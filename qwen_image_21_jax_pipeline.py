"""Qwen-Image-2.1 end to end in native JAX.

Ties together ``qwen_image_21_jax.py`` (the 7.12B DiT) and
``qwen_image_21_jax_vae.py`` (the decoder), adds the flow-match schedule and
the latent bookkeeping, and writes a PNG.

    PYTHONPATH=src python qwen_image_21_jax_pipeline.py \\
        --prompt "..." --output out.png

The text encoder is the one piece still on the host: ``--text-encoder host``
runs Qwen3-VL through PyTorch and hands the hidden states over. Everything
downstream of that -- which is all of the per-step compute -- is JAX.

What the text encoder has to hand over is more particular than it looks, and
three details decide whether the image is right:

* The hidden states are the **last decoder layer's output before the final
  RMSNorm**, not ``last_hidden_state``. transformers 5.x ties that entry to
  the normalised output, so upstream neutralises the norm with a forward hook
  for the duration of the call, and so does this file. Skipping it costs
  about a third of the signal, and it shows up first in rendered text.
* The leading system-role tokens are dropped, by a count derived from the
  tokenised system message rather than hardcoded.
* ``image_pad_mask`` marks which of the remaining positions are image slots.
  The DiT expands each of those four-fold and drops the latents in, so it is
  what ties the two sequences together.

The schedule is transcribed rather than imported: ``sigmas`` are
``linspace(1, 1/steps, steps)`` put through an exponential shift whose ``mu``
depends on the image's token count, then stretched so the last one lands on
``shift_terminal``. It agrees with ``FlowMatchEulerDiscreteScheduler`` to
8e-8, checked with ``--check-schedule``.
"""

import argparse
import glob
import math
import os
import time

import numpy as np

import qwen_image_21_jax as dit
import qwen_image_21_jax_vae as vae

VAE_SCALE_FACTOR = 16  # scale_factor_spatial from the VAE config


def calculate_dimensions(target_area, ratio):
  """The condition image's working size: the target area at its aspect, to 32."""
  width = math.sqrt(target_area * ratio)
  height = width / ratio
  return round(width / 32) * 32, round(height / 32) * 32


MODEL_ID = "Qwen/Qwen-Image-2.1"


def find_snapshot(model_id=MODEL_ID):
  """The local snapshot directory, downloading only what is missing.

  Goes through `huggingface_hub` rather than globbing a cache path, so it
  honours `HF_HOME` / `HF_HUB_CACHE` and works on a machine that has never
  seen this model.
  """
  from huggingface_hub import snapshot_download

  return snapshot_download(model_id)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def flow_match_sigmas(num_steps, image_seq_len, base_image_seq_len=256,
                      max_image_seq_len=8192, base_shift=0.5, max_shift=0.9,
                      shift_terminal=0.02):
  """The sigma schedule, as numpy. Defaults are the shipped scheduler config.

  Returns `num_steps + 1` values, the last one zero, so a step reads
  `sigmas[i]` and `sigmas[i + 1]`.
  """
  sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float64)
  slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
  mu = image_seq_len * slope + base_shift - slope * base_image_seq_len
  sigmas = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
  if shift_terminal is not None:
    one_minus = 1.0 - sigmas
    sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - shift_terminal))
  return np.concatenate([sigmas, [0.0]]).astype(np.float32)


# --------------------------------------------------------------------------
# text encoding, still on the host
# --------------------------------------------------------------------------


_SYS_PROMPT = "Comprehend and analyze the provided prompt."
_TEMPLATE_T2I = (
    f"<|im_start|>system\n{_SYS_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n")
_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{_SYS_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}"
    "<|im_end|>\n<|im_start|>assistant\n")


_PROCESSOR_CACHE = {}


def _get_processor(snapshot):
  """Cached: building it costs over a second, which swamps a short prompt."""
  from transformers import AutoProcessor

  if snapshot not in _PROCESSOR_CACHE:
    _PROCESSOR_CACHE[snapshot] = AutoProcessor.from_pretrained(
        os.path.join(snapshot, "processor"))
  return _PROCESSOR_CACHE[snapshot]


def _tokenize(snapshot, prompts, images=None):
  """Build the template, tokenize, and work out how much to drop.

  Shared by both text-encoder backends so they cannot drift apart.
  """
  processor = _get_processor(snapshot)
  sys_message = [{"role": "system",
                  "content": [{"type": "text", "text": _SYS_PROMPT}]}]
  drop_idx = len(processor.apply_chat_template(
      sys_message, tokenize=True, return_dict=False)[0])
  img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]

  prompts = [p if p else " " for p in prompts]
  if images is None:
    texts = [_TEMPLATE_T2I.format(p) for p in prompts]
    extra = {}
  else:
    texts = [_TEMPLATE_TI2I.format(p) for p in prompts]
    extra = {"images": [img for _ in prompts for img in images]}
  inputs = processor(text=texts, padding=True, padding_side="left",
                     return_tensors="pt", **extra)
  return inputs, drop_idx, img_token_id


def _postprocess(hidden, input_ids, attention_mask, drop_idx, img_token_id):
  """Strip padding and the system prefix, then re-pad to a rectangle.

  `hidden` is `(batch, seq, dim)` numpy. Returns the three arrays the DiT
  takes: the embeddings, which of their positions are real, and which are
  image slots.
  """
  keep = [np.flatnonzero(row)[drop_idx:] for row in attention_mask]
  max_len = max(len(k) for k in keep)
  embeds = np.zeros((len(keep), max_len, hidden.shape[-1]), hidden.dtype)
  mask = np.zeros((len(keep), max_len), bool)
  pad = np.zeros((len(keep), max_len), bool)
  for i, index in enumerate(keep):
    embeds[i, :len(index)] = hidden[i, index]
    mask[i, :len(index)] = True
    pad[i, :len(index)] = input_ids[i, index] == img_token_id
  return embeds, mask, pad


def encode_prompt_host(snapshot, prompts, images=None, dtype=None):
  """Qwen3-VL through PyTorch. The reference the JAX one is checked against."""
  import torch
  from transformers import Qwen3VLForConditionalGeneration

  dtype = dtype or torch.bfloat16
  inputs, drop_idx, img_token_id = _tokenize(snapshot, prompts, images)
  text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
      os.path.join(snapshot, "text_encoder"), dtype=dtype).eval()

  forward = {"input_ids": inputs.input_ids,
             "attention_mask": inputs.attention_mask,
             "output_hidden_states": True}
  for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
    if getattr(inputs, key, None) is not None:
      forward[key] = getattr(inputs, key)

  # Replace the final RMSNorm with the identity for this call: the DiT was
  # trained on the last decoder layer's *unnormalised* output.
  model = getattr(text_encoder.model, "language_model", text_encoder.model)
  handle = model.norm.register_forward_hook(lambda m, args, out: args[0])
  try:
    with torch.no_grad():
      hidden = text_encoder(**forward).hidden_states[-1].float().numpy()
  finally:
    handle.remove()
  return _postprocess(hidden, inputs.input_ids.numpy(),
                      inputs.attention_mask.numpy().astype(bool),
                      drop_idx, img_token_id)


_TEXT_JIT = None


def _text_forward(model, input_ids, cos, sin, inputs_embeds=None,
                  deepstack=None, visual_positions=None):
  """Compiled per sequence length; prompts of one length reuse the program.

  The jitted wrapper is built once and kept -- `nnx.jit` returns a fresh
  callable with a fresh cache each time it is called, so building it inside
  this function recompiles on every prompt and is slower than not jitting.
  """
  global _TEXT_JIT
  if _TEXT_JIT is None:
    from flax import nnx
    _TEXT_JIT = nnx.jit(
        lambda m, i, c, s, e, d, v: m(i, c, s, None, e, d, v),
        static_argnums=())
  return _TEXT_JIT(model, input_ids, cos, sin, inputs_embeds, deepstack,
                   visual_positions)


def load_text_encoder(snapshot, dtype="bfloat16", mesh=None):
  """Build and fill Qwen3-VL's language stack. Keep it resident across calls."""
  import jax.numpy as jnp
  from flax import nnx

  import qwen_image_21_jax_text as text

  model = text._build_modules()(**text.DEFAULTS, dtype=getattr(jnp, dtype),
                                rngs=nnx.Rngs(0))
  text.load_text_model(model, os.path.join(snapshot, "text_encoder"),
                       mesh=mesh, spec_fn=text.param_spec if mesh else None)
  return model


def load_vision_encoder(snapshot, dtype="bfloat16"):
  """Qwen3-VL's vision tower. Small enough (0.5B) to leave replicated."""
  import jax.numpy as jnp
  from flax import nnx

  import qwen_image_21_jax_vision as vision

  model = vision._build_modules()(**vision.DEFAULTS, dtype=getattr(jnp, dtype),
                                  rngs=nnx.Rngs(0))
  vision.load_vision_model(model, os.path.join(snapshot, "text_encoder"))
  return model


def _multimodal_position_ids(snapshot, inputs):
  """The three mRoPE axes for a sequence that contains an image.

  Borrowed from transformers rather than transcribed: it is pure index
  arithmetic over the token types and the patch grid, it runs once per shape
  on the host, and the model it hangs off is instantiated on the meta device
  so no weights exist.
  """
  import torch
  from transformers import AutoConfig, Qwen3VLModel

  config = AutoConfig.from_pretrained(os.path.join(snapshot, "text_encoder"))
  with torch.device("meta"):
    shim = Qwen3VLModel(config)
  shim.rope_deltas = None
  position_ids, _ = shim.get_rope_index(
      inputs.input_ids, image_grid_thw=inputs.image_grid_thw,
      attention_mask=inputs.attention_mask,
      mm_token_type_ids=getattr(inputs, "mm_token_type_ids", None))
  return position_ids[:, 0].numpy()          # (3, seq)


def encode_prompt_jax(snapshot, prompts, images=None, dtype="bfloat16",
                      mesh=None, model=None, vision_model=None):
  """Qwen3-VL natively. With `images`, the vision tower runs too."""
  import jax.numpy as jnp
  from flax import nnx

  import qwen_image_21_jax_text as text
  import qwen_image_21_jax_vision as vision

  inputs, drop_idx, img_token_id = _tokenize(snapshot, prompts, images)
  input_ids = inputs.input_ids.numpy()
  attention_mask = inputs.attention_mask.numpy().astype(bool)
  if not attention_mask.all():
    raise NotImplementedError(
        "left-padded batches need a key mask through the decoder; encode "
        "prompts one at a time until that is wired up")

  if model is None:
    model = load_text_encoder(snapshot, dtype, mesh)

  seq = input_ids.shape[1]
  inputs_embeds = None
  deepstack = None
  visual_positions = None

  if images is None:
    positions = np.tile(np.arange(seq), (3, 1))  # text-only: all axes agree
  else:
    if vision_model is None:
      vision_model = load_vision_encoder(snapshot, dtype)
    grid = inputs.image_grid_thw.numpy()
    vmeta = vision.vision_metadata(grid)
    vcos, vsin = vision.vision_rope(vmeta["position_ids"])
    merged, deepstack = vision_model(
        jnp.asarray(inputs.pixel_values.numpy(), getattr(jnp, dtype)),
        {"interp_indices": jnp.asarray(vmeta["interp_indices"]),
         "interp_weights": jnp.asarray(vmeta["interp_weights"])},
        jnp.asarray(vcos), jnp.asarray(vsin))
    # Drop the merged tokens into the image-pad slots of the text stream.
    visual_positions = jnp.asarray(
        np.flatnonzero(input_ids[0] == img_token_id))
    embeds = model.embed_tokens[...][jnp.asarray(input_ids)]
    inputs_embeds = embeds.at[:, visual_positions].set(
        merged.astype(embeds.dtype))
    positions = _multimodal_position_ids(snapshot, inputs)

  cos, sin = text.mrope_cos_sin(positions, text.DEFAULTS["head_dim"],
                                text.DEFAULTS["rope_theta"],
                                text.DEFAULTS["mrope_section"])
  hidden = np.asarray(_text_forward(
      model, jnp.asarray(input_ids), jnp.asarray(cos), jnp.asarray(sin),
      inputs_embeds, deepstack, visual_positions)).astype(np.float32)
  return _postprocess(hidden, input_ids, attention_mask, drop_idx,
                      img_token_id)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def generate(prompt, height=1024, width=1024, num_inference_steps=20, seed=42,
             snapshot=None, dtype="bfloat16", embeds_cache=None, mesh=None,
             runs=1, text_encoder="jax", image=None, output_resolution=1024,
             kv_cache=True):
  """One image, start to finish. Returns uint8 RGB `(height, width, 3)`.

  With `image`, the condition image is resized to `output_resolution`'s area
  at its own aspect ratio, encoded by the VAE, and its latents are prepended
  to the noise -- the DiT sees one joint sequence with the condition image's
  tokens ahead of the target's.
  """
  import jax
  import jax.numpy as jnp
  import torch
  from flax import nnx

  snapshot = snapshot or find_snapshot()
  jnp_dtype = getattr(jnp, dtype)
  encoder = None

  chips = jax.device_count()
  if mesh is None and chips > 1:
    mesh = jax.make_mesh((chips,), ("tensor",),
                         axis_types=(jax.sharding.AxisType.Auto,))

  # --- condition image ---------------------------------------------------
  condition = None
  if image is not None:
    from diffusers.image_processor import VaeImageProcessor

    processor = VaeImageProcessor(vae_scale_factor=VAE_SCALE_FACTOR * 2)
    cond_w, cond_h = calculate_dimensions(output_resolution**2,
                                          image.size[0] / image.size[1])
    rgba = image.convert("RGBA")
    # The vision encoder reads the resized RGB; the VAE reads all four
    # channels of the same pixels.
    condition = {
        "vision": processor.resize(rgba, width=cond_w, height=cond_h),
        "pixels": processor.preprocess(rgba, width=cond_w, height=cond_h),
        "size": (cond_w, cond_h),
    }
    height = height or cond_h
    width = width or cond_w

  # --- text --------------------------------------------------------------
  if embeds_cache and os.path.exists(embeds_cache):
    cached = np.load(embeds_cache)
    embeds, embeds_mask, pad_mask = (cached["embeds"], cached["mask"],
                                     cached["pad"])
    print(f"text embeddings from {embeds_cache}")
  else:
    t0 = time.perf_counter()
    images = [condition["vision"]] if condition else None
    if text_encoder == "jax":
      encoder = load_text_encoder(snapshot, dtype, mesh)
      embeds, embeds_mask, pad_mask = encode_prompt_jax(
          snapshot, [prompt], images=images, dtype=dtype, mesh=mesh,
          model=encoder)
    else:
      embeds, embeds_mask, pad_mask = encode_prompt_host(snapshot, [prompt],
                                                         images=images)
    print(f"text encoded ({text_encoder}) in {time.perf_counter() - t0:.1f}s")
    if embeds_cache:
      np.savez(embeds_cache, embeds=embeds, mask=embeds_mask, pad=pad_mask)

  # --- latents, seeded the way diffusers seeds them ----------------------
  latent_h = 2 * (height // (VAE_SCALE_FACTOR * 2))
  latent_w = 2 * (width // (VAE_SCALE_FACTOR * 2))
  # `randn_tensor` draws in the pipeline's working dtype, not fp32, and a
  # bfloat16 draw consumes the generator differently -- get this wrong and
  # the image is still good, just a different one, which is the hardest kind
  # of difference to notice.
  generator = torch.Generator("cpu").manual_seed(seed)
  noise = torch.randn((1, 1, 64, latent_h, latent_w), generator=generator,
                      dtype=getattr(torch, dtype))
  # `_pack_latents`: 2.1 consumes latents unpatched, so this is a flatten.
  latents = jnp.asarray(
      noise.view(1, 64, latent_h * latent_w).transpose(1, 2).float().numpy(), jnp_dtype)

  img_shapes = ([] if condition is None else
                [(1, condition["size"][1] // VAE_SCALE_FACTOR,
                  condition["size"][0] // VAE_SCALE_FACTOR)])
  img_shapes.append((1, latent_h, latent_w))
  target_slots = latents.shape[1] // 4
  img_mask = np.concatenate(
      [pad_mask, np.ones((pad_mask.shape[0], target_slots), bool)], axis=1)
  meta = dit.to_device(dit.build_metadata(img_mask, img_shapes, embeds_mask))

  sigmas = flow_match_sigmas(num_inference_steps, latents.shape[1])

  # --- models ------------------------------------------------------------
  _, Transformer = dit._build_modules()
  Decoder, Encoder = vae._build_modules()
  t0 = time.perf_counter()
  model = Transformer(dtype=jnp_dtype, rngs=nnx.Rngs(0))
  dit.load_transformer(model, os.path.join(snapshot, "transformer"), mesh=mesh)
  # The decoder is 0.7 GB; keeping it in fp32 costs nothing and avoids
  # bf16 banding in flat regions of the image.
  decoder = Decoder(**vae.DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  vae.load_decoder(decoder, os.path.join(snapshot, "vae"))
  print(f"weights on {chips} chip(s) in {time.perf_counter() - t0:.1f}s")

  prompt_embeds = jnp.asarray(embeds, jnp_dtype)

  condition_latents = None
  if condition is not None:
    import json as _json
    with open(os.path.join(snapshot, "vae", "config.json")) as handle:
      cfg = _json.load(handle)
    vae_encoder = Encoder(**vae.ENCODER_DEFAULTS, dtype=jnp.float32,
                          rngs=nnx.Rngs(0))
    vae.load_encoder(vae_encoder, os.path.join(snapshot, "vae"))
    pixels = jnp.asarray(
        np.transpose(condition["pixels"].numpy(), (0, 2, 3, 1)), jnp.float32)
    encoded = vae.encode_image(vae_encoder, pixels, cfg["latents_mean"],
                               cfg["latents_std"])
    condition_latents = jnp.transpose(
        encoded.reshape(1, 64, -1), (0, 2, 1)).astype(jnp_dtype)
    print(f"condition image {condition['size']} -> "
          f"{condition_latents.shape[1]} latent tokens")

  def _joint(latents):
    return latents if condition_latents is None else jnp.concatenate(
        [condition_latents, latents], axis=1)

  @nnx.jit
  def denoise(model, latents, embeds, timestep):
    target = latents.shape[1]
    return model(_joint(latents), embeds, timestep, meta)[:, -target:]

  @nnx.jit
  def denoise_extract(model, latents, embeds, timestep):
    target = latents.shape[1]
    out, caches = model(_joint(latents), embeds, timestep, meta, extract=True)
    return out[:, -target:], caches

  @nnx.jit
  def denoise_cached(model, latents, embeds, timestep, caches):
    # The cached program returns the target rows only, which is already the
    # slice the scheduler wants.
    return model(_joint(latents), embeds, timestep, meta, kv_cache=caches)

  @nnx.jit
  def denoise_scan(model, latents, embeds, steps, caches):
    """Every remaining step inside one program.

    Worth about 4%: the gaps between twenty separate launches, not the
    dispatch itself, which is already asynchronous. `steps` is the stacked
    `(sigma, sigma_next)` pairs.
    """
    def body(x, pair):
      sigma, sigma_next = pair
      if caches is None:
        pred = model(_joint(x), embeds, sigma[None], meta)[:, -x.shape[1]:]
      else:
        pred = model(_joint(x), embeds, sigma[None], meta, kv_cache=caches)
      return x + (sigma_next - sigma) * pred, None

    out, _ = jax.lax.scan(body, latents, steps)
    return out

  import json
  with open(os.path.join(snapshot, "vae", "config.json")) as handle:
    vae_config = json.load(handle)

  @nnx.jit
  def decode(decoder, latents):
    unpacked = jnp.transpose(latents, (0, 2, 1)).reshape(
        1, 64, latent_h, latent_w).astype(jnp.float32)
    return vae.decode_latents(decoder, unpacked, vae_config["latents_mean"],
                              vae_config["latents_std"])

  for run in range(runs):
    t_text = 0.0
    if encoder is not None:
      t0 = time.perf_counter()
      encode_prompt_jax(snapshot, [prompt], dtype=dtype, mesh=mesh,
                        model=encoder)
      t_text = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = latents
    caches = None
    first = 0
    if kv_cache:
      # Step 0 fills the cache and cannot be part of the scan.
      noise_pred, caches = denoise_extract(
          model, x, prompt_embeds, jnp.asarray([sigmas[0]], jnp.float32))
      x = x + (sigmas[1] - sigmas[0]) * noise_pred
      first = 1
    pairs = jnp.stack([jnp.asarray(sigmas[first:num_inference_steps]),
                       jnp.asarray(sigmas[first + 1:num_inference_steps + 1])],
                      axis=1)
    x = denoise_scan(model, x, prompt_embeds, pairs, caches)
    x = jax.block_until_ready(x)
    t_steps = time.perf_counter() - t0
    t0 = time.perf_counter()
    pixels = np.asarray(jax.block_until_ready(decode(decoder, x)))
    t_decode = time.perf_counter() - t0
    tag = " (includes compilation)" if run == 0 else ""
    print(f"run {run}: text {t_text:.2f}s + {num_inference_steps} steps "
          f"{t_steps:.2f}s + decode {t_decode:.2f}s = "
          f"{t_text + t_steps + t_decode:.2f}s{tag}", flush=True)

  rgb = ((pixels[0, :, :, :3] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
  return rgb


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prompt", type=str, required=False)
  parser.add_argument("--image", type=str, default=None,
                      help="condition image; omit for text-to-image")
  parser.add_argument("--height", type=int, default=1024)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--num_inference_steps", type=int, default=20)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=str, default="qwen_image_21_jax.png")
  parser.add_argument("--dtype", type=str, default="bfloat16")
  parser.add_argument("--embeds-cache", type=str, default=None,
                      help="npz to reuse host text embeddings across runs")
  parser.add_argument("--no-kv-cache", action="store_true",
                      help="recompute the prefix every step, for comparison")
  parser.add_argument("--text-encoder", choices=("jax", "host"), default="jax",
                      help="'host' runs Qwen3-VL through PyTorch; needed for "
                           "condition images until the vision tower lands")
  parser.add_argument("--runs", type=int, default=1,
                      help="repeat to measure steady state; run 0 compiles")
  parser.add_argument("--check-schedule", action="store_true")
  args = parser.parse_args()

  if args.check_schedule:
    from diffusers import FlowMatchEulerDiscreteScheduler
    snapshot = find_snapshot()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(snapshot, "scheduler"))
    steps, seq = args.num_inference_steps, 4096
    mine = flow_match_sigmas(steps, seq)
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = seq * slope + 0.5 - slope * 256
    scheduler.set_timesteps(steps, sigmas=np.linspace(1.0, 1.0 / steps, steps),
                            mu=mu)
    ref = scheduler.sigmas.numpy()
    print(f"schedule max|diff| {np.abs(ref - mine).max():.3e}")
    return 0

  if not args.prompt:
    parser.error("--prompt is required")
  from PIL import Image

  condition = Image.open(args.image) if args.image else None
  rgb = generate(args.prompt, args.height, args.width,
                 args.num_inference_steps, args.seed, dtype=args.dtype,
                 embeds_cache=args.embeds_cache, runs=args.runs,
                 text_encoder=args.text_encoder, image=condition,
                 kv_cache=not args.no_kv_cache)
  Image.fromarray(rgb).save(args.output)
  print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
