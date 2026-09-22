"""Qwen-Image-2.1's text encoder -- Qwen3-VL's language stack -- in JAX.

The last piece of the native path. ``qwen_image_21_jax.py`` has the DiT and
``qwen_image_21_jax_vae.py`` the decoder; this is what turns a prompt into
the hidden states they are conditioned on.

It is a plain Qwen3 decoder -- 36 layers, 4096 wide, 32 query heads over 8
key/value heads, SwiGLU, RMSNorm -- with three things that are not:

* **Per-head RMSNorm on q and k**, over the head dimension only, applied
  after the projection and before RoPE.
* **Interleaved mRoPE.** Positions are three-dimensional (frame, height,
  width) and the *frequencies* are interleaved between the axes rather than
  sliced into contiguous blocks: with ``mrope_section = [24, 20, 20]``, slots
  ``1, 4, ... 58`` come from the height axis, ``2, 5, ... 59`` from width,
  and everything else -- including all four above 59 -- from the frame axis.
  For a text-only prompt all three axes carry the same position, so this
  collapses to ordinary RoPE; it only bites once a condition image is in the
  sequence. ``rope_theta`` is 5e6, not the usual 1e4.
* **Deepstack injection.** Features from vision layers 8, 16 and 24 are
  *added* into the language stream at image-token positions, in language
  layers 0, 1 and 2 respectively. Text-only prompts never see it, which is
  exactly why it is easy to leave out and then be puzzled by edits.

This model is used as an encoder, not a generator, so two things differ from
how transformers would run it: there is no KV cache, and the output is the
last layer's hidden states **before** the final RMSNorm, which is what the
DiT was trained against.

    JAX_PLATFORMS=cpu PYTHONPATH=src python qwen_image_21_jax_text.py
"""

import glob
import math
import os

import numpy as np

# From text_config in the shipped text_encoder/config.json.
DEFAULTS = dict(num_layers=36, hidden_size=4096, head_dim=128, num_heads=32,
                num_kv_heads=8, intermediate_size=12288, vocab_size=151936,
                rms_norm_eps=1e-6, rope_theta=5_000_000.0,
                mrope_section=(24, 20, 20))


MODEL_ID = "Qwen/Qwen-Image-2.1"


def find_snapshot(model_id=MODEL_ID):
  """The local snapshot directory, downloading only what is missing.

  Goes through `huggingface_hub` rather than globbing a cache path, so it
  honours `HF_HOME` / `HF_HUB_CACHE` and works on a machine that has never
  seen this model.
  """
  from huggingface_hub import snapshot_download

  return snapshot_download(model_id)

def _build_modules():
  """Imports flax lazily so the file can be read without it installed."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, *, rngs=None):
      self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      variance = jnp.mean(x32**2, axis=-1, keepdims=True)
      return (self.weight[...] * (x32 * jax.lax.rsqrt(variance + self.eps))
              ).astype(x.dtype)

  class Linear(nnx.Module):
    """Bias-free; weight stored as torch does, (out, in)."""

    def __init__(self, in_f, out_f, *, dtype, rngs=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype))

    def __call__(self, x):
      return x @ self.weight[...].T

  def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

  def apply_rope(x, cos, sin):
    """`x` is [batch, seq, heads, head_dim]; cos/sin are [seq, head_dim]."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + rotate_half(x) * sin

  class Attention(nnx.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, eps, *, dtype, rngs):
      self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
      self.scale = head_dim**-0.5
      self.q_proj = Linear(hidden, heads * head_dim, dtype=dtype, rngs=rngs)
      self.k_proj = Linear(hidden, kv_heads * head_dim, dtype=dtype, rngs=rngs)
      self.v_proj = Linear(hidden, kv_heads * head_dim, dtype=dtype, rngs=rngs)
      self.o_proj = Linear(heads * head_dim, hidden, dtype=dtype, rngs=rngs)
      # Over the head dimension only, so no reshape is needed afterwards.
      self.q_norm = RMSNorm(head_dim, eps)
      self.k_norm = RMSNorm(head_dim, eps)

    def __call__(self, x, cos, sin, mask=None):
      b, s, _ = x.shape
      q = self.q_norm(self.q_proj(x).reshape(b, s, self.heads, self.head_dim))
      k = self.k_norm(self.k_proj(x).reshape(b, s, self.kv_heads, self.head_dim))
      v = self.v_proj(x).reshape(b, s, self.kv_heads, self.head_dim)
      q = apply_rope(q, cos, sin).astype(v.dtype)
      k = apply_rope(k, cos, sin).astype(v.dtype)
      # jax's attention does grouped queries itself when the head counts
      # differ, so no explicit repeat.
      out = jax.nn.dot_product_attention(q, k, v, mask=mask,
                                         is_causal=mask is None,
                                         scale=self.scale)
      return self.o_proj(out.reshape(b, s, -1))

  class MLP(nnx.Module):
    def __init__(self, hidden, intermediate, *, dtype, rngs):
      self.gate_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.up_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.down_proj = Linear(intermediate, hidden, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

  class DecoderLayer(nnx.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, intermediate, eps,
                 *, dtype, rngs):
      self.input_layernorm = RMSNorm(hidden, eps)
      self.self_attn = Attention(hidden, heads, kv_heads, head_dim, eps,
                                 dtype=dtype, rngs=rngs)
      self.post_attention_layernorm = RMSNorm(hidden, eps)
      self.mlp = MLP(hidden, intermediate, dtype=dtype, rngs=rngs)

    def __call__(self, x, cos, sin, mask=None):
      x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
      return x + self.mlp(self.post_attention_layernorm(x))

  class TextModel(nnx.Module):
    """Qwen3-VL's language model, run as an encoder.

    Returns the last layer's hidden states *before* `norm`. The final
    RMSNorm is loaded anyway so the module round-trips the checkpoint, but
    `__call__` does not apply it: the DiT was trained on the unnormalised
    output, and transformers 5.x quietly ties `hidden_states[-1]` to the
    normalised one, which upstream works around with a forward hook.
    """

    def __init__(self, num_layers=36, hidden_size=4096, head_dim=128,
                 num_heads=32, num_kv_heads=8, intermediate_size=12288,
                 vocab_size=151936, rms_norm_eps=1e-6, rope_theta=5e6,
                 mrope_section=(24, 20, 20), *, dtype, rngs):
      self.embed_tokens = nnx.Param(jnp.zeros((vocab_size, hidden_size), dtype))
      self.layers = nnx.List([
          DecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim,
                       intermediate_size, rms_norm_eps, dtype=dtype, rngs=rngs)
          for _ in range(num_layers)])
      self.norm = RMSNorm(hidden_size, rms_norm_eps)

    def __call__(self, input_ids, cos, sin, mask=None, inputs_embeds=None,
                 deepstack=None, visual_positions=None):
      """`visual_positions` is an int index array, not a boolean mask:
      boolean indexing needs a concrete count, which a traced function does
      not have."""
      x = self.embed_tokens[...][input_ids] if inputs_embeds is None \
          else inputs_embeds
      for index, layer in enumerate(self.layers):
        x = layer(x, cos, sin, mask)
        if deepstack is not None and index < len(deepstack):
          # Add the vision features back in at the image-token positions.
          x = x.at[:, visual_positions].add(deepstack[index].astype(x.dtype))
      return x

  return TextModel


# --------------------------------------------------------------------------
# host prologue: interleaved mRoPE
# --------------------------------------------------------------------------


def mrope_cos_sin(position_ids, head_dim=128, rope_theta=5_000_000.0,
                  mrope_section=(24, 20, 20)):
  """Build the rotary tables for one sequence, on the host.

  Args:
    position_ids: `(3, seq)` -- the frame, height and width position of every
      token. A text-only prompt passes the same `arange` three times.

  Returns:
    `cos`, `sin`, each `(seq, head_dim)`.

  The interleave is the part worth reading twice. Each axis gets its own
  `freqs` over all `head_dim / 2` slots, and then the slots are handed out
  round-robin: slot `3k + 1` to height and `3k + 2` to width for as many `k`
  as that axis's section allows, everything else to frame. The result is
  duplicated, not interleaved, into the full `head_dim` -- `rotate_half`
  pairs `i` with `i + head_dim / 2`.
  """
  position_ids = np.asarray(position_ids, np.float64)
  if position_ids.shape[0] != 3:
    raise ValueError(f"expected (3, seq) positions, got {position_ids.shape}")
  inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2) / head_dim))
  freqs = position_ids[:, :, None] * inv_freq[None, None, :]  # (3, seq, dim/2)

  merged = freqs[0].copy()
  for axis, offset in ((1, 1), (2, 2)):
    stop = mrope_section[axis] * 3
    merged[:, offset:stop:3] = freqs[axis][:, offset:stop:3]
  merged = np.concatenate([merged, merged], axis=-1)
  return np.cos(merged).astype(np.float32), np.sin(merged).astype(np.float32)


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------

_PREFIX = "model.language_model."


def _resolve(model, path):
  obj = model
  for part in path:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def load_text_model(model, text_encoder_dir, mesh=None, spec_fn=None):
  """Read the language half of the checkpoint into the nnx module.

  Skips `lm_head` -- nothing here generates tokens -- and the vision tower,
  which lives in its own module.
  """
  import jax
  from flax import nnx
  from safetensors import safe_open
  from jax.sharding import NamedSharding

  shards = sorted(glob.glob(os.path.join(text_encoder_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {text_encoder_dir}")

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if not key.startswith(_PREFIX):
          continue                      # lm_head, model.visual.*
        name = key[len(_PREFIX):]
        if name == "embed_tokens.weight":
          param = model.embed_tokens
        else:
          param = _resolve(model, name.split("."))
        value = reader.get_tensor(key).astype(param[...].dtype)
        if value.shape != param[...].shape:
          raise ValueError(
              f"{key}: checkpoint {value.shape} vs model {param[...].shape}")
        if mesh is not None and spec_fn is not None:
          value = jax.device_put(
              value, NamedSharding(mesh, spec_fn(name.split("."), value.ndim)))
        param[...] = value
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return model


_COLUMN = frozenset({"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"})
_ROW = frozenset({"o_proj", "down_proj"})


def param_spec(keys, ndim):
  """Tensor-parallel layout, matching the DiT's conventions."""
  from jax.sharding import PartitionSpec as P

  parent = next((k for k in reversed(keys)
                 if k not in ("weight", "value", ".value")), "")
  if ndim == 2 and parent in _COLUMN:
    return P("tensor", None)
  if ndim == 2 and parent in _ROW:
    return P(None, "tensor")
  return P()


# --------------------------------------------------------------------------
# numerical equivalence against transformers
# --------------------------------------------------------------------------


def main():
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from transformers import Qwen3VLForConditionalGeneration

  text_dir = os.path.join(find_snapshot(), "text_encoder")

  TextModel = _build_modules()
  model = TextModel(**DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  load_text_model(model, text_dir)

  torch_model = Qwen3VLForConditionalGeneration.from_pretrained(
      text_dir, dtype=torch.float32).eval()

  rng = np.random.default_rng(0)
  seq = 24
  input_ids = rng.integers(0, 150000, size=(1, seq))
  positions = np.tile(np.arange(seq), (3, 1))  # text-only: all axes agree
  cos, sin = mrope_cos_sin(positions, DEFAULTS["head_dim"],
                           DEFAULTS["rope_theta"], DEFAULTS["mrope_section"])

  # transformers returns hidden_states[-1] tied to the normalised output, so
  # neutralise the final norm the way the diffusers pipeline does.
  language = torch_model.model.language_model
  handle = language.norm.register_forward_hook(lambda m, a, out: a[0])
  try:
    with torch.no_grad():
      ref = torch_model(input_ids=torch.tensor(input_ids),
                        attention_mask=torch.ones((1, seq), dtype=torch.long),
                        output_hidden_states=True).hidden_states[-1].numpy()
  finally:
    handle.remove()

  got = np.asarray(model(jnp.asarray(input_ids), jnp.asarray(cos),
                         jnp.asarray(sin)))
  diff = np.abs(ref - got)
  rel = diff.max() / max(np.abs(ref).max(), 1e-9)
  print(f"hidden {ref.shape}  max|diff| {diff.max():.3e}  relative {rel:.3e}  "
        f"mean|diff| {diff.mean():.3e}")
  ok = rel < 1e-4
  print("MATCH" if ok else "MISMATCH")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
