"""Qwen-Image-Edit on TPU via torchax.

Standalone port of DiffusionServing's qwen-image-edit torchax runner (a serving
framework, not open to customers) into a plain diffusers + torchax script, in the
same spirit as this repo's ``wan_tx_splash_attn.py``.

Ported so far (correctness first, then layering back DiffusionServing's
performance optimizations one at a time, each verified with a real TPU run):
  1. Baseline: DiT + VAE on TPU via torchax, Qwen2.5-VL text encoder on CPU,
     jax.nn.dot_product_attention for SDPA. Verified correct end-to-end.
  2. VAE JIT-compiled encode/decode + H-axis GSPMD spatial sharding of
     activations (weights stay replicated).
  3. DiT joint self-attention -> a splash-flash pallas kernel (removes jax dpa's
     S x S fp32 logits materialization) — this file. Simplified vs
     DiffusionServing's version: their kernel also converts a text-padding
     attn_mask into runtime segment ids, but this repo's vendored diffusers
     version (0.35.0) never actually wires ``encoder_hidden_states_mask`` into an
     ``attn_mask`` for the DiT attention processor (checked by reading
     transformer_qwenimage.py), so there's no mask to convert — always dense
     FullMask. Revisit if a future diffusers sync adds real mask wiring.
  4. Hybrid text encoding: Qwen2.5-VL's 28-layer LANGUAGE MODEL moves to TPU
     (sharded FSDP-style, --text_encoder_cpu_only to disable) — this was ~85% of
     a whole edit's CPU time in DiffusionServing's own profiling.
  5. Vision tower ALSO moves to TPU (--vision_cpu_only to disable), reimplementing
     Qwen2_5_VisionTransformerPretrainedModel.forward (``_VisionTPU``) to take all
     grid_thw-derived metadata (window_index, cu_seqlens, rotary cos/sin, a fused
     block-diagonal attention mask) as precomputed CPU inputs instead of computing
     them internally — the stock forward's ``self.rot_pos_emb``/
     ``self.get_window_index`` do plain-Python loops/``.item()``/``.tolist()`` over
     ``grid_thw``, fine when eager but not when grid_thw is a genuine (non-static)
     jit input. Also requires ``_patch_vision_attention()``: the stock windowed
     attention path splits Q/K/V per window via ``lengths.tolist()`` (also
     untraceable) — replaced with one fused SDPA call over a precomputed
     block-diagonal bool mask, numerically identical to per-window attention.
     Note: this repo's transformers version (4.57.1) has no separate
     ``vision_utils`` module (DiffusionServing's version — 5.14.0 — does); the
     same grid_thw-derived math lives on the model's own methods here instead, so
     the precompute calls ``visual.rot_pos_emb``/``visual.get_window_index``
     directly rather than external ``vision_utils`` functions.
  6. Fast 2x nearest upsample for the VAE decoder (``QwenImageUpsample`` patch):
     torchax deliberately doesn't implement ``aten.upsample_nearest2d`` as a
     direct jax op (falls back to PyTorch's decomposition, dispatched op-by-op)
     — measured via a standalone per-block bisection of the VAE decoder that the
     ONE upsample stage reaching full output resolution cost ~3.4s ALONE (~140x
     its same-resolution neighbors), persisting identically with H-sharding
     fully disabled (so it wasn't the sharding hooks). This VAE only ever
     upsamples by exactly 2x, where nearest and nearest-exact are identical (both
     just replicate each pixel into a 2x2 block — no interpolation math), so
     replaced with ``repeat_interleave``, which IS natively implemented in
     torchax. Fixed VAE decode from ~3.4s to ~0.02s per stage (~0.09s total
     decode+postprocess) — this was THE remaining bottleneck once steps 1-5 were
     done; full steady-state latency (Lightning LoRA, 4 steps, cfg=1.0) went from
     ~4.8s to ~1.45s, matching the user's own v6e-4 DiffusionServing benchmark
     (1.272s) in the same ballpark. Dense/no-LoRA 40-step/cfg=4.0 also verified:
     ~16.7s steady-state (was ~21 min before any optimization).
  Still deferred (tuned for v6e-8 + newer jax/diffusers/transformers than this
  repo's verified stack, higher risk to port): aspect bucketing (recompiles per
  unique image size / prompt length are the price paid instead), FP8 (moot on
  v6e-4 anyway — no native fp8 hardware support until v7/Ironwood).

Architecture:
  - DiT (QwenImageTransformer2DModel, 20B) runs on TPU via torchax, sharded across
    all available chips, with a splash-flash attention kernel for joint
    self-attention.
  - VAE (AutoencoderKLQwenImage) encode/decode run on TPU via torchax, JIT-compiled
    with H-axis spatial sharding of activations (weights replicated).
  - Text encoder (Qwen2.5-VL 7B): both the language model and the vision tower run
    on TPU via torchax; only the processor/tokenizer (cheap CPU string/PIL work)
    stay on CPU.
  - Attention uses ``jax.nn.dot_product_attention`` for everything except the DiT's
    joint self-attention (splash-flash) and the vision tower's windowed attention
    (fused masked SDPA); native torchax SDPA is numerically unstable here (NaN on
    the first forward in bf16).
"""

import argparse
import contextlib
import functools
import math
import re

import jax
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from PIL import Image
import torch
import torchax
from torchax.ops import jaten, ops_registry

from diffusers import QwenImageEditPipeline
from diffusers.models.autoencoders import autoencoder_kl_qwenimage as qvae
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen2_5_vl_mod
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding

if not hasattr(Qwen2_5_VLRotaryEmbedding, "_tpu_autocast_patched"):
  # forward() wraps its float32 cast in torch.autocast(device_type=x.device.type,
  # enabled=False) — a purely defensive disable of any AMBIENT autocast context.
  # torch.autocast's __init__ validates device_type against a registered-backend
  # list even when enabled=False, and torchax's "jax" pseudo-device isn't
  # registered there -> AssertionError. We never enable autocast anywhere in this
  # script, so there's nothing to defensively disable; drop the context entirely
  # (the .float()/.to(dtype=x.dtype) calls already control precision explicitly).
  # rope_type is "default" for Qwen2.5-VL's text config (no dynamic/longrope
  # scaling), so bypassing the @dynamic_rope_update wrapper (which no-ops for
  # "default" anyway — checked its source) is also safe.
  def _rope_forward_no_autocast(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

  Qwen2_5_VLRotaryEmbedding.forward = torch.no_grad()(_rope_forward_no_autocast)
  Qwen2_5_VLRotaryEmbedding._tpu_autocast_patched = True


def _patch_vision_attention():
  """Fused-mask replacement for Qwen2_5_VLVisionAttention.forward (jit-safe).

  The stock non-flash path splits Q/K/V per window via ``lengths.tolist()``
  (untraceable under jit) and runs one SDPA per chunk. This patch runs ONE fused
  SDPA over the whole sequence with a precomputed block-diagonal bool mask
  (``vision_window_mask`` kwarg, True = attend) — numerically identical to
  per-window attention. Full-attention blocks arrive with ``cu_seqlens`` of shape
  (2,) -> mask None (shape check, not a value check, so it's trace-static).
  Idempotent; only applied on the vision-on-TPU path.
  """
  if getattr(qwen2_5_vl_mod.Qwen2_5_VLVisionAttention, "_tpu_fused_patched", False):
    return

  def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = qwen2_5_vl_mod.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    if cu_seqlens.shape[0] == 2:  # full attention (also the 1-window edge case)
      attn_mask = None
    else:
      attn_mask = kwargs.get("vision_window_mask")
      if attn_mask is None:
        raise RuntimeError(
            "patched vision attention needs vision_window_mask for windowed blocks (cu_seqlens has >2 entries)."
        )
    out = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attn_mask, scale=self.scaling
    )
    out = out.transpose(1, 2).reshape(seq_length, -1).contiguous()
    return self.proj(out)

  qwen2_5_vl_mod.Qwen2_5_VLVisionAttention.forward = forward
  qwen2_5_vl_mod.Qwen2_5_VLVisionAttention._tpu_fused_patched = True


class _VisionTPU(torch.nn.Module):
  """Reimplements Qwen2_5_VisionTransformerPretrainedModel.forward, taking all
  grid_thw-derived metadata (window_index, cu_seqlens, cu_window_seqlens, rotary
  cos/sin, the fused-attention mask) as precomputed INPUTS instead of computing
  them internally from ``grid_thw``.

  The stock forward's internal metadata computation (``self.rot_pos_emb``,
  ``self.get_window_index``) does plain-Python loops/``.item()``/``.tolist()``
  over ``grid_thw`` — fine when eager, but ``grid_thw`` is a genuine runtime input
  here (not a jit static arg), so under jit those become
  ConcretizationTypeErrors. Precomputing them on CPU (see ``_vision_precompute``)
  and injecting the results sidesteps needing grid_thw inside the jit at all.
  """

  def __init__(self, visual):
    super().__init__()
    self.visual = visual

  def forward(self, hidden_states, window_index, cu_seqlens, cu_window_seqlens, cos, sin, vision_window_mask):
    visual = self.visual
    hidden_states = visual.patch_embed(hidden_states)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    position_embeddings = (cos, sin)
    for layer_num, blk in enumerate(visual.blocks):
      cu = cu_seqlens if layer_num in visual.fullatt_block_indexes else cu_window_seqlens
      hidden_states = blk(
          hidden_states, cu_seqlens=cu, position_embeddings=position_embeddings, vision_window_mask=vision_window_mask
      )
    hidden_states = visual.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    return hidden_states[reverse_indices, :]


_VISION_PRECOMPUTE_CACHE = {}


def _rot_pos_emb_cpu(visual, grid_thw, cpu_inv_freq):
  """CPU reimplementation of Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb.

  Can't just call ``visual.rot_pos_emb(grid_thw)`` once the visual tower's
  weights (including ``rotary_pos_emb.inv_freq``) have been moved to the XLA
  device for compiling — its forward does ``torch.arange(seqlen,
  device=self.inv_freq.device, ...)``, and creating a fresh tensor directly on
  torchax's "jax" pseudo-device outside a jit trace raises ``NotImplementedError:
  ... 'aten::empty.memory_format' ... 'jax' backend``. Needs its own CPU-resident
  copy of inv_freq, captured before the move (see ``main()``).
  """
  pos_ids = []
  for t, h, w in grid_thw:
    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    hpos_ids = hpos_ids.reshape(
        h // visual.spatial_merge_size, visual.spatial_merge_size, w // visual.spatial_merge_size,
        visual.spatial_merge_size,
    )
    hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
    wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    wpos_ids = wpos_ids.reshape(
        h // visual.spatial_merge_size, visual.spatial_merge_size, w // visual.spatial_merge_size,
        visual.spatial_merge_size,
    )
    wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
  pos_ids = torch.cat(pos_ids, dim=0)
  max_grid_size = int(grid_thw[:, 1:].max())
  seq = torch.arange(max_grid_size, dtype=cpu_inv_freq.dtype)
  rotary_pos_emb_full = torch.outer(seq, cpu_inv_freq)
  return rotary_pos_emb_full[pos_ids].flatten(1)


def _vision_precompute(visual, grid_thw, cpu_inv_freq):
  """CPU: all grid_thw-derived values for the vision tower forward, cached.

  Pure function of grid_thw (fixed per input image size, since this script has no
  aspect bucketing — a new image size means a fresh cache entry and a fresh
  compile of the vision wrapper, same tradeoff already accepted for the DiT's
  img_shapes/txt_seq_lens). Reimplements the exact math from the stock
  Qwen2_5_VisionTransformerPretrainedModel.forward/get_window_index (calling the
  model's own methods directly — this version doesn't have a separate
  vision_utils module — with grid_thw fully concrete on CPU, so all the
  `.item()`/`.tolist()` calls inside those methods are safe here).
  """
  import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

  key = tuple(int(x) for x in grid_thw.flatten())
  cached = _VISION_PRECOMPUTE_CACHE.get(key)
  if cached is not None:
    return cached

  rotary_pos_emb = _rot_pos_emb_cpu(visual, grid_thw, cpu_inv_freq)
  window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
  cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
  cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

  seq_len = rotary_pos_emb.shape[0]
  merge_unit = visual.spatial_merge_unit
  rpe = rotary_pos_emb.reshape(seq_len // merge_unit, merge_unit, -1)[window_index, :, :].reshape(seq_len, -1)
  emb = torch.cat((rpe, rpe), dim=-1)
  cos, sin = emb.cos(), emb.sin()

  cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
  cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

  # Block-diagonal mask over the WINDOW-REORDERED sequence (cu_window_seqlens are
  # boundaries in that order); True = attend, matching torch SDPA semantics.
  s = seq_len
  seg = torch.zeros(s, dtype=torch.long)
  inner = cu_window_seqlens[1:-1].to(torch.long)
  if inner.numel():
    seg[inner] = 1
  seg = torch.cumsum(seg, dim=0)
  mask = (seg[:, None] == seg[None, :]).view(1, 1, s, s)

  result = {
      "window_index": window_index,
      "cu_seqlens": cu_seqlens,
      "cu_window_seqlens": cu_window_seqlens,
      "cos": cos,
      "sin": sin,
      "mask": mask,
  }
  _VISION_PRECOMPUTE_CACHE[key] = result
  return result


def _run_visual(compiled_visual, env, mesh, pixel_values, grid_thw, precompute):
  """TPU: compiled vision tower with injected precomputes -> image_embeds."""
  pv = pixel_values.to(torch.bfloat16)
  to_xla = env.to_xla
  with mesh:
    out = compiled_visual(
        to_xla(pv),
        window_index=to_xla(precompute["window_index"]),
        cu_seqlens=to_xla(precompute["cu_seqlens"]),
        cu_window_seqlens=to_xla(precompute["cu_window_seqlens"]),
        cos=to_xla(precompute["cos"]),
        sin=to_xla(precompute["sin"]),
        vision_window_mask=to_xla(precompute["mask"]),
    )
  return out.torch() if hasattr(out, "torch") else out


if not hasattr(CacheMixin, "cache_context"):
  # cache_context() is a newer CacheMixin method (PAB/FasterCache bookkeeping) not
  # present in this repo's diffusers version. The vendored QwenImageEditPipeline
  # calls it unconditionally around each transformer forward; since this baseline
  # never calls enable_cache(), it's a correctness no-op here (matches upstream's
  # own behavior when no cache hooks are registered).
  @contextlib.contextmanager
  def _cache_context_noop(self, name):
    yield

  CacheMixin.cache_context = _cache_context_noop

# ---------------------------------------------------------------------------
# Monkeypatches (required for correctness under jax.jit, not performance)
# ---------------------------------------------------------------------------

if not getattr(QwenEmbedRope, "_tpu_tuple_patched", False):
  # _compute_video_freqs is @lru_cache'd, AND forward() also memoizes per-instance
  # into self.rope_cache (skipped only under torch.compiler.is_compiling(), which
  # is Dynamo-specific and never true under torchax/jax tracing). Both caches store
  # a jax tracer from the first ("cond") jit trace and return it during the second
  # ("uncond") trace -> UnexpectedTracerError ("intermediate ... escaped the
  # scope"). Unwrap the lru_cache AND fully replace forward() to always recompute
  # (bypass self.rope_cache), while also converting video_fhw back from the
  # hashable tuple jax.jit's static_argnames requires to the nested list the
  # original logic branches on.
  _f = getattr(QwenEmbedRope, "_compute_video_freqs", None)
  if _f is not None and hasattr(_f, "__wrapped__"):
    QwenEmbedRope._compute_video_freqs = _f.__wrapped__

  def _to_lists(x):
    return [_to_lists(e) for e in x] if isinstance(x, (list, tuple)) else x

  def _rope_forward(self, video_fhw, txt_seq_lens, device):
    video_fhw = _to_lists(video_fhw)
    if self.pos_freqs.device != device:
      self.pos_freqs = self.pos_freqs.to(device)
      self.neg_freqs = self.neg_freqs.to(device)

    if isinstance(video_fhw, list):
      video_fhw = video_fhw[0]
    if not isinstance(video_fhw, list):
      video_fhw = [video_fhw]

    vid_freqs = []
    max_vid_index = 0
    for idx, fhw in enumerate(video_fhw):
      frame, height, width = fhw
      video_freq = self._compute_video_freqs(frame, height, width, idx).to(device)
      vid_freqs.append(video_freq)
      if self.scale_rope:
        max_vid_index = max(height // 2, width // 2, max_vid_index)
      else:
        max_vid_index = max(height, width, max_vid_index)

    max_len = max(txt_seq_lens)
    txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
    vid_freqs = torch.cat(vid_freqs, dim=0)
    return vid_freqs, txt_freqs

  QwenEmbedRope.forward = _rope_forward
  QwenEmbedRope._tpu_tuple_patched = True


class _StaticImgShapes:
  """Wraps the compiled DiT to make ``img_shapes``/``txt_seq_lens`` hashable.

  diffusers passes ``img_shapes`` as a (nested) list of plain ints and
  ``txt_seq_lens`` as a plain list of ints; jax.jit ``static_argnames`` requires
  hashable values (also used as concrete Python ints inside RoPE slicing, e.g.
  ``self.pos_freqs[max_vid_index : max_vid_index + max_len]``), so convert both to
  tuples right before invoking the compiled forward. Note: since these aren't
  padded to a fixed length in this baseline, every distinct image aspect or prompt
  token count triggers its own recompile (a perf follow-up, not a correctness
  concern). Everything else is delegated to the wrapped compiled module.
  """

  def __init__(self, inner):
    object.__setattr__(self, "_inner", inner)

  def __getattr__(self, name):
    return getattr(object.__getattribute__(self, "_inner"), name)

  def __call__(self, *args, **kwargs):
    ishapes = kwargs.get("img_shapes")
    if ishapes is not None:
      kwargs["img_shapes"] = tuple(
          tuple(tuple(int(x) for x in dim) for dim in sample) for sample in ishapes
      )
    tsl = kwargs.get("txt_seq_lens")
    if tsl is not None:
      kwargs["txt_seq_lens"] = tuple(int(x) for x in tsl)
    return object.__getattribute__(self, "_inner")(*args, **kwargs)


# ---------------------------------------------------------------------------
# Sharding — param-name regex -> PartitionSpec axes, matched via re.fullmatch.
# Axis "tensor" spans however many chips are available (4 on a v6e-4).
# ---------------------------------------------------------------------------
TRANSFORMER_SHARDINGS = {
    r".*attn\.to_q\.weight": ("tensor", None),
    r".*attn\.to_q\.bias": ("tensor",),
    r".*attn\.to_k\.weight": ("tensor", None),
    r".*attn\.to_k\.bias": ("tensor",),
    r".*attn\.to_v\.weight": ("tensor", None),
    r".*attn\.to_v\.bias": ("tensor",),
    r".*attn\.to_out\.0\.weight": (None, "tensor"),
    r".*attn\.add_q_proj\.weight": ("tensor", None),
    r".*attn\.add_q_proj\.bias": ("tensor",),
    r".*attn\.add_k_proj\.weight": ("tensor", None),
    r".*attn\.add_k_proj\.bias": ("tensor",),
    r".*attn\.add_v_proj\.weight": ("tensor", None),
    r".*attn\.add_v_proj\.bias": ("tensor",),
    r".*attn\.to_add_out\.weight": (None, "tensor"),
    r".*(img_mlp|txt_mlp)\.net\.0\.proj\.weight": ("tensor",),
    r".*(img_mlp|txt_mlp)\.net\.0\.proj\.bias": ("tensor",),
    r".*(img_mlp|txt_mlp)\.net\.2\.weight": (None, "tensor"),
}

# Qwen2.5-VL's language model (Qwen2_5_VLTextModel, ~7B) is too big to replicate on
# top of the sharded DiT (~14GB/chip would OOM alongside the DiT+VAE on a v6e-4's
# 32GB/chip). Its GQA head counts (28 query / 4 KV heads on the standard config)
# aren't necessarily divisible by every mesh size, so instead of clean tensor
# parallelism this shards the OUTPUT dim of each big linear as STORAGE sharding and
# lets GSPMD insert whatever collectives are needed (FSDP-style) — the LM only runs
# once per edit, so the extra communication is a non-issue. Only q/k/v_proj have a
# bias (o_proj does not, matching this repo's Qwen2_5_VLAttention).
LM_SHARDINGS = {
    r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight": ("tensor",),
    r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.bias": ("tensor",),
    r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight": ("tensor",),
    r"embed_tokens\.weight": ("tensor",),
}


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
  result = {}
  for k, v in weight_dict.items():
    for target, sharding in sharding_dict.items():
      if re.fullmatch(target, k) is not None:
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
        break
    else:
      v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    result[k] = v
  return result


def _move_module(module, env):
  """Moves a module's weights onto the mesh (replicated, torchax-eager).

  ``state_dict()`` only covers params + PERSISTENT buffers, but
  ``torchax.compile``'s ``JittableModule`` finds what to shard via
  ``extract_all_buffers``, which walks ``dir()`` on every submodule and grabs ANY
  tensor-valued attribute — registered buffer or not (e.g. RoPE ``inv_freq``,
  registered with ``persistent=False`` so it's excluded from state_dict/
  checkpoints, or a plain instance attribute like ``QwenEmbedRope.pos_freqs``
  above). Anything left as a plain CPU tensor at compile time raises
  ``AttributeError: ... no attribute 'apply_jax_'`` when sharding afterward, so
  mirror the exact same walk here and move whatever it finds.
  """
  with jax.default_device("cpu"):
    state_dict = module.state_dict()
    state_dict = env.to_xla(state_dict)
    module.load_state_dict(state_dict, assign=True)
  from torchax.interop import extract_all_buffers  # pylint: disable=import-outside-toplevel

  _, buffers = extract_all_buffers(module)
  for qual_name, tensor in buffers.items():
    if hasattr(tensor, "apply_jax_"):
      continue
    *prefix, attr = qual_name.split(".")
    owner = module
    for p in prefix:
      owner = getattr(owner, p)
    try:
      setattr(owner, attr, env.to_xla(tensor))
    except Exception:  # pylint: disable=broad-except
      pass  # read-only property or similar — not a real buffer to shard


if not hasattr(qvae.QwenImageUpsample, "_tpu_fast_upsample_patched"):
  # torchax deliberately does NOT implement aten.upsample_nearest2d as a direct
  # jax op (see torchax/ops/jaten.py: "NOTE: skip aten.upsample_nearest2d ...
  # here we are using torch decompositions") — it falls back to PyTorch's
  # generic decomposition, dispatched op-by-op through torchax. Measured via a
  # standalone per-block bisection of the VAE decoder (each upsampling stage
  # timed in isolation, forced to sync): the ONE 2x nearest-exact upsample that
  # reaches full output resolution took ~3.4s ALONE — ~140x its same-resolution
  # neighboring blocks — and this persisted identically with H-axis sharding
  # fully disabled, ruling out the sharding hooks as the cause. This VAE only
  # ever upsamples by exactly 2x, and nearest/nearest-exact are identical for an
  # exact integer scale factor (both just replicate each pixel into a 2x2
  # block, no interpolation math) — so replace it with repeat_interleave, which
  # IS natively implemented in torchax (torch.ops.aten.repeat_interleave).
  def _fast_upsample_forward(self, x):
    sh, sw = self.scale_factor
    if (sh, sw) != (2.0, 2.0):
      raise NotImplementedError(
          f"fast upsample override only handles exact 2x nearest, got scale_factor={self.scale_factor}"
      )
    return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

  qvae.QwenImageUpsample.forward = _fast_upsample_forward
  qvae.QwenImageUpsample._tpu_fast_upsample_patched = True


class _VaeEncode(torch.nn.Module):
  """Jittable VAE encode: mirrors AutoencoderKLQwenImage._encode for T=1.

  Calling encoder/quant_conv with feat_cache=None (the default) is bit-identical to
  the stock cached path at T=1 (single image edit, not multi-frame video) — every
  cache mutation in the stock path is gated on ``if feat_cache is not None``, and a
  conv given cache_x=None applies the same static causal padding. Returns raw
  moments; the DiagonalGaussianDistribution is built outside the jit (tensor-only
  boundary).
  """

  def __init__(self, vae):
    super().__init__()
    self.encoder = vae.encoder
    self.quant_conv = vae.quant_conv

  def forward(self, x):  # (B, 3, 1, H, W) bf16
    return self.quant_conv(self.encoder(x))  # (B, 2*z_dim, 1, H/8, W/8)


class _VaeDecode(torch.nn.Module):
  """Jittable VAE decode: mirrors AutoencoderKLQwenImage._decode for T=1."""

  def __init__(self, vae):
    super().__init__()
    self.post_quant_conv = vae.post_quant_conv
    self.decoder = vae.decoder

  def forward(self, z):  # (B, z_dim, 1, H/8, W/8)
    out = self.decoder(self.post_quant_conv(z))
    return torch.clamp(out, min=-1.0, max=1.0)


def _patch_vae_entrypoints(pipe, compiled_encode, compiled_decode):
  """Routes pipe.vae.encode/decode to the compiled wrappers.

  Instance-level monkeypatch; the diffusers dataclasses are built OUTSIDE the jit
  (DiagonalGaussianDistribution's .mode()/.sample() are cheap eager ops). The
  pipeline and vae.config/dtype/temperal_downsample stay untouched.
  """

  def encode(x, return_dict=True):
    posterior = DiagonalGaussianDistribution(compiled_encode(x))
    if not return_dict:
      return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

  def decode(z, return_dict=True):
    import os  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    if os.environ.get("QWEN_VAE_PROFILE") == "1":
      t0 = time.perf_counter()
      sample = compiled_decode(z)
      _ = float(sample.to(torch.float32).sum())  # force sync -> real compute time
      print(f"[vae] decode compute (sync'd): {time.perf_counter() - t0:.3f}s", flush=True)
    else:
      sample = compiled_decode(z)
    if not return_dict:
      return (sample,)
    return DecoderOutput(sample=sample)

  pipe.vae.encode = encode
  pipe.vae.decode = decode


def _install_vae_shard_hooks(mesh, vae, enc, dec):
  """H-axis GSPMD activation sharding for the VAE (WAN recipe, via nn hooks).

  Entry of each wrapper shards [B,C,T,H,W] on H (1D h-mesh over all chips); GSPMD
  auto-inserts conv halo exchange and auto-pads non-divisible H. A post-hook after
  each mid-block AttentionBlock re-anchors H (its HxW flatten destroys H
  propagation). Wrapper outputs are constrained back to replicated for the
  eager/DiT code downstream. Weights stay replicated on the main mesh.
  """
  h_mesh = Mesh(mesh.devices.reshape(-1), ("h",))
  h_shard = NamedSharding(h_mesh, P(None, None, None, "h", None))
  replicated = NamedSharding(mesh, P())

  def _constrain(x, sharding):
    if hasattr(x, "shard_") and getattr(x, "ndim", 0) == 5:
      x.shard_(sharding)
    return x

  def _pre(_module, args):
    _constrain(args[0], h_shard)
    return None

  def _post_attn(_module, _args, output):
    return _constrain(output, h_shard)

  def _post_replicate(_module, _args, output):
    return _constrain(output, replicated)

  for wrapper in (enc, dec):
    wrapper.register_forward_pre_hook(_pre)
    wrapper.register_forward_hook(_post_replicate)
  n = 0
  for root in (vae.encoder, vae.decoder):
    for mod in root.modules():
      if isinstance(mod, qvae.QwenImageResidualBlock):
        mod.register_forward_pre_hook(_pre)
        n += 1
      elif isinstance(mod, qvae.QwenImageResample):
        # Upsample/downsample Resamples CHANGE the H dimension's size. Forcing
        # the h-sharding constraint right before one (found via _vae_bisect.py:
        # the single upsample2d Resample that reaches full resolution took 4.2s
        # alone, ~140x its neighboring same-size blocks) makes GSPMD reshard
        # across a SIZE CHANGE, not just a same-size re-layout — appears to hit a
        # slow path (probably an implicit all-gather-then-redistribute) rather
        # than the cheap local re-shard a same-size ResidualBlock gets. Only
        # shard Resamples that don't change H (mode="none"); let up/downsampling
        # ones run on whatever sharding naturally propagates in.
        if mod.mode == "none":
          mod.register_forward_pre_hook(_pre)
          n += 1
      elif isinstance(mod, qvae.QwenImageAttentionBlock):
        mod.register_forward_hook(_post_attn)
        n += 1
  print(f"VAE H-shard hooks installed on {n} modules.")


@contextlib.contextmanager
def torchax_paused():
  """Runs the body with torchax dispatch off (for the CPU text encoder)."""
  torchax.disable_globally()
  try:
    yield
  finally:
    torchax.enable_globally()


# ---------------------------------------------------------------------------
# Op overrides
# ---------------------------------------------------------------------------
def _register_op_override(env, op, impl):
  env._ops[op] = ops_registry.Operator(
      op, impl, is_jax_function=False, is_user_defined=True, needs_env=False, is_view_op=False
  )


def _torch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env):
  """conv2d via JAX — torchax's native conv2d has gaps on this stack."""
  jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
  res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
  return env.j2t_iso(res)


def _scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, *, env
):
  """SDPA via jax.nn.dot_product_attention — the one option proven stable here.

  Native torchax SDPA returns NaN on the first DiT forward in bf16 (large
  Qwen2.5-VL embeds overflow the softmax exp). jax's dot_product_attention uses a
  numerically stable softmax, is mask-correct, and needs no manual shard_map (GSPMD
  shards from the input sharding).
  """
  jq, jk, jv = env.t2j_iso((query, key, value))
  # torch SDPA layout (B, H, S, D) -> jax dpa layout (B, S, H, D).
  jq = jnp.swapaxes(jq, 1, 2)
  jk = jnp.swapaxes(jk, 1, 2)
  jv = jnp.swapaxes(jv, 1, 2)
  mask = env.t2j_iso(attn_mask) if attn_mask is not None else None
  out = jax.nn.dot_product_attention(jq, jk, jv, mask=mask, scale=scale, is_causal=is_causal)
  out = jnp.swapaxes(out, 1, 2)
  return env.j2t_iso(out)


# ---------------------------------------------------------------------------
# DiT splash-flash attention (replaces jax dpa for the DiT joint-attention call
# only; VAE/other attention keeps using jax dpa via the fallback below).
#
# NOTE on scope vs DiffusionServing's kernel: their version also converts a
# text-padding attn_mask into runtime splash segment ids. In THIS diffusers
# version (0.35.0, vendored — see Part 1 of the port), the DiT's
# ``encoder_hidden_states_mask`` is never actually wired into an ``attn_mask``
# passed to the attention processor (verified by reading
# transformer_qwenimage.py — attention_mask stays None through the whole call
# chain), so there is no mask to convert here; this always uses a dense FullMask.
# If a future diffusers sync adds real mask wiring, this must be revisited (would
# need the segment-id conversion DiffusionServing's kernel has).
# ---------------------------------------------------------------------------
DIT_SPLASH_BLOCK_Q = 1024
DIT_SPLASH_BLOCK_KV = 1024
DIT_SPLASH_BLOCK_KV_COMPUTE = 512


@functools.lru_cache(maxsize=8)
def _get_dit_splash_kernel(padded_seq_len, num_heads_on_device):
  """Memoized FullMask splash kernel construction (trace-time work).

  Built under ``jax.ensure_compile_time_eval()`` so the internal mask arrays are
  concrete, not tracers — otherwise a kernel built inside one jit trace and reused
  from the cache in a later trace raises UnexpectedTracerError (the same class of
  bug as QwenEmbedRope's rope_cache above, just inside the pallas kernel builder).
  """
  mask = splash_attention.MultiHeadMask(
      [splash_attention.FullMask((padded_seq_len, padded_seq_len)) for _ in range(num_heads_on_device)]
  )
  bsizes = splash_attention.BlockSizes(
      block_q=min(DIT_SPLASH_BLOCK_Q, padded_seq_len),
      block_kv=min(DIT_SPLASH_BLOCK_KV, padded_seq_len),
      block_kv_compute=min(DIT_SPLASH_BLOCK_KV_COMPUTE, padded_seq_len),
  )
  with jax.ensure_compile_time_eval():
    return splash_attention.make_splash_mha(mask=mask, block_sizes=bsizes, head_shards=1, q_seq_shards=1)


def _dit_splash_attention(query, key, value, mesh, scale):
  """Self-attention only (query/key/value share seq length). (B, H, S, D) jax arrays.

  Always pads both q and kv to the same block-aligned length (query.shape[-2] ==
  key.shape[-2] for self-attention) rather than assuming the raw sequence length
  already divides the block size — that assumption is what broke the Wan baseline
  earlier this session on an off-default resolution (`kv_block_size should divide
  kv_seq_len`); always padding sidesteps it entirely.
  """

  def _attention_on_slices(q, k, v):
    scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    q = q * scale_factor

    def _pad(x, multiple):
      seq_len = x.shape[1]
      pad_len = (multiple - seq_len % multiple) % multiple
      if pad_len == 0:
        return x, seq_len
      pad_width = [(0, 0)] * x.ndim
      pad_width[1] = (0, pad_len)
      return jnp.pad(x, pad_width), seq_len

    def _kernel_3d(q_3d, k_3d, v_3d):
      block = max(DIT_SPLASH_BLOCK_Q, DIT_SPLASH_BLOCK_KV)
      q_3d, orig_len = _pad(q_3d, block)
      k_3d, _ = _pad(k_3d, block)
      v_3d, _ = _pad(v_3d, block)
      kernel = _get_dit_splash_kernel(q_3d.shape[1], q_3d.shape[0])
      return kernel(q_3d, k_3d, v_3d)[:, :orig_len, ...]

    return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q, k, v)

  # 1D mesh ("tensor"): shard heads on it (24 heads / 4 chips divides cleanly on
  # v6e-4; the sequence-sharding alternative used on larger meshes needs the joint
  # seq length to divide the chip count, which doesn't always hold).
  tp_axis = mesh.axis_names[0]
  part_spec = P(None, tp_axis, None, None)
  sharded_fn = shard_map(_attention_on_slices, mesh=mesh, in_specs=(part_spec,) * 3, out_specs=part_spec, check_rep=False)
  out = sharded_fn(query, key, value)
  return jax.lax.with_sharding_constraint(out, part_spec)


def _make_dit_splash_sdpa(mesh, env):
  def _splash_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    jq, jk, jv = env.t2j_iso((query, key, value))  # already (B, H, S, D) — no transpose needed
    out = _dit_splash_attention(jq, jk, jv, mesh, scale)
    return env.j2t_iso(out)

  return _splash_sdpa


def _make_sdpa_with_dit_flash(env, mesh):
  """Dispatches DiT joint self-attention to splash, everything else to jax dpa.

  Predicate mirrors DiffusionServing's: non-causal, self-attention (matching q/kv
  seq length), head_dim 128 (unique to the DiT among this pipeline's attention
  calls — the VAE's QwenImageAttentionBlock uses a much larger single-head dim, so
  this reliably distinguishes them without needing an explicit caller tag).
  """
  dense = functools.partial(_scaled_dot_product_attention, env=env)
  splash = _make_dit_splash_sdpa(mesh, env)

  def _dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if not is_causal and query.shape[-1] == 128 and query.shape[-2] == key.shape[-2]:
      return splash(query, key, value, attn_mask=attn_mask, scale=scale)
    return dense(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

  return _dispatch


# ---------------------------------------------------------------------------
# Hybrid VL: vision tower + processor on CPU, Qwen2.5-VL language model on TPU.
# This is the biggest win in DiffusionServing's own profiling (~85% of a whole
# edit was CPU text encoding before this; the LANGUAGE MODEL was the bottleneck,
# not the vision tower — vision-on-CPU alone is only ~0.5s — so unlike the other
# three steps, vision-on-TPU is NOT ported here; it's low priority per their own
# findings and needs extra windowed-attention monkeypatching this repo's
# transformers version wasn't checked against).
# ---------------------------------------------------------------------------
def _build_lm_inputs(model, cpu_embed_weight, input_ids, image_embeds):
  """CPU: token embeds + scatter image features into image-token positions."""
  inputs_embeds = torch.nn.functional.embedding(input_ids, cpu_embed_weight)
  img = image_embeds.to(inputs_embeds.dtype)
  image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=img)
  return inputs_embeds.masked_scatter(image_mask, img)


def _run_lm(compiled_lm, env, mesh, inputs_embeds):
  """TPU: run the compiled language model, return CPU [1, seq, hidden] bf16.

  return_dict=False -> a plain tuple (last_hidden_state,) rather than
  BaseModelOutputWithPast (use_cache=False/no output_hidden_states/attentions
  requested -> ModelOutput.to_tuple() drops the other, all-None fields) since jax
  jit can't return an arbitrary dataclass without a registered pytree node.
  """
  x = env.to_xla(inputs_embeds)
  with mesh:
    out = compiled_lm(
        inputs_embeds=x, position_ids=None, attention_mask=None, use_cache=False, return_dict=False
    )
  hidden = out[0]
  return hidden.torch() if hasattr(hidden, "torch") else hidden


def _encode_prompt_hybrid(
    pipe,
    env,
    mesh,
    compiled_lm,
    cpu_embed_weight,
    prompt,
    negative,
    image,
    need_negative,
    compiled_visual=None,
    cpu_vision_inv_freq=None,
):
  """Reproduces QwenImageEditPipeline._get_qwen_prompt_embeds with the LM on TPU.

  CPU (torchax off): processor -> pixel_values/input_ids; vision tower once (the
  image is identical for pos+neg) — on CPU via get_image_features, or on TPU via
  the compiled _VisionTPU wrapper if `compiled_visual` is given. TPU: the compiled
  Qwen2_5_VLTextModel with position_ids=None, attention_mask=None (batch=1 -> pure
  causal via our SDPA override's is_causal path). Then drop the first `drop_idx`
  template positions, matching the stock method. Returns CPU torch (pe, pem, npe,
  npem); npe/npem are None when `need_negative` is False (true_cfg_scale <= 1 ->
  the pipeline ignores the negative branch, so its LM forward is skipped
  entirely).
  """
  model = pipe.text_encoder.model
  template = pipe.prompt_template_encode
  drop_idx = pipe.prompt_template_encode_start_idx

  with torchax_paused():
    mi_pos = pipe.processor(text=[template.format(prompt)], images=image, padding=True, return_tensors="pt")
    mi_neg = None
    if need_negative:
      mi_neg = pipe.processor(text=[template.format(negative)], images=image, padding=True, return_tensors="pt")

    if compiled_visual is None:
      # get_image_features returns a tuple of per-image embeddings directly in
      # this transformers version (no .pooler_output wrapper, unlike the version
      # DiffusionServing's runner targeted) — torch.cat is the adaptation needed.
      image_embeds = torch.cat(model.get_image_features(mi_pos["pixel_values"], mi_pos["image_grid_thw"]), dim=0)
    else:
      precompute = _vision_precompute(model.visual, mi_pos["image_grid_thw"], cpu_vision_inv_freq)

  if compiled_visual is not None:
    image_embeds = _run_visual(
        compiled_visual, env, mesh, mi_pos["pixel_values"], mi_pos["image_grid_thw"], precompute
    )

  with torchax_paused():
    ie_pos = _build_lm_inputs(model, cpu_embed_weight, mi_pos["input_ids"], image_embeds)
    ie_neg = None
    if need_negative:
      ie_neg = _build_lm_inputs(model, cpu_embed_weight, mi_neg["input_ids"], image_embeds)

  pe = _run_lm(compiled_lm, env, mesh, ie_pos)[:, drop_idx:, :].contiguous()
  pem = torch.ones((1, pe.shape[1]), dtype=torch.long)
  npe = npem = None
  if need_negative:
    npe = _run_lm(compiled_lm, env, mesh, ie_neg)[:, drop_idx:, :].contiguous()
    npem = torch.ones((1, npe.shape[1]), dtype=torch.long)
  return pe, pem, npe, npem


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------
def _edit(
    pipe,
    env,
    mesh,
    args,
    compiled_lm=None,
    cpu_embed_weight=None,
    compiled_visual=None,
    cpu_vision_inv_freq=None,
    profile=False,
):
  import time  # pylint: disable=import-outside-toplevel

  now = time.perf_counter
  img = Image.open(args.image).convert("RGB")
  do_cfg = args.true_cfg_scale > 1

  t0 = now()
  if compiled_lm is not None:
    prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = _encode_prompt_hybrid(
        pipe,
        env,
        mesh,
        compiled_lm,
        cpu_embed_weight,
        args.prompt,
        args.negative_prompt,
        img,
        need_negative=do_cfg,
        compiled_visual=compiled_visual,
        cpu_vision_inv_freq=cpu_vision_inv_freq,
    )
  else:
    with torchax_paused():
      prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
          prompt=[args.prompt], image=img, device="cpu"
      )
      negative_prompt_embeds = negative_prompt_embeds_mask = None
      if do_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
            prompt=[args.negative_prompt], image=img, device="cpu"
        )
  t_vl = now() - t0

  t1 = now()
  to_xla = env.to_xla
  prompt_embeds, prompt_embeds_mask = to_xla(prompt_embeds), to_xla(prompt_embeds_mask)
  if do_cfg:
    negative_prompt_embeds = to_xla(negative_prompt_embeds)
    negative_prompt_embeds_mask = to_xla(negative_prompt_embeds_mask)
  t_prep = now() - t1

  generator = torch.Generator().manual_seed(args.seed)
  step_ts = []
  pipe_kwargs = dict(
      image=img,
      prompt_embeds=prompt_embeds,
      prompt_embeds_mask=prompt_embeds_mask,
      negative_prompt_embeds=negative_prompt_embeds,
      negative_prompt_embeds_mask=negative_prompt_embeds_mask,
      num_inference_steps=args.num_inference_steps,
      true_cfg_scale=args.true_cfg_scale,
      generator=generator,
  )
  if profile:
    # jax dispatch is async, so the tqdm loop finishes almost instantly regardless
    # of real compute time; force a TPU->host sync at each step boundary (like
    # DiffusionServing's own QWEN_PROFILE callback) to get real per-stage timing.
    def _cb(_pipe, _step, _timestep, kw):
      lat = kw.get("latents")
      if lat is not None:
        _ = float(lat.to(torch.float32).sum())
      step_ts.append(now())
      return kw

    pipe_kwargs["callback_on_step_end"] = _cb
    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

  t2 = now()
  with mesh:
    out = pipe(**pipe_kwargs)  # VAE decode + postprocess (numpy/PIL) both happen
    # inside __call__ before it returns -- diffusers pipelines materialize
    # `.images` before returning, so there's no further work to isolate outside
    # this call; can't split VAE-decode-compute from PIL/numpy conversion via
    # wall-clock timestamps from the outside without instrumenting the pipeline
    # itself.
  image = out.images[0]
  t_pipe = now() - t2
  if profile:
    n = len(step_ts)
    if n > 1:
      vae_enc_step0 = step_ts[0] - t2
      denoise = step_ts[-1] - step_ts[0]
      vae_dec_and_postproc = (t2 + t_pipe) - step_ts[-1]
      print(
          f"[stage] VL={t_vl:.2f}s prep={t_prep:.2f}s pipe={t_pipe:.2f}s || "
          f"VAEenc+step0={vae_enc_step0:.2f}s denoise(remaining {n - 1} steps)={denoise:.2f}s "
          f"(~{denoise / max(n - 1, 1):.2f}s/step) VAEdec+postprocess={vae_dec_and_postproc:.2f}s",
          flush=True,
      )
    else:
      print(f"[stage] VL={t_vl:.2f}s prep={t_prep:.2f}s pipe(denoise+VAE)={t_pipe:.2f}s", flush=True)
  return image


def parse_args():
  parser = argparse.ArgumentParser(description="Qwen-Image-Edit on TPU (torchax)")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit")
  parser.add_argument("--image", type=str, required=True, help="Path to the input image")
  parser.add_argument("--prompt", type=str, required=True, help="Edit instruction")
  parser.add_argument("--negative_prompt", type=str, default=" ")
  parser.add_argument("--num_inference_steps", type=int, default=40)
  parser.add_argument("--true_cfg_scale", type=float, default=4.0)
  parser.add_argument("--lightning_lora", type=str, default=None, help="Optional LoRA repo id")
  parser.add_argument("--lightning_lora_weight_name", type=str, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", type=str, default="qwen_edit_output.png")
  parser.add_argument(
      "--text_encoder_cpu_only",
      action="store_true",
      help="Disable the LM-on-TPU hybrid path; run the whole text encoder on CPU (slower, for A/B or debugging).",
  )
  parser.add_argument(
      "--vision_cpu_only",
      action="store_true",
      help="Keep the vision tower on CPU (language model still moves to TPU); for A/B or debugging.",
  )
  parser.add_argument(
      "--vae_shard_off",
      action="store_true",
      help="Keep VAE JIT-compiled but skip the H-axis spatial-sharding hooks; for A/B diagnosing perf.",
  )
  parser.add_argument(
      "--repeat",
      type=int,
      default=1,
      help="Run the edit N times in the same process (only the last is saved) to measure steady-state "
      "per-edit latency once compilation is already warm — the first call always includes first-trace XLA compile.",
  )
  parser.add_argument("--profile", action="store_true", help="Print a VL/prep/pipe timing breakdown per edit.")
  return parser.parse_args()


def main():
  args = parse_args()
  print(args)

  torch.set_default_dtype(torch.bfloat16)

  # 1) Load the pipeline on CPU, plain diffusers (torchax not yet enabled).
  pipe = QwenImageEditPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

  if args.lightning_lora:
    kwargs = {"weight_name": args.lightning_lora_weight_name} if args.lightning_lora_weight_name else {}
    pipe.load_lora_weights(args.lightning_lora, **kwargs)
    # Fuse BEFORE compiling: otherwise the adapter's lora_A/lora_B names don't
    # match TRANSFORMER_SHARDINGS and get silently dropped at compile time.
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    print(f"Loaded + fused LoRA: {args.lightning_lora} ({args.lightning_lora_weight_name or 'default'})")

  # Pin execution device to the torchax XLA device: otherwise diffusers infers cpu
  # (the text encoder stays on CPU) and creates image/noise/latents as plain CPU
  # tensors that then mix with the XLA transformer/VAE.
  pipe_cls = pipe.__class__
  pipe.__class__ = type(
      f"{pipe_cls.__name__}TPU", (pipe_cls,), {"_execution_device": property(lambda _s: torch.device("jax"))}
  )

  # 2) torchax env + mesh over all available chips.
  torchax.enable_globally()
  env = torchax.default_env()
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("tensor",))
  env._mesh = mesh
  env.default_device_or_sharding = NamedSharding(mesh, P())
  print(f"Mesh: {jax.device_count()} chips on axis 'tensor'")

  # 3) Op overrides. SDPA dispatches DiT joint self-attention to a splash-flash
  # kernel (removes jax dpa's S x S fp32 logits materialization); everything else
  # (VAE mid-block attention, etc.) still uses jax dpa.
  _register_op_override(env, torch.nn.functional.conv2d, functools.partial(_torch_conv2d, env=env))
  _register_op_override(
      env, torch.nn.functional.scaled_dot_product_attention, _make_sdpa_with_dit_flash(env, mesh)
  )

  # 4) Compile + shard the DiT. Weights must be moved onto the XLA device BEFORE
  # compiling (torchax.compile's .params/.buffers only carry torchax XLA tensors,
  # with the .apply_jax_ sharding hook, if the wrapped module's parameters were
  # already XLA tensors when compiled). img_shapes must be a jit static arg (its
  # ints are used for RoPE slicing/reshape), so mark it static and wrap with
  # _StaticImgShapes to make the list diffusers passes in hashable first.
  _move_module(pipe.transformer, env)  # also moves pos_embed.pos_freqs/neg_freqs
  opts = torchax.CompileOptions(
      jax_jit_kwargs={"static_argnames": ("return_dict", "img_shapes", "txt_seq_lens")}
  )
  compiled_tf = torchax.compile(pipe.transformer, opts)
  compiled_tf.params = _shard_weight_dict(compiled_tf.params, TRANSFORMER_SHARDINGS, mesh)
  compiled_tf.buffers = _shard_weight_dict(compiled_tf.buffers, TRANSFORMER_SHARDINGS, mesh)
  pipe.transformer = _StaticImgShapes(compiled_tf)

  # 5) VAE: JIT-compile encode/decode (wrapping the whole module only jits its
  # `forward`, which the pipeline never calls, so it would run torchax-eager) +
  # H-axis spatial sharding of activations (weights stay replicated).
  _move_module(pipe.vae, env)
  enc, dec = _VaeEncode(pipe.vae), _VaeDecode(pipe.vae)
  if not args.vae_shard_off:
    _install_vae_shard_hooks(mesh, pipe.vae, enc, dec)
  compiled_encode = torchax.compile(enc)
  compiled_decode = torchax.compile(dec)
  _patch_vae_entrypoints(pipe, compiled_encode, compiled_decode)

  # 6) Text encoder (Qwen2.5-VL): the language model (28 decoder layers) moves to
  # TPU — this is where ~85% of a whole edit's CPU time went in DiffusionServing's
  # own profiling. The vision tower ALSO moves to TPU (its per-window attention
  # fused into one masked SDPA so the graph is trace-safe); processor/tokenizer
  # stay on CPU either way (cheap, string/PIL work).
  compiled_lm = cpu_embed_weight = compiled_visual = cpu_vision_inv_freq = None
  if not args.text_encoder_cpu_only:
    te = pipe.text_encoder
    # Capture the CPU embedding weight BEFORE moving language_model to XLA (which
    # frees the CPU copy) — needed to build inputs_embeds (embed + image-feature
    # scatter) on CPU before the LM runs.
    cpu_embed_weight = te.model.get_input_embeddings().weight.detach().clone()
    # Force stable SDPA attention -> our jax.nn.dot_product_attention override
    # (native eager attention risks the same bf16 overflow-to-NaN seen on the DiT).
    te.config._attn_implementation = "sdpa"
    te.config.text_config._attn_implementation = "sdpa"
    _move_module(te.model.language_model, env)
    lm_opts = torchax.CompileOptions(jax_jit_kwargs={"static_argnames": ("use_cache", "return_dict")})
    compiled_lm = torchax.compile(te.model.language_model, lm_opts)
    compiled_lm.params = _shard_weight_dict(compiled_lm.params, LM_SHARDINGS, mesh)
    compiled_lm.buffers = _shard_weight_dict(compiled_lm.buffers, LM_SHARDINGS, mesh)
    print("Compiled Qwen2.5-VL language model on TPU (hybrid VL path).")

    if not args.vision_cpu_only:
      # Vision tower is small (~1.3GB) — replicate, no sharding needed. Capture
      # rotary_pos_emb.inv_freq on CPU BEFORE the move — _vision_precompute needs
      # it (see _rot_pos_emb_cpu's docstring for why it can't just read the
      # (now-XLA) buffer off the compiled module at request time).
      cpu_vision_inv_freq = te.model.visual.rotary_pos_emb.inv_freq.detach().clone()
      _patch_vision_attention()
      _move_module(te.model.visual, env)
      compiled_visual = torchax.compile(_VisionTPU(te.model.visual))
      print("Compiled Qwen2.5-VL vision tower on TPU (hybrid VL path).")

  print("Model loaded. Running edit...")
  import time  # pylint: disable=import-outside-toplevel

  image = None
  for i in range(args.repeat):
    t0 = time.perf_counter()
    image = _edit(
        pipe,
        env,
        mesh,
        args,
        compiled_lm=compiled_lm,
        cpu_embed_weight=cpu_embed_weight,
        compiled_visual=compiled_visual,
        cpu_vision_inv_freq=cpu_vision_inv_freq,
        profile=args.profile,
    )
    print(f"[timing] edit #{i}: {time.perf_counter() - t0:.2f}s")
  image.save(args.output)
  print(f"Saved edited image to {args.output}")


if __name__ == "__main__":
  main()
