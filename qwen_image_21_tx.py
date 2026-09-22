"""Qwen-Image-2.1 on TPU via torchax.

Sibling of ``qwen_image_edit_tx.py``. Despite the shared name the model is a
different one, not a new revision:

* the DiT is **single-stream** (``QwenImage21Transformer2DModel``, 32 layers,
  32 heads) where Qwen-Image-Edit is double-stream with 60. Each block is just
  ``attn`` plus a SwiGLU ``img_mlp``, every linear is bias-free, the
  LayerNorms carry no weights, and modulation is computed once by the parent
  and sliced per block instead of living in the blocks.
* the text encoder is **Qwen3-VL**, not Qwen2.5-VL.
* the VAE is ``AutoencoderKLQwenImage21``.

One pipeline serves both text-to-image and image-conditioned generation, so
``--image`` is optional here.

**Which branch this lives on matters.** The QwenImage21 classes landed in
PR #14804 on the day the model shipped and are in no tagged release, so this
branch is cut from upstream ``main`` rather than from the 0.33.1 base the
older scripts sit on. That base carried hand-backported copies of
QwenImage/Hunyuan/Cog to make them work; upstream has all of it natively now,
so nothing is lost by moving -- but the older scripts stay on the old branch,
where the backports they were written against still are.

    pip install -e .                 # this branch
    pip install "transformers>=5.17"

What is on device: the DiT (tensor-parallel), the VAE decode, and all of
Qwen3-VL -- its 36 decoder layers tensor-parallel, its vision tower
replicated. Measured on v6e-8 at 1024x1024, 20 steps: 1.36-1.38 s for
text-to-image and 3.03-3.28 s for an edit, against 183 s and 450 s on CPU.

Steady state, with the first run's compilation discarded; ``--runs 4``
reproduces it.

Three things had to be restructured to get there, each because some shape
depends on tensor *contents* and so cannot be traced.

1. **The DiT's prologue.** ``repeats = where(img_mask, 4, 1)`` feeds a
   ``repeat_interleave`` whose output length is data-dependent, the joint
   sequence is filled through a boolean mask, and ``prefix_len`` comes out of
   an ``int()`` on a tensor. None of it depends on the latents or the
   timestep -- it is a function of ``img_mask``, ``img_shapes`` and the prompt
   mask, constant for the whole loop -- so the host computes it once and the
   jitted core takes it ready-made, with the two data-dependent ops rewritten
   as a gather over a precomputed index and an ``index_copy``.

2. **Qwen3-VL.** Same split: everything up to the decoder loop stays on the
   host (``compute_3d_position_ids`` and the three ``get_vision_*`` helpers
   all go through ``.item()``/``.tolist()``), the layers run on device. The
   vision tower's variable-length attention splits q/k/v with
   ``torch.split(lengths.tolist())``; one fused SDPA with a block-diagonal
   mask is the same computation, and a single condition image has one segment
   so the mask is None.

3. **The VAE decode**, which is Wan's video decoder specialised to one frame.
   ``feat_cache`` cannot be None: Wan's ``upsample3d`` branch skips
   ``time_conv`` exactly when a cache list is present and the slot is empty,
   so passing None doubles the temporal axis -- and since
   ``QwenImage21CausalConv3d`` is an ``nn.Conv2d`` that folds the frame away
   with ``squeeze(2)``, the squeeze then silently does nothing and the
   convolution gets a 5-D input against a 2-D weight.

Things deliberately not done, with the measurements behind them:

* **No splash attention.** At 1024x1024 the sequence is only 4096 tokens, and
  attention is 13% of the DiT's FLOPs against the MLP's 60%. Time scales with
  FLOPs (2.25x the tokens costs 2.58x, predicted 2.63x), so a faster
  attention kernel cannot move much.
* **Nothing done about the collectives**, which are 38% of device op time and
  fully exposed -- none of it lands on the async row. Every lever was
  measured: the latency-hiding scheduler and the async-collective-fusion
  family do nothing on v6e, SparseCore collective offload is 5-7% *slower*,
  replicating the MLP to avoid all-reduces loses because it is 60% of the
  FLOPs, and every other mesh is worse (1.65 s per image at 1x8 against 2.00
  at 2x4, 3.06 at 4x2, out of memory at 8x1). Eight-way tensor parallel is
  already the right configuration.
"""

import argparse
import contextlib
import os
import functools
import math
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import torchax
from torchax.ops import jaten


# Single-stream MMDiT. No text stream (`add_q_proj` / `to_add_out` /
# `txt_mlp`) and the FFN is a SwiGLU triple rather than
# `FeedForward(net.0.proj, net.2)`. Every linear is bias-free, so no bias
# rules. At tensor=8: 32 heads -> 4 per chip, inner dim 4096, mlp 12288 ->
# 1536. Both clean.
TRANSFORMER_SHARDINGS = {
    r".*attn\.to_q\.weight": ("tensor", None),
    r".*attn\.to_k\.weight": ("tensor", None),
    r".*attn\.to_v\.weight": ("tensor", None),
    r".*attn\.to_out\.0\.weight": (None, "tensor"),
    r".*img_mlp\.proj\.weight": ("tensor", None),
    r".*img_mlp\.gate_layer\.weight": ("tensor", None),
    r".*img_mlp\.out\.weight": (None, "tensor"),
}

# Qwen3-VL's decoder. A real tensor-parallel split, unlike Qwen2.5-VL, whose
# 28 query and 4 key/value heads divide neither 8 nor each other and forced
# the Edit script into storage sharding: here it is 32 and 8, with a 12288
# intermediate, all divisible by the axis.
LM_SHARDINGS = {
    r".*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight": ("tensor",),
    r".*layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight": ("tensor",),
}


@contextlib.contextmanager
def torchax_paused():
  """Runs plain PyTorch inside an otherwise torchax-enabled process."""
  torchax.disable_globally()
  try:
    yield
  finally:
    torchax.enable_globally()


def _register_op_override(env, op, impl):
  from torchax.ops import ops_registry

  env._ops[op] = ops_registry.Operator(
      op, impl, is_jax_function=False, is_user_defined=True,
      needs_env=False, is_view_op=False,
  )


def _torch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
                  groups=1, *, env):
  """conv2d via jax -- works around torchax's gaps."""
  jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
  res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation,
                           groups)
  return env.j2t_iso(res)


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
  out = {}
  for k, v in weight_dict.items():
    for target, spec in sharding_dict.items():
      if re.fullmatch(target, k) is not None:
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*spec)))
        break
    else:
      v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    out[k] = v
  return out


def _move_module(module, env):
  """Moves a module's weights onto the mesh, replicated.

  `state_dict()` covers params and persistent buffers only, but
  `torchax.compile` finds what to shard with `extract_all_buffers`, which
  grabs any tensor-valued attribute -- including non-persistent ones like
  rope `inv_freq`. Anything still a plain CPU tensor at compile time raises
  `AttributeError: ... no attribute 'apply_jax_'` later, so walk the same way.
  """
  with jax.default_device("cpu"):
    sd = module.state_dict()
    sd = env.to_xla(sd)
    module.load_state_dict(sd, assign=True)
  from torchax.interop import extract_all_buffers

  _, buffers = extract_all_buffers(module)
  for qual, tensor in buffers.items():
    if hasattr(tensor, "apply_jax_"):
      continue
    *prefix, attr = qual.split(".")
    owner = module
    for p in prefix:
      owner = getattr(owner, p)
    try:
      setattr(owner, attr, env.to_xla(tensor))
    except Exception:  # pylint: disable=broad-except
      pass  # read-only property, not a real buffer


# ---------------------------------------------------------------------------
# DiT: host prologue + traceable core
# ---------------------------------------------------------------------------


class _JitCore(torch.nn.Module):
  """The part of the DiT that can be traced.

  Holds the original submodules under their original names, so
  TRANSFORMER_SHARDINGS still matches.
  """

  def __init__(self, model):
    super().__init__()
    self.img_in = model.img_in
    self.txt_in = model.txt_in
    self.time_text_embed = model.time_text_embed
    self.modulation = model.modulation
    self.transformer_blocks = model.transformer_blocks
    self.norm_out = model.norm_out
    self.proj_out = model.proj_out
    self._causal_condition = bool(model.config.causal_condition)

  def forward(self, hidden_states, encoder_hidden_states, timestep,
              expand_idx, image_positions, rotary_emb, modulation_mask,
              key_valid, pad_tokens=0, segments=None):
    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    joint = torch.cat(
        [
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                encoder_hidden_states.shape[0], pad_tokens,
                encoder_hidden_states.shape[2]),
        ],
        dim=1,
    )
    joint = joint[:, expand_idx]  # was repeat_interleave(repeats, dim=1)
    joint = joint.index_copy(1, image_positions, hidden_states)  # was mask set

    timestep = timestep.to(hidden_states.dtype)
    if self._causal_condition:
      timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
    temb = self.time_text_embed(timestep, hidden_states)
    modulation = self.modulation(temb)

    seg = list(segments) if segments is not None else None
    for block in self.transformer_blocks:
      joint = block(
          hidden_states=joint, modulation=modulation, rotary_emb=rotary_emb,
          attention_mask=None, target_token_mask=modulation_mask,
          layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
          segments=seg, key_valid=key_valid,
      )
    joint = self.norm_out(joint, temb, modulation_mask)
    return self.proj_out(joint)


class _Prologue:
  """Host-side prologue plus the jitted core, behind the DiT's signature."""

  def __init__(self, model, core, env):
    self._model = model
    self._core = core
    self._env = env
    self._cache = {}

  def __getattr__(self, name):
    return getattr(self._model, name)

  def _build(self, img_mask_cpu, img_shapes, enc_mask_cpu):
    from diffusers.models.transformers import transformer_qwenimage21 as q21

    repeats = torch.where(img_mask_cpu, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask_cpu[0], repeats)
    expand_idx = torch.repeat_interleave(torch.arange(repeats.shape[0]),
                                         repeats)
    image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
    rotary_emb = self._model.pos_embed(img_shapes[0], image_pad_mask,
                                       device=torch.device("cpu"))
    image_ids, target_token_mask = self._model.build_token_metadata(
        image_pad_mask, img_shapes[0])
    modulation_mask = (target_token_mask
                       if self._model.config.causal_condition else None)

    key_valid = None
    if enc_mask_cpu is not None:
      key_valid = torch.ones(enc_mask_cpu.shape[0], image_pad_mask.shape[0],
                             dtype=torch.bool)
      text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
      vlm_text_positions = ~img_mask_cpu[0][: enc_mask_cpu.shape[1]]
      key_valid[:, text_positions] = enc_mask_cpu.bool()[:, vlm_text_positions]

    prefix_len = int((~target_token_mask).sum())
    segments = q21._qwenimage21_prefix_segments(image_ids, prefix_len)
    return {
        "expand_idx": expand_idx,
        "image_positions": image_positions,
        "rotary_emb": rotary_emb,
        "modulation_mask": modulation_mask,
        "key_valid": key_valid,
        "pad_tokens": math.prod(img_shapes[0][-1]) // 4,
        "segments": tuple(tuple(x) for x in segments) if segments else None,
    }

  def __call__(self, **kwargs):
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    import torchax.tensor as tx

    def cpu(t):
      if t is None:
        return None
      return self._env.j2t_copy(t._elem) if isinstance(t, tx.Tensor) else t

    img_mask = cpu(kwargs["img_mask"])
    enc_mask = cpu(kwargs.get("encoder_hidden_states_mask"))
    img_shapes = kwargs["img_shapes"]

    with torchax_paused():
      key = (
          tuple(tuple(tuple(t) for t in row) for row in img_shapes),
          tuple(img_mask.reshape(-1).tolist()),
          None if enc_mask is None else tuple(enc_mask.reshape(-1).tolist()),
      )
      pre = self._cache.get(key)
      if pre is None:
        pre = self._build(img_mask, img_shapes, enc_mask)
        self._cache[key] = pre

    if not pre.get("_on_device"):
      for k in ("expand_idx", "image_positions", "rotary_emb",
                "modulation_mask", "key_valid"):
        v = pre[k]
        if torch.is_tensor(v):
          pre[k] = self._env.to_xla(v)
        elif isinstance(v, (tuple, list)):
          pre[k] = type(v)(self._env.to_xla(x) if torch.is_tensor(x) else x
                           for x in v)
      pre["_on_device"] = True

    out = self._core(
        kwargs["hidden_states"], kwargs["encoder_hidden_states"],
        kwargs["timestep"], pre["expand_idx"], pre["image_positions"],
        pre["rotary_emb"], pre["modulation_mask"], pre["key_valid"],
        pad_tokens=pre["pad_tokens"], segments=pre["segments"],
    )
    if kwargs.get("return_dict", True):
      return Transformer2DModelOutput(sample=out)
    return (out,)


# ---------------------------------------------------------------------------
# Qwen3-VL: host prologue + traceable decoder and vision tower
# ---------------------------------------------------------------------------


def patch_vision_attention():
  """One fused SDPA instead of a per-segment split.

  The stock non-flash path does `torch.split(tensor, lengths.tolist(), ...)`,
  so the number and size of chunks come from tensor contents. A single
  condition image gives `cu_seqlens` of shape (2,) -- one segment, full
  attention, no mask -- and the branch is on the shape, which is static.
  """
  import torch.nn.functional as F
  from transformers.models.qwen3_vl import modeling_qwen3_vl as m

  if getattr(m.Qwen3VLVisionAttention, "_tpu_fused_patched", False):
    return

  def forward(self, hidden_states, cu_seqlens, position_embeddings=None,
              max_seqlen=None, **kwargs):
    del max_seqlen
    seq_length = hidden_states.shape[0]
    q, k, v = (self.qkv(hidden_states)
               .reshape(seq_length, 3, self.num_heads, -1)
               .permute(1, 0, 2, 3).unbind(0))
    cos, sin = position_embeddings
    q, k = m.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    attn_mask = None if cu_seqlens.shape[0] == 2 else kwargs.get(
        "vision_segment_mask")
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                         scale=self.scaling)
    return self.proj(out.transpose(1, 2).reshape(seq_length, -1).contiguous())

  m.Qwen3VLVisionAttention.forward = forward
  m.Qwen3VLVisionAttention._tpu_fused_patched = True


class Qwen3VLDecoderCore(torch.nn.Module):
  """The 36 decoder layers, returning the pre-norm hidden state.

  The pipeline reads `hidden_states[-1]`, which has to be the last layer's
  output *before* the final RMSNorm -- what the DiT was trained against.
  Upstream gets there with a forward hook that makes `norm` return its input;
  returning the pre-norm state is the same thing without depending on a hook
  firing inside a traced graph.
  """

  def __init__(self, language_model):
    super().__init__()
    self.layers = language_model.layers

  def forward(self, inputs_embeds, position_embeddings, attention_mask,
              text_position_ids, deepstack_embeds=None,
              visual_positions=None, deepstack_depth=0):
    hidden_states = inputs_embeds
    for layer_idx, layer in enumerate(self.layers):
      hidden_states = layer(
          hidden_states, attention_mask=attention_mask,
          position_ids=text_position_ids, past_key_values=None,
          position_embeddings=position_embeddings, use_cache=False,
      )
      if deepstack_embeds is not None and layer_idx < deepstack_depth:
        # was: hidden_states[visual_pos_masks, :] += visual_embeds
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat = flat.index_add(0, visual_positions,
                              deepstack_embeds[layer_idx].to(flat.dtype))
        hidden_states = flat.reshape(hidden_states.shape)
    return hidden_states


class Qwen3VLVisionCore(torch.nn.Module):
  """The vision tower with its grid-dependent prologue lifted out.

  Replicated, not sharded: `qkv` packs q, k and v into one linear laid out
  [3][heads][dim], so splitting that dimension would cut across the
  boundaries. 0.58B is cheap to copy.
  """

  def __init__(self, visual):
    super().__init__()
    self.patch_embed = visual.patch_embed
    self.pos_embed = visual.pos_embed
    self.rotary_pos_emb = visual.rotary_pos_emb
    self.blocks = visual.blocks
    self.deepstack_merger_list = visual.deepstack_merger_list
    self.merger = visual.merger
    self._deepstack_indexes = list(visual.deepstack_visual_indexes)

  def forward(self, hidden_states, interp_indices, interp_weights,
              position_ids, cu_seqlens):
    hidden_states = self.patch_embed(hidden_states)
    pos = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
    hidden_states = hidden_states + pos.to(hidden_states.dtype)
    position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

    deepstack = []
    for layer_num, blk in enumerate(self.blocks):
      hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                          position_embeddings=position_embeddings)
      if layer_num in self._deepstack_indexes:
        merger = self.deepstack_merger_list[
            self._deepstack_indexes.index(layer_num)]
        deepstack.append(merger(hidden_states))
    return self.merger(hidden_states), torch.stack(deepstack, dim=0)


class Qwen3VLSplit:
  """Stands in for the pipeline's `text_encoder`."""

  def __init__(self, model, core, env, vision_core=None):
    self._model = model
    self._core = core
    self._env = env
    self._vision_core = vision_core
    self._vision_cache = {}

  def __getattr__(self, name):
    return getattr(self._model, name)

  def _vision_precompute(self, visual, grid_thw):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    key = tuple(int(x) for x in grid_thw.flatten())
    hit = self._vision_cache.get(key)
    if hit is not None:
      return hit
    interp_indices, interp_weights = (
        get_vision_interpolation_indices_and_weights(
            grid_thw, num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=visual.config.spatial_merge_size, kwargs={}))
    position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size,
                                           kwargs={})
    cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, visual.config,
                                                 kwargs={})
    pre = {"interp_indices": interp_indices, "interp_weights": interp_weights,
           "position_ids": position_ids, "cu_seqlens": cu_seqlens}
    self._vision_cache[key] = pre
    return pre

  def _run_vision(self, visual, pixel_values, grid_thw):
    pre = self._vision_precompute(visual, grid_thw)
    to = self._env.to_xla
    merged, deepstack = self._vision_core(
        to(pixel_values.type(visual.dtype)), to(pre["interp_indices"]),
        to(pre["interp_weights"]), to(pre["position_ids"]),
        to(pre["cu_seqlens"]))
    merged = self._env.j2t_copy(merged._elem)
    deepstack = self._env.j2t_copy(deepstack._elem)
    split = (grid_thw.prod(-1) // visual.config.spatial_merge_size**2).tolist()
    return torch.split(merged, split), list(deepstack)

  def _host_prologue(self, kwargs):
    from transformers.masking_utils import create_causal_mask

    m = self._model.model
    lm = m.language_model
    input_ids = kwargs["input_ids"]
    attention_mask = kwargs.get("attention_mask")
    inputs_embeds = lm.embed_tokens(input_ids)
    deepstack = visual_positions = None

    pixel_values = kwargs.get("pixel_values")
    if pixel_values is not None:
      grid_thw = kwargs["image_grid_thw"]
      if self._vision_core is not None:
        pooler, deep_feats = self._run_vision(m.visual, pixel_values, grid_thw)
      else:
        out = m.get_image_features(pixel_values, grid_thw, return_dict=True)
        pooler, deep_feats = out.pooler_output, out.deepstack_features
      image_embeds = torch.cat(pooler, dim=0).to(inputs_embeds.dtype)
      image_mask, _ = m.get_placeholder_mask(
          input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
      inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
      visual_positions = image_mask[..., 0].reshape(-1).nonzero(
          as_tuple=True)[0]
      deepstack = torch.stack(list(deep_feats), dim=0)

    # compute_3d_position_ids is the model's own wrapper around
    # get_rope_index -- whose second positional is mm_token_type_ids, not
    # image_grid_thw, so calling it directly is easy to get wrong.
    position_ids = m.compute_3d_position_ids(
        input_ids=input_ids, inputs_embeds=inputs_embeds,
        image_grid_thw=kwargs.get("image_grid_thw"),
        video_grid_thw=kwargs.get("video_grid_thw"),
        attention_mask=attention_mask, past_key_values=None,
        mm_token_type_ids=kwargs.get("mm_token_type_ids"))

    if position_ids is None:
      text_position_ids = rope_ids = None
    elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
      text_position_ids, rope_ids = position_ids[0], position_ids[1:]
    elif position_ids.ndim == 2:
      text_position_ids = position_ids
      rope_ids = position_ids[None, ...].expand(3, -1, -1)
    else:
      text_position_ids, rope_ids = None, position_ids

    causal = create_causal_mask(
        config=lm.config, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=None,
        position_ids=text_position_ids)
    if rope_ids is None:
      seq = torch.arange(inputs_embeds.shape[1])
      text_position_ids = seq.view(1, -1).expand(inputs_embeds.shape[0], -1)
      rope_ids = seq.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    position_embeddings = lm.rotary_emb(inputs_embeds, rope_ids)
    return {
        "inputs_embeds": inputs_embeds,
        "position_embeddings": position_embeddings,
        "attention_mask": causal,
        "text_position_ids": text_position_ids,
        "deepstack_embeds": deepstack,
        "visual_positions": visual_positions,
        "deepstack_depth": 0 if deepstack is None else deepstack.shape[0],
    }

  def __call__(self, **kwargs):
    from types import SimpleNamespace
    import torchax.tensor as tx

    def cpu(v):
      if isinstance(v, tx.Tensor):
        return self._env.j2t_copy(v._elem)
      if isinstance(v, dict):
        return {k: cpu(x) for k, x in v.items()}
      if isinstance(v, (list, tuple)):
        return type(v)(cpu(x) for x in v)
      return v

    kwargs = cpu(kwargs)
    # jax carries no int64 unless x64 is on, so the round trip downgrades the
    # integer inputs and get_rope_index then cannot write int64 values into a
    # buffer it sized from input_ids.dtype.
    for name in ("input_ids", "attention_mask", "mm_token_type_ids",
                 "image_grid_thw", "video_grid_thw"):
      v = kwargs.get(name)
      if torch.is_tensor(v) and not torch.is_floating_point(v):
        kwargs[name] = v.to(torch.long)

    with torchax_paused():
      pre = self._host_prologue(kwargs)

    to = self._env.to_xla

    def move(v):
      if torch.is_tensor(v):
        return to(v)
      if isinstance(v, tuple):
        return tuple(move(x) for x in v)
      return v

    hidden = self._core(
        move(pre["inputs_embeds"]), move(pre["position_embeddings"]),
        move(pre["attention_mask"]), move(pre["text_position_ids"]),
        move(pre["deepstack_embeds"]), move(pre["visual_positions"]),
        deepstack_depth=pre["deepstack_depth"])
    return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class _VaeDecode(torch.nn.Module):
  """The VAE decode path as one traceable module.

  Wrapping the module and compiling it only jits `forward`, and the pipeline
  calls `decode`, so the decoder would keep running op by op.
  """

  def __init__(self, vae):
    super().__init__()
    self.post_quant_conv = vae.post_quant_conv
    self.decoder = vae.decoder
    self._conv_num = vae._cached_conv_counts["decoder"]

  def forward(self, z):
    # The pipeline denormalises with statistics built by `torch.tensor(...)`,
    # so the latents arrive as float32.
    z = z.to(self.post_quant_conv.weight.dtype)
    x = self.post_quant_conv(z)
    # feat_cache cannot be None. Wan's upsample3d branch skips `time_conv`
    # exactly when a cache list is present and the slot is still empty -- it
    # writes a "Rep" sentinel and moves on. Pass None and the temporal axis
    # doubles, after which QwenImage21CausalConv3d's `squeeze(2)` silently
    # does nothing and the convolution sees a 5-D input.
    out = self.decoder(x[:, :, 0:1, :, :],
                       feat_cache=[None] * self._conv_num,
                       feat_idx=[0], first_chunk=True)
    return torch.clamp(out, min=-1.0, max=1.0)


def _patch_vae_decode(pipe, compiled_decode):
  from diffusers.models.autoencoders.vae import DecoderOutput

  def decode(z, return_dict=True):
    sample = compiled_decode(z)
    return DecoderOutput(sample=sample) if return_dict else (sample,)

  pipe.vae.decode = decode


# ---------------------------------------------------------------------------


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-2.1")
  p.add_argument("--prompt", type=str, required=True)
  p.add_argument("--image", type=str, default=None,
                 help="Condition image; omit for text-to-image")
  p.add_argument("--height", type=int, default=1024)
  p.add_argument("--width", type=int, default=1024)
  p.add_argument("--num_inference_steps", type=int, default=20)
  p.add_argument("--true_cfg_scale", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--output", type=str, default="qwen_image_21_output.png")
  p.add_argument("--runs", type=int, default=1,
                 help="Repeat the generation; the first pays compilation")
  p.add_argument("--text_encoder_cpu_only", action="store_true",
                 help="Keep all of Qwen3-VL on CPU")
  p.add_argument("--vision_cpu_only", action="store_true",
                 help="Keep only the vision tower on CPU")
  p.add_argument("--vae_decode_eager", action="store_true",
                 help="Skip compiling the VAE decode (it then runs op by op)")
  p.add_argument("--compile_cache_dir", type=str,
                 default="~/.cache/qwen_image_21_jax",
                 help="Persistent XLA compilation cache; \"\" disables it")
  return p.parse_args()


def main():
  args = parse_args()
  print(args)
  torch.set_default_dtype(torch.bfloat16)

  # Without this a cold process pays the whole XLA compile -- 54 s for
  # text-to-image, 169 s for an edit -- before the first image. The cache is
  # keyed on the HLO, so it survives restarts and only misses when a shape or
  # a flag changes.
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    print(f"Compilation cache: {cache}")

  from diffusers import QwenImage21Pipeline

  pipe = QwenImage21Pipeline.from_pretrained(args.model_id,
                                             torch_dtype=torch.bfloat16)
  cls = pipe.__class__
  pipe.__class__ = type(f"{cls.__name__}TPU", (cls,),
                        {"_execution_device": property(
                            lambda _s: torch.device("jax"))})

  torchax.enable_globally()
  env = torchax.default_env()
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)),
              ("tensor",))
  env._mesh = mesh
  env.default_device_or_sharding = NamedSharding(mesh, P())
  print(f"Mesh: {jax.device_count()} chips on axis 'tensor'")

  _register_op_override(env, torch.nn.functional.conv2d,
                        functools.partial(_torch_conv2d, env=env))
  # No SDPA override: the sharded kernel splits the sequence across the mesh
  # and this model's joint sequence -- text plus 4x-expanded image slots -- is
  # not a multiple of eight, so shard_map rejects it. GSPMD shards on heads,
  # which is how TRANSFORMER_SHARDINGS already splits the projections.

  raw_tf = pipe.transformer
  _move_module(raw_tf, env)
  core = torchax.compile(
      _JitCore(raw_tf),
      torchax.CompileOptions(
          jax_jit_kwargs={"static_argnames": ("pad_tokens", "segments")}))
  core.params = _shard_weight_dict(core.params, TRANSFORMER_SHARDINGS, mesh)
  core.buffers = _shard_weight_dict(core.buffers, TRANSFORMER_SHARDINGS, mesh)
  pipe.transformer = _Prologue(raw_tf, core, env)

  _move_module(pipe.vae, env)
  if not args.vae_decode_eager:
    _patch_vae_decode(pipe, torchax.compile(_VaeDecode(pipe.vae)))

  if not args.text_encoder_cpu_only:
    te = pipe.text_encoder
    te.config._attn_implementation = "sdpa"
    te.config.text_config._attn_implementation = "sdpa"
    # Only the layers. `embed_tokens` and `rotary_emb` are used by the host
    # prologue under `torchax_paused()`, and a device tensor there raises
    # "torchax Tensors can only do math within the torchax environment".
    _move_module(te.model.language_model.layers, env)
    lm_core = torchax.compile(
        Qwen3VLDecoderCore(te.model.language_model),
        torchax.CompileOptions(
            jax_jit_kwargs={"static_argnames": ("deepstack_depth",)}))
    lm_core.params = _shard_weight_dict(lm_core.params, LM_SHARDINGS, mesh)
    lm_core.buffers = _shard_weight_dict(lm_core.buffers, LM_SHARDINGS, mesh)

    vision_core = None
    if not args.vision_cpu_only:
      patch_vision_attention()
      _move_module(te.model.visual, env)
      vision_core = torchax.compile(Qwen3VLVisionCore(te.model.visual))
    pipe.text_encoder = Qwen3VLSplit(te, lm_core, env, vision_core)
    print("Qwen3-VL: decoder sharded"
          + (", vision tower replicated" if vision_core else ", vision on CPU"))
  else:
    print("Qwen3-VL: all on CPU")

  image = None
  if args.image:
    from PIL import Image

    image = Image.open(args.image).convert("RGB")

  for i in range(args.runs):
    t0 = time.perf_counter()
    out = pipe(
        prompt=args.prompt, image=image, height=args.height,
        width=args.width, num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        # The pipeline's KV cache is a QwenImage21KVCache object, which jax
        # cannot accept as an argument. Registering it as a pytree node would
        # be the proper fix.
        use_kv_cache=False,
    ).images[0]
    dt = time.perf_counter() - t0
    tag = " (includes compilation)" if i == 0 else ""
    print(f"run {i}: {dt:.2f}s{tag}", flush=True)

  out.save(args.output)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
