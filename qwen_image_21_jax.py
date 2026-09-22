"""Qwen-Image-2.1's DiT in native JAX (flax nnx).

The third backend for this model, and the only one that is a reimplementation
rather than a way of running the PyTorch module. ``qwen_image_21_tx.py``
traces the diffusers model through torchax; ``qwen_image_21_tt.py`` runs it
eagerly through torch_tpu; both reuse the upstream definition. A native JAX
path has no such shortcut: the usual source of JAX diffusion models is
maxdiffusion (``maxdiffusion.models.wan``, ``.flux``, ``.ltx2``,
``.ltx_video``), and there is no ``maxdiffusion.models.qwen`` -- the model
shipped too recently. So the layers have to be written.

**What is here: the whole transformer (7.12B at the real config), a host
prologue, eight-way tensor parallel, and a test that all of it matches
PyTorch numerically.** The VAE and Qwen3-VL are not here; see the notes at
the bottom. Running this file checks a small random configuration against
diffusers on CPU -- one block, then the full model down both attention paths,
then the parameter count at the shipped config:

    JAX_PLATFORMS=cpu PYTHONPATH=src python qwen_image_21_jax.py

`--bench` times one denoising step on whatever devices JAX finds. On a v6e-8
at 1024x1024 (4096 image tokens, joint sequence 4608):

    eight chips, tensor parallel    0.091 s/step
    one chip, unsharded             0.465 s/step

so 5.1x across eight chips, and the whole 14.2 GB model does fit one chip if
throughput matters more than latency -- eight independent chips come to
58 ms per image-step against tensor parallel's 91. Of the 91 ms, all-reduce
is 31%, the matmuls 33% at 62% of memory bandwidth, and 20% is idle.

The split between ``build_metadata`` and the module is the point of the
design, not an implementation detail. Five things in this model are computed
from tensor *contents* -- where the image slots are, how far RoPE has
advanced, which tokens belong to which image block, which prompt positions
are padding -- and none of them can happen inside a traced region. They all
live in ``build_metadata``, which runs once per shape on the host in numpy;
the module takes the results as inputs and is jittable with everything
static. This is the same host-prologue split the torchax port needed, which
is why that work carried over.

Transcription notes, each a place the obvious reading is wrong:

* ``QwenImage21ZeroCenterRMSNorm`` is *not* a mean-subtracting norm. Its
  learnable weight is stored zero-centered, so the effective scale is
  ``weight + 1``. It appears in exactly one place, ``txt_in.text_norm``.
  ``norm_q``/``norm_k`` are plain ``RMSNorm``; using the zero-centered one
  there is off by a factor of ``(w + 1) / w`` and nothing else notices.
* Modulation is not per block. The parent computes one tensor and every block
  splits it into two halves, then each half into a scale and a gate. The gate
  goes through ``tanh`` before it multiplies the residual branch.
* With ``causal_condition`` the modulation tensor has ``batch + 1`` rows: the
  extra one is ``t = 0``, and text and condition-image tokens read it instead
  of the sampled timestep. ``target_token_mask`` chooses per token.
* RoPE is a genuine complex multiply (``use_real=False``), not the
  rotate-half form the other Qwen models use, and its tables are indexed by
  *signed* positions -- image blocks sit on a grid centred on zero, and the
  1024 negative rows are appended after the 8192 positive ones so that plain
  negative indexing lands on them.
* The attention rule is ``(q >= kv) or same_image_block``: causal over the
  joint sequence, bidirectional inside each image. Writing that as one dense
  boolean is correct and 13x slower -- see ``build_metadata``, which instead
  splits the sequence so the long image segments need no 2-D mask at all.
* Comparing against PyTorch **on TPU** needs
  ``jax.config.update("jax_default_matmul_precision", "highest")``. TPU runs
  float32 matmuls at bfloat16 precision by default, and over 32 layers that
  compounds to 29% relative error against a CPU reference -- which looks
  exactly like a broken port. With it set, the loaded 7.12B model matches
  diffusers to 5.8e-5.
* The feed-forward is SwiGLU with the gate on its own projection:
  ``out(silu(gate_layer(x)) * proj(x))``, all three linears bias-free. There
  are no biases anywhere in this model.
"""

import math

import numpy as np

_IMG_TOKENS_PER_SLOT = 4  # each vision-language image slot stands for 2x2 latents


# --------------------------------------------------------------------------
# flax nnx layers
# --------------------------------------------------------------------------


def _build_modules():
  """Imports flax lazily so the file can be read without it installed."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  def _select_rows(params, target_token_mask):
    """Broadcast per-sample modulation over the token axis.

    `params` is `(batch [+ 1], dim)`. Without a mask every token shares row
    `b`. With one, `params` has a trailing `t = 0` row that every token
    outside the target image reads instead.
    """
    if target_token_mask is None:
      return jnp.expand_dims(params, 1)
    real = jnp.expand_dims(params[:-1], 1)  # (batch, 1, dim)
    zero = jnp.expand_dims(params[-1:], 0)  # (1, 1, dim)
    return jnp.where(target_token_mask.reshape(1, -1, 1), real, zero)

  class ZeroCenterRMSNorm(nnx.Module):
    """Effective scale is `weight + 1`, computed in fp32."""

    def __init__(self, dim, eps=1e-6, *, rngs=None):
      self.weight = nnx.Param(jnp.zeros((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      dtype = x.dtype
      x32 = x.astype(jnp.float32)
      rrms = jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + self.eps)
      return (x32 * rrms * (self.weight[...].astype(jnp.float32) + 1)).astype(dtype)

  class RMSNorm(nnx.Module):
    """Plain RMSNorm: scale is `weight`, not `weight + 1`."""

    def __init__(self, dim, eps=1e-6, *, rngs=None):
      self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      rrms = jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + self.eps)
      return x32 * rrms * self.weight[...].astype(jnp.float32)

  class LayerNormNoAffine(nnx.Module):
    def __init__(self, eps=1e-6):
      self.eps = eps

    def __call__(self, x):
      dtype = x.dtype
      x32 = x.astype(jnp.float32)
      mean = jnp.mean(x32, axis=-1, keepdims=True)
      var = jnp.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
      return ((x32 - mean) * jax.lax.rsqrt(var + self.eps)).astype(dtype)

  class Linear(nnx.Module):
    """Bias-free; weight stored as torch does, (out, in)."""

    def __init__(self, in_f, out_f, *, dtype, rngs=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype))

    def __call__(self, x):
      return x @ self.weight[...].T

  def apply_rope(x, freqs_cis):
    """Complex rotary, the `use_real=False` branch. x is [B, S, H, D]."""
    x32 = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    xc = jax.lax.complex(x32[..., 0], x32[..., 1])
    out = xc * jnp.expand_dims(freqs_cis, 1)  # freqs [S, D/2] -> [S, 1, D/2]
    out = jnp.stack([jnp.real(out), jnp.imag(out)], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)

  class Attention(nnx.Module):
    def __init__(self, dim, heads, dim_head, eps, *, dtype, rngs):
      inner = heads * dim_head
      self.heads = heads
      self.scale = dim_head**-0.5
      self.to_q = Linear(dim, inner, dtype=dtype, rngs=rngs)
      self.to_k = Linear(dim, inner, dtype=dtype, rngs=rngs)
      self.to_v = Linear(dim, inner, dtype=dtype, rngs=rngs)
      self.norm_q = RMSNorm(dim_head, eps)
      self.norm_k = RMSNorm(dim_head, eps)
      self.to_out = Linear(inner, dim, dtype=dtype, rngs=rngs)

    def __call__(self, x, rotary_emb, attention_mask=None, segments=None):
      b, s, _ = x.shape
      q = self.to_q(x).reshape(b, s, self.heads, -1)
      k = self.to_k(x).reshape(b, s, self.heads, -1)
      v = self.to_v(x).reshape(b, s, self.heads, -1)
      q = self.norm_q(q).astype(v.dtype)
      k = self.norm_k(k).astype(v.dtype)
      if rotary_emb is not None:
        q = apply_rope(q, rotary_emb)
        k = apply_rope(k, rotary_emb)
      if segments is None:
        out = jax.nn.dot_product_attention(q, k, v, mask=attention_mask,
                                           scale=self.scale)
      else:
        # One call per segment, with a 2-D mask only where one is needed.
        # This is worth the loop: on v6e a masked call costs 35 ms at
        # sequence 4608 against 1.3 ms unmasked, because any non-trivial
        # mask drops XLA out of its fused attention and makes it
        # materialise the whole score matrix. Image blocks need no 2-D mask
        # at all (see `build_metadata`), so the expensive path is left with
        # only the short text runs.
        out = jnp.concatenate(
            [jax.nn.dot_product_attention(q[:, start:end], k[:, :end],
                                          v[:, :end], mask=mask,
                                          scale=self.scale)
             for (start, end), mask in segments], axis=1)
      return self.to_out(out.reshape(b, s, -1))

  class SwiGLU(nnx.Module):
    def __init__(self, hidden, mlp_hidden, *, dtype, rngs):
      self.proj = Linear(hidden, mlp_hidden, dtype=dtype, rngs=rngs)
      self.gate_layer = Linear(hidden, mlp_hidden, dtype=dtype, rngs=rngs)
      self.out = Linear(mlp_hidden, hidden, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      return self.out(jax.nn.silu(self.gate_layer(x)) * self.proj(x))

  class Block(nnx.Module):
    """One single-stream block. Modulation comes from the parent."""

    def __init__(self, dim, heads, dim_head, mlp_ratio=3, eps=1e-6, *,
                 dtype, rngs):
      self.img_norm1 = LayerNormNoAffine(eps)
      self.attn = Attention(dim, heads, dim_head, eps, dtype=dtype, rngs=rngs)
      self.img_norm2 = LayerNormNoAffine(eps)
      self.img_mlp = SwiGLU(dim, dim * mlp_ratio, dtype=dtype, rngs=rngs)

    @staticmethod
    def _modulate(x, mod, target_token_mask=None):
      scale, gate = jnp.split(mod, 2, axis=-1)
      scale = _select_rows(scale, target_token_mask)
      gate = _select_rows(gate, target_token_mask)
      return x * (1 + scale), gate

    def __call__(self, x, modulation, rotary_emb, attention_mask=None,
                 target_token_mask=None, segments=None):
      mod1, mod2 = jnp.split(modulation, 2, axis=-1)
      h, gate1 = self._modulate(self.img_norm1(x), mod1, target_token_mask)
      x = x + jnp.tanh(gate1) * self.attn(h, rotary_emb, attention_mask,
                                          segments)
      h2, gate2 = self._modulate(self.img_norm2(x), mod2, target_token_mask)
      return x + jnp.tanh(gate2) * self.img_mlp(h2)

  class TextProjection(nnx.Module):
    def __init__(self, context_in_dim, hidden, eps=1e-6, *, dtype, rngs):
      self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps)
      self.in_layer = Linear(context_in_dim, hidden, dtype=dtype, rngs=rngs)
      self.out_layer = Linear(hidden, hidden, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      x = self.in_layer(self.text_norm(x))
      return self.out_layer(jax.nn.gelu(x, approximate=True))

  class TimestepEmbeddings(nnx.Module):
    """Sinusoid with `cos` in the first half, then two bias-free linears."""

    def __init__(self, embedding_dim, timestep_dim=256, max_period=10000,
                 time_factor=1000.0, *, dtype, rngs):
      self.timestep_dim = timestep_dim
      self.max_period = max_period
      self.time_factor = time_factor
      self.linear_1 = Linear(timestep_dim, embedding_dim, dtype=dtype, rngs=rngs)
      self.linear_2 = Linear(embedding_dim, embedding_dim, dtype=dtype, rngs=rngs)

    def __call__(self, timestep, dtype):
      half = self.timestep_dim // 2
      freqs = jnp.exp(-math.log(self.max_period)
                      * jnp.arange(half, dtype=jnp.float32) / half)
      args = (self.time_factor * timestep.astype(jnp.float32))[:, None] * freqs[None]
      emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(dtype)
      return self.linear_2(jax.nn.silu(self.linear_1(emb)))

  class AdaLayerNormContinuous(nnx.Module):
    """Final adaptive norm. Scale only -- this variant emits no shift."""

    def __init__(self, embedding_dim, conditioning_dim, eps=1e-6, *, dtype, rngs):
      self.linear = Linear(conditioning_dim, embedding_dim, dtype=dtype, rngs=rngs)
      self.norm = LayerNormNoAffine(eps)

    def __call__(self, x, conditioning, target_token_mask=None):
      scale = self.linear(jax.nn.silu(conditioning).astype(x.dtype))
      return self.norm(x) * (1 + _select_rows(scale, target_token_mask))

  class Transformer(nnx.Module):
    """`QwenImage21Transformer2DModel`, without the KV cache.

    Defaults are the shipped config: 32 blocks of 32 heads x 128, 7.12B
    parameters. Everything content-dependent arrives in `meta`; see
    `build_metadata`.
    """

    def __init__(self, in_channels=64, out_channels=64, num_layers=32,
                 attention_head_dim=128, num_attention_heads=32,
                 context_in_dim=4096, mlp_ratio=3, eps=1e-6,
                 causal_condition=True, *, dtype, rngs):
      dim = num_attention_heads * attention_head_dim
      self.inner_dim = dim
      self.causal_condition = causal_condition
      self.time_text_embed = TimestepEmbeddings(dim, dtype=dtype, rngs=rngs)
      self.txt_in = TextProjection(context_in_dim, dim, eps, dtype=dtype, rngs=rngs)
      self.img_in = Linear(in_channels, dim, dtype=dtype, rngs=rngs)
      # One shared modulation for every block: mod1.scale, mod1.gate,
      # mod2.scale, mod2.gate. The blocks hold no modulation parameters.
      self.modulation = Linear(dim, 4 * dim, dtype=dtype, rngs=rngs)
      # nnx.List, not a plain list: a bare list is treated as a static
      # attribute and refuses to hold modules.
      self.transformer_blocks = nnx.List([
          Block(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                dtype=dtype, rngs=rngs)
          for _ in range(num_layers)
      ])
      self.norm_out = AdaLayerNormContinuous(dim, dim, eps, dtype=dtype, rngs=rngs)
      self.proj_out = Linear(dim, out_channels, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states, encoder_hidden_states, timestep, meta):
      batch = hidden_states.shape[0]
      hidden_states = self.img_in(hidden_states)
      encoder_hidden_states = self.txt_in(encoder_hidden_states)

      # Reserve room for the target image, expand every image slot four-fold,
      # then drop the real latents into the expanded positions. Upstream does
      # this with `repeat_interleave` and a boolean-mask assignment; both
      # become a gather and a scatter against indices from the host.
      pad = jnp.zeros((batch, meta["target_tokens"] // _IMG_TOKENS_PER_SLOT,
                       encoder_hidden_states.shape[2]), encoder_hidden_states.dtype)
      joint = jnp.concatenate([encoder_hidden_states, pad], axis=1)
      joint = jnp.take(joint, meta["expand_index"], axis=1)
      joint = joint.at[:, meta["image_positions"]].set(
          hidden_states.astype(joint.dtype))

      timestep = timestep.astype(joint.dtype)
      target_token_mask = None
      if self.causal_condition:
        timestep = jnp.concatenate([timestep, jnp.zeros((1,), timestep.dtype)])
        target_token_mask = meta["target_token_mask"]
      temb = self.time_text_embed(timestep, joint.dtype)
      modulation = self.modulation(jax.nn.silu(temb))

      for block in self.transformer_blocks:
        joint = block(joint, modulation, meta["rotary_emb"],
                      meta.get("attention_mask"), target_token_mask,
                      meta.get("segments"))

      joint = self.norm_out(joint, temb, target_token_mask)
      return self.proj_out(joint)

  return Block, Transformer


# --------------------------------------------------------------------------
# host prologue
# --------------------------------------------------------------------------


def _rope_table(dim, theta=10000):
  """Positive positions 0..8191 followed by negative -1024..-1.

  The negative rows go last so that indexing the table with a negative
  position lands on them under ordinary wrap-around, which is how upstream
  gets away with signed indices.
  """
  inv = 1.0 / np.power(np.float32(theta),
                       np.arange(0, dim, 2, dtype=np.float32) / dim)
  index = np.concatenate([np.arange(8192, dtype=np.float32),
                          np.arange(1024, dtype=np.float32)[::-1] * -1 - 1])
  return np.exp(1j * np.outer(index, inv)).astype(np.complex64)


def build_metadata(img_mask, img_shapes, encoder_hidden_states_mask=None,
                   axes_dims_rope=(16, 56, 56), theta=10000,
                   rope_tables=None, dense_mask=False):
  """Everything the traced core needs but cannot compute for itself.

  All of it depends on tensor *contents* -- which positions hold images, how
  far RoPE has advanced, which block a token belongs to -- so it runs here,
  on the host, in numpy, once per shape rather than once per denoising step.

  Args:
    img_mask: `(batch, vlm_seq)` bool, `True` at the vision-language
      encoder's image slots. Rows must agree; only row 0 is read.
    img_shapes: per-image `(frame, height, width)` in latent tokens,
      condition images first and the target image last.
    encoder_hidden_states_mask: `(batch, text_seq)` bool marking valid prompt
      tokens. Right padding must be excluded as *keys* on every path.
    rope_tables: output of `_rope_table` per axis, to build once and reuse.

  Returns:
    A dict of numpy arrays, ready to be `jnp.asarray`d and passed to
    `Transformer.__call__`.
  """
  row = np.asarray(img_mask, bool)[0]
  batch = np.asarray(img_mask).shape[0]
  repeats = np.where(row, _IMG_TOKENS_PER_SLOT, 1)
  image_pad_mask = np.repeat(row, repeats)
  expand_index = np.repeat(np.arange(row.shape[0]), repeats)
  image_positions = np.flatnonzero(image_pad_mask)
  seq_len = image_pad_mask.shape[0]

  # Block ids come from the token counts in `img_shapes`, not from runs of
  # `True`: two adjacent condition images form one run but must stay separate
  # blocks, or they would attend to each other bidirectionally.
  block_lengths = [math.prod(shape) for shape in img_shapes]
  if sum(block_lengths) != image_positions.size:
    raise ValueError(
        f"img_shapes accounts for {sum(block_lengths)} image tokens but "
        f"img_mask marks {image_positions.size}.")
  image_ids = np.full(seq_len, -1, np.int32)
  image_ids[image_positions] = np.repeat(np.arange(len(block_lengths)),
                                         block_lengths)
  target_token_mask = np.zeros(seq_len, bool)
  target_token_mask[image_positions[-block_lengths[-1]:]] = True

  # RoPE. Text advances a shared position on all three axes; each image block
  # freezes the frame axis where the preceding text left it and lays its
  # tokens on a height/width grid centred on zero, so a block's spatial
  # positions do not depend on where it sits in the sequence.
  frame_index, image_h, image_w = [], [], []
  cursor = position = 0
  is_image = image_pad_mask.tolist()
  for _, height, width in img_shapes:
    block_start = is_image.index(True, cursor)
    text_len = block_start - cursor
    frame_index.extend(range(position, position + text_len))
    position += text_len
    cursor = block_start + height * width
    frame_index.extend([position] * (height * width))
    position += max(height, width)
    image_h.extend([h for h in range(-(height - height // 2), height // 2)
                    for _ in range(width)])
    image_w.extend([w for _ in range(height)
                    for w in range(-(width - width // 2), width // 2)])
  if cursor < seq_len:
    frame_index.extend(range(position, position + seq_len - cursor))

  frame_index = np.asarray(frame_index, np.int64)
  height_index, width_index = frame_index.copy(), frame_index.copy()
  height_index[image_pad_mask] = image_h
  width_index[image_pad_mask] = image_w
  if rope_tables is None:
    rope_tables = [_rope_table(dim, theta) for dim in axes_dims_rope]
  rotary_emb = np.concatenate(
      [table[index] for table, index
       in zip(rope_tables, (frame_index, height_index, width_index))], axis=-1)

  # Right-padded prompt positions must never be attended to, on any path.
  # Text positions of the joint sequence line up, in order, with the
  # non-image positions of the vision-language sequence; the two are
  # interleaved, so this cannot be sliced off as a prefix. Padded positions
  # stay valid as *queries* so no row is ever fully masked.
  key_valid = None
  if encoder_hidden_states_mask is not None:
    ehs_mask = np.asarray(encoder_hidden_states_mask, bool)
    key_valid = np.ones((batch, seq_len), bool)
    key_valid[:, ~image_pad_mask] = ehs_mask[:, ~row[:ehs_mask.shape[1]]]

  # The attention rule is `(q >= kv) or same_image_block`. Written as one
  # dense boolean that is correct but slow: on v6e any non-trivial mask
  # costs 35 ms per call at sequence 4608 against 1.3 ms unmasked, because
  # it drops XLA out of its fused attention.
  #
  # Splitting the sequence into runs of equal `image_ids` removes almost all
  # of it. Take a run `[start, end)`:
  #
  # * an image block attends to `[0, q]` by causality *and* to all of
  #   `[start, end)` by being one block, and since `q >= start` the union is
  #   exactly `[0, end)` for every row -- no 2-D mask at all;
  # * a text run is plain causal, and needs one, but text runs are short.
  #
  # So the expensive path survives only on the short segments, and the long
  # image segments run unmasked. Prompt padding still applies everywhere, as
  # a `(batch, 1, 1, end)` broadcast that costs nothing.
  segments = []
  start = 0
  while start < seq_len:
    end = start + 1
    while end < seq_len and image_ids[end] == image_ids[start]:
      end += 1
    if image_ids[start] >= 0:
      mask = None if key_valid is None else key_valid[:, None, None, :end]
    else:
      rows = np.arange(start, end)[:, None] >= np.arange(end)[None, :]
      mask = np.broadcast_to(rows, (batch, end - start, end)).copy()
      if key_valid is not None:
        mask &= key_valid[:, None, :end]
      mask = mask[:, None]
    segments.append(((start, end), mask))
    start = end

  meta = {
      "expand_index": expand_index,
      "image_positions": image_positions,
      "image_ids": image_ids,
      "target_token_mask": target_token_mask,
      "target_tokens": block_lengths[-1],
      "rotary_emb": rotary_emb,
      "segments": tuple(segments),
  }
  if dense_mask:
    # The unsegmented form, kept for cross-checking the segmented one.
    q = np.arange(seq_len)[:, None]
    kv = np.arange(seq_len)[None, :]
    same_block = ((image_ids[:, None] == image_ids[None, :])
                  & (image_ids[:, None] >= 0))
    allowed = np.broadcast_to((q >= kv) | same_block,
                              (batch, seq_len, seq_len)).copy()
    if key_valid is not None:
      allowed &= key_valid[:, None, :]
    meta["attention_mask"] = allowed[:, None]
  return meta


def to_device(meta):
  """`build_metadata`'s numpy output as jax arrays.

  Pass the result to the module as a *closure* rather than as a traced
  argument: the segment boundaries are Python ints and have to stay static,
  since they index the sequence.
  """
  import jax.numpy as jnp

  def convert(value):
    if isinstance(value, int) or value is None:
      return value
    if isinstance(value, tuple):  # segments
      return tuple((bounds, convert(mask)) for bounds, mask in value)
    return jnp.asarray(value)

  return {key: convert(value) for key, value in meta.items()}


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------


def _put(param, tensor):
  import jax.numpy as jnp

  value = jnp.asarray(tensor.detach().float().numpy(), dtype=param[...].dtype)
  if value.shape != param[...].shape:
    raise ValueError(f"shape {value.shape} into parameter {param[...].shape}")
  param[...] = value


def copy_block_weights(jax_block, torch_block):
  """diffusers -> nnx, name by name, for one transformer block."""
  a, ta = jax_block.attn, torch_block.attn
  for name in ("to_q", "to_k", "to_v"):
    _put(getattr(a, name).weight, getattr(ta, name).weight)
  _put(a.to_out.weight, ta.to_out[0].weight)  # to_out is [Linear, Dropout]
  _put(a.norm_q.weight, ta.norm_q.weight)
  _put(a.norm_k.weight, ta.norm_k.weight)
  m, tm = jax_block.img_mlp, torch_block.img_mlp
  for name in ("proj", "gate_layer", "out"):
    _put(getattr(m, name).weight, getattr(tm, name).weight)


def copy_transformer_weights(jax_model, torch_model):
  """The whole DiT. Bias-free throughout, so weights are the only state."""
  _put(jax_model.img_in.weight, torch_model.img_in.weight)
  _put(jax_model.proj_out.weight, torch_model.proj_out.weight)
  _put(jax_model.modulation.weight, torch_model.modulation[1].weight)
  _put(jax_model.norm_out.linear.weight, torch_model.norm_out.linear.weight)
  _put(jax_model.txt_in.text_norm.weight, torch_model.txt_in.text_norm.weight)
  _put(jax_model.txt_in.in_layer.weight, torch_model.txt_in.in_layer.weight)
  _put(jax_model.txt_in.out_layer.weight, torch_model.txt_in.out_layer.weight)
  embedder = torch_model.time_text_embed.timestep_embedder
  _put(jax_model.time_text_embed.linear_1.weight, embedder.linear_1.weight)
  _put(jax_model.time_text_embed.linear_2.weight, embedder.linear_2.weight)
  for jax_block, torch_block in zip(jax_model.transformer_blocks,
                                    torch_model.transformer_blocks):
    copy_block_weights(jax_block, torch_block)


# --------------------------------------------------------------------------
# numerical equivalence against diffusers
# --------------------------------------------------------------------------


def _report(tag, ref, got, tol=1e-5):
  ref = np.asarray(ref)
  got = np.asarray(got)
  diff = np.abs(ref - got)
  rel = diff.max() / max(np.abs(ref).max(), 1e-9)
  verdict = "MATCH" if rel < tol else "MISMATCH"
  print(f"{tag:12s} {str(ref.shape):18s} max|diff| {diff.max():.3e}  "
        f"relative {rel:.3e}  {verdict}")
  return rel < tol


def _patch_dispatch():
  """diffusers routes attention through `dispatch_attention_fn`, whose backend
  selection misbehaves under a CUDA torch build on a host with no GPU
  ("permute(sparse_coo): number of dimensions ... does not match"). Swap in
  plain SDPA: the models below are still the real diffusers code.
  """
  import torch
  import diffusers.models.transformers.transformer_qwenimage21 as q21mod

  def _plain_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **_):
    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale)
    return out.permute(0, 2, 1, 3)

  q21mod.dispatch_attention_fn = _plain_attention


def test_block(Block):
  """One block in isolation, with a component-by-component breakdown."""
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from diffusers.models.transformers.transformer_qwenimage21 import (
      QwenImage21TransformerBlock,
  )

  torch.manual_seed(0)
  dim, heads, dim_head, seq = 512, 4, 128, 96  # small; the shapes generalise

  torch_block = QwenImage21TransformerBlock(
      dim=dim, num_attention_heads=heads, attention_head_dim=dim_head,
      mlp_ratio=3, eps=1e-6).eval().float()
  # RMSNorm weights start at one, which would hide a mistake in the
  # zero-centred `+1`; give them something to say.
  with torch.no_grad():
    torch_block.attn.norm_q.weight.normal_(1, 0.1)
    torch_block.attn.norm_k.weight.normal_(1, 0.1)

  jax_block = Block(dim, heads, dim_head, dtype=jnp.float32, rngs=nnx.Rngs(0))
  copy_block_weights(jax_block, torch_block)

  x = torch.randn(1, seq, dim)
  modulation = torch.randn(1, 4 * dim) * 0.1  # (batch, 4*dim), per sample
  ang = torch.randn(seq, dim_head // 2)
  freqs = torch.polar(torch.ones_like(ang), ang)  # complex [S, D/2]

  with torch.no_grad():
    ref = torch_block(hidden_states=x, modulation=modulation,
                      rotary_emb=freqs, attention_mask=None,
                      target_token_mask=None)
    xj, fj = jnp.asarray(x.numpy()), jnp.asarray(freqs.numpy())
    mj = jnp.asarray(modulation.numpy())
    # Component-wise as well, so a mismatch names a layer.
    th, _ = torch_block._modulate(torch_block.img_norm1(x),
                                  modulation.chunk(2, dim=-1)[0], None)
    jh, _ = jax_block._modulate(jax_block.img_norm1(xj),
                                jnp.split(mj, 2, axis=-1)[0])
    ok = _report("  modulate", th.numpy(), jh)
    ok &= _report("  attn", torch_block.attn(hidden_states=th, rotary_emb=freqs),
                  jax_block.attn(jh, fj))
    ok &= _report("  mlp", torch_block.img_mlp(th), jax_block.img_mlp(jh))

  got = jax_block(xj, mj, fj)
  return _report("block", ref.numpy(), got) and ok


def test_transformer(Transformer):
  """The whole model, including the joint sequence and the block-causal mask.

  Upstream computes block-causal attention with a multi-pass SDPA over
  per-image segments; this port uses one dense mask. Comparing the two end to
  end is the stronger test precisely because the implementations differ -- it
  checks the function, not the transcription.
  """
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from diffusers import QwenImage21Transformer2DModel

  torch.manual_seed(0)
  config = dict(in_channels=16, out_channels=16, num_layers=2,
                attention_head_dim=128, num_attention_heads=2,
                context_in_dim=64, mlp_ratio=3, eps=1e-6,
                causal_condition=True)
  # One condition image of 2x2 latents and a 4x4 target, in a 6-token prompt
  # whose last token is padding. img_mask marks slots, four latents each: one
  # slot for the condition image at position 3, four for the target.
  img_shapes = [(1, 2, 2), (1, 4, 4)]
  batch, text_len = 1, 6
  row = np.zeros(text_len + math.prod(img_shapes[-1]) // 4, bool)
  row[3] = True
  row[text_len:] = True
  img_mask = torch.tensor(np.tile(row, (batch, 1)))
  ehs_mask = torch.ones(batch, text_len, dtype=torch.bool)
  ehs_mask[:, -1] = False

  torch_model = QwenImage21Transformer2DModel(**config).eval().float()
  with torch.no_grad():
    for block in torch_model.transformer_blocks:
      block.attn.norm_q.weight.normal_(1, 0.1)
      block.attn.norm_k.weight.normal_(1, 0.1)
    torch_model.txt_in.text_norm.weight.normal_(0, 0.1)

  jax_model = Transformer(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))
  copy_transformer_weights(jax_model, torch_model)

  latents = torch.randn(batch, sum(math.prod(s) for s in img_shapes),
                        config["in_channels"])
  prompt = torch.randn(batch, text_len, config["context_in_dim"])
  timestep = torch.tensor([0.7])

  with torch.no_grad():
    ref = torch_model(hidden_states=latents, encoder_hidden_states=prompt,
                      timestep=timestep, img_shapes=[img_shapes],
                      img_mask=img_mask,
                      encoder_hidden_states_mask=ehs_mask,
                      return_dict=False)[0]

  args = (jnp.asarray(latents.numpy()), jnp.asarray(prompt.numpy()),
          jnp.asarray(timestep.numpy()))
  meta = build_metadata(img_mask.numpy(), img_shapes, ehs_mask.numpy(),
                        dense_mask=True)
  ok = _report("segmented", ref.numpy(), jax_model(*args, to_device(meta)))
  # Same metadata minus the segments, so the module takes the dense path.
  dense = {key: value for key, value in meta.items() if key != "segments"}
  return _report("dense mask", ref.numpy(),
                 jax_model(*args, to_device(dense))) and ok


def test_real_config(Transformer):
  """The shipped config, by shape only.

  `nnx.eval_shape` builds the parameter tree without allocating it, so this
  checks the 7.12B layout -- and that every layer is wired at the real
  dimensions -- for the cost of a dict traversal. Materialising it would be
  14 GB in bfloat16.
  """
  import jax
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from diffusers import QwenImage21Transformer2DModel

  abstract = nnx.eval_shape(
      lambda: Transformer(dtype=jnp.bfloat16, rngs=nnx.Rngs(0)))
  params = nnx.state(abstract, nnx.Param)
  total = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))

  # Upstream's own count, from a model that is likewise never materialised.
  with torch.device("meta"):
    reference = QwenImage21Transformer2DModel()
  expected = sum(p.numel() for p in reference.parameters())

  verdict = "MATCH" if total == expected else f"expected {expected / 1e9:.3f}B"
  print(f"real config  {total / 1e9:.3f}B parameters  {verdict}")
  return total == expected


_COLUMN_PARALLEL = frozenset({"to_q", "to_k", "to_v", "proj", "gate_layer"})
_ROW_PARALLEL = frozenset({"to_out", "out"})


def param_spec(keys, ndim):
  """The tensor-parallel `PartitionSpec` for one parameter, from its path.

  Heads-shard the three attention projections and the two SwiGLU inputs,
  row-shard what follows them so the all-reduce lands once per sub-layer.
  Everything else replicates: the largest of them, `modulation`, is 67M
  parameters against the blocks' 7.1B.

  Shared by `shard_model` and `load_transformer` so a loaded model and a
  built-then-sharded one lay out identically.
  """
  from jax.sharding import PartitionSpec as P

  # Paths end `.../to_q/weight/.value`, so the owning module is the last
  # component that is not one of the variable's own accessors.
  parent = next((k for k in reversed(keys)
                 if k not in ("weight", "value", ".value")), "")
  if ndim == 2 and parent in _COLUMN_PARALLEL:
    return P("tensor", None)
  if ndim == 2 and parent in _ROW_PARALLEL:
    return P(None, "tensor")
  return P()


def shard_model(model, mesh):
  """Move an already-built model onto `mesh` with `param_spec`'s layout."""
  import jax
  from flax import nnx
  from jax.sharding import NamedSharding

  def rule(path, leaf):
    keys = [str(getattr(k, "key", k)) for k in path]
    return jax.device_put(leaf,
                          NamedSharding(mesh, param_spec(keys, leaf.ndim)))

  state = nnx.state(model, nnx.Param)
  nnx.update(model, jax.tree.map_with_path(rule, state))
  return model


# diffusers names a few things differently from the module tree here: the
# modulation is a `Sequential` whose linear is index 1, the timestep embedder
# sits one level deeper, and `to_out` is a `ModuleList` of [Linear, Dropout].
_CHECKPOINT_RENAMES = (
    ("modulation.1.", "modulation."),
    ("time_text_embed.timestep_embedder.", "time_text_embed."),
    (".to_out.0.", ".to_out."),
)


def _resolve(model, path):
  obj = model
  for part in path:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def load_transformer(model, model_dir, mesh=None):
  """Stream the diffusers safetensors shards straight into the nnx model.

  One tensor at a time, placed on its final sharding as it is read, so the
  7.12B never exists twice -- neither as a PyTorch model nor as an
  unsharded device copy.

  Args:
    model: a `Transformer` built at the checkpoint's config.
    model_dir: the `transformer/` folder of a diffusers snapshot.
    mesh: shard as it loads; `None` leaves everything on the default device.

  Returns:
    The same model, updated in place.
  """
  import glob
  import os

  import jax
  from safetensors import safe_open
  from jax.sharding import NamedSharding

  shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {model_dir}")

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        name = key
        for old, new in _CHECKPOINT_RENAMES:
          name = name.replace(old, new)
        keys = name.split(".")
        param = _resolve(model, keys)
        value = reader.get_tensor(key).astype(param[...].dtype)
        if value.shape != param[...].shape:
          raise ValueError(
              f"{key}: checkpoint {value.shape} vs model {param[...].shape}")
        if mesh is not None:
          value = jax.device_put(
              value, NamedSharding(mesh, param_spec(keys, value.ndim)))
        param[...] = value
        loaded += 1

  from flax import nnx
  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return model


def bench(height=1024, width=1024, text_len=512, steps=3):
  """Time one denoising step of the real 7.12B DiT on whatever JAX sees.

  Weights are zeros -- TPU matmul timing does not depend on values -- so this
  measures the graph, not the model. 14.2 GB in bfloat16 fits one v6e chip,
  which makes the number directly comparable to the torch_tpu route's
  single-chip run rather than to the eight-chip torchax one.
  """
  import time

  import jax
  import jax.numpy as jnp
  from flax import nnx

  _, Transformer = _build_modules()
  latent_h, latent_w = height // 16, width // 16  # /8 VAE, /2 packing
  img_shapes = [(1, latent_h, latent_w)]
  target = math.prod(img_shapes[-1])
  row = np.zeros(text_len + target // _IMG_TOKENS_PER_SLOT, bool)
  row[text_len:] = True
  meta = to_device(build_metadata(row[None], img_shapes))
  seq = meta["rotary_emb"].shape[0]
  print(f"{height}x{width}: {target} image tokens, joint sequence {seq}")

  t0 = time.perf_counter()
  model = Transformer(dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  chips = jax.device_count()
  if chips > 1:
    # Auto axes, not Explicit: with explicit sharding a row-parallel matmul
    # has both contracting dims sharded and jax refuses to guess the output
    # sharding. Auto hands that to GSPMD, which inserts the all-reduce.
    mesh = jax.make_mesh((chips,), ("tensor",),
                         axis_types=(jax.sharding.AxisType.Auto,))
    model = shard_model(model, mesh)
  jax.block_until_ready(nnx.state(model, nnx.Param))
  print(f"weights on {chips} chip(s) in {time.perf_counter() - t0:.1f}s")

  latents = jnp.zeros((1, target, 64), jnp.bfloat16)
  prompt = jnp.zeros((1, text_len, 4096), jnp.bfloat16)

  @nnx.jit
  def step(model, latents, prompt, timestep):
    return model(latents, prompt, timestep, meta)

  for i in range(steps):
    t0 = time.perf_counter()
    out = jax.block_until_ready(
        step(model, latents, prompt, jnp.asarray([0.7], jnp.float32)))
    label = " (includes compilation)" if i == 0 else ""
    print(f"step {i}: {time.perf_counter() - t0:.3f}s{label}", flush=True)
  print(f"output {out.shape}")
  return True


def main():
  import sys

  if "--bench" in sys.argv:
    return 0 if bench() else 1
  _patch_dispatch()
  Block, Transformer = _build_modules()
  ok = test_block(Block)
  ok &= test_transformer(Transformer)
  ok &= test_real_config(Transformer)
  print("all match" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())


# Still to write for a full native-JAX pipeline, in the order the port would
# go:
#
# 1. Weight loading straight from the diffusers safetensors rather than
#    through a materialised PyTorch model, which is what
#    `copy_transformer_weights` does above. The name mapping is already
#    there; what is missing is streaming the shards so 7.12B never lands in
#    host memory twice.
# 2. `AutoencoderKLQwenImage21` -- Wan's video VAE specialised to one frame.
#    maxdiffusion already has a JAX Wan VAE
#    (`maxdiffusion.models.wan.autoencoder_kl_wan`), so this is the one piece
#    with a real head start.
# 3. Qwen3-VL. The largest remaining item at 8.77B, and the one with no
#    precedent in maxdiffusion. Running it on the host, as both other
#    backends do, is a reasonable place to stop.
# 4. The KV cache. `causal_condition` makes text and condition-image
#    activations independent of the timestep, so their keys and values are
#    computed once and reused for every later step. Skipped here, as it is in
#    the torchax port.
#
# Performance leads not taken, in the order they look worth it:
#
# * All-reduce is 31% of the step. The two per block come from row-sharding
#   `to_out` and the SwiGLU `out`; holding activations sequence-sharded
#   between sub-layers turns each into a reduce-scatter plus a later
#   all-gather, which moves the same bytes but overlaps better. The torchax
#   port hit the same wall from the other side -- 38% of device time in
#   collectives, fully exposed -- and no XLA flag helped there either.
# * Data parallel beats tensor parallel on throughput by 1.6x here, because
#   the model fits one chip. A serving runner should probably offer both.
# * The matmuls run at 62% of memory bandwidth and only 11% of peak FLOPs, so
#   at batch 1 this is a memory-bound model and the usual lever is a larger
#   batch rather than a better kernel.
