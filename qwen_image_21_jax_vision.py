"""Qwen3-VL's vision tower in JAX, for image-conditioned generation.

The last module of the native path. ``qwen_image_21_jax_text.py`` reads a
prompt; this reads the condition image that goes with it, and feeds the
language stack in two places: the merged tokens that sit at the
``<|image_pad|>`` positions, and three sets of *deepstack* features that are
added into language layers 0, 1 and 2.

Twenty-seven blocks of 1152 with 16 heads, pre-norm LayerNorm (with biases,
unlike everything else in this model), a GELU MLP, and axial 2-D RoPE.
Four things are worth knowing before reading the code:

* **The patch embedding is a matmul.** Upstream writes a ``Conv3d`` whose
  kernel and stride are both ``(2, 16, 16)`` over an input already reshaped
  to one patch per row, so every patch is independent and the convolution is
  a linear map on the flattened patch. Reshaping the weight is exact, not an
  approximation.
* **The learned position grid is resampled per image.** The table is 48x48
  and the image's patch grid is whatever it is, so the two are bridged by a
  bilinear resample expressed as four gather indices and four weights per
  patch. This is the part that goes wrong on large images if it goes wrong
  at all, so ``vision_metadata`` is checked against transformers' own helper
  across a range of sizes, small and large, by running this file.
* **Two different GELUs.** The blocks use the tanh approximation
  (``gelu_pytorch_tanh``); the patch mergers use ``nn.GELU()``, which is the
  exact error-function one. They differ by about 1e-3, which is enough to
  matter and easy to miss.
* **Patches arrive in spatial-merge-block order**, not raster order: the
  2x2 group that becomes one language token is contiguous. Both the position
  ids and the interpolation indices have to be laid out the same way, which
  is why they are computed together.

    JAX_PLATFORMS=cpu PYTHONPATH=src python qwen_image_21_jax_vision.py
"""

import glob
import math
import os

import numpy as np

DEFAULTS = dict(depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
                patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                in_channels=3, out_hidden_size=4096,
                num_position_embeddings=2304, deepstack_indexes=(8, 16, 24),
                rope_theta=10000.0)


# --------------------------------------------------------------------------
# host prologue
# --------------------------------------------------------------------------


def _axis_taps_weights(index, size, side):
  """Bilinear taps with `align_corners=True`, as `F.interpolate` computes them.

  `index` is the patch's row (or column); `size` the grid's height (or
  width); `side` the learned table's side, 48 here. Returns two taps and
  their weights, the taps clamped to the table's edge.
  """
  src = index.astype(np.float64) * (side - 1) / np.maximum(size - 1, 1)
  floor = np.floor(src)
  raw = floor[:, None].astype(np.int64) + np.arange(2)
  taps = np.clip(raw, 0, side - 1)
  distance = np.abs(src[:, None] - floor[:, None] - np.arange(2))
  return taps, np.clip(1.0 - distance, 0.0, None)


def vision_metadata(grid_thw, num_position_embeddings=2304,
                    spatial_merge_size=2):
  """Everything the tower needs that depends on the image's shape.

  Args:
    grid_thw: `(num_images, 3)` of `(frames, patch_rows, patch_cols)`.

  Returns a dict of numpy arrays:
    `interp_indices` `(tokens, 4)` and `interp_weights` `(tokens, 4)` --
      the bilinear resample of the learned position table, as a gather;
    `position_ids` `(tokens, 2)` -- the row and column of every patch, in
      spatial-merge-block order;
    `cu_seqlens` -- image boundaries, for the block-diagonal attention mask.
  """
  grid_thw = np.asarray(grid_thw, np.int64).reshape(-1, 3)
  side = int(round(math.sqrt(num_position_embeddings)))
  merge = spatial_merge_size

  rows, cols, counts = [], [], []
  for frames, height, width in grid_thw:
    grid_r, grid_c = np.meshgrid(np.arange(height), np.arange(width),
                                 indexing="ij")
    block = (height // merge, merge, width // merge, merge)
    # Transposing the two middle axes is what puts each 2x2 merge group
    # together; the language model consumes one token per group.
    grid_r = grid_r.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    grid_c = grid_c.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    rows.append(np.tile(grid_r, frames))
    cols.append(np.tile(grid_c, frames))
    counts.append(frames * height * width)

  row = np.concatenate(rows)
  col = np.concatenate(cols)
  heights = np.repeat(grid_thw[:, 1], counts)
  widths = np.repeat(grid_thw[:, 2], counts)

  h_taps, h_weights = _axis_taps_weights(row, heights, side)
  w_taps, w_weights = _axis_taps_weights(col, widths, side)
  # Outer product of the two axes' taps: (tokens, 2, 2) flattened to 4.
  indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
  weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)

  return {
      "interp_indices": indices.astype(np.int32),
      "interp_weights": weights.astype(np.float32),
      "position_ids": np.stack([row, col], axis=-1).astype(np.int32),
      "cu_seqlens": np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
  }


def vision_rope(position_ids, hidden_size=1152, num_heads=16,
                rope_theta=10000.0):
  """Axial 2-D RoPE: the same frequencies for rows and columns, concatenated.

  `head_dim // 4` frequencies are computed and laid out as
  `[h, w, h, w]` across the head, so `rotate_half` pairs each angle with
  itself a half-head away.
  """
  head_dim = hidden_size // num_heads
  spatial = head_dim // 2
  inv_freq = 1.0 / (rope_theta ** (np.arange(0, spatial, 2) / spatial))
  freqs = np.asarray(position_ids, np.float64)[..., None] * inv_freq
  merged = np.concatenate([freqs[:, 0], freqs[:, 1]], axis=-1)
  merged = np.concatenate([merged, merged], axis=-1)
  return np.cos(merged).astype(np.float32), np.sin(merged).astype(np.float32)


# --------------------------------------------------------------------------
# flax nnx layers
# --------------------------------------------------------------------------


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
  import jax
  import jax.numpy as jnp
  from flax import nnx

  class Linear(nnx.Module):
    """With bias -- the vision tower has them where the rest of the model does not."""

    def __init__(self, in_f, out_f, *, dtype, rngs=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype))
      self.bias = nnx.Param(jnp.zeros((out_f,), dtype))

    def __call__(self, x):
      return x @ self.weight[...].T + self.bias[...]

  class LayerNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, *, rngs=None):
      self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.bias = nnx.Param(jnp.zeros((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      mean = jnp.mean(x32, axis=-1, keepdims=True)
      var = jnp.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
      normed = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
      return (normed * self.weight[...] + self.bias[...]).astype(x.dtype)

  def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

  class Attention(nnx.Module):
    def __init__(self, dim, heads, *, dtype, rngs):
      self.heads = heads
      self.head_dim = dim // heads
      self.scale = self.head_dim**-0.5
      self.qkv = Linear(dim, dim * 3, dtype=dtype, rngs=rngs)
      self.proj = Linear(dim, dim, dtype=dtype, rngs=rngs)

    def __call__(self, x, cos, sin, mask=None):
      seq, dim = x.shape
      qkv = self.qkv(x).reshape(seq, 3, self.heads, self.head_dim)
      q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
      cos32, sin32 = cos[:, None, :], sin[:, None, :]
      q = (q.astype(jnp.float32) * cos32
           + rotate_half(q.astype(jnp.float32)) * sin32).astype(v.dtype)
      k = (k.astype(jnp.float32) * cos32
           + rotate_half(k.astype(jnp.float32)) * sin32).astype(v.dtype)
      out = jax.nn.dot_product_attention(q[None], k[None], v[None], mask=mask,
                                         scale=self.scale)
      return self.proj(out.reshape(seq, dim))

  class MLP(nnx.Module):
    def __init__(self, dim, intermediate, *, dtype, rngs):
      self.linear_fc1 = Linear(dim, intermediate, dtype=dtype, rngs=rngs)
      self.linear_fc2 = Linear(intermediate, dim, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      # gelu_pytorch_tanh, unlike the mergers' exact GELU.
      return self.linear_fc2(jax.nn.gelu(self.linear_fc1(x), approximate=True))

  class Block(nnx.Module):
    def __init__(self, dim, heads, intermediate, *, dtype, rngs):
      self.norm1 = LayerNorm(dim)
      self.attn = Attention(dim, heads, dtype=dtype, rngs=rngs)
      self.norm2 = LayerNorm(dim)
      self.mlp = MLP(dim, intermediate, dtype=dtype, rngs=rngs)

    def __call__(self, x, cos, sin, mask=None):
      x = x + self.attn(self.norm1(x), cos, sin, mask)
      return x + self.mlp(self.norm2(x))

  class PatchMerger(nnx.Module):
    """Folds a 2x2 patch group into one language token.

    `use_postshuffle_norm` decides whether the LayerNorm sees the group
    already folded (deepstack) or each patch on its own (the main merger).
    """

    def __init__(self, dim, merge_size, out_dim, postshuffle, *, dtype, rngs):
      folded = dim * merge_size**2
      self.folded = folded
      self.postshuffle = postshuffle
      self.norm = LayerNorm(folded if postshuffle else dim)
      self.linear_fc1 = Linear(folded, folded, dtype=dtype, rngs=rngs)
      self.linear_fc2 = Linear(folded, out_dim, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      x = self.norm(x.reshape(-1, self.folded) if self.postshuffle else x)
      x = x.reshape(-1, self.folded)
      # Exact GELU here, not the tanh approximation the blocks use.
      return self.linear_fc2(jax.nn.gelu(self.linear_fc1(x), approximate=False))

  class VisionModel(nnx.Module):
    def __init__(self, depth=27, hidden_size=1152, num_heads=16,
                 intermediate_size=4304, patch_size=16, temporal_patch_size=2,
                 spatial_merge_size=2, in_channels=3, out_hidden_size=4096,
                 num_position_embeddings=2304, deepstack_indexes=(8, 16, 24),
                 rope_theta=10000.0, *, dtype, rngs):
      patch = in_channels * temporal_patch_size * patch_size * patch_size
      # The Conv3d, as the linear map it is: kernel == stride, one patch per row.
      self.patch_embed = Linear(patch, hidden_size, dtype=dtype, rngs=rngs)
      self.pos_embed = nnx.Param(
          jnp.zeros((num_position_embeddings, hidden_size), dtype))
      self.blocks = nnx.List([
          Block(hidden_size, num_heads, intermediate_size, dtype=dtype,
                rngs=rngs) for _ in range(depth)])
      self.merger = PatchMerger(hidden_size, spatial_merge_size,
                                out_hidden_size, False, dtype=dtype, rngs=rngs)
      self.deepstack_merger_list = nnx.List([
          PatchMerger(hidden_size, spatial_merge_size, out_hidden_size, True,
                      dtype=dtype, rngs=rngs)
          for _ in deepstack_indexes])
      self.deepstack_indexes = tuple(deepstack_indexes)

    def __call__(self, pixel_values, meta, cos, sin, mask=None):
      """`pixel_values` is `(tokens, patch)`; returns merged tokens and deepstack."""
      x = self.patch_embed(pixel_values)
      taps = self.pos_embed[...][meta["interp_indices"]]
      x = x + jnp.sum(taps * meta["interp_weights"][:, :, None],
                      axis=1).astype(x.dtype)

      deepstack = []
      for index, block in enumerate(self.blocks):
        x = block(x, cos, sin, mask)
        if index in self.deepstack_indexes:
          deepstack.append(
              self.deepstack_merger_list[self.deepstack_indexes.index(index)](x))
      return self.merger(x), deepstack

  return VisionModel


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------

_PREFIX = "model.visual."


def _resolve(model, path):
  obj = model
  for part in path:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def load_vision_model(model, text_encoder_dir):
  """Read the vision half of the checkpoint into the nnx module."""
  import jax
  from flax import nnx
  from safetensors import safe_open

  shards = sorted(glob.glob(os.path.join(text_encoder_dir, "*.safetensors")))
  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if not key.startswith(_PREFIX):
          continue
        name = key[len(_PREFIX):]
        if name.startswith("pos_embed."):
          param = model.pos_embed
        elif name.startswith("patch_embed.proj."):
          param = getattr(model.patch_embed,
                          "bias" if name.endswith("bias") else "weight")
        else:
          param = _resolve(model, name.split("."))
        value = reader.get_tensor(key).astype(param[...].dtype)
        # The Conv3d weight is (embed, channels, t, p, p); flatten the patch.
        param[...] = value.reshape(param[...].shape)
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return model


# --------------------------------------------------------------------------
# numerical equivalence against transformers
# --------------------------------------------------------------------------


def _check_metadata(grid_thw):
  """The host prologue against transformers' own helpers, at one size."""
  import torch
  from transformers.vision_utils import (
      get_vision_interpolation_indices_and_weights, get_vision_position_ids)

  grid = torch.tensor(grid_thw, dtype=torch.long).reshape(-1, 3)
  ref_i, ref_w = get_vision_interpolation_indices_and_weights(
      grid, num_grid_per_side=48, mode="bilinear", align_corners=True,
      spatial_merge_size=2)
  ref_p = get_vision_position_ids(grid, 2)
  mine = vision_metadata(grid_thw)
  same_i = np.array_equal(ref_i.numpy(), mine["interp_indices"])
  weight_diff = np.abs(ref_w.numpy() - mine["interp_weights"]).max()
  same_p = np.array_equal(ref_p.numpy(), mine["position_ids"])
  tokens = mine["position_ids"].shape[0]
  print(f"  grid {str(grid_thw):22s} {tokens:6d} tokens  indices "
        f"{'ok' if same_i else 'DIFF'}  weights {weight_diff:.2e}  "
        f"positions {'ok' if same_p else 'DIFF'}")
  # The weights are a float32 round of the same arithmetic, so 1e-5 not 0.
  return same_i and same_p and weight_diff < 1e-5


def main():
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from transformers import Qwen3VLForConditionalGeneration

  text_dir = os.path.join(find_snapshot(), "text_encoder")

  # The resample is what breaks on large images, so check a spread of sizes
  # before anything else.
  print("host prologue:")
  ok = True
  for grid in ([[1, 4, 4]], [[1, 8, 12]], [[1, 32, 32]], [[1, 64, 64]],
               [[1, 96, 128]], [[1, 4, 4], [1, 8, 8]]):
    ok &= _check_metadata(grid)

  VisionModel = _build_modules()
  model = VisionModel(**DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  load_vision_model(model, text_dir)

  torch_model = Qwen3VLForConditionalGeneration.from_pretrained(
      text_dir, dtype=torch.float32).eval()
  visual = torch_model.model.visual

  rng = np.random.default_rng(0)
  grid = [[1, 8, 12]]
  tokens = 8 * 12
  patch = DEFAULTS["in_channels"] * DEFAULTS["temporal_patch_size"] * 16 * 16
  pixels = rng.standard_normal((tokens, patch), dtype=np.float32)

  with torch.no_grad():
    out = visual(torch.tensor(pixels), torch.tensor(grid, dtype=torch.long))
  ref_merged = out.pooler_output.numpy()
  ref_deep = [f.numpy() for f in out.deepstack_features]

  meta = vision_metadata(grid)
  cos, sin = vision_rope(meta["position_ids"])
  jmeta = {"interp_indices": jnp.asarray(meta["interp_indices"]),
           "interp_weights": jnp.asarray(meta["interp_weights"])}
  merged, deep = model(jnp.asarray(pixels), jmeta, jnp.asarray(cos),
                       jnp.asarray(sin))

  def report(tag, ref, got):
    got = np.asarray(got)
    diff = np.abs(ref - got)
    rel = diff.max() / max(np.abs(ref).max(), 1e-9)
    print(f"  {tag:14s} {str(ref.shape):16s} max|diff| {diff.max():.3e}  "
          f"relative {rel:.3e}")
    return rel < 1e-4

  print("vision tower:")
  ok &= report("merged", ref_merged, merged)
  for i, (r, g) in enumerate(zip(ref_deep, deep)):
    ok &= report(f"deepstack {i}", r, g)
  print("all match" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
