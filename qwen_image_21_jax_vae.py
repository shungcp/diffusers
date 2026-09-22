"""Qwen-Image-2.1's VAE decoder in native JAX (flax nnx).

Companion to ``qwen_image_21_jax.py``, which has the DiT. Together they are
everything between a latent and a picture.

``AutoencoderKLQwenImage21`` is Wan's *video* VAE specialised to a single
frame, and that specialisation is the whole story of this file. Upstream
keeps the 3-D machinery and neutralises it at runtime; here it is compiled
away, which is both faster and far less code:

* ``QwenImage21CausalConv3d`` is already an ``nn.Conv2d`` that squeezes the
  frame axis away and puts it back. With one frame the squeeze is free, so
  every convolution here is an ordinary 2-D one and tensors stay NHWC, which
  is the layout a TPU wants anyway.
* ``upsample3d`` and ``upsample2d`` do the same thing on the first chunk.
  Upstream's temporal branch runs only when the feature cache for that slot
  already holds something; on the first (and here only) chunk it writes a
  ``"Rep"`` sentinel and skips straight to the spatial resample. The
  checkpoint's ``time_conv`` weights are therefore never read -- the loader
  below says so out loud rather than silently ignoring them.
* ``DupUp3D`` with ``first_chunk=True`` expands the frame axis by
  ``factor_t`` and then keeps only the last entry, so with one frame in it is
  one frame out and only the spatial duplication survives.

Getting this wrong is quiet rather than loud: passing ``feat_cache=None``
instead of a list of ``None`` makes upstream take the temporal branch, which
doubles the frame axis, after which the squeeze is a no-op and a 5-D tensor
reaches a 2-D weight. That is what it cost to find the first time.

Run this file to check the decoder against diffusers on CPU:

    JAX_PLATFORMS=cpu PYTHONPATH=src python qwen_image_21_jax_vae.py
"""

import glob
import os

import numpy as np

# decoder_base_dim / dim_mult / temperal_downsample / z_dim / out_channels,
# from the shipped vae/config.json. out_channels is 4, not 3: the checkpoint
# decodes RGBA and the pipeline keeps the alpha only when asked.
DEFAULTS = dict(dim=144, z_dim=64, dim_mult=(1, 2, 4, 8, 8), num_res_blocks=2,
                temperal_downsample=(False, True, True, True), out_channels=4)


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

  class Conv2d(nnx.Module):
    """NHWC convolution holding torch's `(out, in, kh, kw)` weight verbatim.

    Keeping the checkpoint's layout means the loader is a straight copy and
    the transpose happens once, at call time, where XLA folds it into the
    convolution.
    """

    def __init__(self, in_f, out_f, kernel=3, padding=1, *, dtype, rngs=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f, kernel, kernel), dtype))
      self.bias = nnx.Param(jnp.zeros((out_f,), dtype))
      self.padding = padding

    def __call__(self, x):
      pad = ((self.padding, self.padding), (self.padding, self.padding))
      out = jax.lax.conv_general_dilated(
          x, jnp.transpose(self.weight[...], (2, 3, 1, 0)),
          window_strides=(1, 1), padding=pad,
          dimension_numbers=("NHWC", "HWIO", "NHWC"))
      return out + self.bias[...]

  class RMSNormChannels(nnx.Module):
    """`QwenImage21RMS_norm`: L2-normalise over channels, then `sqrt(dim)*gamma`.

    An L2 normalise scaled by `sqrt(dim)` is an RMS normalise, but written
    the other way round and with no epsilon of its own -- `jax.lax.rsqrt` of
    a zero-mean-free sum is what upstream's `F.normalize` does, down to its
    1e-12 floor.
    """

    def __init__(self, dim, *, rngs=None):
      self.gamma = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.scale = dim**0.5

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      norm = jnp.sqrt(jnp.sum(x32**2, axis=-1, keepdims=True))
      normalized = x32 / jnp.maximum(norm, 1e-12)
      return (normalized * self.scale * self.gamma[...]).astype(x.dtype)

  class ResidualBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, *, dtype, rngs):
      self.norm1 = RMSNormChannels(in_dim)
      self.conv1 = Conv2d(in_dim, out_dim, 3, 1, dtype=dtype, rngs=rngs)
      self.norm2 = RMSNormChannels(out_dim)
      self.conv2 = Conv2d(out_dim, out_dim, 3, 1, dtype=dtype, rngs=rngs)
      self.conv_shortcut = (Conv2d(in_dim, out_dim, 1, 0, dtype=dtype, rngs=rngs)
                            if in_dim != out_dim else None)

    def __call__(self, x):
      # The shortcut reads the input before the first norm, not after.
      h = x if self.conv_shortcut is None else self.conv_shortcut(x)
      x = self.conv1(jax.nn.silu(self.norm1(x)))
      x = self.conv2(jax.nn.silu(self.norm2(x)))
      return x + h

  class AttentionBlock(nnx.Module):
    """Single-head attention over the spatial positions, head dim = channels."""

    def __init__(self, dim, *, dtype, rngs):
      self.norm = RMSNormChannels(dim)
      self.to_qkv = Conv2d(dim, dim * 3, 1, 0, dtype=dtype, rngs=rngs)
      self.proj = Conv2d(dim, dim, 1, 0, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      b, h, w, c = x.shape
      qkv = self.to_qkv(self.norm(x)).reshape(b, h * w, 3 * c)
      q, k, v = jnp.split(qkv, 3, axis=-1)
      out = jax.nn.dot_product_attention(q[:, :, None], k[:, :, None],
                                         v[:, :, None])
      return x + self.proj(out.reshape(b, h, w, c))

  class MidBlock(nnx.Module):
    def __init__(self, dim, num_layers=1, *, dtype, rngs):
      self.resnets = nnx.List([ResidualBlock(dim, dim, dtype=dtype, rngs=rngs)
                               for _ in range(num_layers + 1)])
      self.attentions = nnx.List([AttentionBlock(dim, dtype=dtype, rngs=rngs)
                                  for _ in range(num_layers)])

    def __call__(self, x):
      x = self.resnets[0](x)
      for attn, resnet in zip(self.attentions, self.resnets[1:]):
        x = resnet(attn(x))
      return x

  class DupUp(nnx.Module):
    """`QwenImage21DupUp3D` at one frame: a parameter-free 2x2 duplication.

    Upstream expands the channel axis by `factor_t * factor_s**2`, unravels
    it into `(out, factor_t, fs, fs)` and interleaves those into time,
    height and width. With one frame in and `first_chunk=True` the temporal
    copies are thrown away again, so only the spatial `fs x fs` survives --
    but the channel unravel order still has to match, or the duplication
    lands on the wrong pixels.
    """

    def __init__(self, in_channels, out_channels, factor_t, factor_s=2):
      self.out_channels = out_channels
      self.factor_t = factor_t
      self.factor_s = factor_s
      factor = factor_t * factor_s * factor_s
      if out_channels * factor % in_channels:
        raise ValueError(f"{out_channels} * {factor} % {in_channels}")
      self.repeats = out_channels * factor // in_channels

    def __call__(self, x):
      b, h, w, _ = x.shape
      ft, fs, out = self.factor_t, self.factor_s, self.out_channels
      x = jnp.repeat(x, self.repeats, axis=-1)
      x = x.reshape(b, h, w, out, ft, fs, fs)
      x = x[:, :, :, :, ft - 1]                      # first_chunk
      x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))       # b, h, fs, w, fs, out
      return x.reshape(b, h * fs, w * fs, out)

  class Upsampler(nnx.Module):
    """Nearest x2 then a 3x3 convolution.

    `upsample3d` and `upsample2d` are the same module here; the difference
    upstream is a `time_conv` that the first chunk never reaches.
    """

    def __init__(self, dim, out_dim, *, dtype, rngs):
      self.conv = Conv2d(dim, out_dim, 3, 1, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
      return self.conv(x)

  class ResidualUpBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, temperal_upsample,
                 up_flag, *, dtype, rngs):
      dims = [in_dim] + [out_dim] * num_res_blocks
      self.resnets = nnx.List([
          ResidualBlock(d, out_dim, dtype=dtype, rngs=rngs) for d in dims])
      self.upsampler = (Upsampler(out_dim, out_dim, dtype=dtype, rngs=rngs)
                        if up_flag else None)
      self.avg_shortcut = (DupUp(in_dim, out_dim, 2 if temperal_upsample else 1)
                           if up_flag else None)

    def __call__(self, x):
      residual = x
      for resnet in self.resnets:
        x = resnet(x)
      if self.upsampler is not None:
        x = self.upsampler(x)
      if self.avg_shortcut is not None:
        x = x + self.avg_shortcut(residual)
      return x

  class AvgDown(nnx.Module):
    """`QwenImage21AvgDown3D` at one frame: average pooling, done by reshape.

    The temporal factor still matters even with one frame. Upstream pads the
    frame axis up to `factor_t` *at the front* with zeros and then averages,
    so a `factor_t = 2` shortcut on a single frame averages the frame with a
    zero -- it halves. Dropping the pad because "there is only one frame"
    would silently double the shortcut.
    """

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
      factor = factor_t * factor_s * factor_s
      if in_channels * factor % out_channels:
        raise ValueError(f"{in_channels} * {factor} % {out_channels}")
      self.out_channels = out_channels
      self.factor_t = factor_t
      self.factor_s = factor_s
      self.group_size = in_channels * factor // out_channels

    def __call__(self, x):
      b, h, w, c = x.shape
      ft, fs = self.factor_t, self.factor_s
      x = x.reshape(b, h // fs, fs, w // fs, fs, c)
      x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))        # b, H', W', C, fs, fs
      if ft > 1:
        x = jnp.stack([jnp.zeros_like(x), x], axis=4)  # the front zero pad
      else:
        x = x[:, :, :, :, None]
      x = x.reshape(b, h // fs, w // fs, self.out_channels, self.group_size)
      return jnp.mean(x, axis=-1)

  class Downsampler(nnx.Module):
    """Pad one pixel on the right and bottom, then a strided 3x3.

    `downsample3d` and `downsample2d` are the same here, for the same reason
    the upsamplers are: the temporal branch needs a populated cache slot.
    """

    def __init__(self, dim, *, dtype, rngs):
      self.conv = Conv2d(dim, dim, 3, 0, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
      return jax.lax.conv_general_dilated(
          x, jnp.transpose(self.conv.weight[...], (2, 3, 1, 0)),
          window_strides=(2, 2), padding=((0, 0), (0, 0)),
          dimension_numbers=("NHWC", "HWIO", "NHWC")) + self.conv.bias[...]

  class ResidualDownBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, temperal_downsample,
                 down_flag, *, dtype, rngs):
      dims = [in_dim] + [out_dim] * (num_res_blocks - 1)
      self.resnets = nnx.List([
          ResidualBlock(d, out_dim, dtype=dtype, rngs=rngs) for d in dims])
      self.downsampler = (Downsampler(out_dim, dtype=dtype, rngs=rngs)
                          if down_flag else None)
      self.avg_shortcut = AvgDown(in_dim, out_dim,
                                  2 if temperal_downsample else 1,
                                  2 if down_flag else 1)

    def __call__(self, x):
      residual = x
      for resnet in self.resnets:
        x = resnet(x)
      if self.downsampler is not None:
        x = self.downsampler(x)
      return x + self.avg_shortcut(residual)

  class Encoder(nnx.Module):
    """`QwenImage21Encoder3d` at one frame, plus `quant_conv`.

    Returns the posterior's mean, which is what the pipeline reads
    (`sample_mode="argmax"`); the log-variance half of `conv_out` is
    computed and dropped.
    """

    def __init__(self, dim=96, z_dim=64, dim_mult=(1, 2, 4, 8, 8),
                 num_res_blocks=2, temperal_downsample=(False, True, True, True),
                 in_channels=4, *, dtype, rngs):
      dims = [dim * u for u in (1,) + tuple(dim_mult)]
      self.z_dim = z_dim
      self.conv_in = Conv2d(in_channels, dims[0], 3, 1, dtype=dtype, rngs=rngs)
      blocks = []
      for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        down_flag = i != len(dim_mult) - 1
        blocks.append(ResidualDownBlock(
            in_d, out_d, num_res_blocks,
            temperal_downsample[i] if down_flag else False, down_flag,
            dtype=dtype, rngs=rngs))
      self.down_blocks = nnx.List(blocks)
      self.mid_block = MidBlock(dims[-1], 1, dtype=dtype, rngs=rngs)
      self.norm_out = RMSNormChannels(dims[-1])
      self.conv_out = Conv2d(dims[-1], z_dim * 2, 3, 1, dtype=dtype, rngs=rngs)
      self.quant_conv = Conv2d(z_dim * 2, z_dim * 2, 1, 0, dtype=dtype, rngs=rngs)

    def __call__(self, pixels):
      """`pixels` is NHWC in [-1, 1]; returns NHWC latent means."""
      x = self.conv_in(pixels)
      for block in self.down_blocks:
        x = block(x)
      x = self.mid_block(x)
      x = self.conv_out(jax.nn.silu(self.norm_out(x)))
      return self.quant_conv(x)[..., :self.z_dim]

  class Decoder(nnx.Module):
    """`QwenImage21Decoder3d` at one frame, plus `post_quant_conv`."""

    def __init__(self, dim=144, z_dim=64, dim_mult=(1, 2, 4, 8, 8),
                 num_res_blocks=2, temperal_downsample=(False, True, True, True),
                 out_channels=4, *, dtype, rngs):
      temperal_upsample = tuple(reversed(temperal_downsample))
      dims = [dim * u for u in (dim_mult[-1],) + tuple(reversed(dim_mult))]
      self.post_quant_conv = Conv2d(z_dim, z_dim, 1, 0, dtype=dtype, rngs=rngs)
      self.conv_in = Conv2d(z_dim, dims[0], 3, 1, dtype=dtype, rngs=rngs)
      self.mid_block = MidBlock(dims[0], 1, dtype=dtype, rngs=rngs)
      blocks = []
      for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        up_flag = i != len(dim_mult) - 1
        blocks.append(ResidualUpBlock(
            in_dim, out_dim, num_res_blocks,
            temperal_upsample[i] if up_flag else False, up_flag,
            dtype=dtype, rngs=rngs))
      self.up_blocks = nnx.List(blocks)
      self.norm_out = RMSNormChannels(dims[-1])
      self.conv_out = Conv2d(dims[-1], out_channels, 3, 1, dtype=dtype, rngs=rngs)

    def __call__(self, z):
      """`z` is NHWC latents; returns NHWC pixels in [-1, 1]."""
      x = self.conv_in(self.post_quant_conv(z))
      x = self.mid_block(x)
      for block in self.up_blocks:
        x = block(x)
      x = self.conv_out(jax.nn.silu(self.norm_out(x)))
      return jnp.clip(x, -1.0, 1.0)

  return Decoder, Encoder


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------

# `resample` is a Sequential of [Upsample, Conv2d], so the convolution is
# index 1; everything else lines up name for name.
_RENAMES = (("decoder.", ""), ("upsampler.resample.1.", "upsampler.conv."))


def _resolve(model, path):
  obj = model
  for part in path:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def load_decoder(model, vae_dir, verbose=False):
  """Read the decoder half of the VAE checkpoint into the nnx module.

  Skips the encoder, and skips the `time_conv` weights of the `upsample3d`
  resamplers, which a single frame never reaches. Everything else must land,
  and a leftover key is an error rather than a warning -- a silently dropped
  tensor is exactly the failure this is meant to catch.
  """
  import jax.numpy as jnp
  from flax import nnx
  from safetensors import safe_open

  shards = sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {vae_dir}")

  loaded, skipped = 0, []
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if key.startswith("encoder.") or key.startswith("quant_conv."):
          continue
        if ".time_conv." in key:
          skipped.append(key)
          continue
        name = key
        for old, new in _RENAMES:
          name = name.replace(old, new)
        param = _resolve(model, name.split("."))
        value = reader.get_tensor(key).astype(param[...].dtype)
        # RMS_norm stores gamma with broadcasting axes; the module wants (dim,).
        value = value.reshape(param[...].shape)
        param[...] = value
        loaded += 1

  import jax
  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  if verbose:
    print(f"loaded {loaded}, skipped {len(skipped)} unused time_conv tensors")
  return model


ENCODER_DEFAULTS = dict(dim=96, z_dim=64, dim_mult=(1, 2, 4, 8, 8),
                        num_res_blocks=2,
                        temperal_downsample=(False, True, True, True),
                        in_channels=4)

_ENCODER_RENAMES = (("encoder.", ""), ("downsampler.resample.1.", "downsampler.conv."))


def load_encoder(model, vae_dir, verbose=False):
  """Read the encoder half. Skips the unused `time_conv` weights, as the
  decoder loader does, for the same single-frame reason."""
  import jax
  from flax import nnx
  from safetensors import safe_open

  shards = sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")))
  loaded, skipped = 0, 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if key.startswith(("decoder.", "post_quant_conv.")):
          continue
        if ".time_conv." in key:
          skipped += 1
          continue
        name = key
        for old, new in _ENCODER_RENAMES:
          name = name.replace(old, new)
        param = _resolve(model, name.split("."))
        value = reader.get_tensor(key).astype(param[...].dtype)
        param[...] = value.reshape(param[...].shape)
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  if verbose:
    print(f"encoder: loaded {loaded}, skipped {skipped} unused time_conv")
  return model


def encode_image(encoder, pixels, latents_mean, latents_std):
  """`pixels` NHWC in [-1, 1] -> normalised latents, channels second."""
  import jax.numpy as jnp

  latents = jnp.transpose(encoder(pixels), (0, 3, 1, 2))
  mean = jnp.asarray(latents_mean).reshape(1, -1, 1, 1)
  std = jnp.asarray(latents_std).reshape(1, -1, 1, 1)
  return (latents - mean) / std


def decode_latents(decoder, latents, latents_mean, latents_std):
  """Undo the pipeline's latent normalisation, then decode.

  `latents` is `(batch, z, height, width)` the way the DiT leaves them --
  channels second, as torch has it -- and comes back `(batch, height, width,
  channels)`. The un-normalise is `z * std + mean`, the inverse of what the
  pipeline does to a condition image before it encodes one.
  """
  import jax.numpy as jnp

  mean = jnp.asarray(latents_mean).reshape(1, -1, 1, 1)
  std = jnp.asarray(latents_std).reshape(1, -1, 1, 1)
  latents = latents * std + mean
  return decoder(jnp.transpose(latents, (0, 2, 3, 1)))


# --------------------------------------------------------------------------
# numerical equivalence against diffusers
# --------------------------------------------------------------------------


def main():
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from diffusers import AutoencoderKLQwenImage21

  vae_dir = os.path.join(find_snapshot(), "vae")

  Decoder, Encoder = _build_modules()
  decoder = Decoder(**DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  load_decoder(decoder, vae_dir, verbose=True)

  torch_vae = AutoencoderKLQwenImage21.from_pretrained(
      vae_dir, dtype=torch.float32).eval()

  rng = np.random.default_rng(0)
  z = rng.standard_normal((1, DEFAULTS["z_dim"], 8, 8), dtype=np.float32)
  with torch.no_grad():
    ref = torch_vae.decode(torch.tensor(z)[:, :, None]).sample.numpy()
  ref = np.transpose(ref[:, :, 0], (0, 2, 3, 1))  # NCHW -> NHWC, drop frame

  got = np.asarray(decoder(jnp.asarray(np.transpose(z, (0, 2, 3, 1)))))
  diff = np.abs(ref - got)
  rel = diff.max() / max(np.abs(ref).max(), 1e-9)
  print(f"decode {ref.shape}  max|diff| {diff.max():.3e}  relative {rel:.3e}  "
        f"mean|diff| {diff.mean():.3e}")
  ok = rel < 1e-4

  encoder = Encoder(**ENCODER_DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  load_encoder(encoder, vae_dir, verbose=True)
  image = rng.standard_normal((1, 4, 128, 128), dtype=np.float32).clip(-1, 1)
  with torch.no_grad():
    ref_e = torch_vae.encode(torch.tensor(image)[:, :, None]
                             ).latent_dist.mode().numpy()
  ref_e = np.transpose(ref_e[:, :, 0], (0, 2, 3, 1))
  got_e = np.asarray(encoder(jnp.asarray(np.transpose(image, (0, 2, 3, 1)))))
  diff_e = np.abs(ref_e - got_e)
  rel_e = diff_e.max() / max(np.abs(ref_e).max(), 1e-9)
  print(f"encode {ref_e.shape}  max|diff| {diff_e.max():.3e}  "
        f"relative {rel_e:.3e}  mean|diff| {diff_e.mean():.3e}")
  ok &= rel_e < 1e-4
  print("MATCH" if ok else "MISMATCH")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
