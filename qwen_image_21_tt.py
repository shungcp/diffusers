"""Qwen-Image-2.1 on TPU via torch_tpu.

Third backend for the same model, alongside ``qwen_image_21_tx.py``
(torchax). The two reach the device very differently and the difference
decides how much work each one is:

* **torchax** traces the PyTorch model into one XLA graph. That is what makes
  it fast, and it is also why three parts of this model had to be
  restructured -- the DiT's prologue, Qwen3-VL's, and the vision tower's
  attention all derive shapes from tensor *contents*, which a trace cannot
  do.
* **torch_tpu** runs the model eagerly with deferred execution. Nothing is
  traced, so none of those restructurings are needed and the stock diffusers
  model runs as written. The price is that each operator becomes its own
  program.

So this script is short where the torchax one is long, and that is the
interesting part of the comparison rather than an accident.

**One chip, nothing sharded.** The DiT is 14.2 GB in bfloat16 and the VAE
0.7, which fits a v6e chip's 31 GB with the text encoder left on the host;
adding Qwen3-VL's 17.5 GB would not. Tensor-parallel across eight chips means
rewriting the DiT against column- and row-parallel linears, which is a
separate job -- see the notes at the bottom.

Needs diffusers main (the QwenImage21 classes are in no release) and
transformers >= 5.17.

    TORCH_TPU_TOPOLOGY=1,1,1 \\
    TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:<free port> \\
    python qwen_image_21_tt.py --prompt "..."

Both environment variables are required even for a single chip: without them
PjRt refuses to start with "missing required environment variables for
distributed training". ``_auto_env`` fills them in by default; ``--no_auto_env`` opts out.
"""

import argparse
import os
import socket
import time


def _auto_env():
  """Fills in what torch_tpu's PjRt client insists on, for one chip."""
  os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = f"localhost:{port}"
  os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", str(time.time_ns()))
  # The default is 0, which halves host-to-device bandwidth; 16 GiB is where
  # it saturates on v6e.
  os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(16 << 30))
  # The stock LIBTPU_INIT_ARGS some launchers set includes
  # --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true, which
  # aborts at PjRt client creation on v6e ("platform ghostlite").
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      "--xla_tpu_dvfs_p_state=7 "
      "--xla_tpu_enable_latency_hiding_scheduler=true "
      "--xla_tpu_spmd_rng_bit_generator_unsafe=true "
      "--xla_tpu_enable_dot_strength_reduction=true",
  )


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
  p.add_argument("--output", type=str, default="qwen_image_21_tt.png")
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--no_auto_env", action="store_true",
                 help="Do not fill in the TORCH_TPU_* variables")
  return p.parse_args()


def main():
  args = parse_args()
  if not args.no_auto_env:
    _auto_env()
  print(args)

  import torch
  import torch_tpu  # noqa: F401  registers the 'tpu' device
  from diffusers import QwenImage21Pipeline

  dev = torch.device("tpu")
  pipe = QwenImage21Pipeline.from_pretrained(args.model_id,
                                             torch_dtype=torch.bfloat16)
  for name in ("transformer", "vae"):
    n = sum(p.numel() for p in getattr(pipe, name).parameters())
    print(f"{name:12s} {n / 1e9:5.2f}B")

  # DiT and VAE to the chip; Qwen3-VL stays on the host, which is what keeps
  # this inside 31 GB. Unlike the torchax script there is no tracing, so the
  # text encoder needs no splitting -- it just runs.
  pipe.transformer = pipe.transformer.to(dev)
  pipe.vae = pipe.vae.to(dev)
  # The pipeline takes its device from the transformer, so it would move the
  # text encoder's inputs to the chip while its weights are still on the host
  # ("embedding(): tensor is expected to be on tpu, got cpu"). Run the whole
  # encode on the host and move only the result.
  inner_encode = pipe.encode_prompt

  def encode_prompt(*a, **kw):
    kw["device"] = torch.device("cpu")
    embeds, mask, image_pad_mask = inner_encode(*a, **kw)
    to_dev = lambda t: t.to(dev) if t is not None else None
    return to_dev(embeds), to_dev(mask), to_dev(image_pad_mask)

  pipe.encode_prompt = encode_prompt
  print("DiT + VAE on tpu, Qwen3-VL on host")

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
    ).images[0]
    dt = time.perf_counter() - t0
    print(f"run {i}: {dt:.2f}s" + (" (includes compilation)" if i == 0 else ""),
          flush=True)

  out.save(args.output)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()


# Not done, and what it would take:
#
# * **Tensor parallel.** torch_tpu models are usually sharded by rebuilding
#   them against column- and row-parallel linear layers rather than by
#   sharding the diffusers module, so eight-way TP here means a rewrite of
#   QwenImage21Transformer2DModel: 32 single-stream blocks, each
#   attn.{to_q,to_k,to_v} column-parallel, attn.to_out.0 row-parallel,
#   img_mlp.{proj,gate_layer} column-parallel and img_mlp.out row-parallel,
#   every linear bias-free. Simpler than WAN's double-stream MoE, but a
#   rewrite all the same, plus a weight mapping from the diffusers
#   checkpoint.
# * **Qwen3-VL on device.** It would not fit alongside an unsharded DiT, so
#   this waits on the TP work above.
# * **Layerwise host offload** would be the other way to fit more on one
#   chip: park the blocks in pinned host memory already in the device's tiled
#   layout and DMA one in at a time (26.3 GB/s against .to()'s 7.2 for
#   bfloat16, because the layout conversion is paid once at load).
