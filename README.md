Original readme moved to README_original.md

# install
```
pip install -e .
pip install transformers accelerate

# install torchax
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes#subdirectory=torchax
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu]
```

To run:

sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

```
python wan_tx.py
```

# Progress:

(Jun 17)
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Device ┃ Memory usage         ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0      │ 2.09 GiB / 31.25 GiB │      0.00% │
│ 1      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 2      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 3      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 4      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 5      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 6      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 7      │ 2.08 GiB / 31.25 GiB │      0.00% │


## sizes:
### wan 1.3B:
text_encoder 12.537574768066406 G
transformer 2.64891254901886 G
vae 0.23635575734078884 G

### wan 14B
text_encoder 12.537574768066406 G
transformer 26.66874897480011 G
vae 0.23635575734078884 G

## Shapes of weights for 1.3B model:
### vae
```
encoder.conv_in.weight : # (torch.Size([96, 3, 3, 3, 3]), torch.bfloat16)
encoder.conv_in.bias : # (torch.Size([96]), torch.bfloat16)
encoder.down_blocks.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
encoder.down_blocks.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.resample.*.weight : # (torch.Size([384, 384, 3, 3]), torch.bfloat16)
encoder.down_blocks.*.resample.*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.conv_shortcut.weight : # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.conv_shortcut.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.time_conv.weight : # (torch.Size([384, 384, 3, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.time_conv.bias : # (torch.Size([384]), torch.bfloat16)
encoder.mid_block.attentions.*.norm.gamma : # (torch.Size([384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.to_qkv.weight : # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.to_qkv.bias : # (torch.Size([1152]), torch.bfloat16)
encoder.mid_block.attentions.*.proj.weight : # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.proj.bias : # (torch.Size([384]), torch.bfloat16)
encoder.mid_block.resnets.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.mid_block.resnets.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
encoder.mid_block.resnets.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.norm_out.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.conv_out.weight : # (torch.Size([32, 384, 3, 3, 3]), torch.bfloat16)
encoder.conv_out.bias : # (torch.Size([32]), torch.bfloat16)
quant_conv.weight : # (torch.Size([32, 32, 1, 1, 1]), torch.bfloat16)
quant_conv.bias : # (torch.Size([32]), torch.bfloat16)
post_quant_conv.weight : # (torch.Size([16, 16, 1, 1, 1]), torch.bfloat16)
post_quant_conv.bias : # (torch.Size([16]), torch.bfloat16)
decoder.conv_in.weight : # (torch.Size([384, 16, 3, 3, 3]), torch.bfloat16)
decoder.conv_in.bias : # (torch.Size([384]), torch.bfloat16)
decoder.mid_block.attentions.*.norm.gamma : # (torch.Size([384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.to_qkv.weight : # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.to_qkv.bias : # (torch.Size([1152]), torch.bfloat16)
decoder.mid_block.attentions.*.proj.weight : # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.proj.bias : # (torch.Size([384]), torch.bfloat16)
decoder.mid_block.resnets.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
decoder.mid_block.resnets.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
decoder.mid_block.resnets.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.norm*.gamma : # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv*.weight : # (torch.Size([96, 96, 3, 3, 3]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv*.bias : # (torch.Size([96]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.resample.*.weight : # (torch.Size([96, 192, 3, 3]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.resample.*.bias : # (torch.Size([96]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.time_conv.weight : # (torch.Size([768, 384, 3, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.time_conv.bias : # (torch.Size([768]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv_shortcut.weight : # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv_shortcut.bias : # (torch.Size([384]), torch.bfloat16)
decoder.norm_out.gamma : # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
decoder.conv_out.weight : # (torch.Size([3, 96, 3, 3, 3]), torch.bfloat16)
decoder.conv_out.bias : # (torch.Size([3]), torch.bfloat16)
```

### transformer
```
scale_shift_table : # (torch.Size([1, 2, 1536]), torch.float32)
patch_embedding.weight : # (torch.Size([1536, 16, 1, 2, 2]), torch.bfloat16)
patch_embedding.bias : # (torch.Size([1536]), torch.bfloat16)
condition_embedder.time_embedder.linear_*.weight : # (torch.Size([1536, 1536]), torch.float32)
condition_embedder.time_embedder.linear_*.bias : # (torch.Size([1536]), torch.float32)
condition_embedder.time_proj.weight : # (torch.Size([9216, 1536]), torch.bfloat16)
condition_embedder.time_proj.bias : # (torch.Size([9216]), torch.bfloat16)
condition_embedder.text_embedder.linear_*.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
condition_embedder.text_embedder.linear_*.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.scale_shift_table : # (torch.Size([1, 6, 1536]), torch.float32)
blocks.*.attn*.norm_q.weight : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.norm_k.weight : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_q.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_q.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_k.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_k.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_v.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_v.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_out.*.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_out.*.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.norm*.weight : # (torch.Size([1536]), torch.float32)
blocks.*.norm*.bias : # (torch.Size([1536]), torch.float32)
blocks.*.ffn.net.*.proj.weight : # (torch.Size([8960, 1536]), torch.bfloat16)
blocks.*.ffn.net.*.proj.bias : # (torch.Size([8960]), torch.bfloat16)
blocks.*.ffn.net.*.weight : # (torch.Size([1536, 8960]), torch.bfloat16)
blocks.*.ffn.net.*.bias : # (torch.Size([1536]), torch.bfloat16)
proj_out.weight : # (torch.Size([64, 1536]), torch.bfloat16)
proj_out.bias : # (torch.Size([64]), torch.bfloat16)
```

### text encoder
```
shared.weight : # (torch.Size([256384, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.q.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.k.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.v.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.o.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight : # (torch.Size([32, 64]), torch.bfloat16)
encoder.block.*.layer.*.layer_norm.weight : # (torch.Size([4096]), torch.bfloat16)
encoder.block.*.layer.*.DenseReluDense.wi_*.weight : # (torch.Size([10240, 4096]), torch.bfloat16)
encoder.block.*.layer.*.DenseReluDense.wo.weight : # (torch.Size([4096, 10240]), torch.bfloat16)
encoder.final_layer_norm.weight : # (torch.Size([4096]), torch.bfloat16)
```

### adding flash attention inconsistant issue unit test

```
python compare_fa_sharding_consistency.py
...
==============================
      FINAL CONCLUSION
==============================
❌❌❌ [FAILED] The outputs are different.
This provides strong evidence that the problem IS inside the WanTransformer3DModel when Flash Attention is used.
The root cause is likely the self-attention calculation itself being altered by torchax's Flash Attention implementation, when used with this specific tensor sharding strategy.
==============================

```

### wan_tx_splash_attn.py combine the jax pallas splash attention, maxdiffusion vae decoder

```
# on v6e-8:

(venv)$ python wan_tx_splash_attn.py
Load and port Wan 2.1 VAE on tpu
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████| 12/12 [00:01<00:00, 10.11it/s]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.60it/s]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.63it/s]
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Number of devices is:,  8
text_encoder 12.537574768066406 G
transformer 26.66874897480011 G
vae (JAX VAE) - size calculation not implemented
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
100%|█████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:16<00:00,  9.94s/it]
numpy shape: (720, 1280, 3, 81)
100%|█████████████████████████████████████████████████████████████████████████████████████████| 50/50 [06:29<00:00,  7.80s/it]
Iteration 0: 418.294946s
DONE
```

### support flash attention

Current support flash attention to generate correct normal 14B model, 81 frames videos.
Flash attention prevent the huge attention weight which cause OOM.

1.3B model is not yet ready using flash attention since kv_head = 12 cannot divide by 8 tpus.
Disable flash attention for VAE for now since kv_head = 1 in VAE.

Modify flash attention block size to 2048
528s


### multi-host run on v6e-16

1. create tpu vm with v6e-16.
  1. it will create 4 hosts with 4x4 gpus mesh
  2. all the command use gcloud to distribute to all workers.

```
# Remember to replace variable in placeholder
# setup env
export PROJECT_ID=<project_id>
export TPU_NAME=<tpu_name>
export ZONE=<zone>
export ACCELERATOR_TYPE=v6e-16
export RUNTIME_VERSION=v2-alpha-tpuv6e

export ACCOUNT=<account>
export GITHUB_BRANCH=<branch_name>
export GITHUB_ADDRESS=<github_repo_address>

run()
{
  local command=$1
  local worker=${2:-all}
  gcloud compute tpus tpu-vm ssh --zone "${ZONE}" "${ACCOUNT}@${TPU_NAME}" --project "${PROJECT_ID}" --worker=${worker} --command="$command"
}


SETUP_COMMAND="\
set -x && \
sudo apt update && \
sudo apt install -y python3.10-venv && \
python -m venv venv && \
source venv/bin/activate && \
pip install torch --index-url https://download.pytorch.org/whl/cpu && \
pip install jax[tpu] && \
pip install transformers accelerate ftfy tpu-info imageio imageio-ffmpeg tensorflow && \
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes#subdirectory=torchax && \
git clone -b ${GITHUB_BRANCH} ${GITHUB_ADDRESS} || true && \
cd diffusers && \
pip install -e . \
"
# Only need run the first time
# run "${SETUP_COMMAND}"

RUN_COMMAND="\
set -x && \
source ~/venv/bin/activate && \
killall -9 python || true && \
sleep 10 && \
export JAX_COMPILATION_CACHE_DIR="/dev/shm/jax_cache" && \
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && \
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='xla_gpu_per_fusion_autotune_cache_dir' && \
export HF_HUB_CACHE=/dev/shm/hf_cache && \
cd diffusers && \
git fetch && git reset --hard origin/${GITHUB_BRANCH} && \
nohup python wan_tx.py > wan_tx.log 2>&1 & \
"
run "${RUN_COMMAND}"

```
ssh into a VM to collect the log in wan_tx.log and video generated.
