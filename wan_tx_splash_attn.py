import functools
import re
import math
import torch
import torchax
from torchax.ops import ops_registry
import time
import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# Add JAX VAE imports
import flax
from flax import nnx
from maxdiffusion.models.wan.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanUpsample,
    AutoencoderKLWan,
    WanMidBlock,
    WanResidualBlock,
    WanRMS_norm,
    WanResample,
    ZeroPaddedConv2D,
    WanAttentionBlock,
    AutoencoderKLWanCache,
)
from maxdiffusion.models.wan.wan_utils import load_wan_vae
from flax.linen import partitioning as nn_partitioning

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan as TorchAutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from jax.tree_util import register_pytree_node

from transformers import modeling_outputs

from datetime import datetime

# import torchax.ops.jtorch
import traceback
import types
import argparse

#### SETTINGS
# 1.3B
# MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# 14B
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# 720p
FLOW_SHIFT = 5.0 # 5.0 for 720P, 3.0 for 480P
WIDTH = 1280
HEIGHT = 720

# 81 frames
FRAMES = 81
FPS = 16

# step
NUM_STEP = 50
# NUM_STEP = 1

BQSIZE =  3024 # 2240 # 3024 #2520
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024

# <--- NEW: Local Attention Window Size Setting --->
# window_size = (left, right). (128, 0) means each token can attend to itself and the previous 128 tokens.
# Set right=0 to maintain causality for autoregressive models.
# Set to None to use the original full Causal Attention.
WINDOW_SIZE = None

PROFILE_OUT_PATH = "/dev/shm/tensorboard"

USE_DP = True
SP_NUM = 1
USE_FSDP = True

# for shard vae
LOGICAL_AXIS_RULES = (
                    ('conv_out', ('axis','dp','sp')),
                    ('conv_in', ('axis','dp','sp'))
                  )

USE_K_SMOOTH = True

####


axis = 'axis'

# Sharding for tranformers, all the replicated are commented out for speed
transformer_shardings_fsdp = {
# 'scale_shift_table': (), # (torch.Size([1, 2, 1536]), torch.float32)
# 'patch_embedding.weight': (), # (torch.Size([1536, 16, 1, 2, 2]), torch.bfloat16)
# 'patch_embedding.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.time_embedder.linear_1.weight': (None, (axis,'sp'),), # (torch.Size([1536, 256]), torch.float32)
# r'condition_embedder.time_embedder.linear_1.bias': (), # (torch.Size([1536]), torch.float32)
r'condition_embedder.time_embedder.linear_2.weight': ((axis,'sp'), None), # (torch.Size([1536, 1536]), torch.float32)
# r'condition_embedder.time_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.float32)
r'condition_embedder.time_proj.weight': ((axis,'sp'), None,), # (torch.Size([9216, 1536]), torch.bfloat16)
# r'condition_embedder.time_proj.bias': (), # (torch.Size([9216]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_1.weight': (None, (axis,'sp'),), # (torch.Size([1536, 4096]), torch.bfloat16)
# r'condition_embedder.text_embedder.linear_1.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_2.weight': ((axis,'sp'), None), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'condition_embedder.text_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.scale_shift_table': (), # (torch.Size([1, 6, 1536]), torch.float32)
# 'blocks.\d+.attn1.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_q.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn1.to_q.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_k.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn1.to_k.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_v.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn1.to_v.bias': (), # (torch.Size([1536]), torch.bfloat16)
# to_out has 2 submodules, the first is the Linear and second is dropout
r'blocks.\d+.attn1.to_out.0.weight': ((axis,'sp'), None), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn1.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_q.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn2.to_q.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_k.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn2.to_k.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_v.weight': (None, (axis,'sp'),), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn2.to_v.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_out.0.weight': ((axis,'sp'), None), # (torch.Size([1536, 1536]), torch.bfloat16)
# r'blocks.\d+.attn2.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.norm2.weight': (), # (torch.Size([1536]), torch.float32)
# r'blocks.\d+.norm2.bias': (), # (torch.Size([1536]), torch.float32)
r'blocks.\d+.ffn.net.0.proj.weight': (None, (axis,'sp'),), # (torch.Size([8960, 1536]), torch.bfloat16)
# r'blocks.\d+.ffn.net.0.proj.bias': (), # (torch.Size([8960]), torch.bfloat16)
r'blocks.\d+.ffn.net.2.weight': ((axis,'sp'), None), # (torch.Size([1536, 8960]), torch.bfloat16)
# r'blocks.\d+.ffn.net.2.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'proj_out.weight': (None, (axis,'sp'),), # (torch.Size([64, 1536]), torch.bfloat16)
# 'proj_out.bias': (), # (torch.Size([64]), torch.bfloat16)
}

transformer_shardings_tp = {
# 'scale_shift_table': (), # (torch.Size([1, 2, 1536]), torch.float32)
# 'patch_embedding.weight': (), # (torch.Size([1536, 16, 1, 2, 2]), torch.bfloat16)
# 'patch_embedding.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.time_embedder.linear_1.weight': ((axis,'sp'), None), # (torch.Size([1536, 256]), torch.float32)
r'condition_embedder.time_embedder.linear_1.bias': ((axis,'sp'),), # (torch.Size([1536]), torch.float32)
r'condition_embedder.time_embedder.linear_2.weight': (None, (axis,'sp')), # (torch.Size([1536, 1536]), torch.float32)
# 'condition_embedder.time_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.float32)
# 'condition_embedder.time_proj.weight': (), # (torch.Size([9216, 1536]), torch.bfloat16)
# 'condition_embedder.time_proj.bias': (), # (torch.Size([9216]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_1.weight': ((axis,'sp'), None), # (torch.Size([1536, 4096]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_1.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_2.weight': (None, (axis,'sp')), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'condition_embedder.text_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.scale_shift_table': (), # (torch.Size([1, 6, 1536]), torch.float32)
# 'blocks.\d+.attn1.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_q.weight': ((axis,'sp'), None), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_q.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_k.weight': ((axis,'sp'), ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_k.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_v.weight': ((axis,'sp'), ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_v.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
# to_out has 2 submodules, the first is the Linear and second is dropout
r'blocks.\d+.attn1.to_out.0.weight': (None, (axis,'sp')), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_q.weight': ((axis,'sp'), ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_q.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_k.weight': ((axis,'sp'), ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_k.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_v.weight': ((axis,'sp'), ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_v.bias': ((axis,'sp'), ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_out.0.weight': (None, (axis,'sp')), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.norm2.weight': (), # (torch.Size([1536]), torch.float32)
# 'blocks.\d+.norm2.bias': (), # (torch.Size([1536]), torch.float32)
r'blocks.\d+.ffn.net.0.proj.weight': ((axis,'sp'),), # (torch.Size([8960, 1536]), torch.bfloat16)
r'blocks.\d+.ffn.net.0.proj.bias': ((axis,'sp'), ), # (torch.Size([8960]), torch.bfloat16)
r'blocks.\d+.ffn.net.2.weight': (None, (axis,'sp')), # (torch.Size([1536, 8960]), torch.bfloat16)
# 'blocks.\d+.ffn.net.2.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'proj_out.weight': (), # (torch.Size([64, 1536]), torch.bfloat16)
# 'proj_out.bias': (), # (torch.Size([64]), torch.bfloat16)
}

text_encoder_shardings = {
  'shared.weight': ((axis,'dp','sp'), ), # (torch.Size([256384, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.q.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.k.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.v.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.o.weight': (None, (axis,'dp','sp')), # (torch.Size([4096, 4096]), torch.bfloat16)
  # 'encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight': (), # (torch.Size([32, 64]), torch.bfloat16)
  # 'encoder.block.*.layer.*.layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': ((axis,'dp','sp'), ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': ((axis,'dp','sp'), ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, (axis,'dp','sp')), # (torch.Size([4096, 10240]), torch.bfloat16)
  # 'encoder.final_layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
}


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
  result = {}
  for k, v in weight_dict.items():
    for target, sharding in sharding_dict.items():
      if re.fullmatch(target, k) is not None:
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
        break
    else:
      # replicate
      v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))

    result[k] = v
  return result


def flatten_model_output(obj):
  return obj.to_tuple(), type(obj)

def unflatten_model_output(aux, children):
  return aux(*children)

register_pytree_node(
  modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
  flatten_model_output,
  unflatten_model_output)

def make_key(name):
  return re.sub(r'\.\d+\.', '.*.', name)

  
def _get_weights_of_linear(module):

  result = {}

  def fn(start_path, module):
    if isinstance(module, torch.nn.Linear):
      for k, v in module.named_parameters():
        start_path.append(k)
        key = '.'.join(start_path)
        result[key] = v
        start_path.pop()
    else:
      for name, child in module.named_children():
        start_path.append(name)
        fn(start_path, child)
        start_path.pop()
  fn([], module)
  return result


def _print_weights(module):
  all_buffers = dict(module.named_parameters())
  all_buffers.update(module.named_buffers())
  result = {}
  for k, v in all_buffers.items():
    result[make_key(k)] = (v.shape, v.dtype)
  print('{')
  for k, v in result.items():
    print(f"'{k}': (), # {v}")
  print('}')


### Splash attention ###

def _sdpa_reference(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
  L, S = query.size(-2), key.size(-2)
  scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
  if is_causal:
    assert attn_mask is None
    temp_mask = torch.ones(
        L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)
  if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
      attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
      attn_bias += attn_mask
  if enable_gqa:
    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

  attn_weight = query @ key.transpose(-2, -1) * scale_factor
  attn_weight += attn_bias
  attn_weight = torch.softmax(attn_weight, dim=-1)
  if dropout_p > 0:
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  return attn_weight @ value


# <--- MODIFIED: Added window_size parameter to the function signature --->
def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    import jax
    import math
    mesh = env._mesh
    num_heads = query.shape[1]

    # The function that will be sharded across devices.
    def _attention_on_slices(q, k, v):
        import jax.numpy as jnp
        # Scale the query tensor. This happens on each device with its slice of data.
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        # Helper to pad to next multiple
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        # This function operates on a single item from the batch.
        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # self attention
            if k_3d.shape[1] > 10000:
              # Pad q, k, v to next multiple of BQSIZE/BKVSIZE
              q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
              k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
              v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)
            else:
              # do not padding on kv in cross attention. kv length is 512
              q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
              k_3d_padded, k_orig_len = k_3d, k_3d.shape[1]
              v_3d_padded, v_orig_len = v_3d, v_3d.shape[1]

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # ======================= NEW MASK LOGIC =======================
            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )
            # =============================================================

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            # Remove padding if any
            return out[:, :q_orig_len, ...]

        # Map the kernel over the batch dimension.
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # Determine the partitioning spec based on the number of heads.
    if num_heads < mesh.size:
        # Replicated case for VAE. All devices get the full tensor.
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # Sharded case for Transformer. Split along the heads axis.
        # Attn1 self attention, key length is long.
        if key.shape[2] > 10000:
          q_partition_spec = P('dp', 'axis', 'sp', None)
          kv_partition_spec = P('dp', 'axis', None, None)
        else:
          # Attn2 which is cross attention, kv sequence is shorter. All gather the key value cost less.
          q_partition_spec = P('dp', None, ('axis', 'sp'), None)
          kv_partition_spec = P('dp', None, None, None)

    # ALWAYS use shard_map. The partition_spec will control the behavior.
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    out = jax.lax.with_sharding_constraint(out, P('dp', None, ('axis', 'sp'), None))
    return out


# <--- MODIFIED: Added window_size parameter to the function signature --->
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    env=None,
    window_size=None, # <--- NEW
) -> torch.Tensor:
  # Debug prints to understand what's happening
  #print(f"[DEBUG] scaled_dot_product_attention called with:")
  #print(f"  query.shape={query.shape}")
  #print(f"  key.shape={key.shape}")
  #print(f"  value.shape={value.shape}")
  #print(f"  query.shape[-1]={query.shape[-1]}")
  #print(f"  window_size={window_size}")
  #print(f"  env.config.use_tpu_splash_attention={env.config.use_tpu_splash_attention if env else 'None'}")

  if env.config.use_tpu_splash_attention:
    #print(f"[DEBUG] Using splash attention")
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    if USE_K_SMOOTH:
      key_mean = jnp.mean(jkey, axis=2, keepdims=True)
      # jkey_smoothed
      jkey = jkey - key_mean
    # <--- MODIFIED: Pass window_size to the backend function --->
    res = _tpu_splash_attention(jquery, jkey, jvalue, env, scale=scale, is_causal=is_causal, window_size=window_size)
    return env.j2t_iso(res)

  #print(f"[DEBUG] Using reference implementation (fallback)")
  return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                         scale, enable_gqa)

###

# Fix for torch2jax compatibility issue
def load_wan_vae_fixed(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
    """Fixed version of load_wan_vae that avoids torch2jax issues"""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from flax.traverse_util import unflatten_dict, flatten_dict
    
    device_obj = jax.local_devices(backend=device)[0]
    with jax.default_device(device_obj):
        if hf_download:
            ckpt_path = hf_hub_download(
                pretrained_model_name_or_path, subfolder="vae", filename="diffusion_pytorch_model.safetensors"
            )
        print(f"Load and port Wan 2.1 VAE on {device}")

        if ckpt_path is not None:
            tensors = {}
            
            # Use safetensors with numpy framework to avoid torchax interference
            with safe_open(ckpt_path, framework="np") as f:
                for k in f.keys():
                    # Get numpy array directly
                    numpy_tensor = f.get_tensor(k)
                    tensors[k] = jnp.array(numpy_tensor)
            
            flax_state_dict = {}
            cpu = jax.local_devices(backend="cpu")[0]
            
            # Import the utility functions
            from maxdiffusion.models.modeling_flax_pytorch_utils import rename_key, rename_key_and_reshape_tensor, validate_flax_state_dict
            
            for pt_key, tensor in tensors.items():
                renamed_pt_key = rename_key(pt_key)
                # Order matters
                renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
                renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
                renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

                renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
                renamed_pt_key = renamed_pt_key.replace("conv_in.weight", "conv_in.conv.weight")
                renamed_pt_key = renamed_pt_key.replace("conv_out.bias", "conv_out.conv.bias")
                renamed_pt_key = renamed_pt_key.replace("conv_out.weight", "conv_out.conv.weight")
                renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
                renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
                renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
                renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
                renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
                renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
                renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
                renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
                renamed_pt_key = renamed_pt_key.replace("time_conv.bias", "time_conv.conv.bias")
                renamed_pt_key = renamed_pt_key.replace("time_conv.weight", "time_conv.conv.weight")
                renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
                renamed_pt_key = renamed_pt_key.replace("conv_shortcut", "conv_shortcut.conv")
                if "decoder" in renamed_pt_key:
                    renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
                    renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
                if "encoder" in renamed_pt_key:
                    renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
                pt_tuple_key = tuple(renamed_pt_key.split("."))
                flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
                flax_key = tuple(int(item) if isinstance(item, str) and item.isdigit() else item for item in flax_key)
                flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
            
            validate_flax_state_dict(eval_shapes, flax_state_dict)
            flax_state_dict = unflatten_dict(flax_state_dict)
            del tensors
            jax.clear_caches()
        else:
            raise FileNotFoundError(f"Path {ckpt_path} was not found")

        return flax_state_dict

### Sharding VAE ###

def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs

@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = functools.partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model

#####################


# --- Config Wrapper ---
class ConfigWrapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)

def to_torch_recursive(x):
    import torch
    import numpy as np
    if 'ArrayImpl' in str(type(x)):
        return torch.from_numpy(np.array(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    elif hasattr(x, 'sample'):
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    else:
        return x

class VAEProxy:
    def __init__(self, vae, vae_cache, dtype, config):
        self._vae = vae
        self.vae_cache = vae_cache
        self.dtype = dtype
        self.config = config
    def __getattr__(self, name):
        return getattr(self._vae, name)
    def decode(self, *args, **kwargs):
        if 'feat_cache' not in kwargs:
            kwargs['feat_cache'] = self.vae_cache
        out = self._vae.decode(*args, **kwargs)
        return to_torch_recursive(out)

def prepare_video_for_export(video):
    import torch
    import numpy as np
    if isinstance(video, (list, tuple)):
        print("output 是 list/tuple，长度：", len(video))
        return [prepare_video_for_export(v) for v in video]
    if isinstance(video, torch.Tensor):
        print("原始 shape:", video.shape)
        if video.dim() == 5:  # (B, C, T, H, W)
            video = video[0]
        if video.dim() == 4 and video.shape[0] != args.frames:  # (C, T, H, W)
            video = video.permute(1, 0, 2, 3)
        # (T, C, H, W) -> (T, H, W, C)
        if video.shape[-1] == 3:
            pass
        else:
            video = video.permute(0, 2, 3, 1)
        print("转置后 shape:", video.shape)
        if video.shape[-1] > 3:
            video = video[..., :3]
        if video.shape[-1] not in [1, 2, 3, 4]:
            video = torch.unsqueeze(video, -1)
        print("裁剪/补齐后 shape:", video.shape)
        video = video.cpu().numpy()
        video = np.clip(video, 0, 255).astype(np.uint8)
        print("最终 numpy shape:", video.shape)
        # 如果是灰度，自动扩展为 3 通道
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        # 检查每一帧的 channel
        for i, frame in enumerate(video):
            if frame.shape[-1] not in [1, 2, 3, 4]:
                print(f"第{i}帧 shape: {frame.shape}")
        return video
    if isinstance(video, np.ndarray):
        print("numpy shape:", video.shape)
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    print("未知类型：", type(video))
    return video

def sharded_device_put(tensor, sharding):
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)
  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)

def main():

  if hasattr(flax.config, 'flax_always_shard_variable'):
    flax.config.update('flax_always_shard_variable', False)

  # Set JAX config to enable compilation cache
  jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

  torch.set_default_dtype(torch.bfloat16)
  # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  #model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  model_id = args.model_id
  
  # Initialize JAX environment first
  torchax.enable_globally()
  env = torchax.default_env()
  # Create a 2D mesh for FSDP sharding
  
  tp_dim, dp_dim, sp_dim = len(jax.devices()), 1, 1
  if args.use_dp:
    # tp_dim > 8, which is v6e-16, could not divide head_dim=40, need use dp
    print(f"{args.use_dp=}")
    tp_dim //= 2
    dp_dim = 2
  
  if args.sp_num > 1:
    print(f"{args.sp_num=}")
    tp_dim //= args.sp_num
    sp_dim = args.sp_num

  print(f"{tp_dim=}, {dp_dim=}, {sp_dim=}")
     
  # mesh = jax.make_mesh((len(jax.devices()), 1), (axis, 'fsdp'))
  mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
  mesh = Mesh(mesh_devices, (axis,'dp','sp'))

  env.default_device_or_sharding = NamedSharding(mesh, P())
  env._mesh = mesh
  env.config.use_tpu_splash_attention = True

  # Initialize JAX VAE
  key = jax.random.key(0)
  rngs = nnx.Rngs(key)
  
  # Create JAX VAE with default parameters
  wan_vae = AutoencoderKLWan(
      rngs=rngs,
      base_dim=96,
      z_dim=16,
      dim_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      attn_scales=[],
      temperal_downsample=[False, True, True],
      mesh=mesh
  )
  
  with mesh:
    # Create VAE cache
    vae_cache = AutoencoderKLWanCache(wan_vae)
    
    # Load pretrained weights
    graphdef, state = nnx.split(wan_vae)
    params = state.to_pure_dict()
    params = load_wan_vae_fixed(model_id, params, "tpu")
    # 保证全部 replicate 到 mesh 上所有 device
    sharding = NamedSharding(mesh, P())
    params = jax.tree_util.tree_map(lambda x: sharded_device_put(x, sharding), params)
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    wan_vae = nnx.merge(graphdef, params)

    # Shard vae
    p_create_sharded_logical_model = functools.partial(create_sharded_logical_model, logical_axis_rules=LOGICAL_AXIS_RULES)
    wan_vae = p_create_sharded_logical_model(model=wan_vae)
  
  
  # Skip PyTorch VAE loading to avoid torchax interference
  # We'll use JAX VAE directly
  
  # Temporarily disable torchax to load pipeline components
  torchax.disable_globally()
  
  try:
    # flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    flow_shift = args.flow_shift
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    
    # Load pipeline without VAE to avoid torchax interference
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe.scheduler = scheduler
  finally:
    # Re-enable torchax for the rest of the pipeline
    torchax.enable_globally()
  
  # Replace the VAE in the pipeline with our JAX VAE
  vae_config = ConfigWrapper(
      latents_mean=np.array(wan_vae.latents_mean),
      latents_std=np.array(wan_vae.latents_std),
      z_dim=wan_vae.z_dim
  )
  pipe.vae = VAEProxy(wan_vae, vae_cache, torch.bfloat16, vae_config)
  pipe.vae_cache = vae_cache

  # 伪装 config
  vae_config = ConfigWrapper(
      latents_mean=np.array(wan_vae.latents_mean),
      latents_std=np.array(wan_vae.latents_std),
      z_dim=wan_vae.z_dim
  )
  pipe.vae.config = vae_config

  # print('vae=====')
  # _print_weights(pipe.vae)
  # print('trans===')
  # print(_get_weights_of_linear(pipe.transformer).keys())
  # print('encoder===')
  # _print_weights(pipe.text_encoder)
  # return

  def _move_module(module):
    with jax.default_device('cpu'):
      state_dict  = module.state_dict()
      state_dict = env.to_xla(state_dict)
      module.load_state_dict(state_dict, assign=True)

  # Re-enable torchax for the rest of the pipeline
  # torchax.enable_globally()  # Already enabled above
  # env = torchax.default_env()  # Already initialized above
  # mesh = jax.make_mesh((len(jax.devices()), 1), (axis, 'fsdp'))  # Already created above
  # env.default_device_or_sharding = NamedSharding(mesh, P())  # Already set above
  # env._mesh = mesh  # Already set above
  # env.config.use_tpu_splash_attention = True  # Already set above

  # <--- MODIFIED: Override flash attention with custom function, now with window_size --->
  custom_attention = functools.partial(
      scaled_dot_product_attention,
      env=env,
      window_size=args.window_size # Inject the global window size setting here
  )
  # Workaround for the function lack is_view_op argument
  # env.override_op_definition(torch.nn.functional.scaled_dot_product_attention, custom_attention)
  op_to_override = torch.nn.functional.scaled_dot_product_attention
  op_impl = custom_attention
  env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )

  # Compile modules with torchax (skip VAE as it's already JAX)
  vae_options = torchax.CompileOptions(
    methods_to_compile=['decode']
  )
  # Skip VAE compilation as it's already JAX
  # _move_module(pipe.vae)
  # pipe.vae = torchax.compile(pipe.vae)
  
  if args.t5_cpu:
    # 只把 text_encoder 移到 CPU，不做 compile 和 shard
    pipe.text_encoder.to("cpu")
  else:
    # TPU 路径，做 compile 和 shard
    _move_module(pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = _shard_weight_dict(pipe.text_encoder.params, text_encoder_shardings, mesh)
    pipe.text_encoder.buffers = _shard_weight_dict(pipe.text_encoder.buffers, text_encoder_shardings, mesh)

  # the param below is not declared as param or buffer so the module.to('jax') didnt work
  _move_module(pipe.transformer)
  pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
  options = torchax.CompileOptions(
      jax_jit_kwargs={'static_argnames': ('return_dict',)}
  )
  pipe.transformer = torchax.compile(pipe.transformer, options)

  #pipe.to('jax')
  print('Number of devices is:, ', len(jax.devices()))

  if args.use_fsdp:
    transformer_shardings = transformer_shardings_fsdp
  else:
    transformer_shardings = transformer_shardings_tp

  pipe.transformer.params = _shard_weight_dict(pipe.transformer.params, 
                                               transformer_shardings,
                                               mesh)
  pipe.transformer.buffers = _shard_weight_dict(pipe.transformer.buffers, 
                                               transformer_shardings,
                                               mesh)

  # Skip VAE sharding as it's already JAX and handled differently
  # pipe.vae.params = _shard_weight_dict(pipe.vae.params, {}, mesh)
  # pipe.vae.buffers = _shard_weight_dict(pipe.vae.buffers, {}, mesh)

  def move_scheduler(scheduler):
    for k, v in scheduler.__dict__.items():
      if isinstance(v, torch.Tensor):
        setattr(scheduler, k, v.to('jax'))

  #move_scheduler(pipe.scheduler)

  def module_size(module):
    size = 0
    for k, v in module.state_dict().items():
      size += math.prod(v.shape) * v.dtype.itemsize
    return size

  for m in dir(pipe):
      module = getattr(pipe, m, None)
      if isinstance(module, torch.nn.Module):
          print(m, module_size(module) / (1024 * 1024 * 1024), 'G')
      elif m == 'vae' and hasattr(pipe, 'vae_cache'):
          # JAX VAE size calculation
          print(f"{m} (JAX VAE) - size calculation not implemented")


  prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
  # prompt = "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and greenshrubbery covers the cliffs edge. The steep drop from the road down to the beach is adramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."
  negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

  generator = torch.Generator()
  generator.manual_seed(42)
  with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
    # warm up and save video
    pipe_kwargs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 5.0,
        'generator': generator,
        'use_dp': args.use_dp,
    }
    
    output = pipe(**pipe_kwargs).frames[0]
    #print("output type:", type(output), "output shape:", output.shape)
    #if hasattr(output, 'shape'):
    #    print("output shape:", output.shape)
    #elif isinstance(output, (list, tuple)):
    #    for i, v in enumerate(output):
    #        print(f"output[{i}] type: {type(v)}, shape: {getattr(v, 'shape', None)}")
    output = prepare_video_for_export(output)
    if isinstance(output, np.ndarray) and output.ndim == 4 and output.shape[-2] == 3:
        output = output.transpose(3, 0, 1, 2)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{current_datetime}.mp4"
    export_to_video(output, file_name, fps=args.fps)
    print(f"output video done. {file_name}")
    jax.effects_barrier()
    
    if args.profile:
      # profile set fewer step and output latent to skip VAE for now
      # output_type='latent' will skip VAE
      jax.profiler.start_trace(PROFILE_OUT_PATH)
      output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=3,
        num_frames=args.frames,
        guidance_scale=5.0,
        output_type="latent",
        generator=generator,
        use_dp=args.use_dp,
      )
      jax.effects_barrier()
      jax.profiler.stop_trace()
      print("profile done")
    
    # Benchmark loop
    for i in range(1):
      start = time.perf_counter()
      output = pipe(**pipe_kwargs)
      # make sure all computation done
      jax.effects_barrier()
      end = time.perf_counter()  
      print(f'Iteration {i} BKVCOMPUTESIZE={BKVCOMPUTESIZE} BKVSIZE={BKVSIZE}, BQSIZE={BQSIZE}: {end - start:.6f}s')
        
  print('DONE')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--flow_shift", type=float, default=FLOW_SHIFT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEP)
    parser.add_argument("--window_size", type=int, nargs=2, default=None)
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--sp_num", type=int, default=SP_NUM)
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Offload T5 text_encoder to CPU")
    parser.add_argument("--bqsize", type=int, default=BQSIZE, help="Block Q size")
    parser.add_argument("--bkvsize", type=int, default=BKVSIZE, help="Block KV size")
    parser.add_argument("--bkvcomputesize", type=int, default=BKVCOMPUTESIZE, help="Block KV compute size")
    parser.add_argument("--profile", action="store_true", default=False, help="Add profiler")
    parser.add_argument("--use_fsdp", type=bool, default=USE_FSDP, help="Use FSDP")
    parser.add_argument("--use_k_smooth", type=bool, default=USE_K_SMOOTH, help="Use K smooth")
    return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(args)
  BQSIZE = args.bqsize
  BKVSIZE = args.bkvsize
  BKVCOMPUTESIZE = args.bkvcomputesize
  USE_K_SMOOTH = args.use_k_smooth
  main()
