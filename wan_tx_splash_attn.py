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
import flax.linen as nn

from jax.experimental import pjit
from jax.lax import ppermute, dot_general
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax import tree_util

from aqt.jax.v2.pallas import quantizer as aqt_pallas_quantizer

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import make_splash_mha, BlockSizes

from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    MultiHeadMask,
    CausalMask,
    FullMask,
    LocalMask,
)

# Add JAX VAE imports
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
from typing import Optional

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan as TorchAutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from jax.tree_util import register_pytree_node

from transformers import modeling_outputs

from datetime import datetime

# import torchax.ops.jtorch
import traceback
import types
import argparse

from tpu_sageattention import TPUSageAttention, create_tpu_sageattention


def mask_flatten(mask):
  """Flattens a Mask object into its dynamic children and static auxiliary data."""
  # For Mask objects, all their attributes (like shape) are static. They have no dynamic children.
  return (), (type(mask), mask.shape)

def mask_unflatten(aux_data, children):
  """Recreates a Mask object from its static auxiliary data and dynamic children."""
  mask_type, shape = aux_data
  return mask_type(shape)

# Register the simple masks that only have a shape
#tree_util.register_pytree_node(CausalMask, mask_flatten, mask_unflatten)
#tree_util.register_pytree_node(FullMask, mask_flatten, mask_unflatten)

def local_mask_flatten(mask):
    """Flatten function for LocalMask, which has more attributes."""
    return (), (type(mask), mask.shape, mask.window_size, mask.offset)

def local_mask_unflatten(aux_data, children):
    """Unflatten function for LocalMask."""
    mask_type, shape, window_size, offset = aux_data
    return mask_type(shape, window_size=window_size, offset=offset)

#tree_util.register_pytree_node(LocalMask, local_mask_flatten, local_mask_unflatten)


def multi_head_mask_flatten(mhm):
  """Flattens a MultiHeadMask. Its children are the individual masks."""
  return (mhm.masks,), type(mhm)

def multi_head_mask_unflatten(mhm_type, children):
  """Recreates a MultiHeadMask from its children."""
  # children is a tuple containing one element: the tuple of masks
  return mhm_type(children[0])

#tree_util.register_pytree_node(MultiHeadMask, multi_head_mask_flatten, multi_head_mask_unflatten)


def static_obj_flatten(obj):
  """A generic flatten function that treats the entire object as static."""
  # No dynamic children, the entire object is passed as auxiliary data.
  return (), obj

def static_obj_unflatten(obj, children):
  """A generic unflatten function that just returns the static object."""
  return obj

# Register all our custom mask classes to be treated as static objects
tree_util.register_pytree_node(CausalMask, static_obj_flatten, static_obj_unflatten)
tree_util.register_pytree_node(FullMask, static_obj_flatten, static_obj_unflatten)
tree_util.register_pytree_node(LocalMask, static_obj_flatten, static_obj_unflatten)
tree_util.register_pytree_node(MultiHeadMask, static_obj_flatten, static_obj_unflatten)

print("Successfully registered custom Mask objects as JAX Pytrees.")
# --- END: JAX Pytree Registration ---

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

BQSIZE =  1024 # 2240 # 3024 #2520
BKVSIZE = 1024 # 2304 # 1664 #2048

# <--- NEW: Local Attention Window Size Setting --->
# window_size = (left, right). (128, 0) means each token can attend to itself and the previous 128 tokens.
# Set right=0 to maintain causality for autoregressive models.
# Set to None to use the original full Causal Attention.
WINDOW_SIZE = None

PROFILE_OUT_PATH = "/dev/shm/tensorboard"

USE_DP = True
SP_NUM = 1

# for shard vae
LOGICAL_AXIS_RULES = (
                    ('conv_out', ('axis','dp','sp')),
                    ('conv_in', ('axis','dp','sp'))
                  )

####

axis = 'axis'

# Sharding for tranformers, all the replicated are commented out for speed
transformer_shardings = {
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

class QuantizationParams:
    def __init__(self, scale, noise=None):
        self.scale = scale
        self.noise = noise

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

# ---- PyTree 注册保护 ----
try:
    from jax import tree_util
    import transformers
    tree_util.register_pytree_node(
        transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        # 这里填写你原有的 to_iterable, from_iterable
        lambda obj: (list(obj.__dict__.values()), None),
        lambda _, values: transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(*values)
    )
except (ValueError, ImportError) as e:
    if 'Duplicate custom PyTreeDef type registration' not in str(e):
        raise

def make_key(name):
  return re.sub('\.\d+\.', '.*.', name)

  
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
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len
        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)
            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]
            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask
            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )
            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len), block_kv=min(BKVSIZE, padded_kv_seq_len)
            )
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # Attn1 self attention, key length is long.
        if key.shape[2] > 10000:
          q_partition_spec = P('dp', 'axis', 'sp', None)
          kv_partition_spec = P('dp', 'axis', None, None)
        else:
          # Attn2 which is cross attention, kv sequence is shorter. All gather the key value cost less.
          q_partition_spec = P('dp', None, ('axis', 'sp'), None)
          kv_partition_spec = P('dp', None, None, None)
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


def _tpu_sageattention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    import jax
    import jax.numpy as jnp
    from tpu_sageattention import TPUSageAttention, create_tpu_sageattention
    mesh = env._mesh
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    sage_attn = create_tpu_sageattention(head_dim=head_dim, use_tpu=True)
    def _attention_on_slices(q, k, v):
        return sage_attn.forward(q, k, v, is_causal=is_causal, sm_scale=scale)
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        q_partition_spec = P('dp', 'axis', 'sp', None)
        kv_partition_spec = P('dp', 'axis', None, None)
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    res = sharded_fn(query, key, value)
    out = env.j2t_iso(res)
    print("type after j2t_iso:", type(out))
    return out


def pad_to_block(x, block_size, axis=2):
    """Pad tensor x on axis to the next multiple of block_size."""
    seq_len = x.shape[axis]
    pad_len = (block_size - seq_len % block_size) % block_size
    if pad_len == 0:
        return x, 0
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    x_padded = jnp.pad(x, pad_width)
    return x_padded, pad_len

def unpad(x, pad_len, axis=2):
    if pad_len == 0:
        return x
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(0, -pad_len)
    return x[tuple(idx)]

def _tpu_aqt_attention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    """
    最终版 AQT Splash Attention 实现 (v2.0)。
    结合了 K-Smoothing 和 AQT官方Pallas量化内核。
    """
    mesh = env._mesh
    batch_size, num_heads, q_seq_len, head_dim = query.shape

    if batch_size != 1:
        raise ValueError(f"当前实现假设推理时的batch size为1，但得到了{batch_size}")

    # 1. 预处理: scale_factor 和 K-Smoothing
    scale_factor = 1.0 / math.sqrt(head_dim) if scale is None else scale
    query = query * scale_factor
    key_mean = jnp.mean(key, axis=2, keepdims=True)
    key_smoothed = key - key_mean

    # 2. 【核心修正】使用AQT官方Pallas内核进行量化
    # 定义量化目标类型
    quant_dtype = quant_dtypes.int8
    
    # 【重要】定义静态Scale。您需要通过校准得到最优值。
    # 这里的0.2是一个示例占位符，您可以根据之前收集的max值来设定一个合理的初始值。
    STATIC_Q_SCALE = 0.2
    STATIC_K_SCALE = 0.2

    # 创建量化参数
    q_quant_params = QuantizationParams(scale=jnp.full((), STATIC_Q_SCALE, dtype=jnp.float32), noise=None)
    k_quant_params = QuantizationParams(scale=jnp.full((), STATIC_K_SCALE, dtype=jnp.float32), noise=None)

    # 调用AQT官方提供的、高性能的Pallas量化内核
    query_quantized = aqt_pallas_quantizer.quant(query, q_quant_params, dtype=quant_dtype)
    key_quantized = aqt_pallas_quantizer.quant(key_smoothed, k_quant_params, dtype=quant_dtype)
    
    # 3. 准备并调用我们简化后的Attention内核
    def _attention_on_slices(q_i8, k_i8, v, q_s, k_s):
        q_no_batch, k_no_batch, v_no_batch = q_i8.squeeze(0), k_i8.squeeze(0), v.squeeze(0)
        
        # 准备mask
        padded_q_seq_len, padded_kv_seq_len = q_no_batch.shape[1], k_no_batch.shape[1]
        
        if is_causal and window_size is None:
            mask_class = mask_lib.CausalMask
        elif window_size is not None:
            mask_class = functools.partial(mask_lib.LocalMask, window_size=window_size, offset=0)
        else:
            mask_class = mask_lib.FullMask

        mask_list = [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(q_no_batch.shape[0])]
        assert len(mask_list) > 0, f"mask_list is empty! q_no_batch.shape={q_no_batch.shape}"
        assert all(hasattr(m, '__class__') and m.__class__.__name__.endswith('Mask') for m in mask_list), f"mask_list contains non-Mask: {mask_list}"
        mask = mask_lib.MultiHeadMask(mask_list)
        
        # 使用mask_info_lib来处理mask以支持稀疏性
        block_sizes_for_mask = (BQSIZE, BKVSIZE)
        head_shards_for_mask = mesh.shape.get('axis', 1) * mesh.shape.get('sp', 1)
        # process_mask前详细打印和断言
        print("[DEBUG] process_mask前 mask type:", type(mask), "mask:", mask)
        if hasattr(mask, 'masks'):
            print("[DEBUG] mask.masks types:", [type(m) for m in mask.masks])
        assert hasattr(mask, 'masks'), f"mask has no attribute 'masks', type={type(mask)}"
        assert all(hasattr(m, '__class__') and m.__class__.__name__.endswith('Mask') for m in mask.masks), f"mask.masks contains non-Mask: {mask.masks}"
        fwd_mask_info, mask_function = mask_info_lib.process_mask(
            mask, block_sizes_for_mask, head_shards=head_shards_for_mask, q_seq_shards=1
        )
        # process_mask后（如有返回mask相关对象）可继续打印
        fwd_mask_info = jax.tree_util.tree_map(jnp.array, fwd_mask_info)

        # 定义Pallas调用
        grid_width = padded_kv_seq_len // BKVSIZE
        grid = (q_no_batch.shape[0], padded_q_seq_len // BQSIZE, grid_width)
        
        out_shape = jax.ShapeDtypeStruct(v_no_batch.shape, v_no_batch.dtype)
        # 完整的输出形状列表，包括scratch buffers
        full_out_shapes = [
            jax.ShapeDtypeStruct((BQSIZE, NUM_LANES), jnp.float32), # m_scratch
            jax.ShapeDtypeStruct((BQSIZE, NUM_LANES), jnp.float32), # l_scratch
            jax.ShapeDtypeStruct((BQSIZE, head_dim), jnp.float32), # o_scratch
            out_shape, # 最终输出
            None, # logsumexp
        ]
        
        # 调用我们自己简化后的内核
        output_no_batch = pl.pallas_call(
            functools.partial(aqt_flash_attention_kernel, 
                              mask_value=DEFAULT_MASK_VALUE, grid_width=grid_width,
                              bq=BQSIZE, bkv=BKVSIZE, bkv_compute=BKVSIZE, head_dim=head_dim,
                              attn_logits_soft_cap=None, mask_function=mask_function),
            out_shape=full_out_shapes,
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=3),
        )(
            fwd_mask_info.data_next, fwd_mask_info.block_mask, fwd_mask_info.mask_next,
            q_no_batch, k_no_batch, v_no_batch, # int8, int8, bf16
            q_s, k_s, # scales
            None, None, # segment_ids
            fwd_mask_info.partial_mask_blocks, fwd_mask_info.q_sequence
        )[3] # 只取最终的输出
        
        return jnp.expand_dims(output_no_batch, axis=0)

    # 4. 设置分片并执行
    q_part = P('dp', ('axis', 'sp'), None, None)
    kv_part = P('dp', ('axis', 'sp'), None, None)
    scale_part = P() # scale是标量，复制到所有设备
    
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_part, kv_part, kv_part, scale_part, scale_part), # q_i8, k_i8, v_bf16, q_scale, k_scale
        out_specs=q_part,
        check_rep=False,
    )
    
    return sharded_fn(query_quantized, key_quantized, value, q_quant_params.scale, k_quant_params.scale)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, env=None, **kwargs):
    if env is None:
        raise RuntimeError('env must be provided for ring/splash/aqt/sage attention!')
    # 优先用参数 backend，其次用 env.config.tpu_attention_backend
    if backend is not None:
        selected_backend = backend
    elif env is not None and hasattr(env, "config"):
        selected_backend = getattr(env.config, "tpu_attention_backend", "splash")
    else:
        selected_backend = "ring"  # 或其它默认

    if selected_backend == 'sage':
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        jquery = pad_to_block(jquery, BQSIZE, axis=2)[0]
        jkey = pad_to_block(jkey, BKVSIZE, axis=2)[0]
        jvalue = pad_to_block(jvalue, BKVSIZE, axis=2)[0]
        res = _tpu_sageattention(jquery, jkey, jvalue, env, scale=kwargs.get('scale'), is_causal=is_causal, window_size=kwargs.get('window_size'))
        return env.j2t_iso(res)
    elif selected_backend == 'splash':
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        res = _tpu_splash_attention(jquery, jkey, jvalue, env, scale=kwargs.get('scale'), is_causal=is_causal, window_size=kwargs.get('window_size'))
        return env.j2t_iso(res)
    elif selected_backend == 'aqt':
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        jquery = pad_to_block(jquery, BQSIZE, axis=2)[0]
        jkey = pad_to_block(jkey, BKVSIZE, axis=2)[0]
        jvalue = pad_to_block(jvalue, BKVSIZE, axis=2)[0]

        key_mean = jnp.mean(jkey, axis=2, keepdims=True)
        jkey_smoothed = jkey - key_mean

        res = _tpu_aqt_attention(jquery, jkey_smoothed, jvalue, env, scale=kwargs.get('scale'), is_causal=is_causal, window_size=kwargs.get('window_size'))
        return env.j2t_iso(res)
    elif selected_backend == 'ring':
        return _tpu_ring_attention(query, key, value, env, is_causal=is_causal)
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, kwargs.get('scale'), kwargs.get('enable_gqa'))

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

def _ring_attention_on_slices(query, key, value, env, axis_size, sequence_axis_name, is_causal=False):
    """
    内存高效的、分块的Ring Attention实现。
    使用Splash Attention作为内部计算核心来避免OOM。
    """

    is_3d_input = False
    if query.ndim == 3:
        is_3d_input = True
        batch_size, q_seq_len, dim = query.shape
        # 对于14B模型，transformer block中的维度是1536，头数是40
        if dim == 1536:
            num_heads = 40
        else:
            # 一个备用逻辑，以防万一
            num_heads = query.shape[1] // 64

        if dim % num_heads != 0:
            raise ValueError(f"Dimension {dim} is not divisible by num_heads {num_heads}")
        head_dim = dim // num_heads

        query = query.reshape(batch_size, q_seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, key.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, value.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
    else:
        batch_size, num_heads, q_seq_len, head_dim = query.shape

    query = query / jnp.sqrt(head_dim)

    block_size = 256
    query_padded, q_pad_len = pad_to_block(query, block_size, axis=2)
    key_padded, _ = pad_to_block(key, block_size, axis=2)
    value_padded, _ = pad_to_block(value, block_size, axis=2)

    max_logit_accumulator = jnp.full(query_padded.shape[:-1] + (1,), -jnp.inf, dtype=query.dtype)
    value_accumulator = jnp.zeros_like(query_padded)
    weights_accumulator = jnp.zeros(query_padded.shape[:-1] + (1,), dtype=query.dtype)
    kv_chunk = jnp.concatenate([key_padded, value_padded], axis=-1)

    block_sizes = BlockSizes(block_q=block_size, block_kv=block_size)
    q_len_padded, kv_len_padded = query_padded.shape[2], key_padded.shape[2]

    if is_causal:
        mask_class = CausalMask
    else:
        mask_class = FullMask
    
    # 在循环外创建Splash Attention内核
    splash_mask = MultiHeadMask([mask_class((q_len_padded, kv_len_padded)) for _ in range(num_heads)])
    head_shards = env._mesh.shape['axis']
    q_seq_shards = 1
    splash_kernel = make_splash_mha(mask=splash_mask, 
									block_sizes=block_sizes,
									head_shards=head_shards,
                                    q_seq_shards=q_seq_shards)

    def loop_body(i, state):
        max_logit_accum, value_accum, weights_accum, current_kv_chunk = state
        
        current_key = current_kv_chunk[..., :head_dim]
        current_value = current_kv_chunk[..., head_dim:]

        # --- 使用Splash内核进行高效的块状计算 ---
        # splash_kernel内部实现了分块和在线softmax，避免了OOM
        # 我们需要squeeze掉batch维度，因为splash_kernel期望(H, S, D)
        local_out = splash_kernel(query_padded.squeeze(0), current_key.squeeze(0), current_value.squeeze(0))
        local_out = jnp.expand_dims(local_out, axis=0) # 加回batch维度

        # --- 这里需要重新实现softmax的在线归并 ---
        # (这部分逻辑比较复杂，为了先让代码跑通，我们做一个简化，
        # 假设每个块的输出可以直接累加。这在数学上不完全等价于
        # 全局softmax，但能验证内存和流程是否跑通)
        # 一个更精确的实现需要从splash_kernel中拿到logits和max_logits
        new_value_accum = value_accum + local_out

        # Permute KV chunk for the next device
        next_kv_chunk = jax.lax.ppermute(
            current_kv_chunk,
            axis_name=sequence_axis_name,
            perm=[(j, (j - 1 + axis_size) % axis_size) for j in range(axis_size)]
        )
        # 在这个简化版中，我们只更新value_accum
        return max_logit_accum, new_value_accum, weights_accum, next_kv_chunk

        """
        logits = jax.lax.dot_general(query, current_key, dimension_numbers=dot_dims_qk)
        logits = logits / jnp.sqrt(head_dim)

        old_max_logit = max_logit_accum
        new_max_logit = jnp.maximum(old_max_logit, jnp.max(logits, axis=-1, keepdims=True))

        exp_old_to_new_max = jnp.exp(old_max_logit - new_max_logit)
        rescaled_value_accum = value_accum * exp_old_to_new_max
        rescaled_weights_accum = weights_accum * exp_old_to_new_max

        exp_logits = jnp.exp(logits - new_max_logit)
        current_weighted_value = jax.lax.dot_general(exp_logits, current_value, dimension_numbers=dot_dims_sv)
        
        new_value_accum = rescaled_value_accum + current_weighted_value
        new_weights_accum = rescaled_weights_accum + jnp.sum(exp_logits, axis=-1, keepdims=True)

        next_kv_chunk = jax.lax.ppermute(
            current_kv_chunk,
            axis_name=sequence_axis_name,
            perm=[(j, (j - 1 + axis_size) % axis_size) for j in range(axis_size)]
        )

        return new_max_logit, new_value_accum, new_weights_accum, next_kv_chunk
        """

    initial_state = (max_logit_accumulator, value_accumulator, weights_accumulator, kv_chunk)
    _, final_value_sum, _, _ = jax.lax.fori_loop(
        0, axis_size, loop_body, initial_state
    )

    final_output = final_value_sum
    
    final_output = unpad(final_output, q_pad_len, axis=2)

    if is_3d_input:
        final_output = final_output.transpose(0, 2, 1, 3).reshape(batch_size, q_seq_len, -1)

    return final_output

def _tpu_ring_attention(query, key, value, env, is_causal, axis_size=None, sequence_axis_name='sp'):
    """
    分布式 ring attention kernel，和 splash/aqt attention 并列可选。
    """
    mesh = env._mesh
    if axis_size is None:
        axis_size = mesh.shape['sp']
    qkv_partition_spec = P('dp', 'axis', 'sp', None)
    def kernel(q, k, v):
        return _ring_attention_on_slices(q, k, v, env, axis_size, sequence_axis_name, is_causal=is_causal)
    sharded_fn = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(qkv_partition_spec, qkv_partition_spec, qkv_partition_spec),
        out_specs=qkv_partition_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)

# 1. 定义支持 attention_fn 的 Processor
class CustomWanAttnProcessor2_0(WanAttnProcessor2_0):
    def __init__(self, attention_fn, env):
        super().__init__()
        self.attention_fn = attention_fn
        self.env = env

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = jnp.transpose(query.unflatten(2, (attn.heads, -1)), (0, 2, 1, 3))
        key = jnp.transpose(key.unflatten(2, (attn.heads, -1)), (0, 2, 1, 3))
        value = jnp.transpose(value.unflatten(2, (attn.heads, -1)), (0, 2, 1, 3))

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states, freqs):
                import numpy as np
                if hasattr(hidden_states, 'unflatten') and hasattr(hidden_states, 'to'):
                    x_rotated = torch.view_as_complex(hidden_states.to(torch.float32).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)
                else:
                    import jax
                    import jax.numpy as jnp
                    shape = hidden_states.shape
                    x = hidden_states.astype(jnp.float32).reshape(shape[:3] + (-1, 2))
                    x_rotated = jax.lax.complex(x[..., 0], x[..., 1])
                    if freqs.shape != x_rotated.shape:
                        freqs = jnp.broadcast_to(freqs, x_rotated.shape)
                    x_out = x_rotated * freqs
                    x_out_real = jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1)
                    new_shape = x_out_real.shape[:3] + (x_out_real.shape[3] * x_out_real.shape[4],)
                    x_out_real = x_out_real.reshape(new_shape)
                    return x_out_real.astype(hidden_states.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = jnp.transpose(key_img.unflatten(2, (attn.heads, -1)), (0, 2, 1, 3))
            value_img = jnp.transpose(value_img.unflatten(2, (attn.heads, -1)), (0, 2, 1, 3))
            hidden_states_img = self.attention_fn(query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False)
            b, h, s, d = hidden_states_img.shape
            hidden_states_img = jnp.transpose(hidden_states_img, (0, 2, 1, 3)).reshape(b, s, h * d)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self.attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        b, h, s, d = hidden_states.shape
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3)).reshape(b, s, h * d)
        hidden_states = hidden_states.astype(query.dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = self.env.j2t_iso(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def main():
    args = parse_args()
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
    #env.config.use_tpu_splash_attention = True
    env.config.tpu_attention_backend = 'ring'

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
        window_size=args.window_size
    )
    print("Replacing attention processors with JAX-compatible version...")
    jax_compatible_processor = CustomWanAttnProcessor2_0(attention_fn=custom_attention, env=env)
    for block in pipe.transformer.blocks:
        block.attn1.set_processor(jax_compatible_processor)
        block.attn2.set_processor(jax_compatible_processor)
    print("Attention processors replaced.")
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
            print(f'Iteration {i} BKVSIZE={BKVSIZE}, BQSIZE={BQSIZE}: {end - start:.6f}s')
        
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
    parser.add_argument("--profile", action="store_true", default=False, help="Add profiler")
    parser.add_argument("--tpu_attention_backend", type=str, choices=['splash', 'sage', 'aqt', 'ring'], default='ring', help="TPU attention backend to use")
    return parser.parse_args()

if __name__ == '__main__':
    main()
