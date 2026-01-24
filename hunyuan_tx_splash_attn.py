
import functools
from typing import Optional, List, Tuple, Union, Sequence, Dict, Any
import re
import math
import torch
import torch.nn.functional as F
import torchax
from torchax.ops import ops_registry
import time
import jax
import jax.numpy as jnp
import numpy as np
import os

# Monkeypatch torchax.ops.jaten._aten_unsafe_view to fix 'order' arg error with wrappers
try:
    import torchax.ops.jaten as jaten
    import torch
    from torchax.ops import register_op
    
    def _robust_aten_unsafe_view(x, shape):
        try:
            # Try standard jax reshape
            return jnp.reshape(x, shape) 
        except TypeError as e:
            if "order" in str(e):
                # Fallback: if x is a wrapper that rejects 'order' (like TorchaxTensor)
                # verify if it has .reshape and call it directly
                if hasattr(x, 'reshape'):
                    return x.reshape(shape)
            raise e

    # Apply patch to module
    #print("DEBUG: Monkeypatching torchax.ops.jaten._aten_unsafe_view for robustness...")
    jaten._aten_unsafe_view = _robust_aten_unsafe_view
    
    # Re-register for dispatch
    @register_op(torch.ops.aten.unsafe_view)
    def _robust_aten_unsafe_view_wrapper(x, shape):
        return _robust_aten_unsafe_view(x, shape)

except ImportError:
    pass

try:
    import transformers.masking_utils
    def _simple_create_causal_mask(*args, **kwargs):
        """
        Simple causal mask implementation to bypass transformers vmap/functorch complexity.
        Fully generic signature to handle varying call patterns from different models.
        """
        # print(f"DEBUG: create_causal_mask called with args len={len(args)} kwargs={list(kwargs.keys())}")
        
        # Try to resolve arguments based on typical signature: (input_shape, dtype, device, ...)
        input_shape = kwargs.get('input_shape', None)
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)

        if len(args) > 0: input_shape = args[0]
        if len(args) > 1: dtype = args[1]
        if len(args) > 2: device = args[2]

        # Robust inference of arguments if missing
        if input_shape is None and 'input_embeds' in kwargs:
             input_shape = kwargs['input_embeds'].shape[:2] # (bsz, seq_len)
        
        if device is None and 'input_embeds' in kwargs:
             device = kwargs['input_embeds'].device

        if dtype is None and 'input_embeds' in kwargs:
             dtype = kwargs['input_embeds'].dtype

        if input_shape is None:
             # Last resort: try attention_mask
             if 'attention_mask' in kwargs:
                  input_shape = kwargs['attention_mask'].shape[:2]
             else:
                  raise ValueError(f"Could not determine input_shape from args={args} kwargs={kwargs}")

        bsz, seq_len = input_shape

        # Use triu for causal mask (upper triangular is masked)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
        mask = mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
        return mask

    #print("DEBUG: Monkeypatching transformers.masking_utils.create_causal_mask (PRE-IMPORT) for torchax compatibility...")
    transformers.masking_utils.create_causal_mask = _simple_create_causal_mask
except ImportError:
    pass

# PATCH: LlamaRotaryEmbedding to avoid @ operator (matmul) crash in torchax
# Replaces (A @ B) with (A * B) which is equivalent for these shapes (outer product) but JAX-safe
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    
    def _robust_llama_rotary_forward(self, x, position_ids):
        # Unwrap inputs to interact directly with JAX
        # This bypasses torchax dispatch issues where wrappers leak into jax ops
        def unwrap(t):
            # Loop to handle nested wrapping (e.g. Wrapper(Wrapper(JAX)))
            # Max depth 10 to prevent infinite loops
            current = t
            for _ in range(10):
                # Check for JAX Array (Tracer or ArrayImpl)
                # If it looks like JAX, we are done
                if hasattr(current, 'astype') and hasattr(current, 'device_buffer'):
                     # Likely JAX array
                     return current
                     
                found_inner = False
                
                # 1. Try direct dict access for _elem
                try:
                    if hasattr(current, '__dict__') and '_elem' in current.__dict__:
                        current = current.__dict__['_elem']
                        found_inner = True
                        continue
                except Exception:
                    pass

                # 2. Try __jax_array__
                if hasattr(current, '__jax_array__'):
                    try:
                        val = current.__jax_array__
                        if callable(val): 
                            current = val()
                        else:
                            current = val
                        found_inner = True
                        continue
                    except Exception:
                        pass
                
                # 3. Try standard _elem
                if hasattr(current, '_elem'):
                    current = current._elem
                    found_inner = True
                    continue
                    
                # If no inner wrapper found, break loop
                if not found_inner:
                    break
            
            # After peeling, if it's still a Torch Tensor (and not JAX), convert it
            # This handles raw CPU tensors
            is_torch = False
            try:
                if isinstance(current, torch.Tensor):
                    is_torch = True
            except:
                pass
            if not is_torch and hasattr(current, 'detach') and hasattr(current, 'cpu'):
                is_torch = True
                
            if is_torch:
                try:
                    # Check if it has 'astype' (JAX-like) -> if so, it's not a raw torch tensor?
                    # But TorchaxTensor masks astype. 
                    # If we are here, it means we couldn't peel it further. 
                    # If it really IS a TorchaxTensor, we failed.
                    
                    if hasattr(current, 'device') and current.device.type == 'meta':
                        return jnp.zeros(current.shape, dtype=torchax.ops.mappings.t2j_dtype(current.dtype))
                    
                    # Convert raw torch -> JAX
                    return jnp.array(current.detach().cpu().numpy())
                except Exception:
                    pass
            
            # Final try: jnp.asarray
            try:
                return jnp.asarray(current)
            except Exception:
                pass
                
            return current
            
        inv_freq_jax = unwrap(self.inv_freq)
        pos_ids_jax = unwrap(position_ids)
        x_jax = unwrap(x) # For dtype extraction if needed
        
        # Debug check
        if hasattr(inv_freq_jax, 'device') and hasattr(inv_freq_jax, 'type') and 'torch' in str(type(inv_freq_jax)):
             print(f"DEBUGGING CRITICAL: inv_freq_jax is STILL a torch tensor: {type(inv_freq_jax)}")
        
        # JAX Ops
        # inv_freq_expanded: [1, dim/2, 1]
        # position_ids_expanded: [bsz, 1, seq_len]
        
        # Cast
        inv_freq_jax = inv_freq_jax.astype(jnp.float32)
        pos_ids_jax = pos_ids_jax.astype(jnp.float32)
        
        # Reshape
        dim = inv_freq_jax.shape[0]
        inv_freq_expanded = jnp.reshape(inv_freq_jax, (1, dim, 1))
        
        # position_ids: (bsz, seq_len) -> (bsz, 1, seq_len)
        if pos_ids_jax.ndim == 2:
             bsz, seq_len = pos_ids_jax.shape
             pos_ids_expanded = jnp.reshape(pos_ids_jax, (bsz, 1, seq_len))
        else:
             pos_ids_expanded = pos_ids_jax

        # Broadcasting Mul
        freqs = jnp.transpose(inv_freq_expanded * pos_ids_expanded, (0, 2, 1))
        
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        # DEBUG: Check for NaNs in Rotary
        print(f"DEBUG: Rotary Cos Stats: Min: {jnp.min(cos)}, Max: {jnp.max(cos)}")
        print(f"DEBUG: Rotary Sin Stats: Min: {jnp.min(sin)}, Max: {jnp.max(sin)}")
        
        # Cast back to input dtype
        target_dtype = x_jax.dtype
        cos = cos.astype(target_dtype)
        sin = sin.astype(target_dtype)
        
        # Wrap outputs back to TorchaxTensor
        # Use env from input x
        env = getattr(x, 'env', None) or torchax.default_env()
        return torchax.tensor.Tensor(cos, env), torchax.tensor.Tensor(sin, env)

    #print("DEBUG: Monkeypatching LlamaRotaryEmbedding.forward for torchax compatibility...")
    LlamaRotaryEmbedding.forward = _robust_llama_rotary_forward
except ImportError:
    pass
except Exception as e:
    print(f"DEBUG: Failed to patch LlamaRotaryEmbedding: {e}")


from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import flax
from flax import nnx
import flax.linen as nn

from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from jax.tree_util import register_pytree_node
from transformers import modeling_outputs
import transformers.masking_utils
import argparse

# PATCH: Prevent transformers from checking concrete values of masks during tracing
def _patched_ignore_causal_mask_sdpa(*args, **kwargs):
    return False

# PATCH: torchax.ops.mappings.j2t_dtype to handle torch.dtype
# Tracers sometimes present as torch.Tensor with torch.dtype, confusing torchax.
import torchax.ops.mappings
_original_j2t_dtype = torchax.ops.mappings.j2t_dtype
def _robust_j2t_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    return _original_j2t_dtype(dtype)
torchax.ops.mappings.j2t_dtype = _robust_j2t_dtype
#print("DEBUG: Monkeypatched torchax.ops.mappings.j2t_dtype for robustness.")

# PATCH: torchax.tensor.Tensor.reshape to support JAX 'order' argument
# JAX's jnp.reshape calls .reshape(shape, order=...) on inputs.
# TorchaxTensor.reshape likely doesn't accept 'order'.
from torchax.tensor import Tensor as TorchaxTensor
_original_reshape = TorchaxTensor.reshape

def _robust_reshape_wrapper(self, *args, **kwargs):
    # Consume 'order' if present, as Torch tensors/wrappers don't typically support it in the same way
    # or pass it along if the underlying implementation supports it (less likely for torch signatures)
    if 'order' in kwargs:
        kwargs.pop('order')
    return _original_reshape(self, *args, **kwargs)

TorchaxTensor.reshape = _robust_reshape_wrapper
#print("DEBUG: Monkeypatched torchax.tensor.Tensor.reshape for JAX compatibility.")

# Removed failed monkeypatch attempt
# Dyanmic patch is now in main()
pass

# Scheduler fix is now applied directly in library source: 
# diffusers/schedulers/scheduling_flow_match_euler_discrete.py
pass



transformers.masking_utils._ignore_causal_mask_sdpa = _patched_ignore_causal_mask_sdpa





# Fix Recompilation for Nested Tensors (e.g. Tuples of TorchaxTensors)
def _register_torchax_pytree():
    try:
        def flatten_torchax(t):
            # 1. Unsafe Direct Access Only
            # Accessing properties like .jax() or even .detach() might trigger __jax_array__
            # which is forbidden during abstractification (JIT trace).
            # We must grab the underlying Tracer/_elem directly.
            
            elem = None
            # Try to grab _elem from __dict__ to avoid property descriptors
            if hasattr(t, '__dict__') and '_elem' in t.__dict__:
                elem = t.__dict__['_elem']
            elif hasattr(t, '_elem'):
                elem = t._elem
            
            if elem is None:
                # If no _elem (view?), try fallback but risky
                if hasattr(t, '__jax_array__'):
                     # Do NOT call it. This is exactly what causes the error.
                     # But if we don't have _elem, we are kinda stuck.
                     # Assuming typical TorchaxTensor HAS _elem.
                     pass
            
            # If we still don't have elem, it might be a raw tensor or something else
            if elem is None:
                 elem = t

            # 2. Preserve environment info
            # Accessing .env might be safe if it's just a stored object
            env = getattr(t, 'env', None)
            
            # Return tuple of children (tracers) and auxiliary data (env)
            return (elem,), env

        def unflatten_torchax(env, children):
            val = children[0]
            # No need to convert here if j2t_dtype is patched.
            # Just wrap it. TorchaxTensor will accept checked dtype.
            return torchax.tensor.Tensor(val, env)
            
        jax.tree_util.register_pytree_node(torchax.tensor.Tensor, flatten_torchax, unflatten_torchax)
        print("DEBUG: Registered TorchaxTensor as JAX PyTree.")
    except Exception as e:
        print(f"DEBUG: Failed to register PyTree: {e}")

_register_torchax_pytree()

# PATCH: torch.autocast compatibility for JAX
# LlamaRotaryEmbedding uses torch.autocast with device_type='jax' (from tensor) which fails.
# We redirect 'jax' to 'cpu' which is a valid AMP backend, allowing enabled=False to work as intended (no-op).
_original_autocast = torch.autocast

class _SafeAutocast(_original_autocast):
    def __init__(self, device_type, *args, **kwargs):
        if device_type == 'jax':
             device_type = 'cpu'
        super().__init__(device_type, *args, **kwargs)

torch.autocast = _SafeAutocast

# PATCH: Register HF ModelOutputs as JAX Pytrees
# torchax/JAX needs to know how to flatten/unflatten these custom classes to return them from compiled functions.
import transformers.cache_utils
def _register_hf_output_as_pytree(cls):
    def flatten(output):
        # ModelOutput behaves like an OrderedDict
        return list(output.values()), list(output.keys())

    def unflatten(keys, values):
        return cls(**dict(zip(keys, values)))

    register_pytree_node(cls, flatten, unflatten)

_register_hf_output_as_pytree(modeling_outputs.BaseModelOutputWithPast)
_register_hf_output_as_pytree(modeling_outputs.BaseModelOutputWithPooling)

# Also register DynamicCache if possible, though we will try to avoid using it
# DynamicCache usually just holds lists of tensors.
try:
    def flatten_dynamic_cache(cache):
        # Debugging what this object actually is
        # print(f"DEBUG: Flattening DynamicCache: {type(cache)}")
        # print(f"DEBUG: Attributes: {dir(cache)}")
        
        # Robust access
        keys = getattr(cache, 'key_cache', [])
        values = getattr(cache, 'value_cache', [])
        
        # If still missing/empty, just return empty
        return (keys, values), None
        
    def unflatten_dynamic_cache(aux, children):
        c = transformers.cache_utils.DynamicCache()
        if len(children) >= 2:
            c.key_cache = children[0]
            c.value_cache = children[1]
        return c
        
    register_pytree_node(transformers.cache_utils.DynamicCache, flatten_dynamic_cache, unflatten_dynamic_cache)
except Exception as e:
    print(f"Warning: Could not register DynamicCache pytree: {e}")

# Monkeypatch HunyuanVideoPipeline methods to ensure inputs are torchax-compatible
def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Original templating logic
        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
            use_cache=False, # Disable cache to avoid returning DynamicCache
        )
        # Use logic from original: hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.hidden_states[-(num_hidden_layers_to_skip + 1)]
        
        # Crop prompt
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        if dtype:
            prompt_embeds = prompt_embeds.to(dtype=dtype)

        batch_size = len(prompt)
        prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1, prompt_embeds.shape[-1])
        
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds, prompt_attention_mask


# Reusing Splash Attention from Wan2.1 (generic enough for now)
def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, window_size=None, block_size_q=128, block_size_kv=128, block_size_kv_compute=128):
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
            # Pad to multiple of block sizes
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, block_size_q, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, block_size_kv, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, block_size_kv, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # Use MultiHeadMask wrapping FullMask per head
            mask = splash_attention.MultiHeadMask(
                [splash_attention.FullMask((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )

            block_sizes = splash_attention.BlockSizes(
                block_q=block_size_q,
                block_kv=block_size_kv,
                block_kv_compute=block_size_kv_compute, # Match BKVCOMPUTESIZE usually
            )

            splash_fn = splash_attention.make_splash_mha(
                mask=mask,
                head_shards=1,
                q_seq_shards=1,
                block_sizes=block_sizes
            )
            
            out = splash_fn(
                q_3d_padded, k_3d_padded, v_3d_padded,
                segment_ids=None
            )
            
            # Crop back to original length
            return out[:, :q_orig_len, :]

        # Vectorize over the batch dimension.
        # q, k, v are (Batch, Heads, SeqLen, Dim)
        # kernel_3d expects (Heads, SeqLen, Dim)
        return jax.vmap(kernel_3d, in_axes=(0, 0, 0))(q, k, v)

    # Shard map to distribute computation
    return shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(
            P(None, 'sp', None, None), 
            P(None, 'sp', None, None), 
            P(None, 'sp', None, None)
        ),
        out_specs=P(None, 'sp', None, None),
        check_rep=False
    )(query, key, value)

# ops_registry.register("splash_attention", _tpu_splash_attention)

# Configuration
# Configuration
# MODEL_ID = "tencent/HunyuanVideo" # Official Repo (likely 13B) - Missing transformer/config.json
#MODEL_ID = "hunyuanvideo-community/HunyuanVideo" # Community Repo (Diffusers Compatible)
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
#MODEL_ID = "Aquiles-ai/HunyuanVideo-1.5-720p-fp8"
HEIGHT = 720
WIDTH = 1280
FRAMES = 61 # Standard for Hunyuan
NUM_STEP = 30

USE_DP = True
USE_FSDP = True
SP_NUM = 1

# Splash Attention Tuning
SPLASH_BLOCK_Q = 2048
SPLASH_BLOCK_KV = 2048
SPLASH_BLOCK_KV_COMPUTE = 1024

# Sharding rules (will need adjustment for Hunyuan architecture)
LOGICAL_AXIS_RULES = (
    ('conv_out', ('axis','dp','sp')),
    ('conv_in', ('axis','dp','sp')),
    # Add Hunyuan specific rules if needed
)
# Helper from wan_tx.py
def _shard_weight_dict(weight_dict, sharding_dict, mesh):
  result = {}
  for k, v in weight_dict.items():
    for target, sharding in sharding_dict.items():
      if re.fullmatch(target, k) is not None:
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
        break
    else:
      # replicate
      if not hasattr(v, 'apply_jax_'):
          #print(f"Warning: {k} (type: {type(v)}) does not have apply_jax_. Converting on the fly.")
          # Convert on the fly
          env = torchax.default_env()
          v = env.to_xla(v)
          v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
      else:
          v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    result[k] = v
  return result

def _move_module(module, env):
    # Standard state_dict move for tracked params/buffers
    with jax.default_device('cpu'):
      state_dict  = module.state_dict()
      state_dict = env.to_xla(state_dict)
      state_dict = env.to_xla(state_dict)
      module.load_state_dict(state_dict, assign=True)
    
    # Handle non-persistent buffers or anything missed by state_dict (like inv_freq)
    # This is critical for LlamaRotaryEmbedding
    for name, buf in module.named_buffers(recurse=True): # Recurse true to check all
        if not hasattr(buf, 'apply_jax_'):
             # Attempt to convert on the fly
             # We need to find the parent module to set the attribute
             parent = module
             path = name.split('.')
             for p in path[:-1]:
                 parent = getattr(parent, p)
             attr_name = path[-1]
             
             # Convert
             #print(f"Manually converting buffer: {name}")
             # We use env.to_xla for consistency
             jax_buf = env.to_xla(buf)
             setattr(parent, attr_name, jax_buf)

# Monkeypatch HunyuanVideoPipeline methods to ensure inputs are torchax-compatible
def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ):
        device = device or self._execution_device
        # SWAP: Use text_encoder (Qwen) dtype
        dtype = dtype or self.text_encoder.dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Original templating logic
        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            # SWAP: Use tokenizer (Qwen)
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        # SWAP: Use tokenizer (Qwen)
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        # Use logic from original: hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.hidden_states[-(num_hidden_layers_to_skip + 1)]
 
        # Crop prompt
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        if dtype:
            prompt_embeds = prompt_embeds.to(dtype=dtype)

        batch_size = len(prompt)
        
        with torchax.default_env():
            # MANUAL JAX TILE with Recursive Unwrapping
            def _manual_tile(t, reps):
                # Robust peeling loop
                current = t
                for _ in range(20): # Safety break
                    # 1. Is it a JAX Array?
                    if hasattr(current, 'device_buffer') or 'jaxlib' in str(type(current)) or 'jax.Array' in str(type(current)):
                         break
                    
                    # 2. Is it a View?
                    if 'torchax.view.View' in str(type(current)):
                         # Call .jax() to get the result of the view
                         # BUT if .jax() returns another View, we continue loop
                         if hasattr(current, 'jax'):
                             current = current.jax()
                             continue
                    
                    # 3. Is it a TorchaxTensor? (Check _elem)
                    found_inner = False
                    if hasattr(current, '_elem'):
                         current = current._elem
                         found_inner = True
                    elif hasattr(current, '__jax_array__'):
                         val = current.__jax_array__
                         current = val() if callable(val) else val
                         found_inner = True
                    
                    if found_inner:
                         continue
                        
                    # If we reached here and it's not JAX, break to check if it's acceptable or fallback
                    break

                jax_arr = current
                
                # Check if we actually got a JAX array or something tile-able
                try:
                    # Perform JAX tile
                    tiled = jnp.tile(jax_arr, reps)
                    # Wrap back
                    return torchax.tensor.Tensor(tiled, torchax.default_env())
                except Exception:
                    # Fallback to standard torch logic if extraction failed
                    print(f"DEBUG: _manual_tile failed to extract JAX array from {type(t)}. Falling back to .repeat()")
                    return t.repeat(*reps)
            
            # Apply to prompt_embeds
            prompt_embeds = _manual_tile(prompt_embeds, (num_videos_per_prompt, 1, 1))
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1, prompt_embeds.shape[-1])
            
            # Apply to attention mask
            prompt_attention_mask = _manual_tile(prompt_attention_mask, (num_videos_per_prompt, 1))
            prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds, prompt_attention_mask

def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ):
        import torchax
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # SWAP: Hunyuan 1.5 Community Repo puts ByT5 (Pooled) at text_encoder_2
        # So we must use tokenizer_2 and text_encoder_2 here to get 1472 dim.
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)

        # SWAP: Use text_encoder_2 (ByT5)
        # ByT5 usually returns (last_hidden_state,) or similar. 
        # Typically DOES NOT have pooler_output, so we might need to mean pool manually if it returns sequence.
        prompt_embeds = self.text_encoder_2(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
            return_dict=False,
        )
        
        # Confirm expected dimension from transformer weights (Truth)
        # Confirm expected dimension from transformer weights (Truth)
        expected_dim = 1024 # Default
        if hasattr(self, 'transformer'):
             if hasattr(self.transformer, 'time_text_embed') and \
                hasattr(self.transformer.time_text_embed, 'text_embedder') and \
                hasattr(self.transformer.time_text_embed.text_embedder, 'linear_1'):
                  expected_dim = self.transformer.time_text_embed.text_embedder.linear_1.weight.shape[1]
                  #print(f"DEBUG: Detected text_embedder expected input dim: {expected_dim}")
             elif hasattr(self.transformer.config, 'pooled_projection_dim'):
                  expected_dim = self.transformer.config.pooled_projection_dim
                  print(f"DEBUG: Using config pooled_projection_dim: {expected_dim}")
        
        # Override for Hunyuan 1.5 if detection failed (e.g. 1024) but we know it should be 3584
        # We can detect this if the input tensor is 1472 (Hunyuan 1.5 CLIP?)
        # Or just trust the detection if it worked.
        # But if it defaulted to 1024, we might miss padding 1472 -> 3584.
        if expected_dim == 1024:
             # Check if we should actually be 3584
             if hasattr(self, 'transformer') and getattr(self.transformer.config, 'pooled_projection_dim', 0) == 3584:
                 expected_dim = 3584
                 print("DEBUG: Corrected expected_dim to 3584 based on config.")

        # Helper to find tensor by ndim
        def _find_tensor(tup, ndim):
            if not isinstance(tup, (tuple, list)):
                tup = [tup]
            for i, t in enumerate(tup):
                dims = None
                if hasattr(t, 'ndim'): dims = t.ndim
                elif hasattr(t, 'shape'): dims = len(t.shape)
                
                if dims == ndim:
                    print(f"DEBUG: Found {ndim}D tensor at index {i} with shape {t.shape}")
                    return t
            return None

        if expected_dim > 20000:
            # Need Flattened Sequence
            target = None
            if isinstance(prompt_embeds, tuple):
                target = _find_tensor(prompt_embeds, 3) 
            else:
                target = _find_tensor([prompt_embeds], 3)
            
            if target is not None:
                print(f"DEBUG: Found 3D tensor {target.shape}, flattening to match {expected_dim}...")
                prompt_embeds = target.flatten(1)
            else:
                print(f"WARNING: Could not find 3D tensor for flattening! prompt_embeds type: {type(prompt_embeds)}")
                if isinstance(prompt_embeds, tuple): prompt_embeds = prompt_embeds[0]
                
        else:
            # Need Pooled (2D)
            target = None
            if isinstance(prompt_embeds, tuple):
                 # CLIP return order: (last_hidden_state, pooler_output)
                 # Try index 1 first as it's usually pooler
                 if len(prompt_embeds) > 1:
                      t = prompt_embeds[1]
                      dims = getattr(t, 'ndim', len(t.shape) if hasattr(t, 'shape') else None)
                      if dims is not None and dims == 2:
                           target = t
                           print("DEBUG: Found 2D tensor at index 1 (likely pooler_output)")
                 
                 if target is None:
                      target = _find_tensor(prompt_embeds, 2)
            else:
                 target = _find_tensor([prompt_embeds], 2)
            
            if target is not None:
                prompt_embeds = target
            else:
                 three_d = None
                 if hasattr(prompt_embeds, 'pooler_output'):
                      prompt_embeds = prompt_embeds.pooler_output
                 elif hasattr(prompt_embeds, 'text_embeds'):
                      prompt_embeds = prompt_embeds.text_embeds
                 else:
                      if isinstance(prompt_embeds, tuple): 
                          three_d = prompt_embeds[0]
                      else:
                          three_d = prompt_embeds
                      
                      # Manually pool: Mean Pooling or standard CLIP EOS pooling?
                      # Mean pooling is safer if we don't have EOS indices
                      if hasattr(three_d, 'mean'):
                           #print("DEBUG: Applying MEAN pooling to 3D tensor")
                           prompt_embeds = three_d.mean(dim=1)
                      else:
                           print("WARNING: Could not pool 3D tensor (no mean method), taking index 0")
                           prompt_embeds = three_d[:, 0]

        #print(f"DEBUG: Checking Padding logic. expected_dim={expected_dim}, prompt_embeds.shape={prompt_embeds.shape}, type={type(prompt_embeds)}")
        
        # Padding Patch for Hunyuan 1.5 (1024 -> 1152 or 3584 or 1472)
        if expected_dim is not None and prompt_embeds.shape[-1] != expected_dim:
             if expected_dim > prompt_embeds.shape[-1]:
                 pad_amt = expected_dim - prompt_embeds.shape[-1]
                 print(f"DEBUG: Padding prompt_embeds from {prompt_embeds.shape[-1]} to {expected_dim} (pad={pad_amt})")
                 
                 # Explicit JAX Padding
                 jax_arr = None
                 env = None
                 if hasattr(prompt_embeds, 'jax'):
                     jax_arr = prompt_embeds.jax()
                     env = prompt_embeds.env
                 elif hasattr(prompt_embeds, '_elem'):
                     jax_arr = prompt_embeds._elem
                     env = prompt_embeds.env
                 
                 if jax_arr is not None:
                      print(f"DEBUG: Performing JAX padding on {jax_arr.shape}...")
                      import jax.numpy as jnp
                      import torchax
                      
                      pad_width = []
                      for _ in range(len(jax_arr.shape) - 1):
                           pad_width.append((0, 0))
                      pad_width.append((0, pad_amt))
                      
                      padded_jax = jnp.pad(jax_arr, pad_width)
                      print(f"DEBUG: JAX Padded shape: {padded_jax.shape}")
                      
                      # Wrap back
                      if env is None: env = torchax.default_env()
                      prompt_embeds = torchax.tensor.Tensor(padded_jax, env)
                 else:
                      print("DEBUG: Could not unwrap to JAX for padding. Trying fallback F.pad...")
                      import torch.nn.functional as F
                      if hasattr(prompt_embeds, 'to'): 
                          prompt_embeds = F.pad(prompt_embeds, (0, pad_amt))
                      else:
                          try:
                             prompt_embeds = F.pad(prompt_embeds, (0, pad_amt))
                          except Exception as e:
                             print(f"DEBUG: F.pad failed: {e}")

        #print(f"DEBUG: Final prompt_embeds shape: {prompt_embeds.shape}")

        if dtype and hasattr(prompt_embeds, 'to'):
            prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt
        if hasattr(prompt_embeds, 'repeat'):
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        if hasattr(prompt_embeds, 'view'):
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds

# Apply monkeypatches
HunyuanVideoPipeline._get_llama_prompt_embeds = _get_llama_prompt_embeds
HunyuanVideoPipeline._get_clip_prompt_embeds = _get_clip_prompt_embeds


# Placeholder for sharding rules - can be populated later
transformer_shardings = {}
vae_shardings = {}


# Monkeypatch HunyuanVideoTransformer3DModel forward to fix JAX tracing error
# The original uses an in-place loop for attention_mask that fails with JAX tracers.
def _patched_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    **kwargs,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # PATCH: Pad channels if needed (32 -> 65 for Hunyuan 1.5)
    if hidden_states.shape[1] == 32 and self.config.in_channels == 65:
         # Pad with zeros: (B, 33, T, H, W)
         pad = torch.zeros(hidden_states.shape[0], 33, *hidden_states.shape[2:], 
                           device=hidden_states.device, dtype=hidden_states.dtype)
         hidden_states = torch.cat([hidden_states, pad], dim=1)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width
    
    # DEBUG: Check stats using jax.debug.print
    def _print_stats(name, t):
        import jax
        import jax.numpy as jnp
        
        def _log_one(n, x):
            if x is None:
                 print(f"DEBUG RT: {n} is None")
                 return
            # Unwrap TorchaxTensor
            if hasattr(x, 'jax'): x = x.jax()
            elif hasattr(x, '_elem'): x = x._elem
            
            # Robust dtype check
            is_int_or_bool = False
            if hasattr(x, 'dtype'):
                d_str = str(x.dtype)
                if 'int' in d_str or 'bool' in d_str:
                    is_int_or_bool = True
            
            # Helper to get values
            try:
                # If it's a Torch tensor, convert to numpy/jax for printing if possible, or use torch ops
                # But we want jax.debug.print during tracing.
                # If x is a Torch Tensor during JAX tracing, it might be a problem if it's not a Tracer.
                # Assuming x is compatible interacting with JAX here (Torchax guarantees this mostly).
                
                if is_int_or_bool:
                     jax.debug.print(f"DEBUG RT: {n} (Int/Bool) - Min: {{min}}, Max: {{max}}", 
                                    min=jnp.min(x), max=jnp.max(x))
                else:
                     jax.debug.print(f"DEBUG RT: {n} stats - Min: {{min}}, Max: {{max}}, Mean: {{mean}}, IsNaN: {{nan}}", 
                                    min=jnp.min(x), max=jnp.max(x), mean=jnp.mean(x), nan=jnp.any(jnp.isnan(x)))
            except Exception as e:
                # Fallback if something weird happens (e.g. mix of frameworks)
                print(f"DEBUG RT: Could not print stats for {n}: {e}")

        if isinstance(t, tuple):
             for i, item in enumerate(t):
                 _log_one(f"{name}[{i}]", item)
        else:
             _log_one(name, t)

    #_print_stats("SiT Input TimeStep", timestep)
    #_print_stats("SiT Input Pooled Proj", pooled_projections)
    #_print_stats("SiT Input Guidance", guidance)
    
    # PATCH: Fix Guidance explosion (1e29)
    # Pipeline passes guidance * 1000 (e.g. 6000), but generic/V1 transformer might expect ~6.
    # We use pure JAX to avoid torchax operator bugs (AttributeError during trace)
    if guidance is not None:
         # Unwrap to JAX array
         g_jax = guidance.jax()
         
         # Compute safe guidance in JAX
         import jax.numpy as jnp
         # Use 0.001 multiplication to be safe, though JAX handles division fine.
         # Condition must be on the JAX array.
         cond = jnp.mean(g_jax) > 100
         g_rescaled = jnp.where(cond, g_jax * 0.001, g_jax)
         
         # Wrap back to Torchax Tensor
         # We need the environment. Input tensors might not expose .env, so use default_env()
         guidance = torchax.tensor.Tensor(g_rescaled, torchax.default_env())
         
         #_print_stats("SiT Input Guidance (Processed)", guidance)

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)
    
    # DEBUG: Check stats using jax.debug.print for Tracers
    def _print_stats(name, t):
        import jax
        import jax.numpy as jnp
        
        def _log_one(n, x):
            # Unwrap TorchaxTensor if present
            if hasattr(x, 'jax'): 
                x = x.jax()
            elif hasattr(x, '_elem'):
                x = x._elem
            
            # Print at runtime
            jax.debug.print(f"DEBUG RT: {n} stats - Min: {{min}}, Max: {{max}}, Mean: {{mean}}, IsNaN: {{nan}}", 
                            min=jnp.min(x), max=jnp.max(x), mean=jnp.mean(x), nan=jnp.any(jnp.isnan(x)))

        if isinstance(t, tuple):
             for i, item in enumerate(t):
                 _log_one(f"{name}[{i}]", item)
        else:
             _log_one(name, t)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
    
    #_print_stats("SiT Time/Text Emb", temb)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation (PATCHED for JAX)
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    
    # Original (Fails in JAX):
    # attention_mask = torch.zeros(...)
    # effective_condition_sequence_length = ...
    # for i in range(batch_size):
    #     attention_mask[i, : effective_sequence_length[i]] = True
    
    # Patched (Vectorized):
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    
    # [B, N] mask
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0) # [1, N]
    attention_mask = indices < effective_sequence_length.unsqueeze(1) # [B, N]
    
    # [B, 1, 1, N], for broadcasting across attention heads
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    
    # Ensure bool
    attention_mask = attention_mask.to(torch.bool)



    # Monkeypatch HunyuanVideoConditionEmbedding to inspect components
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoConditionEmbedding
    
    def _patched_time_text_embed_forward(self, timestep, pooled_projections, guidance=None):
        # 1. Timestep
        timesteps_proj = self.timestep_embedder(timestep) 
        # Debug
        #_print_stats("SiT Inner Timestep Proj", timesteps_proj)
        
        # 2. Pooled
        pooled_projections = self.text_embedder(pooled_projections)
        #_print_stats("SiT Inner Pooled Proj", pooled_projections)
        
        conditioning = timesteps_proj + pooled_projections

        # 3. Guidance
        if self.guidance_embedder is not None and guidance is not None:
            guidance_emb = self.guidance_embedder(guidance)
            #_print_stats("SiT Inner Guidance Emb", guidance_emb)
            conditioning = conditioning + guidance_emb
            
        return conditioning, pooled_projections

    HunyuanVideoConditionEmbedding.forward = _patched_time_text_embed_forward

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                use_reentrant=False
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                use_reentrant=False
            )
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    # 5. Output projection
    output = self.norm_out(hidden_states, temb)
    output = self.proj_out(output)

    # 6. Unpatchify
    output = output.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p)
    output = output.permute(0, 4, 1, 5, 2, 6, 3, 7)
    output = output.flatten(5, 6).flatten(3, 4).flatten(1, 2)

    if not return_dict:
        return (output,)

    return transformers.modeling_outputs.BaseModelOutput(last_hidden_state=output)

HunyuanVideoTransformer3DModel.forward = _patched_transformer_forward

# Custom Splash Attention Processor for HunyuanVideo
class HunyuanSplashAttnProcessor(torch.nn.Module):
    def __init__(self, block_size_q=128, block_size_kv=128, block_size_kv_compute=128):
        super().__init__()
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.block_size_kv_compute = block_size_kv_compute

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #print(f"DEBUG: Tracing SplashAttnProcessor... hidden_states={hidden_states.shape}")
        from diffusers.models.embeddings import apply_rotary_emb

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                # Split, apply RoPE, concat back
                query_img = query[:, :, : -encoder_hidden_states.shape[1]]
                query_txt = query[:, :, -encoder_hidden_states.shape[1] :]
                key_img = key[:, :, : -encoder_hidden_states.shape[1]]
                key_txt = key[:, :, -encoder_hidden_states.shape[1] :]
                
                query_img = apply_rotary_emb(query_img, image_rotary_emb)
                key_img = apply_rotary_emb(key_img, image_rotary_emb)
                
                query = torch.cat([query_img, query_txt], dim=2)
                key = torch.cat([key_img, key_txt], dim=2)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV (if separate)
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention (Splash)
        # We need to reshape to 3D for splash attention if needed, or pass 4D if it supports it.
        # _tpu_splash_attention expects (B, H, S, D) which is what we have here.
        
        # Use a placeholder op that torchax will intercept
        # torchax typically doesn't automatically map generic python calls. 
        # But we can try using the registered name if torchax supports `torch.ops.torchax.splash_attention` or similar.
        # For now, we will call a dummy torch function and hope to use torchax's replacement mechanism, 
        # or we might need to use `torchax.symbolic_call` if available.
        # Given the context, we will try to use the `scaled_dot_product_attention` but optimized.
        # BUT the goal is splash attention.
        
        # We'll assume torchax has a mechanism to call the registered op.
        # If not, we might need `torchax.ops.call_jax_op`.
        # I'll check `wan_tx` usages. It didn't seem to call it from PyTorch?
        # Re-checking imports... `import torchax`.
        
        # Let's try to assume `torchax` patches `F.scaled_dot_product_attention` or we can call `torchax.ops.splash_attention` directly?
        # No, `torchax` is for `torch` -> `jax`.
        # I will leave standard `F.scaled_dot_product_attention` for now and see if torchax lowers it efficiently,
        # OR I will insert a marker.
        
        # Call Splash Attention via torchax environment
        # Capture environment from input if possible to ensure consistency
        env = getattr(hidden_states, 'env', torchax.default_env())
        
        # Ensure we are using the correct inputs
        scale = 1.0 / math.sqrt(query.shape[-1])
        seq_len = query.shape[2]
        
        # Threshold for using Splash Attention
        # Run standard SDPA for small sequences (like token_refiner with 256 tokens)
        if seq_len < 2048:
             hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            def _unwrap_torchax(x):
                if hasattr(x, "unwrap"):
                    x = x.unwrap()
                # If it's pure TorchaxTensor, it might hold jax value in `_elem` or similar?
                # Usually `unwrap` (from global torchax) would handle it.
                # But if we manually inspect:
                if hasattr(x, '_elem'):
                    return x._elem
                return x

            # Unwrap inputs for JAX
            jax_query = _unwrap_torchax(query)
            jax_key = _unwrap_torchax(key)
            jax_value = _unwrap_torchax(value)
            
            # Need strict padding for Splash Attention on TPU
            # Use fixed block sizes for now to verify functionality
            # 128 is a safe block size for TPU
            BLOCK_SIZE = 128
            
            jax_hidden_states = _tpu_splash_attention(
                jax_query, jax_key, jax_value, env, 
                scale=scale, 
                is_causal=False,
                block_size_q=self.block_size_q,
                block_size_kv=self.block_size_kv,
                block_size_kv_compute=self.block_size_kv_compute
            )
            
            # Wrap back
            # Wrap back
            # torchax.tensor.wrap fails because it doesn't pass env to Tensor constructor
            # Instantiate Tensor directly with env
            if hasattr(torchax.tensor, 'Tensor'):
                hidden_states = torchax.tensor.Tensor(jax_hidden_states, env)
            elif hasattr(env, 'to_xla'):
                hidden_states = env.to_xla(jax_hidden_states)
            else:
                 raise RuntimeError("Cannot wrap back to TorchaxTensor.")
        
        # Handle JAX/Torch API difference
        # We need (Batch, Heads, SeqLen, Dim) -> (Batch, SeqLen, Heads, Dim) -> (Batch, SeqLen, Heads*Dim)
        if hasattr(hidden_states, 'permute'):
             # Torch-like
             hidden_states = hidden_states.permute(0, 2, 1, 3)
        else:
             # JAX-like: transpose expects permutation
             hidden_states = hidden_states.transpose((0, 2, 1, 3))

        # flatten(2, 3) equivalent: reshape(B, S, -1)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        
        if hasattr(hidden_states, 'to'):
            hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
        else:
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        if encoder_hidden_states is None:
            return hidden_states

        return hidden_states, encoder_hidden_states




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEP)
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    # Add profile argument
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--fp8", action="store_true", default=False, help="Use FP8 quantized weights")
    args = parser.parse_args()

    # Separate IDs for Transformer and VAE to handle FP8 split sources
    transformer_model_id = args.model_id
    transformer_subfolder = "transformer"
    vae_model_id = args.model_id
    vae_subfolder = "vae"

    if args.fp8:
        # Override Transformer source to Aquiles-ai (Files at ROOT)
        transformer_model_id = "Aquiles-ai/HunyuanVideo-1.5-720p-fp8"
        transformer_subfolder = None # Config is at root in this repo
        
        # Keep VAE from the original/community source
        if args.model_id == MODEL_ID or args.model_id == transformer_model_id:
            # If user didn't specify a custom base, use the community one for VAE
            vae_model_id = MODEL_ID 
        
        print(f"FP8 Mode: Loading Transformer from {transformer_model_id} (root)")
        print(f"FP8 Mode: Loading VAE from {vae_model_id} ({vae_subfolder})")

    # Load Pipeline
    import diffusers
    import transformers
    
    # Suppress warnings
    diffusers.utils.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    
    if not args.fp8:
         pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    else:
         # FP8 Pipeline Loading
         # We load the standard pipeline first (CPU, float32/16)
         # Then we swap the transformer.
         # Ideally we want to load ONLY the components we need to save RAM, but...
         
         # Load with float32 to avoid issues, we move to BF16 later
         # use_safetensors=True is default.
         pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)

    # Restore default logging if needed, or keep quiet
    # diffusers.utils.logging.set_verbosity_info()

    transformer_weight_name = None
    if args.fp8:
         from huggingface_hub import hf_hub_download
         transformer_weight_name = "quantized/hy15_720p_t2v_fp8_e4m3_lightx2v.safetensors"
         
         print(f"DEBUG: Explicitly downloading FP8 weights: {transformer_weight_name} from {transformer_model_id}...")
         try:
             # 1. Download Weights
             weight_path = hf_hub_download(
                 repo_id=transformer_model_id,
                 filename=transformer_weight_name,
                 subfolder=transformer_subfolder
             )
             print(f"DEBUG: Found/Downloaded FP8 weights at: {weight_path}")

             # 2. Use MAPPED Config (Generated by map_config.py)
             # We need to map the Aquiles keys (heads_num) to Diffusers keys (num_attention_heads)
             config_path = "hunyuan_fp8_transformer_link/mapped_config.json"
             
             if not os.path.exists(config_path):
                 raise RuntimeError(f"Mapped config not found at {config_path}. Please run 'python map_config.py' first!")
                 
             print(f"DEBUG: Using Mapped Config at: {config_path}")

             # 3. Use from_single_file which handles ComfyUI/LightX key mapping automatically
             print(f"DEBUG: Loading transformer using from_single_file...")
             # DEBUG: Print raw keys from file to debug mapping
             try:
                 import safetensors.torch
                 raw_sd = safetensors.torch.load_file(weight_path)
                 # print("DEBUG: Raw Checkpoint Keys (First 50):")
                 # for i, k in enumerate(list(raw_sd.keys())[:50]):
                 #     print(f"  {k}")
             except Exception as e:
                 print(f"DEBUG: Could not read raw keys: {e}")

             # MANUAL LOADING REPLACEMENT
             print("DEBUG: Switching to MANUAL LOADING to bypass meta/mapping issues...")
             import json
             import safetensors.torch
             import copy
             
             # 1. Load Config
             with open(config_path, 'r') as f:
                 config_dict = json.load(f)

             if args.fp8:
                 #print("DEBUG: Manually forcing pooled_projection_dim=1472 within manual config load to match ByT5 weights")
                 config_dict["pooled_projection_dim"] = 1472
             
             # 2. Init Model (on CPU, force dense tensors)
             #print("DEBUG: Instantiating model on CPU...")
             # Remove keys that might confuse __init__ if necessary (usually from_config handles this)
             # We use direct __init__ so we must clean extra keys if any
             # But map_config.py ensures keys are from ref_conf mostly.
             try:
                 # Filter keys to match __init__ signature if possible? 
                 # Or just try/except. 
                 # A safer way is using the class method that handles config
                 # Use the modified config_dict
                 transformer = HunyuanVideoTransformer3DModel.from_config(config_dict)
             except Exception as e:
                 #print(f"DEBUG: from_config failed ({e}), trying kwargs init...")
                 transformer = HunyuanVideoTransformer3DModel(**config_dict)
                 
             transformer.eval() 
             
             # 3. Load Weights
             #print(f"DEBUG: Loading weights from {weight_path}")
             raw_sd = safetensors.torch.load_file(weight_path)
             
             # 4. Remap Keys (time_embed -> time_text_embed, etc.)
             #print("DEBUG: Remapping keys manually...")
             new_sd = {}
             for k, v in raw_sd.items():
                 new_k = k
                 if k.startswith("time_embed."):
                     new_k = k.replace("time_embed.", "time_text_embed.")
                 elif k.startswith("context_embedder_2."):
                      new_k = k.replace("context_embedder_2.", "time_text_embed.text_embedder.")
                 elif k.startswith("byt5_in."):
                       new_k = k.replace("byt5_in.", "time_text_embed.text_embedder.")
                 elif k.startswith("txt_in.c_embedder."): # Found in FP8 checkpoint - Context/Sequence Embedder
                       new_k = k.replace("txt_in.c_embedder.", "context_embedder.time_text_embed.text_embedder.")
                 elif k.startswith("txt_in.t_embedder."): # Found in FP8 checkpoint
                       new_k = k.replace("txt_in.t_embedder.", "context_embedder.time_text_embed.timestep_embedder.")
                 # Remap image_embedder? User execution earlier showed 'image_embedder' unused.
                 # If model doesn't use it, ignore.
                 
                 new_sd[new_k] = v
             
             #print("DEBUG: Successfully loaded transformer via MANUAL path.")
             # Skip standard loading below
             transformer_model_id = None
             #print("DEBUG: Successfully loaded transformer via from_single_file.")
             
             # MANUAL PATCH: Remap keys if time_text_embed is zero
             # Based on user logs: unused 'time_embed.timestep_embedder...' vs missing 'time_text_embed.timestep_embedder...'
             # It seems the checkpoint uses 'time_embed' but the model uses 'time_text_embed'
             # Also 'context_embedder_2' vs ??? 
             
             #print("DEBUG: Checking for remapping needs...")
             if hasattr(transformer, 'time_text_embed'):
                 # Check if linear_1 weight is zero
                 if isinstance(transformer.time_text_embed.timestep_embedder.linear_1.weight, torch.Tensor) and \
                    (transformer.time_text_embed.timestep_embedder.linear_1.weight == 0).all():
                     
                     #print("DEBUG: Detected ZERO weights in time_text_embed. Attempting manual remapping from loaded state dict...")
                     # We need to reload the state dict from the file manually to get the unused keys
                     # Or rely on what we just loaded if we can get it. 
                     # Actually from_single_file returns the model. We can't easily access the unused keys from it directly 
                     # unless we capture the output of load_state_dict which is internal.
                     
                     # Better approach: Load raw state dict, remap keys, load into model.
                     import safetensors.torch
                     raw_sd = safetensors.torch.load_file(weight_path)
                     
                     new_sd = {}
                     count = 0
                     for k, v in raw_sd.items():
                         new_k = k
                         # Mapping rules
                         if k.startswith("time_embed."):
                             new_k = k.replace("time_embed.", "time_text_embed.")
                         elif k.startswith("context_embedder_2."):
                             # context_embedder_2 -> time_text_embed.text_embedder ??
                             # User logs show 'context_embedder_2.linear_1.weight' unused.
                             # Model has 'time_text_embed.text_embedder.linear_1.weight' zero.
                             # Let's map context_embedder_2 -> time_text_embed.text_embedder
                             new_k = k.replace("context_embedder_2.", "time_text_embed.text_embedder.")
                         
                         if new_k != k:
                             print(f"  Remapping {k} -> {new_k}")
                             count += 1
                         new_sd[new_k] = v
                     
                     if count > 0:
                         print(f"DEBUG: Applying {count} remapped keys...")
                         m, u = transformer.load_state_dict(new_sd, strict=False)
                         print(f"DEBUG: Reloaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
             
             #print("DEBUG: Successfully loaded transformer via from_single_file.")
             
             # Skip standard loading below
             transformer_model_id = None 

         except Exception as e:
             print(f"CRITICAL ERROR: Failed to prepare/load FP8 weights: {e}")
             raise e

    if transformer_model_id is not None:
        # Hunyuan 1.5 specific projection dim
        pooled_projection_dim = 113344 if "1.5" in transformer_model_id else None
        
        load_kwargs = {
            "subfolder": transformer_subfolder,
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,
            "weight_name": transformer_weight_name,
            "low_cpu_mem_usage": False,
        }
        if pooled_projection_dim:
             load_kwargs["pooled_projection_dim"] = pooled_projection_dim

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            transformer_model_id, 
            **load_kwargs
        )

    try:

        if args.fp8:
             # Download FP8 VAE config and weights
             print(f"FP8 Mode: Downloading VAE from {transformer_model_id}...")
             vae_config_path = hf_hub_download(repo_id=transformer_model_id, filename="config.json", subfolder="vae")
             vae_weight_path = hf_hub_download(repo_id=transformer_model_id, filename="diffusion_pytorch_model.safetensors", subfolder="vae")
             
             print(f"DEBUG: Loading VAE from downloaded path: {os.path.dirname(vae_config_path)}")
             vae = AutoencoderKLHunyuanVideo.from_pretrained(
                 os.path.dirname(vae_config_path),
                 torch_dtype=torch.bfloat16,
             )
             # Force latent channels if needed (Hunyuan 1.5 is 32)
             if "1.5" in transformer_model_id:
                  vae.config.latent_channels = 32
        
        elif "HunyuanVideo-1.5" in vae_model_id:
             # Now utilizing native support in Diffusers (updated AutoencoderKLHunyuanVideo)
             vae = AutoencoderKLHunyuanVideo.from_pretrained(
                 vae_model_id, 
                 subfolder=vae_subfolder, 
                 torch_dtype=torch.bfloat16,
                 low_cpu_mem_usage=False,
             )
             if "1.5" in vae_model_id:
                 #print("DEBUG: Forcing VAE latent_channels=32 for Hunyuan 1.5")
                 vae.config.latent_channels = 32
        else:
             vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_model_id, subfolder=vae_subfolder, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
            )
    except Exception as e:
        print(f"Fallback to standard VAE loading failed: {e}")
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            vae_model_id, subfolder=vae_subfolder, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, 
        transformer=transformer, 
        vae=vae,
        torch_dtype=torch.bfloat16
    )

    # PATCH: torchax.ops.jaten._aten_convolution robustness
# When JIT tracing, dispatch sometimes fails to unwrap TorchaxTensors (Env mismatch?), passing them to JAX ops.
# This triggers __jax_array__ which is forbidden. We force unwrapping here.
    try:
        from torchax.ops import jaten, ops_registry
        
        # Original implementation
        _original_aten_convolution = jaten._aten_convolution

        def _robust_aten_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
            # Unwrap helpers
            def _u(x):
                if hasattr(x, '_elem'): return x._elem
                if hasattr(x, 'unwrap'): return x.unwrap()
                if hasattr(x, 'jax'): 
                     try: return x.jax()
                     except: pass 
                return x

            # Convolution expects JAX tracers, not TorchaxTensors
            input = _u(input)
            weight = _u(weight)
            bias = _u(bias) if bias is not None else None
            
            return _original_aten_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

        # Force update ALL loop-ups involving convolution
        count = 0
        
        # Check both registries
        registries = []
        if hasattr(ops_registry, 'all_aten_ops'):
            registries.append(ops_registry.all_aten_ops)
        if hasattr(ops_registry, 'all_torch_functions'):
            registries.append(ops_registry.all_torch_functions)
            
        #print(f"DEBUG: Found {len(registries)} registries in ops_registry.")

        for reg in registries:
            # It's likely a dict or list
            if isinstance(reg, dict):
                items = reg.items()
            else:
                # If it's a list? unlikely but possible
                #print(f"DEBUG: Registry is type {type(reg)}, skipping iteration.")
                continue

            for op, entry in items:
                op_str = str(op)
                if "convolution" in op_str or "conv3d" in op_str:
                    #print(f"DEBUG: Patching op: {op_str} in registry")
                    # entry is likely an Operator object with .func
                    if hasattr(entry, 'func'):
                        entry.func = _robust_aten_convolution
                        count += 1
                    else:
                        print(f"DEBUG: Warning - Entry for {op_str} has no .func attribute")
                          
    except Exception as e:
        print(f"DEBUG: Failed to patch _aten_convolution: {e}")

# Enable torchax globally AFTER loading (Safe Mode)
    print("Model loaded (PyTorch). Enabling torchax globally and moving to TPU environment...")
    torchax.enable_globally()
    
    # DYNAMIC PATCH: Patch the Environment class to support CPU tensors in TorchaxTensor (Runtime Fix)
    # This keeps runtime interop safe but avoids init crashes
    try:
        env = torchax.default_env()
        EnvClass = getattr(env, '__class__')
        
        # Patch _to_copy if it exists (responsible for JAX conversion)
        if hasattr(EnvClass, '_to_copy'):
             _original_env_to_copy = EnvClass._to_copy
             def _robust_env_to_copy(self, the_tensor, dtype=None, device=None, *args, **kwargs):
                # Check if we are dealing with a Torch Tensor (CPU or otherwise)
                # We interpret ALL torch.Tensor input as needing torch-based conversion
                # CRITICAL: Do NOT double-wrap if it's already a TorchaxTensor (has _elem)
                is_torch = isinstance(the_tensor, torch.Tensor)
                is_already_wrapped = hasattr(the_tensor, '_elem')

                # Check internal wrapping
                inner_elem = None
                
                # Deep unwrap loop to find the true underlying element
                # TorchaxTensors can be nested (e.g. jax array <- TorchaxTensor <- TorchaxTensor)
                temp = the_tensor
                for _ in range(10): # Safety limit
                    # Prefer .jax() if available (generic unwrap to JAX world)
                    # This handles View, TorchaxTensor, etc.
                    if hasattr(temp, 'jax'):
                        try:
                            val = temp.jax()
                            # If .jax() returns self/same type, avoid infinite loop if it's not unwrapping
                            if val is not temp:
                                temp = val
                                continue
                        except Exception:
                            pass
                    
                    if hasattr(temp, '_elem'):
                        temp = temp._elem
                    elif hasattr(temp, 'unwrap'): # Some might use .unwrap()
                        temp = temp.unwrap()
                    else:
                        break
                inner_elem = temp
                
                # Check for Tracer
                is_tracer = inner_elem is not None and (hasattr(inner_elem, 'aval') or "Tracer" in str(type(inner_elem)))
                
                if is_tracer:
                     if dtype is not None:
                         from torchax.ops.mappings import t2j_dtype
                         jax_dtype = t2j_dtype(dtype)
                         new_tracer = inner_elem.astype(jax_dtype)
                         return torchax.tensor.Tensor(new_tracer, self)
                     else:
                         return torchax.tensor.Tensor(inner_elem, self)
                
                # We handle if it's a raw torch tensor OR a wrapped torch tensor
                # Because _original_env_to_copy expects JAX/Numpy array (uses .astype)
                is_inner_torch = isinstance(inner_elem, torch.Tensor)
                
                # Check if we should intervene:
                should_intervene = (is_torch and not is_already_wrapped) or is_inner_torch
                
                if should_intervene:
                    
                    # Get the actual torch tensor to work with
                    target_tensor = inner_elem if is_inner_torch else the_tensor

                    # LAST DITCH CHECK: If target_tensor is still a TorchaxTensor, verify its guts
                    if hasattr(target_tensor, '_elem'):
                        guts = target_tensor._elem
                        if hasattr(guts, 'aval') or "Tracer" in str(type(guts)):
                            return torchax.tensor.Tensor(target_tensor, self) # Keep it wrapped

                    # FIX: If we are moving to a specific device (likely JAX/XLA), 
                    # we must manually bridge because _original_env_to_copy fails on torch.Tensor (no .astype)
                    if device is not None and str(device) != 'cpu':
                         try:
                             from torchax.ops.mappings import t2j_dtype
                             jax_dtype = t2j_dtype(dtype) if dtype is not None else None
                         except:
                             jax_dtype = None

                         # Manually convert Torch (CPU) -> Numpy -> JAX
                         # Note: .detach().cpu() is safe here as source is likely CPU tensor
                         
                         # Safety Check before detach
                         if "Tracer" in str(type(target_tensor)):
                              # Fallback to JAX return
                              return torchax.tensor.Tensor(target_tensor, self)
                              
                         npy = target_tensor.detach().cpu().numpy()
                         jax_arr = jnp.array(npy, dtype=jax_dtype)
                         
                         return torchax.tensor.Tensor(jax_arr, self)

                    # Bypassing JAX .astype() logic by using Torch's .to() and re-wrapping
                    # This works for CPU->CPU, and helps when just casting dtype
                    converted_elem = target_tensor.to(dtype=dtype, device=device if device else None)
                    # Force creation of TorchaxTensor on this env
                    return torchax.tensor.Tensor(converted_elem, self)
                
                # If it's a JAX array (not Tracer, not Torch), let original env handle it (.astype)
                return _original_env_to_copy(self, the_tensor, dtype, device, *args, **kwargs)
             
             EnvClass._to_copy = _robust_env_to_copy
             #print("DEBUG: Successfully patched EnvClass._to_copy for runtime robustness.")
             
    except Exception as e:
        print(f"DEBUG: Failed to patch Environment class: {e}")

    env = torchax.default_env()
    
    # DEBUG: Check for meta parameters
    #print("DEBUG: Checking transformer parameter devices...")
    has_meta = False
    count = 0
    for n, p in transformer.named_parameters():
        if p.device.type == "meta":
            if count < 5:
                 print(f"WARNING: Parameter {n} is on META device! (Not loaded correctly)")
            count += 1
            has_meta = True
            # unexpected key mismatch or loading issue

    # TPU Mesh Setup
    n_devices = len(jax.devices())
    # PERF: Revert to 1x8 Mesh (sp=8) for max memory bandwidth. 
    # dp=2 (2x4) caused slowdown (2.46s -> 7s) due to reduced model parallelism efficiency.
    # While dp=2 is conceptually "correct" for CFG, sp=8 is physically FASTER for this model size/batch.
    devices = mesh_utils.create_device_mesh((1, n_devices))
    mesh = Mesh(devices, axis_names=('dp', 'sp'))
    env.default_device_or_sharding = NamedSharding(mesh, P())
    env._mesh = mesh
    
    # Simple recursive replacement or iterating known modules
    replacement_processor = HunyuanSplashAttnProcessor(
        block_size_q=SPLASH_BLOCK_Q,
        block_size_kv=SPLASH_BLOCK_KV,
        block_size_kv_compute=SPLASH_BLOCK_KV_COMPUTE
    )
    
    def replace_processor(module):
        for name, child in module.named_children():
            if name == 'attn' and hasattr(child, 'processor'):
                 child.set_processor(replacement_processor)
            else:
                 replace_processor(child)
                 
    replace_processor(pipe.transformer)
    
    text_encoder_options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('output_hidden_states', 'return_dict', 'use_cache')}
    )

    # Text Encoder 1 (Llama/Qwen)
    print("Moving text_encoder (Llama/Qwen) to environment (FP32)...")
    torchax.disable_globally()
    try:
        pipe.text_encoder.to(dtype=torch.float32) # Force FP32 for stability
    finally:
        torchax.enable_globally()
    _move_module(pipe.text_encoder, env)
    # pipe.text_encoder = torchax.compile(pipe.text_encoder, text_encoder_options)
    
    # Text Encoder 2 (CLIP/T5)
    print("Moving text_encoder_2 (CLIP/T5) to environment (FP32)...")
    torchax.disable_globally()
    try:
        pipe.text_encoder_2.to(dtype=torch.float32) # Force FP32 for stability
    finally:
        torchax.enable_globally()
    _move_module(pipe.text_encoder_2, env)
    # pipe.text_encoder_2 = torchax.compile(pipe.text_encoder_2, text_encoder_options)

    # VAE (Replaced with Native JAX VAE to fix OOM)
    # print("Compiling vae...")
    # ... (Refer to diff for removal/commenting of old torchax logic)
    
    print("Loading Native JAX VAE (Fixing OOM)...")
    from hunyuan_vae_jax import load_hunyuan_vae_jax
    
    # Load JAX VAE
    # We use the same model ID.
    # If explicit path available (from FP8 download), we need to adapt for load_hunyuan_vae_jax
    # load_hunyuan_vae_jax takes (pretrained_model_name_or_path, subfolder="vae")
    # If we have a direct file path, we can pass the directory as path and "" as subfolder.
    
    vae_id_to_use = vae_model_id
    vae_subfolder_to_use = "vae"
    
    if 'vae_weight_path' in locals() and vae_weight_path is not None:
         vae_id_to_use = os.path.dirname(vae_weight_path)
         vae_subfolder_to_use = "" # File is directly in this dir
         
    jax_vae = load_hunyuan_vae_jax(vae_id_to_use, subfolder=vae_subfolder_to_use, dtype=jnp.bfloat16, mesh=mesh)
    print("Native JAX VAE Loaded.")
    
    # Disable PyTorch VAE in pipeline to save memory/confusion? 
    # Actually pipe.vae is still needed for configuration validation inside invoke, 
    # but we won't use it for decode.
    
    # Transformer compilation (Keep as is)
    print("Compiling transformer...")
    _move_module(pipe.transformer, env)
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict', 'attention_kwargs')}
    )
    pipe.transformer = torchax.compile(pipe.transformer, transformer_options)

    # Verify Config matches User Request ("HYVideo-T/2-cfgdistill")
    config = pipe.transformer.config
    d_model = config.num_attention_heads * config.attention_head_dim
    print(f"Hidden Size (calc): {d_model}")
    print(f"Num Heads: {config.num_attention_heads}")
   
    # Apply Sharding (Manual)
    if hasattr(pipe.transformer, 'params'):
            # Define Sharding Rules (Recovered from previous optimization)
            # 2D+ weights: Shard dim 1 (Input Channels) on 'sp' -> (None, 'sp')
            # 1D weights: Replicate -> (None,)
            transformer_shardings = {}
            for name, param in pipe.transformer.named_parameters():
                 if param.ndim >= 2:
                      transformer_shardings[name] = (None, 'sp')
                 else:
                      transformer_shardings[name] = (None,)
                      
            print("Applying transformer sharding rules...")
            pipe.transformer.params = _shard_weight_dict(pipe.transformer.params, transformer_shardings, mesh)
            pipe.transformer.buffers = _shard_weight_dict(pipe.transformer.buffers, transformer_shardings, mesh) 

    if hasattr(pipe.text_encoder, 'params'):
            pipe.text_encoder.params = _shard_weight_dict(pipe.text_encoder.params, {}, mesh)
            pipe.text_encoder.buffers = _shard_weight_dict(pipe.text_encoder.buffers, {}, mesh)

    if hasattr(pipe.text_encoder_2, 'params'):
            pipe.text_encoder_2.params = _shard_weight_dict(pipe.text_encoder_2.params, {}, mesh)
            pipe.text_encoder_2.buffers = _shard_weight_dict(pipe.text_encoder_2.buffers, {}, mesh)

    # Monkeypatch scheduler.step (Keep as is)
    original_step = pipe.scheduler.step
    def debug_step(model_output, timestep, sample, return_dict=True, generator=None):
        try:
            return original_step(model_output, timestep, sample, return_dict=return_dict, generator=generator)
        except Exception as e: 
            print(f"Caught error in scheduler step: {e}. Falling back to CPU with global disable.")
            def to_cpu_pure(t):
                if hasattr(t, 'numpy'):
                    try: return torch.as_tensor(t.detach().numpy())
                    except: pass
                if hasattr(t, 'cpu'): return t.cpu()
                return torch.as_tensor(t, device='cpu')

            torchax.disable_globally()
            try:
                model_output_cpu = to_cpu_pure(model_output)
                sample_cpu = to_cpu_pure(sample)
                timestep_cpu = to_cpu_pure(timestep) if torch.is_tensor(timestep) else timestep
                
                if model_output_cpu.numel() == sample_cpu.numel() and model_output_cpu.shape != sample_cpu.shape:
                    if model_output_cpu.shape[-1] == 2 and model_output_cpu.shape[-2] == sample_cpu.shape[-1]:
                         model_output_cpu = model_output_cpu.permute(0, 1, 2, 4, 3)
                    model_output_cpu = model_output_cpu.reshape(sample_cpu.shape)

                model_output_cpu = to_cpu_pure(model_output_cpu)
                timestep_cpu = to_cpu_pure(timestep_cpu)
                sample_cpu = to_cpu_pure(sample_cpu)

                scheduler_instance = original_step.__self__
                for k in dir(scheduler_instance):
                     if k.startswith("__"): continue
                     try: v = getattr(scheduler_instance, k)
                     except: continue
                     if torch.is_tensor(v) or (hasattr(v, 'shape') and hasattr(v, 'dtype')): 
                        if "torchax" in str(type(v)) or hasattr(v, 'env') or (torch.is_tensor(v) and v.device.type != 'cpu'):
                             try: setattr(scheduler_instance, k, to_cpu_pure(v))
                             except Exception: pass
                
                if hasattr(scheduler_instance, 'sigmas'):
                    scheduler_instance.sigmas = to_cpu_pure(scheduler_instance.sigmas)
                if hasattr(scheduler_instance, 'timesteps'):
                    scheduler_instance.timesteps = to_cpu_pure(scheduler_instance.timesteps)

                res = original_step(model_output_cpu, timestep_cpu, sample_cpu, return_dict=return_dict, generator=generator)
                
            finally:
                torchax.enable_globally()
            
            if 'res' in locals():
                 env = torchax.default_env()
                 def to_jax_recursive_final(obj):
                    if torch.is_tensor(obj):
                         if obj.device.type == 'cpu':
                             try:
                                 np_val = obj.detach().numpy()
                                 jax_val = jnp.array(np_val)
                                 return torchax.tensor.Tensor(jax_val, env)
                             except Exception as e:
                                 return obj
                         return obj
                    elif isinstance(obj, tuple):
                        return tuple(to_jax_recursive_final(x) for x in obj)
                    elif isinstance(obj, list):
                        return [to_jax_recursive_final(x) for x in obj]
                    return obj
                 return to_jax_recursive_final(res)
            return None

    pipe.scheduler.step = debug_step
    
    # Monkeypatch prepare_latents (Keep as is)
    original_prepare_latents = pipe.prepare_latents
    def patched_prepare_latents(*args, **kwargs):
        if pipe.transformer.config.in_channels == 65 and pipe.transformer.config.out_channels == 32:
             if len(args) > 1:
                 args_list = list(args)
                 args_list[1] = 32
                 args = tuple(args_list)
                 if 'num_channels_latents' in kwargs: del kwargs['num_channels_latents']
             else:
                 kwargs['num_channels_latents'] = 32
        return original_prepare_latents(*args, **kwargs)
    pipe.prepare_latents = patched_prepare_latents

    # Fast Debug Step (Keep as is)
    @jax.jit
    def jax_euler_step(sample, model_output, sigma, sigma_next):
        dt = sigma_next - sigma
        return sample + dt * model_output

    def fast_debug_step(model_output, timestep, sample, return_dict=True, generator=None):
        try:
            scheduler = original_step.__self__
            env = getattr(sample, 'env', getattr(model_output, 'env', torchax.default_env()))

            def unwrap_to_jax(x):
                if hasattr(x, '_jax_tensor'): return x._jax_tensor
                if hasattr(x, '_elem'): return x._elem
                if hasattr(x, 'cpu') and x.device.type == 'cpu': return jnp.array(x.detach().numpy())
                if isinstance(x, (int, float)): return jnp.array(x)
                return x

            jax_model_output = unwrap_to_jax(model_output)
            jax_sample = unwrap_to_jax(sample)
            jax_sigmas = unwrap_to_jax(scheduler.sigmas)

            jmo_shape = getattr(jax_model_output, 'shape', None)
            js_shape = getattr(jax_sample, 'shape', None)

            if jmo_shape is not None and js_shape is not None:
                 if jnp.prod(jnp.array(jmo_shape)) == jnp.prod(jnp.array(js_shape)):
                      if jmo_shape != js_shape:
                           if jmo_shape[-1] == 2 and jmo_shape[-2] == js_shape[-1]:
                                jax_model_output = jnp.transpose(jax_model_output, (0, 1, 2, 4, 3))
                           jax_model_output = jax_model_output.reshape(js_shape)

            step_index = scheduler.step_index
            if step_index is None:
                 scheduler._init_step_index(timestep)
                 step_index = scheduler.step_index
            
            sigma = jax_sigmas[step_index]
            sigma_next = jax_sigmas[step_index + 1]
            
            prev_sample_jax = jax_euler_step(jax_sample, jax_model_output, sigma, sigma_next)
            
            scheduler._step_index += 1
            res = (torchax.tensor.Tensor(prev_sample_jax, env),)
            
            if not return_dict: return res
            return res

        except Exception as e:
            print(f"DEBUG: Fast Path failed: {e}. Fallback CPU.") 
            return debug_step(model_output, timestep, sample, return_dict, generator)

    pipe.scheduler.step = fast_debug_step

    print("Running inference...")
    
    with mesh, torchax.default_env():
        start = time.perf_counter()
        # Request LATENTS only
        output_latents = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.num_inference_steps,
            output_type="latent"
        ).frames[0]
        
        # JAX Decoding
        print("Decoding latents with JAX VAE...")
        
        # Unwrap Latents
        def unwrap(x):
            if hasattr(x, 'jax'): return x.jax()
            if hasattr(x, 'unwrap'): return x.unwrap()
            if hasattr(x, '_elem'): return x._elem
            return x
        
        jax_latents = unwrap(output_latents)
        
        # Check rank and adjust layout
        # Expected from Diffusers: (C, T, H, W) for single batch item in frames[0]
        # or (B, C, T, H, W) if it was a tensor.
        
        if jax_latents.ndim == 4:
             # (C, T, H, W) -> (1, C, T, H, W)
             jax_latents = jnp.expand_dims(jax_latents, 0)
             
        # Convert (B, C, T, H, W) -> (B, T, H, W, C) for JAX VAE
        # Check for C=16 (Hunyuan 1.0) or C=32 (Hunyuan 1.5)
        if jax_latents.ndim == 5 and jax_latents.shape[1] in [16, 32]: 
             jax_latents = jnp.transpose(jax_latents, (0, 2, 3, 4, 1))
        
        latents_sharding = NamedSharding(mesh, P(None, None, None, 'sp', None))
        jax_latents = jax.device_put(jax_latents, latents_sharding)
        
        # JAX splits Width on 'sp'.
        # We MUST enforce output sharding to avoid gathering full video on each TPU (OOM).
        # Output: (B, T, H, W, C)
        video_sharding = NamedSharding(mesh, P(None, None, None, 'sp', None))
        
        @nnx.jit(out_shardings=video_sharding)
        def decode_fn(model, x):
             return model.decode(x)[0]
        
        decoded_videos = decode_fn(jax_vae, jax_latents)
        
        # Post-process
        # (val / 2 + 0.5).clamp(0, 1)
        decoded_videos = (decoded_videos / 2.0 + 0.5)
        decoded_videos = jnp.clip(decoded_videos, 0.0, 1.0)
 
        # Convert to numpy
        output_cpu = np.array(decoded_videos)

        print(f"Decoded shape: {output_cpu.shape}")
        
        # Usually (B, T, H, W, C)
        output = output_cpu[0] # (T, H, W, C)
        
        jax.effects_barrier()
        end = time.perf_counter()
        print(f"Inference time: {end - start:.4f}s")
        
        output_path = "hunyuan_output.mp4"
        export_to_video(list(output), output_path, fps=24)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()
