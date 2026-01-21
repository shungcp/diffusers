
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

# CRITICAL: Patch transformers BEFORE importing diffusers/transformers models
# This avoids vmap/functorch crashes in masking_utils.

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
    print("DEBUG: Monkeypatching torchax.ops.jaten._aten_unsafe_view for robustness...")
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

    print("DEBUG: Monkeypatching transformers.masking_utils.create_causal_mask (PRE-IMPORT) for torchax compatibility...")
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
        
        # Cast back to input dtype
        target_dtype = x_jax.dtype
        cos = cos.astype(target_dtype)
        sin = sin.astype(target_dtype)
        
        # Wrap outputs back to TorchaxTensor
        # Use env from input x
        env = getattr(x, 'env', None) or torchax.default_env()
        return torchax.tensor.Tensor(cos, env), torchax.tensor.Tensor(sin, env)

    print("DEBUG: Monkeypatching LlamaRotaryEmbedding.forward for torchax compatibility...")
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
print("DEBUG: Monkeypatched torchax.ops.mappings.j2t_dtype for robustness.")

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
print("DEBUG: Monkeypatched torchax.tensor.Tensor.reshape for JAX compatibility.")

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

        # PATCH: Convert to jax to ensure compatibility with compiled text_encoder
        # text_input_ids = text_input_ids.to('jax')
        # prompt_attention_mask = prompt_attention_mask.to('jax')
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)
        
        print(f"DEBUG: text_input_ids type: {type(text_input_ids)}")
        print(f"DEBUG: prompt_attention_mask type: {type(prompt_attention_mask)}")

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
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
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
          print(f"Warning: {k} (type: {type(v)}) does not have apply_jax_. Converting on the fly.")
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
             print(f"Manually converting buffer: {name}")
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

        # PATCH: Convert to jax to ensure compatibility with compiled text_encoder
        # text_input_ids = text_input_ids.to('jax')
        # prompt_attention_mask = prompt_attention_mask.to('jax')
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)
        
        print(f"DEBUG: text_input_ids type: {type(text_input_ids)}")
        print(f"DEBUG: prompt_attention_mask type: {type(prompt_attention_mask)}")

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
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

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

        # PATCH: Convert to jax
        # text_input_ids = text_input_ids.to('jax')
        # prompt_attention_mask = prompt_attention_mask.to('jax')
        
        env = torchax.default_env()
        text_input_ids = env.to_xla(text_input_ids)
        prompt_attention_mask = env.to_xla(prompt_attention_mask)

        prompt_embeds = self.text_encoder_2(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        )
        prompt_embeds = prompt_embeds.pooler_output
        
        if dtype:
            prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
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
    
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

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
    args = parser.parse_args()

    print(f"Initializing HunyuanVideoPipeline with model: {args.model_id}")

    # Load Pipeline (Load BEFORE enabling torchax to avoid safetensors/UntypedStorage issues)
    # Using float32 for loading to avoid precision issues before moving to JAX/BF16 if needed
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, 
        transformer=transformer, 
        vae=vae,
        torch_dtype=torch.float32
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
            
        print(f"DEBUG: Found {len(registries)} registries in ops_registry.")

        for reg in registries:
            # It's likely a dict or list
            if isinstance(reg, dict):
                items = reg.items()
            else:
                # If it's a list? unlikely but possible
                print(f"DEBUG: Registry is type {type(reg)}, skipping iteration.")
                continue

            for op, entry in items:
                op_str = str(op)
                if "convolution" in op_str or "conv3d" in op_str:
                    print(f"DEBUG: Patching op: {op_str} in registry")
                    # entry is likely an Operator object with .func
                    if hasattr(entry, 'func'):
                        entry.func = _robust_aten_convolution
                        count += 1
                    else:
                        print(f"DEBUG: Warning - Entry for {op_str} has no .func attribute")
                
        if count > 0:
            print(f"DEBUG: Successfully brute-force patched {count} convolution ops in registries.")
        else:
            print("DEBUG: WARNING: Could not find any convolution ops in any registry to patch!")
            
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
             print("DEBUG: Successfully patched EnvClass._to_copy for runtime robustness.")
             
    except Exception as e:
        print(f"DEBUG: Failed to patch Environment class: {e}")

    env = torchax.default_env()

    # TPU Mesh Setup
    n_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((1, (n_devices))) # Use all devices
    mesh = Mesh(devices, axis_names=('dp', 'sp'))
    env.default_device_or_sharding = NamedSharding(mesh, P())
    env._mesh = mesh
    
    # Replace Attention Processor
    print("Replacing attention processor with Splash Attention...")
    # We need to walk the model and replace processors
    # Hunyuan transformer uses 'attn1' usually? 
    # Let's check structure. It has transformer_blocks and single_transformer_blocks.
    # Each block has 'attn'.
    
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
    print("DEBUG: Splash Attention ENABLED.")

    # Compilation & Sharding
    print("Compiling modules with torchax...")
    
    # Text Encoders
    # Hunyuan has text_encoder (Llama) and text_encoder_2 (CLIP)
    # We need to move/compile them to avoid mixed tensor errors
    
    text_encoder_options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('output_hidden_states', 'return_dict', 'use_cache')}
    )

    # Text Encoder 1 (Llama)
    print("Compiling text_encoder (Llama)...")
    _move_module(pipe.text_encoder, env)
    pipe.text_encoder = torchax.compile(pipe.text_encoder, text_encoder_options)
    
    # Text Encoder 2 (CLIP)
    print("Compiling text_encoder_2 (CLIP)...")
    _move_module(pipe.text_encoder_2, env)
    pipe.text_encoder_2 = torchax.compile(pipe.text_encoder_2, text_encoder_options)

    # VAE
    print("Compiling vae...")
    # Check if VAE needs specific compile options. wan_tx compiled 'decode'.
    # Hunyuan VAE might need 'decode' too.
    # FIX: 'return_dict' determines output format and control flow, so it MUST be static.
    vae_options = torchax.CompileOptions(
        methods_to_compile=['decode'],
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    _move_module(pipe.vae, env)
    pipe.vae = torchax.compile(pipe.vae, vae_options)
    
    # Transformer
    print("Compiling transformer...")
    # Hunyuan transformer forward signature:
    # forward(hidden_states, timestep, encoder_hidden_states, ...)
    # We might need to handle static args if any.
    # For now, let's try default compilation.
    _move_module(pipe.transformer, env)
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict', 'attention_kwargs')}
    )
    pipe.transformer = torchax.compile(pipe.transformer, transformer_options)

    # Debug Recompilation: Wrap the compiled transformer
    _compiled_transformer_forward = pipe.transformer.forward
    def debug_transformer_forward(*args, **kwargs):
        # Inspect types of key inputs
        # args[0] is usually hidden_states (Tensor)
        # args[1] might be timestep? Or use kwargs.
        
        # Log minimal info to avoid spam, but enough to see changes
        msg = "DEBUG TX CALL: "
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                 msg += f"Arg{i}:{type(arg)} shape={arg.shape} id={id(arg)} | "
            elif isinstance(arg, (int, float)):
                 msg += f"Arg{i}:{type(arg)} val={arg} | "
            else:
                 msg += f"Arg{i}:{type(arg)} | "
        
        for k, v in kwargs.items():
             if hasattr(v, 'shape'):
                  msg += f"K_{k}:{type(v)} shape={v.shape} id={id(v)} | "
             elif isinstance(v, (int, float)):
                  msg += f"K_{k}:{type(v)} val={v} | "
             else:
                  msg += f"K_{k}:{type(v)} | "
        
        print(msg)
        return _compiled_transformer_forward(*args, **kwargs)
    
    pipe.transformer.forward = debug_transformer_forward
    # pipe.transformer is a TorchaxWrapper, so modifying forward instance method might work 
    # if it delegates. If not, we wrapper the object.
    # Actually, pipe.transformer is the wrapper. 
    # .forward is the method.
    # We can assign to the instance.

    
    # Apply Sharding (Manual)
    print("Applying sharding rules...")
    if hasattr(pipe.transformer, 'params'):
            pipe.transformer.params = _shard_weight_dict(pipe.transformer.params, transformer_shardings, mesh)
            pipe.transformer.buffers = _shard_weight_dict(pipe.transformer.buffers, transformer_shardings, mesh)
    
    if hasattr(pipe.vae, 'params'):
            pipe.vae.params = _shard_weight_dict(pipe.vae.params, vae_shardings, mesh)
            pipe.vae.buffers = _shard_weight_dict(pipe.vae.buffers, vae_shardings, mesh)
            
    # Also shard/replicate text encoders
    # For now, just replicate (empty sharding dict)
    if hasattr(pipe.text_encoder, 'params'):
            pipe.text_encoder.params = _shard_weight_dict(pipe.text_encoder.params, {}, mesh)
            pipe.text_encoder.buffers = _shard_weight_dict(pipe.text_encoder.buffers, {}, mesh)

    if hasattr(pipe.text_encoder_2, 'params'):
            pipe.text_encoder_2.params = _shard_weight_dict(pipe.text_encoder_2.params, {}, mesh)
            pipe.text_encoder_2.buffers = _shard_weight_dict(pipe.text_encoder_2.buffers, {}, mesh)

    # Monkeypatch scheduler.step to debug type mismatch
    original_step = pipe.scheduler.step
    def debug_step(model_output, timestep, sample, return_dict=True, generator=None):
        try:
            return original_step(model_output, timestep, sample, return_dict=return_dict, generator=generator)
        except Exception as e: # Catch TypeError and AssertionError
            print(f"Caught error in scheduler step: {e}. Falling back to CPU with global disable.")
            
            # Helper to move to cpu and strip wrappers
            def to_cpu_pure(t):
                if hasattr(t, 'numpy'):
                    try:
                        return torch.as_tensor(t.detach().numpy())
                    except:
                        pass
                if hasattr(t, 'cpu'):
                    return t.cpu()
                return torch.as_tensor(t, device='cpu')

            # We MUST disable globally to stop Torchax from intercepting CPU tensor operations
            torchax.disable_globally()
            try:
                # Convert inputs to pure CPU tensors
                model_output_cpu = to_cpu_pure(model_output)
                sample_cpu = to_cpu_pure(sample)
                timestep_cpu = to_cpu_pure(timestep) if torch.is_tensor(timestep) else timestep
                
                print(f"DEBUG: model_output_cpu shape: {model_output_cpu.shape}")
                print(f"DEBUG: sample_cpu shape: {sample_cpu.shape}")
                
                # Check for shape mismatch and reshape if element counts match
                if model_output_cpu.numel() == sample_cpu.numel() and model_output_cpu.shape != sample_cpu.shape:
                    print(f"DEBUG: Reshaping model_output_cpu from {model_output_cpu.shape} to {sample_cpu.shape}")
                    
                    # Heuristic: if last dim is 2 and we need to merge it into height (dim -2?)
                    # Current: [1, 256, 45, 160, 2] -> Target: [1, 16, 16, 90, 160] (B, C, T, H, W)
                    # 45 * 2 = 90. 160 = 160.
                    # We have [..., H/2, W, 2]. We need [..., H, W].
                    # So we need to bring '2' next to 'H/2'.
                    # [..., 45, 160, 2] -> permute(..., 45, 2, 160) -> reshape(..., 90, 160)
                    if model_output_cpu.shape[-1] == 2 and model_output_cpu.shape[-2] == sample_cpu.shape[-1]:
                         print("DEBUG: Applying permute(0, 1, 2, 4, 3) to fix spatial layout.")
                         model_output_cpu = model_output_cpu.permute(0, 1, 2, 4, 3)
                    
                    model_output_cpu = model_output_cpu.reshape(sample_cpu.shape)

                
                # Sanitize scheduler state (sigmas, etc) which might be TorchaxTensors
                # Access scheduler instance via bound method
                scheduler_instance = original_step.__self__
                for k, v in scheduler_instance.__dict__.items():
                     if torch.is_tensor(v): 
                        # Check if it looks like a TorchaxTensor (via string check or type)
                        # or just blindly convert all to CPU pure tensor
                        if "torchax" in str(type(v)) or hasattr(v, 'env'):
                             try:
                                setattr(scheduler_instance, k, to_cpu_pure(v))
                             except Exception:
                                pass

                # Try running step
                res = original_step(model_output_cpu, timestep_cpu, sample_cpu, return_dict=return_dict, generator=generator)
                
            finally:
                # Re-enable globally
                torchax.enable_globally()
            
            # Convert result back to JAX/TPU
            if 'res' in locals():
                 print(f"DEBUG: Step finished. Output type: {type(res)}")
                 
                 env = torchax.default_env()
                 
                 def to_jax_recursive_final(obj):
                    if torch.is_tensor(obj):
                         if obj.device.type == 'cpu':
                             try:
                                 # Bridge via Numpy -> JAX -> Torchax
                                 # output of .numpy() is numpy array
                                 np_val = obj.detach().numpy()
                                 jax_val = jnp.array(np_val)
                                 return torchax.tensor.Tensor(jax_val, env)
                             except Exception as e:
                                 print(f"DEBUG: Failed to wrap output tensor via numpy: {e}")
                                 return obj
                         return obj
                    elif isinstance(obj, tuple):
                        return tuple(to_jax_recursive_final(x) for x in obj)
                    elif isinstance(obj, list):
                        return [to_jax_recursive_final(x) for x in obj]
                    return obj
                 
                 return to_jax_recursive_final(res)
            
            # Fallback if res not defined
            return None

    pipe.scheduler.step = debug_step

    # Optimized JAX Scheduler Step
    # JIT compile the math kernel for efficiency
    pipe.scheduler.step = debug_step

    # Optimized JAX Scheduler Step
    # JIT compile the PURE math kernel for efficiency
    @jax.jit
    def jax_euler_step(sample, model_output, sigma, sigma_next):
        dt = sigma_next - sigma
        # Pure math: broadcasting should work if shapes match
        return sample + dt * model_output

    def fast_debug_step(model_output, timestep, sample, return_dict=True, generator=None):
        try:
            # 1. Access Scheduler State
            scheduler = original_step.__self__
            
            # CRITICAL: Propagate Environment from Input to prevent Recompilation
            env = getattr(sample, 'env', getattr(model_output, 'env', torchax.default_env()))

            # Helper: Unwrap to JAX
            def unwrap_to_jax(x):
                # Check for explicit JAX tensor wrapper
                if hasattr(x, '_jax_tensor'):
                    return x._jax_tensor
                # 'TorchaxTensor' often uses '_elem' to store the JAX array
                if hasattr(x, '_elem'):
                    return x._elem
                
                # Check if it's a Torch Tensor
                if hasattr(x, 'cpu'):
                     if x.device.type == 'cpu':
                        return jnp.array(x.detach().numpy())
                
                # Primitives
                if isinstance(x, (int, float)):
                    return jnp.array(x)
                
                return x

            jax_model_output = unwrap_to_jax(model_output)
            jax_sample = unwrap_to_jax(sample)
            jax_sigmas = unwrap_to_jax(scheduler.sigmas)

            # 2. Fix Spatial Layout (Permute/Reshape) - EAGER MODE


            # 2. Fix Spatial Layout (Permute/Reshape) - EAGER MODE
            # We do this in Python to ensure correct shapes are passed to JIT
            # Heuristic: [B, PatchL, H_patch, W_full, PatchH] -> [B, C, F, H, W]
            jmo_shape = getattr(jax_model_output, 'shape', None)
            js_shape = getattr(jax_sample, 'shape', None)

            if jmo_shape is not None and js_shape is not None:
                 if jnp.prod(jnp.array(jmo_shape)) == jnp.prod(jnp.array(js_shape)):
                      if jmo_shape != js_shape:
                           # print(f"DEBUG FAST: Reshaping {jmo_shape} -> {js_shape}")
                           if jmo_shape[-1] == 2 and jmo_shape[-2] == js_shape[-1]:
                                jax_model_output = jnp.transpose(jax_model_output, (0, 1, 2, 4, 3))
                           
                           jax_model_output = jax_model_output.reshape(js_shape)

            # 3. Euler Step Logic (JAX)
            step_index = scheduler.step_index
            if step_index is None:
                 scheduler._init_step_index(timestep)
                 step_index = scheduler.step_index
            
            sigma = jax_sigmas[step_index]
            sigma_next = jax_sigmas[step_index + 1]
            
            # Execute Jitted Kernel (Shapes MUST match now)
            prev_sample_jax = jax_euler_step(jax_sample, jax_model_output, sigma, sigma_next)
            
            # 4. Update State
            scheduler._step_index += 1
            
            # 5. Wrap & Return
            print(f"DEBUG FAST: Success. EnvID: {id(env)}")
            res = (torchax.tensor.Tensor(prev_sample_jax, env),)
            
            if not return_dict:
                return res
            return res

        except Exception as e:
            # Uncomment for visibility
            print(f"DEBUG: Fast Path failed: {e}. Fallback CPU.") 
            return debug_step(model_output, timestep, sample, return_dict, generator)

    # Use the fast JAX step by default
    # Use the fast JAX step by default
    pipe.scheduler.step = fast_debug_step

    # REMOVED: Proactive scheduler state wrapping.
    # Moving scheduler tensors to TorchaxTensor explicitly caused AssertionError during set_timesteps
    # because simple CPU math (sigmas * scalar) triggered dispatch without valid env.
    # We let scheduler run on CPU and convert to JAX on-the-fly in fast_debug_step.
    
    # print("Moving scheduler state to JAX device with explicit default_env...")
    # env = torchax.default_env()
    # for k, v in pipe.scheduler.__dict__.items():
    #     if torch.is_tensor(v):
    #          try:
    #             # Use explicit wrapping to ensure environment consistency
    #             # Bridge via Numpy -> JAX -> Torchax
    #             if getattr(v, 'device', torch.device('cpu')).type == 'cpu':
    #                 np_val = v.detach().numpy()
    #             else:
    #                 np_val = v.cpu().detach().numpy()
    # 
    #             jax_val = jnp.array(np_val)
    #             # Use default_env explicitly
    #             wrapped_v = torchax.tensor.Tensor(jax_val, env)
    #             setattr(pipe.scheduler, k, wrapped_v)
    #          except Exception as e:
    #             print(f"Failed to move scheduler buffer {k} to jax: {e}")

    print("Running inference...")
    
    print("Running inference...")
    
    with mesh, torchax.default_env():
        output = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]
        jax.effects_barrier()
        end = time.perf_counter()
        print(f"Inference time: {end - start:.4f}s")
        
        output_path = "hunyuan_output.mp4"
        export_to_video(output, output_path, fps=15)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()
