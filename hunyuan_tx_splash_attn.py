
import functools
from typing import Optional, List, Tuple, Union, Sequence
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
import argparse

# Reusing Splash Attention from Wan2.1 (generic enough for now)
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
            # (H, S, D)
            return splash_attention(
                q_3d, k_3d, v_3d,
                mask=None,
                segment_ids=None,
                block_sizes=None,
            )

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

ops_registry.register("splash_attention", _tpu_splash_attention)

# Configuration
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
HEIGHT = 720
WIDTH = 1280
FRAMES = 61 # Standard for Hunyuan
NUM_STEP = 30

USE_DP = True
USE_FSDP = True
SP_NUM = 1

# Sharding rules (will need adjustment for Hunyuan architecture)
LOGICAL_AXIS_RULES = (
    ('conv_out', ('axis','dp','sp')),
    ('conv_in', ('axis','dp','sp')),
    # Add Hunyuan specific rules if needed
)
axis = 'axis'

# Variable State Metadata Helper
def _add_sharding_rule(vs: nnx.Variable, logical_axis_rules) -> nnx.Variable:
  vs.set_metadata(sharding_rules=logical_axis_rules)
  return vs

@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = functools.partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model

# Custom Splash Attention Processor for HunyuanVideo
class HunyuanSplashAttnProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
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

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEP)
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    args = parser.parse_args()

    print(f"Initializing HunyuanVideoPipeline with model: {args.model_id}")

    # TPU Mesh Setup
    devices = mesh_utils.create_device_mesh((1, 4)) # Adjust based on available devices
    mesh = Mesh(devices, axis_names=('dp', 'sp'))
    
    with mesh:
        # Load Pipeline
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
        
        # Replace Attention Processor
        # We need to do this BEFORE wrapping with torchax if possible, or after?
        # PyTorch modules should be modified before wrapping.
        # print("Swapping Attention Processors...")
        # from diffusers.models.attention_processor import Attention
        # for name, module in pipe.transformer.named_modules():
        #     if isinstance(module, Attention):
        #         module.processor = HunyuanSplashAttnProcessor()

        # Wrap modules with torchax
        print("Wrapping modules with torchax...")
        pipe.transformer = torchax.nnx.TorchModule(pipe.transformer)
        pipe.vae = torchax.nnx.TorchModule(pipe.vae)
        
        # Apply Sharding
        print("Applying sharding rules...")
        # (Assuming we define sharding rules for Hunyuan layers later)
        # pipe.transformer = create_sharded_logical_model(pipe.transformer, LOGICAL_AXIS_RULES)

        print("Running inference...")
        output = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]
        
        output_path = "hunyuan_output.mp4"
        export_to_video(output, output_path, fps=15)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()
