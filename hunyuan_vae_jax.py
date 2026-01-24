# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, List, Sequence, Union, Optional
import os
import json
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from diffusers.utils import logging

# Basic Helper for Tuple Canonicalization
def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank} integers. Got {x}")

class HunyuanVideoCausalConv3d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        use_bias: bool = True,
        mesh: jax.sharding.Mesh = None,
    ):
        self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
        self.stride = _canonicalize_tuple(stride, 3, "stride")
        self.dilation = _canonicalize_tuple(dilation, 3, "dilation")
        
        pad_t, pad_h, pad_w = _canonicalize_tuple(padding, 3, "padding")
        
        self.pad_time = 2 * pad_t
        self.pad_height = pad_h
        self.pad_width = pad_w
        
        # Partitioning Logic
        # Kernel: (T, H, W, In, Out)
        # We shard Output Channel on 'conv_out' logical axis IF possible.
        
        kernel_init = nnx.initializers.xavier_uniform()
        if mesh is not None:
             pass 
             # We will handle sharding explicitly during weight loading or via logical axis rules globally if needed.
             # For now, remove eager partitioning annotation to fix ValueError.
             # kernel_sharding = (None, None, None, None, "sp")
             # kernel_init = nnx.with_partitioning(kernel_init, kernel_sharding)
             
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID', 
            use_bias=use_bias,
            kernel_init=kernel_init, 
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            input_dilation=1, 
            kernel_dilation=self.dilation
        )

    def __call__(self, x):
        pads = (
            (0, 0), # Batch
            (self.pad_time, 0), # Time (Causal)
            (self.pad_height, self.pad_height), # Height
            (self.pad_width, self.pad_width), # Width
            (0, 0) # Channel
        )
        
        if any(p[0] > 0 or p[1] > 0 for p in pads):
             x = jnp.pad(x, pads)
             
        return self.conv(x)


class HunyuanVideoGroupNorm(nnx.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6, affine: bool = True, rngs: nnx.Rngs = None, dtype=jnp.float32, param_dtype=jnp.float32):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nnx.Param(jnp.ones((num_channels,), dtype=param_dtype))
            self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=param_dtype))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x):
        # x: (Batch, Time, Height, Width, Channels)
        # Matches PyTorch GroupNorm logic
        # GroupNorm happens over (C, T, H, W) effectively
        # We need to reshape to (Batch, Groups, Channels//Groups, ...)
        
        x_shape = x.shape
        batch = x_shape[0]
        channels = x_shape[-1] # Assuming channels last
        spatial_dims = x_shape[1:-1] # T, H, W
        
        # Reshape to (Batch, Groups, C//G, ...)
        # Flatten spatial dims for moments
        
        group_size = channels // self.num_groups
        
        # (B, T, H, W, G, C//G)
        x_reshaped = x.reshape(batch, *spatial_dims, self.num_groups, group_size)
        
        # Normalize over (Spatial + C//G)
        # Axes to reduce: T(1), H(2), W(3), C//G(5)
        # Assuming NDHWC input: 0, 1, 2, 3, 4
        # after reshape: 0, 1(T), 2(H), 3(W), 4(G), 5(C_sub)
        # Normalize over (Spatial + C//G)
        # Axes to reduce: T(1), H(2), W(3), C//G(5)
        
        # Manual Mean/Var to avoid f32 upcast of the huge tensor x_reshaped
        # x_reshaped is likely bfloat16. jnp.var might cast to f32.
        
        reduce_axes = (1, 2, 3, 5)
        
        # 1. Mean (Keep in input dtype, assume bfloat16 range is sufficient for mean)
        # Or cast to f32 for stability but only for the reduction result?
        # mean = jnp.mean(x_reshaped, axis=reduce_axes, keepdims=True)
        
        # Better: cast to f32 for accumulation? 
        # But we don't want to cast x_reshaped to f32 (28GB).
        # jnp.mean usually accumulates in f32 if using bfloat16.
        
        mean = jnp.mean(x_reshaped, axis=reduce_axes, keepdims=True)
        
        # 2. Var = Mean(x^2) - Mean(x)^2 OR Mean((x-mu)^2).
        # Mean((x-mu)^2) is safer.
        # x_reshaped (bf16) - mean (bf16) -> diff (bf16).
        # diff^2 (bf16).
        # mean(diff^2) (f32 accumulation).
        
        # We manually compute it to ensure intermediate 'diff' is not promoted to f32.
        diff = x_reshaped - mean
        sq_diff = jnp.square(diff)
        var = jnp.mean(sq_diff, axis=reduce_axes, keepdims=True)
        
        x_norm = diff / jnp.sqrt(var + self.eps)
        
        # Restore shape
        x_norm = x_norm.reshape(x_shape)
        
        if self.affine:
            w = self.weight.value.astype(x.dtype)
            b = self.bias.value.astype(x.dtype)
            x_norm = x_norm * w + b
            
        return x_norm


class HunyuanVideoResnetBlockCausal3D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        rngs: nnx.Rngs = None,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        mesh: jax.sharding.Mesh = None,
    ):
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if non_linearity == "swish":
            self.nonlinearity = nnx.silu
        elif non_linearity == "silu":
            self.nonlinearity = nnx.silu
        elif non_linearity == "mish":
            self.nonlinearity = lambda x: x * jnp.tanh(nnx.softplus(x))
        elif non_linearity == "gelu":
            self.nonlinearity = nnx.gelu
        else:
            raise ValueError(f"Unsupported nonlinearity: {non_linearity}")

        self.norm1 = HunyuanVideoGroupNorm(groups, in_channels, eps=eps, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, kernel_size=3, padding=1, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
        
        self.norm2 = HunyuanVideoGroupNorm(groups, out_channels, eps=eps, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, kernel_size=3, padding=1, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
        
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(in_channels, out_channels, kernel_size=1, padding=0, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
        else:
            self.conv_shortcut = None

    @nnx.remat
    def __call__(self, x):
        h = x
        
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.conv_shortcut is not None:
             x = self.conv_shortcut(x)
             
        return x + h


class HunyuanVideoUpsampleCausal3D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: Tuple[int, int, int] = (2, 2, 2),
        match_channel: bool = False,
        rngs: nnx.Rngs = None,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        mesh: jax.sharding.Mesh = None,
    ):
        self.upsample_factor = upsample_factor
        self.match_channel = match_channel
        
        if match_channel:
             # Hunyuan 1.5 Special Logic: Conv (Expansion) -> PixelShuffle
             # Output channels of Conv must satisfy PixelShuffle compression
             # c_out_final = c_conv_out / (f_t * f_h * f_w)
             # So c_conv_out = out_channels * (f_t * f_h * f_w)
             
             factor_prod = upsample_factor[0] * upsample_factor[1] * upsample_factor[2]
             conv_out_channels = out_channels * factor_prod
             
             self.conv = HunyuanVideoCausalConv3d(
                 in_channels, 
                 conv_out_channels, 
                 kernel_size=3, 
                 padding=1, 
                 rngs=rngs, 
                 dtype=dtype, 
                 param_dtype=param_dtype,
                 mesh=mesh
             )
        else:
             self.conv = HunyuanVideoCausalConv3d(
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 padding=1, 
                 rngs=rngs, 
                 dtype=dtype, 
                 param_dtype=param_dtype,
                 mesh=mesh
             )

    def __call__(self, x):
        # x: (B, T, H, W, C)
        
        if self.match_channel:
             # Logic: Conv -> PixelShuffle
             x = self.conv(x) # (B, T, H, W, C_big)
             
             b, t, h, w, c_big = x.shape
             ft, fh, fw = self.upsample_factor
             c_out = c_big // (ft * fh * fw)
             
             # Reshape to split channels: (B, T, H, W, ft * fh * fw, C_out)
             # Actually better: (B, T, H, W, C_out, ft, fh, fw)
             # Wait, usually PixelShuffle assumes (B, C*r^2, H, W).
             # Here we want (B, C, T*ft, H*fh, W*fw).
             # We need to reshape C_big -> (C_out, ft, fh, fw)
             # And then permute to (B, T, ft, H, fh, W, fw, C_out)
             # And then reshape.
             
             # Let's do reshape C_big -> (ft, fh, fw, C_out)
             # Assuming standard order of channel packing matches PyTorch PixelShuffle(3D equivalent)
             # PyTorch PixelShuffle (DepthToSpace) usually expects (C, r, r) packed.
             
             x = x.reshape(b, t, h, w, c_out, ft, fh, fw)
             
             # We want (B, T*ft, H*fh, W*fw, C_out)
             # Current: (B, T, H, W, C_out, ft, fh, fw)
             # Permute to: (B, T, ft, H, fh, W, fw, C_out)
             
             x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
             
             x = x.reshape(b, t * ft, h * fh, w * fw, c_out)
             
             return x
             
        else:
             # Logic: Interpolate -> Conv
             
             # Logic: Interpolate -> Conv
             
             # Interpolate (Nearest Neighbor) using repeat for efficiency
             # x: (B, T, H, W, C)
             
             if self.upsample_factor[0] > 1:
                 x = jnp.repeat(x, self.upsample_factor[0], axis=1)
             if self.upsample_factor[1] > 1:
                 x = jnp.repeat(x, self.upsample_factor[1], axis=2)
             if self.upsample_factor[2] > 1:
                 x = jnp.repeat(x, self.upsample_factor[2], axis=3)
             
             # Conv
             x = self.conv(x)
             return x


class AutoencoderKLHunyuanVideo(nnx.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        z_dim: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        downsample_match_channel: bool = False,
        upsample_match_channel_config: List[bool] = None, # List of bools for each upblock
        rngs: nnx.Rngs = None,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        mesh: jax.sharding.Mesh = None,
    ):
        self.z_dim = z_dim
        # Encoder (Placeholder for now, or implement if needed)
        # We focus on Decoder for T2V Inference OOM fix.
        self.encoder = None 
        
        # Decoder
        # Pass upsample_match_channel_config to Decoder
        self.decoder = HunyuanVideoDecoder3D(
            in_channels=z_dim,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            upsample_match_channel_config=upsample_match_channel_config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh
        )
        
        # Quant Conv / Post Quant Conv
        # PyTorch: quant_conv (2*z -> 2*z), post_quant_conv (z -> z)
        # JAX Conv: In -> Out
        
        self.quant_conv = HunyuanVideoCausalConv3d(2*z_dim, 2*z_dim, 1, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
        self.post_quant_conv = HunyuanVideoCausalConv3d(z_dim, z_dim, 1, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)

    def decode(self, latents, return_dict=False):
        h = self.post_quant_conv(latents)
        h = self.decoder(h)
        if not return_dict:
            return (h,)
        return h

# Update Decoder to accept config

class HunyuanVideoUpBlock3D(nnx.Module):
    def __init__(self, resnets, upsampler):
        self.resnets = nnx.List(resnets)
        self.upsampler = upsampler

    @nnx.remat
    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler:
            x = self.upsampler(x)
        return x

class HunyuanVideoDecoder3D(nnx.Module):
    def __init__(
        self,
        in_channels: int = 16, 
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        upsample_match_channel_config: List[bool] = None,
        rngs: nnx.Rngs = None,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        mesh: jax.sharding.Mesh = None,
    ):
        self.block_out_channels = block_out_channels
        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, 
            block_out_channels[-1], 
            kernel_size=3, 
            padding=1, 
            rngs=rngs, 
            dtype=dtype, 
            param_dtype=param_dtype,
            mesh=mesh
        )
        
        mid_channels = block_out_channels[-1]
        self.mid_block_resnets = nnx.List([
            HunyuanVideoResnetBlockCausal3D(mid_channels, mid_channels, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh),
            HunyuanVideoResnetBlockCausal3D(mid_channels, mid_channels, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
        ])
        
        reversed_channels = list(reversed(block_out_channels))
        
        if upsample_match_channel_config is None:
             upsample_match_channel_config = [False] * (len(reversed_channels) - 1)
             
        self.up_blocks = nnx.List([])
        
        prev_channels = mid_channels # Start with output of mid_block
        
        for i, out_c in enumerate(reversed_channels):
            in_c = prev_channels
            
            block_resnets = []
            for _ in range(layers_per_block):
                 block_resnets.append(
                     HunyuanVideoResnetBlockCausal3D(
                         in_c if len(block_resnets)==0 else out_c, 
                         out_c,
                         rngs=rngs, dtype=dtype, param_dtype=param_dtype,
                         mesh=mesh
                     )
                 )
            
            upsampler = None
            if i < len(reversed_channels) - 1:
                next_c = reversed_channels[i+1]
                match_val = upsample_match_channel_config[i] if i < len(upsample_match_channel_config) else False
                upsampler = HunyuanVideoUpsampleCausal3D(out_c, next_c, rngs=rngs, dtype=dtype, param_dtype=param_dtype, match_channel=match_val, mesh=mesh) 
                
            self.up_blocks.append(
                HunyuanVideoUpBlock3D(block_resnets, upsampler)
            )
            
            # If upsampler exists, it changed the channels to next_c
            # If no upsampler (last block), we stay at out_c (but loop ends anyway)
            # Actually, Resnet output `out_c`.
            # If Upsampler exists, input is `out_c`, output is `next_c`.
            # So next block input should be `next_c`.
            # If no Upsampler, next block input (if any) would be `out_c`.
            
            if upsampler:
                prev_channels = next_c
            else:
                prev_channels = out_c
            
            
        self.conv_norm_out = HunyuanVideoGroupNorm(32, reversed_channels[-1], rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.conv_act = nnx.silu
        self.conv_out = HunyuanVideoCausalConv3d(reversed_channels[-1], out_channels, kernel_size=3, padding=1, rngs=rngs, dtype=dtype, param_dtype=param_dtype, mesh=mesh)
    
    def __call__(self, x):
         h = self.conv_in(x)
         for resnet in self.mid_block_resnets:
             h = resnet(h)
         for block in self.up_blocks:
             h = block(h)
         h = self.conv_norm_out(h)
         h = self.conv_act(h)
         h = self.conv_out(h)
         return h

def load_hunyuan_vae_jax(pretrained_model_name_or_path, subfolder="vae", dtype=jnp.float32, mesh=None):
    print(f"Loading Hunyuan VAE from {pretrained_model_name_or_path} ...")
    
    # 1. Download/Locate safetensors
    tensors = {}
    if os.path.exists(pretrained_model_name_or_path):
        # Local path logic
        index_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
            shard_files = set(index['weight_map'].values())
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, subfolder, shard_file)
                with safe_open(shard_path, framework="np") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k)
        else:
            ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            with safe_open(ckpt_path, framework="np") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    else:
        # Hub logic
        try:
            from huggingface_hub.utils import EntryNotFoundError
            index_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename="diffusion_pytorch_model.safetensors.index.json")
            load_sharded = True
        except (EntryNotFoundError, Exception):
            load_sharded = False
        
        if load_sharded:
            print("Detected sharded VAE checkpoint, loading shards...")
            with open(index_path, 'r') as f:
                index = json.load(f)
            shard_files = set(index['weight_map'].values())
            for shard_file in shard_files:
                shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=shard_file)
                with safe_open(shard_path, framework="np") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k)
        else:
            print("Loading single VAE checkpoint file...")
            ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename="diffusion_pytorch_model.safetensors")
            with safe_open(ckpt_path, framework="np") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            

    # 2. Inspect Config from Tensors (Heuristic)
    # Detect upsample_match_channel configuration
    # Hunyuan 1.5 usually has 5 blocks: [128, 256, 512, 1024, 1024]
    # UpBlocks count: 4 (since len(channels)-1)
    
    upsample_match_channel_config = []
    # We iterate 4 times for the 4 upsamplers
    for i in range(4): 
        k = f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
        if k in tensors:
            w = tensors[k]
            # w shape: (Out, In, T, H, W)
            out_c, in_c = w.shape[0], w.shape[1]
            if out_c > in_c:
                 print(f"Detected match_channel=True for UpBlock {i} (Out={out_c}, In={in_c})")
                 upsample_match_channel_config.append(True)
            else:
                 upsample_match_channel_config.append(False)
        else:
             # print(f"Warning: Could not find upsampler weight for block {i}, assuming False")
             upsample_match_channel_config.append(False)
             
    # 3. Instantiate Model
    rngs = nnx.Rngs(0)
    
    # Use contextlib.nullcontext if mesh is None
    from contextlib import nullcontext
    mesh_context = mesh if mesh is not None else nullcontext()
    
    with mesh_context:
        model = AutoencoderKLHunyuanVideo(
           in_channels=3,
           out_channels=3,
           z_dim=32, # Hunyuan 1.5 uses 32 latent channels
           block_out_channels=(128, 256, 512, 1024, 1024), # 5 blocks
           layers_per_block=2,
           upsample_match_channel_config=upsample_match_channel_config,
           rngs=rngs,
           dtype=dtype, 
           param_dtype=dtype,
           mesh=mesh
        )
    
    # 4. Load Weights
    # Iterate module parameters and find matching tensor
    
    # Helper to traverse NNX model
    # We can use flatten/unflatten or manual traversal. 
    # Since we need to rename keys, we iterate TENSORS and assign to MODEL.
    
    # Map PyTorch keys to NNX keys/attributes
    # NNX structure:
    # decoder.up_blocks[i]['resnets'][j].conv1.conv.kernel
    # decoder.up_blocks is list of dicts.
    # self.up_blocks is list.
    
    # We need to map:
    # decoder.up_blocks.0.resnets.0.conv1.weight -> decoder.up_blocks[0]['resnets'][0].conv1.conv.kernel
    
    # Let's perform state_dict mapping.
    
    import flax.traverse_util
    graphdef, state = nnx.split(model)
    # state.to_flat_dict() is removed in 0.12+. Use to_pure_dict + flatten_dict.
    flat_state = flax.traverse_util.flatten_dict(state.to_pure_dict())
    new_state_dict = {}
    
    # Create a mapping from PyTorch keys to FlatState keys?
    # Or iterate FlatState keys and find PyTorch tensor?
    
    # Better: Iterate PyTorch keys, transform name, find in FlatState.
    
    # FlatState keys tuple example: ('decoder', 'up_blocks', '0', 'resnets', '0', 'conv1', 'conv', 'kernel')
    # Note: up_blocks[i] is a dict in my implementation?
    # No, nnx.Module doesn't support list of dicts naturally as submodules unless handled?
    # Wait, simple list of dicts might not track submodules if not nnx.List or similar.
    # My Implementation: self.up_blocks = [] ... append({...})
    # These are python lists/dicts. nnx.split might lose them if they are not nnx structures?
    # NNX traverses python trees (lists, dicts, tuples) looking for Variables/Modules.
    # So it SHOULD work.
    
    for pt_key, val in tensors.items():
        path = pt_key.split('.')
        
        # Shortcut mapping for mid_block
        # PyTorch: decoder.mid_block.resnets.0... -> JAX: decoder.mid_block_resnets.0...
        if 'mid_block' in path:
             try:
                 idx = path.index('mid_block')
                 if idx + 1 < len(path) and path[idx+1] == 'resnets':
                      path[idx] = 'mid_block_resnets'
                      # We need to Keep the index '0' after resnets
                      # PyTorch: mid_block.resnets.0
                      # JAX nnx.List: mid_block_resnets.0
                      # So we just remove 'resnets' segment
                      path.pop(idx+1) 
             except ValueError:
                 pass
        
        # Shortcuts for common mappings
        # conv.weight -> conv.kernel (and transpose)
        # conv.bias -> conv.bias
        # norm.weight -> weight (My GroupNorm uses weight/bias parameters, not modules? No, parameters)
        # In HunyuanVideoGroupNorm: self.weight, self.bias are Params.
        
        # PyTorch: norm.weight -> JAX: norm.weight.value (if Param)
        # NNX Flat Dict keys end with the Param name?
        # e.g. ('decoder', '...', 'norm1', 'weight') -> Param
        
        # Detect Layer Type by name?
        
        is_conv = 'conv' in path[-2] or 'conv' in path[-1] # heuristic
        is_norm = 'norm' in path[-2]
        
        # Fix path
        new_path = list(path)
        
        # Fix specific layer names
        # 'resnets.0' -> 0 in list
        # 'upsamplers.0' -> upsampler (it is a single object in my dict, not list)
        
        # My structure: self.up_blocks is list of dicts.
        # dict keys: 'resnets', 'upsampler'.
        # 'resnets' is list.
        # 'upsampler' is Module or None.
        
        # PyTorch: up_blocks.0.resnets.0
        # JAX: up_blocks.0.resnets.0 (Match)
        
        # PyTorch: up_blocks.0.upsamplers.0
        # JAX: up_blocks.0.upsampler (No list)
        
        # Fix 'upsamplers.0' -> 'upsampler'
        for i, p in enumerate(new_path):
            if p == 'upsamplers' and i+1 < len(new_path) and new_path[i+1] == '0':
                new_path[i] = 'upsampler'
                new_path.pop(i+1) # remove '0'
                break
        
        # Fix parameter name
        param_name = new_path[-1]
        
        # Mapping for GroupNorm: gamma -> weight, beta -> bias
        if param_name == 'gamma':
             param_name = 'weight'
             new_path[-1] = 'weight'
        elif param_name == 'beta':
             param_name = 'bias'
             new_path[-1] = 'bias'

        if param_name == 'weight':
            # Check if Conv
            # My Conv module has 'conv' submodule
            # PyTorch: conv1.weight
            # JAX: conv1.conv.kernel (and kernel is transposed)
            
            # Check if this node in JAX is a Conv module?
            # We assume naming convention 'conv' prefix implies conv module wrapper.
            if 'conv' in new_path[-2] or 'quant_conv' in new_path[-2]:
                 new_path[-1] = 'kernel'
                 new_path.insert(-1, 'conv') 
                 
                 # Transpose Weight
                 # PyTorch (Out, In, T, H, W) -> JAX (T, H, W, In, Out)
                 # Permute: (2, 3, 4, 1, 0)
                 # Exception: 1x1 convs? (Out, In, 1, 1, 1) -> (1, 1, 1, In, Out). Same permute.
                 # Transpose Weight
                 # PyTorch (Out, In, T, H, W) -> JAX (T, H, W, In, Out)
                 # Permute: (2, 3, 4, 1, 0)
                 if val.ndim == 5:
                     val = np.transpose(val, (2, 3, 4, 1, 0))
                 elif val.ndim == 4:
                     # Maybe (Out, In, H, W) ?
                     # JAX: (H, W, In, Out)
                     val = np.transpose(val, (2, 3, 1, 0))
                     # Add T dim -> (1, H, W, In, Out) = (0, ...)
                     val = np.expand_dims(val, 0)
                 elif val.ndim == 2:
                     # (Out, In) -> (1, 1, 1, In, Out)
                     val = np.transpose(val, (1, 0))
                     val = np.expand_dims(val, (0, 1, 2))
                 else:
                     print(f"WARNING: Unexpected conv weight shape {val.shape} for {pt_key}. Leaving as is.")
                 
            elif 'norm' in new_path[-2]:
                 # GroupNorm
                 # PyTorch: weight / gamma
                 # JAX: weight (My custom GroupNorm uses .weight)
                 if val.ndim > 1:
                     val = val.reshape(-1)
            elif 'upsamplers' in pt_key: 
                 # upsampler conv handle above?
                 pass
                 
        if param_name == 'bias':
            # Check if Conv
            if 'conv' in new_path[-2] or 'quant_conv' in new_path[-2]:
                 new_path.insert(-1, 'conv') # conv1.bias -> conv1.conv.bias
            elif 'norm' in new_path[-2]:
                 # Flatten bias just in case
                 if val.ndim > 1:
                     val = val.reshape(-1)
            
        
        # Create tuple key (handling numeric strings)
        # However, nnx flat dict keys can be integers for list indices
        # PyTorch path is all strings.
        
        key_tuple = []
        for p in new_path:
             if p.isdigit():
                 key_tuple.append(int(p))
             else:
                 key_tuple.append(p)
                 
        key_tuple = tuple(key_tuple)
        
        # Check alignment with dictionary structure (up_blocks list of dicts)
        # flat_state keys for up_blocks: ('decoder', 'up_blocks', 0, 'resnets', 0, ...)
        # My list of dicts: [ {'resnets': [...]} ]
        # NNX treats list as indices 0..N
        # NNX treats dict as keys 'resnets', 'upsampler' (string)
        # So ('decoder', 'up_blocks', 0, 'resnets') IS correct.
        
        if key_tuple in flat_state:
             arr = jnp.array(val)
             # REMOVED Weight Sharding
             # if mesh is not None and ...
                  
             new_state_dict[key_tuple] = arr
        else:
             # Try generic debug?
             pass 
             # print(f"Warning: Key not found in JAX model: {key_tuple}")
             
    # Update State
    # Update State
    # state = state.replace_by_flat_dict(new_state_dict) # Removed in 0.12+
    
    final_flat_state = flat_state.copy()
    final_flat_state.update(new_state_dict)
    
    new_pure_dict = flax.traverse_util.unflatten_dict(final_flat_state)
    state = nnx.State(new_pure_dict)
    
    # Merge
    model = nnx.merge(graphdef, state)
    
    return model


