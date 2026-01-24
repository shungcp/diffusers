# Copyright 2024 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.accelerate_utils import apply_forward_hook
from ..activations import get_activation
from ..attention_processor import Attention
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_causal_attention_mask(
    num_frames: int, height_width: int, dtype: torch.dtype, device: torch.device, batch_size: int = None
) -> torch.Tensor:
    indices = torch.arange(1, num_frames + 1, dtype=torch.int32, device=device)
    indices_blocks = indices.repeat_interleave(height_width)
    x, y = torch.meshgrid(indices_blocks, indices_blocks, indexing="xy")
    mask = torch.where(x <= y, 0, -float("inf")).to(dtype=dtype)

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class HunyuanVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class HunyuanVideoUpsampleCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        upsample_factor: Tuple[float, float, float] = (2, 2, 2),
        match_channel: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.upsample_factor = upsample_factor
        self.match_channel = match_channel

        if match_channel:
             # Input: C -> Output: C * factor? No, Weights are [8192, 1024].
             # 1024 input, 8192 output. 8192 = 1024 * 8.
             # So we assume pixel shuffle expansion factor is 2*2*2 = 8
             channel_factor = int(upsample_factor[0] * upsample_factor[1] * upsample_factor[2])
             conv_out_channels = out_channels * channel_factor
             self.conv = HunyuanVideoCausalConv3d(in_channels, conv_out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        else:
             self.conv = HunyuanVideoCausalConv3d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Hunyuan 1.5 "PixelShuffle" upsample logic (match_channel=True)
        if self.match_channel:
             # Weights are (C_out*8, C_in, k, k, k). 
             # Standard flow: Conv (Expansion) -> PixelShuffle (DepthToSpace)
             
             # 1. Conv
             hidden_states = self.conv(hidden_states)
             
             # 2. PixelShuffle (DepthToSpace) 3D
             b, c_total, t, h, w = hidden_states.shape
             ft, fh, fw = int(self.upsample_factor[0]), int(self.upsample_factor[1]), int(self.upsample_factor[2])
             
             # Output channels
             c_out = c_total // (ft * fh * fw)
             
             # (B, C*r3, T, H, W) -> (B, C, r, r, r, T, H, W)
             hidden_states = hidden_states.view(b, c_out, ft, fh, fw, t, h, w)
             # Permute to (B, C, T, r, H, r, W, r) -> (B, C, T*r, H*r, W*r)
             hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c_out, t * ft, h * fh, w * fw)
             
             return hidden_states

        # Standard Interpolate -> Conv logic
        num_frames = hidden_states.size(2)

        first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
        first_frame = F.interpolate(
            first_frame.squeeze(2), scale_factor=self.upsample_factor[1:], mode="nearest"
        ).unsqueeze(2)

        if num_frames > 1:
            # See: https://github.com/pytorch/pytorch/issues/81665
            # Unless you have a version of pytorch where non-contiguous implementation of F.interpolate
            # is fixed, this will raise either a runtime error, or fail silently with bad outputs.
            # If you are encountering an error here, make sure to try running encoding/decoding with
            # `vae.enable_tiling()` first. If that doesn't work, open an issue at:
            # https://github.com/huggingface/diffusers/issues
            other_frames = other_frames.contiguous()
            other_frames = F.interpolate(other_frames, scale_factor=self.upsample_factor, mode="nearest")
            hidden_states = torch.cat((first_frame, other_frames), dim=2)
        else:
            hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoDownsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        stride=2,
        match_channel: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels
        self.match_channel = match_channel
        
        # If match_channel, maybe we change out_channels?
        # WEIGHTS: [64, 128, 3, 3, 3] for input 128.
        # This is strictly outputting 64. 
        # Standard block out_channels[0] is 128.
        # So we force output to be smaller? 
        # Actually, let's respect the user-provided shapes.
        # BUT we don't know the logic for sure. 
        # Hypothesis: if match_channel, we assume out_channels calculation is overridden or specific.
        # For now, let's rely on the block passing the correct `out_channels` ? 
        # No, block passes `out_channels` which comes from `block_out_channels`.
        # `block_out_channels[0]` is 128.
        
        # HACK: If match_channel is True, for the FIRST downsampler (128->?), maybe we blindly trust the weight loader?
        # No, we must define the layer.
        
        # Let's assume if match_channel=True, we use a specific ratio?
        # 128 -> 64 (0.5x)?
        
        # OR: PixelUnshuffle?
        # If we use PixelUnshuffle, channels increase.
        
        # Re-read weights: [64, 128]. 
        # This is just a Conv that reduces channels.
        # Normal DownsampleCausal3D:
        # self.conv = HunyuanVideoCausalConv3d(channels, out_channels ... )
        # If channels=128, out_channels=128 (default).
        # We need out_channels=64.
        
        if self.match_channel and out_channels == channels:
             pass

        if match_channel:
             # SpaceToDepth logic:
             # 1. Conv reduces scale.
             # 2. PixelUnshuffle increases scale.
             # Result: Downsample.
             
             # Calculate expected Unshuffle factor
             stride_factor = stride[0] * stride[1] * stride[2]
             
             # For SpaceToDepth, C_in_unshuffle = C_out_final // stride_factor
             # So Conv must output C_in_unshuffle.
             
             conv_out_channels = out_channels // stride_factor
             
             # Force stride=1 for Conv because Unshuffle handles downsampling
             self.conv = HunyuanVideoCausalConv3d(channels, conv_out_channels, kernel_size, stride=1, padding=padding, bias=bias)
             self.stride = stride # save for forward
        else:
            self.conv = HunyuanVideoCausalConv3d(channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.match_channel:
            hidden_states = self.conv(hidden_states)
            
            # PixelUnshuffle (SpaceToDepth)
            # (B, C, T, H, W) -> (B, C*st*sh*sw, T/st, H/sh, W/sw)
            b, c, t, h, w = hidden_states.shape
            try:
                st, sh, sw = self.stride
            except:
                # Fallback if stride is int
                st, sh, sw = (1, self.stride, self.stride) if isinstance(self.stride, int) else self.stride

            hidden_states = hidden_states.view(b, c, t // st, st, h // sh, sh, w // sw, sw)
            hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c * st * sh * sw, t // st, h // sh, w // sw)
            return hidden_states
            
        return self.conv(hidden_states)


class HunyuanVideoResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3, 1, 0)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return hidden_states


class HunyuanVideoMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanVideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.resnets[0], hidden_states)

            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if attn is not None:
                    batch_size, num_channels, num_frames, height, width = hidden_states.shape
                    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                    attention_mask = prepare_causal_attention_mask(
                        num_frames, height * width, hidden_states.dtype, hidden_states.device, batch_size=batch_size
                    )
                    hidden_states = attn(hidden_states, attention_mask=attention_mask)
                    hidden_states = hidden_states.unflatten(1, (num_frames, height, width)).permute(0, 4, 1, 2, 3)

                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

        else:
            hidden_states = self.resnets[0](hidden_states)

            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if attn is not None:
                    batch_size, num_channels, num_frames, height, width = hidden_states.shape
                    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                    attention_mask = prepare_causal_attention_mask(
                        num_frames, height * width, hidden_states.dtype, hidden_states.device, batch_size=batch_size
                    )
                    hidden_states = attn(hidden_states, attention_mask=attention_mask)
                    hidden_states = hidden_states.unflatten(1, (num_frames, height, width)).permute(0, 4, 1, 2, 3)

                hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
        downsample_match_channel: bool = False,
        downsample_out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideoDownsampleCausal3D(
                        out_channels,
                        out_channels=downsample_out_channels or out_channels,
                        padding=downsample_padding,
                        stride=downsample_stride,
                        match_channel=downsample_match_channel,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: Tuple[int, int, int] = (2, 2, 2),
        upsample_match_channel: bool = False,
        upsample_out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideoUpsampleCausal3D(
                        out_channels,
                        out_channels=upsample_out_channels or out_channels,
                        upsample_factor=upsample_scale_factor,
                        match_channel=upsample_match_channel,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideoEncoder3D(nn.Module):
    r"""
    Causal encoder for 3D video-like data introduced in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        downsample_match_channel: bool = False,
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type != "HunyuanVideoDownBlock3D":
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(temporal_compression_ratio))

            if temporal_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block
                )
            elif temporal_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {temporal_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            downsample_out_channels = output_channel
            if downsample_match_channel and bool(add_spatial_downsample or add_time_downsample):
                 if i + 1 < len(block_out_channels):
                      downsample_out_channels = block_out_channels[i+1]

            down_block = HunyuanVideoDownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_stride=downsample_stride,
                downsample_padding=0,
                downsample_match_channel=downsample_match_channel,
                downsample_out_channels=downsample_out_channels,
            )

            self.down_blocks.append(down_block)

            if downsample_match_channel and bool(add_spatial_downsample or add_time_downsample):
                 output_channel = downsample_out_channels

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)

            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)

            hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoDecoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data introduced in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        upsample_match_channel: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1)
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        prev_output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type != "HunyuanVideoUpBlock3D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                if upsample_match_channel:
                     # Hunyuan 1.5 logic: Early Time Upsampling (Inverse of Late Downsampling)
                     add_time_upsample = bool(i < num_time_upsample_layers)
                else:
                     add_time_upsample = bool(
                         i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block
                     )
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)

            upsample_out_channels = output_channel
            if upsample_match_channel:
                 if i < len(block_out_channels) - 1:
                     next_channel = reversed_block_out_channels[i+1]
                     upsample_out_channels = next_channel

            up_block = HunyuanVideoUpBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                upsample_match_channel=upsample_match_channel,
                upsample_out_channels=upsample_out_channels,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = upsample_out_channels

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states)
        else:
            #print(f"DEBUG: VAE Decoder MidBlock Input: {hidden_states.shape}")
            hidden_states = self.mid_block(hidden_states)
            #print(f"DEBUG: VAE Decoder MidBlock Output: {hidden_states.shape}")

            for i, up_block in enumerate(self.up_blocks):
                #print(f"DEBUG: VAE Decoder UpBlock {i} Input: {hidden_states.shape}")
                # Debug weight shapes
                if hasattr(up_block, 'resnets') and len(up_block.resnets) > 0:
                     w = up_block.resnets[0].norm1.weight
                     #print(f"DEBUG: VAE Decoder UpBlock {i} ResNet0 Norm1 Weight: {w.shape}")
                
                if hasattr(up_block, 'upsamplers') and up_block.upsamplers is not None:
                     w = up_block.upsamplers[0].conv.conv.weight
                     #print(f"DEBUG: VAE Decoder UpBlock {i} Upsampler0 Conv Weight: {w.shape}")

                hidden_states = up_block(hidden_states)
                #print(f"DEBUG: VAE Decoder UpBlock {i} Output: {hidden_states.shape}")

        # post-process
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKLHunyuanVideo(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [HunyuanVideo](https://huggingface.co/papers/2412.03603).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
        # Hunyuan 1.5 args
        downsample_match_channel: bool = False,
        upsample_match_channel: bool = False,
    ) -> None:
        super().__init__()
        
        # Register hook to fix 5D weights from Hunyuan 1.5 checkpoints (Conv3d 1x1x1 -> Linear)
        self._register_load_state_dict_pre_hook(self._fix_hunyuan_1_5_weights)

        self.time_compression_ratio = temporal_compression_ratio

        # Hunyuan 1.5 has 5 blocks but defaults only provided 4. Auto-extend if needed.
        if len(block_out_channels) == 5 and len(down_block_types) == 4:
            down_block_types = [
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ]
        
        self.encoder = HunyuanVideoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            downsample_match_channel=downsample_match_channel,
        )

        # Hunyuan 1.5 has 5 blocks but defaults only provided 4. Auto-extend if needed.
        if len(block_out_channels) == 5 and len(up_block_types) == 4:
             up_block_types = [
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ]

        self.decoder = HunyuanVideoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
            upsample_match_channel=upsample_match_channel,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.tile_sample_min_num_frames`), the memory requirement can be lowered.
        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12

    def _fix_hunyuan_1_5_weights(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Pre-hook to squeeze 5D weights (from Conv3d 1x1x1) to 2D for Linear layers 
        used in Attention blocks. Also swaps decoder blocks for Hunyuan 1.5.
        """
        # 0. Check for 5-layer decoder swap (Hunyuan 1.5 specific)
        # 0. Check for 5-layer decoder swap (Hunyuan 1.5 specific)
        has_block_4 = any(f"{prefix}decoder.up_blocks.4" in k for k in state_dict.keys())
        if has_block_4:
             # Check if swap is needed by looking at shapes.
             # Hunyuan 1.5 Architecture (Diffusers): up_blocks.0 should be 1024 channels.
             # Hunyuan 1.5 Checkpoint (Reversed): up_blocks.0 is 128 channels.
             # We check one representative key in block 0.
             check_key = f"{prefix}decoder.up_blocks.0.resnets.0.norm1.weight"
             should_swap = False
             
             if check_key in state_dict:
                 w = state_dict[check_key]
                 if w.shape == (128,):
                     should_swap = True
             
             if should_swap:
                 # print(f"DEBUG: Detected 5-block decoder with REVERSED weights. Swapping blocks...")
                 keys_to_swap = [
                     (f"{prefix}decoder.up_blocks.0", f"{prefix}decoder.up_blocks.4"),
                     (f"{prefix}decoder.up_blocks.1", f"{prefix}decoder.up_blocks.3"),
                 ]
                 
                 all_keys = list(state_dict.keys())
                 
                 for k1_prefix, k2_prefix in keys_to_swap:
                     for k in all_keys:
                         if k.startswith(k1_prefix):
                             suffix = k[len(k1_prefix):]
                             k2 = k2_prefix + suffix
                             if k2 in state_dict:
                                 v1 = state_dict[k]
                                 v2 = state_dict[k2]
                                 state_dict[k] = v2
                                 state_dict[k2] = v1

        # 1. Fix Attention projection weights (5D -> 2D)
        keys_to_modify = []
        for k, v in state_dict.items():
            if "attentions" in k and "weight" in k and v.ndim == 5:
                # Check if it looks like [out, in, 1, 1, 1]
                if v.shape[2:] == (1, 1, 1):
                    keys_to_modify.append(k)
        
        for k in keys_to_modify:
            state_dict[k] = state_dict[k].squeeze(-1).squeeze(-1).squeeze(-1)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_model_file,
        pretrained_model_name_or_path,
        loaded_keys,
        ignore_mismatched_sizes=False,
        assign_to_params_buffers=False,
        hf_quantizer=None,
        low_cpu_mem_usage=True,
        dtype=None,
        keep_in_fp32_modules=None,
        device_map=None,
        offload_state_dict=None,
        offload_folder=None,
        dduf_entries=None,
    ):
        # Allow loading of state dict if resolved_model_file is None
        # (This logic is usually handled by ModelMixin, but if we override we must ensure it works)
        
        # 1. Sanitize state_dict if it is already provided (e.g. from sharded loader)
        if state_dict is not None:
             model._fix_hunyuan_1_5_weights(state_dict, "", None, None, None, None, None)
             
        # 2. If state dict comes from file inside super(), we can't easily intercept it UNLESS we load it first.
        # However, ModelMixin._load_pretrained_model usually expects state_dict to be passed OR loaded from resolved_model_file.
        # If resolved_model_file is valid and state_dict is None, ModelMixin loads it.
        
        # To fix this properly, we must load the state_dict locally if it's not provided, fix it, and then pass it to super().
        
        if state_dict is None and resolved_model_file is not None:
             from diffusers.models.modeling_utils import load_state_dict
             
             # Determine paths
             paths = []
             if isinstance(resolved_model_file, list):
                 paths = resolved_model_file
             else:
                 paths = [resolved_model_file]
             
             # Check if we should intercept
             should_intercept = False
             for p in paths:
                 if p is not None and os.path.isfile(p) and not p.endswith(".index.json"):
                     should_intercept = True
                     break
            
             if should_intercept:
                 # Load ALL files into one state_dict
                 state_dict = {}
                 for p in paths:
                     if p is not None and os.path.isfile(p) and not p.endswith(".index.json"):
                         # Load shard
                         shard_sd = load_state_dict(p)
                         state_dict.update(shard_sd)
                 
                 # Fix weights
                 model._fix_hunyuan_1_5_weights(state_dict, "", None, None, None, None, None)
                 
                 # IMPORTANT: Invalidating resolved_model_file isn't enough, we must pass state_dict.
                 # ModelMixin handles "if state_dict is not None: resolved_model_file = [state_dict]"
                 pass

        return super()._load_pretrained_model(
            model,
            state_dict,
            resolved_model_file,
            pretrained_model_name_or_path,
            loaded_keys,
            ignore_mismatched_sizes,
            assign_to_params_buffers,
            hf_quantizer,
            low_cpu_mem_usage,
            dtype,
            keep_in_fp32_modules,
            device_map,
            offload_state_dict,
            offload_folder,
            dduf_entries,
        )

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_min_num_frames (`int`, *optional*):
                The minimum number of frames required for a sample to be separated into tiles across the frame
                dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
            tile_sample_stride_num_frames (`int`, *optional*):
                The stride between two consecutive frame tiles. This is to ensure that there are no tiling artifacts
                produced across the frame dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_framewise_encoding and num_frames > self.tile_sample_min_num_frames:
            return self._temporal_tiled_encode(x)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                tile = x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                tile = z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec

class AutoencoderKLHunyuanVideo15(AutoencoderKLHunyuanVideo):
    pass

