# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Minimal `dispatch_attention_fn`.

Newer diffusers releases route attention through a full multi-backend dispatcher
(flash-attn / xformers / sage / sdpa / ...) in this module. This repo only needs the
plain `torch.nn.functional.scaled_dot_product_attention` backend — everything else
(op overrides for custom kernels, e.g. torchax/TPU splash attention) is applied by
patching `torch.nn.functional.scaled_dot_product_attention` itself, so this shim just
forwards to it, doing only the (B, S, H, D) <-> (B, H, S, D) layout conversion that
callers (e.g. `QwenDoubleStreamAttnProcessor2_0`) expect.
"""

import torch


def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
    backend=None,  # noqa: ARG001 — accepted for call-site compatibility, unused
) -> torch.Tensor:
    # query/key/value arrive as (batch, seq, heads, head_dim); SDPA wants
    # (batch, heads, seq, head_dim).
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2)
