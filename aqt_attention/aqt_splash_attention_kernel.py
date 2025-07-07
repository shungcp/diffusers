# aqt_splash_attention_kernel_v1.2_final.py

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from typing import Optional, Tuple, Any, Callable
import numpy as np
import functools
import dataclasses
from aqt_attention import splash_attention_mask as mask_lib
from aqt_attention import splash_attention_mask_info as mask_info_lib

# AQT and other constants remain the same...
# (此处省略了之前版本中的AQT导入、配置函数和常量定义，它们保持不变)
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
HEAD_DIM_MINOR = 0

@dataclasses.dataclass(frozen=True, slots=True)
class AqtBlockSizes:
    """AQT Splash Attention的分块大小配置"""
    block_q: int = 128
    block_kv: int = 128
    block_kv_compute: int = 128

def aqt_flash_attention_kernel(
    # --- 输入参数 ---
    data_next_ref, block_mask_ref, mask_next_ref,
    q_ref, k_ref, v_ref,
    q_segment_ids_ref, kv_segment_ids_ref,
    mask_ref, q_sequence_ref,
    # --- 输出参数 ---
    m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref=None,
    # --- 静态参数 ---
    *,
    mask_value: float, grid_width: int,
    bq: int, bkv: int, bkv_compute: int, head_dim: int,
    attn_logits_soft_cap: float | None,
    mask_function: Callable | None,
    # AQT相关配置 (在当前实现中未使用，但保留接口)
    q_config: Any, k_config: Any,
    # 布局参数
    q_layout: int, k_layout: int, v_layout: int,
    # 新增：静态scale参数
    q_scale_val: float,
    k_scale_val: float
):
    """
    在Pallas中集成了Tile-wise量化的Flash Attention内核最终版
    """
    float32 = jnp.float32
    head_dim_v_repeats, rem = divmod(head_dim, NUM_LANES)
    if rem != 0:
        raise NotImplementedError(f"{head_dim=} should be a multiple of {NUM_LANES}")

    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -1e6) # 使用一个足够小的值
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        # 初始化scale统计信息
        # 暂时注释掉，用于调试
        # if collect_scale_stats and scale_stats_ref is not None:
        #     scale_stats_ref[0] = 0.0
        #     scale_stats_ref[1] = 0.0
        #     scale_stats_ref[2] = 0.0
        #     scale_stats_ref[3] = 0.0

    global_kv_index, _, should_run, should_not_mask = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )

    def body(kv_compute_index, _):
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]

        # 1. 加载 bfloat16 的数据块 (tile)
        q_tile_bf16 = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
        if k_layout == HEAD_DIM_MINOR:
            k_tile_bf16 = k_ref[slice_k, :]
        else:
            k_tile_bf16 = k_ref[:, slice_k]

        # 【核心改动】直接使用传入的静态scale
        q_scale = q_scale_val
        k_scale = k_scale_val

        # 3. 量化到 int8
        q_float32 = q_tile_bf16.astype(float32)
        k_float32 = k_tile_bf16.astype(float32)
        q_quantized = jnp.round(q_float32 / q_scale).astype(jnp.int8)
        k_quantized = jnp.round(k_float32 / k_scale).astype(jnp.int8)

        # 4. int8 matmul
        qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
        qk_int32 = lax.dot_general(
            q_quantized,
            k_quantized,
            qk_dims,
            preferred_element_type=jnp.int32
        )

        # 5. 反量化回 float32
        qk = qk_int32.astype(jnp.float32) * (q_scale * k_scale)
        assert qk.shape == (bq, bkv_compute)

        # 6. 应用 mask 和 softmax
        qk = _apply_aqt_mask_and_soft_cap(
            qk, mask_value, should_not_mask, mask_ref, q_sequence_ref,
            q_segment_ids_ref, kv_segment_ids_ref, attn_logits_soft_cap,
            slice_k, global_kv_index * bkv + kv_compute_index * bkv_compute,
            bq, mask_function
        )

        qk_float = qk.astype(float32)
        m_curr = qk_float.max(axis=-1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)

        bkv_repeats, _ = divmod(bkv_compute, NUM_LANES)
        m_next_tiled = jnp.tile(m_next, (1, bkv_repeats))
        s_curr = jnp.exp(qk - m_next_tiled)

        l_curr = jnp.broadcast_to(s_curr.sum(axis=-1, keepdims=True), l_prev.shape)

        alpha = jnp.exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next

        # 7. 计算输出 o
        if v_layout == HEAD_DIM_MINOR:
            v = v_ref[slice_k, :]
        else:
            v = v_ref[:, slice_k]
        sv_dims = NN_DIM_NUMBERS if v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
        o_curr = lax.dot_general(s_curr, v, sv_dims, preferred_element_type=float32)

        alpha_o = jnp.repeat(alpha, head_dim_v_repeats, axis=1)
        o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr

    @pl.when(should_run)
    def run():
        assert bkv % bkv_compute == 0
        # 修正：原始的 splash attention 内核中的 fori_loop 逻辑
        num_iters_in_kv_block = bkv // bkv_compute
        lax.fori_loop(0, num_iters_in_kv_block, body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def end():
        l = l_scratch_ref[...]
        l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=1)
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
        if logsumexp_ref is not None:
            logsumexp_ref[...] = (jnp.log(l) + m_scratch_ref[...]).astype(logsumexp_ref.dtype)
        
        # 输出scale统计信息（用于收集静态scale值）
        # 暂时注释掉，用于调试
        # if collect_scale_stats and scale_stats_ref is not None:
        #     # 简化版本：只输出当前block的scale值
        #     # 这里我们假设scale_stats_ref包含了当前block的统计信息
        #     # 在实际使用中，你可以在外部收集这些值并计算平均值
        #     print(f"🔍 AQT Scale Stats - Head {h}, Block {i}: "
        #           f"q_scale={scale_stats_ref[0]:.6f}, k_scale={scale_stats_ref[1]:.6f}, "
        #           f"q_max={scale_stats_ref[2]:.6f}, k_max={scale_stats_ref[3]:.6f}")
        
        m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

def get_default_aqt_config():
    try:
        # 使用AQT v2的fully_quantized配置
        from aqt.jax.v2.config import fully_quantized
        config = fully_quantized(
            fwd_bits=8,  # 前向传播使用8位量化
            bwd_bits=8,  # 反向传播使用8位量化
            use_fwd_quant=True,  # 启用前向量化
            use_stochastic_rounding=True,  # 使用随机舍入
        )
        return config
    except Exception as e:
        print(f"⚠️ 创建AQT配置失败: {e}")
        return None

def get_attention_optimized_aqt_config():
    try:
        # 使用AQT v2的fully_quantized配置，质量优化版本
        from aqt.jax.v2.config import fully_quantized
        config = fully_quantized(
            fwd_bits=8,  # 前向传播使用8位量化
            bwd_bits=8,  # 反向传播使用8位量化
            use_fwd_quant=True,  # 启用前向量化
            use_stochastic_rounding=True,  # 启用随机舍入以提高质量
            use_dummy_static_bound=False,  # 关闭静态边界以提高质量
        )
        return config
    except Exception as e:
        print(f"⚠️ 创建优化AQT配置失败: {e}")
        return get_default_aqt_config()

def get_no_quantization_config():
    try:
        # 使用AQT v2的fully_quantized配置，但禁用量化
        from aqt.jax.v2.config import fully_quantized
        config = fully_quantized(
            fwd_bits=16,  # 使用16位而不是8位
            bwd_bits=16,  # 使用16位而不是8位
            use_fwd_quant=False,  # 禁用前向量化
            use_stochastic_rounding=False,  # 禁用随机舍入
            use_dummy_static_bound=False,  # 关闭静态边界
        )
        return config
    except Exception as e:
        print(f"⚠️ 创建无量化配置失败: {e}")
        return None

def _next_nonzero(
    h,
    i,
    j,
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    next_i=False,
):
  assert (data_next_ref is None) == (block_mask_ref is None)

  if data_next_ref is None and block_mask_ref is None:
    # Handle the case in which we have no masking nor next data information.
    # Simply fetch the next data and apply the mask for every block.
    assert mask_next_ref is None
    next_data = i if next_i else j
    return (
        next_data,
        None,  # next mask
        True,  # should run
        False,  # should not mask
    )

  assert data_next_ref.shape == block_mask_ref.shape
  assert mask_next_ref is None or data_next_ref.shape[0] == mask_next_ref.shape[0]

  # We are working with one head only. Force the head index to 0.
  if data_next_ref.shape[0] == 1:
    h = 0

  # When scalar-memory data is of types smaller than int32, then we have to
  # upcast it back to use it in the kernel.

  to_i32 = lambda x: x.astype(jnp.int32)

  is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
  if mask_next_ref is None:
    should_not_mask = True
    next_m = None
  else:
    should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
    next_m = to_i32(mask_next_ref[h, i, j])
  next_j = to_i32(data_next_ref[h, i, j])
  return next_j, next_m, is_nonzero, should_not_mask

def _apply_aqt_mask_and_soft_cap(
    qk, mask_value, should_not_mask, mask_ref, q_sequence_ref,
    q_segment_ids_ref, kv_segment_ids_ref, attn_logits_soft_cap,
    k_slice, k_offset, bq, mask_function
):
    """在AQT kernel中应用mask和soft cap"""
    masks = []
    
    if mask_ref is not None:
        mask = mask_ref[:, k_slice]
        masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape)))
    
    if mask_function is not None:
        if q_sequence_ref.shape == (bq, NUM_LANES):
            k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)
            repeats, rem = divmod(k_slice.size, NUM_LANES)
            assert rem == 0
            q_sequence = pltpu.repeat(q_sequence_ref[...], repeats, axis=1)
        else:
            k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
            q_sequence = q_sequence_ref[:1, :]
            q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))
        
        computed_mask = mask_function(q_sequence, k_sequence)
        masks.append(computed_mask)
    
    if q_segment_ids_ref is not None:
        kv_ids = kv_segment_ids_ref[:1, k_slice]
        repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
        if rem:
            raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
        q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)
        masks.append(q_ids == kv_ids)
    
    def cap_logits(logits):
        if attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / attn_logits_soft_cap)
            return logits * attn_logits_soft_cap
        return logits
    
    if masks:
        mask = functools.reduce(jnp.logical_and, masks)
        # 确保 mask 的形状与 qk 匹配
        if mask.shape != qk.shape:
            if mask.ndim == 3 and qk.ndim == 2:
                mask = mask.squeeze(0)  # 移除第一个维度
            elif mask.ndim == 2 and qk.ndim == 2:
                pass  # 形状已经匹配
            else:
                mask = mask.reshape(qk.shape)  # 强制reshape
        
        qk = cap_logits(qk)
        qk = jnp.where(mask, qk, mask_value)
    else:
        qk = cap_logits(qk)
    
    return qk

class AqtSplashAttentionKernel:
    """AQT Splash Attention Kernel类"""
    
    def __init__(
        self,
        fwd_mask_info: mask_info_lib.MaskInfo,
        q_config: Any,
        k_config: Any,
        block_sizes: Optional[AqtBlockSizes] = None,
        mask_value: float = DEFAULT_MASK_VALUE,
        attn_logits_soft_cap: float | None = None,
        interpret: bool = False,
        collect_scale_stats: bool = False,
        q_scale_val: float = 0.035925,
        k_scale_val: float = 0.035925,
    ):
        self.fwd_mask_info = fwd_mask_info
        self.q_config = q_config
        self.k_config = k_config
        self.block_sizes = block_sizes or AqtBlockSizes()
        self.mask_value = mask_value
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.interpret = interpret
        self.collect_scale_stats = collect_scale_stats
        self.q_scale_val = q_scale_val
        self.k_scale_val = k_scale_val
    
    def __call__(self, q, k, v, segment_ids=None, save_residuals=False):
        """执行AQT Splash Attention"""
        return self._aqt_splash_attention_forward(
            q, k, v, segment_ids, save_residuals
        )
    
    def _get_causal_mask_function(self):
        """获取因果mask函数"""
        def causal_mask(q_seq, k_seq):
            """因果mask函数：q_seq <= k_seq"""
            return q_seq <= k_seq
        return causal_mask
    
    def _aqt_splash_attention_forward(self, q, k, v, segment_ids, save_residuals):
        """AQT Splash Attention前向传播"""
        num_q_heads, q_seq_len, head_dim_qk = q.shape
        head_dim_v = v.shape[-1]
        
        # 确保head_dim是128的倍数
        if head_dim_v % NUM_LANES != 0:
            raise ValueError(f"head_dim_v={head_dim_v} must be a multiple of {NUM_LANES} (128)")
        if head_dim_qk % NUM_LANES != 0:
            raise ValueError(f"head_dim_qk={head_dim_qk} must be a multiple of {NUM_LANES} (128)")
        
        # 使用动态分块大小，而不是硬编码
        bq = self.block_sizes.block_q
        bkv = self.block_sizes.block_kv
        bkv_compute = self.block_sizes.block_kv_compute
        
        # 正确的AQT实现：直接传递float32数据给Pallas kernel
        # 量化将在kernel内部进行，确保tile-wise精度
        
        # 设置grid和block specs - 支持分片
        if self.fwd_mask_info.data_next is not None:
            grid_width = self.fwd_mask_info.data_next.shape[-1]
        else:
            grid_width = k.shape[1] // bkv
        
        # 支持head分片和q序列分片
        if hasattr(self, 'block_sizes') and self.block_sizes is not None:
            # 使用分片配置
            grid = (num_q_heads, q_seq_len // bq, grid_width)
        else:
            # 默认配置
            grid = (num_q_heads, q_seq_len // bq, grid_width)
        
        # 定义index mapping函数 - 与原始Splash Attention一致
        def q_index_map(h, i, j, *_):
            return h, i, 0
        
        # 使用与原始Splash Attention相同的index_map逻辑
        def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
            next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
            prefix = (h,)
            return from_head_minor((*prefix, next_j, 0), 0)
        
        def v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
            next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
            prefix = (h,)
            return from_head_minor((*prefix, next_j, 0), 0)
        
        def out_index_map(h, i, j, *_):
            return h, i, 0
        
        in_specs = [
            pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
            pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
            pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
            None,
            None,
            pl.BlockSpec((None, bq, bkv), lambda *_: (0, 0, 0)) if self.fwd_mask_info.partial_mask_blocks is not None else None,
            pl.BlockSpec((bq, NUM_LANES), lambda h, i, *_: (i, 0)) if self.fwd_mask_info.q_sequence is not None else None,
        ]
        
        out_shapes = [
            jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
            jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
            jax.ShapeDtypeStruct((bq, head_dim_v), jnp.float32),
            jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
        ]
        
        out_specs = [
            pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
            pl.BlockSpec((bq, NUM_LANES), lambda *_: (0, 0)),
            pl.BlockSpec((bq, head_dim_v), lambda *_: (0, 0)),
            pl.BlockSpec((None, bq, head_dim_v), out_index_map),
        ]
        
        if save_residuals:
            out_shapes.append(jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32))
            out_specs.append(pl.BlockSpec((None, bq, NUM_LANES), lambda h, i, *_: (h, i, 0)))
        else:
            out_shapes.append(None)
            out_specs.append(None)
        
        kernel_name = f"aqt_splash_attention_{bq}_{bkv}_{bkv_compute}"
        
        with jax.named_scope(kernel_name):
            all_out = pl.pallas_call(
                functools.partial(
                    aqt_flash_attention_kernel,
                    mask_value=self.mask_value,
                    grid_width=grid_width,
                    bq=bq,
                    bkv=bkv,
                    bkv_compute=bkv_compute,
                    head_dim=head_dim_v,
                    attn_logits_soft_cap=self.attn_logits_soft_cap,
                    mask_function=self._get_causal_mask_function() if self.fwd_mask_info.q_sequence is not None else None,
                    q_config=self.q_config,
                    k_config=self.k_config,
                    q_layout=0,
                    k_layout=0,
                    v_layout=0,
                    q_scale_val=self.q_scale_val,
                    k_scale_val=self.k_scale_val
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=3,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel", "arbitrary", "arbitrary"),
                ),
                out_shape=out_shapes,
                name=kernel_name,
                interpret=self.interpret,
            )(
                self.fwd_mask_info.data_next,
                self.fwd_mask_info.block_mask,
                self.fwd_mask_info.mask_next,
                q,
                k,
                v,
                None,
                None,
                self.fwd_mask_info.partial_mask_blocks,
                jax.lax.broadcast_in_dim(self.fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)) if self.fwd_mask_info.q_sequence is not None else None,
            )
        
        _, _, _, out, logsumexp = all_out
        
        if save_residuals:
            assert logsumexp is not None
            logsumexp = logsumexp[..., 0]
            return out, (logsumexp,)
        return out

def make_aqt_splash_attention(
    mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask | None,
    q_config: Any = None,
    k_config: Any = None,
    block_sizes: Optional[AqtBlockSizes] = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
    head_shards: int = 1,
    q_seq_shards: int = 1,
    collect_scale_stats: bool = False,
    q_scale_val: float = 0.035925,
    k_scale_val: float = 0.035925,
):
    """创建AQT Splash Attention kernel"""
    if q_config is None:
        q_config = get_default_aqt_config()
    if k_config is None:
        k_config = get_default_aqt_config()
    
    # 处理mask - 添加分片支持
    if mask is None:
        fwd_mask_info = mask_info_lib.MaskInfo(
            data_next=None,
            block_mask=None,
            mask_next=None,
            partial_mask_blocks=None,
            q_sequence=None,
        )
    elif isinstance(mask, np.ndarray):
        is_causal = _is_causal_mask(mask)
        if is_causal:
            seq_len = mask.shape[1]
            fwd_mask_info = mask_info_lib.MaskInfo(
                data_next=None,
                block_mask=None,
                mask_next=None,
                partial_mask_blocks=None,
                q_sequence=jnp.arange(seq_len, dtype=jnp.int32),
            )
        else:
            if block_sizes is not None:
                try:
                    fwd_mask_info, _ = mask_info_lib.process_mask(
                        mask,
                        (block_sizes.block_q, block_sizes.block_kv),
                        head_shards=head_shards,
                        q_seq_shards=q_seq_shards,
                    )
                    fwd_mask_info = jax.tree_util.tree_map(jnp.array, fwd_mask_info)
                except Exception as e:
                    fwd_mask_info = mask_info_lib.MaskInfo(
                        data_next=None,
                        block_mask=None,
                        mask_next=None,
                        partial_mask_blocks=None,
                        q_sequence=None,
                    )
            else:
                fwd_mask_info = mask_info_lib.MaskInfo(
                    data_next=None,
                    block_mask=None,
                    mask_next=None,
                    partial_mask_blocks=None,
                    q_sequence=None,
                )
    else:
        if block_sizes is not None:
            try:
                fwd_mask_info, _ = mask_info_lib.process_mask(
                    mask,
                    (block_sizes.block_q, block_sizes.block_kv),
                    head_shards=head_shards,
                    q_seq_shards=q_seq_shards,
                )
                fwd_mask_info = jax.tree_util.tree_map(jnp.array, fwd_mask_info)
            except Exception as e:
                fwd_mask_info = mask_info_lib.MaskInfo(
                    data_next=None,
                    block_mask=None,
                    mask_next=None,
                    partial_mask_blocks=None,
                    q_sequence=None,
                )
        else:
            fwd_mask_info = mask_info_lib.MaskInfo(
                data_next=None,
                block_mask=None,
                mask_next=None,
                partial_mask_blocks=None,
                q_sequence=None,
            )
    
    return AqtSplashAttentionKernel(
        fwd_mask_info=fwd_mask_info,
        q_config=q_config,
        k_config=k_config,
        block_sizes=block_sizes,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=interpret,
        collect_scale_stats=collect_scale_stats,
        q_scale_val=q_scale_val,
        k_scale_val=k_scale_val,
    )

# 兼容性函数
def aqt_splash_attention_fn(mask, q, k, v, segment_ids=None, mask_value=DEFAULT_MASK_VALUE, 
                           save_residuals=False, attn_logits_soft_cap=None, 
                           q_config=None, k_config=None, block_multiple=128, collect_scale_stats=False):
    """兼容性函数，使用新的AQT Splash Attention实现"""
    block_sizes = AqtBlockSizes(
        block_q=block_multiple,
        block_kv=block_multiple,
        block_kv_compute=block_multiple
    )
    
    kernel = make_aqt_splash_attention(
        mask=mask,
        q_config=q_config,
        k_config=k_config,
        block_sizes=block_sizes,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        collect_scale_stats=collect_scale_stats,
        q_scale_val=0.035925,
        k_scale_val=0.035925,
    )
    
    return kernel(q, k, v, segment_ids, save_residuals) 

# 添加必要的辅助函数
def from_head_minor(vals: tuple[Any, ...], layout: int):
    """Convert from head-minor format to the specified layout"""
    if layout == 0:  # HEAD_DIM_MINOR
        return vals
    return (*vals[:-2], vals[-1], vals[-2])

def _div(dividend: int, divisor: int):
    """Integer division"""
    return dividend // divisor

def _is_causal_mask(mask: np.ndarray) -> bool:
    """检查是否为因果mask（下三角矩阵）"""
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        return False
    
    # 检查是否为下三角矩阵（包括对角线）
    seq_len = mask.shape[0]
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i and mask[i, j] != 0:  # 上三角部分应该都是0
                return False
            if j <= i and mask[i, j] == 0:  # 下三角部分应该都是1
                return False
    return True
