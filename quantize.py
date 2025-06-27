from torch import nn
import torch
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import torch
from torch.nn import functional as F
import torchax
import torchax.interop

class WeightOnlyPerChannelQuantizedLinear(torch.nn.Module):

  def __init__(
      self,
      in_features,
      out_features,
      bias=False,
      device=None,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    weight = torch.ones(
        (out_features, in_features), dtype=torch.int8, device=device
    )
    self.register_buffer("weight", weight)

    weight_scaler = torch.ones(
        (out_features,), dtype=torch.bfloat16, device=device
    )
    self.register_buffer("weight_scaler", weight_scaler)

    self.is_symmetric_weight = True 

    if not self.is_symmetric_weight:
      zero_point = torch.ones(
          (out_features,), dtype=torch.bfloat16, device=device
      )
      self.register_buffer("zero_point", zero_point)
    else:
      self.register_buffer("zero_point", None)

    if bias:
      bias_tensor = torch.zeros((out_features, ), dtype=torch.bfloat16, device=device)
      self.register_buffer('bias', bias_tensor)


    # Number of bits of weight tensor
    self.n_bit = 8

    # Quantize activation
    self.quantize_activation = True

    # Flag to enable dequantize weight first, then do matmul. Useful for debugging.
    self.run_fake_quantize = False

  def _load_quantized_weights(self, w_q, scale, zp=None):
    """
    Load weights quantized by 'quantize_tensor'.
    """
    self.weight, self.weight_scaler, self.zero_point = load_q_weight_helper(
        w_q, scale, zp, block_size=-1
    )

  def quantize_weight_from_nn_linear(self, weight):
    assert weight.dim() == 2, "Expect 2D weight from torch.nn.Linear."
    assert weight.shape == (
        self.out_features,
        self.in_features,
    ), f"Got unexpected weight of shape {weight.shape}, expected weight shape ({self.out_features}, {self.in_features})."
    w_q, scale, zp = quantize_tensor(
        weight, (1,), self.n_bit, self.is_symmetric_weight, block_size=-1
    )
    self._load_quantized_weights(w_q, scale, zp)

  def forward(self, inputs):
      if not self.quantize_activation:
        result = F.linear(inputs, self.weight)
        result *= self.weight_scaler
        if self.bias is not None:
          result += self.bias
        return result
      else:
        inputs, act_s, _ = quantize_tensor(inputs, reduce_axis=(2,))
        # We have to call jax because we need to specify the output dtype of dot
        # dot(int8, int8)->bf16.
        # This semantic cannot be represented in torch. The inferred output dtype
        # will be int8 in torch, causing the dot result to overflow.
        result = torchax.interop.call_jax(
            jax.lax.dot_general,
            inputs,
            self.weight,
            (((2,), (1)), ((), ())),
            None,
            jnp.bfloat16.dtype,
        )
      result = result * self.weight_scaler
      if self.quantize_activation:
        result = result * act_s
      if not self.is_symmetric_weight:
        zp_out = torch.einsum("...c,z->...z", inputs, self.zero_point)
        result = result - zp_out
      return result

def create_quantized_from_nn_linear(
    float_linear: nn.Linear
):
  obj = WeightOnlyPerChannelQuantizedLinear(
      float_linear.in_features,
      float_linear.out_features,
      float_linear.bias is not None,
      "meta",
  )
  obj.quantize_weight_from_nn_linear(float_linear.weight)
  if float_linear.bias is not None:
    obj.bias = float_linear.bias
  return obj
      

EPS = 1e-5


def quantize_tensor(
    w: torch.Tensor,
    reduce_axis: Union[Tuple[int], int],
    n_bit: int = 8,
    symmetric: bool = True,
    block_size: int = -1,
):
  """
  Quantize weight tensor w along 'reduce_axis'.

  Args:
    w: weight tensor to be quantized.
    reduce_axis: axises along which to quantize.
    n_bit: Quantize to n_bit bits. (Use int8 container for n_bits < 8).
    symmetric: Whether quantization is symmetric.
    block_size: Blocksize for blockwise quantization. -1 for per-channel quant.

  Return:
    w_q: Quantized weight in int8 container
    scale: scalar for quantized tensor
    zero_point: zero_point for quantized tensor, None if symmetric quantization
  """

  assert 0 < n_bit <= 8, "Quantization bits must be between [1, 8]."
  if isinstance(reduce_axis, int):
    reduce_axis = (reduce_axis,)

  if block_size > 0:
    axis = reduce_axis[0]
    w_shape = w.shape
    assert w_shape[axis] % block_size == 0
    w = w.reshape(w_shape[:axis] + (-1, block_size) + w_shape[axis + 1 :])
    reduce_axis = axis + 1

  max_int = 2 ** (n_bit - 1) - 1
  min_int = -(2 ** (n_bit - 1))
  if not symmetric:
    max_val = w.amax(dim=reduce_axis, keepdim=True)
    min_val = w.amin(dim=reduce_axis, keepdim=True)
    scales = (max_val - min_val).clamp(min=EPS) / float(max_int - min_int)
    zero_point = min_int - min_val / scales
  else:
    max_val = w.abs().amax(dim=reduce_axis, keepdim=True)
    max_val = max_val.clamp(min=EPS)
    scales = max_val / max_int
    zero_point = 0

  w = torch.clamp(
      torch.round(w * (1.0 / scales) + zero_point), min_int, max_int
  ).to(torch.int8)

  return w, scales, zero_point if not symmetric else None


def dequantize_tensor(w, scale, zero_point=None):
  """Dequantize tensor quantized by quantize_tensor."""
  if zero_point is not None:
    return (w - zero_point) * scale

  return w * scale


def load_q_weight_helper(w_q, scale, zp=None, block_size=-1):
  """Helper function to update the shape of quantized weight to match
  what quantized linear layer expects."""
  if block_size < 0:
    w_q = w_q.to(torch.int8)
    if zp is not None:
      zp = (zp * scale).squeeze(-1).to(torch.bfloat16)
    scale = scale.squeeze(-1).to(torch.bfloat16)
  else:
    w_q = w_q.permute(1, 2, 0).to(torch.int8)
    if zp is not None:
      zp = (zp * scale).transpose(1, 0).squeeze(-1).to(torch.bfloat16)
    scale = scale.transpose(1, 0).squeeze(-1).to(torch.bfloat16)
  return w_q, scale, zp


def quantize_model(float_model): 
  """Apply quantization to linear layers."""

  def quantize_nn_mod(float_model):
    for name, mod in float_model.named_modules():
      new_mod = None

      if isinstance(mod, torch.nn.Linear):
        new_mod = create_quantized_from_nn_linear(mod)

      if new_mod:
        setattr(float_model, name, new_mod)

  float_model.apply(quantize_nn_mod)
  return float_model
