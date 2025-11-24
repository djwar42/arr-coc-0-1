# <claudes_code_comments>
# ** Function List **
# ceil_div(x, y) - Ceiling integer division for alignment calculations
# align(x, y) - Align x to next multiple of y
# ceil_to_ue8m0(x) - Round tensor values up to nearest power-of-2 for UE8M0 format
# per_token_cast_to_fp8(x, use_ue8m0) - Quantize to FP8 with per-token scaling (128-element blocks)
# per_channel_cast_to_fp8(x, use_ue8m0) - Quantize to FP8 with per-channel scaling
# per_block_cast_to_fp8(x, use_ue8m0) - Quantize to FP8 with per-block (128x128) scaling
# per_custom_dims_cast_to_fp8(x, dims, use_ue8m0) - Quantize to FP8 with custom dimension scaling
#
# ** Technical Review **
# FP8 quantization utilities implementing E4M3 (4-bit exponent, 3-bit mantissa) conversion with scaling factors.
#
# Quantization strategy: FP8 E4M3 has limited dynamic range (~448 max value). Scaling factors (SF) normalize
# values into representable range: x_fp8 = x_fp32 * (1/SF), where SF = amax(x) / 448.
#
# Granularity levels:
# - per_token: 128-element blocks along N-axis, SF shape [M, N/128], for activation quantization
# - per_channel: 128-element blocks along M-axis, SF shape [M/128, N], for weight quantization
# - per_block: 128x128 blocks, SF shape [M/128, N/128], finest granularity for accuracy
# - per_custom_dims: User-specified dimensions for flexible quantization schemes
#
# UE8M0 format: SM100 requires power-of-2 scales packed as unsigned 8-bit exponents (4 per torch.int).
# ceil_to_ue8m0() rounds SF up to nearest power-of-2 via: 2^ceil(log2(abs(SF))).
# This enables efficient hardware decompression at cost of slight overestimation.
#
# Alignment: Functions pad to 128-element boundaries (GEMM block size) then slice to original shape.
# Clamping: amax clamped to 1e-4 minimum prevents division by zero for zero/tiny tensors.
#
# Mathematical foundation: Affine quantization Q = clamp(round(x/scale), -max, max) where
# scale = amax(x) / 448 for E4M3. Dequantization: x_approx = Q * scale.
# </claudes_code_comments>

import torch
from typing import Tuple


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, 128)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf


def per_channel_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(0) % 128 == 0
    m, n = x.shape
    x_view = x.view(-1, 128, n)
    x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()
