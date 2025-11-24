"""FP8 KV Cache Quantization Utilities"""

# <claudes_code_comments>
# ** Function List **
# quantize_k_cache(input_k_cache, dv, tile_size) - Quantizes KV cache to FP8 format with per-tile scaling
# dequantize_k_cache(quant_k_cache, dv, tile_size, d) - Dequantizes FP8 KV cache back to BF16
#
# ** Technical Review **
# This module implements FP8 quantization/dequantization for KV cache compression in FlashMLA.
#
# Quantization Format (656 bytes/token with default d=576, dv=512):
# 1. **NoPE Part** (512 bytes): fp8_e4m3fn quantized values (dv elements)
# 2. **Scale Factors** (16 bytes): 4 × float32 scales (one per 128-element tile)
# 3. **RoPE Part** (64 bytes): Unquantized bfloat16 values (d - dv elements)
#
# Key Design Choices:
#
# Per-Tile Scaling (tile_size=128):
# - Divides NoPE part into 4 tiles (512 / 128 = 4)
# - Each tile has independent scale factor: scale = max(abs(tile)) / 448.0
# - 448.0 = max representable value in fp8_e4m3fn (prevents overflow)
# - Quantized value = original / scale, Dequantized value = quantized * scale
# - Finer granularity than per-token → better accuracy
# - Coarser than per-element → minimal memory overhead (16 bytes)
#
# NoPE vs RoPE Split:
# - NoPE (No Position Encoding): First dv=512 dimensions, quantized to FP8
# - RoPE (Rotary Position Encoding): Last d-dv=64 dimensions, kept in BF16
# - Rationale: RoPE encodes critical position information, quantization degrades accuracy
# - Trade-off: 2× compression on 89% of data, full precision on 11%
#
# Quantization Algorithm (quantize_k_cache):
# for each tile in [0, dv, step=tile_size]:
#     scale[tile] = max(abs(input[tile:tile+128])) / 448.0
#     quantized[tile:tile+128] = input[tile:tile+128] / scale[tile]  # to fp8_e4m3fn
# rope_part = input[dv:]  # keep as bfloat16
# return concat(quantized_nope, scales, rope_part)
#
# Dequantization Algorithm (dequantize_k_cache):
# for each tile in [0, dv, step=tile_size]:
#     dequantized[tile:tile+128] = quantized[tile:tile+128] * scale[tile]  # to bfloat16
# dequantized[dv:] = rope_part
# return dequantized
#
# Memory Savings:
# - Original BF16: d × 2 bytes = 576 × 2 = 1152 bytes/token
# - Quantized FP8: dv × 1 + (dv/128) × 4 + (d-dv) × 2 = 512 + 16 + 128 = 656 bytes/token
# - Compression ratio: 1152 / 656 = 1.76× (43% reduction)
#
# Accuracy Characteristics:
# - FP8 e4m3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
# - Dynamic range: ~1e-9 to 448, suitable for normalized activations
# - Per-tile scaling adapts to local magnitude distribution
# - RoPE preservation maintains positional accuracy
# - Typical error: < 1% relative error on attention scores
#
# Usage in FlashMLA:
# - Quantize KV cache during prefill or after appending new tokens
# - Store quantized cache to save memory (critical for long contexts)
# - Kernel dequantizes on-the-fly during attention computation
# - Decoding kernel reads FP8, converts to BF16 internally, computes in BF16
# </claudes_code_comments>

import torch

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype uint8_t, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty((num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv:dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4:].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size]).max(dim=-1).values / 448.0  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
        cur_quantized_nope = (input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_quantized_nope

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv:dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4:].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_nope * cur_scales

    result = result.view(num_blocks, block_size, 1, d)
    return result
