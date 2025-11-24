"""
FP8 Quantization Mechanics - How to actually convert FP16/BF16 to FP8

<karpathys_code_comments>
** This File's Role **
The low-level quantization functions used by the training loop. Takes FP16/BF16 tensors and
converts them to FP8 E4M3 or E5M2 formats with proper scaling.

** Function List **
quantize_fp8_e4m3(tensor, scale) - Quantize to E4M3 (4-bit exp, 3-bit mantissa)
quantize_fp8_e5m2(tensor, scale) - Quantize to E5M2 (5-bit exp, 2-bit mantissa)
compute_scale_factor(tensor) - Find optimal scale for tensor's dynamic range
dequantize_fp8(tensor, scale) - Convert FP8 back to FP16/BF16

** Technical Deep Dive **
FP8 comes in two flavors:
- E4M3: More precision, smaller range (good for weights/activations)
- E5M2: Less precision, larger range (good for gradients)

The key is per-tensor scaling: find the max value, scale to fit FP8 range, store the scale factor.

Karpathy: This is just careful numerical engineering. The math is straightforward but critical
to get right. One mistake and your model explodes or underflows.
</karpathys_code_comments>
"""

import torch

def quantize_fp8_e4m3(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize tensor to FP8 E4M3 format with scaling"""
    # Karpathy: E4M3 range is ~[-448, 448]. Find max absolute value to scale.
    abs_max = tensor.abs().max()

    # Karpathy: Scale factor maps tensor's range to FP8's range
    scale = 448.0 / abs_max.clamp(min=1e-12)  # Avoid division by zero

    # Karpathy: Scale tensor, clip to FP8 range, convert dtype
    scaled = tensor * scale
    quantized = scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    return quantized, scale.item()

# Karpathy: Just compute scale from tensor statistics, apply it, clip to range. Simple!
