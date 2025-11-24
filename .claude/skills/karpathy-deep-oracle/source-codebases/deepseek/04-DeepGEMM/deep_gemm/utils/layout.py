# <claudes_code_comments>
# ** Function List **
# (All functions imported from deep_gemm_cpp C++ extension)
# get_tma_aligned_size() - Calculate required TMA alignment size
# get_mk_alignment_for_contiguous_layout() - Get M/K axis alignment for grouped contiguous GEMMs
# get_mn_major_tma_aligned_tensor() - Create MN-major TMA-aligned tensor
# get_mn_major_tma_aligned_packed_ue8m0_tensor() - Create TMA-aligned tensor with FP32→UE8M0 packing
# get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor() - K-grouped version of UE8M0 packing
#
# ** Technical Review **
# Layout utilities for DeepGEMM tensor memory alignment and SM90/SM100 compatibility.
#
# TMA (Tensor Memory Accelerator) requirements: SM90+ GPUs use TMA for high-bandwidth async memory transfers.
# TMA requires specific alignment (typically 128 bytes) for optimal performance. These functions ensure
# tensors meet TMA alignment constraints.
#
# Scaling factor layout: FP8 GEMMs require per-block or per-channel scaling factors (SF).
# SM90 uses FP32 SF, SM100 uses packed UE8M0 format (4 UE8M0 values packed into torch.int).
# UE8M0 = unsigned 8-bit exponent-only format for power-of-2 scales.
#
# Grouped layout alignment: MoE grouped GEMMs (m_grouped_*_contiguous) require each expert's token batch
# to align to GEMM M block size (get_mk_alignment_for_contiguous_layout). This enables efficient
# batched execution across varying expert token counts.
#
# Aliases provided for backward compatibility (get_m/k_alignment maps to get_mk_alignment).
# </claudes_code_comments>

from deep_gemm_cpp import (
    get_tma_aligned_size,
    get_mk_alignment_for_contiguous_layout,
    get_mn_major_tma_aligned_tensor,
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor
)

# Some alias
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
