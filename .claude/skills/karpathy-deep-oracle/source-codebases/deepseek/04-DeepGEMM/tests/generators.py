# <claudes_code_comments>
# ** Function List **
# get_arch_major() - Get GPU SM architecture major version (9=SM90, 10=SM100)
# get_ue8m0_usage(kernel_type) - Determine if UE8M0 scaling factors should be used
# get_kernel_types(dtype) - Get supported kernel types (1D1D/1D2D/NoSF) for dtype and architecture
# get_major_ab(allow_a_mn_major, allow_b_mn_major) - Generate valid memory layout combinations
# enumerate_normal(dtype) - Generate test configurations for dense GEMMs (forward/backward)
# enumerate_m_grouped_contiguous(dtype) - Generate M-grouped contiguous GEMM test configs (MoE)
# enumerate_m_grouped_masked(dtype) - Generate M-grouped masked GEMM test configs (MoE decoding)
# enumerate_k_grouped_contiguous() - Generate K-grouped contiguous test configs (MoE weight backward)
# enumerate_sf_layout() - Generate scaling factor layout transformation test configs
# enumerate_k_grouped_sf_layout() - Generate K-grouped scaling factor layout test configs
# enumerate_transpose() - Generate transpose operation test configs
# generate_normal(...) - Create test tensors for dense GEMM with quantization
# generate_m_grouped_contiguous(...) - Create test tensors for M-grouped contiguous MoE GEMM
# generate_m_grouped_masked(...) - Create test tensors for M-grouped masked MoE GEMM
# generate_k_grouped_contiguous(...) - Create test tensors for K-grouped weight backward GEMM
#
# ** Technical Review **
# Test data generation framework for DeepGEMM kernel validation across architectures and data types.
#
# Architecture handling: SM90 (H100) vs SM100 (next-gen) have different capabilities:
# - SM90: FP32 scaling factors, K-major memory layouts, 1D2D kernels
# - SM100: UE8M0 packed scaling factors, supports all layouts (NT/NN/TN/TT), 1D1D/1D2D kernels
# 1D2D being deprecated for SM100, 1D1D preferred for unified codebase.
#
# Kernel types:
# - Kernel1D1D: 1D threadblock cluster, 1D warp tile (modern, supports all features)
# - Kernel1D2D: 1D threadblock cluster, 2D warp tile (legacy, SM90 optimized)
# - KernelNoSF: No scaling factors, for BF16 native GEMMs
#
# Memory layouts:
# - KMajor: Row-major for A, col-major for B (NT layout, standard)
# - MNMajor: Transposed layouts for A/B (TN/NN/TT layouts)
# SM90 supports limited MNMajor, SM100 supports all combinations.
#
# Test case enumeration philosophy:
# - enumerate_normal: Dense GEMMs covering forward (small M) and backward (large M, includes Dgrad/Wgrad)
# - enumerate_m_grouped_contiguous: MoE grouped GEMMs where tokens vary per expert (training/prefill)
# - enumerate_m_grouped_masked: MoE masked GEMMs for CUDA graph decoding (fixed max_m, variable actual_m)
# - enumerate_k_grouped_contiguous: MoE weight backward where K-axis grouped (gradient accumulation)
#
# Data generation strategy:
# - Create BF16 reference matrices, compute reference output in FP32 for accuracy
# - Quantize to FP8 E4M3 with appropriate granularity (per-token/per-channel/per-block)
# - Apply memory layout transformations (transpose for MNMajor)
# - Add accumulation (C += A @ B) vs non-accumulation (D = A @ B) tests
#
# MoE grouping simulation:
# - Actual token counts vary ±30% from expected (random.uniform(0.7, 1.3))
# - Align to GEMM block size (get_mk_alignment_for_contiguous_layout)
# - Use -1 indices to mark padding regions that should produce zero output
# - For K-grouped: each expert has different K dimension (varying hidden states)
#
# Scaling factor layouts:
# - Forward: per-token (A) and per-block (B) for accuracy
# - Backward with accumulation: per-token for both A and B (reduces memory)
# - UE8M0 packing: 4 FP32 scales → 1 torch.int with 4 UE8M0 values for SM100
#
# Output dtype handling: Support torch.float (FP32 accum) and torch.bfloat16 (BF16 output)
# Critical for mixed-precision training (FP32 accum prevents gradient underflow).
# </claudes_code_comments>

import enum
import random
import torch
from typing import Generator, List

from deep_gemm.utils import (
    align, ceil_div,
    per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout
)


class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


def get_arch_major() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major


def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()


def get_kernel_types(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return (KernelType.KernelNoSF, )

    # TODO: SM100 1D2D kernels are going to be deprecated
    # But if you want to test it, please use:
    # `(KernelType.Kernel1D2D, ) if get_arch_major() == 9 else (KernelType.Kernel1D1D, KernelType.Kernel1D2D)`
    return (KernelType.Kernel1D2D, ) if get_arch_major() == 9 else (KernelType.Kernel1D1D, )


def get_major_ab(allow_a_mn_major: bool, allow_b_mn_major: bool) -> Generator:
    for major_a in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
        for major_b in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
            if major_a.is_mn_major() and not allow_a_mn_major:
                continue
            if major_b.is_mn_major() and not allow_b_mn_major:
                continue
            yield major_a, major_b


def enumerate_normal(dtype: torch.dtype) -> Generator:
    assert dtype in (torch.float8_e4m3fn, torch.bfloat16)

    fp32_output_nk = [(256, 7168), (129280, 7168)]
    bf16_output_nk = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
    m_fwd_list, m_bwd_list = [128, 4096], [4096, ]
    nk_list = bf16_output_nk

    # Only BF16 GEMM needs FP32 outputs
    if dtype == torch.bfloat16:
        nk_list += fp32_output_nk

    for kernel_type in get_kernel_types(dtype):
        # Forward
        for m in m_fwd_list:
            for n, k in nk_list:
                out_dtype = torch.float if (n, k) in fp32_output_nk else torch.bfloat16
                yield kernel_type, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, out_dtype

        # TODO: support BF16 SM90 MN-major kernels
        if dtype == torch.bfloat16 and get_arch_major() == 9:
            continue

        # Backward
        for m in m_bwd_list:
            for n, k in nk_list:
                override_major = MajorTypeAB.MNMajor
                override_kernel_type = kernel_type
                if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
                    override_major = MajorTypeAB.KMajor
                    override_kernel_type = KernelType.Kernel1D1D
                yield kernel_type,          m, k, n, MajorTypeAB.KMajor, override_major, False, torch.bfloat16     # Dgrad
                yield override_kernel_type, n, m, k, override_major,     override_major, True,  torch.float        # Wgrad
                yield override_kernel_type, n, m, k, override_major,     override_major, False, torch.bfloat16     # Wgrad


def enumerate_m_grouped_contiguous(dtype: torch.dtype) -> Generator:
    for kernel_type in get_kernel_types(dtype):
        for num_groups, expected_m_per_group, n, k in ((4, 8192, 4096, 7168), (4, 8192, 7168, 2048), (8, 4096, 4096, 7168), (8, 4096, 7168, 2048)):
            for major_a, major_b in get_major_ab(False, get_arch_major() > 9):
                yield kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b


def enumerate_m_grouped_masked(dtype: torch.dtype) -> Generator:
    max_m = 4096
    for kernel_type in get_kernel_types(dtype):
        for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
            for n, k in ((4096, 7168), (7168, 2048), ):
                yield kernel_type, num_groups, max_m, m, n, k


def enumerate_k_grouped_contiguous():
    # Only K-major is supported for SM90
    major_a, major_b = (MajorTypeAB.KMajor, MajorTypeAB.KMajor) if get_arch_major() == 9 \
                       else (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
    # Must with FP32 accumulation and 1D1D kernels
    for num_groups, m, n, expected_k_per_group in (( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),   # EP64
                                                   ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),   # EP32
                                                   (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)):  # EP16
        ks = [align(int(expected_k_per_group * random.uniform(0.7, 1.3)), get_mk_alignment_for_contiguous_layout()) for _ in range(num_groups)]
        yield num_groups, m, n, major_a, major_b, ks, expected_k_per_group


def enumerate_sf_layout():
    for use_ue8m0 in (False, True):
        for with_transpose in (True, False):
            for mn in (4096, 4097, 8192):
                for k in (128, 7168, 7296):
                    for num_groups in (1, 2, 4):
                        yield mn, k, with_transpose, use_ue8m0, num_groups


def enumerate_k_grouped_sf_layout():
    alignment = get_mk_alignment_for_contiguous_layout()
    assert alignment % 128 == 0
    for mn in (4096, 7168):
        for num_groups, avg_k in ((16, 2048), (8, 4096), (72, 384), (128, 256)):
            ks = [align(int(random.uniform(0.7, 1.3) * avg_k), alignment) for _ in range(num_groups)]
            yield mn, ks, num_groups


def enumerate_transpose():
    for mn in (64, 4096, 16384):
        for delta in (0, 101, 202, 303):
            for k in (128, 1024, 4096, 9984, 16384):
                yield mn + delta, k


def generate_normal(m: int, n: int, k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype,
                    kernel_type: KernelType,
                    use_ue8m0: bool = False, use_bf16: bool = False):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d

    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0) if kernel_type.is_1d1d() and accumulate \
            else per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int,
                                  major_a: MajorTypeAB, major_b: MajorTypeAB,
                                  use_ue8m0: bool = False, use_bf16: bool = False):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn),
             torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d


def generate_m_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, n: int, k: int,
                              use_ue8m0: bool = False, use_bf16: bool = False):
    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (torch.empty_like(a, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, ceil_div(k, 128)), device='cuda', dtype=torch.float))
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i], use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)

    return a_fp8, b_fp8, masked_m, d, ref_d


def generate_k_grouped_contiguous(num_groups: int, m: int, n: int, major_a: MajorTypeAB, major_b: MajorTypeAB, ks: List[int], use_ue8m0: bool):
    assert get_mk_alignment_for_contiguous_layout() % 128 == 0
    k = sum(ks)

    a = torch.randn((k, m), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device='cuda', dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    start = 0
    for i, group_k in enumerate(ks):
        end = start + group_k
        ref_d[i] = c[i] + (a[start:end].T @ b[start:end])
        start = end

    a_fp8 = per_channel_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_channel_cast_to_fp8(b, use_ue8m0=use_ue8m0)

    # Transpose for K Major A/B
    if (major_a, major_b) == (MajorTypeAB.KMajor, MajorTypeAB.KMajor):
        a, sfa = a_fp8
        b, sfb = b_fp8
        new_a = torch.empty((sum(ks) * m, ), dtype=a.dtype, device=a.device)
        new_b = torch.empty((sum(ks) * n, ), dtype=b.dtype, device=b.device)
        prefix = 0
        for K in ks:
            new_a[prefix * m : (prefix + K) * m] = a[prefix : prefix + K, ].T.flatten()
            new_b[prefix * n : (prefix + K) * n] = b[prefix : prefix + K, ].T.flatten()
            prefix += K
        a_fp8, b_fp8 = (new_a, sfa.T), (new_b, sfb.T)
    else:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)

    return k, a_fp8, b_fp8, c, d, ref_d
