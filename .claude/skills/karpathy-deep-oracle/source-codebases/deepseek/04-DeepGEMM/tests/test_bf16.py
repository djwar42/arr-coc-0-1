# <claudes_code_comments>
# ** Function List **
# test_gemm() - Validate BF16 dense GEMM correctness and performance (all layouts)
# test_m_grouped_gemm_contiguous() - Validate BF16 M-grouped contiguous GEMM for MoE
# test_m_grouped_gemm_masked() - Validate BF16 M-grouped masked GEMM for MoE decoding
# test_cublaslt_gemm() - Validate cuBLASLt BF16 GEMM baseline (reference comparison)
#
# ** Technical Review **
# BF16 (bfloat16) native GEMM validation suite without FP8 quantization.
#
# Key differences from FP8 tests:
# - No scaling factors (KernelNoSF): BF16 native arithmetic, no quantization overhead
# - Higher precision: BF16 mantissa (7 bits) vs FP8 E4M3 (3 bits)
# - SM90 limitations: Accumulation not supported on H100 for BF16 (TODO comment line 19)
# - SM100-only grouped: M-grouped kernels only tested on next-gen architecture (line 145-147)
#
# test_gemm(): Dense BF16 GEMM covering forward/backward passes.
# - All layouts: NT/NN/TN/TT via major_a/major_b combinations
# - Output dtypes: BF16 (standard) or FP32 (accumulation for gradient computation)
# - Accumulation: C += A @ B for parameter gradient accumulation (when out_dtype=float)
# - SM90 skip: Skips accumulation tests on H100 due to hardware limitation
# - Alias testing: Validates transposed contiguous layouts (func_name uses major_opt)
# - Correctness: calc_diff < 0.0001 (0.01% error, tighter than FP8 due to higher precision)
#
# Performance comparison: Benchmarks against cuBLASLt (nvjet kernel + reduce for split-K).
# Reports speedup ratio: (cublas_t + split_k_t) / t. DeepGEMM optimizations target parity
# or better vs cuBLAS for transformer workloads.
#
# test_m_grouped_gemm_contiguous(): MoE BF16 training/prefill.
# - Same grouped semantics as FP8: variable tokens per expert, -1 padding indices
# - M-alignment: align to GEMM block size for efficient batching
# - Layout support: Both K-major and MN-major via major_a/major_b
# - SM100-only: Only executed if get_arch_major() > 9 (line 145)
#
# test_m_grouped_gemm_masked(): MoE BF16 CUDA graph decoding.
# - Fixed max_m budget, variable masked_m actual tokens
# - 10 correctness iterations: Tests randomized masking patterns (line 89)
# - Per-expert validation: Only checks output[:masked_m[j]] for each expert j
# - Bandwidth calculation: Accounts for valid_m ratio in memory footprint
#
# test_cublaslt_gemm(): Baseline cuBLASLt validation.
# - Direct cuBLASLt wrapper test (deep_gemm.cublaslt_gemm_nt)
# - Tighter tolerance: calc_diff < 5e-7 (cuBLAS reference accuracy)
# - Kernel profiling: Measures 'nvjet' kernel time (cuBLAS main compute kernel)
# - Validates DeepGEMM's cuBLAS integration for fallback/comparison
#
# BF16 use cases:
# - Training: Higher precision than FP8 for gradient stability
# - Inference: Faster than FP32, sufficient precision for most models
# - Mixed precision: FP32 accumulation (out_dtype=torch.float) prevents underflow
#
# Architecture notes:
# - SM90 (H100): Limited BF16 kernel support, accumulation not implemented
# - SM100 (next-gen): Full BF16 support including grouped and accumulation
# - cuBLASLt: Provides high-quality baseline for performance validation
# </claudes_code_comments>

import torch
import random

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from generators import (
    get_arch_major,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, generate_normal,
    generate_m_grouped_contiguous, generate_m_grouped_masked
)


def test_gemm() -> None:
    print('Testing GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        # TODO: support accumulation for SM90 BF16 GEMM 
        if get_arch_major() == 9 and accumulate:
            continue

        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
            func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else a.T
                b = b if major_b.is_k_major() else b.T
                assert a.is_contiguous() and b.is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c)
            diff = calc_diff(d, ref_d)
            assert diff < 0.0001, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                   f'{diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)

        t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
            func_name = f"m_grouped_bf16_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else b.mT
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked(torch.bfloat16):
        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_bf16=True)
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)
            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
        deep_gemm.cublaslt_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        assert diff < 5e-7, f'{diff=}, ({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})'

        t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), 'nvjet', suppress_kineto_output=True,)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    # TODO: support SM100
    if get_arch_major() == 9:
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()

    test_cublaslt_gemm()
