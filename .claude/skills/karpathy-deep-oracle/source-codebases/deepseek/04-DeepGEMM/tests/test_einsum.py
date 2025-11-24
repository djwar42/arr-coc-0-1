# <claudes_code_comments>
# ** Function List **
# test_bmk_bnk_mn() - Test batched GEMM with batch reduction "bmk,bnk->mn"
# test_bhr_hdr_bhd() - Test multi-head attention value projection "bhr,hdr->bhd"
# test_bhd_hdr_bhr() - Test multi-head attention score computation "bhd,hdr->bhr"
#
# ** Technical Review **
# Einsum operation tests for specialized tensor contraction patterns in attention and MoE.
#
# Einsum rationale: Some tensor operations don't fit standard GEMM APIs but are critical for
# transformers. DeepGEMM provides optimized einsum kernels for specific patterns.
#
# test_bmk_bnk_mn(): Batched GEMM with reduction over batch dimension.
# - Pattern: (s, m, k) @ (s, n, k).T → sum over s → (m, n)
# - Use case: MoE expert aggregation where s=num_experts, reducing expert outputs to single result
# - Reference: torch.bmm(a, b.mT).sum(0) computes batched matmul then sums batch dimension
# - Accumulation: Supports c parameter for D = C + einsum(...) when dtype=torch.float
# - Output dtype: FP32 (with accum) or BF16 (no accum) for numerical stability
#
# test_bhr_hdr_bhd(): Attention value projection (scores @ values).
# - Pattern: (b, h, r) @ (h, d, r).T → (b, h, d) where b=batch, h=heads, r=seq_len, d=head_dim
# - Use case: Multi-head attention value computation: attention_weights @ value_matrix
# - Memory layout: y sliced from fy[:, :, :r] tests non-contiguous input handling
# - Reference: torch.einsum('bhr,hdr->bhd', x, y) as ground truth
# - Kernel name: 'nvjet' indicates cuBLASLt backend for this pattern
#
# test_bhd_hdr_bhr(): Attention score computation (query @ key).
# - Pattern: (b, h, d) @ (h, d, r) → (b, h, r) for query-key attention scores
# - Use case: Computing attention logits before softmax in multi-head attention
# - Same memory layout handling as value projection
# - Symmetric to test_bhr_hdr_bhd with swapped query/key roles
#
# Correctness: All tests use calc_diff < 1e-5 (bmk_bnk_mn) or 1e-10 (attention) thresholds.
# Tighter tolerance for attention patterns due to BF16-only computation (no FP8 quantization).
#
# Performance metrics: TFLOPS and GB/s reported. Attention patterns are memory-bound for
# typical transformer dimensions (d=128, r=512), GEMM patterns can be compute-bound.
#
# Design note: Einsum API allows flexible tensor contraction notation while leveraging
# optimized GEMM kernels under the hood. Alternative to verbose reshape/transpose chains.
# </claudes_code_comments>

import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto,
    calc_diff, count_bytes
)


def test_bmk_bnk_mn() -> None:
    print('Testing "bmk, bnk -> mn":')
    for s in (129, 4096, 8192):
        for m, n, k in [(128, 384, 128), (256, 256, 256), (384, 128, 384)]:
            for dtype in (torch.float, torch.bfloat16):
                a = torch.randn((s, m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((s, n, k), dtype=torch.bfloat16, device='cuda')
                d = torch.randn((m, n), dtype=dtype, device='cuda')
                c = d if dtype == torch.float else None

                # Test correctness
                ref_d = (c if dtype == torch.float else 0) + torch.bmm(a.float(), b.float().mT).sum(0)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
                assert calc_diff(d, ref_d) < 1e-5

                t = bench_kineto(lambda: deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c), 'bmn_bnk_mn_gemm_impl', suppress_kineto_output=True)
                print(f' > Perf (b={s:4.0f}, {m=}, {n=}, {k=}, {"FP32" if dtype == torch.float else "BF16"}): ',
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * s * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b) + (d.numel() * 4)) / 1e9 / t:4.0f} GB/s')
    print()


def test_bhr_hdr_bhd():
    print('Testing "bhr, hdr -> bhd":')
    for b in (128, 4096, 8192):
        for h, r, d in [(128, 512, 128)]:
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:.0f} GB/s')
    print()


def test_bhd_hdr_bhr():
    print('Testing "bhd, hdr -> bhr":')
    for b in (128, 4096, 8192):
        for h, r, d in [(128, 512, 128)]:
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_bmk_bnk_mn()
    test_bhr_hdr_bhd()
    test_bhd_hdr_bhr()
