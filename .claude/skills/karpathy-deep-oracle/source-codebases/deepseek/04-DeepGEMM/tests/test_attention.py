# <claudes_code_comments>
# ** Function List **
# apply_skip_head_mid(d, head_splits) - Apply head dimension masking (left+mid+right split)
# test_gemm_skip_head_mid() - Validate GEMM with skip-head-mid attention optimization
# kv_cache_cast_to_fp8(x) - Quantize KV cache to FP8 with packed scaling factors
# generate_cp_test_data(seq_len, seq_len_kv) - Generate context parallelism test indices
# ref_fp8_mqa_logits(...) - Reference implementation for MQA logits computation
# test_mqa_logits() - Validate FP8 Multi-Query Attention logits with context parallelism
# ref_fp8_paged_mqa_logits(...) - Reference implementation for paged MQA logits
# test_paged_mqa_logits() - Validate FP8 paged attention for vLLM-style KV caching
#
# ** Technical Review **
# Specialized attention kernel tests for DeepSeek v3.2 indexer and efficient serving.
#
# test_gemm_skip_head_mid(): DeepSeek v3.2 MLA (Multi-Latent Attention) optimization.
# - Head splits: (left=128, mid=64, right=128) dimensions for Q projection
# - Use case: MLA compresses KV to latent space, reconstructs Q heads in 3 segments
# - Middle segment (mid=64) is computed separately, skipped in main GEMM to save FLOPS
# - Implementation: GEMM computes left+right (256 dims), then insert zero-padded mid (64 dims)
# - apply_skip_head_mid(): Splits heads, inserts zeros for mid, concatenates back
# - Speedup: Reduces GEMM N from 320 to 256, ~20% fewer FLOPS for Q projection
#
# test_mqa_logits(): Multi-Query Attention with context parallelism (CP) for long sequences.
# - MQA: Multiple Q heads (64) share single K/V head (memory efficient for long context)
# - Pattern: Q[seq_len, num_heads, head_dim] @ K[seq_len_kv, head_dim].T → [num_heads, seq_len, seq_len_kv]
# - Weights: Per-query-head scaling (DeepSeek indexer uses learned weights for head fusion)
# - Context parallelism: Splits KV across devices, each rank computes subset of attention
# - Indexing: cu_seqlen_ks (start) and cu_seqlen_ke (end) define KV range per query token
# - generate_cp_test_data(): Creates interleaved CP indices (cp_id=cp_size//3) for testing
# - Masking: Outputs -inf for tokens outside [ks, ke) range (excluded from softmax)
# - Cost: Tracks actual FLOPS (non-masked elements) for accurate TFLOPS calculation
# - clean_logits kernel: Efficiently zeros out-of-range elements (measured separately)
#
# FP8 quantization for MQA:
# - Q: Direct cast to FP8 E4M3 (assumes pre-scaled)
# - KV: per_custom_dims_cast_to_fp8 with dim=(0,) creates per-token scaling factors
# - Scaling factor shape: [seq_len_kv] for memory efficiency vs accuracy trade-off
#
# test_paged_mqa_logits(): vLLM-style paged attention for variable-length batching.
# - Use case: Batch decode with different context lengths per request (context_lens)
# - Paging: KV cache split into blocks (blocksize=64), non-contiguous memory
# - block_tables: [batch_size, max_blocks] maps logical blocks to physical cache indices
# - next_n: Number of new tokens per request (typically 1 for decoding, >1 for prefill)
# - max_model_len: Maximum supported sequence length (111K for long context models)
#
# Paged KV cache format (kv_cache_cast_to_fp8):
# - Input: [num_blocks, block_size, num_heads=1, head_dim] BF16 cache
# - Quantize: Per-token (block_size) scaling, SF shape [num_blocks, block_size]
# - Layout: Interleave FP8 data + FP32 scaling factors: [block_size*(head_dim+4)] bytes
# - Packing: First block_size*head_dim bytes = FP8 values, last block_size*4 bytes = SF
# - Rationale: Co-locate data + SF for efficient cache-line access during decode
#
# Schedule metadata (get_paged_mqa_logits_metadata):
# - Pre-computes block scheduling for SM utilization
# - Inputs: context_lens, blocksize, num_sms (streaming multiprocessors)
# - Optimizes load balancing across variable-length requests
#
# Reference implementations:
# - ref_fp8_mqa_logits(): torch.einsum('mhd,nd->hmn') + masking + weighted sum
# - ref_fp8_paged_mqa_logits(): Block-by-block iteration with paged lookup
# - Both compute in FP32 for numerical accuracy baseline
#
# Correctness validation:
# - Mask alignment: -inf positions must match exactly (torch.equal on neginf_mask)
# - Numerical: calc_diff < 1e-3 on non-masked elements (0.1% error tolerance)
# - Large sequences: Only validate seq_len_kv < 32K (reference too slow otherwise)
#
# Performance metrics:
# - TFLOPS: Based on actual cost (non-masked elements), not theoretical peak
# - clean_logits: Separate timing for zeroing masked elements (memset-like operation)
# - Bandwidth: Accounts for Q, K, V, weights, indices, and output logits
#
# Context parallelism (CP) patterns:
# - disable_cp=True: Standard causal mask (each token attends to [0, seq_len_kv-seq_len+i])
# - disable_cp=False: Interleaved CP ranks (simulates distributed attention sharding)
# - CP efficiency: Measures overhead of non-contiguous KV access patterns
# </claudes_code_comments>

import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8

from generators import get_arch_major, generate_normal, get_ue8m0_usage, get_kernel_types, MajorTypeAB


def apply_skip_head_mid(d: torch.Tensor, head_splits: Tuple[int, int, int]):
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    # Split and insert padding tensor
    d = d.view(m, num_heads, -1)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]

    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


def test_gemm_skip_head_mid() -> None:
    print('Testing GEMM skip head mid:')
    head_splits = (128, 64, 128)

    major_a, major_b = MajorTypeAB.KMajor,  MajorTypeAB.KMajor
    out_dtype, accumulate = torch.bfloat16, False

    for kernel_type in get_kernel_types(dtype=torch.float8_e4m3fn):
        for m in (128, 4096):
            for n, k in [(32768, 512), (8192, 512)]:
                kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
                use_ue8m0 = get_ue8m0_usage(kernel_type)
                disable_ue8m0_cast = not use_ue8m0

                a, b, _, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
                d = apply_skip_head_mid(d, head_splits)
                ref_d = apply_skip_head_mid(ref_d, head_splits)

                deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}'

                t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast),
                                'fp8_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): '
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[ :, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int,  device='cuda')
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, cost_only: bool = False):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end   = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
    return logits, cost


def test_mqa_logits():
    print('Testing FP8 MQA Logits:')
    num_heads, head_dim = 64, 128
    for seq_len in (2048, 4096):
        for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
            for disable_cp in (False, True):
                q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
                kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
                weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
                    ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
                else:
                    ks, ke = generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0, ), False)
                logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

                do_check = (seq_len_kv < 32768)
                if do_check:
                    ref_logits, ref_cost = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

                    ref_neginf_mask = (ref_logits == float('-inf'))
                    neginf_mask = (logits == float('-inf'))
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                    logits = logits.masked_fill(neginf_mask, 0)
                    diff = calc_diff(logits, ref_logits)
                    assert diff < 1e-3, f"{diff=}"
                else:
                    ref_cost = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke, cost_only=True)

                tflops = 2 * ref_cost * num_heads * head_dim / 1e12
                t, clean_t = bench_kineto(lambda: deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke),
                                          ('fp8_mqa_logits', 'clean_logits'))
                clean_bytes = (seq_len * seq_len_kv - ref_cost) * 4 + count_bytes(ks, ke)
                print(f' > S={seq_len:4}, SKV={seq_len_kv:6}, H={num_heads:3}, D={head_dim:3}, CP={0 if disable_cp else 1}: '
                      f'{tflops / t:4.0f} TFLOPS, {t * 1e6:4.0f} us, '
                      f'{(count_bytes(q_fp8, kv_fp8, weights, ks, ke) + ref_cost * 4) / t / 1e9:4.0f} GB/s | '
                      f'clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s')
    print()


def ref_fp8_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                             weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                             max_model_len: int):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device='cuda')
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range(ceil_div(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device='cuda')
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype), float('-inf'))
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
    return logits


def test_paged_mqa_logits():
    print('Testing FP8 Paged MQA Logits:')
    max_model_len = 111 * 1000
    for batch_size, next_n in [(64, 1), (64, 2), (128, 1)]:
        for heads, index_dim in [(64, 128)]:
            for avg_kv in (8192, 32768):
                num_blocks, blocksize = max_model_len * 3, 64

                q = torch.randn((batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.bfloat16)
                kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.bfloat16)
                weights = torch.randn((batch_size * next_n, heads), device='cuda', dtype=torch.float32)

                context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size, )).cuda().to(torch.int32)
                max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = context_lens[i].item()
                    for j in range(ceil_div(ctx_len, blocksize)):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(context_lens, blocksize, deep_gemm.get_num_sms())
                logits = deep_gemm.fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=True)

                ref_logits = ref_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len)
                positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device='cuda') // next_n
                next_n_offset = torch.arange(batch_size * next_n, device='cuda') % next_n
                ref_neginf_mask = ~(positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1))

                neginf_mask = (logits == float('-inf'))
                assert torch.equal(neginf_mask, ref_neginf_mask)

                logits = logits.masked_fill(neginf_mask, 0)
                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"

                sum_lens = sum(context_lens.to(torch.int64))
                tflops = 2 * sum_lens * next_n * heads * index_dim / 1e12
                input_bytes = count_bytes(q_fp8, weights, context_lens) + sum_lens * (index_dim + 4) + (sum_lens / blocksize) * 4
                output_bytes = sum_lens * next_n * 4
                t, clean_t = bench_kineto(lambda: deep_gemm.fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=True),
                                          ('fp8_paged_mqa_logits', 'clean_logits'))
                clean_bytes = (batch_size * next_n * max_model_len - neginf_mask.sum().item()) * 4 + count_bytes(context_lens)
                print(f' > BSZ={batch_size:3}, NextN={next_n:1}, H={heads:2}, D={index_dim:2}, L={avg_kv:6}: '
                      f'{tflops / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, '
                      f'{(input_bytes + output_bytes) / t / 1e9:4.0f} GB/s | '
                      f'clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    test_gemm_skip_head_mid()

    test_mqa_logits()
    test_paged_mqa_logits()
