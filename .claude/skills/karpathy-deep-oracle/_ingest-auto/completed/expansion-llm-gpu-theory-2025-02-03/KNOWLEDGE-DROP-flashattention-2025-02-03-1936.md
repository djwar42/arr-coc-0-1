# KNOWLEDGE DROP: FlashAttention & Attention Kernel Optimization

**Date**: 2025-02-03
**Time**: 19:36
**Runner**: PART 1 of LLM + GPU Integration Expansion
**File Created**: `karpathy/llm-gpu-integration/00-flashattention-internals.md`
**Lines**: 1309

---

## Summary

Created comprehensive documentation on FlashAttention algorithm and its evolution from version 1 through 3, covering the fundamental memory bottleneck in standard attention, block-wise tiling solutions, and hardware-specific optimizations for Hopper GPUs.

## Key Topics Covered

### 1. Standard Attention Memory Bottleneck (~120 lines)
- GPU memory hierarchy (HBM vs SRAM bandwidth: 1.6 TB/s vs 19.5 TB/s on A100)
- O(N²) intermediate matrices (S, P) requiring quadratic HBM accesses
- Why attention is memory-bound, not compute-bound
- Bandwidth-limited performance (~25% GPU utilization on standard implementations)

### 2. FlashAttention Algorithm (~200 lines)
- Core innovation: Block-wise computation keeping data in SRAM
- Online softmax algorithm for numerically stable incremental computation
- Forward pass with tiling (Br × d Q blocks, Bc × d K/V blocks)
- Backward pass with recomputation trade-off (memory vs compute)
- IO complexity: O(N²/M) vs O(N²) for standard attention
- Block size selection for SRAM constraints (64-128 rows typical)
- Performance: 2-4× speedup, enables 32K+ contexts

### 3. FlashAttention-2 Improvements (~150 lines)
- Reduced non-matmul FLOPs (deferred softmax rescaling)
- Sequence parallelism (split across thread blocks, not just heads)
- Work partitioning within thread blocks (producer-consumer warps)
- Performance: 2× faster than FA-1, reaching 225 TFLOPs (72% utilization on A100)

### 4. FlashAttention-3 Hopper Optimizations (~180 lines)
- WGMMA (Warpgroup Matrix Multiply-Accumulate) - 1.5-2× faster Tensor Cores
- TMA (Tensor Memory Accelerator) - asynchronous memory transfers freeing registers
- Inter-warpgroup overlapping (pingpong scheduling)
- Intra-warpgroup overlapping (2-stage pipeline)
- Warp specialization with double buffering
- FP8 low-precision with incoherent processing (Hadamard transform reduces quantization error 2.6×)
- Performance: 740 TFLOPs FP16 (75% H100 utilization), 1.2 PFLOPs FP8

### 5. PyTorch Integration (~120 lines)
- torch.nn.functional.scaled_dot_product_attention (SDPA) API
- Automatic backend selection (FlashAttention, memory-efficient, math fallback)
- Full compatibility with torch.compile for graph fusion
- Custom attention masks (causal, boolean, additive bias, sliding window)
- MultiheadAttention module integration
- PyTorch 2.2+ automatic FlashAttention-2 usage

### 6. ARR-COC Connection (~100 lines)
- Query-aware relevance scoring using cross-attention
- Variable LOD token allocation (64-400 tokens per patch)
- Multi-query attention for efficient KV cache (32× smaller)
- Texture array processing (13 channels)
- Token budget GPU alignment (multiples of 16/64 for Tensor Cores)

## Research Sources

**Papers**:
- FlashAttention (Dao et al., NeurIPS 2022) - arXiv:2205.14135
- FlashAttention-2 (Dao, 2023) - arXiv:2307.08691
- FlashAttention-3 (Shah et al., 2024) - arXiv:2407.08608

**Web Resources**:
- Tri Dao's FlashAttention-3 blog post (comprehensive Hopper optimization explanation)
- PyTorch documentation and tutorials for SDPA
- PyTorch 2.2 release notes (FA-2 integration)
- NVIDIA Hopper whitepaper (WGMMA, TMA details)
- NVIDIA CUTLASS documentation (warp specialization patterns)

**Existing Knowledge Referenced**:
- cuda/01-memory-management-unified.md (memory hierarchy, HBM vs SRAM)
- cuda/05-tensor-core-programming-wmma-mma.md (Tensor Core tile sizes)
- cuda/06-pytorch-jit-torch-compile.md (kernel fusion concepts)
- cuda/07-mixed-precision-training-internals.md (FP8/FP16 formats)
- vllm-knowledge/00-vllm-architecture-pagedattention.md (PagedAttention builds on FA)

## Performance Numbers Documented

**Evolution of FlashAttention Performance**:
- Standard PyTorch: ~80 TFLOPs (25% A100 utilization)
- FlashAttention-1: ~125 TFLOPs (40% utilization) - **2-3× speedup**
- FlashAttention-2: ~225 TFLOPs (72% utilization) - **2× speedup over FA-1**
- FlashAttention-3 FP16: ~740 TFLOPs (75% H100 utilization) - **2× speedup over FA-2**
- FlashAttention-3 FP8: ~1200 TFLOPs (61% H100 utilization) - **1.6× speedup over FP16**

**Context Length Scaling**:
- Before FlashAttention: 2-4K tokens (GPT-3, OPT)
- With FA-1/2: 32-128K tokens (GPT-4, Claude)
- With FA-3: 256K+ tokens FP16, 512K+ tokens FP8

## Implementation Details

**Block-wise Tiling**:
- Typical block sizes: Br=Bc=128 for d=64, Br=Bc=64 for d=128
- SRAM usage: ~100KB per thread block on A100/H100
- Trade-off: Larger blocks → fewer recomputations, smaller blocks → more parallelism

**Online Softmax Algorithm**:
- Maintains running (max, sum) statistics across blocks
- Numerically stable with rescaling
- Enables single-pass computation without materializing full attention matrix

**Hopper-Specific Optimizations**:
- Pingpong scheduling: +50 TFLOPs (8% improvement)
- Intra-warpgroup pipelining: +20-40 TFLOPs (3-6% improvement)
- FP8 with Hadamard: 2.6× lower quantization error vs naive FP8

## Next Steps (Remaining PARTs)

- PART 2: Architecture & GPU constraints (hidden dims, heads, vocab sizes)
- PART 3: Training dynamics (gradient checkpointing, pipeline parallelism, ZeRO)
- PART 4: Inference optimization (KV cache management, continuous batching, PagedAttention deep dive)

---

**Status**: ✓ PART 1 Complete
**Quality**: Comprehensive coverage with citations, performance numbers, code examples, and ARR-COC integration
