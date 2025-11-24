# KNOWLEDGE DROP: Tensor Core Programming (WMMA & MMA)

**Date:** 2025-02-03
**Time:** 19:22
**Part:** 4 of 6
**File:** `cuda/05-tensor-core-programming-wmma-mma.md`
**Lines:** 1029

---

## Summary

Created comprehensive guide on programming NVIDIA Tensor Cores using WMMA API and PTX mma.sync instructions. Covers four generations of Tensor Cores (Volta through Hopper), multiple precision formats (FP8/FP16/BF16/TF32/FP64), and practical optimization strategies for ARR-COC relevance scorers.

---

## Key Knowledge Acquired

### 1. Tensor Core Fundamentals

**What They Are:**
- Specialized hardware for matrix multiply-accumulate (D = A×B + C)
- Warp-level (32 threads) or warpgroup-level (128 threads) operations
- 10-20× speedup over standard CUDA cores for matrix operations

**Generational Evolution:**
- Gen 1 (Volta, sm_70): FP16 only, 16×16×16 tiles
- Gen 2 (Turing, sm_75): Added INT8/INT4, T4 at 65 TFLOPs FP16
- Gen 3 (Ampere, sm_80): TF32/BF16/FP64, A100 at 312 TFLOPs FP16
- Gen 4 (Hopper, sm_90): FP8 formats, H100 at 2000 TFLOPs FP8

### 2. WMMA API Programming

**Core Operations:**
```cpp
// Fragment declaration
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Load, compute, store
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

**Key Characteristics:**
- High-level warp-wide interface
- Portable across architectures (sm_70+)
- Automatic fragment layout handling
- Mixed precision support (FP16 input, FP32 accumulate)

### 3. PTX mma.sync Instructions

**Low-Level Control:**
```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16  d, a, b, c;
```

**Fragment Layout Complexity:**
- Each thread owns non-contiguous elements
- Layout computed via groupID and threadID formulas
- LDMATRIX instruction for SMEM → register loading
- Required for maximum performance tuning

### 4. Precision Formats

**FP8 (Hopper Only):**
- E4M3: 4 exp, 3 mantissa (more precision, less range)
- E5M2: 5 exp, 2 mantissa (more range, less precision)
- 2× throughput vs FP16, half memory footprint
- Requires Transformer Engine for scaling

**TF32 (Ampere+):**
- 8 exp, 10 mantissa (FP32 range, reduced mantissa)
- Automatic in matmul operations
- ~10× speedup over FP32, minimal accuracy loss
- Drop-in replacement for FP32 code

**BF16 vs FP16:**
- BF16: Same range as FP32, better for training
- FP16: More precision, needs gradient scaling
- Both supported on Ampere+ (Gen 3+)

### 5. Hopper WGMMA Features

**Warpgroup Operations (128 threads):**
- Larger tiles: 64×64×16 and beyond
- Asynchronous execution model
- SMEM-only for operand B
- Matrix descriptors replace direct pointers

**Synchronization Primitives:**
```cpp
warpgroup_arrive();           // wgmma.fence.sync.aligned
gemm(...);                    // wgmma.mma_async...
warpgroup_commit_batch();     // wgmma.commit_group
warpgroup_wait<0>();          // wgmma.wait_group 0
```

**Core Matrices:**
- Building blocks: 8 strided × 16 bytes contiguous
- Swizzle modes: None, 32B, 64B, 128B
- Descriptors encode: address, LBO, SBO, swizzle, offset

### 6. Verification and Profiling

**Nsight Compute Metrics:**
```bash
ncu --metrics \
  smsp__sass_thread_inst_executed_op_hmma_,\
  smsp__inst_executed_pipe_tensor \
  ./kernel
```

**Performance Targets:**
- A100 FP16: 312 TFLOPs peak
- H100 FP8: 2000 TFLOPs peak
- Target utilization: >80%
- Occupancy: 50-75% (balance with registers)

### 7. ARR-COC Optimization Strategy

**Current Architecture:**
- Participatory scorer: query @ patches.T
- Shape: [B, D] @ [B, D, N] = [B, N]
- Perfect Tensor Core target (matrix multiply)

**Optimization Phases:**
1. **Baseline WMMA:** 10-20× speedup on A100 (FP16)
2. **Fused Operations:** Combine all three scorers, 2-3× additional
3. **Hopper WGMMA:** Port to H100, 3-5× over A100 WMMA
4. **FP8 Exploration:** 2× over FP16 WGMMA if precision sufficient

**Expected Total Speedup:**
- A100 standard → A100 WMMA: 10-30×
- A100 standard → H100 WGMMA FP16: 30-150×
- A100 standard → H100 WGMMA FP8: 50-200×

---

## Web Research Sources

**Primary:**
1. [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
   - Fourth-gen Tensor Core specs
   - FP8 formats and Transformer Engine
   - A100 vs H100 performance comparisons

2. [Nvidia Tensor Core MMA PTX Programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d)
   - PTX mma.sync syntax
   - Fragment layouts and thread ownership
   - LDMATRIX instruction
   - Complete HGEMM example

3. [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
   - Warpgroup matrix operations
   - Matrix descriptors and core matrices
   - Asynchronous execution model
   - SMEM layout constraints

**Supporting:**
- PTX ISA documentation (referenced, too large to scrape)
- CUTLASS library source code
- NVIDIA architecture whitepapers

---

## Source Documents Referenced

- `vertex-ai-production/01-gpu-optimization-deep.md` (lines 36-99)
  - Tensor Core specifications
  - A100/H100 performance numbers
  - Memory hierarchy and bandwidth

---

## File Statistics

**Structure:**
- 7 main sections
- 1029 total lines
- ~700 lines content (329 lines sources/metadata)

**Coverage:**
1. Tensor Core Architecture: ~100 lines
2. WMMA API Programming: ~200 lines
3. PTX mma.sync: ~150 lines
4. Precision Formats: ~150 lines
5. Verification/Profiling: ~100 lines
6. Hopper WGMMA: ~150 lines
7. ARR-COC Optimization: ~100 lines

---

## Integration Points

**Connects to Existing Knowledge:**
- Links vertex-ai-production GPU optimization
- References CUDA memory management
- Builds on CUDA streams/concurrency
- Relates to PyTorch custom extensions (future)

**Enables Future Work:**
- PyTorch JIT compilation (PART 5)
- Mixed precision training (PART 6)
- ARR-COC relevance scorer implementation
- FlashAttention-style optimizations

---

## Next Steps (for oracle, not runner)

**PART 5:** PyTorch JIT & torch.compile
- TorchScript vs torch.compile
- TorchDynamo, TorchInductor
- CUDA Graph integration
- Compilation modes

**PART 6:** Mixed Precision Training Internals
- torch.cuda.amp deep dive
- GradScaler algorithm
- FP8 training with Transformer Engine
- ARR-COC mixed precision strategy

---

## Validation Checklist

- [✓] File created: `cuda/05-tensor-core-programming-wmma-mma.md`
- [✓] Contains 7 sections as specified
- [✓] ~700+ lines content (1029 total)
- [✓] All citations include sources and links
- [✓] Web research URLs preserved
- [✓] ARR-COC optimization strategies included
- [✓] Practical recommendations provided
- [✓] Sources section complete
- [✓] Checkbox marked in ingestion.md

---

**Status:** ✓ PART 4 COMPLETE

**File:** `/Users/alfrednorth/Desktop/Code/arr-coc-ovis/.claude/skills/karpathy-deep-oracle/cuda/05-tensor-core-programming-wmma-mma.md`

**Ready for:** Oracle review and INDEX.md update
