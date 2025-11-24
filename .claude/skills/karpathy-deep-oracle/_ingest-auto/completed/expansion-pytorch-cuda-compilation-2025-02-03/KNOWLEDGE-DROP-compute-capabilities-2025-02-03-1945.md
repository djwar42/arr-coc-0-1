# KNOWLEDGE DROP: CUDA Compute Capabilities & GPU Architectures

**Date**: 2025-02-03 19:45
**File Created**: `cuda/03-compute-capabilities-gpu-architectures.md`
**Lines**: ~750 lines
**Topic**: GPU architecture evolution, compute capabilities, compilation strategies

---

## What Was Created

Comprehensive guide to NVIDIA GPU compute capabilities from Turing (sm_75) through Blackwell (sm_100, sm_120), covering:

### Architecture Coverage
- **Turing (sm_75)**: T4, RTX 20 series - 2nd gen Tensor Cores, FP16/INT8
- **Ampere (sm_80, sm_86)**: A100, RTX 30 series - 3rd gen Tensor Cores, TF32, BF16, cp.async
- **Ada (sm_89)**: L4, RTX 40 series - 4th gen Tensor Cores with FP8
- **Hopper (sm_90)**: H100, H200 - Transformer Engine, TMA, thread block clusters
- **Blackwell (sm_100, sm_120)**: B100, RTX 50 series - 5th gen Tensor Cores

### Key Technical Content
1. **Compute capability fundamentals** - sm_XX notation, PTX vs cubin, forward compatibility
2. **Architecture-specific features** - Tensor Core generations, memory hierarchies, new instructions
3. **Compilation strategies** - Single-arch vs multi-arch builds, TORCH_CUDA_ARCH_LIST, CMake
4. **Performance optimization** - TF32 for 10× speedup, FP8 for 2× throughput, Tensor Core alignment
5. **ARR-COC integration** - Compiling for A100, enabling FlashAttention-2, future H100 path

---

## Critical Insights for arr-coc-0-1

### 1. Optimal Compilation Flags
```bash
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9+PTX"
```
- sm_80: A100 production (Vertex AI)
- sm_86: RTX 3090 development
- sm_89: L4 inference deployment
- +PTX: Forward compatible with H100

### 2. TF32 Automatic Acceleration (A100)
```python
torch.backends.cuda.matmul.allow_tf32 = True  # Default in PyTorch 1.7+
# Result: 156 TFLOPs vs 19.5 TFLOPs (8× faster matmuls)
```

### 3. Tensor Core Alignment
```python
# Ensure dimensions are multiples of 8 for FP16/BF16 Tensor Cores
batch_size = 16       # Multiple of 8
num_patches = 200     # K=200 (divisible by 8)
texture_channels = 16 # Pad 13 → 16 for optimal performance
```

### 4. Architecture-Specific Performance Differences
**sm_80 vs sm_86 (A100 vs RTX 3090):**
- sm_86 has 2× FP32 throughput per SM
- Both have same Tensor Core count
- Binary compiled for sm_80 runs on sm_86 but slower
- **Always recompile for sm_86 for RTX 3090**

### 5. H100 Future Path (FP8 Training)
When arr-coc migrates to H100:
```bash
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
pip install transformer-engine
# Expected: 4× faster training vs A100 TF32
```

---

## Web Research Sources

1. **NVIDIA Developer** - Official compute capability reference, H100 architecture whitepaper
2. **arnon.dk** - Comprehensive sm_XX mapping, compilation flags, PyTorch integration
3. **NVIDIA GitHub Issues** - sm_90 compatibility, driver requirements
4. **Medium/Technical Blogs** - Architecture comparisons, optimization strategies

**All sources cited with access dates (2025-02-03) and specific URLs in knowledge file.**

---

## Integration Points

### Connects to Existing Knowledge
- `vertex-ai-production/01-gpu-optimization-deep.md` - GPU specs, TF32 details
- `karpathy/practical-implementation/32-vertex-ai-gpu-tpu.md` - Vertex AI pricing
- `cuda/00-streams-concurrency-async.md` - CUDA programming model
- `cuda/01-memory-management-unified.md` - Memory hierarchy

### Enables Future Topics
- PART 3: PyTorch custom CUDA extensions (needs architecture targeting)
- PART 4: Tensor Core programming (builds on generation differences)
- PART 5: torch.compile optimization (architecture-aware code generation)
- PART 6: Mixed precision training (TF32, BF16, FP8 hardware support)

---

## Quick Reference Table

| Architecture | sm_XX | Tensor Cores | Key Feature | arr-coc Use Case |
|--------------|-------|--------------|-------------|------------------|
| Turing | 75 | Gen 2 (FP16) | Cost inference | Legacy support |
| Ampere (A100) | 80 | Gen 3 (TF32) | Training flagship | **Primary target** |
| Ampere (RTX) | 86 | Gen 3 (2× FP32) | Dev workstation | Local development |
| Ada | 89 | Gen 4 (FP8) | Efficient inference | Future inference |
| Hopper | 90 | Gen 4 (TE) | LLM training | Future training |
| Blackwell | 100/120 | Gen 5 | Next-gen | 2025+ path |

---

## Status

✅ **PART 2 COMPLETE**
- File: `cuda/03-compute-capabilities-gpu-architectures.md` (750 lines)
- Citations: 12 web sources with URLs and access dates
- ARR-COC integration: Section 8 with practical examples
- Next: PART 3 (Custom CUDA extensions)
