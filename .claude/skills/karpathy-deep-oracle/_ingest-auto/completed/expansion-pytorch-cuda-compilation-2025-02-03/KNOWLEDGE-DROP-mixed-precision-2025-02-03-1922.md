# KNOWLEDGE DROP: Mixed Precision Training Internals

**Date**: 2025-02-03 19:22
**Runner**: PART 6
**File Created**: `cuda/07-mixed-precision-training-internals.md`
**Lines**: 1,419 lines
**Status**: ✓ Complete

---

## What Was Created

Comprehensive deep dive into PyTorch mixed precision training internals, covering:

1. **torch.cuda.amp.autocast** - Automatic precision casting mechanism
2. **GradScaler algorithm** - Loss scaling and gradient overflow detection
3. **Precision formats** - FP32, FP16, BF16, TF32, FP8 comparison
4. **FP8 training** - Transformer Engine, delayed scaling, MXFP8
5. **Gradient stability** - Debugging underflow/overflow, NaN detection
6. **ARR-COC optimization** - Mixed precision strategies for relevance scorers

---

## Key Technical Insights

### 1. Autocast Context Manager (200 lines)

**Core mechanism:**
- Thread-local state tracking for autocast mode
- Operation-specific whitelist/blacklist (matmul→FP16, softmax→FP32)
- Type promotion for mixed inputs
- Nested context support with enable/disable

**Critical rule:**
```python
# CORRECT: Backward outside autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()  # Outside autocast!
```

### 2. GradScaler Algorithm (200 lines)

**Five methods deep dive:**
1. `scale(loss)` - Multiply loss by scale factor
2. `unscale_(optimizer)` - Divide gradients, check inf/nan
3. `step(optimizer)` - Conditionally update parameters
4. `update()` - Adjust scale based on overflow detection
5. Internal state tracking per optimizer

**Dynamic scaling:**
- Init scale: 65536 (2^16)
- Growth: 2× every 2000 successful iterations
- Backoff: 0.5× on overflow detection
- Prevents gradient underflow in FP16

### 3. Precision Format Comparison (150 lines)

| Format | Bits | Exponent | Mantissa | Range | Hardware |
|--------|------|----------|----------|-------|----------|
| **FP32** | 32 | 8 | 23 | ±3.4e38 | All |
| **TF32** | 32* | 8 | 10 | ±3.4e38 | Ampere+ |
| **BF16** | 16 | 8 | 7 | ±3.4e38 | Ampere+, TPU |
| **FP16** | 16 | 5 | 10 | ±65,504 | Volta+ |
| **FP8 E4M3** | 8 | 4 | 3 | ±448 | Hopper+, Ada+ |
| **FP8 E5M2** | 8 | 5 | 2 | ±57,344 | Hopper+, Ada+ |

**Key finding:** BF16 doesn't need GradScaler (same dynamic range as FP32)

### 4. FP8 Training Details (200 lines)

**Two scaling strategies:**
- **Tensor-wise:** Single FP32 scale per tensor (standard FP8)
- **Block-wise (MXFP8):** E8M0 scale per 32 elements (Blackwell)

**Delayed vs Current Scaling:**
- Delayed: Uses amax history (stable, recommended for LLMs)
- Current: Uses present statistics (reactive, better convergence)

**Transformer Engine integration:**
```python
import transformer_engine.pytorch as te
# Replace nn.Linear with te.Linear
# FP8 autocast handles E4M3 fwd + E5M2 bwd automatically
```

**H100 performance:**
- 3,958 TFLOPs FP8 (vs 1,979 TFLOPs FP16)
- 4× training speedup potential
- 75% memory reduction

### 5. ARR-COC Optimization (150 lines)

**Recommended strategy: BF16 training**

**Why BF16 for ARR-COC:**
- Opponent processing needs stable gradients
- Tension balancing: `compress_score - particularize_score`
- No gradient scaling complexity
- A100 TF32 automatic: 156 TFLOPs (10× vs FP32)

**Three approaches documented:**
1. BF16 training (simplest, stable)
2. TF32 automatic (zero code changes)
3. FP16 custom autocast (maximum speed, complex)

---

## Sources Cited

**PyTorch Official:**
- PyTorch AMP Documentation (https://docs.pytorch.org/docs/stable/amp.html)
- PyTorch AMP Examples (https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- PyTorch AMP Recipe (https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

**NVIDIA Resources:**
- Floating-Point 8 Blog (https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- TransformerEngine GitHub (https://github.com/NVIDIA/TransformerEngine)
- Transformer Engine FP8 Primer (https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

**Web Research:**
- Stack Overflow (GradScaler usage, mixed precision training)
- PyTorch Forums (FP8 support, H100 discussions)
- Reddit r/MachineLearning (BF16 vs FP16 convergence)

**Source Documents:**
- cuda/01-memory-management-unified.md (AMP mentions)
- vertex-ai-production/01-gpu-optimization-deep.md (GPU specs, mixed precision overview)

---

## Connections to Existing Knowledge

**Builds on:**
- `cuda/01-memory-management-unified.md` - Memory bandwidth implications of FP16/FP8
- `vertex-ai-production/01-gpu-optimization-deep.md` - Tensor Core specifications, GPU comparisons

**Complements:**
- PyTorch build system (PART 1) - Compiling with CUDA_ARCH_LIST for Tensor Cores
- Compute capabilities (PART 2) - Understanding sm_80 FP16/TF32, sm_90 FP8 support
- Tensor Core programming (PART 4) - Low-level WMMA with FP16/BF16 formats

**Enables:**
- Custom CUDA extensions (PART 3) - Writing FP16/FP8 kernels
- torch.compile (PART 5) - Compiling mixed precision models
- ARR-COC training optimization - Production deployment strategies

---

## Validation

**File structure:**
- ✓ 7 major sections (Overview, Autocast, GradScaler, Formats, FP8, Stability, ARR-COC)
- ✓ 1,419 lines total (exceeds 800 line target)
- ✓ Code examples with full training loops
- ✓ Comprehensive citations with URLs and access dates
- ✓ ARR-COC specific optimization strategies

**Knowledge completeness:**
- ✓ Autocast internals (thread-local state, operation policies)
- ✓ GradScaler algorithm (all 5 methods, dynamic scaling)
- ✓ All precision formats (FP32/TF32/BF16/FP16/FP8 E4M3/E5M2)
- ✓ FP8 training (Transformer Engine, delayed scaling, MXFP8)
- ✓ Practical debugging (NaN detection, scale monitoring)
- ✓ ARR-COC integration (3 approaches with benchmarks)

**Cross-references:**
- ✓ Links to existing cuda/ files
- ✓ Links to vertex-ai-production/ files
- ✓ References to GPU architectures (A100, H100)
- ✓ Connection to ARR-COC training pipeline

---

## Impact on Oracle Knowledge Base

**Before PART 6:**
- General mixed precision overview in vertex-ai-production
- Brief AMP mention in cuda/01-memory-management
- No GradScaler internals
- No FP8 training details

**After PART 6:**
- Complete autocast mechanism documentation
- Full GradScaler algorithm with all methods
- Precision format comparison table
- FP8 training strategies (delayed vs current scaling)
- MXFP8 block-wise scaling (Blackwell)
- ARR-COC specific optimization guide

**New capabilities unlocked:**
1. Debug mixed precision training failures
2. Choose optimal precision format for use case
3. Implement FP8 training on H100
4. Optimize ARR-COC relevance scorer training
5. Understand gradient scaling dynamics
6. Profile Tensor Core utilization

---

## Next Steps for Oracle

**Immediate:**
- ✓ Checkbox marked in ingestion.md
- ✓ KNOWLEDGE DROP created
- ⏳ Wait for other PARTs to complete

**When all PARTs complete:**
1. Update INDEX.md (add cuda/07-mixed-precision-training-internals.md)
2. Update SKILL.md (increment file count, enhance description)
3. Move folder to _ingest-auto/completed/
4. Git commit with summary

**Future expansions:**
- FP4 training (Blackwell native support)
- NVFP4 microscaling formats
- INT8 quantization comparison
- Mixed precision inference optimization
- Custom precision training loops

---

**PART 6 Status: ✓ COMPLETE**

Knowledge file created: `cuda/07-mixed-precision-training-internals.md` (1,419 lines)
All requirements met. Ready for oracle finalization.
