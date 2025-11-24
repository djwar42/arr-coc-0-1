# KNOWLEDGE DROP: Cloud TPU Architecture & Programming

**Created**: 2025-11-16
**Part**: PART 13 (Batch 4: TPU & Specialized Accelerators)
**File**: gcp-gpu/12-cloud-tpu-architecture-programming.md
**Lines**: 742

## What Was Created

Comprehensive guide to Google Cloud TPU architecture, programming models (JAX and PyTorch XLA), and production deployment strategies.

## Key Sections

1. **TPU Generations Comparison** (v4, v5e, v5p, v6e Trillium)
   - Detailed specifications table
   - Cost-performance trade-offs
   - Use case recommendations

2. **TPU Architecture Deep Dive**
   - TensorCore components (MXU, VPU, VMEM)
   - Memory hierarchy and bandwidth
   - Systolic array architecture (128×128, 256×256)

3. **TPU Networking & Pod Architecture**
   - Inter-Chip Interconnect (ICI) bandwidth
   - 2D vs 3D torus topologies
   - Pod sizes and slice configurations

4. **JAX Programming on TPUs**
   - Data parallelism with pmap
   - Model parallelism with pjit
   - JIT compilation best practices

5. **PyTorch XLA Programming**
   - mark_step() critical requirement
   - Multi-core training patterns
   - Common pitfalls and solutions

6. **Performance Optimization**
   - Batch size optimization (FLOPs-bound training)
   - Matrix dimension alignment (128/256 multiples)
   - Precision optimization (bf16/int8)

7. **TPU vs GPU Decision Framework**
   - Performance comparison
   - Cost analysis
   - Decision tree

8. **arr-coc-0-1 TPU Feasibility**
   - Migration assessment
   - Cost-benefit analysis
   - Recommendations

## Knowledge Integration

**Cited existing knowledge**:
- [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - 835 lines of TPU fundamentals (15+ citations)
- [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) - Vertex AI TPU patterns (8+ citations)

**Web research sources**:
- Google Cloud TPU documentation (official specs)
- JAX ML Scaling Book (TPU architecture deep dive)
- PyTorch XLA documentation (training patterns)
- Google Cloud Blog (v5p, v5e, v6e Trillium announcements)
- SemiAnalysis (TPUv5e cost analysis)
- USENIX NSDI 2024 paper (TPU v4 resiliency at scale)

**Key insights from web research**:
- **v5e cost advantage**: 60-70% lower cost per FLOP vs GPUs
- **v6e Trillium**: 256×256 systolic array (2× larger), 4.7× faster than v5e
- **ICI topology**: 2D torus (v5e/v6e) vs 3D torus (v4/v5p)
- **Batch size requirements**: Need 240+ for v5e FLOPs-bound (from HBM)
- **PyTorch XLA critical**: mark_step() is mandatory after optimizer.step()

## Technical Highlights

**TPU v5e specifications** (cost-optimized):
- 16GB HBM, 197 TF/s (bf16), 394 TF/s (int8)
- 2D torus topology (4 neighbors)
- $1.20/hour single chip, $4.80/hour for 4 chips
- 1.9× better LLM fine-tuning performance per dollar vs v4

**TPU v6e Trillium** (latest):
- 256×256 systolic array (vs 128×128 in v5e)
- 920 TF/s (bf16), 1,840 TF/s (int8)
- 4.7× performance increase over v5e

**JAX advantages**:
- Native XLA compilation (no intermediate layers)
- pmap for data parallelism (automatic gradient reduction)
- pjit for model parallelism (SPMD sharding)
- Static shapes critical for performance

**PyTorch XLA requirements**:
- `xm.mark_step()` after optimizer step (triggers XLA execution)
- `xm.optimizer_step()` for gradient synchronization
- Static shapes (avoid dynamic slicing)
- Minimize CPU transfers (.item() calls)

## arr-coc-0-1 Implications

**TPU feasibility**:
- ✅ Good fit: Transformer-based, large batches, matmul-heavy
- ⚠️ Challenges: Custom CUDA kernels, dynamic shapes, mixed CPU/GPU ops
- **Recommendation**: Don't migrate immediately, prototype on v5e-4 first

**Cost comparison**:
- Current 8×A100: $188.48 per training run
- v5e-32: $76.80 per run (60% cheaper)
- Preemptible v5e-32: $46.08 per run (76% cheaper)

**Migration effort**: 4-6 weeks (JAX port + testing + scaling)

## Quality Checklist

- ✅ Comprehensive coverage (~750 lines)
- ✅ All sources cited with links
- ✅ Web research integrated (15+ sources)
- ✅ Existing knowledge cross-referenced (2 files, 23+ citations)
- ✅ Code examples included (JAX pmap/pjit, PyTorch XLA)
- ✅ Technical depth (systolic arrays, ICI topology, arithmetic intensity)
- ✅ Practical guidance (batch size formulas, optimization patterns)
- ✅ arr-coc-0-1 specific analysis

## Unique Contributions

1. **Detailed TPU generation comparison**: v4, v5e, v5p, v6e specs with cost analysis
2. **Systolic array deep dive**: 128×128 vs 256×256 architecture
3. **ICI topology explanation**: 2D vs 3D torus, bandwidth hierarchy
4. **Batch size optimization formula**: Calculate FLOPs-bound threshold
5. **JAX vs PyTorch XLA comparison**: When to use each framework
6. **arr-coc-0-1 migration plan**: Detailed cost-benefit and effort estimate

---

**Status**: PART 13 complete ✓

**Next**: Mark checkbox in ingestion.md
