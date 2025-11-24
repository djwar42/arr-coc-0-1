# KNOWLEDGE DROP: CUDA Performance Debugging & Profiling

**Runner**: PART 3
**Date**: 2025-11-13
**Status**: ✓ Complete

---

## Knowledge File Created

**Location**: `cuda/10-performance-debugging-profiling-expert.md`
**Size**: ~400 lines
**Focus**: Expert-level performance profiling, bottleneck diagnosis, optimization strategies

---

## Content Overview

### Section 1: Profiling Tools & Workflow (~100 lines)

**Tools Covered:**
- NSight Systems (system-level profiling)
- NSight Compute (kernel-level profiling)
- torch.profiler (PyTorch-specific)

**Key Content:**
- Non-intrusive profiling commands
- NVTX annotation examples
- Timeline view interpretation
- CPU + GPU correlation

**Cited Sources:**
- Meta's AI Performance Profiling (MAIProf)
- Henry Ko's NSight Systems tutorial
- PyTorch profiler documentation

### Section 2: GPU Utilization Debugging (~100 lines)

**Bottlenecks Covered:**
- Low SM (Streaming Multiprocessor) utilization
- Zero Tensor Core utilization
- Underutilized GPU memory

**Key Content:**
- Meta case study: 9.1% → 74% SM utilization
- Occupancy calculation formulas
- Mixed precision diagnostics
- Tensor dimension alignment (multiples of 8/16)

**Real Production Example:**
```
Before optimization:
- SM utilization: 9.1%
- Tensor Core: 0%
- Memory: 47%

After optimization:
- SM utilization: 74%
- Tensor Core: 62%
- Memory: 85%
- Speedup: 4-5× end-to-end
```

### Section 3: Memory Bandwidth Analysis (~100 lines)

**Topics:**
- HBM bandwidth saturation diagnosis
- Memory coalescing (26× speedup example)
- L1/L2 cache optimization
- Shared memory usage patterns

**Key Metrics:**
- A100: 2.0 TB/s peak, 1.4-1.6 TB/s typical (70-80%)
- H100: 3.35 TB/s peak, 2.4-2.8 TB/s typical (70-85%)
- Coalesced vs uncoalesced: 26× performance difference

**Cited Sources:**
- NVIDIA CUDA C++ Best Practices Guide
- Memory bandwidth measurements
- Cache hit rate targets

### Section 4: Expert Optimization Techniques (~100 lines)

**Techniques:**
- Kernel fusion (2-3× speedup for memory-bound ops)
- Block size tuning (128-256 optimal for most cases)
- Warp efficiency (avoiding thread divergence)
- Asynchronous operations (10-20% speedup)

**Common Anti-Patterns:**
1. Small batch sizes (underutilizes GPU)
2. CPU-GPU synchronization in training loop
3. Inefficient DataLoader (single-threaded, no pinning)

**Production Workflow:**
- Meta's 4-step optimization process
- Distributed profiling best practices
- Profiling overhead considerations

---

## Sources Used

### Primary Web Research (accessed 2025-11-13)

1. **Meta Performance Debugging Blog**
   - URL: https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/
   - Content: MAIProf infrastructure, production case study
   - Key insight: 4-5× speedup with 4 simple optimizations

2. **NSight Systems Tutorial**
   - URL: https://henryhmko.github.io/posts/profiling/profiling.html
   - Content: Practical NSight Systems usage, NVTX annotations
   - Key insight: torch.compile profiling gotchas

3. **CUDA Optimization Checklist**
   - URL: https://medium.com/@rimikadhara/cuda-3-your-checklist-for-optimizing-cuda-kernels-68ef2a42332d
   - Content: Three bottleneck categories, systematic approach
   - Key insight: Compute vs memory vs overhead diagnosis

### Supporting Documentation

- PyTorch Profiler Recipe (official docs)
- NVIDIA CUDA C++ Best Practices Guide (memory coalescing)
- NSight Systems User Guide (referenced, too large to scrape)

---

## Knowledge Gaps Filled

### What Was Missing Before

Existing cuda/ files (00-07) covered:
- CUDA streams and concurrency
- Memory management (unified memory, pinned memory)
- Build system and compilation
- Compute capabilities
- Custom CUDA extensions
- Tensor Core programming
- Mixed precision training

**But didn't cover:**
- Systematic profiling workflow (NSight tools)
- Performance bottleneck diagnosis (compute vs memory vs overhead)
- Production-scale profiling (distributed, minimal overhead)
- Real-world optimization case studies
- Common anti-patterns and gotchas

### What This File Adds

**Practical profiling skills:**
- How to run NSight Systems/Compute/torch.profiler
- Interpreting timeline views and metrics
- Adding NVTX annotations for code mapping

**Diagnostic expertise:**
- Identifying bottleneck category (compute/memory/overhead)
- Measuring SM utilization, Tensor Core usage, bandwidth
- Calculating occupancy and warp efficiency

**Optimization strategies:**
- Kernel fusion for memory-bound operations
- Block size tuning for occupancy
- Memory coalescing for bandwidth
- Asynchronous operations for latency hiding

**Production knowledge:**
- Meta's MAIProf infrastructure design
- Distributed profiling best practices
- Profiling overhead considerations
- Real case study: 9.1% → 74% SM utilization

---

## Expert-Level Content Highlights

**Real Production Metrics:**
- SM utilization targets: >80% (good), <50% (poor)
- Tensor Core utilization: >50% (if using mixed precision)
- Memory bandwidth: >70% of peak (A100: >1.4 TB/s, H100: >2.35 TB/s)
- Cache hit rates: L1 >80%, L2 >70%

**Concrete Speedup Examples:**
- Memory coalescing: 26× faster
- Kernel fusion: 2-3× faster (elementwise ops)
- Mixed precision: 2× faster (A100 Tensor Cores)
- Async transfers: 10-20% speedup
- Meta case study: 4-5× end-to-end training time

**Diagnostic Commands:**
```bash
# NSight Systems profiling
nsys profile --trace=cuda,nvtx,osrt -o output python train.py

# NSight Compute kernel analysis
ncu --set full -o kernel_profile python train.py

# Measure bandwidth
ncu --metrics dram__bytes.sum.per_second python script.py

# Check Tensor Core utilization
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
```

---

## Integration with Existing Knowledge

**Connects to:**
- `cuda/00-streams-concurrency-async.md` - Async profiling for overlap
- `cuda/01-memory-management-unified.md` - Memory profiling techniques
- `cuda/05-tensor-core-programming-wmma-mma.md` - Tensor Core utilization debugging
- `cuda/06-pytorch-jit-torch-compile.md` - torch.compile profiling gotchas
- `cuda/07-mixed-precision-training-internals.md` - AMP profiling and Tensor Core activation
- `karpathy/llm-gpu-integration/00-flashattention-internals.md` - Memory-bound optimization example

**Extends knowledge with:**
- Production profiling workflows (Meta's approach)
- Systematic bottleneck diagnosis (not covered elsewhere)
- Real performance metrics and targets
- Common anti-patterns to avoid

---

## File Statistics

- **Total lines**: ~400
- **Code examples**: 20+
- **Diagnostic commands**: 15+
- **Performance metrics**: 10+ with targets
- **Sources cited**: 6 (3 web, 3 docs)
- **Cross-references**: 6 existing files

---

## Completion Notes

**Step 0: Check existing knowledge** ✓
- Reviewed INDEX.md cuda/ section
- Confirmed profiling not deeply covered
- Identified gap: systematic performance debugging

**Step 1: Web research** ✓
- 4 search queries executed
- 3 URLs scraped successfully (Meta blog, NSight tutorial, CUDA optimization)
- 1 URL too large (NVIDIA docs - referenced instead)

**Step 2: Focus areas covered** ✓
- Profiling tools (NSight Systems, Compute, torch.profiler)
- GPU utilization (SM, Tensor Core, memory)
- Memory bandwidth (HBM, coalescing, cache)
- Kernel optimization (fusion, block size, warp efficiency)
- Python overhead (DataLoader, synchronization)
- Production workflows (Meta's approach)

**Step 3: Content extracted** ✓
- Profiling workflow (NSight commands, NVTX annotations)
- Performance metrics (SM %, Tensor Core %, bandwidth)
- Optimization techniques (fusion, coalescing, async)
- Real case study (Meta production model)
- Diagnostic commands (nsys, ncu, torch.profiler)
- Common anti-patterns

**Step 4: Knowledge file written** ✓
- Section 1: Profiling Tools & Workflow (~100 lines)
- Section 2: GPU Utilization Debugging (~100 lines)
- Section 3: Memory Bandwidth Analysis (~100 lines)
- Section 4: Expert Optimization Techniques (~100 lines)
- All sections cite web sources with URLs and access dates
- Cross-references to existing cuda/ files

**Step 5: KNOWLEDGE DROP created** ✓
- This file
- Documents runner, timestamp, status
- Lists knowledge file created, sources used, gaps filled

---

**Status**: PART 3 complete ✓

**Next**: Oracle will update INDEX.md and move to completed/
