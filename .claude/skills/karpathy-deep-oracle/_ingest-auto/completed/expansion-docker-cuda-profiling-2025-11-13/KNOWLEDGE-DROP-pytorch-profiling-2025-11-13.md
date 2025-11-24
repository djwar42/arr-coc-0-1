# KNOWLEDGE DROP: PyTorch Profiling Ecosystem

**Runner**: PART 2
**Timestamp**: 2025-11-13
**Status**: ✅ SUCCESS

---

## Knowledge File Created

**File**: `cuda/11-pytorch-profiling-ecosystem.md`
**Size**: 408 lines
**Topic**: Complete PyTorch profiling tools ecosystem

---

## Content Summary

### Section 1: PyTorch Profiler & Kineto Architecture (120 lines)
- torch.profiler Python API architecture
- Kineto library as C++ profiling backend
- CUPTI dependency explanation (why it's required)
- What breaks without CUPTI (no GPU kernel timelines)
- Basic and advanced profiling usage examples
- Kineto overhead benchmarks (2-15% depending on features)

### Section 2: NSight Tools - CUPTI-Free Profiling (120 lines)
- NSight Systems system-level timeline profiling
- NSight Compute kernel-level deep-dive analysis
- Key distinction: NSight tools DON'T require CUPTI
- Comparison table: torch.profiler vs NSight Systems vs NSight Compute
- When to use which tool (decision matrix)
- NVTX annotation patterns for PyTorch

### Section 3: Profiling Without CUPTI (80 lines)
- Why CUPTI might be missing (Docker runtime images, cloud environments)
- Fallback Strategy 1: NSight Systems (recommended alternative)
- Fallback Strategy 2: CPU-only profiling with torch.profiler
- Fallback Strategy 3: Manual CUDA event timing (zero dependencies)
- Fallback Strategy 4: Simple timing wrapper (production-safe)

### Section 4: Production Profiling Strategies (80 lines)
- Meta's Dynolog + PyTorch Profiler automated approach
- Zero code instrumentation profiling (KINETO_USE_DAEMON)
- Production best practices (short bursts, profiler schedule)
- Docker multi-stage builds (runtime vs devel for CUPTI)
- TensorBoard integration patterns
- HTA (Holistic Trace Analysis) for batch analysis
- Minimal-overhead production monitoring (<0.5% overhead)

---

## Web Research Sources

### PyTorch Official (5 sources)
1. [PyTorch Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html) - API reference and core concepts
2. [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Practical usage examples
3. [Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) - Meta's production approach
4. [Introducing Profiler blog](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) - Profiler architecture
5. [Kineto GitHub](https://github.com/pytorch/kineto) - Profiling library internals

### NVIDIA Profiling Tools (3 sources)
6. [NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/) - System-level profiling
7. [NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Kernel analysis
8. [NVIDIA Forums: Profiler Comparison](https://forums.developer.nvidia.com/t/nsight-compute-vs-nsight-systems-vs-pytorch-profiler/283271) - Community discussion

### Production Debugging (3 sources)
9. [Meta Production Debugging](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/) - MAIProf architecture
10. [HTA Documentation](https://hta.readthedocs.io/) - Trace analysis library
11. [GPU Profiling Survey](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/) - Technical deep-dive

### Additional Research (1 source)
12. [Argonne Lab: PyTorch Profiler for AI](https://www.alcf.anl.gov/sites/default/files/2025-05/PyTorchProfiler_HZheng.pdf) - HPC perspective

---

## Production Context

**CUPTI Investigation Connection** (arr-coc-0-1):

This knowledge file directly addresses the CUPTI profiling challenges discovered during arr-coc-0-1 Cloud Build investigation:

1. **Problem**: torch.profiler requires CUPTI, but nvidia/cuda:*-runtime-* images don't include it
2. **Solution 1**: Use NSight Systems (doesn't need CUPTI) for production profiling
3. **Solution 2**: Add cuda-cupti-12-4 package to runtime images (minimal overhead)
4. **Solution 3**: Multi-stage Docker builds (runtime for prod, devel for profiling)
5. **Solution 4**: Fallback to CPU-only profiling or manual CUDA events

**Key production insights**:
- **Meta's approach**: Zero code instrumentation via Dynolog + optimizer hooks
- **Overhead targets**: <0.5% for always-on monitoring, 5-10% for detailed profiling
- **Profiling strategy**: Short bursts (3-5 iterations) rather than continuous
- **Tool selection**: torch.profiler for PyTorch-specific, NSight Systems for system-wide

---

## Key Technical Insights

### Architecture Differences

**torch.profiler stack**:
```
Python API (torch.profiler)
    ↓
Kineto (C++ profiling library)
    ↓
CUPTI (NVIDIA GPU profiling interface)
    ↓
GPU hardware counters
```

**NSight Systems stack**:
```
NSight Systems CLI
    ↓
GPU driver hooks (NOT CUPTI)
    ↓
GPU kernel start/end events
```

### When CUPTI Fails

**Typical scenario**: Docker runtime image profiling

```bash
# Runtime image (1.5GB, no CUPTI)
nvidia/cuda:12.4.0-runtime-ubuntu22.04
    ↓
torch.profiler fails with CUPTI warning
    ↓
Fallback: Use NSight Systems (works without CUPTI)

# OR add CUPTI to runtime
apt-get install cuda-cupti-12-4  # Only ~200MB
```

### Profiling Overhead Spectrum

| Tool | Overhead | Use Case |
|------|----------|----------|
| Manual CUDA events | ~0.1% | Always-on monitoring |
| NSight Systems | 2-3% | System bottlenecks |
| torch.profiler (CPU) | 2-5% | PyTorch operator analysis |
| torch.profiler (GPU) | 5-10% | Full training profiling |
| torch.profiler (memory) | 8-15% | Memory leak debugging |
| NSight Compute | Very high | Single kernel optimization |

---

## Files Created

1. **cuda/11-pytorch-profiling-ecosystem.md** (408 lines)
   - Complete profiling tools ecosystem guide
   - torch.profiler + Kineto architecture
   - NSight Systems/Compute alternatives
   - Profiling without CUPTI strategies
   - Production deployment patterns

2. **KNOWLEDGE-DROP-pytorch-profiling-2025-11-13.md** (this file)
   - Execution summary
   - Web research sources
   - Production context
   - Technical insights

---

## Next Steps

This knowledge complements existing CUDA profiling files:

- **cuda/10-performance-debugging-profiling-expert.md** - Performance debugging workflows (references torch.profiler)
- **cuda/12-kernel-debugging-internals-expert.md** - NSight Compute debugging (references interactive profiling)
- **cuda/16-assembly-level-debugging-ptx-sass-expert.md** - NSight Compute SASS analysis

**Cross-references to add**:
- Link from cuda/10 to this file for profiler tool selection
- Link from cuda/12 to Section 2 for NSight Compute usage
- Reference this file from INDEX.md under CUDA profiling section

---

**PART 2 execution complete** ✅
