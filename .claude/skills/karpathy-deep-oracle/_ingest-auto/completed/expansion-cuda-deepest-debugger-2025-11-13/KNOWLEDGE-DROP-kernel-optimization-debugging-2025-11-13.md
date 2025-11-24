# KNOWLEDGE DROP: Custom Kernel Optimization Debugging

**Runner**: PART 4
**Timestamp**: 2025-11-13 (execution completed)
**Status**: ✓ SUCCESS

---

## Knowledge File Created

**File**: `cuda/19-custom-kernel-optimization-debugging-expert.md`
**Size**: ~600 lines
**Sections**: 4 major sections

### Section Breakdown

**Section 1: Occupancy Analysis & Tuning (~150 lines)**
- Theoretical vs measured occupancy
- Five occupancy limiters (registers, shared memory, block size, max blocks, barriers)
- Register reduction techniques
- CUDA occupancy calculator usage
- Launch rate limited occupancy
- Register spilling detection

**Section 2: Shared Memory Bank Conflict Detection (~150 lines)**
- 32-bank architecture fundamentals
- NSight Compute bank conflict detection (3 methods)
- Bank conflict patterns & solutions (strided access, SOA vs AOS, 64/128-bit accesses)
- Hardware counter discrepancies and workarounds
- Bank conflict optimization checklist

**Section 3: Warp Efficiency & Stall Reasons (~150 lines)**
- Eight primary warp stall reasons (memory throttle, exec dependency, scoreboard, etc.)
- NSight Compute warp state analysis
- Warp divergence detection & mitigation
- Latency hiding through ILP and TLP
- Warp execution efficiency optimization workflow

**Section 4: Memory Access Optimization (~150 lines)**
- Global memory coalescing fundamentals
- NSight Compute memory metrics (sector utilization, L1/TEX hit rate)
- Memory access pattern optimization (SOA, alignment, vectorization)
- Shared memory usage patterns
- Cache optimization strategies (L1, L2, read-only cache)
- Advanced techniques (async copy, prefetching)

---

## Sources Used

**Web Research (accessed 2025-11-13):**

1. **Medium - CUDA Optimization Checklist**
   - URL: https://medium.com/@rimikadhara/cuda-3-your-checklist-for-optimizing-cuda-kernels-68ef2a42332d
   - Content: Occupancy fundamentals, compute vs memory bound kernels

2. **NVIDIA Developer Forums - Bank Conflicts**
   - URL: https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731
   - Content: NSight Compute bank conflict detection methods, hardware counter issues

3. **AMD GPUOpen - Occupancy Explained**
   - URL: https://gpuopen.com/learn/occupancy-explained/
   - Content: Occupancy definition, latency hiding, limiters (VGPR, LDS, barriers)

4. **MoldStud - CUDA Occupancy Optimization**
   - URL: https://moldstud.com/articles/p-optimizing-cuda-occupancy-discovering-the-best-gpu-configuration-for-performance
   - Content: Block size tuning, grid size optimization, register pressure analysis

5. **Massed Compute - Register Usage Reduction**
   - URL: https://massedcompute.com/faq-answers/?question=How%20can%20I%20reduce%20register%20usage
   - Content: Seven techniques for reducing register pressure

6. **NVIDIA Developer Forums - Warp Stall Reasons**
   - URL: https://forums.developer.nvidia.com/t/stalll-reasons/121598
   - Content: Warp execution efficiency metrics, stall reason breakdown

**Note**: NSight Compute documentation scrape exceeded 25K token limit. Used forum discussions and other sources for NSight metrics.

---

## Gaps Filled

### Before PART 4
Existing CUDA knowledge (files 08-15) covered:
- Basic kernel debugging (file 12)
- Performance profiling with NSight (file 10)
- Multi-GPU troubleshooting (file 11)
- Production deployment (file 15)

**Missing:**
- Deep occupancy analysis (theoretical vs measured, limiters)
- Systematic bank conflict detection & debugging
- Comprehensive warp stall reason analysis
- Memory coalescing optimization workflow

### After PART 4
**Added expert-level knowledge:**
- ✓ Occupancy calculator usage (programmatic API + Excel)
- ✓ Five occupancy limiters with specific reduction strategies
- ✓ Register spilling detection and prevention
- ✓ NSight Compute bank conflict detection (3 independent methods)
- ✓ Bank conflict patterns with code examples (padding, SOA conversion)
- ✓ Eight warp stall reasons with targeted optimizations
- ✓ Warp divergence mitigation patterns
- ✓ ILP vs TLP trade-offs for latency hiding
- ✓ Memory coalescing verification metrics
- ✓ Advanced memory optimization (async copy, prefetching, cache hints)

---

## Absolute Deepest Custom Kernel Debugging

This KNOWLEDGE DROP completes the ABSOLUTE DEEPEST level of custom kernel optimization debugging expertise:

**What Sets This Apart:**
1. **NSight Compute Mastery**: Three independent methods for bank conflict detection (SOL, Warp Stats, Source Page)
2. **Occupancy Nuance**: Understanding when high occupancy HURTS performance (cache thrashing, memory bound)
3. **Hardware Counter Discrepancies**: Knowing when metrics lie (bank conflict counter issues on Turing)
4. **Systematic Workflows**: Decision trees for occupancy → bank conflicts → warp stalls → memory access
5. **Real Production Scenarios**: Register spilling detection, launch rate limiting, 64KB shared memory effects
6. **Quantified Impact**: Performance gain ranges for each optimization (2-4× for coalescing, 3-10× for shared memory)

**Debugging Philosophy:**
- Occupancy is a diagnostic lens, not a goal
- Always profile before optimizing
- Fix highest-impact bottleneck first (systematic prioritization)
- Verify improvements with metrics, not assumptions

This knowledge enables debugging the hardest custom kernel performance issues in production ML training and inference workloads.
