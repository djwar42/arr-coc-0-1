# Knowledge Drop: PTX/SASS Assembly-Level Debugging

**Runner**: PART 1
**Timestamp**: 2025-11-13
**Status**: ✓ COMPLETE

---

## Knowledge File Created

**File**: `cuda/16-assembly-level-debugging-ptx-sass-expert.md`
**Lines**: ~650 lines
**Depth**: ABSOLUTE DEEPEST assembly-level debugging expertise

---

## Content Summary

### Section 1: PTX/SASS Fundamentals & Tools
- Compilation pipeline: CUDA C++ → PTX → SASS → GPU execution
- PTX (virtual ISA) vs SASS (hardware ISA) differences
- cuobjdump detailed commands for PTX/SASS extraction
- PTX ISA fundamentals: virtual registers, special registers, instruction patterns
- SASS architecture-specific instructions (sm_70 to sm_121)
- ISA version differences: Volta, Turing, Ampere, Ada, Hopper, Blackwell

### Section 2: Register & Occupancy Analysis
- ptxas verbose output interpretation (register usage, spilling detection)
- Register spilling: detection, impact measurement, mitigation strategies
- Occupancy calculation from register usage (with A100 examples)
- Per-instruction register allocation analysis
- Occupancy vs performance trade-offs (memory-bound vs compute-bound)

### Section 3: Instruction-Level Optimization
- NSight Compute SASS instruction throughput metrics
- Dual-issue and warp scheduler behavior (4 schedulers per SM)
- Latency hiding: ILP (instruction-level) vs TLP (thread-level parallelism)
- SASS optimization patterns: alignment, predication, reuse flags
- Warp divergence analysis at SASS level

### Section 4: Advanced Assembly Debugging
- Inline PTX assembly syntax and constraint letters (r, l, f, d, h, c)
- Volatile keyword for side effects and memory clobbering
- Advanced inline PTX examples: warp shuffle, custom atomics, memory fences
- Debugging inline PTX issues: constraint mismatches, missing volatile
- Production 7-step assembly debugging workflow
- SASS-level race condition detection with compute-sanitizer
- Architecture-specific debugging: Hopper WGMMA, Ada FP8

---

## Sources Used

**Source Documents** (existing knowledge):
- cuda/05-tensor-core-programming-wmma-mma.md - PTX mma.sync, WMMA, LDMATRIX
- cuda/10-performance-debugging-profiling-expert.md - NSight Compute, optimization
- cuda/12-kernel-debugging-internals-expert.md - Source View, cuda-gdb

**Web Research** (Bright Data MCP, accessed 2025-11-13):
- Tutorial: Understanding GPU Assembly with PTX - eunomia-bpf
  * PTX fundamentals, inline assembly, hands-on examples
- Reversing Nvidia GPU's SASS code - JEB Decompiler
  * SASS ISA sm_70-121, cuobjdump usage, register types
- NVIDIA Docs: Inline PTX Assembly
  * Constraint letters, volatile semantics
- NSight Compute Profiling Guide
  * Instruction throughput, Source View correlation
- NVIDIA Developer Blog: Register Spilling
  * CUDA 13.0 ptxas optimization
- Stack Overflow: ptxas output interpretation
  * Register usage, spill detection
- Stack Overflow: inline PTX volatile
  * Memory clobbering, side effects

---

## Gaps Filled

**Before PART 1:**
- Limited PTX/SASS ISA documentation (only mentioned in cuda/05)
- No cuobjdump detailed usage
- No register pressure analysis workflows
- No inline PTX debugging guidance

**After PART 1:**
- Complete PTX/SASS fundamentals (virtual vs hardware ISA)
- Comprehensive cuobjdump command reference
- ptxas verbose output interpretation expertise
- Register spilling detection and mitigation
- Occupancy calculation from register usage
- Instruction throughput analysis workflows
- Inline PTX assembly syntax and constraints
- Production assembly debugging 7-step workflow
- Architecture-specific debugging (sm_70 to sm_121)

---

## Absolute Deepest Expertise Achieved

This knowledge file provides **ABSOLUTE DEEPEST** assembly-level debugging:

1. **PTX/SASS binary analysis** - cuobjdump extraction, ISA differences
2. **Register pressure diagnosis** - ptxas -v output, spilling detection
3. **Instruction-level optimization** - throughput metrics, dual-issue
4. **Inline assembly debugging** - constraints, volatile, side effects
5. **Production workflows** - 7-step debugging, NSight Compute correlation
6. **Architecture-specific** - Volta through Blackwell differences

Ready for **production VLM training** workloads at hyperscale (1000s of GPUs) with assembly-level performance debugging.
