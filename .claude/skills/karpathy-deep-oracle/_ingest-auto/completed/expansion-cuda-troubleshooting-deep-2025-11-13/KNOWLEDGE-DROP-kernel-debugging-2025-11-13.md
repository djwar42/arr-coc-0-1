# KNOWLEDGE DROP: CUDA Kernel Debugging Internals

**Runner**: PART 1 (expansion-cuda-troubleshooting-deep)
**Timestamp**: 2025-11-13 (execution timestamp)
**Status**: ✅ COMPLETE

---

## Knowledge File Created

**File**: `cuda/12-kernel-debugging-internals-expert.md`
**Size**: ~500 lines (7,638 words)
**Depth**: ULTRA-EXPERT (advanced debugging workflows, production patterns)

---

## Coverage Summary

### Section 1: cuda-gdb Fundamentals (~125 lines)
- cuda-gdb command reference (break, info cuda, cuda thread, cuda block)
- Kernel breakpoint syntax (conditional breakpoints, register breakpoints)
- Thread/warp/block navigation commands
- Memory inspection (shared, local, global with proper casting)
- Step commands and execution control
- cuda-gdb workflow for index error debugging
- Limitations and gotchas (read-only memory, focus changes)

### Section 2: Device-Side Debugging (~125 lines)
- **printf() debugging**: Buffer size limits (1MB default, expandable to 8MB)
- **Argument limits**: 32 arguments maximum (silent failure beyond)
- **Performance impact**: ~1000 cycles per printf, serializes warp execution
- **Conditional printf patterns**: Thread-selective, sampling, error-only
- **Device-side assertions**: assert() usage, trap() for immediate termination
- **TORCH_USE_CUDA_DSA**: PyTorch device-side assertion debugging
- **CUDA_LAUNCH_BLOCKING**: Synchronous error reporting
- **Hexadecimal dumps** for bit-level debugging

### Section 3: NSight Compute Interactive Debugging (~125 lines)
- **Source View debugging**: Correlate CUDA C++ with PTX/SASS assembly
- **Warp analysis**: Detect warp divergence (pred_on vs pred_off metrics)
- **Kernel replay behavior**: Runs kernels 5-10 times for metric collection
- **Side effect warnings**: printf appears multiple times under NSight
- **Replay modes**: `--replay-mode application` (single run) vs `kernel` (multiple)
- **Memory access pattern debugging**: Bank conflicts, coalescing metrics
- **Race condition detection with inconsistent replays**

### Section 4: Advanced Debugging Techniques (~125 lines)
- **Race condition detection**: Intra-warp (impossible) vs inter-warp (possible)
- **compute-sanitizer racecheck**: Detect shared memory race hazards
- **Minimal overhead debugging**: Selective instrumentation with debug macros
- **CUDA event timing**: < 1 μs overhead for performance measurement
- **Memory corruption debugging**: compute-sanitizer memcheck for illegal access
- **Conditional breakpoints**: Thread-specific, register-based, memory watchpoints
- **Warp shuffle debugging**: Active mask requirements, assertion patterns
- **Production debugging workflow**: 5-step process (minimal test → sanitizer → cuda-gdb → NSight)

---

## Sources Used

**Existing Knowledge**:
- cuda/09-runtime-errors-debugging-expert.md (basic cuda-gdb intro, lines 270-296)

**Web Research** (11 primary sources):
1. **NVIDIA cuda-gdb Documentation** - Complete command reference
2. **NVIDIA NSight Compute Profiling Guide** - Interactive debugging workflows
3. **NSight Visual Studio State Inspection** - Warp/thread inspection details
4. **NVIDIA Developer Forums: printf buffer size** - 1MB default, 8MB max, silent truncation
5. **NVIDIA Developer Forums: assert in CUDA** - Device-side assertion behavior
6. **NVIDIA Developer Forums: Race condition within warp** - Warp synchronization guarantees
7. **NVIDIA Developer Blog: Compute Sanitizer** - Modern debugging tool workflows
8. **PyTorch Forums: device-side assert** - TORCH_USE_CUDA_DSA usage
9. **Stack Overflow: Printf 45 arguments** - 32 argument limit discovery
10. **Stack Overflow: NSight kernel replay** - Multiple kernel runs side effects
11. **NASA HECC: NSight Compute** - Source view debugging techniques

**Additional References**:
- Carnegie Mellon cuda-gdb PDF
- Forschungszentrum Jülich CUDA Tools
- SHARCNET Debugging Guide
- Massed Compute practical examples

---

## Gaps Filled

**Beyond existing cuda/09-runtime-errors-debugging-expert.md:**

1. **Deep cuda-gdb workflows**:
   - Conditional breakpoints (thread-specific, register-based)
   - Memory watchpoints for device memory
   - Navigation between threads/warps/blocks with `info cuda` commands

2. **printf debugging patterns**:
   - Buffer size management (cudaDeviceSetLimit)
   - 32 argument hard limit (not documented in Programming Guide!)
   - Performance overhead quantified (~1000 cycles)
   - Production-safe conditional printf patterns

3. **NSight Compute debugging** (not just profiling):
   - Source View as debugging tool (correlate C++ ↔ PTX ↔ SASS)
   - Kernel replay behavior (5-10 runs) and side effects
   - Warp divergence visualization (pred_on/pred_off metrics)
   - Race condition detection through replay inconsistency

4. **Advanced production techniques**:
   - Minimal overhead debugging (debug macros, selective instrumentation)
   - Warp shuffle assertion patterns
   - 5-step production debugging workflow
   - compute-sanitizer tool suite (memcheck, racecheck, synccheck)

---

## Knowledge Depth

**ULTRA-EXPERT level achieved**:
- ✅ Beyond basic "how to use cuda-gdb" tutorials
- ✅ Production debugging patterns (minimal overhead, selective instrumentation)
- ✅ Hard-to-find limits (32 printf args, 1MB buffer, 1000 cycle overhead)
- ✅ NSight Compute as debugger (not just profiler)
- ✅ Real-world debugging workflows (5-step process from minimal test to NSight)
- ✅ Edge cases (warp shuffle active masks, race detection via replay inconsistency)

**Complements existing cuda/ knowledge**:
- cuda/08-compilation-troubleshooting-expert.md (build-time issues)
- cuda/09-runtime-errors-debugging-expert.md (high-level error categories)
- cuda/10-performance-debugging-profiling-expert.md (performance tools)
- **cuda/12-kernel-debugging-internals-expert.md** (deep kernel-level debugging) ← **NEW**

---

## Practical Value

**Immediately useful for**:
1. **Debugging kernel crashes**: cuda-gdb workflows for illegal memory access
2. **Fixing race conditions**: compute-sanitizer racecheck + atomic patterns
3. **Optimizing warp divergence**: NSight Compute Source View + metrics
4. **Production debugging**: Minimal overhead patterns, selective instrumentation
5. **PyTorch CUDA errors**: TORCH_USE_CUDA_DSA + CUDA_LAUNCH_BLOCKING setup

**Target audience**:
- ML engineers debugging custom CUDA kernels (PyTorch extensions, TorchScript fusions)
- VLM developers optimizing vision transformers (warp divergence in attention kernels)
- Production ML teams needing minimal-overhead debugging in deployed systems

---

**PART 1 execution complete** ✓

Next: Oracle will assign PART 2, 3, or 4 (memory leak profiling, mixed precision debugging, or production deployment troubleshooting).
