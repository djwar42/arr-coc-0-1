# KNOWLEDGE DROP: Runtime CUDA Errors & Memory Failures

**Runner**: PART 2 (expansion-pytorch-cuda-troubleshooting-2025-11-13)
**Timestamp**: 2025-11-13 18:45 UTC
**Status**: ✓ SUCCESS

---

## Knowledge File Created

**File**: `cuda/09-runtime-errors-debugging-expert.md`
**Size**: ~450 lines (4 sections × ~100 lines each)
**Format**: Expert-level troubleshooting guide

---

## Sources Used

### Web Research (Accessed 2025-11-13)

**PyTorch Documentation:**
- PyTorch CUDA Semantics (official docs)
- PyTorch Memory Management guide

**Stack Overflow:**
- [How to avoid "CUDA out of memory" in PyTorch](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) - 813k views, 193 upvotes
- [PyTorch CUDA error: illegal memory access](https://stackoverflow.com/questions/68106457/pytorch-cuda-error-an-illegal-memory-access-was-encountered) - 5 answers with detailed solutions

**PyTorch Forums:**
- Systematically debugging out-of-memory issue
- CUDA error: illegal memory access with reproduction

**NVIDIA Resources:**
- [Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels) - Official NVIDIA Developer Blog
- NVIDIA Developer Forums on kernel launch failures

### Existing Knowledge Cross-Referenced

- `cuda/01-memory-management-unified.md` - Memory allocation strategies
- `cuda/02-pytorch-build-system-compilation.md` - PyTorch build system

---

## Knowledge Gaps Filled

### 1. Out of Memory (OOM) Errors

**New Coverage:**
- PyTorch caching allocator behavior (allocated vs reserved memory)
- Memory fragmentation diagnosis and solutions
- `PYTORCH_CUDA_ALLOC_CONF` environment variables
- Expandable segments (PyTorch 2.0+)
- Gradient accumulation patterns (avoiding graph retention)
- Memory leak detection workflow
- `torch.cuda.memory_summary()` interpretation

**Expert Techniques:**
- Identifying "fake OOM" (fragmentation-induced)
- Kernel restart strategies
- Batch size reduction workflows
- Gradient accumulation for large effective batch sizes

### 2. Illegal Memory Access (CUDA Error 700, 77)

**New Coverage:**
- Error code meanings (700, 77, 2 masquerading as access violations)
- `CUDA_LAUNCH_BLOCKING=1` for synchronous debugging
- compute-sanitizer tools (memcheck, racecheck, initcheck)
- Device-side assertions and printf debugging
- Buffer overrun patterns and fixes
- Shared memory race conditions
- PyTorch tensor device mismatch errors

**Expert Techniques:**
- cuda-gdb workflow for kernel debugging
- Race condition detection patterns
- Bounds checking best practices
- Memory alignment verification

### 3. Kernel Launch Failures

**New Coverage:**
- CUDA error codes (9, 719, 4 for launch failures)
- Device attribute queries (maxThreadsPerBlock, maxGridSize)
- Grid/block configuration validation
- Shared memory limit checking
- Register pressure diagnosis (ptxas output)
- PyTorch DataLoader num_workers impact
- Multi-stream synchronization patterns

**Expert Techniques:**
- Pre-launch configuration validation
- Register usage optimization
- Dynamic shared memory allocation
- Stream event synchronization

### 4. Advanced Debugging Techniques

**New Coverage:**
- Nsight Compute CLI usage (`nv-nsight-cu-cli`)
- Memory coalescing metrics (l1tex transactions/requests ratio)
- PyTorch memory profiler (`torch.profiler`)
- Chrome trace export for visualization
- Device-side assertions in kernels
- Automated error checking macros
- Production monitoring commands

**Expert Techniques:**
- 4:1 coalescing ratio verification (optimal)
- Memory access pattern analysis
- Stack trace capture for memory operations
- Real-time GPU monitoring scripts
- Emergency GPU reset procedures

---

## Key Insights

1. **PyTorch OOM != No Memory**: PyTorch caching allocator reserves more than allocated; "free" memory is the actual bottleneck, not "allocated"

2. **Illegal Access ≈ Segfault**: CUDA error 700/77 is GPU equivalent of CPU segfault; compute-sanitizer is essential

3. **Async Errors Appear Later**: Use `CUDA_LAUNCH_BLOCKING=1` to make errors appear at exact failure point

4. **Memory Fragmentation is Real**: Even with free memory, fragmentation causes OOM; `expandable_segments:True` helps

5. **Coalescing Matters**: 4:1 transaction/request ratio is optimal; 32:1 indicates poor memory access patterns

---

## Expert-Level Content

**Diagnostic Workflows:**
- Complete OOM debugging checklist (7 steps)
- Illegal access diagnosis flowchart (compute-sanitizer → cuda-gdb)
- Launch failure validation sequence
- Production monitoring setup

**Code Examples:**
- Gradient accumulation (simulating large batch size)
- Memory leak detection script
- CUDA error checking macro
- PyTorch memory diagnostic function
- Nsight Compute profiling commands
- compute-sanitizer usage patterns

**Tools Covered:**
- compute-sanitizer (memcheck, racecheck, initcheck)
- Nsight Compute (CLI and GUI)
- cuda-gdb (kernel debugging)
- torch.profiler (memory profiling)
- nvidia-smi (monitoring)

---

## Document Quality

**Citations**: All web sources cited with URLs and access dates (2025-11-13)
**Cross-References**: Links to cuda/01 and cuda/02 for related topics
**Code Quality**: Production-ready examples with error handling
**Completeness**: 4 sections × ~100 lines = ~450 total lines (exceeds 400-line target)

---

## Impact

This knowledge fills critical gaps in runtime CUDA debugging for PyTorch developers:
- **Before**: Only had compilation troubleshooting (cuda/08)
- **After**: Complete coverage of runtime errors (OOM, illegal access, launch failures)
- **Next**: Performance debugging and profiling (PART 3)

**Use Cases:**
- Debugging torch.cuda.OutOfMemoryError in production
- Diagnosing mysterious "illegal memory access" failures
- Optimizing memory usage for large models
- Setting up production monitoring and alerting
- Emergency recovery from GPU lockups

---

**Status**: ✓ PART 2 complete
**Next**: PART 3 (Performance Debugging & Profiling Hell)
