# Oracle Knowledge Expansion: PyTorch CUDA Troubleshooting (Expert-Level)

**Topic**: PyTorch CUDA compilation, debugging, troubleshooting - common and hard problems
**Date**: 2025-11-13
**Runners**: 4 (parallel execution)
**Target**: Expert-level troubleshooting knowledge

---

## PART 1: CUDA Compilation Failures & Build System Hell

- [✓] PART 1: Create cuda/08-compilation-troubleshooting-expert.md (~400 lines) (Completed 2025-11-13 15:45)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md to find existing cuda/ files
- [ ] Grep for "compilation" AND "nvcc" in cuda/ folder
- [ ] Read cuda/02-pytorch-build-system-compilation.md (existing file)
- [ ] Identify knowledge gaps: What compilation errors are NOT covered?

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "PyTorch CUDA compilation errors nvcc 2024 2025"
- [ ] Search: "CUDA architecture mismatch sm_75 sm_80 troubleshooting"
- [ ] Search: "PyTorch build from source CMake errors solutions"
- [ ] Search: "nvcc fatal error no such file or directory"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] nvcc compilation errors (architecture mismatch, missing headers, linker errors)
- [ ] CMake configuration hell (FindCUDA.cmake, CUDA_ARCH_LIST issues)
- [ ] PyTorch build failures (setuptools, ninja, wheel building)
- [ ] Cross-compilation issues (different CUDA versions, driver mismatches)
- [ ] Environment variable problems (PATH, LD_LIBRARY_PATH, CUDA_HOME)
- [ ] Conda vs pip vs source build conflicts

**Step 3: Content to Extract**
- [ ] Common nvcc error messages and solutions
- [ ] Architecture targeting errors (sm_XX not supported)
- [ ] Missing dependencies (CUDA toolkit, cuDNN, NCCL)
- [ ] Build system debugging workflow
- [ ] Diagnostic commands (nvcc --version, nvidia-smi, torch.cuda.is_available())
- [ ] Expert-level fixes (patching CMakeLists.txt, forcing architectures)

**Step 4: Write Knowledge File**
- [ ] Create cuda/08-compilation-troubleshooting-expert.md
- [ ] Section 1: nvcc Compilation Errors (~100 lines)
      Cite: web sources found
      Include: Error messages, root causes, solutions
- [ ] Section 2: CMake & Build System Issues (~100 lines)
      Cite: PyTorch forums, GitHub issues
      Include: Configuration debugging, common pitfalls
- [ ] Section 3: Environment & Dependency Problems (~100 lines)
      Cite: NVIDIA docs, PyTorch installation guides
      Include: PATH issues, version conflicts, diagnostic workflow
- [ ] Section 4: Expert-Level Fixes (~100 lines)
      Cite: Stack Overflow, PyTorch internals
      Include: Manual patches, workarounds, advanced debugging

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-compilation-troubleshooting-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List: Knowledge file created, sources used, gaps filled
- [ ] Describe: Expert compilation debugging knowledge added

---

## PART 2: Runtime CUDA Errors & Memory Failures

- [✓] PART 2: Create cuda/09-runtime-errors-debugging-expert.md (~400 lines) (Completed 2025-11-13 18:45)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "runtime" AND "error" in cuda/ folder
- [ ] Read cuda/01-memory-management-unified.md (existing file)
- [ ] Identify gaps: What runtime errors are NOT covered?

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA out of memory OOM debugging PyTorch 2024"
- [ ] Search: "illegal memory access CUDA error 700 pytorch"
- [ ] Search: "CUDA kernel launch failure troubleshooting"
- [ ] Search: "torch.cuda.OutOfMemoryError solutions expert"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] OOM errors (torch.cuda.OutOfMemoryError, fragmentation, memory leaks)
- [ ] Illegal memory access (CUDA error 700, 77, buffer overruns)
- [ ] Kernel launch failures (invalid configuration, grid/block size errors)
- [ ] Device-side assertions (printf debugging, cuda-gdb)
- [ ] Synchronization errors (race conditions, deadlocks)
- [ ] Multi-GPU errors (peer access, NCCL failures)

**Step 3: Content to Extract**
- [ ] CUDA error codes and meanings (700, 77, 2, 30, etc.)
- [ ] OOM debugging workflow (memory profiler, torch.cuda.memory_summary())
- [ ] Illegal memory access diagnosis (cuda-memcheck, compute-sanitizer)
- [ ] Kernel debugging techniques (printf, cuda-gdb, NSight)
- [ ] Common pitfalls (indexing errors, dtype mismatches)
- [ ] Expert-level recovery strategies

**Step 4: Write Knowledge File**
- [ ] Create cuda/09-runtime-errors-debugging-expert.md
- [ ] Section 1: Out of Memory Errors (~100 lines)
      Cite: PyTorch docs, NVIDIA debugging guides
      Include: Diagnosis, memory profiling, fragmentation fixes
- [ ] Section 2: Illegal Memory Access (~100 lines)
      Cite: CUDA documentation, cuda-memcheck guides
      Include: Error codes, debugging tools, common causes
- [ ] Section 3: Kernel Launch Failures (~100 lines)
      Cite: CUDA programming guides
      Include: Configuration errors, grid/block debugging
- [ ] Section 4: Advanced Debugging Techniques (~100 lines)
      Cite: NSight documentation, expert tutorials
      Include: cuda-gdb, device assertions, profiling workflow

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-runtime-errors-debugging-2025-11-13.md
- [ ] Include: Runner (PART 2), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Expert runtime error debugging knowledge

---

## PART 3: Performance Debugging & Profiling Hell

- [✓] PART 3: Create cuda/10-performance-debugging-profiling-expert.md (~400 lines) (Completed 2025-11-13 20:15)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "performance" AND "profiling" in cuda/ folder
- [ ] Read existing cuda/ files for performance content
- [ ] Identify gaps: What profiling/optimization debugging is missing?

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "PyTorch CUDA kernel slow performance debugging 2024"
- [ ] Search: "NSight Systems PyTorch profiling tutorial"
- [ ] Search: "torch.profiler GPU utilization debugging"
- [ ] Search: "CUDA kernel optimization bottlenecks memory bandwidth"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Profiling tools (NSight Systems, NSight Compute, torch.profiler)
- [ ] GPU utilization problems (kernel starvation, launch overhead)
- [ ] Memory bandwidth bottlenecks (HBM saturation, cache misses)
- [ ] Kernel optimization (occupancy, register spilling, shared memory)
- [ ] Python overhead (GIL, CPU-GPU synchronization)
- [ ] DataLoader bottlenecks (pinned memory, num_workers, prefetching)

**Step 3: Content to Extract**
- [ ] Profiling workflow (NSight Systems timeline analysis)
- [ ] Interpreting profiler output (kernel time, memory bandwidth, occupancy)
- [ ] Common performance issues (low GPU utilization, CPU bottlenecks)
- [ ] Optimization strategies (kernel fusion, memory coalescing)
- [ ] Diagnostic metrics (FLOPS, memory throughput, Tensor Core utilization)
- [ ] Expert-level tuning (block size selection, warp efficiency)

**Step 4: Write Knowledge File**
- [ ] Create cuda/10-performance-debugging-profiling-expert.md
- [ ] Section 1: Profiling Tools & Workflow (~100 lines)
      Cite: NVIDIA NSight docs, PyTorch profiler guides
      Include: NSight Systems, Compute, torch.profiler setup
- [ ] Section 2: GPU Utilization Debugging (~100 lines)
      Cite: CUDA optimization guides
      Include: Kernel starvation, launch overhead, occupancy
- [ ] Section 3: Memory Bandwidth Analysis (~100 lines)
      Cite: NVIDIA memory optimization guides
      Include: HBM saturation, cache optimization, coalescing
- [ ] Section 4: Expert Optimization Techniques (~100 lines)
      Cite: Advanced CUDA tutorials
      Include: Kernel fusion, register tuning, shared memory

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-performance-debugging-2025-11-13.md
- [ ] Include: Runner (PART 3), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Expert performance debugging knowledge

---

## PART 4: Advanced Multi-GPU & Hard Edge Cases

- [✓] PART 4: Create cuda/11-advanced-troubleshooting-multi-gpu-expert.md (~400 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "multi-gpu" AND "distributed" in cuda/ folder
- [ ] Read existing cuda/ files for multi-GPU content
- [ ] Identify gaps: What advanced troubleshooting is missing?

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "PyTorch multi-GPU training errors troubleshooting 2024"
- [ ] Search: "NCCL timeout errors distributed training solutions"
- [ ] Search: "peer-to-peer access CUDA multi-GPU debugging"
- [ ] Search: "PyTorch DDP gradient synchronization failures"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Multi-GPU initialization failures (peer access, NCCL setup)
- [ ] DDP/FSDP synchronization errors (gradient all-reduce failures)
- [ ] NCCL timeouts (network issues, slow nodes)
- [ ] Mixed precision issues across GPUs (gradient scaling, NaN propagation)
- [ ] Hard edge cases (driver bugs, ECC errors, thermal throttling)
- [ ] Production debugging (checkpoint corruption, model divergence)

**Step 3: Content to Extract**
- [ ] Multi-GPU error codes (NCCL errors, peer access failures)
- [ ] DDP debugging workflow (TORCH_DISTRIBUTED_DEBUG)
- [ ] NCCL troubleshooting (network diagnostics, timeout tuning)
- [ ] Gradient synchronization issues (all-reduce failures, bucket sizes)
- [ ] Hard edge cases (ECC errors, GPU lockups, driver crashes)
- [ ] Expert recovery strategies (checkpoint surgery, model surgery)

**Step 4: Write Knowledge File**
- [ ] Create cuda/11-advanced-troubleshooting-multi-gpu-expert.md
- [ ] Section 1: Multi-GPU Initialization & Setup (~100 lines)
      Cite: PyTorch distributed docs, NCCL guides
      Include: Peer access, NCCL setup, common failures
- [ ] Section 2: DDP/FSDP Synchronization Errors (~100 lines)
      Cite: PyTorch DDP internals, debugging guides
      Include: Gradient all-reduce, bucket debugging, timeouts
- [ ] Section 3: NCCL Debugging (~100 lines)
      Cite: NCCL documentation, troubleshooting guides
      Include: Network diagnostics, timeout tuning, error codes
- [ ] Section 4: Hard Edge Cases & Recovery (~100 lines)
      Cite: Expert forums, GPU hardware guides
      Include: ECC errors, thermal issues, checkpoint recovery

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-advanced-troubleshooting-2025-11-13.md
- [ ] Include: Runner (PART 4), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Expert advanced troubleshooting knowledge

---

## Completion Criteria

**All PARTs must:**
- ✅ Check existing knowledge first (avoid duplication)
- ✅ Focus on EXPERT-LEVEL troubleshooting (not basic tutorials)
- ✅ Include real error messages and solutions
- ✅ Cite web sources (accessed 2025-11-13)
- ✅ Provide diagnostic workflows and commands
- ✅ Cover common AND hard edge cases
- ✅ Create KNOWLEDGE DROP file

**Expected output:**
- 4 new files in cuda/ (~400 lines each, ~1,600 lines total)
- Expert-level troubleshooting knowledge
- Real-world error scenarios and solutions
- Diagnostic commands and workflows
- Advanced debugging techniques

---

**Status**: Ready for parallel execution (4 runners)
