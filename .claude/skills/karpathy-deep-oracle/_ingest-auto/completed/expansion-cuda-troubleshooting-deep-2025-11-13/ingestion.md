# Oracle Knowledge Expansion: CUDA Troubleshooting DEEP (Ultra-Expert)

**Topic**: Advanced CUDA debugging, memory profiling, mixed precision stability, production deployment
**Date**: 2025-11-13
**Runners**: 4 (parallel execution)
**Target**: ULTRA-EXPERT troubleshooting (beyond common issues)

---

## PART 1: CUDA Kernel Debugging Internals (cuda-gdb, printf, NSight)

- [✓] PART 1: Create cuda/12-kernel-debugging-internals-expert.md (~500 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "cuda-gdb" AND "kernel debugging" in cuda/ folder
- [ ] Read cuda/09-runtime-errors-debugging-expert.md (mentions cuda-gdb)
- [ ] Identify gaps: Deep cuda-gdb workflows, printf debugging patterns

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "cuda-gdb kernel debugging tutorial 2024 2025"
- [ ] Search: "CUDA device-side assertions printf debugging"
- [ ] Search: "NSight Compute kernel debugging workflow"
- [ ] Search: "CUDA kernel breakpoints thread inspection"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] cuda-gdb fundamentals (launching, attaching, kernel breakpoints)
- [ ] Thread-level debugging (inspecting warp state, register values)
- [ ] Device-side assertions (assert(), trap(), CUDA error handling)
- [ ] printf debugging patterns (format specifiers, buffer limits)
- [ ] NSight Compute interactive debugging (kernel replay, warp analysis)
- [ ] Debugging race conditions (warp divergence, shared memory conflicts)
- [ ] Advanced breakpoint techniques (conditional breaks, watchpoints)

**Step 3: Content to Extract**
- [ ] cuda-gdb command reference (break, info cuda, thread, print)
- [ ] Kernel state inspection (registers, shared memory, local memory)
- [ ] Thread/warp/block navigation commands
- [ ] Device-side printf best practices (performance impact, buffer management)
- [ ] NSight Compute Source view debugging
- [ ] Common kernel bug patterns (index errors, race conditions, memory corruption)
- [ ] Production debugging workflows (minimal overhead, selective instrumentation)

**Step 4: Write Knowledge File**
- [ ] Create cuda/12-kernel-debugging-internals-expert.md
- [ ] Section 1: cuda-gdb Fundamentals (~125 lines)
      Cite: NVIDIA cuda-gdb docs, tutorials
      Include: Command reference, kernel breakpoints, thread inspection
- [ ] Section 2: Device-Side Debugging (~125 lines)
      Cite: CUDA C programming guide, assertion docs
      Include: printf patterns, assertions, trap handling
- [ ] Section 3: NSight Compute Interactive Debugging (~125 lines)
      Cite: NSight Compute docs, debugging guides
      Include: Source view, warp analysis, kernel replay
- [ ] Section 4: Advanced Debugging Techniques (~125 lines)
      Cite: Expert tutorials, production debugging guides
      Include: Race condition detection, memory corruption, minimal overhead

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-kernel-debugging-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List: Knowledge file created, sources used, gaps filled
- [ ] Describe: Ultra-expert kernel debugging knowledge

---

## PART 2: Memory Leak Detection & Advanced Profiling

- [✓] PART 2: Create cuda/13-memory-leak-profiling-expert.md (~505 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "memory leak" AND "profiling" in cuda/ folder
- [ ] Read cuda/01-memory-management-unified.md (basic memory management)
- [ ] Read cuda/09-runtime-errors-debugging-expert.md (mentions leak detection)
- [ ] Identify gaps: Deep leak detection, persistent allocations, fragmentation analysis

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA memory leak detection compute-sanitizer 2024"
- [ ] Search: "PyTorch CUDA memory profiling persistent allocations"
- [ ] Search: "CUDA memory fragmentation patterns analysis"
- [ ] Search: "NSight Systems memory allocation tracking"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Memory leak detection tools (compute-sanitizer memcheck, NSight Systems)
- [ ] PyTorch-specific leaks (gradient accumulation, caching allocator issues)
- [ ] Persistent allocations (tracking lifetime, finding retention sources)
- [ ] Fragmentation analysis (visualizing memory blocks, identifying patterns)
- [ ] Memory timeline profiling (allocation/deallocation tracking)
- [ ] Reference counting issues (Python reference leaks affecting GPU memory)
- [ ] Production memory monitoring (real-time tracking, alerting)

**Step 3: Content to Extract**
- [ ] compute-sanitizer memcheck workflow (leak reports, stack traces)
- [ ] PyTorch memory profiler (torch.cuda.memory._record_memory_history)
- [ ] Memory snapshot analysis (pickle files, flamegraphs)
- [ ] Fragmentation metrics (free blocks distribution, largest contiguous)
- [ ] Persistent allocation tracking (weak references, gc module)
- [ ] Memory timeline visualization (NSight Systems memory view)
- [ ] Production monitoring patterns (prometheus metrics, alerting)

**Step 4: Write Knowledge File**
- [ ] Create cuda/13-memory-leak-profiling-expert.md
- [ ] Section 1: Memory Leak Detection Tools (~125 lines)
      Cite: compute-sanitizer docs, PyTorch memory profiler
      Include: Leak detection workflow, stack trace analysis
- [ ] Section 2: PyTorch-Specific Memory Issues (~125 lines)
      Cite: PyTorch forums, memory debugging guides
      Include: Caching allocator, gradient retention, reference cycles
- [ ] Section 3: Fragmentation Analysis (~125 lines)
      Cite: CUDA memory management docs
      Include: Block visualization, patterns, mitigation strategies
- [ ] Section 4: Production Memory Monitoring (~125 lines)
      Cite: Production monitoring guides
      Include: Real-time tracking, alerting, emergency procedures

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-memory-leak-profiling-2025-11-13.md
- [ ] Include: Runner (PART 2), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Ultra-expert memory profiling knowledge

---

## PART 3: Mixed Precision Debugging & Stability Issues

- [✓] PART 3: Create cuda/14-mixed-precision-debugging-expert.md (~500 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "mixed precision" AND "debugging" in cuda/ folder
- [ ] Read cuda/07-mixed-precision-training-internals.md (basic AMP)
- [ ] Identify gaps: NaN debugging, gradient underflow, stability patterns

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "PyTorch AMP NaN debugging gradient underflow 2024"
- [ ] Search: "mixed precision training stability issues solutions"
- [ ] Search: "FP16 vs BF16 numerical stability debugging"
- [ ] Search: "Transformer Engine FP8 debugging NaN propagation"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] NaN detection and debugging (torch.isnan, backward hooks)
- [ ] Gradient underflow/overflow (dynamic scaling issues, GradScaler tuning)
- [ ] Numerical instability patterns (FP16 range limits, BF16 advantages)
- [ ] Loss scaling strategies (fixed vs dynamic, growth interval tuning)
- [ ] FP8 debugging (E4M3 vs E5M2, quantization errors)
- [ ] Layer-specific precision policies (which layers need FP32?)
- [ ] Mixed precision with distributed training (gradient sync precision)

**Step 3: Content to Extract**
- [ ] NaN debugging workflow (detection, isolation, root cause)
- [ ] Gradient scaling diagnostics (growth_interval, backoff_factor)
- [ ] Precision format comparison (numerical stability, range, accuracy)
- [ ] FP16 instability patterns (LayerNorm, Softmax, log operations)
- [ ] BF16 migration strategies (when to switch from FP16)
- [ ] FP8 training debugging (Transformer Engine errors, quantization)
- [ ] Production stability monitoring (NaN detection, automatic fallback)

**Step 4: Write Knowledge File**
- [ ] Create cuda/14-mixed-precision-debugging-expert.md
- [ ] Section 1: NaN Detection & Debugging (~125 lines)
      Cite: PyTorch AMP docs, debugging guides
      Include: Detection hooks, isolation workflow, common causes
- [ ] Section 2: Gradient Scaling Issues (~125 lines)
      Cite: GradScaler documentation, stability guides
      Include: Dynamic scaling tuning, underflow/overflow patterns
- [ ] Section 3: Precision Format Stability (~125 lines)
      Cite: FP16/BF16/FP8 comparison papers
      Include: Numerical stability, format selection, migration
- [ ] Section 4: Production Stability (~125 lines)
      Cite: Production ML guides
      Include: Monitoring, automatic fallback, stability guarantees

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-mixed-precision-debugging-2025-11-13.md
- [ ] Include: Runner (PART 3), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Ultra-expert mixed precision debugging knowledge

---

## PART 4: Production Deployment Troubleshooting (Containers, Drivers, Multi-Tenant)

- [✓] PART 4: Create cuda/15-production-deployment-troubleshooting-expert.md (~500 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "production" AND "deployment" in cuda/ folder
- [ ] Read existing cuda/ files for production content
- [ ] Identify gaps: Container issues, driver mismatches, multi-tenant problems

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA Docker container driver mismatch troubleshooting 2024"
- [ ] Search: "NVIDIA Container Toolkit debugging GPU access denied"
- [ ] Search: "Multi-tenant GPU isolation CUDA MPS MIG troubleshooting"
- [ ] Search: "Kubernetes GPU scheduling CUDA errors production"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Container troubleshooting (NVIDIA Container Toolkit, runtime errors)
- [ ] Driver/container version mismatches (compatibility matrix, forward compat)
- [ ] GPU device access issues (permissions, cgroups, device nodes)
- [ ] Multi-tenant isolation (MPS, MIG, time-slicing, debugging conflicts)
- [ ] Kubernetes GPU scheduling (device plugin errors, resource allocation)
- [ ] Production monitoring (health checks, GPU utilization, error rates)
- [ ] Emergency recovery (GPU reset, driver reload, container restart)

**Step 3: Content to Extract**
- [ ] NVIDIA Container Toolkit troubleshooting (nvidia-docker2, libnvidia-container)
- [ ] Driver compatibility matrix (container runtime vs host driver)
- [ ] GPU device access debugging (/dev/nvidia*, permissions, SELinux)
- [ ] MPS debugging (multi-process service errors, isolation issues)
- [ ] MIG configuration (instance profiles, partition failures)
- [ ] Kubernetes GPU device plugin (daemonset errors, resource claims)
- [ ] Production runbooks (common failures, recovery procedures)

**Step 4: Write Knowledge File**
- [ ] Create cuda/15-production-deployment-troubleshooting-expert.md
- [ ] Section 1: Container & Driver Issues (~125 lines)
      Cite: NVIDIA Container Toolkit docs, Docker debugging
      Include: Runtime errors, driver mismatches, toolkit debugging
- [ ] Section 2: GPU Device Access (~125 lines)
      Cite: Linux device management docs
      Include: Permissions, cgroups, device nodes, SELinux
- [ ] Section 3: Multi-Tenant Isolation (~125 lines)
      Cite: MPS/MIG documentation, multi-tenant guides
      Include: MPS debugging, MIG configuration, conflict resolution
- [ ] Section 4: Production Operations (~125 lines)
      Cite: Production operations guides
      Include: Kubernetes GPU scheduling, monitoring, recovery runbooks

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-production-deployment-2025-11-13.md
- [ ] Include: Runner (PART 4), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Ultra-expert production deployment knowledge

---

## Completion Criteria

**All PARTs must:**
- ✅ Check existing knowledge first (avoid duplication with files 08-11)
- ✅ Focus on ULTRA-EXPERT depth (beyond previous expansion)
- ✅ Include real production scenarios and solutions
- ✅ Cite web sources (accessed 2025-11-13)
- ✅ Provide detailed workflows and commands
- ✅ Cover advanced edge cases and hard problems
- ✅ Create KNOWLEDGE DROP file

**Expected output:**
- 4 new files in cuda/ (~500 lines each, ~2,000 lines total)
- Ultra-expert troubleshooting knowledge
- Advanced debugging techniques
- Production deployment wisdom
- Deep technical expertise

---

**Status**: Ready for parallel execution (4 runners)
**Depth**: ULTRA-EXPERT (going beyond expansion #10)
