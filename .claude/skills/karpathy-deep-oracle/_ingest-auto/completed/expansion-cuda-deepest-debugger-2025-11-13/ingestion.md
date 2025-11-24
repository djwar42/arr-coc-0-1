# Oracle Knowledge Expansion: CUDA DEEPEST DEBUGGER (Absolute Expert)

**Topic**: PTX/SASS assembly, hardware-level debugging, massive-scale distributed, custom kernel optimization
**Date**: 2025-11-13
**Runners**: 4 (parallel execution)
**Target**: ABSOLUTE DEEPEST debugging expertise (assembly, hardware, scale)

---

## PART 1: PTX/SASS Assembly-Level Debugging & Cubin Analysis

- [✓] PART 1: Create cuda/16-assembly-level-debugging-ptx-sass-expert.md (~600 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "PTX" AND "SASS" in cuda/ folder
- [ ] Read cuda/05-tensor-core-programming-wmma-mma.md (mentions PTX)
- [ ] Read cuda/12-kernel-debugging-internals-expert.md (NSight Source View)
- [ ] Identify gaps: Deep PTX/SASS analysis, cubin inspection, assembly optimization

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA PTX SASS assembly debugging cuobjdump 2024"
- [ ] Search: "NSight Compute SASS analysis instruction throughput"
- [ ] Search: "cubin analysis ptxas register usage optimization"
- [ ] Search: "CUDA assembly-level performance debugging"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] PTX/SASS fundamentals (ISA differences, when to use each)
- [ ] cuobjdump analysis (extracting PTX/SASS, symbol tables, relocation info)
- [ ] NSight Compute Source View deep dive (PTX ↔ SASS ↔ C++ correlation)
- [ ] Register pressure analysis (ptxas output, occupancy impact)
- [ ] Instruction throughput analysis (dual-issue, warp scheduler, latency hiding)
- [ ] SASS optimization patterns (load/store alignment, predication, reuse flags)
- [ ] Inline assembly debugging (PTX injection, constraints, volatile)

**Step 3: Content to Extract**
- [ ] PTX ISA reference (most common instructions, addressing modes)
- [ ] SASS instruction formats (Volta/Ampere/Hopper differences)
- [ ] cuobjdump commands (--dump-ptx, --dump-sass, --list-elf)
- [ ] Register usage analysis (ptxas -v output interpretation)
- [ ] Occupancy calculator integration (registers vs shared memory trade-offs)
- [ ] NSight Compute metrics (inst_executed, warp_execution_efficiency)
- [ ] Assembly optimization checklist (coalescing, bank conflicts, warp divergence at SASS level)

**Step 4: Write Knowledge File**
- [ ] Create cuda/16-assembly-level-debugging-ptx-sass-expert.md
- [ ] Section 1: PTX/SASS Fundamentals & Tools (~150 lines)
      Cite: NVIDIA PTX ISA, cuobjdump docs
      Include: cuobjdump commands, PTX vs SASS, ISA versions
- [ ] Section 2: Register & Occupancy Analysis (~150 lines)
      Cite: ptxas documentation, occupancy calculator
      Include: Register pressure diagnosis, spilling detection
- [ ] Section 3: Instruction-Level Optimization (~150 lines)
      Cite: NSight Compute docs, SASS guides
      Include: Throughput analysis, dual-issue, latency hiding
- [ ] Section 4: Advanced Assembly Debugging (~150 lines)
      Cite: Inline PTX guides, assembly optimization
      Include: Inline assembly, SASS patterns, production debugging

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-assembly-debugging-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List: Knowledge file created, sources used, gaps filled
- [ ] Describe: Absolute deepest assembly-level debugging

---

## PART 2: Hardware-Level Debugging (ECC, Memory Controller, GPU Faults)

- [✓] PART 2: Create cuda/17-hardware-level-debugging-gpu-faults-expert.md (~620 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "ECC" AND "hardware" in cuda/ folder
- [ ] Read cuda/11-advanced-troubleshooting-multi-gpu-expert.md (mentions ECC errors)
- [ ] Read cuda/15-production-deployment-troubleshooting-expert.md (driver crashes)
- [ ] Identify gaps: Deep ECC analysis, memory controller debugging, GPU hardware diagnostics

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA ECC error analysis debugging memory controller 2024"
- [ ] Search: "GPU hardware faults Xid errors NVIDIA debugging"
- [ ] Search: "NVML API GPU health monitoring temperature throttling"
- [ ] Search: "GPU memory controller errors row remapping RAS"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] ECC error types (correctable vs uncorrectable, SRAM vs DRAM)
- [ ] Memory controller debugging (row remapping, page retirement, RAS features)
- [ ] GPU hardware monitoring (NVML API, nvidia-smi queries, health checks)
- [ ] Xid error codes (comprehensive list, meaning, recovery procedures)
- [ ] Thermal management debugging (throttling detection, clock speed monitoring)
- [ ] Power management issues (power limit throttling, PCIe power delivery)
- [ ] Hardware fault prediction (predictive analytics, SMART-like monitoring)

**Step 3: Content to Extract**
- [ ] ECC error monitoring (nvidia-smi queries, NVML API calls)
- [ ] Memory error patterns (single-bit vs double-bit, frequency analysis)
- [ ] Row remapping mechanics (when GPU retires rows, impact on capacity)
- [ ] Xid error code reference (31, 43, 45, 48, 61, 62, 63, 64, 79, etc.)
- [ ] Thermal throttling thresholds (slowdown temp, shutdown temp, monitoring)
- [ ] Power debugging (nvidia-smi -q, power limit configuration, PCIe slot limits)
- [ ] Production hardware health checks (automated monitoring, RMA criteria)

**Step 4: Write Knowledge File**
- [ ] Create cuda/17-hardware-level-debugging-gpu-faults-expert.md
- [ ] Section 1: ECC Error Analysis (~150 lines)
      Cite: NVIDIA ECC docs, NVML API reference
      Include: Error types, monitoring, patterns, impact
- [ ] Section 2: Memory Controller & RAS (~150 lines)
      Cite: GPU architecture docs, RAS features
      Include: Row remapping, page retirement, diagnostics
- [ ] Section 3: Xid Errors & GPU Faults (~150 lines)
      Cite: Xid error documentation, recovery guides
      Include: Error code reference, recovery procedures
- [ ] Section 4: Thermal & Power Debugging (~150 lines)
      Cite: Thermal management docs, power specs
      Include: Throttling detection, power monitoring, production health

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-hardware-debugging-2025-11-13.md
- [ ] Include: Runner (PART 2), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Hardware-level GPU fault debugging

---

## PART 3: Massive-Scale Distributed Debugging (100s-1000s of GPUs)

- [✓] PART 3: Create cuda/18-massive-scale-distributed-debugging-expert.md (~620 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "distributed" AND "scale" in cuda/ folder
- [ ] Read cuda/11-advanced-troubleshooting-multi-gpu-expert.md (DDP, NCCL)
- [ ] Read vertex-ai-production/00-distributed-training-patterns.md (if exists)
- [ ] Identify gaps: Massive-scale debugging (100s-1000s GPUs), network debugging, stragglers

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "distributed training 1000 GPU debugging PyTorch 2024"
- [ ] Search: "NCCL InfiniBand debugging network topology bottlenecks"
- [ ] Search: "straggler detection distributed training debugging"
- [ ] Search: "multi-node GPU cluster debugging synchronization"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Network topology debugging (InfiniBand, RoCE, switch fabric, tree topology)
- [ ] NCCL topology analysis (nvidia-smi topo, NCCL_TOPO_DUMP_FILE)
- [ ] Straggler detection (slow nodes, network bottlenecks, imbalanced workloads)
- [ ] Collective operation debugging (all-reduce, all-gather at massive scale)
- [ ] Fault tolerance at scale (elastic training, checkpoint-restart strategies)
- [ ] Network bandwidth analysis (iperf, NCCL tests, SHARP acceleration)
- [ ] Production monitoring (distributed tracing, centralized logging, anomaly detection)

**Step 3: Content to Extract**
- [ ] Network topology best practices (fat-tree, rail-optimized, NCCL tuning)
- [ ] NCCL environment variables for scale (NCCL_TOPO_FILE, NCCL_NET_GDR_LEVEL)
- [ ] Straggler detection techniques (timing analysis, outlier detection)
- [ ] Collective benchmarking (NCCL tests suite, bandwidth analysis)
- [ ] Elastic training patterns (torchelastic, dynamic scaling)
- [ ] Network debugging tools (nvidia-smi topo, ibstat, perftest)
- [ ] Production observability (distributed tracing, Prometheus federation)

**Step 4: Write Knowledge File**
- [ ] Create cuda/18-massive-scale-distributed-debugging-expert.md
- [ ] Section 1: Network Topology & NCCL (~150 lines)
      Cite: NCCL docs, InfiniBand guides
      Include: Topology analysis, NCCL tuning, fabric debugging
- [ ] Section 2: Straggler Detection & Performance (~150 lines)
      Cite: Distributed training optimization papers
      Include: Detection techniques, profiling, load balancing
- [ ] Section 3: Fault Tolerance at Scale (~150 lines)
      Cite: Elastic training docs, checkpoint strategies
      Include: Recovery mechanisms, dynamic scaling
- [ ] Section 4: Production Observability (~150 lines)
      Cite: Observability guides, monitoring stacks
      Include: Distributed tracing, centralized logging, anomaly detection

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-massive-scale-debugging-2025-11-13.md
- [ ] Include: Runner (PART 3), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Massive-scale distributed debugging expertise

---

## PART 4: Custom Kernel Optimization Debugging (Occupancy, Bank Conflicts, Warp Efficiency)

- [✓] PART 4: Create cuda/19-custom-kernel-optimization-debugging-expert.md (~600 lines) (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md cuda/ section
- [ ] Grep for "occupancy" AND "bank conflicts" in cuda/ folder
- [ ] Read cuda/10-performance-debugging-profiling-expert.md (basic optimization)
- [ ] Read cuda/04-pytorch-custom-cuda-extensions.md (kernel writing)
- [ ] Identify gaps: Deep occupancy analysis, shared memory optimization, warp scheduler

**Step 1: Web Research (Bright Data MCP)**
- [ ] Search: "CUDA occupancy optimization register pressure 2024"
- [ ] Search: "shared memory bank conflict debugging NSight Compute"
- [ ] Search: "warp execution efficiency stall reasons analysis"
- [ ] Search: "CUDA kernel optimization checklist expert"
- [ ] Scrape top 3-4 results per query

**Step 2: Research Focus Areas**
- [ ] Occupancy analysis deep dive (register vs shared memory limits, block size tuning)
- [ ] Shared memory bank conflicts (detection, 32-bank architecture, padding strategies)
- [ ] Warp execution efficiency (stall reasons, pipeline utilization, instruction mix)
- [ ] Memory access pattern optimization (coalescing verification, sector utilization)
- [ ] Warp divergence analysis (predication, branch optimization)
- [ ] Latency hiding techniques (ILP vs TLP, occupancy vs resource usage)
- [ ] Custom kernel debugging workflow (NSight Compute metrics interpretation)

**Step 3: Content to Extract**
- [ ] Occupancy calculator usage (CUDA_Occupancy_Calculator.xls, programmatic API)
- [ ] NSight Compute occupancy metrics (achieved vs theoretical, limiters)
- [ ] Bank conflict detection (shared_load_transactions_per_request, ideal=1.0)
- [ ] Warp stall reasons (memory throttle, execution dependency, barrier, dispatch stall)
- [ ] Memory coalescing metrics (l1tex__t_sectors ratio, ideal=1.0 for 128B)
- [ ] ILP analysis (inst_per_warp distribution, pipeline utilization)
- [ ] Optimization decision trees (when to increase occupancy vs reduce divergence)

**Step 4: Write Knowledge File**
- [ ] Create cuda/19-custom-kernel-optimization-debugging-expert.md
- [ ] Section 1: Occupancy Analysis & Tuning (~150 lines)
      Cite: CUDA Best Practices, occupancy docs
      Include: Calculator usage, limiters, block size optimization
- [ ] Section 2: Shared Memory Optimization (~150 lines)
      Cite: Shared memory guides, bank conflict docs
      Include: Bank conflict detection, padding strategies, SOA vs AOS
- [ ] Section 3: Warp Efficiency Analysis (~150 lines)
      Cite: NSight Compute guides, warp scheduler docs
      Include: Stall reasons, divergence, predication
- [ ] Section 4: Memory Access Optimization (~150 lines)
      Cite: Memory optimization guides
      Include: Coalescing verification, sector utilization, caching

**Step 5: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-kernel-optimization-debugging-2025-11-13.md
- [ ] Include: Runner (PART 4), Timestamp, Status
- [ ] List: Knowledge file, sources, gaps filled
- [ ] Describe: Custom kernel optimization debugging expertise

---

## Completion Criteria

**All PARTs must:**
- ✅ Check existing knowledge first (avoid duplication with files 08-15)
- ✅ Focus on ABSOLUTE DEEPEST expertise (assembly, hardware, massive scale, kernel internals)
- ✅ Include real production scenarios from hyperscale training
- ✅ Cite web sources (accessed 2025-11-13)
- ✅ Provide detailed expert-level workflows
- ✅ Cover the hardest, rarest debugging scenarios
- ✅ Create KNOWLEDGE DROP file

**Expected output:**
- 4 new files in cuda/ (~600 lines each, ~2,400 lines total)
- ABSOLUTE DEEPEST debugging knowledge
- Assembly-level expertise
- Hardware fault debugging
- Massive-scale distributed debugging
- Custom kernel optimization mastery

---

**Status**: Ready for parallel execution (4 runners)
**Depth**: ABSOLUTE DEEPEST DEBUGGER (beyond expansion #10 and #11)
**Target**: The hardest CUDA debugging scenarios in existence
