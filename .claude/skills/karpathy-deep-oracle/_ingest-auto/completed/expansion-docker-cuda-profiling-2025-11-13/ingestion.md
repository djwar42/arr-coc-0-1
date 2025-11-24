# Oracle Knowledge Expansion: Docker + CUDA Profiling Expert Knowledge

**Topic**: Docker Multi-Stage Builds, PyTorch Profiling, CMake Build System, NVIDIA Container Packaging
**Date**: 2025-11-13
**Source**: arr-coc-0-1/training/CUPTI_INVESTIGATION_STUDY.md production debugging experience
**Target Folder**: `cuda/` and `karpathy/practical-implementation/`

---

## Overview

This expansion adds expert-level knowledge on 4 critical topics discovered during CUPTI investigation:
1. Docker multi-stage build optimization for ML/CUDA workloads
2. PyTorch profiling tools ecosystem (CUPTI, Kineto, NSight alternatives)
3. PyTorch CMake build system internals (optional dependency detection)
4. NVIDIA container images and CUDA apt packaging structure

All topics focus on **production debugging and optimization** from real-world experience.

---

## PART 1: Create cuda/10-docker-multistage-ml-optimization.md (400 lines)

- [✓] PART 1: Create cuda/10-docker-multistage-ml-optimization.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md to find Docker/container-related files
- [ ] Grep for "docker" AND "multi-stage" in cuda/ and practical-implementation/
- [ ] Identify knowledge gaps: What's NOT covered about Docker ML optimization?

**Step 1: Web Research**
- [ ] Search: "docker multi-stage builds cuda pytorch production optimization 2024"
- [ ] Search: "nvidia cuda runtime vs devel image differences size comparison"
- [ ] Search: "docker buildkit cache mount pytorch cuda compilation 2025"
- [ ] Search: "minimal docker image ml production cuda libraries only"
- [ ] Scrape top 3-4 results per search for detailed content

**Step 2: Extract Key Content**
Research Focus:
- Multi-stage build patterns (builder vs runtime, devel vs slim)
- Layer caching strategies for CUDA/PyTorch builds
- Image size optimization (1.5GB runtime vs 4.5GB devel vs 8GB full)
- What to copy between stages (libraries, headers, profiling tools dilemma)
- BuildKit advanced features (cache mounts, secrets)
- ARG cache invalidation strategies
- Production deployment patterns (minimal runtime, optional dev tools)

**Step 3: Write Knowledge File**
- [ ] Create cuda/10-docker-multistage-ml-optimization.md (~400 lines)
- [ ] Section 1: Multi-Stage Build Fundamentals (~100 lines)
      - Builder vs runtime stages
      - NVIDIA base image variants (runtime, devel, cudnn)
      - Size comparisons and trade-offs
      Cite web sources
- [ ] Section 2: Layer Caching & BuildKit (~100 lines)
      - Cache mount strategies for pip/apt
      - RUN --mount=type=cache patterns
      - ARG placement for cache optimization
      Cite web sources
- [ ] Section 3: Library Selection Between Stages (~100 lines)
      - Essential CUDA libraries (runtime)
      - Optional profiling tools (CUPTI, NSight)
      - Development headers (when needed)
      Cite CUPTI investigation + web sources
- [ ] Section 4: Production Patterns (~100 lines)
      - Minimal runtime images
      - Optional dev layer mounting
      - CI/CD optimization strategies
      Cite web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-docker-multistage-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List web sources used
- [ ] Describe production debugging context from CUPTI investigation

---

## PART 2: Create cuda/11-pytorch-profiling-ecosystem.md (400 lines)

- [✓] PART 2: Create cuda/11-pytorch-profiling-ecosystem.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for profiling-related files
- [ ] Grep for "profiling" OR "kineto" OR "nsight" in cuda/ and llm-gpu-integration/
- [ ] Check existing flashattention-internals.md for profiling references
- [ ] Identify gaps: PyTorch profiler internals, CUPTI alternatives

**Step 1: Web Research**
- [ ] Search: "pytorch profiler kineto cupti requirements alternatives 2024"
- [ ] Search: "nsight systems vs nsight compute vs torch.profiler comparison"
- [ ] Search: "gpu profiling without cupti pytorch production 2025"
- [ ] Search: "pytorch kineto profiling overhead minimal instrumentation"
- [ ] Scrape top 3-4 results per search

**Step 2: Extract Key Content**
Research Focus:
- PyTorch profiler (torch.profiler) - requires CUPTI
- Kineto profiling library (PyTorch's profiling backend)
- NSight Systems (system-level, no CUPTI needed)
- NSight Compute (kernel-level, standalone)
- nvprof (legacy but CUPTI-based)
- Alternatives when CUPTI is missing
- GPU kernel profiling vs CPU profiling
- TensorBoard integration
- Production profiling strategies (minimal overhead)

**Step 3: Write Knowledge File**
- [ ] Create cuda/11-pytorch-profiling-ecosystem.md (~400 lines)
- [ ] Section 1: PyTorch Profiler & Kineto (~120 lines)
      - torch.profiler API
      - Kineto backend architecture
      - CUPTI dependency (why it's needed)
      - What breaks without CUPTI
      Cite PyTorch docs + web sources
- [ ] Section 2: NSight Tools (~120 lines)
      - NSight Systems (timeline profiling)
      - NSight Compute (kernel analysis)
      - No CUPTI required (standalone)
      - When to use which tool
      Cite NVIDIA docs + web sources
- [ ] Section 3: Profiling Without CUPTI (~80 lines)
      - Fallback profilers
      - CPU-only profiling
      - Manual instrumentation
      Cite web sources
- [ ] Section 4: Production Profiling (~80 lines)
      - Minimal overhead strategies
      - Docker image considerations
      - TensorBoard integration
      Cite CUPTI investigation + web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-pytorch-profiling-2025-11-13.md
- [ ] Include: Runner (PART 2), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List web sources used
- [ ] Note CUPTI investigation context

---

## PART 3: Create cuda/12-pytorch-cmake-build-internals.md (400 lines)

- [✓] PART 3: Create cuda/12-pytorch-cmake-build-internals.md (Completed 2025-11-13 15:45)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for build system files
- [ ] Read cuda/02-pytorch-build-system-compilation.md (existing)
- [ ] Identify gaps: CMake detection logic, optional dependencies, false positives

**Step 1: Web Research**
- [ ] Search: "pytorch cmake build system cupti detection header only 2024"
- [ ] Search: "pytorch cmake optional dependencies disable cupti kineto"
- [ ] Search: "FindCUDA.cmake pytorch how it detects libraries headers"
- [ ] Search: "pytorch build from source cmake configuration expert 2025"
- [ ] Scrape top 3-4 results per search

**Step 2: Extract Key Content**
Research Focus:
- PyTorch CMake configuration (FindCUDA.cmake, TORCH_CUDA_ARCH_LIST)
- How PyTorch detects CUPTI (headers vs libraries)
- Why CMake "finds" CUPTI but library doesn't exist (header-only detection)
- Optional dependencies in PyTorch builds
- CMakeCache.txt interpretation
- Build-time vs runtime dependencies
- CMake lies vs reality (claims to find but doesn't validate)
- Force disabling optional features

**Step 3: Write Knowledge File**
- [ ] Create cuda/12-pytorch-cmake-build-internals.md (~400 lines)
- [ ] Section 1: CMake Detection Logic (~120 lines)
      - FindCUDA.cmake internals
      - Header-only vs library detection
      - Why CMake "finds" missing libraries
      Cite PyTorch source + web sources
- [ ] Section 2: Optional Dependencies (~120 lines)
      - CUPTI, Kineto, MKL, MAGMA
      - How to force disable
      - CMakeCache.txt variables
      Cite CUPTI investigation + web sources
- [ ] Section 3: Build vs Runtime Dependencies (~80 lines)
      - What's needed at build time
      - What's needed at runtime
      - Docker stage implications
      Cite web sources
- [ ] Section 4: Debugging CMake Issues (~80 lines)
      - CMakeCache.txt interpretation
      - -DCMAKE_FIND_DEBUG_MODE=ON
      - Common false positive patterns
      Cite CUPTI investigation + web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-pytorch-cmake-2025-11-13.md
- [ ] Include: Runner (PART 3), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List web sources used
- [ ] Describe CMake false positive discovery from CUPTI investigation

---

## PART 4: Create cuda/13-nvidia-container-cuda-packaging.md (400 lines)

- [✓] PART 4: Create cuda/13-nvidia-container-cuda-packaging.md (Completed 2025-11-13 16:45)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for NVIDIA/CUDA packaging files
- [ ] Grep for "nvidia" AND ("container" OR "apt" OR "package") in cuda/
- [ ] Identify gaps: Container image structure, apt package ecosystem

**Step 1: Web Research**
- [ ] Search: "nvidia cuda container images runtime vs devel differences 2024"
- [ ] Search: "cuda-libraries-dev vs cuda-libraries apt packages cupti 2025"
- [ ] Search: "nvidia docker image official structure what packages included"
- [ ] Search: "cuda apt repository setup ubuntu 22.04 correct package versions"
- [ ] Scrape top 3-4 results per search

**Step 2: Extract Key Content**
Research Focus:
- NVIDIA official container image structure (runtime, devel, cudnn variants)
- CUDA apt packages (cuda-libraries-12-4 vs cuda-libraries-dev-12-4)
- Where CUPTI lives (cuda-libraries-dev, /usr/local/cuda/extras/CUPTI)
- CUDA toolkit local installer vs apt packages
- NVIDIA apt repositories setup
- Version matching (driver vs runtime vs toolkit)
- CUDA forward compatibility packages
- What's in each image variant (runtime vs devel vs cudnn)

**Step 3: Write Knowledge File**
- [ ] Create cuda/13-nvidia-container-cuda-packaging.md (~400 lines)
- [ ] Section 1: NVIDIA Container Image Variants (~120 lines)
      - nvidia/cuda:{version}-runtime-ubuntu22.04
      - nvidia/cuda:{version}-devel-ubuntu22.04
      - nvidia/cuda:{version}-cudnn8-runtime-ubuntu22.04
      - Size comparisons, what's included
      Cite NVIDIA docs + web sources
- [ ] Section 2: CUDA Apt Package Ecosystem (~120 lines)
      - cuda-libraries-{version} (runtime)
      - cuda-libraries-dev-{version} (headers + CUPTI)
      - cuda-toolkit-{version} (full toolkit)
      - Package dependencies
      Cite CUPTI investigation + web sources
- [ ] Section 3: CUPTI Location & Installation (~80 lines)
      - /usr/local/cuda/extras/CUPTI/
      - cuda-libraries-dev package contents
      - How to add CUPTI to runtime image
      Cite CUPTI investigation + web sources
- [ ] Section 4: NVIDIA Apt Repository Setup (~80 lines)
      - Adding NVIDIA repositories
      - Version pinning strategies
      - Driver vs runtime compatibility
      Cite web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-nvidia-packaging-2025-11-13.md
- [ ] Include: Runner (PART 4), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List web sources used
- [ ] Note CUPTI discovery (cuda-libraries-dev-12-4 package)

---

## Completion Criteria

All 4 PARTs must:
- [ ] Create knowledge file in cuda/ folder
- [ ] Include web research citations
- [ ] Reference CUPTI investigation context
- [ ] Create individual KNOWLEDGE DROP file
- [ ] Mark checkbox [✓] when complete

**Expected Output:**
- 4 knowledge files (~1,600 lines total)
- 4 KNOWLEDGE DROP files
- Expert-level Docker + CUDA + PyTorch production knowledge
- Direct connection to arr-coc-0-1 CUPTI investigation debugging

---

**Oracle**: Launch all 4 runners in PARALLEL, retry failures once, finalize INDEX.md and git commit.
