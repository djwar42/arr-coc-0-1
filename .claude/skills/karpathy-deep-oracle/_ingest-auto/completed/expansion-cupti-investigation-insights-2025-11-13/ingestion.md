# Oracle Knowledge Expansion: CUPTI Investigation Insights

**Topic**: Deep dive into CUPTI, Docker build strategies, PyTorch CMake detection, CUDA installation methods
**Date**: 2025-11-13
**Source**: CUPTI investigation learnings, NVIDIA docs, PyTorch internals
**Target Folder**: `cuda/` and `karpathy/practical-implementation/`

---

## Overview

This expansion extracts 4 general knowledge areas from the arr-coc-0-1 CUPTI investigation experience.

All runners should focus on PRODUCTION-VALIDATED knowledge from real debugging experience.

---

## PART 1: Create cuda/20-cupti-deep-dive-arr-coc.md (400 lines)

- [✓] PART 1: Create cuda/20-cupti-deep-dive-arr-coc.md (Completed 2025-11-13 21:30)

**Focus**: What is CUPTI? Do we need it for arr-coc? When to skip it?

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/11-pytorch-profiling-ecosystem.md (CUPTI profiling)
- [ ] Read cuda/13-nvidia-container-cuda-packaging.md (CUPTI location)
- [ ] Identify gaps: CUPTI purpose, arr-coc relevance, skip vs use decision

**Step 1: CUPTI Research**
- [ ] Search: "nvidia cupti what is it used for profiling"
- [ ] Search: "cupti vs nsight when to use cuda profiling"
- [ ] Search: "do i need cupti pytorch production deployment"
- [ ] Search: "cupti overhead performance impact"
- [ ] Scrape NVIDIA CUPTI official docs

**Step 2: Extract CUPTI Knowledge**
Research Focus:
- What is CUPTI (CUDA Profiling Tools Interface)
- CUPTI use cases (PyTorch profiler, Kineto, custom profilers)
- CUPTI overhead (runtime cost, memory impact)
- When you NEED CUPTI vs when you DON'T
- **arr-coc specific**: Do we need CUPTI for arr-coc training/inference?
- Production decision tree: CUPTI yes/no

**Step 3: Write Knowledge File**
- [ ] Create cuda/20-cupti-deep-dive-arr-coc.md (~400 lines)
- [ ] Section 1: What is CUPTI (~100 lines)
      - CUPTI architecture (callbacks, activity API, metrics API)
      - CUPTI vs NSight (when to use each)
      - CUPTI dependencies (PyTorch profiler, Kineto)
      Cite: NVIDIA CUPTI docs
- [ ] Section 2: CUPTI Use Cases (~100 lines)
      - Profiling workflows that require CUPTI
      - Profiling workflows WITHOUT CUPTI
      - Overhead measurements
      Cite: PyTorch profiler docs, NSight docs
- [ ] Section 3: Production Decision Tree (~100 lines)
      - When you NEED CUPTI (development, deep profiling)
      - When you DON'T NEED CUPTI (production inference, basic monitoring)
      - Docker image size trade-offs
      Cite: arr-coc-0-1 investigation
- [ ] Section 4: arr-coc Analysis (~100 lines)
      - **Do we need CUPTI for arr-coc training?**
      - **Do we need CUPTI for arr-coc inference?**
      - Recommendation for arr-coc deployment
      Cite: arr-coc architecture, profiling needs

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-cupti-deep-dive-2025-11-13.md
- [ ] Include clear answer: "Do we need CUPTI for arr-coc?"

---

## PARTS 2-4: TBD (User to specify 3 more topics)

---

## DISABLED PARTS (keeping for reference)

## PART 2: Create cuda/21-docker-build-strategy-patterns.md (400 lines)

- [ ] PART 2: Create cuda/21-docker-build-strategy-patterns.md

**Focus**: Docker build strategies from CUPTI investigation (debug builder, devel as runtime, nuclear copy)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/10-docker-multistage-ml-optimization.md (multi-stage builds)
- [ ] Identify gaps: Build debugging strategies, alternative approaches

**Step 1: Docker Build Strategy Research**
- [ ] Search: "docker build debug intermediate stage troubleshooting"
- [ ] Search: "docker devel as runtime vs multi-stage production"
- [ ] Search: "docker copy entire cuda toolkit nuclear approach"
- [ ] Scrape Docker debugging guides

**Step 2: Extract Build Strategies**
Research Focus:
- Debug builder stage techniques (--target=builder, intermediate inspection)
- Devel as runtime (pros/cons, when it makes sense)
- Nuclear copy approach (copy entire /usr/local/cuda)
- Build troubleshooting workflows
- Trade-offs analysis (size vs simplicity vs debuggability)

**Step 3: Write Knowledge File**
- [ ] Create cuda/21-docker-build-strategy-patterns.md (~400 lines)
- [ ] Section 1: Debug Builder Stage (~120 lines)
      - docker build --target=builder
      - Inspecting intermediate layers
      - Interactive debugging (docker run)
      Cite: Docker docs, arr-coc-0-1 investigation
- [ ] Section 2: Devel as Runtime (~100 lines)
      - When to use devel image as runtime
      - Size trade-offs (4.5GB vs 1.5GB)
      - Development vs production
      Cite: NVIDIA container docs
- [ ] Section 3: Nuclear Copy CUDA (~100 lines)
      - Copy entire /usr/local/cuda approach
      - Pros: Guaranteed completeness
      - Cons: Massive size increase
      - When to use nuclear approach
      Cite: arr-coc-0-1 CUPTI investigation
- [ ] Section 4: Strategy Decision Matrix (~80 lines)
      - Build strategy comparison table
      - Debugging workflow recommendations
      - Production deployment best practices
      Cite: Docker best practices

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-docker-strategies-2025-11-13.md

---

## PART 3: Create cuda/22-pytorch-cmake-detection-deep-dive.md (400 lines)

- [ ] PART 3: Create cuda/22-pytorch-cmake-detection-deep-dive.md

**Focus**: PyTorch CMake detection internals (how PyTorch finds CUDA/CUPTI)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/12-pytorch-cmake-build-internals.md (CMake detection)
- [ ] Identify gaps: Deep CMake findings, detection output analysis

**Step 1: PyTorch CMake Research**
- [ ] Search: "pytorch cmake detect cuda cupti verbose output"
- [ ] Search: "cmake find_library how it works detection"
- [ ] Search: "pytorch build log cmake findings cuda"
- [ ] Use mcp__bright-data__web_data_github_repository_file for:
      - PyTorch cmake/Dependencies.cmake source code

**Step 2: Extract CMake Detection Internals**
Research Focus:
- PyTorch CMake detection sequence (CUDA → CUPTI → other)
- CMake find_library() detection logic
- How to interpret CMake findings output
- CMake cache analysis (CMakeCache.txt)
- Detection vs availability gap (found != usable)

**Step 3: Write Knowledge File**
- [ ] Create cuda/22-pytorch-cmake-detection-deep-dive.md (~400 lines)
- [ ] Section 1: PyTorch CMake Detection Sequence (~120 lines)
      - Detection order (CUDA, cuDNN, NCCL, CUPTI)
      - cmake/Dependencies.cmake analysis
      - find_library() patterns
      Cite: PyTorch GitHub source code
- [ ] Section 2: Interpreting CMake Findings (~120 lines)
      - "Found CUDA" vs "CUDA not found"
      - CUPTI detection output
      - CMakeCache.txt reading
      Cite: CMake documentation, arr-coc-0-1 logs
- [ ] Section 3: Detection vs Availability Gap (~80 lines)
      - Headers found, libraries missing
      - Build-time vs runtime validation
      - False positives analysis
      Cite: arr-coc-0-1 CUPTI investigation
- [ ] Section 4: Debugging CMake Detection (~80 lines)
      - CMAKE_FIND_DEBUG_MODE=ON
      - Verbose output interpretation
      - Forcing detection results
      Cite: CMake docs, PyTorch forums

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-pytorch-cmake-2025-11-13.md

---

## PART 4: Create cuda/23-cuda-installation-methods-comparison.md (400 lines)

- [ ] PART 4: Create cuda/23-cuda-installation-methods-comparison.md

**Focus**: CUDA installation methods (local installer vs apt-get vs runfile)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/18-ubuntu-pytorch-cuda-setup.md (Ubuntu apt setup)
- [ ] Identify gaps: Installation method comparison, when to use each

**Step 1: CUDA Installation Methods Research**
- [ ] Search: "cuda local installer vs network installer vs apt comparison"
- [ ] Search: "cuda runfile installation pros cons"
- [ ] Search: "nvidia apt repository vs local cuda installer"
- [ ] Scrape NVIDIA CUDA installation guide

**Step 2: Extract Installation Methods**
Research Focus:
- Local installer (runfile, .run file)
- Network installer (.deb with apt repository)
- apt-get direct (ubuntu apt packages)
- Comparison matrix (size, control, reproducibility)
- Docker context (which method for containers)

**Step 3: Write Knowledge File**
- [ ] Create cuda/23-cuda-installation-methods-comparison.md (~400 lines)
- [ ] Section 1: Local Installer Method (~100 lines)
      - Runfile installation process
      - Pros: Full control, offline install
      - Cons: Manual updates, larger download
      Cite: NVIDIA CUDA installation guide
- [ ] Section 2: Network Installer + apt (~100 lines)
      - .deb repository setup
      - Pros: Automatic updates, selective packages
      - Cons: Network dependency
      Cite: NVIDIA apt guide, Ubuntu docs
- [ ] Section 3: Direct apt-get (~100 lines)
      - apt-get install cuda-toolkit-12-4
      - Pros: Simplest, system package manager
      - Cons: Ubuntu-specific, version delays
      Cite: Ubuntu CUDA packages
- [ ] Section 4: Method Comparison & Docker Context (~100 lines)
      - Comparison matrix (size, speed, reproducibility)
      - Docker-specific recommendations
      - arr-coc-0-1 choice analysis
      Cite: Docker best practices, arr-coc-0-1 investigation

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-cuda-installation-2025-11-13.md

---

## Completion Criteria

All 4 PARTs must:
- [ ] Create knowledge file in cuda/ folder
- [ ] Include production insights from arr-coc-0-1 investigation
- [ ] Create individual KNOWLEDGE DROP file
- [ ] Mark checkbox [✓] when complete

**Expected Output:**
- 4 knowledge files (~1,600 lines total)
- 4 KNOWLEDGE DROP files
- Production-validated CUPTI investigation insights
- Clear answer for arr-coc: "Do we need CUPTI?"

---

**Oracle**: Launch all 4 runners in PARALLEL, extract production insights, answer arr-coc CUPTI question, finalize INDEX.md and git commit.
