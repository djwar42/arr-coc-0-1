# Oracle Knowledge Expansion: Docker CUDA Build Strategies (Complete)

**Topic**: Complete Docker build strategies for PyTorch/CUDA from CUPTI investigation
**Date**: 2025-11-13
**Source**: CUPTI investigation learnings, Docker best practices, PyTorch build internals
**Target Folder**: `cuda/`

---

## Overview

From the CUPTI investigation, we identified 6 critical Docker build strategies for PyTorch/CUDA containers. Each strategy represents a different approach to solving the "CUPTI in production" problem.

This expansion creates 6 comprehensive knowledge files covering all strategies.

---

## PART 1: Create cuda/21-docker-debug-builder-stage.md (~400 lines)

**Topic**: Debug builder stage - Interactive debugging during multi-stage builds

- [✓] PART 1: Create cuda/21-docker-debug-builder-stage.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md - find Docker-related files
- [ ] Grep for "builder" AND "debug" in cuda/ folder
- [ ] Read cuda/10-docker-multistage-ml-optimization.md for context
- [ ] Identify gaps: Interactive debugging, troubleshooting build failures

**Step 1: Web Research**
- [ ] Search: "Docker multi-stage build debugging interactive"
- [ ] Search: "Docker builder stage troubleshooting"
- [ ] Search: "Docker build --target flag debugging"
- [ ] Find: Best practices for debugging intermediate stages

**Step 2: Create Knowledge File Structure**

Section 1: Builder Stage Debugging Fundamentals (~100 lines)
- What is a builder stage vs runtime stage
- Common build failures (missing libraries, CMake errors, compilation)
- Why debug builder stage (CUPTI investigation context)

Section 2: Interactive Debug Techniques (~150 lines)
- Docker build --target=builder (stop at intermediate stage)
- docker run -it builder-image /bin/bash (explore built environment)
- docker commit debugging containers (save state)
- BuildKit --progress=plain (verbose output)
- Inspect layer sizes: docker history

Section 3: CUPTI Investigation Debug Workflow (~100 lines)
- How we debugged PyTorch CMake false positives
- Finding missing libcupti.so in builder stage
- Verifying CUDA library presence
- Testing PyTorch profiler in builder vs runtime

Section 4: Production Debug Patterns (~50 lines)
- RUN ls -la /usr/local/cuda/extras/CUPTI (verification commands)
- RUN find / -name "libcupti.so*" (discovery)
- RUN python -c "import torch; print(torch.cuda.is_available())" (validation)
- Conditional debugging (ARG DEBUG=1 for extra commands)

**Step 3: Citations**
- Cite web sources (Docker docs, Stack Overflow)
- Reference cuda/10-docker-multistage-ml-optimization.md
- Link to cuda/12-pytorch-cmake-build-internals.md (CMake debugging)

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-docker-debug-builder-2025-11-13.md
- Summary: How to debug Docker multi-stage builds interactively
- Key insight: --target=builder + docker run -it = interactive troubleshooting

---

## PART 2: Create cuda/22-cuda-apt-get-installation.md (~450 lines)

**Topic**: Install CUDA via apt-get - Package-based installation strategies

- [✓] PART 2: Create cuda/22-cuda-apt-get-installation.md (Completed 2025-11-13 15:45)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/13-nvidia-container-cuda-packaging.md (apt package structure)
- [ ] Grep for "apt-get" AND "cuda-libraries" in cuda/
- [ ] Identify gaps: Detailed apt-get workflows, package selection

**Step 1: Web Research**
- [ ] Search: "NVIDIA CUDA apt-get installation Ubuntu"
- [ ] Search: "cuda-libraries vs cuda-libraries-dev apt packages"
- [ ] Search: "apt-get install cuda minimal packages"
- [ ] Find official NVIDIA apt repository setup

**Step 2: Create Knowledge File Structure**

Section 1: CUDA Apt Repository Setup (~100 lines)
- Add NVIDIA package repository
- GPG key installation
- apt-get update patterns
- Version pinning (CUDA 12.4 vs 12.6)

Section 2: Package Selection Strategies (~150 lines)
- cuda-libraries (runtime only, ~1GB)
- cuda-libraries-dev (includes CUPTI, ~3GB)
- cuda-toolkit-12-4 (full toolkit, ~5GB)
- Minimal package combinations (what do you really need?)

Section 3: CUPTI-Specific Installation (~100 lines)
- Extract libcupti.so from cuda-libraries-dev
- Selective installation (install, copy, remove pattern)
- Size optimization (1.52GB vs 4.5GB)
- LD_LIBRARY_PATH configuration

Section 4: Multi-Stage Docker Patterns (~100 lines)
- Install in builder, copy to runtime
- apt-get install --no-install-recommends
- apt-get clean && rm -rf /var/lib/apt/lists/*
- Conditional CUPTI installation (ARG ENABLE_PROFILING=0)

**Step 3: Citations**
- NVIDIA CUDA Installation Guide (official)
- Ubuntu package repositories
- Reference cuda/13-nvidia-container-cuda-packaging.md

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-cuda-apt-installation-2025-11-13.md
- Summary: apt-get strategies for minimal CUDA installations
- Key insight: cuda-libraries-dev extraction saves 3GB vs full install

---

## PART 3: Create cuda/23-docker-devel-as-runtime.md (~350 lines)

**Topic**: Use devel as runtime - Simplest approach (but largest images)

- [✓] PART 3: Create cuda/23-docker-devel-as-runtime.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/13-nvidia-container-cuda-packaging.md (devel vs runtime)
- [ ] Grep for "devel" in cuda/
- [ ] Identify gaps: When to use devel directly, tradeoffs

**Step 1: Web Research**
- [ ] Search: "NVIDIA CUDA devel image as runtime production"
- [ ] Search: "Docker image size optimization devel vs runtime"
- [ ] Find: Use cases for devel images in production

**Step 2: Create Knowledge File Structure**

Section 1: Devel vs Runtime Comparison (~100 lines)
- Size: nvidia/cuda:12.4.0-runtime (1.5GB) vs devel (4.5GB)
- Contents: What's in devel that's not in runtime
- CUPTI included by default in devel
- Build tools: nvcc, CMake, headers

Section 2: When to Use Devel as Runtime (~100 lines)
- Development environments (quick iteration)
- Profiling/debugging workflows (CUPTI always available)
- Prototyping (don't optimize yet)
- Single-stage builds (simplicity over size)

Section 3: Production Tradeoffs (~100 lines)
- Cost: 3GB larger images = slower deploys, more bandwidth
- Benefits: No missing library issues, full debugging capability
- Cloud Build: 4.5GB = longer upload/download times
- Vertex AI: Image pull time impact on cold starts

Section 4: CUPTI Investigation Context (~50 lines)
- Why we considered devel as runtime (CUPTI headaches)
- Decision: NOT recommended for arr-coc-0-1 production
- Use case: Temporary debugging (switch to devel, debug, switch back)

**Step 3: Citations**
- NVIDIA NGC container catalog
- Docker best practices
- Reference cuda/10-docker-multistage-ml-optimization.md

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-devel-as-runtime-2025-11-13.md
- Summary: When to use devel images directly vs multi-stage optimization
- Key insight: 3GB cost for convenience - good for dev, bad for prod

---

## PART 4: Create cuda/24-docker-nuclear-copy-cuda.md (~400 lines)

**Topic**: Nuclear copy entire CUDA - Brute force approach (copy everything)

- [✓] PART 4: Create cuda/24-docker-nuclear-copy-cuda.md (Completed 2025-11-13 18:00)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/10-docker-multistage-ml-optimization.md
- [ ] Grep for "COPY --from" in cuda/
- [ ] Identify gaps: Full CUDA copy strategies

**Step 1: Web Research**
- [ ] Search: "Docker copy entire CUDA toolkit multi-stage"
- [ ] Search: "COPY --from builder /usr/local/cuda"
- [ ] Find: Selective copying vs nuclear approach

**Step 2: Create Knowledge File Structure**

Section 1: Nuclear Copy Explained (~100 lines)
- What is nuclear copy (copy /usr/local/cuda entirely)
- Why it's called "nuclear" (overkill but guaranteed to work)
- CUPTI investigation context (frustration → "just copy everything")

Section 2: Implementation Patterns (~150 lines)
- COPY --from=builder /usr/local/cuda /usr/local/cuda
- COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/
- Size impact: Often 2-3GB of unnecessary files
- ENV PATH and LD_LIBRARY_PATH setup

Section 3: When Nuclear Copy Makes Sense (~100 lines)
- Debugging unknown dependencies (what's actually needed?)
- Quick prototyping (optimize later)
- Complex CUDA applications (many libraries)
- Time-constrained debugging (CUPTI investigation example)

Section 4: Optimization After Nuclear (~50 lines)
- Profile which libraries are actually used (ldd, strace)
- Iterative removal (remove, test, repeat)
- Document final minimal set
- Transition to selective copying

**Step 3: Citations**
- Docker multi-stage documentation
- CUDA filesystem layout
- Reference cuda/13-nvidia-container-cuda-packaging.md

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-nuclear-copy-cuda-2025-11-13.md
- Summary: Brute force CUDA copying - when to use, how to optimize later
- Key insight: Nuclear copy = guaranteed to work, then optimize down

---

## PART 5: Create cuda/25-pytorch-cmake-findings-deep.md (~500 lines)

**Topic**: PyTorch CMake findings - How PyTorch detects CUDA/CUPTI (deep dive)

- [✓] PART 5: Create cuda/25-pytorch-cmake-findings-deep.md (Completed 2025-11-13 17:45)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/12-pytorch-cmake-build-internals.md (CMake detection)
- [ ] Grep for "CMake" AND "FindCUDA" in cuda/
- [ ] Identify gaps: Detailed CMake workflow, false positive analysis

**Step 1: Web Research**
- [ ] Search: "PyTorch CMake FindCUDA.cmake CUPTI detection"
- [ ] Search: "CMake header-only detection false positive"
- [ ] Search: GitHub pytorch/pytorch cmake/Modules/
- [ ] Find: Official PyTorch CMake code

**Step 2: Create Knowledge File Structure**

Section 1: PyTorch CMake Detection Flow (~150 lines)
- FindCUDA.cmake workflow
- FindCUPTI.cmake (if exists)
- Header detection: find_path(CUPTI_INCLUDE_DIR cupti.h)
- Library detection: find_library(CUPTI_LIBRARY cupti)
- False positive pattern: headers found, libraries missing

Section 2: CUPTI Investigation Findings (~150 lines)
- CMake found cupti.h in /usr/local/cuda/extras/CUPTI/include
- CMake did NOT find libcupti.so (runtime missing)
- Build succeeded (headers sufficient for compilation)
- Runtime failed (libraries needed for execution)
- Critical discovery: This is a CMake design limitation

Section 3: Verification Commands (~100 lines)
- CMake debugging: cmake -DCMAKE_FIND_DEBUG_MODE=ON
- CMake cache inspection: grep CUPTI CMakeCache.txt
- Manual library search: find / -name "libcupti.so*"
- PyTorch test: python -c "torch.profiler" (crashes without libcupti.so)

Section 4: Solutions and Workarounds (~100 lines)
- Force disable CUPTI: cmake -DUSE_CUPTI=OFF
- Install runtime libraries: apt-get install cuda-libraries-dev
- Minimal extraction: Copy libcupti.so only
- Verification before commit: Test profiler after build

**Step 3: Citations**
- PyTorch GitHub cmake/ directory
- CMake find_package documentation
- Reference cuda/12-pytorch-cmake-build-internals.md
- Cite arr-coc-0-1 CUPTI_INVESTIGATION_STUDY.md findings

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-pytorch-cmake-findings-2025-11-13.md
- Summary: CMake header-only false positive - headers present, libraries absent
- Key insight: CMake build succeeds, runtime fails - critical Docker gotcha

---

## PART 6: Create cuda/26-cuda-local-installer-runfile.md (~450 lines)

**Topic**: Local CUDA installer - .run file installation (alternative to apt-get)

- [✓] PART 6: Create cuda/26-cuda-local-installer-runfile.md (Completed 2025-11-13 16:45)

**Step 0: Check Existing Knowledge**
- [ ] Grep for "runfile" OR "local installer" in cuda/
- [ ] Read cuda/18-ubuntu-pytorch-cuda-setup.md (installation methods)
- [ ] Identify gaps: Runfile installation, customization

**Step 1: Web Research**
- [ ] Search: "NVIDIA CUDA runfile installation Ubuntu"
- [ ] Search: "cuda_12.4.0_550.54.14_linux.run silent install"
- [ ] Search: "CUDA runfile custom installation selective components"
- [ ] Find: NVIDIA official runfile docs

**Step 2: Create Knowledge File Structure**

Section 1: Runfile vs Apt-Get Comparison (~100 lines)
- Distribution methods: runfile (official NVIDIA) vs apt (Ubuntu repos)
- Version control: Runfile = exact version, apt = repository version
- Installation location: /usr/local/cuda vs /usr for apt
- Root required: Both need sudo

Section 2: Runfile Installation Workflow (~150 lines)
- Download: wget https://developer.download.nvidia.com/...
- Permissions: chmod +x cuda_12.4.0_linux.run
- Silent install: ./cuda_*.run --silent --toolkit
- Component selection: --no-drm, --no-man-page, --no-opengl-libs
- Verification: nvcc --version, /usr/local/cuda/bin/nvcc

Section 3: Docker Integration (~100 lines)
- Download in builder stage
- Silent installation (non-interactive)
- Selective component installation (minimal)
- ENV PATH setup: /usr/local/cuda/bin
- Size comparison: Runfile toolkit ~3GB, selective ~1.5GB

Section 4: CUPTI with Runfile (~100 lines)
- CUPTI included in runfile toolkit
- Location: /usr/local/cuda/extras/CUPTI
- libcupti.so location: /usr/local/cuda/extras/CUPTI/lib64
- LD_LIBRARY_PATH configuration
- Selective copy to runtime stage

**Step 3: Citations**
- NVIDIA CUDA Downloads (official)
- CUDA Installation Guide Linux
- Docker runfile examples
- Reference cuda/22-cuda-apt-get-installation.md (comparison)

**Step 4: Create KNOWLEDGE DROP**
- File: KNOWLEDGE-DROP-cuda-local-installer-2025-11-13.md
- Summary: CUDA runfile installation - alternative to apt-get with more control
- Key insight: Runfile = exact version control, component selection flexibility

---

## Execution Plan

**All 6 PARTs will be executed in parallel by oracle-knowledge-runners.**

Each runner:
1. Checks existing knowledge (avoid duplication)
2. Performs web research (Bright Data)
3. Creates comprehensive knowledge file (~350-500 lines)
4. Creates KNOWLEDGE DROP summary
5. Returns SUCCESS or FAILURE

---

## Expected Outputs

**6 new CUDA knowledge files:**
1. cuda/21-docker-debug-builder-stage.md (~400 lines)
2. cuda/22-cuda-apt-get-installation.md (~450 lines)
3. cuda/23-docker-devel-as-runtime.md (~350 lines)
4. cuda/24-docker-nuclear-copy-cuda.md (~400 lines)
5. cuda/25-pytorch-cmake-findings-deep.md (~500 lines)
6. cuda/26-cuda-local-installer-runfile.md (~450 lines)

**Total:** ~2,550 lines of Docker/CUDA build strategy expertise

**All content focused on:** CUPTI investigation insights, Docker optimization, production patterns

---

## Success Criteria

- [ ] All 6 knowledge files created
- [ ] All 6 KNOWLEDGE DROP files created
- [ ] Each file 350-500 lines
- [ ] Web research citations included
- [ ] Cross-references to existing cuda/ files
- [ ] arr-coc-0-1 context included
- [ ] INDEX.md updated with 6 new entries
- [ ] Git committed

---

**Ready to launch 6 parallel runners!** 🚀
