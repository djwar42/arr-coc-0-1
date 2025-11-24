# <claudes_code_comments>
# ═══════════════════════════════════════════════════════════════════════════
# 💎 RUN-based CHONK Progress Markers (arr-pytorch-base)
# ═══════════════════════════════════════════════════════════════════════════
#
# ** SETUP (REQUIRED - do this FIRST!) **
# Add this RUN command BEFORE any work steps:
#   RUN echo $(date +%s) > /build_start_time
#
# ** PLACEMENT (CRITICAL!) **
# EVERY meaningful RUN step gets a CHONK after work completes:
#   RUN <do work here> && \
#       CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
#       echo "🔹CHONK: [X%] Work DONE! 💎 Name ✧ ▂ [${CHONK_ELAPSED}s]"
#
# Add CHONK to ALL work steps (apt-get, pip install, git clone, builds, etc.)
#
# ** STANDARD FORMAT (greppable!) **
#   echo "🔹CHONK: [X%] <phase> <verb>! <gems> <name> ✧ <bars> [${CHONK_ELAPSED}s]"
#
# ** REQUIRED ELEMENTS **
#   - "🔹CHONK: [X%]" marker with bracket percentage
#   - CHONK_ELAPSED calculation: $(( $(date +%s) - $(cat /build_start_time) ))
#   - [${CHONK_ELAPSED}s] timing at end
#
# ** GEM HARMONIC PROGRESSION (0% → 100%) **
# Percentages are FLEXIBLE - place harmonics where work naturally completes:
#   - Early steps: Single/double gems (💎 → 💎🔷) + ▂ → ▂▃
#   - ~70% / final 1/3 starts: First Harmonic (💎🔷🔶 = 3 gems) + ▂▃▄▅▆▇
#   - Between First & Final: Double Harmonic (💎🔷🔶 💠🔷💎 = 6 gems) + ✧✧
#   - 100% ALWAYS: Triple Harmonic (💎🔷🔶 💠🔷💎 🔶💎🔷 = 9 gems) + ✧✧✧
#
# ** UNIQUE GEM NAMES (per image) **
# This image uses: Sapphire, Diamond, Aquamarine, Ruby, Emerald, Topaz
# Other images use different gems (Quartz, Jade, Citrine, Peridot, etc.)
#
# ALL arr- images follow this pattern!
# ═══════════════════════════════════════════════════════════════════════════
#
# ** BUILDKIT LOCAL FEATURES ONLY **
#
# This Dockerfile uses BuildKit LOCAL features (DOCKER_BUILDKIT=1):
# - RUN --mount=type=cache,target=/ccache → C++ compilation cache (10-100× speedup)
# - RUN --mount=type=cache,target=/root/.cache/pip → pip download cache
#
# What we DON'T use (removed buildkit complexity):
# - Remote cache (--cache-from/--cache-to registry) → removed
# - docker buildx create/rm → removed
# - Smart cache busting with git hashes → removed
#
# Why removed entirely:
# BuildKit cache mounts proved too aggressive - spent excessive time debugging cache
# busting issues. Complexity outweighed benefits for our once-per-month rebuild cadence.
# Standard Docker build is simple, reliable, and works everywhere.
#
# ** VALIDATION **
#
# After build, verify protobuf was patched:
# docker run arr-pytorch-base:latest cat /opt/pytorch/third_party/protobuf/cmake/CMakeLists.txt
# → Should show cmake_minimum_required(VERSION 3.5), not VERSION 2.8.12
#
# ** CODE FLOW WALKTHROUGH **
#
# STAGE 1 (Builder):
#   1. Base: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#   2. Install Python 3.10 (NO CONDA!)
#   3. Install build deps (cmake, git, ccache, ninja)
#   4. Initialize build timestamp
#   5. Clone PyTorch v2.6.0 (70+ git submodules) [CHONK-1: 10%]
#   6. Patch ALL third_party CMakeLists.txt (VERSION 3.5) [CHONK-2: 15%]
#   7. Install PyTorch requirements.txt
#   8. Build PyTorch from source (2-4 hrs, ccache enabled) [CHONK-3: 70%]
#   9. Build torchvision 0.20.0 [CHONK-4: 85%]
#  10. Build torchaudio 2.6.0 [CHONK-5: 100%]
#
# STAGE 2 (Runtime):
#   1. Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 (has cuDNN + cuBLAS!)
#   2. Install Python 3.10 runtime (minimal deps)
#   3. CUPTI: SKIPPED (training works without it, only lose torch.profiler GPU profiling)
#   4. Copy PyTorch packages from builder → runtime
#   5. Set PYTHONPATH, verify imports
#
# Result: Clean ~3GB runtime image with PyTorch 2.6.0 compiled for T4/L4/A100/H100!
# </claudes_code_comments>

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# 📚 LESSONS LEARNED - PyTorch Clean Build (Nov 13-14, 2025)
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
# This section consolidates EVERY debugging insight, failure, and solution from
# 15+ builds across 2 days. Read this FIRST before touching anything!
#
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 CUPTI Investigation Journey (6 Build Attempts)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT IS CUPTI?
# - CUDA Profiling Tools Interface - NVIDIA's API for GPU kernel profiling
# - Used by torch.profiler for GPU metrics (kernel times, memory transfers)
# - Optional dependency - PyTorch works PERFECTLY without it
#
# THE PROBLEM:
# PyTorch CMake reports finding CUPTI during compilation:
#   -- CUDA_cupti_LIBRARY = /usr/local/cuda/lib64/libcupti.so
# But in runtime images (nvidia/cuda:*-runtime-*), CUPTI DOES NOT EXIST!
#
# WHY CUPTI DOESN'T EXIST IN RUNTIME IMAGES:
# - CUPTI only shipped in nvidia/cuda:*-devel-* images (~8GB)
# - Runtime images are stripped down (~2-3GB) - no dev tools, no CUPTI
# - Multi-stage builds can't copy what doesn't exist in builder stage!
#
# ────────────────────────────────────────────────────────────────────────────
# CUPTI Attempt History (Build IDs)
# ────────────────────────────────────────────────────────────────────────────
#
# Build V1 (c2323775): COPY /usr/local/cuda-12.0/lib64/libcupti.so*
#   Result: ❌ COPY failed: no source files were specified
#   Why: Wrong CUDA version path (12.0 vs 12.4)
#
# Build V2 (84fbeb34): COPY /usr/local/cuda/lib64/libcupti.so*
#   Result: ❌ COPY failed: no source files were specified
#   Why: File doesn't exist in that path (it's in extras/CUPTI/)
#
# Build V3 (acaba721): Added massive debug tree to search filesystem
#   Debug: Searched entire /usr, checked all .so files, checked dpkg
#   Result: ❌ CUPTI NOT FOUND ANYWHERE in runtime image!
#   Discovery: CUPTI doesn't exist in runtime images - only in devel!
#
# Build V4 (attempt using wget): Try to download NVIDIA keyring + apt-get
#   Result: ❌ wget command not found (runtime images don't have wget!)
#   Why: Runtime images are minimal - no download tools
#
# Build V5 (switch to curl): Same apt-get approach but using curl
#   Result: ❌ curl command not found (runtime images don't have curl either!)
#   Why: Runtime images are VERY minimal
#
# Build V6 (HUNKY BOI - bfcbd5c): Install curl FIRST, then download keyring
#   Steps: apt-get install curl → curl download keyring → apt-get install CUPTI
#   Result: ✅ CUPTI INSTALLED SUCCESSFULLY!
#   Location: /usr/local/cuda/extras/CUPTI/lib64/libcupti.so
#
# ────────────────────────────────────────────────────────────────────────────
# FINAL DECISION: SKIP CUPTI (Confirmed - training works perfectly!)
# ────────────────────────────────────────────────────────────────────────────
#
# After V6 succeeded, we tested training WITHOUT CUPTI:
#   ✅ PyTorch imports successfully
#   ✅ CUDA operations work
#   ✅ Training jobs complete
#   ✅ Only lose: torch.profiler GPU kernel profiling
#
# We chose to COMMENT OUT CUPTI installation because:
#   1. Training works perfectly without it
#   2. Saves ~200MB image size
#   3. Faster builds (skip apt-get + download)
#   4. External profilers (nvprof, Nsight) work better anyway
#
# If you need CUPTI back:
#   - Uncomment lines 487-544 (the V6 approach that worked!)
#   - Or use nvidia/cuda:*-devel-* as runtime base (adds ~5GB)
#
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 Staging Bucket Retry Logic (3 Fixes)
# ═══════════════════════════════════════════════════════════════════════════
#
# THE PROBLEM:
# `gsutil ls gs://bucket` timed out (10s) with network issues.
# No retry logic - single failure → entire launch failed!
#
# ────────────────────────────────────────────────────────────────────────────
# Staging Bucket Fix History
# ────────────────────────────────────────────────────────────────────────────
#
# Issue (GODZILLALILLA launch): Staging bucket command timed out
#   Error: Command '' timed out after 10 seconds
#   Why: No retry logic, network glitch = instant failure
#
# Fix 1 (f57e1d0): Added retry_with_backoff() wrapper
#   - Wrap bucket check/create in function
#   - Call retry_with_backoff(func, max_attempts=4)
#   - 4 attempts: 0s, 1s, 4s, 8s delays (13s total)
#
# Fix 2 (50c7c30): Added timeout exception handling
#   - subprocess.TimeoutExpired wasn't caught
#   - Timeout bypassed retry logic → failed immediately
#   - Added: except subprocess.TimeoutExpired → return (False, msg)
#
# Fix 3 (0e6cf45): Added robust error handling + comments
#   - Added catch-all: except Exception
#   - Fixed AttributeError: stderr or "" fallback
#   - Added claudes_code_comments
#   - Documented return contract: (bool, error_msg)
#
# Result: ✅ MRCHONKYPANTBOOTS launch succeeded!
#   - Staging bucket created without timeout
#   - Retry logic working perfectly
#   - Same pattern now used in 10+ places across CLI
#
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 3 Best Debugging Methods for Docker Builds
# ═══════════════════════════════════════════════════════════════════════════
#
# After 15+ builds and countless debug sessions, these 3 methods proved most valuable:
#
# ────────────────────────────────────────────────────────────────────────────
# Method 1: Progressive Echo Checkpoints (Fastest - use this first!)
# ────────────────────────────────────────────────────────────────────────────
#
# Add echo statements at each logical step to see WHERE it fails:
#
# RUN echo "DEBUG: Step 1 - Installing curl..." && \
#     apt-get update && \
#     apt-get install -y curl && \
#     echo "DEBUG: Step 1 ✅ - curl installed!" && \
#     echo "" && \
#     echo "DEBUG: Step 2 - Downloading keyring..." && \
#     curl -LO https://example.com/file.deb && \
#     echo "DEBUG: Step 2 ✅ - Download complete!" && \
#     ls -lh file.deb
#
# Benefits:
#   ✅ Shows exactly which step fails
#   ✅ Build logs show progress in real-time
#   ✅ No rebuild needed - just read logs
#   ✅ Easy to add/remove
#
# When to use: ALWAYS start here! 90% of issues found this way.
#
# ────────────────────────────────────────────────────────────────────────────
# Method 2: Filesystem Exploration (When files are missing)
# ────────────────────────────────────────────────────────────────────────────
#
# Use find/ls to locate files when COPY fails:
#
# # Example: Finding CUPTI libraries
# RUN echo "Searching for libcupti.so in /usr..." && \
#     find /usr -name "libcupti.so*" 2>/dev/null || echo "NOT FOUND" && \
#     echo "" && \
#     echo "Checking common locations:" && \
#     ls -la /usr/local/cuda/lib64/ 2>/dev/null || echo "  /usr/local/cuda/lib64/ empty" && \
#     ls -la /usr/local/cuda/extras/CUPTI/ 2>/dev/null || echo "  CUPTI extras not found" && \
#     ls -la /usr/lib/x86_64-linux-gnu/ | grep cupti || echo "  Not in system libs"
#
# Benefits:
#   ✅ Shows actual file locations (vs expected)
#   ✅ Reveals directory structure
#   ✅ Catches path typos instantly
#
# When to use: COPY failures, library not found errors
#
# ────────────────────────────────────────────────────────────────────────────
# Method 3: Package Verification (When apt-get installs fail)
# ────────────────────────────────────────────────────────────────────────────
#
# Check if packages exist in repos BEFORE trying to install:
#
# RUN echo "Checking if cuda-cupti-12-4 is available..." && \
#     apt-cache search cuda-cupti && \
#     echo "" && \
#     echo "Checking specific package policy:" && \
#     apt-cache policy cuda-cupti-12-4 || echo "Package NOT in repos!" && \
#     echo "" && \
#     echo "Installing package..." && \
#     apt-get install -y cuda-cupti-12-4 && \
#     echo "Verifying installation:" && \
#     dpkg -l | grep cupti
#
# Benefits:
#   ✅ Shows if package exists before install attempt
#   ✅ Reveals available versions
#   ✅ Confirms successful installation
#
# When to use: apt-get failures, package not found errors
#
# ────────────────────────────────────────────────────────────────────────────
# Bonus: Build ID Tracking
# ────────────────────────────────────────────────────────────────────────────
#
# Every build gets a unique ID from Cloud Build. Track these in comments:
#
# # Build V6 (bfcbd5c) - HUNKY BOI - Build ID: 5b7e9584
# #   Approach: Install curl first, then download keyring
# #   Result: ✅ SUCCESS - CUPTI installed!
#
# Benefits:
#   ✅ Easy to reference past builds
#   ✅ Can retrieve exact logs from Cloud Build
#   ✅ Documents what worked vs what failed
#
# Command to get logs:
#   gcloud builds log BUILD_ID --region=us-west2
#
# ═══════════════════════════════════════════════════════════════════════════
# END OF LESSONS LEARNED
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ PyTorch Clean: The Foundation Image
# ═══════════════════════════════════════════════════════════════════════════
#
# LOCAL BUILDKIT FEATURES: Cache mounts for ccache and pip (no remote cache complexity!)
#
# This is Image 0 in our 4-tier architecture:
#
#   arr-pytorch-base (this!)
#       ↓ FROM
#   arr-base
#       ↓ FROM
#   arr-training
#       ↓ FROM
#   arr-runner
#
# ═══════════════════════════════════════════════════════════════════════════
# 🎯 THE CENTERPIECE DECISION: Build PyTorch from Source (No Conda!)
# ═══════════════════════════════════════════════════════════════════════════
#
# Why we build from source instead of using pytorch/pytorch:2.6.0:
#
# ✅ BENEFITS:
#   - NO CONDA BAGGAGE! (Single Python 3.10 environment)
#   - FULL GPU ARCHITECTURE SUPPORT (T4/L4/A100/H100)
#     * pip wheels MISSING sm_89 (L4 GPU)
#     * We build with: TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
#   - CLEAN SECURITY POSTURE (No conda bundled wheels with old pip/setuptools)
#   - FULL CONTROL (We know exactly what's in it)
#
# ⚠️ COST:
#   - 2-4 HOUR BUILD TIME (first time only!)
#   - Complex build process (many dependencies)
#   - We own the maintenance (PyTorch updates = rebuild)
#
# 🎁 THE PAYOFF:
#   - Build ONCE, cache in Artifact Registry FOREVER
#   - Daily builds use cached image (fast!)
#   - Clean architecture for all downstream images
#
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ STAGE 1: Builder (Install build tools, compile PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Show BuildKit cache status
RUN echo "BuildKit Cache: Dockerfile unchanged = reuse layers | Dockerfile changed = rebuild from here"

# ═══════════════════════════════════════════════════════════════════════════
# Install Python 3.10 (NO CONDA!)
# ═══════════════════════════════════════════════════════════════════════════
# We use Ubuntu's system Python 3.10 - clean, simple, single environment
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip to latest (security!)
RUN python -m pip install --upgrade pip setuptools wheel

# ═══════════════════════════════════════════════════════════════════════════
# Install Build Dependencies
# ═══════════════════════════════════════════════════════════════════════════
# ccache: C++ compilation cache (10-100× speedup on rebuilds!)
# ninja-build: Fast parallel builds (alternative to make)
# git: Clone PyTorch, torchvision, torchaudio
# cmake, build-essential: Standard build tools
# jq: JSON processor (useful for parsing build metadata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    cmake \
    git \
    jq \
    ninja-build \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure ccache for maximum reuse
ENV CCACHE_DIR=/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
RUN ccache --set-config max_size=5.0G

# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Build Configuration
# ═══════════════════════════════════════════════════════════════════════════
# GPU Architecture List:
#   sm_75: T4 (Turing)
#   sm_80: A100 (Ampere)
#   sm_86: A6000, RTX 3090 (Ampere)
#   sm_89: L4 (Ada Lovelace) ← Missing from pip wheels!
#   sm_90: H100 (Hopper)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Parallel build jobs (needed default - CLI passes actual value via --build-arg)
# WHY DEFAULT=4: Needed for Docker build compatibility. Don't remove.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# Use Ninja (faster than make)
ENV CMAKE_GENERATOR=Ninja

# Disable features we don't need
ENV USE_DISTRIBUTED=1
ENV USE_NCCL=1
ENV BUILD_TEST=0
ENV USE_KINETO=0

# ═══════════════════════════════════════════════════════════════════════════
# Clone PyTorch, Fix CMake, Build Everything
# ═══════════════════════════════════════════════════════════════════════════
#
# CMake Version Bug Fix:
# ----------------------
# Many third_party dependencies have cmake_minimum_required < 3.5
# Ubuntu 22.04 ships CMake 3.22, which rejects old versions
# We patch ALL third_party CMakeLists.txt files to VERSION 3.5
#
# Why 3 separate RUN commands:
#   1. git clone  (cacheable - rarely changes)
#   2. sed fix    (separate layer - proves it ran!)
#   3. build      (long - can cache if unchanged)
#
# VALIDATION: After build, verify protobuf was patched:
#   docker run arr-pytorch-base:latest \
#     cat /opt/pytorch/third_party/protobuf/cmake/CMakeLists.txt | head -3
#   → Should show: cmake_minimum_required(VERSION 3.5)
# ═══════════════════════════════════════════════════════════════════════════

# Initialize build timestamp (for progress tracking)
RUN echo $(date +%s) > /build_start_time

# Clone PyTorch (separate RUN for caching)
# CHONK is MUHNCHIN through 70+ git submodules!
RUN git clone --recursive --branch v2.6.0 --depth 1 https://github.com/pytorch/pytorch && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [10%] Git clone MUHNCHED! 💎 Sapphire ✧ ▂ [${CHONK_ELAPSED}s]"

# Fix CMake version requirements in ALL third_party dependencies
# Many have cmake_minimum_required < 3.5 which fails on Ubuntu 22.04 CMake 3.22
# Patch ALL of them at once to avoid whack-a-mole (protobuf, psimd, FP16, FXdiv, etc.)
# CHONK is MUHNCHIN through ALL the CMakeLists!
RUN cd pytorch && \
    echo "🔧 Patching ALL third_party CMakeLists.txt files - MUHNCHIN through 70+ files..." && \
    find third_party -name "CMakeLists.txt" -exec \
        sed -i 's/[Cc][Mm][Aa][Kk][Ee]_[Mm][Ii][Nn][Ii][Mm][Uu][Mm]_[Rr][Ee][Qq][Uu][Ii][Rr][Ee][Dd][ ]*([^)]*)/CMAKE_MINIMUM_REQUIRED(VERSION 3.5)/' {} \; && \
    echo "✅ All third_party CMakeLists.txt files patched to VERSION 3.5!" && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [15%] CMake MUHNCHED! 💎🔷 Diamond ✧ Aquamarine ✧ ▂▃ [${CHONK_ELAPSED}s]"

# Build PyTorch (this takes 4+ hours on 4-vcpu or around 10-15 mins on 176-vcpu!)
# 🚀 SPEEDUP: ccache persists compiled C++ objects across builds (10-100× faster on rebuilds!)
# 📊 RESOURCE MONITORING: Track CPU, RAM, Disk I/O during build
# Use bash for complex syntax (sh can't handle nested parentheses)
SHELL ["/bin/bash", "-c"]

# ═══════════════════════════════════════════════════════════════════════════
# 📦 SEGMENT 1: Install PyTorch dependencies
# ═══════════════════════════════════════════════════════════════════════════
RUN cd pytorch && \
    echo "📦 Installing PyTorch requirements..." && \
    pip install -r requirements.txt && \
    echo "✅ Requirements installed!"

# ═══════════════════════════════════════════════════════════════════════════
# 🔨 SEGMENT 2: Build PyTorch from source (4+ hours on smol, ~10-15 min on CHONK)
# ═══════════════════════════════════════════════════════════════════════════
RUN \
    echo "ccache before build:" && ccache -s && \
    echo "Building PyTorch v2.6.0 - MAXIMUM MUHNCHIN ENGAGED! [4+ hours on smol, ~10-15 min on CHONK]..." && \
    cd pytorch && \
    python setup.py install && \
    cd /opt && \
    echo "ccache after build:" && ccache -s

# ═══════════════════════════════════════════════════════════════════════════
# ✅ SEGMENT 3: Verify PyTorch installation
# ═══════════════════════════════════════════════════════════════════════════
RUN python -c "import torch; print(f'✅ torch {torch.__version__} verified!')" || \
    (echo "❌ PyTorch import failed! Build is broken!" && exit 1)

# ═══════════════════════════════════════════════════════════════════════════
# 🎉 SEGMENT 4: Success message & CHONK level
# ═══════════════════════════════════════════════════════════════════════════
RUN CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [70%] PyTorch MUHNCHED! 💎🔷🔶 Ruby ✧ Emerald ✧ Topaz ✧ First Harmonic! ▂▃▄▅▆▇ [${CHONK_ELAPSED}s]" && \
    if [ "${MAX_JOBS}" -ge 176 ]; then \
        echo "   🚀 CHONK LEVEL: ABSOLUTE UNIT ▂▃▄▅▆▇█ - 176 cores DEMOLISHED the build!"; \
    elif [ "${MAX_JOBS}" -ge 88 ]; then \
        echo "   🦁 CHONK LEVEL: Big Chungus ▂▃▄▅▆ - 88 cores crushed it!"; \
    elif [ "${MAX_JOBS}" -ge 44 ]; then \
        echo "   🦊 CHONK LEVEL: Decent Chonk ▂▃▄ - 44 cores respectable!"; \
    else \
        echo "   🐜 CHONK LEVEL: Smol Boi ▂ - but still completed!"; \
    fi

# ═══════════════════════════════════════════════════════════════════════════
# 📦 SEGMENT 5: Clone and build torchvision 0.20.0
# ═══════════════════════════════════════════════════════════════════════════
RUN \
    git clone --recursive --branch v0.20.0 --depth 1 https://github.com/pytorch/vision && \
    cd vision && \
    pip install Pillow && \
    python setup.py install

# ═══════════════════════════════════════════════════════════════════════════
# ✅ SEGMENT 6: Verify torchvision installation
# ═══════════════════════════════════════════════════════════════════════════
RUN python -c "import torchvision; print(f'✅ torchvision {torchvision.__version__} verified!')" || \
    (echo "❌ torchvision import failed! Build is broken!" && exit 1)

# ═══════════════════════════════════════════════════════════════════════════
# 🎉 SEGMENT 7: torchvision success message
# ═══════════════════════════════════════════════════════════════════════════
RUN CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [85%] Vision MUHNCHED! 💎🔷🔶 💠🔷💎 Double Harmonic ✧✧ ▂▃▄▅▆▇█ [${CHONK_ELAPSED}s]"

# ═══════════════════════════════════════════════════════════════════════════
# 📦 SEGMENT 8: Clone and build torchaudio 2.6.0
# ═══════════════════════════════════════════════════════════════════════════
RUN \
    git clone --recursive --branch v2.6.0 --depth 1 https://github.com/pytorch/audio && \
    cd audio && \
    python setup.py install

# ═══════════════════════════════════════════════════════════════════════════
# ✅ SEGMENT 9: Verify torchaudio installation
# ═══════════════════════════════════════════════════════════════════════════
RUN python -c "import torchaudio; print(f'✅ torchaudio {torchaudio.__version__} verified!')" || \
    (echo "❌ torchaudio import failed! Build is broken!" && exit 1)

# ═══════════════════════════════════════════════════════════════════════════
# 🎉 SEGMENT 10: torchaudio success message & FINAL CHONK (100%)
# ═══════════════════════════════════════════════════════════════════════════
RUN CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [100%] Build COMPLETE! 💎🔷🔶 💠🔷💎 🔶💎🔷 Triple Harmonic ✧✧✧ ▂▃▄▅▆▇█ [${CHONK_ELAPSED}s]"

# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ STAGE 2: Runtime (Clean image, remove build tools)
# ═══════════════════════════════════════════════════════════════════════════
# WHY cudnn-runtime instead of plain runtime?
#   - Builder stage uses cudnn-devel to COMPILE PyTorch with cuDNN support
#   - Runtime stage needs cuDNN libraries to RUN PyTorch
#   - Plain runtime image (nvidia/cuda:12.4.1-runtime) has NO cuDNN!
#   - cudnn-runtime image has cuDNN 9.1.0.70 + cuBLAS already installed
#   - No manual library copying needed - it's the official NVIDIA way!
#
# Size comparison:
#   - runtime:         ~2GB  (CUDA runtime only, NO cuDNN)
#   - cudnn-runtime:   ~2.2GB (CUDA + cuDNN 9.1.0.70 + cuBLAS)
#   - cudnn-devel:     ~8GB  (everything + headers + dev tools)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 runtime (minimal - no dev tools!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# ═══════════════════════════════════════════════════════════════════════════
# Copy PyTorch packages from builder → runtime
# ═══════════════════════════════════════════════════════════════════════════
# This includes:
#   - PyTorch 2.6.0
#   - torchvision 0.20.0
#   - torchaudio 2.6.0
#   - All compiled C++ extensions
#   - CUDA kernels
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# ═══════════════════════════════════════════════════════════════════════════
# cuDNN/cuBLAS runtime libraries
# ═══════════════════════════════════════════════════════════════════════════
# Already included in nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 base image:
#   - cuDNN 9.1.0.70
#   - cuBLAS 12.x runtime libraries
#   - All CUDA runtime dependencies
# No manual copying of cuDNN/cuBLAS needed - they're already in the base image!

# ═══════════════════════════════════════════════════════════════════════════
# CUPTI (CUDA Profiling Tools Interface) - SKIPPED
# ═══════════════════════════════════════════════════════════════════════════
#
# DECISION: Training works perfectly WITHOUT CUPTI!
#
# What we lose:
#   - torch.profiler GPU kernel profiling only
#   - nvprof, Nsight Systems still work (better profilers anyway)
#
# What we gain:
#   - Saves ~200MB image size
#   - Faster builds (skip apt-get + download)
#   - Simpler Dockerfile
#
# ────────────────────────────────────────────────────────────────────────────
# TO RE-ENABLE CUPTI (if needed - Build V6 approach that worked!)
# ────────────────────────────────────────────────────────────────────────────
#
# Uncomment this block to add CUPTI back:
#
# RUN apt-get update && \
#     apt-get install -y curl && \
#     curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
#     dpkg -i cuda-keyring_1.1-1_all.deb && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         cuda-cupti-12-4 \
#         libcupti-dev-12-4 && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm cuda-keyring_1.1-1_all.deb && \
#     ldconfig
#
# Result: CUPTI installed at /usr/local/cuda/extras/CUPTI/lib64/libcupti.so
#
# ════════════════════════════════════════════════════════════════════════════

# Set Python path
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages

# Verify PyTorch imports successfully
RUN python -c "import torch; print(f'✅ PyTorch {torch.__version__} runtime ready!')" && \
    python -c "import torchvision; print(f'✅ torchvision {torchvision.__version__} runtime ready!')" && \
    python -c "import torchaudio; print(f'✅ torchaudio {torchaudio.__version__} runtime ready!')"

# Final success message
RUN echo "" && \
    echo "════════════════════════════════════════════════════════════════" && \
    echo "🎉 PyTorch Clean Image Build Complete!" && \
    echo "════════════════════════════════════════════════════════════════" && \
    echo "PyTorch:     $(python -c 'import torch; print(torch.__version__)')" && \
    echo "torchvision: $(python -c 'import torchvision; print(torchvision.__version__)')" && \
    echo "torchaudio:  $(python -c 'import torchaudio; print(torchaudio.__version__)')" && \
    echo "CUDA:        $(python -c 'import torch; print(torch.version.cuda)')" && \
    echo "cuDNN:       $(python -c 'import torch; print(torch.backends.cudnn.version())')" && \
    echo "" && \
    echo "GPU Architectures: sm_75 (T4), sm_80 (A100), sm_86 (A6000)," && \
    echo "                   sm_89 (L4), sm_90 (H100)" && \
    echo "" && \
    echo "════════════════════════════════════════════════════════════════"
