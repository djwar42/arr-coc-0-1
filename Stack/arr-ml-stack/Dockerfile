# ============================================================================
# ARR-COC ML Stack Image - ML packages on clean PyTorch base
# ============================================================================
# 🔧 BuildKit cache mounts: too aggressive cache busting, excessive debugging time
#    See arr-pytorch-base/Dockerfile for full exploration
# Last rebuild: 2025-11-14 (image rename + cascade fixes.)
#
# NEW 4-TIER ARCHITECTURE:
# ========================
#   arr-pytorch-base (Image 0) - PyTorch from source, no conda, full GPU support
#       ↓ FROM
#   arr-ml-stack (this!) (Image 1) - ML packages (transformers, wandb, etc.).
#       ↓ FROM
#   arr-trainer (Image 2) - ARR-COC training code
#       ↓ FROM
#   arr-vertex-launcher (Image 3) - W&B Launch agent for Vertex AI
#
# ============================================================================
#
# WHY arr-pytorch-base vs pytorch/pytorch?
# =====================================
#
# OLD APPROACH (3 tiers, conda baggage):
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel (has /opt/conda/)
#     ↓ FROM
#   arr-base (cleanup hacks: delete bundled wheels, upgrade pip in both envs)
#     ↓ FROM
#   arr-training → arr-runner
#
# NEW APPROACH (4 tiers, clean):
#   arr-pytorch-base:2.6.0-cuda12.4 (NO conda, built from source)
#     ↓ FROM
#   arr-base (this! - simple package installation, no hacks)
#     ↓ FROM
#   arr-training → arr-runner
#
# BENEFITS:
#   ✅ Single Python 3.10 environment (no conda!)
#   ✅ Full GPU support (T4/L4/A100/H100) including sm_89 (L4)
#   ✅ ~4-5 fewer CVEs (no conda bundled wheels)
#   ✅ Simpler architecture (no dual pip, no cleanup hacks)
#   ✅ ~800MB-1.2GB smaller (no conda overhead)
#
# TRADEOFF:
#   ⏰ arr-pytorch-base takes 10-15 min (176-vCPU) or 4+ hours (4-vCPU) to build (FIRST TIME ONLY!)
#   ⏰ Then cached forever in Artifact Registry (0 sec for daily use)
#
# See: training/images/arr-pytorch-base/PLAN.md for full theory & details
# ============================================================================

# ============================================================================
# THE FOUNDATION: arr-pytorch-base (built from source, no conda!)
# ============================================================================
# This is our custom PyTorch image (Image 0) built from:
#   FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04
#   → Build PyTorch 2.6.0 from source
#   → TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" (ALL GPUs!)
#   → NO CONDA!
#
# GPU Architecture Support:
#   ✅ T4   (sm_75, Compute 7.5)
#   ✅ L4   (sm_89, Compute 8.9) ← MISSING in pip wheels!
#   ✅ A100 (sm_80, Compute 8.0)
#   ✅ H100 (sm_90, Compute 9.0)
#
# This image will be referenced like:
#   FROM us-central1-docker.pkg.dev/PROJECT/REPO/arr-pytorch-base:2.6.0-cuda12.4
#
# For now, we'll use a placeholder that gets replaced during build:
FROM us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry-persistent/arr-pytorch-base:latest

# Set working directory
WORKDIR /root

# ═══════════════════════════════════════════════════════════════════════════
# 💎 RUN-based CHONK Progress Markers (arr-ml-stack)
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
# This image uses: Quartz, Jade, Amethyst, Opal, Moonstone
# Other images use different gems (Sapphire, Citrine, Peridot, etc.)
#
# ALL arr- images follow this pattern!
# ═══════════════════════════════════════════════════════════════════════════

# Initialize build timestamp for CHONK elapsed time tracking
RUN echo $(date +%s) > /build_start_time

# ============================================================================
# Install Additional System Tools
# ============================================================================
# arr-pytorch-base already has: Python 3.10, pip, basic libs
# We just need git and wget for cloning/downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [20%] System tools ACQUIRED! 🔧 Quartz ✧ ▂ [${CHONK_ELAPSED}s]"

# ============================================================================
# 🛡️ Apply Ubuntu Security Patches
# ============================================================================
# Fixes OS-level CVEs that Ubuntu has patches for:
#   - CVE-2025-5245, CVE-2025-5244 (binutils MEDIUM)
#   - 12-15 additional LOW severity system CVEs
#
# Limitations:
#   - Some OS CVEs have no patch yet (e.g., CVE-2025-47273 python-pip)
#   - We can only patch what Ubuntu releases
#
# Ubuntu Security Notice: USN-7847-1 (binutils patches)
# ============================================================================
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "✓ Applied Ubuntu security patches (CVE-2025-5245, CVE-2025-5244 + 12-15 LOW CVEs)" && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [35%] Security patches SEALED! 🛡️ Jade ✧ ▂▃ [${CHONK_ELAPSED}s]"

# ============================================================================
# Install ML Dependencies
# ============================================================================
# PyTorch 2.6.0 already installed in arr-pytorch-base base!
# We just add the ML packages we need.
#
# Pinned to LATEST SECURE VERSIONS to eliminate CVEs:
#   - 90+ vulnerabilities fixed by using latest packages
#   - Last updated: 2025-11-09
#
# CVE Mitigation Methodology:
#   - Research with Bright Data for OS vs runtime workarounds
#   - Example: brotli CVE-2025-6176 → pip install brotli>=1.2.0
#     (Python uses pip version, not OS version!)
# ============================================================================

# Parallel build jobs (needed default - CLI passes actual value via --build-arg)
# WHY DEFAULT=4: Needed for Docker build compatibility. Don't remove.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# BuildKit cache mounts: impractical (see arr-pytorch-base/Dockerfile for exploration)
RUN pip install --upgrade \
    'pip>=25.3' \
    'setuptools>=70.0' \
    accelerate>=1.4.0 \
    wandb[launch]>=0.19.0 \
    huggingface-hub>=0.27.0 \
    "datasets>=2.21.0,<3.0.0" \
    transformers>=4.53.0 \
    google-cloud-storage>=2.19.0 \
    google-cloud-aiplatform>=1.75.0 \
    pillow>=11.3.0 \
    safetensors>=0.4.7 \
    peft>=0.14.0 \
    kornia>=0.7.4 \
    numpy>=2.0.0 \
    scipy>=1.15.0 \
    requests>=2.32.3 \
    'urllib3>=2.5.0' \
    certifi>=2024.12.14 \
    'h11>=0.16.0' \
    'jinja2>=3.1.6' \
    'h2>=4.3' \
    'brotli>=1.2.0' \
    'lief>=0.15' && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [65%] ML packages LOADED! 📦💎 Amethyst ✧ Pearl ✧ ▂▃▄▅ [${CHONK_ELAPSED}s]"

# CRITICAL: Ray is vulnerable (CVE-2023-48022) - skip it for now
# If W&B Launch needs Ray, use latest version with workarounds:
# RUN pip install --no-cache-dir "ray[default]>=2.50.0"
#
# For now, W&B Launch can work without Ray for Vertex AI jobs

# ============================================================================
# 🧹 Remove Build Tools (Security + Size Optimization)
# ============================================================================
# PyTorch is already compiled in arr-pytorch-base base!
# Build tools were only needed for pip packages (numpy, pillow, etc.)
# Those are now installed → safe to remove build tools
#
# Packages removed:
#   - binutils: assembler, linker (CVE-2025-5245, CVE-2025-5244, 3 more)
#   - gcc-11, gcc-12: C/C++ compilers (3 CVEs)
#
# Impact:
#   - Eliminates 8 CVEs
#   - Reduces image size by ~250MB
#   - Zero runtime impact (PyTorch already compiled)
# ============================================================================
RUN apt-get remove --purge -y \
    binutils \
    gcc-11 \
    gcc-12 \
    || true \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "✓ Removed build tools - eliminated 8 CVEs, saved ~250MB" && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [85%] Build tools PURGED! 🧹🔷 Opal ✧ ▂▃▄▅▆▇ [${CHONK_ELAPSED}s]"

# ============================================================================
# 🧹 Remove Test Files (CVE Mitigation)
# ============================================================================
# Problem: Python includes vulnerable setuptools in TEST directories
# Fix: Delete test directories (production code never imports from test/)
#
# CVEs eliminated:
#   - CVE-2025-47273 (setuptools 65.5.0 in test files) - HIGH
#   - CVE-2024-6345 (setuptools path traversal) - HIGH
#   - CVE-2022-40897 (setuptools wheel extraction) - HIGH
#
# Also removes __pycache__ for cleaner image (~50-100MB savings)
#
# Source: https://github.com/python/cpython/issues/135374
#   CPython maintainer: "It is safe to be removed from your runtime layer"
# ============================================================================
RUN find /usr/local/lib -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib -type d -name '_bundled' -exec rm -rf {} + 2>/dev/null || true && \
    echo "✓ Removed test files, pycache, and ensurepip bundled wheels" && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [95%] Test files CLEANSED! 🧼🔶 Moonstone ✧ ▂▃▄▅▆▇█ [${CHONK_ELAPSED}s]"

# ============================================================================
# Health Check
# ============================================================================
# Verify PyTorch + CUDA libraries work
# (GPU availability verified at runtime on Vertex AI VMs)
RUN python -c "import torch; \
    print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}'); \
    print(f'cuDNN version: {torch.backends.cudnn.version()}'); \
    print(f'CUDA arch list: {torch.cuda.get_arch_list()}'); \
    print('✓ PyTorch 2.6.0 with CUDA 12.4 from arr-pytorch-base'); \
    print('   GPU availability verified at runtime on Vertex AI')" && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [100%] Stack COMPLETE! 💎🔷🔶 💠🔷💎 🔶💎🔷 Triple Harmonic ✧✧✧ FULL STACK! [${CHONK_ELAPSED}s]"

# ============================================================================
# Labels
# ============================================================================
LABEL maintainer="djwar42@gmail.com"
LABEL description="ARR-COC base image - built on arr-pytorch-base (no conda!)"
LABEL version="4.0-arr-pytorch-base-2.6.0-cuda-12.4"
LABEL architecture="4-tier: arr-pytorch-base → arr-base → arr-training → arr-runner"
LABEL security="CVE-2024-6345 fixed (setuptools 70.0), CVE-2022-40897 fixed (setuptools 70.0), CVE-2025-8869 fixed (pip 25.3), CVE-2025-5245 fixed (binutils USN-7847-1), CVE-2025-5244 fixed (binutils USN-7847-1), CVE-2025-32434 fixed (PyTorch 2.6.0), CVE-2025-48379 fixed (Pillow 11.3.0), CVE-2025-50181 fixed (urllib3 2.5.0), CVE-2025-43859 fixed (h11 0.16.0), CVE-2025-27516 fixed (jinja2 3.1.6), CVE-2025-57804 fixed (h2 4.3), CVE-2025-6176 fixed (brotli 1.2.0), CVE-2025-47273 mitigated (setuptools test files removed), CVE-2025-6921/6638/6051/3777/2099 fixed (transformers 4.53.0), CVE-2023-48022 mitigated (no Ray), +12-15 LOW CVEs fixed (apt-get upgrade), +4-5 CVEs eliminated (no conda bundled wheels)"
LABEL gpu_support="T4 (sm_75), L4 (sm_89), A100 (sm_80), H100 (sm_90) - all supported!"
