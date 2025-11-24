# Dockerfile.wandb - ARR-COC Training Container for Vertex AI
# 🔧 BuildKit cache mounts: too aggressive cache busting, excessive debugging time
#    See arr-pytorch-base/Dockerfile for full exploration.
# Last rebuild: 2025-11-15 🔥 L4 GPU config + rebuild trigger
# ✨ REBUILD TRIGGER: Testing unified CLOUD BUILDS table with dividers
#
# Uses pre-built base image with all heavy dependencies installed.
# This dramatically speeds up builds: 20min → 3-5min per code change!
#
# Base image includes:
#   - PyTorch 2.4 + CUDA 12.1 + Python 3.10
#   - All ML libraries (accelerate, wandb, transformers, etc)
#   - gcloud CLI
#
# This Dockerfile only installs arr_coc package (fast!)
FROM us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-ml-stack:latest

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# 💎 RUN-based CHONK Progress Markers (arr-trainer)
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
# This image uses: Citrine
# Other images use different gems (Sapphire, Quartz, Peridot, etc.)
#
# ALL arr- images follow this pattern!
# ═══════════════════════════════════════════════════════════════════════════

# Initialize build timestamp for CHONK elapsed time tracking
RUN echo $(date +%s) > /build_start_time

# Install gcloud CLI (required for W&B Launch on Vertex AI)
# Retry logic: GPG key download can fail with network errors
# - --retry 3: Retry up to 3 times on transient errors
# - --retry-delay 2: Wait 2 seconds between retries
# - --max-time 30: 30 second timeout per attempt (small file)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl --retry 3 --retry-delay 2 --max-time 30 \
    https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [30%] gcloud CLI INSTALLED! ☁️ Citrine ✧ ▂▃ [${CHONK_ELAPSED}s]"

# Copy codebase (Training/ is now inside ARR_COC/)
COPY ARR_COC/ ARR_COC/
RUN CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [60%] Training code DEPLOYED! 🎯💎 Citrine ✧ Topaz ✧ ▂▃▄ [${CHONK_ELAPSED}s]"

# Install ARR_COC as a proper Python package (critical for imports!)
COPY Stack/arr-trainer/pyproject.toml .

# Parallel build jobs (needed default - CLI passes actual value via --build-arg)
# WHY DEFAULT=4: Needed for Docker build compatibility. Don't remove.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# BuildKit cache mounts: impractical (see arr-pytorch-base/Dockerfile for exploration)
RUN pip install -e . && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [100%] Trainer READY! 💎🔷🔶 💠🔷💎 🔶💎🔷 Triple Harmonic ✧✧✧ YEET! [${CHONK_ELAPSED}s]"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Healthcheck (optional but useful)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())"

# Entry point for Accelerate (multi-GPU support)
ENTRYPOINT ["accelerate", "launch"]
CMD ["ARR_COC/Training/train.py"]
