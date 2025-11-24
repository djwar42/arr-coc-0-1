# W&B Launch Runner with gcloud CLI 🛡️ (security hardened)
# 🔧 BuildKit cache mounts: too aggressive cache busting, excessive debugging time
#    See arr-pytorch-base/Dockerfile for full exploration
# Ephemeral Cloud Run Job that submits training jobs to Vertex AI
# Last rebuild: 2025-11-15 🔥 L4 GPU config + rebuild trigger
# 🚀 REBUILD TRIGGER: Launch validation ensures GPU config matches .training before submission
#
# Architecture:
#   W&B Queue → Runner (this) → Vertex AI Custom Job → Training Image
#
# Runner responsibilities:
#   - Poll W&B queues for training jobs
#   - Submit jobs to Vertex AI using gcloud/Python SDK
#   - Monitor job status and report to W&B
#   - Timeout: 600s (30 minutes max per runner)

FROM wandb/launch-agent:latest

# Switch to root to install packages
USER root

# ═══════════════════════════════════════════════════════════════════════════
# 💎 RUN-based CHONK Progress Markers (arr-vertex-launcher)
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
# This image uses: Peridot, Garnet, Turquoise, Zircon
# Other images use different gems (Sapphire, Quartz, Citrine, etc.)
#
# ALL arr- images follow this pattern!
# ═══════════════════════════════════════════════════════════════════════════

# Initialize build timestamp for CHONK elapsed time tracking
RUN echo $(date +%s) > /build_start_time

# Security: Upgrade all packages to fix MEDIUM/LOW vulnerabilities
# - Upgrades all existing packages to latest versions with security patches
# - Uses --no-cache to avoid storing package index locally (reduces image size)
# - Targets vulnerabilities in base wandb/launch-agent image dependencies
RUN apk upgrade --no-cache && \
    apk add --no-cache --upgrade \
        ca-certificates \
        openssl \
        libssl3 \
        libcrypto3 && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [25%] System packages UPGRADED! 🔧 Peridot ✧ ▂▃ [${CHONK_ELAPSED}s]"

# Security: Upgrade Python packages to fix CVEs
# - CVE-2025-50181 / CVE-2025-50182: urllib3 2.3 → 2.5
# - CVE-2025-8869: pip → 25.3
# - CVE-2024-12797: cryptography → 44.0.1

# Parallel build jobs (needed default - CLI passes actual value via --build-arg)
# WHY DEFAULT=4: Needed for Docker build compatibility. Don't remove.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# BuildKit cache mounts: impractical (see arr-pytorch-base/Dockerfile for exploration)
RUN pip install --upgrade \
    'pip>=25.3' \
    'urllib3>=2.5.0' \
    'cryptography>=44.0.1' && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [40%] Python packages SECURED! 🔐💎 Garnet ✧ ▂▃▄ [${CHONK_ELAPSED}s]"

# Install gcloud CLI (Alpine Linux - uses apk not apt-get) WEEEE
# Install to /usr/local so all users can access it
#
# Retry logic: 195MB download can fail with HTTP/2 stream errors
# - --retry 5: Retry up to 5 times on transient errors
# - --retry-delay 3: Wait 3 seconds between retries
# - --retry-max-time 600: Give up after 10 minutes total
# - --http1.1: Force HTTP/1.1 to avoid HTTP/2 stream issues
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    bash \
    && cd /usr/local \
    && curl --retry 5 --retry-delay 3 --retry-max-time 600 --http1.1 \
        -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
    && tar -xf google-cloud-cli-linux-x86_64.tar.gz \
    && ./google-cloud-sdk/install.sh --quiet --usage-reporting false --path-update false \
    && chmod -R 755 /usr/local/google-cloud-sdk \
    && rm google-cloud-cli-linux-x86_64.tar.gz \
    && ln -s /usr/local/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud \
    && ln -s /usr/local/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [75%] gcloud CLI READY! ☁️💎🔷 Turquoise ✧ Aquamarine ✧ ▂▃▄▅▆ [${CHONK_ELAPSED}s]"

# Add gcloud to PATH for all users (both methods for redundancy)
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin:${PATH}"

# Set default GCP region
ENV CLOUDSDK_COMPUTE_REGION=us-central1

# Create W&B config directory for user 1000 (launch_agent) BEFORE switching users
# For gcp-vertex: Use noop builder (doesn't build, uses image from resource_args.container_spec)
RUN mkdir -p /home/launch_agent/.config/wandb && \
    printf 'builder:\n  type: noop\n' > /home/launch_agent/.config/wandb/launch-config.yaml && \
    chown -R 1000:1000 /home/launch_agent/.config && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [95%] W&B config READY! 💎🔷🔶 💠🔷💎 Zircon ✧✧ ▂▃▄▅▆▇█ [${CHONK_ELAPSED}s]"

# Copy W&B Vertex spot instance monkey-patch
# CRITICAL SOLUTION: Use sitecustomize.py - auto-loaded by Python on startup!
# Python automatically imports sitecustomize.py BEFORE any other modules load
# This ensures our patch is applied before wandb modules are imported
# Source: https://docs.python.org/3/library/site.html#module-sitecustomize
COPY wandb_vertex_patch.py /app/wandb_vertex_patch.py
COPY sitecustomize.py /usr/lib/python3.13/site-packages/sitecustomize.py
RUN chown 1000:1000 /app/wandb_vertex_patch.py /usr/lib/python3.13/site-packages/sitecustomize.py && \
    chmod 644 /usr/lib/python3.13/site-packages/sitecustomize.py

# Copy fatal error detection wrapper script
# This monitors W&B agent logs and exits immediately on fatal errors
# (machine type not supported, permissions, quota, etc.) instead of waiting 60min timeout
# Also applies wandb_vertex_patch.py BEFORE starting the agent!
COPY entrypoint-wrapper.sh /usr/local/bin/entrypoint-wrapper.sh
RUN chmod +x /usr/local/bin/entrypoint-wrapper.sh && \
    chown 1000:1000 /usr/local/bin/entrypoint-wrapper.sh && \
    CHONK_ELAPSED=$(( $(date +%s) - $(cat /build_start_time) )) && \
    echo "🔹CHONK: [100%] Fatal error detection + SPOT PATCH ARMED! 💎🔷🔶 💠🔷💎 🔶💎🔷 Triple Harmonic ✧✧✧ LIFTOFF! [${CHONK_ELAPSED}s]"

# Switch back to non-root user (if the base image had one)
USER 1000

# Override entrypoint to use our wrapper script
# The wrapper monitors agent logs and exits immediately on fatal errors
ENTRYPOINT ["/usr/local/bin/entrypoint-wrapper.sh"]
