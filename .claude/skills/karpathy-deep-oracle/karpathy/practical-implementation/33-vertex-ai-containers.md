# Vertex AI Container Management: Artifact Registry, Custom Images, and Optimization

**Complete guide to container management for Vertex AI training jobs with W&B Launch integration**

---

## Section 1: Artifact Registry Setup (~180 lines)

### Overview

Google Cloud Artifact Registry is the recommended container image registry for Vertex AI. It replaces the older Container Registry and provides enterprise-grade security, vulnerability scanning, and fine-grained access control.

From [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs) (accessed 2025-01-31):
- Unified repository for containers, language packages, and OS packages
- Native integration with Vertex AI, Cloud Build, and GKE
- Automatic vulnerability scanning and SBOM generation
- Regional and multi-regional storage options

### Creating Artifact Registry Repositories

**Repository Formats Supported:**
- Docker (container images)
- Maven, npm, Python, Go modules
- Apt, Yum (OS packages)
- Generic artifacts

**Create a Docker repository:**

```bash
# Create repository in specific region
gcloud artifacts repositories create ml-training \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML training containers"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

**Location strategies:**
- **Single region**: Lower latency, lower cost (e.g., us-central1)
- **Multi-region**: Higher availability, automatic replication (e.g., us)
- **Choose region close to Vertex AI training location** for faster pulls

From [Artifact Registry Locations](https://cloud.google.com/artifact-registry/docs/repositories/repo-locations):
- Regional repositories: 20+ locations worldwide
- Multi-regional: us, europe, asia
- Pricing varies by region and storage class

### Docker Authentication to Artifact Registry

**Method 1: Standalone Credential Helper (Recommended)**

From [Configure Authentication](https://cloud.google.com/artifact-registry/docs/docker/authentication) (accessed 2025-01-31):

```bash
# Install credential helper
gcloud components install docker-credential-gcr

# Configure Docker to use helper
gcloud auth configure-docker us-central1-docker.pkg.dev

# Verify configuration (adds to ~/.docker/config.json)
cat ~/.docker/config.json
```

**Method 2: Service Account Key (CI/CD)**

```bash
# Create service account
gcloud iam service-accounts create vertex-training \
    --display-name="Vertex AI Training SA"

# Grant Artifact Registry permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-training@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

# Generate and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=vertex-training@PROJECT_ID.iam.gserviceaccount.com

# Authenticate Docker with key
cat key.json | docker login -u _json_key --password-stdin \
    us-central1-docker.pkg.dev
```

**Method 3: Application Default Credentials (Compute Engine/Cloud Run)**

```python
# Automatic authentication when running on GCP
# No explicit configuration needed
# Uses Compute Engine default service account
```

### Image Naming Conventions

**Standard format:**
```
LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE_NAME:TAG
```

**Examples:**
```bash
# Development image
us-central1-docker.pkg.dev/my-project/ml-training/pytorch-train:dev

# Production with semantic versioning
us-central1-docker.pkg.dev/my-project/ml-training/llm-finetuning:1.2.3

# Commit-based tagging
us-central1-docker.pkg.dev/my-project/ml-training/vlm-train:git-abc123f

# Latest (use sparingly in production)
us-central1-docker.pkg.dev/my-project/ml-training/base-ml:latest
```

**Tagging best practices:**
- **Semantic versioning** for production (1.2.3)
- **Git commit SHA** for traceability (git-abc123f)
- **Branch names** for development (dev, staging)
- **Avoid `latest`** in production (non-deterministic)
- **Immutable tags** for reproducibility (don't overwrite tags)

### Repository Permissions

From [Access Control with IAM](https://cloud.google.com/artifact-registry/docs/access-control) (accessed 2025-01-31):

**IAM Roles:**
- `roles/artifactregistry.reader` - Pull images
- `roles/artifactregistry.writer` - Push and pull images
- `roles/artifactregistry.admin` - Full repository management
- `roles/artifactregistry.repoAdmin` - Repository-level admin

**Grant permissions:**

```bash
# Grant read access to Vertex AI service account
gcloud artifacts repositories add-iam-policy-binding ml-training \
    --location=us-central1 \
    --member="serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"

# Grant write access to Cloud Build
gcloud artifacts repositories add-iam-policy-binding ml-training \
    --location=us-central1 \
    --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

### Image Vulnerability Scanning

**Automatic scanning (enabled by default):**

From [Artifact Registry Security](https://cloud.google.com/artifact-registry/docs/analysis):
- Scans for OS package vulnerabilities
- Generates SBOM (Software Bill of Materials)
- Integrates with Container Analysis API
- Continuous monitoring for new CVEs

**View scan results:**

```bash
# List vulnerabilities for an image
gcloud artifacts docker images list us-central1-docker.pkg.dev/PROJECT_ID/ml-training/pytorch-train \
    --include-tags

# Get detailed vulnerability report
gcloud container images describe \
    us-central1-docker.pkg.dev/PROJECT_ID/ml-training/pytorch-train:1.0.0 \
    --show-package-vulnerability
```

**Vulnerability severity levels:**
- CRITICAL - Immediate action required
- HIGH - Fix soon (within days)
- MEDIUM - Fix in next release cycle
- LOW - Monitor and fix when convenient
- MINIMAL - Informational

### Image Cleanup Policies

**Automatic cleanup to manage costs:**

```bash
# Delete images older than 30 days (keep last 5 versions)
gcloud artifacts repositories set-cleanup-policy ml-training \
    --location=us-central1 \
    --policy=cleanup-policy.json

# cleanup-policy.json:
{
  "rules": [
    {
      "condition": {
        "olderThan": "2592000s",  # 30 days
        "packageNamePrefixes": ["pytorch-train"]
      },
      "action": {
        "type": "Delete",
        "keep": 5
      }
    }
  ]
}
```

**Cleanup strategies:**
- **Time-based**: Delete images older than X days
- **Version-based**: Keep only last N versions
- **Tag pattern**: Delete dev/* tags, keep prod/* tags
- **Test before enabling**: Use `--dry-run` flag

### Cost Optimization

**Storage costs:**
- $0.10 per GB/month (standard, regional)
- $0.20 per GB/month (multi-regional)
- Network egress charges apply

**Optimization strategies:**
1. **Use cleanup policies** - Delete old/unused images
2. **Compress layers** - Smaller images = lower storage costs
3. **Share base images** - Reuse common layers across images
4. **Regional storage** - Avoid multi-regional unless needed
5. **Monitor storage usage** - Set up billing alerts

**Check repository size:**

```bash
# List all images with sizes
gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/PROJECT_ID/ml-training \
    --include-tags \
    --format="table(image,size_bytes)"
```

---

## Section 2: Container Image Creation (~200 lines)

### Pre-built Containers Overview

From [Prebuilt Containers for Custom Training](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers) (accessed 2025-01-31):

**Available frameworks (2024-2025):**
- **PyTorch**: 2.0, 2.1, 2.2, 2.3 (with CUDA 11.8, 12.1)
- **TensorFlow**: 2.12, 2.13, 2.14, 2.15 (with CUDA 11.8, 12.1)
- **scikit-learn**: 1.0, 1.3 (CPU only)
- **XGBoost**: 1.6, 1.7 (CPU and GPU)
- **Ray on Vertex AI**: Ray 2.4, 2.5 with MLflow

**Image naming format:**
```
REGION-docker.pkg.dev/vertex-ai/training/FRAMEWORK:VERSION
```

**Examples:**
```python
# PyTorch 2.3 with CUDA 12.1
pytorch_image = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest"

# TensorFlow 2.15 with CUDA 12.1
tensorflow_image = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest"

# Scikit-learn (CPU)
sklearn_image = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest"
```

**Pre-built container advantages:**
- Google-maintained and tested
- Optimized for Vertex AI infrastructure
- Security patches and updates
- No build time required

**Limitations:**
- Fixed package versions
- Limited customization
- May include unnecessary dependencies

### Custom Dockerfile Structure

**Basic template for ML training:**

```dockerfile
# Multi-stage build for optimization
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Final stage (minimal runtime image)
# ============================================
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Non-root user for security
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app
USER trainer

# Entry point (required for Vertex AI)
ENTRYPOINT ["python", "-m", "src.train"]
```

### Base Image Selection

From [Docker Best Practices for ML](https://docs.docker.com/build/building/best-practices/) (accessed 2025-01-31):

**Comparison of base images:**

| Base Image | Size | Use Case | Pros | Cons |
|------------|------|----------|------|------|
| `python:3.10` | 920 MB | Development | All tools included | Very large |
| `python:3.10-slim` | 122 MB | Production | Smaller, faster builds | Missing some tools |
| `python:3.10-alpine` | 45 MB | Minimal deployments | Smallest size | Compatibility issues |
| `nvidia/cuda:12.1-runtime` | 1.8 GB | GPU training | CUDA pre-installed | Large, GPU-specific |
| `gcr.io/deeplearning-platform-release/base-cpu` | 2.1 GB | GCP ML | GCP-optimized | Very large |

**Recommendations:**
- **CPU training**: `python:3.10-slim`
- **GPU training**: `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04`
- **Custom**: Build from `ubuntu:22.04` or `debian:bookworm-slim`

### Installing Dependencies

**requirements.txt best practices:**

```txt
# Pin all versions for reproducibility
torch==2.3.0
transformers==4.38.0
wandb==0.16.3
datasets==2.18.0
accelerate==0.27.2

# Use compatible version specifiers
numpy>=1.24.0,<2.0.0
pillow~=10.2.0

# Hash pinning for security (pip-compile)
# torch==2.3.0 --hash=sha256:abc123...
```

**Multi-stage dependency installation:**

```dockerfile
# Stage 1: Build dependencies (includes compilers)
FROM python:3.10-slim AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchvision==0.18.0

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (no build tools)
FROM python:3.10-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code
WORKDIR /app
COPY . /app
```

### Multi-Stage Builds for Optimization

From [Docker Multi-Stage Builds Best Practices](https://docs.docker.com/build/building/multi-stage/) (accessed 2025-01-31):

**Why multi-stage builds:**
- **Smaller final images** (exclude build tools)
- **Faster deployment** (less data to transfer)
- **Better security** (fewer attack surfaces)
- **Cleaner separation** (build vs runtime)

**Advanced multi-stage example:**

```dockerfile
# ============================================
# Stage 1: Base dependencies
# ============================================
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================
# Stage 2: Build and compile
# ============================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Runtime
# ============================================
FROM base AS runtime

# Copy only installed packages (not build tools)
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages

WORKDIR /app
COPY src/ /app/src/

# Health check for Vertex AI
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import torch; print('OK')" || exit 1

ENTRYPOINT ["python", "-m", "src.train"]
```

**Size comparison:**
- Single-stage build: ~3.2 GB
- Multi-stage build: ~1.4 GB
- **Savings: 56% reduction**

### Entry Point and Command Configuration

From [Vertex AI Custom Container Requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements) (accessed 2025-01-31):

**Vertex AI requirements:**
- Container must run a training script when started
- Must handle SIGTERM for graceful shutdown
- Logs should go to stdout/stderr
- Can accept command-line arguments from Vertex AI

**Entry point patterns:**

```dockerfile
# Pattern 1: Python module execution
ENTRYPOINT ["python", "-m", "trainer.task"]

# Pattern 2: Shell script wrapper
COPY train.sh /app/
RUN chmod +x /app/train.sh
ENTRYPOINT ["/app/train.sh"]

# Pattern 3: Executable with arguments
ENTRYPOINT ["python", "train.py"]
CMD ["--epochs", "10", "--batch-size", "32"]
```

**Handling Vertex AI environment variables:**

```python
# train.py
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)

    # Vertex AI provides these automatically
    model_dir = os.getenv("AIP_MODEL_DIR", "/tmp/model")
    tensorboard_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "")

    args = parser.parse_args()

    # Training logic...
    train(args, model_dir, tensorboard_dir)

if __name__ == "__main__":
    main()
```

### Environment Variable Handling

**Standard Vertex AI environment variables:**

```bash
AIP_MODEL_DIR=/gcs/bucket/model/output
AIP_TENSORBOARD_LOG_DIR=/gcs/bucket/tensorboard
AIP_CHECKPOINT_DIR=/gcs/bucket/checkpoints
AIP_TRAINING_DATA_URI=gs://bucket/data/train
AIP_VALIDATION_DATA_URI=gs://bucket/data/val
```

**Set in Dockerfile:**

```dockerfile
# Default values (can be overridden by Vertex AI)
ENV WANDB_API_KEY=""
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers
ENV HF_HOME=/tmp/huggingface
```

### Complete Dockerfile Examples

**Example 1: PyTorch LLM Fine-tuning**

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/

# Entry point
ENTRYPOINT ["python", "-m", "src.finetune"]
```

**Example 2: VLM Training with W&B**

```dockerfile
FROM python:3.10-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# W&B configuration
ENV WANDB_API_KEY=""
ENV WANDB_PROJECT="vlm-training"
ENV WANDB_ENTITY=""

# Application code
WORKDIR /app
COPY . /app

# Non-root user
RUN useradd -m trainer && chown -R trainer:trainer /app
USER trainer

ENTRYPOINT ["python", "train_vlm.py"]
```

---

## Section 3: Container Optimization (~170 lines)

### Image Size Reduction Techniques

From [Best Practices for Building ML Containers](https://pages.run.ai/hubfs/PDFs/White%20Papers/Best%20Practices%20for%20Building%20Containers%20for%20Machine%20Learning.pdf) (accessed 2025-01-31):

**Top 10 optimization techniques:**

1. **Multi-stage builds** (covered above) - 40-60% reduction
2. **Minimal base images** - Use slim/alpine variants
3. **Layer caching** - Order Dockerfile for cache efficiency
4. **Remove build artifacts** - Clean up after installation
5. **Combine RUN commands** - Reduce layer count
6. **Use .dockerignore** - Exclude unnecessary files
7. **Compress data files** - Use tar.gz for datasets
8. **Minimize dependencies** - Install only what's needed
9. **Use distroless images** - Google's minimal images
10. **Pin package versions** - Enable layer caching

### Layer Caching Strategies

**Dockerfile ordering for optimal caching:**

```dockerfile
# ============================================
# 1. Base image (changes rarely)
# ============================================
FROM python:3.10-slim

# ============================================
# 2. System packages (changes rarely)
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 3. Python dependencies (changes occasionally)
# ============================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# 4. Application code (changes frequently)
# ============================================
WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/

# ============================================
# 5. Entry point (changes rarely)
# ============================================
ENTRYPOINT ["python", "-m", "src.train"]
```

**Why this order matters:**
- **Early layers cached longer** - System packages rarely change
- **Late layers rebuild often** - Application code changes frequently
- **Faster iteration** - Only rebuild changed layers
- **CI/CD efficiency** - Reuse cached layers across builds

**Layer caching in Cloud Build:**

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--cache-from'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-training/pytorch-train:latest'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-training/pytorch-train:$SHORT_SHA'
      - '.'
    id: 'build-image'
```

### Dependency Pinning for Reproducibility

**Why pin dependencies:**
- **Reproducible builds** - Same image every time
- **Avoid breaking changes** - New versions can break code
- **Security auditing** - Know exactly what's installed
- **Layer caching** - Unchanged requirements = cached layer

**Pin all transitive dependencies:**

```bash
# Generate locked requirements with pip-tools
pip install pip-tools
pip-compile requirements.in --output-file=requirements.txt

# Or use Poetry
poetry export --format=requirements.txt --output=requirements.txt

# Or use Pipenv
pipenv requirements > requirements.txt
```

**Example pinned requirements.txt:**

```txt
# Core ML frameworks
torch==2.3.0
torchvision==0.18.0
transformers==4.38.0

# Training utilities
wandb==0.16.3
accelerate==0.27.2

# Data processing
datasets==2.18.0
pillow==10.2.0
numpy==1.26.4

# All transitive dependencies pinned
certifi==2024.2.2
charset-normalizer==3.3.2
huggingface-hub==0.20.3
# ... (100+ more packages)
```

### GPU-Specific Optimizations

From [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html):

**CUDA and cuDNN versions:**
- **CUDA 11.8**: PyTorch 2.0-2.2, TensorFlow 2.12-2.13
- **CUDA 12.1**: PyTorch 2.3+, TensorFlow 2.14+
- **cuDNN 8**: Required for most deep learning frameworks

**Base images with CUDA:**

```dockerfile
# Option 1: NVIDIA official runtime image (recommended)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Option 2: NVIDIA devel image (for compiling CUDA code)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Option 3: Pre-built DL framework with CUDA
FROM nvcr.io/nvidia/pytorch:24.01-py3
```

**Install PyTorch with CUDA support:**

```dockerfile
# Install PyTorch compiled for CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

**GPU optimization flags:**

```python
# train.py - Enable GPU optimizations
import torch

# Enable TF32 (faster on A100/H100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN autotuner (finds fastest algorithms)
torch.backends.cudnn.benchmark = True

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Security Hardening

From [Container Security Best Practices](https://cloud.google.com/architecture/best-practices-for-building-containers):

**1. Non-root user:**

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app

# Switch to non-root user
USER trainer
```

**2. Minimal attack surface:**

```dockerfile
# Remove unnecessary packages after installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Don't install dev tools in production
# (use multi-stage build instead)
```

**3. Scan for vulnerabilities:**

```bash
# Scan image with Trivy
trivy image us-central1-docker.pkg.dev/PROJECT_ID/ml-training/train:v1

# Scan with Google Container Analysis (automatic in Artifact Registry)
gcloud container images describe \
    us-central1-docker.pkg.dev/PROJECT_ID/ml-training/train:v1 \
    --show-package-vulnerability
```

**4. Use trusted base images:**

```dockerfile
# ✅ Good: Official images
FROM python:3.10-slim
FROM nvidia/cuda:12.1-runtime

# ✅ Good: Google-maintained images
FROM gcr.io/deeplearning-platform-release/base-cpu

# ❌ Bad: Unknown third-party images
FROM random-user/ml-base:latest
```

### Container Registry Best Practices

**Image tagging strategy:**

```bash
# Semantic versioning for releases
docker tag train:latest us-central1-docker.pkg.dev/PROJECT/ml-training/train:1.2.3

# Git commit SHA for traceability
docker tag train:latest us-central1-docker.pkg.dev/PROJECT/ml-training/train:git-$(git rev-parse --short HEAD)

# Branch name for development
docker tag train:latest us-central1-docker.pkg.dev/PROJECT/ml-training/train:dev-main

# Date-based for experiments
docker tag train:latest us-central1-docker.pkg.dev/PROJECT/ml-training/train:exp-2025-01-31
```

**Immutable tags policy:**
- **Never overwrite tags** (breaks reproducibility)
- **Use unique tags** for each build (commit SHA, timestamp)
- **Reserve `latest`** for CI/CD only (not production)
- **Pin tags** in Vertex AI job specs (no `latest`)

**Repository organization:**

```
ml-training/
├── base-cpu:2025-01-31        # Base images with common deps
├── base-gpu:2025-01-31
├── pytorch-train:v1.2.3       # Application images
├── tensorflow-train:v2.0.1
└── wandb-sweep:exp-20250131   # Experimental images
```

### W&B Integration in Containers

**Dockerfile with W&B:**

```dockerfile
FROM python:3.10-slim

# Install W&B
RUN pip install --no-cache-dir wandb==0.16.3

# W&B cache directory (mount as volume)
ENV WANDB_DIR=/tmp/wandb
RUN mkdir -p /tmp/wandb && chmod 777 /tmp/wandb

# W&B API key (set at runtime, not in image)
ENV WANDB_API_KEY=""

# Disable prompts
ENV WANDB_CONSOLE=off

# Application code
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
```

**W&B API key management:**

```bash
# Option 1: Environment variable in Vertex AI job spec
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=train-job \
    --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,\
container-image-uri=us-central1-docker.pkg.dev/PROJECT/ml/train:v1,\
environment-variables="WANDB_API_KEY=abc123..."

# Option 2: Secret Manager (recommended)
gcloud secrets create wandb-api-key --data-file=-
# Enter key, then Ctrl+D

# Reference in job:
gcloud ai custom-jobs create \
    --region=us-central1 \
    --worker-pool-spec=...,\
secret-environment-variables="WANDB_API_KEY=wandb-api-key:latest"
```

**W&B artifact logging:**

```python
# train.py
import wandb
import os

# Initialize W&B (API key from environment)
wandb.init(project="vertex-training", entity="my-team")

# Log training artifacts
def train():
    # ... training code ...

    # Log model to W&B
    model_artifact = wandb.Artifact("trained-model", type="model")
    model_artifact.add_dir("./model")
    wandb.log_artifact(model_artifact)

    # Also save to Vertex AI Model Directory
    model_dir = os.getenv("AIP_MODEL_DIR")
    if model_dir:
        save_model(model_dir)
```

---

## Sources

**Web Research:**

From [Google Cloud Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs) (accessed 2025-01-31):
- Repository creation and management
- Docker authentication methods
- IAM roles and permissions

From [Push and Pull Images - Artifact Registry](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling) (accessed 2025-01-31):
- Image naming conventions
- Authentication configuration
- Push/pull workflows

From [Vertex AI Prebuilt Containers](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers) (accessed 2025-01-31):
- Available framework versions (PyTorch 2.0-2.3, TensorFlow 2.12-2.15)
- Image URIs and usage
- Supported CUDA versions

From [Vertex AI Custom Container Requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements) (accessed 2025-01-31):
- Container requirements for Vertex AI
- Entry point and environment variable specifications
- Health check guidelines

From [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/) (accessed 2025-01-31):
- Multi-stage build patterns
- Image size optimization
- Build caching strategies

From [Docker Building Best Practices](https://docs.docker.com/build/building/best-practices/) (accessed 2025-01-31):
- Dockerfile optimization techniques
- Layer caching
- Security hardening

From [Best Practices for Building ML Containers - Run:AI](https://pages.run.ai/hubfs/PDFs/White%20Papers/Best%20Practices%20for%20Building%20Containers%20for%20Machine%20Learning.pdf) (accessed 2025-01-31):
- ML-specific container patterns
- Dependency management
- GPU optimization

From [Multi-Stage Docker Builds for ML - Medium](https://medium.com/@oluoch-odhiambo.medium.com/docker-multi-stage-build-an-effective-strategy-to-building-production-ready-docker-images) (accessed 2025-01-31):
- Production-ready multi-stage patterns
- Size reduction examples
- Build optimization

From [Optimize AI Containers with Multi-Stage Builds - Collabnix](https://collabnix.com/optimize-your-ai-containers-with-docker-multi-stage-builds-a-complete-guide/) (accessed 2025-01-31):
- AI/ML specific optimization strategies
- Artifact copying between stages
- Complete examples

**Additional References:**
- [Artifact Registry IAM Roles](https://cloud.google.com/artifact-registry/docs/access-control)
- [Container Analysis API](https://cloud.google.com/artifact-registry/docs/analysis)
- [GCP Container Best Practices](https://cloud.google.com/architecture/best-practices-for-building-containers)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
