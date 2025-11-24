# Using CUDA Devel Images as Runtime: When Simplicity Trumps Size

**Production knowledge from CUPTI investigation**: Sometimes the simplest solution is to skip multi-stage optimization and use devel images directly. Understanding when this makes sense versus when it's wasteful is critical for Docker strategy decisions.

**Key tradeoff**: 3GB larger images for zero complexity. Good for development, questionable for production.

---

## Section 1: Devel vs Runtime Image Comparison (~100 lines)

### The Three NVIDIA Container Image Types

From [NVIDIA Docker Hub](https://hub.docker.com/r/nvidia/cuda) and [Stack Overflow discussion](https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container) (accessed 2025-11-13):

NVIDIA provides three container image flavors, each with different purposes:

#### Runtime Images: Production-Ready Inference

**Package**: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`

**What's included**:
- CUDA runtime libraries (shared `.so` files)
- libcublas, libcufft, libcurand, libcusparse, libcusolver
- libcudart (CUDA runtime)
- cuDNN runtime libraries (in `-cudnn8-runtime` variant)

**What's NOT included**:
- No compiler toolchain (nvcc)
- No development headers (`.h` files)
- No static libraries (`.a` files)
- No CUPTI profiling libraries
- No debugging tools

**Size**: ~1.5-2.5 GB (runtime), ~3-4 GB (cudnn-runtime)

**Use case**: Deploying pre-compiled applications. Perfect for inference workloads where you've already built your binaries.

**Example**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Your pre-built PyTorch application
COPY ./my_model /app/
CMD ["python", "/app/inference.py"]
```

**What works**:
- PyTorch inference (uses libcublas, libcudnn)
- TensorFlow serving
- Any pre-compiled CUDA application

**What fails**:
- Compiling CUDA kernels (no nvcc)
- Building from source (no headers)
- Profiling with CUPTI (library missing)

#### Devel Images: Full Development Stack

**Package**: `nvidia/cuda:12.4.0-devel-ubuntu22.04`

**What's included**:
- Everything from runtime
- CUDA compiler toolchain (nvcc, ptxas, fatbinary)
- Development headers (`cuda.h`, `cublas_v2.h`, `cupti.h`, etc.)
- Static libraries (`.a` files)
- Debugging tools (cuda-gdb)
- Profiling tools (CUPTI, NSight)
- CMake integration files

**Size**: ~4.5-6 GB (devel), ~7-9 GB (cudnn-devel)

**Use case**: Compiling CUDA applications from source. Standard choice for multi-stage Docker builds (build stage).

**Example multi-stage build** (typical approach):
```dockerfile
# Build stage: compile with devel image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
WORKDIR /build
COPY ./src /build/src
RUN nvcc -o app src/kernel.cu

# Runtime stage: deploy with runtime image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /build/app /app/
CMD ["/app/app"]
```

**Size reduction**: 4.5 GB devel → 1.5 GB runtime (3 GB saved)

### Direct Comparison Table

| Feature | Runtime | Devel | Size Difference |
|---------|---------|-------|-----------------|
| Base image size | ~1.5-2.5 GB | ~4.5-6 GB | +3 GB |
| Shared libraries (.so) | ✓ | ✓ | Same |
| Headers (.h files) | ✗ | ✓ | +500 MB |
| nvcc compiler | ✗ | ✓ | +200 MB |
| Static libraries (.a) | ✗ | ✓ | +1 GB |
| CUPTI profiling | ✗ | ✓ | +100 MB |
| NSight tools | ✗ | ✓ | +300 MB |
| cuda-gdb debugger | ✗ | ✓ | +50 MB |
| CMake files | ✗ | ✓ | +10 MB |
| Documentation | ✗ | ✓ | +500 MB |

**Key insight**: Devel contains ~3 GB of development tools that runtime applications never use.

### What "Using Devel as Runtime" Means

**Standard approach** (multi-stage):
```dockerfile
# Stage 1: Build with devel (4.5 GB)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN compile_my_app

# Stage 2: Deploy with runtime (1.5 GB) ← Final image is small
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /app /app
```

**"Devel as runtime" approach** (single-stage):
```dockerfile
# Single stage: Use devel directly (4.5 GB) ← Final image is large
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
RUN compile_my_app
CMD ["/app/my_app"]
```

**Difference**: Skip the second stage entirely. Your final deployed image includes all the development tools, even though they're never used at runtime.

---

## Section 2: When to Use Devel as Runtime (~100 lines)

### Development Environments: Speed Over Size

**Use case**: Local development, quick iteration

**Why it makes sense**:
- Build once, use everywhere (no multi-stage complexity)
- Full debugging tools available (cuda-gdb, NSight)
- Can recompile on the fly if needed
- CUPTI always present for profiling

**Example workflow**:
```dockerfile
# Development Dockerfile (devel-as-runtime)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app
COPY . /app

# Development dependencies
RUN pip install -e . --no-cache-dir

# Full debugging and profiling tools available
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "train.py"]
```

**Benefits for development**:
- No missing library surprises (everything is there)
- Can attach cuda-gdb to running process
- CUPTI profiling works out of the box
- Faster Docker builds (no multi-stage copying)

**Size doesn't matter because**:
- Local development (not deploying to cloud)
- Build once, run many times
- Developer time > disk space

### Profiling and Debugging Workflows: CUPTI Always Available

**Use case**: Performance investigation, troubleshooting production issues

**Why it makes sense**:
- CUPTI libraries included by default
- No manual extraction needed (see [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md))
- Can enable/disable profiling without rebuilding image

**Example debugging session**:
```bash
# Build with devel image
docker build -t myapp:debug --target devel .

# Run with profiling enabled
docker run --gpus all \
  -e TORCH_PROFILER_ENABLE=1 \
  myapp:debug

# CUPTI works immediately (no libcupti.so missing errors)
```

**From arr-coc-0-1 CUPTI investigation context**:

We spent hours debugging missing CUPTI libraries in runtime images. If we had used devel as runtime during investigation:
- No `libcupti.so: cannot open shared object file` errors
- No multi-stage CUPTI extraction complexity
- Could have focused on actual profiling, not infrastructure

**When debugging is priority**:
- Use devel image directly
- Get CUPTI working first
- Optimize for size later (after confirming profiling works)

### Prototyping: Don't Optimize Yet

**Use case**: MVP development, proof-of-concept, early stage projects

**Why it makes sense**:
- Unknown requirements (might need CUPTI later)
- Fast iteration (no multi-stage refactoring)
- Simplicity (one Dockerfile, one image)

**Example prototyping Dockerfile**:
```dockerfile
# Prototype: devel-as-runtime (simplest approach)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install everything we might need
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    vim

# Copy prototype code
COPY . /workspace
WORKDIR /workspace

# Install Python deps
RUN pip install torch torchvision transformers

# Run prototype
CMD ["python", "prototype.py"]
```

**Optimization can wait**:
- Focus on getting it working first
- Worry about image size after proving concept
- Easier to refactor to multi-stage later than debug missing libs early

**Golden rule for prototyping**: "Make it work, make it right, make it fast" - use devel first, optimize later.

### Single-Stage Builds: When Complexity Isn't Worth It

**Use case**: Simple applications, minimal dependencies

**Why it makes sense**:
- Small codebases don't justify multi-stage complexity
- Maintenance burden of two stages
- 3 GB isn't significant for some contexts

**Example simple application**:
```dockerfile
# Simple CUDA app: devel-as-runtime is fine
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Tiny codebase (one .cu file)
COPY kernel.cu /app/
WORKDIR /app

# Compile
RUN nvcc -o kernel kernel.cu

# Run
CMD ["./kernel"]
```

**When single-stage makes sense**:
- Application code < 100 lines
- No external dependencies
- Not deploying at scale (single instance)
- Internal tools (not customer-facing)

**Cost-benefit analysis**:
- Multi-stage adds 10-20 lines to Dockerfile
- Saves 3 GB per image
- If you only deploy 1-2 instances → not worth the complexity
- If you deploy 100+ instances → definitely worth it

### Educational and Training Use Cases

**Use case**: Teaching CUDA programming, workshops, tutorials

**Why it makes sense**:
- Students need full development environment
- No "why doesn't this work?" debugging sessions
- Everything just works out of the box

**Example workshop Dockerfile**:
```dockerfile
# CUDA Workshop: Use devel directly
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Pre-install common tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    gdb \
    cuda-gdb

# Sample CUDA programs
COPY samples/ /workspace/samples/

# Drop students into shell with everything available
CMD ["/bin/bash"]
```

**Educational benefits**:
- Students can compile and run immediately
- No "missing library" frustration
- Focus on learning CUDA, not Docker optimization

---

## Section 3: Production Tradeoffs: When NOT to Use Devel (~100 lines)

### Cloud Deployment Costs: 3GB Adds Up Fast

**Cost analysis**: At scale, image size directly impacts money and time.

#### Storage Costs

**Container registry pricing** (approximate):
- Docker Hub: $0.10/GB/month
- Google Container Registry: $0.026/GB/month
- AWS ECR: $0.10/GB/month

**Cost impact**:
```
Single image:
- Runtime: 1.5 GB × $0.10 = $0.15/month
- Devel:   4.5 GB × $0.10 = $0.45/month
- Waste:   3 GB × $0.10 = $0.30/month (2x cost)

At scale (100 images):
- Runtime: $15/month
- Devel:   $45/month
- Waste:   $30/month ($360/year wasted)
```

**For large organizations** (1000+ images):
- Wasted storage: 3 TB
- Annual waste: $3,600+

#### Network Transfer Costs

**Image pull bandwidth** (cloud egress pricing):
- AWS: $0.09/GB egress
- GCP: $0.12/GB egress
- Azure: $0.087/GB egress

**Cost per deployment**:
```
Single deployment:
- Runtime: 1.5 GB × $0.09 = $0.135
- Devel:   4.5 GB × $0.09 = $0.405
- Waste:   3 GB × $0.09 = $0.27 (3x data transfer)

Daily deployments (10/day):
- Runtime: $1.35/day = $493/year
- Devel:   $4.05/day = $1,478/year
- Waste:   $2.70/day = $985/year wasted
```

**For high-velocity deployments**:
- 100 deploys/day → $98,500/year wasted on bandwidth alone

#### Deployment Time Impact

**Image pull duration** (1 Gbps network):
- Runtime: 1.5 GB = 12 seconds
- Devel:   4.5 GB = 36 seconds
- Waste:   3 GB = 24 seconds slower (3x longer)

**Cold start latency**:
```
Kubernetes pod scheduling:
- Runtime image: ~15-20 seconds total
- Devel image:   ~40-50 seconds total
- Impact: 2-3x slower cold starts
```

**Vertex AI training jobs** (arr-coc-0-1 context):
```
Image pull for training:
- Runtime: 30 seconds (1.5 GB on Vertex AI network)
- Devel:   90 seconds (4.5 GB)
- Per job: 60 seconds wasted
- 100 jobs/day: 100 minutes/day wasted = 600 hours/year
```

### Google Cloud Build: Upload Time Matters

**From arr-coc-0-1 experience**: We deploy to Google Cloud Build for Vertex AI training.

**Cloud Build source upload**:
```
Local → Cloud Build upload time:
- Runtime Dockerfile produces 1.5 GB image
- Devel Dockerfile produces 4.5 GB image
- Upload time difference: 2-5 minutes (depends on connection)
```

**Build time impact**:
```
Cloud Build process:
1. Upload source context (Dockerfile, code)
2. Build image layers
3. Push to Artifact Registry
4. Pull to Vertex AI

Using devel as runtime:
- Step 3 (push): 3 GB extra = +2-3 minutes
- Step 4 (pull): 3 GB extra = +1-2 minutes
- Total: +3-5 minutes per build
```

**For frequent builds** (arr-coc-0-1 pattern):
- 10 builds/day during development
- 30-50 minutes/day wasted on uploads/downloads
- Developer waiting time = money

### Vertex AI Quotas: Image Size Limits

**Vertex AI Custom Training constraints**:
- Maximum image size: 10 GB (compressed)
- Typical runtime image: 1.5 GB (compressed: ~800 MB)
- Typical devel image: 4.5 GB (compressed: ~2.5 GB)

**Quota considerations**:
```
Adding large model weights:
- Runtime base + 5 GB model = 6.5 GB (within limit)
- Devel base + 5 GB model = 9.5 GB (close to limit!)
- Devel base + 7 GB model = 11.5 GB (exceeds limit!)
```

**From arr-coc-0-1 context**: We need to bundle model weights in training images. Using devel base would leave less room for actual model data.

### Security Surface Area: More Code = More Vulnerabilities

**Container scanning results** (example):

**Runtime image scan**:
```
Critical vulnerabilities: 2
High vulnerabilities: 8
Medium vulnerabilities: 15
Total packages: 200
```

**Devel image scan**:
```
Critical vulnerabilities: 5
High vulnerabilities: 20
Medium vulnerabilities: 45
Total packages: 600
```

**Why more vulnerabilities**:
- nvcc compiler (complex codebase)
- Static libraries (outdated versions)
- Documentation packages (unmaintained)
- Development tools (often not security-hardened)

**Production security principle**: Minimize attack surface by only including runtime dependencies.

### Cache Invalidation: Larger Images = More Cache Misses

**Docker layer caching**:

**Multi-stage runtime** (optimized):
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Small base layer (1.5 GB)
# Cache hit probability: HIGH (base rarely changes)

COPY --from=builder /app /app
# Application layer (~100 MB)
# Cache invalidated only when code changes
```

**Devel as runtime** (large):
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Large base layer (4.5 GB)
# Cache hit probability: MEDIUM (larger images = more frequent rebuilds)

RUN compile_app
# Cache invalidated on ANY code change
# Must re-download 4.5 GB base if cache expires
```

**Cache efficiency**:
- Smaller base images stay in cache longer
- Larger images pushed out of local cache faster
- 3 GB difference = 3x more likely to need re-pull

---

## Section 4: Decision Framework and Best Practices (~50 lines)

### When to Use Devel as Runtime: Decision Tree

```
Should I use devel-as-runtime?

YES if:
├─ Local development only (not deploying to cloud)
├─ Prototyping/MVP (optimization comes later)
├─ Debugging session (temporarily need full tools)
├─ Educational/training (students need everything)
├─ Single-instance deployment (scale doesn't matter)
└─ CUPTI investigation (profiling is priority)

NO if:
├─ Production deployment (optimize for size)
├─ High-frequency deployments (network costs matter)
├─ Cloud-native (Kubernetes, serverless)
├─ Large scale (100+ instances)
├─ Customer-facing (cold start latency matters)
└─ Security-critical (minimize attack surface)
```

### Hybrid Strategy: Devel for Dev, Runtime for Prod

**Best of both worlds**: Use devel locally, runtime in production.

**Example multi-target Dockerfile**:
```dockerfile
# Development target: devel-as-runtime
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS development
WORKDIR /app
COPY . /app
RUN pip install -e . --no-cache-dir
CMD ["python", "train.py"]

# Production target: optimized runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS production
COPY --from=development /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=development /app /app
CMD ["python", "train.py"]
```

**Usage**:
```bash
# Local development
docker build --target development -t myapp:dev .

# Production deployment
docker build --target production -t myapp:prod .
```

**Benefits**:
- Developers get full tools (devel)
- Production gets optimized images (runtime)
- Single Dockerfile maintains both
- Easy to switch between modes

### Temporary Debugging Pattern

**Use case**: Production issue requires debugging, but image is optimized.

**Strategy**: Temporarily switch to devel, debug, then switch back.

**Example workflow**:
```bash
# 1. Production runs optimized runtime image
docker run myapp:prod  # 1.5 GB

# 2. Issue occurs, need CUPTI profiling
# Rebuild with devel temporarily
docker build --build-arg BASE=devel -t myapp:debug .
docker run --gpus all myapp:debug  # 4.5 GB, but has CUPTI

# 3. Debug, find issue, fix code

# 4. Return to optimized production image
docker build --build-arg BASE=runtime -t myapp:prod .
docker run myapp:prod  # Back to 1.5 GB
```

**Parameterized Dockerfile**:
```dockerfile
ARG BASE=runtime
FROM nvidia/cuda:12.4.0-${BASE}-ubuntu22.04

# Same build logic works for both
COPY . /app
RUN compile_app
```

### arr-coc-0-1 Context: Why We Chose Multi-Stage

**Our decision**: Use multi-stage builds with runtime images for production.

**Reasoning**:
1. **Vertex AI costs**: Frequent training job launches (10-20/day during development)
2. **Cloud Build time**: Every minute of upload/download costs developer time
3. **Image size quotas**: Need room for model weights (~5 GB)
4. **CUPTI solution**: Extract libcupti.so in multi-stage (20 MB, not 3 GB)

**Our approach**:
```dockerfile
# Build stage: devel image (4.5 GB)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN pip install torch --no-binary torch  # Compile with CUPTI support

# Runtime stage: optimized (1.52 GB)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /torch

# Minimal CUPTI extraction (20 MB, not 3 GB)
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    rm -rf /var/lib/apt/lists/*
```

**Result**:
- CUPTI works (profiling available)
- Image size: 1.52 GB (vs 4.5 GB with devel-as-runtime)
- Savings: 3 GB per image × 100 builds = 300 GB bandwidth saved
- Build time: 2-3 minutes faster per build

**When we DO use devel**: Local development on MacBook (not deploying to cloud).

---

## Sources

**Source Documents**:
- [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md) - Devel vs runtime package comparison
- [cuda/10-docker-multistage-ml-optimization.md](10-docker-multistage-ml-optimization.md) - Multi-stage build patterns
- arr-coc-0-1/training/CUPTI_INVESTIGATION_STUDY.md - CUPTI investigation context

**Web Research**:
- [Stack Overflow: devel vs runtime tags](https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container) (accessed 2025-11-13) - Community explanation of image types
- [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda) (accessed 2025-11-13) - Official container image documentation
- [NVIDIA Developer Forums: CUDA best practices](https://forums.developer.nvidia.com/t/cuda-best-practices-development-to-production-pipeline/185856) (accessed 2025-11-13) - Production deployment guidance

**Additional References**:
- Docker multi-stage build best practices: https://docs.docker.com/build/building/multi-stage/
- Cloud provider egress pricing: AWS/GCP/Azure documentation
- Vertex AI image size limits: https://cloud.google.com/vertex-ai/docs/training/containers-overview
