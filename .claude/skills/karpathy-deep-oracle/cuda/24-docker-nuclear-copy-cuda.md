# Docker Nuclear Copy CUDA - Brute Force Copying Strategy

**Production context**: From CUPTI investigation debugging and Docker build optimization patterns.

**Key insight**: Copy everything first, optimize later - guaranteed to work, then profile what's actually needed.

---

## Section 1: Nuclear Copy Explained (~100 lines)

### What is Nuclear Copy?

**Nuclear copy** is the brute-force Docker strategy of copying the entire CUDA toolkit directory from a builder stage to runtime, rather than selectively copying individual libraries.

```dockerfile
# Nuclear approach - copy EVERYTHING
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
# Build your app...

FROM nvidia/cuda:12.4.0-base-ubuntu22.04
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
```

**Why "nuclear"?**
- Overkill approach (like using a nuclear bomb when a scalpel would do)
- Guaranteed to include everything you need (and much you don't)
- Maximum redundancy, minimal thinking required
- Trade image size for certainty

From [Docker multi-stage builds documentation](https://docs.docker.com/build/building/multi-stage/) (accessed 2025-11-13):
> "You can selectively copy artifacts from one stage to another, leaving behind everything you don't want in the final image."

Nuclear copy does the opposite - it copies EVERYTHING, then optimizes later.

### Real-World Nuclear Copy Context

**CUPTI investigation scenario** (arr-coc-0-1, 2025-11-13):

After hours of debugging missing CUPTI libraries, frustration led to:
```dockerfile
# Original selective approach (FAILED)
COPY --from=builder /usr/local/cuda/extras/CUPTI /usr/local/cuda/extras/CUPTI
# Runtime error: libcupti.so.12: cannot open shared object file

# Nuclear approach (SUCCESS)
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnv* /usr/lib/x86_64-linux-gnu/
# Just works! (but adds 2-3GB)
```

**Why it worked**:
- CUPTI libraries were in `/usr/lib/`, not `/usr/local/cuda/extras/`
- Wildcards caught all CUDA/NVIDIA libraries
- No need to know exact dependency tree
- Immediate debugging progress

### When Nuclear Copy Makes Sense

**1. Debugging unknown dependencies**:
```dockerfile
# You're getting "library not found" errors
# Don't know which specific .so files are needed
# Nuclear copy = instant solution to identify what works
COPY --from=builder /usr/local/cuda /usr/local/cuda
```

**2. Quick prototyping**:
- Build MVP fast, optimize later
- Focus on functionality, not image size
- Iterate on code, not Dockerfile complexity

**3. Complex CUDA applications**:
- Multiple CUDA libraries (cuBLAS, cuFFT, cuSPARSE, CUPTI, NCCL)
- Custom compiled kernels with unknown dependencies
- Third-party libraries with opaque CUDA requirements

**4. Time-constrained debugging**:
```
Developer: "I need this working in 30 minutes for the demo"
Solution: Nuclear copy → test → document what's needed → optimize after demo
```

**5. Development environments**:
- Local Docker images (not pushing to registry)
- Fast rebuild iterations (BuildKit cache helps)
- Size doesn't matter as much as functionality

### When NOT to Use Nuclear Copy

**❌ Production deployments**:
- 2-3GB larger images = slower cold starts on Vertex AI
- Higher bandwidth costs for image pulls
- More attack surface (unnecessary binaries)

**❌ CI/CD pipelines**:
- Slower builds (more data to copy)
- Larger layer sizes = cache invalidation issues
- Registry storage costs

**❌ Public images**:
- Users downloading 5GB instead of 2GB
- Unprofessional (shows lack of optimization)

**✅ Use selective copying instead** (see Section 4).

### Size Impact of Nuclear Copy

**Typical CUDA toolkit contents**:
```
/usr/local/cuda/
├── bin/                    # 500MB (nvcc, profilers)
├── lib64/                  # 1.5GB (static + shared libraries)
├── include/                # 50MB (headers)
├── extras/                 # 100MB (CUPTI, samples)
├── nvvm/                   # 200MB (LLVM backend)
├── doc/                    # 50MB (documentation)
└── targets/                # 300MB (cross-compilation)
Total: ~2.7GB
```

**Runtime actually needs** (PyTorch example):
```
libcudart.so.12       # 500KB (CUDA runtime)
libcublas.so.12       # 150MB (BLAS)
libcublasLt.so.12     # 200MB (BLAS optimizations)
libcufft.so.12        # 100MB (FFT)
libcurand.so.12       # 50MB (RNG)
libcupti.so.12        # 20MB (profiling, optional)
Total: ~520MB
```

**Nuclear copy waste**: 2.7GB - 520MB = **2.18GB of unnecessary files** (81% waste).

### Nuclear Copy Philosophy

**Trade-off matrix**:
| Factor | Nuclear Copy | Selective Copy |
|--------|-------------|----------------|
| Build time (first) | Fast (simple Dockerfile) | Slow (research dependencies) |
| Build time (cached) | Fast (BuildKit cache) | Fast (BuildKit cache) |
| Image size | 5-6GB | 2-3GB |
| Debugging speed | Instant (everything present) | Slow (iterative testing) |
| Deployment speed | Slow (large image pull) | Fast (smaller image) |
| Production suitability | ❌ No | ✅ Yes |

**Best practice workflow**:
1. Start with nuclear copy (get it working)
2. Profile runtime dependencies (see Section 4)
3. Transition to selective copy (optimize)
4. Document minimal library set
5. Maintain both Dockerfiles (dev = nuclear, prod = selective)

From [Docker best practices](https://docs.docker.com/develop/dev-best-practices/) (accessed 2025-11-13):
> "Optimize for both the build-cache and final image size."

Nuclear copy optimizes for build-cache, selective copy for final image size.

---

## Section 2: Implementation Patterns (~150 lines)

### Basic Nuclear Copy Pattern

**Complete /usr/local/cuda copy**:
```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Build PyTorch from source with CUPTI support
WORKDIR /build
RUN apt-get update && apt-get install -y python3-dev python3-pip git
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /build/pytorch
RUN pip install -r requirements.txt
RUN USE_CUDA=1 USE_CUPTI=1 python setup.py install

# Runtime stage - nuclear copy
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy entire CUDA toolkit (2.7GB)
COPY --from=builder /usr/local/cuda /usr/local/cuda

# Copy all CUDA libraries from system lib (wildcards catch everything)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnv*.so* /usr/lib/x86_64-linux-gnu/

# Copy compiled PyTorch
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch \
                    /usr/local/lib/python3.10/site-packages/torch

# Environment setup
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
```

**What gets copied**:
- `/usr/local/cuda` - Entire CUDA toolkit (2.7GB)
- `libcu*.so*` - All libcublas, libcufft, libcurand, etc. (500MB)
- `libnv*.so*` - NVIDIA driver libraries, nvrtc, etc. (200MB)
- **Total**: ~3.4GB of CUDA files

### Wildcard Patterns Explained

**Understanding shell globs in COPY**:
```dockerfile
# Pattern 1: All shared libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu*.so* /dst/
# Matches:
# - libcublas.so.12
# - libcublas.so.12.4.2.65
# - libcufft.so.11
# - libcupti.so.12
# - libcurand.so.10
# etc.

# Pattern 2: Specific library versions
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu*.so.12* /dst/
# Matches only CUDA 12.x libraries

# Pattern 3: Multiple wildcards
COPY --from=builder /usr/local/cuda/**/lib*.so* /dst/
# Matches all .so files recursively (dangerous - very large!)
```

**Safe wildcard practices**:
```dockerfile
# ✅ GOOD: Specific directory, predictable results
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/

# ⚠️ CAREFUL: Recursive wildcard, unpredictable size
COPY --from=builder /usr/local/cuda/**/*.so* /usr/local/cuda/

# ❌ BAD: Too broad, copies everything
COPY --from=builder /usr/* /usr/
```

### ENV Configuration for Nuclear Copy

**Critical environment variables**:
```dockerfile
# PATH - for CUDA binaries (nvcc, nvidia-smi)
ENV PATH=/usr/local/cuda/bin:$PATH

# LD_LIBRARY_PATH - for shared library discovery
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CUPTI libraries (if using profiler)
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# CUDA installation path (some apps check this)
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# cuDNN location (if using deep learning)
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu
```

**Why these matter**:
- Without `PATH`: `nvcc --version` fails
- Without `LD_LIBRARY_PATH`: `libcublas.so.12: not found`
- Without `CUDA_HOME`: Some build scripts fail to detect CUDA

### Layering Strategy for Nuclear Copy

**Optimize Docker layer cache**:
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
# Layer 1: System packages (changes rarely)
RUN apt-get update && apt-get install -y python3-pip git

# Layer 2: Python dependencies (changes occasionally)
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Layer 3: Build app (changes frequently)
COPY . /build
WORKDIR /build
RUN python setup.py install

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Layer 1: Nuclear CUDA copy (changes never)
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnv* /usr/lib/x86_64-linux-gnu/

# Layer 2: Environment (changes rarely)
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Layer 3: Application code (changes frequently)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY app/ /app/
```

**Cache efficiency**:
- Nuclear CUDA copy layer cached unless CUDA version changes
- Application changes don't invalidate CUDA layer
- Rebuilds only copy changed application files

### Verification After Nuclear Copy

**Test CUDA libraries are present**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/

# Verification step (catches missing libraries early)
RUN ldconfig && \
    ldconfig -p | grep libcublas && \
    ldconfig -p | grep libcufft && \
    ldconfig -p | grep libcupti && \
    echo "All CUDA libraries found!"

# Test CUDA runtime
RUN python3 -c "import torch; assert torch.cuda.is_available()" || \
    (echo "CUDA not available!" && exit 1)
```

**What this catches**:
- Missing library symlinks
- Incorrect `LD_LIBRARY_PATH`
- Broken CUDA installation
- PyTorch CUDA detection failures

### Multi-Architecture Nuclear Copy

**Copy CUDA for multiple platforms**:
```dockerfile
FROM --platform=$BUILDPLATFORM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
ARG TARGETPLATFORM
ARG TARGETARCH

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        export CUDA_ARCH="x86_64-linux-gnu"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        export CUDA_ARCH="aarch64-linux-gnu"; \
    fi

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
ARG TARGETARCH
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/${TARGETARCH}-linux-gnu/libcu* /usr/lib/${TARGETARCH}-linux-gnu/
```

**Build for multiple platforms**:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ml-app:latest .
```

### Size Reporting

**Document nuclear copy overhead**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/

# Report size (for optimization planning)
RUN du -sh /usr/local/cuda && \
    du -sh /usr/lib/x86_64-linux-gnu/libcu* && \
    echo "Total CUDA files: $(du -sh /usr/local/cuda /usr/lib/x86_64-linux-gnu/libcu* | awk '{sum+=$1} END {print sum}')MB"
```

**Output example**:
```
2.7G    /usr/local/cuda
520M    /usr/lib/x86_64-linux-gnu/libcu*
Total CUDA files: 3220MB
```

---

## Section 3: When Nuclear Copy Makes Sense (~100 lines)

### Use Case 1: Debugging Unknown Dependencies

**Scenario**: PyTorch compiled from source, runtime crashes with library errors.

**Selective copy attempt** (FAILED):
```dockerfile
# Try #1: Copy CUPTI from extras directory
COPY --from=builder /usr/local/cuda/extras/CUPTI /usr/local/cuda/extras/CUPTI
# Error: libcupti.so.12: cannot open shared object file

# Try #2: Copy from lib64
COPY --from=builder /usr/local/cuda/lib64/libcupti.so* /usr/local/cuda/lib64/
# Error: still missing (CUPTI not in lib64!)

# Try #3: Search and copy
RUN find /usr -name "libcupti.so*"
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcupti.so* /usr/lib/x86_64-linux-gnu/
# Finally works! But wasted 2 hours...
```

**Nuclear copy solution** (SUCCESS):
```dockerfile
# Copy everything, identify what's needed later
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnv* /usr/lib/x86_64-linux-gnu/

# Works immediately! Profile later to identify minimal set
```

**Debugging workflow**:
1. Nuclear copy → Works!
2. `docker exec -it container bash`
3. `ldd /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so`
4. Document actual dependencies
5. Create selective copy Dockerfile
6. Test selective version
7. Switch to selective for production

**Time saved**: 2 hours debugging → 10 minutes nuclear copy + 30 minutes optimization later.

### Use Case 2: Quick Prototyping

**MVP development pattern**:
```dockerfile
# Day 1-3: MVP with nuclear copy
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
COPY . /app
RUN cd /app && python setup.py install

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# Image size: 5.2GB (don't care yet, functionality > optimization)

# Day 4-5: Optimize after MVP validated
# Profile dependencies, create selective copy version
# Image size: 2.8GB (46% reduction)
```

**Why nuclear copy first**:
- Focus on code functionality, not Docker optimization
- Avoid premature optimization
- Iterate faster (simple Dockerfile)
- Defer size optimization until product validated

### Use Case 3: Complex CUDA Applications

**Multi-library CUDA app**:
```python
# app.py - uses many CUDA libraries
import torch                    # cuBLAS, cuFFT
import torchvision             # cuDNN
from apex import amp           # NCCL for multi-GPU
from torch.profiler import profile  # CUPTI
import cupy                    # cuSPARSE, cuSOLVER
```

**Selective copy nightmare**:
```dockerfile
# Need to copy: cuBLAS, cuFFT, cuDNN, NCCL, CUPTI, cuSPARSE, cuSOLVER
# Each library has dependencies and specific locations
# Risk of missing subtle dependencies
```

**Nuclear copy simplicity**:
```dockerfile
# Just copy everything CUDA-related
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnccl* /usr/lib/x86_64-linux-gnu/
# Guaranteed to work, optimize later
```

### Use Case 4: Time-Constrained Debugging

**Production incident scenario**:
```
11:00 AM: "CUDA profiler broken in production!"
11:05 AM: Try selective CUPTI copy → still broken
11:15 AM: Research CUPTI location → conflicting docs
11:30 AM: Try apt package install → version mismatch
11:45 AM: Nuclear copy → WORKS!
12:00 PM: Deploy fix, incident resolved
```

**Nuclear copy = incident resolution tool**:
- Get production working ASAP
- Profile and optimize in post-incident review
- Document findings for next time

### Use Case 5: Development Environments

**Local development Docker image**:
```dockerfile
# Dockerfile.dev (local only, never pushed)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Nuclear copy for maximum compatibility
# Size doesn't matter (local development)
COPY --from=prebuilt-builder /usr/local/cuda /usr/local/cuda
COPY --from=prebuilt-builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/

# Install debugging tools (wouldn't include in production)
RUN apt-get update && apt-get install -y \
    cuda-gdb \
    nsight-systems \
    nsight-compute \
    vim \
    gdb \
    strace
```

**Why nuclear copy for dev**:
- Developer productivity > image size
- All CUDA tools available
- No "missing library" interruptions
- BuildKit cache makes rebuilds fast

### Anti-Pattern: Nuclear Copy in CI/CD

**❌ DON'T DO THIS**:
```yaml
# GitHub Actions workflow
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: |
          docker build -t ml-app:latest .
          # Dockerfile uses nuclear copy
          # Image size: 5.2GB
          docker push ghcr.io/org/ml-app:latest
          # Push takes 10 minutes (5.2GB upload)
```

**✅ BETTER**:
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build optimized Docker image
        run: |
          docker build -t ml-app:latest -f Dockerfile.prod .
          # Dockerfile.prod uses selective copy
          # Image size: 2.8GB
          docker push ghcr.io/org/ml-app:latest
          # Push takes 4 minutes (46% faster)
```

**CI/CD best practices**:
- Use nuclear copy for local dev
- Use selective copy for CI/CD
- Maintain both Dockerfiles
- Document migration path (nuclear → selective)

---

## Section 4: Optimization After Nuclear (~50 lines)

### Profiling Runtime Dependencies

**Step 1: Run nuclear copy container**:
```bash
docker run --gpus all -it --rm ml-app:nuclear bash
```

**Step 2: Identify loaded libraries**:
```bash
# Method 1: ldd on main executable
ldd /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so

# Output shows actual dependencies:
# libcublas.so.12 => /usr/lib/x86_64-linux-gnu/libcublas.so.12
# libcublasLt.so.12 => /usr/lib/x86_64-linux-gnu/libcublasLt.so.12
# libcudart.so.12 => /usr/lib/x86_64-linux-gnu/libcudart.so.12
# libcufft.so.11 => /usr/lib/x86_64-linux-gnu/libcufft.so.11
# libcupti.so.12 => /usr/lib/x86_64-linux-gnu/libcupti.so.12
```

**Method 2: strace for runtime loading**:
```bash
strace -f -e openat python3 -c "import torch; torch.cuda.is_available()" 2>&1 | grep "\.so"

# Shows which libraries are actually opened:
# openat(AT_FDCWD, "/usr/lib/x86_64-linux-gnu/libcublas.so.12", O_RDONLY) = 3
# openat(AT_FDCWD, "/usr/lib/x86_64-linux-gnu/libcupti.so.12", O_RDONLY) = 4
```

**Method 3: Profile with full test suite**:
```bash
# Run your app's test suite
python3 -m pytest tests/

# Capture library usage
strace -f -e openat python3 -m pytest tests/ 2>&1 | \
    grep "\.so" | \
    grep libcu | \
    sort -u > required_libs.txt
```

### Iterative Removal Process

**Step 1: Create baseline selective Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy only libraries found in profiling
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcublas.so.12* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcublasLt.so.12* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcudart.so.12* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcufft.so.11* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcurand.so.10* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/lib/x86_64-linux-gnu/

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

**Step 2: Build and test**:
```bash
docker build -t ml-app:selective -f Dockerfile.selective .
docker run --gpus all ml-app:selective python3 -c "import torch; assert torch.cuda.is_available()"
```

**Step 3: Iteratively remove libraries**:
```bash
# Try removing libcurand (if profiling showed it's unused)
# Edit Dockerfile, remove libcurand line
docker build -t ml-app:selective-v2 .
docker run --gpus all ml-app:selective-v2 python3 -m pytest
# If tests pass, libcurand not needed!
```

### Document Final Minimal Set

**Create optimization documentation**:
```markdown
# CUDA Library Optimization

## Nuclear Copy (Baseline)
- Size: 5.2GB
- Libraries: All CUDA libraries (~3GB)

## Selective Copy (Optimized)
- Size: 2.8GB (46% reduction)
- Libraries included:
  - libcublas.so.12 (150MB) - Required by PyTorch
  - libcublasLt.so.12 (200MB) - Required by PyTorch
  - libcudart.so.12 (500KB) - CUDA runtime
  - libcufft.so.11 (100MB) - Required by torchvision
  - libcupti.so.12 (20MB) - Required by torch.profiler

## Libraries removed (not used by app):
  - libcurand.so.10 (50MB)
  - libcusparse.so.12 (100MB)
  - libcusolver.so.11 (150MB)
  - /usr/local/cuda/bin/* (500MB)
  - /usr/local/cuda/doc/* (50MB)

## Testing:
```bash
docker run --gpus all ml-app:selective python3 -m pytest tests/
```
```

### Transition Dockerfile Pattern

**Maintain both versions**:
```
project/
├── Dockerfile              # Production (selective copy)
├── Dockerfile.dev          # Development (nuclear copy)
└── docker/
    ├── selective.dockerfile
    └── nuclear.dockerfile
```

**Selective Dockerfile** (production):
```dockerfile
# Optimized for size and deployment speed
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcublas.so.12* /usr/lib/x86_64-linux-gnu/
# ... selective libraries only
```

**Nuclear Dockerfile** (development):
```dockerfile
# Optimized for debugging and compatibility
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcu* /usr/lib/x86_64-linux-gnu/
```

**Build scripts**:
```bash
# build.sh
#!/bin/bash
if [ "$ENV" = "production" ]; then
    docker build -f Dockerfile -t ml-app:prod .
else
    docker build -f Dockerfile.dev -t ml-app:dev .
fi
```

---

## Sources

**Source Documents**:
- [cuda/10-docker-multistage-ml-optimization.md](cuda/10-docker-multistage-ml-optimization.md) - Multi-stage build fundamentals
- [cuda/13-nvidia-container-cuda-packaging.md](cuda/13-nvidia-container-cuda-packaging.md) - CUDA package structure and locations

**Web Research** (accessed 2025-11-13):
- [Multi-stage builds - Docker Docs](https://docs.docker.com/build/building/multi-stage/) - Official multi-stage build documentation
- [Docker best practices - Docker Docs](https://docs.docker.com/develop/dev-best-practices/) - Build optimization guidelines
- [Understanding the Differences Between ADD and COPY - Docker Blog](https://www.docker.com/blog/docker-best-practices-understanding-the-differences-between-add-and-copy-instructions-in-dockerfiles/) - COPY instruction best practices
- [Storage drivers - Docker Docs](https://docs.docker.com/engine/storage/drivers/) - Copy-on-write strategy explanation

**Production Context**:
- arr-coc-0-1 CUPTI investigation (2025-11-13) - Real-world debugging scenario where nuclear copy resolved missing library issues, then optimized to selective copy after profiling

**Additional References**:
- [NVIDIA CUDA filesystem layout](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) - Understanding where CUDA libraries live
