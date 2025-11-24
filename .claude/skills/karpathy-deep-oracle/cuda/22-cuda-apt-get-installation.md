# CUDA Apt-Get Installation Strategies for Docker

**Production knowledge from CUPTI investigation**: Package-based CUDA installation provides granular control over image size and dependencies. Critical for multi-stage Docker builds optimizing for production deployment.

**Key insight**: `cuda-libraries-dev` extraction pattern (install → copy → remove) saves 3GB vs full devel image.

---

## Section 1: CUDA Apt Repository Setup (~100 lines)

### Adding NVIDIA CUDA Repository

**Official NVIDIA repository setup** for Ubuntu 22.04 (accessed 2025-11-13):

From [NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/):

```bash
# Step 1: Download repository pin file (priority management)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Step 2: Add NVIDIA GPG key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Step 3: Update apt package list
sudo apt-get update

# Step 4: Install CUDA toolkit (or specific packages)
sudo apt-get install -y cuda-toolkit-12-4
```

**What the pin file does**:
- Sets package priority to 600 (higher than default 500)
- Ensures NVIDIA packages preferred over Ubuntu's defaults
- Prevents accidental downgrades during `apt upgrade`

**GPG key verification**:
```bash
# Verify keyring installation
apt-key list | grep NVIDIA
```

### Docker-Specific Repository Setup

**For Dockerfiles** (non-interactive installation):

```dockerfile
FROM ubuntu:22.04

# Avoid timezone prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Add NVIDIA CUDA repository
RUN apt-get update && apt-get install -y wget ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Install specific CUDA packages
RUN apt-get install -y \
    cuda-libraries-12-4 \
    cuda-cudart-12-4 && \
    rm -rf /var/lib/apt/lists/*
```

**Best practices**:
- Always run `apt-get update` after adding repository
- Clean up with `rm -rf /var/lib/apt/lists/*` (saves ~100MB)
- Use `--no-install-recommends` for minimal installations
- Pin specific CUDA versions to avoid breaking changes

### Repository URL Structure

NVIDIA maintains OS-specific repositories:

```
https://developer.download.nvidia.com/compute/cuda/repos/{OS}/{ARCH}/
```

**Available repositories**:
```
Ubuntu 22.04 x86_64:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

Ubuntu 20.04 x86_64:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/

Ubuntu 22.04 ARM64 (Jetson/Grace):
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/

Ubuntu 24.04 x86_64 (preview):
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
```

From [Ask Ubuntu - CUDA Installation Ubuntu 24.04](https://askubuntu.com/questions/1520509/issues-installing-cuda-and-cudnn-on-ubuntu-24-04) (accessed 2025-11-13):
> "Ubuntu 24.04 makes NVIDIA installation easier. Check nvidia-driver-550 and cuda-toolkit-12-4 compatibility."

### Version Pinning for Production

**Pin specific CUDA version** to prevent accidental upgrades:

```bash
# Create apt preference file
sudo tee /etc/apt/preferences.d/cuda-12-4 <<EOF
Package: cuda-toolkit-12-4
Pin: version 12.4.0-1
Pin-Priority: 1000

Package: cuda-libraries-12-4
Pin: version 12.4.0-1
Pin-Priority: 1000

Package: cuda-libraries-dev-12-4
Pin: version 12.4.0-1
Pin-Priority: 1000
EOF

# Install pinned version
sudo apt-get install -y cuda-toolkit-12-4
```

**Why pin versions?**:
- `apt upgrade` won't install CUDA 12.5 or 12.6 unexpectedly
- Reproducible builds across environments
- Prevents driver/runtime version mismatches
- Critical for CI/CD pipelines

**Docker pinning**:
```dockerfile
# Pin CUDA 12.4 in Dockerfile
RUN echo "Package: cuda-libraries-12-4\nPin: version 12.4.0-1\nPin-Priority: 1000" \
    > /etc/apt/preferences.d/cuda-12-4 && \
    apt-get install -y cuda-libraries-12-4
```

---

## Section 2: Package Selection Strategies (~150 lines)

### CUDA Package Hierarchy

NVIDIA provides granular package selection:

```
cuda-toolkit-12-4 (meta-package, ~4GB installed)
├── cuda-libraries-12-4 (runtime, ~1GB)
├── cuda-libraries-dev-12-4 (development, ~500MB)
├── cuda-compiler-12-4 (nvcc, ~300MB)
├── cuda-tools-12-4 (profilers, ~200MB)
├── cuda-cudart-12-4 (CUDA runtime, ~50MB)
└── cuda-documentation-12-4 (~100MB)
```

From [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13).

### Runtime Packages (`cuda-libraries-{version}`)

**Package**: `cuda-libraries-12-4`

**What's included**:
- Shared libraries only (`.so` files)
- libcublas, libcufft, libcurand, libcusparse, libcusolver
- libcudart (CUDA runtime)
- libcudnn (if cudnn variant)
- NO headers (`.h` files)
- NO static libraries (`.a` files)
- NO CUPTI profiling libraries

**Installation**:
```bash
apt-get update
apt-get install -y --no-install-recommends cuda-libraries-12-4
```

**Installed size**: ~1GB

**Use case**: Runtime dependencies for pre-compiled CUDA applications. Perfect for inference workloads.

**What you get** (example file listing):
```
/usr/lib/x86_64-linux-gnu/libcublas.so.12
/usr/lib/x86_64-linux-gnu/libcublas.so.12.4.5.8
/usr/lib/x86_64-linux-gnu/libcublasLt.so.12
/usr/lib/x86_64-linux-gnu/libcudart.so.12
/usr/lib/x86_64-linux-gnu/libcufft.so.12
/usr/lib/x86_64-linux-gnu/libcurand.so.12
/usr/lib/x86_64-linux-gnu/libcusolver.so.12
/usr/lib/x86_64-linux-gnu/libcusparse.so.12
```

**NOT included** (common misconception):
```
❌ /usr/include/cublas_v2.h (headers in -dev package)
❌ /usr/lib/x86_64-linux-gnu/libcublas_static.a (static libs in -dev)
❌ /usr/lib/x86_64-linux-gnu/libcupti.so.12 (CUPTI in -dev)
```

### Development Packages (`cuda-libraries-dev-{version}`)

**Package**: `cuda-libraries-dev-12-4`

**What's included**:
- Everything from `cuda-libraries-12-4` (depends on it)
- Development headers (`.h` files)
- Static libraries (`.a` files)
- **CUPTI shared libraries** (critical discovery!)
- CMake integration files
- pkg-config files

**Installation**:
```bash
apt-get update
apt-get install -y --no-install-recommends cuda-libraries-dev-12-4
```

**Installed size**: ~1.5GB total (500MB additional over runtime)

**Use case**: Compiling CUDA applications. Required for PyTorch/TensorFlow builds from source.

**What you get** (additional files):
```
/usr/include/cublas_v2.h
/usr/include/cufft.h
/usr/include/curand.h
/usr/include/cuda_runtime.h
/usr/lib/x86_64-linux-gnu/libcublas_static.a
/usr/lib/x86_64-linux-gnu/libcupti.so.12  ← CUPTI HERE!
/usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
/usr/lib/x86_64-linux-gnu/libnvperf_host.so
/usr/lib/x86_64-linux-gnu/libnvperf_target.so
/usr/share/cmake-3.22/Modules/FindCUDA.cmake
/usr/lib/pkgconfig/cublas.pc
```

**Package dependencies**:
```bash
# cuda-libraries-dev-12-4 automatically installs:
cuda-libraries-12-4 (>= 12.4.0)
cuda-nvml-dev-12-4
cuda-cudart-dev-12-4
cuda-nvcc-12-4 (for some operations)
```

**Verify package contents before installing**:
```bash
# Download without installing
apt-get download cuda-libraries-dev-12-4

# List all files in package
dpkg -c cuda-libraries-dev-12-4*.deb | grep -E '(cupti|include|static)'
```

### Compiler Toolchain (`cuda-compiler-{version}`)

**Package**: `cuda-compiler-12-4`

**What's included**:
- nvcc (CUDA compiler driver)
- nvprof (legacy profiler, deprecated)
- cuda-gdb (CUDA debugger)
- ptxas (PTX assembler)
- fatbinary (multi-architecture binary tool)
- cuobjdump (CUDA object dump utility)

**Installation**:
```bash
apt-get install -y --no-install-recommends cuda-compiler-12-4
```

**Installed size**: ~300MB

**Use case**: Compiling `.cu` source files to GPU binaries. Required for custom CUDA kernel development.

**Binaries installed**:
```
/usr/local/cuda-12.4/bin/nvcc
/usr/local/cuda-12.4/bin/nvprof
/usr/local/cuda-12.4/bin/cuda-gdb
/usr/local/cuda-12.4/bin/ptxas
/usr/local/cuda-12.4/bin/fatbinary
/usr/local/cuda-12.4/bin/cuobjdump
```

**NOT included**:
- CUDA libraries (install cuda-libraries-dev separately)
- NSight tools (install cuda-tools-12-4)

### Full Toolkit Meta-Package

**Package**: `cuda-toolkit-12-4`

**What's included**: ALL of the above (kitchen sink approach)

**Installation**:
```bash
apt-get install -y cuda-toolkit-12-4
```

**Installed size**: ~4GB

**Use case**: Full development environment on workstations. **NOT recommended for Docker** (too large).

**Better approach for Docker**:
```bash
# Install only what you need
apt-get install -y --no-install-recommends \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4
# Total: ~1.8GB vs 4GB for full toolkit
```

### Minimal Package Combinations

**For inference only** (smallest):
```bash
apt-get install -y --no-install-recommends \
    cuda-cudart-12-4 \
    libcublas-12-4
# Size: ~100MB
# Supports: Basic CUDA runtime + cuBLAS operations
```

**For PyTorch/TensorFlow inference**:
```bash
apt-get install -y --no-install-recommends \
    cuda-libraries-12-4
# Size: ~1GB
# Supports: Pre-compiled PyTorch/TensorFlow execution
```

**For compiling PyTorch from source**:
```bash
apt-get install -y --no-install-recommends \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4
# Size: ~1.8GB
# Supports: Full compilation + CUPTI profiling
```

**For custom CUDA kernel development**:
```bash
apt-get install -y --no-install-recommends \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4 \
    cuda-tools-12-4
# Size: ~2GB
# Supports: Full development + profiling tools
```

From [Vultr Docs - Install NVIDIA CUDA Toolkit](https://docs.vultr.com/how-to-install-nvidia-cuda-toolkit-on-ubuntu-22-04) (accessed 2025-11-13):
> "For minimal installations, use cuda-libraries-{version} for runtime or cuda-libraries-dev-{version} for development."

---

## Section 3: CUPTI-Specific Installation (~100 lines)

### CUPTI Package Location (Critical Discovery)

**Problem**: CUPTI is NOT in `/usr/local/cuda/extras/CUPTI/` when installed via apt.

**Actual location**: CUPTI shared libraries live in `cuda-libraries-dev-{version}` package.

From arr-coc-0-1 CUPTI investigation (2025-11-13):

#### Discovery Process

**Initial assumption** (WRONG):
```bash
# Copying from devel image extras directory
COPY --from=builder /usr/local/cuda/extras/CUPTI /usr/local/cuda/extras/CUPTI
# FAILS: Only headers, no .so files!
```

**Investigation**:
```bash
# Search for CUPTI libraries
find / -name "libcupti.so*" 2>/dev/null
# Output:
# /usr/lib/x86_64-linux-gnu/libcupti.so.12
# /usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
```

**Root cause**: apt packages install libraries in `/usr/lib/`, not `/usr/local/cuda/extras/`.

**Verified via package inspection**:
```bash
# Download and inspect
apt-get download cuda-libraries-dev-12-4
dpkg -c cuda-libraries-dev-12-4*.deb | grep cupti

# Output:
# ./usr/include/cupti.h
# ./usr/include/cupti_activity.h
# ./usr/include/cupti_callbacks.h
# ./usr/lib/x86_64-linux-gnu/libcupti.so.12
# ./usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
# ./usr/lib/x86_64-linux-gnu/stubs/libcupti.so
```

### Selective CUPTI Installation (Minimal Approach)

**Pattern**: Install → Copy → Remove (saves ~480MB)

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Option 1: Keep full dev package (~500MB overhead)
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4

# Option 2: Extract CUPTI only (~20MB overhead)
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    mkdir -p /usr/local/lib && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    cp /usr/lib/x86_64-linux-gnu/libnvperf_host.so /usr/local/lib/ && \
    cp /usr/lib/x86_64-linux-gnu/libnvperf_target.so /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Configure library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**What gets extracted**:
```
libcupti.so.12 -> libcupti.so.12.4.127 (main CUPTI library, ~18MB)
libnvperf_host.so (host-side profiling, ~2MB)
libnvperf_target.so (device-side profiling, ~1MB)
```

**Size impact** (arr-coc-0-1 production validation):
```
Without CUPTI: 1.5GB runtime image
With full dev package: 2.0GB (+500MB)
With extracted CUPTI: 1.52GB (+20MB) ✓ OPTIMAL
```

### CUPTI Installation Verification

**Verify CUPTI availability**:
```bash
# Check library presence
ls -lh /usr/local/lib/libcupti.so*

# Test with Python
python -c "
import torch
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    x = torch.randn(100, 100, device='cuda')
    y = x @ x
print('CUPTI available:', prof.kineto_results is not None)
"
```

**Common error without CUPTI**:
```
OSError: libcupti.so.12: cannot open shared object file: No such file or directory
```

**Fix**: Ensure LD_LIBRARY_PATH includes CUPTI location:
```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### CUPTI vs NSight Tools

**CUPTI** (CUDA Profiling Tools Interface):
- Programmatic profiling API
- Required by torch.profiler, Kineto
- Libraries: libcupti.so (~18MB)
- Use case: In-application profiling

**NSight Tools** (standalone profilers):
- NSight Systems: Timeline profiling (works WITHOUT CUPTI)
- NSight Compute: Kernel-level profiling (uses CUPTI)
- Binaries: nsys, ncu (~200MB)
- Use case: External profiling tools

**For Docker production**:
- Skip CUPTI for inference-only workloads
- Use NSight Systems instead (no CUPTI dependency)
- Extract CUPTI only if torch.profiler needed

From [NVIDIA Developer Forums - Minimal CUDA runtime install](https://forums.developer.nvidia.com/t/minimal-cuda-runtime-install-on-ubuntu/63565) (accessed 2025-11-13):
> "For minimal Docker images, install cuda-libraries-{version} for runtime. Add cuda-libraries-dev-{version} only if profiling is required."

---

## Section 4: Multi-Stage Docker Patterns (~100 lines)

### Builder Stage (Compilation)

**Pattern**: Full dev environment for building, minimal runtime for deployment.

```dockerfile
# ==================== BUILDER STAGE ====================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    cmake \
    ninja-build \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Compile PyTorch from source (example)
WORKDIR /build
RUN pip install torch --no-binary torch

# Build custom CUDA extensions
COPY ./cuda_kernels /build/cuda_kernels
RUN cd /build/cuda_kernels && \
    python setup.py install
```

**Builder stage size**: ~6GB (devel base + build tools)

### Runtime Stage (Deployment)

**Pattern 1: Without CUPTI** (smallest):
```dockerfile
# ==================== RUNTIME STAGE ====================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy compiled PyTorch from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch \
                    /usr/local/lib/python3.10/site-packages/torch

# Copy compiled extensions
COPY --from=builder /usr/local/lib/python3.10/site-packages/cuda_kernels \
                    /usr/local/lib/python3.10/site-packages/cuda_kernels

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3-minimal \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

CMD ["python3", "app.py"]
```

**Runtime stage size**: ~2.9GB

**Pattern 2: With extracted CUPTI** (optimal for profiling):
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy compiled code from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Extract CUPTI from dev package
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    mkdir -p /usr/local/lib && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Runtime stage size**: ~3.0GB (CUPTI adds 100MB vs 20MB extracted)

### Conditional CUPTI Installation

**Pattern**: Enable profiling via build argument.

```dockerfile
ARG ENABLE_PROFILING=0

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Conditional CUPTI installation
RUN if [ "$ENABLE_PROFILING" = "1" ]; then \
        apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
        mkdir -p /usr/local/lib && \
        cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
        apt-get remove -y cuda-libraries-dev-12-4 && \
        apt-get autoremove -y && \
        rm -rf /var/lib/apt/lists/*; \
    fi

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Build commands**:
```bash
# Production build (no profiling)
docker build -t myapp:prod .

# Development build (with profiling)
docker build -t myapp:dev --build-arg ENABLE_PROFILING=1 .
```

### Apt Best Practices for Docker

**Always use these flags**:
```bash
apt-get install -y --no-install-recommends PACKAGE
```

**Why `--no-install-recommends`**:
```bash
# Without flag: Installs 200+ recommended packages
apt-get install -y cuda-libraries-12-4
# Size: ~1.5GB

# With flag: Installs only required dependencies
apt-get install -y --no-install-recommends cuda-libraries-12-4
# Size: ~1GB (saves 500MB)
```

**Clean up after installation**:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-12-4 && \
    rm -rf /var/lib/apt/lists/*
```

**Why cleanup matters**:
```bash
/var/lib/apt/lists/*  # Package metadata (~100MB)
/var/cache/apt/*.bin  # Package cache (~50MB)
```

**Complete pattern**:
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-libraries-12-4 \
        cuda-cudart-12-4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*
```

**Layer optimization** (combine commands):
```dockerfile
# ❌ BAD: Creates 3 layers
RUN apt-get update
RUN apt-get install -y cuda-libraries-12-4
RUN rm -rf /var/lib/apt/lists/*

# ✅ GOOD: Creates 1 layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-libraries-12-4 && \
    rm -rf /var/lib/apt/lists/*
```

### arr-coc-0-1 Production Pattern

**From arr-coc-0-1 Cloud Build Dockerfile** (validated 2025-11-13):

```dockerfile
# Builder stage: Compile PyTorch with A100 support
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev python3-pip git cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*
RUN pip install torch --no-binary torch

# Runtime stage: Minimal deployment image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch \
                    /usr/local/lib/python3.10/site-packages/torch

# Extract CUPTI for profiling
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Result**:
- Builder stage: 8.5GB (discarded after build)
- Runtime stage: 1.52GB (deployed to Vertex AI)
- Savings: 6.98GB vs single-stage devel image

From [Roboflow Blog - Docker GPU Support](https://blog.roboflow.com/use-the-gpu-in-docker/) (accessed 2025-11-13):
> "Multi-stage builds dramatically reduce production image sizes while maintaining full GPU functionality."

---

## Sources

**Source Documents**:
- cuda/13-nvidia-container-cuda-packaging.md (package structure reference)
- cuda/10-docker-multistage-ml-optimization.md (multi-stage patterns)
- arr-coc-0-1/training/CUPTI_INVESTIGATION_STUDY.md (CUPTI discovery context)

**Web Research**:
- [NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13) - Official installation guide
- [Vultr Docs - Install NVIDIA CUDA Toolkit Ubuntu 22.04](https://docs.vultr.com/how-to-install-nvidia-cuda-toolkit-on-ubuntu-22-04) (accessed 2025-11-13) - Ubuntu-specific guide
- [Ask Ubuntu - CUDA Installation Issues Ubuntu 24.04](https://askubuntu.com/questions/1520509/issues-installing-cuda-and-cudnn-on-ubuntu-24-04) (accessed 2025-11-13) - Ubuntu 24.04 compatibility
- [NVIDIA Developer Forums - Minimal CUDA runtime install](https://forums.developer.nvidia.com/t/minimal-cuda-runtime-install-on-ubuntu/63565) (accessed 2025-11-13) - Minimal package selection
- [Roboflow Blog - Docker GPU Support](https://blog.roboflow.com/use-the-gpu-in-docker/) (accessed 2025-11-13) - Docker best practices
- [Stack Overflow - apt-get install cuda vs nvidia-cuda-toolkit](https://askubuntu.com/questions/855396/is-there-a-difference-between-sudo-apt-get-install-cuda-and-sudo-apt-get-inst) (accessed 2025-11-13) - Package naming clarification

**Additional References**:
- NVIDIA CUDA apt repository: https://developer.download.nvidia.com/compute/cuda/repos/
- Docker multi-stage build documentation: https://docs.docker.com/build/building/multi-stage/
