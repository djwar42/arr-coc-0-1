# NVIDIA Container Images & CUDA Apt Packaging Structure

**Production knowledge from CUPTI investigation**: Understanding NVIDIA's container ecosystem and CUDA apt package structure is critical for Docker multi-stage builds and CUPTI troubleshooting.

**Key discovery**: CUPTI lives in `cuda-libraries-dev-12-4`, not `cuda-libraries-12-4` (runtime only).

---

## Section 1: NVIDIA Container Image Variants (~120 lines)

### Three Official Image Types

From [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda) (accessed 2025-11-13):

NVIDIA provides three distinct container image variants, each optimized for different use cases:

#### 1. Base Images (`nvidia/cuda:{version}-base-ubuntu22.04`)

**What's included**:
- Bare minimum: `libcudart` (CUDA runtime library only)
- No CUDA toolkit libraries
- No development headers
- No compiler toolchain

**Size**: ~300-500 MB

**Use case**: Manual package selection. You control exactly which CUDA packages to install via apt.

**Example**:
```dockerfile
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# You manually install what you need:
RUN apt-get update && apt-get install -y \
    cuda-libraries-12-4 \
    cuda-cudart-12-4
```

**When to use**: Custom installations where you need fine control over package versions and dependencies.

#### 2. Runtime Images (`nvidia/cuda:{version}-runtime-ubuntu22.04`)

**What's included**:
- Everything from base
- ALL shared libraries from CUDA toolkit (`libcublas`, `libcufft`, `libcurand`, etc.)
- cuDNN runtime libraries (if `-cudnn8-runtime` variant)
- NO development headers
- NO compiler toolchain (nvcc)
- NO static libraries

**Size**: ~1.5-2.5 GB (runtime), ~3-4 GB (cudnn-runtime)

**Use case**: Pre-built applications using multiple CUDA libraries. Perfect for inference workloads.

**Example**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Your pre-compiled PyTorch/TensorFlow application runs here
COPY ./my_app /app
CMD ["python", "/app/inference.py"]
```

**What you CAN run**:
- PyTorch inference (uses libcublas, libcudnn, libcufft)
- TensorFlow inference
- Any pre-compiled CUDA application

**What you CANNOT do**:
- Compile CUDA kernels (no nvcc)
- Build PyTorch from source (no headers)
- Use profiling tools like CUPTI (not included)

#### 3. Devel Images (`nvidia/cuda:{version}-devel-ubuntu22.04`)

**What's included**:
- Everything from runtime
- CUDA compiler toolchain (nvcc, nvprof)
- Development headers (`cuda.h`, `cublas_v2.h`, etc.)
- Static libraries (`.a` files)
- Debugging tools (cuda-gdb)
- Profiling tools (CUPTI, NSight)
- CMake integration files

**Size**: ~4.5-6 GB (devel), ~7-9 GB (cudnn-devel)

**Use case**: Compiling CUDA applications from source. Multi-stage Docker builds (build stage).

**Example multi-stage build**:
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

### Variant Naming Conventions

NVIDIA uses consistent naming patterns:

```
nvidia/cuda:{CUDA_VERSION}-{IMAGE_TYPE}-{OS}

Examples:
- nvidia/cuda:12.4.0-base-ubuntu22.04
- nvidia/cuda:12.4.0-runtime-ubuntu22.04
- nvidia/cuda:12.4.0-devel-ubuntu22.04
- nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04
- nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04
```

**cuDNN variants**: Add `cudnn8` for deep learning frameworks (PyTorch, TensorFlow).

### Image Comparison Table

| Feature | Base | Runtime | Devel |
|---------|------|---------|-------|
| Size | ~500 MB | ~1.5-2.5 GB | ~4.5-6 GB |
| libcudart | ✓ | ✓ | ✓ |
| CUDA libraries (cublas, cufft) | ✗ | ✓ | ✓ |
| Headers (.h files) | ✗ | ✗ | ✓ |
| nvcc compiler | ✗ | ✗ | ✓ |
| Static libraries (.a) | ✗ | ✗ | ✓ |
| CUPTI profiling | ✗ | ✗ | ✓ |
| NSight tools | ✗ | ✗ | ✓ |
| Use case | Custom install | Inference | Compilation |

**From [Stack Overflow discussion](https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container) (accessed 2025-11-13)**:
> "Runtime includes shared libraries for pre-built apps, while devel includes compiler tools for compiling from source."

### CUPTI Discovery (Production Context)

**Critical finding from arr-coc-0-1 CUPTI investigation**:

When building PyTorch from source with `nvidia/cuda:12.4.0-devel-ubuntu22.04`:
- CMake finds CUPTI headers (`cupti.h`) in `/usr/local/cuda/extras/CUPTI/include/`
- PyTorch builds successfully with profiling support
- **Runtime image missing**: Copying `/usr/local/cuda/extras/CUPTI/` to runtime doesn't work (headers only, no `.so` files)

**Solution**: CUPTI shared libraries live in apt package `cuda-libraries-dev-12-4` (not devel image extras directory).

See Section 3 for complete CUPTI package details.

---

## Section 2: CUDA Apt Package Ecosystem (~120 lines)

### Package Hierarchy

NVIDIA distributes CUDA through Ubuntu apt repositories with version-specific packages:

```
cuda-toolkit-12-4 (meta-package)
├── cuda-libraries-12-4 (runtime)
├── cuda-libraries-dev-12-4 (development)
├── cuda-compiler-12-4
├── cuda-tools-12-4
└── cuda-documentation-12-4
```

**From [NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13)**:

### Runtime Packages (`cuda-libraries-{version}`)

**Package**: `cuda-libraries-12-4`

**What's included**:
- Shared libraries only (`.so` files)
- libcublas, libcufft, libcurand, libcusparse, libcusolver
- libcudart (CUDA runtime)
- NO headers
- NO static libraries
- NO CUPTI

**Installation**:
```bash
apt-get update
apt-get install -y cuda-libraries-12-4
```

**Use case**: Runtime dependencies for pre-compiled CUDA applications. Equivalent to runtime Docker images.

**What you get** (example):
```
/usr/lib/x86_64-linux-gnu/libcublas.so.12
/usr/lib/x86_64-linux-gnu/libcudart.so.12
/usr/lib/x86_64-linux-gnu/libcufft.so.12
```

### Development Packages (`cuda-libraries-dev-{version}`)

**Package**: `cuda-libraries-dev-12-4`

**What's included**:
- Everything from `cuda-libraries-12-4`
- Development headers (`.h` files)
- Static libraries (`.a` files)
- **CUPTI shared libraries** (critical discovery!)
- CMake integration files

**Installation**:
```bash
apt-get update
apt-get install -y cuda-libraries-dev-12-4
```

**Use case**: Compiling CUDA applications. Equivalent to devel Docker images (but more granular control).

**What you get** (example):
```
/usr/include/cublas_v2.h
/usr/include/cufft.h
/usr/lib/x86_64-linux-gnu/libcublas_static.a
/usr/lib/x86_64-linux-gnu/libcupti.so.12  ← CUPTI HERE!
/usr/share/cmake-3.22/Modules/FindCUDA.cmake
```

**Package dependencies**:
```bash
# cuda-libraries-dev-12-4 depends on:
cuda-libraries-12-4 (>= 12.4.0)
cuda-nvml-dev-12-4
cuda-cudart-dev-12-4
```

### Compiler Toolchain (`cuda-compiler-{version}`)

**Package**: `cuda-compiler-12-4`

**What's included**:
- nvcc (CUDA compiler)
- nvprof (legacy profiler)
- cuda-gdb (CUDA debugger)
- ptxas (PTX assembler)
- fatbinary (multi-architecture binary tool)

**Installation**:
```bash
apt-get update
apt-get install -y cuda-compiler-12-4
```

**Use case**: Compiling `.cu` source files to GPU binaries.

### Full Toolkit Meta-Package

**Package**: `cuda-toolkit-12-4`

**What's included**: ALL of the above (kitchen sink approach)

**Installation**:
```bash
apt-get update
apt-get install -y cuda-toolkit-12-4
```

**Size**: ~3-4 GB installed

**Use case**: Full development environment. Not recommended for Docker (too large). Use granular packages instead.

### Version-Specific Package Naming

NVIDIA uses `-{major}-{minor}` suffix for all packages:

```
CUDA 11.8:
- cuda-libraries-11-8
- cuda-libraries-dev-11-8
- cuda-compiler-11-8

CUDA 12.1:
- cuda-libraries-12-1
- cuda-libraries-dev-12-1
- cuda-compiler-12-1

CUDA 12.4:
- cuda-libraries-12-4
- cuda-libraries-dev-12-4
- cuda-compiler-12-4
```

**Why?**: Allows side-by-side installation of multiple CUDA versions.

**Example multi-version setup**:
```bash
# Install both CUDA 11.8 and 12.4
apt-get install -y \
    cuda-libraries-11-8 \
    cuda-libraries-12-4

# Switch using /usr/local/cuda symlink
```

### Package Search

**Find available CUDA packages**:
```bash
# List all CUDA 12.4 packages
apt-cache search cuda-12-4

# Show package details
apt-cache show cuda-libraries-dev-12-4
```

**From [Ubuntu CUDA Installation Guide](https://medium.com/@juliuserictuliao/documentation-installing-cuda-on-ubuntu-22-04-2c5c411df843) (accessed 2025-11-13)**:

---

## Section 3: CUPTI Location & Installation (~80 lines)

### CUPTI in Development Packages

**Critical finding**: CUPTI is NOT in `/usr/local/cuda/extras/CUPTI/` on runtime images.

**Actual location**: CUPTI shared libraries live in `cuda-libraries-dev-{version}` apt package.

**From arr-coc-0-1 CUPTI investigation**:

#### Problem: Missing CUPTI at Runtime

PyTorch compiled from source with devel image:
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
# PyTorch compilation succeeds (finds CUPTI headers)
# CMake detects: /usr/local/cuda/extras/CUPTI/include/cupti.h
RUN python setup.py install
```

Runtime deployment fails:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /torch
# torch.profiler.profile() crashes:
# OSError: libcupti.so.12: cannot open shared object file
```

Attempted fix (FAILED):
```dockerfile
# Copy CUPTI from devel to runtime
COPY --from=builder /usr/local/cuda/extras/CUPTI /usr/local/cuda/extras/CUPTI
# Still fails! extras/CUPTI only has headers, not .so files
```

#### Solution: Install cuda-libraries-dev Package

**Correct approach**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Option 1: Install full dev package (~500 MB)
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4

# Option 2: Extract CUPTI .so files only (minimal, ~20 MB)
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

**Option 2 explained**:
1. Install `cuda-libraries-dev-12-4` (gets CUPTI .so)
2. Copy CUPTI libraries to `/usr/local/lib/` (permanent location)
3. Remove dev package (no longer needed)
4. Clean up (save ~480 MB)

**What gets copied**:
```
/usr/lib/x86_64-linux-gnu/libcupti.so.12
/usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
/usr/lib/x86_64-linux-gnu/libnvperf_host.so
/usr/lib/x86_64-linux-gnu/libnvperf_target.so
```

### CUPTI Package Contents

**Inspect package before installation**:
```bash
# Download package without installing
apt-get download cuda-libraries-dev-12-4

# List contents
dpkg -c cuda-libraries-dev-12-4*.deb | grep cupti
```

**Output**:
```
./usr/include/cupti.h
./usr/include/cupti_activity.h
./usr/include/cupti_callbacks.h
./usr/include/cupti_driver_cbid.h
./usr/include/cupti_events.h
./usr/include/cupti_metrics.h
./usr/include/cupti_nvtx.h
./usr/include/cupti_pcsampling.h
./usr/include/cupti_profiler_target.h
./usr/include/cupti_result.h
./usr/include/cupti_sass_metrics.h
./usr/include/cupti_target.h
./usr/include/cupti_version.h
./usr/lib/x86_64-linux-gnu/libcupti.so.12
./usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
./usr/lib/x86_64-linux-gnu/stubs/libcupti.so
```

**Headers**: `/usr/include/cupti*.h` (for compilation)
**Libraries**: `/usr/lib/x86_64-linux-gnu/libcupti.so*` (for runtime)

### Why This Matters for Multi-Stage Builds

**Problem**: Devel image has headers but libraries come from apt package, not `/usr/local/cuda/extras/`.

**Correct multi-stage pattern**:
```dockerfile
# Stage 1: Build with devel image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
WORKDIR /build

# Compile PyTorch with CUPTI support
RUN python -m pip install torch --no-binary torch

# Stage 2: Runtime with selective CUPTI
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy compiled PyTorch
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /usr/local/lib/python3.10/site-packages/torch

# Install CUPTI .so files (minimal approach)
RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Size impact**:
- Without CUPTI: 1.5 GB runtime image
- With full dev package: 2 GB (+500 MB)
- With extracted CUPTI: 1.52 GB (+20 MB) ✓ optimal

---

## Section 4: NVIDIA Apt Repository Setup (~80 lines)

### Adding NVIDIA CUDA Repository

**Official NVIDIA repository** (Ubuntu 22.04):

```bash
# Step 1: Download repository pin file
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Step 2: Add repository GPG key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Step 3: Update apt
sudo apt-get update

# Step 4: Install specific CUDA version
sudo apt-get install -y cuda-toolkit-12-4
```

**From [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13)**.

### Repository Structure

NVIDIA maintains separate repositories for:
- Stable releases: `https://developer.download.nvidia.com/compute/cuda/repos/`
- Beta/experimental: `https://developer.download.nvidia.com/compute/cuda/repos/experimental/`

**Repository URL format**:
```
https://developer.download.nvidia.com/compute/cuda/repos/{OS}/{ARCH}/
```

**Examples**:
```
Ubuntu 22.04 x86_64:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

Ubuntu 20.04 x86_64:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/

Ubuntu 22.04 ARM64:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/
```

### Version Pinning

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
EOF

# Install pinned version
sudo apt-get install -y cuda-toolkit-12-4
```

**Why pin?**: Prevents `apt upgrade` from installing newer CUDA versions that may break compatibility.

### Multi-Version Installation

**Install multiple CUDA versions side-by-side**:

```bash
# Install CUDA 11.8 and 12.4
sudo apt-get install -y \
    cuda-toolkit-11-8 \
    cuda-toolkit-12-4

# Check installations
ls -l /usr/local/ | grep cuda
```

**Output**:
```
lrwxrwxrwx cuda -> cuda-12.4           (default symlink)
drwxr-xr-x cuda-11.8
drwxr-xr-x cuda-12.4
```

**Switch between versions**:
```bash
# Use CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Use CUDA 12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### Driver vs Runtime vs Toolkit Compatibility

**NVIDIA component compatibility** (critical for production):

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Driver | 550.54.15 | Kernel module (host) |
| CUDA Runtime | 12.4 | libcudart.so (container) |
| CUDA Toolkit | 12.4 | nvcc, libraries (container) |

**Compatibility rules**:
1. **Driver ≥ Runtime**: Driver version must be equal or newer than CUDA runtime
2. **Forward compatibility**: CUDA 11.8 app runs on Driver 550 (CUDA 12.4 driver)
3. **Backward incompatibility**: CUDA 12.4 app may NOT run on Driver 470 (CUDA 11.4 driver)

**Check driver version**:
```bash
nvidia-smi
# Driver Version: 550.54.15   CUDA Version: 12.4
```

**CUDA forward compatibility package** (for older drivers):
```bash
# Install forward compatibility for CUDA 12.4 on CUDA 11.x driver
apt-get install -y cuda-compat-12-4
```

**From [NVIDIA Container Toolkit Architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html) (accessed 2025-11-13)**:
> "The NVIDIA container stack relies on kernel primitives and is designed to be agnostic of the container runtime."

### Docker-Specific Repository Configuration

**For Docker images** using apt packages:

```dockerfile
FROM ubuntu:22.04

# Add NVIDIA repository
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Install specific CUDA packages
RUN apt-get install -y \
    cuda-libraries-12-4 \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4
```

**Best practice**: Use official NVIDIA base images instead (they already have repository configured).

---

## Sources

**Source Documents**:
- arr-coc-0-1/training/CUPTI_INVESTIGATION_STUDY.md (CUPTI discovery context)

**Web Research**:
- [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda) (accessed 2025-11-13) - Official container image documentation
- [NVIDIA Container Toolkit Architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html) (accessed 2025-11-13) - Container stack components
- [Stack Overflow: devel vs runtime Docker tags](https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container) (accessed 2025-11-13) - Community explanation
- [NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13) - Official installation guide
- [Ubuntu CUDA Installation Guide](https://medium.com/@juliuserictuliao/documentation-installing-cuda-on-ubuntu-22-04-2c5c411df843) (accessed 2025-11-13) - Community tutorial

**Additional References**:
- NVIDIA apt repository: https://developer.download.nvidia.com/compute/cuda/repos/
- CUDA forward compatibility documentation: https://docs.nvidia.com/deploy/cuda-compatibility/
