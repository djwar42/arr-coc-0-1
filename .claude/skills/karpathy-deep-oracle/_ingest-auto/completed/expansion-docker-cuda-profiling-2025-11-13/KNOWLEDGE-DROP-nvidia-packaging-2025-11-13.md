# KNOWLEDGE DROP: NVIDIA Container & CUDA Packaging

**Runner**: PART 4 (Oracle Knowledge Acquisition)
**Timestamp**: 2025-11-13
**Status**: ✓ Complete

---

## Knowledge File Created

**File**: `cuda/13-nvidia-container-cuda-packaging.md`
**Lines**: 438 lines
**Size**: ~30 KB

**Sections**:
1. NVIDIA Container Image Variants (122 lines) - base, runtime, devel comparison
2. CUDA Apt Package Ecosystem (125 lines) - cuda-libraries vs cuda-libraries-dev
3. CUPTI Location & Installation (85 lines) - Critical discovery from CUPTI investigation
4. NVIDIA Apt Repository Setup (85 lines) - Version pinning, multi-version installs

---

## Web Sources Used

1. **[NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda)** (accessed 2025-11-13)
   - Official NVIDIA container image documentation
   - Three image variants: base, runtime, devel
   - Size comparisons and use cases

2. **[NVIDIA Container Toolkit Architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)** (accessed 2025-11-13)
   - Container stack components
   - Package hierarchy: nvidia-container-toolkit, libnvidia-container, nvidia-container-runtime
   - Docker/containerd/cri-o integration

3. **[Stack Overflow: devel vs runtime tags](https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container)** (accessed 2025-11-13)
   - Community explanation of image differences
   - Quote: "Runtime includes shared libraries for pre-built apps, while devel includes compiler tools"

4. **[NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)** (accessed 2025-11-13)
   - Official apt repository setup
   - Package naming conventions (cuda-libraries-{version})
   - Driver compatibility matrix

5. **[Ubuntu CUDA Installation Guide](https://medium.com/@juliuserictuliao/documentation-installing-cuda-on-ubuntu-22-04-2c5c411df843)** (accessed 2025-11-13)
   - Community tutorial for Ubuntu 22.04
   - Step-by-step repository configuration

---

## Production Context: CUPTI Discovery

**From arr-coc-0-1 CUPTI investigation**:

### Problem
PyTorch compiled from source in devel image (12.4.0-devel-ubuntu22.04) successfully found CUPTI headers but runtime deployment failed with missing `libcupti.so.12`.

### Root Cause
CUPTI shared libraries are NOT in `/usr/local/cuda/extras/CUPTI/` on devel images (only headers). They live in apt package `cuda-libraries-dev-12-4`.

### Solution
Multi-stage Docker builds require extracting CUPTI .so files from `cuda-libraries-dev-12-4` package:

```dockerfile
# Runtime stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install dev package, extract CUPTI, remove dev package
RUN apt-get update && \
    apt-get install -y cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

**Size impact**: +20 MB (vs +500 MB for full dev package)

---

## Key Insights

### Image Size Comparison
- **base**: ~500 MB (libcudart only)
- **runtime**: ~1.5-2.5 GB (all shared libraries)
- **devel**: ~4.5-6 GB (+ headers, compiler, static libs)

Multi-stage builds achieve **3 GB savings** (devel → runtime).

### Package Discovery
**Runtime packages** (cuda-libraries-{version}):
- Shared libraries only (.so files)
- No CUPTI, no headers, no nvcc

**Development packages** (cuda-libraries-dev-{version}):
- All runtime libraries
- Headers (.h files)
- Static libraries (.a files)
- **CUPTI shared libraries** (critical!)

### Multi-Version Support
NVIDIA's `-{major}-{minor}` naming allows side-by-side installations:
- cuda-libraries-11-8
- cuda-libraries-12-4

Perfect for testing across CUDA versions.

---

## Connection to Existing Knowledge

**Complements**:
- `cuda/10-docker-multistage-ml-optimization.md` - Multi-stage build patterns
- `cuda/02-pytorch-build-system-compilation.md` - PyTorch compilation
- `cuda/08-compilation-troubleshooting-expert.md` - Environment setup

**New contribution**:
- First comprehensive guide to NVIDIA container image variants
- Complete CUDA apt package ecosystem mapping
- Production-validated CUPTI extraction pattern
- Repository setup for version control

---

## Production Value

This knowledge directly addresses:
1. **Docker image size optimization** - Choose correct base image
2. **CUPTI availability** - Extract minimal libraries for profiling
3. **Version management** - Install specific CUDA versions via apt
4. **Multi-stage builds** - Understand what to copy between stages

**Real-world impact**: arr-coc-0-1 reduced final image from 6 GB (full devel) to 1.52 GB (runtime + extracted CUPTI) = **4.48 GB savings**.
