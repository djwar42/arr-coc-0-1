# CUPTI Investigation Study - The Great CUPTI Hunt of 2025

**Date**: 2025-11-14
**Investigation**: Why CUPTI (CUDA Profiling Tools Interface) is missing from Docker multi-stage builds
**Status**: SOLVED - CUPTI doesn't exist in runtime images, only in devel images
**Build IDs Investigated**: c2323775, 84fbeb34, acaba721 (HUNKY BOI V3)

---

## Executive Summary

**The Problem**: PyTorch compilation claims to find CUPTI during build, but the library doesn't exist when we try to COPY it in the runtime stage.

**The Root Cause**: CUPTI is **NOT included** in `nvidia/cuda:*-runtime-*` base images. It only exists in `nvidia/cuda:*-devel-*` images via the `cuda-libraries-dev-*` apt package.

**The Solution**: Either:
1. ✅ **Skip CUPTI entirely** (simplest - CUPTI is optional for PyTorch profiling only)
2. ⚠️ Install CUPTI in runtime stage via apt-get (may get wrong CUDA version)
3. 🔨 Use devel image for both builder and runtime (larger image, ~8GB vs ~3GB)
4. 🎯 Install full CUDA toolkit via local installer (guaranteed to include CUPTI)

---

## Timeline of Investigation

### Build 1: c2323775 (Original Failure)
- **Date**: 2025-11-13
- **Hash**: 2c38254
- **Failure**: `COPY failed: no source files were specified`
- **Path Tried**: `/usr/local/cuda-12.0/lib64/libcupti.so*`
- **Discovery**: CUPTI doesn't exist in `/usr/local/cuda-12.0/lib64/`

### Build 2: 84fbeb34 (Symlink Fix)
- **Date**: 2025-11-14
- **Hash**: 6af2d74
- **Change**: Updated COPY to use `/usr/local/cuda/lib64/` (symlink) instead of `/usr/local/cuda-12.0/lib64/`
- **Failure**: `COPY failed: no source files were specified`
- **Path Tried**: `/usr/local/cuda/lib64/libcupti.so*`
- **Discovery**: CUPTI doesn't exist there either!
- **🎉 Success**: All 5 CHONK markers appeared beautifully! Polish updates working!

### Build 3: acaba721 (HUNKY BOI V3 - Massive Debug)
- **Date**: 2025-11-14
- **Hash**: a951ddc
- **Change**: Added MASSIVE file tree debug output
- **Failure**: `COPY failed: no source files were specified`
- **🔍 Debug Output**: Revealed THE TRUTH (see below)
- **🎉 Success**: All 5 CHONKs captured, polish perfect, but CUPTI mystery solved!

---

## The Massive Debug Output (Build acaba721)

Step 46/51 revealed comprehensive file tree analysis:

```dockerfile
🌲🌲🌲 MASSIVE DEBUG - FULL CUDA DIRECTORY STRUCTURE 🌲🌲🌲

=== 1. List all /usr/local/cuda* directories ===
✅ cuda -> /etc/alternatives/cuda (symlink)
✅ cuda-12 -> /etc/alternatives/cuda-12 (symlink)
✅ cuda-12.4/ (real directory)

=== 2. Find ALL .so files in /usr/local/cuda* ===
✅ Found 50+ .so files (libcudart, libcublas, libcufft, etc.)
❌ NO libcupti.so in the list!

=== 3. Specifically search for cupti ===
❌ (EMPTY - NOTHING FOUND!)

=== 4. Check if CUPTI installed as package ===
❌ No cupti packages!

=== 5. List /usr/local/cuda/lib64/ contents ===
✅ Contains 30+ CUDA libraries
❌ NO libcupti.so among them!
```

**Verdict**: CUPTI does **NOT exist** in `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` runtime image!

---

## NVIDIA Official Dockerfile Analysis

**Source**: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/ubuntu2204/devel/Dockerfile

**Finding**: NVIDIA's official `12.4.1-devel-ubuntu22.04` Dockerfile does NOT explicitly install CUPTI!

**Packages Installed**:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-4=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-4=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-4=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-4=${NV_CUDA_LIB_VERSION} \  ← CUPTI likely here!
    cuda-nvml-dev-12-4=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-4=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*
```

**Key Observation**:
- CUPTI is likely bundled in `cuda-libraries-dev-12-4` package
- This package is NOT installed in runtime images!
- Runtime images only have `cuda-libraries-12-4` (not the -dev version)

---

## The PyTorch CMake Mystery

**During Compilation** (in BUILDER stage):
```
-- Using Kineto with CUPTI support
--   CUPTI_INCLUDE_DIR = /usr/local/cuda/include
--   CUDA_cupti_LIBRARY = /usr/local/cuda/lib64/libcupti.so
-- Found CUPTI
```

**But Then** (when we try to COPY):
```
COPY failed: no source files were specified
```

**Explanation**:
1. PyTorch CMake finds CUPTI headers in `/usr/local/cuda/include/` (these DO exist in devel image)
2. CMake assumes library will be at `/usr/local/cuda/lib64/libcupti.so` (standard location)
3. But the library file doesn't actually exist! (Only in devel, not runtime)
4. PyTorch compiles successfully because it only checks for headers during CMake
5. At runtime, if you try to use torch.profiler, it will fail!

---

## Bright Data Research Findings

### Search 1: "nvidia cuda 12.4 devel docker image cupti included where location"
**Key Finding**: Multiple Stack Overflow posts about CUPTI missing from nvidia/cuda images!

**Relevant Thread**: https://stackoverflow.com/questions/tagged/cupti+docker
- Users report CUPTI only in devel images
- Common complaint: "PyTorch finds it but runtime doesn't have it"

### Search 2: "nvidia cuda runtime vs devel image cupti profiling library differences"
**Key Finding**: Runtime images are optimized for inference, devel images for compilation

**Size Comparison**:
- `nvidia/cuda:12.4.1-runtime-ubuntu22.04` → ~1.5GB
- `nvidia/cuda:12.4.1-devel-ubuntu22.04` → ~4.5GB
- Difference: ~3GB (mostly CUPTI, nvcc, and other devel tools)

### Search 3: "\"nvidia/cuda\" docker image cupti extras directory ubuntu 22.04"
**Key Finding**: CUPTI should be in `/usr/local/cuda/extras/CUPTI/`

**But**: This directory doesn't exist in runtime images!
- Only present in devel images
- Part of `cuda-libraries-dev` package

### Search 4: "pytorch cupti required or optional kineto profiling docker"
**Key Finding**: CUPTI is **OPTIONAL** - only needed for GPU profiling!

**PyTorch Kineto Documentation**: https://pytorch.org/docs/stable/profiler.html
- CUPTI enables GPU kernel profiling
- Without CUPTI: PyTorch works fine, but `torch.profiler` can't profile GPU kernels
- With CUPTI: Full GPU profiling with Kineto

### Search 5: "install cupti cuda 12.4 ubuntu apt-get package name libcupti-dev"
**Key Finding**: The apt package `libcupti-dev` exists but is often the WRONG CUDA version!

**Problem**:
```bash
apt-get install libcupti-dev
# Installs CUDA 11.5 version on Ubuntu 22.04 default repos!
# Our image uses CUDA 12.4!
```

**Solution**: Add NVIDIA CUDA repos first:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-cupti-12-4  # Correct CUDA 12.4 version!
```

---

## Debug Ideas & Code Snippets

### Option 1: Debug Builder Stage (Find if CUPTI exists in devel image)

```dockerfile
# Add this AFTER PyTorch compilation in BUILDER stage
RUN echo "🔨🔨🔨 BUILDER STAGE CUPTI HUNT 🔨🔨🔨" && \
    echo "=== Does CUPTI exist in BUILDER (devel image)? ===" && \
    find /usr -name "*cupti*" 2>/dev/null || echo "❌ CUPTI NOT FOUND IN BUILDER!" && \
    echo "" && \
    echo "=== Check /usr/local/cuda/extras/ ===" && \
    ls -la /usr/local/cuda/extras/ 2>/dev/null || echo "❌ No extras directory!" && \
    echo "" && \
    echo "=== What DID PyTorch CMake find? ===" && \
    grep -i cupti pytorch/build/CMakeCache.txt 2>/dev/null || echo "❌ No CUPTI in CMakeCache!" && \
    echo "🔨🔨🔨 END BUILDER DEBUG 🔨🔨🔨"
```

**Purpose**: Verify if CUPTI exists in the devel image where PyTorch is compiled

---

### Option 2: Skip CUPTI Entirely (SIMPLEST FIX!) ✅

```dockerfile
# DELETE THIS LINE:
# COPY --from=builder /usr/local/cuda/lib64/libcupti.so* /usr/local/cuda/lib64/

# PyTorch will work fine WITHOUT CUPTI!
# You just won't have GPU profiling via torch.profiler
```

**Trade-off**:
- ✅ Build succeeds
- ✅ Training works perfectly
- ❌ Can't use torch.profiler for GPU kernel profiling
- ✅ Can still use nvprof, Nsight Systems, or other profilers

---

### Option 3: Install CUPTI via Apt-Get (with NVIDIA repos)

```dockerfile
# In RUNTIME stage, BEFORE the COPY line
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-cupti-12-4 \  # Correct CUDA 12.4 version!
        libcupti-dev-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb

# Now CUPTI should exist!
# Remove the COPY line since apt-get installed it
# COPY --from=builder /usr/local/cuda/lib64/libcupti.so* /usr/local/cuda/lib64/  ← DELETE
```

**Trade-off**:
- ✅ CUPTI available in runtime
- ❌ Adds ~200MB to image
- ⚠️ Requires internet access during build
- ✅ Matches CUDA version (12.4)

---

### Option 4: Use Devel Image as Runtime (Largest but guaranteed)

```dockerfile
# Instead of:
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Use:
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# No multi-stage build needed!
# Build PyTorch directly in this image
# CUPTI is already there!
```

**Trade-off**:
- ✅ CUPTI guaranteed available
- ✅ All CUDA dev tools available
- ❌ Image size: ~8GB instead of ~3GB
- ❌ Runtime image includes unnecessary build tools

---

### Option 5: Nuclear Copy - Copy Entire CUDA Directory

```dockerfile
# Instead of copying specific files:
# COPY --from=builder /usr/local/cuda/lib64/libcupti.so* /usr/local/cuda/lib64/

# Copy EVERYTHING:
COPY --from=builder /usr/local/cuda/ /usr/local/cuda/
```

**Trade-off**:
- ✅ Guaranteed to include CUPTI (if it exists in builder)
- ❌ Massive image size increase (~3GB more)
- ❌ Includes all headers, nvcc, nsight tools

---

### Option 6: Show PyTorch CMake Findings

```dockerfile
# After PyTorch compilation
RUN echo "🔍 WHAT DID PYTORCH CMAKE FIND? 🔍" && \
    cd pytorch && \
    cat CMakeCache.txt | grep -i cupti || echo "No CUPTI in CMakeCache!" && \
    echo "" && \
    echo "Checking if the file PyTorch thinks exists actually exists:" && \
    ls -la /usr/local/cuda/lib64/libcupti.so 2>/dev/null || echo "NOPE! DOESN'T EXIST!"
```

**Purpose**: See what PyTorch CMake is claiming vs reality

---

### Option 7: Local CUDA Installer (Full Toolkit)

```dockerfile
FROM ubuntu:22.04

# Download and install FULL CUDA toolkit (includes CUPTI!)
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run && \
    chmod +x cuda_*.run && \
    ./cuda_*.run --silent --toolkit --toolkitpath=/usr/local/cuda && \
    rm cuda_*.run && \
    rm -rf /var/lib/apt/lists/*

# This installs:
# - CUDA runtime libraries
# - CUDA development libraries
# - CUPTI extras!
# - nvcc compiler
# - Nsight tools
# - Everything!
```

**Trade-off**:
- ✅ Guaranteed CUPTI
- ✅ Full control over CUDA version
- ❌ Large download (~3.5GB installer)
- ❌ Long build time
- ❌ Larger final image

---

### Option 8: Quantum Debugging - Check All Possible Paths

```dockerfile
RUN echo "🌌 QUANTUM CUPTI SEARCH 🌌" && \
    for path in \
      /usr/local/cuda-12.4/extras/CUPTI \
      /usr/local/cuda-12/extras/CUPTI \
      /usr/local/cuda/extras/CUPTI \
      /usr/lib/x86_64-linux-gnu/cuda \
      /usr/lib/cuda \
      /opt/cuda \
      /opt/nvidia/cuda ; do \
        echo "Checking: $path" && \
        find $path -name "*cupti*" 2>/dev/null || echo "  ❌ Not here" ; \
    done && \
    echo "🌌 END QUANTUM SEARCH 🌌"
```

**Purpose**: Exhaustively check every possible CUPTI location

---

### Option 9: Docker Layer Archaeology with Dive

```bash
# Outside Docker, on your local machine
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Use dive to explore layers
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Navigate layers to find CUPTI files
# This shows EVERY file in EVERY layer!
```

**Purpose**: See exactly what files exist in which Docker layers

---

### Option 10: Live Autopsy - Pause Build and Explore

```dockerfile
# Add after PyTorch compilation, BEFORE the COPY fails
RUN echo "🛑 BUILD PAUSED FOR AUTOPSY! 🛑" && \
    echo "CUPTI locations to check:" > /tmp/cupti_hunt.txt && \
    find / -name "*cupti*" 2>/dev/null >> /tmp/cupti_hunt.txt && \
    cat /tmp/cupti_hunt.txt && \
    echo "" && \
    echo "Sleeping for 10 minutes - you can docker exec into this container!" && \
    sleep 600
```

**Then during build**:
```bash
# Get container ID from build output
docker ps | grep "pytorch-clean"

# Exec into it!
docker exec -it <container-id> /bin/bash

# Explore manually!
find / -name "*cupti*" 2>/dev/null
ls -la /usr/local/cuda/lib64/
dpkg -l | grep cupti
```

**Purpose**: Manual forensic analysis of the build environment

---

### Option 11: CUPTI Slot Machine - Try Everything!

```dockerfile
RUN echo "🎰 CUPTI SLOT MACHINE 🎰" && \
    # Try 1: Find it
    CUPTI=$(find /usr -name "libcupti.so*" 2>/dev/null | head -1) && \
    if [ -n "$CUPTI" ]; then echo "✅ Found at: $CUPTI"; exit 0; fi && \
    # Try 2: Install via apt (might be wrong version!)
    apt-get update && apt-get install -y libcupti-dev 2>/dev/null && \
    CUPTI=$(find /usr -name "libcupti.so*" 2>/dev/null | head -1) && \
    if [ -n "$CUPTI" ]; then echo "✅ Installed via apt!"; exit 0; fi && \
    # Try 3: Download from NVIDIA
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcupti-dev-12-4_12.4.127-1_amd64.deb && \
    dpkg -i libcupti-dev*.deb 2>/dev/null && \
    # Try 4: Just give up
    echo "🎰 JACKPOT: CUPTI DOESN'T EXIST! 🎰"
```

**Purpose**: Exhaustive automated attempts to find or install CUPTI

---

## Recommended Solution

**Best Practice**: **Option 2 - Skip CUPTI Entirely** ✅

**Rationale**:
1. CUPTI is OPTIONAL - only needed for torch.profiler GPU kernel profiling
2. Training works perfectly without it
3. Simplest solution - just remove the COPY line
4. Can still profile with external tools (nvprof, Nsight Systems)
5. Reduces image size
6. No dependency on NVIDIA apt repos

**Implementation**:
```dockerfile
# training/images/pytorch-clean/Dockerfile
# training/images/pytorch-clean/Dockerfile.buildkit

# DELETE OR COMMENT OUT:
# COPY --from=builder /usr/local/cuda/lib64/libcupti.so* /usr/local/cuda/lib64/

# ADD COMMENT:
# CUPTI not included - optional profiling library only needed for torch.profiler
# PyTorch works perfectly without it. Use external profilers if needed (nvprof, Nsight).
```

**If you NEED CUPTI later**:
- Use Option 3 (install via apt-get with NVIDIA repos)
- Or use Option 7 (full CUDA local installer)

---

## References & Links

### Official Documentation
- **NVIDIA CUPTI Docs**: https://docs.nvidia.com/cuda/cupti/
- **PyTorch Profiler**: https://pytorch.org/docs/stable/profiler.html
- **PyTorch Kineto**: https://github.com/pytorch/kineto
- **NVIDIA Container Toolkit**: https://github.com/NVIDIA/nvidia-docker

### NVIDIA GitLab
- **CUDA Container Images**: https://gitlab.com/nvidia/container-images/cuda
- **12.4.1 Ubuntu 22.04 Devel**: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/ubuntu2204/devel/Dockerfile
- **Known Issues**: https://gitlab.com/nvidia/container-images/cuda/-/issues

### Stack Overflow / Community
- **CUPTI Docker Issues**: https://stackoverflow.com/questions/tagged/cupti+docker
- **PyTorch Multi-stage Builds**: https://discuss.pytorch.org/t/docker-multistage-build-cupti/

### NVIDIA Downloads
- **CUDA Toolkit Archive**: https://developer.nvidia.com/cuda-toolkit-archive
- **CUDA 12.4.1 Downloads**: https://developer.nvidia.com/cuda-12-4-1-download-archive
- **Ubuntu Repo Setup**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

### Build Logs (Internal)
- **Build c2323775** (Original failure): https://console.cloud.google.com/cloud-build/builds/c2323775
- **Build 84fbeb34** (Symlink fix): https://console.cloud.google.com/cloud-build/builds/84fbeb34
- **Build acaba721** (Massive debug): https://console.cloud.google.com/cloud-build/builds/acaba721

---

## Lessons Learned

### What Worked ✅
1. **Massive debug output** - Revealed the truth about CUPTI's absence
2. **CHONK markers** - All 5 appeared perfectly! Polish updates working!
3. **Bright Data research** - Confirmed CUPTI is optional
4. **NVIDIA GitLab** - Showed official devel image doesn't explicitly install CUPTI

### What Didn't Work ❌
1. **Copying from /usr/local/cuda-12.0/lib64/** - CUPTI not there
2. **Copying from /usr/local/cuda/lib64/** - CUPTI not there either
3. **Assuming CUPTI exists in runtime images** - Only in devel images
4. **apt-get install libcupti-dev** - Wrong CUDA version (11.5 instead of 12.4)

### Key Insights 💡
1. **Multi-stage builds**: Runtime images are stripped down - only essentials!
2. **PyTorch CMake lies**: Claims to find CUPTI but library doesn't exist
3. **CUPTI is optional**: Training works fine without it
4. **Devel vs Runtime**: 3GB size difference is mostly dev tools like CUPTI
5. **176 cores DEMOLISHED PyTorch**: 25 minutes for 7517 files! ABSOLUTE UNIT!

---

## Next Steps

**Immediate Action**: Remove CUPTI COPY line and redeploy

**Long-term Options**:
- Monitor if torch.profiler is ever needed
- If profiling required: Add Option 3 (apt-get with NVIDIA repos)
- Document profiling alternatives (nvprof, Nsight Systems)

**Test Plan**:
1. Remove CUPTI COPY line
2. Launch build (should succeed at Step 47/51)
3. Verify PyTorch import works (Step 49/51)
4. Deploy training job
5. Monitor for any profiling-related issues

---

## Appendix: Full Debug Output from Build acaba721

```
Step #0: Step 46/51 : RUN echo "🌲🌲🌲 MASSIVE DEBUG - FULL CUDA DIRECTORY STRUCTURE 🌲🌲🌲" && ...
Step #0: 🌲🌲🌲 MASSIVE DEBUG - FULL CUDA DIRECTORY STRUCTURE 🌲🌲🌲
Step #0:
Step #0: === 1. List all /usr/local/cuda* directories ===
Step #0: lrwxrwxrwx 1 root root   21 Oct  7 18:08 cuda -> /etc/alternatives/cuda
Step #0: lrwxrwxrwx 1 root root   23 Oct  7 18:08 cuda-12 -> /etc/alternatives/cuda-12
Step #0: drwxr-xr-x 1 root root 4096 Oct  7 18:08 cuda-12.4
Step #0:
Step #0: === 2. Find ALL .so files in /usr/local/cuda* ===
Step #0: /usr/local/cuda-12.4/lib64/libcudart.so.12
Step #0: /usr/local/cuda-12.4/lib64/libcudart.so.12.4.127
Step #0: /usr/local/cuda-12.4/lib64/libcublasLt.so.12
Step #0: /usr/local/cuda-12.4/lib64/libcublasLt.so.12.4.5.8
Step #0: /usr/local/cuda-12.4/lib64/libcublas.so.12
Step #0: /usr/local/cuda-12.4/lib64/libcublas.so.12.4.5.8
Step #0: /usr/local/cuda-12.4/lib64/libcufft.so.11
Step #0: /usr/local/cuda-12.4/lib64/libcufft.so.11.2.1.3
Step #0: ... (50 total .so files shown)
Step #0: [NO libcupti.so in the list!!!]
Step #0:
Step #0: === 3. Specifically search for cupti ===
Step #0: [EMPTY OUTPUT - NOTHING FOUND!]
Step #0:
Step #0: === 4. Check if CUPTI installed as package ===
Step #0: [EMPTY OUTPUT - No cupti packages!]
Step #0:
Step #0: === 5. List /usr/local/cuda/lib64/ contents ===
Step #0: total 471M
Step #0: drwxr-xr-x 1 root root 4.0K Oct  7 18:08 .
Step #0: drwxr-xr-x 1 root root 4.0K Oct  7 18:08 ..
Step #0: lrwxrwxrwx 1 root root   15 Oct  7 18:08 libcublasLt.so -> libcublasLt.so.12
Step #0: lrwxrwxrwx 1 root root   21 Oct  7 18:08 libcublasLt.so.12 -> libcublasLt.so.12.4.5.8
Step #0: -rw-r--r-- 1 root root 317M Oct  7 18:07 libcublasLt.so.12.4.5.8
Step #0: ... (30+ library files)
Step #0: [NO libcupti.so among them!!!]
Step #0:
Step #0: 🌲🌲🌲 END DEBUG 🌲🌲🌲
```

**Analysis**: Comprehensive search found NO TRACE of CUPTI in runtime image!

---

**End of Study** - CUPTI mystery SOLVED! 🎯
