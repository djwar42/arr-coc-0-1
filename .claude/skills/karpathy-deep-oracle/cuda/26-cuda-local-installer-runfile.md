# CUDA Local Installer (.run file) - Installation Guide

**Comprehensive guide to CUDA runfile installation as alternative to apt-get packages**

From CUPTI investigation: Runfile installation provides exact version control and selective component installation for Docker builds.

---

## Section 1: Runfile vs Apt-Get Comparison (~120 lines)

### Distribution Methods Overview

**Two primary CUDA installation approaches on Ubuntu:**

| Method | Distribution | Control | Version |
|--------|-------------|---------|---------|
| **Runfile** | NVIDIA official | High | Exact version (12.4.0) |
| **Apt-get** | Ubuntu repos | Medium | Repository version |
| **Runfile** | Downloaded manually | Component selection | Latest available |
| **Apt-get** | Package manager | Pre-configured | May lag behind |

From [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13):
> "The runfile installation method provides the greatest control over the installation but requires manual intervention. The package manager installation method provides easier installation and upgrade management."

### Key Differences

#### Installation Location

**Runfile:**
```bash
# Default installation path
/usr/local/cuda-12.4/
├── bin/          # nvcc, cuda-gdb
├── include/      # Headers
├── lib64/        # Shared libraries
└── extras/
    └── CUPTI/    # CUPTI headers and libraries
```

**Apt-get:**
```bash
# Apt installs to standard system paths
/usr/bin/nvcc          # Compiler in system bin
/usr/include/cuda/     # Headers in system include
/usr/lib/x86_64-linux-gnu/  # Libraries in system lib
```

**Key difference**: Runfile isolates CUDA in `/usr/local/cuda-X.Y/`, allowing multiple versions. Apt-get integrates into system paths.

#### Version Control

**Runfile advantages:**
- Install exact CUDA version (12.4.0 vs 12.4.1 matters)
- Multiple CUDA versions coexist (`/usr/local/cuda-12.4/`, `/usr/local/cuda-12.6/`)
- Pin to specific driver version (550.54.14)

**Apt-get limitations:**
- Repository may lag (Ubuntu 22.04 had CUDA 11.x when 12.x released)
- System-wide installation (single version)
- Driver updates controlled by package dependencies

From [Ask Ubuntu discussion](https://askubuntu.com/questions/368927/difference-between-installing-cuda-using-nvidia-cuda-toolkit-and-the-run-file) (accessed 2025-11-13):
> "One big difference I have experienced is that the CUDA sample codes are missing when installing through apt-get."

#### Root Privileges Required

**Both methods require sudo:**
```bash
# Runfile
sudo sh cuda_12.4.0_550.54.14_linux.run

# Apt-get
sudo apt-get install cuda-toolkit-12-4
```

No difference in privilege requirements - CUDA installs system-level components (drivers, libraries).

### Component Installation Flexibility

**Runfile allows selective installation:**
```bash
sudo sh cuda_12.4.0_linux.run \
    --toolkit \              # CUDA toolkit only
    --no-drm \              # Skip DRM kernel module
    --no-man-page \         # Skip documentation
    --no-opengl-libs        # Skip OpenGL libraries
```

**Apt-get installs predefined packages:**
```bash
# Install specific components via package selection
sudo apt-get install cuda-libraries-12-4        # Runtime only
sudo apt-get install cuda-libraries-dev-12-4    # + CUPTI
sudo apt-get install cuda-toolkit-12-4          # Everything
```

**Flexibility comparison:**
- **Runfile**: Fine-grained flags control what gets installed
- **Apt-get**: Package-level granularity (coarser control)

### Docker Build Context

**Runfile in Dockerfiles:**
```dockerfile
FROM ubuntu:22.04 AS builder

# Download runfile
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Silent installation
RUN sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit --no-man-page

# Selective copy to runtime
FROM ubuntu:22.04
COPY --from=builder /usr/local/cuda-12.4 /usr/local/cuda
```

**Apt-get in Dockerfiles:**
```dockerfile
FROM ubuntu:22.04

# Add NVIDIA repository
RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA packages
RUN apt-get update && apt-get install -y cuda-toolkit-12-4
```

**Build size comparison:**
- **Runfile**: 3-4GB download, selective install reduces to 1.5-2GB
- **Apt-get**: Package dependencies add overhead, but `--no-install-recommends` helps

From [DEV Community - Install NVIDIA CUDA on Linux](https://dev.to/bybatkhuu/install-nvidia-cuda-on-linux-1040) (accessed 2025-11-13):
> "IMPORTANT! Download and use the .RUN file! It can prevent installing incompatible NVIDIA drivers and gives you more control over the installation."

---

## Section 2: Runfile Installation Workflow (~150 lines)

### Step 1: Download CUDA Runfile

**NVIDIA Developer Download Page:**

Visit: https://developer.nvidia.com/cuda-downloads

Select:
- Operating System: Linux
- Architecture: x86_64
- Distribution: Ubuntu
- Version: 22.04
- Installer Type: **runfile (local)**

**Download via wget:**
```bash
# CUDA 12.4.0 with driver 550.54.14
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# File size: ~3.5GB
# MD5 checksum available on download page
```

**Verify download integrity:**
```bash
md5sum cuda_12.4.0_550.54.14_linux.run
# Compare with official checksum
```

From [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) (accessed 2025-11-13):
- CUDA 12.4.0: Released March 2024
- Includes cuDNN, cuBLAS, CUPTI
- Driver version 550.54.14 bundled

### Step 2: Runfile Permissions

**Make executable:**
```bash
chmod +x cuda_12.4.0_550.54.14_linux.run
```

**Why needed**: Downloaded files default to non-executable (644 permissions).

### Step 3: Interactive Installation

**Run with GUI (X11 required):**
```bash
sudo ./cuda_12.4.0_550.54.14_linux.run
```

**Interactive prompts:**
1. Accept EULA (read and type "accept")
2. Choose components:
   - Driver (yes/no)
   - CUDA Toolkit (yes/no)
   - Samples (yes/no)
   - Documentation (yes/no)
3. Installation path (default: `/usr/local/cuda-12.4`)
4. Symbolic link `/usr/local/cuda` → `/usr/local/cuda-12.4` (yes/no)

**Interactive mode good for:**
- First-time installation
- Desktop/laptop development machines
- Understanding what gets installed

### Step 4: Silent Installation

**Non-interactive (Docker, automation):**
```bash
sudo sh cuda_12.4.0_550.54.14_linux.run \
    --silent \              # No prompts
    --toolkit               # Install toolkit only
```

From [Ask Ubuntu - NVIDIA CUDA silent install](https://askubuntu.com/questions/562667/how-to-install-nvidia-cuda-without-eula-prompts) (accessed 2025-11-13):
> "According to the installer's built-in help text, you should be able to do that by adding the command-line option -silent"

**Critical flags:**
- `--silent`: Accepts EULA automatically, no user interaction
- `--override`: Override GCC version checks (for newer compilers)

### Step 5: Component Selection

**Available flags:**
```bash
# Show all available options
sudo sh cuda_12.4.0_linux.run --help

# Common selective flags:
--toolkit               # CUDA compiler, libraries, headers
--samples              # CUDA sample code
--no-drm               # Skip DRM kernel module
--no-man-page          # Skip manual pages (saves space)
--no-opengl-libs       # Skip OpenGL interop libraries
--no-cuda-11-compat    # Skip CUDA 11 compatibility layer
```

**Minimal installation (toolkit only):**
```bash
sudo sh cuda_12.4.0_550.54.14_linux.run \
    --silent \
    --toolkit \
    --no-drm \
    --no-man-page \
    --no-opengl-libs \
    --installpath=/usr/local/cuda-12.4
```

**Result**: ~1.5GB installation (vs 3GB full install)

### Step 6: Driver Handling

**Important**: Runfile includes NVIDIA driver (550.54.14 in this case).

**Skip driver if already installed:**
```bash
# Check existing driver
nvidia-smi

# If driver already present, skip it
sudo sh cuda_12.4.0_linux.run \
    --silent \
    --toolkit \
    --no-drm        # This skips driver components
```

**Install driver from runfile:**
```bash
# Fresh system, need driver + toolkit
sudo sh cuda_12.4.0_linux.run --silent
# Installs both driver and toolkit
```

**Docker context**: Never install driver in Docker (host provides driver via nvidia-docker).

### Step 7: Verification

**Check installation:**
```bash
# Verify installation location
ls /usr/local/cuda-12.4/

# Check nvcc compiler
/usr/local/cuda-12.4/bin/nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.4, V12.4.131
```

**Environment setup:**
```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc

# Verify
nvcc --version
```

### Step 8: Symbolic Link Management

**Default symlink:**
```bash
# Runfile creates:
/usr/local/cuda -> /usr/local/cuda-12.4

# Check symlink
ls -l /usr/local/cuda
```

**Multiple CUDA versions:**
```bash
# Install CUDA 12.4
sudo sh cuda_12.4.0_linux.run --silent --toolkit

# Install CUDA 12.6 (different version)
sudo sh cuda_12.6.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.6

# Switch between versions:
sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda  # Use 12.4
sudo ln -sf /usr/local/cuda-12.6 /usr/local/cuda  # Use 12.6
```

From [Medium - How to Install Multiple CUDA Versions on WSL2](https://medium.com/@dynotes/how-to-install-multiple-cuda-versions-on-wsl2-ubuntu-22-04-b9c6fc51f252) (accessed 2025-11-13):
> "The recommended approach is to use the runfile installers for CUDA 12.4 and 12.6, ensuring the NVIDIA driver is not installed."

---

## Section 3: Docker Integration (~120 lines)

### Multi-Stage Build with Runfile

**Complete Dockerfile example:**

```dockerfile
# ============================================
# Builder stage: Download and install CUDA
# ============================================
FROM ubuntu:22.04 AS builder

# Install wget
RUN apt-get update && apt-get install -y wget

# Download CUDA runfile
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Silent installation (toolkit only, no driver)
RUN sh cuda_12.4.0_550.54.14_linux.run \
    --silent \
    --toolkit \
    --no-drm \
    --no-man-page \
    --installpath=/usr/local/cuda-12.4

# ============================================
# Runtime stage: Copy selective CUDA files
# ============================================
FROM ubuntu:22.04

# Copy entire CUDA installation
COPY --from=builder /usr/local/cuda-12.4 /usr/local/cuda

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Verify installation
RUN nvcc --version
```

**Size breakdown:**
- Builder stage: ~4.5GB (Ubuntu + runfile + installed CUDA)
- Runtime stage: ~2GB (Ubuntu + copied CUDA toolkit)
- Final image: 2GB (builder discarded)

### Selective Component Copy

**Instead of copying entire CUDA:**

```dockerfile
# Runtime stage with selective copy
FROM ubuntu:22.04

# Copy only runtime libraries
COPY --from=builder /usr/local/cuda-12.4/lib64/*.so* /usr/local/cuda/lib64/

# Copy compiler (if needed)
COPY --from=builder /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda/bin/

# Copy headers (if compiling in runtime)
COPY --from=builder /usr/local/cuda-12.4/include/ /usr/local/cuda/include/

# Copy CUPTI specifically
COPY --from=builder /usr/local/cuda-12.4/extras/CUPTI /usr/local/cuda/extras/CUPTI
```

**Result**: ~1GB runtime image (vs 2GB full copy)

### Runfile Dockerfile Patterns

**Pattern 1: Toolkit in builder, libraries in runtime**
```dockerfile
# Builder: Compile PyTorch with CUDA
FROM ubuntu:22.04 AS builder
RUN sh cuda_12.4.0_linux.run --silent --toolkit
# ... build PyTorch ...

# Runtime: Only CUDA libraries needed
FROM ubuntu:22.04
COPY --from=builder /usr/local/cuda-12.4/lib64 /usr/local/cuda/lib64
COPY --from=builder /build/pytorch/build/lib /app/lib
```

**Pattern 2: Download once, use in stages**
```dockerfile
FROM ubuntu:22.04 AS downloader
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_linux.run

FROM downloader AS builder
RUN sh cuda_12.4.0_linux.run --silent --toolkit

FROM ubuntu:22.04
COPY --from=builder /usr/local/cuda-12.4 /usr/local/cuda
```

**Pattern 3: ARG for version flexibility**
```dockerfile
ARG CUDA_VERSION=12.4.0
ARG CUDA_DRIVER=550.54.14

FROM ubuntu:22.04 AS builder
RUN wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_${CUDA_DRIVER}_linux.run
RUN sh cuda_${CUDA_VERSION}_${CUDA_DRIVER}_linux.run --silent --toolkit

# Build with different CUDA version:
# docker build --build-arg CUDA_VERSION=12.6.0 --build-arg CUDA_DRIVER=560.28.03 .
```

### Non-Interactive Installation Requirements

**Critical for Docker:**
```bash
# All these flags required for automated builds:
--silent              # Accept EULA, no prompts
--toolkit            # Specify what to install
--override           # Override compiler checks
--no-drm             # Skip driver components (Docker doesn't need)
```

**Without --silent, installation hangs:**
```dockerfile
# ❌ BAD - Will hang waiting for EULA acceptance
RUN sh cuda_12.4.0_linux.run --toolkit

# ✅ GOOD - Non-interactive
RUN sh cuda_12.4.0_linux.run --silent --toolkit
```

### Size Optimization Strategies

**Strategy 1: Minimal toolkit installation**
```bash
# Install only essential components
RUN sh cuda_12.4.0_linux.run \
    --silent \
    --toolkit \
    --no-man-page \          # Save 50MB
    --no-opengl-libs \       # Save 100MB
    --no-cuda-11-compat      # Save 200MB
# Result: 1.5GB vs 3GB full install
```

**Strategy 2: Extract only needed files**
```dockerfile
# Install in builder, extract to tarball
FROM ubuntu:22.04 AS builder
RUN sh cuda_12.4.0_linux.run --silent --toolkit
RUN tar czf /tmp/cuda-minimal.tar.gz \
    /usr/local/cuda-12.4/lib64/*.so.* \
    /usr/local/cuda-12.4/bin/nvcc

FROM ubuntu:22.04
COPY --from=builder /tmp/cuda-minimal.tar.gz /tmp/
RUN tar xzf /tmp/cuda-minimal.tar.gz -C / && rm /tmp/cuda-minimal.tar.gz
```

**Strategy 3: Multi-stage with apt cleanup**
```dockerfile
FROM ubuntu:22.04 AS builder
# Install dependencies for runfile
RUN apt-get update && apt-get install -y wget
RUN sh cuda_12.4.0_linux.run --silent --toolkit
# Don't worry about cleanup in builder

FROM ubuntu:22.04
COPY --from=builder /usr/local/cuda-12.4 /usr/local/cuda
# Runtime has clean apt cache automatically
```

From [NVIDIA Developer Forums - Docker CUDA Installation](https://forums.developer.nvidia.com/t/unable-to-get-this-to-install-cuda-12-4-0-550-54-14-linux-run/287094) (accessed 2025-11-13):
> "Install of driver component failed. Docker containers should use --no-drm flag to skip driver installation."

---

## Section 4: CUPTI with Runfile (~110 lines)

### CUPTI Included in Runfile Toolkit

**Key discovery from arr-coc-0-1 CUPTI investigation:**

Runfile CUDA toolkit installation includes CUPTI by default:

```bash
# After runfile installation:
/usr/local/cuda-12.4/
└── extras/
    └── CUPTI/
        ├── include/
        │   ├── cupti.h
        │   ├── cupti_callbacks.h
        │   ├── cupti_events.h
        │   └── ... (all CUPTI headers)
        ├── lib64/
        │   ├── libcupti.so -> libcupti.so.12
        │   ├── libcupti.so.12 -> libcupti.so.12.4.127
        │   ├── libcupti.so.12.4.127
        │   └── libcupti_static.a
        └── samples/
            └── ... (CUPTI sample code)
```

**Compare with apt-get**: Requires separate `cuda-libraries-dev-12-4` package for CUPTI.

### CUPTI File Locations

**Headers (for compilation):**
```bash
/usr/local/cuda-12.4/extras/CUPTI/include/
├── cupti.h                    # Main CUPTI header
├── cupti_activity.h          # Activity API
├── cupti_callbacks.h         # Callback API
├── cupti_driver_cbid.h       # Driver callbacks
├── cupti_events.h            # Event API
├── cupti_metrics.h           # Metrics API
├── cupti_nvtx.h              # NVTX integration
└── cupti_profiler_target.h   # Profiler API
```

**Shared libraries (for runtime):**
```bash
/usr/local/cuda-12.4/extras/CUPTI/lib64/
├── libcupti.so                  # Symlink to versioned
├── libcupti.so.12               # Symlink to patch version
├── libcupti.so.12.4.127         # Actual library
└── libcupti_static.a            # Static library
```

**Size:**
- libcupti.so.12.4.127: ~15MB
- libcupti_static.a: ~20MB
- Headers: ~2MB
- Total CUPTI: ~37MB

### LD_LIBRARY_PATH Configuration

**Environment setup after runfile install:**

```bash
# Add CUPTI to library path
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Or add to system library cache
echo "/usr/local/cuda-12.4/extras/CUPTI/lib64" | sudo tee /etc/ld.so.conf.d/cupti.conf
sudo ldconfig
```

**Verify CUPTI found:**
```bash
# Check if libcupti.so is discoverable
ldconfig -p | grep cupti

# Expected output:
# libcupti.so.12 (libc6,x86-64) => /usr/local/cuda-12.4/extras/CUPTI/lib64/libcupti.so.12
```

**Python/PyTorch verification:**
```python
import torch

# Will fail if libcupti.so not in LD_LIBRARY_PATH
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    # ... profiling code ...
    pass
```

### Docker Copy Strategy for CUPTI

**Pattern 1: Copy entire CUPTI directory**
```dockerfile
FROM builder AS runtime
COPY --from=builder /usr/local/cuda-12.4/extras/CUPTI /usr/local/cuda/extras/CUPTI

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

**Pattern 2: Copy only runtime library**
```dockerfile
FROM builder AS runtime
# Copy just the .so file
COPY --from=builder /usr/local/cuda-12.4/extras/CUPTI/lib64/libcupti.so* /usr/local/cuda/lib64/

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Pattern 3: Conditional CUPTI installation**
```dockerfile
ARG ENABLE_PROFILING=0

FROM builder AS runtime
# Only copy CUPTI if profiling enabled
RUN if [ "$ENABLE_PROFILING" = "1" ]; then \
        cp -r /usr/local/cuda-12.4/extras/CUPTI /usr/local/cuda/extras/; \
    fi
```

**Build with profiling:**
```bash
docker build --build-arg ENABLE_PROFILING=1 -t app:profiling .
```

**Build without profiling (saves 37MB):**
```bash
docker build --build-arg ENABLE_PROFILING=0 -t app:production .
```

### CUPTI Runfile vs Apt-Get Comparison

| Aspect | Runfile | Apt-get |
|--------|---------|---------|
| **Location** | `/usr/local/cuda/extras/CUPTI/` | `/usr/include/`, `/usr/lib/` |
| **Package** | Included in toolkit | Separate `cuda-libraries-dev` |
| **Version control** | Exact CUDA version | Repository version |
| **LD_LIBRARY_PATH** | Must add `/extras/CUPTI/lib64` | Standard `/usr/lib` path |
| **Docker copy** | Copy `/extras/CUPTI` directory | Extract from apt package |
| **Size optimization** | Easy (copy or don't) | Requires apt extraction |

**Runfile advantage for CUPTI**: Single directory, easy to include or exclude.

**Apt-get advantage for CUPTI**: Standard library paths, no LD_LIBRARY_PATH tweaking.

### PyTorch Build with Runfile CUPTI

**Critical for arr-coc-0-1 context:**

```dockerfile
# Builder stage
FROM ubuntu:22.04 AS builder

# Install CUDA via runfile (includes CUPTI)
RUN sh cuda_12.4.0_linux.run --silent --toolkit

# CMake will find CUPTI headers
RUN cmake -S pytorch -B build \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 \
    -DUSE_CUPTI=ON

# PyTorch builds with profiling support
RUN cmake --build build

# Runtime stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy CUPTI library (not just headers!)
COPY --from=builder /usr/local/cuda-12.4/extras/CUPTI/lib64/libcupti.so* /usr/local/cuda/lib64/

# Copy PyTorch
COPY --from=builder /build/lib /app/lib

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Now torch.profiler works
RUN python -c "import torch; torch.profiler.profile"
```

**Key insight**: Runfile places CUPTI in predictable location, easier to copy selectively than extracting from apt packages.

---

## Section 5: Production Best Practices (~100 lines)

### When to Use Runfile

**Runfile recommended for:**

1. **Exact version control required**
   - Production systems need CUDA 12.4.0 (not 12.4.1)
   - Research reproducibility (paper uses CUDA 12.4.0)
   - Multiple CUDA versions on same system

2. **Docker multi-stage builds**
   - Install in builder, selective copy to runtime
   - Control exactly what gets copied
   - Minimize runtime image size

3. **Custom component selection**
   - Need CUPTI but not samples
   - Skip OpenGL libraries (server deployment)
   - Exclude documentation (production)

4. **Bleeding-edge CUDA versions**
   - Latest CUDA release before apt-get packaging
   - Ubuntu repository lags by months

5. **CI/CD automation**
   - Scriptable with --silent flag
   - No repository configuration needed
   - Self-contained installation

**Runfile NOT recommended for:**
- Quick prototyping (apt-get faster)
- System-wide development machines (apt-get easier updates)
- Cloud VMs with auto-updates (package manager integration)

### Apt-Get When Appropriate

**Use apt-get for:**

1. **Desktop development machines**
   - System-wide CUDA installation
   - Automatic security updates
   - Package manager integration

2. **Standard Ubuntu deployments**
   - Ubuntu 22.04 with CUDA 12.x
   - Repository versions sufficient
   - Automatic dependency management

3. **Quick testing**
   - Fast installation (`apt-get install cuda`)
   - Standard library paths (no LD_LIBRARY_PATH)

4. **Cloud VM images**
   - GCP, AWS, Azure provide CUDA images via apt
   - Auto-patching via package manager

### Version Pinning Strategy

**Pin exact CUDA version with runfile:**
```bash
# Download specific version permanently
CUDA_VERSION=12.4.0
CUDA_DRIVER=550.54.14

wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_${CUDA_DRIVER}_linux.run

# Store runfile in artifact repository
# Install from artifact (guarantees version)
```

**Pin with apt-get (more complex):**
```bash
# Install specific version
sudo apt-get install cuda-toolkit-12-4=12.4.0-1

# Hold version (prevent upgrades)
sudo apt-mark hold cuda-toolkit-12-4
```

**Runfile advantage**: Self-contained binary, version never changes.

### Multi-CUDA-Version Workflow

**Development machine with multiple CUDA versions:**

```bash
# Install CUDA 12.4 via runfile
sudo sh cuda_12.4.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.4

# Install CUDA 12.6 via runfile
sudo sh cuda_12.6.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-12.6

# Install CUDA 11.8 via runfile
sudo sh cuda_11.8.0_linux.run --silent --toolkit --installpath=/usr/local/cuda-11.8

# Create environment modules
module load cuda/12.4   # Sets PATH, LD_LIBRARY_PATH for 12.4
module load cuda/12.6   # Switches to 12.6
module load cuda/11.8   # Switches to 11.8
```

**Project-specific CUDA:**
```bash
# Project A uses CUDA 12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Project B uses CUDA 12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

**Not possible with apt-get**: Single system-wide version only.

### Troubleshooting Runfile Installation

**Common issues:**

**Issue 1: Driver already installed**
```bash
# Error: Existing driver detected
# Solution: Skip driver installation
sudo sh cuda_12.4.0_linux.run --silent --toolkit --no-drm
```

**Issue 2: GCC version mismatch**
```bash
# Error: gcc 11 not supported, CUDA requires gcc <= 10
# Solution: Override check (if you know it's compatible)
sudo sh cuda_12.4.0_linux.run --silent --toolkit --override
```

**Issue 3: X11 not available (SSH session)**
```bash
# Error: Cannot display EULA
# Solution: Use --silent flag
sudo sh cuda_12.4.0_linux.run --silent --toolkit
```

**Issue 4: LD_LIBRARY_PATH not set**
```bash
# Symptom: nvcc: command not found
# Solution: Add to PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH

# Symptom: libcudart.so not found
# Solution: Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Issue 5: Multiple CUDA versions conflict**
```bash
# Symptom: Wrong CUDA version active
# Solution: Check symlink
ls -l /usr/local/cuda

# Fix symlink
sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
```

### Uninstallation

**Remove CUDA installed via runfile:**
```bash
# Navigate to CUDA directory
cd /usr/local/cuda-12.4

# Run uninstaller (if exists)
sudo bin/cuda-uninstaller

# Manual removal
sudo rm -rf /usr/local/cuda-12.4
sudo rm /usr/local/cuda  # Remove symlink
```

**Clean environment:**
```bash
# Remove from ~/.bashrc
# Delete these lines:
# export PATH=/usr/local/cuda-12.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Compare with apt-get uninstall:**
```bash
sudo apt-get remove cuda-toolkit-12-4
sudo apt-get autoremove
```

---

## Sources

**Official Documentation:**
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (accessed 2025-11-13)
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) (accessed 2025-11-13)

**Community Resources:**
- [Ask Ubuntu - Difference between runfile and apt-get CUDA installation](https://askubuntu.com/questions/368927/difference-between-installing-cuda-using-nvidia-cuda-toolkit-and-the-run-file) (accessed 2025-11-13)
- [Ask Ubuntu - NVIDIA CUDA silent installation](https://askubuntu.com/questions/562667/how-to-install-nvidia-cuda-without-eula-prompts) (accessed 2025-11-13)
- [DEV Community - Install NVIDIA CUDA on Linux](https://dev.to/bybatkhuu/install-nvidia-cuda-on-linux-1040) (accessed 2025-11-13)
- [Medium - How to Install Multiple CUDA Versions on WSL2](https://medium.com/@dynotes/how-to-install-multiple-cuda-versions-on-wsl2-ubuntu-22-04-b9c6fc51f252) (accessed 2025-11-13)

**Related CUDA Files:**
- [cuda/18-ubuntu-pytorch-cuda-setup.md](18-ubuntu-pytorch-cuda-setup.md) - Ubuntu-specific driver installation
- [cuda/22-cuda-apt-get-installation.md](22-cuda-apt-get-installation.md) - Apt-get package-based installation (comparison)
- [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md) - Docker image variants

**arr-coc-0-1 Context:**
- CUPTI investigation revealed runfile provides predictable CUPTI location
- Multi-stage Docker builds benefit from selective component copying
- Version control critical for reproducible PyTorch builds

---

**Key Takeaways:**

1. **Runfile = exact version control** - Install CUDA 12.4.0 precisely, not "whatever's in apt"
2. **Component selection flexibility** - `--silent --toolkit --no-man-page` for minimal installs
3. **Multiple versions coexist** - `/usr/local/cuda-12.4/` and `/usr/local/cuda-12.6/` simultaneously
4. **CUPTI included** - `/extras/CUPTI/` directory with headers and libraries
5. **Docker-friendly** - Silent installation, selective copying to runtime stage
6. **LD_LIBRARY_PATH required** - Not in standard system paths like apt-get

**When to use runfile over apt-get**: Version control matters, Docker builds, multiple CUDA versions, custom component selection.
