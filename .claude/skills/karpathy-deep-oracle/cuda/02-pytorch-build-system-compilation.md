# PyTorch Build System & Compilation from Source

## Overview

PyTorch compilation from source is a complex process involving CMake configuration, C++ compilation, CUDA kernel building, and Python extension packaging. Understanding this build system is critical for optimizing CUDA architectures, enabling custom features, and reducing compilation time in production environments.

**Why Build from Source:**
- Target specific CUDA compute capabilities (sm_75, sm_80, sm_86, sm_90)
- Enable experimental features not in binary releases
- Optimize for specific hardware (A100, H100, custom GPUs)
- Debug and develop PyTorch core components
- Reduce binary size with targeted architecture builds

**Key Insight from [PyTorch Internals Part II – The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/) (PyTorch Blog, accessed 2025-02-03):**
> "The PyTorch codebase has a variety of components: The core Torch libraries (TH, THC, THNN, THCUNN), Vendor libraries (CuDNN, NCCL), Python Extension libraries, and Additional third-party libraries. A simple invocation of python setup.py install orchestrates CMake, ninja, and setuptools to build all these components."

**Related Knowledge:**
- See [cuda/00-streams-concurrency-async.md](00-streams-concurrency-async.md) for CUDA runtime APIs
- See [cuda/01-memory-management-unified.md](01-memory-management-unified.md) for memory allocation in compiled code
- See [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) for production optimization

---

## Section 1: Build System Architecture (~150 lines)

### PyTorch Build Pipeline

**Complete Build Flow:**
```
setup.py (Python)
    ↓
CMake Configuration (C++)
    ↓
Ninja Build System (Compilation)
    ↓
Backend Libraries (TH, THC, THNN, THCUNN)
    ↓
Python Extensions (torch._C, torch._dl)
    ↓
Setuptools Installation
    ↓
site-packages/torch (Installed Package)
```

From [PyTorch Internals Part II](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/):
> "The build\_all.sh script is essentially a script that runs the CMake configure step on all of these libraries, and then make install. The script creates a build directory and a subdir for each library to build."

### Core Components

**Backend Libraries (torch/lib/):**
```bash
torch/lib/
├── TH/              # Tensor library (CPU)
├── THC/             # Tensor library (CUDA)
├── THNN/            # Neural network primitives (CPU)
├── THCUNN/          # Neural network primitives (CUDA)
├── THPP/            # C++ tensor wrapper
├── libshm/          # Shared memory management
└── nccl/            # NVIDIA Collective Communications Library
```

**Python Extensions (torch/csrc/):**
```bash
torch/csrc/
├── Module.cpp           # Python module initialization
├── Tensor.cpp           # Tensor Python bindings
├── autograd/            # Automatic differentiation
├── cuda/                # CUDA-specific bindings
├── distributed/         # Distributed training
├── jit/                 # TorchScript JIT compiler
├── nn/                  # Neural network operations
└── generic/             # Template-based code generation
```

### Dependencies

**Required Build Dependencies:**
```bash
# CUDA Toolkit
CUDA 11.8+ or 12.1+
cuDNN 8.x
NCCL 2.x

# Build Tools
CMake 3.18+
Ninja (recommended) or Make
Python 3.8+
C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)

# Python Packages
pyyaml
numpy
typing_extensions

# Optional
ccache (compilation caching)
MKL (Intel Math Kernel Library)
```

### Build Modes

**CMAKE_BUILD_TYPE Options:**

| Build Type | Optimization | Debug Symbols | Use Case | Compile Time |
|-----------|--------------|---------------|----------|--------------|
| **Release** | -O3 | No | Production | ~2-4 hours |
| **Debug** | -O0 | Yes | Development | ~3-5 hours |
| **RelWithDebInfo** | -O2 | Yes | Profiling | ~2.5-4.5 hours |
| **MinSizeRel** | -Os | No | Embedded | ~2-4 hours |

**Example Configuration:**
```bash
# Release build (fastest runtime, slowest compile)
export CMAKE_BUILD_TYPE=Release

# Debug build (slower runtime, easier debugging)
export CMAKE_BUILD_TYPE=Debug

# Hybrid (good runtime, debuggable)
export CMAKE_BUILD_TYPE=RelWithDebInfo
```

---

## Section 2: CMake Configuration (~200 lines)

### CUDA Architecture Flags

**TORCH_CUDA_ARCH_LIST Environment Variable:**

From [GitHub Issue #12119](https://github.com/pytorch/pytorch/issues/12119) (accessed 2025-02-03):
> "How can I specify cuda architecture while building pytorch by python setup.py install? For instance: TORCH_CUDA_ARCH_LIST = 5.2"

**Syntax and Examples:**
```bash
# Single architecture (A100 only)
export TORCH_CUDA_ARCH_LIST="8.0"

# Multiple architectures (T4, A100, A10)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Full spectrum (V100, T4, A100, A10, H100)
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# With PTX for forward compatibility
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"
```

**Compute Capability Reference:**

| GPU Model | Architecture | Compute Capability | ARCH_LIST Value |
|-----------|-------------|-------------------|-----------------|
| V100 | Volta | sm_70 | 7.0 |
| T4 | Turing | sm_75 | 7.5 |
| A100 | Ampere | sm_80 | 8.0 |
| A10, RTX 3090 | Ampere | sm_86 | 8.6 |
| L4, RTX 40XX | Ada | sm_89 | 8.9 |
| H100 | Hopper | sm_90 | 9.0 |

**Build Time vs Binary Size Trade-off:**
```bash
# Minimal build (fastest compile, smallest binary)
# A100-only: ~1.5 hours compile, ~800MB binary
export TORCH_CUDA_ARCH_LIST="8.0"

# Multi-arch build (slower compile, larger binary)
# T4+A100+A10+H100: ~2.5 hours compile, ~2.5GB binary
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
```

### Core CMake Flags

**Essential CUDA Flags:**
```bash
# Enable CUDA support
export USE_CUDA=1

# Specify CUDA architectures
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# CUDA toolkit path (if non-standard)
export CUDA_HOME=/usr/local/cuda-12.1

# cuDNN path (if non-standard)
export CUDNN_ROOT=/usr/local/cudnn

# NCCL for multi-GPU
export USE_NCCL=1
```

**Performance Optimization Flags:**
```bash
# Build type (Release for production)
export CMAKE_BUILD_TYPE=Release

# Use Intel MKL for CPU operations
export USE_MKL=1
export MKL_ROOT=/opt/intel/mkl

# Static linking (larger binary, no runtime deps)
export USE_STATIC_MKL=1
export USE_STATIC_CUDNN=1

# Ninja build system (faster than Make)
export CMAKE_GENERATOR=Ninja
```

**Optional Feature Flags:**
```bash
# TensorRT support
export USE_TENSORRT=1

# Distributed training
export USE_DISTRIBUTED=1

# Custom CUDA flags
export CUDA_NVCC_FLAGS="-Xptxas -v"

# Parallel compilation jobs
export MAX_JOBS=32
```

### Environment Variable Reference

**Complete Build Configuration Example:**
```bash
#!/bin/bash
# PyTorch from source build configuration

# CUDA Configuration
export CUDA_HOME=/usr/local/cuda-12.1
export CUDNN_ROOT=/usr/local/cudnn-8.9
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Build Settings
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja
export MAX_JOBS=32

# Optional Libraries
export USE_NCCL=1
export USE_DISTRIBUTED=1
export USE_TENSORRT=0
export USE_MKL=1

# Compilation Cache
export USE_CCACHE=1
export CCACHE_DIR=/var/cache/ccache

# Build PyTorch
python setup.py clean
python setup.py build
python setup.py install
```

---

## Section 3: Compilation Process (~200 lines)

### setup.py Build Workflow

**High-Level Build Steps:**
```python
# From PyTorch setup.py
class build(setuptools.command.build.build):
    def run(self):
        # 1. Build backend dependencies
        self.run_command('build_deps')

        # 2. Generate cwrap bindings
        generate_nn_wrappers()

        # 3. Build Python extensions
        self.run_command('build_ext')

        # 4. Build pure Python modules
        self.run_command('build_py')
```

From [PyTorch Internals Part II](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/):
> "The first thing the install command does is run a command called build_deps – this invokes build_all.sh which runs CMake configure step on all backend libraries and then make install."

### Backend Library Compilation (build_all.sh)

**CMake Build Process for Each Library:**
```bash
# Example: Building TH library
mkdir -p build/TH
cd build/TH

cmake ../../TH \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../tmp_install \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++

ninja install
```

**Backend Build Order (Dependencies):**
```
1. TH (CPU tensors)
2. THNN (CPU neural nets) - depends on TH
3. THC (CUDA tensors) - depends on TH
4. THCUNN (CUDA neural nets) - depends on THC + THNN
5. libshm (shared memory)
6. NCCL (multi-GPU communication)
```

### CUDA Kernel Compilation

**CUDA Compilation Pipeline:**
```
.cu source file (C++ + CUDA)
    ↓ nvcc -c
.cubin object file (GPU binary)
    ↓ nvcc -fatbin (if multi-arch)
.fatbin (multi-architecture binary)
    ↓ link with host code
.so shared library (final output)
```

**Example CUDA Compilation Command:**
```bash
# Single architecture
nvcc -c kernel.cu \
    -arch=sm_80 \
    -O3 \
    -std=c++17 \
    -o kernel.o

# Multi-architecture fatbin
nvcc -c kernel.cu \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90 \
    -O3 -std=c++17 \
    -o kernel.o
```

**PTX Forward Compatibility:**
```bash
# Include PTX for future GPUs
nvcc -c kernel.cu \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90 \
    # ^^^^ PTX code for sm_90+ architectures
```

### Python Extension Compilation

**CUDAExtension Build Example:**
```python
# From setup.py
from torch.utils.cpp_extension import CUDAExtension

ext_modules = [
    CUDAExtension(
        name='torch._C',
        sources=[
            'torch/csrc/Module.cpp',
            'torch/csrc/Tensor.cpp',
            'torch/csrc/autograd/engine.cpp',
            # ... hundreds of source files
        ],
        include_dirs=[
            'torch/lib/tmp_install/include',
            'torch/lib/tmp_install/include/TH',
            'torch/lib/tmp_install/include/THC',
        ],
        library_dirs=[
            'torch/lib',
        ],
        libraries=[
            'shm',
            'cudart',
            'cudnn',
            'nccl',
        ],
        extra_link_args=[
            'torch/lib/libTH.so.1',      # Explicit TH path
            'torch/lib/libTHC.so.1',     # Explicit THC path
            'torch/lib/libTHNN.so.1',    # Explicit THNN path
            'torch/lib/libTHCUNN.so.1',  # Explicit THCUNN path
        ]
    )
]
```

**Why Explicit Library Paths?**

From [PyTorch Internals Part II](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/):
> "We manually specify the paths to the shared libraries we generated directly to the linker. The issue is that Lua Torch installs often set the LD_LIBRARY_PATH variable, and thus we could mistakenly link against a TH library built for Lua Torch instead of the library we have built locally."

### Verification After Build

**Check Compiled CUDA Architectures:**
```python
import torch

# Verify CUDA is available
assert torch.cuda.is_available()

# Check compiled architectures
print(torch.cuda.get_arch_list())
# Output: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'compute_90']

# Verify current GPU is supported
device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(device)
print(f"GPU Compute Capability: {capability[0]}.{capability[1]}")
```

**Verify Shared Library Dependencies:**
```bash
# Check torch._C extension dependencies
ldd /path/to/site-packages/torch/_C.cpython-39-x86_64-linux-gnu.so

# Should show:
# libTH.so.1 => /path/to/torch/lib/libTH.so.1
# libTHC.so.1 => /path/to/torch/lib/libTHC.so.1
# libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12
# libcudnn.so.8 => /usr/local/cudnn/lib64/libcudnn.so.8
```

---

## Section 4: Multi-Architecture Builds (~150 lines)

### Single-Arch vs Multi-Arch Trade-offs

**Single-Architecture Build (Recommended for Production):**
```bash
# A100-only build
export TORCH_CUDA_ARCH_LIST="8.0"
python setup.py install

# Advantages:
# - Fastest compile time (~1.5 hours)
# - Smallest binary size (~800MB)
# - Optimal runtime performance (no arch selection overhead)
# - Simpler deployment (known target GPU)

# Disadvantages:
# - Only runs on sm_80 GPUs (A100)
# - Runtime error on other GPUs
```

**Multi-Architecture Build (Recommended for Distribution):**
```bash
# T4 + A100 + A10 + H100 build
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
python setup.py install

# Advantages:
# - Runs on multiple GPU types
# - Good for heterogeneous clusters
# - PyPI-style distribution compatibility

# Disadvantages:
# - Slower compile time (~2.5 hours)
# - Larger binary size (~2.5GB)
# - Slight runtime overhead (arch selection)
```

### Fatbin Structure

**Understanding CUDA Fatbin:**
```
fatbin (fat binary)
├── sm_75 code (T4)
├── sm_80 code (A100)
├── sm_86 code (A10)
├── sm_90 code (H100)
└── compute_90 PTX (future GPUs)
```

**Runtime Architecture Selection:**
```python
# CUDA runtime automatically selects correct code
# based on GPU compute capability

# Example: Running on A100
device = torch.device('cuda:0')
x = torch.randn(1000, 1000, device=device)

# CUDA runtime:
# 1. Detects GPU is sm_80 (A100)
# 2. Extracts sm_80 code from fatbin
# 3. Loads and executes sm_80 kernels
# 4. Ignores other architectures in fatbin
```

### PTX Forward Compatibility

**Why Include PTX:**
```bash
# Build with PTX for sm_90
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"

# +PTX generates:
# - sm_80 binary (for A100)
# - sm_90 binary (for H100)
# - compute_90 PTX (for future GPUs >= sm_90)
```

**PTX Just-In-Time Compilation:**
```
New GPU (e.g., sm_100 in future)
    ↓
CUDA driver checks fatbin
    ↓
No sm_100 binary found
    ↓
Falls back to compute_90 PTX
    ↓
JIT compiles PTX → sm_100 binary
    ↓
Caches compiled binary
    ↓
Runs on new GPU
```

**Performance Trade-off:**
- **Binary code**: Fastest (pre-compiled)
- **PTX code**: Slower first run (JIT compile), then cached
- **No compatible code**: Runtime error

### Verification Commands

**Check Fatbin Contents:**
```bash
# Extract embedded fatbin from .so
cuobjdump --list-elf /path/to/torch/lib/libTHC.so.1

# Output:
# ELF file    1: sm_75
# ELF file    2: sm_80
# ELF file    3: sm_86
# ELF file    4: sm_90
# PTX file    1: compute_90
```

**Binary Size Analysis:**
```bash
# Compare binary sizes
du -h torch_sm80_only/_C.so        # ~200MB
du -h torch_multi_arch/_C.so       # ~600MB

# Size breakdown per architecture
cuobjdump --dump-elf-all torch_multi_arch/_C.so | grep "\.text"
```

---

## Section 5: Build Time Optimization (~200 lines)

### ccache: Compilation Caching

**What is ccache?**

From [ccache GitHub discussions](https://github.com/ccache/ccache/discussions/1420) (accessed 2025-02-03):
> "ccache is a compiler cache. It speeds up recompilation by caching previous compilations and detecting when the same compilation is being done again. When the hit rate is good, the speedup is excellent."

**Setup and Configuration:**
```bash
# Install ccache
sudo apt-get install ccache  # Ubuntu/Debian
brew install ccache          # macOS

# Configure PyTorch build to use ccache
export USE_CCACHE=1
export CCACHE_DIR=/var/cache/ccache

# Configure ccache limits
ccache --max-size=50G        # Allow 50GB cache
ccache --set-config=compression=true
ccache --set-config=compression_level=6

# Check ccache stats
ccache -s
# Statistics:
#   Cache hits: 12543
#   Cache misses: 1234
#   Hit rate: 91.0%
```

**ccache Performance Impact:**
```bash
# First build (cold cache)
time python setup.py build
# real    120m0.000s  (2 hours)
# ccache hit rate: 0%

# Second build (warm cache, no code changes)
time python setup.py build
# real    8m30.000s   (8.5 minutes)
# ccache hit rate: 95%

# Incremental build (changed 5 files)
time python setup.py build
# real    12m0.000s   (12 minutes)
# ccache hit rate: 92%
```

### Ninja vs Make

**Build System Comparison:**

| Feature | Make | Ninja | Speedup |
|---------|------|-------|---------|
| Parallel build scheduling | Basic | Optimized | 1.3-1.5× |
| Dependency tracking | File timestamps | Content hash | More accurate |
| Build time (32 cores) | ~140 minutes | ~95 minutes | 1.47× faster |
| Incremental rebuild | ~15 minutes | ~8 minutes | 1.88× faster |

**Using Ninja:**
```bash
# Install Ninja
sudo apt-get install ninja-build

# Configure PyTorch to use Ninja
export CMAKE_GENERATOR=Ninja

# Build with Ninja
python setup.py build
# CMake will use Ninja instead of Make
```

**From PyTorch Forums:**
> "Installing ninja and adding MAX_JOBS=4 along with invoking the CUDAExtension setup stuff with -j 4 significantly sped things up."

### Parallel Compilation

**MAX_JOBS Environment Variable:**
```bash
# Limit parallel jobs (avoid OOM on large builds)
export MAX_JOBS=16
python setup.py build

# Automatic detection (uses all CPU cores)
unset MAX_JOBS
python setup.py build
# Will use: os.cpu_count() = 64 cores

# Per-module job control
export MAX_JOBS=32        # C++ compilation
export TORCH_NVCC_FLAGS="-j 16"  # CUDA compilation
```

**Memory Considerations:**
```bash
# High parallelism can cause OOM
# Rule of thumb: 2-4GB RAM per parallel job

# 64GB RAM system:
export MAX_JOBS=16  # 64GB / 4GB = 16 jobs

# 128GB RAM system:
export MAX_JOBS=32  # 128GB / 4GB = 32 jobs

# Monitor during build:
watch -n 1 'free -h && ps aux | grep "c++" | wc -l'
```

### Incremental Builds (Development Mode)

**Setuptools develop Mode:**
```bash
# Install in development mode
python setup.py develop

# Advantages:
# - No copy to site-packages (uses symlinks)
# - Changes to Python code are immediate
# - C++ changes require rebuild (faster than full install)
```

**From [PyTorch Internals Part II](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/):**
> "develop allows us to essentially treat the PyTorch repo itself as if it were in site-packages. If we change a Python source file, the changes are automatically picked up. If we change a C++ Source File, we can re-run the develop command, and it will re-build just that extension."

**Incremental Rebuild Workflow:**
```bash
# Initial setup (full build)
python setup.py develop

# Change C++ file: torch/csrc/autograd/engine.cpp
vim torch/csrc/autograd/engine.cpp

# Rebuild only changed extension (~5-10 minutes)
python setup.py build_ext --inplace

# Test changes immediately
python -c "import torch; torch.autograd.backward(...)"
```

### Cloud Build Optimization (GCP Example)

**arr-coc-0-1 Cloud Build Configuration:**
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--build-arg=MAX_JOBS=32'
      - '--build-arg=TORCH_CUDA_ARCH_LIST=8.0'  # A100 only
      - '--build-arg=USE_CCACHE=1'
      - '--tag=gcr.io/$PROJECT_ID/pytorch-custom:latest'
      - '.'
    timeout: 7200s  # 2 hours

options:
  machineType: 'E2_HIGHCPU_32'  # 32 vCPUs
  diskSizeGb: 200
```

**Dockerfile Optimization:**
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install ccache
RUN apt-get update && apt-get install -y ccache ninja-build

# Configure ccache
ENV USE_CCACHE=1
ENV CCACHE_DIR=/workspace/.ccache
ENV CCACHE_MAXSIZE=20G

# Build PyTorch
ARG TORCH_CUDA_ARCH_LIST=8.0
ARG MAX_JOBS=32

RUN git clone --depth 1 --branch v2.1.0 https://github.com/pytorch/pytorch
WORKDIR /pytorch

# Single-arch build for A100
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_GENERATOR=Ninja

RUN python setup.py install

# Verify build
RUN python -c "import torch; assert torch.cuda.is_available()"
RUN python -c "import torch; print(torch.cuda.get_arch_list())"
```

**Build Time Comparison (GCP Cloud Build):**
```
Configuration 1: Multi-arch + Make + No cache
- Machine: n1-standard-16 (16 vCPUs)
- Time: 3 hours 45 minutes
- Cost: $2.87

Configuration 2: Single-arch + Ninja + ccache (cold)
- Machine: e2-highcpu-32 (32 vCPUs)
- Time: 1 hour 55 minutes
- Cost: $2.14

Configuration 3: Single-arch + Ninja + ccache (warm)
- Machine: e2-highcpu-32 (32 vCPUs)
- Time: 18 minutes
- Cost: $0.35

Savings: 92% time reduction, 88% cost reduction
```

---

## Section 6: Debugging Compilation Errors (~100 lines)

### Common CUDA Compilation Errors

**Error 1: CUDA Not Found**
```
CMake Error: Could not find CUDA
```

**Solution:**
```bash
# Check CUDA installation
which nvcc
# /usr/local/cuda-12.1/bin/nvcc

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

**Error 2: Unsupported CUDA Architecture**
```
nvcc fatal: Unsupported gpu architecture 'compute_90'
```

**Solution:**
```bash
# Your CUDA version doesn't support sm_90
# Check CUDA version and supported architectures
nvcc --help | grep -A 10 "gpu-architecture"

# CUDA 11.8: supports up to sm_90
# CUDA 12.1: supports up to sm_90a

# Downgrade arch list or upgrade CUDA
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  # Remove 9.0
```

**Error 3: ABI Compatibility Mismatch**
```
undefined symbol: _ZN2at6Tensor5fill_ERKc10at7Scalar
```

**Solution:**
```bash
# C++ ABI incompatibility between PyTorch and extensions
# Rebuild with matching ABI flag

export TORCH_CXX11_ABI=1  # New ABI (GCC 5.1+)
# or
export TORCH_CXX11_ABI=0  # Old ABI (compatibility mode)

# Check PyTorch ABI
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

**Error 4: Out of Memory During Compilation**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
```bash
# Reduce parallel jobs
export MAX_JOBS=4  # Instead of 32

# Or add swap space
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Error 5: cuDNN Version Mismatch**
```
error: #error cuDNN version mismatch
```

**Solution:**
```bash
# Check installed cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR

# PyTorch 2.1 requires cuDNN 8.x
# Install matching version:
# https://developer.nvidia.com/cudnn
```

### Verbose Build Debugging

**Enable Detailed Build Output:**
```bash
# Verbose CMake output
export CMAKE_VERBOSE_MAKEFILE=1

# Verbose Ninja output
export NINJA_STATUS="[%f/%t %es] "

# Verbose nvcc output
export CUDA_NVCC_FLAGS="-v -Xptxas=-v"

# Full build log
python setup.py build 2>&1 | tee build.log
```

---

## Section 7: ARR-COC-0-1 Project Context (~50 lines)

### Custom PyTorch Build for arr-coc-0-1

**Project Requirements:**
- Target: A100 GPUs (sm_80) on Vertex AI
- Features: FlashAttention-2, custom CUDA kernels
- Deployment: GCP Cloud Build (~2-4 hours compilation)

**Build Configuration:**
```bash
# arr-coc-0-1/.env.build
TORCH_CUDA_ARCH_LIST=8.0           # A100 only (minimize binary size)
CMAKE_BUILD_TYPE=Release           # Production optimizations
USE_CUDA=1                         # Enable CUDA
USE_CUDNN=1                        # Enable cuDNN
USE_NCCL=1                         # Multi-GPU support
MAX_JOBS=32                        # Parallel compilation
USE_CCACHE=1                       # Compilation caching
CMAKE_GENERATOR=Ninja              # Fast build system
```

**Why Custom Compilation for arr-coc-0-1:**

1. **Tensor Cores Optimization**: Compile with sm_80 for A100 Tensor Core features (TF32, BF16)
2. **FlashAttention-2 Integration**: Requires CUDA 11.8+ and sm_80+ for efficient attention
3. **Texture Processing CUDA Kernels**: Custom kernels for RGB→LAB conversion, Sobel filters
4. **Binary Size**: Single-arch build reduces Docker image size (800MB vs 2.5GB)
5. **Cloud Build Time**: Optimized build completes in ~2 hours vs 4+ hours for multi-arch

**Custom CUDA Extensions in arr-coc-0-1:**
```python
# arr-coc-0-1/setup.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension(
        name='arr_coc.texture_cuda',
        sources=[
            'arr_coc/csrc/texture_ops.cpp',
            'arr_coc/csrc/texture_kernels.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode', 'arch=compute_80,code=sm_80',  # A100 only
            ]
        }
    )
]
```

---

## Sources

**Source Documents:**
- [cuda/00-streams-concurrency-async.md](00-streams-concurrency-async.md) - CUDA API context
- [cuda/01-memory-management-unified.md](01-memory-management-unified.md) - Memory allocation
- [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) - GPU hardware specs

**Web Research:**
- [PyTorch Internals Part II – The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/) - PyTorch Blog (accessed 2025-02-03)
- [Specify Cuda architecture? · Issue #12119](https://github.com/pytorch/pytorch/issues/12119) - GitHub pytorch/pytorch (accessed 2025-02-03)
- [Best Practices to Edit and Compile Pytorch Source Code On Windows](https://github.com/pytorch/pytorch/wiki/Best-Practices-to-Edit-and-Compile-Pytorch-Source-Code-On-Windows) - GitHub Wiki (accessed 2025-02-03)
- [ccache massively speeds up build time · Issue #5018](https://github.com/ccache/ccache/discussions/1420) - GitHub ccache (accessed 2025-02-03)
- [Speeding up C++/CUDA extension build time](https://discuss.pytorch.org/t/speeding-up-c-cuda-extension-build-time/96151) - PyTorch Forums (accessed 2025-02-03)

**Additional References:**
- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/) - Official docs
- [CMake CUDA Architecture Documentation](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) - CMake reference
- [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) - nvcc reference
