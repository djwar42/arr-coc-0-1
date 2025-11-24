# CUDA Compilation Failures & Build System Troubleshooting (Expert-Level)

## Overview

Expert-level troubleshooting guide for CUDA compilation errors, PyTorch build failures, and environment configuration issues. This document covers real-world error messages, root cause analysis, and production-grade solutions from 2024-2025 troubleshooting experiences.

**Why This Matters:**
- PyTorch from-source builds fail with cryptic errors
- Architecture mismatches break deployment on new hardware
- Environment variables cause silent failures
- CMake configuration errors waste hours of build time

**Prerequisite Knowledge:**
- See [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) for basic build process
- See [cuda/03-compute-capabilities-gpu-architectures.md](03-compute-capabilities-gpu-architectures.md) for architecture details

---

## Section 1: nvcc Compilation Errors (~100 lines)

### Error 1.1: Architecture Mismatch (Compile on sm_75, Run on sm_80)

**Symptom:**
```
nvcc fatal: Unsupported gpu architecture 'compute_80'
```

**Root Cause from [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvcc-fatal-unsupported-gpu-architecture/196085) (accessed 2025-11-13):**
- Compiling on Tesla T4 (sm_75, CUDA 11.0) for execution on A100 (sm_80)
- nvcc version doesn't support target architecture
- CUDA toolkit version limits architecture support

**Architecture Support Matrix:**
```
CUDA 10.2: sm_35, sm_37, sm_50, sm_52, sm_60, sm_61, sm_70, sm_75
CUDA 11.0: sm_35 → sm_80 (adds Ampere A100)
CUDA 11.1: sm_35 → sm_86 (adds Ampere RTX 30 series)
CUDA 11.8: sm_35 → sm_90 (adds Hopper H100)
CUDA 12.0: sm_50 → sm_90 (drops Kepler sm_35/37)
CUDA 12.4: sm_50 → sm_100 (adds Blackwell B100)
```

**Solution 1: Use PTX Forward Compatibility**
```bash
# Add PTX intermediate representation for forward compatibility
nvcc -gencode arch=compute_75,code=compute_75 \
     -gencode arch=compute_80,code=sm_80 \
     -o app app.cu

# The compute_75 PTX will JIT compile on sm_80 at runtime
# Slower first run, then cached
```

**Solution 2: Upgrade CUDA Toolkit**
```bash
# Check current nvcc version
nvcc --version

# Upgrade to CUDA 11.8+ for sm_90 support
# Download from https://developer.nvidia.com/cuda-downloads
sudo sh cuda_11.8.0_linux.run

# Verify upgrade
nvcc --version
# Should show: Cuda compilation tools, release 11.8
```

**Solution 3: Multi-Architecture Build**
```bash
# Build for multiple architectures (fatbin)
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -o app app.cu

# Binary will run on sm_75, sm_80, sm_86, sm_90
# Larger binary size (multiple architectures embedded)
```

### Error 1.2: Missing CUDA Headers (cuda.h, cuda_runtime.h)

**Symptom from [GitHub PyTorch Issues](https://github.com/pytorch/pytorch/issues/90757) (accessed 2025-11-13):**
```
fatal error: cuda.h: No such file or directory
#include <cuda.h>
         ^~~~~~~~
compilation terminated.
```

**Root Cause:**
- CUDA_HOME not set or points to wrong location
- conda CUDA toolkit doesn't include nvcc/headers
- Multiple CUDA installations causing confusion

**Diagnostic Commands:**
```bash
# Check if CUDA_HOME is set
echo $CUDA_HOME
# Should output: /usr/local/cuda or /usr/local/cuda-11.8

# Verify cuda.h exists
ls $CUDA_HOME/include/cuda.h
# If fails: CUDA_HOME is wrong or CUDA not installed

# Check for nvcc
which nvcc
# Should be: /usr/local/cuda/bin/nvcc

# Verify CUDA version
nvcc --version
nvidia-smi  # Driver version (may be newer than toolkit)
```

**Solution 1: Set CUDA_HOME Explicitly**
```bash
# Find CUDA installation
sudo find / -name cuda.h 2>/dev/null
# Common locations:
# /usr/local/cuda-11.8/include/cuda.h
# /usr/local/cuda/include/cuda.h

# Set CUDA_HOME temporarily
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make permanent (add to ~/.bashrc)
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Solution 2: Fix Conda CUDA Installation**
```bash
# Conda cudatoolkit doesn't include nvcc by default
# Install full CUDA toolkit from conda-forge
conda install -c conda-forge cudatoolkit-dev

# Or set CUDA_HOME to conda environment
export CUDA_HOME=$CONDA_PREFIX

# Verify cuda.h exists
ls $CONDA_PREFIX/include/cuda.h
```

**Solution 3: Symlink CUDA for Consistency**
```bash
# Create /usr/local/cuda symlink pointing to specific version
sudo ln -sf /usr/local/cuda-11.8 /usr/local/cuda

# Now CUDA_HOME=/usr/local/cuda always works
export CUDA_HOME=/usr/local/cuda
```

### Error 1.3: Linker Errors (Undefined References)

**Symptom from [GitHub PyTorch Issues](https://github.com/pytorch/pytorch/issues/4913) (accessed 2025-11-13):**
```
[ 97%] Linking CXX executable broadcast_test
undefined reference to `std::runtime_error::runtime_error(char const*)'
undefined reference to `cudaGetDeviceCount'
collect2: error: ld returned 1 exit status
```

**Root Cause:**
- Missing -lcudart linker flag
- Wrong library search path
- ABI compatibility mismatch between libraries
- Missing CUDA libraries in LD_LIBRARY_PATH

**Solution 1: Fix Linker Flags**
```bash
# Add CUDA libraries explicitly
nvcc -o app app.cu -lcudart -lcublas -lcusparse

# Or with g++
g++ -o app app.o -L/usr/local/cuda/lib64 -lcudart -lcublas

# CMake: Set CUDA library path
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      ..
```

**Solution 2: Fix LD_LIBRARY_PATH**
```bash
# Add CUDA libraries to runtime path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify libraries are found
ldd app | grep cuda
# Should show: libcudart.so.11.0 => /usr/local/cuda/lib64/libcudart.so.11.0

# If not found, update cache
sudo ldconfig /usr/local/cuda/lib64
```

**Solution 3: ABI Compatibility Check**
```bash
# Check if C++ ABI mismatches exist
# PyTorch pre-1.10 used old ABI, >=1.10 uses new ABI

# Check PyTorch ABI
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# True = new ABI, False = old ABI

# Rebuild with matching ABI
export _GLIBCXX_USE_CXX11_ABI=1  # New ABI
# Or
export _GLIBCXX_USE_CXX11_ABI=0  # Old ABI

# CMake flag
cmake -D_GLIBCXX_USE_CXX11_ABI=1 ..
```

---

## Section 2: CMake & Build System Issues (~100 lines)

### Error 2.1: CMake Can't Find CUDA

**Symptom from [PyTorch Forums](https://discuss.pytorch.org/t/compile-pytorch-source-code-with-cuda-enabled-got-error/208049) (accessed 2025-11-13):**
```
CMake Error: Could not find CUDA
CMake Error at CMakeLists.txt:150 (find_package):
  Could not find a package configuration file provided by "CUDA"
```

**Root Cause:**
- CMAKE_PREFIX_PATH doesn't include CUDA
- FindCUDA.cmake can't locate toolkit
- CUDA_TOOLKIT_ROOT_DIR not set

**Solution 1: Set CMake CUDA Variables**
```bash
# Method 1: Environment variables
export CMAKE_PREFIX_PATH=/usr/local/cuda:$CMAKE_PREFIX_PATH
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Method 2: CMake command-line flags
cmake -DCMAKE_PREFIX_PATH=/usr/local/cuda \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      ..

# Verify CMake found CUDA
cmake .. 2>&1 | grep -i cuda
# Should show: -- Found CUDA: /usr/local/cuda (found version "11.8")
```

**Solution 2: Use CMake's find_package Debug Mode**
```bash
# Enable verbose FindCUDA output
cmake --debug-find -DCMAKE_FIND_DEBUG_MODE=ON ..

# Shows where CMake is looking for CUDA
# Example output:
#   CMake Debug Log at cmake/modules/FindCUDA.cmake:
#     find_path called with
#       HINTS: /usr/local/cuda/include
#       PATHS: /usr/include
```

**Solution 3: Manual FindCUDA.cmake Fix**
```cmake
# Create custom FindCUDA.cmake override
# File: cmake/Modules/FindCUDA.cmake

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
set(CUDA_CUBLAS_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so")

# Then run cmake
cmake -DCMAKE_MODULE_PATH=/path/to/cmake/Modules ..
```

### Error 2.2: TORCH_CUDA_ARCH_LIST Configuration Issues

**Symptom:**
```
CMake Warning:
  Manually-specified variables were not used by the project:
    TORCH_CUDA_ARCH_LIST
```

Or worse:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Root Cause:**
- Built for wrong architecture
- TORCH_CUDA_ARCH_LIST not passed correctly
- Default architecture doesn't match hardware

**Diagnostic:**
```bash
# Check what architectures were compiled
python -c "import torch; print(torch.cuda.get_arch_list())"
# Example output: ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Example output: 8.0  (RTX 3090 = sm_80)

# If GPU arch not in compiled list → rebuild needed
```

**Solution 1: Set TORCH_CUDA_ARCH_LIST Correctly**
```bash
# For PyTorch build, set environment variable
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

# Or CMake flag
cmake -DTORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0" ..

# Verify during build
python setup.py build 2>&1 | grep "arch="
# Should show: -gencode arch=compute_75,code=sm_75 ...
```

**Solution 2: Auto-Detect GPU Architecture**
```python
# File: detect_cuda_arch.py
import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    arch = f"{capability[0]}.{capability[1]}"
    print(f"Detected CUDA architecture: sm_{capability[0]}{capability[1]}")
    print(f"Set: export TORCH_CUDA_ARCH_LIST=\"{arch}\"")
else:
    print("CUDA not available")
```

**Solution 3: Build for All Common Architectures (Universal Binary)**
```bash
# Build time increases, binary size increases, but works everywhere
export TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Or skip old architectures for smaller binary
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"  # Turing, Ampere, Ada, Hopper
```

### Error 2.3: Out of Memory During Compilation

**Symptom:**
```
c++: fatal error: Killed signal terminated program cc1plus
FATAL ERROR: command terminated with exit code 1
```

**Root Cause:**
- Compiler processes killed by OOM killer
- Ninja parallel build uses too much RAM
- Large CUDA kernels require >16GB RAM

**Solution 1: Limit Parallel Jobs**
```bash
# Check available RAM
free -h
# If <32GB RAM, limit parallelism

# Ninja: Set MAX_JOBS
export MAX_JOBS=4  # Limit to 4 parallel jobs
python setup.py install

# Make: Use -j flag
make -j4

# CMake with Ninja
cmake -GNinja -DCMAKE_BUILD_PARALLEL_LEVEL=4 ..
ninja
```

**Solution 2: Use Swap Space**
```bash
# Create 16GB swap file (emergency measure)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
free -h
# Should show 16G swap

# Remove after build
sudo swapoff /swapfile
sudo rm /swapfile
```

**Solution 3: Selective Compilation**
```bash
# Disable optional features to reduce memory usage
export USE_DISTRIBUTED=0
export USE_MKLDNN=0
export USE_NNPACK=0
export BUILD_TEST=0

# Reduce CUDA architectures
export TORCH_CUDA_ARCH_LIST="8.0"  # Only build for your GPU

python setup.py install
```

---

## Section 3: Environment & Dependency Problems (~100 lines)

### Error 3.1: PATH and LD_LIBRARY_PATH Chaos

**Symptom from [PyTorch Forums](https://discuss.pytorch.org/t/pytorch-2-0-1-not-recognizing-cuda-11-4-despite-correct-environment-setup/196939) (accessed 2025-11-13):**
```python
import torch
torch.cuda.is_available()
# Returns: False

torch.version.cuda
# Returns: None or '11.8' but still False
```

**Root Cause:**
- Multiple CUDA installations conflict
- Conda CUDA overrides system CUDA
- LD_LIBRARY_PATH points to wrong libcudart.so
- PyTorch built with different CUDA than runtime

**Diagnostic Workflow:**
```bash
# 1. Check which CUDA PyTorch sees
python -c "import torch; print(torch.version.cuda)"
# Example: 11.8

# 2. Check which CUDA is in PATH
which nvcc
# /usr/local/cuda-11.8/bin/nvcc (good)
# /home/user/anaconda3/bin/nvcc (conda, might conflict)

# 3. Check runtime libraries
python -c "import torch; print(torch._C._cuda_getCompiledVersion())"
# 11080 = CUDA 11.8

# 4. Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | tr ':' '\n'
# Should show /usr/local/cuda-11.8/lib64 FIRST

# 5. Verify library loading
ldd $(python -c "import torch; print(torch._C.__file__)") | grep cuda
# Should show /usr/local/cuda-11.8/lib64/libcudart.so.11.0
```

**Solution 1: Fix Environment Variable Order**
```bash
# CRITICAL: Order matters! System CUDA must come BEFORE conda
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH  # Prepend, not append
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  # Prepend

# Remove conda CUDA from path (if conflicts)
export PATH=$(echo $PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':')

# Verify nvcc is system version
which nvcc
# Should be /usr/local/cuda-11.8/bin/nvcc, NOT conda
```

**Solution 2: Create Clean Environment Script**
```bash
# File: setup_cuda.sh
#!/bin/bash

# Clear existing CUDA paths
export PATH=$(echo $PATH | sed 's|/usr/local/cuda[^:]*:||g')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/usr/local/cuda[^:]*:||g')

# Set clean CUDA environment
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

# Verify
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(which nvcc)"
echo "nvcc version: $(nvcc --version | grep release)"

# Use:
source setup_cuda.sh
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution 3: Conda-Specific Fix**
```bash
# Option A: Install PyTorch with exact CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Option B: Set CUDA_HOME to conda environment
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Option C: Remove conda CUDA, use system CUDA
conda remove cudatoolkit cudnn
export CUDA_HOME=/usr/local/cuda-11.8
```

### Error 3.2: Driver vs Runtime Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: invalid device function
CUDA driver version is insufficient for CUDA runtime version
```

**Diagnostic:**
```bash
# Check driver version (what nvidia-smi reports)
nvidia-smi | grep "Driver Version"
# Example: Driver Version: 450.102.04  CUDA Version: 11.0

# Check runtime version (what nvcc uses)
nvcc --version
# Example: Cuda compilation tools, release 11.8

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
# Example: 11.8

# Problem: Driver 11.0, Runtime 11.8 → Driver too old!
```

**Version Compatibility Matrix:**
```
Driver Version → Maximum CUDA Runtime Supported
450.xx → CUDA 11.0
460.xx → CUDA 11.2
470.xx → CUDA 11.4
510.xx → CUDA 11.6
515.xx → CUDA 11.7
520.xx → CUDA 11.8
525.xx → CUDA 12.0
530.xx → CUDA 12.1
535.xx → CUDA 12.2
545.xx → CUDA 12.3
550.xx → CUDA 12.4
```

**Solution 1: Upgrade NVIDIA Driver**
```bash
# Ubuntu/Debian
sudo ubuntu-drivers autoinstall
# Or specific version
sudo apt install nvidia-driver-535

# CentOS/RHEL
sudo dnf install nvidia-driver

# Verify
nvidia-smi
# Should show newer driver version

# Reboot required
sudo reboot
```

**Solution 2: Downgrade CUDA Runtime**
```bash
# If driver can't be upgraded (e.g., cluster environment)
# Downgrade CUDA toolkit to match driver

# Remove existing CUDA
sudo apt remove --purge cuda-*

# Install older CUDA matching driver
# Driver 450.xx supports CUDA 11.0
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
sudo sh cuda_11.0.3_450.51.06_linux.run
```

**Solution 3: Forward Compatibility (450.80.02+ drivers)**
```bash
# CUDA 11.x introduced forward compatibility
# Install CUDA forward compatibility package

# For driver 450.80.02 to use CUDA 11.8
sudo apt install cuda-compat-11-8

# This allows newer CUDA runtime on older driver
# Verify
nvidia-smi
# Should show "CUDA Version: 11.8" even with driver 450.xx
```

### Error 3.3: cuDNN Version Mismatch

**Symptom:**
```
RuntimeError: cuDNN version mismatch: PyTorch was compiled against 8.9.0 but linked against 8.5.0
```

**Solution 1: Match cuDNN Version**
```bash
# Check cuDNN version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
# Or
python -c "import torch; print(torch.backends.cudnn.version())"

# Download correct cuDNN from NVIDIA
# https://developer.nvidia.com/cudnn

# Install cuDNN 8.9.0 for CUDA 11.8
tar -xzvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

**Solution 2: Rebuild PyTorch with Correct cuDNN**
```bash
# Set cuDNN path before building
export CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8
export CUDNN_INCLUDE_DIR=/usr/local/cuda/include

python setup.py clean
python setup.py install
```

---

## Section 4: Expert-Level Fixes & Advanced Debugging (~100 lines)

### Fix 4.1: Patching CMakeLists.txt for Custom CUDA Paths

**Scenario:** Enterprise cluster with non-standard CUDA installation paths

**Custom CMakeLists.txt Patch:**
```cmake
# File: custom_cuda_paths.cmake
# Include before main CMakeLists.txt

# Override CUDA search paths
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/cuda/11.8" CACHE PATH "CUDA root")
set(CUDA_SDK_ROOT_DIR "/opt/nvidia/hpc_sdk/cuda/11.8" CACHE PATH "CUDA SDK root")

# Force specific CUDA libraries
set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so" CACHE FILEPATH "CUDART library")
set(CUDA_CUBLAS_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so" CACHE FILEPATH "CUBLAS library")
set(CUDA_CUFFT_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so" CACHE FILEPATH "CUFFT library")

# Force nvcc path
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE FILEPATH "NVCC compiler")

# Include directories
include_directories(SYSTEM ${CUDA_TOOLKIT_ROOT_DIR}/include)

# Usage:
# cmake -C custom_cuda_paths.cmake ..
```

### Fix 4.2: Force Architecture Compilation (Override Detection)

**Problem:** CMake auto-detection picks wrong architecture

**Force Specific Architecture:**
```cmake
# Edit torch/CMakeLists.txt or pass as cmake flag

# Method 1: Hardcode in CMakeLists.txt
set(TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6" CACHE STRING "CUDA architectures" FORCE)

# Method 2: Command-line override
cmake -DTORCH_CUDA_ARCH_LIST="8.0" -DCMAKE_CUDA_ARCHITECTURES="80" ..

# Method 3: Environment variable (PyTorch setup.py)
export TORCH_CUDA_ARCH_LIST="8.0"
export FORCE_CUDA=1
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

python setup.py install --force-cuda
```

### Fix 4.3: Verbose Build Debugging

**Enable Maximum Build Verbosity:**
```bash
# PyTorch build with full output
export DEBUG=1
export VERBOSE=1
export MAX_JOBS=1  # Serial build for readable output

python setup.py install 2>&1 | tee build.log

# Search log for specific errors
grep -i "error" build.log
grep -i "cuda" build.log | grep -i "not found"

# CMake verbose
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1

# Ninja verbose
ninja -v
```

### Fix 4.4: Diagnostic Checklist (Production Deployment)

**Pre-Deployment CUDA Environment Verification:**
```bash
#!/bin/bash
# File: verify_cuda_env.sh

echo "=== CUDA Environment Diagnostic ==="

# 1. Driver check
echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed (driver issue)"
    exit 1
fi

# 2. CUDA toolkit check
echo "2. CUDA Toolkit:"
nvcc --version | grep "release"
if [ $? -ne 0 ]; then
    echo "ERROR: nvcc not found (CUDA not in PATH)"
    exit 1
fi

# 3. CUDA_HOME check
echo "3. CUDA_HOME:"
echo $CUDA_HOME
if [ ! -d "$CUDA_HOME/include" ]; then
    echo "ERROR: CUDA_HOME invalid or not set"
    exit 1
fi

# 4. Library path check
echo "4. CUDA Libraries:"
ldconfig -p | grep cudart
if [ $? -ne 0 ]; then
    echo "ERROR: libcudart not in library path"
    exit 1
fi

# 5. PyTorch CUDA check
echo "5. PyTorch CUDA:"
python -c "import torch; print(f'Available: {torch.cuda.is_available()}'); print(f'Version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch CUDA not working"
    exit 1
fi

# 6. cuDNN check
echo "6. cuDNN:"
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"

# 7. Architecture compatibility
echo "7. Architecture Check:"
python -c "import torch; print(f'Compiled for: {torch.cuda.get_arch_list()}'); cap = torch.cuda.get_device_capability(); print(f'GPU requires: sm_{cap[0]}{cap[1]}')"

echo "=== All checks passed! ==="
```

### Fix 4.5: Common Production Errors & Quick Fixes

**Error: "CUDA out of memory" during compilation**
```bash
# Not OOM at runtime, but during build
export MAX_JOBS=1
export PYTORCH_BUILD_VERSION=0  # Disable versioning overhead
export PYTORCH_BUILD_NUMBER=0
python setup.py install
```

**Error: "cc1plus: all warnings being treated as errors"**
```bash
# Disable warnings-as-errors
export CXXFLAGS="-Wno-error"
export CFLAGS="-Wno-error"
cmake -DCMAKE_CXX_FLAGS="-Wno-error" ..
```

**Error: "nvcc: command not found" during PyTorch build**
```bash
# Even with CUDA_HOME set
# Create nvcc wrapper in PATH
sudo ln -s /usr/local/cuda-11.8/bin/nvcc /usr/local/bin/nvcc
```

**Error: "ImportError: libcudart.so.11.0: cannot open shared object file"**
```bash
# Fix library cache
sudo ldconfig /usr/local/cuda/lib64

# Or add to ld.so.conf
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig
```

---

## Workflow: Expert Compilation Troubleshooting Process

**Step 1: Identify Error Category**
```
- nvcc error? → Section 1
- CMake error? → Section 2
- Runtime/import error? → Section 3
- Build system error? → Section 4
```

**Step 2: Run Diagnostic Checklist**
```bash
# Use verify_cuda_env.sh (from Section 4.4)
./verify_cuda_env.sh
```

**Step 3: Check Version Compatibility**
```bash
# Driver → CUDA Runtime → PyTorch compatibility
nvidia-smi  # Driver version
nvcc --version  # CUDA runtime
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

**Step 4: Enable Verbose Logging**
```bash
# Maximum verbosity for error diagnosis
export DEBUG=1
export VERBOSE=1
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1 2>&1 | tee build.log
```

**Step 5: Search Build Log**
```bash
# Find first error (often root cause)
grep -n "error:" build.log | head -1

# Common error patterns
grep -i "unsupported gpu architecture" build.log
grep -i "could not find cuda" build.log
grep -i "undefined reference" build.log
```

---

## Sources

**Web Research (Accessed 2025-11-13):**
- [NVIDIA Developer Forums - nvcc fatal unsupported gpu architecture](https://forums.developer.nvidia.com/t/nvcc-fatal-unsupported-gpu-architecture/196085) - sm_75 to sm_80 compilation issue
- [GitHub PyTorch Issues #90757 - cuda.h missing during torch.compile](https://github.com/pytorch/pytorch/issues/90757) - CUDA_HOME environment variable fix
- [GitHub PyTorch Issues #4913 - CMake error when building from source](https://github.com/pytorch/pytorch/issues/4913) - Undefined reference linker errors
- [PyTorch Forums - Compile PyTorch source code with cuda enabled got error](https://discuss.pytorch.org/t/compile-pytorch-source-code-with-cuda-enabled-got-error/208049) - CMake CUDA detection issues
- [PyTorch Forums - PyTorch 2.0.1 Not Recognizing CUDA 11.4](https://discuss.pytorch.org/t/pytorch-2-0-1-not-recognizing-cuda-11-4-despite-correct-environment-setup/196939) - PATH and LD_LIBRARY_PATH troubleshooting

**Source Documents:**
- [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) - Basic compilation process and dependencies
- [cuda/03-compute-capabilities-gpu-architectures.md](03-compute-capabilities-gpu-architectures.md) - Architecture version compatibility

**Additional References:**
- [NVIDIA CUDA Documentation - Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) - nvcc flags and architecture targeting
- [PyTorch Documentation - Building from Source](https://github.com/pytorch/pytorch#from-source) - Official build instructions
- [Stack Overflow - nvcc missing when installing cudatoolkit](https://stackoverflow.com/questions/56470424/nvcc-missing-when-installing-cudatoolkit) - conda vs system CUDA conflicts

---

**Related Knowledge:**
- [cuda/00-streams-concurrency-async.md](00-streams-concurrency-async.md) - CUDA runtime configuration
- [cuda/01-memory-management-unified.md](01-memory-management-unified.md) - Memory allocation debugging
- [cuda/04-pytorch-custom-cuda-extensions.md](04-pytorch-custom-cuda-extensions.md) - Custom extension compilation
- [cuda/06-pytorch-jit-torch-compile.md](06-pytorch-jit-torch-compile.md) - JIT compilation errors
- [cuda/07-mixed-precision-training-internals.md](07-mixed-precision-training-internals.md) - Precision-related compilation issues
