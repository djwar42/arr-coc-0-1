# PyTorch CMake Build System Internals: Detection Logic & Optional Dependencies

## Overview

PyTorch's CMake build system contains complex detection logic for optional dependencies like CUPTI, Kineto, and other CUDA libraries. Understanding how CMake **detects** vs **validates** libraries is critical for debugging build failures, especially the infamous "CMake found it but library doesn't exist" scenario.

**Key Insight from Production Debugging:**
> CMake's `find_library()` can detect header files and **claim success** even when shared libraries (.so files) are missing. This causes runtime failures when PyTorch attempts to load profiling features that depend on CUPTI.

**Why This Matters:**
- **Docker multi-stage builds**: Copying wrong libraries between stages breaks profiling
- **Runtime vs build-time dependencies**: CMake validates at build time, failures happen at runtime
- **Optional feature detection**: CUPTI/Kineto are "found" based on headers alone
- **False positives**: CMake reports success, but `dlopen()` fails at runtime

**Related Knowledge:**
- See [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) for build process overview
- See [cuda/11-pytorch-profiling-ecosystem.md](11-pytorch-profiling-ecosystem.md) for CUPTI/Kineto profiling context
- See [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md) for CUDA package structure

---

## Section 1: CMake Detection Logic (~120 lines)

### How FindCUDA.cmake Works

**PyTorch CMake Detection Flow:**
```cmake
# From cmake/Dependencies.cmake
find_package(CUDA REQUIRED)

# FindCUDA.cmake searches in order:
1. ${CUDA_TOOLKIT_ROOT_DIR}
2. $ENV{CUDA_HOME}
3. $ENV{CUDA_PATH}
4. /usr/local/cuda
5. /usr/local/cuda-${VERSION}
```

From [PyTorch GitHub Issue #14333](https://github.com/pytorch/pytorch/issues/14333) (accessed 2025-11-13):
> "FindCUDA says CUDA version is (usually determined by nvcc), but the CUDA headers say the version is 9.0. This often occurs when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to non-standard locations, without also setting PATH to point to the correct nvcc."

**Header-Only Detection Problem:**

PyTorch's CMake searches for CUPTI using `find_library()`:
```cmake
# Simplified from cmake/Dependencies.cmake
find_library(CUPTI_LIBRARY
  NAMES cupti
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  NO_DEFAULT_PATH
)

if(CUPTI_LIBRARY)
  set(LIBKINETO_NOCUPTI OFF CACHE STRING "" FORCE)
  message(STATUS "Found CUPTI")
else()
  message(STATUS "Could not find CUPTI library, using CPU-only Kineto")
  set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
endif()
```

From [GitLab pytorch/cmake/Dependencies.cmake](https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/update-test-libtorch-path/cmake/Dependencies.cmake) (accessed 2025-11-13):
> "message(STATUS \"Found CUPTI\"). set(LIBKINETO_NOCUPTI OFF CACHE STRING \"\" FORCE). else(). message(STATUS \"Could not find CUPTI library, using CPU-only Kineto\")"

### Why CMake "Finds" Missing Libraries

**The False Positive Pattern:**

1. **CMake searches for headers** (`cupti.h` in `/usr/local/cuda/extras/CUPTI/include/`)
2. **Headers exist** (from cuda-libraries-dev package)
3. **CMake reports success** ("Found CUPTI")
4. **Shared library missing** (`libcupti.so.12` not in runtime image)
5. **Runtime failure** when PyTorch tries `dlopen("libcupti.so")`

**Example from CUPTI Investigation:**
```bash
# CMake finds headers
$ ls /usr/local/cuda/extras/CUPTI/include/
cupti.h  cupti_activity.h  cupti_callbacks.h  cupti_events.h

# CMake claims success
-- Found CUPTI

# But library is missing
$ ls /usr/local/cuda/extras/CUPTI/lib64/
ls: cannot access '/usr/local/cuda/extras/CUPTI/lib64/': No such file or directory

# Runtime error when profiling
RuntimeError: CUPTI initialization failed
```

### CMake Detection vs Validation

**Detection (Build Time):**
```cmake
find_library(CUPTI_LIBRARY NAMES cupti)
# Returns: CUPTI_LIBRARY-NOTFOUND or /path/to/libcupti.so

if(CUPTI_LIBRARY)
  # CMake thinks CUPTI is available
  set(USE_CUPTI ON)
endif()
```

**Validation (Runtime):**
```python
# PyTorch runtime (Python)
import torch.profiler

with torch.profiler.profile():
    # Attempts to load libcupti.so via dlopen()
    # FAILS if library not in LD_LIBRARY_PATH
    model(input)
```

**The Gap:**
CMake validates at **configure time** (when headers exist), but runtime needs **shared libraries** (`.so` files). In Docker multi-stage builds, headers stay in `devel` stage, libraries may not copy to `runtime` stage.

### Version Mismatch Detection

From [PyTorch Issue #14333](https://github.com/pytorch/pytorch/issues/14333):

**CMake Version Check Logic:**
```cmake
# cmake/public/cuda.cmake
execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} --version
                OUTPUT_VARIABLE NVCC_VERSION_OUTPUT)

# Extract version from nvcc output
string(REGEX MATCH "release ([0-9]+\\.[0-9]+)"
       NVCC_VERSION "${NVCC_VERSION_OUTPUT}")

# Compare with header version
file(READ "${CUDA_INCLUDE_DIRS}/cuda.h" CUDA_H_CONTENTS)
string(REGEX MATCH "define CUDA_VERSION ([0-9]+)"
       CUDA_HEADER_VERSION "${CUDA_H_CONTENTS}")

if(NOT NVCC_VERSION STREQUAL CUDA_HEADER_VERSION)
  message(FATAL_ERROR
    "FindCUDA says CUDA version is ${NVCC_VERSION}, "
    "but headers say ${CUDA_HEADER_VERSION}. "
    "Try: PATH=/usr/local/cuda-${CUDA_HEADER_VERSION}/bin:$PATH")
endif()
```

**Common Mismatch Scenarios:**
```bash
# Scenario 1: Multiple CUDA installations
/usr/local/cuda -> /usr/local/cuda-11.8  # Symlink
/usr/local/cuda-12.1                      # Also installed
nvcc --version  # Reports 11.8
cuda.h          # From 12.1 (wrong path)

# Scenario 2: Environment variable conflict
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-11.8/bin:$PATH
# nvcc from 11.8, headers from 12.1 → mismatch error
```

---

## Section 2: Optional Dependencies (~120 lines)

### Optional Dependency Detection in PyTorch

**PyTorch Optional Features:**
```cmake
# cmake/Dependencies.cmake
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use cuDNN" ON)
option(USE_NCCL "Use NCCL" ON)
option(USE_MKLDNN "Use MKLDNN" ON)
option(USE_KINETO "Use Kineto profiling library" ON)
option(USE_CUPTI_SO "Use CUPTI as shared library" ON)
```

From [PyTorch Forums Discussion](https://discuss.pytorch.org/t/pytorch-profiler-cupti-warning/131011) (accessed 2025-11-13):
> "The default build option is setup with CUPTI off. option(USE_KINETO \"Use Kineto profiling library\" ON) option(USE_CUPTI_SO \"Use CUPTI as a shared library\" ON)"

**Detection Priority Order:**

1. **Explicit CMake flags** (`-DUSE_CUPTI=OFF`)
2. **Environment variables** (`USE_CUPTI=0`)
3. **Automatic detection** (find_library success/failure)
4. **Default values** (from option() directive)

### Force Disabling Optional Dependencies

**Method 1: CMake Command Line**
```bash
# Disable CUPTI explicitly
cmake -DUSE_CUPTI=OFF \
      -DUSE_KINETO=ON \
      -DLIBKINETO_NOCUPTI=ON \
      ..

# Build PyTorch without profiling
python setup.py build
```

**Method 2: Environment Variables**
```bash
# Set before running setup.py
export USE_CUPTI=0
export USE_KINETO=1
export LIBKINETO_NOCUPTI=1

python setup.py install
```

**Method 3: Edit CMakeCache.txt**
```cmake
# After first cmake run, edit build/CMakeCache.txt
USE_CUPTI:BOOL=OFF
LIBKINETO_NOCUPTI:BOOL=ON

# Re-run cmake
cmake --build . --target install
```

### CMakeCache.txt Interpretation

**Reading CMakeCache.txt for Debugging:**
```bash
# Find CUPTI-related settings
grep -i cupti build/CMakeCache.txt

# Output:
# CUPTI_INCLUDE_DIR:PATH=/usr/local/cuda/extras/CUPTI/include
# CUPTI_LIBRARY:FILEPATH=/usr/local/cuda/extras/CUPTI/lib64/libcupti.so
# USE_CUPTI:BOOL=ON
# LIBKINETO_NOCUPTI:BOOL=OFF
```

**Key Variables to Check:**
```cmake
# CUDA detection
CUDA_TOOLKIT_ROOT_DIR:PATH=/usr/local/cuda-12.1
CUDA_NVCC_EXECUTABLE:FILEPATH=/usr/local/cuda/bin/nvcc
CUDA_VERSION:STRING=12.1

# CUPTI detection
CUPTI_LIBRARY:FILEPATH=CUPTI_LIBRARY-NOTFOUND  # ← Missing!
USE_CUPTI:BOOL=ON  # ← But still enabled!
LIBKINETO_NOCUPTI:BOOL=OFF  # ← Kineto expects CUPTI

# This mismatch causes runtime errors
```

**Debugging Pattern:**
```bash
# 1. Check what CMake found
cat build/CMakeCache.txt | grep -E "(CUPTI|KINETO|CUDA_VERSION)"

# 2. Verify library exists
ls -la $(grep CUPTI_LIBRARY build/CMakeCache.txt | cut -d= -f2)

# 3. If library missing, force disable
cmake -DUSE_CUPTI=OFF -DLIBKINETO_NOCUPTI=ON ..
```

### Kineto + CUPTI Dependency Chain

**Kineto's CUPTI Dependency:**
```cmake
# third_party/kineto/libkineto/CMakeLists.txt
if(LIBKINETO_NOCUPTI)
  # CPU-only profiling
  set(KINETO_LIBRARY_TYPE "nocupti")
  message(STATUS "Building Kineto without CUPTI")
else()
  # GPU profiling with CUPTI
  find_library(CUPTI_LIBRARY cupti)
  if(NOT CUPTI_LIBRARY)
    message(FATAL_ERROR "CUPTI required but not found")
  endif()
  set(KINETO_LIBRARY_TYPE "cupti")
endif()
```

**Three Kineto Build Modes:**

| Mode | CMake Flag | CUPTI Required | GPU Profiling |
|------|-----------|----------------|---------------|
| **Full GPU** | `LIBKINETO_NOCUPTI=OFF` | Yes | ✅ Full CUDA profiling |
| **CPU-only** | `LIBKINETO_NOCUPTI=ON` | No | ❌ CPU profiling only |
| **Disabled** | `USE_KINETO=OFF` | No | ❌ No profiling |

**Example: CUPTI Missing Scenario**
```bash
# Build with CUPTI expected
cmake -DUSE_KINETO=ON -DLIBKINETO_NOCUPTI=OFF ..
make -j32

# Runtime error when profiling
>>> import torch.profiler
>>> with torch.profiler.profile():
...     model(x)
RuntimeError: CUPTI initialization failed (libcupti.so.12 not found)

# Solution: Rebuild without CUPTI
cmake -DLIBKINETO_NOCUPTI=ON ..
make -j32
```

---

## Section 3: Build vs Runtime Dependencies (~80 lines)

### What's Needed at Build Time

**PyTorch Build Dependencies:**
```bash
# CUDA Toolkit (headers + libraries)
/usr/local/cuda/include/
  cuda.h
  cuda_runtime.h
  device_launch_parameters.h

/usr/local/cuda/lib64/
  libcudart.so      # CUDA runtime
  libcublas.so      # Linear algebra
  libcufft.so       # FFT operations

# CUPTI (optional, for profiling)
/usr/local/cuda/extras/CUPTI/include/
  cupti.h
  cupti_activity.h

/usr/local/cuda/extras/CUPTI/lib64/
  libcupti.so.12    # Profiling library
```

**CMake Validation at Build Time:**
```cmake
# CMake checks for:
find_library(CUDA_CUDART_LIBRARY cudart)     # Required
find_library(CUDA_CUBLAS_LIBRARY cublas)     # Required
find_library(CUDNN_LIBRARY cudnn)            # Required
find_library(CUPTI_LIBRARY cupti)            # Optional
find_library(NCCL_LIBRARY nccl)              # Optional

# If required library missing → build fails
# If optional library missing → feature disabled (but may not be!)
```

### What's Needed at Runtime

**PyTorch Runtime Dependencies:**
```bash
# Minimal runtime (no profiling)
libcudart.so.12       # CUDA runtime
libcublas.so.12       # Linear algebra
libcublasLt.so.12     # Lightweight BLAS
libcudnn.so.8         # cuDNN

# Optional runtime (for profiling)
libcupti.so.12        # CUPTI profiling
```

**Runtime Library Loading:**
```python
# PyTorch loads libraries via dlopen() at runtime
import torch
torch.cuda.init()  # Loads libcudart, libcublas, etc.

# Profiling triggers CUPTI load
import torch.profiler
with torch.profiler.profile():
    # Attempts: dlopen("libcupti.so.12")
    # If missing → RuntimeError
    pass
```

**LD_LIBRARY_PATH Resolution:**
```bash
# PyTorch searches for libraries in:
1. LD_LIBRARY_PATH (runtime environment variable)
2. RPATH (embedded in PyTorch .so files)
3. /usr/local/cuda/lib64 (default CUDA location)
4. /usr/lib/x86_64-linux-gnu (system libraries)

# Check RPATH in PyTorch
patchelf --print-rpath /path/to/torch/lib/libtorch_cuda.so
# Output: /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# If CUPTI not in RPATH or LD_LIBRARY_PATH → runtime error
```

### Docker Stage Implications

**Multi-Stage Build Example:**

**Stage 1: Builder (devel image)**
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Has headers + libraries
RUN apt-get update && apt-get install -y \
    cuda-libraries-dev-12-1  # Includes CUPTI headers + .so

# Build PyTorch
RUN git clone https://github.com/pytorch/pytorch
WORKDIR pytorch
RUN python setup.py install
# CMake finds CUPTI → sets USE_CUPTI=ON
```

**Stage 2: Runtime (runtime image)**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Only has runtime libraries (NO CUPTI!)
# nvidia/cuda:runtime does NOT include cuda-libraries-dev

# Copy PyTorch from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /usr/local/lib/python3.10/site-packages/torch

# Problem: PyTorch expects CUPTI (from build stage)
# But CUPTI library not copied to runtime stage
# Result: Profiling fails at runtime
```

**The Fix: Copy CUPTI or Disable at Build:**

**Option A: Copy CUPTI to runtime**
```dockerfile
COPY --from=builder /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12 \
                    /usr/local/cuda/extras/CUPTI/lib64/
```

**Option B: Build without CUPTI**
```dockerfile
# In builder stage
ENV LIBKINETO_NOCUPTI=1
RUN python setup.py install
# CMake sets USE_CUPTI=OFF → no CUPTI dependency
```

---

## Section 4: Debugging CMake Issues (~80 lines)

### CMakeCache.txt Interpretation

**Essential CMake Debugging Commands:**
```bash
# View all CUDA-related settings
grep -E "^CUDA|^CUPTI|^USE_" build/CMakeCache.txt

# Check detection results
grep "FOUND\|NOTFOUND" build/CMakeCache.txt

# View library paths
grep "LIBRARY:FILEPATH" build/CMakeCache.txt
```

**Example Output Analysis:**
```cmake
# Good: Library found
CUDA_CUDART_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart.so

# Bad: Library not found (but CMake continues)
CUPTI_LIBRARY:FILEPATH=CUPTI_LIBRARY-NOTFOUND

# Dangerous: Inconsistent state
USE_CUPTI:BOOL=ON                    # ← Feature enabled
CUPTI_LIBRARY:FILEPATH=NOTFOUND      # ← But library missing!
LIBKINETO_NOCUPTI:BOOL=OFF           # ← Kineto expects CUPTI
```

### CMAKE_FIND_DEBUG_MODE

**Enable Detailed Find Logging:**
```bash
# Method 1: Command line
cmake -DCMAKE_FIND_DEBUG_MODE=ON ..

# Method 2: In CMakeLists.txt
set(CMAKE_FIND_DEBUG_MODE ON)
find_library(CUPTI_LIBRARY cupti)
set(CMAKE_FIND_DEBUG_MODE OFF)
```

**Debug Output Example:**
```
-- Checking for module 'cupti'
--   Checking path: /usr/local/cuda/extras/CUPTI/lib64
--   Checking path: /usr/local/cuda/lib64
--   Checking path: /usr/lib/x86_64-linux-gnu
-- Found cupti header: /usr/local/cuda/extras/CUPTI/include/cupti.h
-- Could not find cupti library in any path
-- Result: CUPTI_LIBRARY-NOTFOUND
```

**Interpreting Debug Output:**
- **Found header, missing library**: Header-only detection false positive
- **Wrong path searched**: CUDA_TOOLKIT_ROOT_DIR misconfigured
- **Library exists but not found**: Permissions or naming mismatch

### Common False Positive Patterns

**Pattern 1: Header-Only Detection**
```cmake
# CMake search
find_path(CUPTI_INCLUDE_DIR cupti.h
          PATHS /usr/local/cuda/extras/CUPTI/include)
# → FOUND

find_library(CUPTI_LIBRARY cupti
             PATHS /usr/local/cuda/extras/CUPTI/lib64)
# → NOTFOUND

# But CMake reports success based on headers alone!
if(CUPTI_INCLUDE_DIR)
  message(STATUS "Found CUPTI")  # ← Misleading!
endif()
```

**Pattern 2: Cached False Positives**
```bash
# First build: CUPTI installed
cmake ..  # Sets CUPTI_LIBRARY=/path/to/libcupti.so

# Later: CUPTI uninstalled
rm /usr/local/cuda/extras/CUPTI/lib64/libcupti.so

# Second build: Cache still has old value!
cmake ..  # Uses cached CUPTI_LIBRARY (now invalid)

# Solution: Clear cache
rm -rf build/CMakeCache.txt build/CMakeFiles
cmake ..
```

**Pattern 3: Symlink Confusion**
```bash
# Multiple CUDA versions with symlinks
/usr/local/cuda -> cuda-12.1
/usr/local/cuda-11.8 (also installed)

# CMake finds:
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda  # → 12.1
CUDA_NVCC_EXECUTABLE=/usr/local/cuda-11.8/bin/nvcc

# Result: Version mismatch error
```

### Debugging CUPTI Investigation Context

From **arr-coc-0-1 CUPTI Investigation** (production debugging, 2025-11-13):

**Problem:**
```bash
# Docker build succeeded
cmake ..  # Found CUPTI (headers in devel image)
make install

# Runtime failed
RuntimeError: CUPTI initialization failed
```

**Root Cause Discovery:**
```bash
# CMake claimed success
$ grep CUPTI build/CMakeCache.txt
CUPTI_INCLUDE_DIR=/usr/local/cuda/extras/CUPTI/include  # ✓ Found
CUPTI_LIBRARY=CUPTI_LIBRARY-NOTFOUND                    # ✗ Missing

USE_CUPTI:BOOL=ON         # ← Still enabled!
LIBKINETO_NOCUPTI:BOOL=OFF  # ← Expects CUPTI

# CMake detected headers (from cuda-libraries-dev)
# But library not present in runtime image
# PyTorch built with CUPTI support
# Runtime dlopen() fails
```

**Solution Applied:**
```bash
# Force disable CUPTI at build time
ENV LIBKINETO_NOCUPTI=1
ENV USE_CUPTI=0

# Rebuild PyTorch
python setup.py install

# Verify
$ grep CUPTI build/CMakeCache.txt
LIBKINETO_NOCUPTI:BOOL=ON  # ✓ CPU-only Kineto
USE_CUPTI:BOOL=OFF         # ✓ CUPTI disabled
```

---

## Sources

**Source Documents:**
- [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) - Build process overview
- [cuda/11-pytorch-profiling-ecosystem.md](11-pytorch-profiling-ecosystem.md) - CUPTI/Kineto profiling
- [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md) - CUDA package structure

**Web Research:**
- [PyTorch Issue #14333: FindCUDA error when running cmake](https://github.com/pytorch/pytorch/issues/14333) - GitHub pytorch/pytorch (accessed 2025-11-13)
- [PyTorch cmake/Dependencies.cmake](https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/update-test-libtorch-path/cmake/Dependencies.cmake) - GitLab Maison de la Simulation (accessed 2025-11-13)
- [PyTorch Profiler CUPTI warning](https://discuss.pytorch.org/t/pytorch-profiler-cupti-warning/131011) - PyTorch Forums (accessed 2025-11-13)
- [CMake FindCUDA Module](https://cmake.org/cmake/help/latest/module/FindCUDA.html) - CMake Documentation (accessed 2025-11-13)

**Production Debugging Context:**
- arr-coc-0-1 CUPTI Investigation (2025-11-13) - Docker multi-stage build CUPTI detection false positive discovery
- CMake header-only detection pattern identified during PyTorch compilation troubleshooting
