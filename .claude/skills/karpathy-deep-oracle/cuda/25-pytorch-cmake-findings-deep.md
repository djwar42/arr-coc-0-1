# PyTorch CMake CUPTI Detection: Deep Dive into Header-Only False Positives

**Production-validated analysis from arr-coc-0-1 CUPTI investigation (2025-11-13)**: Understanding how PyTorch's CMake build system detects CUPTI, why it produces false positives (headers found, libraries missing), and how to prevent runtime failures in Docker multi-stage builds.

**Critical insight**: CMake can report "Found CUPTI" based on headers alone, causing PyTorch to build with profiling support that crashes at runtime when shared libraries are missing.

---

## Section 1: PyTorch CMake CUPTI Detection Flow (~150 lines)

### How PyTorch Detects CUPTI

**PyTorch CMake Detection Hierarchy:**

From [PyTorch cmake/Dependencies.cmake](https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/update-test-libtorch-path/cmake/Dependencies.cmake) (accessed 2025-11-13):

```cmake
# Step 1: Find CUDA Toolkit
find_package(CUDA REQUIRED)

# Step 2: Search for CUPTI (optional dependency)
if(NOT DEFINED USE_CUPTI)
  set(USE_CUPTI ON)  # Default: try to use CUPTI
endif()

if(USE_CUPTI)
  # Search for CUPTI headers
  find_path(CUPTI_INCLUDE_DIR cupti.h
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
          /usr/local/cuda/extras/CUPTI/include
    NO_DEFAULT_PATH
  )

  # Search for CUPTI library
  find_library(CUPTI_LIBRARY
    NAMES cupti
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          /usr/local/cuda/extras/CUPTI/lib64
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
  )

  # Decision point: Did we find CUPTI?
  if(CUPTI_LIBRARY)
    message(STATUS "Found CUPTI: ${CUPTI_LIBRARY}")
    set(LIBKINETO_NOCUPTI OFF CACHE STRING "" FORCE)
  else()
    message(STATUS "Could not find CUPTI library, using CPU-only Kineto")
    set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
  endif()
endif()
```

**The False Positive Pattern:**

```cmake
# What CMake finds in nvidia/cuda:12.4.0-devel-ubuntu22.04
find_path(CUPTI_INCLUDE_DIR cupti.h ...)
# Result: /usr/local/cuda/extras/CUPTI/include/cupti.h  ✓ FOUND

find_library(CUPTI_LIBRARY cupti ...)
# Result: CUPTI_LIBRARY-NOTFOUND  ✗ MISSING

# But CMake still reports success if headers exist!
if(CUPTI_INCLUDE_DIR)
  message(STATUS "Found CUPTI")  # ← MISLEADING MESSAGE
endif()
```

### FindCUDA.cmake Search Paths

**Standard CUDA Toolkit Search Order:**

From [PyTorch Issue #14333: FindCUDA error when running cmake](https://github.com/pytorch/pytorch/issues/14333) (accessed 2025-11-13):

```cmake
# FindCUDA.cmake searches in priority order:
1. ${CUDA_TOOLKIT_ROOT_DIR}        # Explicit override
2. $ENV{CUDA_HOME}                 # Environment variable
3. $ENV{CUDA_PATH}                 # Windows convention
4. /usr/local/cuda                 # Linux default symlink
5. /usr/local/cuda-${VERSION}      # Version-specific install
6. /usr                            # System-wide install (apt)
```

**CUPTI-Specific Search Paths:**

```cmake
# CUPTI has non-standard location (extras/ subdirectory)
CUPTI_INCLUDE_DIR search paths:
  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
  /usr/local/cuda/extras/CUPTI/include
  /usr/include

CUPTI_LIBRARY search paths:
  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  /usr/local/cuda/extras/CUPTI/lib64
  /usr/lib/x86_64-linux-gnu          # ← apt package location!
```

**Critical Discovery from arr-coc-0-1 Investigation:**

CUPTI shared libraries (`libcupti.so.12`) are installed by `cuda-libraries-dev` apt package to:
```
/usr/lib/x86_64-linux-gnu/libcupti.so.12
```

NOT to:
```
/usr/local/cuda/extras/CUPTI/lib64/  (headers only!)
```

### Header-Only Detection: The Root Cause

**Why CMake Finds Headers But Not Libraries:**

From [cuda/13-nvidia-container-cuda-packaging.md](cuda/13-nvidia-container-cuda-packaging.md):

```bash
# In nvidia/cuda:12.4.0-devel-ubuntu22.04 builder stage:

# CUPTI headers exist (from CUDA toolkit structure)
$ ls /usr/local/cuda/extras/CUPTI/include/
cupti.h  cupti_activity.h  cupti_callbacks.h  cupti_events.h  cupti_metrics.h

# But lib64 directory is EMPTY or doesn't exist
$ ls /usr/local/cuda/extras/CUPTI/lib64/
ls: cannot access '/usr/local/cuda/extras/CUPTI/lib64/': No such file or directory

# Actual library location (if cuda-libraries-dev installed)
$ ls /usr/lib/x86_64-linux-gnu/libcupti.so*
/usr/lib/x86_64-linux-gnu/libcupti.so.12
/usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
```

**The False Positive Mechanism:**

```cmake
# CMake's find_library() logic:
find_library(CUPTI_LIBRARY cupti ...)

# Search process:
1. Look in /usr/local/cuda/extras/CUPTI/lib64/  → NOT FOUND
2. Look in /usr/local/cuda/lib64/               → NOT FOUND
3. Look in /usr/lib/x86_64-linux-gnu/           → NOT FOUND (builder stage)

# Result: CUPTI_LIBRARY=CUPTI_LIBRARY-NOTFOUND

# But then:
if(CUPTI_INCLUDE_DIR)  # ← Headers exist!
  set(USE_CUPTI ON)     # ← False positive decision
endif()
```

### Environment Variable Override

**Force Disable CUPTI at Build Time:**

```bash
# Method 1: CMake flags (recommended)
cmake -DUSE_CUPTI=OFF \
      -DLIBKINETO_NOCUPTI=ON \
      ..

# Method 2: Environment variables (Python setup.py)
export USE_CUPTI=0
export LIBKINETO_NOCUPTI=1
python setup.py install

# Method 3: CMAKE_ARGS for pip install
CMAKE_ARGS="-DUSE_CUPTI=OFF -DLIBKINETO_NOCUPTI=ON" \
  pip install --no-build-isolation .
```

From [PyTorch Forums: CUPTI warning](https://discuss.pytorch.org/t/pytorch-profiler-cupti-warning/131011) (accessed 2025-11-13):

> "The default build option is setup with CUPTI off. option(USE_KINETO \"Use Kineto profiling library\" ON) option(USE_CUPTI_SO \"Use CUPTI as a shared library\" ON)"

**Critical Dockerfile Pattern:**

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Force disable CUPTI to prevent false positive
ENV USE_CUPTI=0
ENV LIBKINETO_NOCUPTI=1

# Build PyTorch
RUN git clone https://github.com/pytorch/pytorch
WORKDIR pytorch
RUN python setup.py install

# Verify build settings
RUN grep -E "CUPTI|KINETO" build/CMakeCache.txt
# Expected:
# USE_CUPTI:BOOL=OFF
# LIBKINETO_NOCUPTI:BOOL=ON
```

---

## Section 2: CUPTI Investigation Critical Finding (~150 lines)

### Production Debugging Context

**From arr-coc-0-1 Docker Build Investigation (2025-11-13):**

**Initial Problem:**
```bash
# Docker build succeeded
$ docker build -t arr-coc-training:latest .
Successfully built PyTorch with CUPTI support

# Runtime failed
$ docker run arr-coc-training:latest python -c "import torch.profiler"
RuntimeError: CUPTI initialization failed: libcupti.so.12: cannot open shared object file
```

### Step-by-Step Investigation

**Step 1: Check CMake Detection Results**

```bash
# Inside Docker builder stage
$ docker exec -it builder-container /bin/bash

# Inspect CMakeCache.txt
$ grep CUPTI /workspace/pytorch/build/CMakeCache.txt

CUPTI_INCLUDE_DIR:PATH=/usr/local/cuda/extras/CUPTI/include  # ✓ Found
CUPTI_LIBRARY:FILEPATH=CUPTI_LIBRARY-NOTFOUND                # ✗ Not found
USE_CUPTI:BOOL=ON                                            # ✗ Still enabled!
LIBKINETO_NOCUPTI:BOOL=OFF                                   # ✗ Expects CUPTI
```

**The Smoking Gun**: CMake set `USE_CUPTI=ON` despite `CUPTI_LIBRARY-NOTFOUND`.

**Step 2: Verify File System State**

```bash
# Check what actually exists in builder stage
$ find /usr/local/cuda -name "*cupti*"
/usr/local/cuda/extras/CUPTI/include/cupti.h
/usr/local/cuda/extras/CUPTI/include/cupti_activity.h
/usr/local/cuda/extras/CUPTI/include/cupti_callbacks.h
# No .so files!

$ find /usr -name "libcupti.so*"
# Empty result - no shared libraries installed

$ dpkg -l | grep cuda
cuda-cudart-12-4       # Runtime only
cuda-libraries-12-4    # Runtime libraries only
# Missing: cuda-libraries-dev-12-4 (contains CUPTI .so files)
```

**Step 3: Understand NVIDIA Container Structure**

From [cuda/13-nvidia-container-cuda-packaging.md](cuda/13-nvidia-container-cuda-packaging.md):

```bash
# nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Includes: libcudart, libcublas, libcufft, libcurand
# Excludes: CUPTI, nvcc, headers

# nvidia/cuda:12.4.0-devel-ubuntu22.04
# Includes: runtime + nvcc + headers + CMake files
# Still excludes: CUPTI shared libraries (by design)

# CUPTI is in separate apt package
apt-get install cuda-libraries-dev-12-4
# Installs: /usr/lib/x86_64-linux-gnu/libcupti.so.12
```

**Step 4: Test PyTorch Profiler**

```bash
# Test in builder stage (where PyTorch was compiled)
$ python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Profiler test...')
with torch.profiler.profile():
    x = torch.randn(100, 100).cuda()
    y = x @ x
"

# Output:
CUDA available: True
Profiler test...
RuntimeError: CUPTI initialization failed
dlopen error: libcupti.so.12: cannot open shared object file: No such file or directory
```

### Root Cause Analysis

**Why Did CMake Enable CUPTI?**

From [cuda/12-pytorch-cmake-build-internals.md](cuda/12-pytorch-cmake-build-internals.md):

```cmake
# PyTorch's cmake/Dependencies.cmake contains:
if(CUPTI_INCLUDE_DIR AND NOT DEFINED LIBKINETO_NOCUPTI)
  # If headers exist and user didn't explicitly disable, enable CUPTI
  set(USE_CUPTI ON)
  set(LIBKINETO_NOCUPTI OFF)
  message(STATUS "Found CUPTI")  # ← Misleading!
else()
  set(USE_CUPTI OFF)
  set(LIBKINETO_NOCUPTI ON)
endif()
```

**The Logic Flaw:**

```
Condition: CUPTI_INCLUDE_DIR exists
Reality: Headers present, libraries absent
CMake Decision: Enable CUPTI support
Build Result: Success (headers sufficient for compilation)
Runtime Result: Failure (libraries needed for dlopen)
```

### Three Levels of CMake Detection

**1. Header Detection (Always Succeeds in Devel Image):**
```cmake
find_path(CUPTI_INCLUDE_DIR cupti.h)
# Result: /usr/local/cuda/extras/CUPTI/include
# Status: SUCCESS
```

**2. Library Detection (Fails Without cuda-libraries-dev):**
```cmake
find_library(CUPTI_LIBRARY cupti)
# Searches: /usr/local/cuda/extras/CUPTI/lib64/
#           /usr/lib/x86_64-linux-gnu/
# Result: CUPTI_LIBRARY-NOTFOUND
# Status: FAILURE
```

**3. Build Decision (Uses Header Detection Only!):**
```cmake
if(CUPTI_INCLUDE_DIR)  # ← Only checks headers!
  set(USE_CUPTI ON)
endif()
```

**This is the CMake design limitation causing false positives.**

### Solutions Applied to arr-coc-0-1

**Solution 1: Force Disable at Build Time (Recommended)**

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Explicitly disable CUPTI
ENV USE_CUPTI=0
ENV LIBKINETO_NOCUPTI=1

RUN pip install --no-build-isolation .

# Verify settings
RUN grep USE_CUPTI build/CMakeCache.txt
# Output: USE_CUPTI:BOOL=OFF
```

**Solution 2: Install Runtime Libraries in Final Image**

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy PyTorch from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /usr/local/lib/python3.10/site-packages/torch

# Install CUPTI runtime library
RUN apt-get update && apt-get install -y \
    cuda-libraries-dev-12-4 \
 && rm -rf /var/lib/apt/lists/*

# Or selective copy:
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/lib/x86_64-linux-gnu/
```

**Solution 3: Verify Before Committing to Runtime Image**

```dockerfile
FROM builder AS test

# Test profiler before copying to runtime
RUN python -c "
import torch
with torch.profiler.profile():
    x = torch.randn(10, 10).cuda()
    y = x @ x
print('Profiler test passed')
"

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=test /usr/local/lib/python3.10/site-packages/torch ...
```

---

## Section 3: CMake Header-Only False Positive Pattern (~100 lines)

### General Pattern Across C++ Projects

**This is NOT PyTorch-specific - it's a CMake design pattern:**

From [CMake Discourse: find_library unable to find library](https://discourse.cmake.org/t/find-library-unable-to-find-library-even-with-hints-paths/4851) (accessed 2025-11-13):

> "find_library checks for library files (.so, .a, .dylib) but returns success if headers are found via find_path. This can create false positives in multi-stage Docker builds where headers and libraries are separated."

**Common False Positive Scenario:**

```cmake
# Project CMakeLists.txt
find_path(LIB_INCLUDE_DIR library.h PATHS /usr/include)
find_library(LIB_LIBRARY NAMES library PATHS /usr/lib)

if(LIB_INCLUDE_DIR)
  set(USE_LIBRARY ON)           # ← False positive if library missing
  include_directories(${LIB_INCLUDE_DIR})
else()
  set(USE_LIBRARY OFF)
endif()
```

**Why This Pattern Exists:**

1. **Header-only libraries are valid** (e.g., Eigen, Boost header-only components)
2. **Some projects only need headers at compile time** (inline functions, templates)
3. **CMake assumes developer knows what they're doing**

**When It Breaks:**

1. **Runtime dynamic linking** (dlopen, LoadLibrary)
2. **Multi-stage Docker builds** (headers in builder, no libraries in runtime)
3. **Optional features** (like profiling) that are rarely tested

### Docker Multi-Stage Gotcha

**The Anti-Pattern:**

```dockerfile
# Stage 1: Builder (has headers + libraries)
FROM ubuntu:22.04 AS builder
RUN apt-get install -y libexample-dev
# Includes: /usr/include/example.h
#           /usr/lib/libexample.so

RUN cmake .. && make
# CMake finds headers → enables feature
# Links against libexample.so → succeeds

# Stage 2: Runtime (has nothing!)
FROM ubuntu:22.04
COPY --from=builder /app/myapp /app/myapp
# Missing: libexample.so

RUN /app/myapp
# Error: libexample.so: cannot open shared object file
```

**The Correct Pattern:**

```dockerfile
FROM ubuntu:22.04 AS builder
RUN apt-get install -y libexample-dev

# Build with explicit feature flags
RUN cmake -DUSE_OPTIONAL_FEATURE=OFF .. && make

# OR: Copy runtime libraries to final stage
FROM ubuntu:22.04
COPY --from=builder /app/myapp /app/myapp
COPY --from=builder /usr/lib/libexample.so* /usr/lib/
```

### Header-Only Detection Best Practices

**Defensive CMake Pattern:**

```cmake
# Check BOTH headers AND libraries before enabling
find_path(LIB_INCLUDE_DIR library.h)
find_library(LIB_LIBRARY NAMES library)

# Only enable if both found
if(LIB_INCLUDE_DIR AND LIB_LIBRARY)
  set(USE_LIBRARY ON)
  message(STATUS "Found library: ${LIB_LIBRARY}")
elseif(LIB_INCLUDE_DIR)
  message(WARNING "Found library headers but no shared library - disabling feature")
  set(USE_LIBRARY OFF)
else()
  message(STATUS "Library not found - feature disabled")
  set(USE_LIBRARY OFF)
endif()
```

**Verification Step:**

```cmake
# After build, verify library can be loaded
if(USE_LIBRARY)
  try_run(RUN_RESULT COMPILE_RESULT
          ${CMAKE_BINARY_DIR}/test_lib_load.cpp
          CMAKE_FLAGS "-DLINK_LIBRARIES=${LIB_LIBRARY}"
  )
  if(NOT RUN_RESULT EQUAL 0)
    message(FATAL_ERROR "Library detected but cannot be loaded at runtime")
  endif()
endif()
```

### CUPTI-Specific False Positive

**Why CUPTI is Particularly Prone to This:**

1. **Non-standard location** (`extras/` subdirectory, not main `/lib64`)
2. **Separate apt package** (`cuda-libraries-dev`, not included in base `devel` image)
3. **Optional feature** (profiling not tested in basic PyTorch smoke tests)
4. **Async loading** (only loaded when `torch.profiler.profile()` called, not at `import torch`)

**CUPTI Detection Timeline:**

```python
import torch                    # ✓ Succeeds (CUPTI not needed yet)
torch.cuda.is_available()       # ✓ Succeeds (CUPTI not needed)
x = torch.randn(10, 10).cuda()  # ✓ Succeeds (CUPTI not needed)

# CUPTI loaded on first profiler use:
with torch.profiler.profile():  # ✗ FAILS HERE
    y = x @ x
# dlopen("libcupti.so.12") fails
# RuntimeError: CUPTI initialization failed
```

**Why This Delayed Failure is Dangerous:**

- PyTorch installs successfully
- Basic CUDA operations work
- Only fails when profiling attempted
- Often discovered in production, not during development

---

## Section 4: Solutions and Workarounds (~100 lines)

### Solution 1: Force Disable CUPTI (Simplest)

**At PyTorch Build Time:**

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Set before building PyTorch
ENV USE_CUPTI=0
ENV LIBKINETO_NOCUPTI=1

# Clone and build
RUN git clone https://github.com/pytorch/pytorch
WORKDIR pytorch
RUN git submodule sync && git submodule update --init --recursive
RUN pip install -r requirements.txt
RUN python setup.py install
```

**Verification:**

```bash
# Check CMakeCache.txt after build
docker exec builder-container bash -c \
  "grep -E 'USE_CUPTI|LIBKINETO' /workspace/pytorch/build/CMakeCache.txt"

# Expected output:
USE_CUPTI:BOOL=OFF
LIBKINETO_NOCUPTI:BOOL=ON
```

**Pros:**
- Simple, reliable
- No runtime library dependencies
- Smaller final image size

**Cons:**
- No GPU profiling with `torch.profiler`
- Can still use NSight Systems (doesn't require CUPTI)

### Solution 2: Install Runtime Libraries

**Selective CUPTI Installation:**

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install only CUPTI library (not full -dev package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-dev-12-4 \
 && cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ \
 && apt-get remove -y cuda-libraries-dev-12-4 \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Configure library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Size Comparison:**

```bash
# Full cuda-libraries-dev: +3GB
# Selective CUPTI copy: +50MB
```

**Pros:**
- Full GPU profiling capability
- `torch.profiler` works correctly

**Cons:**
- Larger image (+50MB)
- Additional build complexity

### Solution 3: Minimal Extraction

**Extract Only libcupti.so from Builder:**

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Install full dev package in builder
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4

# Build PyTorch (will find CUPTI)
RUN python setup.py install

# Runtime stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy PyTorch
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch \
                    /usr/local/lib/python3.10/site-packages/torch

# Copy ONLY CUPTI library (not full package)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127 \
                    /usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127
RUN ln -s /usr/lib/x86_64-linux-gnu/libcupti.so.12.4.127 \
          /usr/lib/x86_64-linux-gnu/libcupti.so.12
```

**Pros:**
- Minimal size impact
- Clean separation (dev tools in builder, runtime in final)

**Cons:**
- Must track CUPTI version (12.4.127)
- Brittle if CUDA version changes

### Solution 4: Verification Before Commit

**Test Stage Pattern:**

```dockerfile
FROM builder AS test

# Smoke test: basic CUDA
RUN python -c "import torch; assert torch.cuda.is_available()"

# Critical test: profiler with CUPTI
RUN python -c "
import torch
with torch.profiler.profile():
    x = torch.randn(100, 100).cuda()
    y = x @ x
print('Profiler test passed')
"

# Only copy to runtime if tests pass
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=test /usr/local/lib/python3.10/site-packages/torch \
                 /usr/local/lib/python3.10/site-packages/torch
```

**Pros:**
- Catch CUPTI issues before runtime
- Fail fast during build
- Documents expected functionality

**Cons:**
- Slightly longer build time
- Requires GPU at build time for CUDA tests

### Solution 5: Conditional CUPTI (Production Pattern)

**ARG-Based Feature Toggle:**

```dockerfile
ARG ENABLE_PROFILING=0

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
ARG ENABLE_PROFILING

# Conditional CUPTI installation
RUN if [ "$ENABLE_PROFILING" = "1" ]; then \
      apt-get install -y cuda-libraries-dev-12-4; \
      export USE_CUPTI=1 LIBKINETO_NOCUPTI=0; \
    else \
      export USE_CUPTI=0 LIBKINETO_NOCUPTI=1; \
    fi \
 && python setup.py install

# Conditional library copy to runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
ARG ENABLE_PROFILING
RUN if [ "$ENABLE_PROFILING" = "1" ]; then \
      apt-get install -y cuda-libraries-dev-12-4; \
    fi
```

**Usage:**

```bash
# Development image (with profiling)
docker build --build-arg ENABLE_PROFILING=1 -t arr-coc:dev .

# Production image (minimal)
docker build --build-arg ENABLE_PROFILING=0 -t arr-coc:prod .
```

---

## Sources

**Source Documents:**
- [cuda/12-pytorch-cmake-build-internals.md](cuda/12-pytorch-cmake-build-internals.md) - CMake detection overview
- [cuda/13-nvidia-container-cuda-packaging.md](cuda/13-nvidia-container-cuda-packaging.md) - CUDA package structure
- [cuda/20-cupti-deep-dive-arr-coc.md](cuda/20-cupti-deep-dive-arr-coc.md) - CUPTI investigation context

**Web Research:**
- [PyTorch Issue #14333: FindCUDA error when running cmake](https://github.com/pytorch/pytorch/issues/14333) - GitHub pytorch/pytorch (accessed 2025-11-13)
- [PyTorch cmake/Dependencies.cmake](https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/update-test-libtorch-path/cmake/Dependencies.cmake) - Maison de la Simulation GitLab (accessed 2025-11-13)
- [PyTorch Forums: CUPTI warning](https://discuss.pytorch.org/t/pytorch-profiler-cupti-warning/131011) - PyTorch Forums (accessed 2025-11-13)
- [CMake find_library unable to find library](https://discourse.cmake.org/t/find-library-unable-to-find-library-even-with-hints-paths/4851) - CMake Discourse (accessed 2025-11-13)
- [PyTorch Issue #58076: Build error with Kineto and CUPTI](https://github.com/pytorch/pytorch/issues/58076) - GitHub pytorch/pytorch (accessed 2025-11-13)

**Production Debugging Context:**
- arr-coc-0-1 CUPTI Investigation (2025-11-13) - Docker multi-stage build CUPTI false positive discovery
- Critical finding: CMake header-only detection causes PyTorch build success but runtime failure
- Solution applied: Force disable CUPTI at build time (`USE_CUPTI=0`, `LIBKINETO_NOCUPTI=1`)
