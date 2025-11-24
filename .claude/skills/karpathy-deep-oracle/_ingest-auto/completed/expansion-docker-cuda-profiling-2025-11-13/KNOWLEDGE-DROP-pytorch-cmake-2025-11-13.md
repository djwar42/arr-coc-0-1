# KNOWLEDGE DROP: PyTorch CMake Build Internals

**Runner**: PART 3
**Timestamp**: 2025-11-13
**Status**: ✓ SUCCESS

---

## Knowledge File Created

**File**: `cuda/12-pytorch-cmake-build-internals.md` (~400 lines)

**Sections**:
1. CMake Detection Logic (~120 lines) - FindCUDA.cmake, header-only detection, version mismatch
2. Optional Dependencies (~120 lines) - Force disabling CUPTI/Kineto, CMakeCache.txt interpretation
3. Build vs Runtime Dependencies (~80 lines) - What's needed when, Docker stage implications
4. Debugging CMake Issues (~80 lines) - CMAKE_FIND_DEBUG_MODE, false positive patterns, CUPTI investigation

---

## Web Sources Used

1. **[PyTorch Issue #14333: FindCUDA error when running cmake](https://github.com/pytorch/pytorch/issues/14333)** (GitHub, accessed 2025-11-13)
   - CMake version mismatch detection logic
   - CUDA_HOME vs CUDA_NVCC_EXECUTABLE conflicts
   - "FindCUDA says version X but headers say version Y" error

2. **[PyTorch cmake/Dependencies.cmake](https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/update-test-libtorch-path/cmake/Dependencies.cmake)** (GitLab, accessed 2025-11-13)
   - CUPTI detection logic: `find_library(CUPTI_LIBRARY ...)`
   - LIBKINETO_NOCUPTI flag usage
   - Message output: "Found CUPTI" vs "Could not find CUPTI library"

3. **[PyTorch Profiler CUPTI warning](https://discuss.pytorch.org/t/pytorch-profiler-cupti-warning/131011)** (PyTorch Forums, accessed 2025-11-13)
   - Default build option: USE_CUPTI off
   - USE_KINETO and USE_CUPTI_SO options
   - Build configuration recommendations

4. **[CMake FindCUDA Module](https://cmake.org/cmake/help/latest/module/FindCUDA.html)** (CMake Documentation, accessed 2025-11-13)
   - CUDA toolkit search paths
   - FindCUDA.cmake variable reference
   - CUDA_TOOLKIT_ROOT_DIR environment variables

---

## Production Debugging Context

**Source**: arr-coc-0-1 CUPTI Investigation (2025-11-13)

### CMake False Positive Discovery

**Problem Pattern Identified**:
1. Docker multi-stage build (devel → runtime)
2. CMake finds CUPTI headers in devel stage
3. CMake reports "Found CUPTI" based on headers
4. Shared library (`libcupti.so.12`) not copied to runtime stage
5. PyTorch built with `USE_CUPTI=ON`
6. Runtime fails: `RuntimeError: CUPTI initialization failed`

**Root Cause**:
- CMake `find_library()` can succeed on **header detection alone**
- PyTorch cmake/Dependencies.cmake checks for CUPTI headers
- If headers found → sets `LIBKINETO_NOCUPTI=OFF`
- But library validation not enforced until runtime `dlopen()`

**Solution Applied**:
```bash
# Force disable CUPTI at build time
ENV LIBKINETO_NOCUPTI=1
ENV USE_CUPTI=0

# Rebuild PyTorch
python setup.py install
```

**Key Lesson**:
> CMake detection ≠ Runtime availability. Always verify shared libraries exist at runtime, especially in Docker multi-stage builds where headers and libraries may be in different stages.

---

## Knowledge File Highlights

### Header-Only Detection Pattern
```cmake
# CMake searches for CUPTI
find_library(CUPTI_LIBRARY cupti)

# Finds headers (from cuda-libraries-dev)
# Reports success
# But libcupti.so.12 missing
# Runtime: dlopen() fails
```

### CMakeCache.txt Debugging
```bash
# Dangerous pattern found during investigation
USE_CUPTI:BOOL=ON                    # Feature enabled
CUPTI_LIBRARY:FILEPATH=NOTFOUND      # But library missing!
LIBKINETO_NOCUPTI:BOOL=OFF           # Kineto expects CUPTI

# Results in runtime error
```

### Build vs Runtime Gap
- **Build time**: CMake validates headers exist
- **Runtime**: dlopen() validates shared libraries exist
- **Gap**: Headers present, libraries absent → builds succeed, runtime fails

---

## Connection to Existing Knowledge

**Extends**:
- `cuda/02-pytorch-build-system-compilation.md` - Adds CMake detection internals
- `cuda/11-pytorch-profiling-ecosystem.md` - Explains why CUPTI dependency exists

**Complements**:
- `cuda/10-docker-multistage-ml-optimization.md` - Library copying between stages
- `cuda/13-nvidia-container-cuda-packaging.md` - Where CUPTI lives in CUDA packages

**Practical Application**:
- Debugging "CMake found it but runtime fails" errors
- Docker multi-stage build library selection
- Force disabling optional features in PyTorch builds

---

## Line Count

**Total**: 406 lines

**Breakdown**:
- Section 1 (CMake Detection): 124 lines
- Section 2 (Optional Dependencies): 118 lines
- Section 3 (Build vs Runtime): 82 lines
- Section 4 (Debugging): 82 lines

---

**Status**: PART 3 complete ✓
