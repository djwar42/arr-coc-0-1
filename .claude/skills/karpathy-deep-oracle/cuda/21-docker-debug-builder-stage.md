# Docker Builder Stage Interactive Debugging

## Overview

Interactive debugging of Docker multi-stage builds is essential for troubleshooting compilation failures, missing libraries, and build environment issues. This guide covers production-proven techniques for debugging intermediate builder stages, with specific focus on CUDA/PyTorch builds and the CUPTI investigation that revealed critical CMake false positive patterns.

**Key insight from arr-coc-0-1 CUPTI investigation**: Builder stage debugging revealed that PyTorch CMake found CUPTI headers but missed runtime libraries - a "builds successfully, crashes at runtime" pattern that cost hours of debugging until interactive exploration exposed the gap.

---

## Section 1: Builder Stage Debugging Fundamentals (~100 lines)

### Builder vs Runtime Stage Architecture

Multi-stage builds separate compilation (builder) from execution (runtime):

```dockerfile
# Builder stage - Large, has compilation tools
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y build-essential cmake
COPY pytorch-src/ /src/
RUN cd /src && python setup.py build

# Runtime stage - Minimal, production-ready
FROM nvidia/cuda:12.4.0-base-ubuntu22.04
COPY --from=builder /src/build/lib.linux-x86_64-3.10 /usr/local/lib/
CMD ["python3", "train.py"]
```

**Why debug the builder stage?**
- Compilation failures only occur in builder (nvcc errors, missing headers)
- Runtime failures may originate from builder issues (missing .so files)
- Size optimization requires understanding what builder creates

From [Docker Multi-stage Documentation](https://docs.docker.com/build/building/multi-stage/) (accessed 2025-11-13):
> "Each FROM instruction can use a different base, and each begins a new stage of the build. You can selectively copy artifacts from one stage to another."

### Common Build Failures in Builder Stage

**1. Missing libraries (CUPTI example)**
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get install -y cuda-minimal-build-12-4
# PyTorch build finds cupti.h (headers present)
# But libcupti.so missing → runtime import fails
```

**Symptom**: Build succeeds, `import torch.profiler` crashes with:
```
ImportError: libcupti.so.12: cannot open shared object file
```

**2. CMake detection false positives**
```
-- Found CUPTI: /usr/local/cuda/extras/CUPTI/include
[PyTorch build completes successfully]
[Runtime crashes - library not actually available]
```

From arr-coc-0-1 CUPTI investigation (2025-11-13):
- CMake's `find_path(CUPTI_INCLUDE_DIR cupti.h)` succeeds (headers in devel image)
- CMake's `find_library(CUPTI_LIBRARY cupti)` silently fails (no .so files)
- PyTorch build doesn't enforce library presence, only headers
- Result: "Successful" build that crashes at runtime

**3. Compilation errors (architecture mismatch)**
```
nvcc fatal: Unsupported gpu architecture 'compute_80'
```

**Cause**: Building for A100 (sm_80) on T4 (sm_75) base image.

### Why Debug Builder Stage Specifically

**Traditional approach (failed)**:
```bash
docker build -t myapp .
# Build fails at step 27/45
# Must restart entire build to test fix
# Iteration time: 10+ minutes per attempt
```

**Interactive builder debugging (success)**:
```bash
docker build --target=builder -t myapp-builder .
docker run -it myapp-builder /bin/bash
# Explore filesystem, test commands
# Find issue in 2 minutes
# Fix Dockerfile, rebuild
```

**Time savings**:
- Traditional: 10 builds × 10 minutes = 100 minutes to find CUPTI issue
- Interactive: 1 build + 2 minutes exploration = 12 minutes total
- **88% faster debugging**

---

## Section 2: Interactive Debug Techniques (~150 lines)

### Docker Build --target Flag

Stop at intermediate stage for inspection:

```bash
# Dockerfile with named stages
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y \
    build-essential cmake python3-dev
COPY pytorch/ /pytorch
RUN cd /pytorch && python setup.py build

FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS runtime
COPY --from=builder /pytorch/build /usr/local/lib/
CMD ["python3", "train.py"]

# Build ONLY the builder stage
docker build --target=builder -t pytorch-builder .
```

**Key flags**:
- `--target=<stage-name>` - Stop build at named stage
- `-t <tag>` - Tag the intermediate image for later use
- `--progress=plain` - Verbose output (see all RUN commands)

From [Stack Overflow discussion](https://stackoverflow.com/questions/51253987/) (accessed 2025-11-13):
> "When building a Dockerfile with multiple build stages, --target can be used to specify an intermediate build stage by name as a final stage for the resulting image."

**Example: CUPTI investigation workflow**
```bash
# Build builder stage only
docker build --target=builder -t arr-coc-builder \
  --progress=plain \
  --file Dockerfile.cuda \
  .

# Verbose output shows:
# Step 12/45: RUN python setup.py build
# [CMake] Found CUPTI: /usr/local/cuda/extras/CUPTI/include
# [Build succeeds]
```

### Docker Run -it for Interactive Exploration

Explore built environment interactively:

```bash
# Run builder image with interactive shell
docker run -it pytorch-builder /bin/bash

# Inside container - investigate CUPTI
root@container:/# find / -name "libcupti.so*" 2>/dev/null
# [No output - library missing!]

root@container:/# ls -la /usr/local/cuda/extras/CUPTI/
drwxr-xr-x include/  # Headers present
# lib64/ directory missing - no runtime libraries

root@container:/# python3 -c "import torch.profiler"
Traceback: ImportError: libcupti.so.12: cannot open shared object file

# Ah-ha moment: Headers found, libraries missing
# This is why build succeeded but runtime fails
```

**Discovery commands**:
```bash
# Find all CUDA libraries
find /usr/local/cuda -name "*.so*" | head -20

# Check what's actually installed
dpkg -l | grep cuda

# Verify library dependencies
ldd /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so | grep cupti
# libcupti.so.12 => not found

# Test PyTorch GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Docker Commit - Save Debugging State

Preserve container state during investigation:

```bash
# Start interactive session
docker run -it pytorch-builder /bin/bash

# Inside container: Make experimental changes
apt-get update
apt-get install -y cuda-libraries-dev-12-4
# Now libcupti.so appears!

# Exit container (Ctrl+D)

# Save the modified container
docker ps -a  # Find container ID
docker commit <container-id> pytorch-builder-cupti-fixed

# Test the fixed version
docker run -it pytorch-builder-cupti-fixed python3 -c "import torch.profiler"
# Success! No import error
```

**Use case**: Test fixes before updating Dockerfile.

### BuildKit --progress=plain for Verbose Output

See every command's output:

```bash
# Default (compact output)
docker build .
# #12 [builder 7/10] RUN python setup.py build
# #12 DONE 180.5s

# Verbose (see all output)
DOCKER_BUILDKIT=1 docker build --progress=plain .
# #12 [builder 7/10] RUN python setup.py build
# #12 0.523 running build
# #12 1.234 running build_ext
# #12 45.67 -- Found CUPTI: /usr/local/cuda/extras/CUPTI/include
# #12 46.12 -- CUPTI library: CUPTI_LIBRARY-NOTFOUND
# #12 180.5 Finished processing dependencies
```

**Critical for debugging**: See CMake detection messages, compilation warnings, library search paths.

From [Docker BuildKit Documentation](https://docs.docker.com/build/building/best-practices/) (accessed 2025-11-13):
> "BuildKit's --progress flag controls the verbosity of output. Use 'plain' mode to see all command output for debugging."

### Docker History - Inspect Layer Sizes

Understand what builder stage created:

```bash
docker history pytorch-builder

# IMAGE          CREATED          CREATED BY                                      SIZE
# a1b2c3d4e5f6   2 minutes ago    RUN python setup.py build                       2.3GB
# g7h8i9j0k1l2   5 minutes ago    RUN apt-get install build-essential             450MB
# m3n4o5p6q7r8   10 minutes ago   FROM nvidia/cuda:12.4.0-devel                   4.5GB

# Identify largest layers
docker history pytorch-builder --no-trunc --format "{{.Size}}\t{{.CreatedBy}}" | sort -hr | head -10
```

**Use case**: Optimize builder stage by identifying bloated layers.

---

## Section 3: CUPTI Investigation Debug Workflow (~100 lines)

### How We Debugged PyTorch CMake False Positives

**The problem** (arr-coc-0-1, 2025-11-13):
- Cloud Build: PyTorch compilation succeeded (2 hours, A100-targeted)
- Vertex AI Training: `import torch.profiler` crashed immediately
- Error: `libcupti.so.12: cannot open shared object file`
- **Confusion**: Build logs showed "Found CUPTI"

**Step 1: Reproduce locally with --target=builder**
```bash
# Build only builder stage
cd RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1
docker build --target=builder \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0" \
  -t arr-coc-builder \
  --file Dockerfile \
  .

# Build succeeded (same as Cloud Build)
```

**Step 2: Interactive exploration**
```bash
docker run -it arr-coc-builder /bin/bash

# Check PyTorch profiler
python3 -c "import torch.profiler"
# ImportError: libcupti.so.12: cannot open shared object file

# Search for CUPTI files
find /usr/local/cuda/extras/CUPTI -type f
# /usr/local/cuda/extras/CUPTI/include/cupti.h
# /usr/local/cuda/extras/CUPTI/include/cupti_version.h
# [No .so files found]

# Check installed packages
dpkg -l | grep cupti
# [No packages with "cupti" in name]

# Check what cuda packages ARE installed
dpkg -l | grep cuda-12-4 | grep -E "(dev|runtime|libraries)"
# cuda-cudart-dev-12-4
# cuda-minimal-build-12-4
# [cuda-libraries-dev-12-4 NOT installed]
```

**Discovery**: Headers present (from cuda-minimal-build), runtime libraries absent (need cuda-libraries-dev).

**Step 3: Test fix in container**
```bash
# Still inside arr-coc-builder container
apt-get update
apt-get install -y cuda-libraries-dev-12-4

# Now CUPTI libraries appear
ls /usr/local/cuda/extras/CUPTI/lib64/
# libcupti.so  libcupti.so.12  libcupti.so.12.4.127

# Test profiler
python3 -c "import torch.profiler; print('Success!')"
# Success!
```

**Step 4: Understand why CMake lied**
```bash
# Check PyTorch build cache
cat /pytorch/build/CMakeCache.txt | grep CUPTI

# CUPTI_INCLUDE_DIR:PATH=/usr/local/cuda/extras/CUPTI/include
# CUPTI_LIBRARY:FILEPATH=CUPTI_LIBRARY-NOTFOUND
# USE_CUPTI:BOOL=ON

# Ah-ha! CMake found headers, NOT library
# But build continued anyway
```

From cuda/12-pytorch-cmake-build-internals.md:
> "CMake's FindCUDA.cmake uses find_path() for headers and find_library() for .so files separately. If only headers are found, CMake may report 'Found CUPTI' but set CUPTI_LIBRARY=NOTFOUND, allowing builds to complete without runtime support."

### Finding Missing libcupti.so in Builder Stage

**Verification commands** (run in builder container):

```bash
# 1. Filesystem search
find / -name "libcupti.so*" 2>/dev/null
# Expected: /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12.4.127
# Actual (broken): [no results]

# 2. Shared library cache
ldconfig -p | grep cupti
# Expected: libcupti.so.12 (libc6,x86-64) => /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12
# Actual (broken): [no results]

# 3. Package verification
dpkg -L cuda-libraries-dev-12-4 | grep cupti
# Expected: List of CUPTI files
# Actual (broken): Package not installed

# 4. LD_LIBRARY_PATH check
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i cupti
# Expected: /usr/local/cuda/extras/CUPTI/lib64
# Actual (broken): [no results]
```

### Verifying CUDA Library Presence

**Complete verification script**:
```bash
#!/bin/bash
# cuda-verify.sh - Run inside builder container

echo "=== CUDA Toolkit Verification ==="

echo "1. CUDA Version:"
nvcc --version | grep "release"

echo "2. Installed CUDA packages:"
dpkg -l | grep "^ii" | grep "cuda-12-4" | awk '{print $2}'

echo "3. CUPTI presence:"
if [ -d "/usr/local/cuda/extras/CUPTI/lib64" ]; then
    ls -lh /usr/local/cuda/extras/CUPTI/lib64/*.so* 2>/dev/null || echo "CUPTI lib64 dir exists but no .so files"
else
    echo "CUPTI lib64 directory missing"
fi

echo "4. PyTorch CUDA runtime check:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "5. PyTorch profiler check:"
python3 -c "import torch.profiler; print('Profiler import: SUCCESS')" 2>&1

echo "6. Library dependencies:"
ldd $(python3 -c "import torch; print(torch.__file__)" | sed 's/__init__.py$/lib\/libtorch_cuda.so/')  | grep -E "(cupti|cuda)" | head -5
```

**Run verification**:
```bash
docker run -it arr-coc-builder bash < cuda-verify.sh
```

### Testing PyTorch Profiler in Builder vs Runtime

**Builder stage test** (devel image base):
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get install -y cuda-libraries-dev-12-4  # CUPTI included
RUN pip install torch
RUN python3 -c "import torch.profiler; print('Builder: PASS')"
# Output: Builder: PASS
```

**Runtime stage test** (base image):
```dockerfile
FROM nvidia/cuda:12.4.0-base-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
RUN python3 -c "import torch.profiler; print('Runtime: PASS')"
# Output: ImportError: libcupti.so.12 not found
```

**Fix: Selective CUPTI copy**:
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN apt-get install -y cuda-libraries-dev-12-4
RUN pip install torch

FROM nvidia/cuda:12.4.0-base-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# Copy CUPTI runtime (20MB vs 3GB full devel)
COPY --from=builder /usr/local/cuda/extras/CUPTI/lib64/*.so* /usr/local/cuda/extras/CUPTI/lib64/
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
RUN python3 -c "import torch.profiler; print('Runtime: PASS')"
# Output: Runtime: PASS
```

---

## Section 4: Production Debug Patterns (~50 lines)

### RUN Commands for Verification

Embed verification in Dockerfile:

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Install CUDA libraries with verification
RUN apt-get update && apt-get install -y cuda-libraries-dev-12-4 && \
    ls -la /usr/local/cuda/extras/CUPTI/lib64/*.so* || \
    (echo "ERROR: CUPTI libraries not found" && exit 1)

# Build PyTorch with CUPTI verification
RUN cd /pytorch && python setup.py build && \
    python3 -c "import torch.profiler" || \
    (echo "ERROR: torch.profiler import failed" && exit 1)

# Verify CUDA compute capability
RUN python3 -c "import torch; \
    assert torch.cuda.is_available(), 'CUDA not available'; \
    assert torch.cuda.get_device_capability(0)[0] >= 8, 'A100 required'; \
    print('CUDA verification: PASS')"
```

**Pattern**: Fail fast with clear error messages.

### Find Commands for Discovery

```dockerfile
# Search for libraries (useful for debugging)
RUN find /usr/local/cuda -name "libcupti.so*" -exec ls -lh {} \;

# Find all CUDA .so files and their sizes
RUN find /usr/local/cuda -name "*.so*" -type f -exec du -h {} \; | sort -hr | head -20

# Verify shared library dependencies
RUN python3 -c "import torch" && \
    ldd $(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib/libtorch_cuda.so')") | \
    grep "not found" && exit 1 || echo "All dependencies satisfied"
```

### Conditional Debugging with ARG

```dockerfile
ARG DEBUG=0

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Conditional verbose output
RUN if [ "$DEBUG" = "1" ]; then \
        set -x; \
        apt-cache policy cuda-libraries-dev-12-4; \
        dpkg -l | grep cuda; \
    fi && \
    apt-get update && apt-get install -y cuda-libraries-dev-12-4

# Conditional CUPTI verification
RUN if [ "$DEBUG" = "1" ]; then \
        find /usr/local/cuda/extras/CUPTI -type f; \
        python3 -c "import torch.profiler; print('Profiler test: PASS')"; \
    fi
```

**Usage**:
```bash
# Normal build (fast)
docker build -t myapp .

# Debug build (verbose, verification enabled)
docker build --build-arg DEBUG=1 -t myapp .
```

From cuda/10-docker-multistage-ml-optimization.md:
> "Conditional debugging ARGs enable toggling verbose output and verification commands without maintaining separate Dockerfiles. Set DEBUG=1 during troubleshooting, DEBUG=0 for production builds."

---

## Sources

**Web Research** (accessed 2025-11-13):

1. [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/) - Docker Official Documentation
   - --target flag usage for intermediate stages
   - Named stages and selective copying
   - BuildKit vs legacy builder behavior

2. [Building a multi-stage Dockerfile with --target flag](https://stackoverflow.com/questions/51253987/) - Stack Overflow
   - Stopping at specific build stages
   - Interactive debugging workflows
   - Community best practices

3. [Advanced multi-stage build patterns](https://medium.com/@tonistiigi/advanced-multi-stage-build-patterns-6f741b852fae) - Tõnis Tiigi (Docker maintainer)
   - Complex multi-stage architectures
   - Cache optimization patterns
   - Production debugging techniques

4. [Building best practices](https://docs.docker.com/build/building/best-practices/) - Docker Documentation
   - --progress=plain for verbose output
   - Layer optimization
   - BuildKit features

**Cross-References**:

- [cuda/10-docker-multistage-ml-optimization.md](10-docker-multistage-ml-optimization.md) - Multi-stage build fundamentals
- [cuda/12-pytorch-cmake-build-internals.md](12-pytorch-cmake-build-internals.md) - CMake false positive patterns
- [cuda/13-nvidia-container-cuda-packaging.md](13-nvidia-container-cuda-packaging.md) - CUDA package structure (cuda-libraries-dev contains CUPTI)

**Production Context**:

- **arr-coc-0-1 CUPTI Investigation** (2025-11-13):
  - Problem: PyTorch build succeeded, profiler import failed at runtime
  - Root cause: CMake found cupti.h headers, missed libcupti.so libraries
  - Discovery method: `docker run -it arr-coc-builder bash` + `find / -name "libcupti.so*"`
  - Solution: Install cuda-libraries-dev-12-4, copy CUPTI .so to runtime stage
  - Lesson: Interactive builder debugging exposes CMake false positives that build logs hide

**Additional References**:

- [How can I inspect the file system of a failed Docker build?](https://stackoverflow.com/questions/26220957/) - Stack Overflow community patterns
- [Docker buildx debug build](https://github.com/docker/buildx/issues/1104) - BuildKit interactive debugging feature proposals
