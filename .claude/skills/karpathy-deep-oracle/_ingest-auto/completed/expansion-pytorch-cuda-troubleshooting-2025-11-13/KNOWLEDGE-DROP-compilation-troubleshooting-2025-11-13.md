# KNOWLEDGE DROP: CUDA Compilation Troubleshooting (Expert-Level)

**Runner**: PART 1
**Timestamp**: 2025-11-13 15:45
**Status**: ✓ COMPLETE

---

## Knowledge File Created

**File**: `cuda/08-compilation-troubleshooting-expert.md` (425 lines)

**Content Overview**:
- Section 1: nvcc Compilation Errors (architecture mismatch, missing headers, linker errors)
- Section 2: CMake & Build System Issues (FindCUDA.cmake, TORCH_CUDA_ARCH_LIST)
- Section 3: Environment & Dependency Problems (PATH, LD_LIBRARY_PATH, driver versions)
- Section 4: Expert-Level Fixes (CMakeLists.txt patching, diagnostic checklist, production deployment)

---

## Sources Used

**Web Research (Accessed 2025-11-13)**:
1. **NVIDIA Developer Forums** - nvcc fatal unsupported gpu architecture (sm_75 to sm_80)
2. **GitHub PyTorch Issues #90757** - cuda.h missing during torch.compile (CUDA_HOME fix)
3. **GitHub PyTorch Issues #4913** - CMake error when building from source (linker errors)
4. **PyTorch Forums** - Compile PyTorch source with CUDA enabled (CMake detection)
5. **PyTorch Forums** - PyTorch 2.0.1 not recognizing CUDA 11.4 (PATH/LD_LIBRARY_PATH)

**Source Documents**:
- cuda/02-pytorch-build-system-compilation.md (basic compilation - identified ~100 lines of basic errors, expanded to 400+ lines expert-level)

---

## Knowledge Gaps Filled

**Before (existing knowledge)**:
- Basic compilation errors (5 common errors, ~100 lines)
- Generic troubleshooting steps
- Limited environment variable guidance

**After (new expert knowledge)**:
1. **Architecture Mismatch Solutions** - PTX forward compatibility, multi-arch builds, CUDA version matrix
2. **Missing CUDA Headers** - CUDA_HOME diagnostic workflow, conda vs system CUDA conflicts
3. **Linker Error Diagnosis** - ABI compatibility, LD_LIBRARY_PATH ordering, library cache fixes
4. **CMake Configuration Hell** - FindCUDA.cmake debugging, TORCH_CUDA_ARCH_LIST auto-detection
5. **Environment Variable Chaos** - PATH ordering (system before conda), clean environment scripts
6. **Driver/Runtime Mismatch** - Forward compatibility packages, version matrix
7. **cuDNN Version Issues** - Version matching, rebuild procedures
8. **Expert-Level Fixes** - CMakeLists.txt patching, force architecture compilation, verbose debugging
9. **Production Deployment Checklist** - Complete diagnostic script with 7-step verification

---

## Key Expert Additions

**Real-World Error Messages** (from 2024-2025 production systems):
- `nvcc fatal: Unsupported gpu architecture 'compute_80'` (T4 → A100 compilation)
- `fatal error: cuda.h: No such file or directory` (CUDA_HOME not set)
- `undefined reference to cudaGetDeviceCount` (linker flag issues)
- `CMake Error: Could not find CUDA` (CMAKE_PREFIX_PATH not set)
- `RuntimeError: CUDA driver version is insufficient` (450.102.04 driver, 11.8 runtime)

**Diagnostic Commands**:
- Architecture detection: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
- Library loading verification: `ldd $(python -c "import torch; print(torch._C.__file__)") | grep cuda`
- CUDA version compatibility: Driver → Runtime → PyTorch matrix
- PATH ordering check: Remove conda paths, prepend system CUDA
- Complete 7-step diagnostic script for production deployment

**Expert-Level Solutions**:
- Custom CMakeLists.txt patches for non-standard CUDA paths
- Force architecture compilation (override auto-detection)
- Verbose build debugging (DEBUG=1, VERBOSE=1, serial builds)
- Production environment verification script
- Quick fixes for 5 common production errors

---

## Technical Depth

**Audience**: Production ML engineers, cluster administrators, from-source builders

**Coverage**:
- Basic troubleshooting: ✓ (existing knowledge expanded)
- Intermediate debugging: ✓ (CMake, environment variables)
- Expert-level fixes: ✓✓ (patching, forcing, production deployment)
- Hard edge cases: ✓ (driver/runtime mismatch, multi-CUDA installations, conda conflicts)

**Expertise Level**:
- Basic compilation errors → 20%
- Intermediate CMake/environment → 40%
- Expert-level debugging/patching → 30%
- Production deployment/verification → 10%

---

## Integration

**Cross-References Created**:
- Links to cuda/02-pytorch-build-system-compilation.md (basic process)
- Links to cuda/03-compute-capabilities-gpu-architectures.md (architecture versions)
- Links to cuda/00-streams-concurrency-async.md (runtime configuration)
- Links to cuda/04-pytorch-custom-cuda-extensions.md (extension compilation)
- Links to cuda/06-pytorch-jit-torch-compile.md (JIT compilation errors)

**File Location**: `cuda/08-compilation-troubleshooting-expert.md`

**Next Files in Series** (from ingestion plan):
- cuda/09-runtime-errors-debugging-expert.md (PART 2)
- cuda/10-performance-debugging-profiling-expert.md (PART 3)
- cuda/11-advanced-troubleshooting-multi-gpu-expert.md (PART 4)

---

**Status**: PART 1 complete ✓ - Expert compilation troubleshooting knowledge successfully integrated into karpathy-deep-oracle skill.
