# KNOWLEDGE DROP: Docker Multi-Stage ML Optimization

**Runner**: PART 1
**Timestamp**: 2025-11-13
**Status**: ✓ COMPLETE

---

## Knowledge File Created

**File**: `cuda/10-docker-multistage-ml-optimization.md`
**Lines**: 407 lines
**Size**: ~28KB

---

## Content Summary

Created comprehensive guide on Docker multi-stage builds for ML/CUDA production workloads:

### Section 1: Multi-Stage Build Fundamentals (100 lines)
- Builder vs runtime architecture patterns
- NVIDIA base image variants (base, runtime, devel, cudnn)
- Size comparisons: 800MB (base) → 2.9GB (devel) → 5GB (cudnn-devel)
- Key insight: PyTorch bundles CUDA libraries (no separate runtime needed!)

### Section 2: Layer Caching & BuildKit (100 lines)
- BuildKit cache mount strategies (`RUN --mount=type=cache`)
- pip --no-cache-dir pattern (saves 2.5GB)
- ARG cache invalidation strategies
- Multi-stage caching patterns
- BuildKit secrets for private registries

### Section 3: Library Selection Between Stages (100 lines)
- Essential CUDA runtime libraries
- Optional profiling tools (CUPTI, NSight) trade-offs
- Development headers (cuda-minimal-build: 100MB vs 5GB)
- Selective library copying patterns

### Section 4: Production Patterns (100 lines)
- Minimal runtime images (2.9GB final size, 63% reduction)
- Optional dev layer mounting (separate dev/prod targets)
- CI/CD optimization strategies (GitHub Actions cache)
- Layer optimization checklist (production-tested)

---

## Web Sources Used

1. **[Optimizing PyTorch Docker images](https://mveg.es/posts/optimizing-pytorch-docker-images-cut-size-by-60percent/)** - Marton Veges, August 2024
   - Critical discovery: PyTorch includes CUDA binaries
   - 60% size reduction techniques
   - base vs runtime vs devel comparisons

2. **[Docker Multi-stage Documentation](https://docs.docker.com/build/building/multi-stage/)** - Docker Official Docs
   - Multi-stage fundamentals
   - BuildKit vs legacy builder
   - Named stages and target builds

3. **[nvidia/cuda Docker Hub](https://hub.docker.com/r/nvidia/cuda)** - NVIDIA Official
   - Image variant descriptions
   - Size specifications

4. **[BuildKit Cache Backends](https://docs.docker.com/build/cache/backends/)** - Docker Docs
   - Cache mount patterns
   - GitHub Actions integration

---

## Production Debugging Context

This knowledge emerged from arr-coc-0-1 CUPTI investigation:
- Discovered that PyTorch containers can use minimal `base` images
- CUPTI profiling tools optional (50MB+ overhead)
- cuda-minimal-build sufficient for custom kernels (100MB vs 5GB)
- Production builds should prioritize size/security over dev tools

**Real-world impact**:
- Baseline naive Dockerfile: 7.6GB
- Optimized multi-stage: 2.9GB (62% reduction)
- Build time with cache: 30 seconds (vs 10 minutes cold)

---

## Key Takeaways

1. **PyTorch bundles CUDA** - No separate nvidia/cuda:*-runtime needed
2. **base images sufficient** - 800MB vs 1.5GB runtime vs 2.9GB devel
3. **pip --no-cache-dir** - Saves 2.5GB instantly
4. **BuildKit cache mounts** - 92% faster rebuilds
5. **Multi-stage selective copy** - Only transfer compiled artifacts
6. **Production security** - Non-root, minimal packages, no build tools

---

**Knowledge file ready for oracle indexing**: cuda/10-docker-multistage-ml-optimization.md
