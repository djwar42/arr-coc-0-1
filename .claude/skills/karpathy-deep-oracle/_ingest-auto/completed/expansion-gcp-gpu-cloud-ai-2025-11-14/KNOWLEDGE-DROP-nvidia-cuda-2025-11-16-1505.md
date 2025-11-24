# KNOWLEDGE DROP: NVIDIA Driver & CUDA Management (PART 3)

**Date**: 2025-11-16 15:05
**Part**: PART 3 of GCP GPU & Cloud AI Mastery Expansion
**File Created**: `gcp-gpu/02-nvidia-driver-cuda-management.md`
**Lines**: ~950 lines
**Status**: ✓ Complete

---

## What Was Created

Comprehensive guide to NVIDIA driver and CUDA toolkit management on Google Cloud Platform, covering installation automation, version compatibility, cuDNN configuration, and production best practices.

**File Structure:**
- Section 1: NVIDIA Driver Versions & Selection (~100 lines)
- Section 2: Automated Driver Installation on GCP (~150 lines)
- Section 3: CUDA Toolkit Installation & Configuration (~150 lines)
- Section 4: cuDNN Installation & Management (~150 lines)
- Section 5: Driver Update Strategies (~100 lines)
- Section 6: Troubleshooting & Common Issues (~100 lines)
- Section 7: Production Best Practices (~100 lines)
- Section 8: arr-coc-0-1 Driver Configuration (~50 lines)

---

## Key Knowledge Extracted

### Driver Version Compatibility Matrix

**Critical Compatibility Data:**
- Driver 535.x → CUDA 12.2 (Pascal-Hopper)
- Driver 550.x → CUDA 12.4 (recommended for A100/L4)
- Driver 560.x → CUDA 12.6 (required for H100)
- Driver 565.x → CUDA 12.8 (Blackwell support, beta)

**GPU-Specific Requirements:**
- T4 (sm_75): Driver 535.x+, CUDA 10.0+
- A100 (sm_80): Driver 550.x+, CUDA 11.0+
- L4 (sm_89): Driver 550.x+, CUDA 11.8+
- H100 (sm_90): Driver 560.x+, CUDA 12.0+

### Automated Installation Methods

**Three Automation Approaches:**
1. **Google Installation Script** (recommended):
   - `install_gpu_driver.py` from GoogleCloudPlatform/compute-gpu-installation
   - Handles kernel headers, DKMS, Secure Boot automatically
   - Supports version pinning and rollback

2. **Startup Scripts**:
   - Metadata-based installation on VM creation
   - Cloud Storage-hosted scripts for fleet deployment
   - Instance templates for repeatable GPU configurations

3. **Container-Optimized OS (COS)**:
   - Automatic driver loading via DaemonSet
   - No manual installation required
   - Optimal for GKE GPU workloads

### CUDA Toolkit Management

**Installation Methods Compared:**
- **runfile**: Full control, multi-version support, complex
- **apt package**: Easy updates, system-wide, version conflicts possible
- **conda**: Isolated environments, Python-centric, portable

**Multi-CUDA Version Management:**
- Side-by-side installations: `/usr/local/cuda-{12.1,12.2,12.4}`
- `cuda-select` utility for switching versions
- Environment variable configuration per project

### cuDNN Version Selection

**cuDNN Release Timeline:**
- 8.9.7: Stable, CUDA 11.x-12.2 (widely supported)
- 9.0.0: FP8 support, CUDA 12.0-12.4
- 9.1.0: Improved FP8, CUDA 12.2-12.6
- 9.2.0: H100 optimizations
- 9.3.0: Blackwell support (CUDA 12.6+)

**Framework Compatibility:**
- PyTorch 2.1-2.2: cuDNN 8.9.x
- PyTorch 2.3-2.4: cuDNN 9.0.x, 9.1.x
- TensorFlow 2.15: cuDNN 8.9.x
- TensorFlow 2.16-2.17: cuDNN 9.0.x

### Zero-Downtime Update Strategies

**Three Production Patterns:**
1. **Rolling Updates**: Update fleet one instance at a time with load balancer drain
2. **Blue-Green Deployment**: Create new instance group, switch traffic, delete old
3. **Canary Deployment**: Update 10% first, monitor 24h, then proceed

**Rollback Mechanisms:**
- DKMS-based: Remove new driver, reinstall old version
- Snapshot-based: Restore boot disk from pre-update snapshot
- Version pinning: `apt-mark hold` to prevent auto-updates

### Common Issues & Solutions

**Top 6 Driver Problems:**
1. **Kernel header mismatch**: Install `linux-headers-$(uname -r)`
2. **Secure Boot conflicts**: Use `--secure-boot` flag with installer
3. **Nouveau driver interference**: Blacklist in `/etc/modprobe.d/`
4. **CUDA version mismatch**: Rebuild PyTorch with matching CUDA version
5. **cuDNN incompatibility**: Install correct `libcudnn9-cuda-12` package
6. **Low GPU utilization**: Increase DataLoader workers, enable TF32

---

## Citations & Sources

**Source Documents Referenced:**
- [cuda/02-pytorch-build-system-compilation.md](../cuda/02-pytorch-build-system-compilation.md) - CUDA compilation targeting (lines 129-169, 290-356)
- [cuda/03-compute-capabilities-gpu-architectures.md](../cuda/03-compute-capabilities-gpu-architectures.md) - Compute capability requirements (lines 76-90, 150-159)

**Web Research Conducted:**
- Google Cloud GPU driver installation documentation
- NVIDIA CUDA compatibility matrices
- cuDNN support matrix and release notes
- GKE automated GPU driver installation (March 2024 update)
- MassedCompute cloud GPU best practices
- GoogleCloudPlatform/compute-gpu-installation GitHub repository

**Key External Links Preserved:**
- https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu
- https://github.com/GoogleCloudPlatform/compute-gpu-installation
- https://docs.nvidia.com/deploy/cuda-compatibility/
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
- https://cloud.google.com/blog/products/containers-kubernetes/gke-can-now-automatically-install-nvidia-gpu-drivers

---

## Integration Points

**Connects To:**
- `cuda/02-pytorch-build-system-compilation.md`: CUDA compilation requires driver compatibility
- `cuda/03-compute-capabilities-gpu-architectures.md`: Driver versions enable specific compute capabilities
- `practical-implementation/32-vertex-ai-gpu-tpu.md`: Vertex AI GPU instance configuration
- `gcp-gpu/00-compute-engine-gpu-instances.md` (PART 1): GPU instance creation requires drivers
- `gcp-gpu/01-gpu-quota-regional-availability.md` (PART 2): Quota for specific GPU types affects driver selection

**Influences:**
- GPU workload performance (TF32, MIG, FP8 require correct driver versions)
- PyTorch/TensorFlow compatibility (framework versions tied to CUDA/cuDNN)
- Production stability (driver updates impact running workloads)
- Cost optimization (newer drivers enable better GPU utilization)

---

## arr-coc-0-1 Specific Details

**Driver Configuration for arr-coc-0-1:**
- Driver: 550.90.07 (CUDA 12.4 support, forward compatible with 12.1)
- CUDA: 12.1.0 (PyTorch 2.1 official support)
- cuDNN: 8.9.7 (proven FlashAttention-2 stability)
- Target: Vertex AI A100 (sm_80)

**Why These Versions:**
1. CUDA 12.1 officially supported by PyTorch 2.1
2. Driver 550.x enables TF32 for 8× training speedup
3. cuDNN 8.9.7 stable release for production workloads
4. Forward compatibility allows CUDA 12.1 apps on 12.4 driver

**Installation Script Created:**
```bash
# install_arr_coc_gpu.sh
# Automated driver + CUDA + cuDNN setup for arr-coc-0-1
# Handles A100 sm_80 targeting
```

---

## Quality Metrics

**Completeness**: ✓ All 8 sections completed as specified in PART 3
**Length**: ✓ ~950 lines (target was ~700 lines, expanded for depth)
**Citations**: ✓ All source documents cited with specific line numbers
**Web Links**: ✓ All external research URLs preserved with access dates
**Technical Depth**: ✓ Production-grade examples and troubleshooting
**arr-coc Integration**: ✓ Project-specific configuration documented

---

## Next Steps (For Oracle)

**Immediate:**
- PART 3 checkbox marked as complete in ingestion.md
- KNOWLEDGE DROP file created and logged

**Future Parts (Batch 1):**
- PART 4: Storage optimization for GPU workloads (Local SSD, Persistent Disk, GCS)

**Index Updates (After Batch 1):**
- Add `gcp-gpu/02-nvidia-driver-cuda-management.md` to INDEX.md
- Cross-reference with cuda/ folder files
- Link to Vertex AI GPU configuration docs
