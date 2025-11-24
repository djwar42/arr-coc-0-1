# KNOWLEDGE DROP: Production Deployment Troubleshooting (PART 4)

**Runner**: PART 4 - Production Deployment Troubleshooting (Containers, Drivers, Multi-Tenant)
**Timestamp**: 2025-11-13
**Status**: ✅ COMPLETE

## Knowledge File Created

**File**: `cuda/15-production-deployment-troubleshooting-expert.md` (~520 lines)

## Content Summary

### Section 1: NVIDIA Container Toolkit & Docker Runtime Issues
- Container stack architecture (application → CUDA → libnvidia-container → docker → driver)
- Common errors: "Failed to initialize NVML", "nvidia-container-cli: initialization error", GPU not visible
- Driver/container version compatibility matrix (forward compatibility rules)
- SELinux permission debugging and policy generation
- Debug logging configuration (`/etc/nvidia-container-runtime/config.toml`)

### Section 2: GPU Device Access & System-Level Issues
- Device node permissions (`/dev/nvidia*`, `/dev/nvidia-uvm*`)
- Cgroup device access verification
- Missing `/dev/char` symlinks (runc compatibility issue)
- Driver module loading problems (nouveau conflicts)
- Host-container driver mismatch debugging with `nvidia-container-cli info`

### Section 3: Multi-Tenant GPU Isolation (MPS & MIG)
- MPS (Multi-Process Service) fundamentals and per-GPU daemon configuration
- **Critical finding**: MPS does NOT automatically distribute across GPUs in multi-GPU systems
- MPS thread percentage limits (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
- Volta+ MPS error isolation features
- MIG (Multi-Instance GPU) configuration: enable mode, create instances, profile selection
- MIG vs MPS decision tree (isolation vs utilization tradeoffs)

### Section 4: Kubernetes GPU Scheduling & Production Operations
- NVIDIA device plugin installation and verification
- GPU resource requests in pod specs (`nvidia.com/gpu`)
- GPU time-slicing configuration (oversubscribe GPUs 4×)
- Node labeling with GPU types (Node Feature Discovery integration)
- Production monitoring: liveness probes, health checks
- Emergency recovery runbook: drain node, GPU reset, driver reload

## Sources Used

**Official Documentation:**
- NVIDIA Container Toolkit Troubleshooting Guide
- NVIDIA Multi-Instance GPU User Guide
- Kubernetes GPU Scheduling Documentation

**Community Resources:**
- NVIDIA Developer Forums (MPS multi-GPU limitations)
- Medium article: "GPU Sharing That Works: MIG, MPS & Schedulers"
- Jacob Tomlinson blog: Oversubscribing GPUs in Kubernetes

**Technical Discussions:**
- GitHub NVIDIA Container Toolkit Discussion #1133 (/dev/char symlinks)
- NVIDIA Forums: CUDA 11.4.2 docker image driver version mismatch

## Knowledge Gaps Filled

1. **Container Runtime Integration**: Deep dive into libnvidia-container and nvidia-container-runtime-hook
2. **MPS Multi-GPU Limitation**: Documented that MPS requires per-GPU daemon for multi-GPU distribution (not automatic)
3. **MIG vs MPS Selection**: Clear decision tree based on isolation requirements vs utilization goals
4. **Kubernetes GPU Operations**: Time-slicing for oversubscription, production runbooks
5. **SELinux Debugging**: Policy generation for nvidia-docker wrapper permission issues

## Expert-Level Insights

- CDI (Container Device Interface) recommended over legacy `--gpus` flag to prevent GPU loss on container updates
- Volta+ GPUs have MPS client isolation; pre-Volta do not (crash affects all clients)
- Kubernetes time-slicing allows 4× oversubscription but no memory isolation (noisy neighbor risks)
- Driver forward compatibility: Host driver ≥ Required driver for CUDA version
- Production recovery: Always drain node before GPU reset to prevent workload interruption

## Integration with Existing Knowledge

- Complements `cuda/09-runtime-errors-debugging-expert.md` (adds production/container layer)
- Extends `cuda/11-advanced-troubleshooting-multi-gpu-expert.md` (adds orchestration complexity)
- References `cuda/02-pytorch-build-system-compilation.md` (build-time vs runtime driver versions)

## File Statistics

- **Lines**: 520
- **Sections**: 4 (Container Toolkit, Device Access, Multi-Tenant, Kubernetes)
- **Code Examples**: 40+
- **Citations**: 9 sources (accessed 2025-11-13)
- **Depth**: Ultra-expert (production-grade troubleshooting)
