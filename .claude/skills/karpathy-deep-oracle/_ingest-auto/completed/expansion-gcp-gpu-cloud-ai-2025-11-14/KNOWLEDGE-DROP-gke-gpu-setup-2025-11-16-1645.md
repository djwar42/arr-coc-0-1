# KNOWLEDGE DROP: GKE GPU Node Pools & Cluster Setup

**Created**: 2025-11-16 16:45
**Part**: PART 9
**File**: gke-gpu/00-gke-gpu-clusters-setup.md
**Lines**: ~700

## What Was Created

Comprehensive guide to setting up and managing GPU-enabled node pools on Google Kubernetes Engine, covering:

### Core Topics
1. **GPU Node Pool Architecture** - Standard vs Autopilot modes for GPU workloads
2. **Creating GPU Node Pools** - gcloud CLI methods and configuration
3. **NVIDIA Device Plugin** - Automatic installation vs GPU Operator
4. **Resource Requests/Limits** - Proper nvidia.com/gpu configuration
5. **Node Affinity & Taints** - Automatic GPU node isolation
6. **GPU Time-Sharing** - Multi-tenancy configuration
7. **Autoscaling** - Cluster autoscaler with GPU nodes
8. **Monitoring** - DCGM metrics and nvidia-smi
9. **arr-coc-0-1 Config** - Training and inference cluster setups

## Key Insights

**GKE Standard is Essential:**
- Autopilot has limited GPU support (no GPU Operator, no SSH)
- Standard mode provides full control over GPU configuration
- Required for production ML workloads

**NVIDIA GPU Operator:**
- Advanced alternative to default device plugin
- Provides driver lifecycle management
- Enables GPU time-sharing and MIG support
- Must disable default plugin with node label

**GPU Time-Sharing:**
- Allows multiple pods per GPU (e.g., 4 replicas)
- No memory isolation - all pods see full GPU memory
- Good for dev/test, not production training

**Autoscaling Challenges:**
- GPU quota must cover max_nodes × GPUs_per_node
- Scale-up can be slow (node provisioning + driver install)
- Spot/Preemptible saves 60-91% but adds complexity

## Web Research Sources

**Official Documentation:**
- Google Cloud: GPU node pools, DCGM metrics, autoscaling
- NVIDIA: GPU Operator installation, time-slicing configuration
- Kubernetes: GPU scheduling and resource management

**Key URLs:**
- https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
- https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/google-gke.html
- https://cloud.google.com/blog/products/containers-kubernetes/gpu-sharing-with-google-kubernetes-engine

## Integration Points

**Connects to Future Files:**
- orchestration/00-kubernetes-gpu-scheduling.md - K8s GPU patterns
- orchestration/03-ml-workload-patterns-k8s.md - ML workload orchestration
- vertex-ai-production/02-ray-gke-integration.md - Ray on GKE GPUs

**Builds on Existing:**
- gcp-gpu/00-compute-engine-gpu-instances.md - Compute Engine GPU basics
- gcp-gpu/01-gpu-quotas-management.md - GPU quota planning
- gcp-gpu/02-nvidia-driver-cuda-management.md - Driver installation

## arr-coc-0-1 Specific

**Training Configuration:**
- Machine type: a2-highgpu-8g (8× A100 40GB)
- NVSwitch 600 GB/s interconnect
- Autoscaling 0-4 nodes for cost control
- PD-SSD for fast checkpoint storage

**Inference Configuration:**
- Machine type: g2-standard-4 (1× L4 24GB)
- Spot VMs for cost savings
- Autoscaling 1-10 nodes for variable load
- GPU time-sharing for dev/test environments

## Quality Check

- ✅ **700+ lines** of detailed content
- ✅ **All sources cited** with URLs and access dates
- ✅ **Code examples** for all major operations
- ✅ **Troubleshooting section** with common issues
- ✅ **arr-coc-0-1 configurations** for both training and inference
- ✅ **Integration notes** connecting to other knowledge files
- ✅ **Best practices** from Google Cloud and NVIDIA docs

## Notes

- GKE GPU setup is more opinionated than raw Compute Engine
- Time-sharing is unique to GKE (not available on bare Compute Engine)
- DCGM metrics integration is GKE-managed (easier than self-hosted)
- Autoscaling with GPUs requires careful quota planning

PART 9 COMPLETE ✓
