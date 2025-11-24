# GKE GPU Node Pools & Cluster Setup

Comprehensive guide to creating and managing GPU-enabled node pools on Google Kubernetes Engine (GKE) for machine learning workloads.

## Overview

GKE provides native support for NVIDIA GPUs through specialized node pools with GPU-attached machines. This enables running GPU-accelerated ML training, inference, and data processing workloads on Kubernetes.

**Key Capabilities:**
- Multiple GPU types (T4, L4, A100, H100)
- Automatic NVIDIA device plugin installation
- GPU time-sharing for multi-tenancy
- Cluster autoscaling with GPU nodes
- Integration with Cloud Monitoring for GPU metrics

## GPU Node Pool Architecture

### Standard vs Autopilot

**GKE Standard (Recommended for GPUs):**
- Full control over node configuration
- Custom machine types and GPU combinations
- Support for all GPU types
- Manual or automatic driver installation options

**GKE Autopilot (Limited GPU Support):**
- Managed node provisioning
- Limited GPU types (A100, L4, T4)
- No SSH access to nodes
- No DaemonSet support (affects some GPU tools)

**Use Standard mode for production GPU workloads** - provides maximum flexibility and control.

From [NVIDIA GPU Operator with Google GKE](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/google-gke.html) (accessed 2025-11-16):
> For Autopilot Pods, using the GPU Operator is not supported, and you can refer to Deploy GPU workloads in Autopilot.

## Creating GPU Node Pools

### Prerequisites

**1. Enable Required APIs:**
```bash
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
```

**2. Check GPU Quota:**
```bash
# View current GPU quotas
gcloud compute project-info describe --project=PROJECT_ID

# Request quota increase if needed
gcloud compute regions list
```

**Important:** GPU quotas are per-region and per-GPU-type. Request sufficient quota before creating large clusters.

### Method 1: Using gcloud CLI

**Create cluster with GPU node pool:**
```bash
# Create GKE cluster (without GPUs initially)
gcloud container clusters create gpu-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --num-nodes=1 \
    --release-channel=regular

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=gpu-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes=1 \
    --node-labels=gke-no-default-nvidia-gpu-device-plugin=true
```

**GPU-specific flags:**
- `--accelerator type=GPU_TYPE,count=NUM` - GPU type and quantity per node
- `--node-labels=gke-no-default-nvidia-gpu-device-plugin=true` - Disable default device plugin (when using GPU Operator)
- `--metadata disable-legacy-endpoints=true` - Security best practice

### Method 2: Single Command Cluster Creation

From [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/google-gke.html):
```bash
gcloud beta container clusters create demo-cluster \
    --project PROJECT_ID \
    --location us-west1 \
    --release-channel "regular" \
    --machine-type "n1-standard-4" \
    --accelerator "type=nvidia-tesla-t4,count=1" \
    --image-type "UBUNTU_CONTAINERD" \
    --node-labels="gke-no-default-nvidia-gpu-device-plugin=true" \
    --disk-type "pd-standard" \
    --disk-size "1000" \
    --metadata disable-legacy-endpoints=true \
    --max-pods-per-node "110" \
    --num-nodes "1" \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM \
    --enable-ip-alias \
    --tags=nvidia-ingress-all
```

### Available GPU Types

| GPU Type | Use Case | Memory | Performance |
|----------|----------|--------|-------------|
| **nvidia-tesla-t4** | Inference, light training | 16 GB | Entry-level |
| **nvidia-tesla-v100** | Training, HPC | 16 GB | High performance |
| **nvidia-tesla-p100** | Training, HPC | 16 GB | High performance |
| **nvidia-tesla-p4** | Inference | 8 GB | Inference-optimized |
| **nvidia-tesla-a100** | Large-scale training | 40/80 GB | Flagship training |
| **nvidia-a100-80gb** | Large models | 80 GB | Maximum memory |
| **nvidia-l4** | Inference, video | 24 GB | Latest inference GPU |
| **nvidia-h100-80gb** | Next-gen training | 80 GB | Cutting-edge |

**Check availability in your region:**
```bash
gcloud compute accelerator-types list --filter="zone:(us-central1-a)"
```

## NVIDIA Device Plugin Installation

### Automatic Installation (GKE Default)

GKE automatically installs the NVIDIA device plugin DaemonSet on GPU nodes **unless** you disable it.

**Default behavior:**
- Device plugin runs on all GPU nodes
- Exposes `nvidia.com/gpu` resource
- Handles GPU allocation to pods

**To use default device plugin:**
```bash
# Create node pool WITHOUT the disable label
gcloud container node-pools create gpu-pool \
    --cluster=gpu-cluster \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes=1
```

### NVIDIA GPU Operator (Advanced)

For advanced GPU management, use NVIDIA GPU Operator instead of default device plugin.

**Why use GPU Operator:**
- Automated driver lifecycle management
- GPU time-sharing configuration
- Multi-Instance GPU (MIG) support
- Driver upgrades without node recreation

**Installation steps:**

1. **Disable default device plugin:**
```bash
# Already done with node-labels=gke-no-default-nvidia-gpu-device-plugin=true
```

2. **Create namespace:**
```bash
kubectl create ns gpu-operator
```

3. **Apply resource quota:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-operator-quota
  namespace: gpu-operator
spec:
  hard:
    pods: 100
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values:
        - system-node-critical
        - system-cluster-critical
```

4. **Install GPU Operator via Helm:**
```bash
helm install --wait \
    -n gpu-operator \
    nvidia/gpu-operator \
    --version=v24.9.1 \
    --set driver.enabled=false \
    --set cdi.enabled=true \
    --set cdi.default=true
```

From [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/google-gke.html) (accessed 2025-11-16):
> Set the NVIDIA Container Toolkit and driver installation path to /home/kubernetes/bin/nvidia. On GKE node images, this directory is writable and is a stateful location for storing the NVIDIA runtime binaries.

## GPU Resource Requests and Limits

### Pod Configuration

Pods request GPUs using the `nvidia.com/gpu` resource:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
```

**Important rules:**
- GPU must be specified in `limits` (not `requests`)
- GPU count must be an integer (no fractional GPUs without time-sharing)
- `limits` value automatically populates `requests`

### Multiple GPUs per Pod

```yaml
resources:
  limits:
    nvidia.com/gpu: 4  # Request 4 GPUs
```

**Considerations:**
- All GPUs allocated to same node
- Use node affinity if specific GPU topology needed
- Consider NCCL configuration for multi-GPU training

## Node Affinity, Taints, and Tolerations

### Automatic GPU Node Taints

GKE automatically applies taints to GPU nodes:

```yaml
taints:
- key: nvidia.com/gpu
  value: present
  effect: NoSchedule
```

**This prevents non-GPU pods from consuming GPU nodes.**

### GPU Pod Tolerations

GKE automatically adds tolerations to pods requesting GPUs:

```yaml
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

**You don't need to manually add this toleration.**

### Custom Node Affinity

To target specific GPU types:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-specific-pod
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: cloud.google.com/gke-accelerator
            operator: In
            values:
            - nvidia-tesla-a100
            - nvidia-tesla-v100
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
```

**Node labels automatically applied:**
- `cloud.google.com/gke-accelerator=GPU_TYPE`
- `cloud.google.com/gke-accelerator-count=NUM`

## GPU Time-Sharing (Multi-Tenancy)

GPU time-sharing allows multiple pods to share a single physical GPU.

### Configuration via ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  tesla-t4: |
    version: v1
    sharing:
      timeSlicing:
        replicas: 4  # 4 pods can share each T4 GPU
```

### Enable Time-Sharing

```bash
# Apply ConfigMap
kubectl apply -f time-slicing-config.yaml

# Configure GPU Operator for time-sharing
kubectl patch clusterpolicy/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config"}}}}'
```

**Result:**
- Each physical GPU appears as 4 virtual GPUs
- Pods share GPU time via round-robin scheduling
- No memory isolation (all pods see full GPU memory)

From [Google Cloud Blog](https://cloud.google.com/blog/products/containers-kubernetes/gpu-sharing-with-google-kubernetes-engine) (accessed 2025-11-16):
> GPU time-sharing feature on GKE lets multiple containers share a single physical GPU attached to a node, thereby improving its utilization.

### Use Cases for Time-Sharing

**Good for:**
- Development/testing environments
- Inference workloads with low GPU utilization
- Interactive Jupyter notebooks
- Small batch jobs

**Not recommended for:**
- Large-scale training (no performance isolation)
- Latency-sensitive production inference
- Memory-intensive workloads

## Autoscaling GPU Node Pools

### Cluster Autoscaler Configuration

```bash
gcloud container node-pools create gpu-pool \
    --cluster=gpu-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=10 \
    --num-nodes=1
```

**Autoscaler behavior:**
- Scales up when pods are unschedulable due to GPU shortage
- Scales down when nodes are underutilized (default: <50% utilization)
- Respects min/max node limits

### Autoscaling Best Practices

**1. Set appropriate min/max:**
```bash
--min-nodes=1  # Keep 1 node warm for fast pod scheduling
--max-nodes=20 # Prevent runaway costs
```

**2. Request sufficient GPU quota:**
- Quota must cover max_nodes × GPUs_per_node
- Request increases ahead of time (approval can take hours)

**3. Use Preemptible/Spot for cost savings:**
```bash
--spot  # Use Spot VMs (60-91% cheaper)
```

**4. Monitor autoscaling events:**
```bash
kubectl get events --field-selector reason=ScaleUp
kubectl get events --field-selector reason=ScaleDown
```

## Monitoring GPU Utilization

### DCGM Metrics Collection

GKE supports NVIDIA Data Center GPU Manager (DCGM) for detailed GPU metrics.

**Enable DCGM metrics:**
```bash
gcloud container clusters update gpu-cluster \
    --zone=us-central1-a \
    --enable-gpu-metrics
```

**Available metrics in Cloud Monitoring:**
- `nvidia.com/gpu/utilization` - GPU compute utilization (%)
- `nvidia.com/gpu/memory_used` - GPU memory usage (bytes)
- `nvidia.com/gpu/memory_total` - Total GPU memory (bytes)
- `nvidia.com/gpu/temperature` - GPU temperature (°C)
- `nvidia.com/gpu/power_usage` - Power consumption (watts)

From [Collect and view DCGM metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics) (accessed 2025-11-16):
> You can monitor GPU utilization, performance, and health by configuring GKE to send NVIDIA Data Center GPU Manager (DCGM) metrics to Cloud Monitoring.

### Viewing GPU Metrics

**In Cloud Console:**
1. Navigate to Monitoring → Metrics Explorer
2. Resource type: Kubernetes Container
3. Metric: NVIDIA GPU Utilization

**Using gcloud:**
```bash
gcloud monitoring time-series list \
    --filter='metric.type="kubernetes.io/container/accelerator/nvidia.com/gpu/utilization"'
```

### nvidia-smi on Nodes

For debugging, SSH to node and run:
```bash
# Get node name
kubectl get nodes -l cloud.google.com/gke-accelerator

# SSH to node (GKE Standard only)
gcloud compute ssh NODE_NAME --zone=ZONE

# Check GPU status
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

## arr-coc-0-1 GKE GPU Configuration

### Training Cluster Setup

**Requirements:**
- 8× A100 GPUs for multi-GPU training
- NVLink/NVSwitch for fast GPU interconnect
- High-bandwidth persistent disk for checkpoints

**Node pool configuration:**
```bash
gcloud container node-pools create arr-coc-training \
    --cluster=arr-coc-cluster \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-8g \
    --accelerator type=nvidia-tesla-a100,count=8 \
    --num-nodes=1 \
    --disk-type=pd-ssd \
    --disk-size=500 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=4 \
    --node-labels=workload=training,gke-no-default-nvidia-gpu-device-plugin=true
```

**Machine type a2-highgpu-8g includes:**
- 8× A100 40GB GPUs
- 96 vCPUs
- 680 GB RAM
- 600 GB/s NVSwitch interconnect

### Inference Cluster Setup

**Requirements:**
- L4 GPUs for cost-effective inference
- Autoscaling for variable load
- GPU time-sharing for dev/test

**Node pool configuration:**
```bash
gcloud container node-pools create arr-coc-inference \
    --cluster=arr-coc-cluster \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator type=nvidia-l4,count=1 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --spot \
    --node-labels=workload=inference
```

**Machine type g2-standard-4 includes:**
- 1× L4 GPU (24 GB)
- 4 vCPUs
- 16 GB RAM
- Optimized for inference and video workloads

### Pod Deployment Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: arr-coc-training
  labels:
    app: arr-coc
    workload: training
spec:
  nodeSelector:
    workload: training
  containers:
  - name: trainer
    image: gcr.io/PROJECT_ID/arr-coc-trainer:latest
    command: ["python", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 8
    volumeMounts:
    - name: checkpoint-storage
      mountPath: /checkpoints
  volumes:
  - name: checkpoint-storage
    persistentVolumeClaim:
      claimName: arr-coc-checkpoints
```

## Troubleshooting

### Pods Stuck in Pending

**Check GPU availability:**
```bash
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Common issues:**
1. Insufficient GPU quota
2. No nodes with requested GPU type
3. GPU already allocated to other pods
4. Missing tolerations (automatic in GKE)

### GPU Not Detected in Pod

**Verify device plugin:**
```bash
kubectl get pods -n kube-system | grep nvidia-gpu-device-plugin
```

**Check node labels:**
```bash
kubectl get nodes --show-labels | grep accelerator
```

**Test GPU access:**
```bash
kubectl run gpu-test --rm -it \
    --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
    --restart=Never \
    --limits=nvidia.com/gpu=1 \
    -- nvidia-smi
```

### Autoscaler Not Scaling

**Check autoscaler status:**
```bash
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml
```

**Common issues:**
1. Max nodes reached
2. Insufficient GPU quota
3. Node pool not enabled for autoscaling
4. Scale-down disabled

### Performance Issues

**Check GPU utilization:**
```bash
kubectl exec POD_NAME -- nvidia-smi
```

**Verify NVLink connectivity (multi-GPU):**
```bash
kubectl exec POD_NAME -- nvidia-smi topo -m
```

**Monitor NCCL bandwidth:**
```bash
kubectl logs POD_NAME | grep "NCCL"
```

## Sources

**Official Documentation:**
- [Run GPUs in GKE Standard node pools](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) - Google Cloud Documentation (accessed 2025-11-16)
- [NVIDIA GPU Operator with Google GKE](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/google-gke.html) - NVIDIA Documentation (accessed 2025-11-16)
- [Collect and view DCGM metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics) - Google Cloud Documentation (accessed 2025-11-16)

**Web Research:**
- [GPU sharing with Google Kubernetes Engine](https://cloud.google.com/blog/products/containers-kubernetes/gpu-sharing-with-google-kubernetes-engine) - Google Cloud Blog (accessed 2025-11-16)
- [How to Set Up a GPU-Enabled Kubernetes Cluster on GKE](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) - vCluster Blog (accessed 2025-11-16)
- [Cluster multi-tenancy](https://cloud.google.com/kubernetes-engine/docs/concepts/multitenancy-overview) - Google Cloud Documentation (accessed 2025-11-16)

**Community Resources:**
- [Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) - NVIDIA GPU Operator Documentation (accessed 2025-11-16)
- [Monitoring GPU workloads on GKE with NVIDIA DCGM](https://cloud.google.com/blog/products/containers-kubernetes/monitoring-gpu-workloads-on-gke-with-nvidia-data-center-gpu-manager) - Google Cloud Blog (accessed 2025-11-16)

**Related Skills:**
- See orchestration/00-kubernetes-gpu-scheduling.md (to be created) for K8s GPU scheduling patterns
- See orchestration/03-ml-workload-patterns-k8s.md (to be created) for ML-specific workload patterns
- See vertex-ai-production/02-ray-gke-integration.md (to be created) for Ray on GKE GPU clusters
