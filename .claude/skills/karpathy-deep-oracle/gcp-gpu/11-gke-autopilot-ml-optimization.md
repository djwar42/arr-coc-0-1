# GKE Autopilot for ML Workloads

## Overview

GKE Autopilot is Google's fully managed Kubernetes mode that eliminates node management overhead while providing GPU support for ML workloads. Unlike GKE Standard mode (where you configure and manage node pools), Autopilot automatically provisions and scales compute resources based on pod requirements, making it ideal for teams prioritizing simplicity and operational efficiency over granular infrastructure control.

**Key distinction**: Autopilot is **pod-centric** (you request resources per pod), while Standard is **node-centric** (you manage VM instances and node pools).

From [How to Set Up a GPU-Enabled Kubernetes Cluster on GKE](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) (accessed 2025-11-16):
> "Autopilot Mode: For a hands-off approach, GKE Autopilot allows GPU workload deployment without the need for infrastructure management. However, you will incur a flat fee of $0.10 per hour for each cluster, in addition to the control plane cost."

## GPU Support in Autopilot

### Available GPU Types

GKE Autopilot supports the following NVIDIA GPUs for ML workloads (as of 2024-2025):

**Production GPUs:**
- **NVIDIA A100 80GB** - High-performance training for large models
- **NVIDIA L4** - Cost-effective inference and fine-tuning
- **NVIDIA T4** - Entry-level ML workloads and inference
- **NVIDIA H100** - Cutting-edge training for frontier models (newer regions)
- **NVIDIA H200** - Enhanced memory bandwidth for large-scale training
- **NVIDIA B200/GB200** - Next-generation Blackwell architecture (select regions)

From [Deploy GPU workloads in Autopilot | GKE AI/ML](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus) (Google Cloud documentation, 2024-2025):
> "Request and deploy GPU workloads on GKE Autopilot using different GPU quantities and types, such as B200, H200, H100, and A100."

### GPU Request Syntax

In Autopilot, you request GPUs by specifying resource limits in your pod specification:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-pod
spec:
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
        memory: "16Gi"
        cpu: "4"
```

**Critical**: In Autopilot, GPU requests must be specified in **limits** (not requests). Autopilot automatically sets requests equal to limits for GPU resources.

### NVIDIA Device Plugin

Autopilot **automatically installs** the NVIDIA device plugin (no manual DaemonSet required). The plugin:
- Exposes `nvidia.com/gpu` as a schedulable resource
- Manages GPU allocation to pods
- Handles driver installation and updates
- Monitors GPU health

From Reddit discussion [What are the downsides of using GKE Autopilot](https://www.reddit.com/r/kubernetes/comments/1mpzlyj/what_are_the_downsides_of_using_gke_autopilot/) (accessed 2025-11-16):
> "GKE manages driver installation, device plugin deployment, and driver updates during node auto-upgrades."

## Autopilot vs Standard Mode Comparison

### When to Use Autopilot for ML

**Autopilot is ideal for:**
- **Rapid prototyping**: Get GPU clusters running in minutes without node configuration
- **Development/experimentation**: Focus on model development, not infrastructure
- **Inference serving**: Production inference endpoints with auto-scaling
- **Periodic training jobs**: Batch training jobs with variable resource needs
- **Cost-conscious workloads**: Pay only for pod-level resources used
- **Small to medium teams**: Reduce operational burden when infrastructure expertise is limited

From [GKE Autopilot vs. Standard Mode: Which One Should You Choose](https://medium.com/@selvamraju007/gke-autopilot-vs-standard-mode-which-one-should-you-choose-390456bba9d2) (Medium, accessed 2025-11-16):
> "GKE Autopilot is ideal for teams that prioritize simplicity and cost efficiency, while GKE Standard is better for customizable, high-performance workloads."

### When to Use Standard Mode for ML

**Standard mode is better for:**
- **Multi-node distributed training**: 8+ GPUs across multiple nodes with custom networking
- **Fine-grained control**: Specific machine types, network configurations, or storage attachments
- **GPU node pools with DaemonSets**: Custom monitoring, logging, or security agents
- **Persistent GPU workloads**: Long-running training jobs requiring dedicated node pools
- **High-performance networking**: Compact Placement Policy, GPUDirect-TCPX for multi-node
- **Cost optimization at scale**: Sustained Use Discounts, Committed Use Discounts on specific machine types

From [Compare features in Autopilot and Standard clusters](https://docs.cloud.google.com/kubernetes-engine/docs/resources/autopilot-standard-feature-comparison) (Google Cloud documentation):
> "Standard mode gives you fine grained control over machine configuration and resources allocation, but in Autopilot we are not managing nodes directly."

### Feature Comparison Table

| Feature | Autopilot | Standard |
|---------|-----------|----------|
| **Node management** | Automatic | Manual |
| **GPU types** | A100, L4, T4, H100, H200, B200 | All GPU types + custom configs |
| **Pricing model** | Per-pod (CPU, memory, GPU) + $0.10/hr cluster fee | Per-node (VM instances) + $0.10/hr cluster fee |
| **Autoscaling** | Automatic (pod-driven) | Manual (node pool autoscaling) |
| **DaemonSets** | Limited (managed by GKE) | Fully supported |
| **Node SSH access** | Not available | Available |
| **Custom machine types** | Not supported | Fully supported |
| **Spot/Preemptible** | Spot pods supported | Spot node pools supported |
| **Multi-node training** | Possible but not optimized | Optimized (Compact Placement, custom networking) |
| **Setup time** | Minutes | 10-30 minutes |
| **Operational overhead** | Very low | Medium to high |

## Resource Requests and Scheduling

### Pod Resource Configuration

Autopilot enforces specific requirements for GPU pod resource requests:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training
spec:
  containers:
  - name: pytorch-trainer
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 1        # GPU count (1, 2, 4, or 8)
        cpu: "8"                  # CPU cores
        memory: "32Gi"            # Memory
        ephemeral-storage: "10Gi" # Local disk
      # Requests automatically set to match limits in Autopilot
```

**Key rules:**
- **GPU counts**: Must be 1, 2, 4, or 8 (powers of 2)
- **CPU/memory ratios**: Autopilot validates reasonable CPU-to-memory ratios
- **Ephemeral storage**: Default is 10 GiB, can request up to 100 GiB
- **Requests = Limits**: Autopilot automatically sets requests equal to limits for all resources

### Automatic Node Provisioning

When you deploy a GPU pod in Autopilot:

1. **Pod submitted** → Autopilot evaluates resource requirements
2. **Node selection** → GKE automatically selects appropriate machine type (e.g., `n1-standard-8` with T4 GPU)
3. **Node creation** → New node provisioned in ~2-5 minutes
4. **GPU attachment** → NVIDIA drivers installed, device plugin configured
5. **Pod scheduled** → Pod scheduled and containers started
6. **Auto-scaling** → Node scales down after pod completes (idle timeout)

**No manual node pool configuration required**.

### GPU Topology and Multi-GPU

For multi-GPU pods (2, 4, or 8 GPUs), Autopilot automatically provisions nodes with:

**Single-node multi-GPU configurations:**
- **2 GPUs**: `n1-standard-16` with 2× T4 or 2× L4
- **4 GPUs**: `n1-standard-32` with 4× T4 or A100 nodes
- **8 GPUs**: `a2-highgpu-8g` (8× A100 80GB) or `a3-highgpu-8g` (8× H100)

**NVLink support**: A100 and H100 GPU nodes include NVLink for high-bandwidth GPU-to-GPU communication (~600 GB/s on A100, ~900 GB/s on H100).

From [vCluster blog](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) (accessed 2025-11-16):
> "GKE offers a range of NVIDIA GPUs (including H100, A100, L4, B200, GB200 and T4 GPUs) that can be attached to nodes in your clusters, enabling efficient handling of compute-intensive tasks."

## Limitations and Constraints

### Autopilot-Specific Limitations

**1. No DaemonSets (with exceptions)**

Autopilot restricts DaemonSets to prevent node-level customizations that conflict with managed infrastructure. You **cannot** run:
- Custom monitoring agents (e.g., Prometheus node exporter)
- Custom logging collectors (e.g., Fluentd, Filebeat)
- Security scanning DaemonSets (e.g., Falco, Twistlock)
- GPU-specific DaemonSets (e.g., custom DCGM exporters)

**Exception**: GKE-managed DaemonSets (NVIDIA device plugin, GKE logging/monitoring) are automatically deployed.

From Reddit discussion (accessed 2025-11-16):
> "Incompatibility with certain Kubernetes features (e.g., no DaemonSets). Are these limitations still true in 2025? Yes - Autopilot restricts DaemonSets to maintain managed infrastructure consistency."

**2. No Node SSH Access**

You cannot SSH into Autopilot nodes for debugging or custom configuration. This limits:
- Direct GPU driver inspection (`nvidia-smi` on node)
- Custom CUDA toolkit installation
- Manual debugging of node-level issues

**Workaround**: Use `kubectl exec` into pods for container-level debugging, or use GKE logging/monitoring for node metrics.

**3. No Custom Machine Types**

Autopilot automatically selects machine types based on pod resource requests. You cannot:
- Specify exact machine types (e.g., `n1-standard-8` vs `n2-standard-8`)
- Request specific CPU architectures (e.g., Intel vs AMD)
- Configure custom local SSD counts or sizes

**4. Limited Multi-Node Distributed Training**

While Autopilot supports multi-GPU pods (up to 8 GPUs per pod), **multi-node distributed training** (e.g., 16+ GPUs across 2+ nodes) is **possible but not optimized**:

- **No Compact Placement Policy**: Nodes may be spread across zones, increasing latency
- **No GPUDirect-TCPX**: Standard TCP/IP networking (no RDMA-like optimizations)
- **No custom NCCL tuning**: Limited control over inter-node communication settings

For **large-scale distributed training** (128+ GPUs), **Standard mode** is recommended.

**5. No Persistent GPU Reservations**

Autopilot nodes scale down when pods complete. You cannot:
- Reserve dedicated GPU nodes for long-running experiments
- Maintain "warm" GPU nodes for fast job startup
- Pin specific workloads to specific GPU nodes

### Resource Quotas and Limits

Autopilot enforces stricter resource limits than Standard mode:

**Per-pod limits:**
- **Max CPUs**: 110 CPUs per pod
- **Max memory**: 416 GiB per pod
- **Max GPUs**: 8 GPUs per pod
- **Max ephemeral storage**: 100 GiB per pod

**Per-namespace limits:**
- Autopilot enforces resource quotas per namespace to prevent resource exhaustion
- Default quota: 100 pods per namespace (can be increased)

**GPU quotas**: Same regional GPU quotas apply as Standard mode (must request quota increases via IAM & Admin).

## Cost Optimization with Autopilot

### Pricing Model

**Autopilot pricing** = Cluster management fee + Pod-level resources

**1. Cluster management fee**: $0.10/hour per cluster (~$72/month)
   - Same as Standard mode
   - Includes control plane and managed components

**2. Pod-level resource pricing** (pay only for what you request):

| Resource | Price (on-demand) | Price (Spot pods) |
|----------|------------------|-------------------|
| **vCPU** | ~$0.033/hour | ~$0.010/hour (70% savings) |
| **Memory (GB)** | ~$0.0037/hour | ~$0.0011/hour (70% savings) |
| **NVIDIA T4 GPU** | ~$0.325/hour | ~$0.098/hour (70% savings) |
| **NVIDIA L4 GPU** | ~$0.495/hour | ~$0.149/hour (70% savings) |
| **NVIDIA A100 80GB GPU** | ~$2.48/hour | ~$0.744/hour (70% savings) |

From [GKE Pricing Guide: Autopilot vs. Standard (2025)](https://www.devzero.io/blog/gke-pricing) (DevZero, accessed 2025-11-16):
> "Autopilot Mode: You're billed for the CPU, memory, and ephemeral storage that your pods request. Standard Mode: You manage and pay for the underlying VM instances."

### Spot Pods for Cost Savings

Autopilot supports **Spot pods** (equivalent to Preemptible instances in Standard mode):

**Enable Spot pods:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: spot-training
spec:
  nodeSelector:
    cloud.google.com/gke-spot: "true"  # Request Spot pod
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: trainer
    image: pytorch-training:latest
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "8"
```

**Spot pod characteristics:**
- **60-91% cost savings** vs on-demand pods
- **Preemption**: Can be terminated at any time with 30-second warning
- **Availability**: Subject to capacity availability (may not always be available)
- **Best for**: Fault-tolerant workloads with checkpoint-resume capability

**Spot pod best practices:**
- Implement checkpoint-resume in training scripts (save every N steps)
- Use persistent storage (GCS, Persistent Disk) for checkpoints
- Monitor preemption metrics via Cloud Monitoring
- Hybrid approach: Critical jobs on-demand, experimental jobs on Spot

From Reddit discussion (accessed 2025-11-16):
> "Spot pods in Autopilot work similarly to Spot instances in Standard mode - you get significant cost savings but need to handle preemptions gracefully."

### Cost Comparison: Autopilot vs Standard

**Example: 8× A100 80GB training job for 100 hours**

**Autopilot (on-demand):**
- Cluster management: $0.10/hour × 100 hours = $10
- 8× A100 GPUs: $2.48/hour/GPU × 8 × 100 = $1,984
- CPU (64 cores): $0.033/hour × 64 × 100 = $211
- Memory (256 GB): $0.0037/hour × 256 × 100 = $95
- **Total**: ~$2,300

**Standard mode (a2-highgpu-8g on-demand):**
- Cluster management: $0.10/hour × 100 hours = $10
- a2-highgpu-8g instance: ~$23/hour × 100 = $2,300
- **Total**: ~$2,310

**Verdict**: For **single-pod GPU workloads**, Autopilot and Standard have **similar costs**. Autopilot advantage: no idle node costs (pay only when pod is running).

**Autopilot (Spot pods):**
- 70% savings on all resources
- **Total**: ~$700 (vs $2,300 on-demand)

### When Autopilot Saves Money

**Autopilot is cheaper when:**
- **Intermittent workloads**: Training jobs run sporadically (not 24/7)
- **Variable resource needs**: Pod resource requirements change frequently
- **Spot pods**: Workloads are preemption-tolerant
- **No idle nodes**: Standard mode nodes sit idle between jobs

**Standard is cheaper when:**
- **Sustained workloads**: GPUs run continuously (Sustained Use Discounts apply)
- **Committed Use Discounts**: 1-year or 3-year commitments (57% savings)
- **Reserved capacity**: Dedicated GPU node pools for predictable workloads

## Production Deployment Patterns

### Single-GPU Inference Serving

**Use case**: Deploy ML model for inference with autoscaling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vlm-inference
spec:
  replicas: 2  # Start with 2 replicas
  selector:
    matchLabels:
      app: vlm-inference
  template:
    metadata:
      labels:
        app: vlm-inference
    spec:
      containers:
      - name: triton-server
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vlm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vlm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Autopilot benefits:**
- Automatic node provisioning as HPA scales up
- Pay only for running pods (no idle GPU nodes)
- GKE manages driver updates and node health

### Batch Training Jobs (PyTorchJob)

**Use case**: Run distributed training job with Kubeflow Training Operator

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
            command:
            - python
            - train.py
            - --backend=nccl
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "32Gi"
                cpu: "8"
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
            command:
            - python
            - train.py
            - --backend=nccl
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "32Gi"
                cpu: "8"
```

**Autopilot behavior:**
- Creates 4 GPU pods (1 master + 3 workers)
- Automatically provisions 4 GPU nodes (or fewer if pods can colocate)
- Nodes scale down after job completes
- Supports NCCL for multi-GPU communication

**Limitation**: No Compact Placement Policy, so inter-node latency may be higher than Standard mode.

### Spot Pods for Cost-Effective Training

**Use case**: Fault-tolerant training with checkpoint-resume

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: checkpoint-training
spec:
  nodeSelector:
    cloud.google.com/gke-spot: "true"
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: trainer
    image: pytorch-checkpoint:latest
    env:
    - name: CHECKPOINT_DIR
      value: gs://my-bucket/checkpoints  # GCS for persistence
    - name: CHECKPOINT_INTERVAL
      value: "100"  # Save every 100 steps
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
        cpu: "16"
  restartPolicy: OnFailure  # Auto-restart on preemption
```

**Best practices:**
- Store checkpoints in Cloud Storage (survives preemption)
- Implement graceful shutdown handler (SIGTERM signal)
- Monitor preemption events via Cloud Logging
- Use `restartPolicy: OnFailure` for automatic retry

## arr-coc-0-1 Autopilot Feasibility Analysis

### Workload Profile

The **arr-coc-0-1** project (Adaptive Relevance Realization - Contexts Optical Compression - Vision) has the following GPU requirements:

**Training characteristics:**
- **Single-GPU development**: T4/L4 for prototyping and debugging
- **Multi-GPU fine-tuning**: 4-8× A100 80GB for full model training
- **Batch sizes**: Variable (64-400 tokens per patch, K=200 patches)
- **Training duration**: 10-50 hours per experiment
- **Checkpoint frequency**: Every 500-1000 steps

**Inference characteristics:**
- **Single-GPU inference**: L4 or T4 for production serving
- **Latency requirements**: <100ms p99 for real-time vision tasks
- **Throughput**: 10-100 requests/second
- **Autoscaling**: Dynamic scaling based on traffic

### Recommended Deployment Strategy

**Development/experimentation**: ✅ **Use Autopilot**
- Rapid iteration with single-GPU pods (T4/L4)
- Cost-effective Spot pods for non-critical experiments
- No node management overhead
- Fast cluster setup (minutes vs hours)

```yaml
# Development pod
apiVersion: v1
kind: Pod
metadata:
  name: arr-coc-dev
spec:
  nodeSelector:
    cloud.google.com/gke-spot: "true"  # Cost savings
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: trainer
    image: gcr.io/arr-coc/training:latest
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
        cpu: "8"
```

**Production training**: ⚠️ **Consider Standard mode**
- Multi-node distributed training (8+ GPUs) benefits from Compact Placement Policy
- Long-running training jobs (>24 hours) benefit from dedicated node pools
- Fine-grained control over networking and storage configuration
- Committed Use Discounts for cost savings on sustained workloads

**Production inference**: ✅ **Use Autopilot**
- Single-GPU inference pods with HPA for autoscaling
- No idle GPU costs (nodes scale down during low traffic)
- Automatic driver updates and node management
- Production-ready with minimal operational overhead

```yaml
# Production inference deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arr-coc-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: arr-coc-inference
  template:
    metadata:
      labels:
        app: arr-coc-inference
    spec:
      containers:
      - name: triton
        image: gcr.io/arr-coc/inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: arr-coc-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: arr-coc-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Hybrid Approach (Recommended)

**Best of both worlds**:
1. **Autopilot cluster** for development, experimentation, and inference
2. **Standard cluster** for production training (multi-node distributed workloads)
3. **Shared infrastructure**: Cloud Storage for datasets, Artifact Registry for images, Cloud Monitoring for metrics

**Cost breakdown (estimated monthly):**
- **Autopilot cluster** (inference + dev): $500-1,000/month (variable, based on traffic)
- **Standard cluster** (training node pool): $2,000-5,000/month (8× A100 80GB, 50% duty cycle)
- **Total**: $2,500-6,000/month

**Comparison**: Single Standard cluster running 24/7: $10,000-15,000/month

**Savings**: 50-60% cost reduction with hybrid approach

## Monitoring and Observability

### Cloud Monitoring GPU Metrics

Autopilot automatically exports GPU metrics to Cloud Monitoring:

**Available metrics:**
- `container.googleapis.com/accelerator/duty_cycle` - GPU utilization (%)
- `container.googleapis.com/accelerator/memory_used` - GPU memory used (bytes)
- `container.googleapis.com/accelerator/memory_total` - GPU memory total (bytes)

**Create GPU utilization alert:**
```yaml
# Example alert policy (via gcloud)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Low GPU Utilization Alert" \
  --condition-display-name="GPU Utilization < 30%" \
  --condition-threshold-value=0.3 \
  --condition-threshold-duration=600s \
  --condition-filter='resource.type="k8s_container" AND metric.type="container.googleapis.com/accelerator/duty_cycle"'
```

**Dashboard queries:**
```promql
# Average GPU utilization across all pods
avg(container_accelerator_duty_cycle{resource_type="k8s_container"})

# GPU memory usage by pod
sum(container_accelerator_memory_used{resource_type="k8s_container"}) by (pod_name)
```

### Logging and Debugging

**View pod logs:**
```bash
# Standard kubectl logs
kubectl logs gpu-training-pod

# Cloud Logging (more powerful filtering)
gcloud logging read "resource.type=k8s_container AND resource.labels.pod_name=gpu-training-pod" --limit 100
```

**Debug GPU allocation issues:**
```bash
# Check pod events
kubectl describe pod gpu-training-pod

# Check node events
kubectl get events --field-selector involvedObject.kind=Node

# View GPU resource capacity
kubectl get nodes -o json | jq '.items[].status.capacity."nvidia.com/gpu"'
```

**Common issues:**
- **Pod pending**: GPU quota exhausted or no GPU capacity available
- **Pod failed**: OOM (GPU memory exhausted), check `nvidia.com/gpu` limits
- **Slow scheduling**: Autopilot provisioning new GPU node (2-5 minutes)

From GKE documentation:
> "Autopilot automatically applies a toleration so only Pods requesting GPUs are scheduled on GPU nodes. This enables more efficient autoscaling as your GPU nodes can scale down when not needed."

## Migration Path: Standard to Autopilot

### When to Migrate

**Good migration candidates:**
- Development/staging clusters with intermittent GPU usage
- Inference deployments with variable traffic
- Batch training jobs with checkpoint-resume capability
- Single-GPU or small multi-GPU workloads (<8 GPUs per job)

**Not recommended for migration:**
- Multi-node distributed training clusters (16+ GPUs)
- Workloads requiring custom DaemonSets
- Clusters with specific node-level customizations
- Workloads requiring node SSH access for debugging

### Migration Steps

**1. Assess workload compatibility:**
```bash
# Identify DaemonSets (not supported in Autopilot)
kubectl get daemonsets --all-namespaces

# Check node selectors and taints
kubectl get pods --all-namespaces -o json | jq '.items[].spec.nodeSelector'

# Review resource requests/limits
kubectl get pods --all-namespaces -o json | jq '.items[].spec.containers[].resources'
```

**2. Create Autopilot cluster:**
```bash
gcloud container clusters create-auto gpu-autopilot \
  --region=us-central1 \
  --release-channel=regular
```

**3. Migrate workloads:**
```bash
# Export deployments from Standard cluster
kubectl get deployments --all-namespaces -o yaml > deployments.yaml

# Modify resource requests (ensure limits are set)
# Remove node selectors incompatible with Autopilot
# Apply to Autopilot cluster
kubectl apply -f deployments-autopilot.yaml --context=autopilot-cluster
```

**4. Validate and cutover:**
```bash
# Test GPU workload
kubectl apply -f test-gpu-pod.yaml

# Monitor GPU allocation
kubectl describe pod test-gpu-pod | grep -A 10 "nvidia.com/gpu"

# Cutover DNS/load balancer to Autopilot cluster
```

**5. Decommission Standard cluster:**
```bash
# After validation period (1-2 weeks)
gcloud container clusters delete old-standard-cluster --region=us-central1
```

## Best Practices Summary

### Do's ✅

1. **Use Autopilot for inference serving** - Automatic scaling, no idle costs
2. **Use Spot pods for fault-tolerant training** - 60-91% cost savings
3. **Implement checkpoint-resume** - Handle Spot pod preemptions gracefully
4. **Monitor GPU utilization** - Alert on low utilization (<30%) to optimize costs
5. **Set appropriate resource limits** - Avoid over-provisioning (waste) or under-provisioning (OOM)
6. **Use Cloud Storage for checkpoints** - Persist across pod restarts
7. **Enable HPA for inference** - Automatic scaling based on traffic
8. **Test with single-GPU first** - Validate workload before scaling to multi-GPU

### Don'ts ❌

1. **Don't use Autopilot for multi-node distributed training (16+ GPUs)** - Lacks Compact Placement Policy optimization
2. **Don't expect node SSH access** - Use `kubectl exec` for debugging instead
3. **Don't deploy custom DaemonSets** - Autopilot restricts node-level customizations
4. **Don't over-request resources** - Pay for what you request, not what you use
5. **Don't run long-term GPU workloads without cost analysis** - Standard mode may be cheaper with Committed Use Discounts
6. **Don't ignore Spot pod preemptions** - Implement proper error handling and checkpointing
7. **Don't mix Autopilot and Standard for same workload** - Choose one mode per cluster
8. **Don't forget GPU quotas** - Request increases before scaling up

## Decision Framework

**Choose GKE Autopilot when:**
- ✅ Operational simplicity > infrastructure control
- ✅ Intermittent GPU workloads (not 24/7)
- ✅ Single-GPU or small multi-GPU (<8 GPUs per job)
- ✅ Inference serving with autoscaling
- ✅ Rapid prototyping and experimentation
- ✅ Small to medium teams (limited infrastructure expertise)
- ✅ Cost-conscious workloads (pay-per-pod)

**Choose GKE Standard when:**
- ✅ Multi-node distributed training (16+ GPUs)
- ✅ Custom DaemonSets required (monitoring, security)
- ✅ Fine-grained networking control (Compact Placement Policy)
- ✅ Sustained GPU workloads (Committed Use Discounts)
- ✅ Node-level debugging and customization
- ✅ Specific machine type requirements
- ✅ Large-scale production ML infrastructure

**Hybrid approach (recommended):**
- Autopilot for **inference** and **development**
- Standard for **production training** at scale
- Shared infrastructure (Cloud Storage, Monitoring, Artifact Registry)

## Sources

**Google Cloud Documentation:**
- [Deploy GPU workloads in Autopilot | GKE AI/ML](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus) - Official Autopilot GPU guide
- [Compare features in Autopilot and Standard clusters](https://docs.cloud.google.com/kubernetes-engine/docs/resources/autopilot-standard-feature-comparison) - Feature comparison table
- [Google Kubernetes Engine pricing](https://cloud.google.com/kubernetes-engine/pricing) - Official pricing documentation
- [Run GPUs in GKE Standard node pools | GKE AI/ML](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpus) - Standard mode GPU documentation for comparison
- [GKE release notes (new features)](https://docs.cloud.google.com/kubernetes-engine/docs/release-notes-new-features) - Latest GPU support announcements

**Community Resources:**
- [How to Set Up a GPU-Enabled Kubernetes Cluster on GKE](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) (vCluster, accessed 2025-11-16) - Step-by-step GPU cluster setup
- [GKE Pricing Guide: Autopilot vs. Standard (2025)](https://www.devzero.io/blog/gke-pricing) (DevZero, accessed 2025-11-16) - Cost analysis and comparison
- [GKE Autopilot vs. Standard Mode: Which One Should You Choose](https://medium.com/@selvamraju007/gke-autopilot-vs-standard-mode-which-one-should-you-choose-390456bba9d2) (Medium, accessed 2025-11-16) - Use case recommendations
- [What are the downsides of using GKE Autopilot](https://www.reddit.com/r/kubernetes/comments/1mpzlyj/what_are_the_downsides_of_using_gke_autopilot/) (Reddit r/kubernetes, accessed 2025-11-16) - Community discussion on limitations
- [Run GPU workloads on GKE Autopilot](https://cloud.google.com/blog/products/containers-kubernetes/run-gpu-workloads-on-gke-autopilot) (Google Cloud Blog, 2022) - Official Autopilot GPU launch announcement
- [GKE Pricing Explained: How to Choose the Right Plan for You](https://cast.ai/blog/gke-pricing-explained-how-to-choose-the-right-plan-for-you/) (Cast AI, accessed 2025-11-16) - Pricing strategies

**Technical Specifications:**
- [SKU Groups - GKE Autopilot Accelerator Premiums](https://cloud.google.com/skus/sku-groups/gke-autopilot-accelerator-premiums) (Google Cloud pricing SKUs)
- [NVIDIA GPU Operator with Google GKE](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.2/google-gke.html) (NVIDIA documentation) - GPU operator details
