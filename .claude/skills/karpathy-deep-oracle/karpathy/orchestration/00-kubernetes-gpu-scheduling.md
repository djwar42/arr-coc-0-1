# Kubernetes GPU Scheduling: Complete Production Guide

**Knowledge File**: Kubernetes GPU scheduling, NVIDIA GPU Operator, resource management, and best practices for ML/AI workloads

---

## Overview

Kubernetes provides stable support for GPU scheduling through the device plugin framework, enabling AI/ML workloads to access specialized hardware. This guide covers GPU architecture in Kubernetes, the NVIDIA GPU Operator, scheduling patterns, and production best practices.

**Key Capabilities:**
- Automated GPU driver and software stack management
- Dynamic GPU resource allocation and scheduling
- GPU sharing through time-slicing and Multi-Instance GPU (MIG)
- Production-grade monitoring and observability
- Multi-tenant GPU isolation and security

From [Kubernetes Official Documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) (accessed 2025-11-13):
> "Kubernetes includes stable support for managing AMD and NVIDIA GPUs (graphical processing units) across different nodes in your cluster, using device plugins."

---

## Section 1: GPU Architecture in Kubernetes (~100 lines)

### Device Plugin Framework

Kubernetes uses the device plugin framework to expose specialized hardware to containers. GPUs are advertised as custom schedulable resources that pods can request.

**Core Architecture Components:**

From [Kubernetes Device Plugins Documentation](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/):

1. **Device Plugin**: Vendor-specific daemonset that discovers and advertises GPUs
2. **Kubelet**: Manages device plugin registration and resource allocation
3. **Container Runtime**: Provides GPU access to containers (containerd/CRI-O)
4. **GPU Drivers**: Host-level drivers enabling GPU access

**Resource Advertisement:**

GPUs are exposed as custom resources with vendor-specific naming:
- `nvidia.com/gpu` - NVIDIA GPUs
- `amd.com/gpu` - AMD GPUs
- `intel.com/gpu` - Intel GPUs

```yaml
# GPU resource request example
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: gpu-container
    image: tensorflow/tensorflow:latest-gpu
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 NVIDIA GPU
```

**Important GPU Scheduling Rules:**

From [Collabnix Kubernetes GPU Guide](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/) (accessed 2025-11-13):

1. **GPUs must be specified in limits only** - not in requests
2. **GPU limits automatically set GPU requests** to the same value
3. **GPUs are not overcommittable** - exclusive access per container
4. **Fractional GPU requests are not supported** natively (use GPU sharing)
5. **GPU count must be an integer** - cannot request 0.5 GPUs

### Container Runtime Integration

**NVIDIA Container Runtime:**

The NVIDIA Container Runtime hooks into the container lifecycle to:
- Inject GPU device files (`/dev/nvidia*`)
- Mount CUDA libraries and driver files
- Set up GPU isolation between containers
- Enable CUDA API access

**Runtime Configuration:**

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Automatic Node Labeling

From [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/):

> "As an administrator, you can automatically discover and label all your GPU enabled nodes by deploying Kubernetes Node Feature Discovery (NFD)."

**Common GPU Labels Applied:**
- `feature.node.kubernetes.io/pci-10de.present=true` - NVIDIA GPU present
- `nvidia.com/gpu.product` - GPU model (e.g., "Tesla-V100-SXM2-32GB")
- `nvidia.com/gpu.memory` - GPU memory in MB
- `nvidia.com/gpu.count` - Number of GPUs on node
- `nvidia.com/cuda.driver-version` - CUDA driver version

---

## Section 2: NVIDIA GPU Operator Installation (~100 lines)

### What the GPU Operator Does

From [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) (accessed 2025-11-13):

> "The NVIDIA GPU Operator uses the operator framework within Kubernetes to automate the management of all NVIDIA software components needed to provision GPU. These components include the NVIDIA drivers (to enable CUDA), Kubernetes device plugin for GPUs, the NVIDIA Container Toolkit, automatic node labeling using GFD, DCGM based monitoring and others."

**Automated Components:**

1. **NVIDIA GPU Drivers** - Deployed as containers (no host modification)
2. **Kubernetes Device Plugin** - GPU discovery and advertisement
3. **NVIDIA Container Toolkit** - Container runtime integration
4. **GPU Feature Discovery (GFD)** - Automatic node labeling
5. **DCGM Exporter** - GPU metrics for Prometheus
6. **Node Feature Discovery (NFD)** - Hardware feature detection

### Installation Prerequisites

From [Collabnix Installation Guide](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/):

```bash
# Verify GPU nodes exist
kubectl get nodes -o json | jq '.items[].status.capacity'

# Create namespace with privileged access (required for driver installation)
kubectl create namespace gpu-operator
kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged
```

**System Requirements:**
- Kubernetes 1.26+ (stable GPU support)
- Nodes with NVIDIA GPUs
- Supported OS: Ubuntu 22.04/24.04, RHEL 8/9, Rocky Linux 8/9
- Container runtime: containerd 1.6+ or CRI-O 1.23+

### Helm Installation Process

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator (version 25.3.2 as of 2025-11-13)
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version=v25.3.2
```

### Verification Steps

```bash
# Check operator pods (should see 8-10 pods running)
kubectl get pods -n gpu-operator

# Verify GPU nodes are detected
kubectl get nodes -l "nvidia.com/gpu.present=true"

# Check GPU resources available on nodes
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Test GPU access
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:11.0-base --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

**Expected Output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
```

---

## Section 3: GPU Resource Quotas and Limits (~120 lines)

### Resource Quotas for GPU Namespaces

**Namespace-level GPU Quotas:**

```yaml
# GPU resource quota for ML team
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-team
spec:
  hard:
    requests.nvidia.com/gpu: "8"    # Max 8 GPUs total
    limits.nvidia.com/gpu: "8"      # Must match requests
    requests.memory: "128Gi"        # Memory for GPU workloads
    requests.cpu: "64"              # CPU cores
    pods: "20"                      # Limit number of pods
```

**LimitRange for GPU Pods:**

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-team
spec:
  limits:
  - max:
      nvidia.com/gpu: "4"           # Max 4 GPUs per pod
      memory: "64Gi"                # Max memory per pod
      cpu: "32"                     # Max CPU per pod
    min:
      nvidia.com/gpu: "1"           # Min 1 GPU per pod
      memory: "8Gi"                 # Min memory required
      cpu: "4"                      # Min CPU required
    type: Container
```

### Node Taints and Tolerations

**Preventing Non-GPU Workloads on GPU Nodes:**

From [Collabnix Best Practices](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/):

> "Taint GPU nodes to prevent non-GPU workloads from consuming expensive GPU node resources."

```bash
# Taint GPU nodes
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
kubectl taint nodes gpu-node-2 nvidia.com/gpu=present:NoSchedule
```

**Pod with GPU Toleration:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
  containers:
  - name: training
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "8"
```

### Node Affinity for GPU Types

**Scheduling Pods to Specific GPU Models:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: v100-training
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          # Require V100 or A100 GPUs
          - key: "nvidia.com/gpu.product"
            operator: In
            values:
            - "Tesla-V100-SXM2-32GB"
            - "A100-SXM4-40GB"
          # Require nodes with 4+ GPUs
          - key: "nvidia.com/gpu.count"
            operator: Gt
            values: ["4"]
  containers:
  - name: training
    image: training-image:latest
    resources:
      limits:
        nvidia.com/gpu: 4
```

**Common GPU Selection Patterns:**

1. **Cost-optimized inference**: Select T4 GPUs
2. **High-memory training**: Select A100-80GB or H100
3. **Multi-GPU training**: Select nodes with 8+ GPUs
4. **Development**: Use GPU time-slicing on older hardware

---

## Section 4: Multi-GPU Training Jobs (~100 lines)

### Distributed Training with Multiple GPUs

**PyTorchJob with Multi-GPU:**

```yaml
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "pytorch-dist-mnist"
  namespace: ml-jobs
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: kubeflow/pytorch-dist-mnist:latest
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "16Gi"
                cpu: "8"
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: kubeflow/pytorch-dist-mnist:latest
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "16Gi"
                cpu: "8"
```

### Horovod Distributed Training

From [Collabnix Distributed Training](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/):

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: horovod-training
spec:
  parallelism: 4  # 4 worker pods
  template:
    spec:
      containers:
      - name: horovod-worker
        image: horovod/horovod:0.28.1-tf2.11.0-torch1.13.1-py3.8-gpu
        command:
        - horovodrun
        args:
        - -np
        - "4"
        - --host-discovery-script
        - /usr/local/bin/discover_hosts.sh
        - python
        - /training/train.py
        env:
        - name: OMPI_MCA_plm_rsh_agent
          value: "ssh"
        - name: NCCL_DEBUG
          value: "INFO"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
      restartPolicy: Never
```

### GPU Communication Patterns

**NCCL (NVIDIA Collective Communications Library):**

Optimized for multi-GPU and multi-node communication:
- All-reduce operations for gradient synchronization
- Ring and tree algorithms for efficient communication
- GPUDirect RDMA for node-to-node GPU communication
- Automatic topology detection

**Key Environment Variables:**

```yaml
env:
- name: NCCL_DEBUG
  value: "INFO"              # Enable NCCL logging
- name: NCCL_IB_DISABLE
  value: "0"                 # Enable InfiniBand if available
- name: NCCL_NET_GDR_LEVEL
  value: "5"                 # GPUDirect RDMA level
- name: NCCL_SOCKET_IFNAME
  value: "eth0"              # Network interface for NCCL
```

---

## Section 5: GPU Monitoring with DCGM (~80 lines)

### DCGM (Data Center GPU Manager)

The NVIDIA GPU Operator automatically deploys DCGM Exporter for Prometheus metrics.

**Key GPU Metrics Available:**

From [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html):

1. **Utilization Metrics:**
   - `DCGM_FI_DEV_GPU_UTIL` - GPU utilization %
   - `DCGM_FI_DEV_MEM_COPY_UTIL` - Memory bandwidth utilization %
   - `DCGM_FI_DEV_SM_CLOCK` - SM clock frequency

2. **Memory Metrics:**
   - `DCGM_FI_DEV_FB_USED` - Framebuffer memory used (MB)
   - `DCGM_FI_DEV_FB_FREE` - Framebuffer memory free (MB)
   - `DCGM_FI_PROF_DRAM_ACTIVE` - DRAM active cycles

3. **Temperature & Power:**
   - `DCGM_FI_DEV_GPU_TEMP` - GPU temperature (°C)
   - `DCGM_FI_DEV_POWER_USAGE` - Power usage (W)
   - `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` - Total energy (mJ)

### Prometheus Integration

**ServiceMonitor for DCGM:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gpu-metrics
  namespace: gpu-operator
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  endpoints:
  - port: gpu-metrics
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
```

### GPU Alerting Rules

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
  namespace: monitoring
spec:
  groups:
  - name: gpu.rules
    rules:
    - alert: GPUHighUtilization
      expr: DCGM_FI_DEV_GPU_UTIL > 95
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU {{ $labels.gpu }} utilization high"
        description: "GPU utilization is {{ $value }}%"

    - alert: GPUMemoryExhausted
      expr: DCGM_FI_DEV_FB_FREE < 1024
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "GPU {{ $labels.gpu }} memory exhausted"
        description: "Only {{ $value }}MB free GPU memory"

    - alert: GPUTemperatureHigh
      expr: DCGM_FI_DEV_GPU_TEMP > 85
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "GPU {{ $labels.gpu }} temperature high"
        description: "GPU temperature is {{ $value }}°C"
```

---

## Section 6: Production Best Practices (~100 lines)

### Resource Right-Sizing

From [Collabnix Best Practices](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/):

**CPU and Memory Guidelines:**

```yaml
# Well-sized GPU pod
apiVersion: v1
kind: Pod
metadata:
  name: optimized-training
spec:
  containers:
  - name: training
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 4              # 4 GPUs
        memory: "64Gi"                 # 16GB per GPU + overhead
        cpu: "32"                      # 8 CPU cores per GPU
      requests:
        nvidia.com/gpu: 4              # Match limits
        memory: "48Gi"                 # Allow flexibility
        cpu: "24"                      # Allow scheduling flexibility
    env:
    - name: OMP_NUM_THREADS
      value: "8"                       # CPU threads per process
```

**Key Sizing Principles:**

1. **Memory**: 12-16GB RAM per GPU (minimum 8GB)
2. **CPU**: 6-8 CPU cores per GPU for data preprocessing
3. **Storage**: Use high-IOPS storage for training data
4. **Network**: 10Gbps+ for multi-node training

### Init Containers for Model Loading

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-serving
spec:
  initContainers:
  # Download model before GPU allocation
  - name: model-downloader
    image: busybox
    command: ['sh', '-c']
    args:
    - |
      wget -O /models/model.pt https://example.com/model.pt
      echo "Model downloaded successfully"
    volumeMounts:
    - name: model-storage
      mountPath: /models
  containers:
  - name: serving
    image: tritonserver:latest-gpu
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - name: model-storage
      mountPath: /models
  volumes:
  - name: model-storage
    emptyDir:
      sizeLimit: "50Gi"
```

### Security Best Practices

**Pod Security Standards:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-gpu-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: gpu-container
    image: tensorflow/tensorflow:latest-gpu
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
      readOnlyRootFilesystem: true
    resources:
      limits:
        nvidia.com/gpu: 1
```

**Network Policies for GPU Workloads:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gpu-workload-policy
  namespace: ml-jobs
spec:
  podSelector:
    matchLabels:
      workload-type: gpu-training
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ml-platform
    ports:
    - protocol: TCP
      port: 8080
  egress:
  # Allow DNS
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: UDP
      port: 53
  # Allow HTTPS for model downloads
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

### Cost Optimization

**Cluster Autoscaling Configuration:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  # GPU node pool configuration
  scale-down-enabled: "true"
  scale-down-unneeded-time: "10m"      # Scale down after 10m idle
  scale-down-delay-after-add: "10m"    # Wait 10m after scale-up
  max-node-provision-time: "15m"       # Timeout for node provisioning
```

**Spot/Preemptible Instance Tolerations:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-job
spec:
  template:
    spec:
      # Tolerate spot instances for cost savings
      tolerations:
      - key: "cloud.google.com/gke-preemptible"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      - key: "kubernetes.azure.com/scalesetpriority"
        operator: "Equal"
        value: "spot"
        effect: "NoSchedule"
      containers:
      - name: training
        image: training-image:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## arr-coc-0-1 Use Cases

### VLM Training on Multiple GPUs

The ARR-COC architecture benefits from multi-GPU training for:

1. **Texture Array Processing**: Parallel processing of 13-channel texture arrays
2. **Relevance Scorer Training**: Distributed training of propositional, perspectival, and participatory scorers
3. **Opponent Processing Optimization**: Multi-GPU batch processing for balancing training
4. **Quality Adapter Fine-tuning**: Efficient fine-tuning across GPU resources

**Sample ARR-COC Training Job:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: arr-coc-training
  namespace: ml-jobs
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: "nvidia.com/gpu.product"
                operator: In
                values: ["A100-SXM4-40GB", "A100-SXM4-80GB"]
      containers:
      - name: training
        image: arr-coc-trainer:latest
        command: ["python", "train.py"]
        args:
        - --model=arr-coc-vlm
        - --gpus=4
        - --batch-size=32
        - --mixed-precision=bf16
        env:
        - name: NCCL_DEBUG
          value: "INFO"
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "128Gi"
            cpu: "32"
        volumeMounts:
        - name: training-data
          mountPath: /data
        - name: checkpoints
          mountPath: /checkpoints
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: vqa-dataset-pvc
      - name: checkpoints
        persistentVolumeClaim:
          claimName: model-checkpoints-pvc
      restartPolicy: Never
```

### Real-time Inference Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arr-coc-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arr-coc-inference
  template:
    metadata:
      labels:
        app: arr-coc-inference
    spec:
      nodeSelector:
        nvidia.com/gpu.product: "Tesla-T4"  # Cost-effective T4 for inference
      containers:
      - name: inference-server
        image: arr-coc-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/arr-coc-latest"
        - name: BATCH_SIZE
          value: "8"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
```

---

## Sources

**Official Documentation:**
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) - Official Kubernetes documentation (accessed 2025-11-13)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) - Device plugin framework
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) - Complete GPU Operator guide (accessed 2025-11-13)

**Web Research:**
- [Collabnix: Kubernetes and GPU Complete Guide 2025](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/) (accessed 2025-11-13)
  - Comprehensive GPU architecture
  - GPU Operator installation
  - Multi-GPU training patterns
  - Production best practices

**Additional References:**
- NVIDIA DCGM Exporter for Prometheus metrics
- Kubeflow PyTorchJob and TFJob operators
- Horovod distributed training framework
- NCCL (NVIDIA Collective Communications Library)
