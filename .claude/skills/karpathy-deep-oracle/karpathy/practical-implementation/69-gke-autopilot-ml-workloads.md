# GKE Autopilot for ML Workloads: Production Kubernetes Without the Ops Burden

**Knowledge File: GKE Autopilot architecture, GPU support, multi-GPU training, cost optimization, and production ML deployment patterns**

---

## Overview

GKE Autopilot is Google's fully managed Kubernetes mode where Google handles all infrastructure management, node provisioning, scaling, and security patches. For ML workloads, Autopilot removes operational burden while providing GPU support (H100, A100, L4, B200, GB200, T4), automatic scaling, and pay-per-pod pricing.

**Core value proposition:**
- **Zero node management** - No node pool configuration, no manual scaling, no OS patches
- **Pay-per-pod pricing** - Pay only for requested pod resources (CPU, memory, ephemeral storage)
- **Security by default** - Minimal OS, auto-applied patches, workload identity built-in
- **GPU support** - NVIDIA GPUs (H100, A100, L4, T4) with managed driver installation
- **Automatic scaling** - Nodes and pods scale dynamically based on workload demand

From [Hands-Off Kubernetes in 2025](https://medium.com/google-cloud/hands-off-kubernetes-in-2025-a-practical-walkthrough-of-gke-autopilot-04d82833b2ed) (Medium, accessed 2025-01-13):
> "GKE Autopilot turns Kubernetes into a service you consume, not babysit. Pick the right compute class, keep IAM tight, design the network with intention — and let Google handle the grunt work."

**When to use Autopilot for ML:**
- Training jobs with variable GPU requirements (don't want idle GPUs)
- Inference workloads that need to scale to zero during off-peak
- Multi-team ML platforms (each team gets isolated virtual clusters)
- Cost-conscious ML projects (pay-per-pod eliminates node over-provisioning)
- Teams without dedicated Kubernetes ops expertise

---

## Section 1: GKE Autopilot Overview (100 lines)

### 1.1 Autopilot vs Standard Mode

**GKE offers two modes of operation:**

**Standard Mode:**
- **Manual node management** - You configure machine types, node pools, autoscaling
- **Pay for nodes** - Billed for entire VM instances (even if pods use 10% of capacity)
- **Full control** - Custom node configurations, privileged containers, node affinity
- **Use when** - You need specific hardware, custom kernels, or fine-grained node control

**Autopilot Mode:**
- **Fully managed** - Google handles node provisioning, scaling, upgrades, security patches
- **Pay per pod** - Billed only for requested pod resources (CPU, memory, ephemeral storage)
- **Opinionated constraints** - No privileged containers, no node SSH, limited customization
- **Use when** - You want simplicity, cost efficiency, and don't need custom node configs

From [GKE Autopilot vs. Standard Mode](https://medium.com/@selvamraju007/gke-autopilot-vs-standard-mode-which-one-should-you-choose-390456bba9d2) (Medium, accessed 2025-01-13):
> "GKE Autopilot is ideal for teams that prioritize simplicity and cost efficiency, while GKE Standard is better for customizable, high-performance workloads."

**Key architectural difference:**
```
Standard Mode:
User → Manages node pools → Provisions VMs → Schedules pods

Autopilot Mode:
User → Defines pod requests → Google provisions nodes automatically
```

### 1.2 Autopilot Pricing Model

**Pay-per-pod pricing (no node overhead):**
- **CPU**: ~$0.028/vCPU/hour (based on pod requests, not node capacity)
- **Memory**: ~$0.003/GB/hour
- **Ephemeral storage**: Included up to pod memory limit
- **System pods**: Free (logging, monitoring, service mesh helpers)
- **Cluster management**: $0.10/hour/cluster (flat fee)

**Cost comparison example (10 pods, 2 vCPU, 8GB RAM each):**
```
Standard Mode (3 n1-standard-4 nodes):
3 nodes × $0.19/hour × 730 hours/month = $416/month

Autopilot Mode (10 pods × 2 vCPU × 8GB):
(10 × 2 × $0.028 + 10 × 8 × $0.003) × 730 + $73 = $582/month

BUT if pods only run 12 hours/day:
(10 × 2 × $0.028 + 10 × 8 × $0.003) × 365 + $73 = $264/month
```

**When Autopilot saves money:**
- **Variable workloads** - Training jobs that run intermittently
- **Scale-to-zero scenarios** - Inference endpoints with off-peak idle time
- **Multi-tenant platforms** - Many small workloads (avoid node over-provisioning)

From [Hands-Off Kubernetes in 2025](https://medium.com/google-cloud/hands-off-kubernetes-in-2025-a-practical-walkthrough-of-gke-autopilot-04d82833b2ed):
> "A podcast-editing startup compresses audio overnight using GPU compute classes flagged as Spot Pods. By letting jobs run during off-peak hours, the monthly bill lands at roughly half the price of keeping traditional VMs online all day."

### 1.3 Compute Classes (Hardware Selection)

**Autopilot uses "compute classes" to specify hardware requirements:**

**Balanced (default):**
- **General-purpose** ML workloads (inference, lightweight training)
- CPU:Memory ratio ~1:4
- `cloud.google.com/compute-class: Balanced`

**Scale-Out:**
- **Thousands of lightweight pods** (distributed inference, microservices)
- CPU:Memory ratio ~1:2 (more CPU per GB)
- `cloud.google.com/compute-class: Scale-Out`

**Performance:**
- **High-performance compute** (large batch training, data preprocessing)
- Latest CPU generations, higher network bandwidth
- `cloud.google.com/compute-class: Performance`

**Accelerator (GPU):**
- **GPU-accelerated ML** (training, inference with H100/A100/L4/T4)
- Requires `nvidia.com/gpu` resource limit
- Example: `cloud.google.com/gke-accelerator: nvidia-tesla-t4`

**Example pod spec with compute class:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-job
  annotations:
    cloud.google.com/compute-class: Performance
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    resources:
      requests:
        cpu: "16"
        memory: "64Gi"
      limits:
        cpu: "16"
        memory: "64Gi"
```

### 1.4 Security Built-In

**Autopilot enforces security best practices by default:**
- **Workload Identity** - No service account keys in YAML (uses Kubernetes service accounts)
- **Network Policies** - Pod-to-pod traffic control enforced automatically
- **Minimal OS** - Container-Optimized OS with reduced attack surface
- **Automatic patches** - Security updates applied during maintenance windows (no downtime)
- **Binary Authorization** - Optional policy enforcement for container image signing

**No privileged containers:**
```yaml
# ❌ This FAILS in Autopilot
securityContext:
  privileged: true  # Not allowed

# ✅ Use this instead
securityContext:
  runAsNonRoot: true
  capabilities:
    drop:
    - ALL
```

From [Hands-Off Kubernetes in 2025](https://medium.com/google-cloud/hands-off-kubernetes-in-2025-a-practical-walkthrough-of-gke-autopilot-04d82833b2ed):
> "A fintech startup uses Policy Controller to scan every manifest for encryption at rest and minimum resource requests before it reaches the cluster — no more 'forgotten debug container' scandals."

---

## Section 2: ML Training on Autopilot (130 lines)

### 2.1 GPU Support and Driver Installation

**GKE Autopilot supports NVIDIA GPUs:**
- **H100** - Latest flagship GPU (80GB HBM3, 4,000 TFLOPs FP8)
- **A100** - Previous flagship (40GB/80GB, 312 TFLOPs FP16)
- **L4** - Cost-effective inference (24GB, 242 TFLOPs FP16, Ada Lovelace)
- **T4** - Budget-friendly (16GB, 130 TFLOPs FP16, Turing)
- **B200/GB200** - Next-gen Blackwell GPUs (preview)

**GPU driver installation modes:**

**1. Managed drivers (recommended for Autopilot):**
```bash
# GKE handles driver installation automatically
gcloud container node-pools create gpu-pool \
    --accelerator "type=nvidia-tesla-t4,count=1,gpu-driver-version=latest" \
    --cluster=my-cluster \
    --zone=us-central1-a
```

**2. NVIDIA GPU Operator (for Standard mode):**
- Manual installation via Helm
- More control over driver versions
- Required for custom kernels or driver tuning

From [How to Set Up a GPU-Enabled Kubernetes Cluster on GKE](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) (vCluster, accessed 2025-01-13):
> "Autopilot Mode: For a hands-off approach, GKE Autopilot allows GPU workload deployment without the need for infrastructure management."

**Requesting GPUs in pod specs:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-gpu-training
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    command: ["python", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
```

**GPU node provisioning flow:**
```
Pod requests nvidia.com/gpu: 1
    ↓
Autopilot checks available GPU nodes
    ↓
If no capacity: Provisions new GPU node (2-5 minutes)
    ↓
Installs NVIDIA drivers (managed by GKE)
    ↓
Schedules pod to GPU node
    ↓
Pod starts training
```

### 2.2 Multi-GPU Training with PyTorch DDP

**Autopilot supports multi-GPU training via PyTorch DistributedDataParallel (DDP):**

**Single-node multi-GPU (easiest):**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ddp-training-4gpu
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    command:
    - "torchrun"
    - "--nproc_per_node=4"
    - "train.py"
    resources:
      limits:
        nvidia.com/gpu: 4  # Request 4 GPUs on same node
    env:
    - name: MASTER_ADDR
      value: "localhost"
    - name: MASTER_PORT
      value: "29500"
```

**Multi-node multi-GPU (for large-scale training):**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ddp-multinode
spec:
  parallelism: 2  # 2 worker pods
  completions: 2
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
        command:
        - "torchrun"
        - "--nnodes=2"
        - "--nproc_per_node=2"  # 2 GPUs per node
        - "--master_addr=$(MASTER_ADDR)"
        - "--master_port=29500"
        - "train.py"
        resources:
          limits:
            nvidia.com/gpu: 2
        env:
        - name: MASTER_ADDR
          value: "ddp-master-service"  # Kubernetes Service for rank 0 pod
```

**Environment variables for DDP:**
- `MASTER_ADDR` - IP/hostname of rank 0 node
- `MASTER_PORT` - Port for inter-process communication (default 29500)
- `WORLD_SIZE` - Total number of processes (num_nodes × GPUs_per_node)
- `RANK` - Global rank of this process (0 to WORLD_SIZE-1)
- `LOCAL_RANK` - GPU index on this node (0 to GPUs_per_node-1)

**Cost optimization with Spot pods:**
```yaml
metadata:
  annotations:
    cloud.google.com/gke-spot: "true"  # Use Spot GPUs (60-91% cheaper)
spec:
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 1
```

**Warning: Spot interruptions:**
- Spot GPUs can be preempted with 30-second notice
- Use checkpointing every 5-10 minutes to handle interruptions
- Autopilot will re-schedule pod on new node after preemption

### 2.3 Kubernetes Jobs for ML Training

**Use Kubernetes Jobs for batch training (better than Deployments):**

**Simple training job:**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: bert-finetuning
spec:
  backoffLimit: 3  # Retry 3 times on failure
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: huggingface/transformers-pytorch-gpu:4.30.0
        command: ["python", "finetune_bert.py"]
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
          limits:
            nvidia.com/gpu: 1
```

**Job with W&B experiment tracking:**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: resnet-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
        command: ["python", "train_resnet.py"]
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Job with checkpoint persistence (GCS bucket):**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpt-pretraining
spec:
  template:
    spec:
      serviceAccountName: gcs-writer-sa  # Workload Identity
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
        command:
        - "python"
        - "train_gpt.py"
        - "--checkpoint-dir=gs://my-bucket/checkpoints"
        resources:
          limits:
            nvidia.com/gpu: 4
```

### 2.4 PersistentVolumeClaims for Datasets

**Store datasets in GCS buckets (recommended):**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: imagenet-training
spec:
  serviceAccountName: gcs-reader-sa
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    command:
    - "python"
    - "train.py"
    - "--data-dir=gs://my-bucket/imagenet"  # gcsfuse mounts this
    resources:
      limits:
        nvidia.com/gpu: 2
```

**Or use PersistentVolumeClaims (for shared datasets across pods):**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: imagenet-pvc
spec:
  accessModes:
  - ReadOnlyMany  # Multiple pods can read simultaneously
  resources:
    requests:
      storage: 500Gi
  storageClassName: standard-rwo
---
apiVersion: v1
kind: Pod
metadata:
  name: trainer-pod
spec:
  volumes:
  - name: dataset
    persistentVolumeClaim:
      claimName: imagenet-pvc
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    volumeMounts:
    - name: dataset
      mountPath: /data
    resources:
      limits:
        nvidia.com/gpu: 1
```

---

## Section 3: Production Deployment (100 lines)

### 3.1 Deployments and Services for Inference

**Deploy VLM inference endpoint:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llava-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llava
  template:
    metadata:
      labels:
        app: llava
    spec:
      containers:
      - name: inference-server
        image: llava/llava-v1.5-13b:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llava-service
spec:
  selector:
    app: llava
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3.2 Horizontal Pod Autoscaling

**Scale inference pods based on CPU/memory:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llava-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llava-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Custom metrics (e.g., queue depth from Cloud Monitoring):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: custom-metrics-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: pubsub.googleapis.com|subscription|num_undelivered_messages
        selector:
          matchLabels:
            resource.labels.subscription_id: "inference-queue"
      target:
        type: AverageValue
        averageValue: "10"
```

### 3.3 Cost Optimization with Spot Pods

**Use Spot pods for cost-tolerant workloads (60-91% savings):**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: batch-inference
  annotations:
    cloud.google.com/gke-spot: "true"
spec:
  containers:
  - name: inference
    image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 1
```

**Spot interruption handling:**
- **30-second notice** before preemption (via SIGTERM)
- **Graceful shutdown** - Use `preStop` hook to save state
- **Automatic rescheduling** - Autopilot provisions new node and reschedules pod

**preStop hook example:**
```yaml
lifecycle:
  preStop:
    exec:
      command:
      - "sh"
      - "-c"
      - "python save_checkpoint.py && sleep 25"
```

### 3.4 Load Balancing for Inference

**Ingress with HTTPS and global load balancing:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llava-ingress
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "llava-ip"
    networking.gke.io/managed-certificates: "llava-cert"
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1/inference
        pathType: Prefix
        backend:
          service:
            name: llava-service
            port:
              number: 80
---
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: llava-cert
spec:
  domains:
  - api.example.com
```

---

## Section 4: W&B Launch + GKE Integration (70 lines)

### 4.1 Launch Agents on GKE Autopilot

**Deploy W&B Launch agent to run training jobs from queue:**
```bash
# Install agent via Helm
helm repo add wandb https://charts.wandb.ai
helm repo update

helm upgrade --install wandb-agent wandb/operator \
  --namespace wandb \
  --create-namespace \
  --set agent.apiKey=$WANDB_API_KEY \
  --set agent.queue=gke-gpu-queue
```

**Agent configuration (values.yaml):**
```yaml
agent:
  image: wandb/launch-agent:latest
  apiKey: ""  # Set via --set or Secret
  queue: "gke-gpu-queue"
  maxJobs: 5
  resources:
    limits:
      cpu: "1"
      memory: "2Gi"

# Launch jobs will request GPUs via pod specs
jobTemplate:
  spec:
    containers:
    - name: training
      resources:
        limits:
          nvidia.com/gpu: 1
```

### 4.2 Queue-Based Job Scheduling

**W&B Launch workflow:**
```
Developer pushes code → W&B Launch queue
    ↓
Launch agent (running in GKE) polls queue
    ↓
Agent creates Kubernetes Job with GPU request
    ↓
Autopilot provisions GPU node (if needed)
    ↓
Training pod starts, logs to W&B
    ↓
Pod completes, node scales down (if idle)
```

**Example Launch job config:**
```yaml
# launch-config.yaml
job_type: training
resource: kubernetes
resource_args:
  kubernetes_job:
    metadata:
      name: bert-finetuning-{{RUN_ID}}
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: wandb/deeplearning:pytorch-cuda11.8
            command: ["python", "train.py"]
            resources:
              limits:
                nvidia.com/gpu: 2
```

### 4.3 Multi-Node Training Orchestration

**W&B Launch can orchestrate multi-node PyTorch DDP:**
```yaml
# launch-multinode.yaml
job_type: training
resource: kubernetes
resource_args:
  kubernetes_job:
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: ddp-training-{{RUN_ID}}
    spec:
      parallelism: 4  # 4 worker pods
      completions: 4
      template:
        spec:
          containers:
          - name: trainer
            image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
            command:
            - "torchrun"
            - "--nnodes=4"
            - "--nproc_per_node=2"
            - "train.py"
            resources:
              limits:
                nvidia.com/gpu: 2
```

**Agent automatically handles:**
- Job submission to Kubernetes API
- Pod status monitoring
- Log streaming to W&B
- Cleanup after completion

---

## Sources

**Web Research:**

1. [Hands-Off Kubernetes in 2025: A Practical Walkthrough of GKE Autopilot](https://medium.com/google-cloud/hands-off-kubernetes-in-2025-a-practical-walkthrough-of-gke-autopilot-04d82833b2ed) - Medium, Aleksei Aleinikov (accessed 2025-01-13)
   - Autopilot architecture, cost model, compute classes, security defaults
   - Real-world examples: podcast-editing startup, sports analytics, VR hosting

2. [GKE Autopilot vs. Standard Mode: Which One Should You Choose?](https://medium.com/@selvamraju007/gke-autopilot-vs-standard-mode-which-one-should-you-choose-390456bba9d2) - Medium, Selvam Raju (accessed 2025-01-13)
   - Autopilot vs Standard comparison, pricing models, use cases

3. [How to Set Up a GPU-Enabled Kubernetes Cluster on GKE](https://www.vcluster.com/blog/gcp-gke-gpu-cluster) - vCluster (accessed 2025-01-13)
   - GPU setup walkthrough, driver installation, multi-tenancy with vCluster
   - Testing GPU workloads, resource management patterns

4. Google Search: "GKE Autopilot GPU 2024 2025" (accessed 2025-01-13)
   - GPU support on Autopilot (H100, A100, L4, T4, B200, GB200)
   - Managed driver installation, fast-starting nodes for L4 GPUs

5. Google Search: "GKE Autopilot vs Standard ML workloads" (accessed 2025-01-13)
   - Feature comparison, pricing models, when to use each mode

6. Google Search: "GKE Autopilot multi-GPU training Kubernetes" (accessed 2025-01-13)
   - Multi-GPU support, PyTorch DDP patterns, job scheduling

**Existing Knowledge:**

7. [W&B Launch Kubernetes Integration](../26-wandb-launch-kubernetes.md) - Existing file
   - Launch agent deployment, queue-based orchestration, Helm chart configuration

8. [Distributed Training Patterns](../../vertex-ai-production/00-distributed-training-patterns.md) - Existing file
   - PyTorch DDP fundamentals, multi-node communication, gradient synchronization

---

**Knowledge gaps filled:**
- **GKE Autopilot architecture** - Fully managed Kubernetes, pay-per-pod pricing, compute classes
- **GPU support on Autopilot** - H100/A100/L4/T4 GPUs, managed driver installation, fast provisioning
- **ML training patterns** - Multi-GPU DDP, Kubernetes Jobs, checkpoint persistence
- **Production deployment** - HPA, Spot pods, load balancing, cost optimization
- **W&B Launch integration** - Queue-based scheduling, multi-node orchestration on Autopilot

**Connection to ARR-COC:**
- **VLM training** - Run ARR-COC training jobs on Autopilot with automatic GPU provisioning
- **Inference deployment** - Deploy ARR-COC relevance realization as scalable inference endpoint
- **Cost optimization** - Pay-per-pod pricing ideal for variable VLM workloads (avoid idle GPU costs)
- **Multi-team platforms** - Use Autopilot + vCluster for isolated ARR-COC development environments
