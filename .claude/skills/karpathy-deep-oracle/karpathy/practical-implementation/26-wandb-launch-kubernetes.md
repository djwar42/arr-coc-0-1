# W&B Launch Kubernetes Integration

## Overview

W&B Launch on Kubernetes enables automated ML workload execution on Kubernetes clusters, providing ML engineers with a simple interface to use cluster resources while maintaining enterprise-grade GPU scheduling, resource management, and job orchestration capabilities.

From [W&B Launch Kubernetes Tutorial](https://docs.wandb.ai/guides/launch/setup-launch-kubernetes/) (accessed 2025-01-31):
- Launch agent deployed via Helm chart to Kubernetes cluster
- Uses Kaniko builder for Docker image building within cluster
- Supports queue-based job orchestration with priority scheduling
- Integrates with cluster GPU resources for LLM/VLM training

From [W&B Kubernetes Operator Documentation](https://docs.wandb.ai/platform/hosting/operator) (accessed 2025-01-31):
- W&B Kubernetes Operator simplifies deployment and management
- Distributed microservices architecture (multiple pods per service)
- Automatic updates via operator's connection to deploy.wandb.ai
- Production-ready with Prometheus monitoring and OpenTelemetry

## Kubernetes Agent Setup

### Agent Deployment Architecture

**Launch Agent Pod** (`wandb-launch-agent`):
- Polls W&B Launch queue for jobs
- Executes jobs as Kubernetes Job resources
- Handles image building (via Kaniko) or uses pre-built images
- Manages job lifecycle (queued → running → finished)

**Key Components**:
- **Queue Configuration**: FIFO queue with custom resource specs
- **Launch Agent**: Monitors queue, creates Kubernetes Jobs
- **Kaniko Builder**: Builds Docker images within cluster (no Docker daemon needed)
- **Job Pods**: Actual ML training workloads

From [W&B Launch Kubernetes Tutorial](https://docs.wandb.ai/guides/launch/setup-launch-kubernetes/):

```yaml
# Install Launch agent with Helm
helm repo add wandb https://charts.wandb.ai
helm repo update
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```

### Helm Chart Configuration

**values.yaml structure**:

```yaml
agent:
  labels: {}
  apiKey: ''  # W&B API key
  image: wandb/launch-agent:latest
  imagePullPolicy: Always
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

namespace: wandb  # Agent deployment namespace

baseUrl: https://api.wandb.ai

# Target namespaces for job deployment
additionalTargetNamespaces:
  - default
  - wandb

# Launch agent configuration
launchConfig: |
  queues:
    - <queue-name>
  max_jobs: <n-concurrent-jobs>
  environment:
    type: aws  # or gcp, azure
    region: <region>
  registry:
    type: ecr  # or gcr, acr
    uri: <registry-uri>
  builder:
    type: kaniko
    build-context-store: <s3-bucket-uri>

# Git credentials for private repos
gitCreds: |

# Service account annotations (for workload identity)
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account: <gcp-sa>
    azure.workload.identity/client-id: <azure-client-id>
```

### Queue Configuration for Kubernetes

**Kubernetes Job Spec Format**:

From [W&B Launch Kubernetes Tutorial](https://docs.wandb.ai/guides/launch/setup-launch-kubernetes/):

```yaml
spec:
  template:
    spec:
      containers:
        - env:
            - name: MY_ENV_VAR
              value: some-value
          resources:
            requests:
              cpu: 1000m
              memory: 1Gi
metadata:
  labels:
    queue: k8s-test
namespace: wandb
```

**Security Context (auto-injected if not specified)**:

```yaml
spec:
  template:
    backOffLimit: 0
    ttlSecondsAfterFinished: 60
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop:
          - ALL
      seccompProfile:
        type: "RuntimeDefault"
```

### ServiceAccount and RBAC

**Required Permissions**:
- Create/update/delete Kubernetes Jobs
- Create/update/delete Pods
- Access ConfigMaps and Secrets
- Read namespace resources

From [W&B Launch Kubernetes Tutorial](https://docs.wandb.ai/guides/launch/setup-launch-kubernetes/):

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

parquet:
  serviceAccount:
    name: custom-service-account
    create: true
```

**Workload Identity Integration**:
- **GCP**: `iam.gke.io/gcp-service-account` annotation
- **Azure**: `azure.workload.identity/client-id` annotation
- **AWS**: IAM roles for service accounts (IRSA)

## GPU Scheduling on Kubernetes

### Kubernetes GPU Support

From [Kubernetes GPU Scheduling Documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) (accessed 2025-01-31):
- Stable support for AMD and NVIDIA GPUs via device plugins
- GPUs exposed as schedulable resources (e.g., `nvidia.com/gpu`, `amd.com/gpu`)
- GPU resource requests specified in `limits` section only
- Custom resource allocation based on device plugin implementation

**Device Plugin Architecture**:
1. **GPU Drivers**: Installed on nodes by admin
2. **Device Plugin**: DaemonSet exposing GPU resources to kubelet
3. **Extended Resources**: Custom schedulable resources (e.g., `nvidia.com/gpu`)
4. **Scheduler**: Allocates pods to nodes with available GPUs

### GPU Resource Requests

**Basic GPU Request**:

From [Kubernetes GPU Scheduling Documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-job
spec:
  restartPolicy: OnFailure
  containers:
    - name: trainer
      image: "registry.example/llm-trainer:v1"
      resources:
        limits:
          nvidia.com/gpu: 1  # Requesting 1 GPU
```

**Important GPU Request Rules**:
- GPUs specified in `limits` section only
- Can specify `limits` without `requests` (Kubernetes uses limit as request)
- Can specify both `limits` and `requests` (must be equal)
- Cannot specify `requests` without `limits`

### Multi-GPU Job Configuration

**W&B Launch Queue with GPU Allocation**:

```yaml
spec:
  template:
    spec:
      containers:
        - name: training
          resources:
            limits:
              nvidia.com/gpu: 4  # 4 GPUs for multi-GPU training
              cpu: 16000m
              memory: 64Gi
            requests:
              cpu: 16000m
              memory: 64Gi
metadata:
  labels:
    gpu-type: a100
namespace: gpu-training
```

### Node Selection for GPU Types

**Using Node Labels**:

From [Kubernetes GPU Scheduling Documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/):

```bash
# Label nodes with GPU types
kubectl label nodes node1 accelerator=nvidia-a100
kubectl label nodes node2 accelerator=nvidia-h100
kubectl label nodes node3 accelerator=nvidia-t4
```

**Node Affinity for GPU Selection**:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: a100-training-job
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: accelerator
            operator: In
            values: ["nvidia-a100"]
          - key: "gpu.nvidia.com/installed-memory"
            operator: Gt
            values: ["40535"]  # >40GB memory
  containers:
    - name: trainer
      resources:
        limits:
          nvidia.com/gpu: 2
```

### Automatic Node Labeling with NFD

From [Kubernetes GPU Scheduling Documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/):

**Node Feature Discovery (NFD)**:
- Automatically discovers GPU hardware features
- Creates labels for detected features
- Compatible with all supported Kubernetes versions
- Vendor-specific plugins for detailed GPU info

**NFD Feature Labels Example**:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-job-with-nfd
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: "feature.node.kubernetes.io/pci-10.present"
            operator: In
            values: ["true"]  # NFD-detected PCI device
  containers:
    - name: training
      resources:
        limits:
          nvidia.com/gpu: 1
```

**NFD GPU Vendor Plugins**:
- **Intel**: [Intel GPU Device Plugin](https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/gpu_plugin/README.html)
- **NVIDIA**: [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)

### Taints and Tolerations for GPU Nodes

**Taint GPU Nodes** (prevent non-GPU workloads):

```bash
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
```

**Toleration in Pod Spec**:

```yaml
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
  containers:
    - name: training
      resources:
        limits:
          nvidia.com/gpu: 8
```

## Production Deployment Patterns

### Multi-Cluster Setup

**Launch Agent per Cluster**:
- Each Kubernetes cluster has dedicated Launch agent
- Agents poll same W&B Launch queue
- Jobs routed to appropriate cluster via queue configuration
- Load balancing across clusters via queue priority

**Cross-Cluster Configuration**:

```yaml
# Cluster 1 (GKE with A100s)
launchConfig: |
  queues:
    - gpu-a100-queue
  max_jobs: 10
  environment:
    type: gcp
    region: us-central1

# Cluster 2 (EKS with H100s)
launchConfig: |
  queues:
    - gpu-h100-queue
  max_jobs: 5
  environment:
    type: aws
    region: us-west-2
```

### PersistentVolumeClaims for Datasets

**Dataset Storage Pattern**:

From [W&B Kubernetes Operator Documentation](https://docs.wandb.ai/platform/hosting/operator):

```yaml
spec:
  template:
    spec:
      volumes:
        - name: dataset-storage
          persistentVolumeClaim:
            claimName: imagenet-pvc
      containers:
        - name: training
          volumeMounts:
            - name: dataset-storage
              mountPath: /data
          resources:
            limits:
              nvidia.com/gpu: 4
```

**PVC Configuration**:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: imagenet-pvc
  namespace: wandb
spec:
  accessModes:
    - ReadOnlyMany  # Multiple pods can read
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
```

### Secrets Management

**Kubernetes Secrets for API Keys**:

```bash
kubectl create secret generic wandb-secrets \
  --from-literal=WANDB_API_KEY=<key> \
  --from-literal=HF_TOKEN=<token> \
  -n wandb
```

**Reference in Launch Queue**:

```yaml
spec:
  template:
    spec:
      containers:
        - name: training
          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secrets
                  key: WANDB_API_KEY
```

### Resource Quotas and Limits

**Namespace Resource Quota**:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: wandb
spec:
  hard:
    requests.nvidia.com/gpu: "20"  # Max 20 GPUs
    limits.nvidia.com/gpu: "20"
    requests.cpu: "100"
    requests.memory: "500Gi"
```

**LimitRange for Default Resources**:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: wandb
spec:
  limits:
  - max:
      nvidia.com/gpu: "8"  # Max 8 GPUs per pod
    min:
      nvidia.com/gpu: "1"
    type: Container
```

## Cloud Provider Integration

### GKE (Google Kubernetes Engine)

From [Google Cloud Blog - Running W&B Launch on GKE](https://cloud.google.com/blog/products/containers-kubernetes/running-weights-and-biases-launch-ml-platform-on-gke/) (accessed 2025-01-31):

**GKE Cluster Setup**:

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create wandb-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster wandb-cluster \
  --zone us-central1-a \
  --accelerator type=nvidia-tesla-a100,count=4 \
  --machine-type a2-highgpu-4g \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 5
```

**Install NVIDIA GPU Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

**Workload Identity Setup**:

```yaml
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account: wandb-sa@project.iam.gserviceaccount.com
```

### EKS (Amazon Elastic Kubernetes Service)

**EKS Cluster with GPU Nodes**:

```bash
# Create EKS cluster
eksctl create cluster \
  --name wandb-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-ng \
  --node-type p3.8xlarge \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 10 \
  --managed
```

**Install NVIDIA Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**IRSA (IAM Roles for Service Accounts)**:

```yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/wandb-role
```

### AKS (Azure Kubernetes Service)

**AKS GPU Cluster**:

```bash
# Create AKS cluster with GPU node pool
az aks create \
  --resource-group wandb-rg \
  --name wandb-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 10
```

**Install NVIDIA Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**Azure Workload Identity**:

```yaml
serviceAccount:
  annotations:
    azure.workload.identity/client-id: <client-id>
```

## Job Lifecycle and Monitoring

### Job States in W&B Launch

**Job Lifecycle**:
1. **Queued**: Job submitted to queue, waiting for agent
2. **Building**: Agent building Docker image (if needed)
3. **Running**: Kubernetes Job executing
4. **Finished**: Job completed successfully
5. **Failed**: Job failed (pod exit code != 0)
6. **Stopped**: Job manually stopped by user

**Kubernetes Job Spec from Launch**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: wandb-launch-job-abc123
  namespace: wandb
  labels:
    wandb.ai/launch-job: "true"
    wandb.ai/queue: "gpu-training"
spec:
  backoffLimit: 0  # No retries (managed by Launch)
  ttlSecondsAfterFinished: 3600  # Clean up after 1 hour
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: training
          image: <registry>/training:v1
          resources:
            limits:
              nvidia.com/gpu: 2
```

### Monitoring with Prometheus

From [W&B Kubernetes Operator Documentation](https://docs.wandb.ai/platform/hosting/operator):

**W&B Operator Pods**:
- `wandb-app`: Core application (GraphQL API, frontend)
- `wandb-console`: Admin console (/console)
- `wandb-otel`: OpenTelemetry agent (metrics/logs collection)
- `wandb-prometheus`: Prometheus server
- `wandb-parquet`: Database export to Parquet
- `wandb-weave`: Query tables and core features
- `wandb-weave-trace`: LLM tracing framework

**View Logs**:

```bash
# View Launch agent logs
kubectl logs -n wandb-cr deployment/wandb-launch-agent -f

# View training job logs
kubectl logs -n wandb job/wandb-launch-job-abc123 -f

# View all Launch jobs
kubectl get jobs -n wandb -l wandb.ai/launch-job=true
```

### Cost Tracking and Optimization

**Pod Resource Metrics**:

```bash
# Get resource usage for running jobs
kubectl top pods -n wandb --containers

# Get node resource usage
kubectl top nodes
```

**Cost Allocation by Namespace**:
- Use Kubernetes labels for cost attribution
- Track GPU hours per queue/team
- Monitor spot instance usage vs on-demand

**Autoscaling Configuration**:

```yaml
# Horizontal Pod Autoscaler for non-GPU workloads
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wandb-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wandb-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Advanced Patterns

### Gang Scheduling for Multi-GPU Jobs

**Problem**: Multi-GPU jobs need all GPUs available simultaneously

**Solution**: Use gang scheduling with volcano or kube-batch

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  annotations:
    scheduling.volcano.sh/gang: "true"
    scheduling.volcano.sh/min-available: "4"  # Need 4 GPUs
spec:
  template:
    spec:
      schedulerName: volcano
      containers:
        - name: training
          resources:
            limits:
              nvidia.com/gpu: 4
```

### Priority Classes for Job Preemption

**Define Priority Classes**:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-training
value: 1000000
globalDefault: false
description: "High priority for production training"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority-experiments
value: 1000
globalDefault: true
description: "Low priority for experiments"
```

**Use in Launch Queue**:

```yaml
spec:
  template:
    spec:
      priorityClassName: high-priority-training
      containers:
        - name: training
          resources:
            limits:
              nvidia.com/gpu: 8
```

### Spot Instance Integration

**Node Pool with Spot Instances**:

```bash
# GKE spot instance node pool
gcloud container node-pools create spot-gpu-pool \
  --cluster wandb-cluster \
  --spot \
  --accelerator type=nvidia-tesla-a100,count=2 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 20
```

**Toleration for Spot Nodes**:

```yaml
spec:
  template:
    spec:
      tolerations:
      - key: "cloud.google.com/gke-spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
        - name: training
          resources:
            limits:
              nvidia.com/gpu: 2
```

**Checkpoint for Preemption Recovery**:
- Save checkpoints frequently (every N steps)
- Use W&B Artifacts for checkpoint storage
- Launch job automatically retries on preemption

## Troubleshooting

### Common Issues

**1. Pods Pending (Insufficient GPUs)**:

```bash
# Check GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# Check pending pods
kubectl get pods -n wandb --field-selector status.phase=Pending

# Describe pending pod
kubectl describe pod <pod-name> -n wandb
```

**2. Image Pull Errors**:

```bash
# Check image pull secrets
kubectl get secrets -n wandb

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<pass> \
  -n wandb
```

**3. RBAC Permission Errors**:

```bash
# Check service account permissions
kubectl auth can-i create jobs --as=system:serviceaccount:wandb:wandb-launch-agent -n wandb

# View role bindings
kubectl get rolebindings -n wandb
```

**4. GPU Driver Issues**:

```bash
# Verify GPU driver on node
kubectl debug node/<node-name> -it --image=ubuntu
# Inside debug pod:
nvidia-smi
```

### Verification Commands

From [W&B Kubernetes Operator Documentation](https://docs.wandb.ai/platform/hosting/operator):

```bash
# Get W&B operator console password
kubectl get secret wandb-console-password -n default -o jsonpath='{.data.password}' | base64 -d

# Port-forward to console (if ingress not working)
kubectl port-forward svc/wandb-console 8082 -n default

# View W&B Server logs
kubectl logs deployment/wandb-app -n default -f

# Check ingress class
kubectl get ingressclass
```

## Sources

**W&B Documentation:**
- [W&B Launch Kubernetes Tutorial](https://docs.wandb.ai/guides/launch/setup-launch-kubernetes/) (accessed 2025-01-31)
- [W&B Kubernetes Operator Documentation](https://docs.wandb.ai/platform/hosting/operator) (accessed 2025-01-31)
- [Running W&B Launch on GKE](https://cloud.google.com/blog/products/containers-kubernetes/running-weights-and-biases-launch-ml-platform-on-gke/) (accessed 2025-01-31)

**Kubernetes Documentation:**
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) (accessed 2025-01-31)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)

**GPU Vendor Documentation:**
- [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [Intel GPU Device Plugin](https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/gpu_plugin/README.html)

**Additional Resources:**
- [Node Feature Discovery (NFD)](https://kubernetes-sigs.github.io/node-feature-discovery/)
- [W&B Helm Charts](https://github.com/wandb/helm-charts)
