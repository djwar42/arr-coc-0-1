# Vertex AI Fundamentals: Architecture, Custom Jobs, and GCP Setup

**Knowledge File: Deep technical foundations for production ML training on Google Cloud**

---

## Overview

Vertex AI is Google Cloud's unified machine learning platform that consolidates training, deployment, and ML operations into a single service. Unlike AWS SageMaker's toolkit approach with numerous individual services, Vertex AI provides an integrated experience where components work seamlessly together.

**Core Value Proposition:**
- Unified platform reducing configuration overhead
- Managed infrastructure with automatic scaling
- Deep integration with Google Cloud services (GCS, BigQuery, Cloud Logging)
- Production-ready ML workflows without manual VM orchestration

**When to Use Vertex AI:**
- Large-scale distributed training (multi-GPU, multi-node, TPU pods)
- Production ML pipelines requiring end-to-end orchestration
- Teams already invested in GCP ecosystem
- Projects requiring tight integration with BigQuery for data processing

**When NOT to Use Vertex AI:**
- Simple single-GPU experiments (use Colab Pro or Compute Engine directly)
- Extreme cost sensitivity (raw Compute Engine VMs are cheaper)
- Multi-cloud requirements (prefer Kubernetes-based solutions)
- Need for maximum control over infrastructure (use GKE instead)

From [Vertex AI Overview](https://cloud.google.com/vertex-ai/docs/training/overview) (accessed 2025-01-31):
> "Vertex AI operationalizes training at scale by managing the infrastructure, allowing ML engineers to focus on model development rather than DevOps."

---

## Section 1: Vertex AI Architecture and Core Services

### 1.1 Unified ML Platform Components

Vertex AI integrates multiple ML workflow stages:

**Training Services:**
- **Custom Jobs**: Full control over training code and environment
- **Hyperparameter Tuning Jobs**: Automated parameter search with parallel trials
- **AutoML**: No-code model training for standard use cases
- **Pipelines**: Orchestrated multi-step ML workflows (Kubeflow-based)

**Prediction Services:**
- **Online Prediction**: Low-latency synchronous inference (60s timeout)
- **Batch Prediction**: Asynchronous bulk inference over datasets
- **Model Monitoring**: Drift detection and performance tracking

**Data Services:**
- **Datasets**: Managed storage for training data with versioning
- **Feature Store**: Centralized feature management with online/offline serving
- **Labeling**: Integrated data annotation workflows

**Model Management:**
- **Model Registry**: Version control and lineage tracking
- **Model Evaluation**: Automated metrics computation and visualization
- **Experiments**: Track training runs with parameters and metrics

### 1.2 Vertex AI vs Alternatives

**Vertex AI vs AWS SageMaker:**

From [SageMaker vs Vertex AI for Model Inference](https://towardsdatascience.com/sagemaker-vs-vertex-ai-for-model-inference-ef0d503cee76/) (accessed 2025-01-31):
> "Compared to Vertex AI, SageMaker is generally more feature-rich and flexible, without losing sight of its original goal of making ML workflows easy."

Key differences:
- **Autoscaling**: SageMaker scales on QPS metrics; Vertex AI only CPU/GPU utilization
- **Async endpoints**: SageMaker supports 15-min async inference; Vertex AI sync-only (60s timeout)
- **Multi-model endpoints**: SageMaker shares resources across models; Vertex AI only shares URLs
- **Minimum instances**: Vertex AI requires ≥1 instance always; SageMaker can scale to 0 (async mode)
- **Integration**: Vertex AI simpler for GCP-native stacks; SageMaker better for AWS ecosystems

**Vertex AI vs Azure ML:**
- Vertex AI has stronger TPU support for TensorFlow/JAX workloads
- Azure ML better integrates with Microsoft enterprise tools
- Vertex AI's BigQuery integration superior for data-heavy pipelines
- Azure ML more mature notebook environment (Azure ML Studio)

**Vertex AI vs On-Premise:**
- Vertex AI eliminates infrastructure management overhead
- On-prem provides data sovereignty and full control
- Vertex AI auto-scales; on-prem requires manual capacity planning
- Cost: Vertex AI pay-per-use vs on-prem capital investment

### 1.3 Custom Jobs vs AutoML vs Pipelines

**Custom Jobs:**
- Use case: Research experiments, novel architectures, custom training loops
- Control: Full (own Docker container, training code, hyperparameters)
- Complexity: Medium (must write training script and configure resources)
- Cost: Pay only for compute time used

**AutoML:**
- Use case: Standard tasks (image classification, text sentiment, tabular prediction)
- Control: Low (Google's algorithms, limited hyperparameter exposure)
- Complexity: Low (point-and-click interface)
- Cost: Higher per-training-hour vs Custom Jobs

**Pipelines:**
- Use case: Multi-step workflows (data preprocessing → training → evaluation → deployment)
- Control: High (orchestrate any combination of Custom Jobs and services)
- Complexity: High (requires Kubeflow Pipelines SDK knowledge)
- Cost: Pay for pipeline orchestration + individual component costs

### 1.4 Vertex AI Training Architecture

**High-Level Workflow:**
```
1. User submits CustomJob config (Python SDK / gcloud CLI / Console)
2. Vertex AI provisions WorkerPool VMs with specified machine types
3. VMs pull Docker container from Artifact Registry
4. Container runs training script with mounted GCS data
5. Checkpoints/artifacts written to GCS
6. Job completes; VMs automatically terminated
7. Model registered to Model Registry (optional)
```

**Key Components:**
- **CustomJob**: Top-level resource defining the training configuration
- **WorkerPoolSpec**: Specification for each replica group (chief, workers, parameter servers)
- **MachineSpec**: Machine type, accelerators, disk configuration
- **ContainerSpec**: Docker image, entry point, arguments, environment variables
- **PythonPackageSpec**: Alternative to containers for pure Python code

**Architecture Diagram (Conceptual):**
```
┌─────────────────────────────────────────────────────────┐
│ Vertex AI Control Plane                                 │
│  ∙ Job scheduling & orchestration                       │
│  ∙ Resource provisioning                                │
│  ∙ Health monitoring & auto-restart                     │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐             ┌────────▼────────┐
│ WorkerPool 0   │             │ WorkerPool 1    │
│ (Chief Replica)│             │ (Worker Replicas│
│                │             │                 │
│ Machine: n1-.. │◄───TF──────►│ Machine: n1-... │
│ Accelerator:   │    Dist     │ Accelerator:    │
│   NVIDIA_A100  │   Training  │   NVIDIA_A100   │
│ Container:     │   Comms     │ Container:      │
│   my-image:v1  │◄───gRPC────►│   my-image:v1   │
└────────┬───────┘             └─────────┬───────┘
         │                               │
         └───────────┬───────────────────┘
                     │
              ┌──────▼──────┐
              │ Cloud Storage
              │  ∙ Training data
              │  ∙ Checkpoints
              │  ∙ Model artifacts
              └─────────────┘
```

### 1.5 Pricing Structure

**Custom Training Costs (as of 2024-2025):**

**Compute (per hour):**
- n1-standard-4 (4 vCPU, 15GB RAM): ~$0.19/hr
- n1-highmem-8 (8 vCPU, 52GB RAM): ~$0.47/hr
- a2-highgpu-1g (12 vCPU, 85GB RAM, 1x A100 40GB): ~$3.67/hr
- a2-highgpu-8g (96 vCPU, 680GB RAM, 8x A100 40GB): ~$29.39/hr
- TPU v4 pod slice (8 chips): ~$3.60/hr per chip = ~$28.80/hr

**Storage:**
- Cloud Storage (Standard): $0.02/GB/month
- Cloud Storage (Nearline): $0.01/GB/month
- Persistent Disk (SSD): $0.17/GB/month

**Data Egress:**
- Within same region: Free
- Cross-region (same continent): $0.01/GB
- Internet egress: $0.12/GB (first 1TB)

**Cost Optimization Strategies:**
- Use Spot VMs (preemptible): 60-91% discount vs on-demand
- Store checkpoints in Nearline storage (for long-term archives)
- Use local SSD for temporary data during training
- Commit to 1-year/3-year usage: 25-52% discount
- Right-size machine types (don't over-provision CPU/memory)

From [GCP Pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-01-31):
> "Vertex AI Training charges for compute resources used during training, with no additional platform fees."

**Example Cost Calculation:**
Training a VLM on 8x A100 GPUs for 24 hours:
- Compute: $29.39/hr × 24hr = $705.36
- Storage (1TB data + 500GB checkpoints): ~$30/month (prorated to ~$1/day)
- Total: ~$706 for single training run

With Spot VMs (if workload fault-tolerant):
- Compute: $705.36 × 0.30 (70% discount) = $211.61
- Total: ~$212 (66% savings)

---

## Section 2: Custom Jobs Deep Dive

### 2.1 CustomJob vs HyperparameterTuningJob

**CustomJob:**
- Single training run with fixed hyperparameters
- Use case: Final model training, debugging, experimentation
- Resource allocation: Static (you specify exact machine count)
- Example: Train BERT-large on 4x V100 GPUs with learning_rate=2e-5

**HyperparameterTuningJob:**
- Multiple parallel trials exploring hyperparameter space
- Use case: Find optimal learning rate, batch size, layer dimensions
- Resource allocation: Dynamic (Vertex AI schedules trials based on available quota)
- Search algorithms: Grid search, random search, Bayesian optimization
- Early stopping: Automatically terminate underperforming trials

**Key Differences:**
```python
# CustomJob: Single fixed configuration
custom_job = aiplatform.CustomJob(
    display_name="vit-training",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "a2-highgpu-1g"},
        "replica_count": 4,
        "container_spec": {"image_uri": "gcr.io/my-project/vit:v1"}
    }]
)

# HyperparameterTuningJob: Multiple trials
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="vit-hptuning",
    custom_job=custom_job,  # Template for each trial
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
        "batch_size": hpt.DiscreteParameterSpec(values=[16, 32, 64])
    },
    max_trial_count=20,
    parallel_trial_count=5  # Run 5 trials simultaneously
)
```

From [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview) (accessed 2025-01-31):
> "Vertex AI uses Bayesian optimization to intelligently select hyperparameter values, reducing the number of trials needed to find optimal configurations."

### 2.2 WorkerPoolSpec Architecture

Vertex AI supports distributed training via multiple worker pools:

**Worker Pool Roles:**
1. **Chief (Master)**: Coordinates training, manages checkpoints, logs metrics
2. **Workers**: Execute training computation (data parallelism)
3. **Parameter Servers**: Store model parameters in async SGD (TensorFlow-specific)
4. **Evaluators**: Run evaluation during training (separate from training workers)

**Single-Node Training (Most Common):**
```python
worker_pool_specs = [{
    "machine_spec": {
        "machine_type": "n1-highmem-8",
        "accelerator_type": "NVIDIA_TESLA_V100",
        "accelerator_count": 1
    },
    "replica_count": 1,  # Single VM
    "container_spec": {
        "image_uri": "us-docker.pkg.dev/my-project/my-repo/trainer:latest",
        "command": ["python", "train.py"],
        "args": ["--epochs=50", "--batch_size=32"]
    }
}]
```

**Multi-GPU Single-Node (Data Parallel):**
```python
worker_pool_specs = [{
    "machine_spec": {
        "machine_type": "a2-highgpu-4g",  # Machine with 4x A100 GPUs
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 4  # All GPUs on same VM
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "gcr.io/my-project/trainer:v1",
        "command": ["torchrun", "--nproc_per_node=4", "train.py"]  # PyTorch DDP
    }
}]
```

**Multi-Node Distributed Training:**
```python
worker_pool_specs = [
    {  # Chief worker
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/my-project/trainer:v1",
            "env": [{"name": "RANK", "value": "0"}]  # Chief rank
        }
    },
    {  # Worker replicas
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1
        },
        "replica_count": 3,  # 3 additional workers
        "container_spec": {
            "image_uri": "gcr.io/my-project/trainer:v1",
            "env": [{"name": "RANK", "value": "1"}]  # Worker ranks
        }
    }
]
# Total: 4 nodes × 1 GPU = 4 GPUs distributed across nodes
```

**Parameter Server Architecture (TensorFlow):**
```python
worker_pool_specs = [
    {"machine_spec": {...}, "replica_count": 1},   # Chief
    {"machine_spec": {...}, "replica_count": 4},   # Workers
    {"machine_spec": {...}, "replica_count": 2}    # Parameter servers
]
```

### 2.3 Machine Types and Accelerators

**N-Series (General Purpose):**
- n1-standard-{4,8,16,32}: Balanced CPU/memory
- n1-highmem-{2,4,8}: Memory-optimized (6.5GB RAM per vCPU)
- n1-highcpu-{2,4,8}: CPU-optimized (0.9GB RAM per vCPU)
- Use case: Preprocessing, data loading, CPU-heavy augmentation

**N2-Series (Newer Intel CPUs):**
- n2-standard-{4,8,16,32}: 10-30% better price/performance vs N1
- n2-highmem-{2,4,8,16}: Up to 640GB RAM
- Use case: Latest CPU workloads, better single-thread performance

**A2-Series (GPU-Optimized):**
- a2-highgpu-1g: 1× A100 40GB (12 vCPU, 85GB RAM)
- a2-highgpu-2g: 2× A100 40GB (24 vCPU, 170GB RAM)
- a2-highgpu-4g: 4× A100 40GB (48 vCPU, 340GB RAM)
- a2-highgpu-8g: 8× A100 40GB (96 vCPU, 680GB RAM)
- a2-ultragpu-1g: 1× A100 80GB (12 vCPU, 170GB RAM)
- a2-ultragpu-8g: 8× A100 80GB (96 vCPU, 1360GB RAM)
- NVLink/NVSwitch: 600GB/s inter-GPU bandwidth (8-GPU models)

**A3-Series (H100 GPUs - Preview):**
- a3-highgpu-8g: 8× H100 80GB (208 vCPU, 1872GB RAM)
- NVLink Gen4: 900GB/s inter-GPU bandwidth
- Use case: Cutting-edge LLM training (GPT-4 scale)

**G2-Series (Cost-Effective Inference):**
- g2-standard-{4,8,12}: NVIDIA L4 GPUs
- Use case: Inference, not training (L4 has lower FP32 performance)

**Accelerator Selection Guide:**

| Workload | Recommended GPU | Reasoning |
|----------|----------------|-----------|
| Small models (<1B params) | V100 16GB | Cost-effective, sufficient memory |
| Medium models (1-7B params) | A100 40GB | 2.5× faster than V100, more memory |
| Large models (7-70B params) | A100 80GB | Double memory vs 40GB variant |
| Cutting-edge (>70B params) | H100 80GB (A3) | 3× faster than A100 for FP16/BF16 |
| Vision models (high-res) | A100 40/80GB | Large batch sizes, image memory |
| Distributed training | 8× A100 (a2-highgpu-8g) | NVSwitch for efficient all-reduce |

**TPU Options:**
- TPU v4 pods: 4096 chips, 1.1 exaFLOPS (BF16)
- TPU v5e: Cost-optimized, 2× v4 performance per dollar
- TPU v5p: Cutting-edge, 2× v5e performance (limited availability)
- Use case: TensorFlow/JAX workloads at massive scale

From [Vertex AI Machine Types](https://cloud.google.com/vertex-ai/docs/training/configure-compute) (accessed 2025-01-31):
> "A2 VMs with 8 A100 GPUs use NVSwitch for 600GB/s all-to-all GPU connectivity, enabling efficient distributed training."

### 2.4 Container Requirements

**Mandatory Container Capabilities:**
1. **HTTP Health Check Endpoint**: Container must respond to `GET /health` or similar
2. **Training Script Entry Point**: Must run training when container starts
3. **GCS Access**: Container needs `google-cloud-storage` or `gcsfs` for data I/O
4. **Signal Handling**: Gracefully handle SIGTERM for preemption/cancellation
5. **Exit Codes**: Return 0 on success, non-zero on failure

**Environment Variables Provided by Vertex AI:**
```bash
CLUSTER_SPEC         # JSON with worker addresses (for distributed training)
TF_CONFIG           # TensorFlow-specific distributed config
CLOUD_ML_JOB_ID     # Unique job identifier
CLOUD_ML_TRIAL_ID   # Trial ID (for hyperparameter tuning)
AIP_MODEL_DIR       # GCS path to save model artifacts
AIP_CHECKPOINT_DIR  # GCS path for checkpoints
AIP_TENSORBOARD_LOG_DIR  # TensorBoard logging directory
```

**Dockerfile Example:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install training dependencies
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

# Copy training code
COPY train.py /app/
COPY model.py /app/

# Set working directory
WORKDIR /app

# Entry point: Run training script
ENTRYPOINT ["python3", "train.py"]
```

**Key Container Best Practices:**
- Use official NVIDIA CUDA base images for GPU compatibility
- Pin dependency versions (`torch==2.1.0` not `torch>=2.0`)
- Minimize image size (multi-stage builds, slim base images)
- Test locally with `docker run --gpus all` before deployment
- Use Artifact Registry for versioned image storage

### 2.5 Job Lifecycle and States

**Job States:**
1. **QUEUED**: Job submitted, waiting for resources
2. **PREPARING**: VMs provisioning, pulling container image
3. **RUNNING**: Training actively executing
4. **SUCCEEDED**: Job completed successfully (exit code 0)
5. **FAILED**: Job failed (non-zero exit code or error)
6. **CANCELLED**: User-initiated cancellation
7. **CANCELLING**: Cancellation in progress

**Typical Timeline:**
```
QUEUED: 0-5 minutes (depends on quota availability)
PREPARING: 3-8 minutes (VM boot + image pull)
RUNNING: Variable (your training duration)
SUCCEEDED/FAILED: Immediate (VMs auto-terminated)
```

**Monitoring Job State:**
```bash
# Via gcloud CLI
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Via Python SDK
job = aiplatform.CustomJob.get("projects/123/locations/us-central1/customJobs/456")
print(job.state)  # JobState.JOB_STATE_SUCCEEDED
```

**Error Handling:**
- Vertex AI automatically retries failed jobs (configurable)
- Preempted Spot VMs: Job transitions to QUEUED, waits for capacity
- OOM errors: Check `jobDetail.taskDetails` for specific replica failures
- Network failures: Vertex AI retries for transient issues

### 2.6 Output Artifacts and Model Registry

**Artifact Storage Pattern:**
```python
# In training script (train.py)
import os
from google.cloud import storage

# Vertex AI sets this environment variable
model_dir = os.environ["AIP_MODEL_DIR"]  # e.g., gs://my-bucket/models/job-123

# Save model artifacts
torch.save(model.state_dict(), "/tmp/model.pth")

# Upload to GCS
client = storage.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("models/job-123/model.pth")
blob.upload_from_filename("/tmp/model.pth")

print(f"Model saved to {model_dir}/model.pth")
```

**Automatic Model Registry Upload:**
```python
custom_job = aiplatform.CustomJob(
    # ... job config ...
    base_output_dir="gs://my-bucket/outputs"  # Auto-uploads artifacts here
)

# After training, register model
model = aiplatform.Model.upload(
    display_name="vit-large-v1",
    artifact_uri="gs://my-bucket/outputs/model",  # Output from training
    serving_container_image_uri="gcr.io/my-project/serve:v1"
)
print(f"Model registered: {model.resource_name}")
```

**Model Versioning Best Practices:**
- Use semantic versioning: `model-name-v1.0.0`
- Include training date in display name: `bert-large-2025-01-31`
- Tag with hyperparameters: `lr=2e-5_bs=32`
- Store experiment metadata in Model description field

---

## Section 3: GCP Setup Prerequisites

### 3.1 GCP Project Creation and Billing

**Creating a GCP Project:**
```bash
# Via gcloud CLI
gcloud projects create my-ml-project --name="My ML Project"

# Set as active project
gcloud config set project my-ml-project

# Link billing account (required for resource usage)
gcloud billing projects link my-ml-project --billing-account=ABCDEF-123456-GHIJKL
```

**Checking Billing Status:**
```bash
gcloud billing projects describe my-ml-project
```

**Budget Alerts:**
```bash
# Create budget with alerts at 50%, 90%, 100% spend
gcloud billing budgets create \
    --billing-account=ABCDEF-123456-GHIJKL \
    --display-name="ML Training Budget" \
    --budget-amount=5000USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100
```

From [GCP Billing Setup](https://cloud.google.com/billing/docs/how-to/modify-project) (accessed 2025-01-31):
> "All GCP resources require an active billing account. Training jobs will fail if billing is not properly configured."

### 3.2 Enabling Vertex AI APIs

**Required APIs:**
```bash
# Core Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Container Registry (for Docker images)
gcloud services enable containerregistry.googleapis.com

# Artifact Registry (newer, recommended)
gcloud services enable artifactregistry.googleapis.com

# Cloud Storage (for data and artifacts)
gcloud services enable storage.googleapis.com

# Compute Engine (underlying VMs)
gcloud services enable compute.googleapis.com

# Cloud Logging (for job logs)
gcloud services enable logging.googleapis.com
```

**Verification:**
```bash
gcloud services list --enabled | grep aiplatform
# Should output: aiplatform.googleapis.com
```

### 3.3 Service Accounts and IAM Roles

**Default Service Account:**
Vertex AI creates a default service account:
```
<project-number>-compute@developer.gserviceaccount.com
```

**Creating Custom Service Account:**
```bash
# Create service account
gcloud iam service-accounts create vertex-training-sa \
    --display-name="Vertex AI Training Service Account"

# Assign necessary roles
PROJECT_ID=$(gcloud config get-value project)
SA_EMAIL="vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Role 1: Vertex AI User (submit jobs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

# Role 2: Storage Admin (read/write GCS buckets)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.admin"

# Role 3: Artifact Registry Reader (pull Docker images)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.reader"

# Role 4: Logging Writer (write job logs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"
```

**Minimum Required Roles:**
- `roles/aiplatform.user`: Submit and manage training jobs
- `roles/storage.objectViewer`: Read training data from GCS
- `roles/storage.objectCreator`: Write checkpoints/artifacts to GCS
- `roles/artifactregistry.reader`: Pull Docker images

**Best Practice: Principle of Least Privilege:**
```bash
# Grant specific bucket access only (not project-wide storage.admin)
gsutil iam ch serviceAccount:${SA_EMAIL}:objectAdmin gs://my-training-bucket
```

### 3.4 Artifact Registry Setup

**Creating Artifact Registry Repository:**
```bash
# Create Docker repository
gcloud artifacts repositories create ml-containers \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML training containers"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

**Configuring Docker Authentication:**
```bash
# Configure Docker to use gcloud for authentication
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**Pushing Docker Image:**
```bash
# Tag image
IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/ml-containers/trainer:v1"
docker tag trainer:latest $IMAGE_URI

# Push to Artifact Registry
docker push $IMAGE_URI
```

**Image Naming Convention:**
```
<region>-docker.pkg.dev/<project-id>/<repository>/<image>:<tag>
us-central1-docker.pkg.dev/my-project/ml-containers/vit-trainer:v2.1.0
```

### 3.5 Cloud Storage Bucket Creation

**Creating Training Data Bucket:**
```bash
# Create regional bucket (co-located with training VMs)
gsutil mb -l us-central1 -c STANDARD gs://my-training-data

# Create bucket for model artifacts
gsutil mb -l us-central1 -c STANDARD gs://my-model-artifacts

# Set lifecycle policy (auto-delete old checkpoints after 90 days)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 90, "matchesPrefix": ["checkpoints/"]}
    }]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://my-model-artifacts
```

**Bucket Organization Best Practices:**
```
gs://my-training-data/
  ├── datasets/
  │   ├── imagenet/
  │   │   ├── train/
  │   │   └── val/
  │   └── coco/
  └── preprocessed/

gs://my-model-artifacts/
  ├── checkpoints/
  │   ├── job-123/
  │   └── job-124/
  └── models/
      ├── vit-v1/
      └── vit-v2/
```

**Optimizing GCS Performance:**
- Use regional buckets (same region as training VMs)
- Parallel uploads: `gsutil -m cp` for large files
- Composite uploads: Automatic for files >150MB
- Object versioning: Enable for critical datasets

### 3.6 Network Configuration

**Default Network (Sufficient for Most Use Cases):**
Vertex AI uses the `default` VPC network automatically.

**Custom VPC (For Advanced Security):**
```bash
# Create custom VPC
gcloud compute networks create ml-vpc --subnet-mode=custom

# Create subnet in training region
gcloud compute networks subnets create ml-subnet \
    --network=ml-vpc \
    --region=us-central1 \
    --range=10.0.0.0/24

# Configure firewall (allow inter-VM communication for distributed training)
gcloud compute firewall-rules create ml-internal \
    --network=ml-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/24
```

**VPC Peering (Access On-Prem Data):**
```bash
gcloud compute networks peerings create vertex-to-onprem \
    --network=ml-vpc \
    --peer-network=https://www.googleapis.com/compute/v1/projects/onprem-project/global/networks/onprem-vpc
```

**Private Google Access (No Public IPs):**
```bash
gcloud compute networks subnets update ml-subnet \
    --region=us-central1 \
    --enable-private-ip-google-access
```

### 3.7 Quota Management

**Checking Current Quotas:**
```bash
gcloud compute project-info describe --project=$PROJECT_ID \
    | grep -A 5 "NVIDIA_A100_GPUS"
```

**Common Quotas to Request:**
- `NVIDIA_A100_GPUS`: Number of A100 GPUs (default: 0)
- `NVIDIA_V100_GPUS`: Number of V100 GPUs (default: 8)
- `CPUS`: Total CPUs across all VMs (default: 24)
- `PREEMPTIBLE_CPUS`: CPUs for Spot VMs (default: 24)
- `DISKS_TOTAL_GB`: Total disk space (default: 500GB)

**Requesting Quota Increase:**
```bash
# Via Cloud Console
# 1. Navigate to: IAM & Admin > Quotas
# 2. Filter: "a100" or "Service: Compute Engine API"
# 3. Select quota (e.g., "NVIDIA_A100_GPUS in us-central1")
# 4. Click "EDIT QUOTAS"
# 5. Enter new limit + justification
# 6. Submit (typically approved in 1-2 business days)

# Example justification:
# "Training vision-language models for production deployment.
#  Require 8x A100 GPUs for distributed training.
#  Expected usage: 100 hours/month."
```

**Quota Best Practices:**
- Request quotas early (approval takes 24-48 hours)
- Request per-region quotas (not global)
- Start with modest requests (e.g., 8 GPUs), increase iteratively
- Provide specific use case in justification
- Monitor usage with Cloud Monitoring dashboards

From [GCP Quotas](https://cloud.google.com/vertex-ai/docs/quotas) (accessed 2025-01-31):
> "GPU quotas are region-specific. Request increases for the specific region where you plan to run training jobs."

---

## Key Takeaways

**Vertex AI Strengths:**
1. Unified platform: Training, serving, MLOps in one service
2. Deep GCP integration: BigQuery, GCS, Cloud Logging seamless
3. Managed infrastructure: No VM management, auto-scaling
4. TPU access: Cutting-edge hardware for TensorFlow/JAX

**Vertex AI Limitations:**
1. Autoscaling: Only CPU/GPU metrics (no QPS-based scaling)
2. Async inference: Not supported (60s timeout for sync calls)
3. Minimum instances: Cannot scale to 0 (always ≥1 VM)
4. Platform lock-in: GCP-specific APIs (not cloud-agnostic)

**When Vertex AI is the Right Choice:**
- GCP-native tech stacks (BigQuery, Cloud Composer, etc.)
- Large-scale distributed training (multi-node, TPU pods)
- Teams without DevOps capacity (want managed services)
- Strong Google Cloud partnership or credits

**When to Consider Alternatives:**
- Multi-cloud requirements: Use Kubernetes (EKS, GKE, AKS)
- Maximum flexibility: Use raw Compute Engine VMs + custom orchestration
- AWS-centric teams: SageMaker offers more features (async endpoints, QPS scaling, MME)
- Cost-sensitive: Raw VMs are cheaper (but require manual management)

**Next Steps:**
- Review W&B Launch integration (file 31-wandb-launch-vertex-agent.md)
- Understand GPU/TPU selection (file 32-vertex-ai-gpu-tpu.md)
- Learn container optimization (file 33-vertex-ai-containers.md)
- Explore production patterns (file 35-vertex-ai-production-patterns.md)

---

## Sources

**Official Documentation:**
- [Vertex AI Training Overview](https://cloud.google.com/vertex-ai/docs/training/overview) (accessed 2025-01-31)
- [Create Custom Job](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) (accessed 2025-01-31)
- [Distributed Training](https://cloud.google.com/vertex-ai/docs/training/distributed-training) (accessed 2025-01-31)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-01-31)
- [Machine Types](https://cloud.google.com/vertex-ai/docs/training/configure-compute) (accessed 2025-01-31)

**Comparisons and Best Practices:**
- [SageMaker vs Vertex AI for Model Inference](https://towardsdatascience.com/sagemaker-vs-vertex-ai-for-model-inference-ef0d503cee76/) - Julia Turc, Towards Data Science (accessed 2025-01-31)
- [AWS SageMaker vs Google Vertex AI vs Azure ML](https://aws.plainenglish.io/aws-sagemaker-vs-google-vertex-ai-vs-azure-ml-cloud-ml-platform-reality-918c76059904) (accessed 2025-01-31)
- [Google Cloud Vertex AI Development Workflow](https://medium.com/google-cloud/google-cloud-vertex-ai-development-workflow-for-beginners-3b9274be4de3) (accessed 2025-01-31)

**Tutorials:**
- [Custom ML Training with Docker and Vertex AI](https://medium.com/google-cloud/custom-ml-training-with-docker-and-google-cloud-vertex-ai-f8cbbaf23552) (accessed 2025-01-31)
- [Vertex AI Tutorial: Comprehensive Guide](https://www.datacamp.com/tutorial/vertex-ai-tutorial) (accessed 2025-01-31)
