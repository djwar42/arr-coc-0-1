# Vertex AI Custom Jobs - Deep Dive

## Overview

Vertex AI custom jobs (accessed via `gcloud ai custom-jobs create`) provide full control over training infrastructure for machine learning workloads. Unlike AutoML, custom jobs let you run your own training code with complete flexibility over compute resources, distributed training configurations, and container environments.

**Key Capabilities:**
- Run any ML framework (TensorFlow, PyTorch, JAX, scikit-learn, etc.)
- Distribute training across multiple nodes and GPUs/TPUs
- Use custom containers or prebuilt training containers
- Integrate with GCS, Artifact Registry, and other GCP services
- Support for hyperparameter tuning and experiment tracking

From [Vertex AI Custom Training Documentation](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job) (accessed 2025-02-03):
> Custom training jobs (CustomJob resources in the Vertex AI API) are the basic way to run your custom machine learning (ML) training code in Vertex AI.

## gcloud ai custom-jobs Commands

### Core Command Structure

```bash
gcloud ai custom-jobs create \
  --region=REGION \
  --display-name=JOB_NAME \
  --worker-pool-spec=WORKER_POOL_SPEC \
  [--config=CONFIG_FILE]
```

**Essential Flags:**
- `--region`: GCP region (e.g., us-central1, us-west1, europe-west4)
- `--display-name`: Human-readable job name
- `--worker-pool-spec`: Defines compute resources and training code
- `--config`: YAML file with complete job specification (alternative to inline flags)

### Worker Pool Specification

Worker pools define the compute resources and container configuration. You can specify multiple worker pools for distributed training.

**Inline Format:**
```bash
--worker-pool-spec=machine-type=MACHINE_TYPE,\
replica-count=REPLICA_COUNT,\
accelerator-type=ACCELERATOR_TYPE,\
accelerator-count=ACCELERATOR_COUNT,\
container-image-uri=IMAGE_URI
```

**Example - Single Node with 8 V100 GPUs:**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=pytorch-training \
  --worker-pool-spec=machine-type=n1-standard-32,\
replica-count=1,\
accelerator-type=NVIDIA_TESLA_V100,\
accelerator-count=8,\
container-image-uri=gcr.io/my-project/pytorch-trainer:latest
```

From [Vertex AI Training Documentation](https://cloud.google.com/vertex-ai/docs/training/configure-compute) (accessed 2025-02-03):
> If you are using the Google Cloud CLI, then you can use the --worker-pool-spec flag or the --config flag on the gcloud ai custom-jobs create command to specify compute resources.

### Config File Format (YAML)

For complex jobs, use `--config` with a YAML file:

```yaml
displayName: "distributed-pytorch-training"
jobSpec:
  workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-32
      acceleratorType: NVIDIA_TESLA_V100
      acceleratorCount: 8
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/my-project/pytorch-trainer:latest
      args:
        - "--epochs=100"
        - "--batch-size=64"
      env:
        - name: NCCL_DEBUG
          value: INFO
```

**Advantages of Config Files:**
- Better for version control
- Easier to manage complex distributed setups
- Can define environment variables, command args, and more
- Reusable across jobs

## Distributed Training Architecture

### Multi-Node Training Setup

Vertex AI supports distributed training by defining multiple worker pools. The first pool is always the **primary replica** (rank 0), and additional pools are **worker replicas**.

**Example - 4 Nodes with 4 GPUs Each:**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=multi-node-training \
  --worker-pool-spec=machine-type=n1-highmem-16,\
replica-count=1,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=4,\
container-image-uri=gcr.io/my-project/trainer:latest \
  --worker-pool-spec=machine-type=n1-highmem-16,\
replica-count=3,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=4,\
container-image-uri=gcr.io/my-project/trainer:latest
```

From [Vertex AI Distributed Training](https://docs.cloud.google.com/vertex-ai/docs/training/distributed-training) (accessed 2025-02-03):
> You can configure any custom training job as a distributed training job by defining multiple worker pools. All distributed training jobs have one primary replica in the first worker pool.

### Worker Pool Structure

Each worker pool defines:

1. **Machine Type**: CPU/RAM configuration
   - `n1-standard-N`: General purpose (1-96 vCPUs)
   - `n1-highmem-N`: Memory-optimized (2-96 vCPUs, 6.5GB RAM/vCPU)
   - `n1-highcpu-N`: CPU-optimized (2-96 vCPUs)
   - `c2-standard-N`: Compute-optimized (4-60 vCPUs, 4GB RAM/vCPU)
   - `a2-highgpu-N`: GPU-optimized (12-96 vCPUs, up to 16 A100 GPUs)

2. **Accelerators**: GPU/TPU configuration
   - `NVIDIA_TESLA_K80`: 12GB VRAM (legacy)
   - `NVIDIA_TESLA_P4`: 8GB VRAM
   - `NVIDIA_TESLA_T4`: 16GB VRAM (cost-effective)
   - `NVIDIA_TESLA_V100`: 16GB VRAM (high performance)
   - `NVIDIA_TESLA_P100`: 16GB VRAM
   - `NVIDIA_TESLA_A100`: 40GB or 80GB VRAM (highest performance)
   - `TPU_V2`, `TPU_V3`, `TPU_V4`: Cloud TPU pods

3. **Replica Count**: Number of replicas in this pool
   - Primary pool: Usually `replica-count=1`
   - Worker pools: `replica-count=N` for N-1 additional nodes

4. **Container Spec**: Docker image and execution details

### PyTorch Distributed Training Pattern

**Training Code Structure:**
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize distributed training environment."""
    # Vertex AI sets these environment variables
    rank = int(os.environ.get('CLOUD_ML_HP_TUNING_RANK', 0))
    world_size = int(os.environ.get('CLOUD_ML_HP_TUNING_WORLD_SIZE', 1))

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU training
        init_method='env://',  # Use environment variables
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def train():
    rank, world_size, local_rank = setup_distributed()

    # Create model and move to GPU
    model = YourModel()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # Standard training code
            ...
```

From [Vertex AI Distributed Training Guide](https://docs.cloud.google.com/vertex-ai/docs/training/distributed-training) (accessed 2025-02-03):
> Your training code uses TensorFlow or PyTorch and is configured for multi-host data-parallel training with GPUs using NCCL all-reduce.

### Environment Variables

Vertex AI automatically sets these for distributed training:

- `CLOUD_ML_HP_TUNING_RANK`: Global rank (0 = primary replica)
- `CLOUD_ML_HP_TUNING_WORLD_SIZE`: Total number of replicas
- `LOCAL_RANK`: Rank within the current node (for multi-GPU nodes)
- `MASTER_ADDR`: IP address of rank 0 node
- `MASTER_PORT`: Port for communication (default: 23456)

## Custom Container Requirements

Your training container must:

1. **Accept Configuration from Environment Variables**
   - Read hyperparameters from env vars or command-line args
   - Use `AIP_MODEL_DIR` for saving model artifacts
   - Use `AIP_TRAINING_DATA_URI` for input data location

2. **Handle Distributed Training**
   - Initialize distributed backend (NCCL for GPUs, Gloo for CPUs)
   - Use rank/world_size from environment variables
   - Synchronize across replicas

3. **Write Outputs to GCS**
   - Save checkpoints to `gs://bucket/path/`
   - Export final model to Cloud Storage
   - Log metrics for tracking

4. **Exit Cleanly**
   - Return exit code 0 on success
   - Return non-zero on failure
   - Clean up distributed processes

**Example Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage \
    tensorboard \
    wandb

# Copy training code
COPY train.py /app/train.py
COPY utils/ /app/utils/

WORKDIR /app

# Entry point
ENTRYPOINT ["python", "train.py"]
```

From [Vertex AI Custom Container Requirements](https://cloud.google.com/vertex-ai/docs/training/containers-overview) (accessed 2025-02-03):
> A custom container is a Docker image that you create to run your training application. By running your machine learning (ML) training job in a custom container, you can use ML frameworks and libraries that aren't available in Vertex AI's prebuilt containers.

## Resource Pools and Scheduling

### Scheduling Options

Vertex AI provides flexible scheduling for training jobs:

**Standard Scheduling:**
- Immediate execution when resources available
- Pre-allocated resources for duration of job
- Higher cost, guaranteed availability

**Spot VMs (Preemptible):**
- 60-91% cost savings
- Can be interrupted if capacity needed
- Best for fault-tolerant workloads with checkpointing

```yaml
jobSpec:
  scheduling:
    timeout: 86400s  # 24 hours
    restartJobOnWorkerRestart: true
  workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-16
    replicaCount: 4
    containerSpec:
      imageUri: gcr.io/my-project/trainer:latest
```

From [Vertex AI Training Service](https://cloud.google.com/vertex-ai/docs/training/understanding-training-service) (accessed 2025-02-03):
> During a training job, Vertex AI can restart your workers from any worker pool with the same hostname. This can occur for VM maintenance.

### Quotas and Limits

**Per-Region Quotas:**
- GPU quotas (e.g., "NVIDIA_T4_GPUS"): Number of GPUs
- CPU quotas: vCPU count
- TPU quotas: Number of TPU cores
- Worker pool count: Max 10 pools per job
- Replica count: Max 1000 replicas per pool

**Request Quota Increases:**
```bash
# Check current quotas
gcloud compute project-info describe --project=PROJECT_ID

# Request increase via Cloud Console
# IAM & Admin > Quotas > Filter by "Vertex AI" or GPU type
```

From [Vertex AI Quotas](https://docs.cloud.google.com/vertex-ai/docs/quotas) (accessed 2025-02-03):
> Quotas restrict how much of a particular shared Google Cloud resource your project can use. For Vertex AI, quotas apply to training jobs, prediction endpoints, and other resources.

### Resource Pool Best Practices

1. **Start Small, Scale Up**
   - Test on 1 GPU first
   - Verify distributed setup works with 2 nodes
   - Scale to full cluster size

2. **Monitor GPU Utilization**
   - Use `nvidia-smi` logs in training code
   - Check for data loading bottlenecks
   - Ensure batch sizes fully utilize GPUs

3. **Use Appropriate Machine Types**
   - Match CPU/RAM to GPU count (rule of thumb: 4-8 vCPUs per GPU)
   - Use high-memory machines for large models
   - Consider local SSD for I/O-intensive workloads

4. **Handle Preemption**
   - Save checkpoints frequently
   - Resume from last checkpoint on restart
   - Use `restartJobOnWorkerRestart: true`

## Job Monitoring and Debugging

### Viewing Job Status

```bash
# List recent jobs
gcloud ai custom-jobs list --region=us-central1

# Get job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Common Issues

**1. Container Fails to Start**
- Verify image exists in Artifact Registry/GCR
- Check IAM permissions (Vertex AI Service Agent needs Artifact Registry Reader)
- Ensure container can run on specified machine type

**2. Distributed Training Hangs**
- Verify NCCL initialization (set `NCCL_DEBUG=INFO`)
- Check firewall rules (nodes must communicate)
- Ensure all replicas start successfully
- Verify `init_process_group()` timeout is sufficient

**3. Out of Memory Errors**
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training (FP16/BF16)

**4. Slow Training**
- Check data loading (use DataLoader with `num_workers>0`)
- Verify GPU utilization (should be >80%)
- Use distributed data parallel, not model parallel
- Profile with PyTorch Profiler or TensorFlow Profiler

### Interactive Shell Access

Enable web access for debugging:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=debug-job \
  --enable-web-access \
  --worker-pool-spec=machine-type=n1-standard-8,\
replica-count=1,\
container-image-uri=gcr.io/my-project/trainer:latest
```

From [Vertex AI Interactive Shell](https://docs.cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell) (accessed 2025-02-03):
> If you want to create a CustomJob, run the gcloud ai custom-jobs create command, and specify the --enable-web-access flag on this command.

## Advanced Features

### Hyperparameter Tuning

Create multiple trials with different hyperparameters:

```bash
gcloud ai hp-tuning-jobs create \
  --region=us-central1 \
  --display-name=hp-tuning \
  --config=hp_config.yaml
```

**hp_config.yaml:**
```yaml
studySpec:
  metrics:
  - metricId: accuracy
    goal: MAXIMIZE
  parameters:
  - parameterId: learning_rate
    doubleValueSpec:
      minValue: 0.0001
      maxValue: 0.1
    scaleType: UNIT_LOG_SCALE
  - parameterId: batch_size
    discreteValueSpec:
      values: [32, 64, 128, 256]
  algorithm: ALGORITHM_UNSPECIFIED  # Bayesian optimization
  maxTrialCount: 20
  parallelTrialCount: 4

trialJobSpec:
  workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/my-project/trainer:latest
```

### Using Reduction Server

For gradient synchronization at scale (8+ GPUs):

```yaml
workerPoolSpecs:
- machineSpec:
    machineType: a2-highgpu-8g
    acceleratorType: NVIDIA_TESLA_A100
    acceleratorCount: 8
  replicaCount: 4
  containerSpec:
    imageUri: gcr.io/my-project/trainer:latest
  reductionServerSpec:
    replicaCount: 2
    containerSpec:
      imageUri: gcr.io/deeplearning-platform-release/reduction-server:latest
```

From [Vertex AI Reduction Server](https://cloud.google.com/blog/products/ai-machine-learning/speed-up-your-model-training-with-vertex-ai) (accessed 2025-02-03):
> In this post, we'll show you how to speed up training of a PyTorch + Hugging Face model using Reduction Server, a Vertex AI feature that accelerates gradient synchronization.

### Private IP Training

Use VPC networks for security:

```yaml
jobSpec:
  network: projects/PROJECT_ID/global/networks/NETWORK_NAME
  reservedIpRanges:
  - "10.0.0.0/8"
  workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/my-project/trainer:latest
```

## Complete Example: Multi-Node PyTorch Training

**1. Training Script (train.py):**
```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from google.cloud import storage

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    # Load model
    model = YourModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Load data
    dataset = YourDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save model (only rank 0)
    if rank == 0:
        torch.save(model.state_dict(), '/tmp/model.pth')

        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket('my-bucket')
        blob = bucket.blob('models/model.pth')
        blob.upload_from_filename('/tmp/model.pth')

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

**2. Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

RUN pip install google-cloud-storage

COPY train.py /app/train.py
WORKDIR /app

ENTRYPOINT ["python", "train.py"]
```

**3. Build and Push:**
```bash
docker build -t gcr.io/my-project/pytorch-trainer:v1 .
docker push gcr.io/my-project/pytorch-trainer:v1
```

**4. Job Config (job.yaml):**
```yaml
displayName: "multi-node-pytorch"
jobSpec:
  workerPoolSpecs:
  # Primary replica
  - machineSpec:
      machineType: n1-highmem-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 4
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/my-project/pytorch-trainer:v1
  # Worker replicas
  - machineSpec:
      machineType: n1-highmem-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 4
    replicaCount: 3
    containerSpec:
      imageUri: gcr.io/my-project/pytorch-trainer:v1
```

**5. Submit Job:**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --config=job.yaml
```

## Sources

**Google Cloud Documentation:**
- [Create a Custom Training Job](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job) - Vertex AI core training guide (accessed 2025-02-03)
- [Distributed Training](https://docs.cloud.google.com/vertex-ai/docs/training/distributed-training) - Multi-node training architecture (accessed 2025-02-03)
- [Configure Compute Resources](https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute) - Worker pool specifications (accessed 2025-02-03)
- [Custom Containers Overview](https://docs.cloud.google.com/vertex-ai/docs/training/containers-overview) - Container requirements (accessed 2025-02-03)
- [Vertex AI Quotas](https://docs.cloud.google.com/vertex-ai/docs/quotas) - Resource limits and quotas (accessed 2025-02-03)
- [gcloud ai custom-jobs Reference](https://cloud.google.com/sdk/gcloud/reference/beta/ai/custom-jobs/create) - CLI command reference (accessed 2025-02-03)

**Additional References:**
- [PyTorch Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html) - PyTorch official guide
- [Speed Up Training with Reduction Server](https://cloud.google.com/blog/products/ai-machine-learning/speed-up-your-model-training-with-vertex-ai) - Google Cloud Blog (accessed 2025-02-03)
- [Efficient PyTorch Training with Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/efficient-pytorch-training-with-vertex-ai) - Google Cloud Blog (accessed 2025-02-03)
