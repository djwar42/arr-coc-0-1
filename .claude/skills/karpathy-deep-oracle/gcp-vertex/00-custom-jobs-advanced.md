# Vertex AI Custom Jobs Advanced: WorkerPoolSpec Architecture & Multi-Node Training

**Deep dive into advanced Vertex AI Custom Job configurations for production distributed training**

From [Vertex AI Distributed Training Documentation](https://cloud.google.com/vertex-ai/docs/training/distributed-training) (accessed 2025-11-16):
> "Vertex AI Training supports distributed training to reduce training time by running your training application on multiple VMs with GPUs or TPUs."

---

## Section 1: WorkerPoolSpec Architecture (~120 lines)

### 1.1 Worker Pool Fundamentals

**Vertex AI Custom Jobs** use a **worker pool** architecture where each job can define up to **4 worker pools** with different machine specifications and roles.

From [karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md](../karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md) (lines 13-42):
> "Worker Pool Roles:
> - Worker Pool 0: Chief worker (required)
> - Worker Pool 1: Additional workers
> - Worker Pool 2: Reduction Server (optional)
> - Worker Pool 3: Parameter servers (optional)"

**Worker Pool Structure:**

```python
from google.cloud import aiplatform

# Worker Pool 0: Chief + workers on same node
worker_pool_0 = {
    "machine_spec": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "gcr.io/project/training:latest",
        "command": ["python", "train.py"],
        "args": ["--distributed"]
    }
}

# Worker Pool 1: Additional nodes
worker_pool_1 = {
    "machine_spec": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8
    },
    "replica_count": 3,  # 3 additional nodes
    "container_spec": {
        "image_uri": "gcr.io/project/training:latest"
    }
}

# Total: 4 nodes × 8 A100 GPUs = 32 GPUs
```

### 1.2 Chief Worker vs Additional Workers

**Chief Worker (Worker Pool 0)**:
- Required for every Custom Job
- Coordinates distributed training
- Rank 0 in distributed setup
- Handles checkpoint saving
- Manages experiment logging

**Additional Workers (Worker Pool 1+)**:
- Optional for distributed training
- Ranks 1 to N-1
- Participate in gradient computation
- Receive parameters from chief

**Key Difference**: Chief worker has special responsibilities (checkpointing, logging) while additional workers focus purely on computation.

### 1.3 Parameter Servers (Optional)

For **asynchronous training** with parameter server architecture:

```python
# Worker Pool 2: Parameter servers
worker_pool_2 = {
    "machine_spec": {
        "machine_type": "n1-highmem-16",  # CPU-only
        "accelerator_count": 0
    },
    "replica_count": 2,  # 2 parameter servers
    "container_spec": {
        "image_uri": "gcr.io/project/ps-training:latest",
        "args": ["--job-name=ps"]  # TensorFlow parameter server role
    }
}
```

**When to Use Parameter Servers**:
- TensorFlow models with `tf.distribute.experimental.ParameterServerStrategy`
- Large embedding tables that don't fit on GPUs
- Asynchronous training (not recommended for modern LLMs)
- Legacy TensorFlow 1.x codebases

**Modern Alternative**: Use ZeRO-2/ZeRO-3 or FSDP instead of parameter servers for better efficiency.

From [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md) (lines 165-176):
> "ZeRO-2 Memory Breakdown (8 GPUs, 10B params):
> mem_zero2 = 2*params + (2*params/Nd) + (12*params/Nd)  # 37.5 GB per GPU"

### 1.4 Reduction Server Worker Pool

**Reduction Server** is a Google Cloud optimization for multi-node gradient aggregation.

From [karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md](../karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md) (lines 46-76):
> "Reduction Server replaces standard ring all-reduce with a hierarchical reduction pattern optimized for Google Cloud networking.
> Performance Gains:
> - 2× algorithm bandwidth for large models (>1B parameters)
> - 30-40% speedup on 4-8 node training jobs"

```python
# Worker Pool 2: Reduction Server (Google-managed)
worker_pool_2 = {
    "machine_spec": {
        "machine_type": "n1-highcpu-16"
    },
    "replica_count": 2,
    "container_spec": {
        "image_uri": "us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest"
    }
}
```

**Reduction Server Architecture**:
```
Workers (GPU nodes)
    ↓ AllReduce within node (NVLink: 600 GB/s)
Reduction Server Pool
    ↓ Cross-node aggregation (optimized routing)
Workers (GPU nodes)
    ↓ Broadcast aggregated gradients
```

**When to Use Reduction Server**:
- Multi-node training (2+ nodes)
- Models >1B parameters
- Standard NCCL all-reduce showing bottlenecks
- Google Cloud networking (non-InfiniBand)

---

## Section 2: Network Configuration (~140 lines)

### 2.1 VPC Network Basics

**Vertex AI Custom Jobs** can run in three network modes:

1. **Default Network** (not recommended for production)
2. **Custom VPC Network** (recommended)
3. **VPC Peering** (for private connectivity)

From [Vertex AI VPC Peering Documentation](https://cloud.google.com/vertex-ai/docs/general/vpc-peering) (accessed 2025-11-16):
> "VPC Network Peering allows you to set up a private connection to talk to your endpoint without your data ever traversing the public internet."

**Custom VPC Configuration**:

```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="distributed-training-vpc",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8
        },
        "replica_count": 4,
        "container_spec": {
            "image_uri": "gcr.io/project/training:latest"
        },
        # VPC configuration
        "network": "projects/PROJECT_ID/global/networks/custom-vpc",
        "reserved_ip_ranges": ["10.128.0.0/20"]
    }]
)
```

### 2.2 VPC Peering for Private Connectivity

**VPC Peering** establishes a private connection between your VPC and Google's managed VPC.

**Setup Steps**:

1. **Enable Private Services Access**:
```bash
# Reserve IP range for peering
gcloud compute addresses create vertex-ai-range \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=16 \
    --network=custom-vpc

# Create peering connection
gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=vertex-ai-range \
    --network=custom-vpc
```

2. **Configure Custom Job to Use Peering**:
```python
worker_pool_spec = {
    "machine_spec": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "gcr.io/project/training:latest"
    },
    # Enable VPC Peering
    "network": "projects/PROJECT_ID/global/networks/custom-vpc",
    "enable_web_access": False  # Force private connectivity
}
```

**IP Range Considerations**:
- Minimum `/16` range (65,536 IPs) recommended
- Avoid overlap with existing VPC subnets
- Cannot change range after peering is established
- Plan for future scale (more training jobs)

From [Vertex AI VPC Peering Documentation](https://cloud.google.com/vertex-ai/docs/general/vpc-peering) (accessed 2025-11-16):
> "Vertex AI recommends a /28 subnetwork. The subnet supports RFC 1918 and non RFC 1918 addresses with the exception of subnets 100.64.0.0/10."

### 2.3 Shared VPC Configuration

For **multi-project** organizations, **Shared VPC** allows centralized network management.

**Architecture**:
```
Host Project (network-project)
├── Shared VPC: custom-vpc
│   ├── Subnet: training-subnet (us-central1, 10.128.0.0/20)
│   └── Firewall: allow-internal
└── Service Projects
    ├── ml-training-dev (uses shared VPC)
    └── ml-training-prod (uses shared VPC)
```

**Configuration**:

```python
# In service project (ml-training-prod)
from google.cloud import aiplatform

aiplatform.init(
    project="ml-training-prod",
    location="us-central1"
)

job = aiplatform.CustomJob(
    display_name="shared-vpc-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8
        },
        "replica_count": 2,
        "container_spec": {
            "image_uri": "gcr.io/ml-training-prod/training:latest"
        },
        # Reference host project's shared VPC
        "network": "projects/network-project/global/networks/custom-vpc"
    }]
)
```

**IAM Requirements for Shared VPC**:
```bash
# Grant Vertex AI service agent access to shared VPC
gcloud projects add-iam-policy-binding network-project \
    --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/compute.networkUser"
```

### 2.4 Private Service Connect (PSC)

**Private Service Connect** provides private connectivity without VPC Peering.

From [Vertex AI PSC Documentation](https://cloud.google.com/vertex-ai/docs/general/vpc-psc-i-setup) (accessed 2025-11-16):
> "Private Service Connect interface allows you to access Vertex AI services through a private endpoint in your VPC network."

**PSC vs VPC Peering**:

| Feature | VPC Peering | Private Service Connect |
|---------|-------------|-------------------------|
| **IP Range Consumption** | Large (/16) | Small (/28) |
| **Routing Complexity** | Transitive routing issues | Isolated, no transitive routing |
| **Multi-Region** | Complex | Simple (per-region endpoints) |
| **Future-Proof** | Legacy approach | Modern Google Cloud pattern |

**PSC Setup**:

```bash
# Create Private Service Connect endpoint
gcloud compute addresses create vertex-ai-psc \
    --region=us-central1 \
    --subnet=training-subnet \
    --addresses=10.128.0.10

gcloud compute forwarding-rules create vertex-ai-psc-rule \
    --region=us-central1 \
    --network=custom-vpc \
    --address=vertex-ai-psc \
    --target-service-attachment=projects/PROJECT_ID/regions/us-central1/serviceAttachments/vertex-ai
```

---

## Section 3: Persistent Disk Attachment for Checkpointing (~100 lines)

### 3.1 Persistent Disk Fundamentals

**Problem**: Ephemeral boot disks lose checkpoints when jobs are interrupted.

**Solution**: Attach persistent disks to worker pools for durable checkpoint storage.

**Persistent Disk Configuration**:

```python
worker_pool_spec = {
    "machine_spec": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "gcr.io/project/training:latest"
    },
    # Attach persistent disk
    "disk_spec": {
        "boot_disk_type": "pd-ssd",        # Boot disk: 100 GB SSD
        "boot_disk_size_gb": 100,
        # Additional persistent disk for checkpoints
        "persistent_disk_size_gb": 1000,   # 1 TB for model checkpoints
        "persistent_disk_type": "pd-ssd"   # SSD for fast I/O
    }
}
```

**Persistent Disk Types**:

| Type | Throughput | IOPS | Use Case | Cost (per GB/month) |
|------|------------|------|----------|---------------------|
| **pd-standard** | 120 MB/s | 1,500 | Infrequent checkpoint | $0.040 |
| **pd-balanced** | 240 MB/s | 6,000 | Balanced performance | $0.100 |
| **pd-ssd** | 1,200 MB/s | 30,000 | Frequent checkpoint | $0.170 |
| **pd-extreme** | 2,400 MB/s | 120,000 | Ultra-low latency | $0.125 |

**Recommendation**: Use `pd-ssd` for training checkpoints (fast writes critical for large models).

### 3.2 Checkpoint Strategy with Persistent Disk

**Mount Point**: Persistent disk automatically mounted at `/mnt/disks/training_data`

**Training Code Integration**:

```python
import os
import torch
import deepspeed

# Checkpoint directory on persistent disk
checkpoint_dir = "/mnt/disks/training_data/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_with_persistent_checkpoints(model_engine, train_dataloader):
    """Training loop with persistent disk checkpointing"""

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model_engine(batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            # Save checkpoint every 500 steps to persistent disk
            if step % 500 == 0:
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}_step_{step}"
                model_engine.save_checkpoint(
                    save_dir=checkpoint_path,
                    tag=f"checkpoint_{step}"
                )
                print(f"Saved checkpoint to {checkpoint_path}")
```

### 3.3 Cross-Region Checkpoint Syncing

**Problem**: Persistent disk is region-specific. If training fails, resume in different region.

**Solution**: Sync checkpoints to **Cloud Storage** for cross-region durability.

```python
from google.cloud import storage

def sync_checkpoint_to_gcs(local_checkpoint_dir, gcs_bucket, gcs_prefix):
    """Sync persistent disk checkpoints to GCS"""
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)

    for root, dirs, files in os.walk(local_checkpoint_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_checkpoint_dir)
            gcs_path = f"{gcs_prefix}/{relative_path}"

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Synced {local_path} to gs://{gcs_bucket}/{gcs_path}")

# Usage in training loop
if step % 1000 == 0:  # Sync to GCS every 1000 steps
    sync_checkpoint_to_gcs(
        local_checkpoint_dir=checkpoint_dir,
        gcs_bucket="my-training-bucket",
        gcs_prefix=f"checkpoints/job_{job_id}"
    )
```

**Best Practice**:
- Frequent local saves to persistent disk (every 500 steps)
- Less frequent GCS syncs (every 1000-2000 steps)
- Final checkpoint always synced to GCS

---

## Section 4: Preemptible Worker Handling (~120 lines)

### 4.1 Preemptible VMs on Vertex AI

**Preemptible VMs** offer 60-80% cost savings but can be terminated at any time.

From [karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md](../karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md) (lines 849-885):
> "Preemptible Training Configuration:
> - useSpot: true
> - restartJobOnWorkerRestart: true (Auto-restart on preemption)
> - Cost Comparison:
>   - On-demand a2-highgpu-8g: $32.77/hour × 8 hours = $262.16
>   - Preemptible a2-highgpu-8g: $7.86/hour × 12 hours = $94.32
>   - Savings: 64%"

**Configuration**:

```yaml
displayName: preemptible-distributed-training
jobSpec:
  scheduling:
    restartJobOnWorkerRestart: true  # Auto-restart on preemption
    timeout: "86400s"  # 24 hours max

  workerPoolSpecs:
    - machineSpec:
        machineType: a2-highgpu-8g
        acceleratorType: NVIDIA_TESLA_A100
        acceleratorCount: 8
      replicaCount: 4
      containerSpec:
        imageUri: gcr.io/project/training:latest
      # Enable preemptible
      useSpot: true
```

### 4.2 Automatic Restart Strategies

**restartJobOnWorkerRestart Behavior**:

1. **Preemption Event**: Google Cloud terminates VM
2. **Vertex AI Detection**: Job status → `JOB_STATE_PENDING`
3. **Automatic Restart**: New VM allocated in ~2-5 minutes
4. **Checkpoint Resume**: Training continues from last saved checkpoint

**Training Code for Automatic Resume**:

```python
import os
import glob
import deepspeed

def find_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint in directory"""
    checkpoints = glob.glob(f"{checkpoint_dir}/epoch_*_step_*")
    if not checkpoints:
        return None, 0

    # Extract step number from checkpoint path
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("step_")[-1]))
    latest_step = int(latest_checkpoint.split("step_")[-1])

    return latest_checkpoint, latest_step

def train_with_preemption_recovery(model, optimizer, train_dataloader):
    """Training loop with automatic checkpoint resume"""

    checkpoint_dir = "/mnt/disks/training_data/checkpoints"

    # Check for existing checkpoint (resume after preemption)
    latest_checkpoint, resume_step = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config="ds_config.json"
        )

        # Load checkpoint
        _, client_sd = model_engine.load_checkpoint(
            load_dir=latest_checkpoint
        )
        start_step = resume_step
        print(f"Resumed training from step {start_step}")
    else:
        print("Starting training from scratch")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config="ds_config.json"
        )
        start_step = 0

    # Training loop
    for step in range(start_step, total_steps):
        batch = next(iter(train_dataloader))

        outputs = model_engine(batch)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        # Save checkpoint regularly
        if step % 500 == 0:
            model_engine.save_checkpoint(
                save_dir=f"{checkpoint_dir}/step_{step}",
                tag=f"checkpoint_{step}"
            )
```

### 4.3 Multi-Worker Preemption Handling

**Challenge**: In multi-node training, if one worker is preempted, the entire job fails.

**Solution**: Use **gang scheduling** with `restartJobOnWorkerRestart: true`

**Gang Scheduling Behavior**:
- All workers start together
- If any worker fails → all workers restart
- Ensures synchronized state across all workers
- Critical for distributed training (NCCL communication)

**DeepSpeed ZeRO-2 Configuration for Preemptible**:

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,

    "checkpoint": {
        "tag_validation": false,
        "use_node_local_storage": false,  # Use persistent disk, not local
        "parallel_write": {
            "save_opt_states": true
        }
    }
}
```

**Why ZeRO-2 for Preemptible**:
- Faster checkpoint save/load than ZeRO-3
- 40% memory reduction (sufficient for most models)
- Reduces preemption recovery time (critical for spot VMs)

From [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md) (lines 296-305):
> "Stage Comparison Table:
> - ZeRO-2: 95% throughput, 656 GB memory per GPU
> - ZeRO-3: 78% throughput, 350 GB memory per GPU"

---

## Section 5: Environment Variables for Distributed Training (~100 lines)

### 5.1 Vertex AI Automatic Environment Variables

**Vertex AI** automatically sets environment variables for distributed training:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `CLUSTER_SPEC` | JSON describing all workers | `{"cluster":{"worker":["10.128.0.2:2222","10.128.0.3:2222"]}}` |
| `TF_CONFIG` | TensorFlow-specific cluster config | `{"cluster":{"chief":["10.128.0.2:2222"]}}` |
| `WORLD_SIZE` | Total number of processes | `32` (4 nodes × 8 GPUs) |
| `RANK` | Process rank (0 to WORLD_SIZE-1) | `0` (chief), `1-31` (workers) |
| `LOCAL_RANK` | GPU index within node (0-7) | `0-7` |
| `MASTER_ADDR` | Chief worker IP address | `10.128.0.2` |
| `MASTER_PORT` | Chief worker port | `29500` |

**Reading Environment Variables in Python**:

```python
import os
import json

def get_distributed_config():
    """Parse Vertex AI distributed training environment variables"""

    # Process rank and world size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Master address for NCCL
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # TensorFlow cluster config (if using TF)
    tf_config = os.environ.get("TF_CONFIG")
    if tf_config:
        tf_config = json.loads(tf_config)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "master_addr": master_addr,
        "master_port": master_port,
        "tf_config": tf_config
    }

# Usage
config = get_distributed_config()
print(f"Rank {config['rank']}/{config['world_size']}")
print(f"Master: {config['master_addr']}:{config['master_port']}")
```

### 5.2 PyTorch Distributed Initialization

**Standard PyTorch DDP Setup**:

```python
import torch
import torch.distributed as dist

def init_distributed_pytorch():
    """Initialize PyTorch distributed backend using Vertex AI env vars"""

    # Vertex AI provides these automatically
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(local_rank)

    print(f"Initialized rank {rank}/{world_size} on GPU {local_rank}")
    return local_rank, rank, world_size
```

### 5.3 DeepSpeed Initialization

**DeepSpeed with Vertex AI**:

```python
import deepspeed

def init_distributed_deepspeed():
    """Initialize DeepSpeed using Vertex AI environment variables"""

    # DeepSpeed automatically reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    deepspeed.init_distributed(
        dist_backend="nccl",
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1))
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank
```

### 5.4 Custom Environment Variables

**Passing Custom Variables to Training Job**:

```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="custom-env-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8
        },
        "replica_count": 4,
        "container_spec": {
            "image_uri": "gcr.io/project/training:latest",
            # Custom environment variables
            "env": [
                {"name": "MODEL_NAME", "value": "llama-2-7b"},
                {"name": "DATASET_PATH", "value": "gs://bucket/data"},
                {"name": "LEARNING_RATE", "value": "3e-4"},
                {"name": "BATCH_SIZE", "value": "32"},
                {"name": "WANDB_API_KEY", "value": "secret-key"}
            ]
        }
    }]
)
```

**Reading Custom Variables in Training Code**:

```python
import os

# Read custom environment variables
model_name = os.environ["MODEL_NAME"]
dataset_path = os.environ["DATASET_PATH"]
learning_rate = float(os.environ["LEARNING_RATE"])
batch_size = int(os.environ["BATCH_SIZE"])

print(f"Training {model_name} with LR={learning_rate}, Batch={batch_size}")
```

---

## Section 6: arr-coc-0-1 Multi-Worker Training Example (~120 lines)

### 6.1 ARR-COC Architecture on Vertex AI

From [karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md](../karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md) (lines 731-801):
> "ARR-COC Model Components:
> - Qwen3-VL-2B base model: 4 GB (FP16)
> - Texture arrays: 13-channel texture (~3.3 MB per image)
> - Relevance scorers: 3 × 100M params (600 MB)
> - Quality adapter: 50M params (100 MB)
> Total per GPU without distribution: ~29 GB
> With ZeRO-2 on 8 GPUs: ~8 GB per GPU"

### 6.2 Multi-Worker Configuration for arr-coc-0-1

**Recommended Setup**: 4 nodes × 8 A100 (40GB) with ZeRO-2

**Custom Job Specification**:

```yaml
displayName: arr-coc-0-1-multi-worker-training
jobSpec:
  scheduling:
    restartJobOnWorkerRestart: true
    timeout: "172800s"  # 48 hours

  workerPoolSpecs:
    # Worker Pool 0: Chief + 7 workers (node 1)
    - machineSpec:
        machineType: a2-highgpu-8g
        acceleratorType: NVIDIA_TESLA_A100
        acceleratorCount: 8
      replicaCount: 1
      containerSpec:
        imageUri: gcr.io/arr-coc/training:latest
        args:
          - --deepspeed
          - --deepspeed_config=arr_coc_zero2.json
          - --data_path=gs://arr-coc-data/training
          - --output_dir=gs://arr-coc-models/checkpoints
        env:
          - name: NCCL_DEBUG
            value: "INFO"
          - name: NCCL_IB_DISABLE
            value: "0"
      diskSpec:
        bootDiskType: pd-ssd
        bootDiskSize_gb: 200
        persistentDiskSizeGb: 2000  # 2 TB for checkpoints
        persistentDiskType: pd-ssd

    # Worker Pool 1: Additional 3 nodes
    - machineSpec:
        machineType: a2-highgpu-8g
        acceleratorType: NVIDIA_TESLA_A100
        acceleratorCount: 8
      replicaCount: 3
      containerSpec:
        imageUri: gcr.io/arr-coc/training:latest
        args:
          - --deepspeed
          - --deepspeed_config=arr_coc_zero2.json
      diskSpec:
        bootDiskType: pd-ssd
        bootDiskSizeGb: 200
        persistentDiskSizeGb: 2000
        persistentDiskType: pd-ssd

    # Worker Pool 2: Reduction Server (optional but recommended)
    - machineSpec:
        machineType: n1-highcpu-16
      replicaCount: 2
      containerSpec:
        imageUri: us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest

# Total: 4 nodes × 8 GPUs = 32 A100 GPUs
```

### 6.3 DeepSpeed ZeRO-2 Configuration

**arr_coc_zero2.json**:

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    },

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": false,

    "checkpoint": {
        "tag_validation": false,
        "use_node_local_storage": false,
        "parallel_write": {
            "save_opt_states": true
        }
    }
}
```

### 6.4 Training Script for Multi-Worker arr-coc-0-1

**train.py**:

```python
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from arr_coc.texture import TextureArrayGenerator
from arr_coc.knowing import RelevanceScorers
from arr_coc.balancing import TensionBalancer
from arr_coc.attending import RelevanceAllocator

def init_distributed():
    """Initialize distributed training using Vertex AI env vars"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    deepspeed.init_distributed(
        dist_backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def build_arr_coc_model():
    """Build ARR-COC model with Qwen3-VL base"""
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-VL-2B-Instruct",
        torch_dtype=torch.float16
    )

    # Add ARR-COC components
    texture_gen = TextureArrayGenerator(num_channels=13)
    relevance_scorers = RelevanceScorers()
    tension_balancer = TensionBalancer()
    allocator = RelevanceAllocator()

    # Wrap in ARR-COC architecture
    model = ARRCOCModel(
        base_model=base_model,
        texture_gen=texture_gen,
        relevance_scorers=relevance_scorers,
        tension_balancer=tension_balancer,
        allocator=allocator
    )

    return model

def main():
    # Initialize distributed training
    local_rank, rank, world_size = init_distributed()

    # Build model
    model = build_arr_coc_model()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="arr_coc_zero2.json"
    )

    # Load dataset
    dataset = load_arr_coc_dataset("gs://arr-coc-data/training")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=train_sampler
    )

    # Training loop
    checkpoint_dir = "/mnt/disks/training_data/checkpoints"
    for epoch in range(num_epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU
            images = batch["images"].to(local_rank)
            queries = batch["queries"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            # Forward pass through ARR-COC pipeline
            outputs = model_engine(images, queries, labels=labels)
            loss = outputs.loss

            # Backward pass (DeepSpeed handles ZeRO-2 automatically)
            model_engine.backward(loss)
            model_engine.step()

            # Logging (rank 0 only)
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

            # Checkpoint saving (rank 0 only)
            if rank == 0 and step % 500 == 0:
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}_step_{step}"
                model_engine.save_checkpoint(
                    save_dir=checkpoint_path,
                    tag=f"checkpoint_{step}"
                )

                # Sync to GCS
                os.system(f"gsutil -m rsync -r {checkpoint_path} gs://arr-coc-models/checkpoints/epoch_{epoch}_step_{step}")

if __name__ == "__main__":
    main()
```

### 6.5 Launching arr-coc-0-1 Training Job

**Python SDK Launch**:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="arr-coc-project",
    location="us-central1",
    staging_bucket="gs://arr-coc-training"
)

job = aiplatform.CustomJob.from_local_script(
    display_name="arr-coc-0-1-4-node-training",
    script_path="train.py",
    container_uri="gcr.io/arr-coc/training:latest",
    requirements=["deepspeed==0.14.0", "transformers==4.36.0"],
    replica_count=4,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=8,
    boot_disk_type="pd-ssd",
    boot_disk_size_gb=200
)

job.run(
    sync=False,
    service_account="vertex-training@arr-coc-project.iam.gserviceaccount.com"
)

print(f"Job submitted: {job.resource_name}")
print(f"View logs: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
```

---

## Sources

**Vertex AI Official Documentation:**
- [Distributed Training on Vertex AI](https://cloud.google.com/vertex-ai/docs/training/distributed-training) - Google Cloud Docs (accessed 2025-11-16)
- [CustomJobSpec API Reference](https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/CustomJobSpec) - Google Cloud Docs (accessed 2025-11-16)
- [VPC Network Peering for Vertex AI](https://cloud.google.com/vertex-ai/docs/general/vpc-peering) - Google Cloud Docs (accessed 2025-11-16)
- [Private Service Connect for Vertex AI](https://cloud.google.com/vertex-ai/docs/general/vpc-psc-i-setup) - Google Cloud Docs (accessed 2025-11-16)

**Community Resources:**
- [Vertex AI Multi-Worker Training Codelab](https://codelabs.developers.google.com/vertex_multiworker_training) - Google Codelabs (accessed 2025-11-16)
- [Private Service Connect Interface Vertex AI Pipelines](https://codelabs.developers.google.com/psc-interface-pipelines) - Google Codelabs (accessed 2025-11-16)

**Cross-References to Existing Knowledge:**
- [karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md](../karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md) - Lines 13-42 (WorkerPool architecture), 46-76 (Reduction Server), 731-801 (arr-coc-0-1 configuration), 849-885 (Preemptible training)
- [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md) - Lines 165-176 (ZeRO-2 memory), 296-305 (Stage comparison)
- [karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md](../karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md) - Pipeline parallelism patterns
- [karpathy/distributed-training/03-fsdp-vs-deepspeed.md](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - FSDP vs DeepSpeed framework comparison

**Web Research:**
- Search: "Vertex AI Custom Jobs WorkerPoolSpec 2024" (accessed 2025-11-16)
- Search: "Vertex AI multi-worker distributed training" (accessed 2025-11-16)
- Search: "Vertex AI preemptible workers checkpoint resume" (accessed 2025-11-16)
- Search: "Vertex AI VPC network configuration" (accessed 2025-11-16)

---

**Knowledge file complete**: 701 lines
**Created**: 2025-11-16
**Purpose**: Advanced Vertex AI Custom Jobs configuration covering WorkerPoolSpec, VPC networking, persistent disk checkpointing, preemptible workers, environment variables, and arr-coc-0-1 multi-worker example
