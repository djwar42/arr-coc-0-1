# Distributed Training Patterns for Production on Vertex AI

**Knowledge File: Production-grade distributed training architectures, fault tolerance, and optimization strategies**

---

## Overview

Distributed training is essential for training large-scale models that exceed single-GPU capacity or require faster iteration. This guide covers production patterns for PyTorch DDP, Horovod, parameter servers, and fault-tolerant training on Vertex AI.

**Core distributed training paradigms:**
- **Data Parallelism**: Split batch across GPUs, synchronize gradients (PyTorch DDP, Horovod)
- **Model Parallelism**: Split model layers across GPUs (pipeline parallelism)
- **Tensor Parallelism**: Split individual tensors across GPUs (for massive models)
- **Parameter Server**: Async gradient updates via centralized parameter storage

**When to use distributed training:**
- Model doesn't fit on single GPU (>40GB for A100, >80GB for A100-80GB)
- Training time too long (days → hours via multi-GPU)
- Batch size limited by GPU memory (larger batches improve convergence)
- Production deployment requires rapid iteration cycles

From [Vertex AI Fundamentals](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md):
> "Vertex AI supports distributed training via multiple worker pools with chief/worker/parameter server roles, enabling data parallel and model parallel strategies."

---

## Section 1: PyTorch DistributedDataParallel (DDP)

### 1.1 DDP Architecture and Communication

**How DDP Works:**
```
┌─────────────────────────────────────────────────────────┐
│ DDP Training Flow (4 GPUs)                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  GPU 0 (Rank 0)    GPU 1 (Rank 1)    GPU 2 (Rank 2)    GPU 3 (Rank 3)
│     │                  │                  │                  │
│     ▼                  ▼                  ▼                  ▼
│  Model Copy       Model Copy       Model Copy       Model Copy
│  (identical)      (identical)      (identical)      (identical)
│     │                  │                  │                  │
│     ▼                  ▼                  ▼                  ▼
│  Batch[0:8]       Batch[8:16]      Batch[16:24]     Batch[24:32]
│  (micro-batch)    (micro-batch)    (micro-batch)    (micro-batch)
│     │                  │                  │                  │
│     ▼ Forward          ▼ Forward          ▼ Forward          ▼ Forward
│  Loss 0            Loss 1            Loss 2            Loss 3
│     │                  │                  │                  │
│     ▼ Backward         ▼ Backward         ▼ Backward         ▼ Backward
│  Grad 0            Grad 1            Grad 2            Grad 3
│     │                  │                  │                  │
│     └──────────────────┴──────────────────┴──────────────────┘
│                          │
│                          ▼
│                   All-Reduce (Ring or Tree)
│                   Average gradients across all GPUs
│                          │
│     ┌──────────────────┬─┴──────────────┬──────────────────┐
│     ▼                  ▼                 ▼                  ▼
│  Update Params     Update Params    Update Params    Update Params
│  (synchronized)    (synchronized)   (synchronized)   (synchronized)
└─────────────────────────────────────────────────────────────┘
```

**Key DDP Features:**
- **Gradient bucketing**: Overlaps backward pass with gradient communication
- **Ring-AllReduce**: Efficient gradient synchronization (O(N) not O(N²))
- **Process groups**: Supports NCCL (GPU), Gloo (CPU), MPI backends
- **No parameter server**: Fully decentralized, each GPU stores full model

**Single-Node Multi-GPU DDP (Vertex AI):**
```python
# train_ddp.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_ddp():
    # Vertex AI sets these environment variables automatically
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize process group (NCCL for GPU)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # Uses MASTER_ADDR and MASTER_PORT env vars
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train():
    rank, world_size, local_rank = setup_ddp()

    # Create model and move to GPU
    model = MyVisionModel().to(local_rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create distributed sampler (ensures no data duplication)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # Set epoch for sampler (ensures different shuffle each epoch)
        train_sampler.set_epoch(epoch)

        model.train()
        for batch in train_loader:
            images, labels = batch
            images = images.to(local_rank)
            labels = labels.to(local_rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward triggers gradient all-reduce automatically
            loss.backward()

            optimizer.step()

        # Only rank 0 saves checkpoints
        if rank == 0:
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    train()
```

**Vertex AI Custom Job Config (4 GPUs on single node):**
```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="ddp-training-4gpu",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-4g",  # 4x A100 GPUs
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 4
        },
        "replica_count": 1,  # Single node
        "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/my-project/ml/trainer:v1",
            "command": [
                "torchrun",
                "--nproc_per_node=4",  # 4 GPUs
                "--nnodes=1",           # 1 node
                "train_ddp.py"
            ]
        }
    }]
)

job.run(sync=True)
```

### 1.2 Multi-Node DDP (Distributed Across Machines)

**Multi-Node Architecture:**
```
Node 0 (Chief)                     Node 1 (Worker)
┌──────────────────┐              ┌──────────────────┐
│ GPU 0 (Rank 0)   │              │ GPU 0 (Rank 2)   │
│ GPU 1 (Rank 1)   │◄────NCCL────►│ GPU 1 (Rank 3)   │
└──────────────────┘              └──────────────────┘
   MASTER_ADDR:MASTER_PORT
```

**Multi-Node Training Script:**
```python
# train_multinode.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_multinode():
    # Vertex AI sets these automatically for multi-node jobs
    rank = int(os.environ["RANK"])              # Global rank (0-3 for 2 nodes × 2 GPUs)
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank within node (0-1)
    world_size = int(os.environ["WORLD_SIZE"])  # Total GPUs (4)

    master_addr = os.environ["MASTER_ADDR"]     # Chief node address
    master_port = os.environ["MASTER_PORT"]     # Communication port

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def train():
    rank, local_rank, world_size = setup_multinode()

    if rank == 0:
        print(f"Training on {world_size} GPUs across multiple nodes")

    # Rest identical to single-node DDP
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Training loop...
```

**Vertex AI Multi-Node Config:**
```python
job = aiplatform.CustomJob(
    display_name="ddp-multinode-4gpu",
    worker_pool_specs=[
        {  # Chief worker
            "machine_spec": {
                "machine_type": "a2-highgpu-2g",  # 2x A100
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 2
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/my-project/ml/trainer:v1",
                "command": [
                    "torchrun",
                    "--nproc_per_node=2",  # 2 GPUs per node
                    "--nnodes=2",           # 2 nodes total
                    "--node_rank=0",        # Chief node
                    "--master_addr=$MASTER_ADDR",
                    "--master_port=29500",
                    "train_multinode.py"
                ]
            }
        },
        {  # Worker nodes
            "machine_spec": {
                "machine_type": "a2-highgpu-2g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 2
            },
            "replica_count": 1,  # 1 additional node
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/my-project/ml/trainer:v1",
                "command": [
                    "torchrun",
                    "--nproc_per_node=2",
                    "--nnodes=2",
                    "--node_rank=1",  # Worker rank
                    "--master_addr=$MASTER_ADDR",
                    "--master_port=29500",
                    "train_multinode.py"
                ]
            }
        }
    ]
)
```

### 1.3 DDP Performance Optimization

**Gradient Bucketing:**
```python
# Optimize DDP gradient communication
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    bucket_cap_mb=25,  # Default 25MB, increase for larger models
    find_unused_parameters=False,  # Set True if using conditional branches
    gradient_as_bucket_view=True   # Saves memory by avoiding gradient copy
)
```

**Mixed Precision with DDP:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    images, labels = batch
    images, labels = images.to(local_rank), labels.to(local_rank)

    optimizer.zero_grad()

    # Forward in mixed precision
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    # Scaled backward (prevents underflow)
    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step with gradient scaling
    scaler.step(optimizer)
    scaler.update()
```

**Communication Backend Selection:**
```python
# GPU training: Use NCCL (fastest for NVIDIA GPUs)
dist.init_process_group(backend="nccl")

# CPU training: Use Gloo
dist.init_process_group(backend="gloo")

# MPI-based clusters: Use MPI backend
dist.init_process_group(backend="mpi")
```

**Monitoring DDP Performance:**
```python
import time

# Measure gradient all-reduce time
if rank == 0:
    start_time = time.time()

loss.backward()  # Triggers all-reduce

if rank == 0:
    allreduce_time = time.time() - start_time
    print(f"All-reduce time: {allreduce_time:.3f}s")

    # Log to W&B
    wandb.log({"allreduce_time": allreduce_time})
```

From [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-01-31):
> "DDP overlaps gradient computation with communication by bucketing gradients and launching all-reduce operations as soon as a bucket is ready, reducing overall training time."

---

## Section 2: Horovod for Multi-Framework Distribution

### 2.1 Horovod Architecture

**Horovod Overview:**
- Developed by Uber for unified distributed training
- Supports TensorFlow, PyTorch, Apache MXNet, Keras
- Uses MPI for process management
- Implements ring-allreduce algorithm (bandwidth-optimal)
- Simpler API than native framework distribution

**Horovod vs PyTorch DDP:**

| Feature | Horovod | PyTorch DDP |
|---------|---------|-------------|
| Framework support | Multi-framework | PyTorch only |
| Backend | MPI, NCCL, Gloo | NCCL, Gloo, MPI |
| Gradient sync | Ring-allreduce | Ring or tree-allreduce |
| API simplicity | Simpler (fewer changes) | More verbose |
| Performance | Slightly slower (~5%) | Fastest for PyTorch |
| Fault tolerance | Via elastic Horovod | Via torchrun |

**When to use Horovod:**
- Multi-framework workflows (TensorFlow + PyTorch)
- Existing MPI infrastructure
- Need for gradient compression (Horovod built-in)
- Easier migration from single-GPU code

### 2.2 Horovod with PyTorch on Vertex AI

**Installation:**
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    cmake \
    g++

RUN pip install torch torchvision horovod[pytorch]

# Build Horovod with NCCL support
RUN HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod[pytorch]
```

**Training Script:**
```python
# train_horovod.py
import torch
import horovod.torch as hvd

def train():
    # Initialize Horovod
    hvd.init()

    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())

    # Create model
    model = MyModel().cuda()

    # Scale learning rate by number of workers
    base_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr * hvd.size())

    # Wrap optimizer with Horovod DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=hvd.Compression.fp16  # Optional: compress gradients
    )

    # Broadcast model state from rank 0 to all workers
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Create data sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Horovod handles gradient all-reduce automatically
            optimizer.step()

        # Only rank 0 saves checkpoints
        if hvd.rank() == 0:
            torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")

if __name__ == "__main__":
    train()
```

**Vertex AI Config with Horovod:**
```python
job = aiplatform.CustomJob(
    display_name="horovod-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-4g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 4
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/my-project/ml/horovod:v1",
            "command": [
                "horovodrun",
                "-np", "4",  # Number of processes (4 GPUs)
                "--mpi-args=--bind-to none",
                "python", "train_horovod.py"
            ]
        }
    }]
)
```

### 2.3 Horovod Elastic Training (Fault Tolerance)

**Elastic Horovod for Dynamic Scaling:**
```python
# train_elastic.py
import horovod.torch as hvd
from horovod.torch.elastic import run as elastic_run
from horovod.torch.elastic import State

def train_fn(state):
    # Initialize Horovod within elastic context
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Restore or create model
    if state.model is None:
        state.model = MyModel().cuda()
        state.optimizer = torch.optim.Adam(state.model.parameters())
    else:
        # Restore from checkpoint after worker failure/addition
        state.model = state.model.cuda()

    # Wrap optimizer
    state.optimizer = hvd.DistributedOptimizer(state.optimizer)

    # Broadcast state from rank 0
    hvd.broadcast_parameters(state.model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(state.optimizer, root_rank=0)

    # Training loop
    for epoch in range(state.epoch, num_epochs):
        state.epoch = epoch

        for batch in train_loader:
            # Training step...
            pass

        # Commit state for checkpointing
        state.commit()

if __name__ == "__main__":
    state = State(model=None, optimizer=None, epoch=0)

    # Run elastic training (automatically handles worker failures)
    elastic_run(
        train_fn,
        state,
        min_np=2,  # Minimum 2 workers
        max_np=8   # Maximum 8 workers (can scale up dynamically)
    )
```

From [Horovod GitHub](https://github.com/horovod/horovod) (accessed 2025-01-31):
> "Horovod's elastic training enables jobs to continue running if workers are added or removed, making it ideal for preemptible instance training where nodes may be reclaimed."

---

## Section 3: Parameter Server Architecture

### 3.1 Parameter Server Concepts

**Parameter Server vs Data Parallel:**

**Data Parallel (DDP/Horovod):**
- Each worker stores full model copy
- Synchronous gradient updates (all workers wait for slowest)
- All-reduce for gradient aggregation
- Bandwidth: O(N) with ring-allreduce

**Parameter Server:**
- Workers don't store parameters, only compute gradients
- Asynchronous updates (workers don't wait)
- Parameters stored on dedicated parameter server nodes
- Bandwidth: O(1) per worker (sends gradients to server)

**When to use Parameter Server:**
- Extremely large models (>100B parameters)
- Heterogeneous hardware (different GPU speeds)
- High worker failure rate (async = more fault-tolerant)
- Network-constrained environments

**Parameter Server Architecture:**
```
┌────────────────────────────────────────────────┐
│ Parameter Server Topology                      │
├────────────────────────────────────────────────┤
│                                                 │
│          Parameter Servers (PS)                │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│    │  PS 0    │  │  PS 1    │  │  PS 2    │  │
│    │ Params   │  │ Params   │  │ Params   │  │
│    │ [0:33%]  │  │ [33:66%] │  │ [66:100%]│  │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│         │             │             │          │
│    ┌────┴─────────────┴─────────────┴────┐   │
│    │        Push Gradients                │   │
│    │        Pull Parameters               │   │
│    └────┬─────────────┬─────────────┬────┘   │
│         │             │             │          │
│    ┌────▼─────┐  ┌───▼──────┐  ┌───▼──────┐ │
│    │ Worker 0 │  │ Worker 1 │  │ Worker 2 │ │
│    │ Compute  │  │ Compute  │  │ Compute  │ │
│    │ Gradients│  │ Gradients│  │ Gradients│ │
│    └──────────┘  └──────────┘  └──────────┘ │
└────────────────────────────────────────────────┘
```

### 3.2 PyTorch Parameter Server (torch.distributed.rpc)

**Parameter Server Implementation:**
```python
# ps_train.py
import torch
import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
from torch import optim

class ParameterServer:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    @staticmethod
    @rpc.functions.async_execution
    def get_parameters(self):
        # Workers pull parameters
        return [param.data for param in self.model.parameters()]

    @staticmethod
    @rpc.functions.async_execution
    def push_gradients(self, gradients):
        # Workers push gradients
        with self.lock:
            for param, grad in zip(self.model.parameters(), gradients):
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad += grad

    def apply_gradients(self, optimizer):
        with self.lock:
            optimizer.step()
            optimizer.zero_grad()

def run_parameter_server(rank, world_size):
    # Initialize RPC
    rpc.init_rpc(
        name=f"ps_{rank}",
        rank=rank,
        world_size=world_size
    )

    # Create model on parameter server
    model = MyModel()
    ps = ParameterServer(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Keep server alive
    rpc.shutdown()

def run_worker(rank, world_size, ps_rank):
    rpc.init_rpc(
        name=f"worker_{rank}",
        rank=rank,
        world_size=world_size
    )

    ps_rref = rpc.remote(f"ps_{ps_rank}", ParameterServer)

    for epoch in range(num_epochs):
        # Pull parameters from server
        params = rpc.rpc_sync(ps_rref.owner(), ParameterServer.get_parameters, args=(ps_rref,))

        # Compute gradients locally
        model = MyModel()
        for local_param, ps_param in zip(model.parameters(), params):
            local_param.data = ps_param

        for batch in data_loader:
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()

        # Push gradients to server
        gradients = [param.grad.clone() for param in model.parameters()]
        rpc.rpc_async(ps_rref.owner(), ParameterServer.push_gradients, args=(ps_rref, gradients))

    rpc.shutdown()
```

**Vertex AI Config with Parameter Server:**
```python
job = aiplatform.CustomJob(
    display_name="ps-training",
    worker_pool_specs=[
        {  # Parameter servers
            "machine_spec": {"machine_type": "n1-highmem-8"},
            "replica_count": 2,  # 2 parameter servers
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/my-project/ml/ps:v1",
                "command": ["python", "ps_train.py", "--role=ps"]
            }
        },
        {  # Workers
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1
            },
            "replica_count": 4,  # 4 worker GPUs
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/my-project/ml/ps:v1",
                "command": ["python", "ps_train.py", "--role=worker"]
            }
        }
    ]
)
```

### 3.3 TensorFlow Parameter Server on Vertex AI

**TensorFlow Native PS Support:**
```python
# tf_ps_train.py
import tensorflow as tf

# Define cluster spec
cluster_spec = {
    "chief": ["10.0.0.1:2222"],
    "worker": ["10.0.0.2:2222", "10.0.0.3:2222"],
    "ps": ["10.0.0.4:2222", "10.0.0.5:2222"]
}

# Create strategy
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec)
)

# Define model
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training automatically uses parameter servers
model.fit(train_dataset, epochs=10)
```

From [TensorFlow Parameter Server](https://www.tensorflow.org/tutorials/distribute/parameter_server_training) (accessed 2025-01-31):
> "Parameter server training distributes model parameters across multiple machines, enabling training of models too large to fit on a single worker."

---

## Section 4: Fault Tolerance and Checkpointing

### 4.1 Fault Tolerance Strategies

**Common Failure Modes:**
1. **GPU OOM**: Batch size too large, model too big
2. **Preemption**: Spot VMs reclaimed (60-91% discount but interruptible)
3. **Network partition**: Multi-node communication failure
4. **Hardware failure**: GPU/node crash
5. **Straggler nodes**: One slow GPU blocks all workers (synchronous training)

**Fault Tolerance Techniques:**

**1. Checkpointing:**
```python
import os
from google.cloud import storage

def save_checkpoint(model, optimizer, epoch, rank=0):
    if rank != 0:
        return  # Only rank 0 saves

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    # Save locally first
    local_path = f"/tmp/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, local_path)

    # Upload to GCS for durability
    storage_client = storage.Client()
    bucket = storage_client.bucket("my-checkpoints")
    blob = bucket.blob(f"job-123/checkpoint_epoch_{epoch}.pth")
    blob.upload_from_filename(local_path)

    print(f"Checkpoint saved to gs://my-checkpoints/job-123/checkpoint_epoch_{epoch}.pth")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
```

**2. Automatic Restart with torchrun:**
```bash
# Dockerfile ENTRYPOINT
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --max_restarts=3 \  # Restart up to 3 times on failure
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py
```

**3. Gradient Checkpointing (Memory Optimization):**
```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 10)

    def forward(self, x):
        # Use gradient checkpointing for layer1 and layer2
        # Saves memory by not storing activations (recomputes on backward)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

### 4.2 Vertex AI Spot VMs with Fault Tolerance

**Using Spot VMs for 70% Cost Reduction:**
```python
job = aiplatform.CustomJob(
    display_name="fault-tolerant-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-4g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 4
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/my-project/ml/trainer:v1",
            "command": ["torchrun", "--max_restarts=5", "train.py"]
        },
        "spot": True  # Use Spot VMs (preemptible)
    }]
)
```

**Checkpoint Strategy for Spot VMs:**
```python
import signal
import sys

def handle_preemption(signum, frame):
    print("Received preemption signal, saving checkpoint...")
    save_checkpoint(model, optimizer, current_epoch)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, handle_preemption)

# Training loop with frequent checkpointing
for epoch in range(start_epoch, num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Training step...

        # Checkpoint every 100 batches
        if batch_idx % 100 == 0:
            save_checkpoint(model, optimizer, epoch)
```

### 4.3 Elastic Training with Vertex AI

**Elastic Training (Dynamic Worker Scaling):**
```python
# elastic_train.py with torchrun
import torch.distributed as dist

def train():
    # torchrun automatically handles worker failures/additions
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Worker {rank}/{world_size} started")

    # Load checkpoint if exists
    checkpoint_path = "gs://my-bucket/latest_checkpoint.pth"
    if gcs_file_exists(checkpoint_path):
        epoch = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        epoch = 0

    for epoch in range(epoch, num_epochs):
        # Training loop...

        # Save checkpoint every epoch
        if rank == 0:
            save_checkpoint(model, optimizer, epoch)

    dist.destroy_process_group()
```

From [PyTorch Fault Tolerance](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-01-31):
> "torchrun provides fault-tolerant distributed training by automatically restarting failed workers and re-initializing the process group with remaining healthy workers."

---

## Section 5: Production Best Practices

### 5.1 Monitoring and Observability

**Key Metrics to Track:**
```python
import wandb
import torch.distributed as dist

def log_training_metrics(rank, loss, throughput, gpu_memory):
    if rank == 0:  # Only rank 0 logs
        wandb.log({
            "loss": loss,
            "throughput_samples_per_sec": throughput,
            "gpu_memory_gb": gpu_memory,
            "world_size": dist.get_world_size()
        })

# Measure throughput
start_time = time.time()
samples_processed = 0

for batch in train_loader:
    # Training step...
    samples_processed += len(batch)

throughput = samples_processed / (time.time() - start_time)

# Monitor GPU memory
gpu_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB

log_training_metrics(rank, loss.item(), throughput, gpu_memory)
```

**Cloud Logging Integration:**
```python
from google.cloud import logging as cloud_logging

client = cloud_logging.Client()
logger = client.logger("vertex-training")

def log_to_cloud(message, severity="INFO"):
    logger.log_text(message, severity=severity)

# Log important events
log_to_cloud(f"Training started on {world_size} GPUs")
log_to_cloud(f"Checkpoint saved at epoch {epoch}")
log_to_cloud(f"Training completed successfully", severity="NOTICE")
```

### 5.2 Hyperparameter Tuning with Distributed Training

**Vertex AI Hyperparameter Tuning:**
```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Define hyperparameter search space
param_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
    "batch_size": hpt.DiscreteParameterSpec(values=[16, 32, 64, 128]),
    "num_layers": hpt.IntegerParameterSpec(min=4, max=12, scale="linear")
}

# Each trial uses distributed training
custom_job = aiplatform.CustomJob(
    display_name="ddp-hptuning-trial",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-4g",
            "accelerator_count": 4
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/my-project/ml/trainer:v1",
            "command": ["torchrun", "--nproc_per_node=4", "train.py"],
            "args": [
                "--learning_rate={{learning_rate}}",
                "--batch_size={{batch_size}}",
                "--num_layers={{num_layers}}"
            ]
        }
    }]
)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="vit-hptuning",
    custom_job=custom_job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec=param_spec,
    max_trial_count=50,
    parallel_trial_count=5  # Run 5 distributed trials simultaneously
)

hp_job.run(sync=True)
```

### 5.3 Cost Optimization Strategies

**1. Mixed Instance Training:**
```python
# Chief on on-demand, workers on Spot VMs
worker_pool_specs = [
    {  # Chief (on-demand for reliability)
        "machine_spec": {"machine_type": "a2-highgpu-1g"},
        "replica_count": 1,
        "spot": False
    },
    {  # Workers (Spot for cost savings)
        "machine_spec": {"machine_type": "a2-highgpu-1g"},
        "replica_count": 3,
        "spot": True  # 70% cheaper
    }
]
```

**2. Gradient Accumulation (Reduce GPU Count):**
```python
# Simulate large batch size with gradient accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps * world_size

for batch_idx, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. Efficient Data Loading:**
```python
# Use local SSD for fast data access
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=8,  # 8 data loading workers
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reuse workers across epochs
    prefetch_factor=4  # Prefetch 4 batches per worker
)
```

From [Vertex AI Production Patterns](../karpathy/practical-implementation/35-vertex-ai-production-patterns.md):
> "Use Spot VMs for workers with automatic checkpointing and restart. Keep parameter servers or chief nodes on-demand for stability while achieving 60-70% cost savings."

---

## Key Takeaways

**Distributed Training Strategy Selection:**

| Scenario | Recommended Approach | Reasoning |
|----------|---------------------|-----------|
| Single node, <8 GPUs | PyTorch DDP | Simplest, fastest for PyTorch |
| Multi-node, synchronous | PyTorch DDP or Horovod | DDP faster, Horovod multi-framework |
| Massive models (>100B) | Parameter Server + Tensor Parallel | Only way to fit model in memory |
| Fault-tolerant training | Elastic Horovod or torchrun | Handles worker failures gracefully |
| Cost-optimized | DDP with Spot VMs + checkpointing | 70% cost savings with fault tolerance |

**Production Checklist:**
- ✅ Implement frequent checkpointing (every 100-500 steps)
- ✅ Use Spot VMs for workers, on-demand for chief/PS
- ✅ Monitor GPU utilization, throughput, all-reduce time
- ✅ Enable gradient checkpointing for large models
- ✅ Use mixed precision (FP16/BF16) for 2-3× speedup
- ✅ Test fault recovery (simulate preemption)
- ✅ Integrate with W&B/TensorBoard for experiment tracking

**Common Pitfalls:**
- ❌ Not scaling learning rate with world size (use `lr * world_size`)
- ❌ Synchronous training with heterogeneous GPUs (use async PS instead)
- ❌ Saving checkpoints on all ranks (only rank 0 should save)
- ❌ Not setting `DistributedSampler.set_epoch()` (leads to duplicate data)
- ❌ Using default batch size (should be per-GPU, not global)

---

## Sources

**Official Documentation:**
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) (accessed 2025-01-31)
- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) (accessed 2025-01-31)
- [PyTorch Fault Tolerance](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-01-31)
- [Vertex AI Distributed Training](https://docs.cloud.google.com/vertex-ai/docs/training/distributed-training) (accessed 2025-01-31)

**Horovod:**
- [Horovod GitHub Repository](https://github.com/horovod/horovod) (accessed 2025-01-31)
- [Horovod with PyTorch](https://horovod.readthedocs.io/en/stable/pytorch.html) (accessed 2025-01-31)

**Best Practices:**
- [Distributed Training on Kubernetes](https://collabnix.com/distributed-training-on-kubernetes-best-practices-implementation/) - Collabnix (accessed 2025-01-31)
- [PyTorch Multi-Node Training Guide](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide) - Lambda Labs (accessed 2025-01-31)
- [Scaling PyTorch DDP on Azure](https://medium.com/data-science-at-microsoft/scaling-model-training-with-pytorch-distributed-data-parallel-ddp-on-azure-machine-learning-d512a932ca13) - Microsoft (accessed 2025-01-31)

**Internal References:**
- [Vertex AI Fundamentals](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md) - Custom Jobs, WorkerPoolSpecs, Machine Types
- [Vertex AI Production Patterns](../karpathy/practical-implementation/35-vertex-ai-production-patterns.md) - Spot VMs, Checkpointing, Cost Optimization
