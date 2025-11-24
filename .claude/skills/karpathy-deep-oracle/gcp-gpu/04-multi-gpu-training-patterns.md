# Multi-GPU Single-Node Training Patterns on GCP

## Overview

Single-node multi-GPU training is the most common distributed training pattern for modern deep learning. On GCP, A100 and H100 instances provide up to 8 GPUs connected via high-bandwidth NVLink, enabling efficient data parallelism with minimal communication overhead. This guide covers PyTorch DistributedDataParallel (DDP), NCCL optimization, GPU topology awareness, and GCP-specific configuration for maximum training throughput.

**Why Single-Node Multi-GPU Matters:**
- **Cost efficiency**: 8×A100 node cheaper than 8 separate instances
- **NVLink bandwidth**: 600 GB/s (A100) vs 32 GB/s (PCIe), 18× faster GPU-to-GPU communication
- **Reduced latency**: Sub-microsecond GPU communication vs millisecond network latency
- **Simpler orchestration**: No distributed coordination across nodes required
- **GCP integration**: Compact Placement Policies ensure optimal GPU topology

**Performance Impact:**
From [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html) (accessed 2025-11-16):
> "DDP can achieve near-linear scaling efficiency on single-node multi-GPU setups, with 8 GPUs providing 7.5-7.8× speedup over single-GPU training when properly configured."

---

## Section 1: PyTorch DistributedDataParallel (DDP) Fundamentals (100 lines)

### What is DDP?

**DistributedDataParallel** (DDP) is PyTorch's recommended multi-GPU training module. Unlike the older DataParallel, DDP uses **one process per GPU** with **peer-to-peer communication** via NCCL, eliminating the single-GPU bottleneck.

**DDP vs DataParallel:**

| Feature | DataParallel (DP) | DistributedDataParallel (DDP) |
|---------|-------------------|-------------------------------|
| **Process model** | Single process, multi-thread | Multi-process (1 per GPU) |
| **Communication** | CPU scatter/gather via GPU:0 | Direct GPU-to-GPU via NCCL |
| **GPU:0 memory** | Higher (gathers all gradients) | Equal across all GPUs |
| **Scalability** | Poor beyond 4 GPUs | Excellent (scales to 1000s) |
| **Multi-node** | Not supported | Native support |
| **Overhead** | High (CPU bottleneck) | Low (GPU peer-to-peer) |

From [Some PyTorch multi-GPU training tips](https://cerfacs.fr/coop/pytorch-multi-gpu) (Cerfacs COOP Blog, accessed 2025-11-16):
> "DataParallel is not recommended for the following reasons: Communication overhead due to waiting for all GPUs to finish backpropagation, gathering gradients, and broadcasting updated parameters. The memory usage on the master GPU is higher than on other GPUs."

### DDP Architecture

**Process Group:**
- Each GPU runs an independent Python process (rank 0, 1, 2, ..., 7)
- Processes communicate via **NCCL collective operations** (AllReduce)
- Master process (rank 0) handles checkpoint saving, logging

**Gradient Synchronization:**
1. **Forward pass**: Each GPU processes its local batch independently
2. **Backward pass**: Gradients computed locally
3. **AllReduce**: NCCL synchronizes gradients across all GPUs (overlapped with backward pass)
4. **Optimizer step**: Each GPU updates its model replica with synchronized gradients

**Key Insight:**
From [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html):
> "DDP overlaps gradient communication with backward computation. As soon as a gradient for a layer is computed, DDP initiates an AllReduce for that gradient, allowing computation and communication to happen in parallel."

### Basic DDP Setup

**Minimal DDP Example:**

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def ddp_setup(rank: int, world_size: int):
    """Initialize distributed process group.

    Args:
        rank: Unique identifier (0-7 for 8 GPUs)
        world_size: Total number of processes (8 for 8 GPUs)
    """
    import os
    os.environ["MASTER_ADDR"] = "localhost"  # Single-node: localhost
    os.environ["MASTER_PORT"] = "12355"      # Any free port

    # Set device BEFORE init_process_group to avoid GPU:0 congestion
    torch.cuda.set_device(rank)

    # Initialize NCCL process group
    dist.init_process_group(
        backend="nccl",        # NVIDIA GPUs: always use NCCL
        rank=rank,
        world_size=world_size
    )

def train(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Create model and move to GPU
    model = MyModel().to(device)

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Setup data loading with DistributedSampler
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # CRITICAL: Ensures proper shuffling

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  # DDP automatically synchronizes gradients
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 8 for A100-80GB-8
    mp.spawn(train, args=(world_size,), nprocs=world_size)
```

**Critical Details:**

1. **`torch.cuda.set_device(rank)`**: Must be called BEFORE `init_process_group` to prevent GPU:0 memory congestion
2. **`DistributedSampler`**: Ensures each GPU receives unique mini-batches
3. **`sampler.set_epoch(epoch)`**: Required for proper shuffling across epochs
4. **`device_ids=[rank]`**: Tells DDP which GPU this process owns

---

## Section 2: NCCL Optimization for Single-Node Training (120 lines)

### NCCL (NVIDIA Collective Communication Library)

**What is NCCL?**
NCCL provides optimized multi-GPU and multi-node communication primitives for NVIDIA GPUs. It automatically detects GPU topology (PCIe, NVLink, NVSwitch) and selects optimal communication algorithms.

**NCCL Operations:**
- **AllReduce**: Sum gradients across all GPUs, return result to all
- **Broadcast**: Send model from rank 0 to all other ranks
- **AllGather**: Gather tensors from all GPUs to all GPUs
- **ReduceScatter**: Sum and distribute chunks to different GPUs

From [NCCL Official Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html):
> "NCCL automatically uses NVLink when available, achieving up to 600 GB/s bidirectional bandwidth on A100 8-GPU systems, compared to 32 GB/s over PCIe."

### Single-Node NCCL Configuration

**Essential Environment Variables:**

```bash
# Single-node multi-GPU: Disable InfiniBand (not used within node)
export NCCL_IB_DISABLE=1

# Force peer-to-peer access via NVLink
export NCCL_P2P_LEVEL=NVL  # NVL = NVLink, PHB = PCIe

# Use local sockets for single-node (not network interface)
export NCCL_SOCKET_IFNAME=lo

# Debug: Print NCCL topology detection
export NCCL_DEBUG=INFO  # Set to WARN in production
```

**GCP-Specific Configuration:**

On GCP A100/H100 instances, NCCL automatically detects NVLink topology. No manual tuning required for optimal performance.

```bash
# Verify NCCL detects NVLink (check logs):
# NCCL INFO Using network Google 0:gve0
# NCCL INFO Channel 00/08 :    0   1   2   3   4   5   6   7
# NCCL INFO Trees [0] -1/-1/-1->0->1 [1] -1/-1/-1->0->1 ...
# NCCL INFO comm 0x... rank 0 nranks 8 cudaDev 0 busId a00000 - Init COMPLETE
```

**Optimal NCCL Settings for GCP A100-8:**

```python
import os

def configure_nccl_single_node():
    """Optimal NCCL configuration for GCP single-node 8-GPU training."""
    # Disable InfiniBand (single-node doesn't use it)
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Force NVLink peer-to-peer
    os.environ["NCCL_P2P_LEVEL"] = "NVL"

    # Use localhost for single-node coordination
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    # Optimize for single-node NVSwitch topology
    os.environ["NCCL_ALGO"] = "Tree"  # Tree algorithm for NVSwitch

    # Enable GPU Direct RDMA (if supported by GCP instance)
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"

    # CRITICAL: Set before torch.distributed.init_process_group()
```

### NCCL Performance Monitoring

**Verify NCCL is Using NVLink:**

```python
import torch.distributed as dist

def check_nccl_backend():
    """Verify NCCL configuration after process group initialization."""
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        print(f"Backend: {backend}")  # Should be 'nccl'

        # Check NCCL version
        if hasattr(torch.cuda.nccl, "version"):
            version = torch.cuda.nccl.version()
            print(f"NCCL version: {version}")  # 2.18+ recommended for A100
```

**Benchmark AllReduce Bandwidth:**

```python
import torch
import torch.distributed as dist
import time

def benchmark_allreduce(rank, world_size, tensor_size_mb=100):
    """Benchmark NCCL AllReduce bandwidth."""
    device = torch.device(f"cuda:{rank}")

    # Create tensor (100 MB default)
    num_elements = tensor_size_mb * 1024 * 1024 // 4  # 4 bytes per float32
    tensor = torch.randn(num_elements, device=device)

    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()

    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate bandwidth
    data_size = tensor_size_mb * num_iters  # MB transferred
    bandwidth = data_size / elapsed  # MB/s

    if rank == 0:
        print(f"AllReduce bandwidth: {bandwidth:.2f} MB/s")
        print(f"Expected for A100 NVLink: ~75,000 MB/s (600 Gbps)")

    return bandwidth
```

**Expected Performance:**
- **A100 NVLink**: 600 GB/s bidirectional (75,000 MB/s per direction)
- **PCIe Gen4**: 32 GB/s bidirectional (4,000 MB/s per direction)
- **Ratio**: NVLink is **18.75× faster** than PCIe

---

## Section 3: GPU Topology and NVLink Architecture (120 lines)

### Understanding GPU Topology

**Query Topology with nvidia-smi:**

```bash
# Show GPU topology (NVLink connections)
nvidia-smi topo -m

# Example output for A100-8 with NVSwitch:
#       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
# GPU0   X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
# GPU1  NV12   X    NV12  NV12  NV12  NV12  NV12  NV12
# GPU2  NV12  NV12   X    NV12  NV12  NV12  NV12  NV12
# ...
#
# Legend:
#   X    = Self
#   NV#  = NVLink (# = number of links, 12 = full bandwidth)
#   PHB  = PCIe Host Bridge
```

**Key Observations:**
- **NV12**: All GPUs connected via 12 NVLink lanes (full 600 GB/s bandwidth)
- **Symmetric topology**: Each GPU has equal bandwidth to all others
- **NVSwitch**: Enables all-to-all connectivity without PCIe bottleneck

### A100 8-GPU NVLink Topology

**Architecture:**

```
        NVSwitch (600 GB/s per GPU)
            ╱  │  │  │  │  │  │  ╲
          ╱    │  │  │  │  │  │    ╲
        ╱      │  │  │  │  │  │      ╲
    GPU0    GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
     │       │    │    │    │    │    │     │
    HBM2e  HBM2e HBM2e HBM2e ...
    80GB   80GB  80GB  80GB
```

**Specifications:**
- **NVLink 3.0**: 25 GB/s per direction per link
- **12 NVLink lanes per GPU**: 12 × 25 GB/s = 300 GB/s per direction
- **Bidirectional**: 600 GB/s total per GPU
- **NVSwitch**: Non-blocking switch connecting all 8 GPUs

From [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) (accessed 2025-11-16):
> "A single A100 NVLink provides 25 GB/s bandwidth in each direction. The A100 HGX includes eight A100 GPUs connected by the new NVLink-enabled NVSwitch, delivering 600 GB/s bidirectional bandwidth per GPU."

### CPU-GPU NUMA Affinity

**NUMA (Non-Uniform Memory Access):**
On multi-socket CPU systems, memory access latency varies based on which CPU socket owns the memory. For optimal performance, pin GPU processes to CPU cores on the **same NUMA node** as the GPU.

**Check NUMA Topology:**

```bash
# Show CPU-GPU NUMA affinity
nvidia-smi topo -m

# Example output:
#       GPU0  GPU1  ...  CPU0  CPU1
# GPU0   X    NV12       NODE  SYS
# GPU1  NV12   X         NODE  SYS
#
# NODE = Same NUMA node (optimal)
# SYS  = Different NUMA node (higher latency)
```

**Set CPU Affinity:**

```python
import os
import torch

def set_cpu_affinity(rank: int, world_size: int):
    """Pin process to CPUs on same NUMA node as GPU.

    For GCP A100-8 instances, GPUs 0-3 on NUMA node 0, GPUs 4-7 on NUMA node 1.
    """
    # Get CPU list for this GPU's NUMA node
    gpu_device = f"/sys/class/pci_bus/0000:0{rank}/device/numa_node"

    try:
        with open(gpu_device, 'r') as f:
            numa_node = int(f.read().strip())
    except:
        # Fallback: assume GPUs 0-3 on node 0, 4-7 on node 1
        numa_node = 0 if rank < 4 else 1

    # Get CPU list for NUMA node
    cpu_list_file = f"/sys/devices/system/node/node{numa_node}/cpulist"
    with open(cpu_list_file, 'r') as f:
        cpu_list = f.read().strip()

    # Set affinity using taskset
    pid = os.getpid()
    os.system(f"taskset -p -c {cpu_list} {pid}")

    print(f"Rank {rank}: Pinned to NUMA node {numa_node}, CPUs {cpu_list}")
```

From [How does NUMA affinity affect GPU performance](https://massedcompute.com/faq-answers/?question=How%20does%20NUMA%20affinity%20affect%20GPU%20performance%20in%20a%20multi-socket%20system?) (Massed Compute, accessed 2025-11-16):
> "CPU threads should be pinned to cores on the same NUMA node as the GPU to reduce synchronization overhead. Mismatched NUMA affinity can reduce GPU performance by 10-30% due to increased memory access latency."

### GCP Compact Placement Policies

**Compact Placement** ensures VMs are co-located on the same physical hardware, minimizing network latency. For single-node multi-GPU, this is automatic within the VM.

**Multi-node training** (covered in next section) should use Compact Placement:

```bash
# Create compact placement policy
gcloud compute resource-policies create group-placement my-gpu-policy \
    --region=us-central1 \
    --collocation=collocated

# Create instance group with placement policy
gcloud compute instance-groups managed create my-gpu-group \
    --size=4 \
    --template=a100-template \
    --resource-policies=my-gpu-policy \
    --region=us-central1
```

---

## Section 4: Efficient Data Loading for Multi-GPU (100 lines)

### DistributedSampler

**Purpose:** Partition dataset across GPUs so each GPU processes unique samples.

**Basic Usage:**

```python
from torch.utils.data import DataLoader, DistributedSampler

def create_dataloader(dataset, rank, world_size, batch_size=32):
    """Create distributed dataloader for DDP."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # Total number of GPUs
        rank=rank,                # This GPU's rank
        shuffle=True,             # Shuffle within each GPU's partition
        seed=42                   # Ensure same shuffling across GPUs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,   # Per-GPU batch size
        sampler=sampler,          # Use DistributedSampler
        num_workers=4,            # CPU workers for data loading
        pin_memory=True,          # Faster GPU transfer
        prefetch_factor=2,        # Prefetch batches
        persistent_workers=True   # Keep workers alive between epochs
    )

    return dataloader, sampler
```

**Critical: Set Epoch for Proper Shuffling:**

```python
# In training loop
for epoch in range(num_epochs):
    # MUST call set_epoch for shuffling to work correctly
    sampler.set_epoch(epoch)

    for batch in dataloader:
        # Training code
        pass
```

From [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html):
> "Calling the set_epoch() method on the DistributedSampler at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be used in each epoch."

### Optimizing DataLoader Performance

**Key Parameters:**

| Parameter | Recommended Value | Impact |
|-----------|------------------|--------|
| `num_workers` | 4-8 per GPU | Parallel data loading |
| `pin_memory` | True | Faster CPU→GPU transfer |
| `prefetch_factor` | 2-4 | Prefetch batches ahead |
| `persistent_workers` | True | Avoid worker respawn overhead |

**Benchmark Data Loading:**

```python
import time

def benchmark_dataloader(dataloader, num_batches=100):
    """Measure data loading throughput."""
    device = torch.device("cuda:0")

    start = time.time()
    for i, (data, target) in enumerate(dataloader):
        if i >= num_batches:
            break
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

    elapsed = time.time() - start
    throughput = num_batches / elapsed

    print(f"Data loading throughput: {throughput:.2f} batches/sec")
    return throughput
```

**Expected Performance:**
- **Good**: >100 batches/sec (data loading not bottleneck)
- **Bad**: <50 batches/sec (increase `num_workers`)

### Effective Batch Size Scaling

**Global Batch Size:**
With `world_size` GPUs, each with local batch size `B`, the **effective batch size** is `B × world_size`.

**Learning Rate Scaling:**
Linear scaling rule (from [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)):

```python
# Scale learning rate with batch size
base_lr = 1e-4
base_batch_size = 32
world_size = 8

effective_batch_size = base_batch_size * world_size  # 256
scaled_lr = base_lr * (effective_batch_size / base_batch_size)  # 8e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)
```

**Warmup for Large Batch Training:**

```python
def linear_warmup(current_step, warmup_steps, base_lr, target_lr):
    """Linear warmup from base_lr to target_lr."""
    if current_step < warmup_steps:
        return base_lr + (target_lr - base_lr) * (current_step / warmup_steps)
    return target_lr

# Use with scheduler
from torch.optim.lr_scheduler import LambdaLR

warmup_steps = 1000
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: linear_warmup(step, warmup_steps, 0.0, 1.0)
)
```

---

## Section 5: Gradient Accumulation and Mixed Precision (100 lines)

### Gradient Accumulation

**Purpose:** Simulate larger batch sizes when GPU memory is limited.

**Implementation:**

```python
accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)

    # Forward pass
    output = model(data)
    loss = criterion(output, target)

    # Normalize loss by accumulation steps
    loss = loss / accumulation_steps

    # Backward pass (accumulate gradients)
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**With DDP:**
DDP synchronizes gradients **during backward pass**, so accumulation works naturally:

```python
# No special handling needed for DDP + gradient accumulation
# Gradients are synchronized at each backward() call
# Optimizer step happens every accumulation_steps as normal
```

### Automatic Mixed Precision (AMP)

**AMP Benefits:**
- **Memory**: 50% reduction (FP16 vs FP32)
- **Speed**: 2-3× faster on Tensor Cores (A100/H100)
- **Accuracy**: GradScaler prevents underflow

**PyTorch AMP Usage:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass in FP16
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale gradients and step optimizer
        scaler.step(optimizer)
        scaler.update()
```

**AMP + Gradient Accumulation:**

```python
accumulation_steps = 4
scaler = GradScaler()

for i, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)

    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        # Gradient clipping (optional)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### BF16 vs FP16

**On A100/H100, prefer BF16 (bfloat16):**

```python
# BF16 has wider dynamic range, no gradient scaling needed
for data, target in dataloader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    # Forward in BF16
    with autocast(dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()  # No scaler needed!
    optimizer.step()
```

**BF16 vs FP16:**
- **BF16**: Same exponent range as FP32, no overflow/underflow issues, no GradScaler
- **FP16**: Narrower range, requires GradScaler, slightly faster on older GPUs
- **Recommendation**: Use BF16 on A100/H100, FP16 on V100/T4

---

## Section 6: Monitoring and Profiling Multi-GPU Training (80 lines)

### GPU Utilization Monitoring

**nvidia-smi Dashboard:**

```bash
# Real-time GPU monitoring (1-second refresh)
watch -n 1 nvidia-smi

# Monitor specific metrics
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 1
```

**Expected Values for Efficient Training:**
- **GPU utilization**: >90% (if lower, data loading bottleneck)
- **Memory usage**: 70-95% (if <70%, can increase batch size)
- **Temperature**: <85°C (A100 max operating temp)

### NCCL Communication Monitoring

**Enable NCCL Profiling:**

```bash
export NCCL_DEBUG=INFO  # Print NCCL operations
export NCCL_DEBUG_SUBSYS=ALL  # Detailed subsystem info

# Run training
python train.py

# Check logs for:
# - Bandwidth achieved per AllReduce
# - Algorithm used (Ring, Tree, CollNet)
# - Detected topology (NVLink, PCIe)
```

### PyTorch Profiler

**Profile Training Loop:**

```python
from torch.profiler import profile, ProfilerActivity, schedule

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_rank_{rank}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=trace_handler,
    record_shapes=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(dataloader):
        # Training code
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prof.step()  # Signal profiler to advance
```

**Analyze with TensorBoard:**

```bash
tensorboard --logdir=./traces
# Open http://localhost:6006 and view trace
```

---

## Section 7: Checkpointing and Model Saving in DDP (80 lines)

### Save Only from Rank 0

**Critical:** Only rank 0 should save checkpoints to avoid corruption.

```python
def save_checkpoint(model, optimizer, epoch, rank):
    """Save checkpoint from rank 0 only."""
    if rank != 0:
        return  # Other ranks skip saving

    # Extract model from DDP wrapper
    model_state = model.module.state_dict()  # .module is the actual model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    print(f"Rank {rank}: Saved checkpoint for epoch {epoch}")
```

**Load Checkpoint:**

```python
def load_checkpoint(model, optimizer, checkpoint_path, rank):
    """Load checkpoint on all ranks."""
    # Map to correct device
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')

    # Load model state
    model.module.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']

    if rank == 0:
        print(f"Loaded checkpoint from epoch {epoch}")

    return epoch
```

### Collective Calls Warning

**Critical:** Avoid collective calls inside rank-specific code.

```python
# ❌ WRONG: Collective call inside rank check causes deadlock
if rank == 0:
    dist.barrier()  # Only rank 0 calls barrier → DEADLOCK!
    save_checkpoint(...)

# ✅ CORRECT: Collective call before rank check
dist.barrier()  # All ranks call barrier
if rank == 0:
    save_checkpoint(...)
```

---

## Section 8: GCP-Specific Configuration for arr-coc-0-1 (100 lines)

### Instance Selection

**Recommended GCP Instance for Single-Node 8-GPU Training:**

| Instance Type | GPUs | GPU Memory | NVLink | vCPUs | RAM | Cost (us-central1, on-demand) |
|---------------|------|------------|--------|-------|-----|-------------------------------|
| **a2-highgpu-8g** | 8×A100 40GB | 320 GB | 600 GB/s | 96 | 680 GB | ~$35/hour |
| **a2-ultragpu-8g** | 8×A100 80GB | 640 GB | 600 GB/s | 96 | 1360 GB | ~$50/hour |
| **a3-highgpu-8g** | 8×H100 80GB | 640 GB | 900 GB/s | 208 | 1872 GB | ~$60/hour |

**For arr-coc-0-1 Vision-Language Training:**
- **Recommended**: `a2-ultragpu-8g` (8×A100 80GB)
- **Rationale**: Vision transformers require large activations, 80GB per GPU prevents OOM
- **Alternative**: `a2-highgpu-8g` if model fits in 40GB

### Startup Script for DDP Training

**GCP Metadata Startup Script:**

```bash
#!/bin/bash

# Install NVIDIA drivers (if not in image)
/opt/deeplearning/install-driver.sh

# Install CUDA 12.1 + cuDNN 8.9
apt-get update
apt-get install -y cuda-12-1 libcudnn8=8.9.7.*-1+cuda12.1

# Set NCCL environment
cat >> /etc/environment <<EOF
NCCL_IB_DISABLE=1
NCCL_P2P_LEVEL=NVL
NCCL_SOCKET_IFNAME=lo
NCCL_DEBUG=WARN
EOF

# Install PyTorch with CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Clone training repo
cd /workspace
git clone https://github.com/your-org/arr-coc-0-1.git
cd arr-coc-0-1

# Run 8-GPU training
python -m torch.distributed.launch --nproc_per_node=8 train.py \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --fp16 \
    --local_rank $LOCAL_RANK
```

### arr-coc-0-1 Training Configuration

**Optimized Hyperparameters for 8×A100:**

```python
# config.py
class TrainingConfig:
    # Hardware
    num_gpus = 8

    # Data loading
    per_gpu_batch_size = 16       # 16 × 8 = 128 effective batch size
    num_workers = 8                # 8 workers per GPU
    prefetch_factor = 4

    # Optimization
    base_lr = 1e-4
    scaled_lr = base_lr * num_gpus  # 8e-4 for 8 GPUs
    warmup_steps = 1000
    max_steps = 100000

    # Memory optimization
    gradient_accumulation_steps = 4  # 128 × 4 = 512 effective batch
    fp16 = True                      # Or bf16 on A100
    gradient_checkpointing = True    # Trade compute for memory

    # Checkpointing
    save_every = 1000
    checkpoint_dir = "/workspace/checkpoints"
```

**Launch Script:**

```python
# train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(rank, world_size):
    # Setup
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Model
    model = ARRCOCModel(config).to(rank)
    model = DDP(model, device_ids=[rank])

    # Data
    train_dataset = ARRCOCDataset(split="train")
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_gpu_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.scaled_lr)

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    for step in range(config.max_steps):
        for i, batch in enumerate(train_loader):
            images, queries, targets = batch
            images = images.to(rank)
            queries = queries.to(rank)
            targets = targets.to(rank)

            with autocast():
                outputs = model(images, queries)
                loss = criterion(outputs, targets) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if rank == 0 and step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.item()}")

        # Checkpoint
        if rank == 0 and step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, rank)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
```

---

## Sources

**Source Documents:**
- [cuda/00-streams-concurrency-async.md](../cuda/00-streams-concurrency-async.md) - GPU concurrency patterns

**Web Research:**
- [Multi GPU training with DDP - PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html) (accessed 2025-11-16)
- [Some PyTorch multi-GPU training tips - Cerfacs COOP Blog](https://cerfacs.fr/coop/pytorch-multi-gpu) (accessed 2025-11-16)
- [How does NUMA affinity affect GPU performance - Massed Compute](https://massedcompute.com/faq-answers/?question=How%20does%20NUMA%20affinity%20affect%20GPU%20performance%20in%20a%20multi-socket%20system?) (accessed 2025-11-16)
- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) (accessed 2025-11-16)

**Additional References:**
- Google Search: "PyTorch DistributedDataParallel single node multi-GPU 2024" (accessed 2025-11-16)
- Google Search: "NCCL optimization single-node multi-GPU PyTorch 2024" (accessed 2025-11-16)
- Google Search: "GPU affinity CPU pinning NUMA GCP multi-GPU training" (accessed 2025-11-16)
- Google Search: "NVLink bandwidth A100 8-GPU topology 2024" (accessed 2025-11-16)
