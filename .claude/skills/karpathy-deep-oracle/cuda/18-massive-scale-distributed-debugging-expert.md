# CUDA Massive-Scale Distributed Debugging (100s-1000s GPUs) - Expert Guide

**Expert-level guide to debugging distributed training at massive scale: network topology, NCCL optimization, straggler detection, fault tolerance, and production observability**

**Target audience**: ML engineers scaling training to 100+ GPU clusters, infrastructure teams managing multi-node deployments, researchers debugging hyperscale training failures

---

## Overview

Debugging distributed training becomes fundamentally different at massive scale (100s-1000s of GPUs). Issues that are rare on 8 GPUs become common at scale: network bottlenecks dominate, single GPU failures halt entire jobs, stragglers cause severe performance degradation, and debugging visibility becomes challenging.

This guide covers expert-level debugging strategies for:
- **Network topology debugging** (InfiniBand, RoCE, NVLink fabrics, NCCL topology analysis)
- **Straggler detection and mitigation** (slow GPUs, network bottlenecks, load imbalancing)
- **Fault tolerance at scale** (elastic training, checkpoint-restart, communicator shrinking)
- **Production observability** (distributed tracing, centralized logging, anomaly detection)

From [Google Cloud - Stragglers in AI](https://cloud.google.com/blog/products/compute/stragglers-in-ai-a-guide-to-automated-straggler-detection) (accessed 2025-11-13):
> "Slow nodes, or 'stragglers,' can hurt AI model training performance. In distributed computation, the running time of a single distributed task is governed by that of the slowest node."

From [NVIDIA NCCL 2.27 Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) (accessed 2025-11-13):
> "NCCL 2.27 introduces Communicator Shrink, a feature designed to make distributed training more robust, flexible, and efficient. Training jobs running across hundreds or thousands of GPUs are susceptible to device failures. Communicator Shrink enables dynamic exclusion of failed or unnecessary GPUs during training."

---

## Section 1: Network Topology Debugging & NCCL at Scale (~150 lines)

### 1.1 Understanding Network Topologies at Scale

**Common network topologies for massive-scale training:**

```
Fat-Tree Topology (Most Common for 100-1000 GPUs):
┌─────────────────────────────────────────────────────────┐
│ Core Switches (High Radix)                              │
├─────────────────────────────────────────────────────────┤
│   Aggregation Switches                                  │
├─────────────────────────────────────────────────────────┤
│ ToR Switches (Top of Rack)                              │
├───────────┬───────────┬───────────┬───────────┬─────────┤
│ Node 0-7  │ Node 8-15 │ Node16-23 │ Node24-31 │  ...    │
│ (8 GPUs)  │ (8 GPUs)  │ (8 GPUs)  │ (8 GPUs)  │         │
└───────────┴───────────┴───────────┴───────────┴─────────┘

Rail-Optimized Topology (For 1000+ GPUs):
- Multiple independent network planes (rails)
- Each GPU connects to multiple rails for redundancy
- NCCL can stripe communication across rails
- Reduces congestion on single fabric
```

**Intra-node vs inter-node communication:**
- **Intra-node**: NVLink (900 GB/s on NVL72, 600 GB/s on NVL8)
- **Inter-node**: InfiniBand (800 Gb/s NDR, 400 Gb/s HDR) or RoCE (400 Gb/s)
- **Bottleneck**: Inter-node bandwidth is 10-100× slower than intra-node

### 1.2 NCCL Topology Detection and Analysis

**NCCL automatically detects GPU and network topology:**

```bash
# Enable NCCL topology dump
export NCCL_TOPO_DUMP_FILE=nccl_topo.xml

# Run training - NCCL will save topology to XML
python train.py

# Analyze topology
cat nccl_topo.xml
```

**Example NCCL topology output:**
```xml
<system version="1">
  <cpu numaid="0" affinity="0000ffff">
    <gpu dev="0" sm="90" rank="0" gdr="1">
      <nvlink target="1" count="18" />
      <nvlink target="2" count="18" />
      <!-- NVLink connections within node -->
    </gpu>
    <net name="mlx5_0" port="1" guid="..." speed="400000" maxconn="1"/>
    <!-- InfiniBand NIC connections -->
  </cpu>
</system>
```

**Verify GPU topology with nvidia-smi:**
```bash
# Show GPU topology matrix (NVLink connections)
nvidia-smi topo -m

# Example output for DGX H100 (8 GPUs):
#         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
# GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
# GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
# ...
# NV18 = 18 NVLink connections (900 GB/s bidirectional)
```

### 1.3 NCCL Environment Variables for Scale

**Critical NCCL tuning for 100+ GPU clusters:**

```bash
# Network interface selection (critical for multi-NIC nodes)
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Multiple InfiniBand adapters

# Topology awareness
export NCCL_TOPO_FILE=/path/to/custom_topo.xml  # Override auto-detection
export NCCL_NET_GDR_LEVEL=5  # GPUDirect RDMA level (0-5, higher = more aggressive)

# Communication algorithms for scale
export NCCL_ALGO=Tree  # Tree algorithm for large clusters (vs Ring)
export NCCL_PROTO=Simple  # Protocol: Simple, LL (Low Latency), or LL128

# Timeout handling for large clusters
export NCCL_TIMEOUT=3600  # 1 hour timeout (default 600s too short for 1000 GPUs)

# SHARP support (InfiniBand in-network aggregation)
export NCCL_SHARP_ENABLE=1  # Offload AllReduce to network switches
export NCCL_COLLNET_ENABLE=1  # Enable collective network operations

# Multi-rail support (for rail-optimized topologies)
export NCCL_CROSS_NIC=2  # Enable cross-NIC communication
export NCCL_IB_ADAPTIVE_ROUTING=1  # Adaptive routing for congestion avoidance

# Debugging at scale
export NCCL_DEBUG=INFO  # Basic info
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET  # Subsystem-specific logging
```

From [NVIDIA NCCL 2.27 Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) (accessed 2025-11-13):
> "NCCL 2.27 adds support for SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) for both NVLink and InfiniBand fabrics. Traditional ring-based implementations can consume 16 or more SMs, but with NVLink and IB SHARP, this demand is reduced to 6 SMs or fewer, freeing up resources for model computation."

### 1.4 Network Bandwidth Testing at Scale

**Benchmark network performance before training:**

```bash
# Clone NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# Single-node all-reduce test (8 GPUs)
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# Multi-node all-reduce test (128 GPUs across 16 nodes)
mpirun -np 128 \
  --hostfile hostfile.txt \
  --map-by ppr:8:node \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=mlx5_0,mlx5_1 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# Expected output:
#       size         count    type   redop     time   algbw   busbw
#        8MB            1   float     sum    0.234   34.19   64.11  # Good
#      256MB            1   float     sum    2.145  119.37  223.82  # Good
#        8GB            1   float     sum   68.234  117.24  219.82  # Good
```

**Interpreting NCCL test results:**
- **algbw (algorithm bandwidth)**: Actual data throughput
- **busbw (bus bandwidth)**: Effective bandwidth accounting for communication pattern
- **Good performance**: 80-90% of theoretical NIC bandwidth
- **Poor performance**: <50% of theoretical bandwidth (indicates topology/config issues)

### 1.5 Diagnosing Network Bottlenecks

**Symptoms of network bottlenecks at scale:**
- GPU utilization drops during gradient synchronization
- Training throughput doesn't scale linearly with GPU count
- Large variance in step time across iterations

**Diagnosis workflow:**

```bash
# 1. Check for network congestion (run on each node)
ibstat  # InfiniBand status
# Look for: Link state, Rate (should be 400 Gb/s or 800 Gb/s)

# 2. Monitor network bandwidth during training
ib_write_bw  # InfiniBand bandwidth test between nodes
# Run between pairs of nodes to identify slow links

# 3. Profile gradient synchronization time
# Use NCCL_DEBUG=INFO and grep for timing
export NCCL_DEBUG=INFO
python train.py 2>&1 | grep "coll"

# Example output:
# NCCL INFO AllReduce:  128MB  time 45.2ms  algbw 2.83 GB/s  busbw 5.31 GB/s
# NCCL INFO AllReduce: 1024MB  time 180.5ms algbw 5.67 GB/s  busbw 10.63 GB/s

# 4. Compare intra-node vs inter-node bandwidth
# Intra-node should be 10-100x faster than inter-node
```

**Common network bottleneck causes:**
- **Misconfigured RDMA**: Check `NCCL_NET_GDR_LEVEL` and kernel module (`nvidia_peermem`)
- **Single-rail congestion**: Enable multi-rail with `NCCL_CROSS_NIC`
- **Switch oversubscription**: Fat-tree topology with insufficient core switch bandwidth
- **NIC misconfiguration**: Wrong MTU size, PCIe link degradation

### 1.6 InfiniBand-Specific Debugging

**InfiniBand health checks:**

```bash
# Check IB link status
ibstatus
# Output should show: State: Active, Physical state: LinkUp

# Check for errors on IB ports
ibdiagnet -r  # Run InfiniBand diagnostics
# Look for: link errors, congestion, bad cables

# Test RDMA performance
ib_send_bw  # Between two nodes
ib_read_bw  # RDMA read bandwidth

# Check RDMA kernel module
lsmod | grep nvidia_peermem  # Should show nvidia_peermem loaded
```

**InfiniBand subnet manager issues:**
- Large clusters (1000+ GPUs) require high-quality subnet manager (OpenSM)
- Monitor SM for stability: `sminfo` shows current SM state
- Use redundant SMs for fault tolerance

---

## Section 2: Straggler Detection & Performance Debugging (~150 lines)

### 2.1 What Are Stragglers?

**Straggler**: A slow-performing node that holds up the entire training job.

In synchronous distributed training (DDP, FSDP), all GPUs must finish each step before proceeding:
```
Step N Completion Times (128 GPUs):
GPU 0-126:  850ms  (average)
GPU 127:   2100ms  (STRAGGLER - holds up entire job)

Total step time = 2100ms (limited by slowest GPU)
Efficiency = 850ms / 2100ms = 40% (60% wasted time!)
```

**Common straggler causes:**
- **Hardware degradation**: Failing GPU, thermal throttling, ECC errors
- **Network issues**: Slow NIC, degraded InfiniBand link, switch congestion
- **Software bugs**: Imbalanced data loading, synchronization bugs
- **Resource contention**: Other processes consuming GPU/CPU/memory

From [Google Cloud - Stragglers in AI](https://cloud.google.com/blog/products/compute/stragglers-in-ai-a-guide-to-automated-straggler-detection) (accessed 2025-11-13):
> "Straggler nodes are responsible for the delay in synchronization during each iteration's aggregation step. The repetitive nature of SGD (Stochastic Gradient Descent) allows us to detect stragglers by analyzing timing patterns across iterations."

### 2.2 Straggler Detection Techniques

**Technique 1: Per-rank timing analysis**

```python
import torch
import torch.distributed as dist
import time

def detect_stragglers(step_time_ms, rank, world_size):
    """Detect stragglers by comparing step times across all ranks."""
    # Gather all step times to rank 0
    all_times = [torch.zeros(1) for _ in range(world_size)]
    step_time_tensor = torch.tensor([step_time_ms])

    dist.all_gather(all_times, step_time_tensor)

    if rank == 0:
        times = [t.item() for t in all_times]
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5

        # Flag stragglers (>2 std deviations from mean)
        stragglers = []
        for i, t in enumerate(times):
            if t > mean_time + 2 * std_time:
                stragglers.append((i, t))
                print(f"STRAGGLER DETECTED: Rank {i} took {t:.1f}ms (mean {mean_time:.1f}ms)")

        return stragglers
    return []

# Usage in training loop
for step, batch in enumerate(dataloader):
    start = time.time()

    # Forward, backward, optimizer step
    loss = model(batch)
    loss.backward()
    optimizer.step()

    step_time_ms = (time.time() - start) * 1000

    # Check for stragglers every 100 steps
    if step % 100 == 0:
        detect_stragglers(step_time_ms, rank, world_size)
```

**Technique 2: NCCL profiling for communication stragglers**

```bash
# Enable NCCL profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

# Run training and parse logs
python train.py 2>&1 | tee nccl_log.txt

# Analyze NCCL collective times
grep "AllReduce" nccl_log.txt | awk '{print $NF}' | sort -n

# Example output showing straggler:
# Rank 0-126: AllReduce completed in 45-50ms
# Rank 127: AllReduce completed in 180ms  <- STRAGGLER
```

**Technique 3: Automated straggler detection with monitoring**

From [Google Cloud - Stragglers in AI](https://cloud.google.com/blog/products/compute/stragglers-in-ai-a-guide-to-automated-straggler-detection) (accessed 2025-11-13), Google uses automated straggler detection:
- Monitor per-node step times continuously
- Use statistical outlier detection (Z-score, IQR)
- Trigger alerts when stragglers persist for multiple iterations
- Automatically cordon (remove) straggler nodes from cluster

### 2.3 Diagnosing Straggler Root Causes

**Once straggler detected, diagnose the cause:**

```bash
# Run these commands on the straggler node

# 1. Check GPU health
nvidia-smi
# Look for: High temperature (>80C), throttling, ECC errors, low power limit

nvidia-smi -q | grep -E "Temperature|Power|Throttled|ECC|Performance State"

# 2. Check for thermal throttling
nvidia-smi dmon -s pucvmet -i 0 -c 10
# Monitor: pwr (power), temp (temperature), sm (SM utilization), mem (memory utilization)
# Throttling if: temp >85C, pwr hitting limit, sm/mem low despite load

# 3. Check for ECC errors
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv
# Rising ECC errors = failing memory

# 4. Check network performance
ibstat  # InfiniBand link state
ethtool eth0  # Ethernet link speed
iperf3 -c <other_node_ip>  # Network bandwidth test

# 5. Check for resource contention
top  # CPU usage (other processes?)
iostat -x 1  # Disk I/O (slow dataloader?)
```

**Common straggler root causes and fixes:**

| **Symptom** | **Root Cause** | **Fix** |
|-------------|----------------|---------|
| GPU temp >85C | Thermal throttling | Improve cooling, reduce clock speed |
| Rising ECC errors | Failing GPU memory | Replace GPU or disable ECC |
| Low network bandwidth | Degraded NIC/cable | Replace cable, check IB port |
| High CPU load | Dataloader bottleneck | Increase num_workers, pin_memory |
| Persistent slowness | Hardware degradation | Remove node from cluster |

### 2.4 Straggler Mitigation Strategies

**Strategy 1: Timeout-based exclusion**

```python
import torch.distributed as dist
from datetime import timedelta

# Initialize with timeout
dist.init_process_group(
    backend='nccl',
    timeout=timedelta(seconds=300)  # 5 min timeout per operation
)

# If a rank times out during AllReduce, DDP will abort
# Then use NCCL Shrink (see Section 3) to remove failed rank
```

**Strategy 2: Elastic training with dynamic worker scaling**

From [PyTorch Elastic Training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-11-13):
- Use `torchrun` for elastic training
- Dynamically add/remove workers without restarting job
- When straggler detected, remove node and continue training

**Strategy 3: Asynchronous training (advanced)**

For extremely large scales (10,000+ GPUs), consider asynchronous SGD:
- Workers don't wait for stragglers
- Use parameter servers or gossip protocols
- Trade-off: convergence may be slower

### 2.5 Load Balancing at Scale

**Imbalanced workloads cause artificial stragglers:**

```python
# BAD: Imbalanced data distribution
# Last rank gets fewer samples
dataset_size = 1000000
world_size = 128
per_rank_size = dataset_size // world_size  # 7812 samples/rank
# Rank 127 gets 1000000 - 127*7812 = 976 samples (7x fewer!)

# GOOD: Use DistributedSampler for balanced distribution
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True  # Drop incomplete batches for perfect balance
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)
```

**Monitor load balance:**
```python
# Log samples processed per rank
if step % 100 == 0:
    samples_processed = step * batch_size
    print(f"Rank {rank}: {samples_processed} samples processed")
    # All ranks should report similar counts
```

---

## Section 3: Fault Tolerance at Scale (~150 lines)

### 3.1 Why Fault Tolerance Matters

**Failure probability increases with cluster size:**

```
Assume 0.1% GPU failure rate per day (realistic)

8 GPUs:   Probability of failure = 1 - (0.999)^8  = 0.8% per day
128 GPUs: Probability of failure = 1 - (0.999)^128 = 12% per day
1024 GPUs: Probability of failure = 1 - (0.999)^1024 = 64% per day

For 1000 GPU training run lasting 7 days:
Without fault tolerance: 99.7% chance of failure
With fault tolerance: Training continues despite failures
```

**Types of failures at scale:**
- **GPU hardware failure**: ECC errors, overheating, power delivery
- **Network failure**: IB link down, switch failure, congestion
- **Software hang**: Deadlock, NCCL timeout, CUDA error
- **Node failure**: OS crash, kernel panic, power loss

### 3.2 Checkpoint-Restart Strategy

**Basic checkpoint-restart:**

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'rng_state': torch.get_rng_state(),  # For reproducibility
    }

    # Only rank 0 saves
    if dist.get_rank() == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    torch.set_rng_state(checkpoint['rng_state'])

    print(f"Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step

# Training loop with checkpointing
checkpoint_dir = '/shared/checkpoints'
checkpoint_interval = 500  # Steps

for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(dataloader, start=start_step):
        # Training step
        loss = model(batch)
        loss.backward()
        optimizer.step()

        # Save checkpoint periodically
        if step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, step, checkpoint_dir)
```

**Automatic checkpoint loading on restart:**

```python
def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')

    # Try to load latest checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, start_step = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        print("Starting training from scratch")
        start_epoch, start_step = 0, 0

    # Continue training
    train(model, optimizer, start_epoch, start_step)

def find_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint file."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt'))
    if not checkpoints:
        return None
    # Sort by modification time
    return max(checkpoints, key=os.path.getmtime)
```

### 3.3 NCCL Communicator Shrink (Fault Recovery)

From [NVIDIA NCCL 2.27 Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) (accessed 2025-11-13):

**NCCL Communicator Shrink** allows excluding failed GPUs dynamically:

```python
import torch.distributed as dist

# Two modes:
# 1. NCCL_SHRINK_DEFAULT: Planned reconfiguration (waits for operations to complete)
# 2. NCCL_SHRINK_ABORT: Emergency recovery (aborts ongoing operations)

def shrink_communicator_on_failure(comm, excluded_ranks):
    """Shrink communicator to exclude failed ranks."""
    dist.group_start()

    for rank in range(dist.get_world_size()):
        if rank not in excluded_ranks:
            new_comm = dist.comm_shrink(
                comm,
                excluded_ranks,
                mode='abort'  # Abort mode for failure recovery
            )

    dist.group_end()

    return new_comm

# Usage in training loop with error handling
try:
    # Training step
    loss = model(batch)
    loss.backward()
    optimizer.step()
except RuntimeError as e:
    if "NCCL" in str(e):
        # NCCL failure detected
        print(f"NCCL error on rank {rank}: {e}")

        # Identify failed rank (simplified - production needs more robust detection)
        failed_rank = detect_failed_rank()  # Custom logic

        # Shrink communicator
        new_comm = shrink_communicator_on_failure(comm, [failed_rank])

        # Reload checkpoint and continue training with remaining GPUs
        load_checkpoint(model, optimizer, checkpoint_path)
```

**Fault recovery workflow:**
1. Detect failure (NCCL timeout, CUDA error)
2. Identify failed rank(s)
3. Shrink NCCL communicator to exclude failed rank(s)
4. Reload model/optimizer from last checkpoint
5. Continue training with reduced world size

### 3.4 Elastic Training with torchrun

From [PyTorch Elastic Training Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-11-13):

**torchrun provides fault tolerance and elasticity:**

```python
# train.py - Elastic training script
import os
import torch
import torch.distributed as dist

def setup():
    """Setup distributed training (torchrun sets env vars automatically)."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def load_snapshot(snapshot_path):
    """Load training snapshot if exists."""
    if os.path.exists(snapshot_path):
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot['model_state'])
        optimizer.load_state_dict(snapshot['optimizer_state'])
        epoch = snapshot['epoch']
        print(f"Resuming from snapshot: epoch {epoch}")
        return epoch
    return 0

def save_snapshot(snapshot_path, model, optimizer, epoch):
    """Save training snapshot."""
    snapshot = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(snapshot, snapshot_path)
    print(f"Snapshot saved: epoch {epoch}")

def main():
    setup()

    snapshot_path = 'snapshot.pt'
    start_epoch = load_snapshot(snapshot_path)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        for step, batch in enumerate(dataloader):
            # Training step
            loss = model(batch)
            loss.backward()
            optimizer.step()

            # Save snapshot periodically
            if step % 500 == 0:
                save_snapshot(snapshot_path, model, optimizer, epoch)

if __name__ == '__main__':
    main()
```

**Launch with torchrun:**
```bash
# Single node (8 GPUs)
torchrun --standalone --nproc_per_node=8 train.py

# Multi-node (16 nodes, 128 GPUs total)
# On rank 0 node:
torchrun \
  --nnodes=16 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py

# On other nodes (rank 1-15):
torchrun \
  --nnodes=16 \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py
```

**Elastic training benefits:**
- Automatic restart on failure (loads last snapshot)
- Dynamic scaling (add/remove nodes during training)
- No manual intervention needed
- Works with SLURM, Kubernetes, bare metal

### 3.5 Distributed Checkpointing for Large Models

**For massive models (100B+ parameters), single-file checkpoints don't work:**

```python
import torch.distributed.checkpoint as dist_cp

# Distributed checkpoint (saves sharded state across ranks)
def save_distributed_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint with sharding for large models."""
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    # Save sharded checkpoint
    dist_cp.save_state_dict(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(checkpoint_dir),
    )
    print(f"Distributed checkpoint saved: {checkpoint_dir}")

def load_distributed_checkpoint(model, optimizer, checkpoint_dir):
    """Load sharded checkpoint."""
    state_dict = {
        'model': model.state_dict(),  # Template
        'optimizer': optimizer.state_dict(),
    }

    # Load sharded checkpoint
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
    )

    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']
    print(f"Distributed checkpoint loaded: epoch {epoch}")
    return epoch
```

**Benefits of distributed checkpointing:**
- No single-node memory bottleneck (parallel I/O)
- Faster save/load (each rank saves its shard)
- Works with FSDP, ZeRO optimizer sharding
- Checkpoint size = model size / world_size per rank

---

## Section 4: Production Observability (~150 lines)

### 4.1 Distributed Logging Best Practices

**Centralized logging for massive-scale clusters:**

```python
import logging
import sys

def setup_logging(rank, world_size, log_dir='/shared/logs'):
    """Setup logging with rank-specific log files."""
    log_file = f'{log_dir}/rank_{rank}_of_{world_size}.log'

    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}/{world_size}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also log to stdout
        ]
    )

    logger = logging.getLogger(__name__)
    return logger

# Usage in training script
logger = setup_logging(rank, world_size)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()

    # Log metrics (only rank 0 to avoid spam)
    if rank == 0 and step % 10 == 0:
        logger.info(f"Step {step}, Loss: {loss.item():.4f}")

    # Log errors from any rank
    if torch.isnan(loss):
        logger.error(f"NaN loss detected at step {step}")
```

**Aggregate logs for analysis:**
```bash
# View all rank logs sorted by time
tail -f /shared/logs/rank_*.log | sort

# Search for errors across all ranks
grep -r "ERROR" /shared/logs/

# Count NCCL timeouts
grep -r "NCCL timeout" /shared/logs/ | wc -l

# Find which rank had errors
grep -r "CUDA error" /shared/logs/rank_*.log
```

### 4.2 Distributed Metrics Collection

**Collect metrics from all ranks for monitoring:**

```python
import torch.distributed as dist
import time

class DistributedMetrics:
    """Collect and aggregate metrics across all ranks."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.metrics = {}

    def log_metric(self, name, value):
        """Log a metric from this rank."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def aggregate_and_report(self):
        """Aggregate metrics across ranks and report."""
        for name, values in self.metrics.items():
            # Compute local stats
            local_mean = sum(values) / len(values)
            local_tensor = torch.tensor([local_mean]).cuda()

            # Gather from all ranks
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, local_tensor)

            if self.rank == 0:
                all_means = [t.item() for t in gathered_tensors]
                global_mean = sum(all_means) / len(all_means)
                global_std = (sum((x - global_mean)**2 for x in all_means) / len(all_means)) ** 0.5

                print(f"Metric {name}: mean={global_mean:.4f}, std={global_std:.4f}")

                # Detect outliers
                for i, val in enumerate(all_means):
                    if abs(val - global_mean) > 2 * global_std:
                        print(f"  WARNING: Rank {i} is outlier ({val:.4f})")

# Usage
metrics = DistributedMetrics(rank, world_size)

for step, batch in enumerate(dataloader):
    step_start = time.time()

    loss = model(batch)
    loss.backward()
    optimizer.step()

    step_time = time.time() - step_start
    metrics.log_metric('step_time_ms', step_time * 1000)
    metrics.log_metric('loss', loss.item())

    # Report every 100 steps
    if step % 100 == 0:
        metrics.aggregate_and_report()
```

### 4.3 Monitoring GPU Health at Scale

**Automated GPU health monitoring:**

```bash
#!/bin/bash
# monitor_gpus.sh - Monitor GPU health on all nodes

# Run nvidia-smi on all nodes
for node in $(cat nodelist.txt); do
    echo "=== Node $node ==="
    ssh $node 'nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used,ecc.errors.corrected.volatile.total --format=csv,noheader'
done

# Parse for issues
# Temperature >85C
# Power draw <80% of max (throttling)
# ECC errors >0 (memory issues)
```

**Integrate with Prometheus + Grafana:**
```python
# Export metrics to Prometheus
from prometheus_client import start_http_server, Gauge

# Define metrics
gpu_temp = Gauge('gpu_temperature_celsius', 'GPU temperature', ['rank', 'gpu_id'])
gpu_util = Gauge('gpu_utilization_percent', 'GPU utilization', ['rank', 'gpu_id'])
step_time = Gauge('training_step_time_ms', 'Training step time', ['rank'])

# Update metrics in training loop
def update_gpu_metrics(rank):
    for i in range(torch.cuda.device_count()):
        temp = torch.cuda.temperature(i)
        util = torch.cuda.utilization(i)

        gpu_temp.labels(rank=rank, gpu_id=i).set(temp)
        gpu_util.labels(rank=rank, gpu_id=i).set(util)

# Start Prometheus exporter (each rank on different port)
start_http_server(8000 + rank)

# Training loop
for step, batch in enumerate(dataloader):
    step_start = time.time()

    loss = model(batch)
    loss.backward()
    optimizer.step()

    step_time.labels(rank=rank).set((time.time() - step_start) * 1000)
    update_gpu_metrics(rank)
```

### 4.4 Anomaly Detection at Scale

**Detect anomalies in distributed training:**

```python
import numpy as np

class AnomalyDetector:
    """Detect anomalies in distributed training metrics."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.step_times = []
        self.losses = []

    def add_sample(self, step_time, loss):
        """Add a training step sample."""
        self.step_times.append(step_time)
        self.losses.append(loss)

        # Keep only recent samples
        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)
            self.losses.pop(0)

    def detect_anomalies(self):
        """Detect anomalies using statistical methods."""
        anomalies = []

        if len(self.step_times) < 10:
            return anomalies

        # Step time anomalies (Z-score method)
        step_mean = np.mean(self.step_times)
        step_std = np.std(self.step_times)

        latest_step_time = self.step_times[-1]
        z_score = (latest_step_time - step_mean) / (step_std + 1e-8)

        if abs(z_score) > 3:
            anomalies.append(f"Step time anomaly: {latest_step_time:.1f}ms (Z={z_score:.2f})")

        # Loss anomalies (gradient check)
        if len(self.losses) >= 2:
            loss_gradient = self.losses[-1] - self.losses[-2]
            if abs(loss_gradient) > 1.0:  # Sudden loss spike
                anomalies.append(f"Loss spike detected: {self.losses[-1]:.4f} (gradient {loss_gradient:.4f})")

        # NaN/Inf detection
        if np.isnan(self.losses[-1]) or np.isinf(self.losses[-1]):
            anomalies.append(f"NaN/Inf loss detected")

        return anomalies

# Usage in training
detector = AnomalyDetector()

for step, batch in enumerate(dataloader):
    step_start = time.time()

    loss = model(batch)
    loss.backward()
    optimizer.step()

    step_time = (time.time() - step_start) * 1000
    detector.add_sample(step_time, loss.item())

    # Check for anomalies
    anomalies = detector.detect_anomalies()
    for anomaly in anomalies:
        logger.warning(f"Rank {rank}: {anomaly}")
```

### 4.5 Distributed Tracing for Training Jobs

**Trace execution across distributed training job:**

```python
import time
import json

class DistributedTracer:
    """Simple distributed tracer for training jobs."""

    def __init__(self, rank, trace_file):
        self.rank = rank
        self.trace_file = trace_file
        self.events = []

    def record_event(self, name, category, timestamp=None):
        """Record a trace event."""
        if timestamp is None:
            timestamp = time.time()

        event = {
            'name': name,
            'cat': category,
            'ph': 'X',  # Complete event
            'ts': int(timestamp * 1e6),  # Microseconds
            'pid': self.rank,  # Process ID = rank
            'tid': 0,  # Thread ID
        }
        self.events.append(event)

    def save_trace(self):
        """Save trace to file (Chrome tracing format)."""
        with open(f'{self.trace_file}_rank{self.rank}.json', 'w') as f:
            json.dump({'traceEvents': self.events}, f)

# Usage in training loop
tracer = DistributedTracer(rank, 'training_trace')

for step, batch in enumerate(dataloader):
    step_start = time.time()
    tracer.record_event(f'Step {step}', 'training', step_start)

    # Forward pass
    forward_start = time.time()
    loss = model(batch)
    tracer.record_event(f'Forward {step}', 'forward', forward_start)

    # Backward pass
    backward_start = time.time()
    loss.backward()
    tracer.record_event(f'Backward {step}', 'backward', backward_start)

    # Optimizer step (includes AllReduce)
    optim_start = time.time()
    optimizer.step()
    tracer.record_event(f'Optimizer {step}', 'optimizer', optim_start)

# Save trace at end of training
tracer.save_trace()
```

**View traces in Chrome:**
```bash
# Open chrome://tracing in Chrome browser
# Load training_trace_rank0.json, training_trace_rank1.json, etc.
# Visualize execution timeline across all ranks
```

---

## Sources

**Web Research (accessed 2025-11-13):**
- [Google Cloud - Stragglers in AI: Automated Straggler Detection](https://cloud.google.com/blog/products/compute/stragglers-in-ai-a-guide-to-automated-straggler-detection)
- [NVIDIA Developer Blog - Enabling Fast Inference and Resilient Training with NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
- [PyTorch Tutorials - Fault-tolerant Distributed Training with torchrun](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
- [LambdaLabs - Distributed Training Guide (GitHub)](https://github.com/LambdaLabsML/distributed-training-guide)
- [arXiv - Falcon: Pinpointing and Mitigating Stragglers for Large-Scale Training](https://arxiv.org/html/2410.12588v1)

**Related Knowledge Files:**
- [cuda/11-advanced-troubleshooting-multi-gpu-expert.md](cuda/11-advanced-troubleshooting-multi-gpu-expert.md) - Basic multi-GPU debugging
- [cuda/10-performance-debugging-profiling-expert.md](cuda/10-performance-debugging-profiling-expert.md) - Performance debugging fundamentals
- [vertex-ai-production/00-distributed-training-patterns.md](vertex-ai-production/00-distributed-training-patterns.md) - DDP and distributed training patterns

**Additional Resources:**
- NVIDIA NCCL Documentation: [https://docs.nvidia.com/deeplearning/nccl/user-guide/](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- NCCL Tests Repository: [https://github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)
- PyTorch Distributed Documentation: [https://pytorch.org/docs/stable/distributed.html](https://pytorch.org/docs/stable/distributed.html)
