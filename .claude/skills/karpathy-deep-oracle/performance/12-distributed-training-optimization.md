# Distributed Training Optimization

## Overview

Distributed training optimization focuses on efficiently scaling deep learning workloads across multiple GPUs and nodes while minimizing communication overhead, maintaining linear scaling efficiency, and managing complex trade-offs between different parallelism strategies. This guide covers scaling efficiency metrics, framework comparisons (ZeRO vs FSDP), hybrid parallelism strategies, fault tolerance, and production deployment patterns.

**Key Challenge**: Achieving near-linear scaling (85-95% efficiency) when training massive models (100B+ parameters) across hundreds of GPUs while managing communication bottlenecks, load balancing, and stragglers.

From [Multi-Node Distributed Training on GCP](../gcp-gpu/05-multi-node-distributed-training.md):
> "Multinode training is bottlenecked by inter-node communication latencies. Running a training job on 4 GPUs on a single node will be faster than running it on 4 nodes with 1 GPU each."

## Scaling Efficiency Fundamentals

### Linear Scaling Law

**Ideal Linear Scaling**:
```
Throughput = Single_GPU_Throughput × N_GPUs
Speedup = N_GPUs
Efficiency = 100%
```

**Reality**: Communication overhead, synchronization barriers, load imbalance reduce efficiency.

**Typical Scaling Efficiency** (well-optimized systems):
- **8 GPUs (single node)**: 90-95% efficiency
- **64 GPUs (8 nodes)**: 85-90% efficiency
- **512 GPUs (64 nodes)**: 75-85% efficiency
- **1024+ GPUs**: 70-80% efficiency (requires expert optimization)

### Communication Overhead Analysis

**Communication Costs by Parallelism Type**:

| Parallelism Type | Communication Pattern | Overhead | Bandwidth Needs |
|-----------------|----------------------|----------|----------------|
| **Data Parallel** | All-reduce gradients | Medium | ~2× model size per step |
| **Tensor Parallel** | All-gather/reduce-scatter | High | Frequent, small messages |
| **Pipeline Parallel** | Point-to-point (P2P) | Low | Infrequent, large messages |
| **ZeRO-1** | All-reduce optimizer states | Low | Once per step |
| **ZeRO-2** | All-reduce gradients + scatter optimizer | Medium | 2× gradients |
| **ZeRO-3** | All-gather parameters | High | 3× model parameters |

**Communication Volume Calculation**:
```python
# Data Parallel (DDP)
communication_per_step = 2 * model_parameters * 4  # FP32 gradients, bidirectional

# Tensor Parallel (TP=8)
tp_communication = num_layers * (
    hidden_size * seq_length * 2  # All-gather + reduce-scatter per layer
) * TP_SIZE

# ZeRO-3
zero3_communication = model_parameters * 4 * 3  # All-gather params, reduce-scatter gradients
```

From [arXiv:2412.04964 - Flash Communication](https://arxiv.org/abs/2412.04964) (accessed 2025-11-16):
> "Flash Communication, a novel low-bit compression technique designed to alleviate the tensor-parallelism communication bottleneck during inference. Our method substantially boosts intra-node communication speed by more than 3x and reduces the time-to-first-token by 2x."

### Measuring Scaling Efficiency

**Key Metrics**:

1. **Weak Scaling Efficiency**:
   - Keep per-GPU workload constant
   - Measure: `Time(1_GPU) / Time(N_GPUs)`
   - Ideal: Constant time as N increases

2. **Strong Scaling Efficiency**:
   - Keep total workload constant
   - Measure: `(Time(1_GPU) / Time(N_GPUs)) / N`
   - Ideal: 100% efficiency

3. **Model FLOPs Utilization (MFU)**:
   ```python
   # Achieved FLOPs / Theoretical Peak FLOPs
   MFU = (actual_throughput * 6 * params * seq_length) / (
       peak_tflops * num_gpus * 1e12
   )
   ```
   - **Good**: 40-50% MFU (A100/H100)
   - **Excellent**: 55-65% MFU

**Profiling Scaling**:
```python
import torch.distributed as dist
import time

def measure_scaling_efficiency(model, optimizer, dataloader, num_gpus):
    """Measure scaling efficiency across multiple runs."""
    start = time.time()

    for i, batch in enumerate(dataloader):
        if i >= 100:  # Warmup + measurement
            break

        loss = model(batch)
        loss.backward()

        # Synchronization point - measure communication overhead
        comm_start = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        comm_time = time.time() - comm_start

        if dist.get_rank() == 0 and i % 10 == 0:
            compute_time = time.time() - start - comm_time
            print(f"Step {i}: Compute={compute_time:.2f}s, Comm={comm_time:.2f}s")

    total_time = time.time() - start
    efficiency = (baseline_time * baseline_gpus) / (total_time * num_gpus)
    return efficiency
```

## ZeRO vs FSDP: Deep Comparison

### Memory Efficiency Comparison

**ZeRO-1 (Optimizer State Sharding)**:
- Memory savings: ~4× optimizer states
- Communication: Same as DDP (all-reduce gradients)
- Best for: Models that fit in GPU memory

**ZeRO-2 (+ Gradient Sharding)**:
- Memory savings: ~8× (optimizer + gradients)
- Communication: Reduce-scatter gradients, all-gather for optimizer
- Best for: Medium models (7B-30B parameters)

**ZeRO-3 (+ Parameter Sharding)** vs **FSDP**:

| Feature | DeepSpeed ZeRO-3 | PyTorch FSDP |
|---------|-----------------|--------------|
| **Parameter Sharding** | All parameters sharded | All parameters sharded |
| **Memory Savings** | 16× vs DDP | 16× vs DDP |
| **Precision Handling** | Forces FP32 master weights | User-controlled precision |
| **Communication** | All-gather params on-demand | All-gather params on-demand |
| **Gradient Reduction** | Reduce-scatter | Reduce-scatter |
| **Optimizer Step** | FP32 by default | User precision (BF16/FP32) |
| **Setup Complexity** | JSON config file | Python API |
| **Checkpoint Format** | Custom DeepSpeed format | Native PyTorch |

From [Hugging Face: DeepSpeed to FSDP and Back](https://huggingface.co/blog/deepspeed-to-fsdp-and-back) (accessed 2025-11-16):
> "DeepSpeed was performing upcasting internally, and it always keeps its master weights in FP32 by design. This upcasting to full precision meant that the optimizer could converge at learning rates that it would not converge in lower precision."

**Key Difference: Precision Philosophy**

**DeepSpeed ZeRO-3**:
```python
# DeepSpeed forces FP32 optimizer states (good for convergence)
{
  "fp16": {"enabled": true},  # Training in FP16
  "zero_optimization": {
    "stage": 3,
    # Internally: optimizer always runs in FP32
  }
}
```

**FSDP** (flexible precision):
```python
from torch.distributed.fsdp import MixedPrecision

# Option 1: FP32 optimizer (DeepSpeed-like)
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    # Optimizer runs in FP32 (upcast happens automatically in Accelerate 0.30+)
)

# Option 2: BF16 optimizer (memory-constrained)
mixed_precision_low_mem = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    # Optimizer runs in BF16 (saves memory, may need LR tuning)
)
```

### Performance Comparison

From search results (GitHub discussions, 2024):
> "In my experience and setup, I found FSDP to be approximately 10% faster than ZeRO-3. This appears to be because FSDP consolidates parameters more efficiently."

**Throughput Benchmarks** (7B model, 4×A100):

| Framework | Tokens/sec/GPU | Step Time (s) | MFU |
|-----------|---------------|--------------|-----|
| FSDP (mixed precision) | 3158 | 10.4 | 0.41 |
| DeepSpeed ZeRO-3 | 3094 | 10.6 | 0.40 |

**When to Choose Each**:

**Choose DeepSpeed ZeRO-3**:
- Need FP32 optimizer stability (sensitive models)
- Already using DeepSpeed ecosystem (ZeRO-Offload, ZeRO++)
- Training on Azure/AWS with DeepSpeed integration
- Want battle-tested convergence at default settings

**Choose FSDP**:
- Want PyTorch-native solution (easier debugging)
- Need memory-constrained mode (BF16 optimizer)
- Prefer Python API over JSON config
- Want flexible precision control
- Need seamless checkpoint compatibility with PyTorch

## Hybrid Parallelism Strategies

### 3D Parallelism (Data + Tensor + Pipeline)

**Rationale**: Combine parallelism types to exploit hardware hierarchy.

From [Multi-Node Distributed Training](../gcp-gpu/05-multi-node-distributed-training.md):
> "TP: All-reduce within node (NVLink, sub-microsecond latency)
> PP: Point-to-point between nodes (100 Gbps network)
> DP: All-reduce across replicas (overlapped with computation)"

**Configuration Example** (512 GPUs = 64 nodes × 8 GPUs):
```python
# 3D Parallelism Strategy
TENSOR_PARALLEL = 8      # Within node (NVLink)
PIPELINE_PARALLEL = 8    # Across 8 node groups
DATA_PARALLEL = 8        # 8 independent replicas

# Total: TP × PP × DP = 8 × 8 × 8 = 512 GPUs

# Model split:
# - Tensor Parallel: Split each layer across 8 GPUs (same node)
# - Pipeline Parallel: Split layers into 8 stages (across nodes)
# - Data Parallel: 8 replicas for throughput
```

From [arXiv:2503.23186 - Optimizing Distributed Training](https://arxiv.org/html/2503.23186v1) (accessed 2025-11-16):
> "This paper demonstrates that adaptive parallelism strategies can significantly improve the efficiency of distributed deep learning training."

### Hybrid Parallelism Decision Matrix

**Model Size → Strategy**:

| Model Size | GPUs | Recommended Strategy | Reasoning |
|-----------|------|---------------------|-----------|
| **< 7B** | 1-8 | DDP or FSDP | Fits on single node |
| **7B-30B** | 8-32 | FSDP or ZeRO-2 | Needs parameter sharding |
| **30B-70B** | 32-128 | FSDP + TP=4 or ZeRO-3 | Large activations |
| **70B-175B** | 128-512 | TP=8, PP=4, DP=16 | 3D parallelism |
| **175B+** | 512+ | TP=8, PP=16+, DP=4+ | Extreme scale |

**Network Topology Considerations**:

```python
# A3 Mega (8×H100, NVSwitch, GPUDirect-TCPX)
config = {
    "intra_node": {
        "bandwidth": "900 GB/s per GPU (NVSwitch)",
        "latency": "sub-microsecond",
        "best_for": "Tensor Parallel (TP=8)"
    },
    "inter_node": {
        "bandwidth": "200 GB/s (1600 Gbps / 8 GPUs)",
        "latency": "10-50 microseconds",
        "best_for": "Pipeline Parallel, Data Parallel"
    }
}

# Optimal: TP=8 (intra-node), PP/DP (inter-node)
```

### Dynamic Parallelism Switching

From search results (arXiv 2025):
> "Adaptive parallelism strategies can significantly improve efficiency by adjusting to workload characteristics and hardware topology."

**Example: Context-Aware Parallelism**:
```python
def select_parallelism_strategy(model_size, num_gpus, context_length):
    """Select optimal parallelism based on workload."""

    if context_length > 32768:
        # Long context: favor Sequence Parallel
        return {
            "tensor_parallel": 4,
            "sequence_parallel": True,
            "pipeline_parallel": num_gpus // 4
        }
    elif model_size > 100e9:
        # Large model: 3D parallelism
        return {
            "tensor_parallel": 8,
            "pipeline_parallel": 8,
            "data_parallel": num_gpus // 64
        }
    else:
        # Standard: FSDP
        return {"fsdp_sharding": "full"}
```

## Load Balancing Optimization

### Pipeline Parallel Load Balancing

**Challenge**: Uneven layer computational costs cause pipeline bubbles.

**Pipeline Bubble Time**:
```
Bubble_Time = (num_stages - 1) / num_microbatches × microbatch_time
Efficiency = 1 - Bubble_Time / Total_Time
```

**Optimization Strategies**:

1. **Increase Micro-batches**:
   ```python
   # More micro-batches = smaller bubble
   num_microbatches = 4 * num_pipeline_stages  # Rule of thumb

   # Example: 8 stages → 32 micro-batches
   # Bubble overhead: (8-1) / 32 = 21.9% → good
   ```

2. **Layer Balancing**:
   ```python
   # Manual layer assignment (Megatron-LM style)
   layer_distribution = [
       [0, 1, 2, 3],      # Stage 0: 4 layers
       [4, 5, 6, 7, 8],   # Stage 1: 5 layers (has expensive MLP)
       [9, 10, 11, 12],   # Stage 2: 4 layers
       [13, 14, 15, 16]   # Stage 3: 4 layers
   ]
   ```

3. **1F1B Schedule** (One Forward One Backward):
   - Reduces memory vs GPipe
   - Maintains low bubble overhead
   - Most common in production

### Data Parallel Load Balancing

**Gradient Bucketing**:
```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

# Gradient bucketing reduces communication frequency
ddp_model = DDP(
    model,
    bucket_cap_mb=25,  # Default: 25MB buckets
    gradient_as_bucket_view=True  # Avoid extra copy
)

# Communication happens when bucket fills (not per-parameter)
```

**Stragglers Mitigation**:
```python
# Timeout for slow nodes
dist.init_process_group(
    backend="nccl",
    timeout=timedelta(minutes=30)  # Kill stragglers after 30min
)

# Alternative: Backup tasks (Google-style)
# Launch 2 copies of slow tasks, use first to finish
```

## Fault Tolerance and Elasticity

### Checkpoint Strategies for Distributed Training

**Distributed Checkpointing Best Practices**:

```python
import torch.distributed as dist
from torch.distributed.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)

def save_distributed_checkpoint(model, optimizer, epoch, rank):
    """Save checkpoint with FSDP/ZeRO-3 sharded state."""

    # Only rank 0 saves metadata
    if rank == 0:
        metadata = {
            "epoch": epoch,
            "world_size": dist.get_world_size(),
            "model_config": model.config.to_dict(),
        }
        torch.save(metadata, f"checkpoint_epoch_{epoch}/metadata.pt")

    # All ranks save their sharded state
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    save_state_dict(
        state_dict=state_dict,
        storage_writer=FileSystemWriter(f"checkpoint_epoch_{epoch}"),
    )

    dist.barrier()  # Ensure all ranks finish
```

**Checkpoint Frequency Trade-offs**:

| Checkpoint Every | Pros | Cons |
|-----------------|------|------|
| **100 steps** | Fast recovery | High I/O overhead (~5% slowdown) |
| **500 steps** | Balanced | Moderate recovery time |
| **1000 steps** | Low overhead | Lose significant work on failure |
| **Async** | No blocking | Complex, needs extra memory |

### Elastic Training with PyTorch

From [Multi-Node Training](../gcp-gpu/05-multi-node-distributed-training.md):
> "Elastic training with torchrun supports automatic restart on node failure, dynamic node scaling, checkpoint resume on recovery, and membership change detection."

**Elastic Configuration**:
```bash
# Elastic training: 2-4 nodes dynamically
torchrun \
    --nnodes=2:4 \              # Min 2 nodes, max 4 nodes
    --nproc_per_node=8 \
    --max_restarts=3 \          # Max restart attempts
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py
```

**Handling Rank Changes**:
```python
import torch.distributed.elastic.multiprocessing as mp

def train_with_elastic(local_rank):
    """Training function that handles rank changes."""

    # DON'T use rank for critical logic (changes on restart)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Safe: Only master does logging
    is_master = (rank == 0)

    # UNSAFE: Storing rank-specific data
    # rank_data[rank] = ...  # ❌ Breaks on restart

    # Safe: Use worker ID (persistent)
    worker_id = os.environ.get("RANK", "0")  # Environment-based ID
```

## Communication Optimization Techniques

### Gradient Compression

**FP16 Gradient Communication**:
```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

ddp_model = DDP(model)

# FP16 gradient communication (50% bandwidth reduction)
ddp_model.register_comm_hook(
    state=None,
    hook=default_hooks.fp16_compress_hook
)

# Reduces communication: 2× model_size → 1× model_size
```

**PowerSGD Compression**:
```python
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

# PowerSGD: Low-rank gradient approximation
state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=4,  # Rank-4 approximation
    start_powerSGD_iter=100,      # Warmup: 100 iterations
)

ddp_model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)

# 10-100× compression, minimal accuracy loss
```

From [arXiv:2412.04964 - Flash Communication](https://arxiv.org/abs/2412.04964):
> "Low-bit compression technique designed to alleviate the tensor-parallelism communication bottleneck... boosts intra-node communication speed by more than 3×."

### Overlapping Communication and Computation

**DDP Communication Overlap**:
```python
# DDP automatically overlaps gradient reduction with backward pass
# via bucketing and async NCCL calls

# Monitor overlap effectiveness
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # Gradients reduced async during backward
        optimizer.step()

# Check for gaps between compute and communication
prof.export_chrome_trace("trace.json")
```

**FSDP All-Gather Prefetch**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    forward_prefetch=True,          # Prefetch next layer's params
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch for backward
    limit_all_gathers=True,         # Limit memory for prefetch
)

# Prefetching overlaps all-gather with compute
# Reduces waiting time by 20-30%
```

### NCCL Tuning

**Critical Environment Variables**:
```bash
# Network interface selection (CRITICAL for multi-node)
export NCCL_SOCKET_IFNAME=eth0  # Or ib0 for InfiniBand

# Communication algorithm
export NCCL_ALGO=Ring,Tree      # Try both, measure

# Topology awareness
export NCCL_TOPO_FILE=/path/to/topology.xml

# GPUDirect RDMA (if available)
export NCCL_NET_GDR_LEVEL=PHB   # GPU and NIC on same PCIe bridge

# Debugging
export NCCL_DEBUG=INFO          # See NCCL decisions
export NCCL_DEBUG_SUBSYS=ALL    # Detailed logging
```

**Measuring NCCL Performance**:
```bash
# NCCL all-reduce benchmark
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=1

# Test all-reduce performance (multi-node)
mpirun -np 64 -H node0:8,node1:8,...,node7:8 \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Expected bandwidth (A3 Mega cluster):
# - Intra-node: ~600 GB/s per GPU (NVSwitch)
# - Inter-node: ~180-190 GB/s per GPU (GPUDirect-TCPX)
```

## Production Deployment Patterns

### Multi-Node Training Setup (GCP Example)

**Cluster Configuration**:
```bash
# Create compact placement policy (low latency)
gcloud compute resource-policies create group-placement gpu-cluster \
    --collocation=collocated \
    --region=us-central1

# Launch 8-node A3 Mega cluster
for i in {0..7}; do
  gcloud compute instances create node-$i \
      --machine-type=a3-megagpu-8g \
      --accelerator=type=nvidia-h100-80gb,count=8 \
      --resource-policies=gpu-cluster \
      --zone=us-central1-a \
      --metadata=startup-script='#!/bin/bash
        # Install NCCL plugin for GPUDirect-TCPX
        apt-get install -y libnccl2 libnccl-dev google-tcpx-nccl-plugin
        export LD_LIBRARY_PATH=/usr/local/lib/nccl-tcpx:$LD_LIBRARY_PATH
        export NCCL_NET_PLUGIN=libnccl-tcpx.so
      '
done
```

**Training Launch Script**:
```bash
#!/bin/bash
# Master node environment
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

# NCCL optimization for A3 Mega
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_NET_PLUGIN=libnccl-tcpx.so
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring,Tree

# Launch training (run on all nodes)
torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --rdzv_id=training_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_script.py \
        --tensor_parallel=8 \
        --pipeline_parallel=2 \
        --data_parallel=4 \
        --batch_size=1024
```

### Cost Optimization Strategies

**Spot Instance Strategy**:
```python
# Use spot for workers, on-demand for master
cluster_config = {
    "master": {
        "instance_type": "a3-megagpu-8g",
        "preemptible": False,  # On-demand for reliability
        "checkpointing": "every_100_steps",
    },
    "workers": {
        "instance_type": "a3-megagpu-8g",
        "preemptible": True,   # 60-70% cost savings
        "auto_restart": True,
        "max_restarts": 3,
    }
}

# Estimated savings: 50-60% on total cluster cost
# Trade-off: ~10% throughput loss from preemptions
```

**Training Time vs Cost**:

| Configuration | GPUs | Training Time | Cost Estimate | Cost/Hour |
|--------------|------|--------------|---------------|-----------|
| **Single A100** | 1 | 45 days | $32,400 | $30/hr |
| **8×A100 (1 node)** | 8 | 6 days | $4,320 | $30/hr |
| **64×H100 (8 nodes)** | 64 | 18 hours | $4,608 | $256/hr |
| **512×H100 (64 nodes)** | 512 | 3 hours | $6,144 | $2,048/hr |

**Throughput vs Cost Sweet Spot**: 8-16 nodes (64-128 GPUs)
- Good scaling efficiency (85-90%)
- Manageable complexity
- Reasonable cost ($500-1000/hr)

## arr-coc-0-1 Distributed Scaling

### 128 GPU Training Configuration (100B VLM)

**Cluster Setup** (16 nodes × 8 H100):
```python
# 3D Parallelism for 100B parameter vision-language model
config = {
    "tensor_parallel": 8,       # Within node (NVSwitch, 900 GB/s)
    "pipeline_parallel": 4,     # Across 4 node groups
    "data_parallel": 4,         # 4 independent replicas

    # Total: 8 × 4 × 4 = 128 GPUs

    "model_size": "100B parameters",
    "context_length": 4096,
    "global_batch_size": 1024,
    "micro_batch_size": 2,      # Per GPU
}

# Expected performance:
# - Throughput: ~45,000 tokens/sec (all 128 GPUs)
# - MFU: 50-55% (well-optimized)
# - Training time: 1T tokens in ~250 hours (~10 days)
# - Cost: $15,000-$20,000 (spot instances)
```

**Optimization Highlights**:

1. **Gradient Checkpointing** (every 2nd transformer block):
   - Memory savings: 40%
   - Compute overhead: 20%
   - Net benefit: Fits in 80GB H100 memory

2. **Communication Optimization**:
   - NCCL Ring algorithm for DP (large messages)
   - Tree algorithm for TP (latency-sensitive)
   - GPUDirect-TCPX for inter-node (2-3× faster)

3. **Load Balancing**:
   - 32 micro-batches (4 PP stages × 8 micro-batches/stage)
   - Pipeline bubble: ~9% overhead
   - Dynamic layer assignment based on profiling

**Launch Command**:
```bash
# From master node
torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    arr_coc/train_distributed.py \
        --config configs/100b_vlm.yaml \
        --checkpoint_dir=gs://arr-coc-checkpoints/ \
        --wandb_project=arr-coc-100b
```

### Monitoring Distributed Training

**Key Metrics to Track**:
```python
import wandb
import torch.distributed as dist

def log_distributed_metrics(step, loss, throughput, rank):
    """Log metrics from distributed training."""

    if rank == 0:  # Only master logs to wandb
        metrics = {
            "train/loss": loss,
            "train/tokens_per_sec": throughput,
            "train/tokens_per_sec_per_gpu": throughput / dist.get_world_size(),

            # GPU utilization
            "system/gpu_util": torch.cuda.utilization(),
            "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,

            # Communication metrics (if tracked)
            "comm/all_reduce_time_ms": all_reduce_time * 1000,
            "comm/all_gather_time_ms": all_gather_time * 1000,
        }

        wandb.log(metrics, step=step)
```

**Alerting for Issues**:
- GPU utilization < 80% → communication bottleneck
- Loss divergence → learning rate too high or numerical instability
- Throughput drop > 10% → straggler node or network issue

## Sources

**Source Documents**:
- [gcp-gpu/05-multi-node-distributed-training.md](../gcp-gpu/05-multi-node-distributed-training.md) - Multi-node patterns, NCCL config, fault tolerance

**Web Research**:
- [arXiv:2408.12596 - Poplar: Efficient Scaling on Heterogeneous GPU Clusters](https://arxiv.org/abs/2408.12596) (accessed 2025-11-16) - ZeRO extensions for heterogeneous GPUs, 1.02-3.92× throughput improvements
- [Hugging Face: DeepSpeed to FSDP and Back](https://huggingface.co/blog/deepspeed-to-fsdp-and-back) (accessed 2025-11-16) - Precision differences, performance comparisons, migration guide
- [arXiv:2412.04964 - Flash Communication](https://arxiv.org/abs/2412.04964) (accessed 2025-11-16) - Low-bit compression for tensor parallelism, 3× communication speedup
- [arXiv:2503.23186 - Optimizing Distributed Training](https://arxiv.org/html/2503.23186v1) (accessed 2025-11-16) - Adaptive parallelism strategies
- Search: "distributed training scaling efficiency 2024" - Scaling laws, communication overhead analysis
- Search: "ZeRO-3 vs FSDP performance comparison 2024" - Framework comparisons, precision handling
- Search: "tensor parallelism communication overhead 2024" - Communication bottlenecks, optimization techniques
- Search: "hybrid parallelism strategies distributed training 2024" - 3D parallelism, load balancing

**Additional References**:
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html) - Official FSDP API reference
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/config-json/) - ZeRO configuration
- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) - Environment variables, tuning guide
- [Google Cloud GPUDirect-TCPX](https://docs.cloud.google.com/compute/docs/gpus/gpudirect) - A3 Mega network optimization
