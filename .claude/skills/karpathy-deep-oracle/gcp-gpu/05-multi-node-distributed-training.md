# Multi-Node Distributed Training on GCP

## Overview

Multi-node distributed training scales deep learning workloads across multiple machines, each with multiple GPUs. This approach enables training massive models (100B+ parameters) that cannot fit on a single node, though it introduces significant inter-node communication overhead compared to single-node multi-GPU training.

**Key Concept**: Running a training job on 4 GPUs on a single node will be faster than running it on 4 nodes with 1 GPU each, due to inter-node network latency. Multi-node training is essential when model size or dataset scale exceeds single-node capacity.

From [PyTorch Multinode Training Tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html) (accessed 2025-11-16):
> "Multinode training is bottlenecked by inter-node communication latencies. Running a training job on 4 GPUs on a single node will be faster than running it on 4 nodes with 1 GPU each."

## Multi-Node Architecture Patterns

### Master-Worker Configuration

**Architecture Components**:
- **Master Node (Rank 0)**: Coordinates distributed training, handles checkpoint saving, logging
- **Worker Nodes**: Execute training computations, sync gradients with master
- **Rendezvous Backend**: Coordinates process discovery and communication setup

**Environment Variables** (PyTorch DDP):
- `RANK`: Global rank across all nodes (0 to world_size-1)
- `LOCAL_RANK`: GPU ID within each node (0 to GPUs_per_node-1)
- `WORLD_SIZE`: Total number of processes across all nodes
- `MASTER_ADDR`: IP address of master node
- `MASTER_PORT`: Port for inter-node communication (default 29500)

From [PyTorch Multinode Tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html):
> "In single-node settings, we were tracking the `gpu_id` of each device running our training process. `torchrun` tracks this value in an environment variable `LOCAL_RANK` which uniquely identifies each GPU-process on a node. For a unique identifier across all the nodes, `torchrun` provides another variable `RANK` which refers to the global rank of a process."

**GCP-Specific Setup**:
```bash
# On Master Node (10.128.0.2)
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=job_id \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.128.0.2:29500 \
    train.py

# On Worker Nodes
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=job_id \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.128.0.2:29500 \
    train.py
```

### Heterogeneous Scaling

PyTorch `torchrun` supports heterogeneous scaling where different nodes have different GPU counts:

**Example Configuration**:
- Node 0: 8×A100 GPUs
- Node 1: 4×A100 GPUs
- Node 2: 4×A100 GPUs
- Total: 16 GPUs across 3 nodes

From [PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html):
> "Torchrun supports heterogeneous scaling i.e. each of your multinode machines can have different number of GPUs participating in the training job."

## NCCL Configuration for Multi-Node

### Critical Environment Variables

From [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (accessed 2025-11-16):

**Network Interface Selection**:
```bash
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
export NCCL_IB_DISABLE=0        # Enable InfiniBand (if available)
export NCCL_NET_GDR_LEVEL=PHB   # GPU Direct RDMA level
```

**NCCL_SOCKET_IFNAME Options**:
- `eth0`: Default Ethernet interface on GCP
- `^docker0,lo`: Exclude docker and loopback interfaces
- Auto-detection may fail in complex network setups

**NCCL_NET_GDR_LEVEL Values**:
- `LOC`: GPU and NIC must be on same NUMA node
- `PHB`: Use GPU Direct RDMA when GPU and NIC on same PCIe bridge
- `SYS`: Use GPU Direct RDMA even across SMP interconnect (slower)

**Communication Optimization**:
```bash
export NCCL_DEBUG=INFO                    # Verbose logging for debugging
export NCCL_P2P_LEVEL=NVL                # Use NVLink for intra-node
export NCCL_P2P_DISABLE=0                # Enable peer-to-peer
export NCCL_CROSS_NIC=1                  # Use multiple NICs if available
export NCCL_ALGO=Ring,Tree               # Communication algorithms
```

### Network Topology on GCP

**A3 Mega (8×H100 80GB)**:
- 8 GPUs connected via NVSwitch (900 GB/s bisection bandwidth per GPU)
- 1600 Gbps network bandwidth per VM
- GPUDirect-TCPX for optimized multi-node communication
- Designed for 100-node+ clusters

From [Google Cloud A3 Mega Documentation](https://docs.cloud.google.com/compute/docs/gpus/gpudirect):
> "A3 Mega VMs feature 8 NVIDIA H100 GPUs with NVSwitch interconnect and GPUDirect-TCPX networking for high-bandwidth multi-node training."

**A3 High (8×H100 80GB)**:
- 8 GPUs connected via NVSwitch
- 200 Gbps network bandwidth per VM
- Standard RDMA networking

**N1/A2 Instances (Legacy)**:
- PCIe-based GPU interconnect (no NVSwitch)
- Up to 100 Gbps network bandwidth (machine type dependent)
- Limited to smaller-scale multi-node deployments

**Compact Placement Policy**:
GCP Compact Placement Policy co-locates VMs on same physical rack to minimize network latency:

```bash
gcloud compute resource-policies create group-placement gpu-cluster \
    --collocation=collocated \
    --region=us-central1

gcloud compute instances create node-{0..3} \
    --machine-type=a3-megagpu-8g \
    --accelerator=type=nvidia-h100-80gb,count=8 \
    --resource-policies=gpu-cluster \
    --zone=us-central1-a
```

**Benefits**:
- 2-3× lower inter-node latency
- Higher network bandwidth consistency
- Recommended for latency-sensitive multi-node training

## Fault Tolerance and Elasticity

### Checkpoint Strategies

**Distributed Checkpointing**:
```python
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, rank):
    if rank == 0:  # Only master saves
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    dist.barrier()  # Sync all nodes
```

**Checkpoint Frequency**:
- Save every 100-500 steps (depending on step time)
- Use persistent disk snapshots for durability
- Balance checkpoint overhead vs recovery time

From [PyTorch Fault Tolerance](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) (accessed 2025-11-16):
> "When a failure occurs, torchrun logs the errors and restarts the training from the last checkpoint. There is no guarantee that processes will hold the same LOCAL_RANK and RANK after restart."

### Elastic Training with torchrun

**Elastic Configuration**:
```bash
torchrun \
    --nnodes=2:4 \              # Min 2 nodes, max 4 nodes
    --nproc_per_node=8 \
    --max_restarts=3 \          # Max restart attempts
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py
```

**Elastic Features**:
- Automatic restart on node failure
- Dynamic node scaling (add/remove nodes during training)
- Checkpoint resume on recovery
- Membership change detection

**Warning**: Do not use `RANK` for critical logic as ranks may change on restart.

## Pipeline Parallelism Across Nodes

### Pipeline Parallel Strategy

Pipeline parallelism partitions model layers across nodes, processing micro-batches in a pipeline fashion.

From [Megatron-LM Research](https://github.com/NVIDIA/Megatron-LM) (accessed 2025-11-16):

**Pipeline Parallel Configuration**:
- Split model into N stages (typically 4-8)
- Assign each stage to different node/GPU set
- Process micro-batches in pipeline fashion

**Micro-batching**:
```
Global Batch = 1024
Micro-batch Size = 32
Number of Micro-batches = 1024 / 32 = 32

Pipeline Stages:
Node 0: Layers 0-7   (Stage 0)
Node 1: Layers 8-15  (Stage 1)
Node 2: Layers 16-23 (Stage 2)
Node 3: Layers 24-31 (Stage 3)
```

**Pipeline Schedules**:
- **GPipe**: Fills pipeline, then drains (bubble overhead)
- **PipeDream**: Interleaves forward/backward (less bubble)
- **1F1B (One Forward One Backward)**: Reduces memory, maintains efficiency

**Pipeline Efficiency**:
- Bubble time (idle time) = (p-1)/m where p=stages, m=micro-batches
- Example: 4 stages, 32 micro-batches = (4-1)/32 = 9.4% bubble overhead

## Tensor Parallelism Across Nodes

### Tensor Parallel Strategy

Tensor parallelism splits individual layers across multiple GPUs/nodes, distributing matrix multiplications.

From [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM):

**Tensor Parallel Example** (Transformer Layer):
```
Hidden Size = 12,288
Tensor Parallel Size = 8

Per-GPU Partition:
- Attention Heads: 96 / 8 = 12 heads per GPU
- MLP Width: 49,152 / 8 = 6,144 per GPU
```

**Communication Pattern**:
- **All-Gather**: Collect tensor partitions before computation
- **Reduce-Scatter**: Distribute results after computation
- **All-Reduce**: Synchronize gradients

**Tensor Parallel + NCCL Optimization**:
```bash
export NCCL_TOPO_FILE=/path/to/topology.xml  # Topology awareness
export NCCL_ALGO=Ring                        # Ring all-reduce
export NCCL_PROTO=Simple                     # Protocol selection
```

**Multi-Node Tensor Parallel**:
- Best for large model layers (GPT-3 scale: 175B parameters)
- Requires high-bandwidth inter-node networking (100+ Gbps)
- A3 Mega recommended for multi-node tensor parallelism

## 3D Parallelism (Data + Tensor + Pipeline)

### Hybrid Parallelism Strategy

Combines data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP) for maximum scalability.

**Example Configuration** (64 nodes × 8 GPUs = 512 GPUs):
```
Data Parallel: 8 replicas
Tensor Parallel: 8-way (within node)
Pipeline Parallel: 8 stages (across nodes)

Total GPUs: DP × TP × PP = 8 × 8 × 8 = 512 GPUs
```

**Rationale**:
- **TP**: Splits large layers within high-bandwidth node (NVLink/NVSwitch)
- **PP**: Splits model depth across nodes (minimizes inter-node communication)
- **DP**: Replicates model across independent sets (scales throughput)

**Communication Hierarchy**:
1. **TP**: All-reduce within node (NVLink, sub-microsecond latency)
2. **PP**: Point-to-point between nodes (100 Gbps network)
3. **DP**: All-reduce across replicas (overlapped with computation)

From [Megatron-LM Parallelism Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) (accessed 2025-11-16):
> "Megatron Bridge supports various data-parallel and model-parallel deep learning workload deployment methods, which can be mixed together arbitrarily."

## GCP Multi-Node Networking Optimization

### GPUDirect-TCPX (A3 Mega)

GPUDirect-TCPX optimizes GPU-to-GPU communication across nodes by bypassing CPU for network I/O.

From [GCP GPUDirect-TCPX Documentation](https://docs.cloud.google.com/compute/docs/gpus/gpudirect):

**Setup Requirements**:
```bash
# Install NCCL plugin for GPUDirect-TCPX
sudo apt-get install libnccl2 libnccl-dev
sudo apt-get install google-tcpx-nccl-plugin

# Enable GPUDirect-TCPX
export LD_LIBRARY_PATH=/usr/local/lib/nccl-tcpx:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=libnccl-tcpx.so
```

**Performance Benefits**:
- 2-3× higher effective bandwidth for multi-node all-reduce
- Lower CPU utilization (offloads network I/O to GPU)
- Optimized for A3 Mega cluster networking

**Topology Awareness**:
```bash
# Generate topology file
sudo nvidia-smi topo -m > topology.xml

# Use topology file
export NCCL_TOPO_FILE=/path/to/topology.xml
export NCCL_GRAPH_FILE=/path/to/nccl_graph.txt
```

### Network Bandwidth Monitoring

**Monitor NCCL Communication**:
```bash
# Enable NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run training and check logs
python train.py 2>&1 | grep -i "bandwidth\|busbw"
```

**Expected Bandwidth** (A3 Mega 8-node cluster):
- Intra-node (NVSwitch): ~600 GB/s per GPU
- Inter-node (GPUDirect-TCPX): ~180-190 GB/s per GPU
- Theoretical maximum: 200 GB/s (1600 Gbps / 8 GPUs)

**Bandwidth Testing**:
```bash
# NCCL all-reduce performance test
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# Run multi-node all-reduce
mpirun -np 64 -H node0:8,node1:8,...,node7:8 \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Monitoring Multi-Node Training

### Distributed Metrics Collection

**Cloud Monitoring Integration**:
```python
from google.cloud import monitoring_v3
import torch.distributed as dist

def log_training_metrics(loss, throughput, rank):
    if rank == 0:  # Only master logs to Cloud Monitoring
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{PROJECT_ID}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/training/loss"
        series.metric.labels["node_count"] = str(dist.get_world_size() // 8)

        point = series.points.add()
        point.value.double_value = loss
        point.interval.end_time.FromDatetime(datetime.utcnow())

        client.create_time_series(name=project_name, time_series=[series])
```

**Monitor Key Metrics**:
- **GPU Utilization**: Should be >90% during training
- **Network Bandwidth**: Inter-node communication saturation
- **Loss Convergence**: Verify multi-node training matches single-node
- **Throughput**: Samples/sec or tokens/sec per GPU

**DCGM Metrics** (NVIDIA Data Center GPU Manager):
```bash
# Install DCGM
sudo apt-get install datacenter-gpu-manager

# Start DCGM daemon
sudo nv-hostengine

# Monitor GPUs across nodes
dcgmi group -c all_gpus
dcgmi stats -g all_gpus -e
```

### Debugging Multi-Node Issues

**Common Issues**:

1. **Network Connectivity**:
```bash
# Test TCP connectivity between nodes
nc -zv $MASTER_ADDR 29500

# Test SSH between nodes
ssh node-1 hostname
```

2. **NCCL Communication Failures**:
```bash
export NCCL_DEBUG=INFO  # Enable verbose logging
# Check logs for "NET/Socket" or "NET/IB" initialization

# Common errors:
# "No network interface found" → Set NCCL_SOCKET_IFNAME
# "Connection refused" → Check firewall rules
```

3. **Rank Mismatch**:
```python
# Verify rank consistency
import torch.distributed as dist
print(f"Rank {dist.get_rank()}/{dist.get_world_size()}")
dist.barrier()  # Ensure all ranks reach sync point
```

4. **Checkpoint Loading**:
```bash
# Ensure all nodes see shared storage
ls -l /gcs/bucket/checkpoints/

# Use `dist.barrier()` after checkpoint load
```

From [PyTorch Multinode Tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html):

**Troubleshooting Tips**:
> "Ensure that your nodes are able to communicate with each other over TCP. Set env variable `NCCL_DEBUG` to `INFO` to print verbose logs that can help diagnose the issue. Sometimes you might need to explicitly set the network interface for the distributed backend (`export NCCL_SOCKET_IFNAME=eth0`)."

## arr-coc-0-1 Multi-Node Configuration

### 16-Node Training Cluster (128 GPUs Total)

**Cluster Configuration**:
- **Machine Type**: a3-megagpu-8g
- **GPUs per Node**: 8×H100 80GB
- **Total Nodes**: 16
- **Total GPUs**: 128
- **Network**: GPUDirect-TCPX enabled
- **Placement**: Compact Placement Policy for low latency

**Parallelism Strategy** (100B parameter VLM):
```python
# 3D Parallelism Configuration
TENSOR_PARALLEL = 8      # Within node (NVSwitch)
PIPELINE_PARALLEL = 4    # Across 4 node groups
DATA_PARALLEL = 4        # 4 independent replicas

# Total: 8 × 4 × 4 = 128 GPUs
```

**Launch Script**:
```bash
#!/bin/bash
# Master node environment
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

# NCCL optimization
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_NET_PLUGIN=libnccl-tcpx.so
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring,Tree

# Launch training (run on all 16 nodes)
torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_id=arr_coc_training \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    arr_coc/train_distributed.py \
        --tensor_parallel=8 \
        --pipeline_parallel=4 \
        --data_parallel=4 \
        --batch_size=1024 \
        --checkpoint_dir=gs://arr-coc-checkpoints/
```

**Expected Performance**:
- **Throughput**: ~45,000 tokens/sec (across all 128 GPUs)
- **Efficiency**: 85-90% of theoretical (accounting for communication overhead)
- **Training Time**: 100B model on 1T tokens = ~250 hours (~10 days)

**Cost Optimization**:
- Use Spot instances for worker nodes (60% savings)
- Keep master node on-demand (checkpoint reliability)
- Estimated cost: $15,000-$20,000 for full training run

## Sources

**PyTorch Documentation**:
- [Multinode Training Tutorial](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html) - PyTorch official multinode DDP guide (accessed 2025-11-16)
- [Fault-Tolerant Training](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) - Elastic training with torchrun (accessed 2025-11-16)

**NVIDIA Documentation**:
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) - NCCL configuration reference (accessed 2025-11-16)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM) - Advanced parallelism strategies (accessed 2025-11-16)
- [Megatron Bridge Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) - Tensor, pipeline, and data parallelism patterns (accessed 2025-11-16)

**Google Cloud Documentation**:
- [GPUDirect-TCPX Setup](https://docs.cloud.google.com/compute/docs/gpus/gpudirect) - A3 Mega network optimization (accessed 2025-11-16)
- [GPU Network Bandwidth](https://docs.cloud.google.com/compute/docs/gpus/gpu-network-bandwidth) - Network performance specifications (accessed 2025-11-16)

**Web Research**:
- Search: "multi-node PyTorch distributed training GCP 2024 2025"
- Search: "NCCL socket RDMA configuration multi-node GPU"
- Search: "elastic training fault tolerance PyTorch checkpoint resume"
- Search: "multi-node networking GCP 100 Gbps A3 Mega GPU cluster"
- Search: "pipeline parallelism Megatron-LM tensor parallelism multi-node 2024"
