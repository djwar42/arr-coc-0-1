# LLM Training Dynamics & GPU Optimization: Gradient Checkpointing, Pipeline Parallelism, ZeRO, and 3D Parallelism

## Overview

Training large language models (LLMs) with billions to trillions of parameters requires sophisticated memory and computation optimization techniques that leverage GPU hardware capabilities. This document covers four critical training dynamics optimizations: gradient checkpointing (activation recomputation), pipeline parallelism, ZeRO optimizer states partitioning, and tensor parallelism, culminating in 3D parallelism that combines all three techniques.

**Why These Optimizations Matter:**
- **Memory bottleneck**: Training LLMs is memory-limited, not compute-limited
- **Activation memory dominates**: For large batch sizes, activations consume more memory than model parameters
- **Model parallelism enables scale**: Single GPU cannot hold 175B+ parameter models
- **Communication overhead**: Multi-GPU training requires optimizing GPU-to-GPU data movement

**Critical Insight from Training Dynamics:**

From [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md):
- Mixed precision training reduces memory by 50% (FP16/BF16 vs FP32)
- But activation memory still dominates for large models
- Gradient checkpointing trades 33% extra compute for 75% memory savings

**Related Knowledge:**
- See [cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md) for GPU memory hierarchy
- See [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) for AMP and GradScaler
- See [vertex-ai-production/00-distributed-training-patterns.md](../vertex-ai-production/00-distributed-training-patterns.md) for PyTorch DDP

---

## Section 1: Gradient Checkpointing (Activation Recomputation) (200 lines)

### 1.1 The Activation Memory Problem

**Memory Breakdown for LLM Training:**

For a transformer model with:
- L layers
- H hidden dimension
- S sequence length
- B batch size

**Memory consumption:**
```
Parameters:       12 * L * H² (weights stored once)
Optimizer states: 12 * L * H² (for Adam: fp32 weights + 2 moments)
Gradients:        12 * L * H² (fp16 gradients during backward)
Activations:      L * B * S * H (intermediate outputs stored for backward)
```

**Example: GPT-3 175B (L=96, H=12288, B=32, S=2048):**
```python
# Parameters
params = 96 * 12 * (12288**2) * 4 / 1e9  # 175 GB (FP32)

# Optimizer states (Adam)
optimizer = 175 * 2  # 350 GB (2 moments)

# Activations (per layer)
activations_per_layer = 32 * 2048 * 12288 * 4 / 1e9  # 3.2 GB
total_activations = 96 * 3.2  # 307 GB!

# Total: 175 + 350 + 307 = 832 GB (doesn't fit on any GPU!)
```

**Problem:** Activations dominate memory for large batch sizes and long sequences.

From [NVIDIA NeMo Activation Recomputation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html) (accessed 2025-02-03):
> "The input activations of network layers are stored in device memory and are used to compute gradients during back-propagation. When training a LLM with a long sequence length or a large micro-batch size, these input activations can quickly saturate device memory."

### 1.2 Gradient Checkpointing Algorithm

**Core Idea:** Don't store all intermediate activations; checkpoint a few and recompute the rest during backward pass.

**Full Checkpointing (Naive):**
```
Forward:  Store activations for all L layers
Backward: Use stored activations to compute gradients
Memory:   O(L * B * S * H)
```

**Gradient Checkpointing (Optimal):**
```
Forward:  Store activations every √L layers (checkpoints)
Backward: Recompute activations between checkpoints on-the-fly
Memory:   O(√L * B * S * H)
Compute:  +33% (one extra forward pass)
```

**Example: 96-layer model**
```python
# Without checkpointing: Store 96 layers
memory_full = 96 * batch_size * seq_len * hidden_dim

# With checkpointing: Store √96 ≈ 10 checkpoints
memory_checkpoint = 10 * batch_size * seq_len * hidden_dim

# Memory reduction: 96/10 = 9.6× reduction!
```

From [NeurIPS 2024 paper on Activation Recomputation](https://proceedings.neurips.cc/paper_files/paper/2024/file/b063829b922fdeb4fa3472dd3471ff43-Paper-Conference.pdf) (accessed 2025-02-03):
> "Activation recomputation, also known as gradient checkpointing, is a memory-saving technique for training large neural networks. This method trades computation time for memory by recomputing intermediate activations during the backward pass instead of storing them."

### 1.3 PyTorch Implementation

**Basic Gradient Checkpointing:**
```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim)
        self.ffn = FeedForward(hidden_dim)

    def forward(self, x):
        # Regular forward (stores all activations)
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x

class TransformerWithCheckpointing(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Use checkpoint to save memory
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**HuggingFace Transformers Integration:**
```python
from transformers import GPT2LMHeadModel

# Enable gradient checkpointing
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.gradient_checkpointing_enable()

# Training loop (no other changes needed)
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()  # Checkpointing happens automatically
    optimizer.step()
```

### 1.4 Selective Checkpointing Strategies

**Strategy 1: Transformer Layer Checkpointing**

From [NVIDIA NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html) (accessed 2025-02-03):
> "Transformer layer recomputation checkpoints the input of each transformer layer and recomputes the activations for the remaining layers. This technique significantly reduces activation memory usage. However, it increases the per-transformer layer computation cost by 30%."

```python
# NeMo configuration for full transformer layer checkpointing
{
    "activations_checkpoint_granularity": "full",
    "activations_checkpoint_method": "block",
    "activations_checkpoint_num_layers": 96  # Checkpoint all layers
}
```

**Strategy 2: Selective Attention Checkpointing**

From [NVIDIA NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html):
> "Self-attention recomputation checkpoints the inputs of each self-attention block and recomputes the intermediate input activations. This cost-efficient method achieves high memory savings with minimal recomputation cost."

```python
# NeMo configuration for selective checkpointing
{
    "activations_checkpoint_granularity": "selective"
}

# Selective checkpointing targets attention layers:
# - QKV projections: O(H²) memory, expensive to recompute
# - Softmax: O(S²) memory, cheap to recompute ✓
# - Dropout: O(S²) memory, cheap to recompute ✓
# - Output projection: O(H²) memory, expensive to recompute
```

**Strategy 3: Block Checkpointing**
```python
# Checkpoint every N layers (instead of every layer)
{
    "activations_checkpoint_method": "uniform",
    "activations_checkpoint_num_layers": 12  # Checkpoint every 12 layers
}
# For 96-layer model: 96/12 = 8 checkpoints
# Memory: 8 × activation_size (vs 96× without checkpointing)
```

### 1.5 Performance Trade-offs

**Memory vs Compute:**
```
Checkpointing None:     Memory = 100%,  Compute = 100%
Checkpointing Full:     Memory =  25%,  Compute = 133% (33% overhead)
Checkpointing Selective: Memory =  40%,  Compute = 115% (15% overhead)
```

From [Kaitchup Substack on Gradient Checkpointing](https://kaitchup.substack.com/p/gradient-checkpointing-llms) (accessed 2025-02-03):
> "Gradient checkpointing significantly reduces memory usage, cutting activation memory by up to 70%."

**When to Use Gradient Checkpointing:**
1. ✅ Large batch sizes (activation memory > parameter memory)
2. ✅ Long sequences (S > 2048)
3. ✅ Memory-constrained GPUs (e.g., training 13B model on 40GB A100)
4. ❌ Small models (<1B parameters)
5. ❌ Already fitting in memory comfortably

**Optimal Strategy Selection:**
```python
# Decision tree
if activation_memory > available_gpu_memory:
    if sequence_length > 4096:
        use_selective_checkpointing()  # Target attention layers
    else:
        use_full_checkpointing()  # Checkpoint all layers
else:
    no_checkpointing()  # Max speed
```

---

## Section 2: Pipeline Parallelism (200 lines)

### 2.1 Pipeline Parallelism Concepts

**The Model Parallelism Problem:**

When a model is too large for a single GPU, we must split it across multiple GPUs. Two approaches:
1. **Tensor Parallelism**: Split individual layers across GPUs (communication-intensive)
2. **Pipeline Parallelism**: Split sequential layers across GPUs (bubble overhead)

**Pipeline Parallelism Architecture:**
```
GPU 0: Layers  0-23  (Transformer blocks 1-24)
GPU 1: Layers 24-47  (Transformer blocks 25-48)
GPU 2: Layers 48-71  (Transformer blocks 49-72)
GPU 3: Layers 72-95  (Transformer blocks 73-96)
```

**Naive Pipeline (Sequential):**
```
Time →
GPU 0: [Forward] [Backward]        (idle)        (idle)
GPU 1:  (idle)   [Forward] [Backward]        (idle)
GPU 2:  (idle)    (idle)   [Forward] [Backward]
GPU 3:  (idle)    (idle)    (idle)   [Forward] [Backward]

Problem: Only 1 GPU active at a time! 75% idle time (pipeline bubble)
```

### 2.2 GPipe: Micro-Batching to Reduce Bubbles

**Solution: Split batch into micro-batches and pipeline them**

```
Batch size = 8, split into 4 micro-batches (M0, M1, M2, M3)

Time →
GPU 0: [F-M0][F-M1][F-M2][F-M3][B-M3][B-M2][B-M1][B-M0]
GPU 1:  (idle)[F-M0][F-M1][F-M2][F-M3][B-M3][B-M2][B-M1][B-M0]
GPU 2:  (idle) (idle)[F-M0][F-M1][F-M2][F-M3][B-M3][B-M2][B-M1][B-M0]
GPU 3:  (idle) (idle) (idle)[F-M0][F-M1][F-M2][F-M3][B-M3][B-M2][B-M1][B-M0]

F = Forward, B = Backward
Bubble time: 3 idle slots / 12 total = 25% (much better!)
```

**GPipe Algorithm:**
```python
# Micro-batch size calculation
num_gpus = 4
num_microbatches = 8  # More microbatches → smaller bubble
batch_size = 32
microbatch_size = batch_size // num_microbatches  # 4 samples

# Forward pass all microbatches
activations = []
for mb in range(num_microbatches):
    mb_data = data[mb * microbatch_size : (mb+1) * microbatch_size]
    mb_output = model_stage(mb_data)  # Forward through this GPU's layers
    activations.append(mb_output)
    send_to_next_gpu(mb_output)

# Backward pass all microbatches (reverse order)
for mb in reversed(range(num_microbatches)):
    mb_grad = receive_from_next_gpu()
    backward_pass(activations[mb], mb_grad)
```

**Bubble Time Formula:**
```
Bubble fraction = (num_gpus - 1) / num_microbatches

Examples:
4 GPUs, 4 microbatches:  (4-1)/4  = 75% bubble
4 GPUs, 8 microbatches:  (4-1)/8  = 37.5% bubble
4 GPUs, 16 microbatches: (4-1)/16 = 18.75% bubble
```

### 2.3 PipeDream: 1F1B Schedule

**GPipe Problem:** All-forward-then-all-backward creates memory spikes

**PipeDream Solution:** Interleave forward and backward (1F1B)

```
Time →
GPU 0: [F0][F1][F2][F3][B3][B2][B1][B0]
GPU 1:     [F0][F1][F2][F3][B3][B2][B1][B0]
GPU 2:         [F0][F1][F2][F3][B3][B2][B1][B0]
GPU 3:             [F0][F1][F2][F3][B3][B2][B1][B0]

Key: After warmup (3 forward), alternate 1 forward + 1 backward
Memory: Only 4 microbatches in flight (vs 8 in GPipe)
```

**1F1B Implementation:**
```python
def train_1f1b_pipeline(model_stage, num_microbatches):
    num_warmup = num_pipeline_stages - 1  # 3 for 4-stage pipeline

    # Warmup: Fill pipeline with forward passes
    for mb in range(num_warmup):
        activations[mb] = forward(data[mb])
        send_to_next_stage(activations[mb])

    # Steady state: 1 forward, 1 backward
    for mb in range(num_warmup, num_microbatches):
        # Backward for oldest microbatch
        grad = receive_from_next_stage()
        backward(activations[mb - num_warmup], grad)

        # Forward for new microbatch
        activations[mb] = forward(data[mb])
        send_to_next_stage(activations[mb])

    # Cooldown: Drain pipeline with backward passes
    for mb in range(num_microbatches - num_warmup, num_microbatches):
        grad = receive_from_next_stage()
        backward(activations[mb], grad)
```

### 2.4 Megatron-LM Pipeline Parallelism

From [Megatron-LM Paper](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) (accessed 2025-02-03):
> "In particular, we show how to combine pipeline, tensor, and data parallelism, a technique we call PTD-P, to train large language models with good computational efficiency."

**Megatron-LM Improvements:**
1. **Virtual Pipeline Parallelism**: Split each GPU into multiple virtual stages
2. **Interleaved Schedules**: Reduce bubble overhead further
3. **Optimized Communication**: All-reduce gradients across tensor-parallel ranks

**Virtual Pipeline Example:**
```
Instead of:
GPU 0: Layers [0-23]
GPU 1: Layers [24-47]
GPU 2: Layers [48-71]
GPU 3: Layers [72-95]

Use virtual stages (each GPU has 2 stages):
GPU 0: Layers [0-11], [48-59]
GPU 1: Layers [12-23], [60-71]
GPU 2: Layers [24-35], [72-83]
GPU 3: Layers [36-47], [84-95]

Benefit: Smaller bubbles (interleave 8 stages instead of 4)
```

### 2.5 Pipeline Parallelism Configuration

**PyTorch Pipeline Parallelism (PiPPy):**
```python
import torch
from torch.distributed.pipelined import pipeline

# Split model into 4 stages
model = GPT2Model(...)
stages = torch.nn.ModuleList([
    model.layers[0:24],   # Stage 0 → GPU 0
    model.layers[24:48],  # Stage 1 → GPU 1
    model.layers[48:72],  # Stage 2 → GPU 2
    model.layers[72:96],  # Stage 3 → GPU 3
])

# Create pipeline schedule
schedule = pipeline.PipelineSchedule1F1B(
    stages=stages,
    num_microbatches=8,
    loss_fn=cross_entropy_loss
)

# Training loop
for batch in dataloader:
    schedule.step(batch)
```

**DeepSpeed Pipeline Parallelism:**
```python
# deepspeed_config.json
{
    "pipeline": {
        "num_stages": 4,
        "micro_batch_size": 4,
        "activation_checkpoint_interval": 1
    }
}

# Model definition
class GPT2Pipeline(PipelineModule):
    def __init__(self, num_layers=96, **kwargs):
        super().__init__(layers=[
            # DeepSpeed automatically partitions layers
            *[TransformerLayer(**kwargs) for _ in range(num_layers)]
        ], num_stages=4)
```

**Megatron-LM Pipeline Config:**
```bash
# Launch script
python pretrain_gpt.py \
    --pipeline-model-parallel-size 4 \  # 4-stage pipeline
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --micro-batch-size 4 \
    --global-batch-size 32  # 4 microbatches × 8 accumulation steps
```

### 2.6 Performance Considerations

**Optimal Number of Microbatches:**
```
Too few:  Large bubble overhead
Too many: Small microbatch → poor GPU utilization

Rule of thumb: num_microbatches = 4 × num_pipeline_stages
```

**Communication Patterns:**
```
Point-to-point (P2P) between adjacent stages:
GPU 0 ↔ GPU 1: Send activations, receive gradients
GPU 1 ↔ GPU 2: Send activations, receive gradients
GPU 2 ↔ GPU 3: Send activations, receive gradients

Bandwidth requirement: ~12GB/s (PCIe) to ~600GB/s (NVLink)
```

From [vertex-ai-production/00-distributed-training-patterns.md](../vertex-ai-production/00-distributed-training-patterns.md):
- PCIe 4.0: 32GB/s bidirectional
- NVLink 3.0 (A100): 600GB/s per GPU
- NVLink 4.0 (H100): 900GB/s per GPU

---

## Section 3: ZeRO: Zero Redundancy Optimizer (200 lines)

### 3.1 The Optimizer State Memory Problem

**Problem: Adam Optimizer Memory**

For a model with P parameters (in billions):
```
FP16 model parameters:        2P bytes
FP32 optimizer states:
  - FP32 master weights:      4P bytes
  - First moment (momentum):  4P bytes
  - Second moment (variance): 4P bytes
Total optimizer memory:       12P bytes

Example: 175B parameter model (GPT-3)
Optimizer states: 12 × 175B = 2.1 TB!
Model parameters: 2 × 175B = 350 GB
```

**Traditional Data Parallelism Redundancy:**
```
8 GPUs, each stores:
- Full model (350 GB)
- Full optimizer states (2.1 TB)
Total memory across 8 GPUs: 8 × 2.45 TB = 19.6 TB

But only 2.45 TB is actually needed!
Redundancy: 8× duplication
```

### 3.2 ZeRO Stage 1: Optimizer State Partitioning

**Idea:** Each GPU stores only 1/N of the optimizer states

From [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/) (accessed 2025-02-03):
> "Stage 1: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition."

```
8 GPUs, ZeRO Stage 1:
GPU 0: Optimizer states for parameters [0:P/8]
GPU 1: Optimizer states for parameters [P/8:2P/8]
GPU 2: Optimizer states for parameters [2P/8:3P/8]
...
GPU 7: Optimizer states for parameters [7P/8:P]

Each GPU: 2.1 TB / 8 = 262.5 GB optimizer states
Memory savings: 7/8 = 87.5% reduction in optimizer memory
```

**ZeRO Stage 1 Algorithm:**
```python
# During backward pass
def backward_with_zero_stage1(model, loss, optimizer):
    # 1. Compute gradients (normal backward)
    loss.backward()

    # 2. All-reduce gradients (sum across all ranks)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    # 3. Each rank updates only its partition of optimizer states
    rank = dist.get_rank()
    for i, param in enumerate(model.parameters()):
        if i % world_size == rank:
            # This rank owns this parameter's optimizer state
            optimizer.step_single_param(param)

    # 4. All-gather updated parameters
    for param in model.parameters():
        dist.all_gather(param.data)
```

**Memory Breakdown (175B model, 8 GPUs):**
```
Without ZeRO:
  Model:     350 GB per GPU
  Optimizer: 2.1 TB per GPU
  Total:     2.45 TB per GPU

ZeRO Stage 1:
  Model:     350 GB per GPU
  Optimizer: 262.5 GB per GPU (divided by 8)
  Total:     612.5 GB per GPU
  Savings:   75% reduction!
```

### 3.3 ZeRO Stage 2: Gradient Partitioning

From [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Stage 2: The reduced 16-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states."

**Additional Partitioning:** Each GPU stores only 1/N of gradients

```
8 GPUs, ZeRO Stage 2:
GPU 0: Gradients for parameters [0:P/8]
GPU 1: Gradients for parameters [P/8:2P/8]
...
GPU 7: Gradients for parameters [7P/8:P]

Gradient memory per GPU: 350 GB / 8 = 43.75 GB
```

**ZeRO Stage 2 Algorithm (Reduce-Scatter):**
```python
def backward_with_zero_stage2(model, loss):
    # 1. Compute gradients
    loss.backward()

    # 2. Reduce-scatter gradients (each rank gets different shard)
    for i, param in enumerate(model.parameters()):
        grad_shard = reduce_scatter(param.grad, rank=i % world_size)
        # Now each GPU has only 1/N of total gradients

    # 3. Each rank updates its partition of parameters
    optimizer.step()  # Only updates owned shards

    # 4. All-gather parameters for next forward pass
    for param in model.parameters():
        dist.all_gather(param.data)
```

**Memory Breakdown (175B model, 8 GPUs):**
```
ZeRO Stage 2:
  Model:     350 GB per GPU
  Gradients: 43.75 GB per GPU (divided by 8)
  Optimizer: 262.5 GB per GPU (divided by 8)
  Total:     656.25 GB per GPU
  vs Stage 1: Additional 6% savings
```

### 3.4 ZeRO Stage 3: Parameter Partitioning

From [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Stage 3: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes."

**Full Model Partitioning:** Each GPU stores only 1/N of model parameters

```
8 GPUs, ZeRO Stage 3:
GPU 0: Parameters [0:P/8]       + optimizer states for [0:P/8]
GPU 1: Parameters [P/8:2P/8]    + optimizer states for [P/8:2P/8]
GPU 2: Parameters [2P/8:3P/8]   + optimizer states for [2P/8:3P/8]
...
GPU 7: Parameters [7P/8:P]      + optimizer states for [7P/8:P]

Model parameter memory per GPU: 350 GB / 8 = 43.75 GB
```

**ZeRO Stage 3 Algorithm (All-Gather on Demand):**
```python
def forward_with_zero_stage3(model, input):
    # 1. All-gather parameters for current layer
    for layer in model.layers:
        all_gather_layer_params(layer)  # Bring parameters to all GPUs

        # 2. Forward through layer
        output = layer(input)

        # 3. Partition parameters again (free memory)
        partition_layer_params(layer)  # Each GPU keeps 1/N

        input = output

    return output

def backward_with_zero_stage3(model, loss):
    # Similar: all-gather → backward → partition
    for layer in reversed(model.layers):
        all_gather_layer_params(layer)
        backward_through_layer(layer)
        partition_layer_params(layer)
```

**Memory Breakdown (175B model, 8 GPUs):**
```
ZeRO Stage 3:
  Model:     43.75 GB per GPU (divided by 8)
  Gradients: 43.75 GB per GPU (divided by 8)
  Optimizer: 262.5 GB per GPU (divided by 8)
  Total:     350 GB per GPU
  vs No ZeRO: 2.45 TB → 350 GB = 86% reduction!
```

### 3.5 ZeRO-Infinity: CPU and NVMe Offloading

**Problem:** Even ZeRO-3 can't train trillion-parameter models on GPU alone

**Solution:** Offload optimizer states and parameters to CPU/NVMe

From [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "ZeRO-3 includes the infinity offload engine to form ZeRO-Infinity, which can offload to both CPU and NVMe memory for huge memory savings."

```json
// deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",  // or "nvme"
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Memory Breakdown (1T parameter model, 8 GPUs):**
```
GPU Memory (per GPU):
  Activations only: ~100 GB
  Parameters: Offloaded to CPU/NVMe
  Optimizer: Offloaded to CPU/NVMe

CPU Memory (per node, 1TB RAM):
  Parameters: 2 TB / 8 nodes = 250 GB
  Optimizer: 12 TB / 8 nodes = 1.5 TB
  Total: 1.75 TB (fits with swap)

NVMe (per node, 4TB):
  Additional overflow storage
```

**Performance Impact:**
```
No Offload:    100% speed, GPU memory limited
CPU Offload:   70% speed (PCIe bandwidth overhead)
NVMe Offload:  40% speed (NVMe bandwidth overhead)

Trade-off: 2× larger models at 0.7× speed
```

### 3.6 ZeRO Configuration

**DeepSpeed ZeRO Config:**
```json
{
    "zero_optimization": {
        "stage": 3,

        // Memory optimizations
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 500000000,
        "allgather_bucket_size": 500000000,

        // ZeRO-3 specific
        "stage3_max_live_parameters": 1000000000,
        "stage3_max_reuse_distance": 1000000000,
        "stage3_prefetch_bucket_size": 10000000,
        "stage3_param_persistence_threshold": 100000,

        // Offloading
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },

    // Mixed precision
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}
```

**PyTorch FSDP (Alternative to ZeRO):**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload

model = GPT2Model(...)
model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # Similar to ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
)
```

---

## Section 4: Tensor Parallelism (150 lines)

### 4.1 Tensor Parallelism Concepts

**Idea:** Split individual layers across multiple GPUs

Unlike pipeline parallelism (split layers sequentially), tensor parallelism splits each layer:

```
Standard Linear Layer: Y = XW + b
  X: [batch, seq_len, hidden_dim]
  W: [hidden_dim, hidden_dim]
  Y: [batch, seq_len, hidden_dim]

Tensor Parallel (2 GPUs):
  W split column-wise:
    W1: [hidden_dim, hidden_dim/2] on GPU 0
    W2: [hidden_dim, hidden_dim/2] on GPU 1

  Y1 = XW1  (GPU 0 computes half)
  Y2 = XW2  (GPU 1 computes half)
  Y = [Y1 | Y2]  (concatenate)
```

### 4.2 Megatron-LM Tensor Parallelism

From [Megatron-LM Paper](https://arxiv.org/abs/1909.08053):
> "We introduce tensor parallelism to split the computation of a single transformer layer across multiple GPUs, achieving near-linear scaling."

**Column-Parallel Linear:**
```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tensor_parallel_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tensor_parallel_size

        # Each GPU stores output_size / tp_size columns
        self.weight = nn.Parameter(torch.empty(
            input_size,
            output_size // tp_size
        ))

    def forward(self, input):
        # All-gather input across tensor parallel ranks
        input_parallel = input  # Already replicated

        # Local matrix multiplication
        output_parallel = F.linear(input_parallel, self.weight)

        # No communication needed (each GPU has different columns)
        return output_parallel
```

**Row-Parallel Linear:**
```python
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tensor_parallel_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tensor_parallel_size

        # Each GPU stores input_size / tp_size rows
        self.weight = nn.Parameter(torch.empty(
            input_size // tp_size,
            output_size
        ))

    def forward(self, input_parallel):
        # Input already split across dimension
        # Local matrix multiplication
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce to sum results from all GPUs
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        return output_parallel
```

### 4.3 Attention Layer Tensor Parallelism

**Multi-Head Attention Parallelism:**

```
Standard Attention (32 heads):
  Q, K, V projections: [hidden, hidden]
  Split across heads: Each head [hidden, hidden/32]
  Attention computation per head
  Output projection: [hidden, hidden]

Tensor Parallel (4 GPUs, 32 heads → 8 heads per GPU):
  GPU 0: Heads  0-7  (Q, K, V projections for 8 heads)
  GPU 1: Heads  8-15 (Q, K, V projections for 8 heads)
  GPU 2: Heads 16-23 (Q, K, V projections for 8 heads)
  GPU 3: Heads 24-31 (Q, K, V projections for 8 heads)
```

**Implementation:**
```python
class TensorParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, tp_size):
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.heads_per_partition = num_heads // tp_size

        # Column-parallel: Split heads across GPUs
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            tp_size
        )

        # Row-parallel: Reduce across heads
        self.output_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_size
        )

    def forward(self, x):
        # QKV projection (each GPU computes subset of heads)
        qkv = self.qkv_proj(x)
        q, k, v = split_qkv(qkv, self.heads_per_partition)

        # Local attention computation
        attn_output = scaled_dot_product_attention(q, k, v)

        # Output projection (all-reduce across GPUs)
        output = self.output_proj(attn_output)
        return output
```

### 4.4 MLP Layer Tensor Parallelism

**Transformer MLP:**
```
x → Linear(h→4h) → GeLU → Linear(4h→h) → output

Tensor Parallel (2 GPUs):
  First linear (column-parallel):
    GPU 0: [h → 2h]
    GPU 1: [h → 2h]
    No communication needed (split columns)

  Second linear (row-parallel):
    GPU 0: [2h → h]
    GPU 1: [2h → h]
    All-reduce sum outputs
```

**Code:**
```python
class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, tp_size):
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            4 * hidden_size,
            tp_size
        )

        self.fc2 = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            tp_size
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # All-reduce happens here
        return x
```

### 4.5 Communication Costs

**Communication Patterns:**
```
Forward Pass:
  Column-parallel → Row-parallel: All-reduce (sum outputs)

Backward Pass:
  Gradient of row-parallel: All-reduce (sum gradients)
  Gradient of column-parallel: All-gather (replicate gradients)
```

**Bandwidth Requirements:**
```python
# For each transformer layer
hidden_size = 12288
batch_size = 32
seq_len = 2048
tp_size = 4

# All-reduce volume per layer
activation_size = batch_size * seq_len * hidden_size * 2  # FP16
all_reduce_bytes = activation_size  # ~1.6 GB

# Bandwidth needed (96 layers, 2 all-reduces per layer)
total_data = 96 * 2 * 1.6  # ~307 GB per iteration

# With NVLink (600 GB/s for A100):
communication_time = 307 / 600  # 0.51 seconds
```

From [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md):
- NVLink 3.0 (A100): 600 GB/s bidirectional
- NVLink 4.0 (H100): 900 GB/s bidirectional
- Tensor parallelism requires high-bandwidth interconnect (NVLink > PCIe)

---

## Section 5: 3D Parallelism (Combining All Techniques) (100 lines)

### 5.1 3D Parallelism Architecture

**Combining Three Parallelism Dimensions:**

```
Data Parallelism (DP=8):   Replicate model across 8 groups
Tensor Parallelism (TP=4): Split each layer across 4 GPUs
Pipeline Parallelism (PP=2): Split layers into 2 stages

Total GPUs = DP × TP × PP = 8 × 4 × 2 = 64 GPUs
```

**Example: Training GPT-3 175B on 64 GPUs:**
```
Pipeline Parallelism (PP=2):
  Stage 0: Layers 0-47  (first half of model)
  Stage 1: Layers 48-95 (second half of model)

Tensor Parallelism (TP=4 within each pipeline stage):
  GPU 0: 1/4 of stage 0 layers
  GPU 1: 1/4 of stage 0 layers
  GPU 2: 1/4 of stage 0 layers
  GPU 3: 1/4 of stage 0 layers

Data Parallelism (DP=8 across pipeline+tensor groups):
  8 complete replicas of the PP+TP configuration
```

### 5.2 Megatron-LM 3D Parallelism

From [Megatron-LM Paper](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf):
> "We show how to combine pipeline, tensor, and data parallelism, a technique we call PTD-P, to train large language models with good computational efficiency."

**Communication Topology:**
```
Tensor Parallel (TP): All-reduce within node (NVLink)
Pipeline Parallel (PP): P2P between adjacent stages (NVLink or InfiniBand)
Data Parallel (DP): All-reduce across replicas (InfiniBand)
```

**Optimal Parallelism Selection:**
```python
def choose_parallelism(model_size_B, num_gpus, gpu_memory_GB):
    # Start with tensor parallelism (high bandwidth)
    tp_size = 4 if gpu_memory_GB == 40 else 8

    # Add pipeline parallelism if model doesn't fit
    layers_per_gpu = model_size_B / (tp_size * gpu_memory_GB * 0.5)
    if layers_per_gpu > 1:
        pp_size = math.ceil(layers_per_gpu)
    else:
        pp_size = 1

    # Remaining GPUs for data parallelism
    dp_size = num_gpus // (tp_size * pp_size)

    return dp_size, tp_size, pp_size

# Example: 175B model, 512 GPUs, 40GB A100s
dp, tp, pp = choose_parallelism(175, 512, 40)
# Result: DP=64, TP=4, PP=2
```

### 5.3 Implementation Example

**Megatron-LM Launch Script:**
```bash
#!/bin/bash

# 3D Parallelism configuration
WORLD_SIZE=512
DP_SIZE=64
TP_SIZE=4
PP_SIZE=2

# Model configuration
NUM_LAYERS=96
HIDDEN_SIZE=12288
NUM_HEADS=96
SEQ_LENGTH=2048

# Training configuration
GLOBAL_BATCH_SIZE=2048  # Effective batch size
MICRO_BATCH_SIZE=4      # Per-GPU batch size
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_SIZE)))

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=64 \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 500000 \
    --lr 1.5e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-iters 320000 \
    --fp16
```

**DeepSpeed 3D Parallelism:**
```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,

    "pipeline": {
        "num_stages": 2,
        "activation_checkpoint_interval": 1
    },

    "tensor_parallel": {
        "tp_size": 4
    },

    "zero_optimization": {
        "stage": 1,  // ZeRO-1 for data parallelism
        "reduce_bucket_size": 500000000
    },

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}
```

### 5.4 Performance Analysis

**Speedup Calculation:**
```python
# Single GPU baseline (impossible for 175B model)
single_gpu_time = float('inf')  # Doesn't fit

# With 3D parallelism (DP=64, TP=4, PP=2)
effective_compute = 512 * 0.85  # 85% efficiency
speedup = effective_compute / 1  # vs single GPU if it fit

# Efficiency breakdown:
tensor_parallel_efficiency = 0.95   # NVLink overhead
pipeline_parallel_efficiency = 0.92 # 8% bubble
data_parallel_efficiency = 0.98     # All-reduce overhead

overall_efficiency = 0.95 * 0.92 * 0.98 = 0.86
```

**Memory Calculation (175B model):**
```python
# Per GPU memory (DP=64, TP=4, PP=2)
model_memory = 175e9 * 2 / (4 * 2)  # Split across TP and PP
optimizer_memory = 175e9 * 12 / 64  # ZeRO-1 splits optimizer
activation_memory = batch_size * seq_len * hidden * 4  # Micro-batch

total_per_gpu = (model_memory + optimizer_memory + activation_memory) / 1e9
# ≈ 43.75 + 32.8 + 3.2 = 79.75 GB (fits in 80GB A100!)
```

---

## Section 6: ARR-COC Multi-Stage Training (100 lines)

### 6.1 ARR-COC Training Stages

**ARR-COC has three training stages:**

```
Stage 1: Texture Array Extraction (frozen encoder)
  Memory: 13-channel texture generation
  Challenge: 13 channels × batch × patches = large activations

Stage 2: Relevance Scorer Training (3 scorers)
  Memory: 3 separate scorer forwards
  Challenge: Each scorer processes different modalities

Stage 3: Quality Adapter Fine-tuning
  Memory: End-to-end gradient flow
  Challenge: All components active simultaneously
```

### 6.2 Gradient Checkpointing for Texture Processing

**Problem: 13-Channel Texture Array Memory**

```python
# Memory breakdown
batch_size = 32
num_patches = 200  # Variable LOD (64-400 tokens)
num_channels = 13  # RGB, LAB, Sobel, spatial, eccentricity
patch_size = 224

# Activation memory per image
activation_memory = batch_size * num_patches * num_channels * patch_size**2 * 4
# 32 × 200 × 13 × 224² × 4 bytes = 52 GB!
```

**Solution: Checkpoint Texture Channel Computation**
```python
from torch.utils.checkpoint import checkpoint

class TextureExtractor(nn.Module):
    def forward(self, image):
        # Checkpoint each channel group
        rgb_features = checkpoint(self.rgb_encoder, image)
        lab_features = checkpoint(self.lab_encoder, image)
        sobel_features = checkpoint(self.sobel_encoder, image)

        # Combine features
        texture_array = torch.cat([
            rgb_features, lab_features, sobel_features
        ], dim=1)
        return texture_array

# Memory reduction: 13 channels → 4 checkpoints
# 52 GB → 16 GB (69% reduction)
```

### 6.3 Pipeline Parallelism for Multi-Stage Training

**ARR-COC Training Pipeline:**
```
Stage 0 (GPU 0): Texture Array Extraction
  Input: Image [B, 3, H, W]
  Output: Texture [B, 13, num_patches, patch_size²]

Stage 1 (GPU 1): Propositional Knowing
  Input: Texture features
  Output: Information content scores

Stage 2 (GPU 2): Perspectival Knowing
  Input: Texture features
  Output: Salience landscape scores

Stage 3 (GPU 3): Participatory Knowing
  Input: Texture features + Query
  Output: Query-content coupling scores
```

**Implementation:**
```python
class ARRCOCPipeline(nn.Module):
    def __init__(self):
        self.stages = nn.ModuleList([
            TextureExtractor(),           # Stage 0
            PropositionalScorer(),        # Stage 1
            PerspectivalScorer(),         # Stage 2
            ParticipatoryScorer(),        # Stage 3
        ])

    def forward(self, image, query):
        # Micro-batch pipeline
        textures = []
        for mb in range(num_microbatches):
            mb_image = image[mb * mb_size : (mb+1) * mb_size]
            mb_texture = self.stages[0](mb_image)
            textures.append(mb_texture)

            # Send to next stage
            if stage_id == 0:
                send_to_next_stage(mb_texture)

        # Continue pipeline for remaining stages...
```

### 6.4 Mixed Precision for Opponent Processing

From [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md):
> "BF16 is more stable than FP16 for operations with extreme dynamic range."

**Opponent Processing Stability:**
```python
# Opponent processing requires stable gradients
class TensionBalancer(nn.Module):
    def forward(self, compress_score, particularize_score):
        # BF16 for stability
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            tension = compress_score - particularize_score
            balance = torch.sigmoid(tension)  # Needs wide dynamic range
        return balance
```

**Configuration:**
```json
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0
}
```

### 6.5 Future: 3D Parallelism for ARR-COC

**Scaling ARR-COC to Larger Vision-Language Models:**

```
Target: Qwen-VL 72B + ARR-COC Components
  Total parameters: 72B (Qwen) + 2B (ARR-COC) = 74B

Configuration:
  DP = 16 (data parallelism)
  TP = 4  (tensor parallelism for attention)
  PP = 2  (pipeline: Qwen-VL | ARR-COC)

Total GPUs: 16 × 4 × 2 = 128 GPUs

Pipeline Stages:
  Stage 0: Qwen-VL layers 0-39 + Texture Extractor
  Stage 1: Qwen-VL layers 40-79 + 3 Relevance Scorers + Quality Adapter
```

**Estimated Performance:**
```python
# Per-GPU memory (80GB A100)
model_per_gpu = 74e9 * 2 / (4 * 2)  # TP=4, PP=2
optimizer_per_gpu = 74e9 * 12 / 16  # ZeRO-1, DP=16
activations_per_gpu = 4 * 2048 * 12288 * 2  # Micro-batch activations

total = (model_per_gpu + optimizer_per_gpu + activations_per_gpu) / 1e9
# ≈ 18.5 + 55.5 + 0.8 = 74.8 GB (fits!)

# Training throughput
tokens_per_second = 128 * 2048 * 0.85 / 0.5  # 85% efficiency, 0.5s per iteration
# ≈ 444,416 tokens/second
```

---

## Key Takeaways

**Memory Optimization Hierarchy:**
```
1. Gradient Checkpointing:     33% compute → 75% memory savings
2. ZeRO Stage 1:               8× GPU → 7/8 optimizer memory savings
3. ZeRO Stage 2:               + gradient partitioning
4. ZeRO Stage 3:               + parameter partitioning (86% total savings)
5. ZeRO-Infinity:              + CPU/NVMe offload (trillion-scale)
```

**Parallelism Strategy Selection:**
```
Model < 13B:         Data Parallelism only
Model 13-175B:       DP + TP (NVLink required)
Model 175B-1T:       DP + TP + PP (3D parallelism)
Model > 1T:          DP + TP + PP + ZeRO-Infinity
```

**Communication Requirements:**
```
Tensor Parallelism:    NVLink (600 GB/s) within node
Pipeline Parallelism:  NVLink or InfiniBand between nodes
Data Parallelism:      InfiniBand across all nodes
```

**ARR-COC Specific:**
```
Stage 1 (Texture):         Gradient checkpointing (13 channels)
Stage 2 (Scorers):         Mixed precision (BF16 for stability)
Stage 3 (Quality Adapter): End-to-end with gradient accumulation
Future Scale:              3D parallelism for 70B+ VLMs
```

---

## Sources

**NVIDIA Documentation:**
- [Activation Recomputation — NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html) (accessed 2025-02-03)

**DeepSpeed Documentation:**
- [Zero Redundancy Optimizer - DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) (accessed 2025-02-03)

**Academic Papers:**
- [Optimizing Intermediate Memory for Long Sequences](https://proceedings.neurips.cc/paper_files/paper/2024/file/b063829b922fdeb4fa3472dd3471ff43-Paper-Conference.pdf) - NeurIPS 2024 (accessed 2025-02-03)
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) - Megatron-LM paper (accessed 2025-02-03)

**Blog Posts and Tutorials:**
- [The Unreasonable Impact of Gradient Checkpointing for Fine-Tuning](https://kaitchup.substack.com/p/gradient-checkpointing-llms) - Kaitchup Substack (accessed 2025-02-03)
- [Training Transformer Models: Fundamentals and Memory Optimization](https://medium.com/@imabhi1216/training-transformer-models-fundamentals-and-memory-challenges-61679948379a) - Medium (accessed 2025-02-03)
- [A Comprehensive Overview of Optimization Techniques](https://huggingface.co/blog/Isayoften/optimization-rush) - Hugging Face Blog (accessed 2025-02-03)

**Community Resources:**
- [arXiv: Adacc: Adaptive Compression and Activation Recomputation](https://arxiv.org/html/2508.00806v1) (accessed 2025-02-03)
- [Megatron-LM (2) | Zhuo's Blog](https://zhuogege1943.com/blogs/2024/12/26/Megatron-LM-2/) (accessed 2025-02-03)

**Source Documents:**
- [cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md) - GPU memory hierarchy
- [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) - Mixed precision training
- [vertex-ai-production/00-distributed-training-patterns.md](../vertex-ai-production/00-distributed-training-patterns.md) - Distributed training basics
- [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) - GPU specifications

---

**Document Statistics:**
- **Created**: 2025-02-03
- **Lines**: ~850
- **Coverage**: Gradient checkpointing, Pipeline parallelism, ZeRO stages 1-3, Tensor parallelism, 3D parallelism, ARR-COC integration
