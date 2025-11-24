# DeepSpeed ZeRO: Zero Redundancy Optimizer for Large-Scale Training

**Deep dive into memory optimization for trillion-parameter models**

From [DeepSpeed Official Tutorial](https://www.deepspeed.ai/tutorials/zero/) (accessed 2025-11-13):
> "ZeRO is a powerful set of memory optimization techniques that enable effective training of large models with trillions of parameters. Using ZeRO in a DeepSpeed model is quick and easy because all you need is to change a few configurations in the DeepSpeed configuration JSON."

---

## Section 1: ZeRO Overview - What It Is and Why It Exists (~100 lines)

### The Memory Problem

**Standard Data Parallelism Redundancy:**

For a 175B parameter model (GPT-3 scale):
```
FP16 model parameters:        2 × 175B = 350 GB
FP32 optimizer states:
  - FP32 master weights:      4 × 175B = 700 GB
  - First moment (momentum):  4 × 175B = 700 GB
  - Second moment (variance): 4 × 175B = 700 GB
Total optimizer memory:       12 × 175B = 2.1 TB

Traditional 8-GPU Data Parallel:
Each GPU stores: 350 GB + 2.1 TB = 2.45 TB
Total memory: 8 × 2.45 TB = 19.6 TB
Redundancy: 8× duplication of identical data!
```

From [Microsoft Research ZeRO Paper](https://arxiv.org/abs/1910.02054) (2,047 citations):
> "ZeRO leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device used for model training."

### What ZeRO Does Differently

**Core Innovation**: Partition model states across GPUs instead of replicating them

**Three Incremental Stages:**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):

1. **ZeRO-1 (Stage 1)**: Optimizer state partitioning
   - Each GPU stores only 1/N of optimizer states
   - 4× memory reduction for optimizer states
   - Maintains full model copy on each GPU

2. **ZeRO-2 (Stage 2)**: + Gradient partitioning
   - Also partition gradients across GPUs
   - 8× memory reduction (optimizer + gradients)
   - Still maintains full model copy

3. **ZeRO-3 (Stage 3)**: + Parameter partitioning
   - Partition model parameters themselves
   - N× memory reduction (scales with GPU count)
   - All-gather parameters on-demand during forward/backward

### ZeRO vs Model Parallelism

**Key Distinction:**

| Approach | Code Changes | Memory Savings | Communication | Scalability |
|----------|--------------|----------------|---------------|-------------|
| **Model Parallelism** | Extensive | High | Complex | Manual sharding |
| **ZeRO** | None (config only) | High | Automated | Linear scaling |

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "A key appeal of ZeRO is that no model code modifications are required."

### When to Use ZeRO

**Decision Matrix:**

```
Model Size < 7B params:
  └─> ZeRO-0 (standard DDP) if fits in memory
      └─> Highest throughput, simplest setup

Model Size 7-13B params:
  └─> ZeRO-2 for balanced performance
      └─> 40% memory reduction, 95% throughput

Model Size > 13B params:
  └─> ZeRO-3 required
      └─> 65% memory reduction, 78% throughput
```

From [Hugging Face ZeRO Performance Analysis](https://huggingface.co/blog/josh-a/zero-optimization-strategies) (accessed 2025-11-13):
> "ZeRO-3 enables training of significantly larger models on the same hardware, with a 2.7× reduction in memory requirements compared to standard DDP."

---

## Section 2: ZeRO-1, ZeRO-2, ZeRO-3 Detailed Comparison (~150 lines)

### ZeRO-1: Optimizer State Partitioning

**Algorithm:**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Stage 1: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition."

**Memory Breakdown (8 GPUs, 175B model):**

```python
# Without ZeRO-1:
per_gpu_memory = {
    "model_fp16": 350_GB,
    "optimizer_states": 2100_GB,  # 12 × params
    "total": 2450_GB
}

# With ZeRO-1:
per_gpu_memory = {
    "model_fp16": 350_GB,
    "optimizer_states": 262.5_GB,  # 2100 / 8
    "total": 612.5_GB
}
# Savings: 75% reduction!
```

**Implementation Pattern:**

```python
# DeepSpeed ZeRO-1 Config
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8  # Gradient reduction bucket
    }
}

# Training Loop (no code changes needed)
def backward_with_zero_stage1(model, loss, optimizer):
    # 1. Compute gradients (normal backward)
    loss.backward()

    # 2. All-reduce gradients (sum across all ranks)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    # 3. Each rank updates only its partition
    rank = dist.get_rank()
    for i, param in enumerate(model.parameters()):
        if i % world_size == rank:
            optimizer.step_single_param(param)

    # 4. All-gather updated parameters
    for param in model.parameters():
        dist.all_gather(param.data)
```

**Communication Pattern:**
- **Forward**: No communication (all ranks have full model)
- **Backward**: All-reduce gradients (same as DDP)
- **Optimizer**: All-gather parameters after update
- **Overhead**: ~5% vs standard DDP

### ZeRO-2: Gradient Partitioning

**Additional Optimization:**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Stage 2: The reduced 16-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states."

**Memory Breakdown (8 GPUs, 175B model):**

```python
# ZeRO-2 adds gradient partitioning:
per_gpu_memory = {
    "model_fp16": 350_GB,
    "gradients_fp16": 43.75_GB,    # 350 / 8 (partitioned!)
    "optimizer_states": 262.5_GB,   # 2100 / 8
    "total": 656.25_GB
}
# Additional 6% savings vs ZeRO-1
```

**Reduce-Scatter Algorithm:**

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

**Communication Pattern:**
- **Forward**: No communication
- **Backward**: Reduce-scatter gradients (more efficient than all-reduce)
- **Optimizer**: All-gather parameters
- **Overhead**: ~6% vs standard DDP

**Configuration:**

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "We have enabled contiguous_gradients to reduce memory fragmentation during backward pass."

### ZeRO-3: Full Parameter Partitioning

**Complete State Sharding:**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Stage 3: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes."

**Memory Breakdown (8 GPUs, 175B model):**

```python
# ZeRO-3: Everything partitioned
per_gpu_memory = {
    "model_fp16": 43.75_GB,         # 350 / 8 (partitioned!)
    "gradients_fp16": 43.75_GB,     # 350 / 8 (partitioned!)
    "optimizer_states": 262.5_GB,   # 2100 / 8 (partitioned!)
    "total": 350_GB
}
# vs No ZeRO: 2.45 TB → 350 GB = 86% reduction!
```

**All-Gather on Demand:**

```python
def forward_with_zero_stage3(model, input):
    # Layer-by-layer parameter gathering
    for layer in model.layers:
        # 1. All-gather parameters for current layer
        all_gather_layer_params(layer)

        # 2. Forward through layer
        output = layer(input)

        # 3. Partition parameters again (free memory)
        partition_layer_params(layer)

        input = output

    return output

def backward_with_zero_stage3(model, loss):
    # Similar: all-gather → backward → partition
    for layer in reversed(model.layers):
        all_gather_layer_params(layer)
        backward_through_layer(layer)
        partition_layer_params(layer)
```

**Communication Pattern:**
- **Forward**: All-gather parameters layer-by-layer
- **Backward**: All-gather parameters + reduce-scatter gradients
- **Optimizer**: Local update (no communication)
- **Overhead**: ~22% vs standard DDP (4× communication volume)

From [Hugging Face Performance Analysis](https://huggingface.co/blog/josh-a/zero-optimization-strategies):
> "The performance degradation in ZeRO-3 primarily stems from increased all-gather operations for parameter reconstruction. However, on H100 systems with NVSwitch, the high-bandwidth interconnect (~900 GB/s) significantly mitigates this overhead."

**Configuration:**

```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "reduce_bucket_size": 1e7,
        "sub_group_size": 1e9
    }
}
```

### Stage Comparison Table

| Feature | ZeRO-0 (DDP) | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---------|--------------|--------|--------|--------|
| **Parameters** | Replicated | Replicated | Replicated | Partitioned |
| **Gradients** | Replicated | Replicated | Partitioned | Partitioned |
| **Optimizer** | Replicated | Partitioned | Partitioned | Partitioned |
| **Memory/GPU (175B)** | 2.45 TB | 612 GB | 656 GB | 350 GB |
| **Relative Throughput** | 100% | 95% | 95% | 78% |
| **Max Model Size (8×H100)** | ~7B | ~15B | ~15B | ~30B |
| **Communication** | All-reduce | All-reduce + All-gather | Reduce-scatter + All-gather | All-gather (4×) |

---

## Section 3: Memory Optimization Breakdown (~100 lines)

### Understanding Model State Components

**Complete Memory Footprint:**

```python
# For P parameters with Adam optimizer:
memory_components = {
    # Model (forward/backward)
    "fp16_params": 2 * P,              # 2 bytes per param
    "fp16_gradients": 2 * P,           # 2 bytes per param
    "activations": varies_by_batch,    # Depends on batch size

    # Optimizer (Adam)
    "fp32_params": 4 * P,              # Master weights
    "fp32_momentum": 4 * P,            # First moment
    "fp32_variance": 4 * P,            # Second moment

    # Total
    "optimizer_total": 12 * P,
    "model_total": 4 * P + activations
}

# Example: 175B parameters
total_memory = 16 * 175e9 + activations  # ~2.8 TB + activations
```

### ZeRO Memory Savings Per Stage

**Mathematical Formulas:**

From [ZeRO Paper](https://arxiv.org/abs/1910.02054):

**ZeRO-1 Memory per GPU:**
```
M1 = 2Φ + (2Φ + K×Φ)/Nd
```
Where:
- Φ = model parameters
- K = optimizer state multiplier (12 for Adam)
- Nd = data parallel degree (number of GPUs)

**ZeRO-2 Memory per GPU:**
```
M2 = (2Φ + 2Φ + K×Φ)/Nd
```

**ZeRO-3 Memory per GPU:**
```
M3 = (2Φ + 2Φ + K×Φ)/Nd = 16Φ/Nd
```

**Concrete Example (8 GPUs, 10B params):**

```python
params = 10e9
Nd = 8

# ZeRO-0 (standard DDP)
mem_zero0 = 2*params + 2*params + 12*params  # 160 GB
print(f"ZeRO-0: {mem_zero0/1e9:.1f} GB per GPU")

# ZeRO-1
mem_zero1 = 2*params + 2*params + (12*params/Nd)  # 55 GB
print(f"ZeRO-1: {mem_zero1/1e9:.1f} GB per GPU")

# ZeRO-2
mem_zero2 = 2*params + (2*params/Nd) + (12*params/Nd)  # 37.5 GB
print(f"ZeRO-2: {mem_zero2/1e9:.1f} GB per GPU")

# ZeRO-3
mem_zero3 = (16*params)/Nd  # 20 GB
print(f"ZeRO-3: {mem_zero3/1e9:.1f} GB per GPU")
```

**Output:**
```
ZeRO-0: 160.0 GB per GPU
ZeRO-1: 55.0 GB per GPU (66% reduction)
ZeRO-2: 37.5 GB per GPU (77% reduction)
ZeRO-3: 20.0 GB per GPU (87% reduction)
```

### Activation Memory vs Model State

**Critical Distinction:**

Activations are NOT partitioned by ZeRO (they're different per data sample):

```python
# Activation memory (forward pass, per sample):
activation_memory = {
    "transformer_layers": num_layers * hidden_size * sequence_length,
    "attention_scores": num_heads * sequence_length^2,
    "residual_connections": varies
}

# Techniques to reduce activation memory:
techniques = [
    "Gradient checkpointing (75% reduction, 33% compute overhead)",
    "Selective activation recomputation",
    "Flash Attention (reduces attention memory O(N²) → O(N))"
]
```

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):
> "Activation checkpointing enabled: --checkpoint-activations"

### Communication Volume Analysis

**Bandwidth Requirements by Stage:**

```python
# Per training iteration for P parameters:
communication_volume = {
    "ZeRO-0": {
        "forward": 0,
        "backward": 2 * P,  # All-reduce gradients
        "total": 2 * P
    },
    "ZeRO-1": {
        "forward": 0,
        "backward": 2 * P,  # All-reduce gradients
        "optimizer": 2 * P,  # All-gather parameters
        "total": 4 * P
    },
    "ZeRO-2": {
        "forward": 0,
        "backward": 2 * P,  # Reduce-scatter gradients
        "optimizer": 2 * P,  # All-gather parameters
        "total": 4 * P
    },
    "ZeRO-3": {
        "forward": 2 * P,   # All-gather parameters
        "backward": 4 * P,  # All-gather params + reduce-scatter grads
        "total": 6 * P
    }
}

# Example: 10B params, 8 GPUs, NVLink 900 GB/s
params_gb = 10 * 2 / 8  # 2.5 GB per GPU (fp16)
comm_time_zero3 = (6 * params_gb) / 900  # ~17 ms per iteration
```

From [Hugging Face Analysis](https://huggingface.co/blog/josh-a/zero-optimization-strategies):
> "On H100 systems with NVSwitch, the high-bandwidth interconnect (~900 GB/s) significantly mitigates ZeRO-3's overhead compared to PCIe or network-joined systems."

---

## Section 4: PyTorch Integration & Code Examples (~100 lines)

### Minimal DeepSpeed ZeRO Setup

**Step 1: Configuration File (ds_config.json)**

```json
{
    "train_batch_size": 32,
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

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

**Step 2: PyTorch Training Script**

From [DeepSpeed GitHub Examples](https://github.com/deepspeedai/DeepSpeed) (accessed 2025-11-13):

```python
import torch
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel

# Standard PyTorch model
config = GPT2Config(n_layer=48, n_head=16, n_embd=1600)
model = GPT2LMHeadModel(config)

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# Training loop (identical to standard PyTorch!)
for batch in dataloader:
    inputs, labels = batch

    # Forward pass
    outputs = model_engine(inputs, labels=labels)
    loss = outputs.loss

    # Backward pass (DeepSpeed handles ZeRO internally)
    model_engine.backward(loss)

    # Optimizer step (DeepSpeed handles parameter gathering)
    model_engine.step()
```

**That's it!** No changes to model code. ZeRO is transparent.

### ZeRO-3 Specific Patterns

**Model Initialization for Large Models:**

From [DeepSpeed ZeRO-3 Documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html):

```python
import deepspeed
from transformers import AutoModelForCausalLM

# For models > GPU memory, initialize with ZeRO-3 context
with deepspeed.zero.Init(
    data_parallel_group=None,  # Auto-detect
    remote_device="cpu",       # Allocate in CPU initially
    enabled=True               # Enable ZeRO-3
):
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-176b",  # 176B parameters!
        torch_dtype=torch.float16
    )

# DeepSpeed will automatically partition parameters
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config="ds_zero3_config.json"
)
```

**Manual Parameter Gathering (Advanced):**

```python
# If you need to access full parameters (e.g., for initialization)
from deepspeed.zero import GatheredParameters

# Gather weight for custom initialization
with GatheredParameters(
    model.layer.weight,
    modifier_rank=0  # Only rank 0 modifies
):
    if torch.distributed.get_rank() == 0:
        torch.nn.init.xavier_uniform_(model.layer.weight)
    # DeepSpeed will automatically partition after context exit
```

### Multi-Node Training

**Launch Script:**

```bash
# 4 nodes, 8 GPUs each = 32 total GPUs
deepspeed --num_gpus=8 \
          --num_nodes=4 \
          --master_addr=node1.cluster \
          --master_port=29500 \
          --hostfile=hostfile \
          train.py \
          --deepspeed \
          --deepspeed_config ds_zero2_config.json
```

**Hostfile Format:**

```
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

### Checkpoint Saving and Loading

**ZeRO-2 Checkpointing (Simple):**

```python
# Save checkpoint
model_engine.save_checkpoint(save_dir="./checkpoints", tag="epoch_1")

# Load checkpoint
_, client_sd = model_engine.load_checkpoint(
    load_dir="./checkpoints",
    tag="epoch_1"
)
```

**ZeRO-3 Checkpointing (Gather Full State):**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/zero/):

```python
# Enable 16-bit weight gathering in config
{
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}

# Save consolidated fp16 weights
model_engine.save_16bit_model(output_dir="./model", output_file="pytorch_model.bin")
```

**Convert ZeRO Checkpoint to Standard PyTorch:**

```bash
# Offline consolidation (no GPU needed)
cd /path/to/checkpoint_dir
./zero_to_fp32.py . pytorch_model.bin

# Output: Standard PyTorch checkpoint
```

---

## Section 5: arr-coc-0-1 VLM Training Use Cases (~50 lines)

### Qwen3-VL + ARR-COC Architecture

**Memory Requirements:**

```python
# Qwen3-VL-2B base model
qwen_base_memory = {
    "model_params": 2e9 * 2,      # 4 GB (fp16)
    "optimizer": 2e9 * 12,        # 24 GB (Adam)
    "total": 28_GB
}

# ARR-COC additional components
arr_coc_memory = {
    "texture_arrays": 13 * 256 * 256 * 4,  # 13-channel texture
    "relevance_scorers": 3 * 100e6 * 2,     # 3 scorers × 100M params
    "quality_adapter": 50e6 * 2,            # LoRA adapter
    "total": ~2_GB
}

# Total: ~30 GB per GPU without ZeRO
# With ZeRO-2 on 4 GPUs: ~10 GB per GPU
```

### Recommended Configuration for arr-coc-0-1

**ZeRO-2 for 4×A100 (40GB):**

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,

    "fp16": {"enabled": true},

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": false
}
```

**Why ZeRO-2 for arr-coc-0-1:**
- Model fits comfortably with ZeRO-2 (10 GB vs 40 GB available)
- ~95% throughput vs standard DDP
- Simpler debugging than ZeRO-3
- Room for larger batches to test texture array variants

### Scaling to Larger VLMs

**If training Qwen3-VL-7B with ARR-COC (8×A100):**

```python
# Memory estimation
qwen7b_memory = {
    "model_params": 7e9 * 2,      # 14 GB
    "optimizer": 7e9 * 12,        # 84 GB
    "arr_coc": 2_GB,
    "total": 100_GB
}

# Requires ZeRO-3:
per_gpu_zero3 = 100_GB / 8  # ~12.5 GB per GPU
```

**ZeRO-3 Configuration:**

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

From [Hugging Face Analysis](https://huggingface.co/blog/josh-a/zero-optimization-strategies):
> "For production AI workloads, the optimal ZeRO stage depends on the specific use case: ZeRO-2 optimal sweet spot for 9-13B models with 95% throughput retention and 40% memory reduction."

---

## Sources

**Official Documentation:**
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/) - Official implementation guide (accessed 2025-11-13)
- [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/) - Complete config reference
- [DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed) - Source code and examples

**Research Papers:**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - Rajbhandari et al. (2,047 citations)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) - Ren et al. (USENIX ATC 2021)
- [ZeRO-Infinity: Breaking the GPU Memory Wall](https://arxiv.org/abs/2104.07857) - Rajbhandari et al.

**Performance Analysis:**
- [ZeRO Optimization Strategies for Large-Scale Model Training](https://huggingface.co/blog/josh-a/zero-optimization-strategies) - Hugging Face blog (accessed 2025-11-13)
- [Memory Optimizations Toward Training Trillion Parameter Models](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/richards_zero.pdf) - University of Toronto presentation

**Integration Guides:**
- [PyTorch ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html) - PyTorch official tutorial
- [Hugging Face Transformers + DeepSpeed](https://huggingface.co/docs/transformers/main_classes/deepspeed) - Integration guide
