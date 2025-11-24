# PyTorch FSDP vs DeepSpeed ZeRO: Distributed Training Framework Comparison

## Overview

PyTorch Fully Sharded Data Parallel (FSDP) and Microsoft DeepSpeed ZeRO are the two dominant frameworks for memory-efficient distributed training of large models. Both enable training models that cannot fit on a single GPU by sharding model states across workers, but they differ in implementation details, flexibility, and ecosystem integration.

**Key Insight**: FSDP and DeepSpeed ZeRO-3 are functionally equivalent in their core sharding strategy - both shard parameters, gradients, and optimizer states across all workers. The choice between them often comes down to ecosystem preferences (PyTorch native vs Microsoft tooling) and specific feature requirements.

**When This Matters**: Training models >10B parameters where a single GPU cannot hold the full model, optimizer states, and gradients. For arr-coc-0-1 VLM training, understanding these frameworks is critical for scaling to production datasets.

**Related Knowledge**:
- See [llm-gpu-integration/02-training-dynamics-gpu.md](../llm-gpu-integration/02-training-dynamics-gpu.md) for ZeRO stages 1-3 deep dive
- See [practical-implementation/69-gke-autopilot-ml-workloads.md](../practical-implementation/69-gke-autopilot-ml-workloads.md) for multi-GPU deployment

From [HuggingFace Accelerate Documentation](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed) (accessed 2025-11-13):
> The aim of this tutorial is to draw parallels, as well as to outline potential differences, to empower the user to switch seamlessly between these two frameworks.

---

## Section 1: FSDP Overview (80 lines)

### 1.1 What is FSDP?

PyTorch FSDP is a **native PyTorch** implementation of fully sharded data parallelism, introduced in PyTorch 1.11 (March 2022). It borrows heavily from FairScale's FSDP while adding streamlined APIs and performance improvements.

From [PyTorch FSDP Blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) (accessed 2025-11-13):
> FSDP is a type of data-parallel training, but unlike traditional data-parallel, which maintains a per-GPU copy of a model's parameters, gradients and optimizer states, it shards all of these states across data-parallel workers and can optionally offload the sharded model parameters to CPUs.

**Core Mechanism**: FSDP wraps model layers in a nested fashion, gathering full parameters only when needed for computation, then immediately discarding them.

### 1.2 FSDP Sharding Strategies

PyTorch FSDP supports three sharding strategies:

**1. FULL_SHARD (equivalent to ZeRO-3)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

model = FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
)
```
- Shards parameters, gradients, optimizer states across all workers
- Maximum memory savings (N× reduction for N GPUs)
- 1.5× communication volume vs DDP

**2. SHARD_GRAD_OP (equivalent to ZeRO-2)**:
```python
model = FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2 equivalent
)
```
- Shards gradients and optimizer states only
- Parameters replicated across workers
- Same communication volume as DDP

**3. HYBRID_SHARD**:
```python
model = FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```
- Shards within nodes, replicates across nodes
- Reduces expensive inter-node communication
- Critical for multi-node training without InfiniBand

### 1.3 FSDP Workflow

From [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html):

**Forward Pass**:
1. Run `all_gather` to collect parameter shards from all ranks
2. Run forward computation
3. Discard gathered parameters

**Backward Pass**:
1. Run `all_gather` to collect parameter shards
2. Run backward computation
3. Run `reduce_scatter` to sync gradients
4. Discard parameters

**Memory Savings**: For L layers, activation memory drops from O(L) to O(√L) with gradient checkpointing.

### 1.4 FSDP Configuration Example

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Auto-wrap policy: wrap layers with >100M parameters
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000,
)

model = FullyShardedDataParallel(
    model,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
)
```

**Performance on AWS (from PyTorch blog)**:
- GPT-1T (1 trillion parameters): 84 TFLOPS per A100 GPU
- GPT-175B: 159 TFLOPS per A100 GPU (51% of peak 312 TFLOPS)
- Batch size 20, sequence length 512, 128 GPUs

---

## Section 2: DeepSpeed Overview (80 lines)

### 2.1 What is DeepSpeed?

DeepSpeed is Microsoft's deep learning optimization library featuring the **ZeRO (Zero Redundancy Optimizer)** family of memory optimization techniques. DeepSpeed provides a broader toolkit beyond just ZeRO, including inference optimizations, sparse attention, and pipeline parallelism.

From [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054):
> ZeRO powers DP to fit models with arbitrary size as long as there are sufficient number of devices to share the model states.

**Key Advantage**: DeepSpeed offers more flexibility in offloading strategies and has been battle-tested at massive scale (training 1T+ parameter models).

### 2.2 DeepSpeed ZeRO Stages

**ZeRO Stage 1 (Optimizer State Partitioning)**:
```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```
- Partitions optimizer states only
- 4× memory reduction (example-specific)
- Same communication volume as DDP

**ZeRO Stage 2 (+ Gradient Partitioning)**:
```json
{
    "zero_optimization": {
        "stage": 2
    }
}
```
- Partitions optimizer states + gradients
- 8× memory reduction
- Same communication volume as DDP

**ZeRO Stage 3 (+ Parameter Partitioning)**:
```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto"
    }
}
```
- Partitions parameters + gradients + optimizer states
- 64× memory reduction (example-specific)
- 1.5× communication volume vs DDP

### 2.3 DeepSpeed Enhancements

**ZeRO-Offload (CPU Offloading)**:
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```
- Offloads optimizer computations to CPU
- Achieves 40 TFLOPS on 1 V100 for 10B model (vs 30 TFLOPS max with DDP)

**ZeRO-Infinity (NVMe Offloading)**:
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```
- Offloads to NVMe SSD storage
- Enables 10-100T parameter training on single node
- Requires fast NVMe bandwidth (>3GB/s recommended)

**ZeRO++ (2023 Improvements)**:
- Quantized weights (qwZ): INT8 weights → 2× communication reduction
- Hierarchical partitioning (hpZ): Hybrid sharding (shard within node, replicate across nodes)
- Quantized gradients (qgZ): INT4 gradients → additional 2× reduction
- **Total**: 4× communication volume reduction vs ZeRO-3

From [sumanthrh.com guide](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) (accessed 2025-11-13):
> ZeRO++ reduces communication volume by 4x with these three improvements, compared to ZeRO-3.

---

## Section 3: Feature Comparison Table (100 lines)

### 3.1 Core Functionality Comparison

From [HuggingFace Accelerate docs](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed):

| Feature | FSDP | DeepSpeed | Notes |
|---------|------|-----------|-------|
| **Sharding/Partitioning** | `--fsdp_sharding_strategy` | `--zero_stage` | FULL_SHARD = ZeRO-3 |
| **Offloading** | `--fsdp_offload_params` (all-or-nothing) | `--offload_param_device`, `--offload_optimizer_device` (flexible) | DeepSpeed allows separate param/optimizer offload |
| **Model Loading** | `--fsdp_cpu_ram_efficient_loading` | `--zero3_init_flag` (automatic with transformers) | Both reduce RAM during initialization |
| **Checkpointing** | `--fsdp_state_dict_type` (SHARDED/FULL) | `--zero3_save_16bit_model` | FSDP more flexible |
| **Prefetching** | `--fsdp_forward_prefetch`, `--fsdp_backward_prefetch` | Automatic based on hyper-params | FSDP explicit control |
| **Auto-wrapping** | `--fsdp_auto_wrap_policy` | Not needed (transparent) | FSDP requires explicit policy |
| **torch.compile** | `--fsdp_use_orig_params` required | Transparent | FSDP needs flag for compile |
| **Gradient Accumulation** | Transparent | `--gradient_accumulation_steps "auto"` | DeepSpeed needs explicit config |

### 3.2 Offloading Differences

**FSDP**: All-or-nothing offload
```python
cpu_offload = CPUOffload(offload_params=True)  # Offloads params, grads, optimizer
```

**DeepSpeed**: Granular control
```json
{
    "offload_optimizer": {"device": "cpu"},  // Optimizer only
    "offload_param": {"device": "nvme"}      // Parameters to NVMe
}
```

**Why This Matters**: DeepSpeed can offload optimizer to CPU while keeping parameters on GPU, or vice versa. FSDP forces you to offload everything or nothing.

### 3.3 Checkpointing Strategies

**FSDP Sharded Checkpointing** (recommended for large models):
```python
from torch.distributed.fsdp import StateDictType

# Save sharded checkpoint (fast)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    checkpoint = model.state_dict()
    torch.save(checkpoint, f"checkpoint_rank_{rank}.pt")
```

**DeepSpeed ZeRO-3 Checkpointing**:
```python
# Automatic consolidation to rank 0
model_engine.save_checkpoint(save_dir)

# Or use zero_to_fp32.py script for post-conversion
# python zero_to_fp32.py checkpoint_dir output.pt
```

From [HuggingFace docs](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed):
> For large models, consolidating the model to a single rank can be very slow. For quicker checkpointing, for FSDP use `fsdp_state_dict_type: SHARDED_STATE_DICT`, and for DeepSpeed Zero3 use the `zero_to_fp32.py` script to post-convert sharded checkpoints.

### 3.4 Multi-Node Communication Requirements

For model size M, N nodes, G GPUs per node:

**Communication per training step**: ~48 × M/N bits per node

**Example**: 40B model, 2 nodes (16 GPUs total)
- 48 × 40B / 2 = 960 Gb = **120 GB per node per step**

**Network Requirements** (from Stas Bekman's investigations):
- **Large scale (64+ GPUs)**: InfiniBand (1000 Gbps) required
- **Small scale (8-32 GPUs)**: EFA/Ethernet (100-400 Gbps) acceptable
- **AWS P4d instances**: EFA ~340 Gbps (advertised 400 Gbps)

**Solution for limited bandwidth**: Use hybrid/hierarchical sharding
- **FSDP**: `HYBRID_SHARD` strategy
- **DeepSpeed**: `zero_hpz_partition_size` (ZeRO++)

### 3.5 Precision Handling Differences

From [HuggingFace comparison](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed):

| Process | FSDP | DeepSpeed |
|---------|------|-----------|
| **Preparation (flat params)** | Created in `torch_dtype` | Created in FP32 (ignores `torch_dtype`) |
| **Optimizer initialization** | Uses `torch_dtype` | Always FP32 |
| **Training step** | Follows `MixedPrecision` config | Follows `deepspeed_config_file` |
| **Optimizer step** | Occurs in `torch_dtype` | Always FP32 |

**Memory Implication**: DeepSpeed upcasts to FP32 during preparation, potentially causing higher RAM usage with small GPU counts.

**Checkpoint Precision**:
- **FSDP**: Saves in upcasted precision (FP32 if using mixed precision)
- **DeepSpeed**: Can save low precision checkpoints with `--zero3_save_16bit_model`

---

## Section 4: Performance Benchmarks (80 lines)

### 4.1 Scaling Performance

**PyTorch FSDP on AWS** (from PyTorch blog, March 2022):

GPT-175B Model (96 layers, 12288 hidden, 96 heads):
- **Peak throughput**: 159 TFLOPS per A100 (51% of 312 TFLOPS peak)
- **Configuration**: Batch size 20, sequence length 512, 128 GPUs
- **Scaling**: Throughput degraded beyond 128 GPUs due to communication overhead

GPT-1T Model (128 layers, 25600 hidden, 160 heads):
- **Peak throughput**: 84 TFLOPS per A100 (27% of peak)
- **Configuration**: Batch size 4, sequence length 2048, 128 GPUs
- **Bottleneck**: CUDA cache allocator (A100 80GB would improve this)

**DeepSpeed ZeRO Scaling** (from ZeRO paper, 2020):

7.5B Parameter Model on 64 GPUs:
- **ZeRO-1**: 4× memory reduction, same communication as baseline
- **ZeRO-2**: 8× memory reduction, same communication as baseline
- **ZeRO-3**: 64× memory reduction, 1.5× communication vs baseline

### 4.2 Throughput Comparison

From [sumanthrh.com comprehensive guide](https://sumanthrh.com/post/distributed-and-efficient-finetuning/):

**Falcon-180B on 1 × 8xA100-80GB Node** (DeepSpeed + LoRA + Flash Attention):
- **Training time**: 153 minutes for 3 epochs
- **Dataset**: 15K samples (Dolly)
- **Effective batch size**: 64
- **Sequence length**: 2048 tokens
- **Throughput**: >5 samples/second

**Memory Efficiency Comparison** (from DeepSpeed benchmarks):

| Method | Memory per GPU | Max Model Size (8× A100-40GB) |
|--------|----------------|-------------------------------|
| PyTorch DDP | 40 GB | 1.4B params |
| ZeRO-1 | 10 GB | 6B params |
| ZeRO-2 | 5 GB | 12B params |
| ZeRO-3 | 2.5 GB | 40B+ params |
| ZeRO-3 + Offload | 1 GB | 100B+ params |

### 4.3 Communication Overhead

**FSDP FULL_SHARD** (equivalent to ZeRO-3):
```
Communication volume = 3Ψ (where Ψ = number of parameters)
- Ψ for forward all-gather (parameters)
- Ψ for backward reduce-scatter (gradients)
- Ψ for all-gather (updated parameters)
Total = 1.5× DDP baseline (2Ψ)
```

**DeepSpeed ZeRO++** optimizations:
```
With qwZ + qgZ:
- 0.5Ψ for forward all-gather (INT8 quantized weights)
- 0.25Ψ for backward reduce-scatter (INT4 quantized gradients)
- 0.5Ψ for all-gather (updated weights)
Total = 1.25Ψ = 0.625× DDP baseline
```

### 4.4 Real-World Training Costs

**GPT-3 175B Training** (from OpenAI paper):
- **Global batch size**: 2048
- **Tokens**: 300B
- **Hardware**: ~10,000 V100 GPUs

**BLOOM-176B Training** (from BLOOM paper):
- **Global batch size**: 2048
- **Tokens**: 366B
- **Hardware**: 384 A100-80GB GPUs
- **Training time**: ~3.5 months
- **Framework**: Megatron-DeepSpeed (3D parallelism)

From [sumanthrh.com](https://sumanthrh.com/post/distributed-and-efficient-finetuning/):
> For large scale training (64GPUs+), you really do need InfiniBand interconnect with 1000 Gbps. For smaller-scale multi-node training, you can get away with 100-400 Gbps.

---

## Section 5: When to Use Which (60 lines)

### 5.1 Use FSDP When:

**1. PyTorch-Native Preference**
- You want minimal external dependencies
- Tight integration with PyTorch ecosystem
- Easier debugging with PyTorch tools (TorchDynamo, NSight)

**2. Single-Node Training**
```bash
# Simple FSDP launch
torchrun --nproc_per_node=8 train.py --use_fsdp
```
- Less configuration overhead than DeepSpeed
- Native PyTorch distributed launch

**3. Migrating from DDP**
```python
# Easy migration path
from torch.nn.parallel import DistributedDataParallel  # Before
from torch.distributed.fsdp import FullyShardedDataParallel  # After
```

**4. Research/Experimentation**
- Rapid prototyping with new architectures
- No need for JSON config files
- Pythonic configuration

From [PyTorch blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/):
> In future PyTorch versions, we're going to enable users to seamlessly switch between DDP, ZeRO-1, ZeRO-2 and FSDP flavors of data parallelism.

### 5.2 Use DeepSpeed When:

**1. Maximum Flexibility Needed**
- Separate optimizer/parameter offload strategies
- NVMe offloading for 100T+ models
- Custom optimization schedules

**2. Multi-Node Production Training**
```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_hpz_partition_size": 8,  // Hierarchical partitioning
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true
    }
}
```
- ZeRO++ hierarchical partitioning for weak inter-node bandwidth
- Battle-tested at Microsoft scale

**3. Inference Optimization**
- DeepSpeed-Inference for deployment
- INT8/FP16 inference kernels
- Continuous batching support

**4. Comprehensive Logging/Monitoring**
```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "logs/"
    },
    "wandb": {
        "enabled": true,
        "project": "my-training"
    }
}
```

### 5.3 Framework-Agnostic Guidelines

**Memory-Constrained (single GPU)**:
- Use ZeRO-2/FSDP SHARD_GRAD_OP if model fits
- Use ZeRO-3/FSDP FULL_SHARD if model doesn't fit
- Add offloading as last resort (big throughput hit)

**Multi-GPU (same node)**:
- FSDP FULL_SHARD or DeepSpeed ZeRO-3
- Enable gradient checkpointing for large models
- Use BF16 mixed precision

**Multi-Node (with InfiniBand)**:
- FSDP FULL_SHARD or DeepSpeed ZeRO-3
- No hierarchical sharding needed

**Multi-Node (without InfiniBand)**:
- FSDP HYBRID_SHARD or DeepSpeed ZeRO++ (hpZ)
- Critical to avoid inter-node parameter communication bottleneck

From [Ben Gubler's comparison](https://www.bengubler.com/posts/2023-08-29-accelerate-deepspeed-fsdp):
> DeepSpeed and FSDP are two different implementations of the same idea: sharding model parameters, gradients, and optimizer states across data parallel workers while maintaining the simplicity of data parallelism.

### 5.4 arr-coc-0-1 Recommendations

**Phase 1 - Initial Training (Vertex AI, 8× A100-80GB)**:
```bash
# Use FSDP for simplicity
accelerate launch --config_file fsdp_config.yaml train.py
```
- FSDP FULL_SHARD with BF16
- Gradient checkpointing enabled
- No offloading needed (sufficient vRAM)

**Phase 2 - Scaling to Multi-Node**:
```bash
# Use DeepSpeed ZeRO++ with hierarchical partitioning
accelerate launch --config_file deepspeed_config.yaml train.py
```
- ZeRO-3 with hpZ (partition within node)
- Quantized weights/gradients (qwZ/qgZ)
- EFA network (400 Gbps) sufficient

**Why**: arr-coc-0-1's texture array processing (13-channel input) benefits from FSDP's simplicity initially, but multi-node scaling requires DeepSpeed's hierarchical partitioning for cost-effective training.

---

## Sources

**Official Documentation**:
- [PyTorch FSDP API](https://pytorch.org/docs/stable/fsdp.html) - PyTorch documentation
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/) - Microsoft DeepSpeed
- [HuggingFace Accelerate: FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed) (accessed 2025-11-13)

**Research Papers**:
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - Rajbhandari et al., 2020
- [ZeRO++: Extremely Efficient Collective Communication](https://arxiv.org/abs/2306.10209) - Microsoft Research, 2023
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) - Meta, 2023

**Blog Posts & Guides**:
- [Introducing PyTorch FSDP API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) - PyTorch Team, March 2022 (accessed 2025-11-13)
- [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) - Sumanth R Hegde, January 2024 (accessed 2025-11-13)
- [Accelerate vs DeepSpeed vs FSDP](https://www.bengubler.com/posts/2023-08-29-accelerate-deepspeed-fsdp) - Ben Gubler, August 2023

**Community Resources**:
- [Stas Bekman's ML Engineering Guide](https://github.com/stas00/ml-engineering) - Model parallelism section
- [HuggingFace Transformers Distributed Training](https://huggingface.co/docs/transformers/perf_train_gpu_many)
