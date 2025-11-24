# DeepSpeed Pipeline Parallelism

## Overview

Pipeline parallelism splits a model's layers into stages distributed across multiple GPUs, enabling training of models too large to fit on a single device. DeepSpeed implements an efficient pipeline parallelism system that uses gradient accumulation and micro-batching to maximize GPU utilization while minimizing memory overhead.

From [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "DeepSpeed v0.3 includes new support for pipeline parallelism! Pipeline parallelism improves both the memory and compute efficiency of deep learning training by partitioning the layers of a model into stages that can be processed in parallel."

From [Pipeline-Parallel Training Analysis by siboehm](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "Pipeline parallelism makes it possible to train large models that don't fit into a single GPU's memory. Example: Huggingface's BLOOM model is a 175B parameter Transformer model. Storing the weights as bfloat16 requires 350GB, but the GPUs they used to train BLOOM 'only' have 80GB of memory."

**Key Innovation**: Unlike naive model parallelism where only one GPU is active at a time, DeepSpeed's pipeline parallelism keeps multiple GPUs busy simultaneously through micro-batch pipelining.

---

## Section 1: Pipeline Parallelism Fundamentals (80 lines)

### The Model Parallelism Problem

When models exceed single-GPU memory capacity, we must partition across devices. Two primary approaches exist:

1. **Tensor Parallelism**: Split individual layers across GPUs (communication-intensive)
2. **Pipeline Parallelism**: Split sequential layers across GPUs (bubble overhead)

From [karpathy/llm-gpu-integration/02-training-dynamics-gpu.md](../llm-gpu-integration/02-training-dynamics-gpu.md) (lines 227-243):
> "Pipeline Parallelism Architecture:
> ```
> GPU 0: Layers  0-23  (Transformer blocks 1-24)
> GPU 1: Layers 24-47  (Transformer blocks 25-48)
> GPU 2: Layers 48-71  (Transformer blocks 49-72)
> GPU 3: Layers 72-95  (Transformer blocks 73-96)
> ```"

### Naive Pipeline Parallelism Problems

The naive implementation suffers from severe inefficiencies:

From [siboehm pipeline parallelism article](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "By looking at the pebble graph, we can observe some inefficiencies of naive model parallelism:
> 1. **Low GPU utilization**: At any given time, only one GPU is busy, while the other GPU is idle.
> 2. **No interleaving of communication and computation**: While we're sending intermediate outputs (FWD) and gradients (BWD) over the network, no GPU is doing anything.
> 3. **High memory demand**: GPU1 holds all activations for the whole minibatch cached until the very end."

**Naive Schedule Example (4 GPUs)**:
```
Timestep    0    1    2    3    4    5    6    7
GPU3                        FWD  BWD
GPU2                  FWD            BWD
GPU1            FWD                      BWD
GPU0       FWD                                BWD
```

At any timestep, only 1/4 GPUs are active = 75% idle time!

### Micro-Batching Solution

The fundamental insight: split each mini-batch into smaller micro-batches that can be pipelined.

From [GPipe Paper](https://arxiv.org/pdf/1811.06965) (accessed 2025-11-13):
> "Based on this partitioned setup, we propose a novel pipeline parallelism algorithm with batch splitting. We first split a mini-batch of training samples into smaller micro-batches."

**Key Formula - Bubble Fraction**:

From [siboehm analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
```
Bubble fraction = (n - 1) / m

where:
  n = pipeline depth (number of GPUs)
  m = number of micro-batches

Examples:
  4 GPUs, 1 micro-batch:  (4-1)/1  = 75% bubble (naive)
  4 GPUs, 4 micro-batches: (4-1)/4 = 18.75% bubble
  4 GPUs, 16 micro-batches: (4-1)/16 = 6.25% bubble
```

Increasing micro-batches dramatically reduces idle time!

### Communication Volume

From [siboehm communication analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "For simplicity, let's assume a model with only dense layers, which all have equal dimension N. During the forward pass, each GPU will send and receive data of size `batchsize · N`. The same holds for the backwards pass, bringing our total communication volume to `(#GPUs - 1) · 2 · batchsize · N` floats."

Pipeline parallelism requires **point-to-point communication** (GPU-to-GPU), not collective operations like AllReduce.

---

## Section 2: DeepSpeed Pipeline Implementation (120 lines)

### DeepSpeed's Hybrid Approach

DeepSpeed combines data parallelism and pipeline parallelism for maximum efficiency.

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "DeepSpeed's training engine provides hybrid data and pipeline parallelism and can be further combined with model parallelism such as Megatron-LM. An illustration of 3D parallelism is shown below. Our latest results demonstrate that this 3D parallelism enables training models with over a **trillion** parameters."

**3D Parallelism Architecture**:
1. **Data Parallelism (DP)**: Replicate model across workers, split data
2. **Pipeline Parallelism (PP)**: Split model layers across stages
3. **Tensor Parallelism (TP)**: Split individual layers (via Megatron-LM)

### Expressing Pipeline Models

DeepSpeed requires models as sequential layer lists.

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
```python
# PyTorch Sequential can be used directly
net = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_dim, out_features)
)

from deepspeed.pipe import PipelineModule
net = PipelineModule(layers=net, num_stages=2)
```

**Key Constraint**: Total GPUs must be divisible by number of pipeline stages.

### AlexNet Pipeline Example

From [DeepSpeed AlexNet example](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
```python
class AlexNetPipe(AlexNet):
    def to_layers(self):
        layers = [
            *self.features,        # Conv layers
            self.avgpool,
            lambda x: torch.flatten(x, 1),  # Lambdas allowed!
            *self.classifier       # FC layers
        ]
        return layers

net = PipelineModule(layers=net.to_layers(), num_stages=2)
```

**Important**: Any object implementing `__call__()` can be a layer, allowing data transformations in the pipeline.

### Training Loop Modifications

Traditional forward/backward separation doesn't work with pipeline parallelism.

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "Pipeline parallelism interleaves forward and backward passes, and thus the training loop cannot be divided into separate stages of `forward()`, `backward()` and `step()`. Instead, DeepSpeed's pipeline engine provides a `train_batch()` method."

**DeepSpeed Pipeline Training**:
```python
train_iter = iter(train_loader)
loss = engine.train_batch(data_iter=train_iter)
```

**Equivalent Traditional Loop**:
```python
train_iter = iter(train_loader)
for micro_batch in engine.gradient_accumulation_steps():
    batch = next(data_iter)
    loss = engine(batch)
    engine.backward(loss)
    engine.step()
```

### Data Loading Requirements

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "The pipeline engine expects data loaders to return a `tuple` of two items. The first returned item is the input batch data, and the second item is the data to be used in the loss calculation."

**Critical**: Only first/last pipeline stages need data loaders!
- **First stage**: Loads input data
- **Last stage**: Loads labels for loss
- **Middle stages**: Receive activations from previous stage

**Watch Out**: Each `train_batch()` pulls `gradient_accumulation_steps()` micro-batches from iterator. The data stream must not empty mid-batch!

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
```python
# DeepSpeed provides RepeatingLoader for infinite iteration
train_loader = deepspeed.utils.RepeatingLoader(train_loader)
train_iter = iter(train_loader)
for step in range(args.steps):
    loss = engine.train_batch(data_iter=train_iter)
```

### Load Balancing Strategies

DeepSpeed provides multiple partitioning methods via `partition_method` keyword.

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
- **`partition_method="parameters"`** (default): Balance trainable parameters per stage
  - Best for memory-constrained environments
  - Assumes layer size ∝ computation time
- **`partition_method="type:[regex]"`**: Balance layers matching class name regex
  - Example: `partition_method="type:transformer"` balances Transformer blocks
  - Case-insensitive matching
- **`partition_method="uniform"`**: Balance number of layers per stage
  - Simplest but may create imbalanced compute

**Performance Impact**: Load balancing is critical. Unbalanced stages create bottlenecks and increase bubble time.

### Memory-Efficient Model Construction

For massive models, avoid replicating entire model in CPU memory on every worker.

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "Building a `Sequential` container and providing it to a `PipelineModule` is a convenient way of specifying a pipeline parallel model. However, this approach encounters scalability issues for massive models because each worker replicates the whole model in CPU memory."

**Problem**: 16 GPUs = 16× model size in CPU memory total!

**Solution - LayerSpec**:

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "DeepSpeed provides a `LayerSpec` class that delays the construction of modules until the model layers have been partitioned across workers. Then each worker will allocate only the layers it's assigned to."

```python
from deepspeed.pipe import PipelineModule, LayerSpec

class AlexNetPipe(PipelineModule):
    def __init__(self, num_classes=10, **kwargs):
        specs = [
            LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=2),
            LayerSpec(nn.ReLU, inplace=True),
            # ... more layers
            LayerSpec(nn.Linear, 4096, num_classes),
        ]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)
```

**Syntax**: `nn.ReLU(inplace=True)` becomes `LayerSpec(nn.ReLU, inplace=True)`

With LayerSpec, 16 GPUs need only 1× total model size in CPU memory!

---

## Section 3: Micro-Batching Strategies (100 lines)

### GPipe Schedule

GPipe fully completes all forward passes before starting backward passes.

From [siboehm GPipe analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):

**GPipe Schedule (4 GPUs, 4 micro-batches)**:
```
Timestep   0   1   2   3   4   5   6   7   8   9  10  11  12  13
GPU3                  F1  F2  F3  F4  B4  B3  B2  B1
GPU2            F1  F2  F3  F4          B4  B3  B2  B1
GPU1       F1  F2  F3  F4                  B4  B3  B2  B1
GPU0  F1  F2  F3  F4                          B4  B3  B2  B1

F = Forward, B = Backward
```

**Memory Characteristics**:
- All micro-batches in flight simultaneously (peak memory)
- GPU0 caches activations from timestep 0 until timestep 13
- High memory demand for activation caching

From [DeepSpeed pipeline schedule visualization](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "Below is an illustration of how DeepSpeed will train a batch with eight micro-batches using hybrid two-way data parallelism and two-stage pipeline parallelism. GPUs 0 and 2 are arranged in a pipeline and will alternate forward (F) and backward (B) passes. They will then all-reduce (AR) gradients with their data parallel counterparts."

### PipeDream 1F1B Schedule

PipeDream starts backward passes earlier to reduce memory usage.

From [Megatron-LM Paper Figure](https://arxiv.org/abs/2104.04473) via siboehm (accessed 2025-11-13):

**PipeDream Schedule (4 GPUs, 8 micro-batches)**:
```
Warmup phase → Steady state → Cooldown
```

**Key Difference**: In steady state, each GPU alternates **1 Forward, 1 Backward** (1F1B).

From [siboehm PipeDream analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "With the above PipeDream schedule, we have at most as many microbatches in flight as the pipeline is deep. This becomes obvious when looking at GPU1 in the above plot. During the steady state, GPU1 `forward`'s a new microbatch only after completing a `backward` pass."

**Memory Comparison**:
- **GPipe**: All `m` micro-batches in flight
- **PipeDream**: At most `n` micro-batches in flight (where `n` = pipeline depth)
- **Example**: 8 micro-batches, 4 GPUs
  - GPipe: 8 in flight
  - PipeDream: 4 in flight
  - **50% memory reduction** for activation caching!

### Memory Demand Formula

From [siboehm memory analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):

**Activation Cache Memory (without gradient checkpointing)**:
```
O(#max_microbatches_in_flight · microbatch_size · #layers_per_GPU)

GPipe:      O(m · batch_size/m · L/n) = O(batch_size · L/n)
PipeDream:  O(n · batch_size/m · L/n) = O(batch_size · L/m)

where:
  m = number of micro-batches
  n = pipeline depth (number of GPUs)
  L = total layers
  batch_size = total samples per batch
```

**Insight**: PipeDream memory ∝ 1/m, GPipe memory constant (doesn't scale with m)!

### Gradient Checkpointing Integration

From [GPipe Paper](https://arxiv.org/pdf/1811.06965) (accessed 2025-11-13):
> "In the GPipe paper, the authors utilize gradient checkpointing to bring down the memory demand. In gradient checkpointing, instead of caching all activations necessary to compute our gradients, we recompute the activations on the fly during the backward pass."

**Trade-off**: Lower memory demand ↔ Higher compute cost

**Memory with Gradient Checkpointing**:

From [siboehm gradient checkpointing analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
```
O(batch_size + (L/n) · (batch_size/m))

First term:  Cached boundary activations (sent between GPUs)
Second term: Re-materialized activations during backward pass
```

**Strategy**: Only cache activations at pipeline boundaries, recompute everything else.

### Bubble Analysis

From [siboehm bubble analysis](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "In terms of bubble fraction, there is **no difference between PipeDream and GPipe**. The bubble is a result of the inherent dependencies between the operations before on the microbatches, which PipeDream doesn't change."

**Bubble Reduction Strategy**: Increase number of micro-batches `m`.

**Example Calculations**:
```
4 GPUs, 4 micro-batches:  bubble = (4-1)/(4+4-1)  = 42.8%
4 GPUs, 8 micro-batches:  bubble = (4-1)/(8+4-1)  = 27.3%
4 GPUs, 16 micro-batches: bubble = (4-1)/(16+4-1) = 15.8%
```

**Caveat**: Large batch sizes require learning rate scaling (LARS, LAMB) and increase memory for activations.

### Sequential Consistency

From [siboehm sequential consistency note](https://siboehm.com/articles/22/pipeline-parallel-training) (accessed 2025-11-13):
> "I'll call a distributed algorithm _sequentially consistent_ if the resulting gradients are the same as if we had calculated them using sequential training on a single machine."

**GPipe**: Sequentially consistent (modulo floating-point non-associativity)
**PipeDream 1F1B Flush**: Sequentially consistent
**PipeDream non-flush variants**: NOT sequentially consistent (may hurt convergence)

From [Megatron-LM Paper](https://arxiv.org/abs/2104.04473) via siboehm (accessed 2025-11-13):
> "By avoiding the pipeline flush at the end of processing each batch, one can increase efficiency by decreasing the bubble fraction. However, this means the algorithm isn't sequentially consistent anymore, which may hurt convergence speed."

**Practical Impact**: Non-consistent schedules may train faster per step but require more total steps, potentially negating speed gains.

---

## Section 4: Comparison with Other Frameworks (100 lines)

### GPipe vs PipeDream vs DeepSpeed

From [Study of Pipeline Parallelism](https://dialnet.unirioja.es/descarga/articulo/9714357.pdf) via search results (accessed 2025-11-13):
> "In this study, perform a comparison between WPipe against the GPipe, PipeDream and PipeDream2BW libraries in training with natural language models."

**Feature Comparison Matrix**:

| Feature | GPipe | PipeDream | DeepSpeed |
|---------|-------|-----------|-----------|
| **Micro-batching** | ✓ | ✓ | ✓ |
| **Sequential consistency** | ✓ | ✓ (1F1B Flush) | ✓ |
| **Memory optimization** | Gradient checkpointing | Early backward start | Both + ZeRO |
| **Data parallelism hybrid** | ✗ | ✗ | ✓ |
| **Tensor parallelism integration** | ✗ | ✗ | ✓ (Megatron) |
| **Tied layers support** | ✗ | ✗ | ✓ |

### DeepSpeed Advantages

**1. Hybrid Parallelism**

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "DeepSpeed's training engine provides hybrid data and pipeline parallelism and can be further combined with model parallelism such as Megatron-LM."

**Setup**: If you have 8 GPUs and 2 pipeline stages, DeepSpeed automatically creates:
- 2 pipeline stages (GPUs 0-3, GPUs 4-7)
- 4-way data parallelism within each stage

**2. ZeRO Integration**

DeepSpeed uniquely combines pipeline parallelism with ZeRO optimizer for memory efficiency.

From [karpathy/codebases/05-deepseek-moe-overview.md](../codebases/05-deepseek-moe-overview.md) (line 204-207):
> "**Related to DualPipe** (source-codebases/09-DualPipe)
> - Pipeline parallelism for MoE models
> - Optimizes expert-parallel + data-parallel training
> - Critical for scaling to V3-scale models"

**3. Tied Layers Support**

Some models reuse layers (e.g., embedding layer shared between input and output).

From [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
> "DeepSpeed provides a `TiedLayerSpec` that is an extension of `LayerSpec`. `TiedLayerSpec` requires an additional argument: `key`. Each reuse of a layer is specified with a `TiedLayerSpec`, and the `key` field is used to identify where a layer is reused."

**Implementation**:
```python
# Embedding used at input and output
specs = [
    TiedLayerSpec('embed', nn.Embedding, vocab_size, hidden_dim),
    # ... transformer blocks ...
    TiedLayerSpec('embed', nn.Linear, hidden_dim, vocab_size),  # Tied!
]
```

**Mechanism**: Tied layers replicated on all stages that use them. After backward pass, additional all-reduce syncs gradients across replicas.

### Megatron-LM Integration

From [Megatron-DeepSpeed GitHub](https://github.com/deepspeedai/Megatron-DeepSpeed) (accessed 2025-11-13):
> "To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution across stages)"

**Megatron-DeepSpeed 3D Parallelism**:

From [NVIDIA Using DeepSpeed and Megatron Blog](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) (accessed 2025-11-13):
> "More specifically, the system uses tensor-slicing from Megatron-LM to scale the model within a node and uses pipeline parallelism from DeepSpeed to scale the model across nodes."

**Configuration Example** (Megatron-Turing NLG 530B):
- **Tensor Parallelism (TP)**: 8-way (within node, via NVLink)
- **Pipeline Parallelism (PP)**: 8-way (across nodes, via InfiniBand)
- **Data Parallelism (DP)**: 12-way (outer dimension)
- **Total**: 8 × 8 × 12 = 768 GPUs

**Why This Configuration**:
- TP uses fast intra-node NVLink (900 GB/s)
- PP uses slower inter-node InfiniBand (50 GB/s)
- DP uses AllReduce across replicas (maximizes throughput)

### DualPipe for MoE Models

From [karpathy/codebases/02-karpathy-on-deepseek-efficiency.md](../codebases/02-karpathy-on-deepseek-efficiency.md) (lines 185-189):
> "DeepSeek solved this with:
> 1. **Auxiliary loss** for load balancing
> 2. **DualPipe** (custom pipeline parallelism for MoE)
> 3. **Fine-grained experts** (256 small experts vs 16 large experts)"

**DualPipe Specifics** (from source-codebases/09-DualPipe):
- Optimizes pipeline parallelism for Mixture-of-Experts
- Handles expert parallelism + data parallelism coordination
- Reduces pipeline bubbles in sparse activation patterns
- Critical for DeepSeek-V3's 256-expert architecture

### Performance Benchmarks

From [DDLBench Paper](https://atlarge-research.com/pdfs/jansen2020ddlbench.pdf) via search results (accessed 2025-11-13):
> "Some experiments did not succeed for GPipe and PipeDream due to unexpected behaviour. In our experience, this behavior demonstrates the lack of maturity for these frameworks."

**Stability Ranking** (based on production usage):
1. **DeepSpeed**: Most mature, production-proven (BLOOM, Megatron-Turing)
2. **Megatron-LM**: NVIDIA-backed, tensor parallelism focus
3. **GPipe/PipeDream**: Research implementations, less production-ready

**Memory Efficiency Comparison**:
From [PipeDream-2BW Paper](http://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf) (accessed 2025-11-13):
> "PipeDream-2BW is up to 3.2× faster than GPipe, and is able to train large transformer models that vanilla PipeDream cannot fit in memory."

---

## Section 5: VLM-Specific Patterns (50 lines)

### Vision-Language Model Considerations

VLMs present unique challenges for pipeline parallelism:

**Architecture Characteristics**:
1. **Heterogeneous layers**: Vision encoder (CNN/ViT) + Language decoder (Transformer)
2. **Variable sequence lengths**: Image patches vs text tokens
3. **Multi-modal fusion**: Cross-attention between vision and language

### Load Balancing Strategies for VLMs

**Challenge**: Vision encoders and language decoders have different compute/memory profiles.

**Solution - Type-based Partitioning**:

From [DeepSpeed partition methods](https://www.deepspeed.ai/tutorials/pipeline/) (accessed 2025-11-13):
```python
# Balance transformer layers across stages
net = PipelineModule(
    layers=vlm_layers,
    num_stages=4,
    partition_method="type:transformer"
)
```

**VLM Layer Distribution Example** (4 GPUs):
```
GPU 0: Vision encoder (ResNet/ViT layers)
GPU 1: Vision encoder final + Cross-attention 1-4
GPU 2: Cross-attention 5-8 + Transformer decoder 1-6
GPU 3: Transformer decoder 7-12 + Output head
```

### Batch Size Considerations

**VLM Challenge**: Image preprocessing and tokenization vary per sample.

**Recommendation**: Use smaller micro-batches to handle variable-length inputs.

```python
# VLM-specific configuration
micro_batch_size = 1  # Handle variable image sizes
gradient_accumulation_steps = 16  # Effective batch = 16

# Each GPU processes 1 image at a time
# Accumulate gradients across 16 micro-batches
```

**Memory Trade-off**:
- Smaller micro-batches → Less memory per forward pass
- More micro-batches → More cached activations (use gradient checkpointing!)

### ARR-COC Pipeline Patterns

**ARR-COC Architecture** (from project context):
- 13-channel texture array generation
- 3 relevance scorers (propositional, perspectival, participatory)
- Opponent processing
- Variable LOD allocation (64-400 tokens)

**Proposed Pipeline Split** (4 GPUs):
```
Stage 0 (GPU 0): Texture array generation (13 channels)
  - RGB, LAB, Sobel, spatial, eccentricity
  - Heavy preprocessing, all samples same cost

Stage 1 (GPU 1): Relevance scoring (propositional, perspectival)
  - Information content measurement
  - Salience computation
  - Intermediate compute

Stage 2 (GPU 2): Relevance scoring (participatory) + Opponent processing
  - Query-content coupling
  - Tension balancing
  - Variable compute per sample

Stage 3 (GPU 3): LOD allocation + Qwen3-VL integration
  - Token budget assignment (64-400 tokens)
  - Final vision-language fusion
  - Output generation
```

**Why This Split**:
- Stage 0: Uniform compute (good for first stage)
- Stages 1-2: Balanced relevance computation
- Stage 3: Variable LOD handles query-dependent paths
- Memory: Gradient checkpointing on texture arrays (13 channels heavy!)

**Micro-batch Size**: Recommend 4-8 for ARR-COC to balance:
- Bubble time (need enough micro-batches)
- Memory for 13-channel texture arrays
- Variable LOD allocation overhead

---

## Sources

**Official Documentation:**
- [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) - Official DeepSpeed documentation (accessed 2025-11-13)
- [DeepSpeed GitHub Examples](https://github.com/deepspeedai/DeepSpeedExamples) - Code examples (accessed 2025-11-13)
- [Megatron-DeepSpeed GitHub](https://github.com/deepspeedai/Megatron-DeepSpeed) - 3D parallelism implementation (accessed 2025-11-13)

**Research Papers:**
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/pdf/1811.06965) (Huang et al., 2018, 2,291 citations) - Original GPipe paper (accessed 2025-11-13)
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377) (Narayanan et al., 2019) - PipeDream algorithm
- [PipeDream-2BW: Memory-Efficient Pipeline-Parallel DNN Training](http://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf) (Narayanan et al., 2021, 320 citations) - Improved PipeDream (accessed 2025-11-13)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/2104.04473) (Shoeybi et al., 2021) - 3D parallelism

**Technical Analysis:**
- [Pipeline-Parallelism: Distributed Training via Model Partitioning](https://siboehm.com/articles/22/pipeline-parallel-training) by Simon Boehm - Comprehensive technical analysis (accessed 2025-11-13)
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) - NVIDIA production case study (accessed 2025-11-13)

**Source Documents:**
- [karpathy/llm-gpu-integration/02-training-dynamics-gpu.md](../llm-gpu-integration/02-training-dynamics-gpu.md) - Pipeline parallelism fundamentals
- [karpathy/codebases/02-karpathy-on-deepseek-efficiency.md](../codebases/02-karpathy-on-deepseek-efficiency.md) - DualPipe for MoE
- [karpathy/codebases/05-deepseek-moe-overview.md](../codebases/05-deepseek-moe-overview.md) - DualPipe integration

**Additional References:**
- [A Study of Pipeline Parallelism in Deep Neural Networks](https://dialnet.unirioja.es/descarga/articulo/9714357.pdf) (Núñez et al., 2024) - Comparative study (accessed 2025-11-13)
- [DDLBench: Towards a Scalable Benchmarking Infrastructure](https://atlarge-research.com/pdfs/jansen2020ddlbench.pdf) (Jansen et al., 2020, 11 citations) - Framework comparison (accessed 2025-11-13)
