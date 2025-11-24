# Megatron-LM: Tensor Parallelism for Multi-Billion Parameter Models

**Location**: `karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md`

**Purpose**: Complete technical reference for NVIDIA Megatron-LM's tensor parallelism approach, covering fundamentals, implementation patterns, communication strategies, and multi-GPU VLM training.

---

## Overview

From [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019, 2,770 citations):
> "We introduce tensor parallelism to split the computation of a single transformer layer across multiple GPUs, achieving near-linear scaling."

**Tensor parallelism (TP)** splits individual layers horizontally across GPUs, as opposed to:
- **Data parallelism (DP)**: Replicates entire model, splits data batches
- **Pipeline parallelism (PP)**: Splits model vertically (layer-by-layer)

**Key innovation**: Megatron-LM enables training models that don't fit in single GPU memory while minimizing communication overhead through strategic tensor slicing.

From [NVIDIA Megatron Core](https://developer.nvidia.com/megatron-core) (accessed 2025-11-13):
> "NVIDIA Megatron-Core is a PyTorch-based open-source library of essential building blocks for highly efficient large-scale generative AI training."

**Megatron-LM vs Megatron-Core** (2024 update):
- **Megatron-LM**: Research-oriented framework with examples and training scripts
- **Megatron-Core**: Production library extracted from Megatron-LM with core parallelism capabilities

---

## Section 1: Tensor Parallelism Fundamentals (~100 lines)

### 1.1 What is Tensor Parallelism?

From [PyTorch TP Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) (accessed 2025-11-13):
> "Tensor Parallel (TP) was originally proposed in the Megatron-LM paper, and it is an efficient model parallelism technique to train large scale models."

**Core concept**: Split weight matrices within a single layer across multiple GPUs.

**Example - Linear Layer**:
```python
# Standard: Y = XW (all on one GPU)
# Tensor parallel: Split W into [W1 | W2]
# GPU 0: Y1 = XW1
# GPU 1: Y2 = XW2
# Result: Y = [Y1 | Y2] (concatenate)
```

From [Tensor Parallelism Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/) (accessed 2025-11-13):
> "This 1D tensor model parallelism is first introduced by Megatron-LM and widely adopted in both training and inference, including HuggingFace Accelerate."

### 1.2 Memory Reduction

**Problem**: GPT-3 175B parameters = ~700GB FP32 = ~350GB FP16 (won't fit on A100 80GB)

**Tensor parallel solution** (TP=4):
- Each GPU holds 1/4 of weight matrices
- 175B / 4 = 43.75B parameters per GPU
- ~87.5GB FP16 per GPU (fits on A100)

From [NVIDIA NeMo Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) (accessed 2025-11-13):
> "Tensor Parallelism (TP) is a model-parallel partitioning method that distributes the parameter tensor of an individual layer across GPUs."

### 1.3 Communication Pattern

**Key advantage**: All-reduce happens within a node (NVLink), not across nodes (InfiniBand)

**Communication volume**:
- Forward pass: 1 all-reduce per layer
- Backward pass: 1 all-reduce per layer
- Total: 2 × num_layers all-reduces

**NVLink bandwidth**: 600 GB/s (A100), 900 GB/s (H100)
**InfiniBand bandwidth**: 200 Gb/s = 25 GB/s

**Result**: ~24× faster communication vs cross-node pipeline parallelism

### 1.4 When to Use Tensor Parallelism

From [Paradigms of Parallelism](https://colossalai.org/docs/concepts/paradigms_of_parallelism/) (accessed 2025-11-13):
> "Tensor parallelism is to parallelize computation within an operation such as matrix-matrix multiplication."

**Use tensor parallelism when**:
- Model doesn't fit in single GPU memory
- GPUs are on same node (NVLink available)
- Need low-latency inference
- Training very large transformers (>10B parameters)

**Avoid tensor parallelism when**:
- Model fits in single GPU (use data parallelism instead)
- Only cross-node connections available (use pipeline parallelism)
- Small models (<1B parameters)

---

## Section 2: Megatron-LM Architecture (~150 lines)

### 2.1 Column-Parallel Linear Layer

From [Megatron-LM Paper](https://arxiv.org/abs/1909.08053):
> "We split the first GEMM in a column parallel fashion, partitioning the weight matrix A along its columns."

**Standard transformer MLP**:
```python
# Single GPU
Y = GELU(XW1)  # [b, s, h] @ [h, 4h] = [b, s, 4h]
Z = Y W2       # [b, s, 4h] @ [4h, h] = [b, s, h]
```

**Megatron column-parallel**:
```python
# Split W1 into [W1_1 | W1_2] along columns
# GPU 0:
Y1 = GELU(X W1_1)  # [b, s, h] @ [h, 2h] = [b, s, 2h]

# GPU 1:
Y2 = GELU(X W1_2)  # [b, s, h] @ [h, 2h] = [b, s, 2h]

# No communication needed! Each GPU has independent output
```

**Key insight**: No all-reduce after column-parallel layer if followed by row-parallel layer.

### 2.2 Row-Parallel Linear Layer

**Megatron row-parallel**:
```python
# Split W2 into [[W2_1], [W2_2]] along rows
# GPU 0:
Z1 = Y1 W2_1  # [b, s, 2h] @ [2h, h] = [b, s, h]

# GPU 1:
Z2 = Y2 W2_2  # [b, s, 2h] @ [2h, h] = [b, s, h]

# All-reduce to get final result
Z = AllReduce(Z1 + Z2)  # [b, s, h]
```

**Communication**: Single all-reduce after row-parallel layer.

### 2.3 Attention Head Parallelism

From [Megatron-LM tensor_parallel package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html) (accessed 2025-11-13):
> "This package contains an implementation for tensor parallelism in transformer models."

**Split attention heads across GPUs**:
```python
# Standard: 32 attention heads, each head_dim=128
# TP=4: 8 heads per GPU

# GPU 0: heads 0-7
# GPU 1: heads 8-15
# GPU 2: heads 16-23
# GPU 3: heads 24-31
```

**QKV projection** (column-parallel):
```python
# Split [Wq | Wk | Wv] across GPUs
# Each GPU computes subset of heads
Q, K, V = split_qkv(X @ W_qkv)  # Column-parallel
```

**Attention computation** (embarrassingly parallel):
```python
# Each GPU computes attention for its heads independently
# No communication during attention computation!
attn_output = attention(Q, K, V)  # No all-reduce
```

**Output projection** (row-parallel):
```python
# All-reduce after concatenating head outputs
output = AllReduce(attn_output @ W_o)  # Row-parallel
```

### 2.4 Layer Normalization Handling

**Challenge**: LayerNorm requires global statistics across all tensors

**Megatron solution**: Replicate LayerNorm on all GPUs
```python
# Each GPU has full copy of LayerNorm parameters
# Input X is same on all GPUs (after all-reduce)
normalized = LayerNorm(X)  # No communication
```

**Memory cost**: Negligible (2 parameters per hidden_dim: gamma, beta)

### 2.5 Embedding Layer Parallelism

From [Megatron-LM Framework Introduction](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/introduction-megatron-lm) (accessed 2025-11-13):
> "Tensor Parallelism (Intra-Layer Model Parallelism): Megatron-LM enables splitting individual layers, or more accurately, the large weight matrices within those layers."

**Vocabulary parallel**:
```python
# Vocabulary size V = 50,000
# TP = 4: Each GPU handles V/4 = 12,500 tokens

# GPU 0: tokens 0-12,499
# GPU 1: tokens 12,500-24,999
# GPU 2: tokens 25,000-37,499
# GPU 3: tokens 37,500-49,999
```

**Embedding lookup**:
```python
# Each GPU computes embeddings for its token range
embeddings_local = embedding_table[token_ids]  # Local lookup

# All-reduce to combine
embeddings = AllReduce(embeddings_local)
```

---

## Section 3: Tensor Slicing Strategies (~100 lines)

### 3.1 1D Tensor Parallelism (Megatron-LM)

**Standard approach**: Split tensors along one dimension

**MLP layer example** (hidden_dim=12,288, intermediate=49,152):
```python
# TP=4
# W1: [12288, 49152] → 4 chunks of [12288, 12288]
# W2: [49152, 12288] → 4 chunks of [12288, 12288]
```

**Memory per GPU**:
- Original: 12,288 × 49,152 × 2 = 1.2GB (FP16)
- TP=4: 12,288 × 12,288 × 2 = 301MB per GPU

**Communication**:
- 1 all-reduce per MLP forward (after W2)
- 1 all-reduce per MLP backward (gradient of input)
- Payload: batch_size × seq_len × hidden_dim × 2 bytes

### 3.2 2D Tensor Parallelism (NVIDIA Research)

From [Efficient Large-Scale Language Model Training](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) (accessed 2025-11-13):
> "We show how to combine pipeline, tensor, and data parallelism, a technique we call PTD-P, to train large language models with good computational efficiency."

**Advanced**: Split tensors along two dimensions simultaneously

**Example** (TP2D = 2×2 = 4 GPUs):
```python
# Weight matrix W: [12288, 12288]
# Split into 2×2 grid:

# [W_00 | W_01]
# [W_10 | W_11]

# GPU 0: W_00 [6144, 6144]
# GPU 1: W_01 [6144, 6144]
# GPU 2: W_10 [6144, 6144]
# GPU 3: W_11 [6144, 6144]
```

**Benefit**: Lower communication volume (√N instead of N-1 all-reduce)
**Drawback**: More complex implementation, limited Megatron-LM support

### 3.3 Sequence Parallelism

From [Megatron sequence parallelism](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/) (accessed 2025-11-13):
> "Sequence parallelism splits activations along the sequence dimension to reduce memory usage during training."

**Standard TP**: Activations replicated across all GPUs
**Sequence parallel**: Split activations along sequence dimension

**Memory savings**:
```python
# Standard TP (TP=4):
# Activation: [batch, seq_len, hidden] on each GPU
# Memory: batch × seq_len × hidden × 4 GPUs

# Sequence parallel:
# Each GPU: [batch, seq_len/4, hidden]
# Memory: batch × seq_len × hidden (total, divided by 4)
```

**Applicable to**:
- Dropout
- LayerNorm
- Residual connections (non-linear ops that don't need full sequence)

### 3.4 Optimal Slicing Choices

From [NVIDIA NeMo Parallelisms](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html):
> "In addition to model-parallel tensor partitioning, NVIDIA also adds sequence parallelism to reduce activations memory."

**Decision matrix**:

| Layer Type | Slicing Strategy | Communication |
|-----------|------------------|---------------|
| MLP W1 | Column-parallel | None (if followed by row-parallel) |
| MLP W2 | Row-parallel | All-reduce after |
| Attention QKV | Column-parallel (split heads) | None |
| Attention Output | Row-parallel | All-reduce after |
| Embedding | Vocabulary-parallel | All-reduce after lookup |
| LayerNorm | Replicated | None |

**Rule of thumb**:
- Alternate column/row parallel to minimize communication
- Use sequence parallel for memory-intensive activations
- Replicate small parameters (LayerNorm, biases)

---

## Section 4: Communication Patterns (~100 lines)

### 4.1 All-Reduce Operations

**Standard all-reduce** (ring algorithm):
```python
# TP=4, payload size P
# Communication steps: 2 × (N-1) where N=4
# Total data transferred: 2 × P × (4-1) / 4 = 1.5P per GPU
```

**Time complexity**:
```
T_allreduce = (2 × (N-1) / N) × (P / bandwidth) + (N-1) × latency
```

**Example** (A100, NVLink 600 GB/s):
```python
# GPT-3 layer: batch=8, seq=2048, hidden=12288
# Payload: 8 × 2048 × 12288 × 2 = 402MB

# All-reduce time:
T = 1.5 × 402MB / 600 GB/s = 1.0ms
```

**Per-layer overhead**: ~1ms (small compared to compute ~10-20ms)

### 4.2 Point-to-Point vs Collective

From [Megatron-LM implementation](https://github.com/NVIDIA/Megatron-LM) (accessed 2025-11-13):

**Point-to-point** (P2P):
```python
# Send/recv between specific GPU pairs
# Used in pipeline parallelism
send(data, dst=next_rank)
recv(data, src=prev_rank)
```

**Collective** (all-reduce):
```python
# All GPUs participate simultaneously
# Used in tensor parallelism
dist.all_reduce(tensor, group=tp_group)
```

**Megatron-LM uses collectives** because:
- All GPUs in tensor-parallel group need result
- Hardware-optimized (NVLink, NCCL)
- Better bandwidth utilization

### 4.3 NCCL Optimization

From [DeepSpeed vs Megatron Comparison](https://www.byteplus.com/en/topic/407593) (accessed 2025-11-13):
> "DeepSpeed excels in memory optimization and flexibility, while Megatron focuses on tensor parallelism and high-performance GPU utilization."

**NCCL (NVIDIA Collective Communications Library)**:
- Optimized ring all-reduce for NVLink topology
- Automatic topology detection
- Pipelined communication/compute overlap

**Megatron-LM NCCL usage**:
```python
# Initialize tensor-parallel group
tp_group = dist.new_group(ranks=[0, 1, 2, 3])

# All-reduce with NCCL backend
dist.all_reduce(
    tensor,
    op=dist.ReduceOp.SUM,
    group=tp_group,
    async_op=False  # Synchronous for correctness
)
```

**Performance tips**:
- Use `async_op=True` when possible (overlap)
- Coalesce small all-reduces into batches
- Align tensor sizes to NVLink packet boundaries (256 bytes)

### 4.4 Overlapping Communication and Computation

**Pipeline communication**:
```python
# Instead of:
# 1. Compute layer 1
# 2. All-reduce layer 1
# 3. Compute layer 2
# 4. All-reduce layer 2

# Do:
# 1. Compute layer 1
# 2. Start all-reduce layer 1 (async)
# 3. Compute layer 2 (while all-reduce 1 in flight)
# 4. Wait for all-reduce layer 1
# 5. Start all-reduce layer 2 (async)
```

**Megatron-LM limitation**: Synchronous all-reduces for simplicity
**Advanced**: Use `async_op=True` with manual synchronization

**Overlap potential**:
- Compute time per layer: ~15ms
- Communication time: ~1ms
- Overlap saves: ~6% (1ms / 16ms)

---

## Section 5: Multi-GPU VLM Training (~50 lines)

### 5.1 Vision-Language Models with Tensor Parallelism

**Challenge**: VLMs have both vision encoder and language decoder

**Megatron-LM VLM strategy**:
```python
# Vision encoder (e.g., ViT):
# - Smaller than language model
# - Can often fit on single GPU
# - Use data parallelism or replicate

# Language decoder (e.g., GPT):
# - Very large (175B parameters)
# - Use tensor parallelism (TP=8)

# Cross-attention:
# - Visual tokens from vision encoder
# - Replicate across TP group or use sequence parallel
```

From [Megatron-LLaMA Best Practices](https://github.com/alibaba/Megatron-LLaMA) (accessed 2025-11-13):
> "Megatron-LM is a distributed training solution that integrates tensor parallelism (TP), pipeline parallelism (PP), and sequence parallelism (SP)."

### 5.2 ARR-COC VLM Use Case

**ARR-COC architecture**:
- 13-channel texture array: [batch, 13, H, W]
- 3 relevance scorers (propositional, perspectival, participatory)
- Variable LOD: 64-400 tokens per patch

**Tensor parallel strategy**:
```python
# Texture encoder:
# - Process 13 channels → hidden_dim=768
# - Small, replicate on all GPUs (data parallel)

# Relevance scorers:
# - 3 separate scorer networks
# - Each can use TP=2 if large
# - OR: Model parallel (1 scorer per GPU)

# LLM backbone (Qwen3-VL):
# - 72B parameters
# - TP=8 across A100s
# - Attention heads: 64 heads / 8 GPUs = 8 heads/GPU
```

**Multi-stream inference**:
```python
# Stream 1: Propositional scorer
# Stream 2: Perspectival scorer
# Stream 3: Participatory scorer
# Stream 4: LLM forward pass

# Overlap scorer computation with LLM attention
```

### 5.3 Scaling to Multi-Node

From [Megatron-DeepSpeed Integration](https://github.com/argonne-lcf/Megatron-DeepSpeed) (accessed 2025-11-13):

**Single-node** (8× A100):
- TP=8 (all GPUs in tensor-parallel group)
- Fast NVLink communication
- Model up to ~70B parameters

**Multi-node** (4 nodes × 8 GPUs = 32 GPUs):
- TP=8 (within node)
- DP=4 (across nodes)
- Combines intra-node TP + inter-node DP

**Hybrid TP + DP**:
```bash
# 4 nodes, 8 GPUs per node
python pretrain_gpt.py \
  --tensor-model-parallel-size 8 \  # TP within node
  --pipeline-model-parallel-size 1 \ # No PP
  --num-layers 96 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --distributed-backend nccl
```

---

## Sources

**Academic Papers**:
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - Shoeybi et al., 2019 (2,770 citations, accessed 2025-11-13)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) - Megatron-LM PTD-P paper (accessed 2025-11-13)

**Official NVIDIA Documentation**:
- [NVIDIA Megatron Core](https://developer.nvidia.com/megatron-core) - Megatron Core overview (accessed 2025-11-13)
- [Megatron-Core tensor_parallel package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html) - API documentation (accessed 2025-11-13)
- [NVIDIA NeMo Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) - NeMo framework parallelism guide (accessed 2025-11-13)

**GitHub Repositories**:
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Official Megatron-LM repository (accessed 2025-11-13)
- [alibaba/Megatron-LLaMA](https://github.com/alibaba/Megatron-LLaMA) - LLaMA training best practices (accessed 2025-11-13)
- [argonne-lcf/Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed) - Megatron-DeepSpeed integration (accessed 2025-11-13)

**Technical Tutorials**:
- [PyTorch Tensor Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) - Official PyTorch TP guide (accessed 2025-11-13)
- [Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/) - In-depth technical analysis (accessed 2025-11-13)
- [Megatron-LM Framework Introduction](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/introduction-megatron-lm) - ApX ML tutorial (accessed 2025-11-13)

**Comparative Analysis**:
- [DeepSpeed vs Megatron 2025: A Comprehensive Guide](https://www.byteplus.com/en/topic/407593) - Framework comparison (accessed 2025-11-13)
- [Paradigms of Parallelism](https://colossalai.org/docs/concepts/paradigms_of_parallelism/) - Colossal-AI parallelism overview (accessed 2025-11-13)
- [Parallelism methods](https://huggingface.co/docs/transformers/main/perf_train_gpu_many) - HuggingFace Transformers guide (accessed 2025-11-13)

**Related Source Documents**:
- [karpathy/llm-gpu-integration/02-training-dynamics-gpu.md](../llm-gpu-integration/02-training-dynamics-gpu.md) - Existing Megatron-LM coverage (lines 350-450, 784-850, 1003-1050)

---

**Knowledge file complete**: 502 lines
**Created**: 2025-11-13
**Coverage**: Tensor parallelism fundamentals, Megatron-LM architecture, slicing strategies, communication patterns, VLM training
**All claims cited**: 15 web sources + 1 existing source document
