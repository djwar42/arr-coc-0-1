# VLM Memory Footprint Analysis

## Overview

Understanding GPU memory consumption is critical for deploying vision-language models (VLMs) efficiently. Memory requirements scale dramatically with model size, sequence length, and concurrent requests, making careful analysis essential for production deployment. This document provides a comprehensive breakdown of memory components, optimization techniques, and practical guidance for estimating VRAM requirements.

### Why Memory Matters for VLMs

GPU VRAM (Video RAM) is the primary bottleneck for VLM serving due to:
- **Higher bandwidth**: Enables faster data transfer between memory and processing units
- **Lower latency**: Reduces waiting time for memory operations
- **Optimized architecture**: Specifically designed for parallel computing tasks

Unlike system RAM, GPU memory is limited and expensive, making efficient utilization crucial for maximizing throughput and enabling larger batch sizes.

## Memory Components Breakdown

### 1. Model Parameters (Weights)

Model weights represent the learned parameters that define how the VLM processes input data. Memory requirements scale linearly with model size.

**Calculation Formula:**
```
Memory (GB) = Number of Parameters × Bytes per Parameter
```

**Common Precisions:**
- **FP32 (32-bit)**: 4 bytes per parameter
- **FP16/BF16 (16-bit)**: 2 bytes per parameter (standard)
- **INT8 (8-bit)**: 1 byte per parameter
- **INT4 (4-bit)**: 0.5 bytes per parameter

**Examples:**

From [GitHub LLaVA Issue #191](https://github.com/haotian-liu/LLaVA/issues/191) (accessed 2025-01-31):
- **LLaVA-7B** (FP16): 7B × 2 bytes = 14 GB
- **LLaVA-13B** (FP16): 13B × 2 bytes = 26 GB
- **LLaVA-34B** (FP16): 34B × 2 bytes = 68 GB

### 2. Key-Value (KV) Cache Memory

The KV cache stores intermediate attention representations for efficient token generation. This is often the largest memory consumer during inference, especially for long sequences and multiple concurrent requests.

**Per-Token KV Cache Size:**

From [NVIDIA Developer Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-01-31):

```
KV Cache per Token = 2 × num_layers × hidden_size × precision_bytes
```

The factor of 2 accounts for both Key and Value matrices.

**Example: LLaMA-2 13B**
- Layers: 40
- Hidden size: 5120
- Precision: FP16 (2 bytes)
- Key vectors: 40 × 5120 × 2 = 400 KB
- Value vectors: 40 × 5120 × 2 = 400 KB
- **Total per token**: 800 KB

**Scaling with Sequence Length and Batch Size:**

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

```
Total KV Cache = KV_per_token × sequence_length × batch_size
```

For 10 concurrent requests with 8192 tokens:
- 800 KB/token × 8192 tokens × 10 requests = **~65.5 GB**

This demonstrates why KV cache dominates memory consumption for high-throughput serving scenarios.

### 3. Activations and Temporary Buffers

Activations are intermediate outputs from neural network layers during inference, while temporary buffers handle matrix multiplication and other operations.

**Memory Overhead:**

From [UnfoldAI Analysis](https://unfoldai.com/gpu-memory-requirements-for-llms/) (accessed 2025-01-31):

```
Activations + Overhead ≈ 5-10% of (Model Weights + KV Cache)
```

For LLaMA-2 13B example:
- Weights: 26 GB
- KV Cache: 66 GB
- Activations: 0.1 × (26 + 66) = **9.2 GB**

### 4. Memory Fragmentation and Inefficiencies

**Internal Fragmentation**: Memory blocks allocated but not fully utilized, especially when reserving space for maximum sequence length.

**External Fragmentation**: Free memory split into non-contiguous blocks, reducing effective capacity.

From [vLLM PagedAttention Paper](https://arxiv.org/pdf/2309.06180.pdf) (accessed 2025-01-31):
- Static over-provisioning wastes **40-60%** of allocated memory
- Unpredictable sequence lengths lead to inefficient space utilization
- Memory tied up for entire request duration regardless of actual usage

## Complete Memory Calculation

From [UnfoldAI](https://unfoldai.com/gpu-memory-requirements-for-llms/) (accessed 2025-01-31):

```
Total Memory = Model Weights + KV Cache + Activations + Overhead
```

**LLaMA-2 13B Example (8192 tokens, 10 concurrent requests):**
1. **Weights**: 26 GB
2. **KV Cache**: 66 GB
3. **Activations**: 9.2 GB
4. **Total**: **101.2 GB** (requires 3× A100 40GB GPUs)

## Memory Requirements by Model Size

### Single Concurrent Request (4k-128k tokens)

From [UnfoldAI Tables](https://unfoldai.com/gpu-memory-requirements-for-llms/):

| Model | 4k Tokens | 8k Tokens | 32k Tokens | 128k Tokens |
|-------|-----------|-----------|------------|-------------|
| 7B    | 17.6 GB   | 19.8 GB   | 33.0 GB    | 85.8 GB     |
| 13B   | 32.1 GB   | 35.6 GB   | 56.8 GB    | 141.2 GB    |
| 30B   | 72.1 GB   | 78.1 GB   | 114.5 GB   | 259.7 GB    |
| 70B   | 165.6 GB  | 177.1 GB  | 244.1 GB   | 523.3 GB    |
| 175B  | 405.8 GB  | 426.5 GB  | 551.0 GB   | 1,049.6 GB  |

### 10 Concurrent Requests

| Model | 4k Tokens | 8k Tokens | 32k Tokens | 128k Tokens |
|-------|-----------|-----------|------------|-------------|
| 7B    | 37.4 GB   | 59.4 GB   | 191.4 GB   | 719.4 GB    |
| 13B   | 63.8 GB   | 99.0 GB   | 303.6 GB   | 1,128.6 GB  |
| 30B   | 126.5 GB  | 181.5 GB  | 528.0 GB   | 1,914.0 GB  |
| 70B   | 264.0 GB  | 374.0 GB  | 1,034.0 GB | 3,674.0 GB  |

**Key Observations:**
- Memory scales **linearly** with concurrent requests
- KV cache growth dominates for longer sequences
- Multi-GPU deployment essential for larger models

## Batch Size Impact on Memory

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) and [UnfoldAI](https://unfoldai.com/gpu-memory-requirements-for-llms/):

### Batch Size Scaling Pattern

**Memory Components Scaling:**
- **Weights**: Fixed (independent of batch size)
- **KV Cache**: Linear scaling with batch size
- **Activations**: Linear scaling with batch size

**Example: LLaVA-13B at different batch sizes**

| Batch Size | Weights | KV Cache (8k seq) | Activations | Total Memory |
|------------|---------|-------------------|-------------|--------------|
| 1          | 26 GB   | 6.6 GB            | 3.3 GB      | 35.9 GB      |
| 4          | 26 GB   | 26.4 GB           | 5.2 GB      | 57.6 GB      |
| 8          | 26 GB   | 52.8 GB           | 7.9 GB      | 86.7 GB      |
| 16         | 26 GB   | 105.6 GB          | 13.2 GB     | 144.8 GB     |

**Practical Implications:**
- Batch size 1-4: Fits on single A100 80GB
- Batch size 8: Requires multi-GPU or optimization
- Batch size 16+: Mandatory tensor parallelism or quantization

## Optimization Techniques

### 1. Quantization: Reducing Precision

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-01-31):

**Quantization Formats:**

| Format | Bytes/Param | Memory Reduction | Typical Accuracy Impact |
|--------|-------------|------------------|-------------------------|
| FP32   | 4           | Baseline         | Best (reference)        |
| FP16   | 2           | 50%              | Minimal (~0.1%)         |
| INT8   | 1           | 75%              | Low (~0.5-1%)           |
| FP8    | 1           | 75%              | Very Low (~0.2%)        |
| INT4   | 0.5         | 87.5%            | Moderate (~2-3%)        |

**Example: 70B Model Quantization Impact**

| Format | Model Memory | Reduction vs FP32 |
|--------|--------------|-------------------|
| FP32   | 280 GB       | -                 |
| FP16   | 140 GB       | 50%               |
| INT8   | 70 GB        | 75%               |
| INT4   | 35 GB        | 87.5%             |

From [UnfoldAI](https://unfoldai.com/gpu-memory-requirements-for-llms/):

**Quick Estimation Formula:**
```
Memory (GB) = (Parameters × 4 bytes) / (32 / Quantization_Bits) × 1.2
```

The 1.2 factor accounts for 20% overhead.

### 2. FlashAttention: Memory-Efficient Attention

From [FlashAttention Paper](https://arxiv.org/abs/2205.14135) and [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-01-31):

**Key Benefits:**
- **Memory reduction**: 10-20× less memory for attention computation
- **Speed improvement**: 2-4× faster attention on long sequences
- **Exact attention**: Mathematically identical to standard attention
- **IO-aware**: Optimized for GPU memory hierarchy

**Memory Savings Mechanism:**
- **Tiling**: Computes attention in blocks, keeping intermediates in fast SRAM
- **Recomputation**: Avoids storing full attention matrices
- **Fused operations**: Combines multiple attention steps into single kernel

From [INT-FlashAttention Paper](https://arxiv.org/abs/2409.16997) (accessed 2025-01-31):
- **INT8 FlashAttention**: 72% faster inference on Ampere GPUs
- **82% smaller quantization error** compared to FP16
- Compatible with forward workflow, no accuracy degradation

### 3. PagedAttention: Efficient KV Cache Management

From [vLLM PagedAttention](https://vllm.ai/) (accessed 2025-01-31):

**Problem with Static Allocation:**
- Over-provision for max sequence length
- 40-60% memory waste from fragmentation
- Contiguous allocation requirements

**PagedAttention Solution:**
- **Non-contiguous storage**: KV cache in fixed-size blocks
- **Dynamic allocation**: Blocks allocated as tokens generate
- **Memory sharing**: Within and across requests
- **Near-zero waste**: Eliminates internal fragmentation

**Performance Impact:**
- **2-3× larger batch sizes** possible
- **Throughput improvement**: 50-80% for long sequences
- **Memory utilization**: >90% (vs 40-60% traditional)

### 4. Gradient Checkpointing

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

**Concept**: Recompute activations during backward pass instead of storing them.

**Memory Savings:**
```
Memory Reduction = Activation Memory / sqrt(num_layers)
```

**Trade-offs:**
- **Memory**: 50-70% reduction in activation memory
- **Compute**: 30-50% increase in computation time
- **Use case**: Training and fine-tuning, less common for inference

### 5. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

From [GQA Paper](https://arxiv.org/pdf/2305.13245v2.pdf) and [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-01-31):

**Standard Multi-Head Attention (MHA):**
- Separate K, V for each attention head
- Full KV cache per head

**Multi-Query Attention (MQA):**
- **Single shared** K, V across all query heads
- **Memory reduction**: num_heads× smaller KV cache
- **Speed improvement**: Better memory bandwidth utilization
- **Accuracy**: Slight degradation (~1-2% on benchmarks)

**Grouped-Query Attention (GQA):**
- **Balance**: More KV heads than MQA, fewer than MHA
- **Example**: 8 query heads → 2 KV groups (4:1 ratio)
- **Accuracy**: Nearly matches MHA (< 0.5% degradation)
- **Memory**: 4× reduction in example above

**LLaMA 2 70B Uses GQA:**
- 64 query heads
- 8 key-value heads
- **8× KV cache reduction** vs MHA

## Model Parallelism Strategies

From [vLLM Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) and [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-01-31):

### Tensor Parallelism (TP)

**Purpose**: Split individual layers across multiple GPUs

**How It Works:**
- Partition weight matrices horizontally
- Each GPU computes a portion
- Reduction operations combine results

**Memory Distribution:**
```
Per-GPU Memory = Model Weights / TP_size + KV Cache / TP_size
```

**Best For:**
- Single-node multi-GPU setups
- Models too large for single GPU
- High-bandwidth GPU interconnect (NVLink)

**Example: LLaVA-70B on 4× A100 80GB**
- Weights per GPU: 140 GB / 4 = 35 GB
- KV Cache per GPU: Shared across GPUs
- Enables fitting 70B model on 4 GPUs

From [vLLM Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html):
```python
# vLLM tensor parallelism example
vllm serve model_name --tensor-parallel-size 4
```

### Pipeline Parallelism (PP)

**Purpose**: Split model layers across devices vertically

**How It Works:**
- Divide model into N sequential stages
- Each GPU handles subset of layers
- Forward pass: outputs flow through pipeline
- Microbatching reduces pipeline bubbles

**Memory Distribution:**
```
Per-GPU Memory = (Total Layers / PP_size) × Weight per Layer
```

**Challenges:**
- Pipeline bubbles (GPU idle time)
- Communication overhead between stages
- Load imbalancing

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):
- **Naive PP**: 40-50% pipeline bubble
- **With microbatching**: Reduced to 10-20%

### Hybrid Strategies

**Tensor + Pipeline Parallelism:**

From [Red Hat Developer](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm):

```
Total GPUs = TP_size × PP_size
```

**Example: 70B model on 8 GPUs**
- TP=4, PP=2
- Each TP group: 4 GPUs with 35 GB weights each
- Two pipeline stages

**Choosing Strategy:**

| Strategy | Best When | Memory Pattern |
|----------|-----------|----------------|
| TP only | Single node, high bandwidth | Even distribution |
| PP only | Multi-node, limited bandwidth | Per-stage allocation |
| TP+PP | Very large models (100B+) | Hierarchical split |

## Hardware-Specific Considerations

### GPU VRAM Capacities

From [AMD Blog](https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html) (accessed 2025-01-31):

| GPU Model | VRAM | Best For |
|-----------|------|----------|
| NVIDIA T4 | 16 GB | Small models (7B INT8) |
| NVIDIA V100 | 32 GB | Medium models (13B FP16) |
| NVIDIA A100 | 40/80 GB | Large models (70B with optimization) |
| NVIDIA H100 | 80 GB | Largest models (70B+ FP16) |
| AMD MI300X | 192 GB | Multi-modal, long context |

### Memory Bandwidth Impact

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

**Why Bandwidth Matters:**
- Inference is **memory-bandwidth bound**
- Compute speed exceeds memory transfer rate
- Higher bandwidth = better GPU utilization

**GPU Memory Hierarchy:**
```
L1 Cache (fastest, smallest) → SRAM
    ↓
L2 Cache (medium speed/size)
    ↓
HBM/GDDR VRAM (slower, largest)
```

FlashAttention exploits this hierarchy by keeping hot data in L1/L2.

## Practical Estimation Formula

From [UnfoldAI](https://unfoldai.com/gpu-memory-requirements-for-llms/) (accessed 2025-01-31):

**Quick Estimation (with quantization):**

```python
def estimate_memory_gb(
    num_params_billions: float,
    quantization_bits: int = 16,
    sequence_length: int = 2048,
    batch_size: int = 1,
    num_layers: int = 32,
    hidden_size: int = 4096,
    overhead_factor: float = 1.2
) -> float:
    """
    Estimate GPU memory requirements for VLM inference.

    Args:
        num_params_billions: Model size in billions of parameters
        quantization_bits: Bits per parameter (32, 16, 8, or 4)
        sequence_length: Maximum sequence length
        batch_size: Number of concurrent requests
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        overhead_factor: Memory overhead multiplier (default 1.2)

    Returns:
        Estimated memory in GB
    """
    # Model weights
    bytes_per_param = 4 / (32 / quantization_bits)
    model_memory_gb = num_params_billions * bytes_per_param

    # KV cache (2 for key+value, 2 bytes for FP16)
    kv_cache_per_token_kb = 2 * num_layers * hidden_size * 2 / 1024
    kv_cache_gb = (kv_cache_per_token_kb * sequence_length * batch_size) / (1024 * 1024)

    # Total with overhead
    total_memory_gb = (model_memory_gb + kv_cache_gb) * overhead_factor

    return total_memory_gb
```

**Example Usage:**

```python
# LLaVA-13B, FP16, 4k context, batch size 8
memory_required = estimate_memory_gb(
    num_params_billions=13,
    quantization_bits=16,
    sequence_length=4096,
    batch_size=8,
    num_layers=40,
    hidden_size=5120
)
# Result: ~87 GB (requires multi-GPU or INT8)
```

## Deployment Recommendations

### Small Models (7B-13B)

**Target**: Single GPU deployment

From [GitHub LLaVA](https://github.com/haotian-liu/LLaVA/issues/191):

- **7B FP16**: Fits on A100 40GB with batch size 4-8
- **7B INT8**: Fits on T4 16GB with batch size 1-2
- **13B FP16**: Requires A100 40GB for batch size 1-4
- **13B INT8**: Fits on V100 32GB with batch size 2-4

**Optimization Priority:**
1. Use FlashAttention for memory efficiency
2. Consider INT8 quantization for throughput
3. Enable PagedAttention for larger batches

### Medium Models (30B-70B)

**Target**: Multi-GPU with tensor parallelism

From [vLLM Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html):

- **30B FP16**: 2-4× A100 40GB with TP=2-4
- **70B FP16**: 4-8× A100 40GB with TP=4-8
- **70B INT8**: 2-4× A100 80GB with TP=2-4

**Optimization Priority:**
1. Tensor parallelism for weight distribution
2. GQA architecture if training/fine-tuning
3. PagedAttention for high throughput

### Large Models (175B+)

**Target**: Multi-node deployment

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

- **175B FP16**: 8-16× A100 80GB minimum
- **175B INT8**: 4-8× A100 80GB with aggressive optimization
- Hybrid TP+PP strategy required

**Optimization Priority:**
1. Tensor parallelism + Pipeline parallelism
2. INT8 or INT4 quantization essential
3. Advanced KV cache management
4. Continuous batching for throughput

## Common Memory Issues and Solutions

### Issue 1: Out of Memory (OOM) Errors

From [Reddit LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1l8t8n8/huge_vram_usage_with_vllm/):

**Symptoms:**
- `CUDA out of memory` errors
- Model fails to load
- Inference crashes mid-generation

**Solutions:**
1. **Reduce batch size**: Immediate memory relief
2. **Enable quantization**: INT8 reduces memory by 50%
3. **Increase tensor parallelism**: Distribute across more GPUs
4. **Reduce max sequence length**: Lower KV cache allocation
5. **Enable gradient checkpointing**: For training/fine-tuning

### Issue 2: Low GPU Utilization

From [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

**Symptoms:**
- GPU usage < 50%
- High latency despite available memory
- Memory-bandwidth bound

**Solutions:**
1. **Increase batch size**: Better parallelization
2. **Use FlashAttention**: Optimized memory movement
3. **Enable continuous batching**: Fill pipeline gaps
4. **Optimize KV cache**: Use PagedAttention

### Issue 3: Memory Fragmentation

From [vLLM PagedAttention Paper](https://arxiv.org/pdf/2309.06180.pdf):

**Symptoms:**
- Available memory exists but allocation fails
- Unpredictable OOM with variable sequence lengths
- 40-60% memory waste

**Solutions:**
1. **PagedAttention**: Non-contiguous allocation
2. **Fixed block sizes**: Reduce external fragmentation
3. **Memory pooling**: Pre-allocate common sizes
4. **Request batching**: Group similar sequence lengths

## Tools and Frameworks

### vLLM: High-Throughput Inference Engine

From [vLLM GitHub](https://github.com/vllm-project/vllm) (accessed 2025-01-31):

**Key Features:**
- PagedAttention implementation
- Continuous batching
- Tensor parallelism support
- FlashAttention integration
- Near-zero memory waste

**Memory Management:**
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="llava-hf/llava-1.5-13b-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,  # Use 90% of available VRAM
    max_model_len=4096,
    enforce_eager=False  # Use CUDA graphs
)
```

### TensorRT-LLM: Optimized Deployment

From [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (accessed 2025-01-31):

**Features:**
- INT8/INT4/FP8 quantization
- Custom CUDA kernels
- Multi-GPU support
- FlashAttention integration

**Memory Optimizations:**
- Weight-only quantization
- KV cache quantization
- Activation quantization
- Layer fusion

### Hugging Face Transformers

**Memory Utilities:**

```python
from transformers import AutoModelForVision2Seq
import torch

model = AutoModelForVision2Seq.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",
    torch_dtype=torch.float16,
    device_map="auto",  # Automatic device placement
    low_cpu_mem_usage=True
)

# Check memory usage
print(model.get_memory_footprint() / 1e9, "GB")
```

## Monitoring and Profiling

### NVIDIA Tools

From [NVIDIA Developer](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

**nvidia-smi**: Real-time GPU monitoring
```bash
# Watch memory usage
nvidia-smi -l 1  # Update every second

# Show detailed memory info
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

**NVIDIA Nsight Systems**: Detailed profiling
- Memory transfer visualization
- Kernel execution timeline
- Bottleneck identification

### Python Profiling

```python
import torch

# Track peak memory
torch.cuda.reset_peak_memory_stats()
# ... run inference ...
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")

# Memory summary
print(torch.cuda.memory_summary())
```

## Best Practices Summary

From [UnfoldAI](https://unfoldai.com/gpu-memory-requirements-for-llms/) and [NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/):

### Memory Planning

1. **Estimate first**: Use formulas before deployment
2. **Monitor continuously**: Track memory usage in production
3. **Plan for headroom**: Reserve 10-20% for safety
4. **Test configurations**: Validate with realistic workloads

### Optimization Order

1. **FlashAttention**: Nearly free memory savings
2. **PagedAttention**: Eliminates fragmentation
3. **Batch size tuning**: Maximize throughput
4. **Quantization**: If memory-constrained
5. **Model parallelism**: For very large models

### Production Deployment

1. **Start conservative**: Lower batch sizes initially
2. **Scale gradually**: Increase batch size monitoring memory
3. **Implement autoscaling**: Dynamic batch size adjustment
4. **Use continuous batching**: Maximize GPU utilization
5. **Monitor KV cache**: Watch for memory leaks

## Sources

**Research Papers:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (accessed 2025-01-31)
- [INT-FlashAttention: Enabling Flash Attention for INT8 Quantization](https://arxiv.org/abs/2409.16997) (accessed 2025-01-31)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/pdf/2305.13245v2.pdf) (accessed 2025-01-31)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf) (accessed 2025-01-31)

**Technical Blogs:**
- [NVIDIA Developer: Mastering LLM Techniques - Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (accessed 2025-01-31)
- [NVIDIA Developer: Floating-Point 8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-01-31)
- [UnfoldAI: GPU Memory Requirements for LLMs](https://unfoldai.com/gpu-memory-requirements-for-llms/) (accessed 2025-01-31)
- [Red Hat Developer: Distributed Inference with vLLM](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm) (accessed 2025-01-31)

**Documentation:**
- [vLLM Parallelism Documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) (accessed 2025-01-31)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) (accessed 2025-01-31)
- [NVIDIA NeMo Framework: Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) (accessed 2025-01-31)

**Community Resources:**
- [GitHub: LLaVA Memory Requirements Discussion](https://github.com/haotian-liu/LLaVA/issues/191) (accessed 2025-01-31)
- [Reddit LocalLLaMA: vLLM Memory Usage](https://www.reddit.com/r/LocalLLaMA/comments/1l8t8n8/huge_vram_usage_with_vllm/) (accessed 2025-01-31)
- [AMD Blog: Variable Graphics Memory FAQ](https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html) (accessed 2025-01-31)

**Additional References:**
- [Hugging Face Transformers: Multi-GPU Parallelism](https://huggingface.co/docs/transformers/en/perf_train_gpu_many) (accessed 2025-01-31)
- [DigitalOcean: Splitting LLMs Across Multiple GPUs](https://www.digitalocean.com/community/tutorials/splitting-llms-across-multiple-gpus) (accessed 2025-01-31)
