# Inference Speed and Memory Tradeoffs

## Overview

Vision-language model (VLM) inference presents significant computational and memory challenges due to the combined demands of visual and textual processing. Understanding the tradeoffs between speed, memory consumption, and model quality is essential for deploying VLMs effectively across different hardware configurations and use cases.

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1) (arXiv, accessed 2025-01-31):
- Efficient inference is critical as diffusion models and transformers grow in capacity and complexity
- Increased model complexity raises compute costs, latency, and memory requirements
- Techniques like quantization, pruning, and attention optimization can reduce overhead without major performance impacts

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization) (Red Hat Developer, accessed 2025-01-31):
- VLMs are computationally demanding, requiring more processing power and memory than language-only architectures
- Quantized VLMs achieve up to 3.5x speedups while recovering >99% accuracy
- The extra visual modality significantly increases memory and compute requirements

## Memory Requirements Breakdown

### Model Weights

Model weights dominate static memory consumption. The precision format directly impacts memory footprint:

**Weight Memory by Precision:**
- **FP32 (32-bit)**: 4 bytes per parameter
- **FP16/BF16 (16-bit)**: 2 bytes per parameter
- **INT8 (8-bit)**: 1 byte per parameter
- **INT4 (4-bit)**: 0.5 bytes per parameter

From [NVIDIA NIM for Vision Language Models Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html) (NVIDIA, accessed 2025-01-31):

**Llama 3.2 11B Vision Model:**
- BF16: ~60 GB GPU memory, 50 GB disk space
- Requires compute capability >= 7.0 (8.0 for bfloat16)

**Llama 3.2 90B Vision Model:**
- BF16: ~240 GB GPU memory, 200 GB disk space
- FP8: Can reduce memory requirements by ~2x

**Llama 4 Scout 17B (MoE):**
- Total parameters: 109B, Active parameters: 17B
- BF16: ~250 GB GPU memory
- FP8 (dynamic): Same memory as BF16 because quantization happens on-the-fly
- Memory based on total parameters, not just active parameters

**Memory Calculation Example:**
```python
# For a 7B parameter model
params = 7_000_000_000

# Different precision formats
fp32_memory = params * 4 / (1024**3)  # ~26 GB
fp16_memory = params * 2 / (1024**3)  # ~13 GB
int8_memory = params * 1 / (1024**3)  # ~6.5 GB
int4_memory = params * 0.5 / (1024**3)  # ~3.25 GB
```

### Activation Memory

Activation memory scales with batch size, sequence length, and hidden dimensions. For transformer models:

**Activation Memory Factors:**
- Batch size (linear scaling)
- Sequence length (linear to quadratic depending on attention implementation)
- Hidden dimension size
- Number of layers
- Intermediate layer sizes (e.g., FFN expansion factor)

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):
- Self-attention blocks can drive 40% of compute requirements (2.42 out of 6.07 GFLOPS for S/2 model)
- Attention complexity scales quadratically with token sequence length N and linearly with hidden layer size C: O(N²C)

**Typical Activation Memory:**
- Small batches (1-4): 1-5 GB
- Medium batches (8-16): 5-15 GB
- Large batches (32+): 15-40+ GB

### KV Cache Memory

The Key-Value (KV) cache stores computed keys and values from attention layers to avoid redundant computation during autoregressive generation.

From [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) (Hugging Face, accessed 2025-01-31):
- KV caching speeds up AI text generation by remembering past calculations and reusing them
- Stores intermediate states of attention layers during inference
- Benchmark results: 5.21x faster inference with KV caching enabled (T4 GPU)
- Process: Calculate once, store in cache, retrieve for subsequent tokens

**KV Cache Size Calculation:**
```python
# KV cache memory formula
batch_size = 1
seq_length = 2048  # context length
num_layers = 32
hidden_dim = 4096
num_kv_heads = 32
head_dim = hidden_dim // num_kv_heads
bytes_per_element = 2  # for FP16

kv_cache_size = (
    2 *  # keys and values
    batch_size *
    seq_length *
    num_layers *
    num_kv_heads *
    head_dim *
    bytes_per_element
) / (1024**3)  # Convert to GB

# For example: ~2 GB for the configuration above
```

**KV Cache Growth:**
- Scales linearly with sequence length
- Scales linearly with batch size
- Can become memory bottleneck for long contexts
- Larger for models with more attention heads

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):
- Vision models use adaptive average pooling to reduce tokens for efficiency
- Reducing context length (NIM_MAX_MODEL_LEN) helps when KV cache space is limited

### Total Memory Budget

**Complete Memory Formula:**
```
Total GPU Memory = Model Weights + Activations + KV Cache + Framework Overhead

Typical breakdown:
- Model weights: 50-70%
- KV cache: 15-30%
- Activations: 10-20%
- Framework overhead: 5-10%
```

## Inference Speed Factors

### Computation Patterns

**Memory-Bound vs. Compute-Bound:**

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):
- Smaller models (3B-7B) show modest gains (1.1-1.5x) due to lower memory/compute demands
- Larger models (72B+) achieve up to 3.5x speedups - more memory-bound
- Qwen2/2.5-VL models gain most from INT W4A16, suggesting memory-bound behavior

**Memory-Bound Operations:**
- Reading/writing model weights
- Loading KV cache from memory
- Small matrix multiplications (low arithmetic intensity)
- More common in inference, especially with small batch sizes

**Compute-Bound Operations:**
- Large matrix multiplications (high arithmetic intensity)
- Attention computations with large sequence lengths
- More common in training and large-batch inference

### Token Count Impact

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):
- Attention complexity: O(N²C) where N is token count, C is hidden dimension
- Vision transformers process many tokens from image patches
- Reducing token count through compression can significantly speed inference

**Vision Token Examples:**
- Patch size 16×16 on 512×512 image: 1,024 tokens
- Patch size 32×32 on 512×512 image: 256 tokens
- High-resolution documents (1680×2240): Thousands of tokens

### Batch Size Effects

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):

**Low-Latency Scenarios (small batches):**
- INT W4A16 provides lowest response times
- 1.3-1.7x speedup for Pixtral-12B
- Minimal inter-token latency at low request rates

**High-Throughput Scenarios (large batches):**
- W8A8 balances speed and efficiency
- Up to 3.4x higher throughput (Pixtral-Large on H100)
- Supports higher concurrent requests

**Batch Size Tradeoffs:**
```
Small Batches (1-4):
✓ Lower latency per request
✓ Better for interactive applications
✗ Lower GPU utilization
✗ Lower overall throughput

Large Batches (16+):
✓ Higher GPU utilization
✓ Better throughput
✗ Higher latency per request
✗ More memory required
```

### Precision Impact

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):

**Quantization Performance:**
- **FP8 W8A8**: Near-lossless performance, up to 3.5x faster, requires Ada Lovelace/Hopper GPUs
- **INT W8A8**: ~99% accuracy recovery, 1.3-1.5x faster, works on Ampere+ GPUs
- **INT W4A16**: >96% accuracy (larger models), 1.1-1.5x faster for lighter workloads

**Precision Recommendations:**
- FP32: Rarely used in inference (only for high-precision requirements)
- FP16/BF16: Standard baseline, good accuracy
- FP8/INT8: Best balance of speed and accuracy
- INT4: Maximum speed/memory savings, some accuracy loss on small models

## Optimization Techniques

### FlashAttention

FlashAttention is an IO-aware attention algorithm that significantly reduces memory access and speeds up computation.

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):
- FlashAttention leverages tiling and recomputation to speed up attention
- Reduces memory usage while maintaining exact attention computation
- Particularly effective for long sequences

**FlashAttention Benefits:**
- 2-4× faster attention computation
- Reduced memory usage (no need to store full attention matrix)
- Exact computation (not an approximation)
- Works by reordering operations and using GPU memory hierarchy efficiently

**FlashAttention Versions:**
- FlashAttention-1: Initial IO-aware implementation
- FlashAttention-2: Improved parallelization and efficiency
- FlashAttention-3: Hardware-specific optimizations (Hopper architecture)

**When FlashAttention Helps Most:**
- Long sequence lengths (>1024 tokens)
- Limited GPU memory
- Batch inference scenarios
- High-resolution vision tasks

### Quantization Strategies

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):

**Post-Training Quantization (PTQ):**
- No retraining required
- Quick deployment
- Slight accuracy degradation
- INT8: ~1% accuracy loss
- INT4: ~2-4% accuracy loss (model-dependent)

**Quantization-Aware Training (QAT):**
- Incorporates quantization during training
- Better accuracy recovery
- Requires full retraining
- Higher computational cost

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):
- Quantization reduces precision from 32-bit float to 8-bit integers
- Significantly reduces model size and memory usage
- PTQ is attractive for large models where retraining is expensive

**Practical Quantization Formula:**
```python
# Affine quantization transformation
q = round(x / S) + Z

where:
  x: real-valued weight
  S: scaling factor
  Z: zero-point
  q: 8-bit quantized value
```

### KV Cache Optimization

From [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching):

**KV Cache Implementation:**
```python
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            # Concatenate new keys/values
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)
```

**KV Cache Benefits:**
- Avoids redundant computation of past tokens
- 5.21x faster inference (benchmark on T4 GPU)
- Essential for autoregressive generation
- Enabled by default in transformers library

**KV Cache Tradeoffs:**
```
Memory Usage:
✗ Increases linearly with sequence length
✗ Can consume 15-30% of total memory
✓ Manageable with proper context length limits

Speed Gains:
✓ Dramatic speedup for long sequences
✓ More effective with longer generation
✓ Critical for real-time applications
```

### Attention Simplifications

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):

**Grouped Query Attention (GQA):**
- Multiple queries share same keys/values
- Reduces memory bandwidth when memory is bottleneck
- Maintains accuracy while reducing KV cache size

**Multi-Query Attention (MQA):**
- Single K-V head for all queries
- Maximum memory savings
- Slight accuracy degradation

**Mediated Attention:**
- Uses "mediator tokens" to reduce quadratic complexity
- Two-step attention: T×K^T, then Q×T^T
- Complexity: O(NnC) where n << N
- Effective for models where C > N

**Focused Group/Multi-Query Attention:**
- Combines polynomial focusing with grouped queries
- Eliminates softmax, allows K^T×V first
- Complexity: O(NC²) with grouping factor G
- Reduces computation by G² for K^T×V step

### Pruning Strategies

From [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1):

**L2 Norm Pruning:**
- Removes weights with smallest L2 norms
- Unstructured pruning adds sparsity without changing architecture
- Attention head pruning can significantly reduce parameters

**Pruning Results:**
- Even single pruned attention head introduces artifacts
- Suggests attention heads are critical for representational power
- Memory and compute reduction comes with quality-efficiency tradeoff

**Pruning Guidelines:**
- More effective for larger models with excess capacity
- Smaller models (6-head attention) more sensitive to pruning
- Structured pruning (removing entire heads) easier to accelerate than unstructured

## Practical GPU Guidelines

### GPU Selection by Model Size

From [NVIDIA NIM for Vision Language Models Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html):

**Small Models (7-11B parameters):**
- Single GPU: L40S (48GB), A100 (80GB)
- Consumer: Not typically feasible
- Quantization: INT8/INT4 enables consumer hardware

**Medium Models (17-24B parameters):**
- Professional: 1-2× A100/H100
- Quantization helps significantly
- Example: Mistral Small 3.2 24B needs 68GB BF16

**Large Models (70-90B parameters):**
- 4-8× A100 (80GB) or 2-4× H100 (80GB)
- FP8 can halve GPU count
- Example: Llama 3.2 90B needs 240GB BF16

**Mixture-of-Experts Models:**
- Memory based on total parameters, not active
- Llama 4 Scout (109B total, 17B active): 250GB
- Llama 4 Maverick: Requires 8× H100/H200

### Hardware Tier Recommendations

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):

**Consumer/Entry-Level (12-24GB VRAM):**
- Quantized small models only (INT4/INT8)
- Example: 7B model with INT4 quantization
- Limited batch size and context length
- GPUs: RTX 3090, RTX 4090, A10G

**Professional (40-48GB VRAM):**
- Medium models with quantization
- Small models at full precision
- Moderate batch sizes
- GPUs: A40, L40S, RTX A6000

**Data Center (80GB+ VRAM):**
- Large models with quantization
- Medium models at full precision
- High throughput scenarios
- GPUs: A100, H100, H200

**Performance by GPU Tier:**
```
Lower-tier GPUs (A6000):
✓ Greater gains from quantization
✓ Memory bottlenecks reduced by INT W4A16/W8A8
✓ Can serve more requests in parallel with quantization

High-tier GPUs (H100):
✓ Better absolute performance
✓ Still benefit from quantization
✓ FP8 native support on Ada Lovelace/Hopper
```

### Deployment Scenario Guidelines

From [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization):

**Low-Latency (Interactive Applications):**
- Small batch sizes (1-4)
- INT W4A16 provides lowest response times
- Prioritize single-request speed
- Examples: Chatbots, real-time analysis
- Speedup: 1.3-1.7x (Pixtral-12B)

**High-Throughput (Batch Processing):**
- Large batch sizes (16+)
- W8A8 formats provide best throughput
- Prioritize requests per second
- Examples: Document processing, bulk analysis
- Speedup: 1.9-3.4x (Pixtral-Large)

**Balanced (Real-World Production):**
- Medium batch sizes (4-8)
- Balance latency and throughput
- INT W8A8 recommended
- Maintains efficiency across request rates

**Workload-Specific Performance:**

Document VQA (1680×2240, high-resolution):
- Most compute-intensive
- W8A8 delivers highest speedups
- INT W4A16 competitive at low latency

Visual Reasoning (640×480, moderate):
- Medium compute requirements
- Both W8A8 and W4A16 perform well
- Choice depends on batch size

Image Captioning (480×360, light):
- Lightest workload
- INT W4A16 comparable to W8A8
- Memory-bound more than compute-bound

### Memory vs. Speed Optimization

**Optimizing for Memory:**
```
Priority: Fit model in available GPU memory

Techniques:
1. INT4 quantization (4× memory reduction)
2. Reduce context length
3. Smaller batch size
4. Enable KV cache offloading

Trade-offs:
✓ Can run larger models
✓ Lower memory footprint
✗ May sacrifice some speed
✗ Possible accuracy impact (INT4)
```

**Optimizing for Speed:**
```
Priority: Maximize throughput/minimize latency

Techniques:
1. FP8/INT8 quantization (best speed/accuracy)
2. FlashAttention
3. Larger batch sizes (throughput)
4. Enable KV caching

Trade-offs:
✓ Faster inference
✓ Better GPU utilization
✗ Higher memory requirements
✗ May need more powerful GPU
```

**Balanced Approach:**
```
Priority: Best overall efficiency

Configuration:
- INT8 quantization
- KV caching enabled
- FlashAttention when available
- Batch size matched to latency requirements
- Context length appropriate for use case

Results:
✓ Good speed and accuracy
✓ Reasonable memory usage
✓ Works on professional-tier GPUs
✓ Suitable for most production scenarios
```

### Context Length Considerations

From [NVIDIA NIM for Vision Language Models Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html):

**Memory Impact of Context Length:**
- KV cache grows linearly with context length
- Longer contexts require more memory
- Can reduce NIM_MAX_MODEL_LEN when memory-constrained

**Context Length by Model:**
- Llama 3.2: 128K tokens maximum
- Llama 4 Maverick: 1M tokens (H200), 430K tokens (H100)
- Cosmos Reason1: Context length varies by precision

**Reducing Context Length:**
```bash
# Set maximum model length to reduce KV cache size
export NIM_MAX_MODEL_LEN=32768  # Instead of default 128K

# Trade-offs:
✓ Significant memory savings
✓ Faster iteration times
✗ Cannot process longer documents
✗ May need chunking strategies
```

## Practical Configuration Examples

### Example 1: Chatbot on Consumer Hardware

**Goal**: Interactive chatbot with 7B VLM on RTX 4090 (24GB)

**Configuration:**
```
Model: 7B vision-language model
Quantization: INT4 (W4A16)
Batch size: 1
Context length: 8K tokens
KV caching: Enabled

Memory breakdown:
- Model weights: ~3.5 GB
- KV cache: ~2 GB
- Activations: ~1 GB
- Framework: ~1 GB
Total: ~7.5 GB (comfortable fit)

Performance:
- Latency: ~50-100ms per token
- Throughput: 10-20 tokens/second
- Quality: ~96% of BF16 baseline
```

### Example 2: Document Analysis Pipeline

**Goal**: High-throughput document processing with 24B model on 4× A100

**Configuration:**
```
Model: 24B vision-language model
Quantization: INT8 (W8A8)
Batch size: 16
Context length: 32K tokens
KV caching: Enabled
FlashAttention: Enabled

Memory breakdown (per GPU):
- Model weights: ~12 GB (distributed)
- KV cache: ~25 GB
- Activations: ~15 GB
- Framework: ~3 GB
Total per GPU: ~55 GB

Performance:
- Throughput: 100+ requests/hour
- Latency: ~2-3s per document
- Quality: >99% of BF16 baseline
- Speedup: 2.2× over BF16
```

### Example 3: Research/Development Setup

**Goal**: Maximum flexibility for model experimentation on H100

**Configuration:**
```
Model: 11B vision-language model
Quantization: BF16 (full precision)
Batch size: 4
Context length: 128K tokens
KV caching: Enabled
FlashAttention: Enabled

Memory breakdown:
- Model weights: ~22 GB
- KV cache: ~30 GB
- Activations: ~8 GB
- Framework: ~3 GB
Total: ~63 GB

Performance:
- Latency: ~80-120ms per token
- Throughput: 8-12 tokens/second
- Quality: Full baseline accuracy
- Flexibility: Can experiment with settings
```

## Monitoring and Profiling

**Key Metrics to Track:**
```python
# Memory monitoring
torch.cuda.max_memory_allocated()  # Peak memory
torch.cuda.memory_summary()  # Detailed breakdown

# Speed profiling
import time
start = time.time()
output = model.generate(...)
latency = time.time() - start

# Throughput calculation
tokens_generated = len(output)
throughput = tokens_generated / latency  # tokens/second
```

**Performance Indicators:**
- **Memory utilization**: 70-90% is optimal (room for KV cache growth)
- **GPU compute**: >80% during inference is good
- **Batch size**: Increase until memory limit or latency constraints
- **Latency**: Monitor p50, p95, p99 for tail latencies

## Sources

**Source Documents:**
- None (web research only for this PART)

**Web Research:**
- [Optimizing Inference in Transformer-Based Models](https://arxiv.org/html/2509.17894v1) - arXiv:2509.17894 (accessed 2025-01-31)
- [Enable 3.5 times faster vision language models with quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization) - Red Hat Developer (accessed 2025-01-31)
- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching) - Hugging Face Blog (accessed 2025-01-31)
- [Support Matrix — NVIDIA NIM for Vision Language Models (VLMs)](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html) - NVIDIA Documentation (accessed 2025-01-31)

**Additional References:**
- FlashAttention papers and implementations on GitHub
- vLLM documentation for inference optimization
- NVIDIA TensorRT-LLM optimization guides
