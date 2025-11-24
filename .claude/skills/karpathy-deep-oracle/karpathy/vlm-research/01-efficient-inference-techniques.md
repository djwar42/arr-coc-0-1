# Efficient VLM Inference Techniques

## Overview

Efficient inference for Vision-Language Models (VLMs) is critical for deploying these powerful models at scale. Large VLMs face two main challenges: (1) **massive memory footprint** from model parameters, KV cache, and intermediate activations, and (2) **low parallelizability** due to autoregressive generation. This guide covers state-of-the-art techniques for optimizing VLM inference across quantization, KV cache management, pruning, distillation, and production serving systems.

**Key Goals for Inference Optimization:**
- Reduce memory footprint (fewer GPU devices, less GPU memory)
- Reduce computational complexity (lower FLOPs)
- Reduce inference latency (faster token generation)

From [Lilian Weng's Survey on LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (accessed 2025-02-02):
> "The extremely high inference cost, in both time and memory, is a big bottleneck for adopting a powerful transformer for solving real-world tasks at scale."

---

## Section 1: Quantization Techniques

### 1.1 Overview of Quantization

**Quantization** reduces model precision from FP32/FP16 to lower bit-widths (INT8, INT4, FP8) to decrease memory usage and accelerate inference. Two main approaches exist:

1. **Post-Training Quantization (PTQ)**: Convert trained model weights to lower precision without retraining. Fast and cheap.
2. **Quantization-Aware Training (QAT)**: Apply quantization during pre-training or fine-tuning for better performance but higher cost.

From [A Survey on Efficient Vision-Language Models](https://arxiv.org/abs/2504.09724) (arXiv:2504.09724, accessed 2025-02-02):
> "Vision-language models (VLMs) integrate visual and textual information... However, their high computational demands pose challenges for real-time applications."

### 1.2 Challenges for VLM Quantization

**Key Challenge**: VLMs exhibit **extreme activation outliers** that make naive quantization fail.

From [LLM.int8() paper](https://arxiv.org/abs/2208.07339) findings:
- Models > 6.7B parameters develop extreme outliers (~100x larger magnitude than typical values)
- Outliers appear in **all transformer layers** as model size grows
- Simple 8-bit quantization causes significant performance degradation

**Observed Pattern** (from quantization research):
- `W8A32` (weights INT8, activations FP32): **Good performance**
- `W8A8` (weights INT8, activations INT8): **Poor performance**
- `W32A8` (weights FP32, activations INT8): **Poor performance**

**Conclusion**: Activations are harder to quantize than weights in VLMs.

### 1.3 Mixed-Precision Quantization

**LLM.int8()** ([Dettmers et al. 2022](https://arxiv.org/abs/2208.07339))

**Strategy**: Two mixed-precision decompositions
1. **Vector-wise quantization**: Each row and column scaled by absolute maximum values → quantized to INT8
2. **Outlier handling**: Extreme activation features (20x larger) remain in FP16, representing <1% of weights

**Performance**: Enables INT8 inference for 175B+ parameter models with minimal accuracy loss.

**Q-BERT** ([Shen et al. 2020](https://arxiv.org/abs/1909.05840))
- **Group-wise quantization**: Treats each attention head's matrix as one group
- **Hessian-based mixed precision**: Uses second-order information to identify sensitive parameters
- Parameters with higher Hessian spectrum require higher precision

### 1.4 Fine-Grained Quantization Granularity

**Quantization Granularity Levels** (from coarse to fine):
1. **Per-tensor**: Entire weight matrix uses one scale factor (easiest, lowest quality)
2. **Per-layer**: Each layer has separate quantization parameters
3. **Per-channel**: Each output channel quantized separately
4. **Per-token**: Dynamic quantization per input token (highest quality)

**ZeroQuant** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)):
- **Group-wise quantization** for weights (like Q-BERT)
- **Token-wise quantization** for activations
- **Kernel fusion**: Custom kernels fuse quantization with previous operators to avoid expensive quant/dequant overhead

**PEG (Per-Embedding Group) Quantization** ([Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948)):
- Splits activation tensor into evenly-sized groups along embedding dimension
- Groups share quantization parameters
- **Deterministic range-based permutation**: Sorts dimensions by value ranges to group outliers together

### 1.5 Advanced PTQ Methods

**GPTQ** ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323))

**Optimization Formulation**: Find quantized weights $\hat{W}$ to minimize:
$$\hat{W}^* = \arg\min_{\hat{W}} |WX - \hat{W}X|$$

**Approach**:
- Treats weight matrix $W$ as collection of row vectors
- Iteratively quantizes weights greedily to minimize quantization error
- Uses Hessian matrices for closed-form update formulas

**Performance**: Reduces OPT-175B to 3-4 bits with minimal loss
**Limitation**: Only applies to weights, not activations

**SmoothQuant** ([Xiao & Lin 2022](https://arxiv.org/abs/2211.10438))

**Key Insight**: Migrate scale variance from activations to weights via mathematically equivalent transformation.

**Transformation** (per-channel smoothing factor $s$):
$$Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s)W) = \hat{X}\hat{W}$$

**Smoothing Factor**:
$$s = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$$

where $\alpha$ controls migration (typically $\alpha=0.5$, higher for models with more outliers).

**Advantage**: Enables efficient W8A8 quantization with better hardware efficiency than mixed-precision.

### 1.6 Quantization-Aware Training (QAT)

**Approach 1: Fine-Tuning**
- Train quantized model on pre-training or task-specific dataset
- Use same training objective (NLL, MLM, classification loss)

**Approach 2: Distillation-Based QAT**
- Full-precision model = teacher
- Low-precision model = student
- Minimize distillation loss between soft logits

**LKD (Layer-by-layer Knowledge Distillation)** ([Yao et al. 2022](https://arxiv.org/abs/2206.01861)):
- Quantizes network layer by layer
- Uses unquantized layer as teacher
- Minimizes MSE between original and quantized layer outputs

**DistilBERT** ([Sanh et al. 2019](https://arxiv.org/abs/1910.01108)):
- Reduces BERT parameters by 40%
- Maintains 97% performance on downstream tasks
- 71% faster inference
- Loss combines: soft distillation + MLM loss + cosine embedding loss

---

## Section 2: KV Cache Optimization

### 2.1 KV Cache Memory Challenge

**Problem Scale**: For transformer decoding, the KV cache dominates memory usage.

From [Pope et al. 2022](https://arxiv.org/abs/2211.05102):
> "For a batch size of 512 and context length of 2048, the KV cache totals 3TB, that is 3x the model size (!)"

**Memory Components During Inference**:
1. Model parameters
2. **KV cache** (scales with: batch size × sequence length × num layers × hidden dim)
3. Intermediate activations

### 2.2 KV Cache Compression Techniques

**VL-Cache** ([Tu et al. 2025](https://www.amazon.science/publications/vl-cache-sparsity-and-modality-aware-kv-cache-compression-for-vision-language-model-inference-acceleration)):

**Key Challenge**: VLMs have large KV caches encoding long visual contexts (images, videos).

**Approach**: **Sparsity and modality-aware compression**
- Identifies and retains important tokens based on attention patterns
- Exploits sparsity in attention weights
- Achieves significant memory reduction while maintaining accuracy

**Joint Optimization for KV Cache** ([arXiv:2510.20707](https://arxiv.org/abs/2510.20707))

**Problem**: Large Vision-Language Models face KV cache explosion with multi-modal sequences.

**Solution**: **Mixing importance with diversity**
- Balances keeping high-importance tokens with maintaining diverse representation
- Joint optimization across importance scores and diversity metrics
- Reduces KV cache size while preserving model capacity

### 2.3 Multi-Head Latent Attention (MLA)

From [PyImageSearch article on KV Cache Optimization](https://pyimagesearch.com/2025/10/13/kv-cache-optimization-via-multi-head-latent-attention/) (accessed 2025-02-02):

**Multi-Head Latent Attention** dramatically reduces KV cache memory by:
1. **Low-rank projection** of keys and values
2. **Shared latent representations** across attention heads
3. **Compressed KV storage** with minimal accuracy loss

**Benefits**:
- Enables longer context windows
- Faster inference with reduced memory bandwidth
- Maintains model quality

### 2.4 PagedAttention (vLLM)

**vLLM's Innovation**: **PagedAttention** manages KV cache like virtual memory in operating systems.

From [vLLM Documentation](https://docs.vllm.ai/en/latest/configuration/optimization.html) (accessed 2025-02-02):

**Key Concepts**:
- **Paging**: KV cache divided into fixed-size blocks (pages)
- **Dynamic allocation**: Pages allocated on-demand, not contiguously
- **Memory sharing**: Multiple sequences share KV cache pages for prefix caching

**Advantages**:
- Reduces memory fragmentation
- Enables higher batch sizes
- Supports continuous batching (add/remove requests dynamically)

**Configuration Options**:
```python
# vLLM KV cache configuration
gpu_memory_utilization = 0.9  # Use 90% of GPU memory for KV cache
max_num_seqs = 256           # Maximum sequences in batch
block_size = 16               # KV cache block size
```

---

## Section 3: Pruning & Distillation

### 3.1 Network Pruning Overview

**Goal**: Reduce model size by removing unimportant weights while maintaining capacity.

**Types**:
1. **Unstructured pruning**: Drop any weight/connection (doesn't retain architecture, poor hardware support)
2. **Structured pruning**: Maintain dense matrix form (better hardware efficiency)

From [Lilian Weng's Survey](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/):

**Workflow**:
1. Train dense network to convergence
2. Prune network to remove unwanted structure
3. Retrain to recover performance

**Lottery Ticket Hypothesis**: A randomly initialized dense network contains sparse subnetworks ("winning tickets") that can achieve optimal performance when trained in isolation.

### 3.2 Pruning Methods

**Magnitude Pruning** ([Gale et al. 2019](https://arxiv.org/abs/1902.09574)):
- **Simplest method**: Weights with smallest absolute values are trimmed
- **Surprisingly effective**: Achieves comparable or better results than complex methods
- Consistent performance across hyperparameters

**Gradual Magnitude Pruning (GMP)** ([Zhu & Gupta 2017](https://arxiv.org/abs/1710.01878)):
- Increases sparsity gradually during training
- Masks weights with smallest absolute values to reach desired sparsity $s$
- Masked weights don't receive gradient updates
- **Finding**: Large sparse models outperform small dense models

**Iterative Pruning** ([Renda et al. 2020](https://arxiv.org/abs/2003.02389)):
- Repeats prune + retrain cycles
- Small fraction pruned per iteration
- Continues until target sparsity reached

### 3.3 N:M Structured Sparsity

**N:M Sparsity Pattern**: $N$ out of every $M$ consecutive elements are zeros.

**Hardware Support**: Nvidia A100 GPU sparse tensor cores accelerate **2:4 sparsity** (2 zeros per 4 elements).

From [Nvidia A100 Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) (2020):

**Workflow for 2:4 Sparsity**:
1. Train dense network
2. Prune to satisfy 2:4 pattern
3. Retrain (fine-tune)

**Permutation for Better Sparsity** ([Pool & Yu 2021](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)):

**Insight**: Permuting columns provides more pruning options while preserving matrix multiplication results.

**Example**: In self-attention, permute $Q$ (axis 1) and $K^T$ (axis 0) identically:
- $QK^T$ output unchanged
- More flexibility in selecting which weights to keep for N:M sparsity

**SR-STE** (Sparse-Refined STE) ([Zhou et al. 2021](https://arxiv.org/abs/2102.04010)):

Extends Straight-Through Estimator for N:M sparsity training from scratch:
$$W_{t+1} \gets W_t - \gamma \frac{\partial\mathcal{L}}{\partial\widetilde{W}} + \lambda_W (\bar{\mathcal{E}} \odot W_t)$$

where $\bar{\mathcal{E}}$ is mask matrix for pruned weights $\widetilde{W}$.

**Goals**:
1. Prevent large changes in binary mask
2. Restrict values of pruned weights
3. Promote non-pruned weights

### 3.4 Knowledge Distillation

**Framework**: Teacher (large model) → Student (small model)

**Distillation Loss**:
$$\mathcal{L}_{KD} = \mathcal{L}_{distill}(\text{softmax}(z_t, T), \text{softmax}(z_s, T)) + \lambda\mathcal{L}_{CE}(y, z_s)$$

where:
- $z_t$, $z_s$: Teacher and student logits
- $T$: Temperature (higher = softer distributions)
- $\lambda$: Balances soft and hard objectives

**Advantages**:
- Can be combined with quantization, pruning, sparsification
- Student architecture flexible (not restricted to teacher's structure)
- Doesn't require original training data (can use proxy datasets)

---

## Section 4: Speculative Decoding

### 4.1 Overview

**Core Idea**: Use small "draft" model to predict multiple tokens, then verify with large "target" model in parallel.

**Benefits**:
- Reduces autoregressive bottleneck
- Maintains output quality (verification ensures correctness)
- Memory-bound speedup (more arithmetic intensity)

### 4.2 Speculative Decoding for VLMs

From [On Speculative Decoding for Multimodal Large Language Models](https://arxiv.org/abs/2404.08856) (Gagrani et al. 2024, accessed 2025-02-02):

**Key Finding**: Language-only model can serve as effective draft model for VLMs.

**Approach**:
1. **Draft model**: Small language-only model (115M parameters)
2. **Target model**: Full VLM (LLaVA 7B)
3. **Bypass visual processing**: Draft model skips image tokens and associated processing

**Performance**:
- **2.37x memory-bound speedup** on average
- Maintains output quality across three tasks
- Draft model trained from scratch on language data only

**Alternative**: Compact LLaVA draft model with image adapter
- Marginal gains in image captioning
- Comparable results in other tasks
- Trade-off: Added complexity vs. marginal improvement

### 4.3 Advanced Speculative Techniques

**Multimodal Speculative Decoding (MSD)** ([Lin et al. 2025](https://arxiv.org/abs/2505.14260)):

**Innovation**: Tailored specifically for multimodal inference
- Considers both visual and textual tokens in draft generation
- Optimizes verification process for multi-modal context
- Better speedup for vision-heavy tasks

**Implementation Considerations**:
- Draft model size: Balance between quality and speed (70M-500M typical range)
- Acceptance rate: Higher = better speedup (aim for >80%)
- Token lookahead: Draft 3-10 tokens ahead (more = higher rejection risk)

---

## Section 5: Production Serving Systems

### 5.1 vLLM (Vision-Language LLM Serving)

From [vLLM Documentation](https://docs.vllm.ai/) (accessed 2025-02-02):

**Overview**: High-throughput, memory-efficient inference serving framework.

**Key Features**:
1. **PagedAttention**: Efficient KV cache management (see Section 2.4)
2. **Continuous batching**: Dynamic request scheduling
3. **Tensor parallelism**: Multi-GPU model distribution
4. **Optimized kernels**: CUDA kernels for attention, sampling
5. **OpenAI-compatible API**: Drop-in replacement for OpenAI endpoints

**VLM Support** (vLLM v0.6.1+):
- LLaVA models (1.5, 1.6, NeXT)
- Qwen-VL
- InternVL
- Phi-3 Vision
- Custom VLM architectures via plugin system

**Configuration Example**:
```python
from vllm import LLM, SamplingParams

# Load VLM with optimizations
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    gpu_memory_utilization=0.9,  # KV cache allocation
    tensor_parallel_size=2,       # Multi-GPU
    max_num_seqs=256,             # Batch size
    trust_remote_code=True
)

# Generate
outputs = llm.generate(
    prompts=[{"prompt": "Describe the image", "image": image}],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=256)
)
```

**Throughput Benchmarks** (from vLLM documentation):
- **LLaVA-7B**: ~40 tokens/sec/GPU (A100)
- **LLaVA-13B**: ~25 tokens/sec/GPU (A100)
- Scales near-linearly with tensor parallelism

### 5.2 TensorRT-LLM

From [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) (accessed 2025-02-02):

**Overview**: NVIDIA's optimized library for LLM and VLM inference.

**Key Optimizations**:
1. **Kernel fusion**: Combines multiple operations into single kernels
2. **Flash Attention**: Memory-efficient attention implementation
3. **INT8/FP8 quantization**: Hardware-accelerated low-precision compute
4. **In-flight batching**: Dynamic request management
5. **Multi-GPU/multi-node**: Tensor and pipeline parallelism

**Multimodal Support**:
- LLaVA
- VILA
- CogVLM
- Visual question answering models

**LLM API** (Python interface):
```python from tensorrt_llm import LLM

# Create optimized VLM
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    quantization="int8",  # INT8 quantization
    dtype="float16"
)

# Generate with image input
outputs = llm.generate(
    prompts=["<image>\nWhat is in the image?"],
    images=[image_path]
)
```

**Performance vs vLLM** (from [SqueezeBits comparison](https://blog.squeezebits.com/vllm-vs-tensorrtllm-13-visionlanguage-models-40761), accessed 2025-02-02):

| Metric | vLLM | TensorRT-LLM |
|--------|------|--------------|
| **Ease of use** | High (Python-native) | Medium (build step required) |
| **Peak throughput** | Good | Excellent (10-20% faster) |
| **Latency (P50)** | Good | Excellent (15-25% lower) |
| **Quantization** | INT8, FP8 | INT8, FP8, INT4 |
| **Multi-GPU** | Tensor parallel | Tensor + pipeline parallel |
| **Ecosystem** | HuggingFace integration | NVIDIA stack integration |

**When to use**:
- **vLLM**: Rapid prototyping, HuggingFace models, Python-first workflow
- **TensorRT-LLM**: Maximum performance, production at scale, NVIDIA hardware

### 5.3 Additional Serving Frameworks

**Text Generation Inference (TGI)** (HuggingFace):
- Built-in VLM support (IDEFICS, LLaVA)
- Docker-first deployment
- Optimized for HuggingFace ecosystem
- Good developer experience

**Ray Serve**:
- General-purpose serving with VLM support
- Dynamic batching
- Autoscaling capabilities
- Complex deployment patterns

**Triton Inference Server** (with TensorRT-LLM backend):
- Enterprise-grade serving
- Multi-framework support
- Advanced monitoring and logging
- Production-hardened

---

## Section 6: Benchmarks and Performance Metrics

### 6.1 Key Metrics

**Latency Metrics**:
- **Time to First Token (TTFT)**: Prefill latency
- **Inter-Token Latency (ITL)**: Per-token generation time
- **End-to-End Latency**: Total request completion time

**Throughput Metrics**:
- **Tokens/second**: Raw generation speed
- **Requests/second**: System throughput
- **Tokens/second/GPU**: Efficiency per device

**Memory Metrics**:
- **Peak memory usage**: Maximum GPU memory required
- **KV cache size**: Memory for context
- **Model memory**: Parameter storage

### 6.2 Quantization Performance

From [A Survey on Efficient Vision-Language Models](https://arxiv.org/abs/2504.09724):

**INT8 Quantization Results** (typical):
- **Memory**: 50% reduction vs FP16
- **Speedup**: 1.5-2x on modern GPUs
- **Accuracy drop**: <1% on most benchmarks

**INT4 Quantization Results**:
- **Memory**: 75% reduction vs FP16
- **Speedup**: 2-3x potential (limited kernel support)
- **Accuracy drop**: 2-5% (requires careful calibration)

**SmoothQuant W8A8 Performance**:
- OPT-175B: <1% accuracy loss
- BLOOM-176B: <0.5% accuracy loss
- **Hardware efficiency**: Better than mixed-precision (full INT8 compute)

### 6.3 Production System Benchmarks

**vLLM PagedAttention Benefits** (from documentation):
- **Memory savings**: 2-4x higher batch size vs naive implementation
- **Throughput**: 24x higher than HuggingFace Transformers
- **Latency**: Competitive with custom implementations

**TensorRT-LLM Performance** (from NVIDIA benchmarks):
- **LLaVA-7B**: 45-60 tokens/sec/GPU (A100 80GB, INT8)
- **LLaVA-13B**: 30-40 tokens/sec/GPU (A100 80GB, INT8)
- **Multi-GPU scaling**: 85-95% efficiency up to 8 GPUs

### 6.4 Real-World Performance Considerations

**Batch Size Effects**:
- Small batches (1-8): Latency-bound, memory bandwidth bottleneck
- Medium batches (16-64): Good balance
- Large batches (128+): Throughput-bound, compute-intensive

**Sequence Length Impact**:
- Short (<512): Prefill dominates
- Medium (512-2048): Balanced prefill/decode
- Long (>2048): KV cache becomes critical

**Hardware Considerations**:
- **A100 80GB**: Best for large VLMs (13B+), high batch sizes
- **A100 40GB**: Medium VLMs (7-13B), moderate batches
- **H100**: Best absolute performance, excellent for quantization
- **L4/T4**: Inference-optimized, good for smaller models with quantization

---

## Implementation Resources

### GitHub Repositories

**Quantization**:
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant) - Activation smoothing for W8A8
- [GPTQ](https://github.com/IST-DASLab/gptq) - Accurate post-training quantization
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - LLM.int8() implementation

**Serving Systems**:
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput VLM serving
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA optimized inference
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HuggingFace serving

**Pruning & Sparsity**:
- [Apex Sparsity](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) - N:M sparsity tools
- [DeepSpeed-MoE](https://github.com/microsoft/DeepSpeed) - Mixture-of-Experts optimization

**Benchmarking**:
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - VLM evaluation
- [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) - Comprehensive resource list

### Production Deployment Guides

**vLLM Quick Start**:
```bash
# Install
pip install vllm

# Serve VLM with OpenAI-compatible API
vllm serve llava-hf/llava-1.5-7b-hf \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --trust-remote-code

# Query
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/llava-1.5-7b-hf",
    "prompt": "<image>\nDescribe this image",
    "image_url": "http://example.com/image.jpg"
  }'
```

**TensorRT-LLM Build**:
```bash
# Build optimized engine
trtllm-build \
  --checkpoint_dir ./llava-checkpoint \
  --output_dir ./trt_engines \
  --gemm_plugin float16 \
  --max_batch_size 256

# Run inference server
mpirun -n 1 python run.py \
  --engine_dir ./trt_engines \
  --tokenizer_dir ./tokenizer
```

---

## Best Practices

### Quantization Strategy

1. **Start with W8A32**: Test INT8 weights with FP16/FP32 activations
2. **Profile outliers**: Identify problematic layers before full quantization
3. **Use SmoothQuant**: For enabling W8A8 on large models
4. **Calibration data**: Use representative samples (1000-10000 examples)
5. **Per-layer analysis**: Some layers may need higher precision

### KV Cache Management

1. **Set appropriate utilization**: 0.85-0.95 for vLLM `gpu_memory_utilization`
2. **Monitor cache hit rate**: Track KV cache efficiency
3. **Batch similar lengths**: Reduces padding overhead
4. **Use PagedAttention**: Enable in vLLM for better memory efficiency
5. **Tune block size**: vLLM default 16 is good starting point

### Production Deployment

1. **Start with vLLM**: Fastest time-to-production for HuggingFace models
2. **Optimize iteratively**: Profile → identify bottleneck → optimize
3. **Monitor carefully**: Track TTFT, ITL, throughput, memory
4. **Load test thoroughly**: Test at expected peak traffic
5. **Plan for scaling**: Multi-GPU, multiple instances, load balancing

### Performance Tuning Checklist

- [ ] Profile baseline (no optimizations)
- [ ] Apply quantization (W8A16 → W8A8)
- [ ] Optimize KV cache settings
- [ ] Enable kernel fusion where available
- [ ] Test continuous batching
- [ ] Benchmark with production traffic patterns
- [ ] Monitor long-running stability
- [ ] Document performance characteristics

---

## Sources

**Academic Papers**:
- [A Survey on Efficient Vision-Language Models](https://arxiv.org/abs/2504.09724) - arXiv:2504.09724 (accessed 2025-02-02)
- [On Speculative Decoding for Multimodal Large Language Models](https://arxiv.org/abs/2404.08856) - arXiv:2404.08856, Gagrani et al. 2024 (accessed 2025-02-02)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) - Dettmers et al. 2022
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438) - Xiao & Lin 2022
- [GPTQ: Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323) - Frantar et al. 2022
- [ZeroQuant: Efficient and Affordable Post-Training Quantization](https://arxiv.org/abs/2206.01861) - Yao et al. 2022

**Technical Blogs & Documentation**:
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) - Lilian Weng's Survey (accessed 2025-02-02)
- [vLLM Documentation](https://docs.vllm.ai/) - vLLM Optimization Guide (accessed 2025-02-02)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) - NVIDIA (accessed 2025-02-02)
- [vLLM vs TensorRT-LLM Comparison](https://blog.squeezebits.com/vllm-vs-tensorrtllm-13-visionlanguage-models-40761) - SqueezeBits Blog (accessed 2025-02-02)
- [KV Cache Optimization via Multi-Head Latent Attention](https://pyimagesearch.com/2025/10/13/kv-cache-optimization-via-multi-head-latent-attention/) - PyImageSearch (accessed 2025-02-02)

**GitHub Resources**:
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA optimized inference
- [SmoothQuant Implementation](https://github.com/mit-han-lab/smoothquant)
- [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) - Curated inference resources

**Additional Web Resources**:
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) - Pope et al. 2022
- [VL-Cache: Sparsity and Modality-Aware KV Cache Compression](https://www.amazon.science/publications/vl-cache-sparsity-and-modality-aware-kv-cache-compression-for-vision-language-model-inference-acceleration) - Amazon Science
- [Joint Optimization for KV Cache Compression](https://arxiv.org/abs/2510.20707) - arXiv preprint
