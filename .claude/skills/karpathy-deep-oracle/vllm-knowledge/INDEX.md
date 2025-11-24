# vLLM Inference - Index

**vLLM: High-throughput LLM inference engine**

**Total Files**: 15 knowledge files
**Coverage**: PagedAttention, continuous batching, scheduling, CUDA kernels, deployment
**Focus**: Production LLM/VLM serving at scale

---

## Overview

vLLM is a fast and memory-efficient inference engine for LLMs and VLMs:
- **PagedAttention**: Virtual memory-inspired KV cache management
- **Continuous batching**: Dynamic request scheduling
- **High throughput**: 2-24× faster than baseline
- **CUDA optimized**: Custom kernels for Tensor Cores

---

## Topics Covered

### PagedAttention
- Virtual memory for KV cache
- Block-level memory management
- Memory fragmentation reduction
- Copy-on-write optimization

### Continuous Batching
- Dynamic request scheduling
- Iteration-level batching
- Preemption and swapping
- Priority queuing

### CUDA Kernels
- Attention kernel optimization
- Tensor Core utilization
- Flash Attention integration
- Memory-efficient operations

### Scheduling & Throughput
- Request scheduling algorithms
- Batch size optimization
- Memory capacity management
- Latency-throughput tradeoffs

### Deployment
- Multi-GPU deployment
- Tensor parallelism
- Pipeline parallelism
- Distributed serving

### VLM Support
- Vision-language model inference
- Multi-modal token processing
- Image-text attention
- Memory optimization for large vision encoders

---

## Cross-References

**Related Folders**:
- `cuda/` - CUDA fundamentals
- `karpathy/llm-gpu-integration/` - FlashAttention, GPU architecture
- `karpathy/inference-optimization/` - Quantization, KV cache
- `karpathy/distributed-training/` - Multi-GPU patterns

**GCP Deployment**:
- `gcp-vertex/` - Vertex AI serving
- `gcp-gpu/` - GPU selection (A100, H100, L4)

---

## Quick Navigation

**Getting started:**
→ Read PagedAttention fundamentals first

**Production deployment:**
→ Check scheduling and throughput optimization files

**Multi-GPU:**
→ See tensor parallelism and distributed serving files

**VLM inference:**
→ Focus on vision-language specific optimization files

---

## Performance

**Typical speedups with vLLM:**
- 2-4× throughput vs PyTorch (simple batching)
- 10-24× throughput vs naive implementation
- 55% memory savings via PagedAttention
- Near-zero waste with continuous batching

---

**Last Updated**: 2025-11-21
