# FlashMLA: Multi-head Latent Attention Kernel Library
## Architectural Overview

**Created**: 2025-10-28
**Status**: Phase 1 Complete

---

## Executive Summary

FlashMLA is DeepSeek's library of optimized CUDA attention kernels powering the DeepSeek-V3 and DeepSeek-V3.2 models. It implements Multi-head Latent Attention (MLA) with both sparse and dense patterns, achieving 410-1450 TFlops on modern NVIDIA GPUs (H800 Hopper, B200 Blackwell).

**Key Innovation**: Token-level sparse attention with FP8 KV cache quantization for memory-efficient inference.

---

## Architecture Overview

### Repository Structure

```
FlashMLA/
├── flash_mla/                  # Python interface
│   ├── __init__.py            # Public API exports
│   └── flash_mla_interface.py # Main Python bindings
│
├── csrc/                       # CUDA kernel implementations
│   ├── pybind.cpp             # PyBind11 bindings
│   ├── params.h               # Shared parameter structures
│   ├── utils.h                # Utility functions
│   │
│   ├── smxx/                  # Architecture-agnostic code
│   │   ├── get_mla_metadata.cu  # Tile scheduler metadata generation
│   │   └── mla_combine.cu       # Split-KV result combination
│   │
│   ├── sm90/                  # Hopper (H800) kernels
│   │   ├── decode/
│   │   │   ├── dense/         # Dense decoding kernel
│   │   │   └── sparse_fp8/    # FP8 sparse decoding kernel
│   │   └── prefill/
│   │       └── sparse/        # Sparse prefill kernel
│   │
│   ├── sm100/                 # Blackwell (B200) kernels
│   │   ├── decode/
│   │   │   └── sparse_fp8/    # FP8 sparse decoding kernel
│   │   └── prefill/
│   │       ├── dense/         # Dense MHA forward/backward
│   │       └── sparse/        # Sparse prefill kernel
│   │
│   └── cutlass/               # NVIDIA CUTLASS library (submodule)
│
├── tests/                      # Test and benchmark suites
│   ├── test_flash_mla_decoding.py  # Decoding kernel tests
│   ├── test_flash_mla_prefill.py   # Prefill kernel tests
│   ├── test_fmha_sm100.py          # SM100 MHA tests
│   ├── quant.py                    # FP8 quantization utilities
│   └── lib.py                      # Test utilities
│
├── benchmark/                  # Performance benchmarking
│   ├── bench_flash_mla.py     # Benchmark runner
│   └── visualize.py           # Results visualization
│
├── docs/                       # Technical documentation
│   ├── 20250422-new-kernel-deep-dive.md
│   └── 20250929-hopper-fp8-sparse-deep-dive.md
│
└── setup.py                    # Build configuration
```

---

## Core Components

### 1. Python Interface (`flash_mla/`)

**Purpose**: High-level API for PyTorch integration

**Key Functions**:
- `get_mla_metadata()`: Generates tile scheduler metadata for decoding
- `flash_mla_with_kvcache()`: Main decoding kernel (sparse/dense)
- `flash_mla_sparse_fwd()`: Sparse prefill kernel
- `flash_attn_varlen_*()`: Dense MHA prefill/training kernels

**Design**: Thin wrapper around CUDA kernels via PyBind11 extension

---

### 2. CUDA Kernels (`csrc/`)

#### 2.1 Architecture-Specific Kernels

**SM90 (Hopper H800)**:
- Dense decoding: BF16 KV cache, MQA mode
- Sparse FP8 decoding: 410 TFlops, FP8 KV cache
- Sparse prefill: 640 TFlops

**SM100 (Blackwell B200)**:
- Sparse FP8 decoding: 350 TFlops (not fully optimized yet)
- Dense MHA prefill: 1460 TFlops fwd, 1000 TFlops bwd
- Sparse prefill: 1450 TFlops

#### 2.2 Shared Components (`smxx/`)

- **Tile Scheduler**: Distributes work across SMs for split-KV parallelism
- **Combine Kernel**: Merges partial results from split-KV computation

---

### 3. MLA Modes

**MQA (Multi-Query Attention)**:
- `head_dim_k = 576`, `head_dim_v = 512`
- Used for decoding and sparse prefill
- Single KV head, multiple query heads

**MHA (Multi-Head Attention)**:
- `head_dim_k = 192/128`, `head_dim_v = 128`
- Used for dense prefill/training
- Multiple KV heads, standard transformer attention

---

## Key Technical Features

### 1. FP8 KV Cache Quantization

**Format** (656 bytes/token):
- **512 bytes**: Quantized NoPE (512 × fp8_e4m3)
- **16 bytes**: Scale factors (4 × float32, one per 128 elements)
- **128 bytes**: RoPE embeddings (64 × bfloat16, unquantized for accuracy)

**Benefits**:
- 2× memory reduction vs BF16
- Maintains accuracy via per-group scaling
- RoPE preserved in full precision

---

### 2. Token-Level Sparse Attention

**Purpose**: DeepSeek Sparse Attention (DSA) from DeepSeek-V3.2

**Mechanism**:
- Attention computed only for top-k tokens per query
- `indices` tensor specifies which KV tokens to attend
- Invalid indices marked as -1

**Performance**:
- Sparse FP8 decoding: 410 TFlops (H800)
- Sparse prefill: 640 TFlops (H800), 1450 TFlops (B200)

---

### 3. Paged Attention / Blocked KV Cache

**Layout**:
- KV cache divided into fixed-size blocks (default 64 tokens)
- `block_table` maps logical positions to physical blocks
- Enables non-contiguous memory allocation

**Benefits**:
- Memory efficiency for variable-length sequences
- Reduces fragmentation in batch inference
- Compatible with vLLM-style serving

---

### 4. Split-KV Parallelism

**Strategy**:
- Divides KV sequence across multiple thread blocks
- Each block computes partial attention results
- Combine kernel merges results using log-sum-exp trick

**Benefits**:
- Scales to very long sequences (1M+ tokens)
- Better SM utilization on large GPUs
- Tile scheduler optimizes work distribution

---

## Performance Characteristics

### H800 SXM5 (SM90)

| Kernel | Mode | Throughput | Configuration |
|--------|------|------------|---------------|
| Dense Decoding | Memory-bound | 3000 GB/s | Large batch/seqlen |
| Dense Decoding | Compute-bound | 660 TFlops | High head count |
| Sparse FP8 Decoding | Compute-bound | 410 TFlops | topk=64-256 |
| Sparse Prefill | Compute-bound | 640 TFlops | MQA mode |

### B200 (SM100)

| Kernel | Mode | Throughput | Configuration |
|--------|------|------------|---------------|
| Dense MHA Prefill Fwd | Compute-bound | 1460 TFlops | MHA mode |
| Dense MHA Prefill Bwd | Compute-bound | 1000 TFlops | MHA mode |
| Sparse Prefill | Compute-bound | 1450 TFlops | MQA mode |
| Sparse FP8 Decoding | Compute-bound | 350 TFlops | Not optimized |

---

## Code Flow Examples

### Decoding Workflow (Inference)

```python
# 1. Generate tile scheduler metadata (once per batch)
tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,  # num_q_tokens_per_head_k
    h_kv,
    h_q,
    is_fp8_kvcache=True,
    topk=128  # Enable sparse attention
)

# 2. Loop over layers
for layer_idx in range(num_layers):
    q_i = model.get_query(layer_idx)  # [batch, s_q, h_q, d]
    kvcache_i = model.get_kvcache(layer_idx)  # [num_blocks, block_size, h_kv, d]

    # 3. Run attention kernel
    o_i, lse_i = flash_mla.flash_mla_with_kvcache(
        q_i,
        kvcache_i,
        block_table,
        cache_seqlens,
        dv=512,
        tile_scheduler_metadata,
        num_splits,
        is_fp8_kvcache=True,
        indices=sparse_indices  # [batch, s_q, topk]
    )

    # 4. Continue model forward pass
    hidden_i = model.mlp(o_i)
```

### Sparse Prefill Workflow

```python
# Single-call operation (no batch dimension)
out, max_logits, lse = flash_mla.flash_mla_sparse_fwd(
    q,        # [s_q, h_q, d_qk]
    kv,       # [s_kv, h_kv, d_qk]
    indices,  # [s_q, h_kv, topk]
    sm_scale=1.0 / math.sqrt(d_qk),
    d_v=512
)
```

---

## Build System

**Compiler**: NVCC 12.8+ (12.9+ for SM100)

**Dependencies**:
- PyTorch 2.0+
- CUDA 12.8+
- CUTLASS (git submodule)

**Architecture Selection**:
- `FLASH_MLA_DISABLE_SM90=1`: Disable Hopper kernels
- `FLASH_MLA_DISABLE_SM100=1`: Disable Blackwell kernels
- `NVCC_THREADS=32`: Parallel compilation threads

**Optimization Flags**:
- `-O3`: Aggressive optimization
- `--use_fast_math`: Fast math approximations
- `--expt-relaxed-constexpr`: Relaxed compile-time evaluation
- `--register-usage-level=10`: Register usage reporting

---

## Key Design Choices

### 1. Why Separate SM90/SM100 Kernels?

- **SM100 New Features**: Enhanced tensor cores, larger shared memory, new instructions
- **Performance**: Architecture-specific tuning for optimal throughput
- **Flexibility**: Disable unsupported architectures at build time

### 2. Why FP8 for Sparse Attention?

- **Memory**: 2× reduction critical for long-context inference
- **Speed**: FP8 tensor cores on Hopper/Blackwell
- **Accuracy**: Per-group scaling maintains precision for RoPE-free components

### 3. Why Paged/Blocked KV Cache?

- **vLLM Compatibility**: Industry-standard serving framework
- **Memory Efficiency**: Reduces fragmentation, enables dynamic batching
- **Flexibility**: Non-contiguous allocation for variable-length sequences

### 4. Why Split-KV Parallelism?

- **Long Sequences**: Scales to 1M+ tokens per sequence
- **GPU Utilization**: Keeps all SMs busy even with small batch sizes
- **Numerical Stability**: Log-sum-exp trick for combining partial softmax

---

## Testing Strategy

### Correctness Tests

1. **Reference Validation**: Compare against PyTorch reference implementation
2. **Edge Cases**: Zero-length sequences, all-invalid indices, causal masking
3. **Quantization**: FP8 quant/dequant validation
4. **Tolerance**: Configurable atol/rtol for numerical precision

### Performance Benchmarks

1. **Memory-Bound**: Large batch/seqlen, measure GB/s
2. **Compute-Bound**: High head count, measure TFlops
3. **Latency**: Token generation speed (tokens/sec)
4. **Comparison**: vs FlashAttention, vs PyTorch SDPA

---

## Future Directions

### Potential Optimizations

1. **SM100 Sparse Decoding**: Further tuning (currently 350 TFlops, target 500+)
2. **GQA Support**: Grouped-Query Attention for backward pass
3. **INT4 Quantization**: 4× memory reduction for ultra-long contexts
4. **Async Copy**: Overlap compute and memory transfers

### Integration Opportunities

1. **vLLM**: Native integration for production serving
2. **DeepSpeed**: Distributed inference support
3. **TensorRT-LLM**: Export to NVIDIA's inference engine
4. **ROCm/AMD**: Port to AMD GPUs (community versions exist)

---

## References

### Papers
- [DeepSeek-V3 Paper](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V3.2 Paper with DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

### Related Projects
- [FlashAttention](https://github.com/dao-AILab/flash-attention)
- [CUTLASS](https://github.com/nvidia/cutlass)

### Technical Deep-Dives
- [New Kernel Deep-Dive (2025-04-22)](docs/20250422-new-kernel-deep-dive.md)
- [Hopper FP8 Sparse Deep-Dive (2025-09-29)](docs/20250929-hopper-fp8-sparse-deep-dive.md)

---

## Summary

FlashMLA is a highly optimized attention kernel library that enables DeepSeek's Multi-head Latent Attention with:

✅ **Performance**: 410-1450 TFlops across Hopper/Blackwell
✅ **Memory Efficiency**: FP8 KV cache with 2× reduction
✅ **Sparsity**: Token-level sparse attention for long contexts
✅ **Flexibility**: Dense/sparse, prefill/decoding, MQA/MHA modes
✅ **Production-Ready**: Paged attention, split-KV parallelism, robust testing

The codebase demonstrates state-of-the-art CUDA kernel engineering for modern transformer inference.
