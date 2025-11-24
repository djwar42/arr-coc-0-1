# 3FS (FP8-LM) - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepSeek-FP8
**Purpose**: 3-stage FP8 training system for LLMs
**Key Results**: 37% faster training, 39% less memory

## Directory Structure

```
02-3FS/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE
├── CMakeLists.txt           # Build configuration
├── Cargo.toml               # Rust dependencies
│
├── fp8_gemm/                # FP8 GEMM kernels ⭐
│   ├── csrc/                   # CUDA source
│   └── python/                 # Python bindings
│
├── fp8_trainer/             # Training integration
│   ├── megatron/               # Megatron-LM patches
│   └── examples/               # Usage examples
│
├── tests/                   # Test suite
└── third-party/             # Dependencies
```

## Key Concepts

### 3-Stage FP8 Training
1. **Stage 1**: FP8 forward pass (E4M3 format)
2. **Stage 2**: FP8 backward pass (E5M2 format)
3. **Stage 3**: FP8 optimizer states

### Performance Gains
- **37% faster** than BF16 training
- **39% less memory** usage
- **H100 optimized** with native FP8 support

## Key Files

| Component | Description | Keywords |
|-----------|-------------|----------|
| `fp8_gemm/` | Custom FP8 GEMM kernels | Tensor Cores, cuBLAS |
| `fp8_trainer/` | Training loop integration | Megatron, gradient scaling |
| `tests/` | Correctness verification | numerical accuracy |

## Quick Start

```bash
# Build FP8 kernels
mkdir build && cd build
cmake .. && make -j

# Run training
python fp8_trainer/examples/train_gpt.py
```

## Cross-References

**DeepSeek efficiency analysis**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
**Related**: `04-DeepGEMM` (GEMM optimizations), `11-FlashMLA` (memory efficiency)
