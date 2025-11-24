# DeepGEMM - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepGEMM
**Purpose**: Optimized GEMM kernels for Tensor Core utilization
**Target**: H100/H800 GPUs with FP8 support

## Directory Structure

```
04-DeepGEMM/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE
├── CMakeLists.txt           # Build configuration
├── setup.py                 # Python package setup
│
├── csrc/                    # CUDA source ⭐
│   ├── gemm_kernels.cu         # Core GEMM implementations
│   ├── tensor_core_ops.cu      # Tensor Core primitives
│   └── fp8_utils.cu            # FP8 format utilities
│
├── deep_gemm/               # Python package
│   ├── __init__.py
│   ├── gemm.py                 # High-level API
│   └── utils.py                # Utilities
│
├── tests/                   # Test suite
├── build.sh                 # Build script
├── install.sh               # Installation script
├── develop.sh               # Development setup
└── third-party/             # Dependencies
```

## Key Concepts

### Tensor Core Optimization
- **Warp-level matrix ops**: WMMA/MMA instructions
- **Memory coalescing**: Optimized shared memory layout
- **FP8/FP16/BF16**: Multiple precision support

### Performance Features
- **Persistent kernels**: Reduce launch overhead
- **Split-K**: Better GPU utilization
- **Tile quantization**: Per-tile scaling factors

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `csrc/gemm_kernels.cu` | Core GEMM implementations | CUTLASS, Tensor Cores |
| `csrc/tensor_core_ops.cu` | Low-level Tensor Core ops | WMMA, MMA.sync |
| `deep_gemm/gemm.py` | Python API | torch.autograd |

## Quick Start

```bash
# Build
bash build.sh

# Install
pip install -e .

# Test
python -m pytest tests/
```

## Cross-References

**DeepSeek efficiency**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
**Related**: `02-3FS` (FP8 training), `11-FlashMLA` (attention optimization)
