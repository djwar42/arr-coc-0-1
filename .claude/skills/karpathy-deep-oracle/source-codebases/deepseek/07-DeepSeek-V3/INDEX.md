# DeepSeek-V3 - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepSeek-V3
**Purpose**: Latest flagship model architecture
**Scale**: 671B total params, 37B active

## Directory Structure

```
07-DeepSeek-V3/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── README_WEIGHTS.md        # Model weights info
├── LICENSE-CODE
├── LICENSE-MODEL
│
├── inference/               # Inference implementation ⭐
│   ├── model.py                # V3 model architecture
│   ├── generate.py             # Text generation
│   ├── configs/                # Model configs
│   └── kernels/                # Custom CUDA kernels
│
└── figures/                 # Architecture diagrams
```

## Key Concepts

### Architecture Innovations
- **MLA (Multi-head Latent Attention)**: Memory-efficient attention
- **DeepSeekMoE**: Fine-grained expert routing
- **FP8 training**: 3-stage mixed precision

### Performance
- **Cost efficient**: 89× cheaper than comparable models
- **Strong performance**: GPT-4 level on benchmarks
- **Open weights**: Available on HuggingFace

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `inference/model.py` | V3 model implementation | MLA, MoE, RoPE |
| `inference/generate.py` | Generation pipeline | KV cache, sampling |
| `README_WEIGHTS.md` | Weight download guide | HuggingFace, quantization |

## Quick Start

```bash
# Download weights (from HuggingFace)
# See README_WEIGHTS.md for instructions

# Inference
python inference/generate.py \
    --model_path deepseek-ai/DeepSeek-V3 \
    --prompt "Hello, world!"
```

## Model Features

| Feature | Description |
|---------|-------------|
| MLA | 5× KV cache reduction |
| MoE | 256 experts, 8 active |
| FP8 | 37% faster training |

## Cross-References

**MoE details**: `05-DeepSeek-MoE/INDEX.md`, `karpathy/codebases/05-deepseek-moe-overview.md`
**Attention**: `11-FlashMLA/INDEX.md`
**Training efficiency**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
