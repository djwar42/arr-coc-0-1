# Qwen3-VL - Source Codebase Index

**Repository**: https://github.com/QwenLM/Qwen2.5-VL (Alibaba)
**Purpose**: Vision-language model with dynamic resolution
**Note**: Included for architecture comparison

## Directory Structure

```
13-Qwen3-VL/
├── INDEX.md                 # This file
├── README.md                # Original documentation (1205 lines)
├── LICENSE
├── requirements_web_demo.txt
│
├── qwen-vl-finetune/        # Fine-tuning toolkit ⭐
│   ├── finetune.py             # Main training script
│   ├── data_utils.py           # Data processing
│   └── configs/                # Training configs
│
├── qwen-vl-utils/           # Utility package
│   ├── src/                    # Source code
│   └── setup.py                # Package setup
│
├── evaluation/              # Evaluation scripts
│   ├── eval_*.py               # Benchmark evaluations
│   └── benchmarks/             # Benchmark data
│
├── cookbooks/               # Usage examples
│   └── *.ipynb                 # Jupyter notebooks
│
├── docker/                  # Docker setup
├── web_demo_mm.py           # Multimodal demo
└── requirements_web_demo.txt
```

## Key Concepts

### Architecture Innovations
- **M-RoPE**: Multi-axis rotary position encoding
- **DeepStack**: Multi-layer vision injection
- **Dynamic resolution**: Native aspect ratio support

### Features
- **Video understanding**: Temporal encoding
- **Document OCR**: High-resolution support
- **Multilingual**: Multiple language support

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `qwen-vl-finetune/finetune.py` | Training script | LoRA, full fine-tune |
| `qwen-vl-utils/` | Utility package | preprocessing, tokenization |
| `evaluation/` | Benchmark scripts | VQA, OCR benchmarks |
| `web_demo_mm.py` | Interactive demo | Gradio |

## Quick Start

```bash
# Install utils
pip install qwen-vl-utils

# Fine-tune
python qwen-vl-finetune/finetune.py \
    --model Qwen/Qwen2.5-VL-7B \
    --data your_data.json

# Web demo
pip install -r requirements_web_demo.txt
python web_demo_mm.py
```

## Model Variants

| Model | Vision | Language | Total |
|-------|--------|----------|-------|
| 2B | 675M | 1.5B | 2.2B |
| 7B | 675M | 7B | 7.7B |
| 72B | 675M | 72B | 72.7B |

## Cross-References

**Dedicated oracle**: `qwen3vl-oracle` (detailed analysis)
**Related VLMs**: `08-DeepSeek-VL2`, `14-Ovis-2-5`
**Position encoding**: `karpathy/vision-language/02-rope-multiaxis-encoding.md`
