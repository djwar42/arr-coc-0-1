# DeepSeek-VL2 - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepSeek-VL2
**Purpose**: Vision-language v2 with dynamic tiling
**Paper**: DeepSeek_VL2_paper.pdf included

## Directory Structure

```
08-DeepSeek-VL2/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── DeepSeek_VL2_paper.pdf   # Technical paper ⭐
├── LICENSE-CODE
├── LICENSE-MODEL
├── Makefile
├── requirements.txt
├── pyproject.toml
│
├── deepseek_vl2/            # Main package ⭐
│   ├── models/                 # Model implementations
│   │   ├── modeling_deepseek_vl_v2.py
│   │   ├── vision_encoder.py
│   │   └── projector.py
│   ├── serve/                  # Serving utilities
│   └── utils/                  # Utilities
│
├── inference.py             # Basic inference
├── web_demo.py              # Gradio demo
└── images/                  # Demo images
```

## Key Concepts

### Architecture
- **Dynamic tiling**: Adaptive image resolution
- **MoE decoder**: Efficient language modeling
- **Vision projector**: Bridge vision-language

### Performance
- **Multi-resolution**: Native aspect ratio support
- **Strong OCR**: Document understanding
- **Efficient**: MoE for compute efficiency

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `deepseek_vl2/models/modeling_deepseek_vl_v2.py` | Main VL model | multimodal, fusion |
| `deepseek_vl2/models/vision_encoder.py` | Vision backbone | ViT, SigLIP |
| `inference.py` | Basic inference | generation |
| `web_demo.py` | Interactive demo | Gradio |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Inference
python inference.py \
    --model deepseek-ai/deepseek-vl2-small \
    --image test.png \
    --prompt "Describe this image"

# Web demo
python web_demo.py
```

## Model Variants

| Model | Vision | Language | Total |
|-------|--------|----------|-------|
| Small | 400M | 1.3B | 1.7B |
| Base | 400M | 7B | 7.4B |
| Large | 400M | 27B | 27.4B |

## Cross-References

**Related VLMs**: `06-DeepSeek-OCR`, `14-Ovis-2-5`
**MoE architecture**: `05-DeepSeek-MoE/INDEX.md`
**Efficiency analysis**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
