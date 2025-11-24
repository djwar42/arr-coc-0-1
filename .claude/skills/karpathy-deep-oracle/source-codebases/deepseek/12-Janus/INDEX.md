# Janus - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/Janus
**Purpose**: Unified multimodal understanding and generation
**Paper**: janus_pro_tech_report.pdf included

## Directory Structure

```
12-Janus/
├── INDEX.md                 # This file
├── README.md                # Original documentation (744 lines)
├── janus_pro_tech_report.pdf # Technical paper ⭐
├── LICENSE-CODE
├── LICENSE-MODEL
├── Makefile
├── requirements.txt
├── pyproject.toml
│
├── janus/                   # Main package ⭐
│   ├── models/                 # Model implementations
│   │   ├── modeling_janus.py      # Main Janus model
│   │   ├── vision_encoder.py      # Vision backbone
│   │   ├── image_decoder.py       # Image generation
│   │   └── projector.py           # Modality bridge
│   ├── utils/                  # Utilities
│   └── serve/                  # Serving
│
├── inference.py             # Understanding inference
├── generation_inference.py  # Image generation inference
├── interactivechat.py       # Interactive demo
│
├── demo/                    # Demo scripts
└── images/                  # Demo images
```

## Key Concepts

### Unified Multimodal
- **Understanding**: Image → Text (VQA, captioning)
- **Generation**: Text → Image (diffusion-based)
- **Single model**: Both directions in one

### Architecture
- **Decoupled encoders**: Separate vision paths
- **Shared LLM**: Common language backbone
- **Conditional generation**: LLM-guided diffusion

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `janus/models/modeling_janus.py` | Main model | unified multimodal |
| `janus/models/image_decoder.py` | Generation decoder | diffusion, VAE |
| `inference.py` | Understanding inference | VQA, captioning |
| `generation_inference.py` | Image generation | text-to-image |
| `interactivechat.py` | Interactive demo | Gradio |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Understanding (image → text)
python inference.py \
    --model deepseek-ai/Janus-Pro-7B \
    --image test.png

# Generation (text → image)
python generation_inference.py \
    --model deepseek-ai/Janus-Pro-7B \
    --prompt "A beautiful sunset"

# Interactive demo
python interactivechat.py
```

## Model Variants

| Model | Size | Understanding | Generation |
|-------|------|---------------|------------|
| Janus-1.3B | 1.3B | ✓ | ✓ |
| Janus-Pro-7B | 7B | ✓ | ✓ |

## Cross-References

**Related VLMs**: `08-DeepSeek-VL2`, `14-Ovis-2-5`
**MoE backbone**: `05-DeepSeek-MoE/INDEX.md`
