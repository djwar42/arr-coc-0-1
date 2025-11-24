# Ovis 2.5 - Source Codebase Index

**Type**: Complete source code + documentation
**Size**: ~22MB (includes PDFs, images)
**Repository**: https://github.com/AIDC-AI/Ovis

## Overview

Ovis 2.5 is a multimodal vision-language model featuring:
- **Native resolution processing** (no fixed image size)
- **Visual Embedding Table (VET)** - probabilistic token generation
- **5-phase training pipeline** (VET → Multimodal → Instruction → RL)
- **Thinking mode** for complex reasoning

## Directory Structure

```
14-Ovis-2-5/
├── INDEX.md                    # This file
├── README.md                   # Original repo README
├── LICENSE                     # Apache 2.0
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
│
├── ovis/                       # Main package
│   ├── __init__.py
│   ├── model/                  # Core model implementation
│   │   ├── modeling_ovis.py       # Main Ovis model (1,200+ lines)
│   │   ├── configuration_ovis.py  # Model configuration
│   │   ├── conversation_formatter.py  # Chat templates
│   │   └── vit/                   # Vision transformer
│   │       ├── modeling_siglip2_navit.py    # SigLIP 2 NaViT
│   │       └── configuration_siglip2_navit.py
│   │
│   ├── train/                  # Training infrastructure
│   │   ├── train.py               # Main training script
│   │   ├── arguments.py           # Training arguments
│   │   ├── callback.py            # Training callbacks
│   │   └── dataset/               # Dataset loaders
│   │       ├── caption_dataset.py
│   │       ├── conversation_dataset.py
│   │       ├── multimodal_dataset.py
│   │       └── ovis2_5_sft_datainfo.json
│   │
│   ├── serve/                  # Inference & demos
│   │   ├── infer_basic_demo.py    # Basic inference
│   │   ├── infer_think_demo.py    # Thinking mode demo
│   │   └── web_ui.py              # Gradio web UI
│   │
│   └── util/                   # Utilities
│       ├── constants.py
│       └── utils.py
│
├── docs/                       # Documentation & assets
│   ├── Ovis25_arch.png            # Architecture diagram
│   ├── Ovis2_5_Tech_Report.pdf    # Technical report
│   ├── ovis_logo.png
│   ├── license/                   # License files
│   └── performance/               # Benchmark images
│
├── scripts/                    # Training scripts
│   ├── run_ovis2_5_sft.sh         # SFT training script
│   └── zero_configs/              # DeepSpeed configs
│       ├── zero0_cp.json
│       ├── zero1_cp.json
│       ├── zero2_cp.json
│       └── zero3_cp.json
│
└── plugin/
    └── mdp3.py                    # Multimodal data processor
```

## Key Files

### Core Model (`ovis/model/`)

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `modeling_ovis.py` | ~1,200 | Main model: Ovis, VisualTokenizer, VisualEmbedding | VET, forward pass, generation |
| `configuration_ovis.py` | ~200 | Model hyperparameters | config, hidden_size, num_layers |
| `conversation_formatter.py` | ~300 | Chat template formatting | Qwen, Llama, conversation |

### Vision Transformer (`ovis/model/vit/`)

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `modeling_siglip2_navit.py` | ~800 | SigLIP 2 NaViT vision encoder | NaViT, native resolution, patches |
| `configuration_siglip2_navit.py` | ~150 | Vision model config | patch_size, image_size |

### Training (`ovis/train/`)

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `train.py` | ~400 | Main training loop | HuggingFace Trainer, DDP |
| `arguments.py` | ~200 | Training arguments | lr, batch_size, epochs |
| `callback.py` | ~100 | Training callbacks | logging, checkpointing |

### Datasets (`ovis/train/dataset/`)

| File | Description | Keywords |
|------|-------------|----------|
| `caption_dataset.py` | Image captioning data | COCO, Flickr |
| `conversation_dataset.py` | Multi-turn conversations | dialogue, chat |
| `multimodal_dataset.py` | General multimodal data | mixed modality |

### Inference (`ovis/serve/`)

| File | Description | Keywords |
|------|-------------|----------|
| `infer_basic_demo.py` | Basic image Q&A | inference, generation |
| `infer_think_demo.py` | Thinking mode reasoning | chain-of-thought |
| `web_ui.py` | Gradio interface | demo, UI |

## Key Concepts

### Visual Embedding Table (VET)
- Replaces traditional projection layer
- Generates probabilistic visual tokens
- Aligns vision and language embedding spaces
- See: `modeling_ovis.py` → `VisualEmbedding` class

### Native Resolution Processing
- No fixed image size constraint
- Dynamic patch extraction
- Preserves aspect ratio
- See: `modeling_siglip2_navit.py` → `forward()` method

### Thinking Mode
- Chain-of-thought reasoning
- `<think>...</think>` tags for reasoning
- Explicit reflection before answering
- See: `infer_think_demo.py`

## Quick Start

### Inference
```python
from ovis.model.modeling_ovis import Ovis

model = Ovis.from_pretrained("AIDC-AI/Ovis2.5-Qwen3-32B")
response = model.chat(image, "What is in this image?")
```

### Training
```bash
bash scripts/run_ovis2_5_sft.sh
```

## Related Knowledge

See `karpathy-deep-oracle/ovis-2-5/` for detailed documentation:
- `architecture/` - Model architecture deep dives
- `concepts/` - Key concepts (VET, structural alignment)
- `training/` - 5-phase training pipeline
- `examples/` - Code examples
- `usage/` - Quickstart and fine-tuning guides

## Cross-References

- **DeepSeek comparison**: Similar probabilistic token approach in DeepSeek-OCR
- **Qwen integration**: Uses Qwen3 as LLM backbone
- **NaViT**: Native resolution ViT from Google
