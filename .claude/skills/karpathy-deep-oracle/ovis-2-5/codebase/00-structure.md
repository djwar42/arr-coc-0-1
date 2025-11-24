# Codebase Structure

**Category**: Codebase
**Code**: `RESEARCH/Ovis25/Ovis/`

## Repository Organization

```
ovis/
├── model/                      # Model implementation
│   ├── modeling_ovis.py           # Main model classes
│   ├── configuration_ovis.py      # Config
│   ├── conversation_formatter.py  # Chat templates
│   └── vit/
│       ├── modeling_siglip2_navit.py  # Vision transformer
│       └── configuration_siglip2_navit.py
│
├── train/                      # Training code
│   ├── train.py                   # Main training script
│   ├── arguments.py               # Training arguments
│   ├── callback.py                # Training callbacks
│   └── dataset/
│       ├── caption_dataset.py     # P1 dataset
│       ├── conversation_dataset.py # P2+ dataset
│       └── multimodal_dataset.py  # Data collator
│
└── serve/                      # Inference scripts
    ├── infer_basic_demo.py        # Basic examples
    ├── infer_think_demo.py        # Thinking mode
    └── web_ui.py                  # Gradio interface
```

## Key Files

**Model**: `modeling_ovis.py` - VisualEmbedding, VisualTokenizer, Ovis
**ViT**: `modeling_siglip2_navit.py` - SigLIP 2 NaViT
**Training**: `train.py` - Main training loop
**Inference**: `infer_basic_demo.py` - Examples
