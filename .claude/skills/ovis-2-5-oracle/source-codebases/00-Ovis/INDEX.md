# Ovis 2.5 Codebase Reference

**Type**: Reference-based documentation (no local copy)
**Actual Location**: `RESEARCH/Ovis25/Ovis/`

## Overview

This oracle provides **reference-based documentation** for the Ovis 2.5 codebase rather than preserving a complete local copy. All code references point to the actual repository location.

## Why Reference-Based?

- **Active Development**: Codebase is under active development
- **Space Efficiency**: No need for duplicate copies
- **Live References**: Documentation always points to current code
- **Flexibility**: Can reference multiple versions if needed

## Codebase Location

```
RESEARCH/Ovis25/Ovis/
├── model/                      # Model implementation
│   ├── modeling_ovis.py           # Main Ovis model
│   ├── configuration_ovis.py      # Configuration
│   ├── conversation_formatter.py  # Chat templates
│   └── vit/                       # Vision transformer
│       ├── modeling_siglip2_navit.py
│       └── configuration_siglip2_navit.py
├── train/                      # Training infrastructure
│   ├── train.py                   # Main training script
│   ├── arguments.py               # Training arguments
│   ├── callback.py                # Callbacks
│   └── dataset/                   # Dataset loaders
│       ├── caption_dataset.py
│       ├── conversation_dataset.py
│       └── multimodal_dataset.py
└── serve/                      # Inference & demos
    ├── infer_basic_demo.py        # Basic inference
    ├── infer_think_demo.py        # Thinking mode
    └── web_ui.py                  # Gradio UI
```

## Key Files

### Core Model
- **`modeling_ovis.py`** - Main model classes (Ovis, VisualTokenizer, VisualEmbedding)
- **`configuration_ovis.py`** - Model configuration and hyperparameters

### Vision Processing
- **`modeling_siglip2_navit.py`** - SigLIP 2 NaViT vision transformer
- **`configuration_siglip2_navit.py`** - Vision model configuration

### Training
- **`train.py`** - Main training loop with DeepSpeed integration
- **`arguments.py`** - Training argument dataclasses
- **`dataset/`** - P1 (captions), P2+ (conversations) dataloaders

### Inference
- **`infer_basic_demo.py`** - Basic inference examples
- **`infer_think_demo.py`** - Thinking mode demonstrations
- **`web_ui.py`** - Gradio web interface

## Documentation Structure

Complete code documentation is available in the oracle's `codebase/` folder:

- **[00-structure.md](../../codebase/00-structure.md)** - Repository organization
- **[01-modeling-ovis.md](../../codebase/01-modeling-ovis.md)** - Main model file analysis
- **[02-visual-tokenizer-impl.md](../../codebase/02-visual-tokenizer-impl.md)** - Visual tokenizer implementation
- **[03-conversation-formatter.md](../../codebase/03-conversation-formatter.md)** - Chat template system
- **[04-training-loop.md](../../codebase/04-training-loop.md)** - Training infrastructure
- **[05-datasets.md](../../codebase/05-datasets.md)** - Dataset implementations

## Architecture Documentation

For architectural understanding, see:

- **[architecture/00-overview.md](../../architecture/00-overview.md)** - System design
- **[architecture/03-visual-embedding-table.md](../../architecture/03-visual-embedding-table.md)** - VET innovation
- **[architecture/02-visual-tokenizer.md](../../architecture/02-visual-tokenizer.md)** - Tokenization design

## Usage

This oracle references the live codebase. All line numbers and file paths in documentation point to:

```
RESEARCH/Ovis25/Ovis/
```

To explore the actual code, navigate to that directory.

## Oracle Documentation vs Live Code

**Oracle provides**:
- High-level architecture understanding
- Component relationships and data flow
- Key implementation insights with line references
- Training pipeline documentation

**Live codebase contains**:
- Full implementation details
- Complete function signatures
- Unit tests and examples
- Configuration files and scripts

---

**Oracle Type**: Reference-based documentation
**Last Updated**: 2025-10-28
**Codebase Status**: External reference (not copied locally)
