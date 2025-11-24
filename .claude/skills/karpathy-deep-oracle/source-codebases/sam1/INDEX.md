# SAM 1 Source Code

**Repository**: https://github.com/facebookresearch/segment-anything
**Release**: April 2023 (Meta AI)
**Size**: 22M (18 Python files)
**Cloned**: November 21, 2025

---

## Overview

Segment Anything Model (SAM) - First promptable segmentation model trained on SA-1B dataset (11M images, 1.1B masks).

### Key Features

- **Zero-shot transfer**: Trained once, segments any object in any image
- **Promptable interface**: Click, box, or mask prompts
- **Three models**: ViT-H (huge), ViT-L (large), ViT-B (base)
- **Real-time inference**: Interactive segmentation in browser

---

## Directory Structure

```
sam1/
├── segment_anything/          # Core library (18 Python files)
│   ├── __init__.py
│   ├── build_sam.py          # Model builder
│   ├── predictor.py          # Interactive predictor
│   ├── automatic_mask_generator.py  # Automatic mode
│   ├── modeling/             # Architecture (6 files)
│   │   ├── sam.py            # Main SAM class
│   │   ├── image_encoder.py  # ViT-H encoder
│   │   ├── prompt_encoder.py # Point/box encoder
│   │   ├── mask_decoder.py   # Segmentation head
│   │   ├── transformer.py    # Attention blocks
│   │   └── common.py         # Shared components
│   └── utils/                # Utilities (3 files)
│       ├── transforms.py     # Image preprocessing
│       ├── amg.py            # Automatic mask generation
│       └── onnx.py           # ONNX export
├── scripts/                  # CLI tools
│   ├── amg.py                # Automatic mask CLI
│   └── export_onnx_model.py  # ONNX conversion
├── notebooks/                # Jupyter examples (3)
│   ├── predictor_example.ipynb
│   ├── automatic_mask_generator_example.ipynb
│   └── onnx_model_example.ipynb
├── demo/                     # React web app
│   └── [React TypeScript frontend]
├── LICENSE                   # Apache 2.0
├── README.md                 # Documentation
├── setup.py                  # Installation
└── requirements.txt          # Dependencies
```

---

## Core Architecture Files

### Main Entry Points

1. **`build_sam.py`** - Model factory
   - `build_sam_vit_h()` - ViT-Huge (632M params)
   - `build_sam_vit_l()` - ViT-Large (308M params)
   - `build_sam_vit_b()` - ViT-Base (91M params)

2. **`predictor.py`** - Interactive segmentation
   - `SamPredictor` class
   - Methods: `set_image()`, `predict()`, `set_torch_image()`

3. **`automatic_mask_generator.py`** - Automatic mode
   - `SamAutomaticMaskGenerator` class
   - Generates all masks without prompts

### Architecture Components

**`modeling/sam.py`** - Main SAM class
- Combines image encoder + prompt encoder + mask decoder
- Handles forward pass for segmentation

**`modeling/image_encoder.py`** - ViT-H Backbone
- Vision Transformer (ViT) architecture
- Input: 1024×1024 RGB image
- Output: 64×64×256 image embeddings

**`modeling/prompt_encoder.py`** - Prompt Processing
- Point prompts (foreground/background clicks)
- Box prompts (bounding boxes)
- Mask prompts (coarse masks)
- Output: 256-dim sparse embeddings + 256-dim dense embeddings

**`modeling/mask_decoder.py`** - Segmentation Head
- Lightweight transformer decoder
- Predicts multiple masks (3 per prompt)
- Confidence scores for ambiguity handling

**`modeling/transformer.py`** - Attention Mechanisms
- Two-way transformer blocks
- Query-key-value attention
- Used in mask decoder

**`modeling/common.py`** - Shared Utilities
- MLPBlock, LayerNorm2d
- Positional encodings
- Activation functions

---

## Usage Patterns

### Pattern 1: Interactive Segmentation

```python
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)

# Predict with point prompt
masks, scores, logits = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1]  # 1=foreground, 0=background
)
```

### Pattern 2: Automatic Mask Generation

```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate all masks
masks = mask_generator.generate(image)  # Returns list of dicts
```

### Pattern 3: ONNX Export for Browser

```python
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
onnx_model = SamOnnxModel(sam, return_single_mask=True)

# Export decoder (encoder pre-computed in browser)
torch.onnx.export(onnx_model, ..., "sam_decoder.onnx")
```

---

## Model Checkpoints

Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints

| Model | Params | Size | Download |
|-------|--------|------|----------|
| ViT-H | 632M   | 2.4GB | `sam_vit_h_4b8939.pth` |
| ViT-L | 308M   | 1.2GB | `sam_vit_l_0b3195.pth` |
| ViT-B | 91M    | 358MB | `sam_vit_b_01ec64.pth` |

---

## SA-1B Dataset

**Largest segmentation dataset ever created**:
- 11 million images
- 1.1 billion masks
- Licensed images from providers

**Format**: Images + JSON annotations (RLE compressed masks)

**Download**: https://ai.meta.com/datasets/segment-anything/

---

## Installation

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Requirements**:
- Python ≥ 3.8
- PyTorch ≥ 1.7
- torchvision ≥ 0.8

---

## Citation

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## Cross-References

- **SAM 2**: `../sam2/` - Video + image segmentation
- **SAM 3**: `../sam3/` - Text + visual prompts
- **Research**: `.claude/skills/karpathy-deep-oracle/sam-general/` - SAM concepts

---

**Last Updated**: November 21, 2025
**Source**: GitHub clone (--depth 1, main branch)
**Files**: 18 Python files (core library only)
