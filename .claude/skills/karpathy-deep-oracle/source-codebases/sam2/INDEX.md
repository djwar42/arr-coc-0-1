# SAM 2 Source Code

**Repository**: https://github.com/facebookresearch/sam2
**Release**: July 2024 (SAM 2), October 2024 (SAM 2.1)
**Size**: 63M (63 Python files)
**Cloned**: November 21, 2025

---

## Overview

Segment Anything Model 2 (SAM 2) - Unified model for promptable segmentation in **images and videos**. Adds temporal consistency with streaming memory attention.

### Key Features

- **Unified architecture**: Single model for images + videos
- **Streaming memory**: Efficient cross-frame attention
- **Hiera encoder**: 6× faster than SAM 1's ViT-H
- **Real-time video**: 44 FPS on A100 GPU
- **SAM 2.1**: +2.8 J&F improvement (Oct 2024)

---

## Directory Structure

```
sam2/
├── sam2/                     # Core library (63 Python files)
│   ├── __init__.py
│   ├── build_sam.py         # Model builder
│   ├── sam2_image_predictor.py    # Image API (SAM 1-like)
│   ├── sam2_video_predictor.py    # Video API (streaming)
│   ├── modeling/            # Architecture (35+ files)
│   │   ├── sam2_base.py     # Base SAM 2 model
│   │   ├── backbones/       # Hiera encoder
│   │   │   ├── hieradet.py
│   │   │   ├── image_encoder.py
│   │   │   └── utils.py
│   │   ├── memory_attention.py    # Cross-frame attention
│   │   ├── memory_encoder.py      # Temporal features
│   │   ├── position_encoding.py   # Spatial encoding
│   │   └── sam/             # SAM components (6 files)
│   │       ├── prompt_encoder.py
│   │       ├── mask_decoder.py
│   │       ├── transformer.py
│   │       └── ...
│   ├── utils/               # Utilities (8 files)
│   │   ├── amg.py           # Automatic mask generation
│   │   ├── misc.py          # Helpers
│   │   └── transforms.py    # Image preprocessing
│   └── configs/             # Model configurations (YAML)
│       ├── sam2_hiera_t.yaml     # Tiny (91.2 FPS)
│       ├── sam2_hiera_s.yaml     # Small (84.8 FPS)
│       ├── sam2_hiera_b+.yaml    # Base+ (64.1 FPS)
│       └── sam2_hiera_l.yaml     # Large (39.5 FPS)
├── training/                # Training code (SAM 2.1+)
│   ├── README.md
│   ├── train.py
│   ├── dataset/             # Data loading
│   └── utils/               # Training utilities
├── demo/                    # Web demo
│   ├── frontend/            # React TypeScript
│   └── backend/             # Flask GraphQL
├── notebooks/               # Jupyter examples (11)
│   ├── image_predictor_example.ipynb
│   ├── video_predictor_example.ipynb
│   ├── automatic_mask_generator_example.ipynb
│   └── ...
├── tools/                   # CLI utilities
│   ├── vos_inference.py     # Video object segmentation
│   └── ...
├── sav_dataset/             # SA-V dataset tools
├── checkpoints/             # Model weights (download separately)
├── LICENSE                  # Apache 2.0
├── README.md
└── INSTALL.md
```

---

## Core Architecture Files

### Main Entry Points

**`sam2_image_predictor.py`** - Image segmentation
- `SAM2ImagePredictor` class
- Same API as SAM 1 (backward compatible)
- Methods: `set_image()`, `predict()`

**`sam2_video_predictor.py`** - Video segmentation
- `SAM2VideoPredictor` class
- Streaming memory for object tracking
- Methods: `init_state()`, `add_new_points()`, `propagate_in_video()`

**`build_sam.py`** - Model factory
- `build_sam2()` - Loads model from config
- `build_sam2_video_predictor()` - Video API
- `build_sam2_camera_predictor()` - Real-time camera

### Architecture Components

**`modeling/sam2_base.py`** - Base Model
- Unified architecture for images + videos
- Image encoder (Hiera) + prompt encoder + mask decoder
- Memory attention for temporal consistency

**`modeling/backbones/hieradet.py`** - Hiera Encoder
- Hierarchical vision transformer
- Multi-scale features (better than ViT for video)
- 6× faster than SAM 1's ViT-H
- Input: 1024×1024 RGB → Output: Multi-scale feature pyramid

**`modeling/memory_attention.py`** - Cross-Frame Attention
- Self-attention over current frame + memory bank
- Cross-attention to object memories
- Independent per-object tracking

**`modeling/memory_encoder.py`** - Temporal Encoding
- Encodes current frame predictions into memory
- Stores object features for future frames
- Memory bank management (add/remove objects)

**`modeling/sam/` - SAM Components**
- Same prompt encoder, mask decoder, transformer as SAM 1
- Shared with image + video pipelines

---

## Model Variants

Download from: https://github.com/facebookresearch/sam2#model-checkpoints

| Model | Params | Speed (FPS) | J&F (SA-V) | Checkpoint |
|-------|--------|-------------|------------|------------|
| Hiera-Tiny | 38.9M | 91.2 | 75.0 | `sam2_hiera_tiny.pt` |
| Hiera-Small | 46M | 84.8 | 78.9 | `sam2_hiera_small.pt` |
| Hiera-Base+ | 80.8M | 64.1 | 80.8 | `sam2_hiera_base_plus.pt` |
| Hiera-Large | 224.4M | 39.5 | 81.0 (SAM 2.1) | `sam2_hiera_large.pt` |

---

## Usage Patterns

### Pattern 1: Image Segmentation (SAM 1-compatible)

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
sam2_checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

# Same API as SAM 1
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1]
)
```

### Pattern 2: Video Segmentation

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Initialize video state
inference_state = predictor.init_state(video_path=<video_path>)

# Add points on first frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=[[x, y]],
    labels=[1]
)

# Propagate through video
for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
    # Process masks for this frame
    ...
```

### Pattern 3: Real-time Camera

```python
from sam2.build_sam import build_sam2_camera_predictor

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Initialize with first frame
predictor.load_first_frame(frame)

# Add object
predictor.add_new_prompt(frame_idx=0, obj_id=1, points=[[x, y]], labels=[1])

# Track in subsequent frames
for frame in camera_stream:
    masks = predictor.track(frame)
```

---

## SA-V Dataset

**Video segmentation training dataset**:
- 50.9K videos
- 642.6K masklets (spatio-temporal masks)
- 35.5K unique object instances

**Download**: https://ai.meta.com/datasets/segment-anything-video/

---

## SAM 2.1 Updates (October 18, 2024)

**Performance improvements** (no architecture changes):
- +2.8 J&F on SA-V (78.2% → 81.0%)
- Better occlusion handling
- 3× fewer interactions for same accuracy

**New releases**:
- Training code (fine-tune on custom data)
- Web demo code (React + Flask)
- Developer suite (full toolkit)

See: `.claude/skills/karpathy-deep-oracle/sam-2/07-sam21-updates.md`

---

## Training (SAM 2.1+)

**Requirements**:
- 8× A100 80GB GPUs
- PyTorch 2.3.1+
- CUDA 12.1+

```bash
cd training
python train.py \
  --config configs/sam2_hiera_l_mose.yaml \
  --num-gpus 8
```

See: `training/README.md` for full details

---

## Installation

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**Requirements**:
- Python ≥ 3.10
- PyTorch ≥ 2.3.1
- torchvision ≥ 0.18.1

---

## Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv:2408.00714},
  year={2024}
}
```

---

## Cross-References

- **SAM 1**: `../sam1/` - Image-only segmentation
- **SAM 3**: `../sam3/` - Text + visual prompts
- **Research**: `.claude/skills/karpathy-deep-oracle/sam-2/` - SAM 2 concepts

---

**Last Updated**: November 21, 2025
**Source**: GitHub clone (--depth 1, main branch)
**Files**: 63 Python files
**Training Code**: Included (SAM 2.1+)
