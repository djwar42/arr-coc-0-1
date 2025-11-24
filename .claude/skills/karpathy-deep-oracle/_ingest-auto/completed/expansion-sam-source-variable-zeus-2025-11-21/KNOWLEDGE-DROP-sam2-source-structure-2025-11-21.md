# SAM 2 Source Code Repository Structure

**Repository**: https://github.com/facebookresearch/sam2
**Discovery Date**: 2025-11-21
**Worker**: VARIABLE ZEUS Worker 2

---

## Repository Overview

SAM 2 (Segment Anything Model 2) is Meta's foundation model for promptable visual segmentation in both images and videos. The repository contains inference code, model checkpoints, training code, and web demo.

**Key Stats**:
- **Stars**: 17.8k
- **Forks**: 2.2k
- **License**: Apache 2.0 + BSD-3-Clause (cc_torch)
- **Latest Update**: SAM 2.1 (September 29, 2024)

---

## Top-Level Repository Structure

```
facebookresearch/sam2/
├── .github/workflows/          # CI/CD workflows
├── assets/                     # Documentation images (architecture diagrams)
├── checkpoints/                # Model checkpoint downloads
│   └── download_ckpts.sh      # Script to download all checkpoints
├── demo/                       # Web demo (frontend + backend)
│   └── README.md              # Demo deployment instructions
├── notebooks/                  # Jupyter notebook examples
│   ├── image_predictor_example.ipynb
│   ├── video_predictor_example.ipynb
│   └── automatic_mask_generator_example.ipynb
├── sam2/                       # 🔥 CORE SOURCE CODE DIRECTORY
│   ├── __init__.py
│   ├── build_sam.py           # Model builders
│   ├── sam2_image_predictor.py
│   ├── sam2_video_predictor.py
│   ├── automatic_mask_generator.py
│   ├── modeling/              # Model architecture
│   ├── configs/               # Model configurations
│   └── utils/                 # Utility functions
├── sav_dataset/               # SA-V (Segment Anything Video) dataset tools
│   └── README.md              # Dataset download/usage instructions
├── tools/                     # Additional tools
├── training/                  # Training code (SAM 2.1+)
│   └── README.md              # Training instructions
├── setup.py                   # Package installation
├── pyproject.toml             # Project metadata
└── README.md                  # Main documentation
```

---

## Core Source Directory: `sam2/`

### Top-Level API Files

**Image Prediction API**:
- `sam2_image_predictor.py` - `SAM2ImagePredictor` class
  - Static image segmentation
  - Similar interface to original SAM
  - Methods: `set_image()`, `predict()`
  - Supports point/box/mask prompts

**Video Prediction API**:
- `sam2_video_predictor.py` - `SAM2VideoPredictor` class
  - Video object tracking & segmentation
  - Streaming memory architecture
  - Methods: `init_state()`, `add_new_points_or_box()`, `propagate_in_video()`
  - Multi-object tracking with independent per-object inference

**Model Building**:
- `build_sam.py` - Factory functions for building SAM 2 models
  - `build_sam2()` - Build from config + checkpoint
  - `build_sam2_video_predictor()` - Video predictor builder
  - Supports: Hiera-T, Hiera-S, Hiera-B+, Hiera-L

**Automatic Segmentation**:
- `automatic_mask_generator.py` - Automatic mask generation (no prompts)
  - Grid-based prompting
  - NMS for mask deduplication
  - Similar to original SAM's AMG

### `sam2/modeling/` - Model Architecture

**Core Architecture**:
- `sam2_base.py` - Base SAM 2 model class
  - Unified image/video segmentation
  - Memory attention mechanism
  - Streaming inference

**Backbone Encoders** (`backbones/`):
- `image_encoder.py` - Hiera vision encoder
  - Hierarchical transformer
  - Multi-scale features
  - Replaces ViT-H from SAM 1
- `hieradet.py` - Hierarchical detector variant
  - Detection-specific modifications

**Memory System**:
- `memory_attention.py` - Streaming memory attention
  - Cross-frame attention
  - Temporal consistency
  - Object permanence tracking
- `memory_encoder.py` - Memory encoder module
  - Frame feature encoding
  - Memory bank management

**SAM Components** (`sam/`):
- `prompt_encoder.py` - Prompt encoder
  - Point/box/mask embedding
  - Positional encodings
- `mask_decoder.py` - Mask decoder
  - Transformer-based decoder
  - Multi-mask prediction
- `transformer.py` - Transformer blocks
  - Attention mechanisms
  - MLP layers

### `sam2/configs/` - Model Configurations

**SAM 2.1 Configs** (`sam2.1/`):
- `sam2.1_hiera_t.yaml` - Tiny (38.9M params, 91.2 FPS)
- `sam2.1_hiera_s.yaml` - Small (46M params, 84.8 FPS)
- `sam2.1_hiera_b+.yaml` - Base Plus (80.8M params, 64.1 FPS)
- `sam2.1_hiera_l.yaml` - Large (224.4M params, 39.5 FPS)

**SAM 2 Configs** (`sam2/`):
- `sam2_hiera_t.yaml` - Original July 2024 release
- `sam2_hiera_s.yaml`
- `sam2_hiera_b+.yaml`
- `sam2_hiera_l.yaml`

**Config Structure**:
- Model architecture parameters
- Image encoder settings (Hiera depth, width)
- Memory attention configuration
- Mask decoder parameters
- Compilation flags (`compile_image_encoder`)

### `sam2/utils/` - Utility Functions

- `transforms.py` - Image/video transforms
  - Resizing, normalization
  - Data augmentation
- `amg.py` - Automatic mask generation utilities
  - Grid sampling
  - NMS operations
- `misc.py` - Miscellaneous utilities
  - Tensor operations
  - Device management

---

## Model Checkpoints

**SAM 2.1 Checkpoints** (September 2024):
- `sam2.1_hiera_tiny.pt` (38.9M)
- `sam2.1_hiera_small.pt` (46M)
- `sam2.1_hiera_base_plus.pt` (80.8M)
- `sam2.1_hiera_large.pt` (224.4M)

**Performance** (SA-V test J&F):
- Tiny: 76.5%
- Small: 76.6%
- Base+: 78.2%
- Large: 79.5%

**Download**:
```bash
cd checkpoints
./download_ckpts.sh
```

---

## Training Code (`training/`)

**Released in SAM 2.1** (September 30, 2024):
- `train.py` - Main training loop
- `dataset/` - Dataset loaders
  - SA-V dataset support
  - Custom dataset adapters
- `configs/` - Training configurations
  - Hyperparameters
  - Data augmentation settings

**Capabilities**:
- Fine-tuning on custom datasets
- Full model training from scratch
- Image and/or video training

---

## Example Notebooks (`notebooks/`)

**Image Prediction**:
- `image_predictor_example.ipynb`
  - Point prompts
  - Box prompts
  - Mask refinement

**Video Prediction**:
- `video_predictor_example.ipynb`
  - Object tracking
  - Multi-object segmentation
  - Temporal propagation

**Automatic Mask Generation**:
- `automatic_mask_generator_example.ipynb`
  - Zero-shot segmentation
  - Grid-based prompting
  - Mask filtering

**Colab Links**:
- All notebooks have Google Colab versions
- Free GPU runtime for inference

---

## Web Demo (`demo/`)

**Components**:
- Frontend: React/TypeScript
- Backend: Python/FastAPI
- Deployment: Docker Compose

**Features**:
- Interactive video annotation
- Real-time mask propagation
- Multi-object tracking UI
- Export annotations

**Deployment**:
```bash
docker-compose up
```

---

## Dataset Tools (`sav_dataset/`)

**SA-V Dataset** (Segment Anything Video):
- **Largest video segmentation dataset**
- 50,900 videos
- 642,600 masklets (spatio-temporal masks)
- Diverse domains (nature, sports, medicine)

**Tools**:
- Download scripts
- Data loaders
- Visualization utilities

---

## Installation Requirements

**Python**: >=3.10
**PyTorch**: >=2.5.1
**TorchVision**: >=0.20.1

**Dependencies**:
- `torch` - Deep learning framework
- `torchvision` - Vision utilities
- `hydra-core` - Configuration management
- `iopath` - I/O utilities
- `numpy` - Numerical operations
- `pillow` - Image processing
- `tqdm` - Progress bars

**Optional**:
- `jupyter` - Notebook support
- `matplotlib` - Visualization

**Custom CUDA Kernel**:
- Connected components algorithm (cc_torch)
- Requires `nvcc` compiler
- Optional (post-processing only)

---

## Key Architecture Differences from SAM 1

### Vision Encoder
- **SAM 1**: ViT-H (Vision Transformer)
- **SAM 2**: Hiera (Hierarchical Transformer)
  - Multi-scale features
  - More efficient
  - Better for video

### Memory System
- **SAM 1**: Stateless (per-image)
- **SAM 2**: Streaming memory
  - Cross-frame attention
  - Temporal consistency
  - Object tracking

### Video Support
- **SAM 1**: Image-only
- **SAM 2**: Native video support
  - Frame-by-frame propagation
  - Masklet tracking
  - Multi-object handling

---

## Estimated File Counts

**Core `sam2/` directory**:
- Python files: ~40-50 files
- Config YAML files: ~10 files
- Total: ~50-60 files

**Breakdown by subdirectory**:
- `modeling/`: ~20-25 Python files
  - `backbones/`: ~5 files
  - `sam/`: ~5 files
  - Core: ~10-15 files
- `configs/`: ~10 YAML files
- `utils/`: ~5-8 Python files
- Top-level API: ~5 Python files

**Full repository**:
- Python source: ~50-60 files
- Notebooks: 3 files
- Documentation: ~10 MD files
- Config/Setup: ~5 files
- **Total**: ~70-80 files (excluding training/)

---

## Key Classes & Functions

### `SAM2ImagePredictor`
```python
predictor = SAM2ImagePredictor(model)
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    box=box
)
```

### `SAM2VideoPredictor`
```python
predictor = build_sam2_video_predictor(config, checkpoint)
state = predictor.init_state(video_path)

# Add prompts
frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    state, frame_idx, obj_id, points, labels
)

# Propagate through video
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    # Process masks
    pass
```

### `SAM2AutomaticMaskGenerator`
```python
mask_generator = SAM2AutomaticMaskGenerator(model)
masks = mask_generator.generate(image)
```

---

## Module Dependencies

**Image Predictor** depends on:
- `modeling/sam2_base.py` - Base model
- `modeling/backbones/image_encoder.py` - Hiera encoder
- `modeling/sam/prompt_encoder.py` - Prompt encoding
- `modeling/sam/mask_decoder.py` - Mask decoding
- `utils/transforms.py` - Image preprocessing

**Video Predictor** depends on:
- All image predictor dependencies
- `modeling/memory_attention.py` - Temporal attention
- `modeling/memory_encoder.py` - Frame encoding
- Video-specific utilities

---

## HuggingFace Integration

**Model Hub**:
- `facebook/sam2-hiera-tiny`
- `facebook/sam2-hiera-small`
- `facebook/sam2-hiera-base-plus`
- `facebook/sam2-hiera-large`

**Loading**:
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
```

---

## Sources

**Primary Source**:
- GitHub Repository: https://github.com/facebookresearch/sam2
- Accessed: 2025-11-21

**Documentation**:
- Main README: https://github.com/facebookresearch/sam2/blob/main/README.md
- Installation Guide: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
- Training Guide: https://github.com/facebookresearch/sam2/blob/main/training/README.md
- Demo Guide: https://github.com/facebookresearch/sam2/blob/main/demo/README.md

**Research**:
- Paper: SAM 2: Segment Anything in Images and Videos (arXiv:2408.00714)
- Project Page: https://ai.meta.com/sam2
- Dataset: https://ai.meta.com/datasets/segment-anything-video

---

## Discovery Status

✅ **COMPLETE** - Repository structure mapped
✅ **Core directories identified** - sam2/, modeling/, configs/
✅ **Key files documented** - Predictors, encoders, decoders
✅ **Architecture understood** - Hiera + Memory Attention
✅ **File counts estimated** - ~50-60 Python files in sam2/
✅ **API surface mapped** - Image/Video predictors, AMG

**Next Steps**:
- VARIABLE ZEUS Worker 3: Fetch individual Python files
- Add claudes_code_comments to all source files
- Create detailed module documentation
