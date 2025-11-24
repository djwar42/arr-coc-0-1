# SAM 1 Repository Structure - Complete Source Code Map

**Repository**: https://github.com/facebookresearch/segment-anything
**Purpose**: Segment Anything Model (SAM) - Foundation model for promptable image segmentation
**License**: Apache 2.0
**Stars**: 52.6k | **Forks**: 6.2k
**Date Accessed**: 2025-11-21

---

## Repository Overview

The Segment Anything Model (SAM) produces high-quality object masks from input prompts (points, boxes) and can generate masks for all objects in an image. Trained on 11M images and 1.1B masks with strong zero-shot performance.

**Key Features**:
- Promptable segmentation (points, boxes, text)
- Automatic mask generation
- Three model sizes (ViT-H, ViT-L, ViT-B)
- ONNX export for web deployment
- SA-1B dataset (1.1 billion masks)

---

## Complete File Structure

### 📁 Root Directory

```
segment-anything/
├── .flake8                    # Python linter config
├── .gitignore                 # Git ignore rules
├── CODE_OF_CONDUCT.md         # Community guidelines
├── CONTRIBUTING.md            # Contribution instructions
├── LICENSE                    # Apache 2.0 license
├── README.md                  # Main documentation
├── linter.sh                  # Linting script
├── setup.cfg                  # Setup configuration
├── setup.py                   # Package installation script
├── assets/                    # Images and diagrams
├── demo/                      # Web demo (React app)
├── notebooks/                 # Jupyter example notebooks
├── scripts/                   # Command-line tools
└── segment_anything/          # Main Python package
```

---

## 📦 Core Package: `segment_anything/`

### Top-Level Files (Public API)

| File | Purpose | Key Classes/Functions | Dependencies |
|------|---------|----------------------|--------------|
| **`__init__.py`** | Package exports | Exports all public APIs | All submodules |
| **`build_sam.py`** | Model builders | `sam_model_registry`, `build_sam_vit_h()`, `build_sam_vit_l()`, `build_sam_vit_b()`, `build_sam()` | `modeling.sam` |
| **`predictor.py`** | Image-level API | `SamPredictor` class (set_image, predict, set_torch_image) | `modeling.sam`, `utils.transforms` |
| **`automatic_mask_generator.py`** | Auto segmentation | `SamAutomaticMaskGenerator` class (generate, generate_masks) | `predictor`, `utils.amg` |

**Typical Usage Flow**:
```python
# 1. Build model
from segment_anything import sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="path/to/checkpoint")

# 2. Create predictor
from segment_anything import SamPredictor
predictor = SamPredictor(sam)
predictor.set_image(image)

# 3. Get masks
masks, _, _ = predictor.predict(points=[[x, y]], labels=[1])
```

---

### 🏗️ `modeling/` - Model Architecture

| File | Purpose | Key Classes | Architecture Details |
|------|---------|-------------|---------------------|
| **`sam.py`** | Main model | `Sam` class | Integrates image encoder, prompt encoder, mask decoder |
| **`image_encoder.py`** | Vision backbone | `ImageEncoderViT` class | ViT-H/L/B encoder (processes 1024×1024 images) |
| **`prompt_encoder.py`** | Prompt processing | `PromptEncoder` class | Encodes points, boxes, masks into embeddings |
| **`mask_decoder.py`** | Mask generation | `MaskDecoder` class | Lightweight decoder (IoU prediction + mask output) |
| **`transformer.py`** | Attention blocks | `TwoWayTransformer`, `TwoWayAttentionBlock`, `Attention` | Cross-attention between image and prompt embeddings |
| **`common.py`** | Shared components | `MLPBlock`, `LayerNorm2d` | Building blocks used across modules |

**Architecture Flow**:
```
Input Image (HxW)
    ↓
ImageEncoderViT (ViT-H: 1024×1024 → 64×64 embeddings)
    ↓
[Image Embeddings] + [Prompt Embeddings from PromptEncoder]
    ↓
TwoWayTransformer (cross-attention)
    ↓
MaskDecoder → Output Masks + IoU scores
```

**Model Sizes**:
- **ViT-H** (default): 636M params, best quality
- **ViT-L**: 308M params, balanced
- **ViT-B**: 89M params, fastest

---

### 🔧 `utils/` - Utilities

| File | Purpose | Key Functions | Used By |
|------|---------|---------------|---------|
| **`transforms.py`** | Image preprocessing | `ResizeLongestSide`, `apply_image()`, `apply_boxes()`, `apply_coords()` | `predictor.py` |
| **`amg.py`** | Auto mask generation helpers | `generate_crop_boxes()`, `calculate_stability_score()`, `build_point_grid()`, `build_all_layer_point_grids()` | `automatic_mask_generator.py` |
| **`onnx.py`** | ONNX export utilities | `SamOnnxModel` class (wraps decoder for export) | `scripts/export_onnx_model.py` |

**Transform Pipeline**:
```python
# ResizeLongestSide: Scales image to 1024 on longest side
transform = ResizeLongestSide(1024)
image = transform.apply_image(original_image)  # HWC numpy array
coords = transform.apply_coords(points, original_image.shape[:2])
```

---

## 📝 `scripts/` - Command-Line Tools

| File | Purpose | Usage | Output |
|------|---------|-------|--------|
| **`amg.py`** | Automatic mask generation CLI | `python scripts/amg.py --checkpoint <path> --model-type vit_h --input <image/folder> --output <dir>` | JSON masks in COCO RLE format |
| **`export_onnx_model.py`** | ONNX export | `python scripts/export_onnx_model.py --checkpoint <path> --model-type vit_h --output <path.onnx>` | ONNX model file (for web deployment) |

**Automatic Mask Generation**:
- Generates masks for ALL objects in image
- No prompts needed (fully automatic)
- Outputs JSON with COCO RLE masks + metadata

---

## 📓 `notebooks/` - Example Notebooks

| Notebook | Purpose | Demonstrates |
|----------|---------|--------------|
| **`predictor_example.ipynb`** | Prompted segmentation | Using points, boxes, and masks as prompts |
| **`automatic_mask_generator_example.ipynb`** | Automatic segmentation | Generating all masks in an image |
| **`onnx_model_example.ipynb`** | ONNX inference | Using exported ONNX model for deployment |

---

## 🌐 `demo/` - Web Demo

**Purpose**: Single-page React app demonstrating ONNX model in browser

**Structure**:
```
demo/
├── README.md          # Demo setup instructions
├── package.json       # npm dependencies
├── public/            # Static assets
├── src/               # React source code
│   ├── App.tsx        # Main app component
│   ├── components/    # UI components
│   └── helpers/       # ONNX inference helpers
└── build/             # Production build
```

**Key Features**:
- In-browser inference using ONNX runtime
- Web workers for multithreading
- Interactive point/box prompting

---

## 🎨 `assets/` - Documentation Assets

| File | Purpose |
|------|---------|
| **`model_diagram.png`** | SAM architecture diagram |
| **`masks1.png`** | Example segmentation results |
| **`masks2.jpg`** | More example results |
| **`notebook1.png`** | Notebook screenshot |
| **`notebook2.png`** | Notebook screenshot |

---

## Module Dependency Graph

```
┌─────────────────────────────────────────┐
│         Top-Level API                   │
│  build_sam.py, predictor.py,           │
│  automatic_mask_generator.py           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         modeling/                       │
│  sam.py (main model)                    │
│    ├── image_encoder.py (ViT)          │
│    ├── prompt_encoder.py               │
│    ├── mask_decoder.py                 │
│    └── transformer.py (attention)      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         utils/                          │
│  transforms.py, amg.py, onnx.py        │
└─────────────────────────────────────────┘
```

---

## Key Classes and Their Relationships

### 1. **Sam** (modeling/sam.py)
**Purpose**: Main model class integrating all components

**Attributes**:
- `image_encoder`: ImageEncoderViT instance
- `prompt_encoder`: PromptEncoder instance
- `mask_decoder`: MaskDecoder instance

**Methods**:
- `forward()`: Full forward pass
- `postprocess_masks()`: Rescale masks to original image size

### 2. **SamPredictor** (predictor.py)
**Purpose**: High-level API for image-level prediction

**Key Methods**:
- `set_image(image)`: Preprocess and encode image (caches embeddings)
- `predict(point_coords, point_labels, box, mask_input)`: Generate masks from prompts
- `set_torch_image()`: Direct tensor input (skips transforms)

**Usage Pattern**:
```python
predictor = SamPredictor(sam)
predictor.set_image(image)  # Encode once
masks1 = predictor.predict(points=[[x1, y1]])  # Fast (uses cached embeddings)
masks2 = predictor.predict(points=[[x2, y2]])  # Fast (no re-encoding)
```

### 3. **SamAutomaticMaskGenerator** (automatic_mask_generator.py)
**Purpose**: Automatic mask generation for entire image

**Parameters**:
- `points_per_side`: Grid resolution for sampling (default: 32)
- `pred_iou_thresh`: Quality threshold (default: 0.88)
- `stability_score_thresh`: Stability threshold (default: 0.95)

**Output Format**:
```python
[
  {
    'segmentation': <numpy array>,  # Binary mask
    'area': <int>,                  # Mask area in pixels
    'bbox': [x, y, w, h],           # Bounding box
    'predicted_iou': <float>,       # Model's quality prediction
    'stability_score': <float>,     # Mask stability
    'point_coords': [[x, y]]        # Prompt point used
  },
  ...
]
```

---

## Model Checkpoints

**Download URLs**:
- **ViT-H** (default): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **ViT-L**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- **ViT-B**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

**Loading**:
```python
from segment_anything import sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
```

---

## Dataset: SA-1B

**Overview**:
- 11 million images
- 1.1 billion masks
- Largest segmentation dataset ever

**Data Format** (JSON per image):
```python
{
  "image": {
    "image_id": int,
    "width": int,
    "height": int,
    "file_name": str
  },
  "annotations": [
    {
      "id": int,
      "segmentation": dict,      # COCO RLE format
      "bbox": [x, y, w, h],
      "area": int,
      "predicted_iou": float,
      "stability_score": float,
      "crop_box": [x, y, w, h],
      "point_coords": [[x, y]]
    }
  ]
}
```

**Decoding Masks**:
```python
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

---

## Installation & Dependencies

**Python Requirements**:
- Python >= 3.8
- PyTorch >= 1.7
- torchvision >= 0.8

**Install via pip**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Install from source**:
```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

**Optional Dependencies**:
```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx jupyter
```

---

## File Count Summary

| Category | Count | Examples |
|----------|-------|----------|
| **Core API** | 4 files | `__init__.py`, `build_sam.py`, `predictor.py`, `automatic_mask_generator.py` |
| **Modeling** | 6 files | `sam.py`, `image_encoder.py`, `prompt_encoder.py`, `mask_decoder.py`, `transformer.py`, `common.py` |
| **Utils** | 3 files | `transforms.py`, `amg.py`, `onnx.py` |
| **Scripts** | 2 files | `amg.py`, `export_onnx_model.py` |
| **Notebooks** | 3 files | `predictor_example.ipynb`, `automatic_mask_generator_example.ipynb`, `onnx_model_example.ipynb` |
| **Config** | 5 files | `setup.py`, `setup.cfg`, `.flake8`, `.gitignore`, `linter.sh` |
| **Docs** | 3 files | `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` |
| **Demo** | ~10-15 files | React app (TypeScript, components, helpers) |
| **Assets** | 5 images | PNG/JPG diagrams and examples |

**Total Python Files**: ~18-20 files
**Total Repository Size**: ~25-30 files (excluding demo node_modules)

---

## Usage Patterns

### Pattern 1: Prompted Segmentation
```python
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Set image (encode once)
predictor.set_image(image)

# Predict with point
masks, scores, logits = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1]  # 1=foreground, 0=background
)

# Predict with box
masks, scores, logits = predictor.predict(
    box=[x1, y1, x2, y2]
)
```

### Pattern 2: Automatic Mask Generation
```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate all masks
masks = mask_generator.generate(image)
# Returns list of dicts with 'segmentation', 'area', 'bbox', etc.
```

### Pattern 3: ONNX Export & Deployment
```bash
# 1. Export decoder to ONNX
python scripts/export_onnx_model.py \
  --checkpoint sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --output sam_decoder.onnx

# 2. Use in web browser (see demo/)
# Image encoder runs in Python/CUDA
# Mask decoder runs in browser with ONNX Runtime
```

---

## Sources

**GitHub Repository**:
- https://github.com/facebookresearch/segment-anything

**Documentation**:
- README: https://github.com/facebookresearch/segment-anything/blob/main/README.md
- Paper: https://ai.facebook.com/research/publications/segment-anything/
- Project Page: https://segment-anything.com/
- Dataset: https://ai.facebook.com/datasets/segment-anything/

**Accessed**: 2025-11-21
**Repository Stars**: 52.6k
**Repository Forks**: 6.2k
**License**: Apache 2.0

---

## Next Steps for VARIABLE ZEUS

**Discovery Complete**: ✅
**Total Files Identified**: ~25-30 files
**Python Source Files**: ~18-20 files

**Next Phase**: Fetch Workers
**Workers Needed**: ~25 workers (one per file)
**Each Worker**: Fetch ONE file + create detailed KNOWLEDGE-DROP about its structure, classes, and functions

**Files to Fetch** (Priority Order):
1. **Core API** (4 files): `__init__.py`, `build_sam.py`, `predictor.py`, `automatic_mask_generator.py`
2. **Modeling** (6 files): `sam.py`, `image_encoder.py`, `prompt_encoder.py`, `mask_decoder.py`, `transformer.py`, `common.py`
3. **Utils** (3 files): `transforms.py`, `amg.py`, `onnx.py`
4. **Scripts** (2 files): `amg.py`, `export_onnx_model.py`
5. **Config** (2 files): `setup.py`, `requirements.txt`

**Integration Phase**: Main worker reviews all KNOWLEDGE-DROPs + adds claudes_code_comments to ALL Python files
