# SAM (Segment Anything Model): Complete Research Study

**Research Date:** 2025-11-20
**Status:** Comprehensive Deep Dive
**Model Family:** SAM, SAM 2, SAM 3 (November 2025)

---

## Table of Contents

1. [Overview](#overview)
2. [SAM 1: The Original Foundation Model](#sam-1-the-original-foundation-model)
3. [SAM 2: Video Segmentation](#sam-2-video-segmentation)
4. [SAM 3: Text-Prompted Segmentation](#sam-3-text-prompted-segmentation)
5. [Technical Architecture](#technical-architecture)
6. [Training Methodology](#training-methodology)
7. [Code Examples](#code-examples)
8. [Applications](#applications)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Research Papers](#research-papers)
11. [Resources](#resources)

---

## Overview

**Segment Anything Model (SAM)** is a family of **foundation models for image and video segmentation** developed by Meta AI (FAIR). The SAM family represents a paradigm shift in computer vision, enabling:

✅ **Promptable segmentation** (points, boxes, text, masks)
✅ **Zero-shot generalization** to new domains
✅ **Real-time inference** on diverse imagery
✅ **Foundation model** for downstream tasks

### The SAM Family (2023-2025)

| Model | Release | Focus | Key Innovation |
|-------|---------|-------|----------------|
| **SAM** | April 2023 | Image segmentation | Promptable masks from 1.1B annotations |
| **SAM 2** | August 2024 | Video segmentation | Streaming memory for temporal consistency |
| **SAM 3** | November 2025 | Text + Visual prompts | Open-vocabulary concept segmentation |

### Why SAM Matters

**Before SAM:**
- Segmentation models required:
  - Large annotated datasets for each domain
  - Task-specific training
  - Domain expertise

**With SAM:**
- **Zero-shot** transfer to new domains
- **Promptable** interface (no retraining needed)
- **General-purpose** foundation model

---

## SAM 1: The Original Foundation Model

### 📄 Paper Information

**Title:** Segment Anything
**arXiv:** https://arxiv.org/abs/2304.02643
**Published:** April 5, 2023

**Authors:** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick (Meta AI)

**GitHub:** https://github.com/facebookresearch/segment-anything

### 🎯 Core Contributions

1. **Task:** Promptable segmentation
2. **Model:** SAM (Segment Anything Model)
3. **Dataset:** SA-1B (1.1B masks, 11M images)

### 🔬 Key Features

**Promptable Interface:**
- **Point prompts:** Click to segment
- **Box prompts:** Bounding box → mask
- **Mask prompts:** Rough mask → refined mask
- **Text prompts:** (via CLIP integration)

**Zero-Shot Performance:**
- Generalizes to new image distributions
- No task-specific training required
- Robust across domains (natural, medical, satellite, etc.)

**Automatic Mask Generation:**
- Generate **all** masks in image
- Hierarchical segmentation
- No prompts needed

### 📊 Training Details

**Dataset:** SA-1B
- 11 million images
- 1.1 billion masks
- Class-agnostic annotations

**Data Engine (3 stages):**

1. **Assisted-Manual (120k images):**
   - Expert annotators
   - SAM assists annotation
   - 6× faster than manual

2. **Semi-Automatic (180k images):**
   - SAM generates masks
   - Humans refine
   - Confidence-based selection

3. **Fully Automatic (11M images):**
   - SAM generates all masks
   - Human quality verification
   - 100 masks/image average

**Training Strategy:**
- Pre-train on synthetic data
- Fine-tune on SA-1B
- Promptable training objective

### 🏗️ Architecture (Detailed in [Technical Architecture](#technical-architecture))

```
Input Image (1024×1024)
    ↓
ViT-H Image Encoder (MAE pre-trained)
    ↓
Prompt Encoder (sparse + dense)
    ↓
Mask Decoder (lightweight)
    ↓
Output Masks (multiple candidates)
```

### 📦 Model Checkpoints

**Three sizes available:**

| Model | Params | Checkpoint | Size | Speed |
|-------|--------|-----------|------|-------|
| **ViT-H** | 636M | `sam_vit_h_4b8939.pth` | 2.4 GB | Slow, best quality |
| **ViT-L** | 308M | `sam_vit_l_0b3195.pth` | 1.2 GB | Medium |
| **ViT-B** | 91M | `sam_vit_b_01ec64.pth` | 375 MB | Fast, good quality |

**Download:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 💻 Installation & Basic Usage

**Install:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Quick Start:**
```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# Load image
image = np.array(Image.open("image.jpg"))
predictor.set_image(image)

# Segment with point prompt
input_point = np.array([[500, 375]])  # (x, y)
input_label = np.array([1])  # 1=foreground, 0=background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Get 3 mask candidates
)

# Best mask
best_mask = masks[np.argmax(scores)]
```

**Automatic Mask Generation:**
```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam)

# Generate all masks
masks = mask_generator.generate(image)

# Each mask in masks is a dict:
# {
#   'segmentation': binary mask (H, W),
#   'area': int,
#   'bbox': [x, y, w, h],
#   'predicted_iou': float,
#   'stability_score': float
# }

print(f"Generated {len(masks)} masks")
```

### 🌐 Interactive Demo

**Official Playground:** https://www.aidemos.meta.com/segment-anything

---

## SAM 2: Video Segmentation

### 📄 Paper Information

**Title:** SAM 2: Segment Anything in Images and Videos
**arXiv:** https://arxiv.org/abs/2408.00714
**Published:** August 2024

**Authors:** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer (Meta AI)

**GitHub:** https://github.com/facebookresearch/sam2

### 🎯 Core Innovation

**Unified model for images AND videos:**
- **Real-time video segmentation**
- **Streaming memory architecture**
- **Temporal consistency** across frames
- **6× faster than SAM 1** on images

### 🔬 Key Features

**Video Segmentation:**
- **Promptable tracking:** Click once, track throughout video
- **Propagation:** Automatically propagates masks across frames
- **Occlusion handling:** Tracks objects through occlusions
- **Real-time:** Processes 44 FPS on A100 GPU

**Image Improvements:**
- **6× faster** than SAM 1 on images
- **Better accuracy** on challenging cases
- **Same promptable interface**

### 🏗️ Architecture: Streaming Memory

```
Video Frames (1, 2, 3, ..., T)
    ↓
Per-Frame Image Encoder (Hiera transformer)
    ↓
Memory Attention Module
    ├─ Memory Bank (recent frames)
    └─ Prompt Memory (user clicks)
    ↓
Mask Decoder
    ↓
Tracked Masks (temporal consistency)
```

**Key Innovation:** **Memory Attention**
- Stores recent frame features
- Attends to memory for temporal context
- Enables real-time streaming video processing

### 📊 Training Dataset

**SA-V (Segment Anything Video):**
- Built using **data engine** (like SA-1B)
- **Largest video segmentation dataset**
- Covers diverse videos

### 📦 Model Checkpoints

| Model | Config | Checkpoint | Size |
|-------|--------|-----------|------|
| **SAM 2 Tiny** | sam2_hiera_t.yaml | sam2_hiera_tiny.pt | 154 MB |
| **SAM 2 Small** | sam2_hiera_s.yaml | sam2_hiera_small.pt | 185 MB |
| **SAM 2 Base+** | sam2_hiera_b+.yaml | sam2_hiera_base_plus.pt | 310 MB |
| **SAM 2 Large** | sam2_hiera_l.yaml | sam2_hiera_large.pt | 900 MB |

### 💻 Installation & Usage

**Install:**
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**Video Segmentation:**
```python
from sam2.build_sam import build_sam2_video_predictor
import torch

# Load SAM 2 for video
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Initialize video
with torch.inference_mode():
    state = predictor.init_state(video_path="video.mp4")

    # Add point prompt on frame 0
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=[[200, 300]],  # (x, y)
        labels=[1]  # foreground
    )

    # Propagate masks through entire video
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks: (1, H, W) for each object
        # Save or visualize masks
        pass
```

**Image Segmentation (6× faster than SAM 1):**
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=[[500, 375]],
    point_labels=[1]
)
```

### 📈 Performance Improvements

**vs SAM 1 (Images):**
- **6× faster** inference
- **Higher accuracy** on complex scenes
- **Better occlusion handling**

**Video:**
- **44 FPS** on A100 GPU (real-time)
- **3× fewer interactions** needed vs prior methods
- **State-of-the-art** on video segmentation benchmarks

### 🌐 Interactive Demo

**Official Demo:** https://sam2.metademolab.com/demo

---

## SAM 3: Text-Prompted Segmentation

### 📄 Paper Information

**Title:** SAM 3: Segment Anything with Concepts
**Meta AI Research:** https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
**GitHub:** https://github.com/facebookresearch/sam3
**Released:** November 19, 2025

### 🎯 Core Innovation

**Open-vocabulary segmentation:**
- **Text prompts:** "Segment all cats" → masks of all cats
- **Visual exemplars:** Show example → find similar objects
- **Concept detection:** Detect 270K unique concepts
- **Video + Image:** Unified model for both

### 🔬 Key Features

**Text-Based Prompting:**
```python
# Natural language prompts
"Find all red apples"
"Segment the person wearing a hat"
"Detect all stop signs"
```

**Exhaustive Segmentation:**
- Finds **all instances** of a concept in image/video
- Not just one object - ALL matching objects
- 270K concept vocabulary (50× more than existing benchmarks)

**Performance:**
- **75-80% of human performance** on SA-CO benchmark
- **54.1 cgF1** on SA-Co/Gold dataset
- **37.2 cgF1** on LVIS instance segmentation

### 🏗️ Architecture

**848M total parameters:**

```
Text Prompt / Visual Exemplar
    ↓
Detector (finds all instances)
    ├─ Vision Encoder (shared with Tracker)
    └─ Text Encoder (for language understanding)
    ↓
Tracker (tracks across frames)
    └─ Memory Bank (temporal consistency)
    ↓
Masks for all concept instances
```

**Novel Components:**
1. **Presence Token:** Improves discrimination between similar prompts
2. **Decoupled Detector-Tracker:** Separate detection and tracking
3. **Open-Vocabulary Capability:** Understands 270K concepts

### 📊 Training Dataset: SA-Co

**SA-Co (Segment Anything with Concepts):**
- **270K unique concepts** (50× more than LVIS)
- **Three evaluation sets:**
  - **SA-Co/Gold:** High-quality image annotations
  - **SA-Co/Silver:** Larger image dataset
  - **SA-Co/VEval:** Video evaluation set

### 📦 Installation & Requirements

**Requirements:**
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+

**Install:**
```bash
conda create -n sam3 python=3.12
conda activate sam3

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 💻 Usage Examples

**Image Segmentation with Text:**
```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("image.jpg")
inference_state = processor.set_image(image)

# Text prompt
output = processor.set_text_prompt(
    state=inference_state,
    prompt="red apple"  # Find all red apples
)

masks = output["masks"]    # All masks matching "red apple"
boxes = output["boxes"]    # Bounding boxes
scores = output["scores"]  # Confidence scores

print(f"Found {len(masks)} red apples")
```

**Video Segmentation with Text:**
```python
from sam3.model_builder import build_sam3_video_predictor

# Load video predictor
video_predictor = build_sam3_video_predictor()

# Start session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="video.mp4"
    )
)

session_id = response["session_id"]

# Add text prompt for frame 0
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="person wearing red shirt"
    )
)

# Propagate through video
# (automatically tracks "person wearing red shirt" through all frames)
```

**Visual Exemplar Prompting:**
```python
# Use visual example instead of text
output = processor.set_visual_prompt(
    state=inference_state,
    exemplar_box=[100, 100, 200, 200]  # BBox of example object
)

# Finds all visually similar objects
similar_masks = output["masks"]
```

### 📈 Performance Benchmarks

**SA-Co/Gold:**
- SAM 3: **54.1 cgF1**
- Human baseline: **72.8 cgF1**
- SAM 3 achieves **75-80% of human performance**

**LVIS (Instance Segmentation):**
- SAM 3: **37.2 cgF1**
- State-of-the-art open-vocabulary performance

**Video (SA-V):**
- **30.3 cgF1** on SA-V test
- **50.8 cgF1** on YT-Temporal-1B test

### 🔧 Advanced Features

**SAM 3 Agent (Complex Queries):**
```python
from sam3.agent import SAM3Agent

agent = SAM3Agent(model)

# Complex natural language query
result = agent.process_query(
    image=image,
    query="Find all animals that are not cats or dogs"
)

# Agent breaks down query:
# 1. Detect all animals
# 2. Filter out cats
# 3. Filter out dogs
# 4. Return remaining masks
```

**Batched Inference:**
```python
# Process multiple images efficiently
batch_results = processor.batch_inference(
    images=[img1, img2, img3],
    prompts=["red car", "red car", "blue car"]
)
```

---

## Technical Architecture

### SAM 1 Architecture (Detailed)

#### Image Encoder: Vision Transformer (ViT-H)

**Base:** Masked Autoencoder (MAE) pre-trained ViT
**Configuration:**
- Model: **ViT-H/16** (Huge, 16×16 patches)
- Input: **1024 × 1024 × 3** RGB image
- Output: **64 × 64 × 256** embedding (16× downscale)
- Attention: **14×14 windowed attention** + **4 global attention blocks**

**Architecture Details:**
```python
# ViT-H Parameters
num_layers = 32
hidden_dim = 1280
num_heads = 16
mlp_ratio = 4.0
patch_size = 16

# Total params: ~636M
```

**Processing Flow:**
```
Image (1024×1024×3)
    ↓ Patch Embedding
Patches (64×64×256)
    ↓ Transformer Blocks (32 layers)
    ├─ Windowed Self-Attention (14×14)
    ├─ Global Attention (every 8th layer)
    └─ MLP (4× expansion)
    ↓
Feature Map (64×64×256)
```

#### Prompt Encoder

**Sparse Prompts (points, boxes):**
- **Positional encoding:** Sine-cosine embeddings
- **Prompt type embedding:** Learned embedding per type
- **Combined:** Sum of position + type

**Dense Prompts (masks):**
- **Convolutional embedding:** 1×1 conv on input mask
- **Downsampled:** Match image embedding resolution

**Text Prompts (CLIP):**
- **Text encoder:** Off-the-shelf CLIP
- **Embedding:** 512-dim text features
- **Projection:** Linear layer to match prompt embedding dim

```python
# Point prompt encoding
def encode_point(coord, label):
    pos_enc = positional_encoding(coord)  # Sine-cosine
    type_enc = learned_embedding[label]   # 1=fg, 0=bg
    return pos_enc + type_enc

# Box prompt encoding
def encode_box(box):
    # Encode as 2 points (top-left, bottom-right)
    tl = encode_point(box[:2], label=2)  # Top-left
    br = encode_point(box[2:], label=3)  # Bottom-right
    return torch.stack([tl, br])

# Mask prompt encoding
def encode_mask(mask):
    # Convolutional embedding
    return conv_embedding(mask)  # (64×64×256)
```

#### Mask Decoder

**Modified Transformer Decoder:**
- **2 decoder blocks**
- **Prompt self-attention:** Prompts attend to each other
- **Cross-attention (2-way):**
  - Prompt → Image embedding
  - Image embedding → Prompt
- **Dynamic mask prediction head**

**Multi-Mask Output:**
- Generates **3 mask candidates** per prompt
- Each with **predicted IoU score**
- User selects best (or use highest IoU)

```python
class MaskDecoder(nn.Module):
    def __init__(self):
        self.transformer = Transformer(
            num_layers=2,
            hidden_dim=256,
            num_heads=8
        )

        # Learnable mask tokens (3 outputs)
        self.mask_tokens = nn.Embedding(3, 256)

        # Learnable IoU token
        self.iou_token = nn.Embedding(1, 256)

        # Prediction heads
        self.mask_head = MLP(256, 256, 1)  # Per-pixel logits
        self.iou_head = MLP(256, 256, 3)   # IoU scores

    def forward(self, image_emb, prompt_emb):
        # Combine tokens
        tokens = torch.cat([
            self.mask_tokens.weight,  # 3 mask tokens
            self.iou_token.weight,    # 1 IoU token
            prompt_emb                # User prompts
        ], dim=0)

        # Transformer decoder (2 blocks)
        tokens, image_emb = self.transformer(
            query=tokens,
            key=image_emb,
            value=image_emb
        )

        # Predict masks (3 candidates)
        mask_tokens = tokens[:3]
        masks = self.mask_head(mask_tokens, image_emb)  # (3, H, W)

        # Predict IoU scores
        iou_token = tokens[3]
        iou_scores = self.iou_head(iou_token)  # (3,)

        return masks, iou_scores
```

### SAM 2 Architecture

#### Hiera Image Encoder

**Replaces ViT-H with Hiera:**
- **Hierarchical Vision Transformer**
- **Faster than ViT** (6× speedup)
- **Multi-scale features**

**Stages:**
```
Stage 1: 256×256 resolution
Stage 2: 128×128 resolution
Stage 3: 64×64 resolution
Stage 4: 32×32 resolution (final embedding)
```

#### Memory Attention Module

**Streaming Memory for Videos:**

```python
class MemoryAttention(nn.Module):
    def __init__(self):
        self.memory_bank = []  # Stores recent frames
        self.prompt_memory = []  # Stores user prompts

    def forward(self, current_frame_features, prompts):
        # Cross-attention to memory
        attended_memory = cross_attention(
            query=current_frame_features,
            key=self.memory_bank,
            value=self.memory_bank
        )

        # Combine current + memory
        fused = current_frame_features + attended_memory

        # Update memory bank (FIFO)
        self.memory_bank.append(current_frame_features)
        if len(self.memory_bank) > MAX_MEMORY:
            self.memory_bank.pop(0)

        return fused
```

**Benefits:**
- **Temporal consistency:** Uses past frames
- **Streaming:** Doesn't need full video in memory
- **Real-time:** Efficient attention mechanism

### SAM 3 Architecture

#### Detector-Tracker Design

**Decoupled architecture:**
- **Detector:** Finds all instances of concept in single frame
- **Tracker:** Propagates detections across video frames
- **Shared vision encoder:** Both use same image features

**Presence Token:**
- Novel learnable token
- Improves discrimination between similar concepts
- Example: "red apple" vs "green apple"

```python
class SAM3Detector(nn.Module):
    def __init__(self):
        self.vision_encoder = HieraEncoder()
        self.text_encoder = CLIPTextEncoder()
        self.presence_token = nn.Embedding(1, 512)  # NEW!

        self.detection_head = DetectionHead()

    def forward(self, image, text_prompt):
        # Encode image
        image_features = self.vision_encoder(image)

        # Encode text
        text_features = self.text_encoder(text_prompt)

        # Add presence token
        text_features = text_features + self.presence_token.weight

        # Cross-modal fusion
        fused = cross_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )

        # Detect all instances
        detections = self.detection_head(fused)
        return detections  # All masks matching concept
```

---

## Training Methodology

### SAM 1 Training

**Three-Stage Data Engine:**

#### Stage 1: Assisted-Manual (120K images)
```
Human Annotator
    ↓ Clicks points
SAM (pre-trained on public data)
    ↓ Generates mask
Human
    ↓ Refines if needed
Final Mask
```

**Result:** 6× faster than manual annotation

#### Stage 2: Semi-Automatic (180K images)
```
SAM (re-trained on Stage 1 data)
    ↓ Generates confident masks automatically
Human
    ↓ Annotates remaining objects
Final Masks (100+ per image)
```

**Result:** Mix of automatic + manual annotations

#### Stage 3: Fully Automatic (11M images)
```
SAM (re-trained on Stage 1+2)
    ↓ Generates all masks (no human in loop)
Human Quality Check
    ↓ Random sampling validation
Final Dataset: SA-1B
```

**Result:** 1.1B masks across 11M images

### SAM 2 Training

**Data Engine for Videos:**
1. **Annotate key frames** (manual)
2. **Propagate with SAM 2** (automatic)
3. **Human refinement** on errors
4. **Re-train SAM 2** on refined data
5. **Repeat** for more videos

**Result:** SA-V dataset (largest video segmentation dataset)

### SAM 3 Training

**Open-Vocabulary Training:**
1. **Image-text pairs:** Web-scale datasets
2. **Concept vocabulary:** 270K unique concepts from captions
3. **Automatic annotation:** SAM 3 generates masks for concepts
4. **Human verification:** Quality check on samples

**Result:** SA-Co benchmark (270K concepts)

---

## Code Examples

### Example 1: Point-Based Segmentation (SAM 1)

```python
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

# Load image
image = plt.imread("image.jpg")
predictor.set_image(image)

# Multiple points (foreground + background)
input_points = np.array([
    [500, 375],  # Foreground point
    [300, 200]   # Background point
])
input_labels = np.array([1, 0])  # 1=fg, 0=bg

# Predict
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False  # Single best mask
)

# Visualize
plt.imshow(image)
plt.imshow(masks[0], alpha=0.5, cmap='jet')
plt.show()
```

### Example 2: Box-Based Segmentation (SAM 1)

```python
# Bounding box prompt
input_box = np.array([100, 100, 500, 400])  # [x1, y1, x2, y2]

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=True  # Get 3 candidates
)

# Show all 3 candidates
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (mask, score) in enumerate(zip(masks, scores)):
    axes[i].imshow(image)
    axes[i].imshow(mask, alpha=0.5)
    axes[i].set_title(f"Mask {i+1}, Score: {score:.3f}")
    axes[i].axis('off')
plt.show()
```

### Example 3: Automatic Everything Mode (SAM 1)

```python
from segment_anything import SamAutomaticMaskGenerator

# Configure mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # Grid density
    pred_iou_thresh=0.86,          # Quality threshold
    stability_score_thresh=0.92,   # Stability threshold
    crop_n_layers=1,               # Multi-crop layers
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100       # Filter tiny masks
)

# Generate all masks
masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")

# Masks sorted by area (largest first)
for mask in masks[:5]:
    print(f"Area: {mask['area']}, IoU: {mask['predicted_iou']:.3f}")
```

### Example 4: Video Tracking (SAM 2)

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Build predictor
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Load video
video_dir = "./videos/my_video"  # Directory of frames

with torch.inference_mode():
    # Initialize
    state = predictor.init_state(video_path=video_dir)

    # Add prompt on first frame (frame 0)
    frame_idx = 0
    obj_id = 1

    # Click on object to track
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=[[300, 400]],  # (x, y) click
        labels=[1]  # Foreground
    )

    # Propagate through video
    video_segments = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        video_segments[frame_idx] = {
            out_obj_id: masks[i]
            for i, out_obj_id in enumerate(obj_ids)
        }

    # Save masks
    for frame_idx, obj_masks in video_segments.items():
        for obj_id, mask in obj_masks.items():
            # Save or visualize mask
            pass
```

### Example 5: Text-Based Segmentation (SAM 3)

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Build model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("street_scene.jpg")
state = processor.set_image(image)

# Text prompt
output = processor.set_text_prompt(
    state=state,
    prompt="car"  # Find all cars
)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print(f"Found {len(masks)} cars")

# Visualize
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(image)
for mask, box, score in zip(masks, boxes, scores):
    if score > 0.5:  # Confidence threshold
        plt.imshow(mask, alpha=0.3)
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='red', linewidth=2
        ))
plt.title(f"Found {len(masks)} cars")
plt.show()
```

### Example 6: Fine-Tuning SAM

```python
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

# Load pre-trained SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

# Freeze image encoder
for param in sam.image_encoder.parameters():
    param.requires_grad = False

# Fine-tune mask decoder only
for param in sam.mask_decoder.parameters():
    param.requires_grad = True

# Optimizer
optimizer = torch.optim.AdamW(
    sam.mask_decoder.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# Training loop
sam.train()
for epoch in range(10):
    for batch in dataloader:
        images = batch['images'].cuda()
        gt_masks = batch['masks'].cuda()
        points = batch['points'].cuda()
        labels = batch['labels'].cuda()

        # Forward
        with torch.no_grad():
            image_embeddings = sam.image_encoder(images)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None
        )

        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            low_res_masks, gt_masks
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Applications

### 1. Medical Imaging

**Use Cases:**
- Tumor segmentation
- Organ delineation
- Cell counting in microscopy
- X-ray/CT/MRI analysis

**Example:**
```python
# Medical image segmentation
medical_image = load_medical_scan("ct_scan.nii.gz")

# SAM for organ segmentation
predictor.set_image(medical_image)

# Doctor clicks on organ
organ_mask, _, _ = predictor.predict(
    point_coords=[[doctor_click_x, doctor_click_y]],
    point_labels=[1]
)

# Quantify organ volume
organ_volume = compute_volume(organ_mask, voxel_spacing)
print(f"Organ volume: {organ_volume} cm³")
```

**Research:**
- **MedSAM:** Fine-tuned SAM for medical imaging
- **SAM-Med2D:** Medical-specific SAM variant
- Papers: arXiv search "SAM medical imaging"

### 2. Autonomous Driving

**Use Cases:**
- Lane detection
- Pedestrian segmentation
- Vehicle tracking
- Road sign detection

**Example:**
```python
# Autonomous driving perception
from sam2 import build_sam2_video_predictor

# Load dashcam video
predictor = build_sam2_video_predictor(...)
state = predictor.init_state("dashcam_video.mp4")

# Detect pedestrian on frame 0
predictor.add_new_points(
    state, frame_idx=0, obj_id=1,
    points=[[pedestrian_x, pedestrian_y]], labels=[1]
)

# Track through video (for collision prediction)
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    pedestrian_mask = masks[0]
    # Update trajectory prediction
```

### 3. Agriculture

**Use Cases:**
- Crop health monitoring
- Plant counting
- Disease detection
- Yield estimation

**Example:**
```python
# SAM 3 for plant detection
from sam3 import build_sam3_image_model

model = build_sam3_image_model()
processor = Sam3Processor(model)

# Aerial farm image
farm_image = Image.open("drone_farm.jpg")
state = processor.set_image(farm_image)

# Detect all tomato plants
output = processor.set_text_prompt(state, "tomato plant")
plant_masks = output["masks"]

print(f"Detected {len(plant_masks)} tomato plants")

# Estimate yield
total_area = sum(mask.sum() for mask in plant_masks)
estimated_yield = total_area * YIELD_PER_PIXEL
print(f"Estimated yield: {estimated_yield} kg")
```

### 4. Robotics

**Use Cases:**
- Object grasping
- Scene understanding
- Human-robot interaction
- Navigation

**Example:**
```python
# Robot grasping with SAM
robot_camera_image = robot.get_camera_image()

# Segment target object
predictor.set_image(robot_camera_image)
masks, _, _ = predictor.predict(
    point_coords=[[target_x, target_y]],
    point_labels=[1]
)

# Compute grasp pose
grasp_pose = compute_grasp_from_mask(masks[0])
robot.execute_grasp(grasp_pose)
```

### 5. Content Creation

**Use Cases:**
- Background removal
- Object extraction for compositing
- Video editing (rotoscoping)
- AR/VR asset creation

**Example:**
```python
# Remove background
image = load_image("portrait.jpg")
predictor.set_image(image)

# Click on person
person_mask, _, _ = predictor.predict(
    point_coords=[[center_x, center_y]],
    point_labels=[1]
)

# Extract person (transparent background)
person_rgba = np.dstack([image, person_mask * 255])
save_image(person_rgba, "person_no_bg.png")
```

### 6. Satellite Imagery

**Use Cases:**
- Building detection
- Road extraction
- Forest monitoring
- Disaster assessment

**Example:**
```python
# SAM 3 for building detection
satellite_image = load_satellite_image("city.tif")
state = processor.set_image(satellite_image)

# Detect all buildings
output = processor.set_text_prompt(state, "building")
building_masks = output["masks"]

# Count buildings
print(f"Detected {len(building_masks)} buildings")

# Compute urban density
urban_area = sum(mask.sum() for mask in building_masks)
urban_density = urban_area / total_image_area
print(f"Urban density: {urban_density * 100:.2f}%")
```

### 7. E-Commerce

**Use Cases:**
- Product image segmentation
- Virtual try-on
- 3D model generation (with SAM 3D)
- Background replacement

**Example:**
```python
# Product segmentation for e-commerce
product_image = load_image("product_photo.jpg")

# Automatic segmentation (no prompts)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(product_image)

# Largest mask is usually the product
product_mask = max(masks, key=lambda x: x['area'])['segmentation']

# Replace background
new_background = load_image("studio_background.jpg")
composite = product_mask[..., None] * product_image + \
            (1 - product_mask[..., None]) * new_background

save_image(composite, "product_studio.jpg")
```

---

## Performance Benchmarks

### SAM 1 Benchmarks

**Zero-Shot Transfer (23 datasets):**

| Dataset Type | SAM (zero-shot) | Fully Supervised |
|-------------|-----------------|------------------|
| **Natural Images** | 85-90% IoU | 90-95% IoU |
| **Medical** | 70-80% IoU | 85-90% IoU |
| **Satellite** | 75-85% IoU | 85-90% IoU |

**Speed (A100 GPU):**
- Image encoding: **~100 ms** (ViT-H)
- Prompt encoding + mask decode: **~10 ms**
- **Total:** ~110 ms per mask (real-time)

**Model Size vs Accuracy:**

| Model | Params | Size | Speed (ms) | IoU (avg) |
|-------|--------|------|-----------|-----------|
| ViT-H | 636M | 2.4 GB | 110 | **0.88** |
| ViT-L | 308M | 1.2 GB | 80 | 0.86 |
| ViT-B | 91M | 375 MB | 50 | 0.83 |

### SAM 2 Benchmarks

**Image Segmentation:**
- **6× faster** than SAM 1
- **Accuracy:** Slightly better than SAM 1 on most benchmarks

**Video Segmentation:**

| Benchmark | SAM 2 | Prior SOTA | Improvement |
|-----------|-------|-----------|-------------|
| **DAVIS** | 82.5% J&F | 75.0% J&F | **+7.5%** |
| **YouTube-VOS** | 76.4% J&F | 70.2% J&F | **+6.2%** |

**Speed (Video):**
- **44 FPS** on A100 GPU (real-time)
- **3× fewer user interactions** needed vs prior methods

### SAM 3 Benchmarks

**SA-Co/Gold (270K concepts):**
- SAM 3: **54.1 cgF1**
- Human: **72.8 cgF1**
- **SAM 3 achieves 75-80% of human performance**

**LVIS (Instance Segmentation):**
- SAM 3: **37.2 cgF1**
- Prior SOTA (open-vocab): ~30 cgF1

**Video (SA-V):**
- **30.3 cgF1** on SA-V test
- **50.8 cgF1** on YT-Temporal-1B test

---

## Research Papers

### Core SAM Papers

**SAM 1:**
```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```
🔗 https://arxiv.org/abs/2304.02643

**SAM 2:**
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```
🔗 https://arxiv.org/abs/2408.00714

**SAM 3:**
- Paper link: TBD (check https://ai.meta.com/research/)

### Survey Papers

**SAM Survey:**
```bibtex
@article{zhang2023survey,
  title={A Survey on Segment Anything Model (SAM): Vision Foundation Model Meets Prompt Engineering},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.06211},
  year={2023}
}
```
🔗 https://arxiv.org/abs/2306.06211

**SAM to SAM 2 Comparison:**
🔗 https://arxiv.org/abs/2408.06305

### Application Papers

**Medical Imaging:**
- "Unleashing the Potential of SAM for Medical Adaptation"
- arXiv: https://arxiv.org/abs/2403.18271

**Remote Sensing:**
- "SAM for Remote Sensing Applications: From Zero to One Shot"
- arXiv: https://arxiv.org/abs/2306.16623

**3D Segmentation:**
- "SAM3D: Segment Anything in 3D Scenes"
- GitHub: https://github.com/Pointcept/SegmentAnything3D

---

## Resources

### Official Resources

| Resource | Link |
|----------|------|
| **SAM GitHub** | https://github.com/facebookresearch/segment-anything |
| **SAM 2 GitHub** | https://github.com/facebookresearch/sam2 |
| **SAM 3 GitHub** | https://github.com/facebookresearch/sam3 |
| **Meta AI Research** | https://ai.meta.com/research/ |
| **SA-1B Dataset** | https://ai.meta.com/datasets/segment-anything/ |
| **Playground Demo** | https://www.aidemos.meta.com/segment-anything |
| **SAM 2 Demo** | https://sam2.metademolab.com/demo |

### Community Resources

**Documentation:**
- Ultralytics SAM Docs: https://docs.ultralytics.com/models/sam/
- HuggingFace SAM: https://huggingface.co/docs/transformers/model_doc/sam
- Encord SAM Guide: https://encord.com/blog/segment-anything-model-explained/

**Tutorials:**
- Roboflow SAM Tutorial: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- Encord Fine-tuning: https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/

**Papers With Code:**
- SAM Papers: https://paperswithcode.com/method/sam
- SA-1B Dataset: https://paperswithcode.com/dataset/sa-1b

### Tools & Integrations

**Ultralytics YOLOv8 + SAM:**
```bash
pip install ultralytics
```
```python
from ultralytics import SAM

model = SAM('sam_b.pt')
results = model('image.jpg')
```

**Grounding DINO + SAM:**
- Text → Bounding Box (Grounding DINO)
- Bounding Box → Mask (SAM)
- GitHub: https://github.com/IDEA-Research/Grounded-Segment-Anything

**Label Studio + SAM:**
- Interactive annotation tool
- GitHub: https://github.com/HumanSignal/label-studio

### Pre-trained Checkpoints

**SAM 1:**
- ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

**SAM 2:**
- Available on GitHub: https://github.com/facebookresearch/sam2

**SAM 3:**
- Available on HuggingFace: https://huggingface.co/facebook/sam3

---

## Conclusion

The **Segment Anything Model (SAM) family** represents a paradigm shift in computer vision:

### Key Achievements

✅ **Foundation Model:** First promptable segmentation model
✅ **Zero-Shot:** Generalizes to unseen domains without retraining
✅ **Scale:** Trained on 1.1B masks (SA-1B dataset)
✅ **Speed:** Real-time inference (SAM 2: 44 FPS)
✅ **Versatility:** Images + Videos (SAM 2) + Text prompts (SAM 3)

### Evolution Timeline

- **April 2023:** SAM 1 - Promptable image segmentation
- **August 2024:** SAM 2 - Video segmentation with streaming memory
- **November 2025:** SAM 3 - Open-vocabulary text-based segmentation

### Impact

**Research:**
- 700,000+ downloads
- 1,000+ papers citing SAM
- New research direction: foundation models for segmentation

**Industry:**
- Medical imaging (tumor detection, organ segmentation)
- Autonomous driving (perception systems)
- E-commerce (product segmentation)
- Content creation (background removal, video editing)
- Robotics (scene understanding, grasping)

### Future Directions

**Potential improvements:**
- Higher resolution outputs
- Faster inference (mobile deployment)
- 3D segmentation (point clouds, meshes)
- Multi-modal fusion (LiDAR + RGB)
- Interactive refinement tools

**Open Questions:**
- How to best combine SAM with LLMs?
- Can SAM enable new applications in AR/VR?
- What's the limit of zero-shot generalization?

---

**The SAM family has fundamentally changed how we approach segmentation - from task-specific models to general-purpose foundation models that "segment anything"!**

---

**Last Updated:** 2025-11-20
**Status:** Complete research study covering SAM 1, 2, and 3
**Next:** Monitor for new SAM releases and applications!
