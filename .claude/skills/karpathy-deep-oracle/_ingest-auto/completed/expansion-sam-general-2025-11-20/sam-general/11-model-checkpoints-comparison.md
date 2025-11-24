# SAM Model Checkpoints Comparison

## Section 1: Checkpoint Overview

### Introduction to SAM Model Variants

The Segment Anything Model (SAM) is released in three checkpoint sizes, each built on a different
Vision Transformer (ViT) backbone. These variants offer different trade-offs between
segmentation quality, inference speed, and computational requirements.

### The Three SAM Checkpoints

From [Segment Anything paper](https://arxiv.org/abs/2304.02643) (arXiv:2304.02643):

| Model | Parameters | Checkpoint File | Download Size | Primary Use Case |
|-------|------------|-----------------|---------------|------------------|
| **ViT-H (Huge)** | 636M | `sam_vit_h_4b8939.pth` | 2.4 GB | Best quality, production |
| **ViT-L (Large)** | 308M | `sam_vit_l_0b3195.pth` | 1.2 GB | Balanced performance |
| **ViT-B (Base)** | 91M | `sam_vit_b_01ec64.pth` | 375 MB | Fast inference, edge deployment |

### Download Links

Official Meta AI download URLs:
```bash
# ViT-H (Huge) - Best quality
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (Large) - Balanced
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (Base) - Fastest
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Checkpoint Architecture Foundation

All three checkpoints share the same overall SAM architecture:

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 562-598:

```
Input Image (1024x1024)
    |
ViT Image Encoder (varies by checkpoint)
    |
Prompt Encoder (same across all)
    |
Mask Decoder (same across all)
    |
Output Masks
```

The **image encoder** is the primary differentiator between checkpoints - it determines
parameter count, inference speed, and segmentation quality.

### Key Architectural Constants

Across all checkpoints:
- **Input resolution**: 1024 x 1024 pixels
- **Output embedding**: 64 x 64 x 256
- **Prompt encoder**: Identical architecture
- **Mask decoder**: 2 transformer decoder blocks
- **Patch size**: 16 x 16 pixels

---

## Section 2: ViT-H (Huge) - Maximum Quality

### Overview

ViT-H is the flagship SAM checkpoint, offering the highest segmentation quality at the
cost of increased computational requirements. This is the model used in the original
SAM paper benchmarks.

### Technical Specifications

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 569-585:

```python
# ViT-H Configuration
num_layers = 32
hidden_dim = 1280
num_heads = 16
mlp_ratio = 4.0
patch_size = 16

# Total parameters: ~636M
```

### Architecture Details

**Image Encoder:**
- 32 transformer layers
- 1280-dimensional embeddings
- 16 attention heads per layer
- Windowed attention (14x14) with 4 global attention blocks
- MAE (Masked Autoencoder) pre-trained initialization

**Processing Flow:**
```
Image (1024x1024x3)
    | Patch Embedding
Patches (64x64x256)
    | Transformer Blocks (32 layers)
    |-- Windowed Self-Attention (14x14)
    |-- Global Attention (every 8th layer)
    |-- MLP (4x expansion)
    |
Feature Map (64x64x256)
```

### Performance Characteristics

From [Roboflow SAM Tutorial](https://blog.roboflow.com/how-to-use-segment-anything-model-sam/):

**Inference Speed (A100 GPU):**
- Image encoding: ~100 ms
- Prompt encoding + mask decode: ~10 ms
- **Total per mask**: ~110 ms

**Memory Requirements:**
- Model weights: 2.4 GB
- GPU VRAM: ~8-12 GB (inference)
- Fine-tuning: 24+ GB GPU

### Quality Benchmarks

From [Medical Imaging Study](https://www.sciencedirect.com/science/article/abs/pii/S1361841523003213):

- SAM with ViT-H showed **better overall performance** than ViT-B
- Achieves highest IoU scores across diverse segmentation benchmarks
- Best performance on challenging edge cases and fine details

**Zero-Shot Performance:**
- Natural images: 85-90% IoU
- Medical imaging: 70-80% IoU
- Satellite imagery: 75-85% IoU

### When to Use ViT-H

**Recommended for:**
- Production deployments where quality is paramount
- Medical imaging and scientific applications
- Detailed segmentation requiring precise boundaries
- Research benchmarking
- Batch processing where speed is not critical

**Not ideal for:**
- Real-time applications
- Edge/mobile deployment
- Resource-constrained environments
- Interactive applications requiring instant feedback

### Loading ViT-H

```python
from segment_anything import sam_model_registry, SamPredictor

# Load ViT-H model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

predictor = SamPredictor(sam)
```

---

## Section 3: ViT-L (Large) - Balanced Performance

### Overview

ViT-L offers a balanced trade-off between segmentation quality and computational
efficiency. It provides quality close to ViT-H with significantly reduced resource
requirements.

### Technical Specifications

**ViT-L Configuration:**
```python
num_layers = 24
hidden_dim = 1024
num_heads = 16
mlp_ratio = 4.0
patch_size = 16

# Total parameters: ~308M
```

### Architecture Details

**Image Encoder:**
- 24 transformer layers (vs 32 in ViT-H)
- 1024-dimensional embeddings (vs 1280 in ViT-H)
- 16 attention heads per layer
- Same windowed + global attention pattern as ViT-H

### Performance Characteristics

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 1319-1325:

**Inference Speed:**
- **~80 ms** per mask (vs ~110 ms for ViT-H)
- **27% faster** than ViT-H

**Memory Requirements:**
- Model weights: 1.2 GB (50% smaller than ViT-H)
- GPU VRAM: ~6-8 GB (inference)
- Fine-tuning: 16+ GB GPU

### Quality Benchmarks

From [IEEE Performance Study](https://ieeexplore.ieee.org/iel7/6287639/10380310/10493013.pdf):

- **SAM (ViT-L) demonstrated superior performance** in 192 test cases
- Only marginal quality reduction compared to ViT-H
- Significantly better than ViT-B across benchmarks

**Performance Comparison:**
| Metric | ViT-H | ViT-L | Difference |
|--------|-------|-------|------------|
| Average IoU | 0.88 | 0.86 | -2.3% |
| Speed (ms) | 110 | 80 | +37.5% faster |
| Parameters | 636M | 308M | -51.6% |

### When to Use ViT-L

**Recommended for:**
- Production deployments with balanced requirements
- Applications needing good quality with faster inference
- Server-based deployments with moderate GPU resources
- Interactive applications with reasonable latency tolerance
- Fine-tuning experiments (lower memory requirements)

**Not ideal for:**
- Applications requiring maximum possible quality
- Real-time or edge deployment
- Mobile applications

### Loading ViT-L

```python
from segment_anything import sam_model_registry, SamPredictor

# Load ViT-L model
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
sam.to(device="cuda")

predictor = SamPredictor(sam)
```

---

## Section 4: ViT-B (Base) - Speed Optimized

### Overview

ViT-B is the smallest and fastest SAM checkpoint, designed for applications where
inference speed and resource efficiency are prioritized over maximum segmentation quality.

### Technical Specifications

**ViT-B Configuration:**
```python
num_layers = 12
hidden_dim = 768
num_heads = 12
mlp_ratio = 4.0
patch_size = 16

# Total parameters: ~91M
```

### Architecture Details

**Image Encoder:**
- 12 transformer layers (vs 32 in ViT-H, 24 in ViT-L)
- 768-dimensional embeddings
- 12 attention heads per layer
- Same architectural pattern, scaled down

### Performance Characteristics

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 1319-1325:

**Inference Speed:**
- **~50 ms** per mask
- **2.2x faster** than ViT-H
- **1.6x faster** than ViT-L

**Memory Requirements:**
- Model weights: 375 MB (6.4x smaller than ViT-H)
- GPU VRAM: ~3-4 GB (inference)
- Fine-tuning: 8+ GB GPU

### Quality Benchmarks

From [Roboflow SAM Tutorial](https://blog.roboflow.com/how-to-use-segment-anything-model-sam/):

- ViT-H improves substantially over ViT-B
- Still achieves competitive segmentation quality for many tasks
- Acceptable for applications where speed matters more than precision

**Performance Comparison:**
| Metric | ViT-H | ViT-B | Difference |
|--------|-------|-------|------------|
| Average IoU | 0.88 | 0.83 | -5.7% |
| Speed (ms) | 110 | 50 | +120% faster |
| Parameters | 636M | 91M | -85.7% |

### When to Use ViT-B

**Recommended for:**
- Edge deployment and embedded systems
- Real-time or interactive applications
- Mobile and resource-constrained devices
- Rapid prototyping and experimentation
- Applications where speed > quality
- Fine-tuning with limited GPU memory

**Not ideal for:**
- Medical imaging requiring high precision
- Production systems where quality is critical
- Detailed boundary segmentation
- Benchmarking against state-of-the-art

### Loading ViT-B

```python
from segment_anything import sam_model_registry, SamPredictor

# Load ViT-B model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device="cuda")

predictor = SamPredictor(sam)
```

### ViT-B Fine-Tuning Advantage

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 1033-1096:

ViT-B is often preferred for fine-tuning due to lower memory requirements:

```python
import torch
from segment_anything import sam_model_registry

# Load pre-trained ViT-B
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
```

---

## Section 5: Performance Benchmarks

### Comprehensive Model Comparison

From [SAM Study](../source-documents/SAM_STUDY_GENERAL.md) lines 1302-1358:

| Model | Params | Size | Speed (ms) | IoU (avg) | Use Case |
|-------|--------|------|-----------|-----------|----------|
| **ViT-H** | 636M | 2.4 GB | 110 | **0.88** | Best quality |
| **ViT-L** | 308M | 1.2 GB | 80 | 0.86 | Balanced |
| **ViT-B** | 91M | 375 MB | 50 | 0.83 | Fast/Edge |

### SAM vs YOLO Comparison

From [Ultralytics Docs](https://docs.ultralytics.com/models/sam/):

| Model | Size (MB) | Parameters (M) | Speed (CPU ms/im) |
|-------|-----------|----------------|-------------------|
| SAM-b (ViT-B) | 375 | 93.7 | 49,401 |
| MobileSAM | 40.7 | 10.1 | 25,381 |
| FastSAM-s | 23.7 | 11.8 | 55.9 |
| YOLOv8n-seg | **6.7** | **3.4** | **24.5** |
| YOLO11n-seg | **5.9** | **2.9** | **30.1** |

**Key Insight**: YOLO models are dramatically smaller and faster than SAM variants.
YOLO11n-seg is **13.2x smaller** and **864x faster** than SAM-b on CPU.

### Zero-Shot Transfer Performance

SAM zero-shot performance across different domains:

| Domain | ViT-H | ViT-L | ViT-B | Fully Supervised |
|--------|-------|-------|-------|------------------|
| Natural Images | 85-90% | 83-88% | 80-85% | 90-95% |
| Medical | 70-80% | 68-78% | 65-75% | 85-90% |
| Satellite | 75-85% | 73-83% | 70-80% | 85-90% |

### GPU Memory Usage

**Inference Memory:**
| Model | GPU VRAM | Batch Size 1 | Batch Size 8 |
|-------|----------|--------------|--------------|
| ViT-H | 8-12 GB | ~4 GB | ~16 GB |
| ViT-L | 6-8 GB | ~3 GB | ~12 GB |
| ViT-B | 3-4 GB | ~2 GB | ~6 GB |

**Fine-Tuning Memory (Mask Decoder Only):**
| Model | Minimum VRAM | Recommended |
|-------|--------------|-------------|
| ViT-H | 24 GB | 32+ GB |
| ViT-L | 16 GB | 24+ GB |
| ViT-B | 8 GB | 12+ GB |

### Throughput Comparison

**Images per Second (A100 GPU):**
| Model | Single Image | Batched (8) |
|-------|--------------|-------------|
| ViT-H | ~9 img/s | ~40 img/s |
| ViT-L | ~12 img/s | ~55 img/s |
| ViT-B | ~20 img/s | ~90 img/s |

### Quality vs Speed Trade-off

```
Quality
  ^
  |   * ViT-H (0.88 IoU)
  |
  |      * ViT-L (0.86 IoU)
  |
  |          * ViT-B (0.83 IoU)
  |
  +---------------------------------> Speed
        50ms    80ms    110ms
```

---

## Section 6: Selection Guidelines

### Decision Matrix

| Requirement | Recommended Model | Reason |
|-------------|-------------------|--------|
| Maximum quality | ViT-H | Highest IoU, best fine details |
| Production balanced | ViT-L | Good quality, reasonable speed |
| Real-time | ViT-B | Fastest inference |
| Edge/Mobile | ViT-B | Smallest footprint |
| Fine-tuning (limited GPU) | ViT-B | Lowest memory requirements |
| Research benchmarks | ViT-H | Standard for comparisons |
| Medical imaging | ViT-H | Critical accuracy needs |
| Interactive annotation | ViT-L | Balance speed & quality |

### Application-Based Recommendations

**Medical Imaging & Scientific:**
- **Primary**: ViT-H
- **Alternative**: ViT-L for faster annotation
- **Reasoning**: Precision is critical for diagnostics

**Autonomous Driving:**
- **Primary**: ViT-B or ViT-L
- **Alternative**: Consider MobileSAM or FastSAM
- **Reasoning**: Real-time processing requirements

**Content Creation / Background Removal:**
- **Primary**: ViT-L
- **Alternative**: ViT-B for batch processing
- **Reasoning**: Balance between quality and throughput

**Dataset Annotation:**
- **Primary**: ViT-L for interactive
- **Alternative**: ViT-B for large-scale automation
- **Reasoning**: Speed matters for annotation workflows

**Robotics / Embedded:**
- **Primary**: ViT-B
- **Alternative**: MobileSAM or FastSAM
- **Reasoning**: Resource constraints

### Hardware-Based Selection

**High-End GPU (24+ GB VRAM):**
- Use ViT-H for all tasks
- Can fine-tune any checkpoint

**Mid-Range GPU (8-16 GB VRAM):**
- ViT-L for inference
- ViT-B for fine-tuning

**Consumer GPU (4-8 GB VRAM):**
- ViT-B recommended
- Consider MobileSAM

**CPU Only:**
- ViT-B strongly recommended
- Expect very slow inference (~50 seconds)

### Cost-Benefit Analysis

**ViT-H: Premium Quality**
- Storage cost: 2.4 GB
- Compute cost: High
- Quality benefit: Maximum
- Best for: Production, research

**ViT-L: Best Value**
- Storage cost: 1.2 GB
- Compute cost: Moderate
- Quality benefit: High (only 2% below ViT-H)
- Best for: Most applications

**ViT-B: Efficiency Champion**
- Storage cost: 375 MB
- Compute cost: Low
- Quality benefit: Good
- Best for: Speed-critical, edge

---

## Section 7: ARR-COC Integration

### Checkpoint Selection for ARR-COC

For ARR-COC training workflows, the checkpoint selection depends on the specific use case:

**Auto-Annotation Pipeline:**
```python
# For generating training data with SAM
# Use ViT-L for balance of quality and speed

from segment_anything import sam_model_registry

# Load appropriate checkpoint based on hardware
def select_sam_checkpoint(available_vram_gb: float) -> str:
    """Select SAM checkpoint based on available GPU memory."""
    if available_vram_gb >= 12:
        return "vit_h"  # Maximum quality
    elif available_vram_gb >= 6:
        return "vit_l"  # Balanced
    else:
        return "vit_b"  # Resource-constrained

# Load model
checkpoint_type = select_sam_checkpoint(get_available_vram())
sam = sam_model_registry[checkpoint_type](
    checkpoint=f"sam_{checkpoint_type}_*.pth"
)
```

### Integration with ARR-COC Training

**Pre-processing with SAM:**
```python
# Generate segmentation masks for training data
from segment_anything import SamAutomaticMaskGenerator

def generate_training_masks(
    images: List[np.ndarray],
    sam_checkpoint: str = "vit_l"
) -> List[Dict]:
    """Generate masks for ARR-COC training data."""
    sam = sam_model_registry[sam_checkpoint](checkpoint=f"sam_{sam_checkpoint}_*.pth")
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92
    )

    all_masks = []
    for image in images:
        masks = mask_generator.generate(image)
        all_masks.append(masks)

    return all_masks
```

### Performance Considerations for ARR-COC

**Batch Processing:**
- ViT-H: ~9 img/s (best quality training data)
- ViT-L: ~12 img/s (recommended for most ARR-COC workflows)
- ViT-B: ~20 img/s (large-scale data generation)

**Memory Management:**
```python
# Clear GPU memory between batches
import torch

def process_batch_with_sam(batch, sam_model):
    results = []
    for item in batch:
        result = sam_model(item)
        results.append(result)

    # Clear cache after processing
    torch.cuda.empty_cache()
    return results
```

### Recommended Configurations

**Development/Testing:**
- Checkpoint: ViT-B
- Reason: Fast iteration, lower resource usage

**Production Data Generation:**
- Checkpoint: ViT-L
- Reason: Good quality with reasonable throughput

**High-Quality Annotations:**
- Checkpoint: ViT-H
- Reason: Maximum mask quality for critical training data

---

## Sources

**Source Documents:**
- [SAM_STUDY_GENERAL.md](../source-documents/SAM_STUDY_GENERAL.md) - Comprehensive SAM research study

**Web Research:**
- [Segment Anything paper](https://arxiv.org/abs/2304.02643) - arXiv:2304.02643 (accessed 2025-11-20)
- [Roboflow SAM Tutorial](https://blog.roboflow.com/how-to-use-segment-anything-model-sam/) (accessed 2025-11-20)
- [Ultralytics SAM Documentation](https://docs.ultralytics.com/models/sam/) (accessed 2025-11-20)
- [Medical Imaging SAM Study](https://www.sciencedirect.com/science/article/abs/pii/S1361841523003213) (accessed 2025-11-20)
- [IEEE Performance Analysis](https://ieeexplore.ieee.org/iel7/6287639/10380310/10493013.pdf) (accessed 2025-11-20)

**GitHub Repositories:**
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official SAM implementation

**Additional References:**
- [HQ-SAM Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f828e38160f31935cfe9f67503ad17c-Paper-Conference.pdf) - NeurIPS 2023
