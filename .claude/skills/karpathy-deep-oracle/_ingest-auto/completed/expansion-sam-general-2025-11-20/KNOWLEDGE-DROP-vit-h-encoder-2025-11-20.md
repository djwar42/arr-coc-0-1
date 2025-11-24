# SAM: ViT-H Image Encoder Architecture

**PART 31/42 - Vision Transformer Huge (ViT-H) Encoder**

**Date**: 2025-11-20
**Source**: SAM paper Section 3.1, ViT architecture (Dosovitskiy et al. 2021)

---

## ViT-H Overview

**Full Name**: Vision Transformer - Huge variant

**Purpose**: Extract rich visual features from input images for mask prediction.

**Key Stats**:
- **Parameters**: 630 million (largest component of SAM)
- **Depth**: 32 transformer blocks
- **Hidden dim**: 1,280
- **Attention heads**: 16
- **Patch size**: 16×16 pixels

**Pre-training**: MAE (Masked Autoencoder) on ImageNet-1K

---

## Architecture Components

### 1. Patch Embedding

**Input**: RGB image (1024×1024×3)

**Patchification**:
```python
# Divide image into 16×16 patches
num_patches = (1024 / 16) × (1024 / 16) = 4,096 patches

# Each patch = 16×16×3 = 768 values
# Linear projection: 768 → 1,280 (hidden_dim)
```

**Output**: 4,096 patch tokens of dimension 1,280

### 2. Positional Encoding

**Method**: Absolute learned 2D positional embeddings

**Why Learned?**: Better than fixed sinusoidal for high-res images

**Encoding**:
```python
# For each patch at (i, j) in 64×64 grid
pos_embed[i, j] = LearnedEmbedding2D(i, j)  # 1,280-dim vector

# Add to patch tokens
patch_tokens = patch_tokens + pos_embed
```

**Benefit**: Encodes spatial relationships (adjacent patches, distance)

### 3. Transformer Blocks (32 layers)

**Each block**:
```python
# Layer Norm → Multi-Head Attention → Residual
x = x + MultiHeadAttention(LayerNorm(x))

# Layer Norm → MLP → Residual
x = x + MLP(LayerNorm(x))
```

**Multi-Head Attention** (16 heads):
- Query, Key, Value projections: 1,280 → 16 heads × 80 dim
- Attention: Softmax(QK^T / sqrt(80)) × V
- Concatenate heads → Linear projection back to 1,280

**MLP** (4× expansion):
- Linear: 1,280 → 5,120 (GELU activation)
- Linear: 5,120 → 1,280

### 4. Output Features

**After 32 blocks**:
- **Shape**: 4,096 patches × 1,280 dims
- **Resolution**: 64×64 feature map (16× downsampling from 1024×1024)
- **Receptive field**: Global (attention spans entire image)

**Key Property**: Each 64×64 feature encodes context from the entire 1024×1024 image!

---

## MAE Pre-Training

**Method**: Masked Autoencoder (He et al. 2022)

**Training Task**:
1. Mask 75% of patches randomly
2. Encode visible 25% with ViT
3. Decoder predicts masked patches (pixel reconstruction)

**Why MAE?**
- **Self-supervised**: No labels required (learns from images alone)
- **Robust features**: Forces encoder to learn context (can't just memorize)
- **Scales well**: Trained on 1.2M ImageNet images

**Transfer to SAM**:
- Use MAE-pretrained ViT-H as SAM's image encoder
- Fine-tune on SA-1B dataset with mask prediction task
- MAE initialization accelerates convergence (3× faster than random init)

---

## Windowed vs. Global Attention

### SAM ViT-H: Hybrid Approach

**Blocks 1-16**: Window-based attention (local focus)
**Blocks 17-32**: Global attention (full image context)

### Windowed Attention (Blocks 1-16)

**Method**: Divide 64×64 feature map into 8×8 windows → attend within each window

**Window size**: 8×8 patches

**Computation**:
```python
# Full attention: O(4096^2) = 16.7M operations
# Windowed attention: O(64 * 8^2) = 4,096 operations (4,096× cheaper!)
```

**Benefit**: Focus on local textures/edges before global reasoning

### Global Attention (Blocks 17-32)

**Method**: Attend across all 4,096 patches (full receptive field)

**Why**: Integrate context from entire image for semantic understanding

**Example**: To segment a "wheel", model must see the entire "car" (global context)

---

## Why ViT-H vs. ViT-B/ViT-L?

**Model Variants**:
- **ViT-B** (Base): 86M params, 12 layers
- **ViT-L** (Large): 307M params, 24 layers
- **ViT-H** (Huge): 630M params, 32 layers

**SAM ViT-H Performance** (vs. ViT-B baseline):
- **+8.2 mIoU** on COCO instance segmentation
- **+12.1 mIoU** on ADE20K semantic segmentation
- **+6.5% higher** zero-shot transfer accuracy

**Why?**:
1. **More capacity**: 630M params capture richer visual patterns
2. **Deeper reasoning**: 32 layers enable multi-hop inference
3. **Better generalization**: Larger model reduces overfitting on SA-1B

**Cost**:
- **Inference time**: 180ms (ViT-H) vs. 50ms (ViT-B) on A100 GPU
- **Memory**: 2.4GB (ViT-H) vs. 340MB (ViT-B)

**Trade-off**: SAM prioritizes accuracy over speed (foundation model philosophy)

---

## Feature Extraction Workflow

**Step-by-Step**:
```python
# 1. Input image (1024×1024×3)
image = preprocess(raw_image)  # Normalize, resize

# 2. Patchify (4,096 patches of 16×16 pixels)
patches = divide_into_patches(image, patch_size=16)

# 3. Embed patches (4,096 × 1,280)
patch_tokens = linear_projection(patches)

# 4. Add positional encoding
patch_tokens = patch_tokens + pos_embed

# 5. Pass through 32 transformer blocks
features = ViT_H_32_blocks(patch_tokens)  # Still 4,096 × 1,280

# 6. Output feature map (64×64×1,280)
feature_map = features.reshape(64, 64, 1,280)
```

**Result**: 64×64×1,280 feature map fed to prompt encoder + mask decoder

---

## Comparison: ViT vs. CNN Encoders

### ViT-H (SAM)
- **Architecture**: Transformer (self-attention)
- **Receptive field**: Global from layer 1 (attention spans entire image)
- **Strengths**: Long-range dependencies, context integration
- **Weaknesses**: Slower inference, higher memory

### ResNet-101 (Traditional)
- **Architecture**: Convolutional (local filters)
- **Receptive field**: Gradual expansion (limited early layers)
- **Strengths**: Fast, efficient, good for local textures
- **Weaknesses**: Struggles with global context

**SAM Choice**: ViT-H for foundation model (prioritize generalization over speed)

**Alternative**: Lightweight models (SAM-Fast) use distilled ViTs or hybrid CNN-Transformer encoders for speed.

---

## Implementation Details

### Memory Optimization

**Gradient Checkpointing**:
- Save activations only at selected layers (not all 32)
- Recompute intermediate activations during backward pass
- **Trade-off**: 2× slower training, 60% less memory

**Mixed Precision (FP16)**:
- Store weights in FP16, accumulate gradients in FP32
- **Benefit**: 2× faster, 50% less memory, negligible accuracy loss

### Batch Processing

**Patch-Level Batching**:
```python
# Process 4,096 patches in parallel across GPU cores
# Each batch = 256 patches → 16 batches per image
```

**Multi-Image Batching**:
- Standard batch size: 8-16 images per GPU (A100)
- Total patches: 8 images × 4,096 patches = 32,768 patches processed in parallel

---

## Ablation Studies

**Impact of Encoder Depth** (SAM paper Table 5):

| Encoder | Depth | Params | COCO mIoU | ADE20K mIoU |
|---------|-------|--------|-----------|-------------|
| ViT-B | 12 | 86M | 42.1 | 35.3 |
| ViT-L | 24 | 307M | 48.7 | 42.8 |
| ViT-H | 32 | 630M | 50.3 | 47.5 |

**Insight**: Each depth doubling → +6-7 mIoU improvement

**Impact of MAE Pre-Training** (SAM paper Table 6):

| Initialization | COCO mIoU | Zero-Shot Transfer (23 datasets avg) |
|----------------|-----------|--------------------------------------|
| Random | 44.2 | 58.3 |
| ImageNet supervised | 46.8 | 61.7 |
| MAE (ImageNet) | 50.3 | 68.9 |

**Insight**: MAE pre-training crucial for zero-shot generalization (+10.6 mIoU!)

---

## Limitations

### 1. Fixed Input Resolution
- **Constraint**: 1024×1024 pixels (due to positional embeddings)
- **Workaround**: Resize larger images (may lose detail) or crop-based processing

### 2. Computational Cost
- **180ms per image** (A100 GPU)
- **Barrier**: Real-time video segmentation (30 FPS = 33ms per frame)

### 3. No Temporal Modeling
- **ViT-H**: Processes each frame independently (no memory)
- **Solution**: SAM 2 adds streaming memory for video

---

## ARR-COC Integration (5%)

### ViT-H as Relevance Realization Engine

**Connection**: ViT-H's attention mechanism = **dynamic salience landscape construction**.

**ARR-COC Mapping**:
1. **Patch tokens** = propositional features (what is present)
2. **Attention weights** = perspectival focus (what is relevant)
3. **Global context** = participatory integration (how features relate)

**Insight**: Each transformer block refines relevance (early: local edges, late: global objects).

### Opponent Processing in Multi-Head Attention

**Hypothesis**: 16 attention heads implement opponent processing:
- **Heads 1-8**: Feature detection (edges, textures, colors)
- **Heads 9-16**: Context integration (relationships, scene structure)

**ARR-COC**: Opponent processing balances focal attention (object) vs. peripheral context (scene).

**Future Work**: Visualize attention heads to test this hypothesis (mechanistic interpretability).

---

## Key Takeaways

1. **ViT-H**: 630M params, 32 layers, 64×64×1,280 output features
2. **MAE Pre-Training**: Self-supervised on ImageNet → robust zero-shot transfer
3. **Hybrid Attention**: Windowed (local) + global (context) for efficiency
4. **Trade-off**: Accuracy vs. speed (ViT-H slow but best performance)
5. **ARR-COC**: ViT-H attention = relevance realization (propositional → perspectival → participatory)

**Next**: PART 32 - Prompt Encoder Architecture

---

**References**:
- Kirillov et al., "Segment Anything" (ICCV 2023) - Section 3.1
- Dosovitskiy et al., "An Image is Worth 16×16 Words" (ICLR 2021) - ViT
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022) - MAE
