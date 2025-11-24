# SAM: Training Procedure & Loss Functions

**PART 34/42 - Training on SA-1B Dataset**

**Date**: 2025-11-20
**Source**: SAM paper Section 4, Training details

---

## Training Overview

**Dataset**: SA-1B (1.1 billion masks, 11 million images)

**Stages**:
1. **Image encoder pre-training**: MAE on ImageNet (self-supervised)
2. **SAM training**: Full model on SA-1B (supervised)

**Duration**:
- MAE pre-training: ~2 weeks on 128 A100 GPUs
- SAM training: ~90 hours on 256 A100 GPUs

**Total GPU-hours**: ~50,000 A100-hours

---

## Stage 1: MAE Pre-Training (Image Encoder)

### Method

**Masked Autoencoder (MAE)**: Self-supervised pre-training on ImageNet-1K

**Task**:
1. Mask 75% of image patches randomly
2. Encode visible 25% with ViT-H
3. Decoder reconstructs masked patches (pixel-level)

**Loss**: Mean Squared Error (MSE) between reconstructed and original pixels

**Why MAE?**:
- No labels required (learns from raw images)
- Forces encoder to understand context (can't memorize)
- Transfers well to downstream tasks (segmentation)

### Training Hyperparameters

```python
# Optimizer
optimizer = AdamW(lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))

# Learning rate schedule
lr_schedule = CosineAnnealing(warmup_epochs=40, max_epochs=1600)

# Batch size
batch_size = 4096  # Across 128 A100 GPUs (32 per GPU)

# Augmentations
transforms = [
    RandomResizedCrop(224),  # Standard ImageNet size
    RandomHorizontalFlip(),
    ColorJitter(0.4, 0.4, 0.4)
]
```

**Checkpoint**: After MAE, ViT-H encoder has 630M pre-trained parameters

---

## Stage 2: SAM Training (Full Model)

### Training Data (SA-1B)

**Images**: 11 million (diverse domains)
**Masks**: 1.1 billion (100 masks/image average)

**Data Engine Stages**:
- Stage 1 (Assisted-manual): 4.3M masks (120K images)
- Stage 2 (Semi-automatic): 10.2M masks (180K images)
- Stage 3 (Fully automatic): 1.09B masks (10.7M images)

**Training Split**: Use all SA-1B data (no validation set, zero-shot eval on other datasets)

### Training Procedure

**Curriculum**:
1. **Warm-up (5 epochs)**: Train only prompt encoder + mask decoder (freeze ViT-H)
2. **Joint training (90 epochs)**: Train all parameters end-to-end

**Why Warm-up?**:
- MAE-pretrained ViT-H already good → avoid catastrophic forgetting
- Let prompt encoder + decoder catch up first

### Prompt Sampling Strategy

**Random prompt types per training example**:
```python
# Sample prompt type with probability
prompt_type = sample([
    (0.5, "point"),      # 50% point prompts
    (0.3, "box"),        # 30% box prompts
    (0.15, "mask"),      # 15% mask prompts
    (0.05, "no_prompt")  # 5% no prompt (for automatic mask generation)
])

# Sample number of points (if point prompt)
if prompt_type == "point":
    num_points = sample([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
    # 40% single point, 30% two points, etc.

    # Sample foreground/background ratio
    num_fg = random(1, num_points)
    num_bg = num_points - num_fg
```

**Why Random?**: Forces SAM to handle all prompt types (generalization)

---

## Loss Functions

### 1. Focal Loss (Mask Prediction)

**Purpose**: Handle class imbalance (most pixels are background)

**Formula**:
```python
# For each pixel
p_t = p if y == 1 else (1 - p)  # p = predicted prob, y = ground truth label

# Focal loss (down-weights easy examples)
focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)

# Hyperparameters
alpha = 0.25  # Balance positive/negative
gamma = 2.0   # Focus on hard examples
```

**Why Focal Loss?**:
- Standard cross-entropy: Overwhelmed by easy background pixels
- Focal loss: Down-weights easy examples → focuses on hard boundary pixels

**Example**:
- Easy pixel (p=0.99 for background): Loss ≈ 0.001 (negligible)
- Hard pixel (p=0.60 for object edge): Loss ≈ 0.22 (large)

### 2. Dice Loss (Overlap Maximization)

**Purpose**: Directly optimize IoU (overlap between prediction and ground truth)

**Formula**:
```python
# Numerator: 2 * intersection
intersection = sum(pred * gt)  # Element-wise product

# Denominator: sum of predictions + ground truth
union = sum(pred) + sum(gt)

# Dice coefficient (range: 0-1, higher = better)
dice = (2 * intersection + smooth) / (union + smooth)

# Dice loss (minimize)
dice_loss = 1 - dice

# Smooth term avoids division by zero (smooth = 1.0)
```

**Why Dice Loss?**:
- Focal loss: Pixel-level (doesn't directly optimize overlap)
- Dice loss: Directly maximizes IoU (more aligned with evaluation metric)

### 3. IoU Prediction Loss (MSE)

**Purpose**: Train IoU token to predict mask quality

**Formula**:
```python
# Ground truth IoU
gt_iou = intersection(pred_mask, gt_mask) / union(pred_mask, gt_mask)

# Predicted IoU (from IoU token)
pred_iou = MLP(output_tokens[3])  # Range: 0-1

# Mean Squared Error
iou_loss = (pred_iou - gt_iou)^2
```

**Why MSE?**: Simple, effective for regression (predicting continuous IoU values)

### 4. Combined Loss

**Total training loss**:
```python
loss_total = lambda_focal * focal_loss +
             lambda_dice * dice_loss +
             lambda_iou * iou_loss

# Weights (tuned on validation)
lambda_focal = 20.0
lambda_dice = 1.0
lambda_iou = 1.0
```

**Multi-Mask Training**:
```python
# Compute loss for all 3 masks
losses = [loss_total(mask_i, gt_mask) for mask_i in [mask1, mask2, mask3]]

# Backprop only through best-matching mask (min loss)
best_idx = argmin(losses)
loss_final = losses[best_idx]
```

**Why Best-Match?**: Encourages exploration (decoder tries different hypotheses)

---

## Training Hyperparameters

### Optimizer

```python
optimizer = AdamW(
    lr=8e-4,           # Base learning rate
    weight_decay=0.1,  # Regularization
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### Learning Rate Schedule

**Cosine annealing with warm-up**:
```python
# Warm-up (first 250 iterations)
lr = base_lr * (iteration / 250)  # Linear ramp-up

# Cosine decay (after warm-up)
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * iteration / max_iterations))

# Values
base_lr = 8e-4
min_lr = 0 (no lower bound)
```

**Why Cosine?**: Smooth decay → better convergence than step decay

### Batch Size

```python
# Global batch size (across all GPUs)
batch_size = 256 images

# Per-GPU batch size (256 A100 GPUs)
per_gpu_batch_size = 1 image/GPU

# Gradient accumulation (if needed)
accumulation_steps = 4  # Effective batch size = 4 × 256 = 1024
```

**Why Small Per-GPU Batch?**: 1024×1024 images + ViT-H = high memory (10GB per image)

### Data Augmentation

```python
transforms = [
    RandomResizedCrop(1024, scale=(0.8, 1.2)),  # Slight zoom in/out
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # No rotation (preserve spatial structure for points/boxes)
]
```

**Why Minimal Augmentation?**: Prompts are spatially specific (rotation would break point/box coordinates)

---

## Training Efficiency Techniques

### 1. Mixed Precision (FP16)

**Method**: Store weights in FP16, accumulate gradients in FP32

**Implementation**:
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    masks = model(image, prompts)
    loss = compute_loss(masks, gt_masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefit**: 2× faster training, 50% less memory

### 2. Gradient Checkpointing

**Method**: Recompute activations during backward pass instead of storing

**Benefit**: 60% less memory, 30% slower training (net win for large models)

### 3. Distributed Data Parallel (DDP)

**Setup**: 256 A100 GPUs across 32 nodes (8 GPUs per node)

**Synchronization**:
```python
# All-reduce gradients across GPUs after backward pass
dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
gradients = gradients / world_size  # Average
```

**Benefit**: Near-linear scaling (256× speedup with 256 GPUs)

---

## Evaluation During Training

**Metrics** (on held-out SA-1B subset):
- **mIoU**: Mean Intersection over Union (mask overlap)
- **mAP**: Mean Average Precision (detection-style metric)
- **Boundary F1**: Precision/recall on mask boundaries

**Checkpointing**: Save every 10 epochs + best model (highest mIoU)

**Early Stopping**: None (train for full 90 epochs)

---

## Ablation Studies

**Impact of Loss Components** (SAM paper Table 10):

| Loss Components | COCO mIoU |
|-----------------|-----------|
| Focal only | 42.3 |
| Dice only | 44.7 |
| Focal + Dice (SAM) | 50.3 |
| Focal + Dice + IoU | 50.3 (same, IoU helps selection not prediction) |

**Insight**: Combining focal + dice crucial (+5.6 mIoU over focal alone)

**Impact of Multi-Mask Training** (SAM paper Table 11):

| Training Strategy | COCO mIoU | Ambiguity Handling |
|-------------------|-----------|---------------------|
| Single mask (best-match) | 46.2 | Poor |
| 3 masks, backprop all | 48.9 | Good |
| 3 masks, best-match (SAM) | 50.3 | Excellent |

**Insight**: Best-match strategy forces exploration (+1.4 mIoU)

**Impact of Prompt Sampling** (SAM paper Table 12):

| Prompt Distribution | COCO mIoU (Point) | COCO mIoU (Box) |
|---------------------|-------------------|-----------------|
| 100% points | 48.2 | 40.3 |
| 100% boxes | 38.1 | 52.1 |
| Mixed (SAM: 50% point, 30% box, 15% mask, 5% none) | 47.8 | 51.9 |

**Insight**: Mixed prompt training → robust to all prompt types

---

## ARR-COC Integration (5%)

### Training as Relevance Learning

**Connection**: SAM training = **learning to realize relevance from prompts**.

**ARR-COC Mapping**:
1. **Propositional knowing**: Focal loss learns pixel-level boundaries (what is object vs. background)
2. **Perspectival knowing**: Multi-mask training learns multiple valid interpretations (perspectival ambiguity)
3. **Participatory knowing**: Prompt encoder learns to interpret user intent (co-create segmentation)

**Insight**: SAM training iteratively refines relevance realization across all 3 ways of knowing!

### Opponent Processing in Loss Functions

**Hypothesis**: Focal + Dice loss implement opponent processing:
- **Focal loss** = local precision (pixel-level boundaries)
- **Dice loss** = global recall (overall object coverage)

**ARR-COC**: Balances focal attention (boundaries) vs. peripheral integration (object as whole).

---

## Key Takeaways

1. **Training**: MAE pre-training (ViT-H) → SAM training (full model on SA-1B)
2. **Loss**: Focal (class imbalance) + Dice (IoU) + MSE (IoU prediction)
3. **Multi-Mask**: Backprop through best-matching mask (encourages exploration)
4. **Prompt Sampling**: Mixed (50% point, 30% box, 15% mask, 5% none)
5. **ARR-COC**: Training = learning relevance realization (propositional → perspectival → participatory)

**Next**: PART 35 - Deployment & Inference Optimization

---

**References**:
- Kirillov et al., "Segment Anything" (ICCV 2023) - Section 4
- SAM training code: `segment_anything/training/`
- Focal loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- Dice loss: Milletari et al., "V-Net" (3DV 2016)
