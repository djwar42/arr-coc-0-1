# Loss Functions for Segmentation Training

## Overview

Loss functions guide segmentation model training. SAM uses **Focal Loss + Dice Loss** (20:1 ratio) + IoU prediction loss.

**Key challenges:**
- Extreme class imbalance (background >> foreground)
- Fine boundary precision
- Mask quality at multiple granularities
- Confidence calibration

## SAM Combined Loss

```python
# SAM uses: 20 * focal_loss + 1 * dice_loss + iou_loss

class SAMFullLoss(nn.Module):
    def forward(self, pred_masks, pred_iou, target_masks):
        # Focal loss (handles class imbalance + hard examples)
        focal = sigmoid_focal_loss(pred_masks, target_masks, alpha=0.25, gamma=2.0)

        # Dice loss (region overlap)
        dice = dice_loss(pred_masks, target_masks)

        # IoU prediction (confidence calibration)
        actual_iou = compute_iou(pred_masks, target_masks)
        iou_loss = F.mse_loss(pred_iou, actual_iou)

        # Combined
        total = 20.0 * focal + 1.0 * dice + iou_loss
        return total
```

## Loss Components

**1. Focal Loss** (α=0.25, γ=2.0)
- Handles class imbalance (background >> foreground)
- Focuses on hard examples via `(1-p_t)^γ` weighting
- Used by RetinaNet, SAM

**2. Dice Loss**
- Optimizes region overlap directly
- Formula: `1 - (2*intersection / (pred + target))`
- Less sensitive to small objects than BCE

**3. IoU Prediction Loss**
- Trains model to estimate its own mask quality
- Used for ranking multiple mask outputs
- MSE between predicted and actual IoU

## Why 20:1 Ratio?

- Focal loss values: ~0.01-0.1 (smaller range)
- Dice loss values: ~0.1-0.5 (larger range)
- 20:1 balances their contributions
- Focal handles pixel-wise, Dice handles regional

## ARR-COC Application

**Spatial relevance loss:**
```python
class SpatialRelevanceLoss(nn.Module):
    def forward(self, pred_masks, pred_iou, text_alignment):
        # Mask quality (from SAM)
        mask_loss = 20 * focal + 1 * dice

        # Confidence
        iou_loss = F.mse_loss(pred_iou, actual_iou)

        # Text-region alignment
        alignment_loss = contrastive_loss(text_embeds, region_embeds)

        return mask_loss + iou_loss + alignment_loss
```

**Sources**: SAM Paper, Focal Loss (RetinaNet), Dice Loss (V-Net)
