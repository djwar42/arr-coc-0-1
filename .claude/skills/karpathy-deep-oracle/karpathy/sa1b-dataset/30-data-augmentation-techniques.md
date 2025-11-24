# Data Augmentation for SA-1B Training

## Overview

Data augmentation improves SAM's robustness to scale, rotation, and photometric variations. Key techniques: **scale jitter, horizontal flip, color jitter**.

## SAM Augmentation Pipeline

```python
# From SAM training recipe
augmentation = {
    "horizontal_flip": True,         # 50% probability
    "scale_jitter": [0.1, 2.0],     # Random scale 0.1-2.0×
    "crop_size": 1024,               # Always 1024×1024 after aug
}
```

## Common Augmentations

**Geometric:**
- Horizontal flip: 50% probability
- Random scale: 0.1-2.0× (extreme range for diverse scales)
- Random crop: Extract 1024×1024 patch
- Rotation: Usually avoided (not in SAM recipe)

**Photometric:**
- Color jitter: Brightness ±10%, contrast ±10%
- Gaussian blur: σ=0.5-2.0
- Random grayscale: 10% probability

## PyTorch Implementation

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=1.9, p=1.0),  # 0.1-2.0×
    A.RandomCrop(height=1024, width=1024),
    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
], additional_targets={'mask': 'mask'})
```

## ARR-COC Application

**Document layout augmentation:**
```python
# For document spatial grounding training
doc_augment = A.Compose([
    A.RandomScale(scale_limit=0.3, p=0.8),  # 0.7-1.3× for documents
    A.Rotate(limit=2, p=0.5),  # Small rotation for scanned docs
    A.RandomBrightnessContrast(p=0.5),  # Lighting variations
])
```

**Sources**: SAM Paper Appendix, Albumentations Documentation
