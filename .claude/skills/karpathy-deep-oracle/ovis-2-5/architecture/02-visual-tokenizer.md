# Visual Tokenizer

**Category**: Architecture
**Related**: [01-navit-vision.md](01-navit-vision.md), [03-visual-embedding-table.md](03-visual-embedding-table.md)
**Code**: `ovis/model/modeling_ovis.py:36-189`

## Overview

The Visual Tokenizer (VT) converts continuous visual features from NaViT into **probability distributions** over the visual vocabulary for VET lookup.

**Pipeline**: Image → NaViT → Visual Head → Probabilities → VET

## Components

### 1. ViT Encoder (NaViT)

Processes image at native resolution:
```python
visual_features = navit(pixel_values, grid_thw)
# Output: [batch, num_patches, 1152]
```

### 2. Visual Head

Projects to visual vocabulary size:
```python
class VisualHead(nn.Module):
    def __init__(self, hidden_size=1152, vocab_size=16384):
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.layernorm = nn.LayerNorm(vocab_size)

    def forward(self, x):
        logits = self.linear(x)           # [B, N, 16384]
        logits = self.layernorm(logits)   # Normalize
        probs = F.softmax(logits, dim=-1) # Probabilities
        return probs
```

### 3. Smart Resize Algorithm

**Purpose**: Preserve aspect ratio within pixel budget

```python
def smart_resize(height, width, factor=28, min_pixels=448*448, max_pixels=1792*1792):
    """
    Resize to multiples of factor while preserving aspect ratio

    factor=28: Ensures divisibility by patch_size (14) after /2 downsampling
    """
    # Current pixel count
    pixels = height * width

    # Calculate scale needed
    if pixels > max_pixels:
        scale = sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = sqrt(min_pixels / pixels)
    else:
        scale = 1.0

    # Apply scale
    new_h = int(height * scale)
    new_w = int(width * scale)

    # Round to factor
    new_h = round(new_h / factor) * factor
    new_w = round(new_w / factor) * factor

    # Ensure within bounds
    while new_h * new_w > max_pixels:
        new_h -= factor
    while new_h * new_w < min_pixels:
        new_h += factor

    return new_h, new_w
```

## Complete Forward Pass

```python
class VisualTokenizer(nn.Module):
    def forward(self, pixel_values, grid_thw):
        """
        Convert images to probability distributions

        Args:
            pixel_values: [B, 3, H, W]
            grid_thw: [B, 3] with [t, h, w] grid dimensions

        Returns:
            probabilities: [B, num_patches, vocab_size]
        """
        # 1. Encode through ViT
        visual_features = self.vit(pixel_values, grid_thw)
        # [B, num_patches, 1152]

        # 2. Visual head projection
        logits = self.visual_head(visual_features)
        # [B, num_patches, 16384]

        # 3. Softmax to probabilities
        probabilities = F.softmax(logits, dim=-1)
        # [B, num_patches, 16384]

        return probabilities
```

## Probability Distribution Quality

### Sharp Distribution (Confident)
```python
probs = [0.01, 0.02, 0.90, 0.04, 0.03]
# Model confident: 90% weight on one embedding
```

### Soft Distribution (Uncertain)
```python
probs = [0.18, 0.22, 0.25, 0.20, 0.15]
# Model uncertain: spread across multiple embeddings
```

**Impact**:
- Sharp → More discrete (text-like)
- Soft → More continuous (smooth)
- Model learns optimal sharpness

## Related Topics

- [01-navit-vision.md](01-navit-vision.md) - ViT backbone
- [03-visual-embedding-table.md](03-visual-embedding-table.md) - Uses probabilities
- [../codebase/02-visual-tokenizer-impl.md](../codebase/02-visual-tokenizer-impl.md) - Implementation
