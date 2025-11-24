# Visual Tokenizer Implementation

**Category**: Codebase
**File**: `ovis/model/modeling_ovis.py:36-189`

## Class: VisualTokenizer

### __init__()

Initializes:
- SigLIP 2 NaViT encoder
- Visual head (projection layer)
- Image preprocessing

### smart_resize() (lines 59-98)

```python
def smart_resize(self, height, width, factor=28,
                 min_pixels=448*448, max_pixels=1792*1792):
    """
    Resize to preserve aspect ratio within pixel budget

    Returns: (new_height, new_width)
    """
```

**Algorithm**:
1. Calculate current pixels
2. Scale if outside range
3. Round to multiples of factor
4. Ensure within bounds

### preprocess()

```python
def preprocess(self, image, min_pixels, max_pixels):
    """
    Preprocess image for model

    Steps:
    1. smart_resize
    2. Convert to tensor
    3. Normalize
    """
```

### forward()

```python
def forward(self, pixel_values, grid_thw):
    """
    Complete tokenization pipeline

    Returns: probability distributions
    """
    # 1. ViT encoding
    features = self.vit(pixel_values, grid_thw)

    # 2. Visual head
    logits = self.visual_head(features)

    # 3. Softmax
    return F.softmax(logits, dim=-1)
```
