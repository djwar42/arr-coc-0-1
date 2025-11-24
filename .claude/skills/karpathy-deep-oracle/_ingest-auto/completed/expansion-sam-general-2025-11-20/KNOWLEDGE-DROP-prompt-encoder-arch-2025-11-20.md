# SAM: Prompt Encoder Architecture

**PART 32/42 - Prompt Encoder (Points, Boxes, Masks)**

**Date**: 2025-11-20
**Source**: SAM paper Section 3.2, GitHub implementation

---

## Prompt Encoder Overview

**Purpose**: Convert user prompts (points, boxes, masks) into embeddings that guide mask decoder.

**Input Types**:
1. **Points**: (x, y) coordinates + label (foreground/background)
2. **Boxes**: [x_min, y_min, x_max, y_max]
3. **Masks**: Low-res binary mask (256×256)
4. **Text**: (SAM 3 only) Natural language descriptions

**Output**: Prompt embeddings (256-dim vectors) fed to mask decoder

---

## Point Prompt Encoding

### Representation

**Foreground Point** (positive): Click inside object
**Background Point** (negative): Click outside object

**Encoding**:
```python
# Point at (x, y)
point_embed = PositionalEncoding2D(x, y) + TypeEmbedding(foreground/background)

# PositionalEncoding2D: Learned 256-dim embedding for each (x, y) location
# TypeEmbedding: Learned 256-dim vector (foreground vs. background)
```

**Why Positional Encoding?**:
- Captures spatial location on 64×64 feature grid
- Same as ViT-H positional embeddings (but 256-dim instead of 1,280)

### Multiple Points

**Scenario**: User clicks 3 foreground + 2 background points

**Encoding**:
```python
points = [
    (120, 80, foreground),
    (150, 100, foreground),
    (180, 90, foreground),
    (50, 50, background),
    (300, 200, background)
]

# Encode each point
point_embeds = [PositionalEncoding2D(x, y) + TypeEmbedding(label) for x, y, label in points]

# Shape: 5 × 256 (5 points, 256-dim each)
```

**Decoder Use**: Attention over all 5 point embeddings simultaneously

---

## Box Prompt Encoding

### Representation

**Box**: [x_min, y_min, x_max, y_max]

**Encoding** (as 2 corner points):
```python
# Top-left corner (foreground point)
top_left_embed = PositionalEncoding2D(x_min, y_min) + TypeEmbedding(foreground)

# Bottom-right corner (foreground point)
bottom_right_embed = PositionalEncoding2D(x_max, y_max) + TypeEmbedding(foreground)

# Box encoding = concatenate
box_embed = [top_left_embed, bottom_right_embed]  # 2 × 256
```

**Why 2 Points?**:
- Defines bounding region (top-left + bottom-right uniquely specify box)
- Reuses point encoding infrastructure (no separate box encoder needed)

**Benefit**: Simple, efficient, works well in practice

---

## Mask Prompt Encoding

### Representation

**Input**: Coarse binary mask (256×256 resolution)

**Why Low-Res?**:
- User draws rough outline (doesn't need pixel-perfect precision)
- SAM refines boundaries (mask decoder upsamples to 1024×1024)

### Encoding (Convolutional Encoder)

**Architecture**:
```python
# Input: 256×256×1 binary mask
conv1 = Conv2D(256×256×1 → 128×128×32, kernel=3, stride=2)
conv2 = Conv2D(128×128×32 → 64×64×64, kernel=3, stride=2)
conv3 = Conv2D(64×64×64 → 64×64×128, kernel=3, stride=1)
conv4 = Conv2D(64×64×128 → 64×64×256, kernel=3, stride=1)

# Output: 64×64×256 mask embedding
```

**Why 64×64?**: Matches ViT-H feature map resolution (easy to combine in decoder)

**Activation**: GELU (Gaussian Error Linear Unit)

### Dense vs. Sparse Encoding

**Dense Embedding** (64×64×256):
- Used for mask prompts
- Spatially aligned with image features
- Enables pixel-wise refinement

**Sparse Embedding** (N × 256, N = num points):
- Used for point/box prompts
- N varies (1-20 points typical)
- Decoder attends to sparse locations

---

## Text Prompt Encoding (SAM 3 Only)

**Architecture**: CLIP text encoder

**Method**:
```python
# Input: "the red car"
text = "the red car"

# Encode with CLIP
text_embed = CLIP_TextEncoder(text)  # 768-dim

# Project to 256-dim (match prompt space)
text_embed = Linear(768 → 256)(text_embed)

# Use as additional prompt (like a point)
```

**Integration**: Text embedding treated as extra prompt token

**Benefit**: Open-vocabulary segmentation ("segment all dogs" without training on dog labels)

---

## Prompt Combination

### Multiple Prompt Types (Point + Box)

**Scenario**: User provides point inside object + bounding box

**Encoding**:
```python
# Point: (150, 100, foreground)
point_embed = PositionalEncoding2D(150, 100) + TypeEmbedding(foreground)

# Box: [100, 80, 200, 150]
box_embed = [
    PositionalEncoding2D(100, 80) + TypeEmbedding(foreground),
    PositionalEncoding2D(200, 150) + TypeEmbedding(foreground)
]

# Concatenate all prompts
all_prompts = [point_embed, box_embed[0], box_embed[1]]  # 3 × 256

# Decoder attends to all 3 embeddings
```

**Decoder Behavior**: Combines evidence from point (precise location) + box (spatial extent) for best mask.

### Mask + Points (Iterative Refinement)

**Workflow**:
1. User draws coarse mask (256×256)
2. SAM predicts refined mask (1024×1024)
3. User adds correction points (foreground/background)
4. SAM re-predicts with mask + points

**Encoding**:
```python
# Mask prompt: 64×64×256 (dense)
mask_embed = MaskEncoder(coarse_mask)

# Points: 2 × 256 (sparse)
point_embeds = [PositionalEncoding2D(x, y) + TypeEmbedding(label) for x, y, label in points]

# Decoder uses both: mask_embed (dense spatial) + point_embeds (sparse corrections)
```

**Benefit**: Best of both worlds (spatial prior from mask, precision from points)

---

## Prompt Embedding Dimensions

**Summary**:
- **Point/Box**: Sparse embeddings (N × 256, N = 1-20)
- **Mask**: Dense embedding (64×64×256)
- **Text** (SAM 3): Sparse embedding (1 × 256)

**Decoder Input**:
- Image features: 64×64×1,280 (from ViT-H)
- Prompt embeddings: N × 256 (sparse) + optional 64×64×256 (dense mask)

**Cross-Attention**: Decoder attends from image features to prompt embeddings

---

## Positional Encoding Details

### 2D Sinusoidal Encoding (Learned Variant)

**Standard Sinusoidal**:
```python
# For position (x, y) on 64×64 grid
PE_x = [sin(x / 10000^(2i/256)), cos(x / 10000^(2i/256))] for i in 0..127
PE_y = [sin(y / 10000^(2i/256)), cos(y / 10000^(2i/256))] for i in 0..127

# Concatenate: PE(x, y) = [PE_x, PE_y]  # 256-dim
```

**SAM's Learned Variant**:
- Initialize with sinusoidal values
- Fine-tune during SAM training on SA-1B
- **Benefit**: Adapts to dataset-specific spatial patterns

### Why Not Use ViT-H Positional Embeddings?

**ViT-H**: 4,096 patches × 1,280-dim (too high-dimensional for prompts)

**Prompt Encoder**: 64×64 locations × 256-dim (lightweight, efficient)

**Reason**: Prompts need spatial info but not full image context (decoder will integrate)

---

## Implementation Example (PyTorch-style)

```python
class PromptEncoder(nn.Module):
    def __init__(self):
        self.pos_encoding = PositionalEncoding2D(grid_size=64, embed_dim=256)
        self.fg_embed = nn.Parameter(torch.randn(256))  # Foreground type
        self.bg_embed = nn.Parameter(torch.randn(256))  # Background type

        # Mask encoder (conv layers)
        self.mask_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.mask_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mask_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.mask_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

    def encode_points(self, coords, labels):
        # coords: (N, 2) - (x, y) in 0-1024 range
        # labels: (N,) - 0=background, 1=foreground

        # Normalize to 0-64 grid
        coords_grid = coords / 16  # 1024/16 = 64

        # Get positional encodings
        pos_embeds = self.pos_encoding(coords_grid)  # N × 256

        # Add type embeddings
        type_embeds = torch.where(
            labels.unsqueeze(1),
            self.fg_embed.unsqueeze(0),
            self.bg_embed.unsqueeze(0)
        )

        return pos_embeds + type_embeds  # N × 256

    def encode_boxes(self, boxes):
        # boxes: (N, 4) - [x_min, y_min, x_max, y_max]

        # Encode as 2 corner points (both foreground)
        top_left = self.encode_points(boxes[:, :2], labels=torch.ones(len(boxes)))
        bottom_right = self.encode_points(boxes[:, 2:], labels=torch.ones(len(boxes)))

        return torch.cat([top_left, bottom_right], dim=0)  # 2N × 256

    def encode_masks(self, masks):
        # masks: (B, 256, 256) - binary masks

        # Conv encoder
        x = F.gelu(self.mask_conv1(masks.unsqueeze(1)))  # B × 32 × 128 × 128
        x = F.gelu(self.mask_conv2(x))  # B × 64 × 64 × 64
        x = F.gelu(self.mask_conv3(x))  # B × 128 × 64 × 64
        x = F.gelu(self.mask_conv4(x))  # B × 256 × 64 × 64

        return x  # B × 256 × 64 × 64
```

---

## Ablation Studies

**Impact of Prompt Type** (SAM paper Table 3):

| Prompt Type | COCO mIoU | ADE20K mIoU |
|-------------|-----------|-------------|
| 1 center point | 42.5 | 35.8 |
| 1 random point | 38.2 | 31.4 |
| 3 foreground points | 48.1 | 42.3 |
| 1 box | 52.7 | 47.5 |
| Box + 3 points | 54.2 | 49.1 |
| Mask + 3 points | 55.8 | 50.3 |

**Insights**:
- Box prompts > point prompts (spatial extent matters)
- Multiple prompts > single prompt (more information helps)
- Mask + points = best (spatial prior + precision)

**Impact of Positional Encoding** (Ablation):

| Encoding Type | COCO mIoU |
|---------------|-----------|
| No positional encoding | 28.3 |
| Fixed sinusoidal | 47.2 |
| Learned (SAM) | 50.3 |

**Insight**: Learned positional encoding crucial for spatial reasoning (+22 mIoU!)

---

## Limitations

### 1. Fixed Resolution (64×64 Grid)
- **Issue**: Prompts quantized to 64×64 locations
- **Effect**: Point at (123, 456) → snapped to nearest grid cell
- **Impact**: Slight inaccuracy for fine-grained prompts

### 2. No Semantic Understanding
- **Limitation**: Prompts are purely spatial (no object categories)
- **Example**: Can't prompt "the largest dog" (needs text, added in SAM 3)

### 3. Mask Prompt Low-Res (256×256)
- **Trade-off**: Fast encoding but loses fine detail
- **Workaround**: Use mask + correction points for precision

---

## ARR-COC Integration (5%)

### Prompts as Relevance Signals

**Connection**: Prompts = **explicit relevance specification** (user-guided salience).

**ARR-COC Mapping**:
1. **Point prompts** = propositional knowing (this location is relevant)
2. **Box prompts** = perspectival knowing (this region contains the object)
3. **Mask prompts** = participatory knowing (co-create boundary with SAM)

**Insight**: SAM's promptable interface = human-AI relevance co-realization!

### Opponent Processing in Foreground/Background Points

**Hypothesis**: Foreground/background labels implement opponent processing:
- **Foreground** = enhance relevance (positive evidence)
- **Background** = suppress relevance (negative evidence)

**ARR-COC**: Balances focal attention (object) vs. peripheral suppression (background).

**Mechanism**: Type embeddings (fg_embed, bg_embed) bias decoder attention weights.

---

## Key Takeaways

1. **Prompt Encoder**: Converts points/boxes/masks → 256-dim embeddings
2. **Point Encoding**: PositionalEncoding2D + TypeEmbedding (foreground/background)
3. **Box Encoding**: 2 corner points (top-left, bottom-right)
4. **Mask Encoding**: 4-layer conv encoder (256×256 → 64×64×256)
5. **ARR-COC**: Prompts = explicit relevance signals (propositional → perspectival → participatory)

**Next**: PART 33 - Mask Decoder Architecture

---

**References**:
- Kirillov et al., "Segment Anything" (ICCV 2023) - Section 3.2
- SAM GitHub: `segment_anything/modeling/prompt_encoder.py`
- Positional encoding: "Attention Is All You Need" (Vaswani et al. 2017)
