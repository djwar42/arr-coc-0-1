# NaViT: Native-Resolution Vision Transformer

**Category**: Architecture
**Related**: [00-overview.md](00-overview.md), [02-visual-tokenizer.md](02-visual-tokenizer.md)
**Code**: `ovis/model/vit/modeling_siglip2_navit.py`

## Overview

NaViT (Native Vision Transformer) is Ovis's vision backbone based on **SigLIP 2**, designed to process images at their native resolution without fixed tiling or aspect ratio distortion.

**Key Innovation**: Dynamic grid encoding that adapts to arbitrary image sizes while preserving spatial relationships through RoPE (Rotary Position Embeddings).

## Architecture

### Base Model: SigLIP 2

**SigLIP** (Sigmoid Loss for Language-Image Pre-training):
- Trained on image-text pairs
- 400M examples from web
- Strong zero-shot capabilities
- Better than CLIP on many benchmarks

**SigLIP 2 Configuration**:
```python
{
    "hidden_size": 1152,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "intermediate_size": 4304,
    "patch_size": 14,
    "image_size": 384  # Base training size
}
```

### Native Resolution Support

**Traditional ViT Problem**:
```
Fixed size: 224×224 or 336×336
Wide image (1920×1080) → Resize → Distortion
Tall image (1080×1920) → Resize → Distortion
```

**NaViT Solution**:
```
Native resolution: Preserve aspect ratio
Wide image (1920×1080) → Process at native size
Tall image (1080×1920) → Process at native size
```

**Supported Resolutions**:
- Minimum: 448×448
- Maximum: 1792×1792
- Any aspect ratio within range

### RoPE Integration

**Rotary Position Embeddings** applied in **every ViT block**:

```python
class VisionTransformerBlock(nn.Module):
    def forward(self, x, rope):
        # Self-attention with RoPE
        q, k = self.qkv_proj(x)
        q = rope.apply(q)  # Rotate queries
        k = rope.apply(k)  # Rotate keys
        attn = scaled_dot_product(q, k, v)
        # ... rest of block
```

**Why RoPE?**:
- **Relative positions**: Better than absolute
- **Extrapolation**: Generalizes to unseen sizes
- **Spatial awareness**: Improved object localization

### Grid Encoding (grid_thw)

**Purpose**: Track spatial layout for different resolutions

```python
# For 1024×1024 image:
grid_thw = [1, 73, 73]  # [temporal, height, width]
# 1 = single image (not video)
# 73 = (1024 // 14) = 73 patches per dimension

# For 1792×896 image:
grid_thw = [1, 128, 64]
# 128 = (1792 // 14) patches vertically
# 64 = (896 // 14) patches horizontally
```

**Usage**:
- Passed through pipeline
- Used for RoPE calculations
- Enables variable-size processing

## Implementation Details

### Patch Embedding

```python
# Convert image to patches
patch_size = 14  # 14×14 patches
image = [B, 3, H, W]
patches = unfold(image, kernel=14, stride=14)
# Output: [B, (H//14) × (W//14), 14×14×3]

# Linear projection
patch_embeds = linear(patches.flatten(-2))
# Output: [B, num_patches, hidden_size]
```

### Position Encoding

**RoPE Application**:
```python
def apply_rope(q, k, positions):
    """
    Apply rotary position embeddings

    positions: [batch, seq_len, 2] with (row, col) for each patch
    """
    # Compute rotation matrices
    freqs = compute_freqs(positions)

    # Rotate query and key
    q_rot = rotate(q, freqs)
    k_rot = rotate(k, freqs)

    return q_rot, k_rot

# Used in every attention layer
for layer in vit_layers:
    q, k, v = layer.qkv(x)
    q, k = apply_rope(q, k, grid_positions)
    attn = attention(q, k, v)
```

### Multi-Resolution Processing

**Training** (progressive resolution):
```
Phase P1: 448²-896²   → Basic features
Phase P2: 448²-1792²  → High-res capability
Phase P3-P5: 448²-1792² → Maintain
```

**Inference** (smart resize):
```python
def smart_resize(image, min_pixels=448*448, max_pixels=1792*1792):
    """
    Resize preserving aspect ratio within pixel budget
    """
    h, w = image.size
    pixels = h * w

    if pixels < min_pixels:
        # Scale up
        scale = sqrt(min_pixels / pixels)
        new_h, new_w = int(h * scale), int(w * scale)
    elif pixels > max_pixels:
        # Scale down
        scale = sqrt(max_pixels / pixels)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        # Already in range
        new_h, new_w = h, w

    return image.resize((new_w, new_h))
```

## Advantages

### 1. No Aspect Ratio Distortion

**Traditional**:
```
1920×1080 → 336×336 → Stretched (loses info)
```

**NaViT**:
```
1920×1080 → 1792×896 → Preserved (keeps info)
```

### 2. Better Document Understanding

**Documents often have extreme aspect ratios**:
- A4 page: 2480×3508 (tall)
- Wide spreadsheet: 3000×1000 (wide)

**NaViT handles naturally** without distortion.

### 3. Efficient Token Usage

**Fixed tiling** (standard):
```
Small image (512×512) → Padded to 1024×1024 → Waste tokens
```

**Native resolution**:
```
Small image (512×512) → Process at 448×448 → Efficient
```

## Comparison to Other ViTs

| Feature | CLIP ViT | InternViT | SigLIP NaViT |
|---------|----------|-----------|--------------|
| **Resolution** | Fixed 224/336 | Fixed 448 | Variable 448-1792 |
| **Aspect Ratio** | Square only | Square only | Any ratio |
| **Position Encoding** | Absolute | Learned | RoPE (relative) |
| **Training Data** | 400M pairs | 5.5B | 400M pairs |
| **Extrapolation** | Poor | Moderate | Excellent |

## Performance Characteristics

### Speed

```
Resolution → Patches → Attention Cost
448×448    → 1024    → O(1024²) = ~1M ops
896×896    → 4096    → O(4096²) = ~16M ops (16× slower)
1792×1792  → 16384   → O(16384²) = ~268M ops (268× slower)
```

**Optimization**: Flash Attention reduces to O(N) memory

### Memory

```
448×448:   ~500MB activation memory
896×896:   ~2GB activation memory
1792×1792: ~8GB activation memory
```

**Trade-off**: Higher resolution = better quality but slower/more memory

## Code References

**Main Implementation**: `ovis/model/vit/modeling_siglip2_navit.py`

**Key Classes**:
```python
class SigLIP2NaViT(nn.Module):
    """Native-resolution SigLIP 2 vision transformer"""

    def forward(self, pixel_values, grid_thw):
        # Patch embedding
        x = self.patch_embed(pixel_values)

        # Add position embeddings (RoPE computed on-the-fly)
        positions = compute_grid_positions(grid_thw)

        # Process through transformer blocks with RoPE
        for block in self.blocks:
            x = block(x, positions)

        return x
```

**Configuration**: `ovis/model/vit/configuration_siglip2_navit.py`

## Related Topics

- [00-overview.md](00-overview.md) - System architecture
- [02-visual-tokenizer.md](02-visual-tokenizer.md) - Uses NaViT features
- [../concepts/02-native-resolution.md](../concepts/02-native-resolution.md) - Concept deep dive
- [../concepts/04-rope-positional.md](../concepts/04-rope-positional.md) - RoPE details
