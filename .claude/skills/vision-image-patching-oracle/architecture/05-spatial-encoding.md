# Spatial Encoding

**Position embeddings and spatial schemas for organizing visual tokens**

## Overview

Spatial encoding provides the language model with crucial information about where visual tokens came from in the original image, enabling proper spatial reasoning and multi-region understanding.

## Why Spatial Encoding Matters

### The Position Problem

**Scenario**: Image divided into 6 slices (3 columns × 2 rows)

**Without spatial encoding**:
- LLM receives: `[slice_1_tokens, slice_2_tokens, ..., slice_6_tokens]`
- LLM doesn't know: Which slice is top-left? Which is bottom-right?
- Result: Cannot answer "What's in the top-right corner?"

**With spatial encoding**:
- LLM receives: `[(pos=0,0, tokens), (pos=0,1, tokens), ...]`
- LLM knows: Spatial relationships between slices
- Result: Accurate spatial reasoning

### Real-World Example

**Query**: "Describe the text in the upper-left corner"

**Image**: 1920×1080 divided into 3 slices horizontally

**Without spatial encoding**:
```
[slice_left_tokens | slice_middle_tokens | slice_right_tokens]
```
LLM doesn't know which is "upper-left" → random guess

**With spatial encoding**:
```
[(x=0, y=0, slice_left_tokens) | (x=1, y=0, slice_middle) | (x=2, y=0, slice_right)]
```
LLM knows x=0 is leftmost → correct answer

## Core Spatial Encoding Approaches

### 1. Position Embeddings

**Principle**: Add learned vectors encoding spatial position

#### Absolute Position Embeddings

**From [source-documents/18_Vision Transformer Wikipedia](../source-documents/18_Vision transformer - Wikipedia.md)**:

**Original ViT approach**:
```python
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # Learnable position embeddings for each patch position
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

    def forward(self, patch_tokens):
        # patch_tokens: (batch, num_patches, embed_dim)
        return patch_tokens + self.pos_embedding
```

**Characteristics**:
- **Fixed grid size**: Works for 14×14 patch grid (196 patches)
- **Learned**: Position information learned during training
- **Simple**: Direct addition to token embeddings

**Limitation**: Requires interpolation for different resolutions

#### Relative Position Embeddings

**Principle**: Encode relative distances between tokens

**From [source-documents/02_Vision Transformers and CNN Variants Survey](../source-documents/02_A survey of the Vision Transformers and their CNN-Transformer based Variants - arXiv.md)**:

```python
def relative_position_bias(query_pos, key_pos):
    """
    Compute relative position bias for attention

    Args:
        query_pos: (num_queries, 2) - (row, col) positions
        key_pos: (num_keys, 2) - (row, col) positions

    Returns:
        bias: (num_queries, num_keys) - attention bias
    """
    # Compute relative positions
    rel_pos = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)  # (Q, K, 2)

    # Convert to bias indices
    rel_row = rel_pos[:, :, 0]  # Vertical distance
    rel_col = rel_pos[:, :, 1]  # Horizontal distance

    # Look up learned bias
    bias = position_bias_table[rel_row, rel_col]

    return bias
```

**Benefits**:
- **Flexible**: Works with any grid size
- **Relative reasoning**: Models spatial relationships directly
- **Better extrapolation**: Generalizes to new resolutions

#### 2D Sinusoidal Position Encoding

**Extension of Transformer positional encoding to 2D**:

```python
def sinusoidal_2d_encoding(height, width, embed_dim):
    """
    Generate 2D sinusoidal position encodings

    Args:
        height, width: Grid dimensions
        embed_dim: Embedding dimension

    Returns:
        pos_encoding: (height, width, embed_dim)
    """
    # Create position grids
    y_pos = torch.arange(height).unsqueeze(1).expand(height, width)
    x_pos = torch.arange(width).unsqueeze(0).expand(height, width)

    # Frequency bands
    freq_bands = torch.exp(torch.arange(0, embed_dim, 2) *
                          -(math.log(10000.0) / embed_dim))

    # Sinusoidal encoding
    y_embed = torch.zeros(height, width, embed_dim // 2)
    x_embed = torch.zeros(height, width, embed_dim // 2)

    for i, freq in enumerate(freq_bands):
        y_embed[:, :, i] = torch.sin(y_pos * freq)
        x_embed[:, :, i] = torch.sin(x_pos * freq)

    # Alternate y and x dimensions
    pos_encoding = torch.zeros(height, width, embed_dim)
    pos_encoding[:, :, 0::2] = y_embed
    pos_encoding[:, :, 1::2] = x_embed

    return pos_encoding
```

**Characteristics**:
- **Training-free**: No learned parameters
- **Continuous**: Supports any resolution
- **Interpretable**: Clear frequency structure

### 2. Spatial Schema (LLaVA-UHD)

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

**Concept**: Organize slice tokens in a spatial grid that reflects image layout

#### Grid-Based Organization

**For 3×2 slice grid** (3 columns, 2 rows):

```python
def create_spatial_schema(slices, grid_shape):
    """
    Organize slice tokens with spatial structure

    Args:
        slices: List of token sequences for each slice
        grid_shape: (num_rows, num_cols)

    Returns:
        organized_tokens: Spatially-organized token sequence
    """
    num_rows, num_cols = grid_shape

    # Add row/column position tokens
    organized_tokens = []

    for row in range(num_rows):
        for col in range(num_cols):
            slice_idx = row * num_cols + col
            slice_tokens = slices[slice_idx]

            # Create position token
            pos_token = create_position_token(row, col, num_rows, num_cols)

            # Prepend position token to slice
            organized_tokens.append(pos_token)
            organized_tokens.extend(slice_tokens)

    return torch.cat(organized_tokens)
```

**Position token format**:
```
<slice|row={r}|col={c}|total_rows={R}|total_cols={C}>
```

**Example sequence** for 2×2 grid:
```
<slice|0|0|2|2> [slice_00_tokens]
<slice|0|1|2|2> [slice_01_tokens]
<slice|1|0|2|2> [slice_10_tokens]
<slice|1|1|2|2> [slice_11_tokens]
```

#### Benefits of Spatial Schema

1. **Explicit positioning**: LLM directly sees grid coordinates
2. **Relative reasoning**: Easy to compute "adjacent to" relationships
3. **Flexible**: Works for any grid configuration
4. **Interpretable**: Clear semantic meaning

### 3. RoPE (Rotary Position Embedding)

**From [source-documents/05_Circle-RoPE](../source-documents/05_Circle-RoPE_ Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models - arXiv.md)**:

**Principle**: Encode positions via rotation in complex space

**Circle-RoPE innovation**: Separate encoding for X and Y dimensions

```python
def circle_rope_2d(positions, dim):
    """
    Apply Circle-RoPE to 2D positions

    Args:
        positions: (num_tokens, 2) - (row, col) positions
        dim: Embedding dimension

    Returns:
        position_encoding: Complex rotations for attention
    """
    # Separate frequency bands for X and Y
    theta_x = 10000 ** (-2 * torch.arange(dim // 4) / dim)
    theta_y = 10000 ** (-2 * torch.arange(dim // 4) / dim)

    # Extract X, Y positions
    x_pos = positions[:, 1]  # Column
    y_pos = positions[:, 0]  # Row

    # Compute rotation angles
    angles_x = x_pos.unsqueeze(1) * theta_x.unsqueeze(0)
    angles_y = y_pos.unsqueeze(1) * theta_y.unsqueeze(0)

    # Create complex rotations
    rope_x = torch.polar(torch.ones_like(angles_x), angles_x)
    rope_y = torch.polar(torch.ones_like(angles_y), angles_y)

    # Combine X and Y encodings
    rope_2d = torch.cat([rope_x, rope_y], dim=1)

    return rope_2d
```

**Benefits**:
- **Decoupled dimensions**: X and Y encoded independently
- **Long-range**: Better for large images
- **Rotation-based**: Geometric interpretation

## Advanced Spatial Encoding

### Multi-Scale Position Encoding

**For hierarchical image processing**:

```python
def multiscale_position_encoding(patches, scales=[1, 2, 4]):
    """
    Encode positions at multiple scales

    Args:
        patches: Image patches
        scales: List of downsampling factors

    Returns:
        Hierarchical position encodings
    """
    encodings = []

    for scale in scales:
        # Downsample position grid
        h_scaled = h // scale
        w_scaled = w // scale

        # Generate position encoding at this scale
        pos_enc = sinusoidal_2d_encoding(h_scaled, w_scaled, embed_dim)

        # Upsample back to original resolution
        pos_enc_upsampled = F.interpolate(pos_enc, size=(h, w))

        encodings.append(pos_enc_upsampled)

    # Concatenate multi-scale encodings
    return torch.cat(encodings, dim=-1)
```

**Use case**: Encode both local position and global context

### Aspect Ratio Aware Encoding

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

**Challenge**: Wide images (16:9) vs tall images (9:16) need different encodings

**Solution**: Normalize positions by aspect ratio

```python
def aspect_ratio_normalized_encoding(positions, aspect_ratio):
    """
    Normalize positions by aspect ratio

    Args:
        positions: (num_tokens, 2) - (row, col) in image
        aspect_ratio: width / height

    Returns:
        normalized_positions: Scale-invariant positions
    """
    rows, cols = positions[:, 0], positions[:, 1]

    # Normalize so both dimensions span [0, 1]
    if aspect_ratio > 1:  # Wide image
        normalized_rows = rows / rows.max()
        normalized_cols = (cols / cols.max()) / aspect_ratio
    else:  # Tall image
        normalized_rows = (rows / rows.max()) * aspect_ratio
        normalized_cols = cols / cols.max()

    return torch.stack([normalized_rows, normalized_cols], dim=1)
```

**Benefit**: Position encodings consistent across aspect ratios

### Slice Boundary Encoding

**For multi-slice images**:

```python
def encode_slice_boundaries(slice_positions):
    """
    Add special encoding for slice boundaries

    Args:
        slice_positions: (num_slices, 4) - (x_min, y_min, x_max, y_max)

    Returns:
        boundary_tokens: Special tokens marking slice edges
    """
    boundary_tokens = []

    for i, (x_min, y_min, x_max, y_max) in enumerate(slice_positions):
        # Encode slice bounding box
        bbox_token = create_bbox_token(x_min, y_min, x_max, y_max)

        # Encode neighboring slices
        neighbors = find_adjacent_slices(i, slice_positions)
        neighbor_token = create_neighbor_token(neighbors)

        boundary_tokens.append([bbox_token, neighbor_token])

    return boundary_tokens
```

**Purpose**: Help LLM understand relationships across slice boundaries

## Practical Considerations

### Position Encoding Resolution

**Tradeoff**: Fine-grained vs coarse position information

**Recommendations**:
- **Patch-level** (14×14 grid): Standard, most use cases
- **Super-resolution** (28×28 virtual grid): For very high-resolution
- **Coarse** (7×7 grid): For computational efficiency

### Interpolation for Variable Resolutions

**Problem**: Pretrained position embeddings are fixed-size

**Solution**: Interpolate position embeddings

```python
def interpolate_position_embeddings(pos_embed, old_size, new_size):
    """
    Resize position embeddings for new resolution

    Args:
        pos_embed: (old_size**2, dim) - original embeddings
        old_size: Original grid size (e.g., 14)
        new_size: New grid size (e.g., 24)

    Returns:
        resized_embed: (new_size**2, dim) - resized embeddings
    """
    # Reshape to 2D grid
    pos_embed_2d = pos_embed.reshape(old_size, old_size, -1)

    # Interpolate using bilinear
    pos_embed_resized = F.interpolate(
        pos_embed_2d.permute(2, 0, 1).unsqueeze(0),
        size=(new_size, new_size),
        mode='bilinear'
    )

    # Reshape back
    return pos_embed_resized.squeeze(0).permute(1, 2, 0).reshape(-1, dim)
```

**From [source-documents/02_Vision Transformers Survey](../source-documents/02_A survey of the Vision Transformers and their CNN-Transformer based Variants - arXiv.md)**: Interpolation works well for modest resolution changes (up to 2×)

### Combining Multiple Encoding Schemes

**Hybrid approach**:

```python
def combined_spatial_encoding(patches, positions):
    """
    Combine multiple spatial encoding methods

    Args:
        patches: Visual tokens
        positions: (num_tokens, 2) - grid positions

    Returns:
        encoded_tokens: Tokens with combined position information
    """
    # 1. Absolute position embeddings (learned)
    abs_pos_embed = learned_position_embedding(positions)

    # 2. Sinusoidal encoding (training-free)
    sin_pos_embed = sinusoidal_2d_encoding(positions)

    # 3. Relative position bias (for attention)
    rel_pos_bias = compute_relative_bias(positions)

    # Combine encodings
    tokens_with_pos = patches + abs_pos_embed + sin_pos_embed

    return tokens_with_pos, rel_pos_bias
```

**Benefits**: Robustness and flexibility

## Spatial Reasoning Tasks

### Task Types Requiring Spatial Encoding

1. **Directional queries**:
   - "What's in the top-right corner?"
   - "What's to the left of the red car?"

2. **Relative positioning**:
   - "Is the cat above or below the dog?"
   - "Which object is closest to the center?"

3. **Multi-region understanding**:
   - "Compare the left and right halves of the image"
   - "What's the relationship between objects in different slices?"

4. **Document layout**:
   - "Read the text in the second column"
   - "What's the caption for the figure on page 3?"

### Performance Impact

**From LLaVA-UHD experiments**:

**With spatial schema**:
- Spatial reasoning: 82.3% accuracy
- Multi-slice questions: 76.8% accuracy

**Without spatial schema**:
- Spatial reasoning: 51.2% accuracy (near random)
- Multi-slice questions: 42.1% accuracy

**Insight**: Spatial encoding is critical for spatial reasoning

## Design Guidelines

### Choosing Position Encoding

**Factors to consider**:
1. **Resolution flexibility**: Need variable resolutions? → Sinusoidal or RoPE
2. **Training budget**: Limited training? → Sinusoidal (training-free)
3. **Performance priority**: Best accuracy? → Learned absolute + relative bias
4. **Aspect ratio variety**: Wide range? → Circle-RoPE or normalized encoding

### Implementation Checklist

✅ **Position information**: Each token must have position metadata
✅ **Spatial relationships**: LLM can infer adjacency and relative positions
✅ **Slice boundaries**: Clear demarcation between image slices
✅ **Aspect ratio handling**: Consistent encoding across shapes
✅ **Resolution adaptivity**: Works for multiple resolutions

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md) - Spatial schema design
- [05_Circle-RoPE](../source-documents/05_Circle-RoPE_ Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models - arXiv.md) - 2D rotary embeddings
- [18_Vision Transformer Wikipedia](../source-documents/18_Vision transformer - Wikipedia.md) - ViT position embeddings
- [02_Vision Transformers Survey](../source-documents/02_A survey of the Vision Transformers and their CNN-Transformer based Variants - arXiv.md) - Position encoding variants

## Related Documents

- [01-patch-fundamentals.md](01-patch-fundamentals.md) - Basic patching concepts
- [03-native-resolution.md](03-native-resolution.md) - Spatial schema in native resolution processing
- [../models/02-llava-uhd.md](../models/02-llava-uhd.md) - Spatial schema implementation example
- [../concepts/00-image-tokenization.md](../concepts/00-image-tokenization.md) - Token generation and positioning
