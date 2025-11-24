# 2D Positional Encoding in Vision Transformers

## Overview

Two-dimensional positional encoding extends the 1D positional encoding concept from NLP transformers to handle the inherent 2D spatial structure of images. Unlike text sequences which have natural 1D ordering, images require position information along both height and width dimensions.

### Why 2D vs 1D for Images

**The fundamental problem:** Transformers are permutation-invariant without positional information. For images split into patches, this means:
- Patches can be arbitrarily rearranged without affecting output
- Spatial relationships between patches are lost
- Local structure and global layout become meaningless

**1D approach limitations:**
- Treats 2D image as flattened 1D sequence (raster scan order)
- Loses explicit height/width spatial information
- Cannot distinguish between horizontal and vertical neighbors

**2D approach advantages:**
- Explicitly encodes (height, width) position for each patch
- Preserves 2D spatial relationships
- Better represents image structure

From [A 2D Semantic-Aware Position Encoding for Vision Transformers](https://arxiv.org/html/2505.09466v1) (arXiv:2505.09466, accessed 2025-01-31):
- "We propose a semantic-aware approach to position encoding, integrating it with the inherent 2D spatial structure of image inputs"
- Standard 1D encoding fails to capture 2D spatial relationships critical for vision tasks

## Absolute 2D Positional Encodings

### Learned 2D Position Tables

**Concept:** Learn separate embeddings for each spatial position (h, w) during training.

**Implementation:**
```python
# Learned 2D position embedding (PyTorch-style)
num_patches_h = image_height // patch_size  # e.g., 14
num_patches_w = image_width // patch_size   # e.g., 14
embed_dim = 768

# Separate embeddings for each axis
pos_embed_h = nn.Parameter(torch.randn(1, num_patches_h, embed_dim // 2))
pos_embed_w = nn.Parameter(torch.randn(1, num_patches_w, embed_dim // 2))

# Combine: broadcast and concatenate
# pos_embed_h: (1, H, D/2) -> (1, H, 1, D/2) -> (1, H, W, D/2)
# pos_embed_w: (1, W, D/2) -> (1, 1, W, D/2) -> (1, H, W, D/2)
pos_h_broadcast = pos_embed_h.unsqueeze(2).expand(1, num_patches_h, num_patches_w, embed_dim // 2)
pos_w_broadcast = pos_embed_w.unsqueeze(1).expand(1, num_patches_h, num_patches_w, embed_dim // 2)

# Concatenate to get full 2D position encoding
pos_embed_2d = torch.cat([pos_h_broadcast, pos_w_broadcast], dim=-1)
# Shape: (1, H, W, D) -> flatten to (1, H*W, D) for transformer input
```

**Characteristics:**
- **Trainable:** Position encodings updated via backpropagation
- **Flexible:** Can adapt to specific dataset characteristics
- **Memory cost:** O(H × embed_dim/2 + W × embed_dim/2) parameters
- **Fixed resolution:** Cannot extrapolate to different image sizes

From [GitHub: 2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer) (accessed 2025-01-31):
- Learned positional encoding on CIFAR10: **86.52% accuracy**
- Outperforms no position encoding (79.63%) by significant margin
- Comparable to sinusoidal encoding (86.09%)

### Sinusoidal 2D (Absolute)

**Concept:** Extend sinusoidal encoding from NLP to 2D by applying sine/cosine functions separately to height and width dimensions.

**Mathematical formulation:**
```
For position (h, w) and embedding dimension d:

# Split embedding dimension into two halves
d_h = d // 2  # dimensions for height encoding
d_w = d // 2  # dimensions for width encoding

# Height encoding (first half of embedding)
PE_h(h, 2i)   = sin(h / 10000^(2i/d_h))
PE_h(h, 2i+1) = cos(h / 10000^(2i/d_h))

# Width encoding (second half of embedding)
PE_w(w, 2j)   = sin(w / 10000^(2j/d_w))
PE_w(w, 2j+1) = cos(w / 10000^(2j/d_w))

# Final 2D encoding
PE_2D(h, w) = [PE_h(h, 0:d_h), PE_w(w, 0:d_w)]
```

**Implementation:**
```python
def get_2d_sinusoidal_encoding(height, width, embed_dim):
    """Generate 2D sinusoidal position encoding.

    Args:
        height: Number of patches in height dimension
        width: Number of patches in width dimension
        embed_dim: Embedding dimension (must be even)

    Returns:
        Tensor of shape (height, width, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    half_dim = embed_dim // 2

    # Generate frequency scaling
    div_term = torch.exp(torch.arange(0, half_dim, 2).float() *
                        -(math.log(10000.0) / half_dim))

    # Height encoding
    pos_h = torch.arange(height).unsqueeze(1)  # (H, 1)
    pe_h = torch.zeros(height, half_dim)
    pe_h[:, 0::2] = torch.sin(pos_h * div_term)
    pe_h[:, 1::2] = torch.cos(pos_h * div_term)

    # Width encoding
    pos_w = torch.arange(width).unsqueeze(1)  # (W, 1)
    pe_w = torch.zeros(width, half_dim)
    pe_w[:, 0::2] = torch.sin(pos_w * div_term)
    pe_w[:, 1::2] = torch.cos(pos_w * div_term)

    # Combine via broadcasting
    pe_h_broadcast = pe_h.unsqueeze(1).expand(height, width, half_dim)
    pe_w_broadcast = pe_w.unsqueeze(0).expand(height, width, half_dim)

    # Concatenate
    pe_2d = torch.cat([pe_h_broadcast, pe_w_broadcast], dim=-1)

    return pe_2d  # (H, W, D)
```

**Characteristics:**
- **No learned parameters:** Fixed mathematical function
- **Extrapolation:** Can generate encodings for any image size
- **Smooth transitions:** Neighboring positions have similar encodings
- **Frequency spectrum:** Different dimensions capture different spatial scales

From [Position Embeddings for Vision Transformers, Explained](https://medium.com/data-science/position-embeddings-for-vision-transformers-explained-a6f9add341d5) (accessed 2025-01-31):
- Sinusoidal encoding preserves spatial structure visible in visualizations
- Both position information and token content are present in combined representation
- "You can see the structure from the original tokens, as well as the structure in the position embedding!"

### Factorization Strategies

**Height-Width Decomposition:**

Instead of learning a full (H × W) position table, decompose into separate height and width components:

```
# Full 2D table (memory intensive)
pos_embed_full = Parameter(H, W, D)  # H*W*D parameters

# Factorized (memory efficient)
pos_embed_h = Parameter(H, D/2)       # H*D/2 parameters
pos_embed_w = Parameter(W, D/2)       # W*D/2 parameters
# Total: (H + W) * D/2 parameters << H*W*D
```

**Benefits:**
- Massive parameter reduction for large images
- Forces model to learn axis-aligned position representations
- Easier to extrapolate to different resolutions

**Drawbacks:**
- Cannot represent diagonal or rotation-dependent patterns
- Assumes position effects are axis-separable

## Relative 2D Positional Encodings

### Relative Position Bias (Swin Transformer Style)

**Concept:** Instead of absolute positions, encode the relative distance between patches in attention computation.

**Key insight:** For attention between patches at positions (h₁, w₁) and (h₂, w₂), what matters is their relative offset:
```
Δh = h₁ - h₂
Δw = w₁ - w₂
```

**Implementation approach:**
```python
# Relative position bias table
# For max_relative_distance = 7 (e.g., 14x14 patches)
rel_pos_h = 2 * max_relative_distance + 1  # -7 to +7 = 15
rel_pos_w = 2 * max_relative_distance + 1  # -7 to +7 = 15

# Learnable bias table
relative_position_bias_table = Parameter(
    rel_pos_h * rel_pos_w,  # All possible (Δh, Δw) combinations
    num_heads               # Separate bias per attention head
)

# During attention computation:
# 1. Calculate relative position indices for all patch pairs
# 2. Look up bias from table
# 3. Add bias to attention scores

attention_scores = Q @ K.T / sqrt(d_k) + relative_position_bias
attention_weights = softmax(attention_scores)
```

**Characteristics:**
- **Translation invariance:** Same relative offset gets same bias regardless of absolute position
- **Learned:** Bias values optimized during training
- **Per-head:** Different attention heads can use different position biases
- **Bounded:** Typically clip max relative distance to save parameters

From [GitHub: 2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer) (accessed 2025-01-31):
- Relative position encoding on CIFAR10: **90.57% accuracy**
- **Best performer** among all encoding types tested
- On CIFAR100: **65.11% accuracy** (also best)
- Significantly outperforms absolute encodings

### 2D Relative Attention

**Extended formulation** for 2D images:

```
For query patch q at position (h_q, w_q) and key patch k at (h_k, w_k):

# Calculate 2D relative offsets
Δh = h_q - h_k
Δw = w_q - w_k

# Clip to maximum distance (for bounded parameter table)
Δh_clipped = clip(Δh, -max_dist, max_dist)
Δw_clipped = clip(Δw, -max_dist, max_dist)

# Convert to table index
index = (Δh_clipped + max_dist) * (2*max_dist + 1) + (Δw_clipped + max_dist)

# Look up bias
bias = relative_position_bias_table[index]

# Apply in attention
attention_score(q, k) = (q · k) / sqrt(d_k) + bias
```

**Memory requirements:**
```
Parameters = (2*max_relative_distance + 1)² × num_heads

Example: max_dist=7, heads=8
Parameters = 15² × 8 = 1,800 values
```

### Translation Invariance Properties

**Key advantage of relative encoding:**

```
# Absolute encoding: Position (0,0) and (5,5) get different codes
PE_abs(0, 0) ≠ PE_abs(5, 5)

# Relative encoding: Relationship between patches preserved
# Patch pair [(0,0), (1,0)] has same relative encoding as [(5,5), (6,5)]
# Both have Δh=1, Δw=0

This means:
- Model learns patterns that transfer across image locations
- Same edge detector works top-left and bottom-right
- Natural data augmentation through position invariance
```

## Implementation Comparison

### Performance Comparison

From [GitHub: 2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer) experimental results on compact ViT (800k parameters):

| Encoding Type | CIFAR10 Accuracy | CIFAR100 Accuracy | Parameters Added |
|---------------|------------------|-------------------|------------------|
| **No Position** | 79.63% | 53.25% | 0 |
| **Learnable** | 86.52% | 60.87% | 8,192 |
| **Sinusoidal (Absolute)** | 86.09% | 59.73% | 0 |
| **Relative** | **90.57%** | **65.11%** | 2,304 |
| **Rotary (RoPE)** | 88.49% | 62.88% | 0 |

**Key findings:**
- Position encoding provides **~7-11% accuracy boost** (critical)
- Relative encoding outperforms all others by **~2-5%**
- Sinusoidal and learned absolute encodings comparable
- Rotary (RoPE) offers good performance with zero parameters

### Memory Considerations

**Parameter counts for 224×224 image, patch_size=16, embed_dim=768:**

```
Image produces: 14×14 = 196 patches

# Learned 2D (factorized)
pos_embed_h: 14 × 384 = 5,376 parameters
pos_embed_w: 14 × 384 = 5,376 parameters
Total: 10,752 parameters

# Learned 2D (full, not typical)
pos_embed_full: 196 × 768 = 150,528 parameters

# Sinusoidal 2D
Parameters: 0 (computed on-the-fly)

# Relative (max_dist=7, 8 heads)
Bias table: 15 × 15 × 8 = 1,800 parameters

# RoPE
Parameters: 0 (rotation applied mathematically)
```

**Inference speed considerations:**
- Learned/Sinusoidal: Fast (simple addition to patch embeddings)
- Relative: Moderate (requires attention bias computation)
- RoPE: Moderate (rotation computation per attention layer)

### When to Use Which Approach

**Learned 2D position encoding:**
- ✅ Good default choice
- ✅ Fixed input resolution
- ✅ Dataset-specific adaptation needed
- ❌ Cannot extrapolate to different resolutions

**Sinusoidal 2D encoding:**
- ✅ No learned parameters (saves memory)
- ✅ Can extrapolate to different image sizes
- ✅ Smooth position transitions
- ❌ May underperform learned on small datasets

**Relative 2D encoding:**
- ✅ **Best accuracy** in many benchmarks
- ✅ Translation invariance
- ✅ Works well with local attention (Swin)
- ❌ Slightly more complex implementation

**Rotary (RoPE) 2D:**
- ✅ Zero parameters
- ✅ Relative position awareness
- ✅ Extrapolates to longer sequences
- ⚠️ More complex implementation than sinusoidal

## Handling the Classification Token

Vision transformers typically prepend a learnable `[CLS]` token to the sequence. Different encoding strategies handle this token differently:

### Position Encoding for CLS Token

**Option 1: No position for CLS**
```python
# CLS token gets zero position encoding
cls_token = cls_token + 0  # No positional info
patch_tokens = patch_tokens + pos_embed_2d
```

**Option 2: Learnable CLS position**
```python
# CLS token learns its own position
cls_pos = Parameter(1, 1, embed_dim)
cls_token = cls_token + cls_pos
patch_tokens = patch_tokens + pos_embed_2d
```

**Option 3: Special position index**
```python
# RoPE approach: CLS gets position (0, 0)
# Patch positions start at (1, 1)
# This avoids rotation for CLS (no positional bias)
```

From [GitHub: 2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer):
- **Learnable:** CLS token learns its encoding
- **Sinusoidal:** Patch tokens receive sinusoidal encoding; CLS learns its own
- **Relative:** CLS excluded from distance calculations; fixed index (0) represents its distance
- **RoPE:** X and Y positions start at 1 for patches, reserving 0 for CLS (no rotation applied)

### Best Practices

**Most common approach (ViT, DeiT, etc.):**
```python
# Learned position for CLS + learned/sinusoidal for patches
cls_token = repeat(cls_token, '1 1 d -> b 1 d', b=batch_size)
cls_pos = Parameter(1, 1, embed_dim)

# Apply positions
cls_token = cls_token + cls_pos
patch_tokens = patch_tokens + pos_embed_2d

# Concatenate
tokens = torch.cat([cls_token, patch_tokens], dim=1)
```

This approach:
- Allows CLS to learn optimal position in sequence
- Doesn't force CLS to fit spatial grid structure
- Works well in practice

## Advanced: Interpolation for Variable Resolutions

### Position Encoding Interpolation

**Problem:** Model trained on 224×224 images (14×14 patches) needs to handle 384×384 images (24×24 patches).

**Solution:** Interpolate learned position encodings to new size.

```python
def interpolate_pos_encoding(pos_embed, height, width):
    """
    Interpolate 2D position encoding to new spatial size.

    Args:
        pos_embed: Original encoding (1, H_old*W_old, D)
        height: New height in patches
        width: New width in patches

    Returns:
        Interpolated encoding (1, H_new*W_new, D)
    """
    num_patches = height * width
    N = pos_embed.shape[1]

    if num_patches == N:
        return pos_embed

    # Separate CLS token if present
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]

    # Get original dimensions
    h_old = w_old = int(math.sqrt(patch_pos_embed.shape[1]))

    # Reshape to 2D grid
    patch_pos_embed = patch_pos_embed.reshape(1, h_old, w_old, -1)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, D, H, W)

    # Interpolate using bicubic
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(height, width),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, D)
    patch_pos_embed = patch_pos_embed.flatten(1, 2)  # (1, H*W, D)

    # Re-add CLS token
    pos_embed = torch.cat([class_pos_embed.unsqueeze(1), patch_pos_embed], dim=1)

    return pos_embed
```

**Interpolation quality factors:**
- Bicubic interpolation generally works best
- Sinusoidal encodings don't need interpolation (can compute directly)
- Relative encodings naturally handle any resolution (just extend bias table)

## Sources

**Web Research:**

- [A 2D Semantic-Aware Position Encoding for Vision Transformers](https://arxiv.org/html/2505.09466v1) - arXiv:2505.09466 (accessed 2025-01-31)
  - Proposes semantic-aware 2D positional encoding with dynamic adaptation
  - Integrates position encoding with inherent 2D spatial structure

- [GitHub: s-chh/2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer) (accessed 2025-01-31)
  - Comprehensive comparison of 2D positional encoding strategies
  - Experimental results on CIFAR10/CIFAR100
  - Implementation code for learned, sinusoidal, relative, and RoPE encodings
  - CLS token handling for each encoding type

- [Position Embeddings for Vision Transformers, Explained](https://medium.com/data-science/position-embeddings-for-vision-transformers-explained-a6f9add341d5) (accessed 2025-01-31)
  - Detailed mathematical walkthrough with visualizations
  - Demonstrates position invariance of transformers
  - Code examples for sinusoidal 2D encoding

- [Vision Transformer with 2D Explicit Position Encoding](https://arxiv.org/html/2403.13298v1) - arXiv:2403.13298 (accessed 2025-01-31)
  - Investigates explicit 2D coordinate position encoding
  - Concatenates 2D positional coordinates to token embeddings

**Additional References:**

- [Positional Embeddings in Transformer Models](https://iclr-blogposts.github.io/2025/blog/positional-embedding/) - ICLR Blogposts 2026 (accessed 2025-01-31)
  - Examines positional encoding techniques across transformers
  - Emphasizes importance for 2D vision applications

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Original ViT paper
  - Ablation studies on 1D vs 2D vs relative position embeddings
  - Finds little difference between position encoding types (all help significantly vs none)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
  - Introduced sinusoidal positional encoding
  - Establishes transformer position-invariance problem
