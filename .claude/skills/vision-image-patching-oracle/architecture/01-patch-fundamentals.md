# Patch Fundamentals: How Image Patching Works

**The core mechanics of converting images to visual tokens**

## The Basic Pipeline

```
Input Image → Patch Division → Flatten → Project → Add Position → Visual Tokens
(H×W×3)      (N×P×P×3)       (N×D_p)  (N×D)    (N×D)         (N×D)
```

Where:
- **H, W**: Image height and width
- **P**: Patch size (typically 14 or 16)
- **N**: Number of patches = (H/P) × (W/P)
- **D_p**: Patch dimension = P × P × 3
- **D**: Embedding dimension (768, 1024, 1536, etc.)

---

## Step 1: Patch Division

### Grid Partitioning

**Divide image into non-overlapping patches**

```python
# Input: image [B, 3, 336, 336]
# Output: patches [B, 576, 14×14×3]

patch_size = 14
patches = rearrange(image,
                   'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                   p1=patch_size, p2=patch_size)

# Result shape: [batch, num_patches, patch_dim]
# [B, 576, 588] where 588 = 14×14×3
```

**Visual Representation:**

```
┌─────────────────────┐
│ Original 336×336    │      ┌──┬──┬──┬───┬──┐
│                     │ →    │P1│P2│P3│...│576
│     Image           │      ├──┼──┼──┼───┼──┤
│                     │      │  │  │  │   │  │
└─────────────────────┘      └──┴──┴──┴───┴──┘
                              24×24 patch grid
```

### Key Properties

**No Overlap**: Each pixel belongs to exactly one patch
- Efficient: No redundant computation
- Clean: Each patch is independent token

**Fixed Spatial Layout**: Patches maintain 2D grid structure
- Preserves neighborhood relationships
- Enables 2D positional encoding

**Size Constraint**: Image dimensions must be divisible by patch size
- Common practice: Resize/pad images to multiples of patch size
- Alternative: Variable patching (LLaVA-UHD, Ovis)

---

## Step 2: Flatten & Project

### Flattening

**Convert 2D patches to 1D vectors**

```python
# Each patch: [14, 14, 3] → [588]
flattened = patches.flatten(start_dim=2)
```

**Why Flatten?**
- Transformers expect 1D sequences
- Linear projection requires 1D input
- Loses explicit 2D structure (but position encoding restores it)

### Linear Projection

**Map patch vectors to embedding space**

```python
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        patch_dim = patch_size * patch_size * in_chans
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, patches):
        # patches: [B, N, 588]
        # output: [B, N, 1024]
        return self.proj(patches)
```

**Learnable Transformation:**
- **Input**: Raw pixel values [588 dims]
- **Output**: Semantic embeddings [1024 dims]
- **Purpose**: Map visual features to same space as text tokens

**Alternative**: Convolutional projection (equivalent but faster)

```python
self.proj = nn.Conv2d(3, embed_dim,
                     kernel_size=patch_size,
                     stride=patch_size)
```

---

## Step 3: Positional Encoding

### Why Position Information?

**Problem**: Flattening destroys spatial layout
- Patch [5,5] and patch [20,20] look identical to transformer
- But position is critical for understanding (top vs bottom, left vs right)

**Solution**: Add position embeddings to each patch

### Learned 2D Positional Embeddings (ViT Standard)

```python
class ViT(nn.Module):
    def __init__(self, img_size=336, patch_size=14, embed_dim=1024):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 576

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

    def forward(self, patches):
        # patches: [B, 576, 1024]
        # Add position info
        x = patches + self.pos_embed  # Broadcasting
        return x
```

**Characteristics:**
- **Learned**: Optimized during training
- **Fixed Size**: Must match training resolution
- **2D-Aware**: Can encode both x and y positions

### Sinusoidal Position Encoding (Alternative)

```python
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate sinusoidal position embeddings
    embed_dim: embedding dimension
    grid_size: number of patches per side (24 for 336×336, patch=14)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # W, H
    grid = np.stack(grid, axis=0)  # [2, grid_size, grid_size]

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

# Each position gets unique sinusoidal pattern
# Extrapolates better to unseen resolutions than learned
```

**Characteristics:**
- **Fixed**: Not learned, deterministic
- **Extrapolates**: Works better on resolution changes
- **Separable**: Often split into x_pos and y_pos components

### RoPE (Rotary Position Embedding)

**Used in**: Modern LLMs, some VLMs (Ovis, DeepSeek-OCR)

```python
# Encode relative positions via rotation
# Naturally handles 2D layouts
# Better extrapolation properties
```

**Reference**: `Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models`

---

## Step 4: CLS Token (Optional)

### Global Image Representation

**Add special [CLS] token at sequence start**

```python
class ViT(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, patch_embeds):
        B = patch_embeds.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # Prepend to sequence
        x = torch.cat([cls_tokens, patch_embeds], dim=1)  # [B, N+1, D]
        return x
```

**Purpose**:
- **Aggregation**: Attends to all patches, learns global representation
- **Output**: Used for classification (in ViT)
- **VLMs**: Less common, often skip CLS and use all patch tokens

---

## Complete Code Example

### Standard ViT Patching

```python
import torch
import torch.nn as nn
from einops import rearrange

class ViTPatchEmbed(nn.Module):
    """Vision Transformer Patch Embedding"""

    def __init__(self,
                 img_size=336,
                 patch_size=14,
                 in_chans=3,
                 embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 576

        # Patch embedding projection
        self.proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size,
                             stride=patch_size)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

    def forward(self, x):
        # x: [B, 3, 336, 336]
        B, C, H, W = x.shape

        # Patch projection: [B, 3, 336, 336] → [B, 1024, 24, 24]
        x = self.proj(x)

        # Flatten: [B, 1024, 24, 24] → [B, 1024, 576]
        x = x.flatten(2)

        # Transpose: [B, 1024, 576] → [B, 576, 1024]
        x = x.transpose(1, 2)

        # Add position embeddings
        x = x + self.pos_embed

        return x  # [B, 576, 1024]

# Usage
embed = ViTPatchEmbed(img_size=336, patch_size=14, embed_dim=1024)
image = torch.randn(2, 3, 336, 336)
tokens = embed(image)
print(tokens.shape)  # torch.Size([2, 576, 1024])
```

---

## Patch Size Impact

### Common Patch Sizes

| Patch Size | 224×224 | 336×336 | 448×448 | Use Case |
|------------|---------|---------|---------|----------|
| **14×14** | 256 tok | 576 tok | 1024 tok | High detail |
| **16×16** | 196 tok | 441 tok | 784 tok | Standard (ViT) |
| **32×32** | 49 tok | 110 tok | 196 tok | Efficient |

**Trade-off:**
- **Smaller patches** = More tokens = Finer detail BUT slower
- **Larger patches** = Fewer tokens = Faster BUT coarser

### Resolution Scaling

**Key Insight**: Token count grows quadratically with resolution

```
Token Count = (H / P) × (W / P) = H×W / P²

Doubling resolution (H,W → 2H,2W) quadruples tokens!
```

**Example (patch_size=14):**
- 224×224 → 256 tokens
- 448×448 → 1024 tokens (4×)
- 896×896 → 4096 tokens (16×!)

**Why This Matters:**
- Transformer attention: O(N²) complexity
- 4× tokens = 16× computation for attention!
- Motivates compression & adaptive strategies

---

## Variations & Extensions

### Overlapping Patches

**Standard**: Non-overlapping grid
**Alternative**: Sliding window with stride < patch_size

```python
# Overlapping patches (stride=8, patch=16)
patches = F.unfold(image, kernel_size=16, stride=8)
# More patches, more computation, better local modeling
```

**Trade-off**: Better local detail BUT more tokens

### Hierarchical Patching

**Multi-scale approach**: Different patch sizes at different stages

```
Stage 1: 32×32 patches (coarse, 49 tokens)
Stage 2: 16×16 patches (medium, 196 tokens)
Stage 3: 8×8 patches (fine, 784 tokens)
```

**Used in**: Swin Transformer, Pyramid Vision Transformer

**Reference**: `ResFormer: Scaling ViTs With Multi-Resolution Training`

### Irregular Patches

**Beyond grids**: Superpixel segmentation, saliency-based regions

**Adaptive Patch Transformer (APT)**: Mixed patch sizes per image
- Large patches for simple regions
- Small patches for complex regions

**Reference**: `Accelerating Vision Transformers with Adaptive Patch Sizes`

---

## Key Takeaways

### 1. Patching = Visual Tokenization
Patches are the "words" of vision - discrete units of visual information

### 2. Size Matters
Patch size controls resolution-efficiency tradeoff
- Industry standard: 14×14 or 16×16
- Adaptive approaches: Mixed sizes per content

### 3. Position is Critical
Flattening loses spatial structure - position encoding restores it
- Learned embeddings (ViT)
- Sinusoidal (extrapolation)
- RoPE (relative positions)

### 4. Quadratic Scaling is Expensive
Token count grows as (resolution / patch_size)²
- Motivates compression
- Drives adaptive patching
- Requires careful resolution management

---

## Related Documentation

- **[architecture/00-overview.md](00-overview.md)** - Patching paradigms
- **[architecture/02-adaptive-patching.md](02-adaptive-patching.md)** - Variable patch sizes
- **[concepts/00-image-tokenization.md](../concepts/00-image-tokenization.md)** - Theoretical foundations
- **[concepts/01-patch-size-tradeoffs.md](../concepts/01-patch-size-tradeoffs.md)** - Size selection
- **[techniques/00-fixed-patching.md](../techniques/00-fixed-patching.md)** - Implementation details

---

**Next**: Explore [adaptive patching](02-adaptive-patching.md) for content-aware strategies
