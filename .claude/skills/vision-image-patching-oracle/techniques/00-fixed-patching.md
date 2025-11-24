# Fixed Patching

**Standard grid-based image division - the foundational approach of Vision Transformers**

## Overview

Fixed patching divides images into uniform rectangular patches regardless of content. This is the canonical approach introduced by Vision Transformer (ViT) and used throughout the vision-language ecosystem.

**From [source-documents/18_Vision Transformer](../source-documents/18_Vision transformer - Wikipedia.md)** and **[00_Comprehensive Study](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)**:

**Core principle**: Treat image patches as "visual words" analogous to text tokens

## Basic Algorithm

### 1. Image Division

```python
def divide_into_patches(image, patch_size=16):
    """
    Divide image into non-overlapping patches

    Args:
        image: [C, H, W] tensor (channels, height, width)
        patch_size: Size of square patches (P)

    Returns:
        patches: [N, P*P*C] where N = (H/P) * (W/P)
    """
    C, H, W = image.shape
    P = patch_size

    # Ensure image dimensions are divisible by patch size
    assert H % P == 0 and W % P == 0, \
        f"Image size {H}×{W} not divisible by patch size {P}"

    # Number of patches per dimension
    n_h = H // P
    n_w = W // P

    # Reshape: [C, H, W] → [C, n_h, P, n_w, P]
    patches = image.reshape(C, n_h, P, n_w, P)

    # Rearrange: [C, n_h, P, n_w, P] → [n_h, n_w, P, P, C]
    patches = patches.permute(1, 3, 2, 4, 0)

    # Flatten patches: [n_h, n_w, P, P, C] → [n_h*n_w, P*P*C]
    patches = patches.reshape(n_h * n_w, P * P * C)

    return patches  # Shape: [N, patch_dim] where patch_dim = P²*C
```

### 2. Efficient Implementation (einops)

```python
from einops import rearrange

def divide_into_patches_einops(image, patch_size=16):
    """Einops version - more readable"""
    patches = rearrange(
        image,
        'c (n_h p_h) (n_w p_w) -> (n_h n_w) (p_h p_w c)',
        p_h=patch_size,
        p_w=patch_size
    )
    return patches
```

**Example**:
```python
image = torch.randn(3, 224, 224)  # RGB image
patches = divide_into_patches_einops(image, patch_size=16)
print(patches.shape)  # [196, 768]
# 196 = (224/16)² patches
# 768 = 16×16×3 = patch dimension
```

### 3. Linear Embedding

```python
class PatchEmbedding(nn.Module):
    """Project patches to embedding dimension"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Divide into patches: [B, N, P²C]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=self.patch_size, p2=self.patch_size)

        # Project to embedding: [B, N, D]
        x = self.proj(x)

        return x
```

### 4. Alternative: Convolutional Embedding

```python
class ConvPatchEmbedding(nn.Module):
    """Use convolution for patch embedding (more efficient)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Convolution with stride=patch_size effectively divides into patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]

        # Flatten spatial dimensions: [B, D, N] → [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        return x
```

**Benefits of conv approach**:
- Slightly faster (optimized conv kernels)
- Equivalent mathematically (conv is linear operation)
- Standard in most implementations

## Position Encoding

Fixed patching requires position information since patches are processed as unordered set.

### Learnable 2D Positional Embeddings

```python
class PositionalEmbedding(nn.Module):
    """Standard ViT positional encoding"""
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        # Learnable position embeddings (including CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, N+1, D] (including CLS token)
        return x + self.pos_embed
```

### 2D Sinusoidal Encoding

```python
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sinusoidal position embeddings

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (H = W for square images)

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (W, H) order
    grid = np.stack(grid, axis=0)  # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate sin/cos embeddings from 2D grid"""
    assert embed_dim % 2 == 0

    # Use half embed_dim for each dimension
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sinusoidal embeddings"""
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
```

### Position Interpolation

For handling different resolutions at inference:

```python
def interpolate_pos_encoding(pos_embed, n_patches_new, n_patches_old):
    """
    Interpolate position embeddings for different number of patches

    Args:
        pos_embed: [1, n_patches_old+1, embed_dim] (includes CLS)
        n_patches_new: New number of patches
        n_patches_old: Original number of patches

    Returns:
        interpolated: [1, n_patches_new+1, embed_dim]
    """
    # Extract CLS and patch embeddings
    cls_embed = pos_embed[:, 0:1]  # [1, 1, D]
    patch_embed = pos_embed[:, 1:]  # [1, N_old, D]

    # Reshape to 2D grid
    grid_size_old = int(n_patches_old ** 0.5)
    grid_size_new = int(n_patches_new ** 0.5)

    patch_embed = patch_embed.reshape(1, grid_size_old, grid_size_old, -1)
    patch_embed = patch_embed.permute(0, 3, 1, 2)  # [1, D, H_old, W_old]

    # Bicubic interpolation
    patch_embed = F.interpolate(
        patch_embed,
        size=(grid_size_new, grid_size_new),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, n_patches_new, -1)

    # Concatenate CLS
    return torch.cat([cls_embed, patch_embed], dim=1)
```

## Complete Fixed Patching Pipeline

```python
class FixedPatchVisionEncoder(nn.Module):
    """Complete vision encoder with fixed patching"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = ConvPatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: [B, C, H, W] → [B, N, D]
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        return x  # [B, N+1, D]
```

## Standard Configurations

**From [source-documents/00_Comprehensive Study](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)**:

### ViT-Base/16
```python
config = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'n_patches': 196  # (224/16)²
}
```

### ViT-Large/14
```python
config = {
    'img_size': 224,
    'patch_size': 14,
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'n_patches': 256  # (224/14)²
}
```

### CLIP ViT-L/14 (336px)
```python
config = {
    'img_size': 336,
    'patch_size': 14,
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'n_patches': 576  # (336/14)²
}
```

## Advantages

**Simplicity**:
- Straightforward implementation
- Predictable behavior
- Easy to debug

**Efficiency**:
- Vectorized operations
- GPU-friendly (uniform tensor shapes)
- No conditional logic

**Pretraining compatibility**:
- Most pretrained models use fixed patching
- Can leverage existing checkpoints
- Well-established training recipes

**Predictable token count**:
- Known sequence length before forward pass
- Easier memory planning
- Batch size optimization

## Limitations

**Content agnostic**:
- Treats all regions equally
- Sky gets same tokens as detailed text
- No adaptation to content complexity

**Aspect ratio distortion**:
- Requires square images (or padding/cropping)
- Distorts non-square images
- Information loss or computational waste

**Resolution limitations**:
- High-res images require many tokens
- Quadratic scaling: 2× resolution → 4× tokens
- Can exceed context limits

**Fixed granularity**:
- Single patch size for all images
- Cannot zoom into details
- Cannot compress simple regions

## Practical Tips

### Choosing Patch Size

**Guidelines**:
- **Classification**: 16×16 or 32×32 (fewer tokens needed)
- **Dense prediction**: 8×8 or 16×16 (more spatial detail)
- **Vision-language**: 14×14 or 16×16 (balance detail/efficiency)
- **Text-heavy**: 14×14 (finer granularity helps OCR)

### Memory Management

**Token count formula**:
```python
def compute_token_count(img_size, patch_size):
    return (img_size // patch_size) ** 2

# Examples
compute_token_count(224, 16)  # 196 tokens
compute_token_count(336, 14)  # 576 tokens
compute_token_count(672, 16)  # 1764 tokens - may exceed limits!
```

**Memory per image** (rough estimate):
```python
def estimate_memory_mb(img_size, patch_size, embed_dim, depth):
    n_tokens = (img_size // patch_size) ** 2
    # Activations per layer: roughly n_tokens × embed_dim × 4 bytes (FP32)
    memory_per_layer = n_tokens * embed_dim * 4 / (1024**2)
    return memory_per_layer * depth

# ViT-B/16 on 224×224
estimate_memory_mb(224, 16, 768, 12)  # ~69 MB
# ViT-L/14 on 672×672
estimate_memory_mb(672, 16, 1024, 24)  # ~1.6 GB!
```

### Implementation Checklist

- [ ] Verify image dimensions divisible by patch size
- [ ] Initialize position embeddings properly (trunc_normal for learnable)
- [ ] Include CLS token if using global pooling
- [ ] Implement position interpolation for variable resolutions
- [ ] Add dropout if training from scratch
- [ ] Normalize images (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for ImageNet)

## Primary Sources

- [18_Vision Transformer Wikipedia](../source-documents/18_Vision transformer - Wikipedia.md)
- [00_Comprehensive Study](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)
- [02_ViT Survey](../source-documents/02_A survey of the Vision Transformers and their CNN-Transformer based Variants - arXiv.md)

## Related Documents

- [../architecture/01-patch-fundamentals.md](../architecture/01-patch-fundamentals.md) - Conceptual overview
- [01-adaptive-patching.md](01-adaptive-patching.md) - Content-aware alternative
- [../concepts/01-patch-size-tradeoffs.md](../concepts/01-patch-size-tradeoffs.md) - Choosing patch size
- [../examples/00-basic-patching.md](../examples/00-basic-patching.md) - Code examples
