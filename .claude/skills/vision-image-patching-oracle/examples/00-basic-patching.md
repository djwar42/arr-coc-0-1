# Basic Patching Example

**Simple 16×16 patch division implementation**

## Standard ViT-Style Patching

```python
import torch
import torch.nn as nn

class BasicPatchEmbedding(nn.Module):
    """Standard fixed-size patch embedding like ViT"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolution-based patch extraction
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            patches: (batch, 196, 768)
        """
        # Extract patches with convolution
        x = self.proj(x)  # (batch, 768, 14, 14)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch, 768, 196)
        
        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)  # (batch, 196, 768)
        
        return x

# Usage
patch_embed = BasicPatchEmbedding(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768
)

# Process image
image = torch.randn(1, 3, 224, 224)
patches = patch_embed(image)
print(f"Patches shape: {patches.shape}")  # (1, 196, 768)
```

## Manual Patch Extraction

```python
def extract_patches_manual(image, patch_size=16):
    """
    Extract patches manually without convolution
    
    Args:
        image: (batch, channels, height, width)
        patch_size: Size of each patch
    
    Returns:
        patches: (batch, num_patches, patch_size*patch_size*channels)
    """
    batch, channels, height, width = image.shape
    
    # Calculate number of patches
    num_h = height // patch_size
    num_w = width // patch_size
    
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            # Extract patch
            patch = image[
                :, :,
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            
            # Flatten patch
            patch = patch.reshape(batch, -1)
            patches.append(patch)
    
    # Stack all patches
    patches = torch.stack(patches, dim=1)
    
    return patches

# Usage
image = torch.randn(1, 3, 224, 224)
patches = extract_patches_manual(image, patch_size=16)
print(f"Patches shape: {patches.shape}")  # (1, 196, 768)
```

## Adding Position Embeddings

```python
class PatchEmbeddingWithPosition(nn.Module):
    """Patch embedding with learnable position embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        
    def forward(self, x):
        # Extract patches
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        return x
```

## Primary Sources

- [18_Vision Transformer](../source-documents/18_Vision transformer - Wikipedia.md)

## Related Documents

- [../techniques/00-fixed-patching.md](../techniques/00-fixed-patching.md)
- [../models/01-vit.md](../models/01-vit.md)
- [../architecture/01-patch-fundamentals.md](../architecture/01-patch-fundamentals.md)
