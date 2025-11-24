# Vision Transformer (ViT)

**The foundational fixed-patching architecture for vision transformers**

## Overview

Vision Transformer introduced the concept of treating images as sequences of patches, establishing the standard patching approach used by most modern VLMs.

## Architecture

**From [source-documents/18_Vision Transformer](../source-documents/18_Vision transformer - Wikipedia.md)**:

### Patch Extraction

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.proj(x)  # (batch, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (batch, 196, 768)
        return x
```

### Position Embeddings

```python
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768):
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.pos_embed
```

## Standard Configurations

**ViT-Base/16**: 224×224 image, 16×16 patches, 196 tokens
**ViT-Large/14**: 224×224 image, 14×14 patches, 256 tokens  
**ViT-Large/14**: 336×336 image, 14×14 patches, 576 tokens (CLIP)

## Primary Sources

- [18_Vision Transformer Wikipedia](../source-documents/18_Vision transformer - Wikipedia.md)
- [00_Comprehensive Study of ViT](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)
- [02_ViT Survey](../source-documents/02_A survey of the Vision Transformers and their CNN-Transformer based Variants - arXiv.md)

## Related Documents

- [../techniques/00-fixed-patching.md](../techniques/00-fixed-patching.md)
- [../architecture/01-patch-fundamentals.md](../architecture/01-patch-fundamentals.md)
