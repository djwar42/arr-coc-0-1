# SAM Implementation

**File**: `deepencoder/sam_vary_sdpa.py`

## Key Components

### Initialization (lines 120-183)

```python
class SamVisionTower(nn.Module):
    def __init__(self):
        # Patch embedding
        self.patch_embed = PatchEmbed(...)  # 16×16 patches

        # 12 transformer blocks (window attention)
        self.blocks = nn.ModuleList([...])

        # COMPRESSION LAYERS (KEY!)
        self.neck = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),  # Channel reduction
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
```

### Forward Pass (lines 300-350)

```python
def forward(self, x):
    # Patch embed
    x = self.patch_embed(x)  # [B, 4096, 768]

    # Window attention blocks
    for block in self.blocks:
        x = block(x)

    # Reshape for conv
    x = x.reshape(B, 64, 64, 768).permute(0, 3, 1, 2)  # [B, 768, 64, 64]

    # COMPRESS!
    x = self.neck(x)     # [B, 256, 64, 64]
    x = self.net_2(x)    # [B, 512, 32, 32]
    x = self.net_3(x)    # [B, 1024, 16, 16]

    return x  # 256 spatial patches
```

## Window Attention

**Lines 400-450**: Standard SAM window attention
- 8×8 windows
- O(N) complexity
- Local pattern capture

**See Also**:
- [../architecture/deepencoder.md](../architecture/deepencoder.md) - SAM+CLIP design
- [../architecture/compression.md](../architecture/compression.md) - Compression mechanism
