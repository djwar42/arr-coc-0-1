# CLIP Implementation

**File**: `deepencoder/clip_sdpa.py`

## Key Components

### Initialization (lines 200-250)

```python
class CLIPVisionTower(nn.Module):
    def __init__(self):
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))

        # 24 transformer blocks (global attention)
        self.transformer = Transformer(...)
```

### Forward Pass (lines 300-380)

```python
def forward(self, x):
    # x: [B, 1024, 16, 16] from SAM

    # Flatten spatial
    x = x.flatten(2).transpose(1, 2)  # [B, 256, 1024]

    # Add CLS token
    cls = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)  # [B, 257, 1024]

    # Global attention (24 layers)
    x = self.transformer(x)

    return x  # [B, 257, 1024]
```

## Key Difference from Standard CLIP

**Standard CLIP**: Processes raw images independently

**DeepSeek-OCR CLIP**: Uses SAM's compressed output as patch embeddings
- Input: SAM features (not raw image)
- Already compressed 16×
- Adds semantic understanding on top

**See Also**:
- [../architecture/deepencoder.md](../architecture/deepencoder.md) - Serial SAM→CLIP design
- [sam-implementation.md](sam-implementation.md) - SAM details
