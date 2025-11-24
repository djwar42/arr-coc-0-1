# 16× Compression Mechanism

## Overview

DeepSeek-OCR achieves **16× spatial compression** inside the SAM encoder, reducing 4096 tokens to 256 patches before expensive CLIP processing.

**Location**: `deepencoder/sam_vary_sdpa.py:166-183`

## Implementation

```python
# Neck: Channel reduction
self.neck = nn.Sequential(
    nn.Conv2d(768, 256, kernel_size=1, bias=False),  # 768→256 channels
    LayerNorm2d(256),
    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
)

# Spatial compression: 64×64 → 16×16
self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 64→32
self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) # 32→16
```

## Flow

```
SAM Output: [B, 768, 64, 64] (4096 spatial locations)
    ↓
Neck: [B, 256, 64, 64] (channel reduction)
    ↓
net_2 (stride=2): [B, 512, 32, 32] (2× spatial compression)
    ↓
net_3 (stride=2): [B, 1024, 16, 16] (2× spatial compression)
    ↓
Final: [B, 1024, 16, 16] = 256 patches (16× compression)
```

**Math**: 64×64 → 32×32 → 16×16 = 4096 → 1024 → 256 tokens

## Why 16×?

**Too little compression (4×, 8×)**:
- CLIP still expensive
- Memory issues
- Slow inference

**Too much compression (32×, 64×)**:
- Information loss
- OCR quality drops
- Fine details lost

**16× is empirically optimal**: Quality vs efficiency sweet spot

## File References

- `deepencoder/sam_vary_sdpa.py:166-183` - Compression layers
- `deepencoder/sam_vary_sdpa.py:300-350` - Forward pass

**See Also**:
- [deepencoder.md](deepencoder.md) - Full SAM+CLIP architecture
- [../concepts/optical-compression.md](../concepts/optical-compression.md) - Why compression works
