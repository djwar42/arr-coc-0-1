# DeepEncoder Architecture

## Overview

DeepEncoder is the vision component of DeepSeek-OCR (380M params), designed as a **serial SAM → CLIP architecture** with built-in 16× compression.

**Key Innovation**: Process high-res cheaply (SAM), compress spatially, then extract semantics expensively (CLIP).

## Architecture Diagram

```
Input Image (1024×1024)
    ↓
Patch Embedding (16×16 patches)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ SAM-base (80M params)                                       │
│                                                             │
│ Input: [B, 3, 1024, 1024]                                   │
│    ↓                                                        │
│ Patch Embed: [B, 4096, 768] (64×64 patches)                │
│    ↓                                                        │
│ 12 Transformer Blocks (Window Attention)                   │
│ - Cheap: O(N) complexity                                    │
│ - Local: 8×8 windows                                        │
│    ↓                                                        │
│ Neck (Conv 768→256 + LayerNorm)                            │
│    ↓                                                        │
│ net_2 (Conv 256→512, stride=2)  # spatial /2               │
│    ↓                                                        │
│ net_3 (Conv 512→1024, stride=2) # spatial /2               │
│    ↓                                                        │
│ Output: [B, 1024, 16, 16] = 256 spatial patches            │
│                                                             │
│ Compression: 4096 → 256 tokens (16×)                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CLIP-large (300M params)                                    │
│                                                             │
│ Input: SAM output as patch embeddings                       │
│ Flatten: [B, 1024, 16, 16] → [B, 256, 1024]                │
│ Add CLS token: [B, 257, 1024]                              │
│    ↓                                                        │
│ 24 Transformer Blocks (Global Attention)                   │
│ - Expensive: O(N²) complexity                               │
│ - Dense: Full attention matrix                              │
│ - Semantic: Rich understanding                              │
│    ↓                                                        │
│ Output: [B, 257, 1024]                                      │
│                                                             │
│ (CLS token will be dropped later)                          │
└─────────────────────────────────────────────────────────────┘
```

## Why Serial Architecture?

### Alternative: Parallel SAM + CLIP

```
          Input Image
         /           \
      SAM           CLIP
         \           /
          Concatenate
```

**Problems**:
- Both process full 4096 tokens
- CLIP on 4096 tokens = O(4096²) = 16M operations
- Memory explosion!
- Slow!

### Chosen: Serial SAM → Compress → CLIP

```
Input Image
    ↓
  SAM (cheap window attention)
    ↓
  Compress 4096 → 256
    ↓
  CLIP (expensive global attention)
```

**Advantages**:
- SAM: O(4096) on high-res
- Compress: Cheap convolutions
- CLIP: O(256²) = 65K operations (256× faster!)
- Memory efficient
- Fast!

## SAM Component

**File**: `deepencoder/sam_vary_sdpa.py`

### Initialization (lines 120-183)

```python
class SamVisionTower(nn.Module):
    def __init__(self, embed_dim=768, out_chans=256):
        # Standard SAM trunk (12 transformer blocks)
        self.trunk = ...  # Window attention

        # Compression layers (THE KEY INNOVATION!)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),  # 768→256
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )

        # Spatial compression (16× reduction)
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
```

### Forward Pass (lines 300-350)

```python
def forward(self, x):
    # x: [B, 3, 1024, 1024]

    # Patch embedding
    x = self.patch_embed(x)  # [B, 4096, 768]

    # 12 transformer blocks (window attention)
    for block in self.blocks:
        x = block(x)  # [B, 4096, 768]

    # Reshape for convolutions
    B, N, C = x.shape
    H = W = int(N**0.5)  # 64
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 64, 64]

    # Compression!
    x = self.neck(x)        # [B, 256, 64, 64]
    x = self.net_2(x)       # [B, 512, 32, 32] - spatial /2
    x = self.net_3(x)       # [B, 1024, 16, 16] - spatial /2

    return x  # [B, 1024, 16, 16] = 256 spatial patches
```

**Key Insight**: Compression happens INSIDE SAM, not as separate module!

### Window Attention

**Why window attention?**
- Cheap: O(N) instead of O(N²)
- Local patterns captured (fine-grained details)
- High resolution processing feasible

**How it works**:
- Divide 64×64 into 8×8 windows
- Attention within each window only
- No cross-window attention (cheap!)

## CLIP Component

**File**: `deepencoder/clip_sdpa.py`

### Initialization (lines 200-250)

```python
class CLIPVisionTower(nn.Module):
    def __init__(self, embed_dim=1024):
        # 24 transformer blocks (global attention)
        self.transformer = ...

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
```

### Forward Pass (lines 300-380)

```python
def forward(self, x):
    # x: [B, 1024, 16, 16] from SAM

    # Flatten spatial dimensions
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # [B, 256, 1024]

    # Add CLS token
    cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 1024]
    x = torch.cat([cls_tokens, x], dim=1)  # [B, 257, 1024]

    # 24 transformer blocks (GLOBAL attention!)
    for block in self.blocks:
        x = block(x)  # [B, 257, 1024]

    return x  # [B, 257, 1024]
```

**Key**: Uses SAM's compressed output, doesn't process original image!

### Global Attention

**Why global attention?**
- Semantic understanding (big picture)
- Cross-patch relationships
- Rich features for LLM

**Cost**:
- Expensive: O(N²) where N=257
- Only feasible because SAM compressed first!

## Feature Fusion

**File**: `deepencoder/build_linear.py`

**Code** (lines 40-80):

```python
class MlpProjector(nn.Module):
    def __init__(self, in_features=2048, out_features=1280):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, sam_features, clip_features):
        # sam_features: [B, 1024, 16, 16]
        # clip_features: [B, 257, 1024]

        # Drop CLIP CLS token
        clip_features = clip_features[:, 1:, :]  # [B, 256, 1024]

        # Flatten SAM
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, 256, 1024]

        # Concatenate
        fused = torch.cat([clip_features, sam_flat], dim=-1)  # [B, 256, 2048]

        # Project to language space
        output = self.linear(fused)  # [B, 256, 1280]

        return output
```

**Why concatenate?**
- SAM: Fine-grained local details
- CLIP: Global semantic understanding
- Both needed for OCR!

## Computational Analysis

### SAM (80M params)

**FLOPs**:
- Patch embedding: ~1.6 GFLOPs
- 12 window attention blocks: ~60 GFLOPs
- Compression layers: ~3 GFLOPs
- **Total**: ~65 GFLOPs

**Memory**:
- Activations: ~1.2 GB (4096 tokens × 768 dim)
- Small thanks to window attention!

### CLIP (300M params)

**FLOPs**:
- 24 global attention blocks: ~180 GFLOPs
- **Total**: ~180 GFLOPs

**Memory**:
- Activations: ~260 MB (257 tokens × 1024 dim)
- Small thanks to compression!

### Total DeepEncoder

**FLOPs**: ~245 GFLOPs
**Memory**: ~1.5 GB
**Speed**: ~50ms on A100 (base mode)

**Comparison** (if CLIP processed 4096 tokens):
- FLOPs: ~2,800 GFLOPs (11× more!)
- Memory: ~16 GB (10× more!)
- Speed: ~800ms (16× slower!)

**Conclusion**: Serial architecture with compression = massive savings!

## Multi-Resolution Support

DeepEncoder handles 73-421 tokens with **same weights**!

**How?**
- Positional encoding interpolation
- All resolutions trained simultaneously
- No separate models needed

**Example** (Tiny vs Large):

**Tiny (512×512)**:
- SAM input: [B, 3, 512, 512]
- Patches: 32×32 = 1024
- After compression: 8×8 = 64 patches
- CLIP processes: 65 tokens (64 + CLS)

**Large (1280×1280)**:
- SAM input: [B, 3, 1280, 1280]
- Patches: 80×80 = 6400
- After compression: 20×20 = 400 patches
- CLIP processes: 401 tokens (400 + CLS)

**Same architecture, different resolutions!**

## Design Rationale

**Q**: Why not just use CLIP?
**A**: CLIP would need 4096 tokens → too expensive

**Q**: Why not just use SAM?
**A**: SAM lacks semantic understanding needed for OCR

**Q**: Why serial, not parallel?
**A**: Compression before expensive CLIP = memory/speed wins

**Q**: Why 16× compression specifically?
**A**: Empirically optimal balance:
- Less: CLIP too expensive
- More: Quality degradation

## File References

**Implementation**:
- `deepencoder/sam_vary_sdpa.py` - SAM component
- `deepencoder/clip_sdpa.py` - CLIP component
- `deepencoder/build_linear.py` - Feature fusion
- `deepseek_ocr.py:394-407` - Integration code

**See Also**:
- [compression.md](compression.md) - How 16× compression works
- [../code-reference/sam-implementation.md](../code-reference/sam-implementation.md) - SAM code details
- [../code-reference/clip-implementation.md](../code-reference/clip-implementation.md) - CLIP code details
