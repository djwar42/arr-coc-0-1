# Hiera: Hierarchical Vision Transformer (SAM 2 Encoder)

**"A Hierarchical Vision Transformer without the Bells-and-Whistles"**

**Source**: Bolya, D., et al. (2023). "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles." ICML 2023
**Impact**: SAM 2's secret weapon - 6× faster than ViT-H
**Philosophy**: Simpler design, better performance

---

## What is Hiera?

### The Problem: ViT is Flat and Slow

**Standard ViT** (used in SAM 1):
- All tokens attend to all tokens (global attention)
- Single-scale features (no hierarchy)
- Expensive: O(N²) complexity
- Slow for high-resolution images

**Example**:
- 1024×1024 image → 256×256 tokens
- Attention matrix: 256² × 256² = massive!
- Result: 0.5 FPS for video

### Hiera's Solution: Hierarchy + Simplicity

**Key Idea**: Multi-scale processing (like CNNs) with transformer power

**Design Principles**:
1. **Hierarchical**: Process features at multiple scales
2. **Simple**: Remove unnecessary complexity ("bells-and-whistles")
3. **Fast**: Local attention → fewer computations
4. **Effective**: Better than ViT on speed AND accuracy

**Result**: 6× faster than ViT-H, same or better accuracy!

---

## Architecture Overview

### 4-Stage Hierarchical Design

```
Input Image (H × W × 3)
    ↓
[STAGE 1] H/4 × W/4 × C₁
    ↓ (patch merging)
[STAGE 2] H/8 × W/8 × C₂
    ↓ (patch merging)
[STAGE 3] H/16 × W/16 × C₃
    ↓ (patch merging)
[STAGE 4] H/32 × W/32 × C₄
    ↓
Multi-Scale Features
```

**Key Point**: Each stage processes at a different spatial resolution!

### Stage Details (Hiera-L for SAM 2)

| Stage | Resolution | Channels | # Blocks | Purpose |
|-------|-----------|----------|---------|---------|
| 1 | H/4 × W/4 | 192 | 2 | High-res, local details |
| 2 | H/8 × W/8 | 384 | 3 | Mid-level features |
| 3 | H/16 × W/16 | 768 | 11 | Semantic features |
| 4 | H/32 × W/32 | 1536 | 3 | Global context |

**Total Parameters**: 212M (vs. ViT-H: 632M)

---

## Core Design: Remove the "Bells-and-Whistles"

### What ViT Has (that Hiera removes!)

**1. Class Token**
- ViT: Uses [CLS] token for classification
- Hiera: No [CLS] token! Use spatial pooling instead

**Why Remove?**
- Segmentation doesn't need global [CLS]
- Spatial features more useful
- Simpler design

**2. Positional Encoding**
- ViT: Learned absolute positional embeddings
- Hiera: No explicit positional encoding!

**Why Remove?**
- Hierarchical structure provides implicit position
- Patch merging creates spatial relationships
- Works better for variable-resolution images

**3. Layer Norm before MLP**
- ViT: Pre-norm (before attention & MLP)
- Hiera: Post-norm (after attention & MLP)

**Why Change?**
- Post-norm more stable for hierarchical design
- Better gradient flow across stages

**4. Separate Attention/MLP Blocks**
- ViT: Attention block → MLP block (separate)
- Hiera: Fused transformer block (combined)

**Why Fuse?**
- Fewer operations
- Better memory locality
- Faster inference

---

## Hierarchical Processing: How It Works

### Stage Progression

**Stage 1**: High-Resolution Local Features
```
Input: 1024×1024 image
Patch Embedding (stride 4)
    ↓
256×256 tokens (H/4 × W/4)
    ↓
2 Transformer Blocks (local attention within 7×7 windows)
    ↓
Output: 256×256 × 192 channels
```

**Patch Merging** (Stage 1 → Stage 2):
```
256×256 tokens
    ↓
Group into 2×2 patches → merge
    ↓
128×128 tokens (H/8 × W/8)
    ↓
Channels: 192 → 384 (double!)
```

**Stage 2-4**: Repeat (higher-level, lower-resolution)

### Multi-Scale Features for Segmentation

**Why Hierarchy Helps SAM 2**:

**Stage 1** (H/4): Fine details (object boundaries, textures)
**Stage 2** (H/8): Mid-level structures (object parts)
**Stage 3** (H/16): Semantic features (object category, context)
**Stage 4** (H/32): Global context (scene understanding)

**SAM 2 Mask Decoder** uses features from **all 4 stages**:
- Coarse features (Stage 4): Locate object roughly
- Fine features (Stage 1): Refine boundaries precisely
- Result: High-quality masks with precise edges!

---

## Speed: Why Hiera is 6× Faster

### Complexity Analysis

**ViT-H (flat, global attention)**:
- Tokens: N = (H/16) × (W/16) = 4,096 (for 1024×1024 image)
- Attention complexity: O(N²) = O(16M operations)
- All blocks: 32 layers × 16M = 512M operations

**Hiera-L (hierarchical, local attention)**:
- Stage 1: 256² tokens, local attention (7×7 windows) = O(256² × 49)
- Stage 2: 128² tokens, local attention = O(128² × 49)
- Stage 3: 64² tokens, mixed local+global = O(64² × 64²) (small!)
- Stage 4: 32² tokens, global = O(32² × 32²) (tiny!)
- **Total**: Much less than flat ViT!

**Speedup**: 6× faster (88× for video when combined with streaming!)

### Memory Efficiency

**ViT-H**:
- Single-scale features: H/16 × W/16 × 1280 channels
- Large intermediate activations

**Hiera-L**:
- Multi-scale features: Smaller at each stage
- Efficient patch merging (no redundant computation)
- Result: 2-3× less memory

---

## Pre-Training: MAE for Hiera

### Masked Autoencoding (MAE)

**SAM 2 uses MAE-pretrained Hiera** (not supervised ImageNet!)

**MAE Process**:
1. Randomly mask 75% of image patches
2. Encode visible patches with Hiera encoder
3. Decode → reconstruct masked patches
4. Loss: Pixel-level reconstruction

**Why MAE for Segmentation?**
- Learns to reconstruct fine details (good for masks!)
- Unsupervised (scales to massive datasets)
- Better than supervised pre-training for dense prediction

**Hiera + MAE** pre-training:
- SA-1B dataset: 11M images
- 75% masking ratio
- 400 epochs
- Result: Strong feature extractor for segmentation!

---

## Local vs. Global Attention

### Attention Strategy per Stage

**Stage 1-2**: Local Windowed Attention
- Attention within 7×7 windows only
- O(window_size²) complexity per token
- Captures local details (edges, textures)

**Stage 3**: Mixed Local + Global
- Alternate between local and global blocks
- Balance detail and context

**Stage 4**: Global Attention
- All tokens attend to all tokens
- Small number of tokens (32×32), so cheap!
- Captures scene-level context

**Efficiency**:
- Most computation in Stages 1-2 (local, fast)
- Global attention only when cheap (Stage 4, few tokens)
- Result: Speed without sacrificing global understanding!

---

## Hiera vs. Other Hierarchical Transformers

### Swin Transformer

**Swin**:
- Shifted windows for hierarchical processing
- Complex shifting logic
- Relative position bias

**Hiera**:
- Simple patch merging (no shifting)
- No positional encoding
- Faster and simpler!

**Performance**: Hiera matches Swin, 1.3× faster

### Pyramid Vision Transformer (PVT)

**PVT**:
- Hierarchical, but with spatial reduction attention
- Complex attention mechanism
- Heavy on memory

**Hiera**:
- Simpler attention (local windowed)
- Less memory
- Faster!

**Performance**: Hiera better than PVT, 2× faster

### ViT (Flat Transformer)

**ViT**:
- Single-scale, global attention
- Slow but simple

**Hiera**:
- Multi-scale, mixed attention
- Fast and simple!

**Performance**: Hiera matches ViT accuracy, 6× faster

---

## Implementation Details (SAM 2 Configuration)

### Hiera-L (SAM 2 default)

**Model Size**: 212M parameters

**Stage Configuration**:
```python
stages = [
    # Stage 1: H/4 × W/4
    {"resolution": "H/4", "channels": 192, "blocks": 2, "heads": 3},
    # Stage 2: H/8 × W/8
    {"resolution": "H/8", "channels": 384, "blocks": 3, "heads": 6},
    # Stage 3: H/16 × W/16
    {"resolution": "H/16", "channels": 768, "blocks": 11, "heads": 12},
    # Stage 4: H/32 × W/32
    {"resolution": "H/32", "channels": 1536, "blocks": 3, "heads": 24},
]
```

**Attention**:
- Head dimension: 64
- Window size: 7×7 (Stages 1-2)
- MLP ratio: 4.0

### Hiera-B+ (Faster, smaller)

**Model Size**: 84M parameters
**Speed**: 31 FPS (vs. 44 FPS for Hiera-L)
**Use Case**: Mobile/embedded deployment

---

## Code Example: Hiera Forward Pass

```python
import torch
import torch.nn as nn

class HieraStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            HieraBlock(dim, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B, H, W, C)
        for block in self.blocks:
            x = block(x)
        return x

class HieraBlock(nn.Module):
    """Simple transformer block without bells-and-whistles"""
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=dim * 4)

    def forward(self, x):
        # Post-norm design (Hiera)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    """Merge 2x2 patches → halve resolution, double channels"""
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        # Group into 2x2 patches
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).flatten(3)  # (B, H/2, W/2, 4C)

        # Project to 2C
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)

        return x

# Full Hiera encoder
class Hiera(nn.Module):
    def __init__(self):
        super().__init__()

        # Patch embedding (stride 4)
        self.patch_embed = nn.Conv2d(3, 192, kernel_size=7, stride=4, padding=3)

        # 4 stages
        self.stage1 = HieraStage(dim=192, depth=2, num_heads=3)
        self.merge1 = PatchMerging(dim=192)

        self.stage2 = HieraStage(dim=384, depth=3, num_heads=6)
        self.merge2 = PatchMerging(dim=384)

        self.stage3 = HieraStage(dim=768, depth=11, num_heads=12)
        self.merge3 = PatchMerging(dim=768)

        self.stage4 = HieraStage(dim=1536, depth=3, num_heads=24)

    def forward(self, x):
        # x: (B, 3, H, W)
        feats = []

        x = self.patch_embed(x)  # (B, 192, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, 192)

        x = self.stage1(x)
        feats.append(x)  # Stage 1 features
        x = self.merge1(x)

        x = self.stage2(x)
        feats.append(x)  # Stage 2 features
        x = self.merge2(x)

        x = self.stage3(x)
        feats.append(x)  # Stage 3 features
        x = self.merge3(x)

        x = self.stage4(x)
        feats.append(x)  # Stage 4 features

        return feats  # Multi-scale features!
```

---

## Why Hiera Works for SAM 2

### 1. Multi-Scale Features for Masks

**SAM 2 needs**:
- Fine details (object boundaries)
- Semantic understanding (what object is)
- Global context (scene relationships)

**Hiera provides** all 3 via hierarchical stages!

### 2. Speed for Real-Time Video

**SAM 2 processes**:
- 44 FPS video @ 480p
- Requires fast encoder!

**Hiera delivers** 6× speedup vs. ViT-H!

### 3. Memory Efficiency

**SAM 2 memory constraints**:
- Streaming memory bank (14 frames)
- Must fit on single GPU

**Hiera helps** with efficient multi-scale design!

---

## Ablation Studies: What Makes Hiera Fast?

### Removing "Bells-and-Whistles"

| Component Removed | Speedup | Accuracy Change |
|------------------|---------|-----------------|
| [CLS] token | +8% faster | No change |
| Positional encoding | +5% faster | +0.3% better |
| Pre-norm → Post-norm | +3% faster | +0.1% better |
| Fused transformer block | +12% faster | No change |
| **Total (Hiera)** | **+32% faster** | **+0.4% better!** |

**Key Insight**: Simpler design = faster AND better!

### Hierarchical vs. Flat

| Architecture | FLOPs | Params | Speed | ImageNet Top-1 |
|-------------|-------|--------|-------|----------------|
| ViT-L (flat) | 361 GFLOPs | 307M | 1.0× | 82.6% |
| Hiera-L (hierarchical) | 218 GFLOPs | 212M | 1.66× | 82.8% |

**Key Insight**: Hierarchy reduces computation without hurting accuracy!

---

## ARR-COC Connection (10%)

### Hierarchical Processing = Perspectival Knowing

**Propositional** (semantic labels):
- Stage 4: Global scene understanding ("This is a kitchen")
- Stage 3: Object category ("This is a person")

**Perspectival** (spatial relationships):
- Stage 2: Object parts and their relationships ("Head connected to body")
- Stage 1: Fine spatial details ("Object boundary is here")

**Participatory** (interactive refinement):
- Multi-scale features allow flexible prompting
- User clicks → all scales contribute to mask
- Coarse-to-fine refinement

### Hierarchical Relevance Realization

**Hiera's multi-scale design** mirrors cognitive multi-level processing:

**Low-level** (Stage 1): Attention to fine details (edges, textures)
**Mid-level** (Stages 2-3): Object parts and relationships
**High-level** (Stage 4): Scene context and global meaning

**Relevance flows bottom-up** (details → context) and **top-down** (context → details)!

---

## Key Takeaways

1. **