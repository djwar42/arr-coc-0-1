# Hiera Image Encoder in SAM 2

## Overview

The **Hiera (Hierarchical) encoder** is a key architectural innovation in SAM 2 that replaces the ViT-H image encoder from SAM 1. Hiera is a hierarchical vision transformer designed to be "fast, powerful, and simple" - achieving 6× speedup over ViT while maintaining or improving accuracy.

**Key Innovation**: By pre-training with MAE (Masked Autoencoder), Hiera can strip out all the "bells-and-whistles" from complex hierarchical transformers while actually improving performance.

---

## Section 1: Hiera Architecture

### Core Design Philosophy

From [Hiera Paper](https://arxiv.org/abs/2306.00989) (arXiv:2306.00989, ICML 2023 Oral):

> "Modern hierarchical vision transformers have added several vision-specific components in the pursuit of supervised classification performance. While these components lead to effective accuracies and attractive FLOP counts, the added complexity actually makes these transformers slower than their vanilla ViT counterparts."

**The Insight**: Complex components like relative position encodings, shifted windows, and decomposed attention were added to compensate for ViTs lacking spatial biases. But with MAE pre-training, these biases are **learned** rather than hand-coded.

### Architectural Simplification

**Components Removed from MViT**:
- Relative position embeddings
- Decomposed relative position embeddings
- Kernel Q/K/V projection
- Res path in attention blocks

**What Remains - Pure Hierarchical Design**:
```python
# Hiera's simplified architecture
class HieraBlock(nn.Module):
    def __init__(self, dim, num_heads):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)  # Standard attention!
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim)  # Standard MLP

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Model Configurations

From [GitHub facebookresearch/hiera](https://github.com/facebookresearch/hiera):

| Model | Model Name | Params | IN-1K Top-1 | Speed (im/s) |
|-------|------------|--------|-------------|--------------|
| Hiera-T | `hiera_tiny_224` | ~6M | 82.8% | 2758 |
| Hiera-S | `hiera_small_224` | ~22M | 83.8% | 2211 |
| Hiera-B | `hiera_base_224` | ~51M | 84.5% | 1556 |
| Hiera-B+ | `hiera_base_plus_224` | ~70M | 85.2% | 1247 |
| Hiera-L | `hiera_large_224` | ~214M | 86.1% | 531 |
| Hiera-H | `hiera_huge_224` | ~672M | 86.9% | 274 |

**Speed measured on A100 fp16** - significantly faster than ViT counterparts.

### Token Reordering for Efficiency

Hiera uses a **token reordering** strategy at the start of the network:

```python
# From hiera_utils.py
class Roll(nn.Module):
    """Reorders tokens for efficient hierarchical processing"""
    def forward(self, x):
        # Reorder tokens to optimize memory access patterns
        return reordered_x

class Unroll(nn.Module):
    """Restores original token order for downstream tasks"""
    def forward(self, x):
        return original_order_x
```

**Important**: Tokens are NOT in spatial order by default. Use `return_intermediates=True` for downstream tasks:

```python
output, intermediates = model(x, return_intermediates=True)
```

---

## Section 2: Hierarchical Design

### Multi-Stage Architecture

Unlike flat ViT which maintains constant resolution and features, Hiera follows a **hierarchical/pyramidal** design:

```
Input Image (224×224×3)
    ↓
Stage 1: High resolution, few features
    - Resolution: 56×56 (for 224 input)
    - Channels: 96 (Hiera-B)
    ↓
Stage 2: Medium resolution
    - Resolution: 28×28
    - Channels: 192
    ↓
Stage 3: Lower resolution
    - Resolution: 14×14
    - Channels: 384
    ↓
Stage 4: Final embedding
    - Resolution: 7×7
    - Channels: 768
```

### Why Hierarchical is Better

From the Hiera paper:

> "Vision transformers like ViT use the same spatial resolution and number of features throughout the whole network. But this is inefficient: the early layers don't need that many features, and the later layers don't need that much spatial resolution."

**Efficiency Benefits**:
1. **Early layers**: Few features needed → low compute
2. **Late layers**: Low resolution needed → low memory
3. **Multi-scale features**: Natural FPN-like structure for detection/segmentation

### Comparison with Other Hierarchical ViTs

| Model | Added Components | Result |
|-------|------------------|--------|
| **Swin** | Shifted windows, relative position bias | Slower actual inference |
| **MViT** | Pooling attention, decomposed position | Complex implementation |
| **Hiera** | **None** (with MAE pre-training) | Simpler AND faster |

### Resolution Reduction Strategy

Hiera uses **Mask Unit Attention** for efficient downsampling:

```python
# Pooling between stages
class MaskUnitAttention(nn.Module):
    """Attention with built-in spatial reduction"""
    def __init__(self, dim, num_heads, q_stride=(2, 2)):
        self.q_stride = q_stride
        # Q undergoes pooling, K/V stay same

    def forward(self, x):
        # Pool Q to reduce resolution
        q = self.pool(self.q_proj(x))  # (H/2, W/2)
        k = self.k_proj(x)              # (H, W)
        v = self.v_proj(x)              # (H, W)

        # Cross-resolution attention
        return self.attention(q, k, v)  # Output: (H/2, W/2)
```

---

## Section 3: Speed Improvements

### 6× Faster Than ViT-H

SAM 2 with Hiera achieves **6× speedup** over SAM 1 with ViT-H on image segmentation:

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714) (arXiv:2408.00714):

> "SAM 2 achieves higher accuracy (58.9 mIoU with 1 click) than SAM (58.1 mIoU with 1 click), without using any extra data and while being 6× faster."

### Sources of Speedup

**1. Hierarchical vs Flat Processing**
```
ViT-H (SAM 1):
- 64×64 tokens at all 32 layers
- Total attention cost: O(N² × L) where N=4096

Hiera (SAM 2):
- Stage 1: 56×56 tokens, 2 layers
- Stage 2: 28×28 tokens, 3 layers
- Stage 3: 14×14 tokens, 16 layers
- Stage 4: 7×7 tokens, 3 layers
- Much less total computation!
```

**2. No Complex Attention Variants**
- No shifted windows (Swin complexity)
- No relative position (saves computation per layer)
- Standard nn.MultiheadAttention (highly optimized)

**3. Better Memory Access Patterns**
- Token reordering improves cache efficiency
- Hierarchical structure reduces memory bandwidth

### Benchmark Results

From [GitHub facebookresearch/hiera](https://github.com/facebookresearch/hiera):

**Image Classification (ImageNet-1K)**:
| Model | Top-1 | Speed (im/s) | vs ViT |
|-------|-------|--------------|--------|
| ViT-B | 82.9% | ~1000 | 1× |
| Hiera-B | 84.5% | 1556 | **1.5×** |
| ViT-H | 86.9% | ~150 | 1× |
| Hiera-H | 86.9% | 274 | **1.8×** |

**Video Classification (Kinetics-400)**:
| Model | Top-1 | Speed (clip/s) |
|-------|-------|----------------|
| Hiera-B (16×224) | 84.0% | 133.6 |
| Hiera-L (16×224) | 87.3% | 40.8 |
| Hiera-H (16×224) | 87.8% | 20.9 |

### Optimizations Available

**PyTorch 2.0+ Scaled Dot Product Attention**:
```python
# Hiera automatically uses this if available
torch.nn.functional.scaled_dot_product_attention()

# Results in even faster inference than published numbers
```

---

## Section 4: Multi-Scale Features

### Natural Feature Pyramid

Hiera's hierarchical design produces multi-scale features naturally:

```python
# Extract features at multiple scales
output, intermediates = model(x, return_intermediates=True)

# intermediates contains features from each stage:
# Stage 1: (B, 56, 56, 96)   - Fine details
# Stage 2: (B, 28, 28, 192)  - Medium features
# Stage 3: (B, 14, 14, 384)  - Semantic features
# Stage 4: (B, 7, 7, 768)    - Global context
```

### Integration with SAM 2 Decoder

SAM 2's mask decoder uses multi-scale features from Hiera:

```python
# SAM 2 feature extraction
class SAM2ImageEncoder(nn.Module):
    def __init__(self):
        self.hiera = HieraBackbone()
        self.fpn_neck = FPN(...)  # Feature Pyramid Network

    def forward(self, x):
        # Get hierarchical features
        features = self.hiera(x, return_intermediates=True)

        # Combine scales with FPN
        multi_scale = self.fpn_neck(features)

        return multi_scale
```

### Benefits for Segmentation

**Why Multi-Scale Matters**:
1. **Fine boundaries**: High-resolution early features
2. **Object recognition**: Mid-level semantic features
3. **Global context**: Low-resolution late features
4. **Efficient decoding**: Only process what's needed at each scale

### SAM 2 Resolution Stages

In SAM 2, Hiera produces features at these resolutions:

```
Input: 1024×1024 (SAM 2 default)
    ↓
Stage 1: 256×256 resolution
Stage 2: 128×128 resolution
Stage 3: 64×64 resolution (main features)
Stage 4: 32×32 resolution (final embedding)
```

The mask decoder primarily uses Stage 3 (64×64) features, matching SAM 1's output resolution.

---

## Section 5: Pre-Training with MAE

### MAE: The Key Enabler

From the Hiera paper:

> "By pretraining with a strong visual pretext task (MAE), we can strip out all the bells-and-whistles from a state-of-the-art multi-stage vision transformer without losing accuracy."

**MAE (Masked Autoencoder)** teaches spatial reasoning that other architectures hard-code:
- Relative positions → Learned from reconstruction
- Local vs global attention → Learned from context
- Scale awareness → Learned from multi-scale masking

### MAE Pre-training Process

```python
# Hiera MAE pre-training
class HieraMAE(nn.Module):
    def __init__(self):
        self.encoder = Hiera()
        self.decoder = MAEDecoder()

    def forward(self, x, mask_ratio=0.6):
        # Mask random patches (60% for images, 90% for video)
        visible, mask = self.random_mask(x, mask_ratio)

        # Encode visible patches
        latent = self.encoder(visible)

        # Decode to reconstruct
        pred = self.decoder(latent, mask)

        # Loss: reconstruct masked patches
        # Note: Normalized pixel targets!
        loss = self.reconstruction_loss(pred, x, mask)

        return loss, pred
```

### Mask Ratios

- **Images**: 60% mask ratio
- **Video**: 90% mask ratio (temporal redundancy)

### Pre-trained Checkpoints

From [GitHub facebookresearch/hiera](https://github.com/facebookresearch/hiera):

**ImageNet-1K MAE**:
```python
# Load MAE pre-trained model
model = torch.hub.load("facebookresearch/hiera",
                       model="hiera_base_224",
                       checkpoint="mae_in1k")

# Or with decoder for continued pre-training
mae_model = torch.hub.load("facebookresearch/hiera",
                           model="mae_hiera_base_224",
                           checkpoint="mae_in1k")
```

**Kinetics-400 MAE (Video)**:
```python
model = torch.hub.load("facebookresearch/hiera",
                       model="hiera_base_16x224",
                       checkpoint="mae_k400")
```

### Normalized Pixel Targets

**Important Implementation Detail**:

Hiera MAE uses **normalized pixel loss** - patches are normalized before prediction:

```python
def get_pixel_label_2d(self, x, mask):
    # Normalize each patch independently
    patches = patchify(x)  # (B, N, P*P*3)
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True)
    normalized = (patches - mean) / (var + 1e-6).sqrt()
    return normalized
```

To visualize predictions, unnormalize using ground truth.

---

## Section 6: Integration with Memory System

### SAM 2 Memory Architecture

Hiera encoder integrates with SAM 2's streaming memory system for video:

```python
class SAM2VideoPredictor(nn.Module):
    def __init__(self):
        self.image_encoder = Hiera()
        self.memory_attention = MemoryAttention()
        self.memory_bank = MemoryBank()
        self.mask_decoder = MaskDecoder()

    def forward(self, frame, memory_state):
        # Encode current frame with Hiera
        frame_features = self.image_encoder(frame)

        # Attend to memory of previous frames
        attended_features = self.memory_attention(
            query=frame_features,
            memory=memory_state
        )

        # Decode mask
        mask = self.mask_decoder(attended_features)

        # Update memory with current frame
        self.memory_bank.add(frame_features)

        return mask
```

### Per-Frame Encoding Efficiency

Hiera's speed is critical for video:

**SAM 1 (ViT-H)**: ~100ms per frame encoding
**SAM 2 (Hiera)**: ~17ms per frame encoding

This enables **44 FPS** real-time video segmentation on A100.

### Multi-Scale Features for Memory

Memory attention uses Hiera's hierarchical features:

```python
class MemoryAttention(nn.Module):
    def forward(self, current_features, memory):
        # Attend at multiple scales
        attended = []
        for scale_features in current_features:
            scale_memory = memory.get_scale(scale_features.resolution)
            scale_attended = self.cross_attention(
                query=scale_features,
                key=scale_memory,
                value=scale_memory
            )
            attended.append(scale_attended)

        return attended
```

### Memory Bank Design

From the SAM 2 paper:

```python
class MemoryBank:
    def __init__(self, max_frames=6):
        self.recent_frames = []  # FIFO queue
        self.max_frames = max_frames

    def add(self, frame_features):
        self.recent_frames.append(frame_features)
        if len(self.recent_frames) > self.max_frames:
            self.recent_frames.pop(0)  # Remove oldest
```

**Benefits**:
- Temporal consistency across frames
- Handles occlusions (object temporarily hidden)
- Real-time streaming (no need for full video in memory)

---

## Section 7: ARR-COC Integration

### Using Hiera Features for Training

For ARR-COC training with hierarchical features:

```python
class HieraFeatureExtractor(nn.Module):
    """Extract multi-scale features from Hiera for ARR-COC"""
    def __init__(self, hiera_model):
        super().__init__()
        self.hiera = hiera_model

        # Freeze encoder during training (optional)
        for param in self.hiera.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Get all intermediate features
        output, intermediates = self.hiera(
            images,
            return_intermediates=True
        )

        # Return multi-scale features
        return {
            'fine': intermediates[0],    # High resolution
            'medium': intermediates[1],   # Medium features
            'coarse': intermediates[2],   # Semantic features
            'global': output              # Final embedding
        }
```

### Fine-Tuning Strategy

```python
# Load pre-trained Hiera
hiera = torch.hub.load("facebookresearch/hiera",
                       model="hiera_base_224",
                       checkpoint="mae_in1k_ft_in1k")

# For segmentation fine-tuning:
# 1. Freeze early stages (keep learned representations)
for stage in hiera.stages[:2]:
    for param in stage.parameters():
        param.requires_grad = False

# 2. Fine-tune later stages + add task head
# This preserves pre-trained features while adapting to task
```

### Speed Considerations for ARR-COC

**Training**:
- Hiera's efficiency reduces training time significantly
- Multi-scale features available without extra computation
- MAE pre-training provides strong initialization

**Inference**:
- 6× faster than ViT-H baseline
- Real-time video processing possible
- Efficient memory usage with hierarchical design

### Integration with Memory-Efficient Training

```python
# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class HieraWithCheckpointing(nn.Module):
    def forward(self, x):
        for stage in self.stages:
            # Checkpoint each stage to save memory
            x = checkpoint(stage, x)
        return x
```

---

## Sources

**Papers**:
- [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/abs/2306.00989) - arXiv:2306.00989 (ICML 2023 Oral, accessed 2025-11-20)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - arXiv:2408.00714 (accessed 2025-11-20)

**GitHub Repositories**:
- [facebookresearch/hiera](https://github.com/facebookresearch/hiera) - Official Hiera implementation
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) - SAM 2 with Hiera encoder

**Additional References**:
- [Hugging Face Hiera Models](https://huggingface.co/facebook/hiera-base-224-mae-hf)
- [Hugging Face SAM 2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/sam2)

**Source Document**:
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - SAM 2 architecture section (lines 696-745)

---

## Key Takeaways

1. **Simplicity wins**: MAE pre-training eliminates need for complex architectural components
2. **Speed matters**: 6× faster enables real-time video segmentation
3. **Multi-scale is natural**: Hierarchical design provides FPN-like features automatically
4. **MAE is crucial**: Pre-training teaches spatial biases that were previously hard-coded
5. **Memory integration**: Hiera's efficiency enables streaming memory for video
