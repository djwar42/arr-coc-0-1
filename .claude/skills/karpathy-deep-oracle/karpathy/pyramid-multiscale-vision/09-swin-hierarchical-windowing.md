# Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows

## Overview

Swin Transformer (Shifted Window Transformer) is a hierarchical vision transformer that serves as a general-purpose backbone for computer vision tasks. Introduced by Liu et al. (Microsoft Research Asia, ICCV 2021), it addresses key challenges in adapting transformers from NLP to vision: large variations in visual entity scales and high pixel resolution compared to text tokens.

**Key Innovation**: A hierarchical pyramid architecture with **shifted window-based self-attention** that limits computation to non-overlapping local windows while enabling cross-window connections through window shifting.

**Why Dominant**: Swin brought CNN-like inductive biases (locality, hierarchy) to vision transformers while maintaining linear computational complexity relative to image size, making it compatible with dense prediction tasks (detection, segmentation) where standard ViT struggles.

From [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (Liu et al., ICCV 2021, accessed 2025-01-31):
- 87.3% top-1 accuracy on ImageNet-1K
- 58.7 box AP and 51.1 mask AP on COCO test-dev
- 53.5 mIoU on ADE20K semantic segmentation
- Surpasses previous SOTA by +2.7 box AP on COCO

From [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) (Liu et al., CVPR 2022, accessed 2025-01-31):
- Scales to 3 billion parameters (largest dense vision model at the time)
- 84.0% top-1 accuracy on ImageNet-V2
- Handles images up to 1,536×1,536 resolution

## Swin Architecture: 4-Stage Hierarchical Pyramid

Swin Transformer builds hierarchical feature maps by merging image patches in deeper layers, creating a pyramid structure similar to CNNs (ResNet, VGG).

### Stage-by-Stage Design

**Stage 1 (High Resolution):**
- Input: H×W×3 image
- Patch Embedding: 4×4 patches → H/4 × W/4 tokens
- Initial channels: C = 96
- Feature map: 56×56 (for 224×224 input)
- Swin Transformer blocks with W-MSA and SW-MSA

**Stage 2 (Medium Resolution):**
- Patch Merging: 2×2→1 spatial downsampling
- Channel doubling: 96 → 192
- Feature map: 28×28
- Dimensions: H/8 × W/8 × 2C

**Stage 3 (Low Resolution):**
- Patch Merging: 192 → 384 channels
- Feature map: 14×14
- Dimensions: H/16 × W/16 × 4C

**Stage 4 (Lowest Resolution):**
- Patch Merging: 384 → 768 channels
- Feature map: 7×7
- Dimensions: H/32 × W/32 × 8C

### Patch Embedding and Merging

**Initial Patch Embedding (Stage 1):**
```
Input: 224×224×3 image
Patch size: 4×4 pixels
Output: 56×56×96 feature map
Operation: Linear projection of flattened patches
```

**Patch Merging (Between Stages):**
```python
# Conceptual illustration
# Input: H×W×C feature map
# Group 2×2 neighboring patches:
# [x0, x1]  →  Concatenate → [x0, x1, x2, x3] (4C channels)
# [x2, x3]
#
# Linear layer: 4C → 2C (reduces channels while downsampling spatially)
# Output: H/2 × W/2 × 2C
```

Patch merging creates hierarchical representations while doubling channel capacity, mimicking CNN downsampling with max/avg pooling.

From [Official Swin Transformer Implementation](https://github.com/microsoft/Swin-Transformer) (accessed 2025-01-31):
- Typical model sizes: Swin-T (28M params), Swin-S (50M), Swin-B (88M), Swin-L (197M)
- FLOPs scale linearly with image resolution (unlike quadratic scaling in ViT)

## Shifted Window Attention: W-MSA and SW-MSA

The core innovation enabling efficiency: compute self-attention within local windows, then shift windows to enable cross-window connections.

### Window-Based Multi-Head Self-Attention (W-MSA)

**Standard Self-Attention (Global):**
- Complexity: O((HW)²) for H×W feature map
- Problem: Prohibitively expensive for high-resolution images
- Example: 56×56 = 3136 tokens → 9.8M query-key pairs

**Window-Based Self-Attention:**
- Partition feature map into M×M non-overlapping windows
- Typical window size: 7×7 patches
- Complexity per window: O(M²)
- Total complexity: O(HW·M²) — **linear in image size**

**W-MSA Process:**
```
Feature map: 56×56 patches
Window size: 7×7
Number of windows: (56/7) × (56/7) = 8×8 = 64 windows
Tokens per window: 7×7 = 49

Complexity:
- Global attention: O(56² × 56²) = O(9.8M × 3136) ≈ 30B operations
- Window attention: O(64 windows × 49² patches) ≈ 154K operations per layer
- Speedup: ~200,000x reduction in quadratic term
```

### Shifted Window Multi-Head Self-Attention (SW-MSA)

**Problem with W-MSA**: No communication between windows — features are isolated within 7×7 regions, limiting receptive field growth.

**Solution: Shifted Window Scheme**
- Layer L: Regular window partitioning (W-MSA)
- Layer L+1: Shift windows by (M/2, M/2) pixels (SW-MSA)
- Effect: Windows from layer L+1 span boundaries of layer L windows

**Shifting Mechanism:**
```
Layer L (W-MSA):                 Layer L+1 (SW-MSA):
┌───┬───┬───┬───┐                ┌─┬─────┬─────┬──┐
│ A │ A │ B │ B │                │ │     │     │  │
├───┼───┼───┼───┤    Shift by    ├─┼─────┼─────┼──┤
│ A │ A │ B │ B │  ←  (3,3)  ←   │ │  A' │  B' │  │
├───┼───┼───┼───┤                ├─┼─────┼─────┼──┤
│ C │ C │ D │ D │                │ │  C' │  D' │  │
├───┼───┼───┼───┤                ├─┼─────┼─────┼──┤
│ C │ C │ D │ D │                │ │     │     │  │
└───┴───┴───┴───┘                └─┴─────┴─────┴──┘

Window A' contains patches from regions A, B, C, D of Layer L
→ Cross-window information flow established
```

By alternating W-MSA and SW-MSA across layers, Swin achieves:
1. **Local mixing** within windows (efficient computation)
2. **Cross-window propagation** (shifted windows connect previously isolated regions)
3. **Growing receptive field** (information spreads across entire image through successive shifts)

### Cyclic Shift + Masking for Efficient Computation

**Challenge**: After shifting, the window grid becomes irregular with windows of different sizes at the borders.

**Naive Solution**: Pad and handle irregular windows → Wastes computation on padded regions

**Efficient Solution: Cyclic Shift + Attention Masking**

From [arXiv:2103.14030](https://arxiv.org/pdf/2103.14030) (accessed 2025-01-31):

**Step 1: Cyclic Shift**
```python
# Shift feature map cyclically (like np.roll)
shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
# Now we have regular MxM windows again (but patches from different regions)
```

**Step 2: Attention Masking**
```python
# Create attention mask to prevent attention between non-adjacent patches
# Mask shape: (num_windows, M*M, M*M)
# Mask[i,j] = 0 if patches i,j should attend to each other
# Mask[i,j] = -100 (or -inf) if patches are from non-adjacent regions

# Apply mask in softmax:
attention_scores = (Q @ K.T) / sqrt(d) + attention_mask
attention_weights = softmax(attention_scores)  # Masked positions → ~0 weight
```

**Step 3: Reverse Cyclic Shift**
```python
# After attention, shift back to original positions
x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
```

**Computational Benefit**:
- Regular window partitioning maintained (efficient batch computation)
- No padding overhead
- Masking prevents spurious attention across cyclically wrapped boundaries
- Same complexity as W-MSA: O(HW·M²)

From [Swin Transformer implementation guides](https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03) (accessed 2025-01-31):
- Typical shift size: M/2 = 3 or 4 (for M=7 or 8 window size)
- Attention mask computed once and reused across layers
- Cyclic shift enables efficient batched computation across all windows

### Alternating W-MSA and SW-MSA Blocks

**Swin Transformer Block Structure:**
```
Block 2l (even layers):
  LayerNorm → W-MSA → Residual
  LayerNorm → MLP → Residual

Block 2l+1 (odd layers):
  LayerNorm → SW-MSA → Residual
  LayerNorm → MLP → Residual
```

**Information Flow**:
- Depth 2: Local attention (7×7 windows)
- Depth 4: Cross-window connections established (effective 13×13 receptive field)
- Depth 8: 25×25 effective receptive field
- By final layer: Global receptive field achieved through successive shifts

**Computational Complexity Comparison**:
```
Input: H×W feature map, Window size M×M

Global Self-Attention (ViT):
  Ω(MSA) = 4HWC² + 2(HW)²C

Window-Based Self-Attention (W-MSA):
  Ω(W-MSA) = 4HWC² + 2M²HWC

Shifted Window Attention (SW-MSA):
  Ω(SW-MSA) = 4HWC² + 2M²HWC  (same as W-MSA, masking is essentially free)

For H=W=112, C=128, M=7:
  MSA: 19.7B FLOPs
  W-MSA: 1.6B FLOPs (~12x speedup)
```

## Swin-v2 Improvements: Scaling to Billions of Parameters

Swin Transformer V2 (CVPR 2022) introduced three key improvements for training stability and scalability:

### 1. Residual-Post-Norm + Cosine Attention

**Problem in Swin v1**: Training instability when scaling to large models (1B+ parameters), especially with high-resolution images.

**Solution 1: Res-Post-Norm**
```python
# Swin v1 (Pre-Norm):
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))

# Swin v2 (Post-Norm):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```
- Post-norm stabilizes activations during early training
- Prevents amplitude explosion in deep networks

**Solution 2: Scaled Cosine Attention**
```python
# Swin v1 (Dot-Product Attention):
Attention = softmax(Q @ K^T / sqrt(d))

# Swin v2 (Cosine Attention):
Attention = softmax(τ * cos(Q, K))
where cos(Q, K) = (Q @ K^T) / (||Q|| * ||K||)
      τ is learnable temperature parameter
```

**Benefits**:
- Cosine similarity bounds attention logits between [-1, 1]
- Prevents attention collapse (all queries attend to same key)
- Temperature τ adapts to different layers' needs

### 2. Log-Spaced Continuous Position Bias (CPB)

**Problem**: Swin v1 uses relative position bias learned for specific resolution. Fine-tuning on higher resolutions requires extrapolation (poor performance).

**Swin v1 Relative Position Bias**:
```python
# Learned bias table for relative positions within window
bias_table = nn.Parameter(torch.zeros((2*M-1) * (2*M-1), num_heads))
# Index by relative position: bias_table[relative_position_index]
```

**Swin v2 Continuous Position Bias**:
```python
# Meta-network generates bias from continuous relative coordinates
def CPB(relative_coords):
    # relative_coords: continuous values, any resolution
    # Apply log-spacing transformation
    log_coords = sign(relative_coords) * log(1 + |relative_coords|)

    # Small MLP maps log-spaced coords to bias
    bias = MLP(log_coords)  # Output: bias per head
    return bias
```

**Advantages**:
- Smooth interpolation/extrapolation across resolutions
- Pre-train 256×256 → Fine-tune 1536×1536 without retraining position embeddings
- Log-spacing: relative positions close to 0 are more important (local context)

From [Swin Transformer V2 paper](https://arxiv.org/abs/2111.09883) (accessed 2025-01-31):
- Enables training with images up to 1,536×1,536 resolution
- Successfully transfers from 256×256 pre-training to 1536×1536 fine-tuning

### 3. SimMIM Self-Supervised Pre-Training

**Goal**: Reduce labeled data requirements for large-scale training.

**SimMIM (Simple Masked Image Modeling)**:
```python
# Mask random patches (40% masking ratio typical)
masked_image = mask_patches(image, mask_ratio=0.4)

# Predict raw pixel values of masked patches
predictions = SwinTransformerV2(masked_image)

# L1 loss on masked regions only
loss = L1_loss(predictions[masked_patches], original_pixels[masked_patches])
```

**Key Differences from MAE (Masked Autoencoder)**:
- Simpler: Direct pixel prediction (no separate decoder)
- Lighter masking: 40% vs 75% in MAE
- Raw pixel targets (no tokenization/quantization)

**Results**:
- 40x less labeled data than Google's billion-level models
- 40x less training time
- Comparable or better performance on downstream tasks

## Performance Benchmarks

### ImageNet-1K Classification

From [Swin Transformer GitHub](https://github.com/microsoft/Swin-Transformer) and papers (accessed 2025-01-31):

| Model | Params | FLOPs | Top-1 Acc | Top-5 Acc |
|-------|--------|-------|-----------|-----------|
| Swin-T | 28M | 4.5G | 81.3% | 95.5% |
| Swin-S | 50M | 8.7G | 83.0% | 96.2% |
| Swin-B | 88M | 15.4G | 83.5% | 96.5% |
| Swin-L | 197M | 34.5G | **87.3%** | 98.0% |

**ImageNet-V2 (Swin-v2)**:
- Swin-V2-G (3B params): **84.0% top-1 accuracy**
- Previous SOTA: ~82%

### COCO Object Detection

From [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) (accessed 2025-01-31):

**Cascade Mask R-CNN + Swin Backbone:**
| Backbone | Box AP | Mask AP | Params | FLOPs |
|----------|--------|---------|--------|-------|
| ResNet-50 | 46.9 | 40.3 | 82M | 739G |
| ResNeXt-101 | 48.1 | 41.4 | 140M | 972G |
| Swin-T | 50.5 | 43.7 | 86M | 745G |
| Swin-S | 51.9 | 45.0 | 107M | 838G |
| Swin-B | 51.9 | 45.0 | 145M | 982G |
| Swin-L | **58.7** | **51.1** | 284M | 1382G |

**Improvement over SOTA**:
- +2.7 box AP on COCO test-dev
- +2.6 mask AP
- Surpasses previous best by significant margin

### ADE20K Semantic Segmentation

| Backbone | mIoU (val) | mIoU (test) | Params |
|----------|------------|-------------|--------|
| ResNet-101 | 44.9 | - | 86M |
| Swin-T | 46.1 | 45.8 | 60M |
| Swin-S | 49.3 | 48.3 | 81M |
| Swin-B | 48.1 | 47.6 | 121M |
| Swin-L | **53.5** | **52.1** | 234M |

**Improvement**: +3.2 mIoU over previous SOTA on ADE20K

### Computational Efficiency

**FLOPs vs Accuracy Trade-off**:
- Swin-B achieves ViT-L accuracy with **3x fewer FLOPs**
- Swin-L achieves best-in-class accuracy with comparable FLOPs to other large models

**Inference Speed** (from various benchmarks):
- Swin-T: ~480 images/sec (V100 GPU, batch size 128)
- Swin-B: ~210 images/sec
- Comparable or faster than equivalent ViT models due to linear complexity

**Memory Efficiency**:
- Window attention reduces memory footprint significantly
- Can process higher resolution images than global attention ViTs

From [Microsoft Research blog](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/articles/five-reasons-to-embrace-transformer-in-computer-vision/) (accessed 2025-01-31):
- Swin's hierarchical design makes it compatible with existing dense prediction frameworks (FPN, U-Net architectures)
- Direct drop-in replacement for CNN backbones in detection/segmentation pipelines

## Implementation Details

### Key Hyperparameters

From [Official implementation](https://github.com/microsoft/Swin-Transformer) (accessed 2025-01-31):

**Model Configurations**:
```
Swin-T: depths=[2,2,6,2],  num_heads=[3,6,12,24],  embed_dim=96
Swin-S: depths=[2,2,18,2], num_heads=[3,6,12,24],  embed_dim=96
Swin-B: depths=[2,2,18,2], num_heads=[4,8,16,32],  embed_dim=128
Swin-L: depths=[2,2,18,2], num_heads=[6,12,24,48], embed_dim=192
```

**Window Configuration**:
- Window size M: 7 (most common), or 8 for high-res images
- Shift size: M//2 = 3 or 4
- MLP ratio: 4 (hidden dim = 4 × embed_dim)

**Training (ImageNet-1K)**:
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.05)
- Learning rate: 1e-3 with cosine decay
- Batch size: 1024 (across 8 GPUs)
- Epochs: 300
- Augmentation: RandAugment, Mixup, CutMix, Random Erasing
- Regularization: Stochastic Depth (drop_path)

**Transfer Learning**:
- ImageNet-21K pre-training → Fine-tune on ImageNet-1K
- Higher resolution fine-tuning (384×384 or 512×512)
- Bicubic interpolation for position bias if needed (v1), CPB handles naturally (v2)

### PyTorch Implementation Sketch

From [PyTorch documentation](https://docs.pytorch.org/vision/main/models/swin_transformer.html) and tutorials (accessed 2025-01-31):

```python
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (M, M)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape  # N = M*M tokens per window
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        attn = attn + self.relative_position_bias  # (B, num_heads, N, N)

        # Apply attention mask (for shifted windows)
        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # Broadcast over heads

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA or SW-MSA"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, mask_matrix):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows: (B, H, W, C) -> (B*num_windows, M, M, C)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

**Helper Functions**:
```python
def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows
    Args:
        x: (B, H, W, C)
        window_size: int (M)
    Returns:
        windows: (B*num_windows, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse of window_partition
    Args:
        windows: (B*num_windows, M, M, C)
        window_size: int (M)
        H, W: original feature map height, width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

**Usage**:
```python
from torchvision.models import swin_t, swin_b, swin_v2_b

# Load pre-trained Swin Transformer
model = swin_b(weights='IMAGENET1K_V1')
model.eval()

# Inference
image = torch.randn(1, 3, 224, 224)
output = model(image)  # (1, 1000) logits
```

## Practical Considerations

### When to Use Swin Transformer

**Excellent for**:
- **Object detection**: Hierarchical features + FPN-compatible design
- **Semantic segmentation**: Multi-scale representations ideal for dense prediction
- **Instance segmentation**: Mask R-CNN + Swin backbone achieves SOTA
- **High-resolution images**: Linear complexity enables processing large images
- **Video understanding**: Temporal extension (Video Swin Transformer)

**Compared to ViT**:
- Swin: Better for dense prediction, more efficient for high-res images
- ViT: Simpler architecture, better for classification-only tasks with aggressive augmentation

**Compared to CNNs (ResNet, ConvNeXt)**:
- Swin: Larger receptive field through self-attention, better long-range dependencies
- CNNs: Faster inference on edge devices, better inductive bias for small datasets

### Fine-Tuning Tips

From [Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/swin) (accessed 2025-01-31):

**For Downstream Tasks**:
1. **Classification**: Replace classification head, fine-tune with lower learning rate (1e-5 to 1e-4)
2. **Detection/Segmentation**: Use Swin as backbone in existing frameworks (MMDetection, Detectron2)
3. **Higher Resolution**: Swin-v2's CPB enables smooth transfer; Swin-v1 requires careful position bias handling

**Data Efficiency**:
- ImageNet-21K pre-training significantly improves small-dataset performance
- Self-supervised SimMIM pre-training reduces labeled data needs

**Compute Requirements**:
- Swin-T trainable on single GPU (8GB VRAM) with batch size 32-64
- Swin-B/L requires distributed training (4-8 GPUs typical)
- Gradient checkpointing reduces memory at cost of 20-30% slower training

## Sources

**Primary Papers:**
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) - Liu et al., ICCV 2021 (accessed 2025-01-31)
- [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) - Liu et al., CVPR 2022 (accessed 2025-01-31)

**Official Implementation:**
- [Microsoft Swin-Transformer GitHub](https://github.com/microsoft/Swin-Transformer) (accessed 2025-01-31)

**Technical Guides:**
- [Swin Transformer in Depth: Architecture and PyTorch Implementation](https://medium.com/@ovularslan/swin-transformer-in-depth-architecture-and-pytorch-implementation-a11e44d65aef) - Medium (accessed 2025-01-31)
- [Swin (Shifted Window Transformer) Explained](https://www.lightly.ai/blog/swin-transformer) - Lightly AI (accessed 2025-01-31)
- [PyTorch Swin Transformer Documentation](https://docs.pytorch.org/vision/main/models/swin_transformer.html) (accessed 2025-01-31)

**Additional References:**
- [Five Reasons to Embrace Transformer in Computer Vision](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/articles/five-reasons-to-embrace-transformer-in-computer-vision/) - Microsoft Research (accessed 2025-01-31)
- [Swin Transformer from Scratch in PyTorch](https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03) - Python in Plain English (accessed 2025-01-31)
