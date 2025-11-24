# ViT-H Image Encoder: SAM's Vision Backbone

## Overview

The **Vision Transformer Huge (ViT-H)** serves as the image encoder backbone for SAM (Segment Anything Model), providing the critical visual feature extraction that enables SAM's remarkable segmentation capabilities. Pre-trained using Masked Autoencoder (MAE) methodology, ViT-H represents one of the largest and most capable vision transformers deployed in production segmentation systems.

**Key Specifications:**
- **Parameters**: 636M (632M in some references)
- **Architecture**: ViT-H/16 (Huge variant, 16x16 patches)
- **Input Resolution**: 1024 x 1024 x 3 RGB
- **Output**: 64 x 64 x 256 feature embedding
- **Downscaling Factor**: 16x spatial reduction
- **Pre-training**: MAE on ImageNet-1K

The ViT-H encoder transforms high-resolution images into dense feature representations that capture both local details and global context, essential for accurate segmentation across diverse domains.

---

## Section 1: ViT-H Architecture

### Transformer Configuration

The ViT-H encoder follows the standard Vision Transformer architecture with "Huge" scaling:

```python
# ViT-H/16 Configuration
config = {
    "num_layers": 32,           # Transformer blocks
    "hidden_dim": 1280,         # Embedding dimension
    "num_heads": 16,            # Attention heads
    "mlp_ratio": 4.0,           # MLP expansion ratio
    "patch_size": 16,           # Patch dimensions
    "image_size": 1024,         # Input resolution
    "num_patches": 64 * 64,     # 4096 patches total
    "total_params": "636M"      # Total parameters
}
```

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) (lines 569-585):
> The image encoder is based on a standard Vision Transformer (ViT) pre-trained by MAE. Specifically, we use the ViT-H/16 variant.

### Architecture Components

**1. Patch Embedding Layer**
- Converts 16x16 pixel patches to 1280-dimensional tokens
- Linear projection: (16 x 16 x 3) -> 1280

**2. Positional Embeddings**
- Learnable position encodings added to each patch token
- Sine-cosine initialization option
- 4096 position embeddings (64 x 64 grid)

**3. Transformer Encoder Blocks (32 layers)**
Each block contains:
- Multi-head self-attention (16 heads)
- Layer normalization (pre-norm)
- MLP with GELU activation
- Residual connections

**4. Output Projection**
- Projects 1280-d features to 256-d for decoder compatibility
- Maintains spatial structure: 64 x 64 x 256

### Processing Flow

```
Image (1024 x 1024 x 3)
    |
    v
Patch Embedding (16x16 patches)
    |
    v
Patches (64 x 64 x 1280)
    |
    v
+ Positional Embeddings
    |
    v
32 Transformer Blocks
    |-- Windowed Self-Attention (14x14)
    |-- Global Attention (every 8th layer)
    |-- MLP (4x expansion)
    |
    v
Output Projection
    |
    v
Feature Map (64 x 64 x 256)
```

### SAM-Specific Modifications

SAM's ViT-H includes adaptations from standard ViT:

**Windowed Attention:**
- Uses 14x14 window attention for efficiency
- Global attention blocks interspersed (every 4-8 layers)
- Reduces quadratic complexity while maintaining global context

**Relative Position Bias:**
- Adds relative position information to attention scores
- Improves spatial understanding for segmentation

From [Medical SAM Adapter Paper](https://www.sciencedirect.com/science/article/pii/S1361841525000945):
> The ViT-H/16 variant employs 14x14 windowed attention with 4 global attention blocks.

---

## Section 2: MAE Pre-Training

### Masked Autoencoder Methodology

The MAE pre-training approach, introduced by He et al. (2022), enables efficient training of large vision transformers through:

**Core Principle:**
- Mask 75% of image patches randomly
- Encoder processes only visible patches (25%)
- Lightweight decoder reconstructs full image
- Learn representations through reconstruction task

From [Hugging Face vit-mae-huge](https://huggingface.co/facebook/vit-mae-huge):
> During pre-training, one randomly masks out a high portion (75%) of the image patches. First, the encoder is used to encode the visual patches.

### Pre-training Configuration

```python
# MAE Pre-training Settings
mae_config = {
    "masking_ratio": 0.75,        # 75% patches masked
    "decoder_depth": 8,           # Decoder transformer blocks
    "decoder_dim": 512,           # Decoder embedding dimension
    "reconstruction_target": "pixels",  # Normalized pixel values
    "loss": "MSE",                # Mean squared error
    "optimizer": "AdamW",
    "base_lr": 1.5e-4,
    "weight_decay": 0.05,
    "epochs": 1600,               # Long training schedule
    "dataset": "ImageNet-1K"      # 1.28M images
}
```

### Why High Masking Ratio Works

From [MAE Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf):

**Information Redundancy:**
- Images have high spatial redundancy
- Missing patches can't be trivially interpolated
- Forces learning of semantic understanding

**Efficiency Benefits:**
- Encoder processes only 25% of patches
- 3x+ training speedup
- Reduced memory consumption

**Representation Quality:**
- High masking creates challenging pretext task
- Model learns gestalt understanding
- Better transfer to downstream tasks

### MAE Architecture Details

**Asymmetric Encoder-Decoder:**
```
Visible Patches (25%)
    |
    v
ViT-H Encoder (636M params)
    |
    v
Latent Representation
    |
    + Mask Tokens (75%)
    |
    v
Lightweight Decoder (8 blocks, 512-d)
    |
    v
Reconstructed Image
```

**Key Design Decisions:**
- No mask tokens in encoder (critical for efficiency)
- Decoder only used during pre-training
- Normalized pixel reconstruction improves quality

### Training Results

From MAE experiments:
- **ImageNet-1K Fine-tuning**: 87.8% top-1 accuracy (ViT-H)
- **Linear Probing**: 73.5% (800 epochs)
- **Transfer Performance**: Outperforms supervised pre-training

---

## Section 3: Patch Embedding

### Patch Extraction Process

The patch embedding layer converts raw pixels into transformer tokens:

**Step 1: Image Partitioning**
```python
def create_patches(image, patch_size=16):
    """
    Divide image into non-overlapping patches.

    Args:
        image: (B, 3, 1024, 1024) input tensor
        patch_size: 16 pixels

    Returns:
        patches: (B, 4096, 768) patch tokens
                 4096 = 64 * 64 patches
                 768 = 16 * 16 * 3 pixels per patch
    """
    B, C, H, W = image.shape
    num_patches = (H // patch_size) * (W // patch_size)  # 4096

    # Reshape: (B, C, H, W) -> (B, num_patches, patch_size^2 * C)
    patches = image.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, num_patches, -1)

    return patches
```

**Step 2: Linear Projection**
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=1280):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 4096

        # Single conv layer acts as linear projection
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.proj(x)  # (B, 1280, 64, 64)
        x = x.flatten(2).transpose(1, 2)  # (B, 4096, 1280)
        return x
```

### Patch Statistics

**For 1024x1024 Input:**
- Patch size: 16 x 16 pixels
- Grid dimensions: 64 x 64
- Total patches: 4,096
- Pixels per patch: 768 (16 x 16 x 3)
- Embedding dimension: 1,280

**Resolution Trade-offs:**
- Smaller patches (8x8): More tokens, higher compute, finer detail
- Larger patches (32x32): Fewer tokens, faster, coarser detail
- 16x16: Optimal balance for segmentation tasks

### Learned vs Fixed Embeddings

SAM uses a **learned linear projection** rather than fixed features:

**Benefits of Learned Projection:**
- Adapts to task requirements during pre-training
- Can learn to emphasize relevant features
- End-to-end optimization with transformer

**Alternative Approaches:**
- ResNet feature maps (used in earlier methods)
- Frequency-based embeddings
- Hybrid CNN-Transformer stems

---

## Section 4: Position Encoding

### Positional Embedding Types

ViT-H uses **learnable absolute position embeddings**:

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches=4096, embed_dim=1280):
        super().__init__()
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        return x + self.pos_embed
```

### Position Encoding Methods Comparison

**1. Learnable Absolute (Used in SAM)**
```python
pos_embed = nn.Parameter(torch.zeros(1, N, D))
```
- Pros: Flexible, learns task-specific patterns
- Cons: Fixed to training resolution

**2. Sinusoidal Fixed**
```python
def get_sinusoidal_encoding(num_patches, dim):
    position = torch.arange(num_patches).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    pe = torch.zeros(num_patches, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```
- Pros: Generalizes to longer sequences
- Cons: Less flexible

**3. 2D Sinusoidal (Spatial Aware)**
```python
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size)
    grid_w = torch.arange(grid_size)
    grid = torch.meshgrid(grid_h, grid_w)
    grid = torch.stack(grid, dim=0).reshape(2, -1)

    # Apply sincos to each dimension
    pos_embed = get_1d_sincos_from_grid(embed_dim // 2, grid)
    return pos_embed
```
- Pros: Better for 2D spatial structure
- Cons: More complex

### Relative Position Bias

SAM's windowed attention uses **relative position bias**:

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

        # Compute relative position index
        coords = torch.stack(torch.meshgrid([
            torch.arange(window_size),
            torch.arange(window_size)
        ]))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.relative_position_index = relative_coords.sum(-1)

    def forward(self, x):
        # Add relative position bias to attention scores
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)

        attn = attn + relative_position_bias.permute(2, 0, 1)
        return attn
```

### Resolution Interpolation

For different input resolutions, position embeddings are interpolated:

```python
def interpolate_pos_encoding(pos_embed, new_size, old_size):
    """
    Interpolate position embeddings for different resolutions.
    """
    # Reshape to 2D grid
    pos_embed_2d = pos_embed.reshape(1, old_size, old_size, -1)
    pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)

    # Bicubic interpolation
    pos_embed_2d = F.interpolate(
        pos_embed_2d,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back
    pos_embed_new = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
    return pos_embed_new
```

---

## Section 5: Model Variants

### SAM Model Size Comparison

SAM offers three ViT encoder variants:

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) (lines 140-144):

| Model | Params | Checkpoint | Size | Speed |
|-------|--------|-----------|------|-------|
| **ViT-H** | 636M | `sam_vit_h_4b8939.pth` | 2.4 GB | Slow, best quality |
| **ViT-L** | 308M | `sam_vit_l_0b3195.pth` | 1.2 GB | Medium |
| **ViT-B** | 91M | `sam_vit_b_01ec64.pth` | 375 MB | Fast, good quality |

### Architecture Specifications

**ViT-H (Huge)**
```python
vit_h_config = {
    "embed_dim": 1280,
    "depth": 32,
    "num_heads": 16,
    "mlp_ratio": 4,
    "global_attn_indexes": [7, 15, 23, 31],  # 4 global layers
    "window_size": 14,
    "params": "636M"
}
```

**ViT-L (Large)**
```python
vit_l_config = {
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4,
    "global_attn_indexes": [5, 11, 17, 23],
    "window_size": 14,
    "params": "308M"
}
```

**ViT-B (Base)**
```python
vit_b_config = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "global_attn_indexes": [2, 5, 8, 11],
    "window_size": 14,
    "params": "91M"
}
```

### Performance vs Efficiency Trade-offs

**Quality Comparison (IoU on SA-1B test):**
- ViT-H: 0.88 average IoU
- ViT-L: 0.86 average IoU
- ViT-B: 0.83 average IoU

**Speed Comparison (A100 GPU):**
- ViT-H: ~100ms encode + 10ms decode = 110ms total
- ViT-L: ~70ms encode + 10ms decode = 80ms total
- ViT-B: ~40ms encode + 10ms decode = 50ms total

**Use Case Recommendations:**
- **ViT-H**: Research, offline processing, highest accuracy needed
- **ViT-L**: Production systems with GPU, balanced performance
- **ViT-B**: Edge deployment, real-time applications, resource-constrained

### Scaling Law Observations

From MAE experiments, larger models show consistent improvements:

```
Model Size vs ImageNet Accuracy:
ViT-B (91M):  83.3% (fine-tune)
ViT-L (308M): 84.9% (fine-tune)
ViT-H (636M): 87.8% (fine-tune)
```

**Scaling Benefits:**
- More capacity for complex patterns
- Better generalization to new domains
- Improved zero-shot transfer

**Scaling Costs:**
- Quadratic attention complexity
- Memory requirements
- Training time

---

## Section 6: Computational Requirements

### Memory and Compute Analysis

**ViT-H Forward Pass:**
```python
# Memory estimation for ViT-H
batch_size = 1
num_patches = 4096
embed_dim = 1280
num_layers = 32

# Patch embeddings: 4096 * 1280 * 4 bytes = 20.97 MB
# Attention (per layer): 4096 * 4096 * 16 heads * 4 bytes = 1.07 GB
# Total attention: 1.07 GB * 32 layers = ~34 GB (with activation checkpointing: ~2 GB)

# FLOPs estimation
attention_flops = 2 * num_patches ** 2 * embed_dim  # per layer
mlp_flops = 2 * 4 * num_patches * embed_dim ** 2    # per layer
total_flops = num_layers * (attention_flops + mlp_flops)
# ~180 GFLOPs per image
```

### GPU Memory Requirements

**Training (with gradients):**
- ViT-H: ~32 GB (batch size 1)
- ViT-L: ~16 GB (batch size 1)
- ViT-B: ~8 GB (batch size 1)

**Inference (forward only):**
- ViT-H: ~4-6 GB
- ViT-L: ~2-3 GB
- ViT-B: ~1-2 GB

### Optimization Strategies

**1. Gradient Checkpointing**
```python
# Trade compute for memory
def forward_with_checkpointing(self, x):
    for block in self.blocks:
        x = torch.utils.checkpoint.checkpoint(block, x)
    return x
```

**2. Mixed Precision Training**
```python
# FP16 for most operations
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**3. Flash Attention**
```python
# Memory-efficient attention
from flash_attn import flash_attn_func

# Standard: O(N^2) memory
# Flash: O(N) memory with tiling
attn_output = flash_attn_func(q, k, v)
```

**4. Windowed Attention (SAM uses this)**
```python
# Reduce complexity from O(N^2) to O(N * window^2)
# N = 4096, window = 14
# Standard: 4096^2 = 16.7M operations
# Windowed: 4096 * 196 = 803K operations (20x reduction)
```

### Inference Optimization

**TensorRT Optimization:**
```python
# Convert to TensorRT for 2-3x speedup
import tensorrt as trt

# Build TRT engine
builder = trt.Builder(logger)
network = builder.create_network()
# ... parse ONNX model ...
engine = builder.build_cuda_engine(network)
```

**ONNX Export:**
```python
# Export for deployment
torch.onnx.export(
    model,
    dummy_input,
    "vit_h_encoder.onnx",
    opset_version=17,
    input_names=['image'],
    output_names=['features'],
    dynamic_axes={'image': {0: 'batch'}}
)
```

### Benchmarks

**A100 GPU (40GB):**
- ViT-H encode: 95-105ms per image
- Throughput: ~10 images/second

**V100 GPU (32GB):**
- ViT-H encode: 150-180ms per image
- Throughput: ~6 images/second

**RTX 3090 (24GB):**
- ViT-H encode: 120-140ms per image
- Throughput: ~7 images/second

---

## Section 8: ARR-COC Integration

### ViT-H as Relevance Feature Extractor

The ViT-H image encoder provides the foundation for relevance-aware segmentation in ARR-COC systems. Its hierarchical attention mechanism naturally implements aspects of the 4P knowing framework:

**Propositional Knowing:**
The encoder's learned representations capture factual visual knowledge:
- Object boundaries and shapes
- Texture patterns and colors
- Spatial relationships between regions

**Procedural Knowing:**
The attention mechanism implements procedural relevance allocation:
- Windowed attention: Local feature processing
- Global attention: Long-range dependency modeling
- Layer-by-layer refinement: Progressive understanding

**Perspectival Knowing:**
Multi-head attention provides multiple "perspectives" on the same visual input:
- 16 attention heads per layer
- Each head can specialize in different feature types
- Ensemble of perspectives for robust representations

**Participatory Knowing:**
The encoder's features enable interactive segmentation:
- Features encode affordances for interaction
- Prompt injection points for user guidance
- Iterative refinement through feedback

### Integration Code Example

```python
# ARR-COC ViT-H Integration for Relevance-Guided Segmentation
import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class RelevanceAwareEncoder(nn.Module):
    """
    Wrapper for SAM's ViT-H encoder with relevance weighting.

    Implements 4P-aware feature extraction:
    - Propositional: Base visual features
    - Procedural: Attention-based processing
    - Perspectival: Multi-head diverse views
    - Participatory: User-guided relevance
    """

    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()

        # Load SAM ViT-H encoder
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder.to(device)

        # Relevance weighting module
        self.relevance_attention = nn.MultiheadAttention(
            embed_dim=256,  # SAM output dimension
            num_heads=8,
            batch_first=True
        )

        # 4P integration layers
        self.propositional_proj = nn.Linear(256, 256)
        self.participatory_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, user_prompt_embedding=None):
        """
        Extract relevance-weighted features.

        Args:
            image: (B, 3, 1024, 1024) input image
            user_prompt_embedding: Optional (B, N, 256) prompt features

        Returns:
            features: (B, 64, 64, 256) relevance-weighted features
            attention_weights: Attention maps for interpretability
        """
        # Base feature extraction (Propositional)
        features = self.image_encoder(image)  # (B, 256, 64, 64)
        B, C, H, W = features.shape

        # Reshape for attention
        features_flat = features.flatten(2).permute(0, 2, 1)  # (B, 4096, 256)

        # Apply propositional projection
        features_prop = self.propositional_proj(features_flat)

        # Participatory: User-guided relevance weighting
        if user_prompt_embedding is not None:
            # Cross-attention with user prompts
            features_attended, attn_weights = self.relevance_attention(
                query=features_prop,
                key=user_prompt_embedding,
                value=user_prompt_embedding
            )

            # Relevance gating
            relevance_scores = self.participatory_gate(features_attended)
            features_weighted = features_prop * relevance_scores
        else:
            features_weighted = features_prop
            attn_weights = None

        # Reshape back to spatial
        features_out = features_weighted.permute(0, 2, 1).view(B, C, H, W)

        return features_out, attn_weights

    def extract_perspectival_features(self, features, num_perspectives=4):
        """
        Extract multiple perspective views of the features.

        Implements perspectival knowing through feature decomposition.
        """
        B, C, H, W = features.shape
        perspective_dim = C // num_perspectives

        perspectives = []
        for i in range(num_perspectives):
            start_idx = i * perspective_dim
            end_idx = (i + 1) * perspective_dim
            perspective = features[:, start_idx:end_idx, :, :]
            perspectives.append(perspective)

        return perspectives


# Usage Example
def arr_coc_segmentation_pipeline(image_path, prompt_points):
    """
    Complete ARR-COC segmentation with ViT-H encoder.
    """
    from PIL import Image
    import numpy as np

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((1024, 1024))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0) / 255.0

    # Initialize encoder
    encoder = RelevanceAwareEncoder(
        checkpoint_path="sam_vit_h_4b8939.pth",
        device='cuda'
    )

    # Encode prompt points to embeddings
    # (In practice, use SAM's prompt encoder)
    prompt_embedding = encode_points(prompt_points)  # (1, N, 256)

    # Extract relevance-weighted features
    features, attention = encoder(
        image_tensor.cuda(),
        prompt_embedding.cuda()
    )

    # features: (1, 256, 64, 64) - Ready for mask decoder
    # attention: Interpretable relevance maps

    return features, attention
```

### Relevance Realization Through ViT-H

The ViT-H encoder naturally implements relevance realization through:

**1. Hierarchical Attention**
- Early layers: Low-level feature detection
- Middle layers: Part-whole relationships
- Late layers: Semantic understanding
- Relevance emerges through layer progression

**2. Spatial Relevance Maps**
The attention patterns reveal what the model considers relevant:
```python
def extract_attention_relevance(encoder, image):
    """Extract attention maps as relevance indicators."""
    attention_maps = []

    # Hook into transformer blocks
    def attention_hook(module, input, output):
        # Extract attention weights
        attn = output[1]  # (B, heads, N, N)
        attention_maps.append(attn)

    # Register hooks
    for block in encoder.blocks:
        block.attn.register_forward_hook(attention_hook)

    # Forward pass
    _ = encoder(image)

    # Aggregate attention maps across layers
    relevance_map = torch.stack(attention_maps).mean(dim=0)
    return relevance_map
```

**3. Feature-Level Relevance**
Different feature channels encode different aspects of relevance:
- Boundary channels: Edge and contour relevance
- Texture channels: Surface property relevance
- Semantic channels: Object identity relevance

### ARR-COC Training Considerations

When fine-tuning ViT-H for ARR-COC applications:

**Recommended Approach:**
1. Freeze early layers (generic features)
2. Fine-tune late layers (task-specific)
3. Train relevance attention modules
4. Use curriculum learning for 4P integration

**Loss Functions:**
- Segmentation loss: Dice + BCE
- Relevance alignment loss: Attention supervision
- 4P consistency loss: Multi-perspective agreement

---

## Sources

**Source Documents:**
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Lines 140-144, 566-598

**Web Research:**
- [Hugging Face vit-mae-huge](https://huggingface.co/facebook/vit-mae-huge) - Model card and usage (accessed 2025-11-20)
- [MAE Paper - CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) - Kaiming He et al. (accessed 2025-11-20)
- [Medical SAM Adapter Paper](https://www.sciencedirect.com/science/article/pii/S1361841525000945) - Wu et al. 2025 (accessed 2025-11-20)
- [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) - Masked Autoencoders Are Scalable Vision Learners

**Additional References:**
- [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything) - Official implementation
- [Original ViT Paper](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al. "An Image is Worth 16x16 Words"

---

**Last Updated:** 2025-11-20
**Part:** PART 7 of SAM General Expansion
**Lines:** ~700
