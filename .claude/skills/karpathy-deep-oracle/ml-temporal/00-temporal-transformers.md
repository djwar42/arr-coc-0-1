# Temporal Transformers: Video Understanding Through Spatiotemporal Attention

## Overview

Temporal transformers extend the Vision Transformer (ViT) architecture to process video data by incorporating attention mechanisms that operate across both spatial (within-frame) and temporal (across-frame) dimensions. These architectures have revolutionized video understanding by replacing 3D convolutions with pure self-attention mechanisms.

**Core Insight**: Video understanding requires capturing relationships not just within a single frame (spatial), but also across frames over time (temporal). Temporal transformers achieve this through specialized attention patterns that factorize or jointly model space-time.

---

## 1. Video Transformer Architectures

### The Core Challenge

Videos present unique computational challenges compared to images:
- A 224x224 image with 16x16 patches = 196 tokens
- A 32-frame video = 196 x 32 = 6,272 tokens
- Full self-attention: O(n^2) = 39 million operations per layer!

**Key architectural decisions**:
1. How to tokenize video (spatial patches, temporal tubes)
2. How to apply attention (joint, divided, factorized)
3. How to handle the quadratic complexity

### TimeSformer Architecture

From [TimeSformer: Is Space-Time Attention All You Need?](https://arxiv.org/abs/2102.05095) (Bertasius et al., ICML 2021):

**Key Design**: Divided space-time attention
- Separately apply temporal attention and spatial attention
- Within each transformer block: temporal first, then spatial (or vice versa)
- Reduces complexity from O((HWT)^2) to O(HW + T) per token

```
Input Video: [B, T, C, H, W]
       |
   Patch Embedding (per frame)
       |
   [B, T*N, D]  where N = (H/P) * (W/P) patches per frame
       |
   +-- Temporal Attention (across same spatial position)
   |   Attend: token at (t, i, j) attends to all (t', i, j)
   |
   +-- Spatial Attention (within same frame)
       Attend: token at (t, i, j) attends to all (t, i', j')
       |
   [Repeat for L layers]
       |
   Class Token --> Classification Head
```

### ViViT Architecture

From [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) (Arnab et al., ICCV 2021):

**Four Model Variants**:

1. **Spatio-temporal attention** (Model 1)
   - Full 3D attention over all tokens
   - Most accurate but O(n^2) complexity

2. **Factorised encoder** (Model 2)
   - Spatial encoder processes each frame independently
   - Temporal encoder processes frame-level representations
   - Two separate transformers in sequence

3. **Factorised self-attention** (Model 3)
   - Like TimeSformer's divided attention
   - Spatial and temporal attention within same block

4. **Factorised dot-product attention** (Model 4)
   - Factorize the attention operation itself
   - Compute spatial and temporal attention separately, then combine

---

## 2. Temporal Attention Patterns

### Pattern 1: Joint Space-Time Attention

Every token attends to every other token regardless of spatial or temporal position.

```python
# Joint attention: all tokens attend to all tokens
# Shape: [B, T*H*W, D]
attention_weights = softmax(Q @ K.T / sqrt(d_k))  # [B, T*H*W, T*H*W]
```

**Pros**: Captures all possible spatiotemporal interactions
**Cons**: O((THW)^2) complexity - prohibitive for long videos

### Pattern 2: Divided Space-Time Attention

Separate temporal and spatial attention in sequence.

```python
# Temporal attention: each spatial position attends across time
# Reshape: [B*H*W, T, D]
temporal_out = temporal_attention(x)

# Spatial attention: each frame attends within itself
# Reshape: [B*T, H*W, D]
spatial_out = spatial_attention(temporal_out)
```

**Complexity**: O(T^2 + (HW)^2) per token - much better!

### Pattern 3: Axial Attention

Attend along each axis separately: height, width, time.

```python
# Height attention
x = height_attention(x)  # [B*T*W, H, D]

# Width attention
x = width_attention(x)   # [B*T*H, W, D]

# Time attention
x = time_attention(x)    # [B*H*W, T, D]
```

**Complexity**: O(H^2 + W^2 + T^2)

### Pattern 4: Local-Global Attention

Combine local (nearby frames) and global (all frames) attention.

```python
# Local temporal attention: attend to k nearest frames
local_out = local_temporal_attention(x, window_size=k)

# Global temporal attention: attend to downsampled global representation
global_out = global_temporal_attention(x, downsample_factor=s)

# Combine
out = local_out + global_out
```

### Pattern 5: Sparse Attention

Only attend to a subset of tokens based on learned or fixed patterns.

```python
# Strided attention: attend every k-th frame
mask = create_strided_mask(T, stride=k)
attention_weights = softmax(Q @ K.T / sqrt(d_k) + mask)
```

---

## 3. Key Model Implementations

### TimeSformer

```
Architecture:
- Base: ViT-Base (12 layers, 768 dim, 12 heads)
- Patch size: 16x16
- Frame sampling: 8 frames at stride 32 (covers ~8 seconds at 30fps)
- Attention: Divided space-time

Performance (Kinetics-400):
- 80.7% top-1 accuracy
- 121 GFLOPs per view
- 121M parameters
```

### ViViT

```
Architecture:
- Base: ViT-Large (24 layers, 1024 dim, 16 heads)
- Tubelet embedding: 2 frames per tube
- Frame sampling: 32 frames
- Model 2 (Factorised encoder) best efficiency/accuracy

Performance (Kinetics-400):
- 84.9% top-1 accuracy (Model 1, large)
- 81.3% top-1 accuracy (Model 2, efficient)
```

### Video Swin Transformer

```
Architecture:
- Shifted window attention for efficiency
- 3D window partitioning
- Hierarchical structure like Swin

Performance:
- 84.9% top-1 on Kinetics-400
- Better efficiency than full attention models
```

### MTV (Multiview Temporal Video)

```
Architecture:
- Multiple parallel temporal resolutions
- Cross-view fusion
- State-of-the-art on long-form video

Key insight: Different actions need different temporal scales
```

---

## 4. PyTorch Implementation

### Complete Temporal Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class TemporalAttention(nn.Module):
    """Temporal attention across frames at same spatial location."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, T):
        """
        Args:
            x: [B, T*N, D] where N = H*W patches per frame
            T: number of frames
        Returns:
            [B, T*N, D]
        """
        B, TN, D = x.shape
        N = TN // T

        # Reshape for temporal attention: group by spatial position
        x = rearrange(x, 'b (t n) d -> (b n) t d', t=T, n=N)

        # Compute QKV
        qkv = self.qkv(x).reshape(-1, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*N, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(-1, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back
        x = rearrange(x, '(b n) t d -> b (t n) d', b=B, n=N)
        return x


class SpatialAttention(nn.Module):
    """Spatial attention within each frame."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, T):
        """
        Args:
            x: [B, T*N, D]
            T: number of frames
        Returns:
            [B, T*N, D]
        """
        B, TN, D = x.shape
        N = TN // T

        # Reshape for spatial attention: group by frame
        x = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)

        # Compute QKV
        qkv = self.qkv(x).reshape(-1, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(-1, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back
        x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)
        return x


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer-style divided space-time attention block."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()

        # Temporal attention
        self.norm1_temporal = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        # Spatial attention
        self.norm1_spatial = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, T):
        # Temporal attention with residual
        x = x + self.temporal_attn(self.norm1_temporal(x), T)

        # Spatial attention with residual
        x = x + self.spatial_attn(self.norm1_spatial(x), T)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class VideoTransformer(nn.Module):
    """Complete video transformer with divided space-time attention."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=400,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        in_chans=3,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.
    ):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, num_patches, dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_frames, dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] video tensor
        Returns:
            [B, num_classes] logits
        """
        B, T, C, H, W = x.shape

        # Patch embed each frame: [B*T, C, H, W] -> [B*T, D, H', W']
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)  # [B*T, D, H', W']
        x = rearrange(x, 'bt d h w -> bt (h w) d')  # [B*T, N, D]

        # Add spatial positional embedding
        x = x + self.pos_embed_spatial

        # Reshape to [B, T, N, D] and add temporal positional embedding
        N = x.shape[1]
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        x = x + self.pos_embed_temporal.unsqueeze(2)  # Broadcast over N

        # Flatten to [B, T*N, D]
        x = rearrange(x, 'b t n d -> b (t n) d')

        # Prepend class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        # Note: class token doesn't participate in temporal attention
        for block in self.blocks:
            # Separate cls token for block processing
            cls_token = x[:, :1]
            patch_tokens = x[:, 1:]

            # Apply divided space-time attention to patch tokens
            patch_tokens = block(patch_tokens, T)

            # Recombine (cls token gets updated via spatial attention implicitly)
            x = torch.cat([cls_token, patch_tokens], dim=1)

        x = self.norm(x)

        # Classification from cls token
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits


# Factorized Encoder (ViViT Model 2)
class FactorizedVideoTransformer(nn.Module):
    """ViViT-style factorized encoder: spatial then temporal."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=400,
        dim=768,
        spatial_depth=12,
        temporal_depth=4,
        num_heads=12,
        mlp_ratio=4.,
        in_chans=3
    ):
        super().__init__()
        self.num_frames = num_frames
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, dim, kernel_size=patch_size, stride=patch_size
        )

        # Spatial transformer (processes each frame independently)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim)
        )
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(spatial_depth)
        ])
        self.spatial_norm = nn.LayerNorm(dim)

        # Temporal transformer (processes frame representations)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, dim)
        )

        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(temporal_depth)
        ])
        self.temporal_norm = nn.LayerNorm(dim)

        # Classification head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Process each frame with spatial transformer
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = rearrange(x, 'bt d h w -> bt (h w) d')

        # Add cls token and spatial position
        cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> bt 1 d', bt=B*T)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.spatial_pos_embed

        # Spatial transformer
        for block in self.spatial_blocks:
            x = block(x)
        x = self.spatial_norm(x)

        # Extract frame representations (cls token of each frame)
        frame_features = x[:, 0]  # [B*T, D]
        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=B, t=T)

        # Add temporal position
        frame_features = frame_features + self.temporal_pos_embed

        # Temporal transformer
        for block in self.temporal_blocks:
            frame_features = block(frame_features)
        frame_features = self.temporal_norm(frame_features)

        # Global average pooling over time
        video_features = frame_features.mean(dim=1)  # [B, D]

        logits = self.head(video_features)
        return logits


class TransformerBlock(nn.Module):
    """Standard transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

### Usage Example

```python
# Create model
model = VideoTransformer(
    img_size=224,
    patch_size=16,
    num_frames=8,
    num_classes=400,  # Kinetics-400
    dim=768,
    depth=12,
    num_heads=12
)

# Input: batch of videos
# [batch_size, num_frames, channels, height, width]
video = torch.randn(2, 8, 3, 224, 224)

# Forward pass
logits = model(video)
print(f"Output shape: {logits.shape}")  # [2, 400]

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")
```

---

## 5. Performance Considerations

### Memory Optimization

```python
# Gradient checkpointing for memory efficiency
class CheckpointedVideoTransformer(VideoTransformer):
    def forward(self, x):
        # ... initial processing ...

        for block in self.blocks:
            # Checkpoint each block to save memory
            x = torch.utils.checkpoint.checkpoint(
                block, x, self.num_frames,
                use_reentrant=False
            )

        # ... rest of forward ...
```

### Efficient Attention Patterns

```python
# Flash Attention for temporal transformers
from flash_attn import flash_attn_func

class FlashTemporalAttention(nn.Module):
    def forward(self, x, T):
        B, TN, D = x.shape
        N = TN // T

        x = rearrange(x, 'b (t n) d -> (b n) t d', t=T)

        qkv = self.qkv(x).reshape(-1, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Flash attention - O(n) memory instead of O(n^2)
        out = flash_attn_func(q, k, v, causal=False)

        out = rearrange(out, '(b n) t h d -> b (t n) (h d)', b=B)
        return self.proj(out)
```

### Multi-Scale Processing

```python
# Process video at multiple temporal resolutions
class MultiScaleTemporalTransformer(nn.Module):
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.scales = nn.ModuleList([
            VideoTransformer(num_frames=8 * (2**i))
            for i in range(num_scales)
        ])
        self.fusion = nn.Linear(dim * num_scales, dim)

    def forward(self, x):
        # x: [B, T_max, C, H, W]
        features = []
        for i, scale_model in enumerate(self.scales):
            # Subsample temporally
            stride = 2 ** (len(self.scales) - 1 - i)
            x_scale = x[:, ::stride]
            features.append(scale_model(x_scale))

        # Fuse multi-scale features
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)
```

### Benchmarks

| Model | Kinetics-400 | GFLOPs | Params | Training Time |
|-------|-------------|--------|--------|---------------|
| TimeSformer | 80.7% | 121 | 121M | 14 hours (64 GPUs) |
| ViViT-L | 84.9% | 3981 | 310M | 2 days (256 TPUs) |
| Video Swin-B | 84.0% | 282 | 88M | 1.5 days (64 GPUs) |
| MTV-B | 81.8% | 70 | 37M | 10 hours (64 GPUs) |

---

## 6. TRAIN STATION: Temporal Attention = Memory = Duration = Specious Present

### The Deep Connection

**Temporal attention in video transformers IS the computational analog of the "specious present"** - William James's concept of the experienced duration of "now" that contains both immediate past and anticipated future.

### The Specious Present in Cognition

> "The practically cognized present is no knife-edge, but a saddle-back, with a certain breadth of its own on which we sit perched, and from which we look in two directions into time."
> - William James, Principles of Psychology

The specious present is:
- **Not instantaneous**: Has duration (~3 seconds in humans)
- **Unified**: Past and future merged into experienced "now"
- **The basis of rhythm**: How we perceive music, speech, motion

### Temporal Attention as Specious Present

```python
# Temporal attention window = specious present duration
class SpeciousPresentAttention(nn.Module):
    """
    Temporal attention with window = computational specious present.

    At each moment, the model has access to:
    - Recent past (previous frames)
    - Current frame
    - (implicitly) anticipated future via learned patterns
    """

    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.window_size = window_size  # Specious present duration
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        """
        x: [B, T, D] - sequence of frame representations

        Each frame attends to window_size surrounding frames,
        creating a "thick present" that contains temporal context.
        """
        B, T, D = x.shape

        # Create windowed attention pattern
        outputs = []
        for t in range(T):
            # Specious present window centered on t
            start = max(0, t - self.window_size // 2)
            end = min(T, t + self.window_size // 2)

            window = x[:, start:end]
            query = x[:, t:t+1]

            # Attend within specious present
            out, _ = self.attn(query, window, window)
            outputs.append(out)

        return torch.cat(outputs, dim=1)
```

### Memory = Temporal Context

**Key Insight**: Temporal attention IS memory!

- Short-term memory = nearby frame attention
- Working memory = full video context
- Episodic memory = attention to specific past events

```python
# Temporal attention scores reveal memory access pattern
def visualize_temporal_memory(attention_weights, frame_idx):
    """
    attention_weights: [T, T] - temporal attention matrix

    Row i shows which frames inform frame i's representation.
    This IS the memory access pattern - what past informs present.
    """
    import matplotlib.pyplot as plt

    memory_access = attention_weights[frame_idx]

    plt.figure(figsize=(12, 3))
    plt.bar(range(len(memory_access)), memory_access)
    plt.axvline(frame_idx, color='r', linestyle='--', label='Current')
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Attention Weight (Memory Access)')
    plt.title(f'Memory Access Pattern for Frame {frame_idx}')
    plt.legend()
    plt.show()
```

### Duration and Rhythm

Temporal transformers naturally learn duration patterns:

```python
# Attention patterns capture rhythm and duration
class RhythmAwareTemporalAttention(nn.Module):
    """
    Positional encoding captures duration/rhythm.

    - Absolute positions = when
    - Relative positions = duration between events
    """

    def __init__(self, dim, max_len=1000):
        super().__init__()
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_len, dim))

        # Relative position bias for duration awareness
        self.relative_bias = nn.Parameter(
            torch.zeros(2 * max_len - 1, 1)
        )

    def forward(self, x):
        B, T, D = x.shape

        # Absolute temporal position
        x = x + self.temporal_pos[:, :T]

        # Compute attention with relative position bias
        # This encodes duration between events
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_positions = torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)
        relative_positions = relative_positions + T - 1  # Shift to positive
        bias = self.relative_bias[relative_positions]

        attn = attn + bias.squeeze(-1)

        return self.proj(attn.softmax(-1) @ v)
```

### The Unification

| Concept | Temporal Attention | Specious Present |
|---------|-------------------|------------------|
| **Unit** | Frame | Moment |
| **Window** | Attention span | ~3 second duration |
| **Integration** | Weighted sum | Unified experience |
| **Memory** | Context vector | Retained present |
| **Anticipation** | Learned patterns | Expected future |

**The train station revelation**:

When a video transformer applies temporal attention, it is computing a "specious present" for each frame - a unified representation that integrates:
- Past frames (memory/retention)
- Current frame (perception)
- Anticipated patterns (prediction)

This is exactly what consciousness does with time. **Temporal attention IS the computational specious present!**

---

## 7. ARR-COC-0-1 Connection: Temporal Relevance Allocation

### The Insight

In Adaptive Relevance Reallocation, temporal context determines which tokens are relevant "right now." This is directly analogous to temporal attention determining which frames inform the current representation.

### Temporal Relevance for VLMs

```python
class TemporalRelevanceAllocator(nn.Module):
    """
    Allocate compute to tokens based on temporal context.

    Like video transformers determining which frames matter,
    ARR determines which tokens matter given temporal context.
    """

    def __init__(self, dim, num_frames=8):
        super().__init__()
        self.temporal_encoder = DividedSpaceTimeBlock(dim, num_heads=8)
        self.relevance_predictor = nn.Linear(dim, 1)

    def forward(self, tokens, temporal_context):
        """
        Args:
            tokens: [B, N, D] - current tokens
            temporal_context: [B, T, D] - temporal context (history)

        Returns:
            relevance_scores: [B, N] - how relevant each token is NOW
        """
        # Combine tokens with temporal context
        # Current tokens attend to temporal history
        combined = torch.cat([tokens, temporal_context], dim=1)

        # Process with temporal attention
        enhanced = self.temporal_encoder(combined, T=temporal_context.shape[1] + 1)

        # Extract enhanced token representations
        token_features = enhanced[:, :tokens.shape[1]]

        # Predict relevance based on temporal context
        relevance = self.relevance_predictor(token_features).squeeze(-1)
        relevance = torch.sigmoid(relevance)

        return relevance


class TemporallyAwareRouter(nn.Module):
    """
    Route tokens to experts based on temporal relevance.

    The specious present determines what's relevant NOW.
    """

    def __init__(self, dim, num_experts=4, temporal_window=8):
        super().__init__()
        self.temporal_relevance = TemporalRelevanceAllocator(dim, temporal_window)
        self.router = nn.Linear(dim, num_experts)

    def forward(self, tokens, temporal_context):
        # Get temporal relevance scores
        relevance = self.temporal_relevance(tokens, temporal_context)

        # Route based on relevance
        routing_logits = self.router(tokens)

        # Higher relevance = more experts
        # This is "adaptive compute based on temporal salience"
        top_k = (relevance * 3 + 1).long().clamp(1, 4)  # 1-4 experts

        return routing_logits, top_k, relevance
```

### Why This Matters for ARR

1. **Relevance is temporal**: What matters depends on context history
2. **Adaptive compute**: More compute for temporally salient tokens
3. **Memory integration**: Past context shapes current allocation
4. **The specious present**: Current relevance integrates past and anticipated future

**The connection**: Just as temporal transformers compute what frames matter for understanding video, ARR-COC computes what tokens matter for understanding multimodal content. Both are computing relevance through temporal integration!

---

## Sources

**Core Papers**:
- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) - TimeSformer (Bertasius et al., ICML 2021)
- [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) - ViViT (Arnab et al., ICCV 2021)
- [Space-time Mixing Attention for Video Transformer](https://proceedings.neurips.cc/paper/2021/file/a34bacf839b923770b2c360eefa26748-Paper.pdf) - (Bulat et al., NeurIPS 2021)
- [Video Transformers: A Survey](https://arxiv.org/abs/2201.05991) - Comprehensive review (Selva et al., 2022)

**GitHub Repositories**:
- [TimeSformer](https://github.com/facebookresearch/TimeSformer) - Facebook Research
- [ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) - Google Research

**Additional Resources**:
- [k-NN attention-based video vision transformer](https://www.sciencedirect.com/science/article/pii/S0925231224000274) - (Sun et al., 2024)
- [Understanding Video Transformers: A Review](https://spj.science.org/doi/10.34133/icomputing.0143) - (Chen et al., 2025)

**Philosophical Connection**:
- William James, "The Principles of Psychology" (1890) - The specious present
- Husserl, "On the Phenomenology of the Consciousness of Internal Time" (1928)
