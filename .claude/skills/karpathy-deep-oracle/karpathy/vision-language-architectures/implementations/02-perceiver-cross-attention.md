# Perceiver Cross-Attention Implementation

## Overview

The Perceiver architecture, introduced by DeepMind in 2021, uses **latent queries with cross-attention** to handle high-dimensional inputs efficiently. Unlike standard transformers that apply self-attention directly to inputs, Perceiver uses a small set of learned latent variables that cross-attend to the inputs, dramatically reducing computational complexity.

**Key Innovation**: Cross-attention from latents (queries) to inputs (keys/values) creates a bottleneck that scales linearly with input size rather than quadratically.

From [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) (DeepMind, 2021):
- Latent variables produce queries (Q)
- Input data produces keys (K) and values (V)
- Complexity: O(M × N) where M = num_latents, N = input_length
- Standard attention: O(N²)

## Architecture Components

### 1. Latent Array Initialization

The core of Perceiver is a learnable set of latent variables:

```python
import torch
import torch.nn as nn

class PerceiverLatents(nn.Module):
    """Learnable latent array that compresses input information"""

    def __init__(
        self,
        num_latents: int = 256,      # Number of latent variables
        latent_dim: int = 1024,       # Dimension of each latent
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Randomly initialized, learned during training
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Expand latents for batch processing

        Args:
            batch_size: Number of samples in batch

        Returns:
            Latent array of shape (batch_size, num_latents, latent_dim)
        """
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)
```

**Design Rationale** (from Hugging Face blog):
- Latents act as a bottleneck, forcing efficient information compression
- Number of latents determines how much information can flow through
- Typical values: 64-256 for images, 256-512 for complex modalities
- Higher latent_dim allows richer representations

### 2. Cross-Attention Module

The critical component: latents query the input data.

```python
class PerceiverCrossAttention(nn.Module):
    """
    Cross-attention where latents attend to inputs

    Latents → Queries
    Inputs → Keys, Values
    """

    def __init__(
        self,
        latent_dim: int = 1024,       # Dimension of latents
        input_dim: int = 768,          # Dimension of input features
        num_heads: int = 8,            # Number of attention heads
        head_dim: int = 64,            # Dimension per head
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        # Latents produce queries
        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)

        # Inputs produce keys and values
        self.to_k = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(input_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        latents: torch.Tensor,  # (batch, num_latents, latent_dim)
        inputs: torch.Tensor,   # (batch, input_len, input_dim)
    ) -> torch.Tensor:
        """
        Cross-attention from latents to inputs

        Args:
            latents: Latent queries (B, M, D_latent)
            inputs: Input features (B, N, D_input)

        Returns:
            Updated latents (B, M, D_latent)
        """
        batch_size, num_latents, _ = latents.shape
        _, input_len, _ = inputs.shape

        # Generate Q, K, V
        q = self.to_q(latents)  # (B, M, inner_dim)
        k = self.to_k(inputs)    # (B, N, inner_dim)
        v = self.to_v(inputs)    # (B, N, inner_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_latents, self.num_heads, self.head_dim)
        k = k.view(batch_size, input_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, input_len, self.num_heads, self.head_dim)

        # Transpose to (B, heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (B, heads, M, head_dim)
        k = k.transpose(1, 2)  # (B, heads, N, head_dim)
        v = v.transpose(1, 2)  # (B, heads, N, head_dim)

        # Attention scores: Q @ K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (B, heads, M, N) - each latent attends to all inputs

        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, heads, M, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, num_latents, self.inner_dim)

        return self.to_out(out)  # (B, M, latent_dim)
```

**Key Properties**:
- Output shape matches input latents shape (B, M, latent_dim)
- Attention matrix is (M × N), not (N × N)
- Each latent can attend to entire input sequence
- Complexity: O(M × N × D) vs O(N² × D) for self-attention

### 3. Latent Transformer (Self-Attention)

After cross-attention, latents undergo self-attention to process information:

```python
class PerceiverSelfAttention(nn.Module):
    """Self-attention among latents after cross-attention"""

    def __init__(
        self,
        latent_dim: int = 1024,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        # Standard QKV projections
        self.to_qkv = nn.Linear(latent_dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Self-attention among latents

        Args:
            latents: (B, M, latent_dim)

        Returns:
            Updated latents (B, M, latent_dim)
        """
        batch_size, num_latents, _ = latents.shape

        # Generate Q, K, V from latents
        qkv = self.to_qkv(latents).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(
                batch_size, num_latents, self.num_heads, self.head_dim
            ).transpose(1, 2),
            qkv
        )

        # Standard self-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, num_latents, self.inner_dim)
        return self.to_out(out)


class PerceiverTransformerBlock(nn.Module):
    """Transformer block for latent self-attention"""

    def __init__(
        self,
        latent_dim: int = 1024,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_dim)
        self.attn = PerceiverSelfAttention(
            latent_dim, num_heads, head_dim, dropout
        )
        self.ln2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * mlp_ratio, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        latents = latents + self.attn(self.ln1(latents))
        # MLP with residual
        latents = latents + self.mlp(self.ln2(latents))
        return latents
```

### 4. Complete Perceiver Encoder

Combining cross-attention and self-attention:

```python
class PerceiverEncoder(nn.Module):
    """
    Complete Perceiver encoder with iterative attention

    Architecture:
    1. Cross-attention: latents ← inputs
    2. Self-attention: latents ← latents (repeated)
    """

    def __init__(
        self,
        num_latents: int = 256,
        latent_dim: int = 1024,
        input_dim: int = 768,
        num_self_attn_layers: int = 6,
        num_cross_attn_heads: int = 1,    # Paper uses 1
        num_self_attn_heads: int = 8,
        cross_head_dim: int = 64,
        self_head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Learnable latent array
        self.latents = PerceiverLatents(num_latents, latent_dim)

        # Cross-attention: latents attend to inputs
        self.cross_attn = PerceiverCrossAttention(
            latent_dim=latent_dim,
            input_dim=input_dim,
            num_heads=num_cross_attn_heads,
            head_dim=cross_head_dim,
            dropout=dropout,
        )
        self.cross_attn_ln = nn.LayerNorm(latent_dim)

        # Self-attention layers for latents
        self.self_attn_blocks = nn.ModuleList([
            PerceiverTransformerBlock(
                latent_dim=latent_dim,
                num_heads=num_self_attn_heads,
                head_dim=self_head_dim,
                dropout=dropout,
            )
            for _ in range(num_self_attn_layers)
        ])

    def forward(
        self,
        inputs: torch.Tensor,  # (B, N, input_dim)
    ) -> torch.Tensor:
        """
        Encode inputs into latent space

        Args:
            inputs: Input features (B, N, input_dim)

        Returns:
            Latent representations (B, M, latent_dim)
        """
        batch_size = inputs.shape[0]

        # Initialize latents for batch
        latents = self.latents(batch_size)  # (B, M, latent_dim)

        # Cross-attention: compress inputs into latents
        latents = latents + self.cross_attn(latents, inputs)
        latents = self.cross_attn_ln(latents)

        # Self-attention: process latent representations
        for block in self.self_attn_blocks:
            latents = block(latents)

        return latents  # (B, M, latent_dim)
```

**Complexity Analysis**:
- Cross-attention: O(M × N × D) where M << N
- Self-attention: O(M² × D) - cheap since M is small
- Total: O(M × N × D + L × M² × D) where L = num layers
- For M=256, N=50176 (224×224 image): ~200× reduction in cross-attn cost

## Perceiver Resampler (Flamingo Variant)

From [Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (DeepMind, 2022):

The Perceiver Resampler is a variant used in Flamingo to compress variable-length visual features into a fixed number of tokens.

```python
class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler from Flamingo

    Key differences from base Perceiver:
    - Outputs fixed number of visual tokens
    - Used to compress variable-sized vision encoder outputs
    - Enables integration with frozen language models
    """

    def __init__(
        self,
        dim: int = 1024,                  # Feature dimension
        depth: int = 6,                   # Number of self-attention layers
        num_latents: int = 64,            # Fixed output size
        dim_head: int = 64,
        heads: int = 8,
        num_time_embeds: int = 4,         # For temporal sequences
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.dim = dim

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Temporal position embeddings (for video frames)
        self.time_embeds = nn.Parameter(
            torch.randn(num_time_embeds, 1, dim)
        ) if num_time_embeds > 0 else None

        # Cross-attention from latents to media
        self.cross_attn = PerceiverCrossAttention(
            latent_dim=dim,
            input_dim=dim,
            num_heads=heads,
            head_dim=dim_head,
            dropout=dropout,
        )

        # Self-attention layers
        self.layers = nn.ModuleList([
            PerceiverTransformerBlock(
                latent_dim=dim,
                num_heads=heads,
                head_dim=dim_head,
                mlp_ratio=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        media: torch.Tensor,  # (B, T, N, D) - batch, time, seq_len, dim
    ) -> torch.Tensor:
        """
        Resample media features to fixed number of tokens

        Args:
            media: Input features (B, T, N, D)
                   B = batch size
                   T = number of frames/time steps
                   N = sequence length per frame
                   D = feature dimension

        Returns:
            Resampled features (B, T, num_latents, D)
        """
        batch, time, seq_len, dim = media.shape

        # Expand latents for batch and time
        latents = self.latents.unsqueeze(0).unsqueeze(0)
        latents = latents.expand(batch, time, -1, -1)  # (B, T, M, D)

        # Add temporal position embeddings
        if self.time_embeds is not None:
            time_emb = self.time_embeds[:time]  # (T, 1, D)
            media = media + time_emb.unsqueeze(0)

        # Flatten time and batch for processing
        latents = latents.reshape(batch * time, self.num_latents, dim)
        media = media.reshape(batch * time, seq_len, dim)

        # Cross-attention: latents attend to media
        latents = self.cross_attn(latents, media)

        # Self-attention layers
        for layer in self.layers:
            latents = layer(latents)

        latents = self.norm(latents)

        # Reshape back to (B, T, M, D)
        latents = latents.reshape(batch, time, self.num_latents, dim)

        return latents
```

**Use Case in Flamingo**:
- Vision encoder outputs variable shapes: (B, T, N_i, D) where N_i varies
- Perceiver Resampler → fixed shape: (B, T, 64, D)
- Fixed-size output can be fed into frozen language model
- Acts as learned adaptive pooling

From [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) implementation notes:
- Typical config: 64 latents, 6 layers, 8 heads
- Applied per-frame then concatenated temporally
- Enables few-shot visual question answering

## Complete Vision-Language Example

Putting it all together for image classification:

```python
class PerceiverImageClassifier(nn.Module):
    """
    Complete example: Image classification with Perceiver

    Pipeline:
    1. Image patches → embeddings
    2. Cross-attention: latents ← patch embeddings
    3. Self-attention: process latents
    4. Classification head
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        channels: int = 3,
        num_latents: int = 512,
        latent_dim: int = 1024,
        num_self_attn_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Patchify image
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, latent_dim)

        # Position embeddings for patches
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, latent_dim)
        )

        # Perceiver encoder
        self.perceiver = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=latent_dim,
            num_self_attn_layers=num_self_attn_layers,
            num_cross_attn_heads=1,
            num_self_attn_heads=num_heads,
            dropout=dropout,
        )

        # Classification head
        self.norm = nn.LayerNorm(latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches

        Args:
            images: (B, C, H, W)

        Returns:
            Patches: (B, num_patches, patch_dim)
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # Reshape to patches
        patches = images.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, -1, C * p * p)

        return patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images

        Args:
            images: (B, C, H, W)

        Returns:
            Logits: (B, num_classes)
        """
        # Extract patches
        patches = self.patchify(images)  # (B, N, patch_dim)

        # Embed patches
        x = self.patch_embedding(patches)  # (B, N, latent_dim)
        x = x + self.pos_embedding

        # Encode with Perceiver
        latents = self.perceiver(x)  # (B, M, latent_dim)

        # Global average pooling over latents
        x = latents.mean(dim=1)  # (B, latent_dim)

        # Classification
        x = self.norm(x)
        logits = self.head(x)  # (B, num_classes)

        return logits
```

## Implementation Comparison

### Perceiver vs Standard Transformer

```python
# Standard Transformer: O(N²) complexity
class StandardTransformer:
    def forward(self, x):  # x: (B, N, D)
        # Self-attention: Q, K, V all from x
        q = k = v = x  # (B, N, D)
        attn = softmax(q @ k.T / sqrt(d))  # (B, N, N) - EXPENSIVE
        out = attn @ v  # (B, N, D)
        return out

# Perceiver: O(M × N) complexity
class PerceiverTransformer:
    def forward(self, x):  # x: (B, N, D)
        latents = self.latents  # (B, M, D) where M << N

        # Cross-attention: Q from latents, K,V from x
        q = latents  # (B, M, D)
        k = v = x    # (B, N, D)
        attn = softmax(q @ k.T / sqrt(d))  # (B, M, N) - CHEAP
        out = attn @ v  # (B, M, D)

        # Self-attention only on latents
        self_attn(out)  # (B, M, M) - VERY CHEAP
        return out
```

### Memory Footprint

For image classification (224×224, patch_size=16):
- Input sequence length N = 196
- Latents M = 64

```
Standard Transformer:
- Attention matrix: 196 × 196 = 38,416 elements
- Memory: O(N²) per head

Perceiver:
- Cross-attention: 64 × 196 = 12,544 elements
- Self-attention: 64 × 64 = 4,096 elements
- Total: 16,640 elements (2.3× reduction)
- Memory: O(M × N + M²) ≈ O(M × N)
```

## Training Considerations

### Initialization

From the DeepMind paper:
- Latents: Xavier/Glorot initialization (`torch.randn`)
- Cross-attention weights: Smaller initialization to prevent gradient explosion
- Position embeddings: Standard normal or learned from scratch

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Parameter):
        torch.nn.init.normal_(module, std=0.02)
```

### Learning Rate Scheduling

Recommended schedule (from Perceiver IO paper):
- Warmup: 10,000 steps
- Peak LR: 1e-4 for latents, 5e-5 for rest
- Cosine decay to 1e-6
- Gradient clipping: 1.0

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW([
    {'params': model.perceiver.latents.parameters(), 'lr': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if 'latents' not in n], 'lr': 5e-5},
], weight_decay=0.01)

scheduler = CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
```

### Gradient Checkpointing

For large models, use gradient checkpointing on self-attention blocks:

```python
from torch.utils.checkpoint import checkpoint

class PerceiverEncoder(nn.Module):
    def forward(self, inputs):
        latents = self.latents(inputs.shape[0])
        latents = self.cross_attn(latents, inputs)

        # Checkpoint self-attention blocks
        for block in self.self_attn_blocks:
            latents = checkpoint(block, latents)

        return latents
```

## Key Insights

**From Perceiver paper**:
1. Cross-attention bottleneck forces efficient compression
2. Number of latents trades off between capacity and speed
3. Works on any modality without architectural changes
4. Iterative self-attention refines latent representations

**From Flamingo paper**:
1. Resampler provides fixed-size output for integration with LLMs
2. Temporal embeddings handle video/sequential inputs
3. Learned queries more flexible than fixed pooling
4. Enables few-shot learning through efficient conditioning

**Implementation Tips**:
- Start with fewer latents (64-128) for debugging
- Single cross-attention head often sufficient (paper uses 1)
- More self-attention layers (6-8) improve quality
- Position embeddings critical for spatial understanding

## Comparison with Other Approaches

| Method | Complexity | Latents | Use Case |
|--------|-----------|---------|----------|
| Standard Transformer | O(N²) | No | General sequence modeling |
| Perceiver | O(M × N) | Fixed | High-dimensional inputs |
| Perceiver IO | O(M × N + M × K) | Fixed | Flexible outputs (K) |
| Perceiver Resampler | O(M × N) | Fixed | Vision-language interface |

## Sources

**Research Papers**:
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) - DeepMind (2021)
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) - DeepMind (2021)
- [Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) - DeepMind (2022)

**Implementation References**:
- [lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch) - Clean PyTorch implementation
- [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) - Perceiver Resampler variant
- [HuggingFace Perceiver Guide](https://huggingface.co/blog/perceiver) - Detailed architecture explanation
- [DeepMind Research Repository](https://github.com/deepmind/deepmind-research/tree/master/perceiver) - Official JAX implementation

**Additional Resources**:
- [Yannic Kilcher - Perceiver Explained](https://www.youtube.com/watch?v=P_xeshTnPZg) - Video walkthrough
- [The Annotated Perceiver](https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb) - Line-by-line code explanation
