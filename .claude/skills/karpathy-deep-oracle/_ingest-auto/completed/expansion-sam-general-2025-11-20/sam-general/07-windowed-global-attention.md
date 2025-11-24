# SAM Windowed and Global Attention Mechanisms

## Section 1: Attention Mechanism Overview

### The Computational Challenge of Vision Transformers

Vision Transformers (ViTs) revolutionized computer vision by applying the transformer architecture to images. However, standard self-attention presents a fundamental challenge: **quadratic computational complexity**.

**The Math Problem:**
For an image with N tokens (patches):
- Standard self-attention: O(N^2) complexity
- For a 1024x1024 image with 16x16 patches: N = 64 x 64 = 4,096 tokens
- Attention matrix size: 4,096 x 4,096 = ~16.7 million elements

This quadratic scaling makes full global attention extremely expensive for high-resolution images commonly used in segmentation tasks.

### SAM's Solution: Hybrid Attention Strategy

SAM's image encoder employs a sophisticated **hybrid attention strategy** that combines:

1. **Windowed Attention** (local context) - For most transformer blocks
2. **Global Attention** (full image context) - For specific strategic blocks

From the SAM paper (arXiv:2304.02643):
> "We use the ViT-H/16 variant, which employs 14x14 windowed attention and four equally-spaced global attention blocks"

**Architecture Configuration:**
- Total transformer blocks: 32 (ViT-H)
- Windowed attention blocks: 28 (87.5%)
- Global attention blocks: 4 (12.5%)
- Global attention at layers: 7, 15, 23, 31 (equally spaced)

### Why This Hybrid Approach?

**Computational Efficiency:**
- Windowed attention: O(W^2 x N/W^2) where W is window size
- With 14x14 windows on 64x64 feature map: ~21x more efficient than global

**Feature Hierarchy:**
- Local patterns (edges, textures) captured by windowed attention
- Global relationships (object context, scene understanding) by global attention
- Strategic placement ensures information flows across the entire image

**Sources:**
- SAM Paper: https://arxiv.org/abs/2304.02643
- Medical SAM Adapter: https://arxiv.org/html/2304.12620v7

---

## Section 2: Windowed Attention

### Concept and Motivation

Windowed attention restricts self-attention computation to local, non-overlapping windows within the feature map. Instead of each token attending to all other tokens, it only attends to tokens within its local window.

**Key Insight from Swin Transformer (arXiv:2103.14030):**
> "The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connections."

### Implementation in SAM

SAM implements windowed attention by partitioning the 64x64 feature map into smaller windows:

**Window Partitioning Process:**
```python
def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition into non-overlapping windows with padding if needed.

    Input: (B, H, W, C) - e.g., (B, 64, 64, 1280)
    Output: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # Calculate padding for divisibility
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # Reshape to windows: (B, H//ws, ws, W//ws, ws, C)
    x = x.view(B, Hp // window_size, window_size,
               Wp // window_size, window_size, C)

    # Rearrange: (B * num_windows, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)
```

**Example with SAM's Configuration:**
- Feature map: 64x64 tokens
- Window size: 14x14
- Padding: 64 -> 70 (5 windows per dimension)
- Total windows: 5 x 5 = 25 windows
- Each window: 14 x 14 = 196 tokens

### Attention Within Windows

Within each window, standard multi-head self-attention is applied:

```python
class Attention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # Attention computation (within window only)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative positional encoding
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q,
                                          self.rel_pos_h,
                                          self.rel_pos_w,
                                          (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        return self.proj(x)
```

### Window Unpartitioning

After attention, windows are merged back to the original spatial layout:

```python
def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Reverse window partition and remove padding.

    Input: (B * num_windows, ws, ws, C)
    Output: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    # Reshape to spatial grid
    x = windows.view(B, Hp // window_size, Wp // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, -1)

    # Remove padding
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x
```

### Advantages of Windowed Attention

1. **Linear Complexity**: O(W^2 x N/W^2) instead of O(N^2)
2. **Memory Efficiency**: Smaller attention matrices fit in GPU memory
3. **Local Feature Learning**: Ideal for textures, edges, local patterns
4. **Batch Processing**: Windows can be processed in parallel

### Limitations Addressed by Global Attention

- **No Cross-Window Communication**: Information isolated within windows
- **Limited Receptive Field**: Cannot capture long-range dependencies
- **Object Fragmentation**: Large objects spanning multiple windows lack coherence

---

## Section 3: Global Attention

### Full Image Context

Global attention allows every token to attend to every other token across the entire feature map. This provides the model with full image context, essential for understanding:

- Object relationships and scene composition
- Long-range spatial dependencies
- Coherent segmentation of large objects
- Contextual disambiguation

### SAM's Strategic Placement

SAM places global attention blocks at strategic intervals:

**Layer Distribution (ViT-H with 32 blocks):**
- Layer 7: First global attention (after 7 windowed blocks)
- Layer 15: Second global attention (quarter point)
- Layer 23: Third global attention (three-quarter point)
- Layer 31: Final global attention (before output)

```python
# From SAM's ImageEncoderViT
self.blocks = nn.ModuleList()
for i in range(depth):  # depth = 32 for ViT-H
    block = Block(
        dim=embed_dim,
        num_heads=num_heads,
        # Global attention when i in [7, 15, 23, 31]
        window_size=window_size if i not in global_attn_indexes else 0,
        input_size=(img_size // patch_size, img_size // patch_size),
    )
    self.blocks.append(block)
```

### Why Equal Spacing?

The equally-spaced distribution serves critical purposes:

1. **Progressive Feature Integration**
   - Early global: Establish initial global context
   - Middle global: Refine cross-region relationships
   - Late global: Final integration before output

2. **Information Propagation**
   - Windowed blocks refine local features
   - Global blocks propagate information across windows
   - Alternating pattern ensures balanced learning

3. **Computational Balance**
   - Only 4/32 = 12.5% are global (expensive)
   - 87.5% windowed (efficient)
   - Optimal trade-off between quality and speed

### Global Attention Implementation

When `window_size = 0`, the block applies attention to the full feature map:

```python
class Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Conditional windowing
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # Apply attention (windowed or global)
        x = self.attn(x)

        # Unpartition if windowed
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

### Impact on Segmentation Quality

Global attention is crucial for SAM's segmentation performance:

**Object Coherence:**
- Connects distant parts of the same object
- Ensures consistent segmentation across regions
- Handles occluded or scattered object parts

**Contextual Understanding:**
- Scene-level semantic understanding
- Relationship between foreground and background
- Disambiguation of similar-looking regions

**Prompt Integration:**
- Global context helps interpret sparse prompts
- Single click can influence entire image understanding
- Box prompts gain meaning from global relationships

---

## Section 4: Interleaved Pattern

### The Hybrid Architecture Design

SAM's interleaved pattern creates a sophisticated information flow:

```
Layer 1-7:   [W] [W] [W] [W] [W] [W] [G]
Layer 8-15:  [W] [W] [W] [W] [W] [W] [W] [G]
Layer 16-23: [W] [W] [W] [W] [W] [W] [W] [G]
Layer 24-31: [W] [W] [W] [W] [W] [W] [W] [G]

W = Windowed Attention (14x14)
G = Global Attention (full 64x64)
```

### Information Flow Dynamics

**Phase 1 (Layers 1-7): Local Feature Extraction**
- Build rich local representations
- Capture textures, edges, local patterns
- First global block integrates initial context

**Phase 2 (Layers 8-15): Mid-Level Integration**
- Refine local features with global context
- Begin forming object-level representations
- Second global ensures cross-region coherence

**Phase 3 (Layers 16-23): High-Level Semantics**
- Abstract semantic features emerge
- Object relationships solidify
- Third global for semantic consistency

**Phase 4 (Layers 24-31): Final Integration**
- Prepare features for mask decoder
- Final refinements and integration
- Last global ensures complete image understanding

### Comparison with Alternative Designs

**Pure Global Attention (Standard ViT):**
- Computational cost: Prohibitive for high-resolution
- Memory usage: Cannot fit large images
- Advantage: Maximum context at every layer

**Pure Windowed (No Global):**
- Problem: No cross-window communication
- Result: Fragmented object representations
- Missing: Scene-level understanding

**Shifted Windows (Swin Transformer):**
- Alternates window positions between layers
- Creates cross-window connections via overlap
- Different approach to global context

**SAM's Hybrid (Windowed + Strategic Global):**
- Best of both worlds
- Efficient local processing
- Strategic global integration
- Optimal for segmentation tasks

### Benefits for Segmentation

1. **Multi-Scale Understanding**
   - Local: Fine-grained boundaries
   - Global: Object-level semantics

2. **Efficient Processing**
   - 87.5% efficient windowed computation
   - 12.5% strategic global integration

3. **Prompt Sensitivity**
   - Local attention near prompt location
   - Global context for interpretation

4. **Robust Generalization**
   - Local patterns transfer across domains
   - Global context adapts to new scenes

---

## Section 5: Computational Efficiency

### Complexity Analysis

**Standard Global Self-Attention:**
```
Complexity: O(N^2)
For 64x64 feature map: O(4096^2) = O(16.7M)
```

**Windowed Self-Attention:**
```
Complexity: O(W^2 x num_windows)
For 14x14 windows on 64x64: O(196 x 25) = O(4,900)
Speedup: 16.7M / 4,900 = ~3,400x faster per layer
```

### Memory Savings

**Attention Matrix Memory:**
- Global: 4096 x 4096 x 4 bytes = 67 MB per head
- Windowed: 196 x 196 x 25 windows x 4 bytes = 3.8 MB per head

**With 16 heads:**
- Global: 1.07 GB per layer
- Windowed: 61 MB per layer
- Savings: ~94% memory reduction

### SAM's Overall Efficiency

**Per-Layer Computation (ViT-H):**
- 28 windowed layers: 28 x O(4,900) = O(137K)
- 4 global layers: 4 x O(16.7M) = O(67M)
- Total dominated by global layers

**Effective Speedup:**
Without windowed attention (all global):
- 32 x O(16.7M) = O(534M)

With hybrid approach:
- O(137K) + O(67M) = O(67M)
- Speedup: ~8x overall

### Real-World Performance

**Inference Speed (A100 GPU):**
- Image encoding: ~100ms for ViT-H
- Prompt encoding + mask decode: ~10ms
- Total: ~110ms per mask

**Throughput:**
- Single image: ~9 images/second
- Batch processing: Higher with efficient batching

### Scaling Properties

**Resolution Scaling:**
The hybrid approach scales better with resolution:

| Resolution | Global Only | SAM Hybrid |
|------------|-------------|------------|
| 512x512    | 1x          | 1x         |
| 1024x1024  | 4x slower   | 1.5x slower|
| 2048x2048  | 16x slower  | 2.5x slower|

The windowed attention component scales linearly with resolution while global attention scales quadratically, making SAM's approach increasingly beneficial at higher resolutions.

---

## Section 6: Implementation Details

### Relative Positional Encoding

SAM enhances both windowed and global attention with **decomposed relative positional embeddings**:

```python
def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Add decomposed relative positional embeddings to attention scores.

    Height and width are encoded separately for efficiency.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    # Get relative position matrices
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # (q_h, k_h, head_dim)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # (q_w, k_w, head_dim)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    # Compute position biases via Einstein summation
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # Add to attention scores
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn
```

**Why Decomposed?**
- Full 2D: (2H-1) x (2W-1) = 127 x 127 = 16,129 parameters
- Decomposed: (2H-1) + (2W-1) = 254 parameters
- **98.4% parameter reduction** with minimal quality loss

### Attention with Position Encoding

The complete attention computation:

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True,
                 use_rel_pos=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if use_rel_pos:
            # Learnable relative position embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim)
            )
```

### MAE Pre-training Foundation

SAM's image encoder uses **Masked Autoencoder (MAE) pre-training**:

1. **Pre-training Task**: Reconstruct masked image patches (75% masked)
2. **Learned Features**: Rich visual representations without task bias
3. **Transfer Benefit**: Strong initialization for segmentation fine-tuning

This pre-training is crucial for the attention mechanisms to work effectively:
- Windowed attention learns meaningful local patterns
- Global attention learns semantic relationships
- Both benefit from MAE's self-supervised visual understanding

### Block Structure

Complete transformer block with attention selection:

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 window_size=0, input_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            input_size=input_size if window_size == 0
                       else (window_size, window_size)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition if windowed attention
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Window unpartition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## Section 7: ARR-COC Integration

### Relevance to Custom Model Training

Understanding SAM's attention mechanisms provides valuable insights for ARR-COC training:

**Architecture Decisions:**
- Hybrid attention strategies applicable to custom vision models
- Trade-off between efficiency and global context
- Strategic placement of global attention blocks

### Potential Applications

**Custom Segmentation Models:**
```python
# Example: Adapting windowed attention for ARR-COC
class CustomViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=14,
                 use_global=False):
        # Similar structure to SAM blocks
        # Configurable window size and global attention
        pass
```

**Efficiency Considerations:**
- High-resolution images benefit from windowed attention
- Global attention for semantic integration points
- Balance based on GPU memory and latency requirements

### Training Insights

**From SAM's Success:**
1. Pre-training matters: MAE provides strong foundations
2. Hybrid is optimal: Pure windowed or global suboptimal
3. Strategic globals: 4 equally-spaced beats random placement

**For ARR-COC Models:**
- Consider attention mechanism when designing custom encoders
- Windowed attention enables higher resolution training
- Global attention critical for semantic tasks

### Implementation Patterns

**Configurable Attention:**
```python
# Pattern for flexible attention configuration
def create_encoder(
    depth=24,
    global_attn_indexes=[5, 11, 17, 23],  # Every 6th layer
    window_size=14
):
    blocks = []
    for i in range(depth):
        ws = 0 if i in global_attn_indexes else window_size
        blocks.append(Block(window_size=ws))
    return nn.ModuleList(blocks)
```

**Memory-Efficient Training:**
- Gradient checkpointing with windowed attention
- Mixed precision for attention computation
- Efficient attention implementations (Flash Attention compatible)

### Future Directions

**Potential Improvements:**
- Learned window placement instead of fixed
- Adaptive window sizes based on content
- Sparse global attention for further efficiency
- Cross-window communication without full global

---

## Sources

### Primary References

**SAM Paper:**
- Kirillov et al., "Segment Anything" (arXiv:2304.02643)
- https://arxiv.org/abs/2304.02643
- Accessed: 2025-11-20

**Swin Transformer:**
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (arXiv:2103.14030)
- https://arxiv.org/abs/2103.14030
- Cited by 38,000+

### Technical Resources

**Medical SAM Adapter:**
- Wu et al., "Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation" (arXiv:2304.12620)
- https://arxiv.org/html/2304.12620v7
- Detailed ViT-H/16 configuration

**SAM Detailed Explanation:**
- Chau Tuan Kien, "Segment Anything Model (SAM) — Detailed Explanation"
- https://chautuankien.medium.com/segment-anything-model-sam-detailed-explanation-21698094cd56
- Accessed: 2025-11-20

### Code References

**Official SAM Repository:**
- https://github.com/facebookresearch/segment-anything
- Image encoder implementation
- Attention mechanism code

**Ultralytics SAM Documentation:**
- https://docs.ultralytics.com/reference/models/sam/modules/encoders/
- Encoder module documentation

### Additional Reading

**Swin Transformer Resources:**
- Microsoft GitHub: https://github.com/microsoft/Swin-Transformer
- Lightly AI Explanation: https://www.lightly.ai/blog/swin-transformer

**Attention Mechanism Research:**
- ScienceDirect: Parameter-efficient adaptation studies
- Nature Scientific Reports: Dual-scale attention methods

---

*Last Updated: 2025-11-20*
*Knowledge Domain: SAM Architecture - Attention Mechanisms*
*For: Karpathy Deep Oracle - ARR-COC Training System*
