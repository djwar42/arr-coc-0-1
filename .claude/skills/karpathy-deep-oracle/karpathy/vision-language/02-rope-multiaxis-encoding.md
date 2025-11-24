# RoPE Multi-Axis Position Encoding

## Overview

Rotary Position Embedding (RoPE) extends beyond 1D text sequences to encode multi-dimensional spatial and temporal positions in vision-language models. Multi-axis RoPE applies rotational position encoding across multiple coordinate axes (height, width, time) to capture the rich positional structure of images and videos.

This document covers the mathematical foundations, architectural extensions, and implementation strategies for multi-axis RoPE in vision transformers and multimodal models.

## RoPE Fundamentals

### Why Rotation Over Addition

Traditional position encodings (learned absolute, sinusoidal) add positional information to embeddings. RoPE instead **rotates** embeddings in high-dimensional space, providing several advantages:

**Key benefits:**
- **Relative position dependency**: Attention scores naturally depend on relative distances through dot products of rotated vectors
- **Length extrapolation**: Can generalize to longer sequences than seen during training
- **No additional parameters**: Position information encoded through rotation matrices (though frequencies can be learned)
- **Theoretical elegance**: Rotation preserves vector norms and enables geometric interpretation

From [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021):
> RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation.

### 1D RoPE Mathematics

For 1D sequences (text), RoPE divides the embedding dimension D into D/2 pairs and rotates each pair by an angle proportional to position:

**Rotation formula for position t:**
```
RoPE_1D(x, t) = [
  [cos(ω₁t)  -sin(ω₁t)   0         0        ...  0         0       ]
  [sin(ω₁t)   cos(ω₁t)   0         0        ...  0         0       ]
  [0          0          cos(ω₂t) -sin(ω₂t)  ...  0         0       ]
  [0          0          sin(ω₂t)  cos(ω₂t)  ...  0         0       ]
  [⋮          ⋮          ⋮         ⋮         ⋱    ⋮         ⋮       ]
  [0          0          0         0        ...  cos(ω_{D/2}t) -sin(ω_{D/2}t)]
  [0          0          0         0        ...  sin(ω_{D/2}t)  cos(ω_{D/2}t)]
] × x
```

**Frequency assignment:**
The i-th dimension pair rotates at frequency ωᵢ, typically using log-spaced frequencies:

```
ωᵢ = ω_min × (ω_max / ω_min)^(i / (D/2 - 1))
```

**Common frequency ranges:**
- Language models: ω_min = 1.0, ω_max = 10,000
- Vision models: ω_min = 0.2-1.0, ω_max = 20-100 (due to shorter spatial sequences)

From [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/) (accessed 2025-01-31):
> As more frequencies are added, their periodic oscillations cancel each other out, resulting in an attention map concentrated at a specific 1d coordinate position.

## Multi-Axis RoPE Extensions

### 2D RoPE for Images

Images require encoding both height and width positions. Two main approaches:

#### Axial RoPE (Simple Decomposition)

**Strategy:** Apply 1D RoPE independently to each spatial axis

```
- First D/2 dimensions: Rotate according to x-position (width)
- Remaining D/2 dimensions: Rotate according to y-position (height)
```

**Implementation:**
```python
# Axial RoPE
theta_x = freqs * x_positions  # Shape: [H, W, D/4]
theta_y = freqs * y_positions  # Shape: [H, W, D/4]

# First half rotates on x-axis, second half on y-axis
cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
```

**Limitation:** Cannot uniquely attend to specific (key, relative position) pairs
- Attending to a token means attending to all tokens in the same row OR column
- Half the query rotates based only on x, half only on y
- No coupling between spatial dimensions

From [N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/) (Jerry Xiong, accessed 2025-01-31):
> The first half of a query, which rotates according to x-position, contributes the same amount to the attention score for a key regardless of the key's y-position. Similarly, the second half contributes the same amount regardless of x-position.

#### Mixed RoPE (Full 2D Coupling)

**Strategy:** Rotate each dimension pair according to arbitrary 2D directions, not just axes

From [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298) (Heo et al., 2024):

**Key insight:** Instead of constraining frequency directions to [1,0] and [0,1], use unit vectors distributed across the full 2D circle:

```
θᵢ = ωᵢ × (uᵢ · position_2D)
```

Where:
- uᵢ = unit direction vector for i-th dimension pair
- position_2D = [x, y] coordinates
- ωᵢ = frequency magnitude

**Direction selection strategies:**

1. **Random initialization + learning** (original Mixed RoPE):
   - Sample uᵢ uniformly from unit circle
   - Treat frequency vectors fᵢ = ωᵢ × uᵢ as learnable parameters

2. **Golden ratio rotation** (Golden Gate RoPE):
   - Deterministic initialization using φ = (1 + √5)/2 (golden ratio)
   - Rotate i-th direction by angle: i × π/φ ≈ 1.9416 radians
   - No learning required, better extrapolation

From [N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/):
> By selecting {uᵢ} uniformly from the unit circle rather than constraining them to axis-aligned directions, RoPE can produce concentrated attention maps in 2D!

**Performance comparison (ImageNet-1K ViT B/16):**

| Method | Learned | ω_min | ω_max | Valid Acc (224px) | Valid Acc (384px) |
|--------|---------|-------|-------|-------------------|-------------------|
| SinCos | - | 1.0 | 100.0 | 78.71% | 77.29% |
| Axial RoPE | - | 0.2 | 20.0 | 79.58% | 79.57% |
| Mixed RoPE | ✓ | 0.2 | 20.0 | 79.73% | 79.83% |
| Golden Gate RoPE | - | 0.2 | 20.0 | 79.78% | 80.41% |

### 3D RoPE for Video

Video adds temporal dimension to spatial encoding. Several approaches:

#### RoPE-3D (Separate Temporal + Spatial)

From [VRoPE: Rotary Position Embedding for Video Large Language Models](https://arxiv.org/abs/2502.11664):

**Approach:** Separate positional encodings for spatial (width, height) and temporal (frame index) dimensions

**Limitation:** When video tokens are flattened into 1D sequence, temporal coherence can be lost
- Spatial neighbors from same frame may be far apart in token sequence
- RoPE-3D fails to maintain proper temporal relationships

#### Interleaved M-RoPE (Qwen3-VL Approach)

From [Qwen3-VL Technical Documentation](https://qwen.ai/blog) and [Revisiting Multimodal Positional Encoding in Vision-Language Models](https://arxiv.org/html/2510.23095v1) (accessed 2025-01-31):

**Strategy:** Fine-grained, round-robin distribution of frequencies across all three axes

**Key properties:**
- Full-frequency allocation over time, width, and height
- Each axis encoded with complete frequency spectrum
- Better long-horizon video reasoning
- Robust positional embeddings for variable resolutions

**MRoPE-Interleave implementation:**
```python
# Instead of dividing D into 3 blocks (t, h, w)
# Interleave frequency assignments across axes
for i in range(D // 2):
    axis = i % 3  # Round-robin: time, height, width, time, ...
    if axis == 0:
        theta[i] = freq[i] * temporal_position
    elif axis == 1:
        theta[i] = freq[i] * height_position
    else:
        theta[i] = freq[i] * width_position
```

**Benefits over block allocation:**
- Ensures each dimension gets high AND low frequencies
- Prevents frequency starvation on any axis
- More balanced spatial-temporal encoding

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v1):
> Increasing the proportion of channels assigned to the temporal axis reduces the available high-frequency capacity for spatial dimensions. MRoPE-Interleave employs a fine-grained, round-robin distribution of channels to ensure each axis is encoded with the full frequency spectrum.

### 4D RoPE and Beyond

From [4D Rotary Position Embeddings](https://www.emergentmind.com/topics/4d-rotary-position-embeddings) (accessed 2025-01-31):

**Applications:**
- Spatiotemporal data (3D space + time)
- Multi-resolution representations
- Cross-modal alignment (image + audio + text positions)

**Mathematical extension:**
```
RoPE_ND(x, t_N) = rotation_matrix(ω₁⟨u₁, t⟩, ω₂⟨u₂, t⟩, ..., ω_{D/2}⟨u_{D/2}, t⟩) × x
```

Where:
- t_N = N-dimensional position vector
- uᵢ = N-dimensional unit direction vectors
- ⟨uᵢ, t⟩ = dot product (position measured along direction uᵢ)

## Axis Decomposition Strategies

### Frequency Allocation

**Critical design decision:** How to distribute D/2 frequency pairs across N axes?

**Option 1: Block allocation**
```
Dimensions 0 to D/(2N): Axis 1
Dimensions D/(2N) to 2D/(2N): Axis 2
...
```
- Simpler implementation
- Each axis gets D/(2N) frequencies
- Risk of frequency starvation on some axes

**Option 2: Interleaved allocation**
```
Dimension i assigned to axis (i mod N)
```
- More balanced frequency distribution
- Every axis gets full frequency spectrum
- Better for video and high-dimensional positions

### Frequency Magnitude Selection

**Key principle:** Adjust frequency ranges based on expected position ranges

**For normalized positions in [-1.0, 1.0]:**

| Domain | ω_min | ω_max | Rationale |
|--------|-------|-------|-----------|
| Language (long sequences) | 1.0 | 10,000 | Need very low frequencies for distant tokens |
| Vision (224×224) | 0.2-1.0 | 20-100 | Shorter spatial extents, higher minimum frequency |
| Video (temporal) | 0.1-0.5 | 10-50 | Moderate temporal distances |

From [N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/):
> Since the sequence lengths relevant for language modeling are typically much longer than the side length of image inputs for vision transformers, the minimum and maximum frequency magnitudes should be adjusted to compensate.

### Zero-Frequency Components

**Recent finding:** Setting some frequencies to zero can improve performance

From ImageNet experiments (Jerry Xiong, 2025-01-31):
- Setting 8/32 frequencies to zero improved validation accuracy
- Inspired by ModdedNanoGPT and learned frequency analysis
- Frequencies learned to be near-zero during Mixed RoPE training

**Implementation:**
```python
n_zero_freqs = round(p_zero_freqs * n_freqs)
omega = torch.cat([
    torch.zeros(n_zero_freqs),
    min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs)
])
```

## Vision-Language Applications

### Qwen3-VL M-RoPE Architecture

From [Qwen3-VL Documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl) (accessed 2025-01-31):

**Design:**
- Interleaved M-RoPE for temporal + spatial encoding
- Dynamic resolution handling through position scaling
- Compatible with both image and video inputs

**Key features:**
1. Full-frequency allocation across all axes
2. Position scaling for resolution changes at inference
3. No learned parameters (purely geometric)

### Dynamic Resolution Handling

**Challenge:** Train at one resolution (e.g., 224×224), infer at another (e.g., 384×384)

**Solution:** Scale positions to maintain [-1.0, 1.0] range

```python
# Training: 224×224 patches span [-1.0, 1.0]
train_x = torch.linspace(-1.0, 1.0, 224 // patch_size)

# Inference: 384×384 patches still span [-1.0, 1.0]
# Adjacent patches are closer together in coordinate space
infer_x = torch.linspace(-1.0, 1.0, 384 // patch_size)
```

**Results (ViT B/16 trained at 224×224):**

| Method | 224×224 (train) | 384×384 (test) | Δ Accuracy |
|--------|-----------------|----------------|------------|
| SinCos | 78.71% | 77.29% | -1.42% |
| Axial RoPE | 79.58% | 79.57% | -0.01% |
| Mixed RoPE | 79.73% | 79.83% | +0.10% |
| Golden Gate RoPE | 79.78% | 80.41% | +0.63% |

**Key insight:** RoPE-based methods maintain or even improve accuracy at higher resolutions!

### Temperature Scaling for Resolution Changes

**Technique:** Adjust softmax temperature to account for increased token count

```python
temp = log(new_res² / patch_size²) / log(old_res² / patch_size²)
attention_scores = attention_logits / temp
```

From ImageNet-1K experiments:
- 384×384 with temperature scaling: 80.41% (Golden Gate RoPE)
- Better than training resolution (79.78% at 224×224)

## Implementation Details

### Precomputed Rotation Matrices

**For fixed input sizes** (e.g., ViTs with constant image resolution):

```python
class GoldenGateRoPE2D(nn.Module):
    def __init__(self, image_size, n_heads, head_dim, min_freq, max_freq):
        # Precompute all rotation angles
        theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
        self.register_buffer("cos_HWhF", torch.cos(theta_HWhF))
        self.register_buffer("sin_HWhF", torch.sin(theta_HWhF))

    def forward(self, input_NHWhd):
        x, y = input_NHWhd.chunk(2, dim=-1)
        x_out = x * self.cos_HWhF - y * self.sin_HWhF
        y_out = x * self.sin_HWhF + y * self.cos_HWhF
        return torch.cat([x_out, y_out], dim=-1)
```

**Benefits:**
- No runtime computation of sin/cos
- Fixed memory cost
- Fast inference

### Dynamic Position Input

**For variable-length sequences** (language models, variable resolution):

```python
class RoPEND(nn.Module):
    def __init__(self, pos_dim, n_heads, head_dim, min_freq, max_freq):
        # Store frequency vectors only
        self.register_buffer("freqs_hFP", directions_hFP * omega_F)

    def forward(self, input_NLhd, pos_NLP):
        # Compute rotations on-the-fly
        theta_NLhF = (self.freqs_hFP * pos_NLP[..., None, None, :]).sum(dim=-1)
        cos_NLhF = torch.cos(theta_NLhF)
        sin_NLhF = torch.sin(theta_NLhF)
        # Apply rotations...
```

**Trade-off:**
- More flexible (handles any position)
- Slightly slower (computes sin/cos each forward pass)
- Lower memory (only stores frequencies, not full rotation matrices)

### Frequency Direction Initialization

**Golden ratio method** (recommended for fixed frequencies):

From [N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/):

```python
# 2D case
phi = (math.sqrt(5) - 1) * math.pi / 2  # Golden ratio angle
for i in range(n_freqs):
    angle = i * phi
    directions[i] = [math.cos(angle), math.sin(angle)]
```

**N-dimensional case:**
```python
def phi(m):
    """Compute m-dimensional golden ratio"""
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x

def make_directions_nd(n_freqs, pos_dim):
    g = phi(pos_dim)
    alpha = (1.0 / g) ** torch.arange(1, pos_dim + 1)
    i = torch.arange(1, n_freqs + 1).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    # Map to Gaussian via inverse CDF, then normalize
    directions = torch.erfinv(2.0 * z - 1.0)
    return directions / directions.norm(dim=1, keepdim=True)
```

## Code Patterns

### Minimal 2D RoPE Implementation

```python
def rope_2d_forward(q, k, h_pos, w_pos, freqs_h, freqs_w):
    """
    Args:
        q, k: [batch, heads, height*width, head_dim]
        h_pos, w_pos: [height, width] position grids
        freqs_h, freqs_w: [head_dim // 4] frequency arrays
    """
    batch, heads, seq_len, head_dim = q.shape

    # Split into dimension pairs for rotation
    q1, q2, q3, q4 = q.chunk(4, dim=-1)
    k1, k2, k3, k4 = k.chunk(4, dim=-1)

    # Compute rotation angles
    theta_h = freqs_h * h_pos.flatten().unsqueeze(-1)  # [H*W, D/4]
    theta_w = freqs_w * w_pos.flatten().unsqueeze(-1)

    cos_h, sin_h = torch.cos(theta_h), torch.sin(theta_h)
    cos_w, sin_w = torch.cos(theta_w), torch.sin(theta_w)

    # Apply rotations (first half on height, second half on width)
    q_out = torch.cat([
        q1 * cos_h - q2 * sin_h,  # Pair 1 rotates on height
        q1 * sin_h + q2 * cos_h,
        q3 * cos_w - q4 * sin_w,  # Pair 2 rotates on width
        q3 * sin_w + q4 * cos_w
    ], dim=-1)

    k_out = torch.cat([
        k1 * cos_h - k2 * sin_h,
        k1 * sin_h + k2 * cos_h,
        k3 * cos_w - k4 * sin_w,
        k3 * sin_w + k4 * cos_w
    ], dim=-1)

    return q_out, k_out
```

### Frequency Calculation

```python
def get_rope_frequencies(head_dim, min_freq=0.2, max_freq=20.0, p_zero=0.0):
    """Generate log-spaced frequencies with optional zero components"""
    n_freqs = head_dim // 2
    n_zero = int(p_zero * n_freqs)

    freqs = torch.cat([
        torch.zeros(n_zero),
        min_freq * (max_freq / min_freq) ** torch.linspace(
            0, 1, n_freqs - n_zero
        )
    ])
    return freqs
```

### Rotation Matrix Construction

```python
def build_2d_rotation_matrix(theta):
    """
    Args:
        theta: [n_freqs] rotation angles
    Returns:
        rotation_matrix: [2*n_freqs, 2*n_freqs] block-diagonal
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Build block-diagonal matrix
    blocks = []
    for c, s in zip(cos_theta, sin_theta):
        blocks.append(torch.tensor([[c, -s], [s, c]]))

    return torch.block_diag(*blocks)
```

## Performance Considerations

### Computational Complexity

**RoPE operations per token:**
- Frequency computation: O(D) multiplications
- Sin/cos evaluation: O(D) transcendental ops
- Rotation application: O(D) multiply-adds

**Optimization opportunities:**
1. **Precomputation**: Store cos/sin for fixed positions
2. **Fused kernels**: Combine rotation with attention QK computation
3. **Low precision**: bfloat16 sufficient for rotations

### Memory Requirements

**Precomputed approach** (fixed resolution):
```
Memory = H × W × n_heads × (head_dim / 2) × 2 (cos + sin) × 4 bytes
```

Example (224×224 image, 16×16 patches, 12 heads, head_dim=64):
```
14 × 14 × 12 × 32 × 2 × 4 = 301 KB per layer
```

**Dynamic approach:**
```
Memory = n_heads × (head_dim / 2) × pos_dim × 4 bytes (frequency vectors only)
```

Example (12 heads, head_dim=64, 2D positions):
```
12 × 32 × 2 × 4 = 3 KB per layer
```

### Training Stability

**Observations from experiments:**

1. **Frequency magnitude tuning is critical**
   - Too high: Poor local attention, unstable gradients
   - Too low: Cannot distinguish distant positions
   - Sweet spot: ω_min = 0.2-1.0, ω_max = 20-100 for vision

2. **Learned vs fixed frequencies**
   - Mixed RoPE (learned): Better in-distribution performance
   - Golden Gate RoPE (fixed): Better extrapolation to new resolutions
   - Learned frequencies tend toward zero during training (suggests redundancy)

3. **Initialization matters for learned approaches**
   - Random uniform from circle: Works but suboptimal
   - Golden ratio spacing: Better starting point
   - Large initialization std (0.5) needed for learned absolute PE

## Best Practices

### Choosing RoPE Variant

**Decision tree:**

```
Fixed input size (e.g., ViT)?
├─ Yes: Use precomputed rotation matrices
│   └─ Golden Gate RoPE for best extrapolation
└─ No: Use dynamic position input
    └─ Consider Mixed RoPE if training from scratch
```

**For vision transformers:**
1. Start with Golden Gate RoPE (π/φ spacing)
2. Tune frequency range (search ω_min, ω_max)
3. Try p_zero_freqs ≈ 0.25 (25% frequencies set to zero)
4. Use position scaling for resolution changes

**For video models:**
1. Use interleaved M-RoPE (Qwen3-VL approach)
2. Lower frequencies for temporal axis (longer distances)
3. Test with variable-length videos during validation

### Hyperparameter Tuning

**Priority order:**
1. **Frequency range** (ω_min, ω_max): Highest impact
2. **Zero frequency ratio** (p_zero_freqs): Secondary optimization
3. **Direction spacing**: Only if using learned frequencies

**Typical ranges:**
```python
# Vision (224×224 images, normalized positions)
omega_min = [0.2, 0.5, 1.0]  # Search these values
omega_max = omega_min * [20, 50, 100]

# Video (temporal + spatial)
omega_min_temporal = [0.1, 0.2, 0.5]
omega_min_spatial = [0.5, 1.0, 2.0]
```

### Debugging Tips

**Visualization: Cosine similarity heatmaps**
```python
def visualize_rope_attention(model, img_size=(14, 14)):
    """Plot attention pattern induced by RoPE"""
    query = torch.randn(1, n_heads, 1, head_dim)

    similarities = []
    for h in range(img_size[0]):
        for w in range(img_size[1]):
            pos = torch.tensor([[h, w]])
            rotated_q = rope_module(query, pos)
            sim = (query * rotated_q).sum() / (query.norm() * rotated_q.norm())
            similarities.append(sim)

    plt.imshow(torch.tensor(similarities).reshape(img_size))
```

**Expected pattern:** Concentrated peak at queried position, gradual decay with distance

**Check for issues:**
- Flat heatmap → Frequencies too low or too high
- Multiple peaks → Frequency aliasing (reduce ω_max)
- No central peak → Implementation bug in rotation

## Limitations and Challenges

### Current Limitations

1. **Computational overhead**: Sin/cos operations slower than learned embeddings
2. **Hyperparameter sensitivity**: Frequency ranges must be tuned per domain
3. **Mixed results on learning**: Learned frequency vectors sometimes underperform fixed
4. **High-dimensional positions**: Direction selection becomes more complex in >3D

### Open Questions

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v1):

1. **Optimal frequency allocation**: Block vs interleaved vs learned allocation?
2. **Asymmetric axis treatment**: Should temporal get different frequency distribution than spatial?
3. **Cross-modal alignment**: How to handle different position spaces (pixels vs mel-frequency bins)?

### Active Research Directions

- **Adaptive RoPE**: Dynamically adjust frequencies based on input content
- **Hierarchical RoPE**: Different frequency ranges at different transformer layers
- **Compressed RoPE**: Reduce computational cost while preserving benefits
- **Multi-scale RoPE**: Encode multiple resolution levels simultaneously

## Sources

### Papers

**Core RoPE:**
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021 (original RoPE paper)
- [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298) - Heo et al., 2024 (Mixed RoPE, ECCV 2024)

**Multi-axis extensions:**
- [Revisiting Multimodal Positional Encoding in Vision-Language Models](https://arxiv.org/html/2510.23095v1) - Huang et al., 2025 (accessed 2025-01-31)
- [VRoPE: Rotary Position Embedding for Video Large Language Models](https://arxiv.org/abs/2502.11664) - Liu et al., 2025 (accessed 2025-01-31)

**Applications:**
- [Qwen3-VL Technical Report](https://qwen.ai/blog) - Qwen Team, 2025 (accessed 2025-01-31)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - Ravi et al., 2024

### Web Resources

- [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/) - LearnOpenCV (accessed 2025-01-31)
- [N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/) - Jerry Xiong, 2025 (accessed 2025-01-31)
- [4D Rotary Position Embeddings](https://www.emergentmind.com/topics/4d-rotary-position-embeddings) - Emergent Mind (accessed 2025-01-31)
- [Multi-Modal Rotary Position Embedding](https://www.emergentmind.com/topics/multi-modal-rotary-position-embedding-mm-rope) - Emergent Mind (accessed 2025-01-31)

### Implementation References

- [GitHub - naver-ai/rope-vit](https://github.com/naver-ai/rope-vit) - Official PyTorch implementation (accessed 2025-01-31)
- [Qwen3-VL HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl) - (accessed 2025-01-31)

---

**Document version:** 1.0
**Last updated:** 2025-01-31
**Lines:** 648
