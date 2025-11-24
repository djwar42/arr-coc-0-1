# Interleaved-MRoPE (Multimodal Rotary Position Embedding)

**Category**: Concepts
**Related**: [architecture/01-positional-encoding.md](../architecture/01-positional-encoding.md), [codebase/02-rope2d.md](../codebase/02-rope2d.md)
**Code**: `qwen-vl-finetune/qwenvl/data/rope2d.py`

## What is Interleaved-MRoPE?

Interleaved-MRoPE is Qwen3-VL's **revolutionary positional encoding scheme** that allocates **full-frequency ranges** to three separate dimensions: temporal, height, and width.

**Key Innovation**: Instead of sharing frequencies across dimensions (like Qwen2-VL's M-RoPE), each dimension gets the complete frequency spectrum.

## Evolution of Position Encoding

### Standard RoPE (Language Models)
```python
# 1D sequence position
position_ids = [0, 1, 2, 3, 4, ..., L]  # Single dimension

# Frequency bands distributed across embedding dimensions
theta_i = 10000^(-2i/d)  for i in [0, d/2]
```

**Use**: Text-only transformers (Llama, GPT, etc.)

### M-RoPE (Qwen2-VL)
```python
# 3D position: [temporal, height, width]
position_ids = [
    [t1, t2, t3, ...],  # Temporal dimension
    [h1, h2, h3, ...],  # Height dimension
    [w1, w2, w3, ...]   # Width dimension
]

# Frequencies SHARED across dimensions
# Each dimension gets subset of frequency bands
temporal_freqs = theta[0:d/3]
height_freqs = theta[d/3:2d/3]
width_freqs = theta[2d/3:d]
```

**Limitation**: Frequency sharing reduces representational capacity

### Interleaved-MRoPE (Qwen3-VL)
```python
# 3D position: [temporal, height, width]
position_ids = [
    [t1, t2, t3, ...],  # Temporal dimension
    [h1, h2, h3, ...],  # Height dimension
    [w1, w2, w3, ...]   # Width dimension
]

# FULL frequencies for EACH dimension
# Each dimension gets ALL frequency bands (interleaved)
temporal_freqs = theta[0::3]  # Every 3rd frequency
height_freqs = theta[1::3]    # Every 3rd frequency (offset 1)
width_freqs = theta[2::3]     # Every 3rd frequency (offset 2)

# Result: All three dimensions span full frequency spectrum
```

**Advantage**: Maximum representational capacity for each dimension

## Mathematical Formulation

### RoPE Basics

For a position `m` and frequency `θ_i`:

```
Rotation matrix R(m, θ_i):
  [cos(m·θ_i)  -sin(m·θ_i)]
  [sin(m·θ_i)   cos(m·θ_i)]

Apply to query/key pairs:
  q' = R(m, θ) · q
  k' = R(n, θ) · k

Attention score preserves relative distance:
  q' · k' = q · R(m-n, θ) · k
```

### Interleaved-MRoPE

For 3D position `(t, h, w)`:

```python
# Frequency bands
theta = [10000^(-2i/d) for i in range(d)]

# Interleaved allocation
temporal_theta = theta[0::3]  # Indices: 0, 3, 6, 9, ...
height_theta = theta[1::3]    # Indices: 1, 4, 7, 10, ...
width_theta = theta[2::3]     # Indices: 2, 5, 8, 11, ...

# Apply rotations
R_t = Rotation(t, temporal_theta)
R_h = Rotation(h, height_theta)
R_w = Rotation(w, width_theta)

# Combined transformation
q' = R_w(R_h(R_t(q)))
```

**Result**: Each dimension encoded across ALL frequency bands

## Why Full-Frequency Allocation?

### Problem with Shared Frequencies

**Limited expressiveness**:
```
M-RoPE (Qwen2):
  Temporal: θ[0:d/3]   → Only low frequencies
  Height:   θ[d/3:2d/3] → Only mid frequencies
  Width:    θ[2d/3:d]   → Only high frequencies
```

**Issues**:
- Temporal dimension can't express fine-grained details (missing high frequencies)
- Spatial dimensions can't capture long-range patterns (missing low frequencies)
- Arbitrary frequency division

### Solution: Interleaved Allocation

**Full expressiveness**:
```
Interleaved-MRoPE (Qwen3):
  Temporal: θ[0::3] → Low AND high frequencies
  Height:   θ[1::3] → Low AND high frequencies
  Width:    θ[2::3] → Low AND high frequencies
```

**Benefits**:
- ✅ Each dimension spans full frequency spectrum
- ✅ Capture both fine-grained and long-range patterns
- ✅ Symmetric treatment of dimensions
- ✅ Better long-horizon video reasoning

## Long-Horizon Video Reasoning

### Challenge

Videos with many frames need to maintain:
- **Temporal coherence**: Events separated by many frames
- **Spatial consistency**: Objects moving across frames
- **Detail preservation**: Fine-grained changes over time

### How Interleaved-MRoPE Helps

**Low frequencies** (temporal):
```python
theta_low = 10000^(-2·0/d) = 1.0

# Position 0:   cos(0·1.0) = 1.0
# Position 100: cos(100·1.0) ≈ 0.86

# Still captures relationship across 100 frames!
```

**High frequencies** (temporal):
```python
theta_high = 10000^(-2·(d/6)/d) = 0.1  # d/6 because every 3rd

# Position 0: cos(0·0.1) = 1.0
# Position 1: cos(1·0.1) = 0.995

# Distinguishes adjacent frames!
```

**Result**: Can model both:
- Frame-to-frame transitions (high freq)
- Long-term event sequences (low freq)

## Example: 100-Frame Video

### Input
```
Video: 100 frames, each 224×224
After sampling: 50 frames (2-frame temporal patches)
```

### Position IDs (Qwen3)
```python
# Temporal positions (with timestamp tokens)
# Each frame gets its own timestamp token
Frames: <t0>, <t1>, <t2>, ..., <t49>

# Position IDs for each frame's patches (4×4 grid = 16 patches)
Frame 0 patches: t=[0,0,0,0,...], h=[0,0,1,1,...], w=[0,1,0,1,...]
Frame 1 patches: t=[0,0,0,0,...], h=[0,0,1,1,...], w=[0,1,0,1,...]  # t still 0!
...
Frame 49 patches: t=[0,0,0,0,...], h=[0,0,1,1,...], w=[0,1,0,1,...] # t still 0!

# Temporal info encoded in <t0>...<t49> tokens, NOT in position IDs
```

### Why This Works

**Separation of concerns**:
- Position IDs: Encode **spatial** structure (h, w) with full frequencies
- Timestamp tokens: Encode **temporal** structure with learnable embeddings
- Result: Best of both worlds

**vs Qwen2.5-VL**:
```python
# Qwen2.5: Temporal in position IDs
Frame 0: t=[0·25, 0·25, ...] = [0, 0, ...]
Frame 1: t=[1·25, 1·25, ...] = [25, 25, ...]
...
Frame 49: t=[49·25, 49·25, ...] = [1225, 1225, ...]

# Problem: Large temporal indices → potential overflow
```

## Comparison Table

| Feature | Standard RoPE | M-RoPE (Qwen2) | Interleaved-MRoPE (Qwen3) |
|---------|--------------|----------------|---------------------------|
| Dimensions | 1 (sequence) | 3 (t, h, w) | 3 (t, h, w) + timestamps |
| Frequency allocation | Full spectrum | Shared (1/3 each) | Interleaved (full each) |
| Temporal encoding | N/A | Position IDs | Timestamp tokens |
| Long-horizon support | Limited | Good | Excellent |
| Spatial expressiveness | N/A | Limited | Full |
| Video frame density | N/A | Fixed | Variable |

## ARR-COC Compatibility

Interleaved-MRoPE is **perfect for ARR-COC** variable compression:

**Why?**
- ✅ Position IDs generated **after** tokenization
- ✅ Each token gets unique (t, h, w) regardless of compression
- ✅ Works with variable token budgets per patch
- ✅ Timestamp approach allows sparse/dense frame sampling

**No modifications needed!**

```python
# ARR-COC with variable budgets
relevance_scores = allocator(image, query)
token_budgets = map_to_budgets(relevance_scores)  # 64-400 per patch

# Process with variable budgets
for patch, budget in zip(patches, token_budgets):
    max_pixels = budget * (patch_size ** 2)
    resized = smart_resize(patch, max_pixels=max_pixels)
    features = vit_encode(resized)

# M-RoPE automatically handles variable sequence lengths!
position_ids = get_rope_index_3(...)  # Works seamlessly
```

## Related Topics

- [architecture/01-positional-encoding.md](../architecture/01-positional-encoding.md) - Full implementation
- [architecture/03-timestamp-alignment.md](../architecture/03-timestamp-alignment.md) - Timestamp tokens
- [codebase/02-rope2d.md](../codebase/02-rope2d.md) - Code walkthrough
- [concepts/02-timestamp-encoding.md](02-timestamp-encoding.md) - Temporal modeling

## Code Reference

**File**: `qwen-vl-finetune/qwenvl/data/rope2d.py`
**Function**: `get_rope_index_3()` (lines 112-235)
**Key line**: Line 197 - `t_index = torch.arange(llm_grid_t)` (always returns [0,0,0,...])
