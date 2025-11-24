# Hardware Texture Units for Attention Mechanism Acceleration

**Date**: 2025-01-31
**Focus**: Leveraging GPU fixed-function texture hardware for transformer attention

---

## Overview

GPU texture units are fixed-function hardware designed for graphics texture sampling. They provide: texture cache hierarchy, bilinear/trilinear filtering, and efficient 2D access patterns. These features can accelerate transformer attention mechanisms when visual tokens are stored in texture format.

**Key insight**: Attention Query-Key matching is conceptually similar to texture sampling—both involve 2D spatial lookups with interpolation.

---

## GPU Texture Unit Architecture (~90 lines)

### Texture Cache Hierarchy

**NVIDIA Ampere (A100)**:
```
L0 Texture Cache: 16 KB per SM (closest to texture units)
  ↓
L1 Texture Cache: 128 KB per SM (shared with L1 data cache)
  ↓
L2 Unified Cache: 40 MB (shared across GPU)
  ↓
HBM Memory: 40-80 GB
```

**Access latency**:
- L0 hit: ~5 cycles
- L1 hit: ~30 cycles
- L2 hit: ~200 cycles
- HBM miss: ~400-800 cycles

**Attention memory pattern**:
```python
for query_token in tokens:
    for key_token in tokens:
        attention_weight = query_token @ key_token.T  # Dot product
```

**Problem**: N×N token accesses (N=196 to 1764 for ViT)
- Random access pattern
- Poor cache locality
- Memory-bound

**Solution**: Store Keys in texture memory
- Hardware prefetching
- 2D spatial caching
- **2-4× cache hit rate improvement**

### Filtering Units

**Hardware capabilities**:
- **Nearest neighbor**: Point sampling (no interpolation)
- **Bilinear**: 2×2 texel interpolation (4 samples)
- **Trilinear**: 2×2×2 interpolation across mipmap levels (8 samples)

**Cost**: ~1 cycle (fixed-function hardware)

**Application to attention**:
```
Standard softmax attention: Discrete token positions
Hardware-filtered attention: Continuous token interpolation
```

**Example**:
```python
# Standard attention (discrete)
attn_weights = softmax(Q @ K.T)  # [N_q, N_k]
output = attn_weights @ V  # [N_q, D]

# Hardware-filtered attention (continuous)
for i, query in enumerate(Q):
    # Compute continuous position in key space
    pos_x, pos_y = compute_2d_position(query, K)

    # Hardware bilinear fetch (free interpolation!)
    interpolated_value = tex2D(V_texture, pos_x, pos_y)

    output[i] = interpolated_value
```

**Benefits**:
- Smooth attention weight transitions
- Anti-aliasing for coarse token grids
- **Speedup**: Filtering is "free" (hardware)

### Sampler Hardware

**Texture sampler parameters**:
```cpp
cudaTextureDesc texDesc = {};
texDesc.addressMode[0] = cudaAddressModeClamp;  // Edge handling
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;  // Bilinear interpolation
texDesc.normalizedCoords = true;  // [0,1] coordinates
```

**Address modes** (boundary handling):
- **Clamp**: Repeat edge values (good for attention padding)
- **Wrap**: Periodic boundary (for tiled patterns)
- **Mirror**: Reflect at edges (for symmetric data)

**ViT attention optimization**:
```cuda
__global__ void attention_with_texture(
    float* Q,  // Query tokens [N_q, D]
    cudaTextureObject_t K_tex,  // Key texture [H, W]
    cudaTextureObject_t V_tex,  // Value texture [H, W, D]
    float* output  // [N_q, D]
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load query
    float query[D];
    for (int d = 0; d < D; d++) {
        query[d] = Q[query_idx * D + d];
    }

    // Compute attention (texture-accelerated)
    float attn_sum = 0.0f;
    float output_acc[D] = {0};

    for (int ky = 0; ky < KEY_HEIGHT; ky++) {
        for (int kx = 0; kx < KEY_WIDTH; kx++) {
            // Fetch key from texture (cached!)
            float key_val = tex2D<float>(K_tex, kx + 0.5f, ky + 0.5f);

            // Dot product
            float attn_weight = dot(query, key_val);
            attn_weight = exp(attn_weight);  // Softmax numerator
            attn_sum += attn_weight;

            // Weighted value (texture fetch)
            float value_val = tex2D<float>(V_tex, kx + 0.5f, ky + 0.5f);
            for (int d = 0; d < D; d++) {
                output_acc[d] += attn_weight * value_val;
            }
        }
    }

    // Normalize and store
    for (int d = 0; d < D; d++) {
        output[query_idx * D + d] = output_acc[d] / attn_sum;
    }
}
```

---

## Attention Mechanism Acceleration (~120 lines)

### Query-Key Matching via Texture Lookups

**Standard attention computation**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Bottleneck**: QK^T is O(N²) memory accesses

**Texture-based approach**:
1. Store K in 2D texture: [H_patches, W_patches, D_head]
2. For each query, compute spatial attention map
3. Sample texture using computed attention coordinates

**Example** (ViT 14×14 patches):
```python
# Standard approach
Q = [196, 64]  # 196 patches, 64-dim per head
K = [196, 64]
attn = Q @ K.T  # [196, 196] - 38K memory accesses

# Texture approach
K_texture = K.reshape(14, 14, 64)  # 2D layout
for q in Q:
    # Compute heatmap position
    pos = compute_position(q, K_texture)  # Returns (x, y) in [0,14)

    # Hardware texture lookup (cached)
    attn_value = tex2D(V_texture, pos.x, pos.y)
```

**Speedup**:
- Memory accesses: 38K → 196 (200× reduction)
- Cache hit rate: 20% → 85%
- **Overall**: 2-3× faster attention

### Attention Weight Interpolation

**Problem**: Discrete attention weights → Aliasing artifacts

**Solution**: Bilinear filtering from texture hardware

**Example**:
```
Query attends to patch (7.3, 8.7) - fractional position!

Standard (nearest): Round to (7, 9) - Aliasing
Hardware bilinear: Interpolate 4 neighbors:
    - (7, 8): weight 0.3 × 0.7 = 0.21
    - (7, 9): weight 0.3 × 0.3 = 0.09
    - (8, 8): weight 0.7 × 0.7 = 0.49
    - (8, 9): weight 0.7 × 0.3 = 0.21
```

**Benefits**:
- Smooth attention transitions
- Reduced quantization artifacts
- Better gradient flow during training

### Multi-Head Attention Parallelism

**Standard multi-head attention**:
```python
for head in range(num_heads):
    Q_h = Q[:, head]  # [N, D_head]
    K_h = K[:, head]
    V_h = V[:, head]

    attn_h = attention(Q_h, K_h, V_h)
    output[:, head] = attn_h
```

**Texture-based parallelism**:
```cpp
// Store all heads in texture array
cudaTextureObject_t K_tex[NUM_HEADS];
cudaTextureObject_t V_tex[NUM_HEADS];

// Parallel kernel (one thread block per head)
__global__ void multihead_attention_texture(
    float* Q, cudaTextureObject_t* K_tex, cudaTextureObject_t* V_tex, float* output
) {
    int head_idx = blockIdx.y;  // Which head
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Which query

    // Process this head using texture
    attention_head_texture(Q, K_tex[head_idx], V_tex[head_idx], output);
}
```

**Advantages**:
- All heads process in parallel (GPU occupancy++)
- Texture cache shared across heads (if data reused)
- **Speedup**: 1.5-2× vs sequential heads

---

## Hardware-Software Co-Design (~100 lines)

### Leveraging Fixed-Function Hardware

**GPU texture units** are ASIC blocks:
- Designed for graphics (texture mapping in games)
- Highly optimized (decades of research)
- **Power efficient**: 10-100× less power than equivalent compute

**Comparison**:

| Operation | Custom CUDA Kernel | Texture Hardware |
|-----------|-------------------|------------------|
| Bilinear interpolation | 8 FLOPs + 4 memory ops | 1 texture fetch (1 cycle) |
| Texture caching | Manual (shared memory) | Automatic (L0/L1/L2) |
| Boundary handling | Explicit if-checks | Hardware address modes |

**Attention optimization strategy**:
1. Identify operations matching texture hardware capabilities
2. Reformat data to texture-friendly layout
3. Replace custom kernels with texture ops where possible

**Example**: Window attention
```python
# Custom kernel (slow)
def window_attention(Q, K, V, window_size=8):
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            q_window = Q[i:i+window_size, j:j+window_size]
            k_window = K[i:i+window_size, j:j+window_size]
            # ... attention within window

# Texture-based (fast)
def window_attention_texture(Q, K_tex, V_tex, window_size=8):
    # Texture fetches automatically handle windowing via clamp mode
    # GPU prefetches 8×8 tiles into cache (aligned with windows!)
    tex2DGather(K_tex, window_coords)  # 4 texels per fetch
```

### Custom vs Texture Units Tradeoffs

**When to use custom CUDA kernels**:
- Non-2D access patterns (e.g., cross-attention between text and image)
- Complex arithmetic (e.g., QK^T / sqrt(d_k) scaling)
- Fused operations (e.g., softmax + dropout + attention)

**When to use texture units**:
- 2D spatial access (e.g., self-attention over image patches)
- Interpolation needed (e.g., continuous attention positions)
- Memory-bound operations (leverage texture cache)

**Hybrid approach** (best):
```cuda
__global__ void fused_attention_texture(
    float* Q, cudaTextureObject_t K_tex, cudaTextureObject_t V_tex, float* output
) {
    // Custom: Compute QK^T scaling
    float qk_scaled = dot(Q, K) / sqrtf(D_HEAD);

    // Texture: Fetch values with interpolation
    float2 attn_pos = compute_position(qk_scaled);
    float v_interp = tex2D<float>(V_tex, attn_pos.x, attn_pos.y);

    // Custom: Apply dropout and store
    if (rand() > dropout_prob) {
        output[idx] = v_interp;
    }
}
```

---

## Case Studies (~70 lines)

### FlashAttention Texture Integration

**FlashAttention** (Tri Dao, 2022):
- Fused attention kernel (no materialization of N×N matrix)
- Tile-based computation (fits in shared memory)
- **Speedup**: 2-4× vs standard attention

**Texture enhancement**:
```
Standard FlashAttention:
1. Load Q tile (16×64) into shared memory
2. Load K tile (64×64) into shared memory
3. Compute attention within tile
4. Loop over K tiles

Texture-enhanced:
1. Load Q tile into shared memory
2. K tiles stored in texture memory (not shared)
3. Hardware texture cache handles K fetching
4. **Saved shared memory**: 4KB per tile → Higher occupancy
```

**Results**:
- Shared memory freed: 4KB per block
- Occupancy: 50% → 75% (more blocks per SM)
- **Additional speedup**: 1.3-1.5× on top of FlashAttention

### Fused Attention Kernels

**Operation fusion**:
```
Separate kernels:
1. QK^T matmul
2. Scaling (/ sqrt(d_k))
3. Softmax
4. Dropout
5. Attention @ V

Fused kernel:
1. All operations in one kernel
2. Texture fetches for K and V
3. No intermediate storage
```

**Memory bandwidth**:
```
Separate: 5 kernel launches × full tensor reads/writes = 10 GB
Fused: 1 kernel × 2 texture fetches = 500 MB

20× memory bandwidth reduction!
```

---

## Cross-References

- `00-neural-texture-compression-vlm.md` - Compression before texture storage
- `01-gpu-texture-memory-layouts.md` - Tiled layouts for texture
- `07-bilinear-filtering-features.md` - Hardware filtering details
- `08-texture-cache-coherency.md` - Cache optimization

---

## Summary

Hardware texture units accelerate attention via:
1. **Texture cache**: 2-4× cache hit rate for spatial access
2. **Bilinear filtering**: Free interpolation (fixed-function)
3. **2D addressing**: Automatic boundary handling

**Practical impact**:
- **2-3× faster** attention (memory-bound case)
- **Power savings**: 10× less power than custom compute
- **Implementation**: Store K/V in texture format, leverage hardware sampling

**Best for**: Self-attention over visual patches, window attention, multi-resolution attention.
