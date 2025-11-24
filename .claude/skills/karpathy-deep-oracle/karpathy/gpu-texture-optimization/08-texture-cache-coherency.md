# Texture Cache Coherency in Vision-Language Fusion

**Date**: 2025-01-31
**Focus**: Optimizing GPU cache for multimodal transformer memory access

---

## Overview

Vision-language models fuse visual tokens (2D spatial structure) with text tokens (1D sequential structure). These different access patterns create cache coherency challenges. Texture memory optimization improves cache hit rates for the visual modality.

---

## GPU Cache Hierarchy (~80 lines)

### L1 Texture Cache

**NVIDIA Ampere (A100)**:
- **Size**: 128 KB per SM
- **Associativity**: Varies (typically 4-8 way)
- **Line size**: 128 bytes
- **Optimized for**: 2D spatial locality

**Access pattern**:
```
Visual token (i, j) likely accesses neighbors:
(i-1,j-1) (i-1,j) (i-1,j+1)
(i,  j-1) (i,  j) (i,  j+1)
(i+1,j-1) (i+1,j) (i+1,j+1)

All 9 neighbors fit in 1-2 cache lines (texture layout)
```

### L2 Unified Cache

**Size**: 40 MB (shared across GPU)
**Sharing**: All SMs, all memory types

**Contention**:
```
Visual tokens: 32 × 196 × 768 × 2 = 9.6 MB
Text tokens: 32 × 77 × 768 × 2 = 3.8 MB
Total: 13.4 MB (fits in L2!)
```

**Cache partitioning**: L2 dynamically allocates to hottest data

### Memory Bandwidth Bottlenecks

**A100 specs**:
- **HBM bandwidth**: 1.6 TB/s
- **L2 bandwidth**: 4 TB/s (internal)
- **L1 bandwidth**: 10+ TB/s (per SM)

**Attention memory traffic**:
```
QK^T for 196 tokens:
- Read Q: 196 × 768 × 2 = 301 KB
- Read K: 196 × 768 × 2 = 301 KB
- Write output: 196 × 196 × 2 = 77 KB

Total: 679 KB per attention layer

If L2 miss: 679 KB ÷ 1.6 TB/s = 0.4 μs (slow)
If L2 hit: 679 KB ÷ 4 TB/s = 0.17 μs (fast)
```

---

## Access Pattern Optimization (~110 lines)

### Spatial Locality in Attention

**Self-attention over image patches**:
```
Query (7,7) attends to:
- Local neighbors (high weight): (6,6), (7,6), (8,8), ...
- Distant patches (low weight): (0,0), (13,13), ...
```

**Optimized access**:
1. **Tile-based processing**: Process 8×8 query tiles together
2. **Reuse keys**: Load key tile once for all queries in tile
3. **Texture memory**: Automatic caching of 2D neighbors

**CUDA implementation**:
```cuda
__global__ void tiledAttention(
    cudaTextureObject_t keyTex,  // Keys in texture
    float* queries,
    float* output
) {
    __shared__ float queryTile[64][768];  // 8×8 tile

    // Load query tile
    int tileId = blockIdx.x;
    for (int i = 0; i < 64; i++) {
        queryTile[i] = queries[tileId * 64 + i];
    }
    __syncthreads();

    // Process all queries in tile
    for (int q = 0; q < 64; q++) {
        for (int ky = 0; ky < 14; ky++) {
            for (int kx = 0; kx < 14; kx++) {
                // Texture fetch (cached!)
                float key = tex2D<float>(keyTex, kx, ky);
                float attn = dot(queryTile[q], key);
                // ...
            }
        }
    }
}
```

**Cache benefit**: Keys reused 64× (once per query in tile)

### Cache-Friendly Patch Ordering

**Row-major order** (standard ViT):
```
Patch sequence: (0,0), (0,1), ..., (0,13), (1,0), (1,1), ...
Access pattern: Sequential in row, jump between rows
Cache miss rate: ~60%
```

**Z-order** (texture-friendly):
```
Patch sequence: (0,0), (1,0), (0,1), (1,1), (2,0), (3,0), ...
Access pattern: 2×2 tiles, preserves 2D locality
Cache miss rate: ~20%
```

**Hilbert curve** (optimal):
```
Space-filling curve that never jumps far
Access pattern: Always visits nearby patches
Cache miss rate: ~15%
```

**Implementation**: Reorder patches during preprocessing

### Tiling Strategies

**Attention tiling** (FlashAttention-style):
```
Block size: 16×16 (256 tokens)
Tile Q, K, V into blocks
Process: Block(Q) × Block(K) → Block(output)

Memory access per block:
- Load Q block: 256 × 768 × 2 = 393 KB
- Load K block: 256 × 768 × 2 = 393 KB
- Write output: 256 × 256 × 2 = 131 KB
Total: 917 KB (fits in L2!)
```

**Texture memory benefit**:
- K blocks stored in texture format
- Hardware prefetching across blocks
- **Cache hit rate**: 75% → 90%

---

## Vision-Language Fusion Challenges (~90 lines)

### Different Access Patterns

**Visual tokens** (2D):
```
Token (i,j) neighbors: (i±1, j±1)
Access: 2D spatial
Cache: Texture cache (L1)
```

**Text tokens** (1D):
```
Token[t] neighbors: Token[t±1]
Access: Sequential
Cache: Data cache (L1)
```

**Cross-modal attention**:
```
Text query → Visual keys
- Query: Sequential access (1D)
- Keys: 2D access

Problem: Different caching strategies conflict!
```

### Cross-Modal Cache Sharing

**L2 cache partitioning**:
```
Ideal:
- 70% allocated to visual tokens (larger, 2D)
- 30% allocated to text tokens (smaller, 1D)

Actual:
- L2 dynamically allocates based on access frequency
- May evict visual tokens for text (suboptimal)
```

**Solution**: Bias L2 towards visual data
```cuda
// Hint to L2 cache (CUDA 11.8+)
cudaStreamSetAttribute(stream, cudaStreamAttributePreferredCacheConfig, cudaFuncCachePreferL2);
```

### Memory Pressure in Multimodal Models

**Memory footprint** (batch=32):
```
Visual tokens: 32 × 196 × 1024 × 2 = 12.9 MB
Text tokens: 32 × 77 × 1024 × 2 = 5.0 MB
Attention QKV: 3 × (12.9 + 5.0) = 53.7 MB
Total: 71.6 MB (exceeds L2 cache!)
```

**Cache thrashing**: Repeated eviction/reload of data

**Solutions**:
1. **Token compression**: 12.9 MB → 3.2 MB (4× compression)
2. **Fused kernels**: Reduce intermediate storage
3. **Gradient checkpointing**: Trade compute for memory

---

## Optimization Strategies (~70 lines)

### Prefetching

**Manual prefetching**:
```cuda
__global__ void attention WithPrefetch(
    cudaTextureObject_t keyTex,
    float* queries
) {
    // Prefetch next iteration's keys
    if (threadIdx.x == 0) {
        __prefetch_global(&keys[nextBlock]);
    }

    // Process current block
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float key = tex2D<float>(keyTex, kx, ky);
        // ... attention ...
    }
}
```

**Hardware prefetching**: Texture memory does this automatically

### Cache Blocking

**Standard attention** (no blocking):
```python
output = softmax(Q @ K.T) @ V  # Materializes N×N matrix
```

**Blocked attention**:
```python
for q_block in range(0, N, BLOCK_SIZE):
    for k_block in range(0, N, BLOCK_SIZE):
        attn_block = softmax(Q[q_block] @ K[k_block].T)
        output[q_block] += attn_block @ V[k_block]
```

**Cache benefit**: Each block fits in L2

### Batch Size Tuning

**Small batch** (e.g., batch=8):
- Memory footprint: 18 MB (fits in L2)
- Cache hit rate: 95%
- GPU utilization: 40% (underutilized)

**Large batch** (e.g., batch=128):
- Memory footprint: 288 MB (exceeds L2)
- Cache hit rate: 60%
- GPU utilization: 90%

**Optimal** (A100): Batch=32-64
- Memory: ~70-140 MB
- Cache hit: 80-85%
- GPU util: 80-90%

---

## Cross-References

- `01-gpu-texture-memory-layouts.md` - Tiled memory
- `02-hardware-texture-units-attention.md` - Cache hierarchy
- FlashAttention paper - Tiling strategies

---

## Summary

Texture cache coherency for VLMs:
- **2D spatial access**: Texture memory → 90% cache hit
- **Cross-modal fusion**: Separate caching for visual (2D) vs text (1D)
- **L2 pressure**: Token compression, fused kernels
- **Optimal batch size**: 32-64 for A100

**Practical impact**:
- **2× faster** cross-modal attention (cache optimization)
- **Memory savings**: 4× via token compression
