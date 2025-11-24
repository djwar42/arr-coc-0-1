# GPU Bilinear Filtering for Learned Visual Features

**Date**: 2025-01-31
**Focus**: Hardware interpolation for smooth VLM token transitions

---

## Overview

GPU bilinear filtering interpolates 2×2 texels in hardware (1 cycle). For VLMs, enables smooth attention transitions, continuous token positions, and anti-aliasing for coarse grids.

---

## Hardware (~60 lines)

**Bilinear interpolation** at position (2.7, 3.4):
```
Weights:
- (2,3): 0.3 × 0.6 = 0.18
- (3,3): 0.7 × 0.6 = 0.42
- (2,4): 0.3 × 0.4 = 0.12
- (3,4): 0.7 × 0.4 = 0.28

Result = weighted sum of 4 neighbors
Cost: 1 texture fetch (hardware)
```

**CUDA setup**:
```cpp
texDesc.filterMode = cudaFilterModeLinear;  // Bilinear
float val = tex2D<float>(tex, 2.7f, 3.4f);  // Interpolated
```

---

## Applications (~100 lines)

### Smooth Attention Transitions

**Problem**: Discrete tokens cause aliasing

**Solution**: Continuous attention positions
```cuda
__global__ void smoothAttention(cudaTextureObject_t attnTex) {
    // Fractional position
    float pos = threadIdx.x + 0.5f;

    // Hardware bilinear
    float attn = tex2D<float>(attnTex, pos, 0);
}
```

### Continuous Token Resolution

**Dynamic token budgets**:
```python
# Continuous scaling
resolution = 0.7  # Between 196 and 1764 tokens
num_tokens = 196 + (1764 - 196) * resolution

# Sample at fractional positions (bilinear)
for i in range(int(num_tokens)):
    pos = i / num_tokens
    token = tex2D(features, pos * width, pos * height)
```

### ARR-COC Smooth Saccades

**Relevance shifts**: Smooth focus interpolation
```cuda
// Focus moves from (5.2, 7.8) to (8.1, 9.3)
for (float t = 0; t < 1.0f; t += 0.1f) {
    float focusX = lerp(5.2f, 8.1f, t);
    float focusY = lerp(7.8f, 9.3f, t);

    // Smooth foveation (bilinear + mipmap)
    float token = tex2DLod<float>(image, x, y, compute_lod(focusX, focusY));
}
```

**Result**: No jarring transitions during attention shifts

---

## Performance (~50 lines)

**Hardware vs Manual**:

| Method | Latency | Code |
|--------|---------|------|
| Manual | 10 cycles | 15 lines CUDA |
| Hardware | 1 cycle | 1 line (tex2D) |

**Bandwidth**: Same (neighbors cached together)

**Quality**: Identical (IEEE float precision)

---

## Cross-References

- `02-hardware-texture-units-attention.md` - Texture units
- `04-anisotropic-sampling-foveated.md` - Anisotropic filtering

---

## Summary

GPU bilinear filtering for VLMs:
- **Free interpolation** (hardware, 1 cycle)
- **Smooth attention** (no discrete artifacts)
- **Continuous tokens** (fractional positions)
- **ARR-COC compatible** (smooth saccades)

**Implementation**: Set `filterMode = Linear`, use continuous coordinates
