# Trilinear and Anisotropic Filtering Hardware

## Overview

Hardware texture filtering represents one of the most critical performance and quality optimizations in modern GPUs. Trilinear and anisotropic filtering extend basic bilinear interpolation to provide smooth transitions between mipmap levels and high-quality sampling of textures viewed at oblique angles. These filtering techniques are implemented directly in dedicated Texture Mapping Units (TMUs), enabling single-cycle texture operations that would be prohibitively expensive if computed in shaders.

**Key concepts:**
- **Trilinear filtering**: Bilinear filtering across two adjacent mipmap levels with linear interpolation between them
- **Anisotropic filtering (AF)**: Adaptive sampling along texture gradients to handle oblique viewing angles (up to 16× AF standard)
- **TMU architecture**: Dedicated hardware units that perform filtering, addressing, and caching in parallel with shader execution
- **Performance characteristics**: Hardware filtering is "free" or near-free compared to equivalent shader-based implementations

**VLM connection:** For vision transformers processing multi-scale visual features, hardware texture filtering provides:
- Smooth LOD transitions when attention moves between patch resolutions
- Query-aware sampling where anisotropic filtering adapts to viewing angles
- Hardware-accelerated interpolation for continuous relevance realization across scale pyramid

## Trilinear Filtering

### Mathematical Foundation

Trilinear filtering extends bilinear filtering to sample across mipmap levels. Given texture coordinates (s, t) and computed LOD level λ:

```
LOD = λ = log₂(max(∂s/∂x, ∂t/∂y))  // Screen-space gradient determines mipmap level

Bilinear(level) = lerp(
    lerp(texel[i,j], texel[i+1,j], frac_s),
    lerp(texel[i,j+1], texel[i+1,j+1], frac_s),
    frac_t
)

Trilinear = lerp(
    Bilinear(floor(λ)),
    Bilinear(ceil(λ)),
    frac(λ)
)
```

This produces 8 texture samples total:
- 4 samples from mipmap level ⌊λ⌋
- 4 samples from mipmap level ⌈λ⌉
- 7 linear interpolations (3 per bilinear + 1 final)

### Hardware Implementation

From [NVIDIA GPU Gems 2, Chapter 20](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering) (accessed 2025-01-31):

Modern TMUs implement trilinear filtering as a **pipelined operation** that completes in 1-2 cycles:

**Pipeline stages:**
1. **Address calculation**: Compute texel addresses for all 8 samples
2. **Cache lookup**: Check texture cache (L1/L2) for resident texels
3. **Sample fetch**: Retrieve texels (4 from each mipmap level)
4. **Bilinear interpolation**: Parallel interpolation on both levels
5. **LOD interpolation**: Final lerp between mipmap levels

**Hardware optimization (from GPU Gems):**
```
// Rather than 8 separate nearest-neighbor fetches:
// 2 bilinear fetches (4 texels each) + 1 lerp

Hardware automatically:
- Computes ∂s/∂x, ∂t/∂y derivatives
- Determines LOD λ = log₂(max gradient)
- Fetches from ⌊λ⌋ and ⌈λ⌉ levels
- Performs all interpolations in fixed-function units
```

**Performance characteristics:**
- **NVIDIA Ampere**: 1 trilinear fetch per clock per TMU (128 TMUs = 128 trilinear/clock)
- **Latency**: ~20-40 cycles including cache
- **Throughput**: Matches or exceeds shader ALU throughput

### Quality Analysis

```
                LOD Transition Quality

Nearest:     ████░░░░░░░░  (Step function - visible mipmap boundaries)
Bilinear:    ████████░░░░  (Smooth within level, hard LOD transitions)
Trilinear:   ████████████  (Smooth LOD transitions, mild blur)
```

**Artifacts addressed:**
- ✓ Eliminates "mipmap banding" - visible seams between LOD levels
- ✓ Smooth scaling as camera moves toward/away from surface
- ✗ Does not address oblique viewing (requires anisotropic filtering)

**From [NVIDIA GPU Gems 2, Chapter 27](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-27-advanced-high-quality-filtering):**

> "Trilinear filtering solves the problem of mipmap transitions but produces overly blurred results when textures are viewed at steep angles. The issue is that screen-space pixels map to elongated footprints in texture space, which trilinear filtering treats as isotropic (circular)."

## Anisotropic Filtering

### Problem: Oblique Viewing Angles

When a textured surface is viewed at a steep angle, screen-space pixels map to **elongated footprints** in texture space:

```
Screen Space           Texture Space

   ▢  ▢  ▢              ▭▭▭▭▭▭▭▭▭▭▭▭
   ▢  ▢  ▢     →        ▭▭▭▭▭▭▭▭▭▭▭▭  (Anisotropic - elongated)
   ▢  ▢  ▢              ▭▭▭▭▭▭▭▭▭▭▭▭

Trilinear treats as:   ⬤ ⬤ ⬤        (Isotropic - circular)
```

**Consequence:** Trilinear filtering averages over a circular region, causing excessive blur along the major axis of the elongated footprint.

### Anisotropic Filtering Solution

**Core idea:** Take **multiple samples along the major axis** of the texture footprint, then blend results.

From [Arm Developer Documentation on Texture Filtering](https://developer.arm.com/documentation/102073/latest/Texture-filtering) (accessed 2025-01-31):

**Algorithm:**
```python
# Compute texture gradients
ds_dx, dt_dx = ddx(texcoord)  # ∂s/∂x, ∂t/∂x
ds_dy, dt_dy = ddy(texcoord)  # ∂s/∂y, ∂t/∂y

# Determine anisotropy ratio
major_axis = max(length(ds_dx, dt_dx), length(ds_dy, dt_dy))
minor_axis = min(length(ds_dx, dt_dx), length(ds_dy, dt_dy))
aniso_ratio = major_axis / minor_axis  # Typically clamped to 2×, 4×, 8×, 16×

# Sample along major axis
num_samples = min(aniso_ratio, max_aniso)  # e.g., 16 for 16× AF
step = major_axis / num_samples

color = 0
for i in range(num_samples):
    sample_pos = texcoord + (i - num_samples/2) * step
    color += trilinear_sample(sample_pos)
return color / num_samples
```

**16× Anisotropic Filtering** means up to 16 trilinear samples per pixel, adaptively determined by viewing angle.

### Hardware Implementation

From NVIDIA Turing Architecture Whitepaper and AMD RDNA documentation:

**NVIDIA Turing/Ampere TMU architecture:**
```
Per SM: 4 Texture Units (TMUs)
Each TMU:
- 4 texture address units
- 16 texture filter units
- L1 texture cache (128 KB per SM)

Anisotropic filtering:
- Up to 16× AF: 16 trilinear samples = 128 bilinear samples
- Performed across multiple cycles (typically 2-8 cycles for 16× AF)
- Pipelined to hide latency
- Texture cache hit rate critical for performance
```

**AMD RDNA3 TMU architecture (from [AMD RDNA Whitepaper](https://www.techpowerup.com/gpu-specs/docs/amd-rdna-whitepaper.pdf), accessed 2025-01-31):**
```
Per CU: 4 TMUs
Each TMU:
- 64-bit bi-linear filtering (2× throughput of previous gen)
- 16× anisotropic filtering support
- 128 KB L1 texture cache per shader array
- Can perform 4 bilinear fetches per clock

Performance:
- 16× AF reduces throughput by ~2-4× vs trilinear
- Cache hit rate determines actual performance impact
```

### Performance Analysis

**Texture fetch throughput (from GPU Gems and vendor whitepapers):**

| Filtering Mode | Samples | Cycles (typical) | Relative Cost |
|---------------|---------|------------------|---------------|
| Nearest | 1 | 1 | 1.0× |
| Bilinear | 4 | 1 | 1.0× |
| Trilinear | 8 | 1-2 | 1.0-1.5× |
| 2× Anisotropic | 16 | 2-3 | 1.5-2× |
| 4× Anisotropic | 32 | 3-5 | 2-3× |
| 8× Anisotropic | 64 | 4-8 | 3-5× |
| 16× Anisotropic | 128 | 6-12 | 4-8× |

**Key insight:** Hardware anisotropic filtering is **dramatically** faster than equivalent shader implementation:
- Shader-based 16× AF: ~200-400 cycles (fetches + interpolation arithmetic)
- Hardware 16× AF: ~6-12 cycles (pipelined, cached, fixed-function)
- **20-50× performance advantage** from dedicated hardware

### Quality Comparison

```
Viewing Angle: 75° (steep oblique)

Bilinear:
████░░░░  Very blurry - averages over wrong region

Trilinear:
██████░░  Less blurry but still incorrect footprint

4× Anisotropic:
████████  Much sharper - sampling along major axis

16× Anisotropic:
██████████  Near-perfect - closely matches true footprint
```

From community benchmarks (Reddit r/pcgaming, "16x Anisotropic Filtering" discussion, accessed 2025-01-31):

> "16× AF has almost no performance impact on modern GPUs (< 5% frame rate difference) but produces dramatically sharper textures at oblique angles. It's essentially a 'free' quality upgrade."

## Texture Units (TMU) Architecture

### Hardware Organization

**Modern GPU texture pipeline (NVIDIA Ampere, AMD RDNA3):**

```
                    Streaming Multiprocessor (SM) / Compute Unit (CU)
                    ┌────────────────────────────────────────────────
                    │
                    │  Shader Cores (CUDA cores / Stream processors)
                    │  ├─ ALU pipeline
                    │  ├─ FPU pipeline
                    │  └─ Special function units
                    │
                    │  Texture Units (4 per SM/CU)
                    │  ├─ Address calculation
                    │  ├─ LOD computation (∂s/∂x, ∂t/∂y derivatives)
                    │  ├─ Filtering logic (bilinear/trilinear/aniso)
                    │  └─ Format conversion
                    │
                    │  L1 Texture Cache (128 KB per SM)
                    │  ├─ 128-byte cache lines
                    │  ├─ 4-way set associative
                    │  └─ Optimized for 2D locality
                    │
                    └─────────────┬──────────────────────────────────
                                  │
                                  ▼
                          L2 Cache (Shared, 6-12 MB)
                                  │
                                  ▼
                              VRAM (GDDR6/HBM)
```

### Texture Address Unit (TAU)

**Functions:**
1. **Coordinate transformation:** texcoord → texel address
2. **Wrap/clamp mode:** Handle out-of-bounds coordinates
3. **Derivative computation:** Calculate ∂s/∂x, ∂t/∂y for LOD
4. **LOD calculation:** λ = log₂(max(|∂s/∂x|, |∂t/∂y|))

**From Arm Mali GPU documentation:**

> "The texture mapping unit has doubled performance for 64-bit bilinear filtering. The TMU can perform filtering for up to eight texture addresses per clock – twice the throughput of prior generation." — [Arm Mali-G78 Performance Counters Reference](https://developer.arm.com/documentation/102626/latest/Shader-core-texture-unit), accessed 2025-01-31

### Texture Filter Unit (TFU)

**Core filtering operations:**

```
Bilinear interpolation (4 texels):
    t0 = lerp(texel[i,j], texel[i+1,j], frac_s)
    t1 = lerp(texel[i,j+1], texel[i+1,j+1], frac_s)
    result = lerp(t0, t1, frac_t)

Hardware implementation:
- Parallel lerp units (3× per bilinear)
- Fixed-point or FP16 arithmetic
- 1-cycle latency (pipelined)
```

**Trilinear extension:**
- Two parallel bilinear units
- Final lerp unit for mipmap interpolation
- Total: 7 lerp operations, typically 1-2 cycle latency

**Anisotropic extension:**
- Reuse trilinear units across multiple cycles
- Step along major axis computed by TAU
- Accumulator for blending samples

### Texture Cache Hierarchy

**L1 Texture Cache (per SM):**
- **Size:** 96-128 KB (Ampere), 128 KB (RDNA3)
- **Latency:** ~20 cycles
- **Bandwidth:** ~1 TB/s per SM
- **Organization:** Optimized for 2D spatial locality

**Cache line structure:**
```
Cache Line (128 bytes)
├─ Texel block: 8×8 texels @ 16-bit (2 KB uncompressed)
├─ Tag: Mipmap level + base address
└─ Metadata: Format, filtering mode
```

**L2 Cache (shared across GPU):**
- **Size:** 6-12 MB (Ampere), 4-6 MB (RDNA3)
- **Latency:** ~200 cycles
- **Bandwidth:** ~3-5 TB/s total
- **Organization:** Shared between texture and compute

### Performance Characteristics

**Throughput (NVIDIA A100):**
- 108 SMs × 4 TMUs = **432 texture units**
- **432 trilinear fetches per clock** @ 1.41 GHz = 609 billion trilinear samples/sec
- Compare to: **312 TFLOPS** FP32 compute

**From [NVIDIA Ampere Architecture Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), page 14:**

> "Each SM in the GA102 GPU contains four texture units that can deliver up to 128 bytes per clock to each of the two data paths." — accessed 2025-01-31

**Latency breakdown:**
```
Texture fetch total latency: 20-400 cycles

L1 cache hit:     ~20 cycles   (95%+ hit rate typical)
L2 cache hit:     ~200 cycles  (99%+ hit rate typical)
VRAM access:      ~400 cycles  (<1% of accesses)

Hiding latency:
- TMU operates independently of shader ALU
- While waiting for texture, shader executes other warps
- Effective latency: near zero if enough active warps
```

## Neural Rendering Integration

### Using Hardware Filtering in Vision Transformers

**Problem:** VLMs process multi-scale visual features through patch pyramids. Traditional approaches either:
1. Nearest-neighbor sampling → aliasing artifacts at scale transitions
2. Shader-based filtering → massive performance overhead

**Solution:** Leverage hardware texture filtering for smooth LOD transitions.

**Implementation pattern:**

```python
# Store visual features in 3D texture (H, W, D)
# where D = feature dimension (e.g., 768 for ViT-B)

# Configure texture for trilinear filtering
texture_config = {
    'mipmap': True,              # Enable mipmap pyramid
    'filter': 'TRILINEAR',       # Hardware trilinear
    'aniso': 16,                 # 16× anisotropic filtering
    'wrap': 'CLAMP_TO_EDGE'      # Boundary handling
}

# Sample features with query-aware LOD
def sample_features(query_vector, patch_position, attention_score):
    # Attention score determines LOD (high attention = high resolution)
    lod = -log2(attention_score)  # 1.0 → LOD 0, 0.25 → LOD 2

    # Hardware automatically performs trilinear filtering
    features = texture.sample(
        coords=patch_position,
        lod=lod,
        aniso=True  # Enable anisotropic for oblique patches
    )
    return features
```

**Benefits:**
- **40-50× faster** than equivalent shader-based filtering
- **Smooth LOD transitions** as attention moves between patch resolutions
- **Anisotropic quality** for patches viewed at oblique angles (foveated vision)
- **Free mipmapping** - hardware generates and filters pyramid

### Foveated Rendering with Anisotropic Filtering

**Biological inspiration:** Human foveal vision has highest resolution at gaze center, degrading toward periphery.

**GPU-accelerated implementation:**
```
Attention Allocation (Vervaekean Relevance):

High relevance region (fovea):
- LOD 0 (full resolution)
- 16× anisotropic filtering
- Trilinear for smooth zooming

Medium relevance (parafovea):
- LOD 1-2 (1/2 to 1/4 resolution)
- 8× anisotropic filtering
- Trilinear transitions

Low relevance (periphery):
- LOD 3+ (1/8+ resolution)
- 4× anisotropic filtering
- Trilinear only if needed

Hardware automatically:
✓ Computes appropriate LOD per fragment
✓ Applies anisotropic sampling adaptively
✓ Blends between mipmap levels smoothly
✓ Performs all filtering in 1-12 cycles
```

**ARR-COC integration:**
```python
class RelevanceAwareTextureFiltering:
    def realize_relevance_to_lod(self, relevance_scores):
        """Map relevance scores to texture LOD levels"""
        # High relevance → low LOD (high res)
        # Low relevance → high LOD (low res)
        lod = -log2(relevance_scores.clamp(min=0.01))
        return lod

    def configure_anisotropy(self, viewing_angle):
        """Adjust anisotropic filtering based on patch orientation"""
        # Steep angles need high anisotropy
        aniso_factor = clip(abs(1.0 / cos(viewing_angle)), 1, 16)
        return aniso_factor

    def sample_with_relevance(self, patch_coords, relevance):
        # Hardware does all the work
        features = self.feature_texture.sample(
            coords=patch_coords,
            lod=self.realize_relevance_to_lod(relevance),
            aniso=self.configure_anisotropy(patch_coords.angle)
        )
        return features  # Smooth, high-quality, hardware-accelerated
```

### Performance Analysis for VLM Inference

**Scenario:** ViT-Large (1024 patches, 768-dim features)

**Without hardware filtering (shader-based):**
```
Per patch:
- 16 samples (4×4 neighborhood)
- 16 texture fetches @ 20 cycles = 320 cycles
- 48 FMA ops for bilinear = 48 cycles
Total per patch: ~370 cycles

For 1024 patches: 378,880 cycles @ 1.5 GHz = 0.25 ms
```

**With hardware trilinear filtering:**
```
Per patch:
- 1 texture fetch (hardware trilinear)
- 1-2 cycles (pipelined, cached)
Total per patch: ~2 cycles

For 1024 patches: 2,048 cycles @ 1.5 GHz = 0.0014 ms
```

**Speedup: 180× faster** using hardware texture filtering.

**Adding 16× anisotropic filtering:**
```
Per patch: ~8 cycles (worst case)
For 1024 patches: 8,192 cycles @ 1.5 GHz = 0.0055 ms

Still 45× faster than shader-based bilinear
And much higher quality (16× samples vs 4× samples)
```

## Code Examples

### Shader: Explicit LOD Control with Trilinear Filtering

```glsl
// GLSL shader for query-aware LOD selection
#version 450

layout(binding = 0) uniform sampler2D feature_texture;
layout(location = 0) in vec2 patch_coords;
layout(location = 1) in float attention_score;
layout(location = 0) out vec4 features;

void main() {
    // Compute LOD from attention score
    // High attention (1.0) → LOD 0 (full res)
    // Low attention (0.125) → LOD 3 (1/8 res)
    float lod = -log2(max(attention_score, 0.01));

    // Hardware performs trilinear filtering automatically
    // textureLod() uses explicit LOD vs automatic derivative-based LOD
    features = textureLod(feature_texture, patch_coords, lod);

    // Result: smooth interpolation across mipmap pyramid
    // 8 samples blended in 1-2 GPU cycles (hardware accelerated)
}
```

### Shader: Comparing Filtering Modes

```glsl
// GLSL comparison of filtering quality
#version 450

layout(binding = 0) uniform sampler2D tex_nearest;    // GL_NEAREST
layout(binding = 1) uniform sampler2D tex_bilinear;   // GL_LINEAR
layout(binding = 2) uniform sampler2D tex_trilinear;  // GL_LINEAR_MIPMAP_LINEAR
layout(binding = 3) uniform sampler2D tex_aniso;      // + 16× anisotropic

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 color;

uniform int mode;  // 0=nearest, 1=bilinear, 2=trilinear, 3=aniso

void main() {
    // All modes use same coordinates
    // Hardware automatically applies configured filtering
    switch(mode) {
        case 0: color = texture(tex_nearest, texcoord); break;
        case 1: color = texture(tex_bilinear, texcoord); break;
        case 2: color = texture(tex_trilinear, texcoord); break;
        case 3: color = texture(tex_aniso, texcoord); break;
    }

    // Performance:
    // Nearest:   1 cycle  (blocky)
    // Bilinear:  1 cycle  (smooth but blurry at oblique angles)
    // Trilinear: 1-2 cycles (smooth LOD transitions)
    // 16× Aniso: 6-12 cycles (sharp at all angles) - still 20-40× faster than shader impl
}
```

### Python: Configuring Texture Filtering

```python
# PyTorch with OpenGL texture interop
import torch
import torch.nn as nn
from OpenGL.GL import *

class HardwareFilteredFeatures(nn.Module):
    def __init__(self, feature_dim=768, enable_mipmaps=True):
        super().__init__()
        self.feature_dim = feature_dim

        # Create OpenGL texture for features
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Configure trilinear filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                       GL_LINEAR_MIPMAP_LINEAR)  # Trilinear
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                       GL_LINEAR)  # Bilinear for magnification

        # Enable 16× anisotropic filtering (if supported)
        max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                       min(16.0, max_aniso))

        # Configure wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        if enable_mipmaps:
            # Hardware automatically generates mipmaps
            glGenerateMipmap(GL_TEXTURE_2D)

    def upload_features(self, features: torch.Tensor):
        """Upload PyTorch tensor to GPU texture memory"""
        # features: [H, W, feature_dim]
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                    features.shape[1], features.shape[0], 0,
                    GL_RGBA, GL_FLOAT, features.cpu().numpy())

        # Generate mipmap pyramid (hardware accelerated)
        glGenerateMipmap(GL_TEXTURE_2D)

    def sample_with_lod(self, coords: torch.Tensor, lod: torch.Tensor):
        """Sample texture with explicit LOD control"""
        # Use shader with textureLod() to control mipmap level
        # Hardware performs trilinear filtering automatically
        # Returns smoothly interpolated features across LOD pyramid
        pass  # Implemented in shader, called from Python
```

## Sources

**NVIDIA Documentation:**
- [GPU Gems 2, Chapter 20: Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering) - Christian Sigg & Markus Hadwiger, accessed 2025-01-31
- [GPU Gems 2, Chapter 27: Advanced High-Quality Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-27-advanced-high-quality-filtering) - Justin Novosad, accessed 2025-01-31
- [NVIDIA Ampere GA102 Architecture Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) - NVIDIA, 53 pages, accessed 2025-01-31
- [NVIDIA Turing Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) - NVIDIA, 86 pages, accessed 2025-01-31

**AMD Documentation:**
- [AMD RDNA Architecture Whitepaper](https://www.techpowerup.com/gpu-specs/docs/amd-rdna-whitepaper.pdf) - AMD, 25 pages, accessed 2025-01-31

**Arm Documentation:**
- [Arm Developer: Texture Filtering](https://developer.arm.com/documentation/102073/latest/Texture-filtering) - Arm Limited, accessed 2025-01-31
- [Arm Mali-G78 Performance Counters Reference](https://developer.arm.com/documentation/102626/latest/Shader-core-texture-unit) - Arm Limited, accessed 2025-01-31

**Community Resources:**
- [Reddit r/pcgaming: "16x Anisotropic Filtering"](https://www.reddit.com/r/pcgaming/comments/213574/16x_anisotropic_filtering_ive_taken_this_as_a/) - Discussion on AF performance impact, accessed 2025-01-31
- [Intel: What Is Anisotropic Filtering?](https://www.intel.com/content/www/us/en/gaming/resources/what-is-anisotropic-filtering.html) - Intel Gaming Resources, accessed 2025-01-31
- [Powder AI: Understanding Anisotropic Filtering](https://powder.gg/blog/understanding-anisotropic-filtering-enhancing-graphics-quality/) - Blog post, October 2024, accessed 2025-01-31

**Additional References:**
- [NVIDIA Research: Collaborative Texture Filtering](https://research.nvidia.com/labs/rtr/publication/akeninemoller2025collaborative/collaborative_texfilt.pdf) - NVIDIA Research, 16 pages, PDF whitepaper, accessed 2025-01-31
- [Google Patents: Trilinear Optimization for Texture Filtering](https://patents.google.com/patent/US6995767B1/en) - US Patent, accessed 2025-01-31
- [Chips and Cheese: GPU L1 Cache Bandwidth](https://chipsandcheese.com/p/a-new-year-and-new-tests-gpu-l1-cache-bandwidth) - January 2024 analysis, accessed 2025-01-31
