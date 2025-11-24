# Texture Sampling and Filtering Techniques

## Overview

Texture sampling and filtering are fundamental techniques in 3D graphics that determine how texture maps appear when rendered at various distances, angles, and resolutions. These processes bridge the gap between discrete texture data (texels) and continuous screen-space pixels, directly impacting visual quality and performance.

**Core Challenge:** When mapping a 2D texture grid onto a 3D surface viewed from arbitrary angles and distances, texel-to-pixel correspondence rarely aligns perfectly. Without filtering, textures exhibit either blocky pixelation or excessive blur. Filtering methods approximate pixel colors by intelligently sampling and interpolating texel data.

## Texture Coordinate Systems

### UV Mapping Fundamentals

Texture coordinates (UV) define how 2D textures map onto 3D geometry:

- **U axis:** Horizontal texture dimension (typically 0.0 to 1.0)
- **V axis:** Vertical texture dimension (typically 0.0 to 1.0)
- **Normalized coordinates:** UV values outside [0,1] trigger wrapping/clamping behavior

From [WebGPU Unleashed Tutorial](https://shi-yan.github.io/webgpuunleashed/Basics/mipmapping_and_anisotropic_filtering.html) (accessed 2025-10-31):
> "We derive the texture coordinates and positions based on the vertex index. Our goal is to generate a rectangle that covers the entire canvas or screen, effectively rendering the complete image across this area."

### Wrapping Modes

**Repeat (Wrap):** UV coordinates > 1.0 wrap around (tiling)
```
UV = 2.3 → wraps to 0.3
```

**Clamp to Edge:** Coordinates beyond [0,1] clamp to boundary
```
UV = 1.5 → clamps to 1.0
```

**Mirror Repeat:** Alternates between normal and mirrored texture
```
UV = 1.3 → mirrors to 0.7
```

### Fractional Pixel Offsets

From [Bart Wronski's analysis](https://bartwronski.com/2020/04/14/bilinear-texture-filtering-artifacts-alternatives-and-frequency-domain-analysis/) (accessed 2025-10-31):

> "With a subpixel offset of 0.0, linear weights will be [1.0, 0.0], and with a subpixel offset of 0.5, they will be [0.5, 0.5]. Those two extremes correspond to a very different image filters!"

**Key insight:** Filter weights vary dramatically based on fractional UV positions, creating position-dependent filtering effects. At offset 0.0, filtering is identity (no blur); at 0.5, maximum blur occurs.

## Mipmapping: Automatic LOD Selection

### Mipmap Structure

Mipmaps are precomputed texture pyramids storing progressively lower-resolution versions:

```
Level 0: 2048×2048 (original)
Level 1: 1024×1024
Level 2: 512×512
...
Level N: 1×1 (single averaged pixel)
```

**Level calculation:**
```
mipLevelCount = ceil(log₂(max(width, height)))
```

From [WebGPU Unleashed](https://shi-yan.github.io/webgpuunleashed/Basics/mipmapping_and_anisotropic_filtering.html):
> "Each level typically contains the same image at different resolutions, with level 0 being the original resolution. At level 1, we quarter the original image, and so on until the entire image is downsized to a single pixel representing the average color of the original texture map."

### Automatic Mipmap Generation

**GPU-accelerated approach:**
1. Create texture with multiple mip levels
2. Render each level by sampling from previous level
3. Use linear interpolation to downsample
4. Store results in progressively smaller textures

**WebGPU example:**
```javascript
const textureDescriptor = {
    size: { width: imgBitmap.width, height: imgBitmap.height },
    format: 'rgba8unorm',
    mipLevelCount: Math.ceil(Math.log2(Math.max(imgBitmap.width, imgBitmap.height))),
    usage: GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_DST |
           GPUTextureUsage.RENDER_ATTACHMENT
};
```

### Mipmap LOD Selection

The GPU automatically selects appropriate mipmap levels based on:
- **Screen-space derivatives:** How quickly texture coordinates change per pixel
- **Distance from camera:** Farther objects use lower-resolution mips
- **Viewing angle:** Oblique angles may require different sampling strategies

**Trilinear interpolation process:**
1. Calculate ideal mipmap level from screen-space gradients
2. Select two nearest mipmap levels (floor and ceil)
3. Sample both levels with bilinear filtering
4. Linearly interpolate between the two results

## Point Sampling (Nearest-Neighbor)

**Method:** Select single nearest texel, no interpolation.

**Characteristics:**
- Fastest possible sampling (single texture fetch)
- Sharp edges, visible pixel boundaries
- Produces "blocky" or "pixelated" appearance
- Useful for pixel-art rendering or exact texel lookup

**Use cases:**
- Retro/pixel-art aesthetic
- Lookup tables (LUTs) requiring exact values
- Debug visualization of texture data

## Bilinear Filtering

### Algorithm

From [Bart Wronski's technical analysis](https://bartwronski.com/2020/04/14/bilinear-texture-filtering-artifacts-alternatives-and-frequency-domain-analysis/):
> "Bilinear filtering is a texture (or more generally, signal) interpolation filter that is separable – it is a linear filter applied on the x axis of the image (along the width), and then a second filter applied along the y axis (along the height)."

**1D Linear Interpolation (Tent Filter):**
```
color = texel_N × (1 - t) + texel_N+1 × t
```
where `t` is fractional position between samples.

**2D Bilinear (4 samples):**
```
Sample texels: [x0,y0], [x1,y0], [x0,y1], [x1,y1]
Weights: (1-tx)×(1-ty), tx×(1-ty), (1-tx)×ty, tx×ty
```

### Hardware Acceleration

**Critical performance advantage:** Modern GPUs perform bilinear filtering in a single instruction with effectively **zero additional cost** over point sampling.

From [GarageFarm analysis](https://garagefarm.net/blog/texture-filtering-techniques-for-sharper-3d-renders) (accessed 2025-10-31):
> "All modern GPUs can do bilinear filtering of an input texture in a single hardware instruction. This involves both memory access (together with potentially uncompressing block compressed textures into local cache), as well as the interpolation arithmetic."

### Bilinear Artifacts

**Pyramid artifacts:** Each texel creates a "pyramid" of influence in screen space when magnified.

From Bart Wronski:
> "Separable filtering applies first a 1D triangle filter in one direction, then in another direction, multiplying the weights. Those two 1D ramps multiplied together result in a pyramid – this means that every input pixel will 'splat' a small pyramid."

**Mach bands:** Human visual system perceives discontinuities in derivatives (C0 continuous but not C1), creating visible brightness transitions.

**Variance loss:** Position-dependent blurring causes contrast reduction varying with fractional pixel offsets:
- Offset 0.0: No filtering (sharp)
- Offset 0.5: Maximum blur (50% variance reduction)
- Creates "pulsing" artifacts during motion

**Frequency response inconsistency:** Different subpixel positions produce dramatically different low-pass filtering characteristics, leading to visible texture shimmer during camera movement.

## Trilinear Filtering

### Enhanced Mipmap Interpolation

Trilinear filtering extends bilinear by interpolating between mipmap levels, eliminating harsh transitions.

**Algorithm:**
1. Calculate fractional mipmap level `L` from screen-space derivatives
2. Select levels `floor(L)` and `ceil(L)`
3. Bilinear sample from both levels → `color_A`, `color_B`
4. Linear interpolate: `final = color_A × (1-f) + color_B × f`
   where `f = fract(L)`

From [GarageFarm](https://garagefarm.net/blog/texture-filtering-techniques-for-sharper-3d-renders):
> "Trilinear Filtering interpolates across mipmap levels for better consistency. Trilinear filtering softens these transitions by performing bilinear filtering on two different mipmap levels and interpolating the results."

### Benefits and Costs

**Advantages:**
- Eliminates "mipmap popping" (sudden LOD transitions)
- Smooth gradients across distance changes
- Better temporal stability

**Disadvantages:**
- 2× texture bandwidth vs bilinear (8 samples vs 4)
- Additional ALU for level interpolation
- Can appear overly blurry compared to bilinear

## Anisotropic Filtering

### The Oblique Angle Problem

Standard isotropic filtering (bilinear/trilinear) assumes uniform sampling in all directions. This fails when viewing surfaces at steep angles, where screen-space pixel footprints become elongated rather than square.

**Example:** Long corridor floor stretching to horizon
- Pixels near horizon cover many texels horizontally
- But few texels vertically
- Isotropic filter blurs both directions equally → loss of detail

### Anisotropic Solution

From [Wikipedia](https://en.wikipedia.org/wiki/Anisotropic_filtering) (referenced in research):
> "In 3D computer graphics, anisotropic filtering (AF) is a technique that improves the appearance of textures, especially on surfaces viewed at sharp angles."

**Method:** Take multiple samples along the direction of greatest texture compression.

From [GarageFarm](https://garagefarm.net/blog/texture-filtering-techniques-for-sharper-3d-renders):
> "Anisotropic Filtering selectively samples from mipmaps based on screen space distortion... Anisotropic filtering preserves texture clarity at sharp viewing angles, keeping details crisp where other methods blur."

**Sample count:** Typically 2x, 4x, 8x, 16x AF
- 16x AF = up to 16 samples per pixel along anisotropic direction
- Samples distributed intelligently based on texture footprint shape

### Performance Characteristics

**Quality vs. Cost tradeoff:**
- **2x-4x AF:** Modest quality improvement, ~10-20% performance cost
- **8x-16x AF:** Significant quality gain, ~20-30% cost
- Modern GPUs optimize AF heavily; often less expensive than theoretical cost

**When to use:**
- Always enable for ground/floor textures
- Essential for architectural visualization
- Critical for racing games (road textures)
- Recommended for any oblique viewing angles

From research findings:
> "Most game engines today support anisotropic filtering up to 16x, and it's become a standard for high-quality renders in both Direct3D and OpenGL pipelines."

## LOD Calculation for Texture Sampling

### Screen-Space Derivative Method

Modern GPUs calculate texture LOD using partial derivatives:

**GLSL/WGSL automatic derivatives:**
```glsl
// Automatic LOD calculation
vec4 color = texture(sampler, uv); // GPU computes derivatives internally

// Manual LOD specification
vec4 color = textureLod(sampler, uv, lodLevel);

// Derivative functions
float dFdx_uv = dFdx(uv); // Rate of UV change in screen-space X
float dFdy_uv = dFdy(uv); // Rate of UV change in screen-space Y
```

**LOD calculation principle:**
1. Compute how fast UV coordinates change per pixel
2. Larger gradients → need lower resolution mip
3. Select mip level where texel size ≈ pixel size

### Manual LOD Control

**Use cases for explicit LOD:**
- Temporal stability (prevent shimmering)
- Artistic control over blur/sharpness
- Virtual texturing systems
- Custom filtering schemes

**Implementation:**
```glsl
float lod = 0.5 * log2(max(dot(dFdx(uv), dFdx(uv)),
                             dot(dFdy(uv), dFdy(uv))));
vec4 color = textureLod(sampler, uv, lod + lodBias);
```

## Texture Atlas Sampling Strategies

### Atlas Organization

Texture atlases pack multiple textures into single large texture for:
- Reduced draw calls
- Better GPU cache coherency
- Simplified material management

**Common layouts:**
- Grid-based: Fixed-size tiles in regular grid
- Packed: Arbitrary rectangles using bin-packing
- Virtual: Indirection table to sparse physical pages

### Border and Bleeding Issues

**Problem:** Bilinear/trilinear filtering samples neighboring texels, causing "bleeding" between atlas regions.

**Solutions:**

**1. Padding/Border pixels:**
```
Add 1-2 pixel border around each atlas region
Fill with edge-clamped or wrapped texel data
```

**2. UV coordinate clamping:**
```glsl
// Clamp UVs to safe interior region
vec2 uv_min = atlasUV.xy + halfTexel;
vec2 uv_max = atlasUV.zw - halfTexel;
vec2 safe_uv = clamp(uv, uv_min, uv_max);
```

**3. Mipmap considerations:**
- Generate mipmaps BEFORE atlas packing, or
- Ensure border padding grows with mip levels, or
- Use per-region mipmap chains (virtual texturing)

### Atlas-Aware Filtering

For large atlases with mipmaps, consider:

**Derivative clamping:**
```glsl
// Prevent sampling beyond region boundaries at high mip levels
vec2 duvdx = dFdx(uv);
vec2 duvdy = dFdy(uv);
float maxDerivative = regionSize * 0.5;
duvdx = clamp(duvdx, -maxDerivative, maxDerivative);
duvdy = clamp(duvdy, -maxDerivative, maxDerivative);
vec4 color = textureGrad(sampler, uv, duvdx, duvdy);
```

## ARR-COC Multi-Channel Sampling Patterns

### Relevance-Aware Texture Sampling

ARR-COC's adaptive relevance realization framework extends to texture sampling through multi-channel patterns that align with cognitive processing:

**Propositional Channel (Information Content):**
- High-frequency detail textures (normals, roughness)
- Sharp sampling (point/bilinear) for semantic features
- Preserves edge information critical for shape recognition

**Perspectival Channel (Salience Landscapes):**
- Color/albedo textures
- Anisotropic filtering for oblique surfaces
- Adaptive LOD based on query-content coupling

**Participatory Channel (Query-Content Coupling):**
- Context-aware mipmap selection
- Attention-modulated filtering sharpness
- Dynamic LOD bias based on relevance scores

### 32×32 Grid UV Strategy

For ARR-COC's visual token compression (64-400 tokens per patch):

**Grid-aligned sampling:**
```python
# 32×32 patch extraction with relevance-aware filtering
patch_uvs = compute_patch_grid(image_size=(224, 224), grid=(32, 32))

for patch in patches:
    relevance_score = compute_relevance(patch, query)

    if relevance_score > HIGH_THRESHOLD:
        # High relevance: sharp sampling, preserve detail
        filter_mode = BILINEAR
        lod_bias = -0.5  # Sharper
        token_budget = 400
    elif relevance_score > MED_THRESHOLD:
        # Medium relevance: balanced filtering
        filter_mode = TRILINEAR
        lod_bias = 0.0
        token_budget = 196
    else:
        # Low relevance: aggressive blur, compress
        filter_mode = TRILINEAR
        lod_bias = +1.0  # Blurrier
        token_budget = 64

    patch_tokens = sample_patch(patch_uvs, filter_mode, lod_bias)
    compressed_tokens = compress(patch_tokens, token_budget)
```

**Benefits:**
- Computational efficiency: Filter complexity matches relevance
- Perceptual optimization: Detail where it matters
- Adaptive compression: LOD serves dual purpose (rendering + token allocation)

### Multi-Scale Texture Hierarchies

ARR-COC can leverage texture mipmap pyramids as natural LOD structure:

**Alignment with cognitive framework:**
- **Knowing:** Measure information content per mip level
- **Balancing:** Navigate compress↔particularize tension via LOD
- **Attending:** Map relevance scores to mipmap levels
- **Realizing:** Execute LOD-aware sampling and token allocation

**Implementation sketch:**
```python
def arr_coc_texture_sampling(image, query, grid_size=32):
    # Generate mipmap pyramid
    mipmaps = generate_mipmaps(image)

    # Compute relevance per grid patch
    relevance_map = compute_patch_relevance(image, query, grid_size)

    # Map relevance to LOD (higher relevance → higher res mip)
    lod_map = relevance_to_lod(relevance_map, num_mips=len(mipmaps))

    # Sample from appropriate mip levels
    tokens = []
    for patch_id, lod in enumerate(lod_map):
        mip_level = int(lod)
        fractional = lod - mip_level

        # Bilinear between mip levels (trilinear equivalent)
        sample_A = sample_mip(mipmaps[mip_level], patch_id)
        sample_B = sample_mip(mipmaps[mip_level + 1], patch_id)
        patch_data = lerp(sample_A, sample_B, fractional)

        # Allocate tokens based on relevance
        token_count = relevance_to_tokens(relevance_map[patch_id])
        patch_tokens = compress_patch(patch_data, token_count)
        tokens.extend(patch_tokens)

    return tokens
```

## Advanced Filtering Techniques

### Bicubic Filtering (B-Spline)

From [Bart Wronski](https://bartwronski.com/2020/04/14/bilinear-texture-filtering-artifacts-alternatives-and-frequency-domain-analysis/):
> "The most common bicubic filter used in graphics is one that is also called B-Spline bicubic... It looks way smoother and seems to reconstruct shapes much more faithfully than bilinear!"

**Characteristics:**
- Samples 16 texels (4×4 neighborhood)
- C1 continuous (smooth derivatives)
- Reduces Mach banding artifacts
- Can be optimized to 4 bilinear samples via clever UV offsetting

**Optimization (GPU Gems 2 technique):**
```glsl
// 4-tap bicubic approximation using bilinear hardware
vec2 uv_cubic = compute_bicubic_uvs(uv, texelSize);
vec4 c0 = texture(sampler, uv_cubic.xy);
vec4 c1 = texture(sampler, uv_cubic.zw);
// ... weight and combine samples
```

### Biquadratic Filtering

Recent alternative offering sharpness between bilinear and bicubic:
- Samples 9 texels (3×3 neighborhood)
- Reconstructs quadratic function
- Less blur than B-spline bicubic
- Lower cost than full 16-tap bicubic

**Use case:** When bilinear is too soft but bicubic too expensive.

### Windowed Sinc (Lanczos)

High-quality resampling for offline/slow rendering:
- Preserves high frequencies better than cubic
- Minimal aliasing when downsampling
- Risk of ringing artifacts (negative filter weights)
- Too expensive for real-time use

## Performance Optimization Strategies

### Filter Selection Guidelines

**Real-time rendering:**
```
Point sampling:     ~1.0x cost (baseline)
Bilinear:           ~1.0x cost (hardware accelerated)
Trilinear:          ~1.2-1.5x cost (2× bandwidth)
Anisotropic 4x:     ~1.3-1.8x cost
Anisotropic 16x:    ~1.5-2.5x cost
Bicubic (4-tap):    ~2.0-3.0x cost
```

**Adaptive filtering:**
- Foreground objects: Anisotropic 8x-16x
- Background/distant: Trilinear or bilinear
- UI elements: Bilinear or sharp bicubic
- Effects/particles: Bilinear (motion hides artifacts)

### Bandwidth Optimization

**Texture compression:** Use BC/ASTC formats
- Reduces bandwidth 4:1 to 8:1
- Hardware-decompressed during sampling
- Minimal quality loss for most content

**Mipmap streaming:** Load high-res mips on-demand
- Keep low mips resident
- Stream in detail as camera approaches
- Essential for large open-world games

**Atlas packing:** Reduce texture binds
- Group related textures
- Improve cache coherency
- Enable batch rendering

## Common Artifacts and Solutions

### Texture Shimmering

**Cause:** Temporal aliasing from under-sampled high-frequency content

**Solutions:**
- Increase anisotropic filtering level
- Apply temporal anti-aliasing (TAA)
- Add subtle LOD bias (+0.25 to +0.5)
- Ensure proper mipmap generation

### Blurry Textures

**Causes:**
- Overly aggressive LOD bias
- Trilinear filtering on near surfaces
- Mipmap generation with excessive blur

**Solutions:**
- Use negative LOD bias (-0.25 to -0.5) cautiously
- Sharpen mipmaps during generation
- Switch to bilinear for nearby objects
- Apply post-process sharpening (carefully)

### Moiré Patterns

**Cause:** Aliasing from repeating high-frequency patterns (checkerboards, grids)

**Solutions:**
- Ensure proper mipmapping
- Increase anisotropic filtering
- Add subtle noise/variation to break repetition
- Pre-filter textures to remove problematic frequencies

From research:
> "With this naive implementation, the results are clearly visible. On the near side of the checkerboard, we can distinctly see the checkerboard pattern. However, as we look towards the far side, the pattern begins to distort into stripes, no longer resembling a checkerboard."

## Sources

**Web Research:**

- [WebGPU Unleashed: Mipmapping and Anisotropic Filtering](https://shi-yan.github.io/webgpuunleashed/Basics/mipmapping_and_anisotropic_filtering.html) - Shi Yan (accessed 2025-10-31)
  - Comprehensive WebGPU tutorial on mipmap generation and filtering
  - Code examples for automatic mipmap creation
  - Shader implementation details

- [Bilinear texture filtering – artifacts, alternatives, and frequency domain analysis](https://bartwronski.com/2020/04/14/bilinear-texture-filtering-artifacts-alternatives-and-frequency-domain-analysis/) - Bart Wronski, April 2020 (accessed 2025-10-31)
  - Deep technical analysis of filtering artifacts
  - Frequency domain perspective on filter behavior
  - Comparison of bilinear, bicubic, and biquadratic filters
  - Variance loss and Mach band effects

- [Texture Filtering: Techniques for Sharper 3D Renders](https://garagefarm.net/blog/texture-filtering-techniques-for-sharper-3d-renders) - GarageFarm.NET (accessed 2025-10-31)
  - Practical overview of filtering methods
  - Performance characteristics and trade-offs
  - Applications in games, VFX, and archviz

**Additional References:**

- Wikipedia: [Anisotropic Filtering](https://en.wikipedia.org/wiki/Anisotropic_filtering)
- Wikipedia: [Trilinear Filtering](https://en.wikipedia.org/wiki/Trilinear_filtering)
- GPU Gems 2: Fast Third-Order Texture Filtering (NVIDIA, referenced in research)
- Mitchell & Netravali: Reconstruction Filters in Computer Graphics (classic paper, referenced in research)

**Related Oracle Knowledge:**

- See applications/ folder for VLM visualization strategies
- See concepts/ for Level-of-Detail theory
- See architecture/ for ARR-COC framework integration
