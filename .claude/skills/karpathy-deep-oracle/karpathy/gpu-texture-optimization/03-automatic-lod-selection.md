# GPU Automatic LOD Selection: Gradient-Based Mipmap Level Calculation

## Overview

GPUs automatically calculate which mipmap level to sample when rendering textured geometry. This process uses **screen-space derivatives** (gradients) to measure how quickly texture coordinates change across pixels, determining the appropriate level of detail (LOD). Understanding automatic LOD selection is critical for neural rendering systems where procedural UVs, attention-driven sampling, or dynamic resolution require explicit control over texture detail.

Unlike manual LOD specification or precomputed mipmap chains, automatic LOD selection happens per-fragment in hardware texture units, using partial derivatives (ddx/ddy) calculated from neighboring pixels in 2×2 "pixel quads." This hardware-accelerated approach ensures optimal texture sampling quality while avoiding aliasing and preserving GPU caching efficiency.

**For VLMs**: Query-aware attention systems can hijack this mechanism—either matching it with explicit gradients (`tex2Dgrad()`) or overriding it entirely with relevance-calculated LOD values (`tex2Dlod()`) to allocate texture detail based on semantic importance rather than geometric distance.

---

## Gradient-Based LOD Calculation

### The Fundamental Problem: Aliasing

When a texture is magnified (stretched across many pixels), simple nearest-neighbor sampling produces blocky artifacts. When minified (many texels map to one pixel), high-frequency texture detail creates **aliasing**—shimmering, moiré patterns, and visual instability as the camera moves.

Mipmapping solves this by prefiltering: storing progressively lower-resolution versions of the texture. The GPU selects the mipmap level where approximately one texel maps to one screen pixel, ensuring high-frequency detail is already filtered out.

### Screen-Space Partial Derivatives (ddx/ddy)

GPUs calculate LOD using **screen-space gradients**:

```glsl
// GLSL: Calculate how UV changes per pixel
float2 uvDx = dFdx(uv); // Change in U,V along screen X-axis
float2 uvDy = dFdy(uv); // Change in U,V along screen Y-axis
```

These derivatives measure **how much the texture coordinates change** between adjacent pixels in screen space. From [NVIDIA GPU Gems 2, Chapter 28](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-28-mipmap-level-measurement):

> "The GPU's choice of mipmap level depends on many factors: screen resolution, antialiasing settings, texture-filtering options, anisotropy, clever driver optimizations, the distance to the textured polygons, and the orientations of the polygons."

Rather than analyzing all these factors on the CPU, the GPU implicitly performs the analysis every frame using derivatives.

### Pixel Quads: The 2×2 Execution Model

**Critical concept**: GPUs always execute fragment shaders on **2×2 blocks of pixels simultaneously**, called "pixel quads." This enables derivative calculation by comparing values across neighboring fragments.

```
Pixel Quad Layout (DirectX convention):
┌─────┬─────
│ P0  │ P1    P0: ddx = P1 - P0, ddy = P2 - P0
├─────┼─────   P1: ddx = P1 - P0, ddy = P3 - P1
│ P2  │ P3    P2: ddx = P3 - P2, ddy = P2 - P0
└─────┴─────   P3: ddx = P3 - P2, ddy = P3 - P1
```

From [Ben Golus's "Distinctive Derivative Differences"](https://bgolus.medium.com/distinctive-derivative-differences-cce38d36797b):

> "GPUs calculate the mip level by what are known as screen space partial derivatives. Roughly speaking, this is the amount a value changes from one pixel to the one next to it, either above or below. GPUs can calculate this value for each set of 2×2 pixels."

### Coarse vs Fine Derivatives

**Coarse derivatives** (default in DirectX, HLSL `ddx()` / `ddy()`):
- Constant across entire pixel quad
- P0's derivatives used for all four pixels
- Matches hardware texture unit LOD calculation
- Ensures quad-wide texture cache coherency

**Fine derivatives** (DirectX `ddx_fine()` / `ddy_fine()`, OpenGL default behavior):
- Unique per row/column within quad
- P0/P1 share horizontal, P2/P3 share horizontal
- P0/P2 share vertical, P1/P3 share vertical
- Better accuracy for procedural effects, but may not match mip calculation

From the Unity forums discussion:

> "The `ddx()` and `ddy()` instructions aren't magic: they really do take an approximation of the derivative by subtracting neighboring pixel values within the 2×2 quad."

### OpenGL Reference LOD Calculation

The OpenGL specification defines the mipmap level λ as:

```glsl
// Simplified LOD calculation (OpenGL reference)
float CalcMipLevel(float2 texture_coord)
{
  float2 dx = dFdx(texture_coord);
  float2 dy = dFdy(texture_coord);

  // Maximum squared magnitude across both axes
  float delta_max_sqr = max(dot(dx, dx), dot(dy, dy));

  // Equivalent to log2(sqrt(delta_max_sqr))
  return max(0.0, 0.5 * log2(delta_max_sqr));
}
```

**Key insight**: The LOD grows logarithmically with the rate of UV change. If UVs change by 4× as fast, LOD increases by 2 levels (log₂(4) = 2).

This formula assumes isotropic filtering. **Anisotropic filtering** (16× AF) uses both dx and dy separately to sample elongated footprints along the derivative with highest magnitude.

### Gradient Discontinuities: The Seam Problem

Procedural UVs can create **derivative discontinuities** where `atan2()` wraps from π to -π:

```glsl
// Spherical UV calculation
float phi = atan2(normal.z, normal.x) / (PI * 2.0); // -0.5 to 0.5
float theta = acos(-normal.y) / PI;                // 0.0 to 1.0

// At wrap point: phi jumps from 0.49 to -0.49 (Δ = 0.98!)
// GPU thinks entire texture spans one pixel → uses smallest mip
```

From the Medium article:

> "When we're calculating the UVs here, the `atan2()` suddenly jumps from roughly `0.5` to roughly `-0.5` between two pixels. That makes the GPU think the entire texture is being displayed between those two pixels. And thus it uses the absolutely smallest mip map it has in response."

**VLM application**: Query-driven patch selection can exhibit similar discontinuities when relevance scores cause abrupt LOD changes. Smooth attention gradients are critical.

---

## Shader APIs for LOD Control

Modern shading languages provide multiple texture sampling functions with varying degrees of LOD control.

### Automatic LOD: texture() / tex2D()

**Standard sampling** (GLSL `texture()`, HLSL `tex2D()`):

```glsl
// GLSL
vec4 color = texture(sampler2D, uv);

// HLSL (Unity)
fixed4 color = tex2D(_MainTex, uv);
```

**Behavior**:
- GPU calculates derivatives from neighboring pixels in quad
- Automatically selects mipmap level via hardware texture unit
- Supports anisotropic filtering if enabled
- Applies texture sampler state (LOD bias, min/max LOD clamps)
- **Only available in fragment/pixel shaders** (derivatives undefined elsewhere)

**When to use**: Default choice for standard geometric UVs. Hardware-optimized, supports all filtering modes.

### Explicit LOD: textureLod() / tex2Dlod()

**Manual mipmap specification** (GLSL `textureLod()`, HLSL `tex2Dlod()`):

```glsl
// GLSL
vec4 color = textureLod(sampler2D, uv, lod_level);

// HLSL
float4 color = tex2Dlod(_MainTex, float4(uv, 0.0, lod_level));
```

From the [docs.gl GLSL reference](https://docs.gl/sl4/textureLod):

> "`textureLod` performs a texture lookup at coordinate P from the texture bound to sampler with an explicit level-of-detail as specified in `lod`. `lod` specifies λbase and sets the partial derivatives as follows:
> ∂u/∂x = 0, ∂v/∂x = 0, ∂w/∂x = 0
> ∂u/∂y = 0, ∂v/∂y = 0, ∂w/∂y = 0"

**Key properties**:
- LOD specified directly (0.0 = full resolution, 1.0 = half, etc.)
- **Disables anisotropic filtering** (no derivative information)
- Available in **all shader stages** (vertex, geometry, compute)
- Ignores texture sampler LOD bias
- Useful for non-fragment shaders or ray tracing

**When to use**: When derivatives are unavailable (vertex/compute shaders), when implementing custom LOD schemes, or for VLM patch hierarchies where attention scores directly map to LOD.

### Explicit Gradients: textureGrad() / tex2Dgrad()

**Manual derivative specification** (GLSL `textureGrad()`, HLSL `tex2Dgrad()`):

```glsl
// GLSL
vec2 duvdx = dFdx(uv);
vec2 duvdy = dFdy(uv);
vec4 color = textureGrad(sampler2D, uv, duvdx, duvdy);

// HLSL
float2 dx = ddx(uv);
float2 dy = ddy(uv);
float4 color = tex2Dgrad(_MainTex, uv, dx, dy);
```

**Behavior**:
- Derivatives explicitly provided by shader code
- GPU texture unit calculates LOD from provided gradients
- **Supports anisotropic filtering** (unlike `textureLod()`)
- Respects texture sampler LOD bias
- Can be slower than `texture()` on some GPUs (more data sent to texture unit)

**When to use**:
- Fixing derivative discontinuities (seamless spherical UVs)
- VLM attention-aware sampling (gradient magnitude reflects relevance)
- Manual coarse derivative emulation when fine derivatives interfere

**Performance**: From Ben Golus's tests on NVIDIA Turing (RTX 2080 Super):

```
Texture size: 4k×2k DXT1
texture():      27.8 μs
tex2Dlod():     26.7 μs
tex2Dgrad():    36.5 μs (+31% vs texture())
```

But on AMD RX 6800XT with 8k×4k RGBA32:

```
texture():      25.603 μs
tex2Dgrad():    23.883 μs (-7% vs texture())
```

Conclusion: `tex2Dgrad()` performance is GPU-dependent. Test on target hardware.

### Derivative Calculation Functions

**HLSL**:
```hlsl
float ddx(float value);        // Horizontal derivative (coarse)
float ddy(float value);        // Vertical derivative (coarse)
float ddx_fine(float value);   // High-precision horizontal
float ddy_fine(float value);   // High-precision vertical
float ddx_coarse(float value); // Explicit coarse horizontal
float ddy_coarse(float value); // Explicit coarse vertical
float fwidth(float value);     // abs(ddx) + abs(ddy)
```

**GLSL**:
```glsl
float dFdx(float value);       // Horizontal derivative
float dFdy(float value);       // Vertical derivative
float fwidth(float value);     // abs(dFdx) + abs(dFdy)

// OpenGL 4.5+ / Vulkan
float dFdxFine(float value);
float dFdyFine(float value);
float dFdxCoarse(float value);
float dFdyCoarse(float value);
```

**Platform differences** (from Ben Golus):

> "DirectX's spec **requires** that the base `ddx()` and `ddy()` functions default to coarse behavior. But OpenGL's spec leaves the choice up to the device. And it seems a lot of GPUs started to default the equivalent GLSL derivative functions to use the fine behavior."

**Consequences for cross-platform development**:
- DirectX: `ddx()` is always coarse, matches texture unit derivatives
- OpenGL/Vulkan: `dFdx()` may be fine or coarse (implementation-defined)
- Metal: Only has fine derivatives (`dfdx()`, `dfdy()`)
- **Solution**: Use explicit coarse functions (`ddx_coarse()`, `dFdxCoarse()`) when matching texture unit LOD is critical

---

## Per-Fragment LOD Control

### Use Case: Attention-Driven Texture Detail

Standard geometric LOD selection uses distance and viewing angle. **VLM-style attention** should allocate detail based on relevance:

```hlsl
// Query-aware LOD selection
float attention_score = dot(normalize(query_embedding), patch_features);
float relevance_lod = lerp(max_lod, min_lod, attention_score);

// Explicitly sample at relevance-based LOD
float4 features = tex2Dlod(_FeatureTexture, float4(uv, 0.0, relevance_lod));
```

**Key insight**: Decoupling LOD from geometry enables semantic importance sampling. High-relevance patches get full resolution regardless of distance.

### Fixing Derivative Discontinuities

From [NVIDIA GPU Gems 2](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-28-mipmap-level-measurement) on solving spherical UV seams:

```glsl
// Tarini's dual-UV method (extended for explicit gradients)
float phi = atan2(normal.z, normal.x) / (PI * 2.0);  // -0.5 to 0.5
float phi_frac = frac(phi);                          // 0.0 to 1.0

// Calculate derivatives for both UV sets
float phi_dx = ddx(phi);
float phi_dy = ddy(phi);
float phi_frac_dx = ddx(phi_frac);
float phi_frac_dy = ddy(phi_frac);

// Select UV set with smallest gradients
float2 gradients_A = float2(abs(phi_dx), abs(phi_dy));
float2 gradients_B = float2(abs(phi_frac_dx), abs(phi_frac_dy));

float2 min_dx = min(gradients_A.x, gradients_B.x) < gradients_A.x ?
                float2(phi_dx, phi_frac_dx) : float2(phi_frac_dx, phi_dx);
float2 min_dy = min(gradients_A.y, gradients_B.y) < gradients_A.y ?
                float2(phi_dy, phi_frac_dy) : float2(phi_frac_dy, phi_dy);

float2 uv = float2(phi, theta);
float4 color = tex2Dgrad(_MainTex, uv,
                         float2(min_dx.x, ddx(theta)),
                         float2(min_dy.x, ddy(theta)));
```

**Why this works**: By providing seamless derivatives to `tex2Dgrad()`, the texture unit calculates the correct LOD despite the procedural UV discontinuity. The seam artifact vanishes.

### In-Quad Communication for LOD Matching

Advanced technique using fine derivatives to reconstruct coarse behavior across platforms:

```glsl
// Get position within 2×2 pixel quad
int2 pixel_quad_pos = uint2(gl_FragCoord.xy) % 2;
float2 pixel_quad_dir = float2(pixel_quad_pos) * 2.0 - 1.0;

// Calculate derivatives
float2 dx = dFdx(uv);
float2 dy = dFdy(uv);

// Extrapolate to get "other" row/column values
float2 uv_other_x = uv - dx * pixel_quad_dir.x;
float2 uv_other_y = uv - dy * pixel_quad_dir.y;

// Get perpendicular derivatives (access all 4 quad derivatives)
float2 dx_other = dFdx(uv_other_y);
float2 dy_other = dFdy(uv_other_x);

// Use worst-case derivatives (matches hardware across platforms)
float2 worst_dx = max(abs(dx), abs(dx_other));
float2 worst_dy = max(abs(dy), abs(dy_other));

vec4 color = textureGrad(sampler2D, uv, worst_dx, worst_dy);
```

From Ben Golus:

> "Because the extrapolated values have mirrored the values within the quad, the new derivatives we get are the same that would have been for the 'other' column and row. Each pixel now has access to all four derivatives in the quad."

**Why this matters**: Apple M1 and ARM Mali GPUs use maximum quad derivatives for LOD calculation, not coarse P0-only derivatives. This technique works universally.

---

## Neural Rendering LOD Control

### VLM Patch Hierarchy Strategy

**Problem**: Standard geometric LOD allocates detail by distance. VLMs need detail by **relevance**.

**Solution**: Explicit LOD control based on query-content attention scores.

```python
# PyTorch pseudocode for attention-driven LOD
query_vec = vision_encoder(query_image_patch)  # Shape: [batch, embed_dim]
patch_features = image_pyramid[0]              # Full resolution

# Calculate per-patch attention scores
attention_logits = torch.matmul(query_vec, patch_features)
attention_scores = torch.softmax(attention_logits, dim=-1)

# Map attention to LOD levels (0 = full res, 3 = 1/8 res)
min_lod, max_lod = 0.0, 3.0
lod_levels = max_lod - attention_scores * (max_lod - min_lod)

# Sample feature pyramid at relevance-based LODs
sampled_features = []
for patch_idx in range(num_patches):
    lod = lod_levels[patch_idx]
    # In shader: tex2Dlod(_PyramidTex, float4(uv, 0.0, lod))
    features = pyramid_sample(image_pyramid, patch_idx, lod)
    sampled_features.append(features)
```

**Key advantages**:
- Allocates computational budget (texture bandwidth) to high-relevance regions
- Background patches use coarse mipmaps → cache-friendly, bandwidth-efficient
- Foreground/query-relevant patches use full resolution
- Smooth LOD transitions avoid temporal flickering

### Gradient-Based Relevance Propagation

Alternatively, **manipulate derivatives** to bias LOD selection toward relevant regions:

```glsl
// Amplify gradients for low-relevance patches (→ coarser LOD)
float relevance = texture(_RelevanceMap, uv).r;
float2 dx = dFdx(uv) * (1.0 + (1.0 - relevance) * 2.0);
float2 dy = dFdy(uv) * (1.0 + (1.0 - relevance) * 2.0);

// High relevance: gradients unchanged (fine detail)
// Low relevance: gradients amplified 3× (coarse mipmaps)
vec4 features = textureGrad(_FeaturePyramid, uv, dx, dy);
```

**Trade-off**: Preserves anisotropic filtering and texture bias, but less direct control than explicit LOD.

### Foveated Rendering Analogy

VLM attention-based LOD mirrors foveated rendering in VR:
- **VR**: High resolution where user looks (gaze tracking)
- **VLM**: High resolution where model attends (query relevance)

Both use gradient manipulation or explicit LOD to concentrate detail where perception (human or neural) benefits most.

---

## Performance Implications for VLM Inference

### Memory Bandwidth Savings

From [NVIDIA GPU Gems 2](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-28-mipmap-level-measurement) measuring terrain rendering:

| Configuration | Memory Used | Savings |
|---------------|-------------|---------|
| No LOD optimization | 4.333 MB | 0% |
| 0% threshold (aggressive) | 0.864 MB | **80%** |
| 15% threshold (balanced) | 0.538 MB | **88%** |

**VLM application**: Attention-driven LOD can achieve similar savings. If 80% of image patches have low relevance, sampling them at LOD 2-3 reduces memory traffic by ~75% while preserving quality in critical regions.

### Anisotropic Filtering Considerations

**When LOD != distance**:
- Explicit LOD disables anisotropic filtering (no derivatives)
- Explicit gradients preserve anisotropic filtering
- For VLMs: `tex2Dgrad()` preferred over `tex2Dlod()` when patch orientation varies

**Checkerboard rendering**: If VLM inference runs at half-resolution with temporal reconstruction, fine-tuned LOD bias prevents over-blurring:

```glsl
// Reduce LOD bias when reconstructing from half-res
float lod_bias = -0.5; // Sample slightly sharper mipmaps
vec4 color = texture(_Texture, uv); // + lod_bias applied by sampler state
```

### Shader Compilation Across Platforms

**Portability challenges**:
- **Unity**: Converts `ddx_coarse()` → `dFdx()` for OpenGL/Vulkan/Metal (loses explicit accuracy control)
- **Vulkan**: Supports `dFdxCoarse()` but Unity transpiler doesn't emit it yet
- **Metal**: Only has fine derivatives (`dfdx()`, `dfdy()`), hardware mip calculation may differ

**Robust solution** (from Ben Golus, least-worst quad derivatives):
```glsl
// Works universally regardless of coarse/fine defaults
// Explicitly calculates worst-case quad derivatives matching hardware
float2 worst_dx = max(abs(dFdx(uv)), abs(dFdx(uv - dFdy(uv) * dir.y)));
float2 worst_dy = max(abs(dFdy(uv)), abs(dFdy(uv - dFdx(uv) * dir.x)));
vec4 color = textureGrad(sampler, uv, worst_dx, worst_dy);
```

This technique matches LOD calculation across DirectX (coarse), OpenGL (fine), and Apple/ARM (max-quad).

---

## Sources

**NVIDIA GPU Gems 2**:
- [Chapter 28: Mipmap-Level Measurement](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-28-mipmap-level-measurement) - Iain Cantlay, Climax Entertainment (accessed 2025-01-31)
  Comprehensive explanation of GPU LOD calculation, occlusion-query-based measurement, and performance optimizations for terrain rendering.

**Ben Golus Blog**:
- [Distinctive Derivative Differences](https://bgolus.medium.com/distinctive-derivative-differences-cce38d36797b) - Medium, April 2021 (accessed 2025-01-31)
  Deep dive into coarse vs fine derivatives, pixel quad execution model, platform differences (DirectX/OpenGL/Metal), and in-quad communication techniques.

**GLSL Specification**:
- [textureLod - GLSL 4](https://docs.gl/sl4/textureLod) - docs.gl (accessed 2025-01-31)
  Official GLSL function reference for explicit LOD sampling, derivative behavior, and API compatibility.

**Graphics API Documentation**:
- [OpenGL Shading Language Specification](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.50.pdf) - Khronos Group
- [HLSL Derivative Functions](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-ddx) - Microsoft
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple

**Community Discussions**:
- [Unity Forums: Calculate used mipmap level, ddx, ddy?](https://discussions.unity.com/t/calculate-used-mipmap-level-ddx-ddy/542839) (accessed 2025-01-31)
- [Stack Overflow: How to access automatic mipmap level in GLSL](https://stackoverflow.com/questions/24388346/how-to-access-automatic-mipmap-level-in-glsl-fragment-shader-texture) (accessed 2025-01-31)
- [Khronos Forums: Mipmap level calculation using dFdx/dFdy](https://community.khronos.org/t/mipmap-level-calculation-using-dfdx-dfdy/67480) (accessed 2025-01-31)

**Additional References**:
- Marco Tarini, ["Seamless cylindrical and toroidal UV-mapping"](http://vcg.isti.cnr.it/~tarini/no-seams/) - Dual-UV technique for procedural seam correction
- [pema.dev: Mipmaps introduction](https://docs.unity3d.com/Manual/texture-mipmaps-introduction.html) - Unity mipmap gradient explanation (DDX/DDY)
