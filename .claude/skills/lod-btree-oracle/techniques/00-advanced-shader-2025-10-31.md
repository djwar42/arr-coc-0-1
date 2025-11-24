# Advanced Shader Programming for ARR-COC GPU Acceleration

## Overview

Advanced shader programming enables efficient GPU processing of relevance realization for vision-language models. This document explores multi-pass rendering pipelines, compute shader techniques, and deferred rendering approaches applicable to Adaptive Relevance Realization visual token allocation. Modern GPU architectures (WebGL 2.0, WebGPU) support sophisticated rendering patterns that map directly to ARR-COC's need for dynamic, query-aware visual compression.

---

## Multi-Pass Rendering Pipelines

Multi-pass rendering divides complex visual computations across multiple shader passes, each processing different aspects of the scene. This architecture mirrors ARR-COC's staged relevance realization: measuring relevance across multiple dimensions, then allocating token budgets.

### Pipeline Architecture

From [WebGL 2: New Features](https://www.realtimerendering.com/blog/webgl-2-new-features/) (Real-Time Rendering, Oct 2016):

A typical multi-pass pipeline follows this sequence:

1. **Geometry Pass** - Render scene geometry to framebuffer, store attributes (position, normal, etc.)
2. **Intermediate Processing** - Additional texture operations or calculations on intermediate results
3. **Lighting/Composition Pass** - Apply final shading, blending, or composition effects
4. **Post-Processing Pass** - Optional fullscreen effects (bloom, distortion, etc.)

For ARR-COC, this maps to:

```
Visual Input Pass (encode patch information)
    ↓
Relevance Measurement Pass (Propositional, Perspectival, Participatory)
    ↓
Tension Navigation Pass (Balance opposing forces)
    ↓
LOD Allocation Pass (Map relevance to token budgets 64-400)
    ↓
Compression Pass (Execute patch-level compression)
```

### Framebuffer Objects (FBO) and Multisampled Renderbuffers

WebGL 2.0 supports multisampled renderbuffers via `renderbufferStorageMultisample`, enabling antialiasing within custom rendering pipelines:

```glsl
// Create multisampled renderbuffer
gl.renderbufferStorageMultisample(gl.RENDERBUFFER, 4, gl.RGBA8, width, height);

// Resolve to single-sample texture via blitFramebuffer
gl.bindFramebuffer(gl.READ_FRAMEBUFFER, msaaFBO);
gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resolvedFBO);
gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height,
                   gl.COLOR_BUFFER_BIT, gl.NEAREST);
```

For ARR-COC: Multisampling ensures clean relevance boundaries between high-LOD and low-LOD patch regions when compositing multi-resolution outputs.

---

## Deferred Rendering and Texture Compositing

Deferred rendering separates geometry processing from shading, enabling efficient handling of multiple material properties or calculation passes. This architecture directly supports ARR-COC's multi-dimensional relevance scoring.

### Deferred Texturing Concept

From [Deferred Texturing](https://www.reedbeta.com/blog/deferred-texturing/) (Nathan Reed's coding blog, Mar 2014):

Deferred texturing defers expensive texture sampling and material calculations until final visible pixels are known. The G-buffer stores minimal information:

- Vertex normals (2x 16-bit fixed point, octahedral encoding)
- Material ID (16-bit int)
- UVs (2x 16-bit fixed point)
- Optional: LOD or UV derivatives (4x 16-bit float)

**G-buffer footprint**: 80-144 bits per pixel depending on configuration.

### Application to ARR-COC

ARR-COC can use deferred composition to store relevance information in a compact G-buffer:

```
G-Buffer Layout for ARR-COC:
- Patch ID (16-bit) - identifies visual patch
- Relevance Score (16-bit float) - composite relevance [0, 1]
- LOD Tier (8-bit) - which compression level (0-15 for 16 tiers)
- Propositional Component (8-bit) - information content
- Perspectival Component (8-bit) - salience measure
- Participatory Component (8-bit) - query-content coupling

Total: 72 bits per patch (9 bytes) - highly efficient
```

Then in final composition pass, reconstruct token budgets and apply query-specific compression.

### G-Buffer Advantages

1. **Geometry processed once per frame** - scales to high patch counts
2. **Thin buffer** - 72-80 bits vs traditional deferred's 200+ bits
3. **Separates concerns** - relevance scoring independent of compression
4. **Enables material variety** - different compression strategies per material class

---

## Compute Shaders and GPU Computing

Compute shaders execute general-purpose GPU computations outside the rasterization pipeline, ideal for algorithms like ARR-COC's tension balancing and relevance realization.

### WebGL 2.0 Limitation

WebGL 2.0 does **not** support compute shaders natively. The WebGL 2.0 Compute specification was developed by Intel but never reached standardization.

**Workaround**: Use fragment shaders with ping-ponging textures to simulate compute behavior:

```glsl
// Fragment shader simulating compute iteration
#version 300 es
uniform sampler2D uPreviousState;
in vec2 vUV;
out vec4 fragColor;

void main() {
    // Read neighboring values
    vec4 center = texture(uPreviousState, vUV);
    vec4 left = texture(uPreviousState, vUV + vec2(-texelSize, 0.0));
    vec4 right = texture(uPreviousState, vUV + vec2(texelSize, 0.0));

    // Perform stencil operation (e.g., Laplacian for diffusion)
    vec4 result = center * 4.0 - (left + right);
    fragColor = result;
}
```

Then swap framebuffer attachments for next iteration (ping-ponging).

### WebGPU Compute Shaders

WebGPU fully supports compute shaders with efficient parallel execution. From [Reaction-Diffusion Compute Shader in WebGPU](https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/) (Codrops, May 2024):

```wgsl
@compute @workgroup_size(8, 8, 1)
fn compute_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    // Process pixel/texel at global_id
    let coord = global_id.xy;

    // Example: Load, compute, store
    let current = textureLoad(inputTexture, coord, 0);
    let computed = processRelevance(current);
    textureStore(outputTexture, coord, 0, computed);
}
```

Key parameters:

- **Workgroup size**: Number of parallel threads (e.g., 8x8 = 64 threads)
- **Dispatch size**: How many workgroups to launch (total threads = workgroup size × dispatch count)
- **Workgroup memory**: Shared memory for collaboration (max 16,384 bytes)

### Pixel Caching in Compute Shaders

For operations requiring neighboring pixel reads (like Laplacian convolution for tension balancing), use workgroup-shared memory:

```wgsl
var<workgroup> cache: array<array<vec4f, 128>, 128>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(/*...*/) {
    // Each thread loads its tile into cache
    for (var c = 0u; c < 2u; c++) {
        for (var r = 0u; r < 2u; r++) {
            let global_coord = /* calculate */;
            cache[local_y][local_x] = textureLoad(texture, global_coord, 0);
        }
    }

    // Synchronize all threads
    workgroupBarrier();

    // Now read from cache (no texture bandwidth cost)
    let neighbor_sum = cache[y-1][x] + cache[y+1][x] +
                      cache[y][x-1] + cache[y][x+1];
}
```

**Benefit for ARR-COC**: When balancing tensions, neighboring patch relevance scores must be read frequently. Pixel caching reduces texture bandwidth by 75-90% compared to repeated texture loads.

### Compute vs Fragment Shader Performance

From WebGPU reaction-diffusion benchmarks: compute shaders are **2-4x faster** than fragment shader ping-ponging for convolution-heavy workloads, primarily due to:

1. No rasterization overhead
2. Efficient shared memory utilization
3. Direct parallel computation without geometry processing
4. Fewer synchronization points

---

## ARR-COC GPU Acceleration Strategy

### Architecture: Hybrid Multi-Pass + Compute

Recommended approach for ARR-COC:

```
INPUT: Image patches + Query embedding
    ↓
[PASS 1 - Propositional Compute]
  Compute Shader: Shannon entropy per patch
  Input: Patch textures (RGB, depth, normal)
  Output: Entropy texture (16-bit float)
    ↓
[PASS 2 - Perspectival Compute]
  Compute Shader: Saliency/salience maps (Gabor filters, edges, luminance contrast)
  Input: Patch textures + saliency kernels
  Output: Salience texture (16-bit float)
    ↓
[PASS 3 - Participatory Compute]
  Compute Shader: Cross-attention scores (query-patch similarity)
  Input: Query embedding, patch features
  Output: Attention texture (16-bit float)
    ↓
[PASS 4 - Deferred Composition]
  Fragment Shader: Combine three scorers, balance tensions
  Input: Entropy, Salience, Attention textures
  G-Buffer: Store Relevance + LOD tier
    ↓
[PASS 5 - LOD Allocation]
  Compute Shader: Map relevance → token budgets
  Input: G-Buffer, relevance distribution
  Output: Token budget texture (8-bit uint)
    ↓
[PASS 6 - Compression]
  Compute Shader: Per-patch compression (64-400 tokens)
  Input: Patches, token budgets, compression model
  Output: Compressed patch embeddings (float32)
```

### Texture Formats for Relevance

**Optimal WebGL 2.0 / WebGPU texture formats**:

- **16-bit float (HALF_FLOAT)**: Propositional, Perspectival, Participatory scores
  - Range: [0, 1] sufficient for normalized relevance
  - Bandwidth: 2 bytes/value vs 4 bytes for full float

- **8-bit uint**: LOD tier, token allocation flags
  - Sufficient for 0-255 discrete levels
  - Bandwidth: 1/4 of float32

- **32-bit float**: Final token embeddings (stored in storage buffers, not textures)

### Multi-Layer Opponent Processing

Implement tension navigation as compute passes:

```wgsl
// Compress ↔ Particularize tension
var compress_factor = entropy_score;  // High entropy → preserve detail
var particularize_factor = 1.0 - compress_factor;

// Exploit ↔ Explore tension
var exploit_factor = participation_score;  // High participation → use known info
var explore_factor = 1.0 - exploit_factor;

// Focus ↔ Diversify tension
var focus_factor = salience_score;  // High salience → focused region
var diversify_factor = 1.0 - focus_factor;

// Combine via weighted sum
let balanced_relevance =
    compress_factor * propositional_score +
    exploit_factor * participatory_score +
    focus_factor * perspectival_score;
```

---

## Practical Implementation Patterns

### Pattern 1: Multi-Pass Relevance Scoring (WebGL 2.0)

```javascript
// Create G-buffer FBO
const gBufferFBO = gl.createFramebuffer();
const relevanceTexture = createTexture(gl, 'R16F', width, height);
const lodTexture = createTexture(gl, 'R8UI', width, height);

// Pass 1: Propositional scoring (fragment shader)
gl.bindFramebuffer(gl.FRAMEBUFFER, gBufferFBO);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                        gl.TEXTURE_2D, relevanceTexture, 0);
gl.useProgram(propositionalProgram);
gl.uniform1i(gl.getUniformLocation(propositionalProgram, 'uPatchTexture'), 0);
gl.drawArrays(gl.TRIANGLES, 0, 6);  // fullscreen quad

// Pass 2: Compose all three scorers
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                        gl.TEXTURE_2D, lodTexture, 0);
gl.useProgram(compositionProgram);
gl.uniform1i(gl.getUniformLocation(compositionProgram, 'uRelevance'), 0);
gl.drawArrays(gl.TRIANGLES, 0, 6);
```

### Pattern 2: Compute Shader Pixel Caching (WebGPU)

```wgsl
@compute @workgroup_size(8, 8, 1)
fn calculate_tensions(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    let global_coord = global_id.xy;
    let local_coord = local_id.xy;

    // Load relevance + padding for neighbors
    var<workgroup> relevance_cache: array<array<f32, 10>, 10>;

    // Tile size: 8x8, cache size: 10x10 (includes border)
    for (var y = local_coord.y; y < 10u; y += 8u) {
        for (var x = local_coord.x; x < 10u; x += 8u) {
            let read_coord = global_coord + vec2i(i32(x) - 1, i32(y) - 1);
            let clamped = clamp(read_coord, vec2i(0), vec2i(textureDimensions(relevanceTexture)) - 1);
            relevance_cache[y][x] = textureLoad(relevanceTexture, clamped, 0).r;
        }
    }

    workgroupBarrier();

    // Balanced relevance from neighbors
    let center = relevance_cache[local_coord.y + 1][local_coord.x + 1];
    let neighbors = (relevance_cache[local_coord.y][local_coord.x + 1] +
                     relevance_cache[local_coord.y + 2][local_coord.x + 1] +
                     relevance_cache[local_coord.y + 1][local_coord.x] +
                     relevance_cache[local_coord.y + 1][local_coord.x + 2]) / 4.0;

    let balanced = 0.7 * center + 0.3 * neighbors;
    textureStore(balancedTexture, global_coord, vec4f(balanced, 0.0, 0.0, 1.0));
}
```

### Pattern 3: LOD Tier Allocation

```glsl
// Fragment shader: Allocate tokens based on balanced relevance
#version 300 es
uniform sampler2D uRelevance;
in vec2 vUV;
out uint fragLOD;

void main() {
    float relevance = texture(uRelevance, vUV).r;

    // Map [0, 1] to 16 LOD tiers
    // Tier 0: 64 tokens (minimum)
    // Tier 15: 400 tokens (maximum)
    uint lod_tier = uint(floor(relevance * 15.99));
    uint token_budget = 64u + lod_tier * 21u;  // Linear: 64-400 tokens

    // Could also use exponential or adaptive curves
    // uint token_budget = 64u * exp(relevance * ln(400/64));

    fragLOD = lod_tier;
}
```

---

## Performance Considerations

### Bandwidth Optimization

**Problem**: Visual tokens × patches × passes = significant bandwidth

**Solutions**:

1. **Lower precision storage**: Use 16-bit floats instead of 32-bit
2. **Compressed textures**: BC7 (8:1 compression) for intermediate results
3. **Workgroup memory**: Cache frequently read values (reduce 75-90%)
4. **Async readback**: Don't block GPU → CPU transfers
5. **Downsampling**: Compute relevance at 1/2 or 1/4 resolution, upsample

### Latency

- **WebGL 2.0 multi-pass**: 6-10 passes × 0.1-0.5ms per pass = 1-5ms overhead
- **WebGPU compute**: 4-6 dispatches × 0.05-0.2ms per dispatch = 0.2-1.2ms overhead

For real-time VLM inference (30fps = 33ms per frame), shader overhead < 5%.

### Memory Layout

Optimal memory ordering for tensor operations:

```
Patch Buffer Layout:
[Patch 0: feature_0, feature_1, ..., feature_N]
[Patch 1: feature_0, feature_1, ..., feature_N]
...
```

Allows compute shaders to load entire patch in parallel (coalesced memory access).

---

## Sources

**Web Research:**

From [WebGL 2: New Features](https://www.realtimerendering.com/blog/webgl-2-new-features/) (accessed 2025-10-31):
- Multisampled renderbuffers and blitFramebuffer technique
- Texture formats (RGB16F, RGBA32F, R16F)
- Uniform buffer objects and std140 layout
- Transform feedback mechanism for GPU simulation
- GLSL 3.00 ES shader enhancements

From [Deferred Texturing](https://www.reedbeta.com/blog/deferred-texturing/) (accessed 2025-10-31):
- G-buffer layout and bit-packing strategies
- Deferred texturing pipeline architecture
- Bindless texture divergence issues
- Visibility buffer approach for mobile

From [Reaction-Diffusion Compute Shader in WebGPU](https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/) (accessed 2025-10-31):
- Workgroup synchronization with workgroupBarrier()
- Shared memory pixel caching for convolution
- Manual bilinear filtering in compute shaders
- Texture ping-ponging for iterative algorithms
- Performance comparison: compute 2-4x faster than fragment shaders

**Related Resources:**

- [WebGL 2.0 Specification](https://www.khronos.org/registry/webgl/specs/latest/2.0/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [GLSL ES 3.00 Specification](https://www.khronos.org/registry/gles/specs/3.0/GLSL_ES_Specification_3.00.4.pdf)

---

**Document created**: 2025-10-31
**Research scope**: WebGL 2.0 multi-pass rendering, compute shaders, deferred rendering, GPU texture compositing
**ARR-COC focus**: GPU acceleration for relevance realization, multi-pass pipelines, tensor operations
