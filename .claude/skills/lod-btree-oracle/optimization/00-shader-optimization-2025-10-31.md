# Shader Optimization and Performance

**Topic**: GPU shader optimization, profiling, and performance analysis
**Created**: 2025-10-31
**Scope**: Performance profiling, instruction optimization, mobile optimization, precision, ARR-COC shader performance

---

## Overview

Shader optimization is critical for real-time graphics performance, particularly on mobile and resource-constrained platforms. Modern GPUs execute thousands of shader instances in parallel, making per-shader efficiency multiply across the entire scene. This guide covers profiling tools, optimization techniques, and best practices for achieving maximum shader performance.

**Key Performance Factors:**
- Instruction count (ALU operations vs memory access)
- Memory bandwidth (texture sampling, buffer access)
- Register pressure (VGPR/SGPR usage, occupancy)
- Branch divergence (dynamic vs static branching)
- Precision requirements (lowp/mediump/highp tradeoffs)

---

## Performance Profiling Tools

### GPU Vendor Profilers

**Arm Performance Studio (Mobile)**

From [Arm Performance Studio 2024.4](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31):

Frame Advisor shader analysis metrics:
- **Cycle cost breakdown**: Per functional unit (arithmetic, load/store, varying, texture)
- **Register usage**: Work registers, uniform registers, occupancy impact
- **Stack spills**: Compiler-generated memory allocations (very expensive on GPU)
- **FP16 usage**: Percentage of operations at 16-bit precision
- **Shortest/longest path cycles**: Control flow impact analysis

Key metrics per shader program:
```
Shortest path cycles: Min execution time
Longest path cycles: Max execution time (worst case)
Total emitted cycles: All instructions regardless of control flow
Work registers: General purpose read-write per thread
Uniform registers: Read-only constants per program
Occupancy: % of maximum shader core thread capacity
FP16 usage %: Arithmetic at 16-bit or lower precision
```

**NVIDIA Nsight Graphics**

From [Advanced API Performance](https://developer.nvidia.com/blog/tag/advanced-api-performance/) resources (accessed 2025-10-31):

GPU Trace analysis capabilities:
- Peak-Performance-Percentage (P3) method for bottleneck identification
- Per-draw shader instruction analysis
- Warp occupancy and register pressure visualization
- Memory bandwidth utilization tracking
- Dynamic branching cost analysis

Nsight shader debugging features:
- Live VGPR (vector general purpose register) tracking
- Instruction-level performance counters
- Divergence visualization for warp execution
- Cache hit/miss analysis for texture/buffer access

**AMD Radeon GPU Profiler**

From [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

Radeon Developer Tool Suite components:
- **RGP (Radeon GPU Profiler)**: Low-level optimization, wave occupancy analysis
- **RGA (Radeon GPU Analyzer)**: Offline shader compilation, ISA inspection
- **RDP (Radeon Developer Panel)**: Driver experiments for low-level control

RGA shader analysis:
- VGPR/SGPR register usage per shader stage
- Instruction count per functional unit (VALU, SALU, VMEM, etc)
- Wave occupancy based on register pressure
- GCN/RDNA ISA disassembly for manual optimization

### Web-Based Profiling

**WebGL Fragment Cost Analysis**

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31):

Browser developer tools integration:
- Chrome: GPU profiler in DevTools Performance tab
- Firefox: Canvas debugger with shader call stacks
- Shader complexity warnings in console

Fragment shader profiling considerations:
- Fragment shaders run many more times than vertex shaders
- Per-pixel operations multiply by screen resolution
- Overdraw amplifies fragment shader cost
- Mobile fillrate constraints more severe than desktop

---

## Instruction Count Optimization

### ALU vs Memory Operations

**Arithmetic Unit (ALU) Operations**

From [Arm GPU Best Practices](https://developer.arm.com/documentation/101897/0301) and [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

Instruction cost hierarchy (fastest to slowest):
1. **FP16 arithmetic**: 2x throughput vs FP32, lower register pressure
2. **FP32 arithmetic**: Standard precision operations
3. **Transcendental functions**: sin/cos/exp/log (use built-ins, not custom implementations)
4. **FP64 operations**: Very slow on consumer GPUs (avoid unless required)

ALU optimization strategies:
- Prefer built-in functions (dot, normalize, mix) over manual implementations
- Move calculations from fragment to vertex shader when possible
- Use MAD (multiply-add) instructions when available
- Reduce live ranges of variables to minimize register pressure

**Memory Operations Cost**

Texture sampling is expensive:
- L1 cache miss: ~100-200 cycles
- L2 cache miss: ~400-600 cycles
- DRAM access: ~800+ cycles
- Dependent texture reads amplify latency (result of one sample used for next lookup)

Buffer/SSBO access patterns:
- Coalesced access (neighboring threads access neighboring memory): Fast
- Random/scattered access: Slow, poor cache utilization
- Uniform buffer access: Cached, relatively cheap
- Storage buffer reads: Dependent on access pattern

### Vertex vs Fragment Shader Work Distribution

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31):

**Prefer vertex shader calculations:**

Fragment shaders execute per-pixel, vertex shaders execute per-vertex. For a 1080p screen with a mesh of 1000 triangles:
- Vertex shader invocations: ~3,000
- Fragment shader invocations: ~2,073,600 (1920×1080)

Optimization approach:
```glsl
// BAD: Per-fragment calculation
// Fragment shader
varying vec2 texCoord;
void main() {
    vec2 animated = texCoord + time * speed; // Runs 2M times
    gl_FragColor = texture2D(tex, animated);
}

// GOOD: Per-vertex calculation with interpolation
// Vertex shader
attribute vec2 aTexCoord;
varying vec2 vAnimatedCoord;
void main() {
    vAnimatedCoord = aTexCoord + time * speed; // Runs 3K times
    gl_Position = ...;
}
// Fragment shader
varying vec2 vAnimatedCoord;
void main() {
    gl_FragColor = texture2D(tex, vAnimatedCoord); // Interpolated cheaply
}
```

When to keep work in fragment shader:
- Per-pixel lighting requirements
- High-frequency detail (normal mapping)
- Screen-space effects
- Dense vertex meshes (more vertices than pixels)

---

## Branching and Control Flow

### Dynamic Branching Cost

From [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) and [Arm Performance Studio](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31):

**Branch Divergence Problem:**

GPUs execute shader threads in groups (warps/wavefronts):
- NVIDIA: 32 threads per warp
- AMD: 32-64 threads per wavefront
- ARM: 4-16 threads per execution group

When threads diverge at a branch, both paths execute:
```glsl
if (condition) {
    // Path A
} else {
    // Path B
}
// If 16 threads take path A, 16 take path B:
// All 32 threads execute BOTH paths (50% masked out each time)
// Actual cost: Path A + Path B
```

**Branch Optimization Strategies:**

1. **Static/Uniform branching** (all threads take same path):
```glsl
uniform bool useFeature; // Same for all threads
if (useFeature) {
    // No divergence, only active path executes
}
```

2. **Early return** (reduce work for some threads):
```glsl
if (alpha < 0.01) {
    discard; // Early exit for transparent pixels
    // Subsequent expensive calculations avoided
}
// Expensive lighting calculations...
```

3. **Branchless alternatives** (when divergence cost > both paths):
```glsl
// BAD: Dynamic branch
float result;
if (x > 0.5) {
    result = expensiveA(x);
} else {
    result = expensiveB(x);
}

// GOOD: Branchless (if both paths are cheap)
float resultA = expensiveA(x);
float resultB = expensiveB(x);
float result = mix(resultB, resultA, step(0.5, x));
```

**Compile-Time Constants:**

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices):

Use shader permutations for major feature branches:
```glsl
// Compile different shader variants
#ifdef NORMAL_MAPPING
    vec3 normal = texture2D(normalMap, uv).xyz * 2.0 - 1.0;
#else
    vec3 normal = vNormal;
#endif
```

Trade-off: More shader variants = longer compile times, more memory.
Best for: Features with low cardinality (2-4 options), high performance impact.

---

## Precision Qualifiers

### GLSL Precision System

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31):

**WebGL 1 / ESSL 100 minimum requirements:**

| Precision | Type Analog | Range | Min Value | Accuracy |
|-----------|-------------|-------|-----------|----------|
| **highp float** | float24 | (-2^62, 2^62) | 2^-62 | 2^-16 relative |
| **mediump float** | IEEE float16 | (-2^14, 2^14) | 2^-14 | 2^-10 relative |
| **lowp float** | 10-bit fixed | (-2, 2) | 2^-8 | 2^-8 absolute |
| **highp int** | int17 | (-2^16, 2^16) | - | - |
| **mediump int** | int11 | (-2^10, 2^10) | - | - |
| **lowp int** | int9 | (-2^8, 2^8) | - | - |

**WebGL 2 / ESSL 300 minimum requirements:**

| Precision | Type Analog | Range | Min Value | Accuracy |
|-----------|-------------|-------|-----------|----------|
| **highp float** | IEEE float32 | (-2^126, 2^127) | 2^-126 | 2^-24 relative |
| **mediump float** | IEEE float16 | (-2^14, 2^14) | 2^-14 | 2^-10 relative |
| **lowp float** | 10-bit fixed | (-2, 2) | 2^-8 | 2^-8 absolute |
| **highp int** | int32 | [-2^31, 2^31] | - | - |
| **mediump int** | int16 | [-2^15, 2^15] | - | - |
| **lowp int** | int9 | [-2^8, 2^8] | - | - |

**Default Precision in Shaders:**

Vertex shader:
```glsl
precision highp float;
precision highp int;
precision lowp sampler2D;
precision lowp samplerCube;
```

Fragment shader:
```glsl
precision mediump int;
precision lowp sampler2D;
precision lowp samplerCube;
// NO default for float! Must specify.
```

### Precision Selection Guidelines

**Always require highp:**
- Position calculations (vertex positions, clip space)
- Depth values (z-buffer precision)
- High-precision UV coordinates (large textures)
- Coordinate transformations

**Safe for mediump:**
- Color calculations (HDR requires more consideration)
- Normal vectors (renormalize after interpolation)
- Simple UV coordinates (< 4K textures)
- Lighting calculations (specular can show artifacts)

**Safe for lowp:**
- Final color output (0-1 range)
- Simple flags/booleans
- Normalized vectors (if renormalized frequently)
- Texture coordinates for small textures

**Fragment shader precision pattern:**

From [Arm GPU Best Practices](https://developer.arm.com/documentation/101897/0301):

```glsl
// Always check for highp support, fallback to mediump
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

// Explicit precision for performance-critical samplers
uniform highp sampler2D depthMap;    // Depth requires precision
uniform mediump sampler2D colorMap;  // Color can use mediump
uniform lowp sampler2D noiseMap;     // Noise doesn't need precision
```

### Performance Impact of Precision

From [Arm Performance Studio 2024.4](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31):

**FP16 (mediump) vs FP32 (highp) benefits:**

1. **Arithmetic throughput**: 2x faster (can execute 2× FP16 ops per cycle)
2. **Register pressure**: 2x more values fit in same register storage
3. **Memory bandwidth**: 2x less data transferred
4. **Occupancy**: Higher thread count possible with lower register usage

**Measuring FP16 usage:**

Arm Frame Advisor FP16 usage metric:
- Target: > 80% of arithmetic at FP16 for mobile
- < 50% indicates opportunity for optimization
- Sort shaders by lowest FP16% to find optimization targets

**Precision reduction strategy:**

```glsl
// Analyze per-variable, not whole shader
precision mediump float; // Default

// Override specific high-precision needs
highp vec4 position;      // Vertex positions
highp float depth;        // Depth calculations
highp mat4 mvpMatrix;     // Transformation matrices

mediump vec3 normal;      // Normals (renormalize after)
mediump vec2 texCoord;    // Texture coordinates
mediump vec3 color;       // Color values

lowp vec4 finalColor;     // Final output
```

---

## Mobile GPU Optimization

### Mobile vs Desktop Architecture Differences

From [Arm GPU Best Practices](https://developer.arm.com/documentation/101897/0301) and [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

**Tile-Based Deferred Rendering (TBDR):**

Mobile GPUs (Arm Mali, Apple, Qualcomm Adreno, Imagination PowerVR):
- Divide framebuffer into tiles (16×16 to 32×32 pixels)
- Process entire tile in on-chip cache
- Minimize memory bandwidth to DRAM

TBDR benefits:
- Automatic pixel-level early-Z rejection
- Efficient blending within tile memory
- Lower power consumption

TBDR optimization opportunities:
- `invalidateFramebuffer()` for temporary attachments (depth/stencil)
- Avoid reading back framebuffer mid-frame (breaks tiling)
- Minimize attachment count (color + depth + stencil = bandwidth)

**Bandwidth Constraints:**

Mobile memory bandwidth is the primary bottleneck:
- Desktop GPU: 400-900 GB/s
- High-end mobile: 30-60 GB/s (~10x slower)

Bandwidth reduction strategies:
1. Lower resolution rendering (scale down, upscale with bilinear)
2. Compressed texture formats (ETC2, ASTC)
3. Reduce overdraw (Z-prepass selective, not universal)
4. Lower precision (FP16 textures, render targets)

### Fillrate Optimization

From [Wonderland Engine WebXR Profiling](https://wonderlandengine.com/news/profiling-webxr-applications/) (accessed 2025-10-31):

**Fillrate definition:** Pixels per second the GPU can process

Fragment shader cost directly impacts fillrate:
```
Effective fillrate = Peak fillrate / Fragment shader complexity
```

**Measuring fillrate bottlenecks:**

1. Reduce shader complexity to minimal:
```glsl
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Solid color
}
```

2. If frame rate dramatically increases → fillrate/fragment shader bound
3. If frame rate unchanged → vertex/geometry/CPU bound

**Fillrate optimization techniques:**

Reduce overdraw:
- Depth pre-pass (selective, for complex shaders only)
- Occlusion culling
- Draw order: front-to-back for opaque, back-to-front for transparent

Simplify fragment shaders:
- Move calculations to vertex shader
- Use texture lookups instead of expensive math (gradient maps)
- LOD shaders (simpler shaders for distant objects)

Mobile-specific fillrate concerns:
- 4K mobile displays stress fillrate severely
- Dynamic resolution scaling essential for VR/AR
- Render to lower resolution, upscale final composite

---

## Texture Sampling Optimization

### Texture Access Patterns

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) and [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

**Mipmapping for 3D scenes:**

Always use mipmaps for textures that vary in distance:
- Memory overhead: Only 33% (1 + 1/4 + 1/16 + 1/64 + ...)
- Performance benefit: Massive for distant/minified textures

```glsl
// Enable mipmapping
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
gl.generateMipmap(gl.TEXTURE_2D);
```

Without mipmaps for distant textures:
- Poor cache locality (neighboring pixels sample distant texels)
- Aliasing artifacts (undersampling high-frequency detail)
- Reduced texture cache hit rate

**When to skip mipmaps:**
- 2D UI elements (never zoomed out)
- Textures always viewed at 1:1 pixel ratio
- Video textures (can't generate mipmaps for external data)

**Dependent Texture Reads:**

Avoid texture lookups that depend on previous lookups:
```glsl
// BAD: Dependent reads (each read waits for previous)
vec4 offset = texture2D(offsetMap, uv);
vec4 color = texture2D(colorMap, uv + offset.xy); // Must wait for first read

// BETTER: Independent reads (can issue in parallel)
vec4 offset = texture2D(offsetMap, uv);
vec4 color1 = texture2D(colorMap, uv);
vec4 result = mix(color1, color2, offset.x); // Combine results
```

Exception: When dependent read is intentional design (parallax mapping, etc.)

### Texture Compression

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31):

**GPU Compressed Formats:**

Benefits over JPG/PNG:
- Smaller in GPU memory (not just download size)
- Faster texture sampling (less memory bandwidth)
- Decompression happens in texture cache hardware

Universal support matrix:
```
Desktop:    WEBGL_compressed_texture_s3tc (DXT1/DXT5)
Android:    WEBGL_compressed_texture_etc (ETC2)
iOS:        WEBGL_compressed_texture_pvrtc (PVRTC)
Modern:     WEBGL_compressed_texture_astc (highest quality)
```

**Basis Universal:**

Single compressed file that transcodes at load time:
- ~3-4× smaller than uncompressed over-the-wire
- Transcodes to platform-native format (S3TC/ETC2/PVRTC)
- Similar quality to JPEG, faster than runtime
- Library: [Basis Universal](https://github.com/BinomialLLC/basis_universal)

Compression best practices:
- Use for color/albedo maps (not for normals, roughness)
- Quality acceptable for artistic content, not precision data
- Authoring tools: compressonator, ARM texture compression tool

---

## Register Pressure and Occupancy

### Understanding GPU Occupancy

From [Arm Performance Studio 2024.4](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) and [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

**Occupancy definition:** Percentage of maximum concurrent threads actually running

GPU cores can context-switch between threads to hide latency:
- Texture fetch takes 400 cycles
- If only 1 thread: GPU stalls for 400 cycles
- If 8 threads: Switch to other threads while waiting

**Register Pressure:**

Work registers (VGPRs) are divided among active threads:
```
Max threads = Total VGPR pool / VGPRs per thread
```

Example (AMD RDNA):
- Total VGPRs: 256 per SIMD
- Shader uses 64 VGPRs: Max 4 wavefronts (256/64)
- Shader uses 32 VGPRs: Max 8 wavefronts (256/32)

**Reducing Register Pressure:**

1. Use lower precision (2× more FP16 values per register)
2. Reduce variable live ranges:
```glsl
// BAD: Long live range
vec3 temp1 = expensiveCalc1(input);
vec3 temp2 = expensiveCalc2(input);
vec3 temp3 = expensiveCalc3(input);
vec3 result = temp1 + temp2 + temp3; // All temps live simultaneously

// GOOD: Short live ranges
vec3 result = vec3(0.0);
result += expensiveCalc1(input); // temp1 dead after this
result += expensiveCalc2(input); // temp2 dead after this
result += expensiveCalc3(input); // temp3 dead after this
```

3. Simplify shader (fewer operations = fewer temps)
4. Break into multiple passes (if register spilling occurs)

**Stack Spills (Critical Problem):**

From [Arm Performance Studio 2024.4](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4):

When compiler runs out of registers, variables spill to memory:
- Extremely expensive (memory latency per access)
- Indicates shader is too complex

Prevention:
1. Reduce precision (FP32 → FP16)
2. Reduce variable live ranges
3. Simplify shader logic
4. Split into multiple shader passes

RGP/Frame Advisor metrics:
- Stack spills > 0 bytes: Immediate optimization target
- Look for "Stack" in shader ISA output

---

## ARR-COC Shader Performance

### 13-Channel Efficient Processing

ARR-COC processes 13-channel textures for multi-way relevance knowing:
- RGB (3): Propositional information content
- RGBA (4): Perspectival salience landscapes
- RGBA (4): Participatory query coupling
- RG (2): Metadata (patch LOD level, coordinates)

**Optimization Strategy:**

1. **Precision selection per channel:**
```glsl
precision mediump float; // Default

// Propositional (information content): mediump sufficient
mediump vec3 propositional = texture(propTex, uv).rgb;

// Perspectival (salience): mediump for visual salience
mediump vec4 perspectival = texture(salTex, uv);

// Participatory (query coupling): may require highp for precision
highp vec4 participatory = texture(queryTex, uv);

// Metadata: lowp for discrete LOD levels
lowp vec2 metadata = texture(metaTex, uv).rg;
```

2. **Batch texture sampling:**
```glsl
// Issue all independent texture reads together (parallelizable)
mediump vec3 prop = texture(propTex, uv).rgb;
mediump vec4 persp = texture(perspTex, uv);
highp vec4 partic = texture(particTex, uv);
lowp vec2 meta = texture(metaTex, uv).rg;

// Then process results (GPU can fetch while computing)
float relevance = computeRelevance(prop, persp, partic, meta);
```

3. **Avoid dependent reads in relevance calculation:**
```glsl
// BAD: Each LOD level requires new texture fetch
float lod = texture(metaTex, uv).r;
vec3 detail = textureLod(detailTex, uv, lod); // Dependent read

// GOOD: Pre-compute LOD in vertex shader or use automatic mipmaps
varying float vLod; // From vertex shader
vec3 detail = textureLod(detailTex, uv, vLod); // Independent of fragment texture reads
```

### Multi-Resolution LOD Processing

ARR-COC uses 64-400 token budgets per patch (variable resolution):

**Vertex shader optimization:**
```glsl
// Vertex shader: Compute per-patch LOD
attribute vec3 aPosition;
attribute float aPatchRelevance; // From CPU relevance realization

varying float vLodLevel;
varying vec2 vTexCoord;

void main() {
    // Map relevance (0-1) to token budget (64-400) to LOD (0-5)
    float tokenBudget = mix(64.0, 400.0, aPatchRelevance);
    vLodLevel = log2(tokenBudget / 64.0); // 0 = 64 tokens, 2.64 = 400 tokens

    vTexCoord = aPosition.xy * 0.5 + 0.5;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
```

**Fragment shader selective detail:**
```glsl
// Fragment shader: Sample appropriate LOD
precision mediump float;

varying float vLodLevel;
varying vec2 vTexCoord;

uniform sampler2D uDetailPyramid; // Mipmap pyramid

void main() {
    // Automatic LOD selection from vertex shader vLodLevel
    vec4 detail = textureLod(uDetailPyramid, vTexCoord, vLodLevel);

    // Process 13 channels efficiently
    vec3 propositional = detail.rgb;
    // ... additional channel processing

    gl_FragColor = computeRelevanceVisualization(detail);
}
```

**Mobile optimization for 13-channel processing:**

Combine channels into fewer textures:
```glsl
// Instead of 4 separate textures (13 channels):
// - Texture 1: RGBA (Propositional RGB + Perspectival A)
// - Texture 2: RGBA (Perspectival GBA + Participatory A)
// - Texture 3: RGBA (Participatory GBA + Metadata RG)

uniform mediump sampler2D uChannelPack1; // 4 channels
uniform mediump sampler2D uChannelPack2; // 4 channels
uniform highp sampler2D uChannelPack3;   // 4 channels + metadata

// Reduces texture unit usage: 3 instead of 4
// Reduces memory bandwidth: Packed access patterns
```

---

## Shader Compilation and Variants

### Compile-Time Optimization

From [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31):

**Parallel compilation:**

Don't query shader compile status until needed:
```javascript
// BAD: Synchronous compilation (blocks each shader)
gl.compileShader(vs);
if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) { /* error */ }
gl.compileShader(fs);
if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) { /* error */ }
gl.linkProgram(prog);

// GOOD: Asynchronous compilation (parallel)
gl.compileShader(vs);
gl.compileShader(fs);
gl.linkProgram(prog); // Compiles can happen in parallel
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    // Check shader logs only on link failure
    console.error(gl.getShaderInfoLog(vs));
    console.error(gl.getShaderInfoLog(fs));
}
```

**KHR_parallel_shader_compile extension:**

Non-blocking compilation status:
```javascript
const ext = gl.getExtension('KHR_parallel_shader_compile');
gl.compileShader(vs);
gl.compileShader(fs);
gl.linkProgram(prog);

// Later (e.g., next frame):
if (ext && gl.getProgramParameter(prog, ext.COMPLETION_STATUS_KHR)) {
    // Program ready, check link status
    if (gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        gl.useProgram(prog); // Start using
    }
}
```

### Shader Permutation Management

From [The Shader Permutation Problem](https://therealmjp.github.io/posts/shader-permutations-part1/) (accessed 2025-10-31):

**Permutation explosion:**

Features combine multiplicatively:
```
Features: Normal mapping (2), Shadows (2), Fog (2), Skinning (2)
Permutations: 2 × 2 × 2 × 2 = 16 shader variants
```

**Strategies:**

1. **Uber-shader with dynamic branches** (fewer variants, runtime cost):
```glsl
uniform bool useNormalMap;
uniform bool useShadows;

if (useNormalMap) { normal = texture(...); }
if (useShadows) { shadow = computeShadow(...); }
```
- Pros: 1 shader variant
- Cons: Branch divergence cost, larger shader

2. **Static permutations** (more variants, optimal runtime):
```glsl
#ifdef NORMAL_MAPPING
    normal = texture(normalMap, uv).xyz * 2.0 - 1.0;
#endif

#ifdef SHADOWS
    shadow = computeShadow(shadowPos);
#endif
```
- Pros: No runtime branches, optimal code
- Cons: Combinatorial explosion, compile time

3. **Hybrid approach** (balance):
- Static permutations for high-impact features (lighting model, material type)
- Dynamic branches for low-cost options (debug visualization, minor toggles)

---

## Profiling and Debugging Workflow

### Systematic Optimization Process

From [NVIDIA Advanced API Performance](https://developer.nvidia.com/blog/tag/advanced-api-performance/) and [Arm Performance Studio](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31):

**1. Identify Bottleneck (P3 Method):**

Peak-Performance-Percentage analysis:
- GPU Time / Theoretical Peak = Bottleneck indicator
- < 25%: Severe bottleneck
- 50-70%: Moderate optimization opportunity
- > 90%: Near optimal

**2. Categorize Bottleneck:**

Performance categories:
- **ALU-bound**: High arithmetic instruction count
- **Memory-bound**: High texture/buffer access count
- **Bandwidth-bound**: Large data transfers
- **Occupancy-bound**: Low thread count due to register pressure
- **Frontend-bound**: Draw call overhead, state changes

**3. Apply Targeted Optimization:**

| Bottleneck | Optimization Approach |
|------------|----------------------|
| ALU | Reduce instruction count, move to vertex shader, use FP16 |
| Memory | Reduce texture samples, improve cache coherency, use mipmaps |
| Bandwidth | Compressed textures, lower precision, reduce resolution |
| Occupancy | Lower register usage (FP16, simplify shader), reduce locals |
| Frontend | Batch draw calls, reduce state changes, use instancing |

**4. Measure and Iterate:**

Establish baseline:
```
Frame time: 16.7ms (60 fps target)
GPU time: 12ms
Shader X time: 4ms (33% of GPU time)
```

After optimization:
```
Shader X time: 2ms (16% of GPU time) → 2ms improvement
Frame time: 14.7ms → Bottleneck may shift
```

Re-profile to find next bottleneck (iterative process).

---

## Best Practices Summary

### Mobile Shader Optimization Checklist

From [Arm GPU Best Practices](https://developer.arm.com/documentation/101897/0301), [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices), and [GPU Optimization for GameDev](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31):

**Precision:**
- [ ] Use `mediump` as default for fragment shaders
- [ ] Explicitly use `highp` only for positions, depth, transforms
- [ ] Target > 80% FP16 usage (check with profiler)
- [ ] Use `lowp` for final color output and simple flags

**Instruction Count:**
- [ ] Move calculations from fragment to vertex shader when possible
- [ ] Use built-in functions (dot, normalize, mix) over manual implementations
- [ ] Avoid expensive transcendentals (sin, cos, sqrt) in hot paths
- [ ] Reduce live variable ranges to minimize register pressure

**Texture Access:**
- [ ] Always use mipmaps for 3D scene textures
- [ ] Use compressed formats (ASTC > ETC2/PVRTC > S3TC)
- [ ] Batch independent texture reads together
- [ ] Avoid dependent texture lookups
- [ ] Use appropriate filtering (bilinear for most, trilinear for quality)

**Branching:**
- [ ] Use uniform/static branches for feature toggles
- [ ] Minimize dynamic branching in inner loops
- [ ] Consider branchless alternatives for simple conditions
- [ ] Use shader permutations for major feature variants

**Mobile Specific:**
- [ ] Call `invalidateFramebuffer()` for temporary attachments
- [ ] Minimize overdraw (selective Z-prepass, not always)
- [ ] Target 50-60% shader occupancy minimum
- [ ] Avoid reading back framebuffer mid-frame (breaks tiling)
- [ ] Use lower resolution + upscaling for heavy effects

**Compilation:**
- [ ] Compile shaders in parallel (don't check status until link)
- [ ] Use KHR_parallel_shader_compile when available
- [ ] Warm up shader cache early (compile at load, not first use)
- [ ] Limit permutation count (< 100 variants ideally)

**Profiling:**
- [ ] Profile on target mobile hardware (not just desktop)
- [ ] Measure shader cycle cost per functional unit
- [ ] Check register usage and occupancy
- [ ] Identify stack spills (critical: should be 0 bytes)
- [ ] Compare FP16 usage percentage

---

## Tools and Resources

### Profiling Tools

**Cross-Platform:**
- [RenderDoc](https://renderdoc.org/) - Frame debugger with shader profiling
- [Compiler Explorer (Godbolt)](https://godbolt.org/) - Online shader compiler with ISA output

**Mobile:**
- [Arm Performance Studio](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio) - Frame Advisor, Graphics Analyzer
- [Arm GPU Analyzer (RGA)](https://gpuopen.com/rga/) - Offline shader analysis

**Desktop:**
- [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics) - GPU Trace, shader debugging
- [AMD Radeon GPU Profiler](https://gpuopen.com/rgp/) - Low-level optimization, wave occupancy
- [Intel GPA](https://software.intel.com/content/www/us/en/develop/tools/graphics-performance-analyzers.html) - Graphics performance analysis

**Web:**
- [Shader Playground](http://shader-playground.timjones.io/) - Multi-compiler shader analysis
- Browser DevTools (Chrome GPU profiler, Firefox Canvas debugger)

### Learning Resources

**Documentation:**
- [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31)
- [Arm GPU Best Practices Developer Guide](https://developer.arm.com/documentation/101897/0301) (accessed 2025-10-31)
- [GPU Optimization for GameDev (Gist)](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31)

**Talks and Presentations:**
- [Arm Performance Studio 2024.4 - Shader Analysis Features](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31)
- [Advanced API Performance - NVIDIA](https://developer.nvidia.com/blog/tag/advanced-api-performance/) (accessed 2025-10-31)
- [Low-Level Thinking in High-Level Shading Languages](https://www.gdcvault.com/play/1018182/Low-Level-Thinking-in-High) - Emil Persson

---

## Sources

**Web Research:**
- [Arm Performance Studio 2024.4 - Shader Analysis Features](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-performance-studio-2024-4) (accessed 2025-10-31)
- [WebGL Best Practices - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) (accessed 2025-10-31)
- [GPU Optimization for GameDev - GitHub Gist](https://gist.github.com/silvesthu/505cf0cbf284bb4b971f6834b8fec93d) (accessed 2025-10-31)
- [Arm GPU Best Practices Developer Guide](https://developer.arm.com/documentation/101897/0301) (referenced 2025-10-31)
- [NVIDIA Advanced API Performance](https://developer.nvidia.com/blog/tag/advanced-api-performance/) (referenced 2025-10-31)
- [Wonderland Engine - Profiling WebXR Applications](https://wonderlandengine.com/news/profiling-webxr-applications/) (referenced 2025-10-31)
- [The Shader Permutation Problem - Matt Pettineo](https://therealmjp.github.io/posts/shader-permutations-part1/) (referenced 2025-10-31)

**Additional References:**
- [Unity Mobile Optimization](https://discussions.unity.com) (referenced from search results)
- [PreMortem Games - Unity 6 GPU Optimization](https://premortem.games/2024/10/14/advanced-gpu-optimization-techniques-in-unity-6-insights-from-unite-2024/) (referenced from search results)
