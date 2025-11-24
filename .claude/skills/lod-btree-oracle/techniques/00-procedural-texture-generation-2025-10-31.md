# Procedural Texture Generation

**Topic**: GPU-based procedural texture synthesis using shader noise functions
**Date**: 2025-10-31
**Scope**: Fragment shader techniques, noise algorithms, real-time generation

---

## Overview

Procedural texture generation creates textures algorithmically at runtime using mathematical functions rather than loading pre-baked image files. This approach offers infinite variation, eliminates texture memory overhead, ensures seamless tiling, and enables dynamic parameter-driven appearance control. Core to procedural texturing are noise functions - smooth random functions that form the building blocks for natural-looking patterns like wood grain, marble veining, cloud formations, and terrain features.

**Key Advantages:**
- Memory efficiency (no texture storage)
- Resolution independence (infinite detail)
- Parametric control (runtime variation)
- Seamless tiling (no edge artifacts)
- Animation capability (time-varying patterns)

**Primary Applications:**
- Surface detail generation (bump maps, displacement)
- Material appearance (wood, stone, organic patterns)
- Environmental effects (clouds, fire, water)
- Terrain synthesis (height maps, biome distribution)
- Visual effects (distortion, turbulence fields)

---

## Shader-Based Texture Generation

### Fragment Shader Pipeline

Procedural textures execute in fragment shaders, generating pixel colors through mathematical evaluation rather than texture sampling. Each fragment independently computes its color based on spatial coordinates, enabling massively parallel GPU execution.

**Basic Structure:**
```glsl
// Fragment shader procedural texture
varying vec2 v_texcoord;  // UV coordinates from vertex shader
uniform float u_time;      // Animation parameter
uniform float u_scale;     // Pattern scale control

float noise(vec2 p);       // Noise function (various types)

void main() {
    // Scale UV space
    vec2 st = v_texcoord * u_scale;

    // Generate procedural value
    float pattern = noise(st);

    // Map to color
    vec3 color = vec3(pattern);

    gl_FragColor = vec4(color, 1.0);
}
```

**Coordinate Space Management:**
- UV coordinates (0-1 range) provide base positioning
- Scaling factors control pattern frequency
- Tiling achieved through modulo arithmetic (`fract()`)
- Multiple octaves combine different frequencies

### Noise Function Fundamentals

Noise functions are the foundation of procedural generation. They produce smooth, band-limited random values with specific spatial coherence properties.

**Essential Characteristics:**
1. **Repeatability**: Same input always yields same output
2. **Smoothness**: Continuous values with controlled derivatives
3. **Bounded Range**: Typically [-1, 1] or [0, 1]
4. **No Obvious Patterns**: Appears random despite being deterministic
5. **Spatial Invariance**: Frequency remains constant under translation

From [The Book of Shaders - Noise](https://thebookofshaders.com/11/) (accessed 2025-10-31):
> "The unpredictability of these textures could be called 'random,' but they don't look like the random we were playing with before. The 'real world' is such a rich and complex place! How can we approximate this variety computationally?"

**1D Noise Implementation:**
```glsl
// Simple 1D noise with smooth interpolation
float noise(float x) {
    float i = floor(x);  // Integer part
    float f = fract(x);  // Fractional part

    // Smooth interpolation curve (Hermite)
    float u = f * f * (3.0 - 2.0 * f);

    // Interpolate between integer lattice values
    return mix(rand(i), rand(i + 1.0), u);
}
```

**2D Noise Extension:**
```glsl
// 2D value noise
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners of cell
    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Bilinear interpolation
    return mix(mix(a, b, u.x),
               mix(c, d, u.x), u.y);
}
```

### Fractional Brownian Motion (fBm)

Multiple octaves of noise at different frequencies create fractal-like complexity resembling natural textures.

**Octave Accumulation:**
```glsl
float fbm(vec2 st) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    // Accumulate 6 octaves
    for(int i = 0; i < 6; i++) {
        value += amplitude * noise(st * frequency);
        frequency *= 2.0;  // Lacunarity (frequency multiplier)
        amplitude *= 0.5;  // Persistence (amplitude decay)
    }
    return value;
}
```

**Control Parameters:**
- **Octaves**: Number of noise layers (more = more detail)
- **Lacunarity**: Frequency increase per octave (typically 2.0)
- **Persistence**: Amplitude decay per octave (typically 0.5)
- **Initial Frequency**: Base pattern scale

---

## Noise Function Types

### Perlin Noise

Developed by Ken Perlin (1983, improved 2002), Perlin noise interpolates random gradients at lattice points rather than random values, producing smoother, more natural-looking patterns.

**Algorithm Overview:**
1. Subdivide space into integer lattice grid
2. Assign random gradient vectors to lattice points
3. Calculate dot product of gradients with fractional positions
4. Interpolate results using smooth curve (Hermite or quintic)

From [NVIDIA GPU Gems 2 - Chapter 26](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-26-implementing-improved-perlin-noise) (accessed 2025-10-31):
> "Perlin's improved noise algorithm meets all these requirements... It produces a repeatable pseudorandom value for each input position, has a known range (usually [-1, 1]), has band-limited spatial frequency (that is, it is smooth), doesn't show obvious repeating patterns, and its spatial frequency is invariant under translation."

**Classic Perlin Implementation:**
```glsl
// Gradient noise implementation
float perlin(vec2 P) {
    vec2 i = floor(P);
    vec2 f = fract(P);

    // Quintic interpolation curve (improved Perlin)
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Gradient vectors at corners
    float a = dot(grad(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(grad(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(grad(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(grad(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
```

**Improved Interpolation Curve:**

From GPU Gems 2:
> "An improvement by Perlin to his original non-simplex noise is the replacement of the cubic Hermite curve (f(x) = 3x² - 2x³) with a quintic interpolation curve (f(x) = 6x⁵ - 15x⁴ + 10x³). This makes both ends of the curve more 'flat' so each border gracefully stitches with the next one. In other words, you get a more continuous transition between the cells."

**Gradient Generation:**
```glsl
// Permutation-based gradient lookup
vec2 grad(vec2 p) {
    // Hash function to select gradient
    int idx = int(hash(p)) & 15;

    // 16 possible gradient directions
    float u = idx < 8 ? 1.0 : -1.0;
    float v = (idx & 7) < 4 ? 1.0 : -1.0;

    return vec2(u, v);
}
```

**GPU-Optimized Perlin:**

From GPU Gems 2 on optimization:
> "The reference implementation uses six recursive lookups into the permutation table. Instead we can precalculate a 256×256-pixel RGBA 2D texture that contains four values in each texel, and use a single 2D lookup... The unoptimized implementation compiles to 81 Pixel Shader 2.0 instructions, including 22 texture lookups. After optimization, it is 53 instructions, only nine of which are texture lookups."

### Simplex Noise

Ken Perlin's 2001 improvement reduces computational complexity by using simplex grids (triangular in 2D) instead of hypercubes.

**Key Improvements:**
- Lower computational cost (N+1 corners vs 2^N corners)
- Better visual isotropy (less directional artifacts)
- Scales better to higher dimensions
- Continuous gradients everywhere

From The Book of Shaders:
> "Ken smartly noticed that although the obvious choice for a space-filling shape is a square, the simplest shape in 2D is the equilateral triangle. So he started by replacing the squared grid for a simplex grid of equilateral triangles... The simplex shape for N dimensions is a shape with N + 1 corners. In other words one fewer corner to compute in 2D, 4 fewer corners in 3D and 11 fewer corners in 4D! That's a huge improvement!"

**Simplex Grid Structure:**
- 2D: Equilateral triangles (3 corners vs 4 for square)
- 3D: Tetrahedra (4 corners vs 8 for cube)
- 4D: 4-simplices (5 corners vs 16 for hypercube)

**2D Simplex Implementation:**
```glsl
// Simplex noise (Ian McEwan implementation)
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0

    // Skew input space to determine simplex cell
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);

    // Determine which simplex we're in
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    // Offsets for other corners
    vec2 x1 = x0.xy + C.xx - i1;
    vec2 x2 = x0.xy + C.zz;

    // Permutations and gradient calculations
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));

    // Gradients and contributions
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
    m = m * m * m * m;

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    return 130.0 * dot(m, g.x * a0 + g.y * h);
}
```

### Worley Noise (Cellular Noise)

Created by Steven Worley (1996), Worley noise generates cellular patterns by computing distances to randomly distributed feature points.

From [The Book of Shaders - Cellular Noise](https://thebookofshaders.com/12/) (accessed 2025-10-31):
> "Steven Worley wrote a paper called 'A Cellular Texture Basis Function'. In it, he describes a procedural texturing technique now extensively used by the graphics community... Cellular Noise is based on distance fields, the distance to the closest one of a set of feature points."

**Algorithm Concept:**
1. Divide space into grid cells
2. Place random feature point in each cell
3. For each pixel, find distance to nearest feature point
4. Use distance to generate pattern

**Basic Distance Field:**
```glsl
float worley(vec2 st) {
    // Scale and tile
    st *= 3.0;
    vec2 i_st = floor(st);
    vec2 f_st = fract(st);

    float min_dist = 1.0;

    // Check 3×3 neighborhood
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));

            // Random point in neighbor cell
            vec2 point = random2(i_st + neighbor);

            // Distance calculation
            vec2 diff = neighbor + point - f_st;
            float dist = length(diff);

            min_dist = min(min_dist, dist);
        }
    }

    return min_dist;
}
```

**Distance Metric Variations:**
- **Euclidean**: `length(diff)` - circular cells
- **Manhattan**: `abs(diff.x) + abs(diff.y)` - square cells
- **Chebyshev**: `max(abs(diff.x), abs(diff.y))` - cross patterns
- **Minkowski**: `pow(pow(abs(diff.x), p) + pow(abs(diff.y), p), 1/p)` - variable shapes

**Voronoi Diagrams:**

Worley noise naturally generates Voronoi diagrams (cellular partitions) by tracking which feature point is closest:

```glsl
vec2 voronoi(vec2 st) {
    st *= 3.0;
    vec2 i_st = floor(st);
    vec2 f_st = fract(st);

    float min_dist = 1.0;
    vec2 min_point;

    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = random2(i_st + neighbor);
            vec2 diff = neighbor + point - f_st;
            float dist = length(diff);

            if(dist < min_dist) {
                min_dist = dist;
                min_point = point;  // Store closest point
            }
        }
    }

    return min_point;  // Returns unique ID per cell
}
```

From The Book of Shaders:
> "This algorithm can also be interpreted from the perspective of the points and not the pixels. In that case it can be described as: each point grows until it finds the growing area from another point. This mirrors some of the growth rules in nature. Living forms are shaped by this tension between an inner force to expand and grow, and limitations by outside forces."

**Advanced Worley Variations:**
- **F1**: Distance to closest point (standard)
- **F2**: Distance to second-closest point
- **F2 - F1**: Cell border highlighting
- **Crackle**: `1.0 - F1 * F2` for cracked patterns

### Value Noise

Simplest noise type - interpolates random scalar values at lattice points rather than gradients.

**Characteristics:**
- Fast computation (no gradient calculations)
- More "blocky" appearance than gradient noise
- Useful for simple patterns and quick prototypes

```glsl
float valueNoise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Smooth interpolation
    f = f * f * (3.0 - 2.0 * f);

    // Random values at corners
    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
```

---

## GPU Procedural Generation Techniques

### Domain Warping

Distorting noise input coordinates creates complex, organic patterns.

**Basic Warping:**
```glsl
vec2 q = vec2(fbm(st + vec2(0.0, 0.0)),
              fbm(st + vec2(5.2, 1.3)));

vec2 r = vec2(fbm(st + 4.0 * q + vec2(1.7, 9.2)),
              fbm(st + 4.0 * q + vec2(8.3, 2.8)));

float pattern = fbm(st + 4.0 * r);
```

**Turbulence (Absolute Value):**
```glsl
float turbulence(vec2 st) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;

    for(int i = 0; i < 6; i++) {
        value += amplitude * abs(noise(st * frequency));
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}
```

### Ridged Multifractal

Inverted noise creates ridge-like features useful for terrain and mountains.

```glsl
float ridgedMF(vec2 st) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for(int i = 0; i < 6; i++) {
        float n = noise(st * frequency);
        n = 1.0 - abs(n);  // Invert and absolute
        n = n * n;         // Sharpen ridges
        value += n * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}
```

### Rotation and Transformation

Rotating noise space creates directional patterns.

```glsl
mat2 rotate2d(float angle) {
    return mat2(cos(angle), -sin(angle),
                sin(angle),  cos(angle));
}

float pattern = noise(rotate2d(noise(st) * 6.28) * st);
```

### Pattern Combination

Layering multiple noise types creates rich textures.

```glsl
// Wood grain pattern
float wood(vec2 st) {
    // Base rings
    float rings = fract(st.x * 10.0 + noise(st * 5.0) * 0.5);

    // Add grain texture
    float grain = noise(st * 50.0) * 0.1;

    // Combine
    return rings + grain;
}

// Marble pattern
float marble(vec2 st) {
    vec2 q = vec2(fbm(st), fbm(st + vec2(5.2, 1.3)));
    return fbm(st + 4.0 * q);
}
```

### Time-Based Animation

Animating noise creates dynamic effects.

```glsl
uniform float u_time;

// Flowing clouds
float clouds(vec2 st) {
    vec2 flow = st + vec2(u_time * 0.1, u_time * 0.05);
    return fbm(flow * 3.0);
}

// Pulsing energy
float energy(vec2 st) {
    float t = u_time * 2.0;
    return noise(vec3(st * 5.0, t)) * 0.5 + 0.5;
}
```

---

## Real-Time Texture Synthesis

### Performance Optimization

**Texture Lookups vs Computation:**

From GPU Gems 2:
> "Procedural noise is typically implemented in today's shaders using precomputed 3D textures. Implementing noise directly in the pixel shader has several advantages: It requires less texture memory, the period is large, results match existing CPU implementations exactly, allows four-dimensional noise, and interpolation is higher quality... The obvious disadvantage is computational expense."

**Optimization Strategies:**
1. **Precompute Permutation Tables**: Store in textures instead of computing
2. **Reduce Octaves**: Use fewer fBm iterations for distant surfaces
3. **LOD-Based Complexity**: Simpler noise for lower detail levels
4. **Caching Results**: Store in render targets for reuse
5. **Vectorization**: Process multiple noise calls in parallel

**Example LOD System:**
```glsl
float lodNoise(vec2 st, float lod) {
    int octaves = int(mix(2.0, 8.0, 1.0 - lod));

    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for(int i = 0; i < octaves; i++) {
        value += amplitude * noise(st * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}
```

### Texture Coordinate Strategies

**Seamless Tiling:**
```glsl
// Periodic noise (repeats at interval)
float tiledNoise(vec2 st, float period) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Wrap coordinates
    i = mod(i, period);

    // Generate noise with wrapping
    return periodicNoise(i, f, period);
}
```

**Triplanar Mapping:**
```glsl
// Avoid UV distortion on arbitrary geometry
vec3 triplanarNoise(vec3 pos, vec3 normal) {
    vec3 blending = abs(normal);
    blending = normalize(max(blending, 0.00001));
    float b = blending.x + blending.y + blending.z;
    blending /= b;

    float xaxis = noise(pos.yz);
    float yaxis = noise(pos.xz);
    float zaxis = noise(pos.xy);

    return xaxis * blending.x +
           yaxis * blending.y +
           zaxis * blending.z;
}
```

### Multi-Channel Generation

Generate multiple texture maps simultaneously for efficiency.

```glsl
// Combined material properties
struct MaterialMaps {
    float albedo;
    float roughness;
    float height;
    vec3 normal;
};

MaterialMaps generateMaterial(vec2 st) {
    MaterialMaps maps;

    // Use different octaves for variety
    maps.albedo = fbm(st * 2.0);
    maps.roughness = fbm(st * 5.0);
    maps.height = fbm(st * 3.0);

    // Derive normal from height
    float h = maps.height;
    float hx = fbm((st + vec2(0.01, 0.0)) * 3.0);
    float hy = fbm((st + vec2(0.0, 0.01)) * 3.0);
    maps.normal = normalize(vec3(h - hx, h - hy, 0.1));

    return maps;
}
```

---

## ARR-COC Dynamic Texture Patterns

### Query-Aware Texture Generation

Procedural textures can adapt to vision-language model queries by modulating noise parameters based on semantic relevance.

**Relevance-Driven Detail:**
```glsl
uniform float u_relevance;  // From ARR-COC relevance scores

float adaptiveTexture(vec2 st) {
    // More octaves for high-relevance regions
    int octaves = int(mix(2.0, 8.0, u_relevance));

    // Increase frequency in important areas
    float scale = mix(1.0, 4.0, u_relevance);

    return fbm(st * scale, octaves);
}
```

**Semantic Pattern Modulation:**
```glsl
// Different noise types based on query context
float semanticNoise(vec2 st, int queryType) {
    if(queryType == TEXTURE_ROUGH) {
        return worley(st);  // Cellular for roughness
    } else if(queryType == TEXTURE_SMOOTH) {
        return perlin(st);  // Smooth gradients
    } else if(queryType == TEXTURE_ORGANIC) {
        return fbm(st);     // Fractal complexity
    }
    return noise(st);
}
```

### 32×32 Patch Texture Synthesis

Generate unique procedural textures for each vision patch based on LOD allocation.

**Patch-Specific Patterns:**
```glsl
// Unique seed per patch from grid coordinates
vec2 patchSeed = vec2(patchX, patchY) / 32.0;

// Generate patch-specific variation
float patchNoise(vec2 uv, vec2 seed) {
    // Offset noise space based on patch location
    vec2 offset = random2(seed) * 100.0;

    // Generate noise with unique offset
    return noise(uv + offset);
}
```

**LOD-Adaptive Texture Detail:**
```glsl
// Tokens allocated: 64 (low) to 400 (high)
float lodMultiplier = float(tokens - 64) / (400.0 - 64.0);

// Adjust pattern complexity
int octaves = int(mix(1.0, 6.0, lodMultiplier));
float frequency = mix(0.5, 4.0, lodMultiplier);

float texture = fbm(st * frequency, octaves);
```

### Multi-Scale Texture Hierarchies

Generate consistent textures across multiple LOD levels.

**Hierarchical Noise:**
```glsl
// Base pattern same across LODs
float basePattern = noise(st * baseFreq);

// Add detail based on LOD level
float detail = 0.0;
if(lod >= 2) detail += noise(st * baseFreq * 2.0) * 0.5;
if(lod >= 3) detail += noise(st * baseFreq * 4.0) * 0.25;
if(lod >= 4) detail += noise(st * baseFreq * 8.0) * 0.125;

return basePattern + detail;
```

**Mipmap-Style Blending:**
```glsl
// Blend between LOD levels
float lod_fract = fract(targetLOD);
float texLow = proceduralTexture(st, floor(targetLOD));
float texHigh = proceduralTexture(st, ceil(targetLOD));
return mix(texLow, texHigh, lod_fract);
```

---

## Material Pattern Applications

### Natural Materials

**Wood Grain:**
```glsl
float woodPattern(vec2 st) {
    // Radial rings
    float radius = length(st - 0.5);
    float rings = sin(radius * 50.0 + noise(st * 5.0) * 3.0);

    // Grain direction
    float angle = atan(st.y - 0.5, st.x - 0.5);
    float grain = noise(vec2(angle * 10.0, radius * 20.0));

    return rings * 0.7 + grain * 0.3;
}
```

**Marble Veining:**
```glsl
float marblePattern(vec2 st) {
    vec2 q = vec2(fbm(st), fbm(st + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(st + 4.0 * q), fbm(st + 4.0 * q + vec2(1.7, 9.2)));

    float pattern = fbm(st + 4.0 * r);
    return sin(st.y * 10.0 + pattern * 10.0) * 0.5 + 0.5;
}
```

**Stone Texture:**
```glsl
float stonePattern(vec2 st) {
    // Base roughness
    float base = worley(st * 3.0);

    // Surface detail
    float detail = fbm(st * 10.0) * 0.3;

    // Cracks
    float cracks = smoothstep(0.1, 0.15, worley(st * 8.0));

    return base * 0.6 + detail + cracks * 0.2;
}
```

### Environmental Effects

**Cloud Formation:**
```glsl
float clouds(vec2 st, float time) {
    vec2 flow = st + vec2(time * 0.05, 0.0);

    float clouds = fbm(flow * 2.0);
    clouds = smoothstep(0.3, 1.0, clouds);

    // Add wisps
    clouds += fbm(flow * 8.0) * 0.2;

    return clouds;
}
```

**Water Ripples:**
```glsl
float waterPattern(vec2 st, float time) {
    vec2 wave1 = st + vec2(time * 0.1, time * 0.15);
    vec2 wave2 = st - vec2(time * 0.08, time * 0.12);

    float ripples = sin(noise(wave1 * 10.0) * 6.28) * 0.5;
    ripples += sin(noise(wave2 * 8.0) * 6.28) * 0.5;

    return ripples * 0.5 + 0.5;
}
```

**Fire Effect:**
```glsl
float firePattern(vec2 st, float time) {
    // Rising flame
    vec2 flow = st + vec2(0.0, -time * 0.5);

    // Turbulent noise
    float flame = turbulence(flow * 3.0);

    // Fade top
    flame *= 1.0 - st.y;

    // Color intensity
    return pow(flame, 2.0);
}
```

---

## Implementation Considerations

### GLSL Noise Libraries

Several optimized implementations exist for production use:

**Popular Libraries:**
- **webgl-noise** (Ashima Arts): Optimized Simplex implementations
- **lygia**: Comprehensive generative functions library
- **stegu noise**: Stefan Gustavson's reference implementations
- **Book of Shaders patterns**: Educational examples

**Integration Example:**
```glsl
// Include library
#include "noise/simplex.glsl"

// Use in shader
float pattern = snoise(st * 5.0);
```

### Permutation Table Management

From GPU Gems 2:
> "Pixel shaders do not currently support indexing into constant memory, so instead we store these tables in textures and use texture lookups to access them. The texture addressing is set to wrap (or repeat) mode, so we don't have to worry about extending the tables to avoid indexing past the end of the array."

**Texture-Based Permutation:**
```glsl
uniform sampler2D u_permTexture;
uniform sampler2D u_gradTexture;

float perm(float x) {
    return texture2D(u_permTexture, vec2(x / 256.0, 0.5)).r * 256.0;
}

vec2 grad(float x) {
    return texture2D(u_gradTexture, vec2(x / 16.0, 0.5)).xy;
}
```

### Quality vs Performance Trade-offs

**Performance Metrics (approximate):**
- Value Noise: ~20-30 instructions
- Perlin Noise: ~40-60 instructions
- Simplex Noise: ~50-70 instructions
- Worley Noise: ~60-100 instructions (9×9 search)

**Optimization Strategies:**
1. **Reduce octaves** for fBm (2-4 instead of 6-8)
2. **Use value noise** for background details
3. **Precompute** static patterns in textures
4. **LOD switching** based on distance/importance
5. **Simplify** for mobile/low-end GPUs

### Debugging Techniques

**Visualize Noise Components:**
```glsl
// Show individual octaves
if(debugMode == 1) {
    color = vec3(noise(st * freq1));  // Octave 1
} else if(debugMode == 2) {
    color = vec3(noise(st * freq2));  // Octave 2
}

// Show blend weights
color = vec3(u_relevance);  // Relevance visualization
```

---

## Sources

**Core Documentation:**
- [The Book of Shaders - Noise](https://thebookofshaders.com/11/) - Patricio Gonzalez Vivo, accessed 2025-10-31
- [The Book of Shaders - Cellular Noise](https://thebookofshaders.com/12/) - Worley/Voronoi implementation, accessed 2025-10-31
- [NVIDIA GPU Gems 2 - Chapter 26: Implementing Improved Perlin Noise](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-26-implementing-improved-perlin-noise) - Simon Green, accessed 2025-10-31

**Research Papers:**
- Perlin, Ken (2002). "Improving Noise." ACM Transactions on Graphics (Proceedings of SIGGRAPH 2002)
- Worley, Steven (1996). "A Cellular Texture Basis Function"
- Gustavson, Stefan (2005). "Simplex noise demystified"

**Additional References:**
- [Inigo Quilez - Articles](http://www.iquilezles.org/www/index.htm) - Voronoi techniques, voro-noise
- [LYGIA Generative Functions](https://lygia.xyz/generative) - Production shader library

---

## Related Topics

- **[Texture Sampling and Filtering](00-texture-sampling-filtering-2025-10-31.md)** - Mipmap LOD selection for procedural textures
- **[Texture Mapping Techniques](00-texture-mapping-techniques-2025-10-31.md)** - UV coordinate strategies for noise functions
- **Visual Embedding Compression** - Query-aware texture detail allocation
- **GPU Performance Optimization** - Balancing procedural computation with memory access

---

**Last Updated**: 2025-10-31
**Relevance**: GPU texture synthesis, shader programming, real-time rendering, ARR-COC visual processing
