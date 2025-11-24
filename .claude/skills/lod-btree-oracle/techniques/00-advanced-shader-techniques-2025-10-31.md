# Advanced Shader Techniques

## Overview

This document covers advanced shader programming techniques including vertex displacement, parallax mapping, geometry shaders, tessellation, and screen-space effects. These techniques enable sophisticated real-time rendering effects while maintaining performance through GPU optimization.

**Core Categories:**
- **Vertex Displacement**: Heightmap-based geometry modification
- **Parallax Mapping**: Depth illusion through texture coordinate manipulation
- **Geometry Shaders**: Dynamic primitive generation and transformation
- **Tessellation**: GPU-driven subdivision for adaptive level of detail
- **Screen-Space Effects**: Post-processing techniques (SSAO, SSR, bloom)

## Vertex Displacement Techniques

### Heightmap-Based Displacement

Vertex displacement uses shaders to alter 3D model geometry via normal extrusion, pushing vertices outwards based on heightmap texture data.

From [Interactive Map Shader: Vertex Displacement](https://www.alanzucconi.com/2019/07/03/interactive-map-01/) (accessed 2025-10-31):

**Basic Normal Extrusion:**
```glsl
// Surface shader vertex modifier
void vert(inout appdata_base v)
{
    v.vertex.xyz += v.normal * _Amount;
}
```

**Heightmap-Modulated Displacement:**
```glsl
void vert(inout appdata_base v)
{
    float height = tex2Dlod(_HeightMap, float4(v.texcoord.xy, 0, 0)).r;
    v.vertex.xyz += v.normal * height * _Amount;
}
```

**Key Differences from CPU Displacement:**
- Executes in parallel on GPU (vs sequential CPU)
- Does not modify mesh collider (requires separate update)
- Renders with original vertex buffer (displacement visual only)
- Can use tex2Dlod for mipmap-aware sampling

**Flat Plane Optimization:**
```glsl
// For flat surfaces, normal is constant
float3 normal = float3(0, 1, 0);  // Model-space up
vertex.xyz += normal * height * _Amount;
```

### Procedural Displacement

From [Vertex Displacement - Advanced Materials](https://www.youtube.com/watch?v=sYLbdO3VB8c) (accessed 2025-10-31):

**Noise-Based Displacement:**
- Perlin/Simplex noise for organic surfaces
- Worley noise for cellular patterns
- Fractal noise for terrain detail layers
- Time-based animation for dynamic effects

**Use Cases:**
- Ocean wave simulation
- Terrain generation from noise
- Animated cloth/flags
- Particle deformation fields

## Parallax Mapping Family

### Basic Parallax Mapping

From [Parallax Mapping - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Parallax-Mapping) (accessed 2025-10-31):

Parallax mapping creates depth illusion by offsetting texture coordinates based on view direction and heightmap, without adding geometry.

**Core Algorithm:**
```glsl
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{
    float height = texture(depthMap, texCoords).r;
    vec2 p = viewDir.xy / viewDir.z * (height * height_scale);
    return texCoords - p;
}
```

**Why Division by viewDir.z:**
- viewDir is normalized, z ∈ [0.0, 1.0]
- When parallel to surface: z ≈ 0, division creates larger offset
- When perpendicular: z ≈ 1, division creates smaller offset
- Adjusts offset scale based on viewing angle automatically

**Heightmap vs Depthmap:**
- Heightmap: white = raised, black = low (add to position)
- Depthmap: white = deep, black = surface (subtract from position)
- Depthmap more intuitive for "carving" into surfaces

### Steep Parallax Mapping

Improves accuracy by layering depth samples to find intersection point.

From [Parallax Mapping - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Parallax-Mapping) (accessed 2025-10-31):

**Layer-Based Algorithm:**
```glsl
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{
    // Step 1: Define constants
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers,
                         max(dot(vec3(0.0, 0.0, 1.0), viewDir), 0.0));

    // Step 2: Calculate layer depth and offset per layer
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 P = viewDir.xy * height_scale;
    vec2 deltaTexCoords = P / numLayers;

    // Step 3: Iterate through layers
    vec2 currentTexCoords = texCoords;
    float currentDepthMapValue = texture(depthMap, currentTexCoords).r;

    while(currentLayerDepth < currentDepthMapValue)
    {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = texture(depthMap, currentTexCoords).r;
        currentLayerDepth += layerDepth;
    }

    return currentTexCoords;
}
```

**Adaptive Sampling:**
- Viewing perpendicular: fewer layers (8-10)
- Viewing at angle: more layers (32+)
- Balances quality vs performance dynamically

**Limitations:**
- Visible layer stepping artifacts
- Requires blur/interpolation for smoothness

### Parallax Occlusion Mapping (POM)

From [Parallax Mapping - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Parallax-Mapping) (accessed 2025-10-31):

Most accurate parallax technique using linear interpolation between depth layers.

**Interpolation Step:**
```glsl
// After steep parallax loop, interpolate between layers
vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

// Depth distances from layer boundaries
float afterDepth  = currentDepthMapValue - currentLayerDepth;
float beforeDepth = texture(depthMap, prevTexCoords).r -
                   currentLayerDepth + layerDepth;

// Linear interpolation weight
float weight = afterDepth / (afterDepth - beforeDepth);
vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);
```

**Quality vs Performance:**
- Near photo-realistic depth at steep angles
- 8-32 samples typically sufficient
- Slight blur reduces layer artifacts
- Industry standard for real-time depth illusion

**Edge Artifact Fix:**
```glsl
// Discard fragments sampling outside [0,1] range
texCoords = ParallaxMapping(fs_in.TexCoords, viewDir);
if(texCoords.x > 1.0 || texCoords.y > 1.0 ||
   texCoords.x < 0.0 || texCoords.y < 0.0)
    discard;
```

### Parallax Mapping Best Practices

From [Parallax Shift - FJORD.STYLE](https://fjord.style/parallax-shift) (accessed 2025-10-31):

**Spacing Options:**
- `equal_spacing`: Uniform subdivision (simple, predictable)
- `fractional_odd_spacing`: Odd subdivisions with long/short segments
- `fractional_even_spacing`: Even subdivisions with varied lengths

**Common Issues:**
- Breaks down at grazing angles (use angle-based layer count)
- Incorrect steep height changes (POM fixes this)
- Self-shadowing artifacts (add bias parameter)
- Border sampling errors (clamp or discard)

**Typical Parameters:**
- height_scale: 0.05 - 0.1 (material dependent)
- Min layers: 8-10
- Max layers: 32-64
- Bias: 0.001 - 0.025

## Geometry Shaders

From [Geometry Shader - LearnOpenGL](https://learnopengl.com/Advanced-OpenGL/Geometry-Shader) (accessed 2025-10-31):

Geometry shaders sit between vertex and fragment shaders, transforming primitives and generating new geometry on-the-fly.

### Pipeline Position

```
Vertex Shader → [Tessellation Control] → [Tessellation Primitive Gen] →
[Tessellation Evaluation] → Geometry Shader → Rasterization → Fragment Shader
```

### Basic Geometry Shader Structure

**Input/Output Layout:**
```glsl
#version 330 core
layout (points) in;                         // Input primitive type
layout (line_strip, max_vertices = 2) out;  // Output type + max vertices

void main() {
    gl_Position = gl_in[0].gl_Position + vec4(-0.1, 0.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4(0.1, 0.0, 0.0, 0.0);
    EmitVertex();

    EndPrimitive();
}
```

**Input Primitive Types:**
- `points`: GL_POINTS (1 vertex)
- `lines`: GL_LINES, GL_LINE_STRIP (2 vertices)
- `lines_adjacency`: GL_LINES_ADJACENCY (4 vertices)
- `triangles`: GL_TRIANGLES, GL_TRIANGLE_STRIP (3 vertices)
- `triangles_adjacency`: GL_TRIANGLES_ADJACENCY (6 vertices)

**Output Primitive Types:**
- `points`: Individual points
- `line_strip`: Connected line segments
- `triangle_strip`: Connected triangles (N-2 triangles from N vertices)

### Built-in Variables

```glsl
in gl_Vertex {
    vec4  gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
} gl_in[];  // Array sized by input primitive vertex count
```

### Practical Applications

**Normal Visualization:**
```glsl
// Visualize surface normals as lines
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT {
    vec3 normal;
} gs_in[];

const float MAGNITUDE = 0.4;
uniform mat4 projection;

void GenerateLine(int index)
{
    gl_Position = projection * gl_in[index].gl_Position;
    EmitVertex();
    gl_Position = projection * (gl_in[index].gl_Position +
                               vec4(gs_in[index].normal, 0.0) * MAGNITUDE);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    GenerateLine(0); // First vertex normal
    GenerateLine(1); // Second vertex normal
    GenerateLine(2); // Third vertex normal
}
```

**Dynamic Shape Generation:**
From [Geometry Shader - LearnOpenGL](https://learnopengl.com/Advanced-OpenGL/Geometry-Shader) (accessed 2025-10-31):

```glsl
// Generate house from single point
layout (points) in;
layout (triangle_strip, max_vertices = 5) out;

void build_house(vec4 position)
{
    gl_Position = position + vec4(-0.2, -0.2, 0.0, 0.0);  // bottom-left
    EmitVertex();
    gl_Position = position + vec4( 0.2, -0.2, 0.0, 0.0);  // bottom-right
    EmitVertex();
    gl_Position = position + vec4(-0.2,  0.2, 0.0, 0.0);  // top-left
    EmitVertex();
    gl_Position = position + vec4( 0.2,  0.2, 0.0, 0.0);  // top-right
    EmitVertex();
    gl_Position = position + vec4( 0.0,  0.4, 0.0, 0.0);  // roof peak
    EmitVertex();
    EndPrimitive();
}
```

**Use Cases:**
- Point sprites (particles, vegetation)
- Fur/grass rendering (extruded quads from surface)
- Wireframe rendering (lines from triangles)
- Shadow volume generation
- Procedural detail geometry

### Performance Considerations

From [The performance of tessellation and geometry shaders](https://www.reddit.com/r/vulkan/comments/1flx96k/) (accessed 2025-10-31):

**Geometry Shader Drawbacks:**
- Serialized execution (not fully parallel)
- Limited output primitive count
- High register pressure
- Slower than tessellation for similar tasks

**When to Use:**
- Simple per-primitive operations
- Normal visualization (debugging)
- Generating few additional primitives
- Effects requiring primitive manipulation

**Alternatives:**
- Tessellation shaders (faster for subdivision)
- Compute shaders (arbitrary geometry generation)
- Mesh shaders (modern replacement)

## Tessellation Shaders

From [Tessellation - LearnOpenGL Guest Articles](https://learnopengl.com/Guest-Articles/2021/Tessellation/Tessellation) (accessed 2025-10-31):

Tessellation shaders dynamically subdivide patches on GPU for adaptive level of detail.

### Three-Stage Pipeline

**1. Tessellation Control Shader (TCS):**
- Determines tessellation levels per patch
- Passes vertex data through
- Sets inner/outer subdivision factors

**2. Tessellation Primitive Generator:**
- Fixed-function stage (no shader code)
- Generates intermediate points
- Creates abstract patch coordinates (u,v) in [0,1]

**3. Tessellation Evaluation Shader (TES):**
- Evaluates each generated point
- Transforms from patch space to world space
- Applies displacement/heightmap
- Required stage (TCS optional)

### CPU Setup

```cpp
// Specify vertices per patch (must match TCS)
glPatchParameteri(GL_PATCH_VERTICES, 4);  // Quad patches

// Draw patches
glBindVertexArray(terrainVAO);
glDrawArrays(GL_PATCHES, 0, 4 * rez * rez);
```

### Tessellation Control Shader

```glsl
#version 410 core
layout (vertices=4) out;  // Must match glPatchParameteri

in vec2 TexCoord[];
out vec2 TextureCoord[];

void main()
{
    // Pass through vertex data
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    TextureCoord[gl_InvocationID] = TexCoord[gl_InvocationID];

    // Set tessellation levels (only once per patch)
    if (gl_InvocationID == 0)
    {
        // Outer levels (one per edge)
        gl_TessLevelOuter[0] = 16;
        gl_TessLevelOuter[1] = 16;
        gl_TessLevelOuter[2] = 16;
        gl_TessLevelOuter[3] = 16;

        // Inner levels (interior subdivision)
        gl_TessLevelInner[0] = 16;
        gl_TessLevelInner[1] = 16;
    }
}
```

**Tessellation Level Mapping:**
- Outer[0]: Bottom edge (vertices 0-1)
- Outer[1]: Right edge (vertices 1-3)
- Outer[2]: Top edge (vertices 2-3)
- Outer[3]: Left edge (vertices 0-2)
- Inner[0]: Horizontal subdivision
- Inner[1]: Vertical subdivision

### Tessellation Evaluation Shader

```glsl
#version 410 core
layout (quads, fractional_odd_spacing, ccw) in;

uniform sampler2D heightMap;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec2 TextureCoord[];
out float Height;

void main()
{
    // Patch coordinates (0-1 for each dimension)
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // Bilinear interpolation of texture coordinates
    vec2 t00 = TextureCoord[0];
    vec2 t01 = TextureCoord[1];
    vec2 t10 = TextureCoord[2];
    vec2 t11 = TextureCoord[3];

    vec2 t0 = (t01 - t00) * u + t00;
    vec2 t1 = (t11 - t10) * u + t10;
    vec2 texCoord = (t1 - t0) * v + t0;

    // Sample height from heightmap
    Height = texture(heightMap, texCoord).y * 64.0 - 16.0;

    // Bilinear interpolation of vertex positions
    vec4 p00 = gl_in[0].gl_Position;
    vec4 p01 = gl_in[1].gl_Position;
    vec4 p10 = gl_in[2].gl_Position;
    vec4 p11 = gl_in[3].gl_Position;

    vec4 p0 = (p01 - p00) * u + p00;
    vec4 p1 = (p11 - p10) * u + p10;
    vec4 p = (p1 - p0) * v + p0;

    // Compute normal and displace
    vec4 uVec = p01 - p00;
    vec4 vVec = p10 - p00;
    vec4 normal = normalize(vec4(cross(vVec.xyz, uVec.xyz), 0));

    p += normal * Height;

    // Output transformed position
    gl_Position = projection * view * model * p;
}
```

**Layout Qualifiers:**
- Primitive type: `quads`, `triangles`, `isolines`
- Spacing: `equal_spacing`, `fractional_odd_spacing`, `fractional_even_spacing`
- Winding: `ccw` (counter-clockwise), `cw` (clockwise)

### Dynamic Level of Detail

From [Tessellation - LearnOpenGL Guest Articles](https://learnopengl.com/Guest-Articles/2021/Tessellation/Tessellation) (accessed 2025-10-31):

Distance-based tessellation adapts subdivision based on camera proximity.

```glsl
// In TCS, only invocation 0 sets levels
if(gl_InvocationID == 0)
{
    // Constants
    const int MIN_TESS_LEVEL = 4;
    const int MAX_TESS_LEVEL = 64;
    const float MIN_DISTANCE = 20;
    const float MAX_DISTANCE = 800;

    // Transform vertices to eye space
    vec4 eyeSpacePos00 = view * model * gl_in[0].gl_Position;
    vec4 eyeSpacePos01 = view * model * gl_in[1].gl_Position;
    vec4 eyeSpacePos10 = view * model * gl_in[2].gl_Position;
    vec4 eyeSpacePos11 = view * model * gl_in[3].gl_Position;

    // Normalized distance [0,1]
    float distance00 = clamp((abs(eyeSpacePos00.z) - MIN_DISTANCE) /
                            (MAX_DISTANCE - MIN_DISTANCE), 0.0, 1.0);
    float distance01 = clamp((abs(eyeSpacePos01.z) - MIN_DISTANCE) /
                            (MAX_DISTANCE - MIN_DISTANCE), 0.0, 1.0);
    float distance10 = clamp((abs(eyeSpacePos10.z) - MIN_DISTANCE) /
                            (MAX_DISTANCE - MIN_DISTANCE), 0.0, 1.0);
    float distance11 = clamp((abs(eyeSpacePos11.z) - MIN_DISTANCE) /
                            (MAX_DISTANCE - MIN_DISTANCE), 0.0, 1.0);

    // Interpolate tessellation levels (use closer vertex per edge)
    float tessLevel0 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(distance10, distance00));
    float tessLevel1 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(distance00, distance01));
    float tessLevel2 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(distance01, distance11));
    float tessLevel3 = mix(MAX_TESS_LEVEL, MIN_TESS_LEVEL, min(distance11, distance10));

    gl_TessLevelOuter[0] = tessLevel0;
    gl_TessLevelOuter[1] = tessLevel1;
    gl_TessLevelOuter[2] = tessLevel2;
    gl_TessLevelOuter[3] = tessLevel3;

    gl_TessLevelInner[0] = max(tessLevel1, tessLevel3);
    gl_TessLevelInner[1] = max(tessLevel0, tessLevel2);
}
```

**LOD Strategies:**
- Distance-based (as above)
- Screen-space size (project to screen, measure area)
- Roughness-based (higher detail where terrain is complex)
- Silhouette-based (more detail at edges)
- Hybrid (combine multiple factors)

### Tessellation vs Geometry Shaders

**Tessellation Advantages:**
- Parallel execution (much faster)
- Adaptive detail levels
- Efficient for terrain/surfaces
- Hardware-optimized fixed-function stage

**When to Use Tessellation:**
- Terrain rendering
- Displacement mapping
- Curved surface subdivision
- Adaptive LOD systems
- Water/cloth simulation

## Screen-Space Effects

Screen-space effects operate on rendered framebuffers rather than geometry, enabling efficient post-processing.

### Screen-Space Ambient Occlusion (SSAO)

From [SSAO - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/SSAO) (accessed 2025-10-31):

SSAO approximates ambient occlusion by sampling depth buffer in screen space, darkening occluded areas.

**Core Concept:**
- Sample depth around each fragment in hemisphere
- Count samples inside geometry
- Occluded fragments receive less ambient light

**G-Buffer Requirements:**
```glsl
// Geometry pass outputs
layout (location = 0) out vec4 gPosition;  // View-space position
layout (location = 1) out vec3 gNormal;    // View-space normal
layout (location = 2) out vec4 gAlbedoSpec; // Surface color
```

**Hemisphere Sample Kernel Generation:**
```cpp
std::vector<glm::vec3> ssaoKernel;
for (unsigned int i = 0; i < 64; ++i)
{
    glm::vec3 sample(
        randomFloats(generator) * 2.0 - 1.0,  // x: [-1, 1]
        randomFloats(generator) * 2.0 - 1.0,  // y: [-1, 1]
        randomFloats(generator)                // z: [0, 1] (hemisphere)
    );
    sample = glm::normalize(sample);
    sample *= randomFloats(generator);

    // Weight samples toward origin
    float scale = (float)i / 64.0;
    scale = lerp(0.1f, 1.0f, scale * scale);
    sample *= scale;

    ssaoKernel.push_back(sample);
}
```

**Random Rotation Noise:**
```cpp
// 4x4 tiled rotation texture
std::vector<glm::vec3> ssaoNoise;
for (unsigned int i = 0; i < 16; i++)
{
    glm::vec3 noise(
        randomFloats(generator) * 2.0 - 1.0,
        randomFloats(generator) * 2.0 - 1.0,
        0.0f  // Rotate around z-axis only
    );
    ssaoNoise.push_back(noise);
}
```

**SSAO Fragment Shader:**
```glsl
#version 330 core
out float FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];
uniform mat4 projection;

const vec2 noiseScale = vec2(800.0/4.0, 600.0/4.0);
const float radius = 0.5;
const float bias = 0.025;

void main()
{
    // Retrieve G-buffer data
    vec3 fragPos = texture(gPosition, TexCoords).xyz;
    vec3 normal = texture(gNormal, TexCoords).rgb;
    vec3 randomVec = texture(texNoise, TexCoords * noiseScale).xyz;

    // Create TBN matrix (tangent space -> view space)
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    // Sample kernel and accumulate occlusion
    float occlusion = 0.0;
    for(int i = 0; i < 64; ++i)
    {
        // Transform sample to view space
        vec3 samplePos = TBN * samples[i];
        samplePos = fragPos + samplePos * radius;

        // Project to screen space
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        // Sample depth at offset
        float sampleDepth = texture(gPosition, offset.xy).z;

        // Range check + occlusion test
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / 64.0);
    FragColor = occlusion;
}
```

**SSAO Blur Pass:**
```glsl
// Simple box blur to remove noise pattern
void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoInput, 0));
    float result = 0.0;
    for (int x = -2; x < 2; ++x)
    {
        for (int y = -2; y < 2; ++y)
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(ssaoInput, TexCoords + offset).r;
        }
    }
    FragColor = result / 16.0;
}
```

**Applying SSAO to Lighting:**
```glsl
vec3 ambient = vec3(0.3 * Diffuse * AmbientOcclusion);
```

**Key Parameters:**
- Kernel size: 32-64 samples (quality vs performance)
- Radius: 0.5-1.0 (occlusion search distance)
- Bias: 0.01-0.05 (prevents self-occlusion artifacts)
- Noise texture: 4x4 tiled (reduces banding)

### Screen-Space Reflections (SSR)

**Core Algorithm:**
- Ray march through depth buffer in screen space
- Reflect view ray across surface normal
- Find intersection with scene geometry
- Sample color at intersection point

**Use Cases:**
- Wet surfaces (rain, puddles)
- Metallic reflections
- Glass/mirror surfaces
- Water reflections

**Limitations:**
- Only reflects visible geometry
- Breaks at screen edges
- Expensive for high sample counts

### Bloom

From [Bloom - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Bloom) (accessed 2025-10-31):

Bloom simulates light bleeding from bright areas into surrounding pixels.

**Two-Pass Process:**
1. **Bright Pass**: Extract pixels above brightness threshold
2. **Blur Pass**: Gaussian blur bright regions
3. **Combine**: Additive blend with original scene

**Threshold Extraction:**
```glsl
vec3 color = texture(scene, TexCoords).rgb;
float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
if(brightness > threshold)
    BrightColor = vec4(color, 1.0);
else
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
```

**Gaussian Blur (Separable):**
- Horizontal blur pass
- Vertical blur pass
- Multiple passes for larger blur radius

**Final Combine:**
```glsl
vec3 scene = texture(sceneTexture, TexCoords).rgb;
vec3 bloom = texture(bloomTexture, TexCoords).rgb;
FragColor = vec4(scene + bloom * bloomStrength, 1.0);
```

## ARR-COC Advanced Rendering Integration

### Relevance-Based Tessellation

ARR-COC's Vervaekean framework can drive adaptive tessellation based on query relevance:

**Query-Aware LOD:**
```glsl
// In TCS: tessellation level from relevance score
float relevance = queryRelevanceMap(patchCenter, query);
float tessLevel = mix(MIN_TESS, MAX_TESS, relevance);
```

**Salience-Driven Detail:**
- High salience regions: maximum tessellation
- Low salience regions: minimal tessellation
- Opponent processing balances detail vs performance

### Vervaekean Shader Architecture

**Four Ways of Knowing Applied:**

1. **Propositional** (Statistical Information):
   - Heightmap entropy guides tessellation density
   - High-frequency detail requires more subdivision

2. **Perspectival** (Salience Landscapes):
   - User attention heat map influences LOD allocation
   - Gaze-tracked regions receive higher detail

3. **Participatory** (Query-Content Coupling):
   - Query-aware displacement magnitude
   - Relevant features emphasized through displacement

4. **Procedural** (Learned Skills):
   - Learned tessellation policies from usage patterns
   - Adaptive bias/threshold parameters per material type

### LOD-Based Parallax Selection

**Shader Technique Selection by Budget:**
```glsl
// ARR-COC token budget determines parallax quality
if (tokenBudget > 300) {
    texCoords = ParallaxOcclusionMapping(texCoords, viewDir, 32);
} else if (tokenBudget > 150) {
    texCoords = SteepParallaxMapping(texCoords, viewDir, 16);
} else if (tokenBudget > 64) {
    texCoords = BasicParallaxMapping(texCoords, viewDir);
} else {
    // No parallax, just normal mapping
}
```

**Opponent Processing Trade-offs:**
- Compress ↔ Particularize: Uniform detail vs focused complexity
- Exploit ↔ Explore: Known optimal settings vs experimental quality
- Focus ↔ Diversify: High detail in ROI vs balanced distribution

## Best Practices

### Performance Optimization

**Vertex Displacement:**
- Use tex2Dlod in vertex shaders (required for mipmaps)
- Minimize texture fetches (cache height value)
- Prefer compute shaders for CPU-visible mesh updates
- Keep displacement moderate (avoid extreme stretching)

**Parallax Mapping:**
- Use adaptive layer count (8-32 based on angle)
- Cache viewDir calculation in vertex shader
- Discard fragments at edges to prevent artifacts
- Combine with normal mapping for best results

**Geometry Shaders:**
- Minimize usage (tessellation often faster)
- Keep max_vertices low (<10 typically)
- Avoid in performance-critical paths
- Consider mesh shaders (modern alternative)

**Tessellation:**
- Use frustum culling before tessellation
- Implement backface culling in TCS
- Clamp tessellation levels to hardware limits
- Balance inner/outer levels (avoid cracks)

**Screen-Space Effects:**
- Render at half resolution (upscale with bilateral filter)
- Use temporal sampling (accumulate over frames)
- Implement early-out tests (sky pixels, far distances)
- Cache G-buffer data across multiple effects

### Common Pitfalls

**Vertex Displacement:**
- Forgetting `addshadow` pragma (incorrect self-shadowing)
- Not updating colliders (physics mismatch)
- Extreme displacement causing triangle inversion

**Parallax Mapping:**
- Too few layers (visible stepping)
- No range check (artifacts at steep angles)
- Missing bias (self-intersection acne)
- Incorrect texture wrapping (edge discontinuities)

**Geometry Shaders:**
- Generating too many primitives (bandwidth bottleneck)
- Not calling EndPrimitive (rendering breaks)
- Mismatched input/output layout (link errors)

**Tessellation:**
- Forgetting to set GL_PATCH_VERTICES
- TCS vertex count mismatch with patch size
- Crack artifacts (inconsistent edge tessellation)
- Missing TES (required shader stage)

**SSAO:**
- Too small radius (only local occlusion)
- No blur pass (visible noise pattern)
- Insufficient samples (banding artifacts)
- Missing range check (incorrect distant occlusion)

## Sources

**Web Research:**
- [Tessellation - LearnOpenGL](https://learnopengl.com/Guest-Articles/2021/Tessellation/Tessellation) (accessed 2025-10-31)
- [Interactive Map Shader: Vertex Displacement - Alan Zucconi](https://www.alanzucconi.com/2019/07/03/interactive-map-01/) (accessed 2025-10-31)
- [Geometry Shader - LearnOpenGL](https://learnopengl.com/Advanced-OpenGL/Geometry-Shader) (accessed 2025-10-31)
- [Parallax Mapping - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/Parallax-Mapping) (accessed 2025-10-31)
- [SSAO - LearnOpenGL](https://learnopengl.com/Advanced-Lighting/SSAO) (accessed 2025-10-31)
- [Parallax Shift - FJORD.STYLE](https://fjord.style/parallax-shift) (accessed 2025-10-31)
- [The performance of tessellation and geometry shaders - Reddit](https://www.reddit.com/r/vulkan/comments/1flx96k/) (accessed 2025-10-31)

**Video Tutorials:**
- [Vertex Displacement - Advanced Materials - Ben Cloward](https://www.youtube.com/watch?v=sYLbdO3VB8c) (accessed 2025-10-31)
- [Parallax Occlusion Optimization - Ben Cloward](https://www.youtube.com/watch?v=8hThP-Yni_o) (accessed 2025-10-31)

**Additional References:**
- [Advanced Shader Techniques in Unreal Engine - SDLC Corp](https://sdlccorp.com/post/advanced-shader-techniques-in-unreal-engine/) (accessed 2025-10-31)
- [Using Vertex Texture Displacement for Water Rendering - Game Developer](https://www.gamedeveloper.com/programming/using-vertex-texture-displacement-for-realistic-water-rendering) (accessed 2025-10-31)
