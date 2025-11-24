# Texture Mapping Techniques for 3D Rendering

**Created**: 2025-10-31
**Topic**: Texture mapping techniques - UV mapping, triplanar, projective, cube mapping, spherical mapping
**Relevance**: ARR-COC-VIS 32×32 grid UV coordinate strategies, adaptive texture LOD

---

## Overview

Texture mapping applies 2D images onto 3D surfaces to add visual detail without geometric complexity. Different mapping techniques solve specific challenges: UV mapping for artist-controlled unwrapping, triplanar for complex geometry without UVs, projective for dynamic camera-space effects, cube mapping for environment reflections, and spherical mapping for globe-like surfaces.

For ARR-COC-VIS, the 32×32 visual token grid creates a unique texture coordinate challenge - each patch (64-400 tokens) needs efficient texture sampling strategies that adapt to relevance-based LOD allocation.

---

## UV Mapping (Traditional Unwrapping)

### Core Concept

UV mapping unwraps 3D surfaces onto a 2D texture space using artist-defined coordinates (U, V) stored per vertex. The mesh stores UV coordinates that map vertices to texture positions.

**Key components**:
- **U axis**: Horizontal texture coordinate (typically 0-1 range)
- **V axis**: Vertical texture coordinate (typically 0-1 range)
- **UV unwrapping**: Process of flattening 3D geometry to 2D texture space
- **UV islands**: Disconnected pieces of UV layout
- **Seams**: Edges where UV islands separate

From [Catlike Coding - Triplanar Mapping](https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/) (accessed 2025-10-31):
> "The usual way to perform texture mapping is by using the UV coordinates stored per-vertex in a mesh. But this is not the only way to do it. Sometimes, there are no UV coordinates available. For example, when working with procedural geometry of arbitrary shapes."

### UV Coordinate Systems

**Standard UV layout** (0,0 to 1,1):
- Origin typically at bottom-left or top-left (varies by rendering API)
- Values outside 0-1 range handled by wrapping modes:
  - **Repeat/Wrap**: Tiles texture infinitely
  - **Clamp**: Extends edge pixels
  - **Mirror**: Alternates normal/flipped tiles

**UV transformations**:
```glsl
// Basic UV transformation
vec2 transformedUV = uv * tiling + offset;

// With rotation
mat2 rotation = mat2(cos(angle), -sin(angle),
                     sin(angle), cos(angle));
vec2 rotatedUV = rotation * (uv - 0.5) + 0.5;
```

### Unwrapping Strategies

**Planar projection** (single direction):
- Projects from X, Y, or Z axis
- Simple but causes stretching on non-planar surfaces
- Good for terrain viewed from above

**Cylindrical unwrapping**:
- Wraps around one axis (like wrapping paper around cylinder)
- Good for tubes, arms, legs
- Seam where cylinder edge meets

**Spherical unwrapping**:
- Projects from sphere center outward
- Good for heads, planets
- Pole distortion at top/bottom

**Box mapping**:
- Projects from 6 cube faces
- Reduces stretching compared to planar
- Creates 6 separate UV islands

### Challenges

From [Stack Overflow - Generating UV Coordinates](https://stackoverflow.com/questions/18578943/automatically-generating-uv-coordinates-algorithms) (accessed 2025-10-31):

**Stretching**: Non-uniform scaling when 3D surface curvature doesn't match 2D layout
**Seams**: Visible discontinuities where UV islands separate
**Texture density**: Inconsistent texel-to-surface ratios across mesh
**Manual work**: Artist time required for quality unwrapping

---

## Triplanar Mapping (UV-Free World-Space Projection)

### Core Algorithm

Triplanar mapping projects textures from three orthogonal planes (X, Y, Z axes) and blends based on surface normal direction. No UV coordinates required.

From [Ben Golus - Normal Mapping for Triplanar Shader](https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a) (accessed 2025-10-31):
> "World Space Triplanar Projection Mapping is a technique that applies textures to an object from three directions using the world space position. Three planes. This is also called 'UV free texturing', 'world space projected UVs' and a few other names."

**Basic implementation**:
```glsl
// Sample texture from 3 planes using world position
vec2 uvX = worldPos.zy;  // YZ plane (X-facing)
vec2 uvY = worldPos.xz;  // XZ plane (Y-facing)
vec2 uvZ = worldPos.xy;  // XY plane (Z-facing)

vec3 colorX = texture(albedoMap, uvX).rgb;
vec3 colorY = texture(albedoMap, uvY).rgb;
vec3 colorZ = texture(albedoMap, uvZ).rgb;

// Calculate blend weights from surface normal
vec3 blend = abs(worldNormal);
blend /= (blend.x + blend.y + blend.z);  // Normalize to sum = 1.0

// Blend the three projections
vec3 finalColor = colorX * blend.x +
                  colorY * blend.y +
                  colorZ * blend.z;
```

### Blend Weight Calculation

From [Catlike Coding - Triplanar Mapping](https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/):

**Linear blend** (basic):
```glsl
vec3 blend = abs(normal);
blend /= (blend.x + blend.y + blend.z);
```

**Sharpened blend** (reduces visible transitions):
```glsl
vec3 blend = abs(normal);
blend = max(blend - offset, 0.0);  // offset typically 0.2-0.5
blend = pow(blend, exponent);      // exponent typically 2-8
blend /= (blend.x + blend.y + blend.z);
```

**Height-based blend** (uses texture height data):
```glsl
vec3 heights = vec3(heightX, heightY, heightZ) + (blend * 3.0);
float heightStart = max(max(heights.x, heights.y), heights.z) - heightBlending;
vec3 h = max(heights - heightStart, 0.0);
blend = h / (h.x + h.y + h.z);
```

### Normal Map Handling

From [Ben Golus - Normal Mapping for Triplanar Shader](https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a):

**Problem**: Tangent space normal maps expect specific UV orientation - using mesh tangents with triplanar UVs produces incorrect lighting.

**Solution approaches**:

**1. Basic Swizzle** (reorient normals to world space):
```glsl
vec3 tnormalX = unpackNormal(texture(normalMap, uvX));
vec3 tnormalY = unpackNormal(texture(normalMap, uvY));
vec3 tnormalZ = unpackNormal(texture(normalMap, uvZ));

// Swizzle to match world orientation
vec3 worldNormalX = tnormalX.zyx;
vec3 worldNormalY = tnormalY.xzy;
vec3 worldNormalZ = tnormalZ.xyz;

// Account for negative-facing surfaces
vec3 axisSign = sign(worldNormal);
tnormalX.z *= axisSign.x;
tnormalY.z *= axisSign.y;
tnormalZ.z *= axisSign.z;
```

**2. Whiteout Blend** (blend normal map with surface normal):
```glsl
vec3 whiteoutBlend(vec3 mapNormal, vec3 surfaceNormal) {
    vec3 n;
    n.xy = mapNormal.xy + surfaceNormal.xy;
    n.z = abs(mapNormal.z) * surfaceNormal.z;
    return n;
}

// Apply to each projection
vec3 normalX = whiteoutBlend(tnormalX, worldNormal.zyx).zyx;
vec3 normalY = whiteoutBlend(tnormalY, worldNormal.xzy).xzy;
vec3 normalZ = whiteoutBlend(tnormalZ, worldNormal);
```

**3. Reoriented Normal Mapping (RNM)** - highest quality:
```glsl
vec3 rnmBlend(vec3 n1, vec3 n2) {
    n1 += vec3(0, 0, 1);
    n2 *= vec3(-1, -1, 1);
    return n1 * dot(n1, n2) / n1.z - n2;
}
```

### Mirroring Prevention

From [Catlike Coding - Triplanar Mapping](https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/):

Without correction, textures mirror on opposite-facing surfaces:

```glsl
// Fix mirroring by negating U coordinate based on normal direction
if (normal.x < 0) uvX.x = -uvX.x;
if (normal.y < 0) uvY.x = -uvY.x;
if (normal.z >= 0) uvZ.x = -uvZ.x;  // Z is opposite
```

### Use Cases

**Terrain**: Avoids UV stretching on cliffs and overhangs
**Procedural geometry**: No manual UV unwrapping needed
**Voxel/Minecraft-style**: Natural fit for cubic blocks
**Rock formations**: Complex shapes with difficult UV layouts

**Performance**: 3× texture samples (9× with normal maps) - more expensive than standard UV mapping

---

## Projective Texturing (Camera/Light-Space Projection)

### Core Concept

Projects texture from viewpoint of camera or light source, like a slide projector casting image onto surfaces. Uses projection matrix to transform world positions into texture coordinates.

**Basic projective mapping**:
```glsl
// Transform world position to projector clip space
vec4 projectorClip = projectorViewProjection * vec4(worldPos, 1.0);

// Perspective divide and convert to 0-1 UV range
vec2 projectorUV = (projectorClip.xy / projectorClip.w) * 0.5 + 0.5;

// Sample texture
vec3 projectedColor = texture(projectorTexture, projectorUV).rgb;
```

### Applications

**Shadow mapping**: Project shadow depth texture from light's viewpoint
**Decals**: Dynamic detail addition (bullet holes, graffiti)
**Caustics**: Light patterns through water/glass
**Flashlight/spotlight effects**: Dynamic lighting projections
**Blob shadows**: Simple character shadows

### Handling Projection Boundaries

**Behind projector**: Clip or fade pixels behind projection plane
**Outside frustum**: Clamp or fade at texture boundaries
**Oblique surfaces**: Correct for perspective distortion

```glsl
// Depth check - only project on surfaces in front
if (projectorClip.z < 0.0) discard;

// UV bounds check
if (projectorUV.x < 0.0 || projectorUV.x > 1.0 ||
    projectorUV.y < 0.0 || projectorUV.y > 1.0) discard;
```

---

## Cube Mapping (Environment Mapping)

### Core Concept

Stores 6 square textures representing view in each cardinal direction (+X, -X, +Y, -Y, +Z, -Z). Sample using 3D direction vector rather than 2D UV coordinates.

**Sampling cube maps**:
```glsl
// Use reflection or view direction to sample environment
vec3 reflectDir = reflect(viewDir, normal);
vec3 envColor = textureCube(environmentMap, reflectDir).rgb;
```

### Cube Map Generation

**Static**: Pre-rendered from scene center
**Dynamic**: Real-time rendering from object position (6 cameras, 1 per face)
**Paraboloid mapping**: Alternative using 2 textures instead of 6

### Applications

**Skyboxes**: Distant environment backgrounds
**Reflections**: Metallic/glossy surface reflections
**Environment lighting**: Image-based lighting (IBL)
**Irradiance maps**: Diffuse global illumination approximation

From [Real-Time Rendering Resources](https://www.realtimerendering.com) (accessed 2025-10-31):
> Environment mapping using cube maps has become standard for real-time reflections and physically-based lighting in modern game engines.

### Cube Map Coordinate Mapping

From [Stack Overflow - Cubic Mapping Algorithms](https://stackoverflow.com/questions/18578943/automatically-generating-uv-coordinates-algorithms) (accessed 2025-10-31):

**Direction to face selection**:
```glsl
vec3 absDir = abs(direction);
int faceIndex;
vec2 uv;

// Determine dominant axis
if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
    // +X or -X face
    faceIndex = (direction.x > 0.0) ? 0 : 1;
    uv = direction.zy / absDir.x;
} else if (absDir.y >= absDir.z) {
    // +Y or -Y face
    faceIndex = (direction.y > 0.0) ? 2 : 3;
    uv = direction.xz / absDir.y;
} else {
    // +Z or -Z face
    faceIndex = (direction.z > 0.0) ? 4 : 5;
    uv = direction.xy / absDir.z;
}

// Convert to 0-1 range
uv = uv * 0.5 + 0.5;
```

### Seamless Cube Maps

**Edge filtering**: Ensure colors match across cube face boundaries
**Mipmapping**: Pre-filter for different roughness levels
**Importance sampling**: Better quality for specular reflections

---

## Spherical Mapping

### Core Concept

Maps texture onto sphere using latitude/longitude coordinates (like Earth map). Single texture wraps completely around sphere.

**Spherical coordinate conversion**:
```glsl
vec3 sphericalToUV(vec3 direction) {
    float u = 0.5 + atan(direction.z, direction.x) / (2.0 * PI);
    float v = 0.5 - asin(direction.y) / PI;
    return vec2(u, v);
}
```

### Advantages & Disadvantages

**Advantages**:
- Single texture (vs 6 for cube map)
- Natural for spherical objects (planets, balls)
- Efficient storage

**Disadvantages**:
- Polar distortion (stretching at poles)
- Seam at longitude 0/360°
- Non-uniform texel density

### Variations

**Equirectangular** (standard lat/long):
- Most common for 360° panoramas
- Heavy distortion at poles

**Mercator projection**:
- Preserves angles but distorts area
- Extreme polar stretching

**Stereographic projection**:
- Conformal mapping
- Less polar distortion

---

## ARR-COC-VIS Texture Coordinate Strategies

### 32×32 Grid Patch UV Mapping

For ARR-COC-VIS's adaptive visual token grid, texture coordinates must support:

**Variable LOD per patch** (64-400 tokens):
- High-relevance patches: Dense sampling (400 tokens = 20×20 sub-grid)
- Low-relevance patches: Sparse sampling (64 tokens = 8×8 sub-grid)
- Medium patches: 144 tokens (12×12), 256 tokens (16×16)

**UV coordinate strategies**:

**1. Hierarchical UV grids**:
```python
# Generate patch-specific UV coordinates based on relevance
def generate_patch_uvs(patch_idx, token_count):
    # Patch location in 32×32 grid
    patch_row = patch_idx // 32
    patch_col = patch_idx % 32

    # Patch UV bounds (each patch = 1/32 of image)
    u_min = patch_col / 32.0
    u_max = (patch_col + 1) / 32.0
    v_min = patch_row / 32.0
    v_max = (patch_row + 1) / 32.0

    # Sub-grid resolution based on token count
    grid_size = int(sqrt(token_count))  # 8×8, 12×12, 16×16, or 20×20

    # Generate dense UV samples within patch bounds
    uvs = []
    for i in range(grid_size):
        for j in range(grid_size):
            u = u_min + (j / grid_size) * (u_max - u_min)
            v = v_min + (i / grid_size) * (v_max - v_min)
            uvs.append((u, v))

    return uvs
```

**2. Adaptive sampling with mipmaps**:
- High-relevance: Sample base mipmap level (full detail)
- Medium-relevance: Sample mipmap level 1-2 (reduced detail)
- Low-relevance: Sample mipmap level 3-4 (minimal detail)

**3. Triplanar for 3D vision tasks**:
- For point cloud visualization or 3D scene understanding
- No UV unwrapping needed
- Blend based on surface orientation

**4. Cube mapping for panoramic vision**:
- 360° camera inputs
- 6-face cube map sampled by viewing direction
- Efficient for wide field-of-view sensors

### Query-Aware Texture Sampling

From ARR-COC-VIS perspective:

**Participatory knowing** (query-content coupling):
- Sample texture regions relevant to query
- Skip texture detail in irrelevant patches

**Propositional knowing** (information content):
- Texture entropy guides sampling density
- High-frequency texture regions get more tokens

**Perspectival knowing** (salience):
- Edge detection in texture guides UV sampling
- Texture gradients indicate importance

---

## Advanced Techniques (2024-2025)

### Mesh Maps for Texture Enhancement

From [Real-Time Rendering Resources](https://www.realtimerendering.com) (accessed 2025-10-31):

**Ambient occlusion (AO) maps**: Pre-baked shadowing in crevices
**Curvature maps**: Edge detection for wear/weathering
**Thickness maps**: Subsurface scattering approximation
**World-space normal maps**: Eliminate tangent space calculations

### Procedural Texture Coordinates

**Generated at runtime**:
- Perlin noise-based UVs for organic variation
- Hash-based UVs for unique per-instance texturing
- Flow maps for animated water/lava

**Advantages**:
- No storage for UV data
- Infinite variation
- Automatic LOD through noise octaves

### Virtual Texturing & Sparse Virtual Textures

From [Adobe Substance 3D Painter - Sparse Virtual Textures](https://helpx.adobe.com/substance-3d-painter/painting/fill-projections/tri-planar-projection.html) (accessed 2025-10-31):

**Page-based streaming**:
- Load only visible texture regions
- Massive texture resolutions (16K+)
- Reduced VRAM usage

**Megatextures**:
- Entire scene in single massive texture
- Unique detail everywhere (no tiling)
- Streaming from disk

---

## Performance Considerations

### Texture Sample Cost

**Standard UV mapping**: 1-3 samples (albedo, normal, roughness)
**Triplanar mapping**: 3-9 samples (3× per texture type)
**Cube mapping**: 1 sample (but 6-face texture storage)
**Projective texturing**: 1-2 samples + transformation overhead

### Optimization Strategies

**Texture atlasing**: Combine multiple textures into single atlas
**Mipmap usage**: Pre-filtered LOD pyramid for distant surfaces
**Anisotropic filtering**: Better quality on oblique surfaces
**Texture compression**: BC7/ASTC for reduced memory bandwidth

From [ARR-COC-VIS context]:

**Token budget constraints**:
- High-relevance patches: Can afford expensive triplanar (400 tokens)
- Low-relevance patches: Stick to simple UV (64 tokens)
- Dynamic LOD: Adjust sampling complexity based on relevance score

---

## Comparison Table

| Technique | UV Required | Samples | Stretching | Use Case |
|-----------|-------------|---------|------------|----------|
| **UV Mapping** | Yes | 1-3 | Depends on unwrap | Standard textured meshes |
| **Triplanar** | No | 3-9 | Minimal | Complex/procedural geo |
| **Projective** | No | 1-2 | Can be severe | Dynamic decals, shadows |
| **Cube Map** | No | 1 | None (directional) | Environment, reflections |
| **Spherical** | No | 1 | Pole distortion | Panoramas, planets |

---

## Implementation References

### Unity-Specific

From [Catlike Coding - Triplanar Mapping](https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/):
- Custom surface function approach
- No default UV interpolator needed
- Whiteout blend for normal maps

### General GLSL/HLSL

From [Ben Golus - Normal Mapping for Triplanar Shader](https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a):
- Cotangent frame reconstruction from partial derivatives
- Cross-product tangent generation
- Reoriented Normal Mapping (RNM) technique

### Adobe Substance 3D Painter

From [Substance 3D Painter - Tri-planar Projection](https://helpx.adobe.com/substance-3d-painter/painting/fill-projections/tri-planar-projection.html) (accessed 2025-10-31):
- Artist-friendly triplanar parameters
- Hardness control for blend transitions
- Physical size scaling for consistent texel density

---

## Sources

**Web Research**:

- [Catlike Coding - Triplanar Mapping](https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/) (accessed 2025-10-31) - Comprehensive Unity implementation of triplanar mapping, normal handling, and blend optimization
- [Ben Golus - Normal Mapping for a Triplanar Shader](https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a) (accessed 2025-10-31) - Deep dive into correct triplanar normal mapping techniques including Whiteout blend, RNM, and tangent reconstruction
- [Adobe Substance 3D Painter - Tri-planar Projection](https://helpx.adobe.com/substance-3d-painter/painting/fill-projections/tri-planar-projection.html) (accessed 2025-10-31) - Production texture painting tool's triplanar implementation and parameters
- [Real-Time Rendering Resources](https://www.realtimerendering.com) (accessed 2025-10-31) - Industry-standard reference for modern rendering techniques
- [Stack Overflow - Generating UV Coordinates Algorithms](https://stackoverflow.com/questions/18578943/automatically-generating-uv-coordinates-algorithms) (accessed 2025-10-31) - Technical discussion of UV generation and cubic mapping algorithms

**Additional References**:
- GPU Gems 3 - Generating Complex Procedural Terrains (Nvidia, 2003)
- Blending in Detail (Stephen Hill) - Normal map blending techniques

---

## Related Oracle Knowledge

- `00-texture-sampling-filtering-2025-10-31.md` - Mipmapping, anisotropic filtering, LOD calculation
- `00-procedural-texture-generation-2025-10-31.md` - Shader-based texture synthesis, noise functions

**ARR-COC-VIS Integration**: The 32×32 patch grid with variable token allocation (64-400) requires adaptive texture coordinate strategies that match relevance-based LOD - high-salience patches use dense UV sampling or expensive triplanar, while low-relevance patches use sparse sampling or simple UV mapping to preserve token budget for more important visual regions.
