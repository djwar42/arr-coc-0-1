# Progressive Buffers: View-Dependent Geometry and Texture LOD

## Overview

Progressive buffers enable smooth, view-dependent LOD transitions for both geometry and texture by maintaining progressive refinement structures that can be rendered at any intermediate quality level.

## Primary Sources

From [15-Progressive Buffers](../source-documents/15-Progressive%20Buffers_%20View-dependent%20Geometry%20and%20Texture%20LOD%20Rendering.md):
- Progressive mesh representation
- Texture pyramids and progressive transmission
- View-dependent refinement criteria
- Smooth LOD transitions

## Key Concepts

### Progressive Representation

**What it is**: Data structure supporting continuous LOD spectrum, not discrete levels.

**Benefits**:
- Smooth transitions (no popping)
- Exact detail budget matching
- View-dependent refinement
- Network streaming friendly

**Contrast with discrete LOD**:
- Discrete: 3-5 fixed levels, abrupt transitions
- Progressive: Continuous spectrum, smooth refinement

### Progressive Meshes

**Hoppe's progressive mesh** (1996):
- Base mesh (coarse)
- Series of vertex split operations
- Apply splits to refine
- Remove splits to coarsen

**Representation**:
```
ProgressiveMesh = {
  base_mesh,
  vertex_split_operations[n]
}
```

**Rendering at quality q**:
```
1. Start with base_mesh
2. Apply first q split operations
3. Render resulting mesh
```

**Benefits**:
- Exact polygon budget control
- Smooth refinement
- Streaming transmission

### Progressive Textures

**Texture pyramid** (mipmaps):
- Level 0: Full resolution (e.g., 2048×2048)
- Level 1: 1024×1024
- Level 2: 512×512
- ...
- Level n: 1×1

**Progressive transmission**:
- Send coarse levels first
- Refine with finer levels
- Display at any stage

**Wavelet-based progressive**:
- Decompose texture into wavelets
- Transmit low-frequency coefficients first
- Progressively add high-frequency detail

## View-Dependent Refinement

### Geometry Refinement Criteria

**Distance-based**:
```
refine_priority(split) = geometric_error / distance²
```

**Screen-space error**:
```
projected_error = geometric_error * focal_length / distance
refine if projected_error > threshold
```

**Silhouette preservation**:
- Higher priority for silhouette edges
- Lower priority for interior regions
- Maintains object outline quality

**View-frustum culling**:
- Don't refine outside frustum
- Lower priority near frustum boundary
- Highest priority for central view

### Texture Refinement Criteria

**Mipmap selection**:
```
mip_level = log2(distance / texture_size)
```

**Screen-space filtering**:
- Compute texture footprint on screen
- Select mip level to avoid aliasing
- Anisotropic filtering for oblique views

**Importance-based**:
- Salient textures: Higher resolution
- Background textures: Lower resolution
- Task-relevant: Highest priority

## Implementation Strategies

### Geometry LOD Pipeline

**1. Priority Computation**
```glsl
// Compute shader
for each vertex_split in refinement_queue:
  error = split.geometric_error
  dist = distance(split.position, camera)
  priority = error / (dist * dist)
  split.priority = priority
```

**2. Priority Sorting**
```
sort(refinement_queue, by: priority, descending)
```

**3. Refinement Budget Allocation**
```
polygon_budget = target_poly_count - current_poly_count
refine_count = polygon_budget / 2  // Each split adds 2 polygons
apply_top_n_splits(refine_count)
```

**4. Rendering**
```
render(refined_mesh)
```

### Texture LOD Pipeline

**1. Mip Level Calculation**
```glsl
// Fragment shader
float dist = length(worldPos - cameraPos);
float mip = log2(dist / textureSize);
mip = clamp(mip, 0, max_mip_level);
```

**2. Trilinear Filtering**
```glsl
// Blend between mip levels for smoothness
vec4 color = textureLod(tex, uv, mip);
```

**3. Anisotropic Filtering**
```glsl
// Handle oblique viewing angles
vec4 color = textureGrad(tex, uv, dpdx, dpdy);
```

## Smooth Transitions

### Geometry Geomorphing

**Problem**: Vertex split creates discontinuity

**Solution**: Interpolate vertex positions over time
```
vertex_pos_current = lerp(pos_before_split, pos_after_split, t)
```
Where t = time since split / transition_duration

**Benefits**:
- Eliminates popping
- Smooth visual flow
- Imperceptible transitions

**Cost**: 10-20% overhead (extra vertex attribute + interpolation)

### Texture Blending

**Problem**: Abrupt mip level changes visible

**Solution**: Trilinear filtering (built into GPUs)
- Sample two adjacent mip levels
- Blend based on fractional mip level

**Alternative**: Time-based blending
```glsl
vec4 color_old = textureLod(tex, uv, mip_old);
vec4 color_new = textureLod(tex, uv, mip_new);
vec4 final = mix(color_old, color_new, transition_alpha);
```

## Progressive Transmission and Streaming

### Network Transmission

**Progressive mesh streaming**:
1. Send base mesh (small, < 1KB)
2. Send vertex splits progressively
3. Refine as data arrives
4. Early display with coarse quality

**Texture streaming**:
1. Send low-res mip levels first (e.g., 64×64)
2. Progressively send higher resolution
3. Display improves as data arrives
4. Responsive even on slow connections

### Disk Streaming

**Out-of-core rendering**:
- Full quality data exceeds memory
- Stream from disk as needed
- Progressive structures ideal (read only what's needed)

**Level-of-detail streaming**:
- High LOD data on disk
- Load based on view-dependent priorities
- Unload low-priority data

## Performance Characteristics

### Geometry LOD Cost

**CPU cost**:
- Priority computation: O(n) for n active splits
- Sorting: O(n log n)
- Refinement application: O(k) for k splits applied

**GPU cost**:
- Rendering: O(p) for p polygons
- Same as non-progressive mesh at same polygon count

**Memory**:
- Base mesh + split records
- Typically 1.5-2× single LOD level
- Amortized over smooth transitions

### Texture LOD Cost

**Mipmapping overhead**:
- Memory: 1.33× (pyramid vs single level)
- Bandwidth: Slight increase (cache-friendly)
- Filtering: Hardware-accelerated (minimal cost)

## Quality Metrics

### Geometric Error

**Hausdorff distance**:
- Maximum distance from simplified mesh to original
- Conservative metric (worst-case)

**Quadric error**:
- Sum of squared distances
- Optimized by edge collapse algorithms

**Screen-space error**:
- Project error to screen
- User-perceptible metric (< 1 pixel imperceptible)

### Texture Quality

**Aliasing**: Avoid undersampling (Nyquist frequency)

**Blurriness**: Avoid oversampling (wasted memory/bandwidth)

**Anisotropy**: Handle oblique angles properly

## Integration with Other Techniques

### Progressive + Foveated Rendering

**View-dependent + foveated**:
- Central vision: High refinement priority
- Peripheral: Low refinement priority
- Smooth LOD gradient matches acuity

### Progressive + Occlusion Culling

**Visibility-driven refinement**:
- Visible objects: Refine normally
- Partially occluded: Reduce refinement
- Fully occluded: No refinement (or unload)

### Progressive + Streaming

**Network streaming**:
- Progressive ideal for incremental loading
- Display coarse immediately
- Refine as bandwidth allows

## Practical Applications

### Video Games

**Open world games**:
- Stream world progressively
- Refine near player
- Coarsen distant regions

**Multiplayer**:
- Progressive meshes for player models
- Network-friendly representation

### CAD/Visualization

**Large models** (buildings, mechanical):
- Progressive meshes for interactive viewing
- Refine as needed
- No discrete LOD popping

### VR/AR

**High resolution requirements**:
- Progressive refinement for large scenes
- Foveated + progressive combination
- Smooth transitions critical (comfort)

## Cross-References

- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - LOD basics
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - Refinement criteria
- [techniques/00-foveated-rendering.md](00-foveated-rendering.md) - Foveated integration
- [applications/00-video-games.md](../applications/00-video-games.md) - Game applications

## Key Takeaways

1. **Continuous LOD**: Progressive buffers support smooth spectrum, not discrete levels
2. **View-dependent**: Refine based on visibility, distance, importance
3. **Smooth transitions**: Geomorphing and blending eliminate popping
4. **Streaming-friendly**: Progressive transmission enables early display
5. **Proven techniques**: Progressive meshes (Hoppe 1996) widely adopted

**Design principle**: Represent geometry and textures progressively, compute view-dependent refinement priorities, allocate detail budget to highest priority regions, smooth transitions via interpolation, stream progressively for network/disk efficiency.
