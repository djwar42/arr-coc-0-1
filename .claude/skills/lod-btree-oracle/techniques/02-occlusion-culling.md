# Occlusion Culling with LOD

## Overview

Occlusion culling eliminates rendering of objects hidden behind other geometry. When combined with parallel LOD, provides multiplicative performance benefits by reducing both polygon count (LOD) and polygon rendering (culling).

## Primary Sources

From [11-Integrating Occlusion Culling with Parallel LOD](../source-documents/11-Integrating%20Occlusion%20Culling%20with%20Parallel%20LOD%20for%20Rendering%20Complex%203D%20Environments%20on%20GPU%20-%20People.md):
- GPU-based occlusion culling
- Parallel LOD selection
- Integration strategies
- Performance analysis for complex 3D environments

## Key Concepts

### Occlusion Culling

**What it is**: Avoid rendering objects completely hidden by other objects.

**Benefits**:
- Skip hidden geometry (no vertex processing)
- Reduce overdraw (no pixel shading for occluded)
- Major savings in complex scenes (urban, indoor)

**Methods**:
- Software: CPU-based frustum + occlusion tests
- Hardware: GPU occlusion queries
- Hybrid: CPU coarse + GPU fine

### Level of Detail (LOD)

**What it is**: Render distant/less important objects with reduced geometric complexity.

**Benefits**:
- Fewer polygons processed
- Lower memory bandwidth
- Simpler shading

### Synergy: Culling + LOD

**Multiplicative benefits**:
- LOD reduces polygon count for visible objects
- Culling eliminates hidden objects entirely
- Combined: Render only visible geometry at appropriate detail

**Example**:
- Scene: 10M polygons
- LOD: 2M polygons (80% reduction)
- Culling: 1M polygons visible (50% of LOD set)
- Final: 1M polygons rendered (90% total reduction)

## Integration Strategies

### Parallel GPU Pipeline

From source 11, GPU-parallel approach:

**Stage 1: Coarse Culling (CPU)**
- Frustum culling
- Portal/PVS (Potentially Visible Set)
- Coarse occlusion (large occluders only)
- Output: Potentially visible objects

**Stage 2: LOD Selection (GPU)**
- Compute shader: Distance-based LOD selection
- Per-object LOD level determination
- Generate draw commands
- Output: LOD-selected objects

**Stage 3: Fine Occlusion Culling (GPU)**
- Hardware occlusion queries
- Test visibility against depth buffer
- Conditional rendering (skip if occluded)
- Output: Visible objects at appropriate LOD

**Stage 4: Rendering (GPU)**
- Render visible objects only
- Appropriate LOD level
- Efficient instancing

### Hierarchical Approaches

**Spatial hierarchy** (octree, BVH, BSP):
- Organize objects hierarchically
- Cull/LOD entire branches
- Reduces per-object overhead

**Temporal coherence**:
- Reuse previous frame visibility
- Update incrementally
- Amortize expensive queries

## GPU Occlusion Queries

### Hardware Queries

**OpenGL/Vulkan occlusion queries**:
```
1. Render scene's depth (no shading)
2. Issue occlusion query
3. Render object's bounding box
4. Check query result (pixels passed depth test?)
5. If pixels passed: Render full object
   Else: Skip object
```

**Challenges**:
- Latency (query results delayed)
- CPU-GPU synchronization stalls

**Solutions**:
- Multi-frame delay (use previous frame results)
- Conservative visibility (render if uncertain)
- Hierarchical queries (test groups first)

### Compute Shader Occlusion

**Modern approach** (from source 11):
```
ComputeShader OcclusionTest:
  Load depth buffer
  For each object:
    Project bounding box to screen
    Sample depth buffer at box corners
    If all samples > object depth: Visible
    Else: Occluded
  Write visibility flags
```

**Benefits**:
- No CPU-GPU synchronization
- Parallel processing
- Can combine with LOD selection

## LOD Selection Strategies

### Distance-Based LOD

**Simple metric**:
```
LOD_level = floor(distance / LOD_threshold)
```

**Screen-space error**:
```
projected_error = geometric_error / distance
LOD_level = select_LOD_for_error(projected_error)
```

### View-Frustum LOD

**Inside frustum**: Full LOD selection

**Outside frustum but near**: Simplified LOD (might come into view)

**Far outside frustum**: Minimal LOD or cull entirely

### Occlusion-Aware LOD

**If partially occluded**:
- Reduce LOD (less geometry to test)
- Lower quality acceptable (partially hidden)

**If fully visible**:
- Full LOD selection based on distance/importance

## Performance Optimization

### Batching and Instancing

**Group objects**:
- Same LOD level → batch draws
- Same material → reduce state changes
- Instancing for repeated objects

**From source 11**:
- GPU-driven instancing
- Indirect draw calls
- Minimal CPU involvement

### Hierarchical Occlusion

**Test large occluders first**:
- Buildings, terrain, large objects
- Cull entire regions
- Fine-grained tests only for remaining

**Hierarchical depth buffer**:
- Mipmap chain of depth buffer
- Test against appropriate mip level
- Coarse tests fast, fine tests accurate

### Temporal Coherence

**Frame-to-frame similarity**:
- Most objects visibility unchanged
- Reuse previous results
- Update changed regions only

**Predictive**:
- Predict camera motion
- Precompute visibility
- Reduce per-frame cost

## Practical Implementation

### Three-Pass Rendering

**Pass 1: Depth Pre-Pass**
- Render scene with minimal shaders (depth only)
- Low LOD acceptable (conservative depth)
- Establishes occlusion relationships

**Pass 2: Occlusion + LOD**
- GPU compute: Test visibility + select LOD
- Generate draw call list
- Compact and batch

**Pass 3: Final Rendering**
- Render visible objects at selected LOD
- Full shading
- Minimal wasted work

### Single-Pass with Early-Z

**Modern GPUs**:
- Early depth test (before fragment shader)
- Occlusion "free" if depth pre-computed
- Combine with LOD selection

**Pipeline**:
```
1. Sort front-to-back (maximize early-Z benefit)
2. Render with LOD selection
3. Early-Z culls occluded fragments
4. Shading only for visible
```

## Scene Complexity Management

### Urban Environments

**Challenge**: Dense geometry, high occlusion, varied LOD needs

**Strategy** (from source 11):
- Coarse PVS (portals between street blocks)
- Building-level occlusion
- Per-object LOD within visible buildings
- Aggressive culling between blocks

**Results**: 10-20× reduction in rendered geometry

### Indoor Environments

**Challenge**: Many small rooms, high occlusion, portals

**Strategy**:
- Portal-based visibility
- Room-level LOD
- Aggressive culling through doorways
- Hierarchical occlusion queries

### Outdoor Terrains

**Challenge**: Large open spaces, less occlusion

**Strategy**:
- Horizon culling
- Terrain LOD (heightfield tessellation)
- Object LOD (trees, rocks, etc.)
- Occlusion less important (fewer large occluders)

## Quality Considerations

### Popping Artifacts

**Problem**: Sudden LOD changes visible

**Solutions**:
- Smooth LOD transitions (alpha blending)
- Hysteresis in LOD selection
- Time-based interpolation
- Geomorphing (vertex position interpolation)

### Culling Errors

**Problem**: False positives (object culled but actually visible)

**Solutions**:
- Conservative culling (err on side of rendering)
- Expand bounding boxes slightly
- Multi-frame verification
- User-tunable aggressiveness

## Integration with Other Techniques

### Culling + LOD + Foveated Rendering

**Three-way optimization**:
1. Cull occluded objects
2. LOD visible distant objects
3. Foveated rendering for peripheral regions

**Combined savings**: 95%+ in complex VR scenes

### Culling + LOD + Streaming

**Stream based on visibility + LOD**:
- High-priority: Visible, high LOD
- Medium-priority: Visible, low LOD
- Low-priority: Occluded but near
- Don't load: Occluded and far

## Cross-References

- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - LOD basics
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - LOD calculation
- [integration/00-lod-with-culling.md](../integration/00-lod-with-culling.md) - Integration details
- [applications/00-video-games.md](../applications/00-video-games.md) - Game applications

## Key Takeaways

1. **Multiplicative benefits**: Culling + LOD = massive savings
2. **GPU-driven pipeline**: Modern GPUs handle both efficiently
3. **Hierarchical approach**: Coarse tests first, fine tests for remaining
4. **Temporal coherence**: Reuse previous frame visibility
5. **Conservative culling**: Err on side of rendering to avoid artifacts

**Design principle**: Use spatial hierarchy for coarse culling, GPU compute for parallel LOD + occlusion, render only visible geometry at appropriate detail, exploit temporal coherence for efficiency.
