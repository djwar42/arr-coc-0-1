# Integrating LOD with Occlusion Culling

## Overview

LOD and occlusion culling provide multiplicative performance benefits when combined. LOD reduces polygon count per object, while culling eliminates entire objects from rendering. Together they enable rendering of massive scenes in real-time.

## Primary Sources

From [11-Integrating Occlusion Culling with Parallel LOD](../source-documents/11-Integrating%20Occlusion%20Culling%20with%20Parallel%20LOD%20for%20Rendering%20Complex%203D%20Environments%20on%20GPU%20-%20People.md):
- GPU-based occlusion culling
- Parallel LOD computation
- Unified rendering pipeline

From [06-Binary Space Partitioning](../source-documents/06-Binary%20space%20partitioning%20-%20Wikipedia.md):
- BSP trees for visibility determination
- Spatial organization for efficient culling

## Multiplicative Performance Benefits

### Individual Techniques
**LOD alone**: 3-10x polygon reduction
**Culling alone**: 2-5x object reduction (depends on scene occlusion)

**Combined**: 6-50x total speedup (LOD × Culling)

### Why Multiply, Not Add?
**Reason**: LOD and culling operate on different levels.

**Example scene**:
- 1000 objects, 10K polygons each = 10M polygons total
- **LOD reduces**: 10M → 2M polygons (5x reduction)
- **Culling removes**: 800 objects hidden → 200 objects visible
- **Combined**: 200 objects × 2K polygons = 400K polygons (25x reduction)

**Key insight**: Culling eliminates objects entirely, LOD simplifies what remains.

## Unified Pipeline Architecture

### Integrated Processing
From [11-Integrating Occlusion Culling]:

**Traditional (separate)**:
1. Occlusion culling (CPU)
2. LOD selection (CPU)
3. Rendering (GPU)

**Bottleneck**: CPU limits throughput

**Unified (GPU-based)**:
1. Coarse culling (GPU compute)
2. Parallel LOD selection (GPU compute)
3. Visibility refinement (GPU compute)
4. Rendering (GPU raster)

**Advantage**: All processing on GPU, massive parallelism.

### GPU Compute Pipeline
**Implementation**:
```glsl
// Compute Shader 1: Coarse Occlusion Culling
layout(local_size_x = 256) in;
void main() {
    uint object_id = gl_GlobalInvocationID.x;
    if (object_id >= num_objects) return;

    BoundingBox bbox = objects[object_id].bounds;
    if (!frustum_visible(bbox, camera)) {
        visibility[object_id] = false;
        return;
    }

    // Hardware occlusion query
    bool occluded = texture(depth_pyramid, bbox_center).r > bbox_depth;
    visibility[object_id] = !occluded;
}

// Compute Shader 2: LOD Selection for Visible Objects
void main() {
    uint object_id = visible_objects[gl_GlobalInvocationID.x];

    float distance = length(camera.pos - objects[object_id].center);
    float screen_size = project_size(objects[object_id].bounds, camera);

    int lod = select_lod(distance, screen_size);
    lod_levels[object_id] = lod;
}

// Compute Shader 3: Generate Draw Commands
void main() {
    uint object_id = visible_objects[gl_GlobalInvocationID.x];
    int lod = lod_levels[object_id];

    DrawCommand cmd;
    cmd.vertex_count = objects[object_id].lod_vertex_counts[lod];
    cmd.instance_count = 1;
    cmd.first_vertex = objects[object_id].lod_offsets[lod];
    cmd.base_instance = object_id;

    draw_commands[gl_GlobalInvocationID.x] = cmd;
}
```

**Result**: Entire pipeline executes on GPU in < 1ms for 10K objects.

## BSP Trees for Visibility

### Spatial Organization
From [06-Binary Space Partitioning]:

**Structure**: Recursive binary subdivision of 3D space

**Properties**:
- **Front-to-back traversal**: Natural occlusion culling order
- **Back-to-front traversal**: Painter's algorithm rendering
- **Hierarchical LOD**: Store LOD per BSP node

### BSP-Based Culling + LOD
**Algorithm**:
```python
def render_bsp_with_lod(node, camera, lod_threshold):
    if node is None:
        return

    # Determine which side of partition camera is on
    camera_side = which_side(camera.position, node.partition)

    if camera_side == FRONT:
        render_bsp_with_lod(node.back, camera, lod_threshold)  # Render back first (occluded)
        if node.visible(camera):
            lod = select_lod(node, camera, lod_threshold)
            render_node(node, lod)
        render_bsp_with_lod(node.front, camera, lod_threshold)  # Render front last (occluders)
    else:
        render_bsp_with_lod(node.front, camera, lod_threshold)
        if node.visible(camera):
            lod = select_lod(node, camera, lod_threshold)
            render_node(node, lod)
        render_bsp_with_lod(node.back, camera, lod_threshold)
```

**Advantage**: Automatic front-to-back ordering enables early-Z culling + LOD.

## Hierarchical Culling with LOD

### Bounding Volume Hierarchies (BVH)
**Structure**: Tree of bounding boxes containing scene objects

**Combined algorithm**:
1. **Frustum cull** BVH node (if outside, skip entire subtree)
2. **Occlusion cull** BVH node (if occluded, skip entire subtree)
3. **LOD select** for node (if far enough, render as single LOD object)
4. **Recurse** to children (if node too close, need detail)

**Benefit**: Culling entire groups with single test, LOD for distant groups.

### LOD Clusters
**Concept**: Group objects, treat cluster as single LOD object when far.

**Implementation**:
- Cluster nearby objects into bounding box
- Compute representative center and size
- Distance LOD: Far away → render cluster as single simplified mesh
- Distance LOD: Nearby → recurse, render individual objects with their own LOD

**Performance**: 10-100x reduction for distant object groups.

## Occlusion Query Optimization

### Hierarchical Z-Buffer (HZB)
**Technique**: Mipmap pyramid of depth buffer for fast occlusion queries.

**Structure**:
- Level 0: Full-resolution depth (2048×2048)
- Level 1: 1024×1024 (max of 2×2 pixels)
- Level 2: 512×512
- ...
- Level 11: 1×1 (furthest depth in scene)

**Occlusion test**:
```glsl
bool is_occluded(BoundingBox bbox, mat4 view_proj, texture2D hzb) {
    // Project bbox to screen space
    vec4 screen_rect = project_bbox(bbox, view_proj);

    // Select HZB level matching screen size
    int mip_level = compute_mip(screen_rect.size());

    // Sample HZB at bbox center
    float nearest_occluder = textureLod(hzb, screen_rect.center, mip_level).r;

    // Compare with bbox depth
    return bbox.min_depth > nearest_occluder;  // Occluded if farther than occluder
}
```

**Speed**: <1 cycle per query with hardware filtering.

### Two-Phase Culling
**Phase 1: Coarse culling** (CPU or compute)
- Frustum culling
- Coarse occlusion (previous frame HZB)
- LOD selection for potentially visible objects

**Phase 2: Fine culling** (GPU)
- Render occluders at LOD0 (high detail)
- Generate current frame HZB
- Hardware occlusion queries for remaining objects
- Final LOD adjustment based on actual visibility

**Result**: <1% false positives, <5% false negatives.

## Practical Integration Strategies

### Conservative LOD with Culling
**Problem**: Aggressive LOD can miss small features that become visible when close objects are culled.

**Solution**:
1. Perform occlusion culling first
2. Re-evaluate LOD after culling (objects may be closer to camera now)
3. Boost LOD for newly visible objects

**Example**: Building behind tree. Tree culled when camera moves → building becomes visible → boost building LOD.

### Temporal Coherence
**Observation**: Visibility and LOD change slowly between frames.

**Optimization**:
- Cache previous frame visibility
- Cache previous frame LOD
- Only recompute for objects with changed distance/visibility
- Amortize expensive queries over multiple frames

**Result**: 2-5x faster culling + LOD selection via temporal caching.

## Performance Analysis

### CPU vs. GPU Pipeline
From [11-Integrating Occlusion Culling]:

**CPU Pipeline** (traditional):
- Culling: 10ms for 10K objects
- LOD selection: 5ms
- Total: 15ms (66 FPS max)

**GPU Pipeline** (unified):
- Culling + LOD: 1ms for 10K objects
- Rendering: 8ms
- Total: 9ms (111 FPS)

**Speedup**: 15x faster culling+LOD, 1.7x overall framerate improvement.

### Memory Bandwidth
**Challenge**: Fetching object data for culling + LOD

**Optimization**:
- **Struct-of-Arrays**: Pack culling data separately from rendering data
- **Cache-friendly**: Arrange objects in spatial order
- **GPU resident**: Keep all object data on GPU, avoid CPU↔GPU transfers

**Result**: 5-10x bandwidth reduction.

## Case Studies

### Open-World Games (GTA V, Red Dead Redemption 2)
**Challenge**: Render 50+ km² with millions of objects

**Strategy**:
- BSP trees for coarse culling (city blocks)
- Occlusion culling per-building
- 4-5 LOD levels per object
- Streaming combined with culling + LOD

**Result**: 60 FPS with 500K+ objects in view.

### Architectural Visualization (Autodesk)
**Challenge**: Buildings with millions of components

**Strategy**:
- BVH for hierarchical culling
- LOD clusters (room-level groups)
- GPU compute pipeline
- Conservative occlusion (render occluders first)

**Result**: Real-time navigation of 10M+ polygon buildings.

### CAD/Manufacturing (Siemens NX)
**Challenge**: Assembly models with 100K+ parts

**Strategy**:
- Hierarchical culling by assembly structure
- LOD per part (3-4 levels)
- Occlusion culling for hidden internals
- View-frustum culling

**Result**: Interactive manipulation of massive assemblies.

## Connection to ARR-COC-VIS

Integrated culling + LOD demonstrates synergistic resource optimization:

**Shared principles**:
- **Eliminate irrelevant data**: Culling removes hidden objects ↔ We skip irrelevant patches
- **Simplify remaining data**: LOD reduces visible geometry ↔ We compress visible patches
- **Hierarchical processing**: BVH culling ↔ Hierarchical relevance scoring
- **Multiplicative benefits**: Culling × LOD → Large speedup ↔ Patch selection × Token compression → Large savings

**Key difference**: Spatial culling (visibility) vs. semantic culling (relevance).

**Adaptation**: Occlusion culling (binary: visible or not) maps to our relevance realization (continuous: 64-400 tokens). Both eliminate unnecessary computation.

## Summary

Integrating LOD with occlusion culling provides multiplicative performance improvements (6-50x) by eliminating hidden objects and simplifying visible ones. Our relevance realization framework combines similar strategies: eliminate irrelevant patches (low relevance) and compress relevant ones (variable token budgets).

**Core insight**: Don't render what you can't see (culling). Don't detail what you don't need (LOD). We extend this to VLMs: Don't process what isn't relevant (relevance-based selection), and compress what is (query-aware compression).

---

*This document synthesizes occlusion culling and LOD integration from real-time rendering research, demonstrating effective combination of complementary optimization strategies.*
