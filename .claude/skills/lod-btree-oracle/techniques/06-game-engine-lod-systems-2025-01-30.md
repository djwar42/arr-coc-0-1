# Game Engine LOD Systems and Texture Streaming

**Date**: 2025-01-30
**Parent**: [00-foveated-rendering.md](00-foveated-rendering.md)
**Cross-Domain**: Game rendering techniques applicable to VLM token allocation

---

## Overview

Real-time game engines have pioneered advanced LOD (Level of Detail) systems that dynamically allocate computational resources based on viewer position, screen space, and perceptual importance. These techniques directly parallel VLM token allocation challenges: both must balance quality vs efficiency, handle multi-scale content, and adapt to dynamic viewing conditions.

**Core insight**: Game engines solve "what level of detail to render where" - VLMs solve "what token density to allocate where". The algorithms are remarkably similar.

---

## Unreal Engine Nanite: Virtualized Geometry

### What Is Nanite?

Nanite is Unreal Engine 5's virtualized geometry system that renders **pixel-scale detail** from massive polygon meshes (billions of triangles) by dynamically streaming and rasterizing only visible triangles.

**Key innovation**: Eliminates traditional LOD mesh creation by automatically generating hierarchical cluster DAGs and streaming triangle data like texture mipmaps.

### Technical Architecture

**Cluster DAG (Directed Acyclic Graph)**:
- Meshes decomposed into 128-triangle clusters
- Clusters hierarchically grouped into parent clusters (8:1 reduction)
- Forms multi-level tree structure similar to image pyramids
- Runtime selects appropriate cluster level per screen-space error

**Streaming System**:
```
Screen Space Error Budget → Cluster Selection → Triangle Streaming
    ↓                           ↓                      ↓
Pixel threshold (1-2px)    Select cluster level   Stream 64KB pages
```

**Performance characteristics** (from UE5.6):
- Streams geometry on-demand (similar to texture streaming)
- ~1-2 pixel screen-space error threshold
- Automatic LOD transitions (no pop-in)
- Scales to billions of triangles in viewport

### Relevance to VLM Token Allocation

**Direct parallels**:

1. **Hierarchical representation**: Cluster DAGs ↔ Token merge/prune hierarchies
2. **Screen-space error metrics**: Pixel error ↔ Perceptual relevance scores
3. **Dynamic budgets**: Texture pool ↔ Token budget (576-4096 tokens)
4. **Streaming**: Geometry pages ↔ Progressive token loading

**Nanite's LOD selection algorithm**:
```cpp
// Simplified Nanite cluster selection
float screenSpaceError = projectedBoundsSize / distanceToCamera;
int clusterLevel = selectClusterLevel(screenSpaceError, errorThreshold);
// errorThreshold typically 1-2 pixels

// VLM equivalent
float visualRelevance = queryAttentionScore(patch, query);
int tokenBudget = allocateTokens(visualRelevance, totalBudget);
// Similar trade-off: quality vs resource budget
```

---

## Texture Streaming and Mipmap Systems

### Texture Streaming Fundamentals

Game engines stream textures at appropriate mipmap levels based on:
- **Distance to camera**: Farther objects use lower mip levels
- **Screen coverage**: Texels-per-pixel ratio determines mip selection
- **Texture pool budget**: Limited VRAM (e.g., 512MB-2GB streaming pool)

**Mipmap pyramid**:
```
Level 0: 4096×4096 (16MB)    ← High detail, close-up
Level 1: 2048×2048 (4MB)
Level 2: 1024×1024 (1MB)
Level 3: 512×512 (256KB)
...
Level 10: 4×4 (16 bytes)     ← Low detail, distant
```

### Texture Streaming Pool Budget

**Unreal Engine streaming pool**:
- `r.Streaming.PoolSize` = VRAM budget for textures
- Dynamic allocation: High-priority textures loaded at higher mips
- Eviction policy: LRU (least recently used) when pool exhausted

**Budget allocation formula**:
```
mipLevel = clamp(
    log2(screenPixels / texturePixels),  // Screen space coverage
    minMip,
    maxMip
)

// Analogous VLM token allocation
tokenDensity = clamp(
    relevanceScore * maxTokensPerPatch,
    minTokens,   // 64 tokens
    maxTokens    // 400 tokens
)
```

### Anisotropic Filtering

Handles textures viewed at oblique angles by sampling multiple mipmap levels along anisotropic axes.

**Why relevant to VLMs**: Visual patches viewed "obliquely" by query attention may need different token densities along different dimensions (horizontal vs vertical scene features).

---

## Traditional LOD Systems vs Nanite

### Traditional Mesh LODs

**Artist-created LOD chain**:
```
LOD0: 100K triangles (0-10m viewing distance)
LOD1: 25K triangles  (10-50m)
LOD2: 5K triangles   (50-100m)
LOD3: 1K triangles   (100m+)
```

**Transition strategies**:
- **Discrete switching**: Pop-in artifacts, but simple
- **Alpha blending**: Smooth transition, double rendering cost
- **Geometric morphing**: Vertices interpolate between LOD levels

**Performance** (from Reddit UE5 discussions):
- Overdraw-focused traditional LODs: ~60 FPS
- Nanite (upfront cost, better scaling): ~55-65 FPS (varies by scene)
- **Trade-off**: Nanite has higher baseline cost but scales better with complexity

### When to Use Traditional LODs vs Nanite

**Use traditional LODs**:
- Simple geometric objects (< 10K triangles)
- Mobile/low-end hardware
- Foliage (Nanite foliage support improved in UE5.6)

**Use Nanite**:
- Highly detailed meshes (> 100K triangles)
- Environments with many unique assets
- When LOD creation workflow is bottleneck

**VLM analogy**:
- Traditional LODs ↔ Fixed patch sizes (ViT-style)
- Nanite ↔ Adaptive token allocation (ARR-COC-VIS style)

---

## Hierarchical Z-Buffer (Hi-Z) and Occlusion Culling

### GPU-Driven Occlusion Culling

Modern game engines use **hierarchical depth buffers** to cull occluded geometry before rasterization.

**Hi-Z pyramid**:
```
Level 0: Full resolution depth (1920×1080)
Level 1: Half resolution (960×540) - max depth per 2×2 block
Level 2: Quarter resolution (480×270)
...
Level N: Single depth value (scene bounding depth)
```

**Culling test**:
```cpp
// Test if object bounding box is occluded
bool isOccluded(BoundingBox box, HiZBuffer hiz) {
    float closestDepth = hiz.queryMinDepth(box.screenRect);
    return box.maxDepth < closestDepth;  // Behind other geometry
}
```

### Relevance to Vision Transformers

**VLM occlusion-aware token allocation**:
- Some image regions fully occluded by foreground objects
- Could use depth-aware pruning (if depth available)
- Or use saliency/segmentation to detect occluded background

**Current VLM approaches lack spatial occlusion reasoning** - game engines handle this efficiently with Hi-Z.

---

## Real-Time Performance Metrics

### Nanite Performance Data (UE5.5-5.6)

From community benchmarks and Epic documentation:

**Triangle throughput**:
- **Software rasterization**: 1-2 billion triangles/frame @ 60 FPS
- **Hardware rasterization fallback**: For translucent, masked materials

**Memory footprint**:
- Compressed cluster data: ~50% of original mesh size
- Streaming overhead: 64KB granularity pages
- Texture streaming pool: Independent budget (512MB-2GB typical)

**Latency**:
- Upfront frame cost: Higher than traditional (5-10ms baseline)
- Scaling: Better with scene complexity (marginal cost per object low)

### Lessons for VLM Systems

**Resource budgets parallel**:

| Game Engine | VLM System |
|------------|-----------|
| Texture pool (MB) | Token budget (count) |
| Triangle budget (millions) | Patch count × tokens/patch |
| Frame time (16.6ms @ 60 FPS) | Inference latency (ms) |
| LOD transitions | Token merge/prune schedules |

**Key insight**: Game engines use **heterogeneous budgets** (geometry + texture + shader) - VLMs could similarly use heterogeneous token types (visual + positional + semantic).

---

## Code Example: LOD Selection Algorithm

```python
# Game engine LOD selection (simplified)
def select_lod_level(object_bounds, camera_pos, screen_height):
    """Select LOD level based on screen-space coverage."""
    distance = np.linalg.norm(object_bounds.center - camera_pos)
    projected_size = object_bounds.radius / distance  # Simple projection
    screen_coverage = projected_size * screen_height  # Pixels

    # Discrete LOD levels
    if screen_coverage > 200:
        return 0  # Highest detail
    elif screen_coverage > 100:
        return 1
    elif screen_coverage > 50:
        return 2
    else:
        return 3  # Lowest detail

# VLM token allocation equivalent
def allocate_patch_tokens(patch, query, max_budget=400, min_budget=64):
    """Allocate tokens to patch based on query relevance."""
    relevance = compute_attention_score(patch, query)
    screen_coverage = estimate_patch_importance(patch)  # Analogous

    # Continuous allocation
    token_budget = min_budget + (max_budget - min_budget) * relevance
    token_budget = np.clip(token_budget, min_budget, max_budget)

    return int(token_budget)
```

---

## Cross-References

**Related LOD oracle files**:
- [00-foveated-rendering.md](00-foveated-rendering.md) - Gaze-aware LOD
- [01-peripheral-degradation.md](01-peripheral-degradation.md) - Distance-based quality reduction
- [algorithms/00-btree-traversal.md](../algorithms/00-btree-traversal.md) - Hierarchical traversal (like cluster DAGs)

**VLM token allocation files**:
- [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Token budgeting strategies
- [00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md](00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md) - Hierarchical token reduction

---

## References

**Academic**:
- arXiv 2507.08142 (Jul 2025): "A Technical Review of Unreal Engine" - LOD and Nanite analysis
- Nature Scientific Reports s41598-025-94464-6 (2025): "High-resolution image reflection removal by Laplacian pyramid network" - Multi-scale processing

**Industry Documentation**:
- Epic Games: "Nanite Virtualized Geometry in Unreal Engine" (UE5 documentation)
- Epic Games: "Texture Streaming Configuration in Unreal Engine"
- Game Developer (2023): "Level-of-detail as artistic codec"

**Community Discussions**:
- Reddit r/unrealengine: "Is Nanite good or bad for performance?" (2024-2025 discussions)
- Unreal Engine Forums: "UE5.5+ Feedback: Performance innovations beyond frame smearing" (May 2023-2025)
- Reddit r/unrealengine: "The Witcher 4 - Gameplay UE 5.6 Tech Demo | State of Unreal 2025" (June 2025, 40+ comments)

**Performance Benchmarks**:
- UE5.6 State of Unreal 2025: Nanite foliage improvements, streaming optimizations
- YouTube (GamaWooDev): "Stop Wasting Your Time! UE5 LODs vs Nanite Explained"

**UE5.6+ Specific References**:
- Tom Looman: "Unreal Engine 5.6 Performance Highlights" (July 2025) - https://www.tomlooman.com/unreal-engine-5-6-performance-highlights/
- Epic Games: "All the big news and announcements from the State of Unreal 2025" (June 2025)
- Epic Games: "Unreal Engine 5.6 Release Notes" (June 2025)
- CD PROJEKT RED: "The Witcher 4 Unreal Engine 5 Tech Demo" (State of Unreal 2025)

---

---

## Unreal Engine 5.6+ Improvements (June 2025)

### Nanite Foliage Performance Enhancements

**Major breakthrough**: UE5.6 introduces **Nanite Foliage**, optimizing dense vegetation rendering with production-ready performance at 60 FPS on current-gen consoles.

**Key improvements** (from Tom Looman's UE5.6 Performance Highlights):

1. **Distance culling for CSM**: Fixed distance culling bug for Nanite rendering into Cascaded Shadow Maps
   - Now correctly sets up culling view overrides
   - Per-instance cull distance (critical for foliage) works properly

2. **Single-view specialization**: Optimized instance cull shader for visibility buffer
   - Compiler removes loop overhead
   - Significantly lowers register pressure
   - Faster per-frame culling

3. **Static geometry instance cull**: Specialized path for static geo (`r.Nanite.StaticGeometryInstanceCull`)
   - Reduces cost and register pressure
   - Disabled by default due to edge cases

4. **Aggregate instance draw distance**: Hierarchical culling for small instances
   - Chunks track aggregate draw distances
   - **Massive win for foliage**: Scenes with many small instances using per-instance culling
   - Reduces overdraw significantly

5. **Chunk-based instance culling**: Changed from cell-based to 64-instance chunks
   - Supports GPU-updated instances (PCG)
   - Improves scenes with large amounts of procedural foliage
   - Better scaling for dense vegetation

**Performance metrics** (The Witcher 4 Tech Demo):
- **60 FPS** on PlayStation 5 with raytracing enabled
- Dense foliage scenes with high fidelity
- Memory-efficient rendering (better than traditional foliage systems)

**Upcoming UE5.7**: Nanite Foliage leaves experimental, becomes production-ready

```cpp
// UE5.6 Nanite foliage optimization example
// Aggregate draw distance culling for small instances
struct FNaniteFoliageChunk {
    float AggregateDrawDistance;  // Max draw dist for chunk
    uint32 InstanceCount;         // 64 instances per chunk
};

// Hierarchical culling pseudocode
bool CullChunk(FNaniteFoliageChunk chunk, float viewDistance) {
    if (viewDistance > chunk.AggregateDrawDistance) {
        return true;  // Cull entire chunk (64 instances at once)
    }
    return false;  // Proceed to per-instance culling
}
```

### Fast Geometry Streaming Plugin (Experimental)

**Collaboration**: Epic + CD PROJEKT RED (The Witcher 4)

**Major open-world streaming improvements**:

1. **Per-frame budgets**: Tunable time budgets for streaming operations
   - AddToWorld budget (actor spawning)
   - RemoveFromWorld budget (actor cleanup)
   - Mesh streaming budget

2. **Asynchronous physics**: Physics state creation/destruction off GameThread
   - **Huge win**: Removes streaming hitches
   - Parallel physics setup during level loading

3. **Incremental EndPlay**: Improved RemoveFromWorld (spread over multiple frames)
   - No more massive spikes when unloading levels
   - Time-sliced actor cleanup

4. **Unified time budget**: Shared budget for ProcessAsyncLoading + UpdateLevelStreaming
   - Better allocation under load
   - High-priority streaming handled efficiently

**Example CitySample budgets** (from UE5.6 docs):
```ini
; Fast Geometry Streaming budgets (CitySample reference)
s.LevelStreaming.AddToWorldBudget=0.5ms    ; Per-frame actor spawn budget
s.LevelStreaming.RemoveFromWorldBudget=0.3ms  ; Per-frame cleanup budget
s.Streaming.MeshStreamingBudget=2.0ms      ; Geometry streaming budget
```

**Performance impact**:
- ~100μs improvement in CitySample (UE5.6)
- Smoother streaming in dense open worlds
- Reduced hitches during fast camera movement

### Virtual Shadow Maps Optimizations

**Receiver masks**: New clipmap culling for dense scenes (`r.Shadow.Virtual.UseReceiverMask`)
- Significantly improves culling effectiveness with dynamic lights
- Reduces rendered geometry for distant shadows
- Potential for artifacts with DOF resolution bias (off by default)

**Dynamic geometry culling**: Clipmap far plane set to visible range only
- `r.Shadow.Virtual.Clipmap.CullDynamicTightly` (default: true)
- Greatly reduces geometry in some cases

**Performance** (console @ 1080p):
- Typical savings: ~0.5-1.0ms per frame
- Higher savings in foliage-heavy scenes

### Renderer Parallelization

**Refactored RHI API**: Remove render thread constraints
- Full multi-threading for supported platforms
- **Major win**: Render thread often the bottleneck in UE titles
- Better CPU utilization on modern hardware

### GPU Profiler 2.0

**New profiling architecture**:
- **Two GPU tracks**: Graphics + Compute (separate visualization)
- Async compute now properly displayed (pre-5.6 was misleading)
- Pipeline bubbles, fence waits, cross-queue dependencies visible

**Impact on understanding performance**:
- Pre-5.6: Async compute hidden, stats confusing
- Post-5.6: Clear visibility into GPU work distribution

---

## Parallels: Game Engine Streaming → VLM Token Allocation

### Geometry Streaming → Progressive Token Loading

**Game engine concept**:
```
Distance-based LOD → Stream high-res geometry on-demand → Unload when far
```

**VLM equivalent**:
```
Relevance-based LOD → Load high-res tokens for salient patches → Prune when irrelevant
```

**Key insight**: Both systems manage **limited pool budgets** (VRAM vs token count) with **dynamic allocation** based on importance metrics.

### Nanite's Screen-Space Error → Perceptual Relevance

**Nanite uses** 1-2 pixel threshold for LOD selection (imperceptible error).

**VLM could use** Just Noticeable Difference (JND) threshold for token allocation:
- Patches below JND threshold → minimum tokens (64)
- Patches above JND threshold → scale tokens based on importance (64-400)

### Hierarchical Culling → Hierarchical Token Pruning

**Nanite**: Cull 64-instance chunks before per-instance tests

**VLM**: Prune entire patch groups (8×8 regions) before per-patch allocation

**Performance benefit**: **~8× reduction** in culling/allocation operations

---

## Future Directions

**Game engine → VLM knowledge transfer**:

1. **Cluster DAG hierarchies** → Token merge hierarchies (8:1 reduction ratio proven effective)
2. **Screen-space error thresholds** → Perceptual relevance thresholds (1-2 pixel error ≈ JND)
3. **Texture streaming pools** → Progressive token loading (load high-res tokens on-demand)
4. **Hi-Z occlusion culling** → Depth-aware token pruning (remove occluded regions)
5. **Fast Geometry Streaming** → Fast Token Streaming (async loading, time budgets)
6. **Nanite Foliage chunk culling** → VLM patch group culling (hierarchical relevance)

**Research questions**:
- Can VLMs benefit from "geometry streaming" style progressive token loading?
- Should token budgets be heterogeneous (like geometry + texture budgets)?
- Can we adapt Nanite's 1-2 pixel error threshold to perceptual token importance?
- How can async token loading (like async physics) reduce inference latency?
- Can we use chunk-based token culling (64-token groups) for efficiency?

