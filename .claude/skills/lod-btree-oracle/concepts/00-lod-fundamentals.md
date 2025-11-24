# LOD Fundamentals: Level of Detail Basics

**What is Level of Detail and why does it matter?**

**Sources**: Progressive Buffers paper, Wikipedia LOD article, Unity/Unreal documentation

---

## Core Concept

**Level of Detail (LOD)** is a technique that reduces the number of GPU operations by decreasing the workload on graphics pipeline stages, usually vertex transformations, for objects that contribute less to the final image quality.

The fundamental insight: **Not all objects need the same level of geometric or texture detail.**

### Why LOD Matters

**Performance**: Modern scenes contain millions of polygons. Rendering everything at full detail is computationally prohibitive.

**Perceptual Equivalence**: Objects far from camera or in peripheral vision can be simplified without noticeable quality loss.

**Scalability**: LOD enables rendering complexity to scale with available hardware resources.

---

## Types of LOD

### 1. Discrete LOD

Predefined levels of geometric complexity, typically labeled LOD0 (highest detail) through LOD3+ (lowest detail).

**Example hierarchy:**
- **LOD0**: 10,000 triangles (close-up)
- **LOD1**: 2,500 triangles (mid-range)
- **LOD2**: 500 triangles (far)
- **LOD3**: 100 triangles (very far)

**Advantages:**
- Simple to implement
- Predictable memory footprint
- Easy artist control

**Disadvantages:**
- "Popping" artifacts during transitions
- Wasted detail levels (objects jump between levels)
- Fixed memory overhead

### 2. Continuous LOD

Smooth, progressive transitions between detail levels without abrupt changes.

**Implementation approaches:**
- **Progressive meshes**: Incremental vertex/triangle additions
- **Geomorphing**: Smooth interpolation between LOD levels
- **GPU tessellation**: Hardware-based dynamic subdivision

**Advantages:**
- No visible popping
- Optimal detail allocation
- Smooth visual experience

**Disadvantages:**
- More complex implementation
- Potential CPU overhead for mesh updates
- Requires sophisticated blending

---

## LOD Selection Criteria

### Distance-Based LOD

Most common approach: Select LOD based on camera-to-object distance.

**Formula (simplified):**
```
LOD_level = floor(distance / threshold)
```

**Improvements:**
- **Screen-space error**: Consider projected size, not just distance
- **Hysteresis**: Different thresholds for increasing/decreasing detail
- **LOD bias**: User-adjustable quality preference

### View-Dependent LOD

More sophisticated selection considering:
- **Viewing angle**: Objects viewed edge-on can use lower detail
- **Projected screen area**: Percentage of screen occupied
- **Velocity**: Fast-moving objects can use lower detail
- **Attention**: Gaze-aware LOD for VR/AR

### Perceptual LOD

Based on human visual system characteristics:
- **Foveal vs peripheral**: Higher detail in center of vision
- **Visual acuity**: Detail requirements decrease with eccentricity
- **Change blindness**: Transitions during saccades go unnoticed

**See also**: [concepts/03-transjective-relevance.md](03-transjective-relevance.md)

---

## Rendering Pipeline Integration

### Where LOD Fits

```
Scene Graph Traversal
    ↓
View Frustum Culling
    ↓
LOD Selection ← (distance, screen-space, attention)
    ↓
Occlusion Culling
    ↓
Rendering
```

### GPU Pipeline Considerations

**Vertex Stage:**
- Fewer vertices to transform
- Reduced vertex shader invocations

**Geometry Stage:**
- Optional tessellation for continuous LOD
- Geometry shader amplification

**Fragment Stage:**
- Smaller triangles = fewer fragments
- Texture LOD (mipmapping) runs parallel

---

## Common LOD Artifacts

### Popping

**Cause**: Abrupt transition between discrete LOD levels
**Solutions**:
- Continuous LOD
- Geomorphing
- Temporal blending
- Transition during motion/distraction

### Hysteresis Oscillation

**Cause**: Object repeatedly crossing LOD threshold
**Solution**: Use different thresholds for increase/decrease

```
Increase detail at: distance < 50m
Decrease detail at: distance > 55m
```

### LOD Selection Lag

**Cause**: Delayed LOD updates during rapid camera movement
**Solution**:
- Predictive LOD based on velocity
- Aggressive LOD bias during motion
- Async LOD selection

---

## LOD Metrics

### Geometric Complexity

**Triangle count**: Primary metric, directly impacts vertex processing
**Vertex count**: Actual transform workload
**Draw calls**: CPU overhead for rendering

### Visual Quality

**Screen-space error**: Pixel deviation from original mesh
**Silhouette preservation**: Maintaining object outline
**Normal deviation**: Surface orientation changes

### Performance

**Frame time**: Total rendering duration
**GPU utilization**: Vertex/fragment shader load
**Memory bandwidth**: Texture/geometry fetch overhead

---

## Best Practices

### Artist Workflow

1. **Author LOD0 at highest needed quality**
2. **Generate LOD1-3 via automatic simplification**
3. **Manually adjust for important features**
4. **Test LOD transitions in-game**
5. **Optimize LOD thresholds per scene**

### Technical Implementation

**Threshold Tuning:**
- Start with distance-based
- Add screen-space error
- Profile and adjust per object type
- Consider worst-case viewpoints

**Memory Management:**
- Stream LOD levels on demand
- Discard unused detail levels
- Share textures across LOD levels
- Use texture atlases

**Debugging:**
- Visualization modes showing active LOD
- Color-coded LOD levels
- Performance overlays
- LOD transition triggers

---

## LOD for Different Content Types

### Static Meshes
- Precomputed LOD levels
- Automatic simplification works well
- Focus on silhouette preservation

### Skeletal Meshes (Characters)
- LOD affects both mesh and animation
- Bone count reduction at lower LOD
- Joint simplification
- Animation LOD (reduced update rate)

### Foliage
- Billboard imposters at far distances
- Wind animation LOD
- Clustering for distant trees
- GPU instancing for efficiency

### Terrain
- Heightfield tessellation
- Tile-based streaming
- Geometry clipmaps
- See: [algorithms/03-heightfield-tessellation.md](../algorithms/03-heightfield-tessellation.md)

---

## Modern LOD Advances (2024-2025)

### Neural LOD

**LODGE (2025)**: Level-of-detail Gaussian Splatting
- Neural rendering with LOD
- Real-time quality
- Learned detail allocation

### GPU-Driven Rendering

- Compute shader LOD selection
- Mesh shaders for dynamic detail
- GPU culling pipelines
- Reduces CPU bottleneck

### Adaptive Tessellation

- Hardware tessellation stages
- View-dependent subdivision
- Watertight crack-free rendering
- See: [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md)

---

## LOD and ARR-COC-VIS Connection

Level of Detail naturally maps to **Vervaeke's relevance realization framework**:

**Propositional (Information)**: Statistical content determines base detail requirements
**Perspectival (Salience)**: Screen-space error and visual attention guide allocation
**Participatory (Coupling)**: View-dependent LOD realizes agent-arena relationship
**Procedural (Skills)**: Learned quality adaptation (like our Quality Adapter)

LOD is fundamentally about **realizing relevance** - determining what level of detail is relevant given:
- **Distance** (objective constraint)
- **Attention** (subjective focus)
- **Performance** (participatory constraint)

This is **transjective optimization**: detail emerges from the relationship between viewer and scene.

---

## Further Reading

**Concepts:**
- [concepts/01-bsp-btree-basics.md](01-bsp-btree-basics.md) - Spatial organization
- [concepts/03-transjective-relevance.md](03-transjective-relevance.md) - Perceptual LOD

**Techniques:**
- [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md) - View-dependent rendering
- [techniques/01-peripheral-degradation.md](../techniques/01-peripheral-degradation.md) - Perceptual LOD

**Algorithms:**
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - Selection strategies
- [algorithms/02-terrain-synthesis.md](../algorithms/02-terrain-synthesis.md) - Terrain-specific LOD

---

**Key Takeaway**: LOD is not just a performance optimization - it's a form of **relevance realization** that allocates computational resources based on transjective coupling between viewer and scene.
