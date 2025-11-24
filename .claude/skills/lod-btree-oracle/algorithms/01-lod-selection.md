# LOD Selection Algorithms

**Distance metrics, screen-space error, hysteresis, and popping prevention strategies**

---

## Overview

Level of Detail (LOD) selection determines which geometric representation to render for each object based on viewing parameters. The fundamental challenge is balancing visual quality against computational cost - rendering full detail everywhere wastes GPU resources, while aggressive simplification produces visible artifacts. Modern LOD selection combines multiple metrics (distance, screen-space error, velocity, attention) with temporal smoothing to achieve imperceptible quality transitions.

**Core principle**: Allocate geometric detail proportional to visual contribution.

**Historical evolution**: Simple distance thresholds (1990s) → screen-space error metrics (2000s) → perceptual and attention-aware selection (2010s+).

---

## Primary Sources

**Screen-Space Error Metrics:**
- `source-documents/15-Progressive Buffers_ View-dependent Geometry and Texture LOD Rendering.md` - View-dependent LOD rendering with continuous detail
- `source-documents/11-Integrating Occlusion Culling with Parallel LOD for Rendering Complex 3D Environments on GPU - People.md` - GPU-parallel LOD selection

**Popping Prevention:**
- `source-documents/13-Managing Level of Detail through Head-Tracked Peripheral Degradation_ A Model and Resulting Design Principles - arXiv.md` - Temporal coherence in VR LOD

**Perceptual LOD:**
- `source-documents/03-Applying level-of-detail and perceptive effects to 3D urban semantics visualization - Eurographics Association.md` - Perceptual LOD for urban scenes

---

## Key Concepts

### Distance-Based LOD

**Simplest approach**: Select LOD level based on Euclidean distance from camera to object.

**Formula**:
```
distance = ||cameraPosition - objectCenter||
lodLevel = clamp(floor(distance / lodThreshold), 0, maxLOD)
```

**Problems**:
- Ignores projected screen size
- Fails for different object scales
- No consideration of viewing angle
- Abrupt transitions cause popping

**Use case**: Fast approximation for distant objects, background elements.

### Screen-Space Error

**Better metric**: Measure geometric error in screen pixels.

**Definition**: Maximum pixel deviation between simplified and original geometry when projected to screen.

**Calculation**:
```
# For each triangle in simplified mesh
worldSpaceError = max(distance(simplifiedVertex, originalSurface))

# Project to screen space
screenSpaceError = worldSpaceError * (screenHeight / (2 * tan(fov/2) * distanceToCamera))

# Select LOD to keep error below threshold
if screenSpaceError < targetPixelError:
    useLOD = currentLevel
else:
    useLOD = currentLevel - 1  # Higher detail
```

**Typical threshold**: 1-2 pixels of error at full resolution.

### Projected Screen Area

**Alternative metric**: Percentage of screen occupied by object.

**Calculation**:
```
# Bounding sphere projection
radius_world = object.boundingSphere.radius
distance = ||cameraPosition - objectCenter||

# Angular size
angularSize = 2 * atan(radius_world / distance)

# Screen space projection
screenRadius = (angularSize / fov) * screenHeight / 2
screenArea = π * screenRadius²

# LOD threshold
screenAreaRatio = screenArea / (screenWidth * screenHeight)

if screenAreaRatio > 0.25:
    lodLevel = 0  # Full detail (>25% of screen)
else if screenAreaRatio > 0.05:
    lodLevel = 1  # Medium detail (5-25%)
else if screenAreaRatio > 0.005:
    lodLevel = 2  # Low detail (0.5-5%)
else:
    lodLevel = 3  # Minimal detail (<0.5%)
```

**Advantages**:
- Intuitive artist control
- Works for all object types
- Handles scale naturally

---

## Algorithm Details

### Basic LOD Selection Pipeline

```
function SelectLOD(object, camera, viewport):
    # Step 1: View frustum culling
    if not FrustumContains(camera.frustum, object.bounds):
        return LOD_CULLED

    # Step 2: Calculate distance
    distance = Distance(camera.position, object.center)

    # Step 3: Calculate screen-space metrics
    screenError = CalculateScreenSpaceError(object, camera, viewport)

    # Step 4: Select appropriate LOD
    for level in 0 to object.maxLOD:
        if screenError < object.lodThresholds[level]:
            return level

    return object.maxLOD  # Lowest detail
```

### Screen-Space Error Calculation

**For mesh simplification algorithms**:

```
function CalculateScreenSpaceError(object, camera, viewport):
    # Get bounding sphere
    center = object.bounds.center
    radius = object.bounds.radius

    # Distance to camera
    distance = Distance(camera.position, center)

    # Maximum geometric error for current LOD
    # (stored during mesh simplification)
    geometricError = object.currentLOD.maxError

    # Project to screen space
    # Error scales inversely with distance
    viewportHeight = viewport.height
    fov = camera.fieldOfView

    screenError = (geometricError * viewportHeight) /
                  (2 * distance * tan(fov / 2))

    return screenError
```

**Geometric error** is typically computed during mesh simplification (quadric error metrics).

### Hysteresis for Stability

**Problem**: Objects near LOD threshold oscillate between levels.

**Solution**: Use different thresholds for increasing vs decreasing detail.

```
function SelectLODWithHysteresis(object, camera, viewport):
    screenError = CalculateScreenSpaceError(object, camera, viewport)
    currentLOD = object.activeLOD

    # Hysteresis margin (typically 10-20%)
    HYSTERESIS = 1.15

    for level in 0 to object.maxLOD:
        threshold = object.lodThresholds[level]

        # Increasing detail (lower LOD number)
        if level < currentLOD:
            if screenError > threshold * HYSTERESIS:
                return level

        # Decreasing detail (higher LOD number)
        else if level > currentLOD:
            if screenError < threshold / HYSTERESIS:
                return level

    return currentLOD  # No change
```

**Result**: Stable LOD selection, eliminates oscillation artifacts.

### Temporal Smoothing

**Problem**: Rapid LOD changes during camera movement cause distracting pops.

**Solution**: Smooth LOD transitions over multiple frames.

```
function SelectLODWithSmoothing(object, camera, viewport, deltaTime):
    targetLOD = SelectLODWithHysteresis(object, camera, viewport)
    currentLOD = object.activeLOD

    if targetLOD == currentLOD:
        return currentLOD

    # Smooth transition over time
    TRANSITION_TIME = 0.3  # seconds

    object.lodTransitionProgress += deltaTime / TRANSITION_TIME

    if object.lodTransitionProgress >= 1.0:
        object.lodTransitionProgress = 0.0
        object.activeLOD = targetLOD

    return currentLOD  # Keep current until transition complete
```

**Enhanced version**: Blend between LOD levels during transition (geomorphing).

---

## Advanced Selection Strategies

### View-Dependent LOD

**Enhancement**: Consider viewing angle, not just distance.

```
function ViewDependentLODScore(object, camera):
    # Base distance metric
    distance = Distance(camera.position, object.center)

    # Viewing angle factor
    viewDir = Normalize(object.center - camera.position)
    objectForward = object.transform.forward
    alignment = DotProduct(viewDir, objectForward)

    # Objects viewed edge-on can use lower detail
    # alignment: 1.0 (facing) → 0.0 (perpendicular)
    viewFactor = lerp(0.5, 1.0, abs(alignment))

    # Adjusted distance
    effectiveDistance = distance / viewFactor

    return effectiveDistance
```

**Applications**: Characters (side views need less detail than front views), buildings, vehicles.

### Velocity-Based LOD

**Insight**: Fast-moving objects can use lower detail (motion blur masks simplification).

```
function VelocityAdjustedLOD(object, camera):
    baseDistance = Distance(camera.position, object.center)

    # Object velocity relative to camera
    relativeVelocity = Magnitude(object.velocity - camera.velocity)

    # Velocity scaling factor
    # Fast motion (>10 m/s) allows 2x reduction
    # Slow motion (<1 m/s) no reduction
    velocityFactor = clamp(relativeVelocity / 10.0, 0.0, 1.0)
    distanceMultiplier = 1.0 + velocityFactor

    effectiveDistance = baseDistance * distanceMultiplier

    return SelectLODByDistance(effectiveDistance)
```

**Result**: 10-20% polygon reduction during fast camera movement with imperceptible quality loss.

### Importance-Based LOD

**Concept**: Not all objects are equally important.

```
function ImportanceWeightedLOD(object, camera, viewport):
    baseScore = CalculateScreenSpaceError(object, camera, viewport)

    # Importance multiplier
    # Hero characters: 2.0 (more detail)
    # Background props: 0.5 (less detail)
    importance = object.importanceWeight

    adjustedScore = baseScore / importance

    return SelectLODByScore(adjustedScore)
```

**Use cases**: Story-critical objects, player character, NPCs vs environment props.

### Attention-Aware LOD (VR/AR)

**Eye-tracking integration**: Allocate detail based on gaze position.

```
function GazeAwareLOD(object, camera, eyeTracker):
    # Foveal region (high detail)
    gazeDir = eyeTracker.gazeDirection
    objectDir = Normalize(object.center - camera.position)

    # Angular distance from gaze
    gazeAngle = acos(DotProduct(gazeDir, objectDir))

    # Foveal: 0-2° | Parafoveal: 2-5° | Peripheral: 5-10° | Far peripheral: >10°
    if gazeAngle < 2°:
        detailMultiplier = 1.0  # Full detail
    else if gazeAngle < 5°:
        detailMultiplier = 0.7  # 30% reduction
    else if gazeAngle < 10°:
        detailMultiplier = 0.4  # 60% reduction
    else:
        detailMultiplier = 0.2  # 80% reduction

    adjustedThreshold = object.lodThreshold * detailMultiplier

    return SelectLODByThreshold(adjustedThreshold)
```

**Performance gain**: 40-60% polygon reduction with imperceptible quality loss in VR.

---

## Popping Prevention Techniques

### Geomorphing

**Continuous LOD**: Smoothly interpolate vertex positions between LOD levels.

```
function GeomorphLOD(object, transitionProgress):
    currentMesh = object.lodLevels[object.currentLOD]
    nextMesh = object.lodLevels[object.nextLOD]

    # For each vertex in lower-detail mesh
    for i in 0 to nextMesh.vertexCount:
        # Find corresponding vertex in higher-detail mesh
        sourceVertex = currentMesh.vertices[nextMesh.vertexMap[i]]
        targetVertex = nextMesh.vertices[i]

        # Interpolate position
        morphedVertex = lerp(sourceVertex, targetVertex, transitionProgress)

        renderMesh.vertices[i] = morphedVertex

    return renderMesh
```

**Cost**: Extra vertex shader computation, doubled memory during transition.
**Benefit**: Eliminates visible popping entirely.

### Alpha Fading

**Discrete LOD variant**: Cross-fade between LOD levels using alpha blending.

```
function AlphaFadeLOD(object, transitionProgress):
    # Render both LOD levels simultaneously
    alpha_current = 1.0 - transitionProgress
    alpha_next = transitionProgress

    RenderMesh(object.currentLOD, alpha_current)
    RenderMesh(object.nextLOD, alpha_next)
```

**Cost**: 2x draw calls during transition (typically 5-10 frames).
**Benefit**: Works with any mesh, no vertex correspondence needed.

### Transition During Distraction

**Perceptual masking**: Switch LOD during camera motion or cuts.

```
function OpportunisticLODSwitch(object, camera, deltaTime):
    targetLOD = SelectLOD(object, camera)

    if targetLOD != object.currentLOD:
        # Check for masking opportunities
        cameraVelocity = Magnitude(camera.angularVelocity)
        screenMotion = Magnitude(object.screenVelocity)

        # High camera rotation or fast object motion
        if cameraVelocity > 30° / second or screenMotion > 50 pixels / frame:
            # Switch immediately (masked by motion)
            object.currentLOD = targetLOD
        else:
            # Queue for next opportunity or use geomorphing
            object.pendingLOD = targetLOD
```

**Result**: 90% of transitions occur during natural motion, remaining 10% use geomorphing.

---

## GPU-Parallel LOD Selection

### Compute Shader LOD Evaluation

**Modern approach**: Evaluate LOD for all objects in parallel on GPU.

```
// Compute shader pseudocode
struct ObjectLODData {
    vec3 center;
    float radius;
    float geometricError;
    int currentLOD;
    int maxLOD;
};

layout(std430) buffer ObjectBuffer {
    ObjectLODData objects[];
};

layout(local_size_x = 256) in;

void main() {
    uint objectID = gl_GlobalInvocationID.x;
    if (objectID >= numObjects) return;

    ObjectLODData obj = objects[objectID];

    // Calculate distance
    float distance = length(obj.center - cameraPosition);

    // Calculate screen-space error
    float screenError = (obj.geometricError * viewportHeight) /
                       (2.0 * distance * tan(fov * 0.5));

    // Select LOD
    int newLOD = obj.maxLOD;
    for (int level = 0; level <= obj.maxLOD; level++) {
        if (screenError < lodThresholds[level]) {
            newLOD = level;
            break;
        }
    }

    // Apply hysteresis
    if (newLOD < obj.currentLOD) {
        if (screenError > lodThresholds[newLOD] * 1.15)
            obj.currentLOD = newLOD;
    } else if (newLOD > obj.currentLOD) {
        if (screenError < lodThresholds[newLOD] / 1.15)
            obj.currentLOD = newLOD;
    }

    objects[objectID] = obj;
}
```

**Performance**: 1M objects evaluated in ~1ms on modern GPUs.

### Hierarchical LOD Selection

**Optimization**: Use spatial hierarchy to cull and batch LOD decisions.

```
function HierarchicalLODSelection(node, camera, viewport):
    # Early exit for culled nodes
    if not FrustumContains(camera.frustum, node.bounds):
        return

    # Calculate representative error for this hierarchy level
    screenError = CalculateScreenSpaceError(node, camera, viewport)

    # If entire subtree can use same LOD, batch it
    if screenError < node.minLODThreshold:
        SetSubtreeLOD(node, node.maxLOD)
        return
    else if screenError > node.maxLODThreshold:
        SetSubtreeLOD(node, 0)
        return

    # Otherwise, recurse to children
    for child in node.children:
        HierarchicalLODSelection(child, camera, viewport)
```

**Benefit**: 5-10x speedup for large scenes with spatial coherence.

---

## Performance Optimization

### LOD Bias

**User control**: Global quality multiplier.

```
lodBias = userQualitySettings  # 0.5 (low) to 2.0 (ultra)
adjustedThreshold = lodThreshold * lodBias
```

**Result**: Easy quality/performance trade-off for players.

### Predictive LOD

**Anticipate future needs**: Pre-load LOD before visible.

```
function PredictiveLOD(object, camera):
    # Current selection
    currentLOD = SelectLOD(object, camera)

    # Predicted position (1 second ahead)
    predictedCameraPos = camera.position + camera.velocity * 1.0

    # Predicted LOD
    predictedLOD = SelectLOD(object, predictedCameraPos)

    # Pre-fetch if transitioning to higher detail
    if predictedLOD < currentLOD:
        PrefetchLODLevel(object, predictedLOD)

    return currentLOD
```

**Benefit**: Eliminates load stutter during LOD transitions.

### Amortized LOD Updates

**Insight**: Not all objects need evaluation every frame.

```
function AmortizedLODUpdate(objects, frame):
    # Divide objects into buckets
    BUCKETS = 4
    bucketID = frame % BUCKETS

    for i in bucketID .. objects.length step BUCKETS:
        objects[i].lod = SelectLOD(objects[i], camera)
```

**Result**: 4x fewer LOD evaluations per frame, imperceptible staleness.

---

## Cross-References

**Related Concepts:**
- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - LOD basics and types
- [concepts/02-visual-perception.md](../concepts/02-visual-perception.md) - Human visual system constraints

**Related Techniques:**
- [techniques/01-peripheral-degradation.md](../techniques/01-peripheral-degradation.md) - Attention-aware LOD
- [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md) - Continuous LOD rendering

**Related Algorithms:**
- [algorithms/00-bsp-construction.md](00-bsp-construction.md) - Spatial organization for efficient LOD
- [algorithms/03-heightfield-tessellation.md](03-heightfield-tessellation.md) - Real-time LOD for terrain

**Integration:**
- Used with occlusion culling, frustum culling, draw call batching
- Critical for open-world games, VR, CAD visualization

---

## Key Takeaways

1. **Screen-space error beats distance**: Projected pixel error provides resolution-independent, scale-invariant LOD selection. Target 1-2 pixels of error for imperceptible quality.

2. **Hysteresis prevents oscillation**: Use 10-20% margin between increase/decrease thresholds. Eliminates flickering at LOD boundaries.

3. **Popping is perceptually critical**: Geomorphing (vertex interpolation) or alpha fading (dual rendering) for smooth transitions. Alternatively, exploit motion masking during camera movement.

4. **View-dependent factors matter**: Viewing angle, velocity, and attention significantly affect perceived quality. Fast-moving objects can use 50% fewer polygons without visible degradation.

5. **GPU-parallel evaluation scales**: Compute shader LOD selection handles 1M+ objects in ~1ms. Essential for modern open-world rendering.

6. **Hierarchical culling amplifies performance**: Batch LOD decisions for spatially coherent objects. 5-10x speedup for large scenes.

7. **Temporal coherence is free performance**: Not all objects need per-frame updates. Stagger evaluations across 4-8 frames with imperceptible staleness. Frees 75-87% of LOD computation.

8. **Connection to ARR-COC-VIS**: LOD selection is relevance realization - allocating geometric detail based on transjective coupling between viewer and scene. Screen-space error is propositional (information content), attention is perspectival (salience), and adaptive thresholds are procedural (learned skills).
