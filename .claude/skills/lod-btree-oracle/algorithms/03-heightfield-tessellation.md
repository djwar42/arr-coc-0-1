# Heightfield Tessellation Algorithms

**Real-time watertight tessellation, crack prevention, GPU implementation, and view-dependent LOD**

---

## Overview

Heightfield tessellation converts discrete elevation data into continuous triangle meshes for rendering. The core challenge is maintaining visual quality (no cracks or T-junctions) while achieving real-time performance through view-dependent level of detail. Modern solutions use GPU compute shaders to maintain adaptive triangulations that are guaranteed watertight, incrementally updated each frame, and support unlimited subdivision for extreme zoom ranges.

**Key innovation**: Watertight triangulation during parallel GPU processing via graph coloring and adjacency tracking.

**Critical properties**: No visual artifacts, stable 60 FPS, dynamic heightfield updates, higher-order interpolation support.

---

## Primary Sources

**Core Paper:**
- `source-documents/19-Watertight Incremental Heightfield Tessellation - Research Unit of Computer Graphics _ TU Wien.md` - Cornel et al. (2022), TU Wien VRVis

**Key Contributions:**
- **Guaranteed watertight** triangulation (no T-junctions)
- **Incremental updates** (cache previous frame's result)
- **Unlimited subdivisions** (no maximum LOD restriction)
- **GPU-based** implementation (compute shaders)
- **Graph coloring** for thread-safe parallel processing

**Use Case:**
- Flood simulation visualization (Austria, 572 km × 293 km, 1m resolution)
- Multiple nested heightfields (terrain + water)
- Real-time updates (60 FPS)

---

## Key Concepts

### Watertight Triangulation

**Definition**: Every edge is shared by exactly two triangles (or one at boundaries).

**Problem**: T-junctions cause cracks.

```
T-junction (BAD):          Watertight (GOOD):
    A-------B                  A---C---B
    |       |                  |  / \  |
    |   C   |                  | /   \ |
    D-------E                  D-------E
```

**T-junction**: Point C on edge AB is not a vertex of triangle ABD.
**Crack**: Gap appears in rendered mesh as triangles don't share edge.

### Graph Coloring for Parallelism

**Insight**: Triangles of different colors never share edges.

**4-coloring** (initial triangulation):
- Color grid cells alternately
- Adjacent triangles have different colors
- Triangles of same color can be processed in parallel

**8-coloring** (after first subdivision):
- Subdivision creates 4 new triangle types (colors 4-7)
- Maintain property: adjacent triangles differ in color
- Still allows parallel processing in 8 sequential batches

### Incremental Tessellation

**Key optimization**: Reuse previous frame's triangulation.

**Pipeline**:
1. Previous frame: Triangulation cached in GPU memory
2. Camera moves: Calculate new LOD requirements
3. Update: Only subdivide/merge changed triangles
4. Render: Use updated triangulation

**Benefit**: Avoid full reconstruction (expensive sampling operations minimized).

### LOD Metrics

**Screen-space edge length** (simplest, used in source paper):

```
lodState = KEEP  # Default

maxEdgeLength = max(
    screenLength(triangle.edge0),
    screenLength(triangle.edge1),
    screenLength(triangle.edge2)
)

if maxEdgeLength > targetLength * 2:
    lodState = SUBDIVIDE
else if maxEdgeLength < targetLength / 2:
    lodState = MERGE
```

**Target length**: User-defined (typically 10 pixels).

**Alternative metrics**: Screen-space error, geometric error, curvature-based.

---

## Algorithm Details

### Data Structures

**Triangle representation**:
```
struct Triangle {
    // Semantic flags (1 byte)
    bool deleted;
    uint2 lodState;         // SUBDIVIDE, MERGE, KEEP
    uint3 graphColor;       // 0-7

    // Heightfield reference
    uint8 heightfieldIndex; // For multiple heightfields

    // Hierarchy
    int32 parentIndex;      // -1 if root
    int32 firstChildIndex;  // -1 if leaf

    // Adjacency (counter-clockwise order)
    int32[3] adjacentIndices;  // -1 if boundary
                               // [1] is hypothenuse neighbor

    // Geometry
    int32[3] vertexIndices;    // [1], [2] are hypothenuse endpoints
};
```

**Memory per triangle**: 32 bytes (optimized with bitfields).

**Vertex buffer**: Separate array of 3D positions + heightfield index.

**Free index buffers**: Track deleted triangles/vertices for reuse.

### Initial Triangulation

**Goal**: Create coarse regular grid of right triangles.

```
function InitialTriangulation(heightfield):
    # Determine grid size (target ~2000 cells)
    k = FindPowerOf2CellSize(heightfield, targetCells=2000)
    gridWidth = ceil(heightfield.width / k)
    gridHeight = ceil(heightfield.height / k)

    triangleBuffer = []

    # Create two triangles per cell
    for y in 0 to gridHeight:
        for x in 0 to gridWidth:
            # Alternate diagonal direction for 4-coloring
            if (x + y) % 2 == 0:
                # Diagonal from top-left to bottom-right
                t0 = CreateTriangle(
                    vertices: [(x,y), (x+1,y), (x,y+1)],
                    color: 0
                )
                t1 = CreateTriangle(
                    vertices: [(x+1,y+1), (x,y+1), (x+1,y)],
                    color: 1
                )
            else:
                # Diagonal from bottom-left to top-right
                t0 = CreateTriangle(
                    vertices: [(x,y+1), (x,y), (x+1,y+1)],
                    color: 2
                )
                t1 = CreateTriangle(
                    vertices: [(x+1,y), (x+1,y+1), (x,y)],
                    color: 3
                )

            # Set adjacency
            SetAdjacency(t0, t1, gridWidth, gridHeight, x, y)

            triangleBuffer.append(t0, t1)

    return triangleBuffer
```

**Result**: 2 * gridWidth * gridHeight triangles, 4-colored, watertight.

### Vertex Sampling

**Heightfield interpolation** (flexible interface):

```
function SampleHeight(x, y, heightfieldIndex):
    # Simple: Bilinear interpolation
    return BilinearSample(heightfields[heightfieldIndex], x, y)

    # Advanced: Adaptive bicubic (C1 continuity)
    # return AdaptiveBicubic(heightfields[heightfieldIndex], x, y)
```

**When to sample**:
1. **Initial triangulation**: Sample all vertices
2. **Subdivision**: Sample new vertex at edge midpoint
3. **Heightfield update**: Resample all existing vertices

**Cost**: Expensive for higher-order interpolation (justifies caching).

### LOD State Update

**Compute shader** (parallel for all triangles):

```
kernel UpdateLODState(triangleBuffer, camera, viewport):
    triangleID = get_global_id()
    triangle = triangleBuffer[triangleID]

    if triangle.deleted:
        return

    # Frustum culling
    if not FrustumIntersects(camera, triangle.bounds):
        triangle.lodState = KEEP
        return

    # Calculate screen-space edge lengths
    screenLengths[3]
    for i in 0 to 2:
        v0 = vertexBuffer[triangle.vertexIndices[i]]
        v1 = vertexBuffer[triangle.vertexIndices[(i+1)%3]]

        p0 = ProjectToScreen(v0, camera, viewport)
        p1 = ProjectToScreen(v1, camera, viewport)

        screenLengths[i] = length(p1 - p0)

    maxLength = max(screenLengths)

    # Determine LOD state
    if maxLength > targetLength * 2.0:
        triangle.lodState = SUBDIVIDE
    else if maxLength < targetLength / 2.0:
        triangle.lodState = MERGE
    else:
        triangle.lodState = KEEP

    triangleBuffer[triangleID] = triangle
```

### Subdivision Rules

**8 triangle types** emerge from recursive longest-edge bisection:

```
Types 0-3: Initial triangulation (4 colors)
Types 4-7: After first subdivision (4 new colors)

Subdivision rules:
- Type 0 → Type 4 (child 0) + Type 5 (child 1)
- Type 1 → Type 6 (child 0) + Type 7 (child 1)
- Type 2 → Type 4 (child 0) + Type 5 (child 1)
- Type 3 → Type 6 (child 0) + Type 7 (child 1)
- Type 4 → Type 0 (child 0) + Type 1 (child 1)
- Type 5 → Type 2 (child 0) + Type 3 (child 1)
- Type 6 → Type 0 (child 0) + Type 1 (child 1)
- Type 7 → Type 2 (child 0) + Type 3 (child 1)
```

**Key property**: Adjacent triangles always have consecutive color values (differ by 1).

**Subdivision constraint**: Only subdivide if neighbor along hypothenuse also subdivides (prevents T-junctions).

### Subdivision Algorithm

**Process triangles by graph color** (4 passes: colors 0, 2, 4, 6):

```
kernel Subdivide(triangleBuffer, color):
    triangleID = get_global_id()
    t0 = triangleBuffer[triangleID]

    # Check if this triangle should be processed
    if t0.deleted or t0.graphColor != color:
        return

    # Get neighbor along hypothenuse (adjacentIndices[1])
    t1 = triangleBuffer[t0.adjacentIndices[1]]

    # Validation checks
    if t1.deleted or
       t1.graphColor != color + 1 or
       t0.firstChildIndex != -1 or
       t1.firstChildIndex != -1 or
       t1.adjacentIndices[1] != triangleID:
        return  # Cannot subdivide

    # Check if subdivision needed (either triangle or neighbors want it)
    needSubdivision = (
        t0.lodState == SUBDIVIDE or
        t1.lodState == SUBDIVIDE or
        triangleBuffer[t0.adjacentIndices[0]].lodState == SUBDIVIDE or
        triangleBuffer[t0.adjacentIndices[2]].lodState == SUBDIVIDE or
        triangleBuffer[t1.adjacentIndices[0]].lodState == SUBDIVIDE or
        triangleBuffer[t1.adjacentIndices[2]].lodState == SUBDIVIDE
    )

    if not needSubdivision:
        return

    # === Perform subdivision ===

    # 1. Create new vertex at hypothenuse midpoint
    v0 = vertexBuffer[t0.vertexIndices[1]]
    v1 = vertexBuffer[t0.vertexIndices[2]]
    newVertex = (v0 + v1) / 2
    newVertex.z = SampleHeight(newVertex.x, newVertex.y, t0.heightfieldIndex)

    newVertexIndex = AllocateVertex()
    vertexBuffer[newVertexIndex] = newVertex

    # 2. Create four child triangles
    # Based on subdivision rules and adjacency relationships
    (t00, t01, t10, t11) = CreateChildTriangles(t0, t1, newVertexIndex)

    # 3. Update parent triangles
    t0.firstChildIndex = GetTriangleIndex(t00)
    t1.firstChildIndex = GetTriangleIndex(t10)

    # 4. Update neighbor adjacency
    UpdateNeighborAdjacency(t0.adjacentIndices[0], t0, t01)
    UpdateNeighborAdjacency(t0.adjacentIndices[2], t0, t00)
    UpdateNeighborAdjacency(t1.adjacentIndices[0], t1, t11)
    UpdateNeighborAdjacency(t1.adjacentIndices[2], t1, t10)

    # 5. Calculate LOD state for new triangles
    for child in [t00, t01, t10, t11]:
        child.lodState = CalculateLODState(child, camera, viewport)
```

**Synchronization**: Process each color group sequentially (GPU sync between kernel dispatches).

### Merging Algorithm

**Reverse of subdivision** (4 passes: colors 0, 2, 4, 7):

```
kernel Merge(triangleBuffer, color):
    triangleID = get_global_id()
    t = triangleBuffer[triangleID]

    if t.deleted or t.graphColor != color or t.parentIndex == -1:
        return

    # Get parent and neighbor parent
    t0 = triangleBuffer[t.parentIndex]
    t1 = triangleBuffer[t0.adjacentIndices[1]]

    # Get all four children
    t00 = triangleBuffer[t0.firstChildIndex]
    t01 = triangleBuffer[t0.firstChildIndex + 1]
    t10 = triangleBuffer[t1.firstChildIndex]
    t11 = triangleBuffer[t1.firstChildIndex + 1]

    # Validation checks
    if t1.deleted or
       t0.graphColor >= t1.graphColor or
       t00.firstChildIndex != -1 or  # Children are not leaves
       t01.firstChildIndex != -1 or
       t10.firstChildIndex != -1 or
       t11.firstChildIndex != -1 or
       t00.lodState != MERGE or  # All children want to merge
       t01.lodState != MERGE or
       t10.lodState != MERGE or
       t11.lodState != MERGE or
       t0.lodState == SUBDIVIDE or  # Parents don't want subdivision
       t1.lodState == SUBDIVIDE:
        return  # Cannot merge

    # === Perform merging ===

    # 1. Update parent adjacency (restore pre-subdivision state)
    t0.adjacentIndices[0] = t01.adjacentIndices[1]
    t0.adjacentIndices[2] = t00.adjacentIndices[2]
    t1.adjacentIndices[0] = t10.adjacentIndices[1]
    t1.adjacentIndices[2] = t11.adjacentIndices[2]

    # 2. Update neighbor adjacency
    UpdateNeighborAdjacency(t0.adjacentIndices[0], t01, t0)
    UpdateNeighborAdjacency(t0.adjacentIndices[2], t00, t0)
    UpdateNeighborAdjacency(t1.adjacentIndices[0], t10, t1)
    UpdateNeighborAdjacency(t1.adjacentIndices[2], t11, t1)

    # 3. Mark children as deleted
    for child in [t00, t01, t10, t11]:
        child.deleted = true
        FreeTriangleIndex(GetIndex(child))

    # 4. Free shared vertex
    sharedVertexIndex = t00.vertexIndices[0]  # New vertex from subdivision
    vertexBuffer[sharedVertexIndex].deleted = true
    FreeVertexIndex(sharedVertexIndex)

    # 5. Remove child references from parents
    t0.firstChildIndex = -1
    t1.firstChildIndex = -1
```

### Tessellation Loop

**Main update cycle**:

```
function TessellationLoop(triangleBuffer, camera, viewport):
    # 1. Update LOD state for all triangles
    UpdateLODState_Kernel(triangleBuffer, camera, viewport)

    iterationsRemaining = MAX_ITERATIONS  # Typically 10-20
    minChanges = 10  # Termination threshold

    while iterationsRemaining > 0:
        changesThisIteration = 0

        # 2. Subdivision (4 passes by color)
        for color in [0, 2, 4, 6]:
            numSubdivided = Subdivide_Kernel(triangleBuffer, color)
            changesThisIteration += numSubdivided

        # 3. Merging (4 passes by color)
        for color in [0, 2, 4, 7]:
            numMerged = Merge_Kernel(triangleBuffer, color)
            changesThisIteration += numMerged

        # 4. Check termination
        if changesThisIteration < minChanges:
            break

        iterationsRemaining -= 1

    # 5. Optional: Compact buffer (remove deleted triangles)
    if frameCount % COMPACT_INTERVAL == 0:
        CompactTriangleBuffer(triangleBuffer)

    return triangleBuffer
```

---

## Performance Optimizations

### Time Budget

**Guarantee stable frame rate**: Limit tessellation loop iterations.

```
function TessellationLoopWithBudget(triangleBuffer, maxTime):
    startTime = GetTime()

    while GetTime() - startTime < maxTime:
        changesThisIteration = 0

        # Subdivision passes
        for color in [0, 2, 4, 6]:
            if GetTime() - startTime > maxTime:
                break
            changesThisIteration += Subdivide_Kernel(triangleBuffer, color)

        # Merging passes
        for color in [0, 2, 4, 7]:
            if GetTime() - startTime > maxTime:
                break
            changesThisIteration += Merge_Kernel(triangleBuffer, color)

        if changesThisIteration < minChanges:
            break

    return triangleBuffer
```

**Typical budget**: 2-4 ms per frame (leaves 12-14 ms for rendering at 60 FPS).

### Buffer Compaction

**Problem**: Deleted triangles leave holes in buffer.

**Solution**: Periodically compact (every 60-120 frames).

```
kernel CompactTriangleBuffer(triangleBuffer):
    # Parallel stream compaction
    # Move non-deleted triangles to front of buffer

    triangleID = get_global_id()
    triangle = triangleBuffer[triangleID]

    if not triangle.deleted:
        # Compute new position via prefix sum
        newIndex = PrefixSum(triangleID, !deleted)
        compactedBuffer[newIndex] = triangle

        # Update references in neighbors
        for neighbor in triangle.adjacentIndices:
            # Atomic update neighbor's adjacency
            AtomicUpdate(neighbor, triangleID, newIndex)
```

**Benefit**: Maintains cache coherency, prevents unbounded growth.

### Asynchronous Execution

**Overlap frame N rendering with frame N+1 tessellation**:

```
function PipelinedRendering():
    # Frame 0: Initialize
    tessellationFuture = TessellationLoop_Async(triangleBuffer, camera0)

    for frame in 1 to N:
        # Render previous frame's tessellation
        Render(triangleBuffer)

        # Wait for previous tessellation to complete
        Wait(tessellationFuture)

        # Start tessellation for next frame
        tessellationFuture = TessellationLoop_Async(triangleBuffer, camera[frame])
```

**Benefit**: Hides tessellation latency (1-2 ms becomes "free").

---

## Implementation Details

### GPU Memory Layout

**Separate arrays** (better cache coherency than struct-of-arrays):

```
// Triangle buffer (8M elements = 256 MB)
array<uint8> triangleFlags;          // 8 MB
array<uint8> triangleHeightfieldIdx; // 8 MB
array<int32> triangleParents;        // 32 MB
array<int32> triangleFirstChild;     // 32 MB
array<int32x3> triangleAdjacency;    // 96 MB
array<int32x3> triangleVertices;     // 96 MB

// Vertex buffer (8M elements = 128 MB)
array<float3> vertexPositions;       // 96 MB
array<uint8> vertexHeightfieldIdx;   // 8 MB

// Free index buffers (for deleted elements)
array<int32> freeTriangleIndices;    // 32 MB
array<int32> freeVertexIndices;      // 32 MB

// Counters (atomic)
atomic<int32> freeTriangleCount;
atomic<int32> freeVertexCount;
```

**Total GPU memory**: ~376 MB (fixed allocation).

### Higher-Order Interpolation

**C0 continuity** (bilinear): Fast but visible facets.

**C1 continuity** (bicubic): Smoother, more expensive.

```
function AdaptiveBicubicSampling(heightfield, x, y):
    # Find containing cell in adaptive quadtree
    cell = heightfield.FindCell(x, y)

    # Get 16 control points (4×4 stencil)
    controlPoints = GatherControlPoints(cell, x, y)

    # Cubic Hermite interpolation
    # Requires derivatives at corners (computed once, cached)
    localX = (x - cell.x0) / cell.width
    localY = (y - cell.y0) / cell.height

    # Hermite basis functions
    h00 = (1 + 2*localX) * (1 - localX)^2
    h10 = localX * (1 - localX)^2
    h01 = localX^2 * (3 - 2*localX)
    h11 = localX^2 * (localX - 1)

    # Apply in both dimensions
    height = 0
    for i in 0 to 3:
        for j in 0 to 3:
            weight = BicubicWeight(h00, h10, h01, h11, i, j)
            height += controlPoints[i][j] * weight

    return height
```

**Cost**: 4-6x slower than bilinear, but cached vertex positions amortize cost.

### Multiple Heightfields

**Use case**: Terrain base + dynamic water layer.

```
function RenderMultipleHeightfields(heightfields):
    # Single unified triangle buffer
    # Each triangle has heightfieldIndex attribute

    for heightfield in heightfields:
        # Update only triangles belonging to this heightfield
        kernel UpdateVertices(triangleBuffer, heightfield.index):
            triangleID = get_global_id()
            triangle = triangleBuffer[triangleID]

            if triangle.heightfieldIndex != heightfield.index:
                return

            for vertexIndex in triangle.vertexIndices:
                vertex = vertexBuffer[vertexIndex]
                vertex.z = SampleHeight(
                    vertex.x, vertex.y, heightfield.index
                )
                vertexBuffer[vertexIndex] = vertex

    # Render entire triangle buffer (all heightfields)
    DrawTriangleBuffer(triangleBuffer)
```

**Benefit**: Unified tessellation, coherent LOD across layers.

---

## Cross-References

**Related Concepts:**
- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - LOD systems overview
- [concepts/01-bsp-btree-basics.md](../concepts/01-bsp-btree-basics.md) - Spatial tree structures

**Related Techniques:**
- [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md) - View-dependent mesh LOD
- [techniques/02-occlusion-culling.md](../techniques/02-occlusion-culling.md) - Visibility optimization

**Related Algorithms:**
- [algorithms/01-lod-selection.md](01-lod-selection.md) - LOD metric calculation
- [algorithms/02-terrain-synthesis.md](02-terrain-synthesis.md) - Heightfield generation

**Applications:**
- Flood simulation visualization (source paper)
- Flight simulators (terrain rendering)
- Open-world games (landscape streaming)
- GIS visualization (digital elevation models)

---

## Key Takeaways

1. **Watertight triangulation requires adjacency tracking**: Store references to adjacent triangles for each edge. Graph coloring enables parallel processing without race conditions. 8 sequential passes (4 subdivision + 4 merging) per iteration.

2. **Incremental updates are critical**: Caching previous frame's triangulation reduces cost from O(n) full reconstruction to O(k) incremental changes where k << n. Enables expensive interpolation methods (bicubic) by minimizing sampling operations.

3. **Longest-edge bisection prevents T-junctions**: Recursively split triangles along hypothenuse (longest edge). Only subdivide if neighbor also subdivides. Automatically maintains watertightness without complex stitching.

4. **Graph coloring enables GPU parallelism**: 4-coloring for initial triangulation, 8-coloring after subdivision. Triangles of same color never share edges → process concurrently. 8 sequential kernel dispatches per iteration.

5. **Time budgets guarantee stable performance**: Limit tessellation loop iterations based on time budget (2-4 ms). Convergence not required every frame - partial updates acceptable. Worst case: use previous frame's tessellation.

6. **Unlimited subdivision via recursive coloring**: No maximum LOD restriction (unlike hardware tessellation's 64x limit). Subdivision rules maintain valid 8-coloring indefinitely. Enables extreme zoom ranges (country-scale to 1m resolution).

7. **Flexible interpolation via sampler interface**: Decouples tessellation from heightfield sampling. Supports bilinear (fast), bicubic (smooth), adaptive quadtree (variable resolution). Vertex caching amortizes expensive sampling.

8. **Connection to relevance realization**: Tessellation is procedural knowing - learned skills (subdivision rules) applied dynamically. LOD state calculation is perspectival knowing (screen-space salience). Incremental updates are participatory knowing (agent-arena coupling through camera movement).
