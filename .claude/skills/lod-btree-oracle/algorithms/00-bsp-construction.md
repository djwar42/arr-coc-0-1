# BSP Tree Construction Algorithms

**Binary Space Partitioning tree generation, polygon splitting, and optimization strategies**

---

## Overview

Binary Space Partitioning (BSP) trees recursively subdivide 3D space using hyperplanes to create a hierarchical spatial data structure. Originally developed for visible surface determination in computer graphics, BSP trees solve the painter's algorithm problems of sorting polygons and handling overlaps. The construction algorithm transforms an unsorted list of polygons into a binary tree where each node contains a partitioning plane and references to polygons in front of and behind that plane.

**Key challenge**: Choosing optimal partitioning planes to minimize polygon splits while maintaining tree balance.

**Historical context**: Developed by Fuchs, Kedem, and Naylor (1980) for real-time rendering in flight simulators, later popularized by id Software's Doom (1993) and Quake engines.

---

## Primary Sources

**Core Algorithm:**
- `source-documents/06-Binary space partitioning - Wikipedia.md` - Complete BSP construction algorithm, traversal, historical development, and applications

**Foundational Papers:**
- Fuchs, Kedem, Naylor (1980) - "On Visible Surface Generation by A Priori Tree Structures"
- Naylor's PhD thesis (1981) - First empirical data on tree size and polygon splitting
- Naylor (1993) - "Constructing Good Partitioning Trees" - optimal tree characteristics

---

## Key Concepts

### What is a BSP Tree?

A BSP tree is a binary tree data structure where:
- **Nodes** contain a partitioning hyperplane (defined by a polygon)
- **Left subtree** contains polygons behind the plane
- **Right subtree** contains polygons in front of the plane
- **Leaf nodes** contain polygons that can be rendered in arbitrary order

### Hyperplane Partitioning

In 3D graphics, hyperplanes are defined by polygons:
- **Front side**: Direction the polygon normal points
- **Back side**: Opposite direction
- **Coincident**: Polygons lying exactly in the plane

The plane equation: `Ax + By + Cz + D = 0` where `(A, B, C)` is the normal vector.

### Polygon Classification

For each polygon relative to partitioning plane:
1. **Wholly in front**: All vertices have positive distance to plane
2. **Wholly behind**: All vertices have negative distance to plane
3. **Intersecting**: Vertices on both sides - requires splitting
4. **Coincident**: All vertices lie in plane (distance ≈ 0)

### Convexity and Splitting

**Key property**: Splitting a convex polygon with a plane produces two convex polygons.

**Splitting algorithm**:
1. Find intersection points where polygon edges cross the plane
2. Create two new polygons from the split segments
3. Preserve texture coordinates and attributes via interpolation

---

## Algorithm Details

### Recursive BSP Construction

**Input**: Unsorted list of polygons
**Output**: BSP tree with hierarchical spatial organization

```
function BuildBSPTree(polygonList):
    if polygonList is empty:
        return null

    # Step 1: Choose partitioning polygon
    partitionPoly = SelectPartitioningPolygon(polygonList)

    # Step 2: Create node
    node = new BSPNode()
    node.plane = ExtractPlane(partitionPoly)
    node.polygons = [partitionPoly]

    # Step 3: Classify and distribute remaining polygons
    frontList = []
    backList = []

    for poly in polygonList (excluding partitionPoly):
        classification = ClassifyPolygon(poly, node.plane)

        if classification == FRONT:
            frontList.add(poly)
        else if classification == BACK:
            backList.add(poly)
        else if classification == COINCIDENT:
            node.polygons.add(poly)
        else if classification == INTERSECTING:
            (frontPoly, backPoly) = SplitPolygon(poly, node.plane)
            frontList.add(frontPoly)
            backList.add(backPoly)

    # Step 4: Recursively build subtrees
    node.front = BuildBSPTree(frontList)
    node.back = BuildBSPTree(backList)

    return node
```

### Polygon Classification Algorithm

```
function ClassifyPolygon(polygon, plane):
    EPSILON = 0.001  # Tolerance for floating-point comparison

    numFront = 0
    numBack = 0
    numCoincident = 0

    for vertex in polygon.vertices:
        distance = DotProduct(plane.normal, vertex) + plane.D

        if distance > EPSILON:
            numFront++
        else if distance < -EPSILON:
            numBack++
        else:
            numCoincident++

    # Determine overall classification
    if numFront > 0 and numBack == 0:
        return FRONT
    else if numBack > 0 and numFront == 0:
        return BACK
    else if numFront == 0 and numBack == 0:
        return COINCIDENT
    else:
        return INTERSECTING
```

### Polygon Splitting Algorithm

**Critical operation**: Intersecting polygons must be split cleanly.

```
function SplitPolygon(polygon, plane):
    EPSILON = 0.001
    frontVertices = []
    backVertices = []

    for i in 0 to polygon.numVertices:
        v1 = polygon.vertices[i]
        v2 = polygon.vertices[(i + 1) % polygon.numVertices]

        d1 = DotProduct(plane.normal, v1) + plane.D
        d2 = DotProduct(plane.normal, v2) + plane.D

        # Add v1 to appropriate list
        if d1 > EPSILON:
            frontVertices.add(v1)
        else if d1 < -EPSILON:
            backVertices.add(v1)
        else:
            frontVertices.add(v1)
            backVertices.add(v1)

        # Check if edge crosses plane
        if (d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON):
            # Calculate intersection point
            t = d1 / (d1 - d2)
            intersection = v1 + t * (v2 - v1)

            frontVertices.add(intersection)
            backVertices.add(intersection)

    frontPoly = new Polygon(frontVertices)
    backPoly = new Polygon(backVertices)

    return (frontPoly, backPoly)
```

### Partitioning Plane Selection

**Most critical decision**: Choice of partition polygon dramatically affects tree quality.

**Selection criteria**:

1. **Minimize splits**: Choose polygon that splits fewest others
2. **Balance tree**: Prefer polygons that divide roughly equally
3. **Alignment**: Favor axis-aligned planes (faster classification)
4. **Cost function**: Combine multiple heuristics

**Simple selection heuristic**:
```
function SelectPartitioningPolygon(polygonList):
    bestPoly = null
    bestScore = INFINITY

    for candidate in polygonList:
        plane = ExtractPlane(candidate)

        numSplits = 0
        numFront = 0
        numBack = 0

        for poly in polygonList:
            classification = ClassifyPolygon(poly, plane)
            if classification == INTERSECTING:
                numSplits++
            else if classification == FRONT:
                numFront++
            else if classification == BACK:
                numBack++

        # Cost function: prioritize minimizing splits
        # Secondary: balance front/back distribution
        balance = abs(numFront - numBack)
        score = numSplits * 8 + balance

        if score < bestScore:
            bestScore = score
            bestPoly = candidate

    return bestPoly
```

**Advanced heuristics** (Naylor 1993):
- Expected-case cost models instead of worst-case
- Probabilistic binary search tree analogy
- Huffman coding parallels
- Multi-resolution object representation

---

## Implementation Considerations

### Tree Balance vs Splits

**Trade-off**: Fewer splits vs better balance

**Balanced tree**: O(log n) traversal time
**Unbalanced tree**: O(n) traversal time in worst case

**Strategy**: Weight split cost heavily, but consider balance as tie-breaker.

### Floating-Point Precision

**Problem**: Rounding errors can cause incorrect classification.

**Solutions**:
- Use epsilon tolerance (typically 0.001 to 0.0001)
- Snap vertices to grid for robustness
- Handle degenerate cases explicitly

### Memory Management

**Polygon growth**: Final polygon count often 2-4x original.

**Space considerations**:
- Store only necessary plane information (normal + D)
- Share vertex data between polygons
- Use indexed geometry

**Typical memory**: 32-64 bytes per node (plane, polygon list, child pointers).

### Preprocessing Optimization

**BSP generation is offline preprocessing**:
- One-time cost for static geometry
- Can afford expensive optimization
- Trade generation time for runtime performance

**Typical generation time**:
- Simple scenes: Seconds
- Complex levels (10K+ polygons): Minutes
- Optimization passes can double build time but improve runtime 20-30%

---

## Advanced Techniques

### Auto-Partitioning (Axis-Aligned)

**Simpler alternative**: Use axis-aligned planes instead of polygon planes.

**Benefits**:
- Faster classification (single coordinate test)
- More predictable tree structure
- Better cache coherency

**Used in**: Quake engine hybrid approach (BSP + axis-aligned subdivision).

### Set Operations on BSP Trees

**CSG operations** (Thibault & Naylor 1987):
- Union, intersection, difference of solid objects
- Merge two BSP trees to form new BSP tree
- Real-time boolean operations

**Applications**:
- Level editing with "brushes" (Quake, Unreal)
- Dynamic destructible environments
- Exact collision detection O(log n * log n)

### Moving Objects

**Challenge**: BSP trees are static structures.

**Solutions**:
1. **Separate dynamic objects**: Render with Z-buffer after BSP background
2. **Tree merging**: Merge moving object's BSP with static environment (expensive)
3. **Loose BSP nodes**: Store dynamic objects in containing nodes

**Practical approach**: Hybrid rendering (static BSP + dynamic Z-buffer).

### 2D BSP Trees

**Simplification**: 2D scenes use line segments instead of polygons.

**Applications**:
- 2D game engines
- UI rendering
- Vector graphics
- Image compression (Radha 1993)

**Partitioning**: Use line (not plane) to split space.

---

## Optimization Strategies

### Multi-Pass Construction

**Basic approach**: Single pass with local decisions.

**Multi-pass refinement**:
1. **Initial pass**: Quick construction with simple heuristic
2. **Analysis pass**: Identify poorly balanced subtrees
3. **Refinement pass**: Rebuild problem areas with better partitioning

**Result**: 10-20% better tree quality, 2-3x longer build time.

### Bounding Volume Hierarchies

**Hybrid approach**: Use BSP for spatial partitioning, BVH for object grouping.

**Benefits**:
- Faster culling (test groups, not individual polygons)
- Better cache utilization
- Reduced tree depth

### Portal-Based Visibility

**Extension** (Teller 1992): Pre-compute potentially visible sets (PVS).

**Approach**:
- Divide scene into convex regions (rooms)
- Calculate which regions are visible from each room
- Store visibility data in BSP structure

**Performance**: Near-constant rendering cost regardless of scene complexity.

---

## Performance Characteristics

### Construction Time

**Naive algorithm**: O(n²) polygon classifications
**Optimized**: O(n log n) with spatial hashing

**Typical times** (1990s hardware):
- 1,000 polygons: ~1 second
- 10,000 polygons: ~30 seconds
- 100,000 polygons: ~10 minutes

**Modern hardware**: 10-100x faster, but scenes also larger.

### Tree Size

**Polygon explosion**: Number of polygons after splitting.

**Empirical data** (Naylor 1981, Space Shuttle model):
- Original: 5,174 polygons
- After BSP: 7,283 polygons (41% increase)
- Tree nodes: 12,457

**Good heuristics**: Keep explosion under 50%.

### Runtime Traversal

**Complexity**: O(log n) for balanced tree, O(n) worst case.

**Typical depth**: 15-25 levels for 10K polygon scene.

**Traversal cost**: ~10-15 plane tests per visible polygon.

---

## Cross-References

**Related Concepts:**
- [concepts/01-bsp-btree-basics.md](../concepts/01-bsp-btree-basics.md) - BSP tree fundamentals and spatial organization
- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - How BSP fits into LOD systems

**Related Techniques:**
- [techniques/02-occlusion-culling.md](../techniques/02-occlusion-culling.md) - BSP-based visibility determination
- [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md) - Modern alternatives to BSP

**Related Algorithms:**
- [algorithms/01-lod-selection.md](01-lod-selection.md) - View-dependent detail selection works with BSP structure

**Application Context:**
- Used in: Doom, Quake, Half-Life, Counter-Strike (via GoldSrc/Source engines)
- Modern alternatives: Octrees, BVH, GPU-driven culling

---

## Key Takeaways

1. **BSP construction is recursive partitioning**: Choose plane, split polygons, recurse on front and back sets.

2. **Partition selection is critical**: Minimize splits first, balance second. Good heuristics reduce polygon explosion from 100%+ to 40-50%.

3. **Polygon splitting must preserve convexity**: Intersection points calculated via linear interpolation along edges.

4. **Trade-offs are unavoidable**: No single heuristic produces optimal trees for all scenes. Expected-case models outperform worst-case analysis.

5. **Static preprocessing enables runtime performance**: One-time construction cost (seconds to minutes) enables O(log n) runtime queries and linear-time front-to-back traversal.

6. **Floating-point robustness matters**: Epsilon tolerance prevents misclassification. Typical epsilon: 0.001 for world-space coordinates.

7. **Modern relevance**: While less common in 2025 (GPU-driven pipelines dominate), BSP principles inform spatial acceleration structures, CSG operations, and visibility preprocessing. The core insight - hierarchical space subdivision for efficient queries - remains fundamental to computer graphics.
