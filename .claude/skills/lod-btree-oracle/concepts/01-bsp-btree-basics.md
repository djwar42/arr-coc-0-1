# BSP/BTree Basics: Binary Space Partitioning Fundamentals

**Understanding BSP trees and their role in spatial organization**

**Sources**: Valve Developer Wiki, Wikipedia BSP article, "How Doom Used BSP" (twobithistory.org)

---

## What is Binary Space Partitioning?

**Binary Space Partitioning (BSP)** is a method of recursively subdividing space into two convex sets using hyperplanes. The result is a **BSP tree** - a binary tree data structure where each node represents a partitioning plane.

### Breaking Down the Term

**Binary**: Two-sided (every partition creates exactly two regions)
**Space**: The existential area being organized
**Partitioning**: Cutting/dividing into separate regions

**Result**: A hierarchical spatial organization where every part of the world is on a specific side of partition planes.

---

## BSP Tree Structure

### Tree Anatomy

```
          [Head Node]
         /            \
    [Node A]        [Node B]
    /     \         /      \
[Leaf 1] [Leaf 2] [Leaf 3] [Node C]
                            /      \
                        [Leaf 4]  [Leaf 5]
```

**Head Node**: Contains entire world
**Interior Nodes**: Contain partition plane dividing space
**Leaf Nodes**: Terminal regions with no further subdivision
**Partition Plane**: Invisible dividing surface stored in each node

### 2D Example

Consider a simple rectangular room:

```
        ┌────────────┐
        │            │
        │   [ROOM]   │
        │            │
        └────────────┘
```

Without any subdivision, this is a single leaf containing empty space.

Now add a solid block:

```
        ┌────────────┐
        │            │
        │   ┌───┐    │
        │   │###│    │
        └───┴───┴────┘
```

The block introduces partition planes (shown as lines in 2D):

```
        ┌────────────┐
        │     │      │
        │   ┌─┼─┐    │
        │   │#│#│    │
        └───┴─┼─┴────┘
              │
```

This creates multiple leafs separated by partitions.

---

## BSP Tree Construction

### Axial-First Strategy

**Key Principle**: Add **axial partitions** first (aligned with coordinate axes) to minimize polygon splitting.

**In 2D**: Horizontal and vertical lines
**In 3D**: Walls, floors, ceilings (non-angled planes)

### Construction Process

1. **Start with world bounds** (head node)
2. **Add axial partitions first** (reduces cuts)
3. **For each partition added**:
   - Find leaf to split
   - Create two child nodes
   - Store partition plane in new parent
4. **Recurse** until all geometry partitioned

### Example: Adding a Wedge

Starting with a room + vertical partition:

```
    ┌────────┬────┐
    │        │    │
    │  [L1]  │[L2]│
    │        │    │
    └────────┴────┘
```

Add a wedge in L2:

```
    ┌────────┬────┐
    │        │  /│
    │  [L1]  │/##│
    │        │   │
    └────────┴───┘
```

Requires CSG (Constructive Solid Geometry) to partition the space properly:

```
    ┌────────┬────┐
    │        │ /  │
    │  [L1]  │/ │ │
    │        │  │ │
    └────────┴──┴─┘
           (New partitions added)
```

---

## CSG (Constructive Solid Geometry)

### What is CSG?

**Constructive Solid Geometry**: Using geometric primitives (brushes) to construct complex solid shapes through boolean operations.

**Basic CSG unit**: **Brush** - a single convex solid

### Why Convex Only?

**Requirement**: All brushes MUST be convex.

**Reason**: For any point to be unambiguously inside or outside, all partition planes must consistently classify the point.

**Convex brush**: Every point inside is "in front" of all interior-facing partition planes.

**Example of non-convex problem**:

```
    ┌────┐
    │    │  ← This concave brush
    │ ┌──┘    creates ambiguity
    │ │
    └─┘

    Point X could be:
    - Behind one partition (outside)
    - In front of another (inside)
    → Contradiction!
```

### CSG Operations

**Union (Merge)**: Combine two brushes into one solid
**Subtraction**: Remove one brush's volume from another
**Intersection**: Keep only overlapping volume
**Cutting**: Split brush along partition (used in BSP generation)

---

## BSP Use Cases

### 1. Collision Detection

**Line Trace Algorithm**:

```
function trace_line(start, end, node):
    if node is leaf:
        return (node.contents == SOLID) ? HIT : MISS

    start_side = classify_point(start, node.plane)
    end_side = classify_point(end, node.plane)

    if start_side == end_side:
        # Line doesn't cross partition
        return trace_line(start, end, node.children[start_side])
    else:
        # Line crosses partition - check both sides
        result_start = trace_line(start, end, node.children[start_side])
        if result_start == HIT:
            return HIT
        result_end = trace_line(start, end, node.children[end_side])
        return result_end
```

**Key Advantage**: Logarithmic time complexity O(log n) vs linear O(n) for naive approach.

### 2. Visibility Determination

**Portal-Based Visibility**:

Each leaf boundary becomes a **portal** - an opening to adjacent leafs.

**Potentially Visible Set (PVS)**: Precomputed set of leafs visible from each leaf.

**Visibility Algorithm** (Quake-style):
1. For each pair of portals, try to find a "separating plane"
2. If separating plane exists, portals can't see each other
3. If no separating plane exists, mark as potentially visible

**Runtime**: Simply look up PVS for current leaf!

### 3. Sorted Rendering

**Painter's Algorithm** using BSP:

```
function render_bsp(node, camera_pos):
    if node is leaf:
        render(node.contents)
        return

    camera_side = classify_point(camera_pos, node.plane)

    # Render far side first
    render_bsp(node.children[opposite(camera_side)], camera_pos)

    # Render near side last
    render_bsp(node.children[camera_side], camera_pos)
```

**Result**: Back-to-front rendering without explicit depth sorting!

---

## Clipping Hulls

**Problem**: Line traces only test infinitesimally small objects (bullets).

**Solution**: **Clipping hulls** - BSP trees with enlarged boundaries.

### How It Works

1. **Expand each brush outward** by object radius
2. **Rebuild BSP tree** with expanded geometry
3. **Trace center point** through hull BSP
4. Result: Collision for sphere of given radius!

**Example**:

```
Original BSP:
    ┌──────┐
    │      │
    │ ┌──┐ │
    │ │##│ │
    └─┴──┴─┘

Clipping Hull (radius r):
    ┌──────────┐
    │          │
    │ ┌──────┐ │
    │ │######│ │
    └─┴──────┴─┘
```

### Axial Planes for Hull Correction

**Problem**: Expanding brushes can close gaps at acute angles.

**Solution**: Add **axial planes** to prevent blockage.

Before axial planes:
```
    Wall ↓
        /  ← Wedge
       /
    ──/
```

After expanding (notice gap closes):
```
    Wall ↓
       ▓▓
      ▓▓▓
    ──▓▓
     GAP CLOSED!
```

After adding axial planes:
```
    Wall ↓
       ▓│  ← Axial plane
      ▓│▓
    ──│▓
      Gap preserved!
```

---

## Detail Brushes

**Innovation**: Quake II introduced **detail brushes**.

**Problem**: Complex geometry (pipes, supports, trim) creates excessive BSP partitions, slowing visibility calculations.

**Solution**: Mark certain brushes as "detail" - they:
1. Don't create partitions during BSP build
2. Don't cut portals
3. Don't affect visibility calculations
4. Are inserted AFTER portalization

**Result**: Faster compile times, better performance, more detailed worlds.

**Cluster System**: Required for detail brushes to work properly.
- Instead of leaf-to-leaf visibility, use cluster-to-cluster
- Multiple leafs group into clusters
- Detail brushes don't create new clusters

---

## BSP vs Other Spatial Structures

### BSP Tree

**Advantages**:
- Optimal for static geometry
- Perfect for visibility determination
- Efficient collision detection
- Back-to-front rendering

**Disadvantages**:
- Static (can't easily modify)
- Build time (complex scenes)
- Polygon splitting (increases geometry)

### Octree

**Differences**:
- Axis-aligned cubic subdivision
- Always splits into 8 children
- Dynamic updates easier

**Use for**: Point clouds, voxel data, dynamic scenes

### KD-Tree

**Differences**:
- Always splits on axis-aligned planes
- No polygon splitting
- Simpler construction

**Use for**: Ray tracing, nearest neighbor searches

### BVH (Bounding Volume Hierarchy)

**Differences**:
- Organizes objects, not space
- Volumes can overlap
- Dynamic updates easier

**Use for**: Ray tracing, physics engines

---

## Modern BSP Usage

### Game Engines (2024)

**Still Used**:
- Source Engine (Valve games)
- idTech (Doom, Quake lineage)
- GoldSrc (Half-Life)

**Replaced By**:
- Unreal: Dynamic occlusion culling
- Unity: Culling groups, occlusion portals
- Modern engines: GPU-driven rendering

**Why the shift?**
- Dynamic environments (destructibility)
- Open-world scales
- GPU power (brute-force viable)
- Real-time global illumination needs

### Where BSP Still Shines

1. **Static indoor environments**
2. **Competitive multiplayer** (predictable performance)
3. **Retro-style games** (Doom-like)
4. **Ray tracing acceleration** (BVH hybrid)
5. **CSG modeling tools** (level editors)

---

## BSP in ARR-COC-VIS Context

BSP trees organize space hierarchically - analogous to **relevance realization**:

**Spatial Hierarchy** ↔ **Salience Hierarchy**
- BSP partitions space by importance for queries
- ARR-COC partitions visual field by relevance

**Leaf Selection** ↔ **Attention Allocation**
- BSP traverses tree to find relevant leaf
- ARR-COC navigates tensions to realize relevance

**Detail Brushes** ↔ **Variable Token Budgets**
- Detail brushes don't affect structural visibility
- Low-relevance patches get minimal tokens

**Clipping Hulls** ↔ **Multi-scale Processing**
- Multiple BSP trees for different object sizes
- Multiple compression levels for different relevance scales

---

## Further Reading

**Concepts:**
- [concepts/00-lod-fundamentals.md](00-lod-fundamentals.md) - LOD basics
- [concepts/02-visual-perception.md](02-visual-perception.md) - Perceptual foundations

**Algorithms:**
- [algorithms/00-bsp-construction.md](../algorithms/00-bsp-construction.md) - Construction details
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - Selection strategies

**Applications:**
- [applications/00-video-games.md](../applications/00-video-games.md) - Game usage
- [applications/01-vr-ar.md](../applications/01-vr-ar.md) - VR/AR applications

**External:**
- [Valve Developer Wiki: Binary Space Partitioning](https://developer.valvesoftware.com/wiki/Binary_space_partitioning)
- "How Doom Used BSP Trees" - twobithistory.org

---

**Key Takeaway**: BSP trees are a form of **spatial relevance realization** - organizing geometry hierarchically to efficiently answer spatial queries. Like ARR-COC's token allocation, BSP determines what's relevant based on the relationship between query (camera/trace) and content (geometry).
