# Adaptive Spatial Subdivision for LOD Allocation

**Dynamic knowledge addition**: 2025-01-30
**Source**: ARR-COC-VIS Dialogue 15, BSP/Quadtree algorithms, computational geometry
**Parent**: [00-bsp-construction.md](00-bsp-construction.md), [01-lod-selection.md](01-lod-selection.md)

---

## Overview

Adaptive spatial subdivision dynamically partitions images based on **content-driven boundaries** rather than fixed grids. Using BSP trees, quadtrees, and importance-gradient detection, subdivision creates irregular but semantically meaningful regions that respect natural boundaries in the image.

**Key Principle**: Split where important, merge where uniform → efficient representation

---

## Binary Space Partitioning (BSP) for Images

### Traditional BSP

**Classic use** ([00-bsp-construction.md](00-bsp-construction.md)):
- 3D graphics: Split polygons for rendering order
- Collision detection: Spatial indexing

**Adapted for Images**:
- Split image recursively
- Splitting criterion: Content complexity
- Result: Variable-sized regions

### Importance-Gradient Split Points

```python
def adaptive_bsp_split(region, importance_map, min_size=16):
    """Recursively split image based on importance gradients"""

    if region.width < min_size or region.height < min_size:
        return [region]  # Too small, stop

    # Compute importance gradient within region
    region_importance = importance_map[region.bounds]
    grad_x, grad_y = np.gradient(region_importance)

    # Find strongest gradient direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Determine split axis (perpendicular to strongest gradient)
    if np.mean(np.abs(grad_x)) > np.mean(np.abs(grad_y)):
        # Strong horizontal gradient → split vertically
        split_axis = 'vertical'
        gradient_profile = np.mean(grad_x, axis=0)
    else:
        # Strong vertical gradient → split horizontally
        split_axis = 'horizontal'
        gradient_profile = np.mean(grad_y, axis=1)

    # Find split point at maximum gradient
    split_point = np.argmax(np.abs(gradient_profile))

    # Split region
    if split_axis == 'vertical':
        left = Region(region.x, region.y, split_point, region.height)
        right = Region(region.x + split_point, region.y,
                      region.width - split_point, region.height)
    else:
        top = Region(region.x, region.y, region.width, split_point)
        bottom = Region(region.x, region.y + split_point,
                       region.width, region.height - split_point)

    # Recurse
    if split_axis == 'vertical':
        return (adaptive_bsp_split(left, importance_map, min_size) +
                adaptive_bsp_split(right, importance_map, min_size))
    else:
        return (adaptive_bsp_split(top, importance_map, min_size) +
                adaptive_bsp_split(bottom, importance_map, min_size))
```

**Splitting Strategy**:
1. Compute importance gradient
2. Split perpendicular to strongest gradient (cuts across boundary)
3. Split at maximum gradient (right at boundary)
4. Recurse on subregions

**Result**: Regions naturally align with content boundaries (edges between objects)

---

## Quadtree Subdivision

### Standard Quadtree

**Recursive 4-way split**:
```python
class QuadTreeNode:
    def __init__(self, bounds, depth=0, max_depth=5):
        self.bounds = bounds  # (x, y, width, height)
        self.depth = depth
        self.children = None  # [NW, NE, SW, SE] if split
        self.is_leaf = True

    def split(self):
        """Split into 4 quadrants"""
        x, y, w, h = self.bounds
        hw, hh = w // 2, h // 2

        self.children = [
            QuadTreeNode((x, y, hw, hh), self.depth + 1),         # NW
            QuadTreeNode((x + hw, y, w - hw, hh), self.depth + 1), # NE
            QuadTreeNode((x, y + hh, hw, h - hh), self.depth + 1), # SW
            QuadTreeNode((x + hw, y + hh, w - hw, h - hh), self.depth + 1) # SE
        ]
        self.is_leaf = False
```

### Relevance-Based Quadtree

```python
def build_relevance_quadtree(image, importance_map, query, budget=273):
    """Build quadtree splitting high-importance regions"""

    root = QuadTreeNode((0, 0, image.width, image.height))
    nodes = [root]
    leaves = []

    while nodes:
        node = nodes.pop(0)

        # Compute region importance
        region_imp = importance_map[node.bounds]
        region_score = compute_relevance(region_imp, query)

        # Compute region variance (complexity)
        region_variance = np.var(region_imp)

        # Split criteria:
        # 1. High importance AND high variance (complex important region)
        # 2. Not too deep (prevent infinite recursion)
        # 3. Not too small (min size limit)

        should_split = (
            region_score > importance_threshold and
            region_variance > variance_threshold and
            node.depth < max_depth and
            min(node.bounds[2], node.bounds[3]) > min_size
        )

        if should_split:
            node.split()
            nodes.extend(node.children)
        else:
            leaves.append(node)

    # Now have variable-sized regions in leaves
    # Allocate tokens based on importance
    return allocate_tokens_to_leaves(leaves, budget)
```

**Adaptive Behavior**:
- Important + complex regions → Split (need detail)
- Important + uniform regions → Don't split (waste of tokens)
- Unimportant regions → Don't split regardless of variance

### Quadtree Pruning

```python
def prune_quadtree_by_relevance(root, importance_threshold):
    """Merge low-importance siblings back together"""

    def should_merge(node):
        if node.is_leaf or node.children is None:
            return False

        # Check if all children are leaves
        if not all(child.is_leaf for child in node.children):
            return False

        # Compute children importance scores
        child_scores = [
            compute_importance(child.bounds, importance_map)
            for child in node.children
        ]

        # If all children have low importance, merge them
        if all(score < importance_threshold for score in child_scores):
            return True

        return False

    def merge_children(node):
        """Convert node back to leaf"""
        node.children = None
        node.is_leaf = True

    # Bottom-up traversal
    def prune_recursive(node):
        if node.is_leaf:
            return

        # Recurse on children first
        for child in node.children:
            prune_recursive(child)

        # After recursion, check if should merge
        if should_merge(node):
            merge_children(node)

    prune_recursive(root)
    return root
```

**Pruning Benefits**:
- Reduces number of regions
- Saves tokens on unimportant areas
- Concentrates budget on important regions

---

## K-d Tree Subdivision

### Alternating Axis Splits

```python
class KDNode:
    def __init__(self, bounds, axis='x', depth=0):
        self.bounds = bounds
        self.axis = axis  # 'x' or 'y'
        self.split_value = None
        self.left = None
        self.right = None
        self.is_leaf = True

def build_kd_tree(image, importance_map, depth=0, max_depth=8):
    """K-d tree with alternating x/y splits"""

    # Determine split axis (alternate)
    axis = 'x' if depth % 2 == 0 else 'y'

    # Find median importance along axis
    if axis == 'x':
        importance_profile = np.mean(importance_map, axis=0)  # Average over y
    else:
        importance_profile = np.mean(importance_map, axis=1)  # Average over x

    # Find split point (median for balanced tree)
    split_value = len(importance_profile) // 2

    # Or split at gradient maximum (content-aware)
    # split_value = np.argmax(np.abs(np.gradient(importance_profile)))

    node = KDNode(image.bounds, axis, depth)
    node.split_value = split_value

    # Split region
    if axis == 'x':
        left_bounds = (image.x, image.y, split_value, image.height)
        right_bounds = (image.x + split_value, image.y,
                       image.width - split_value, image.height)
    else:
        left_bounds = (image.x, image.y, image.width, split_value)
        right_bounds = (image.x, image.y + split_value,
                       image.width, image.height - split_value)

    # Recurse if not too deep
    if depth < max_depth:
        node.left = build_kd_tree(Region(left_bounds), importance_map, depth + 1)
        node.right = build_kd_tree(Region(right_bounds), importance_map, depth + 1)
        node.is_leaf = False

    return node
```

**K-d Tree Properties**:
- Balanced (similar to BSP)
- Alternating splits (prevents skew)
- Efficient spatial queries
- Good for nearest-neighbor search

---

## Stratified Sampling with Subdivision

### Tiered Allocation via Quadtree

```python
def stratified_quadtree_allocation(image, importance_map, budget=273):
    """Combine quadtree subdivision with tiered allocation"""

    # Build quadtree
    root = build_relevance_quadtree(image, importance_map)

    # Collect all leaf nodes
    leaves = get_all_leaves(root)

    # Score each leaf
    leaf_scores = [compute_relevance(leaf.bounds, importance_map)
                   for leaf in leaves]

    # Sort by score
    sorted_leaves = sorted(zip(leaves, leaf_scores),
                          key=lambda x: x[1], reverse=True)

    # Assign to tiers (like Homunculus Protocol)
    tier_allocations = [
        {'count': 20, 'tokens_per': 8},  # High tier
        {'count': 30, 'tokens_per': 3},  # Mid tier
        {'count': 23, 'tokens_per': 1},  # Low tier
    ]

    tokens = []
    idx = 0

    for tier in tier_allocations:
        tier_leaves = sorted_leaves[idx:idx + tier['count']]
        idx += tier['count']

        for leaf, score in tier_leaves:
            # Extract tokens from leaf region
            region_tokens = encode_region(
                leaf.bounds,
                num_tokens=tier['tokens_per']
            )
            tokens.extend(region_tokens)

    return tokens  # Total: 20*8 + 30*3 + 23*1 = 273
```

**Benefits**:
- Adaptive regions (quadtree)
- Fixed budget (tiered allocation)
- Best of both worlds!

---

## Content-Aware Boundaries

### Edge-Aligned Splitting

```python
def edge_aware_split(region, edge_map):
    """Split along strong edges"""

    # Compute edge strength in region
    region_edges = edge_map[region.bounds]

    # Sum edge strength along potential split lines
    horizontal_edge_sums = np.sum(region_edges, axis=1)  # Each row
    vertical_edge_sums = np.sum(region_edges, axis=0)    # Each column

    # Find strongest edge line
    max_h_edge = np.max(horizontal_edge_sums)
    max_v_edge = np.max(vertical_edge_sums)

    if max_h_edge > max_v_edge:
        # Split horizontally along strongest horizontal edge
        split_row = np.argmax(horizontal_edge_sums)
        return 'horizontal', split_row
    else:
        # Split vertically along strongest vertical edge
        split_col = np.argmax(vertical_edge_sums)
        return 'vertical', split_col
```

**Why Edge-Aligned?**:
- Edges = object boundaries
- Splitting along edges → regions contain whole objects
- Similar to semantic atlas but cheaper (no SAM needed)

### Perceptual Grouping

```python
def perceptual_grouping_split(region, image):
    """Split based on Gestalt principles"""

    # Extract region
    region_img = image[region.bounds]

    # Compute grouping cues
    color_coherence = compute_color_variance(region_img)
    texture_coherence = compute_texture_variance(region_img)
    proximity = compute_spatial_clustering(region_img)

    # If region has multiple distinct groups, split
    if has_multiple_groups(color_coherence, texture_coherence, proximity):
        # Find boundary between groups
        split_line = find_grouping_boundary(region_img)
        return split_line
    else:
        # Single coherent group, don't split
        return None
```

**Gestalt Principles**:
- Similarity (color, texture)
- Proximity (spatial clustering)
- Continuity (smooth contours)
- Closure (completed shapes)

**Application**: Split when region contains perceptually distinct groups

---

## RoPE 2D for Region Position Encoding

### Rotary Position Embedding for Regions

```python
def rope_2d_for_regions(regions, image_shape):
    """Encode region positions using RoPE 2D"""

    def compute_rope_2d(x, y, dim):
        """RoPE for 2D coordinates"""
        # Frequency bands
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Position encoding
        pos_x = x * freqs
        pos_y = y * freqs

        # Interleave sin/cos
        emb_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=-1).flatten()
        emb_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=-1).flatten()

        # Concatenate
        return torch.cat([emb_x, emb_y])

    region_embeddings = []

    for region in regions:
        # Normalize coordinates to [0, 1]
        center_x = (region.x + region.width / 2) / image_shape[0]
        center_y = (region.y + region.height / 2) / image_shape[1]

        # Compute RoPE
        pos_emb = compute_rope_2d(center_x, center_y, dim=256)

        region_embeddings.append(pos_emb)

    return torch.stack(region_embeddings)
```

**Why RoPE for Regions?**:
- Preserves spatial relationships
- Relative position encoding (region A left of region B)
- Compatible with transformer architecture
- Used in Qwen3-VL (see qwen3vl-oracle)

---

## Comparison: Grid vs BSP vs Quadtree

| Aspect | Grid | BSP | Quadtree | K-d Tree |
|--------|------|-----|----------|----------|
| **Structure** | Regular | Irregular | Hierarchical | Alternating |
| **Splits** | Fixed | Content | Recursive 4-way | Recursive 2-way |
| **Boundary Respect** | ❌ Fragments | ✅ Edges | ⚠️ Partial | ⚠️ Partial |
| **Complexity** | O(1) | O(N log N) | O(N log N) | O(N log N) |
| **Memory** | O(N) | O(N) | O(N log N) | O(N log N) |
| **Flexibility** | Low | High | Medium | Medium |
| **Best For** | Uniform | Complex edges | Multi-scale | Balanced |

**When to use each**:
- **Grid**: Simple scenes, fast needed
- **BSP**: Complex boundaries, irregular objects
- **Quadtree**: Multi-scale content, hierarchical
- **K-d Tree**: Nearest-neighbor queries, spatial indexing

---

## Hybrid Subdivision Strategies

### Quadtree + Semantic Atlas

```python
def quadtree_atlas_hybrid(image, importance_map):
    """Coarse quadtree + SAM refinement"""

    # Stage 1: Coarse quadtree (fast, approximate)
    quadtree = build_quadtree(image, max_depth=3)  # 8×8 max
    coarse_regions = get_leaves(quadtree)

    # Stage 2: Score regions
    region_scores = [score_region(r, importance_map) for r in coarse_regions]

    # Stage 3: Refine high-importance regions with SAM
    refined_regions = []

    for region, score in zip(coarse_regions, region_scores):
        if score > threshold:
            # High importance → SAM refinement
            sam_regions = sam.generate(image[region.bounds])
            refined_regions.extend(sam_regions)
        else:
            # Low importance → keep coarse
            refined_regions.append(region)

    return refined_regions
```

**Benefits**:
- Fast coarse structure (quadtree)
- Semantic refinement where needed (SAM)
- Computational efficiency

### BSP + Grid Hybrid

```python
def bsp_grid_hybrid(image, importance_map):
    """BSP for foreground, grid for background"""

    # Detect foreground (high importance)
    foreground_mask = importance_map > threshold

    # BSP on foreground
    foreground_regions = bsp_split(
        image,
        mask=foreground_mask,
        split_criterion='gradient'
    )

    # Grid on background
    background_mask = ~foreground_mask
    background_patches = uniform_grid_sample(
        image,
        mask=background_mask,
        patch_size=32
    )

    return foreground_regions + background_patches
```

---

## Open Research Questions

1. **Optimal stopping criterion**: When to stop subdividing?
2. **Split vs merge**: Top-down (split) vs bottom-up (merge) better?
3. **Learning split decisions**: Can we learn splitting policy?
4. **Subdivision + transformer**: How to integrate with attention?
5. **3D extension**: Octrees for video/3D scenes?

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [00-bsp-construction.md](00-bsp-construction.md) - BSP tree fundamentals
- [01-lod-selection.md](01-lod-selection.md) - LOD calculation
- [concepts/01-bsp-btree-basics.md](../concepts/01-bsp-btree-basics.md) - BTree structures
- [techniques/00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](../techniques/00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Tiered allocation
- [integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md) - Relevance scoring

**Other Oracles**:
- **computer-vision-foundation-oracle**: SAM segmentation
- **qwen3vl-oracle**: RoPE 2D position encoding
- **vision-image-patching-oracle**: Adaptive patch strategies

---

**Last Updated**: 2025-01-30
**Status**: Synthesis from ARR-COC-VIS Dialogue 15, BSP/Quadtree algorithms
**Relevance**: ★★★★☆ (Content-aware alternative to fixed grids)
