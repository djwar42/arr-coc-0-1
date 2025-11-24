---
summary: whereby Karpathy discovers the fundamental problem with uniform 64×64 grids fragmenting semantic units like text boxes awkwardly split across patches 47, 48, 49 each seeing only "Te", "xt ", "Box" respectively, realizing rigid geometric boundaries ignore semantic structure, proposing instead adaptive spatial partitioning where boundaries align with semantic regions through techniques like segment-anything masks, quadtree decomposition subdividing high-importance areas into smaller cells while merging low-importance areas into larger cells, Voronoi tessellation around semantic keypoints, or learned boundary detection, enabling the rigid grid to dissolve into semantically meaningful partitions that respect object boundaries, text regions, and visual structure rather than imposing arbitrary Cartesian coordinates that fragment coherent visual elements across multiple incompatible patches
---

# Part 15: Boundaries and Subdivisions
*Wherein Karpathy and the LOD Oracle discover adaptive spatial partitioning, and the rigid grid dissolves into semantic boundaries*

---

## Opening: The Fragmented Text Box

*Scene: The Dirac Sea. KARPATHY is staring at a clay tablet showing a document with a text box awkwardly split across three grid cells.*

**KARPATHY:** *[Frustrated, pointing at tablet]*

Look at this. Text box—one semantic unit—fragmented across patches 47, 48, and 49.

*[Sketches]*

```
Uniform 64×64 grid:
╔═══╦═══╦═══╗
║   ║   ║   ║
╠═══╬═══╬═══╣
║ Te║xt ║Box║  ← "Text Box" split across 3 patches!
╠═══╬═══╬═══╣
║   ║   ║   ║
╚═══╩═══╩═══╝

Each patch sees:
  Patch 47: "Te"
  Patch 48: "xt "
  Patch 49: "Box"

LLM has to reconstruct: "Oh, these 3 fragments form 'Text Box'"
```

The boundaries are **arbitrary**. They don't respect content. We're fighting against our own representation.

*[LOD ORACLE materializes, carrying tome labeled "BSP Trees 1980-2025"]*

**LOD ORACLE:** *[Looking at the tablet, nods slowly]*

You've hit the fundamental problem with uniform grids.

*[Opens tome to page showing BSP tree diagrams]*

Binary Space Partitioning. We solved this in rendering 40 years ago.

**Don't split uniformly. Split at boundaries.**

**KARPATHY:** *[Looks up]*

Split at boundaries? You mean... let the patches adapt to content?

**LOD ORACLE:** *[Sets down tome]*

Exactly. Your text box becomes ONE patch. A formula becomes its own region. The table gets subdivided where cells are.

*[Sketches new diagram]*

```
Adaptive subdivision:
┌─────────────────────┐
│ Title (1 patch)     │
├─────────────┬───────┤
│ Formula     │ Text  │  ← Each semantic region
│ (fine grid) │ Box   │     gets its own partition
│  ┌─┬─┐      │ (1    │
│  ├─┼─┤      │patch) │
│  └─┴─┘      │       │
├─────────────┴───────┤
│ Footer (1 patch)    │
└─────────────────────┘

Variable patch sizes!
Boundaries follow content, not arbitrary grid.
```

**KARPATHY:** *[Excited]*

This is... different from everything we've explored. Not vortices, not spirals. This is **adaptive geometry**.

How do we detect where to split?

---

## Exploration 1: Boundary Detection Methods

**LOD ORACLE:** *[Flipping through tome]*

That's the question. In rendering, we split along polygon edges—geometry is known.

For images? You need to **find** the boundaries.

*[Inscribes on tablet]*

```python
# Method 1: Edge Detection
def find_split_boundaries(image):
    """
    Split where edges are strongest.

    Intuition: Edges = boundaries between semantic regions.
    """
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Project to axes
    vertical_edges = edges.sum(axis=0)    # Column-wise sum
    horizontal_edges = edges.sum(axis=1)  # Row-wise sum

    # Peaks = strong boundaries
    vert_peaks = find_peaks(vertical_edges)
    horiz_peaks = find_peaks(horizontal_edges)

    return vert_peaks, horiz_peaks
```

**KARPATHY:** *[Studying]*

So we find edge peaks, split there. But edges don't always align with semantic boundaries.

A photo of a person—there are edges everywhere (hair, clothing, background). We'd over-subdivide.

**LOD ORACLE:** *[Nods]*

True. Edge detection is low-level. You need something semantic.

*[New tablet]*

```python
# Method 2: Segment Anything Model (SAM)
def detect_semantic_boundaries(image):
    """
    Use SAM to find object masks.

    Each mask = one semantic region.
    Masks give us precise boundaries!
    """
    from segment_anything import SamPredictor

    predictor = SamPredictor(sam_model)
    predictor.set_image(image)

    # Auto-generate masks
    masks = predictor.generate_masks()

    # Each mask is a semantic region
    # Compute bounding boxes
    bboxes = [mask_to_bbox(mask) for mask in masks]

    return bboxes  # [(x1,y1,x2,y2), ...]
```

**KARPATHY:** *[Pause]*

Wait. SAM. Segment Anything.

We already have SAM in DeepSeek-OCR's architecture! It's the encoder—processes image, outputs compressed features.

*[Excited]*

What if SAM's intermediate masks GUIDE subdivision? Use SAM twice:
1. Generate masks (detect regions)
2. Encode each region independently

**LOD ORACLE:** *[Intrigued]*

Dual-use SAM. I like it.

But SAM generates hundreds of masks. You can't have 200 patches—you said 273 tokens total, not 273 patches.

**KARPATHY:** *[Thinking]*

Filter by importance. SAM gives masks, we score them by query relevance, keep top-K.

Or... hierarchical. SAM gives masks at multiple scales (small objects, large objects). We choose granularity based on importance.

**LOD ORACLE:** *[Cautious]*

Possible. But now you're running SAM inference, computing masks, filtering, encoding...

What's the computational cost?

**KARPATHY:** *[Grimaces]*

Good point. SAM is expensive. ViT-H backbone, 600M parameters.

If we're running SAM anyway (for encoding), extracting masks is cheap. But if we need EXTRA SAM passes...

*[Looks at LOD Oracle]*

Is there a lighter way to detect boundaries?

---

## Exploration 2: Lightweight Boundary Detection

**LOD ORACLE:** *[New tablet]*

Importance gradients. You're already computing importance. Use its gradient to find boundaries.

```python
def importance_gradient_boundaries(importance_map):
    """
    High gradient in importance = boundary between regions.

    Cheap! Just derivatives of what you already have.
    """
    # importance_map: [H, W]

    grad_y, grad_x = np.gradient(importance_map)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold: strong gradients = boundaries
    boundaries = grad_magnitude > threshold

    # Find connected components of strong gradients
    # These are boundary lines

    return boundaries
```

**KARPATHY:** *[Nodding]*

This is lightweight. But gradients are noisy. We'll get boundaries everywhere.

Unless...

*[Sketches]*

What if we smooth the importance map first? Gaussian blur, then gradient. Reduces noise, keeps major boundaries.

```python
importance_smoothed = gaussian_blur(importance_map, sigma=5)
gradients = compute_gradient(importance_smoothed)
major_boundaries = gradients > high_threshold
```

**LOD ORACLE:** *[Approving]*

Better. And you can tune sigma—larger sigma = coarser boundaries, smaller = finer.

But I see another issue: **how do you go from boundary map to subdivision?**

Boundaries tell you WHERE to split. But you still need to build the tree structure.

---

## Exploration 3: Quadtree vs BSP Tree

**KARPATHY:** *[New direction]*

Okay, two approaches:

**Quadtree**: Always split into 4 quadrants (top-left, top-right, bottom-left, bottom-right)

**BSP Tree**: Split along ONE axis at a time (vertical or horizontal)

Which is better for documents?

**LOD ORACLE:** *[Opens tome to comparison page]*

Depends on content structure.

**Quadtree**:
```
Splits into 4 equal quadrants:
┌───────┬───────┐
│  TL   │  TR   │
├───────┼───────┤
│  BL   │  BR   │
└───────┴───────┘

Pros:
  - Simple, symmetric
  - Natural recursion (subdivide each quadrant)
  - Good for images with scattered content

Cons:
  - Forces 4-way split even if not needed
  - Doesn't align with rectilinear content (text, tables)
```

**BSP Tree**:
```
Splits along one axis:
┌─────────────┐         ┌─────────────┐
│             │    OR   │             │
├─────────────┤         │      │      │
│             │         │      │      │
└─────────────┘         └──────┴──────┘
 Horizontal split        Vertical split

Pros:
  - Flexible (choose best axis per region)
  - Aligns with documents (columns, rows)
  - Can create non-square regions

Cons:
  - More complex (decide axis at each split)
  - Harder to balance (tree depth can vary)
```

**KARPATHY:** *[Thinking out loud]*

Documents have structure. Columns of text (vertical split), rows of sections (horizontal split).

BSP feels right. But deciding WHICH axis to split...

*[Looks at LOD Oracle]*

How do rendering engines decide?

**LOD ORACLE:** *[Flips to algorithms section]*

Heuristics. For rendering:
- Split along longest axis (keep regions square-ish)
- Split where it minimizes polygon count on each side
- Split along major boundaries (walls, object edges)

For your case:
```python
def choose_split_axis(region, importance_map):
    """
    Decide: split horizontally or vertically?

    Heuristics:
      1. Aspect ratio (split long axis)
      2. Gradient direction (split perpendicular to strong gradient)
      3. Content flow (columns → split vertically)
    """
    height, width = region.shape

    # Heuristic 1: Split long axis
    if width > 1.5 * height:
        return 'vertical'
    elif height > 1.5 * width:
        return 'horizontal'

    # Heuristic 2: Gradient analysis
    grad_y, grad_x = np.gradient(importance_map[region])

    if grad_x.std() > grad_y.std():
        # Strong vertical gradients → split vertically
        return 'vertical'
    else:
        # Strong horizontal gradients → split horizontally
        return 'horizontal'
```

**KARPATHY:** *[Excited]*

I like heuristic 2. Gradients tell us where importance changes. Split perpendicular to that.

But wait—where ALONG the axis do we split? Center? Or at the boundary?

**LOD ORACLE:** *[Inscribing]*

At the boundary! That's the whole point.

```python
def find_split_position(region, importance_map, axis='vertical'):
    """
    Find best position along axis to split.

    Best = where gradient is strongest (major boundary).
    """
    if axis == 'vertical':
        # Sum gradients along columns
        proj = np.gradient(importance_map[region], axis=1).sum(axis=0)
    else:  # horizontal
        # Sum gradients along rows
        proj = np.gradient(importance_map[region], axis=0).sum(axis=1)

    # Find peak (strongest boundary)
    split_pos = np.argmax(np.abs(proj))

    return split_pos
```

Text box spans columns 20-60. Boundary at column 19 and 61.

Gradient will peak at 19 and 61. We split there, text box becomes one region!

---

## Exploration 4: Recursive Subdivision Algorithm

**KARPATHY:** *[Putting it together]*

Okay, let me sketch the full algorithm:

```python
def recursive_subdivide(region, importance_map, depth=0, max_depth=6):
    """
    Recursively subdivide region based on importance boundaries.

    Stopping criteria:
      - Max depth reached
      - Region is small (< min_size)
      - Region is homogeneous (low gradient variance)
    """
    # Base cases
    if depth >= max_depth:
        return [region]  # Stop, this is a leaf

    if region.width < min_size or region.height < min_size:
        return [region]  # Too small to subdivide

    # Check homogeneity
    grad = np.gradient(importance_map[region])
    if grad.std() < homogeneity_threshold:
        return [region]  # Uniform region, no need to split

    # Recursive case: subdivide
    axis = choose_split_axis(region, importance_map)
    split_pos = find_split_position(region, importance_map, axis)

    if axis == 'vertical':
        left = Region(region.x, region.y, split_pos, region.y + region.height)
        right = Region(split_pos, region.y, region.x + region.width, region.y + region.height)
    else:  # horizontal
        top = Region(region.x, region.y, region.x + region.width, split_pos)
        bottom = Region(region.x, split_pos, region.x + region.width, region.y + region.height)

    # Recurse
    return (
        recursive_subdivide(left, importance_map, depth+1, max_depth) +
        recursive_subdivide(right, importance_map, depth+1, max_depth)
    )

# Usage
full_image = Region(0, 0, width, height)
patches = recursive_subdivide(full_image, importance_map)

print(f"Created {len(patches)} adaptive patches (vs 4096 uniform)")
```

**LOD ORACLE:** *[Studying]*

Good structure. But I see problems.

**Problem 1**: You might get too many patches (over-subdivide) or too few (under-subdivide).

How do you control the count?

**KARPATHY:** *[Frowning]*

Hmm. Adjust max_depth? But that's global. Some regions need depth 8, others depth 2.

Maybe... adaptive stopping based on importance?

```python
# Add to stopping criteria:
importance_sum = importance_map[region].sum()
if importance_sum < importance_threshold:
    return [region]  # Low importance, stop subdividing
```

High-importance regions → subdivide deeply
Low-importance regions → stop early

**LOD ORACLE:** *[Nods]*

Better. But you still might get variable counts. 150 patches one image, 400 another.

Batching becomes hard again. Fixed 273 tokens was the whole point!

**KARPATHY:** *[Realizes]*

Oh. Right.

So we need to either:
1. **Prune** afterwards (merge low-importance regions until we have exactly 273)
2. **Sample** patches (if we get 500 patches, select top 273 by importance)
3. **Variable patches per region** (large region = more tokens, small region = fewer)

*[Pause]*

Option 3 brings us back to variable allocation...

**LOD ORACLE:** *[Gentle]*

Exactly. You're rediscovering the same trade-offs.

Adaptive subdivision is powerful, but it creates irregular structure. And irregular structure is hard to batch, hard to make differentiable.

*[New tablet]*

Let me show you a compromise.

---

## Exploration 5: Fixed Quadtree with Importance Pruning

**LOD ORACLE:** *[Inscribing]*

**Idea**: Build a full quadtree to fixed depth, then prune unimportant regions.

```python
class QuadNode:
    def __init__(self, bbox, depth=0):
        self.bbox = bbox  # (x, y, width, height)
        self.depth = depth
        self.children = []  # 4 children if subdivided, empty if leaf

    def is_leaf(self):
        return len(self.children) == 0

def build_full_quadtree(image_size, max_depth=6):
    """
    Build complete quadtree (every node subdivided to max_depth).

    Depth 6 → 4^6 = 4096 leaves (same as 64×64 grid!)
    """
    root = QuadNode(bbox=(0, 0, image_size, image_size), depth=0)

    def subdivide(node):
        if node.depth >= max_depth:
            return  # Leaf

        # Create 4 quadrants
        x, y, w, h = node.bbox
        hw, hh = w//2, h//2

        node.children = [
            QuadNode((x, y, hw, hh), node.depth + 1),      # Top-left
            QuadNode((x+hw, y, hw, hh), node.depth + 1),   # Top-right
            QuadNode((x, y+hh, hw, hh), node.depth + 1),   # Bottom-left
            QuadNode((x+hw, y+hh, hw, hh), node.depth + 1) # Bottom-right
        ]

        for child in node.children:
            subdivide(child)

    subdivide(root)
    return root

def prune_quadtree(root, importance_map):
    """
    Merge nodes with low importance.

    If all 4 children are leaves AND have low importance, merge them.
    """
    def can_merge(node):
        # Check if all children are leaves with low importance
        if not all(child.is_leaf() for child in node.children):
            return False

        total_importance = sum(
            importance_map[child.bbox].mean()
            for child in node.children
        )

        return total_importance / 4 < merge_threshold

    def prune(node):
        if node.is_leaf():
            return

        # Recurse first (bottom-up pruning)
        for child in node.children:
            prune(child)

        # Try to merge
        if can_merge(node):
            node.children = []  # Remove children, become leaf

    prune(root)
    return root
```

**KARPATHY:** *[Eyes lighting up]*

So we start with uniform 4096, then collapse low-importance regions back to their parents.

Formula region: high importance → stays subdivided (16 small patches)
Margin: low importance → collapses (1 large patch)

**Result**: Variable resolution, but structurally clean!

**LOD ORACLE:** *[Nods]*

And you can control final count:

```python
def prune_to_budget(root, importance_map, budget=273):
    """
    Prune until we have exactly 'budget' leaves.
    """
    leaves = get_all_leaves(root)

    while len(leaves) > budget:
        # Find merge that loses least importance
        best_merge = find_best_merge_candidate(root, importance_map)
        merge_node(best_merge)
        leaves = get_all_leaves(root)

    return root
```

**KARPATHY:** *[Thinking]*

This is clean. Quadtree structure = easy to implement. Pruning = adaptive.

But it's still axis-aligned splits. Your text box example—if it's at angle, quadtree won't help.

**LOD ORACLE:** *[Honest]*

True. Quadtree assumes rectilinear content. For rotated objects, you'd need oriented bounding boxes (OBB).

But documents? 99% rectilinear. Text, tables, figures—all axis-aligned.

Quadtree is appropriate for the domain.

---

## Exploration 6: Query-Driven Subdivision

**KARPATHY:** *[New idea]*

Everything so far is image-driven—subdivide based on image boundaries.

But what about query-driven?

*[Inscribes]*

```
Query: "What's the formula in the top-left?"

Response: Subdivide top-left region finely, rest coarsely.

╔═══════════╦═╦═╗
║ Fine      ║ ║ ║ ← Top-left: fine (query-relevant)
║  ┌──┬──┐  ║ ║ ║
║  ├──┼──┤  ║ ║ ║
║  └──┴──┘  ║ ║ ║
╠═══════════╩═╩═╣
║               ║ ← Bottom: coarse (query-irrelevant)
╚═══════════════╝
```

Subdivision depth controlled by query relevance!

**LOD ORACLE:** *[Intrigued]*

Foveation meets subdivision. Subdivide where query "looks".

How do you determine relevance before subdividing?

**KARPATHY:** *[Thinking]*

Chicken-and-egg. Need to subdivide to get features. Need features to determine relevance.

Maybe... coarse-to-fine?

```python
def query_driven_subdivision(image, query, max_depth=6):
    """
    Start coarse, subdivide query-relevant regions.

    Multi-pass:
      1. Divide into 4×4 grid (16 regions)
      2. Score each region by query relevance
      3. Subdivide top-K relevant regions
      4. Repeat until budget exhausted
    """
    # Pass 1: Coarse grid
    regions = create_grid(4, 4)  # 16 regions

    for depth in range(max_depth):
        # Score regions
        features = [encode_region(image, r) for r in regions]
        relevance = [query_relevance(f, query) for f in features]

        # Subdivide top-K
        top_k_indices = np.argsort(relevance)[-K:]

        new_regions = []
        for i, region in enumerate(regions):
            if i in top_k_indices:
                # Subdivide this region
                new_regions.extend(subdivide_quad(region))
            else:
                # Keep as-is
                new_regions.append(region)

        regions = new_regions

        if len(regions) >= budget:
            break

    return regions[:budget]
```

**LOD ORACLE:** *[Thoughtful]*

Iterative refinement. I like it.

But encoding each region separately—that's expensive. You're running vision encoder K times.

Unless...

**KARPATHY:** *[Realizes]*

Unless we use hierarchical features! Vision encoder outputs multi-scale features (like FPN).

Use low-res features for coarse relevance, high-res for fine.

```python
# Single encoder pass
feature_pyramid = vision_encoder(image)
# {scale_1: [16, 768], scale_2: [64, 768], scale_3: [256, 768]}

# Use coarse features for coarse regions
coarse_regions = grid_4x4
coarse_features = sample_features(feature_pyramid['scale_1'], coarse_regions)
relevance_coarse = score_relevance(coarse_features, query)

# Subdivide relevant regions, use finer features
fine_regions = subdivide(top_k_coarse_regions)
fine_features = sample_features(feature_pyramid['scale_2'], fine_regions)
```

**LOD ORACLE:** *[Impressed]*

Now you're thinking efficiently.

Feature pyramid = natural match for hierarchical subdivision.

And this is differentiable! Features flow from encoder, relevance is learned.

---

## Exploration 7: Segment Anything Integration

**KARPATHY:** *[Back to SAM idea]*

Okay, let's seriously consider SAM.

DeepSeek-OCR already uses SAM as encoder. What if we extract its intermediate masks?

**LOD ORACLE:** *[Opens tome to SAM section]*

SAM architecture:

```
Image → SAM Encoder (ViT-H) → Feature Map [64×64×256]
                              ↓
                        Mask Decoder → Masks

SAM encoder outputs dense features.
Mask decoder generates masks from features.
```

Your idea: Use masks to define regions?

**KARPATHY:** *[Nods]*

```python
def sam_guided_subdivision(image, sam_encoder, budget=273):
    """
    Use SAM masks as subdivision boundaries.

    1. Run SAM encoder → features
    2. Run mask generation → N masks
    3. Filter masks by importance/size
    4. Use top-K masks as regions
    """
    # Encode
    features = sam_encoder(image)  # [64, 64, 256]

    # Generate masks
    masks = sam_mask_generator(features)  # List of binary masks

    # Score masks
    mask_scores = []
    for mask in masks:
        # Score = importance within mask
        bbox = mask_to_bbox(mask)
        importance = compute_importance(features, bbox, query)
        mask_scores.append(importance)

    # Select top-K masks
    top_k_indices = np.argsort(mask_scores)[-budget:]
    selected_masks = [masks[i] for i in top_k_indices]

    # Convert masks to bboxes (regions)
    regions = [mask_to_bbox(mask) for mask in selected_masks]

    return regions
```

**LOD ORACLE:** *[Considering]*

Interesting. SAM gives you semantic boundaries for free (it's already running).

But SAM generates OBJECT masks. What if your document has 500 words? 500 masks?

You need to decide which to keep, which to merge.

**KARPATHY:** *[Thinking]*

Filter by area? Discard tiny masks (punctuation, small symbols).

Or hierarchical? SAM has multi-scale masks. Keep coarse masks, subdivide into fine where needed.

**LOD ORACLE:** *[Cautious]*

Possible. But SAM isn't trained for documents. It's trained on natural images.

Text, tables, formulas—these might not segment cleanly.

Worth trying, but don't assume it works.

**KARPATHY:** *[Nods]*

Fair. Experimental direction. Test on DocVQA, see if masks are useful.

---

## Exploration 8: Differentiability and Training

**LOD ORACLE:** *[New concern]*

We've talked implementation. But can you TRAIN this?

Subdivision based on image content—where do gradients flow?

**KARPATHY:** *[Pauses]*

Hmm. If split positions are determined by gradient peaks, that's not differentiable.

`argmax(gradient)` has zero gradient.

Unless...

**LOD ORACLE:** *[Inscribes]*

**Soft splits**. Instead of hard boundaries, use smooth weighting.

```python
def soft_subdivision(image, split_position, temperature=0.1):
    """
    Instead of hard split at position x:
      - Region A = [0, x)
      - Region B = [x, width)

    Use soft split:
      - Pixel i belongs to A with weight sigmoid((x - i) / temp)
      - Pixel i belongs to B with weight 1 - sigmoid((x - i) / temp)
    """
    x_coords = torch.arange(image.width)

    # Soft membership to region A
    weight_A = torch.sigmoid((split_position - x_coords) / temperature)
    weight_B = 1 - weight_A

    # Features for region A = weighted sum
    features_A = (image * weight_A.unsqueeze(-1)).sum(dim=1)
    features_B = (image * weight_B.unsqueeze(-1)).sum(dim=1)

    return features_A, features_B
```

Split position is continuous parameter. Sigmoid makes it differentiable.

**KARPATHY:** *[Excited]*

So during training, split positions are learned! Gradients flow through sigmoid.

During inference, we can sharpen (temperature → 0) to get hard splits.

**LOD ORACLE:** *[Nods]*

Exactly. Soft during training, hard during inference.

Similar to Gumbel-Softmax for discrete sampling.

**KARPATHY:** *[But then frowns]*

But now we're back to learning split positions. That's a lot of parameters.

For a quadtree depth 6, we have 2^6-1 = 63 internal nodes. Each needs a split position.

63 parameters isn't terrible, but... they're not shared across images. Each image has different optimal splits.

**LOD ORACLE:** *[Thoughtful]*

Right. This works if you're learning a FIXED subdivision (same for all images).

For adaptive (per-image) subdivision, you need to compute split positions from image features.

```python
class LearnedSubdivision(nn.Module):
    def __init__(self):
        super().__init__()
        # Learn how to compute split positions from features
        self.split_predictor = nn.Linear(768, 1)  # feature → split position

    def forward(self, image_features):
        # Predict split position from features
        split_pos = self.split_predictor(image_features).sigmoid()
        # sigmoid → [0,1] range

        # Use soft split
        regions = soft_subdivision(image_features, split_pos)

        return regions
```

**KARPATHY:** *[Nods slowly]*

So the subdivision is learned, but it's adaptive per image (split_pos depends on features).

Differentiable end-to-end.

But complex. And we're learning subdivision on top of importance on top of encoding...

*[Looks at LOD Oracle]*

Is this over-engineering?

**LOD ORACLE:** *[Honest]*

Maybe. You're adding complexity to solve boundary fragmentation.

Simple question: **How often does boundary fragmentation hurt accuracy?**

If text boxes fragmented across 3 patches only hurts 5% of examples, is complex subdivision worth it?

Or can you get 80% of the benefit with simple quadtree pruning (no learning)?

**KARPATHY:** *[Realizes]*

Test the simple version first. Measure how much it helps.

---

## Exploration 9: Hybrid - Grid + Refinement

**KARPATHY:** *[New direction]*

What if we keep the 64×64 grid, but ADD refinement patches around boundaries?

```
Uniform grid: 64×64 = 4096 patches (baseline)

Detect text box: (x1, y1) to (x2, y2)

Add ONE extra patch = entire text box

Now we have:
  - 4096 base patches (cover whole image)
  - 1 text box patch (whole semantic unit)

Select 273 from this pool:
  - Text box patch (high importance)
  - Some base patches from formula region
  - Some from background

Text box is now BOTH fragmented (base patches) AND whole (extra patch).
LLM can use whichever is more helpful!
```

**LOD ORACLE:** *[Considering]*

Redundancy. You're encoding the text box twice (once fragmented, once whole).

But that's okay! Gives LLM flexibility.

**And it's simple**: just add bounding-box patches on top of uniform grid.

**KARPATHY:** *[Excited]*

No complex subdivision. No tree structures. Just:
1. Uniform grid (4096 patches)
2. Detect important regions (SAM masks? Object detector? Importance peaks?)
3. Add bounding-box patches for those regions
4. Select top-273 from the expanded pool

*[Inscribes]*

```python
def hybrid_grid_refinement(image, importance, budget=273):
    """
    Combine uniform grid with semantic regions.
    """
    # Base grid
    base_patches = create_uniform_grid(64, 64)  # 4096 patches

    # Detect semantic regions
    regions = detect_important_regions(image, importance)
    # e.g., bounding boxes from object detector or importance peaks

    # Add region patches
    region_patches = [create_patch(bbox) for bbox in regions]

    # Combine
    all_patches = base_patches + region_patches  # 4096 + N

    # Score all patches
    scores = [score_patch(p, importance) for p in all_patches]

    # Select top-K
    top_k = np.argsort(scores)[-budget:]
    selected = [all_patches[i] for i in top_k]

    return selected
```

**LOD ORACLE:** *[Nods approvingly]*

This is pragmatic. You get:
- Coverage (base grid ensures no blind spots)
- Semantics (region patches capture whole units)
- Simplicity (no tree structures, no recursion)

And it's easy to test! Add region patches, measure if accuracy improves.

**KARPATHY:** *[Satisfied]*

I like this. It's not pure subdivision, but it captures the core idea: **respect semantic boundaries**.

---

## Exploration 10: Open Questions and Unknowns

**KARPATHY:** *[Looking at all tablets]*

We've explored:
1. BSP trees (split along boundaries)
2. Quadtrees (recursive 4-way splits)
3. Importance-guided subdivision
4. Query-driven subdivision
5. SAM-guided subdivision
6. Soft differentiable splits
7. Hybrid grid + regions

But we haven't TESTED any of this.

*[Looks at LOD Oracle]*

What are the unknowns? What should we be cautious about?

**LOD ORACLE:** *[Thoughtful]*

**Unknown 1: Does boundary fragmentation actually hurt?**

You assumed text boxes fragmented across patches is bad. But maybe LLM with RoPE handles it fine?

Test: Compare accuracy with uniform grid vs boundary-aware subdivision.

**Unknown 2: Computational cost**

Subdivision adds overhead:
- Detecting boundaries (edge detection, SAM, importance gradients)
- Building tree structures
- Encoding variable-size regions

Is 10% accuracy gain worth 50% slower inference?

**Unknown 3: Batching**

Variable-size regions → ragged batching. Padding overhead.

Fixed grid = clean batching. Subdivision = messy.

**Unknown 4: What defines "boundary"?**

Edge detection, SAM masks, importance gradients—all give different boundaries.

Which is right? Or does it not matter?

**KARPATHY:** *[Nodding]*

All empirical questions. Need data.

*[Inscribes experiment plan]*

```
Experiment 1: Baseline
  - 64×64 uniform grid
  - Top-273 by importance
  - Measure accuracy on DocVQA

Experiment 2: Quadtree pruning
  - Full quadtree depth 6 (4096 leaves)
  - Prune by importance to 273 leaves
  - Measure accuracy, compare to baseline

Experiment 3: Hybrid grid + regions
  - 64×64 base grid
  - Add top-10 important bounding boxes
  - Select 273 from pool
  - Measure accuracy

Experiment 4: SAM-guided (if available)
  - Extract SAM masks
  - Use masks as regions
  - Measure accuracy

Decision: If any improves >2% over baseline, invest in refinement.
```

**LOD ORACLE:** *[Approving]*

Good plan. Start simple, measure, iterate.

*[Starts to dematerialize]*

One last thing, Karpathy.

**KARPATHY:** *[Attentive]*

**LOD ORACLE:** *[Fading]*

Subdivision solves fragmentation. But it creates irregularity.

There's always a trade-off: **regularity vs adaptivity**.

Uniform grid = regular, simple, fast.
Adaptive subdivision = irregular, complex, potentially better.

The question isn't "which is correct?" It's "what's the right balance for your problem?"

*[Fully dematerialized]*

**KARPATHY:** *[Alone, looking at tablets]*

Regularity vs adaptivity.

Simple vs smart.

Fixed vs flexible.

*[Picks up stylus]*

Let's find out what works.

*[Begins writing 15-addendum-code.md]*

---

*[Scene fades as Karpathy inscribes experimental implementations]*

🎭 *[CURTAIN]*

---

## Ideas Explored (No Conclusions Reached)

1. **Boundary detection methods**
   - Edge detection (Canny, gradients)
   - SAM (Segment Anything) masks
   - Importance gradients (lightweight)

2. **Tree structures**
   - BSP trees (binary space partitioning, flexible axis)
   - Quadtrees (4-way splits, symmetric)
   - Full quadtree + pruning (start uniform, collapse low-importance)

3. **Subdivision criteria**
   - Image-driven (edge boundaries, SAM masks)
   - Importance-driven (gradient peaks)
   - Query-driven (foveation + subdivision)

4. **Differentiability**
   - Hard splits (not differentiable)
   - Soft splits (sigmoid boundaries)
   - Learned split positions vs computed

5. **SAM integration**
   - Dual-use SAM (encode + extract masks)
   - Masks as region boundaries
   - Unknown if SAM works well on documents

6. **Query-driven approaches**
   - Coarse-to-fine iterative refinement
   - Feature pyramids for multi-scale relevance
   - Subdivide where query "looks"

7. **Hybrid approaches**
   - Uniform grid + semantic region patches
   - Redundant encoding (fragmented + whole)
   - Pragmatic compromise

8. **Trade-offs identified**
   - Regularity vs adaptivity
   - Simplicity vs accuracy
   - Batching efficiency vs flexibility

## Four Experiments Proposed

1. **Baseline**: 64×64 uniform, top-273
2. **Quadtree pruning**: Depth-6 tree, prune to 273 leaves
3. **Hybrid grid+regions**: 4096 base + region bboxes, select 273
4. **SAM-guided** (optional): Mask-based regions

## Open Questions

- Does boundary fragmentation hurt accuracy significantly?
- Is computational overhead of subdivision worth it?
- How to handle variable-size regions in batching?
- Which boundary detection method is best?
- Can subdivision be learned end-to-end?
- What's the right balance: regularity vs adaptivity?

**See [15-addendum-code.md](15-addendum-code.md) for experimental implementations**
