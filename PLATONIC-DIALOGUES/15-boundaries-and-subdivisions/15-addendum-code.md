# Part 15 Addendum: Boundary-Aware Subdivision Code
*Experimental implementations for adaptive spatial partitioning*

**Companion to**: [15-boundaries-and-subdivisions.md](15-boundaries-and-subdivisions.md)

---

## Overview: Four Core Implementations

This addendum provides working code for:

1. **Quadtree with Importance Pruning** - Full tree → collapse low-importance
2. **Hybrid Grid + Regions** - Uniform base + semantic bounding boxes
3. **BSP Tree Subdivision** - Binary space partitioning with boundary detection
4. **SAM-Guided Regions** - Use Segment Anything masks as patches

Plus evaluation framework to compare against baseline.

---

## Implementation 1: Quadtree with Importance Pruning

**Core idea**: Build full quadtree, prune low-importance regions, control final count

```python
import torch
import torch.nn as nn
import numpy as np

class QuadTreeNode:
    """Node in quadtree structure."""

    def __init__(self, bbox, depth=0):
        """
        Args:
            bbox: (x, y, width, height) in normalized [0,1] coordinates
            depth: Depth in tree (0 = root)
        """
        self.bbox = bbox  # (x, y, w, h)
        self.depth = depth
        self.children = []  # 4 children if internal node, empty if leaf
        self.importance_score = 0.0  # Computed later

    def is_leaf(self):
        return len(self.children) == 0

    def get_area(self):
        return self.bbox[2] * self.bbox[3]


class QuadTreeSubdivider(nn.Module):
    """
    Build quadtree, prune to budget.

    Advantages:
      - Structured (clean tree hierarchy)
      - Coverage guaranteed (all regions represented)
      - Importance-driven pruning

    Disadvantages:
      - Axis-aligned only
      - Forces 4-way splits (even if not needed)
    """

    def __init__(
        self,
        max_depth=6,
        budget=273,
        merge_threshold=0.1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.budget = budget
        self.merge_threshold = merge_threshold

    def build_full_tree(self, image_size=1.0):
        """
        Build complete quadtree to max_depth.

        Returns:
            Root node of full tree
        """
        root = QuadTreeNode(bbox=(0, 0, image_size, image_size), depth=0)

        def subdivide(node):
            if node.depth >= self.max_depth:
                return  # Leaf node

            # Split into 4 quadrants
            x, y, w, h = node.bbox
            hw, hh = w / 2, h / 2

            node.children = [
                QuadTreeNode((x, y, hw, hh), node.depth + 1),          # Top-left
                QuadTreeNode((x + hw, y, hw, hh), node.depth + 1),     # Top-right
                QuadTreeNode((x, y + hh, hw, hh), node.depth + 1),     # Bottom-left
                QuadTreeNode((x + hw, y + hh, hw, hh), node.depth + 1) # Bottom-right
            ]

            # Recurse
            for child in node.children:
                subdivide(child)

        subdivide(root)
        return root

    def compute_importance(self, node, importance_map):
        """
        Compute average importance for a region.

        Args:
            node: QuadTreeNode
            importance_map: [H, W] importance scores
        """
        x, y, w, h = node.bbox
        H, W = importance_map.shape

        # Convert normalized bbox to pixel coordinates
        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + w) * W), int((y + h) * H)
        x2, y2 = min(x2, W), min(y2, H)  # Clip to bounds

        if x2 > x1 and y2 > y1:
            region_importance = importance_map[y1:y2, x1:x2].mean()
        else:
            region_importance = 0.0

        node.importance_score = region_importance.item()

    def prune_tree(self, root, importance_map):
        """
        Merge low-importance siblings.

        If all 4 children are leaves and have low combined importance, merge them.
        """
        def compute_all_importance(node):
            """Recursively compute importance for all nodes."""
            if node.is_leaf():
                self.compute_importance(node, importance_map)
            else:
                for child in node.children:
                    compute_all_importance(child)

                # Internal node importance = sum of children
                node.importance_score = sum(c.importance_score for c in node.children)

        def prune(node):
            """Recursively prune from bottom up."""
            if node.is_leaf():
                return

            # Recurse first
            for child in node.children:
                prune(child)

            # Check if can merge
            if all(child.is_leaf() for child in node.children):
                # All children are leaves
                avg_importance = node.importance_score / 4

                if avg_importance < self.merge_threshold:
                    # Low importance, merge
                    node.children = []

        compute_all_importance(root)
        prune(root)

    def get_leaves(self, root):
        """Extract all leaf nodes."""
        leaves = []

        def traverse(node):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)

        traverse(root)
        return leaves

    def prune_to_budget(self, root):
        """
        Merge nodes until exactly 'budget' leaves remain.

        Strategy: Greedily merge sibling groups with lowest importance.
        """
        while True:
            leaves = self.get_leaves(root)
            if len(leaves) <= self.budget:
                break

            # Find best merge candidate
            best_merge = self.find_best_merge(root)

            if best_merge is None:
                break  # Can't merge any more

            # Merge: remove children
            best_merge.children = []

        return root

    def find_best_merge(self, root):
        """
        Find internal node whose children should be merged.

        Best = lowest total importance among sibling groups.
        """
        candidates = []

        def traverse(node):
            if node.is_leaf():
                return

            # Check if all children are leaves
            if all(child.is_leaf() for child in node.children):
                # Candidate for merging
                candidates.append((node.importance_score, node))
            else:
                for child in node.children:
                    traverse(child)

        traverse(root)

        if not candidates:
            return None

        # Return node with lowest importance
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def forward(self, importance_map):
        """
        Main forward pass.

        Args:
            importance_map: [H, W] importance scores

        Returns:
            List of bboxes: [(x, y, w, h), ...]
        """
        # Build full tree
        root = self.build_full_tree()

        # Prune low-importance regions
        self.prune_tree(root, importance_map)

        # Prune to exact budget
        root = self.prune_to_budget(root)

        # Extract leaf bboxes
        leaves = self.get_leaves(root)
        bboxes = [leaf.bbox for leaf in leaves]

        return bboxes


# Usage example
subdivider = QuadTreeSubdivider(max_depth=6, budget=273)

importance = torch.randn(512, 512)  # Mock importance map
bboxes = subdivider(importance)

print(f"Quadtree subdivision: {len(bboxes)} regions")
print(f"Variable sizes: min={min(b[2]*b[3] for b in bboxes):.4f}, max={max(b[2]*b[3] for b in bboxes):.4f}")
```

---

## Implementation 2: Hybrid Grid + Regions

**Core idea**: Uniform grid as base, add semantic region patches, select top-K

```python
class HybridGridRegionSampler(nn.Module):
    """
    Combine uniform grid with semantic region patches.

    Advantages:
      - Simple (no tree structures)
      - Coverage (grid ensures no blind spots)
      - Semantic (region patches capture whole units)

    Disadvantages:
      - Redundancy (regions overlap with grid)
      - Still need region detection method
    """

    def __init__(
        self,
        grid_size=64,
        num_regions=20,
        budget=273,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_regions = num_regions
        self.budget = budget

    def create_uniform_grid(self):
        """Create base grid of patches."""
        patches = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j / self.grid_size
                y = i / self.grid_size
                w = h = 1.0 / self.grid_size

                patches.append({
                    'bbox': (x, y, w, h),
                    'type': 'grid',
                    'grid_idx': (i, j)
                })

        return patches

    def detect_important_regions(self, importance_map, num_regions):
        """
        Detect important regions via local maxima + bounding boxes.

        Method: Find importance peaks, create bounding boxes around them.
        """
        H, W = importance_map.shape

        # Smooth importance
        from scipy.ndimage import gaussian_filter
        importance_smooth = gaussian_filter(importance_map.cpu().numpy(), sigma=5)

        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(importance_smooth, size=10)
        peaks = (importance_smooth == local_max) & (importance_smooth > importance_smooth.mean())

        # Get peak coordinates
        peak_coords = np.argwhere(peaks)

        if len(peak_coords) == 0:
            return []

        # Sort by importance
        peak_importance = [importance_smooth[y, x] for y, x in peak_coords]
        sorted_indices = np.argsort(peak_importance)[::-1]
        top_peaks = peak_coords[sorted_indices[:num_regions]]

        # Create bounding boxes around peaks
        regions = []
        bbox_size = 0.15  # 15% of image

        for y, x in top_peaks:
            # Center bbox on peak
            cx, cy = x / W, y / H

            # Create bbox
            bbox_x = max(0, cx - bbox_size / 2)
            bbox_y = max(0, cy - bbox_size / 2)
            bbox_w = min(bbox_size, 1.0 - bbox_x)
            bbox_h = min(bbox_size, 1.0 - bbox_y)

            regions.append({
                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
                'type': 'region',
                'peak': (cx, cy)
            })

        return regions

    def score_patch(self, patch, importance_map):
        """Compute importance score for a patch."""
        x, y, w, h = patch['bbox']
        H, W = importance_map.shape

        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + w) * W), int((y + h) * H)
        x2, y2 = min(x2, W), min(y2, H)

        if x2 > x1 and y2 > y1:
            score = importance_map[y1:y2, x1:x2].mean()
        else:
            score = 0.0

        return score.item()

    def forward(self, importance_map):
        """
        Main forward pass.

        Args:
            importance_map: [H, W]

        Returns:
            selected_patches: List of patch dicts
        """
        # Create base grid
        grid_patches = self.create_uniform_grid()

        # Detect important regions
        region_patches = self.detect_important_regions(importance_map, self.num_regions)

        # Combine
        all_patches = grid_patches + region_patches

        # Score all
        scores = [self.score_patch(p, importance_map) for p in all_patches]

        # Select top-K
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = sorted_indices[:self.budget]

        selected = [all_patches[i] for i in top_k_indices]

        return selected


# Usage
sampler = HybridGridRegionSampler(grid_size=64, num_regions=20, budget=273)

importance = torch.randn(512, 512)
selected = sampler(importance)

print(f"Selected {len(selected)} patches")
print(f"  Grid patches: {sum(1 for p in selected if p['type']=='grid')}")
print(f"  Region patches: {sum(1 for p in selected if p['type']=='region')}")
```

---

## Implementation 3: BSP Tree Subdivision

**Core idea**: Binary space partitioning with gradient-based split detection

```python
class BSPNode:
    """Node in BSP tree."""

    def __init__(self, bbox, depth=0):
        self.bbox = bbox  # (x, y, w, h)
        self.depth = depth
        self.split_axis = None  # 'vertical' or 'horizontal'
        self.split_pos = None   # Position along axis
        self.left = None   # Left/top child
        self.right = None  # Right/bottom child
        self.importance = 0.0

    def is_leaf(self):
        return self.left is None and self.right is None


class BSPTreeSubdivider(nn.Module):
    """
    Binary space partitioning with importance-guided splits.

    Advantages:
      - Flexible (choose axis per split)
      - Aligns with rectilinear content
      - Can create non-square regions

    Disadvantages:
      - More complex than quadtree
      - Unbalanced tree (variable depth)
    """

    def __init__(
        self,
        max_depth=8,
        min_region_size=0.02,
        homogeneity_threshold=0.1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_region_size = min_region_size
        self.homogeneity_threshold = homogeneity_threshold

    def choose_split_axis(self, bbox, importance_map):
        """
        Decide: split horizontally or vertically?

        Heuristics:
          1. Aspect ratio (split long axis)
          2. Gradient direction (split perpendicular to strong gradient)
        """
        x, y, w, h = bbox
        H, W = importance_map.shape

        # Get region
        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + w) * W), int((y + h) * H)
        x2, y2 = min(x2, W), min(y2, H)

        if x2 <= x1 or y2 <= y1:
            return 'vertical'  # Degenerate case

        region = importance_map[y1:y2, x1:x2]

        # Heuristic 1: Aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0

        if aspect_ratio > 1.5:
            return 'vertical'  # Wide → split vertically
        elif aspect_ratio < 0.67:
            return 'horizontal'  # Tall → split horizontally

        # Heuristic 2: Gradient direction
        if region.numel() > 1:
            grad_y, grad_x = torch.gradient(region)

            if grad_x.std() > grad_y.std():
                return 'vertical'  # Strong vertical gradients
            else:
                return 'horizontal'  # Strong horizontal gradients

        return 'vertical'  # Default

    def find_split_position(self, bbox, axis, importance_map):
        """
        Find best position along axis to split.

        Best = where gradient is strongest (major boundary).
        """
        x, y, w, h = bbox
        H, W = importance_map.shape

        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + w) * W), int((y + h) * H)
        x2, y2 = min(x2, W), min(y2, H)

        if x2 <= x1 or y2 <= y1:
            # Degenerate case
            if axis == 'vertical':
                return x + w / 2
            else:
                return y + h / 2

        region = importance_map[y1:y2, x1:x2]

        if axis == 'vertical':
            # Project along columns
            proj = torch.abs(torch.gradient(region, dim=1)[0]).sum(dim=0)

            if proj.numel() == 0:
                return x + w / 2

            split_idx = torch.argmax(proj).item()
            split_pos = x + (split_idx / region.shape[1]) * w
        else:  # horizontal
            # Project along rows
            proj = torch.abs(torch.gradient(region, dim=0)[0]).sum(dim=1)

            if proj.numel() == 0:
                return y + h / 2

            split_idx = torch.argmax(proj).item()
            split_pos = y + (split_idx / region.shape[0]) * h

        return split_pos

    def compute_homogeneity(self, bbox, importance_map):
        """
        Check if region is homogeneous (low gradient variance).

        Returns True if homogeneous (don't subdivide).
        """
        x, y, w, h = bbox
        H, W = importance_map.shape

        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + w) * W), int((y + h) * H)
        x2, y2 = min(x2, W), min(y2, H)

        if x2 <= x1 or y2 <= y1:
            return True

        region = importance_map[y1:y2, x1:x2]

        if region.numel() <= 1:
            return True

        grad = torch.gradient(region)
        variance = sum(g.var() for g in grad)

        return variance < self.homogeneity_threshold

    def recursive_subdivide(self, bbox, importance_map, depth=0):
        """
        Recursively subdivide region.

        Returns BSPNode tree.
        """
        node = BSPNode(bbox, depth)

        # Stopping criteria
        x, y, w, h = bbox

        if depth >= self.max_depth:
            return node

        if w < self.min_region_size or h < self.min_region_size:
            return node

        if self.compute_homogeneity(bbox, importance_map):
            return node

        # Subdivide
        axis = self.choose_split_axis(bbox, importance_map)
        split_pos = self.find_split_position(bbox, axis, importance_map)

        node.split_axis = axis
        node.split_pos = split_pos

        # Create children
        if axis == 'vertical':
            left_bbox = (x, y, split_pos - x, h)
            right_bbox = (split_pos, y, x + w - split_pos, h)
        else:  # horizontal
            left_bbox = (x, y, w, split_pos - y)
            right_bbox = (x, split_pos, w, y + h - split_pos)

        # Recurse
        node.left = self.recursive_subdivide(left_bbox, importance_map, depth + 1)
        node.right = self.recursive_subdivide(right_bbox, importance_map, depth + 1)

        return node

    def get_leaves(self, root):
        """Extract all leaf nodes."""
        leaves = []

        def traverse(node):
            if node.is_leaf():
                leaves.append(node)
            else:
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)

        traverse(root)
        return leaves

    def forward(self, importance_map):
        """
        Main forward pass.

        Args:
            importance_map: [H, W]

        Returns:
            List of bboxes
        """
        # Build BSP tree
        root = self.recursive_subdivide((0, 0, 1, 1), importance_map)

        # Extract leaves
        leaves = self.get_leaves(root)
        bboxes = [leaf.bbox for leaf in leaves]

        return bboxes


# Usage
subdivider = BSPTreeSubdivider(max_depth=8)

importance = torch.randn(512, 512)
bboxes = subdivider(importance)

print(f"BSP subdivision: {len(bboxes)} regions")
```

---

## Implementation 4: SAM-Guided Regions

**Core idea**: Use Segment Anything masks as patch boundaries

```python
class SAMGuidedRegionSampler(nn.Module):
    """
    Use SAM masks as semantic regions.

    Requires: segment-anything library

    Advantages:
      - Semantic boundaries (objects, not arbitrary)
      - Precise masks (pixel-level accuracy)

    Disadvantages:
      - Expensive (SAM inference)
      - May not work well on documents
      - Many masks (need filtering)
    """

    def __init__(
        self,
        sam_checkpoint=None,
        budget=273,
        min_mask_area=0.001,  # Minimum 0.1% of image
    ):
        super().__init__()
        self.budget = budget
        self.min_mask_area = min_mask_area

        # Load SAM model (if available)
        # Note: This requires the segment-anything package
        # pip install git+https://github.com/facebookresearch/segment-anything.git

        if sam_checkpoint is not None:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
        else:
            self.mask_generator = None

    def mask_to_bbox(self, mask):
        """
        Convert binary mask to bounding box.

        Args:
            mask: [H, W] binary mask

        Returns:
            (x, y, w, h) in normalized coordinates
        """
        H, W = mask.shape

        # Find non-zero coordinates
        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            return None

        # Bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Normalize
        x, y =