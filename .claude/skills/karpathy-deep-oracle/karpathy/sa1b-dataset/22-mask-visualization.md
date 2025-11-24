# KNOWLEDGE DROP: Mask Visualization Techniques for SA-1B

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 23)
**File Created**: `sa1b-dataset/22-mask-visualization.md`

---

## What Was Created

**Knowledge File**: Mask Visualization Techniques (~700 lines)

**8 Sections**:
1. Matplotlib Overlay Fundamentals
2. Color Mapping for Multiple Masks
3. Alpha Blending Techniques
4. Multi-Mask Display Strategies
5. Interactive Visualization Tools
6. Saving Visualizations
7. Publication-Quality Figures
8. **ARR-COC-0-1** (10%): Visualization for spatial relevance debugging

---

## Key Insights

### Basic Mask Overlay with Matplotlib

From [Medium: Comprehensive Guide to Overlaying Segmentation Masks](https://medium.com/the-owl/comprehensive-guide-to-overlaying-segmentation-masks-in-python-86b67dd93fad):

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_single_mask(image, mask, alpha=0.5):
    """
    Overlay single binary mask on image.

    Args:
        image: numpy array (H, W, 3) RGB image
        mask: numpy array (H, W) binary mask
        alpha: transparency of mask overlay

    Returns:
        Overlay image as numpy array
    """
    # Ensure image is float
    image = image.astype(np.float32) / 255.0 if image.max() > 1 else image

    # Create colored mask (red)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask  # Red channel

    # Alpha blend
    overlay = image * (1 - alpha * mask[:, :, np.newaxis]) + \
              colored_mask * alpha * mask[:, :, np.newaxis]

    return np.clip(overlay, 0, 1)


# Display
plt.figure(figsize=(10, 10))
plt.imshow(visualize_single_mask(image, mask))
plt.axis('off')
plt.show()
```

### Color Mapping for Multiple Masks

```python
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def assign_mask_colors(n_masks, colormap='tab20'):
    """
    Assign distinct colors to multiple masks.

    Args:
        n_masks: Number of masks
        colormap: Matplotlib colormap name

    Returns:
        List of RGBA colors
    """
    cmap = cm.get_cmap(colormap)
    colors = [cmap(i / n_masks) for i in range(n_masks)]
    return colors


def visualize_multiple_masks(image, masks, alpha=0.5, colormap='tab20'):
    """
    Overlay multiple masks with distinct colors.

    Args:
        image: RGB image (H, W, 3)
        masks: Binary masks (N, H, W)
        alpha: Transparency
        colormap: Colormap for mask colors

    Returns:
        Overlay image
    """
    image = image.astype(np.float32) / 255.0 if image.max() > 1 else image

    # Get colors for each mask
    n_masks = len(masks)
    colors = assign_mask_colors(n_masks, colormap)

    # Create composite overlay
    overlay = image.copy()

    for mask, color in zip(masks, colors):
        # Create colored mask
        colored = np.zeros_like(image)
        for c in range(3):
            colored[:, :, c] = color[c]

        # Apply mask
        mask_3d = mask[:, :, np.newaxis]
        overlay = overlay * (1 - alpha * mask_3d) + colored * alpha * mask_3d

    return np.clip(overlay, 0, 1)
```

### Alpha Blending Techniques

From [SimpleITK: Results Visualization](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html):

```python
def alpha_blend_overlay(image, mask, color, alpha=0.5):
    """
    Alpha blending creates semi-transparent overlay.

    Formula: output = image * (1 - alpha) + colored_mask * alpha
    """
    colored_mask = np.zeros_like(image)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]

    blended = image * (1 - alpha) + colored_mask * alpha

    return blended


def edge_highlighted_overlay(image, mask, edge_color=(1, 0, 0), fill_alpha=0.3):
    """
    Overlay with highlighted edges.
    """
    from scipy import ndimage

    # Find edges using gradient
    edges = ndimage.sobel(mask.astype(float))
    edges = (np.abs(edges) > 0).astype(float)

    # Fill overlay
    overlay = alpha_blend_overlay(image, mask, (0, 1, 0), fill_alpha)

    # Edge overlay (stronger)
    overlay = alpha_blend_overlay(overlay, edges, edge_color, 0.8)

    return overlay
```

### Multi-Mask Display Grid

```python
def create_mask_grid(image, masks, metadata=None, cols=4, figsize=(16, 16)):
    """
    Display multiple masks in a grid.

    Args:
        image: Original image
        masks: Array of masks (N, H, W)
        metadata: Optional dict with mask metadata
        cols: Number of columns
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_masks = len(masks)
    rows = (n_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, mask) in enumerate(zip(axes, masks)):
        # Create overlay for this mask
        overlay = visualize_single_mask(image, mask, alpha=0.6)

        ax.imshow(overlay)
        ax.axis('off')

        # Add metadata title
        if metadata:
            iou = metadata.get('predicted_ious', [0]*n_masks)[i]
            area = metadata.get('areas', [0]*n_masks)[i]
            ax.set_title(f'IoU: {iou:.2f}, Area: {area:.0f}', fontsize=8)

    # Hide empty subplots
    for ax in axes[n_masks:]:
        ax.axis('off')

    plt.tight_layout()
    return fig


def create_hierarchical_view(image, masks, areas):
    """
    Display masks organized by size hierarchy.
    """
    # Sort by area
    sorted_indices = np.argsort(areas)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Large masks
    large_idx = sorted_indices[:len(sorted_indices)//3]
    large_overlay = visualize_multiple_masks(image, masks[large_idx])
    axes[0].imshow(large_overlay)
    axes[0].set_title('Large Masks')
    axes[0].axis('off')

    # Medium masks
    med_idx = sorted_indices[len(sorted_indices)//3:2*len(sorted_indices)//3]
    med_overlay = visualize_multiple_masks(image, masks[med_idx])
    axes[1].imshow(med_overlay)
    axes[1].set_title('Medium Masks')
    axes[1].axis('off')

    # Small masks
    small_idx = sorted_indices[2*len(sorted_indices)//3:]
    small_overlay = visualize_multiple_masks(image, masks[small_idx])
    axes[2].imshow(small_overlay)
    axes[2].set_title('Small Masks')
    axes[2].axis('off')

    return fig
```

### Interactive Visualization

```python
def create_interactive_viewer(image, masks, metadata):
    """
    Create interactive mask viewer with slider.
    """
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)

    # Initial display
    overlay = visualize_single_mask(image, masks[0])
    im = ax.imshow(overlay)
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Mask',
        0, len(masks) - 1,
        valinit=0, valstep=1
    )

    def update(val):
        idx = int(slider.val)
        overlay = visualize_single_mask(image, masks[idx])
        im.set_data(overlay)

        # Update title with metadata
        iou = metadata.get('predicted_ious', [0]*len(masks))[idx]
        area = metadata.get('areas', [0]*len(masks))[idx]
        ax.set_title(f'Mask {idx}: IoU={iou:.3f}, Area={area:.0f}')

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return fig


def create_clickable_viewer(image, masks):
    """
    Click to toggle individual mask visibility.
    """
    visible = np.ones(len(masks), dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 10))

    def update_display():
        overlay = image.copy().astype(float) / 255.0
        for i, (mask, vis) in enumerate(zip(masks, visible)):
            if vis:
                color = assign_mask_colors(len(masks))[i]
                overlay = alpha_blend_overlay(overlay, mask, color[:3], 0.5)
        ax.imshow(overlay)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Find clicked mask
        for i, mask in enumerate(masks):
            if mask[y, x]:
                visible[i] = not visible[i]
                break

        update_display()

    fig.canvas.mpl_connect('button_press_event', on_click)
    update_display()

    return fig
```

### Saving Visualizations

```python
def save_visualization(image, masks, output_path, dpi=150):
    """
    Save publication-quality visualization.
    """
    overlay = visualize_multiple_masks(image, masks)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(overlay)
    ax.axis('off')

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()


def save_mask_comparison(image, gt_mask, pred_mask, output_path):
    """
    Save side-by-side comparison of ground truth vs prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth
    gt_overlay = visualize_single_mask(image, gt_mask, alpha=0.5)
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    pred_overlay = visualize_single_mask(image, pred_mask, alpha=0.5)
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Complete SA-1B Visualizer Class

```python
class SA1BVisualizer:
    """
    Comprehensive visualization tools for SA-1B dataset.
    """

    def __init__(self, colormap='tab20', default_alpha=0.5):
        self.colormap = colormap
        self.default_alpha = default_alpha

    def visualize_sample(self, image, masks, metadata=None, max_masks=50):
        """
        Visualize SA-1B sample with all masks.
        """
        # Limit masks for clarity
        if len(masks) > max_masks:
            # Select by area
            areas = metadata.get('areas', np.ones(len(masks)))
            indices = np.argsort(areas)[::-1][:max_masks]
            masks = masks[indices]

        overlay = visualize_multiple_masks(
            image, masks, self.default_alpha, self.colormap
        )

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(overlay)
        ax.set_title(f'{len(masks)} masks visualized')
        ax.axis('off')

        return fig

    def visualize_annotation_quality(self, image, masks, ious, stabilities):
        """
        Visualize masks colored by quality metrics.
        """
        # Sort by IoU
        sorted_idx = np.argsort(ious)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Top quality masks
        top_masks = masks[sorted_idx[:20]]
        top_overlay = visualize_multiple_masks(image, top_masks)
        axes[0].imshow(top_overlay)
        axes[0].set_title(f'Top 20 by IoU (avg: {ious[sorted_idx[:20]].mean():.3f})')
        axes[0].axis('off')

        # Bottom quality masks
        bottom_masks = masks[sorted_idx[-20:]]
        bottom_overlay = visualize_multiple_masks(image, bottom_masks)
        axes[1].imshow(bottom_overlay)
        axes[1].set_title(f'Bottom 20 by IoU (avg: {ious[sorted_idx[-20:]].mean():.3f})')
        axes[1].axis('off')

        return fig

    def create_granularity_comparison(self, image, masks, areas):
        """
        Compare mask granularity levels.
        """
        return create_hierarchical_view(image, masks, areas)
```

---

## Research Performed

**Web sources consulted**:
1. [Medium: Overlaying Segmentation Masks Guide](https://medium.com/the-owl/comprehensive-guide-to-overlaying-segmentation-masks-in-python-86b67dd93fad)
2. [StackOverflow: Overlay image segmentation](https://stackoverflow.com/questions/31877353)
3. [SimpleITK: Results Visualization](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html)
4. [LearnOpenCV: Alpha Blending](https://learnopencv.com/alpha-blending-using-opencv-cpp-python/)
5. [Kaggle: Overlaying Masks for Segmentation](https://www.kaggle.com/code/tuynlc/overlaying-a-mask-on-an-image-for-segmentation)

**Source document**:
- SAM_DATASET_SA1B.md (lines 80-100: mask visualization requirements)

---

## ARR-COC-0-1 Integration (10%)

### Visualization for Spatial Relevance Debugging

```python
class ARRCOCVisualizer(SA1BVisualizer):
    """
    Extended visualizer for ARR-COC spatial relevance debugging.
    """

    def visualize_spatial_relationships(self, image, masks, containment_matrix):
        """
        Visualize containment relationships between masks.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Mask overlay
        overlay = visualize_multiple_masks(image, masks)
        axes[0].imshow(overlay)
        axes[0].set_title('All Masks')
        axes[0].axis('off')

        # Containment heatmap
        im = axes[1].imshow(containment_matrix, cmap='hot')
        axes[1].set_title('Containment Matrix')
        axes[1].set_xlabel('Contained Mask')
        axes[1].set_ylabel('Container Mask')
        plt.colorbar(im, ax=axes[1])

        return fig

    def visualize_attention_weights(self, image, masks, attention_weights):
        """
        Visualize model attention on masks.
        """
        # Weight colors by attention
        weighted_colors = []
        for i, weight in enumerate(attention_weights):
            # Higher attention = more red
            color = (weight, 0, 1 - weight, 0.5)
            weighted_colors.append(color)

        fig, ax = plt.subplots(figsize=(12, 12))

        overlay = image.copy().astype(float) / 255.0
        for mask, color in zip(masks, weighted_colors):
            overlay = alpha_blend_overlay(overlay, mask, color[:3], color[3])

        ax.imshow(overlay)
        ax.set_title('Attention-Weighted Masks (red=high attention)')
        ax.axis('off')

        return fig
```

**Benefits**:
- **Debug spatial learning**: Visualize containment relationships
- **Attention analysis**: See what model focuses on
- **Quality assessment**: Identify problematic masks

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: 15+ (overlays, grids, interactive)
- **Sections**: 8 (7 technical + 1 ARR-COC at 10%)
- **Web sources**: 5 cited with URLs
- **Completion time**: ~40 minutes

---

## Next Steps

**PART 24** will cover: Preprocessing Pipelines & Transforms
- Synchronized image-mask transforms
- Data augmentation for segmentation
- Albumentations integration
