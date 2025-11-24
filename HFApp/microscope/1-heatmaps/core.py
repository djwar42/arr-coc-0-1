"""
Heatmap visualizations for relevance scores.

Creates smooth, interpolated heatmaps overlaid on original images
to show WHERE the system found relevance.

Usage:
    heatmap_img = draw_heatmap(scores, image)
    fig = create_heatmap_figure(scores, image, title="Information Score")
"""

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO


def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if hasattr(x, 'cpu'):  # PyTorch tensor
        return x.cpu().numpy()
    return np.asarray(x)


def _get_ndim(x):
    """Get number of dimensions (works for both numpy and torch)."""
    if hasattr(x, 'dim'):  # PyTorch tensor
        return x.dim()
    return np.asarray(x).ndim


def draw_heatmap(
    scores: torch.Tensor,  # [H, W] or [B, H, W]
    image: Image.Image = None,
    colormap: str = 'hot',
    alpha: float = 0.5,
    interpolation: str = 'bilinear'
) -> Image.Image:
    """
    Draw heatmap from score tensor.

    Args:
        scores: Relevance scores [H, W] or [B, H, W]
        image: Optional background image for overlay
        colormap: Matplotlib colormap ('hot', 'viridis', 'plasma', 'inferno')
        alpha: Transparency for overlay (0=transparent, 1=opaque)
        interpolation: 'nearest', 'bilinear', 'bicubic'

    Returns:
        PIL Image of heatmap (with optional image overlay)
    """
    # Handle batch dimension
    if _get_ndim(scores) == 3:
        scores = scores[0]

    # Convert to numpy
    scores_np = _to_numpy(scores)

    # Normalize to [0, 1]
    scores_norm = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)

    # Resize to image size if image provided
    if image is not None:
        H, W = image.height, image.width
        # Use cv2 for interpolation
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC
        }
        scores_resized = cv2.resize(scores_norm, (W, H), interpolation=interp_map[interpolation])
    else:
        scores_resized = scores_norm
        H, W = scores_resized.shape

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(scores_resized)[:, :, :3]  # Drop alpha channel
    heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)

    # Convert to PIL
    heatmap_img = Image.fromarray(heatmap_uint8)

    # Overlay on image if provided
    if image is not None:
        # Blend
        result = Image.blend(image.convert('RGB'), heatmap_img, alpha=alpha)
        return result
    else:
        return heatmap_img


def create_heatmap_figure(
    scores: torch.Tensor,
    image: Image.Image,
    title: str = "Relevance Heatmap",
    colormap: str = 'hot',
    show_colorbar: bool = True
) -> Image.Image:
    """
    Create complete heatmap figure with image + heatmap + colorbar.

    Returns:
        PIL Image of matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis('off')

    # Right: Heatmap overlay
    heatmap = draw_heatmap(scores, image, colormap=colormap, alpha=0.6)
    im = axes[1].imshow(heatmap)
    axes[1].set_title(title, fontsize=10)
    axes[1].axis('off')

    # Colorbar
    if show_colorbar:
        # Create colorbar on the heatmap
        scores_np = _to_numpy(scores[0]) if _get_ndim(scores) == 3 else _to_numpy(scores)
        norm = plt.Normalize(vmin=scores_np.min(), vmax=scores_np.max())
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


def create_multi_heatmap_figure(
    score_dict: dict,  # {name: scores_tensor}
    image: Image.Image,
    suptitle: str = "Relevance Score Comparison"
) -> Image.Image:
    """
    Create figure with multiple heatmaps side-by-side.

    Args:
        score_dict: {name: scores_tensor} mapping
        image: Background image
        suptitle: Overall figure title

    Returns:
        PIL Image of matplotlib figure
    """
    n_heatmaps = len(score_dict)
    fig, axes = plt.subplots(1, n_heatmaps, figsize=(5 * n_heatmaps, 5))

    if n_heatmaps == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, score_dict.items()):
        heatmap = draw_heatmap(scores, image, colormap='hot', alpha=0.6)
        ax.imshow(heatmap)
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    plt.suptitle(suptitle, fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


# === TESTS ===

def test_heatmaps():
    """Test heatmap visualization"""
    print("Testing heatmap visualization...")

    # Create dummy scores
    scores = torch.rand(32, 32)

    # Create dummy image
    image = Image.new('RGB', (512, 512), color='lightblue')

    # Test heatmap drawing
    heatmap = draw_heatmap(scores, image, colormap='hot')
    assert heatmap.size == image.size
    print("✓ Heatmap drawing works")

    # Test figure creation
    fig = create_heatmap_figure(scores, image, title="Test Heatmap")
    assert isinstance(fig, Image.Image)
    print("✓ Figure creation works")

    # Test multi-heatmap
    score_dict = {
        'Score 1': torch.rand(32, 32),
        'Score 2': torch.rand(32, 32),
        'Score 3': torch.rand(32, 32)
    }
    multi_fig = create_multi_heatmap_figure(score_dict, image)
    assert isinstance(multi_fig, Image.Image)
    print("✓ Multi-heatmap works")

    print("✓ All heatmap tests passed!")


if __name__ == "__main__":
    test_heatmaps()
