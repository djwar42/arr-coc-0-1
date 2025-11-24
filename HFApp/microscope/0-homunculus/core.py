"""
Homunculus visualization - shows which patches were selected vs rejected.

The "homunculus" is the visual representation of attention allocation:
- Selected patches (top-K) shown with green borders or highlights
- Rejected patches shown dimmed or with red overlay
- Shows WHERE the system focused, the core validation of query-awareness

Usage:
    homunculus_img = draw_homunculus(image, selected_indices, grid_size=32)
    fig = create_homunculus_figure(image, selected_indices, query="Where is the cat?")
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
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


def draw_homunculus(
    image: Image.Image,
    selected_indices: torch.Tensor,  # [K] or [B, K]
    grid_size: int = 32,
    style: str = 'overlay'  # 'overlay', 'borders', or 'heatmap'
) -> Image.Image:
    """
    Draw homunculus showing selected patches.

    Args:
        image: Original PIL image
        selected_indices: Indices of selected patches [K] or [B, K]
        grid_size: Number of patches per side (32 → 32x32 = 1024 patches)
        style: Visualization style
            - 'overlay': Red transparent overlay on rejected patches
            - 'borders': Green borders on selected patches
            - 'heatmap': Binary heatmap showing selection

    Returns:
        PIL Image with homunculus overlay
    """
    # Handle batch dimension
    if _get_ndim(selected_indices) == 2:
        selected_indices = selected_indices[0]  # Take first in batch

    selected_set = set(_to_numpy(selected_indices).tolist())

    # Create canvas
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw, 'RGBA')

    patch_h = image.height // grid_size
    patch_w = image.width // grid_size

    if style == 'overlay':
        # Red overlay on REJECTED patches
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx not in selected_set:
                    x0, y0 = j * patch_w, i * patch_h
                    x1, y1 = x0 + patch_w, y0 + patch_h
                    # Semi-transparent red
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 100))

    elif style == 'borders':
        # Green borders on SELECTED patches
        for idx in selected_set:
            i = idx // grid_size
            j = idx % grid_size
            x0, y0 = j * patch_w, i * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h
            # Thick green border
            draw.rectangle([x0, y0, x1, y1], outline='lime', width=3)

    elif style == 'heatmap':
        # Binary heatmap (selected = white, rejected = dark)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                x0, y0 = j * patch_w, i * patch_h
                x1, y1 = x0 + patch_w, y0 + patch_h

                if idx in selected_set:
                    # Selected: semi-transparent white
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 120))
                else:
                    # Rejected: semi-transparent black
                    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 150))

    return img_draw


def create_homunculus_figure(
    image: Image.Image,
    selected_indices: torch.Tensor,
    query: str,
    grid_size: int = 32,
    show_stats: bool = True
) -> Image.Image:
    """
    Create complete homunculus figure with image + overlay + metadata.

    Returns:
        PIL Image of matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nQuery: {query[:50]}...", fontsize=10)
    axes[0].axis('off')

    # Right: Homunculus overlay
    homunculus = draw_homunculus(image, selected_indices, grid_size, style='overlay')
    axes[1].imshow(homunculus)

    K = len(selected_indices) if _get_ndim(selected_indices) == 1 else selected_indices.shape[1]
    N = grid_size * grid_size
    coverage = K / N * 100

    title = f"Homunculus (Top-{K} Selected)\n{coverage:.1f}% coverage"
    axes[1].set_title(title, fontsize=10)
    axes[1].axis('off')

    # Stats annotation
    if show_stats:
        stats_text = f"""
Selected: {K} / {N} patches
Compression: {N/K:.1f}×
Red overlay = Rejected regions
"""
        axes[1].text(
            0.02, 0.98, stats_text.strip(),
            transform=axes[1].transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.suptitle("ARR-COC Patch Selection (Homunculus)", fontweight='bold')
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


def create_homunculus_grid(
    image: Image.Image,
    selected_indices: torch.Tensor,
    grid_size: int = 32
) -> np.ndarray:
    """
    Create binary grid showing selection (for use in heatmaps).

    Returns:
        np.ndarray of shape [grid_size, grid_size] with 1=selected, 0=rejected
    """
    if _get_ndim(selected_indices) == 2:
        selected_indices = selected_indices[0]

    grid = np.zeros((grid_size, grid_size))

    for idx in _to_numpy(selected_indices):
        i = idx // grid_size
        j = idx % grid_size
        grid[i, j] = 1.0

    return grid


# === TESTS ===

def test_homunculus():
    """Test homunculus visualization"""
    print("Testing homunculus visualization...")

    # Create dummy image
    image = Image.new('RGB', (512, 512), color='lightblue')

    # Create dummy selection (top-left quadrant)
    selected = torch.tensor([i * 32 + j for i in range(16) for j in range(16)])  # 256 patches

    # Test overlay style
    homunculus = draw_homunculus(image, selected, grid_size=32, style='overlay')
    assert homunculus.size == image.size
    print("✓ Overlay style works")

    # Test borders style
    homunculus = draw_homunculus(image, selected, grid_size=32, style='borders')
    assert homunculus.size == image.size
    print("✓ Borders style works")

    # Test figure creation
    fig = create_homunculus_figure(image, selected, query="Test query", grid_size=32)
    assert isinstance(fig, Image.Image)
    print("✓ Figure creation works")

    # Test grid creation
    grid = create_homunculus_grid(image, selected, grid_size=32)
    assert grid.shape == (32, 32)
    assert grid.sum() == 256  # 16x16 selected
    print("✓ Grid creation works")

    print("✓ All homunculus tests passed!")


if __name__ == "__main__":
    test_homunculus()
