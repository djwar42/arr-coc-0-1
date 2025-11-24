"""
Channel Grid Visualization - Show all 13 texture channels in a grid layout.

Inspired by game engine material inspectors (Unity/Unreal "Texture Properties" panel).
Shows each channel with semantic labels and proper normalization.

Usage:
    fig = create_channel_grid(textures, patch_idx=(5, 10))
    grid_img = visualize_all_channels(textures, layout='4x4')
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from io import BytesIO


# Channel metadata
CHANNEL_INFO = {
    0: {'name': 'Red (R)', 'group': 'Color', 'cmap': 'Reds'},
    1: {'name': 'Green (G)', 'group': 'Color', 'cmap': 'Greens'},
    2: {'name': 'Blue (B)', 'group': 'Color', 'cmap': 'Blues'},
    3: {'name': 'Lightness (L*)', 'group': 'Perceptual', 'cmap': 'gray'},
    4: {'name': 'Green-Red (a*)', 'group': 'Perceptual', 'cmap': 'RdYlGn'},
    5: {'name': 'Sobel-X (∂x)', 'group': 'Edges', 'cmap': 'seismic'},
    6: {'name': 'Sobel-Y (∂y)', 'group': 'Edges', 'cmap': 'seismic'},
    7: {'name': 'Edge Magnitude', 'group': 'Edges', 'cmap': 'hot'},
    8: {'name': 'Position-Y', 'group': 'Spatial', 'cmap': 'viridis'},
    9: {'name': 'Position-X', 'group': 'Spatial', 'cmap': 'plasma'},
    10: {'name': 'Eccentricity', 'group': 'Spatial', 'cmap': 'YlOrRd'},
    11: {'name': 'Saliency', 'group': 'Derived', 'cmap': 'hot'},
    12: {'name': 'Luminance', 'group': 'Derived', 'cmap': 'gray'},
}


def create_channel_grid(
    textures: torch.Tensor,  # [B, 13, H, W]
    batch_idx: int = 0,
    layout: str = '4x4',  # '4x4', '3x5', '2x7'
    figsize: tuple = (16, 16),
    show_colorbar: bool = True
) -> Image.Image:
    """
    Create grid showing all 13 channels with labels.

    Args:
        textures: Texture array [B, 13, H, W]
        batch_idx: Which image in batch to visualize
        layout: Grid layout ('4x4', '3x5', '2x7')
        figsize: Figure size
        show_colorbar: Show colorbar for each channel

    Returns:
        PIL Image of channel grid
    """
    if layout == '4x4':
        rows, cols = 4, 4
    elif layout == '3x5':
        rows, cols = 3, 5
    elif layout == '2x7':
        rows, cols = 2, 7
    else:
        rows, cols = 4, 4

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for ch in range(13):
        row = ch // cols
        col = ch % cols
        ax = fig.add_subplot(gs[row, col])

        # Get channel data
        channel_data = textures[batch_idx, ch].cpu().numpy()

        # Get metadata
        info = CHANNEL_INFO[ch]

        # Plot
        im = ax.imshow(channel_data, cmap=info['cmap'], interpolation='nearest')
        ax.set_title(f"[{ch}] {info['name']}\n({info['group']})", fontsize=9)
        ax.axis('off')

        # Colorbar
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Fill empty subplot if layout leaves space
    if 13 < rows * cols:
        ax_empty = fig.add_subplot(gs[rows-1, cols-1])
        ax_empty.text(0.5, 0.5, 'RGB Composite\n(Channels 0-2)',
                     ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        # Show RGB composite
        rgb = textures[batch_idx, 0:3].permute(1, 2, 0).cpu().numpy()
        ax_empty.imshow(rgb)
        ax_empty.axis('off')

    plt.suptitle('13-Channel Texture Array Breakdown', fontsize=14, fontweight='bold')

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


def visualize_all_channels(
    textures: torch.Tensor,
    batch_idx: int = 0,
    show_stats: bool = True
) -> Image.Image:
    """
    Simplified channel visualization with statistics.

    Shows all channels in 4x4 grid with min/max/mean annotations.
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for ch in range(13):
        ax = axes[ch]

        # Get data
        data = textures[batch_idx, ch].cpu().numpy()
        info = CHANNEL_INFO[ch]

        # Plot
        im = ax.imshow(data, cmap=info['cmap'], interpolation='bilinear')
        title = f"[{ch}] {info['name']}"

        if show_stats:
            title += f"\nmin={data.min():.2f} max={data.max():.2f}"

        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Last subplot: RGB composite
    ax = axes[13]
    rgb = textures[batch_idx, 0:3].permute(1, 2, 0).cpu().numpy()
    ax.imshow(rgb)
    ax.set_title("RGB Composite", fontsize=8)
    ax.axis('off')

    plt.suptitle('Texture Channel Visualization', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


def create_channel_inspector(
    textures: torch.Tensor,
    patch_idx: tuple,  # (y, x) position in grid
    batch_idx: int = 0
) -> Image.Image:
    """
    Inspect a SINGLE PATCH across all 13 channels.

    Shows bar chart of channel values for one specific patch.
    Useful for understanding "what does this patch look like in texture space?"
    """
    y, x = patch_idx

    # Extract single patch values
    patch_values = textures[batch_idx, :, y, x].cpu().numpy()  # [13]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of values
    colors = [CHANNEL_INFO[i]['cmap'] for i in range(13)]
    x_pos = np.arange(13)

    ax1.bar(x_pos, patch_values, color='steelblue', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([CHANNEL_INFO[i]['name'] for i in range(13)],
                         rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title(f'Channel Values at Patch ({y}, {x})', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Right: Show patch location on RGB composite
    rgb = textures[batch_idx, 0:3].permute(1, 2, 0).cpu().numpy()
    ax2.imshow(rgb)

    # Highlight the patch
    H, W = rgb.shape[:2]
    patch_h, patch_w = H // textures.shape[2], W // textures.shape[3]
    rect_y, rect_x = y * patch_h, x * patch_w

    from matplotlib.patches import Rectangle
    rect = Rectangle((rect_x, rect_y), patch_w, patch_h,
                     linewidth=3, edgecolor='lime', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title(f'Patch Location (Grid: {textures.shape[2]}×{textures.shape[3]})', fontsize=11)
    ax2.axis('off')

    plt.suptitle('Texture Patch Inspector', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


# === TESTS ===

def test_channel_grid():
    """Test channel grid visualization"""
    print("Testing channel grid...")

    # Create dummy textures
    textures = torch.rand(1, 13, 32, 32)

    # Test grid creation
    grid = create_channel_grid(textures, layout='4x4')
    assert isinstance(grid, Image.Image)
    print("✓ Channel grid works")

    # Test all channels
    viz = visualize_all_channels(textures)
    assert isinstance(viz, Image.Image)
    print("✓ All channels visualization works")

    # Test patch inspector
    inspector = create_channel_inspector(textures, patch_idx=(10, 15))
    assert isinstance(inspector, Image.Image)
    print("✓ Patch inspector works")

    print("✓ All channel grid tests passed!")


if __name__ == "__main__":
    test_channel_grid()
