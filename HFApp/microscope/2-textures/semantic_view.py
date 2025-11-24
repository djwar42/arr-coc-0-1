"""
Semantic View - Group channels by meaning (Color, Edges, Spatial, Derived).

Instead of showing 13 individual channels, show 4 semantic groups:
1. Color (RGB + LAB)
2. Edges (Sobel Gx/Gy/Mag)
3. Spatial (Position X/Y + Eccentricity)
4. Derived (Saliency + Luminance)

Usage:
    groups = create_semantic_groups(textures)
    fig = visualize_by_meaning(textures)
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from io import BytesIO


def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if hasattr(x, 'cpu'):  # PyTorch tensor
        return x.cpu().numpy()
    return np.asarray(x)


SEMANTIC_GROUPS = {
    'Color': {
        'channels': [0, 1, 2, 3, 4],
        'names': ['Red', 'Green', 'Blue', 'Lightness (L*)', 'Green-Red (a*)'],
        'description': 'RGB + Perceptual color (LAB)',
        'color': 'blue'
    },
    'Edges': {
        'channels': [5, 6, 7],
        'names': ['Sobel-X (∂x)', 'Sobel-Y (∂y)', 'Edge Magnitude'],
        'description': 'Sobel edge detection',
        'color': 'orange'
    },
    'Spatial': {
        'channels': [8, 9, 10],
        'names': ['Position-Y', 'Position-X', 'Eccentricity'],
        'description': 'Spatial coordinates + distance from center',
        'color': 'green'
    },
    'Derived': {
        'channels': [11, 12],
        'names': ['Saliency', 'Luminance'],
        'description': 'Derived/reused channels',
        'color': 'purple'
    }
}


def create_semantic_groups(
    textures: torch.Tensor,  # [B, 13, H, W]
    batch_idx: int = 0
) -> dict:
    """
    Group texture channels by semantic meaning.

    Returns:
        dict with keys: 'Color', 'Edges', 'Spatial', 'Derived'
        Each contains: {channels: [...], tensors: [...], names: [...]}
    """
    result = {}

    for group_name, group_info in SEMANTIC_GROUPS.items():
        ch_indices = group_info['channels']
        result[group_name] = {
            'channels': ch_indices,
            'tensors': [textures[batch_idx, ch] for ch in ch_indices],
            'names': group_info['names'],
            'description': group_info['description'],
            'color': group_info['color']
        }

    return result


def visualize_by_meaning(
    textures: torch.Tensor,
    batch_idx: int = 0,
    figsize: tuple = (16, 12)
) -> Image.Image:
    """
    Visualize texture channels grouped by semantic meaning.

    Creates 4 rows (one per semantic group) with channels shown horizontally.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.3)

    groups = create_semantic_groups(textures, batch_idx)

    row = 0
    for group_name in ['Color', 'Edges', 'Spatial', 'Derived']:
        group = groups[group_name]

        # Group title (spans first column)
        ax_title = fig.add_subplot(gs[row, 0])
        ax_title.text(0.5, 0.5, f"**{group_name}**\n\n{group['description']}",
                     ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor=group['color'], alpha=0.3))
        ax_title.axis('off')

        # Show each channel in this group
        for i, (tensor, name) in enumerate(zip(group['tensors'], group['names'])):
            col = i + 1
            if col >= 6:
                break  # Max 5 channels per row

            ax = fig.add_subplot(gs[row, col])
            data = _to_numpy(tensor)

            ax.imshow(data, cmap='gray', interpolation='bilinear')
            ax.set_title(name, fontsize=8)
            ax.axis('off')

        row += 1

    plt.suptitle('Texture Channels by Semantic Meaning', fontsize=14, fontweight='bold')

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


def create_semantic_composite(
    textures: torch.Tensor,
    group_name: str,  # 'Color', 'Edges', 'Spatial', 'Derived'
    batch_idx: int = 0,
    method: str = 'mean'  # 'mean', 'max', 'pca'
) -> torch.Tensor:
    """
    Create single composite image from a semantic group.

    Args:
        textures: Texture array
        group_name: Which semantic group
        batch_idx: Which batch
        method: How to combine channels
            - 'mean': Average all channels
            - 'max': Max across channels
            - 'pca': PCA to reduce to 1 channel (TODO)

    Returns:
        Single-channel tensor [H, W]
    """
    group_info = SEMANTIC_GROUPS[group_name]
    ch_indices = group_info['channels']

    # Extract channels
    group_tensors = textures[batch_idx, ch_indices]  # [num_channels, H, W]

    if method == 'mean':
        composite = group_tensors.mean(dim=0)
    elif method == 'max':
        composite = group_tensors.max(dim=0)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    return composite


def visualize_semantic_composites(
    textures: torch.Tensor,
    batch_idx: int = 0
) -> Image.Image:
    """
    Show one composite per semantic group (4 total).

    Useful for seeing "what does Color look like overall" without
    looking at 5 separate channels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, group_name in enumerate(['Color', 'Edges', 'Spatial', 'Derived']):
        ax = axes[i]

        # Create composite
        composite = create_semantic_composite(textures, group_name, batch_idx, method='mean')
        composite_np = composite.cpu().numpy()

        # Plot
        group_color = SEMANTIC_GROUPS[group_name]['color']
        im = ax.imshow(composite_np, cmap='gray', interpolation='bilinear')
        ax.set_title(f"{group_name} Composite\n({SEMANTIC_GROUPS[group_name]['description']})",
                    fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Semantic Group Composites (Mean Aggregation)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


# === TESTS ===

def test_semantic_view():
    """Test semantic view"""
    print("Testing semantic view...")

    # Create dummy textures
    textures = torch.rand(1, 13, 32, 32)

    # Test grouping
    groups = create_semantic_groups(textures)
    assert len(groups) == 4
    assert 'Color' in groups
    print("✓ Semantic grouping works")

    # Test visualization
    viz = visualize_by_meaning(textures)
    assert isinstance(viz, Image.Image)
    print("✓ Semantic visualization works")

    # Test composite creation
    color_comp = create_semantic_composite(textures, 'Color', method='mean')
    assert color_comp.shape == (32, 32)
    print("✓ Semantic composite works")

    # Test composites visualization
    comps = visualize_semantic_composites(textures)
    assert isinstance(comps, Image.Image)
    print("✓ Composites visualization works")

    print("✓ All semantic view tests passed!")


if __name__ == "__main__":
    test_semantic_view()
