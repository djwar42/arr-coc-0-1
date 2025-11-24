"""
False Color Visualization - Color-code channels by semantic meaning.

Game rendering concept: In shader debugging, we use false color to show:
- Red = X-axis data
- Green = Y-axis data
- Blue = Z-axis data
- etc.

We apply this to texture channels: color-code by MEANING, not pixel value.

Usage:
    false_img = apply_false_color(textures, mode='semantic')
    composite = create_false_color_composite(textures, channels=[0,1,7])
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO


def apply_false_color(
    textures: torch.Tensor,  # [B, 13, H, W]
    mode: str = 'semantic',  # 'semantic', 'edges', 'spatial'
    batch_idx: int = 0
) -> Image.Image:
    """
    Apply false color encoding based on semantic meaning.

    Args:
        textures: Texture array
        mode: Which semantic grouping to visualize
            - 'semantic': Color/Edges/Spatial as RGB
            - 'edges': Sobel Gx/Gy/Mag as RGB
            - 'spatial': Position X/Y/Eccentricity as RGB
        batch_idx: Which batch item

    Returns:
        PIL Image with false color encoding
    """
    if mode == 'semantic':
        # Red = Color intensity (average RGB)
        # Green = Edge strength (Sobel magnitude)
        # Blue = Eccentricity

        R = textures[batch_idx, 0:3].mean(dim=0)  # Average of RGB
        G = textures[batch_idx, 7]  # Sobel magnitude
        B = textures[batch_idx, 10]  # Eccentricity

    elif mode == 'edges':
        # Sobel components
        R = textures[batch_idx, 5]  # Sobel-X (normalized)
        G = textures[batch_idx, 6]  # Sobel-Y (normalized)
        B = textures[batch_idx, 7]  # Sobel magnitude

    elif mode == 'spatial':
        # Spatial encoding
        R = textures[batch_idx, 8]  # Position-Y
        G = textures[batch_idx, 9]  # Position-X
        B = textures[batch_idx, 10]  # Eccentricity

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Stack and normalize
    rgb = torch.stack([R, G, B], dim=0)  # [3, H, W]

    # Normalize each channel to [0, 1]
    for i in range(3):
        ch = rgb[i]
        rgb[i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    # Convert to PIL
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb_np)


def create_false_color_composite(
    textures: torch.Tensor,
    channels: list = [0, 7, 10],  # RGB mapping to channels
    batch_idx: int = 0,
    normalize: bool = True
) -> Image.Image:
    """
    Create custom false color composite by selecting 3 channels.

    Args:
        textures: Texture array [B, 13, H, W]
        channels: List of 3 channel indices to map to RGB
        batch_idx: Which batch
        normalize: Whether to normalize each channel to [0,1]

    Returns:
        PIL Image with false color composite
    """
    assert len(channels) == 3, "Must provide exactly 3 channels for RGB"

    # Extract channels
    R = textures[batch_idx, channels[0]]
    G = textures[batch_idx, channels[1]]
    B = textures[batch_idx, channels[2]]

    rgb = torch.stack([R, G, B], dim=0)  # [3, H, W]

    # Normalize if requested
    if normalize:
        for i in range(3):
            ch = rgb[i]
            rgb[i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    # Convert to PIL
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb_np)


def create_false_color_comparison(
    textures: torch.Tensor,
    batch_idx: int = 0
) -> Image.Image:
    """
    Show multiple false color modes side-by-side.

    Returns:
        PIL Image with 4 false color variants
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original RGB
    rgb = textures[batch_idx, 0:3].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB", fontsize=10)
    axes[0, 0].axis('off')

    # Semantic false color
    semantic = apply_false_color(textures, mode='semantic', batch_idx=batch_idx)
    axes[0, 1].imshow(semantic)
    axes[0, 1].set_title("Semantic\nR=Color G=Edges B=Eccentric", fontsize=9)
    axes[0, 1].axis('off')

    # Edges false color
    edges = apply_false_color(textures, mode='edges', batch_idx=batch_idx)
    axes[1, 0].imshow(edges)
    axes[1, 0].set_title("Edges\nR=Sobel-X G=Sobel-Y B=Magnitude", fontsize=9)
    axes[1, 0].axis('off')

    # Spatial false color
    spatial = apply_false_color(textures, mode='spatial', batch_idx=batch_idx)
    axes[1, 1].imshow(spatial)
    axes[1, 1].set_title("Spatial\nR=Pos-Y G=Pos-X B=Eccentric", fontsize=9)
    axes[1, 1].axis('off')

    plt.suptitle("False Color Encoding Comparison", fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = Image.open(buf)
    plt.close()

    return result


# === TESTS ===

def test_false_color():
    """Test false color visualization"""
    print("Testing false color...")

    # Create dummy textures
    textures = torch.rand(1, 13, 32, 32)

    # Test semantic mode
    semantic = apply_false_color(textures, mode='semantic')
    assert isinstance(semantic, Image.Image)
    print("✓ Semantic false color works")

    # Test edges mode
    edges = apply_false_color(textures, mode='edges')
    assert isinstance(edges, Image.Image)
    print("✓ Edges false color works")

    # Test custom composite
    custom = create_false_color_composite(textures, channels=[0, 7, 10])
    assert isinstance(custom, Image.Image)
    print("✓ Custom composite works")

    # Test comparison
    comparison = create_false_color_comparison(textures)
    assert isinstance(comparison, Image.Image)
    print("✓ Comparison works")

    print("✓ All false color tests passed!")


if __name__ == "__main__":
    test_false_color()
