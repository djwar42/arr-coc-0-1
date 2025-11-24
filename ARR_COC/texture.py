"""
ARR_COC/texture.py - 13-Channel Texture Array Generation

Converts RGB images into multi-channel texture representations for relevance scoring.

Channels (0-12):
  0-2:   RGB (normalized [0,1])
  3-4:   LAB L* and a* (b* dropped for MVP)
  5-7:   Sobel edges (Gx, Gy, magnitude)
  8-9:   Spatial position (y_norm, x_norm)
  10:    Eccentricity (distance from center)
  11:    Saliency proxy (reuse Sobel magnitude)
  12:    Luminance (reuse LAB L*)

No CLIP, no PCA, no temporal - pure vision-based features for MVP.
"""

import torch
import torch.nn.functional as F
import kornia


def generate_texture_array(
    image: torch.Tensor,  # [B, 3, H, W] RGB in [0,1]
    target_size: int = 32  # Output 32×32 patches
) -> torch.Tensor:
    """
    Generate 13-channel texture array from RGB image.

    Args:
        image: [B, 3, H, W] RGB image, values in [0, 1]
        target_size: Output spatial resolution (32 → [B, 13, 32, 32])

    Returns:
        textures: [B, 13, target_size, target_size]
    """
    B, _, H, W = image.shape
    device = image.device

    # Downsample to intermediate resolution for processing
    # Use 16x target size for better feature extraction
    intermediate_size = target_size * 16  # 512×512 for 32×32 output
    image_intermediate = F.interpolate(
        image,
        size=(intermediate_size, intermediate_size),
        mode='bilinear',
        align_corners=False
    )

    # Initialize output
    textures = torch.zeros(B, 13, target_size, target_size, device=device)

    # === CHANNELS 0-2: RGB ===
    textures[:, 0:3] = F.interpolate(
        image_intermediate,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )

    # === CHANNELS 3-4: LAB (L* and a*) ===
    # Convert RGB → LAB using kornia
    lab = kornia.color.rgb_to_lab(image_intermediate)

    # L* channel (lightness)
    textures[:, 3] = F.interpolate(
        lab[:, 0:1],
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    # a* channel (green-red)
    textures[:, 4] = F.interpolate(
        lab[:, 1:2],
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    # Normalize LAB to [0,1]
    textures[:, 3] = textures[:, 3] / 100.0  # L* is in [0, 100]
    textures[:, 4] = (textures[:, 4] + 128.0) / 255.0  # a* is in [-128, 127]

    # === CHANNELS 5-7: Sobel Edges ===
    # Convert to grayscale for edge detection
    gray = kornia.color.rgb_to_grayscale(image_intermediate)

    # Spatial gradient (returns [B, C, 2, H, W] where last dim is [dx, dy])
    grads = kornia.filters.spatial_gradient(gray, mode='sobel', normalized=True)
    sobel_x = grads[:, :, 0, :, :]  # X gradient [B, 1, H, W]
    sobel_y = grads[:, :, 1, :, :]  # Y gradient [B, 1, H, W]

    # Magnitude
    sobel_mag = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)

    # Downsample edges
    textures[:, 5] = F.interpolate(
        sobel_x, size=(target_size, target_size), mode='bilinear', align_corners=False
    ).squeeze(1)
    textures[:, 6] = F.interpolate(
        sobel_y, size=(target_size, target_size), mode='bilinear', align_corners=False
    ).squeeze(1)
    textures[:, 7] = F.interpolate(
        sobel_mag, size=(target_size, target_size), mode='bilinear', align_corners=False
    ).squeeze(1)

    # Normalize edges to [0,1]
    for c in [5, 6, 7]:
        textures[:, c] = (textures[:, c] - textures[:, c].min()) / (
            textures[:, c].max() - textures[:, c].min() + 1e-8
        )

    # === CHANNELS 8-9: Spatial Position ===
    # Normalized (y, x) coordinates in [0, 1]
    y_coords = torch.linspace(0, 1, target_size, device=device)
    x_coords = torch.linspace(0, 1, target_size, device=device)

    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

    textures[:, 8] = Y.unsqueeze(0).expand(B, -1, -1)  # y_norm
    textures[:, 9] = X.unsqueeze(0).expand(B, -1, -1)  # x_norm

    # === CHANNEL 10: Eccentricity ===
    # Distance from image center, normalized to [0, 1]
    center_y, center_x = 0.5, 0.5
    eccentricity = torch.sqrt((Y - center_y)**2 + (X - center_x)**2)
    eccentricity = eccentricity / eccentricity.max()  # Normalize to [0,1]

    textures[:, 10] = eccentricity.unsqueeze(0).expand(B, -1, -1)

    # === CHANNEL 11: Simple Saliency ===
    # NOTE (v0.1 LIMITATION): Reusing Sobel magnitude as saliency proxy
    # This is a DUPLICATE of channel 7, wasting 1/13 texture capacity
    # TODO (v0.2): Replace with unique feature (gradient orientation, texture entropy)
    # See AUDIT_FINDINGS.md Finding #1
    textures[:, 11] = textures[:, 7]  # Same as edge magnitude

    # === CHANNEL 12: Luminance ===
    # NOTE (v0.1 LIMITATION): Reusing LAB L* channel
    # This is a DUPLICATE of channel 3, wasting 1/13 texture capacity
    # TODO (v0.2): Replace with unique feature (color variance, local contrast)
    # See AUDIT_FINDINGS.md Finding #1
    textures[:, 12] = textures[:, 3]  # Same as L*

    return textures


# === TESTS ===

def test_texture_array():
    """Test texture array generation."""
    print("Testing texture array generation...")

    # Create dummy RGB image
    image = torch.rand(2, 3, 512, 512)

    # Generate textures
    textures = generate_texture_array(image, target_size=32)

    # Check shape
    assert textures.shape == (2, 13, 32, 32), f"Expected (2, 13, 32, 32), got {textures.shape}"

    # Check value ranges
    assert textures.min() >= 0.0, "Texture values should be >= 0"
    assert textures.max() <= 1.0, "Texture values should be <= 1"

    # Check specific channels
    # RGB should be in [0,1]
    assert (textures[:, 0:3] >= 0).all() and (textures[:, 0:3] <= 1).all()

    # Position channels should span [0,1]
    assert textures[:, 8].min() < 0.1, "y should start near 0"
    assert textures[:, 8].max() > 0.9, "y should end near 1"
    assert textures[:, 9].min() < 0.1, "x should start near 0"
    assert textures[:, 9].max() > 0.9, "x should end near 1"

    # Eccentricity should peak at corners
    assert textures[:, 10].max() > 0.7, "Eccentricity should be high at corners"

    print("✓ Texture array tests passed")


if __name__ == "__main__":
    test_texture_array()
