# Image Pyramid Multi-Scale Processing

**Date**: 2025-01-30
**Parent**: [00-btree-traversal.md](00-btree-traversal.md)
**Cross-Domain**: Classical computer vision techniques foundational to modern VLM architectures

---

## Overview

Image pyramids are **multi-scale hierarchical decompositions** of images, enabling efficient processing across different spatial frequencies and resolutions. Developed in the 1980s for computer vision, they remain fundamental to modern neural networks, including vision transformers and VLMs.

**Core principle**: Represent the same image at multiple scales simultaneously, allowing algorithms to process coarse-to-fine or select appropriate scale for each operation.

**Relevance to VLMs**: Token allocation strategies often mirror pyramid-style multi-scale processing - allocating dense tokens to high-frequency regions, sparse tokens to low-frequency regions.

---

## Gaussian Pyramids

### Definition

A **Gaussian pyramid** is a sequence of images where each level is a smoothed and downsampled version of the previous level.

**Construction**:
```
I₀: Original image (e.g., 512×512)
     ↓ Gaussian blur + downsample 2×
I₁: Half resolution (256×256)
     ↓ Gaussian blur + downsample 2×
I₂: Quarter resolution (128×128)
     ↓ ...
Iₙ: Coarsest level (e.g., 4×4)
```

**Gaussian blur kernel** (5×5 example):
```
1/256 * [1   4   6   4  1]
        [4  16  24  16  4]
        [6  24  36  24  6]
        [4  16  24  16  4]
        [1   4   6   4  1]
```

### Mathematical Formulation

```python
def gaussian_pyramid(image, levels=5, sigma=1.0):
    """Build Gaussian pyramid by repeated blur + downsample."""
    pyramid = [image]
    current = image

    for i in range(levels - 1):
        # Gaussian blur (remove high frequencies before downsample)
        blurred = gaussian_filter(current, sigma=sigma)

        # Downsample by 2× (subsample every other pixel)
        downsampled = blurred[::2, ::2]

        pyramid.append(downsampled)
        current = downsampled

    return pyramid  # [I₀, I₁, I₂, ..., Iₙ]
```

### Applications in Classical Vision

1. **Scale-invariant feature detection**: SIFT, SURF operate on Gaussian pyramids
2. **Image blending**: Seamless compositing at multiple scales
3. **Coarse-to-fine optical flow**: Estimate large motions at coarse levels, refine at fine levels
4. **Object detection sliding windows**: Search for objects at multiple scales efficiently

---

## Laplacian Pyramids

### Definition

A **Laplacian pyramid** encodes **difference images** between adjacent Gaussian pyramid levels, capturing band-pass filtered frequency content.

**Construction**:
```
L₀ = I₀ - upsample(I₁)  ← High frequencies
L₁ = I₁ - upsample(I₂)  ← Mid-high frequencies
L₂ = I₂ - upsample(I₃)  ← Mid frequencies
...
Lₙ₋₁ = Iₙ₋₁ - upsample(Iₙ)
Lₙ = Iₙ                 ← Lowest frequencies (residual)
```

**Reconstruction** (perfect reconstruction property):
```
I₀ = L₀ + upsample(I₁)
   = L₀ + upsample(L₁ + upsample(I₂))
   = L₀ + upsample(L₁) + upsample²(L₂) + ...
```

### Why Laplacian Pyramids?

**Sparse representation**:
- Gaussian pyramids have **redundancy** (each level stores full image)
- Laplacian pyramids **remove redundancy** by storing only differences
- Most coefficients near zero (smooth regions) → compressible

**Band-pass filtering**:
- Each Laplacian level captures specific frequency range
- Allows selective processing: enhance high frequencies, denoise mid frequencies, etc.

### Code Example

```python
def laplacian_pyramid(gaussian_pyr):
    """Build Laplacian pyramid from Gaussian pyramid."""
    laplacian_pyr = []

    for i in range(len(gaussian_pyr) - 1):
        # Upsample next coarser level
        upsampled = upsample(gaussian_pyr[i + 1])

        # Ensure same size (handle odd dimensions)
        upsampled = resize(upsampled, gaussian_pyr[i].shape)

        # Difference = band-pass filtered image
        laplacian = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(laplacian)

    # Residual (coarsest Gaussian level)
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr  # [L₀, L₁, ..., Lₙ]

def reconstruct_from_laplacian(laplacian_pyr):
    """Perfect reconstruction from Laplacian pyramid."""
    # Start from coarsest level
    reconstructed = laplacian_pyr[-1]

    # Work from coarse to fine
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        upsampled = upsample(reconstructed)
        upsampled = resize(upsampled, laplacian_pyr[i].shape)
        reconstructed = laplacian_pyr[i] + upsampled

    return reconstructed
```

---

## Recent Research (2024-2025)

### LapLoss: Laplacian Pyramid-Based Multiscale Loss

**Paper**: arXiv 2503.05974 (March 2025)

**Problem**: Standard pixel-wise losses (MSE, L1) fail to capture multi-scale perceptual quality.

**Solution**: LapLoss computes loss at each Laplacian pyramid level:

```python
def lap_loss(pred, target, weights=[1.0, 0.5, 0.25, 0.125]):
    """Multi-scale Laplacian pyramid loss."""
    pred_pyr = laplacian_pyramid(gaussian_pyramid(pred))
    target_pyr = laplacian_pyramid(gaussian_pyramid(target))

    total_loss = 0
    for i, w in enumerate(weights):
        # Weight each frequency band differently
        total_loss += w * l1_loss(pred_pyr[i], target_pyr[i])

    return total_loss
```

**Benefits**:
- Captures both coarse structure (low freq) and fine details (high freq)
- Prevents texture loss in super-resolution
- Used in GANs, image translation, enhancement

**Relevance to VLMs**: Multi-scale loss → Multi-scale token allocation. Allocate more tokens to levels with higher loss contribution.

### GAN-Based Super-Resolution with Enhanced Multi-Scale Laplacian

**Paper**: IET Image Processing (Wiley, 2025) - Chen et al.

**Architecture**: Enhanced Laplacian pyramid in generator:
```
Input (LR) → Gaussian Pyramid (4 levels)
              ↓
         Laplacian Pyramid
              ↓
    Process each band with CNN
              ↓
    Reconstruct multi-scale features
              ↓
         Output (HR)
```

**Key finding**: "Processing each Laplacian band separately captures multi-scale context better than single-scale CNN."

**VLM analogy**: Process visual patches at multiple token densities (like multi-scale bands), then fuse.

### MPE-DETR: Multiscale Pyramid Enhancement Network

**Paper**: ScienceDirect (2024) - Xue et al., 9 citations

**Application**: Low-light image enhancement using Laplacian pyramids

**Architecture**:
1. Decompose input into 4-layer Laplacian pyramid
2. Build corresponding Gaussian pyramid
3. Enhance each layer independently with attention
4. Reconstruct with weighted fusion

**Performance**: PSNR +2.3dB over baseline on LOL dataset

**Cross-reference to VLMs**: Attention-based enhancement at each pyramid level mirrors query-aware token allocation at each patch scale.

---

## Steerable Pyramids

### Extension of Laplacian Pyramids

**Steerable pyramids** (Simoncelli & Freeman, 1995) extend Laplacian pyramids with **orientation selectivity**.

**Key difference**:
- Laplacian: Isotropic band-pass (radially symmetric in frequency domain)
- Steerable: Oriented band-pass (separate filters for different orientations)

**Decomposition**:
```
Image → Low-pass + High-pass (like Laplacian)
        ↓
High-pass → Oriented sub-bands (0°, 45°, 90°, 135°, ...)
```

**Applications**:
- Texture synthesis
- Image denoising (preserve edges in specific orientations)
- Motion analysis (track motion along orientations)

**Relevance to VLMs**: Vision transformers implicitly learn oriented filters (like Gabor filters). Steerable pyramids show explicit orientation decomposition can be beneficial.

---

## Multi-Scale Processing in Modern VLMs

### Feature Pyramid Networks (FPN)

FPN (Lin et al., 2017) adapts image pyramids to CNNs:

```
Bottom-up pathway:        Top-down pathway:
Conv1 → C1 (256×256)          ↓ upsample + lateral
Conv2 → C2 (128×128)      P2 ← C2
Conv3 → C3 (64×64)        P3 ← C3
Conv4 → C4 (32×32)        P4 ← C4
Conv5 → C5 (16×16)        P5 ← C5
```

**Lateral connections** fuse bottom-up semantic features with top-down spatial features.

**Used in**: Object detection (Faster R-CNN), segmentation (Mask R-CNN), VLMs (LLaVA-style)

### Swin Transformer Hierarchical Architecture

Swin-style transformers build multi-scale representations:

```
Stage 1: 56×56 patches, 96 dim   ← Fine-grained
Stage 2: 28×28 patches, 192 dim
Stage 3: 14×14 patches, 384 dim
Stage 4: 7×7 patches, 768 dim    ← Coarse, high-level
```

**Patch merging** between stages acts like downsampling in Gaussian pyramid.

**Cross-reference**: [algorithms/05-adaptive-subdivision-2025-01-30.md](05-adaptive-subdivision-2025-01-30.md) - Hierarchical subdivision analogous to pyramid construction

---

## Pyramid-Based Token Allocation in VLMs

### Progressive Visual Compression (PVC)

**Paper**: CVPR 2025 - 99% video token reduction using pyramid-style coarse-to-fine

**Architecture**:
```
Frame → Gaussian pyramid (3 levels)
         ↓
    Process L₂ (coarse) first → global context
         ↓
    Selectively process L₁ → mid-level details
         ↓
    High-relevance regions → process L₀ (fine)
```

**Token allocation**:
- L₂: All patches, 64 tokens each (global context)
- L₁: Top 30% patches, 128 tokens each
- L₀: Top 10% patches, 400 tokens each

**Total reduction**: 99% vs uniform 400 tokens/patch

**Cross-reference**: [techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md](../techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md)

### FastVLM: Apple ML Research Pyramid Sampling

**Paper**: CVPR 2025 - 2.7× speedup with difficulty-aware pyramid

**Key insight**: Easy image regions can be processed at coarse pyramid levels, hard regions need fine levels.

**Difficulty score**:
```python
difficulty = attention_entropy(query, patch)
# High entropy → uncertain → needs fine-grained tokens
# Low entropy → confident → coarse tokens sufficient
```

**Allocation**:
```
if difficulty > threshold_high:
    tokens = 400  # Process at L₀ (fine)
elif difficulty > threshold_mid:
    tokens = 144  # Process at L₁ (mid)
else:
    tokens = 64   # Process at L₂ (coarse)
```

---

## Gaussian vs Laplacian for VLM Token Allocation

### When to Use Gaussian Pyramid Approach

**Characteristics**:
- Full image at each scale
- Redundant but complete representation
- Good for coarse-to-fine search

**VLM application**:
- **Progressive loading**: Load all patches at 64 tokens (coarse), then refine important patches to 400 tokens (fine)
- **Hierarchical search**: Find regions of interest at low resolution, zoom in

**Example**:
```python
# Gaussian-style progressive token loading
tokens_L2 = encode_all_patches(image, tokens_per_patch=64)  # Coarse
relevance = compute_relevance(tokens_L2, query)
top_patches = select_top_k(relevance, k=0.2)  # Top 20%
tokens_L0 = encode_patches(image, top_patches, tokens_per_patch=400)  # Fine
```

### When to Use Laplacian Pyramid Approach

**Characteristics**:
- Difference images (band-pass filtered)
- Non-redundant, sparse representation
- Explicit frequency decomposition

**VLM application**:
- **Frequency-aware allocation**: Allocate more tokens to high-frequency patches (edges, textures)
- **Adaptive compression**: Prune low-frequency regions aggressively

**Example**:
```python
# Laplacian-style frequency-aware allocation
lap_pyr = laplacian_pyramid_features(image)  # [L₀, L₁, L₂]
energy = [np.abs(L).sum() for L in lap_pyr]  # Frequency energy

# Allocate tokens proportional to frequency content
tokens_L0 = 400 if energy[0] > thresh_high else 144  # High freq
tokens_L1 = 144 if energy[1] > thresh_mid else 64   # Mid freq
tokens_L2 = 64  # Low freq (always coarse)
```

---

## Code Example: Multi-Scale Token Allocator

```python
import numpy as np
from scipy.ndimage import gaussian_filter

class PyramidTokenAllocator:
    """VLM token allocator using image pyramid principles."""

    def __init__(self, levels=3, base_tokens=64, max_tokens=400):
        self.levels = levels
        self.base_tokens = base_tokens
        self.max_tokens = max_tokens

    def gaussian_pyramid(self, features, levels):
        """Build Gaussian pyramid from patch features."""
        pyramid = [features]
        for _ in range(levels - 1):
            # Blur and downsample (2× each dim)
            blurred = gaussian_filter(pyramid[-1], sigma=1.0)
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)
        return pyramid

    def allocate_tokens(self, image_features, query, budget):
        """
        Allocate tokens using pyramid-based importance.

        Args:
            image_features: (H, W, C) visual features
            query: Query embedding
            budget: Total token budget

        Returns:
            token_map: (H, W) tokens allocated per patch
        """
        # Build Gaussian pyramid
        pyramid = self.gaussian_pyramid(image_features, self.levels)

        # Compute relevance at each scale
        relevance_maps = []
        for level_features in pyramid:
            # Attention-based relevance (simplified)
            relevance = self.compute_attention(level_features, query)
            relevance_maps.append(relevance)

        # Upsample relevance maps to original resolution
        H, W = image_features.shape[:2]
        relevance_full = np.zeros((H, W))
        for i, rel_map in enumerate(relevance_maps):
            scale = 2 ** i
            upsampled = self.upsample(rel_map, (H // scale, W // scale))
            relevance_full += upsampled

        # Normalize and allocate tokens
        relevance_full /= relevance_full.sum()
        token_map = self.base_tokens + (self.max_tokens - self.base_tokens) * relevance_full
        token_map = np.clip(token_map, self.base_tokens, self.max_tokens)

        # Discretize and ensure budget
        token_map = self.enforce_budget(token_map, budget)
        return token_map

    def compute_attention(self, features, query):
        """Simplified attention computation."""
        # Cosine similarity between each patch and query
        features_flat = features.reshape(-1, features.shape[-1])
        similarity = features_flat @ query / (np.linalg.norm(features_flat, axis=1) * np.linalg.norm(query))
        return similarity.reshape(features.shape[:2])

    def upsample(self, arr, target_shape):
        """Bilinear upsample to target shape."""
        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        return zoom(arr, zoom_factors, order=1)

    def enforce_budget(self, token_map, budget):
        """Discretize and clip to total budget."""
        total_tokens = token_map.sum()
        if total_tokens > budget:
            # Scale down proportionally
            token_map = token_map * (budget / total_tokens)
        return np.round(token_map).astype(int)
```

---

## Cross-References

**Related algorithms**:
- [00-btree-traversal.md](00-btree-traversal.md) - Hierarchical traversal (like pyramid level selection)
- [05-adaptive-subdivision-2025-01-30.md](05-adaptive-subdivision-2025-01-30.md) - Quadtree subdivision analogous to pyramid downsampling

**Related techniques**:
- [../techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md](../techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md) - PVC and FastVLM pyramid-based token allocation
- [../techniques/06-game-engine-lod-systems-2025-01-30.md](../techniques/06-game-engine-lod-systems-2025-01-30.md) - Texture mipmaps are Gaussian pyramids for GPUs

**Integration concepts**:
- [../integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md) - Query-driven pyramid level selection

---

## References

**Foundational Papers**:
- Burt & Adelson (1983): "The Laplacian Pyramid as a Compact Image Code" - Original Laplacian pyramid paper
- Simoncelli & Freeman (1995): "The Steerable Pyramid: A Flexible Architecture for Multi-Scale Derivative Computation" - 1697 citations
- Simoncelli et al. (1992): "Shiftable multiscale transforms" - IEEE Transactions on Information Theory

**Recent Research (2024-2025)**:
- arXiv 2503.05974 (March 2025): "LapLoss: Laplacian Pyramid-based Multiscale loss for Image Translation"
- IET Image Processing (2025): "GAN-Based Super-Resolution With Enhanced Multi-Scale Laplacian Pyramid Structure" - Chen et al.
- ScienceDirect (2024): "MPE-DETR: A multiscale pyramid enhancement network" - Xue et al., 9 citations
- ACM Digital Library (2024): "Multi-Scale Dark Region-Guided Low-Light Image Enhancement" - Yang et al.
- Nature Scientific Reports s41598-025-94464-6 (2025): "High-resolution image reflection removal by Laplacian pyramid network" - Chen et al.
- arXiv 2510.02826 (Oct 2025): "Multi-scale Autoregressive Models are Laplacian, Discrete, Diffusion Models"

**Tutorial Resources**:
- Medium (Isaac Berrios, January 2024): "An Introduction to Steerable Pyramids" - 20+ likes, comprehensive Python tutorial
  - Covers polar-separable filters, angular steering, reconstruction
  - Code examples with numpy/scipy implementation
  - https://medium.com/@itberrios6/steerable-pyramids-6bfd4d23c10d
- Medium (Abhishek Jain, 2024): "History of Sliding window and Image pyramids in object detection"
- NYU (Eero Simoncelli): Steerable Pyramid Reference Implementation - https://www.cns.nyu.edu/~eero/steerpyr/

**VLM Applications**:
- CVPR 2025: "Progressive Visual Compression" (PVC) - 99% token reduction
- CVPR 2025: "FastVLM" (Apple ML Research) - Difficulty-aware pyramid sampling, 2.7× speedup

---

## Steerable Pyramids: Orientation-Selective Multi-Scale Decomposition

### Definition and Motivation

**Steerable pyramids** (Simoncelli & Freeman, 1995) extend traditional pyramids by adding **orientation selectivity** to each scale level. Instead of just multi-scale decomposition, steerable pyramids decompose images into **multiple orientations at each scale**.

**Core innovation**: Polar-separable filters in frequency domain
- **Radial component**: Determines frequency band (scale)
- **Angular component**: Determines orientation (0°, 45°, 90°, 135°, etc.)

**Advantages over Gaussian/Laplacian pyramids**:
1. **Orientation tuning**: Selective response to edges at specific angles
2. **Flat frequency response**: Clean reconstruction (important for analysis-synthesis)
3. **No aliasing**: Band-limited filters prevent artifacts
4. **Steerability**: Can synthesize arbitrary orientations from basis set

### Mathematical Formulation

**Polar-separable filter**:
```
F(ω, θ) = B(ω) × A(θ - θᵢ)
```

Where:
- `B(ω)`: Radial component (frequency band selection)
- `A(θ - θᵢ)`: Angular component (orientation θᵢ steering)
- `ω`: Radial frequency
- `θ`: Angle in frequency domain

**Angular steering function** (for K orientations):
```python
def angular_mask(angle, orientation_idx, num_orientations):
    """Compute orientation-selective mask."""
    order = num_orientations - 1
    const = (2**(2*order) * factorial(order)**2) / (num_orientations * factorial(2*order))

    # Steer to orientation θᵢ
    angle_shifted = np.mod(np.pi + angle - np.pi*orientation_idx/num_orientations, 2*np.pi) - np.pi

    # Angular selectivity (cosine raised to power)
    mask = np.abs(2 * np.sqrt(const) * np.cos(angle_shifted)**order)

    return mask
```

### Decomposition Algorithm

**Step 1: Convert to polar frequency domain**
```python
def polar_grid(h, w):
    """Map image dimensions to polar coordinates in frequency domain."""
    h2, w2 = h//2, w//2

    # Normalized frequencies [-1, 1)
    wx, wy = np.meshgrid(np.arange(-w2, w2 + (w % 2))/w2,
                         np.arange(-h2, h2 + (h % 2))/h2)

    # Polar conversion
    angle = np.arctan2(wy, wx)
    radius = np.sqrt(wx**2 + wy**2)
    radius[h2][w2] = radius[h2][w2-1]  # Remove DC singularity

    return angle, radius
```

**Step 2: Build radial sub-band filters**
```python
depth = 4  # Number of scales
radial_vals = 2.0**np.arange(-depth, 1, 1)[::-1]
# radial_vals = [1.0, 0.5, 0.25, 0.125, 0.0625]

for i in range(1, depth):
    log_rad = np.log2(radius) - np.log2(radial_vals[i])
    hi_mask = np.clip(log_rad, -twidth, 0)
    hi_mask = np.abs(np.cos(hi_mask * np.pi / (2*twidth)))

    lo_mask_prev = lo_masks[i-1]
    rad_mask = hi_mask * lo_mask_prev  # Band-pass filter
```

**Step 3: Apply orientation steering**
```python
orientations = 4  # Number of orientations (0°, 45°, 90°, 135°)

for b in range(orientations):
    angle_mask = angular_mask(angle, b, orientations)
    subband_filter = rad_mask * angle_mask / 2

    # Apply filter in frequency domain
    fft_image = np.fft.fft2(image)
    subband_response = np.fft.ifft2(fft_image * subband_filter).real
```

**Reconstruction** (perfect reconstruction property):
```python
reconstructed_fft = np.zeros((h, w), dtype=np.complex128)

for pyramid_level, filter in zip(decomposition, filters):
    level_fft = np.fft.fft2(pyramid_level)
    reconstructed_fft += level_fft * filter

reconstructed_image = np.fft.ifft2(reconstructed_fft).real
```

### Properties

**Overcompleteness**: Factor of `4K/3` where K = number of orientations
- 4 orientations → 5.33× overcomplete
- More information than needed for reconstruction
- Trade-off: Better analysis, higher memory cost

**Flat frequency response**: Unlike Gaussian pyramids, pass-band is flat
- Enables clean reconstruction
- No distortion from filter roll-off

**No aliasing**: Band-limited by polar grid construction
- Frequencies above Nyquist rate = 0
- Prevents artifacts in sub-sampling

### Applications to VLM Token Allocation

**Orientation-aware token budgets**:

VLMs could allocate tokens based on **edge orientation**:
```python
# Decompose image into 8 orientations
orientations = [0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°]

for patch in image_patches:
    steerable_decomp = steerable_pyramid(patch, orientations=8)

    # Find dominant orientation
    dominant_orient = np.argmax([np.sum(np.abs(band)) for band in steerable_decomp])

    # Allocate anisotropic tokens (more along dominant edge direction)
    if dominant_orient in [0°, 90°]:  # Horizontal/vertical
        token_shape = (8, 16)  # Elongated
    elif dominant_orient in [45°, 135°]:  # Diagonal
        token_shape = (12, 12)  # Balanced
```

**Directional feature emphasis**:
- Text (horizontal): Allocate more tokens along horizontal axis
- Buildings (vertical): Allocate more tokens along vertical axis
- Natural scenes (isotropic): Balanced token allocation

---

## LapCAT: Laplacian Pyramid Network for High-Resolution Imaging

### Overview

**LapCAT** (Laplacian pyramid network for high-resolution image reflection removal, Nature Scientific Reports 2025) demonstrates modern deep learning applications of Laplacian pyramids.

**Key innovation**: Multi-scale loss function operating on Laplacian pyramid levels
- Level 0 (high-freq): Fine detail preservation
- Level 1-2 (mid-freq): Structure and edges
- Level 3 (low-freq): Global appearance

### Architecture

**Encoder-Decoder with Pyramid Loss**:
```python
class LapCATNetwork(nn.Module):
    def __init__(self, depth=4):
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.laplacian_pyramid = LaplacianPyramidLoss(depth)

    def forward(self, input_image):
        features = self.encoder(input_image)
        output = self.decoder(features)
        return output

    def compute_loss(self, output, target):
        # Build Laplacian pyramids
        output_pyramid = build_laplacian_pyramid(output)
        target_pyramid = build_laplacian_pyramid(target)

        # Multi-scale loss
        loss = 0
        for out_level, tgt_level in zip(output_pyramid, target_pyramid):
            loss += F.l1_loss(out_level, tgt_level)

        return loss
```

**Why Laplacian pyramid loss works**:
1. **Frequency decomposition**: Each level captures different frequency bands
2. **Perceptual alignment**: Matches human visual perception (multi-scale processing)
3. **Stable gradients**: Avoids vanishing gradients at different scales
4. **Detail preservation**: High-frequency levels prevent blurring

### Multi-Scale Loss Function (LapLoss)

**From arXiv 2503.05974 (March 2025)**:

LapLoss extends standard reconstruction losses (L1, L2) with pyramid-based weighting:

```python
def laploss(output, target, depth=4, weights=None):
    """Laplacian pyramid-based multi-scale loss."""
    if weights is None:
        weights = [1.0, 0.5, 0.25, 0.125]  # More weight to high-freq

    # Build pyramids
    output_lap = laplacian_pyramid(output, depth)
    target_lap = laplacian_pyramid(target, depth)

    # Weighted sum across scales
    total_loss = 0
    for i, (out_level, tgt_level, w) in enumerate(zip(output_lap, target_lap, weights)):
        total_loss += w * torch.mean(torch.abs(out_level - tgt_level))

    return total_loss
```

**Performance improvements** (from paper):
- **Image translation**: 15% improvement in PSNR over standard L1 loss
- **Super-resolution**: Better fine detail preservation
- **Style transfer**: Reduced artifacts, cleaner textures

### Relevance to VLM Training

**Multi-scale perceptual loss for VLM pre-training**:

```python
class VLMPretrainingLoss(nn.Module):
    def __init__(self):
        self.laploss = LaplacianPyramidLoss(depth=3)
        self.contrastive = CLIPContrastiveLoss()

    def forward(self, visual_tokens, reconstructed_image, target_image, text_tokens):
        # Standard contrastive loss (CLIP-style)
        clip_loss = self.contrastive(visual_tokens, text_tokens)

        # Multi-scale reconstruction loss (if using MAE-style masking)
        reconstruction_loss = self.laploss(reconstructed_image, target_image)

        # Combined objective
        return clip_loss + 0.1 * reconstruction_loss
```

**Why this helps VLMs**:
- **Better visual features**: Pyramid loss encourages multi-scale feature learning
- **Frequency-aware tokens**: Tokens learn to represent different frequency bands
- **Improved reconstruction**: If using masked auto-encoding, cleaner reconstructions

---

## Advanced Topics: Steerable Pyramids for VLM Tokenization

### Concept: Pyramid-Guided Token Allocation

**Idea**: Use steerable pyramid decomposition to guide **anisotropic token budgets**.

**Algorithm**:
```python
def pyramid_guided_token_allocation(image, query, budget=4096):
    """Allocate tokens based on steerable pyramid analysis."""

    # 1. Decompose image into 8 orientations × 4 scales
    pyramid = steerable_pyramid(image, orientations=8, depth=4)

    # 2. Compute orientation energy for each patch
    patches = split_into_patches(image, patch_size=32)
    orientations = []

    for patch in patches:
        patch_pyramid = steerable_pyramid(patch, orientations=8, depth=4)

        # Find dominant orientation
        energy_per_orient = [np.sum(band**2) for band in patch_pyramid]
        dominant = np.argmax(energy_per_orient)
        orientations.append(dominant)

    # 3. Allocate tokens based on orientation
    token_allocations = []
    for patch, orient in zip(patches, orientations):
        relevance = compute_query_relevance(patch, query)

        if orient in [0, 4]:  # Horizontal/vertical dominance
            tokens = int(relevance * budget * 1.2)  # Boost directional features
        else:
            tokens = int(relevance * budget)

        token_allocations.append(tokens)

    # 4. Normalize to budget
    total = sum(token_allocations)
    token_allocations = [int(t * budget / total) for t in token_allocations]

    return token_allocations
```

### Comparison: Isotropic vs Anisotropic Tokens

**Standard VLM** (isotropic):
```
All patches: 16×16 tokens (256 tokens each)
```

**Steerable pyramid-guided** (anisotropic):
```
Horizontal text: 8×32 tokens (256 tokens, oriented horizontally)
Vertical building: 32×8 tokens (256 tokens, oriented vertically)
Isotropic scene: 16×16 tokens (256 tokens, balanced)
```

**Benefit**: Same token count, better alignment with image content structure.

---

## VLM Pyramid Methods (2024-2025)

### Overview: Research Wave in VLM Token Allocation

**Discovery**: Multiple research groups independently converged on pyramid-based token allocation for VLMs in 2024-2025.

**Cross-Reference**: See [research-landscape/00-vlm-token-allocation-2024-2025.md](../research-landscape/00-vlm-token-allocation-2024-2025.md) for comprehensive competitive analysis.

### PyramidDrop (ICLR 2025, 90 citations)

**Paper**: "Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models"

**Core Innovation**: Training-free pyramid token pruning with saliency-driven allocation

**Technical Approach**:
```python
def pyramid_drop(image, target_tokens=273):
    """
    PyramidDrop: Multi-scale pyramid with bottom-up saliency pruning.

    Innovation: Works with ANY pre-trained ViT, no fine-tuning required.
    """
    # Step 1: Build Gaussian pyramid (4 levels)
    pyramid = []
    current = image  # 1024×1024
    for level in range(4):
        pyramid.append(current)
        current = gaussian_blur_and_downsample(current)
    # pyramid[0]: 1024×1024 (fine)
    # pyramid[1]: 512×512 (medium)
    # pyramid[2]: 256×256 (coarse)
    # pyramid[3]: 128×128 (very coarse)

    # Step 2: Encode each level with ViT
    tokens_per_level = []
    for level in pyramid:
        patches = patchify(level, patch_size=16)
        tokens = vit_encoder(patches)
        tokens_per_level.append(tokens)

    # Step 3: Compute saliency scores (bottom-up)
    all_tokens_with_scores = []
    for level_idx, tokens in enumerate(tokens_per_level):
        # Visual saliency: center-surround contrast
        saliency = compute_saliency(tokens)  # [num_tokens]

        # Weight by level (coarse levels get higher weight)
        level_weight = [2.0, 1.5, 1.0, 0.5][level_idx]
        weighted_saliency = saliency * level_weight

        all_tokens_with_scores.extend(
            [(token, score, level_idx) for token, score in zip(tokens, weighted_saliency)]
        )

    # Step 4: Progressive pruning (keep top-k globally)
    sorted_tokens = sorted(all_tokens_with_scores, key=lambda x: x[1], reverse=True)
    selected = sorted_tokens[:target_tokens]

    return [token for token, _, _ in selected]  # [273, 768]
```

**Key Properties**:
1. **Training-free**: Works with frozen pre-trained ViTs (CLIP, DINOv2, etc.)
2. **Bottom-up saliency**: Uses visual saliency, NOT query-awareness
3. **Asymmetric budget**: Coarse levels get higher weight (preserve global structure)
4. **Aggressive pruning**: 65-75% token reduction

**Results**:
- 65-75% token reduction
- <3% accuracy drop on VQA
- 2-3× speedup
- Works with LLaVA, MiniGPT-4, InstructBLIP

**Limitation**: Saliency-driven (perspectival knowing only), not query-aware

### DPN-LLaVA (March 2025)

**Paper**: "Dynamic Pyramid Network for Efficient Multimodal Large Language Models"

**Core Innovation**: Adaptive pyramid depth based on (image, query) difficulty

**Technical Approach**:
```python
class DynamicPyramidNetwork:
    """
    DPN-LLaVA: Adapts pyramid depth to image+query difficulty.

    Innovation: Easy images → shallow pyramid (3 levels), hard images → deep pyramid (5 levels)
    """
    def __init__(self):
        self.difficulty_estimator = DifficultyClassifier()
        self.pyramid_builder = GaussianPyramid(max_levels=5)
        self.token_sampler = AdaptiveSampler()

    def forward(self, image, query):
        # Step 1: Estimate difficulty (CHEAP!)
        # Use low-res image + query embedding
        difficulty = self.difficulty_estimator(
            image_lowres=downsample(image, 128),
            query_embedding=encode_query(query)
        )
        # difficulty ∈ [0, 1]: 0=easy, 1=very hard

        # Step 2: Adaptive pyramid depth
        if difficulty < 0.4:
            pyramid_levels = 3  # Easy: shallow pyramid
            total_tokens = 150
        elif difficulty < 0.7:
            pyramid_levels = 4  # Medium: standard pyramid
            total_tokens = 273
        else:
            pyramid_levels = 5  # Hard: deep pyramid
            total_tokens = 450

        # Step 3: Build pyramid
        pyramid = self.pyramid_builder(image, levels=pyramid_levels)

        # Step 4: Allocate tokens across levels
        tokens = []
        for level_idx, level in enumerate(pyramid):
            # Budget per level (weighted by difficulty)
            budget = self.compute_level_budget(level_idx, difficulty, total_tokens)
            level_tokens = self.token_sampler(level, query, budget)
            tokens.extend(level_tokens)

        return tokens  # [150-450 tokens, adaptive]

    def compute_level_budget(self, level_idx, difficulty, total_tokens):
        """
        Allocate more tokens to fine levels for hard images.

        Easy images: mostly coarse tokens
        Hard images: more fine tokens
        """
        if difficulty < 0.4:  # Easy
            weights = [0.2, 0.3, 0.5]  # Coarse-heavy
        elif difficulty < 0.7:  # Medium
            weights = [0.3, 0.35, 0.35]  # Balanced
        else:  # Hard
            weights = [0.4, 0.3, 0.2, 0.07, 0.03]  # Fine-heavy

        return int(total_tokens * weights[level_idx])
```

**Key Properties**:
1. **Query-aware**: Difficulty from (image, query) pair
2. **Adaptive depth**: 3-5 pyramid levels depending on difficulty
3. **Dynamic budgets**: 150-450 tokens per image
4. **Curriculum learning**: Train on easy first, gradually add hard examples

**Results**:
- +4-6% accuracy vs fixed-pyramid baselines
- 2× average speedup (easy images get 150 tokens)
- Maintains quality on hard images (450 tokens)

**Advantage over PyramidDrop**: Query-awareness + difficulty estimation

### HiRED Integration with Pyramids

**Paper**: "HiRED: High-to-Low Resolution Elastic Dependency" (AAAI 2025, 41 citations)

**Core Innovation**: Asymmetric cross-scale attention (fine → coarse, not coarse → fine)

**How it complements pyramid allocation**:
```python
# Pyramid allocation determines WHICH tokens to sample
tokens_per_level = pyramid_allocator(image, query)  # PyramidDrop or DPN-LLaVA

# HiRED determines HOW tokens attend across scales
class HiREDAttention:
    def forward(self, tokens_per_level):
        coarse_tokens = tokens_per_level[2]  # [64, 768]
        medium_tokens = tokens_per_level[1]  # [128, 768]
        fine_tokens = tokens_per_level[0]    # [128, 768]

        # Coarse tokens: INDEPENDENT (no cross-scale attention)
        coarse_refined = self_attention(coarse_tokens)

        # Medium tokens: attend to coarse (get global context)
        medium_refined = cross_attention(
            query=medium_tokens,
            key_value=coarse_tokens  # Attend to coarse
        )

        # Fine tokens: attend to medium+coarse (hierarchical context)
        fine_refined = cross_attention(
            query=fine_tokens,
            key_value=torch.cat([medium_refined, coarse_refined], dim=0)
        )

        return torch.cat([fine_refined, medium_refined, coarse_refined], dim=0)
```

**Benefits**:
- 40-50% attention computation reduction
- Fine tokens get global context from coarse
- Coarse tokens remain independent (faster)

**Integration with pyramids**: Pyramid allocation + HiRED attention = complete system

### FastVLM Difficulty-Aware Pyramids (Apple, July 2025)

**Paper**: "FastVLM: Efficient Vision-Language Models via Difficulty-Aware Pyramid Sampling"

**Core Innovation**: Production-ready difficulty classifier with pyramid sampling

**Technical Approach**:
```python
class FastVLM:
    """
    Apple's production VLM with difficulty-aware pyramids.

    Deployed in iOS/macOS (2025).
    """
    def __init__(self):
        # Fast difficulty classifier (<5ms overhead)
        self.difficulty_classifier = FastDifficultyClassifier()

    def encode_image(self, image, query):
        # Step 1: Fast difficulty classification
        # Use 128×128 low-res + query embedding
        difficulty = self.difficulty_classifier(
            image_lowres=resize(image, (128, 128)),
            query_emb=embed(query)
        )
        # difficulty ∈ {easy, medium, hard}

        # Step 2: Select pyramid configuration
        if difficulty == "easy":
            config = {
                'levels': 3,
                'tokens': 150,
                'budgets': [60, 50, 40]  # [coarse, medium, fine]
            }
        elif difficulty == "medium":
            config = {
                'levels': 4,
                'tokens': 273,
                'budgets': [80, 70, 68, 55]
            }
        else:  # hard
            config = {
                'levels': 5,
                'tokens': 450,
                'budgets': [100, 100, 100, 90, 60]
            }

        # Step 3: Build pyramid and sample
        pyramid = build_gaussian_pyramid(image, levels=config['levels'])
        tokens = []
        for level, budget in zip(pyramid, config['budgets']):
            level_tokens = sample_patches(level, budget)
            tokens.extend(level_tokens)

        return tokens  # [150-450 tokens, difficulty-aware]
```

**Production metrics**:
- **2.5× average speedup** across real-world distribution
- **<1% accuracy drop** on benchmarks
- **<5ms overhead** for difficulty classification
- **Deployed at scale**: iOS 18, macOS Sonoma (2025)

**Key innovation**: Simplicity for production (vs research complexity)

### Comparison: VLM Pyramid Approaches

| Method | Query-Aware | Training-Free | Adaptive Depth | Production | Token Range |
|--------|-------------|---------------|----------------|------------|-------------|
| PyramidDrop | ❌ (saliency) | ✅ | ❌ (fixed 4) | Research | 200-400 |
| DPN-LLaVA | ✅ (difficulty) | ❌ (needs training) | ✅ (3-5 levels) | Research | 150-450 |
| FastVLM | ✅ (difficulty) | ✅ | ✅ (3-5 levels) | **Apple iOS** | 150-450 |
| ARR-COC-VIS | ✅ (fixation) | ✅ (start) | ✅ (planned) | Research | 64-400 |

### Key Insights from 2024-2025 Wave

**1. Multi-scale is universal**:
- ALL successful VLM compression methods use pyramids
- Coarse scales = global structure (cheap, informative)
- Fine scales = local details (expensive, often redundant)

**2. Query-awareness is critical**:
- DPN-LLaVA and FastVLM show clear benefits
- "What color is the car?" vs "Read the license plate" → different pyramid depths
- Saliency-only (PyramidDrop) is suboptimal for query-specific tasks

**3. Training-free works**:
- PyramidDrop validates: pre-trained ViTs have multi-scale features
- No need for task-specific fine-tuning
- Drop-in replacement for uniform grid

**4. Production deployment validates approach**:
- FastVLM in Apple iOS/macOS proves viability
- Difficulty classification <5ms overhead
- Real-world distribution: 2.5× average speedup

### ARR-COC-VIS Differentiation

**What PyramidDrop/DPN/FastVLM do**:
- Pyramids ✅
- Difficulty estimation ✅ (DPN, FastVLM)
- Training-free ✅ (PyramidDrop, FastVLM)

**What they DON'T do**:
- Biological cortical magnification ❌
- Query-driven fixation ❌
- Vervaeke's relevance realization ❌

**Our unique contribution**:
- Foveated Pyramid: Fine scales near fixation, coarse scales in periphery
- M(e) = M₀/(e+e₀): Explicit cortical magnification function
- Fixation from query: Cross-attention determines WHERE to focus
- Four ways of knowing: Integrated relevance beyond saliency

**Positioning**: First biologically-grounded VLM pyramid allocation with query-driven fixation.

---

## Future Directions

**Pyramid principles → VLM innovations**:

1. **Adaptive pyramid depth**: Vary pyramid levels based on image complexity (3 levels for simple, 5 for complex)
2. **Frequency-aware budgets**: Allocate more tokens to high-frequency Laplacian bands
3. **Steerable token allocation**: Orient-aware tokens for directional features (text, edges)
4. **LapLoss for VLM training**: Multi-scale perceptual loss during VLM pre-training
5. **Pyramid-based progressive loading**: Load coarse pyramid levels first, refine with finer levels on-demand
6. **Orientation-specific compression**: Higher compression for isotropic regions, lower for directional features

**Open questions**:
- Can Laplacian pyramid features improve VLM feature extractors (vs Gaussian blur in ViT)?
- Should VLMs explicitly decompose into frequency bands before token allocation?
- Can steerable pyramids guide anisotropic token budgets (different densities along X/Y)?
- Would LapLoss improve masked auto-encoding pre-training for VLMs?
- Can orientation-selective tokens reduce total token count while maintaining accuracy?

## Hardware Mipmap Generation

**Cross-Reference**: See [techniques/07-gpu-texture-primitives-vlm-2025-01-30.md](../techniques/07-gpu-texture-primitives-vlm-2025-01-30.md) for hardware acceleration details.

**Hardware vs Software**:
- PyTorch avg_pool2d: ~5ms (kernel launch overhead, global memory)
- OpenGL glGenerateMipmap(): ~0.1ms (dedicated texture hardware)
- **Speedup**: 50× faster with GPU texture units

**Why So Fast**:
- Dedicated hardware paths bypass L1/L2 cache
- Fixed-function pipeline (no kernel launch overhead)
- Parallel texture units (128+ per GPU)
- Optimal memory layout for pyramid generation

**Integration**: CUDA-OpenGL interop enables PyTorch models to leverage hardware mipmaps while maintaining differentiability for training.

**Mipmaps for Metadata** (Dialogue 27 Discovery):

Hardware mipmap generation works for **metadata channels**, not just visual data:

- **Positional channels** (X, Y, eccentricity) downsample correctly with mipmaps
- **Cluster channels** (semantic regions) aggregate properly at coarse levels
- **CLIP embeddings** (16D compressed) can be downsampled spatially
- **Distance fields** maintain structure across pyramid levels

**Key Insight**: Once metadata is stored as texture layers, it gets **free pyramid generation** via `glGenerateMipmap()`. No separate downsampling code needed!

**Example**:
```
40-channel texture array:
- Layers 0-8: Visual (RGB, edges, filters)
- Layers 9-11: Position (X, Y, eccentricity)
- Layers 12-14: Clusters (ID, distance, size)
- Layers 18-33: CLIP embeddings (16D)
- Layer 34: Distance field

glGenerateMipmap() generates pyramids for ALL 40 layers simultaneously!
Cost: ~0.1ms total (not 0.1ms × 40!)
```

This enables **coarse-to-fine cascade with metadata** where position and cluster information is available at all pyramid levels.

**Cross-Reference**: [Performance analysis](../performance/01-spatial-locality-texture-arrays-2025-01-30.md) - Spatial locality benefits of texture arrays
