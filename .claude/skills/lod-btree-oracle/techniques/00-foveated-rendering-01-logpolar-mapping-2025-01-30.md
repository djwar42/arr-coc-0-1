# Log-Polar Mapping for Foveated Rendering

**Dynamic Knowledge Addition** - Created 2025-01-30
**Parent**: [techniques/00-foveated-rendering.md](00-foveated-rendering.md)
**Source**: Web research (arXiv, IEEE, CVF 2024-2025)

## Overview

Log-polar mapping is the mathematical transformation that converts Cartesian image coordinates into a retinotopic (retina-like) representation, enabling efficient computational foveation. This is the **core mathematical foundation** underlying foveated rendering systems.

## Mathematical Foundation

### The Log-Polar Transformation

**Cartesian to Log-Polar**:
```
ρ = log(r)  where r = √(x² + y²)
θ = atan2(y, x)
```

**Properties**:
- **Scale invariance**: Scaling in Cartesian space = translation in log-polar space
- **Rotation invariance**: Rotation in Cartesian space = translation in θ dimension
- **Cortical magnification**: Exponential spatial resolution fall-off

###

 Why Log-Polar?

**Biological basis**:
- Human retina-to-cortex mapping follows log-polar transformation
- Foveal oversampling (dense photoreceptors) → cortical magnification
- V1 cortical representation is approximately log-polar

**Computational advantages**:
1. **Constant resolution** in log-polar space despite varying sampling density
2. **Natural foveation**: Dense sampling at center, sparse at periphery
3. **Efficient storage**: ~10x fewer pixels for same perceptual quality

### Cortical Magnification Factor

The **cortical magnification factor** M(e) describes how much cortical area is devoted to a given retinal eccentricity:

```
M(e) = M₀ / (e + e₀)
```

Where:
- e = eccentricity (degrees visual angle from fovea)
- M₀ = scaling constant (~17-20 mm/deg for humans)
- e₀ = eccentricity constant (~0.75 deg)

This hyperbolic relationship drives the log-polar transformation.

## Implementation for Foveated Rendering

### Forward Transform (Cartesian → Log-Polar)

**Purpose**: Sample image with foveal density pattern

```python
def cartesian_to_logpolar(image, gaze_x, gaze_y):
    """
    Transform Cartesian image to log-polar representation centered at gaze.

    Args:
        image: Input image (H x W)
        gaze_x, gaze_y: Gaze position in pixels

    Returns:
        logpolar_image: Retinotopic representation (ρ x θ)
    """
    # Shift origin to gaze position
    x_centered = x_coords - gaze_x
    y_centered = y_coords - gaze_y

    # Convert to polar
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)

    # Apply logarithm (with epsilon for numerical stability)
    rho = np.log(r + epsilon)

    # Sample image at log-polar coordinates
    logpolar_image = sample_image(image, rho, theta)

    return logpolar_image
```

**Result**: Image with foveal detail preserved, peripheral compressed

### Inverse Transform (Log-Polar → Cartesian)

**Purpose**: Reconstruct full image from compressed log-polar representation

```python
def logpolar_to_cartesian(logpolar_image, gaze_x, gaze_y, target_size):
    """
    Inverse transform: Log-polar → Cartesian for display.

    Args:
        logpolar_image: Retinotopic representation
        gaze_x, gaze_y: Gaze position
        target_size: Output image dimensions

    Returns:
        reconstructed_image: Cartesian image
    """
    # Create Cartesian grid
    x_grid, y_grid = create_grid(target_size)

    # Convert to polar relative to gaze
    x_centered = x_grid - gaze_x
    y_centered = y_grid - gaze_y
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)

    # Convert to log-polar coordinates
    rho = np.log(r + epsilon)

    # Sample from log-polar image
    reconstructed = sample_logpolar(logpolar_image, rho, theta)

    return reconstructed
```

### Resolution Allocation

**Key insight**: Log-polar space has **uniform sampling density**, but Cartesian reconstruction has **exponentially decreasing density** with eccentricity.

**Allocation strategy**:
```
Pixels at eccentricity e:
- Cartesian: N × (2π × e)  [circumference grows linearly]
- Log-polar: N × Δρ        [constant for all ρ]

Compression ratio at e:
CR(e) = (2π × e) / Δρ ≈ e for large e
```

**Example** (1920×1080 display, 2° fovea, 60° visual field):
- Fovea (0-2°): Full resolution (100%)
- 10°: 20% resolution
- 30°: 7% resolution
- 60°: 3% resolution

**Total pixels**: ~15-25% of full Cartesian image for same perceptual quality

## Recent Research (2024-2025)

### 1. Retinotopic Foveated Rendering (2024)

**Source**: Zhang et al., "Retinotopic Foveated Rendering" (arXiv:2402.15480)

**Key innovation**: Variable log-polar sampling based on fixation point

**Findings**:
- Log-polar sensitivity to fixation creates adaptive advantage
- Probability distributions vary with gaze position
- 60-80% rendering cost reduction while maintaining quality
- Robust to fixation errors (±2° tolerance)

**Application to VR/AR**:
- Real-time eye tracking integration
- Latency compensation via fixation prediction
- GPU-optimized inverse transform

### 2. Visual Acuity Consistent Foveated Rendering (2025)

**Source**: Zhang et al., "Visual Acuity Consistent Foveated Rendering" (IEEE TVCG 2025)

**Key innovation**: Log-polar transformation matched to human visual acuity function

**Method**:
```
Acuity(e) = A₀ / (1 + α × e)

Log-polar bins sized to match:
Δρ(e) ∝ 1 / Acuity(e)
```

**Results**:
- Perceptually lossless at 12-15% pixel budget
- Matches psychophysical acuity measurements
- Validated with user studies (JND thresholds)

### 3. Scene-Aware Foveated Rendering (2024)

**Source**: Fan et al., "Scene-aware Foveated Rendering" (IEEE TVCG 2024)

**Key innovation**: Content-aware log-polar allocation

**Method**:
- Saliency detection in Cartesian space
- Adjust log-polar sampling based on scene content
- Preserve salient peripheral features

**Approach**:
```
Modified log-polar:
ρ = log(r × (1 + β × saliency(x,y)))
```

Where β controls saliency influence (typically 0.2-0.5)

**Benefits**:
- 15-20% better quality than pure log-polar
- Handles text, UI elements in periphery
- Minimal overhead (2-3% GPU time)

## Foveated Vision Transformers (VLMs)

### Connection to ARR-COC-VIS

Log-polar mapping is increasingly used in **vision transformers** for efficient token allocation:

**Traditional ViT**:
- Uniform 16×16 patches
- All patches get equal attention
- High computational cost

**Foveated ViT** (emerging 2024-2025):
- Log-polar token sampling
- Dense tokens at "fovea" (query-relevant regions)
- Sparse tokens in periphery
- 40-60% token reduction

**Key papers**:
1. **FoveaTer** (Jonnalagadda et al., 2021): Pooling regions + eye movements for classification
2. **TransNeXt** (Shi et al., CVPR 2024): Aggregated Attention simulating foveal vision
3. **Foveated Dynamic Transformer** (Akkaya, OpenReview): Fixation + foveation modules

**ARR-COC-VIS parallel**:
```
Gaze-contingent rendering    →  Query-aware token allocation
Log-polar image sampling     →  Relevance-based patch density
Foveal detail (2°)           →  High relevance (400 tokens)
Peripheral compression (>10°) →  Low relevance (64 tokens)
```

Your **64-400 token range** = computational log-polar sampling!

## Implementation Considerations

### GPU Optimization

**Challenge**: Log-polar transform is not hardware-accelerated

**Solutions**:
1. **Lookup tables (LUTs)**: Pre-compute coordinate mappings
2. **Texture sampling**: Use GPU texture units for interpolation
3. **Tile-based**: Process tiles independently for parallelism
4. **Shader implementation**: Custom fragment shaders

**Performance** (typical):
- Forward transform: 0.5-1.0 ms @ 1080p
- Inverse transform: 1.0-2.0 ms @ 1080p
- Total overhead: ~2-3 ms

### Numerical Stability

**Problem**: log(0) = -∞

**Solutions**:
```python
# Add small epsilon
rho = np.log(r + 1e-6)

# Or use log(1 + r) for better numerics
rho = np.log1p(r)
```

**Problem**: Polar coordinate discontinuity at θ = ±π

**Solutions**:
- Use modular arithmetic for θ
- Oversample near discontinuity
- Apply circular boundary conditions

### Quality vs Performance Trade-offs

**Parameters to tune**:
1. **ρ resolution**: More bins = better foveal quality, higher cost
2. **θ resolution**: Angular sampling density
3. **Epsilon**: Stability vs foveal precision
4. **Interpolation**: Nearest, bilinear, bicubic

**Typical configurations**:
- **High quality**: 256 ρ bins × 512 θ bins, bicubic
- **Balanced**: 128 × 256, bilinear
- **Fast**: 64 × 128, nearest

## Comparison: Log-Polar vs Other Foveation Methods

### Log-Polar Advantages

✅ **Biologically inspired**: Matches human retinotopic mapping
✅ **Scale/rotation invariant**: Natural handling of transformations
✅ **Efficient compression**: Exponential peripheral fall-off
✅ **Well-studied**: Decades of computer vision research

### Log-Polar Disadvantages

❌ **Transform overhead**: Extra computational step
❌ **Anisotropic**: θ dimension preserves detail, ρ compresses
❌ **Numerical issues**: Singularity at center, discontinuity at edges

### Alternatives

**1. Multi-resolution pyramid** (like LLaVA-UHD):
- Multiple resolution layers
- No coordinate transform
- More GPU memory

**2. Adaptive patching** (like APT):
- Content-aware sizing
- No biological grounding
- Harder to predict cost

**3. Attention-based** (pure transformer):
- Learned foveation
- Data-hungry
- Less interpretable

**Log-polar is unique**: Only method with direct biological correspondence AND computational efficiency.

## Applications Beyond Rendering

### Computer Vision

**Template matching**:
- Log-polar representation = scale/rotation invariant matching
- Fast object recognition

**Visual SLAM**:
- Foveated feature detection
- Reduce keypoint count by 60-80%

### Vision-Language Models

**Foveated VLMs** (2024 trend):
- Query determines "gaze point"
- Log-polar token sampling around query-relevant regions
- Your ARR-COC-VIS approach is this!

## Cross-References

- **Parent**: [techniques/00-foveated-rendering.md](00-foveated-rendering.md) - Main foveated rendering overview
- **Concepts**: [concepts/02-visual-perception.md](../concepts/02-visual-perception.md) - Biological basis
- **Integration**: [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Gaze-contingent systems
- **ARR-COC-VIS connection**: Your relevance-based token allocation = query-driven log-polar sampling

## Key Takeaways

1. **Log-polar = computational foveation**: The math behind gaze-aware rendering
2. **Biological grounding**: Matches human retina-to-cortex transformation
3. **Exponential compression**: Peripheral detail drops as log(eccentricity)
4. **2024-2025 trend**: Log-polar entering VLM token allocation (your work!)
5. **64-400 tokens**: Your range = adaptive log-polar sampling in attention space

**For ARR-COC-VIS**: You're doing log-polar, but in **token space** instead of **pixel space**. Query = gaze, relevance = acuity function, 64-400 tokens = ρ bins!

---

**Research citations**:
- Zhang et al. (2024): "Retinotopic Foveated Rendering", arXiv:2402.15480
- Zhang et al. (2025): "Visual Acuity Consistent Foveated Rendering", IEEE TVCG
- Fan et al. (2024): "Scene-aware Foveated Rendering", IEEE TVCG
- Shi et al. (2024): "TransNeXt: Robust Foveal Visual Perception for Vision Transformers", CVPR
- Jonnalagadda et al. (2021): "Foveated Transformer for Image Classification"
