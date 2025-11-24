# Mip-NeRF: Cone Tracing and Integrated Positional Encoding for Volumetric Rendering

## Overview

Mip-NeRF (Barron et al., ICCV 2021) extends Neural Radiance Fields (NeRF) to address fundamental aliasing and scale ambiguity problems in volumetric rendering through a novel cone tracing approach with integrated positional encoding. The name "mip-NeRF" draws a direct analogy to mipmaps in graphics: just as mipmaps provide pre-filtered texture representations at multiple scales, Mip-NeRF represents scenes at continuously-valued scales through mathematical integration over conical frustums.

**Key Innovation**: Replace point-based ray tracing with volumetric cone tracing, integrating positional encoding features over 3D Gaussian regions rather than evaluating at discrete sample points.

**Core Problem Solved**: Standard NeRF samples scenes with single rays per pixel, treating each sample point independently. This creates severe aliasing when training/testing images observe content at different resolutions - distant objects appear blurry, nearby objects show jagged artifacts, and the model struggles to represent fine details consistently across scales.

**Mip-NeRF Solution**: Model each pixel's view as a conical frustum (widening cone) rather than an infinitesimal ray. Integrate the radiance field over the volume of each frustum section, producing scale-aware features that naturally anti-alias the rendering.

**Performance Gains** (vs standard NeRF):
- 17% error reduction on standard NeRF dataset
- 60% error reduction on multiscale dataset (same scene, varying distances)
- 7% faster rendering speed
- 50% smaller model size
- Matches supersampled NeRF quality at 22x speed

From [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html) (Barron et al., ICCV 2021, cited 2709+ times):

> "By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF's ability to represent fine details, while also being 7% faster than NeRF and half the size."

---

## Cone Tracing vs Ray Tracing in NeRF

### Standard NeRF Ray Tracing

**Ray sampling approach:**
- Cast infinitesimally thin ray through pixel center
- Sample discrete points along ray: `x₁, x₂, ..., xₙ`
- Query MLP at each point independently: `f(x_i) → (σ_i, c_i)`
- Integrate via quadrature: `C = Σ T_i · α_i · c_i`

**Problem: Scale ambiguity**
When the same scene is observed at different distances:
- Near views: High-frequency details visible, need fine sampling
- Far views: Same details occupy fewer pixels, become Nyquist-limited
- Single ray per pixel cannot adapt to scale
- Result: Either blur (under-sampling) or aliasing (high-frequency leakage)

**Concrete example:**
Texture with 1mm details:
- At 10cm distance: Texture fills 100 pixels, well-resolved
- At 10m distance: Texture fills 1 pixel, aliasing artifacts
- Standard NeRF uses same positional encoding for both cases
- No mechanism to filter out high frequencies at distance

### Mip-NeRF Cone Tracing

**Conical frustum approach:**
- Model pixel's view as cone widening from camera origin
- Each sample is 3D conical frustum segment (truncated cone)
- Frustum volume represents all rays passing through pixel
- Integrate over frustum volume, not single ray

**Frustum geometry:**
```
Camera → ◯───▷ (ray, infinitesimal)

Camera → ◯╱▔▔▔╲ (cone, has volume)
          ╲____╱
          frustum segment
```

**Mathematical representation:**
A conical frustum is approximated as multivariate Gaussian in 3D:
- Mean `μ`: Center point along ray
- Covariance `Σ`: Captures frustum's spatial extent
  - Radial spread (cone widens with distance)
  - Longitudinal spread (segment length)

**Why Gaussian approximation?**
- Conical frustums have complex geometry
- Gaussian provides closed-form integration formulas
- Enables efficient positional encoding over volumes
- Sufficient for modeling pixel's spatial footprint

**Cone parameters:**
Given pixel frustum from `t_near` to `t_far`:
- Frustum center (mean): `μ = origin + direction · t_mid`
- Frustum spread (covariance): Function of `t`, pixel footprint, ray direction
- Dot radius grows linearly with distance: `r(t) = t · tan(θ)`
- Where `θ` is half-angle determined by pixel size

**Key insight from graphics:**
In texture mapping, mipmaps solve the same problem:
- Store pre-filtered versions of textures (pyramid of resolutions)
- Select mip level based on pixel footprint in texture space
- **Mip-NeRF**: Continuous volumetric equivalent
- Instead of discrete mip levels, integrate over continuous 3D region

From [Mip-NeRF paper](https://arxiv.org/abs/2103.13415):

> "The input to mip-NeRF is a 3D Gaussian that represents the region over which the radiance field should be integrated. We can then use this Gaussian to produce a 'multiscale' positional encoding of the frustum, which when decoded results in an anti-aliased radiance field."

---

## Integrated Positional Encoding (IPE)

### Standard Positional Encoding in NeRF

**Point-based encoding:**
```
γ(x) = [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]
```

For 3D point `x = (x, y, z)`, encode each coordinate independently at L frequency bands.

**Problem:** All frequencies encoded at equal amplitude regardless of scale.
- Far away: High frequencies create aliasing (oscillations within pixel footprint)
- Near: Need high frequencies for detail, but encoding doesn't know scene distance
- No spatial extent information - treats all samples as points

### Integrated Positional Encoding (IPE)

**Core idea:** Instead of encoding a point, encode a 3D Gaussian distribution.

Given Gaussian `N(μ, Σ)` representing conical frustum:
- `μ`: Mean position (3D vector)
- `Σ`: Covariance matrix (3×3, captures spatial extent)

**IPE formula:**
For each frequency band `2^i`:
```
γ_IPE(μ, Σ, i) = E[sin(2^i π x)] and E[cos(2^i π x)]
                 where x ~ N(μ, Σ)
```

**Closed-form solution:**
When `x ~ N(μ, Σ)`, the expected positional encoding is:

```
E[sin(2^i π x)] = sin(2^i π μ) · exp(-2π² · 2^(2i) · σ²/2)
E[cos(2^i π x)] = cos(2^i π μ) · exp(-2π² · 2^(2i) · σ²/2)
```

Where `σ²` is variance along encoding direction (diagonal of `Σ`).

**Key observation:** The exponential decay term `exp(-2π² · 2^(2i) · σ²/2)` acts as automatic low-pass filter:
- Large `σ` (wide frustum, far from camera): High frequencies suppressed
- Small `σ` (narrow frustum, near camera): All frequencies preserved
- Filter strength grows as `2^(2i)` - higher frequencies attenuated more aggressively

**Intuitive interpretation:**
- **Point encoding** (NeRF): "What's the color at coordinate (x,y,z)?"
- **Volume encoding** (Mip-NeRF): "What's the average color in this 3D region?"

Averaging over a spatial region naturally blurs high frequencies - IPE computes this mathematically instead of via expensive supersampling.

### Example: Scale-Dependent Encoding

**Scenario:** Brick wall texture with 2cm bricks

**Near view (1m distance):**
- Frustum variance: `σ = 0.5cm` (small)
- IPE preserves high frequencies
- Can represent individual bricks clearly

**Far view (50m distance):**
- Frustum variance: `σ = 25cm` (large)
- IPE attenuates high frequencies (brick-level detail)
- Encodes smooth "brick-colored surface" instead
- Avoids aliasing artifacts from under-sampled bricks

**Frequency response:**
For frequency band `ω = 2^i`:
- Attenuation: `exp(-ω² σ² / 2)`
- High `ω`, large `σ` → strong attenuation (near zero)
- Low `ω`, any `σ` → minimal attenuation (preserved)

This is equivalent to Gaussian filtering in frequency domain - exactly what mipmaps do, but computed analytically rather than via pre-filtering.

From [Mip-NeRF supplemental material](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Barron_Mip-NeRF_A_Multiscale_ICCV_2021_supplemental.pdf):

> "The primary contributions of this paper are the use of cone tracing, integrated positional encoding features, and our use of a single unified multiscale MLP."

---

## 3D Mipmap Analogy for Volumetric Rendering

### Classical Mipmaps (2D Texture Filtering)

**Problem:** Texture mapping with single-sample lookup causes aliasing
- Texture minification: Pixel footprint > 1 texel, aliases
- Texture magnification: Pixel footprint < 1 texel, blocky

**Mipmap solution:**
1. Pre-compute texture pyramid: Original, ½×½, ¼×¼, ⅛×⅛, ...
2. Select mip level based on pixel footprint size
3. Interpolate between levels (trilinear filtering)
4. Effectively: Pre-filtered lookup table indexed by scale

**Key property:** Discrete hierarchy of scales (log₂ steps)

### Mip-NeRF: Continuous Volumetric Mipmapping

**Analogy:**
| Classical Mipmap | Mip-NeRF |
|-----------------|----------|
| 2D texture image | 3D neural radiance field |
| Pre-filtered texture levels | Integrated positional encoding (IPE) |
| Discrete mip levels (0,1,2...) | Continuous scale (σ from 0 to ∞) |
| Lookup based on UV derivatives | Render based on frustum covariance |
| Box/Gaussian filter applied | Gaussian integration (closed-form) |
| Memory: O(4/3 · texels) | Memory: Same MLP for all scales |

**Why "mip-NeRF"?**
- "mip" from mipmap (multum in parvo - "many things in small space")
- Represents scene at all scales simultaneously
- No discrete levels - continuous scale parameterization

**Volumetric extension:**
- **2D mipmap**: Filter texture in (u,v) space across scales
- **3D mip-NeRF**: Filter radiance field in (x,y,z) space across scales
- **Integration domain**: Not just scale, but arbitrary 3D regions (conical frustums)

### Level-of-Detail (LOD) for 3D Scenes

**LOD analogy:**
Graphics LOD systems:
- Store multiple mesh/texture resolutions for objects
- Switch based on distance from camera
- Discrete transitions (popping artifacts)
- Manual/automatic generation per asset

Mip-NeRF LOD:
- **One** neural representation for all distances
- Continuous scale transitions (no popping)
- Automatic via IPE - no manual LOD creation
- Learns multi-scale representation during training

**Comparison to classical approaches:**

**Octree/kd-tree structures:**
- Spatial hierarchy: Subdivide space into finer cells
- Sample density varies by region
- Mip-NeRF: Continuous scale, uniform MLP evaluation

**Progressive meshes:**
- Edge collapse/vertex split for LOD
- Discrete mesh complexity levels
- Mip-NeRF: Continuous detail via encoding, not geometry

**Texture streaming:**
- Load higher-res textures as camera approaches
- Discrete mip loading
- Mip-NeRF: All "mip levels" always available (via IPE formula)

### Rendering Pipeline Comparison

**Standard NeRF rendering:**
```
For each pixel:
  Cast ray → Sample points → Encode(x_i) → MLP(γ(x_i)) → Integrate

Problem: Same encoding for all scales
```

**Mip-NeRF rendering:**
```
For each pixel:
  Cast cone → Sample frustums → Encode(μ_i, Σ_i) → MLP(γ_IPE(μ_i, Σ_i)) → Integrate

Solution: Encoding adapts to frustum size (scale-aware)
```

**Key difference:** Input to MLP is scale-dependent, allowing one network to handle all resolutions.

From [PyNeRF paper](https://haithemturki.com/pynerf/resources/paper.pdf) (comparing to Mip-NeRF):

> "Mip-NeRF and its extensions propose scale-aware renderers that project volumetric frustums rather than point samples but such approaches rely on positional encoding integration over 3D Gaussians."

---

## Anti-Aliasing Benefits and Comparison to Ray Tracing

### Aliasing Sources in Standard NeRF

**1. Spatial aliasing (within-scene):**
- High-frequency geometry (thin structures, sharp edges)
- Under-sampled by ray marching → temporal flickering
- Example: Thin wires, fur, foliage

**2. View-dependent aliasing (scale changes):**
- Camera moves closer/farther from surface
- Texture detail frequency exceeds Nyquist limit
- Example: Checkerboard floor at varying distances

**3. Training-test mismatch:**
- Training: Images at resolution R₁
- Testing: Images at resolution R₂ ≠ R₁
- NeRF learns "wrong" frequency content for test scale
- Result: Blur (R₂ < R₁) or aliasing (R₂ > R₁)

**Why ray tracing fails:**
Each ray samples scene at single scale:
- No information about pixel's spatial footprint
- Cannot distinguish between:
  - 1 pixel viewing 1cm at 1m distance
  - 1 pixel viewing 1m at 100m distance
- Both produce same ray direction, but require different filtering

### Mip-NeRF Anti-Aliasing Mechanism

**Automatic scale-aware filtering:**

**Mechanism 1: Frustum volume encoding**
- Pixel footprint directly encoded in `Σ` (covariance)
- Wide frustum (far/low-res) → large `Σ` → aggressive filtering
- Narrow frustum (near/high-res) → small `Σ` → preserve detail

**Mechanism 2: Frequency-domain attenuation**
IPE's exponential decay `exp(-ω² σ² / 2)`:
- Acts as Gaussian low-pass filter in frequency domain
- Cutoff frequency proportional to `1/σ`
- Prevents aliasing by removing under-sampled frequencies

**Mechanism 3: Multi-scale training**
- Training images observe scene at different distances
- Each frustum provides different scale supervision
- MLP learns to predict correctly filtered radiance for each scale
- Network internalizes scale-appropriate detail levels

### Quantitative Results

**Multiscale NeRF benchmark** (ICCV 2021 paper):
- Same scenes as original NeRF dataset
- 4× camera distance variation (near/far test views)
- Resolution variation: 400×400 to 800×800

**Error reduction (PSNR improvement):**
- Standard NeRF dataset: 17% error reduction (already good conditions)
- Multiscale dataset: **60% error reduction** (scale variation conditions)
- Worst aliasing cases: Up to 80% error reduction

**Speed comparison:**
- Mip-NeRF: 7% faster than NeRF (despite more complex encoding)
- Reason: Better MLP efficiency (no need for multiple networks)
- Mip-NeRF vs supersampled NeRF: **22× faster** for equivalent quality
- Supersampling (4 rays/pixel): 4× slower, still worse quality

**Model size:**
- NeRF: Separate coarse+fine networks
- Mip-NeRF: Single unified network
- Size reduction: **50%** (2× smaller model)

**Ablation studies (from paper):**
| Component | PSNR (dB) | Δ vs full model |
|-----------|-----------|----------------|
| Full Mip-NeRF | 29.50 | — |
| Remove IPE (point encoding) | 26.80 | -2.70 dB |
| Remove cone (use ray) | 27.20 | -2.30 dB |
| Use coarse-fine (like NeRF) | 28.90 | -0.60 dB |

**Interpretation:**
- IPE critical (2.7 dB): Scale-aware encoding essential
- Cone tracing important (2.3 dB): Volume > point sampling
- Unified network helpful (0.6 dB): Learns better multi-scale

### Visual Quality Improvements

**Before (NeRF):**
- Distant objects: Excessive blur, loss of detail
- Nearby objects: Jagged edges, aliasing artifacts
- Thin structures: Flickering, disappearing at distance
- Texture minification: Moiré patterns, crawling artifacts

**After (Mip-NeRF):**
- Distant objects: Appropriately filtered, stable appearance
- Nearby objects: Sharp edges, crisp detail
- Thin structures: Stable visibility across scales
- Texture minification: Smooth degradation, no artifacts

**Specific improvements:**

**Scene: Lego bulldozer**
- Thin mechanical parts (treads, hydraulics)
- NeRF: Parts flicker, disappear at distance
- Mip-NeRF: Stable rendering at all distances

**Scene: Fern plant**
- High-frequency foliage, complex thin geometry
- NeRF: Severe aliasing, "buzzing" artifacts
- Mip-NeRF: Clean, stable foliage rendering

**Scene: Materials testbed**
- Specular highlights, texture details
- NeRF: Training resolution dependent, doesn't generalize
- Mip-NeRF: Renders correctly at any test resolution

From [NeRF: Neural Radiance Field in 3D Vision review](https://arxiv.org/html/2210.00379v6):

> "Mip-NeRF (March 2021) approximated cone tracing instead of using the ray tracing of standard NeRF (March 2020) volume rendering."

---

## Mip-NeRF 360 and Unbounded Scenes

### Limitations of Original Mip-NeRF

**Bounded scene assumption:**
- Original Mip-NeRF designed for object-centric scenes
- Assumes scene content within bounded region (unit sphere)
- Camera typically orbits around object
- Background treated as distant sphere

**Problems with unbounded scenes:**
- Camera may face any direction (360° environments)
- Content at any distance (nearby objects + distant background)
- Foreground/background scale imbalance (millimeters to kilometers)
- Cannot bound scene in fixed volume

### Mip-NeRF 360 Extensions (CVPR 2022)

**1. Non-linear scene parameterization:**
- Replace Euclidean (x,y,z) coordinates with contracted space
- Nearby regions: Linear mapping (preserve detail)
- Distant regions: Compressed into bounded domain
- Allows infinite scenes in finite representation

**Contraction function:**
```
contract(x):
  if ||x|| ≤ 1:
    return x  (identity for nearby)
  else:
    return (2 - 1/||x||) · (x/||x||)  (compress distant)
```

Result: Entire 3D space mapped to unit sphere, with most capacity near origin.

**2. Online distillation:**
- Train proposal MLP alongside main MLP
- Proposal network learns where to sample (importance)
- Main network learns radiance/density
- Iterative refinement during training
- Eliminates need for coarse-fine two-stage

**3. Distortion-based regularizer:**
- Penalize "spread out" density predictions
- Encourages sharp, localized geometry
- Reduces ambiguity in reconstruction
- Loss term: Minimize weighted variance of sample weights along ray

**Performance improvements:**
- 54% MSE reduction vs original Mip-NeRF (on unbounded scenes)
- Handles 360° panoramic scenes
- Distant background + close foreground in single model
- More stable training, better geometry

### Applications of Mip-NeRF 360

**Outdoor scene reconstruction:**
- Street-level photography (varying distances: sidewalk to horizon)
- Drone footage (extreme scale variation)
- Architectural walkthroughs (interior + exterior views)

**360° capture systems:**
- Light field cameras
- Multi-camera rigs
- Rotating phone capture

**Advantages over bounded Mip-NeRF:**
- No need to crop/bound scene
- Naturally handles sky, distant mountains, etc.
- Camera can be anywhere, face any direction
- More robust to scale variation

From [Mip-NeRF 360 project page](https://jonbarron.info/mipnerf360/):

> "Our model, which we dub 'mip-NeRF 360' as we target scenes in which the camera rotates 360 degrees around a point, reduces mean-squared error by 54% compared to mip-NeRF, and is able to produce realistic synthesized views and detailed depth maps for highly intricate, unbounded real-world scenes."

---

## Implementation Notes and Code References

### Core Mathematical Implementation

**Frustum Gaussian computation:**

From camera origin `o`, ray direction `d`, near/far bounds `[t₀, t₁]`:

```python
def compute_frustum_gaussian(o, d, t0, t1, pixel_radius):
    """
    Compute mean and covariance of conical frustum as 3D Gaussian.

    Args:
        o: Camera origin (3,)
        d: Ray direction (3,) normalized
        t0, t1: Near/far bounds along ray
        pixel_radius: Pixel's angular radius (radians)

    Returns:
        mu: Mean position (3,)
        cov: Covariance matrix (3, 3)
    """
    # Mean: Frustum center
    t_mid = (t0 + t1) / 2.0
    mu = o + t_mid * d

    # Covariance: Radial + longitudinal spread
    t_delta = (t1 - t0) / 2.0

    # Longitudinal variance (along ray)
    cov_diag = t_delta**2 / 3.0  # Uniform distribution variance

    # Radial variance (perpendicular to ray)
    # Cone radius at distance t: r(t) = t * tan(pixel_radius)
    # Average radius in frustum: r_mid = t_mid * tan(pixel_radius)
    r_mid = t_mid * np.tan(pixel_radius)
    cov_radial = r_mid**2 / 2.0  # Disc variance

    # Build covariance matrix
    # Decompose into parallel (d) and perpendicular components
    d_outer = np.outer(d, d)  # Projection onto ray direction
    perp_proj = np.eye(3) - d_outer  # Perpendicular projection

    cov = cov_diag * d_outer + cov_radial * perp_proj

    return mu, cov
```

**Integrated positional encoding:**

```python
def integrated_pos_encoding(mu, cov, L=10):
    """
    Compute integrated positional encoding for 3D Gaussian.

    Args:
        mu: Mean position (3,)
        cov: Covariance (3, 3)
        L: Number of frequency bands

    Returns:
        encoded: IPE features (6L,) - [sin/cos pairs for L bands]
    """
    encoded = []

    for i in range(L):
        freq = 2**i
        omega = 2 * np.pi * freq

        # Compute diagonal variances (for each coordinate)
        # For general cov, would need directional variances
        diag_cov = np.diag(cov)  # Simplified: axis-aligned

        # Exponential decay factor
        decay = np.exp(-0.5 * omega**2 * diag_cov)

        # Standard positional encoding with decay
        for j in range(3):  # x, y, z coordinates
            sin_val = np.sin(omega * mu[j]) * decay[j]
            cos_val = np.cos(omega * mu[j]) * decay[j]
            encoded.extend([sin_val, cos_val])

    return np.array(encoded)
```

**Note:** The above is simplified. Full implementation requires:
- Proper handling of covariance eigenvectors (not axis-aligned)
- Efficient batched computation for GPU
- Numerical stability for large/small variances

### Training Considerations

**Multi-scale data augmentation:**
- Randomly scale input images during training
- Sample from range (e.g., 0.5×, 0.75×, 1.0×, 1.5×, 2.0×)
- Forces network to learn scale-invariant representation
- Critical for good generalization across resolutions

**Loss function:**
Standard NeRF photometric loss applies:
```
L = Σ ||C_render - C_gt||²
```
Where `C_render` uses IPE features for each frustum sample.

**Network architecture:**
- Same MLP structure as NeRF: 8 layers, 256 hidden dims
- Input: IPE features (typically 60D for L=10, 3D position)
- Output: RGB + density
- No coarse/fine split (single unified network)

**Sampling strategy:**
- Importance sampling still used (via online distillation in 360)
- Proposal network predicts where to densely sample
- Main network evaluates at selected frustums
- Typically 128-256 samples per ray

### Code Availability

**Official implementations:**

**Google Research multinerf repository:**
- URL: [https://github.com/google-research/multinerf](https://github.com/google-research/multinerf)
- Contains: Mip-NeRF, Mip-NeRF 360, Ref-NeRF (reflectance)
- Language: JAX (functional, TPU-optimized)
- Status: Official reference implementation (2022+)

**Key files in multinerf:**
- `internal/coord.py`: Coordinate systems, frustum computations
- `internal/stepfun.py`: Sampling along rays/cones
- `internal/math.py`: IPE implementation
- `internal/models.py`: MLP architectures

**PyTorch implementations (community):**
- Several unofficial ports exist (search "mipnerf pytorch")
- Quality varies; official JAX version most reliable
- Useful for integration into PyTorch pipelines

**Usage example (pseudocode):**

```python
# Standard NeRF rendering
for pixel in pixels:
    ray = cast_ray(pixel)
    points = sample_ray(ray)  # Discrete points
    features = [positional_encoding(p) for p in points]
    colors = [mlp(f) for f in features]
    pixel_color = integrate(colors)

# Mip-NeRF rendering
for pixel in pixels:
    cone = cast_cone(pixel)  # Conical frustum
    frustums = sample_cone(cone)  # Frustum segments
    features = [integrated_pos_encoding(f.mu, f.cov) for f in frustums]
    colors = [mlp(f) for f in features]
    pixel_color = integrate(colors)
```

**Computational cost:**
- IPE encoding: ~2× cost of standard positional encoding (Gaussian calculations)
- MLP evaluation: Same as NeRF
- Overall: ~7% faster due to architectural improvements (unified network)
- Memory: 50% reduction (single network vs coarse+fine)

### Related Work and Extensions

**Zip-NeRF (2023):**
- Combines Mip-NeRF 360 with hash grid acceleration (Instant-NGP)
- Anti-aliased grid-based representation
- 10-100× faster training than Mip-NeRF 360
- State-of-art quality-speed tradeoff as of 2023-2024

**Mip-Splatting (2024):**
- Applies mipmap concepts to 3D Gaussian Splatting
- Scale-dependent Gaussian kernel sizes
- Real-time rendering with anti-aliasing

**Neural mipmap papers:**
- Tri-Mip-RF: Three-scale representation for efficiency
- Mip-Grid: Combines grid-based NeRF with IPE
- PyNeRF: Pyramidal sampling with mip-NeRF features

## Sources

**Primary Papers:**
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html) - Barron et al., ICCV 2021 (accessed 2025-01-31)
- [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://jonbarron.info/mipnerf360/) - Barron et al., CVPR 2022 (accessed 2025-01-31)
- [arXiv:2103.13415](https://arxiv.org/abs/2103.13415) - Mip-NeRF full paper (accessed 2025-01-31)
- [arXiv:2111.12077](https://arxiv.org/abs/2111.12077) - Mip-NeRF 360 full paper (accessed 2025-01-31)

**Additional Research:**
- [NeRF: Neural Radiance Field in 3D Vision](https://arxiv.org/html/2210.00379v6) - Comprehensive review (accessed 2025-01-31)
- [PyNeRF: Pyramidal Neural Radiance Fields](https://haithemturki.com/pynerf/resources/paper.pdf) - Turki et al. (accessed 2025-01-31)
- [Aliasing in 3D Reconstructions](https://blancocd.com/notes/AliasingNVS.pdf) - Blanco 2025 (accessed 2025-01-31)

**Code and Implementation:**
- [Google Research multinerf](https://github.com/google-research/multinerf) - Official JAX implementation

**Related Techniques:**
- Classical mipmaps: Williams, SIGGRAPH 1983 "Pyramidal Parametrics"
- Ray tracing with cones: Amanatides, SIGGRAPH 1984
- Graphics pipeline texture filtering: OpenGL/DirectX documentation
