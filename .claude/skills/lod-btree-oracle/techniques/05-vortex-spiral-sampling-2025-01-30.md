# Vortex and Spiral Sampling for LOD Allocation

**Dynamic knowledge addition**: 2025-01-30
**Source**: ARR-COC-VIS Dialogue 14, computational geometry, biological vision
**Parent**: [00-foveated-rendering.md](00-foveated-rendering.md), [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md)

---

## Overview

Vortex and spiral sampling strategies provide **exploratory, biologically-inspired** approaches to LOD allocation. Unlike grid sampling (uniform, rigid) or semantic atlases (object-based), vortex sampling creates **importance centers** with dense sampling nearby and sparse sampling far away, following logarithmic spiral patterns reminiscent of galaxy formation and biological growth.

**Key Innovation**: Continuous sampling patterns that respect importance gradients while maintaining spatial coherence.

---

## Discrete Vortex Sampling

### Basic Concept

**Problem**: Grid sampling fragments objects, semantic atlases are expensive
**Solution**: Sample densely around importance centers, sparsely elsewhere

```python
def discrete_vortex_sampling(image, importance_map, num_centers=5, radius=50):
    """Sample patches around identified importance centers"""

    # Step 1: Find importance centers (local maxima)
    centers = find_local_maxima(importance_map, num_peaks=num_centers)

    # Step 2: For each center, sample within radius
    sampled_patches = []

    for center in centers:
        # Sample density decreases with distance from center
        for distance in range(0, radius, step=5):
            # Angular sampling (more points at larger radius)
            num_angles = max(4, int(2 * np.pi * distance / 10))

            for angle in np.linspace(0, 2*np.pi, num_angles, endpoint=False):
                # Polar to Cartesian
                x = center.x + distance * np.cos(angle)
                y = center.y + distance * np.sin(angle)

                if in_bounds(x, y, image.shape):
                    patch = extract_patch(image, (x, y))
                    sampled_patches.append(patch)

    return sampled_patches
```

**Characteristics**:
- Centers at high-importance regions
- Radial sampling (concentric circles)
- Density ∝ 1/distance (more samples near center)
- Spatial coherence (neighbors are nearby in importance space)

**Use Cases**:
- Exploratory search (don't know where important regions are)
- Multi-object scenes (multiple vortex centers)
- Texture-rich images (spiral captures patterns)

---

## Logarithmic Spiral Sampling

### Mathematical Foundation

**Logarithmic (Equiangular) Spiral**:
```
r(θ) = a * e^(b*θ)

Where:
  r = radius from center
  θ = angle (0 to 2π)
  a = initial radius
  b = growth rate (controls tightness)
```

**Special case - Golden Spiral** (b = ln(φ)/90°):
```python
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618

def golden_spiral_sampling(center, num_points=273):
    """Sample along golden spiral - optimal packing"""
    points = []

    # Golden angle ≈ 137.5°
    golden_angle = 2 * np.pi * (1 - 1/PHI)

    for i in range(num_points):
        # Angle increases by golden angle each step
        theta = i * golden_angle

        # Radius grows logarithmically
        r = a * np.exp(b * theta)

        # Convert to Cartesian
        x = center.x + r * np.cos(theta)
        y = center.y + r * np.sin(theta)

        points.append((x, y))

    return points
```

**Properties**:
- Self-similar at all scales (fractal-like)
- Optimal packing (minimal overlap, maximum coverage)
- Appears in nature: shells, galaxies, sunflower seeds
- Smooth exploration (no discontinuous jumps)

**Biological Grounding**:
- Nautilus shell: Golden spiral growth
- Sunflower seeds: Fibonacci spiral (related to golden ratio)
- Galaxy arms: Logarithmic spiral structure
- Human eye saccades: Spiral-like exploration patterns

---

## Continuous Soft Vortex Fields

### Problem with Discrete Vortices

**Discrete vortex issues**:
- Hard boundaries between vortex regions
- Difficult to optimize (non-differentiable)
- Sensitive to center placement

**Solution**: Continuous soft vortex field

```python
def compute_soft_vortex_field(image, importance_map):
    """Create differentiable vortex influence field"""

    # Compute importance gradient (direction of increasing importance)
    grad_x, grad_y = np.gradient(importance_map)

    # Vortex field: spiral around high-importance regions
    vortex_field = np.zeros_like(importance_map)

    for x in range(image.width):
        for y in range(image.height):
            # Distance to nearest high-importance region
            dist_to_peak = distance_to_nearest_peak(x, y, importance_map)

            # Vortex strength decreases with distance
            strength = np.exp(-dist_to_peak / sigma)

            # Spiral direction (perpendicular to gradient)
            spiral_x = -grad_y[x, y]
            spiral_y = grad_x[x, y]

            # Normalize
            magnitude = np.sqrt(spiral_x**2 + spiral_y**2) + 1e-8
            spiral_x /= magnitude
            spiral_y /= magnitude

            # Weighted vortex vector
            vortex_field[x, y] = strength * np.array([spiral_x, spiral_y])

    return vortex_field
```

**Advantages**:
- Differentiable (can backprop through it!)
- Smooth transitions between regions
- Adaptive to importance landscape
- Can be learned end-to-end

### Differentiable Sampling

```python
class DifferentiableSpiralSampler(nn.Module):
    """Learn optimal spiral sampling parameters"""

    def __init__(self):
        # Learnable parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.1))  # b in r = ae^(bθ)
        self.initial_radius = nn.Parameter(torch.tensor(5.0))  # a
        self.angle_step = nn.Parameter(torch.tensor(137.5 * np.pi/180))  # Golden angle

    def forward(self, image, center, num_points=273):
        """Sample along learned spiral"""
        points = []

        for i in range(num_points):
            theta = i * self.angle_step
            r = self.initial_radius * torch.exp(self.growth_rate * theta)

            x = center[0] + r * torch.cos(theta)
            y = center[1] + r * torch.sin(theta)

            # Differentiable bilinear sampling
            sampled_feature = grid_sample(image, (x, y))
            points.append(sampled_feature)

        return torch.stack(points)
```

**Training**: End-to-end with task loss, parameters adjust to optimal spiral shape

---

## Adaptive Whirlwind Sampling

### Density-Varying Spirals

**Concept**: Adjust spiral tightness based on local importance

```python
def adaptive_whirlwind(image, importance_map, budget=273):
    """Spiral sampling with adaptive density"""

    # Find center (highest importance)
    center = argmax(importance_map)

    samples = []
    theta = 0
    radius = initial_radius

    while len(samples) < budget:
        # Current position
        x = center.x + radius * np.cos(theta)
        y = center.y + radius * np.sin(theta)

        if in_bounds(x, y, image.shape):
            # Sample patch
            samples.append(extract_patch(image, (x, y)))

            # Adaptive angle step (smaller step = tighter spiral)
            local_importance = importance_map[int(x), int(y)]

            # High importance → tight spiral (small angle step)
            # Low importance → loose spiral (large angle step)
            angle_step = base_angle * (1 - local_importance)

            theta += angle_step

            # Radius grows logarithmically
            radius = initial_radius * np.exp(growth_rate * theta)
        else:
            # Hit boundary, stop
            break

    return samples
```

**Behavior**:
- Important regions: Tight spiral (many samples)
- Unimportant regions: Loose spiral (few samples)
- Automatic budget allocation
- Single continuous path (good for sequential processing)

---

## Multi-Center Vortex Systems

### Multiple Importance Peaks

**Problem**: Single vortex misses multiple objects
**Solution**: Multiple vortices with interference

```python
def multi_vortex_sampling(image, importance_map, num_vortices=5):
    """Multiple spirals from different centers"""

    # Find multiple importance peaks
    centers = find_local_maxima(importance_map, num_peaks=num_vortices)

    # Score each center by importance
    center_scores = [importance_map[c.x, c.y] for c in centers]
    total_score = sum(center_scores)

    # Allocate budget proportionally
    budget_per_center = [
        int(273 * score / total_score)
        for score in center_scores
    ]

    # Sample from each vortex
    all_samples = []
    for center, budget in zip(centers, budget_per_center):
        spiral_samples = spiral_sampling(image, center, num_points=budget)
        all_samples.extend(spiral_samples)

    return all_samples
```

**Budget Allocation**:
- Vortex importance ∝ peak height
- More samples for higher-importance vortices
- Total always sums to 273

### Vortex Interference

```python
def compute_vortex_interference(centers, strengths):
    """Model interaction between multiple vortices"""

    def vortex_field_at_point(x, y):
        total_field = np.zeros(2)

        for center, strength in zip(centers, strengths):
            # Vector from point to center
            dx = center.x - x
            dy = center.y - y
            distance = np.sqrt(dx**2 + dy**2) + 1e-8

            # Vortex induces spiral flow
            # Perpendicular to radial direction
            spiral_x = -dy / distance
            spiral_y = dx / distance

            # Strength decreases with distance
            field_strength = strength * np.exp(-distance / sigma)

            total_field += field_strength * np.array([spiral_x, spiral_y])

        return total_field

    return vortex_field_at_point
```

**Emergent Behavior**:
- Vortices attract samples (like gravity)
- Multiple vortices create complex flow patterns
- Samples follow natural paths in combined field
- Similar to fluid dynamics!

---

## Comparison to Other Strategies

| Method | Coverage | Coherence | Adaptiveness | Complexity |
|--------|----------|-----------|--------------|------------|
| **Grid** | Uniform | Low (fragments) | Static | O(N) |
| **Top-K Grid** | Sparse | Low | Query-aware | O(N log N) |
| **Semantic Atlas** | Semantic | High (whole objects) | Query-aware | O(N²) (SAM) |
| **Vortex** | Gradient | Medium | Importance | O(N) |
| **Spiral** | Exploratory | High (continuous) | Adaptive | O(N) |
| **Multi-Vortex** | Multi-region | High | Importance | O(KN) |

**Vortex Advantages**:
- Exploratory (good for unknown scenes)
- Spatially coherent (smooth paths)
- Biologically inspired
- Differentiable (can learn)

**Vortex Disadvantages**:
- Doesn't respect object boundaries (unlike atlas)
- Center placement critical
- May miss small distant objects

---

## Hybrid Approaches

### Grid-Vortex Hybrid

```python
def grid_vortex_hybrid(image, importance_map, query):
    """Combine grid coverage with vortex exploration"""

    # Stage 1: Coarse grid for broad coverage (40% budget)
    grid_samples = uniform_grid_sample(image, num_patches=109)

    # Stage 2: Identify high-importance regions
    grid_importance = [score(patch, query) for patch in grid_samples]
    top_regions = top_k(grid_samples, grid_importance, k=3)

    # Stage 3: Vortex sampling around top regions (60% budget)
    vortex_samples = []
    for region in top_regions:
        spiral = spiral_sampling(image, region.center, num_points=55)
        vortex_samples.extend(spiral)

    return grid_samples + vortex_samples  # 109 + 164 = 273
```

**Benefits**:
- Grid ensures coverage (no blind spots)
- Vortex provides detail where needed
- Balanced exploration-exploitation

### Atlas-Spiral Hybrid

```python
def atlas_spiral_hybrid(image, importance_map):
    """Semantic regions + spiral within regions"""

    # Stage 1: SAM segmentation
    regions = sam.generate(image)

    # Stage 2: Score regions
    region_scores = [score_region(r, importance_map) for r in regions]

    # Stage 3: Select top regions
    top_regions = top_k(regions, region_scores, k=20)

    # Stage 4: Spiral sample within each region
    samples = []
    for region in top_regions:
        # Spiral from region center, constrained to region mask
        center = region.centroid
        spiral_points = spiral_sampling(
            image,
            center,
            num_points=13,  # 20 regions × 13 ≈ 260 tokens
            mask=region.mask  # Only sample inside region
        )
        samples.extend(spiral_points)

    return samples
```

**Benefits**:
- Semantic boundaries respected (atlas)
- Smooth exploration within objects (spiral)
- Best of both worlds

---

## Biological Inspiration

### Human Eye Movements

**Saccadic Patterns**:
- Not random, not grid-like
- Spiral-like exploration of scenes
- More fixations near salient regions
- Logarithmic spacing (more at start, sparse later)

**Foveated Vision + Saccades**:
```
Fixation 1: Center of image (initial landing point)
  → High detail at center, low periphery

Saccade to next salient region (spiral-like path)

Fixation 2: Salient object
  → High detail at object, medium context

Saccade to next region...
```

**Vortex sampling mimics this**:
- Centers = fixation points
- Spiral = saccade path
- Density gradient = foveated resolution

### Natural Patterns

**Fibonacci Spirals in Nature**:
- Sunflower seeds: 21 spirals clockwise, 34 counterclockwise (Fibonacci numbers)
- Nautilus shell: Golden ratio spiral
- Galaxy arms: Logarithmic spiral structure
- Pine cones: Dual spiral pattern

**Why spirals?**:
- Optimal packing (maximum seeds in sunflower head)
- Efficient growth (each new element finds space)
- Structural stability
- **Efficient coverage of space!**

**Implication for VLMs**: Nature found spirals optimal for resource allocation → good prior for token allocation!

---

## Implementation Patterns

### Fast Spiral Generation

```python
def fast_spiral_points(num_points=273, growth=0.1):
    """Generate spiral points efficiently"""
    # Pre-allocate
    points = np.zeros((num_points, 2))

    # Golden angle
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi)

    # Vectorized generation
    indices = np.arange(num_points)
    thetas = indices * golden_angle
    radii = np.exp(growth * thetas)

    points[:, 0] = radii * np.cos(thetas)
    points[:, 1] = radii * np.sin(thetas)

    return points
```

### GPU-Accelerated Vortex Field

```python
import torch

def gpu_vortex_field(importance_map, device='cuda'):
    """Compute vortex field on GPU"""
    importance = torch.from_numpy(importance_map).to(device)

    # Gradient (direction to peaks)
    grad_x = torch.gradient(importance, dim=0)[0]
    grad_y = torch.gradient(importance, dim=1)[0]

    # Perpendicular (spiral direction)
    spiral_x = -grad_y
    spiral_y = grad_x

    # Normalize
    magnitude = torch.sqrt(spiral_x**2 + spiral_y**2) + 1e-8
    spiral_x /= magnitude
    spiral_y /= magnitude

    # Stack into vector field
    vortex_field = torch.stack([spiral_x, spiral_y], dim=-1)

    return vortex_field
```

---

## Open Research Questions

1. **Optimal growth rate**: Is golden ratio optimal for VLMs, or task-specific?
2. **Number of vortices**: How many centers needed for coverage?
3. **Vortex vs atlas**: When does spiral beat semantic segmentation?
4. **Learning spiral parameters**: Can we learn growth rate end-to-end?
5. **Multi-scale spirals**: Should we have spirals at multiple scales?

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [00-foveated-rendering.md](00-foveated-rendering.md) - Foveation fundamentals
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - LOD calculation
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Importance scoring
- [integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md) - Explore dimension

**Other Oracles**:
- **john-vervaeke-oracle**: Exploit-explore tension (vortex = explore!)
- **computer-vision-foundation-oracle**: SAM for region detection
- **vision-image-patching-oracle**: Adaptive patching strategies

---

**Last Updated**: 2025-01-30
**Status**: Synthesis from ARR-COC-VIS Dialogue 14, biological vision, computational geometry
**Relevance**: ★★★★☆ (Exploratory alternative to grid/atlas)
