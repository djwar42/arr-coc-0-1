# Part 14 Addendum: Vortex Experimental Implementations
*Executable code for vortex, spiral, and whirlwind token allocation experiments*

**Companion to**: [14-vortices-and-whirlwinds.md](14-vortices-and-whirlwinds.md)

---

## Overview: Three Core Experiments + Extensions

This addendum provides working implementations for:

**Core Experiments** (to be tested):
1. **Discrete Vortex Sampling** - Importance peaks → vortex centers → discrete token selection
2. **Continuous Soft Vortex Field** - Blended features via vortex strength weights
3. **Spiral Pattern Sampling** - Logarithmic spiral paths from vortex centers

**Extensions** (exploratory):
4. **Vortex Repulsion** - Keep vortices spatially distributed
5. **Adaptive Whirlwind** - Spiral tightness varies with density
6. **Voronoi Tessellation** - Organic cell boundaries
7. **Wavelet-Based Allocation** - Frequency domain importance

---

## Experiment 1: Discrete Vortex Sampling

**Core idea**: Find importance peaks, use as vortex centers, sample tokens within vortex radius

```python
import torch
import torch.nn as nn
import numpy as np

class DiscreteVortexSampler(nn.Module):
    """
    Find vortex centers from importance peaks, allocate tokens discretely.

    Advantages:
      - No learned parameters (data-driven vortex placement)
      - Continuous boundaries (no grid artifacts)
      - Natural coverage (vortices spread across peaks)

    Disadvantages:
      - Non-differentiable peak finding
      - Fixed number of vortices
    """

    def __init__(
        self,
        num_vortices=8,
        total_tokens=273,
        grid_size=64,  # Assume 64×64 patch grid
    ):
        super().__init__()
        self.num_vortices = num_vortices
        self.total_tokens = total_tokens
        self.grid_size = grid_size

    def find_importance_peaks(
        self,
        importance_scores,  # [N] flat importance
        positions,          # [N, 2] positions
    ):
        """
        Find local maxima in importance field.

        Method:
          1. Reshape to 2D grid
          2. Apply max pooling with small kernel
          3. Peaks = locations where value == max_pooled value
          4. Select top-K peaks by importance
        """
        N = len(importance_scores)

        # Reshape to grid (assume square grid)
        importance_grid = importance_scores.reshape(
            self.grid_size, self.grid_size
        )

        # Max pooling to find local maxima
        kernel_size = 5
        padding = kernel_size // 2

        max_pooled = torch.nn.functional.max_pool2d(
            importance_grid.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze()

        # Peaks: locations where original == max_pooled
        is_peak = (importance_grid == max_pooled).float()
        peak_importance = importance_grid * is_peak

        # Flatten and select top-K
        flat_peaks = peak_importance.flatten()
        top_k_indices = torch.topk(flat_peaks, k=self.num_vortices).indices

        # Convert to (x, y) positions
        y_coords = (top_k_indices // self.grid_size).float() / (self.grid_size - 1)
        x_coords = (top_k_indices % self.grid_size).float() / (self.grid_size - 1)

        vortex_positions = torch.stack([x_coords, y_coords], dim=1)  # [K, 2]

        return vortex_positions

    def allocate_tokens_to_vortices(self, vortex_positions, importance_scores):
        """
        Decide how many tokens each vortex gets based on local importance.

        Method: Compute importance within radius of each vortex.
        """
        K = len(vortex_positions)

        # For simplicity: equal allocation
        # (Could be proportional to vortex importance sum)
        tokens_per_vortex = torch.full((K,), self.total_tokens // K, dtype=torch.long)

        # Adjust for rounding
        remainder = self.total_tokens - tokens_per_vortex.sum()
        tokens_per_vortex[0] += remainder

        return tokens_per_vortex

    def sample_within_vortex(
        self,
        vortex_center,      # [2]
        positions,          # [N, 2]
        importance_scores,  # [N]
        num_tokens,
        radius=0.25,
    ):
        """
        Sample tokens within vortex radius by importance.
        """
        # Compute distances to vortex center
        distances = torch.norm(positions - vortex_center, dim=1)

        # Patches within radius
        in_radius = (distances < radius)

        # If not enough patches in radius, expand
        if in_radius.sum() < num_tokens:
            # Take all in radius + closest others
            in_radius_indices = torch.where(in_radius)[0]
            out_radius_indices = torch.where(~in_radius)[0]

            # Sort out-radius by distance
            out_distances = distances[out_radius_indices]
            sorted_out = out_radius_indices[torch.argsort(out_distances)]

            # Combine
            needed = num_tokens - len(in_radius_indices)
            selected = torch.cat([in_radius_indices, sorted_out[:needed]])
        else:
            # Select top-K by importance within radius
            in_radius_indices = torch.where(in_radius)[0]
            in_radius_importance = importance_scores[in_radius_indices]

            top_k = torch.topk(in_radius_importance, k=num_tokens).indices
            selected = in_radius_indices[top_k]

        return selected

    def forward(self, importance_scores, positions):
        """
        Main forward pass.

        Args:
          importance_scores: [N] importance per patch
          positions: [N, 2] (x, y) in [0, 1]

        Returns:
          selected_indices: [total_tokens]
        """
        # Step 1: Find vortex centers
        vortex_positions = self.find_importance_peaks(importance_scores, positions)

        # Step 2: Allocate tokens to vortices
        tokens_per_vortex = self.allocate_tokens_to_vortices(
            vortex_positions, importance_scores
        )

        # Step 3: Sample within each vortex
        selected_indices = []
        for k in range(self.num_vortices):
            vortex_center = vortex_positions[k]
            num_tokens = tokens_per_vortex[k].item()

            if num_tokens > 0:
                selected = self.sample_within_vortex(
                    vortex_center, positions, importance_scores, num_tokens
                )
                selected_indices.extend(selected.tolist())

        return torch.tensor(selected_indices[:self.total_tokens], dtype=torch.long)


# Usage example
sampler = DiscreteVortexSampler(num_vortices=8, total_tokens=273)

importance = torch.randn(4096)  # Mock importance scores
positions = torch.rand(4096, 2)  # Mock positions

selected_idx = sampler(importance, positions)
print(f"Selected {len(selected_idx)} tokens via vortex sampling")
```

---

## Experiment 2: Continuous Soft Vortex Field

**Core idea**: Blend patch features using vortex field weights (fully differentiable)

```python
class ContinuousSoftVortexSampler(nn.Module):
    """
    Create virtual tokens as weighted combinations of patches.

    Each vortex generates tokens at different "depths" along radial path.
    Fully differentiable - no discrete selection!

    Advantages:
      - Completely differentiable
      - No top-K gradients issues
      - Smooth allocation

    Disadvantages:
      - Higher compute (weighted sums)
      - Less interpretable (what IS a virtual token?)
    """

    def __init__(
        self,
        num_vortices=8,
        total_tokens=273,
        vortex_radius=0.3,
    ):
        super().__init__()
        self.num_vortices = num_vortices
        self.total_tokens = total_tokens
        self.vortex_radius = vortex_radius

    def compute_vortex_field(self, positions, vortex_centers):
        """
        Compute vortex field strength at each patch position.

        Returns:
          field_strength: [N, K] - strength of vortex k at patch i
        """
        # Distances: [N, K]
        distances = torch.cdist(positions, vortex_centers)

        # Gaussian decay
        # strength = exp(-distance² / (2 * radius²))
        field_strength = torch.exp(
            -distances**2 / (2 * self.vortex_radius**2)
        )

        # Normalize per vortex (each vortex has unit total strength)
        field_strength = field_strength / (field_strength.sum(dim=0, keepdim=True) + 1e-8)

        return field_strength  # [N, K]

    def create_virtual_tokens(
        self,
        patch_features,  # [N, D]
        vortex_field,    # [N, K]
    ):
        """
        Create virtual tokens as weighted combinations.

        For each vortex, create tokens at different "depths":
          - depth 0: center (high weight at vortex center)
          - depth max: periphery (distributed weight)
        """
        N, D = patch_features.shape
        K = vortex_field.shape[1]

        tokens_per_vortex = self.total_tokens // K

        virtual_tokens = []

        for k in range(K):
            vortex_strength = vortex_field[:, k]  # [N]

            for depth in range(tokens_per_vortex):
                # Radial weight: higher depth → broader distribution
                # depth 0 → concentrate at center
                # depth max → spread to periphery

                depth_normalized = depth / tokens_per_vortex  # [0, 1]

                # Exponential decay: focus at center for low depth
                radial_weight = torch.exp(-depth_normalized * 3)  # High for depth=0

                # Combined weight
                combined = vortex_strength * radial_weight

                # Normalize
                combined = combined / (combined.sum() + 1e-8)

                # Weighted sum of features
                virtual_token = (combined.unsqueeze(1) * patch_features).sum(dim=0)  # [D]

                virtual_tokens.append(virtual_token)

        # Stack and trim to exact count
        virtual_tokens = torch.stack(virtual_tokens[:self.total_tokens])  # [total_tokens, D]

        return virtual_tokens

    def forward(
        self,
        patch_features,  # [N, D]
        positions,       # [N, 2]
        vortex_centers,  # [K, 2] (could be from importance peaks or learned)
    ):
        """
        Main forward pass.

        Returns:
          virtual_tokens: [total_tokens, D] - blended features
        """
        # Compute vortex field
        vortex_field = self.compute_vortex_field(positions, vortex_centers)

        # Create virtual tokens
        virtual_tokens = self.create_virtual_tokens(patch_features, vortex_field)

        return virtual_tokens


# Usage example
sampler = ContinuousSoftVortexSampler(num_vortices=8, total_tokens=273)

patch_features = torch.randn(4096, 768)  # Mock features
positions = torch.rand(4096, 2)
vortex_centers = torch.rand(8, 2)  # Could be from importance peaks

virtual_tokens = sampler(patch_features, positions, vortex_centers)
print(f"Created {len(virtual_tokens)} virtual tokens via soft vortex field")
print(f"Shape: {virtual_tokens.shape}")  # [273, 768]
```

---

## Experiment 3: Spiral Pattern Sampling

**Core idea**: Sample tokens along logarithmic spiral path from vortex

```python
class SpiralPatternSampler(nn.Module):
    """
    Sample tokens along golden ratio logarithmic spiral.

    Spiral equation (polar):
      r(θ) = a * exp(b * θ)

    Where b = log(φ) / (π/2) for golden ratio spiral.

    Advantages:
      - Natural spatial coherence (tokens are spatially adjacent)
      - Better angular coverage than radial
      - Aesthetically pleasing (if that matters)

    Disadvantages:
      - More complex than top-K
      - Unclear if it helps accuracy
    """

    def __init__(
        self,
        total_tokens=273,
        growth_rate=None,  # If None, use golden ratio
    ):
        super().__init__()
        self.total_tokens = total_tokens

        # Golden ratio spiral
        if growth_rate is None:
            phi = (1 + np.sqrt(5)) / 2
            self.growth_rate = np.log(phi) / (np.pi / 2)
        else:
            self.growth_rate = growth_rate

    def compute_spiral_coordinates(
        self,
        positions,      # [N, 2]
        spiral_center,  # [2]
    ):
        """
        Convert patch positions to spiral coordinates.

        Returns:
          s: [N] - position along spiral path (0 at center, increases outward)
        """
        # Cartesian to polar relative to spiral center
        rel_x = positions[:, 0] - spiral_center[0]
        rel_y = positions[:, 1] - spiral_center[1]

        r = torch.sqrt(rel_x**2 + rel_y**2)
        theta = torch.atan2(rel_y, rel_x)

        # For logarithmic spiral: r = a * exp(b * θ)
        # Solving for θ: θ = log(r/a) / b
        # Position along spiral s = arclength ≈ r / exp(b*θ)

        # Spiral coordinate (how far along spiral)
        s = r / (torch.exp(self.growth_rate * theta) + 1e-6)

        return s, r, theta

    def sample_along_spiral(
        self,
        positions,
        importance_scores,
        spiral_center,
        num_tokens,
    ):
        """
        Select tokens along spiral path, prioritizing importance.
        """
        s, r, theta = self.compute_spiral_coordinates(positions, spiral_center)

        # Score: importance × spiral proximity
        # Patches close to ideal spiral path get higher scores

        # Ideal spiral: r_ideal(θ) = a * exp(b * θ)
        # For each patch, compute how far it is from ideal spiral
        a = 0.01  # Initial radius (small, near center)
        r_ideal = a * torch.exp(self.growth_rate * theta)

        spiral_deviation = torch.abs(r - r_ideal)

        # Score: high importance AND close to spiral
        spiral_score = importance_scores / (spiral_deviation + 0.01)

        # Select top-K by spiral score
        selected = torch.topk(spiral_score, k=num_tokens).indices

        return selected

    def forward(
        self,
        positions,
        importance_scores,
        spiral_center,  # [2] - could be highest importance location
    ):
        """
        Main forward pass.

        Args:
          positions: [N, 2]
          importance_scores: [N]
          spiral_center: [2] - center of spiral

        Returns:
          selected_indices: [total_tokens]
        """
        selected = self.sample_along_spiral(
            positions,
            importance_scores,
            spiral_center,
            self.total_tokens
        )

        return selected


# Usage example
sampler = SpiralPatternSampler(total_tokens=273)

positions = torch.rand(4096, 2)
importance = torch.randn(4096)
spiral_center = positions[torch.argmax(importance)]  # Center at peak importance

selected_idx = sampler(positions, importance, spiral_center)
print(f"Selected {len(selected_idx)} tokens via spiral sampling")
```

---

## Extension 1: Vortex Repulsion (Stability)

**Core idea**: Add loss term to prevent vortex collapse

```python
def vortex_repulsion_loss(vortex_positions, min_distance=0.15):
    """
    Penalize vortices that are too close together.

    Used during training if vortex positions are learned.

    Args:
      vortex_positions: [K, 2]
      min_distance: Minimum allowed distance between vortices

    Returns:
      repulsion_loss: scalar
    """
    K = len(vortex_positions)

    # Pairwise distances
    distances = torch.cdist(vortex_positions, vortex_positions)  # [K, K]

    # Mask diagonal (self-distances)
    mask = ~torch.eye(K, dtype=torch.bool, device=distances.device)
    distances = distances[mask].reshape(K, K-1)

    # Penalize distances below threshold
    violations = torch.clamp(min_distance - distances, min=0)  # [K, K-1]

    # Loss: sum of squared violations
    repulsion_loss = (violations ** 2).sum()

    return repulsion_loss


# Usage in training loop
class LearnedVortexSampler(nn.Module):
    """
    Vortex positions are learned parameters (not derived from data).
    """

    def __init__(self, num_vortices=8, total_tokens=273):
        super().__init__()
        self.vortex_positions = nn.Parameter(torch.rand(num_vortices, 2))
        self.total_tokens = total_tokens

    def forward(self, patch_features, positions, importance):
        # Sample tokens based on learned vortex positions
        # (Implementation similar to DiscreteVortexSampler)
        pass

    def compute_loss(self, task_loss):
        """
        Add repulsion loss to prevent collapse.
        """
        repulsion = vortex_repulsion_loss(self.vortex_positions)

        total_loss = task_loss + 0.1 * repulsion

        return total_loss


# Training example
model = LearnedVortexSampler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    task_loss = compute_task_loss(batch)  # e.g., LLM accuracy
    total_loss = model.compute_loss(task_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Vortex positions will spread out due to repulsion loss!
```

---

## Extension 2: Adaptive Whirlwind (Dynamic Spiral)

**Core idea**: Spiral tightness adapts to local importance density

```python
class AdaptiveWhirlwindSampler(nn.Module):
    """
    Spiral growth rate varies based on local importance density.

    High density → tight spiral (more tokens per revolution)
    Low density → loose spiral (broader coverage)
    """

    def __init__(self, total_tokens=273):
        super().__init__()
        self.total_tokens = total_tokens
        self.base_growth_rate = nn.Parameter(torch.tensor(0.3))
        self.density_sensitivity = nn.Parameter(torch.tensor(0.5))

    def compute_radial_density(self, positions, importance, center, num_bins=10):
        """
        Compute average importance as function of radius from center.

        Returns:
          densities: [num_bins] - average importance per radial bin
        """
        # Distances from center
        distances = torch.norm(positions - center, dim=1)

        # Create radial bins
        max_dist = distances.max()
        bin_edges = torch.linspace(0, max_dist, num_bins + 1)

        densities = []
        for i in range(num_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if mask.any():
                avg_importance = importance[mask].mean()
                densities.append(avg_importance.item())
            else:
                densities.append(0.0)

        return torch.tensor(densities, device=importance.device)

    def adaptive_growth_rate(self, radius, densities, bin_edges):
        """
        Growth rate varies with radius based on local density.

        b(r) = base_rate / (1 + sensitivity * density(r))

        High density → small b → tight spiral
        Low density → large b → loose spiral
        """
        # Find which bin this radius falls into
        bin_idx = torch.searchsorted(bin_edges, radius)
        bin_idx = torch.clamp(bin_idx, 0, len(densities) - 1)

        local_density = densities[bin_idx]

        adaptive_b = self.base_growth_rate / (
            1 + self.density_sensitivity * local_density
        )

        return adaptive_b

    def sample_adaptive_spiral(self, positions, importance, center, num_tokens):
        """
        Sample along spiral with adaptive growth rate.
        """
        # Compute radial density profile
        densities = self.compute_radial_density(positions, importance, center)

        # Convert to polar
        rel_x = positions[:, 0] - center[0]
        rel_y = positions[:, 1] - center[1]
        r = torch.sqrt(rel_x**2 + rel_y**2)
        theta = torch.atan2(rel_y, rel_x)

        # For each patch, compute adaptive growth rate at its radius
        # (Simplified: use base_growth_rate for now, full implementation would
        #  compute per-patch adaptive rate)

        # Score based on spiral proximity with adaptive rate
        # (Implementation similar to SpiralPatternSampler but with varying b)

        # For now, fall back to standard spiral
        # (Full adaptive implementation is complex)

        spiral_sampler = SpiralPatternSampler(num_tokens)
        selected = spiral_sampler.sample_along_spiral(
            positions, importance, center, num_tokens
        )

        return selected

    def forward(self, positions, importance, center):
        selected = self.sample_adaptive_spiral(
            positions, importance, center, self.total_tokens
        )
        return selected


# Note: Full adaptive whirlwind is complex. This is a simplified scaffold.
```

---

## Extension 3: Voronoi Tessellation

**Core idea**: Organic cell boundaries, each patch belongs to nearest vortex

```python
def voronoi_allocation(
    positions,       # [N, 2]
    vortex_centers,  # [K, 2]
    importance,      # [N]
    total_tokens=273,
):
    """
    Voronoi tessellation: each patch belongs to nearest vortex.

    Then allocate tokens proportionally to cell importance.

    Advantages:
      - Natural, organic boundaries
      - No overlap between regions
      - Differentiable if using soft assignment

    Disadvantages:
      - Still discrete assignment (nearest vortex)
      - Sensitive to vortex positions
    """
    N = len(positions)
    K = len(vortex_centers)

    # Compute distances to all vortices
    distances = torch.cdist(positions, vortex_centers)  # [N, K]

    # Hard assignment: nearest vortex
    nearest_vortex = torch.argmin(distances, dim=1)  # [N]

    # Compute importance per Voronoi cell
    cell_importance = torch.zeros(K)
    for k in range(K):
        mask = (nearest_vortex == k)
        if mask.any():
            cell_importance[k] = importance[mask].sum()

    # Allocate tokens proportionally
    importance_fraction = cell_importance / (cell_importance.sum() + 1e-8)
    tokens_per_cell = (importance_fraction * total_tokens).long()

    # Adjust for rounding
    while tokens_per_cell.sum() < total_tokens:
        tokens_per_cell[torch.argmax(cell_importance)] += 1
    while tokens_per_cell.sum() > total_tokens:
        tokens_per_cell[torch.argmax(tokens_per_cell)] -= 1

    # Select top-K within each cell
    selected_indices = []
    for k in range(K):
        mask = (nearest_vortex == k)
        cell_patches = torch.where(mask)[0]
        cell_scores = importance[mask]

        k_tokens = tokens_per_cell[k].item()
        if k_tokens > 0 and len(cell_patches) > 0:
            if len(cell_patches) >= k_tokens:
                top_k = torch.topk(cell_scores, k=k_tokens).indices
                selected = cell_patches[top_k]
            else:
                selected = cell_patches

            selected_indices.extend(selected.tolist())

    return torch.tensor(selected_indices[:total_tokens], dtype=torch.long)


# Usage
vortex_centers = torch.rand(8, 2)
positions = torch.rand(4096, 2)
importance = torch.randn(4096)

selected = voronoi_allocation(positions, vortex_centers, importance, 273)
print(f"Voronoi selected {len(selected)} tokens")
```

---

## Extension 4: Wavelet-Based Allocation

**Core idea**: Importance in frequency domain, allocate to high-coefficient wavelets

```python
import pywt

def wavelet_allocation(
    importance_scores,  # [N] flat
    positions,          # [N, 2]
    grid_size=64,
    total_tokens=273,
    wavelet='db4',
):
    """
    Transform importance field to wavelet domain,
    allocate tokens to regions with high wavelet coefficients.

    Wavelets decompose into scales (frequency bands):
      - Low frequency (cA): Overall structure
      - High frequency (cH, cV, cD): Details

    Allocate more tokens to detail-rich regions.

    Advantages:
      - Frequency-aware allocation
      - Multi-scale analysis
      - Classic signal processing approach

    Disadvantages:
      - Requires rectangular grid
      - Not query-aware (unless importance already is)
      - Complex implementation
    """
    # Reshape importance to 2D grid
    importance_grid = importance_scores.reshape(grid_size, grid_size).cpu().numpy()

    # 2D wavelet decomposition
    coeffs = pywt.dwt2(importance_grid, wavelet)
    cA, (cH, cV, cD) = coeffs

    # cA: low-freq (overall), cH/cV/cD: high-freq (details)

    # Compute magnitude of detail coefficients
    detail_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)

    # Allocate tokens proportional to detail magnitude
    # (More tokens where high-frequency content is strong)

    # Upsample detail_magnitude to original size
    from scipy.ndimage import zoom
    scale = grid_size / detail_magnitude.shape[0]
    detail_upsampled = zoom(detail_magnitude, scale, order=1)

    # Flatten
    detail_flat = torch.tensor(detail_upsampled.flatten(), dtype=torch.float32)

    # Select top-K by detail magnitude
    selected = torch.topk(detail_flat, k=total_tokens).indices

    return selected


# Usage (requires pywt)
# pip install PyWavelets
importance = torch.randn(4096)
positions = torch.rand(4096, 2)

selected = wavelet_allocation(importance, positions, grid_size=64, total_tokens=273)
print(f"Wavelet allocated {len(selected)} tokens")
```

---

## Evaluation Framework: Test All Methods

```python
class TokenAllocationBenchmark:
    """
    Test suite to compare all allocation methods on same dataset.

    Metrics:
      - Accuracy (DocVQA)
      - Speed (ms per image)
      - Coverage (% of image regions with ≥1 token)
      - Spatial coherence (avg distance between consecutive tokens)
    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def test_method(self, sampler, name):
        """
        Run sampler on dataset, measure metrics.
        """
        import time

        accuracies = []
        times = []
        coverages = []

        for image, query, ground_truth in self.dataset:
            # Compute importance
            importance = self.model.compute_importance(image, query)
            positions = self.model.get_positions(image)

            # Sample tokens
            start = time.time()
            selected_idx = sampler(importance, positions)
            elapsed = (time.time() - start) * 1000  # ms

            # Forward through LLM
            tokens = self.model.get_tokens(image)[selected_idx]
            prediction = self.model.llm_forward(tokens, query)

            # Metrics
            acc = self.compute_accuracy(prediction, ground_truth)
            cov = self.compute_coverage(selected_idx, positions)

            accuracies.append(acc)
            times.append(elapsed)
            coverages.append(cov)

        # Aggregate
        results = {
            'name': name,
            'accuracy': np.mean(accuracies),
            'time_ms': np.mean(times),
            'coverage': np.mean(coverages),
        }

        return results

    def compare_all(self):
        """
        Compare stratified, vortex discrete, vortex continuous, spiral.
        """
        methods = [
            (StratifiedTokenSampler(grid=(4,4)), "Stratified 4x4"),
            (DiscreteVortexSampler(num_vortices=8), "Vortex Discrete"),
            (ContinuousSoftVortexSampler(num_vortices=8), "Vortex Continuous"),
            (SpiralPatternSampler(), "Spiral"),
            # Add more methods here
        ]

        results = []
        for sampler, name in methods:
            print(f"Testing {name}...")
            result = self.test_method(sampler, name)
            results.append(result)

        # Print comparison table
        print("\n" + "="*70)
        print(f"{'Method':<25} | {'Accuracy':<10} | {'Time (ms)':<12} | {'Coverage':<10}")
        print("="*70)
        for r in results:
            print(f"{r['name']:<25} | {r['accuracy']:<10.2%} | {r['time_ms']:<12.1f} | {r['coverage']:<10.2%}")
        print("="*70)

        return results

    def compute_coverage(self, selected_indices, all_positions, grid=(8,8)):
        """
        Coverage = % of grid cells that have at least 1 selected token.
        """
        grid_h, grid_w = grid

        selected_positions = all_positions[selected_indices]

        # Assign to grid cells
        grid_x = (selected_positions[:, 0] * grid_w).long().clamp(0, grid_w-1)
        grid_y = (selected_positions[:, 1] * grid_h).long().clamp(0, grid_h-1)

        # Count unique cells
        occupied_cells = set()
        for x, y in zip(grid_x.tolist(), grid_y.tolist()):
            occupied_cells.add((x, y))

        coverage = len(occupied_cells) / (grid_h * grid_w)

        return coverage


# Usage
# benchmark = TokenAllocationBenchmark(docvqa_dataset, vlm_model)
# results = benchmark.compare_all()
```

---

## Visualization Tools

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_vortex_allocation(
    image,
    vortex_centers,
    selected_indices,
    all_positions,
    importance_scores,
):
    """
    Visualize vortex centers and selected tokens.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original image with vortex centers
    axes[0].imshow(image)
    axes[0].scatter(
        vortex_centers[:, 0] * image.shape[1],
        vortex_centers[:, 1] * image.shape[0],
        c='red', s=200, marker='*', edgecolors='white', linewidths=2,
        label='Vortex centers'
    )
    axes[0].set_title("Image with Vortex Centers")
    axes[0].legend()
    axes[0].axis('off')

    # Plot 2: Importance heatmap
    grid_size = int(np.sqrt(len(importance_scores)))
    importance_grid = importance_scores.reshape(grid_size, grid_size)

    axes[1].imshow(importance_grid, cmap='hot', interpolation='nearest')
    axes[1].set_title("Importance Field")
    axes[1].axis('off')

    # Plot 3: Selected tokens
    axes[2].imshow(image, alpha=0.3)

    selected_positions = all_positions[selected_indices]

    # Color by vortex assignment
    distances = torch.cdist(selected_positions, vortex_centers)
    vortex_assignment = torch.argmin(distances, dim=1)

    for k in range(len(vortex_centers)):
        mask = (vortex_assignment == k).numpy()
        if mask.any():
            axes[2].scatter(
                selected_positions[mask, 0].numpy() * image.shape[1],
                selected_positions[mask, 1].numpy() * image.shape[0],
                s=30, alpha=0.7, label=f'Vortex {k}'
            )

    axes[2].set_title(f"Selected Tokens (n={len(selected_indices)})")
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_spiral_path(spiral_center, positions, selected_indices, grid_size=64):
    """
    Visualize spiral sampling pattern.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all patches (faint)
    ax.scatter(positions[:, 0], positions[:, 1], c='lightgray', s=10, alpha=0.3)

    # Plot selected tokens (colored by selection order)
    selected_pos = positions[selected_indices]

    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
    ax.scatter(selected_pos[:, 0], selected_pos[:, 1], c=colors, s=50, edgecolors='black', linewidths=0.5)

    # Plot spiral center
    ax.scatter(spiral_center[0], spiral_center[1], c='red', s=300, marker='*', edgecolors='white', linewidths=2)

    # Draw lines connecting selected tokens (spiral path)
    for i in range(len(selected_pos) - 1):
        ax.plot(
            [selected_pos[i, 0], selected_pos[i+1, 0]],
            [selected_pos[i, 1], selected_pos[i+1, 1]],
            'k-', alpha=0.2, linewidth=0.5
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("Spiral Sampling Pattern (colored by selection order)")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## Summary: What to Test First

**Recommended testing order**:

1. **Baseline**: Stratified 4×4 (already implemented in Dialogue 13)

2. **Discrete Vortex**: Test if boundary-free allocation helps
   - Expectation: +0.5-1.0% accuracy, +10-15% time
   - Hypothesis: Helps on documents with formulas spanning regions

3. **Continuous Soft Vortex**: Test differentiability gains
   - Expectation: Similar accuracy to discrete, +20-30% time (weighted sums)
   - Hypothesis: Better for end-to-end training

4. **Spiral Sampling** (if vortices show promise): Test spatial coherence
   - Expectation: +0.2-0.5% over discrete vortex
   - Hypothesis: Helps on radially-structured images

5. **Extensions** (if core experiments work): Repulsion, whirlwinds, Voronoi, wavelets
   - Test individually based on failure modes discovered

**Decision tree**:
```
Run Discrete Vortex vs Stratified
  │
  ├─ If vortex wins by >1%
  │   → Invest in continuous + spiral variants
  │
  ├─ If vortex wins by 0.3-0.7%
  │   → Marginal gain, keep as option but don't prioritize
  │
  └─ If vortex loses or ties
      → Stick with stratified, explore other directions (wavelets?)
```

**All code in this addendum is EXPERIMENTAL** - test, measure, decide based on data!

∿◇∿ The vortices await testing ∿◇∿
