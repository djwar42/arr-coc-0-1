# Neural Network Manifolds: Where Geometry Meets Deep Learning

## Overview

Neural networks don't just learn functions - they learn to transform data manifolds. This file explores how deep networks manipulate the geometric structure of data, how to measure intrinsic dimensionality, and why understanding manifolds is crucial for building better models.

**The Core Insight**: Every layer of a neural network performs a continuous transformation on the data manifold. Understanding these transformations reveals why networks succeed or fail, and how to design better architectures.

From [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) (Colah's blog, accessed 2025-11-23):
- Neural network layers are homeomorphisms (topology-preserving transformations)
- Classification requires untangling manifolds in representation space
- Topological constraints determine minimum network width

From [Intrinsic dimension of data representations in deep neural networks](https://arxiv.org/abs/1905.12784) (Ansuini et al., 2019, cited by 401):
- Intrinsic dimension (ID) measures the minimal parameters needed to describe a representation
- ID is typically much smaller than layer width
- ID follows a characteristic "hunchback" pattern through network layers

---

## Section 1: The Data Manifold Hypothesis

### What Is the Manifold Hypothesis?

The manifold hypothesis states that **natural high-dimensional data lies on or near a low-dimensional manifold** embedded in the ambient space.

From [Wikipedia - Manifold hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis) (accessed 2025-11-23):
- High-dimensional observations (images, audio, text) actually have far fewer degrees of freedom
- The "true" data lies on a curved surface in high-dimensional space
- Most of the ambient space contains no valid data points

**Why This Matters for Deep Learning:**

```
Image Space: 256 x 256 x 3 = 196,608 dimensions
But: Natural images form a manifold of ~100-1000 intrinsic dimensions

Most points in 196,608-D space are NOT valid images!
```

### Mathematical Formulation

A d-dimensional manifold M embedded in n-dimensional space (d << n):

```
M subset of R^n
dim(M) = d

For natural images:
- n = number of pixels * channels (very large)
- d = intrinsic degrees of freedom (much smaller)
```

From [A Manifold Learning Perspective on Representation Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8625121/) (Schuster, 2021, cited by 9):
- Manifold learning finds the low-dimensional structure
- Representation learning creates an encoder mapping to it
- Deep learning combines both with hierarchical feature extraction

### Evidence for the Manifold Hypothesis

**Theoretical Evidence:**
- Natural transformations (rotation, translation, lighting) form continuous paths
- These paths define smooth manifolds in pixel space
- Physical constraints limit degrees of freedom

**Experimental Evidence:**

From [The training process of many deep networks explores...](https://www.pnas.org/doi/10.1073/pnas.2310002121) (Mao et al., 2024, cited by 27):
- Training explores a low-dimensional manifold in prediction space
- Networks with different architectures converge to similar manifold regions
- The manifold structure is determined by the data, not the architecture

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ManifoldAnalyzer:
    """
    Analyze the manifold structure of data and representations.

    The manifold hypothesis suggests that high-dimensional data
    lies on a low-dimensional manifold. This class provides tools
    to verify and measure this structure.
    """

    def __init__(self, k_neighbors: int = 20):
        self.k = k_neighbors

    def estimate_local_dimension(
        self,
        X: torch.Tensor,
        method: str = 'mle'
    ) -> float:
        """
        Estimate intrinsic dimension using maximum likelihood.

        Based on: "Maximum Likelihood Estimation of Intrinsic Dimension"
        (Levina & Bickel, 2004)

        Args:
            X: Data tensor of shape (n_samples, n_features)
            method: Estimation method ('mle' or 'twonn')

        Returns:
            Estimated intrinsic dimension
        """
        X_np = X.detach().cpu().numpy()
        n_samples = X_np.shape[0]

        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X_np)
        distances, _ = nbrs.kneighbors(X_np)

        # Remove self-distance (first column)
        distances = distances[:, 1:]

        if method == 'mle':
            # Maximum likelihood estimator
            # ID = 1 / mean(log(r_k / r_j) for j < k)
            estimates = []
            for i in range(n_samples):
                r_k = distances[i, -1]  # k-th neighbor distance
                if r_k > 0:
                    # Sum log ratios
                    log_ratios = np.log(r_k / distances[i, :-1])
                    log_ratios = log_ratios[distances[i, :-1] > 0]
                    if len(log_ratios) > 0:
                        estimates.append(len(log_ratios) / np.sum(log_ratios))

            return np.mean(estimates) if estimates else 0.0

        elif method == 'twonn':
            # Two-NN estimator (simpler, uses only 2 neighbors)
            # More robust for small samples
            r1 = distances[:, 0]
            r2 = distances[:, 1]

            # Filter valid ratios
            valid = (r1 > 0) & (r2 > 0)
            mu = r2[valid] / r1[valid]

            # ID = log(n) / mean(log(mu))
            return np.log(np.sum(valid)) / np.mean(np.log(mu))

        else:
            raise ValueError(f"Unknown method: {method}")

    def manifold_distance(
        self,
        X: torch.Tensor,
        geodesic: bool = True
    ) -> torch.Tensor:
        """
        Compute pairwise distances respecting manifold structure.

        If geodesic=True, approximates geodesic distances using
        graph shortest paths (Isomap-style).

        Args:
            X: Data tensor of shape (n_samples, n_features)
            geodesic: Whether to compute geodesic distances

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Euclidean distances
        diff = X.unsqueeze(0) - X.unsqueeze(1)
        euclidean = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

        if not geodesic:
            return euclidean

        # Approximate geodesic with graph distances
        # Build k-NN graph
        k = min(self.k, X.shape[0] - 1)
        _, indices = torch.topk(euclidean, k + 1, largest=False, dim=1)

        # Initialize with infinity
        n = X.shape[0]
        geodesic_dist = torch.full((n, n), float('inf'), device=X.device)

        # Set neighbor distances
        for i in range(n):
            for j in indices[i]:
                geodesic_dist[i, j] = euclidean[i, j]
                geodesic_dist[j, i] = euclidean[i, j]

        # Floyd-Warshall for shortest paths
        for k in range(n):
            geodesic_dist = torch.minimum(
                geodesic_dist,
                geodesic_dist[:, k:k+1] + geodesic_dist[k:k+1, :]
            )

        return geodesic_dist


# Example: Verify manifold hypothesis on synthetic data
def demonstrate_manifold_hypothesis():
    """
    Show that points on a manifold have lower intrinsic dimension
    than their ambient dimension.
    """
    analyzer = ManifoldAnalyzer(k_neighbors=10)

    # Generate data on a 2D manifold embedded in 100D
    n_samples = 500
    ambient_dim = 100
    intrinsic_dim = 2

    # Swiss roll: 2D manifold in 3D, then embed in 100D
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))
    height = 10 * np.random.rand(n_samples)

    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)

    # Embed in higher dimension with random projection
    low_dim = np.stack([x, y, z], axis=1)
    projection = np.random.randn(3, ambient_dim)
    high_dim = low_dim @ projection

    # Add small noise
    high_dim += 0.1 * np.random.randn(*high_dim.shape)

    X = torch.tensor(high_dim, dtype=torch.float32)

    # Estimate intrinsic dimension
    id_mle = analyzer.estimate_local_dimension(X, method='mle')
    id_twonn = analyzer.estimate_local_dimension(X, method='twonn')

    print(f"Ambient dimension: {ambient_dim}")
    print(f"True intrinsic dimension: {intrinsic_dim}")
    print(f"Estimated ID (MLE): {id_mle:.2f}")
    print(f"Estimated ID (TwoNN): {id_twonn:.2f}")

    return X, analyzer

# X, analyzer = demonstrate_manifold_hypothesis()
```

---

## Section 2: Representation Manifolds in Neural Networks

### How Networks Transform Manifolds

Each neural network layer transforms the data manifold:

From [Deep Networks as Paths on the Manifold of Neural Representations](https://proceedings.mlr.press/v221/lange23a.html) (Lange et al., 2023, cited by 3):
- Layer-wise computation is a path along a high-dimensional manifold
- Each layer moves representations toward more linearly separable configurations
- The path reveals the network's computational strategy

**Key Insight from Colah's Blog:**

> "Each layer stretches and squishes space, but it never cuts, breaks, or folds it. Intuitively, we can see that it preserves topological properties."

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldTransformationTracker(nn.Module):
    """
    Track how a neural network transforms data manifolds layer by layer.

    This reveals the geometric operations the network performs
    to achieve classification or other tasks.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Storage for activations
        self.activations = []
        self.id_estimates = []

    def forward(self, x: torch.Tensor, track_manifold: bool = False):
        """
        Forward pass with optional manifold tracking.

        Args:
            x: Input tensor
            track_manifold: Whether to store intermediate representations

        Returns:
            Output tensor
        """
        if track_manifold:
            self.activations = [x.detach().clone()]

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.tanh(x)  # Smooth activation preserves topology

            if track_manifold:
                self.activations.append(x.detach().clone())

        # Final layer (no activation for classification logits)
        x = self.layers[-1](x)

        if track_manifold:
            self.activations.append(x.detach().clone())

        return x

    def analyze_manifold_evolution(self, analyzer: 'ManifoldAnalyzer'):
        """
        Analyze how intrinsic dimension changes through the network.

        This reveals the "compression-expansion" pattern typical
        of classification networks.
        """
        if not self.activations:
            raise ValueError("No activations stored. Run forward with track_manifold=True")

        self.id_estimates = []

        for i, activation in enumerate(self.activations):
            if activation.shape[1] >= 3:  # Need enough dimensions
                id_est = analyzer.estimate_local_dimension(activation)
                self.id_estimates.append(id_est)
                print(f"Layer {i}: dim={activation.shape[1]}, ID={id_est:.2f}")
            else:
                self.id_estimates.append(activation.shape[1])
                print(f"Layer {i}: dim={activation.shape[1]} (too small for ID)")

        return self.id_estimates

    def compute_layer_jacobian(
        self,
        x: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute the Jacobian of a layer's transformation.

        The Jacobian reveals local stretching/compression of the manifold.
        Singular values indicate how much each direction is scaled.

        Args:
            x: Input to the layer
            layer_idx: Which layer to analyze

        Returns:
            Jacobian matrix
        """
        x = x.requires_grad_(True)

        # Forward through layers up to layer_idx
        h = x
        for i in range(layer_idx + 1):
            h = self.layers[i](h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)

        # Compute Jacobian
        batch_size = x.shape[0]
        output_dim = h.shape[1]
        input_dim = x.shape[1]

        jacobian = torch.zeros(batch_size, output_dim, input_dim)

        for i in range(output_dim):
            grad = torch.autograd.grad(
                h[:, i].sum(),
                x,
                create_graph=True,
                retain_graph=True
            )[0]
            jacobian[:, i, :] = grad

        return jacobian

    def manifold_curvature_proxy(
        self,
        x: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Estimate local manifold curvature via Jacobian analysis.

        High curvature indicates the network is "bending" the manifold
        significantly at this point.

        Args:
            x: Input tensor
            layer_idx: Which layer to analyze

        Returns:
            Curvature estimate per sample
        """
        jacobian = self.compute_layer_jacobian(x, layer_idx)

        # Singular value decomposition
        # High condition number = high curvature
        curvature = []

        for i in range(jacobian.shape[0]):
            U, S, V = torch.svd(jacobian[i])
            # Condition number as curvature proxy
            condition = S[0] / (S[-1] + 1e-8)
            curvature.append(condition.item())

        return torch.tensor(curvature)


class ManifoldAwareClassifier(nn.Module):
    """
    A classifier that explicitly considers manifold structure.

    Uses geodesic-aware distance in the final classification layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_prototypes_per_class: int = 5
    ):
        super().__init__()

        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Learnable class prototypes on the manifold
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class

        self.prototypes = nn.Parameter(
            torch.randn(n_classes, n_prototypes_per_class, hidden_dim // 2)
        )

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify based on distance to class prototypes.

        This is more manifold-aware than a linear classifier
        because it considers the local neighborhood structure.
        """
        # Encode to representation manifold
        z = self.encoder(x)  # (batch, hidden_dim // 2)

        # Compute distance to each prototype
        # Shape: (batch, n_classes, n_prototypes)
        distances = torch.cdist(
            z.unsqueeze(1),  # (batch, 1, dim)
            self.prototypes  # (n_classes, n_prototypes, dim)
        ).squeeze(1)

        # Min distance to any prototype of each class
        # Shape: (batch, n_classes)
        min_distances = distances.min(dim=-1)[0]

        # Convert to logits (smaller distance = higher logit)
        logits = -min_distances / self.temperature

        return logits

    def get_class_manifold_structure(self):
        """
        Analyze the manifold structure learned for each class.

        Returns information about prototype spread and separation.
        """
        results = {}

        for c in range(self.n_classes):
            class_prototypes = self.prototypes[c]  # (n_prototypes, dim)

            # Intra-class spread
            intra_dist = torch.cdist(class_prototypes, class_prototypes)
            spread = intra_dist.mean().item()

            # Inter-class separation
            other_prototypes = torch.cat([
                self.prototypes[i] for i in range(self.n_classes) if i != c
            ])
            inter_dist = torch.cdist(class_prototypes, other_prototypes)
            separation = inter_dist.min().item()

            results[c] = {
                'spread': spread,
                'separation': separation,
                'ratio': separation / (spread + 1e-8)
            }

        return results
```

### The Hunchback Pattern

From Ansuini et al. (2019), intrinsic dimension follows a characteristic pattern:

```
Input → [Expansion] → Peak ID → [Compression] → Output

Early layers: ID increases (disentangling features)
Middle layers: ID peaks (maximum information)
Late layers: ID decreases (task-relevant compression)
```

This pattern reflects:
1. **Expansion**: Unfolding tangled manifolds
2. **Compression**: Discarding task-irrelevant dimensions

---

## Section 3: Intrinsic Dimensionality Estimation

### Why Intrinsic Dimension Matters

From [Measuring the Intrinsic Dimension of Objective Landscapes](https://openreview.net/pdf?id=ryup8-WCW) (Li et al., 2018, cited by 551):
- Many problems can be solved in a much lower-dimensional subspace
- The intrinsic dimension reveals the "true complexity" of a task
- Networks are often vastly overparameterized relative to intrinsic dimension

**Key Finding**: MNIST can be solved with ~750 intrinsic parameters, not 7M!

### Estimation Methods

```python
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

class IntrinsicDimensionEstimator:
    """
    Multiple methods for estimating intrinsic dimensionality.

    Different methods have different assumptions:
    - MLE: Assumes locally uniform density
    - TwoNN: Robust to density variations
    - PCA: Assumes linear manifold (baseline)
    - Correlation dimension: Fractal-based
    """

    @staticmethod
    def mle_estimator(
        X: np.ndarray,
        k: int = 20,
        averaging: str = 'global'
    ) -> float:
        """
        Maximum Likelihood Estimation of intrinsic dimension.

        From: Levina & Bickel (2004)

        Args:
            X: Data array (n_samples, n_features)
            k: Number of neighbors
            averaging: 'global' or 'local'

        Returns:
            Estimated intrinsic dimension
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]

        # Find k+1 nearest neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:]  # Remove self

        # Local ID estimates
        local_ids = []

        for i in range(n_samples):
            r_k = distances[i, -1]
            if r_k > 0:
                # MLE formula: ID = (k-1) / sum(log(r_k / r_j))
                log_sum = np.sum(np.log(r_k / (distances[i, :-1] + 1e-10)))
                if log_sum > 0:
                    local_ids.append((k - 1) / log_sum)

        if averaging == 'global':
            # Harmonic mean (more robust)
            return len(local_ids) / np.sum(1.0 / np.array(local_ids))
        else:
            return np.mean(local_ids)

    @staticmethod
    def twonn_estimator(X: np.ndarray) -> float:
        """
        Two Nearest Neighbors estimator.

        From: Facco et al. (2017)
        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"

        More robust than MLE for non-uniform densities.

        Args:
            X: Data array (n_samples, n_features)

        Returns:
            Estimated intrinsic dimension
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]

        # Find 3 nearest neighbors (self + 2)
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)

        r1 = distances[:, 1]  # First neighbor
        r2 = distances[:, 2]  # Second neighbor

        # Compute mu = r2 / r1
        valid = r1 > 0
        mu = r2[valid] / r1[valid]

        # Sort and compute empirical CDF
        mu_sorted = np.sort(mu)
        n = len(mu_sorted)

        # Fit: log(1 - F(mu)) = -d * log(mu)
        # Using least squares on log scale
        F = np.arange(1, n + 1) / n

        # Avoid log(0)
        valid_F = F < 1
        log_mu = np.log(mu_sorted[valid_F])
        log_1_minus_F = np.log(1 - F[valid_F])

        # Linear regression: log(1-F) = -d * log(mu)
        # d = -cov(log_mu, log_1_minus_F) / var(log_mu)
        d = -np.polyfit(log_mu, log_1_minus_F, 1)[0]

        return d

    @staticmethod
    def pca_estimator(
        X: np.ndarray,
        variance_threshold: float = 0.95
    ) -> int:
        """
        PCA-based dimension estimate.

        Finds the number of components explaining threshold variance.
        Assumes linear manifold (baseline method).

        Args:
            X: Data array (n_samples, n_features)
            variance_threshold: Cumulative variance to explain

        Returns:
            Number of principal components needed
        """
        pca = PCA()
        pca.fit(X)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        return int(np.searchsorted(cumsum, variance_threshold) + 1)

    @staticmethod
    def correlation_dimension(
        X: np.ndarray,
        r_min: float = None,
        r_max: float = None,
        n_points: int = 20
    ) -> float:
        """
        Correlation dimension estimator.

        Based on the correlation integral:
        C(r) ~ r^d as r -> 0

        Args:
            X: Data array (n_samples, n_features)
            r_min, r_max: Range of radii to consider
            n_points: Number of radii to sample

        Returns:
            Estimated correlation dimension
        """
        # Compute all pairwise distances
        distances = pdist(X)

        if r_min is None:
            r_min = np.percentile(distances, 1)
        if r_max is None:
            r_max = np.percentile(distances, 50)

        # Sample radii logarithmically
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)

        # Compute correlation integral
        n_pairs = len(distances)
        C = np.array([np.sum(distances < r) / n_pairs for r in radii])

        # Filter valid points
        valid = C > 0
        log_r = np.log(radii[valid])
        log_C = np.log(C[valid])

        # Estimate dimension as slope
        if len(log_r) < 2:
            return 0.0

        slope, _ = np.polyfit(log_r, log_C, 1)
        return slope


def analyze_network_intrinsic_dimension(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    n_samples: int = 1000
):
    """
    Analyze how intrinsic dimension changes through a network.

    This reveals the compression/expansion pattern characteristic
    of deep learning.

    Args:
        model: Neural network model
        data_loader: Data to analyze
        n_samples: Number of samples to use

    Returns:
        Dictionary mapping layer names to ID estimates
    """
    estimator = IntrinsicDimensionEstimator()

    # Collect activations
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Collect samples
    model.eval()
    samples_collected = 0

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            model(x)
            samples_collected += x.shape[0]

            if samples_collected >= n_samples:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Estimate ID for each layer
    results = {}

    for name, acts in activations.items():
        # Concatenate and flatten
        acts = torch.cat(acts, dim=0)[:n_samples]
        acts = acts.view(acts.shape[0], -1).numpy()

        if acts.shape[1] >= 10:  # Need enough dimensions
            id_mle = estimator.mle_estimator(acts, k=min(20, acts.shape[0] // 2))
            id_twonn = estimator.twonn_estimator(acts)

            results[name] = {
                'output_dim': acts.shape[1],
                'id_mle': id_mle,
                'id_twonn': id_twonn,
                'compression_ratio': acts.shape[1] / id_mle
            }

            print(f"{name}: dim={acts.shape[1]}, "
                  f"ID_MLE={id_mle:.1f}, ID_TwoNN={id_twonn:.1f}")

    return results
```

### Performance Considerations

**Computational Complexity:**
- MLE: O(n * k * d) for k-NN search
- TwoNN: O(n * log(n)) with efficient k-NN
- Correlation dimension: O(n^2) for pairwise distances

**GPU Acceleration:**

```python
def gpu_knn_distances(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    GPU-accelerated k-NN distance computation.

    For large datasets, this is significantly faster than CPU sklearn.

    Args:
        X: Data tensor on GPU (n_samples, n_features)
        k: Number of neighbors

    Returns:
        Distances to k nearest neighbors (n_samples, k)
    """
    # Compute all pairwise distances
    # Using squared distances for efficiency
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    distances_sq = X_norm + X_norm.t() - 2 * X @ X.t()
    distances_sq = torch.clamp(distances_sq, min=0)  # Numerical stability

    # Find k smallest (excluding self)
    distances_sq.fill_diagonal_(float('inf'))
    topk_sq, _ = torch.topk(distances_sq, k, largest=False, dim=1)

    return torch.sqrt(topk_sq)
```

---

## Section 4: Code - Complete Manifold Dimension Estimation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

class NeuralManifoldAnalyzer:
    """
    Comprehensive tool for analyzing neural network manifolds.

    Provides:
    - Intrinsic dimension estimation
    - Layer-wise manifold analysis
    - Topological change detection
    - Visualization support

    Based on research from:
    - Ansuini et al. (2019) - ID in deep networks
    - Li et al. (2018) - Intrinsic dimension of objectives
    - Colah (2014) - NN manifold topology
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.k = k_neighbors
        self.device = device

    def estimate_id(
        self,
        X: torch.Tensor,
        method: str = 'ensemble'
    ) -> Dict[str, float]:
        """
        Estimate intrinsic dimension with multiple methods.

        Args:
            X: Data tensor (n_samples, n_features)
            method: 'mle', 'twonn', 'pca', or 'ensemble'

        Returns:
            Dictionary of ID estimates
        """
        X = X.to(self.device)
        X_flat = X.view(X.shape[0], -1)

        results = {}

        # Maximum Likelihood Estimation
        if method in ['mle', 'ensemble']:
            results['mle'] = self._mle_id(X_flat)

        # Two Nearest Neighbors
        if method in ['twonn', 'ensemble']:
            results['twonn'] = self._twonn_id(X_flat)

        # PCA baseline
        if method in ['pca', 'ensemble']:
            results['pca'] = self._pca_id(X_flat)

        # Ensemble: geometric mean
        if method == 'ensemble':
            values = [v for v in results.values() if v > 0]
            results['ensemble'] = np.exp(np.mean(np.log(values)))

        return results

    def _mle_id(self, X: torch.Tensor) -> float:
        """MLE intrinsic dimension estimator (GPU-accelerated)."""
        n_samples = X.shape[0]
        k = min(self.k, n_samples - 1)

        # Compute k-NN distances on GPU
        distances = self._knn_distances(X, k)

        # MLE formula
        r_k = distances[:, -1:]  # (n, 1)
        ratios = r_k / (distances[:, :-1] + 1e-10)  # (n, k-1)
        log_ratios = torch.log(ratios)

        # Local estimates
        local_ids = (k - 1) / log_ratios.sum(dim=1)

        # Filter outliers
        valid = (local_ids > 0) & (local_ids < X.shape[1])

        if valid.sum() == 0:
            return 1.0

        # Harmonic mean
        return (valid.sum() / (1.0 / local_ids[valid]).sum()).item()

    def _twonn_id(self, X: torch.Tensor) -> float:
        """TwoNN intrinsic dimension estimator."""
        # Get 2 nearest neighbor distances
        distances = self._knn_distances(X, 2)

        r1 = distances[:, 0]
        r2 = distances[:, 1]

        # Filter valid
        valid = r1 > 0
        mu = r2[valid] / r1[valid]

        # Sort for CDF
        mu_sorted, _ = torch.sort(mu)
        n = len(mu_sorted)

        # Empirical CDF
        F = torch.arange(1, n + 1, device=self.device).float() / n

        # Fit: log(1 - F) = -d * log(mu)
        valid_F = F < 0.9  # Avoid boundary effects
        log_mu = torch.log(mu_sorted[valid_F])
        log_1_minus_F = torch.log(1 - F[valid_F])

        # Linear regression
        X_reg = torch.stack([log_mu, torch.ones_like(log_mu)], dim=1)
        coeffs = torch.linalg.lstsq(X_reg, log_1_minus_F).solution

        return -coeffs[0].item()

    def _pca_id(self, X: torch.Tensor, threshold: float = 0.95) -> int:
        """PCA-based dimension estimate."""
        # Center data
        X_centered = X - X.mean(dim=0)

        # Compute covariance
        cov = X_centered.t() @ X_centered / (X.shape[0] - 1)

        # Eigendecomposition
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = torch.flip(eigenvalues, [0])  # Descending

        # Cumulative variance
        total_var = eigenvalues.sum()
        cumsum = torch.cumsum(eigenvalues, dim=0) / total_var

        return int((cumsum < threshold).sum().item()) + 1

    def _knn_distances(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-NN distances on GPU."""
        # Pairwise squared distances
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        dist_sq = X_norm + X_norm.t() - 2 * X @ X.t()
        dist_sq = torch.clamp(dist_sq, min=0)

        # Exclude self
        dist_sq.fill_diagonal_(float('inf'))

        # Top-k smallest
        topk_sq, _ = torch.topk(dist_sq, k, largest=False, dim=1)

        return torch.sqrt(topk_sq)

    def analyze_layer_wise(
        self,
        model: nn.Module,
        X: torch.Tensor,
        layer_types: tuple = (nn.Linear, nn.Conv2d)
    ) -> Dict[str, Dict]:
        """
        Analyze intrinsic dimension at each layer.

        Args:
            model: Neural network
            X: Input data
            layer_types: Types of layers to analyze

        Returns:
            Layer-wise ID analysis
        """
        activations = {}
        hooks = []

        # Register hooks
        def hook_fn(name):
            def hook(module, input, output):
                # Flatten spatial dimensions
                out = output.detach()
                if out.dim() > 2:
                    out = out.view(out.shape[0], -1)
                activations[name] = out
            return hook

        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        model.eval()
        with torch.no_grad():
            model(X.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Analyze each layer
        results = {}

        for name, acts in activations.items():
            if acts.shape[1] >= 10:
                id_est = self.estimate_id(acts, method='ensemble')

                results[name] = {
                    'output_dim': acts.shape[1],
                    **id_est,
                    'compression': acts.shape[1] / id_est['ensemble']
                }

        return results

    def topological_complexity(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate topological complexity of a classification task.

        Based on: How tangled are the class manifolds?

        Args:
            X: Data tensor
            y: Class labels

        Returns:
            Complexity metrics
        """
        X = X.to(self.device)
        y = y.to(self.device)

        classes = torch.unique(y)
        n_classes = len(classes)

        # Compute class-wise statistics
        class_means = []
        class_covs = []
        class_ids = []

        for c in classes:
            mask = y == c
            X_c = X[mask]

            class_means.append(X_c.mean(dim=0))
            class_covs.append(torch.cov(X_c.t()))
            class_ids.append(self.estimate_id(X_c)['ensemble'])

        # Inter-class distances
        mean_stack = torch.stack(class_means)
        inter_dist = torch.cdist(mean_stack, mean_stack)

        # Average separation
        avg_separation = inter_dist[torch.triu(torch.ones_like(inter_dist), diagonal=1) > 0].mean()

        # Average spread (using covariance trace)
        avg_spread = sum(torch.trace(cov) for cov in class_covs) / n_classes

        # Separation ratio
        sep_ratio = avg_separation / (torch.sqrt(avg_spread) + 1e-8)

        return {
            'avg_class_id': sum(class_ids) / len(class_ids),
            'global_id': self.estimate_id(X)['ensemble'],
            'separation_ratio': sep_ratio.item(),
            'complexity_score': sum(class_ids) / (sep_ratio.item() + 1e-8)
        }


def manifold_regularized_loss(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    activations: torch.Tensor,
    lambda_manifold: float = 0.01
) -> torch.Tensor:
    """
    Loss function with manifold regularization.

    Encourages smooth manifold structure in representations.
    Based on: Contractive autoencoders (Rifai et al., 2011)

    Args:
        model: Neural network
        X: Input data
        y: Labels
        activations: Hidden layer activations
        lambda_manifold: Regularization strength

    Returns:
        Combined loss
    """
    # Classification loss
    logits = model(X)
    cls_loss = F.cross_entropy(logits, y)

    # Manifold regularization: penalize Frobenius norm of Jacobian
    # This encourages locally flat manifold (contractive)
    jacobian_norm = 0.0

    for i in range(activations.shape[1]):
        grad = torch.autograd.grad(
            activations[:, i].sum(),
            X,
            create_graph=True,
            retain_graph=True
        )[0]
        jacobian_norm += (grad ** 2).sum()

    jacobian_norm /= activations.shape[0]

    return cls_loss + lambda_manifold * jacobian_norm


# Example usage
def example_manifold_analysis():
    """
    Demonstrate manifold analysis on a simple network.
    """
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 500
    n_features = 100
    n_classes = 5

    # Generate data on class manifolds
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    # Simple network
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes)
    )

    # Analyze
    analyzer = NeuralManifoldAnalyzer(k_neighbors=15)

    # Input ID
    input_id = analyzer.estimate_id(X)
    print(f"Input ID: {input_id}")

    # Layer-wise analysis
    print("\nLayer-wise ID:")
    layer_results = analyzer.analyze_layer_wise(model, X)
    for name, stats in layer_results.items():
        print(f"  {name}: dim={stats['output_dim']}, "
              f"ID={stats['ensemble']:.1f}, "
              f"compression={stats['compression']:.1f}x")

    # Topological complexity
    complexity = analyzer.topological_complexity(X, y)
    print(f"\nTopological complexity: {complexity}")

    return analyzer, model, X, y

# analyzer, model, X, y = example_manifold_analysis()
```

---

## Section 5: TRAIN STATION - Manifold = Embedding = Representation

### The Grand Unification

**TRAIN STATION**: Three concepts that are fundamentally the same geometric object:

```
MANIFOLD    =    EMBEDDING    =    REPRESENTATION
   |                |                    |
   v                v                    v
 Curved         Low-dim              Feature
 surface        projection           space

All describe: The true structure underlying high-dim data
```

### Why They're the Same

**1. Manifold Learning View:**
- Data lies on a curved surface in high-D
- Goal: Discover this surface structure
- Methods: Isomap, LLE, t-SNE, UMAP

**2. Embedding View:**
- Map high-D data to low-D representation
- Goal: Preserve important relationships
- Methods: Word2Vec, Node2Vec, autoencoders

**3. Representation Learning View:**
- Learn useful features for downstream tasks
- Goal: Extract task-relevant information
- Methods: Deep learning, self-supervised learning

### The Unified Perspective

```python
class UnifiedManifoldView:
    """
    Demonstrates the equivalence of manifold/embedding/representation.

    THE TRAIN STATION: These are all the same thing!

    - Manifold: The geometric structure of data
    - Embedding: The coordinates on that structure
    - Representation: The features that capture it
    """

    @staticmethod
    def manifold_to_embedding(manifold_points: torch.Tensor) -> torch.Tensor:
        """
        A manifold IS an embedding.

        The manifold coordinates ARE the embedding coordinates.
        There's no transformation - they're the same object.
        """
        return manifold_points  # Identity!

    @staticmethod
    def embedding_to_representation(embedding: torch.Tensor) -> torch.Tensor:
        """
        An embedding IS a representation.

        The embedding coordinates ARE the representation features.
        """
        return embedding  # Identity!

    @staticmethod
    def representation_to_manifold(representation: torch.Tensor) -> torch.Tensor:
        """
        A representation defines a manifold.

        The representation space IS a manifold.
        """
        return representation  # Identity!

    @staticmethod
    def unified_learning(
        encoder: nn.Module,
        decoder: nn.Module,
        X: torch.Tensor,
        method: str = 'autoencoder'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn the unified manifold/embedding/representation.

        All three views are satisfied by learning a good encoder:
        - Manifold: The image of the encoder
        - Embedding: The encoder output
        - Representation: The features for tasks

        Args:
            encoder: Maps data to low-dimensional space
            decoder: Reconstructs from low-dim (for training)
            X: Input data
            method: Learning method

        Returns:
            The unified representation and reconstruction
        """
        # Encode = find manifold = compute embedding = extract representation
        z = encoder(X)

        # Decode = map back to ambient space
        X_reconstructed = decoder(z)

        return z, X_reconstructed


class ManifoldAwareTransformer(nn.Module):
    """
    Transformer that explicitly respects manifold structure.

    THE INSIGHT: Attention computes relationships ON the manifold.

    Standard attention: softmax(QK^T / sqrt(d))
    Manifold attention: softmax(geodesic_similarity(Q, K))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        manifold_aware: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.manifold_aware = manifold_aware

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if manifold_aware:
            # Learn local metric tensor
            self.metric = nn.Parameter(torch.eye(self.d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with manifold-aware attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq, d_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.manifold_aware:
            # Manifold-aware similarity using learned metric
            # This respects local curvature
            Q_metric = Q @ self.metric
            scores = Q_metric @ K.transpose(-2, -1) / np.sqrt(self.d_head)
        else:
            # Standard dot-product attention
            scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_head)

        # Attention weights
        attn = F.softmax(scores, dim=-1)

        # Apply to values
        out = attn @ V

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out
```

### Connections to Other Topics

**Manifold Learning meets Predictive Coding:**
- Predictions flow on the manifold
- Prediction errors measure deviation from manifold
- Learning = refining the manifold model

**Manifold Learning meets Active Inference:**
- Beliefs form a manifold in probability space
- Free energy measures distance on this manifold
- Inference = geodesic descent on belief manifold

**Manifold Learning meets Loss Landscapes:**
- Parameter space forms a manifold
- Loss function defines geometry
- Optimization = navigation on parameter manifold

---

## Section 6: ARR-COC-0-1 Connection - Relevance Manifold

### Relevance as Manifold Geometry (10%)

The ARR-COC-0-1 relevance reranker can be understood through manifold geometry:

```python
class RelevanceManifoldReranker:
    """
    Relevance reranking using manifold structure.

    KEY INSIGHT: Relevant items form a coherent manifold.
    Irrelevant items scatter randomly in representation space.

    A good relevance model should:
    1. Learn the manifold of relevant content
    2. Measure distance to this manifold
    3. Rank by manifold membership
    """

    def __init__(
        self,
        encoder: nn.Module,
        manifold_dim: int,
        n_prototypes: int = 100
    ):
        self.encoder = encoder
        self.manifold_dim = manifold_dim

        # Prototype points on relevance manifold
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, manifold_dim)
        )

        # Intrinsic dimension tracker
        self.analyzer = NeuralManifoldAnalyzer()

    def compute_relevance(
        self,
        query: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance as distance to learned manifold.

        Args:
            query: Query representation
            items: Item representations

        Returns:
            Relevance scores (higher = more relevant)
        """
        # Encode to manifold space
        query_encoded = self.encoder(query)
        items_encoded = self.encoder(items)

        # Distance to nearest prototype
        query_to_proto = torch.cdist(query_encoded, self.prototypes).min(dim=-1)[0]
        items_to_proto = torch.cdist(items_encoded, self.prototypes).min(dim=-1)[0]

        # Combined relevance: close to query AND on manifold
        query_item_dist = torch.cdist(query_encoded, items_encoded).squeeze(0)
        manifold_membership = -items_to_proto

        relevance = -query_item_dist + 0.5 * manifold_membership

        return relevance

    def analyze_relevance_manifold(
        self,
        relevant_items: torch.Tensor,
        irrelevant_items: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze the manifold structure of relevant vs irrelevant items.

        HYPOTHESIS: Relevant items have lower intrinsic dimension
        (they form a more structured manifold).

        Args:
            relevant_items: Items known to be relevant
            irrelevant_items: Items known to be irrelevant

        Returns:
            Manifold structure comparison
        """
        rel_encoded = self.encoder(relevant_items)
        irrel_encoded = self.encoder(irrelevant_items)

        rel_id = self.analyzer.estimate_id(rel_encoded)['ensemble']
        irrel_id = self.analyzer.estimate_id(irrel_encoded)['ensemble']

        return {
            'relevant_id': rel_id,
            'irrelevant_id': irrel_id,
            'id_ratio': irrel_id / (rel_id + 1e-8),
            'manifold_coherence': 1 / (rel_id + 1)
        }


class ManifoldGuidedTokenAllocation:
    """
    Token allocation guided by manifold structure.

    Allocate more tokens to regions that:
    1. Have high local intrinsic dimension (complex)
    2. Are far from the learned manifold (surprising)
    3. Have high curvature (need more representation)
    """

    def __init__(self, base_tokens: int, max_tokens: int):
        self.base_tokens = base_tokens
        self.max_tokens = max_tokens

    def allocate(
        self,
        content_embeddings: torch.Tensor,
        analyzer: NeuralManifoldAnalyzer
    ) -> torch.Tensor:
        """
        Allocate tokens based on manifold complexity.

        Args:
            content_embeddings: Content representations
            analyzer: Manifold analyzer

        Returns:
            Token allocation per content item
        """
        n_items = content_embeddings.shape[0]

        # Estimate local complexity
        local_ids = []

        for i in range(n_items):
            # Use surrounding items for local estimate
            start = max(0, i - 10)
            end = min(n_items, i + 10)
            local_data = content_embeddings[start:end]

            if local_data.shape[0] >= 5:
                local_id = analyzer.estimate_id(local_data)['mle']
            else:
                local_id = 1.0

            local_ids.append(local_id)

        local_ids = torch.tensor(local_ids)

        # Normalize to [0, 1]
        complexity = (local_ids - local_ids.min()) / (local_ids.max() - local_ids.min() + 1e-8)

        # Map to token allocation
        tokens = self.base_tokens + complexity * (self.max_tokens - self.base_tokens)

        return tokens.long()
```

### Design Implications for ARR-COC-0-1

**1. Use Manifold-Aware Embeddings:**
- Train encoders that preserve manifold structure
- Measure relevance as manifold distance, not just cosine similarity

**2. Adaptive Token Allocation:**
- More tokens for high-ID regions (complex content)
- Fewer tokens for low-ID regions (simple/structured)

**3. Quality Metrics:**
- ID of relevant set should be lower than ID of random set
- This indicates the model has learned coherent relevance structure

---

## Performance Notes

### Computational Costs

| Method | Complexity | Memory | GPU-Friendly |
|--------|------------|--------|--------------|
| MLE ID | O(n * k * d) | O(n * k) | Yes |
| TwoNN ID | O(n * log n) | O(n) | Yes |
| PCA ID | O(min(n, d)^2 * max(n, d)) | O(n * d) | Yes |
| Correlation | O(n^2) | O(n^2) | Limited |

### Practical Guidelines

**For ID Estimation:**
- Use k = 10-20 neighbors for stable estimates
- Need at least 50 * ID samples for reliable estimate
- TwoNN is most robust for non-uniform densities

**For Layer-wise Analysis:**
- Sample 1000-5000 activations
- Compare ID to layer width for compression ratio
- Look for the "hunchback" pattern

**Memory Optimization:**
```python
# Batch processing for large datasets
def batched_id_estimation(X, batch_size=1000, analyzer=None):
    """
    Estimate ID on large datasets by batching.
    """
    if analyzer is None:
        analyzer = NeuralManifoldAnalyzer()

    n_samples = X.shape[0]
    id_estimates = []

    for i in range(0, n_samples, batch_size):
        batch = X[i:i + batch_size]
        if batch.shape[0] >= 50:
            id_est = analyzer.estimate_id(batch)['ensemble']
            id_estimates.append(id_est)

    return np.mean(id_estimates)
```

---

## Sources

**Primary References:**

- [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) - Colah's blog (2014)
- [Intrinsic dimension of data representations in deep neural networks](https://arxiv.org/abs/1905.12784) - Ansuini et al. (2019), cited by 401
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://openreview.net/pdf?id=ryup8-WCW) - Li et al. (2018), cited by 551

**Additional Sources:**

- [A Manifold Learning Perspective on Representation Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8625121/) - Schuster (2021)
- [Deep Networks as Paths on the Manifold of Neural Representations](https://proceedings.mlr.press/v221/lange23a.html) - Lange et al. (2023)
- [The training process of many deep networks explores...](https://www.pnas.org/doi/10.1073/pnas.2310002121) - Mao et al. (2024)
- [Representation Learning via Manifold Flattening](https://jmlr.org/papers/v25/23-0615.html) - Psenka et al. (2024)
- [Wikipedia - Manifold hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis)

**Intrinsic Dimension Methods:**

- Levina & Bickel (2004) - MLE estimator
- Facco et al. (2017) - TwoNN estimator
- [Intrinsic dimension estimation: Advances and open problems](https://math.umd.edu/~rvbalan/RESEARCH/DimensionEstimation/camastra2016.pdf) - Camastra (2016)

---

## Summary

Neural network manifolds provide a geometric lens for understanding deep learning:

1. **Data Manifold Hypothesis**: High-D data lies on low-D manifolds
2. **Layer Transformations**: Networks continuously deform manifolds
3. **Intrinsic Dimension**: Measures true complexity vs ambient dimension
4. **Topological Constraints**: Determine minimum network width

**THE TRAIN STATION**: Manifold = Embedding = Representation
- All three describe the same geometric structure
- Understanding one means understanding all
- This unification connects manifold learning, representation learning, and deep learning

**For ARR-COC-0-1**: Relevance forms a manifold - coherent relevant items cluster on a low-dimensional structure while irrelevant items scatter. Measuring distance to this manifold provides geometrically-grounded relevance scores.
