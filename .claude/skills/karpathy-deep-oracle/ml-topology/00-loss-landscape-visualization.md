# Loss Landscape Visualization

## Overview

Loss landscape visualization reveals the geometry of neural network optimization surfaces, providing crucial insights into why certain architectures train better and generalize well. The foundational work by Li et al. (2018) introduced filter normalization techniques that enable meaningful comparisons across different network architectures, fundamentally changing how we understand deep learning optimization.

**Core Insight**: The loss landscape is a high-dimensional surface where gradient descent navigates toward minima. Visualizing 2D slices of this surface reveals dramatic differences between architectures - skip connections create smooth, convex basins while deep networks without them develop chaotic, non-convex topologies.

---

## Section 1: The Challenge of Visualizing High-Dimensional Loss Landscapes

### The Dimensionality Problem

Modern neural networks have millions to billions of parameters, creating loss landscapes in correspondingly high-dimensional spaces. Direct visualization is impossible, requiring dimensionality reduction techniques.

**Key Approaches:**

1. **1D Linear Interpolation**: Plot loss along the line between two minima
2. **2D Random Direction Plots**: Sample loss in a plane defined by two random directions
3. **PCA-Based Directions**: Use principal components of optimization trajectory
4. **Hessian Eigenvector Directions**: Align with curvature information

### Mathematical Formulation

For a neural network with parameters theta*, the loss landscape is visualized by:

```
L(alpha, beta) = L(theta* + alpha * d1 + beta * d2)
```

Where:
- `theta*` is the trained model's parameters
- `d1, d2` are direction vectors (random or computed)
- `alpha, beta` are scalar coordinates in the visualization plane

### Why Naive Visualization Fails

**Scale Invariance Problem**: Neural networks have inherent scale invariance - you can multiply one layer's weights by a constant and divide the next layer's by the same constant without changing the function. This makes random directions misleading because:

- Different layers have different weight magnitudes
- Filters within layers have varying scales
- Comparing sharpness across architectures becomes meaningless

**Solution**: Filter Normalization

From [Li et al. 2018 - Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913):

> "We introduce a simple 'filter normalization' method that helps us visualize loss function curvature and make meaningful side-by-side comparisons between loss functions."

---

## Section 2: Filter Normalization Technique

### The Core Innovation

Filter normalization scales each filter in the random direction to have the same norm as the corresponding filter in the trained model. This removes scale ambiguity and enables meaningful comparisons.

### Mathematical Definition

For each filter `i` in layer `l`, the normalized direction is:

```
d_i^l = ||theta_i^l|| * (r_i^l / ||r_i^l||)
```

Where:
- `theta_i^l` is the trained filter
- `r_i^l` is the random direction for that filter
- `||.||` denotes the Frobenius norm

### Implementation in PyTorch

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import copy

class FilterNormalizedDirection:
    """
    Generate filter-normalized random directions for loss landscape visualization.

    Based on Li et al. 2018: "Visualizing the Loss Landscape of Neural Nets"
    https://arxiv.org/abs/1712.09913
    """

    def __init__(self, model: nn.Module, ignore_bn: bool = True):
        """
        Args:
            model: The trained neural network
            ignore_bn: Whether to ignore BatchNorm parameters
        """
        self.model = model
        self.ignore_bn = ignore_bn
        self.state_dict = copy.deepcopy(model.state_dict())

    def create_random_direction(self) -> Dict[str, torch.Tensor]:
        """Create a filter-normalized random direction."""
        direction = {}

        for name, param in self.state_dict.items():
            # Skip BatchNorm statistics and biases if requested
            if self.ignore_bn and ('running_mean' in name or
                                   'running_var' in name or
                                   'num_batches_tracked' in name):
                direction[name] = torch.zeros_like(param)
                continue

            if self.ignore_bn and 'bias' in name:
                direction[name] = torch.zeros_like(param)
                continue

            # Generate random direction
            random_dir = torch.randn_like(param)

            # Apply filter normalization
            if len(param.shape) >= 2:  # Conv or Linear layer
                direction[name] = self._filter_normalize(param, random_dir)
            else:
                # For 1D params, just scale to match norm
                if param.norm() > 1e-10:
                    direction[name] = random_dir * (param.norm() / random_dir.norm())
                else:
                    direction[name] = random_dir

        return direction

    def _filter_normalize(self, weights: torch.Tensor,
                         direction: torch.Tensor) -> torch.Tensor:
        """
        Normalize direction filter-wise to match weight norms.

        For conv layers: each filter is a 3D tensor (out_channels, in_channels, k, k)
        For linear layers: each filter is a row (in_features,)
        """
        normalized = torch.zeros_like(direction)

        if len(weights.shape) == 4:  # Conv layer
            for i in range(weights.shape[0]):  # For each output filter
                filter_weights = weights[i]
                filter_dir = direction[i]

                weight_norm = filter_weights.norm()
                dir_norm = filter_dir.norm()

                if dir_norm > 1e-10 and weight_norm > 1e-10:
                    normalized[i] = filter_dir * (weight_norm / dir_norm)
                else:
                    normalized[i] = filter_dir

        elif len(weights.shape) == 2:  # Linear layer
            for i in range(weights.shape[0]):  # For each output neuron
                weight_norm = weights[i].norm()
                dir_norm = direction[i].norm()

                if dir_norm > 1e-10 and weight_norm > 1e-10:
                    normalized[i] = direction[i] * (weight_norm / dir_norm)
                else:
                    normalized[i] = direction[i]
        else:
            normalized = direction

        return normalized

    def get_perturbed_model(self, alpha: float,
                           direction: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Get model with parameters perturbed along direction.

        new_params = original_params + alpha * direction
        """
        model_copy = copy.deepcopy(self.model)
        new_state_dict = {}

        for name, param in self.state_dict.items():
            new_state_dict[name] = param + alpha * direction[name]

        model_copy.load_state_dict(new_state_dict)
        return model_copy


def compute_loss_landscape_1d(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    alphas: np.ndarray,
    device: torch.device = torch.device('cuda')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D loss landscape along a filter-normalized direction.

    Args:
        model: Trained model
        dataloader: Data for loss computation
        criterion: Loss function
        alphas: Array of perturbation magnitudes
        device: Computation device

    Returns:
        losses: Loss values at each alpha
        accuracies: Accuracy values at each alpha
    """
    model = model.to(device)
    dir_generator = FilterNormalizedDirection(model, ignore_bn=True)
    direction = dir_generator.create_random_direction()

    losses = []
    accuracies = []

    for alpha in alphas:
        # Get perturbed model
        perturbed_model = dir_generator.get_perturbed_model(alpha, direction)
        perturbed_model = perturbed_model.to(device)
        perturbed_model.eval()

        # Compute loss and accuracy
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = perturbed_model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        losses.append(total_loss / total)
        accuracies.append(100. * correct / total)

    return np.array(losses), np.array(accuracies)


def compute_loss_landscape_2d(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    alphas: np.ndarray,
    betas: np.ndarray,
    device: torch.device = torch.device('cuda')
) -> np.ndarray:
    """
    Compute 2D loss landscape along two filter-normalized directions.

    Args:
        model: Trained model
        dataloader: Data for loss computation
        criterion: Loss function
        alphas: Array of perturbation magnitudes for direction 1
        betas: Array of perturbation magnitudes for direction 2
        device: Computation device

    Returns:
        loss_surface: 2D array of loss values
    """
    model = model.to(device)
    dir_generator = FilterNormalizedDirection(model, ignore_bn=True)

    # Create two orthogonal directions
    direction1 = dir_generator.create_random_direction()
    direction2 = dir_generator.create_random_direction()

    loss_surface = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb model along both directions
            new_state_dict = {}
            for name in dir_generator.state_dict.keys():
                new_state_dict[name] = (dir_generator.state_dict[name] +
                                       alpha * direction1[name] +
                                       beta * direction2[name])

            perturbed_model = copy.deepcopy(model)
            perturbed_model.load_state_dict(new_state_dict)
            perturbed_model = perturbed_model.to(device)
            perturbed_model.eval()

            # Compute loss
            total_loss = 0.0
            total = 0

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = perturbed_model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    total += targets.size(0)

            loss_surface[i, j] = total_loss / total

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Computed row {i+1}/{len(alphas)}")

    return loss_surface
```

### Why Filter Normalization Matters

**Without Normalization**:
- Sharpness comparisons are meaningless
- Different architectures look arbitrarily different
- Cannot compare batch sizes, learning rates, etc.

**With Normalization**:
- Reveals true geometric differences
- Enables fair architecture comparisons
- Sharpness correlates with generalization

---

## Section 3: Sharp vs Flat Minima and Generalization

### The Sharp/Flat Hypothesis

**Flat Minima Hypothesis** (Hochreiter & Schmidhuber, 1997):
- Flat minima generalize better than sharp minima
- Can be specified with lower precision
- More robust to perturbations

From [Keskar et al. 2017 - On Large-Batch Training](https://arxiv.org/abs/1609.04836):

> "Large-batch methods tend to converge to sharp minimizers... small-batch methods converge to flat minimizers."

### Mathematical Definition of Sharpness

**Epsilon-Sharpness** (Keskar et al.):

```
sharpness(theta, epsilon) = max_{||delta|| <= epsilon} [L(theta + delta) - L(theta)] / (1 + L(theta))
```

This measures the maximum loss increase within an epsilon-ball around the minimum.

### The Controversy: Sharp Minima CAN Generalize

**Key Counter-Paper**: [Dinh et al. 2017 - Sharp Minima Can Generalize](https://arxiv.org/abs/1703.04933)

**Main Argument**: Sharpness is not invariant to reparameterization. You can take a flat minimum and make it arbitrarily sharp by rescaling layers - without changing the function!

```python
def demonstrate_reparameterization_invariance():
    """
    Show that sharpness depends on parameterization, not just function.

    For any flat minimum, we can create an equivalent sharp minimum
    by rescaling consecutive layers.
    """
    # Original network: y = W2 * ReLU(W1 * x)
    # Reparameterized: y = (alpha * W2) * ReLU((1/alpha) * W1 * x)
    # Same function, different geometry!

    # With alpha >> 1:
    # - First layer weights shrink -> curvature in W1 direction decreases
    # - Second layer weights grow -> curvature in W2 direction increases DRAMATICALLY
    # - Overall sharpness can be made arbitrarily large

    print("Sharpness is NOT invariant to reparameterization!")
    print("This means raw sharpness cannot directly predict generalization.")
```

### Resolution: What Actually Matters

**Modern Understanding**:

1. **Volume of Basin**: Flat minima occupy larger volume in parameter space
2. **Robustness to Noise**: Flat minima are robust to perturbations along ALL directions
3. **PAC-Bayes Bounds**: Connect flatness to generalization through information theory
4. **Hessian Spectrum**: The full spectrum of eigenvalues matters, not just "sharpness"

```python
def compute_hessian_spectrum_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    num_eigenvalues: int = 20,
    device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """
    Compute Hessian spectrum metrics for sharpness analysis.

    Uses power iteration for top eigenvalues (more tractable than full Hessian).

    Returns:
        metrics: Dictionary with spectral metrics
    """
    from torch.autograd.functional import hvp

    model = model.to(device)
    model.eval()

    # Get total parameters
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)

    def compute_loss():
        total_loss = 0.0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss * inputs.size(0)
            total += inputs.size(0)
        return total_loss / total

    # Power iteration for top eigenvalue
    v = [torch.randn_like(p) for p in params]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    eigenvalues = []

    for _ in range(num_eigenvalues):
        # Hessian-vector product
        loss = compute_loss()
        grads = torch.autograd.grad(loss, params, create_graph=True)

        hv = torch.autograd.grad(
            grads, params, grad_outputs=v, retain_graph=True
        )

        # Rayleigh quotient gives eigenvalue estimate
        eigenvalue = sum((hvi * vi).sum() for hvi, vi in zip(hv, v))
        eigenvalues.append(eigenvalue.item())

        # Update v (power iteration)
        v = list(hv)
        v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
        v = [vi / v_norm for vi in v]

        # Deflation for next eigenvalue (simplified)
        # Full implementation would subtract projection

    return {
        'top_eigenvalue': max(eigenvalues),
        'trace_estimate': sum(eigenvalues),  # Approximate
        'spectral_norm': max(abs(e) for e in eigenvalues),
        'condition_number_estimate': max(eigenvalues) / (min(eigenvalues) + 1e-10)
    }
```

---

## Section 4: Architecture Effects on Loss Landscape

### Skip Connections: The Great Convexifier

From [Li et al. 2018](https://www.cs.umd.edu/~tomg/projects/landscapes/):

> "Skip connections cause a dramatic 'convexification' of the loss landscape."

**Key Findings:**

1. **VGG-56 (no skip)**: Chaotic, highly non-convex surface with many saddle points
2. **ResNet-56 (with skip)**: Smooth, nearly convex basin
3. **DenseNet-121 (elaborate skip)**: Even smoother landscape

### Why Skip Connections Help

**Gradient Flow Theory**:
- Skip connections provide direct gradient paths
- Prevents vanishing gradients in deep networks
- Enables training of very deep networks

**Loss Landscape Theory**:
- Skip connections prevent "shattering" of the loss surface
- Maintain smoothness as depth increases
- Create wider basins of attraction

```python
class ResidualBlock(nn.Module):
    """
    Residual block that helps maintain smooth loss landscapes.

    The identity shortcut provides:
    1. Direct gradient flow (no vanishing gradients)
    2. Loss landscape smoothness
    3. Effective ensemble of shallow networks
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The skip connection is the key to smooth landscapes
        identity = self.shortcut(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection: enables training of 1000+ layer networks
        out += identity
        out = torch.relu(out)

        return out


def analyze_architecture_landscape(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    resolution: int = 51,
    range_val: float = 1.0,
    device: torch.device = torch.device('cuda')
) -> Dict[str, np.ndarray]:
    """
    Analyze loss landscape characteristics for a given architecture.

    Returns:
        results: Dictionary with landscape analysis
    """
    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)

    # Compute 2D surface
    surface = compute_loss_landscape_2d(
        model, dataloader, criterion, alphas, betas, device
    )

    # Compute landscape metrics
    center_loss = surface[resolution//2, resolution//2]

    # Convexity measure: how often is second derivative positive?
    gradient_x = np.gradient(surface, axis=0)
    gradient_y = np.gradient(surface, axis=1)
    hessian_xx = np.gradient(gradient_x, axis=0)
    hessian_yy = np.gradient(gradient_y, axis=1)

    convexity = np.mean((hessian_xx > 0) & (hessian_yy > 0))

    # Smoothness: variance of gradients
    smoothness = 1.0 / (np.var(gradient_x) + np.var(gradient_y) + 1e-10)

    # Basin width: distance to significant loss increase
    threshold = center_loss * 1.5
    basin_mask = surface < threshold
    basin_width = np.sqrt(np.sum(basin_mask) / (resolution ** 2)) * 2 * range_val

    return {
        'surface': surface,
        'center_loss': center_loss,
        'convexity_measure': convexity,
        'smoothness': smoothness,
        'basin_width': basin_width,
        'max_loss': np.max(surface),
        'loss_range': np.max(surface) - np.min(surface)
    }
```

### Width vs Depth Trade-off

**Wider Networks**:
- Smoother loss landscapes
- More parameters but easier optimization
- Better gradient flow per layer

**Deeper Networks**:
- More expressive but chaotic landscapes
- Need skip connections for trainability
- Higher capacity but harder optimization

---

## Section 5: Complete Loss Landscape Visualization Pipeline

### Full Implementation with Visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import copy

class LossLandscapeVisualizer:
    """
    Complete pipeline for neural network loss landscape visualization.

    Based on Li et al. 2018: "Visualizing the Loss Landscape of Neural Nets"

    Features:
    - Filter normalization for fair comparisons
    - 1D and 2D visualization
    - Multiple direction types (random, PCA, Hessian)
    - Contour, surface, and interpolation plots
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.state_dict = copy.deepcopy(model.state_dict())

    def create_filter_normalized_direction(
        self,
        ignore_bn: bool = True
    ) -> dict:
        """Create a filter-normalized random direction."""
        direction = {}

        for name, param in self.state_dict.items():
            # Handle BatchNorm layers
            if ignore_bn and any(bn_key in name for bn_key in
                                ['running_mean', 'running_var',
                                 'num_batches_tracked', 'bias']):
                direction[name] = torch.zeros_like(param)
                continue

            # Random direction
            d = torch.randn_like(param)

            # Filter normalization
            if len(param.shape) >= 2:
                # Normalize each filter
                for i in range(param.shape[0]):
                    param_norm = param[i].norm()
                    d_norm = d[i].norm()
                    if d_norm > 1e-10 and param_norm > 1e-10:
                        d[i] = d[i] * (param_norm / d_norm)
            else:
                # Scalar normalization
                if param.norm() > 1e-10:
                    d = d * (param.norm() / d.norm())

            direction[name] = d

        return direction

    def evaluate_at_point(
        self,
        direction1: dict,
        direction2: Optional[dict],
        alpha: float,
        beta: float = 0.0
    ) -> tuple:
        """Evaluate loss and accuracy at a point in direction space."""
        # Create perturbed parameters
        new_state = {}
        for name in self.state_dict:
            new_state[name] = self.state_dict[name] + alpha * direction1[name]
            if direction2 is not None:
                new_state[name] = new_state[name] + beta * direction2[name]

        # Load into model
        model_copy = copy.deepcopy(self.model)
        model_copy.load_state_dict(new_state)
        model_copy.eval()

        # Evaluate
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model_copy(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        return total_loss / total, 100.0 * correct / total

    def plot_1d_landscape(
        self,
        num_points: int = 51,
        range_val: float = 1.0,
        figsize: tuple = (10, 4)
    ) -> tuple:
        """
        Plot 1D loss and accuracy curves.

        Returns:
            fig, axes: Matplotlib figure and axes
            alphas: x-coordinates
            losses: loss values
            accuracies: accuracy values
        """
        alphas = np.linspace(-range_val, range_val, num_points)
        direction = self.create_filter_normalized_direction()

        losses = []
        accuracies = []

        for i, alpha in enumerate(alphas):
            loss, acc = self.evaluate_at_point(direction, None, alpha)
            losses.append(loss)
            accuracies.append(acc)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_points}")

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(alphas, losses, 'b-', linewidth=2)
        axes[0].set_xlabel('Distance along direction')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Landscape (1D)')
        axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(alphas, accuracies, 'g-', linewidth=2)
        axes[1].set_xlabel('Distance along direction')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Landscape (1D)')
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig, axes, alphas, np.array(losses), np.array(accuracies)

    def plot_2d_contour(
        self,
        num_points: int = 31,
        range_val: float = 1.0,
        figsize: tuple = (8, 6),
        levels: int = 30,
        log_scale: bool = True
    ) -> tuple:
        """
        Plot 2D loss contour.

        Returns:
            fig, ax: Matplotlib figure and axis
            surface: 2D loss array
        """
        alphas = np.linspace(-range_val, range_val, num_points)
        betas = np.linspace(-range_val, range_val, num_points)

        direction1 = self.create_filter_normalized_direction()
        direction2 = self.create_filter_normalized_direction()

        surface = np.zeros((num_points, num_points))

        total_evals = num_points * num_points
        eval_count = 0

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                loss, _ = self.evaluate_at_point(
                    direction1, direction2, alpha, beta
                )
                surface[i, j] = loss
                eval_count += 1

            if (i + 1) % 5 == 0:
                print(f"Row {i+1}/{num_points} complete "
                      f"({100*eval_count/total_evals:.1f}%)")

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)

        X, Y = np.meshgrid(alphas, betas)

        if log_scale:
            plot_surface = np.log10(surface + 1e-6)
            label = 'Log10(Loss)'
        else:
            plot_surface = surface
            label = 'Loss'

        contour = ax.contourf(X, Y, plot_surface.T, levels=levels, cmap='viridis')
        plt.colorbar(contour, ax=ax, label=label)

        ax.plot(0, 0, 'r*', markersize=15, label='Trained minimum')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_title('Loss Landscape Contour (2D)')
        ax.legend()

        return fig, ax, surface

    def plot_3d_surface(
        self,
        num_points: int = 31,
        range_val: float = 1.0,
        figsize: tuple = (12, 8),
        log_scale: bool = True,
        elev: int = 30,
        azim: int = 45
    ) -> tuple:
        """
        Plot 3D loss surface.

        Returns:
            fig, ax: Matplotlib figure and 3D axis
            surface: 2D loss array
        """
        # Compute surface (can reuse from contour)
        alphas = np.linspace(-range_val, range_val, num_points)
        betas = np.linspace(-range_val, range_val, num_points)

        direction1 = self.create_filter_normalized_direction()
        direction2 = self.create_filter_normalized_direction()

        surface = np.zeros((num_points, num_points))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                loss, _ = self.evaluate_at_point(
                    direction1, direction2, alpha, beta
                )
                surface[i, j] = loss

            if (i + 1) % 5 == 0:
                print(f"Row {i+1}/{num_points} complete")

        # 3D Plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(alphas, betas)

        if log_scale:
            Z = np.log10(surface.T + 1e-6)
            zlabel = 'Log10(Loss)'
        else:
            Z = surface.T
            zlabel = 'Loss'

        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                               edgecolor='none', alpha=0.8)

        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_zlabel(zlabel)
        ax.set_title('Loss Landscape Surface (3D)')
        ax.view_init(elev=elev, azim=azim)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)

        return fig, ax, surface

    def plot_linear_interpolation(
        self,
        model2: nn.Module,
        num_points: int = 51,
        figsize: tuple = (10, 4)
    ) -> tuple:
        """
        Plot loss along linear interpolation between two minima.

        Used to study mode connectivity and basin structure.

        Args:
            model2: Second trained model to interpolate to

        Returns:
            fig, axes, alphas, losses, accuracies
        """
        state1 = self.state_dict
        state2 = model2.state_dict()

        alphas = np.linspace(0, 1, num_points)
        losses = []
        accuracies = []

        for alpha in alphas:
            # Linear interpolation
            interp_state = {}
            for name in state1:
                interp_state[name] = (1 - alpha) * state1[name] + alpha * state2[name]

            model_copy = copy.deepcopy(self.model)
            model_copy.load_state_dict(interp_state)
            model_copy.eval()

            # Evaluate
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in self.dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model_copy(inputs)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

            losses.append(total_loss / total)
            accuracies.append(100.0 * correct / total)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(alphas, losses, 'b-', linewidth=2)
        axes[0].set_xlabel('Interpolation (0=Model1, 1=Model2)')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Linear Interpolation: Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(alphas, accuracies, 'g-', linewidth=2)
        axes[1].set_xlabel('Interpolation (0=Model1, 1=Model2)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Linear Interpolation: Accuracy')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig, axes, alphas, np.array(losses), np.array(accuracies)


# Example usage
def visualize_loss_landscape_example():
    """
    Complete example of loss landscape visualization.
    """
    import torchvision
    import torchvision.transforms as transforms

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2
    )

    # Load pretrained model (example with ResNet18)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    # Create visualizer
    criterion = nn.CrossEntropyLoss()
    visualizer = LossLandscapeVisualizer(
        model, testloader, criterion, device
    )

    # Generate visualizations
    print("Generating 1D landscape...")
    fig1, _, alphas, losses, accs = visualizer.plot_1d_landscape(
        num_points=41, range_val=0.5
    )
    plt.savefig('loss_landscape_1d.png', dpi=150)

    print("\nGenerating 2D contour...")
    fig2, _, surface = visualizer.plot_2d_contour(
        num_points=21, range_val=0.5
    )
    plt.savefig('loss_landscape_2d_contour.png', dpi=150)

    print("\nGenerating 3D surface...")
    fig3, _, _ = visualizer.plot_3d_surface(
        num_points=21, range_val=0.5
    )
    plt.savefig('loss_landscape_3d.png', dpi=150)

    print("\nVisualization complete!")

    return visualizer


if __name__ == "__main__":
    visualize_loss_landscape_example()
```

---

## Section 6: Performance Optimization

### Computational Considerations

**Memory Requirements:**
- Each evaluation requires full model copy
- For 50x50 grid: 2500 forward passes
- GPU memory: ~2-4GB for typical CIFAR models

**Time Optimization Strategies:**

1. **Parallel Evaluation**: Use multiple GPUs
2. **Reduced Dataset**: Subsample for faster evaluation
3. **Progressive Resolution**: Start coarse, refine interesting regions
4. **Caching**: Store directions and reuse

```python
def optimized_landscape_computation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    resolution: int = 51,
    range_val: float = 1.0,
    num_workers: int = 4,
    subset_fraction: float = 0.1  # Use 10% of data for speed
) -> np.ndarray:
    """
    Optimized loss landscape computation with multiple speedups.

    Optimizations:
    - Dataset subsampling
    - Batch evaluation
    - Progress tracking

    Time estimate for CIFAR-10:
    - Full dataset, 51x51: ~2 hours on single GPU
    - 10% subset, 51x51: ~12 minutes
    - 10% subset, 31x31: ~4 minutes
    """
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Subsample dataset
    dataset = dataloader.dataset
    subset_size = int(len(dataset) * subset_fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    fast_loader = torch.utils.data.DataLoader(
        subset, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Setup
    dir_gen = FilterNormalizedDirection(model)
    d1 = dir_gen.create_random_direction()
    d2 = dir_gen.create_random_direction()

    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)

    surface = np.zeros((resolution, resolution))

    start_time = time.time()
    total_points = resolution * resolution

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb model
            new_state = {}
            for name in dir_gen.state_dict:
                new_state[name] = (dir_gen.state_dict[name] +
                                  alpha * d1[name] +
                                  beta * d2[name])

            model.load_state_dict(new_state)
            model.eval()

            # Fast evaluation
            total_loss = 0.0
            total = 0

            with torch.no_grad():
                for inputs, targets in fast_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    total_loss += loss.item() * inputs.size(0)
                    total += inputs.size(0)

            surface[i, j] = total_loss / total

        # Progress
        elapsed = time.time() - start_time
        points_done = (i + 1) * resolution
        eta = elapsed / points_done * (total_points - points_done)
        print(f"Row {i+1}/{resolution} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # Restore original parameters
    model.load_state_dict(dir_gen.state_dict)

    return surface
```

### GPU Memory Management

```python
def memory_efficient_evaluation(model, dataloader, criterion, device):
    """
    Memory-efficient loss evaluation with gradient checkpointing.
    """
    model.eval()

    total_loss = 0.0
    total = 0

    # Clear cache before evaluation
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Use automatic mixed precision for memory savings
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total += targets.size(0)

            # Free memory
            del outputs, loss

    return total_loss / total
```

---

## Section 7: TRAIN STATION - Loss Landscape = Free Energy Landscape = Affordance Space

### The Deep Unification

**TRAIN STATION**: Loss landscapes, free energy landscapes, and affordance spaces are topologically equivalent concepts representing navigation through possibility spaces.

### Loss Landscape = Free Energy Landscape

In the Free Energy Principle framework (Friston):

```
F = E_q[log q(z) - log p(x,z)]
  = KL[q(z) || p(z|x)] - log p(x)
```

**The connection:**
- **Neural Network Loss**: `L(theta) = -log p(data | theta)`
- **Variational Free Energy**: `F = -log p(x) + KL[q || p]`

Both are quantities to minimize through gradient descent on a high-dimensional landscape!

```python
def free_energy_loss_equivalence():
    """
    Demonstrate that neural network loss IS a free energy functional.

    Cross-entropy loss = negative log likelihood = surprise = free energy
    """
    # Neural network forward pass
    # p(y|x, theta) = softmax(f_theta(x))

    # Cross-entropy loss
    # L = -sum_i y_i * log(p_i)
    # L = -log p(y_true | x, theta)

    # This IS the surprisal (negative log probability)
    # Minimizing loss = minimizing surprise = minimizing free energy

    # The loss landscape is literally a free energy landscape!
    pass
```

### Loss Landscape = Affordance Space

**Gibson's Affordances**: Possibilities for action in an environment.

**Neural Network Weights**: Define what actions (predictions) are possible.

**The topology is the same:**

| Loss Landscape | Affordance Space |
|----------------|------------------|
| Minimum = trained model | Attractor = stable action |
| Gradient = update direction | Gradient = action tendency |
| Basin = solution family | Basin = action category |
| Saddle = transition | Saddle = action boundary |

### Unified View: Navigation Through Possibility Space

```python
class UnifiedLandscapeTheory:
    """
    Loss landscape = Free energy landscape = Affordance landscape

    All three describe navigation through possibility spaces:
    - Loss: Parameter space of neural network
    - Free energy: Belief space of agent
    - Affordance: Action space of embodied agent

    TRAIN STATION: They meet at the concept of GRADIENT DESCENT
    through a HIGH-DIMENSIONAL SPACE toward ATTRACTORS (minima).
    """

    def __init__(self):
        # The deep unity
        self.equivalences = {
            'minimum': {
                'loss': 'Trained model parameters',
                'free_energy': 'Stable belief state',
                'affordance': 'Reliable action pattern'
            },
            'gradient': {
                'loss': 'Direction to reduce loss',
                'free_energy': 'Direction to reduce surprise',
                'affordance': 'Direction toward goal'
            },
            'basin': {
                'loss': 'Family of similar solutions',
                'free_energy': 'Region of stable beliefs',
                'affordance': 'Category of related actions'
            },
            'saddle_point': {
                'loss': 'Optimization challenge',
                'free_energy': 'Belief uncertainty',
                'affordance': 'Action ambiguity'
            },
            'curvature': {
                'loss': 'Sharpness -> generalization',
                'free_energy': 'Precision -> confidence',
                'affordance': 'Specificity -> skill'
            }
        }

    def why_this_matters(self) -> str:
        """
        Why this unification is profound.
        """
        return """
        1. OPTIMIZATION = INFERENCE = ACTION
           - Training a neural network
           - Updating beliefs about the world
           - Learning to act skillfully
           All follow gradient descent through possibility space!

        2. ARCHITECTURE = EMBODIMENT = UMWELT
           - Skip connections shape loss landscape
           - Body structure shapes free energy landscape
           - Environment shapes affordance landscape
           The 'container' determines the topology!

        3. GENERALIZATION = ROBUSTNESS = ADAPTABILITY
           - Flat minima generalize to new data
           - Low free energy states are robust to perturbation
           - Good affordances transfer to new situations
           Basin width predicts success!

        4. VISUALIZATION = UNDERSTANDING
           - See loss landscape -> understand training
           - See free energy landscape -> understand cognition
           - See affordance landscape -> understand behavior
           Topology reveals dynamics!
        """
```

---

## Section 8: ARR-COC-0-1 Connection - Relevance Landscape Navigation

### Relevance as Loss Landscape Navigation

In ARR-COC (Adaptive Relevance Realization), computing relevance for vision-language models involves navigating a relevance landscape:

```python
class RelevanceLandscapeNavigator:
    """
    ARR-COC relevance computation as landscape navigation.

    The 'relevance landscape' has:
    - Minima: Optimal token allocations
    - Gradients: Directions to improve allocation
    - Basins: Families of similar allocations
    - Sharpness: Sensitivity to perturbations

    Just like loss landscape visualization reveals optimization dynamics,
    visualizing the relevance landscape reveals allocation dynamics!
    """

    def __init__(self, vlm_model, image, prompt):
        self.model = vlm_model
        self.image = image
        self.prompt = prompt

    def compute_relevance_surface(
        self,
        token_range: tuple = (128, 4096),
        resolution: int = 50
    ) -> np.ndarray:
        """
        Compute relevance-performance surface.

        X-axis: Number of visual tokens
        Y-axis: Relevance threshold
        Z-axis: Task performance (or proxy)

        This is analogous to loss landscape visualization!
        """
        tokens = np.linspace(token_range[0], token_range[1], resolution)
        thresholds = np.linspace(0.0, 1.0, resolution)

        surface = np.zeros((resolution, resolution))

        for i, num_tokens in enumerate(tokens):
            for j, threshold in enumerate(thresholds):
                # Apply token allocation
                performance = self.evaluate_allocation(
                    int(num_tokens), threshold
                )
                surface[i, j] = performance

        return surface

    def evaluate_allocation(self, num_tokens: int,
                           threshold: float) -> float:
        """
        Evaluate performance with given allocation parameters.
        """
        # This would call the actual VLM with modified allocation
        # Returns a performance metric (accuracy, perplexity, etc.)
        pass

    def find_optimal_allocation(self) -> tuple:
        """
        Navigate the relevance landscape to find optimal allocation.

        Uses gradient-based optimization - just like training!
        """
        # Start from default allocation
        # Follow gradients toward better performance
        # Stop at a minimum (optimal allocation)
        pass

    def analyze_allocation_robustness(self, allocation: tuple) -> dict:
        """
        Analyze sharpness of the allocation minimum.

        Flat minima in relevance space = robust allocations
        Sharp minima = allocations sensitive to perturbations

        This directly mirrors loss landscape analysis!
        """
        # Compute local curvature around allocation
        # High curvature = sharp = fragile allocation
        # Low curvature = flat = robust allocation
        pass


def relevance_landscape_insights():
    """
    Key insights from loss landscape theory for ARR-COC.

    1. ALLOCATION BASINS
       Different prompts/images may have different "basins"
       of good allocations. Understanding basin structure
       helps design adaptive allocation.

    2. SHARP VS FLAT ALLOCATIONS
       Allocations that work across many prompts (flat)
       vs allocations that work for specific prompts (sharp).
       Trade-off between specialization and generalization.

    3. MODE CONNECTIVITY
       Can we smoothly interpolate between good allocations?
       If yes, can learn continuous allocation policies.
       If no, may need discrete switching.

    4. ARCHITECTURAL EFFECTS
       Just as skip connections smooth loss landscapes,
       certain VLM architectures may create smoother
       relevance landscapes (easier to optimize allocation).
    """
    return {
        'basin_structure': 'Understand families of good allocations',
        'flatness': 'Prefer robust allocations over brittle ones',
        'connectivity': 'Enable smooth adaptation between allocations',
        'architecture': 'Design VLMs for smooth relevance landscapes'
    }
```

### Practical Applications for ARR-COC

1. **Allocation Visualization**: Plot relevance-performance surfaces to understand allocation dynamics
2. **Robustness Analysis**: Identify allocations with flat basins (robust to perturbations)
3. **Adaptive Navigation**: Use gradient information to dynamically adjust allocation
4. **Architecture Search**: Find VLM architectures with smooth relevance landscapes

---

## Sources

### Primary Paper

- [Li et al. 2018 - Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) - NeurIPS 2018 (2769+ citations)

### Implementation

- [GitHub: tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape) - Official PyTorch implementation
- [UMD Project Page](https://www.cs.umd.edu/~tomg/projects/landscapes/) - Additional visualizations and resources

### Related Work

- [Goodfellow et al. 2015 - Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/abs/1412.6544) - Linear interpolation method
- [Keskar et al. 2017 - On Large-Batch Training: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) - Sharp vs flat minima and batch size
- [Dinh et al. 2017 - Sharp Minima Can Generalize for Deep Nets](https://arxiv.org/abs/1703.04933) - Reparameterization invariance counterargument
- [He et al. 2019 - Asymmetric Valleys: Beyond Sharp and Flat Local Minima](https://papers.nips.cc/paper/8524-asymmetric-valleys-beyond-sharp-and-flat-local-minima.pdf) - NeurIPS 2019
- [Baldassi et al. 2020 - Shaping the Learning Landscape in Neural Networks](https://www.pnas.org/doi/10.1073/pnas.1908636117) - PNAS

### Interactive Tools

- [3D Interactive Visualizer](http://www.telesens.co/loss-landscape-viz/viewer.html) - by Ankur Mohan
- [losslandscape.com](https://losslandscape.com/) - Artistic renderings

### Blog Posts

- [The Generalization Mystery: Sharp vs Flat Minima](https://www.inference.vc/sharp-vs-flat-minima-are-still-a-mystery-to-me/) - Ferenc Huszar's analysis

---

## Key Takeaways

1. **Filter Normalization is Essential**: Without it, loss landscape comparisons are meaningless due to scale invariance

2. **Architecture Shapes Landscape**: Skip connections dramatically smooth the loss surface, enabling training of deep networks

3. **Sharp/Flat Debate Continues**: Flatness correlates with generalization but isn't invariant to reparameterization - the full Hessian spectrum matters

4. **TRAIN STATION Unification**: Loss landscapes = Free energy landscapes = Affordance spaces - all are possibility spaces navigated by gradient descent

5. **Practical Tool**: Loss landscape visualization helps understand why certain architectures train better and diagnose optimization problems

6. **ARR-COC Application**: Relevance landscapes can be visualized similarly to understand and optimize token allocation dynamics

---

*The loss landscape is not just a mathematical abstraction - it's the terrain that determines whether training succeeds or fails, whether models generalize or overfit, and whether architectures scale or collapse. Visualizing this terrain transforms deep learning from alchemy to engineering.*
