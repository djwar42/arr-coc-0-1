# Morphogenetic Field Learning: Neural Fields as Implicit Representations

## Overview

Neural fields (also known as implicit neural representations or coordinate-based networks) represent signals as continuous functions parameterized by neural networks. This approach has deep connections to morphogenetic fields in developmental biology - both encode spatial information implicitly and guide pattern formation through gradient-based processes.

**The Key Insight**: A morphogenetic field in biology guides cell differentiation through concentration gradients of morphogens. A neural field guides rendering/reconstruction through learned mappings from coordinates to values. Both are implicit, continuous, and gradient-based!

---

## Section 1: Neural Fields (NeRF-Style)

### What Are Neural Fields?

Neural fields represent signals (images, 3D shapes, audio, video) as continuous functions:

```
f: R^n -> R^m
```

Where:
- Input: Spatial/temporal coordinates (x, y, z, t)
- Output: Signal values (RGB color, density, audio amplitude)
- Function: Neural network (typically MLP)

From [Neural Fields in Visual Computing and Beyond](https://arxiv.org/pdf/2111.11426.pdf) (Xie et al., 2022, cited by Computer Graphics Forum):
> "Neural fields... are coordinate-based neural networks, neural implicits, or neural implicit representations."

### NeRF: Neural Radiance Fields

From [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf) (Mildenhall et al., 2020):

The core idea maps 5D coordinates to color and density:

```
F: (x, y, z, theta, phi) -> (r, g, b, sigma)
```

Where:
- (x, y, z): 3D position
- (theta, phi): Viewing direction
- (r, g, b): Color
- sigma: Volume density

### Volume Rendering Equation

The key rendering equation integrates along camera rays:

```python
C(r) = integral[t_n to t_f] T(t) * sigma(r(t)) * c(r(t), d) dt

where T(t) = exp(-integral[t_n to t] sigma(r(s)) ds)
```

- C(r): Expected color along ray r
- T(t): Accumulated transmittance (how much light penetrates)
- sigma: Volume density
- c: Color

### Positional Encoding

From [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739) (Tancik et al., 2020):

Standard MLPs struggle with high-frequency details. Solution: encode coordinates with Fourier features:

```python
gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
            sin(2^1 * pi * p), cos(2^1 * pi * p),
            ...,
            sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]
```

This maps low-dimensional coordinates to high-dimensional space, enabling networks to learn high-frequency functions.

---

## Section 2: SIREN - Sinusoidal Implicit Representations

### The Problem with ReLU Networks

From [SIREN: Implicit Neural Representations with Periodic Activation Functions](https://www.vincentsitzmann.com/siren/) (Sitzmann et al., NeurIPS 2020, cited 3691 times):

> "Current network architectures for such implicit neural representations are incapable of modeling signals with fine detail, and fail to represent a signal's spatial and temporal derivatives."

ReLU networks:
- Produce piecewise linear outputs
- Derivatives are piecewise constant
- Second derivatives are zero almost everywhere
- Cannot accurately represent gradients of signals

### The SIREN Solution

Replace ReLU with sinusoidal activations:

```python
phi(x) = sin(omega_0 * W * x + b)
```

Where omega_0 is a hyperparameter controlling frequency (typically 30).

**Key Benefits**:
- Derivatives are also sinusoids (still smooth!)
- Can represent signals AND their derivatives
- 10+ dB higher PSNR than baselines
- Converges significantly faster

### Why Periodic Activations Work

From the SIREN paper:

1. **Spectral bias**: Neural networks naturally learn low frequencies first. Sinusoids inject high-frequency capacity directly into the architecture.

2. **Derivative preservation**: For f(x) = sin(omega * x), derivatives are:
   - f'(x) = omega * cos(omega * x)
   - f''(x) = -omega^2 * sin(omega * x)

   All derivatives remain smooth and well-behaved!

3. **Solving PDEs**: Since derivatives are accurate, SIRENs can solve boundary value problems like Eikonal, Poisson, Helmholtz, and wave equations.

### SIREN Initialization

From the paper's theoretical analysis:

```python
# First layer: uniform distribution
W_1 ~ U(-1/input_dim, 1/input_dim)

# Hidden layers: scaled for variance preservation
W_i ~ U(-sqrt(6/(input_dim * omega_0^2)), sqrt(6/(input_dim * omega_0^2)))
```

This initialization ensures:
- Pre-activations are normally distributed
- Activations are arcsine distributed
- Variance is preserved across layers

---

## Section 3: Morphogenetic Gradients in Neural Fields

### The Morphogenesis Connection

In developmental biology, morphogenetic fields are regions where cells receive positional information through concentration gradients of signaling molecules (morphogens). Cells differentiate based on their position in this gradient field.

**The parallel**:
- Morphogen concentration -> Network output value
- Spatial position -> Input coordinates
- Gradient of morphogen -> Gradient of network output
- Cell fate determination -> Pixel/voxel value assignment

### Gradient-Based Pattern Formation

Both systems use gradients for:

1. **Spatial organization**: Higher concentration/value at source, decreasing with distance
2. **Threshold-based decisions**: Different outputs based on local gradient values
3. **Self-organization**: Local rules produce global patterns
4. **Regeneration**: The field contains all information needed to reconstruct

### Neural Field Gradients

For a neural field f(x), we can compute:

```python
# Gradient (first derivative)
grad_f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)

# Laplacian (second derivative)
laplacian = sum([torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g),
                                      create_graph=True)[0] for g in grad_f])
```

These gradients are crucial for:
- **Eikonal equation**: |grad(f)| = 1 for signed distance functions
- **Poisson equation**: laplacian(f) = g for image reconstruction from gradients
- **Wave equation**: d^2f/dt^2 = c^2 * laplacian(f) for wave propagation

### Morphogenetic Field as Energy Landscape

Like free energy landscapes in active inference, morphogenetic fields can be viewed as:

```python
# Position x seeks to minimize "developmental potential"
E(x) = field_value(x)

# "Force" on development = negative gradient
F = -grad(E)

# Cells follow the morphogenetic gradient
dx/dt = -grad(E)
```

---

## Section 4: PyTorch Implementation - Complete Neural Field System

### 4.1 Basic SIREN Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SineLayer(nn.Module):
    """
    Sinusoidal activation layer for SIREN networks.

    From Sitzmann et al., NeurIPS 2020:
    - Uses sin activation with omega_0 scaling
    - Special initialization for variance preservation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """SIREN-specific initialization for stable training."""
        with torch.no_grad():
            if self.is_first:
                # First layer: larger range
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                # Hidden layers: scaled by omega_0
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Complete SIREN network for implicit neural representations.

    Maps coordinates to signal values with accurate gradient representation.
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        outermost_linear: bool = True
    ):
        super().__init__()

        # Build network
        layers = []

        # First layer
        layers.append(SineLayer(in_features, hidden_features,
                               omega_0=omega_0, is_first=True))

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                   omega_0=omega_hidden))

        # Output layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            # Initialize output layer for bounded outputs
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / omega_hidden
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                   omega_0=omega_hidden))

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., in_features] coordinate tensor

        Returns:
            [..., out_features] signal values
        """
        return self.net(coords)

    def forward_with_gradients(
        self,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also computes spatial gradients.

        Essential for:
        - Eikonal equation (SDF)
        - Poisson equation (image from gradients)
        - Normal computation
        """
        coords = coords.requires_grad_(True)
        output = self.forward(coords)

        # Compute gradient w.r.t. input coordinates
        grad = torch.autograd.grad(
            output,
            coords,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

        return output, grad


class PositionalEncoding(nn.Module):
    """
    Fourier feature encoding for coordinate inputs.

    Maps low-dimensional coordinates to high-dimensional space
    to help networks learn high-frequency functions.

    From Tancik et al., "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self,
        in_features: int = 3,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Generate frequency bands
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)

        self.register_buffer('freq_bands', freq_bands)

        # Output dimension
        self.out_features = in_features * num_frequencies * 2
        if include_input:
            self.out_features += in_features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., in_features] coordinates

        Returns:
            [..., out_features] encoded features
        """
        # Reshape for broadcasting: [..., in_features, 1]
        coords_expanded = coords.unsqueeze(-1)

        # Scale by frequencies: [..., in_features, num_frequencies]
        scaled = coords_expanded * self.freq_bands * np.pi

        # Apply sin and cos
        sin_features = torch.sin(scaled)
        cos_features = torch.cos(scaled)

        # Concatenate: [..., in_features * num_frequencies * 2]
        features = torch.cat([sin_features, cos_features], dim=-1)
        features = features.view(*coords.shape[:-1], -1)

        if self.include_input:
            features = torch.cat([coords, features], dim=-1)

        return features
```

### 4.2 NeRF-Style Volume Rendering

```python
class NeuralRadianceField(nn.Module):
    """
    NeRF-style neural radiance field with volume rendering.

    Maps 5D coordinates (position + direction) to color and density.
    """

    def __init__(
        self,
        pos_dim: int = 3,
        dir_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_pos_freq: int = 10,
        num_dir_freq: int = 4,
        skip_connect: int = 4
    ):
        super().__init__()

        # Positional encodings
        self.pos_encoder = PositionalEncoding(pos_dim, num_pos_freq)
        self.dir_encoder = PositionalEncoding(dir_dim, num_dir_freq)

        pos_encoded_dim = self.pos_encoder.out_features
        dir_encoded_dim = self.dir_encoder.out_features

        # Main network (processes position)
        self.layers_before_skip = nn.ModuleList()
        self.layers_after_skip = nn.ModuleList()

        # Before skip connection
        self.layers_before_skip.append(nn.Linear(pos_encoded_dim, hidden_dim))
        for i in range(1, skip_connect):
            self.layers_before_skip.append(nn.Linear(hidden_dim, hidden_dim))

        # After skip connection
        self.layers_after_skip.append(nn.Linear(hidden_dim + pos_encoded_dim, hidden_dim))
        for i in range(skip_connect + 1, num_layers - 1):
            self.layers_after_skip.append(nn.Linear(hidden_dim, hidden_dim))

        # Density output (sigma)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

        # Color branch (incorporates viewing direction)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.color_layer = nn.Sequential(
            nn.Linear(hidden_dim + dir_encoded_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [..., 3] 3D positions
            directions: [..., 3] viewing directions

        Returns:
            colors: [..., 3] RGB colors
            densities: [..., 1] volume densities
        """
        # Encode inputs
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)

        # Process through network
        x = pos_encoded

        # Before skip
        for layer in self.layers_before_skip:
            x = F.relu(layer(x))

        # Skip connection
        x = torch.cat([x, pos_encoded], dim=-1)

        # After skip
        for layer in self.layers_after_skip:
            x = F.relu(layer(x))

        # Density output (non-negative)
        sigma = F.relu(self.sigma_layer(x))

        # Color output (direction-dependent)
        features = self.feature_layer(x)
        color_input = torch.cat([features, dir_encoded], dim=-1)
        rgb = self.color_layer(color_input)

        return rgb, sigma


def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    white_background: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable volume rendering along rays.

    Implements the classic NeRF volume rendering equation:
    C(r) = sum_i T_i * alpha_i * c_i

    where:
    - T_i = prod_{j<i} (1 - alpha_j)  [transmittance]
    - alpha_i = 1 - exp(-sigma_i * delta_i)  [opacity]

    Args:
        rgb: [batch, num_samples, 3] colors at sample points
        sigma: [batch, num_samples, 1] densities
        z_vals: [batch, num_samples] depth values along rays
        white_background: whether to use white background

    Returns:
        rgb_map: [batch, 3] rendered colors
        depth_map: [batch] expected depth
        weights: [batch, num_samples] per-sample weights
    """
    # Compute distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Last distance is infinite (or large value)
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)

    # Convert density to alpha (opacity)
    # alpha = 1 - exp(-sigma * dist)
    sigma = sigma.squeeze(-1)
    alpha = 1.0 - torch.exp(-sigma * dists)

    # Compute transmittance
    # T_i = prod_{j<i} (1 - alpha_j)
    # Use exclusive cumulative product
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]

    # Compute weights
    weights = alpha * transmittance

    # Render color
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)

    # Render depth
    depth_map = (weights * z_vals).sum(dim=-1)

    # Background
    if white_background:
        acc_map = weights.sum(dim=-1)
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, weights
```

### 4.3 Morphogenetic Field Implementation

```python
class MorphogeneticField(nn.Module):
    """
    Neural field that mimics morphogenetic field behavior.

    Models pattern formation through learned concentration gradients,
    similar to how morphogens guide development in biology.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        omega_0: float = 30.0,
        num_morphogens: int = 1
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.num_morphogens = num_morphogens

        # Use SIREN for smooth gradients
        self.concentration_field = SIREN(
            in_features=spatial_dim,
            out_features=num_morphogens,
            hidden_features=hidden_dim,
            hidden_layers=num_layers,
            omega_0=omega_0
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Get morphogen concentration at coordinates."""
        return self.concentration_field(coords)

    def get_gradient(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute morphogenetic gradient (like diffusion gradient in biology).

        This is the "force" that guides pattern formation.
        """
        coords = coords.requires_grad_(True)
        concentration = self.forward(coords)

        # Compute spatial gradient for each morphogen
        gradients = []
        for i in range(self.num_morphogens):
            grad = torch.autograd.grad(
                concentration[..., i].sum(),
                coords,
                create_graph=True
            )[0]
            gradients.append(grad)

        return torch.stack(gradients, dim=-1)  # [..., spatial_dim, num_morphogens]

    def get_laplacian(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian (diffusion term in morphogen dynamics).

        For reaction-diffusion systems: dc/dt = D * laplacian(c) + R(c)
        """
        coords = coords.requires_grad_(True)
        concentration = self.forward(coords)

        # First derivative
        grad = torch.autograd.grad(
            concentration.sum(),
            coords,
            create_graph=True
        )[0]

        # Second derivative (Laplacian)
        laplacian = 0
        for i in range(self.spatial_dim):
            grad_i = grad[..., i].sum()
            grad_ii = torch.autograd.grad(
                grad_i,
                coords,
                create_graph=True
            )[0][..., i]
            laplacian = laplacian + grad_ii

        return laplacian

    def cell_fate_from_field(
        self,
        coords: torch.Tensor,
        thresholds: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine cell fate based on morphogen concentration.

        Mimics threshold-based cell differentiation in French flag model.

        Args:
            coords: [..., spatial_dim] positions
            thresholds: [num_fates] concentration thresholds

        Returns:
            fates: [...] cell fate indices
        """
        concentration = self.forward(coords).squeeze(-1)

        # Determine fate based on thresholds
        fates = torch.zeros_like(concentration, dtype=torch.long)
        for i, thresh in enumerate(thresholds):
            fates = torch.where(concentration > thresh,
                               torch.full_like(fates, i + 1),
                               fates)

        return fates


class ReactionDiffusionField(MorphogeneticField):
    """
    Neural field that learns reaction-diffusion dynamics.

    Combines:
    - Spatial representation (SIREN)
    - Temporal dynamics (reaction-diffusion)
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_morphogens: int = 2,  # e.g., activator + inhibitor
        diffusion_coeffs: Optional[torch.Tensor] = None
    ):
        super().__init__(
            spatial_dim=spatial_dim + 1,  # +1 for time
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_morphogens=num_morphogens
        )

        if diffusion_coeffs is None:
            diffusion_coeffs = torch.ones(num_morphogens)
        self.register_buffer('diffusion_coeffs', diffusion_coeffs)

    def forward_spacetime(
        self,
        spatial_coords: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Get concentration at spatial position and time.

        Args:
            spatial_coords: [..., spatial_dim] positions
            time: [..., 1] or scalar time value

        Returns:
            concentration: [..., num_morphogens]
        """
        if time.dim() == 0:
            time = time.expand(*spatial_coords.shape[:-1], 1)

        spacetime = torch.cat([spatial_coords, time], dim=-1)
        return self.forward(spacetime)
```

### 4.4 Training Utilities

```python
def train_neural_field_image(
    model: SIREN,
    image: torch.Tensor,
    num_iterations: int = 5000,
    lr: float = 1e-4,
    batch_size: int = 8192,
    device: str = 'cuda'
) -> list:
    """
    Train a neural field to represent an image.

    Args:
        model: SIREN model
        image: [H, W, C] image tensor
        num_iterations: training iterations
        lr: learning rate
        batch_size: pixels per batch
        device: computation device

    Returns:
        losses: list of loss values
    """
    model = model.to(device)
    image = image.to(device)

    H, W, C = image.shape

    # Create coordinate grid
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    coords = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    coords = coords.reshape(-1, 2)  # [H*W, 2]

    # Flatten image
    pixels = image.reshape(-1, C)  # [H*W, C]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for i in range(num_iterations):
        # Sample random batch
        idx = torch.randint(0, coords.shape[0], (batch_size,), device=device)
        batch_coords = coords[idx]
        batch_pixels = pixels[idx]

        # Forward pass
        pred = model(batch_coords)

        # MSE loss
        loss = F.mse_loss(pred, batch_pixels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 500 == 0:
            psnr = -10 * np.log10(loss.item())
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}, PSNR: {psnr:.2f} dB")

    return losses


def train_sdf_from_pointcloud(
    model: SIREN,
    points: torch.Tensor,
    normals: torch.Tensor,
    num_iterations: int = 10000,
    lr: float = 1e-4,
    num_samples: int = 8192,
    device: str = 'cuda'
) -> list:
    """
    Train a neural field as a signed distance function from point cloud.

    Uses Eikonal loss: |grad(f)| = 1

    Args:
        model: SIREN model (out_features=1)
        points: [N, 3] surface points
        normals: [N, 3] surface normals
        num_iterations: training iterations
        lr: learning rate
        num_samples: samples per batch
        device: computation device

    Returns:
        losses: list of loss values
    """
    model = model.to(device)
    points = points.to(device)
    normals = normals.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for i in range(num_iterations):
        # Sample surface points
        idx = torch.randint(0, points.shape[0], (num_samples,), device=device)
        batch_points = points[idx]
        batch_normals = normals[idx]

        # Sample random off-surface points
        off_surface = torch.randn(num_samples, 3, device=device)

        # Compute SDF and gradients at surface
        batch_points.requires_grad_(True)
        sdf_surface = model(batch_points)
        grad_surface = torch.autograd.grad(
            sdf_surface.sum(),
            batch_points,
            create_graph=True
        )[0]

        # Compute gradients at random points
        off_surface.requires_grad_(True)
        sdf_off = model(off_surface)
        grad_off = torch.autograd.grad(
            sdf_off.sum(),
            off_surface,
            create_graph=True
        )[0]

        # Loss components
        # 1. Surface constraint: SDF = 0 at surface
        surface_loss = sdf_surface.abs().mean()

        # 2. Normal constraint: gradient = normal at surface
        normal_loss = (1 - F.cosine_similarity(grad_surface, batch_normals, dim=-1)).mean()

        # 3. Eikonal constraint: |grad| = 1 everywhere
        eikonal_loss = ((grad_surface.norm(dim=-1) - 1) ** 2).mean()
        eikonal_loss += ((grad_off.norm(dim=-1) - 1) ** 2).mean()

        # Total loss
        loss = surface_loss + 0.1 * normal_loss + 0.1 * eikonal_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 1000 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

    return losses
```

---

## Section 5: TRAIN STATION - Neural Field = NeRF = Implicit = Morphogenetic

### The Grand Unification

**ALL of these are the SAME computational pattern**:

| Concept | Input | Output | Gradient Meaning |
|---------|-------|--------|-----------------|
| Neural Field | coordinates | signal value | spatial derivative |
| NeRF | (x,y,z,dir) | (color, density) | surface normal |
| SDF | position | distance | surface normal |
| Morphogenetic Field | cell position | concentration | developmental force |
| Energy Landscape | state | energy | gradient descent direction |

### Why They're Topologically Equivalent

```
Coffee Cup = Donut thinking:

Neural Field     : R^n -> R^m
    |                    |
    | homeomorphic       | homeomorphic
    v                    v
SDF Function     : R^3 -> R
    |                    |
    | homeomorphic       | homeomorphic
    v                    v
Morphogen Field  : tissue -> concentration

They all share:
- Continuous mapping
- Differentiable everywhere
- Gradient-based optimization
- Implicit representation
- Coordinate-based queries
```

### The Deeper Connection to Active Inference

From active inference (see 05-axiom-architecture-deep.md):

```python
# Active inference: minimize free energy
F = -log p(o|s) + KL[q(s)||p(s)]

# Neural field: minimize reconstruction loss
L = ||f(x) - y||^2 + R(theta)

# Both are:
# 1. Variational inference
# 2. Gradient-based optimization
# 3. Learning implicit representations
```

### Train Station Connections

**Connection to Predictive Coding**:
- Neural field = learned prediction of signal
- Training = minimizing prediction error
- Gradient = error signal for updating

**Connection to Message Passing**:
- Coordinate queries = messages
- Network layers = message passing
- Output = aggregated messages

**Connection to Self-Organization**:
- Pattern formation without explicit programming
- Local rules (network weights) produce global structure
- Emergence of complex representations

### The Mathematical Bridge

The key insight is that neural fields solve BOUNDARY VALUE PROBLEMS:

```python
# Signed Distance Function (Eikonal equation)
|grad(f)| = 1
f(surface) = 0

# Morphogen diffusion (steady state)
D * laplacian(c) = -source
c(boundary) = fixed

# Heat equation (steady state)
laplacian(T) = 0
T(boundary) = fixed

# ALL are:
# 1. PDEs defined implicitly
# 2. Solved by gradient descent on neural network
# 3. Gradient information is ESSENTIAL
```

This is why SIREN (with accurate gradients) outperforms ReLU networks - the gradients ARE the solution!

---

## Section 6: ARR-COC Connection - Implicit Relevance Fields

### Relevance as a Learned Field

In ARR-COC (Adaptive Relevance Realization for Concept-Object Crossreferencing), we can model relevance as an implicit neural field:

```python
class ImplicitRelevanceField(nn.Module):
    """
    Model relevance as a continuous field over (token, concept) space.

    Instead of discrete relevance scores, learn a smooth relevance manifold.
    """

    def __init__(
        self,
        token_dim: int = 768,
        concept_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()

        # Use SIREN for smooth relevance landscape
        self.relevance_field = SIREN(
            in_features=token_dim + concept_dim,
            out_features=1,
            hidden_features=hidden_dim,
            hidden_layers=num_layers,
            omega_0=30.0
        )

    def forward(
        self,
        token_features: torch.Tensor,
        concept_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance at (token, concept) coordinate.

        Args:
            token_features: [batch, seq_len, token_dim]
            concept_features: [batch, num_concepts, concept_dim]

        Returns:
            relevance: [batch, seq_len, num_concepts]
        """
        batch, seq_len, _ = token_features.shape
        _, num_concepts, _ = concept_features.shape

        # Create all pairs
        tokens_exp = token_features.unsqueeze(2).expand(-1, -1, num_concepts, -1)
        concepts_exp = concept_features.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # Concatenate to form coordinates
        coords = torch.cat([tokens_exp, concepts_exp], dim=-1)

        # Query relevance field
        relevance = self.relevance_field(coords).squeeze(-1)

        return torch.sigmoid(relevance)

    def relevance_gradient(
        self,
        token_features: torch.Tensor,
        concept_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of relevance field.

        The gradient tells us:
        - How to modify token features to increase relevance
        - How to modify concept features to increase relevance

        This is analogous to morphogenetic gradients guiding development!
        """
        token_features = token_features.requires_grad_(True)
        concept_features = concept_features.requires_grad_(True)

        relevance = self.forward(token_features, concept_features)

        # Compute gradients
        grads = torch.autograd.grad(
            relevance.sum(),
            [token_features, concept_features],
            create_graph=True
        )

        return grads[0], grads[1]  # d_relevance/d_token, d_relevance/d_concept
```

### Benefits for ARR-COC

1. **Continuous relevance landscape**: Smooth interpolation between discrete token-concept pairs
2. **Gradient-based optimization**: Can optimize token allocation by following relevance gradients
3. **Implicit compression**: Entire relevance matrix encoded in network weights
4. **Morphogenetic interpretation**: Tokens "develop" relevance through gradient fields

### The 10% Connection

Neural fields provide a principled way to:
- Model relevance as a continuous field
- Use gradients for token allocation (like morphogenetic guidance)
- Achieve implicit compression (relevance encoded in weights)
- Enable smooth interpolation for novel queries

---

## Performance Considerations

### Memory Efficiency

Neural fields store signals in network weights:
- Image: H x W x 3 pixels -> ~1MB network
- 3D shape: Infinite resolution -> ~5MB network
- Video: T x H x W x 3 -> ~10MB network

**Trade-off**: Slower query time vs. massive compression

### GPU Optimization

```python
# Batch coordinate queries for efficiency
@torch.jit.script
def batched_query(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 65536
) -> torch.Tensor:
    """
    Query neural field in chunks to avoid OOM.
    """
    outputs = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i + chunk_size]
        outputs.append(model(chunk))
    return torch.cat(outputs, dim=0)
```

### Training Tips

1. **Learning rate**: Start with 1e-4 for SIREN, can go higher than ReLU
2. **Omega_0**: Higher = more high-frequency detail, but harder to optimize
3. **Batch size**: Larger is better (random sampling helps)
4. **Positional encoding**: More frequencies = more detail, but slower training

### Benchmarks

From SIREN paper:
- **Image fitting**: SIREN achieves 10+ dB higher PSNR than ReLU
- **Convergence**: 2-5x faster than ReLU with positional encoding
- **Gradient accuracy**: Only SIREN accurately represents first and second derivatives

---

## Summary

Neural fields (SIREN, NeRF, implicit representations) provide a powerful way to represent continuous signals:

1. **Implicit representation**: Function, not samples
2. **Continuous**: Query at any coordinate
3. **Differentiable**: Accurate gradients everywhere
4. **Compressed**: Entire signal in network weights

The connection to morphogenetic fields is deep:
- Both use gradients to guide pattern formation
- Both are implicit (the field contains all information)
- Both support self-organization through local rules

This TRAIN STATION unifies:
- **Neural Fields = NeRF = SDF = Morphogenetic Field**
- **Gradient Descent = Morphogenetic Guidance = Free Energy Minimization**
- **Implicit Representation = Developmental Potential = Energy Landscape**

For ARR-COC: Model relevance as an implicit field, use gradients for token allocation!

---

## Sources

**Key Papers:**
- [SIREN: Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) - Sitzmann et al., NeurIPS 2020 (cited 3691 times)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf) - Mildenhall et al., ECCV 2020
- [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739) - Tancik et al., NeurIPS 2020
- [Neural Fields in Visual Computing and Beyond](https://arxiv.org/pdf/2111.11426.pdf) - Xie et al., CGF 2022

**Web Resources:**
- [SIREN Project Page](https://www.vincentsitzmann.com/siren/) (accessed 2025-11-23)
- [Neural Fields FAQ - Brown University](https://neuralfields.cs.brown.edu/faq.html) (accessed 2025-11-23)
- [Awesome Implicit Representations](https://github.com/vsitzmann/awesome-implicit-representations) - GitHub

**Implementations:**
- [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch)
- [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
- [PyTorch3D NeRF Tutorial](https://pytorch3d.org/tutorials/fit_simple_neural_radiance_field)

**Related Concepts:**
- Morphogenetic fields in developmental biology (Wolpert's French Flag Model)
- Reaction-diffusion systems (Turing patterns)
- Active inference and free energy principle
