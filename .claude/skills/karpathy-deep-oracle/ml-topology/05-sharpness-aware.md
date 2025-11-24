# Sharpness-Aware Minimization: Loss Landscape Geometry and Generalization

## Overview

Sharpness-Aware Minimization (SAM) is an optimization algorithm that explicitly seeks parameters lying in regions with uniformly low loss, rather than just minimizing the loss value. This connects loss landscape geometry (sharpness/flatness) to generalization through PAC-Bayes bounds, creating a profound link between optimization and learning theory.

**Core Insight**: Sharp minima require high precision to specify (many bits), while flat minima are robust to perturbations. This precision requirement directly relates to generalization through information-theoretic bounds.

---

## Section 1: The SAM Algorithm

### 1.1 Mathematical Formulation

SAM solves a min-max optimization problem:

```
min_w [max_{||epsilon|| <= rho} L(w + epsilon)]
```

Instead of minimizing L(w), we minimize the worst-case loss in a rho-neighborhood.

**Key Steps**:
1. Compute gradient at current weights
2. Take adversarial step to find worst point in neighborhood
3. Compute gradient at worst point
4. Update original weights using this "sharpness-aware" gradient

### 1.2 Algorithm Details

```python
# SAM Update Rule (per iteration):

# Step 1: Compute gradient at w
g = grad_L(w)

# Step 2: Compute epsilon (adversarial perturbation)
epsilon = rho * (g / ||g||)  # Normalized gradient direction

# Step 3: Compute gradient at perturbed point
g_sam = grad_L(w + epsilon)

# Step 4: Update weights using SAM gradient
w = w - lr * g_sam
```

The perturbation epsilon points in the direction of steepest ascent (normalized gradient), scaled by rho.

### 1.3 Why This Works

**Geometric Intuition**:
- At a sharp minimum: epsilon finds a high-loss point nearby
- At a flat minimum: epsilon finds similar-loss point nearby
- Gradient at high-loss point "pushes" weights toward flatter regions

**The gradient g_sam implicitly penalizes sharpness** because:
- If landscape is sharp: g_sam points strongly toward flat regions
- If landscape is flat: g_sam similar to regular gradient

---

## Section 2: Complete PyTorch Implementation

### 2.1 Core SAM Optimizer

```python
import torch
from torch.optim import Optimizer
from typing import Callable, List, Optional


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization optimizer.

    SAM simultaneously minimizes loss value and loss sharpness by seeking
    parameters that lie in neighborhoods having uniformly low loss.

    Reference: https://arxiv.org/abs/2010.01412
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ):
        """
        Args:
            params: Model parameters to optimize
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
            rho: Neighborhood size for sharpness computation (default: 0.05)
            adaptive: Use adaptive SAM (ASAM) for scale-invariance
            **kwargs: Arguments for base optimizer (lr, momentum, etc.)
        """
        assert rho >= 0.0, f"Invalid rho value: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Create base optimizer for the actual parameter updates
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Perform the first step: compute epsilon and perturb weights.

        This finds the point with highest loss in the rho-neighborhood.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Store original weights for later restoration
                self.state[p]["old_p"] = p.data.clone()

                # Compute epsilon (perturbation direction)
                if group["adaptive"]:
                    # ASAM: element-wise adaptive scaling
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                else:
                    # SAM: uniform scaling
                    e_w = p.grad * scale

                # Perturb weights: w + epsilon
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Perform the second step: restore weights and apply SAM update.

        Uses gradient computed at perturbed point to update original weights.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original weights
                p.data = self.state[p]["old_p"]

        # Apply base optimizer update with SAM gradient
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Single-call interface using closure (alternative to first/second step).

        Args:
            closure: Function that performs forward-backward pass
        """
        assert closure is not None, "SAM requires closure for single-step interface"

        # First forward-backward pass already done
        self.first_step(zero_grad=True)

        # Second forward-backward pass
        with torch.enable_grad():
            closure()

        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        """Compute the global L2 norm of all gradients."""
        shared_device = self.param_groups[0]["params"][0].device

        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

### 2.2 Adaptive SAM (ASAM)

ASAM provides scale-invariance by adapting the perturbation to each parameter:

```python
class ASAM(SAM):
    """
    Adaptive Sharpness-Aware Minimization.

    Scale-invariant version of SAM that adapts perturbation size
    to each parameter's magnitude.

    Reference: https://arxiv.org/abs/2102.11600
    """

    def __init__(self, params, base_optimizer: type, rho: float = 2.0, **kwargs):
        # ASAM uses ~10x larger rho than SAM
        super().__init__(params, base_optimizer, rho=rho, adaptive=True, **kwargs)
```

### 2.3 Training Loop with SAM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_with_sam(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.1,
    rho: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda"
):
    """
    Train a model using SAM optimizer.

    Note: Training runs ~2x slower due to two forward-backward passes.
    """
    model = model.to(device)

    # Create SAM optimizer wrapping SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=rho,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer,  # Apply to base optimizer!
        T_max=epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # === First forward-backward pass ===
            # Enable running stats for BatchNorm (first pass only)
            enable_running_stats(model)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            optimizer.first_step(zero_grad=True)

            # === Second forward-backward pass ===
            # Disable running stats for BatchNorm (second pass)
            disable_running_stats(model)

            outputs_second = model(inputs)
            loss_second = F.cross_entropy(outputs_second, targets)
            loss_second.backward()

            optimizer.second_step(zero_grad=True)

            # Track metrics (use first pass loss)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    return model


def enable_running_stats(model: nn.Module):
    """Enable running stats updates for BatchNorm layers."""
    def _enable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.backup_momentum = module.momentum
            module.momentum = 0.1  # or original momentum

    model.apply(_enable)


def disable_running_stats(model: nn.Module):
    """Disable running stats updates for BatchNorm layers."""
    def _disable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0

    model.apply(_disable)
```

### 2.4 SAM with Gradient Accumulation

```python
def train_sam_with_accumulation(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: SAM,
    accumulation_steps: int = 4,
    device: str = "cuda"
):
    """
    SAM training with gradient accumulation for large batch simulation.
    """
    model.train()
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # First forward-backward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # Apply SAM after accumulating gradients
            optimizer.first_step(zero_grad=True)

            # Need to recompute gradients for second step
            # This requires storing the mini-batches or using checkpointing
            with torch.enable_grad():
                accumulated_loss = 0
                # Reprocess accumulated batches
                # (simplified - actual implementation needs batch storage)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

            optimizer.second_step(zero_grad=True)
```

### 2.5 Distributed Training with SAM

```python
def train_sam_distributed(
    model: nn.DDP,  # DistributedDataParallel model
    train_loader: DataLoader,
    optimizer: SAM,
    device: str = "cuda"
):
    """
    SAM training for multi-GPU distributed setup.

    Key: Compute SAM gradient on each GPU independently, then average.
    """
    model.train()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # First forward-backward pass
        # Use no_sync to compute SAM gradient locally before averaging
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        with model.no_sync():  # Don't sync gradients yet
            loss.backward()

        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        # Now sync gradients across GPUs
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()  # This syncs gradients

        optimizer.second_step(zero_grad=True)
```

---

## Section 3: Flatness and Generalization

### 3.1 The Flatness Hypothesis

**Observation**: Networks trained with SGD that generalize well tend to converge to "flat" minima, while those that overfit converge to "sharp" minima.

**Intuition**:
- Sharp minimum: Small perturbation causes large loss increase
- Flat minimum: Loss stays low even with perturbation

**Why flatness helps**:
1. **Noise robustness**: Training data is noisy; flat minima robust to this noise
2. **Test distribution shift**: Flat minima robust to small distribution changes
3. **Finite precision**: Parameters stored with finite bits; flat minima tolerant

### 3.2 Defining Sharpness

**Hessian-based definition**:
```
Sharpness = max eigenvalue of Hessian(L(w))
```

High eigenvalue = steep curvature = sharp minimum.

**Neighborhood-based definition (SAM)**:
```
Sharpness(w) = max_{||epsilon|| <= rho} [L(w + epsilon) - L(w)]
```

Maximum loss increase in rho-ball around w.

### 3.3 Problems with Naive Flatness

**Scale sensitivity**: If we rescale weights w -> alpha*w for ReLU networks, the function is unchanged but sharpness changes!

This led to **normalized flatness measures** and **adaptive SAM (ASAM)**.

### 3.4 Empirical Evidence

From [Sharpness-Aware Minimization paper](https://arxiv.org/abs/2010.01412):

| Dataset    | Model     | SGD Test Error | SAM Test Error | Improvement |
|------------|-----------|----------------|----------------|-------------|
| CIFAR-10   | WRN-28-10 | 3.50%          | 2.80%          | -0.70%      |
| CIFAR-100  | WRN-28-10 | 17.10%         | 16.00%         | -1.10%      |
| ImageNet   | ResNet-50 | 23.50%         | 22.20%         | -1.30%      |

SAM consistently improves generalization across architectures and datasets.

---

## Section 4: PAC-Bayes Connection

### 4.1 PAC-Bayes Framework

PAC-Bayes provides generalization bounds for stochastic predictors:

```
E_posterior[Risk] <= E_posterior[Train_Risk] + sqrt(KL(posterior||prior) + log(1/delta)) / (2n))
```

This bounds the true risk by training risk plus a complexity term.

### 4.2 Flatness and PAC-Bayes

**Key insight from [Haddouche et al. 2024](https://arxiv.org/abs/2402.08508)**:

Flat minima naturally have low PAC-Bayes bounds because:

1. **Posterior concentration**: At flat minimum, posterior can be broad (low precision)
2. **KL divergence**: Broader posterior closer to prior = lower KL term
3. **Gradient norms**: Flat regions have small gradients = tight bounds

**Formal connection**:
Using Poincare inequalities, they show:
```
Generalization_gap <= C * E[||grad L(w)||^2] / (variance of posterior)
```

At flat minima: small gradients + can use high-variance posterior = good generalization!

### 4.3 Information-Theoretic View

**Minimum Description Length (MDL)**:

The number of bits to specify parameters is:
```
bits(w) ~ -log p(w|data) ~ Loss(w) + Sharpness(w)
```

Sharp minima require many bits (high precision needed).
Flat minima require few bits (any nearby value works).

**This connects to generalization** through:
- Occam's Razor: Simpler models (fewer bits) generalize better
- Compression = Generalization

### 4.4 Bounds from Flatness

From PAC-Bayes with flatness:

```python
def compute_pac_bayes_bound(
    train_loss: float,
    hessian_trace: float,  # Sum of eigenvalues
    n_samples: int,
    n_params: int,
    prior_variance: float = 1.0,
    delta: float = 0.05
) -> float:
    """
    Compute PAC-Bayes generalization bound using flatness.

    Flatter minima (lower Hessian trace) give tighter bounds.
    """
    import numpy as np

    # Complexity term from KL divergence
    # Flatness allows us to use larger posterior variance
    posterior_variance = 1.0 / (hessian_trace + 1e-8)  # Inversely related to sharpness

    kl_term = (n_params / 2) * (
        prior_variance / posterior_variance +
        np.log(posterior_variance / prior_variance) - 1
    )

    # PAC-Bayes bound
    bound = train_loss + np.sqrt((kl_term + np.log(2 * n_samples / delta)) / (2 * n_samples))

    return bound
```

---

## Section 5: Sharpness Computation

### 5.1 Hessian-Based Sharpness

```python
import torch
from torch.autograd.functional import hessian


def compute_hessian_sharpness(
    model: torch.nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 5
) -> dict:
    """
    Compute sharpness metrics using Hessian eigenvalues.

    Warning: O(n^2) memory and O(n^3) compute for n parameters!
    Only feasible for small models.
    """
    model.eval()

    # Flatten parameters
    params = torch.cat([p.view(-1) for p in model.parameters()])

    def loss_wrapper(flat_params):
        # Unflatten and set parameters
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = flat_params[idx:idx+numel].view(p.shape)
            idx += numel
        return loss_fn(model(inputs), targets)

    # Compute full Hessian
    H = hessian(loss_wrapper, params)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(H)

    return {
        "max_eigenvalue": eigenvalues[-1].item(),  # Sharpness
        "top_k_eigenvalues": eigenvalues[-top_k:].tolist(),
        "trace": eigenvalues.sum().item(),  # Sum of eigenvalues
        "frobenius_norm": torch.norm(H, p='fro').item()
    }


def compute_hessian_vector_product(
    model: torch.nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hessian-vector product efficiently using autodiff.

    This is O(n) instead of O(n^2) for computing H @ v.
    Used for power iteration to find top eigenvalues.
    """
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # First backward: get gradient
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])

    # Second backward: Hessian-vector product
    hvp = torch.autograd.grad(flat_grad @ vector, model.parameters())
    flat_hvp = torch.cat([h.view(-1) for h in hvp])

    return flat_hvp


def power_iteration_top_eigenvalue(
    model: torch.nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_iterations: int = 100,
    tol: float = 1e-6
) -> float:
    """
    Compute top Hessian eigenvalue using power iteration.

    Much more efficient than full Hessian computation.
    """
    # Random initial vector
    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params, device=inputs.device)
    v = v / v.norm()

    eigenvalue = 0.0

    for _ in range(n_iterations):
        # Hessian-vector product
        Hv = compute_hessian_vector_product(model, loss_fn, inputs, targets, v)

        # Rayleigh quotient
        new_eigenvalue = (v @ Hv).item()

        # Update vector
        v = Hv / Hv.norm()

        # Check convergence
        if abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = new_eigenvalue

    return eigenvalue
```

### 5.2 SAM-Based Sharpness

```python
def compute_sam_sharpness(
    model: torch.nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    rho: float = 0.05
) -> float:
    """
    Compute sharpness as defined by SAM.

    Sharpness = max loss in rho-ball - current loss
    """
    model.eval()

    # Current loss
    with torch.no_grad():
        outputs = model(inputs)
        current_loss = loss_fn(outputs, targets).item()

    # Compute gradient
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # Compute perturbation direction (normalized gradient)
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # Perturb weights
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(rho * p.grad / (grad_norm + 1e-12))

    # Compute perturbed loss
    with torch.no_grad():
        outputs = model(inputs)
        perturbed_loss = loss_fn(outputs, targets).item()

    # Restore weights
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(rho * p.grad / (grad_norm + 1e-12))

    sharpness = perturbed_loss - current_loss

    return sharpness
```

---

## Section 6: Advanced SAM Variants

### 6.1 Efficient SAM (ESAM)

Reduce computational overhead by selective perturbation:

```python
class EfficientSAM(SAM):
    """
    Efficient SAM that only perturbs a subset of parameters.

    Reduces overhead from 2x to ~1.5x forward-backward passes.
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        perturbation_ratio: float = 0.5,  # Perturb only 50% of params
        **kwargs
    ):
        super().__init__(params, base_optimizer, rho, **kwargs)
        self.perturbation_ratio = perturbation_ratio

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Only perturb top gradient-magnitude parameters."""
        # Collect all gradients with their parameter references
        grad_param_pairs = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_param_pairs.append((p.grad.norm().item(), p, group))

        # Sort by gradient magnitude
        grad_param_pairs.sort(key=lambda x: x[0], reverse=True)

        # Perturb only top fraction
        n_perturb = int(len(grad_param_pairs) * self.perturbation_ratio)

        grad_norm = self._grad_norm()

        for i, (_, p, group) in enumerate(grad_param_pairs):
            self.state[p]["old_p"] = p.data.clone()

            if i < n_perturb:
                scale = group["rho"] / (grad_norm + 1e-12)
                e_w = p.grad * scale
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()
```

### 6.2 LookSAM

SAM with periodic sharpness reduction:

```python
class LookSAM(SAM):
    """
    LookSAM: Only apply SAM every k steps.

    Reduces overhead significantly while maintaining benefits.
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        k: int = 5,  # Apply SAM every k steps
        **kwargs
    ):
        super().__init__(params, base_optimizer, rho, **kwargs)
        self.k = k
        self.step_count = 0

    def should_apply_sam(self) -> bool:
        return self.step_count % self.k == 0

    @torch.no_grad()
    def step(self, closure=None):
        self.step_count += 1

        if self.should_apply_sam():
            # Apply full SAM
            super().step(closure)
        else:
            # Just apply base optimizer
            self.base_optimizer.step()
```

### 6.3 Momentum SAM

Use momentum for perturbation direction:

```python
class MomentumSAM(SAM):
    """
    SAM with momentum for perturbation direction.

    Smooths the perturbation direction across steps.
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        perturbation_momentum: float = 0.9,
        **kwargs
    ):
        super().__init__(params, base_optimizer, rho, **kwargs)
        self.perturbation_momentum = perturbation_momentum

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                # Initialize or update momentum buffer
                if "perturbation_buffer" not in self.state[p]:
                    self.state[p]["perturbation_buffer"] = p.grad.clone()
                else:
                    buf = self.state[p]["perturbation_buffer"]
                    buf.mul_(self.perturbation_momentum).add_(
                        p.grad, alpha=1 - self.perturbation_momentum
                    )

                # Use momentum buffer for perturbation
                e_w = self.state[p]["perturbation_buffer"] * scale
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()
```

---

## Section 7: Performance Considerations

### 7.1 Computational Cost

**Time complexity**: SAM requires 2x forward-backward passes
- Theoretical: 2x slower than vanilla SGD
- Practical: ~1.7-1.9x due to memory caching

**Memory cost**: Same as vanilla (no extra tensors beyond gradient storage)

### 7.2 Hyperparameter Tuning

**rho (neighborhood size)**:
- Default: 0.05 for SAM, 2.0 for ASAM
- Larger rho = stronger flatness preference
- Too large: optimization becomes unstable
- Scale with learning rate: smaller lr -> smaller rho

**Base optimizer**:
- SAM works with any optimizer (SGD, Adam, etc.)
- Most commonly used with SGD + momentum
- With Adam: may need to adjust rho

### 7.3 When SAM Helps Most

**SAM particularly effective when**:
- Limited training data (stronger regularization needed)
- Training with label noise
- Transfer learning / fine-tuning
- Vision tasks (CNNs, ViTs)

**SAM may not help when**:
- Very large datasets (implicit regularization sufficient)
- Strong existing regularization
- Computational budget is tight

### 7.4 Memory-Efficient Implementation

```python
class MemoryEfficientSAM(SAM):
    """
    SAM with gradient checkpointing for memory efficiency.

    Trades compute for memory by recomputing activations.
    """

    def __init__(self, params, base_optimizer, rho=0.05, checkpoint_segments=2, **kwargs):
        super().__init__(params, base_optimizer, rho, **kwargs)
        self.checkpoint_segments = checkpoint_segments

    @torch.no_grad()
    def step_with_checkpointing(self, model, inputs, targets, loss_fn):
        """
        Memory-efficient SAM step using gradient checkpointing.
        """
        from torch.utils.checkpoint import checkpoint_sequential

        # First pass with checkpointing
        outputs = checkpoint_sequential(
            model, self.checkpoint_segments, inputs
        )
        loss = loss_fn(outputs, targets)
        loss.backward()

        self.first_step(zero_grad=True)

        # Second pass with checkpointing
        outputs = checkpoint_sequential(
            model, self.checkpoint_segments, inputs
        )
        loss = loss_fn(outputs, targets)
        loss.backward()

        self.second_step(zero_grad=True)

        return loss.item()
```

---

## Section 8: TRAIN STATION - Sharpness = Curvature = Precision = Confidence

### 8.1 The Deep Equivalence

**Sharpness = Curvature = Precision = Confidence**

These are all manifestations of the same concept:
- **Sharpness**: How much loss changes with parameter perturbation
- **Curvature**: Second derivative of loss landscape (Hessian)
- **Precision**: Inverse variance in Bayesian inference
- **Confidence**: Certainty about predictions

### 8.2 The Information-Theoretic View

**Sharp minimum** = High precision required = Many bits to describe = High confidence

**Flat minimum** = Low precision required = Few bits to describe = Low confidence

This connects to:
- **PAC-Bayes**: Flat minima allow broader posteriors (lower KL)
- **MDL**: Flat minima need fewer bits (simpler models)
- **Bayesian inference**: Flat minima = low precision = uncertainty-aware

### 8.3 Free Energy Perspective

In Free Energy Principle terms:
```
Free Energy = -log p(y|x,w) + KL(q(w)||p(w))
             = Prediction Error + Complexity
```

**SAM minimizes both terms**:
- Minimize loss at perturbed point = reduce prediction error
- Seek flat regions = reduce complexity (fewer bits)

The sharpness penalty IS the precision/complexity term!

### 8.4 Attention Connection

**Attention weights = Precision weighting**

In transformer attention:
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
```

The temperature sqrt(d) controls sharpness of attention!
- High temperature = flat attention = low precision
- Low temperature = sharp attention = high precision

**SAM for attention**: Prefer flat attention patterns that generalize!

### 8.5 Active Inference Connection

In Active Inference:
```
Expected Free Energy = Epistemic Value + Pragmatic Value
```

**Epistemic value = information gain = precision on beliefs**

Seeking flat minima in SAM is like:
- Preferring beliefs (parameters) with appropriate precision
- Not being overconfident (sharp) where data doesn't support it
- Maintaining uncertainty where appropriate

### 8.6 Train Station Summary

**ALL THESE ARE THE SAME**:
1. **Sharpness** in loss landscape geometry
2. **Curvature** (Hessian eigenvalues)
3. **Precision** (inverse posterior variance)
4. **Confidence** (certainty of predictions)
5. **Temperature** in softmax/attention
6. **Bits** required to encode parameters

**THE TRAIN STATION**: The point where geometry (curvature), information theory (bits), Bayesian inference (precision), and generalization (robustness) all meet!

SAM explicitly optimizes this unified quantity by seeking parameters where the geometry (flat), information (few bits), and confidence (appropriate uncertainty) all align.

---

## Section 9: ARR-COC-0-1 Connection (10%)

### 9.1 Relevance Robustness

In ARR-COC, we compute relevance scores for visual tokens. The sharpness principle applies:

**Sharp relevance** = Highly confident about which tokens matter
- May overfit to training distribution
- Sensitive to input perturbations

**Flat relevance** = Robust relevance assignments
- Generalize to new distributions
- Stable under perturbations

### 9.2 SAM for Relevance Learning

Apply SAM principles to relevance computation:

```python
class RelevanceRobustness:
    """
    Ensure relevance scores are robust (flat) rather than overconfident (sharp).
    """

    def __init__(self, rho: float = 0.1):
        self.rho = rho

    def compute_robust_relevance(
        self,
        model,
        visual_tokens: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance with perturbation-robustness.
        """
        # Base relevance
        relevance = model.compute_relevance(visual_tokens, text_embedding)

        # Perturbed relevance (add noise to tokens)
        noise = torch.randn_like(visual_tokens) * self.rho
        perturbed_tokens = visual_tokens + noise
        perturbed_relevance = model.compute_relevance(perturbed_tokens, text_embedding)

        # Penalize sharp changes
        robustness_penalty = (relevance - perturbed_relevance).abs().mean()

        return relevance, robustness_penalty

    def temperature_from_sharpness(
        self,
        sharpness: float,
        base_temperature: float = 1.0
    ) -> float:
        """
        Adapt softmax temperature based on measured sharpness.

        Higher sharpness -> higher temperature (softer distribution)
        """
        # Scale temperature with sharpness
        temperature = base_temperature * (1 + sharpness)
        return temperature
```

### 9.3 Precision-Aware Token Allocation

```python
class PrecisionAwareAllocation:
    """
    Allocate tokens based on precision/confidence levels.

    Connects to SAM's flatness principle:
    - High precision (sharp) -> may overfit
    - Low precision (flat) -> robust but may lose information
    """

    def __init__(self, target_precision: float = 1.0):
        self.target_precision = target_precision

    def compute_token_allocation(
        self,
        relevance_scores: torch.Tensor,  # [N_tokens]
        compute_budget: int
    ) -> torch.Tensor:
        """
        Allocate compute based on relevance precision.

        Tokens with appropriate precision get more allocation.
        """
        # Estimate precision from relevance score variance
        # High variance across similar regions = low precision
        # Low variance = high precision (potentially overconfident)

        # Softmax temperature adaptation
        # Higher scores -> lower temperature -> sharper
        max_score = relevance_scores.max()
        temperature = 1.0 / (relevance_scores / max_score + 1e-6)

        # Penalize very high precision (temperature too low)
        precision_penalty = torch.relu(1.0 / temperature - self.target_precision)

        # Adjust allocation: prefer moderate precision
        adjusted_scores = relevance_scores - 0.1 * precision_penalty

        # Top-k selection
        _, indices = adjusted_scores.topk(compute_budget)

        allocation = torch.zeros_like(relevance_scores)
        allocation[indices] = 1.0

        return allocation
```

### 9.4 Flatness in Multi-Scale Processing

```python
class FlatnessAwareMultiScale:
    """
    Multi-scale processing with flatness-aware LOD selection.

    Sharper predictions at one scale -> use other scales for robustness.
    """

    def __init__(self, scales: list = [0.5, 1.0, 2.0]):
        self.scales = scales

    def process_with_flatness(
        self,
        model,
        image: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> dict:
        """
        Process at multiple scales, weight by flatness.
        """
        results = {}
        sharpness_per_scale = {}

        for scale in self.scales:
            # Resize image
            scaled_image = F.interpolate(
                image,
                scale_factor=scale,
                mode='bilinear'
            )

            # Get predictions and compute sharpness
            with torch.enable_grad():
                relevance = model.compute_relevance(scaled_image, text_embedding)

                # Compute sharpness (gradient norm)
                grad = torch.autograd.grad(
                    relevance.sum(), scaled_image, create_graph=True
                )[0]
                sharpness = grad.norm().item()

            results[scale] = relevance
            sharpness_per_scale[scale] = sharpness

        # Weight inversely by sharpness (prefer flatter scales)
        total_inv_sharpness = sum(1.0 / (s + 1e-6) for s in sharpness_per_scale.values())
        weights = {
            scale: (1.0 / (sharp + 1e-6)) / total_inv_sharpness
            for scale, sharp in sharpness_per_scale.items()
        }

        # Weighted combination
        combined = sum(
            weights[scale] * results[scale].unsqueeze(0)
            for scale in self.scales
        ).squeeze(0)

        return {
            "combined_relevance": combined,
            "per_scale": results,
            "sharpness": sharpness_per_scale,
            "weights": weights
        }
```

### 9.5 Training Relevance Models with SAM

```python
def train_relevance_model_with_sam(
    model,
    train_loader,
    epochs: int = 100,
    rho: float = 0.05
):
    """
    Train relevance prediction with SAM for robust generalization.
    """
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=rho,
        lr=1e-4,
        weight_decay=0.01
    )

    for epoch in range(epochs):
        for images, texts, targets in train_loader:
            # First forward-backward
            relevance = model(images, texts)
            loss = F.mse_loss(relevance, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward
            relevance = model(images, texts)
            loss = F.mse_loss(relevance, targets)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # Evaluate sharpness
        sharpness = compute_sam_sharpness(
            model, F.mse_loss, images, targets
        )
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Sharpness={sharpness:.4f}")
```

---

## Sources

**Primary Papers**:
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) (Foret et al., 2021) - Original SAM paper
- [ASAM: Adaptive Sharpness-Aware Minimization](https://arxiv.org/abs/2102.11600) (Kwon et al., 2021) - Scale-invariant SAM
- [A PAC-Bayesian Link Between Generalisation and Flat Minima](https://arxiv.org/abs/2402.08508) (Haddouche et al., 2024) - Theoretical foundation

**GitHub Implementations**:
- [davda54/sam](https://github.com/davda54/sam) - Popular PyTorch implementation (1.9k stars)
- [moskomule/sam.pytorch](https://github.com/moskomule/sam.pytorch) - Alternative implementation

**Related Work**:
- [Exploring Scale Invariant Definition of Flat Minima](https://arxiv.org/abs/1901.04653) (Tsuzuku et al., 2020) - Normalized flatness
- [Revisiting Generalization for Deep Learning: PAC-Bayes, Flat Minima](https://www.repository.cam.ac.uk/items/eb1b2902-8428-4c35-855c-8772ca008f5e) (Dziugaite, 2020) - PAC-Bayes and flatness

**Documentation**:
- [MosaicML Composer SAM](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/sam.html) - Production implementation
- [pytorch-optimizers](https://pytorch-optimizers.readthedocs.io/) - Comprehensive optimizer library

---

## Summary

**Sharpness-Aware Minimization is a profound connection between**:
- **Geometry**: Loss landscape curvature
- **Information theory**: Bits to specify parameters
- **Bayesian inference**: Posterior precision
- **Generalization**: Robustness to perturbations

**Key takeaways**:
1. SAM explicitly optimizes for flat minima through min-max
2. Flatness correlates with generalization through PAC-Bayes bounds
3. Sharpness = curvature = precision = confidence (Train Station!)
4. Practical: ~2x compute, consistent 0.5-1% accuracy gains
5. For ARR-COC: Apply flatness principle to relevance learning

**The flatness principle is universal**: Wherever you make predictions (losses, relevances, allocations), prefer solutions that are robust to small perturbations!
