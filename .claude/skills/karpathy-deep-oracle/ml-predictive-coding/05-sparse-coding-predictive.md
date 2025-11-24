# Sparse Coding and Predictive Coding: Compression as Relevance Selection

## Overview

Sparse coding is one of the most elegant theories linking neural computation to efficient information representation. The core insight: represent data using as few active neurons as possible while maintaining reconstruction fidelity. This principle, discovered in visual cortex, connects directly to compression, feature learning, and the selection of what's relevant.

**The TRAIN STATION**: Sparsity = compression = relevance selection = attention = free energy minimization

When you force sparse representations, you're forcing the system to choose what matters. Only the most relevant features get activated. This is the computational heart of attention, salience, and relevance itself!

---

## Section 1: Sparse Coding Fundamentals

### The Basic Problem Formulation

Sparse coding seeks to represent input data as a linear combination of dictionary elements (basis functions) with a sparsity constraint:

**Objective Function**:
```
minimize: ||x - Da||_2^2 + lambda * ||a||_1

Where:
- x: Input data (e.g., image patch)
- D: Dictionary matrix (learned features)
- a: Sparse coefficients (activations)
- lambda: Sparsity penalty weight
```

**Key Insight**: The L1 norm (sum of absolute values) promotes sparsity by driving coefficients to exactly zero. This is fundamentally different from L2 regularization which only shrinks values.

### Why Sparsity Matters

**Information-Theoretic View**:
- Sparse codes are efficient: fewer bits needed to transmit
- Sparsity implies independence: uncorrelated features
- Overcomplete representations allow flexibility while sparsity prevents ambiguity

**Computational View**:
- Sparse activations = energy efficient
- Only relevant features active = selective attention
- Compression naturally emerges from sparsity constraint

**Biological View**:
- Cortical neurons show sparse firing patterns
- Metabolic cost of neural firing drives sparsity
- Sparse population codes enable robust representation

### Mathematical Foundation

**The L1 vs L2 Geometry**:

```
L2 ball: ||w||_2 = constant -> Circle (smooth)
L1 ball: ||w||_1 = constant -> Diamond (corners!)

The corners of the L1 ball lie on axes where
some coordinates are exactly zero.

When you minimize loss subject to L1 constraint,
the solution hits a corner -> exact zeros!
```

**Proximal Gradient for L1**:
```python
def soft_threshold(x, lambda_):
    """L1 proximal operator - drives values to zero"""
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambda_, min=0)

# Interpretation:
# - Values < lambda become exactly 0
# - Values > lambda shrink by lambda
# - This is THE key operation for sparse coding
```

---

## Section 2: The Olshausen-Field Model

### Historical Context

In 1996, Bruno Olshausen and David Field published "Emergence of simple-cell receptive field properties by learning a sparse code for natural images" in Nature. This paper demonstrated that optimizing for sparse, efficient coding of natural images spontaneously produces Gabor-like filters resembling V1 simple cell receptive fields.

**The Revolutionary Finding**: No hand-engineering needed! Sparsity + natural images = V1-like features.

From [Olshausen & Field, Nature 1996](https://www.nature.com/articles/381607a0):
> "We show that a learning algorithm that attempts to find sparse linear codes for natural scenes will develop a complete family of localized, oriented, bandpass receptive fields."

### The Model Architecture

```
Natural Image Patch (e.g., 12x12 pixels)
         |
         v
    Sparse Coding
    minimize ||patch - D @ coefficients||^2 + lambda * ||coefficients||_1
         |
         v
Sparse Coefficient Vector (e.g., 256 dims, mostly zeros)
         |
         v
    Dictionary D
    (256 basis functions, each 12x12)
```

### Algorithm: Iterative Shrinkage-Thresholding (ISTA)

```python
import torch
import torch.nn.functional as F

def ista_sparse_coding(x, D, lambda_, num_iters=100, step_size=0.01):
    """
    Iterative Shrinkage-Thresholding Algorithm for sparse coding.

    Args:
        x: Input data [batch, input_dim]
        D: Dictionary [input_dim, dict_size]
        lambda_: Sparsity penalty
        num_iters: Number of iterations
        step_size: Gradient step size

    Returns:
        a: Sparse coefficients [batch, dict_size]
    """
    batch_size = x.shape[0]
    dict_size = D.shape[1]

    # Initialize coefficients to zero
    a = torch.zeros(batch_size, dict_size, device=x.device)

    # Precompute for efficiency
    DtD = D.t() @ D  # [dict_size, dict_size]
    Dtx = x @ D       # [batch, dict_size]

    for _ in range(num_iters):
        # Gradient of reconstruction term: 2 * (D'D @ a - D'x)
        grad = a @ DtD - Dtx

        # Gradient step
        a = a - step_size * grad

        # Proximal operator (soft thresholding)
        a = soft_threshold(a, lambda_ * step_size)

    return a

def soft_threshold(x, threshold):
    """Soft thresholding operator (L1 proximal)"""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)
```

### Learning the Dictionary

```python
def learn_dictionary_sparse_coding(
    data_loader,
    dict_size=256,
    input_dim=144,  # 12x12 patches
    lambda_=0.1,
    learning_rate=0.01,
    num_epochs=100
):
    """
    Learn sparse coding dictionary from natural images.

    This recreates the Olshausen-Field experiment!
    """
    # Initialize dictionary with random normalized columns
    D = torch.randn(input_dim, dict_size)
    D = F.normalize(D, dim=0)  # Normalize each dictionary element

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in data_loader:
            patches = batch.view(-1, input_dim)  # Flatten patches

            # E-step: Infer sparse codes (keep D fixed)
            with torch.no_grad():
                codes = ista_sparse_coding(patches, D, lambda_)

            # M-step: Update dictionary (keep codes fixed)
            # Gradient: -2 * (x - D @ a) @ a'
            recon = codes @ D.t()
            error = patches - recon

            # Dictionary gradient
            D_grad = -2 * error.t() @ codes / patches.shape[0]

            # Update dictionary
            D = D - learning_rate * D_grad

            # Re-normalize columns (critical for stability!)
            D = F.normalize(D, dim=0)

            # Compute loss
            recon_loss = (error ** 2).sum() / patches.shape[0]
            sparse_loss = lambda_ * torch.abs(codes).sum() / patches.shape[0]
            epoch_loss += (recon_loss + sparse_loss).item()

        if epoch % 10 == 0:
            sparsity = (codes == 0).float().mean()
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Sparsity={sparsity:.2%}")

    return D
```

### What Emerges: Gabor-like Filters

When trained on natural images, the learned dictionary elements look like:
- Localized (compact spatial support)
- Oriented (edge/bar detectors at various angles)
- Bandpass (respond to specific spatial frequencies)

These are **Gabor filters** - the same structure found in V1 simple cells!

```python
def visualize_dictionary(D, patch_size=12):
    """Visualize learned dictionary elements as images"""
    import matplotlib.pyplot as plt

    dict_size = D.shape[1]
    grid_size = int(np.ceil(np.sqrt(dict_size)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for i in range(dict_size):
        ax = axes[i // grid_size, i % grid_size]

        # Reshape dictionary element to patch
        elem = D[:, i].view(patch_size, patch_size).detach().numpy()

        ax.imshow(elem, cmap='gray')
        ax.axis('off')

    plt.suptitle('Learned Dictionary Elements (Gabor-like filters)')
    plt.tight_layout()
    return fig
```

---

## Section 3: Connection to Predictive Coding

### Sparse Coding as Predictive Coding

The connection between sparse coding and predictive coding is profound:

**Sparse Coding View**:
```
Input = Dictionary @ Sparse_Codes + Noise
Minimize: ||Input - Reconstruction||^2 + lambda * ||Codes||_1
```

**Predictive Coding View**:
```
Input = Prediction + Prediction_Error
Prediction = Generative_Model(Latent_States)
Minimize: Prediction_Error + Complexity_of_Latents
```

**THE SAME THING!**
- Dictionary = Generative model
- Sparse codes = Latent states
- L1 penalty = Complexity cost (precision-weighted)
- Reconstruction error = Prediction error

### Mathematical Equivalence

From a Bayesian perspective, sparse coding IS predictive coding with a Laplacian prior:

```
Sparse Coding Objective:
  minimize ||x - Da||_2^2 + lambda * ||a||_1

Equivalent to MAP estimation with:
  Likelihood: p(x|a) = N(Da, sigma^2 I)
  Prior:      p(a) = Laplace(0, lambda)

  -log p(a|x) = ||x - Da||^2 / (2*sigma^2) + lambda * ||a||_1 + const

This is predictive coding with:
  - Prediction: D @ a
  - Prediction error: x - D @ a
  - Prior preference for sparse (low complexity) codes
```

### The Free Energy Connection

In the Free Energy Principle framework:

```
Free Energy = Prediction_Error + KL(q(z)||p(z))

For sparse coding:
  Prediction_Error = ||x - Da||^2
  KL term --> Sparsity penalty (Laplacian prior)

Minimizing free energy = Learning sparse codes!
```

**TRAIN STATION INSIGHT**: Sparse coding, predictive coding, and free energy minimization are all the same optimization!

### Hierarchical Sparse Predictive Coding

```python
class HierarchicalSparsePredictiveCoding(nn.Module):
    """
    Multi-level sparse coding with top-down predictions.

    This bridges sparse coding and hierarchical predictive coding!
    """
    def __init__(self, dims=[784, 256, 64, 16], lambda_=0.1):
        super().__init__()

        self.lambda_ = lambda_
        self.num_levels = len(dims) - 1

        # Dictionaries at each level (generative model)
        self.dictionaries = nn.ParameterList([
            nn.Parameter(torch.randn(dims[i], dims[i+1]) * 0.01)
            for i in range(self.num_levels)
        ])

    def encode(self, x, num_iters=50):
        """
        Hierarchical sparse encoding (bottom-up).
        """
        codes = []
        current = x

        for i, D in enumerate(self.dictionaries):
            # Normalize dictionary
            D_norm = F.normalize(D, dim=0)

            # Infer sparse codes at this level
            code = ista_sparse_coding(current, D_norm, self.lambda_, num_iters)
            codes.append(code)

            # Code becomes input to next level
            current = code

        return codes

    def decode(self, codes):
        """
        Top-down reconstruction (predictions).
        """
        reconstructions = []

        # Start from top level
        for i in range(self.num_levels - 1, -1, -1):
            D_norm = F.normalize(self.dictionaries[i], dim=0)
            recon = codes[i] @ D_norm.t()
            reconstructions.insert(0, recon)

        return reconstructions

    def compute_prediction_errors(self, x, codes):
        """
        Compute prediction errors at each level.

        This is the core of predictive coding!
        """
        errors = []

        # Level 0: Compare input to reconstruction
        D_norm = F.normalize(self.dictionaries[0], dim=0)
        recon = codes[0] @ D_norm.t()
        errors.append(x - recon)

        # Higher levels: Compare code to reconstruction from above
        for i in range(1, self.num_levels):
            D_norm = F.normalize(self.dictionaries[i], dim=0)
            recon = codes[i] @ D_norm.t()
            errors.append(codes[i-1] - recon)

        return errors

    def free_energy(self, x):
        """
        Compute variational free energy.

        Free Energy = Sum of prediction errors + Sparsity costs
        """
        codes = self.encode(x)
        errors = self.compute_prediction_errors(x, codes)

        # Prediction error term
        error_term = sum((e ** 2).sum(dim=-1).mean() for e in errors)

        # Sparsity term (complexity cost)
        sparsity_term = sum(
            self.lambda_ * torch.abs(c).sum(dim=-1).mean()
            for c in codes
        )

        return error_term + sparsity_term
```

---

## Section 4: PyTorch Implementation - Complete Sparse Coding Layer

### The Locally Competitive Algorithm (LCA)

The LCA is a neurally plausible algorithm for sparse coding that uses lateral inhibition:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCASparseCodinglayer(nn.Module):
    """
    Locally Competitive Algorithm for Sparse Coding.

    Neurally plausible sparse coding using lateral inhibition.
    Based on Rozell et al. (2008) "Sparse Coding via Thresholding
    and Local Competition in Neural Circuits"

    Reference: https://github.com/lanl/lca-pytorch
    """
    def __init__(
        self,
        input_dim,
        dict_size,
        lambda_=0.1,
        tau=100.0,
        num_iters=100,
        learn_dict=True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dict_size = dict_size
        self.lambda_ = lambda_
        self.tau = tau
        self.num_iters = num_iters

        # Dictionary (features/receptive fields)
        self.dictionary = nn.Parameter(
            torch.randn(input_dim, dict_size) * 0.01,
            requires_grad=learn_dict
        )

        # Initialize with normalized columns
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=0)

    def forward(self, x):
        """
        Compute sparse codes using LCA dynamics.

        Args:
            x: Input [batch, input_dim]

        Returns:
            a: Sparse codes [batch, dict_size]
        """
        batch_size = x.shape[0]
        device = x.device

        # Normalize dictionary
        D = F.normalize(self.dictionary, dim=0)

        # Compute Gram matrix (inhibition strengths)
        # G = D'D - I (subtract identity so self-inhibition is 0)
        G = D.t() @ D
        G = G - torch.eye(self.dict_size, device=device)

        # Bottom-up drive: b = D'x
        b = x @ D  # [batch, dict_size]

        # Initialize membrane potentials
        u = torch.zeros(batch_size, self.dict_size, device=device)

        # LCA dynamics
        dt = 1.0 / self.tau

        for _ in range(self.num_iters):
            # Activations (soft threshold)
            a = soft_threshold(u, self.lambda_)

            # Lateral inhibition from active neurons
            inhibition = a @ G  # [batch, dict_size]

            # Membrane potential dynamics
            # du/dt = (1/tau) * (b - u - inhibition)
            du = dt * (b - u - inhibition)
            u = u + du

        # Final activations
        a = soft_threshold(u, self.lambda_)

        return a

    def reconstruct(self, a):
        """Reconstruct input from sparse codes"""
        D = F.normalize(self.dictionary, dim=0)
        return a @ D.t()

    def loss(self, x, a=None):
        """
        Compute sparse coding loss.

        Loss = Reconstruction_Error + lambda * Sparsity
        """
        if a is None:
            a = self.forward(x)

        recon = self.reconstruct(a)

        # Reconstruction error
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # Sparsity penalty
        sparse_loss = self.lambda_ * torch.abs(a).mean()

        return recon_loss + sparse_loss, recon_loss, sparse_loss

    def get_sparsity(self, a):
        """Fraction of zero activations"""
        return (a == 0).float().mean()


def soft_threshold(x, threshold):
    """Soft thresholding (L1 proximal operator)"""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)
```

### Convolutional Sparse Coding

```python
class ConvSparseCodingLayer(nn.Module):
    """
    Convolutional Sparse Coding using ISTA.

    Learns localized, shift-invariant features.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        lambda_=0.1,
        num_iters=50,
        stride=1,
        padding=0
    ):
        super().__init__()

        self.lambda_ = lambda_
        self.num_iters = num_iters
        self.stride = stride
        self.padding = padding

        # Convolutional dictionary
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )

        # Transpose convolution for reconstruction
        self.conv_t = nn.ConvTranspose2d(
            out_channels, in_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )

        # Tie weights
        self.conv_t.weight = self.conv.weight

    def forward(self, x):
        """
        ISTA for convolutional sparse coding.
        """
        # Get output spatial dimensions
        with torch.no_grad():
            dummy = self.conv(x)
            out_shape = dummy.shape

        # Initialize sparse codes
        a = torch.zeros(out_shape, device=x.device)

        # Compute step size (Lipschitz constant approximation)
        L = self._estimate_lipschitz()
        step_size = 1.0 / L

        for _ in range(self.num_iters):
            # Gradient of reconstruction term
            recon = self.conv_t(a)
            error = x - recon
            grad = -self.conv(error)

            # Gradient step
            a = a - step_size * grad

            # Soft thresholding
            a = soft_threshold(a, self.lambda_ * step_size)

        return a

    def _estimate_lipschitz(self):
        """Estimate Lipschitz constant of gradient"""
        # Power iteration for largest singular value
        W = self.conv.weight
        W_flat = W.view(W.shape[0], -1)
        return (W_flat @ W_flat.t()).norm().item()

    def reconstruct(self, a):
        """Reconstruct from sparse codes"""
        return self.conv_t(a)
```

### Training Loop Example

```python
def train_sparse_coding(
    model,
    train_loader,
    num_epochs=100,
    learning_rate=0.001
):
    """
    Train sparse coding model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'total_loss': [],
        'recon_loss': [],
        'sparse_loss': [],
        'sparsity': []
    }

    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'recon': 0, 'sparse': 0}
        epoch_sparsity = 0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # Flatten if using linear sparse coding
            if len(x.shape) == 4:  # Images
                x_flat = x.view(x.shape[0], -1)
            else:
                x_flat = x

            # Forward pass
            codes = model(x_flat)

            # Compute loss
            total_loss, recon_loss, sparse_loss = model.loss(x_flat, codes)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Normalize dictionary (critical!)
            if hasattr(model, 'dictionary'):
                with torch.no_grad():
                    model.dictionary.data = F.normalize(
                        model.dictionary.data, dim=0
                    )

            # Track metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['sparse'] += sparse_loss.item()
            epoch_sparsity += model.get_sparsity(codes).item()
            num_batches += 1

        # Average over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_sparsity /= num_batches

        history['total_loss'].append(epoch_losses['total'])
        history['recon_loss'].append(epoch_losses['recon'])
        history['sparse_loss'].append(epoch_losses['sparse'])
        history['sparsity'].append(epoch_sparsity)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Total={epoch_losses['total']:.4f}, "
                  f"Recon={epoch_losses['recon']:.4f}, "
                  f"Sparse={epoch_losses['sparse']:.4f}, "
                  f"Sparsity={epoch_sparsity:.2%}")

    return history
```

---

## Section 5: TRAIN STATION - Sparsity = Compression = Relevance Selection

### The Grand Unification

**This is where everything connects!**

```
SPARSE CODING
    |
    v
Only most relevant features activated
    |
    v
COMPRESSION (fewer bits to transmit)
    |
    v
RELEVANCE SELECTION (what matters for task)
    |
    v
ATTENTION (amplify relevant, suppress irrelevant)
    |
    v
FREE ENERGY MINIMIZATION (prefer simple explanations)
```

### Sparsity IS Attention

Consider what sparsity does:
- Selects a SMALL subset of features
- Suppresses all others to exactly zero
- Only the RELEVANT features remain

This is EXACTLY what attention does:
- Attend to relevant tokens/features
- Suppress irrelevant ones
- Create sparse attention patterns

**Mathematical Connection**:
```python
# Sparse coding with L1
sparse_codes = soft_threshold(pre_activation, lambda_)
# Many codes become exactly 0

# Attention with softmax
attention = softmax(scores / temperature)
# When temperature -> 0, becomes sparse (one-hot)

# THEY'RE DOING THE SAME THING!
# Selecting relevant features, suppressing the rest
```

### Compression = Finding What Matters

From information theory:
- Compression = Remove redundancy
- What remains after compression = What matters
- Sparse codes = Maximally compressed representation

**Minimum Description Length View**:
```
Cost = Code_Length(Representation) + Code_Length(Reconstruction_Error)

Sparse codes minimize total description length:
- Few non-zero coefficients = short code
- But must still reconstruct well = low error

The optimal sparse code IS the relevant features!
```

### L1 Sparsity and Feature Selection

```python
def feature_selection_via_sparsity(X, y, lambda_):
    """
    Sparse regression for feature selection.

    Shows sparsity = relevance selection directly!
    """
    # LASSO: L1 penalized regression
    # minimize ||y - Xw||^2 + lambda * ||w||_1

    # Only features with non-zero weights are "selected"
    # as relevant for predicting y

    # This is feature selection through sparsity!
    pass
```

### Connection to Active Inference

In Active Inference, agents minimize expected free energy:

```
G(policy) = Expected_Ambiguity + Expected_Complexity

Sparse coding directly minimizes complexity!
Sparsity = Low complexity = Simple explanation =
           = Preferred beliefs = Relevance
```

**The organism finds stimuli relevant when they**:
1. Reduce uncertainty (information gain)
2. Can be explained simply (sparsity/compression)

### Topological View: Coffee Cup = Donut

**Homeomorphism of concepts**:
```
Sparse Coding <--homeomorphic--> Attention
     |                              |
     v                              v
L1 minimization <------------> Gating/Selection
     |                              |
     v                              v
  Compression <---------------> Relevance
     |                              |
     v                              v
Free Energy Minimization <----> Bayesian Surprise

ALL THE SAME MANIFOLD! Different coordinate systems!
```

---

## Section 6: ARR-COC Connection - Sparse Token Selection

### Relevance Realization Through Sparsity

In ARR-COC, we're computing relevance for token allocation. Sparse coding provides a principled foundation:

**Key Insight**: The most relevant tokens are those that would have non-zero coefficients in a sparse reconstruction of the information need.

### Sparse Token Allocation Architecture

```python
class SparseTokenAllocator(nn.Module):
    """
    Allocate computation budget using sparse coding principles.

    Instead of soft attention over all tokens, use HARD sparsity
    to select only the most relevant tokens for processing.
    """
    def __init__(
        self,
        embed_dim,
        num_tokens,
        budget_fraction=0.1,  # Only activate 10% of tokens
        lambda_base=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.budget_fraction = budget_fraction
        self.lambda_base = lambda_base

        # Query projection for relevance scoring
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Dictionary of "relevance patterns"
        self.relevance_dict = nn.Parameter(
            torch.randn(embed_dim, num_tokens) * 0.01
        )

    def forward(self, queries, keys, values):
        """
        Sparse token selection.

        Args:
            queries: [batch, num_queries, embed_dim]
            keys: [batch, num_tokens, embed_dim]
            values: [batch, num_tokens, embed_dim]

        Returns:
            output: [batch, num_queries, embed_dim]
            selection_mask: Which tokens were selected
        """
        batch_size = queries.shape[0]
        num_queries = queries.shape[1]

        # Project queries
        q = self.query_proj(queries)  # [batch, num_queries, embed_dim]

        # Compute relevance scores
        # Shape: [batch, num_queries, num_tokens]
        scores = torch.bmm(q, keys.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        # Apply sparse selection via soft thresholding
        # Adaptive lambda based on score distribution
        lambda_adaptive = self.lambda_base * scores.abs().mean()

        # Sparse selection (differentiable)
        selection = soft_threshold(scores, lambda_adaptive)

        # Normalize selected weights (like attention but sparse!)
        selection_sum = selection.abs().sum(dim=-1, keepdim=True) + 1e-8
        selection_normalized = selection / selection_sum

        # Aggregate values from selected tokens
        output = torch.bmm(selection_normalized, values)

        # Compute actual sparsity achieved
        sparsity = (selection == 0).float().mean()

        # Create selection mask for interpretability
        selection_mask = (selection != 0)

        return output, selection_mask, sparsity

    def get_selected_tokens(self, queries, keys):
        """
        Get indices of selected (relevant) tokens.

        Useful for interpretability and debugging.
        """
        with torch.no_grad():
            q = self.query_proj(queries)
            scores = torch.bmm(q, keys.transpose(-2, -1)) / (self.embed_dim ** 0.5)
            lambda_adaptive = self.lambda_base * scores.abs().mean()
            selection = soft_threshold(scores, lambda_adaptive)

            # Get top-k non-zero indices
            num_selected = int(self.num_tokens * self.budget_fraction)
            _, indices = selection.abs().topk(num_selected, dim=-1)

        return indices


def soft_threshold(x, threshold):
    """Differentiable soft thresholding"""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)
```

### Hierarchical Sparse Relevance

```python
class HierarchicalSparseRelevance(nn.Module):
    """
    Multi-scale sparse relevance computation.

    Coarse-to-fine relevance selection:
    1. Sparse selection at region level
    2. Sparse selection at patch level within selected regions
    3. Sparse selection at token level within selected patches
    """
    def __init__(
        self,
        embed_dim,
        num_regions=16,
        patches_per_region=16,
        tokens_per_patch=16,
        lambda_coarse=0.3,
        lambda_fine=0.1
    ):
        super().__init__()

        # Coarse level: region selection
        self.region_selector = SparseTokenAllocator(
            embed_dim, num_regions,
            budget_fraction=0.25, lambda_base=lambda_coarse
        )

        # Fine level: patch selection within regions
        self.patch_selector = SparseTokenAllocator(
            embed_dim, patches_per_region,
            budget_fraction=0.25, lambda_base=lambda_fine
        )

        # Finest level: token selection within patches
        self.token_selector = SparseTokenAllocator(
            embed_dim, tokens_per_patch,
            budget_fraction=0.25, lambda_base=lambda_fine
        )

    def forward(self, query, hierarchy):
        """
        Hierarchical sparse selection.

        Args:
            query: [batch, embed_dim] - what we're looking for
            hierarchy: Dict with 'regions', 'patches', 'tokens'

        Returns:
            selected_tokens: The sparsely selected relevant tokens
            selection_path: Which regions/patches/tokens were selected
        """
        batch_size = query.shape[0]

        # Expand query for all levels
        q = query.unsqueeze(1)  # [batch, 1, embed_dim]

        # Level 1: Select regions
        regions = hierarchy['regions']  # [batch, num_regions, embed_dim]
        _, region_mask, _ = self.region_selector(q, regions, regions)

        # Level 2: Select patches within selected regions
        # (Only process selected regions for efficiency)
        selected_region_indices = region_mask.squeeze(1).nonzero()[:, 1]
        patches = hierarchy['patches']  # [batch, num_regions, patches_per_region, embed_dim]

        # ... continue hierarchical selection

        return selected_tokens, selection_path
```

### Sparsity as Computational Budget

```python
class BudgetAwareSparseAttention(nn.Module):
    """
    Attention with explicit computational budget via sparsity.

    Key idea: Lambda controls the trade-off between
    - Reconstruction quality (using more tokens)
    - Computational cost (using fewer tokens)

    Higher lambda = sparser = faster but less accurate
    Lower lambda = denser = slower but more accurate
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, lambda_budget):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            lambda_budget: Sparsity level (higher = fewer tokens used)
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # SPARSE attention via soft thresholding
        # Instead of softmax, use L1 proximal!
        sparse_attn = soft_threshold(scores, lambda_budget)

        # Normalize (handle rows that are all zeros)
        row_sums = sparse_attn.abs().sum(dim=-1, keepdim=True) + 1e-8
        sparse_attn = sparse_attn / row_sums

        # Apply attention
        output = torch.matmul(sparse_attn, v)

        # Merge heads
        output = output.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # Compute actual FLOPs savings
        sparsity = (sparse_attn == 0).float().mean()
        theoretical_speedup = 1.0 / (1.0 - sparsity + 1e-8)

        return output, sparsity, theoretical_speedup
```

### Why This Matters for ARR-COC

1. **Principled Selection**: Sparsity provides a mathematically principled way to select relevant tokens
2. **Computational Efficiency**: Sparse operations are faster (only compute on non-zero elements)
3. **Interpretability**: Easy to see which tokens were selected (non-zero = selected)
4. **Adaptive Budget**: Lambda controls the relevance/efficiency trade-off
5. **Hierarchical**: Natural extension to multi-scale processing

**The Core Insight**: Relevance realization IS sparse coding. The brain doesn't attend to everything - it selects what matters through sparsity.

---

## Performance Considerations

### Computational Complexity

**Dense Attention**: O(n^2 * d)
**Sparse Attention with k active**: O(k * n * d) where k << n

For 10% sparsity: **10x theoretical speedup!**

### GPU Optimization

```python
# Use sparse operations when available
if sparsity > 0.9:  # Very sparse
    # Convert to sparse format
    sparse_attn = sparse_attn.to_sparse()
    output = torch.sparse.mm(sparse_attn, v)
else:
    # Dense is faster for moderate sparsity
    output = torch.mm(sparse_attn, v)
```

### Memory Efficiency

```python
# Sparse codes require less memory
# Only store (index, value) pairs for non-zero elements

def sparse_storage_size(tensor, sparsity):
    """
    Memory required for sparse storage.
    """
    num_elements = tensor.numel()
    num_nonzero = int(num_elements * (1 - sparsity))

    # Each non-zero: index (int64=8 bytes) + value (float32=4 bytes)
    sparse_bytes = num_nonzero * (8 + 4)
    dense_bytes = num_elements * 4

    return sparse_bytes, dense_bytes, dense_bytes / max(sparse_bytes, 1)
```

### Sparsity vs Accuracy Trade-off

```python
def find_optimal_lambda(model, val_loader, lambda_range, target_sparsity):
    """
    Find lambda that achieves target sparsity with minimal reconstruction error.
    """
    results = []

    for lambda_ in lambda_range:
        model.lambda_ = lambda_

        total_error = 0
        total_sparsity = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                codes = model(batch)
                error = F.mse_loss(model.reconstruct(codes), batch)
                sparsity = model.get_sparsity(codes)

                total_error += error.item()
                total_sparsity += sparsity.item()
                num_batches += 1

        avg_error = total_error / num_batches
        avg_sparsity = total_sparsity / num_batches

        results.append({
            'lambda': lambda_,
            'error': avg_error,
            'sparsity': avg_sparsity
        })

        print(f"Lambda={lambda_:.4f}: Error={avg_error:.4f}, Sparsity={avg_sparsity:.2%}")

    # Find lambda closest to target sparsity
    best = min(results, key=lambda r: abs(r['sparsity'] - target_sparsity))

    return best['lambda'], results
```

---

## Sources

**Foundational Papers**:
- [Olshausen & Field, 1996](https://www.nature.com/articles/381607a0) - "Emergence of simple-cell receptive field properties" (Nature, cited 7800+)
- [Olshausen & Field, 1997](https://www.sciencedirect.com/science/article/pii/S0042698997001697) - "Sparse coding with an overcomplete basis set" (Vision Research, cited 5000+)
- [Olshausen & Field, 2004](https://pubmed.ncbi.nlm.nih.gov/15321069/) - "Sparse coding of sensory inputs" (Current Opinion in Neurobiology, cited 2000+)

**Algorithms**:
- [Rozell et al., 2008](https://bpb-us-e1.wpmucdn.com/blogs.rice.edu/dist/c/3448/files/2014/07/neco2008.pdf) - "Sparse Coding via Thresholding and Local Competition in Neural Circuits"

**Implementations**:
- [LCA-PyTorch](https://github.com/lanl/lca-pytorch) - Los Alamos National Lab implementation (accessed 2025-11-23)
- [Sparse Autoencoders L1](https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/) - DebuggerCafe PyTorch tutorial (accessed 2025-11-23)

**Dictionary Learning**:
- [Transformer Circuits - Monosemantic Features](https://transformer-circuits.pub/2023/monosemantic-features) - Dictionary learning for LLM interpretability

**Connection to Predictive Coding**:
- [Willmore et al., 2011](https://journals.physiology.org/doi/10.1152/jn.00594.2010) - "Sparse coding in striate and extrastriate visual cortex" (J Neurophysiol)

**Theoretical Foundations**:
- Andrew Ng - [CS294A Sparse Autoencoder Lecture Notes](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) (Stanford)

---

## Summary

Sparse coding reveals a fundamental principle of neural computation: represent information using the minimum necessary resources while maintaining fidelity. This principle connects:

1. **Efficient coding**: Sparsity = compression = minimal description length
2. **Feature learning**: Gabor filters emerge from sparsity on natural images
3. **Predictive coding**: Sparse codes = latent states with Laplacian priors
4. **Attention**: Sparsity = selection = what matters = relevance
5. **Free energy**: Sparsity minimizes complexity (KL term)

**THE TRAIN STATION INSIGHT**: Sparsity, compression, and relevance selection are topologically equivalent. They're all doing the same thing - finding what matters and suppressing the rest. This is the computational foundation of attention, salience, and relevance realization itself.

For ARR-COC, sparse coding provides a principled foundation for token allocation. Instead of soft attention over all tokens, use hard sparsity to select only the truly relevant ones. This gives:
- Interpretability (see what was selected)
- Efficiency (only compute on selected)
- Principled selection (optimization-based)
- Budget control (lambda = efficiency dial)

The brain doesn't process everything equally - it selects what matters through sparsity. ARR-COC can do the same.
