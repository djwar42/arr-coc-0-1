# Neural Cellular Automata: Differentiable Model of Morphogenesis

## Overview

Neural Cellular Automata (NCA) represent a paradigm shift in deep learning: instead of designing monolithic networks, we learn **local update rules** that enable self-organization into complex global patterns. This is the computational analog of morphogenesis - how a single cell becomes an organism through purely local interactions.

**The Core Insight**: A simple neural network, applied identically to every cell, can learn to grow, maintain, and regenerate complex patterns through local communication with neighbors.

From [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca) (Mordvintsev et al., 2020):
> "Morphogenesis is one of the most striking examples of a phenomenon called self-organisation. Cells, the tiny building blocks of bodies, communicate with their neighbors to decide the shape of organs and body plans."

---

## Section 1: NCA Fundamentals

### What is a Neural Cellular Automaton?

Traditional Cellular Automata (CA) use discrete states and handcrafted rules (like Conway's Game of Life). Neural CA replace these with:
- **Continuous state vectors** (typically 16 channels)
- **Learned update rules** (small neural networks)
- **Differentiable operations** (enables gradient-based learning)

### Cell State Representation

Each cell has a state vector of k dimensions (typically k=16):
```
state = [R, G, B, alpha, hidden_1, ..., hidden_12]
```

- **Channels 0-2**: RGB color (visible output)
- **Channel 3**: Alpha (alive/dead marker, alpha > 0.1 = alive)
- **Channels 4-15**: Hidden channels (cell memory/signaling)

From [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca):
> "Hidden channels don't have a predefined meaning, and it's up to the update rule to decide what to use them for. They can be interpreted as concentrations of some chemicals, electric potentials or some other signaling mechanism."

### The Update Rule

The NCA update happens in distinct phases:

1. **Perception**: Each cell perceives its 3x3 neighborhood via Sobel filters
2. **Update computation**: Neural network processes perceived state
3. **Stochastic application**: Random subset of cells update (asynchronous)
4. **Alive masking**: Dead cells are zeroed out

This creates a **differentiable PDE approximation**:

```
ds/dt = f(s, grad_x(s), grad_y(s))
```

Where f is the learned neural network update rule.

---

## Section 2: Differentiable CA Rules

### Perception via Convolution

The key insight is using **fixed Sobel filters** to sense gradients:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Perception(nn.Module):
    """Perceive local neighborhood using Sobel filters."""

    def __init__(self, n_channels=16):
        super().__init__()
        self.n_channels = n_channels

        # Sobel filters for gradient estimation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 8.0

        sobel_y = sobel_x.T

        # Identity filter (pass through state)
        identity = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)

        # Stack filters: [3, 1, 3, 3]
        filters = torch.stack([identity, sobel_x, sobel_y])

        # Expand for all channels: [3*n_channels, 1, 3, 3]
        self.filters = filters.repeat(n_channels, 1, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] cell states
        Returns:
            perception: [B, 3*C, H, W] - state + gradients
        """
        B, C, H, W = x.shape

        # Apply filters with groups=C (depthwise)
        # This gives us identity, grad_x, grad_y for each channel
        perception = F.conv2d(
            x,
            self.filters.to(x.device),
            padding=1,
            groups=C
        )

        return perception
```

### The Update Network

A small MLP processes the perception vector:

```python
class UpdateRule(nn.Module):
    """Neural network that computes state updates."""

    def __init__(self, n_channels=16, hidden_dim=128):
        super().__init__()

        # Input: 3 * n_channels (state + grad_x + grad_y)
        perception_dim = 3 * n_channels

        self.net = nn.Sequential(
            nn.Conv2d(perception_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_channels, 1, bias=False)
        )

        # Initialize final layer to zero for "do nothing" initial behavior
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, perception):
        """
        Args:
            perception: [B, 3*C, H, W]
        Returns:
            update: [B, C, H, W] - incremental state update
        """
        return self.net(perception)
```

### Stochastic Updates (Breaking Synchrony)

Real cells don't have a global clock. We simulate asynchrony:

```python
def stochastic_update(state, update, update_rate=0.5):
    """Apply updates stochastically to break synchrony.

    Args:
        state: [B, C, H, W] current state
        update: [B, C, H, W] computed update
        update_rate: probability each cell updates

    Returns:
        new_state: [B, C, H, W]
    """
    B, C, H, W = state.shape

    # Random mask per cell (not per channel)
    mask = (torch.rand(B, 1, H, W, device=state.device) < update_rate).float()

    return state + update * mask
```

### Alive Masking

Cells only live if they or their neighbors have alpha > threshold:

```python
def alive_masking(state, alpha_threshold=0.1):
    """Zero out cells that are not alive or adjacent to alive cells.

    Args:
        state: [B, C, H, W]
        alpha_threshold: threshold for "mature" cells

    Returns:
        masked_state: [B, C, H, W]
    """
    # Get alpha channel
    alpha = state[:, 3:4, :, :]

    # Max pool to include neighbors
    alive = F.max_pool2d(alpha, 3, stride=1, padding=1) > alpha_threshold

    return state * alive.float()
```

---

## Section 3: Growing Patterns

### Complete NCA Model

```python
class NeuralCA(nn.Module):
    """Complete Neural Cellular Automaton."""

    def __init__(self, n_channels=16, hidden_dim=128, update_rate=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.update_rate = update_rate

        self.perception = Perception(n_channels)
        self.update_rule = UpdateRule(n_channels, hidden_dim)

    def forward(self, x, steps=1, training=True):
        """Run NCA for multiple steps.

        Args:
            x: [B, C, H, W] initial state
            steps: number of CA iterations
            training: use stochastic updates if True

        Returns:
            final_state: [B, C, H, W]
        """
        for _ in range(steps):
            # Perceive neighborhood
            perception = self.perception(x)

            # Compute update
            update = self.update_rule(perception)

            # Apply stochastically during training
            if training:
                x = stochastic_update(x, update, self.update_rate)
            else:
                x = x + update

            # Mask dead cells
            x = alive_masking(x)

        return x

    def seed_state(self, batch_size, height, width, device='cuda'):
        """Create initial seed state (single cell in center).

        Returns:
            seed: [B, C, H, W] with single alive cell in center
        """
        state = torch.zeros(batch_size, self.n_channels, height, width, device=device)

        # Set center cell alive with hidden channels = 1
        center_h, center_w = height // 2, width // 2
        state[:, 3:, center_h, center_w] = 1.0  # alpha and hidden = 1

        return state
```

### Training Loop with Pool Sampling

The key innovation is the **sample pool** for long-term stability:

```python
class SamplePool:
    """Pool of states for training temporal stability."""

    def __init__(self, size, state_shape):
        self.size = size
        self.pool = [None] * size
        self.state_shape = state_shape

    def sample(self, batch_size):
        """Sample batch from pool."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = []
        for i in indices:
            if self.pool[i] is None:
                # Initialize with seed
                batch.append(self.create_seed())
            else:
                batch.append(self.pool[i].clone())
        return indices, torch.stack(batch)

    def update(self, indices, states):
        """Update pool with final states."""
        for i, state in zip(indices, states):
            self.pool[i] = state.detach().cpu()


def train_nca(model, target_image, pool, num_iterations=10000):
    """Train NCA to grow target pattern.

    Args:
        model: NeuralCA instance
        target_image: [C, H, W] target RGBA image
        pool: SamplePool instance
        num_iterations: training steps
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    target = target_image.unsqueeze(0)  # [1, C, H, W]

    for iteration in range(num_iterations):
        # Sample from pool
        indices, batch = pool.sample(batch_size=32)
        batch = batch.to(device)

        # Replace highest loss sample with seed (prevent forgetting)
        with torch.no_grad():
            losses = compute_loss(model(batch, steps=1), target)
            worst_idx = losses.argmax()
            batch[worst_idx] = model.seed_state(1, H, W, device)[0]

        # Run for random number of steps
        steps = np.random.randint(64, 96)
        output = model(batch, steps=steps)

        # Compute loss (L2 on RGBA)
        loss = F.mse_loss(output[:, :4], target.expand(32, -1, -1, -1))

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient normalization for stability
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data / (param.grad.data.norm() + 1e-8)

        optimizer.step()

        # Update pool with output states
        pool.update(indices, output)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
```

### Learning to Regenerate

To learn regeneration, we **damage samples during training**:

```python
def damage_state(state, damage_radius=5):
    """Apply circular damage to state.

    Args:
        state: [B, C, H, W]
        damage_radius: radius of damage circle

    Returns:
        damaged: [B, C, H, W]
    """
    B, C, H, W = state.shape
    damaged = state.clone()

    for b in range(B):
        # Random center within pattern
        cy = np.random.randint(H // 4, 3 * H // 4)
        cx = np.random.randint(W // 4, 3 * W // 4)

        # Create circular mask
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((x - cx)**2 + (y - cy)**2) < damage_radius**2

        # Zero out damaged region
        damaged[b, :, mask] = 0.0

    return damaged


def train_with_regeneration(model, target, pool):
    """Training with damage for regeneration capability."""

    for iteration in range(num_iterations):
        indices, batch = pool.sample(32)

        # Damage some samples (lowest loss = best formed)
        with torch.no_grad():
            losses = compute_loss(model(batch, steps=1), target)
            sorted_indices = losses.argsort()

            # Damage the 3 best-formed samples
            for i in sorted_indices[:3]:
                batch[i] = damage_state(batch[i:i+1])[0]

            # Replace worst with seed
            batch[sorted_indices[-1]] = model.seed_state(1, H, W, device)[0]

        # Train as usual
        output = model(batch, steps=np.random.randint(64, 96))
        loss = F.mse_loss(output[:, :4], target)

        # ... optimization ...
```

---

## Section 4: Complete Implementation

### Full Training Script

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================
# NEURAL CELLULAR AUTOMATA - FULL IMPLEMENTATION
# ============================================

class NeuralCA(nn.Module):
    """Neural Cellular Automaton for morphogenesis."""

    def __init__(self, n_channels=16, hidden_dim=128, update_rate=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.update_rate = update_rate

        # Perception filters (fixed)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

        # Update rule network
        perception_dim = 3 * n_channels
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_channels, 1, bias=False)
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, x):
        """Compute perception vector for each cell."""
        # Depthwise convolution for gradients
        def depthwise_conv(x, kernel):
            return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1),
                          padding=1, groups=x.shape[1])

        grad_x = depthwise_conv(x, self.sobel_x)
        grad_y = depthwise_conv(x, self.sobel_y)

        return torch.cat([x, grad_x, grad_y], dim=1)

    def forward(self, x, steps=1, training=True):
        """Run NCA for multiple steps."""
        for _ in range(steps):
            # Perceive
            perception = self.perceive(x)

            # Compute update
            update = self.update_net(perception)

            # Stochastic update
            if training:
                mask = (torch.rand_like(x[:, :1]) < self.update_rate).float()
                x = x + update * mask
            else:
                x = x + update

            # Alive masking
            alpha = x[:, 3:4]
            alive = F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
            x = x * alive.float()

        return x

    def seed(self, batch_size, height, width, device):
        """Create seed state."""
        state = torch.zeros(batch_size, self.n_channels, height, width, device=device)
        state[:, 3:, height//2, width//2] = 1.0
        return state


def load_target(path, size=64):
    """Load and preprocess target image."""
    img = Image.open(path).convert('RGBA').resize((size, size))
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)


def train():
    """Full training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load target
    target = load_target('target.png', size=64).to(device)
    target = target.unsqueeze(0)  # [1, 4, 64, 64]

    # Model
    model = NeuralCA(n_channels=16, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Sample pool
    pool_size = 1024
    pool = [None] * pool_size

    # Training
    for step in range(10000):
        # Sample batch
        batch_size = 32
        indices = np.random.choice(pool_size, batch_size, replace=False)

        batch = []
        for i in indices:
            if pool[i] is None:
                batch.append(model.seed(1, 64, 64, device)[0])
            else:
                batch.append(pool[i].clone().to(device))
        batch = torch.stack(batch)

        # Replace worst with seed
        with torch.no_grad():
            output = model(batch.clone(), steps=1)
            losses = ((output[:, :4] - target)**2).mean(dim=[1,2,3])
            worst = losses.argmax()
            batch[worst] = model.seed(1, 64, 64, device)[0]

            # Damage best samples for regeneration
            if step > 2000:
                best = losses.argsort()[:3]
                for i in best:
                    # Random circular damage
                    cy, cx = np.random.randint(16, 48, size=2)
                    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
                    mask = ((x - cx)**2 + (y - cy)**2) < 25
                    batch[i, :, mask.to(device)] = 0.0

        # Forward pass
        n_steps = np.random.randint(64, 96)
        output = model(batch, steps=n_steps)

        # Loss
        loss = F.mse_loss(output[:, :4], target.expand(batch_size, -1, -1, -1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient normalization
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= (p.grad.norm() + 1e-8)

        optimizer.step()

        # Update pool
        for i, idx in enumerate(indices):
            pool[idx] = output[i].detach().cpu()

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    return model


if __name__ == '__main__':
    model = train()
    torch.save(model.state_dict(), 'nca_model.pt')
```

---

## Section 5: Performance Optimization

### GPU Efficiency

From [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca):
> "Each cell would require approximately 10Kb of ROM to store the 'cell genome': neural network weights and the control code, and about 256 bytes of RAM for the cell state."

**Key optimizations**:

1. **Depthwise separable convolutions**: Perception uses grouped convolutions
2. **Small network**: Only ~8,000 parameters per model
3. **Quantization**: WebGL demo uses 8-bit weights
4. **Batched updates**: Process all cells in parallel

### Memory Considerations

```python
# Memory per cell:
# - State: 16 channels * 4 bytes = 64 bytes
# - Perception: 48 channels * 4 bytes = 192 bytes
# - Hidden: 128 channels * 4 bytes = 512 bytes
# Total intermediate: ~768 bytes per cell

# For 64x64 grid:
# - 4096 cells * 768 bytes = 3.1 MB per sample
# - Batch of 32 = ~100 MB (fits easily in GPU)
```

### Training Speed

```python
# Typical training metrics:
# - 10,000 iterations
# - ~3 hours on single GPU
# - Batch size 32
# - 64-96 steps per iteration

# Inference is VERY fast:
# - Single step: ~0.1ms on GPU
# - Can run 10,000+ steps per second
```

### Quantization for Deployment

```python
def quantize_model(model, bits=8):
    """Quantize model for efficient deployment.

    From the Distill article:
    "We use the non-linear arctan function to compress
    the unbounded activation values to the bounded segment,
    preserving the highest accuracy around zero."
    """
    quantized = {}
    scale = 2**(bits - 1) - 1

    for name, param in model.named_parameters():
        # Compress to bounded range
        compressed = torch.atan(param) / (np.pi / 2)  # [-1, 1]
        quantized[name] = (compressed * scale).round().to(torch.int8)

    return quantized
```

---

## Section 6: TRAIN STATION - NCA = Morphogenesis = Self-Organization = Emergence

### The Grand Unification

**NCA reveals deep connections between:**

```
              EMERGENCE
                  |
    +-------------+-------------+
    |             |             |
Self-Organization  Morphogenesis  Autopoiesis
    |             |             |
    +-------------+-------------+
                  |
          NEURAL CELLULAR AUTOMATA
                  |
    +-------------+-------------+
    |             |             |
Message Passing   PDEs/Fields   Free Energy
    |             |             |
GNN/Attention   Reaction-Diffusion  Active Inference
```

### Connection 1: NCA = Morphogenesis

From [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca):
> "Most multicellular organisms begin their life as a single egg cell - a single cell whose progeny reliably self-assemble into highly complex anatomies."

**The parallel is exact:**
- Biological cell = NCA cell
- DNA/genome = learned update rule
- Chemical gradients = Sobel-perceived state gradients
- Morphogen signaling = hidden channel communication
- Pattern formation = loss minimization

### Connection 2: NCA = Self-Organization

From [Self-Organising Textures](https://distill.pub/selforg/2021/textures):
> "Each cell must be able to take on the role of any other cell - as a result they tend to generalize well to unseen situations."

**Self-organization principles:**
- No central controller
- Identical local rules everywhere
- Global pattern emerges from local interactions
- Robustness through redundancy

### Connection 3: NCA = Reaction-Diffusion

The NCA update rule is a **learned** reaction-diffusion system:

```
Standard Turing pattern:
  du/dt = D_u * laplacian(u) + f(u, v)
  dv/dt = D_v * laplacian(v) + g(u, v)

NCA (generalized):
  ds/dt = neural_net(s, grad_x(s), grad_y(s))
```

The NCA learns WHAT reaction-diffusion dynamics produce the target pattern.

### Connection 4: NCA = Message Passing

NCAs are **graph neural networks** on a grid:
- Nodes = cells
- Edges = 3x3 neighborhoods
- Message = perceived gradient
- Aggregation = convolution
- Update = learned MLP

This connects to:
- Belief propagation
- Predictive coding
- Bioelectric signaling

### Connection 5: NCA = Free Energy Minimization

The training process minimizes a free energy functional:
- **Loss** = prediction error = surprise
- **Pattern stability** = attractor in state space
- **Regeneration** = returning to free energy minimum after perturbation

From the active inference perspective:
- Cells are minimizing prediction error
- The "prediction" is the target pattern
- Local actions reduce global free energy

### The Profound Insight

**What makes a single cell become a lizard?**

The NCA demonstrates that you only need:
1. Local sensing (gradients)
2. Local communication (hidden channels)
3. Local identical rules (shared weights)
4. Selection pressure (loss function)

**The genome encodes the local rule, not the global pattern!**

This is why organisms can:
- Regenerate (return to attractor)
- Scale (rule works at any grid size)
- Adapt (local robustness = global robustness)

---

## Section 7: ARR-COC-0-1 Connection - Self-Organizing Relevance

### How NCA Principles Apply to Token Allocation

**The Core Analogy**:
- NCA cells = image tokens
- Cell state = token relevance score
- Perception = attention to neighbors
- Update rule = learned relevance computation
- Target pattern = optimal relevance distribution

### Self-Organizing Token Selection

```python
class RelevanceNCA(nn.Module):
    """NCA-style self-organizing token relevance.

    Instead of centrally computing relevance,
    let tokens self-organize to find important regions.
    """

    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()

        # Each token perceives its spatial neighbors
        self.perceive = nn.Conv2d(embed_dim, 3 * embed_dim, 3, padding=1, groups=embed_dim)

        # Update rule (same for all tokens)
        self.update = nn.Sequential(
            nn.Conv2d(3 * embed_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1)  # Output: relevance update
        )

    def forward(self, x, steps=10):
        """Self-organize relevance scores.

        Args:
            x: [B, C, H, W] token features
            steps: NCA iterations

        Returns:
            relevance: [B, 1, H, W] self-organized scores
        """
        B, C, H, W = x.shape

        # Initialize relevance uniformly
        relevance = torch.ones(B, 1, H, W, device=x.device) / (H * W)

        for _ in range(steps):
            # Combine features with current relevance
            state = x * relevance

            # Perceive neighborhood
            perception = self.perceive(state)

            # Compute update
            delta = self.update(perception)

            # Stochastic update for robustness
            mask = (torch.rand_like(relevance) < 0.5).float()
            relevance = relevance + delta * mask

            # Normalize to sum to 1
            relevance = F.softmax(relevance.view(B, -1), dim=-1).view(B, 1, H, W)

        return relevance
```

### Benefits of Self-Organizing Relevance

1. **Robustness**: Like regenerating organisms, can recover from perturbations
2. **Adaptability**: Same rule works for different image sizes
3. **Efficiency**: Parallel local computation
4. **Emergence**: Complex relevance patterns from simple rules

### Integration with ARR-COC

```python
class NCAGuidedAttention(nn.Module):
    """Use NCA-style self-organization to guide attention."""

    def __init__(self, embed_dim, num_heads, nca_steps=5):
        super().__init__()
        self.nca = RelevanceNCA(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.nca_steps = nca_steps

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        H = W = int(N**0.5)

        # Reshape for NCA
        x_spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Self-organize relevance
        relevance = self.nca(x_spatial, steps=self.nca_steps)

        # Use relevance to weight attention
        relevance_flat = relevance.view(B, -1, 1)  # [B, N, 1]

        # Weighted attention
        x_weighted = x * relevance_flat
        output, _ = self.attention(x_weighted, x_weighted, x_weighted)

        return output
```

### The Vision

**From centralized to self-organized relevance:**

Instead of:
- Single forward pass computes all relevance
- Global pooling loses spatial information
- Fixed computation regardless of content

We get:
- Iterative refinement through local interactions
- Spatial structure preserved
- Computation adapts to content complexity

This mirrors how biological attention works - not a single computation, but an emergent property of many local interactions reaching consensus.

---

## Sources

**Primary Sources - Distill Publications:**
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca) - Mordvintsev et al., 2020 (DOI: 10.23915/distill.00023)
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures) - Niklasson et al., 2021 (DOI: 10.23915/distill.00027.003)

**Web Research (accessed 2025-11-23):**
- [GitHub: MonashDeepNeuron/Neural-Cellular-Automata](https://github.com/MonashDeepNeuron/Neural-Cellular-Automata) - Implementation reference
- [Studying Growth with Neural Cellular Automata](http://greydanus.github.io/2022/05/24/studying-growth/) - Sam Greydanus
- [Attention-based Neural Cellular Automata](https://proceedings.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf) - NeurIPS 2022
- [arXiv: A Path to Universal Neural Cellular Automata](https://arxiv.org/html/2505.13058v1) - 2025
- [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/) - Google Research
- [HyperNCA: Growing Neural Networks with NCA](https://arxiv.org/pdf/2204.11674) - Najarro et al., 2022

**Related Works:**
- Turing, A.M. (1952) "The chemical basis of morphogenesis" - Original pattern formation theory
- Gray-Scott model - Reaction-diffusion reference
- Lenia - Chan, B.W. (2019) - Continuous CA lifeforms

**Implementation References:**
- [Colab Notebook](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb) - Official implementation
- WebGL demo code in original articles
