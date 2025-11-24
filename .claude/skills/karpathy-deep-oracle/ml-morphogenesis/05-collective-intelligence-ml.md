# Collective Intelligence in Machine Learning

## Overview

Collective intelligence studies the group intelligence that emerges from interactions of many individuals. In machine learning, this manifests as **ensembles**, **mixture of experts (MoE)**, **swarm optimization**, and **emergent behavior** from simple rules. The deep insight is that ALL of these are the same pattern: multiple agents making local decisions that aggregate into global intelligence.

From [Collective Intelligence for Deep Learning: A Survey](https://arxiv.org/abs/2111.14377) (Ha & Tang, 2021, cited 126 times):
> "Collective behavior, commonly observed in nature, tends to produce systems that are robust, adaptable, and have less rigid assumptions about the environment configuration."

---

## Section 1: Ensemble Methods as Collective Decision-Making

### The Wisdom of Crowds Principle

Ensemble methods are the simplest form of collective intelligence in ML:
- Multiple models vote on predictions
- Diversity reduces correlated errors
- Aggregation smooths individual mistakes

### Types of Ensembles

**Bagging (Bootstrap Aggregating)**:
- Train models on bootstrap samples
- Average predictions (regression) or vote (classification)
- Random Forest = bagging + feature randomization

**Boosting**:
- Sequential training, focus on mistakes
- AdaBoost, Gradient Boosting, XGBoost
- Weak learners become strong collectively

**Stacking**:
- Train meta-learner on base model predictions
- Learns optimal combination strategy
- Two-level collective decision

### PyTorch Ensemble Implementation

```python
import torch
import torch.nn as nn
from typing import List

class EnsembleCollective(nn.Module):
    """
    Ensemble as collective intelligence.
    Multiple models vote, collective decides.

    This is the simplest TRAIN STATION:
    Ensemble = voting = democracy = swarm consensus
    """

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = 'mean',  # 'mean', 'vote', 'weighted'
        learnable_weights: bool = False
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.n_models = len(models)

        if learnable_weights:
            # Learnable combination weights
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        else:
            self.register_buffer('weights', torch.ones(self.n_models) / self.n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gather predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Stack: [n_models, batch, ...]
        stacked = torch.stack(predictions, dim=0)

        if self.aggregation == 'mean':
            # Simple averaging - democratic vote
            return stacked.mean(dim=0)

        elif self.aggregation == 'weighted':
            # Weighted combination - expertise weighting
            weights = torch.softmax(self.weights, dim=0)
            # [n_models, 1, 1, ...] for broadcasting
            weight_shape = [self.n_models] + [1] * (stacked.dim() - 1)
            weights = weights.view(*weight_shape)
            return (stacked * weights).sum(dim=0)

        elif self.aggregation == 'vote':
            # Hard voting for classification
            votes = stacked.argmax(dim=-1)  # [n_models, batch]
            # Mode across models
            return torch.mode(votes, dim=0).values

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measure collective disagreement.
        High disagreement = high uncertainty.

        TRAIN STATION: Uncertainty = diversity = entropy
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)

        # Variance across models
        variance = stacked.var(dim=0)

        return variance


class DiverseEnsembleTrainer:
    """
    Train ensemble with diversity encouragement.

    Key insight: Collective intelligence requires DIVERSITY.
    If all models agree, no wisdom of crowds!
    """

    def __init__(
        self,
        ensemble: EnsembleCollective,
        diversity_weight: float = 0.1
    ):
        self.ensemble = ensemble
        self.diversity_weight = diversity_weight

    def diversity_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Encourage models to disagree (within reason).

        Negative correlation learning:
        Models should make different errors.
        """
        n = len(predictions)
        if n < 2:
            return torch.tensor(0.0)

        # Compute pairwise correlations
        total_correlation = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Flatten predictions
                p_i = predictions[i].flatten()
                p_j = predictions[j].flatten()

                # Center
                p_i_centered = p_i - p_i.mean()
                p_j_centered = p_j - p_j.mean()

                # Correlation
                correlation = (p_i_centered * p_j_centered).sum()
                correlation = correlation / (p_i_centered.norm() * p_j_centered.norm() + 1e-8)

                total_correlation += correlation
                count += 1

        # Penalize high correlation (encourage diversity)
        return total_correlation / count if count > 0 else torch.tensor(0.0)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        optimizers: List[torch.optim.Optimizer]
    ) -> dict:
        """
        Train step with diversity regularization.
        """
        # Get individual predictions
        predictions = []
        for model in self.ensemble.models:
            pred = model(x)
            predictions.append(pred)

        # Individual losses
        individual_losses = []
        for pred in predictions:
            loss = criterion(pred, y)
            individual_losses.append(loss)

        # Diversity loss
        div_loss = self.diversity_loss(predictions)

        # Total loss per model
        total_losses = []
        for i, loss in enumerate(individual_losses):
            # Each model: task loss - diversity bonus
            # Negative because we WANT diversity
            total = loss + self.diversity_weight * div_loss
            total_losses.append(total)

        # Backward for each model
        for optimizer, loss in zip(optimizers, total_losses):
            optimizer.zero_grad()

        # Sum and backward
        total_loss = sum(total_losses)
        total_loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'diversity_loss': div_loss.item(),
            'individual_losses': [l.item() for l in individual_losses]
        }
```

### Performance Notes

**Memory**: O(n_models * model_size) - all models in memory
**Compute**: O(n_models * forward_pass) - can parallelize
**Latency**: Limited by slowest model (if parallel) or sum (if sequential)

**Optimization Tips**:
- Use model parallelism: different models on different GPUs
- Batch predictions across models
- Prune ensemble to smallest subset maintaining accuracy

---

## Section 2: Mixture of Experts (MoE)

### Core Concept

MoE is **learned routing** rather than fixed ensemble aggregation:
- Gating network decides which experts to use per input
- Sparse activation: only k experts compute per token
- Scales model capacity without proportional compute increase

From [Mixture of Experts Explained](https://huggingface.co/blog/moe) (Hugging Face, 2023):
> "MoE enables models to be pretrained with far less compute, which means you can dramatically scale up the model or dataset size with the same compute budget as a dense model."

### MoE Architecture

**Components**:
1. **Expert Networks**: Typically FFN layers (can be any architecture)
2. **Gating Network (Router)**: Learns which experts for which inputs
3. **Combiner**: Aggregates expert outputs

**Key Equation**:
```
y = sum_i G(x)_i * E_i(x)
```
where G is the gating function and E_i are experts.

### PyTorch MoE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Expert(nn.Module):
    """
    Single expert network.
    Each expert specializes in different input patterns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKGating(nn.Module):
    """
    Top-K gating with load balancing.

    TRAIN STATION CONNECTION:
    Gating = attention = precision weighting = relevance selection

    The router IS an attention mechanism:
    - Query: input token
    - Keys: expert embeddings (implicit in W_gate)
    - Values: expert outputs
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1.0,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        # Gating network
        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            gates: [batch, top_k] - softmax weights for selected experts
            indices: [batch, top_k] - which experts selected
            load_balance_loss: scalar - auxiliary loss for balanced routing
        """
        batch_size = x.size(0)

        # Compute logits
        logits = self.w_gate(x)  # [batch, num_experts]

        # Add noise during training (exploration)
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * F.softplus(self.w_noise(x))
            logits = logits + noise * self.noise_std

        # Top-K selection
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        # Softmax over selected experts only
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Load balancing loss
        # Encourages uniform expert utilization
        load_balance_loss = self._compute_load_balance_loss(logits, top_k_indices)

        return top_k_gates, top_k_indices, load_balance_loss

    def _compute_load_balance_loss(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary loss to prevent expert collapse.

        Without this, model learns to always use same experts.
        This is the "democracy enforcement" of collective intelligence!
        """
        batch_size = logits.size(0)

        # Fraction of tokens routed to each expert
        # Count how often each expert is selected
        expert_mask = F.one_hot(indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # [batch, num_experts]

        # Fraction per expert
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        fraction_tokens = tokens_per_expert / (batch_size * self.top_k)

        # Mean routing probability to each expert
        routing_probs = F.softmax(logits, dim=-1)
        mean_routing_prob = routing_probs.mean(dim=0)  # [num_experts]

        # Load balance loss = dot product of fractions
        # Minimized when both are uniform
        loss = (fraction_tokens * mean_routing_prob).sum() * self.num_experts

        return loss


class MixtureOfExperts(nn.Module):
    """
    Full MoE layer with sparse expert activation.

    COLLECTIVE INTELLIGENCE INSIGHT:
    Each expert is a "specialist" in the collective.
    The router is the "coordinator" that assembles the right team.

    This is EXACTLY like:
    - Cell differentiation (each cell type = expert)
    - Division of labor in ant colonies
    - Brain regions specializing in different functions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = TopKGating(
            input_dim, num_experts, top_k,
            capacity_factor=capacity_factor
        )

        # For tracking expert utilization
        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse expert computation.

        Args:
            x: [batch, input_dim]

        Returns:
            output: [batch, output_dim]
            aux_loss: load balancing loss (if requested)
        """
        batch_size = x.size(0)

        # Get routing decisions
        gates, indices, load_balance_loss = self.gate(x, self.training)
        # gates: [batch, top_k]
        # indices: [batch, top_k]

        # Compute expert outputs (sparse computation)
        # Only compute for experts that are selected
        expert_outputs = torch.zeros(
            batch_size, self.top_k, x.size(-1),
            device=x.device, dtype=x.dtype
        )

        # Efficient batched computation
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # [batch]

            # Group by expert for efficient batching
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    expert_outputs[mask, k] = expert_output

                    # Track utilization
                    if self.training:
                        self.expert_counts[e] += mask.sum().item()

        # Combine expert outputs with gates
        # gates: [batch, top_k, 1] for broadcasting
        gates_expanded = gates.unsqueeze(-1)
        output = (expert_outputs * gates_expanded).sum(dim=1)

        if return_aux_loss:
            aux_loss = load_balance_loss * self.aux_loss_weight
            return output, aux_loss

        return output, None

    def get_expert_utilization(self) -> torch.Tensor:
        """
        Return fraction of tokens sent to each expert.
        Useful for monitoring collective behavior.
        """
        total = self.expert_counts.sum()
        if total == 0:
            return torch.ones(self.num_experts) / self.num_experts
        return self.expert_counts / total

    def reset_expert_counts(self):
        """Reset utilization tracking."""
        self.expert_counts.zero_()


class SwitchTransformerLayer(nn.Module):
    """
    Switch Transformer style: route to single expert (top-1).

    Key insight from Switch Transformers paper:
    - Routing to 1 expert works as well as 2+
    - Simpler routing = more stable training
    - Capacity factor handles overflow

    TRAIN STATION:
    Switch = binary decision = discrete routing = cell fate decision
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        # Expert capacity per batch
        # = (tokens_per_batch / num_experts) * capacity_factor

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            router_loss: auxiliary loss
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Flatten to [num_tokens, d_model]
        x_flat = x.view(-1, d_model)

        # Route
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing
        expert_indices = router_probs.argmax(dim=-1)  # [num_tokens]
        expert_gate = router_probs.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)

        # Compute capacity
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        # Dispatch to experts with capacity constraint
        output = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            mask = (expert_indices == e)
            token_indices = mask.nonzero(as_tuple=True)[0]

            # Capacity constraint
            if len(token_indices) > capacity:
                if self.drop_tokens:
                    # Drop overflow tokens (use residual)
                    token_indices = token_indices[:capacity]
                # else: process all (for inference)

            if len(token_indices) > 0:
                expert_input = x_flat[token_indices]
                expert_output = self.experts[e](expert_input)
                # Weight by routing probability
                gate_values = expert_gate[token_indices].unsqueeze(-1)
                output[token_indices] = expert_output * gate_values

        # Router z-loss for stability
        router_z_loss = (router_logits ** 2).mean()

        # Load balance loss
        # Fraction of tokens per expert
        tokens_per_expert = F.one_hot(expert_indices, self.num_experts).float().sum(0)
        fraction_tokens = tokens_per_expert / num_tokens
        # Mean routing prob per expert
        mean_router_prob = router_probs.mean(0)
        # Loss
        load_balance_loss = (fraction_tokens * mean_router_prob).sum() * self.num_experts

        total_aux_loss = load_balance_loss + 0.001 * router_z_loss

        # Reshape output
        output = output.view(batch_size, seq_len, d_model)

        return output, total_aux_loss
```

### Performance Notes

**Memory**: O(num_experts * expert_size) - all experts loaded
**Compute**: O(top_k * expert_compute) - sparse activation
**Throughput**: Higher than dense model with same parameters

**Key Trade-offs**:
- More experts = better quality but diminishing returns after 256
- Higher capacity factor = better quality but more communication
- Top-1 vs Top-2: Top-1 is simpler and similarly effective

From [Switch Transformers](https://arxiv.org/abs/2101.03961):
> "Switch Transformers achieved a 4x pre-train speed-up over T5-XXL"

---

## Section 3: Swarm Intelligence Optimization

### Particle Swarm Optimization (PSO)

PSO optimizes by simulating a swarm:
- Each particle = candidate solution
- Particles move based on personal best and global best
- Collective exploration finds global optima

### PSO for Neural Network Training

```python
import torch
import torch.nn as nn
from typing import Callable, List
import copy

class Particle:
    """
    Single particle in the swarm.

    TRAIN STATION:
    Particle = agent = explorer = cell in tissue

    Each particle:
    - Has position (network weights)
    - Has velocity (update direction)
    - Remembers personal best
    - Knows about global best
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device
    ):
        self.device = device

        # Position = flattened weights
        self.position = self._flatten_params(model)

        # Velocity
        self.velocity = torch.zeros_like(self.position)

        # Personal best
        self.best_position = self.position.clone()
        self.best_fitness = float('inf')

    def _flatten_params(self, model: nn.Module) -> torch.Tensor:
        """Flatten all model parameters into single vector."""
        return torch.cat([p.data.flatten() for p in model.parameters()])

    def _unflatten_params(self, flat: torch.Tensor, model: nn.Module):
        """Unflatten vector back into model parameters."""
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = flat[idx:idx + numel].view(p.shape)
            idx += numel


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for neural networks.

    COLLECTIVE INTELLIGENCE:
    - No gradients needed!
    - Population-based exploration
    - Emergent optimization from simple rules

    TRAIN STATION CONNECTION:
    PSO = evolution = exploration-exploitation = free energy minimization

    Each particle minimizes its "free energy" (loss)
    while sharing information with neighbors.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        num_particles: int = 20,
        w: float = 0.7,      # Inertia weight
        c1: float = 1.5,     # Cognitive (personal best) weight
        c2: float = 1.5,     # Social (global best) weight
        device: torch.device = torch.device('cpu')
    ):
        self.model_fn = model_fn
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.device = device

        # Create reference model for param shapes
        self.reference_model = model_fn().to(device)

        # Initialize swarm
        self.particles = []
        for _ in range(num_particles):
            model = model_fn().to(device)
            # Random initialization
            for p in model.parameters():
                p.data = torch.randn_like(p.data) * 0.1
            particle = Particle(model, device)
            self.particles.append(particle)

        # Global best
        self.global_best_position = self.particles[0].position.clone()
        self.global_best_fitness = float('inf')

    def evaluate_fitness(
        self,
        particle: Particle,
        loss_fn: Callable[[nn.Module], torch.Tensor]
    ) -> float:
        """
        Evaluate particle fitness (lower is better).

        Fitness = loss = surprise = free energy
        """
        # Load weights into reference model
        particle._unflatten_params(particle.position, self.reference_model)

        # Compute loss
        with torch.no_grad():
            fitness = loss_fn(self.reference_model).item()

        return fitness

    def step(
        self,
        loss_fn: Callable[[nn.Module], torch.Tensor]
    ) -> dict:
        """
        One PSO iteration.

        The collective behavior:
        1. Each particle evaluates its position
        2. Updates personal best
        3. Updates global best
        4. Moves based on personal + social information
        """
        # Evaluate all particles
        for particle in self.particles:
            fitness = self.evaluate_fitness(particle, loss_fn)

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.clone()

            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.clone()

        # Update velocities and positions
        for particle in self.particles:
            r1 = torch.rand_like(particle.velocity)
            r2 = torch.rand_like(particle.velocity)

            # Velocity update
            # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)

            particle.velocity = self.w * particle.velocity + cognitive + social

            # Position update
            particle.position = particle.position + particle.velocity

        return {
            'global_best_fitness': self.global_best_fitness,
            'mean_fitness': sum(p.best_fitness for p in self.particles) / len(self.particles)
        }

    def get_best_model(self) -> nn.Module:
        """Return model with global best weights."""
        model = self.model_fn().to(self.device)
        # Unflatten global best into model
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = self.global_best_position[idx:idx + numel].view(p.shape)
            idx += numel
        return model

    def train(
        self,
        loss_fn: Callable[[nn.Module], torch.Tensor],
        num_iterations: int = 100,
        verbose: bool = True
    ) -> nn.Module:
        """
        Full PSO training loop.

        No gradients! Pure collective exploration.
        """
        for i in range(num_iterations):
            stats = self.step(loss_fn)

            if verbose and i % 10 == 0:
                print(f"Iteration {i}: "
                      f"Global best = {stats['global_best_fitness']:.4f}, "
                      f"Mean = {stats['mean_fitness']:.4f}")

        return self.get_best_model()


class HybridPSOGradient:
    """
    Hybrid: PSO for global search + gradient for local refinement.

    Best of both worlds:
    - PSO explores broadly (avoids local minima)
    - Gradient descent refines locally (efficient)

    TRAIN STATION:
    This is like morphogenesis:
    - Bioelectric patterns (PSO) guide global structure
    - Local chemical gradients (backprop) refine details
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        num_particles: int = 10,
        gradient_steps_per_pso: int = 10,
        device: torch.device = torch.device('cpu')
    ):
        self.pso = ParticleSwarmOptimizer(
            model_fn, num_particles, device=device
        )
        self.gradient_steps = gradient_steps_per_pso
        self.device = device

    def train(
        self,
        train_loader,
        criterion: nn.Module,
        num_pso_iterations: int = 50,
        lr: float = 0.01
    ) -> nn.Module:
        """
        Alternating PSO and gradient descent.
        """
        def loss_fn(model):
            model.eval()
            total_loss = 0
            count = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                total_loss += criterion(out, y)
                count += 1
                if count >= 5:  # Sample subset for speed
                    break
            return total_loss / count

        for pso_iter in range(num_pso_iterations):
            # PSO step
            self.pso.step(loss_fn)

            # Gradient refinement on top particles
            for particle in self.pso.particles[:3]:  # Top 3
                model = self.pso.model_fn().to(self.device)
                particle._unflatten_params(particle.position, model)

                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

                for _ in range(self.gradient_steps):
                    for x, y in train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        out = model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        break  # One batch per step

                # Update particle position
                particle.position = particle._flatten_params(model)

        return self.pso.get_best_model()
```

### Performance Notes

**PSO vs Gradient Descent**:
- PSO: O(num_particles * forward_pass) per iteration
- No backward pass needed!
- Better for non-differentiable objectives
- Worse for high-dimensional problems (neural nets are high-dim)

**When to use PSO**:
- Hyperparameter optimization
- Architecture search
- Non-differentiable loss functions
- Avoiding local minima

---

## Section 4: Collective Decision-Making Networks

### Voting Networks

Neural networks that explicitly implement voting mechanisms:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingLayer(nn.Module):
    """
    Explicit voting mechanism in neural network.

    Each "voter" (sub-network) casts a vote.
    Final decision = aggregated votes.

    TRAIN STATION:
    Voting = soft attention = MoE gating = democracy
    """

    def __init__(
        self,
        input_dim: int,
        num_voters: int,
        hidden_dim: int,
        output_dim: int,
        voting_type: str = 'soft'  # 'soft', 'hard', 'weighted'
    ):
        super().__init__()
        self.num_voters = num_voters
        self.voting_type = voting_type

        # Each voter is a small network
        self.voters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_voters)
        ])

        # Confidence/weight predictor per voter
        self.confidence = nn.ModuleList([
            nn.Linear(input_dim, 1)
            for _ in range(num_voters)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get votes from all voters
        votes = []
        confidences = []

        for voter, conf in zip(self.voters, self.confidence):
            vote = voter(x)
            confidence = torch.sigmoid(conf(x))
            votes.append(vote)
            confidences.append(confidence)

        # Stack: [num_voters, batch, output_dim]
        votes = torch.stack(votes, dim=0)
        confidences = torch.stack(confidences, dim=0)  # [num_voters, batch, 1]

        if self.voting_type == 'soft':
            # Weighted average by confidence
            weights = F.softmax(confidences, dim=0)
            return (votes * weights).sum(dim=0)

        elif self.voting_type == 'hard':
            # Majority voting
            # Each voter's argmax
            voter_decisions = votes.argmax(dim=-1)  # [num_voters, batch]
            # Mode across voters
            final_decision = torch.mode(voter_decisions, dim=0).values
            # Convert back to one-hot
            return F.one_hot(final_decision, votes.size(-1)).float()

        elif self.voting_type == 'weighted':
            # Confidence-weighted sum (no softmax)
            return (votes * confidences).sum(dim=0) / (confidences.sum(dim=0) + 1e-8)


class ConsensusNetwork(nn.Module):
    """
    Network that reaches consensus through iterative message passing.

    Inspired by:
    - Bioelectric networks in cells
    - Opinion dynamics in social networks
    - Belief propagation

    TRAIN STATION:
    Consensus = equilibrium = attractor = steady state

    Agents exchange messages until they agree.
    This is message passing / predictive coding!
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        message_dim: int,
        num_iterations: int = 5
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_iterations = num_iterations

        # Initial state encoder
        self.encoder = nn.Linear(state_dim, message_dim)

        # Message computation
        self.message_fn = nn.Sequential(
            nn.Linear(message_dim * 2, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )

        # State update
        self.update_fn = nn.GRUCell(message_dim, message_dim)

        # Final decision
        self.decoder = nn.Linear(message_dim, state_dim)

    def forward(self, initial_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_states: [batch, num_agents, state_dim]

        Returns:
            consensus_states: [batch, num_agents, state_dim]
        """
        batch_size = initial_states.size(0)

        # Encode initial states
        # [batch, num_agents, message_dim]
        states = self.encoder(initial_states)

        # Iterative consensus
        for _ in range(self.num_iterations):
            # Compute messages between all pairs
            # Mean-field approximation: aggregate all neighbors

            # Mean state of all other agents
            # [batch, 1, message_dim]
            mean_others = states.mean(dim=1, keepdim=True)
            # Broadcast: [batch, num_agents, message_dim]
            mean_others = mean_others.expand_as(states)

            # Compute messages
            # [batch, num_agents, message_dim * 2]
            message_input = torch.cat([states, mean_others], dim=-1)
            # [batch * num_agents, message_dim * 2]
            message_input_flat = message_input.view(-1, message_input.size(-1))

            messages = self.message_fn(message_input_flat)
            messages = messages.view(batch_size, self.num_agents, -1)

            # Update states with messages (GRU)
            states_flat = states.view(-1, states.size(-1))
            messages_flat = messages.view(-1, messages.size(-1))

            states_flat = self.update_fn(messages_flat, states_flat)
            states = states_flat.view(batch_size, self.num_agents, -1)

        # Decode final consensus
        consensus = self.decoder(states)

        return consensus

    def get_agreement_level(self, states: torch.Tensor) -> torch.Tensor:
        """
        Measure how much agents agree (low variance = high agreement).
        """
        variance = states.var(dim=1).mean(dim=-1)
        agreement = 1 / (1 + variance)
        return agreement
```

---

## Section 5: The TRAIN STATION - Collective = MoE = Ensemble = Swarm = Cells

### The Grand Unification

**All collective intelligence systems share the same structure**:

```
                    COLLECTIVE INTELLIGENCE
                           |
           +---------------+---------------+
           |               |               |
        ENSEMBLE          MoE           SWARM
           |               |               |
     Multiple models  Multiple experts  Multiple agents
           |               |               |
     Aggregate votes  Route + combine   Share info
           |               |               |
           +---------------+---------------+
                           |
                     SAME PATTERN!
                           |
              +------------+------------+
              |            |            |
           NEURONS       CELLS      SOCIAL
           (brain)     (tissue)   (society)
```

### The Isomorphisms

**1. Ensemble = MoE (with uniform gating)**
```python
# Ensemble: all experts, equal weight
ensemble_output = sum(expert(x) for expert in experts) / n_experts

# MoE: learned gating
moe_output = sum(gate[i] * expert[i](x) for i in range(n_experts))

# When gate = uniform: MoE = Ensemble!
```

**2. MoE = Attention (with experts as values)**
```python
# Attention: Q @ K^T -> weights for V
attn_weights = softmax(Q @ K.T)
attn_output = attn_weights @ V

# MoE: router(x) -> weights for experts
expert_weights = softmax(router(x))
moe_output = sum(expert_weights[i] * expert[i](x))

# Router = Q @ K^T (implicit keys in router weights)
# Experts = V (values)
```

**3. Swarm = Distributed Optimization**
```python
# Swarm: particles explore, share best
for particle in swarm:
    particle.move_toward(global_best)

# Gradient descent: all params move together
params = params - lr * gradient

# Swarm = distributed gradient descent with exploration
```

**4. Cells in Tissue = Experts in MoE**
```python
# Cell differentiation: cells become specialized
cell_type = differentiation_signal(position, neighbors)
output = cell_type_function(input)

# MoE: experts are specialized
expert_id = router(input)
output = expert[expert_id](input)

# Cell fate = expert routing!
```

### The Deep Connection to Morphogenesis

**Morphogenesis = collective intelligence of cells**:
- Each cell = an "agent" that processes signals
- Cell-cell communication = message passing
- Pattern formation = emergent collective behavior
- Cell differentiation = expert specialization

**MoE is a simplified morphogenetic system**:
- Experts = cell types
- Router = morphogen gradient
- Sparse activation = tissue-specific expression
- Load balancing = avoiding tumor (all cells same type)

### The Unified Equation

```python
class UnifiedCollective(nn.Module):
    """
    All collective systems in one framework.

    output = sum_i w_i(x) * f_i(x)

    Where:
    - Ensemble: w_i = 1/n (uniform)
    - MoE: w_i = router(x)[i] (learned)
    - Attention: w_i = softmax(q @ k_i) (content-based)
    - Voting: w_i = confidence_i (explicit)

    THE TRAIN STATION:
    All collective intelligence = weighted sum of specialists
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_specialists: int,
        weighting: str = 'learned'  # 'uniform', 'learned', 'confidence'
    ):
        super().__init__()

        # Specialists (experts/voters/models)
        self.specialists = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_specialists)
        ])

        # Weighting mechanism
        self.weighting = weighting
        if weighting == 'learned':
            self.router = nn.Linear(input_dim, num_specialists)
        elif weighting == 'confidence':
            self.confidence = nn.ModuleList([
                nn.Linear(input_dim, 1) for _ in range(num_specialists)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get specialist outputs
        outputs = [s(x) for s in self.specialists]
        outputs = torch.stack(outputs, dim=0)  # [n, batch, output]

        # Get weights
        if self.weighting == 'uniform':
            weights = torch.ones(len(self.specialists), device=x.device)
            weights = weights / len(self.specialists)
        elif self.weighting == 'learned':
            weights = F.softmax(self.router(x), dim=-1)  # [batch, n]
            weights = weights.T.unsqueeze(-1)  # [n, batch, 1]
        elif self.weighting == 'confidence':
            weights = torch.stack([
                torch.sigmoid(c(x)) for c in self.confidence
            ], dim=0)  # [n, batch, 1]
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted combination
        if self.weighting == 'uniform':
            return outputs.mean(dim=0)
        else:
            return (outputs * weights).sum(dim=0)
```

---

## Section 6: ARR-COC-0-1 Connection - Multi-Agent Relevance

### The Core Insight

**Relevance scoring can be collective**:
- Different "experts" for different content types
- Route image regions to appropriate scorers
- Aggregate with learned or uncertainty-weighted combination

### Application to Adaptive Token Allocation

```python
class CollectiveRelevanceScorer(nn.Module):
    """
    Multi-agent relevance scoring for ARR-COC.

    Different experts for:
    - Faces (specialized face relevance)
    - Text (OCR relevance)
    - Objects (object salience)
    - Scene (global context)

    The router decides which expert(s) to consult.
    This matches how the brain processes:
    - FFA for faces
    - Visual word form area for text
    - Etc.
    """

    def __init__(
        self,
        feature_dim: int,
        num_experts: int = 4,
        top_k: int = 2
    ):
        super().__init__()

        # Expert names for interpretability
        self.expert_names = ['face', 'text', 'object', 'scene'][:num_experts]

        # Specialized relevance scorers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_experts)
        ])

        # Router based on content
        self.router = nn.Linear(feature_dim, num_experts)
        self.top_k = top_k

    def forward(
        self,
        features: torch.Tensor,
        return_expert_contributions: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, num_tokens, feature_dim]

        Returns:
            relevance_scores: [batch, num_tokens]
        """
        # Route
        routing_logits = self.router(features)  # [batch, tokens, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Top-K selection for efficiency
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # Renormalize

        # Compute expert scores
        batch, num_tokens, _ = features.shape
        scores = torch.zeros(batch, num_tokens, device=features.device)

        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # [batch, tokens]
            weight = top_k_weights[..., k]  # [batch, tokens]

            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = features[mask]
                    expert_score = self.experts[e](expert_input).squeeze(-1)
                    scores[mask] += weight[mask] * expert_score

        if return_expert_contributions:
            return scores, routing_weights

        return scores

    def get_expert_interpretation(
        self,
        features: torch.Tensor
    ) -> dict:
        """
        Interpret which experts are used for which tokens.
        Useful for understanding what drives relevance.
        """
        routing_logits = self.router(features)
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Dominant expert per token
        dominant = routing_probs.argmax(dim=-1)

        interpretation = {}
        for i, name in enumerate(self.expert_names):
            mask = (dominant == i)
            interpretation[name] = {
                'fraction': mask.float().mean().item(),
                'avg_confidence': routing_probs[..., i].mean().item()
            }

        return interpretation


class SwarmTokenAllocation(nn.Module):
    """
    Allocate tokens using swarm-inspired optimization.

    Idea: Each token "competes" for allocation budget.
    High-relevance tokens attract more budget (like resources in ecology).

    This is more dynamic than fixed top-K:
    - Tokens can "communicate" through competition
    - Budget naturally flows to important regions
    """

    def __init__(
        self,
        num_iterations: int = 5,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.temperature = temperature

    def forward(
        self,
        relevance_scores: torch.Tensor,
        total_budget: int
    ) -> torch.Tensor:
        """
        Args:
            relevance_scores: [batch, num_tokens]
            total_budget: total tokens to allocate

        Returns:
            allocation: [batch, num_tokens] - probability of selection
        """
        batch_size, num_tokens = relevance_scores.shape

        # Initialize allocation as uniform
        allocation = torch.ones_like(relevance_scores) / num_tokens

        # Iterative reallocation (swarm dynamics)
        for _ in range(self.num_iterations):
            # Tokens with high relevance and low current allocation
            # should "attract" more budget
            need = relevance_scores - allocation

            # Softmax to get redistribution
            redistribution = F.softmax(need / self.temperature, dim=-1)

            # Blend with current allocation
            allocation = 0.5 * allocation + 0.5 * redistribution

        # Final normalization
        allocation = allocation / allocation.sum(dim=-1, keepdim=True)

        return allocation

    def sample_tokens(
        self,
        allocation: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Sample tokens according to allocation distribution.
        """
        # Multinomial sampling
        indices = torch.multinomial(allocation, num_samples, replacement=False)
        return indices
```

### Benefits for Relevance-Based Token Allocation

1. **Specialization**: Different experts for different content types
2. **Adaptivity**: Routing changes based on input
3. **Efficiency**: Sparse activation saves compute
4. **Interpretability**: Can see which expert drove the decision
5. **Robustness**: Multiple experts = less sensitivity to individual failures

---

## Performance Summary

### Ensemble Methods

| Aspect | Consideration |
|--------|--------------|
| Memory | O(n * model_size) |
| Compute | O(n * forward) |
| Quality | Generally better than single model |
| Latency | Can parallelize |

### Mixture of Experts

| Aspect | Consideration |
|--------|--------------|
| Memory | O(n_experts * expert_size) |
| Compute | O(top_k * expert_compute) - SPARSE! |
| Quality | Scales better than dense |
| Latency | Communication overhead for distributed |

### Swarm Optimization

| Aspect | Consideration |
|--------|--------------|
| Memory | O(n_particles * model_size) |
| Compute | O(n_particles * forward) - no backward |
| Quality | Good for global search |
| Use case | Non-differentiable objectives |

---

## Sources

**Primary Research**:
- [Collective Intelligence for Deep Learning](https://arxiv.org/abs/2111.14377) - Ha & Tang, 2021 (arXiv:2111.14377, accessed 2025-11-23)
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - Hugging Face Blog, 2023 (accessed 2025-11-23)
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2022

**Additional References**:
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017
- [GShard: Scaling Giant Models](https://arxiv.org/abs/2006.16668) - Lepikhin et al., 2020
- [ST-MoE: Stable and Transferable](https://arxiv.org/abs/2202.08906) - Zoph et al., 2022
- [Deep Reinforcement Learning for Swarm Systems](https://jmlr.org/papers/volume20/18-476/18-476.pdf) - JMLR, 2019

**Web Resources**:
- [DataCamp MoE Guide](https://www.datacamp.com/blog/mixture-of-experts-moe)
- [Machine Learning Mastery - Mixture of Experts](https://www.machinelearningmastery.com/mixture-of-experts/)

---

## Key Takeaways

1. **All collective systems = weighted sum of specialists**
2. **MoE = learned attention over experts**
3. **Ensemble = MoE with uniform gating**
4. **Swarm = distributed exploration with information sharing**
5. **Morphogenesis = collective intelligence of cells**
6. **The TRAIN STATION: Collective = MoE = Ensemble = Swarm = Cells**

The deep insight: **intelligence emerges from simple agents following simple rules**. Whether neurons in a brain, cells in a tissue, particles in a swarm, or experts in an MoE - the pattern is the same. The whole is greater than the sum of its parts because of the interactions, the routing, the communication.

For ARR-COC-0-1: Use MoE-style routing for relevance scoring, with different experts specializing in different content types. The router learns to consult the right experts for each image region.
