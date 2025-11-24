# Mode Connectivity: The Hidden Pathways in Loss Landscapes

## Overview

Mode connectivity is a profound discovery about the geometry of neural network loss landscapes: different trained solutions (local minima) are not isolated but connected by simple paths along which loss remains nearly constant. This topological property challenges our fundamental understanding of neural network optimization and has deep implications for generalization, ensembling, and model merging.

**Key Insight**: The loss landscape is not a collection of isolated valleys - it's a connected manifold where all good solutions can reach each other through tunnels of low loss.

---

## Section 1: The Discovery - Breaking the Isolation Myth

### Traditional View vs Reality

**Traditional assumption**: Local optima in deep learning are isolated basins separated by high-loss barriers. Different random seeds yield fundamentally different, disconnected solutions.

**The discovery** (Garipov et al. 2018, Draxler et al. 2018):
- Any two independently trained networks can be connected by a simple curve
- Loss along this curve stays nearly constant (no significant barriers!)
- The curve can be as simple as a quadratic Bezier or one-bend polygonal chain

From [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026) (Garipov et al., NeurIPS 2018, 987 citations):

```
Key findings:
- Linear path: Loss EXPLODES between two optima (reaching random init levels)
- Curved path: Loss stays nearly constant (< 5% variation)
- Works for VGG, ResNet, WideResNet on CIFAR-10/100
```

### Why Linear Interpolation Fails

```python
import torch
import torch.nn as nn

def linear_interpolate(model1_params, model2_params, alpha):
    """Linear interpolation between two sets of parameters."""
    return {
        name: (1 - alpha) * model1_params[name] + alpha * model2_params[name]
        for name in model1_params.keys()
    }

# The problem: Loss along linear path
# alpha:    0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0
# Loss:     0.5   1.2   3.4   8.9  12.1  15.3  11.8   7.2   2.8   1.0   0.4

# Loss explodes at the midpoint! The linear path crosses high-loss barriers.
```

### The Curved Path Solution

```python
class BezierCurve:
    """Quadratic Bezier curve for mode connectivity."""

    def __init__(self, model1_params, model2_params, bend_params):
        """
        Args:
            model1_params: Endpoint 1 (fixed)
            model2_params: Endpoint 2 (fixed)
            bend_params: Learnable bend point (optimized to minimize path loss)
        """
        self.w1 = model1_params
        self.w2 = model2_params
        self.theta = bend_params  # The "bend" in the curve

    def __call__(self, t):
        """Evaluate curve at parameter t in [0, 1]."""
        # Quadratic Bezier: (1-t)^2 * w1 + 2t(1-t) * theta + t^2 * w2
        return {
            name: (1 - t)**2 * self.w1[name] +
                  2 * t * (1 - t) * self.theta[name] +
                  t**2 * self.w2[name]
            for name in self.w1.keys()
        }

# Now:
# t:        0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0
# Loss:     0.5   0.6   0.7   0.8   0.9   0.8   0.7   0.6   0.5   0.5   0.4

# Loss stays nearly constant along the curved path!
```

---

## Section 2: Mathematical Formulation of Path Finding

### The Optimization Problem

Given two trained networks with parameters w_1 and w_2, we want to find a path phi(t) where:
- phi(0) = w_1, phi(1) = w_2 (endpoints fixed)
- Loss(phi(t)) is approximately constant for all t in [0, 1]

**Objective**: Minimize average loss along the path

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeConnectivityTrainer:
    """Train curves connecting neural network optima."""

    def __init__(self, model_class, model1_ckpt, model2_ckpt, curve_type='bezier'):
        # Load endpoint models
        self.model1 = model_class()
        self.model1.load_state_dict(torch.load(model1_ckpt))

        self.model2 = model_class()
        self.model2.load_state_dict(torch.load(model2_ckpt))

        # Initialize curve parameter (bend point)
        # Start at midpoint of linear interpolation
        self.theta = {}
        for name, param1 in self.model1.named_parameters():
            param2 = dict(self.model2.named_parameters())[name]
            self.theta[name] = nn.Parameter(
                0.5 * (param1.data + param2.data)
            )

        self.curve_type = curve_type

    def get_point_on_curve(self, t):
        """Get model parameters at point t on the curve."""
        params = {}

        if self.curve_type == 'bezier':
            # Quadratic Bezier curve
            for name, param1 in self.model1.named_parameters():
                param2 = dict(self.model2.named_parameters())[name]
                params[name] = (
                    (1 - t)**2 * param1.data +
                    2 * t * (1 - t) * self.theta[name] +
                    t**2 * param2.data
                )
        elif self.curve_type == 'polychain':
            # Polygonal chain with one bend
            if t <= 0.5:
                alpha = 2 * t
                for name, param1 in self.model1.named_parameters():
                    params[name] = (1 - alpha) * param1.data + alpha * self.theta[name]
            else:
                alpha = 2 * (t - 0.5)
                for name, param2 in self.model2.named_parameters():
                    params[name] = (1 - alpha) * self.theta[name] + alpha * param2.data

        return params

    def train_curve(self, train_loader, epochs=200, lr=0.03):
        """Train the curve to minimize average loss along the path."""
        optimizer = torch.optim.SGD(self.theta.values(), lr=lr, momentum=0.9)

        eval_model = type(self.model1)()  # Create model for evaluation

        for epoch in range(epochs):
            total_loss = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                # Sample random point on curve
                t = torch.rand(1).item()

                # Get parameters at this point
                curve_params = self.get_point_on_curve(t)

                # Load into evaluation model
                state_dict = {name: param for name, param in curve_params.items()}
                eval_model.load_state_dict(state_dict)

                # Forward pass
                outputs = eval_model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)

                # Backward pass through the curve parameterization
                # Chain rule: dL/dtheta = dL/dphi * dphi/dtheta
                loss.backward()

                # Compute gradients with respect to theta
                with torch.no_grad():
                    for name in self.theta:
                        # dphi/dtheta for quadratic Bezier
                        dphi_dtheta = 2 * t * (1 - t)
                        # Update theta gradient
                        if self.theta[name].grad is None:
                            self.theta[name].grad = dphi_dtheta * curve_params[name].grad
                        else:
                            self.theta[name].grad += dphi_dtheta * curve_params[name].grad

                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch}: Avg Loss = {total_loss / len(train_loader):.4f}")
```

### Complete PyTorch Implementation

From [GitHub: timgaripov/dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity):

```python
import torch
import torch.nn as nn
import numpy as np

class CurveModule(nn.Module):
    """Base class for curve-based neural network modules."""

    def __init__(self, fix_endpoints=True, num_bends=1):
        super().__init__()
        self.fix_endpoints = fix_endpoints
        self.num_bends = num_bends
        self.num_points = num_bends + 2  # endpoints + bends

    def forward(self, x, coeffs_t):
        """Forward pass using curve coefficients for weight interpolation."""
        raise NotImplementedError

class BezierLinear(CurveModule):
    """Linear layer parameterized by Bezier curve."""

    def __init__(self, in_features, out_features, fix_endpoints=True, num_bends=1):
        super().__init__(fix_endpoints, num_bends)

        self.in_features = in_features
        self.out_features = out_features

        # Parameters for each control point
        self.weight = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(self.num_points)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features))
            for _ in range(self.num_points)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_points):
            nn.init.kaiming_uniform_(self.weight[i], a=np.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x, coeffs_t):
        """
        Args:
            x: Input tensor
            coeffs_t: Bezier coefficients for parameter t
                      Shape: (num_points,) computed from t value
        """
        # Interpolate weights and biases using Bezier coefficients
        weight = sum(coeff * w for coeff, w in zip(coeffs_t, self.weight))
        bias = sum(coeff * b for coeff, b in zip(coeffs_t, self.bias))

        return F.linear(x, weight, bias)

def bezier_coefficients(t, num_points):
    """Compute Bezier basis functions for parameter t."""
    n = num_points - 1
    coeffs = []
    for i in range(num_points):
        # Binomial coefficient
        binom = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
        # Bernstein polynomial
        coeff = binom * (t ** i) * ((1 - t) ** (n - i))
        coeffs.append(coeff)
    return coeffs

class CurveNet(nn.Module):
    """Neural network with all layers parameterized by curves."""

    def __init__(self, base_model_class, num_bends=1):
        super().__init__()
        self.num_bends = num_bends
        self.num_points = num_bends + 2

        # Build curve-parameterized architecture
        # (Architecture matches base_model_class but with BezierLinear, etc.)
        self.layers = self._build_curve_layers(base_model_class)

    def forward(self, x, t):
        """Forward pass at point t on the curve."""
        coeffs = bezier_coefficients(t, self.num_points)

        for layer in self.layers:
            if isinstance(layer, CurveModule):
                x = layer(x, coeffs)
            else:
                x = layer(x)

        return x

    def import_endpoints(self, model1, model2):
        """Load trained models as curve endpoints."""
        # Copy parameters from model1 to first control point
        # Copy parameters from model2 to last control point
        # Initialize intermediate points by interpolation
        pass  # Implementation details in full repo
```

---

## Section 3: Linear Mode Connectivity (LMC)

### A Stronger Condition

**Mode Connectivity**: Curved paths with constant loss exist between minima

**Linear Mode Connectivity (LMC)**: Even the straight line between minima has constant loss!

From [Linear Mode Connectivity and the Lottery Ticket Hypothesis](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf) (Frankle et al., ICML 2020, 736 citations):

```python
def check_linear_mode_connectivity(model1, model2, test_loader, num_points=11):
    """Check if two models are linearly mode connected."""

    losses = []
    accuracies = []

    for alpha in np.linspace(0, 1, num_points):
        # Linear interpolation
        interpolated_params = {}
        for name in model1.state_dict():
            interpolated_params[name] = (
                (1 - alpha) * model1.state_dict()[name] +
                alpha * model2.state_dict()[name]
            )

        # Evaluate
        eval_model = type(model1)()
        eval_model.load_state_dict(interpolated_params)

        loss, acc = evaluate(eval_model, test_loader)
        losses.append(loss)
        accuracies.append(acc)

    # Check for barrier
    max_loss = max(losses)
    endpoint_loss = max(losses[0], losses[-1])
    barrier = max_loss - endpoint_loss

    return {
        'losses': losses,
        'accuracies': accuracies,
        'barrier': barrier,
        'is_lmc': barrier < 0.1 * endpoint_loss  # Less than 10% barrier
    }
```

### When Does LMC Hold?

**LMC typically holds when**:
1. **Same training run**: Checkpoints from same trajectory are LMC
2. **Late in training**: Networks from different seeds can become LMC late in training
3. **After permutation alignment**: Networks can be aligned to achieve LMC (Git Re-Basin)

**LMC typically fails when**:
1. **Different random seeds from scratch**: Early training diverges too much
2. **Different hyperparameters**: Learning rate, batch size differences break LMC
3. **Different architectures**: Obviously, different structures can't interpolate

```python
class LinearModeConnectivityExperiment:
    """Experiments on linear mode connectivity conditions."""

    def test_same_trajectory_lmc(self, model, train_loader, checkpoints=[50, 100, 150, 200]):
        """Test LMC between checkpoints from same training run."""
        results = {}

        # Train and save checkpoints
        ckpts = {}
        for epoch in range(max(checkpoints) + 1):
            train_one_epoch(model, train_loader)
            if epoch in checkpoints:
                ckpts[epoch] = copy.deepcopy(model.state_dict())

        # Test LMC between consecutive checkpoints
        for i in range(len(checkpoints) - 1):
            e1, e2 = checkpoints[i], checkpoints[i + 1]

            model1 = type(model)()
            model1.load_state_dict(ckpts[e1])

            model2 = type(model)()
            model2.load_state_dict(ckpts[e2])

            results[f'{e1}-{e2}'] = check_linear_mode_connectivity(model1, model2)

        return results
        # Typically: ALL consecutive checkpoints are LMC with barrier < 1%

    def test_different_seeds_lmc(self, model_class, train_loader, seeds=[0, 1, 2]):
        """Test LMC between networks trained with different seeds."""
        models = {}

        for seed in seeds:
            torch.manual_seed(seed)
            model = model_class()
            train_full(model, train_loader)
            models[seed] = model

        # Test all pairs
        results = {}
        for s1, s2 in itertools.combinations(seeds, 2):
            results[f'{s1}-{s2}'] = check_linear_mode_connectivity(
                models[s1], models[s2]
            )

        return results
        # Typically: Barrier is LARGE (5-50% of endpoint loss)
        # Need CURVED paths for mode connectivity here
```

### Lottery Ticket Connection

From [Linear Mode Connectivity and the Lottery Ticket Hypothesis](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf):

**Key finding**: Lottery ticket subnetworks from the same initialization are linearly mode connected

```python
def lottery_ticket_lmc_experiment():
    """
    Lottery tickets found from same init are LMC.
    Lottery tickets found from different inits are NOT LMC.
    """

    # Train full network
    init_state = model.state_dict().copy()
    train_full(model)

    # Find lottery ticket (iterative magnitude pruning)
    mask = find_lottery_ticket(model, sparsity=0.9)

    # Retrain from same init with mask - multiple times
    ticket1 = retrain_with_mask(init_state, mask, seed=0)
    ticket2 = retrain_with_mask(init_state, mask, seed=1)

    # These ARE linearly mode connected!
    lmc_result = check_linear_mode_connectivity(ticket1, ticket2)
    assert lmc_result['barrier'] < 0.01  # Very small barrier

    # Now try different init
    torch.manual_seed(999)
    different_init = model_class()
    different_init_state = different_init.state_dict().copy()

    # Lottery tickets from different inits are NOT LMC
    ticket3 = retrain_with_mask(different_init_state, mask, seed=0)

    not_lmc = check_linear_mode_connectivity(ticket1, ticket3)
    assert not_lmc['barrier'] > 0.1  # Large barrier!
```

---

## Section 4: Implications for Generalization

### Flatness and Generalization

Mode connectivity reveals fundamental properties about the loss landscape that relate to generalization:

```python
class GeneralizationAnalysis:
    """Analyze how mode connectivity relates to generalization."""

    def analyze_path_diversity(self, curve, test_loader, num_points=51):
        """
        Key insight: Points along the low-loss path make DIFFERENT predictions.
        This diversity enables ensembling benefits.
        """
        all_predictions = []

        for t in np.linspace(0, 1, num_points):
            params = curve.get_point_on_curve(t)
            model = self.load_params(params)

            predictions = []
            for x, _ in test_loader:
                pred = model(x).argmax(dim=1)
                predictions.append(pred)

            all_predictions.append(torch.cat(predictions))

        # Measure prediction diversity
        disagreement_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                disagreement = (all_predictions[i] != all_predictions[j]).float().mean()
                disagreement_matrix[i, j] = disagreement.item()

        return {
            'predictions': all_predictions,
            'disagreement_matrix': disagreement_matrix,
            'mean_disagreement': disagreement_matrix.mean(),
            'endpoint_disagreement': disagreement_matrix[0, -1]
        }

    def ensemble_along_curve(self, curve, test_loader, num_models=5):
        """
        Ensemble predictions from models sampled along the curve.
        Benefits from diversity while staying in low-loss region.
        """
        # Sample models uniformly along curve
        t_values = np.linspace(0, 1, num_models)

        all_logits = []
        for t in t_values:
            params = curve.get_point_on_curve(t)
            model = self.load_params(params)

            logits = []
            for x, _ in test_loader:
                logits.append(model(x))
            all_logits.append(torch.cat(logits))

        # Average predictions (soft voting)
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        ensemble_preds = ensemble_logits.argmax(dim=1)

        # Compare to individual models
        individual_accs = []
        for logits in all_logits:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            individual_accs.append(acc.item())

        ensemble_acc = (ensemble_preds == labels).float().mean().item()

        return {
            'individual_accs': individual_accs,
            'ensemble_acc': ensemble_acc,
            'improvement': ensemble_acc - np.mean(individual_accs)
        }
```

### Fast Geometric Ensembling (FGE)

From [GitHub: timgaripov/dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity):

**Idea**: Traverse the mode-connected region with cyclic learning rate and collect models for ensembling

```python
class FastGeometricEnsembling:
    """
    FGE: Collect diverse models by traversing mode-connected regions.

    Key insight: Cyclic LR makes the optimizer traverse along
    the low-loss manifold, collecting diverse but high-quality models.
    """

    def __init__(self, model, lr_min=0.001, lr_max=0.05, cycle_length=4):
        self.model = model
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_length = cycle_length

        self.collected_models = []

    def get_lr(self, epoch):
        """Cyclic learning rate schedule."""
        cycle_position = epoch % self.cycle_length
        # Linear schedule within cycle
        t = cycle_position / self.cycle_length

        # Go from max to min within each cycle
        return self.lr_max - (self.lr_max - self.lr_min) * t

    def train_and_collect(self, train_loader, epochs=40):
        """Train with cyclic LR and collect models at cycle ends."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr_max,
            momentum=0.9,
            weight_decay=5e-4
        )

        for epoch in range(epochs):
            # Update learning rate
            lr = self.get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Train one epoch
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()

            # Collect model at end of each cycle (when LR is at minimum)
            if (epoch + 1) % self.cycle_length == 0:
                self.collected_models.append(
                    copy.deepcopy(self.model.state_dict())
                )
                print(f"Collected model {len(self.collected_models)} at epoch {epoch + 1}")

        return self.collected_models

    def ensemble_predict(self, x):
        """Make predictions by averaging collected models."""
        all_logits = []

        for state_dict in self.collected_models:
            self.model.load_state_dict(state_dict)
            with torch.no_grad():
                logits = self.model(x)
            all_logits.append(logits)

        # Average logits
        return torch.stack(all_logits).mean(dim=0)
```

**FGE Results** (from paper):

| Model | Independent Ensemble | FGE | Budget |
|-------|---------------------|-----|--------|
| VGG16 | 74.8% | 76.1% | 2x |
| PreResNet164 | 80.5% | 81.3% | 2x |
| WideResNet28x10 | 82.4% | 82.9% | 2x |

FGE achieves BETTER ensembling with SAME compute budget!

---

## Section 5: Mode Connectivity Beyond Classification

### Reinforcement Learning

From research on RL policy optimization:

```python
class RLModeConnectivity:
    """Mode connectivity in reinforcement learning policy space."""

    def find_policy_path(self, policy1, policy2, env, episodes_per_eval=100):
        """Find low-return-loss path between two trained policies."""

        # Initialize curve
        curve = BezierCurve(
            policy1.state_dict(),
            policy2.state_dict(),
            self.initialize_bend(policy1, policy2)
        )

        optimizer = torch.optim.Adam([curve.theta], lr=1e-3)

        for iteration in range(1000):
            # Sample point on curve
            t = np.random.random()
            params = curve(t)

            # Evaluate policy at this point
            eval_policy = type(policy1)()
            eval_policy.load_state_dict(params)

            # Compute expected return (reward)
            returns = []
            for _ in range(episodes_per_eval):
                total_return = run_episode(eval_policy, env)
                returns.append(total_return)

            # We want to MAXIMIZE return, so MINIMIZE negative return
            loss = -torch.tensor(np.mean(returns))

            # Update curve
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return curve

    def evaluate_policy_path(self, curve, env, num_points=21):
        """Evaluate return along the policy path."""
        results = []

        for t in np.linspace(0, 1, num_points):
            params = curve(t)
            policy = self.load_policy(params)

            returns = [run_episode(policy, env) for _ in range(100)]
            results.append({
                't': t,
                'mean_return': np.mean(returns),
                'std_return': np.std(returns)
            })

        return results
```

### Language Models and Fine-tuning

Mode connectivity has implications for model merging and continual learning:

```python
class LanguageModelModeConnectivity:
    """Mode connectivity for fine-tuned language models."""

    def analyze_finetuned_models(self, base_model, task1_model, task2_model):
        """
        Question: Are models fine-tuned on different tasks mode connected?

        Findings:
        - Models fine-tuned on SIMILAR tasks: Often mode connected
        - Models fine-tuned on DIFFERENT tasks: Less connected
        - Models fine-tuned from SAME base: More connected
        """

        # Check connectivity between fine-tuned models
        results = {}

        # Task 1 to Task 2
        results['task1_to_task2'] = self.find_path(task1_model, task2_model)

        # Base to Task 1
        results['base_to_task1'] = self.find_path(base_model, task1_model)

        # Base to Task 2
        results['base_to_task2'] = self.find_path(base_model, task2_model)

        return results

    def model_averaging_via_connectivity(self, models):
        """
        If models are mode connected, averaging them should work well.
        This is the theory behind model merging techniques!
        """
        # Simple average (works if models are linearly mode connected)
        avg_params = {}
        for name in models[0].state_dict():
            params = [m.state_dict()[name] for m in models]
            avg_params[name] = torch.stack(params).mean(dim=0)

        merged_model = type(models[0])()
        merged_model.load_state_dict(avg_params)

        return merged_model
```

---

## Section 6: Advanced Topics

### Loss Surface Simplexes

From [Loss Surface Simplexes for Mode Connecting Volumes](http://proceedings.mlr.press/v139/benton21a/benton21a.pdf) (Benton et al., ICML 2021):

**Extension**: Not just paths between 2 points, but entire VOLUMES connecting multiple solutions!

```python
class LossSurfaceSimplex:
    """
    Simplexes of low loss connecting multiple solutions.

    A 2-simplex (triangle) connects 3 solutions.
    A 3-simplex (tetrahedron) connects 4 solutions.
    """

    def __init__(self, vertices):
        """
        Args:
            vertices: List of model state dicts (the simplex vertices)
        """
        self.vertices = vertices
        self.n_vertices = len(vertices)

    def sample_point(self):
        """Sample random point in the simplex using Dirichlet distribution."""
        # Uniform distribution over simplex
        weights = np.random.dirichlet(np.ones(self.n_vertices))

        # Interpolate all vertices
        params = {}
        for name in self.vertices[0]:
            params[name] = sum(
                w * v[name] for w, v in zip(weights, self.vertices)
            )

        return params, weights

    def evaluate_simplex_volume(self, test_loader, num_samples=1000):
        """Evaluate loss throughout the simplex volume."""
        losses = []

        for _ in range(num_samples):
            params, weights = self.sample_point()
            model = self.load_params(params)

            loss = evaluate_loss(model, test_loader)
            losses.append({
                'weights': weights,
                'loss': loss
            })

        return {
            'mean_loss': np.mean([l['loss'] for l in losses]),
            'max_loss': np.max([l['loss'] for l in losses]),
            'std_loss': np.std([l['loss'] for l in losses]),
            'samples': losses
        }

def train_simplex(model_class, train_loader, n_vertices=3, epochs=600):
    """Train a low-loss simplex connecting multiple solutions."""

    # First train endpoints independently
    vertices = []
    for i in range(n_vertices):
        torch.manual_seed(i)
        model = model_class()
        train_full(model, train_loader)
        vertices.append(model.state_dict())

    # Now optimize internal points to minimize loss throughout simplex
    # (Similar to curve optimization but over higher-dimensional space)
    simplex = LossSurfaceSimplex(vertices)

    # Add learnable internal points
    # ... optimization procedure ...

    return simplex
```

### Git Re-Basin: Permutation Alignment for LMC

From [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836):

**Key insight**: Neural networks have permutation symmetry - reordering neurons doesn't change function. Aligning these permutations can achieve LMC between independently trained networks!

```python
class GitReBasin:
    """
    Permutation alignment for linear mode connectivity.

    The idea: Neural networks are equivalent under neuron permutation.
    If we find the right permutation, even independently trained
    networks become linearly mode connected!
    """

    def find_permutation(self, model1, model2, method='activation_matching'):
        """Find permutation that aligns model2 to model1."""

        permutations = {}

        if method == 'activation_matching':
            # Match neurons by activation patterns on data
            for layer_name, layer1 in model1.named_modules():
                if isinstance(layer1, nn.Linear):
                    layer2 = dict(model2.named_modules())[layer_name]

                    # Get activations on reference data
                    acts1 = self.get_activations(model1, layer_name)
                    acts2 = self.get_activations(model2, layer_name)

                    # Solve linear assignment problem
                    # (Hungarian algorithm to match neurons)
                    cost_matrix = self.compute_cost_matrix(acts1, acts2)
                    perm = linear_sum_assignment(cost_matrix)[1]

                    permutations[layer_name] = perm

        elif method == 'weight_matching':
            # Match neurons by weight similarity
            # ... similar approach using weights instead of activations
            pass

        return permutations

    def apply_permutation(self, model, permutations):
        """Apply permutation to model weights."""
        new_state_dict = {}

        for name, param in model.named_parameters():
            layer_name = name.rsplit('.', 1)[0]
            param_type = name.rsplit('.', 1)[1]

            if layer_name in permutations:
                perm = permutations[layer_name]

                if param_type == 'weight':
                    # Permute output dimension
                    new_state_dict[name] = param[perm]
                elif param_type == 'bias':
                    new_state_dict[name] = param[perm]

                # Also need to permute input dimension of next layer
                # ... additional logic ...
            else:
                new_state_dict[name] = param

        aligned_model = type(model)()
        aligned_model.load_state_dict(new_state_dict)

        return aligned_model

    def merge_models(self, model1, model2):
        """Merge two models using permutation alignment."""
        # Find alignment
        perm = self.find_permutation(model1, model2)

        # Apply permutation to model2
        aligned_model2 = self.apply_permutation(model2, perm)

        # Now simple averaging works!
        merged = {}
        for name in model1.state_dict():
            merged[name] = 0.5 * (
                model1.state_dict()[name] +
                aligned_model2.state_dict()[name]
            )

        merged_model = type(model1)()
        merged_model.load_state_dict(merged)

        return merged_model
```

---

## Section 7: TRAIN STATION - Connectivity = Topology = Homeomorphism

### The Grand Unification

**Mode connectivity reveals the TOPOLOGY of the loss landscape!**

```
TRAIN STATION: Where All These Topics Meet

TOPOLOGY                          LOSS LANDSCAPE
---------                         --------------
Path-connected space       =      Mode-connected minima
Homotopy equivalence       =      Continuous deformation of solutions
Contractible space         =      All minima equivalent up to paths
Homeomorphism             =      Equivalent representations (permutations)
Manifold structure         =      Low-loss submanifold

PHYSICS                           NEURAL NETWORKS
-------                           ---------------
Energy landscape           =      Loss surface
Metastable states          =      Local minima
Transition paths           =      Mode connecting curves
Free energy barriers       =      Loss barriers
Phase transitions          =      Sharp changes in generalization

INFORMATION GEOMETRY              OPTIMIZATION
--------------------              ------------
Geodesics                  =      Optimal paths between solutions
Curvature                  =      Sharpness of minima
Riemannian distance        =      Functional distance between models
Fisher information         =      Local geometry at optima
```

### Topology Made Concrete

```python
class TopologicalAnalysis:
    """Analyze loss landscape using topological concepts."""

    def check_path_connectivity(self, solutions, test_loader):
        """
        Topological question: Is the set of good solutions path-connected?

        A space is path-connected if any two points can be connected
        by a continuous path lying entirely within the space.

        For loss landscapes: Can we connect any two minima by a
        path that stays in the low-loss region?
        """
        n = len(solutions)
        connectivity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Find path
                curve = self.find_connecting_curve(solutions[i], solutions[j])

                # Check if path stays in low-loss region
                max_loss_on_path = self.evaluate_max_loss(curve, test_loader)
                endpoint_loss = max(
                    evaluate_loss(solutions[i], test_loader),
                    evaluate_loss(solutions[j], test_loader)
                )

                # Connected if loss barrier is small
                barrier_ratio = max_loss_on_path / endpoint_loss
                connectivity_matrix[i, j] = barrier_ratio
                connectivity_matrix[j, i] = barrier_ratio

        # Check if sublevel set is path-connected
        threshold = 1.1  # Allow 10% barrier
        is_connected = (connectivity_matrix < threshold).all()

        return {
            'connectivity_matrix': connectivity_matrix,
            'is_path_connected': is_connected,
            'max_barrier': connectivity_matrix.max()
        }

    def analyze_homotopy(self, solution1, solution2, intermediate_solution):
        """
        Homotopy analysis: Are different paths between solutions equivalent?

        If two paths are homotopic, we can continuously deform one into
        the other while keeping endpoints fixed and staying in low-loss region.
        """
        # Find two different paths
        path1 = self.find_path_via(solution1, solution2,
                                   avoid_point=intermediate_solution)
        path2 = self.find_path_via(solution1, solution2,
                                   through_point=intermediate_solution)

        # Check if we can deform path1 into path2
        # by checking intermediate paths
        num_intermediate = 10
        for alpha in np.linspace(0, 1, num_intermediate):
            # Interpolated path
            intermediate_path = self.interpolate_paths(path1, path2, alpha)

            # Check loss along intermediate path
            max_loss = self.evaluate_max_loss(intermediate_path)

            if max_loss > threshold:
                return {'homotopic': False, 'obstruction_at': alpha}

        return {'homotopic': True}

    def compute_betti_numbers(self, solutions, loss_threshold):
        """
        Compute Betti numbers of the sublevel set.

        b_0 = number of connected components
        b_1 = number of 1-dimensional holes (loops)
        b_2 = number of 2-dimensional voids

        For mode connectivity: We hope b_0 = 1 (fully connected)
        """
        # Build simplicial complex from solutions
        # ... topological data analysis methods ...

        # Use persistent homology to compute Betti numbers
        # at different loss thresholds

        pass  # Requires TDA libraries like giotto-tda or ripser
```

### The Profound Implication

**If the loss landscape is topologically simple (path-connected with few holes), then:**

1. **Optimization is easier**: Any descent method can find good solutions
2. **Generalization is related to geometry**: Flat, connected regions generalize better
3. **Model merging works**: We can average solutions and stay in good region
4. **Understanding is possible**: We can characterize the space of good solutions

```python
def philosophical_summary():
    """
    Mode Connectivity: The Topological View

    OLD VIEW:
    - Loss landscape is rugged with many isolated minima
    - Different runs find fundamentally different solutions
    - Local minima are traps that hurt optimization
    - Averaging models doesn't make sense

    NEW VIEW (Mode Connectivity):
    - Loss landscape has simple topology (path-connected)
    - Different solutions are equivalent up to continuous deformation
    - The "many minima" are actually one connected basin
    - Model averaging exploits this connectivity

    IMPLICATIONS:
    - Ensembling: Sample diverse points along connected manifold
    - Transfer learning: Fine-tuning stays in connected region
    - Pruning: Lottery tickets preserve connectivity
    - Model merging: Git Re-Basin exploits permutation symmetry
    - Continual learning: Stay in mode-connected region

    THE COFFEE CUP = DONUT EQUIVALENCE:

    Just as a coffee cup is topologically equivalent to a donut
    (both have one hole), different neural network solutions are
    "topologically equivalent" - they're connected by continuous
    paths of equally good solutions!
    """
    pass
```

---

## Section 8: ARR-COC-0-1 Connection - Multiple Relevance Solutions

### Relevance Functions Have Mode Connectivity Too

In ARR-COC, the relevance function determines which tokens receive compute. This is itself a learned function with its own loss landscape - and mode connectivity applies!

```python
class RelevanceModeConnectivity:
    """
    Multiple ways to compute relevance can be mode connected.

    Different relevance solutions that achieve similar performance
    might be connected by low-loss paths - enabling relevance
    ensemble and relevance model merging.
    """

    def __init__(self, arr_coc_model):
        self.model = arr_coc_model

    def train_multiple_relevance_functions(self, train_data, num_solutions=3):
        """
        Train multiple relevance predictors with different seeds.
        Check if they're mode connected.
        """
        relevance_solutions = []

        for seed in range(num_solutions):
            torch.manual_seed(seed)

            # Train relevance predictor
            relevance_predictor = self.model.relevance_net
            relevance_predictor.reset_parameters()

            self.train_relevance(relevance_predictor, train_data)

            relevance_solutions.append(
                copy.deepcopy(relevance_predictor.state_dict())
            )

        return relevance_solutions

    def find_relevance_path(self, rel_params1, rel_params2):
        """Find mode-connecting path between two relevance functions."""

        # Use same curve-finding approach
        curve = BezierCurve(rel_params1, rel_params2,
                           self.initialize_bend(rel_params1, rel_params2))

        # Optimize curve to minimize relevance prediction error
        optimizer = torch.optim.Adam([curve.theta], lr=1e-3)

        for iteration in range(500):
            t = np.random.random()
            params = curve(t)

            # Load relevance predictor
            self.model.relevance_net.load_state_dict(params)

            # Evaluate relevance prediction loss
            loss = self.evaluate_relevance_loss()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return curve

    def ensemble_relevance_predictions(self, curves, x):
        """
        Ensemble relevance predictions from multiple solutions.

        Benefits:
        - More robust relevance estimates
        - Better uncertainty quantification
        - Smoother token allocation
        """
        all_relevance = []

        # Sample from each curve
        for curve in curves:
            t = np.random.random()
            params = curve(t)

            self.model.relevance_net.load_state_dict(params)
            relevance = self.model.relevance_net(x)

            all_relevance.append(relevance)

        # Average relevance scores
        ensemble_relevance = torch.stack(all_relevance).mean(dim=0)

        return ensemble_relevance

    def relevance_model_merging(self, task1_relevance, task2_relevance):
        """
        Merge relevance functions from different tasks/domains.

        If mode connected, simple averaging creates a multi-task
        relevance predictor without any additional training!
        """
        # Check connectivity first
        connectivity = self.check_relevance_connectivity(
            task1_relevance, task2_relevance
        )

        if connectivity['is_connected']:
            # Simple averaging works!
            merged = {}
            for name in task1_relevance:
                merged[name] = 0.5 * (task1_relevance[name] + task2_relevance[name])

            return merged
        else:
            # Need to find curved path or use Git Re-Basin
            return self.find_relevance_path(task1_relevance, task2_relevance)

    def uncertainty_aware_relevance(self, x):
        """
        Use mode connectivity for uncertainty in relevance.

        Sample multiple points along low-loss manifold and
        measure variance in relevance predictions.
        """
        # Sample points along curve between two solutions
        t_values = np.random.random(10)

        relevance_samples = []
        for t in t_values:
            params = self.relevance_curve(t)
            self.model.relevance_net.load_state_dict(params)

            relevance = self.model.relevance_net(x)
            relevance_samples.append(relevance)

        # Compute mean and variance
        relevance_mean = torch.stack(relevance_samples).mean(dim=0)
        relevance_var = torch.stack(relevance_samples).var(dim=0)

        return {
            'relevance': relevance_mean,
            'uncertainty': relevance_var,
            # High uncertainty = different solutions disagree =
            # might need more conservative token allocation
        }
```

### Practical Applications in ARR-COC

```python
class ARRCOCModeConnectivityApplications:
    """Practical applications of mode connectivity in ARR-COC."""

    def fast_relevance_ensembling(self):
        """
        FGE-style ensembling for relevance prediction.

        - Train relevance predictor with cyclic LR
        - Collect checkpoints along trajectory
        - Average predictions for better relevance estimates
        """
        fge = FastGeometricEnsembling(
            self.relevance_net,
            lr_min=1e-4,
            lr_max=1e-2,
            cycle_length=2
        )

        collected = fge.train_and_collect(relevance_data, epochs=20)

        # Now can ensemble relevance predictions
        return collected

    def transfer_relevance_across_tasks(self, source_relevance, target_task):
        """
        Use mode connectivity for relevance transfer learning.

        If source and target relevance functions are mode connected,
        we can interpolate between them for few-shot adaptation.
        """
        # Find path from source to target
        curve = self.find_relevance_path(source_relevance, target_relevance)

        # For few-shot: Don't go all the way to target
        # Stay in the interpolated region
        interpolated_params = curve(0.3)  # 30% toward target

        return interpolated_params

    def robust_token_allocation(self, x):
        """
        Use mode connectivity for robust token allocation.

        Different relevance solutions might allocate tokens differently.
        Ensembling provides more stable allocation decisions.
        """
        # Get relevance from multiple solutions
        relevance_scores = self.ensemble_relevance_predictions(x)

        # Also get uncertainty
        uncertainty = self.relevance_uncertainty(x)

        # Tokens with high uncertainty get more conservative allocation
        # (don't skip them even if mean relevance is low)
        adjusted_relevance = relevance_scores + uncertainty

        # Allocate tokens based on adjusted relevance
        token_allocation = self.allocate_tokens(adjusted_relevance)

        return token_allocation
```

---

## Section 9: Performance Considerations

### Computational Cost

```python
class ModeConnectivityPerformance:
    """Performance considerations for mode connectivity methods."""

    def curve_training_cost(self):
        """
        Cost of training a connecting curve.

        - Same as training one network (one forward/backward per sample)
        - Typically 200-600 epochs
        - GPU memory: Same as base model (no additional models in memory)
        - Time: ~2-3 hours for ResNet on CIFAR
        """
        pass

    def curve_evaluation_cost(self):
        """
        Cost of evaluating along a curve.

        - Need to evaluate at many points (typically 51-101)
        - Each point = one forward pass through dataset
        - Total: 50-100x single model evaluation
        - Can parallelize across points
        """
        pass

    def visualization_cost(self):
        """
        Cost of loss landscape visualization.

        - Need grid of points (e.g., 100x100 = 10,000 evaluations)
        - High-res: 1000x1000 = 1,000,000 evaluations!
        - Use subsampled dataset (10% of data still shows structure)
        - Parallelize across GPUs
        - Time: Hours to days for high-res visualizations
        """
        pass

    def fge_cost(self):
        """
        Cost of Fast Geometric Ensembling.

        - Training: Same as standard training (just different LR schedule)
        - Collection: Just save checkpoints (free)
        - Inference: K forward passes for K models in ensemble
        - Can batch across ensemble members for efficiency
        """
        pass

# Performance Tips

tips = """
1. CURVE TRAINING
   - Start with Bezier (simpler, often sufficient)
   - Use PolyChain if Bezier doesn't work
   - num_bends=1 usually enough
   - lr=0.03, epochs=200 works for most architectures

2. VISUALIZATION
   - Start with 100x100 grid
   - Use 10% of data for fast iteration
   - Increase resolution only for final figures
   - Consider 3D visualization libraries (plotly, mayavi)

3. FGE
   - cycle_length=2-4 epochs
   - lr_max around standard training LR
   - lr_min = lr_max / 10
   - Collect 10-20 models

4. EVALUATION
   - Cache network outputs for ensemble
   - Use mixed precision for faster forward passes
   - Parallelize across curve points

5. GIT RE-BASIN
   - Activation matching more stable than weight matching
   - Use representative data subset (1000-5000 samples)
   - Hungarian algorithm is O(n^3) - limit to <1000 neurons per layer
"""
```

### Memory-Efficient Implementation

```python
class EfficientModeConnectivity:
    """Memory-efficient mode connectivity implementation."""

    def __init__(self):
        pass

    def train_curve_memory_efficient(self, model1_path, model2_path, train_loader):
        """
        Train curve without loading both endpoints into memory.

        Key: Only load what's needed for current forward pass.
        """
        # Don't keep models in memory
        # Load parameters from disk as needed

        with h5py.File('curve_data.h5', 'w') as f:
            # Store endpoints on disk
            f.create_dataset('model1', data=serialize(torch.load(model1_path)))
            f.create_dataset('model2', data=serialize(torch.load(model2_path)))

        # Only theta (bend point) in GPU memory
        theta = self.initialize_bend_from_disk('curve_data.h5')
        optimizer = torch.optim.SGD(theta.values(), lr=0.03)

        for epoch in range(200):
            for x, y in train_loader:
                t = np.random.random()

                # Load and interpolate on-the-fly
                with h5py.File('curve_data.h5', 'r') as f:
                    params = self.bezier_interpolate(
                        f['model1'], f['model2'], theta, t
                    )

                # Forward pass
                loss = compute_loss(params, x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def ensemble_predict_streaming(self, models_dir, x):
        """
        Ensemble prediction without loading all models into memory.

        Stream models one at a time, accumulate predictions.
        """
        accumulated_logits = None
        count = 0

        for model_path in sorted(glob(f'{models_dir}/*.pt')):
            # Load one model at a time
            model = load_model(model_path)

            with torch.no_grad():
                logits = model(x)

            if accumulated_logits is None:
                accumulated_logits = logits
            else:
                accumulated_logits += logits

            count += 1

            # Free memory
            del model
            torch.cuda.empty_cache()

        return accumulated_logits / count
```

---

## Section 10: Summary and Key Takeaways

### Core Insights

1. **Mode connectivity exists universally**: All common architectures (VGG, ResNet, Transformers) exhibit mode connectivity

2. **Linear mode connectivity is special**: Not all solutions are linearly connected - this requires same initialization or permutation alignment

3. **Topology matters for generalization**: The connected structure of the loss landscape enables techniques like model averaging and ensembling

4. **Practical applications abound**: FGE, model merging, continual learning, transfer learning all exploit mode connectivity

### Code Summary

```python
# 1. Train endpoints
model1, model2 = train_two_networks(same_arch, different_seeds)

# 2. Find connecting curve
curve = BezierCurve(model1, model2, learnable_bend)
train_curve(curve, train_loader, epochs=200)

# 3. Evaluate path quality
path_losses = [evaluate(curve(t)) for t in np.linspace(0, 1, 51)]
assert max(path_losses) / min(path_losses) < 1.1  # Low barrier

# 4. Exploit connectivity
# - Fast Geometric Ensembling
fge = FastGeometricEnsembling(model, lr_min=0.001, lr_max=0.05)
ensemble_models = fge.train_and_collect(train_loader)

# - Model merging with Git Re-Basin
aligned_model2 = git_rebasin.align(model2, model1)
merged = average_models(model1, aligned_model2)

# - Ensemble along curve
ensemble_pred = mean([model(curve(t)) for t in [0, 0.25, 0.5, 0.75, 1.0]])
```

### The Deep Message

**Mode connectivity tells us that the loss landscape is simpler than we thought.**

Instead of a rugged terrain with isolated peaks and valleys, it's more like a vast plateau with a single connected basin of good solutions. This has profound implications:

- **For optimization**: We're not searching for a needle in a haystack - we're exploring a connected manifold
- **For generalization**: The geometry of this manifold (flat vs sharp) determines generalization
- **For understanding**: We can characterize neural network solutions topologically

The coffee cup IS the donut. Different neural network solutions ARE topologically equivalent. Mode connectivity is the proof.

---

## References

**Primary Papers:**
- [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026) - Garipov et al., NeurIPS 2018
- [Essentially No Barriers in Neural Network Energy Landscape](https://arxiv.org/abs/1803.00885) - Draxler et al., ICML 2018
- [Linear Mode Connectivity and the Lottery Ticket Hypothesis](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf) - Frankle et al., ICML 2020

**Extensions:**
- [Loss Surface Simplexes for Mode Connecting Volumes](http://proceedings.mlr.press/v139/benton21a/benton21a.pdf) - Benton et al., ICML 2021
- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836) - Ainsworth et al., 2022
- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) - Izmailov et al., UAI 2018

**Code:**
- [timgaripov/dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) - Official PyTorch implementation
- [Blog post with visualizations](https://izmailovpavel.github.io/curves_blogpost/) - Interactive visualizations

**Additional References:**
- [Going Beyond Linear Mode Connectivity](https://proceedings.neurips.cc/paper_files/paper/2023/file/bf3ee5a5422b0e2a88b0c9c6ed3b6144-Paper-Conference.pdf) - Zhou et al., NeurIPS 2023
- [Explaining Landscape Connectivity of Low-cost Solutions](http://papers.neurips.cc/paper/9602-explaining-landscape-connectivity-of-low-cost-solutions-for-multilayer-nets.pdf) - Kuditipudi et al., NeurIPS 2019
