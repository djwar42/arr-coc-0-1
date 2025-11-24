# Biologically Plausible Backpropagation Alternatives

## Overview

Backpropagation is the workhorse of deep learning, but it has fundamental biological implausibilities that have driven decades of research into alternative learning algorithms. This document explores why backprop isn't biologically plausible, and the major alternatives: target propagation, feedback alignment, equilibrium propagation, and local Hebbian learning rules.

**TRAIN STATION**: Local rules = Hebbian = self-organization = free energy minimization

The deep connection: ALL these alternatives are trying to achieve the same thing - credit assignment through local computations. This is exactly what biological neural circuits do, and it's also what emerges from free energy minimization and self-organizing systems.

---

## Section 1: Why Backpropagation Isn't Biologically Plausible

### The Weight Transport Problem

The most fundamental issue with backpropagation:

```
Forward pass:  y = W * x
Backward pass: grad_x = W^T * grad_y
```

**The problem**: The backward pass requires the TRANSPOSE of the forward weights.

This implies:
1. **Symmetric synapses** - Each synapse must know the weight of its reverse connection
2. **Perfect weight copying** - Forward and backward pathways must be identical
3. **No biological mechanism** - Neurons don't have access to synaptic weights of other neurons

### The Derivative Problem

Backprop requires computing f'(x) for the activation function:
- Neurons would need to compute their own derivatives
- No known biological mechanism for this
- Different neurons use different "activation functions"

### The Separate Phase Problem

Backprop needs distinct forward and backward phases:
- Information flows forward during inference
- Errors flow backward during learning
- Brain must somehow separate these temporally or spatially

### The Non-Local Credit Assignment Problem

To update weight W_ij:
```python
delta_W_ij = learning_rate * error_j * activation_i
```

But error_j depends on ALL downstream weights and errors - this is non-local information!

### Summary of Biological Implausibilities

From [Scellier & Bengio, 2017](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full):

| Issue | Backprop Requirement | Biological Reality |
|-------|---------------------|-------------------|
| Weight transport | W^T available | No synaptic weight access |
| Derivatives | Exact f'(x) | Unknown mechanism |
| Phases | Separate forward/backward | Continuous processing |
| Locality | Global error info | Only local signals |

---

## Section 2: Target Propagation

### Core Idea

Instead of propagating errors backward, propagate **targets** for each layer.

From [Ahmad et al., 2020](https://arxiv.org/abs/2006.06438) (GAIT-prop):

> "Target propagation proposes to solve this implausibility by using a top-down model of neural activity to convert an error at the output into layer-wise and plausible 'targets' for every unit."

### How It Works

```
Forward:  h_i = f(W_i * h_{i-1})
Target:   t_{i-1} = g(V_i * t_i)  # V is inverse/feedback pathway
Update:   delta_W_i = (t_{i-1} - h_{i-1}) * h_{i-1}^T
```

Where:
- g is a learned inverse function
- V_i learns to approximate the inverse of f(W_i * .)

### The Difference Reconstruction Loss

To learn good inverses:
```python
L_rec = ||h_{i-1} - g(V_i * f(W_i * h_{i-1}))||^2
```

This encourages V to be a true inverse of the forward computation.

### GAIT-prop: Exact Equivalence

GAIT-prop (Gradient Adjusted Incremental Target Propagation) achieves exact equivalence to backprop:

**Key insight**: When weight matrices are orthogonal, target propagation = backpropagation!

```python
# Target is a small perturbation of forward pass
t_i = h_i + epsilon * grad_h_i
```

### PyTorch Implementation: Target Propagation Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetPropLayer(nn.Module):
    """
    Target Propagation layer with learned inverse.

    Forward: y = activation(W @ x + b)
    Inverse: x_hat = inverse_activation(V @ y + c)
    """
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()

        # Forward pathway
        self.W = nn.Linear(in_features, out_features)

        # Feedback/inverse pathway (learned separately)
        self.V = nn.Linear(out_features, in_features)

        # Activation
        if activation == 'relu':
            self.activation = F.relu
            self.inverse_activation = F.relu  # Approximate inverse
        elif activation == 'tanh':
            self.activation = torch.tanh
            self.inverse_activation = torch.tanh

        # For storing activations
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x.detach().clone()
        self.output = self.activation(self.W(x))
        return self.output

    def compute_target(self, target_output):
        """
        Compute target for input given target for output.
        Uses learned inverse pathway.
        """
        return self.inverse_activation(self.V(target_output))

    def reconstruction_loss(self):
        """
        Loss to train the inverse pathway.
        """
        if self.input is None or self.output is None:
            return 0.0

        # Reconstruct input from output
        reconstructed = self.inverse_activation(self.V(self.output))
        return F.mse_loss(reconstructed, self.input)

    def local_update(self, target_input, lr=0.01):
        """
        Update forward weights using local target.

        The update is Hebbian-like: based on difference between
        target and actual input.
        """
        if self.input is None:
            return

        # Local error signal
        error = target_input - self.input

        # Update (simplified - actual would use proper gradient)
        with torch.no_grad():
            # delta_W proportional to error @ input^T
            delta = error.T @ self.output / error.shape[0]
            self.W.weight.data -= lr * delta.T


class TargetPropNetwork(nn.Module):
    """
    Full network using target propagation.
    """
    def __init__(self, layer_sizes):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                TargetPropLayer(layer_sizes[i], layer_sizes[i+1])
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def propagate_targets(self, output_target):
        """
        Propagate targets backward through network.
        """
        targets = [output_target]
        current_target = output_target

        # Go backward through layers
        for layer in reversed(self.layers):
            current_target = layer.compute_target(current_target)
            targets.insert(0, current_target)

        return targets

    def train_step(self, x, y, lr_forward=0.01, lr_inverse=0.01):
        """
        One training step with target propagation.
        """
        # Forward pass
        output = self.forward(x)

        # Compute output error
        output_error = y - output
        output_target = output + output_error  # Target = desired output

        # Propagate targets backward
        targets = self.propagate_targets(output_target)

        # Update each layer using its local target
        for i, layer in enumerate(self.layers):
            layer.local_update(targets[i], lr=lr_forward)

        # Update inverse pathways
        inverse_loss = sum(layer.reconstruction_loss() for layer in self.layers)

        return F.mse_loss(output, y), inverse_loss
```

### Performance Notes

From research:
- Target propagation achieves ~95-98% of backprop performance on MNIST/CIFAR
- Requires careful tuning of inverse learning
- Orthogonality regularization improves results significantly
- More memory efficient (no need to store all activations)

---

## Section 3: Feedback Alignment

### Core Idea

The simplest biologically plausible alternative: **use random fixed feedback weights!**

From [Lillicrap et al., 2016](https://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks):

> "Random feedback weights support learning in deep neural networks."

### How It Works

```
Forward:  y = W * x
Backward: grad_x = B * grad_y  # B is RANDOM and FIXED!
```

**Why does this work?** The forward weights W learn to align with the random B!

### Mathematical Insight

Over training, the angle between W^T and B decreases:
```
angle(W^T, B) -> small
```

The network essentially learns to make the random feedback useful.

### Variants

**1. Feedback Alignment (FA)**
- Each layer uses random B_i
- Error propagates layer by layer

**2. Direct Feedback Alignment (DFA)**
- Error goes directly from output to each layer
- Even simpler, sometimes works better

```python
# DFA: Each layer gets output error directly
grad_h_i = B_i @ output_error  # Direct, not through layers
```

### PyTorch Implementation: Feedback Alignment

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedbackAlignmentLinear(nn.Module):
    """
    Linear layer with feedback alignment.
    Uses random fixed feedback weights for backprop.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        # Forward weights (learned)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Feedback weights (random, fixed)
        self.register_buffer(
            'feedback',
            torch.randn(in_features, out_features) * 0.01
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class FeedbackAlignmentFunction(torch.autograd.Function):
    """
    Custom autograd for feedback alignment.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, feedback):
        ctx.save_for_backward(input, feedback)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, feedback = ctx.saved_tensors

        # Gradient for input uses RANDOM feedback weights
        grad_input = grad_output @ feedback.T

        # Gradient for weight uses normal gradient
        grad_weight = grad_output.T @ input

        # Gradient for bias
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class DirectFeedbackAlignmentNetwork(nn.Module):
    """
    Network using Direct Feedback Alignment.
    Error goes directly from output to each hidden layer.
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        self.layers = nn.ModuleList()
        self.feedbacks = nn.ParameterList()

        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))

            # Direct feedback from output to this layer (except last)
            if i < len(sizes) - 2:
                fb = torch.randn(sizes[i+1], output_size) * 0.01
                self.feedbacks.append(nn.Parameter(fb, requires_grad=False))

        self.activations = []

    def forward(self, x):
        self.activations = [x]

        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            self.activations.append(x)

        # Output layer (no activation)
        x = self.layers[-1](x)
        self.activations.append(x)

        return x

    def dfa_backward(self, output_error):
        """
        Direct Feedback Alignment backward pass.

        Each layer receives error directly from output,
        projected through its random feedback matrix.
        """
        gradients = []

        # Output layer gradient (normal)
        grad_output = output_error

        # Hidden layers get direct feedback
        for i in range(len(self.layers) - 2, -1, -1):
            # Direct feedback from output error
            if i < len(self.feedbacks):
                local_error = self.activations[i+1] * (
                    output_error @ self.feedbacks[i].T
                )
            else:
                local_error = grad_output

            # Compute weight gradient
            grad_weight = local_error.T @ self.activations[i]
            gradients.insert(0, grad_weight)

        return gradients

    def train_step(self, x, y, lr=0.01):
        """
        Training step with DFA.
        """
        # Forward
        output = self.forward(x)
        loss = F.mse_loss(output, y)

        # DFA backward
        output_error = output - y
        gradients = self.dfa_backward(output_error)

        # Update weights
        with torch.no_grad():
            for i, (layer, grad) in enumerate(zip(self.layers[:-1], gradients)):
                layer.weight.data -= lr * grad / x.shape[0]

        return loss


# Example usage
def train_dfa_example():
    """
    Example: Train a network with Direct Feedback Alignment.
    """
    # Create network
    net = DirectFeedbackAlignmentNetwork(
        input_size=784,
        hidden_sizes=[500, 200],
        output_size=10
    )

    # Dummy data
    x = torch.randn(32, 784)
    y = F.one_hot(torch.randint(0, 10, (32,)), 10).float()

    # Training loop
    for epoch in range(100):
        loss = net.train_step(x, y, lr=0.01)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return net
```

### Performance Notes

From research:
- Feedback Alignment: ~98% of backprop on MNIST
- Direct Feedback Alignment: Sometimes even better!
- Works surprisingly well on ImageNet with modifications
- Alignment between W^T and B emerges during training
- Very memory efficient (no need to store W for backward)

---

## Section 4: Equilibrium Propagation

### Core Idea

From [Scellier & Bengio, 2017](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full):

> "We introduce Equilibrium Propagation, a learning framework for energy-based models. It involves only one kind of neural computation, performed in both phases of training."

**Key insight**: Use the SAME dynamics for inference and learning!

### The Energy-Based View

The network minimizes an energy function:
```python
E(s) = (1/2) * sum(s_i^2) - (1/2) * sum(W_ij * rho(s_i) * rho(s_j)) - sum(b_i * rho(s_i))
```

### Two Phases

**Free Phase** (inference):
- Network settles to equilibrium with beta = 0
- This is the prediction

**Nudged Phase** (learning):
- Add small beta * Cost to energy
- Network settles to new equilibrium
- The DIFFERENCE between phases gives gradient!

### The Beautiful Learning Rule

```python
delta_W_ij = (1/beta) * (rho(s_i^beta) * rho(s_j^beta) - rho(s_i^0) * rho(s_j^0))
```

This is a **contrastive Hebbian rule**!

### Why It Works

The gradient of the objective is encoded in the difference between:
- Free equilibrium (prediction)
- Nudged equilibrium (slightly better prediction)

As beta -> 0, this exactly computes the gradient.

### Connection to STDP

The learning rule can be written as:
```python
dW_ij/dt = rho(s_i) * d(rho(s_j))/dt
```

This matches Spike-Timing Dependent Plasticity observations!

### PyTorch Implementation: Equilibrium Propagation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EqPropNetwork(nn.Module):
    """
    Equilibrium Propagation network.

    Based on continuous Hopfield networks with symmetric weights.
    """
    def __init__(self, layer_sizes, rho='hard_sigmoid'):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Symmetric weights between layers
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(layer_sizes) - 1):
            W = torch.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(torch.zeros(layer_sizes[i+1])))

        # Activation function
        if rho == 'hard_sigmoid':
            self.rho = lambda x: torch.clamp(x, 0, 1)
            self.rho_prime = lambda x: ((x > 0) & (x < 1)).float()
        elif rho == 'sigmoid':
            self.rho = torch.sigmoid
            self.rho_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def energy(self, states, x, beta=0, target=None):
        """
        Compute total energy function F = E + beta * C

        E = sum_i (s_i^2 / 2) - sum_{i,j} W_ij * rho(s_i) * rho(s_j) - sum_i b_i * rho(s_i)
        C = ||y - target||^2 / 2
        """
        E = 0

        # Include input in full state
        full_states = [x] + states

        # Squared term
        for s in states:
            E += 0.5 * (s ** 2).sum()

        # Interaction terms (symmetric weights)
        for i, W in enumerate(self.weights):
            rho_i = self.rho(full_states[i])
            rho_j = self.rho(full_states[i+1])
            E -= (rho_i @ W * rho_j).sum()

        # Bias terms
        for i, b in enumerate(self.biases):
            E -= (b * self.rho(states[i])).sum()

        # Cost term
        if beta > 0 and target is not None:
            y = states[-1]  # Output
            C = 0.5 * ((y - target) ** 2).sum()
            E += beta * C

        return E

    def compute_gradients(self, states, x):
        """
        Compute -dF/ds for each state.
        This determines the dynamics.
        """
        grads = []
        full_states = [x] + states

        for i in range(len(states)):
            s = states[i]

            # Gradient of squared term
            grad = -s

            # Gradient from interaction with previous layer
            W_prev = self.weights[i]
            rho_prev = self.rho(full_states[i])
            grad += self.rho_prime(s) * (rho_prev @ W_prev)

            # Gradient from interaction with next layer (if not output)
            if i < len(states) - 1:
                W_next = self.weights[i+1]
                rho_next = self.rho(full_states[i+2])
                grad += self.rho_prime(s) * (W_next @ rho_next)

            # Gradient from bias
            grad += self.rho_prime(s) * self.biases[i]

            grads.append(grad)

        return grads

    def relax(self, x, n_iters=100, epsilon=0.5, beta=0, target=None):
        """
        Relax network to equilibrium.

        Uses gradient descent on energy function.
        """
        batch_size = x.shape[0]

        # Initialize states
        states = []
        for size in self.layer_sizes[1:]:
            states.append(torch.zeros(batch_size, size))

        # Relax to equilibrium
        for _ in range(n_iters):
            grads = self.compute_gradients(states, x)

            # Add nudging force if beta > 0
            if beta > 0 and target is not None:
                # Force on output layer toward target
                grads[-1] += beta * (target - states[-1])

            # Update states
            for i in range(len(states)):
                states[i] = states[i] + epsilon * grads[i]
                states[i] = torch.clamp(states[i], 0, 1)  # Keep in valid range

        return states

    def train_step(self, x, y, beta=0.5, n_free=100, n_nudge=20, lr=0.01):
        """
        One training step with Equilibrium Propagation.

        1. Free phase: relax to equilibrium with beta=0
        2. Nudged phase: relax to equilibrium with beta>0
        3. Update weights using contrastive Hebbian rule
        """
        # Free phase
        states_free = self.relax(x, n_iters=n_free, beta=0)

        # Nudged phase (start from free phase equilibrium)
        states_nudge = [s.clone() for s in states_free]
        for _ in range(n_nudge):
            grads = self.compute_gradients(states_nudge, x)
            grads[-1] += beta * (y - states_nudge[-1])
            for i in range(len(states_nudge)):
                states_nudge[i] = states_nudge[i] + 0.5 * grads[i]
                states_nudge[i] = torch.clamp(states_nudge[i], 0, 1)

        # Contrastive Hebbian update
        full_free = [x] + states_free
        full_nudge = [x] + states_nudge

        with torch.no_grad():
            for i, W in enumerate(self.weights):
                rho_i_free = self.rho(full_free[i])
                rho_j_free = self.rho(full_free[i+1])
                rho_i_nudge = self.rho(full_nudge[i])
                rho_j_nudge = self.rho(full_nudge[i+1])

                # Contrastive Hebbian rule
                delta_W = (1/beta) * (
                    rho_i_nudge.T @ rho_j_nudge -
                    rho_i_free.T @ rho_j_free
                ) / x.shape[0]

                W.data += lr * delta_W

        # Compute loss
        loss = F.mse_loss(states_free[-1], y)
        return loss


def train_eqprop_example():
    """
    Example: Train with Equilibrium Propagation.
    """
    net = EqPropNetwork([784, 500, 10])

    # Dummy data
    x = torch.randn(32, 784)
    y = F.one_hot(torch.randint(0, 10, (32,)), 10).float()

    for epoch in range(50):
        loss = net.train_step(x, y, beta=0.5, lr=0.1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return net
```

### Performance Notes

From [Scellier & Bengio, 2017]:
- Achieves 0.00% training error on MNIST
- 2-3% test error with proper architecture
- Requires long relaxation (hundreds of iterations)
- Very promising for analog neuromorphic hardware
- Has been scaled to ConvNets (Laborieux et al., 2021)

---

## Section 5: Local Hebbian Learning Rules

### The Hebbian Principle

> "Neurons that fire together, wire together."

```python
delta_W_ij = eta * x_i * y_j
```

### The Problem with Pure Hebbian Learning

Pure Hebbian learning:
- Only strengthens connections
- Leads to unbounded weight growth
- No mechanism for credit assignment

### Modern Hebbian-Inspired Rules

**1. Oja's Rule (Normalized Hebbian)**
```python
delta_W = eta * y * (x - y * W)
```
Performs PCA!

**2. BCM Rule (Bienenstock-Cooper-Munro)**
```python
delta_W = eta * y * (y - theta) * x
```
Where theta is a sliding threshold.

**3. Contrastive Hebbian Learning**
```python
delta_W = eta * (x_+ * y_+ - x_- * y_-)
```
Difference between two phases.

### PyTorch Implementation: Local Hebbian Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLayer(nn.Module):
    """
    Layer with local Hebbian learning rule.

    Implements several variants of Hebbian learning.
    """
    def __init__(self, in_features, out_features, rule='oja'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rule = rule

        # Weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        # For BCM rule: sliding threshold
        if rule == 'bcm':
            self.register_buffer('theta', torch.ones(out_features))

        # Store activations
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x.detach()
        self.output = F.linear(x, self.weight)
        return self.output

    def hebbian_update(self, lr=0.01):
        """
        Apply local Hebbian learning rule.
        """
        if self.input is None or self.output is None:
            return

        x = self.input  # (batch, in)
        y = self.output  # (batch, out)

        with torch.no_grad():
            if self.rule == 'basic':
                # Basic Hebbian: delta_W = eta * y^T @ x
                delta_W = y.T @ x / x.shape[0]

            elif self.rule == 'oja':
                # Oja's rule: delta_W = eta * y @ (x - y @ W)
                # Normalizes weights, performs PCA
                delta_W = y.T @ (x - y @ self.weight) / x.shape[0]

            elif self.rule == 'bcm':
                # BCM rule: delta_W = eta * y * (y - theta) @ x
                # Sliding threshold provides homeostasis
                modulation = y * (y - self.theta.unsqueeze(0))
                delta_W = modulation.T @ x / x.shape[0]

                # Update threshold (slow)
                self.theta = 0.99 * self.theta + 0.01 * (y ** 2).mean(0)

            elif self.rule == 'anti_hebbian':
                # Anti-Hebbian for decorrelation
                delta_W = -y.T @ x / x.shape[0]

            self.weight.data += lr * delta_W


class LocalLearningNetwork(nn.Module):
    """
    Network trained with local learning rules only.

    Each layer learns independently using Hebbian rules.
    No backpropagation!
    """
    def __init__(self, layer_sizes, rule='oja'):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                HebbianLayer(layer_sizes[i], layer_sizes[i+1], rule=rule)
            )

        # Final layer uses supervised signal
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        # Hebbian layers (unsupervised)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Output layer
        x = self.layers[-1](x)
        return x

    def local_train_step(self, x, y=None, lr_hebbian=0.01, lr_supervised=0.01):
        """
        Training step with local rules.

        Hidden layers use Hebbian learning.
        Output layer uses supervised gradient (or also Hebbian).
        """
        # Forward pass
        output = self.forward(x)

        # Hebbian updates for hidden layers
        for layer in self.layers[:-1]:
            layer.hebbian_update(lr=lr_hebbian)

        # Supervised update for output layer
        if y is not None:
            loss = F.mse_loss(output, y)

            # Simple gradient update for output
            with torch.no_grad():
                error = output - y
                self.layers[-1].weight.data -= lr_supervised * (
                    error.T @ self.layers[-1].input / x.shape[0]
                )

            return loss

        return None


class InfoNCEHebbianLayer(nn.Module):
    """
    Hebbian layer with Information Bottleneck objective.

    Based on recent work connecting Hebbian learning to
    information-theoretic objectives.
    """
    def __init__(self, in_features, out_features, temperature=0.1):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.temperature = temperature

    def forward(self, x):
        return F.linear(x, self.weight)

    def contrastive_hebbian_update(self, x_pos, x_neg, lr=0.01):
        """
        Contrastive Hebbian update.

        Strengthen connections for positive pairs,
        weaken for negative pairs.
        """
        y_pos = self.forward(x_pos)
        y_neg = self.forward(x_neg)

        with torch.no_grad():
            # Positive phase: strengthen
            delta_pos = y_pos.T @ x_pos / x_pos.shape[0]

            # Negative phase: weaken
            delta_neg = y_neg.T @ x_neg / x_neg.shape[0]

            # Contrastive update
            self.weight.data += lr * (delta_pos - delta_neg)


def train_local_example():
    """
    Example: Train with local Hebbian rules.
    """
    net = LocalLearningNetwork([784, 500, 200, 10], rule='oja')

    # Dummy data
    x = torch.randn(32, 784)
    y = F.one_hot(torch.randint(0, 10, (32,)), 10).float()

    for epoch in range(100):
        loss = net.local_train_step(x, y, lr_hebbian=0.001, lr_supervised=0.01)
        if epoch % 20 == 0 and loss is not None:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return net
```

### Performance Notes

Local Hebbian learning:
- Works well for unsupervised feature learning
- Can achieve ~95% on MNIST with proper setup
- Recently shown to match backprop in some settings (Journé et al., 2022)
- Very efficient: no backward pass needed
- Natural for neuromorphic hardware

---

## Section 6: TRAIN STATION - Local Rules = Hebbian = Self-Organization

### The Deep Unification

All biologically plausible alternatives converge on the same fundamental principle:

**LOCAL COMPUTATION + CONTRASTIVE LEARNING = GRADIENT DESCENT**

```
Target Propagation:  local targets - local actual
Feedback Alignment:  random projection of global error
Equilibrium Prop:    nudged state - free state
Hebbian Learning:    positive phase - negative phase
```

### The Free Energy Connection

From Friston's Free Energy Principle, learning is:

```python
delta_W = -dF/dW
```

Where F is free energy. This naturally leads to:
- Local computation (only local variables)
- Contrastive phases (prediction vs observation)
- Hebbian-like rules (correlation-based)

### Why Local Rules Lead to Self-Organization

Local rules + global objective = emergent structure!

This is exactly what happens in:
- Morphogenesis (local signals, global pattern)
- Neural development (Hebbian wiring, functional circuits)
- Deep learning (local gradients, global features)

### The Mathematical Connection

All methods approximate:
```python
dL/dW = dL/dy * dy/dW
```

But they compute dL/dy differently:
- **Backprop**: Exact transpose
- **Target prop**: Learned inverse
- **Feedback alignment**: Random projection
- **Eq prop**: Energy gradient
- **Hebbian**: Correlation difference

### Train Station Summary

```
         LOCAL RULES
              |
     +--------+--------+
     |        |        |
  Hebbian   Target   Equilibrium
              |
     +--------+--------+
     |                 |
Free Energy      Self-Organization
     |                 |
     +--------+--------+
              |
      EMERGENT STRUCTURE
```

**The train station**: Every path leads to the same destination -
credit assignment through local computation emerges from the
fundamental requirement that learning be physically realizable!

---

## Section 7: ARR-COC-0-1 - Local Relevance Computation

### Connection to Token Allocation

ARR-COC allocates computational resources (tokens) based on relevance.
The biologically plausible methods suggest:

**Local relevance computation without global backprop!**

### Proposed Approach: Local Attention as Hebbian Learning

```python
class LocalRelevanceAttention(nn.Module):
    """
    Attention mechanism learned with local Hebbian rules.

    Each attention head learns to weight features using
    only local information.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, Key, Value projections
        self.q_proj = HebbianLayer(dim, dim, rule='oja')
        self.k_proj = HebbianLayer(dim, dim, rule='oja')
        self.v_proj = HebbianLayer(dim, dim, rule='oja')

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        # Compute Q, K, V with Hebbian-learned projections
        q = self.q_proj(x.view(-1, D)).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x.view(-1, D)).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x.view(-1, D)).view(B, N, self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, D)

        return self.out_proj(out), attn

    def local_update(self, lr=0.001):
        """
        Update projections using local Hebbian learning.
        """
        self.q_proj.hebbian_update(lr)
        self.k_proj.hebbian_update(lr)
        self.v_proj.hebbian_update(lr)
```

### Benefits for ARR-COC

1. **No global backprop needed** - Each layer learns independently
2. **Biological plausibility** - Could run on neuromorphic hardware
3. **Memory efficiency** - No need to store activations
4. **Online learning** - Can adapt continuously
5. **Local relevance** - Each region computes its own importance

### Relevance as Free Energy Minimization

In the free energy framework:
- **Relevant tokens** = High precision (low uncertainty)
- **Irrelevant tokens** = Low precision (high uncertainty)

Local learning naturally discovers this structure!

```python
# Relevance score from local computation
relevance = precision * prediction_error
          = attention_weight * feature_magnitude
```

### Implementation Note

For ARR-COC, we could:
1. Use Hebbian-learned attention for initial relevance estimates
2. Apply equilibrium propagation for fine-tuning
3. Use feedback alignment for the allocation network

This gives us the best of both worlds:
- **Biological plausibility** from local rules
- **Performance** from gradient-based fine-tuning

---

## Summary

### Key Takeaways

1. **Backprop is biologically implausible** due to weight transport, derivative computation, and non-locality

2. **Target propagation** uses learned inverses to propagate targets instead of errors

3. **Feedback alignment** surprisingly works with random fixed feedback weights

4. **Equilibrium propagation** uses energy-based dynamics with contrastive Hebbian learning

5. **Local Hebbian rules** can achieve competitive performance with proper setup

6. **All methods converge** on local computation + contrastive learning

### Performance Comparison

| Method | MNIST | CIFAR-10 | Biological Plausibility |
|--------|-------|----------|------------------------|
| Backprop | 99%+ | 95%+ | Low |
| Target Prop | 98% | 90% | Medium |
| Feedback Align | 98% | 85% | High |
| Equilibrium Prop | 97% | 85% | Very High |
| Hebbian | 95% | 80% | Very High |

### Future Directions

1. **Scaling** - Can these methods work on ImageNet/transformers?
2. **Hardware** - Neuromorphic implementations
3. **Hybrid** - Combine methods for best performance
4. **Theory** - Deeper understanding of why they work

---

## Sources

**Source Documents:**
- Ingestion plan from PLATONIC-DIALOGUES/67

**Web Research (accessed 2025-11-23):**
- [Equilibrium Propagation Paper](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full) - Scellier & Bengio, 2017
- [GAIT-prop Paper](https://arxiv.org/abs/2006.06438) - Ahmad et al., 2020
- [Direct Feedback Alignment](https://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks) - Nokland, 2016
- [PyTorch-Hebbian](https://arxiv.org/abs/2102.00428) - Talloen et al., 2021
- [Scaling Equilibrium Propagation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7930909/) - Laborieux et al., 2021

**Additional References:**
- Lillicrap et al. (2016) - Random feedback weights support learning
- Bengio et al. (2015) - Towards biologically plausible deep learning
- Xie & Seung (2003) - Equivalence of backprop and contrastive Hebbian
- Journé et al. (2022) - Hebbian deep learning without feedback
- Hinton (2022) - Forward-forward algorithm

**Code References:**
- [Equilibrium Propagation GitHub](https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop)
- PyTorch implementations derived from papers

---

## Train Station Connections

This document connects to:
- `../ml-active-inference/` - Free energy minimization
- `../ml-train-stations/00-loss-free-energy-relevance.md` - Loss = Free Energy
- `../ml-morphogenesis/03-self-organizing-nn.md` - Self-organization
- `../ml-train-stations/04-self-organization-unified.md` - Self-org everywhere

**The coffee cup = donut equivalence**:
Backprop and Hebbian learning are topologically equivalent -
they're just different ways of navigating the same optimization landscape!
