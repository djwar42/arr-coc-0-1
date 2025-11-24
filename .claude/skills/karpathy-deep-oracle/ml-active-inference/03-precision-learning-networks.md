# Precision Learning Networks: Attention as Learnable Uncertainty

## Overview

Precision learning networks represent a fundamental unification between attention mechanisms in deep learning and precision weighting in predictive coding and active inference frameworks. The core insight is that **attention IS precision** - both are mechanisms for modulating the gain or reliability of information channels based on estimated uncertainty.

This document explores how neural networks can learn to estimate and use precision (inverse variance) as a first-class computational primitive, enabling dynamic uncertainty quantification and adaptive information routing.

---

## 1. Precision as Learnable Parameter

### 1.1 From Fixed Noise to Learned Uncertainty

Traditional neural networks assume homoscedastic noise - constant variance across all predictions:

```python
# Traditional: Fixed variance assumption
loss = MSELoss(prediction, target)  # Assumes sigma^2 = 1
```

Precision learning networks instead output both mean and precision:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrecisionNetwork(nn.Module):
    """
    Network that outputs both predictions and their precision.

    The precision (inverse variance) tells us how confident
    the network is about each prediction.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for mean and log-precision
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_precision_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.features(x)

        # Predict mean
        mu = self.mean_head(features)

        # Predict log-precision (more numerically stable than precision directly)
        # log_precision = log(1/sigma^2) = -2*log(sigma)
        log_precision = self.log_precision_head(features)

        # Clamp for numerical stability
        log_precision = torch.clamp(log_precision, min=-10, max=10)

        return mu, log_precision

    def precision_weighted_loss(self, mu, log_precision, target):
        """
        Negative log-likelihood under heteroscedastic Gaussian.

        NLL = 0.5 * precision * (mu - target)^2 - 0.5 * log(precision)

        This loss automatically balances:
        - Accuracy: precision * error^2 (high precision = big penalty for errors)
        - Calibration: -log(precision) (prevents precision from going to infinity)
        """
        precision = torch.exp(log_precision)

        # Squared error weighted by precision
        weighted_error = 0.5 * precision * (mu - target) ** 2

        # Regularization term that prevents trivial solution
        # (can't just predict infinite variance to make loss 0)
        precision_penalty = -0.5 * log_precision

        loss = weighted_error + precision_penalty
        return loss.mean()
```

### 1.2 Why Learn Precision?

Learning precision provides several key benefits:

**1. Uncertainty Quantification**: The network tells us where it's confident vs uncertain
**2. Robust Training**: High-noise regions contribute less to gradients
**3. Attention-like Behavior**: Precision acts as soft attention over features/predictions

From [Sluijterman et al., 2024](https://www.sciencedirect.com/science/article/pii/S0925231224007008):
> "The optimal training of Mean Variance Estimation networks requires careful balancing between the prediction accuracy and uncertainty calibration terms."

### 1.3 The Precision-Weighted Prediction Error

The fundamental computation in precision learning:

```python
def precision_weighted_prediction_error(prediction, target, precision):
    """
    Core computation shared by:
    - Heteroscedastic neural networks
    - Predictive coding
    - Active inference
    - Kalman filtering

    Higher precision = prediction error matters MORE
    Lower precision = prediction error matters LESS

    This is EXACTLY what attention does!
    """
    error = target - prediction
    weighted_error = precision * error
    return weighted_error
```

---

## 2. Heteroscedastic Neural Networks

### 2.1 Architecture Patterns

Heteroscedastic networks predict input-dependent variance, enabling the network to express "I don't know" for uncertain inputs.

```python
class HeteroscedasticRegressor(nn.Module):
    """
    Full heteroscedastic regression network with:
    - Mean prediction
    - Variance prediction (input-dependent)
    - Proper negative log-likelihood loss
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=1):
        super().__init__()

        # Build shared trunk
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Mean head
        self.mu_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, output_dim)
        )

        # Log-variance head (outputs log(sigma^2))
        self.logvar_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, output_dim)
        )

        # Initialize logvar head to predict small variance initially
        nn.init.constant_(self.logvar_head[-1].bias, -2.0)

    def forward(self, x):
        features = self.trunk(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)

        # Clamp for stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar

    def nll_loss(self, mu, logvar, target):
        """
        Negative log-likelihood for Gaussian with learned variance.

        -log p(y|x) = 0.5 * log(2*pi*sigma^2) + 0.5 * (y-mu)^2 / sigma^2
                    = 0.5 * logvar + 0.5 * (y-mu)^2 * exp(-logvar) + const
        """
        precision = torch.exp(-logvar)  # 1/sigma^2

        nll = 0.5 * logvar + 0.5 * precision * (target - mu) ** 2

        return nll.mean()

    def predict_with_uncertainty(self, x):
        """Get prediction with confidence intervals."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x)
            std = torch.exp(0.5 * logvar)

        return {
            'mean': mu,
            'std': std,
            'lower_95': mu - 1.96 * std,
            'upper_95': mu + 1.96 * std
        }
```

### 2.2 Training Considerations

From [Deka et al., 2024](https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_TAGIV_2024_preprint.pdf):

```python
class HeteroscedasticTrainer:
    """
    Training utilities for heteroscedastic networks.

    Key challenges:
    1. Variance collapse (predicting constant high variance)
    2. Mean-variance coupling (errors in mean affect variance learning)
    3. Numerical stability with extreme precisions
    """

    def __init__(self, model, lr=1e-3, min_logvar=-10, max_logvar=10):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        mu, logvar = self.model(x)

        # Clamp logvar for stability
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        # NLL loss
        loss = self.model.nll_loss(mu, logvar, y)

        # Optional: Add regularization to prevent variance collapse
        # This encourages the network to actually use the variance
        variance_regularization = 0.01 * torch.mean(torch.exp(logvar))

        total_loss = loss + variance_regularization
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'loss': loss.item(),
            'mean_logvar': logvar.mean().item(),
            'std_logvar': logvar.std().item()
        }
```

### 2.3 Analytical Variance Inference

From [Deka et al., 2024](https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_TAGIV_2024_preprint.pdf), the Approximate Gaussian Variance Inference (AGVI) method enables analytical (closed-form) inference of variance:

```python
class AGVILayer(nn.Module):
    """
    Approximate Gaussian Variance Inference layer.

    Instead of learning variance with a separate head,
    derive it analytically from prediction errors.

    Key insight: Variance can be estimated from the
    running average of squared prediction errors.
    """

    def __init__(self, output_dim, momentum=0.99):
        super().__init__()
        self.momentum = momentum

        # Running estimate of variance per output dimension
        self.register_buffer('running_var', torch.ones(output_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0))

    def forward(self, mu, target=None):
        if self.training and target is not None:
            # Update running variance estimate
            with torch.no_grad():
                batch_var = ((target - mu) ** 2).mean(dim=0)

                if self.num_batches_tracked == 0:
                    self.running_var.copy_(batch_var)
                else:
                    self.running_var.mul_(self.momentum).add_(
                        batch_var, alpha=1 - self.momentum
                    )
                self.num_batches_tracked.add_(1)

        # Return current variance estimate
        return self.running_var.expand_as(mu)
```

---

## 3. Attention = Precision Weighting

### 3.1 The Core Equivalence

This is the **TRAIN STATION** where attention mechanisms meet precision weighting:

From [Feldman & Friston, 2010](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00215/full):
> "We suggested recently that attention can be understood as inferring the level of uncertainty or precision during hierarchical perception."

From [Mazzaglia et al., 2022](https://arxiv.org/abs/2207.06415):
> "In active inference implementations, precision has been employed as a form of attention, to decide on which transitions the model should focus."

```python
class AttentionAsPrecision(nn.Module):
    """
    Demonstrates the equivalence between attention and precision weighting.

    Standard attention: softmax(QK^T / sqrt(d)) * V
    Precision weighting: precision * prediction_error

    Both are GAIN CONTROL mechanisms that modulate information flow
    based on estimated reliability/relevance.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores = precision weights
        # Higher score = higher precision = more reliable information
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Softmax converts to proper precision weights (sum to 1)
        # This is like normalizing precisions across sources
        precision_weights = F.softmax(scores, dim=-1)

        # Weighted combination = precision-weighted average
        # Each value is weighted by its estimated reliability
        output = torch.matmul(precision_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output), precision_weights

    def precision_interpretation(self):
        """
        How to interpret attention weights as precision:

        1. Query = "what am I looking for" (prediction)
        2. Key = "what do I have" (sensory data)
        3. QK similarity = "how relevant is this data to my prediction"
        4. High similarity = high precision = trust this data more

        This is EXACTLY predictive coding:
        - Prediction errors from high-precision channels drive learning
        - Prediction errors from low-precision channels are ignored
        """
        pass
```

### 3.2 Gain Control: The Unified Mechanism

Both attention and precision implement **gain control** - multiplicative modulation of signals:

```python
class GainControlModule(nn.Module):
    """
    Unified gain control mechanism that can be interpreted as:
    - Attention (ML perspective)
    - Precision weighting (Bayesian perspective)
    - Synaptic gain modulation (neuroscience perspective)

    All three are the SAME computation!
    """

    def __init__(self, dim):
        super().__init__()

        # Gain predictor (outputs log-gain for stability)
        self.gain_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, context=None):
        """
        Apply gain control to input x, optionally using context.

        Args:
            x: Input signals (e.g., prediction errors)
            context: Optional context for computing gain
        """
        if context is None:
            context = x

        # Predict log-gain (precision)
        log_gain = self.gain_predictor(context)

        # Convert to gain (precision)
        gain = torch.exp(log_gain)

        # Apply gain control (precision weighting)
        output = gain * x

        return output, gain

    def as_attention(self, queries, keys, values):
        """Interpret as attention mechanism."""
        # Similarity = precision estimate
        similarity = torch.matmul(queries, keys.transpose(-2, -1))
        precision = F.softmax(similarity, dim=-1)
        return torch.matmul(precision, values)

    def as_precision_weighting(self, prediction_error, estimated_precision):
        """Interpret as precision weighting."""
        return estimated_precision * prediction_error

    def as_synaptic_gain(self, presynaptic, gain_modulation):
        """Interpret as synaptic gain modulation."""
        return gain_modulation * presynaptic
```

### 3.3 State-Dependent Precision

A key insight from Friston's work: precision depends on the STATE of the world, not just the data.

```python
class StateDependentPrecision(nn.Module):
    """
    Precision that depends on hidden states, not just inputs.

    This is crucial for attention because:
    - The SAME visual input has different precision depending on
      where you're attending (searchlight metaphor)
    - Precision is a property of the generative model, not the data

    From Feldman & Friston 2010:
    "If the precision depends on the states, one can explain many
    aspects of attention."
    """

    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()

        # State evolution (slow dynamics)
        self.state_update = nn.GRUCell(input_dim, state_dim)

        # Precision prediction from state
        self.precision_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, output_dim)
        )

        # Prediction from state and input
        self.predictor = nn.Linear(state_dim + input_dim, output_dim)

    def forward(self, x, state=None):
        batch_size = x.shape[0]

        if state is None:
            state = torch.zeros(batch_size, self.state_update.hidden_size, device=x.device)

        # Update state based on input
        new_state = self.state_update(x, state)

        # Predict precision from state (not from input!)
        log_precision = self.precision_predictor(new_state)
        precision = torch.exp(log_precision)

        # Make prediction
        prediction = self.predictor(torch.cat([new_state, x], dim=-1))

        return prediction, precision, new_state

    def attention_interpretation(self):
        """
        The state encodes WHERE attention is directed.

        Example: Posner cueing task
        - Cue presented on right -> state encodes "right has high precision"
        - Target on right -> high precision -> fast, accurate detection
        - Target on left -> low precision -> slow, inaccurate detection

        The state persists and decays slowly, explaining:
        - Benefits of valid cues
        - Costs of invalid cues
        - Time course of attentional effects
        """
        pass
```

---

## 4. Code: Precision-Weighted Prediction Errors

### 4.1 Complete Predictive Coding Layer with Precision

```python
class PrecisionWeightedPredictiveCodingLayer(nn.Module):
    """
    A single layer of a predictive coding network with learned precision.

    This implements the core equations from Friston's free energy formulation:
    - Prediction errors are weighted by precision
    - Precision is learned from data
    - State updates minimize precision-weighted prediction error
    """

    def __init__(self, input_dim, state_dim, output_dim, n_iterations=10):
        super().__init__()
        self.n_iterations = n_iterations

        # Generative model: state -> prediction
        self.generative = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, output_dim)
        )

        # State dynamics
        self.state_transition = nn.Linear(state_dim, state_dim)

        # Precision predictor (state-dependent)
        self.log_precision = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, output_dim)
        )

        # Learning rate for state updates
        self.state_lr = nn.Parameter(torch.tensor(0.1))

        # Initial state
        self.initial_state = nn.Parameter(torch.zeros(state_dim))

        self.state_dim = state_dim
        self.output_dim = output_dim

    def forward(self, target, n_iterations=None):
        """
        Perform iterative inference to find state that minimizes
        precision-weighted prediction error.

        This is the core of predictive coding / active inference.
        """
        if n_iterations is None:
            n_iterations = self.n_iterations

        batch_size = target.shape[0]

        # Initialize state
        state = self.initial_state.expand(batch_size, -1).clone()
        state.requires_grad_(True)

        # Iterative inference
        for _ in range(n_iterations):
            # Generate prediction from current state
            prediction = self.generative(state)

            # Compute prediction error
            error = target - prediction

            # Compute precision (state-dependent!)
            log_prec = self.log_precision(state)
            precision = torch.exp(torch.clamp(log_prec, -10, 10))

            # Precision-weighted prediction error
            weighted_error = precision * error

            # Free energy (to minimize)
            free_energy = 0.5 * (weighted_error ** 2).sum(dim=-1) - 0.5 * log_prec.sum(dim=-1)

            # Compute gradient of free energy w.r.t. state
            grad = torch.autograd.grad(
                free_energy.sum(), state, create_graph=True
            )[0]

            # Update state to minimize free energy
            state = state - self.state_lr * grad

        # Final predictions
        final_prediction = self.generative(state)
        final_log_precision = self.log_precision(state)

        return {
            'state': state,
            'prediction': final_prediction,
            'log_precision': final_log_precision,
            'precision': torch.exp(final_log_precision),
            'error': target - final_prediction
        }

    def loss(self, target):
        """
        Compute free energy loss for training.
        """
        result = self.forward(target)

        precision = result['precision']
        error = result['error']
        log_precision = result['log_precision']

        # Free energy
        free_energy = 0.5 * (precision * error ** 2).sum(dim=-1)
        free_energy = free_energy - 0.5 * log_precision.sum(dim=-1)

        return free_energy.mean()
```

### 4.2 Multi-Scale Precision Network

```python
class HierarchicalPrecisionNetwork(nn.Module):
    """
    Hierarchical network with precision at each level.

    Higher levels predict lower levels.
    Precision at each level modulates prediction error.

    This implements the hierarchical predictive coding scheme
    from Friston's free energy principle.
    """

    def __init__(self, dims=[784, 256, 64, 16]):
        super().__init__()

        self.n_levels = len(dims) - 1

        # Generative models (top-down predictions)
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i+1], dims[i+1]),
                nn.ReLU(),
                nn.Linear(dims[i+1], dims[i])
            )
            for i in range(self.n_levels)
        ])

        # Inference models (bottom-up recognition)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i]),
                nn.ReLU(),
                nn.Linear(dims[i], dims[i+1])
            )
            for i in range(self.n_levels)
        ])

        # Precision predictors at each level
        self.precision_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i+1], dims[i+1] // 2),
                nn.ReLU(),
                nn.Linear(dims[i+1] // 2, dims[i])
            )
            for i in range(self.n_levels)
        ])

    def forward(self, x, n_iterations=5):
        """
        Hierarchical inference with precision weighting.
        """
        batch_size = x.shape[0]

        # Bottom-up pass to initialize states
        states = [x]
        for encoder in self.encoders:
            states.append(encoder(states[-1]))

        # Iterative refinement with precision weighting
        for _ in range(n_iterations):
            errors = []
            precisions = []

            # Top-down predictions and precision-weighted errors
            for level in range(self.n_levels - 1, -1, -1):
                # Prediction from level above
                prediction = self.generators[level](states[level + 1])

                # Precision at this level
                log_precision = self.precision_predictors[level](states[level + 1])
                precision = torch.exp(torch.clamp(log_precision, -10, 10))

                # Prediction error
                if level == 0:
                    error = x - prediction
                else:
                    error = states[level] - prediction

                # Precision-weighted error
                weighted_error = precision * error

                errors.append(error)
                precisions.append(precision)

                # Update state at level below (except for input)
                if level > 0:
                    # Gradient of free energy
                    grad = weighted_error
                    states[level] = states[level] + 0.1 * grad

        return {
            'states': states,
            'errors': errors,
            'precisions': precisions,
            'reconstruction': self.generators[0](states[1])
        }
```

### 4.3 Precision-Weighted Attention Layer

```python
class PrecisionWeightedAttention(nn.Module):
    """
    Attention mechanism explicitly formulated as precision weighting.

    Instead of softmax(QK^T), we learn explicit precision weights
    for each key-value pair.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Precision predictor for each attention position
        self.precision_predictor = nn.Sequential(
            nn.Linear(self.d_k * 2, self.d_k),
            nn.ReLU(),
            nn.Linear(self.d_k, 1)
        )

    def forward(self, x, return_precision=False):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute precision for each Q-K pair
        # Shape: (batch, heads, seq, seq, 2*d_k)
        Q_expanded = Q.unsqueeze(3).expand(-1, -1, -1, seq_len, -1)
        K_expanded = K.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        QK_concat = torch.cat([Q_expanded, K_expanded], dim=-1)

        # Predict log-precision
        log_precision = self.precision_predictor(QK_concat).squeeze(-1)
        log_precision = torch.clamp(log_precision, -10, 10)

        # Normalize precision across keys (like softmax in standard attention)
        precision = F.softmax(log_precision, dim=-1)

        # Precision-weighted values
        output = torch.matmul(precision, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        if return_precision:
            return output, precision
        return output
```

---

## 5. TRAIN STATION: Attention = Precision = Gain Control

### 5.1 The Unified View

This is where three different fields converge on the SAME mechanism:

| Deep Learning | Bayesian/Active Inference | Neuroscience |
|--------------|---------------------------|--------------|
| Attention weights | Precision weighting | Synaptic gain |
| softmax(QK^T) | exp(-log_variance) | Post-synaptic responsiveness |
| Multiplicative gating | Inverse variance | Gain modulation |
| "What to focus on" | "How reliable is this" | "How strongly to respond" |

### 5.2 Mathematical Equivalence

```python
def demonstrate_equivalence():
    """
    Show that attention, precision, and gain control are the same.
    """
    # Same computation, three interpretations:

    # 1. Attention
    query = torch.randn(1, 64)
    keys = torch.randn(10, 64)
    values = torch.randn(10, 64)

    attention_scores = torch.matmul(query, keys.T) / 8.0
    attention_weights = F.softmax(attention_scores, dim=-1)
    attended = torch.matmul(attention_weights, values)

    # 2. Precision weighting
    predictions = torch.randn(10, 64)
    errors = values - predictions
    precisions = attention_weights  # SAME WEIGHTS
    weighted_errors = precisions.unsqueeze(-1) * errors

    # 3. Gain control
    inputs = values
    gains = attention_weights  # SAME WEIGHTS
    gated_inputs = gains.unsqueeze(-1) * inputs

    print("Attention, precision weighting, and gain control")
    print("all use the SAME multiplicative weighting mechanism")
    print(f"Weights sum to 1: {attention_weights.sum().item():.4f}")
```

### 5.3 Implications for Architecture Design

```python
class UnifiedAttentionPrecisionGain(nn.Module):
    """
    A module that makes the attention-precision-gain equivalence explicit.

    Design principles:
    1. Weights represent PRECISION (inverse variance)
    2. High precision = high attention = high gain
    3. Precision should be learned, not hand-coded
    4. Precision can be state-dependent
    """

    def __init__(self, dim, n_sources):
        super().__init__()

        # Learn precision as a function of content
        self.precision_network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_sources)
        )

        # State for dynamic precision (attention persistence)
        self.register_buffer('precision_state', torch.zeros(n_sources))
        self.state_momentum = 0.9

    def forward(self, sources, context=None):
        """
        Combine multiple sources with learned precision weighting.

        Args:
            sources: (batch, n_sources, dim) - multiple input sources
            context: (batch, dim) - optional context for precision prediction
        """
        batch_size, n_sources, dim = sources.shape

        if context is None:
            context = sources.mean(dim=1)

        # Predict log-precision for each source
        log_precision = self.precision_network(context)

        # Add persistent state (attention inertia)
        log_precision = log_precision + self.precision_state

        # Update state
        with torch.no_grad():
            self.precision_state = (
                self.state_momentum * self.precision_state +
                (1 - self.state_momentum) * log_precision.mean(dim=0)
            )

        # Convert to weights
        precision_weights = F.softmax(log_precision, dim=-1)

        # Precision-weighted combination
        output = torch.einsum('bn,bnd->bd', precision_weights, sources)

        return output, precision_weights
```

---

## 6. ARR-COC-0-1: Dynamic Precision in Token Allocation

### 6.1 Connection to Adaptive Token Routing

The precision learning framework directly applies to ARR-COC's challenge of allocating computational tokens to image regions:

```python
class PrecisionBasedTokenAllocator(nn.Module):
    """
    Allocate tokens to image regions based on estimated precision.

    Key insight: Regions with HIGH PRECISION need FEWER tokens
    (we're already confident about them). Regions with LOW PRECISION
    need MORE tokens (high uncertainty = needs more computation).

    This is the OPPOSITE of standard attention, which allocates
    more to "important" things. Here we allocate more to
    "uncertain" things that need resolution.
    """

    def __init__(self, dim, n_regions, total_tokens):
        super().__init__()
        self.total_tokens = total_tokens

        # Precision estimator per region
        self.precision_estimator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # Token value predictor
        self.token_value = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, region_features):
        """
        Allocate tokens inversely proportional to precision.

        High precision (confident) -> few tokens
        Low precision (uncertain) -> many tokens
        """
        batch_size, n_regions, dim = region_features.shape

        # Estimate precision per region
        log_precision = self.precision_estimator(region_features).squeeze(-1)
        precision = torch.exp(torch.clamp(log_precision, -5, 5))

        # Inverse precision = uncertainty = need for computation
        uncertainty = 1.0 / (precision + 1e-6)

        # Allocate tokens proportionally to uncertainty
        # Normalize to sum to total_tokens
        allocation_weights = uncertainty / uncertainty.sum(dim=-1, keepdim=True)
        token_allocation = allocation_weights * self.total_tokens

        # Round to integers (differentiable approximation)
        # Use straight-through estimator
        token_allocation_int = token_allocation.round()
        token_allocation = token_allocation + (
            token_allocation_int - token_allocation
        ).detach()

        return {
            'precision': precision,
            'uncertainty': uncertainty,
            'token_allocation': token_allocation,
            'allocation_weights': allocation_weights
        }
```

### 6.2 Adaptive Precision During Inference

```python
class AdaptivePrecisionVLM(nn.Module):
    """
    VLM component with adaptive precision estimation.

    During inference:
    1. Initial quick pass with few tokens -> estimate precision
    2. Allocate more tokens to low-precision (uncertain) regions
    3. Refine predictions in uncertain regions
    4. Repeat until precision threshold met or budget exhausted
    """

    def __init__(self, vision_encoder, n_regions, min_precision=0.8):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.n_regions = n_regions
        self.min_precision = min_precision

        # Precision estimator
        self.precision_head = nn.Linear(vision_encoder.dim, 1)

    def forward(self, image, token_budget, max_iterations=3):
        """
        Iteratively refine low-precision regions.
        """
        # Split image into regions
        regions = self.split_into_regions(image, self.n_regions)

        # Initial encoding with uniform tokens
        initial_tokens = token_budget // self.n_regions
        features = []
        precisions = []

        for region in regions:
            feat = self.vision_encoder(region, n_tokens=initial_tokens)
            prec = torch.sigmoid(self.precision_head(feat.mean(dim=1)))
            features.append(feat)
            precisions.append(prec)

        features = torch.stack(features, dim=1)
        precisions = torch.stack(precisions, dim=1).squeeze(-1)

        remaining_budget = token_budget - initial_tokens * self.n_regions

        # Iteratively refine low-precision regions
        for iteration in range(max_iterations):
            if remaining_budget <= 0:
                break

            # Find regions below precision threshold
            low_precision_mask = precisions < self.min_precision
            if not low_precision_mask.any():
                break

            # Allocate remaining budget to low-precision regions
            n_low = low_precision_mask.sum().item()
            extra_tokens = remaining_budget // max(n_low, 1)

            # Re-encode low-precision regions with more tokens
            for i in range(self.n_regions):
                if low_precision_mask[0, i]:
                    feat = self.vision_encoder(
                        regions[i],
                        n_tokens=initial_tokens + extra_tokens
                    )
                    prec = torch.sigmoid(self.precision_head(feat.mean(dim=1)))
                    features[:, i] = feat
                    precisions[:, i] = prec.squeeze()

            remaining_budget -= extra_tokens * n_low

        return {
            'features': features,
            'precisions': precisions,
            'tokens_used': token_budget - remaining_budget
        }
```

### 6.3 Training with Precision-Aware Loss

```python
class PrecisionAwareRelevanceLoss(nn.Module):
    """
    Loss function that incorporates precision estimation.

    The network learns to:
    1. Predict relevance scores
    2. Estimate precision (confidence) of those predictions
    3. Allocate computation based on precision

    Loss penalizes:
    - Wrong predictions (scaled by precision - confident errors cost more)
    - Miscalibrated precision (overconfident or underconfident)
    """

    def __init__(self, calibration_weight=0.1):
        super().__init__()
        self.calibration_weight = calibration_weight

    def forward(self, predictions, precisions, targets):
        """
        Compute precision-aware loss.

        Args:
            predictions: (batch, n_regions) predicted relevance
            precisions: (batch, n_regions) estimated precision
            targets: (batch, n_regions) ground truth relevance
        """
        # Prediction error
        error = (predictions - targets) ** 2

        # Precision-weighted error
        # High precision = errors cost more (you were confident!)
        weighted_error = precisions * error

        # Calibration loss
        # Precision should match actual accuracy
        # If precision is high but error is high -> bad calibration
        actual_accuracy = 1.0 / (1.0 + error)  # Proxy for accuracy
        calibration_error = (precisions - actual_accuracy) ** 2

        # Total loss
        loss = weighted_error.mean() + self.calibration_weight * calibration_error.mean()

        return {
            'total_loss': loss,
            'weighted_error': weighted_error.mean(),
            'calibration_error': calibration_error.mean()
        }
```

---

## 7. Performance Considerations

### 7.1 Computational Overhead

```python
# Precision learning adds ~20-50% compute overhead:
# - Extra forward pass for precision prediction
# - More complex loss computation
# - Additional memory for precision values

# Optimization strategies:
# 1. Amortize precision prediction (share across layers)
# 2. Use low-rank precision approximations
# 3. Quantize precision values (e.g., 3-5 discrete levels)

class EfficientPrecisionEstimator(nn.Module):
    """
    Efficient precision estimation with minimal overhead.
    """

    def __init__(self, dim, n_precision_levels=5):
        super().__init__()

        # Discrete precision levels (more efficient than continuous)
        self.precision_levels = nn.Parameter(
            torch.linspace(0.1, 2.0, n_precision_levels)
        )

        # Light predictor
        self.predictor = nn.Linear(dim, n_precision_levels)

    def forward(self, x):
        # Predict which precision level
        logits = self.predictor(x.mean(dim=-2))  # Pool first
        probs = F.softmax(logits, dim=-1)

        # Soft selection of precision level
        precision = (probs * self.precision_levels).sum(dim=-1)

        return precision
```

### 7.2 Memory Efficiency

```python
class MemoryEfficientPrecisionNetwork(nn.Module):
    """
    Memory-efficient implementation using gradient checkpointing.
    """

    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([
            PrecisionWeightedPredictiveCodingLayer(dim, dim, dim)
            for _ in range(6)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Checkpoint to save memory
            x = torch.utils.checkpoint.checkpoint(
                layer,
                x,
                use_reentrant=False
            )
        return x
```

### 7.3 GPU Optimization

```python
# Key optimizations for GPU:
# 1. Fuse precision computation with attention
# 2. Use mixed precision (FP16 for features, FP32 for precision)
# 3. Batch precision predictions

@torch.cuda.amp.autocast()
def optimized_precision_attention(q, k, v, precision_net):
    """
    Fused precision-weighted attention for GPU efficiency.
    """
    # Compute attention in FP16
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Compute precision in FP32 for stability
    with torch.cuda.amp.autocast(enabled=False):
        precision = precision_net(scores.float())

    # Apply precision weighting
    weighted_scores = scores * precision.half()
    attn_weights = F.softmax(weighted_scores, dim=-1)

    return torch.matmul(attn_weights, v)
```

---

## 8. Sources

### Web Research (accessed 2025-11-23)

- [Feldman & Friston, 2010: Attention, Uncertainty, and Free-Energy](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00215/full) - Frontiers in Human Neuroscience
  - Foundational paper on attention as precision optimization
  - Cited by 1727+

- [Mazzaglia et al., 2022: The Free Energy Principle for Perception and Action](https://arxiv.org/abs/2207.06415) - arXiv:2207.06415
  - Deep learning perspective on active inference
  - Cited by 96

- [Sluijterman et al., 2024: Optimal training of Mean Variance Estimation networks](https://www.sciencedirect.com/science/article/pii/S0925231224007008) - Neurocomputing
  - Practical guidance on heteroscedastic network training
  - Cited by 30

- [Deka et al., 2024: Analytically Tractable Heteroscedastic Uncertainty](https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_TAGIV_2024_preprint.pdf) - TAGIV preprint
  - AGVI method for analytical variance inference
  - Cited by 8

- [Immer et al., 2023: Effective Bayesian Heteroscedastic Regression](https://openreview.net/forum?id=A6EquH0enk) - NeurIPS 2023
  - Laplace approximation for heteroscedastic networks
  - Cited by 24

### Additional References

- [Parr & Friston, 2017: Working Memory, Attention, and Salience](https://www.nature.com/articles/s41598-017-15249-0) - Scientific Reports
  - Cited by 295

- [Brown et al., 2011: Active Inference, Attention, and Motor Preparation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3177296/) - Frontiers in Psychology
  - Cited by 278

- [Mirza et al., 2019: Bayesian Model of Selective Attention](https://www.nature.com/articles/s41598-019-50138-8) - Scientific Reports
  - Cited by 107

---

## Summary

Precision learning networks unify three fundamental computational mechanisms:

1. **Attention** (deep learning): Learn what to focus on
2. **Precision weighting** (Bayesian inference): Weight by inverse variance
3. **Gain control** (neuroscience): Modulate synaptic responsiveness

The key insights:
- Precision is LEARNABLE, not fixed
- Precision depends on STATE, not just input
- High precision = trust this channel = attend to it
- Learning precision = learning attention = learning what's reliable

For ARR-COC, this means:
- Estimate precision (confidence) of relevance predictions
- Allocate MORE tokens to LOW precision (uncertain) regions
- Train with precision-aware loss for calibrated uncertainty
- Use precision as the principled basis for adaptive computation

The TRAIN STATION where all roads meet: **Attention mechanism = precision = gain control** - the same multiplicative weighting that modulates information flow based on estimated reliability.
