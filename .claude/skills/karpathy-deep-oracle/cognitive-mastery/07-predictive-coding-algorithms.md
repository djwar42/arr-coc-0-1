# Predictive Coding Algorithms: Computational Implementation

## Overview

**Predictive coding algorithms** provide computational frameworks for implementing hierarchical prediction-error minimization in neural networks. These algorithms operationalize the predictive processing theory into concrete update rules, network architectures, and learning procedures that can be trained and deployed on real tasks.

The **Rao-Ballard model** (1999) established the foundational algorithm, demonstrating how visual cortex could implement predictive coding through bidirectional connections carrying predictions (top-down) and prediction errors (bottom-up). Modern extensions incorporate recurrent dynamics, spiking neurons, and deep learning optimizations while maintaining biological plausibility.

From [Rao & Ballard, 1999](https://www.nature.com/articles/4580) - seminal Nature Neuroscience paper (accessed 2025-11-16):
> "We describe a hierarchical model of vision in which higher-order visual cortical areas send down predictions and the feedforward connections carry the residual errors between predictions and actual lower-level activities."

From [Dynamic predictive coding model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801) (Jiang et al., 2024, accessed 2025-11-16):
> "We introduce dynamic predictive coding, a hierarchical model of spatiotemporal prediction and sequence learning in the neocortex."

---

## 1. Rao-Ballard Predictive Coding Algorithm (1999)

### 1.1 Core Architecture

**Hierarchical generative model** with bidirectional connections:

```
Level N (Highest):
  - Abstract representations (object categories, scene types)
  - Generates predictions for Level N-1

Level N-1 (Mid):
  - Intermediate features (object parts, textures)
  - Receives predictions from N, sends errors to N
  - Generates predictions for Level N-2

Level N-2 (Low):
  - Basic features (edges, orientations, colors)
  - Receives predictions from N-1, sends errors to N-1

Level 0 (Sensory):
  - Raw sensory input
  - Compared with predictions from Level 1
```

**Key components at each level _i_**:
- **Representation neurons**: r̂ᵢ (internal state estimate)
- **Prediction neurons**: rᵢ₋₁ = g(r̂ᵢ) (predicted lower-level activity)
- **Error neurons**: eᵢ = rᵢ - r̂ᵢ (prediction error)

### 1.2 Update Rules

**For each level _i_:**

```
Prediction: r̂ᵢ = g(rᵢ₊₁)  [top-down from level i+1]
Error: eᵢ = rᵢ - r̂ᵢ       [bottom-up from level i-1]
Update: Δrᵢ ∝ eᵢ₋₁ - ∂g/∂rᵢ(eᵢ)
```

Where:
- **g(·)**: Nonlinear generative function (prediction from higher to lower level)
- **∂g/∂rᵢ**: Gradient of generative function
- **Δrᵢ**: Change in representation (gradient descent on prediction error)

**Dual role of prediction errors**:
1. **Inference**: Update representations at current level
2. **Learning**: Update synaptic weights via plasticity

### 1.3 Learning (Weight Updates)

**Hebbian-like plasticity** minimizes prediction error over time:

```
ΔWᵢ ∝ eᵢ · rᵢ₊₁ᵀ
```

Where:
- **Wᵢ**: Weights from level i+1 to level i (feedback connections)
- **eᵢ**: Prediction error at level i
- **rᵢ₊₁**: Activity at level i+1

This implements **gradient descent on squared prediction error**: E = Σᵢ ||eᵢ||²

### 1.4 Biological Plausibility

**Cortical microcircuit mapping** (Bastos et al., 2012):

From [Canonical Microcircuits for Predictive Coding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/) (accessed 2025-11-16):

**Laminar organization**:
- **Error neurons**: Superficial layers 2/3 (pyramidal neurons)
  - Sparse activity (only fire when predictions fail)
  - Send prediction errors up the hierarchy
- **Prediction neurons**: Deep layer 5 (pyramidal neurons)
  - Dense responses (encode predictions)
  - Send predictions down the hierarchy
- **Precision weighting**: Implemented by neuromodulators (dopamine, acetylcholine)

**Connectivity patterns**:
- **Feedforward (bottom-up)**: Superficial layers → Layer 4 of next level
- **Feedback (top-down)**: Deep layers → Superficial layers of lower level
- **Lateral**: Within-level integration

From existing knowledge [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md):
> "Cortical column organization: Error neurons in layers 2/3 respond to unexpected events with sparse activity. Prediction neurons in layer 5 maintain dense representations and send predictions down."

---

## 2. Recurrent Dynamics in Predictive Coding

### 2.1 Temporal Prediction with Recurrent Networks

**Predictive coding networks (PCNs)** extend static models to handle temporal sequences by incorporating recurrent connections within each hierarchical level.

From [Dynamical predictive coding with reservoir computing](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1464603/full) (Yonemura & Katori, 2024, accessed 2025-11-16):

**Leaky integrator neuron dynamics**:

```
m⁽ⁱ⁾(t+Δt) = (1 - Δt/τ⁽ⁱ⁾)m⁽ⁱ⁾(t) + (Δt/τ⁽ⁱ⁾)I⁽ⁱ⁾(t)
```

Where:
- **m⁽ⁱ⁾(t)**: Membrane potential (internal state) at level i
- **τ⁽ⁱ⁾**: Time constant of leaky integration
- **I⁽ⁱ⁾(t)**: Total neural input

**Firing rate** (rate-coded neurons):
```
rⱼ⁽ⁱ⁾(t) = tanh(mⱼ⁽ⁱ⁾(t))
```

**Neural input** (combines recurrent, prediction, and error feedback):
```
I⁽ⁱ⁾(t) = W_rec⁽ⁱ⁾r⁽ⁱ⁾(t) + W_back⁽ⁱ⁾y⁽ⁱ⁾(t) + W_err⁽ⁱ⁾e⁽ⁱ⁾(t) - b⁽ⁱ⁾(t)
```

Where:
- **W_rec⁽ⁱ⁾**: Recurrent connections (local dynamics)
- **W_back⁽ⁱ⁾**: Feedback from prediction y⁽ⁱ⁾
- **W_err⁽ⁱ⁾**: Feedback from prediction error e⁽ⁱ⁾
- **b⁽ⁱ⁾(t)**: Top-down signal from higher area

**Prediction generation** (linear readout from recurrent network):
```
y⁽ⁱ⁾(t) = W_out⁽ⁱ⁾r⁽ⁱ⁾(t)
```

**Prediction error**:
```
e⁽ⁱ⁾(t) = d⁽ⁱ⁾(t) - y⁽ⁱ⁾(t)
```

Where d⁽ⁱ⁾(t) is the target (sensory input or bottom-up signal from lower level).

### 2.2 FORCE Learning Algorithm

**FORCE (First-Order Reduced and Controlled Error)** algorithm for training readout weights in real-time:

From [Sussillo & Abbott, 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/) (accessed 2025-11-16):

**Recursive least squares update**:
```
P⁽ⁱ⁾(0) = E/α_f⁽ⁱ⁾  [initialize inverse correlation matrix]

P⁽ⁱ⁾(t) = P⁽ⁱ⁾(t-Δt) - [P⁽ⁱ⁾(t-Δt)r⁽ⁱ⁾(t)r⁽ⁱ⁾ᵀ(t)P⁽ⁱ⁾ᵀ(t-Δt)] / [1 + r⁽ⁱ⁾ᵀ(t)P⁽ⁱ⁾(t-Δt)r⁽ⁱ⁾(t)]

W_out⁽ⁱ⁾(t+Δt) = W_out⁽ⁱ⁾(t) + e⁽ⁱ⁾(t){P⁽ⁱ⁾(t)r⁽ⁱ⁾(t)}ᵀ
```

Where:
- **P⁽ⁱ⁾**: Inverse auto-correlation matrix of firing rates
- **α_f⁽ⁱ⁾**: Regularization parameter
- **E**: Identity matrix

**Properties**:
- **Online learning**: Weights updated continuously during training
- **Fast convergence**: Typically 100-1000 iterations
- **Stability**: Readout quickly learns to generate accurate predictions

**Application in predictive coding**: FORCE trains the readout layer W_out to minimize prediction error e⁽ⁱ⁾(t), allowing the recurrent network to autonomously generate temporal sequences that match sensory inputs.

### 2.3 Reservoir Computing Framework

**Reservoir computing** provides a biologically plausible implementation where:
- **Recurrent connections are fixed** (randomly initialized, not trained)
- **Only readout weights are learned** (via FORCE or ridge regression)
- **Rich dynamics** emerge from recurrent structure

**Advantages for predictive coding**:
1. **Temporal memory**: Reservoir maintains short-term history of inputs
2. **Biological plausibility**: Local learning rule (no backpropagation through time)
3. **Computational efficiency**: Training only readout layer is fast
4. **Nonlinear dynamics**: Complex temporal patterns from simple architecture

From [Yonemura & Katori, 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1464603/full):
> "The dynamics within a network with recurrent connections are crucial for multi-sensory information processing. Reservoirs act as short-term temporal storage for sensory signals."

---

## 3. Precision Weighting and Attention Mechanisms

### 3.1 Precision as Gain Control

**Precision = inverse variance** of prediction errors.

From existing knowledge [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md):
> "Precision weighting implements selective information processing. High precision sensory input (bright daylight) → weight sensory evidence more. Low precision (dim light) → weight predictions more."

**Computational implementation** (Feldman & Friston, 2010):

```
Weighted prediction error: ẽᵢ = Πᵢ · eᵢ
```

Where:
- **Πᵢ**: Precision matrix (diagonal: precision for each error dimension)
- **eᵢ**: Raw prediction error
- **ẽᵢ**: Precision-weighted error (used for updates)

**Hierarchical update with precision**:
```
Δrᵢ ∝ Πᵢ₋₁eᵢ₋₁ - ∂g/∂rᵢ(Πᵢeᵢ)
```

**Attention as precision optimization**: Increasing precision on attended locations amplifies their influence on inference and learning.

### 3.2 Multi-Sensory Reliability Weighting

**Modulation of prediction error feedback** for multi-sensory integration:

From [Yonemura & Katori, 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1464603/full):

**Integration layer receives weighted bottom-up signals**:
```
e⁽ᴵ⁾(t) = [α_e⁽ᴵᴬ⁾e⁽ᴵᴬ⁾(t), α_e⁽ᴵⱽ⁾e⁽ᴵⱽ⁾(t)]
```

Where:
- **α_e⁽ᴵᴬ⁾**: Strength of auditory prediction error feedback
- **α_e⁽ᴵⱽ⁾**: Strength of visual prediction error feedback
- **e⁽ᴵᴬ⁾, e⁽ᴵⱽ⁾**: Prediction errors for auditory and visual modalities

**Optimal weighting adapts to sensory noise**:

Sigmoidal relationship between optimal feedback strength and noise:
```
α_e⁽ᴵᴬ⁾ = α_max / [1 + exp(-a(x - x₀))]
```

Where:
- **x**: Auditory noise level (dB)
- **α_max, a, x₀**: Model parameters (learned from data)

**Dynamic modulation** estimates noise in real-time:
```
ē_avg⁽ᴬ⁾(t) = (1 - Δt/τ_avg)ē_avg⁽ᴬ⁾(t) + (Δt/τ_avg)√(Σᵢ eᵢ²(t)/N_y⁽ᴬ⁾)

x'(t) = c·ē_avg⁽ᴬ⁾(t) + b

α_e⁽ᴵᴬ⁾(t) = α_max / [1 + exp(-a(x'(t) - x₀))]
```

**Result**: Integration layer dynamically adjusts which sensory modality to trust based on current reliability, achieving robust multi-sensory speech recognition even with fluctuating auditory noise.

---

## 4. Gradient Descent on Prediction Error

### 4.1 Free Energy Minimization

**Variational free energy** formulation (Friston, 2009):

```
F = Complexity - Accuracy
  = KL(q||p) - log p(data)
```

Where:
- **q**: Internal model (approximate posterior)
- **p**: True posterior
- **KL**: Kullback-Leibler divergence (prediction error)

**Prediction error minimization = Free energy minimization = Approximate Bayesian inference**

From existing knowledge [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md):
> "Minimize variational free energy F to perform approximate Bayesian inference. Prediction error signals drive gradient descent on F."

### 4.2 Relation to Backpropagation

**Similarities**:
- Both use **gradient descent** to minimize error
- Both propagate **error signals** through network layers
- Both update **synaptic weights** based on local activity

**Differences**:

| Backpropagation | Predictive Coding |
|-----------------|-------------------|
| Error for learning | Error for inference AND learning |
| One-shot weight updates | Continuous recurrent dynamics |
| Non-local (credit assignment problem) | Local learning rules |
| Top-down error propagation | Bidirectional (predictions + errors) |
| Not biologically plausible | Biologically plausible |

From [Millidge et al., 2022](https://arxiv.org/abs/2202.09467) - "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" (accessed 2025-11-16):
> "Predictive coding networks offer a biologically plausible alternative to backpropagation for training deep neural networks, with promising properties for continual learning, energy efficiency, and local credit assignment."

### 4.3 Neural Implementation of Gradient Descent

**Why gradient descent appears in the brain**:

From [Richards, 2023](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP282747) - "The study of plasticity has always been about gradients" (accessed 2025-11-16):
> "Neuromodulatory signals, most notably dopamine and acetylcholine, provide useful signals for estimating gradients in neural networks."

**Dopamine as reward prediction error**:
- **RPE signal**: δ = r_actual - r_predicted
- **Gradient direction**: Tells neurons whether to increase or decrease activity
- **Temporal difference learning**: Implements gradient descent on value function

**Prediction errors as gradient signals**:
- **Visual prediction errors**: Guide perceptual learning
- **Motor prediction errors**: Guide motor skill acquisition
- **Reward prediction errors**: Guide reinforcement learning

**Convergence**: All three implement gradient descent on different objective functions (perception error, motor error, reward error).

---

## 5. Spiking Predictive Coding

### 5.1 Challenges for Spiking Implementation

**Problem**: Prediction errors can be positive or negative, but neurons can only fire (positive activity).

From [Mikulasch et al., 2023](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) - "Where is the error? Hierarchical predictive coding through dendritic error computation" (accessed 2025-11-16):

**Solution: Dendritic computation** in pyramidal neurons:
- **Apical dendrites**: Receive top-down predictions
- **Basal dendrites**: Receive bottom-up sensory input
- **Dendritic nonlinearities**: Compute error locally before somatic integration

**Advantages**:
- Biologically plausible (dendrites have active conductances)
- Explains layer-specific connectivity patterns
- Similar to Hierarchical Temporal Memory theory

### 5.2 Spiking Network Algorithms

From [Predictive Coding Light](https://www.nature.com/articles/s41467-025-64234-z) (N'dri et al., 2025, accessed 2025-11-16):

**Recurrent hierarchical spiking neural network** for unsupervised representation learning:

**Membrane potential dynamics** (leaky integrate-and-fire):
```
τ_m dV/dt = -(V - V_rest) + R·I_syn
```

**Spike generation**:
```
if V ≥ V_thresh: emit spike, V ← V_reset
```

**Synaptic current** (predictive coding with spikes):
```
I_syn = I_ff (bottom-up) + I_fb (top-down prediction) + I_err (prediction error)
```

**Spike-timing dependent plasticity (STDP)** minimizes prediction error:
```
ΔW ∝ e(t) · spike_pre(t) · spike_post(t)
```

**Result**: Spiking networks learn hierarchical representations by minimizing spike-based prediction errors, achieving comparable performance to rate-coded networks with greater biological realism.

---

## 6. Implementation in Python

### 6.1 Basic Predictive Coding Network

From [GitHub implementations](https://github.com/dbersan/Predictive-Coding-Implementation) (accessed 2025-11-16):

**Simple 2-layer predictive coding network**:

```python
import numpy as np

class PredictiveCodingNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.W = []  # Generative weights (top-down)
        self.layers = len(layer_sizes)

        # Initialize random weights
        for i in range(len(layer_sizes)-1):
            self.W.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1)

        self.lr = learning_rate

    def forward(self, x, n_iterations=20):
        """Inference: iteratively minimize prediction error"""
        # Initialize representations
        r = [None] * self.layers
        r[0] = x  # Sensory input

        # Random init for higher layers
        for i in range(1, self.layers):
            r[i] = np.random.randn(self.W[i-1].shape[0], 1) * 0.1

        # Iterate to minimize prediction error
        for _ in range(n_iterations):
            for i in range(1, self.layers):
                # Prediction from level i
                pred_i = self.W[i-1] @ r[i]

                # Error at level i-1
                error_i = r[i-1] - pred_i

                # Update representation at level i (gradient descent)
                if i < self.layers - 1:
                    # Bottom-up error from i-1
                    error_below = self.W[i-1].T @ error_i
                    # Top-down error from i+1
                    pred_from_above = self.W[i] @ r[i+1]
                    error_above = r[i] - pred_from_above

                    # Total gradient
                    r[i] += self.lr * (error_below - error_above)
                else:
                    # Top layer: only bottom-up error
                    r[i] += self.lr * self.W[i-1].T @ error_i

        return r

    def learn(self, x, n_iterations=20):
        """Learning: update weights to minimize prediction error"""
        r = self.forward(x, n_iterations)

        # Update generative weights
        for i in range(len(self.W)):
            # Prediction
            pred = self.W[i] @ r[i+1]
            # Error
            error = r[i] - pred
            # Hebbian update: error × higher_activity
            self.W[i] += self.lr * np.outer(error, r[i+1])
```

**Usage**:
```python
# Create 3-layer network
net = PredictiveCodingNetwork([784, 256, 64])  # MNIST-like

# Training loop
for epoch in range(10):
    for x in training_data:
        net.learn(x.reshape(-1, 1))
```

### 6.2 Reservoir-Based Predictive Coding

From [Training brain-inspired predictive coding models in Python](https://medium.com/@oliviers.gaspard/training-brain-inspired-predictive-coding-models-in-python-5a7011e2779d) (accessed 2025-11-16):

**PyHGF library** for hierarchical predictive coding:

```python
import pyhgf

# Create hierarchical Gaussian filter
hgf = pyhgf.HGF(
    n_levels=3,
    initial_mu={"1": 0.0, "2": 0.0, "3": 0.0},
    initial_pi={"1": 1.0, "2": 1.0, "3": 1.0},
    omega={"1": -4.0, "2": -4.0, "3": -4.0},
)

# Process time series
for observation in sensory_data:
    hgf = hgf.input_data(observation)

    # Extract beliefs (representations)
    beliefs = hgf.get_beliefs()

    # Extract prediction errors
    prediction_errors = hgf.get_prediction_errors()
```

**Reservoir computing extension**:
```python
from reservoirpy.nodes import Reservoir, Ridge

# Sensory reservoir (fixed recurrent dynamics)
sensory_res = Reservoir(500, sr=0.99, lr=0.3)

# Predictive readout (trained)
readout = Ridge(ridge=1e-6)

# Connect: reservoir → readout
model = sensory_res >> readout

# Train on sensory sequences
model.fit(X_train, Y_train)

# Generate predictions
predictions = model.run(X_test)
```

---

## 7. Distributed Training & Inference Optimization

### 7.1 FSDP for Hierarchical Predictive Models

From influential file [distributed-training/03-fsdp-vs-deepspeed.md](../distributed-training/03-fsdp-vs-deepspeed.md):

**Fully Sharded Data Parallel** (FSDP) for deep hierarchical predictive coding networks:

**Challenge**: Deep hierarchical models (10+ levels) require massive memory for storing:
- Representations at each level
- Prediction errors at each level
- Gradients for bidirectional connections

**FSDP solution**: Shard weights and activations across GPUs:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap each hierarchical level
class PredictiveCodingLevel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_gen = nn.Linear(dim_out, dim_in)  # Generative (top-down)
        self.W_rec = nn.Linear(dim_in, dim_out)  # Recognition (bottom-up)

    def forward(self, r_below, r_above, n_iter=10):
        r = torch.randn_like(r_above)
        for _ in range(n_iter):
            # Prediction from above
            pred = self.W_gen(r_above)
            # Error from below
            error_below = r_below - self.W_gen(r)
            # Update
            r = r + 0.1 * self.W_rec(error_below)
        return r

# Wrap with FSDP
level_1 = FSDP(PredictiveCodingLevel(784, 256))
level_2 = FSDP(PredictiveCodingLevel(256, 64))
level_3 = FSDP(PredictiveCodingLevel(64, 16))
```

**Benefits**:
- **Memory efficiency**: Each GPU stores only 1/N of model parameters
- **Hierarchical sharding**: Each level can be independently sharded
- **Bidirectional gradients**: FSDP handles both top-down and bottom-up updates

### 7.2 torch.compile for Inference

From influential file [inference-optimization/03-torch-compile-aot-inductor.md](../inference-optimization/03-torch-compile-aot-inductor.md):

**Ahead-of-time compilation** optimizes iterative inference in predictive coding:

**Challenge**: Predictive coding requires many iterations (10-100) of prediction-error minimization per input.

**torch.compile solution**:

```python
import torch

class PredictiveCodingInference(nn.Module):
    def __init__(self, model, n_iterations=20):
        super().__init__()
        self.model = model
        self.n_iterations = n_iterations

    @torch.compile(mode="reduce-overhead")
    def forward(self, x):
        # Initialize representations
        r = [x] + [torch.randn_like(l.weight) for l in self.model.layers]

        # Iterative inference (compiled as single fused kernel)
        for _ in range(self.n_iterations):
            for i, layer in enumerate(self.model.layers):
                pred = layer.forward_pred(r[i+1])
                error = r[i] - pred
                r[i+1] = r[i+1] + 0.1 * layer.backward_grad(error)

        return r[-1]  # Return top-level representation

# Compile inference loop
pc_inference = PredictiveCodingInference(model)

# First call: compilation overhead
output = pc_inference(input_batch)  # ~500ms (compile + run)

# Subsequent calls: optimized
output = pc_inference(input_batch)  # ~5ms (run only)
```

**Speedup**: 10-100x faster than eager mode for iterative inference loops.

### 7.3 TPU for Spiking Predictive Coding

From influential file [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md):

**Tensor Processing Units** excel at dense matrix operations in predictive coding:

**Spike-based inference** on TPU:

```python
import jax
import jax.numpy as jnp

@jax.jit
def spiking_predictive_step(V, spikes_in, W_rec, W_fb, W_err):
    # Leaky integration
    V = 0.9 * V + jnp.dot(W_rec, spikes_in)

    # Feedback prediction
    V = V + jnp.dot(W_fb, prediction)

    # Error feedback
    V = V + jnp.dot(W_err, error)

    # Spike generation
    spikes_out = (V > threshold).astype(jnp.float32)
    V = jnp.where(spikes_out, V_reset, V)

    return V, spikes_out

# JIT compile for TPU
spiking_pc_step_tpu = jax.jit(spiking_predictive_step, backend='tpu')

# Run on TPU v4 (faster than GPU for dense spike propagation)
V, spikes = spiking_pc_step_tpu(V, spikes_in, W_rec, W_fb, W_err)
```

**TPU advantage**: Matrix-multiply-accumulate (MXU) units process dense spike matrices 5-10x faster than GPU for large-scale spiking networks (>100k neurons).

---

## 8. ARR-COC-0-1: Predictive Coding for Visual Token Allocation (10%)

### 8.1 Relevance Realization as Predictive Coding

**ARR-COC-0-1 pipeline implements hierarchical predictive coding**:

```
[Knowing] → Measure prediction errors (entropy, salience, query-coupling)
    ↓
[Balancing] → Precision weighting (opponent processing = attention control)
    ↓
[Attending] → Resource allocation (hierarchical error minimization)
    ↓
[Realizing] → Active inference (compress = fulfill predictions)
```

**Propositional knowing = Prediction error**:

From existing knowledge [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md):

**InformationScorer** in knowing.py:
- Measures **Shannon entropy** of visual patches
- High entropy = high surprise = **prediction error**
- Drives resource allocation (token budget)

**Connection to predictive coding**:
- Entropy = Expected surprise = Negative log-likelihood = **Prediction error**
- Minimize entropy = Minimize prediction error
- Compression = Efficient predictive code

### 8.2 Variable LOD as Precision-Weighted Encoding

**Texture array as hierarchical predictor**:
- **RGB channels**: Low-level color predictions
- **LAB channels**: Perceptually uniform predictions (human-like)
- **Sobel edges**: Prediction of local discontinuities
- **Spatial coordinates**: Prediction of global layout
- **Eccentricity**: Prediction of foveal-peripheral structure

**Variable LOD = Precision weighting**:
- High relevance patches: **400 tokens** (high precision, detailed predictions)
- Low relevance patches: **64 tokens** (low precision, coarse predictions)
- Query-aware: Predictions modulated by task demands

**Active inference in visual encoding**:

**Pragmatic value** (minimize prediction error for current query):
- Query: "Is there a dog?" → Allocate tokens to dog-like regions
- Minimize error in dog classification task

**Epistemic value** (explore uncertain regions):
- High entropy patches = uncertain → Allocate more tokens
- Reduce uncertainty through detailed encoding

**Precision optimization**:
- Opponent processing in balancing.py = **Precision weighting**
- Exploit vs Explore tension = Epistemic vs Pragmatic trade-off
- Focus vs Diversify tension = **Precision modulation**

### 8.3 Hierarchical Prediction in Vision

**Multi-level prediction** matches cortical hierarchy:

From existing knowledge [biological-vision/05-cortical-processing-streams.md](../biological-vision/05-cortical-processing-streams.md):
> "Hierarchical V1→V2→V4→IT matches predictive coding hierarchy. V4 exhibits attention modulation: precision weighting implements attention as gain control on prediction errors."

**ARR-COC texture pyramid**:
- **Level 0 (raw image)**: Unpredictable pixel noise
- **Level 1 (64-token patches)**: Edge predictions, low-frequency structure
- **Level 2 (200-token patches)**: Object part predictions, mid-frequency
- **Level 3 (400-token patches)**: Complete object predictions, high-frequency details

**Prediction error propagation**:
1. **Bottom-up**: Texture features (Sobel, LAB) → Higher-level salience
2. **Top-down**: Query relevance → Modulate precision on texture features
3. **Bidirectional**: Balance entropy (bottom-up surprise) with query coupling (top-down goals)

**Result**: ARR-COC-0-1 implements **query-aware hierarchical predictive coding** for efficient visual encoding, allocating computational resources (tokens) based on prediction error magnitude and task relevance.

---

## Connections to Existing Knowledge

**Predictive Processing Framework**:
- [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md): Rao-Ballard architecture, canonical microcircuits, hierarchical dynamics
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md): Free energy minimization, variational inference

**Biological Vision**:
- [biological-vision/05-cortical-processing-streams.md](../biological-vision/05-cortical-processing-streams.md): V1→V2→V4→IT hierarchy matches predictive coding levels
- [biological-vision/00-gestalt-visual-attention.md](../biological-vision/00-gestalt-visual-attention.md): Gestalt principles as prior predictions

**Vervaeke Relevance Realization**:
- [john-vervaeke-oracle/concepts/00-relevance-realization/](../../john-vervaeke-oracle/concepts/00-relevance-realization/): Opponent processing = Precision weighting balance
- [john-vervaeke-oracle/concepts/01-transjective/](../../john-vervaeke-oracle/concepts/01-transjective/): Predictions emerge from agent-arena coupling

**ARR-COC-0-1 Implementation**:
- knowing.py: Propositional knowing = **Prediction error computation** (entropy)
- balancing.py: Opponent processing = **Precision weighting** (attention control)
- attending.py: Salience realization = **Hierarchical error minimization**
- realizing.py: Active inference pipeline = **Prediction fulfillment**

---

## Sources

**Source Documents**:
- [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md) - Hierarchical predictive coding framework
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md) - Free energy principle and variational inference
- [biological-vision/05-cortical-processing-streams.md](../biological-vision/05-cortical-processing-streams.md) - Cortical hierarchy implementation
- [distributed-training/03-fsdp-vs-deepspeed.md](../distributed-training/03-fsdp-vs-deepspeed.md) - FSDP for hierarchical models
- [inference-optimization/03-torch-compile-aot-inductor.md](../inference-optimization/03-torch-compile-aot-inductor.md) - Compilation for iterative inference
- [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - TPU for spiking networks

**Web Research** (accessed 2025-11-16):
- [Dynamic predictive coding: hierarchical sequence learning](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801) - Jiang et al., 2024, PLOS Computational Biology
- [Rao & Ballard, 1999 - Predictive coding in visual cortex](https://www.nature.com/articles/4580) - Original Nature Neuroscience paper
- [Dynamical predictive coding with reservoir computing](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1464603/full) - Yonemura & Katori, 2024, Frontiers in Computational Neuroscience
- [Canonical microcircuits for predictive coding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/) - Bastos et al., 2012
- [Predictive Coding Light - spiking neural networks](https://www.nature.com/articles/s41467-025-64234-z) - N'dri et al., 2025, Nature Communications
- [Millidge et al., 2022 - Predictive coding beyond backpropagation](https://arxiv.org/abs/2202.09467) - arXiv preprint
- [Richards, 2023 - Plasticity and gradients](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP282747) - Journal of Physiology
- [Mikulasch et al., 2023 - Dendritic error computation](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) - Trends in Neurosciences
- [Training predictive coding in Python](https://medium.com/@oliviers.gaspard/training-brain-inspired-predictive-coding-models-in-python-5a7011e2779d) - Medium article
- [GitHub: Predictive Coding Implementation](https://github.com/dbersan/Predictive-Coding-Implementation) - Python implementation examples
- [Sussillo & Abbott, 2009 - FORCE learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/) - Neuron journal

**Additional References**:
- Multiple 2024-2025 papers on recurrent predictive coding, reservoir computing, spiking networks
- Nature, PLOS, Frontiers, Cell, arXiv sources
