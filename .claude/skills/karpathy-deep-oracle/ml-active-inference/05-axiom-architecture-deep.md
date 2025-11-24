# AXIOM Architecture Deep Dive: Active Inference Meets Object-Centric Learning

## Overview

AXIOM (Active eXpanding Inference with Object-centric Models) represents a paradigm shift in reinforcement learning architecture. Developed by VERSES AI, it abandons the traditional deep learning approach of overparameterized networks trained via backpropagation in favor of a fully Bayesian, gradient-free architecture built on mixture models.

**Core Innovation**: AXIOM learns to play arcade games in minutes (10,000 steps) without neural networks, gradient-based optimization, or replay buffers - achieving human-like sample efficiency through structured priors and online Bayesian inference.

**Key Insight**: Instead of learning everything from scratch with massive data, AXIOM incorporates "core priors" - abstract organizational principles that constrain the hypothesis space while leaving specific structural details to be discovered online.

---

## Section 1: AXIOM Architecture Details

### 1.1 Architectural Philosophy

AXIOM addresses three fundamental challenges simultaneously:

1. **Sample Efficiency**: Learn from minimal experience like humans
2. **Structured Priors**: Incorporate object-centric inductive biases
3. **Uncertainty Quantification**: Maintain full Bayesian posteriors for principled exploration

**The Core Prior Approach**:
- World consists of discrete, extensive entities (objects)
- Objects follow piecewise linear dynamics
- Interactions between objects are sparse and local
- Rewards are linked directly to interaction events

### 1.2 The Four Mixture Models

AXIOM's architecture consists of four interconnected mixture models:

```
Input Image -> sMM -> Object Slots -> iMM -> Identity Codes
                                        |
                                        v
                        rMM <- Action/Reward -> tMM -> Next State
                         |                       ^
                         +-> Switch States ------+
```

#### Slot Mixture Model (sMM)
**Purpose**: Segment pixels into object-centric representations

**Mechanism**:
- Explains each pixel as belonging to one of K object slots
- Each slot proposes: position (p), color (c), shape/extent (e)
- Competitive assignment: pixel assigned to slot with highest likelihood

```python
# sMM likelihood for pixel n at time t
p(y_n | x^(k), sigma_c^(k), z_k) = N(A*x^(k), diag([B*x^(k), sigma_c^(k)]))

# Where:
# A*x^(k) = [position, color]  (mean)
# B*x^(k) = extent (spatial variance)
# sigma_c = color variance (learned per slot)
```

**Key Features**:
- No neural encoder - direct mixture model on pixel data
- Automatically grows new slots when new objects appear
- Prunes unused slots via truncated stick-breaking prior

#### Identity Mixture Model (iMM)
**Purpose**: Assign discrete identity codes to objects

**Why Identity Matters**:
- Enables type-specific dynamics (ball vs paddle)
- Allows remapping when environment changes (color swap)
- Supports generalization across instances of same type

```python
# iMM clusters color+shape features
p([c^(k), e^(k)]^T | z_type^(k)) = Product_j N(mu_j, Sigma_j)^(z_j_type)

# With Normal-Inverse-Wishart priors on (mu, Sigma)
p(mu_j, Sigma_j^-1) = NIW(m_j, kappa_j, U_j, n_j)
```

**Result**: Each object gets a discrete "type token" used by dynamics model

#### Transition Mixture Model (tMM)
**Purpose**: Model object dynamics as switching linear systems

**Key Insight**: Complex trajectories decompose into simpler linear "motion verbs"

```python
# tMM: Piecewise linear dynamics
p(x_t^(k) | x_{t-1}^(k), s_t^(k)) = Product_l N(D_l * x^(k) + b_l, 2I)^(s_l)

# Where:
# D_l, b_l = linear dynamics parameters for mode l
# s_t^(k) = switch state selecting which mode
```

**Motion Library Examples**:
- "Falling" (gravity): D = [[1,0],[0,1]], b = [0, -g]
- "Bouncing": D = [[1,0],[0,-1]], b = [0, 0]
- "Sliding left": D = [[1,0],[0,1]], b = [-v, 0]

**Important**: Dynamics are SHARED across all objects - learn once, reuse everywhere!

#### Recurrent Mixture Model (rMM)
**Purpose**: Model object interactions and predict switch states

**This is the "brain" of AXIOM** - it captures:
- Object-to-object interactions (collisions)
- Action effects on dynamics
- Reward associations

```python
# rMM models joint distribution of features
f^(k) = (C*x^(k), g(x^(1:K)))  # Continuous: own state + interaction features
d^(k) = (z_type, s_tmm, action, reward)  # Discrete: identity, switch, action, reward

p(f^(k), d^(k) | s_rmm^(k)) = Product_m [N(f; mu_m, Sigma_m) * Product_i Cat(d_i; alpha_m,i)]^(s_m_rmm)
```

**Interaction Features g(x^(1:K))**:
- Distance to nearest object
- X/Y displacement to nearest object
- Identity of nearest object

**Result**: rMM clusters predict which tMM mode (switch state) to use next

### 1.3 Information Flow

Complete forward pass through AXIOM:

```
1. Observe frame y_t
2. sMM: Segment into K object slots {x_t^(k)}
3. iMM: Assign identity codes {z_type^(k)}
4. For each object k:
   a. Compute interaction features with nearest object
   b. rMM: Infer switch state s_t^(k) from (position, identity, distance, action)
   c. tMM: Predict next state x_{t+1}^(k) using dynamics mode s_t^(k)
5. Predict reward from rMM clusters
6. Plan using expected free energy over imagined trajectories
```

---

## Section 2: How AXIOM Differs from Transformers

### 2.1 Fundamental Architectural Differences

| Aspect | Transformers | AXIOM |
|--------|-------------|-------|
| **Learning** | Gradient descent | Variational Bayes |
| **Memory** | Replay buffer | None (online) |
| **Parameters** | Fixed (millions) | Growing (thousands) |
| **Representation** | Distributed activations | Explicit objects |
| **Uncertainty** | Point estimates | Full posteriors |
| **Computation** | Matrix multiplications | Mixture inference |

### 2.2 No Attention Mechanism

**Transformers**: Use attention to route information between positions

```python
# Transformer attention
Q, K, V = linear(x), linear(x), linear(x)
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**AXIOM**: Uses mixture assignment as implicit attention

```python
# AXIOM "attention" via mixture assignment
responsibility = exp(log_likelihood(x, component_k)) / sum_j(exp(log_likelihood(x, component_j)))
# Hard assignment: argmax over responsibilities
```

**Key Difference**:
- Attention is soft-weighted combination
- Mixture is winner-take-all assignment

### 2.3 No Learned Embeddings

**Transformers**: Project inputs through learned embedding matrices

```python
# Transformer embedding
token_embedding = embedding_matrix[token_id]
position_embedding = positional_encoding[position]
x = token_embedding + position_embedding
```

**AXIOM**: Raw features are directly interpretable

```python
# AXIOM uses raw features
x^(k) = [position_x, position_y, color_r, color_g, color_b, extent_x, extent_y]
# No embedding - these ARE the representations
```

### 2.4 No Feedforward Networks

**Transformers**: MLP blocks transform representations

```python
# Transformer FFN
h = gelu(linear1(x))
out = linear2(h)
```

**AXIOM**: Linear transformations only (within tMM)

```python
# AXIOM dynamics (purely linear)
x_next = D @ x + b  # That's it!
```

### 2.5 Structure Learning vs Fixed Architecture

**Transformers**: Fixed number of layers, heads, dimensions

**AXIOM**: Structure grows with data

```python
# AXIOM structure learning
if max_likelihood < threshold:
    add_new_component()  # Grow model

# Periodic reduction
if model_evidence_increases_after_merge:
    merge_components()  # Shrink model
```

### 2.6 Online vs Batch Learning

**Transformers**: Train on batches, require multiple epochs

```python
# Transformer training
for epoch in range(100):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**AXIOM**: Single-pass online learning

```python
# AXIOM learning
for frame in stream:
    # E-step: Infer latent states
    posteriors = infer_states(frame)
    # M-step: Update parameters
    update_parameters(posteriors, frame)
    # Done! No replay, no gradients
```

---

## Section 3: Belief Representation vs Activations

### 3.1 What Transformers Store

Transformers maintain **activations** - point estimates of hidden states:

```python
# Transformer hidden state
h = torch.Tensor([0.3, -0.7, 0.1, ...])  # Just numbers
# No uncertainty information!
```

### 3.2 What AXIOM Stores

AXIOM maintains **belief distributions** - full posteriors over parameters:

```python
# AXIOM belief state (example: NIW posterior)
belief = {
    'mean': m,           # Expected mean
    'precision_scale': kappa,  # Confidence in mean
    'scatter_matrix': U,  # Expected covariance structure
    'degrees_of_freedom': n  # Sample count
}
```

### 3.3 Implications for Uncertainty

**Transformers**: Must use auxiliary methods for uncertainty

```python
# Dropout-based uncertainty (hacky)
predictions = [model(x, dropout=True) for _ in range(100)]
uncertainty = std(predictions)

# Or train separate uncertainty head
mean, log_var = model(x)
```

**AXIOM**: Uncertainty is native

```python
# AXIOM predictive distribution
# For NIW posterior, the predictive is Student-t:
predictive = StudentT(
    df = n - D + 1,
    loc = m,
    scale = U * (kappa + 1) / (kappa * (n - D + 1))
)
# Uncertainty comes FREE from the posterior!
```

### 3.4 Belief Updates

**How beliefs evolve in AXIOM**:

```python
# Conjugate update for Normal-Inverse-Wishart
def update_niw(prior, data_point):
    m_0, kappa_0, U_0, n_0 = prior

    # Update posterior parameters
    kappa_n = kappa_0 + 1
    n_n = n_0 + 1
    m_n = (kappa_0 * m_0 + data_point) / kappa_n

    delta = data_point - m_0
    U_n = U_0 + (kappa_0 / kappa_n) * outer(delta, delta)

    return m_n, kappa_n, U_n, n_n
```

**Key Property**: Updates are exact (no approximation error accumulation)

### 3.5 Belief State for Planning

AXIOM uses beliefs directly in planning:

```python
# Expected free energy computation
def expected_free_energy(policy, beliefs):
    G = 0
    for tau in range(horizon):
        # Utility: Expected log reward
        utility = E_q[log p(r_tau | o_tau, policy)]

        # Information gain: KL divergence
        # How much would we learn about rMM parameters?
        info_gain = KL(q(alpha | o_tau, policy) || q(alpha))

        G += -utility + info_gain
    return G
```

**Critical Insight**: Information gain requires uncertainty - transformers can't do this naturally!

---

## Section 4: Code - AXIOM-Style Layers

### 4.1 Slot Mixture Model Implementation

```python
import numpy as np
from scipy.special import digamma, gammaln

class SlotMixtureModel:
    """Object-centric slot segmentation via Gaussian mixture."""

    def __init__(self, max_slots=10, alpha_0=0.1):
        self.max_slots = max_slots
        self.alpha_0 = alpha_0  # Stick-breaking concentration

        # Slot parameters (position, color, extent)
        self.slots = []  # List of slot posteriors
        self.mixing_weights = np.ones(max_slots) * alpha_0

    def add_slot(self, initial_observation):
        """Add new slot initialized from observation."""
        slot = {
            'position_mean': initial_observation[:2],
            'color_mean': initial_observation[2:5],
            'extent': np.array([0.1, 0.1]),  # Initial size
            'count': 1
        }
        self.slots.append(slot)

    def compute_responsibilities(self, pixels):
        """E-step: Compute soft assignments of pixels to slots."""
        # pixels: (N, 5) - [x, y, r, g, b]
        N = len(pixels)
        K = len(self.slots)

        if K == 0:
            return np.zeros((N, 1))

        log_resp = np.zeros((N, K))

        for k, slot in enumerate(self.slots):
            # Position likelihood
            pos_diff = pixels[:, :2] - slot['position_mean']
            pos_ll = -0.5 * np.sum(pos_diff**2 / slot['extent']**2, axis=1)

            # Color likelihood
            color_diff = pixels[:, 2:5] - slot['color_mean']
            color_ll = -0.5 * np.sum(color_diff**2 / 0.01, axis=1)

            log_resp[:, k] = pos_ll + color_ll + np.log(self.mixing_weights[k])

        # Normalize
        log_resp -= np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)

        return resp

    def update_slots(self, pixels, responsibilities):
        """M-step: Update slot parameters given responsibilities."""
        for k, slot in enumerate(self.slots):
            r_k = responsibilities[:, k]
            N_k = r_k.sum()

            if N_k > 1:
                # Update position
                slot['position_mean'] = (r_k[:, None] * pixels[:, :2]).sum(0) / N_k

                # Update color
                slot['color_mean'] = (r_k[:, None] * pixels[:, 2:5]).sum(0) / N_k

                # Update extent (variance)
                pos_diff = pixels[:, :2] - slot['position_mean']
                slot['extent'] = np.sqrt((r_k[:, None] * pos_diff**2).sum(0) / N_k + 1e-6)

                slot['count'] = N_k

        # Update mixing weights
        counts = np.array([s['count'] for s in self.slots])
        self.mixing_weights[:len(self.slots)] = counts / counts.sum()

    def should_add_slot(self, pixels, responsibilities):
        """Check if new slot needed based on reconstruction error."""
        max_resp = responsibilities.max(axis=1)
        poorly_explained = (max_resp < 0.5).sum() / len(pixels)
        return poorly_explained > 0.1 and len(self.slots) < self.max_slots

    def infer(self, image):
        """Full inference: segment image into slots."""
        # Flatten image to pixels
        H, W, C = image.shape
        pixels = np.zeros((H * W, 5))
        for i in range(H):
            for j in range(W):
                pixels[i * W + j] = [
                    2 * i / H - 1,  # Normalized position
                    2 * j / W - 1,
                    image[i, j, 0] / 255,  # RGB
                    image[i, j, 1] / 255,
                    image[i, j, 2] / 255
                ]

        # Initialize if needed
        if len(self.slots) == 0:
            # K-means++ style initialization
            idx = np.random.randint(len(pixels))
            self.add_slot(pixels[idx])

        # EM iteration
        for _ in range(5):
            resp = self.compute_responsibilities(pixels)

            if self.should_add_slot(pixels, resp):
                # Add slot at poorly explained pixel
                max_resp = resp.max(axis=1)
                worst_idx = np.argmin(max_resp)
                self.add_slot(pixels[worst_idx])
                resp = self.compute_responsibilities(pixels)

            self.update_slots(pixels, resp)

        return [s['position_mean'] for s in self.slots]
```

### 4.2 Transition Mixture Model Implementation

```python
class TransitionMixtureModel:
    """Switching linear dynamical system for object dynamics."""

    def __init__(self, state_dim=7, max_modes=20, alpha_0=0.1):
        self.state_dim = state_dim
        self.max_modes = max_modes
        self.alpha_0 = alpha_0

        # Each mode: (D, b) for x_next = D @ x + b
        self.modes = []
        self.mode_counts = np.zeros(max_modes)

    def add_mode(self, x_prev, x_curr):
        """Add new dynamics mode from single transition."""
        # Initialize D as identity, b as the residual
        D = np.eye(self.state_dim)
        b = x_curr - x_prev

        mode = {
            'D': D,
            'b': b,
            'sum_xx': np.outer(x_prev, x_prev),
            'sum_xy': np.outer(x_prev, x_curr),
            'count': 1
        }
        self.modes.append(mode)

    def predict(self, x, mode_idx):
        """Predict next state using specified mode."""
        mode = self.modes[mode_idx]
        return mode['D'] @ x + mode['b']

    def compute_mode_likelihoods(self, x_prev, x_curr):
        """Compute likelihood of transition under each mode."""
        K = len(self.modes)
        if K == 0:
            return np.array([])

        log_liks = np.zeros(K)
        for k, mode in enumerate(self.modes):
            pred = self.predict(x_prev, k)
            error = x_curr - pred
            # Fixed covariance of 2*I
            log_liks[k] = -0.25 * np.sum(error**2)
            log_liks[k] += np.log(self.mode_counts[k] + self.alpha_0)

        return log_liks

    def infer_mode(self, x_prev, x_curr):
        """Infer which dynamics mode generated transition."""
        log_liks = self.compute_mode_likelihoods(x_prev, x_curr)

        if len(log_liks) == 0:
            self.add_mode(x_prev, x_curr)
            return 0

        # Check if new mode needed
        max_ll = np.max(log_liks)
        threshold = np.log(self.alpha_0) - 0.5 * self.state_dim  # Prior predictive

        if max_ll < threshold and len(self.modes) < self.max_modes:
            self.add_mode(x_prev, x_curr)
            return len(self.modes) - 1

        return np.argmax(log_liks)

    def update_mode(self, mode_idx, x_prev, x_curr):
        """Update mode parameters with new transition."""
        mode = self.modes[mode_idx]

        # Accumulate sufficient statistics
        mode['sum_xx'] += np.outer(x_prev, x_prev)
        mode['sum_xy'] += np.outer(x_prev, x_curr)
        mode['count'] += 1

        # Solve for D, b via least squares
        # x_curr = D @ x_prev + b
        # Augment: [x_curr] = [D, b] @ [x_prev; 1]
        n = mode['count']
        if n > self.state_dim:
            # Regularized least squares
            reg = 0.01 * np.eye(self.state_dim)
            D_new = mode['sum_xy'] @ np.linalg.inv(mode['sum_xx'] + reg)
            mode['D'] = D_new

        self.mode_counts[mode_idx] = mode['count']
```

### 4.3 Recurrent Mixture Model Implementation

```python
class RecurrentMixtureModel:
    """Models interactions and predicts switch states."""

    def __init__(self, max_clusters=100, alpha_0=0.1):
        self.max_clusters = max_clusters
        self.alpha_0 = alpha_0

        # Each cluster models joint distribution over:
        # - Continuous: own position, distance to nearest
        # - Discrete: own identity, nearest identity, action, reward, next switch
        self.clusters = []

    def add_cluster(self, continuous_features, discrete_features):
        """Add new interaction cluster."""
        cluster = {
            # Continuous (NIW posterior)
            'cont_mean': continuous_features.copy(),
            'cont_scatter': np.eye(len(continuous_features)) * 0.01,
            'cont_kappa': 1,
            'cont_nu': len(continuous_features) + 2,

            # Discrete (Dirichlet posteriors for each categorical)
            'discrete_counts': {},
            'count': 1
        }

        # Initialize discrete counts
        for key, value in discrete_features.items():
            cluster['discrete_counts'][key] = {value: 1}

        self.clusters.append(cluster)

    def compute_cluster_likelihood(self, cluster, cont_feat, disc_feat):
        """Compute likelihood of observation under cluster."""
        # Continuous: Student-t predictive
        diff = cont_feat - cluster['cont_mean']
        scale = cluster['cont_scatter'] * (cluster['cont_kappa'] + 1) / (
            cluster['cont_kappa'] * (cluster['cont_nu'] - len(cont_feat) + 1))

        cont_ll = -0.5 * diff @ np.linalg.inv(scale + 1e-6 * np.eye(len(diff))) @ diff

        # Discrete: Categorical likelihood with Dirichlet prior
        disc_ll = 0
        for key, value in disc_feat.items():
            counts = cluster['discrete_counts'].get(key, {})
            total = sum(counts.values()) + self.alpha_0
            count = counts.get(value, 0) + self.alpha_0 / 10
            disc_ll += np.log(count / total)

        return cont_ll + disc_ll

    def infer_cluster(self, cont_feat, disc_feat):
        """Infer which interaction cluster explains current state."""
        if len(self.clusters) == 0:
            self.add_cluster(cont_feat, disc_feat)
            return 0

        log_liks = np.array([
            self.compute_cluster_likelihood(c, cont_feat, disc_feat)
            for c in self.clusters
        ])

        # Add mixing weight prior
        counts = np.array([c['count'] for c in self.clusters])
        log_liks += np.log(counts + self.alpha_0)

        # Check if new cluster needed
        max_ll = np.max(log_liks)
        threshold = np.log(self.alpha_0) - 5  # Prior predictive threshold

        if max_ll < threshold and len(self.clusters) < self.max_clusters:
            self.add_cluster(cont_feat, disc_feat)
            return len(self.clusters) - 1

        return np.argmax(log_liks)

    def update_cluster(self, cluster_idx, cont_feat, disc_feat):
        """Update cluster with new observation."""
        cluster = self.clusters[cluster_idx]

        # Update continuous (NIW)
        n = cluster['count']
        old_mean = cluster['cont_mean']

        cluster['cont_kappa'] += 1
        cluster['cont_nu'] += 1
        cluster['cont_mean'] = (n * old_mean + cont_feat) / (n + 1)

        delta = cont_feat - old_mean
        cluster['cont_scatter'] += (n / (n + 1)) * np.outer(delta, delta)

        # Update discrete (Dirichlet)
        for key, value in disc_feat.items():
            if key not in cluster['discrete_counts']:
                cluster['discrete_counts'][key] = {}
            counts = cluster['discrete_counts'][key]
            counts[value] = counts.get(value, 0) + 1

        cluster['count'] += 1

    def predict_switch(self, cluster_idx):
        """Predict next tMM switch state from cluster."""
        cluster = self.clusters[cluster_idx]
        switch_counts = cluster['discrete_counts'].get('next_switch', {})

        if not switch_counts:
            return 0

        # Return mode (most likely switch)
        return max(switch_counts, key=switch_counts.get)

    def predict_reward(self, cluster_idx):
        """Predict reward from cluster."""
        cluster = self.clusters[cluster_idx]
        reward_counts = cluster['discrete_counts'].get('reward', {})

        if not reward_counts:
            return 0

        # Return expected reward
        total = sum(reward_counts.values())
        return sum(r * c / total for r, c in reward_counts.items())
```

### 4.4 Complete AXIOM Agent

```python
class AXIOMAgent:
    """Complete AXIOM agent for game playing."""

    def __init__(self, action_space=4):
        self.smm = SlotMixtureModel(max_slots=10)
        self.imm = IdentityMixtureModel(max_types=10)  # Simplified
        self.tmm = TransitionMixtureModel(state_dim=7, max_modes=20)
        self.rmm = RecurrentMixtureModel(max_clusters=100)

        self.action_space = action_space
        self.prev_slots = None
        self.prev_identities = None

    def observe(self, image, action, reward):
        """Process observation and update world model."""
        # 1. Segment into slots
        slot_positions = self.smm.infer(image)

        # Simplified: use positions as full state
        slots = [np.concatenate([pos, np.zeros(5)]) for pos in slot_positions]

        if self.prev_slots is not None:
            # 2-4. Update models for each object
            for k, (curr, prev) in enumerate(zip(slots, self.prev_slots)):
                if k >= len(self.prev_slots):
                    break

                # Find nearest other object
                distances = [np.linalg.norm(curr[:2] - s[:2])
                           for i, s in enumerate(slots) if i != k]
                if distances:
                    nearest_idx = np.argmin(distances)
                    nearest_dist = distances[nearest_idx]
                else:
                    nearest_dist = 1.0

                # Build features for rMM
                cont_feat = np.array([curr[0], curr[1], nearest_dist])
                disc_feat = {
                    'action': action,
                    'reward': int(reward > 0) - int(reward < 0)  # Discretize
                }

                # Infer and update rMM
                cluster_idx = self.rmm.infer_cluster(cont_feat, disc_feat)

                # Infer and update tMM
                mode_idx = self.tmm.infer_mode(prev, curr)

                # Link rMM cluster to tMM mode
                disc_feat['next_switch'] = mode_idx
                self.rmm.update_cluster(cluster_idx, cont_feat, disc_feat)
                self.tmm.update_mode(mode_idx, prev, curr)

        self.prev_slots = slots

    def plan(self, horizon=10, n_rollouts=64):
        """Plan using expected free energy."""
        if self.prev_slots is None or len(self.rmm.clusters) == 0:
            return np.random.randint(self.action_space)

        best_action = 0
        best_efe = float('inf')

        for action in range(self.action_space):
            total_efe = 0

            for _ in range(n_rollouts):
                # Simulate trajectory
                slots = [s.copy() for s in self.prev_slots]
                cumulative_reward = 0

                for tau in range(horizon):
                    a = action if tau == 0 else np.random.randint(self.action_space)

                    for k, slot in enumerate(slots):
                        # Find nearest
                        distances = [np.linalg.norm(slot[:2] - s[:2])
                                   for i, s in enumerate(slots) if i != k]
                        nearest_dist = min(distances) if distances else 1.0

                        cont_feat = np.array([slot[0], slot[1], nearest_dist])
                        disc_feat = {'action': a, 'reward': 0}

                        # Predict using rMM
                        cluster_idx = self.rmm.infer_cluster(cont_feat, disc_feat)
                        switch = self.rmm.predict_switch(cluster_idx)
                        reward = self.rmm.predict_reward(cluster_idx)

                        # Predict next state using tMM
                        if switch < len(self.tmm.modes):
                            slots[k] = self.tmm.predict(slot, switch)

                        cumulative_reward += reward

                # EFE = -utility (negative reward)
                total_efe -= cumulative_reward

            avg_efe = total_efe / n_rollouts

            if avg_efe < best_efe:
                best_efe = avg_efe
                best_action = action

        return best_action
```

---

## Section 5: TRAIN STATION - Axiom = Bayesian NN = Uncertainty Quantification

### 5.1 The Deep Connection

**TRAIN STATION INSIGHT**: AXIOM, Bayesian Neural Networks, and Uncertainty Quantification are all manifestations of the same underlying principle - representing knowledge as probability distributions rather than point estimates.

```
AXIOM             =    Bayesian NN      =    Uncertainty Quantification
   |                       |                          |
Mixture posteriors   Weight distributions     Predictive distributions
   |                       |                          |
Conjugate updates    Variational inference    Posterior predictive
   |                       |                          |
Object-centric       Distributed repr.        Task-agnostic
```

### 5.2 Why This Matters

**The Core Principle**: To make decisions under uncertainty, you need to know WHAT you don't know.

| Approach | How It Represents Uncertainty |
|----------|------------------------------|
| **Standard NN** | Doesn't (point estimates) |
| **Dropout** | Approximates via sampling |
| **Ensemble** | Multiple models disagree |
| **BNN** | Distributions over weights |
| **AXIOM** | Distributions over structure |

### 5.3 AXIOM as Bayesian Neural Network

AXIOM can be viewed as a highly structured BNN:

```python
# Standard BNN: p(w) -> p(y|x,w)
# AXIOM: p(theta) -> p(y|x,theta) where theta has STRUCTURE

# BNN weight posterior
p(w|D) = p(D|w) * p(w) / p(D)

# AXIOM parameter posterior (e.g., for iMM)
p(mu, Sigma | D) = p(D | mu, Sigma) * p(mu, Sigma) / p(D)
# But with CONJUGATE priors -> exact inference!
```

**Key Difference**: AXIOM's structure makes inference tractable

### 5.4 Uncertainty Types in AXIOM

**Epistemic Uncertainty** (model uncertainty):
- Which cluster does this interaction belong to?
- How many dynamics modes exist?
- What are the dynamics parameters?

```python
# Epistemic uncertainty in tMM
# Parameter posterior gives uncertainty in D, b
mode_uncertainty = mode['count']  # More data -> less uncertainty
```

**Aleatoric Uncertainty** (data uncertainty):
- Where exactly is this object?
- What color is it?
- How much noise in dynamics?

```python
# Aleatoric uncertainty in sMM
slot_extent = slot['extent']  # Captures spread of object
color_variance = sigma_c  # Noise in color observation
```

### 5.5 Information-Theoretic Planning

AXIOM uses uncertainty for principled exploration:

```python
# Expected Free Energy
G = -E[log p(reward)] + KL[q(params|future) || q(params)]
#   ^ Utility term      ^ Information gain term

# Information gain drives exploration:
# - High when model uncertain about interaction outcome
# - Low when cluster well-established
# - Naturally decays as agent learns
```

**This is impossible without uncertainty quantification!**

### 5.6 Comparison: AXIOM vs BNN Approaches

```python
# MC Dropout (approximate)
def predict_with_uncertainty_mcdropout(model, x, n_samples=100):
    preds = [model(x, training=True) for _ in range(n_samples)]
    mean = np.mean(preds, axis=0)
    var = np.var(preds, axis=0)
    return mean, var  # Approximate!

# Deep Ensemble (expensive)
def predict_with_uncertainty_ensemble(models, x):
    preds = [m(x) for m in models]
    mean = np.mean(preds, axis=0)
    var = np.var(preds, axis=0)
    return mean, var  # Requires K forward passes

# AXIOM (exact and efficient)
def predict_with_uncertainty_axiom(tmm, x, mode_idx):
    mode = tmm.modes[mode_idx]
    mean = mode['D'] @ x + mode['b']
    # Uncertainty from NIW posterior -> Student-t predictive
    var = mode['scatter'] * (mode['kappa'] + 1) / (
        mode['kappa'] * (mode['nu'] - dim + 1))
    return mean, var  # Exact! Single forward pass!
```

### 5.7 The Topological Equivalence

**Coffee Cup = Donut Topology**:

```
Loss minimization     <-->  Free energy minimization
Gradient descent      <-->  Variational inference
Point estimates       <-->  Posterior distributions
Backpropagation       <-->  Message passing
Attention weights     <-->  Mixture responsibilities
Layer activations     <-->  Belief states
```

AXIOM shows that you can get the SAME computational power with a DIFFERENT mathematical formulation that naturally includes uncertainty.

---

## Section 6: ARR-COC-0-1 Connection - Uncertainty in Relevance

### 6.1 The Connection

ARR-COC-0-1 (Attention-Relevance-Routing for Compute-On-Chip) deals with allocating computational resources based on **relevance**. AXIOM's uncertainty quantification provides a principled foundation for this:

**Key Insight**: Relevance should be a DISTRIBUTION, not a point estimate!

### 6.2 Relevance as Belief Distribution

Instead of:
```python
relevance = model(token)  # Point estimate
```

Use:
```python
relevance_distribution = posterior(token)  # Full uncertainty
```

### 6.3 AXIOM-Inspired Token Routing

```python
class UncertainRelevanceRouter:
    """Route tokens using uncertainty-aware relevance."""

    def __init__(self, n_experts=4):
        # Each expert has posterior over relevance
        self.expert_posteriors = [
            {'alpha': np.ones(10), 'beta': np.ones(10)}  # Beta posteriors
            for _ in range(n_experts)
        ]

    def compute_relevance_distribution(self, token, expert_idx):
        """Get full distribution over relevance."""
        posterior = self.expert_posteriors[expert_idx]

        # Beta posterior -> Beta predictive
        alpha = posterior['alpha'][token.type]
        beta = posterior['beta'][token.type]

        return {
            'mean': alpha / (alpha + beta),
            'var': (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)),
            'alpha': alpha,
            'beta': beta
        }

    def route_with_uncertainty(self, token):
        """Route considering both expected relevance AND uncertainty."""
        best_expert = 0
        best_score = -float('inf')

        for i in range(len(self.expert_posteriors)):
            dist = self.compute_relevance_distribution(token, i)

            # UCB-style: mean + exploration bonus
            exploration_bonus = np.sqrt(dist['var'])
            score = dist['mean'] + exploration_bonus

            if score > best_score:
                best_score = score
                best_expert = i

        return best_expert

    def update_after_routing(self, expert_idx, token, was_relevant):
        """Bayesian update of expert posterior."""
        posterior = self.expert_posteriors[expert_idx]

        if was_relevant:
            posterior['alpha'][token.type] += 1
        else:
            posterior['beta'][token.type] += 1
```

### 6.4 Expected Free Energy for Token Allocation

```python
def compute_token_efe(token, allocation, belief_state):
    """Expected free energy for token allocation decision."""

    # Utility: Expected relevance contribution
    expected_relevance = belief_state['relevance_mean'][allocation]
    utility = np.log(expected_relevance + 1e-6)

    # Information gain: How much would this allocation teach us?
    current_uncertainty = belief_state['relevance_var'][allocation]
    # High uncertainty -> high information gain
    info_gain = 0.5 * np.log(1 + current_uncertainty)

    # EFE = -utility + info_gain (we minimize this)
    return -utility + info_gain

def select_allocation(token, possible_allocations, belief_state):
    """Select allocation minimizing expected free energy."""
    efes = [compute_token_efe(token, a, belief_state)
            for a in possible_allocations]
    return possible_allocations[np.argmin(efes)]
```

### 6.5 Practical Applications in ARR-COC

**1. Adaptive Token Budgets**:
```python
# High uncertainty tokens get more compute
if token_uncertainty > threshold:
    allocate_more_layers(token)
```

**2. Expert Selection with Exploration**:
```python
# Balance exploitation (best expert) with exploration (uncertain experts)
expert = select_by_thompson_sampling(expert_posteriors)
```

**3. Dynamic Precision Adjustment**:
```python
# AXIOM-style precision learning for attention
precision = 1 / uncertainty
weighted_value = precision * value
```

**4. Online Relevance Learning**:
```python
# Update relevance model online like AXIOM
# No replay buffer needed!
for token, feedback in stream:
    update_relevance_posterior(token, feedback)
```

### 6.6 The Full Picture

```
AXIOM Architecture        ARR-COC Application
===============================================
sMM (segmentation)    ->  Token segmentation/parsing
iMM (identity)        ->  Token type classification
tMM (dynamics)        ->  Relevance dynamics over sequence
rMM (interactions)    ->  Token-token relevance interactions

Belief states         ->  Relevance distributions
Information gain      ->  Exploration in routing
Expected utility      ->  Expected relevance contribution
Active inference      ->  Adaptive token allocation
```

---

## Section 7: Performance Notes

### 7.1 Computational Efficiency

From the AXIOM paper benchmarks:

| Model | Parameters | Update Time | Planning Time |
|-------|------------|-------------|---------------|
| BBF | 6.47M | 135ms | N/A |
| DreamerV3 | 420M | 221ms | 823ms |
| **AXIOM** | **0.3-1.6M** | **18ms** | **252-534ms** |

**Key Advantages**:
- 20x fewer parameters than DreamerV3
- 7x faster model updates than BBF
- No GPU required for basic inference

### 7.2 Sample Efficiency

AXIOM achieves higher cumulative reward than baselines on 8/10 games within 10,000 steps.

**Why So Efficient?**:
1. Single-pass learning (no replay)
2. Exact Bayesian updates (no gradient noise)
3. Structure learning (right model complexity)
4. Information-seeking exploration (efficient data collection)

### 7.3 Memory Efficiency

```python
# Standard RL
replay_buffer = 1_000_000 * transition_size  # ~4GB

# AXIOM
total_memory = sum(
    cluster['count'] * feature_dim  # Just sufficient statistics!
    for cluster in rmm.clusters
)  # ~10MB
```

### 7.4 Scaling Considerations

**Current Limitations**:
- Visual complexity: Works best with simple sprites
- Game complexity: 10,000 steps is ~12 minutes of play
- Action space: Tested with 2-4 discrete actions

**Future Directions**:
- Hierarchical sMM for complex visuals
- Deeper tMM for long-horizon credit assignment
- Continuous action support

---

## Sources

### Primary Sources

**AXIOM Paper and Code**:
- [AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models](https://arxiv.org/html/2505.24784v1) - arXiv:2505.24784 (accessed 2025-11-23)
- [VERSES AI Research Blog - AXIOM](https://www.verses.ai/research-blog/axiom-mastering-arcade-games-in-minutes-with-active-inference-and-structure-learning) (accessed 2025-11-23)
- [GitHub: VersesTech/axiom](https://github.com/VersesTech/axiom) - Official implementation
- [GitHub: VersesTech/gameworld](https://github.com/VersesTech/gameworld) - Benchmark environments

### Background on Active Inference

**Free Energy Principle and Active Inference**:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces: A synthesis.

### Related Work

**Object-Centric Learning**:
- Locatello, F., et al. (2020). Object-centric learning with slot attention.
- Greff, K., et al. (2019). Multi-object representation learning with iterative variational inference.

**Switching Linear Dynamical Systems**:
- Linderman, S., et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems.

**Bayesian Neural Networks**:
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks.

### Additional Resources

**VERSES AI**:
- [VERSES AI Official Site](https://www.verses.ai)
- [Active Inference Research](https://www.verses.ai/active-inference-research)

**Uncertainty Quantification**:
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning?
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles.

---

## Summary

AXIOM represents a fundamental rethinking of how to build intelligent agents. By abandoning the transformer paradigm in favor of structured Bayesian mixture models, it achieves:

1. **Human-like sample efficiency**: Learn games in minutes, not hours
2. **Native uncertainty quantification**: No approximations or tricks
3. **Interpretable representations**: Objects, dynamics, interactions - all readable
4. **Online learning**: No replay buffer, no batch training
5. **Gradient-free optimization**: Exact conjugate updates

The TRAIN STATION connection reveals that AXIOM, Bayesian NNs, and uncertainty quantification are topologically equivalent - different views of the same principle that knowledge should be represented as distributions. This insight directly applies to ARR-COC: token relevance should be uncertain and adaptive, not fixed and deterministic.

**The future of efficient AI may not be bigger transformers, but smarter priors.**
