# Attention = Precision = Salience: The Core Gain Control Unification

**TRAIN STATION PHILOSOPHY**: This is where neuroscience, cognitive science, and machine learning meet at the same platform. Attention mechanisms in transformers, precision weighting in predictive coding, salience networks in the brain, and relevance in active inference are all **THE SAME FUNDAMENTAL COMPUTATION**: dynamic gain control that amplifies task-relevant signals.

**ML-HEAVY**: PyTorch implementations, architectural patterns, performance implications.

---

## 1. The Core Unification: All Attention is Gain Control

### 1.1 What Attention Actually Does

At its core, attention is **multiplicative gain modulation**:

```
output = gain × signal
```

This appears across all domains:

**Neuroscience**:
- V4 neurons show 20-30% firing rate increases when stimulus is attended
- Modulation is multiplicative, not additive
- Effect scales with neuron's tuning preference

**Predictive Coding**:
- Precision (inverse variance) weights prediction errors
- Higher precision = stronger influence on inference
- π_i = 1/σ²_i (precision = inverse variance)

**Machine Learning Attention**:
- Attention weights α_i scale value vectors
- Softmax ensures competition (limited resources)
- Multi-head attention = multiple gain fields

**Active Inference**:
- Precision controls sensory vs. prior influence
- Expected precision determines attention allocation
- Salience = expected information gain × precision

### 1.2 The Mathematical Unity

All these mechanisms share the same core equation:

```python
# Universal attention/precision/salience equation
weighted_signal = gain_factor * signal + (1 - gain_factor) * baseline

# Where gain_factor comes from:
# - Neuroscience: tuning similarity to attended feature
# - Predictive coding: inverse variance (precision)
# - ML attention: softmax(Q·K^T / sqrt(d))
# - Active inference: expected free energy minimization
```

The gain factor is always:
1. **Data-dependent**: Changes based on input
2. **Normalized**: Limited resources require competition
3. **Multiplicative**: Scales signal, doesn't add to it
4. **Selective**: Enhances some signals at expense of others

---

## 2. Neuroscience: Attention as Neural Gain Control

### 2.1 Feature Similarity Gain Model (FSGM)

**The biological evidence** (Treue & Martinez-Trujillo, 1999; Lindsay & Miller, 2018):

Neurons modulate their firing rates according to similarity between:
- Their preferred stimulus (tuning)
- The attended stimulus

```
firing_rate_attended = baseline_firing * (1 + β * similarity_to_attended)

Where:
- β = attention strength (~0.2-0.3 experimentally)
- similarity ∈ [-1, 1] based on tuning curves
```

**Key findings**:
- Multiplicative scaling (not additive)
- Bidirectional (suppress non-attended features)
- Spatially global for features
- Stronger in later visual areas (V4, IT > V1)

### 2.2 Salience Network: Detecting What Matters

**The salience network** (anterior insula + anterior cingulate cortex):

Functions to identify the most **relevant** stimuli among internal/external inputs:
- Bottom-up salience: stimulus-driven (pop-out, novelty)
- Top-down salience: goal-driven (task relevance)
- Salience = relevance = "worthiness of attention"

**Computational role**:
```python
salience_map = bottom_up_features + top_down_goals + prediction_error

# Then:
attention_allocation = softmax(salience_map / temperature)
```

The salience network IS a meta-controller for attention/precision allocation!

### 2.3 Precision Weighting in Neural Circuits

**Precision = confidence = reliability = inverse variance**

Neural implementation:
```
E[error | precision] = precision_weight * prediction_error

High precision (confident prediction):
  → Large weight on prediction error
  → Strong learning signal
  → Attention to this input

Low precision (uncertain):
  → Small weight on prediction error
  → Weak learning signal
  → Ignore this input
```

**Neural mechanisms for precision**:
- Gain modulation of neural responses (attention!)
- Neuromodulators (acetylcholine, norepinephrine)
- Oscillatory synchronization (gamma band coherence)

From Lindsay (2020): "Attention IS the important ability to flexibly control limited computational resources."

---

## 3. Predictive Coding: Precision-Weighted Prediction Errors

### 3.1 The Precision-Weighted Update Rule

Predictive coding learns by minimizing precision-weighted prediction errors:

```
∂μ/∂t = -∂F/∂μ = Π_ε · ε

Where:
- μ = prediction
- ε = prediction error (observation - prediction)
- Π_ε = precision matrix (inverse covariance)
- F = free energy
```

**Precision determines influence**:
- High precision errors → strong updates → attention
- Low precision errors → weak updates → ignore

### 3.2 Precision as Attention Mechanism

```python
import torch
import torch.nn as nn

class PrecisionWeightedPredictiveCoding(nn.Module):
    """Predictive coding with learnable precision weights"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.prediction_net = nn.Linear(hidden_dim, input_dim)
        self.error_net = nn.Linear(input_dim, hidden_dim)

        # Precision weights (learnable!)
        self.log_precision = nn.Parameter(torch.zeros(input_dim))

    def forward(self, observation, hidden_state):
        # Generate prediction
        prediction = self.prediction_net(hidden_state)

        # Compute prediction error
        error = observation - prediction

        # Apply precision weighting (THIS IS ATTENTION!)
        precision = torch.exp(self.log_precision)
        weighted_error = precision * error

        # Update hidden state based on weighted error
        hidden_update = self.error_net(weighted_error)
        new_hidden = hidden_state + 0.1 * hidden_update

        return new_hidden, prediction, weighted_error

    def get_attention_weights(self):
        """Precision weights ARE attention weights"""
        return torch.softmax(self.log_precision, dim=0)
```

**Key insight**: Precision weighting in predictive coding is mathematically equivalent to attention weighting in transformers!

### 3.3 Expected Precision and Active Inference

**Expected precision** guides information seeking:

```
Expected precision = E[Π | current beliefs]

Action selection:
  a* = argmax_a E[precision of observations | action a]
```

This explains:
- **Epistemic value**: Actions that reduce uncertainty (high expected precision gain)
- **Pragmatic value**: Actions that achieve goals
- **Attention allocation**: Look where you expect to learn most

**Biological implementation**: Dopamine may signal precision/confidence (Friston et al., 2014)

---

## 4. Machine Learning: Attention Mechanisms

### 4.1 Self-Attention as Precision-Weighted Message Passing

**The transformer attention equation**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

This IS precision weighting!
```

Breaking it down:
1. `QK^T`: Similarity (like neural tuning similarity!)
2. `/ √d_k`: Temperature (normalize scale)
3. `softmax()`: Competition (limited resources!)
4. `× V`: Gain modulation (multiply signal by weight!)

### 4.2 Unified Attention-Precision Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UnifiedAttentionPrecision(nn.Module):
    """
    Attention = Precision = Gain Control

    Three equivalent views:
    1. ML Attention: Query-key similarity scaling
    2. Precision weighting: Inverse-variance weighting
    3. Neural gain: Feature-similarity gain modulation
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1,
                 use_precision=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_precision = use_precision

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Precision estimation network (optional)
        if use_precision:
            self.precision_net = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus()  # Ensures positive precision
            )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None,
                return_attention=False):
        batch_size = query.size(0)

        # Linear projections in batch from d_model => n_heads x d_k
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k)

        # Transpose for attention: (batch, n_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ===== CORE ATTENTION = GAIN CONTROL =====

        # 1. Compute similarity (like neural tuning similarity!)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 2. Optional: Add precision weighting
        if self.use_precision:
            # Estimate precision for each key position
            key_flat = key.view(-1, self.d_model)
            precision = self.precision_net(key_flat)
            precision = precision.view(batch_size, 1, 1, -1)

            # Precision modulates attention scores
            # High precision = more confident = stronger signal
            scores = scores * precision

        # 3. Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax = competition (limited resources!)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. Apply gain to values (multiplicative modulation!)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        if return_attention:
            return output, attention_weights
        return output

    def get_effective_precision(self, query, key):
        """
        Extract precision weights (attention weights)
        Shows equivalence to precision weighting!
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if self.use_precision:
            key_flat = key.view(-1, self.d_model)
            precision = self.precision_net(key_flat)
            precision = precision.view(batch_size, 1, 1, -1)
            scores = scores * precision

        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights  # These ARE precision weights!


# ===== EXAMPLE USAGE =====

def demo_attention_precision_equivalence():
    """Demonstrate that attention = precision weighting"""

    # Setup
    batch_size = 2
    seq_len = 10
    d_model = 512

    # Create model
    model = UnifiedAttentionPrecision(
        d_model=d_model,
        n_heads=8,
        use_precision=True
    )

    # Create dummy data
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attention = model(x, x, x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights sum to 1: {attention.sum(dim=-1)[0, 0]}")

    # Show precision view
    precision_weights = model.get_effective_precision(x, x)
    print(f"\nPrecision weights (attention) shape: {precision_weights.shape}")
    print(f"These ARE the same as attention weights!")

    return model, x, output, attention


# ===== FEATURE SIMILARITY GAIN (NEUROSCIENCE) =====

class FeatureSimilarityGainAttention(nn.Module):
    """
    Neural-inspired attention based on feature tuning similarity
    Implements the Feature Similarity Gain Model (Treue & Martinez-Trujillo)
    """

    def __init__(self, d_model, n_features=128):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Learnable feature tuning curves
        self.feature_centers = nn.Parameter(
            torch.randn(n_features, d_model)
        )

        # Gain modulation strength (β in FSGM)
        self.beta = nn.Parameter(torch.tensor(0.3))

    def compute_tuning_similarity(self, x, attended_feature):
        """
        Compute similarity between input and preferred features
        Similar to neural tuning curves!
        """
        # x: (batch, seq_len, d_model)
        # attended_feature: (batch, d_model)

        batch_size, seq_len, _ = x.shape

        # Compute similarity to all feature centers
        # (batch, seq_len, n_features)
        x_expanded = x.unsqueeze(2)  # (batch, seq_len, 1, d_model)
        centers = self.feature_centers.unsqueeze(0).unsqueeze(0)

        # Cosine similarity (like neural tuning!)
        similarities = F.cosine_similarity(
            x_expanded, centers, dim=-1
        )

        # Find which feature is attended
        attended_expanded = attended_feature.unsqueeze(1)
        attended_sim = F.cosine_similarity(
            attended_expanded.unsqueeze(2),
            centers.squeeze(0).squeeze(0),
            dim=-1
        )
        best_feature = torch.argmax(attended_sim, dim=-1)

        # Extract similarity to attended feature
        batch_idx = torch.arange(batch_size).unsqueeze(1)
        seq_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        tuning_sim = similarities[batch_idx, seq_idx, best_feature.unsqueeze(1)]

        return tuning_sim

    def forward(self, x, attended_feature=None):
        """
        Apply feature similarity gain modulation

        Args:
            x: Input features (batch, seq_len, d_model)
            attended_feature: Feature to attend (batch, d_model)
                            If None, uses mean of x

        Returns:
            Gain-modulated features
        """
        if attended_feature is None:
            attended_feature = x.mean(dim=1)

        # Compute tuning similarity
        tuning_sim = self.compute_tuning_similarity(x, attended_feature)

        # Apply FSGM: firing_rate = baseline * (1 + β * similarity)
        gain = 1.0 + self.beta * tuning_sim.unsqueeze(-1)

        # Multiplicative modulation (THIS IS ATTENTION!)
        output = x * gain

        return output, gain
```

### 4.3 Multi-Head Attention = Multiple Precision Fields

**Each attention head = different precision/salience computation**

```python
class MultiHeadPrecisionAttention(nn.Module):
    """
    Multi-head attention = Multiple precision hypotheses

    Each head captures different aspects of relevance/precision:
    - Head 1: Spatial precision
    - Head 2: Semantic precision
    - Head 3: Temporal precision
    - etc.
    """

    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([
            UnifiedAttentionPrecision(d_model, n_heads=1)
            for _ in range(n_heads)
        ])
        self.combine = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x):
        # Each head computes its own precision weighting
        head_outputs = [head(x, x, x) for head in self.heads]

        # Combine multiple precision perspectives
        combined = torch.cat(head_outputs, dim=-1)
        output = self.combine(combined)

        return output
```

---

## 5. The TRAIN STATION: Where All Mechanisms Meet

### 5.1 The Core Equivalence

**Attention (ML) = Precision (PC) = Salience (Neuro) = Relevance (AI)**

All implement the same computation:

```python
def universal_gain_control(signal, context, gain_computer):
    """
    Universal attention/precision/salience mechanism

    Args:
        signal: Input to be modulated
        context: What determines relevance/precision
        gain_computer: Function that computes gain from context

    Returns:
        Gain-modulated signal
    """
    # 1. Compute relevance/precision/salience
    gain_factors = gain_computer(signal, context)

    # 2. Normalize (limited resources!)
    gain_factors = softmax(gain_factors)

    # 3. Apply multiplicative modulation
    output = gain_factors * signal

    return output
```

**Instantiations**:

```python
# ML Attention
def ml_attention(Q, K, V):
    gain = softmax(Q @ K.T / sqrt(d_k))
    return gain @ V

# Precision Weighting
def precision_weighting(error, variance):
    precision = 1 / variance
    gain = precision / sum(precision)
    return gain * error

# Neural Gain Modulation
def neural_gain(firing_rate, tuning_similarity, beta):
    gain = 1 + beta * tuning_similarity
    return gain * firing_rate

# Salience-Based Selection
def salience_selection(stimuli, goals):
    salience = bottom_up(stimuli) + top_down(goals)
    gain = softmax(salience)
    return gain * stimuli
```

### 5.2 Why They're the Same

**Common computational principles**:

1. **Limited Resources**: Can't process everything equally
2. **Context-Dependent**: What's relevant changes with task/goals
3. **Multiplicative**: Scales signal strength, not content
4. **Competitive**: Emphasizing one thing de-emphasizes others
5. **Dynamic**: Updates moment-to-moment

**Common neural implementation**:
- Gain modulation of neural responses
- Synchronized oscillations (gamma for attention, beta for precision)
- Neuromodulatory control (ACh, DA, NE)

### 5.3 Integrated Architecture

```python
class IntegratedAttentionPrecisionSalience(nn.Module):
    """
    Full integration of all attention/precision/salience mechanisms

    This is what the brain might be doing!
    """

    def __init__(self, d_model, n_heads=8):
        super().__init__()

        # Bottom-up saliency (stimulus-driven)
        self.bottom_up_salience = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Top-down attention (goal-driven)
        self.top_down_attention = UnifiedAttentionPrecision(
            d_model, n_heads
        )

        # Precision estimation (confidence/uncertainty)
        self.precision_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )

        # Feature similarity gain (neural tuning)
        self.feature_gain = FeatureSimilarityGainAttention(d_model)

        # Integration weights
        self.integration_weights = nn.Parameter(
            torch.ones(4) / 4  # Bottom-up, top-down, precision, FSGM
        )

    def forward(self, x, goal_context=None):
        """
        Integrate all sources of attention/precision/salience
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Bottom-up salience (what stands out?)
        salience_scores = self.bottom_up_salience(x).squeeze(-1)
        salience_gain = F.softmax(salience_scores, dim=-1).unsqueeze(-1)

        # 2. Top-down attention (what's relevant to goals?)
        if goal_context is None:
            goal_context = x.mean(dim=1, keepdim=True)
        td_output, td_weights = self.top_down_attention(
            goal_context.expand(-1, seq_len, -1),
            x, x,
            return_attention=True
        )
        td_gain = td_weights.mean(dim=1)  # Average over heads

        # 3. Precision weighting (how confident?)
        precision = self.precision_estimator(x).squeeze(-1)
        precision_gain = precision / precision.sum(dim=-1, keepdim=True)
        precision_gain = precision_gain.unsqueeze(-1)

        # 4. Feature similarity gain (neural tuning)
        if goal_context is not None:
            fg_output, fg_gain = self.feature_gain(
                x, goal_context.squeeze(1)
            )
        else:
            fg_output, fg_gain = self.feature_gain(x)

        # Normalize FSGM gain to [0, 1]
        fg_gain_norm = (fg_gain - fg_gain.min()) / (
            fg_gain.max() - fg_gain.min() + 1e-8
        )

        # 5. Integrate all sources
        w = F.softmax(self.integration_weights, dim=0)

        combined_gain = (
            w[0] * salience_gain +
            w[1] * td_gain.mean(dim=1).unsqueeze(-1) +
            w[2] * precision_gain +
            w[3] * fg_gain_norm
        )

        # Apply integrated gain modulation
        output = x * combined_gain

        return output, {
            'salience': salience_gain,
            'top_down': td_gain,
            'precision': precision_gain,
            'feature_similarity': fg_gain_norm,
            'combined': combined_gain,
            'integration_weights': w
        }


# ===== DEMONSTRATION =====

def demonstrate_train_station():
    """Show all mechanisms computing the same thing"""

    batch_size = 4
    seq_len = 16
    d_model = 256

    # Create integrated model
    model = IntegratedAttentionPrecisionSalience(d_model, n_heads=8)

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    goal = torch.randn(batch_size, 1, d_model)

    # Forward pass
    output, gains = model(x, goal)

    print("=" * 60)
    print("TRAIN STATION: All Mechanisms Computing Gain Control")
    print("=" * 60)

    print(f"\nInput shape: {x.shape}")
    print(f"Goal shape: {goal.shape}")
    print(f"Output shape: {output.shape}")

    print(f"\n--- Gain Sources ---")
    for name, gain in gains.items():
        if name != 'integration_weights':
            print(f"{name:20s}: shape {gain.shape}, "
                  f"mean {gain.mean():.4f}, std {gain.std():.4f}")

    print(f"\n--- Integration Weights ---")
    w = gains['integration_weights']
    sources = ['Bottom-up salience', 'Top-down attention',
               'Precision', 'Feature similarity']
    for source, weight in zip(sources, w):
        print(f"{source:25s}: {weight:.4f}")

    print("\n" + "=" * 60)
    print("All sources contribute to unified gain control!")
    print("Attention = Precision = Salience = Relevance")
    print("=" * 60)

    return model, output, gains
```

---

## 6. Signal Detection Theory: Precision and Criteria

### 6.1 Precision Affects Sensitivity, Attention Affects Criteria

From Lindsay & Miller (2018):

**Feature-based attention**: Primarily decreases criteria (threshold)
- Shifts representations toward attended category
- Lowers detection threshold
- Increases true positives before false positives

**Spatial attention**: Primarily increases sensitivity (d')
- Amplifies differences in representations
- Separates signal from noise better
- Increases discriminability

```python
def signal_detection_with_attention(signals, attention_type='feature'):
    """
    Model how attention/precision affects signal detection
    """
    # Signal and noise distributions
    signal_dist = torch.distributions.Normal(1.0, 1.0)
    noise_dist = torch.distributions.Normal(0.0, 1.0)

    if attention_type == 'feature':
        # Feature attention: Shift means (criteria change)
        signal_mean_shift = 0.5
        signal_dist = torch.distributions.Normal(
            1.0 + signal_mean_shift, 1.0
        )
        # Sensitivity (d') increases slightly
        # Criteria decreases (threshold lower)

    elif attention_type == 'spatial':
        # Spatial attention: Reduce variance (sensitivity increase)
        precision_factor = 2.0
        signal_std = 1.0 / precision_factor
        noise_std = 1.0 / precision_factor

        signal_dist = torch.distributions.Normal(1.0, signal_std)
        noise_dist = torch.distributions.Normal(0.0, noise_std)
        # Sensitivity (d') increases substantially
        # Criteria less affected

    # Compute d-prime (sensitivity)
    d_prime = (signal_dist.mean - noise_dist.mean) / torch.sqrt(
        (signal_dist.variance + noise_dist.variance) / 2
    )

    return d_prime
```

**Key insight**: Different attention mechanisms affect different aspects of detection!

---

## 7. Performance Implications

### 7.1 Computational Costs

**Attention mechanisms**:
- Standard attention: O(n²) in sequence length
- Linear attention: O(n) with approximations
- Local attention: O(n × window_size)

**Precision computation**:
- Heteroscedastic networks: Extra forward pass for variance
- Learned precision: Minimal cost (parallel to values)
- Dynamic precision: Amortized in attention computation

### 7.2 When to Use Which

**Use standard transformer attention when**:
- Sequence modeling (NLP, time series)
- Need global context
- Hardware supports efficient attention (Flash Attention, etc.)

**Use precision weighting when**:
- Uncertainty quantification needed
- Noisy data sources
- Active learning / exploration
- Predictive coding architectures

**Use neural gain modulation when**:
- Biologically-inspired models
- Need interpretable attention
- Feature-based tasks (not sequential)
- Vision tasks with spatial/feature attention

---

## 8. ARR-COC-0-1 Connections (10%)

### 8.1 Relevance AS Attention/Precision

**In ARR-COC dialogue system**:

```python
class DialogueAttentionPrecisionRelevance:
    """
    Relevance in dialogue = Attention/Precision over context

    User utterance → What's relevant from history?
    → Precision-weighted context retrieval
    → Attention over relevant segments
    → Generate response
    """

    def compute_relevance(self, query, dialogue_history):
        # Relevance = Attention weights!
        attention_scores = query @ dialogue_history.T
        relevance_weights = softmax(attention_scores)

        # Precision = Confidence in each context element
        precision = estimate_precision(dialogue_history)

        # Combined: Relevant AND confident
        combined_weights = relevance_weights * precision
        combined_weights = combined_weights / combined_weights.sum()

        relevant_context = combined_weights @ dialogue_history
        return relevant_context
```

### 8.2 Token Budget = Limited Resources

**Token allocation in LLMs = Attention under resource constraints**:

- Fixed context window = Limited resources
- Which dialogue turns get tokens? = Attention allocation
- How confident in each turn? = Precision weighting
- What's most relevant? = Salience computation

**This IS the same problem**:
- Neuroscience: Which stimuli to process?
- ML: Which tokens to attend to?
- Dialogue: Which context to include?

All solved by: **Dynamic gain control that amplifies relevant signals**

### 8.3 Future Extensions

**Adaptive context windows**:
```python
# Allocate more tokens to high-precision, high-salience content
token_allocation = salience * precision * available_tokens
```

**Multi-scale attention**:
```python
# Different timescales in dialogue
# - Immediate: Last 2 turns (high precision)
# - Recent: Last 10 turns (medium precision)
# - Background: Full history (low precision, high-level gist)
```

---

## 9. Code Examples: Complete Implementations

### 9.1 Minimal Attention-Precision Module

```python
class MinimalAttentionPrecision(nn.Module):
    """Simplest possible attention = precision implementation"""

    def __init__(self, d_model):
        super().__init__()
        self.scale = math.sqrt(d_model)

    def forward(self, query, key, value, return_precision=False):
        # Compute similarity (QK^T)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale (temperature)
        scores = scores / self.scale

        # Softmax = competition = limited resources
        precision_weights = F.softmax(scores, dim=-1)

        # Apply gain (multiply by values)
        output = torch.matmul(precision_weights, value)

        if return_precision:
            return output, precision_weights  # These ARE precision weights!
        return output
```

### 9.2 Complete Vision Attention with FSGM

```python
class VisionAttentionFSGM(nn.Module):
    """
    Vision attention using Feature Similarity Gain Model
    For object detection / classification with attention
    """

    def __init__(self, in_channels, num_classes, beta=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.beta = beta  # Gain modulation strength

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        # Class-specific feature templates (tuning curves!)
        self.class_templates = nn.Parameter(
            torch.randn(num_classes, 256)
        )

        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, attended_class=None):
        # Extract features
        features = self.features(x)  # (batch, 256, h, w)

        # Global average pool
        pooled = F.adaptive_avg_pool2d(features, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # (batch, 256)

        if attended_class is not None:
            # Apply FSGM: gain = 1 + β * similarity_to_attended
            attended_template = self.class_templates[attended_class]

            # Compute similarity (cosine)
            similarity = F.cosine_similarity(
                pooled.unsqueeze(1),
                attended_template.unsqueeze(0),
                dim=-1
            )

            # Apply gain modulation
            gain = 1.0 + self.beta * similarity.unsqueeze(-1)
            pooled = pooled * gain

        # Classify
        logits = self.classifier(pooled)

        return logits
```

---

## 10. The Train Station Summary

**Coffee cup = donut moment**: Attention, precision, salience, and relevance are **topologically equivalent** — they're all the same computation viewed from different angles!

**The core equation** (appears everywhere):

```
output = softmax(similarity / temperature) × value
```

**Where**:
- **Similarity**: QK^T (transformers), tuning curves (neuro), expected information (active inference)
- **Temperature**: √d_k (attention), inverse variance (precision), urgency (salience)
- **Value**: Information to modulate (always)
- **Softmax**: Competition for limited resources (always)

**Implementation checklist**:
- ✅ Multiplicative gain (not additive)
- ✅ Normalized weights (sum to 1, limited resources)
- ✅ Context-dependent (query-key interaction)
- ✅ Competitive (softmax or winner-take-all)
- ✅ Differentiable (for learning)

**Biological validation**:
- Attention modulates firing rates multiplicatively ✅
- Precision weights prediction errors ✅
- Salience network detects relevant stimuli ✅
- Effects propagate through hierarchy ✅

**Machine learning validation**:
- Transformers use this for SOTA performance ✅
- Predictive coding networks learn efficiently ✅
- Vision models improve with attention ✅
- All use same core mechanism ✅

**This is THE TRAIN STATION**: Where neuroscience, cognitive science, machine learning, and active inference all meet, using the exact same fundamental computation.

---

## Sources

**Source Documents:**
- None (this synthesis draws from web research)

**Web Research:**
- Lindsay, G. W. (2020). ["Attention in Psychology, Neuroscience, and Machine Learning"](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00029/full) - Frontiers in Computational Neuroscience (Accessed 2025-11-23)
  - Comprehensive review of attention across disciplines
  - Feature similarity gain model (FSGM)
  - Relationship between tuning and function

- Lindsay, G. W. & Miller, K. D. (2018). ["How biological attention mechanisms improve task performance in a large-scale visual system model"](https://elifesciences.org/articles/38105) - eLife (Accessed 2025-11-23)
  - FSGM in CNNs
  - Attention as multiplicative gain control
  - Signal detection theory with attention
  - Feature vs. spatial attention effects

- Fecteau, J. H. & Munoz, D. P. (2006). "Salience, relevance, and firing: a priority map for target selection" - Trends in Cognitive Sciences
  - Salience map as attention mechanism
  - Bottom-up vs. top-down salience

- Menon, V. (2010). "Saliency, switching, attention and control: a network model of insula function" - PMC
  - Salience network architecture
  - Attention switching mechanisms

**Additional References:**
- Treue, S. & Martinez-Trujillo, J. C. (1999). Feature-based attention influences motion processing gain in macaque visual cortex
- Reynolds, J. H. & Heeger, D. J. (2009). The normalization model of attention
- Vaswani, A. et al. (2017). "Attention is All You Need" - NIPS
- Friston, K. (2010). The free-energy principle: a unified brain theory?

---

**TRAIN STATION COORDINATES**:
- **Latitude**: Attention mechanisms (ML)
- **Longitude**: Precision weighting (Predictive Coding)
- **Altitude**: Gain control (Neuroscience)
- **Time**: Dynamic relevance (Active Inference)

**All trains arrive at the same station: Multiplicative gain control that amplifies task-relevant signals under resource constraints.**
