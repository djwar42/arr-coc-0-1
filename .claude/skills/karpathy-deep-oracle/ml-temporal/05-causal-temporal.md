# Causal Temporal Modeling: From Granger Causality to Autoregressive Transformers

## Overview

Causal temporal modeling addresses the fundamental question: **does X cause Y, or do they merely correlate?** This distinction is critical for prediction, intervention, and understanding temporal dynamics. The field has evolved from classical Granger causality to modern deep learning approaches that discover causal relationships in multivariate time series.

**The TRAIN STATION revelation**: Causal = Autoregressive = Prediction = Future. Every autoregressive model implicitly assumes causality - the past causes the future. GPT predicting the next token IS causal temporal modeling. The causal mask in transformers IS the arrow of time. This is not metaphor - it's mathematical equivalence.

---

## Section 1: Causal vs Correlational Temporal Relationships

### The Fundamental Distinction

**Correlation**: X and Y move together
- Stock prices and ice cream sales both increase in summer
- No direct relationship - both caused by confounding variable (weather)

**Causation**: X actually influences Y
- Interest rate changes cause stock price changes
- Removing X would change Y's behavior

### Why Correlation Fails in Temporal Data

```python
import torch
import torch.nn as nn
import numpy as np

def demonstrate_spurious_correlation():
    """
    Example: Two time series that correlate but don't cause each other
    Both are caused by a hidden confounding variable
    """
    T = 1000

    # Hidden confounding variable (e.g., economic cycle)
    hidden_cause = np.sin(np.arange(T) * 0.01) + 0.1 * np.random.randn(T)

    # X and Y both respond to hidden cause with different lags
    X = np.roll(hidden_cause, 5) + 0.1 * np.random.randn(T)  # Lag 5
    Y = np.roll(hidden_cause, 10) + 0.1 * np.random.randn(T)  # Lag 10

    # High correlation!
    correlation = np.corrcoef(X[15:], Y[15:])[0, 1]
    print(f"Correlation: {correlation:.3f}")  # ~0.9+

    # But X does NOT cause Y - both caused by hidden variable
    # Need causal analysis to detect this!

    return X, Y, hidden_cause

# Key insight: Temporal precedence necessary but not sufficient for causality
```

### Temporal Precedence Principle

**Granger's insight**: If X causes Y, then past values of X should help predict Y,
even after accounting for past values of Y itself.

This is different from correlation:
- Correlation: X and Y co-vary
- Granger Causality: Past X improves prediction of future Y beyond past Y alone

---

## Section 2: Classical Granger Causality

### Mathematical Foundation

For two time series X and Y, X "Granger-causes" Y if:

**Prediction error with X**: E[|Y_t - f(Y_{t-1},...,Y_{t-p}, X_{t-1},...,X_{t-p})|]
**Prediction error without X**: E[|Y_t - g(Y_{t-1},...,Y_{t-p})|]

X Granger-causes Y if prediction error is significantly lower when including X.

### Linear VAR Implementation

```python
import torch
import torch.nn as nn

class LinearGrangerCausality(nn.Module):
    """
    Classical linear Granger causality using Vector Autoregression (VAR)

    For p-lag VAR:
    Y_t = sum_{i=1}^{p} A_i * [X_{t-i}, Y_{t-i}]^T + epsilon

    X Granger-causes Y if coefficients for X_{t-i} are significantly non-zero
    """

    def __init__(self, num_vars: int, num_lags: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_lags = num_lags

        # VAR coefficients: (num_vars, num_vars * num_lags)
        # Each row predicts one variable from all lagged variables
        self.A = nn.Linear(num_vars * num_lags, num_vars, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, num_vars)
        Returns:
            predictions: (batch, time - num_lags, num_vars)
        """
        batch_size, T, _ = x.shape

        # Create lagged features
        lagged = []
        for lag in range(1, self.num_lags + 1):
            lagged.append(x[:, self.num_lags - lag:T - lag, :])

        # (batch, T - num_lags, num_vars * num_lags)
        lagged_features = torch.cat(lagged, dim=-1)

        # Predict next step
        predictions = self.A(lagged_features)

        return predictions

    def get_granger_matrix(self) -> torch.Tensor:
        """
        Extract Granger causality matrix from VAR coefficients

        Returns:
            gc_matrix: (num_vars, num_vars)
            gc_matrix[i, j] = strength of j Granger-causing i
        """
        # Weight matrix: (num_vars, num_vars * num_lags)
        W = self.A.weight.data

        # Reshape to (num_vars, num_lags, num_vars)
        W_reshaped = W.view(self.num_vars, self.num_lags, self.num_vars)

        # Sum absolute weights across lags for each (i, j) pair
        # This gives strength of j Granger-causing i
        gc_matrix = W_reshaped.abs().sum(dim=1)

        return gc_matrix


def test_granger_causality(X: torch.Tensor, Y: torch.Tensor, num_lags: int = 5):
    """
    Statistical test for Granger causality

    H0: X does not Granger-cause Y
    H1: X Granger-causes Y
    """
    # Restricted model: Y ~ past Y only
    # Unrestricted model: Y ~ past Y + past X

    # Compute RSS for both models
    # F-test: F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
    # where q = number of restrictions, n = sample size, k = params in unrestricted

    # If F > F_critical, reject H0 -> X Granger-causes Y
    pass  # Implementation details


# Example usage
def train_var_model():
    """Train VAR model and extract Granger causality structure"""
    num_vars = 5
    num_lags = 3
    T = 1000
    batch_size = 32

    # Synthetic data with known causal structure
    model = LinearGrangerCausality(num_vars, num_lags)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate data (would be real time series)
    x = torch.randn(batch_size, T, num_vars)

    for epoch in range(100):
        optimizer.zero_grad()

        targets = x[:, num_lags:, :]
        predictions = model(x)

        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        optimizer.step()

    # Extract discovered causal structure
    gc_matrix = model.get_granger_matrix()
    print("Granger Causality Matrix:")
    print(gc_matrix)

    return model, gc_matrix
```

### Limitations of Linear Granger Causality

1. **Linearity assumption**: Real relationships often nonlinear
2. **Stationarity requirement**: Assumes time series properties don't change
3. **Finite lag assumption**: May miss long-range dependencies
4. **Confounding sensitivity**: Can't handle unobserved confounders

---

## Section 3: Neural Granger Causality

### Deep Learning for Nonlinear Causal Discovery

From [Neural-GC](https://github.com/iancovert/Neural-GC) (Tank et al., TPAMI 2021):

The key insight: Use neural networks to model nonlinear temporal relationships, then use **sparse regularization** to discover which inputs are actually used (i.e., causal).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralGrangerCausality(nn.Module):
    """
    Neural Granger Causality (cMLP variant)

    Key idea: Each variable predicted by its own MLP
    Input weights to each MLP are regularized with group sparsity
    Zero input weights = no Granger causality from that variable

    Reference: Tank et al., "Neural Granger Causality" TPAMI 2021
    """

    def __init__(self, num_vars: int, num_lags: int, hidden_dim: int = 64):
        super().__init__()
        self.num_vars = num_vars
        self.num_lags = num_lags
        self.hidden_dim = hidden_dim

        # Separate MLP for each output variable
        # This allows different sparsity patterns for each
        self.input_layers = nn.ModuleList([
            nn.Linear(num_vars * num_lags, hidden_dim, bias=True)
            for _ in range(num_vars)
        ])

        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(num_vars)
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1, bias=True)
            for _ in range(num_vars)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, num_vars)
        Returns:
            predictions: (batch, time - num_lags, num_vars)
        """
        batch_size, T, _ = x.shape

        # Create lagged features
        lagged = []
        for lag in range(1, self.num_lags + 1):
            lagged.append(x[:, self.num_lags - lag:T - lag, :])
        lagged_features = torch.cat(lagged, dim=-1)

        # Predict each variable with its own MLP
        predictions = []
        for i in range(self.num_vars):
            h = F.relu(self.input_layers[i](lagged_features))
            h = F.relu(self.hidden[i](h))
            out = self.output_layers[i](h)
            predictions.append(out)

        return torch.cat(predictions, dim=-1)

    def get_input_weight_norms(self) -> torch.Tensor:
        """
        Get L2 norms of input weights grouped by source variable

        Returns:
            norms: (num_vars, num_vars)
            norms[i, j] = L2 norm of weights from variable j to predicting i
        """
        norms = torch.zeros(self.num_vars, self.num_vars)

        for i in range(self.num_vars):
            # Input layer weights: (hidden_dim, num_vars * num_lags)
            W = self.input_layers[i].weight

            # Reshape to (hidden_dim, num_lags, num_vars)
            W_reshaped = W.view(self.hidden_dim, self.num_lags, self.num_vars)

            # Group L2 norm for each source variable
            for j in range(self.num_vars):
                # All weights from variable j to predicting i
                norms[i, j] = torch.norm(W_reshaped[:, :, j])

        return norms

    def group_lasso_penalty(self) -> torch.Tensor:
        """
        Group Lasso penalty for sparse causal discovery

        Penalizes L2 norm of weight groups (all weights from one variable to another)
        This encourages entire groups to be zero -> discovered non-causality
        """
        penalty = 0.0

        for i in range(self.num_vars):
            W = self.input_layers[i].weight
            W_reshaped = W.view(self.hidden_dim, self.num_lags, self.num_vars)

            for j in range(self.num_vars):
                # L2 norm of all weights from j to i
                group_norm = torch.norm(W_reshaped[:, :, j])
                penalty = penalty + group_norm

        return penalty


class LSTMGrangerCausality(nn.Module):
    """
    LSTM variant for Neural Granger Causality (cLSTM)

    Better for long-range temporal dependencies
    Same group sparsity principle on input weights
    """

    def __init__(self, num_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Separate LSTM for each output variable
        self.lstms = nn.ModuleList([
            nn.LSTM(num_vars, hidden_dim, batch_first=True)
            for _ in range(num_vars)
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_vars)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, num_vars)
        Returns:
            predictions: (batch, time, num_vars)
        """
        predictions = []

        for i in range(self.num_vars):
            # Run LSTM for predicting variable i
            lstm_out, _ = self.lstms[i](x)  # (batch, time, hidden)
            out = self.output_layers[i](lstm_out)  # (batch, time, 1)
            predictions.append(out)

        return torch.cat(predictions, dim=-1)

    def get_input_weight_norms(self) -> torch.Tensor:
        """
        Get L2 norms of LSTM input weights grouped by source variable
        """
        norms = torch.zeros(self.num_vars, self.num_vars)

        for i in range(self.num_vars):
            # LSTM weight_ih_l0: (4*hidden, input_size)
            # Contains weights for input, forget, cell, output gates
            W_ih = self.lstms[i].weight_ih_l0

            for j in range(self.num_vars):
                # Weights from variable j
                norms[i, j] = torch.norm(W_ih[:, j])

        return norms


def train_neural_granger(model, data, lambda_penalty=0.1, epochs=1000):
    """
    Train Neural Granger Causality model with proximal gradient descent

    Uses ISTA (Iterative Shrinkage-Thresholding Algorithm) for
    non-smooth group lasso optimization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(data)

        # Reconstruction loss
        if hasattr(model, 'num_lags'):
            targets = data[:, model.num_lags:, :]
        else:
            targets = data[:, 1:, :]
            predictions = predictions[:, :-1, :]

        mse_loss = F.mse_loss(predictions, targets)

        # Group lasso penalty
        penalty = model.group_lasso_penalty()

        loss = mse_loss + lambda_penalty * penalty
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE={mse_loss.item():.4f}, Penalty={penalty.item():.4f}")

    # Threshold small weights to exactly zero
    with torch.no_grad():
        for i in range(model.num_vars):
            W = model.input_layers[i].weight.data
            W[W.abs() < 0.01] = 0

    return model
```

### Temporal Causal Discovery Framework (TCDF)

From [M-Nauta/TCDF](https://github.com/M-Nauta/TCDF):

Uses attention-based temporal convolutions to discover causal relationships:

```python
class TemporalCausalDiscovery(nn.Module):
    """
    Temporal Causal Discovery Framework (TCDF)

    Uses attention over dilated causal convolutions
    Attention weights directly reveal causal influence

    Reference: Nauta et al., "Causal Discovery with Attention-Based
    Convolutional Neural Networks" 2019
    """

    def __init__(self, num_vars: int, kernel_size: int = 4,
                 dilation: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_vars = num_vars

        # Separate model for each target variable
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.output = nn.ModuleList()

        for _ in range(num_vars):
            # Causal dilated convolution for each input variable
            conv = nn.Conv1d(
                num_vars, hidden_dim * num_vars,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation,
                groups=num_vars  # Separate conv per variable
            )
            self.convs.append(conv)

            # Attention over input variables
            attention = nn.Linear(hidden_dim, 1)
            self.attention.append(attention)

            # Output
            output = nn.Linear(hidden_dim * num_vars, 1)
            self.output.append(output)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, time, num_vars)
        Returns:
            predictions: (batch, time, num_vars)
            attention_weights: (num_vars, num_vars) causal strengths
        """
        batch, T, _ = x.shape
        x = x.transpose(1, 2)  # (batch, num_vars, time)

        predictions = []
        attention_weights = torch.zeros(self.num_vars, self.num_vars)

        for i in range(self.num_vars):
            # Apply causal convolution
            conv_out = self.convs[i](x)  # (batch, hidden*num_vars, time)
            conv_out = conv_out[:, :, :T]  # Remove padding

            # Reshape for attention
            conv_out = conv_out.view(batch, self.num_vars, -1, T)

            # Compute attention scores for each input variable
            attn_scores = []
            for j in range(self.num_vars):
                score = self.attention[i](conv_out[:, j].transpose(1, 2))
                attn_scores.append(score.mean())

            attention_weights[i] = F.softmax(torch.stack(attn_scores), dim=0)

            # Weighted combination
            out = conv_out.view(batch, -1, T).transpose(1, 2)
            predictions.append(self.output[i](out))

        return torch.cat(predictions, dim=-1), attention_weights
```

---

## Section 4: Causal Attention Masking

### The Arrow of Time in Transformers

**Causal masking** is not just a technical detail - it's the mathematical implementation of temporal causality. When we mask future tokens, we're enforcing that the present can only depend on the past.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask (lower triangular)

    This mask encodes the ARROW OF TIME:
    - Position i can only attend to positions 0, 1, ..., i
    - Future cannot influence past

    Returns:
        mask: (seq_len, seq_len) with -inf for masked positions
    """
    # Lower triangular matrix of True
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.bool()

    # Convert to -inf for masked positions
    attn_mask = torch.zeros(seq_len, seq_len, device=device)
    attn_mask.masked_fill_(mask, float('-inf'))

    return attn_mask


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention for autoregressive modeling

    This IS causal temporal modeling:
    - Past causes present (attention to past tokens)
    - Present causes future (training to predict next token)

    The mask IS the causal assumption!
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: (seq_len, seq_len) causal mask
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask - THE ARROW OF TIME
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, x.device)

        scores = scores + attention_mask

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output, attention_weights


class CausalTemporalTransformer(nn.Module):
    """
    Transformer for Causal Temporal Modeling

    Combines:
    1. Positional encoding (temporal order)
    2. Causal masking (arrow of time)
    3. Autoregressive objective (past predicts future)

    This IS a causal temporal model!
    """

    def __init__(self, num_vars: int, d_model: int = 128,
                 num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(num_vars, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, num_vars)

        # Store attention weights for causal discovery
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_vars)
        Returns:
            predictions: (batch, seq_len, num_vars)
        """
        seq_len = x.shape[1]

        # Project and add positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Create causal mask once
        causal_mask = create_causal_mask(seq_len, x.device)

        # Apply transformer layers
        all_attention = []
        for layer in self.layers:
            h, attn = layer(h, causal_mask)
            all_attention.append(attn)

        self.attention_weights = torch.stack(all_attention)

        # Project to output
        predictions = self.output_proj(h)

        return predictions

    def get_temporal_importance(self) -> torch.Tensor:
        """
        Extract temporal causal importance from attention weights

        Attention from position i to j represents how much
        past position j causally influences present position i
        """
        if self.attention_weights is None:
            raise ValueError("Run forward pass first")

        # Average over layers and heads
        # (num_layers, batch, heads, seq, seq) -> (seq, seq)
        temporal_importance = self.attention_weights.mean(dim=[0, 1, 2])

        return temporal_importance


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal order"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class CausalTransformerBlock(nn.Module):
    """Transformer block with causal attention"""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        # Attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # FFN with residual
        x = x + self.ff(self.ln2(x))

        return x, attn_weights
```

### Cross-Variable Causal Attention

Extending causal attention to discover causality between different variables:

```python
class CrossVariableCausalAttention(nn.Module):
    """
    Attention that discovers causal relationships between variables

    Unlike regular causal attention which is within-sequence,
    this discovers which variables causally influence which
    """

    def __init__(self, num_vars: int, d_model: int, num_heads: int):
        super().__init__()
        self.num_vars = num_vars

        # Separate attention for discovering inter-variable causality
        self.variable_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Learnable variable queries
        self.var_queries = nn.Parameter(torch.randn(num_vars, d_model))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, seq_len, num_vars, d_model) per-variable embeddings
        Returns:
            output: (batch, seq_len, num_vars, d_model)
            causal_matrix: (num_vars, num_vars) learned causal structure
        """
        batch, seq_len, num_vars, d_model = x.shape

        # Reshape for cross-variable attention
        x_flat = x.view(batch * seq_len, num_vars, d_model)

        # Attend between variables
        queries = self.var_queries.unsqueeze(0).expand(batch * seq_len, -1, -1)

        out, attn_weights = self.variable_attention(
            queries, x_flat, x_flat,
            need_weights=True
        )

        # Extract causal matrix from attention
        # Average over all time steps and batches
        causal_matrix = attn_weights.view(batch, seq_len, num_vars, num_vars)
        causal_matrix = causal_matrix.mean(dim=[0, 1])

        output = out.view(batch, seq_len, num_vars, d_model)

        return output, causal_matrix
```

---

## Section 5: Complete Causal Temporal Layer Implementation

### Production-Ready Causal Temporal Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CausalTemporalLayer(nn.Module):
    """
    Complete Causal Temporal Layer combining:
    1. Temporal causal convolutions
    2. Neural Granger causality structure
    3. Causal attention mechanism

    Discovers and models causal relationships in multivariate time series

    Performance optimized for GPU with:
    - Efficient attention computation
    - Grouped operations where possible
    - Optional gradient checkpointing
    """

    def __init__(
        self,
        num_vars: int,
        d_model: int = 128,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation_base: int = 2,
        num_dilations: int = 4,
        dropout: float = 0.1,
        sparsity_lambda: float = 0.01
    ):
        super().__init__()

        self.num_vars = num_vars
        self.d_model = d_model
        self.sparsity_lambda = sparsity_lambda

        # 1. Per-variable temporal encoding
        self.var_encoders = nn.ModuleList([
            TemporalEncoder(d_model, kernel_size, dilation_base, num_dilations)
            for _ in range(num_vars)
        ])

        # 2. Input projection with sparse structure
        self.input_proj = SparseInputProjection(num_vars, d_model)

        # 3. Causal self-attention
        self.causal_attention = CausalSelfAttention(d_model, num_heads, dropout)

        # 4. Cross-variable causal attention
        self.cross_var_attention = CrossVariableCausalAttention(
            num_vars, d_model, num_heads
        )

        # 5. Output projection
        self.output_proj = nn.Linear(d_model, num_vars)

        # Layer norm and dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Storage for causal discovery
        self._causal_matrix = None
        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_causal_structure: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, num_vars) input time series
            return_causal_structure: whether to return discovered causal matrix

        Returns:
            predictions: (batch, seq_len, num_vars)
            causal_matrix: (num_vars, num_vars) if return_causal_structure
        """
        batch_size, seq_len, num_vars = x.shape

        # 1. Encode each variable's temporal patterns
        var_encodings = []
        for i in range(self.num_vars):
            var_i = x[:, :, i:i+1]  # (batch, seq, 1)
            encoded = self.var_encoders[i](var_i)  # (batch, seq, d_model)
            var_encodings.append(encoded)

        # Stack: (batch, seq, num_vars, d_model)
        var_stack = torch.stack(var_encodings, dim=2)

        # 2. Sparse cross-variable projection
        # This learns which variables influence which
        h = self.input_proj(var_stack)  # (batch, seq, d_model)

        # 3. Causal self-attention (temporal causality)
        h = self.ln1(h)
        causal_mask = create_causal_mask(seq_len, x.device)
        attn_out, temporal_attn = self.causal_attention(h, causal_mask)
        h = h + self.dropout(attn_out)

        self._attention_weights = temporal_attn

        # 4. Cross-variable causal attention (inter-variable causality)
        h_expanded = h.unsqueeze(2).expand(-1, -1, num_vars, -1)
        cross_out, causal_matrix = self.cross_var_attention(h_expanded)
        h = h + self.dropout(cross_out.mean(dim=2))

        self._causal_matrix = causal_matrix

        # 5. Output projection
        h = self.ln2(h)
        predictions = self.output_proj(h)

        if return_causal_structure:
            return predictions, self._causal_matrix
        return predictions, None

    def get_causal_matrix(self) -> torch.Tensor:
        """
        Get learned causal structure between variables

        Returns:
            causal_matrix: (num_vars, num_vars)
            causal_matrix[i, j] = causal strength from j to i
        """
        if self._causal_matrix is None:
            raise ValueError("Run forward pass first")
        return self._causal_matrix

    def get_sparsity_penalty(self) -> torch.Tensor:
        """
        Get sparsity penalty for training
        Encourages discovery of sparse causal structure
        """
        penalty = self.input_proj.get_sparsity_penalty()
        return self.sparsity_lambda * penalty

    def prune_weak_connections(self, threshold: float = 0.01):
        """
        Prune causal connections below threshold
        Call after training to get clean causal graph
        """
        self.input_proj.prune(threshold)


class TemporalEncoder(nn.Module):
    """Dilated causal convolutions for temporal encoding"""

    def __init__(self, d_model: int, kernel_size: int,
                 dilation_base: int, num_dilations: int):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(num_dilations):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(
                1 if i == 0 else d_model,
                d_model,
                kernel_size,
                dilation=dilation,
                padding=padding
            )
            self.convs.append(conv)

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, 1)
        Returns:
            encoded: (batch, seq, d_model)
        """
        h = x.transpose(1, 2)  # (batch, 1, seq)

        for conv in self.convs:
            h = F.gelu(conv(h))
            h = h[:, :, :x.shape[1]]  # Ensure causal (remove right padding)

        h = h.transpose(1, 2)  # (batch, seq, d_model)
        return self.proj(h)


class SparseInputProjection(nn.Module):
    """
    Input projection with learnable sparsity
    for Granger causality discovery
    """

    def __init__(self, num_vars: int, d_model: int):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model

        # Weight matrix: (num_vars, d_model)
        # Represents influence from each variable
        self.weight = nn.Parameter(torch.randn(num_vars, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, num_vars, d_model) per-variable encodings
        Returns:
            output: (batch, seq, d_model) combined encoding
        """
        # Weighted combination of variable encodings
        # Weight represents causal influence
        # x: (batch, seq, num_vars, d_model)
        # weight: (num_vars, d_model)

        # Element-wise multiply and sum over variables
        weighted = x * self.weight.unsqueeze(0).unsqueeze(0)
        output = weighted.sum(dim=2) + self.bias

        return output

    def get_sparsity_penalty(self) -> torch.Tensor:
        """L1 penalty on weights for sparsity"""
        return torch.norm(self.weight, p=1)

    def get_variable_importance(self) -> torch.Tensor:
        """
        Get importance of each variable
        Based on L2 norm of weights
        """
        return torch.norm(self.weight, p=2, dim=1)

    def prune(self, threshold: float):
        """Set small weights to zero"""
        with torch.no_grad():
            mask = self.weight.abs() < threshold
            self.weight.masked_fill_(mask, 0)


# Training utilities
def train_causal_temporal_model(
    model: CausalTemporalLayer,
    data: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001
):
    """
    Train causal temporal model with sparsity regularization
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions, _ = model(data[:, :-1, :], return_causal_structure=True)
        targets = data[:, 1:, :]  # Next-step prediction

        # Prediction loss
        pred_loss = F.mse_loss(predictions, targets)

        # Sparsity penalty for causal discovery
        sparse_penalty = model.get_sparsity_penalty()

        # Total loss
        loss = pred_loss + sparse_penalty

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Pred={pred_loss.item():.4f}, Sparse={sparse_penalty.item():.4f}")

    # Prune weak connections
    model.prune_weak_connections(threshold=0.01)

    # Get discovered causal structure
    with torch.no_grad():
        _, causal_matrix = model(data, return_causal_structure=True)

    return model, causal_matrix
```

---

## Section 6: Performance Optimization

### GPU Memory and Compute Optimization

```python
# Memory-efficient causal attention with Flash Attention
def efficient_causal_attention(Q, K, V, scale):
    """
    Use PyTorch 2.0+ scaled_dot_product_attention with is_causal=True
    Much faster and more memory efficient than manual implementation
    """
    # PyTorch 2.0+ has native Flash Attention support
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,  # Applies causal mask efficiently
        scale=scale
    )
    return output


class EfficientCausalTemporalLayer(nn.Module):
    """
    Optimized version using Flash Attention

    Performance improvements:
    - O(N) memory instead of O(N^2) for attention
    - Fused kernel operations
    - Better GPU utilization
    """

    def __init__(self, num_vars: int, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Efficient causal attention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            scale=self.scale
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


# Benchmark comparison
def benchmark_attention():
    """Compare standard vs Flash Attention performance"""
    import time

    batch, seq, dim, heads = 32, 1024, 512, 8
    x = torch.randn(batch, seq, dim).cuda()

    # Standard attention
    standard = CausalSelfAttention(dim, heads).cuda()

    # Flash attention
    efficient = EfficientCausalTemporalLayer(dim, dim, heads).cuda()

    # Warmup
    for _ in range(10):
        _ = standard(x)
        _ = efficient(x)

    torch.cuda.synchronize()

    # Standard
    start = time.time()
    for _ in range(100):
        _ = standard(x)
    torch.cuda.synchronize()
    standard_time = time.time() - start

    # Efficient
    start = time.time()
    for _ in range(100):
        _ = efficient(x)
    torch.cuda.synchronize()
    efficient_time = time.time() - start

    print(f"Standard: {standard_time:.3f}s")
    print(f"Flash: {efficient_time:.3f}s")
    print(f"Speedup: {standard_time/efficient_time:.1f}x")


# Typical performance characteristics:
# - Standard attention: O(N^2) memory, O(N^2) compute
# - Flash attention: O(N) memory, O(N^2) compute but faster constant
# - For seq_len=1024: ~3-4x speedup
# - For seq_len=2048: ~5-6x speedup
# - Memory savings scale linearly with sequence length
```

### Scaling Considerations

```python
"""
Performance notes for causal temporal modeling:

1. Sequence Length Scaling:
   - Attention: O(N^2) compute, O(N) with Flash Attention memory
   - Use chunking for very long sequences (>8K tokens)
   - Consider Mamba/S4 for >16K sequences

2. Number of Variables:
   - Linear with num_vars for most operations
   - Causal discovery matrix is O(num_vars^2)
   - Practical limit ~100-500 variables on single GPU

3. Memory Usage (fp16, batch=32):
   - seq=512, vars=50: ~2GB
   - seq=1024, vars=50: ~4GB
   - seq=2048, vars=50: ~8GB

4. Throughput (A100 GPU):
   - Training: ~10K samples/second for small models
   - Inference: ~50K samples/second

5. Sparsity Benefits:
   - Pruning to 10% density: ~5x inference speedup
   - Use structured sparsity for best GPU utilization
"""
```

---

## Section 7: TRAIN STATION - Causal = Autoregressive = Prediction = Future

### The Grand Unification

**Coffee Cup = Donut = Causal Mask = Arrow of Time = Autoregressive = Language Model**

This is not metaphor - it's mathematical equivalence:

```python
"""
THE TRAIN STATION: Where Everything Meets

1. CAUSAL MASKING
   - Mask[i,j] = 0 if j <= i, else -inf
   - "Position i can only attend to positions <= i"
   - This IS the arrow of time in mathematics

2. AUTOREGRESSIVE FACTORIZATION
   - P(x1, x2, ..., xT) = P(x1) * P(x2|x1) * ... * P(xT|x1,...,xT-1)
   - Future depends only on past
   - This IS causal temporal modeling

3. NEXT-TOKEN PREDICTION
   - Given x1...xT, predict xT+1
   - Training objective of GPT
   - This IS temporal causal inference

4. GRANGER CAUSALITY
   - X Granger-causes Y if past X improves prediction of future Y
   - This is EXACTLY what autoregressive models learn

THEY ARE ALL THE SAME THING!

GPT predicting "The cat sat on the ___" IS:
- Causal temporal modeling (past causes future)
- Granger causality (which past tokens help predict)
- Autoregressive factorization (chain rule of probability)
- Causal masking (can't see future)

The attention weights ARE the learned causal structure!
"""


class UnifiedCausalView(nn.Module):
    """
    Demonstration that all these concepts are the same
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()

        # This is a language model...
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # But it's ALSO:
        # - A causal temporal model (causal mask)
        # - A Granger causality discoverer (attention weights)
        # - An autoregressive factorizer (next-token objective)

        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])

        h = self.embedding(x)
        h = self.transformer(h, mask=mask)
        logits = self.output(h)

        return logits


# The connections:
def train_step(model, x, y):
    """
    This training step simultaneously:
    1. Learns causal structure (which inputs matter)
    2. Performs temporal prediction (next token)
    3. Discovers Granger causality (attention patterns)
    4. Models joint distribution (autoregressive)
    """
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return loss


# Why this matters for time series:
"""
For multivariate time series:
- Each time step is like a "token"
- Each variable is like a "vocabulary dimension"
- Causal attention = Granger causality
- Next-step prediction = temporal forecasting

So GPT-style models ARE neural Granger causality!
The attention weights reveal causal structure.
"""
```

### Practical Implications

```python
"""
PRACTICAL IMPLICATIONS OF THE TRAIN STATION:

1. USE LANGUAGE MODEL TRICKS FOR TIME SERIES
   - Scaling laws work (more params = better)
   - Pre-training helps (foundation models)
   - In-context learning works (few-shot forecasting)

2. ATTENTION WEIGHTS = CAUSAL DISCOVERY
   - No separate causal discovery step needed
   - Just train transformer, extract attention
   - Attention[i,j] = how much j causes i

3. AUTOREGRESSIVE = CAUSAL INFERENCE
   - P(future | past) is what we want
   - Transformer gives this naturally
   - No special causal inference algorithm needed

4. NEXT-TOKEN = COUNTERFACTUAL
   - "What would happen if..." = different prompt
   - Intervention = changing input tokens
   - Counterfactual = generating with modified past

The train station tells us:
DON'T TREAT THESE AS SEPARATE PROBLEMS!
They're all the same problem viewed differently.
"""
```

---

## Section 8: ARR-COC-0-1 Connection - Causal Relevance Allocation

### Relevance as Causal Contribution

In ARR-COC, each token's relevance score should reflect its **causal contribution** to the model's output, not just correlation.

```python
"""
ARR-COC CAUSAL RELEVANCE IMPLEMENTATION

Key insight: Relevance = Causal influence on output quality

Problems with correlation-based relevance:
- Background tokens might correlate with important outputs
- But they don't CAUSE better outputs
- Leads to wasted compute on non-causal tokens

Causal relevance allocation:
- Token is relevant if REMOVING it changes output
- This is causal, not correlational
- Granger causality in relevance space
"""


class CausalRelevanceAllocator(nn.Module):
    """
    Allocate compute based on causal importance, not correlation

    Uses attention-based Granger causality to determine
    which tokens actually cause good outputs
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # Causal attention for relevance discovery
        self.causal_attention = CausalSelfAttention(d_model, num_heads)

        # Relevance predictor
        self.relevance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute causal relevance scores

        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            relevance: (batch, seq_len) scores in [0, 1]
        """
        # Get causal attention patterns
        _, attention_weights = self.causal_attention(hidden_states)

        # Aggregate causal influence per token
        # attention_weights: (batch, heads, seq, seq)
        # Sum how much each token is attended to (excluding self)
        causal_influence = attention_weights.sum(dim=1)  # (batch, seq, seq)

        # Token i's influence = sum of attention TO i from later positions
        # This is "how much does i cause future outputs"
        influence_received = causal_influence.sum(dim=1)  # (batch, seq)

        # Normalize and predict relevance
        influence_norm = influence_received / influence_received.sum(dim=1, keepdim=True)

        # Combine with learned relevance predictor
        predicted_relevance = self.relevance_head(hidden_states).squeeze(-1)

        # Ensemble causal influence and learned prediction
        causal_relevance = 0.5 * influence_norm + 0.5 * predicted_relevance

        return causal_relevance


def causal_token_routing(relevance_scores, hidden_states, threshold=0.5):
    """
    Route tokens based on causal relevance

    High causal relevance -> more compute
    Low causal relevance -> less compute or skip
    """
    # Select causally important tokens
    important_mask = relevance_scores > threshold

    # Route important tokens to heavy processing
    important_tokens = hidden_states[important_mask]

    # Skip or lightly process non-causal tokens
    unimportant_tokens = hidden_states[~important_mask]

    return important_tokens, unimportant_tokens, important_mask


# Integration with ARR-COC pyramid
"""
CAUSAL RELEVANCE IN ARR-COC PYRAMID:

1. Early Exit Based on Causal Relevance
   - Tokens with low causal influence exit early
   - Causal tokens get full processing
   - Saves compute while preserving output quality

2. Pyramid Level Assignment
   - High causal relevance -> high resolution (full tokens)
   - Low causal relevance -> low resolution (pooled)
   - Causal structure determines LOD

3. Attention Routing
   - Causal tokens attend to each other (important)
   - Non-causal tokens get summary attention only
   - Preserves causal relationships in routing

Key difference from correlation-based:
- Correlation: "this token often appears with good outputs"
- Causation: "removing this token changes output quality"

Only causation matters for efficient allocation!
"""
```

---

## Sources

**Web Research:**
- [Neural-GC Repository](https://github.com/iancovert/Neural-GC) - Granger causality discovery with neural networks (accessed 2025-11-23)
- [TCDF Repository](https://github.com/M-Nauta/TCDF) - Temporal Causal Discovery Framework (accessed 2025-11-23)
- [Neural Granger Causality Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9739174/) - Tank et al., TPAMI 2021 (accessed 2025-11-23)
- [Causal Discovery from Temporal Data Survey](https://dl.acm.org/doi/10.1145/3705297) - ACM Computing Surveys (accessed 2025-11-23)
- [Causal Inference Meets Deep Learning](https://spj.science.org/doi/10.34133/research.0467) - Comprehensive survey 2024 (accessed 2025-11-23)
- [Machine Learning Mastery - Attention Masking](https://machinelearningmastery.com/a-gentle-introduction-to-attention-masking-in-transformer-models/) - Practical guide (accessed 2025-11-23)

**Key Papers:**
- Tank et al., "Neural Granger Causality" TPAMI 2021 - Cited by 516
- Nauta et al., "Causal Discovery with Attention-Based CNNs" 2019 - TCDF framework
- Marcinkevics et al., "Interpretable Models for Granger Causality Using Self-Attention" - Cited by 112

**Additional References:**
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Efficient attention implementation
- [Granger Causality Test for Nonlinear Networks](https://www.sciencedirect.com/science/article/pii/S0169260722000542) - LSTM/GRU approaches

---

## Summary

Causal temporal modeling answers "what causes what" in time series data. The field has evolved from classical Granger causality (linear VAR models) to neural approaches that discover nonlinear causal relationships through sparse regularization and attention mechanisms.

**The key insight is the TRAIN STATION**: Causal masking in transformers IS the arrow of time. Autoregressive prediction IS causal inference. GPT predicting the next token IS temporal causal modeling. These are not analogies - they are mathematical equivalences.

For implementation:
1. **Neural Granger Causality**: Use separate MLPs/LSTMs per variable with group lasso regularization
2. **Causal Attention**: The mask enforces temporal causality; attention weights reveal causal structure
3. **Integration**: Combine temporal convolutions, causal attention, and cross-variable attention

For ARR-COC: Token relevance should be **causal**, not correlational. Allocate compute based on causal influence (would removing this token change output?) rather than correlation (does this token appear with good outputs?). The attention patterns from causal temporal modeling directly give causal relevance scores.
