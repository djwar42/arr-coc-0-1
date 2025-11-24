# State Space Models: S4, Mamba, and Efficient Sequence Modeling

## Overview

State Space Models (SSMs) represent a paradigm shift in sequence modeling, offering linear-time complexity while maintaining the ability to capture long-range dependencies. Unlike transformers with their quadratic attention bottleneck, SSMs process sequences efficiently through continuous-time dynamics discretized for computation, enabling processing of sequences with millions of tokens.

**Key Innovation**: SSMs compress the entire history into a fixed-size state, making them O(n) in time and O(1) in space per step, compared to transformers' O(n^2) time and O(n) space.

---

## Section 1: State Space Fundamentals

### Continuous-Time State Space Model

The classical SSM originates from control theory and describes a system through differential equations:

```
h'(t) = A*h(t) + B*x(t)    # State evolution (Equation 1a)
y(t) = C*h(t) + D*x(t)      # Output mapping (Equation 1b)
```

Where:
- **h(t)**: Hidden state (N-dimensional) - compressed representation of history
- **x(t)**: Input signal (1-dimensional per channel)
- **y(t)**: Output signal
- **A**: State transition matrix (N x N) - "How to evolve/forget the state"
- **B**: Input projection matrix (N x 1) - "What to remember from input"
- **C**: Output projection matrix (1 x N) - "How to read the state"
- **D**: Skip connection (1 x 1) - Direct input-to-output pathway

### Discretization: Continuous to Discrete

Real-world sequences are discrete (tokens, samples). We discretize using Zero-Order Hold (ZOH):

```python
# Zero-Order Hold discretization
A_bar = exp(delta * A)
B_bar = (A^-1) * (A_bar - I) * B

# Resulting discrete equations
h[k] = A_bar * h[k-1] + B_bar * x[k]
y[k] = C * h[k] + D * x[k]
```

Where **delta** is the step size/dwell time - larger delta means more focus on current input.

### Three Equivalent Representations

SSMs have a beautiful duality - three ways to compute the same thing:

**1. Recurrent Mode** (O(n) sequential, good for inference):
```python
h[k] = A_bar * h[k-1] + B_bar * x[k]
y[k] = C * h[k]
```

**2. Convolutional Mode** (O(n log n) parallel, good for training):
```python
# Kernel: K = [C*B, C*A*B, C*A^2*B, ..., C*A^(L-1)*B]
y = conv1d(x, K)  # FFT-based fast convolution
```

**3. Attention-like Mode** (for analysis):
```python
# Can be written as a structured attention matrix
y = M @ x  # Where M is a structured matrix
```

### Why States Work: Compression of History

From [Mamba Explained](https://thegradient.pub/mamba-explained/) (accessed 2025-11-23):

> "The efficiency vs. effectiveness tradeoff of sequence models is characterized by how well they compress their state: efficient models must have a small state, while effective models must have a state that contains all necessary information from the context."

The state is a **lossy compression** of the past - like JPEG for images. The key insight is that this compression can be sufficient for prediction if done intelligently (selectively).

---

## Section 2: S4 - Structured State Spaces

### The HiPPO Framework

S4 builds on HiPPO (High-Order Polynomial Projection Operator), which provides a principled way to initialize the A matrix for optimal memory of continuous signals.

From [Hazy Research S4 Blog](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1) (accessed 2025-11-23):

> "The Structured State Space (S4) is a new sequence model based on the state space model that is continuous-time in nature, excels at modeling long dependencies, and is very computationally efficient."

**HiPPO-LegS (Legendre)**: Optimal for bounded memory
```python
# A matrix initialization for memorizing recent history
A[n,k] = -sqrt(2n+1) * sqrt(2k+1) if n > k else
         -sqrt(2n+1) * sqrt(2k+1) if n == k else
         sqrt(2n+1) * sqrt(2k+1)
```

### The S4 Parameterization

The computational bottleneck: A is N x N, computing A^L for kernel is O(N^2 * L).

**S4's Solution**: Diagonal Plus Low-Rank (DPLR) structure
```python
A = V * Lambda * V^(-1) - P * Q^T
```

Where Lambda is diagonal. This enables O(N + L) computation via:
1. Cauchy kernel computation in frequency domain
2. FFT-based convolution

### S4D: Diagonal State Spaces

Simplified version using purely diagonal A:

```python
class S4D(nn.Module):
    def __init__(self, d_model, d_state, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize diagonal A matrix (log-space for stability)
        log_A_real = torch.log(torch.linspace(dt_min, dt_max, d_state))
        self.log_A_real = nn.Parameter(log_A_real)

        # B, C, D parameters
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        # Learnable step size
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.01)))

    def forward(self, x):
        """
        x: (batch, length, d_model)
        """
        batch, length, _ = x.shape

        # Get discretized parameters
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real)  # Negative for stability

        # Zero-order hold discretization
        A_bar = torch.exp(dt * A)  # (d_state,)
        B_bar = (1 - A_bar) / (-A) * dt  # Simplified for diagonal A

        # Compute convolution kernel
        # K[i] = C @ (A_bar^i * B_bar)
        powers = torch.arange(length, device=x.device).unsqueeze(1)  # (L, 1)
        kernel = (A_bar.unsqueeze(0) ** powers)  # (L, d_state)
        kernel = kernel * B_bar.unsqueeze(0)  # (L, d_state)
        K = (kernel @ self.C.T).T  # (d_model, L)

        # Convolutional mode - FFT for efficiency
        x_f = torch.fft.rfft(x.transpose(1, 2), n=2*length)  # (batch, d_model, L)
        K_f = torch.fft.rfft(K, n=2*length)  # (d_model, L)
        y = torch.fft.irfft(x_f * K_f, n=2*length)[:, :, :length]

        # Add skip connection
        y = y.transpose(1, 2) + x * self.D

        return y
```

### Performance Characteristics

From S4 paper (arXiv:2111.00396):
- **Path-X benchmark**: First model to solve (16K sequence length)
- **Long Range Arena**: State-of-the-art on all 6 tasks
- **Speed**: Comparable to transformers on training, faster on inference

---

## Section 3: Mamba - Selective State Spaces

### The Selectivity Problem

Standard SSMs apply the same A, B matrices to every input - no context dependence. This limits their effectiveness compared to attention.

From [Mamba paper](https://arxiv.org/abs/2312.00752) (arXiv:2312.00752, Gu & Dao):

> "A fundamental principle for building sequence models is selectivity: or the context-aware ability to focus on or filter out inputs into a sequential state."

### Selective State Space Model (S6)

Mamba makes B, C, and delta **input-dependent**:

```python
class SelectiveSSM(nn.Module):
    """Mamba's Selective State Space Model"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection (expand dimension)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # Selection parameters - these depend on input!
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A parameter (not input-dependent, diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, length, d_model)
        """
        batch, length, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (batch, length, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, length, d_inner)

        # Short convolution
        x = x.transpose(1, 2)  # (batch, d_inner, length)
        x = self.conv1d(x)[:, :, :length]
        x = x.transpose(1, 2)  # (batch, length, d_inner)
        x = F.silu(x)

        # Compute input-dependent B, C, dt
        x_dbl = self.x_proj(x)  # (batch, length, d_state*2 + 1)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt)  # Ensure positive step size

        # Get A
        A = -torch.exp(self.A_log)  # (d_state,)

        # Selective scan (recurrent mode for simplicity)
        y = self.selective_scan(x, dt.squeeze(-1), A, B, C)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def selective_scan(self, x, dt, A, B, C):
        """
        Selective scan with input-dependent B, C, dt

        x: (batch, length, d_inner)
        dt: (batch, length)
        A: (d_state,)
        B: (batch, length, d_state)
        C: (batch, length, d_state)
        """
        batch, length, d_inner = x.shape
        d_state = A.shape[0]

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device)

        outputs = []
        for t in range(length):
            # Discretize A and B for this timestep
            dt_t = dt[:, t:t+1, None]  # (batch, 1, 1)
            A_bar = torch.exp(dt_t * A)  # (batch, 1, d_state)
            B_bar = dt_t * B[:, t:t+1, :]  # (batch, 1, d_state)

            # State update: h = A_bar * h + B_bar * x
            x_t = x[:, t:t+1, :, None]  # (batch, 1, d_inner, 1)
            h = A_bar.unsqueeze(2) * h + B_bar.unsqueeze(2) * x_t.squeeze(1)

            # Output: y = C * h
            C_t = C[:, t, :].unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)  # (batch, d_inner)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, length, d_inner)

        # Add skip connection
        y = y + x * self.D

        return y
```

### Hardware-Aware Selective Scan

The selective scan can't use FFT (matrices are input-dependent). Mamba uses custom CUDA kernels with:
- **Kernel fusion**: Avoid memory I/O
- **Recomputation**: Trade compute for memory
- **Block-wise parallel scan**: GPU-efficient

Result: 5x faster than transformers at inference!

### Mamba Architecture

```python
class MambaBlock(nn.Module):
    """Complete Mamba block with residual"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class Mamba(nn.Module):
    """Full Mamba model"""

    def __init__(self, vocab_size, d_model=768, n_layers=24, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
```

---

## Section 4: Efficient Long-Range Dependencies

### Why SSMs Excel at Long Range

**1. Linear Memory**: State size is fixed regardless of sequence length
```python
# Transformer: KV cache grows with sequence
memory_transformer = O(n * d_model * n_layers)

# SSM: State is fixed
memory_ssm = O(d_state * d_model * n_layers)  # Independent of n!
```

**2. No Quadratic Attention**:
```python
# Transformer attention
attention_cost = O(n^2 * d_model)

# SSM scan
scan_cost = O(n * d_state * d_model)
```

**3. Continuous-Time Foundation**: Natural handling of irregular sampling

### Long Range Arena Results

From S4 paper, performance on LRA benchmark (sequence lengths 1K-16K):

| Task | Length | S4 | Transformer |
|------|--------|-----|-------------|
| ListOps | 2K | 59.60 | 36.37 |
| Text | 4K | 86.82 | 64.27 |
| Retrieval | 4K | 90.90 | 57.46 |
| Image | 1K | 88.65 | 42.44 |
| Pathfinder | 1K | 94.20 | 71.40 |
| **Path-X** | **16K** | **96.35** | **FAIL** |

### Scaling Laws

Mamba matches transformer scaling laws while being more efficient:

From Mamba paper:
- Mamba-3B matches Transformer-6B on language modeling
- Linear scaling enables million-token contexts
- Training is parallelizable (unlike RNNs)

---

## Section 5: Complete Implementation

### Production-Ready SSM Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SSMLayer(nn.Module):
    """
    Production SSM layer supporting both training (conv) and inference (recurrent)

    Features:
    - Selective (input-dependent) B, C, dt
    - Efficient convolutional training mode
    - Fast recurrent inference mode
    - Proper initialization for long-range
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Determine dt projection rank
        self.dt_rank = d_model // 16 if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # Selection projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt projection
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to log-uniform between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
            + torch.log(torch.tensor(dt_min))
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter - initialized for long-range
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x, inference_params=None):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        batch, seqlen, _ = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # Selective parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, d_inner)

        # SSM
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = self.selective_scan_ref(x, dt, A, B, C, self.D.float())

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan_ref(self, u, delta, A, B, C, D):
        """
        Reference implementation of selective scan

        u: (B, L, d_inner)
        delta: (B, L, d_inner)
        A: (d_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        D: (d_inner,)
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Scan
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C[:, i].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        return y + u * D

    def step(self, x, state):
        """
        Single step for autoregressive inference

        x: (B, D)
        state: (B, d_inner, d_state)
        Returns: (B, D), new_state
        """
        # Similar to forward but for single token
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Update conv state and apply conv
        # (Implementation would maintain conv buffer)
        x = F.silu(x)

        # Selective params
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # State update
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)

        new_state = dA * state + dB * x.unsqueeze(-1)
        y = (new_state * C.unsqueeze(1)).sum(dim=-1)
        y = y + x * self.D

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y), new_state
```

### Performance Optimizations

```python
# CUDA kernel pseudocode for parallel selective scan
# Uses work-efficient parallel scan algorithm

def parallel_selective_scan_cuda(u, delta, A, B, C, D):
    """
    GPU-efficient selective scan

    Key optimizations:
    1. Fuse discretization with scan
    2. Block-parallel scan with associative operator
    3. Recompute rather than store intermediates
    """
    # Load to shared memory
    # Compute local prefix sums within blocks
    # Combine block results
    # Store outputs
    pass

# Memory usage comparison
def memory_analysis(seq_len, d_model, n_layers):
    # Transformer KV cache
    kv_cache = 2 * seq_len * d_model * n_layers  # K and V

    # SSM state
    d_state = 16  # Typical value
    ssm_state = d_state * d_model * n_layers

    # For seq_len = 1M, d_model = 4096, n_layers = 32:
    # Transformer: 2 * 1M * 4096 * 32 = 256 GB
    # SSM: 16 * 4096 * 32 = 2 MB

    return {
        "transformer_bytes": kv_cache * 2,  # float16
        "ssm_bytes": ssm_state * 2
    }
```

---

## Section 6: TRAIN STATION - SSM = RNN = Memory = Markov = Temporal

### The Grand Unification

**SSM IS RNN IS MEMORY IS MARKOV**

All temporal models are fundamentally doing the same thing: compressing history into a state that's sufficient for prediction.

```
      SSM                    RNN                   Markov
       |                      |                      |
   h' = Ah + Bx          h = f(Wh + Ux)         P(x|history)
       |                      |                      |
       +----------+-----------+                      |
                  |                                  |
            HIDDEN STATE = SUFFICIENT STATISTIC = MEMORY
                  |
                  v
        Future is independent of past given state
```

### The Markov Property

**Key Insight**: The state h makes the process Markov - given h[t], the future is independent of the past.

```python
# All these are equivalent!

# SSM view
h[t+1] = A_bar * h[t] + B_bar * x[t]  # Linear dynamics

# RNN view
h[t+1] = tanh(W @ h[t] + U @ x[t])    # Nonlinear dynamics

# Markov view
P(future | h[t]) = P(future | x[0:t])  # Sufficient statistic

# Memory view
h[t] = compress(x[0:t])               # Lossy compression of history
```

### Attention as Infinite State

Transformers are SSMs with infinite state dimension!

```python
# Attention: Store everything
state_attention = [x[0], x[1], ..., x[t-1]]  # State grows with t

# SSM: Compress into fixed size
state_ssm = A^(t-1) * B * x[0] + A^(t-2) * B * x[1] + ... + B * x[t-1]
```

The tradeoff:
- **Attention**: Perfect recall, but O(n^2) cost
- **SSM**: Lossy recall, but O(n) cost
- **Mamba**: Smart lossy recall via selectivity

### Predictive Coding Connection

From predictive coding perspective:
```python
# Prediction error drives learning
prediction = C @ h[t]
error = x[t] - prediction

# State update incorporates error
h[t+1] = A @ h[t] + B @ x[t]  # x[t] contains surprise!

# This IS Kalman filtering!
```

### Active Inference Connection

From active inference:
```python
# State as belief
mu = h  # Mean of belief about hidden causes

# Precision-weighted prediction error
epsilon = Pi @ (x - g(mu))

# State update minimizes free energy
d_mu/dt = -partial_F/partial_mu

# SSM IS variational inference with linear generative model!
```

---

## Section 7: ARR-COC Connection - Efficient Temporal Relevance

### Temporal Relevance as State Evolution

In ARR-COC, computing relevance over video requires tracking what matters over time. SSMs provide the perfect framework:

```python
class TemporalRelevanceSSM(nn.Module):
    """
    SSM for tracking relevance across video frames

    The state compresses relevance history efficiently,
    enabling O(1) relevance updates per frame.
    """

    def __init__(self, d_features, d_state=64):
        super().__init__()
        self.ssm = SelectiveSSM(d_features, d_state)
        self.relevance_head = nn.Linear(d_features, 1)

    def forward(self, frame_features):
        """
        frame_features: (batch, n_frames, d_features)
        Returns: relevance scores (batch, n_frames)
        """
        # SSM processes temporal sequence
        temporal_features = self.ssm(frame_features)

        # Predict frame relevance
        relevance = self.relevance_head(temporal_features).squeeze(-1)

        return torch.sigmoid(relevance)

    def stream_inference(self, frame_feature, state):
        """
        Single-frame inference for real-time processing

        frame_feature: (batch, d_features)
        state: (batch, d_inner, d_state)
        """
        # One SSM step
        output, new_state = self.ssm.step(frame_feature, state)

        # Relevance for this frame
        relevance = torch.sigmoid(self.relevance_head(output))

        return relevance, new_state
```

### Advantages for Video Understanding

**1. Linear Complexity**: Process hour-long videos efficiently
```python
# Transformer attention for 1 hour @ 30fps
n_frames = 30 * 60 * 60  # 108,000 frames
attention_cost = n_frames ** 2  # 11.6 billion operations per layer!

# SSM
ssm_cost = n_frames * d_state  # 1.7 million operations (6800x less!)
```

**2. Streaming Inference**: Real-time relevance updates
```python
# ARR-COC can update relevance as video streams
while video_streaming:
    frame = get_next_frame()
    features = extract_features(frame)
    relevance, state = model.stream_inference(features, state)

    if relevance > threshold:
        allocate_more_tokens(frame)
```

**3. Multi-Scale Temporal**: Different timescales for different features
```python
class MultiScaleTemporalRelevance(nn.Module):
    """Different SSMs for different temporal scales"""

    def __init__(self, d_features):
        super().__init__()
        # Fast changes (object motion)
        self.fast_ssm = SelectiveSSM(d_features, d_state=16, dt_min=0.01)

        # Medium changes (scene transitions)
        self.medium_ssm = SelectiveSSM(d_features, d_state=32, dt_min=0.1)

        # Slow changes (narrative arc)
        self.slow_ssm = SelectiveSSM(d_features, d_state=64, dt_min=1.0)

        self.combine = nn.Linear(d_features * 3, d_features)
```

**4. Selective Attention to Relevance Changes**: Mamba's selectivity naturally focuses on relevance shifts
```python
# The selection mechanism learns to:
# - Remember frames that change relevance (B controls what enters state)
# - Forget frames that maintain relevance (A controls decay)
# - Output relevance based on context (C reads state)
```

### Integration with Pyramid LOD

Combine SSM temporal modeling with pyramid spatial hierarchy:

```python
class ARRCOCTemporalPyramid(nn.Module):
    """
    Temporal SSMs at each pyramid level

    Higher levels: Coarse temporal modeling (slow changes)
    Lower levels: Fine temporal modeling (fast changes)
    """

    def __init__(self, pyramid_channels):
        super().__init__()
        self.level_ssms = nn.ModuleList([
            SelectiveSSM(
                channels,
                d_state=16 * (i + 1),  # More state at higher levels
                dt_min=0.001 * (2 ** i)  # Slower at higher levels
            )
            for i, channels in enumerate(pyramid_channels)
        ])

    def forward(self, pyramid_features_sequence):
        """
        Process temporal sequence at each pyramid level
        """
        outputs = []
        for level, (ssm, features) in enumerate(
            zip(self.level_ssms, pyramid_features_sequence)
        ):
            # features: (batch, time, height, width, channels)
            b, t, h, w, c = features.shape

            # Flatten spatial, process temporal
            flat = features.view(b * h * w, t, c)
            temporal = ssm(flat)

            # Reshape back
            outputs.append(temporal.view(b, t, h, w, c))

        return outputs
```

---

## Section 8: Performance Notes and Best Practices

### Training Considerations

**1. Use Convolutional Mode for Training**
```python
# Parallel FFT-based convolution
# O(n log n) but highly parallelizable
y = fft_conv(x, kernel)
```

**2. Initialization Matters**
```python
# A should be initialized for long-range memory
# S4: HiPPO initialization
# Mamba: Simple negative exponential

A = -torch.exp(torch.linspace(0.001, 0.1, d_state))  # Covers multiple timescales
```

**3. dt Initialization**
```python
# Initialize dt to cover desired timescale range
dt_init = torch.exp(
    torch.rand(d_inner) * (log_dt_max - log_dt_min) + log_dt_min
)
```

### Inference Optimizations

**1. Recurrent Mode for Generation**
```python
# O(1) per token, perfect for autoregressive
h = A_bar * h + B_bar * x
y = C * h
```

**2. State Caching**
```python
# Cache state between calls (like KV cache but much smaller)
class MambaCache:
    def __init__(self, batch_size, d_inner, d_state, n_layers, device):
        self.states = [
            torch.zeros(batch_size, d_inner, d_state, device=device)
            for _ in range(n_layers)
        ]
```

**3. Quantization**
```python
# SSMs are quantization-friendly
# State dynamics are smooth, no attention spikes
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### Memory Efficiency

```python
# Comparison for 1B parameter model, 100K context
#
# Transformer:
#   Weights: 2 GB
#   KV Cache: 100K * 4096 * 32 * 2 * 2 = 50 GB
#   Total: 52 GB
#
# Mamba:
#   Weights: 2 GB
#   State: 16 * 4096 * 48 * 2 = 6 MB
#   Total: 2 GB
```

---

## Sources

**Primary Papers:**
- [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) - Gu et al., 2021 (arXiv:2111.00396)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - Gu & Dao, 2023 (arXiv:2312.00752)

**Web Research (accessed 2025-11-23):**
- [Hazy Research S4 Blog](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1) - Stanford Hazy Research
- [Mamba Explained](https://thegradient.pub/mamba-explained/) - The Gradient, Kola Ayonrinde
- [The Annotated S4](https://srush.github.io/annotated-s4/) - Alexander Rush

**GitHub Implementations:**
- [state-spaces/s4](https://github.com/state-spaces/s4) - Official S4 repository
- [state-spaces/mamba](https://github.com/state-spaces/mamba) - Official Mamba repository

**Additional References:**
- [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) - Gu et al., 2020
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) - Maarten Grootendorst
- [Introduction to State Space Models (SSM)](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train) - Hugging Face
