# Temporal Hierarchies in Neural Networks

## Overview

Temporal hierarchies implement **multi-timescale processing** where different layers or modules operate at different temporal resolutions. This mirrors how biological systems process information across multiple timescales simultaneously - from millisecond neural firing to minutes of working memory to hours of episodic memory.

**Core Insight**: Just as Feature Pyramid Networks (FPN) process spatial information at multiple scales, temporal hierarchies process temporal information at multiple timescales.

From [Yamashita & Tani 2008](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000220) (cited 637 times):
> "We demonstrate that functional hierarchy can self-organize through multiple timescales in neural activity, without explicit spatial hierarchical structure."

From [Koutnik et al. 2014 - A Clockwork RNN](http://proceedings.mlr.press/v32/koutnik14.pdf) (cited 775 times):
> "A Clockwork RNN (CW-RNN) partitions the hidden layer into modules with different clock speeds, processing inputs at their own temporal granularity."

---

## Section 1: Multi-Timescale Processing Fundamentals

### The Hierarchy of Time in Neural Systems

**Biological Inspiration:**
- Sensory cortex: ~10-50ms timescales
- Association cortex: ~100-300ms timescales
- Prefrontal cortex: ~1-5 second timescales
- Hippocampus: minutes to hours

From [Golesorkhi et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7933253/) (cited 113 times):
> "Together, we demonstrate that the temporal hierarchy of (shorter and longer) intrinsic neural timescales converges with the spatial topography of the cortex."

### Key Principles

**1. Timescale Separation:**
- Fast dynamics for rapid responses
- Slow dynamics for context and memory
- Intermediate dynamics bridge the gap

**2. Information Flow:**
- Bottom-up: Fast to slow (abstraction)
- Top-down: Slow to fast (prediction/control)

**3. Clock Rates:**
- Different modules update at different frequencies
- Exponential spacing common (1, 2, 4, 8, ...)

### Mathematical Framework

For a hierarchical temporal system with L levels:

```
Level l timescale: tau_l = tau_0 * r^l

where:
  tau_0 = base timescale
  r = timescale ratio (typically 2-4)
  l = level index

Update frequency: f_l = 1/tau_l
```

**Continuous-time formulation:**
```
dx_l/dt = (1/tau_l) * (-x_l + f(W_l * x_l + U_l * x_{l-1}))
```

---

## Section 2: Clockwork RNN Architecture

### Core Concept

The Clockwork RNN partitions the hidden state into **modules** that update at different clock rates. This creates an explicit temporal hierarchy without needing multiple separate recurrent layers.

From [arXiv:1402.3511](https://arxiv.org/abs/1402.3511):
> "A Clockwork RNN (CW-RNN) partitions the hidden layer into modules processing inputs at different temporal granularity, reducing parameters and computation."

### Architecture Details

**Module Structure:**
- Hidden state h divided into g modules: h = [h_1, h_2, ..., h_g]
- Each module h_i has clock period T_i
- Module i only updates when timestep t mod T_i == 0

**Clock Periods:**
- Typically exponential: T_i = 2^(i-1)
- Example with 4 modules: [1, 2, 4, 8]

**Connection Structure:**
- Module i receives input from modules j where j >= i
- Slower modules (larger T) inform faster modules
- Creates natural information flow from slow to fast

### Update Equations

```python
# At timestep t, for module i:
if t % T_i == 0:
    # Module updates
    h_i(t) = tanh(W_i * x(t) + sum_{j>=i}(U_ij * h_j(t-1)) + b_i)
else:
    # Module maintains state
    h_i(t) = h_i(t-1)
```

### Advantages

**1. Reduced Computation:**
- Slow modules update infrequently
- O(g * n^2/g^2) vs O(n^2) for standard RNN

**2. Better Long-Range Dependencies:**
- Slow modules maintain information longer
- Exponential receptive field growth

**3. Parameter Efficiency:**
- Block-diagonal weight matrices
- Fewer connections to learn

---

## Section 3: Dilated Temporal Convolutions

### From WaveNet to TCN

Dilated convolutions achieve multi-timescale processing through **exponentially increasing dilation rates**, creating receptive fields that grow exponentially with depth.

From [Temporal Convolutional Networks for Action Segmentation (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf) (cited 2555 times):
> "Our Dilated TCN uses a deep series of dilated temporal convolutions adapted from the WaveNet model."

### Dilation Pattern

**Standard Pattern:**
```
Layer 1: dilation = 1   (looks at t-1, t)
Layer 2: dilation = 2   (looks at t-2, t)
Layer 3: dilation = 4   (looks at t-4, t)
Layer 4: dilation = 8   (looks at t-8, t)
...
```

**Receptive Field Growth:**
```
Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)

For k=2 and dilations [1,2,4,8]:
RF = 1 + 2 * 1 * 15 = 31 timesteps
```

### WaveNet vs TCN

**WaveNet (2016):**
- Causal dilated convolutions
- Gated activation units
- Skip connections from each layer
- Originally for audio generation

**TCN (2017):**
- Simplified architecture
- Residual blocks
- No gating necessary
- Better for many sequence tasks

### Hierarchical TCN Patterns

**Encoder-Decoder TCN:**
```
Encoder: Increasing dilations (1, 2, 4, 8, ...)
         Captures context at multiple scales

Decoder: Decreasing dilations (8, 4, 2, 1)
         Generates output at multiple scales
```

**Multi-Scale TCN:**
```
Parallel branches with different dilation patterns:
  Branch 1: [1, 2, 4]     - Fine temporal details
  Branch 2: [4, 8, 16]    - Medium-scale patterns
  Branch 3: [16, 32, 64]  - Long-range dependencies

Outputs concatenated or fused
```

---

## Section 4: Hierarchical Multiscale RNN (HM-RNN)

### Architecture Overview

From [Hierarchical Multiscale Recurrent Neural Networks (ICLR 2017)](https://openreview.net/pdf?id=S1di0sfgl) (cited 709 times):
> "We propose a novel multiscale RNN model, which can learn the hierarchical multiscale structure from temporal data without explicit boundary information."

### Key Innovation: Learned Boundaries

Unlike Clockwork RNN with fixed clock periods, HM-RNN **learns when to update** each level through boundary detectors.

**Three Operations per Layer:**
1. **UPDATE**: Process new information
2. **COPY**: Maintain previous state
3. **FLUSH**: Reset and propagate to higher level

### Boundary Detection

Each layer l has a boundary detector z_l:
```
z_l(t) = hard_sigmoid(W_z * h_l(t))

If z_l(t) = 1: FLUSH operation
   - Reset h_l to initial state
   - Send summarized information to layer l+1

If z_l(t) = 0:
   - UPDATE or COPY based on lower layer boundary
```

### Information Flow

**Bottom-up (during FLUSH):**
```
c_l+1(t) = summarize(h_l(t-1), h_l(t))
```

**Top-down (continuous):**
```
h_l(t) = f(h_l(t-1), c_l-1(t), h_l+1(t-1))
```

---

## Section 5: PyTorch Implementation - Hierarchical Temporal Network

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

class ClockworkRNNCell(nn.Module):
    """
    Clockwork RNN cell with modules updating at different clock rates.

    From Koutnik et al. 2014: "A Clockwork RNN"
    http://proceedings.mlr.press/v32/koutnik14.pdf
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_modules: int = 4,
        clock_base: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_modules = num_modules
        self.module_size = hidden_size // num_modules

        # Clock periods: [1, 2, 4, 8, ...]
        self.clock_periods = [clock_base ** i for i in range(num_modules)]

        # Input weights for each module
        self.W_in = nn.ModuleList([
            nn.Linear(input_size, self.module_size)
            for _ in range(num_modules)
        ])

        # Recurrent weights - module i receives from modules j >= i
        # This creates the hierarchical structure
        self.W_rec = nn.ModuleList([
            nn.Linear(
                self.module_size * (num_modules - i),  # Modules i to num_modules
                self.module_size
            )
            for i in range(num_modules)
        ])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_size]
            h: Hidden state [batch, hidden_size]
            timestep: Current timestep (for clock gating)

        Returns:
            New hidden state [batch, hidden_size]
        """
        batch_size = x.size(0)

        # Split hidden state into modules
        h_modules = h.split(self.module_size, dim=1)
        new_h_modules = []

        for i in range(self.num_modules):
            period = self.clock_periods[i]

            if timestep % period == 0:
                # Module updates
                # Get input from higher-indexed modules (slower)
                h_upper = torch.cat(h_modules[i:], dim=1)

                # Compute new state
                input_contrib = self.W_in[i](x)
                rec_contrib = self.W_rec[i](h_upper)
                new_h_i = torch.tanh(input_contrib + rec_contrib)
            else:
                # Module maintains state (copy)
                new_h_i = h_modules[i]

            new_h_modules.append(new_h_i)

        return torch.cat(new_h_modules, dim=1)


class DilatedTemporalBlock(nn.Module):
    """
    Single block of dilated temporal convolution with residual connection.

    Based on TCN architecture from Lea et al. 2017 and WaveNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Calculate padding for causal convolution
        # Output length = Input length for causal conv
        self.padding = (kernel_size - 1) * dilation

        # Two layers of dilated convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

        # Normalization and regularization
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, channels, time]

        Returns:
            Output [batch, channels, time]
        """
        # First conv layer
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Remove future timesteps (causal)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # Second conv layer
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # Causal trimming
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Residual connection
        residual = self.residual(x)

        return F.relu(out + residual)


class HierarchicalTCN(nn.Module):
    """
    Hierarchical Temporal Convolutional Network with multiple timescales.

    Features:
    - Exponentially increasing dilations
    - Multi-scale feature extraction
    - Residual connections
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_levels = num_levels

        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)

        # Build temporal blocks with exponential dilations
        # Dilation pattern: 1, 2, 4, 8, ...
        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            dilation = 2 ** i
            self.blocks.append(
                DilatedTemporalBlock(
                    hidden_size, hidden_size,
                    kernel_size, dilation, dropout
                )
            )

        # Output projection
        self.output_proj = nn.Conv1d(hidden_size, output_size, 1)

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(
            kernel_size, num_levels
        )

    def _calculate_receptive_field(
        self,
        kernel_size: int,
        num_levels: int
    ) -> int:
        """Calculate total receptive field of the network."""
        rf = 1
        for i in range(num_levels):
            dilation = 2 ** i
            rf += 2 * (kernel_size - 1) * dilation
        return rf

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input [batch, time, features] or [batch, features, time]

        Returns:
            output: Final output [batch, output_size, time]
            intermediates: Features from each level for multi-scale analysis
        """
        # Ensure [batch, features, time] format
        if x.dim() == 3 and x.size(1) > x.size(2):
            x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)

        # Process through hierarchical blocks
        intermediates = []
        for block in self.blocks:
            x = block(x)
            intermediates.append(x)

        # Output projection
        output = self.output_proj(x)

        return output, intermediates


class MultiTimescaleRNN(nn.Module):
    """
    RNN with explicit multiple timescales through learnable time constants.

    Inspired by:
    - Yamashita & Tani 2008: "Emergence of Functional Hierarchy"
    - Kurikawa et al. 2021: "Multiple-Timescale Neural Networks"
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        timescales: List[float],
        output_size: int
    ):
        super().__init__()

        assert len(hidden_sizes) == len(timescales)

        self.num_levels = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        # Time constants (tau) for each level
        # Larger tau = slower dynamics
        self.register_buffer(
            'timescales',
            torch.tensor(timescales)
        )

        # Layers for each timescale level
        self.layers = nn.ModuleList()

        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # Each level receives:
            # - Input (or previous level output)
            # - Own recurrent connection
            # - Connection from adjacent levels (if any)

            total_input = prev_size
            if i > 0:
                total_input += hidden_sizes[i-1]  # From level below
            if i < self.num_levels - 1:
                total_input += hidden_sizes[i+1]  # From level above

            self.layers.append(
                nn.Linear(total_input + hidden_size, hidden_size)
            )
            prev_size = hidden_size

        # Output projection
        self.output_proj = nn.Linear(sum(hidden_sizes), output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input sequence [batch, time, features]
            hidden_states: Initial hidden states for each level

        Returns:
            output: Network output [batch, time, output_size]
            final_hidden: Final hidden states
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, h_size, device=x.device)
                for h_size in self.hidden_sizes
            ]

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_hidden = []

            for i in range(self.num_levels):
                # Gather inputs for this level
                inputs = [x_t if i == 0 else hidden_states[i-1]]

                # Add connections from adjacent levels
                if i > 0:
                    inputs.append(hidden_states[i-1])
                if i < self.num_levels - 1:
                    inputs.append(hidden_states[i+1])

                # Add recurrent connection
                inputs.append(hidden_states[i])

                # Concatenate all inputs
                layer_input = torch.cat(inputs, dim=1)

                # Compute candidate activation
                candidate = torch.tanh(self.layers[i](layer_input))

                # Apply timescale-dependent update
                # h_new = (1 - 1/tau) * h_old + (1/tau) * candidate
                alpha = 1.0 / self.timescales[i]
                new_h = (1 - alpha) * hidden_states[i] + alpha * candidate

                new_hidden.append(new_h)

            hidden_states = new_hidden

            # Combine all levels for output
            combined = torch.cat(hidden_states, dim=1)
            outputs.append(combined)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)

        # Project to output
        output = self.output_proj(outputs)

        return output, hidden_states


class HierarchicalTemporalPredictor(nn.Module):
    """
    Complete hierarchical temporal prediction system combining:
    - Clockwork-style gated updates
    - TCN-style dilated convolutions
    - Multi-timescale dynamics

    For tasks requiring prediction at multiple temporal scales.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_scales: int = 4,
        use_tcn: bool = True,
        use_clockwork: bool = True
    ):
        super().__init__()

        self.num_scales = num_scales
        self.use_tcn = use_tcn
        self.use_clockwork = use_clockwork

        # TCN encoder for hierarchical features
        if use_tcn:
            self.tcn_encoder = HierarchicalTCN(
                input_size, hidden_size, hidden_size,
                num_levels=num_scales
            )
        else:
            self.input_proj = nn.Linear(input_size, hidden_size)

        # Clockwork RNN for temporal dynamics
        if use_clockwork:
            self.clockwork = ClockworkRNNCell(
                hidden_size, hidden_size,
                num_modules=num_scales
            )
        else:
            self.rnn = nn.GRUCell(hidden_size, hidden_size)

        # Multi-scale prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, output_size)
            for _ in range(num_scales)
        ])

        # Timescale-specific attention for combining predictions
        self.scale_attention = nn.Linear(hidden_size, num_scales)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: Input [batch, time, features]
            hidden: Initial hidden state

        Returns:
            output: Combined prediction [batch, time, output_size]
            hidden: Final hidden state
            info: Dictionary with multi-scale predictions and attention
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(
                batch_size,
                self.clockwork.hidden_size if self.use_clockwork else x.size(-1),
                device=x.device
            )

        # Process through TCN if enabled
        if self.use_tcn:
            # TCN expects [batch, features, time]
            x_tcn = x.transpose(1, 2)
            tcn_out, intermediates = self.tcn_encoder(x_tcn)
            x_processed = tcn_out.transpose(1, 2)  # Back to [batch, time, features]
        else:
            x_processed = self.input_proj(x)

        # Process through temporal dynamics
        outputs = []
        scale_predictions = []
        scale_attentions = []

        for t in range(seq_len):
            x_t = x_processed[:, t, :]

            # Update hidden state
            if self.use_clockwork:
                hidden = self.clockwork(x_t, hidden, t)
            else:
                hidden = self.rnn(x_t, hidden)

            # Multi-scale predictions
            scale_preds = [head(hidden) for head in self.prediction_heads]
            scale_pred_tensor = torch.stack(scale_preds, dim=1)  # [batch, scales, output]

            # Attention over scales
            attn_logits = self.scale_attention(hidden)  # [batch, scales]
            attn_weights = F.softmax(attn_logits, dim=1)  # [batch, scales]

            # Weighted combination
            combined = torch.einsum(
                'bs,bso->bo',
                attn_weights,
                scale_pred_tensor
            )

            outputs.append(combined)
            scale_predictions.append(scale_pred_tensor)
            scale_attentions.append(attn_weights)

        # Stack outputs
        output = torch.stack(outputs, dim=1)
        scale_predictions = torch.stack(scale_predictions, dim=1)
        scale_attentions = torch.stack(scale_attentions, dim=1)

        info = {
            'scale_predictions': scale_predictions,
            'scale_attentions': scale_attentions,
            'receptive_field': self.tcn_encoder.receptive_field if self.use_tcn else None
        }

        return output, hidden, info


# ============================================================================
# USAGE EXAMPLE AND TRAINING
# ============================================================================

def demo_hierarchical_temporal():
    """Demonstrate hierarchical temporal network usage."""

    print("=" * 60)
    print("Hierarchical Temporal Networks Demo")
    print("=" * 60)

    # Configuration
    batch_size = 16
    seq_len = 100
    input_size = 32
    hidden_size = 64
    output_size = 10
    num_scales = 4

    # Create input
    x = torch.randn(batch_size, seq_len, input_size)

    # 1. Test Clockwork RNN
    print("\n1. Clockwork RNN Cell:")
    cw_cell = ClockworkRNNCell(input_size, hidden_size, num_modules=4)
    h = torch.zeros(batch_size, hidden_size)

    for t in range(10):
        h = cw_cell(x[:, t, :], h, t)
    print(f"   Output shape: {h.shape}")
    print(f"   Clock periods: {cw_cell.clock_periods}")

    # 2. Test Hierarchical TCN
    print("\n2. Hierarchical TCN:")
    tcn = HierarchicalTCN(input_size, hidden_size, output_size, num_levels=4)
    output, intermediates = tcn(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Receptive field: {tcn.receptive_field} timesteps")
    print(f"   Intermediate shapes: {[f.shape for f in intermediates]}")

    # 3. Test Multi-Timescale RNN
    print("\n3. Multi-Timescale RNN:")
    timescales = [1.0, 2.0, 4.0, 8.0]  # tau values
    mt_rnn = MultiTimescaleRNN(
        input_size,
        hidden_sizes=[32, 32, 32, 32],
        timescales=timescales,
        output_size=output_size
    )
    output, final_hidden = mt_rnn(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Timescales (tau): {timescales}")

    # 4. Test Complete Hierarchical Predictor
    print("\n4. Hierarchical Temporal Predictor:")
    predictor = HierarchicalTemporalPredictor(
        input_size, hidden_size, output_size,
        num_scales=num_scales,
        use_tcn=True,
        use_clockwork=True
    )
    output, hidden, info = predictor(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Scale predictions shape: {info['scale_predictions'].shape}")
    print(f"   Scale attentions shape: {info['scale_attentions'].shape}")

    # Show attention distribution
    avg_attention = info['scale_attentions'].mean(dim=(0, 1))
    print(f"   Average attention per scale: {avg_attention.detach().numpy()}")

    # 5. Performance estimation
    print("\n5. Performance Characteristics:")

    # Count parameters
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Estimate memory
    mem_mb = total_params * 4 / (1024 * 1024)  # float32
    print(f"   Model memory: {mem_mb:.2f} MB")

    # Timing
    import time

    predictor.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = predictor(x)
        elapsed = (time.time() - start) / 100

    print(f"   Inference time: {elapsed*1000:.2f} ms")
    print(f"   Throughput: {seq_len/elapsed:.0f} timesteps/sec")

    print("\n" + "=" * 60)

    return predictor


if __name__ == "__main__":
    model = demo_hierarchical_temporal()
```

---

## Section 6: Performance Considerations

### Computational Complexity

**Clockwork RNN:**
- Standard RNN: O(n^2) per timestep
- CW-RNN with g modules: O(g * (n/g)^2) = O(n^2/g)
- **Speedup factor: g** (number of modules)

**TCN:**
- O(L * k * n * T) where L=layers, k=kernel, n=channels, T=time
- Parallelizable over time dimension (unlike RNN)
- GPU-efficient due to convolution optimization

**Memory Usage:**

| Architecture | Memory Complexity | Notes |
|-------------|------------------|-------|
| Standard RNN | O(T * n) | Sequential, can't parallelize |
| Clockwork RNN | O(T * n) | Same but fewer computations |
| TCN | O(L * n * T) | All timesteps stored |
| Transformer | O(T^2 * n) | Full attention matrix |

### GPU Optimization Tips

**1. Batch Clock Updates:**
```python
# Instead of checking each module separately
# Create mask for all modules at once
clock_mask = torch.tensor([
    t % period == 0 for period in clock_periods
])
```

**2. Fused Operations:**
```python
# Use torch.jit.script for clock gating
@torch.jit.script
def clockwork_update(h, update, mask):
    return torch.where(mask, update, h)
```

**3. Memory-Efficient TCN:**
```python
# Use gradient checkpointing for deep TCNs
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    for block in self.blocks:
        x = checkpoint(block, x)
    return x
```

### Training Strategies

**1. Curriculum Learning:**
- Start with shorter sequences
- Gradually increase to full length
- Helps slow modules learn first

**2. Timescale Initialization:**
```python
# Initialize slow modules with smaller weights
for i, module in enumerate(self.modules):
    scale = 1.0 / (2 ** i)  # Smaller for slower modules
    module.weight.data *= scale
```

**3. Loss Weighting:**
```python
# Weight loss by timescale for multi-scale predictions
timescale_weights = [1.0, 0.5, 0.25, 0.125]  # More weight on fast scales
total_loss = sum(w * loss_fn(pred, target)
                 for w, pred in zip(timescale_weights, scale_preds))
```

---

## Section 7: TRAIN STATION - Temporal Hierarchy = FPN = Predictive Coding = Friston

### The Deep Unification

**All hierarchical temporal systems share the same fundamental structure:**

```
                    SLOW/COARSE
                        |
                    +-------+
                    | Level |  <-- Predictions down
                    |   3   |
                    +-------+
                        |
                    +-------+
                    | Level |
                    |   2   |
                    +-------+
                        |
                    +-------+
                    | Level |  --> Errors up
                    |   1   |
                    +-------+
                        |
                    FAST/FINE
```

### Equivalence Table

| Concept | Temporal Hierarchy | FPN (Vision) | Predictive Coding | Friston Free Energy |
|---------|-------------------|--------------|-------------------|---------------------|
| **Bottom-up** | Fast to slow | High-res to low-res | Prediction errors | Sensory evidence |
| **Top-down** | Slow to fast | Low-res to high-res | Predictions | Prior expectations |
| **Skip connections** | Direct paths | Lateral connections | Error channels | Precision weighting |
| **Timescale** | Clock periods | Spatial scale | Temporal depth | Markov blanket |
| **Purpose** | Context | Multi-scale features | Minimize surprise | Minimize free energy |

### The Coffee Cup = Donut Insight

From [Friston 2009 - Predictive Coding under Free-Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) (cited 1956 times):
> "A key aspect of these models is their hierarchical form, which induces empirical priors on the representation at any level."

**Why they're the same:**

1. **Hierarchical Generative Models:**
   - All create predictions from coarse to fine
   - All propagate errors from fine to coarse
   - All have lateral connections at each level

2. **Information Compression:**
   - Higher levels = more abstract/slower/coarser
   - Lower levels = more detailed/faster/finer
   - Compression ratio increases with level

3. **Predictive Structure:**
   - Top-down = "What should be happening"
   - Bottom-up = "What is actually happening"
   - Difference = error/surprise/prediction error

### Mathematical Unification

**Temporal Hierarchy Update:**
```
h_l(t) = f(h_l(t-1), input_l(t), h_{l+1}(t-1))
```

**FPN Update:**
```
F_l = upsample(F_{l+1}) + lateral(C_l)
```

**Predictive Coding Update:**
```
mu_l = mu_l + alpha * (prediction_error_l - precision * backward_error_{l+1})
```

**Free Energy Update:**
```
q(s_l) = argmin_q F(q, s_{l-1}, s_{l+1})
```

**All are the same structure!**
- Level l depends on its own state
- Level l receives from level l+1 (above)
- Level l sends to level l-1 (below)
- Bidirectional information flow

### Implementation Bridge

```python
class UnifiedHierarchicalLevel(nn.Module):
    """
    Generic hierarchical level that unifies:
    - Temporal hierarchy (Clockwork, HM-RNN)
    - Feature Pyramid (FPN)
    - Predictive Coding
    - Active Inference
    """

    def __init__(self, dim: int, timescale: float = 1.0):
        super().__init__()

        # Own state transition
        self.state_transition = nn.Linear(dim, dim)

        # Top-down pathway (predictions from above)
        self.top_down = nn.Linear(dim, dim)

        # Bottom-up pathway (evidence from below)
        self.bottom_up = nn.Linear(dim, dim)

        # Timescale (larger = slower dynamics)
        self.timescale = timescale

    def forward(
        self,
        state: torch.Tensor,
        from_above: torch.Tensor,
        from_below: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            new_state: Updated representation
            error: Prediction error to send upward
        """
        # Prediction from above (what we expect)
        prediction = self.top_down(from_above)

        # Evidence from below (what we observe)
        evidence = self.bottom_up(from_below)

        # Prediction error
        error = evidence - prediction

        # State update with timescale
        alpha = 1.0 / self.timescale
        candidate = torch.tanh(
            self.state_transition(state) + error
        )
        new_state = (1 - alpha) * state + alpha * candidate

        return new_state, error
```

---

## Section 8: ARR-COC-0-1 Connection - Multi-Scale Temporal Relevance (10%)

### Application to Visual Token Relevance

**ARR-COC-0-1 computes relevance at multiple temporal scales:**

```python
class MultiScaleTemporalRelevance(nn.Module):
    """
    Temporal relevance scoring at multiple timescales.

    For VLM token allocation:
    - Fast scale: Frame-level relevance (immediate visual salience)
    - Medium scale: Segment-level relevance (action/event context)
    - Slow scale: Video-level relevance (overall narrative importance)
    """

    def __init__(
        self,
        token_dim: int,
        num_scales: int = 3,
        timescales: List[float] = [1.0, 4.0, 16.0]
    ):
        super().__init__()

        self.num_scales = num_scales

        # Temporal processors at each scale
        self.temporal_processors = nn.ModuleList([
            nn.GRU(token_dim, token_dim, batch_first=True)
            for _ in range(num_scales)
        ])

        # Timescale-specific relevance heads
        self.relevance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, token_dim // 2),
                nn.ReLU(),
                nn.Linear(token_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_scales, num_scales),
            nn.Softmax(dim=-1)
        )

        # Time constants
        self.register_buffer(
            'timescales',
            torch.tensor(timescales)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_scale_breakdown: bool = False
    ) -> torch.Tensor:
        """
        Args:
            tokens: Video tokens [batch, time, num_tokens, dim]

        Returns:
            relevance: Multi-scale relevance [batch, time, num_tokens]
        """
        batch, time, num_tokens, dim = tokens.shape

        # Reshape for temporal processing: [batch * num_tokens, time, dim]
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(
            batch * num_tokens, time, dim
        )

        # Process at each timescale
        scale_relevances = []

        for i, (processor, head, tau) in enumerate(zip(
            self.temporal_processors,
            self.relevance_heads,
            self.timescales
        )):
            # Apply timescale-dependent smoothing
            # Larger tau = more temporal smoothing
            alpha = 1.0 / tau

            # Process through GRU
            hidden, _ = processor(tokens_flat)

            # Exponential smoothing for timescale
            smoothed = torch.zeros_like(hidden)
            smoothed[:, 0] = hidden[:, 0]
            for t in range(1, time):
                smoothed[:, t] = (1 - alpha) * smoothed[:, t-1] + alpha * hidden[:, t]

            # Compute relevance at this scale
            relevance = head(smoothed).squeeze(-1)  # [batch*tokens, time]
            relevance = relevance.reshape(batch, num_tokens, time).permute(0, 2, 1)
            scale_relevances.append(relevance)

        # Stack scales: [batch, time, num_tokens, num_scales]
        all_scales = torch.stack(scale_relevances, dim=-1)

        # Compute scale fusion weights
        # Each token gets its own scale weighting
        scale_weights = self.scale_fusion(
            all_scales.mean(dim=-2)  # Average over time for stable weights
        )
        scale_weights = scale_weights.unsqueeze(2).expand_as(all_scales)

        # Weighted combination
        final_relevance = (all_scales * scale_weights).sum(dim=-1)

        if return_scale_breakdown:
            return final_relevance, scale_relevances

        return final_relevance


class HierarchicalTokenBudget(nn.Module):
    """
    Allocate token budgets hierarchically across temporal scales.

    Key insight: Different timescales need different token densities:
    - Fast (frame): Many tokens for visual detail
    - Medium (segment): Moderate tokens for action context
    - Slow (video): Few tokens for narrative summary
    """

    def __init__(
        self,
        total_budget: int,
        num_scales: int = 3,
        budget_ratios: List[float] = [0.6, 0.3, 0.1]
    ):
        super().__init__()

        self.total_budget = total_budget
        self.num_scales = num_scales

        # Budget allocation per scale
        self.register_buffer(
            'budget_ratios',
            torch.tensor(budget_ratios)
        )

        # Learnable budget adjustment
        self.budget_adjuster = nn.Sequential(
            nn.Linear(num_scales * 2, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        scale_relevances: List[torch.Tensor],
        content_complexity: torch.Tensor
    ) -> List[int]:
        """
        Args:
            scale_relevances: Relevance at each scale
            content_complexity: Overall content complexity [batch]

        Returns:
            budgets: Token budget for each scale
        """
        # Compute mean relevance per scale
        mean_relevances = torch.stack([
            r.mean(dim=(1, 2)) for r in scale_relevances
        ], dim=1)  # [batch, num_scales]

        # Combine with base ratios
        combined = torch.cat([
            mean_relevances,
            self.budget_ratios.unsqueeze(0).expand(mean_relevances.size(0), -1)
        ], dim=1)

        # Compute adjusted ratios
        adjusted_ratios = self.budget_adjuster(combined)

        # Scale by complexity (more complex = honor base ratios more)
        final_ratios = (
            content_complexity.unsqueeze(1) * self.budget_ratios +
            (1 - content_complexity.unsqueeze(1)) * adjusted_ratios
        )

        # Convert to integer budgets
        budgets = (final_ratios * self.total_budget).round().int()

        # Ensure total equals budget (adjust largest)
        diff = self.total_budget - budgets.sum(dim=1, keepdim=True)
        largest_idx = budgets.argmax(dim=1, keepdim=True)
        budgets.scatter_add_(1, largest_idx, diff)

        return budgets.tolist()[0]  # Return for single batch
```

### Performance Considerations for ARR-COC

**Memory vs Accuracy Trade-off:**

| Scale | Tokens | Memory | Detail Level |
|-------|--------|--------|--------------|
| Fast (1-2 frames) | 60% | High | Object boundaries, textures |
| Medium (5-10 frames) | 30% | Medium | Actions, movements |
| Slow (full video) | 10% | Low | Scene type, narrative |

**Adaptive Strategy:**
- Action videos: More medium-scale tokens
- Static scenes: More fast-scale tokens
- Complex narratives: More slow-scale tokens

---

## Sources

**Core Papers:**

- [Koutnik et al. 2014 - A Clockwork RNN](http://proceedings.mlr.press/v32/koutnik14.pdf) - ICML 2014, cited 775 times
- [Chung et al. 2017 - Hierarchical Multiscale RNN](https://openreview.net/pdf?id=S1di0sfgl) - ICLR 2017, cited 709 times
- [Yamashita & Tani 2008 - Emergence of Functional Hierarchy](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000220) - PLOS Comp Bio, cited 637 times
- [Lea et al. 2017 - Temporal Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf) - CVPR 2017, cited 2555 times

**Predictive Coding & Free Energy:**

- [Friston & Kiebel 2009 - Predictive Coding under Free-Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) - Phil Trans Roy Soc, cited 1956 times
- [Millidge et al. 2021 - Predictive Coding: A Theoretical and Experimental Review](https://arxiv.org/pdf/2107.12979) - arXiv, cited 299 times
- [Jiang et al. 2024 - Dynamic Predictive Coding](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801) - PLOS Comp Bio, cited 45 times

**Biological Temporal Hierarchy:**

- [Golesorkhi et al. 2021 - Temporal Hierarchy of Intrinsic Neural Timescales](https://pmc.ncbi.nlm.nih.gov/articles/PMC7933253/) - Neuroimage, cited 113 times
- [Kurikawa et al. 2021 - Multiple-Timescale Neural Networks](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.743537/full) - Front Comp Neurosci, cited 17 times
- [Quax et al. 2020 - Adaptive Time Scales in Recurrent Neural Networks](https://www.nature.com/articles/s41598-020-68169-x) - Scientific Reports, cited 27 times

**TCN Resources:**

- [GitHub: keras-tcn](https://github.com/philipperemy/keras-tcn) - Popular Keras implementation
- [GitHub: pytorch-tcn](https://github.com/paul-krug/pytorch-tcn) - Flexible PyTorch implementation
- [Naoki Shibuya's TCN Tutorial](https://naokishibuya.github.io/blog/2021-09-13-temporal-convolutional-networks/) - Clear explanation with visuals

**Additional References:**

- [Zheng et al. 2024 - Temporal Dendritic Heterogeneity](https://www.nature.com/articles/s41467-023-44614-z) - Nature Comm, cited 110 times
- [Moro et al. 2024 - Temporal Hierarchy in Spiking Neural Networks](https://arxiv.org/abs/2407.18838) - arXiv, cited 8 times

---

## Summary

**Temporal hierarchies enable multi-timescale processing through:**

1. **Clockwork RNN**: Fixed clock periods, module-based updates
2. **Dilated Convolutions**: Exponential receptive field growth
3. **HM-RNN**: Learned boundaries, adaptive timescales
4. **Multi-timescale RNN**: Continuous time constants

**TRAIN STATION unification:**
- Temporal hierarchy = FPN = Predictive coding = Friston free energy
- All have bidirectional hierarchical structure
- All pass predictions down, errors up
- All compress information at higher levels

**Key insight**: The hierarchy of timescales in neural networks mirrors both:
- Biological cortical hierarchies (sensory to prefrontal)
- Information-theoretic compression (fine to coarse)
- Predictive processing (fast predictions, slow context)

This unification suggests that **multi-scale temporal processing is fundamental to intelligence** - whether biological or artificial.
