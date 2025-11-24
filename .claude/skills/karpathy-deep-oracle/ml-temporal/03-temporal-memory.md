# Temporal Memory Networks: Neural Turing Machines and Differentiable Neural Computers

## Overview

Memory-augmented neural networks (MANNs) extend standard neural networks with external memory resources, enabling them to learn algorithms, store and retrieve information over long time horizons, and generalize to sequences far longer than seen during training. This document covers Neural Turing Machines (NTMs), Differentiable Neural Computers (DNCs), and their temporal memory mechanisms.

**TRAIN STATION**: Memory = retention = history = context = attention over time

---

## 1. Memory-Augmented Architectures

### The Fundamental Problem

Standard RNNs (including LSTMs) suffer from fundamental limitations:

```
# LSTM hidden state capacity is FIXED
hidden_size = 512  # This is ALL your memory!

# For a 1000-step sequence:
# - Must compress ALL relevant info into 512 dims
# - Gradient flow through 1000 steps
# - Catastrophic forgetting of early information
```

**Memory-augmented networks solve this by separating:**
- **Computation** (controller network) from **Storage** (external memory)
- Like a CPU vs RAM in computer architecture

### Architecture Components

**1. Controller Network**
- Usually LSTM or feedforward network
- Reads inputs, produces outputs
- Generates memory interface signals

**2. External Memory Matrix**
- M: (N x W) matrix where N = number of slots, W = memory width
- Differentiable read/write operations
- Persists across time steps

**3. Read/Write Heads**
- Attention mechanisms for memory access
- Content-based and location-based addressing
- Multiple read heads, typically one write head

### Mathematical Framework

At each time step t:

```python
# Controller receives input and previous reads
h_t = controller(x_t, r_{t-1}, h_{t-1})

# Controller emits interface vector
interface = linear(h_t)

# Interface parsed into memory operations
k_t, beta_t, g_t, s_t, gamma_t, e_t, a_t = parse(interface)

# Memory operations
w_t = attention(k_t, beta_t, M_{t-1})  # Address
r_t = read(M_{t-1}, w_t)               # Read
M_t = write(M_{t-1}, w_t, e_t, a_t)    # Write

# Output
y_t = output_layer(h_t, r_t)
```

---

## 2. Neural Turing Machines (NTM)

### Original Architecture (Graves et al., 2014)

From [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (arXiv:1410.5401):

> "We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to-end."

### Addressing Mechanisms

**Content-Based Addressing:**
```python
def content_addressing(memory, key, strength):
    """
    Cosine similarity between key and memory rows

    Args:
        memory: (N, W) memory matrix
        key: (W,) query vector
        strength: scalar sharpening factor

    Returns:
        (N,) attention weights
    """
    # Cosine similarity
    similarity = F.cosine_similarity(
        memory,                    # (N, W)
        key.unsqueeze(0),          # (1, W)
        dim=1
    )  # (N,)

    # Sharpened softmax
    weights = F.softmax(strength * similarity, dim=0)
    return weights
```

**Location-Based Addressing:**
```python
def location_addressing(prev_weights, gate, shift, sharpen):
    """
    Interpolate, shift, and sharpen weights

    Args:
        prev_weights: (N,) previous attention weights
        gate: scalar interpolation gate [0, 1]
        shift: (shift_range,) shift distribution
        sharpen: scalar sharpening factor >= 1
    """
    # Interpolation with content weights (assumed done before)
    # w_g = g * w_c + (1-g) * w_{t-1}

    # Circular convolution for shifting
    shifted = circular_convolve(prev_weights, shift)

    # Sharpening
    sharpened = shifted ** sharpen
    weights = sharpened / sharpened.sum()

    return weights
```

### Read and Write Operations

**Reading:**
```python
def read(memory, weights):
    """
    Weighted sum of memory rows

    Args:
        memory: (N, W) memory matrix
        weights: (N,) attention weights

    Returns:
        (W,) read vector
    """
    return torch.matmul(weights, memory)  # (W,)
```

**Writing:**
```python
def write(memory, weights, erase, add):
    """
    Erase then add to memory

    Args:
        memory: (N, W) memory matrix
        weights: (N,) write weights
        erase: (W,) erase vector [0, 1]
        add: (W,) add vector

    Returns:
        (N, W) updated memory
    """
    # Erase
    erase_matrix = torch.outer(weights, erase)  # (N, W)
    memory = memory * (1 - erase_matrix)

    # Add
    add_matrix = torch.outer(weights, add)  # (N, W)
    memory = memory + add_matrix

    return memory
```

---

## 3. Differentiable Neural Computers (DNC)

### Improvements Over NTM

DNCs (Graves et al., 2016, Nature) add:

1. **Temporal link matrix** - Tracks write order
2. **Usage vector** - Memory allocation mechanism
3. **Dynamic memory allocation** - Free list management

### Temporal Link Matrix

The key innovation for temporal memory:

```python
class TemporalLinkage:
    """
    Tracks temporal order of memory writes
    """

    def __init__(self, num_slots):
        self.N = num_slots
        # Link matrix: L[i,j] = strength of link from i to j
        self.L = torch.zeros(num_slots, num_slots)
        # Precedence weights: which locations were recently written
        self.p = torch.zeros(num_slots)

    def update(self, write_weights):
        """
        Update links based on write operation

        Args:
            write_weights: (N,) current write weights
        """
        # Update link matrix
        # L[i,j] represents: after writing to i, we wrote to j

        # Remove old links to current write locations
        self.L = self.L * (1 - write_weights.unsqueeze(0))  # (N, N)
        self.L = self.L * (1 - write_weights.unsqueeze(1))  # (N, N)

        # Add new links from previous write to current
        self.L = self.L + torch.outer(write_weights, self.p)

        # Update precedence
        self.p = (1 - write_weights.sum()) * self.p + write_weights

    def forward_weights(self, read_weights):
        """Get next memory location in write order"""
        return torch.matmul(self.L.T, read_weights)

    def backward_weights(self, read_weights):
        """Get previous memory location in write order"""
        return torch.matmul(self.L, read_weights)
```

### Memory Allocation

```python
class MemoryAllocator:
    """
    Manages memory allocation using usage statistics
    """

    def __init__(self, num_slots):
        self.N = num_slots
        self.usage = torch.zeros(num_slots)

    def update_usage(self, read_weights, write_weights, free_gates):
        """
        Update usage based on operations

        Args:
            read_weights: list of (N,) read weight vectors
            write_weights: (N,) write weights
            free_gates: list of scalars, one per read head
        """
        # Memory retention
        retention = torch.ones(self.N)
        for r_w, f_g in zip(read_weights, free_gates):
            retention = retention * (1 - f_g * r_w)

        # Update usage
        self.usage = (self.usage + write_weights
                      - self.usage * write_weights) * retention

    def allocation_weights(self):
        """
        Get allocation weights for new writes
        Returns weights favoring unused memory locations
        """
        # Sort by usage (ascending)
        sorted_usage, indices = torch.sort(self.usage)

        # Compute allocation weights
        # Higher weight for less-used locations
        cumprod = torch.cumprod(sorted_usage, dim=0)
        alloc = (1 - sorted_usage) * torch.cat([
            torch.ones(1),
            cumprod[:-1]
        ])

        # Unsort
        allocation = torch.zeros_like(alloc)
        allocation.scatter_(0, indices, alloc)

        return allocation
```

---

## 4. Complete PyTorch Implementation

### Full DNC Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DNController(nn.Module):
    """LSTM controller for DNC"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        return self.lstm(x, hidden)


class DNCMemory(nn.Module):
    """
    DNC Memory Module with temporal linkage

    Reference: pytorch-dnc by ixaxaar
    https://github.com/ixaxaar/pytorch-dnc
    """

    def __init__(
        self,
        num_slots: int = 128,
        slot_size: int = 64,
        num_read_heads: int = 4,
    ):
        super().__init__()
        self.N = num_slots      # Number of memory locations
        self.W = slot_size      # Width of each memory slot
        self.R = num_read_heads

        # Interface vector sizes
        self.interface_size = (
            self.W * self.R +  # Read keys
            self.R +          # Read strengths
            self.W +          # Write key
            1 +               # Write strength
            self.W +          # Erase vector
            self.W +          # Write vector
            self.R +          # Free gates
            1 +               # Allocation gate
            1 +               # Write gate
            3 * self.R        # Read modes
        )

    def reset(self, batch_size: int, device: torch.device):
        """Initialize memory state"""
        return {
            'memory': torch.zeros(batch_size, self.N, self.W, device=device),
            'usage': torch.zeros(batch_size, self.N, device=device),
            'link': torch.zeros(batch_size, self.N, self.N, device=device),
            'precedence': torch.zeros(batch_size, self.N, device=device),
            'read_weights': torch.zeros(batch_size, self.R, self.N, device=device),
            'write_weights': torch.zeros(batch_size, self.N, device=device),
            'read_vectors': torch.zeros(batch_size, self.R, self.W, device=device),
        }

    def content_address(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        strengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Content-based addressing

        Args:
            memory: (B, N, W)
            keys: (B, num_heads, W)
            strengths: (B, num_heads)

        Returns:
            (B, num_heads, N) attention weights
        """
        # Normalize
        memory_norm = F.normalize(memory, dim=2)  # (B, N, W)
        keys_norm = F.normalize(keys, dim=2)      # (B, H, W)

        # Cosine similarity
        similarity = torch.bmm(keys_norm, memory_norm.transpose(1, 2))  # (B, H, N)

        # Apply strength and softmax
        return F.softmax(strengths.unsqueeze(2) * similarity, dim=2)

    def update_usage(
        self,
        usage: torch.Tensor,
        read_weights: torch.Tensor,
        write_weights: torch.Tensor,
        free_gates: torch.Tensor
    ) -> torch.Tensor:
        """Update memory usage statistics"""
        # Compute retention
        retention = torch.prod(
            1 - free_gates.unsqueeze(2) * read_weights,
            dim=1
        )  # (B, N)

        # Update usage
        usage = (usage + write_weights - usage * write_weights) * retention
        return usage

    def allocation_weights(self, usage: torch.Tensor) -> torch.Tensor:
        """Compute allocation weights from usage"""
        B, N = usage.shape

        # Sort by usage
        sorted_usage, indices = torch.sort(usage, dim=1)

        # Compute cumulative product for allocation
        cumprod = torch.cumprod(sorted_usage + 1e-8, dim=1)
        alloc_sorted = (1 - sorted_usage) * torch.cat([
            torch.ones(B, 1, device=usage.device),
            cumprod[:, :-1]
        ], dim=1)

        # Unsort
        allocation = torch.zeros_like(alloc_sorted)
        allocation.scatter_(1, indices, alloc_sorted)

        return allocation

    def update_link(
        self,
        link: torch.Tensor,
        precedence: torch.Tensor,
        write_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update temporal link matrix and precedence"""
        B, N, _ = link.shape

        # Expand write weights for broadcasting
        w_i = write_weights.unsqueeze(2)  # (B, N, 1)
        w_j = write_weights.unsqueeze(1)  # (B, 1, N)

        # Update link matrix
        link = (1 - w_i - w_j) * link + torch.bmm(
            write_weights.unsqueeze(2),
            precedence.unsqueeze(1)
        )

        # Zero diagonal
        link = link * (1 - torch.eye(N, device=link.device))

        # Update precedence
        precedence = (1 - write_weights.sum(dim=1, keepdim=True)) * precedence + write_weights

        return link, precedence

    def forward(
        self,
        interface: torch.Tensor,
        state: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Memory forward pass

        Args:
            interface: (B, interface_size) from controller
            state: dictionary of memory state tensors

        Returns:
            read_vectors: (B, R, W)
            new_state: updated state dict
        """
        B = interface.shape[0]

        # Parse interface vector
        idx = 0

        # Read keys and strengths
        read_keys = interface[:, idx:idx + self.W * self.R].view(B, self.R, self.W)
        idx += self.W * self.R
        read_strengths = F.softplus(interface[:, idx:idx + self.R])
        idx += self.R

        # Write key and strength
        write_key = interface[:, idx:idx + self.W].view(B, 1, self.W)
        idx += self.W
        write_strength = F.softplus(interface[:, idx:idx + 1])
        idx += 1

        # Erase and write vectors
        erase = torch.sigmoid(interface[:, idx:idx + self.W])
        idx += self.W
        write_vector = interface[:, idx:idx + self.W]
        idx += self.W

        # Gates
        free_gates = torch.sigmoid(interface[:, idx:idx + self.R])
        idx += self.R
        alloc_gate = torch.sigmoid(interface[:, idx:idx + 1])
        idx += 1
        write_gate = torch.sigmoid(interface[:, idx:idx + 1])
        idx += 1

        # Read modes (backward, content, forward)
        read_modes = F.softmax(
            interface[:, idx:idx + 3 * self.R].view(B, self.R, 3),
            dim=2
        )

        # === Memory Operations ===

        # 1. Update usage
        usage = self.update_usage(
            state['usage'],
            state['read_weights'],
            state['write_weights'],
            free_gates
        )

        # 2. Compute write weights
        content_w = self.content_address(
            state['memory'], write_key, write_strength
        ).squeeze(1)  # (B, N)

        alloc_w = self.allocation_weights(usage)  # (B, N)

        write_weights = write_gate * (
            alloc_gate * alloc_w + (1 - alloc_gate) * content_w
        )  # (B, N)

        # 3. Write to memory
        memory = state['memory'].clone()
        erase_matrix = torch.bmm(
            write_weights.unsqueeze(2),
            erase.unsqueeze(1)
        )  # (B, N, W)
        memory = memory * (1 - erase_matrix)

        add_matrix = torch.bmm(
            write_weights.unsqueeze(2),
            write_vector.unsqueeze(1)
        )  # (B, N, W)
        memory = memory + add_matrix

        # 4. Update temporal link
        link, precedence = self.update_link(
            state['link'],
            state['precedence'],
            write_weights
        )

        # 5. Compute read weights
        content_r = self.content_address(
            memory, read_keys, read_strengths
        )  # (B, R, N)

        # Forward and backward from temporal links
        forward_w = torch.bmm(
            state['read_weights'],
            link.transpose(1, 2)
        )  # (B, R, N)
        backward_w = torch.bmm(
            state['read_weights'],
            link
        )  # (B, R, N)

        # Combine read modes
        read_weights = (
            read_modes[:, :, 0:1] * backward_w +
            read_modes[:, :, 1:2] * content_r +
            read_modes[:, :, 2:3] * forward_w
        )  # (B, R, N)

        # 6. Read from memory
        read_vectors = torch.bmm(read_weights, memory)  # (B, R, W)

        # Update state
        new_state = {
            'memory': memory,
            'usage': usage,
            'link': link,
            'precedence': precedence,
            'read_weights': read_weights,
            'write_weights': write_weights,
            'read_vectors': read_vectors,
        }

        return read_vectors, new_state


class DNC(nn.Module):
    """
    Complete Differentiable Neural Computer

    Based on: Hybrid computing using a neural network with
    dynamic external memory (Nature 2016)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_slots: int = 128,
        slot_size: int = 64,
        num_read_heads: int = 4,
    ):
        super().__init__()

        self.num_read_heads = num_read_heads
        self.slot_size = slot_size

        # Memory module
        self.memory = DNCMemory(num_slots, slot_size, num_read_heads)

        # Controller input: original input + read vectors
        controller_input = input_size + num_read_heads * slot_size

        # Controller
        self.controller = DNController(
            controller_input, hidden_size, num_layers
        )

        # Interface layer
        self.interface_layer = nn.Linear(
            hidden_size, self.memory.interface_size
        )

        # Output layer
        self.output_layer = nn.Linear(
            hidden_size + num_read_heads * slot_size,
            output_size
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass

        Args:
            x: (B, T, input_size) input sequence
            state: optional previous state

        Returns:
            output: (B, T, output_size)
            final_state: state dict
        """
        B, T, _ = x.shape
        device = x.device

        # Initialize state
        if state is None:
            state = {
                'memory_state': self.memory.reset(B, device),
                'controller_hidden': None,
            }

        memory_state = state['memory_state']
        controller_hidden = state['controller_hidden']

        outputs = []

        for t in range(T):
            # Get current input
            x_t = x[:, t:t+1, :]  # (B, 1, input_size)

            # Concatenate with previous read vectors
            read_flat = memory_state['read_vectors'].view(
                B, 1, -1
            )  # (B, 1, R*W)
            controller_input = torch.cat([x_t, read_flat], dim=2)

            # Controller forward
            controller_out, controller_hidden = self.controller(
                controller_input, controller_hidden
            )  # (B, 1, hidden)

            # Generate interface
            interface = self.interface_layer(
                controller_out.squeeze(1)
            )  # (B, interface_size)

            # Memory operations
            read_vectors, memory_state = self.memory(
                interface, memory_state
            )  # (B, R, W)

            # Output
            output_input = torch.cat([
                controller_out.squeeze(1),
                read_vectors.view(B, -1)
            ], dim=1)  # (B, hidden + R*W)

            output = self.output_layer(output_input)  # (B, output_size)
            outputs.append(output)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (B, T, output_size)

        final_state = {
            'memory_state': memory_state,
            'controller_hidden': controller_hidden,
        }

        return outputs, final_state


# Example usage
def demo_dnc():
    """Demonstrate DNC on copy task"""

    # Create model
    dnc = DNC(
        input_size=8,
        output_size=8,
        hidden_size=64,
        num_slots=32,
        slot_size=16,
        num_read_heads=2,
    )

    # Sample input
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 8)

    # Forward pass
    output, state = dnc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Memory shape: {state['memory_state']['memory'].shape}")

    return dnc, output, state


if __name__ == "__main__":
    demo_dnc()
```

---

## 5. Sparse Access Memory (SAM)

### Scaling to Large Memories

For very large memories (>1000 slots), full attention is O(N^2). SAM uses sparse access:

```python
class SparseAccessMemory(nn.Module):
    """
    Memory with sparse read/write operations

    Uses approximate nearest neighbor search for scaling
    """

    def __init__(
        self,
        num_slots: int = 10000,
        slot_size: int = 64,
        num_reads: int = 4,
        num_writes: int = 1,
    ):
        super().__init__()
        self.N = num_slots
        self.W = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes

    def sparse_attention(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        k: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k attention weights

        Args:
            memory: (B, N, W)
            keys: (B, H, W)
            k: number of slots to attend to

        Returns:
            weights: (B, H, k)
            indices: (B, H, k)
        """
        B, N, W = memory.shape
        H = keys.shape[1]

        # Compute all similarities
        # In practice, use FAISS for approximate search
        similarity = torch.bmm(
            F.normalize(keys, dim=2),
            F.normalize(memory, dim=2).transpose(1, 2)
        )  # (B, H, N)

        # Top-k
        values, indices = torch.topk(similarity, k, dim=2)
        weights = F.softmax(values, dim=2)  # (B, H, k)

        return weights, indices

    def sparse_read(
        self,
        memory: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Read from sparse attention

        Args:
            memory: (B, N, W)
            weights: (B, H, k)
            indices: (B, H, k)

        Returns:
            read_vectors: (B, H, W)
        """
        B, H, k = weights.shape
        W = memory.shape[2]

        # Gather memory slots
        indices_expanded = indices.unsqueeze(3).expand(-1, -1, -1, W)
        memory_expanded = memory.unsqueeze(1).expand(-1, H, -1, -1)
        selected = torch.gather(memory_expanded, 2, indices_expanded)  # (B, H, k, W)

        # Weighted sum
        read_vectors = torch.sum(
            weights.unsqueeze(3) * selected, dim=2
        )  # (B, H, W)

        return read_vectors
```

---

## 6. Performance Considerations

### Memory and Compute Costs

**DNC Complexity per time step:**
- Content addressing: O(N * W * H) where H = num heads
- Link matrix operations: O(N^2) - the bottleneck!
- Memory read/write: O(N * W)

**Scaling strategies:**
1. **Sparse attention** (SAM): O(k * W * H) where k << N
2. **Factored memory**: Separate key/value like transformers
3. **Hierarchical memory**: Multi-level addressing

### GPU Optimization

```python
# Use batched operations
# BAD: Loop over heads
for h in range(num_heads):
    weights[h] = attention(memory, keys[h])

# GOOD: Batched attention
weights = batched_attention(memory, keys)  # All heads at once

# Use efficient indexing for sparse access
# BAD: Python loop
selected = []
for b in range(B):
    selected.append(memory[b, indices[b]])

# GOOD: Advanced indexing
selected = torch.gather(memory, 1, indices.unsqueeze(-1).expand(-1, -1, W))
```

### Training Tips

1. **Gradient clipping** - Memory operations can have exploding gradients
2. **Curriculum learning** - Start with short sequences
3. **Memory initialization** - Small random values, not zeros
4. **Interface vector scaling** - Use layer norm on interface

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)

# Layer norm on interface
self.interface_layer = nn.Sequential(
    nn.Linear(hidden_size, interface_size),
    nn.LayerNorm(interface_size)
)
```

---

## 7. TRAIN STATION: Memory = Retention = History = Context

### The Core Unification

**Memory IS attention over time!**

```
MANN Memory Access    ≡  Transformer Attention  ≡  Context Window
     ↓                        ↓                        ↓
   w_t = softmax(q·K^T)    a = softmax(QK^T/√d)    Context retrieval
   r_t = w_t · M           out = a · V              Token retrieval
```

### Why This Is the Same Thing

**1. Memory = Key-Value Store**
```python
# DNC Memory
memory_matrix  # (N, W) - N slots of width W

# Transformer KV Cache
key_cache      # (T, d_k) - T timesteps
value_cache    # (T, d_v)

# Same structure! Keys → addresses, Values → content
```

**2. Attention = Content-Based Addressing**
```python
# DNC content addressing
sim = cosine_sim(query, memory)
weights = softmax(strength * sim)

# Transformer attention
sim = query @ keys.T / sqrt(d)
weights = softmax(sim)

# Identical! Just different normalization
```

**3. Temporal Links = Causal Masking**
```python
# DNC temporal links
# L[i,j] = 1 if wrote to i then j
forward_read = L.T @ current_weights

# Transformer causal mask
# mask[i,j] = 1 if i can attend to j (j <= i)
masked_attention = attention * causal_mask

# Both encode temporal order!
```

### The Deep Connection

| MANN Concept | Transformer Equivalent | Biological Analog |
|--------------|----------------------|-------------------|
| Memory matrix | KV cache | Synaptic weights |
| Write head | Write to KV cache | Long-term potentiation |
| Read head | Query attention | Memory retrieval |
| Temporal links | Causal mask / RoPE | Temporal coding |
| Usage vector | Token importance | Synaptic tagging |
| Allocation | Dynamic context | Memory consolidation |

### Transformers ARE Memory Networks

The transformer revolution can be understood as:

1. **Fixed temporal structure** - Position encodings replace learned links
2. **Parallel memory access** - All positions attend simultaneously
3. **Implicit write** - Every forward pass "writes" new KV
4. **Content-only addressing** - No location-based addressing

```python
# Transformer = DNC with specific choices
class TransformerAsMANN:
    """
    Transformer attention as memory-augmented network
    """

    def __init__(self, d_model, num_heads, max_len):
        # Memory size = context length
        self.memory_slots = max_len

        # Content addressing only
        self.content_attention = True
        self.location_attention = False

        # Fixed temporal structure
        self.temporal_links = "positional_encoding"

        # Parallel multi-head access
        self.num_read_heads = num_heads
        self.num_write_heads = num_heads  # Write = input projection
```

---

## 8. ARR-COC Connection: Relevance Memory

### Memory for Token Relevance (10%)

In ARR-COC-0-1, we need to track which tokens were relevant over time:

**Relevance Memory Architecture:**

```python
class RelevanceMemory(nn.Module):
    """
    Memory system for tracking token relevance

    Stores: which tokens have been useful for which goals
    Retrieves: relevance predictions for current tokens
    """

    def __init__(
        self,
        token_dim: int = 768,
        memory_slots: int = 256,
        slot_size: int = 128,
    ):
        super().__init__()

        # Token to memory projection
        self.key_proj = nn.Linear(token_dim, slot_size)
        self.value_proj = nn.Linear(token_dim, slot_size)

        # Memory matrix
        self.register_buffer(
            'memory',
            torch.zeros(memory_slots, slot_size)
        )
        self.register_buffer(
            'memory_usage',
            torch.zeros(memory_slots)
        )

        # Relevance output
        self.relevance_head = nn.Linear(slot_size, 1)

    def write_relevant_tokens(
        self,
        tokens: torch.Tensor,      # (B, T, D)
        relevance: torch.Tensor,   # (B, T) actual relevance
    ):
        """
        Write highly relevant tokens to memory
        for future retrieval
        """
        # Select top-k relevant tokens
        _, top_indices = torch.topk(relevance, k=16, dim=1)

        # Get tokens at these positions
        selected = torch.gather(
            tokens, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )

        # Project to memory format
        keys = self.key_proj(selected)     # (B, k, slot_size)
        values = self.value_proj(selected) # (B, k, slot_size)

        # Write using content-based allocation
        # ... (similar to DNC write)

    def read_relevance_prediction(
        self,
        tokens: torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:
        """
        Predict relevance based on memory

        Returns: (B, T) relevance scores
        """
        # Query memory
        queries = self.key_proj(tokens)  # (B, T, slot_size)

        # Content-based attention to memory
        similarity = torch.matmul(
            F.normalize(queries, dim=-1),
            F.normalize(self.memory, dim=-1).T
        )  # (B, T, N)

        # Soft read
        weights = F.softmax(similarity * 10, dim=-1)
        read_vectors = torch.matmul(weights, self.memory)  # (B, T, slot_size)

        # Predict relevance
        relevance = self.relevance_head(read_vectors).squeeze(-1)  # (B, T)

        return torch.sigmoid(relevance)
```

### Memory-Based Relevance Benefits

1. **Historical context** - "This token was relevant before in similar contexts"
2. **Pattern learning** - Implicit learning of relevance patterns
3. **Temporal consistency** - Relevance predictions informed by history
4. **Few-shot adaptation** - Quick memory updates for new domains

### Implementation Strategy

```python
class MemoryAugmentedRelevanceScorer(nn.Module):
    """
    Token relevance scoring with memory augmentation
    """

    def __init__(self, config):
        super().__init__()

        # Base scorer (feedforward)
        self.base_scorer = TokenRelevanceScorer(config)

        # Memory module
        self.memory = RelevanceMemory(
            token_dim=config.hidden_size,
            memory_slots=256,
            slot_size=128
        )

        # Combination
        self.combine = nn.Linear(2, 1)

    def forward(self, tokens, goal_embedding):
        # Base prediction
        base_relevance = self.base_scorer(tokens, goal_embedding)

        # Memory prediction
        memory_relevance = self.memory.read_relevance_prediction(tokens)

        # Combine
        combined = torch.stack([base_relevance, memory_relevance], dim=-1)
        final_relevance = self.combine(combined).squeeze(-1)

        return torch.sigmoid(final_relevance)

    def update_memory(self, tokens, actual_relevance):
        """Call after computing actual relevance"""
        self.memory.write_relevant_tokens(tokens, actual_relevance)
```

---

## Sources

### Primary References

**Neural Turing Machines:**
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) - Graves, Wayne, Danihelka (arXiv:1410.5401, 2014)
  - Original NTM paper with content and location-based addressing
  - Demonstrated copy, sort, recall tasks

**Differentiable Neural Computers:**
- [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101) - Graves et al. (Nature 2016)
  - Temporal link matrix and memory allocation
  - Graph traversal and question answering

### Implementation References

**PyTorch DNC:**
- [pytorch-dnc](https://github.com/ixaxaar/pytorch-dnc) - ixaxaar (accessed 2025-11-23)
  - DNC, SAM, and SDNC implementations
  - MIT licensed, production-ready code
  - Includes sparse access memory with FAISS

### Additional Resources

**Memory-Augmented Networks:**
- [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns) - Olah & Carter (Distill 2016)
  - Interactive visualizations of NTMs

**Sparse Memory:**
- [Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes](http://papers.nips.cc/paper/6298-scaling-memory-augmented-neural-networks-with-sparse-reads-and-writes.pdf) - Rae et al. (NIPS 2016)
  - SAM architecture for large-scale memory

**Improved Addressing:**
- [Improving Differentiable Neural Computers Through Memory Masking](https://openreview.net/pdf?id=HyGEM3C9KQ) - Csordas & Schmidhuber
  - DNC-DMS with better content matching

---

## Summary

Memory-augmented neural networks provide a principled way to separate computation from storage, enabling neural networks to learn algorithms and maintain long-term dependencies. The key insight is that memory access through attention IS the mechanism for temporal reasoning - and this same mechanism powers modern transformers.

**Key Takeaways:**

1. **MANNs = Neural Networks + External Memory + Attention**
2. **NTM pioneered content + location addressing**
3. **DNC added temporal links for sequence reasoning**
4. **SAM enables scaling to very large memories**
5. **Transformers are MANNs with parallel, content-only access**
6. **Memory = retention = history = context = attention over time**

For ARR-COC, memory augmentation provides a natural way to track relevance patterns across time, enabling more informed token allocation decisions based on historical context.

---

*Memory: The train station where past meets present to inform future*
