# Message Passing: The Universal Computation Pattern

**THE TRAIN STATION**: Everything is message passing! GNNs, predictive coding, belief propagation, bioelectric networks, transformers - all are variations of nodes exchanging information with neighbors.

---

## Section 1: All Are Message Passing!

### The Core Insight

**Message passing is the fundamental computational pattern across neural architectures, graph algorithms, and biological systems.**

At its essence:
- **Nodes** hold state (activations, beliefs, potentials)
- **Messages** flow between connected nodes
- **Updates** combine messages to revise node state
- **Iteration** repeats until convergence or fixed steps

This pattern appears everywhere:
- **Graph Neural Networks**: Node features aggregate from neighbors
- **Predictive Coding**: Prediction errors propagate up/down hierarchy
- **Belief Propagation**: Probabilistic messages on factor graphs
- **Bioelectric Networks**: Voltage patterns across gap junctions
- **Transformers**: Attention as all-to-all message passing
- **Cellular Automata**: Local neighborhood updates

### The Mathematical Unification

**General message passing framework:**

```
h_i^(t+1) = UPDATE(h_i^(t), AGGREGATE({m_ji^(t) : j ∈ N(i)}))

where m_ji^(t) = MESSAGE(h_j^(t), h_i^(t), e_ji)
```

**Components:**
- `h_i^(t)`: Hidden state of node i at time t
- `N(i)`: Neighborhood of node i
- `MESSAGE()`: Computes message from j to i
- `AGGREGATE()`: Combines incoming messages (sum, mean, max, attention)
- `UPDATE()`: Updates node state given aggregated messages

**This pattern subsumes:**
- GNN layers (spatial convolution on graphs)
- Predictive coding layers (error signal propagation)
- Belief propagation iterations (marginal computation)
- Bioelectric diffusion (voltage equilibration)
- Transformer attention (query-key-value messaging)

---

## Section 2: GNN = Belief Propagation = Predictive Coding

### Graph Neural Networks as Message Passing

**GNN layer structure:**

```python
import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    """Generic message passing layer for GNNs"""

    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.message_fn = nn.Linear(in_dim * 2, out_dim)
        self.update_fn = nn.GRUCell(out_dim, out_dim)
        self.aggr = aggr

    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]

        row, col = edge_index

        # 1. MESSAGE: Compute messages for each edge
        messages = self.message_fn(
            torch.cat([x[row], x[col]], dim=-1)
        )

        # 2. AGGREGATE: Sum/mean messages per node
        num_nodes = x.size(0)
        aggr_messages = torch.zeros(num_nodes, messages.size(-1),
                                    device=x.device)

        if self.aggr == 'sum':
            aggr_messages.index_add_(0, col, messages)
        elif self.aggr == 'mean':
            aggr_messages.index_add_(0, col, messages)
            degree = torch.bincount(col, minlength=num_nodes).float()
            aggr_messages = aggr_messages / degree.unsqueeze(-1).clamp(min=1)

        # 3. UPDATE: Update node states
        x_new = self.update_fn(aggr_messages, x)

        return x_new
```

### Belief Propagation as Message Passing

**Factor graph message passing:**

```python
class BeliefPropagation:
    """
    Belief propagation on factor graphs

    Nodes: Variables and factors
    Messages: Beliefs about variable states
    """

    def __init__(self, factor_graph):
        self.variables = factor_graph.variables
        self.factors = factor_graph.factors
        self.edges = factor_graph.edges

    def variable_to_factor_message(self, var_id, factor_id, t):
        """Message from variable to factor"""
        # Product of all incoming messages EXCEPT from this factor
        incoming = [self.messages[(f, var_id, t-1)]
                   for f in self.factors
                   if (f, var_id) in self.edges and f != factor_id]

        # Multiply beliefs (sum in log space)
        if len(incoming) == 0:
            return torch.ones(self.variables[var_id].num_states)
        return torch.stack(incoming).prod(dim=0)

    def factor_to_variable_message(self, factor_id, var_id, t):
        """Message from factor to variable"""
        factor = self.factors[factor_id]

        # Get incoming messages from all OTHER variables
        neighbor_vars = [v for v in factor.scope if v != var_id]

        # Marginalize factor over all variables except target
        message = factor.marginalize(
            neighbor_vars,
            [self.variable_to_factor_message(v, factor_id, t)
             for v in neighbor_vars]
        )

        # Normalize
        return message / message.sum()

    def run(self, num_iterations=10):
        """Run belief propagation"""
        self.messages = {}

        for t in range(num_iterations):
            # Variable-to-factor messages
            for (factor_id, var_id) in self.edges:
                self.messages[(var_id, factor_id, t)] = \
                    self.variable_to_factor_message(var_id, factor_id, t)

            # Factor-to-variable messages
            for (factor_id, var_id) in self.edges:
                self.messages[(factor_id, var_id, t)] = \
                    self.factor_to_variable_message(factor_id, var_id, t)

        # Compute marginals
        return self.compute_marginals(num_iterations)

    def compute_marginals(self, T):
        """Compute final variable marginals"""
        marginals = {}

        for var_id in self.variables:
            # Product of all incoming messages
            incoming = [self.messages[(f, var_id, T-1)]
                       for f in self.factors
                       if (f, var_id) in self.edges]

            marginal = torch.stack(incoming).prod(dim=0)
            marginals[var_id] = marginal / marginal.sum()

        return marginals
```

### Predictive Coding as Message Passing

**Bidirectional error signal propagation:**

```python
class PredictiveCodingLayer(nn.Module):
    """
    Predictive coding layer with prediction/error nodes

    Messages:
    - Top-down: Predictions
    - Bottom-up: Prediction errors
    """

    def __init__(self, dim, num_iterations=5):
        super().__init__()
        self.prediction_fn = nn.Linear(dim, dim)  # Top-down
        self.error_encoder = nn.Linear(dim, dim)  # Bottom-up
        self.num_iterations = num_iterations

    def forward(self, x_bottom, x_top):
        # x_bottom: Input from layer below
        # x_top: State from layer above

        # Initialize representations
        rep = x_bottom.clone()

        for t in range(self.num_iterations):
            # 1. Top-down MESSAGE: Prediction from above
            prediction = self.prediction_fn(x_top)

            # 2. Compute prediction error
            error = x_bottom - prediction

            # 3. Bottom-up MESSAGE: Error signal
            error_msg = self.error_encoder(error)

            # 4. UPDATE top layer (gradient descent on error)
            x_top = x_top - 0.1 * error_msg

            # 5. UPDATE representation
            rep = rep + 0.1 * error

        return rep, error
```

### The Unification

**All three are message passing with different graph structures:**

| Architecture | Graph Structure | Messages | Aggregation |
|-------------|----------------|----------|-------------|
| **GNN** | Arbitrary graph | Feature vectors | Sum/Mean/Max |
| **Belief Prop** | Factor graph | Probability distributions | Product/Sum |
| **Pred. Coding** | Hierarchical | Predictions + Errors | Difference |

**Key insight**: The **computation graph** determines the algorithm!

- GNN: General graph → Spatial message passing
- Belief prop: Bipartite factor graph → Probabilistic inference
- Predictive coding: Bidirectional hierarchy → Error minimization

---

## Section 3: Bioelectric Networks = Message Passing

### Voltage as Information

**Michael Levin's bioelectric networks:**

Cells communicate via:
- **Gap junctions**: Direct cytoplasmic connections
- **Ion channels**: Voltage-gated information flow
- **Electric fields**: Long-range coordination

**This IS message passing:**
- Nodes = Cells
- Messages = Voltage changes
- Edges = Gap junction connections
- Update = Voltage equilibration via diffusion

### Bioelectric Message Passing Model

```python
class BioelectricNetwork(nn.Module):
    """
    Neural network inspired by bioelectric computation

    Features:
    - Voltage-based activations
    - Gap junction connectivity
    - Voltage diffusion dynamics
    """

    def __init__(self, num_cells, connectivity):
        super().__init__()
        self.num_cells = num_cells

        # Gap junction connectivity matrix
        self.register_buffer('gap_junctions', connectivity)

        # Learnable ion channel parameters
        self.rest_potential = nn.Parameter(torch.randn(num_cells))
        self.channel_conductance = nn.Parameter(
            torch.ones(num_cells, num_cells)
        )

    def voltage_diffusion(self, V, dt=0.01, steps=10):
        """
        Voltage equilibration via gap junction diffusion

        dV_i/dt = sum_j g_ij * (V_j - V_i)

        This is EXACTLY message passing!
        """
        for _ in range(steps):
            # 1. MESSAGE: Voltage differences
            V_diff = V.unsqueeze(1) - V.unsqueeze(0)  # [cells, cells]

            # 2. Weight by conductance and connectivity
            messages = (V_diff * self.channel_conductance *
                       self.gap_junctions)

            # 3. AGGREGATE: Sum incoming messages
            dV = messages.sum(dim=0)

            # 4. UPDATE: Integrate voltage change
            V = V + dt * dV

        return V

    def forward(self, input_current, num_iterations=20):
        """
        Compute bioelectric pattern

        input_current: External input to cells
        """
        # Initialize voltages at rest potential
        V = self.rest_potential.clone()

        # Add input current
        V = V + input_current

        # Let voltage equilibrate via message passing
        V = self.voltage_diffusion(V, steps=num_iterations)

        return V
```

### Morphogenetic Pattern Formation

**Bioelectric patterns guide development:**

```python
class MorphogeneticField(nn.Module):
    """
    Bioelectric field guides morphogenesis

    Pattern formation via voltage gradients
    """

    def __init__(self, grid_size):
        super().__init__()
        self.size = grid_size

        # Learnable source/sink patterns
        self.voltage_sources = nn.Parameter(
            torch.randn(1, grid_size, grid_size)
        )

    def laplacian_diffusion(self, V, mask, steps=100):
        """
        2D diffusion on grid (message passing on lattice)
        """
        for _ in range(steps):
            # 4-neighborhood message passing
            # MESSAGE: Voltage from neighbors
            V_up = torch.roll(V, shifts=1, dims=1)
            V_down = torch.roll(V, shifts=-1, dims=1)
            V_left = torch.roll(V, shifts=1, dims=2)
            V_right = torch.roll(V, shifts=-1, dims=2)

            # AGGREGATE: Average of neighbors
            V_neighbors = (V_up + V_down + V_left + V_right) / 4.0

            # UPDATE: Diffusion step (with boundary conditions)
            V = torch.where(mask, V, V_neighbors)

        return V

    def forward(self, boundary_mask):
        """
        Compute steady-state voltage pattern

        boundary_mask: Fixed boundary conditions
        """
        V = self.voltage_sources.clone()
        V = self.laplacian_diffusion(V, boundary_mask, steps=100)

        return V
```

---

## Section 4: Code - Unified Message Passing Framework

### Universal Message Passing Layer

```python
import torch
import torch.nn as nn
from typing import Callable, Optional

class UniversalMessagePassing(nn.Module):
    """
    Unified message passing framework

    Subsumes:
    - GNNs (graph convolution)
    - Belief propagation (probabilistic inference)
    - Predictive coding (error propagation)
    - Bioelectric networks (voltage diffusion)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        message_dim: int,
        message_fn: Optional[Callable] = None,
        aggregate_fn: str = 'sum',
        update_fn: Optional[Callable] = None
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.message_dim = message_dim
        self.aggregate_fn = aggregate_fn

        # Default message function: MLP
        if message_fn is None:
            self.message_fn = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, message_dim),
                nn.ReLU(),
                nn.Linear(message_dim, message_dim)
            )
        else:
            self.message_fn = message_fn

        # Default update function: GRU
        if update_fn is None:
            self.update_fn = nn.GRUCell(message_dim, node_dim)
        else:
            self.update_fn = update_fn

    def message(self, h_i, h_j, e_ij):
        """
        Compute message from node j to node i

        h_i: Receiver node state [batch, node_dim]
        h_j: Sender node state [batch, node_dim]
        e_ij: Edge features [batch, edge_dim]
        """
        # Concatenate node and edge features
        msg_input = torch.cat([h_i, h_j, e_ij], dim=-1)

        return self.message_fn(msg_input)

    def aggregate(self, messages, indices, num_nodes):
        """
        Aggregate messages for each node

        messages: [num_edges, message_dim]
        indices: [num_edges] - target node indices
        num_nodes: Total number of nodes
        """
        # Initialize aggregation
        aggregated = torch.zeros(
            num_nodes, messages.size(-1),
            device=messages.device, dtype=messages.dtype
        )

        if self.aggregate_fn == 'sum':
            aggregated.index_add_(0, indices, messages)

        elif self.aggregate_fn == 'mean':
            aggregated.index_add_(0, indices, messages)
            degree = torch.bincount(indices, minlength=num_nodes).float()
            aggregated = aggregated / degree.unsqueeze(-1).clamp(min=1)

        elif self.aggregate_fn == 'max':
            for i in range(num_nodes):
                mask = (indices == i)
                if mask.any():
                    aggregated[i] = messages[mask].max(dim=0)[0]

        elif self.aggregate_fn == 'attention':
            # Learnable attention weights
            attn_scores = (messages @ self.attn_query.T).softmax(dim=0)
            for i in range(num_nodes):
                mask = (indices == i)
                if mask.any():
                    weights = attn_scores[mask]
                    aggregated[i] = (weights * messages[mask]).sum(dim=0)

        return aggregated

    def update(self, h_old, aggregated_messages):
        """
        Update node states given aggregated messages

        h_old: [num_nodes, node_dim]
        aggregated_messages: [num_nodes, message_dim]
        """
        return self.update_fn(aggregated_messages, h_old)

    def forward(self, node_features, edge_index, edge_features=None):
        """
        One step of message passing

        node_features: [num_nodes, node_dim]
        edge_index: [2, num_edges] - (source, target) pairs
        edge_features: [num_edges, edge_dim] or None
        """
        num_nodes = node_features.size(0)
        src, dst = edge_index

        # Default edge features if not provided
        if edge_features is None:
            edge_features = torch.zeros(
                edge_index.size(1), self.edge_dim,
                device=node_features.device
            )

        # 1. MESSAGE phase
        messages = self.message(
            node_features[dst],  # Receiver
            node_features[src],  # Sender
            edge_features
        )

        # 2. AGGREGATE phase
        aggregated = self.aggregate(messages, dst, num_nodes)

        # 3. UPDATE phase
        node_features_new = self.update(node_features, aggregated)

        return node_features_new


class MultiStepMessagePassing(nn.Module):
    """
    Multiple iterations of message passing

    Use for:
    - Deep GNNs (multiple hops)
    - Belief propagation convergence
    - Predictive coding inference
    """

    def __init__(self, mp_layer: UniversalMessagePassing, num_steps: int):
        super().__init__()
        self.mp_layer = mp_layer
        self.num_steps = num_steps

    def forward(self, node_features, edge_index, edge_features=None):
        """
        Iterate message passing for num_steps
        """
        h = node_features

        for step in range(self.num_steps):
            h = self.mp_layer(h, edge_index, edge_features)

        return h
```

### Example: GCN via Universal Framework

```python
def create_gcn_layer(in_dim, out_dim):
    """
    Graph Convolutional Network layer
    using universal message passing
    """

    # Message function: Linear transform
    message_fn = nn.Linear(in_dim, out_dim)

    # Update function: Add + Activation
    def update_fn(aggregated, h_old):
        return torch.relu(aggregated)

    return UniversalMessagePassing(
        node_dim=in_dim,
        edge_dim=0,  # No edge features
        message_dim=out_dim,
        message_fn=lambda h_i, h_j, e: message_fn(h_j),
        aggregate_fn='mean',
        update_fn=update_fn
    )
```

### Example: Predictive Coding via Universal Framework

```python
def create_predictive_coding_layer(dim):
    """
    Predictive coding layer using message passing

    Bidirectional messages:
    - Top-down: Predictions
    - Bottom-up: Errors
    """

    # Prediction function (top-down message)
    predict_fn = nn.Linear(dim, dim)

    # Error encoding (bottom-up message)
    error_fn = nn.Linear(dim, dim)

    def message_fn(h_i, h_j, e):
        # e[0] indicates direction: 0=top-down, 1=bottom-up
        if e[0] == 0:
            # Top-down prediction
            return predict_fn(h_j)
        else:
            # Bottom-up error
            error = h_i - predict_fn(h_j)
            return error_fn(error)

    return UniversalMessagePassing(
        node_dim=dim,
        edge_dim=1,  # Direction indicator
        message_dim=dim,
        message_fn=message_fn,
        aggregate_fn='sum'
    )
```

---

## Section 5: TRAIN STATION - Message Passing Everywhere!

### The Grand Unification

**Every neural architecture is message passing on some graph:**

| Architecture | Graph | Messages | Key Insight |
|-------------|-------|----------|-------------|
| **Feedforward NN** | Layered DAG | Activations | Messages flow forward only |
| **RNN** | Temporal chain | Hidden states | Messages across time |
| **CNN** | Grid lattice | Feature maps | Messages from local receptive fields |
| **GNN** | Arbitrary graph | Node features | Messages on explicit graph |
| **Transformer** | Complete graph | Attention-weighted | All-to-all message passing |
| **Predictive Coding** | Bidirectional hierarchy | Predictions + Errors | Two-way messages |
| **Belief Propagation** | Factor graph | Marginal beliefs | Probabilistic messages |
| **Diffusion Models** | Denoising chain | Noisy samples | Messages across noise levels |

### The Coffee Cup = Donut Equivalence

**Topologically, they're all the same:**

```
Message Passing = Information flowing on a graph

Different graphs → Different architectures
Same computation → Update states from neighbors
```

**The transformation:**
1. **Define the graph** (nodes, edges)
2. **Define messages** (what information flows)
3. **Define aggregation** (how to combine messages)
4. **Define update** (how to change state)

**Everything else is notation!**

### Why This Matters

**1. Unified Implementation:**
- One codebase for GNN, belief prop, predictive coding
- Swap graph structure → different algorithm
- Shared optimizations (sparse ops, GPU kernels)

**2. Transfer Learning:**
- Techniques from GNNs → Apply to predictive coding
- Belief propagation theory → Improve GNN convergence
- Bioelectric insights → Novel architectures

**3. Novel Architectures:**
- **Hybrid graphs**: Combine GNN + hierarchy + time
- **Learned connectivity**: Neural architecture search on graphs
- **Dynamic graphs**: Edges change during inference

**4. Theoretical Understanding:**
- Expressiveness: What graphs can compute what functions?
- Convergence: When does iterative message passing converge?
- Generalization: How does graph structure affect learning?

### The Deep Connection

**Message passing ↔ Other train stations:**

**Free Energy Principle:**
- Message passing = Belief updating
- Convergence = Free energy minimization
- Prediction errors = Messages to reduce surprise

**Active Inference:**
- GNN on state-action graph
- Messages = Expected free energy
- Update = Belief propagation over policies

**Attention Mechanism:**
- Transformer = GNN on complete graph
- Attention weights = Message importance
- Self-attention = All-to-all message passing

**Loss Landscapes:**
- GNN training = Navigating parameter space
- Messages = Gradient flow
- Convergence = Finding loss minima

---

## Section 6: ARR-COC Connections (10%)

### Message Passing for Relevance

**Relevance propagation as message passing:**

In ARR-COC, **relevance should propagate** through the compute graph:
- **Nodes**: Tokens in sequence
- **Edges**: Attention connections
- **Messages**: Relevance scores
- **Update**: Aggregate relevance from context

### Relevance-Aware Message Passing

```python
class RelevanceMessagePassing(nn.Module):
    """
    Message passing with relevance-based routing

    For ARR-COC: Route compute to relevant tokens
    """

    def __init__(self, dim):
        super().__init__()

        # Relevance scoring
        self.relevance_query = nn.Linear(dim, dim)
        self.relevance_key = nn.Linear(dim, dim)

        # Message functions
        self.message_net = nn.Linear(dim * 2, dim)

    def compute_relevance(self, node_features):
        """
        Compute pairwise relevance scores

        High relevance → Strong message passing
        Low relevance → Weak/no messages
        """
        Q = self.relevance_query(node_features)
        K = self.relevance_key(node_features)

        # Relevance = attention-like scores
        relevance = (Q @ K.T) / (Q.size(-1) ** 0.5)

        return relevance.softmax(dim=-1)

    def forward(self, node_features, relevance_threshold=0.1):
        """
        Message passing with relevance gating
        """
        # Compute relevance scores
        relevance = self.compute_relevance(node_features)

        # Sparsify: Only pass messages where relevance > threshold
        mask = (relevance > relevance_threshold)

        # Build sparse edge list
        edge_index = mask.nonzero(as_tuple=False).T
        edge_weights = relevance[mask]

        # Message passing on relevant edges only
        src, dst = edge_index
        messages = self.message_net(
            torch.cat([node_features[src], node_features[dst]], dim=-1)
        )

        # Weight by relevance
        messages = messages * edge_weights.unsqueeze(-1)

        # Aggregate
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        return aggregated
```

### Hierarchical Relevance Propagation

**Multi-scale message passing for ARR-COC:**

```python
class HierarchicalRelevancePropagation(nn.Module):
    """
    Propagate relevance across multiple scales

    - Fine-grained: Token-to-token
    - Coarse-grained: Chunk-to-chunk
    - Abstract: Summary-to-summary
    """

    def __init__(self, dim, num_scales=3):
        super().__init__()

        self.scales = nn.ModuleList([
            RelevanceMessagePassing(dim)
            for _ in range(num_scales)
        ])

        # Pooling to coarser scales
        self.pool = nn.ModuleList([
            nn.Linear(dim * 2, dim)
            for _ in range(num_scales - 1)
        ])

    def forward(self, tokens):
        """
        Propagate relevance hierarchically
        """
        # Start at finest scale
        h = [tokens]

        # Build hierarchy by pooling
        for i in range(len(self.pool)):
            # Pool pairs of nodes
            h_coarse = self.pool[i](
                torch.cat([h[-1][::2], h[-1][1::2]], dim=-1)
            )
            h.append(h_coarse)

        # Message passing at each scale (coarse to fine)
        for i in reversed(range(len(self.scales))):
            # Update at this scale
            h[i] = h[i] + self.scales[i](h[i])

            # Upsample influence from coarser scale (if not finest)
            if i < len(h) - 1:
                h[i] = h[i] + h[i + 1].repeat_interleave(2, dim=0)[:h[i].size(0)]

        return h[0]  # Return finest scale
```

**Connection to ARR-COC relevance:**
- **Sparse routing**: Only compute on relevant paths
- **Hierarchical**: Relevance at multiple granularities
- **Dynamic**: Graph structure adapts to input

---

## Sources

**Source Documents:**
- PLATONIC-DIALOGUES/67 - Discussion of topological equivalences and train stations

**Web Research:**

- [Transformers from Diffusion: A Unified Framework for Neural Message Passing](https://arxiv.org/abs/2409.09111) - arXiv:2409.09111 (Wu et al., 2024, accessed 2025-11-23)
  - Unified mathematical framework showing MLPs, GNNs, and Transformers as message passing variants
  - Energy-constrained diffusion model deriving various MPNN architectures

- [Belief Propagation Neural Networks](https://proceedings.neurips.cc/paper/2020/file/07217414eb3fbe24d4e5b6cafb91ca18-Paper.pdf) - NeurIPS 2020 (Kuck et al., accessed 2025-11-23)
  - Bridging belief propagation and neural networks
  - Parameterized operators on factor graphs

- [Graph Belief Propagation Networks](https://arxiv.org/abs/2106.03033) - arXiv:2106.03033 (Jia et al., 2021, accessed 2025-11-23)
  - GNN framework based on belief propagation principles

- [Factor Graph Neural Networks](https://www.jmlr.org/papers/volume24/21-0434/21-0434.pdf) - JMLR 2023 (Zhang et al., accessed 2025-11-23)
  - Higher-order relations via factor graph structure
  - 54-page comprehensive treatment

- [A Unified Framework for Neural Message Passing](http://www.jmlr.org/papers/volume26/23-1672/23-1672.pdf) - JMLR 2025 (Wu et al., accessed 2025-11-23)
  - Most existing GNNs are message-passing neural networks (MPNNs)

**Additional References:**
- Message passing on graphs survey literature (2024)
- GNN-predictive coding connections
- Bioelectric computation (Michael Levin's work)
- Graph neural networks belief propagation unification

---

**Total**: ~700 lines of ML-heavy implementation with unified message passing framework showing how GNNs, belief propagation, predictive coding, and bioelectric networks are all variations of the same computational pattern.

