# BATCH 5: GNN Cognitive Fingerprint Research

## Graph Neural Networks Overview

GNNs process graph-structured data through message passing between nodes.

### Core Message Passing Framework

```python
# General GNN layer
def gnn_layer(node_features, edge_index, edge_features=None):
    messages = []
    for src, dst in edge_index:
        # Message computation
        msg = message_fn(node_features[src], node_features[dst], edge_features)
        messages.append((dst, msg))

    # Aggregation
    aggregated = aggregate(messages)  # sum, mean, max, attention

    # Update
    new_features = update_fn(node_features, aggregated)
    return new_features
```

## Key GNN Architectures

### Graph Attention Networks (GAT)

**Key Innovation:** Attention-weighted message aggregation

```python
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim * num_heads)
        self.a = nn.Linear(2 * out_dim, 1)  # Attention

    def forward(self, x, edge_index):
        # Transform features
        h = self.W(x).view(-1, num_heads, out_dim)

        # Compute attention scores
        src, dst = edge_index
        alpha = self.compute_attention(h[src], h[dst])
        alpha = softmax(alpha, dst)  # Softmax per node

        # Weighted aggregation
        return scatter_add(alpha * h[src], dst)
```

**Attention Formula:**
```
α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
```

### GraphSAGE (Sample and Aggregate)

**Key Innovation:** Inductive learning via neighborhood sampling

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator='mean'):
        super().__init__()
        self.W = nn.Linear(2 * in_dim, out_dim)
        self.aggregator = aggregator

    def forward(self, x, neighbors):
        # Aggregate neighbor features
        if self.aggregator == 'mean':
            neigh_agg = neighbors.mean(dim=1)
        elif self.aggregator == 'max':
            neigh_agg = neighbors.max(dim=1)[0]
        elif self.aggregator == 'lstm':
            neigh_agg = self.lstm(neighbors)

        # Concatenate and transform
        combined = torch.cat([x, neigh_agg], dim=-1)
        return F.relu(self.W(combined))
```

**Aggregation Options:**
- Mean: Simple average
- Max: Select maximum
- LSTM: Sequential processing
- Pooling: Learned attention

### Graph Convolutional Networks (GCN)

**Simplest GNN formula:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

Where:
- Ã = A + I (adjacency + self-loops)
- D̃ = degree matrix of Ã
- H^(l) = node features at layer l
- W^(l) = learnable weights

## Cognitive Fingerprint as Graph

### User Interest Graph Structure

**Nodes:**
- User's interests/topics
- Previously viewed content
- Query history patterns

**Edges:**
- Co-occurrence in sessions
- Semantic similarity
- Temporal proximity

```python
# Build cognitive fingerprint graph
class CognitiveGraph:
    def __init__(self, user_history):
        # Nodes: interests, queries, viewed items
        self.nodes = extract_concepts(user_history)

        # Edges: relationships between concepts
        self.edges = build_edges(
            co_occurrence=compute_cooccurrence(user_history),
            semantic=compute_similarity(self.nodes),
            temporal=compute_temporal_edges(user_history)
        )
```

### Graph-Based User Embedding

```python
class CognitiveFingerprint(nn.Module):
    def __init__(self, num_concepts, hidden_dim):
        super().__init__()
        self.concept_embed = nn.Embedding(num_concepts, hidden_dim)
        self.gnn = GAT(hidden_dim, hidden_dim, num_heads=4)
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, user_graph):
        # Initial node features
        x = self.concept_embed(user_graph.nodes)

        # Message passing
        x = self.gnn(x, user_graph.edges)

        # Global readout
        fingerprint = self.readout(x.mean(dim=0))
        return fingerprint
```

## Heterogeneous Graph Neural Networks

### Multi-Type Nodes and Edges

For cognitive fingerprints, we have different types:

**Node Types:**
- Query nodes (what user asked)
- Content nodes (what user viewed)
- Interest nodes (high-level concepts)

**Edge Types:**
- Query-Content (retrieval)
- Content-Interest (categorization)
- Interest-Interest (semantic)

```python
class HeteroGNN(nn.Module):
    def forward(self, hetero_graph):
        # Different transforms per node type
        query_h = self.query_encoder(hetero_graph.query_nodes)
        content_h = self.content_encoder(hetero_graph.content_nodes)
        interest_h = self.interest_encoder(hetero_graph.interest_nodes)

        # Type-specific message passing
        for edge_type in hetero_graph.edge_types:
            src_type, relation, dst_type = edge_type
            messages = self.message_fn[relation](...)
            # Aggregate per type
```

## Integration with Spicy Lentil

### GNN for Channel Weighting

The cognitive fingerprint graph determines which visual channels matter:

```python
# Cognitive fingerprint → channel weights
fingerprint = cognitive_gnn(user_graph)
channel_weights = channel_predictor(fingerprint)

# Apply to visual features
weighted_features = visual_features * channel_weights
```

### Graph-Based Relevance Field

The 9 pathways can be connected as a graph:
```python
# Pathway connectivity graph
pathway_graph = nn.Graph(
    nodes=[propositional, perspectival, procedural, participatory,
           prehension, comprehension, apprehension, reprehension, extension],
    edges=[
        (propositional, comprehension),  # Facts → Synthesis
        (perspectival, prehension),      # Salience → Grasping
        # ... more connections
    ]
)

# Message passing between pathways
pathway_features = pathway_gnn(pathway_features, pathway_graph)
```

### Dynamic Graph Evolution

As processing proceeds, the cognitive graph updates:
```python
# After each attention step
new_edges = detect_new_connections(current_output)
cognitive_graph.add_edges(new_edges)

# Re-process with updated graph
fingerprint = cognitive_gnn(cognitive_graph)
```

## Performance and Complexity

### Computational Complexity

| Method | Time | Space |
|--------|------|-------|
| GCN | O(|E| × d²) | O(|V| × d) |
| GAT | O(|E| × d² × H) | O(|V| × d × H) |
| GraphSAGE | O(|V| × k × d²) | O(|V| × d) |

Where: |V| = nodes, |E| = edges, d = dimension, H = heads, k = samples

### Benchmark Results

**Node Classification (Cora/Citeseer/PubMed):**
- GAT: 83.0% / 72.5% / 79.0%
- GraphSAGE: 78.9% / 68.5% / 76.4%
- GCN: 81.5% / 70.3% / 79.0%

## Key Implementation Details

### Aggregation Functions

```python
# Different aggregation strategies
def aggregate(messages, method='attention'):
    if method == 'mean':
        return messages.mean(dim=0)
    elif method == 'max':
        return messages.max(dim=0)[0]
    elif method == 'attention':
        weights = softmax(attention_score(messages))
        return (weights * messages).sum(dim=0)
    elif method == 'lstm':
        return lstm(messages)[-1]
```

### Readout for Graph-Level Representation

```python
def graph_readout(node_features, method='mean'):
    if method == 'mean':
        return node_features.mean(dim=0)
    elif method == 'sum':
        return node_features.sum(dim=0)
    elif method == 'attention':
        weights = softmax(attention(node_features))
        return (weights * node_features).sum(dim=0)
    elif method == 'hierarchical':
        # DiffPool or similar
        return hierarchical_pool(node_features)
```

## Recommendations for Spicy Lentil

1. **Use GAT:** Attention-based aggregation fits the relevance theme
2. **Heterogeneous graph:** Different node/edge types for rich structure
3. **Dynamic updates:** Graph evolves with processing
4. **Combine with FiLM:** Graph output → FiLM conditioning

---

**Sources:**
- "Graph Attention Networks" - ICLR 2018
- "Inductive Representation Learning on Large Graphs" (GraphSAGE) - NeurIPS 2017
- "Semi-Supervised Classification with Graph Convolutional Networks" - ICLR 2017
- "Modeling Relational Data with Graph Convolutional Networks" - ESWC 2018
