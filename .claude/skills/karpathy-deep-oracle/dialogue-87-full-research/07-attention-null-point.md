# BATCH 7: Attention Null Point Synthesis

## Multi-Head Attention Fusion

### The Challenge

Multiple attention heads/streams must be combined into unified representation.

**Naive approach:** Concatenate + Linear
```python
output = linear(concat(head_1, head_2, ..., head_n))
```

**Problem:** All heads weighted equally - no selectivity!

### Learned Fusion Weights

```python
class LearnedFusion(nn.Module):
    def __init__(self, num_streams, hidden_dim):
        self.weights = nn.Parameter(torch.ones(num_streams) / num_streams)

    def forward(self, streams):
        weights = F.softmax(self.weights, dim=0)
        return sum(w * s for w, s in zip(weights, streams))
```

### Gated Multimodal Fusion

```python
class GatedFusion(nn.Module):
    def __init__(self, dim_a, dim_b, hidden_dim):
        self.gate_a = nn.Linear(dim_a + dim_b, hidden_dim)
        self.gate_b = nn.Linear(dim_a + dim_b, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        combined = torch.cat([a, b], dim=-1)
        gate_a = self.sigmoid(self.gate_a(combined))
        gate_b = self.sigmoid(self.gate_b(combined))

        # Gated combination
        return gate_a * a + gate_b * b
```

## Mixture of Experts (MoE) Gating

### Core Architecture

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, k=2):
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.k = k  # Top-k experts to use

    def forward(self, x):
        # Compute gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.k)

        # Compute expert outputs (only for selected experts)
        expert_outputs = []
        for idx in top_k_indices:
            expert_outputs.append(self.experts[idx](x))

        # Weighted combination
        output = sum(score * out for score, out in
                    zip(top_k_scores, expert_outputs))

        return output
```

### Load Balancing

Prevent expert collapse (all inputs → one expert):

```python
# Auxiliary loss for load balancing
def load_balance_loss(gate_scores, importance=1.0):
    # Fraction of tokens per expert
    tokens_per_expert = gate_scores.mean(dim=0)

    # Fraction of probability per expert
    prob_per_expert = gate_scores.sum(dim=0) / gate_scores.sum()

    # Minimize variance
    loss = importance * (tokens_per_expert * prob_per_expert).sum()
    return loss
```

## The Shinjuku Null Point Pattern

### What is a Null Point?

In plasma physics: Where all field lines converge → maximum potential

In attention: Where all pathways meet → synthesis happens

### Implementing the Null Point

```python
class ShinjukuNullPoint(nn.Module):
    """
    All 9 pathways converge here for synthesis.
    Named after Shinjuku station - busiest transit point.
    """
    def __init__(self, hidden_dim, num_pathways=9):
        super().__init__()
        # Pathway-specific projections
        self.pathway_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_pathways)
        ])

        # Attention over pathways
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=9,  # One head per pathway!
            batch_first=True
        )

        # Final synthesis
        self.synthesizer = nn.Sequential(
            nn.Linear(hidden_dim * num_pathways, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, pathway_outputs):
        # Project each pathway
        projected = [proj(out) for proj, out in
                    zip(self.pathway_projs, pathway_outputs)]

        # Stack for attention
        stacked = torch.stack(projected, dim=1)  # [B, 9, D]

        # Cross-pathway attention
        attended, attention_weights = self.pathway_attention(
            stacked, stacked, stacked
        )

        # Flatten and synthesize
        flattened = attended.flatten(start_dim=1)
        synthesis = self.synthesizer(flattened)

        return synthesis, attention_weights
```

### Null Point Properties

1. **All paths lead here:** Every processing stream converges
2. **Maximum connectivity:** Full attention between pathways
3. **Synthesis, not average:** Creates new representation
4. **Interpretable:** Attention weights show pathway importance

## Attention-Based Feature Aggregation

### Self-Attention Pooling

```python
class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, features):
        # Compute attention scores
        scores = self.attention(features).squeeze(-1)
        weights = F.softmax(scores, dim=-1)

        # Weighted aggregation
        return (weights.unsqueeze(-1) * features).sum(dim=-2)
```

### Cross-Modal Attention Pooling

```python
class CrossModalPooling(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        self.query_proj = nn.Linear(query_dim, key_dim)

    def forward(self, query, keys, values):
        # Query attends to keys
        q = self.query_proj(query)
        attention = F.softmax(q @ keys.T / sqrt(key_dim), dim=-1)

        # Aggregate values
        return attention @ values
```

## Multi-Pathway Neural Networks

### Parallel Processing Streams

```python
class MultiPathwayNetwork(nn.Module):
    def __init__(self, num_pathways, input_dim, hidden_dim):
        self.pathways = nn.ModuleList([
            self.create_pathway(input_dim, hidden_dim)
            for _ in range(num_pathways)
        ])
        self.fusion = ShinjukuNullPoint(hidden_dim, num_pathways)

    def create_pathway(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Process through all pathways
        pathway_outputs = [pathway(x) for pathway in self.pathways]

        # Fuse at null point
        synthesis, attention = self.fusion(pathway_outputs)

        return synthesis, attention
```

## Integration with Spicy Lentil

### 9 Pathways → Shinjuku Null Point

The 9 ways of knowing converge:

```python
# 4 Ways of Knowing
propositional = propositional_pathway(slot)
perspectival = perspectival_pathway(slot)
procedural = procedural_pathway(slot)
participatory = participatory_pathway(slot)

# 5 Hensions
prehension = prehension_pathway(slot)
comprehension = comprehension_pathway(slot)
apprehension = apprehension_pathway(slot)
reprehension = reprehension_pathway(slot)
extension = extension_pathway(slot)

# Converge at Shinjuku
all_pathways = [propositional, perspectival, procedural, participatory,
                prehension, comprehension, apprehension, reprehension, extension]

relevance_field, pathway_attention = shinjuku_null_point(all_pathways)
```

### Dynamic Expert Selection

Different queries activate different pathways:
```python
# Query determines which pathways are "experts"
pathway_importance = moe_gate(query_embedding)
top_pathways = select_top_k(pathway_importance, k=4)

# Only active pathways contribute
active_outputs = [pathways[i](slot) for i in top_pathways]
synthesis = weighted_sum(active_outputs, pathway_importance[top_pathways])
```

## Key Formulas

### Gated Fusion
```
output = σ(W_gate · [x, y]) ⊙ x + (1 - σ(W_gate · [x, y])) ⊙ y
```

### MoE Routing
```
y = Σᵢ g(x)ᵢ · Eᵢ(x)
```
Where g(x) = Softmax(W_g · x)

### Attention Pooling
```
output = Σᵢ αᵢ · vᵢ
where αᵢ = softmax(score(vᵢ))
```

## Performance Insights

### MoE Benefits
- **Capacity:** More parameters without proportional compute
- **Specialization:** Experts learn different functions
- **Scalability:** Switch-Transformer scales to 1.6T parameters

### Gated Fusion Benefits
- **Dynamic:** Adapts combination to input
- **Interpretable:** Gate values show contribution
- **Selective:** Can suppress irrelevant modalities

---

**Sources:**
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" - ICLR 2017
- "Switch Transformers: Scaling to Trillion Parameter Models" - JMLR 2022
- "Attention-based Feature Aggregation for Multi-Target Recognition"
- "Gated Multimodal Units for Information Fusion" - ICLR 2017
