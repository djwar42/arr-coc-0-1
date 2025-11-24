# KNOWLEDGE DROP: Bioelectric Computing

**Date**: 2025-11-23 20:30
**Source**: ml-morphogenesis/01-bioelectric-computing.md
**Topic**: Bioelectric computation without neurons

---

## Core Insight

Bioelectric networks are the evolutionary precursor to neural networks, using identical hardware (ion channels, gap junctions) to perform computation. The key insight is that **all computation is gradient descent** - whether in weight space (ML), morphospace (biology), or potential space (physics).

---

## Key Concepts

### 1. Bioelectric = Neural (Same Hardware)
```
Neural Network:    Ion Channels + Synapses + Neurotransmitters
Bioelectric Net:   Ion Channels + Gap Junctions + Signaling Molecules

Same components, different problem space:
- Neural: Navigate 3D behavioral space
- Bioelectric: Navigate anatomical morphospace
```

### 2. Voltage Patterns = Computation
- Spatial bioelectric patterns encode morphogenetic GOALS
- Patterns exist BEFORE structures form (prepatterns)
- Stable but rewritable (like memory)
- Can encode counterfactuals (what SHOULD be, not what IS)

### 3. Gap Junctions = Cognitive Glue
- Bidirectional (unlike neural synapses)
- Voltage-gated (activity-dependent)
- Enable collective intelligence emergence
- Scale cell-level goals to tissue-level goals

---

## TRAIN STATION: Gradient = Field = Potential

**THE UNIFICATION**:
```
Bioelectric Potential (Vmem)
    = Electric Potential (Physics)
    = Neural Activation (Neuroscience)
    = Morphogenetic Field (Dev Bio)
    = Free Energy (Active Inference)
```

All are gradient flows on energy landscapes!

---

## Code Pattern

```python
class BioelectricRelevance(nn.Module):
    """Compute relevance via iterative settling (like Vmem dynamics)"""

    def forward(self, x):
        relevance = self.initial_relevance(x)

        # Iterate until attractor (settled state)
        for _ in range(n_iterations):
            coupling = self.compute_coupling(x)
            delta = torch.einsum('bst,bt->bs', coupling, relevance)
            relevance = relevance + 0.1 * delta

        return torch.sigmoid(relevance)
```

---

## ARR-COC Connection

**Relevance = Voltage Potential in Information Space**

Traditional attention: ONE forward pass
Bioelectric-inspired: ITERATE until settling

Consider:
1. Initial coarse relevance
2. Iterative refinement via "gap junction" communication
3. Converge to stable relevance allocation

---

## Key Sources

- Manicka & Levin 2019 - BEN can compute logic gates
- Levin 2023 - Bioelectric networks as cognitive glue
- Mathews & Levin 2017 - Gap junctions in pattern regulation

---

## One-Line Summary

**Bioelectric computing shows that collective intelligence emerges from gradient dynamics on potential fields - the same principle underlying both neural networks and morphogenesis.**
