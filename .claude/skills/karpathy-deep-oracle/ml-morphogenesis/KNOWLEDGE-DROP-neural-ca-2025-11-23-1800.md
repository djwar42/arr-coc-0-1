# KNOWLEDGE DROP: Neural Cellular Automata

**Date**: 2025-11-23 18:00
**Source**: PART 19 - ML Morphogenesis Expansion
**File Created**: ml-morphogenesis/00-neural-cellular-automata.md

---

## Summary

Created comprehensive knowledge file on Neural Cellular Automata (NCA), covering:

1. **NCA Fundamentals** - Cell states, perception, update rules
2. **Differentiable CA Rules** - Sobel filters, stochastic updates, alive masking
3. **Growing Patterns** - Pool sampling, regeneration training
4. **Complete Implementation** - Full PyTorch code (~8k parameters)
5. **Performance Optimization** - GPU efficiency, quantization
6. **TRAIN STATION** - NCA = Morphogenesis = Self-Organization = Emergence
7. **ARR-COC Connection** - Self-organizing token relevance

---

## Key Insights

### The Core NCA Architecture

```python
# Only ~8,000 parameters!
class NeuralCA(nn.Module):
    def __init__(self, n_channels=16, hidden_dim=128):
        # Fixed perception (Sobel filters)
        # Learned update (small MLP)
        # Stochastic application
        # Alive masking
```

### The Morphogenesis Analogy

| Biology | NCA |
|---------|-----|
| Cell | Grid cell |
| DNA/Genome | Learned update rule |
| Chemical gradients | Sobel-perceived gradients |
| Morphogens | Hidden channels |
| Pattern formation | Loss minimization |

### TRAIN STATION Unification

**NCA reveals the deep equivalence:**
- NCA = Morphogenesis (biological pattern formation)
- NCA = Self-Organization (no central controller)
- NCA = Reaction-Diffusion (learned Turing patterns)
- NCA = Message Passing (GNN on grid)
- NCA = Free Energy Minimization (attractor dynamics)

The key insight: **The genome encodes local rules, not global patterns!**

---

## ARR-COC Application

**Self-Organizing Token Relevance:**

Instead of computing relevance centrally, let tokens self-organize:

```python
class RelevanceNCA(nn.Module):
    """Tokens iterate to find consensus on relevance."""

    def forward(self, x, steps=10):
        relevance = uniform_init()
        for _ in range(steps):
            # Perceive neighbors
            # Compute local update
            # Stochastic application
            # Normalize
        return relevance  # Emerged from local interactions!
```

Benefits:
- Robust to perturbations (like regeneration)
- Adapts to content complexity
- Parallel local computation
- Emergent global patterns

---

## Implementation Highlights

### Training Tricks

1. **Sample Pool**: Maintain pool of states, sample + update
2. **Worst Replacement**: Replace highest-loss with seed (prevent forgetting)
3. **Gradient Normalization**: Per-parameter L2 norm for stability
4. **Damage Training**: Circular damage for regeneration capability

### Performance Notes

- 64x64 grid, batch 32: ~100 MB GPU memory
- Single step: ~0.1ms inference
- Training: ~3 hours for 10k iterations
- Quantizable to 8-bit for deployment

---

## Code Patterns to Remember

### Perception

```python
# Sobel + Identity depthwise conv
perception = torch.cat([x, grad_x, grad_y], dim=1)
```

### Stochastic Update

```python
mask = (torch.rand_like(x[:, :1]) < 0.5).float()
x = x + update * mask
```

### Alive Masking

```python
alive = F.max_pool2d(alpha, 3, padding=1) > 0.1
x = x * alive.float()
```

---

## Sources

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca) - Distill 2020
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures) - Distill 2021
- [Official Colab](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)

---

## Integration Notes

This file establishes the foundation for the ml-morphogenesis directory. Key connections:

- **Next**: Bioelectric computing (01), GNN morphogenesis (02)
- **Related**: Message passing unification, self-organization
- **TRAIN STATION**: Central to morphogenesis = emergence = autopoiesis theme

The NCA paradigm is a powerful example of how simple local rules can produce complex global behavior - exactly the insight needed for understanding both biological development and distributed AI systems.
