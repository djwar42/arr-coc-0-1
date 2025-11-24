# KNOWLEDGE DROP: Mode Connectivity - The Hidden Pathways

**Date**: 2025-11-23 16:30
**Source**: Web Research + Garipov et al. 2018, Draxler et al. 2018, Frankle et al. 2020
**File Created**: ml-topology/02-mode-connectivity.md
**Lines**: ~700

---

## Core Knowledge Acquired

### Mode Connectivity Discovery (2018)

**The breakthrough**: Different trained neural networks (local minima) are NOT isolated - they're connected by simple curved paths along which loss remains nearly constant.

- **Linear interpolation**: Loss EXPLODES between two optima (reaching random init levels)
- **Curved path (Bezier/PolyChain)**: Loss stays nearly constant (< 5-10% variation)
- **Universal phenomenon**: Works for VGG, ResNet, WideResNet, Transformers, RL policies

**Papers**:
- Garipov et al. "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs" (NeurIPS 2018, 987 citations)
- Draxler et al. "Essentially No Barriers in Neural Network Energy Landscape" (ICML 2018)

### Linear Mode Connectivity (LMC)

**Stronger condition**: Even the STRAIGHT LINE between minima has constant loss

**When LMC holds**:
- Checkpoints from same training trajectory
- Lottery tickets from same initialization
- After permutation alignment (Git Re-Basin)

**When LMC fails**:
- Different random seeds from scratch
- Different hyperparameters
- Different architectures

### Key Methods Implemented

1. **Bezier Curve Training**: Parameterize path, minimize average loss along path
2. **Fast Geometric Ensembling (FGE)**: Cyclic LR to traverse low-loss manifold, collect models
3. **Loss Surface Simplexes**: Extend to volumes connecting 3+ solutions
4. **Git Re-Basin**: Permutation alignment to achieve LMC between any two networks

---

## TRAIN STATION: Connectivity = Topology = Homeomorphism

**The Grand Unification**:

```
TOPOLOGY                    =    LOSS LANDSCAPE
Path-connected space        =    Mode-connected minima
Homotopy equivalence        =    Continuous deformation of solutions
Homeomorphism               =    Equivalent representations (permutations)

PHYSICS                     =    NEURAL NETWORKS
Energy landscape            =    Loss surface
Transition paths            =    Mode connecting curves
Phase transitions           =    Sharp changes in generalization
```

**The Coffee Cup = Donut insight**: Just as they're topologically equivalent (one hole), different neural network solutions are equivalent - they're connected by continuous paths of equally good solutions!

---

## PyTorch Implementation Highlights

### Bezier Curve Path Finding

```python
class BezierCurve:
    def __init__(self, w1, w2, theta):  # endpoints + learnable bend
        self.w1, self.w2, self.theta = w1, w2, theta

    def __call__(self, t):
        # (1-t)^2 * w1 + 2t(1-t) * theta + t^2 * w2
        return {name: (1-t)**2*self.w1[name] + 2*t*(1-t)*self.theta[name] + t**2*self.w2[name]
                for name in self.w1}

# Train by minimizing E_{t~U[0,1]}[Loss(curve(t))]
```

### Fast Geometric Ensembling

```python
class FGE:
    def train_and_collect(self, train_loader, epochs=40, cycle_length=4):
        for epoch in range(epochs):
            lr = self.lr_max - (self.lr_max - self.lr_min) * (epoch % cycle_length) / cycle_length
            # ... train with cyclic LR ...
            if (epoch + 1) % cycle_length == 0:
                self.collected_models.append(copy.deepcopy(model.state_dict()))
```

---

## ARR-COC Connection: Multiple Relevance Solutions

**Key insight**: Relevance functions have mode connectivity too!

- Different relevance predictors with same performance are connected
- Enables: Relevance ensembling, relevance transfer, uncertainty-aware allocation
- High variance along mode = uncertain relevance = more conservative token allocation

---

## Performance Notes

- **Curve training**: Same cost as training one network (~200-600 epochs)
- **Visualization**: 100x100 grid = 10K evaluations, 1000x1000 = 1M evaluations
- **FGE**: Same training cost, K forward passes at inference for K models
- **Git Re-Basin**: Hungarian algorithm O(n^3), limit to <1000 neurons/layer

---

## Key Takeaways

1. Loss landscape is SIMPLER than we thought - one connected basin
2. Model averaging works BECAUSE of mode connectivity
3. Ensembling exploits diversity along connected manifold
4. Topology of loss landscape determines generalization
5. Permutation symmetry enables model merging (Git Re-Basin)

**The deep message**: We're not searching for needles in a haystack - we're exploring a connected manifold where all good solutions can reach each other.
