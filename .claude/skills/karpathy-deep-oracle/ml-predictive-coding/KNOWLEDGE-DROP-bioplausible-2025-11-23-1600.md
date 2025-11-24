# KNOWLEDGE DROP: Biologically Plausible Backprop Alternatives

**Date**: 2025-11-23 16:00
**Source**: Web research + ML literature
**Target**: ml-predictive-coding/01-bioplausible-backprop.md

---

## What Was Created

Comprehensive guide (~700 lines) covering biologically plausible alternatives to backpropagation:

### Sections Created

1. **Why Backprop Isn't Biologically Plausible**
   - Weight transport problem
   - Derivative computation problem
   - Separate phase problem
   - Non-local credit assignment

2. **Target Propagation**
   - Core idea: propagate targets, not errors
   - GAIT-prop with exact equivalence to backprop
   - PyTorch implementation with learned inverses

3. **Feedback Alignment**
   - Random fixed feedback weights
   - Direct Feedback Alignment variant
   - Why it works: forward weights align with random feedback

4. **Equilibrium Propagation**
   - Energy-based view
   - Two phases: free and nudged
   - Contrastive Hebbian learning rule
   - Connection to STDP

5. **Local Hebbian Learning Rules**
   - Oja's rule (PCA)
   - BCM rule
   - Contrastive Hebbian
   - InfoNCE-based variants

6. **TRAIN STATION**: Local Rules = Hebbian = Self-Organization
   - All methods converge on local + contrastive
   - Free energy connection
   - Mathematical unification

7. **ARR-COC Connection**: Local Relevance Computation
   - Hebbian-learned attention
   - Relevance as free energy minimization
   - No global backprop needed

---

## Key Code Implementations

### Target Propagation Layer
```python
class TargetPropLayer(nn.Module):
    def __init__(self, in_features, out_features):
        # Forward pathway W
        # Inverse pathway V (learned)
    def compute_target(self, target_output):
        # Propagate target backward through learned inverse
    def local_update(self, target_input, lr):
        # Hebbian-like update based on target-actual difference
```

### Direct Feedback Alignment
```python
class DirectFeedbackAlignmentNetwork(nn.Module):
    def __init__(self, sizes):
        # Random fixed feedback matrices for each layer
    def dfa_backward(self, output_error):
        # Each layer gets direct feedback from output
```

### Equilibrium Propagation
```python
class EqPropNetwork(nn.Module):
    def energy(self, states, x, beta, target):
        # E + beta * C
    def relax(self, x, n_iters, beta):
        # Gradient descent on energy to equilibrium
    def train_step(self, x, y, beta):
        # Free phase -> Nudged phase -> Contrastive Hebbian update
```

### Local Hebbian Learning
```python
class HebbianLayer(nn.Module):
    def hebbian_update(self, lr):
        if rule == 'oja':
            delta_W = y.T @ (x - y @ W)  # PCA
        elif rule == 'bcm':
            delta_W = (y * (y - theta)).T @ x  # Sliding threshold
```

---

## TRAIN STATION Discovery

**The deep unification**:

```
LOCAL COMPUTATION + CONTRASTIVE LEARNING = GRADIENT DESCENT
```

All methods compute:
- Target Prop: local targets - local actual
- Feedback Align: random projection of global error
- Equilibrium Prop: nudged state - free state
- Hebbian: positive phase - negative phase

**Why this matters**: Credit assignment through local computation emerges from the fundamental requirement that learning be physically realizable!

---

## Performance Summary

| Method | MNIST | Bio-Plausibility |
|--------|-------|-----------------|
| Backprop | 99%+ | Low |
| Target Prop | 98% | Medium |
| Feedback Align | 98% | High |
| Equilibrium Prop | 97% | Very High |
| Hebbian | 95% | Very High |

---

## Sources Used

- Scellier & Bengio (2017) - Equilibrium Propagation
- Ahmad et al. (2020) - GAIT-prop
- Nokland (2016) - Direct Feedback Alignment
- Lillicrap et al. (2016) - Random feedback weights
- Laborieux et al. (2021) - Scaling to ConvNets

---

## ARR-COC Application

**Key insight**: Local relevance computation without global backprop!

Proposed approach:
1. Hebbian-learned attention for initial relevance
2. Equilibrium propagation for fine-tuning
3. Feedback alignment for allocation network

Benefits:
- No global backprop needed
- Memory efficient
- Online learning capable
- Neuromorphic-ready
