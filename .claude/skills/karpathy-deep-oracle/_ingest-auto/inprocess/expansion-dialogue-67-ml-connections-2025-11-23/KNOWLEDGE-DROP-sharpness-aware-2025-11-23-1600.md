# KNOWLEDGE DROP: Sharpness-Aware Minimization

**Date**: 2025-11-23 16:00
**File Created**: ml-topology/05-sharpness-aware.md
**Lines**: ~700

## Summary

Created comprehensive knowledge file covering Sharpness-Aware Minimization (SAM), the optimization algorithm that explicitly seeks flat minima for better generalization.

## Key Content

### 1. SAM Algorithm (Complete)
- Mathematical formulation of min-max optimization
- Step-by-step algorithm details
- Geometric intuition for why SAM works

### 2. PyTorch Implementation (Complete)
- Core SAM optimizer class (~100 lines)
- Adaptive SAM (ASAM) variant
- Full training loop with BatchNorm handling
- Gradient accumulation support
- Distributed training with DDP

### 3. Flatness and Generalization
- The flatness hypothesis explained
- Sharpness definitions (Hessian-based, SAM-based)
- Empirical evidence with benchmarks
- Scale-sensitivity problems and solutions

### 4. PAC-Bayes Connection
- PAC-Bayes framework overview
- How flat minima give tight bounds
- Information-theoretic interpretation (MDL)
- Formal bounds using Poincare inequalities

### 5. Sharpness Computation Code
- Full Hessian computation (small models)
- Power iteration for top eigenvalue (efficient)
- SAM-based sharpness measure

### 6. Advanced SAM Variants
- Efficient SAM (selective perturbation)
- LookSAM (periodic application)
- Momentum SAM (smoothed perturbation)

### 7. Performance Considerations
- 2x computational overhead
- Hyperparameter tuning (rho, base optimizer)
- When SAM helps most
- Memory-efficient implementation

### 8. TRAIN STATION (Core Unification)
**Sharpness = Curvature = Precision = Confidence**

All these are the same:
- Sharpness in loss landscape
- Hessian eigenvalues (curvature)
- Inverse posterior variance (precision)
- Certainty of predictions (confidence)
- Temperature in softmax
- Bits to encode parameters

### 9. ARR-COC Connection (10%)
- Relevance robustness through perturbation
- Precision-aware token allocation
- Flatness-aware multi-scale processing
- Training relevance models with SAM

## Sources

**Papers**:
- Foret et al. 2021 - Original SAM paper (arXiv:2010.01412)
- Kwon et al. 2021 - ASAM paper (arXiv:2102.11600)
- Haddouche et al. 2024 - PAC-Bayes flatness link (arXiv:2402.08508)
- Tsuzuku et al. 2020 - Normalized flatness (arXiv:1901.04653)

**Code**:
- https://github.com/davda54/sam (1.9k stars)
- https://github.com/moskomule/sam.pytorch

## Key Insight

**The flatness principle is universal**: Wherever you make predictions, prefer solutions robust to small perturbations. This applies to loss optimization (SAM), relevance computation (ARR-COC), and any decision-making under uncertainty.

The TRAIN STATION reveals that sharpness/curvature/precision/confidence are all manifestations of the same underlying concept - how certain we are about our predictions and how many bits are needed to specify them.
