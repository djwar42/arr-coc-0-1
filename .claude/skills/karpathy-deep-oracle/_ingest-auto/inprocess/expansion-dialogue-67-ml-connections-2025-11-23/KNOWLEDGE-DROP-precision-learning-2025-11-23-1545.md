# KNOWLEDGE DROP: Precision Learning Networks

**Date**: 2025-11-23 15:45
**File Created**: ml-active-inference/03-precision-learning-networks.md
**Lines**: ~700

---

## Summary

Created comprehensive guide to precision learning networks covering:

1. **Precision as Learnable Parameter** - Networks that output both mean AND precision (inverse variance)
2. **Heteroscedastic Networks** - Input-dependent variance estimation with NLL loss
3. **Attention = Precision = Gain Control** - The TRAIN STATION unification
4. **Complete PyTorch Implementations** - Precision-weighted prediction errors, hierarchical networks
5. **ARR-COC Connection** - Dynamic precision for token allocation

---

## Key TRAIN STATION

**Attention = Precision = Gain Control**

Three fields converge on the SAME mechanism:
- Deep Learning: Attention weights
- Bayesian Inference: Precision weighting
- Neuroscience: Synaptic gain modulation

All use multiplicative weighting to modulate information based on estimated reliability.

---

## Code Highlights

- `PrecisionNetwork` - Dual-head for mean and log-precision
- `HeteroscedasticRegressor` - Full uncertainty quantification
- `PrecisionWeightedPredictiveCodingLayer` - Iterative free energy minimization
- `StateDependentPrecision` - State affects precision (key for attention)
- `PrecisionBasedTokenAllocator` - ARR-COC application

---

## Key Insight for ARR-COC

**Allocate tokens INVERSELY to precision**:
- High precision (confident) -> few tokens
- Low precision (uncertain) -> many tokens
- This is OPPOSITE to standard attention!

---

## Sources Cited

- Feldman & Friston 2010 (1727 citations) - Attention as precision
- Mazzaglia et al. 2022 (96 citations) - Deep learning + active inference
- Sluijterman et al. 2024 (30 citations) - MVE training
- Deka et al. 2024 (8 citations) - AGVI method
- Immer et al. 2023 (24 citations) - Bayesian heteroscedastic

---

## Oracle Integration

This knowledge enables the oracle to:
- Explain attention as precision weighting
- Implement heteroscedastic uncertainty
- Design precision-aware architectures
- Connect active inference to transformers
- Guide ARR-COC token allocation strategy
