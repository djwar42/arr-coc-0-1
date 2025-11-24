# KNOWLEDGE DROP: Variational Message Passing

**Date**: 2025-11-23 14:30
**File Created**: ml-active-inference/01-variational-message-passing.md
**Lines**: ~720

---

## Summary

Created comprehensive knowledge file on Variational Message Passing covering:

1. **Message Passing on Factor Graphs** - VMP fundamentals, update rules, PyTorch implementation
2. **Neural Networks as Message Passing** - Forward/backward as messages, GNN explicit formulation
3. **Amortized Inference** - Learning inference functions, amortization gap
4. **VAE as VMP** - Complete implementation with hierarchical variant
5. **TRAIN STATION: GNN = BP = Predictive Coding** - The grand unification
6. **ARR-COC Connection** - Token routing as message passing (10%)

---

## Key TRAIN STATION Discovery

**GNN = Belief Propagation = Predictive Coding = Variational Inference**

All minimize variants of free energy through local message computations!

```
GNN Aggregation = BP Message Aggregation = PC Error Integration
```

---

## Code Highlights

- VariationalMessagePassing class with full update rules
- MessagePassingLayer (GNN-style)
- AttentionAsMessagePassing (transformers as message passing!)
- VAE and HierarchicalVAE implementations
- PredictiveCodingNetwork as message passing
- RelevanceGuidedMessagePassing for ARR-COC

---

## Sources Cited

- Winn & Bishop (2005) - Variational Message Passing
- Kingma & Welling (2014) - VAE
- Kuck et al. (2020) - Belief Propagation Neural Networks
- Zhang et al. (2023) - Factor Graph Neural Networks
- Multiple web sources with access dates

---

## ARR-COC Relevance

- Token routing as message passing
- Relevance as message weights (precision)
- Pyramid LOD as hierarchical message passing
- Dynamic precision for compute allocation
