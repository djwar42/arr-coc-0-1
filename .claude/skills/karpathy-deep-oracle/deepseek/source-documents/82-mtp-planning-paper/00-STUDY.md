# Multi-Token Prediction for Planning - Study

**Source**: arXiv (Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - MTP Research

---

## 📝 TL;DR

Research paper on how MTP (Multi-Token Prediction) helps Transformers learn transitive relations for path planning. Shows MTP enables models to infer A→C from A→B and B→C. Proposes Next-Token Injection (NTI) and Transformer transfer layer. Validates on synthetic graphs + Blocksworld benchmark.

---

## 🎯 Key Concepts

- **Problem**: Standard Transformers fail transitive planning (can't infer A→C from separate A→B, B→C paths)
- **MTP Solution**: Predicting multiple future tokens provides richer supervision signals
- **Transfer Layer**: Gradually learns multi-step adjacency, enables transitive reachability
- **Enhancements**: NTI (Next-Token Injection), Transformer-based transfer layer

---

## 🔗 Connections

- **13-multi-token-prediction**: DeepSeek's MTP implementation
- **04-deepseek-r1-paper**: R1 uses improved planning via reasoning

---

## 💭 Karpathy Take

Cool theoretical analysis of WHY MTP helps planning. Most MTP papers say "it works better" without explaining mechanism. This one actually proves MTP learns compositional structure through transfer layers. Useful if you're designing planning systems.
