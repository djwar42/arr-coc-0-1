# SGLang Multi-Token Prediction - Study
**Source**: LMSYS Org
**Date Processed**: 2025-10-28
**Category**: Model Architectures (Multi-Token Prediction)
## 📝 TL;DR
Accelerating SGLang inference with multi-token prediction. Predict N tokens at once instead of 1. Reduces sequential dependency. 1.5-2x speedup for code generation.
## 💭 Karpathy Take
Multi-token prediction (used in V3) speeds up inference by reducing autoregressive steps. Trade-off: more compute per step, fewer steps total. Net win for long sequences.
