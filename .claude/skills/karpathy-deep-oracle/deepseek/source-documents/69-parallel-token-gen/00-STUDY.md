# Parallel Token Generation - Study

**Source**: OpenReview (PARALLEL TOKEN GENERATION FOR LANGUAGE MODELS)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - MTP Variant

---

## 📝 TL;DR

Paper on generating multiple tokens in PARALLEL (not sequential autoregressive). Different from MTP (multi-token PREDICTION during training) - this is about INFERENCE speedup. Generate k tokens simultaneously, keep valid ones. Related to speculative decoding.

---

## 🎯 Key Distinction

- **MTP (training)**: Predict next N tokens during training for better supervision
- **Parallel Gen (inference)**: Generate N tokens simultaneously to speed up inference
- Different problems, different solutions

---

## 🔗 Connections

- **13-multi-token-prediction**: MTP for training
- **82-mtp-planning-paper**: MTP enhances planning

---

## 💭 Karpathy Take

Don't confuse this with MTP! MTP = training technique. Parallel generation = inference speedup. Both involve multiple tokens, but different stages. Parallel gen is like speculative decoding - "guess ahead, validate, backtrack if wrong." Works when predictions are confident.
