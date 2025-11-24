# Multi-Token Prediction - Study

**Source**: arXiv (Better & Faster Large Language Models via Multi-token Prediction)
**Date Processed**: 2025-10-28
**Category**: Model Architectures (Training Objective)

---

## 📝 TL;DR

**Multi-Token Prediction (MTP)** - Train models to predict multiple future tokens simultaneously (not just next token).

**Key benefits**:
- Better sample efficiency during training
- Improved performance on downstream tasks
- Enables speculative decoding for faster inference
- Used in DeepSeek-V3

**Trade-off**: More compute during training, but worth it for performance gains.

---

## 🎯 The Core Idea

### Standard Next-Token Prediction
```
Given: [w1, w2, w3]
Predict: w4
Loss: Cross-entropy on w4
```

### Multi-Token Prediction
```
Given: [w1, w2, w3]
Predict: w4, w5, w6  (next 3 tokens!)
Loss: Cross-entropy on w4 + w5 + w6
```

**Why this helps**: Forces model to learn longer-range dependencies and better planning.

---

## 📊 Results

**Training improvements**:
- Better sample efficiency (learn more per token)
- Improved generalization
- Better performance on reasoning benchmarks

**Inference benefits**:
- Can use for speculative decoding
- Faster generation without quality loss

**DeepSeek-V3 usage**:
- V3 uses MTP as training objective
- Contributes to strong benchmark performance
- Combined with other innovations (MoE, MLA, FP8)

---

## 💡 Key Insights

**On training objective**:
- Predicting multiple tokens = harder task
- Harder task → better learned representations
- Small compute overhead during training

**On speculative decoding**:
- MTP model can guess future tokens
- Validate guesses cheaply
- Speedup inference significantly

**On V3**:
- MTP is one piece of the puzzle
- Not revolutionary alone, but helps
- Combined with other innovations = powerful

---

## 🔗 Cross-References

**Used in**:
- DeepSeek-V3 (training objective)

**Connects to**:
- [V3 Technical Report](../01-deepseek-v3-technical-report/00-STUDY.md) - V3's use of MTP
- `knowledge-categories/model-architectures/03-multi-token-prediction.md`

---

**Last Updated**: 2025-10-28
**Status**: Core technique study
