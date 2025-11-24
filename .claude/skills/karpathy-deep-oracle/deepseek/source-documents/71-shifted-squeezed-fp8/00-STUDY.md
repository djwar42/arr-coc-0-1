# Shifted and Squeezed FP8 Format - Study

**Source**: Paper (SHIFTED AND SQUEEZED 8-BIT FLOATING POINT FORMAT FOR LOW-PRECISION TRAINING OF DEEP NEURAL NETWORKS)
**Date Processed**: 2025-10-29
**Category**: LOW - FP8 Format Variant

---

## 📝 TL;DR

Research on specific FP8 format variant - "shifted and squeezed" exponent/mantissa encoding. Alternative FP8 design to standard E5M2/E4M3. Likely NOT what DeepSeek uses (they use standard FP8 formats), but relevant for understanding FP8 design space.

---

## 🔗 Connections

- **05-fp8-lm-paper**: DeepSeek's FP8 training (standard formats)
- **10-fine-grained-fp8**: FP8 implementation details

---

## 💭 Karpathy Take

Academic FP8 format exploration. The FP8 design space is: how do you split 8 bits between exponent and mantissa? Standard is E5M2 or E4M3. This paper explores shifting/squeezing strategies. Interesting for format designers, not directly relevant to using DeepSeek's FP8 models.
