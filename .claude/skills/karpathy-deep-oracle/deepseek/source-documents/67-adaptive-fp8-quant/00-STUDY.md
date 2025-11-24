# Adaptive FP8 Quantization Methodology - Study

**Source**: JuSER (Novel adaptive quantization methodology for 8-bit floating-point DNN training)
**Date Processed**: 2025-10-29
**Category**: LOW - FP8 Research

---

## 📝 TL;DR

Research on adaptive quantization for FP8 training. "Adaptive" = adjust quantization dynamically during training based on tensor statistics. Alternative approach to fixed FP8 formats. Academic exploration of FP8 training strategies.

---

## 🔗 Connections

- **05-fp8-lm-paper**: DeepSeek's FP8 training
- **10-fine-grained-fp8**: Fine-grained FP8 details

---

## 💭 Karpathy Take

FP8 training research. The challenge: FP8 range is limited, so you need good scaling factors. "Adaptive" means adjusting scales dynamically vs fixed. Trade-off: adaptive = better quality, fixed = simpler hardware. DeepSeek likely uses carefully chosen fixed scales optimized per layer.
