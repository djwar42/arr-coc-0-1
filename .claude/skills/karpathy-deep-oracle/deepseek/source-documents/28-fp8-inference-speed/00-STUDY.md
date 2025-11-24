# 33% Faster Inference with FP8 - Study

**Source**: Baseten (33% faster LLM inference with FP8 quantization)
**Date Processed**: 2025-10-28
**Category**: FP8 & Quantization (Inference Optimization)

---

## 📝 TL;DR

FP8 quantization speeds up inference by 33% with minimal accuracy loss. Post-training quantization (PTQ) converts FP16 → FP8 without retraining. Memory bandwidth bound operations benefit most. Practical deployment guide.

---

## 🎯 Key Points

### Performance Gains
- **Throughput**: +33% tokens/second
- **Memory**: 2x reduction (FP8 vs FP16)
- **Latency**: ~25-30% improvement
- **Accuracy**: <1% degradation on most tasks

### When FP8 Helps Most
- Memory-bandwidth-bound operations (KV cache reads)
- Large batch sizes
- Long context lengths
- GPU memory constrained deployments

### PTQ vs QAT
- **PTQ** (Post-Training Quantization): No retraining, fast to deploy, slightly lower quality
- **QAT** (Quantization-Aware Training): Requires retraining, better quality, what DeepSeek uses
- **Tradeoff**: PTQ is good enough for most deployments

---

## 💭 Karpathy Take

FP8 is basically free performance if your hardware supports it (H100, A100 80GB). The 33% speedup is real and the accuracy hit is minimal.

The key insight: most LLM inference is memory-bandwidth-bound, not compute-bound. FP8 halves memory traffic → faster inference. Simple.

PTQ is great for quick wins. QAT (what DeepSeek did for V3 training) is better but requires access to training infra. For most people deploying models, PTQ is the way to go.

---

## 🔗 Connections
- **05-fp8-lm-paper**: FP8 training (QAT)
- **01-deepseek-v3-technical-report**: V3 uses FP8 training
- **19-vllm-mla-fp8-optimization**: vLLM deployment with FP8
