# Llama 4 - Meta Multimodal Models - Study

**Source**: Meta AI (The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation - Meta AI.md)
**Date Processed**: 2025-10-29
**Category**: LOW - Not DeepSeek (Competitor Model)

---

## 📝 TL;DR

Meta's Llama 4 announcement - multimodal MoE models. Scout (17B active, 16 experts), Maverick (17B active, 128 experts), Behemoth (288B active, 16 experts). Compares to DeepSeek V3 on reasoning/coding. Not directly relevant to DeepSeek knowledge base, but useful for competitive context.

---

## 🎯 Key Concepts

### Three Models
1. **Scout**: 17B active, 16 experts, fits on 1x H100, 10M context
2. **Maverick**: 17B active, 128 experts, beats GPT-4o, comparable to DeepSeek V3
3. **Behemoth**: 288B active, 16 experts, outperforms GPT-4.5/Claude 3.7/Gemini 2.0 Pro

### Key Claims
- First open-weight natively multimodal models
- MoE architecture (like DeepSeek V3)
- Distilled from Behemoth teacher model
- Maverick: "comparable results to DeepSeek V3 on reasoning and coding—at less than half the active parameters"

---

## 💡 Why This Matters

**Competitive Landscape**: Meta directly compares Maverick to DeepSeek V3. Shows DeepSeek is now the benchmark others compare against.

**MoE Adoption**: Both Meta and DeepSeek converged on MoE + distillation strategy. Industry trend confirmed.

---

## 🔧 Karpathy-Style Implementation Notes

```python
# Llama 4 is open-weight on Hugging Face
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-Scout",  # or Maverick
    trust_remote_code=True
)

# Similar usage to DeepSeek models
```

Not much code here - it's a product announcement, not a technical paper.

---

## 🔗 Connections

- **01-deepseek-v3-technical-report**: V3 architecture that Llama 4 compares against
- **03-deepseek-moe-paper**: MoE fundamentals (both use similar approach)
- **04-deepseek-r1-paper**: Reasoning capabilities that Maverick matches

---

## 💭 Karpathy Take

This isn't a DeepSeek document lol. It's Meta's Llama 4 launch. But it's interesting BECAUSE they explicitly compare to DeepSeek V3. Quote: "achieving comparable results to the new DeepSeek v3 on reasoning and coding—at less than half the active parameters."

Translation: "We're as good as DeepSeek but smaller!" Classic competitive positioning.

Also notable: Llama 4 uses MoE (16 or 128 experts) just like DeepSeek. Everyone's converging on the same architecture - MoE + distillation + multimodal. The tricks that work, work for everyone.

Open-weight is cool - Meta's playing the same open source game as DeepSeek. Both betting that free/open beats closed/proprietary in the long run.

Anyway, this doc is here because it mentions DeepSeek as a benchmark. That's it. ¯\_(ツ)_/¯
