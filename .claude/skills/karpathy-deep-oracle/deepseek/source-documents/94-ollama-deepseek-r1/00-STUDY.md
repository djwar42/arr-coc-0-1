# DeepSeek-R1 on Ollama - Study

**Source**: Ollama Library (deepseek-r1 - Ollama.md)
**Date Processed**: 2025-10-29
**Category**: LOW - Tool Implementation / Deployment

---

## 📝 TL;DR

Ollama makes running DeepSeek-R1 locally stupidly easy - `ollama run deepseek-r1` and you're off. Models range from 1.5B (1.1GB) to 671B (404GB), all with 128K context. Distilled versions perform surprisingly well, proving you can teach small models to reason like big ones.

---

## 🎯 Key Concepts

### Model Sizes Available
- **1.5B - 70B**: Distilled from R1-671B reasoning traces
- **671B**: Full model, recently upgraded to R1-0528
- **Context**: 128K tokens (160K for 671B)
- **Downloads**: 68.3M total (pretty popular lol)

### Distillation Strategy
The R1 team distilled reasoning patterns from the 671B model into smaller Qwen/Llama models. Key insight: it's cheaper to learn reasoning from a smart teacher than discover it yourself via RL.

---

## 💡 Why This Matters

**Local Deployment Made Trivial**: No API calls, no rate limits, no internet required. Just `ollama run deepseek-r1:8b` and you've got an 8B reasoning model running on your laptop.

**Distillation Success**: Shows that reasoning isn't magic - it's a learnable pattern. Small models (1.5B!) can pick up reasoning behaviors from larger models through fine-tuning on 800k curated samples.

---

## 🔧 Karpathy-Style Implementation Notes

```bash
# Run default (8B distilled)
ollama run deepseek-r1

# Run full 671B model (if you have 404GB lying around)
ollama run deepseek-r1:671b

# Run tiny 1.5B model (actually pretty capable)
ollama run deepseek-r1:1.5b

# Update to latest version
ollama pull deepseek-r1
```

**Base Models Used for Distillation**:
- Qwen-2.5 series (1.5B, 7B, 8B, 14B, 32B) → Apache 2.0
- Llama 3.1-8B-Base → Llama 3.1 license
- Llama 3.3-70B-Instruct → Llama 3.3 license

All fine-tuned with 800k reasoning samples from R1-671B.

---

## 🔗 Connections

- **04-deepseek-r1-paper**: Core R1 architecture and training
- **07-grpo-theory**: How R1 learned to reason via RL
- **16-esft-marktech**: Expert fine-tuning (similar distillation concept)

---

## 💭 Karpathy Take

This is what open source AI should look like - dead simple deployment, multiple size options, MIT license for commercial use. The distillation results are honestly impressive... a 1.5B model that can reason? That's wild.

The fact that Ollama has 68M downloads across R1 models shows people want local inference. No vendor lock-in, no API costs piling up while you debug. Just download and run. ¯\_(ツ)_/¯

Coolest part: reasoning patterns are transferable. You don't need 671B parameters to think step-by-step - you just need good training data from a model that already knows how. Makes you wonder what else we can distill...
