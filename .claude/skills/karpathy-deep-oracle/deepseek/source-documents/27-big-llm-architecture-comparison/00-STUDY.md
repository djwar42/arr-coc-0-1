# Big LLM Architecture Comparison (Raschka) - Study

**Source**: Ahead of AI - Sebastian Raschka (The Big LLM Architecture Comparison)
**Date Processed**: 2025-10-28
**Category**: Model Comparison (Architecture Survey)

---

## 📝 TL;DR

Sebastian Raschka's comprehensive comparison of major LLM architectures: GPT, LLaMA, Gemini, DeepSeek-V3, etc. Compares attention mechanisms, MoE designs, context lengths, training approaches. DeepSeek-V3 notable for MLA + aux-loss-free balancing + FP8.

---

## 🎯 Key Comparisons

### Attention Mechanisms
- **GPT/LLaMA**: Standard multi-head attention
- **DeepSeek-V2/V3**: MLA (93% KV cache reduction)
- **Gemini**: Standard + sparse attention variants
- **Verdict**: MLA is the clear winner for memory efficiency

### MoE Architectures
- **GPT-4** (rumored): Coarse-grained experts
- **DeepSeek-V3**: Fine-grained + shared experts
- **Mixtral**: Standard TopK routing
- **Verdict**: DeepSeek's fine-grained + aux-loss-free approach is most sophisticated

### Context Length
- **GPT-4**: 128k tokens
- **Claude-3**: 200k tokens
- **DeepSeek-V3**: 128k tokens
- **Gemini**: 1M+ tokens (but expensive)
- **Verdict**: Gemini wins on length, MLA makes DeepSeek efficient at 128k

### Training Efficiency
- **DeepSeek-V3**: $5.5M for 671B params
- **LLaMA-3**: ~$10M+ for 70B params
- **GPT-4**: ???M (OpenAI won't say)
- **Verdict**: DeepSeek crushes on cost efficiency (FP8 + pipeline parallelism)

### Architecture Philosophy
- **GPT**: Dense, standard attention, proprietary
- **LLaMA**: Dense, standard attention, open weights
- **DeepSeek**: MoE + MLA + FP8, open everything
- **Verdict**: DeepSeek most innovative on architecture

---

## 💭 Karpathy Take

Raschka's comparisons are always good. This one confirms what we already knew: DeepSeek-V3 is architecturally ahead of most models.

The MLA advantage is real - 93% less KV cache means you can actually run these models without renting a datacenter. The aux-loss-free MoE balancing is clever (no load balancing loss term needed). And the FP8 training is how they got to $5.5M.

Other models (GPT, LLaMA, Gemini) are iterating on standard architectures. DeepSeek is pushing architectural boundaries: MLA, fine-grained MoE, FP8, multi-token prediction, etc.

GPT-4 is probably still better at some tasks (OpenAI has more RLHF data and proprietary techniques), but from an architecture standpoint, DeepSeek is more interesting.

The "open everything" philosophy helps too. GPT-4's architecture is a black box. DeepSeek publishes papers, code, weights. Better for science.

---

## 🔗 Connections
- **01-deepseek-v3-technical-report**: V3 architecture details
- **02-deepseek-v2-technical-report**: V2 architecture (MLA introduction)
- **03-deepseek-moe-paper**: Fine-grained MoE design
- **05-fp8-lm-paper**: FP8 training efficiency
- **06-mla-explained**: MLA mechanics
