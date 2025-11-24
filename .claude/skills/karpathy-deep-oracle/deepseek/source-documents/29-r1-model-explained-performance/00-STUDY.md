# R1 Model Explained - Performance Analysis - Study

**Source**: (DeepSeek R1 Model Explained: How MLA and MoE Architectures Power Its Performance)
**Date Processed**: 2025-10-28
**Category**: DeepSeek Models (R1 Architecture Analysis)

---

## 📝 TL;DR

R1 performance breakdown: MLA (memory efficiency) + MoE (parameter efficiency) + GRPO (reasoning). 671B total params, 37B active. Matches o1 on reasoning benchmarks, 95% cheaper. Architecture enables performance.

---

## 🎯 Architecture → Performance

### MLA Impact
- 93% KV cache reduction
- Enables longer contexts at same memory
- 2.4x throughput improvement
- Critical for reasoning (long CoT sequences)

### MoE Impact
- 671B total, 37B active per token
- 18x parameter efficiency vs dense
- Cost: $5.5M training vs $50M+ for dense equivalent
- Maintains quality with sparse activation

### GRPO Impact
- Learned reasoning through RL
- No critic needed (simpler than PPO)
- 71% on AIME (vs 15.6% baseline)
- Reasoning emerges from relative rewards

### Combined Effect
MLA + MoE + GRPO = GPT-4o-level reasoning at 1/20th the cost

---

## 💭 Karpathy Take

This is the "why R1 works" explainer. Each architectural choice directly enables the performance:
- MLA → long CoT sequences fit in memory
- MoE → affordable to train 671B params
- GRPO → discovers reasoning without massive supervised data

It's not one trick, it's the system. Remove any piece and performance tanks.

---

## 🔗 Connections
- **04-deepseek-r1-paper**: Full R1 technical details
- **01-deepseek-v3-technical-report**: V3 base architecture
- **06-mla-explained**: MLA mechanics
- **03-deepseek-moe-paper**: MoE design
- **07-grpo-theory**: GRPO algorithm
