# Understanding DeepSeek R1 Blog - Study

**Source**: Blog (Understanding DeepSeek R1 | Christian B. B. Houmann)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - Educational Overview

---

## 📝 TL;DR

Clear blog post explaining R1 fundamentals: multi-stage training (cold-start SFT → large-scale RL) + GRPO (Group Relative Policy Optimization). R1-Zero = pure RL, R1 = SFT+RL. Cost advantage: $0.14-0.55 per M tokens vs o1's $15. MIT licensed.

---

## 🎯 Key Points

- **R1 vs R1-Zero**: R1 uses cold-start SFT, R1-Zero is pure RL
- **GRPO**: Compares multiple outputs per prompt, no separate critic needed
- **Chain-of-Thought**: Thinks in `<think>` tags before answering
- **Cost**: 10-100x cheaper than OpenAI o1
- **Open**: MIT license, full training methodology published

---

## 🔗 Connections

- **04-deepseek-r1-paper**: Full technical paper
- **07-grpo-theory**: GRPO deep dive

---

## 💭 Karpathy Take

Good explainer for people who don't want to read the full paper. The cost comparison ($0.55 vs $15) is the killer feature - makes reasoning models actually usable at scale. Open source + cheap + good = game changer.
