# Pure RL for Math Reasoning - Study

**Source**: RUG (No Supervision, No Problem: Pure Reinforcement Learning Improves Mathematical Reasoning in Small Language Models)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - RL Research

---

## 📝 TL;DR

Research on pure RL (no supervised data) for math reasoning in small models. Similar to R1-Zero approach (pure RL without cold-start SFT). Shows RL alone can discover reasoning without supervised examples. Math problems provide clear rewards (correct/incorrect).

---

## 🎯 Key Insight

Math is ideal for pure RL: verifiable answers = clear reward signal. Don't need supervised reasoning traces - model discovers them via trial-and-error. Related to R1-Zero's pure RL experiments.

---

## 🔗 Connections

- **04-deepseek-r1-paper**: R1-Zero uses pure RL
- **07-grpo-theory**: GRPO for RL training

---

## 💭 Karpathy Take

Interesting result: you CAN learn reasoning from scratch with RL if reward signal is clear. Math works because answers are checkable. For open-ended tasks, pure RL struggles (no clear reward). That's why R1 uses cold-start SFT for non-math domains - gives model initial strategy before RL refines it.
