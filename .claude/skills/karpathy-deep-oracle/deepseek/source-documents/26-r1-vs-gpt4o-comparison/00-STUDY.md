# R1 vs GPT-4o Comparison - Study

**Source**: arXiv (Comparative Analysis of OpenAI GPT-4o and DeepSeek R1 for Scientific Text Categorization)
**Date Processed**: 2025-10-28
**Category**: Model Comparison (Benchmarking)

---

## 📝 TL;DR

Empirical comparison of R1 vs GPT-4o on scientific text categorization using prompt engineering. R1 competitive with GPT-4o, sometimes better. Key finding: prompt engineering matters more than model choice for many tasks. R1 costs 95% less.

---

## 🎯 Key Findings

### Performance
- R1 matches or exceeds GPT-4o on most scientific text tasks
- Particularly strong on structured reasoning tasks
- GPT-4o slightly better on nuanced language understanding

### Cost
- R1: $0.14-0.55/M input, $2.19/M output
- GPT-4o: $2.50-5.00/M input, $10.00/M output
- ~95% cost reduction for R1

### Prompt Engineering Impact
- Well-designed prompts reduce model performance gap
- R1 benefits more from structured prompts (CoT-style)
- GPT-4o more robust to poorly-designed prompts

---

## 💭 Karpathy Take

Unsurprising results but good to have empirical validation. R1 competes with GPT-4o for way less money. For scientific text classification, reasoning models like R1 work well because you can structure the problem as step-by-step logic.

The prompt engineering finding is interesting: good prompts matter MORE than model choice. A well-prompted R1 beats a poorly-prompted GPT-4o.

Practical takeaway: For most structured tasks (classification, extraction, reasoning), use R1 and save 95%. For creative writing or nuanced language tasks, maybe GPT-4o is still worth it.

---

## 🔗 Connections
- **04-deepseek-r1-paper**: R1 technical details
- **22-understanding-r1-christian**: R1 explainer
- **23-grpo-multistage-training**: How R1 was trained
