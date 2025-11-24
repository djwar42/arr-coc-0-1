# ESFT Paper - arXiv - Study

**Source**: arXiv (Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models)
**Date Processed**: 2025-10-29
**Category**: HIGH - Core ESFT Paper

---

## 📝 TL;DR

THE definitive ESFT paper on arXiv. "Let the Expert Stick to His Last" = train only task-relevant experts in MoE models. Core contribution: identify which experts matter for a task, freeze the rest. 90% memory reduction, 30% time savings. Accepted to EMNLP 2024.

---

## 🎯 Key Innovation

Traditional fine-tuning: update all parameters. ESFT: score experts by task relevance → train only top-k% → rest stay frozen. Result: massive efficiency gains with maintained performance.

---

## 🔗 Connections

- **93-esft-github-repo**: Implementation code
- **92-esft-vanilla-lite-hf**: Base model for ESFT
- **16-esft-marktech**: Higher-level ESFT explanation
- **03-deepseek-moe-paper**: MoE architecture being fine-tuned

---

## 💭 Karpathy Take

This is THE paper for ESFT. The title's a proverb: "shoemaker should stick to shoes" → experts should stick to what they're good at. Don't waste compute training irrelevant experts. Smart insight, practical impact. EMNLP acceptance shows community values efficiency research, not just benchmark chasing.
