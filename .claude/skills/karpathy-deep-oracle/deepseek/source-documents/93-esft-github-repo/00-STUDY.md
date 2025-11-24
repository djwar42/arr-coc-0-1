# ESFT GitHub Repository - Study

**Source**: GitHub (deepseek-ai/ESFT: Expert Specialized Fine-Tuning - GitHub.md)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - Implementation / Training Code

---

## 📝 TL;DR

Official ESFT training code is out - fine-tune MoE models by only updating task-relevant experts instead of all of them. Saves up to 90% memory and 30% time. Accepted to EMNLP 2024. Comes with eval scripts, expert scoring, config generation, and training. MIT licensed. Pretty straightforward to use tbh.

---

## 🎯 Key Concepts

### Core Idea
**Problem**: Fine-tuning full MoE models wastes compute on irrelevant experts
**Solution**: Score each expert's relevance to the task, only train the top-p% that matter

### Pipeline (4 Steps)
1. **Evaluate**: Run model on task samples to see expert activation patterns
2. **Score**: Calculate which experts actually contribute to this specific task
3. **Config**: Generate expert selection config (e.g., top 20% of experts)
4. **Train**: Fine-tune only the selected experts with expert parallelism

### Key Scripts
- `eval_multigpu.py` - Evaluate model performance on datasets
- `get_expert_scores.py` - Calculate per-expert relevance scores
- `generate_expert_config.py` - Create training config for selected experts
- `train_ep.py` - Train with expert parallel (optimized multi-GPU)

---

## 💡 Why This Matters

**Efficiency Gains**: Why update 64 experts when only 12 matter for your task? ESFT identifies the relevant ones and ignores the rest. Result: 90% less memory, 30% faster training.

**Practical Deployment**: This isn't theoretical - it's production code. 708 GitHub stars, 260 forks, used by people actually deploying MoE models.

**Hyperparameters Matter**: `score_function` and `top_p` control expert selection. Trade-off between efficiency (fewer experts) and performance (more experts). Typical: top 20% works well.

---

## 🔧 Karpathy-Style Implementation Notes

```bash
# Install deps (pretty minimal)
pip install transformers torch safetensors accelerate

# Download pre-trained adapters
bash scripts/download_adapters.sh

# Evaluate on translation task
python eval_multigpu.py \
    --eval_dataset=translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=all_models/adapters/token/translation \
    --output_path=results/completions/token/translation.jsonl

# Get expert scores (which experts fire for this task?)
python scripts/expert/get_expert_scores.py \
    --eval_dataset=intent \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --output_dir=results/expert_scores/intent \
    --n_sample_tokens=131072 \
    --world_size=4 \
    --gpus_per_rank=2

# Generate expert config (select top 20% of experts)
python scripts/expert/generate_expert_config.py \
    --eval_dataset=intent \
    --expert_scores_dir=results/expert_scores/intent \
    --output_path=results/expert_configs/intent.json \
    --score_function=token \
    --top_p=0.2

# Train selected experts only
torchrun --nproc-per-node=8 train_ep.py \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --expert_config=results/expert_configs/intent.json \
    --train_dataset=intent \
    --save_opt_states \
    --train_config=configs/base.yaml \
    --output_dir=results/checkpoints/test/eval_intent
```

**Base Model**: Uses `deepseek-ai/ESFT-vanilla-lite` - a MoE model ready for ESFT training.

---

## 🔗 Connections

- **03-deepseek-moe-paper**: Core MoE architecture being fine-tuned
- **08-aux-loss-free-balancing**: Load balancing without aux loss (related to expert routing)
- **16-esft-marktech**: Higher-level explanation of ESFT concept
- **17-dynamic-expert-specialization**: Similar idea of task-specific expert selection

---

## 💭 Karpathy Take

This is what good open source ML looks like - paper + code + working examples. The 4-step pipeline makes sense: evaluate → score → select → train. No magic, just common sense applied to MoE fine-tuning.

The `top_p=0.2` hyperparameter is interesting... basically saying "20% of your experts are doing 80% of the work for this task." Pareto principle strikes again lol.

Expert parallel training is smart - if you're only updating 20% of experts, distribute them across GPUs efficiently. No point syncing gradients for frozen experts.

One question: How sensitive is performance to `top_p`? Like, is there a sharp cliff at 15% or graceful degradation? Would be cool to see a curve of accuracy vs. expert count... but that's what experiments are for ¯\_(ツ)_/¯

Pretty cool that this got into EMNLP 2024. Shows the community cares about practical efficiency, not just benchmark numbers.
