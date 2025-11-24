# ESFT (Efficient Supervised Fine-Tuning) - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/ESFT
**Purpose**: Efficient fine-tuning for MoE models
**Key Innovation**: Expert-aware selective fine-tuning

## Directory Structure

```
10-ESFT/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE-CODE
├── LICENSE-MODEL
│
├── esft.py                  # Core ESFT implementation ⭐
├── train.py                 # Main training script
├── train_ep.py              # Expert-parallel training
├── __init__.py
├── utils.py
│
├── deepseek/                # Model integration
│   └── modeling.py             # DeepSeek model patches
│
├── datasets/                # Data processing
├── configs/                 # Training configs
├── scripts/                 # Helper scripts
├── benchmarks.py            # Evaluation benchmarks
├── eval_multigpu.py         # Multi-GPU evaluation
└── results/                 # Benchmark results
```

## Key Concepts

### Efficient SFT for MoE
- **Expert selection**: Only fine-tune activated experts
- **Gradient routing**: Sparse backward pass
- **Memory efficient**: Reduced optimizer states

### Benefits
- **Faster training**: Skip inactive experts
- **Lower memory**: Only track active parameters
- **Better generalization**: Preserve expert specialization

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `esft.py` | Core implementation | expert selection, sparse |
| `train.py` | Training loop | HuggingFace Trainer |
| `train_ep.py` | Expert-parallel version | distributed |
| `benchmarks.py` | Evaluation suite | MMLU, HumanEval |

## Quick Start

```bash
# Standard training
python train.py \
    --model deepseek-ai/deepseek-moe-16b \
    --data your_data.json \
    --output_dir ./output

# Expert-parallel training
torchrun --nproc_per_node=8 train_ep.py ...

# Evaluation
python eval_multigpu.py --model ./output
```

## Cross-References

**MoE architecture**: `05-DeepSeek-MoE/INDEX.md`
**Expert parallel**: `03-DeepEP/INDEX.md`
**Efficiency analysis**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
