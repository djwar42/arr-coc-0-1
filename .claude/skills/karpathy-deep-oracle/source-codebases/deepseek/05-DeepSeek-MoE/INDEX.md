# DeepSeek-MoE - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepSeek-MoE
**Purpose**: Mixture of Experts architecture (671B total, 37B active)
**Paper**: DeepSeekMoE.pdf included

## Directory Structure

```
05-DeepSeek-MoE/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── DeepSeekMoE.pdf          # Technical paper ⭐
├── LICENSE-CODE
├── LICENSE-MODEL
├── requirements.txt
│
├── modeling_deepseek.py     # Main model implementation ⭐
│
├── finetune/                # Fine-tuning scripts
│   ├── finetune.py             # Main training script
│   ├── data_utils.py           # Data processing
│   └── configs/                # Training configs
│
└── images/                  # Architecture diagrams
```

## Key Concepts

### MoE Architecture
- **671B total parameters**: Full model size
- **37B active parameters**: Per-token compute
- **Fine-grained experts**: 2048 experts, top-k routing

### Expert Routing
- **Load balancing**: Auxiliary loss for even distribution
- **Expert capacity**: Maximum tokens per expert
- **Shared experts**: Always-active base experts

## Key Files

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `modeling_deepseek.py` | ~1000 | Full MoE model | DeepSeekMoE, MoEGate, experts |
| `finetune/finetune.py` | ~500 | Training loop | DeepSpeed, gradient checkpointing |
| `DeepSeekMoE.pdf` | - | Technical report | architecture, training details |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fine-tune
python finetune/finetune.py \
    --model_path deepseek-ai/deepseek-moe-16b \
    --data_path your_data.json
```

## Model Variants

| Model | Total Params | Active Params | Experts |
|-------|--------------|---------------|---------|
| 16B | 16.4B | 2.8B | 64 |
| 145B | 145B | 22B | 128 |
| 671B | 671B | 37B | 256 |

## Cross-References

**MoE overview**: `karpathy/codebases/05-deepseek-moe-overview.md`
**DeepSeek efficiency**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
**Related**: `09-DualPipe` (pipeline parallelism), `03-DeepEP` (expert parallelism)
