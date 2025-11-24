# DualPipe - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DualPipe
**Purpose**: Pipeline parallelism for MoE models
**Key Innovation**: Dual-direction pipeline scheduling

## Directory Structure

```
09-DualPipe/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE
├── setup.py
│
├── dualpipe/                # Main package ⭐
│   ├── __init__.py
│   ├── scheduler.py            # Pipeline scheduler
│   ├── pipeline.py             # Pipeline execution
│   └── utils.py                # Utilities
│
├── examples/                # Usage examples
│   ├── train_moe.py            # MoE training example
│   └── configs/                # Example configs
│
└── images/                  # Pipeline diagrams
```

## Key Concepts

### Dual-Direction Pipeline
- **Bidirectional scheduling**: Forward + backward overlap
- **Bubble reduction**: Minimize idle GPU time
- **MoE optimized**: Expert-parallel aware

### Benefits
- **Higher throughput**: Better GPU utilization
- **Scalable**: Efficient at 100s of GPUs
- **Memory efficient**: Activation checkpointing integration

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `dualpipe/scheduler.py` | Pipeline scheduling | 1F1B, interleaved |
| `dualpipe/pipeline.py` | Execution engine | stages, microbatches |
| `examples/train_moe.py` | MoE training | DeepSeek integration |

## Quick Start

```bash
# Install
pip install -e .

# Run example
python examples/train_moe.py --config configs/default.yaml
```

## Cross-References

**Expert parallel**: `03-DeepEP/INDEX.md`
**MoE architecture**: `05-DeepSeek-MoE/INDEX.md`
**Efficiency analysis**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
