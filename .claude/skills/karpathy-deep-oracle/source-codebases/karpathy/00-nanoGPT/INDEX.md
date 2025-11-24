# nanoGPT - Source Codebase Index

**Repository**: https://github.com/karpathy/nanoGPT
**Philosophy**: Simplest, fastest GPT training. Prioritizes teeth over education.
**Size**: ~600 lines of core code

## Directory Structure

```
00-nanoGPT/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE                  # MIT
│
├── model.py                 # GPT model (~300 lines) ⭐
├── train.py                 # Training loop (~300 lines) ⭐
├── sample.py                # Inference/generation
├── bench.py                 # Benchmarking
├── configurator.py          # Config system
│
├── config/                  # Training configurations
│   ├── train_gpt2.py           # GPT-2 124M reproduction
│   ├── train_shakespeare_char.py
│   ├── finetune_shakespeare.py
│   └── eval_gpt2*.py           # Evaluation configs
│
├── data/                    # Dataset preparation
│   ├── shakespeare_char/       # Character-level Shakespeare
│   ├── shakespeare/            # BPE Shakespeare
│   └── openwebtext/            # OpenWebText for GPT-2
│
├── assets/                  # Images for README
├── scaling_laws.ipynb       # Scaling experiments
└── transformer_sizing.ipynb # Model sizing guide
```

## Key Files

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `model.py` | ~300 | GPT Transformer implementation | CausalSelfAttention, Block, GPT class |
| `train.py` | ~300 | Training loop with DDP | AdamW, gradient accumulation, checkpointing |
| `sample.py` | ~100 | Generation/inference | temperature, top_k, sampling |
| `configurator.py` | ~50 | Config override system | CLI args, config files |

## Quick Start

**3-minute Shakespeare (character-level):**
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

**GPT-2 reproduction (4 days on 8xA100):**
```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**Fine-tune from GPT-2:**
```bash
python train.py config/finetune_shakespeare.py
```

## Key Concepts

- **Plain and readable**: No abstractions, direct PyTorch
- **Hackable**: Easy to modify for experiments
- **Multi-GPU**: DDP support built-in
- **PyTorch 2.0**: torch.compile for 135ms/iter

## Cross-References

**Detailed analysis**: `karpathy/codebases/00-overview.md`
**Mathematical optimizations**: `karpathy/codebases/01-mathematical-optimizations.md`
**Compare with nanochat**: `01-nanochat/INDEX.md`
