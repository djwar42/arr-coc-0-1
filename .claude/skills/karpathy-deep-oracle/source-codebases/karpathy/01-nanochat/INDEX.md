# nanochat - Source Codebase Index

**Repository**: https://github.com/karpathy/nanochat
**Philosophy**: "The best ChatGPT that $100 can buy" - Full-stack minimal hackable LLM
**Size**: ~8K lines of core code

## Directory Structure

```
01-nanochat/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── LICENSE                  # MIT
│
├── nanochat/                # Core package ⭐
│   ├── gpt.py                  # GPT Transformer
│   ├── tokenizer.py            # BPE wrapper (GPT-4 style)
│   ├── dataloader.py           # Distributed data loading
│   ├── adamw.py                # Distributed AdamW
│   ├── muon.py                 # Distributed Muon optimizer
│   ├── engine.py               # Inference with KV cache
│   ├── core_eval.py            # CORE score evaluation
│   └── *.py                    # Additional modules
│
├── scripts/                 # Pipeline scripts
│   ├── train_tokenizer.py      # Step 1: Tokenizer training
│   ├── train_base.py           # Step 2: Pretraining
│   ├── train_mid.py            # Step 3: Midtraining
│   ├── train_sft.py            # Step 4: SFT
│   ├── train_rl.py             # Step 5: RLHF
│   ├── eval_*.py               # Evaluation scripts
│   └── chat_web.py             # Web UI server
│
├── tasks/                   # Evaluation benchmarks
│   ├── arc_challenge/          # ARC-Challenge
│   ├── gsm8k/                   # GSM8K math
│   ├── hellaswag/               # HellaSwag
│   ├── humaneval/               # HumanEval code
│   └── mmlu/                    # MMLU
│
├── rustbpe/                 # Rust BPE tokenizer
│   ├── src/                    # Rust source
│   └── Cargo.toml              # Rust config
│
├── dev/                     # Development utilities
├── tests/                   # Test suite
├── speedrun.sh              # $100 speedrun script ⭐
├── run1000.sh               # Extended training
└── pyproject.toml           # Python package config
```

## Key Files

| File | Lines | Description | Keywords |
|------|-------|-------------|----------|
| `nanochat/gpt.py` | ~400 | GPT Transformer with RoPE | CausalSelfAttention, RMSNorm |
| `nanochat/tokenizer.py` | ~200 | BPE tokenizer wrapper | GPT-4 style, Rust backend |
| `nanochat/engine.py` | ~300 | Inference engine | KV cache, streaming |
| `scripts/train_sft.py` | ~500 | Supervised fine-tuning | conversation format |
| `scripts/train_rl.py` | ~600 | RLHF training | reward model, PPO |
| `speedrun.sh` | ~200 | Complete $100 pipeline | 4 hours, 8xH100 |

## Training Pipeline

1. **Tokenizer** (`train_tokenizer.py`) - Custom BPE on training data
2. **Pretraining** (`train_base.py`) - Base model on raw text
3. **Midtraining** (`train_mid.py`) - Domain adaptation
4. **SFT** (`train_sft.py`) - Conversation fine-tuning
5. **RLHF** (`train_rl.py`) - Reinforcement learning from feedback
6. **Evaluation** (`eval_*.py`) - CORE, ARC, GSM8K, HumanEval, MMLU

## Quick Start

**$100 Speedrun (4 hours on 8xH100):**
```bash
bash speedrun.sh
```

**Serve Web UI:**
```bash
python -m scripts.chat_web
# Visit http://YOUR_IP:8000/
```

**Extended Training ($800, 33 hours):**
```bash
bash run1000.sh
```

## Model Tiers

| Tier | Cost | Time | FLOPs | Performance |
|------|------|------|-------|-------------|
| d20 | $100 | 4h | ~4e19 | Basic chat |
| d26 | $300 | 12h | ~1e20 | Beats GPT-2 |
| d32 | $800 | 33h | ~3e20 | 1.9B params |

## Cross-References

**Detailed analysis**: `karpathy/codebases/00-overview.md`
**Mathematical optimizations**: `karpathy/codebases/01-mathematical-optimizations.md`
**Compare with nanoGPT**: `00-nanoGPT/INDEX.md`
