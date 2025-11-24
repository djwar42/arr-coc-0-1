# Codebase Overview

Andrej Karpathy's two flagship LLM training repositories: nanoGPT and nanochat.

## Source Codebases

Located in `../source-codebases/`:
- **00-nanoGPT** - The simplest, fastest GPT training repository
- **01-nanochat** - Full-stack ChatGPT clone (tokenization → web UI)

---

## nanoGPT

**Philosophy**: Simplest, fastest repository for training/finetuning medium-sized GPTs. Prioritizes teeth over education.

**Core Files:**
- `train.py` (~300 lines) - Boilerplate training loop
- `model.py` (~300 lines) - GPT model definition
- `sample.py` - Sampling/inference script

**Capabilities:**
- Reproduces GPT-2 (124M) in ~4 days on 8xA100
- Character-level or BPE tokenization
- Finetuning from pretrained checkpoints
- Multi-GPU training with DDP
- PyTorch 2.0 compile support

**Quick Start:**
```bash
# Character-level Shakespeare (3 minutes on GPU)
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char

# GPT-2 reproduction (4 days on 8xA100)
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**Key Features:**
- Plain, readable code - easy to hack
- Loads GPT-2 weights from OpenAI
- Efficient: ~135ms/iter with torch.compile
- Supports CPU, GPU, Apple Silicon (MPS)

**Architecture:**
```
nanoGPT/
├── train.py           # Training loop
├── model.py           # GPT Transformer
├── sample.py          # Inference
├── config/            # Training configs
│   ├── train_gpt2.py
│   └── finetune_shakespeare.py
└── data/              # Dataset preparation
    ├── shakespeare/
    └── openwebtext/
```

---

## nanochat

**Philosophy**: "The best ChatGPT that $100 can buy" - Full-stack minimal hackable LLM training.

**Scope:** Complete pipeline from raw text to chatbot:
1. Tokenization (custom Rust BPE)
2. Pretraining (base model)
3. Midtraining (domain adaptation)
4. SFT (supervised fine-tuning)
5. RLHF (reinforcement learning)
6. Evaluation (multiple benchmarks)
7. Web UI (ChatGPT-like interface)

**Models:**
- **d20 ($100 tier)** - 4 hours on 8xH100, ~4e19 FLOPs
- **d26 ($300 tier)** - 12 hours, outperforms GPT-2
- **d32 ($800 tier)** - 33 hours, 1.9B parameters, 38B tokens

**Quick Start:**
```bash
# $100 speedrun (4 hours on 8xH100)
bash speedrun.sh

# Serve web UI
python -m scripts.chat_web
# Visit http://YOUR_IP:8000/
```

**Core Modules:**
```
nanochat/
├── gpt.py             # GPT Transformer
├── tokenizer.py       # BPE wrapper (GPT-4 style)
├── dataloader.py      # Distributed data loading
├── adamw.py           # Distributed AdamW optimizer
├── muon.py            # Distributed Muon optimizer
├── engine.py          # Inference with KV cache
├── core_eval.py       # CORE score evaluation
└── ui.html            # Web interface
```

**Training Scripts:**
```
scripts/
├── tok_train.py       # Train tokenizer
├── base_train.py      # Pretrain base model
├── mid_train.py       # Midtraining
├── chat_sft.py        # Supervised fine-tuning
├── chat_rl.py         # Reinforcement learning
├── chat_eval.py       # Evaluate on benchmarks
└── chat_web.py        # Serve web UI
```

**Evaluation Tasks:**
```
tasks/
├── arc.py             # Science questions
├── gsm8k.py           # Grade school math
├── humaneval.py       # Python coding
├── mmlu.py            # Multiple choice (broad)
└── smoltalk.py        # Conversational dataset
```

**Key Features:**
- Single cohesive minimal codebase (~8K lines, 45 files)
- Fully hackable - no framework complexity
- Custom Rust BPE tokenizer (fast training)
- Distributed training on 8xH100
- Complete report card with metrics
- CPU/MPS support for testing

**Example Report Card:**
```
| Metric          | BASE   | MID    | SFT    | RL     |
|-----------------|--------|--------|--------|--------|
| CORE            | 0.2219 | -      | -      | -      |
| ARC-Challenge   | -      | 0.2875 | 0.2807 | -      |
| ARC-Easy        | -      | 0.3561 | 0.3876 | -      |
| GSM8K           | -      | 0.0250 | 0.0455 | 0.0758 |
| HumanEval       | -      | 0.0671 | 0.0854 | -      |
| MMLU            | -      | 0.3111 | 0.3151 | -      |
| ChatCORE        | -      | 0.0730 | 0.0884 | -      |
```

---

## Codebase Comparison

| Aspect | nanoGPT | nanochat |
|--------|---------|----------|
| **Scope** | Pretraining only | Full pipeline (tok→deploy) |
| **Code Size** | ~600 lines core | ~8K lines (45 files) |
| **Goal** | Reproduce GPT-2 | Build ChatGPT clone |
| **Training** | Base models | Base + SFT + RL |
| **Eval** | Loss only | Multiple benchmarks |
| **Inference** | Simple sampling | Web UI + KV cache |
| **Cost** | $0-$10K+ | $100-$1000 |

**When to Use:**

**nanoGPT:**
- Learning GPT architecture basics
- Quick experiments with Transformers
- Finetuning existing models
- Research prototyping

**nanochat:**
- Building complete LLM application
- Understanding full training pipeline
- Low-budget ChatGPT training
- End-to-end hackable system

---

## Architecture Insights

Both repos share Karpathy's design philosophy:

**Principles:**
- ✅ **Minimal** - No unnecessary abstractions
- ✅ **Readable** - Plain Python, well-commented
- ✅ **Hackable** - Easy to modify and experiment
- ✅ **Practical** - Actually works on real hardware
- ✅ **Educational** - Clear implementation over flexibility

**Not Frameworks:**
- No giant config objects
- No model factories
- No if-then-else monsters
- Single cohesive codebase
- Maximally forkable

---

## References

**nanoGPT:**
- Repo: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Full source: `../source-codebases/00-nanoGPT/`

**nanochat:**
- Repo: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- Full source: `../source-codebases/01-nanochat/`
- Demo: [nanochat.karpathy.ai](https://nanochat.karpathy.ai/)
