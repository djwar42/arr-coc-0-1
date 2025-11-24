# nanochat Speedrun: $100 ChatGPT in 4 Hours

**The fastest way to feel the magic**

## Primary Sources

From [nanochat README](../../source-codebases/karpathy/01-nanochat/README.md):
- Complete speedrun walkthrough
- Efficiency philosophy and design
- Cost-performance tradeoffs

From [nanochat codebase](../../source-codebases/karpathy/01-nanochat/):
- speedrun.sh script
- Full pipeline implementation
- ~8K lines, 45 files total

---

## The Pitch

> "The best ChatGPT that $100 can buy." — Karpathy

**What nanochat is**:
- Full-stack LLM implementation in a single, clean, minimal, hackable, dependency-lite codebase
- Entire pipeline: tokenization → pretraining → finetuning → evaluation → inference → web serving
- Designed to run on a single 8×H100 node
- "Fully yours - fully configurable, tweakable, hackable, and trained by you from start to end"

**What you get for $100**:
- 1.9 billion parameter model (d32 depth)
- Trained on 38 billion tokens
- 4 hours on 8×H100 GPUs
- ChatGPT-like web UI to talk to your LLM

**Reality check**:
> "They make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing."

---

## Quick Start

Boot up an 8×H100 GPU box from your favorite provider (Karpathy uses [Lambda](https://lambda.ai/service/gpu-cloud)):

### Simple Version

```bash
bash speedrun.sh
```

### Production Version (Recommended)

Launch inside a screen session to survive disconnects:

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

**Screen tips**:
- Watch it go: Stay in the session
- Detach: `Ctrl-a d`
- Monitor: `tail speedrun.log`
- Reattach: `screen -r speedrun`

**Then wait 4 hours** ☕

---

## What Happens During the Speedrun

### Complete Pipeline (5 Core Stages)

**Stage 1: Tokenization (Rust-powered speed)**
- Custom BPE tokenizer built in Rust
- Vocabulary: 65,536 tokens
- Why Rust? "GPU-fast way faster than standard Python libraries"
- Training uses Rust tokenizer
- Inference switches to OpenAI's tiktoken (proven reliability)

> "Using Rust makes it GPU fast way faster than standard Python libraries for this step which really helps speed up getting the data ready for training"
>
> — *Source: [source-documents/00-...nanochat podcast](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Stage 2: Pre-training (Base model)**
- Learn raw language from massive web scrapes (FineWeb EDU)
- Goal: Grammar, facts, sentence structure, general world knowledge
- "Just reads a ton of text... Learn what language looks like"

**Stage 3: Mid-training (Conversational structure)**
- Shift gears: Feed conversational data (e.g., SmallTalk dataset)
- Introduce special tokens:
  - `<user>` - marks user messages
  - `<assistant>` - marks assistant responses
  - `<assistant_end>` - stop token (when to stop talking!)
- Teaches the structure of conversation: who's talking, when to respond

> "Here they shift gears They start feeding it conversational data... And crucially they introduce special chat tokens like user and assistant... teach the model the structure of a conversation Who's talking when to respond and importantly when to stop talking"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Bug note:** Early versions had a learning rate decay bug in mid-training script. Fixed now, but "highlights that even in these minimal systems tiny details in the training setup can really matter."

**Stage 4: SFT (Supervised fine-tuning)**
- Polish with high-quality curated examples
- Question-answer pairs
- Align behavior to be helpful and follow instructions

**Stage 5: RL (Optional - GRPO for skill boosts)**
- GRPO = Generalized Policy Optimization
- Optional but highly effective for specific skills
- Example: Math problem solving (GSM8K benchmark)

**Why RL specifically for math?**

> "SFT is good for style and factual recall from examples but complex reasoning like multi-step math problems benefits from the model getting feedback on the final answer RL methods like GRPO let the model try things out and get rewarded for correct solutions which helps it learn those step-by-step reasoning paths more effectively than just seeing examples"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**SFT → Style and facts**
**RL → Complex reasoning**

### Evaluation & Deployment

6. **Evaluation** - CORE, ARC, GSM8K, HumanEval, MMLU, ChatCORE
7. **Web UI deployment** - Serve ChatGPT-like interface

### Result

**Model specs**:
- d32 (32 layers in Transformer)
- 1.9B parameters
- 38B training tokens
- 4e19 FLOPs capability

**Performance**:
> "It's a bit like talking to a kindergartener :)" — Karpathy

---

## Architecture Deep Dive: The Core Files

### nanochat/gpt.py - The Transformer Brain

**Modern, high-performance components:**

**1. RMS Norm (instead of Layer Norm)**
- More stable during training
- Simpler computation
- Better for small models

**2. Rotary Positional Embeddings (RoPE)**
- Helps model handle sequence length better
- More flexible than fixed positional encodings
- Better extrapolation to longer sequences

**3. Grouped Query Attention (GQA)**
- Drastically cuts KV cache size
- Faster generation, lower memory
- Essential for making $100 speedrun practical!

> "GQA isn't that usually for massive models to save memory during inference Why put it in a nano chat that's actually a really smart choice GQA drastically cuts down the size of the key value tach needed during inference This means faster generation and lower memory use even for a smaller model"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Why GQA matters:** Advanced features enable minimalism to be practical!

**4. RoReLU² Activation Function**
- Not just ReLU, literally `F.relu(x).square()`
- "Simple effective nonlinearity"
- Unusual but works!

### nanochat/engine.py - Inference Engine

**Two-phase generation:**

**1. Prefill Phase (fast)**
- Process initial prompt
- Compute all KV pairs for context
- Happens once

**2. Decode Phase (streaming)**
- Generate response token by token
- Stream output to user
- KV caching essential here!

**KV Cache:**
- Stores calculated keys and values from previous tokens
- Avoids recomputing every single time
- "Essential for low latency streaming"

**Tool Use Integration:**
- Engine watches for special tokens like `<python_start>`
- Pauses generation
- Calls actual Python interpreter in safe sandbox
- Gets result
- Lets model continue with result

> "If the model outputs say Python start the engine pauses generation calls an actual Python interpreter in a safe sandbox to run the code the model generated gets the result and then lets the model continue"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**It can actually DO things, not just talk about them!**

### nanochat/config.py - Control Panel

**Your experimentation dashboard:**
- `depth` - Number of layers (e.g., 32 for d32)
- `n_head` - Number of attention heads
- `n_embd` - Embedding dimensions
- All tweakable without digging into gpt.py

**Scaling tip:** Want deeper model? Just change `depth` parameter!

### The No-Padding Strategy

**Problem with padding:**
- Sequences have different lengths
- Usually pad shorter ones with fake tokens
- Need complex attention masks ("ignore these")
- Wastes GPU computation on meaningless tokens

**nanochat solution: No padding at all!**
- Careful batching of real data only
- Attention masks cover real data only
- Special tokens (like `<assistant_end>`) signal stops clearly
- Strips out entire layer of complexity

> "It avoids padding entirely It relies on careful batching attention masks that only cover the real data and those special tokens like assistant tend to signal clearly where generation should stop It strips out that whole layer of complexity"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Result:** Simpler code, no wasted computation!

---

## Reproducibility: The Auto-Generated Report

### nanochat/report.py

**After every training run:**
- Automatically generates `report.md`
- Logs everything: environment, hyperparameters, metrics
- Makes comparing runs reliable

> "There's a nano chat report Script After a training run finishes it automatically spits out a report.md file... it logs everything Your environment details the exact hyperparameters you used key performance metrics"
>
> — *Source: [source-documents/00-...](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Why this matters:**
- No more "wait, what hyperparameters did I use?"
- Easy A/B testing of changes
- Full experimental provenance
- Science, not guessing!

---

## Serving Your LLM

Once training completes, activate your virtual environment:

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

**Access the UI**:
- Lambda users: Use public IP + port
- Example: `http://209.20.xxx.xxx:8000/`

**What to try**:
- Ask it to write stories or poems
- "Tell me who you are" → See hallucinations in action
- "Why is the sky blue?" → Get a real answer
- "Why is the sky green?" → Get creative fiction

**Remember**: "Speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener"

---

## The Report Card

After completion, check `report.md` for the full evaluation summary:

### Example Metrics

```
Total wall clock time: 3h51m

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |
```

**What this means**:
- BASE = Base model after pre-training
- MID = After midtraining
- SFT = After supervised fine-tuning
- RL = After RLHF (optional, may be missing)

---

## Scaling to Bigger Models

**The tiers**:
- **$100** → d32 model, 4 hours, GPT-1 level
- **$300** → d26 model, 12 hours, slightly above GPT-2
- **$1000** → 41.6 hours, "nice round number"

### Upgrading to d26 ($300 tier)

Only three changes needed in `speedrun.sh`:

**1. Download more data shards**:
```bash
# Get the number of parameters, multiply 20 to get tokens,
# multiply by 4.8 to get chars, divide by 250 million to get number of shards.
# todo need to improve this...
python -m nanochat.dataset -n 450 &
```

**2. Increase model depth, reduce batch size** (to avoid OOM):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
```

**3. Use same batch size during midtraining**:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

**That's it!**

---

## Memory Management: The OOM Game

**Key principle**: Adjust `device_batch_size` until things fit

**Default**: 32 (works on 80GB H100)

**If you OOM, reduce progressively**:
- Try 16 (good for larger models)
- Try 8
- Try 4
- Try 2
- Try 1
- Less than 1? "You'll have to know a bit more what you're doing and get more creative"

**What happens when you reduce batch size**:
> "Code automatically compensates by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute"

**Translation**: Same results, longer time. The math still works out.

---

## Computing Flexibility

### GPU Options

**Best**: 8×H100 (fastest)
- Speedrun: ~4 hours
- Cost: ~$24/hr

**Good**: 8×A100 (Ampere)
- "Will run just fine... but a bit slower"

**Works**: Single GPU
- Omit `torchrun` from commands
- "Will produce ~identical results"
- "You'll have to wait 8 times longer"
- Automatically switches to gradient accumulation

### Alternative Devices

**CPU / MPS** (Macbook):
- Possible but slow
- See `dev/runcpu.sh` for smaller configs
- "You're not going to get too far without GPUs"
- "At least you'll be able to run the code paths and maybe train a tiny LLM with some patience"

**Other backends** (xpu, mps, etc):
- "Most of the code is fairly vanilla PyTorch so it should run on anything that supports that"
- "But I haven't implemented this out of the box so it might take a bit of tinkering"

---

## Codebase Philosophy

### Stats

**Size**:
- ~330KB packaged
- ~8,304 lines
- 44 files
- ~83,497 tokens (fits in any modern LLM context)
- 2,004 uv.lock dependency lines

### Design Principles

**Minimal**:
- "Single, clean, minimal, hackable, dependency-lite codebase"
- No unnecessary abstractions
- Plain Python, well-commented

**Hackable**:
- "Very easy to hack to your needs"
- Designed to be forked and modified
- Educational over production-ready

**Self-contained**:
- Everything you need in one place
- Can package entire repo and ask LLM questions about it
- Works offline once data is downloaded

### File Structure Philosophy

```
nanochat/
├── adamw.py              # Distributed AdamW optimizer
├── checkpoint_manager.py # Save/Load model checkpoints
├── common.py             # Misc utilities, quality of life
├── configurator.py       # A superior alternative to argparse
├── dataloader.py         # Tokenizing Distributed Data Loader
├── engine.py             # Efficient model inference with KV Cache
├── gpt.py                # The GPT nn.Module Transformer
└── ...
```

**Note the naming**: Every file name tells you exactly what it does. No guessing needed.

---

## Customization

### Infusing Identity

**Guide**: [Infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)

**How**: Synthetic data generation + mixing into midtraining and SFT stages

**What you can tune**:
- Personality
- Speaking style
- Domain expertise
- Response patterns

### Adding Abilities

**Guide**: [Counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164)

**Examples**:
- Better reasoning
- Tool use
- Specific skills
- Domain knowledge

---

## Asking Questions About nanochat

### Method 1: Package and Feed to LLM

Use [files-to-prompt](https://github.com/simonw/files-to-prompt):

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh \
  --ignore "*target*" --cxml > packaged.txt
```

**Result**:
- ~330KB file
- Well below 100K tokens
- Copy-paste to Claude/GPT-4

### Method 2: DeepWiki

Visit the repo on [DeepWiki](https://deepwiki.com/):
- Change `github.com` to `deepwiki.com` in URL
- Ask questions interactively

**Karpathy's approach**: "I recommend using DeepWiki from Devin/Cognition to ask questions of this repo"

---

## What You're Actually Getting

### The Reality

**Performance tier**: GPT-1 to GPT-2 level (2019)

**Compared to modern LLMs**:
> "Falls dramatically short of modern Large Language Models like GPT-5"

**Behavior**:
- Makes mistakes frequently
- Naive and silly responses
- Hallucinates extensively
- "A bit like children. It's kind of amusing."

### The Value

**Not performance, but ownership**:
> "What makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end"

**Educational value**:
- Understand every step of the pipeline
- See exactly where compute goes
- Learn by modifying and experimenting
- No black boxes

**Philosophical point**:
- GPT-4 is amazing but opaque
- nanochat is modest but transparent
- "Teeth over education" — prioritizes what works, but makes learning accessible

---

## Cost-Performance Analysis

### The $100 Tier

**Cost breakdown**:
- 8×H100 @ $24/hr
- 4 hours runtime
- Total: ~$96

**What you get**:
- 1.9B parameters
- 38B training tokens
- Kindergarten-level ChatGPT
- Complete ownership

**ROI**: Educational value is immense; practical value is "amusing"

### The $800 Tier

**Setup**: Same node, longer training
- ~33 hours
- ~$800 total

**Result**:
- Outperforms GPT-2 (2019)
- Still dramatically short of GPT-4
- But yours end-to-end

### Going Beyond

**Multi-million dollar models** (GPT-4 territory):
- Not accessible via nanochat
- "LLMs are famous for their multi-million dollar capex"
- nanochat shows the principles at accessible scale

---

## Technical Deep Dive: The speedrun.sh Script

### What It Actually Does

**Stage 1: Data Preparation**
```bash
python -m nanochat.dataset -n 450 &  # Download training shards in background
```

**Stage 2: Tokenizer**
```bash
# Train BPE tokenizer on sample data
```

**Stage 3: Base Training**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=32 --device_batch_size=32
```

**Stage 4: Midtraining**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
  --device_batch_size=32
```

**Stage 5: SFT**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.sft_train
```

**Stage 6: RLHF** (optional)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.rl_train
```

**Stage 7: Evaluation**
```bash
python -m scripts.eval_core
python -m scripts.eval_arc
python -m scripts.eval_gsm8k
# ... etc
```

**Stage 8: Serve**
```bash
python -m scripts.chat_web
```

---

## Common Issues and Solutions

### OOM / Out of VRAM

**Solution**: Reduce `--device_batch_size`
- Start at 32
- Halve until it fits: 16 → 8 → 4 → 2 → 1

**Impact**: Training takes longer, results identical

### Not Enough Data Shards

**Symptom**: Code loops over same data (more epochs)
**Impact**: "Decreasing learning speed a bit"
**Solution**: Download more shards via `-n` parameter

### GPU Compatibility

**Less than 80GB VRAM**:
- Tune `device_batch_size` aggressively
- May need to reduce model size (`--depth`)

**Non-NVIDIA GPUs**:
- "Might take a bit of tinkering"
- PyTorch should support it
- Not tested out of the box

---

## Key Takeaways

**Efficiency**:
- 4 hours, $100 → Complete ChatGPT clone
- Single script, end-to-end
- No complicated infrastructure

**Reality**:
- Performance is kindergarten-level
- But that's still GPT-1/GPT-2 tier!
- Enough to be useful for learning

**Philosophy**:
> "Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints" — The Karpathy way

**The real win**:
- Understanding > performance
- Ownership > capability
- Transparency > black boxes

---

## Related Content

- [Four-Stage Pipeline](../training-llms/01-four-stage-pipeline.md) - Training stages explained
- [nanoGPT Comparison](../codebase/00-overview.md) - nanochat vs nanoGPT
- [Memory Management](02-memory-optimization.md) - OOM solutions
- [Customization Guide](03-customization.md) - Adding personality and abilities
