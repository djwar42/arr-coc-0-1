# The Four-Stage LLM Training Pipeline

**Karpathy's recipe for training GPT assistants from scratch**

## Primary Sources

From [State of GPT](../../source-documents/karpathy/32-State of GPT _ BRK216HFS.md):
- Complete 4-stage pipeline breakdown
- Pre-training, SFT, reward modeling, RLHF
- Computational requirements and scaling

From [nanochat README](../../source-codebases/karpathy/01-nanochat/README.md):
- Practical implementation of full pipeline
- $100-$1000 tier training costs
- Real-world efficiency examples

---

## Overview: The Emerging Recipe

> "This is all very new and still rapidly evolving" — Karpathy

The current recipe for training GPT assistants looks like this:

```
┌─────────────────┐    ┌──────────┐    ┌──────────────┐    ┌──────┐
│ Pre-training    │ → │ SFT      │ → │ Reward Model │ → │ RLHF │
│ (99% compute)   │   │ (hours)  │   │ (hours)      │   │ (days)│
└─────────────────┘    └──────────┘    └──────────────┘    └──────┘
   Base Model         SFT Model       Reward Model       Assistant
```

**Key insight**: Pre-training is where ALL the computational work happens. The other three stages are fine-tuning stages that are "much more along the lines of a small few number of gpus and hours or days."

---

## Stage 1: Pre-training (The Big One)

**What it is**: Train a base model on internet-scale data to predict the next token

**Computational Reality**:
- 99% of training compute time and FLOPs
- Thousands of GPUs in a cluster
- ~1 month of training potentially
- Several million dollars (e.g., $2-3M for 65B model)

### Data Mixture

Example from Meta's LLaMA:

| Source          | Description                    |
|-----------------|--------------------------------|
| Common Crawl    | Web scrape (bulk of data)     |
| C4              | Cleaned Common Crawl          |
| GitHub          | Code repositories             |
| Wikipedia       | High-quality encyclopedic     |
| Books           | Long-form text                |
| arXiv           | Scientific papers             |
| Stack Exchange  | Q&A forums                    |

These are "mixed up together and then they are sampled according to some given proportions."

### Tokenization

Before training, text must be converted to integers:

- **Method**: Byte Pair Encoding (BPE) — "iteratively merges little text chunks and groups them into tokens"
- **Result**: Lossless translation between text and integer sequences
- **Vocabulary size**: ~50,000 tokens typically

### Scale Examples

**GPT-3** (3 years old as of 2023):
- Vocabulary: 50,257 tokens
- Context length: 2,048 tokens
- Parameters: 175 billion
- Training tokens: 300 billion

**LLaMA 65B** (more recent):
- Vocabulary: 32,000 tokens
- Context length: 2,048-4,096 tokens
- Parameters: 65 billion
- Training tokens: 1.4 trillion
- Training time: ~21 days on 2,000 GPUs

**Key insight**: "You shouldn't judge the power of a model just by the number of parameters that it contains." LLaMA 65B outperforms GPT-3 175B because it's trained for longer (1.4T vs 300B tokens).

### The Training Objective

**Simple**: Predict the next token in a sequence

**How it works**:
1. Lay out tokens in batches (B × T arrays)
2. For each position, look at all previous tokens
3. Predict what comes next
4. Update weights to make correct predictions more likely

**Visual analogy from New York Times GPT on Shakespeare**:

At initialization → Complete random gibberish
After training → "by the end you see that the Transformer has learned about words and where to put spaces and where to put commas"

### What Base Models Learn

> "These models basically in the process of language modeling learn very powerful General representations" — Karpathy

**Emergent capabilities**:
- Understanding text structure
- Multitasking (forced to handle many tasks in next-token prediction)
- Knowledge across vast domains
- Patterns, grammar, semantics

**But**: Base models are NOT assistants — they just want to "complete documents"

---

## Stage 2: Supervised Fine-Tuning (SFT)

**What it is**: Train the base model on high-quality prompt-response pairs

**Data requirements**:
- Small but HIGH quality datasets
- Tens of thousands of examples typically
- Human contractors following labeling guidelines
- Format: `[prompt, ideal response]` pairs

### Example Training Data

**Prompt**:
> "Can you write a short introduction about the relevance of the term 'monopsony'?"

**Ideal response** (written by human contractor):
> Follows extensive labeling documentation to be:
> - Helpful
> - Truthful
> - Harmless

### The Algorithm

**Still language modeling!** Nothing changed algorithmically — just swapping the training set:

- Before: Internet documents (high quantity, low quality)
- Now: QA prompt-response (low quantity, high quality)

### Result

After SFT, you get an **actual assistant** that will try to answer questions. These models work "to some extent" but can be improved further with RLHF.

---

## Stage 3: Reward Modeling

**What it is**: Train a model to score the quality of completions

**Data format shift**: From responses to **comparisons**

### How It Works

1. Take the SFT model
2. Generate multiple completions for each prompt (e.g., 3 completions)
3. Ask humans to rank them
4. Train a reward model to predict these rankings

### Example

**Prompt**: "Write a program that checks if a string is a palindrome"

**Three completions** generated by SFT model → Humans rank them

**Reward model training**:
- Lay out prompt + completion in rows
- Append special `[reward]` token at end
- Model predicts reward for that completion
- Train to match human rankings

**Note**: "These are very difficult things to do to compare some of these predictions and this can take people even hours for a single prompt completion pair"

### Output

A reward model that can score arbitrary completions for any prompt — kept fixed for next stage.

---

## Stage 4: Reinforcement Learning (RLHF)

**What it is**: Use the reward model to train the policy to generate high-scoring completions

### Process

1. Get large collection of prompts
2. Initialize policy at SFT model
3. Generate completions (yellow tokens)
4. Reward model scores each completion
5. Apply language modeling loss weighted by rewards

**Example**:
- Row 1: High reward (+2.5) → All tokens reinforced (higher probability)
- Row 2: Low reward (-1.2) → All tokens discouraged (lower probability)

### Result

An **RLHF model** (like ChatGPT) that generates completions scoring high according to the reward model.

---

## SFT vs RLHF: Why Bother?

**Simple answer**: "It just works better"

Humans prefer RLHF models > SFT models > prompted base models

### Why Does RLHF Work Better?

**Karpathy's hypothesis**: Asymmetry between generating vs comparing

**Example — Write a haiku about paper clips**:

As a contractor:
- **Generating**: Hard! You might not be good at writing haikus
- **Comparing**: Easy! You can appreciate good vs bad haikus

"This asymmetry makes it so that comparisons are a better way to potentially leverage yourself as a human and your judgment to create a slightly better model"

### Trade-offs

**RLHF models lose some entropy**:
- More "peaky" — less diverse outputs
- More consistent but less creative

**When to use base models**:
- Tasks requiring high diversity
- "N things → more things like it"
- Example: "Generate cool Pokemon names" (base model with high entropy works better)

---

## nanochat: Full Pipeline in Practice

**The speedrun**: Entire pipeline (tokenization → web serving) in one script

### The $100 Tier

**Setup**: 8×H100 GPUs at $24/hr
**Total time**: 4 hours
**Total cost**: ~$100

**One command**:
```bash
bash speedrun.sh
```

**What it does**:
1. Tokenizer training
2. Base model pre-training
3. Midtraining (domain adaptation)
4. SFT (supervised fine-tuning)
5. RLHF (reinforcement learning)
6. Evaluation (CORE, ARC, GSM8K, HumanEval, MMLU)
7. Web UI deployment

**Result**: 1.9B parameter model (d32 depth) trained on 38B tokens

> "It's a bit like talking to a kindergartener" — Karpathy on the $100 model

### The $800 Tier

**Setup**: Same 8×H100 node
**Time**: ~33 hours
**Cost**: ~$800

**Result**: "Enough to outperform GPT-2 of 2019" but still "falls dramatically short of modern Large Language Models like GPT-5"

### Efficiency Philosophy

**Karpathy's design**:
- "Single, clean, minimal, hackable, dependency-lite codebase"
- ~8K lines of code, 45 files
- Fully configurable and tweakable
- "Fully yours... trained by you from start to end"

**On model capabilities**:
> "They make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing."

---

## Scaling Laws

### How to Scale Up

From speedrun ($100) to bigger models:

**$300 tier** (d26 model, ~12 hours):
- Slightly outperforms GPT-2 CORE score
- Changes needed: More data shards, `--depth=26`, reduce batch size

**$1000 tier** (~41.6 hours):
- "Just because it's a nice round number"

### Memory Management

**Key principle**: Adjust `device_batch_size` until it fits:
- Default: 32
- If OOM: Try 16, 8, 4, 2, or 1
- "Code automatically compensates by increasing gradient accumulation loops, simply turning parallel compute to sequential compute"

### Computing Flexibility

**nanochat runs on**:
- 8×H100 (fastest, ~4 hours)
- 8×A100 (slightly slower)
- Single GPU (8× longer, omit `torchrun`)
- CPU / MPS (for tinkering, very slow)

"Most of the code is fairly vanilla PyTorch so it should run on anything that supports that"

---

## Key Takeaways

**Pre-training dominates**:
- 99% of compute
- Months, millions of dollars
- Where knowledge is learned

**Fine-tuning is cheap**:
- SFT: Hours, small GPUs
- Reward modeling: Hours
- RLHF: Days, small GPUs

**Practical implementation** (nanochat):
- Full pipeline in 4 hours for $100
- Results are "amusing" but functional
- Fully hackable and yours

**Philosophy**:
> "While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of modern Large Language Models... But what makes nanochat unique is that it is fully yours"

---

## Related Content

- [Pre-training Details](02-pretraining-deep-dive.md) - Token prediction mechanics
- [SFT Techniques](03-supervised-finetuning.md) - Dataset collection and labeling
- [RLHF Implementation](04-rlhf-explained.md) - Reward models and PPO
- [nanochat Speedrun](../practical-implementation/01-nanochat-speedrun.md) - Full walkthrough
