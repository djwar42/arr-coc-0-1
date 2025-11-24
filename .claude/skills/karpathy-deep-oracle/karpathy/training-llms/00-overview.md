# Training LLMs - From Pre-training to RLHF

**The Complete Pipeline for Building ChatGPT-Style Models**

---

## The Four-Stage Recipe

> "This is all very new and still rapidly evolving but so far the recipe looks something like this... we have four major stages: pre-training, supervised fine-tuning, reward modeling, reinforcement learning"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

```
Stage 1: Pre-training (99% of compute)
    ↓
Stage 2: Supervised Fine-Tuning (hours/days)
    ↓
Stage 3: Reward Modeling (comparison data)
    ↓
Stage 4: Reinforcement Learning (align to preferences)
```

---

## Stage 1: Pre-Training (The Foundation)

**Goal:** Learn language from massive internet text

### The Data Pipeline: Download and Process the Internet

**Example: FineWeb Dataset (Hugging Face)**
- 44 terabytes of disk space
- 15 trillion tokens
- Could fit on a single large hard drive!

> "This company called hugging face uh collected and created and curated this data set called Fine web... ends up being only about 44 terabyt of dis space... this is not a huge amount of data at the end of the day"
>
> — *Source: [source-documents/27-Deep Dive into LLMs like ChatGPT.md]*

**Starting point: Common Crawl**
- Indexing internet since 2007
- 2.7 billion web pages (as of 2024)
- Crawlers follow links, index everything
- Raw HTML from the entire web

### Multi-Stage Filtering Process

**1. URL Filtering:**
- Block malware websites
- Filter spam and marketing sites
- Remove racist/adult content
- Eliminate low-quality domains

**2. Text Extraction:**
- Convert HTML → plain text
- Remove navigation, CSS, JavaScript
- Keep only valuable content
- Complex heuristics to get it right

**3. Language Filtering:**
- Detect language per page
- FineWeb: >65% English content
- Design decision affects multilingual performance
- Filtering Spanish → worse at Spanish later

**4. Deduplication:**
- Remove duplicate content
- Prevent memorization
- Reduce redundancy

**5. PII Removal:**
- Detect personal information
- Filter addresses, Social Security numbers
- Protect privacy

**Data Scale:**
- GPT-3: 300B tokens, 175B parameters
- LLaMA: 1.4T tokens, 65B parameters
- More tokens > More parameters

> "llama is a significantly more powerful model and intuitively that's because the model is trained for significantly longer... you shouldn't judge the power of a model just by the number of parameters"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Training objective:** Predict next token

```python
loss = 0
for position in sequence:
    prediction = model(sequence[:position])
    loss += cross_entropy(prediction, sequence[position])
```

**Cost:**
- GPT-3 (175B): Several million dollars
- LLaMA (65B): ~$2M, 21 days on 2000 GPUs

**Output:** **Base model** (document completer, not assistant)

---

## Base Models vs Assistants

**Base models** (GPT-2, LLaMA base):
- Complete documents
- Don't answer questions directly
- Require careful prompting
- Still have entropy (diverse outputs)

**Assistant models** (ChatGPT, GPT-4):
- Answer questions directly
- Follow instructions
- More helpful, harmless, honest (HHH)
- Less entropy (more focused)

> "base models are not assistants they don't want to answer questions they just want to complete documents"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## Stage 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach model to be a helpful assistant

**Data:** High-quality prompt-response pairs (tens of thousands)

**Example:**
```
Prompt: "Explain quantum computing to a 10-year-old"
Response: "Imagine a coin spinning in the air. While it spins,
          it's both heads and tails at once! Quantum computers
          use particles that can be multiple things at once..."
```

**Data collection:**
- Human contractors write ideal responses
- Follow labeling guidelines (helpful, truthful, harmless)
- Extensive documentation on tone, style, boundaries

**Training:** Continue language modeling on this curated data

```python
# Same objective, different data
loss = cross_entropy(model(prompt), ideal_response)
```

**Cost:** Much cheaper than pre-training (days, not months)

**Output:** **SFT model** (basic assistant)

---

## Stage 3: Reward Modeling

**Goal:** Learn what responses humans prefer

**Key insight:** **Comparing is easier than generating**

> "There's this asymmetry between how easy computationally it is to compare versus generate... judging which one of these is good is much easier task"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Data format:** Prompt + Multiple completions + Rankings

**Example:**
```
Prompt: "Write a function to check if string is palindrome"

Completion A: [working code, no comments]
Completion B: [working code, clear comments, docstring]
Completion C: [buggy code]

Ranking: B > A > C
```

**Training:**
- Take SFT model
- Generate multiple completions
- Humans rank them
- Train model to predict rankings

**Output:** **Reward model** (scores completion quality, not generative)

---

## Stage 4: Reinforcement Learning (RLHF)

**Goal:** Optimize model to generate high-reward completions

**Algorithm:** PPO (Proximal Policy Optimization)

**Process:**
1. Generate completions with SFT model (policy)
2. Score completions with reward model
3. Update policy to increase high-reward completions
4. Repeat

**Training:**
```python
for batch of prompts:
    completions = policy_model.generate(prompts)
    rewards = reward_model(prompts, completions)

    # Reinforce high-reward completions
    loss = -mean(rewards × log_probs(completions))
    policy_model.update(loss)
```

**KL penalty:** Keep model close to SFT version (prevent "reward hacking")

**Cost:** Days/weeks on smaller clusters

**Output:** **RLHF model** (ChatGPT, GPT-4)

---

## Why RLHF Works Better Than SFT

**SFT limitations:**
- Contractors may not be experts (can't write perfect haikus)
- Hard to write ideal responses for complex tasks
- Model limited by human demonstration quality

**RLHF advantages:**
- Humans can judge quality even if they can't create it
- Model explores and discovers better strategies
- Leverages human preferences, not just demonstrations

> "one answer that is kind of not that exciting is that it just works better... humans just prefer outputs from rlhf models"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## Trade-offs: Base vs SFT vs RLHF

| Metric | Base | SFT | RLHF |
|--------|------|-----|------|
| **Helpfulness** | Low | Medium | High |
| **Instruction following** | Poor | Good | Excellent |
| **Entropy (diversity)** | High | Medium | Low |
| **Cost** | Highest | Low | Medium |
| **Creative tasks** | Best | Good | Okay |

**When to use base models:**
- Need high diversity (brainstorming)
- Few-shot prompting
- Generate "more like this"

**When to use RLHF models:**
- Direct question answering
- Instruction following
- Production assistants

---

## Data Mixture Strategy

**Pre-training data composition** (example from LLaMA):
```
CommonCrawl:      67%  (web scrapes)
C4:               15%  (filtered web)
GitHub:            4.5% (code)
Wikipedia:         4.5% (high-quality)
Books:             4.5% (long-form)
ArXiv:             2.5% (academic)
StackExchange:     2%   (Q&A)
```

**Why this matters:**
- More code → better at programming
- More math → better at reasoning
- More diverse → better generalization

---

## Tokenization (The Pre-processing Step)

**Byte Pair Encoding (BPE):**
1. Start with characters
2. Iteratively merge frequent pairs
3. Build vocabulary (~50K tokens)

**Example:**
```
"understanding" → ["under", "stand", "ing"]
"Anthropic" → ["An", "throp", "ic"]
```

**Why it matters:**
- Common words = 1 token (efficient)
- Rare words = multiple tokens (flexible)
- Vocabulary size affects model size

---

## Training Hyperparameters (nanoGPT Example)

```python
# Model architecture
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 50257
block_size = 1024

# Training
batch_size = 12
learning_rate = 3e-4
max_iters = 600000
warmup_iters = 2000

# Regularization
weight_decay = 0.1
dropout = 0.1
grad_clip = 1.0
```

*Source: [source-codebases/00-nanoGPT/](../source-codebases/00-nanoGPT/)*

---

## Scaling Laws

**Key findings:**
- Loss scales predictably with compute
- More compute → lower loss → better model
- Optimal balance: model size vs training tokens

**Chinchilla scaling:**
- For N parameters, train on ~20N tokens
- GPT-3 (175B) trained on 300B tokens (undertrained!)
- LLaMA (65B) trained on 1.4T tokens (better!)

**Implication:** Training longer > Bigger model

---

## Distributed Training

**Data Parallel (DP):**
- Replicate model on multiple GPUs
- Each GPU processes different batch
- Sync gradients after backward pass

**Model Parallel (MP):**
- Split model layers across GPUs
- Pipeline forward/backward passes
- For models too large for single GPU

**Example (nanoGPT):**
```bash
# 8 GPUs, data parallel
torchrun --nproc_per_node=8 train.py config/train_gpt2.py
```

---

## Evaluation Metrics

**During training:**
- Training loss (should decrease)
- Validation loss (should decrease, may plateau)
- Gradient norms (should be stable)
- Learning rate (should decay)

**After training:**
- Perplexity (lower = better)
- Zero-shot benchmarks (LAMBADA, HELM)
- Human evaluation (preference tests)

---

## nanochat: Complete Pipeline in $100

> "$100 ChatGPT in 4 hours... this trains a 1.9B parameter model on 38B tokens, achieving GPT-2 level performance"
>
> — *Source: [source-documents/00-...nanochat](../../source-documents/karpathy/00-2025-10-28%20https___www.youtube.com_watch_v=0PGRqDy9C04.md)*

**Full speedrun:**
1. Tokenizer training (Rust BPE)
2. Base model pre-training (1.9B params, 38B tokens)
3. Midtraining (conversational data)
4. SFT (instruction following)
5. RLHF (optional, for math)
6. Evaluation (CORE, ARC, GSM8K, HumanEval, MMLU)
7. Web UI deployment

**Hardware:** 8×H100 GPUs for 4 hours

---

## Common Training Issues

**Loss not decreasing:**
- Learning rate too low/high
- Bad data mixture
- Tokenization issues
- Vanishing/exploding gradients

**Loss explodes (NaN):**
- Learning rate too high
- Gradient clipping too loose
- Numerical instability
- Bad initialization

**Model memorizes training data:**
- Overtrained (too many epochs)
- Data duplication
- Insufficient regularization

**Model is dumb:**
- Not enough data
- Poor data quality
- Undertrained (not enough tokens)
- Model too small

---

## Karpathy's Training Wisdom

> "pre-training stage is where all of the computational work basically happens this is 99% of the training compute time... the other three stages are fine-tuning stages that are much more along the lines of a small few number of gpus and hours or days"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Key lessons:**
- Pre-training is expensive, fine-tuning is cheap
- More tokens > bigger model
- SFT is achievable, RLHF is research territory
- Start small, verify pipeline works, then scale

---

## Next Steps

**Practical guides:**
- [../practical-implementation/](../practical-implementation/) - nanoGPT and nanochat walkthroughs
- [../gpt-architecture/](../gpt-architecture/) - Model architecture details

**Source code:**
- `source-codebases/00-nanoGPT/` - Pre-training pipeline
- `source-codebases/01-nanochat/` - Full training pipeline (base → RLHF)

**Primary sources:**
- `source-documents/32-State of GPT...` - Complete training overview
- `source-documents/31-Let's reproduce GPT-2...` - Practical reproduction guide
- `codebase/00-overview.md` - nanoGPT vs nanochat comparison

---

## The Big Picture

**Modern LLM training:**
```
1. Pre-train on internet (months, $millions)
2. Fine-tune on instructions (days, $thousands)
3. Align with human preferences (weeks, $tens of thousands)
4. Deploy and iterate
```

**Result:** Models that can chat, code, reason, and assist.

All from predicting the next token.
