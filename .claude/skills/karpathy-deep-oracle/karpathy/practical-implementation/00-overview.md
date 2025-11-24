# Practical Implementation - From Code to Running Models

**nanoGPT and nanochat: Minimalist Implementations That Actually Work**

---

## Philosophy: Hackable > Framework

> "The simplest, fastest repository for training/finetuning medium-sized GPTs... prioritizes teeth over education... it is very easy to hack to your needs"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

**Core values:**
- ✅ Minimal code (~600 lines for nanoGPT core)
- ✅ Readable (plain Python, well-commented)
- ✅ Hackable (easy to modify and experiment)
- ✅ Not a framework (single cohesive codebase)
- ✅ Maximally forkable

---

## nanoGPT: Pre-training in ~600 Lines

### Quick Start (3 Minutes on GPU)

**Train on Shakespeare:**
```bash
# 1. Prepare data (creates train.bin and val.bin)
python data/shakespeare_char/prepare.py

# 2. Train (3 minutes on A100)
python train.py config/train_shakespeare_char.py

# 3. Sample
python sample.py --out_dir=out-shakespeare-char
```

**Config details (train_shakespeare_char.py):**
- Context size: 256 characters
- Feature channels: 384
- Architecture: 6-layer Transformer, 6 heads per layer
- Training time: ~3 minutes on A100
- Best validation loss: 1.4697

**Output:**
```
ANGELO: And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO: I thank your eyes against it.
```

> "lol ¯\_(ツ)_/¯ . Not bad for a character-level model after 3 minutes of training on a GPU."
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

### Training on CPU (MacBook / Cheap Computer)

**No GPU? No problem!** Dial down the hyperparameters:

```bash
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --log_interval=1 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0
```

**Key changes for CPU:**
- `--device=cpu` (use CPU instead of GPU)
- `--compile=False` (disable PyTorch 2.0 compile)
- `--block_size=64` (context: 256→64 characters)
- `--batch_size=12` (batch: 64→12 examples)
- Smaller network: 4 layers, 4 heads, 128 dims (vs 6/6/384)
- `--eval_iters=20` (faster but noisier validation)
- `--dropout=0.0` (no regularization for tiny network)

**Training time:** ~3 minutes on CPU
**Validation loss:** 1.88 (vs 1.47 on GPU)

**Output:**
```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive
```

> "Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

**Apple Silicon tip:** Add `--device=mps` (Metal Performance Shaders) for 2-3× speedup!

### Core Files

**train.py** (~300 lines):
- Training loop
- Data loading
- Gradient accumulation
- Checkpointing
- Logging (wandb integration)

**model.py** (~300 lines):
- GPT architecture
- Multi-head attention
- Feed-forward networks
- Layer normalization
- Positional embeddings

**sample.py** (~100 lines):
- Inference/sampling
- Temperature control
- Top-k/top-p sampling

**config/** (various):
- Hyperparameter configurations
- Pre-set configs for Shakespeare, GPT-2, etc.

---

## Reproducing GPT-2 (124M)

> "When openi released gpt2 this was 2019... today you can reproduce this model in roughly an hour or probably less even and it will cost you about 10 bucks if you want to do this on the cloud"
>
> — *Source: [source-documents/21-...GPT-2 reproduction](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**The GPT-2 Miniseries (4 models):**
```
124M params:  12 layers, 768 dims, 12 heads (we're targeting this)
355M params:  24 layers, 1024 dims, 16 heads
774M params:  36 layers, 1280 dims, 20 heads
1.5B params:  48 layers, 1600 dims, 25 heads
```

**Modern reproduction costs:**
- **Time:** ~1 hour or less
- **Cost:** ~$10 on cloud GPUs
- **Hardware:** 8×GPU (H100s or A100s)
- **Result:** Beat OpenAI's original validation loss

### Full Reproduction Process

**1. Download and tokenize OpenWebText:**
```bash
# Downloads OpenWebText dataset (open reproduction of OpenAI's WebText)
# Creates train.bin and val.bin with GPT-2 BPE token IDs
python data/openwebtext/prepare.py
```

**What is OpenWebText?**
- Open reproduction of OpenAI's (private) WebText
- GPT-2 BPE tokenization
- Stored as raw uint16 bytes in sequence

**2. Train (distributed across 8 GPUs):**
```bash
# Single node with 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**Multi-node training (GPU go brrrr):**
```bash
# Master node (IP: 123.456.123.456):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=123.456.123.456 --master_port=1234 train.py

# Worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=123.456.123.456 --master_port=1234 train.py
```

**⚠️ No Infiniband?** Prepend `NCCL_IB_DISABLE=1` to launches (multinode will work but slower)

**3. Validation loss tracking:**
- Start: ~10.0 (random initialization)
- Target: Beat ~2.85 (OpenAI's GPT-2 124M finetuned on OWT)
- Training: ~4 days on 8×A100 40GB
- Result: ~2.85 validation loss

> "We see that we go from doing that task not very well because we're initializing from scratch all the way to doing that task quite well... and hopefully we're going to beat the gpt2 124 M model"
>
> — *Source: [source-documents/21-...](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**Note on dataset domain gap:**
- GPT-2 was trained on closed WebText
- OpenWebText is best-effort reproduction
- GPT-2 124M on OWT gets ~3.11 val loss
- After finetuning on OWT → comes down to ~2.85
- So our reproduction matching ~2.85 is appropriate!

### Loading Pretrained Weights (For Testing)

**Problem:** OpenAI released TensorFlow weights
**Solution:** Use Hugging Face's PyTorch conversion

```python
from transformers import GPT2LMHeadModel

# Load GPT-2 124M (confusingly just called "gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# For actual GPT-2 1.5B, use:
# model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
```

**Vocabulary:** 50,257 tokens
- 50,000 BPE merges
- 256 byte tokens (leaves)
- 1 special end-of-text token

### Visualizing Learned Embeddings

**Position embeddings (1024 × 768):**
- Each row = one position (0-1023)
- Each column = one dimension (0-767)
- Learns sinusoidal-like patterns during training

> "You can see that this has structure... these positional embeddings end up learning these sinusoids and cosiness that sort of like represent each of these positions"
>
> — *Source: [source-documents/21-...](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**What to look for:**
- Smooth curves = well-trained
- Jagged curves = undertrained
- Random noise = not trained yet

> "Because they're a bit more Jagged and they're kind of noisy you can tell that this model was not fully trained and the more trained this model was the more you would expect to smooth this out"
>
> — *Source: [source-documents/21-...](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

---

## Finetuning Pretrained Models

**Philosophy:** Start with pretrained checkpoint, train with smaller learning rate

> "Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

### Finetuning on Shakespeare (Few Minutes!)

**Step 1: Prepare Shakespeare data with GPT-2 tokenizer**
```bash
cd data/shakespeare
python prepare.py  # Uses OpenAI BPE tokenizer (not character-level!)
# Creates train.bin and val.bin in seconds
```

**Step 2: Finetune from GPT-2 checkpoint**
```bash
python train.py config/finetune_shakespeare.py
```

**Key config overrides (finetune_shakespeare.py):**
- `init_from='gpt2'` (or 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
- Much shorter training time
- Smaller learning rate
- Can decrease `block_size` if running out of memory

**Training time:** Few minutes on single GPU
**Best checkpoint:** Saved to `out-shakespeare/` (lowest validation loss)

**Step 3: Sample from finetuned model**
```bash
python sample.py --out_dir=out-shakespeare
```

**Output:**
```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I lie,
I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.
```

> "Whoa there, GPT, entering some dark place over there"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

### OpenAI GPT-2 Baselines on OpenWebText

**Evaluation of official GPT-2 checkpoints:**

```bash
python train.py config/eval_gpt2.py        # 124M
python train.py config/eval_gpt2_medium.py # 350M
python train.py config/eval_gpt2_large.py  # 774M
python train.py config/eval_gpt2_xl.py     # 1558M
```

**Results:**

| Model | Params | Train Loss | Val Loss |
|-------|--------|------------|----------|
| gpt2 | 124M | 3.11 | 3.12 |
| gpt2-medium | 350M | 2.85 | 2.84 |
| gpt2-large | 774M | 2.66 | 2.67 |
| gpt2-xl | 1558M | 2.56 | 2.54 |

**Important note:** GPT-2 was trained on closed WebText, OpenWebText is reproduction. Domain gap exists! After finetuning GPT-2 124M on OWT directly → loss reaches ~2.85, which becomes the appropriate reproduction baseline.

---

## llm.c: Training in Pure C/CUDA

**Philosophy:** Even more minimal than nanoGPT - no Python at all!

> "LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch or 107MB of cPython"
>
> — *Source: [source-documents/35-karpathy_llm.c...](../../source-documents/karpathy/35-karpathy_llm.c_%20LLM%20training%20in%20simple%2C%20raw%20C_CUDA%20-%20GitHub.md)*

**Goal:** Reproduce GPT-2 miniseries (124M→1.5B) in pure C/CUDA

**Performance:** ~7% faster than PyTorch Nightly

### Three Implementations

**1. CPU reference (train_gpt2.c):**
- ~1,000 lines of clean code
- Single file, fp32 only
- Educational, simple to understand
- Perfect for learning

**2. GPU bleeding edge (train_gpt2.cu):**
- Full CUDA implementation
- Multi-GPU support
- Mixed precision (fp32, bf16, fp16)
- Production-grade performance

**3. Parallel PyTorch (train_gpt2.py):**
- Tweaked nanoGPT
- For comparison/verification
- Same results, different language

### Quick Start

```bash
# Download starter pack (model, tokenizer, tiny shakespeare)
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh

# Build and run (1 GPU, fp32)
make train_gpt2fp32cu
./train_gpt2fp32cu
```

**Starter pack includes:**
- GPT-2 124M model (fp32 + bf16)
- Debug state (for unit testing)
- GPT-2 tokenizer
- Tokenized TinyShakespeare dataset

### Why Pure C/CUDA?

**Benefits:**
- No 245MB PyTorch dependency
- No 107MB Python dependency
- Understand every single line
- Learn CUDA from scratch
- Maximum performance control

**Good for:**
- Learning CUDA
- Single/multi-GPU training
- Understanding the full stack
- Maximum performance tuning

**Not good for:**
- Multi-node distributed training → Use PyTorch
- Complex mixed-precision setups → PyTorch handles better
- Beginners → Start with nanoGPT (Python)

### Learning Path

**Start: fp32 legacy versions**
- Simpler, more portable
- "Checkpointed" early in project history
- Frozen in time for learning
- Easier to understand CUDA basics

**Graduate: bleeding edge**
- Mixed precision
- Multi-GPU coordination
- Latest kernel optimizations

**Debugging tip:**
```bash
# Replace -O3 with -g in Makefile
make train_gpt2cu  # Now debuggable in VSCode/GDB
```

**Community:**
- GitHub Discussions
- Discord: #llmc on Zero to Hero
- Discord: #llmdotc on gpumode

---

## nanoGPT Configuration

**Example (train_shakespeare_char.py):**
```python
# Model
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2

# Training
batch_size = 64
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

# System
device = 'cuda'
compile = True  # PyTorch 2.0
```

**Tweak these to:**
- Increase model size (more layers, heads, embeddings)
- Change training duration (max_iters)
- Adjust learning rate schedule
- Enable/disable compilation

---

## Fine-tuning Pretrained Models

**Load GPT-2 checkpoint and fine-tune:**

```bash
# Fine-tune GPT-2 on Shakespeare
python train.py config/finetune_shakespeare.py
```

**Key config changes:**
```python
init_from = 'gpt2'  # Start from pretrained
learning_rate = 1e-4  # Much smaller than training from scratch
max_iters = 2000  # Shorter fine-tuning
```

**Output (after fine-tuning):**
```
THEODORE: Thou shalt sell me to the highest bidder:
if I die, I sell thee to the first;
if I go mad, I sell thee to the second...
```

> "Whoa there, GPT, entering some dark place over there."
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

---

## nanochat: Full ChatGPT Pipeline

### Architecture Overview

**Size:** ~8K lines across 45 files

**Pipeline stages:**
1. **Tokenizer training** (`scripts/tok_train.py`)
2. **Base pre-training** (`scripts/base_train.py`)
3. **Midtraining** (`scripts/mid_train.py`) - Conversational adaptation
4. **SFT** (`scripts/chat_sft.py`) - Instruction following
5. **RLHF** (`scripts/chat_rl.py`) - Preference alignment
6. **Evaluation** (`scripts/chat_eval.py`) - Benchmarks
7. **Web UI** (`scripts/chat_web.py`) - Deployment

### Speedrun ($100 in 4 Hours)

```bash
# Run complete pipeline
bash speedrun.sh

# Serve web interface
python -m scripts.chat_web
```

**Hardware:** 8×H100 GPUs
**Cost:** ~$100
**Result:** 1.9B parameter conversational model

> "The best ChatGPT that $100 can buy"
>
> — *nanochat philosophy*

### Key Files

**nanochat/gpt.py:**
- Transformer implementation
- RMS norm (not LayerNorm)
- Rotary embeddings (RoPE)
- Grouped query attention (GQA)
- ReLU² activation

**nanochat/engine.py:**
- Two-phase generation (prefill + decode)
- KV caching for efficiency
- Tool use (Python interpreter integration)
- Streaming generation

**config.py:**
- Central configuration
- Easy scaling (adjust n_layer, n_embd, etc.)
- Device, batch size, training params

---

## Key Implementation Details

### 1. Data Loading

**nanoGPT approach:**
```python
# Memory-mapped numpy array (fast, efficient)
data = np.memmap('train.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y
```

**Why memory-mapped?**
- Doesn't load full dataset into RAM
- Fast random access
- Works with datasets larger than memory

### 2. Training Loop

```python
# Simplified nanoGPT training loop
for iter in range(max_iters):
    # Get batch
    X, Y = get_batch('train')

    # Forward pass
    logits, loss = model(X, Y)

    # Backward pass
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Decay learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Log and checkpoint
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter % save_interval == 0:
        torch.save(model.state_dict(), f'ckpt_{iter}.pt')
```

### 3. Sampling/Generation

```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    idx: (B, T) array of token indices
    Generate max_new_tokens more tokens
    """
    for _ in range(max_new_tokens):
        # Crop to block_size
        idx_cond = idx[:, -block_size:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Get last token logits
        logits = logits[:, -1, :] / temperature

        # Optional top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

---

## Efficiency Techniques

### PyTorch 2.0 Compile

```python
model = GPT(config)
model = torch.compile(model)  # One line = 2× speedup!
```

**Impact:** ~135ms/iter instead of ~250ms/iter

### Flash Attention

```python
# Automatic in PyTorch 2.0+
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Impact:** 3× faster, lower memory

### Mixed Precision Training

```python
# Use torch.cuda.amp for automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits, loss = model(X, Y)
scaler.scale(loss).backward()
```

**Impact:** Faster training, lower memory

---

## Evaluation

**nanoGPT benchmarks:**
- Character-level loss on Shakespeare
- Validation loss on OpenWebText
- Perplexity comparisons

**nanochat benchmarks:**
- **CORE:** Base model quality
- **ARC-Challenge/Easy:** Science reasoning
- **GSM8K:** Math problems
- **HumanEval:** Code generation
- **MMLU:** General knowledge

---

## Common Modifications

### 1. Changing Model Size

```python
# Bigger model
n_layer = 24  # from 12
n_embd = 1024  # from 768
n_head = 16  # from 12
```

### 2. Changing Context Length

```python
block_size = 2048  # from 1024
```

### 3. Adding New Data

```python
# Create custom data loader
python data/my_dataset/prepare.py
python train.py --data_dir=data/my_dataset
```

### 4. Custom Tokenizers

```python
# Use different tokenizer (e.g., SentencePiece)
from sentencepiece import SentencePieceProcessor
tokenizer = SentencePieceProcessor(model_file='my_tokenizer.model')
```

---

## MacBook Training (Yes, Really!)

**For learning on cheap hardware:**

```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000
```

**Result:** ~3 minutes on CPU, loss ~1.88

> "GLEORKEN VINGHARD III: Whell's the couse, the came light gacks..."
>
> — *CPU-trained Shakespeare model*

**With Apple Silicon:**
Add `--device=mps` for 2-3× speedup!

---

## Multi-GPU Training

### Single Node (8 GPUs)

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### Multi-Node (2 nodes, 8 GPUs each)

**Master node:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
```

**Worker node:**
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
```

---

## Debugging Tips

**Loss not decreasing?**
1. Check data loading (print batch samples)
2. Verify gradient flow (print gradient norms)
3. Try smaller model first
4. Lower learning rate

**NaN loss?**
1. Reduce learning rate
2. Enable gradient clipping
3. Check for numerical instability
4. Use mixed precision carefully

**Out of memory?**
1. Reduce batch size
2. Reduce model size
3. Enable gradient checkpointing
4. Use smaller block_size

---

## Next Steps

**Experiment:**
- Train on custom datasets
- Modify architectures
- Implement new optimizers
- Add evaluation metrics

**Scale up:**
- Move from Shakespeare to larger datasets
- Increase model size gradually
- Use multiple GPUs

**Source code:**
- `source-codebases/00-nanoGPT/` - Complete nanoGPT implementation
- `source-codebases/01-nanochat/` - Full nanochat pipeline

**Primary sources:**
- `source-documents/36-karpathy_nanoGPT...` - nanoGPT documentation
- `source-documents/00-...nanochat` - nanochat overview
- `codebase/00-overview.md` - Detailed comparison

---

## Karpathy's Implementation Philosophy

**Minimal, readable, hackable.**

> "Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints"
>
> — *nanoGPT README*

Not a framework. Not a library. **A single, cohesive, forkable codebase.**

Learn by doing. Modify with confidence.
