# GPT Architecture - The Transformer Explained

**How GPT Actually Works (Karpathy's Plain-English Guide)**

---

## What is GPT?

**GPT = Generative Pre-trained Transformer**

Breaking it down:
- **Generative:** Predicts next token in sequence
- **Pre-trained:** Trained on massive internet text
- **Transformer:** Specific neural network architecture

At its core: **A text completion engine.**

> "These Transformers are just like token simulators... they don't know what they don't know like they just imitate the next token"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## The Big Picture

```
Text → Tokens → Embeddings → Transformer Blocks → Output Probabilities
```

**Input:** "The cat sat on the"
**Output:** [0.001 (dog), 0.834 (mat), 0.089 (chair), ...]
**Sample:** "mat"

Repeat this billions of times = trained GPT.

---

## Why Transformers? (Deriving from First Principles)

### The Problem: CNNs Failed for Text

**In 2017:** CNNs dominated image processing but failed at NLP

> "CNNs weren't quite as good at these NLP tasks. In fact, for almost all of them CNNs were significantly worse than humans. For many tasks, CNNs were so bad as to be completely unusable."
>
> — *Source: [source-documents/19-...Algorithmic Simplicity](../../source-documents/karpathy/19-2025-10-28%20https___www.youtube.com_watch_v=kWLed8o5M2Y&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=10&pp=iAQB.md)*

**Why CNNs work for images:**
- Nearby pixels are strongly related (similar colors)
- Knowing one pixel helps predict neighbors
- Local structure dominates

**Why CNNs fail for text:**
- Related words can be far apart in sentences
- Example: "The dog spun around so fast that it caught its **tail**"
- "dog" and "tail" are related but separated by 10 words!
- CNNs only combine nearby words in early layers

**The fundamental issue:** Long-range dependencies

> "the dog spun around so fast that it caught its tail. In this sentence it's clear that the 'tail' belongs to the 'dog' and so these 2 words are strongly related. But since it isn't until the final layer that information from these words can be combined, in the early layers the CNN will attempt to combine them with other words in the sentence which are less related, and get itself confused."
>
> — *Source: [source-documents/19-...](../../source-documents/karpathy/19-2025-10-28%20https___www.youtube.com_watch_v=kWLed8o5M2Y&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=10&pp=iAQB.md)*

### The Solution: All-Pairs Processing

**Key insight:** Process every pair of words simultaneously!

**From:**
```
CNN: Combine neighbors → larger groups → full sequence
Problem: Related words far apart can't combine early
```

**To:**
```
Transformer: All pairs at once → immediate long-range connections
```

**Pairwise convolutional layers:**
- Apply neural net to EVERY pair of words
- Distance doesn't matter anymore
- Related words connect immediately

**Result:** For n words, create n² pair vectors

### Problem 1: Word Order Lost

**Issue:** Averaging all permutations loses order!

"the fat cat" vs "the cat fat" → Same permutations → Same output

**Solution:** Positional encodings
- Attach position to each word vector
- Each vector = (word identity, position in sentence)
- Model can now distinguish word order

### Problem 2: Exponential Growth

**Issue:** n² vectors explode with layers!

- 3 words → 9 pairs → 81 vectors (after layer 2) → 729 vectors → ...
- Need thousands of dimensions per vector
- Completely impractical

**Solution:** Reduce n² back to n between layers

**How to reduce while preserving information?**

### The Attention Mechanism (Derived!)

**Goal:** Sum n² vectors (all pairs) down to n vectors (one per word)

**Problem:** Simple averaging = blurry mess (like averaging images)

> "imagine what happens when you take the average of a bunch of images: you just end up with a blurry mess. The same thing happens with our vectors, when you try to cram a bunch of different vectors together, you just get noise"
>
> — *Source: [source-documents/19-...](../../source-documents/karpathy/19-2025-10-28%20https___www.youtube.com_watch_v=kWLed8o5M2Y&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=10&pp=iAQB.md)*

**Solution:** Weighted sum (attention!)
- ONE large vector (important pair) preserves information
- Other vectors small (close to zero)
- Neural net learns to output large vectors for important pairs

**But which pairs are important?** Context-dependent!

Example 1: "there was a tree on fire and it was roaring"
- Important pair: (roaring, fire)

Example 2: "there was a lion on fire and it was roaring"
- Important pair: (roaring, lion)

**Same pair ("roaring", "fire") needs:**
- Large weight in sentence 1
- Zero weight in sentence 2

**Cannot determine from pair alone!**

### The Attention Formula (Step by Step)

**Step 1: Score all pairs**
- Second neural net outputs importance score for each pair
- Compares scores within each column

**Step 2: Normalize scores**
```python
# Raw scores might be: [10, 90, 0, 0, ...]
# Relative importance: [10/100, 90/100, 0/100, ...] = [0.1, 0.9, 0, ...]
relative_importance = exp(score) / sum(exp(scores))  # Softmax!
```

**Step 3: Weighted sum**
```python
output = sum(relative_importance[i] * pair_vector[i])
```

**Why exponential?** Ensure positive numbers for normalization!

**This is self-attention:**
- Model decides which inputs to "pay attention to"
- Data-dependent weighting
- Preserves important information, discards unimportant

### Optimizations → Actual Transformer

**Optimization 1: Linear projections**
- Replace pair representation neural nets with linear functions
- Much faster (n² neural nets → n² matrix multiplies)
- Move non-linearity to AFTER attention (the MLP layer!)

**Optimization 2: Bilinear scoring**
- Replace scoring neural nets with bilinear forms (dot products)
- Query-Key mechanism emerges naturally
- Still simple, still effective

**Optimization 3: Multi-head attention**
- Sometimes need info from multiple pairs
- Run attention mechanism multiple times in parallel
- Each "head" can focus on different relationships

**Final architecture:**
```
Self-Attention (linear + bilinear)
    ↓
MLP (non-linear processing)
    ↓
Repeat
```

**Result:** Modern transformer architecture derived from first principles!

---

## Core Components

### 1. Tokenization (The Necessary Evil)

> "Tokenization is my least favorite part of working with large language models but unfortunately it is necessary to understand in some detail because it is fairly hairy gnarly and there's a lot of hidden foot guns to be aware of"
>
> — *Source: [source-documents/26-...tokenization video](../../source-documents/karpathy/26-2025-10-28%20https___www.youtube.com_watch_v=zduSFxRajkE.md)*

**Problem:** Computers don't understand words.
**Solution:** Convert text to numbers (tokens).

**From Characters to Tokens:**
```
Characters: "hello world" (11 chars)
          ↓
UTF-8 bytes: [104, 101, 108, 108, 111, 32, ...] (11 bytes)
          ↓
BPE tokens: [15339, 1917] (2 tokens for GPT-4!)
```

**Why not just use characters?**
- Sequence would be TOO LONG
- Transformers have limited context (e.g., 8K tokens)
- Need to compress for efficiency

**Byte Pair Encoding (BPE):**
1. Start with individual bytes (256 symbols)
2. Find most common consecutive pairs
3. Merge them into new tokens
4. Repeat until vocab size ~100K

**Result:** Variable-length chunks
- "hello" → single token
- "hello world" → two tokens
- " world" (with space) → different token than "world"

**Vocabulary sizes:**
- GPT-2: ~50,257 tokens
- GPT-4: ~100,277 tokens
- More tokens = denser representation = sees more context

**Key insight:** Common words = single token, rare words = multiple tokens

> "Before we can actually train on this data we need to go through one more pre-processing step and that is tokenization and this is basically a translation of the raw text into sequences of integers"

---

## The Transformer Block: Map-Reduce Architecture

**Attention = Reduce (Communication)**
- Tokens exchange information
- Weighted sum (pooling/aggregation)
- Each token gathers from previous tokens
- 1024 tokens communicate with each other

**MLP = Map (Individual Processing)**
- Each token processes independently
- No information exchange
- Parallel computation per token
- Think about gathered information

> "Attention is a communication operation it is where all the tokens... this is where the tokens communicate this is where they exchange information... whereas MLP this happens at every single token individually there's no information being collected or exchanged... the attention is the reduce and the MLP is the map and what you end up with is that the Transformer just ends up just being a repeated application of map produce"
>
> — *Source: [source-documents/21-...GPT-2 reproduction](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**Pattern:**
```
Reduce (attention) → Map (MLP) → Reduce (attention) → Map (MLP) → ...
    ↓                    ↓              ↓                    ↓
 Gather info         Process        Gather more         Process more
```

Each block iteratively refines representations in the residual stream.

---

## The History of Attention (Before "Attention is All You Need")

### 2003: Neural Language Modeling Begins
**First neural network for language modeling** (Bengio et al., 2003)
- Used multi-layer perceptron (MLP)
- Input: 3 words → Output: probability for 4th word
- Foundation for later work

### 2014: Sequence-to-Sequence Models
**Problem:** How to translate English → French with variable length?

**Solution:** Encoder-decoder with LSTMs
```
Encoder LSTM: Reads English sentence → Creates context vector
Decoder LSTM: Takes context vector → Generates French
```

**Bottleneck identified:** Entire input sentence compressed into single vector!

> "We conjectured that the use of a fixed lung Vector is a bottleneck in improving the performance of the basic encoder decoder architecture"
>
> — *Source: [source-documents/11-...Transformers United](../../source-documents/karpathy/11-2025-10-28%20https___www.youtube.com_watch_v=XfpMkf4rD6E&t=615s.md)*

### 2015: Attention Mechanism Born
**Paper:** "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al.)

**Key insight:** Allow decoder to "look back" at all encoder states, not just final context vector

**The first attention:**
- Context = weighted sum of encoder hidden states
- Weights = softmax of compatibility scores between decoder state and encoder states

**Origin story (from email with Dimitri Bahdanau):**
> "One day I had this thought that it would be nice to enable the decoder RNN to learn to search where to put the cursor on the source sequence this was sort of inspired by translation exercises that um learning English in my middle school involved you gaze shifts back and forth between source and Target sequence as you translate"
>
> — *Source: [source-documents/11-...Transformers United](../../source-documents/karpathy/11-2025-10-28%20https___www.youtube.com_watch_v=XfpMkf4rD6E&t=615s.md)*

**The name "Attention":** Suggested by Yoshua Bengio on final paper pass (originally called "RNN search")

### 2017: "Attention is All You Need"
**Radical idea:** Delete the RNNs. Keep ONLY attention!

**Why it worked:**
1. Attention operates over sets (no sequential processing needed)
2. Added positional encoding (attention doesn't know position by itself)
3. Adopted ResNet-style residual connections
4. Interspersed attention with MLPs
5. Introduced multi-head attention (parallel attention operations)
6. Used layer normalization
7. Found good hyperparameters (4× expansion in MLP, etc.)

**Remarkably resilient:** 2017 architecture still used today with minimal changes!

Only major change: Pre-norm (put layer norms before blocks, not after)

---

## Attention as Message Passing on Directed Graphs

**Karpathy's interpretation:** Attention is communication, not just a mechanism

### The Graph View

**Each token = node in directed graph**
- Node stores a private data vector
- Emits three vectors: Query, Key, Value

**Communication round:**
```python
# At node i:
query = W_q @ node_data[i]      # What I'm looking for
key = W_k @ node_data[i]        # What I have
value = W_v @ node_data[i]      # What I'll communicate

# Look at all inputs (incoming edges):
for j in incoming_nodes:
    score = query @ key[j]      # How interesting is j to me?

# Normalize scores
attention_weights = softmax(scores)

# Weighted sum of values
update = sum(attention_weights[j] * value[j] for j in incoming_nodes)
```

> "To me attention is kind of like the communication phase of the Transformer... it's really just a data dependent message passing on directed graphs"
>
> — *Source: [source-documents/11-...Transformers United](../../source-documents/karpathy/11-2025-10-28%20https___www.youtube.com_watch_v=XfpMkf4rD6E&t=615s.md)*

### Multi-Head = Multiple Message Channels

**Each head = independent communication channel**
- Same graph, different Q/K/V weights
- Different heads seek different information
- All updates happen in parallel

**Layers = Iterative refinement**
- Communication happens multiple times
- Each layer with different weights
- Features refined through repeated message passing

### Encoder vs Decoder Connectivity

**Encoder (bidirectional):**
```
All tokens fully connected to each other
[tok1] ↔ [tok2] ↔ [tok3] ↔ [tok4]
```

**Decoder (causal/autoregressive):**
```
Triangular structure - no future connections
[tok1] → [tok2] → [tok3] → [tok4]
  ↓       ↓        ↓
```

**Cross-attention (decoder accessing encoder):**
- Queries from decoder tokens
- Keys and Values from encoder (top layer only!)
- Decoder "looks at" fully-processed encoder states

---

## Pre-Normalization vs Post-Normalization

**GPT-2 improvement over original Transformer:**

**Post-norm (original Transformer):**
```
x → [Attention] → Add → LayerNorm →
    [MLP] → Add → LayerNorm → Output
```
Problems:
- Normalization inside residual path
- Gradients must flow through norms
- Less clean gradient flow

**Pre-norm (GPT-2):**
```
x → LayerNorm → [Attention] → Add →
    LayerNorm → [MLP] → Add → Output
```
Benefits:
- Clean residual pathway from supervision to inputs
- Gradients flow unchanged through residual stream
- Blocks contribute additively over time

> "You actually prefer to have a single clean residual stream all the way from supervision all the way down to the inputs the tokens and this is very desirable and nice because the gradients that flow from the top... addition just distributes gradients during the backwards state to both of its branches equally so... the gradients from the top flows straight to the inputs the tokens through the residual Pathways unchanged"
>
> — *Source: [source-documents/21-...GPT-2 reproduction](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**Additional change:** Final LayerNorm added before output projection

---

## GELU Activation (Why Not ReLU?)

**ReLU problem: Dead neurons**
```python
relu(x) = max(0, x)
```
- Flat at zero = zero gradient
- Neurons can get "stuck" with no adaptation
- No learning in flat region

**GELU solution:**
```python
gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```
- Smooth everywhere
- Always has gradient
- Empirically better performance

> "The G always contributes a local gradient and so there's always going to be a change always going to be an adaptation and sort of smoothing it out ends up empirically working better in practice"
>
> — *Source: [source-documents/21-...GPT-2 reproduction](../../source-documents/karpathy/21-2025-10-28%20https___www.youtube.com_watch_v=l8pRSuU81PU&t=14379s.md)*

**Why "approximate"?**
- Historical: ERF function was slow in TensorFlow (2019)
- Tanh approximation was faster
- BERT and GPT-2 adopted it
- Today: No real reason, but we stick with it for exact reproduction

**Modern variants:** SwiGLU (LLaMA 3+), but GPT-2 uses approximate GELU
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

### 2. Embeddings

**Problem:** Tokens are just numbers. Need semantic meaning.

**Solution:** Embed tokens into high-dimensional space (e.g., 768 dimensions)

```python
token = 464  # "the"
embedding = embedding_table[token]  # [0.23, -0.41, 0.67, ...]
```

**Similar words get similar embeddings:**
- "cat" and "dog" are close
- "cat" and "democracy" are far

### 3. Positional Encodings

**Problem:** Transformer sees all tokens simultaneously (no sequence order!)

**Solution:** Add position information

**Options:**
- **Sinusoidal:** sin/cos functions (original Transformer)
- **Learned:** Train position embeddings (GPT-2)
- **Rotary (RoPE):** Rotation-based (modern, efficient)

**Why it matters:** "Dog bites man" ≠ "Man bites dog"

### 4. Self-Attention (The Magic)

**The key innovation of Transformers.**

**What it does:**
For each token, look at ALL previous tokens and decide which ones are relevant.

**Example:**
```
"The cat sat on the mat because it was tired"
```

When processing "it":
- High attention to "cat" (subject)
- Low attention to "the", "was" (less relevant)

**How it works:**
1. Each token creates Query, Key, Value vectors
2. Compare Query of current token with Keys of all tokens
3. Use comparison scores to weight the Values
4. Output = weighted sum of Values

```python
# Simplified self-attention
scores = Q @ K.T / sqrt(d_k)  # Compatibility scores
attention = softmax(scores)    # Normalize to probabilities
output = attention @ V         # Weighted sum of values
```

> "Attention Scaling (1/√d_k): Variance stabilization proof, Prevents softmax saturation, Maintains gradient flow"
>
> — *Source: [codebase/01-mathematical-optimizations.md](../codebase/01-mathematical-optimizations.md)*

### 5. Multi-Head Attention

**Problem:** Single attention mechanism is limiting.

**Solution:** Run multiple attention operations in parallel (e.g., 12 heads)

**Benefit:** Different heads learn different patterns:
- Head 1: Subject-verb relationships
- Head 2: Noun-adjective pairs
- Head 3: Long-range dependencies
- etc.

```python
# Multi-head attention
for i in range(num_heads):
    head_output[i] = self_attention(Q[i], K[i], V[i])

output = concatenate(head_outputs) @ W_output
```

### 6. Feed-Forward Networks

After attention, each token passes through a simple MLP:

```python
def feed_forward(x):
    hidden = relu(x @ W1 + b1)  # Expand dimension
    output = hidden @ W2 + b2    # Project back
    return output
```

**Why?** Processes each token independently, adds nonlinearity, increases model capacity.

### 7. Residual Connections

**Problem:** Deep networks have gradient flow issues.

**Solution:** Skip connections (add input to output)

```python
# Without residual
x = attention(x)
x = feed_forward(x)

# With residual (better!)
x = x + attention(x)
x = x + feed_forward(x)
```

**Benefit:** Gradients flow directly through network, enables deep models.

### 8. Layer Normalization

**Stabilizes training** by normalizing activations:

```python
def layer_norm(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + epsilon)
```

**Modern variant: RMS Norm** (simpler, faster)

```python
def rms_norm(x):
    return x / sqrt(mean(x²) + epsilon)
```

---

## Transformer Block (Put It All Together)

```python
def transformer_block(x):
    # 1. Self-attention with residual
    x = x + multi_head_attention(layer_norm(x))

    # 2. Feed-forward with residual
    x = x + feed_forward(layer_norm(x))

    return x
```

**Stack 12-96 of these = GPT model.**

---

## Full GPT Architecture

```
Input Text: "The cat"
    ↓
Tokenize: [464, 3797]
    ↓
Embed + Position: [[0.23, -0.41, ...], [0.67, 0.12, ...]]
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
    ...
    ↓
Transformer Block N
    ↓
Output Layer: [vocab_size probabilities]
    ↓
Sample: "sat"
```

---

## GPT-2 Specifications

| Model | Layers | Hidden Size | Heads | Parameters |
|-------|--------|-------------|-------|------------|
| GPT-2 (124M) | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 350M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1558M |

*Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

---

## Context Length (The Finite Memory)

GPT has a **fixed context window** (e.g., 2048 tokens for GPT-2).

**What this means:**
- Model only sees last N tokens
- Can't remember earlier text
- Like reading with a narrow window sliding across page

**Modern models:**
- GPT-3: 2048 tokens
- GPT-4: 8K-128K tokens
- Claude: 200K tokens

---

## Causal Masking (Left-to-Right Only)

GPT can ONLY look at previous tokens, not future ones.

**Why?** During training, we predict next token. Can't "cheat" by seeing it!

```
When predicting token 4:
Can see: [token1, token2, token3]
Cannot see: [token4, token5, ...]
```

This is called **causal** or **autoregressive** modeling.

---

## The Training Objective

**Goal:** Predict next token accurately

```python
for position in sequence:
    # See all previous tokens
    context = sequence[:position]

    # Predict next token
    prediction = model(context)

    # Calculate loss
    loss += cross_entropy(prediction, sequence[position])
```

**That's it.** Train on billions of tokens. Model learns language.

---

## Why Transformers Win

**Compared to RNNs (LSTMs, GRUs):**
- ✅ Parallelizable (process all tokens simultaneously)
- ✅ No vanishing gradients (attention connects all positions)
- ✅ Longer memory (direct connections via attention)
- ✅ Scales better (more data = better performance)

**Transformers unlocked large language models.**

---

## nanoGPT: Minimal Implementation

Karpathy's nanoGPT implements GPT in ~300 lines:

```python
# source-codebases/00-nanoGPT/model.py (simplified)

class GPT(nn.Module):
    def __init__(self, config):
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        # Embed tokens
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(len(idx)))
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

> "The code itself is plain and readable: train.py is a ~300-line boilerplate training loop and model.py a ~300-line GPT model definition"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

---

## Advanced Optimizations

### Flash Attention
- O(N²) memory → O(N) memory
- 3× speedup with identical math
- CUDA kernel optimization

### Grouped Query Attention (GQA)
- Fewer Key/Value heads than Query heads
- Drastically reduces KV cache size
- Faster inference, lower memory

### RoPE (Rotary Position Embeddings)
- Encodes position via rotation matrices
- Better length extrapolation
- More efficient than learned embeddings

*See [codebase/01-mathematical-optimizations.md](../codebase/01-mathematical-optimizations.md) for deep dives.*

---

## Karpathy's Architecture Philosophy

> "Prioritizes teeth over education... it is very easy to hack to your needs"
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

**Design principles:**
- ✅ Minimal code (no unnecessary abstractions)
- ✅ Readable (plain Python, well-commented)
- ✅ Hackable (easy to modify)
- ✅ Practical (uses what works)

**Not a framework. A single cohesive codebase.**

---

## Common Misconceptions

**❌ "Attention is all you need"**
→ No. You also need: embeddings, FFN, normalization, residuals, training data, compute...

**❌ "GPT understands language"**
→ No. It predicts next tokens based on patterns in training data.

**❌ "Bigger is always better"**
→ Not quite. Compute budget matters more than model size alone.

**✅ "Transformers are incredibly effective pattern matchers"**
→ Yes. That's why they work so well.

---

## Next Steps

**Deep dives:**
- [../training-llms/](../training-llms/) - How to train these architectures
- [../practical-implementation/](../practical-implementation/) - nanoGPT walkthrough

**Explore code:**
- `source-codebases/00-nanoGPT/model.py` - Full implementation
- `source-codebases/01-nanochat/nanochat/gpt.py` - Production variant

**Primary sources:**
- `source-documents/32-State of GPT...` - High-level overview
- `source-documents/31-Let's reproduce GPT-2...` - Implementation details
- `codebase/01-mathematical-optimizations.md` - Advanced techniques

---

## The Big Picture

**Transformer architecture enables:**
- Parallel processing (fast training)
- Long-range dependencies (attention)
- Scalability (billions of parameters)
- Transfer learning (pre-train once, fine-tune many)

**Result:** Language models that can write, reason, code, and more.

All from predicting the next token.
