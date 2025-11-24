# Neural Network Fundamentals - Overview

**Karpathy's Bottom-Up Approach to Understanding Neural Networks**

---

## Philosophy: Build From Scratch

> "Today I'm going to build it from scratch - that means no machine learning libraries, no pre-built Frameworks, only numpy Python and some maths"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

Karpathy's teaching philosophy: **Start simple, build intuition, scale gradually.**

The best way to understand neural networks is to implement one from scratch. No magic, no abstractions - just math and code you can see and touch.

---

## Character-Level Language Modeling (The makemore Project)

**Goal:** Build models that generate text one character at a time

> "makemore as the name suggests makes more of things that you give it... when you look at names.txt you'll find that it's a very large data set of names... if you train make more on this data set it will learn to make more of things like this"
>
> — *Source: [source-documents/06-...makemore](../../source-documents/karpathy/06-2025-10-28%20https___www.youtube.com_watch_v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&pp=iAQB.md)*

**Training data:** 32,000 names (e.g., Emma, Olivia, Isabella)
**Output:** New unique names (e.g., Dontel, Irot, Zhendi)

### What is Character-Level Modeling?

**Not word-level:** We don't treat "Emma" as a single unit
**Character-level:** We break it down: E → m → m → a

Each name becomes a sequence of character prediction tasks:
```
Input:  [START] → E → m → m → a
Output: E → m → m → a → [END]
```

**Key insight:** One word = many training examples packed together!

The word "Isabella" teaches us:
- `[START] → I` (I starts names)
- `I → s` (s follows I)
- `Is → a` (a follows Is)
- `Isa → b` (b follows Isa)
- ...
- `Isabella → [END]` (Isabella ends here)

---

## The Bigram Model (Simplest Language Model)

**Definition:** Predict next character using ONLY the previous character

> "in the bigram language model we're always working with just two characters at a time so we're only looking at one character that we are given and we're trying to predict the next character in the sequence"
>
> — *Source: [source-documents/06-...makemore](../../source-documents/karpathy/06-2025-10-28%20https___www.youtube.com_watch_v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&pp=iAQB.md)*

### How It Works: Simple Counting

**Step 1: Count all bigrams**
```python
# From names like "Emma", "Olivia", "Eva"
bigrams = {
    ('[START]', 'E'): 3,  # E starts 3 words
    ('E', 'm'): 1,
    ('m', 'm'): 1,
    ('m', 'a'): 2,
    ('a', '[END]'): 3,    # a ends 3 words
    # ... thousands more
}
```

**Step 2: Build 27×27 count matrix**
- Rows = first character (what we have)
- Columns = second character (what comes next)
- Cell value = how many times this happens

**Step 3: Convert counts to probabilities**
```python
# Row for 'm' might look like:
counts:        [516, 2500, 150, ...]  # m→., m→a, m→b, ...
probabilities: [0.1, 0.48, 0.03, ...]  # Divide by row sum
```

**Step 4: Sample from probabilities**
```python
# Start with [START] token
current = '[START]'

while current != '[END]':
    # Get row for current character
    probs = probability_matrix[current]

    # Sample next character
    next_char = sample_from(probs)
    print(next_char)

    current = next_char
```

### Why Bigram Models Are Terrible

Generated names: "h", "yanu", "o'reilly", "mor"

> "I'll be honest with you this doesn't look right... the reason these samples are so terrible is that bigram language model is actually just like really terrible"
>
> — *Source: [source-documents/06-...makemore](../../source-documents/karpathy/06-2025-10-28%20https___www.youtube.com_watch_v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&pp=iAQB.md)*

**The problem:** No long-term memory!

When generating "h" as a complete name:
- Model sees: "previous character was [START]"
- Model thinks: "h sometimes follows [START]"
- Model doesn't know: "h was the ONLY character so far!"
- Result: Outputs `[END]` immediately → name is just "h"

**The model is myopic:** It only sees one character back, never the full context.

### The torch.multinomial Function

**Sampling from probability distributions:**

```python
import torch

# Create probability distribution
probs = torch.tensor([0.6, 0.3, 0.1])  # 60%, 30%, 10%

# Sample 20 times
g = torch.Generator().manual_seed(2147483647)  # Deterministic
samples = torch.multinomial(probs, num_samples=20, replacement=True, generator=g)

# Results: mostly 0s, some 1s, few 2s
# [0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
```

**Why replacement=True matters:**
- `True`: Sample, put back, sample again (can get duplicates)
- `False`: Sample, remove from pool (no duplicates)

For language modeling, we always want `replacement=True` because we can repeat characters!

### From Bigrams to Neural Networks

**The progression (makemore series):**
1. ✅ Bigram (counting) - START HERE
2. Multilayer Perceptron (MLP)
3. Recurrent Neural Networks (RNNs)
4. LSTMs/GRUs
5. Transformers (GPT-2 equivalent!)

Each step adds more context and intelligence.

**Bigram limitations:**
- Only sees 1 character back
- No sense of "this is the first character" vs "this is the 10th character"
- Pure statistics, no learning of patterns

**Neural networks solve this:** Can look back many characters, learn patterns, understand context.

---

## Multi-Layer Perceptrons (MLPs): The Next Step

### The Problem with Counting-Based Bigrams

**Exponential explosion with more context!**

> "the problem with this approach though is that if we are to take more context into account when predicting the next character in a sequence things quickly blow up"
>
> — *Source: [source-documents/08-...makemore Part 2](../../source-documents/karpathy/08-2025-10-28%20https___www.youtube.com_watch_v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB.md)*

**The explosion:**
- 1 character context: 27 possibilities (manageable)
- 2 character context: 27 × 27 = 729 possibilities
- 3 character context: 27 × 27 × 27 = 19,683 possibilities
- 4+ characters: Completely impractical!

**The data sparsity problem:**
- Most combinations never appear in training data
- Not enough counts to create reliable probabilities
- Table grows exponentially, data doesn't

**Example:** "Emma" appears once in training
- 1-char: "E" appears maybe 100 times → good statistics
- 2-char: "Em" appears 5 times → okay statistics
- 3-char: "Emm" appears 1 time → terrible statistics!
- 4-char: "Emma" appears 1 time → useless!

**Solution needed:** Compress the representation, generalize across similar contexts.

### The Bengio et al. 2003 Approach

**Paper:** "A Neural Probabilistic Language Model" (Bengio et al., 2003)

> "this is the paper that we're going to first look at and then implement... it's very readable interesting and has a lot of interesting ideas"
>
> — *Source: [source-documents/08-...](../../source-documents/karpathy/08-2025-10-28%20https___www.youtube.com_watch_v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB.md)*

**Key insight:** Use embeddings to share knowledge!

**The paper's setup:**
- 17,000 word vocabulary (word-level, not character-level)
- Embed each word into 30-dimensional space
- Initially random, then learned via backpropagation

**Why embeddings help:**

Example: "a dog was running in a ___"
- Problem: Exact phrase never seen in training
- Solution: Maybe saw "the dog was running in a ___"
- If embeddings learned "a" ≈ "the", can transfer knowledge!

Similarly:
- "dog" ≈ "cat" (both animals)
- "running" ≈ "walking" (both motion verbs)
- Transfer through embedding space!

> "suppose that the exact phrase a dog was running in a has never occurred in a training data... but maybe you've seen the phrase the dog was running in a blank and maybe your network has learned that a and the are like frequently are interchangeable with each other... you can transfer knowledge through that embedding and you can generalize in that way"
>
> — *Source: [source-documents/08-...](../../source-documents/karpathy/08-2025-10-28%20https___www.youtube.com_watch_v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB.md)*

### MLP Architecture (Character-Level Adaptation)

**Karpathy's implementation:** Character-level instead of word-level

**Structure:**
1. **Embedding layer** (lookup table C)
   - 27 characters × 2 dimensions = 54 parameters
   - Each character → 2D vector (learned, not random!)
   - Block size 3 → 3 characters → 3×2 = 6 inputs

2. **Hidden layer** (with tanh activation)
   - Input: 6 numbers (concatenated embeddings)
   - Hidden neurons: 100 (hyperparameter choice)
   - Weights: 6×100 = 600 parameters
   - Biases: 100 parameters
   - Activation: tanh (squashes to -1 to 1)

3. **Output layer**
   - Input: 100 hidden neurons
   - Output: 27 logits (one per character)
   - Weights: 100×27 = 2,700 parameters
   - Biases: 27 parameters

**Total:** ~3,400 parameters

**Forward pass:**
```python
# 1. Embed characters (lookup)
emb = C[X]  # Shape: (batch, block_size, emb_dim)

# 2. Concatenate embeddings
emb_concat = emb.view(batch_size, -1)  # Flatten to (batch, 6)

# 3. Hidden layer
h = torch.tanh(emb_concat @ W1 + b1)  # (batch, 100)

# 4. Output layer
logits = h @ W2 + b2  # (batch, 27)

# 5. Softmax → probabilities
probs = F.softmax(logits, dim=1)
```

### Block Size (Context Length)

**Block size = how many previous characters to look at**

```python
block_size = 3  # Look at 3 characters to predict the 4th
```

**Example: "emma" with block_size=3:**
```
... → e  (predict 'e' given nothing)
..e → m  (predict 'm' given '..e')
.em → m  (predict 'm' given '.em')
emm → a  (predict 'a' given 'emm')
mma → .  (predict END given 'mma')
```

**Padding:** Use dots `...` for initial context

**Trade-off:**
- Larger block size = more context = better predictions
- But: More parameters, more computation
- Typical: 3-10 characters

### PyTorch Embedding Tricks

**Problem:** How to embed all integers in a batch simultaneously?

**Solution 1: Index with tensors (FAST)**
```python
C = torch.randn(27, 2)  # Embedding table
X = torch.tensor([[5, 13, 13], [1, 13, 13]])  # Batch of indices

emb = C[X]  # Shape: (2, 3, 2) - Just works! Fast!
```

**Solution 2: One-hot encoding (SLOW, don't use)**
```python
# Convert to one-hot, then matrix multiply
one_hot = F.one_hot(X, num_classes=27).float()
emb = one_hot @ C  # Same result, much slower!
```

> "we can interpret this first piece here this embedding of the integer we can either think of it as the integer indexing into a lookup table c but equivalently we can also think of this... as a first layer of this bigger neural net"
>
> — *Source: [source-documents/08-...](../../source-documents/karpathy/08-2025-10-28%20https___www.youtube.com_watch_v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB.md)*

**PyTorch indexing is powerful:**
- Can index with integers: `C[5]`
- Can index with lists: `C[[5, 6, 7]]`
- Can index with tensors: `C[torch.tensor([5, 6, 7])]`
- Can index with multi-dimensional tensors: `C[X]` where X is (32, 3)

### Flattening: view() vs concatenate()

**Problem:** Need to flatten (32, 3, 2) → (32, 6) for hidden layer

**Solution 1: torch.cat() (SLOW - creates new memory)**
```python
# Manually concatenate each embedding
parts = [emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]]
flat = torch.cat(parts, dim=1)  # (32, 6)
```

**Solution 2: view() (FAST - no new memory!)**
```python
flat = emb.view(32, 6)  # Instant! Just changes metadata
# Or let PyTorch infer:
flat = emb.view(emb.shape[0], -1)  # -1 = "figure it out"
```

**Why view() is fast:**
- Tensors have underlying storage (1D array of bytes)
- view() only changes shape/stride metadata
- No data copied or moved
- Just reinterprets same memory differently

> "in pytorch this operation calling that view is extremely efficient and the reason for that is that in each tensor there's something called the underlying storage and the storage is just the numbers always as a one-dimensional vector... no memory is being changed copied moved or created when we call that view"
>
> — *Source: [source-documents/08-...](../../source-documents/karpathy/08-2025-10-28%20https___www.youtube.com_watch_v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB.md)*

### Broadcasting Pitfalls

**Always check shapes when adding biases!**

```python
h = some_matrix @ W1 + b1  # Does this work correctly?
```

**Check:**
```python
print((some_matrix @ W1).shape)  # (32, 100)
print(b1.shape)                  # (100,)
# Broadcasting: (32, 100) + (100,)
#   → (32, 100) + (1, 100)  # Add fake dim on left
#   → (32, 100) + (32, 100)  # Copy vertically
# Result: Each row gets same bias - CORRECT!
```

**If shapes were different, could broadcast incorrectly!**

Always verify: "Is the bias being added to every example in the batch the same way?"

### Training the MLP

**Loss function:** Negative log-likelihood (same as bigram)

```python
# Get probabilities for correct characters
correct_probs = probs[torch.arange(32), y]

# Average negative log probability
loss = -correct_probs.log().mean()
```

**Initially:** Loss ~17 (untrained, random predictions)
**After training:** Loss decreases, predictions improve!

**Total parameters:** ~3,400 (much better than 27³ = 19,683 counting table!)

---

## What is a Neural Network?

### The Fruit Classification Example

**Problem:** Discover which mysterious purple spiky fruits are poisonous

> "We've discovered a peculiar new fruit, which is purple and spiky with orange spots, and extremely delicious. Strangely though, *some* of them seem to be poisonous"
>
> — *Source: [source-documents/15-...](../../source-documents/karpathy/15-2025-10-28%20https___www.youtube.com_watch_v=hfMk-kjRv4c&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=4&pp=iAQB.md)*

**Two features:**
- Size of orange spots (0-10 scale)
- Length of spikes (0-10 scale)

**Plot data on graph:**
```
  Spike
  Length  •      •   (safe)
    ↑     •  •  •
   10   •  •    •••  (poisonous)
    │  •   •••
    0 ─────────→ Spot Size
      0        10
```

**Decision boundary:** Draw a line separating safe from poisonous

This is what a neural network learns: **finding decision boundaries in high-dimensional space**.

For 2 features → 2D graph
For 784 pixels → 784-dimensional space (yikes!)

But the concept is the same: separate classes with boundaries.

---

### Single Neuron (The Building Block)

At its core, a neural network is surprisingly simple:

**Single Neuron:**
- Takes multiple inputs
- Multiplies each by a weight
- Adds a bias
- Outputs the weighted sum

```
output = (input₁ × weight₁) + (input₂ × weight₂) + ... + bias
```

For the fruit example:
```python
# Inputs: spot_size, spike_length
decision = (spot_size × w₁) + (spike_length × w₂) + bias
if decision > 0:
    return "poisonous"
else:
    return "safe"
```

The weights and bias define the decision boundary!

---

### Multiple Neurons (A Network)

**Multiple Neurons (A Network):**
- Stack neurons in layers
- Each neuron connects to all neurons in previous layer
- All connections = lots of weighted sums
- *But wait...* this is just matrix multiplication!

---

## The Magic of Linear Algebra

Instead of calculating each neuron individually:

```python
# DON'T DO THIS (slow, tedious)
output1 = w1*x1 + w2*x2 + w3*x3 + bias1
output2 = w4*x1 + w5*x2 + w6*x3 + bias2
# ...hundreds more...
```

Do this:

```python
# DO THIS (fast, elegant)
outputs = inputs @ weights + biases
```

One line. One dot product. That's the power of linear algebra.

> "Many many years ago some of these guys invented the dot product which converts all this mumbo jumbo into to this one python line pretty convenient huh"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

---

## Why Nonlinearity Matters

**Problem:**
If neural networks just multiply and add, they're basically fancy linear regression. Not very powerful.

**Solution: Activation Functions**

Add ReLU (Rectified Linear Unit):
```python
output = max(0, input)
```

Looks simple, but it introduces **nonlinearity** - the ability to learn complex patterns.

> "We want to introduce some nonlinear and we do that with reu... it will be better at understanding nonlinear data just trust me on this one"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

---

## The Output Layer: Softmax

For classification (like recognizing digits):

```python
# Convert weird numbers into probabilities
probabilities = softmax(outputs)
# Now: [0.1, 0.05, 0.7, 0.15] → sums to 1.0
```

> "We slap a softmax activation function... it just converts a bunch of weird numbers that the network outputs into a probability distribution which is just a scary way of saying that it tells you what the network thinks is the likelihood of each class being the correct one"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

Makes the network's confidence explicit: "I'm 70% sure this is a 5."

---

## Key Concepts (Karpathy's Way)

### 1. Forward Pass
Push data through the network, layer by layer:
- Input → Hidden layers → Output
- At each step: multiply, add, activate

### 2. Loss Function
How wrong is the model?
- Compare prediction to ground truth
- Calculate a single number (the loss)
- Lower loss = better model

### 3. Backpropagation
**The learning algorithm** (see [01-backpropagation.md](01-backpropagation.md))
- Figure out which weights caused the error
- Update weights to reduce error
- Repeat thousands of times

### 4. Learning Rate: The Critical Hyperparameter

**Too high:** Model diverges (loss goes to infinity)
> "If we do that then the network is going to stumble all over the place"

**Too low:** Model learns too slowly
- Taking baby steps forever

**Solution:** Start big, get smaller
```python
# Learning rate decay
lr = initial_lr * decay_factor ** epoch
```

> "The higher this number is the more rapidly the network is going to change its parameters... we want to take big steps at first and then smaller and smaller steps once we get closer to the solution"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

### 5. Optimizers: Smarter Updates

**SGD (Stochastic Gradient Descent):** Basic but effective

**With momentum:** Remember previous updates
- Like a ball rolling downhill
- Builds momentum in consistent directions
- Dampens oscillations

**Adaptive methods:** AdaGrad, RMSprop, Adam
- Adjust learning rate per parameter
- Usually overkill for simple problems

> "You can also do a bunch of fancy with them like implementing momentum adaptative gradients root mean Square propagation we don't really need that we're happy with our little SGD Optimizer"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

---

## The Three-Step Training Loop

```
1. Forward pass: Calculate output
2. Calculate loss: How wrong were we?
3. Backward pass: Update weights
→ Repeat until loss stops decreasing
```

Simple. Effective. This is the foundation of all modern AI.

---

## Karpathy's Learning Path

**Start here:**
1. Single neurons (just multiplication and addition)
2. Small networks (a few layers)
3. Backpropagation by hand (see the math)
4. micrograd tutorial (build it yourself)
5. Character-level models (simple but complete)
6. Scale up gradually (GPT, transformers, etc.)

**Don't start here:**
- ❌ Giant pre-trained models
- ❌ Complex frameworks (TensorFlow, PyTorch... yet)
- ❌ Advanced architectures (transformers, attention)
- ❌ Research papers (save for later)

---

## Why This Approach Works

> "The reason I bring all of this up is because I think to a large extent prompting is just making up for this sort of cognitive difference between these two kind of architectures"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

Understanding the fundamentals lets you:
- **Debug better** - Know what's happening under the hood
- **Innovate faster** - Modify architectures with confidence
- **Avoid cargo culting** - Understand *why* things work

---

## Essential Math (Don't Panic)

You need:
- ✅ Multiplication and addition (you got this)
- ✅ Derivatives (just slopes)
- ✅ Chain rule (combine derivatives)
- ✅ Matrix multiplication (dot products)

You don't need:
- ❌ PhD-level calculus
- ❌ Abstract algebra
- ❌ Measure theory
- ❌ Advanced topology

Karpathy shows the math, but explains it in plain English. Always.

---

## Real Results: MNIST & Fashion-MNIST

**From scratch neural network performance:**

**MNIST (handwritten digits):**
- 97.42% accuracy on test set
- After tweaking parameters and implementing mini-batches
- Can confidently classify 0-9 digits

> "Let's see what he thinks of this okay he predicted a nine nice... wow he's 99.99% sure that this is a nine low key it's getting kind of cocky"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

**Fashion-MNIST (clothing items):**
- 87% accuracy
- Same neural network code, just swapped dataset
- Classification: t-shirts, pants, bags, sneakers, boots

> "Training training training and boom 87% accuracy not bad huh... I literally copy pasted the same neural net code from amnest and got 87% so I'm happy with that"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

**What the mistakes look like:**
- Usually understandable misclassifications
- "Come on who who did this" (on ambiguous handwriting)
- Even simple networks get surprisingly good

---

## Next Steps

**Deep Dives:**
- [01-backpropagation.md](01-backpropagation.md) - How neural networks actually learn
- [02-micrograd.md](02-micrograd.md) - Build a tiny automatic differentiation engine
- [03-gradient-descent.md](03-gradient-descent.md) - Optimization mechanics

**Primary Source Materials:**
- `source-documents/34-The spelled-out intro to neural networks.md` - Complete micrograd walkthrough
- `source-documents/13-...` - Building neural networks from scratch video
- Zero to Hero video series - Entire learning path

---

## Karpathy's Philosophy

**Minimal:** No unnecessary abstractions
**Readable:** Plain Python, well-commented
**Hackable:** Easy to modify and experiment
**Practical:** Prioritizes what actually works
**Educational:** Clear explanations over flexibility

> "lol ¯\_(ツ)_/¯ . Not bad for a character-level model after 3 minutes of training on a GPU."
>
> — *Source: [source-documents/36-karpathy_nanoGPT...](../../source-documents/karpathy/36-karpathy_nanoGPT_%20The%20simplest%2C%20fastest%20repository%20for%20training_finetuning%20medium-sized%20GPTs.%20-%20GitHub.md)*

Start small. Build intuition. Scale gradually. That's the Karpathy way.
