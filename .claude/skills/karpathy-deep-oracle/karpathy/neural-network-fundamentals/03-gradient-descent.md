# Gradient Descent - The Optimization Engine

**How Neural Networks Find the Right Weights**

---

## The Mountain Analogy

Imagine you're stuck on a mountain in thick fog. You can't see the valley, but you can feel which direction slopes downward.

**Gradient descent:**
- Take a step in the steepest downward direction
- Repeat until you reach the bottom
- Hope you find the global minimum (not just a local dip)

That's training a neural network in a nutshell.

---

## The Math (Keep It Simple)

**Goal:** Minimize loss function L(weights)

**How:** Update weights in the direction that reduces loss

```
new_weight = old_weight - learning_rate × gradient
```

That's it. That's gradient descent.

**Gradient:** Direction of steepest *increase*
**Negative gradient:** Direction of steepest *decrease*

We move *opposite* the gradient to reduce loss.

---

## Vanilla Gradient Descent

```python
for epoch in range(num_epochs):
    # 1. Compute loss on ALL training data
    loss = compute_loss(model, all_training_data)

    # 2. Compute gradients
    gradients = compute_gradients(loss)

    # 3. Update weights
    for param, grad in zip(model.parameters(), gradients):
        param -= learning_rate * grad
```

**Problem:** Computing loss on ALL data is slow.
**Solution:** Stochastic Gradient Descent (SGD).

---

## Stochastic Gradient Descent (SGD)

Instead of using all data, use **one example at a time:**

```python
for epoch in range(num_epochs):
    for single_example in shuffle(training_data):
        loss = compute_loss(model, single_example)
        gradients = compute_gradients(loss)

        for param, grad in zip(model.parameters(), gradients):
            param -= learning_rate * grad
```

**Pros:**
- Much faster updates
- Can escape local minima (noise helps!)
- Works with huge datasets

**Cons:**
- Noisy updates (jittery path)
- Doesn't utilize hardware efficiently

---

## Mini-Batch Gradient Descent (The Sweet Spot)

Use a **batch** of examples (typically 32, 64, 128):

```python
for epoch in range(num_epochs):
    for batch in create_batches(training_data, batch_size=64):
        loss = compute_loss(model, batch)
        gradients = compute_gradients(loss)

        for param, grad in zip(model.parameters(), gradients):
            param -= learning_rate * grad
```

**Why batches?**
- ✅ Faster than single examples (GPU parallelization)
- ✅ Smoother than single examples (less noise)
- ✅ Better than full dataset (faster iterations)

**This is what everyone uses in practice.**

---

## The Learning Rate Problem

**Too high:** Model diverges
```
Loss: 0.5 → 1.2 → 5.3 → NaN (💥 explosion!)
```

**Too low:** Model learns too slowly
```
Loss: 0.500 → 0.499 → 0.498 → 😴 (weeks of training)
```

**Just right:** Smooth, steady descent
```
Loss: 0.500 → 0.301 → 0.152 → 0.089 → 0.045 (🎯)
```

---

## Learning Rate Schedules

**Don't use a fixed learning rate.** Decay it over time.

### Step Decay
```python
lr = initial_lr × 0.1^(epoch // decay_every)
```

### Exponential Decay
```python
lr = initial_lr × decay_rate^epoch
```

### Cosine Annealing (Karpathy's favorite)
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π × t / T))
```

**Why cosine?**
- Smooth decay (no sudden drops)
- Reaches minimum gracefully
- Often works best in practice

> "Cosine LR Schedule: Smooth optimization landscape, Better final convergence"
>
> — *Source: [codebase/01-mathematical-optimizations.md](../codebase/01-mathematical-optimizations.md)*

---

## Advanced Optimizers

### SGD with Momentum

Remember previous gradients (like a ball rolling down a hill):

```python
velocity = momentum × velocity - learning_rate × gradient
param += velocity
```

**Pros:**
- Accelerates in consistent directions
- Dampens oscillations
- Escapes small local minima

**Typical momentum:** 0.9

### Adam (Adaptive Moment Estimation)

Combines momentum with adaptive learning rates:

```python
# First moment (momentum)
m = beta1 × m + (1 - beta1) × gradient

# Second moment (adaptive learning rate)
v = beta2 × v + (1 - beta2) × gradient²

# Update
param -= lr × m / (sqrt(v) + epsilon)
```

**Pros:**
- Per-parameter learning rates
- Works well out-of-the-box
- Less hyperparameter tuning

**Typical values:**
- beta1 = 0.9
- beta2 = 0.999
- lr = 1e-3 or 3e-4

**Adam is the default optimizer for most tasks.**

### AdamW (Adam with Weight Decay)

Adam + proper weight decay (regularization):

```python
# Standard Adam update
param -= lr × m / (sqrt(v) + epsilon)

# Add weight decay
param -= lr × weight_decay × param
```

**Why separate weight decay?**
- Better regularization
- Decouples optimization from regularization
- Standard in modern training

---

## Warm-up: Starting Slow

**Problem:** Large learning rates at initialization can destabilize training.

**Solution:** Start with a small LR, gradually increase to max:

```python
if step < warmup_steps:
    lr = max_lr × (step / warmup_steps)
else:
    lr = cosine_schedule(step - warmup_steps)
```

**Typical warmup:** 1-5% of total training steps

---

## Gradient Clipping

**Problem:** Exploding gradients (gradients become huge)

**Solution:** Clip gradients to maximum norm:

```python
total_norm = sqrt(sum(grad² for grad in all_gradients))
if total_norm > max_norm:
    for grad in all_gradients:
        grad *= max_norm / total_norm
```

**Typical max_norm:** 1.0 or 5.0

**Prevents:** NaN losses, unstable training

---

## Weight Decay (L2 Regularization)

Penalize large weights to prevent overfitting:

```python
loss = prediction_loss + weight_decay × sum(param² for param in weights)
```

**Effect:**
- Encourages smaller weights
- Reduces overfitting
- Improves generalization

**Typical weight_decay:** 0.1, 0.01, 0.001

---

## Batch Size Effects

**Small batches (8-32):**
- More noise → better generalization?
- Slower training (fewer parallel operations)
- Fits in smaller GPUs

**Large batches (128-1024):**
- Less noise → faster convergence
- Faster training (more parallelization)
- Requires larger GPUs
- May generalize worse (debated)

**Typical choice:** 32-128 for most tasks

---

## Learning Rate Finder

**Don't guess the learning rate. Find it:**

1. Start with tiny LR (1e-8)
2. Train for a few batches
3. Double LR after each batch
4. Plot loss vs LR
5. Choose LR just before loss explodes

```
Loss
  |
  |     ___________  ← choose here
  |   /             \
  | /                 \_____ (explosion)
  |___________________
      LR (log scale)
```

**Best LR:** Where loss decreases fastest (steepest slope)

---

## Karpathy's Optimization Wisdom

> "The amount the network changes its parameters is called The Learning rate so the higher this number is the more rapidly the network is going to change its parameters well why don't we put it super high so it learns super fast well if we do that then the network is going to stumble all over the place"
>
> — *Source: [source-documents/13-...](../../source-documents/karpathy/13-2025-10-28%20https___www.youtube.com_watch_v=cAkMcPfY_Ns&list=PLDbgk_jPPZaDvtbw9USUw91AV85y0xX2S&index=8&pp=iAQB.md)*

**Key insights:**
- Start high, decay over time
- Big steps at first (far from solution)
- Small steps later (near solution)
- Use schedules, don't use fixed LR

---

## Common Training Issues

### Loss not decreasing
- ❌ Learning rate too low
- ❌ Gradients vanishing
- ❌ Wrong loss function

### Loss exploding (NaN)
- ❌ Learning rate too high
- ❌ Gradients exploding
- ❌ Numerical instability

### Loss plateaus early
- ❌ Learning rate decayed too fast
- ❌ Stuck in local minimum
- ❌ Model capacity too small

### Training is slow
- ❌ Batch size too small
- ❌ Learning rate too low
- ❌ Too much regularization

---

## Karpathy's Optimizer Choice

**For nanoGPT (small to medium models):**
- AdamW
- Cosine LR schedule with warmup
- Gradient clipping (norm 1.0)
- Weight decay 0.1

**For nanochat (production system):**
- Muon optimizer (custom, experimental)
- Same scheduling strategy
- Aggressive hyperparameter tuning

**Default recommendation:**
```python
optimizer = AdamW(params, lr=3e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)
warmup_scheduler = LinearWarmup(optimizer, warmup_steps=1000)
```

---

## Practical Training Checklist

✅ Use mini-batch gradient descent (batch size 32-128)
✅ Use AdamW optimizer (beta1=0.9, beta2=0.999)
✅ Start with LR 3e-4 or 1e-3
✅ Use cosine annealing schedule
✅ Add warmup (1000-5000 steps)
✅ Clip gradients (max norm 1.0)
✅ Add weight decay (0.01-0.1)
✅ Monitor loss plots carefully
✅ Save checkpoints frequently

---

## Next Steps

**Master optimization:**
- Experiment with different optimizers
- Tune learning rates systematically
- Understand when to use which optimizer

**Move on to:**
- [../gpt-architecture/](../gpt-architecture/) - Build transformers
- [../training-llms/](../training-llms/) - Scale to large models
- [../practical-implementation/](../practical-implementation/) - Real training scripts

**Primary sources:**
- `source-documents/13-...` - Training neural networks from scratch
- `codebase/01-mathematical-optimizations.md` - Advanced optimization techniques

---

## The Big Picture

**Gradient descent is the engine of deep learning:**
- Converts loss into weight updates
- Scales to billions of parameters
- Works for any differentiable model
- Foundation of all neural network training

**Modern recipe:**
AdamW + Cosine schedule + Warmup + Gradient clipping = 🚀

> "Profile before optimizing. Hardware-aware design. Simple > Complex. Optimize the hot path."
>
> — *Karpathy's optimization philosophy from [codebase/02-karpathy-on-deepseek-efficiency.md](../codebase/02-karpathy-on-deepseek-efficiency.md)*
