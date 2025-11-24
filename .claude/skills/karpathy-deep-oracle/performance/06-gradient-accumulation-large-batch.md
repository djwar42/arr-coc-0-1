# Gradient Accumulation & Large Batch Training

**Simulating large batches when GPU memory is limited**

---

## Section 1: Gradient Accumulation Fundamentals (~90 lines)

### What is Gradient Accumulation?

**Core Concept**: Accumulate gradients over multiple forward/backward passes before updating model weights.

From [PyTorch Gradient Accumulation Guide](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3) (accessed 2025-11-16):
> "Gradient accumulation allows training with effective batch sizes larger than what fits in GPU memory by accumulating gradients over multiple mini-batches before performing an optimizer step."

**Basic Formula**:
```
effective_batch_size = micro_batch_size × gradient_accumulation_steps × num_gpus
```

### Why Use Gradient Accumulation?

**Memory Constraints**:
```python
# Without gradient accumulation (OOM!)
batch_size = 64  # Requires 48GB GPU memory
loss.backward()
optimizer.step()

# With gradient accumulation (fits in 12GB!)
micro_batch_size = 8  # Uses 6GB GPU memory
accumulation_steps = 8
# Effective batch = 8 × 8 = 64

for i in range(accumulation_steps):
    loss = model(micro_batch[i])
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients

optimizer.step()  # Update once with accumulated gradients
optimizer.zero_grad()
```

**Benefits**:
- Train with large effective batch sizes on limited hardware
- Better training stability (larger batches = less noisy gradients)
- Enable scaling to batch sizes beyond single-GPU capacity
- Cost-effective alternative to multi-GPU setups

**Tradeoffs**:
- Slightly slower training (more forward/backward passes per update)
- No reduction in memory for activations (only gradients accumulate)
- Must handle learning rate scaling properly

### Mathematical Equivalence

**Gradient accumulation should be mathematically equivalent to large batch training**:

```python
# Large batch (if it fits)
loss = cross_entropy(model(batch_64), labels_64)
loss.backward()

# Gradient accumulation (equivalent!)
total_loss = 0
for micro_batch in split_into_8(batch_64):
    loss = cross_entropy(model(micro_batch), labels)
    loss = loss / 8  # Critical: scale by accumulation steps
    loss.backward()
    total_loss += loss.item()

# Both should produce identical weight updates
```

**Why the scaling?**: Without dividing by `accumulation_steps`, gradients would be `N×` larger than normal.

From [Weights & Biases Gradient Accumulation Tutorial](https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5) (accessed 2025-11-16):
> "Dividing the loss by the number of accumulation steps ensures the gradient magnitude remains consistent with training on the full batch size."

### PyTorch Implementation Pattern

**Basic Pattern**:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Scale loss
    loss = loss / accumulation_steps

    # Backward pass (accumulates gradients)
    loss.backward()

    # Update weights every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Gotcha**: Don't forget to handle the final batch if `len(dataloader) % accumulation_steps != 0`.

---

## Section 2: Memory Savings Analysis (~80 lines)

### Memory Breakdown: What Gets Saved?

**GPU Memory Components**:
```python
# Without gradient accumulation (batch_size = 64):
model_parameters    = 2GB   # FP16 weights
optimizer_states    = 6GB   # AdamW (momentum + variance)
gradients          = 2GB   # Same size as parameters
activations        = 12GB  # Stored for backward pass
total_memory       = 22GB  # Doesn't fit in 16GB GPU!

# With gradient accumulation (micro_batch = 8, steps = 8):
model_parameters    = 2GB   # Same
optimizer_states    = 6GB   # Same
gradients          = 2GB   # Same (accumulated, not replicated)
activations        = 1.5GB # Only for micro-batch!
total_memory       = 11.5GB # Fits in 16GB GPU!
```

**Key Insight**: Gradient accumulation primarily saves **activation memory**, not gradient or parameter memory.

From [DeepSpeed ZeRO Documentation](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md):
> "Activations are the dominant memory consumer for large batch training. Gradient accumulation reduces activation memory linearly with accumulation steps."

### Practical Memory Calculations

**Rule of Thumb**:
```python
# Maximum batch size without accumulation
max_batch = GPU_memory / (
    model_size × 2  # Parameters (FP16)
    + model_size × 8  # Optimizer (AdamW FP32)
    + activation_memory_per_sample
)

# With gradient accumulation
micro_batch = GPU_memory / (
    model_size × 10  # Model + optimizer (same)
    + activation_memory_per_sample  # Only micro-batch activations
)

effective_batch = micro_batch × accumulation_steps
```

**Example: GPT-2 Medium (355M params)**:
```
Single Sample Activations: ~50MB
GPU Memory: 16GB
Overhead (model + optimizer): ~8GB

Without accumulation:
max_batch = (16GB - 8GB) / 50MB = 160 samples

With 4× accumulation:
micro_batch = (16GB - 8GB) / 50MB = 160 samples
effective_batch = 160 × 4 = 640 samples!
```

### Memory vs Compute Tradeoff

**Time Cost**:
```python
# Baseline (batch_size = 64)
time_per_step = forward_time + backward_time + optimizer_time
            = 100ms + 100ms + 10ms = 210ms

# Gradient accumulation (micro_batch = 8, steps = 8)
time_per_step = 8 × (forward + backward) + optimizer
            = 8 × (12.5ms + 12.5ms) + 10ms = 210ms

# Identical time! (if activations aren't checkpoint-ed)
```

**But with activation checkpointing** (common for memory savings):
```python
time_per_step = 8 × (12.5ms + 25ms) + 10ms = 310ms
# 47% slower due to recomputation
```

From [PyTorch Forums: Gradient Accumulation Performance](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593) (accessed 2025-11-16):
> "Gradient accumulation adds minimal overhead if activations aren't checkpointed. The bottleneck is typically data loading, not accumulation itself."

### Combining with Other Techniques

**Gradient Accumulation + Mixed Precision**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with autocast():  # FP16 forward pass
        loss = model(batch) / accumulation_steps

    scaler.scale(loss).backward()  # Scaled backward

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)  # Unscale and update
        scaler.update()
        optimizer.zero_grad()
```

**Gradient Accumulation + Gradient Checkpointing**:
```python
# Ultra-memory-efficient (but slower)
model.gradient_checkpointing_enable()

accumulation_steps = 16  # Can use larger accumulation!
micro_batch_size = 2     # Tiny micro-batches

# Effective batch = 2 × 16 = 32
# Memory usage: Minimal
# Training time: ~2× slower
```

---

## Section 3: Large Batch Training Stability (~100 lines)

### The Large Batch Problem

**Challenge**: Large batches can destabilize training and hurt generalization.

From [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) (4,780 citations, accessed 2025-11-16):
> "When the minibatch size is multiplied by k, multiply the learning rate by k. This linear scaling rule allows us to scale to large minibatches without loss of accuracy."

**Why large batches are hard**:
```
Small batch (32):
- Noisy gradients (high variance)
- Explores loss landscape better
- Better generalization
- Slower convergence

Large batch (8192):
- Smooth gradients (low variance)
- Gets stuck in sharp minima
- Worse generalization
- Faster convergence (if stable)
```

### LAMB Optimizer (Layer-wise Adaptive Moments)

**Designed specifically for large batch training**.

From [LAMB: Layer-wise Adaptive Moments optimizer for Batch training](https://arxiv.org/abs/1904.00962) (1,200+ citations):
> "LAMB achieves state-of-the-art performance with batch sizes up to 64K without loss of accuracy."

**How LAMB Works**:
```python
# Standard Adam
gradient_update = lr * gradient / (sqrt(variance) + epsilon)

# LAMB (adds layer-wise adaptation)
for layer in model.layers:
    # Compute trust ratio
    weight_norm = torch.norm(layer.weight)
    gradient_norm = torch.norm(layer.gradient)
    trust_ratio = weight_norm / (gradient_norm + epsilon)

    # Scale update by trust ratio
    adam_update = lr * gradient / (sqrt(variance) + epsilon)
    lamb_update = trust_ratio * adam_update

    layer.weight -= lamb_update
```

**Key Innovation**: Different layers need different learning rates at large batch sizes.

**PyTorch Implementation**:
```python
from torch.optim import Optimizer

class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 weight_decay=0.01, eps=1e-6):
        defaults = dict(lr=lr, betas=betas,
                       weight_decay=weight_decay, eps=eps)
        super(LAMB, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Momentum and variance
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Adam update
                adam_step = exp_avg / bias_correction1
                adam_step /= (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])

                # Weight decay
                adam_step.add_(p.data, alpha=group['weight_decay'])

                # Trust ratio (LAMB's key innovation)
                weight_norm = torch.norm(p.data)
                adam_norm = torch.norm(adam_step)
                trust_ratio = weight_norm / (adam_norm + group['eps'])

                # Update with trust ratio
                p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)
```

### LARS Optimizer (Layer-wise Adaptive Rate Scaling)

**Alternative to LAMB, focuses on layer-wise learning rate scaling**.

From [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) (2,600+ citations):
> "LARS uses a separate learning rate for each layer that is proportional to the ratio of the weight norm to the gradient norm."

**LARS Formula**:
```python
# For each layer
local_lr = global_lr * trust_coefficient * (
    ||weights|| / (||gradients|| + weight_decay × ||weights||)
)
```

**When to use LAMB vs LARS**:
```
LARS:
- Best for ResNet, ConvNets
- Vision tasks (ImageNet)
- Batch sizes 8K-32K
- Simpler implementation

LAMB:
- Best for Transformers, BERT
- NLP tasks (language modeling)
- Batch sizes 16K-64K
- Better generalization
```

From [Revisiting LARS for Large Batch Training](https://arxiv.org/abs/2309.14053) (accessed 2025-11-16):
> "LARS employs adaptive rate scaling to improve gradient descent on a per-layer basis, enhancing training stability when batch sizes exceed 8K."

---

## Section 4: Learning Rate Scaling Rules (~100 lines)

### Linear Scaling Rule

**Golden Rule**: When batch size increases by k, multiply learning rate by k.

From [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677):
> "Applying the linear scaling rule along with a warmup strategy allows us to seamlessly scale between small and large minibatches (up to 8K)."

**Mathematical Justification**:
```python
# Small batch (batch_size = 32, lr = 0.1)
gradient_32 = compute_gradient(batch_32)  # Noisy estimate
weight_update = -0.1 * gradient_32

# Large batch (batch_size = 256, lr = 0.8)
gradient_256 = compute_gradient(batch_256)  # Less noise, 8× samples
weight_update = -0.8 * gradient_256

# Gradients average out, so:
# gradient_256 ≈ gradient_32 (expected value)
# To maintain same weight update magnitude:
# lr_new = lr_old × (batch_new / batch_old)
```

**Practical Example**:
```python
# Base configuration
base_batch_size = 256
base_lr = 0.1

# Scaled configuration (4× larger batch)
scaled_batch_size = 1024
scaled_lr = base_lr * (scaled_batch_size / base_batch_size)
scaled_lr = 0.1 * 4 = 0.4
```

### Learning Rate Warmup

**Problem**: Linear scaling can cause training instability early on.

**Solution**: Gradually increase learning rate from 0 to target over warmup period.

From [How Should the Learning Rate Change as the Batch Size Changes](https://www.geeksforgeeks.org/deep-learning/how-should-the-learning-rate-change-as-the-batch-size-changes/) (accessed 2025-11-16):
> "The linear scaling rule posits that the learning rate should be adjusted in direct proportion to the batch size, but warmup prevents early training divergence."

**Warmup Schedule**:
```python
def linear_warmup(current_step, warmup_steps, base_lr, target_lr):
    if current_step < warmup_steps:
        # Linear interpolation
        return base_lr + (target_lr - base_lr) * (current_step / warmup_steps)
    else:
        return target_lr

# Example
warmup_steps = 5000
base_lr = 0.0  # Start from zero
target_lr = 0.4  # Linearly scaled LR

for step in range(total_steps):
    lr = linear_warmup(step, warmup_steps, base_lr, target_lr)
    optimizer.param_groups[0]['lr'] = lr
```

**Common Warmup Strategies**:
```python
# 1. Linear warmup (most common)
lr_t = base_lr + (target_lr - base_lr) * (t / warmup_steps)

# 2. Gradual warmup (smoother)
lr_t = target_lr * (t / warmup_steps) ** 2

# 3. Constant warmup (simple)
lr_t = base_lr if t < warmup_steps else target_lr
```

**Warmup Duration Rules**:
```
Batch Size     | Warmup Steps
---------------|-------------
≤ 512          | 0-500 steps
512-2048       | 1000 steps
2048-8192      | 5000 steps
8192-32768     | 10000 steps
> 32768        | 20000 steps
```

### Square Root Scaling Rule

**Alternative to linear scaling for very large batches**.

From [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) (1,446 citations):
> "For extremely large batches, square root scaling provides better generalization than linear scaling."

**Formula**:
```python
# Linear scaling (standard)
lr_linear = base_lr * (batch_size / base_batch_size)

# Square root scaling (large batches)
lr_sqrt = base_lr * sqrt(batch_size / base_batch_size)

# Comparison
base_lr = 0.1
base_batch = 256

batch_8k:
  linear: 0.1 * (8192/256) = 3.2
  sqrt:   0.1 * sqrt(32) = 0.565

batch_32k:
  linear: 0.1 * 125 = 12.5  # Too large!
  sqrt:   0.1 * 11.2 = 1.12  # More stable
```

**When to use square root scaling**:
- Batch sizes > 8K
- Training shows instability with linear scaling
- Prioritizing generalization over training speed

### Practical Learning Rate Schedule

**Complete schedule combining warmup, scaling, and decay**:
```python
def get_learning_rate(step, total_steps, base_lr, batch_size,
                      base_batch_size, warmup_steps):
    # 1. Compute scaled LR
    scaling_factor = batch_size / base_batch_size
    target_lr = base_lr * scaling_factor

    # 2. Warmup phase
    if step < warmup_steps:
        lr = target_lr * (step / warmup_steps)
    else:
        # 3. Cosine decay after warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = target_lr * 0.5 * (1 + math.cos(math.pi * progress))

    return lr

# Usage
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_learning_rate(
        step, total_steps=100000, base_lr=0.001,
        batch_size=1024, base_batch_size=256, warmup_steps=5000
    ) / 0.001  # Normalize by base_lr
)
```

From [Why Warmup the Learning Rate? Underlying Mechanisms](https://proceedings.neurips.cc/paper_files/paper/2024/file/ca98452d4e9ecbc18c40da2aa0da8b98-Paper-Conference.pdf) (45 citations, accessed 2025-11-16):
> "Warmup prevents early training divergence by allowing the optimizer to build momentum gradually when using large learning rates from linear scaling."

---

## Section 5: Gradient Clipping with Accumulation (~80 lines)

### Why Gradient Clipping Matters

**Problem**: Gradient accumulation can amplify gradient explosion.

```python
# Without accumulation (batch_size = 64)
gradient_norm = 5.0  # Normal
clip_value = 1.0
clipped_gradient = min(gradient_norm, clip_value) = 1.0

# With accumulation (4 steps)
# Step 1: gradient_norm = 5.0
# Step 2: accumulated_norm = 10.0
# Step 3: accumulated_norm = 15.0
# Step 4: accumulated_norm = 20.0  # Exploded!
```

### Correct Clipping Implementation

**Option 1: Clip after accumulation** (most common):
```python
accumulation_steps = 4
max_grad_norm = 1.0
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        # Clip BEFORE optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

**Option 2: Clip every micro-batch** (more conservative):
```python
accumulation_steps = 4
max_grad_norm = 1.0
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    # Clip after each backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Which to use?**:
```
Clip after accumulation (Option 1):
- Standard practice
- Equivalent to large batch
- Use max_grad_norm = 1.0

Clip every micro-batch (Option 2):
- More aggressive clipping
- Better for unstable training
- Use max_grad_norm = 1.0 / sqrt(accumulation_steps)
```

### Global Norm Clipping

**Most effective clipping method for Transformers**:
```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    Clip gradients by global norm.

    Args:
        parameters: Model parameters
        max_norm: Maximum norm value
        norm_type: Type of norm (2.0 = L2 norm)
    """
    # Compute global norm across all parameters
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type)
            for p in parameters if p.grad is not None
        ]),
        norm_type
    )

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # Only clip if necessary
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    return total_norm
```

**Practical clipping values**:
```python
# Transformers (BERT, GPT)
max_grad_norm = 1.0

# Vision models (ResNet, ViT)
max_grad_norm = 5.0

# RNNs (LSTM, GRU)
max_grad_norm = 0.5

# Large batch (>8K)
max_grad_norm = 0.25  # More conservative
```

### Per-Parameter Clipping

**Alternative: Clip each parameter independently**:
```python
def clip_grad_value_(parameters, clip_value):
    """
    Clip gradients by value (elementwise).
    Less common than global norm clipping.
    """
    for p in parameters:
        if p.grad is not None:
            p.grad.data.clamp_(-clip_value, clip_value)

# Usage
clip_grad_value_(model.parameters(), clip_value=10.0)
```

**When to use per-parameter clipping**:
- RNNs with vanishing gradients
- Very deep networks (>100 layers)
- Debugging gradient explosion

---

## Section 6: Distributed Gradient Accumulation (~100 lines)

### DDP + Gradient Accumulation

**Challenge**: Combine Data Parallel training with gradient accumulation.

From [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html):
> "When using gradient accumulation with DDP, you must manually control when gradients are synchronized across GPUs."

**Problem: Unnecessary synchronization**:
```python
# Bad: Syncs gradients every backward (slow!)
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Implicit all-reduce here!

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Result: 4× slower (syncing 4 times instead of 1)
```

**Solution: Use `no_sync()` context**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # Skip sync for all but last accumulation step
    if (i + 1) % accumulation_steps != 0:
        with model.no_sync():
            loss = model(batch) / accumulation_steps
            loss.backward()  # No sync!
    else:
        loss = model(batch) / accumulation_steps
        loss.backward()  # Sync here
        optimizer.step()
        optimizer.zero_grad()
```

**Performance improvement**:
```
Without no_sync():
  4 GPUs, 4 accumulation steps
  Time per update: 100ms × 4 syncs = 400ms

With no_sync():
  Time per update: 100ms × 1 sync = 100ms
  Speedup: 4×!
```

### DeepSpeed Gradient Accumulation

**Automatic handling with DeepSpeed ZeRO**:

From [DeepSpeed ZeRO Optimizer](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md):
> "DeepSpeed automatically optimizes gradient accumulation with ZeRO, reducing communication overhead by 50-80%."

```python
# DeepSpeed config
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 8,  # Auto-computed: 1024/(32×4 GPUs)

    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,  # Overlap communication with computation
        "contiguous_gradients": true,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}

# Training loop (same as normal)
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)  # DeepSpeed handles accumulation
    model.step()
```

**DeepSpeed benefits**:
- Automatic accumulation step calculation
- Gradient bucketing for efficient all-reduce
- Communication overlap (compute while syncing)
- ZeRO optimizer state partitioning

### FSDP Gradient Accumulation

**Fully Sharded Data Parallel with accumulation**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=my_wrap_policy,
    limit_all_gathers=True  # Important for accumulation!
)

accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # FSDP requires no manual no_sync
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

From [FSDP vs DeepSpeed Comparison](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md):
> "FSDP automatically handles gradient synchronization during accumulation, with built-in optimizations for reducing all-gather calls."

### Communication-Efficient Gradient Accumulation

**Advanced: Gradient compression**:
```python
# Compress gradients during sync (FP16 instead of FP32)
model = DDP(
    model,
    gradient_as_bucket_view=True,
    broadcast_buffers=False,
)

# Hook for FP16 gradient compression
def fp16_compress_hook(state, bucket):
    # Compress gradients to FP16
    compressed = bucket.buffer().half()
    # All-reduce in FP16 (2× less bandwidth)
    compressed.div_(world_size)
    dist.all_reduce(compressed)
    # Decompress back to FP32
    bucket.buffer().copy_(compressed.float())
    return bucket.buffer()

model.register_comm_hook(state=None, hook=fp16_compress_hook)
```

**Communication savings**:
```
8 GPUs, 7B model, gradient_accumulation_steps = 4

Without compression:
  Gradient size: 28GB per sync
  Bandwidth: 28GB × 1 sync = 28GB

With FP16 compression:
  Gradient size: 14GB per sync
  Bandwidth: 14GB × 1 sync = 14GB
  Savings: 50%
```

---

## Section 7: Performance Considerations (~90 lines)

### Throughput Analysis

**Measuring samples/second with gradient accumulation**:
```python
import time

accumulation_steps = 4
micro_batch_size = 8
effective_batch = micro_batch_size * accumulation_steps * num_gpus

start_time = time.time()
total_samples = 0

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

        # Count after optimizer step
        total_samples += effective_batch

        if total_samples >= 10000:
            break

elapsed_time = time.time() - start_time
throughput = total_samples / elapsed_time
print(f"Throughput: {throughput:.2f} samples/sec")
```

**Expected throughput impact**:
```
Baseline (no accumulation):
  Batch size: 32
  Throughput: 1000 samples/sec

Gradient accumulation (4 steps):
  Micro-batch: 8, effective: 32
  Throughput: 950 samples/sec  # 5% slower

Why slower?
- More kernel launches (4× forward/backward)
- Less GPU parallelism (smaller batches)
- But: same effective batch size!
```

### Memory-Time Tradeoffs

**Choosing optimal accumulation steps**:
```python
def optimal_accumulation_steps(
    target_batch_size,
    available_memory_gb,
    model_size_gb,
    activation_memory_per_sample_mb
):
    """
    Calculate optimal gradient accumulation steps.

    Returns:
        (micro_batch_size, accumulation_steps)
    """
    # Memory for model + optimizer
    overhead = model_size_gb * 10  # Model + optimizer states

    # Available for activations
    activation_budget = (available_memory_gb - overhead) * 1024

    # Max micro-batch that fits
    micro_batch = int(activation_budget / activation_memory_per_sample_mb)

    # Accumulation steps needed
    accumulation_steps = target_batch_size // micro_batch

    return micro_batch, accumulation_steps

# Example: GPT-2 Medium
micro_batch, accum = optimal_accumulation_steps(
    target_batch_size=256,
    available_memory_gb=16,
    model_size_gb=1.4,
    activation_memory_per_sample_mb=50
)
print(f"Micro-batch: {micro_batch}, Accumulation: {accum}")
# Output: Micro-batch: 90, Accumulation: 3
```

### Profiling Gradient Accumulation

**Identify bottlenecks with PyTorch Profiler**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for i, batch in enumerate(dataloader):
        loss = model(batch) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if i >= 10:  # Profile 10 steps
            break

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**What to look for**:
```
Key metrics:
1. GPU utilization: Should be >90%
   - Low utilization → increase micro-batch size

2. Memory usage: Should be near max
   - Headroom → can increase micro-batch

3. Time breakdown:
   - Forward pass: 40%
   - Backward pass: 50%
   - Optimizer: 10%
   - If optimizer >15% → too many accumulation steps
```

### Optimal Batch Size Selection

**Decision framework**:
```python
# Step 1: Find maximum micro-batch that fits
max_micro_batch = find_max_batch_size(model, gpu_memory)

# Step 2: Choose target effective batch
# Rule of thumb: 0.1% to 1% of dataset size
dataset_size = 1_000_000
target_batch = int(dataset_size * 0.001)  # 1000

# Step 3: Calculate accumulation steps
num_gpus = 8
accumulation_steps = target_batch // (max_micro_batch * num_gpus)

# Step 4: Adjust for performance
if accumulation_steps > 8:
    # Too many steps → slower training
    # Reduce target batch or add more GPUs
    target_batch = max_micro_batch * num_gpus * 8
    accumulation_steps = 8

print(f"Micro-batch: {max_micro_batch}")
print(f"GPUs: {num_gpus}")
print(f"Accumulation: {accumulation_steps}")
print(f"Effective batch: {max_micro_batch * num_gpus * accumulation_steps}")
```

**Performance guidelines**:
```
Accumulation Steps | Performance Impact
-------------------|-------------------
1-2                | Optimal (no overhead)
3-4                | Good (<5% slower)
5-8                | Acceptable (<10% slower)
9-16               | Noticeable (<20% slower)
>16                | Significant (>20% slower)
```

---

## Section 8: arr-coc-0-1 Gradient Accumulation Strategy (~80 lines)

### Training Configuration

**arr-coc-0-1 gradient accumulation setup**:

From [arr-coc-0-1/training/train.py](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/train.py):
```python
gradient_accumulation_steps: int = 4

# Effective batch calculation
effective_batch_size = (
    micro_batch_size × gradient_accumulation_steps × num_gpus
    = 32 × 4 × 8 = 1024
)
```

**Why 1024 effective batch?**:
- Stable training for vision-language models
- Linear scaling rule: lr = 1e-4 × (1024/256) = 4e-4
- Fits in 8× A100 40GB GPUs (32GB per micro-batch)
- Sweet spot: large enough for stability, small enough for generalization

### Memory Budget Analysis

**arr-coc-0-1 memory breakdown (per GPU)**:
```
Model components:
- Vision encoder (frozen): 1.2GB
- Qwen3-VL decoder: 4.8GB
- ARR-COC adapter (trainable): 0.8GB
- Total parameters: 6.8GB

Training memory:
- FP16 parameters: 6.8GB
- Optimizer states (AdamW): 20.4GB
- Gradients: 6.8GB
- Activations (micro-batch=32): 8GB
- Total: ~42GB

With gradient checkpointing:
- Activations reduced: 8GB → 2GB
- Total: ~36GB (fits in A100 40GB!)
```

**Without gradient accumulation would require**:
```
Batch size 1024 on single GPU:
- Activations: 8GB × (1024/32) = 256GB
- Total: 284GB (doesn't fit!)

Solution: Gradient accumulation
- Split across 8 GPUs: 1024 / 8 = 128 per GPU
- Further split: 128 / 4 = 32 micro-batch
- Activations: 8GB (fits comfortably)
```

### Training Loop Implementation

**Simplified training loop with accumulation**:
```python
def train_epoch(
    model, dataloader, optimizer, scheduler,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(
            pixel_values=batch['images'],
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )

        # Scale loss by accumulation steps
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        # Update every N steps
        if (i + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping (global norm)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log effective batch loss
            total_loss += outputs.loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / 25  # Every 25 updates
                print(f"Step {(i+1)//4}, Loss: {avg_loss:.4f}")
                total_loss = 0
```

### Learning Rate Schedule

**arr-coc-0-1 LR schedule with warmup**:
```python
from torch.optim.lr_scheduler import OneCycleLR

# Base configuration
base_batch_size = 256
base_lr = 1e-4

# Scaled configuration
effective_batch_size = 1024
scaled_lr = base_lr * (effective_batch_size / base_batch_size)
scaled_lr = 4e-4  # 4× larger

# OneCycle scheduler with warmup
scheduler = OneCycleLR(
    optimizer,
    max_lr=scaled_lr,
    epochs=10,
    steps_per_epoch=len(dataloader) // gradient_accumulation_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25.0,  # Initial lr = max_lr / 25
    final_div_factor=1e4
)

# LR progression:
# Steps 0-500:    1.6e-5 → 4e-4 (warmup)
# Steps 500-5000: 4e-4 → 4e-8 (cosine decay)
```

### Generation Loss Fix

**Critical: Variable sequence length handling**:

From [PyTorch Gradient Accumulation Reproducibility](https://muellerzr.github.io/blog/gradient_accumulation_part2.html) (accessed 2025-11-16):
> "For generation tasks, gradient accumulation requires accounting for variable sequence lengths across accumulation steps to maintain reproducibility."

**arr-coc-0-1 corrected loss calculation**:
```python
def compute_loss_with_accumulation(model, batch_samples, num_items_in_batch):
    """
    Compute loss accounting for variable sequence lengths.

    Args:
        batch_samples: List of micro-batches
        num_items_in_batch: Total non-padding tokens across all micro-batches
    """
    total_loss = 0

    for batch in batch_samples:
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch['labels']

        # Compute loss with "sum" reduction
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum'
        )

        # Normalize by TOTAL tokens (not micro-batch tokens)
        loss = loss / num_items_in_batch
        loss.backward()
        total_loss += loss.item()

    return total_loss

# Usage
num_update_steps = len(dataloader) // gradient_accumulation_steps

for update_step in range(num_update_steps):
    # Prefetch accumulation_steps batches
    batch_samples = [next(dataloader_iter) for _ in range(gradient_accumulation_steps)]

    # Count total non-padding tokens
    num_items = sum([
        (batch['labels'] != -100).sum().item()
        for batch in batch_samples
    ])

    # Compute loss with proper normalization
    loss = compute_loss_with_accumulation(model, batch_samples, num_items)

    optimizer.step()
    optimizer.zero_grad()
```

**Why this matters for arr-coc-0-1**:
- Vision-language pairs have variable text lengths
- Image descriptions: 10-100 tokens (10× variance!)
- Without proper normalization: training instability
- With correction: reproducible, stable training

### Deployment Considerations

**Inference doesn't use gradient accumulation**:
```python
# Training mode (gradient accumulation)
effective_batch = 1024
micro_batch = 32
accumulation_steps = 4

# Inference mode (no accumulation)
inference_batch = 128  # Much larger!
# No backward pass → only activations use memory
# Can fit 4× larger batches
```

**Real-world arr-coc-0-1 usage**:
```
Training:
- 8× A100 GPUs
- Micro-batch: 32 per GPU
- Accumulation: 4 steps
- Effective batch: 1024
- Throughput: ~120 samples/sec
- Cost: $20/hour (Vertex AI)

Inference (Gradio demo):
- 1× A100 GPU
- Batch size: 64
- Throughput: ~800 samples/sec
- Cost: $3/hour (HuggingFace Space)
```

---

## Sources

**Source Documents:**
- [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md) - DeepSpeed ZeRO gradient accumulation
- [karpathy/distributed-training/03-fsdp-vs-deepspeed.md](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - FSDP accumulation strategies
- [karpathy/training-llms/01-four-stage-pipeline.md](../karpathy/training-llms/01-four-stage-pipeline.md) - LLM training fundamentals

**Web Research:**
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Linear scaling rule (4,780 citations, accessed 2025-11-16)
- [PyTorch Gradient Accumulation Guide](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3) - HuggingFace implementation (accessed 2025-11-16)
- [Gradient Accumulation Reproducibility](https://muellerzr.github.io/blog/gradient_accumulation_part2.html) - Variable sequence length fix (accessed 2025-11-16)
- [How to Implement Gradient Accumulation in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5) - W&B tutorial (accessed 2025-11-16)
- [Large Batch Optimization with LAMB](https://arxiv.org/abs/1904.00962) - LAMB optimizer (1,200+ citations)
- [Large Batch Training with LARS](https://arxiv.org/abs/1708.03888) - LARS optimizer (2,600+ citations)
- [Revisiting LARS for Large Batch Training](https://arxiv.org/abs/2309.14053) - LARS improvements (accessed 2025-11-16)
- [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) - Square root scaling (1,446 citations)
- [Why Warmup the Learning Rate?](https://proceedings.neurips.cc/paper_files/paper/2024/file/ca98452d4e9ecbc18c40da2aa0da8b98-Paper-Conference.pdf) - Warmup mechanisms (45 citations, accessed 2025-11-16)
- [How Should Learning Rate Scale with Batch Size](https://www.geeksforgeeks.org/deep-learning/how-should-the-learning-rate-change-as-the-batch-size-changes/) - GeeksforGeeks guide (accessed 2025-11-16)

**Implementation References:**
- [arr-coc-0-1/training/train.py](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/train.py) - Production gradient accumulation
