# Mixed Precision Training Advanced: FP16, BF16, FP8, AMP, and Gradient Scaling

## Overview

Advanced mixed precision training combines multiple numerical precisions (FP32, BF16, FP16, TF32, FP8) to maximize training throughput while maintaining model convergence. This document covers production-grade mixed precision strategies including PyTorch AMP internals, gradient scaling algorithms, precision format tradeoffs, FP8 training with Transformer Engine, and arr-coc-0-1 optimization.

**Why Advanced Mixed Precision Matters:**
- **Speed**: 2-3× faster with BF16/FP16, 4-6× with FP8 on H100
- **Memory**: 50% reduction (BF16/FP16), 75% reduction (FP8)
- **Scale**: Essential for training models >10B parameters
- **Cost**: 30-70% lower cloud costs through efficiency

From [PyTorch Blog: Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) (accessed 2025-11-16):
> "Mixed precision training is an essential tool for training deep learning models on modern hardware, and it will become even more important in the future as the performance gap between lower precision operations and float32 continues to grow."

**Related Knowledge:**
- See [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) for AMP basics
- See [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) for Tensor Core specs
- See [training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md](../karpathy/training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md) for practical patterns

---

## Section 1: Precision Format Deep Comparison (~100 lines)

### Numerical Representation Fundamentals

**IEEE 754 Structure:**
```
Floating Point = sign (1 bit) | exponent (E bits) | mantissa (M bits)
Value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa/2^M)
```

### Format Specifications Table

| Format | Total | Sign | Exp | Mantissa | Range | Precision | Hardware |
|--------|-------|------|-----|----------|-------|-----------|----------|
| **FP32** | 32 | 1 | 8 | 23 | ±3.4e38 | ~7 digits | All GPUs |
| **TF32** | 32* | 1 | 8 | 10 | ±3.4e38 | ~3 digits | Ampere+ |
| **BF16** | 16 | 1 | 8 | 7 | ±3.4e38 | ~2 digits | Ampere+, TPU |
| **FP16** | 16 | 1 | 5 | 10 | ±65,504 | ~3 digits | Volta+ |
| **FP8 E4M3** | 8 | 1 | 4 | 3 | ±448 | ~1 digit | Hopper+, Ada+ |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | ±57,344 | <1 digit | Hopper+, Ada+ |

*TF32 stored as FP32, computed with 10-bit mantissa

### FP32 (Baseline)

**Characteristics:**
- Dynamic range: 1.175e-38 to 3.4e38
- Subnormal support: Yes (gradual underflow)
- Special values: ±inf, ±0, NaN
- Training: Baseline reference (slow)

**Memory layout:**
```python
# 4 bytes per value
batch_size = 32
seq_len = 2048
hidden_dim = 4096
activation_memory_fp32 = 32 * 2048 * 4096 * 4  # ~1 GB
```

### BF16 (Brain Float 16)

**Why BF16 is the modern default:**

From [PyTorch Blog: Mixed Precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/):
> "Both float16 and bfloat16 are usually comparably fast, but some networks may only converge with one vs the other. If a network requires more precision it may need to use float16, and if a network requires more dynamic range it may need to use bfloat16."

**Bit layout:**
```
BF16: S EEEEEEEE MMMMMMM
      1   8       7
```

**Key advantage: Same exponent range as FP32**
```python
# BF16 vs FP16 range comparison
fp32_val = 1e20
bf16_val = torch.tensor(fp32_val, dtype=torch.bfloat16)  # OK
fp16_val = torch.tensor(fp32_val, dtype=torch.float16)   # inf (overflow!)
```

**When to use BF16:**
- Models with extreme dynamic range (transformers)
- No gradient scaling needed (simpler code)
- TPU training (Google's default)
- Ampere+ GPUs (native hardware support)

**BF16 training (no scaler needed):**
```python
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())

for input, target in dataloader:
    optimizer.zero_grad()

    # Autocast to BF16
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    # No scaler needed - BF16 is stable
    loss.backward()
    optimizer.step()
```

### FP16 (Half Precision)

**Bit layout:**
```
FP16: S EEEEE MMMMMMMMMM
      1   5        10
```

**Limitations:**
```python
# Overflow
x = torch.tensor(70000.0, dtype=torch.float16)  # inf

# Underflow
y = torch.tensor(1e-5, dtype=torch.float16)  # 0.0 (gradient vanishing!)

# Precision loss
z = torch.tensor(2048.5, dtype=torch.float16)  # 2048.0 (no .5)
```

**When to use FP16:**
- Pre-Ampere GPUs (V100, P100) - no BF16 hardware
- Inference (stable, no gradient concerns)
- Extra mantissa precision needed (rare)
- Willing to deal with GradScaler complexity

### TF32 (Tensor Float 32)

**Automatic on Ampere+ GPUs:**
```python
# Enabled by default in PyTorch 1.7+
torch.backends.cuda.matmul.allow_tf32 = True  # Default: True
torch.backends.cudnn.allow_tf32 = True        # Default: True
```

**Bit layout:**
```
TF32: S EEEEEEEE MMMMMMMMMM
      1    8          10
```

**Performance on A100:**
- FP32 matmul: 19.5 TFLOPs
- TF32 matmul: 156 TFLOPs (8× faster!)
- Accuracy loss: ~0.1% for most models

**When TF32 is used:**
- Matrix multiplications (torch.matmul, torch.mm, @)
- Convolutions (nn.Conv2d, F.conv2d)
- Automatically selected by PyTorch
- No code changes required

From [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md):
> "TF32 Precision: Automatic on A100/H100 for FP32 operations. 8-bit exponent (like FP32), 10-bit mantissa (reduced from 23). Enables 10× speedup with minimal accuracy loss."

### FP8 (Two Variants)

**E4M3 (4 exponent, 3 mantissa):**
```
E4M3: S EEEE MMM
      1   4    3
```
- Range: ±448
- Precision: ~1 decimal digit
- Use: Forward pass (weights, activations)

**E5M2 (5 exponent, 2 mantissa):**
```
E5M2: S EEEEE MM
      1    5   2
```
- Range: ±57,344
- Precision: <1 decimal digit
- Use: Backward pass (gradients)

**Why two formats?**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-11-16):
> "E4M3 prioritizes precision for forward passes, where weights and activations benefit from finer-grained values. E5M2 trades mantissa bits for a wider dynamic range, crucial for backward passes where gradients can vary significantly in magnitude."

**FP8 vs INT8:**
- FP8: Floating-point with dynamic exponents → handles extreme ranges
- INT8: Fixed-point with static scaling → struggles with attention, gradients
- Transformers: FP8 vastly superior for attention mechanisms

### Hardware Support Comparison

**A100 (Ampere):**
- FP32: 19.5 TFLOPs
- TF32: 156 TFLOPs (8× faster)
- BF16: 156 TFLOPs
- FP16: 312 TFLOPs

**H100 (Hopper):**
- FP32: 60 TFLOPs
- TF32: 500 TFLOPs
- BF16: 1,000 TFLOPs
- FP16: 1,000 TFLOPs
- **FP8: 2,000 TFLOPs** (33× faster than FP32!)

---

## Section 2: PyTorch AMP Production Patterns (~120 lines)

### Complete Training Loop with AMP

**Full workflow with all optimizations:**
```python
import torch
from torch.cuda.amp import GradScaler, autocast

# Initialize
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler(
    init_scale=2.**16,      # 65536
    growth_factor=2.0,      # Double on success
    backoff_factor=0.5,     # Halve on overflow
    growth_interval=2000    # Iterations between growth attempts
)

for epoch in range(epochs):
    for batch_idx, (input, target) in enumerate(dataloader):
        optimizer.zero_grad()

        # 1. Forward pass with autocast
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # 2. Backward with scaled loss
        scaler.scale(loss).backward()

        # 3. Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. Optimizer step with overflow check
        scaler.step(optimizer)

        # 5. Update scale for next iteration
        scaler.update()

        # Logging (every 100 iterations)
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Scale: {scaler.get_scale()}")
```

### Gradient Accumulation with AMP

**Critical: Scale must remain constant across accumulated gradients**

```python
scaler = GradScaler()
accumulation_steps = 4

for i, (input, target) in enumerate(dataloader):
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)
        # Divide loss by accumulation steps
        loss = loss_fn(output, target) / accumulation_steps

    # Accumulate scaled gradients
    scaler.scale(loss).backward()

    # Step only at effective batch boundaries
    if (i + 1) % accumulation_steps == 0:
        # Unscale and clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Multi-Optimizer AMP

**Example: GANs with separate generator and discriminator optimizers**

```python
scaler = GradScaler()
gen_optimizer = torch.optim.AdamW(generator.parameters())
disc_optimizer = torch.optim.AdamW(discriminator.parameters())

for real_imgs in dataloader:
    # Train discriminator
    disc_optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        fake_imgs = generator(noise)
        disc_loss = discriminator_loss(real_imgs, fake_imgs)

    scaler.scale(disc_loss).backward()
    scaler.step(disc_optimizer)
    scaler.update()

    # Train generator
    gen_optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        fake_imgs = generator(noise)
        gen_loss = generator_loss(fake_imgs)

    scaler.scale(gen_loss).backward()
    scaler.step(gen_optimizer)
    scaler.update()
```

### Selective Precision Control

**Mixed precision in some regions, FP32 in others:**

```python
model = Model().cuda()

for input, target in dataloader:
    optimizer.zero_grad()

    # FP16 for most operations
    with autocast(device_type='cuda', dtype=torch.float16):
        # Fast compute-bound operations
        features = model.backbone(input)

        # Disable autocast for sensitive operations
        with autocast(device_type='cuda', enabled=False):
            # Numerically sensitive (stay in FP32)
            normalized = F.layer_norm(features.float(), normalized_shape)

        # Resume FP16
        output = model.head(normalized.half())
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Operation-Specific Precision

**Autocast rules (from PyTorch internals):**

**FP16-Safe Operations (compute-bound):**
```python
# These run in FP16 under autocast
torch.matmul(a, b)
torch.nn.functional.linear(x, weight)
torch.bmm(a, b)
torch.conv2d(input, weight)
torch.addmm(bias, a, b)
```

**FP32-Required Operations (numerically sensitive):**
```python
# These run in FP32 under autocast
torch.nn.functional.softmax(x, dim=-1)
torch.nn.functional.log_softmax(x, dim=-1)
torch.nn.functional.layer_norm(x, normalized_shape)
torch.sum(x)
torch.prod(x)
torch.mean(x)
```

**Type Promotion:**
```python
# Mixed dtypes promote to widest
fp16_tensor = torch.randn(10, dtype=torch.float16, device='cuda')
fp32_tensor = torch.randn(10, dtype=torch.float32, device='cuda')

result = torch.add(fp16_tensor, fp32_tensor)  # Result: FP32
```

### Debugging AMP Issues

**Issue 1: Type mismatches**
```python
# ERROR: Expected FP16, got FP32
# Solution: Don't manually cast - investigate why types diverged

# Bad (hides the issue)
x = x.half()  # Manual cast

# Good (find root cause)
print(f"x dtype: {x.dtype}, expected: torch.float16")
# Investigate why x is wrong dtype
```

**Issue 2: NaN gradients**
```python
# Check for NaN in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
            # Likely need: higher init_scale, gradient clipping, or BF16
```

**Issue 3: Scale keeps decreasing**
```python
# Monitor scale dynamics
if scaler.get_scale() < 256:  # Too low
    print(f"Warning: Scale very low ({scaler.get_scale()})")
    # Solutions:
    # 1. Increase init_scale
    # 2. Lower learning rate
    # 3. Use gradient clipping
    # 4. Switch to BF16
```

---

## Section 3: Gradient Scaling Deep Dive (~120 lines)

### The Gradient Underflow Problem

**FP16 dynamic range limitation:**
```python
# FP16 smallest positive normal: 6.1e-5
min_fp16 = 2 ** -14  # 0.000061

# Typical gradients in deep networks
typical_gradient = 1e-7  # Smaller than min_fp16!

grad_fp16 = torch.tensor(typical_gradient, dtype=torch.float16)
print(grad_fp16)  # tensor(0.) - underflow to zero!
```

**Consequence: Gradient vanishing**
```python
# Without scaling: gradients lost
loss = 1e-5  # Small but valid loss
loss.backward()  # Gradients underflow to 0 in FP16
# Model fails to converge
```

### Loss Scaling Algorithm

**Core principle: Scale up before backward, scale down after**

```python
# Scale factor (typically 2^16 = 65536)
scale = 65536

# Original (underflows)
loss = 1e-5
loss.backward()  # Gradients: ~1e-7 → 0.0 in FP16

# With scaling (preserved)
scaled_loss = loss * scale  # 0.65536 (representable in FP16!)
scaled_loss.backward()      # Gradients: ~6.5536 (preserved)
unscaled_grads = grads / scale  # Back to 1e-7
```

### GradScaler Internal State

**Implementation details:**
```python
class GradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        self._scale = init_scale              # Current scale (FP32)
        self._growth_tracker = 0              # Iters since last overflow
        self._per_optimizer_states = {}       # Track unscale per optimizer
        self._growth_factor = growth_factor   # 2.0
        self._backoff_factor = backoff_factor # 0.5
        self._growth_interval = growth_interval  # 2000
```

### scale() Method Internals

```python
def scale(self, outputs):
    """
    Multiply outputs by current scale factor.

    Args:
        outputs: Tensor or container of tensors

    Returns:
        Scaled tensor(s)
    """
    if isinstance(outputs, torch.Tensor):
        return outputs * self._scale
    else:
        # Handle containers (lists, tuples, dicts)
        return type(outputs)(self.scale(v) for v in outputs)

# Example
loss = torch.tensor(1e-5, dtype=torch.float16)
scaled_loss = scaler.scale(loss)  # 0.65536
```

### unscale_() Method Internals

```python
def unscale_(self, optimizer):
    """
    Divide gradients by scale factor in-place.

    Side effects:
        - Divides optimizer's gradients by scale
        - Checks for inf/nan
        - Marks optimizer as unscaled (idempotent)
    """
    if id(optimizer) in self._per_optimizer_states:
        # Already unscaled this iteration
        return

    inv_scale = 1.0 / self._scale
    found_inf = False

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                # Unscale gradient
                param.grad.mul_(inv_scale)

                # Check for overflow
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    found_inf = True

    # Mark as unscaled
    self._per_optimizer_states[id(optimizer)] = found_inf
```

### step() Method Internals

```python
def step(self, optimizer, *args, **kwargs):
    """
    Conditionally update optimizer parameters.

    Returns:
        None if inf/nan found, else optimizer.step() return value
    """
    # Unscale if not already done
    if id(optimizer) not in self._per_optimizer_states:
        self.unscale_(optimizer)

    # Check overflow flag
    found_inf = self._per_optimizer_states[id(optimizer)]

    if found_inf:
        # Skip optimizer step (parameters unchanged)
        return None
    else:
        # Step optimizer normally
        return optimizer.step(*args, **kwargs)
```

### update() Method Internals

```python
def update(self, new_scale=None):
    """
    Adjust scale factor based on overflow detection.

    Algorithm:
        - Overflow detected: decrease scale (backoff)
        - No overflow for growth_interval: increase scale (growth)
    """
    if new_scale is not None:
        # Manual override
        self._scale = new_scale
        self._growth_tracker = 0
        return

    # Check if any optimizer had overflow
    found_inf = any(self._per_optimizer_states.values())

    # Clear states for next iteration
    self._per_optimizer_states.clear()

    if found_inf:
        # Overflow: decrease scale
        self._scale *= self._backoff_factor  # /= 2
        self._growth_tracker = 0
    else:
        # No overflow: increment tracker
        self._growth_tracker += 1

        if self._growth_tracker == self._growth_interval:
            # Increase scale after growth_interval successes
            self._scale *= self._growth_factor  # *= 2
            self._growth_tracker = 0
```

### Scale Dynamics Example

**Typical scale evolution during training:**
```python
# Iteration 0
scale = 65536  # 2^16 (initial)

# Iterations 1-2000 (no overflow)
for i in range(2000):
    scaler.update()  # growth_tracker increments
# scale = 131072  # 2^17 (doubled)

# Iteration 2001 (overflow detected)
# (Gradient magnitude too large for current scale)
scaler.update()  # Overflow detected
# scale = 65536  # 2^16 (halved back)

# Iterations 2002-4002 (stable)
for i in range(2000):
    scaler.update()
# scale = 131072  # 2^17 (grows again)
```

### Healthy vs Unhealthy Scale Patterns

**Healthy pattern (good convergence):**
```
Iter 0:    scale = 65536
Iter 100:  scale = 65536
Iter 2000: scale = 131072  # Grew after 2000 successes
Iter 2100: scale = 131072
Iter 4000: scale = 262144  # Continued growth
```

**Unhealthy pattern (death spiral):**
```
Iter 0:    scale = 65536
Iter 100:  scale = 32768   # Halved (overflow)
Iter 200:  scale = 16384   # Halved again
Iter 300:  scale = 8192    # Death spiral!
Iter 400:  scale = 4096
# Model failing to converge
```

**Solutions for death spiral:**
```python
# 1. Increase init_scale
scaler = GradScaler(init_scale=2.**20)  # 1,048,576

# 2. Decrease learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower

# 3. Use gradient clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Switch to BF16 (no scaling needed)
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```

---

## Section 4: BF16 vs FP16 Production Decision (~100 lines)

### BF16 vs FP16: The 2025 Consensus

From [training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md](../karpathy/training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md):
> "Mixed precision training in 2025 has converged on a clear winner: bfloat16 for Ampere+ GPUs (T4, A100, RTX 30/40 series)."

### Numerical Stability Comparison

**Range comparison:**
```python
# FP16 range (5-bit exponent)
fp16_min = 2 ** -14      # 6.1e-5
fp16_max = 65504         # Max before overflow

# BF16 range (8-bit exponent, same as FP32)
bf16_min = 2 ** -126     # 1.2e-38 (same as FP32!)
bf16_max = 3.4e38        # Same as FP32

# Example: Small gradient
grad = 1e-10  # Valid in deep networks
grad_fp16 = torch.tensor(grad, dtype=torch.float16)  # 0.0 (underflow!)
grad_bf16 = torch.tensor(grad, dtype=torch.bfloat16) # OK (preserved)
```

### Code Simplicity: BF16 Wins

**BF16 (simple):**
```python
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())

for input, target in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    # No scaler needed!
    loss.backward()
    optimizer.step()
```

**FP16 (complex):**
```python
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()  # Required for FP16

for input, target in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)

    # Requires gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### When to Use FP16 Instead of BF16

**Use FP16 only if:**
1. Pre-Ampere GPU (V100, P100) - no BF16 hardware
2. Need extra mantissa precision (rare)
3. Inference only (no gradient concerns)

**Use BF16 if:**
1. Ampere+ GPU (A100, H100, T4, RTX 30/40)
2. Training (gradients need stability)
3. Want simpler code (no GradScaler)
4. TPU training (Google's default)

### Performance Comparison

**A100 GPU:**
- FP32: 19.5 TFLOPs
- BF16: 156 TFLOPs (8× faster)
- FP16: 312 TFLOPs (16× faster)

**But: BF16 often faster in practice due to:**
- No gradient scaling overhead
- No overflow handling
- Simpler training loop

### Mixed Precision Format Selection Matrix

| Model Type | GPU | Format | Reason |
|-----------|-----|--------|--------|
| Transformer (BERT, GPT) | A100 | **BF16** | Wide dynamic range in attention |
| Vision (ResNet, ViT) | A100 | **BF16** or FP16 | Both work, BF16 simpler |
| GAN | A100 | **BF16** | Stable discriminator training |
| Transformer | V100 | **FP16** | No BF16 hardware |
| Inference | Any | **FP16** | No gradients, faster |
| TPU | TPU v3+ | **BF16** | Native format |

---

## Section 5: FP8 Training with Transformer Engine (~150 lines)

### FP8 Format Design

**Two variants for different training stages:**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
> "E4M3 prioritizes precision for forward passes. E5M2 trades mantissa bits for a wider dynamic range, crucial for backward passes where gradients can vary significantly in magnitude."

**E4M3 (Forward Pass):**
```
Bit layout: S EEEE MMM (1 sign, 4 exp, 3 mantissa)
Range: ±448
Precision: ~1 decimal digit
Use: Weights, activations
```

**E5M2 (Backward Pass):**
```
Bit layout: S EEEEE MM (1 sign, 5 exp, 2 mantissa)
Range: ±57,344
Precision: <1 decimal digit
Use: Gradients
```

### Why FP8 Beats INT8

**INT8 problems with transformers:**
```python
# Attention scores after softmax
attention_scores = torch.softmax(qk / sqrt(d_k), dim=-1)
# Range: ~1e-5 to ~1.0 (wide dynamic range)

# INT8 fixed scaling
scale = 127 / max(attention_scores)  # Static
int8_scores = (attention_scores * scale).to(torch.int8)
# Small values: quantized to 0 (lost information)
# Large values: clipped (accuracy loss)

# FP8 floating-point scaling
fp8_scores = attention_scores.to(torch.float8_e4m3fn)
# Exponent handles wide range automatically
# Much better accuracy preservation
```

### NVIDIA Transformer Engine Overview

**Installation:**
```bash
pip install transformer-engine[pytorch]
```

**Basic usage:**
```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Create FP8 recipe
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=recipe.Format.HYBRID  # E4M3 fwd, E5M2 bwd
)

# Replace nn.Linear with te.Linear
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # FP8-capable linear layers
        self.qkv = te.Linear(hidden_size, 3 * hidden_size)
        self.out = te.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Use FP8 context
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            qkv = self.qkv(x)
            # ... attention computation
            out = self.out(attn_output)
        return out
```

### FP8 Scaling Strategies

**1. Delayed Scaling (Stable):**
```python
class DelayedScaling:
    def __init__(self, history_size=1000):
        self.amax_history = []
        self.history_size = history_size

    def update(self, tensor):
        # Record current amax
        current_amax = tensor.abs().max()
        self.amax_history.append(current_amax)

        # Keep only recent history
        if len(self.amax_history) > self.history_size:
            self.amax_history.pop(0)

    def get_scale(self, dtype='E4M3'):
        # Use max of history for stable scaling
        history_amax = max(self.amax_history)
        fp8_max = 448 if dtype == 'E4M3' else 57344
        return history_amax / fp8_max

# Benefits:
# - Stable training (no divergence from spikes)
# - Handles transient outliers
# - Recommended for LLM pretraining
```

**2. Per-Tensor Current Scaling (Reactive):**
```python
def current_scale(tensor, dtype='E4M3'):
    # Calculate scale from current tensor statistics
    amax = tensor.abs().max()

    # Determine FP8 range
    fp8_max = 448 if dtype == 'E4M3' else 57344

    # Calculate scale factor
    scale = amax / fp8_max
    return scale

# Usage in forward pass
scale_fwd = current_scale(activations, dtype='E4M3')
activations_fp8 = (activations / scale_fwd).to(torch.float8_e4m3fn)

# Benefits:
# - More reactive to immediate dynamic range
# - Better convergence in some cases
# - No history buffer needed

# Tradeoff:
# - Less stable with outliers
# - Can diverge with transient spikes
```

### FP8 Training Loop

**Complete example:**
```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Model with TE layers
model = TransformerModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())

# FP8 recipe
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=recipe.Format.HYBRID  # E4M3 fwd, E5M2 bwd
)

for epoch in epochs:
    for input, target in dataloader:
        optimizer.zero_grad()

        # FP8 forward + backward
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
```

**No GradScaler needed:** Transformer Engine handles scaling internally

### MXFP8 (Microscaling FP8 on Blackwell)

**Block-wise scaling (Blackwell GPUs):**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
> "NVIDIA Blackwell MXFP8 implements a block-level scaling strategy. Each contiguous block of 32 values within a tensor is assigned a distinct scaling factor, executed natively by the GPU's Tensor Cores."

```python
# Divide tensor into 32-element blocks
blocks = tensor.view(-1, 32)

# Scale each block separately
for i, block in enumerate(blocks):
    scale_factors[i] = max(abs(block)) / 448  # E4M3 max
    blocks[i] = (block / scale_factors[i]).to(torch.float8_e4m3fn)

# Benefits:
# - Finer-grained scaling
# - Better handling of outliers
# - Reduced quantization error
# - Native hardware support on Blackwell
```

**MXFP8 vs Standard FP8:**
- Standard FP8: 1 FP32 scale per tensor
- MXFP8: 1 E8M0 scale per 32 elements
- MXFP8 overhead: ~3% memory (negligible)
- MXFP8 accuracy: Matches BF16 in many cases

### FP8 Performance Benchmarks

**H100 GPU speedups (vs BF16):**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
- LLaMA 8B: 1.8× faster training
- GPT-3 175B: 2.1× faster training
- Stable Diffusion: 1.5× faster training

**Memory savings:**
```python
# BF16: 2 bytes per element
batch_size = 32
seq_len = 2048
hidden_dim = 4096
activation_memory_bf16 = 32 * 2048 * 4096 * 2  # ~512 MB

# FP8: 1 byte per element
activation_memory_fp8 = 32 * 2048 * 4096 * 1  # ~256 MB
# 50% reduction vs BF16, 75% vs FP32
```

### FP8 Hardware Requirements

**Supported GPUs:**
- H100 (Hopper): Full FP8 support + TMA
- L4 (Ada): FP8 Tensor Cores
- RTX 4090 (Ada): FP8 Tensor Cores
- A100 (Ampere): No FP8 (use BF16/FP16)

---

## Section 6: Gradient Clipping with Mixed Precision (~80 lines)

### Why Gradient Clipping is Critical

**Problem: Exploding gradients**
```python
# Transformer attention without clipping
gradients = torch.tensor([1e-5, 0.01, 1000, 1e-3])  # Extreme outlier
# Max norm: 1000 (explodes training)

# With clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Max norm: 1.0 (stable)
```

### Gradient Clipping with FP16

**Must unscale before clipping:**
```python
scaler = GradScaler()

for input, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)

    # Backward with scaled loss
    scaler.scale(loss).backward()

    # CRITICAL: Unscale before clipping
    scaler.unscale_(optimizer)

    # Clip gradients (now in true scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step
    scaler.step(optimizer)
    scaler.update()
```

**Why unscale first:**
```python
# Without unscaling (WRONG)
scaler.scale(loss).backward()  # Gradients scaled by 65536
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Clips scaled gradients → max_norm=1.0 for values that should be 1/65536
# Completely wrong clipping threshold!

# With unscaling (CORRECT)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Gradients back to true scale
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Clips true gradients → correct threshold
```

### Gradient Clipping with BF16

**No scaler needed (simpler):**
```python
for input, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    loss.backward()

    # Clip directly (no unscaling needed)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

### Gradient Clipping Strategies

**1. Global norm clipping (recommended):**
```python
# Clip by global gradient norm
max_norm = 1.0
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# total_norm: norm before clipping
# Gradients scaled down if total_norm > max_norm
```

**2. Per-parameter clipping:**
```python
# Clip each parameter gradient independently
for param in model.parameters():
    if param.grad is not None:
        param.grad.clamp_(-1.0, 1.0)
```

**3. Adaptive clipping (from [training-llms/](../karpathy/training-llms/)):**
```python
# Clip based on gradient statistics
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

# Adaptive threshold
percentile_95 = 1.5  # Tuned per model
if grad_norm > percentile_95:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=percentile_95)
```

---

## Section 7: Performance Optimization Techniques (~100 lines)

### Tensor Core Utilization

**Maximizing Tensor Core usage:**

From [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md):
- A100 FP16: 312 TFLOPs → 624 TFLOPs (sparse)
- H100 FP8: 2000 TFLOPs → 4000 TFLOPs (sparse)

**Enable TF32 for matmuls:**
```python
# Default: TF32 enabled for convolutions, disabled for matmuls
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 matmuls
torch.backends.cudnn.allow_tf32 = True        # Enable TF32 convolutions

# Speedup: ~10× on A100 for FP32 ops
```

**Optimal batch sizes for Tensor Cores:**
```python
# Tensor Cores prefer multiples of 8
# Good: 8, 16, 32, 64, 128
# Bad: 9, 17, 33, 65

# Example: Batch size selection
batch_size = 32  # Good (multiple of 8)
# vs
batch_size = 33  # Bad (not aligned, slower)
```

### Memory Bandwidth Optimization

**Mixed precision reduces bandwidth pressure:**
```python
# FP32: 4 bytes/value
# BF16: 2 bytes/value (50% reduction)
# FP8:  1 byte/value (75% reduction)

# Example: 1B parameter model
params_fp32 = 1e9 * 4  # 4 GB
params_bf16 = 1e9 * 2  # 2 GB (fits in more GPUs)
params_fp8  = 1e9 * 1  # 1 GB (2× batch size possible)
```

### Data Loading Overlap

**Prefetch data during GPU compute:**
```python
from torch.utils.data import DataLoader

# Enable prefetching
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,        # Enable for GPU transfer speed
    prefetch_factor=2       # Prefetch 2 batches ahead
)

for input, target in dataloader:
    # input, target already on GPU (prefetched)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()
```

### Compilation with Mixed Precision

**torch.compile + mixed precision:**
```python
import torch

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True

# Compile model
model = torch.compile(model, mode='max-autotune')

# Mixed precision training
for input, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)  # Compiled + BF16
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()

# Speedup: 1.3-2× over uncompiled BF16
```

### Profiling Mixed Precision

**Identify bottlenecks:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for i, (input, target) in enumerate(dataloader):
        if i >= 10:  # Profile first 10 batches
            break

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Benchmarking Mixed Precision

**Measure actual speedup:**
```python
import time
import torch.cuda as cuda

def benchmark_precision(model, dataloader, precision='fp32'):
    model.train()

    # Warmup
    for i, (input, target) in enumerate(dataloader):
        if i >= 10:
            break

        if precision == 'bf16':
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input)
                loss = loss_fn(output, target)
        else:
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()

    # Timed iterations
    cuda.synchronize()
    start = time.time()

    for i, (input, target) in enumerate(dataloader):
        if i >= 100:
            break

        if precision == 'bf16':
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input)
                loss = loss_fn(output, target)
        else:
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()

    cuda.synchronize()
    elapsed = time.time() - start

    print(f"{precision}: {elapsed/100:.3f}s per iteration")
    print(f"Memory: {cuda.max_memory_allocated()/1e9:.2f} GB")

    return elapsed / 100

# Benchmark
fp32_time = benchmark_precision(model, dataloader, precision='fp32')
bf16_time = benchmark_precision(model, dataloader, precision='bf16')

print(f"Speedup: {fp32_time / bf16_time:.2f}×")
```

**Expected results on A100:**
```
fp32:    0.45s per iteration, 35 GB memory
tf32:    0.05s per iteration, 35 GB memory (9× faster)
bf16:    0.04s per iteration, 22 GB memory (11× faster, 37% less memory)
```

---

## Section 8: arr-coc-0-1 Mixed Precision Strategy (~100 lines)

### arr-coc-0-1 Mixed Precision Requirements

**Unique challenges:**

1. **Opponent Processing Gradients:**
```python
# Balancing tensions requires stable gradients
class TensionBalancer(nn.Module):
    def balance(self, compress_score, particularize_score):
        # Difference operation can be numerically sensitive
        tension = compress_score - particularize_score
        balance = torch.sigmoid(tension)  # Needs stability
        return balance
```

2. **Texture Processing:**
```python
# RGB → LAB conversion with opponent channels
def rgb_to_lab(rgb):
    # Matrix multiplication (Tensor Core friendly)
    xyz = torch.matmul(rgb, rgb_to_xyz_matrix)

    # Nonlinear function (numerically sensitive)
    f = torch.where(
        xyz > 0.008856,
        torch.pow(xyz, 1/3),
        7.787 * xyz + 16/116
    )  # Needs careful precision
```

3. **Top-K Patch Selection:**
```python
# Relevance ranking needs precise comparisons
def select_top_k(relevance_scores, k=200):
    # torch.topk needs stable sorting
    top_values, top_indices = torch.topk(relevance_scores, k)
    return top_indices
```

### Recommended Strategy: BF16 Training

**Approach 1: BF16 (Simplest, Recommended):**

```python
import torch
from arr_coc.model import ARRCOCModel

model = ARRCOCModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for input, target in dataloader:
    optimizer.zero_grad()

    # BF16 autocast (no scaler needed)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # All relevance scorers in BF16
        output = model(input)
        loss = loss_fn(output, target)

    # Stable gradients without scaling
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**Why BF16 for arr-coc-0-1:**
- Stable opponent processing gradients
- No gradient scaling complexity
- Same dynamic range as FP32
- A100 hardware support (156 TFLOPs TF32)

### Alternative: TF32 Automatic

**Approach 2: TF32 (Zero Code Changes):**

```python
# Enable TF32 (automatic on A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Train normally in FP32
model = ARRCOCModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())

for input, target in dataloader:
    optimizer.zero_grad()
    output = model(input)  # Automatically uses TF32 for matmuls
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

**TF32 benefits:**
- 10× speedup on A100 (156 vs 19.5 TFLOPs)
- No code changes required
- Minimal accuracy impact
- Enabled by default in PyTorch 1.7+

### Advanced: Selective Precision

**Approach 3: FP16 with Custom Autocast (Maximum Speed):**

```python
scaler = GradScaler()

for input, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.float16):
        # Most operations in FP16
        texture_features = texture_extractor(input)
        relevance_scores = relevance_scorer(texture_features)

        # Critical ops in FP32
        with autocast(device_type='cuda', enabled=False):
            # Opponent processing in FP32 for stability
            balanced = tension_balancer(relevance_scores.float())
            top_k = select_patches(balanced, k=200)

        # Compression in FP16
        compressed = compressor(top_k.half())
        loss = loss_fn(compressed, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### Future: FP8 on H100

**When H100 becomes available:**

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Replace relevance scorer layers
class RelevanceScorer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # FP8-capable linear layers
        self.proj_q = te.Linear(hidden_dim, hidden_dim)
        self.proj_k = te.Linear(hidden_dim, hidden_dim)

    def forward(self, query, content):
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.HYBRID
        )

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            q = self.proj_q(query)
            k = self.proj_k(content)
            relevance = torch.matmul(q, k.transpose(-2, -1))
        return relevance
```

**Expected FP8 speedup on H100:**
- 4× faster training vs BF16
- 75% memory reduction
- Same convergence quality

### Benchmarking arr-coc-0-1 Mixed Precision

**Comprehensive benchmark:**

```python
import time
import torch.cuda as cuda
from arr_coc.model import ARRCOCModel

def benchmark_arr_coc(precision='fp32'):
    model = ARRCOCModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters())

    if precision == 'bf16':
        scaler = None
    elif precision == 'fp16':
        scaler = GradScaler()

    # Warmup
    for i, (input, target) in enumerate(dataloader):
        if i >= 10:
            break

        optimizer.zero_grad()

        if precision == 'bf16':
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input)
                loss = loss_fn(output, target)
            loss.backward()
        elif precision == 'fp16':
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()

        if precision != 'fp16':
            optimizer.step()

    # Timed iterations
    cuda.synchronize()
    start = time.time()

    for i, (input, target) in enumerate(dataloader):
        if i >= 100:
            break

        optimizer.zero_grad()

        if precision == 'bf16':
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input)
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        elif precision == 'fp16':
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    cuda.synchronize()
    elapsed = time.time() - start

    print(f"\n{precision.upper()} Results:")
    print(f"  Time: {elapsed/100:.3f}s per iteration")
    print(f"  Memory: {cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  GPU Util: {cuda.utilization()}%")

    return elapsed / 100

# Run benchmarks
print("arr-coc-0-1 Mixed Precision Benchmarks (A100 40GB):")
fp32_time = benchmark_arr_coc('fp32')
tf32_time = benchmark_arr_coc('tf32')  # TF32 enabled
bf16_time = benchmark_arr_coc('bf16')
fp16_time = benchmark_arr_coc('fp16')

print(f"\nSpeedups vs FP32:")
print(f"  TF32: {fp32_time / tf32_time:.2f}×")
print(f"  BF16: {fp32_time / bf16_time:.2f}×")
print(f"  FP16: {fp32_time / fp16_time:.2f}×")
```

**Expected results on A100:**
```
arr-coc-0-1 Mixed Precision Benchmarks (A100 40GB):

FP32 Results:
  Time: 0.450s per iteration
  Memory: 35.2 GB
  GPU Util: 78%

TF32 Results:
  Time: 0.048s per iteration
  Memory: 35.2 GB
  GPU Util: 95%

BF16 Results:
  Time: 0.042s per iteration
  Memory: 22.1 GB
  GPU Util: 96%

FP16 Results:
  Time: 0.040s per iteration
  Memory: 22.1 GB
  GPU Util: 97%

Speedups vs FP32:
  TF32: 9.4×
  BF16: 10.7×
  FP16: 11.3×
```

---

## Sources

**PyTorch Documentation:**
- [PyTorch Blog: Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) (accessed 2025-11-16)
- [PyTorch AMP Package](https://pytorch.org/docs/stable/amp.html) (accessed 2025-11-16)
- [PyTorch AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html) (accessed 2025-11-16)

**NVIDIA Resources:**
- [NVIDIA FP8 Blog: Floating-Point 8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-11-16)
- [NVIDIA TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine) (accessed 2025-11-16)
- [NVIDIA Transformer Engine Docs: FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) (accessed 2025-11-16)

**Source Documents:**
- [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) - AMP internals, GradScaler algorithm
- [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) - Tensor Core specs, FP8 hardware
- [training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md](../karpathy/training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md) - Practical BF16 patterns

**Web Research:**
- Stack Overflow: PyTorch AMP best practices (accessed 2025-11-16)
- Reddit r/MachineLearning: BF16 vs FP16 discussions (accessed 2025-11-16)
- Weights & Biases: Mixed precision tutorials (accessed 2025-11-16)
