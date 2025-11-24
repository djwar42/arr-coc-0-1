# Mixed Precision Training Internals: torch.cuda.amp, GradScaler, and FP8

## Overview

Mixed precision training combines multiple numerical precisions during neural network training to accelerate computation while maintaining model accuracy. This document covers the internals of PyTorch's Automatic Mixed Precision (AMP) system, the GradScaler algorithm for gradient stability, precision format comparisons (FP16/BF16/TF32/FP8), and FP8 training with Transformer Engine.

**Why Mixed Precision Matters:**
- **Speed**: 2-3× training speedup with FP16/BF16, up to 4× with FP8
- **Memory**: 50% memory reduction (FP16/BF16 vs FP32), 75% with FP8
- **Hardware**: Utilizes Tensor Cores on NVIDIA GPUs (Volta+, Ampere+, Hopper+)
- **Cost**: Reduces training costs by 30-70% through faster iteration

From [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html) (accessed 2025-02-03):
> "torch.amp provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and other operations use lower precision floating point datatype (lower_precision_fp): such as torch.float16 (half) or torch.bfloat16."

**Related Knowledge:**
- See [cuda/01-memory-management-unified.md](01-memory-management-unified.md) for memory bandwidth implications
- See [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) for Tensor Core specifications

---

## Section 1: Mixed Precision Overview (100 lines)

### What is Mixed Precision Training?

Mixed precision training strategically employs lower precision formats (FP16, BF16, FP8) for computationally intensive operations while retaining FP32 precision where numerical stability is critical.

**Core Principle:**
```python
# Traditional FP32 training
model = Model().cuda()  # FP32 parameters
output = model(input)   # FP32 computation
loss = loss_fn(output, target)  # FP32 loss
loss.backward()  # FP32 gradients

# Mixed precision training
model = Model().cuda()  # FP32 parameters (master copy)
with torch.cuda.amp.autocast():
    output = model(input)  # FP16 computation
    loss = loss_fn(output, target)  # FP16 loss
scaler.scale(loss).backward()  # Scaled FP16 gradients
scaler.step(optimizer)  # Update FP32 master weights
```

### Two-Component System

**1. torch.autocast (Automatic Casting):**
- Context manager that automatically casts operations to lower precision
- Applies operation-specific policies (matmul→FP16, softmax→FP32)
- Maintains computation graph compatibility

**2. GradScaler (Gradient Scaling):**
- Prevents gradient underflow in FP16
- Scales loss before backward pass
- Detects gradient overflow/NaN
- Dynamically adjusts scale factor

### Historical Context

**Before Mixed Precision (2017):**
- All training in FP32
- A100 GPU: 19.5 TFLOPs FP32

**With FP16/BF16 (2018+):**
- 2-3× speedup
- A100 GPU: 312 TFLOPs FP16, 156 TFLOPs TF32

**With FP8 (2022+):**
- 4× speedup potential
- H100 GPU: 3,958 TFLOPs FP8

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-02-03):
> "With the growth of large language models (LLMs), deep learning is advancing both model architecture design and computational efficiency. Mixed precision training, which strategically employs lower precision formats like brain floating point 16 (BF16) for computationally intensive operations while retaining the stability of 32-bit floating-point (FP32) where needed, has been a key strategy for accelerating training."

### When Mixed Precision Helps

**Ideal Use Cases:**
- Large transformer models (BERT, GPT, LLaMA)
- Computer vision (ResNet, ViT, ConvNeXt)
- Matrix-heavy operations (attention, FFN)
- Batch sizes that fit in memory with FP32

**Challenging Use Cases:**
- Small models (overhead > speedup)
- Numerically unstable operations
- Models with extreme dynamic range
- Operations without Tensor Core support

---

## Section 2: torch.cuda.amp.autocast Internals (200 lines)

### Autocast Context Manager

The `autocast` context manager wraps forward pass regions to enable automatic precision casting.

**Basic Usage:**
```python
# CUDA autocast
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

# Device-agnostic (PyTorch 2.0+)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
```

**Implementation Details:**
- Uses thread-local state to track autocast mode
- Propagates through function calls automatically
- Maintains separate state per device type (cuda, cpu, xpu)

### Operation-Specific Casting Rules

PyTorch maintains a whitelist/blacklist of operations for autocast:

**FP16-Safe Operations (Compute-Bound):**
```python
# These operations run in FP16
torch.matmul(a, b)           # Matrix multiplication
torch.nn.functional.linear   # Linear layers
torch.bmm(a, b)             # Batch matrix multiply
torch.addmm(bias, a, b)     # Matrix multiply-add
torch.baddbmm(bias, a, b)   # Batch matrix multiply-add
torch.conv2d(input, weight) # Convolutions
```

**FP32-Required Operations (Numerically Sensitive):**
```python
# These operations run in FP32
torch.nn.functional.softmax(x, dim=-1)
torch.nn.functional.log_softmax(x, dim=-1)
torch.nn.functional.layer_norm(x, normalized_shape)
torch.nn.functional.batch_norm(x, ...)
torch.sum(x)
torch.prod(x)
```

**Type Promotion Operations:**
```python
# These promote to widest input type
torch.add(fp16_tensor, fp32_tensor)  # Result: FP32
torch.mul(fp16_tensor, fp32_tensor)  # Result: FP32
torch.cat([fp16_tensor, fp32_tensor])  # Result: FP32
```

From [PyTorch AMP Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) (accessed 2025-02-03):
> "Autocast automatically chooses the precision for operations to improve performance while maintaining accuracy."

### Casting Algorithm

**Step-by-step execution:**

1. **Enter autocast context:**
   ```python
   # Thread-local state updated
   thread_local.autocast_enabled = True
   thread_local.autocast_dtype = torch.float16
   thread_local.autocast_device = 'cuda'
   ```

2. **Operation dispatch:**
   ```python
   # For each operation, check policy
   if op in autocast_fp16_ops:
       # Cast inputs to FP16
       fp16_inputs = [x.half() if x.dtype == torch.float32 else x
                      for x in inputs]
       result = op(*fp16_inputs)
   elif op in autocast_fp32_ops:
       # Cast inputs to FP32
       fp32_inputs = [x.float() if x.dtype == torch.float16 else x
                      for x in inputs]
       result = op(*fp32_inputs)
   else:
       # Use input dtype
       result = op(*inputs)
   ```

3. **Exit autocast context:**
   ```python
   thread_local.autocast_enabled = False
   # Restore previous state
   ```

### Nested Autocast Contexts

Autocast contexts can be nested and disabled:

```python
# Outer autocast enabled
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # FP16 operations
    x = torch.matmul(a, b)

    # Disable autocast for specific region
    with torch.autocast(device_type='cuda', enabled=False):
        # FP32 operations
        y = torch.matmul(c, d)

    # FP16 operations resume
    z = torch.matmul(x, y)
```

### Autocast and Backward Pass

**Critical Rule: Do NOT use autocast in backward pass**

From [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html):
> "Backward passes under autocast are not recommended. Backward ops run in the same dtype autocast chose for corresponding forward ops."

```python
# CORRECT
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
# Backward outside autocast
scaler.scale(loss).backward()

# INCORRECT
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()  # BAD: backward under autocast
```

**Reason:**
- Backward ops automatically use same dtype as forward
- Autocast in backward can cause dtype mismatches
- GradScaler handles backward precision correctly

### Memory Implications

**Activation Memory:**
```python
# FP32: 4 bytes per element
batch_size = 32
seq_len = 512
hidden_dim = 768
activation_memory_fp32 = batch_size * seq_len * hidden_dim * 4  # ~48 MB

# FP16: 2 bytes per element
activation_memory_fp16 = batch_size * seq_len * hidden_dim * 2  # ~24 MB
# 50% memory reduction
```

**Parameter Storage:**
```python
# Master weights stay in FP32
model_params_fp32 = sum(p.numel() for p in model.parameters()) * 4

# Additional FP16 copy (if using GradScaler)
model_params_fp16 = sum(p.numel() for p in model.parameters()) * 2

# Total: 1.5× FP32 memory (not 0.5×!)
# But activations are 0.5×, so net reduction ~30-40%
```

---

## Section 3: GradScaler Algorithm Deep Dive (200 lines)

### The Gradient Underflow Problem

FP16 has limited dynamic range: smallest positive normal value is ~6e-5.

**Problem:**
```python
# FP32 gradient
grad_fp32 = 1e-7  # Representable

# FP16 gradient
grad_fp16 = torch.tensor(1e-7, dtype=torch.float16)  # Underflows to 0!
print(grad_fp16)  # tensor(0.)
```

**Consequence:**
- Small gradients flush to zero
- Model fails to converge
- Gradient information is lost

### Loss Scaling Solution

**Core Idea:** Scale loss by large factor before backward, then unscale gradients before optimizer step.

```python
# Without scaling
loss = 1e-5  # Small loss
loss.backward()  # Gradients underflow to zero

# With scaling
scale = 65536  # 2^16
scaled_loss = loss * scale  # 0.65536
scaled_loss.backward()  # Gradients preserved
unscaled_grads = grads / scale  # Restore original magnitudes
```

### GradScaler Implementation

**Initialization:**
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler(
    init_scale=2.**16,    # Initial scale factor (65536)
    growth_factor=2.0,    # Multiply scale on success
    backoff_factor=0.5,   # Divide scale on overflow
    growth_interval=2000  # Iterations between growth attempts
)
```

**Internal State:**
```python
class GradScaler:
    def __init__(self, ...):
        self._scale = init_scale  # Current scale factor (FP32)
        self._growth_tracker = 0  # Iterations since last overflow
        self._per_optimizer_states = {}  # Track unscale per optimizer
```

### GradScaler Training Loop

**Complete workflow:**

```python
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # 1. Forward pass with autocast
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # 2. Scale loss and backward
        scaler.scale(loss).backward()

        # 3. Unscale gradients (optional, for clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. Optimizer step with overflow check
        scaler.step(optimizer)

        # 5. Update scale for next iteration
        scaler.update()
```

### GradScaler.scale() Method

**Purpose:** Multiply loss by scale factor

```python
def scale(self, outputs):
    """
    Multiply outputs by the current scale factor.

    Args:
        outputs: Tensor or container of tensors (loss)

    Returns:
        Scaled tensor(s)
    """
    if isinstance(outputs, torch.Tensor):
        # Simple case: single tensor
        return outputs * self._scale
    else:
        # Container case: scale each tensor
        return type(outputs)(self.scale(v) for v in outputs)
```

**Example:**
```python
loss = torch.tensor(1e-5, dtype=torch.float16)
scaled_loss = scaler.scale(loss)  # 0.65536 (65536 * 1e-5)
scaled_loss.backward()  # Backward on scaled loss
```

### GradScaler.unscale_() Method

**Purpose:** Divide gradients by scale factor

```python
def unscale_(self, optimizer):
    """
    Unscale gradients of optimizer's assigned parameters in-place.

    Args:
        optimizer: Torch optimizer

    Side effects:
        - Divides optimizer's gradients by scale
        - Checks for inf/nan in gradients
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

                # Check for inf/nan
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    found_inf = True

    # Mark as unscaled
    self._per_optimizer_states[id(optimizer)] = found_inf
```

**When to call unscale_():**

```python
# Case 1: Gradient clipping (must unscale first)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)  # Step knows grads already unscaled

# Case 2: Inspecting gradients
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
print(model.fc.weight.grad.abs().max())  # True unscaled gradient magnitude

# Case 3: No need to call (step does it internally)
scaler.scale(loss).backward()
scaler.step(optimizer)  # Internally calls unscale_()
```

### GradScaler.step() Method

**Purpose:** Conditionally update optimizer parameters

```python
def step(self, optimizer, *args, **kwargs):
    """
    Step optimizer if gradients are finite, skip if inf/nan.

    Args:
        optimizer: Torch optimizer
        *args, **kwargs: Passed to optimizer.step()

    Returns:
        Optional return value from optimizer.step()
    """
    # Unscale if not already done
    if id(optimizer) not in self._per_optimizer_states:
        self.unscale_(optimizer)

    # Check if inf/nan was found
    found_inf = self._per_optimizer_states[id(optimizer)]

    if found_inf:
        # Skip optimizer step
        return None
    else:
        # Step optimizer normally
        return optimizer.step(*args, **kwargs)
```

**Step skipping behavior:**
```python
# Iteration with valid gradients
scaler.step(optimizer)  # Calls optimizer.step()

# Iteration with inf/nan gradients
scaler.step(optimizer)  # Skips optimizer.step()
# Model parameters unchanged this iteration
```

### GradScaler.update() Method

**Purpose:** Adjust scale factor based on overflow detection

```python
def update(self, new_scale=None):
    """
    Update scale factor for next iteration.

    Args:
        new_scale: Optional manual scale override

    Algorithm:
        - If overflow occurred: decrease scale (backoff)
        - If no overflow for growth_interval: increase scale (growth)
    """
    if new_scale is not None:
        # Manual scale override
        self._scale = new_scale
        self._growth_tracker = 0
        return

    # Check if any optimizer had inf/nan
    found_inf = any(self._per_optimizer_states.values())

    # Clear per-optimizer states
    self._per_optimizer_states.clear()

    if found_inf:
        # Overflow detected: decrease scale
        self._scale *= self.backoff_factor  # Divide by 2
        self._growth_tracker = 0
    else:
        # No overflow: increment tracker
        self._growth_tracker += 1

        if self._growth_tracker == self.growth_interval:
            # Increase scale after growth_interval successful iterations
            self._scale *= self.growth_factor  # Multiply by 2
            self._growth_tracker = 0
```

**Scale dynamics example:**
```python
# Initial scale
scale = 65536  # 2^16

# 2000 iterations without overflow
for i in range(2000):
    scaler.update()  # No overflow
# scale = 131072  # 2^17 (doubled)

# Overflow on next iteration
# (gradient magnitude too large)
scaler.update()  # Overflow detected
# scale = 65536  # 2^16 (halved back)
```

### Gradient Accumulation with GradScaler

**Problem:** Scale must remain constant across accumulated gradients

```python
# CORRECT: Update scale only at effective batch boundaries
scaler = GradScaler()
accumulation_steps = 4

for i, (input, target) in enumerate(data):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target) / accumulation_steps

    # Accumulate scaled gradients
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        # Unscale and step at effective batch boundary
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()  # Update scale AFTER effective batch
        optimizer.zero_grad()
```

---

## Section 4: Precision Formats Comparison (150 lines)

### Floating-Point Representation Basics

**IEEE 754 Format:**
```
sign | exponent | mantissa
 1 bit | E bits | M bits
```

**Value calculation:**
```
value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa/2^M)
```

### FP32 (32-bit Float)

**Bit layout:**
```
sign (1) | exponent (8) | mantissa (23)
```

**Properties:**
- Range: ±1.4e-45 to ±3.4e38
- Precision: ~7 decimal digits
- Exponent bias: 127
- Smallest positive normal: 1.175e-38

**Memory:**
- 4 bytes per value
- A100 40GB: ~10 billion FP32 values

### FP16 (16-bit Half Precision)

**Bit layout:**
```
sign (1) | exponent (5) | mantissa (10)
```

**Properties:**
- Range: ±6.1e-5 to ±65,504
- Precision: ~3 decimal digits
- Exponent bias: 15
- Smallest positive normal: 6.1e-5

**Limitations:**
```python
# Overflow
x = torch.tensor(70000.0, dtype=torch.float16)  # inf

# Underflow
y = torch.tensor(1e-5, dtype=torch.float16)  # 0.0

# Precision loss
z = torch.tensor(2048.5, dtype=torch.float16)  # 2048.0 (no .5)
```

**When to use:**
- Forward pass (with autocast)
- Gradients (with scaling)
- Inference
- CUDA devices with Tensor Cores

From [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md):
- A100: 312 TFLOPs FP16 (vs 19.5 TFLOPs FP32)
- H100: 1,979 TFLOPs FP16

### BF16 (Brain Float 16)

**Bit layout:**
```
sign (1) | exponent (8) | mantissa (7)
```

**Properties:**
- Range: ±1.4e-45 to ±3.4e38 (same as FP32)
- Precision: ~2 decimal digits
- Exponent bias: 127
- Smallest positive normal: 1.175e-38

**Key advantage:** Same dynamic range as FP32

```python
# BF16 vs FP16 range comparison
fp32_val = 1e20
bf16_val = torch.tensor(fp32_val, dtype=torch.bfloat16)  # OK
fp16_val = torch.tensor(fp32_val, dtype=torch.float16)   # inf!
```

**When to use BF16 over FP16:**
- Models with extreme dynamic range
- No gradient scaling needed (stable training)
- TPU training (Google TPUs prefer BF16)
- Ampere+ GPUs (hardware support)

**BF16 Training without GradScaler:**
```python
# BF16 doesn't need gradient scaling
model = Model().cuda()

for input, target in data:
    optimizer.zero_grad()

    # Autocast to BF16
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    # No scaler needed!
    loss.backward()
    optimizer.step()
```

### TF32 (Tensor Float 32)

**Bit layout:**
```
sign (1) | exponent (8) | mantissa (10)
```

**Properties:**
- Hybrid format: FP32 range, FP16 mantissa
- Automatic on Ampere+ GPUs
- Transparent to user code
- Used internally by Tensor Cores

**Automatic activation:**
```python
# Enabled by default in PyTorch 1.7+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Performance:**
```python
# A100 GPU
# FP32 matmul: 19.5 TFLOPs
# TF32 matmul: 156 TFLOPs (8× faster!)
# Accuracy loss: minimal (~0.1% for most models)
```

**When TF32 is used:**
- Matrix multiplications on Ampere+ GPUs
- No code changes required
- Transparent precision reduction
- Not user-controllable per-operation

From [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md):
> "TF32 Precision (Tensor Float 32): Automatic on A100/H100 for FP32 operations. 8-bit exponent (like FP32), 10-bit mantissa (reduced from 23). Enables 10x speedup with minimal accuracy loss. Enabled by default in PyTorch 1.7+."

### FP8 (8-bit Float)

FP8 has two variants designed for different training stages:

**E4M3 (4 exponent, 3 mantissa bits):**
```
sign (1) | exponent (4) | mantissa (3)
```
- Range: ±448
- Precision: ~1 decimal digit
- Use: Forward pass (weights, activations)

**E5M2 (5 exponent, 2 mantissa bits):**
```
sign (1) | exponent (5) | mantissa (2)
```
- Range: ±57,344
- Precision: <1 decimal digit
- Use: Backward pass (gradients)

**Why two formats?**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
> "E4M3, a format with 4 exponent 3 mantissa bits, prioritizes precision for forward passes, where weights and activations benefit from finer-grained values. E5M2, short for 5 exponent bits and 2 mantissa bits, trades mantissa bits for a wider dynamic range. This broader range is crucial for backward passes, where gradients can vary significantly in magnitude."

**FP8 vs INT8:**
- FP8: Floating-point with dynamic exponents
- INT8: Fixed-point with static scaling
- FP8 handles extreme ranges better (attention, gradients)
- INT8 struggles with transformer architectures

### Precision Format Comparison Table

| Format | Bits | Exponent | Mantissa | Range | Use Case | Hardware |
|--------|------|----------|----------|-------|----------|----------|
| **FP32** | 32 | 8 | 23 | ±3.4e38 | Baseline | All |
| **TF32** | 32* | 8 | 10 | ±3.4e38 | Automatic | Ampere+ |
| **BF16** | 16 | 8 | 7 | ±3.4e38 | Stable training | Ampere+, TPU |
| **FP16** | 16 | 5 | 10 | ±65,504 | Fast training | Volta+ |
| **FP8 E4M3** | 8 | 4 | 3 | ±448 | Forward pass | Hopper+, Ada+ |
| **FP8 E5M2** | 8 | 5 | 2 | ±57,344 | Backward pass | Hopper+, Ada+ |

*TF32 stored as FP32 but computed with 10-bit mantissa

---

## Section 5: FP8 Training with Transformer Engine (200 lines)

### NVIDIA Transformer Engine Overview

Transformer Engine is a library for accelerating transformer models using FP8 on Hopper (H100) and Ada (L4, RTX 40XX) GPUs.

**Key features:**
- Automatic FP8 scaling (per-tensor or block-wise)
- Delayed scaling algorithm
- Tensor-wise and block-wise MXFP8
- Integration with PyTorch, JAX, PaddlePaddle

From [NVIDIA TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine) (accessed 2025-02-03):
> "Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper, Ada, and later GPUs."

### FP8 Scaling Strategies

**1. Tensor-wise Scaling (Standard FP8):**

Single FP32 scaling factor per tensor:
```python
# Forward pass
scale_fwd = max(abs(tensor)) / 448  # E4M3 max
tensor_fp8 = (tensor / scale_fwd).to(torch.float8_e4m3fn)

# Backward pass
scale_bwd = max(abs(grads)) / 57344  # E5M2 max
grads_fp8 = (grads / scale_bwd).to(torch.float8_e5m2)
```

**2. Block-wise Scaling (MXFP8 on Blackwell):**

Separate scaling factor per 32-element block:
```python
# Divide tensor into 32-element blocks
blocks = tensor.view(-1, 32)

# Scale each block separately
for i, block in enumerate(blocks):
    scale_factors[i] = max(abs(block)) / 448
    blocks[i] = (block / scale_factors[i]).to(torch.float8_e4m3fn)
```

**MXFP8 advantages:**
- Finer-grained scaling
- Better handling of outliers
- Reduced quantization error
- Native hardware support on Blackwell

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
> "The fundamental distinction between standard FP8 and MXFP8 lies in their granular scaling mechanism. NVIDIA Blackwell MXFP8 implements a block-level scaling strategy. Specifically, each contiguous block of 32 values within a tensor is assigned a distinct scaling factor, a process executed natively by the GPU's Tensor Cores."

### Delayed Scaling Algorithm

**Problem:** Current-iteration statistics can cause instability

**Solution:** Use historical amax values for scaling

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

    def get_scale(self):
        # Use max of history for stable scaling
        history_amax = max(self.amax_history)
        return history_amax / 448  # E4M3 max
```

**Benefits:**
- Stable training without divergence
- Handles transient spikes gracefully
- Recommended for LLM pretraining

### Per-Tensor Current Scaling

**Algorithm:** Calculate scale from current tensor statistics

```python
def current_scale(tensor, dtype='E4M3'):
    # Get max absolute value
    amax = tensor.abs().max()

    # Determine FP8 range
    if dtype == 'E4M3':
        fp8_max = 448
    else:  # E5M2
        fp8_max = 57344

    # Calculate scale factor
    scale = amax / fp8_max

    return scale

# Usage in forward pass
scale_fwd = current_scale(activations, dtype='E4M3')
activations_fp8 = (activations / scale_fwd).to(torch.float8_e4m3fn)
```

**Benefits over delayed scaling:**
- More reactive to immediate dynamic range
- Better convergence in some cases
- No history buffer needed

**Tradeoff:**
- Less stable with outliers
- Can diverge with transient spikes

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
> "Unlike delayed scaling, where transient spikes in training can contaminate the amax history and potentially lead to divergence, per-tensor current scaling dynamically adjusts the scale based on the present data range. This immediate responsiveness helps optimize the FP8 representation and has been observed to improve model convergence during training."

### Transformer Engine API

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

**FP8 autocast context:**
```python
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    # Inside this context:
    # - te.Linear layers use FP8
    # - Regular nn.Linear uses BF16/FP16
    # - Other ops follow autocast rules
    output = model(input)
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
    fp8_format=recipe.Format.HYBRID
)

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # FP8 forward + backward
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
```

**No GradScaler needed:** Transformer Engine handles scaling internally

### FP8 Memory Savings

**Activation memory:**
```python
# BF16: 2 bytes per element
batch_size = 32
seq_len = 2048
hidden_dim = 4096
activation_memory_bf16 = batch_size * seq_len * hidden_dim * 2  # ~512 MB

# FP8: 1 byte per element
activation_memory_fp8 = batch_size * seq_len * hidden_dim * 1  # ~256 MB
# 50% reduction vs BF16, 75% vs FP32
```

**Scaling factor overhead:**
```python
# Tensor-wise: 1 FP32 per tensor
num_tensors = 100
scaling_overhead = num_tensors * 4  # 400 bytes (negligible)

# MXFP8 block-wise: 1 E8M0 per 32 elements
num_elements = 32 * 2048 * 4096
num_blocks = num_elements // 32
scaling_overhead_mxfp8 = num_blocks * 1  # ~8 MB (3% overhead)
```

### FP8 Performance Benchmarks

**H100 GPU speedups (vs BF16):**

From [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/):
- LLaMA 8B: 1.8× faster training
- GPT-3 175B: 2.1× faster training
- Stable Diffusion: 1.5× faster training

**Convergence comparison:**
- MXFP8 matches BF16 perplexity
- No accuracy degradation observed
- Same final model quality

**Hardware requirements:**
- H100 (Hopper): Full FP8 support + TMA
- L4 (Ada): FP8 Tensor Cores
- RTX 4090 (Ada): FP8 Tensor Cores
- A100 (Ampere): No FP8 (use BF16/FP16)

---

## Section 6: Gradient Stability and Debugging (100 lines)

### Understanding Gradient Underflow

**FP16 representable range:**
```python
import torch

# Smallest positive normal FP16
min_fp16 = 2 ** -14  # 6.1e-5
print(f"Min FP16: {min_fp16}")

# Gradient smaller than this underflows
grad = torch.tensor(1e-6, dtype=torch.float16)
print(f"1e-6 in FP16: {grad}")  # 0.0
```

**When underflow happens:**
- Deep networks (many layers)
- Small learning rates
- Batch normalization (divides by std)
- Attention with many heads

### Loss Scaling Guidelines

**Choosing initial scale:**
```python
# Conservative (fewer overflows)
scaler = GradScaler(init_scale=2.**10)  # 1024

# Standard (balanced)
scaler = GradScaler(init_scale=2.**16)  # 65536 (default)

# Aggressive (fewer underflows)
scaler = GradScaler(init_scale=2.**20)  # 1048576
```

**Monitoring scale dynamics:**
```python
scaler = GradScaler()

for i, (input, target) in enumerate(data):
    # ... training step
    scaler.update()

    # Log scale every 100 iterations
    if i % 100 == 0:
        print(f"Iter {i}: scale = {scaler.get_scale()}")
```

**Healthy scale patterns:**
```python
# Good: Scale grows slowly, few overflows
Iter 0: scale = 65536.0
Iter 100: scale = 65536.0
Iter 200: scale = 131072.0  # Grew after 2000 iters
Iter 300: scale = 131072.0

# Bad: Scale keeps decreasing (persistent overflows)
Iter 0: scale = 65536.0
Iter 100: scale = 32768.0   # Halved
Iter 200: scale = 16384.0   # Halved again
Iter 300: scale = 8192.0    # Death spiral!
```

### Detecting NaN Gradients

**Manual gradient checking:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")
```

**GradScaler automatic detection:**
```python
# GradScaler detects inf/nan during unscale_()
scaler.scale(loss).backward()
scaler.step(optimizer)  # Skips if inf/nan found
scaler.update()  # Reduces scale if inf/nan found
```

### BF16 Advantage: No Gradient Scaling Needed

**Why BF16 is more stable:**
```python
# FP16: Range ±65504
fp16_grad = torch.tensor(1e-38, dtype=torch.float16)  # 0.0 (underflow)

# BF16: Range ±3.4e38 (same as FP32)
bf16_grad = torch.tensor(1e-38, dtype=torch.bfloat16)  # OK!
```

**BF16 training (no scaler):**
```python
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())

for input, target in data:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    # No scaler needed!
    loss.backward()
    optimizer.step()
```

### Debugging Mixed Precision Issues

**Issue 1: Model diverges with FP16**

Solution:
```python
# Try BF16 instead
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```

**Issue 2: Scale keeps decreasing**

Solutions:
```python
# 1. Increase init_scale
scaler = GradScaler(init_scale=2.**20)

# 2. Decrease learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower

# 3. Use gradient clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

**Issue 3: No speedup from mixed precision**

Checks:
```python
# 1. Verify Tensor Core usage
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Check GPU support
print(torch.cuda.get_device_capability())  # Should be (7,0)+ for FP16

# 3. Profile with NSight Systems
# Look for HMMA (Half Matrix Multiply-Accumulate) operations
```

---

## Section 7: ARR-COC Mixed Precision Optimization (150 lines)

### ARR-COC Training Requirements

ARR-COC relevance scorers have unique mixed precision needs:

**1. Opponent Processing Gradients:**
```python
# Balancing tensions requires stable gradients
class TensionBalancer(nn.Module):
    def balance(self, compress_score, particularize_score):
        # Difference operation can be numerically sensitive
        tension = compress_score - particularize_score
        balance = torch.sigmoid(tension)  # Needs FP32 for stability
        return balance
```

**Solution:** Use BF16 for stability
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    tension = balancer.balance(compress, particularize)
```

**2. Texture Processing:**
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

**Solution:** Keep conversion in FP32, textures in FP16
```python
with torch.autocast(device_type='cuda', enabled=False):
    lab = rgb_to_lab(rgb.float())  # FP32 conversion

# Use FP16 textures after conversion
with torch.autocast(device_type='cuda', dtype=torch.float16):
    texture_features = texture_extractor(lab.half())
```

**3. Top-K Patch Selection:**
```python
# Relevance ranking needs precise comparisons
def select_top_k(relevance_scores, k=200):
    # torch.topk needs stable sorting
    top_values, top_indices = torch.topk(relevance_scores, k)
    return top_indices
```

**Solution:** Use FP32 for ranking
```python
with torch.autocast(device_type='cuda', enabled=False):
    top_indices = select_top_k(relevance_scores.float(), k=200)
```

### Recommended Mixed Precision Strategy for ARR-COC

**Approach 1: BF16 Training (Simplest)**

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

**Why BF16 for ARR-COC:**
- Stable opponent processing gradients
- No gradient scaling complexity
- Same dynamic range as FP32
- A100 hardware support (156 TFLOPs TF32)

**Approach 2: TF32 Automatic (Zero Code Changes)**

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

**Approach 3: FP16 with Custom Autocast (Maximum Speed)**

```python
scaler = GradScaler()

for input, target in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Most operations in FP16
        texture_features = texture_extractor(input)
        relevance_scores = relevance_scorer(texture_features)

        # Critical ops in FP32
        with torch.autocast(device_type='cuda', enabled=False):
            # Opponent processing in FP32
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

### Future: FP8 Training on H100

**When available:**
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
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            q = self.proj_q(query)
            k = self.proj_k(content)
            relevance = torch.matmul(q, k.transpose(-2, -1))
        return relevance
```

**Expected speedup on H100:**
- 4× faster training vs BF16
- 75% memory reduction
- Same convergence quality

**Caveats:**
- H100 required (expensive)
- Transformer Engine integration needed
- More complex debugging

### Benchmarking ARR-COC Mixed Precision

**Metrics to track:**
```python
import time
import torch.cuda as cuda

# Benchmark training iteration
def benchmark_precision(model, dataloader, precision='fp32'):
    model.train()

    # Warmup
    for i, (input, target) in enumerate(dataloader):
        if i >= 10:
            break
        with torch.autocast(device_type='cuda', dtype=precision):
            output = model(input)
            loss = loss_fn(output, target)
        loss.backward()

    # Timed iterations
    cuda.synchronize()
    start = time.time()

    for i, (input, target) in enumerate(dataloader):
        if i >= 100:
            break
        with torch.autocast(device_type='cuda', dtype=precision):
            output = model(input)
            loss = loss_fn(output, target)
        loss.backward()

    cuda.synchronize()
    elapsed = time.time() - start

    print(f"{precision}: {elapsed/100:.3f}s per iteration")
    print(f"Memory: {cuda.max_memory_allocated()/1e9:.2f} GB")
```

**Expected results on A100:**
```
fp32:    0.45s per iteration, 35 GB memory
tf32:    0.05s per iteration, 35 GB memory (10× faster)
bf16:    0.04s per iteration, 22 GB memory (11× faster)
fp16:    0.04s per iteration, 22 GB memory (11× faster)
```

---

## Sources

**PyTorch Documentation:**
- [PyTorch AMP Package](https://docs.pytorch.org/docs/stable/amp.html) (accessed 2025-02-03)
- [PyTorch AMP Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) (accessed 2025-02-03)
- [PyTorch AMP Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) (accessed 2025-02-03)

**NVIDIA Resources:**
- [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) - NVIDIA Developer Blog (accessed 2025-02-03)
- [NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine) - GitHub repository (accessed 2025-02-03)
- [Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) - NVIDIA Docs

**Web Research:**
- Stack Overflow discussions on GradScaler usage (accessed 2025-02-03)
- PyTorch Forums on FP8 support and mixed precision (accessed 2025-02-03)

**Source Documents:**
- [cuda/01-memory-management-unified.md](01-memory-management-unified.md) - Mentions AMP
- [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) - GPU specs and mixed precision overview

**Additional References:**
- Reddit r/MachineLearning discussions on BF16 vs FP16 training
- Medium articles on GradScaler implementation
- Weights & Biases tutorials on autocast and mixed precision
