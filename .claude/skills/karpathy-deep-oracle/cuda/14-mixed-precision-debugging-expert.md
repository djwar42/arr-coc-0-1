# Mixed Precision Debugging & Stability: Ultra-Expert Guide

## Overview

This guide covers ultra-expert level mixed precision debugging for production deep learning training. Mixed precision training (FP16/BF16/FP8) accelerates training 2-4× but introduces numerical instability challenges that can stall training for days. This document focuses on **debugging NaN propagation, gradient underflow/overflow, GradScaler tuning, and production stability monitoring**.

**Why This Matters:**
- Large model training can fail **days** into a run due to precision issues
- NaN detection is reactive - by the time you see NaN loss, damage is done
- Understanding **early warning signs** saves compute time and money
- Different precision formats (FP16/BF16/FP8) have different failure modes

**Prerequisites:**
- Read [cuda/07-mixed-precision-training-internals.md](07-mixed-precision-training-internals.md) for AMP basics
- Understand [cuda/09-runtime-errors-debugging-expert.md](09-runtime-errors-debugging-expert.md) for general debugging

From [Medium: Solving the Limits of Mixed Precision Training](https://medium.com/@jbensnyder/solving-the-limits-of-mixed-precision-training-231019128b4b) (accessed 2025-11-13):
> "Often large training jobs can be stalled for days in order to deal with numeric instabilities. Understanding the model's internal state in the early training stages can inform if a model will be prone to instabilities later in training."

---

## Section 1: NaN Detection & Root Cause Analysis (~125 lines)

### The NaN Propagation Problem

NaNs propagate through computation graphs, making root cause identification difficult. By the time you see `loss=nan`, the actual source may be many layers upstream.

**Common NaN Sources:**
1. **Division by zero** - BatchNorm with zero variance, attention with zero denominators
2. **Log of negative/zero** - Cross-entropy with invalid logits, log-softmax edge cases
3. **Gradient overflow** - FP16 max is 65,504; gradients easily exceed this
4. **Invalid operations** - sqrt of negatives, arccos of values > 1.0

From [PyTorch GitHub Issue #40497](https://github.com/pytorch/pytorch/issues/40497) (accessed 2025-11-13):
> "I'm using autocast with GradScaler to train on mixed precision. For small dataset, it works fine. But when I trained on bigger dataset, after few epochs (3-4), the loss turns to nan."

### NaN Detection Hooks (Early Detection)

**Basic backward hook for NaN detection:**

```python
import torch

def register_nan_hooks(model):
    """Register hooks to detect NaN gradients during backward pass."""

    def check_grad_nan(module, grad_input, grad_output):
        # Check output gradients
        for i, grad in enumerate(grad_output):
            if grad is not None and torch.isnan(grad).any():
                print(f"NaN in output gradient {i} of {module.__class__.__name__}")
                print(f"  Grad shape: {grad.shape}")
                print(f"  NaN count: {torch.isnan(grad).sum().item()}")
                # Optionally save gradient for inspection
                torch.save(grad, f"nan_grad_{module.__class__.__name__}_{i}.pt")

        # Check input gradients
        for i, grad in enumerate(grad_input):
            if grad is not None and torch.isnan(grad).any():
                print(f"NaN in input gradient {i} of {module.__class__.__name__}")

    # Register hook on all modules
    for name, module in model.named_modules():
        module.register_full_backward_hook(check_grad_nan)
```

**Usage:**
```python
model = MyModel()
register_nan_hooks(model)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()  # Hooks fire here if NaN detected
    scaler.step(optimizer)
    scaler.update()
```

### Advanced: Layer-by-Layer NaN Isolation

**Binary search for NaN source:**

```python
def find_nan_layer(model, input_data, target):
    """Isolate which layer produces NaN via binary search."""

    # Get all modules
    modules = list(model.modules())

    # Forward pass, checking intermediate outputs
    x = input_data
    for i, module in enumerate(modules):
        x = module(x)

        if torch.isnan(x).any():
            print(f"NaN first appeared at layer {i}: {module.__class__.__name__}")
            print(f"  Input shape: {input_data.shape}")
            print(f"  Output shape: {x.shape}")
            print(f"  NaN locations: {torch.isnan(x).nonzero()}")

            # Check module parameters
            for name, param in module.named_parameters():
                if torch.isnan(param).any():
                    print(f"  Parameter '{name}' contains NaN")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"  Gradient '{name}' contains NaN")

            return module, i

    return None, -1
```

### GradScaler NaN Detection Internals

**How GradScaler detects overflow/NaN:**

```python
# Inside GradScaler.unscale_() (simplified)
def unscale_(self, optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                # Unscale gradient
                param.grad.div_(self._scale)

                # Check for inf/nan
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    self._found_inf_per_device[param.device] = 1.0
                    # Scale will be reduced on next update()
```

**What happens when NaN detected:**
1. `scaler.step(optimizer)` **skips** the weight update
2. `scaler.update()` reduces scale factor by `backoff_factor` (default 0.5)
3. Next iteration tries again with lower scale

**Problem:** This is **reactive**. By the time NaN appears, model may be in unstable state.

### Proactive NaN Prevention

**Pre-gradient clipping NaN check:**

```python
def safe_backward_step(scaler, loss, optimizer, model, max_grad_norm=1.0):
    """Safer backward pass with multiple safety checks."""

    # 1. Check loss is finite before backward
    if not torch.isfinite(loss):
        print(f"WARNING: Loss is {loss.item()}, skipping backward")
        return False

    # 2. Scaled backward
    scaler.scale(loss).backward()

    # 3. Unscale before gradient clipping (required!)
    scaler.unscale_(optimizer)

    # 4. Check for NaN/Inf in gradients BEFORE clipping
    found_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN/Inf in gradient of {name}")
                found_nan = True

    if found_nan:
        # Reset gradients and skip update
        optimizer.zero_grad()
        scaler.update()  # Reduce scale factor
        return False

    # 5. Gradient clipping (on unscaled gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 6. Optimizer step
    scaler.step(optimizer)
    scaler.update()

    return True
```

### Layer-Specific Numerical Instability Patterns

**Common unstable operations in FP16:**

1. **LayerNorm with small epsilon:**
```python
# BAD: epsilon too small for FP16
layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)  # Will underflow!

# GOOD: larger epsilon for FP16
layer_norm = nn.LayerNorm(hidden_size, eps=1e-7)
```

2. **Softmax on large logits:**
```python
# BAD: logits can overflow FP16 max (65504)
attention = torch.softmax(qk / sqrt(d_k), dim=-1)

# GOOD: subtract max for numerical stability
qk_max = qk.max(dim=-1, keepdim=True)[0]
attention = torch.softmax((qk - qk_max) / sqrt(d_k), dim=-1)
```

3. **Log-softmax numerical issues:**
```python
# BAD: log(0) = -inf for zero probabilities
log_probs = torch.log(torch.softmax(logits, dim=-1))

# GOOD: use log_softmax (numerically stable)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
```

---

## Section 2: Gradient Underflow/Overflow & GradScaler Tuning (~125 lines)

### The Gradient Underflow Problem

**FP16 range limitations:**
- **Minimum**: 5.96e-8 (subnormal), 6.10e-5 (normal)
- **Maximum**: 65,504

**Problem:** In billion-parameter models, gradients span 10+ orders of magnitude. When GradScaler adjusts for large gradients, small gradients underflow to zero.

From [Medium article](https://medium.com/@jbensnyder/solving-the-limits-of-mixed-precision-training-231019128b4b) (accessed 2025-11-13):
> "The problem is now the multiplier is so low that the smaller gradients are dropping to zero. The scaler doesn't monitor for zero gradients, only NaNs... Over days of training, and with large batch sizes, the probability of outlier cases producing NaNs increases."

### Diagnosing Gradient Underflow

**Monitor zero gradient frequency:**

```python
def track_gradient_underflow(model, log_every=100):
    """Track percentage of gradients that are zero (underflowed)."""

    total_params = 0
    zero_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += param.grad.numel()
            zero_grads += (param.grad == 0).sum().item()

    underflow_pct = 100.0 * zero_grads / total_params if total_params > 0 else 0

    return {
        'total_params': total_params,
        'zero_grads': zero_grads,
        'underflow_pct': underflow_pct
    }

# Usage in training loop
if step % 100 == 0:
    stats = track_gradient_underflow(model)
    print(f"Step {step}: {stats['underflow_pct']:.2f}% gradients are zero")

    # WARNING: >5% zero gradients indicates instability
    if stats['underflow_pct'] > 5.0:
        print("WARNING: High gradient underflow detected!")
```

### GradScaler Tuning Strategies

**Default GradScaler parameters:**
```python
scaler = torch.cuda.amp.GradScaler(
    init_scale=65536.0,      # Initial loss scale (2^16)
    growth_factor=2.0,        # Multiply scale by this when stable
    backoff_factor=0.5,       # Divide scale by this when NaN detected
    growth_interval=2000,     # Steps between scale increases
    enabled=True
)
```

**Problem scenarios:**

**Scenario 1: Scale drops too low (gradient underflow)**
```python
# Symptom: Gradients frequently zero, loss plateaus
# Solution: Increase init_scale, slower backoff

scaler = torch.cuda.amp.GradScaler(
    init_scale=131072.0,      # 2^17 instead of 2^16
    backoff_factor=0.75,      # Reduce less aggressively (0.75 instead of 0.5)
    growth_interval=1000,     # Grow back faster
)
```

**Scenario 2: Scale too high (frequent overflow)**
```python
# Symptom: Frequent NaN detection, scale constantly backing off
# Solution: Lower init_scale, faster backoff

scaler = torch.cuda.amp.GradScaler(
    init_scale=32768.0,       # 2^15 instead of 2^16
    backoff_factor=0.5,       # Standard backoff
    growth_interval=4000,     # More conservative growth
)
```

**Scenario 3: Wide gradient range (transformers, LLMs)**
```python
# Symptom: Some layers overflow, others underflow
# Solution: Per-parameter gradient scaling (expensive!)
# Alternative: Switch to BF16 (covered in Section 3)
```

### Dynamic Loss Scaling Diagnostics

**Log GradScaler state:**

```python
class GradScalerMonitor:
    """Monitor GradScaler behavior over training."""

    def __init__(self, scaler):
        self.scaler = scaler
        self.history = {
            'step': [],
            'scale': [],
            'growth_tracker': [],
            'found_inf': []
        }

    def log_state(self, step):
        """Log current scaler state."""
        self.history['step'].append(step)
        self.history['scale'].append(self.scaler.get_scale())
        self.history['growth_tracker'].append(self.scaler._growth_tracker)

        # Check if inf/nan was found this step
        found_inf = any(self.scaler._found_inf_per_device.values())
        self.history['found_inf'].append(found_inf)

    def plot_history(self):
        """Visualize scaler behavior."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot scale factor over time
        ax1.plot(self.history['step'], self.history['scale'])
        ax1.set_ylabel('Loss Scale')
        ax1.set_yscale('log')
        ax1.set_title('GradScaler Loss Scale Over Training')
        ax1.grid(True)

        # Plot overflow events
        overflow_steps = [s for s, f in zip(self.history['step'], self.history['found_inf']) if f]
        ax2.scatter(overflow_steps, [1]*len(overflow_steps), c='red', alpha=0.5)
        ax2.set_ylabel('Overflow Event')
        ax2.set_xlabel('Training Step')
        ax2.set_title('Gradient Overflow Events')

        plt.tight_layout()
        plt.savefig('gradscaler_diagnostics.png')

# Usage
monitor = GradScalerMonitor(scaler)

for step, batch in enumerate(dataloader):
    # ... training code ...
    monitor.log_state(step)

    if step % 1000 == 0:
        monitor.plot_history()
```

### Advanced: Adaptive Loss Scaling

**Per-layer loss scaling (from "Adaptive Loss Scaling for Mixed Precision Training" paper):**

```python
class PerLayerGradScaler:
    """Adaptive loss scaling per layer group."""

    def __init__(self, model, init_scale=65536.0):
        self.scalers = {}

        # Create separate scaler for each layer group
        for name, param in model.named_parameters():
            layer_id = name.split('.')[0]  # First component of name
            if layer_id not in self.scalers:
                self.scalers[layer_id] = torch.cuda.amp.GradScaler(
                    init_scale=init_scale
                )

    def scale(self, loss):
        # Use first scaler for loss (all scalers should scale loss same)
        return list(self.scalers.values())[0].scale(loss)

    def step(self, optimizer):
        # Step each scaler (they check their own parameters)
        for scaler in self.scalers.values():
            scaler.step(optimizer)

    def update(self):
        # Update all scalers
        for scaler in self.scalers.values():
            scaler.update()
```

**Note:** Per-layer scaling helps but adds overhead. For most cases, switching to BF16 is more practical.

### Gradient Clipping with Mixed Precision

**CRITICAL: Unscale before clipping!**

```python
# ❌ WRONG: Clip scaled gradients
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # WRONG!
scaler.step(optimizer)

# ✅ CORRECT: Unscale, then clip, then step
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Must unscale first!
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Why:** Clipping scaled gradients by max_norm=1.0 effectively clips at `1.0 * scale_factor`, which defeats the purpose of gradient clipping.

---

## Section 3: Precision Format Stability (FP16 vs BF16 vs FP8) (~125 lines)

### Numerical Stability Comparison

**Format specifications:**

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | ~7 decimal digits |
| FP16 | 16 | 5 | 10 | ±65,504 | ~3 decimal digits |
| BF16 | 16 | 8 | 7 | ±3.4e38 | ~2 decimal digits |
| TF32 | 19 | 8 | 10 | ±3.4e38 | ~3 decimal digits |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | ~0.9 decimal digits |
| FP8 E5M2 | 8 | 5 | 2 | ±57,344 | ~0.6 decimal digits |

From [Beam Cloud: BF16 vs FP16](https://www.beam.cloud/blog/bf16-vs-fp16) (accessed 2025-11-13):
> "BFloat16 offers better stability during training than FP16. The increased dynamic range means fewer underflow/overflow issues, especially in deep transformer models."

### When to Use Each Format

**FP16 (best for):**
- ✅ Small-medium models (< 1B parameters)
- ✅ Computer vision (ResNet, EfficientNet)
- ✅ Models without extreme gradient ranges
- ❌ Large transformers (GPT, BERT variants)
- ❌ Training from scratch (vs fine-tuning)

**BF16 (best for):**
- ✅ Large language models (GPT, LLaMA, T5)
- ✅ Deep transformers (> 12 layers)
- ✅ Training from scratch
- ✅ Models with numerical instability in FP16
- ❌ Older GPUs (requires Ampere/Hopper)

**FP8 (best for):**
- ✅ Hopper GPUs (H100/H200) - 4× speedup
- ✅ Extremely large models (> 70B parameters)
- ✅ Inference optimization
- ❌ Training without Transformer Engine
- ❌ Models requiring high precision

### Migrating from FP16 to BF16

**PyTorch migration (minimal changes):**

```python
# FP16 training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():  # Defaults to FP16
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# BF16 training (PyTorch 1.12+)
scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaling!
with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Specify BF16
    output = model(input)
    loss = criterion(output, target)
loss.backward()  # No scaling needed
optimizer.step()
```

**Key difference:** BF16 has same range as FP32, so gradient scaling is typically **not needed**. You can still use GradScaler for compatibility, but set `enabled=False`.

### TensorFloat32 (TF32) - The Middle Ground

**Enabling TF32 (PyTorch 1.7+, Ampere+ GPUs):**

```python
# Enable TF32 for matmul and cuDNN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# No other code changes needed - TF32 replaces FP32 automatically
model = Model().cuda()
optimizer = Adam(model.parameters())

# Training loop unchanged
for batch in dataloader:
    output = model(batch)  # Uses TF32 internally
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**TF32 characteristics:**
- 19-bit format (8 exp + 10 mantissa + sign)
- Same range as FP32, more precision than BF16
- ~8× faster than FP32 on Ampere/Hopper
- Transparent replacement (no code changes)
- Enabled by default in PyTorch < 1.11, disabled in 1.12+

### FP8 Training with Transformer Engine

**NVIDIA Transformer Engine setup:**

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# FP8 recipe configuration
fp8_recipe = recipe.DelayedScaling(
    margin=0,              # Scaling margin
    interval=1,            # Amax history interval
    fp8_format=recipe.Format.HYBRID,  # E4M3 for forward, E5M2 for backward
    amax_history_len=1024,            # Amax history length
    amax_compute_algo="max"           # Amax computation algorithm
)

# Wrap model layers with TE
model = te.Linear(in_features, out_features)

# Training with FP8
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

**FP8 format selection:**
- **E4M3** (4 exponent, 3 mantissa): Better precision, use for forward pass
- **E5M2** (5 exponent, 2 mantissa): Better range, use for backward pass
- **HYBRID**: E4M3 forward, E5M2 backward (recommended)

### Debugging FP8 NaN Propagation

**FP8-specific instabilities:**

1. **Quantization error accumulation:**
```python
# Check FP8 scaling factors
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) as fp8_context:
    output = model(input)

    # Inspect amax (absolute maximum) values
    for name, amax in fp8_context.get_amax_dict().items():
        print(f"{name}: amax={amax}")

        # WARNING: amax > 440 (FP8 E4M3 max) indicates overflow risk
        if amax > 440:
            print(f"  WARNING: {name} approaching FP8 E4M3 limit!")
```

2. **Delayed scaling convergence:**
```python
# Increase amax_history_len if frequent overflow
fp8_recipe = recipe.DelayedScaling(
    amax_history_len=2048,  # Longer history (default 1024)
    interval=2,              # Update every 2 steps (default 1)
)
```

3. **Mixed FP8/BF16 fallback:**
```python
# Some layers may be too unstable for FP8
class MixedPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = te.Linear(768, 3072)      # FP8
        self.layer2 = te.LayerNorm(3072)        # FP8
        self.layer3 = nn.Linear(3072, 768)      # BF16 fallback

    def forward(self, x):
        with te.fp8_autocast(enabled=True):
            x = self.layer1(x)
            x = self.layer2(x)
        # layer3 runs in BF16 (outside FP8 context)
        x = self.layer3(x.to(torch.bfloat16))
        return x
```

### Format Selection Decision Tree

```
Is model < 1B params and vision-focused?
├─ YES → Try FP16 first
└─ NO ↓

Is model a transformer with > 12 layers?
├─ YES → Use BF16
└─ NO ↓

Do you have Ampere/Hopper GPU?
├─ YES → Try BF16, consider TF32
└─ NO → Stick with FP16, tune GradScaler carefully

Is model > 70B params on Hopper GPU?
├─ YES → Evaluate FP8 with Transformer Engine
└─ NO → BF16 is likely optimal
```

---

## Section 4: Production Stability Monitoring & Automatic Fallback (~125 lines)

### Early Warning System for Instability

**Gradient health metrics to track:**

```python
class GradientHealthMonitor:
    """Monitor gradient health for early instability detection."""

    def __init__(self, model, alert_thresholds=None):
        self.model = model
        self.thresholds = alert_thresholds or {
            'zero_grad_pct': 5.0,      # Alert if >5% grads are zero
            'large_grad_pct': 1.0,     # Alert if >1% grads near overflow
            'grad_norm_growth': 10.0,  # Alert if grad norm grows 10× in 100 steps
        }
        self.history = []

    def compute_metrics(self, step):
        """Compute gradient health metrics."""
        total_params = 0
        zero_grads = 0
        large_grads = 0  # Near FP16 max (65504)
        grad_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                total_params += param.grad.numel()
                zero_grads += (param.grad == 0).sum().item()
                large_grads += (param.grad.abs() > 60000).sum().item()
                grad_norm += param.grad.norm().item() ** 2

        grad_norm = (grad_norm ** 0.5) if total_params > 0 else 0.0

        metrics = {
            'step': step,
            'zero_grad_pct': 100.0 * zero_grads / total_params if total_params > 0 else 0,
            'large_grad_pct': 100.0 * large_grads / total_params if total_params > 0 else 0,
            'grad_norm': grad_norm,
        }

        self.history.append(metrics)
        return metrics

    def check_alerts(self, metrics):
        """Check for alert conditions."""
        alerts = []

        # Alert 1: High zero gradient percentage
        if metrics['zero_grad_pct'] > self.thresholds['zero_grad_pct']:
            alerts.append(f"High zero gradients: {metrics['zero_grad_pct']:.2f}%")

        # Alert 2: High large gradient percentage
        if metrics['large_grad_pct'] > self.thresholds['large_grad_pct']:
            alerts.append(f"Gradients near overflow: {metrics['large_grad_pct']:.2f}%")

        # Alert 3: Gradient norm explosion
        if len(self.history) > 100:
            old_norm = self.history[-100]['grad_norm']
            growth = metrics['grad_norm'] / old_norm if old_norm > 0 else 0
            if growth > self.thresholds['grad_norm_growth']:
                alerts.append(f"Gradient norm exploding: {growth:.2f}× growth")

        return alerts

# Usage
monitor = GradientHealthMonitor(model)

for step, batch in enumerate(dataloader):
    # Training step
    scaler.scale(loss).backward()

    # Check gradient health
    metrics = monitor.compute_metrics(step)
    alerts = monitor.check_alerts(metrics)

    if alerts:
        print(f"Step {step} ALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
        # Optionally trigger automatic remediation

    scaler.step(optimizer)
    scaler.update()
```

### Automatic Precision Fallback

**Auto-switch from FP16 to BF16 on instability:**

```python
class AutoPrecisionTrainer:
    """Automatically fallback to BF16 if FP16 unstable."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.use_fp16 = True
        self.consecutive_failures = 0
        self.max_failures = 10

        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch, criterion):
        """Single training step with auto fallback."""

        # Select dtype based on current mode
        dtype = torch.float16 if self.use_fp16 else torch.bfloat16

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=dtype):
            output = self.model(batch['input'])
            loss = criterion(output, batch['target'])

        # Check if loss is finite
        if not torch.isfinite(loss):
            self.consecutive_failures += 1
            print(f"Non-finite loss detected ({self.consecutive_failures}/{self.max_failures})")

            if self.consecutive_failures >= self.max_failures:
                if self.use_fp16:
                    print("Switching from FP16 to BF16 due to instability")
                    self.use_fp16 = False
                    self.scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaling
                    self.consecutive_failures = 0
                else:
                    raise RuntimeError("Training unstable even in BF16!")

            return None  # Skip this step

        # Reset failure counter on success
        self.consecutive_failures = 0

        # Backward pass
        if self.use_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()
```

### Production Monitoring Dashboard

**Metrics to track in production:**

```python
import wandb  # Or TensorBoard

class ProductionMonitor:
    """Comprehensive production monitoring for mixed precision training."""

    def __init__(self, project_name):
        wandb.init(project=project_name)

    def log_precision_metrics(self, step, model, scaler, loss):
        """Log all precision-related metrics."""

        # Basic metrics
        metrics = {
            'loss': loss.item() if torch.isfinite(loss) else float('nan'),
            'loss_scale': scaler.get_scale(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        # Gradient statistics
        grad_stats = self._compute_grad_stats(model)
        metrics.update(grad_stats)

        # Parameter statistics
        param_stats = self._compute_param_stats(model)
        metrics.update(param_stats)

        wandb.log(metrics, step=step)

    def _compute_grad_stats(self, model):
        """Compute gradient statistics."""
        grad_norms = []
        zero_count = 0
        total_count = 0

        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                zero_count += (param.grad == 0).sum().item()
                total_count += param.grad.numel()

        return {
            'grad_norm_mean': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
            'grad_norm_max': max(grad_norms) if grad_norms else 0,
            'grad_zero_pct': 100.0 * zero_count / total_count if total_count > 0 else 0,
        }

    def _compute_param_stats(self, model):
        """Compute parameter statistics."""
        param_norms = []

        for param in model.parameters():
            param_norms.append(param.norm().item())

        return {
            'param_norm_mean': sum(param_norms) / len(param_norms) if param_norms else 0,
            'param_norm_max': max(param_norms) if param_norms else 0,
        }
```

### Emergency Recovery Procedures

**Checkpoint rollback on instability:**

```python
class SafeCheckpointer:
    """Checkpoint with automatic rollback on instability."""

    def __init__(self, save_dir, keep_last_n=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []

    def save(self, model, optimizer, scaler, step, metrics):
        """Save checkpoint with metadata."""
        checkpoint = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'metrics': metrics,
        }

        path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)

        self.checkpoints.append((step, path, metrics))

        # Keep only last N checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_path, _ = self.checkpoints.pop(0)
            old_path.unlink()

    def rollback_to_stable(self, model, optimizer, scaler):
        """Rollback to most recent stable checkpoint."""
        # Find checkpoint with finite loss
        for step, path, metrics in reversed(self.checkpoints):
            if torch.isfinite(torch.tensor(metrics['loss'])):
                print(f"Rolling back to step {step}")
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scaler.load_state_dict(checkpoint['scaler_state'])
                return step

        raise RuntimeError("No stable checkpoint found!")

# Usage
checkpointer = SafeCheckpointer(save_dir="checkpoints")

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch, optimizer, scaler)

    # Save checkpoint every 1000 steps
    if step % 1000 == 0:
        metrics = {'loss': loss.item()}
        checkpointer.save(model, optimizer, scaler, step, metrics)

    # Check for instability
    if not torch.isfinite(loss):
        print("Instability detected, rolling back...")
        step = checkpointer.rollback_to_stable(model, optimizer, scaler)
```

### Real-World Deployment Checklist

**Pre-deployment validation:**

- [ ] Test on small dataset with FP16 for 1000 steps
- [ ] Monitor zero gradient percentage (should be < 5%)
- [ ] Check GradScaler behavior (scale should stabilize)
- [ ] Validate on outlier/difficult batches
- [ ] Set up monitoring dashboards (WandB/TensorBoard)
- [ ] Configure automatic checkpointing (every 500-1000 steps)
- [ ] Implement auto-fallback to BF16 if available
- [ ] Set up alerts for gradient health metrics
- [ ] Document precision format choice and reasoning
- [ ] Test recovery procedures (rollback, precision fallback)

**Runtime monitoring (every 100 steps):**
- Loss finite check
- Gradient norm tracking
- Zero gradient percentage
- GradScaler scale factor
- Parameter norm stability

**Alert thresholds:**
- Loss NaN → immediate rollback
- Zero grads > 5% → warning, consider BF16
- Grad norm exploding (10× in 100 steps) → reduce LR
- Scale factor < 1024 → gradient underflow likely

---

## Sources

**Source Documents:**
- [cuda/07-mixed-precision-training-internals.md](07-mixed-precision-training-internals.md) - AMP and GradScaler basics
- [cuda/09-runtime-errors-debugging-expert.md](09-runtime-errors-debugging-expert.md) - General debugging context

**Web Research (accessed 2025-11-13):**
- [Medium: Solving the Limits of Mixed Precision Training](https://medium.com/@jbensnyder/solving-the-limits-of-mixed-precision-training-231019128b4b) - Ben Snyder, comprehensive guide to gradient underflow, CLIP case study, BF16 migration
- [PyTorch GitHub Issue #40497: Mixed precision causes NaN loss](https://github.com/pytorch/pytorch/issues/40497) - Real-world NaN debugging case, GradScaler interaction with transformers
- [PyTorch Docs: Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html) - Official GradScaler documentation, autocast op reference
- [Beam Cloud: BF16 vs FP16](https://www.beam.cloud/blog/bf16-vs-fp16) - Precision format comparison, numerical stability analysis
- [NVIDIA Developer Forums: FP16/FP8 Training Stability](https://forums.developer.nvidia.com/t/fp16-fp8-training-stability-what-s-working-and-what-s-failing/341705) - Community experiences with stability issues

**Additional References:**
- Adaptive Loss Scaling for Mixed Precision Training (arXiv:1910.12385)
- NVIDIA Transformer Engine documentation
- PyTorch mixed precision training best practices
