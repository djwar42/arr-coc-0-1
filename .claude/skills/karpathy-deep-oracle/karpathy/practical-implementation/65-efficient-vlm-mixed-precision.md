# Efficient VLM Mixed Precision Training

Complete guide to mixed precision training (FP16/BF16) for vision-language models, covering automatic mixed precision (AMP), memory optimization, numerical stability, and performance tuning.

## Overview

**Mixed precision training**: Use FP16/BF16 for compute, FP32 for critical operations
- **Speed**: 2-3x faster on modern GPUs (Ampere/Ada)
- **Memory**: 40-50% reduction
- **Risk**: Numerical instability if not done carefully

**Formats**:
- **FP32 (float32)**: Standard precision, safe, slow
- **FP16 (half)**: 2x faster, 2x less memory, risk of overflow/underflow
- **BF16 (bfloat16)**: Same speed/memory as FP16, more stable (same exponent range as FP32)

**Recommendation**: **Use BF16 on Ampere+ GPUs (A100, RTX 4090, H100)**

## PyTorch Automatic Mixed Precision (AMP)

### Basic Usage

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler (for FP16 only, not needed for BF16)
scaler = GradScaler()

model = VLMModel().cuda()
optimizer = AdamW(model.parameters(), lr=1e-4)

for images, texts in dataloader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast(dtype=torch.float16):  # or torch.bfloat16
        outputs = model(images, texts)
        loss = outputs['loss']

    # Backward pass with gradient scaling (FP16 only)
    scaler.scale(loss).backward()

    # Unscale gradients and clip
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
```

### BF16 (Simpler, Recommended)

```python
# BF16: No gradient scaler needed!
for images, texts in dataloader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):
        outputs = model(images, texts)
        loss = outputs['loss']

    loss.backward()  # No scaler needed

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

## FP16 vs BF16 Comparison

| Aspect | FP16 | BF16 |
|--------|------|------|
| Speed | 2-3x faster | 2-3x faster |
| Memory | 50% of FP32 | 50% of FP32 |
| Range | ±65,504 | ±3.4×10^38 (same as FP32) |
| Precision | Higher | Lower |
| Stability | Requires gradient scaling | More stable, no scaling needed |
| Overflow risk | High | Low |
| GPU support | All CUDA GPUs | Ampere+ (A100, RTX 3090/4090, H100) |
| **Recommendation** | Use for older GPUs | **Preferred for Ampere+** |

## Critical Operations to Keep in FP32

### 1. Loss Functions with Exponentials

```python
# Softmax, log_softmax → keep in FP32
with autocast(dtype=torch.bfloat16):
    logits = model(images, texts)  # BF16

# Cast to FP32 before softmax
logits_fp32 = logits.float()
loss = F.cross_entropy(logits_fp32, labels)  # FP32
```

**Why**: `exp()` in softmax can overflow in FP16

### 2. Layer Normalization

```python
# LayerNorm should use FP32 for stability
class StableLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Convert to FP32, normalize, convert back
        dtype = x.dtype
        x_fp32 = x.float()
        normalized = self.norm(x_fp32)
        return normalized.to(dtype)
```

**Note**: PyTorch LayerNorm automatically uses FP32 internally (since PyTorch 1.10)

### 3. Reductions (sum, mean)

```python
# Large reductions → use FP32
with autocast(dtype=torch.bfloat16):
    features = model.encoder(inputs)  # [B, N, D] in BF16

# Cast to FP32 for summation
features_fp32 = features.float()
pooled = features_fp32.mean(dim=1)  # FP32
```

### 4. Batch Normalization

```python
# BatchNorm should use FP32 running stats
class StableBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

        # Force running stats to FP32
        self.bn.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.float32))
        self.bn.register_buffer('running_var', torch.ones(num_features, dtype=torch.float32))

    def forward(self, x):
        return self.bn(x)
```

## Gradient Scaling (FP16 Only)

### Why Gradient Scaling?

FP16 range: ±65,504. Many gradients are < 1e-5 → **underflow to zero** → vanishing gradients.

**Solution**: Scale loss up by 2^N before backward, scale gradients down by 2^N after.

### Dynamic Gradient Scaling

```python
scaler = GradScaler(
    init_scale=2**16,      # Initial scale factor
    growth_factor=2.0,     # Multiply scale if no overflow
    backoff_factor=0.5,    # Divide scale if overflow
    growth_interval=2000   # Steps between scale increases
)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    with autocast(dtype=torch.float16):
        loss = model(batch)

    # Scale loss, backward
    scaler.scale(loss).backward()

    # Unscale before clipping (important!)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step (skips if NaN/Inf detected)
    scaler.step(optimizer)

    # Update scale for next iteration
    scaler.update()
```

### Manual Gradient Scaling (Advanced)

```python
# Custom scaling for specific layers
class CustomScaler:
    def __init__(self, scale=65536.0):
        self.scale = scale

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(self.scale)

    def step(self, optimizer):
        # Check for NaN/Inf
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters()
                      if p.grad is not None)

        if not has_nan:
            optimizer.step()
            return True
        else:
            print("Skipping step due to NaN gradients")
            return False
```

## Memory Optimization Techniques

### 1. Mixed Precision + Gradient Checkpointing

```python
# Combine for maximum memory savings
model = VLMModel()
model.vision_encoder.gradient_checkpointing_enable()

# Training loop
with autocast(dtype=torch.bfloat16):
    outputs = model(images, texts)  # Uses checkpointing internally
    loss = outputs['loss']

loss.backward()
```

**Memory savings**: 60-70% reduction vs FP32 without checkpointing

### 2. Optimizer State in FP32 (Critical)

```python
# WRONG: Optimizer state in BF16 → numerical issues
optimizer = AdamW(model.parameters(), lr=1e-4)
for param_group in optimizer.param_groups:
    param_group['dtype'] = torch.bfloat16  # ❌ BAD

# RIGHT: Keep optimizer state in FP32
optimizer = AdamW(model.parameters(), lr=1e-4)
# Optimizer states (momentum, variance) stay in FP32 automatically ✓
```

### 3. Model Weights Storage

```python
# Store model in BF16, optimizer keeps FP32 copy
model = model.to(dtype=torch.bfloat16)  # Model weights in BF16

optimizer = AdamW(model.parameters(), lr=1e-4)
# Optimizer maintains FP32 master copy internally
```

## Troubleshooting Mixed Precision Issues

### Issue 1: Loss Goes to NaN

**Diagnosis**:
```python
# Check where NaNs appear
with autocast(dtype=torch.float16):
    outputs = model(images, texts)

for name, value in outputs.items():
    if torch.isnan(value).any():
        print(f"NaN detected in {name}")
```

**Fixes**:

1. **Switch to BF16** (if on Ampere+):
```python
# FP16 → BF16
with autocast(dtype=torch.bfloat16):  # More stable
    outputs = model(images, texts)
```

2. **Increase gradient scale init** (FP16):
```python
scaler = GradScaler(init_scale=2**20)  # Larger initial scale
```

3. **Disable AMP for specific modules**:
```python
with autocast(dtype=torch.float16):
    vision_feats = model.vision_encoder(images)

# Disable autocast for problematic cross-attention
with autocast(enabled=False):
    fused = model.cross_attention(vision_feats.float(), text_feats.float())
```

### Issue 2: Gradient Underflow (FP16)

**Symptom**: Gradients become zero, no learning

**Diagnosis**:
```python
# Check gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_magnitude = param.grad.abs().max().item()
        if grad_magnitude < 1e-7:
            print(f"Gradient underflow in {name}: {grad_magnitude:.2e}")
```

**Fix**:
```python
# Increase loss scale
scaler = GradScaler(init_scale=2**20)  # Higher scale prevents underflow
```

### Issue 3: Slow Training (No Speedup)

**Causes**:
- Using FP16/BF16 on old GPU (pre-Volta)
- Too small batch size (< 32)
- CPU bottleneck in data loading

**Fixes**:

1. **Use larger batch size**:
```python
# Small batch: No speedup
batch_size = 16  # ❌

# Larger batch: 2x speedup
batch_size = 128  # ✓ Utilizes tensor cores
```

2. **Enable TF32** (Ampere+):
```python
# Enable TF32 for matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

3. **Profile to find bottleneck**:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    for _ in range(10):
        outputs = model(images, texts)
        outputs['loss'].backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices

### Training Configuration

```yaml
# Recommended settings for BF16 (Ampere+ GPUs)
mixed_precision:
  dtype: bfloat16
  enabled: true

training:
  batch_size: 256  # Larger batch for better utilization
  gradient_clipping: 1.0
  gradient_accumulation: 4

torch_settings:
  allow_tf32: true  # Extra speedup on Ampere+
  cudnn_benchmark: true
```

### For FP16 (Older GPUs)

```yaml
mixed_precision:
  dtype: float16
  enabled: true
  grad_scaler:
    init_scale: 65536  # 2^16
    growth_factor: 2.0
    backoff_factor: 0.5
    growth_interval: 2000
```

### Validation/Inference

```python
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    # Use same precision as training
    with autocast(dtype=torch.bfloat16):
        for images, texts in dataloader:
            outputs = model(images, texts)
            # ... compute metrics
```

## Performance Benchmarks

### Training Speed (LLaVA-7B, 1 epoch on COCO)

| Precision | GPU | Time | Memory | Accuracy |
|-----------|-----|------|--------|----------|
| FP32 | A100 80GB | 12 hours | 76GB | 78.5% |
| FP16 + scaler | A100 80GB | 5 hours | 42GB | 78.3% |
| BF16 | A100 80GB | 4.5 hours | 40GB | 78.4% |
| BF16 + TF32 | A100 80GB | 4 hours | 40GB | 78.4% |

### Memory Usage (BLIP-2, Stage 1)

| Configuration | Memory per GPU |
|---------------|----------------|
| FP32 baseline | 78GB (OOM on A100) |
| BF16 | 42GB |
| BF16 + gradient checkpointing | 28GB |
| BF16 + checkpointing + flash attention | 24GB |

## Key Takeaways

1. **Use BF16 on Ampere+ GPUs** - simpler, more stable than FP16
2. **FP16 requires gradient scaling** - BF16 doesn't
3. **Keep optimizer state in FP32** - critical for stability
4. **Disable AMP for problematic ops** - softmax, large reductions
5. **Enable TF32** for extra speedup on Ampere+
6. **Larger batches** → better tensor core utilization

---

**Sources**:
- Micikevicius, P. et al. (2018). "Mixed Precision Training"
- NVIDIA AMP Documentation: https://pytorch.org/docs/stable/amp.html
- Kalamkar, D. et al. (2019). "A Study of BFLOAT16 for Deep Learning Training"
