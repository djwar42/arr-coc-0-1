# VLM Gradient Flow Debugging

Practical guide to debugging gradient flow issues in vision-language models, including vanishing/exploding gradients, layer-wise analysis, and fixing common training failures.

## Overview

Gradient flow problems are common in VLMs due to:
- **Deep architectures**: Vision encoders (24+ layers) + LLMs (32+ layers)
- **Multiple modalities**: Different gradient scales from vision vs text
- **Cross-modal connections**: Projection/resampler layers as bottlenecks
- **Frozen components**: Gradients must flow through frozen encoders

**Symptoms**: Loss plateau, NaN loss, slow convergence, poor alignment

## Monitoring Gradient Flow

### Essential Logging

```python
def log_gradient_statistics(model, step):
    """
    Log gradient norms per layer/module
    """
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_max = param.grad.abs().max().item()

            grad_stats[f"grad_norm/{name}"] = grad_norm
            grad_stats[f"grad_mean/{name}"] = grad_mean
            grad_stats[f"grad_max/{name}"] = grad_max

    # Log to wandb/tensorboard
    wandb.log(grad_stats, step=step)

    # Alert on issues
    for name, norm in grad_stats.items():
        if norm > 100:
            logging.warning(f"Step {step}: Large gradient {name}: {norm:.2f}")
        if norm < 1e-7:
            logging.warning(f"Step {step}: Vanishing gradient {name}: {norm:.2e}")
```

### Gradient Flow Visualization

```python
def visualize_gradient_flow(model):
    """
    Plot gradient flow through layers
    """
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.grad is not None and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu())
            max_grads.append(param.grad.abs().max().cpu())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label="max")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, label="mean")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"gradient_flow_step_{step}.png")
    plt.close()
```

## Common Gradient Flow Issues

### Issue 1: Vanishing Gradients in Projection Layer

**Symptom**: Projection layer gradients < 1e-6, model not learning

**Diagnosis**:
```python
# Check gradient flow through projection
proj_grad = model.projection.weight.grad.norm().item()
print(f"Projection gradient norm: {proj_grad:.2e}")

if proj_grad < 1e-6:
    print("WARNING: Vanishing gradients in projection layer!")
```

**Causes**:
- Vision encoder outputs too large (||v|| > 50)
- Vision encoder outputs too small (||v|| < 0.1)
- Too many frozen layers before projection

**Fixes**:

1. **Add LayerNorm before projection**:
```python
class StableProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)  # Normalize vision features
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, vision_feats):
        normed = self.norm(vision_feats)
        return self.proj(normed)
```

2. **Learnable scaling**:
```python
class ScaledProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))  # Learnable scale
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(self.scale * x)
```

3. **Check vision encoder output scale**:
```python
# During training
vision_output = vision_encoder(images)
print(f"Vision output norm: {vision_output.norm(dim=-1).mean():.3f}")
# Should be 1-10, not 0.01 or 100
```

### Issue 2: Exploding Gradients

**Symptom**: Loss → NaN after 100-1000 steps, gradient norms > 100

**Diagnosis**:
```python
def check_grad_explosion(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > 100:
        print(f"ALERT: Gradient explosion! Norm: {total_norm:.2f}")
        # Print worst offenders
        for name, p in model.named_parameters():
            if p.grad is not None:
                norm = p.grad.norm().item()
                if norm > 10:
                    print(f"  {name}: {norm:.2f}")
```

**Fixes**:

1. **Gradient clipping** (mandatory):
```python
# Clip by global norm
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # BLIP-2 uses 1.0, LLaVA uses 0.5
)
```

2. **Lower learning rate**:
```python
# If gradients still explode with clipping:
learning_rate = 1e-4  # Try 5e-5 or 1e-5
```

3. **Check attention scores**:
```python
# In cross-attention forward
scores = (query @ key.T) / sqrt(d_k)

# Log attention score range
print(f"Attention scores: min={scores.min():.2f}, max={scores.max():.2f}")
# Should be -10 to 10, not -100 to 100

# If too large, increase temperature
scores = scores / temperature  # temperature > 1.0
```

### Issue 3: Gradient Blockage at Frozen Boundaries

**Symptom**: Layers after frozen encoder have tiny gradients

**Diagnosis**:
```python
# Check gradient flow across frozen/unfrozen boundary
frozen_encoder_output = vision_encoder(images)  # Frozen
projection_output = projection(frozen_encoder_output)  # Trainable

# After backward pass
proj_grad = projection.weight.grad.norm().item()
print(f"Projection gradient: {proj_grad:.2e}")

# If < 1e-6, gradients are blocked
```

**Root cause**: Frozen encoder produces constant outputs → no gradient signal

**Fixes**:

1. **Unfreeze last N layers** of vision encoder:
```python
# Unfreeze last 4 layers
for layer in vision_encoder.layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True
```

2. **Add dropout after frozen encoder**:
```python
class ProjectionWithDropout(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Adds noise to gradients
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.dropout(x)  # Even with frozen encoder, gradients flow
        return self.proj(x)
```

3. **Use residual connections**:
```python
class ResidualProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.proj(x)  # Residual helps gradient flow
```

### Issue 4: Imbalanced Gradients Between Modalities

**Symptom**: Vision path gradients >> text path gradients (or vice versa)

**Diagnosis**:
```python
def check_gradient_balance(model):
    vision_grads = []
    text_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'vision' in name:
                vision_grads.append(param.grad.norm().item())
            elif 'text' in name:
                text_grads.append(param.grad.norm().item())

    vision_avg = np.mean(vision_grads)
    text_avg = np.mean(text_grads)
    ratio = vision_avg / (text_avg + 1e-8)

    print(f"Vision grad norm: {vision_avg:.2e}")
    print(f"Text grad norm: {text_avg:.2e}")
    print(f"Ratio: {ratio:.2f}")

    if ratio > 10 or ratio < 0.1:
        print("WARNING: Imbalanced gradients between modalities!")
```

**Fixes**:

1. **Separate learning rates**:
```python
optimizer = AdamW([
    {'params': model.vision_encoder.parameters(), 'lr': 1e-5},
    {'params': model.text_encoder.parameters(), 'lr': 1e-4},
    {'params': model.projection.parameters(), 'lr': 1e-3},
])
```

2. **Gradient scaling**:
```python
# In backward pass
loss.backward()

# Scale vision gradients
for name, param in model.vision_encoder.named_parameters():
    if param.grad is not None:
        param.grad *= 0.1  # Scale down if vision grads too large
```

3. **Loss term balancing**:
```python
# If vision loss dominates
total_loss = 0.5 * vision_loss + 1.0 * text_loss  # Rebalance
```

## Advanced Debugging Techniques

### Gradient Checkpointing Impact

```python
# With gradient checkpointing (saves memory, affects gradients)
model.vision_encoder.gradient_checkpointing_enable()

# Check if gradients change
before = get_grad_norm(model)
model.vision_encoder.gradient_checkpointing_disable()
after = get_grad_norm(model)

print(f"Gradient norm change: {before:.3f} → {after:.3f}")
# Small difference (<10%) is OK, large difference indicates numerical issues
```

### Numerical Stability Check

```python
def check_numerical_stability(model, images, texts):
    """
    Check for NaNs/Infs in forward and backward pass
    """
    # Forward pass
    outputs = model(images, texts)

    print("Forward pass check:")
    for name, value in outputs.items():
        has_nan = torch.isnan(value).any()
        has_inf = torch.isinf(value).any()
        if has_nan or has_inf:
            print(f"  {name}: NaN={has_nan}, Inf={has_inf}")

    # Backward pass
    loss = outputs['loss']
    loss.backward()

    print("Backward pass check:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any()
            has_inf = torch.isinf(param.grad).any()
            if has_nan or has_inf:
                print(f"  {name}: NaN={has_nan}, Inf={has_inf}")
```

### Layer-wise Learning Rate Finder

```python
def find_layer_learning_rates(model, dataloader, base_lr=1e-3):
    """
    Find optimal LR for each layer group
    """
    layer_groups = {
        'vision': [p for n, p in model.named_parameters() if 'vision' in n],
        'text': [p for n, p in model.named_parameters() if 'text' in n],
        'projection': [p for n, p in model.named_parameters() if 'proj' in n],
    }

    for group_name, params in layer_groups.items():
        lrs = [base_lr * 10**i for i in range(-2, 3)]  # 0.01x to 100x
        best_lr = None
        best_loss = float('inf')

        for lr in lrs:
            optimizer = AdamW(params, lr=lr)
            loss = train_one_batch(model, next(iter(dataloader)), optimizer)

            if loss < best_loss:
                best_loss = loss
                best_lr = lr

        print(f"{group_name} best LR: {best_lr:.2e} (loss: {best_loss:.4f})")
```

## Quick Debugging Checklist

```python
def debug_gradient_flow(model, images, texts):
    """
    Run all gradient flow checks
    """
    print("=" * 50)
    print("GRADIENT FLOW DEBUG REPORT")
    print("=" * 50)

    # 1. Forward pass
    outputs = model(images, texts)
    loss = outputs['loss']

    # 2. Backward pass
    loss.backward()

    # 3. Check for NaNs
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"\n1. NaN check: {'FAIL ❌' if has_nan else 'PASS ✓'}")

    # 4. Check gradient norms
    total_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    print(f"2. Total gradient norm: {total_norm:.2f} {'✓' if 0.1 < total_norm < 10 else '❌'}")

    # 5. Check projection layer
    proj_norm = model.projection.weight.grad.norm().item()
    print(f"3. Projection gradient: {proj_norm:.2e} {'✓' if proj_norm > 1e-6 else '❌'}")

    # 6. Check modality balance
    vision_norm = sum(p.grad.norm().item()**2 for n, p in model.named_parameters()
                      if 'vision' in n and p.grad is not None)**0.5
    text_norm = sum(p.grad.norm().item()**2 for n, p in model.named_parameters()
                    if 'text' in n and p.grad is not None)**0.5
    ratio = vision_norm / (text_norm + 1e-8)
    print(f"4. Vision/Text ratio: {ratio:.2f} {'✓' if 0.1 < ratio < 10 else '❌'}")

    # 7. Check attention (if applicable)
    if hasattr(model, 'cross_attention'):
        attn_weights = outputs.get('attention_weights')
        if attn_weights is not None:
            entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean()
            print(f"5. Attention entropy: {entropy:.2f} {'✓' if entropy > 1.0 else '❌ (collapsed)'}")

    print("=" * 50)
```

## Best Practices

1. **Always log gradients** during initial training (first 1000 steps)
2. **Use gradient clipping** (max_norm=1.0) from step 1
3. **Monitor projection layer** - it's the critical bottleneck
4. **Check numerical stability** every 100 steps
5. **Use mixed precision carefully** - can cause gradient underflow
6. **Visualize gradient flow** when debugging issues

---

**Sources:**
- Pascanu, R. et al. (2013). "On the difficulty of training Recurrent Neural Networks"
- Ba, J. et al. (2016). "Layer Normalization"
- Zhang, H. et al. (2019). "Residual Networks and Why They Work"
- Practical debugging techniques from BLIP-2, LLaVA, Flamingo codebases
