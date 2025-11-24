# Cross-Attention Training Stability

Practical guide to stabilizing cross-attention layers in vision-language models, covering gradient clipping, initialization strategies, warmup schedules, and debugging unstable training.

## Overview

Cross-attention layers connect visual and textual modalities in VLMs but are notoriously unstable during training due to:
- **Gradient explosion**: Large gradients from attention score exp()
- **Attention collapse**: All queries attend to single key (mode collapse)
- **Vanishing gradients**: Deep cross-attention stacks lose signal
- **Feature scale mismatch**: Vision and text embeddings on different scales

**Used by**: Flamingo, BLIP/BLIP-2 (Q-Former), Perceiver-based models

**Note**: Many modern VLMs (LLaVA, Qwen-VL) avoid cross-attention by using simple concatenation, which is more stable but less flexible.

## Common Instability Patterns

### Pattern 1: Gradient Explosion

**Symptoms**:
- Loss spikes to NaN after few hundred steps
- Attention logits grow unbounded (>100)
- Parameters become NaN

**Root cause**:
```python
# Softmax in attention amplifies large values
scores = Q @ K.T / sqrt(d_k)  # Can be large
attn = softmax(scores)         # exp() explodes if scores > 20
```

**Solutions**:

1. **Gradient clipping** (mandatory):
```yaml
training:
  max_grad_norm: 1.0  # Clip gradients to norm 1.0
  grad_clip_type: "norm"  # or "value" for per-parameter clipping
```

2. **Attention logit temperature**:
```python
class StableC

rossAttention(nn.Module):
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.attention = nn.MultiheadAttention(dim, num_heads=8)

    def forward(self, query, key, value):
        # Scale attention scores
        attn_output, attn_weights = self.attention(
            query, key, value,
            attn_mask=None
        )
        # Learnable temperature prevents explosion
        attn_weights = attn_weights / self.temperature.clamp(min=0.1)
        return attn_output
```

3. **Use pre-LayerNorm architecture**:
```python
# BAD: Post-LN (unstable)
x = x + CrossAttention(x, visual_features)
x = LayerNorm(x)

# GOOD: Pre-LN (stable)
x = x + CrossAttention(LayerNorm(x), visual_features)
```

### Pattern 2: Attention Collapse

**Symptoms**:
- All attention weights focus on single token
- Attention entropy decreases to near-zero
- Model ignores most visual information

**Diagnosis**:
```python
def check_attention_collapse(attn_weights):
    # attn_weights: [batch, num_heads, query_len, key_len]
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
    avg_entropy = entropy.mean()
    print(f"Attention entropy: {avg_entropy:.3f}")
    # Healthy: > 2.0, Collapsed: < 0.5
    return avg_entropy < 0.5
```

**Solutions**:

1. **Attention dropout**:
```python
attention = nn.MultiheadAttention(
    embed_dim=768,
    num_heads=12,
    dropout=0.1  # Dropout on attention weights
)
```

2. **Entropy regularization loss**:
```python
def attention_entropy_loss(attn_weights, target_entropy=2.0):
    # Encourage diverse attention
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
    loss = F.mse_loss(entropy.mean(), torch.tensor(target_entropy))
    return loss

total_loss = task_loss + 0.01 * attention_entropy_loss(attn_weights)
```

3. **Multi-head attention** (more heads = more diversity):
```python
# Use more heads for cross-attention than self-attention
self_attention_heads = 12
cross_attention_heads = 16  # More diversity
```

### Pattern 3: Feature Scale Mismatch

**Symptoms**:
- Visual features dominate or are ignored
- Unstable early training (first 1000 steps)
- Attention weights either all ~0 or all ~1

**Root cause**:
```python
# Vision features: ||v|| ≈ 10-50 (from CLIP)
# Text features:   ||t|| ≈ 1-5 (from LLM embeddings)
# Attention: Q @ K.T produces mismatched scales
```

**Solutions**:

1. **Feature normalization before cross-attention**:
```python
class NormalizedCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=12)

    def forward(self, text_features, visual_features):
        # Normalize both modalities
        text_norm = self.query_norm(text_features)
        visual_norm = self.key_norm(visual_features)
        output, _ = self.cross_attn(text_norm, visual_norm, visual_norm)
        return output
```

2. **Learnable scaling factor**:
```python
class ScaledCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=12)
        self.scale = nn.Parameter(torch.ones(1))  # Learnable

    def forward(self, query, key, value):
        scaled_key = key * self.scale
        return self.cross_attn(query, scaled_key, value)
```

## Initialization Strategies

### Xavier/Glorot for Cross-Attention (Standard)

```python
def init_cross_attention(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

cross_attention.apply(init_cross_attention)
```

### Small Random Init for Query Projections

```python
def init_query_projection(module):
    # Initialize query projection with smaller values
    if 'query' in module.name:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Small std
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

**Rationale**: Smaller query projections → smaller attention logits → more stable early training

### Zero Init for Output Projection (Flamingo-style)

```python
# Initialize cross-attention output projection to zero
# Model starts by passing through residual connection
nn.init.zeros_(cross_attn.out_proj.weight)
nn.init.zeros_(cross_attn.out_proj.bias)
```

**Effect**: Cross-attention starts as identity, gradually learns to incorporate visual features

## Warmup Strategies

### Learning Rate Warmup

```python
def get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Use longer warmup for cross-attention
warmup_steps = int(0.1 * total_training_steps)  # 10% warmup
```

### Layer-wise Warmup

```python
def layer_wise_warmup(model, current_step, warmup_steps):
    # Gradually unfreeze cross-attention layers
    num_layers = len(model.cross_attention_layers)
    active_layers = int((current_step / warmup_steps) * num_layers) + 1

    for i, layer in enumerate(model.cross_attention_layers):
        if i < active_layers:
            layer.requires_grad_(True)
        else:
            layer.requires_grad_(False)
```

### Attention Dropout Warmup

```python
# Start with high dropout, decrease over time
def get_attention_dropout(current_step, total_steps):
    start_dropout = 0.3
    end_dropout = 0.1
    progress = current_step / total_steps
    return start_dropout + (end_dropout - start_dropout) * progress

# Update during training
model.cross_attention.dropout = get_attention_dropout(step, total_steps)
```

## Gated Cross-Attention (Flamingo Method)

**Most stable cross-attention design**:

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=12)
        self.gate = nn.Parameter(torch.zeros(1))  # Initialized to 0

    def forward(self, text_features, visual_features):
        # Cross-attend to visual features
        attn_out, _ = self.cross_attn(
            text_features,
            visual_features,
            visual_features
        )
        # Gating: starts at 0 (no visual influence), learns to open
        gated_out = torch.tanh(self.gate) * attn_out
        return text_features + gated_out  # Residual connection
```

**Why stable**:
1. Gate initialized to 0 → model starts without visual influence
2. tanh(gate) bounds output to [-1, 1]
3. Gradual learning of visual integration
4. Can't explode (bounded by tanh)

**Used by**: Flamingo (introduced), adopted by many subsequent models

## Gradient Clipping Configurations

### Global Norm Clipping (Recommended)

```python
# Clip total gradient norm across all parameters
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Flamingo uses 1.0, BLIP-2 uses 0.5
)
```

### Per-Layer Norm Clipping

```python
# Clip cross-attention layers separately
cross_attn_params = [p for n, p in model.named_parameters()
                     if 'cross_attn' in n]
other_params = [p for n, p in model.named_parameters()
                if 'cross_attn' not in n]

torch.nn.utils.clip_grad_norm_(cross_attn_params, max_norm=0.5)  # Stricter
torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)       # Standard
```

### Adaptive Clipping

```python
def adaptive_grad_clip(model, clip_percentile=1.0):
    # Clip based on gradient norm percentile
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    if grad_norms:
        threshold = np.percentile(grad_norms, clip_percentile * 100)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=threshold)
```

## Debugging Unstable Training

### Monitor Attention Weights

```python
def log_attention_statistics(attn_weights, step):
    # attn_weights: [batch, heads, queries, keys]
    stats = {
        'max': attn_weights.max().item(),
        'min': attn_weights.min().item(),
        'mean': attn_weights.mean().item(),
        'std': attn_weights.std().item(),
        'entropy': compute_entropy(attn_weights),
    }

    wandb.log({f"attention/{k}": v for k, v in stats.items()}, step=step)

    # Alert if unstable
    if stats['max'] > 0.99:
        logging.warning(f"Step {step}: Attention collapse detected!")
    if stats['std'] > 0.4:
        logging.warning(f"Step {step}: High attention variance!")
```

### Monitor Gradient Norms

```python
def log_gradient_norms(model, step):
    total_norm = 0
    cross_attn_norm = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2

            if 'cross_attn' in name:
                cross_attn_norm += param_norm ** 2

    total_norm = total_norm ** 0.5
    cross_attn_norm = cross_attn_norm ** 0.5

    wandb.log({
        'grad_norm/total': total_norm,
        'grad_norm/cross_attention': cross_attn_norm,
    }, step=step)

    if total_norm > 100:
        logging.error(f"Step {step}: Gradient explosion! Norm: {total_norm}")
```

### Check Feature Scales

```python
def check_feature_scales(text_features, visual_features, step):
    text_norm = text_features.norm(dim=-1).mean().item()
    visual_norm = visual_features.norm(dim=-1).mean().item()

    ratio = visual_norm / (text_norm + 1e-8)

    wandb.log({
        'features/text_norm': text_norm,
        'features/visual_norm': visual_norm,
        'features/scale_ratio': ratio,
    }, step=step)

    if ratio > 10 or ratio < 0.1:
        logging.warning(f"Step {step}: Feature scale mismatch! Ratio: {ratio:.2f}")
```

## Best Practices Summary

### Training Configuration

```yaml
# Recommended settings for stable cross-attention training
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.05
  betas: [0.9, 0.98]  # Higher beta2 for stability

lr_schedule:
  warmup_steps: 5000  # 10% of training
  schedule: "cosine"

gradient_clipping:
  max_grad_norm: 1.0
  clip_type: "norm"

cross_attention:
  num_heads: 12
  dropout: 0.1
  use_gating: true       # Flamingo-style gates
  init_method: "zero"    # Zero init output projection
  temperature: 1.0       # Learnable temperature

monitoring:
  log_attention_every: 100
  log_gradients_every: 100
  check_nan_every: 10
```

### Code Template

```python
class StableCrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        # Pre-LayerNorm for stability
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)

        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Gating mechanism (Flamingo-style)
        self.gate = nn.Parameter(torch.zeros(1))

        # Zero init output projection
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, text_features, visual_features):
        # Normalize inputs
        text_norm = self.query_norm(text_features)
        visual_norm = self.key_norm(visual_features)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            text_norm, visual_norm, visual_norm
        )

        # Gated residual
        gated_out = torch.tanh(self.gate) * attn_out
        output = text_features + gated_out

        return output, attn_weights
```

---

**Sources:**
- Alayrac, J. et al. (2022). "Flamingo: a Visual Language Model for Few-Shot Learning"
- Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training"
- Vaswani, A. et al. (2017). "Attention is All You Need"
- Xiong, R. et al. (2020). "On Layer Normalization in the Transformer Architecture"
