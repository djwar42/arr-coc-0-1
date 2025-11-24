# Vision Encoder Freezing Strategies

Complete guide to when, what, and how to freeze vision encoders during VLM training, including analysis of compute/performance trade-offs, stage-specific strategies, and adaptive unfreezing techniques.

## Overview

Vision encoder freezing is a critical decision in VLM training that affects:
- **Compute cost**: Frozen encoders save 40-60% training time
- **Memory**: Frozen encoders use ~30% less GPU memory (no gradients/optimizer states)
- **Alignment quality**: Unfrozen encoders can adapt better to downstream tasks
- **Stability**: Frozen encoders prevent catastrophic forgetting

**Key principle**: Freeze when leveraging strong pretrained features; unfreeze when adaptation is needed.

## Standard Freezing Strategies by Training Stage

### Stage 1: Pre-Alignment (Feature Alignment)

**Standard approach: FREEZE vision encoder**

```yaml
frozen:
  vision_encoder: true
  language_model: true
trainable:
  projection_layer: true
```

**Rationale**:
- CLIP/SigLIP already learned strong visual features
- Alignment layer just needs to map to LLM space
- Saves massive compute (only train 2-8M projection parameters)

**Exception - Unfreeze when**:
- Using non-CLIP encoder (e.g., DINOv2, MAE)
- Target domain differs significantly from CLIP training (medical, satellite images)
- Have compute budget and want optimal alignment

**Trade-offs**:

| Approach | Training Time | GPU Memory | Alignment Quality | Risk |
|----------|---------------|------------|-------------------|------|
| Frozen | 1x (baseline) | 1x | Good | Low |
| Unfrozen | 2.5-3x | 1.4x | Better | Medium (overfitting) |

### Stage 2: Multimodal Pre-Training

**Standard approach: FREEZE vision encoder (most models)**

```yaml
frozen:
  vision_encoder: true
  language_model: true  # or LoRA
trainable:
  visual_abstractor: true
  lora_adapters: true  # if using PEFT
```

**Used by**: BLIP-2, LLaVA, Qwen-VL, MiniGPT-4

**Alternative: UNFREEZE for better alignment (advanced)**

```yaml
frozen:
  language_model: true
trainable:
  vision_encoder: true
  visual_abstractor: true
```

**Used by**: InternVL, CogVLM, Vary

**Decision matrix**:

```python
def should_unfreeze_vision_stage2(context):
    if context.compute_budget == "limited":
        return False  # Keep frozen

    if context.domain_shift == "large":  # Medical, remote sensing
        return True  # Unfreeze and adapt

    if context.vision_encoder == "CLIP":
        return False  # CLIP features strong enough

    if context.target_tasks == "fine_grained":  # OCR, charts, diagrams
        return True  # Need adaptation

    return False  # Default: keep frozen
```

### Stage 3: Instruction Tuning

**Standard approach: KEEP FROZEN from Stage 2**

```yaml
frozen:
  vision_encoder: true  # IMPORTANT: Keep frozen
trainable:
  visual_abstractor: true
  language_model: true  # Via LoRA/QLoRA
```

**Rationale**:
- Vision features already well-aligned from Stage 2
- Unfreezing risks catastrophic forgetting
- Instruction tuning should focus on language understanding
- Saves memory for larger LLM fine-tuning

**Exception - Unfreeze for specialized domains**:
```yaml
# Only if ALL conditions met:
# 1. Domain-specific images (medical, satellite)
# 2. Have large instruction dataset (50k+)
# 3. Can afford 2-3x longer training
frozen:
  vision_encoder: false
trainable:
  vision_encoder: true
  visual_abstractor: true
  language_model: true
```

## Partial Freezing Strategies

### Layer-wise Freezing

Freeze early layers, train later layers:

```python
# Freeze bottom 80% of vision encoder
total_layers = len(vision_encoder.layers)
freeze_until = int(0.8 * total_layers)

for i, layer in enumerate(vision_encoder.layers):
    if i < freeze_until:
        layer.requires_grad_(False)  # Freeze
    else:
        layer.requires_grad_(True)   # Train
```

**Rationale**: Early layers capture generic features (edges, textures), later layers are task-specific

**Used by**: Some CogVLM variants, domain adaptation studies

**Benefits**:
- 60% of compute savings vs full unfreezing
- Better adaptation than full freezing
- Intermediate memory usage

**When to use**:
- Domain adaptation (natural → medical images)
- Fine-grained recognition tasks
- Mid-range compute budget

### Attention-only Unfreezing

Freeze convolutional/feedforward, train attention:

```python
for name, param in vision_encoder.named_parameters():
    if 'attn' in name or 'attention' in name:
        param.requires_grad = True   # Train attention
    else:
        param.requires_grad = False  # Freeze FFN/conv
```

**Benefits**:
- 40% of full training compute
- Allows spatial reasoning adaptation
- Preserves low-level feature extraction

**Use cases**:
- Spatial reasoning tasks (NLVR2, referring expression)
- Multi-object scenarios
- Video understanding (temporal attention)

### Norm Layer Unfreezing (Cheapest Adaptation)

Only train LayerNorm/BatchNorm parameters:

```python
for name, param in vision_encoder.named_parameters():
    if 'norm' in name or 'bn' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**Benefits**:
- <5% compute overhead
- Allows feature distribution adaptation
- Extremely memory efficient

**Surprisingly effective for**:
- Domain shift (natural → document images)
- Style transfer
- Lighting/color adaptation

## Adaptive Unfreezing (Progressive Training)

Start frozen, gradually unfreeze during training:

### Linear Unfreezing Schedule

```python
def unfreeze_schedule(current_step, total_steps, vision_encoder):
    # Unfreeze layers linearly over training
    unfreeze_ratio = current_step / total_steps
    num_layers = len(vision_encoder.layers)
    num_trainable = int(unfreeze_ratio * num_layers)

    for i in range(num_layers):
        trainable = (i >= num_layers - num_trainable)
        vision_encoder.layers[i].requires_grad_(trainable)
```

**Schedule**:
- Steps 0-25%: Fully frozen
- Steps 25-75%: Gradually unfreeze from top layer down
- Steps 75-100%: Fully unfrozen

**Benefits**:
- Stable early training (no gradient explosion)
- Progressive adaptation
- Best of both worlds (efficiency + adaptation)

### Layer Discrimination with Different Learning Rates

Keep all layers trainable but use different LRs:

```python
param_groups = [
    {'params': vision_encoder.layers[:12].parameters(), 'lr': 1e-6},   # Early layers: very small LR
    {'params': vision_encoder.layers[12:20].parameters(), 'lr': 5e-6}, # Mid layers: small LR
    {'params': vision_encoder.layers[20:].parameters(), 'lr': 1e-5},   # Late layers: normal LR
]

optimizer = AdamW(param_groups)
```

**Rationale**: Early layers should change minimally, later layers need more adaptation

**Used by**: Some BERT fine-tuning approaches, transferable to VLMs

## Model-Specific Freezing Recommendations

### BLIP-2

**Recommended**:
```yaml
stage_1:
  vision_encoder: freeze  # CLIP features sufficient
  q_former: train
  llm: freeze

stage_2:
  vision_encoder: freeze
  q_former: train
  llm: freeze
```

**Reasoning**: Q-Former designed to work with frozen models

### LLaVA

**Recommended**:
```yaml
stage_1:
  vision_encoder: freeze  # CLIP ViT-L
  projection: train
  llm: freeze

stage_2:
  vision_encoder: freeze  # Keep frozen
  projection: train
  llm: train (LoRA)
```

**Exception - High-res LLaVA**:
```yaml
# For 768x768+ images, consider unfreezing last 4 layers
vision_encoder[:-4]: freeze
vision_encoder[-4:]: train  # Adapt to higher resolution
```

### InternVL

**Recommended**:
```yaml
stage_1:
  vision_encoder: train  # InternViT co-trained
  resampler: train
  llm: freeze

stage_2:
  vision_encoder: train  # Continue joint training
  resampler: train
  llm: train (LoRA)
```

**Reasoning**: InternVL philosophy is joint vision-language training

### CogVLM

**Recommended**:
```yaml
stage_1:
  vision_encoder: freeze
  vision_expert: train   # Parallel vision branch
  language_expert: train

stage_2:
  vision_encoder: freeze
  vision_expert: train
  language_expert: train
```

**Reasoning**: MoE architecture with separate experts

## Domain-Specific Guidelines

### Natural Images → Natural Images

**Scenario**: COCO, ImageNet, general photos

**Recommendation**: **Freeze vision encoder**
- CLIP features highly transferable
- Save compute for LLM training

```yaml
frozen:
  vision_encoder: true
```

### Natural Images → Documents/OCR

**Scenario**: PDFs, receipts, charts, diagrams

**Recommendation**: **Unfreeze last 6-8 layers**
- CLIP trained on natural images
- Text patterns need adaptation

```yaml
frozen:
  vision_encoder[:16]: true   # ViT-L has 24 layers
  vision_encoder[16:]: false  # Adapt last 8 layers
```

### Natural Images → Medical/Scientific

**Scenario**: X-rays, microscopy, MRI, satellite

**Recommendation**: **Full unfreezing or domain-specific pretrained encoder**
- Large domain gap from CLIP training
- Consider using domain-pretrained encoders (RadImageNet, SatMAE)

```yaml
frozen:
  vision_encoder: false  # Full adaptation needed
```

### Natural Images → Video

**Scenario**: Video understanding tasks

**Recommendation**: **Freeze spatial, train temporal**
- CLIP spatial features work well
- Add trainable temporal attention

```python
for layer in vision_encoder.spatial_layers:
    layer.requires_grad_(False)

for layer in vision_encoder.temporal_layers:
    layer.requires_grad_(True)
```

## Compute and Memory Analysis

### Training Cost Comparison (ViT-L/14 example)

| Strategy | GPU Hours (8xA100) | Memory per GPU | Final Accuracy |
|----------|-------------------|----------------|----------------|
| Fully frozen | 100h (baseline) | 24GB | 78.5% (VQAv2) |
| Last 4 layers | 140h (+40%) | 28GB | 79.8% (+1.3%) |
| Last 8 layers | 180h (+80%) | 32GB | 80.2% (+1.7%) |
| Fully unfrozen | 250h (+150%) | 40GB | 80.5% (+2.0%) |

**Cost-benefit analysis**: Last 4 layers offers best trade-off for most use cases

### Memory Breakdown

```python
# ViT-L/14 (307M parameters)
memory_frozen = {
    'model_weights': 1.2GB,       # FP32 weights
    'activations': 4GB,           # Forward pass
    'total': 5.2GB
}

memory_unfrozen = {
    'model_weights': 1.2GB,
    'activations': 4GB,
    'gradients': 1.2GB,           # Backward pass
    'optimizer_state': 2.4GB,     # AdamW momentum + variance
    'total': 8.8GB                # 70% more memory
}
```

## Debugging Frozen Encoder Issues

### Issue 1: Poor Performance with Frozen Encoder

**Symptoms**: Model accuracy plateaus below expectations

**Diagnosis**:
```python
# Check CLIP similarity of your data
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Compute similarity on your dataset
similarities = []
for image, text in dataset:
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = clip(**inputs)
    sim = outputs.logits_per_image.item()
    similarities.append(sim)

avg_sim = np.mean(similarities)
print(f"Average CLIP similarity: {avg_sim:.3f}")

# If avg_sim < 0.25: Consider unfreezing
# If avg_sim > 0.30: Frozen should work fine
```

**Solutions**:
- If CLIP similarity < 0.25: Unfreeze last 6-8 layers
- If dataset is very different: Use domain-specific pretrained encoder
- If still poor: Full unfreezing needed

### Issue 2: Gradient Flow Problems

**Symptoms**: Loss doesn't decrease, gradients vanish

**Check gradient flow**:
```python
# Monitor gradients through projection layer
def check_gradient_flow(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm:.6f}")

# If projection layer has tiny gradients (< 1e-6):
# Vision encoder features might be poorly scaled
```

**Solution**:
```python
# Add learnable scaling factor
class ScaledProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.proj(self.scale * x)
```

### Issue 3: Overfitting to Projection Layer

**Symptoms**: Training loss decreases but validation stagnates

**Solution**:
```python
# Add dropout to projection
projection = nn.Sequential(
    nn.Linear(vision_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(0.1),  # Add regularization
    nn.Linear(hidden_dim, llm_dim)
)
```

## Best Practices Summary

### ✅ DO

1. **Freeze by default in Stages 1-2** unless you have specific reasons not to
2. **Keep frozen in Stage 3** to preserve learned alignment
3. **Use partial freezing** (last N layers) for domain adaptation
4. **Monitor CLIP similarity** to decide freezing strategy
5. **Consider compute budget** - frozen = 2-3x faster training

### ❌ DON'T

1. **Don't unfreeze without good reason** - wastes compute for minimal gain
2. **Don't unfreeze in Stage 3** unless domain-specific and have large dataset
3. **Don't forget to freeze after Stage 2** - catastrophic forgetting risk
4. **Don't use same LR for frozen and unfrozen** - use discriminative LRs
5. **Don't train vision encoder with high LR** - use 10-100x lower than projection

## Quick Decision Guide

```
START
  ↓
Is vision encoder CLIP/SigLIP?
  ├─ Yes → FREEZE (go to Q2)
  └─ No → UNFREEZE (use lower LR 1e-6)
  ↓
Q2: Is domain shift large? (medical, satellite, documents)
  ├─ Yes → UNFREEZE last 6-8 layers
  └─ No → KEEP FROZEN (go to Q3)
  ↓
Q3: Have large compute budget (100+ GPU days)?
  ├─ Yes → Consider full unfreezing for +1-2% accuracy
  └─ No → KEEP FROZEN (most efficient)
  ↓
Q4: In Stage 3 (instruction tuning)?
  ├─ Yes → MUST FREEZE (prevent catastrophic forgetting)
  └─ No → Follow Q1-Q3 decisions
```

---

**Sources:**
- Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training"
- Liu, H. et al. (2024). "Visual Instruction Tuning"
- Chen, Z. et al. (2024). "InternVL: Scaling up Vision Foundation Models"
- Wang, W. et al. (2023). "CogVLM: Visual Expert for Pretrained Language Models"
- Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
