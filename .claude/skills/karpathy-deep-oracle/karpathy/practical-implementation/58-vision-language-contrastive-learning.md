# Vision-Language Contrastive Learning Training

Complete training strategies for vision-language contrastive learning (CLIP-style) including InfoNCE loss, temperature tuning, negative sampling, and large-scale training recipes.

## Overview

**Contrastive learning** aligns vision and language by pulling positive image-text pairs together and pushing negative pairs apart in a shared embedding space. This is the foundation of CLIP, ALIGN, BLIP, and most modern vision-language models.

**Key Insight**: Large batch sizes = more in-batch negatives = better discrimination.

## InfoNCE Loss Formulation

### The Core Loss Function

```python
import torch
import torch.nn.functional as F

def info_nce_loss(image_embeds, text_embeds, temperature=0.07):
    """
    InfoNCE (Noise Contrastive Estimation) loss for vision-language alignment.

    Args:
        image_embeds: [B, D] normalized image embeddings
        text_embeds: [B, D] normalized text embeddings
        temperature: Temperature parameter τ (default 0.07)

    Returns:
        loss: Scalar loss value
    """
    batch_size = image_embeds.shape[0]

    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Compute similarity matrix: [B, B]
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=logits.device)

    # Symmetric loss: image->text + text->image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2

    return loss
```

**Mathematics**:

For image $i$ and text $t$:
```
similarity(i, t) = (image_embed(i) · text_embed(t)) / τ

P(t | i) = exp(similarity(i, t)) / Σ_k exp(similarity(i, t_k))

Loss = -log P(t_positive | i)
```

## Temperature Parameter τ

### Tuning Temperature

**Temperature controls the "sharpness" of the distribution**:

- **Low τ (0.01)**: Very confident, sharp distribution (harder learning)
- **High τ (0.5)**: Softer distribution (easier learning)
- **Optimal τ**: 0.05-0.07 for most VLM tasks

### Temperature as Learnable Parameter

**CLIP approach**: Initialize τ as learnable parameter

```python
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize as log(1/0.07) ≈ 2.66
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale.requires_grad = True

    def forward(self, image_embeds, text_embeds):
        # Clamp to prevent exp overflow
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # Equivalent to dividing by temperature
        logits = image_embeds @ text_embeds.T * logit_scale

        labels = torch.arange(len(logits), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2
```

**Why Learnable**: Model learns optimal temperature during training (typically converges to τ ≈ 0.05-0.07).

### Temperature Sweep Results

| τ Value | Zero-Shot Acc | Training Stability |
|---------|---------------|-------------------|
| 0.01    | 45.2%         | Unstable (gradients explode) |
| 0.03    | 58.7%         | Moderate |
| 0.05    | 63.4%         | Good |
| **0.07** | **65.1%**    | **Excellent** ✓ |
| 0.10    | 62.8%         | Good |
| 0.20    | 58.3%         | Too soft |

**Recommendation**: Start with τ=0.07 or make it learnable.

## Batch Size Effects

### Why Batch Size Matters

**In-batch negatives**: For batch size B, each sample has (B-1) negatives

```
Batch = 256  → 255 negatives per sample
Batch = 1024 → 1023 negatives per sample
Batch = 32768 (CLIP) → 32767 negatives! 🚀
```

**Larger batch = more diverse negatives = better discrimination**

### Training Recipe by Batch Size

#### Small Batch (256-512)

```yaml
batch_size: 512
learning_rate: 1e-3
warmup_steps: 2000
total_steps: 100k
optimizer: AdamW
weight_decay: 0.1

# Expect moderate performance
zero_shot_accuracy: ~55-60%
```

**Use case**: Limited GPUs (1-2), small datasets

#### Medium Batch (1024-2048)

```yaml
batch_size: 2048
learning_rate: 3e-3
warmup_steps: 5000
total_steps: 200k
optimizer: LAMB  # Better for large batch
weight_decay: 0.05

# Good performance
zero_shot_accuracy: ~63-67%
```

**Use case**: 4-8 GPUs, standard pre-training

#### Large Batch (8192-32768) - CLIP Scale

```yaml
batch_size: 32768
learning_rate: 5e-4
warmup_steps: 10000
total_steps: 400k
optimizer: LAMB
weight_decay: 0.2

# SOTA performance
zero_shot_accuracy: ~70-76%
```

**Use case**: 32+ GPUs, massive datasets (LAION-400M+)

**Scaling Rule**: Learning rate ∝ sqrt(batch_size)

## Negative Sampling Strategies

### 1. In-Batch Negatives (Standard)

**Default strategy**: All other samples in batch are negatives

```python
def in_batch_negatives(image_embeds, text_embeds):
    # All off-diagonal elements are negatives
    logits = image_embeds @ text_embeds.T / temperature

    # Diagonal = positives, off-diagonal = negatives
    return logits
```

**Pros**: Simple, efficient, no extra computation
**Cons**: Limited by batch size

### 2. Hard Negative Mining

**Strategy**: Select hardest negatives (highest similarity to positive)

```python
def hard_negative_mining(image_embeds, text_embeds, k=128):
    """
    Select k hardest negatives based on similarity scores.
    """
    # Compute all similarities
    similarities = image_embeds @ text_embeds.T  # [B, B]

    # For each positive pair, select k hardest negatives
    hard_neg_indices = []
    for i in range(len(similarities)):
        # Exclude positive (diagonal)
        sim = similarities[i].clone()
        sim[i] = -float('inf')

        # Get top-k most similar (hardest) negatives
        _, indices = torch.topk(sim, k)
        hard_neg_indices.append(indices)

    return hard_neg_indices
```

**Impact**: +2-5% accuracy improvement, slightly slower training

### 3. Momentum-Based Negatives (MoCo-style)

**Strategy**: Maintain queue of negatives from previous batches

```python
class MomentumQueue:
    def __init__(self, queue_size=65536, dim=512):
        self.queue = torch.randn(queue_size, dim)
        self.queue = F.normalize(self.queue, dim=1)
        self.queue_ptr = 0

    def dequeue_and_enqueue(self, keys):
        """Update queue with new keys."""
        batch_size = keys.shape[0]

        # Replace oldest keys with new ones
        self.queue[self.queue_ptr:self.queue_ptr + batch_size] = keys
        self.queue_ptr = (self.queue_ptr + batch_size) % len(self.queue)

    def get_negatives(self):
        return self.queue.clone()

# Usage
momentum_queue = MomentumQueue(queue_size=65536)
queue_negatives = momentum_queue.get_negatives()

# Compute contrastive loss with both in-batch and queue negatives
logits = torch.cat([
    image_embeds @ text_embeds.T,           # In-batch
    image_embeds @ queue_negatives.T        # Queue
], dim=1) / temperature
```

**Pros**: Huge negative pool (65k+), better discrimination
**Cons**: More complex, requires momentum encoder

### 4. Cross-Batch Negatives

**Strategy**: Gather negatives from all GPUs (distributed training)

```python
import torch.distributed as dist

def gather_from_all_gpus(tensor):
    """Gather tensors from all GPUs."""
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)

# Usage
all_image_embeds = gather_from_all_gpus(image_embeds)
all_text_embeds = gather_from_all_gpus(text_embeds)

# Now compute loss with global negatives
logits = image_embeds @ all_text_embeds.T / temperature
```

**Impact**: Effective batch size = local_batch × num_gpus

## Complete Training Recipe

### CLIP-Style Recipe (Recommended)

```yaml
# Model
vision_encoder: ViT-L/14 (or B/16 for smaller)
text_encoder: Transformer (12 layers, 512 dim)
embedding_dim: 512
temperature: 0.07 (learnable)

# Training
batch_size: 32768  # Across all GPUs
gradient_accumulation: 1
mixed_precision: bf16

# Optimizer: LAMB (for large batch)
optimizer: LAMB
learning_rate: 5e-4
weight_decay: 0.2
beta1: 0.9
beta2: 0.98
eps: 1e-6

# Schedule
warmup_steps: 10000  # Linear warmup
total_steps: 400000
lr_schedule: cosine_decay
min_lr: 0

# Gradient
gradient_clipping: 1.0  # Max norm

# Data
dataset: LAION-400M (or LAION-5B)
image_size: 224x224
augmentation:
  - random_resized_crop
  - random_horizontal_flip
  - color_jitter

# Hardware
gpus: 256x A100 (for batch 32768)
training_time: ~2 weeks
cost: ~$150k cloud (CLIP scale)
```

### Smaller Budget Recipe

```yaml
# Model
vision_encoder: ViT-B/16
text_encoder: Transformer (6 layers, 512 dim)
embedding_dim: 512
temperature: 0.07

# Training
batch_size: 2048  # More realistic
gradient_accumulation: 4
mixed_precision: bf16

# Optimizer: AdamW
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 0.1
beta1: 0.9
beta2: 0.999

# Schedule
warmup_steps: 5000
total_steps: 200000
lr_schedule: cosine_decay
min_lr: 1e-5

# Data
dataset: CC3M + CC12M (15M pairs)
image_size: 224x224

# Hardware
gpus: 8x A100
training_time: ~1 week
cost: ~$5k cloud
```

## Loss Function Variants

### 1. Symmetric Contrastive Loss (Standard)

```python
loss = (loss_image_to_text + loss_text_to_image) / 2
```

**Use case**: Equal importance to both directions

### 2. Asymmetric Loss (Text-Heavy)

```python
loss = 0.7 * loss_image_to_text + 0.3 * loss_text_to_image
```

**Use case**: Prioritize image→text retrieval (captions)

### 3. Soft Targets (Momentum Distillation)

**ALBEF approach**: Use momentum model to generate soft targets

```python
with torch.no_grad():
    momentum_logits = momentum_model(images, texts) / temperature
    soft_targets = F.softmax(momentum_logits, dim=-1)

# Student model learns from soft targets
student_logits = model(images, texts) / temperature
loss = F.kl_div(
    F.log_softmax(student_logits, dim=-1),
    soft_targets,
    reduction='batchmean'
)
```

**Impact**: Smoother training, +1-2% accuracy

## Training Stability Tips

### 1. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why**: Prevents exploding gradients from large batch sizes.

### 2. Warmup Schedule

```python
def linear_warmup_cosine_decay(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

**Why**: Warmup prevents early training instability.

### 3. Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16/BF16
        loss = contrastive_loss(images, texts)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Impact**: 2-3× speedup, half memory usage

## Dataset Requirements

### Minimum Dataset Size

| Model Size | Min Pairs | Recommended Pairs |
|------------|-----------|-------------------|
| Small (ViT-B, 6L text) | 1M | 3M |
| Medium (ViT-B, 12L text) | 3M | 15M |
| Large (ViT-L, 12L text) | 15M | 100M+ |
| CLIP-scale (ViT-L/14) | 100M | 400M-5B |

**Rule of Thumb**: Need 1000× more data than model parameters.

### Data Quality vs Quantity

**CLIP approach**: Massive noisy data (400M pairs from web)
**ALIGN approach**: Even larger, noisier (1.8B pairs)
**Filtered approach**: Smaller, cleaner (CC3M, CC12M)

**Trade-off**:
- More data (noisy) → Better zero-shot, worse fine-tuning
- Less data (clean) → Worse zero-shot, better fine-tuning

## Evaluation Metrics

### During Training

Track these metrics on validation set:

```python
# 1. Contrastive loss
val_loss = compute_contrastive_loss(val_images, val_texts)

# 2. Image-text retrieval
recall_at_1 = compute_recall(image_embeds, text_embeds, k=1)
recall_at_5 = compute_recall(image_embeds, text_embeds, k=5)
recall_at_10 = compute_recall(image_embeds, text_embeds, k=10)

# 3. Zero-shot classification (if available)
zeroshot_acc = evaluate_zeroshot(model, imagenet_val)
```

### Final Evaluation

**Benchmarks**:
- Zero-shot ImageNet accuracy
- Image-text retrieval (Flickr30k, COCO)
- Visual reasoning (VQA, NLVR2)
- Compositionality (Winoground)

## Common Issues & Solutions

### Issue: Training Loss Not Decreasing

**Causes**:
1. Temperature too low (<0.01)
2. Learning rate too high
3. Batch size too small (<256)

**Solutions**:
- Set τ=0.07 or make learnable
- Reduce LR by 10×
- Increase batch size or use cross-batch negatives

### Issue: Model Collapse (All Embeddings Similar)

**Symptoms**: Loss plateaus around 6-7, all similarities ~0.5

**Solutions**:
- Add weight decay (0.1-0.2)
- Use layer normalization on embeddings
- Verify embeddings are normalized before dot product
- Check for bugs in positive pair construction

### Issue: OOM During Training

**Solutions**:
- Reduce batch size, use gradient accumulation
- Enable activation checkpointing
- Use BF16 instead of FP32
- Reduce image resolution (224→128)
- Use queue-based negatives instead of cross-batch

## Advanced Techniques

### 1. Multi-Positive Contrastive Learning

**Scenario**: Multiple valid texts for one image

```python
# For each image, we have K positive texts
logits = image_embeds @ text_embeds.T / temperature  # [B, B*K]

# Mark all K positives for each image
labels = torch.arange(B).repeat_interleave(K)

loss = F.cross_entropy(logits, labels)
```

### 2. Hierarchical Contrastive Learning

**Strategy**: Contrastive loss at multiple scales

```python
# Patch-level contrastive
patch_loss = contrastive_loss(patch_embeds, word_embeds)

# Image-level contrastive
image_loss = contrastive_loss(image_embeds, sentence_embeds)

total_loss = 0.3 * patch_loss + 0.7 * image_loss
```

## References

**Papers**:
- CLIP: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- ALIGN: Jia et al. "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (2021)
- ALBEF: Li et al. "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation" (2021)

**Implementations**:
- OpenCLIP: https://github.com/mlfoundations/open_clip
- HuggingFace: transformers.CLIPModel
- LAVIS: Salesforce BLIP-2 (uses ITC loss in Stage 1)

**Datasets**:
- LAION-400M/5B: https://laion.ai/
- CC3M/CC12M: Google Conceptual Captions
- YFCC15M: Yahoo Flickr Creative Commons

## Summary

**Contrastive Learning TL;DR**:

```yaml
Core Loss: InfoNCE (pull positives, push negatives)
Temperature: 0.05-0.07 (or learnable)
Batch Size: Bigger = better (32k for SOTA)
Negatives: In-batch + hard negatives
Optimizer: AdamW (small) / LAMB (large batch)
LR Schedule: Linear warmup + cosine decay
Dataset: 100M+ pairs for strong zero-shot

Recipe:
  CLIP-scale: 256 GPUs, 400k steps, $150k, τ=0.07
  Budget: 8 GPUs, 200k steps, $5k, τ=0.07
```

Master contrastive learning → unlock powerful vision-language alignment! 🔥
