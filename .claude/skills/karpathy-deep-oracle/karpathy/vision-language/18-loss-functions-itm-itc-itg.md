# VLM Loss Functions: ITM, ITC, ITG

Deep dive into the three fundamental vision-language pre-training objectives: Image-Text Matching (ITM), Image-Text Contrastive (ITC), and Image-grounded Text Generation (ITG), with implementations and training strategies.

## Overview

Modern VLMs use combinations of three complementary objectives:

**ITC (Image-Text Contrastive)**: Align global image-text representations via contrastive learning
**ITM (Image-Text Matching)**: Binary classification - do image and text match?
**ITG (Image-grounded Text Generation)**: Generate text conditioned on image (captioning)

**Used by**: CLIP (ITC only), BLIP/BLIP-2 (all three), ALBEF (ITC+ITM), Florence (ITC)

## 1. Image-Text Contrastive (ITC)

### Concept

Learn a shared embedding space where matching image-text pairs are close, non-matching pairs are far.

**Architecture**:
```
Image → Vision Encoder → image_emb [D]
Text  → Text Encoder   → text_emb [D]

similarity = cosine(image_emb, text_emb)
```

### InfoNCE Loss Implementation

```python
def itc_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Image-Text Contrastive loss (CLIP-style)

    Args:
        image_embeds: [B, D] normalized image embeddings
        text_embeds: [B, D] normalized text embeddings
        temperature: scaling factor for logits
    """
    batch_size = image_embeds.shape[0]

    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature  # [B, B]

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=logits.device)

    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)  # Image → Text
    loss_t2i = F.cross_entropy(logits.T, labels)  # Text → Image

    loss = (loss_i2t + loss_t2i) / 2
    return loss
```

### Key Hyperparameters

**Temperature τ**: Controls how "peaked" the distribution is
- **Low τ (0.01-0.05)**: Sharp distribution, confident predictions (harder training)
- **High τ (0.1-0.2)**: Smooth distribution, less confident (easier training)
- **Standard**: 0.07 (CLIP), 0.05 (BLIP-2)

```python
# Learnable temperature (better than fixed)
class ITCLoss(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

    def forward(self, image_embeds, text_embeds):
        temperature = self.temp.exp()
        # ... rest of ITC loss
```

**Batch size**: Critical for ITC!
- Small batch (256): Only 256 negatives per sample
- Large batch (4096): 4096 negatives → better discrimination
- **CLIP**: 32k batch size across GPUs
- **BLIP-2**: 2320 (Stage 1)

### Hard Negative Mining

Improve ITC by focusing on hard negatives:

```python
def itc_loss_with_hard_negatives(image_embeds, text_embeds, temperature=0.07, hard_ratio=0.5):
    batch_size = image_embeds.shape[0]

    # Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Similarity matrix
    sim_matrix = torch.matmul(image_embeds, text_embeds.T) / temperature

    # Get hard negatives (high similarity but wrong match)
    with torch.no_grad():
        # For each image, find hardest negative texts
        sim_i2t = sim_matrix.clone()
        sim_i2t.fill_diagonal_(-float('inf'))  # Mask positive pairs
        hard_neg_idx = sim_i2t.topk(k=int(batch_size * hard_ratio), dim=-1).indices

    # Standard ITC + weighted hard negatives
    labels = torch.arange(batch_size, device=sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)

    # Add hard negative term
    hard_neg_sim = sim_matrix.gather(1, hard_neg_idx)
    hard_neg_loss = -torch.log(1 - torch.sigmoid(hard_neg_sim)).mean()

    return loss + 0.1 * hard_neg_loss  # Weight hard negative loss
```

## 2. Image-Text Matching (ITM)

### Concept

Binary classification: Does this image-text pair match?

**Architecture**:
```
Image features [B, N, D] ──┐
                          ├─→ Cross-Encoder → Binary Classifier → {0, 1}
Text features [B, L, D] ───┘
```

### Implementation

```python
class ITMHead(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification

    def forward(self, multimodal_embeds):
        # multimodal_embeds: [B, D] fused image-text representation
        logits = self.fc(multimodal_embeds)  # [B, 2]
        return logits

def itm_loss(model, images, texts, match_labels):
    """
    Image-Text Matching loss

    Args:
        images: [B, 3, H, W]
        texts: [B, L] token IDs
        match_labels: [B] binary labels (1=match, 0=no match)
    """
    # Extract and fuse features
    multimodal_embeds = model.encode_multimodal(images, texts)  # [B, D]

    # Binary classification
    logits = model.itm_head(multimodal_embeds)  # [B, 2]

    # Cross-entropy loss
    loss = F.cross_entropy(logits, match_labels)
    return loss
```

### Hard Negative Sampling (Critical for ITM)

Random negatives are too easy; use hard negatives from ITC:

```python
def create_itm_batch(images, texts, image_embeds, text_embeds):
    """
    Create ITM training batch with hard negatives

    Returns:
        images: [2B, ...] positive + hard negative images
        texts: [2B, ...] positive + hard negative texts
        labels: [2B] match labels
    """
    batch_size = images.shape[0]

    # Positive pairs (original batch)
    pos_images = images
    pos_texts = texts
    pos_labels = torch.ones(batch_size)

    # Hard negative pairs (high ITC similarity but wrong match)
    with torch.no_grad():
        sim_matrix = torch.matmul(
            F.normalize(image_embeds, dim=-1),
            F.normalize(text_embeds, dim=-1).T
        )
        sim_matrix.fill_diagonal_(-float('inf'))

        # Sample hard negatives (top-k similar but wrong)
        hard_neg_idx = sim_matrix.topk(k=1, dim=-1).indices.squeeze(-1)

    neg_images = images
    neg_texts = texts[hard_neg_idx]  # Mismatch text for each image
    neg_labels = torch.zeros(batch_size)

    # Combine positive and negative
    all_images = torch.cat([pos_images, neg_images], dim=0)
    all_texts = torch.cat([pos_texts, neg_texts], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    return all_images, all_texts, all_labels
```

## 3. Image-grounded Text Generation (ITG)

### Concept

Language modeling conditioned on image: P(text | image)

**Architecture**:
```
Image → Vision Encoder → image_features
Text  → Decoder (causal LM)

logits = Decoder(text_tokens, image_features)
loss = CrossEntropy(logits, target_tokens)
```

### Implementation

```python
def itg_loss(model, images, caption_tokens):
    """
    Image-grounded Text Generation (captioning) loss

    Args:
        images: [B, 3, H, W]
        caption_tokens: [B, L] caption token IDs
    """
    # Encode image
    image_features = model.vision_encoder(images)  # [B, N, D]

    # Generate caption with image conditioning
    # Shift tokens for causal LM
    input_ids = caption_tokens[:, :-1]  # Remove last token
    labels = caption_tokens[:, 1:]      # Remove first token (BOS)

    # Forward through decoder
    outputs = model.text_decoder(
        input_ids=input_ids,
        encoder_hidden_states=image_features,  # Cross-attention
        labels=labels
    )

    # Language modeling loss (ignoring padding)
    loss = outputs.loss
    return loss
```

### Label Smoothing (Important for ITG)

```python
def itg_loss_with_smoothing(logits, labels, smoothing=0.1, ignore_index=-100):
    """
    ITG loss with label smoothing

    Args:
        logits: [B, L, V] predicted logits (V = vocab size)
        labels: [B, L] target token IDs
        smoothing: label smoothing factor
    """
    vocab_size = logits.size(-1)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]

    # Create smoothed targets
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (vocab_size - 1))

        # Set true label probability
        true_dist.scatter_(2, labels.unsqueeze(-1), 1.0 - smoothing)

        # Mask padding
        true_dist[labels == ignore_index] = 0

    # Compute loss
    loss = -(true_dist * log_probs).sum(dim=-1)
    loss = loss[labels != ignore_index].mean()

    return loss
```

## Combined Training (BLIP-2 Style)

### Multi-Objective Training

```python
class VLMWithThreeObjectives(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ViT()
        self.q_former = QFormer()  # BLIP-2 Q-Former
        self.text_decoder = GPTDecoder()

        self.itc_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts, mode='all'):
        # Extract features
        image_feats = self.vision_encoder(images)  # [B, 197, 768]
        text_feats = self.q_former.text_encoder(texts)  # [B, L, 768]

        # Q-Former processes image features
        query_output = self.q_former(image_feats)  # [B, 32, 768]

        losses = {}

        # 1. ITC Loss
        if mode in ['itc', 'all']:
            image_embeds = query_output.mean(dim=1)  # [B, 768]
            text_embeds = text_feats[:, 0, :]  # [B, 768] CLS token
            losses['itc'] = itc_loss(image_embeds, text_embeds, self.itc_temp.exp())

        # 2. ITM Loss
        if mode in ['itm', 'all']:
            # Create hard negatives
            itm_images, itm_texts, itm_labels = create_itm_batch(
                images, texts, image_embeds, text_embeds
            )
            multimodal = self.q_former.cross_encode(itm_images, itm_texts)
            losses['itm'] = itm_loss(multimodal, itm_labels)

        # 3. ITG Loss
        if mode in ['itg', 'all']:
            losses['itg'] = itg_loss(self.text_decoder, query_output, texts)

        return losses

# Training step
def train_step(model, batch):
    images, texts = batch['images'], batch['texts']

    losses = model(images, texts, mode='all')

    # Weighted combination
    total_loss = (
        1.0 * losses['itc'] +
        1.0 * losses['itm'] +
        1.0 * losses['itg']
    )

    return total_loss, losses
```

### Loss Balancing Strategies

**Fixed weights**:
```python
loss = 1.0 * L_itc + 1.0 * L_itm + 1.0 * L_itg
```

**Dynamic weighting** (uncertainty-based):
```python
class DynamicLossWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))  # Learn loss weights

    def forward(self, losses):
        # losses: {'itc': ..., 'itm': ..., 'itg': ...}
        precision = torch.exp(-self.log_vars)

        total = (
            precision[0] * losses['itc'] + self.log_vars[0] +
            precision[1] * losses['itm'] + self.log_vars[1] +
            precision[2] * losses['itg'] + self.log_vars[2]
        )
        return total
```

**Curriculum** (stage-based):
```python
def get_loss_weights(epoch, total_epochs):
    if epoch < total_epochs * 0.3:
        # Early: Focus on alignment (ITC)
        return {'itc': 2.0, 'itm': 1.0, 'itg': 0.5}
    elif epoch < total_epochs * 0.7:
        # Mid: Balanced
        return {'itc': 1.0, 'itm': 1.0, 'itg': 1.0}
    else:
        # Late: Focus on generation (ITG)
        return {'itc': 0.5, 'itm': 0.5, 'itg': 2.0}
```

## Training Best Practices

### Stage 1: ITC-only (Alignment)

```yaml
duration: 50k steps
objectives: ITC only
batch_size: 2048
learning_rate: 3e-4
purpose: Learn shared embedding space
```

### Stage 2: ITC + ITM (Refined Alignment)

```yaml
duration: 100k steps
objectives: ITC + ITM
batch_size: 2048
loss_weights: [1.0, 1.0]
learning_rate: 1e-4
purpose: Hard negative mining, better discrimination
```

### Stage 3: All Objectives (Generative Capabilities)

```yaml
duration: 200k steps
objectives: ITC + ITM + ITG
batch_size: 1024  # Smaller due to decoder
loss_weights: [0.5, 0.5, 2.0]  # Emphasize generation
learning_rate: 5e-5
purpose: Caption generation, full multimodal understanding
```

## Key Takeaways

- **ITC**: Global alignment, requires large batch sizes (1024+)
- **ITM**: Fine-grained matching, MUST use hard negatives
- **ITG**: Generation capability, use label smoothing (0.1)
- **Combined**: Start ITC-only, gradually add ITM/ITG
- **Loss weights**: Equal weights (1:1:1) work well, curriculum helps

---

**Sources:**
- Radford, A. et al. (2021). "Learning Transferable Visual Models" (CLIP)
- Li, J. et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training"
- Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training"
- Li, J. et al. (2021). "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation" (ALBEF)
