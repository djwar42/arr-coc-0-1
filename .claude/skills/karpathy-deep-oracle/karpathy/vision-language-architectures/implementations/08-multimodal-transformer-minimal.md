# Minimal Multimodal Transformer Architecture

## Overview

A minimal multimodal transformer combines vision and language processing in a single, unified architecture. Unlike dual-encoder approaches (like CLIP), minimal multimodal transformers use shared transformer layers to process both modalities jointly, enabling rich cross-modal interactions through self-attention.

**Key characteristics:**
- **Shared transformer backbone** - Same layers process vision + text
- **Modality-specific embeddings** - Separate embedding layers for images and text
- **Joint attention** - Vision and text tokens attend to each other
- **Simple pre-training objectives** - MLM, ITM, ITC for self-supervised learning
- **Minimal complexity** - Educational architecture, not state-of-the-art scale

This architecture is inspired by BERT's success in NLP, adapted for multimodal learning where an image is treated as a sequence of visual tokens that can be processed alongside text tokens.

From [RobinDong/tiny_multimodal](https://github.com/RobinDong/tiny_multimodal):
> "A simple and 'tiny' implementation of many multimodal models. It supports training/finetuning/deploying these tiny-sized models."

## Architecture Components

### 1. Modality Embeddings

**Vision Embeddings:**

```python
import torch
import torch.nn as nn
import math

class PatchEmbeddings(nn.Module):
    """
    Convert image into patches and project to embedding space.
    Similar to Vision Transformer (ViT) patch embedding.
    """
    def __init__(self, image_size=224, patch_size=16, num_channels=3, hidden_size=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional projection: non-overlapping patches
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        # -> (batch_size, hidden_size, num_patches_h, num_patches_w)
        x = self.projection(x)

        # Flatten spatial dimensions
        # -> (batch_size, hidden_size, num_patches)
        x = x.flatten(2)

        # Transpose to sequence format
        # -> (batch_size, num_patches, hidden_size)
        x = x.transpose(1, 2)
        return x


class ImageEmbeddings(nn.Module):
    """
    Complete image embedding with [CLS] token and position embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            config['image_size'],
            config['patch_size'],
            config['num_channels'],
            config['hidden_size']
        )

        # Learnable [CLS] token for image classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['hidden_size']))

        # Position embeddings for each patch + [CLS]
        num_positions = self.patch_embeddings.num_patches + 1
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_positions, config['hidden_size'])
        )

        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]

        # Get patch embeddings
        embeddings = self.patch_embeddings(pixel_values)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
```

From [Implementing Vision Transformer (ViT) from Scratch](https://towardsdatascience.com/implementing-vision-transformer-vit-from-scratch-3e192c6155f0/):
> "The input image is split into small patches, which are then flattened into sequences of vectors. These vectors are then processed by a transformer encoder."

**Text Embeddings:**

```python
class TextEmbeddings(nn.Module):
    """
    Standard BERT-style text embeddings with token, position, and type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        max_position_embeddings = config['max_position_embeddings']

        # Token embeddings (vocabulary)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # Token type embeddings (for text only, usually 0)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_length = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Token type embeddings (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
```

### 2. Joint Attention Mechanism

The core innovation is processing vision and text tokens together in shared transformer layers:

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention that processes both vision and text tokens jointly.
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def transpose_for_scores(self, x):
        # (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_size)
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        # Final projection
        output = self.output_projection(context_layer)
        return output
```

**Key insight:** When we concatenate vision and text tokens into a single sequence, the attention mechanism naturally learns cross-modal interactions. Vision tokens can attend to text tokens and vice versa, enabling the model to ground language in visual content.

### 3. Transformer Layer

```python
class TransformerLayer(nn.Module):
    """
    Single transformer layer with self-attention and feed-forward network.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.layer_norm1 = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)

        # Feed-forward network with residual connection
        intermediate_output = nn.functional.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.layer_norm2(hidden_states + layer_output)

        return hidden_states


class TransformerEncoder(nn.Module):
    """
    Stack of transformer layers that process multimodal sequences.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config['num_hidden_layers'])
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
```

### 4. Complete Multimodal Model

```python
class MinimalMultimodalTransformer(nn.Module):
    """
    Minimal multimodal transformer that processes vision and language jointly.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Modality-specific embeddings
        self.image_embeddings = ImageEmbeddings(config)
        self.text_embeddings = TextEmbeddings(config)

        # Shared transformer encoder
        self.encoder = TransformerEncoder(config)

        # Modality type embeddings to distinguish vision vs text
        self.modality_type_embeddings = nn.Embedding(2, config['hidden_size'])

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Forward pass with optional vision and/or text inputs.

        Args:
            pixel_values: (batch_size, channels, height, width) - Images
            input_ids: (batch_size, seq_length) - Text token IDs
            attention_mask: (batch_size, total_seq_length) - Attention mask
        """
        embeddings_list = []
        modality_ids_list = []

        # Process vision if provided
        if pixel_values is not None:
            vision_embeddings = self.image_embeddings(pixel_values)
            batch_size, num_patches, _ = vision_embeddings.shape

            # Add vision modality type embedding (type_id = 0)
            vision_type_ids = torch.zeros(
                batch_size, num_patches,
                dtype=torch.long,
                device=pixel_values.device
            )
            vision_modality_embeddings = self.modality_type_embeddings(vision_type_ids)
            vision_embeddings = vision_embeddings + vision_modality_embeddings

            embeddings_list.append(vision_embeddings)
            modality_ids_list.append(vision_type_ids)

        # Process text if provided
        if input_ids is not None:
            text_embeddings = self.text_embeddings(input_ids)
            batch_size, seq_length, _ = text_embeddings.shape

            # Add text modality type embedding (type_id = 1)
            text_type_ids = torch.ones(
                batch_size, seq_length,
                dtype=torch.long,
                device=input_ids.device
            )
            text_modality_embeddings = self.modality_type_embeddings(text_type_ids)
            text_embeddings = text_embeddings + text_modality_embeddings

            embeddings_list.append(text_embeddings)
            modality_ids_list.append(text_type_ids)

        # Concatenate vision and text embeddings
        multimodal_embeddings = torch.cat(embeddings_list, dim=1)

        # Prepare attention mask
        if attention_mask is None:
            total_length = multimodal_embeddings.shape[1]
            attention_mask = torch.ones(batch_size, total_length, device=multimodal_embeddings.device)

        # Convert attention mask to attention bias
        # (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Pass through transformer encoder
        encoder_output = self.encoder(multimodal_embeddings, extended_attention_mask)

        return encoder_output
```

## Pre-training Objectives

Three complementary objectives are commonly used for pre-training minimal multimodal transformers:

### 1. Image-Text Contrastive Learning (ITC)

Learn aligned representations by maximizing similarity between matched image-text pairs:

```python
class ITCLoss(nn.Module):
    """
    Image-Text Contrastive (ITC) learning objective.
    Aligns vision and text representations in a shared embedding space.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        self.vision_projection = nn.Linear(hidden_size, config['projection_dim'])
        self.text_projection = nn.Linear(hidden_size, config['projection_dim'])
        self.temperature = nn.Parameter(torch.ones([]) * config['temperature'])

    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (batch_size, hidden_size) - [CLS] token from vision
            text_features: (batch_size, hidden_size) - [CLS] token from text
        """
        # Project to shared embedding space
        vision_embeds = self.vision_projection(vision_features)
        text_embeds = self.text_projection(text_features)

        # Normalize
        vision_embeds = nn.functional.normalize(vision_embeds, dim=-1)
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(vision_embeds, text_embeds.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = vision_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Bidirectional contrastive loss
        loss_v2t = nn.functional.cross_entropy(logits, labels)
        loss_t2v = nn.functional.cross_entropy(logits.t(), labels)

        return (loss_v2t + loss_t2v) / 2
```

From [Align before Fuse](https://arxiv.org/pdf/2107.07651):
> "We pre-train ALBEF with three objectives: image-text contrastive learning (ITC) on the unimodal encoders, masked language modeling (MLM) and image-text matching (ITM)."

### 2. Masked Language Modeling (MLM)

Predict masked text tokens using both text context and visual information:

```python
class MLMLoss(nn.Module):
    """
    Masked Language Modeling objective for multimodal learning.
    Predicts masked text tokens using vision + text context.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size']

        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, text_hidden_states, labels, masked_positions):
        """
        Args:
            text_hidden_states: (batch_size, text_seq_length, hidden_size)
            labels: (batch_size, text_seq_length) - Original token IDs
            masked_positions: (batch_size, text_seq_length) - Boolean mask of masked positions
        """
        # Get hidden states for masked positions only
        masked_hidden = text_hidden_states[masked_positions]

        # Predict token logits
        prediction_scores = self.mlm_head(masked_hidden)

        # Compute loss only for masked tokens
        masked_labels = labels[masked_positions]
        loss = nn.functional.cross_entropy(prediction_scores, masked_labels)

        return loss


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Prepare masked input for MLM.

    Strategy:
    - 80% of time: replace with [MASK]
    - 10% of time: replace with random token
    - 10% of time: keep original token (but still predict it)
    """
    labels = input_ids.clone()

    # Create random mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Don't mask special tokens
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special_tokens_mask |= (labels == token_id)
    masked_indices &= ~special_tokens_mask

    # 80%: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10%: replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10%: keep original (no replacement)

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    return input_ids, labels, masked_indices
```

### 3. Image-Text Matching (ITM)

Binary classification to predict whether an image-text pair is matched or mismatched:

```python
class ITMLoss(nn.Module):
    """
    Image-Text Matching objective.
    Predicts whether image and text are semantically matched.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']

        self.itm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)  # Binary: matched or not
        )

    def forward(self, multimodal_cls_token, labels):
        """
        Args:
            multimodal_cls_token: (batch_size, hidden_size) - [CLS] from joint encoding
            labels: (batch_size,) - 1 for matched, 0 for mismatched
        """
        logits = self.itm_head(multimodal_cls_token)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


def create_negative_samples(image_batch, text_batch, negative_ratio=0.5):
    """
    Create negative image-text pairs for ITM by shuffling.

    Args:
        image_batch: Original images
        text_batch: Original texts (matched with images)
        negative_ratio: Fraction of batch to make into negative pairs

    Returns:
        images, texts, labels (1=matched, 0=mismatched)
    """
    batch_size = len(image_batch)
    num_negative = int(batch_size * negative_ratio)

    # Create labels: first portion is positive, rest is negative
    labels = torch.ones(batch_size, dtype=torch.long)
    labels[batch_size - num_negative:] = 0

    # Shuffle text for negative samples
    negative_indices = torch.randperm(num_negative)
    text_batch_shuffled = text_batch.clone()
    text_batch_shuffled[batch_size - num_negative:] = text_batch[negative_indices]

    return image_batch, text_batch_shuffled, labels
```

### Combined Pre-training

```python
def pretrain_step(model, batch, itc_loss_fn, mlm_loss_fn, itm_loss_fn, optimizer):
    """
    Single pre-training step combining all three objectives.
    """
    images = batch['pixel_values']
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # 1. Image-Text Contrastive (ITC)
    # Forward pass for vision and text separately first
    vision_output = model(pixel_values=images)
    text_output = model(input_ids=input_ids, attention_mask=attention_mask)

    vision_cls = vision_output[:, 0, :]  # [CLS] token for vision
    text_cls = text_output[:, 0, :]      # [CLS] token for text

    itc_loss = itc_loss_fn(vision_cls, text_cls)

    # 2. Masked Language Modeling (MLM)
    # Mask text tokens
    masked_input_ids, mlm_labels, masked_positions = mask_tokens(input_ids, tokenizer)

    # Forward with masked text + images
    multimodal_output = model(
        pixel_values=images,
        input_ids=masked_input_ids,
        attention_mask=attention_mask
    )

    # Extract text hidden states (after vision tokens)
    num_vision_tokens = images.shape[0] * ((images.shape[-1] // 16) ** 2 + 1)
    text_hidden = multimodal_output[:, num_vision_tokens:, :]

    mlm_loss = mlm_loss_fn(text_hidden, mlm_labels, masked_positions)

    # 3. Image-Text Matching (ITM)
    # Create negative samples
    images_itm, input_ids_itm, itm_labels = create_negative_samples(images, input_ids)

    # Forward with positive + negative pairs
    multimodal_output_itm = model(
        pixel_values=images_itm,
        input_ids=input_ids_itm,
        attention_mask=attention_mask
    )

    itm_cls = multimodal_output_itm[:, 0, :]  # Multimodal [CLS]
    itm_loss = itm_loss_fn(itm_cls, itm_labels)

    # Combined loss
    total_loss = itc_loss + mlm_loss + itm_loss

    # Backward and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'itc_loss': itc_loss.item(),
        'mlm_loss': mlm_loss.item(),
        'itm_loss': itm_loss.item()
    }
```

## Training Configuration

Minimal configuration for educational/experimental purposes:

```python
config = {
    # Image settings
    'image_size': 224,
    'patch_size': 16,
    'num_channels': 3,

    # Model architecture
    'hidden_size': 384,              # Smaller than BERT-base (768)
    'num_hidden_layers': 6,          # Half of BERT-base (12)
    'num_attention_heads': 6,
    'intermediate_size': 1536,       # 4 * hidden_size

    # Text settings
    'vocab_size': 30522,             # BERT vocabulary
    'max_position_embeddings': 512,

    # Dropout
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,

    # Pre-training
    'projection_dim': 256,           # For ITC loss
    'temperature': 0.07,             # For contrastive learning

    # Training
    'batch_size': 256,
    'learning_rate': 1e-4,
    'warmup_steps': 10000,
    'max_steps': 1000000,
}

# Initialize model
model = MinimalMultimodalTransformer(config)

# Initialize losses
itc_loss = ITCLoss(config)
mlm_loss = MLMLoss(config)
itm_loss = ITMLoss(config)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

## Comparison: Minimal vs Production Models

| Aspect | Minimal (This Guide) | BERT-base | ALBEF | BLIP-2 |
|--------|---------------------|-----------|-------|---------|
| Hidden Size | 384 | 768 | 768 | 768 |
| Layers | 6 | 12 | 12 | 12 |
| Heads | 6 | 12 | 12 | 12 |
| Parameters | ~30M | ~110M | ~220M | ~3.4B |
| Pre-training Data | Conceptual Captions 3M | BookCorpus + Wikipedia | 14M images | 129M images |
| Training Time | ~1 week (1 GPU) | ~1 month (multi-GPU) | ~2 weeks (multi-GPU) | ~weeks (TPU) |

**The minimal architecture can be trained on:**
- Single GPU (RTX 3080 Ti or similar)
- Conceptual Captions 3M or COCO Captions
- 1-2 weeks for reasonable convergence

## Key Insights

**1. Shared vs Dual Encoders:**

Minimal multimodal transformers use **shared layers** for both modalities, unlike CLIP's dual-encoder approach. This enables:
- Richer cross-modal interactions (vision attends to text directly)
- More parameter-efficient (one encoder instead of two)
- Better for tasks requiring deep multimodal fusion (VQA, captioning)

**2. Modality Type Embeddings:**

Critical for the model to distinguish vision vs text tokens:
```python
# Without modality embeddings: model is confused
vision_tokens = [v1, v2, v3, ...]
text_tokens = [t1, t2, t3, ...]

# With modality embeddings: model knows what's what
vision_tokens = [v1+type0, v2+type0, v3+type0, ...]
text_tokens = [t1+type1, t2+type1, t3+type1, ...]
```

**3. Pre-training Objective Synergy:**

- **ITC**: Aligns vision and text in embedding space (coarse alignment)
- **ITM**: Forces deep interaction through joint encoding (fine-grained alignment)
- **MLM**: Enables language understanding with visual grounding

Each objective complements the others, creating robust multimodal representations.

## Practical Tips

**1. Start Small:**
- Begin with hidden_size=256, 4 layers to verify training works
- Scale up once pipeline is stable

**2. Monitor Attention Maps:**
```python
# Extract attention during evaluation
with torch.no_grad():
    output = model(images, text, return_attention=True)
    attention_maps = output['attention_maps']

# Visualize cross-modal attention
# How much do text tokens attend to image patches?
text_to_vision_attention = attention_maps[:, :, text_range, vision_range]
```

**3. Hard Negative Mining:**

For ITM loss, use hard negatives (similar but incorrect pairs) instead of random:
```python
# Find hardest negative in batch (highest similarity but wrong)
similarities = torch.matmul(vision_embeds, text_embeds.t())
hard_negative_idx = similarities.topk(k=2, dim=1)[1][:, 1]  # 2nd highest
```

**4. Gradient Checkpointing:**

Save memory for larger models:
```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, hidden_states):
    for layer in self.layers:
        hidden_states = checkpoint(layer, hidden_states)
    return hidden_states
```

## Sources

**Web Research:**

From [RobinDong/tiny_multimodal](https://github.com/RobinDong/tiny_multimodal) (accessed 2025-01-31):
- Simple implementation patterns for multimodal models
- Training on Conceptual Captions 12M dataset
- Minimal architecture suitable for single-GPU training

From [Implementing Vision Transformer (ViT) from Scratch](https://towardsdatascience.com/implementing-vision-transformer-vit-from-scratch-3e192c6155f0/) (accessed 2025-01-31):
- Patch embedding implementation details
- Position encoding strategies
- Transformer encoder architecture

From [Align before Fuse (ALBEF)](https://arxiv.org/pdf/2107.07651) (accessed 2025-01-31):
- ITC, MLM, ITM pre-training objectives
- Momentum distillation techniques
- Vision-language alignment strategies

**Search Results:**
- "site:github.com minimal multimodal transformer PyTorch implementation" (Google, accessed 2025-01-31)
- "simple multimodal transformer tutorial PyTorch BERT ViT" (Google, accessed 2025-01-31)
- "multimodal transformer pre-training PyTorch example MLM ITM" (Google, accessed 2025-01-31)
- "CLIP BERT multimodal transformer shared encoder architecture PyTorch" (Google, accessed 2025-01-31)
- "image text contrastive learning ITM ITC MLM pre-training objectives" (Google, accessed 2025-01-31)

**Additional References:**
- BERT paper: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- Vision Transformer (ViT): Dosovitskiy et al., "An Image is Worth 16x16 Words"
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
- BLIP-2: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders"

**Related Oracle Knowledge:**
- [00-minimal-vlm-pytorch.md](00-minimal-vlm-pytorch.md) - Foundational VLM implementation
- [01-qformer-blip2-implementation.md](01-qformer-blip2-implementation.md) - Query-based architecture
- [02-perceiver-cross-attention.md](02-perceiver-cross-attention.md) - Alternative attention patterns
- [07-fusion-strategies.md](07-fusion-strategies.md) - Early/mid/late fusion comparison
