# Query-Conditioned Visual Attention Implementation

## Overview

Query-conditioned attention enables vision-language models to dynamically focus on image regions relevant to a text query (question, instruction, or prompt). Unlike standard vision transformers that process images independently of text, query-conditioned mechanisms embed question awareness directly into visual encoding.

**Core Principle**: The text query guides which visual features receive attention, creating query-aware visual representations optimized for answering specific questions about an image.

**Relevance to ARR-COC**: Query-conditioned attention is closely related to ARR-COC's relevance realization framework - both use text queries to dynamically allocate computational resources (attention/tokens) to relevant image regions.

From [Question aware vision transformer for multimodal reasoning](https://www.amazon.science/publications/question-aware-vision-transformer-for-multimodal-reasoning) (Amazon Science, accessed 2025-01-31):
- QA-ViT embeds question awareness directly within the vision encoder
- Results in dynamic visual features focusing on relevant image aspects to posed questions
- Model-agnostic approach applicable to various VL architectures

## Architecture Components

### 1. Query Embedding Module

**Purpose**: Convert text query into embedding space compatible with visual features.

**PyTorch Implementation**:

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class QueryEmbedder(nn.Module):
    """Embed text queries for visual attention conditioning."""

    def __init__(self, text_encoder='bert-base-uncased', hidden_dim=768, output_dim=512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.text_encoder = BertModel.from_pretrained(text_encoder)

        # Project text features to visual feature space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, questions):
        """
        Args:
            questions: List of text questions
        Returns:
            query_embeds: [B, output_dim] query embeddings
        """
        # Tokenize
        tokens = self.tokenizer(
            questions,
            padding=True,
            return_tensors='pt',
            max_length=64,
            truncation=True
        ).to(self.text_encoder.device)

        # Encode text
        with torch.no_grad():
            text_outputs = self.text_encoder(**tokens)

        # Use [CLS] token as query representation
        cls_embed = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Project to visual space
        query_embeds = self.projection(cls_embed)  # [B, output_dim]

        return query_embeds
```

**Key Design Choices**:
- Use pretrained language model (BERT/GPT) for robust text understanding
- Project text embeddings to match visual feature dimensionality
- [CLS] token provides global question representation
- Can also use mean pooling or last hidden state

### 2. Query-Conditioned Visual Attention

**Core Mechanism**: Use query embeddings as attention queries to weight visual features.

From [Visual-Question-Answering-using-Stacked-Attention-Networks](https://github.com/williamcfrancis/Visual-Question-Answering-using-Stacked-Attention-Networks) (GitHub, accessed 2025-01-31):
- Stacked attention layers enable multi-step reasoning
- First attention layer focuses broadly, second layer refines focus
- Point-wise multiplication fuses image and question embeddings

**PyTorch Implementation**:

```python
class QueryConditionedAttention(nn.Module):
    """Apply query-aware attention to visual features."""

    def __init__(self, visual_dim=512, query_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        assert visual_dim % num_heads == 0, "visual_dim must be divisible by num_heads"

        # Query-Key-Value projections
        self.q_proj = nn.Linear(query_dim, visual_dim)
        self.k_proj = nn.Linear(visual_dim, visual_dim)
        self.v_proj = nn.Linear(visual_dim, visual_dim)

        # Output projection
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, visual_features, query_embed, attention_mask=None):
        """
        Args:
            visual_features: [B, N, visual_dim] - patch features from vision encoder
            query_embed: [B, query_dim] - text query embedding
            attention_mask: Optional [B, N] mask for valid patches
        Returns:
            attended_features: [B, N, visual_dim] - query-aware visual features
            attention_weights: [B, num_heads, N] - attention scores
        """
        B, N, C = visual_features.shape

        # Project query (text) -> [B, 1, visual_dim]
        Q = self.q_proj(query_embed).unsqueeze(1)  # [B, 1, visual_dim]

        # Project visual features for keys and values
        K = self.k_proj(visual_features)  # [B, N, visual_dim]
        V = self.v_proj(visual_features)  # [B, N, visual_dim]

        # Reshape for multi-head attention
        Q = Q.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, N]
                float('-inf')
            )

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, 1, N]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, H, 1, D]

        # Reshape and project output
        attended = attended.transpose(1, 2).reshape(B, 1, C)  # [B, 1, visual_dim]
        output = self.out_proj(attended)  # [B, 1, visual_dim]

        # Also return attended features for all patches (weighted by attention)
        # Expand attended features to all patch positions
        attended_features = output.expand(B, N, C)  # [B, N, visual_dim]

        return attended_features, attn_weights.squeeze(2)  # [B, H, N]
```

### 3. Stacked Query-Conditioned Attention

**Multi-Step Reasoning**: Stack multiple attention layers for iterative refinement.

```python
class StackedQueryAttention(nn.Module):
    """Multiple attention layers for multi-step visual reasoning."""

    def __init__(self, visual_dim=512, query_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        self.num_layers = num_layers

        # Stack multiple attention layers
        self.attention_layers = nn.ModuleList([
            QueryConditionedAttention(visual_dim, query_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Layer normalization after each attention
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(visual_dim) for _ in range(num_layers)
        ])

        # Query update mechanism (optional)
        self.query_updaters = nn.ModuleList([
            nn.Linear(visual_dim + query_dim, query_dim)
            for _ in range(num_layers - 1)
        ])

    def forward(self, visual_features, query_embed):
        """
        Args:
            visual_features: [B, N, visual_dim]
            query_embed: [B, query_dim]
        Returns:
            attended_features: [B, N, visual_dim] - final attended features
            all_attention_weights: List of [B, num_heads, N] - attention at each layer
        """
        all_attention_weights = []
        current_query = query_embed
        current_visual = visual_features

        for i, (attn_layer, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Apply query-conditioned attention
            attended, attn_weights = attn_layer(current_visual, current_query)
            all_attention_weights.append(attn_weights)

            # Residual connection and layer norm
            current_visual = norm(current_visual + attended)

            # Update query for next layer (use attended visual info)
            if i < self.num_layers - 1:
                # Pool visual features using attention
                pooled_visual = torch.sum(
                    current_visual * attn_weights.mean(dim=1, keepdim=True).transpose(-1, -2),
                    dim=1
                )  # [B, visual_dim]

                # Concatenate with original query and update
                combined = torch.cat([pooled_visual, current_query], dim=-1)
                current_query = self.query_updaters[i](combined)

        return current_visual, all_attention_weights
```

## Relevance Scoring Mechanisms

### 1. Additive Attention (Bahdanau-style)

**For VQA tasks where query and visual features interact non-linearly**:

```python
class AdditiveQueryAttention(nn.Module):
    """Additive (Bahdanau) attention for query-visual interaction."""

    def __init__(self, visual_dim=512, query_dim=512, hidden_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, visual_features, query_embed):
        """
        Args:
            visual_features: [B, N, visual_dim]
            query_embed: [B, query_dim]
        Returns:
            relevance_scores: [B, N] - relevance of each patch to query
        """
        B, N, _ = visual_features.shape

        # Project visual and query to shared space
        visual_proj = self.visual_proj(visual_features)  # [B, N, hidden_dim]
        query_proj = self.query_proj(query_embed).unsqueeze(1)  # [B, 1, hidden_dim]

        # Additive combination with tanh activation
        combined = torch.tanh(visual_proj + query_proj)  # [B, N, hidden_dim]

        # Score each patch
        scores = self.score_proj(combined).squeeze(-1)  # [B, N]

        # Softmax to get relevance distribution
        relevance = torch.softmax(scores, dim=-1)

        return relevance
```

### 2. Bilinear Attention

**For capturing complex query-visual interactions**:

```python
class BilinearQueryAttention(nn.Module):
    """Bilinear attention for rich query-visual interaction."""

    def __init__(self, visual_dim=512, query_dim=512):
        super().__init__()
        # Bilinear weight matrix
        self.W = nn.Parameter(torch.randn(query_dim, visual_dim))

    def forward(self, visual_features, query_embed):
        """
        Args:
            visual_features: [B, N, visual_dim]
            query_embed: [B, query_dim]
        Returns:
            relevance_scores: [B, N]
        """
        # Bilinear scoring: q^T W v
        # query_embed: [B, query_dim]
        # W: [query_dim, visual_dim]
        # visual_features: [B, N, visual_dim]

        # Compute q^T W
        qW = torch.matmul(query_embed, self.W)  # [B, visual_dim]

        # Compute (q^T W) v for all patches
        scores = torch.matmul(qW.unsqueeze(1), visual_features.transpose(-1, -2))  # [B, 1, N]
        scores = scores.squeeze(1)  # [B, N]

        # Softmax
        relevance = torch.softmax(scores, dim=-1)

        return relevance
```

## Complete VQA Model with Query-Conditioned Attention

**Integrating all components into an end-to-end system**:

```python
class QueryConditionedVQAModel(nn.Module):
    """Complete VQA model with query-conditioned visual attention."""

    def __init__(
        self,
        vision_encoder_name='google/vit-base-patch16-224',
        text_encoder_name='bert-base-uncased',
        hidden_dim=768,
        num_attention_layers=2,
        num_attention_heads=8,
        num_answer_classes=3000,
        dropout=0.1
    ):
        super().__init__()

        # Vision encoder (ViT)
        from transformers import ViTModel
        self.vision_encoder = ViTModel.from_pretrained(vision_encoder_name)
        visual_dim = self.vision_encoder.config.hidden_size

        # Query embedder
        self.query_embedder = QueryEmbedder(
            text_encoder=text_encoder_name,
            hidden_dim=768,
            output_dim=hidden_dim
        )

        # Query-conditioned attention stack
        self.query_attention = StackedQueryAttention(
            visual_dim=visual_dim,
            query_dim=hidden_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads
        )

        # Fusion and answer prediction
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answer_classes)
        )

    def forward(self, images, questions):
        """
        Args:
            images: [B, 3, H, W] - input images
            questions: List of text questions
        Returns:
            logits: [B, num_answer_classes] - answer predictions
            attention_maps: List of attention weights for visualization
        """
        B = images.shape[0]

        # Encode image
        vision_outputs = self.vision_encoder(pixel_values=images)
        visual_features = vision_outputs.last_hidden_state  # [B, N, visual_dim]

        # Encode query
        query_embed = self.query_embedder(questions)  # [B, hidden_dim]

        # Apply query-conditioned attention
        attended_visual, attention_maps = self.query_attention(
            visual_features, query_embed
        )

        # Pool attended visual features
        attended_pooled = attended_visual.mean(dim=1)  # [B, visual_dim]

        # Fuse visual and textual features
        combined = torch.cat([attended_pooled, query_embed], dim=-1)  # [B, visual_dim + hidden_dim]
        fused = self.fusion(combined)  # [B, hidden_dim]

        # Predict answer
        logits = self.classifier(fused)  # [B, num_answer_classes]

        return logits, attention_maps

# Training example
def train_step(model, images, questions, answers, optimizer, criterion):
    """Single training step."""
    optimizer.zero_grad()

    # Forward pass
    logits, _ = model(images, questions)

    # Compute loss
    loss = criterion(logits, answers)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute accuracy
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == answers).float().mean()

    return loss.item(), accuracy.item()

# Inference with attention visualization
def visualize_attention(model, image, question):
    """Visualize what the model attends to for a given question."""
    model.eval()
    with torch.no_grad():
        logits, attention_maps = model(image.unsqueeze(0), [question])

        # Get predicted answer
        pred_idx = torch.argmax(logits, dim=-1).item()

        # Get attention from last layer, averaged across heads
        final_attention = attention_maps[-1].mean(dim=1).squeeze(0)  # [N]

        return pred_idx, final_attention
```

## Query-Aware Vision Transformer (QA-ViT)

**Advanced approach: Embed question awareness directly in vision encoder**

From [Question aware vision transformer for multimodal reasoning](https://www.amazon.science/publications/question-aware-vision-transformer-for-multimodal-reasoning) (Amazon Science, accessed 2025-01-31):

```python
class QAViTBlock(nn.Module):
    """Vision Transformer block with query-aware attention."""

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Standard self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        # Query-conditioned cross-attention
        self.query_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, query_embed=None):
        """
        Args:
            x: [B, N, dim] - patch embeddings
            query_embed: [B, 1, dim] - query embedding (optional)
        Returns:
            x: [B, N, dim] - query-aware features
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Query-conditioned attention (if query provided)
        if query_embed is not None:
            # Use query as query, visual features as key/value
            x = x + self.query_attn(
                query_embed.expand(-1, x.shape[1], -1),  # Expand query to all patches
                self.norm2(x),
                self.norm2(x)
            )[0]

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x
```

## Connection to ARR-COC Relevance Realization

**Query-conditioned attention implements key aspects of Vervaekean relevance realization**:

1. **Participatory Knowing** (Query-Content Coupling)
   - Query embedding represents agent's perspective
   - Attention weights measure relevance between query and visual patches
   - Transjective: emerges from relationship, not objective properties

2. **Dynamic Resource Allocation**
   - Attention weights allocate "computational budget" to relevant regions
   - Similar to ARR-COC's token budget allocation (64-400 tokens per patch)
   - Both optimize for query-relevant information

3. **Multi-Step Reasoning** (Stacked Attention)
   - First layer: broad focus (explore)
   - Second layer: refined focus (exploit)
   - Balances Compress↔Particularize tension

**Key Difference**:
- Query-conditioned attention: Soft weighting (all patches processed)
- ARR-COC: Hard allocation (variable tokens per patch based on relevance)
- ARR-COC more efficient for deployment (fewer total tokens)

## Training Strategies

### 1. End-to-End Training

```python
# Setup
model = QueryConditionedVQAModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for images, questions, answers in train_loader:
        images = images.to(device)
        answers = answers.to(device)

        loss, acc = train_step(model, images, questions, answers, optimizer, criterion)

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, Acc={acc:.4f}")
```

### 2. Two-Stage Training

**Stage 1**: Train without query conditioning (standard ViT)
**Stage 2**: Freeze vision encoder, train query attention layers

```python
# Stage 1: Standard vision encoder training
for param in model.query_attention.parameters():
    param.requires_grad = False

# Train vision encoder...

# Stage 2: Fine-tune with query conditioning
for param in model.vision_encoder.parameters():
    param.requires_grad = False

for param in model.query_attention.parameters():
    param.requires_grad = True

# Fine-tune query attention...
```

### 3. Attention Supervision (Optional)

**Use human attention maps to guide learning**:

```python
def attention_supervision_loss(pred_attention, target_attention):
    """
    Args:
        pred_attention: [B, N] - model's attention weights
        target_attention: [B, N] - human gaze heatmap
    Returns:
        loss: Scalar
    """
    # KL divergence between predicted and target attention
    pred_attention = pred_attention + 1e-8  # Avoid log(0)
    target_attention = target_attention + 1e-8

    loss = F.kl_div(
        pred_attention.log(),
        target_attention,
        reduction='batchmean'
    )

    return loss

# Combined loss
total_loss = task_loss + lambda_attn * attention_supervision_loss(pred_attn, target_attn)
```

## Evaluation and Visualization

### 1. Attention Map Visualization

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_query_attention(model, image_path, question, patch_size=16):
    """Visualize where the model attends for a given question."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Get attention weights
    with torch.no_grad():
        _, attention_maps = model(image_tensor, [question])

    # Use final layer attention, average across heads
    attention = attention_maps[-1].mean(dim=1).squeeze(0).cpu().numpy()  # [N]

    # Reshape to 2D grid (assuming square patches)
    H = W = int(np.sqrt(len(attention)))
    attention_map = attention.reshape(H, W)

    # Resize to original image size
    from scipy.ndimage import zoom
    h, w = image.size
    attention_resized = zoom(attention_map, (h // (patch_size * H), w // (patch_size * W)))

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title(f"Attention Map\nQuestion: {question}")
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(attention_resized, alpha=0.6, cmap='hot')
    axes[2].set_title("Attention Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
```

### 2. Attention Rollout

**Aggregate attention across multiple layers**:

```python
def attention_rollout(attention_maps):
    """
    Compute attention rollout across layers.

    Args:
        attention_maps: List of [B, num_heads, N] attention weights
    Returns:
        rollout: [B, N] - aggregated attention
    """
    # Average across heads for each layer
    layer_attns = [attn.mean(dim=1) for attn in attention_maps]  # List of [B, N]

    # Initialize with identity
    B, N = layer_attns[0].shape
    rollout = torch.eye(N).unsqueeze(0).expand(B, -1, -1).to(layer_attns[0].device)

    # Multiply attention matrices
    for attn in layer_attns:
        # Convert attention weights to matrix form [B, N, N]
        attn_matrix = attn.unsqueeze(-1).expand(-1, -1, N)
        rollout = torch.matmul(rollout, attn_matrix)

    # Return attention to first patch (often [CLS] token)
    return rollout[:, 0, :]
```

## Performance Optimization

### 1. Efficient Attention Computation

```python
# Use Flash Attention for faster computation
from torch.nn.functional import scaled_dot_product_attention

def efficient_query_attention(Q, K, V, mask=None):
    """Memory-efficient attention using PyTorch 2.0 SDPA."""
    return scaled_dot_product_attention(
        Q, K, V,
        attn_mask=mask,
        dropout_p=0.0 if not training else 0.1,
        is_causal=False
    )
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, questions, answers in train_loader:
    optimizer.zero_grad()

    # Mixed precision forward pass
    with autocast():
        logits, _ = model(images, questions)
        loss = criterion(logits, answers)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Benchmarks and Results

**VQA v2.0 Dataset** (from literature):
- Baseline (no query conditioning): 48.2% accuracy
- Stacked attention (2 layers): 54.8% accuracy
- QA-ViT: 58.3% accuracy

**Attention Quality Metrics**:
- Attention Accuracy: IoU between predicted attention and human gaze
- Relevance Correlation: Spearman correlation with ground truth relevance scores

## Sources

**Web Research:**

- [Visual-Question-Answering-using-Stacked-Attention-Networks](https://github.com/williamcfrancis/Visual-Question-Answering-using-Stacked-Attention-Networks) (GitHub, accessed 2025-01-31)
  - Stacked attention architecture for VQA
  - Multi-step reasoning with iterative attention refinement
  - Achieved 54.82% accuracy on VQA v2.0

- [Question aware vision transformer for multimodal reasoning](https://www.amazon.science/publications/question-aware-vision-transformer-for-multimodal-reasoning) (Amazon Science, accessed 2025-01-31)
  - QA-ViT architecture embeds question awareness in vision encoder
  - Model-agnostic approach applicable to various VL architectures
  - Dynamic visual features focusing on query-relevant aspects

**Related Concepts:**
- [mechanisms/00-query-conditioned-attention.md](../mechanisms/00-query-conditioned-attention.md) - Architectural overview
- [training/00-vlm-pretraining-strategies.md](../training/00-vlm-pretraining-strategies.md) - Training approaches
- [../../../arr-coc-ovis/arr_coc/attending.py](../../../arr-coc-ovis/arr_coc/attending.py) - ARR-COC relevance realization

**Connection to ARR-COC Project:**
- Query-conditioned attention implements participatory knowing (query-content coupling)
- Attention weights realize relevance dynamically based on query
- Stacked attention enables multi-step reasoning (balancing exploration vs exploitation)
- Soft weighting vs ARR-COC's hard token allocation (trade-off: flexibility vs efficiency)
