# Q-Former (BLIP-2) Implementation Guide

## Overview

Q-Former (Querying Transformer) is the lightweight bridging module introduced in BLIP-2 that connects frozen vision encoders with frozen large language models. It addresses the modality gap through a two-stage pre-training strategy and achieves state-of-the-art vision-language performance with minimal trainable parameters (188M).

**Key Innovation**: Q-Former uses learnable query tokens that extract visual features most relevant to text through cross-attention, dramatically reducing computational cost compared to end-to-end training.

From [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597) (Salesforce Research, 2023, accessed 2025-01-31):
- Outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters
- Bridges frozen ViT encoders (e.g., ViT-L/14) with frozen LLMs (OPT, FlanT5)
- Two-stage training: (1) vision-language representation learning, (2) vision-to-language generative learning

## Architecture Components

### 1. Learnable Query Embeddings

Q-Former uses a fixed set of learnable query tokens as input to extract visual information:

```python
import torch
import torch.nn as nn

class QueryEmbeddings(nn.Module):
    """Learnable query tokens for Q-Former"""
    def __init__(self, num_queries=32, hidden_dim=768):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings (32 x 768 by default)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )

    def forward(self, batch_size):
        # Expand queries to batch size
        return self.query_tokens.expand(batch_size, -1, -1)
```

**Design rationale**: The compact query representation (32 × 768 = 24,576 dimensions) is much smaller than frozen image features (e.g., 257 × 1024 = 263,168 for ViT-L/14), forcing queries to extract only the most relevant visual information for text.

### 2. Q-Former Transformer Architecture

From [HuggingFace BLIP-2 Documentation](https://huggingface.co/docs/transformers/v4.36.1/model_doc/blip-2) (accessed 2025-01-31):

Q-Former consists of two transformer submodules sharing self-attention layers:
1. **Image transformer**: Interacts with frozen image encoder via cross-attention
2. **Text transformer**: Functions as both text encoder and text decoder

```python
class QFormerLayer(nn.Module):
    """Single Q-Former transformer layer with self-attention and cross-attention"""
    def __init__(self, hidden_dim=768, num_heads=12, cross_attention_freq=2):
        super().__init__()

        # Self-attention (shared between image and text transformers)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to frozen image features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, queries, image_features, attention_mask=None):
        # Self-attention among queries
        residual = queries
        queries_norm = self.self_attn_norm(queries)
        queries_attn, _ = self.self_attn(
            queries_norm, queries_norm, queries_norm,
            attn_mask=attention_mask
        )
        queries = residual + queries_attn

        # Cross-attention to image features
        residual = queries
        queries_norm = self.cross_attn_norm(queries)
        queries_cross, _ = self.cross_attn(
            queries_norm,           # Query
            image_features,         # Key
            image_features,         # Value
        )
        queries = residual + queries_cross

        # Feed-forward
        residual = queries
        queries_norm = self.ffn_norm(queries)
        queries_ffn = self.ffn(queries_norm)
        queries = residual + queries_ffn

        return queries
```

### 3. Attention Masking Strategies

From [Papers Explained 155: BLIP 2](https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65) (accessed 2025-01-31):

Q-Former employs different attention masks for different pre-training objectives:

```python
class AttentionMasks:
    """Creates attention masks for different Q-Former training objectives"""

    @staticmethod
    def itc_mask(num_queries, num_text_tokens):
        """Image-Text Contrastive (ITC) - Unimodal mask

        Queries and text cannot see each other (prevent information leak)

        Returns:
            mask: [num_queries + num_text_tokens, num_queries + num_text_tokens]
        """
        total_len = num_queries + num_text_tokens
        mask = torch.zeros(total_len, total_len)

        # Queries can attend to queries
        mask[:num_queries, :num_queries] = 1

        # Text can attend to text
        mask[num_queries:, num_queries:] = 1

        return mask

    @staticmethod
    def itg_mask(num_queries, num_text_tokens):
        """Image-grounded Text Generation (ITG) - Multimodal causal mask

        Queries can attend to each other but not text.
        Text tokens can attend to all queries and previous text tokens.

        Returns:
            mask: [num_queries + num_text_tokens, num_queries + num_text_tokens]
        """
        total_len = num_queries + num_text_tokens
        mask = torch.zeros(total_len, total_len)

        # Queries can attend to queries
        mask[:num_queries, :num_queries] = 1

        # Text can attend to all queries
        mask[num_queries:, :num_queries] = 1

        # Text can attend to previous text (causal)
        for i in range(num_text_tokens):
            mask[num_queries + i, num_queries:num_queries + i + 1] = 1

        return mask

    @staticmethod
    def itm_mask(num_queries, num_text_tokens):
        """Image-Text Matching (ITM) - Bidirectional mask

        All queries and text can attend to each other (full attention)

        Returns:
            mask: [num_queries + num_text_tokens, num_queries + num_text_tokens]
        """
        total_len = num_queries + num_text_tokens
        return torch.ones(total_len, total_len)
```

### 4. Complete Q-Former Module

```python
class QFormer(nn.Module):
    """Complete Q-Former implementation for BLIP-2

    Based on BERT-base architecture with cross-attention layers
    """
    def __init__(
        self,
        num_queries=32,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        cross_attention_freq=2,
        encoder_hidden_dim=1408,  # ViT-L/14 hidden dim
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable query embeddings
        self.query_embeddings = QueryEmbeddings(num_queries, hidden_dim)

        # Projection layer for image features (if dimensions don't match)
        self.image_proj = nn.Linear(encoder_hidden_dim, hidden_dim) \
            if encoder_hidden_dim != hidden_dim else nn.Identity()

        # Q-Former transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                cross_attention_freq=cross_attention_freq
            )
            for _ in range(num_layers)
        ])

        # Text embedding layer (initialized from BERT)
        self.text_embeddings = nn.Embedding(30522, hidden_dim)  # BERT vocab size

    def forward(
        self,
        image_features,      # [batch, num_patches, encoder_hidden_dim]
        text_input_ids=None, # [batch, seq_len] (optional)
        attention_mask_type='bidirectional',  # 'unimodal', 'causal', 'bidirectional'
    ):
        batch_size = image_features.shape[0]

        # Get learnable query embeddings
        queries = self.query_embeddings(batch_size)

        # Project image features to Q-Former dimension
        image_features = self.image_proj(image_features)

        # Optional: Add text embeddings
        if text_input_ids is not None:
            text_embeds = self.text_embeddings(text_input_ids)
            # Concatenate queries and text
            combined = torch.cat([queries, text_embeds], dim=1)
        else:
            combined = queries

        # Create attention mask based on objective
        if attention_mask_type == 'unimodal':
            mask = AttentionMasks.itc_mask(self.num_queries,
                                          text_input_ids.shape[1] if text_input_ids is not None else 0)
        elif attention_mask_type == 'causal':
            mask = AttentionMasks.itg_mask(self.num_queries,
                                          text_input_ids.shape[1] if text_input_ids is not None else 0)
        else:  # bidirectional
            mask = AttentionMasks.itm_mask(self.num_queries,
                                          text_input_ids.shape[1] if text_input_ids is not None else 0)

        # Apply Q-Former layers
        hidden_states = combined
        for layer in self.layers:
            hidden_states = layer(hidden_states, image_features, mask)

        # Extract query outputs (first num_queries tokens)
        query_outputs = hidden_states[:, :self.num_queries]

        return query_outputs
```

## Two-Stage Pre-training Strategy

### Stage 1: Vision-Language Representation Learning

From [BLIP-2 Paper](https://arxiv.org/abs/2301.12597) (accessed 2025-01-31):

Three objectives are jointly optimized with frozen image encoder:

```python
class Stage1PreTraining(nn.Module):
    """Stage 1: Bootstrap vision-language representation from frozen image encoder"""
    def __init__(self, qformer, vision_encoder):
        super().__init__()
        self.qformer = qformer
        self.vision_encoder = vision_encoder

        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Projection heads for each objective
        self.itc_projection = nn.Linear(768, 256)  # Image-Text Contrast
        self.itm_head = nn.Linear(768, 2)          # Image-Text Matching

    def forward(self, images, text_input_ids, text_targets):
        # Extract frozen image features
        with torch.no_grad():
            image_features = self.vision_encoder(images)

        # 1. Image-Text Contrastive Learning (ITC)
        query_outputs_itc = self.qformer(
            image_features,
            text_input_ids,
            attention_mask_type='unimodal'
        )
        # Use highest similarity between any query and text
        image_embeds = self.itc_projection(query_outputs_itc)
        text_embeds = self.itc_projection(query_outputs_itc)  # From text transformer
        itc_loss = contrastive_loss(image_embeds, text_embeds)

        # 2. Image-grounded Text Generation (ITG)
        query_outputs_itg = self.qformer(
            image_features,
            text_input_ids,
            attention_mask_type='causal'
        )
        itg_loss = language_modeling_loss(query_outputs_itg, text_targets)

        # 3. Image-Text Matching (ITM)
        query_outputs_itm = self.qformer(
            image_features,
            text_input_ids,
            attention_mask_type='bidirectional'
        )
        # Average query outputs for matching score
        pooled = query_outputs_itm.mean(dim=1)
        itm_logits = self.itm_head(pooled)
        itm_loss = matching_loss(itm_logits, labels)

        return itc_loss + itg_loss + itm_loss
```

**Training details** (from paper):
- Pre-trained for 250,000 steps
- Uses same data as BLIP (129M images)
- Initialized with BERT-base weights (cross-attention layers randomly initialized)
- Optimizer: AdamW with learning rate 1e-4

### Stage 2: Vision-to-Language Generative Learning

```python
class Stage2PreTraining(nn.Module):
    """Stage 2: Bootstrap vision-to-language generation from frozen LLM"""
    def __init__(self, qformer, vision_encoder, language_model):
        super().__init__()
        self.qformer = qformer
        self.vision_encoder = vision_encoder
        self.language_model = language_model

        # Freeze both vision encoder and LLM
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Linear projection to LLM embedding dimension
        self.query_to_llm = nn.Linear(768, language_model.config.hidden_size)

    def forward(self, images, text_input_ids, text_targets):
        # Extract frozen image features
        with torch.no_grad():
            image_features = self.vision_encoder(images)

        # Get query outputs from Q-Former
        query_outputs = self.qformer(
            image_features,
            attention_mask_type='bidirectional'
        )

        # Project queries to LLM embedding space
        query_embeds_llm = self.query_to_llm(query_outputs)

        # Get text embeddings from LLM
        text_embeds = self.language_model.get_input_embeddings()(text_input_ids)

        # Prepend query embeddings as "soft visual prompts"
        inputs_embeds = torch.cat([query_embeds_llm, text_embeds], dim=1)

        # Forward through frozen LLM with language modeling objective
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            labels=text_targets
        )

        return outputs.loss
```

**Training details** (from paper):
- Pre-trained for 80,000 steps
- Language models tested: OPT (2.7B, 6.7B, 30B, 66B), FlanT5 (3B, 11B, XXL)
- Only Q-Former and projection layer are trainable
- Enables leveraging LLM capabilities without full fine-tuning

## Hyperparameters and Configuration

From [HuggingFace BLIP-2 Configuration](https://huggingface.co/docs/transformers/v4.36.1/model_doc/blip-2) (accessed 2025-01-31):

```python
class Blip2Config:
    """Standard BLIP-2 configuration"""

    # Q-Former configuration
    qformer_config = {
        'vocab_size': 30522,              # BERT vocab
        'hidden_size': 768,                # Hidden dimension
        'num_hidden_layers': 12,           # Number of layers
        'num_attention_heads': 12,         # Attention heads
        'intermediate_size': 3072,         # FFN dimension
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'layer_norm_eps': 1e-12,
        'cross_attention_frequency': 2,    # Add cross-attention every N layers
        'encoder_hidden_size': 1408,       # ViT-L/14 dimension
    }

    # Vision encoder configuration
    vision_config = {
        'hidden_size': 1408,               # ViT-L/14
        'intermediate_size': 6144,
        'num_hidden_layers': 39,           # Use 2nd-to-last layer
        'num_attention_heads': 16,
        'image_size': 224,
        'patch_size': 14,
    }

    # Number of learnable query tokens
    num_query_tokens = 32
```

## Usage Example

```python
import torch
from transformers import AutoTokenizer

# Initialize components
vision_encoder = load_pretrained_vit()  # e.g., ViT-L/14 from CLIP
qformer = QFormer(
    num_queries=32,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    cross_attention_freq=2,
    encoder_hidden_dim=1408
)
language_model = load_pretrained_llm()  # e.g., OPT-2.7B or FlanT5-XL

# Freeze pretrained models
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in language_model.parameters():
    param.requires_grad = False

# Forward pass
images = torch.randn(4, 3, 224, 224)
text_input_ids = torch.randint(0, 30522, (4, 20))

# Extract image features
image_features = vision_encoder(images)  # [4, 257, 1408]

# Get query outputs
query_outputs = qformer(
    image_features,
    text_input_ids,
    attention_mask_type='bidirectional'
)  # [4, 32, 768]

print(f"Query outputs shape: {query_outputs.shape}")
print(f"Trainable parameters: {sum(p.numel() for p in qformer.parameters() if p.requires_grad):,}")
# Trainable parameters: ~188M (much smaller than frozen models)
```

## Key Design Decisions

1. **Compact query representation**: 32 × 768 = 24,576 dimensions forces information bottleneck
2. **BERT initialization**: Leverages strong language priors from pre-trained BERT
3. **Frozen encoders**: Enables bootstrapping from powerful pre-trained models without catastrophic forgetting
4. **Two-stage training**: First learns vision-language alignment, then learns to condition LLM
5. **Multiple attention masks**: Different masks for different objectives (contrastive, generative, matching)

## Implementation References

**Source Documents:**
- None (pure web research)

**Web Research:**
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597) - Original paper (accessed 2025-01-31)
- [HuggingFace BLIP-2 Documentation](https://huggingface.co/docs/transformers/v4.36.1/model_doc/blip-2) - Official implementation docs (accessed 2025-01-31)
- [Papers Explained 155: BLIP 2](https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65) - Architecture walkthrough (accessed 2025-01-31)
- [Salesforce LAVIS GitHub](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) - Official implementation repository (accessed 2025-01-31)
- [kyegomez/qformer GitHub](https://github.com/kyegomez/qformer) - Community implementation in Zeta blocks (accessed 2025-01-31)

**Additional References:**
- [HuggingFace Transformers Issues](https://github.com/huggingface/transformers/issues/25245) - Q-Former implementation discussions (accessed 2025-01-31)
- [Rohit Bandaru's VLM Guide](https://rohitbandaru.github.io/blog/Vision-Language-Models/) - Q-Former vs Perceiver comparison (accessed 2025-01-31)
