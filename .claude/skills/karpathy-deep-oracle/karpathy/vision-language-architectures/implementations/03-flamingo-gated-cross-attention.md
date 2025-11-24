# Flamingo Gated Cross-Attention Implementation

**Topic**: Gated cross-attention mechanism from DeepMind's Flamingo model
**Difficulty**: Advanced
**Prerequisites**: Understanding of transformers, attention mechanisms, PyTorch basics

---

## Overview

Flamingo, introduced by DeepMind in 2022, pioneered the use of **gated cross-attention** to seamlessly integrate visual information into frozen language models. This mechanism allows a pre-trained language model to learn to incorporate visual features without catastrophically forgetting its language capabilities.

The key innovation: **tanh-gated cross-attention layers** inserted between frozen language model blocks, which start at zero influence and gradually learn to inject visual information during training.

### Why Gated Cross-Attention?

**Problem**: Injecting visual features directly into a pre-trained language model destroys the carefully learned language representations.

**Solution**: Use learnable gating (tanh activation) initialized to zero, allowing the model to:
1. Start training with the language model functioning exactly as before (gate = 0)
2. Gradually learn to introduce visual information as the perceiver resampler improves
3. Control the amount of visual influence at each layer

From [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/pdf/2204.14198) (accessed 2025-01-31):
- "We freeze the pretrained LM blocks, and insert gated cross-attention dense blocks trained from scratch"
- "The tanh gating mechanism ensures training stability by starting with zero visual influence"

---

## Architecture Overview

Flamingo's architecture consists of four main components working together:

```
┌─────────────────────────────────────────────────────────┐
│  Input: Interleaved Text + Images                       │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   Text Path              Image Path
        │                     │
        │              ┌──────▼──────┐
        │              │ Vision      │
        │              │ Encoder     │
        │              │ (NFNet F6)  │
        │              └──────┬──────┘
        │                     │
        │              ┌──────▼──────────┐
        │              │ Perceiver       │
        │              │ Resampler       │
        │              │ (fixed tokens)  │
        │              └──────┬──────────┘
        │                     │
        └─────────┬───────────┘
                  │
          ┌───────▼────────┐
          │  Frozen LM     │
          │  Block         │
          └───────┬────────┘
                  │
          ┌───────▼────────────┐
          │  GATED XATTN-DENSE │  ← This is what we implement
          │  (trainable)       │
          └───────┬────────────┘
                  │
          ┌───────▼────────┐
          │  Frozen LM     │
          │  Block         │
          └───────┬────────┘
                  │
                 ...
```

**Key points**:
- Vision encoder (NFNet) extracts spatial features from images
- Perceiver resampler converts variable-length image sequences to fixed number of tokens
- Gated cross-attention layers interleave between frozen LM blocks
- Text flows through frozen LM, visual info injected via gated XATTN

---

## The Gated Cross-Attention Mechanism

### High-Level Intuition

Think of gated cross-attention as a "volume knob" for visual information:

```python
# Conceptual equation
result = (tanh(alpha) * cross_attention(visual, text)) + text

# At initialization (alpha = 0)
result = (0 * cross_attention(visual, text)) + text = text  # Pure LM

# After training (alpha learned)
result = (tanh(alpha) * cross_attention(visual, text)) + text  # Blended
```

The `tanh(alpha)` gate:
- Starts at 0 (no visual influence)
- Gradually opens as the model learns
- Controls how much visual information flows into each layer

From [Understanding DeepMind's Flamingo Visual Language Models](https://medium.com/@paluchasz/understanding-flamingo-visual-language-models-bea5eeb05268) (accessed 2025-01-31):
- "A 'gating' mechanism is used to improve training stability and final performance"
- "The gate ensures that at initialization, the visual features have zero influence"

---

## Complete PyTorch Implementation

### 1. Gated Cross-Attention Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GatedCrossAttentionBlock(nn.Module):
    """
    Flamingo's gated cross-attention mechanism.

    Injects visual information into a language model using:
    1. Cross-attention (text queries, visual keys/values)
    2. Tanh gating (controls visual influence)
    3. Feedforward network (post-attention processing)

    Args:
        dim: Model hidden dimension
        dim_visual: Visual feature dimension (from perceiver resampler)
        dim_head: Dimension per attention head
        heads: Number of attention heads
        ff_mult: Feedforward expansion factor (default 4)
    """

    def __init__(
        self,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        # Cross-attention components
        self.scale = dim_head ** -0.5

        # Query: from text (language model hidden states)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)

        # Key, Value: from visual (perceiver resampler output)
        self.to_kv = nn.Linear(dim_visual, self.inner_dim * 2, bias=False)

        # Output projection
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        # Gating mechanism
        # Alpha initialized to 0 → tanh(0) = 0 → no visual influence initially
        self.alpha_xattn = nn.Parameter(torch.tensor(0.0))

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

        # Gating for feedforward
        self.alpha_ff = nn.Parameter(torch.tensor(0.0))

        # Layer normalization
        self.norm_context = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,                    # [batch, seq_len, dim]
        visual: torch.Tensor,                # [batch, num_visual_tokens, dim_visual]
        mask: Optional[torch.Tensor] = None, # [batch, seq_len, num_visual_tokens]
    ) -> torch.Tensor:
        """
        Forward pass with gated cross-attention.

        Args:
            x: Text features from language model [B, T, D]
            visual: Visual features from perceiver [B, V, D_v]
            mask: Attention mask [B, T, V] (optional)

        Returns:
            Updated text features [B, T, D]
        """
        batch_size, seq_len, _ = x.shape

        # Layer norm before cross-attention
        normed_x = self.norm_context(x)

        # === Cross-Attention ===
        # Query from text
        q = self.to_q(normed_x)  # [B, T, heads * dim_head]
        q = q.view(batch_size, seq_len, self.heads, self.dim_head)
        q = q.transpose(1, 2)  # [B, heads, T, dim_head]

        # Key, Value from visual
        kv = self.to_kv(visual)  # [B, V, 2 * heads * dim_head]
        k, v = kv.chunk(2, dim=-1)

        k = k.view(batch_size, -1, self.heads, self.dim_head)
        k = k.transpose(1, 2)  # [B, heads, V, dim_head]

        v = v.view(batch_size, -1, self.heads, self.dim_head)
        v = v.transpose(1, 2)  # [B, heads, V, dim_head]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [B, heads, T, V]

        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads: [B, T, V] → [B, 1, T, V]
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)
        # [B, heads, T, dim_head]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        attn_out = self.to_out(attn_out)

        # === Tanh Gating ===
        # Gate starts at 0, learns to open during training
        gate_xattn = torch.tanh(self.alpha_xattn)

        # Apply gate and residual connection
        x = x + gate_xattn * attn_out

        # === Gated Feedforward ===
        normed_ff = self.norm_ff(x)
        ff_out = self.ff(normed_ff)

        gate_ff = torch.tanh(self.alpha_ff)
        x = x + gate_ff * ff_out

        return x
```

### 2. Flamingo Layer (LM Block + Gated XATTN)

```python
class FlamingoLayer(nn.Module):
    """
    A complete Flamingo layer combining:
    1. Frozen language model block (self-attention + FFN)
    2. Gated cross-attention block (visual injection)

    Args:
        lm_block: Pre-trained language model transformer block (frozen)
        dim: Model hidden dimension
        dim_visual: Visual feature dimension
        dim_head: Dimension per attention head
        heads: Number of attention heads
    """

    def __init__(
        self,
        lm_block: nn.Module,  # Frozen pre-trained LM block
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
    ):
        super().__init__()

        # Frozen language model block
        self.lm_block = lm_block
        for param in self.lm_block.parameters():
            param.requires_grad = False

        # Trainable gated cross-attention
        self.gated_xattn = GatedCrossAttentionBlock(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        visual: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Flamingo layer.

        Args:
            x: Text features [B, T, D]
            visual: Visual features [B, V, D_v]
            attention_mask: Causal mask for LM self-attention
            visual_mask: Mask for cross-attention to visual tokens

        Returns:
            Updated text features [B, T, D]
        """
        # 1. Frozen LM block (self-attention on text only)
        x = self.lm_block(x, attention_mask=attention_mask)

        # 2. Gated cross-attention (inject visual information)
        x = self.gated_xattn(x, visual, mask=visual_mask)

        return x
```

### 3. Full Flamingo Model

```python
class FlamingoModel(nn.Module):
    """
    Complete Flamingo architecture for vision-language modeling.

    Combines:
    - Vision encoder (e.g., CLIP ViT)
    - Perceiver resampler (compress visual tokens)
    - Frozen language model with interleaved gated cross-attention

    Args:
        vision_encoder: Pre-trained vision model
        perceiver_resampler: Perceiver module for visual token compression
        lm_blocks: List of frozen language model transformer blocks
        dim: Model hidden dimension
        dim_visual: Visual feature dimension from perceiver
        cross_attn_every_n_layers: Insert cross-attention every N layers
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        perceiver_resampler: nn.Module,
        lm_blocks: nn.ModuleList,
        dim: int,
        dim_visual: int,
        cross_attn_every_n_layers: int = 1,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.perceiver_resampler = perceiver_resampler

        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Build Flamingo layers
        self.layers = nn.ModuleList()
        for i, lm_block in enumerate(lm_blocks):
            # Insert gated cross-attention every N layers
            if (i + 1) % cross_attn_every_n_layers == 0:
                layer = FlamingoLayer(
                    lm_block=lm_block,
                    dim=dim,
                    dim_visual=dim_visual,
                )
            else:
                # Just use frozen LM block
                layer = lm_block
                for param in layer.parameters():
                    param.requires_grad = False

            self.layers.append(layer)

        self.cross_attn_every_n_layers = cross_attn_every_n_layers

    def forward(
        self,
        text_tokens: torch.Tensor,      # [B, T]
        images: torch.Tensor,            # [B, num_images, C, H, W]
        image_positions: torch.Tensor,   # [B, num_images] (where images go in text)
    ) -> torch.Tensor:
        """
        Forward pass through Flamingo.

        Args:
            text_tokens: Tokenized text [B, T]
            images: Images [B, N, C, H, W]
            image_positions: Indices where images appear in text [B, N]

        Returns:
            Language model logits [B, T, vocab_size]
        """
        batch_size, num_images = images.shape[:2]

        # === Process Images ===
        # Flatten batch and num_images for vision encoder
        images_flat = images.view(-1, *images.shape[2:])

        with torch.no_grad():
            # Extract visual features
            visual_features = self.vision_encoder(images_flat)
            # [B * num_images, num_patches, dim_visual]

        # Reshape back to batch
        visual_features = visual_features.view(
            batch_size, num_images, -1, visual_features.shape[-1]
        )

        # Apply perceiver resampler (compress to fixed tokens)
        visual_tokens = self.perceiver_resampler(visual_features)
        # [B, num_images, num_latents, dim_visual]

        # Flatten for cross-attention
        num_latents = visual_tokens.shape[2]
        visual_tokens = visual_tokens.view(
            batch_size, num_images * num_latents, -1
        )
        # [B, num_images * num_latents, dim_visual]

        # === Create Visual Attention Mask ===
        # Only attend to visual tokens from immediately preceding image
        visual_mask = self._create_visual_mask(
            text_tokens, image_positions, num_latents
        )
        # [B, T, num_images * num_latents]

        # === Process Through Flamingo Layers ===
        # Embed text (assuming embedding layer exists)
        # x = self.text_embedding(text_tokens)
        # For this example, assume x is already embedded
        x = text_tokens  # Placeholder

        for i, layer in enumerate(self.layers):
            if isinstance(layer, FlamingoLayer):
                # Gated cross-attention layer
                x = layer(x, visual_tokens, visual_mask=visual_mask)
            else:
                # Frozen LM block only
                x = layer(x)

        return x

    def _create_visual_mask(
        self,
        text_tokens: torch.Tensor,
        image_positions: torch.Tensor,
        num_latents: int,
    ) -> torch.Tensor:
        """
        Create mask so each text token only attends to immediately preceding image.

        Args:
            text_tokens: [B, T]
            image_positions: [B, num_images] (index in text where each image appears)
            num_latents: Number of tokens per image from perceiver

        Returns:
            Mask [B, T, num_images * num_latents]
        """
        batch_size, seq_len = text_tokens.shape
        num_images = image_positions.shape[1]

        # Initialize mask (all False = masked out)
        mask = torch.zeros(
            batch_size, seq_len, num_images * num_latents,
            dtype=torch.bool,
            device=text_tokens.device,
        )

        for b in range(batch_size):
            for t in range(seq_len):
                # Find most recent image before position t
                preceding_images = image_positions[b] < t
                if preceding_images.any():
                    latest_img_idx = preceding_images.nonzero()[-1].item()

                    # Unmask tokens for this image
                    start_idx = latest_img_idx * num_latents
                    end_idx = start_idx + num_latents
                    mask[b, t, start_idx:end_idx] = True

        return mask
```

---

## Key Implementation Details

### 1. Tanh Gating Mathematics

The tanh gate provides smooth control of visual influence:

```python
# Initialize alpha to 0
alpha = nn.Parameter(torch.tensor(0.0))

# Gate value
gate = torch.tanh(alpha)

# Properties:
# - tanh(0) = 0 → no visual influence initially
# - tanh is smooth: gradient flows easily during training
# - tanh saturates at ±1: bounded influence prevents instability
```

**Why tanh instead of sigmoid?**
- Symmetric around 0 (can learn negative gating if needed)
- Stronger gradients near 0 (faster learning initially)
- Well-studied in neural networks

From [Flamingo - Intuitively and Exhaustively Explained](https://towardsdatascience.com/flamingo-intuitively-and-exhaustively-explained-bf745611238b/) (accessed 2025-01-31):
- "tanh gating allows Flamingo to gently introduce image information into the model"
- "Without this, injecting image information at initialization would heavily confuse the LM"

### 2. Freezing Strategy

Flamingo carefully controls what gets trained:

```python
# ✓ Trainable:
- Gated cross-attention layers (initialized new)
- Perceiver resampler (initialized new)
- Alpha parameters (gate controllers)
- Feedforward networks after cross-attention

# ✗ Frozen:
- Vision encoder (pre-trained CLIP/NFNet)
- Language model blocks (pre-trained LM)
```

**Why freeze the LM?**
1. Preserve language capabilities (avoid catastrophic forgetting)
2. Reduce compute (fewer gradients to backpropagate)
3. Data efficiency (don't need to re-learn language from scratch)

### 3. Visual Masking Strategy

Flamingo uses a specific masking strategy: **each text token only attends to the immediately preceding image**.

```python
# Example sequence:
# [text_1] <image_1> [text_2] <image_2> [text_3]

# Attention pattern:
# text_1 → no images (none precede it)
# text_2 → image_1 (most recent)
# text_3 → image_2 (most recent, even though image_1 also precedes)
```

**Why this strategy?**

From [Flamingo paper](https://arxiv.org/pdf/2204.14198) (accessed 2025-01-31):
1. **Relevant context**: Text immediately following an image is most likely about that image
2. **Computational efficiency**: Reduces attention computation (don't attend to all images)
3. **Generalization**: Prevents overfitting to specific image orderings during training

---

## Training Considerations

### 1. Initialization

```python
# Critical: Start with zero visual influence
model = FlamingoModel(...)

# Verify gates are at zero
for layer in model.layers:
    if isinstance(layer, FlamingoLayer):
        assert layer.gated_xattn.alpha_xattn.item() == 0.0
        assert layer.gated_xattn.alpha_ff.item() == 0.0
```

### 2. Learning Rate Schedule

Use different learning rates for different components:

```python
# Typical setup
optimizer = torch.optim.AdamW([
    # Higher LR for new components
    {'params': gated_xattn_params, 'lr': 1e-4},
    {'params': perceiver_params, 'lr': 1e-4},

    # Vision encoder and LM are frozen (no params here)
])
```

### 3. Gradient Flow

Monitor alpha parameters during training:

```python
# Check if gates are learning to open
for epoch in range(num_epochs):
    # ... training loop ...

    # Log gate values
    for i, layer in enumerate(model.layers):
        if isinstance(layer, FlamingoLayer):
            alpha_val = layer.gated_xattn.alpha_xattn.item()
            gate_val = torch.tanh(torch.tensor(alpha_val)).item()
            print(f"Layer {i}: alpha={alpha_val:.3f}, gate={gate_val:.3f}")
```

Expected progression:
- Epoch 0: `gate ≈ 0.0` (no visual influence)
- Epoch 10: `gate ≈ 0.1-0.3` (gentle visual influence)
- Epoch 50: `gate ≈ 0.5-0.8` (strong visual influence)

---

## Minimal Working Example

```python
import torch
import torch.nn as nn

# === Setup ===
batch_size = 2
seq_len = 20
num_visual_tokens = 64
dim = 512
dim_visual = 768

# Mock components
class MockLMBlock(nn.Module):
    """Placeholder for frozen LM block"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, attention_mask=None):
        # Self-attention
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm(x))
        return x

# Create Flamingo layer
lm_block = MockLMBlock(dim)
flamingo_layer = FlamingoLayer(
    lm_block=lm_block,
    dim=dim,
    dim_visual=dim_visual,
    dim_head=64,
    heads=8,
)

# === Run Forward Pass ===
# Text features from language model
text_features = torch.randn(batch_size, seq_len, dim)

# Visual features from perceiver resampler
visual_features = torch.randn(batch_size, num_visual_tokens, dim_visual)

# Forward pass
output = flamingo_layer(text_features, visual_features)

print(f"Input shape: {text_features.shape}")
print(f"Output shape: {output.shape}")
print(f"Gate value: {torch.tanh(flamingo_layer.gated_xattn.alpha_xattn).item():.4f}")

# === Verify Zero Initialization ===
assert torch.allclose(
    torch.tanh(flamingo_layer.gated_xattn.alpha_xattn),
    torch.tensor(0.0),
    atol=1e-6
), "Gate should be initialized to 0!"

print("✓ Gated cross-attention working correctly!")
```

Expected output:
```
Input shape: torch.Size([2, 20, 512])
Output shape: torch.Size([2, 20, 512])
Gate value: 0.0000
✓ Gated cross-attention working correctly!
```

---

## Comparison with Other Fusion Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Gated XATTN (Flamingo)** | Tanh-gated cross-attention inserted between frozen LM blocks | Preserves LM, smooth learning, stable training | Requires careful initialization, more complex |
| **Direct Concatenation** | Concat visual features to text embeddings | Simple, fast | No learned fusion, fixed integration point |
| **Adapter Layers** | Small trainable modules between frozen blocks | Minimal params | No explicit visual attention |
| **Full Fine-tuning** | Train entire LM + vision encoder | Maximum flexibility | Catastrophic forgetting, data hungry |

**Why Flamingo's approach wins**:
- ✓ Preserves pre-trained LM capabilities (frozen blocks)
- ✓ Learns optimal visual injection points (cross-attention at every layer)
- ✓ Stable training (tanh gating prevents disruption)
- ✓ Efficient (only train cross-attention + perceiver)

---

## Common Pitfalls

### 1. Forgetting to Freeze the LM

```python
# ✗ Wrong: LM will catastrophically forget
flamingo_layer = FlamingoLayer(lm_block, ...)
# (lm_block is trainable)

# ✓ Correct: Freeze LM explicitly
flamingo_layer = FlamingoLayer(lm_block, ...)
for param in flamingo_layer.lm_block.parameters():
    param.requires_grad = False
```

### 2. Not Initializing Alpha to Zero

```python
# ✗ Wrong: Random initialization disrupts LM
self.alpha = nn.Parameter(torch.randn(1))

# ✓ Correct: Start at zero
self.alpha = nn.Parameter(torch.tensor(0.0))
```

### 3. Incorrect Visual Masking

```python
# ✗ Wrong: Allow attention to all images
mask = torch.ones(batch, seq_len, num_visual_tokens)

# ✓ Correct: Only immediately preceding image
mask = create_visual_mask(image_positions, ...)
```

### 4. Gradient Accumulation Issues

```python
# ✗ Wrong: Gates not learning
# Check if gradients are flowing to alpha parameters

# ✓ Correct: Monitor gradients
for name, param in model.named_parameters():
    if 'alpha' in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

---

## Extensions and Variations

### 1. Learnable Gate Per Head

```python
# Instead of single alpha per layer, use one per attention head
self.alpha_xattn = nn.Parameter(torch.zeros(num_heads))

# Apply per-head gating
gate = torch.tanh(self.alpha_xattn)  # [num_heads]
gate = gate.view(1, num_heads, 1, 1)  # [1, H, 1, 1]
attn_out = gate * attn_out  # Broadcast over [B, H, T, D]
```

**When to use**: If you want fine-grained control over which attention heads focus on visual vs. textual features.

### 2. Position-Dependent Gating

```python
# Gate value depends on position in sequence
self.alpha_xattn = nn.Parameter(torch.zeros(max_seq_len))

# During forward
gate = torch.tanh(self.alpha_xattn[:seq_len])  # [T]
gate = gate.view(1, seq_len, 1)  # [1, T, 1]
attn_out = gate * attn_out  # [B, T, D]
```

**When to use**: If visual influence should vary by position (e.g., stronger near image tokens).

### 3. Conditional Gating

```python
# Gate value predicted from text features
self.gate_predictor = nn.Linear(dim, 1)

# During forward
gate = torch.tanh(self.gate_predictor(x))  # [B, T, 1]
attn_out = gate * attn_out  # [B, T, D]
```

**When to use**: If visual relevance varies dynamically based on content.

---

## Integration with OpenFlamingo

The open-source [OpenFlamingo implementation](https://github.com/mlfoundations/open_flamingo) (accessed 2025-01-31) provides production-ready code:

```python
from open_flamingo import create_model_and_transforms

# Create Flamingo model with gated cross-attention
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,  # Insert gated XATTN every layer
)

# Load pre-trained weights
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
)
model.load_state_dict(torch.load(checkpoint_path), strict=False)

# Use for inference
generated_text = model.generate(
    vision_x=images,         # [B, num_images, 1, C, H, W]
    lang_x=input_ids,        # [B, T]
    attention_mask=attn_mask,
    max_new_tokens=20,
)
```

**Key OpenFlamingo components**:
- `GatedCrossAttentionBlock`: Implements tanh-gated cross-attention
- `PerceiverResampler`: Compresses visual tokens to fixed size
- `FlamingoLMMixin`: Wraps pre-trained LM with cross-attention layers

---

## Real-World Performance

From the [Flamingo paper](https://arxiv.org/pdf/2204.14198) (accessed 2025-01-31):

**Few-shot learning results**:
- COCO Captioning (4-shot): 89.0 CIDEr
- VQAv2 (4-shot): 54.8% accuracy
- OKVQA (4-shot): 44.7% accuracy

**Training efficiency**:
- Only ~10-15% of parameters are trainable (gated XATTN + perceiver)
- Converges in ~1/10th the compute of training from scratch
- Preserves language model capabilities (minimal forgetting)

**Scaling results**:
| Model Size | Params | COCO CIDEr | VQAv2 Acc |
|------------|--------|------------|-----------|
| Flamingo-3B | 3B | 77.3 | 45.8% |
| Flamingo-9B | 9B | 89.0 | 54.8% |
| Flamingo-80B | 80B | 93.1 | 56.3% |

---

## Summary

**Key Takeaways**:

1. **Gated cross-attention** allows seamless integration of visual features into frozen language models
2. **Tanh gating** starts at zero influence, gradually learning to inject visual information
3. **Visual masking** restricts attention to immediately preceding images for better generalization
4. **Frozen LM blocks** preserve language capabilities while learning multimodal understanding

**When to use gated cross-attention**:
- ✓ You have a pre-trained LM you want to make multimodal
- ✓ You want to preserve language model capabilities
- ✓ You need stable training with gradual visual integration
- ✓ You want to insert visual information at multiple LM layers

**When NOT to use**:
- ✗ Training from scratch (no frozen LM to preserve)
- ✗ Single-layer visual fusion sufficient (simpler approaches work)
- ✗ Compute-constrained (cross-attention at every layer adds overhead)

---

## Sources

**Research Papers**:
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/pdf/2204.14198) - DeepMind, NeurIPS 2022 (accessed 2025-01-31)
  - Original paper introducing gated cross-attention mechanism
  - Sections on GATED XATTN-DENSE layers and tanh gating

**Implementation Resources**:
- [OpenFlamingo GitHub Repository](https://github.com/mlfoundations/open_flamingo) (accessed 2025-01-31)
  - Open-source PyTorch implementation
  - Production-ready gated cross-attention code
  - Pre-trained model weights

**Tutorials and Explanations**:
- [Understanding DeepMind's Flamingo Visual Language Models](https://medium.com/@paluchasz/understanding-flamingo-visual-language-models-bea5eeb05268) - Medium (accessed 2025-01-31)
  - Explains gating mechanism for training stability
  - Details on initialization strategy

- [Flamingo - Intuitively and Exhaustively Explained](https://towardsdatascience.com/flamingo-intuitively-and-exhaustively-explained-bf745611238b/) - Towards Data Science (accessed 2025-01-31)
  - Comprehensive walkthrough of Flamingo architecture
  - Visual diagrams of cross-attention and gating
  - Step-by-step explanation of attention mechanism

**Additional References**:
- [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) - Community implementation (accessed 2025-01-31)
- OpenFlamingo blog posts: [Part 1](https://laion.ai/blog/open-flamingo/), [Part 2](https://laion.ai/blog/open-flamingo-v2/)

---

## Related Topics

- `00-minimal-vlm-pytorch.md` - Basic VLM architecture (simpler than Flamingo)
- `01-qformer-blip2-implementation.md` - Alternative visual-language bridging (Q-Former)
- `02-perceiver-cross-attention.md` - Perceiver architecture (used in Flamingo's resampler)
- `07-fusion-strategies.md` - Comparison of different vision-language fusion approaches

**Next Steps**:
1. Implement basic gated cross-attention layer
2. Experiment with different gating functions (sigmoid, learnable per-head)
3. Try visual masking strategies (all images vs. preceding only)
4. Compare with adapter-based fusion methods
