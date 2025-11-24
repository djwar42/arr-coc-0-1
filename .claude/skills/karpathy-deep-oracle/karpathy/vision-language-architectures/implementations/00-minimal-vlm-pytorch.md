# Minimal Vision-Language Model (VLM) in PyTorch

## Overview

A minimal Vision-Language Model (VLM) combines visual understanding with natural language processing in the simplest possible architecture. This guide covers building a VLM from scratch in pure PyTorch, focusing on educational clarity over production optimization.

**Core Architecture**: Image Encoder + Projection Layer + Text Decoder

**Typical Size**: 200M-300M parameters (e.g., 85M vision encoder + 135M language decoder + projection)

**Training Time**: ~3-6 hours on single H100 GPU for basic competence

## Three Essential Components

### 1. Vision Encoder (Image → Features)
Extracts visual features from images into dense embeddings.

**Common Choices**:
- **Vision Transformer (ViT)** - Most popular (used in CLIP, BLIP, LLaVA)
- **ResNet/ConvNet** - Simpler alternative for quick prototyping
- **SigLip Vision Encoder** - Improved numerical stability over CLIP

### 2. Multimodal Projection (Vision → Language Space)
Projects vision features into the language model's embedding space.

**Implementation Options**:
- **Single Linear Layer** - Simplest approach
- **MLP (2-3 layers)** - More expressive, commonly used
- **Perceiver Resampler** - Advanced (reduces token count)

### 3. Language Decoder (Generate Text)
Autoregressive model that generates text conditioned on visual features.

**Common Choices**:
- **GPT-2/GPT-Neo** - Popular for prototypes
- **LLaMA/Gemma** - Modern lightweight decoders
- **SmolLM** - Tiny efficient models (135M params)

## Minimal PyTorch Implementation

### Vision Encoder: Simple ViT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    """Minimal Vision Transformer for image encoding"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding: Convert image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]

        # Patchify image
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x  # (B, num_patches, embed_dim)

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention + MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x
```

### Multimodal Projection Layer

```python
class MultimodalProjection(nn.Module):
    """Project vision features to language embedding space"""
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=2048):
        super().__init__()
        # Simple MLP projection
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, language_dim)
        )

    def forward(self, vision_features):
        # vision_features: (B, num_patches, vision_dim)
        # Output: (B, num_patches, language_dim)
        return self.proj(vision_features)
```

### Language Decoder: Minimal GPT

```python
class LanguageDecoder(nn.Module):
    """Minimal GPT-style decoder for text generation"""
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim)
        )

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, visual_tokens=None):
        # input_ids: (B, seq_len)
        B, T = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, T, embed_dim)
        x = x + self.pos_embed[:, :T, :]

        # Prepend visual tokens if provided
        if visual_tokens is not None:
            x = torch.cat([visual_tokens, x], dim=1)

        # Apply decoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, seq_len, vocab_size)
        return logits

class DecoderBlock(nn.Module):
    """Decoder transformer block with causal attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CausalSelfAttention(nn.Module):
    """Causal self-attention with masking for autoregressive generation"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
```

### Complete VLM Model

```python
class MinimalVLM(nn.Module):
    """Complete minimal Vision-Language Model"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        vocab_size=50257,
        vision_dim=768,
        language_dim=768
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_dim
        )
        self.projection = MultimodalProjection(
            vision_dim=vision_dim,
            language_dim=language_dim
        )
        self.language_decoder = LanguageDecoder(
            vocab_size=vocab_size,
            embed_dim=language_dim
        )

    def forward(self, images, input_ids):
        # Encode images
        vision_features = self.vision_encoder(images)

        # Project to language space
        visual_tokens = self.projection(vision_features)

        # Generate text
        logits = self.language_decoder(input_ids, visual_tokens)
        return logits

    @torch.no_grad()
    def generate(self, image, prompt_ids, max_new_tokens=50, temperature=1.0):
        """Autoregressive text generation"""
        self.eval()

        # Encode image once
        vision_features = self.vision_encoder(image.unsqueeze(0))
        visual_tokens = self.projection(vision_features)

        # Start with prompt
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.language_decoder(generated, visual_tokens)

            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == 50256:  # EOS token for GPT-2
                break

        return generated
```

## Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_vlm(model, train_loader, num_epochs=10, learning_rate=3e-4):
    """Simple training loop for VLM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, input_ids, target_ids) in enumerate(train_loader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits = model(images, input_ids)

            # Compute loss (shift for autoregressive prediction)
            loss = criterion(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                target_ids[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

# Example usage
if __name__ == '__main__':
    # Create model
    model = MinimalVLM(
        img_size=224,
        patch_size=16,
        vocab_size=50257,
        vision_dim=768,
        language_dim=768
    )

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {num_params/1e6:.1f}M')

    # Train (assuming you have a DataLoader)
    # train_vlm(model, train_loader, num_epochs=10)
```

## Key Design Decisions

### Vision Encoder Choice

**ViT (Vision Transformer)**:
- Used in: CLIP, BLIP, LLaVA, PaliGemma
- Pros: Strong performance, self-attention captures global context
- Cons: Requires more data, ~85M parameters for base model

**ResNet/ConvNet**:
- Pros: Simpler, faster training, good for prototypes
- Cons: Less expressive, local receptive fields only

**Recommendation**: Start with ViT for best alignment with modern VLMs

### Projection Layer Design

**Single Linear Layer**:
```python
self.proj = nn.Linear(vision_dim, language_dim)
```
- Simplest, works surprisingly well
- Used in early CLIP implementations

**MLP (2-3 layers)**:
```python
self.proj = nn.Sequential(
    nn.Linear(vision_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, language_dim)
)
```
- More expressive, better alignment
- Used in BLIP, LLaVA, most modern VLMs

**Recommendation**: Use 2-layer MLP for better performance

### Text Decoder Choice

**For Learning**:
- Character-level decoder (like nanoGPT) - simplest
- Small GPT-2 (124M params) - good balance

**For Production**:
- GPT-2 Medium/Large (355M-774M) - better quality
- LLaMA/Gemma variants - modern, efficient
- SmolLM (135M) - tiny but capable

## Optimization Techniques

### 1. Gradient Checkpointing
Reduces memory usage during training:
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x)  # Save memory
    return x
```

### 2. Mixed Precision Training
Speeds up training on modern GPUs:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(images, input_ids)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Freeze Vision Encoder
For faster finetuning on limited data:
```python
# Freeze vision encoder parameters
for param in model.vision_encoder.parameters():
    param.requires_grad = False

# Only train projection + decoder
trainable_params = (
    list(model.projection.parameters()) +
    list(model.language_decoder.parameters())
)
optimizer = optim.AdamW(trainable_params, lr=3e-4)
```

## Common Pitfalls & Solutions

### Problem 1: Visual Tokens Ignored
**Symptom**: Model generates text unrelated to image

**Solution**: Check that visual tokens are properly integrated
```python
# BAD: Visual tokens concatenated but never attended to
x = torch.cat([visual_tokens, text_tokens], dim=1)

# GOOD: Ensure causal mask allows attending to visual prefix
# Visual tokens should be at the START of sequence
visual_tokens = self.projection(vision_features)  # (B, N_vis, D)
text_embed = self.token_embed(input_ids)  # (B, N_text, D)
x = torch.cat([visual_tokens, text_embed], dim=1)  # Visual first!
```

### Problem 2: Training Instability
**Symptom**: Loss explodes or NaN values

**Solution**: Apply proper normalization
```python
# Use LayerNorm before attention and MLP
x = x + self.attn(self.norm1(x))  # Pre-norm
x = x + self.mlp(self.norm2(x))

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Problem 3: Poor Generation Quality
**Symptom**: Repetitive or incoherent text

**Solution**: Use better sampling strategies
```python
def sample_top_p(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumsum_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()

    return torch.multinomial(probs, num_samples=1)
```

## VRAM Requirements

**Model Size**: ~222M parameters (85M vision + 135M language + 2M projection)

**Training VRAM** (single H100):
- Batch size 1: ~4.5 GB
- Batch size 8: ~5 GB
- Batch size 16: ~7 GB
- Batch size 32: ~11 GB

**Inference VRAM**:
- Single image + generation: ~1.5 GB

**Memory-Efficient Training**:
```python
# Enable gradient checkpointing
model.vision_encoder.gradient_checkpointing = True

# Use bfloat16 mixed precision
with torch.autocast('cuda', dtype=torch.bfloat16):
    logits = model(images, input_ids)
```

## Real-World Example: Image Captioning

```python
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer

# Load pretrained components
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = MinimalVLM(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('vlm_checkpoint.pt'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('cat.jpg')
image_tensor = transform(image)

# Generate caption
prompt = "A photo of"
prompt_ids = torch.tensor([tokenizer.encode(prompt)])

generated_ids = model.generate(
    image_tensor,
    prompt_ids,
    max_new_tokens=50,
    temperature=0.7
)

caption = tokenizer.decode(generated_ids[0])
print(f"Caption: {caption}")
# Output: "A photo of a tabby cat sitting on a wooden floor."
```

## Sources

**GitHub Implementations**:
- [nanoVLM](https://github.com/juyoungml/nanoVLM) - Minimal 222M param VLM in pure PyTorch (~750 lines)
- [seemore](https://github.com/AviSoori1x/seemore) - Educational VLM with ViT encoder from scratch
- [VisualLanguageModel](https://github.com/aiishwarrya/VisualLanguageModel) - Custom VLM using SigLip contrastive learning

**Tutorials**:
- "Coding a Multimodal (Vision) Language Model from scratch" - Umar Jamil (YouTube, accessed 2025-01-31)
- "Implementation of Vision language models (VLM) from scratch" - Medium (accessed 2025-01-31)
- PyTorch Vision Language Model Tutorial (various, accessed 2025-01-31)

**Papers**:
- "Visual Instruction Tuning" (LLaVA) - https://arxiv.org/pdf/2304.08485.pdf
- "BLIP: Bootstrapping Language-Image Pre-training" - https://arxiv.org/pdf/2201.12086
- "Large Multimodal Models: Notes on CVPR 2023 Tutorial" - https://arxiv.org/pdf/2306.14895.pdf

**Web Research** (accessed 2025-01-31):
- Hugging Face nanoVLM blog: https://huggingface.co/blog/nanovlm
- Building Vision Language Models tutorial: https://sachinruk.github.io/blog/2024-08-11-vision-language-models.html
- Simple OpenAI CLIP implementation: Towards Data Science
