# Phase P1: VET Pre-training

**Category**: Training
**Related**: [00-overview.md](00-overview.md), [../architecture/03-visual-embedding-table.md](../architecture/03-visual-embedding-table.md)

## Goal

Initialize Visual Embedding Table (VET) with meaningful visual vocabulary before full multimodal training.

## Configuration

**Trainable**:
- Visual Tokenizer (partial ViT layers)
- Visual Embedding Table (VET)
- Visual Head

**Frozen**:
- Most ViT layers
- Entire LLM (Qwen3)

## Data

**Type**: Image-caption pairs
**Size**: ~100M examples
**Format**: `(image, caption_text)`

**Sources**:
- CC12M (Conceptual Captions)
- LAION subsets
- Web-scraped image-text pairs

## Resolution

**Range**: 448²-896²
**RoPE**: Disabled (activated in P2)
**Why lower res**: Faster VET initialization

## Training

```yaml
epochs: 1
learning_rate: 5e-5
batch_size: 1280
optimizer: AdamW
scheduler: Cosine with warmup
warmup_steps: 5000
```

## What Happens

1. Image → ViT → visual features
2. Visual head → probabilities
3. VET lookup → embeddings
4. LLM generates caption
5. Loss on caption tokens only
6. Gradients update VET + visual_head

**Result**: VET learns visual concepts, visual_head learns to generate meaningful distributions.

## Related

- [00-overview.md](00-overview.md)
- [../architecture/03-visual-embedding-table.md](../architecture/03-visual-embedding-table.md)
