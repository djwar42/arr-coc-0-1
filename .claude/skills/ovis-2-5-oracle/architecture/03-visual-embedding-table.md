# Visual Embedding Table (VET)

**Category**: Architecture
**Related**: [00-overview.md](00-overview.md), [02-visual-tokenizer.md](02-visual-tokenizer.md)
**Code**: `ovis/model/modeling_ovis.py:25-34`

## Overview

The Visual Embedding Table (VET) is Ovis's **core innovation** - a learnable table of discrete visual embeddings that enables **structural alignment** between visual and textual representations.

**Key Insight**: Instead of using continuous visual features (standard approach), Ovis uses probabilistic lookups into a discrete embedding table, making visual tokens structurally similar to text tokens.

## Core Concept

### Standard Approach (Continuous)

```python
# Most VLMs
visual_features = vit_encoder(image)  # [B, N, 768]
visual_embeddings = projection(visual_features)  # [B, N, D_llm]
# Problem: Continuous features, structurally different from discrete text tokens
```

### Ovis Approach (Probabilistic Discrete)

```python
# Ovis
visual_features = vit_encoder(image)  # [B, N, 768]
probabilities = visual_head(visual_features)  # [B, N, vocab_size]
visual_embeddings = probabilities @ embedding_table  # [B, N, D_llm]
# Solution: Probabilistic weighted sum of discrete embeddings
```

## Implementation

**File**: `ovis/model/modeling_ovis.py:25-34`

```python
class VisualEmbedding(nn.Embedding):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probabilistic lookup in embedding table

        Args:
            x: Probability distribution [batch, num_patches, vocab_size]

        Returns:
            Embeddings [batch, num_patches, embedding_dim]
        """
        if x.dtype in [torch.int32, torch.int64]:
            # Standard discrete lookup (fallback)
            return F.embedding(x, self.weight)

        # Probabilistic lookup (Ovis way)
        return torch.matmul(x, self.weight)  # [B, N, V] @ [V, D] = [B, N, D]
```

## Mathematical Formulation

### Discrete Lookup (Standard Text)

```
text_token_id = 42
embedding = embedding_table[42]  # One-hot: [0,...,0,1,0,...,0]
```

### Probabilistic Lookup (Ovis Vision)

```
probabilities = [0.05, 0.15, 0.60, 0.15, 0.05, ...]  # Softmax output
embedding = Σᵢ (pᵢ × embedding_table[i])

= 0.05 × emb[0] + 0.15 × emb[1] + 0.60 × emb[2] + ...

# Weighted sum of discrete embeddings
```

## Why VET Matters

### 1. Structural Alignment

**Problem**: LLMs are trained on discrete text tokens
- Text: Discrete embeddings from token IDs
- Vision (standard): Continuous feature vectors
- **Mismatch**: Different data types, different semantic spaces

**Solution**: Make vision discrete (but soft)
- Text: `embedding_table[token_id]`
- Vision: `probabilities @ embedding_table`
- **Aligned**: Both use same embedding table structure

### 2. Improved Cross-Modal Learning

**Standard VLMs**:
```
Text path:   token_id → embedding_table[id] → LLM
Vision path: features → projection       → LLM
```
Two different pathways, hard to align semantically.

**Ovis**:
```
Text path:   token_id → one_hot → embedding_table → LLM
Vision path: features → softmax → embedding_table → LLM
```
Same embedding table, structurally similar inputs.

### 3. Better Training Dynamics

**Continuous features**:
- Gradients flow through projection layer
- Vision space separate from text space
- Harder to learn cross-modal correspondences

**Probabilistic discrete**:
- Gradients flow through embedding table
- Vision uses same discrete space as text
- Clearer semantic alignment signals

## Training VET

### Phase P1: VET Pre-training

**Goal**: Initialize Visual Embedding Table

**Setup**:
- **Trainable**: VT (partial), VET, Visual Head
- **Frozen**: Most ViT, all LLM
- **Data**: Image-caption pairs
- **Loss**: Next token prediction

**Why This Phase?**
- Learn meaningful visual vocabulary
- Align visual embeddings with LLM's text space
- Initialize before full multimodal training

**Process**:
```
1. Image → ViT → visual features
2. Visual head → probabilities
3. VET lookup → embeddings
4. LLM forward → predictions
5. Loss on caption tokens
6. Gradients update VET, visual head
```

### Phase P2+: Full Training

VET continues to adapt during full multimodal training while learning alongside CLIP and LLM.

## VET Configuration

**From `configuration_ovis.py`**:

```python
visual_vocab_size = 16384  # Size of VET (number of discrete embeddings)
hidden_size = 1280          # Embedding dimension (matches Qwen3)
```

**Visual Vocabulary**:
- 16,384 discrete embeddings
- Each embedding: 1280-dimensional vector
- Total parameters: 16,384 × 1280 = ~21M params

## Probability Distribution Quality

### Sharp vs Soft Distributions

**Sharp (confident)**:
```
probabilities = [0.01, 0.02, 0.90, 0.05, 0.02]
# Model is confident about specific embedding
```

**Soft (uncertain)**:
```
probabilities = [0.18, 0.22, 0.25, 0.20, 0.15]
# Model hedges, blends multiple embeddings
```

**Impact**:
- Sharp → More discrete (closer to text tokens)
- Soft → More continuous (smoother gradients)
- Model learns optimal sharpness during training

## Comparison to Standard Approaches

| Approach | Vision Representation | Alignment | Training |
|----------|----------------------|-----------|----------|
| **Standard VLM** | Continuous features | Projection layer | Separate spaces |
| **BLIP-2 Q-Former** | Learned queries | Cross-attention | Complex architecture |
| **Flamingo** | Continuous + cross-attn | Gated layers | Many parameters |
| **Ovis (VET)** | Probabilistic discrete | Shared embedding table | Structurally aligned |

## VET vs Traditional Codebooks

### Vector Quantization (VQ-VAE style)

```python
# Traditional codebook (hard assignment)
codes = argmax(similarities)  # Discrete indices
embeddings = codebook[codes]  # Hard lookup
# Problem: Non-differentiable argmax
```

### Ovis VET (soft assignment)

```python
# Ovis (soft assignment)
probabilities = softmax(logits)  # Soft distribution
embeddings = probabilities @ VET  # Soft lookup
# Solution: Fully differentiable
```

**Key Difference**: Ovis uses soft probabilistic lookup, not hard vector quantization.

## Interpretability

### Analyzing VET Assignments

```python
# Get probability distribution
probabilities = visual_head(visual_features)  # [B, N, 16384]

# Find most confident patches
confidence = probabilities.max(dim=-1)  # [B, N]
top_patches = torch.topk(confidence, k=10)

# Find most used embeddings
usage = probabilities.sum(dim=(0, 1))  # [16384]
top_embeddings = torch.topk(usage, k=100)

# Analyze: Which visual patterns map to which embeddings?
```

## Performance Impact

### With VET (Ovis)
- **Alignment**: Excellent cross-modal understanding
- **Training**: Stable, clear gradients
- **Inference**: Fast (single matmul)
- **Memory**: Modest (21M extra params)

### Without VET (ablation)
- **Alignment**: Weaker cross-modal understanding
- **Training**: Less stable, noisier gradients
- **Performance**: Lower on multimodal benchmarks

**Conclusion**: VET is essential for Ovis's strong performance.

## Related Topics

- [00-overview.md](00-overview.md) - System architecture
- [02-visual-tokenizer.md](02-visual-tokenizer.md) - Generating probabilities
- [../concepts/00-structural-alignment.md](../concepts/00-structural-alignment.md) - Theory deep dive
- [../concepts/01-probabilistic-vte.md](../concepts/01-probabilistic-vte.md) - Mathematical details
- [../training/01-phase-p1-vet.md](../training/01-phase-p1-vet.md) - Training VET

## Code References

**VET Class**: `ovis/model/modeling_ovis.py:25-34`
**Config**: `ovis/model/configuration_ovis.py` - `visual_vocab_size`, `hidden_size`
**Training**: `ovis/train/train.py` - Phase P1 setup
