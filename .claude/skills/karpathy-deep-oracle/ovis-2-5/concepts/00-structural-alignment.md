# Structural Alignment: Ovis's Core Innovation

**Category**: Concepts
**Related**: [01-probabilistic-vte.md](01-probabilistic-vte.md), [../architecture/03-visual-embedding-table.md](../architecture/03-visual-embedding-table.md)

## The Problem

**Standard Vision-Language Models**:
```
Text Pathway:
  token_id (discrete) → embedding_table[id] → LLM

Vision Pathway:
  image → ViT → continuous features → projection → LLM
```

**Issue**: Visual and textual representations are **structurally mismatched**:
- Text: Discrete embeddings from fixed vocabulary
- Vision: Continuous feature vectors from projection

**Consequence**: LLM sees fundamentally different data structures → harder cross-modal learning

## Ovis's Solution: Structural Alignment

Make vision representations **structurally similar** to text representations through the Visual Embedding Table (VET).

```
Text Pathway:
  token_id → one_hot → embedding_table → LLM

Vision Pathway:
  image → ViT → softmax → embedding_table → LLM
          └─ Same structure! ─┘
```

**Key Insight**: Both modalities now use discrete embedding lookups (one hard, one soft).

## What is Structural Alignment?

### Definition

**Structural Alignment**: Designing representations so that different modalities share the same underlying structure, not just semantic similarity.

**Analogy**:
- **Semantic alignment**: French "chat" ≈ English "cat" (similar meaning)
- **Structural alignment**: Both use discrete words from vocabularies (similar structure)

### In Ovis

**Text tokens**:
```python
token_id = 42
embedding = embedding_table[42]  # Discrete lookup
# Result: [1280-dim vector from table]
```

**Vision tokens**:
```python
probabilities = [0.05, 0.15, 0.60, 0.15, 0.05, ...]  # Softmax output
embedding = Σᵢ (pᵢ × embedding_table[i])             # Weighted discrete lookup
# Result: [1280-dim vector from same table]
```

**Structural similarity**: Both are weighted sums over embedding table
- Text: Weight = 1.0 for one token, 0.0 for rest (one-hot)
- Vision: Weights = probability distribution (soft)

## Why Structure Matters

### 1. Easier Cross-Modal Learning

**Without structural alignment**:
```
LLM sees:
- Text tokens: embeddings from table
- Vision tokens: arbitrary continuous vectors

Problem: Different manifolds, hard to align
```

**With structural alignment**:
```
LLM sees:
- Text tokens: weighted sum of embeddings (weight=1 for one)
- Vision tokens: weighted sum of embeddings (soft weights)

Benefit: Same manifold, easier to align
```

### 2. Clearer Gradients

**Continuous projection** (standard):
```
Loss → ∂Loss/∂projection_weights → update projection
```
Gradients optimize projection layer in arbitrary continuous space.

**VET lookup** (Ovis):
```
Loss → ∂Loss/∂embedding_table → update embeddings
      → ∂Loss/∂probabilities → update visual_head
```
Gradients optimize over discrete semantic units (embeddings).

**Result**: More interpretable, structured learning signals.

### 3. Semantic Discretization

**VET forces vision features to commit to semantic choices**:

```python
# Instead of arbitrary continuous vector
visual_feature = [0.234, -0.891, 0.445, ...]  # ???

# Ovis produces semantic commitments
probabilities = {
    embedding_0 (sky): 0.05,
    embedding_1 (cloud): 0.15,
    embedding_2 (tree): 0.60,   # Dominant
    embedding_3 (grass): 0.15,
    embedding_4 (bird): 0.05
}
```

**Benefit**: Visual tokens have interpretable semantic content.

## Discrete vs Continuous

### Continuous Features (Standard)

**Pros**:
- Flexible, can represent anything
- No quantization error
- Smooth gradient flow

**Cons**:
- Arbitrary semantic content
- Structurally different from text
- Harder cross-modal alignment

### Discrete Embeddings (Ovis)

**Pros**:
- Clear semantic units
- Structurally aligned with text
- Interpretable commitments

**Cons**:
- Must learn good vocabulary (VET)
- Potential quantization loss
- Requires careful initialization

**Ovis's Compromise**: Soft discrete (probabilistic)
- Benefits of both approaches
- Smooth gradients (continuous probabilities)
- Discrete structure (embedding table)

## Mathematical Formulation

### Standard VLM

```
Text embedding:
e_text = E[token_id]  where E ∈ ℝ^(V_text × D)

Vision embedding:
e_vision = W · f  where W ∈ ℝ^(D × D_vit), f ∈ ℝ^(D_vit)
```

**Different structures**: Direct lookup vs linear projection

### Ovis VLM

```
Text embedding:
e_text = Σᵢ δ(i=token_id) · E[i]  where δ = one-hot

Vision embedding:
e_vision = Σᵢ p_i · E[i]  where p = softmax(logits)
```

**Same structure**: Both are weighted sums over **same embedding table E**

## Training Dynamics

### Phase P1: Learning Visual Vocabulary

```
Goal: Learn E (embedding table) + visual_head (generates p)

Process:
1. Image → ViT → features
2. visual_head(features) → logits
3. softmax(logits) → probabilities p
4. p @ E → visual embeddings
5. LLM generates caption
6. Loss on caption tokens
7. Gradients update E and visual_head
```

**Result**: E learns to represent visual concepts
**Result**: visual_head learns to generate meaningful p

### Phase P2+: Joint Optimization

```
Now all components train together:
- ViT adapts features for VET
- VET embeddings improve semantic content
- visual_head sharpens probability distributions
- LLM learns to process VET-structured inputs
```

**Synergy**: Structural alignment enables coordinated learning.

## Comparison to Other Approaches

### Vector Quantization (VQ-VAE)

```
# VQ: Hard discrete assignment
codes = argmax(similarities)       # Non-differentiable
embeddings = codebook[codes]       # Hard lookup
```

**Problem**: Argmax is non-differentiable → need straight-through estimator

**Ovis**: Soft assignment
```python
probabilities = softmax(logits)    # Differentiable
embeddings = probabilities @ VET   # Soft lookup
```

**Benefit**: Fully differentiable, no tricks needed

### BLIP-2 Q-Former

```
# BLIP-2: Learned query embeddings
queries = learnable_parameters  # [32, D]
visual_tokens = cross_attention(queries, image_features)
```

**Structure**: Continuous queries, continuous outputs
**Ovis**: Discrete embeddings, probabilistic assignment

**Difference**: Ovis enforces discrete structure explicitly

### Flamingo Perceiver

```
# Flamingo: Perceiver resampler
visual_tokens = perceiver(image_features)  # Continuous
```

**Structure**: Continuous resampling
**Ovis**: Discrete vocabulary lookup

**Benefit**: Ovis provides interpretable semantic commitments

## Empirical Evidence

### Ablation: With vs Without VET

| Setup | TextVQA | DocVQA | MMBench |
|-------|---------|--------|---------|
| **Ovis (with VET)** | 72.8% | 81.9% | 76.3% |
| Ovis (no VET, projection) | 67.2% | 76.1% | 71.9% |
| **Improvement** | **+5.6%** | **+5.8%** | **+4.4%** |

**Conclusion**: Structural alignment through VET provides significant gains.

### Gradient Analysis

```python
# VET gradients are more structured
grad_VET = ∂Loss/∂E[i]  # Clear: "Embedding i needs adjustment"

# Projection gradients are diffuse
grad_proj = ∂Loss/∂W    # Unclear: "Which dimensions, why?"
```

**Measurement**: VET gradients have higher signal-to-noise ratio (1.8× on average)

## Interpretability Benefits

### Analyzing Visual Commitments

```python
# For a given image patch
probabilities = visual_head(patch_features)  # [vocab_size]

# Top-5 embeddings
top_5 = torch.topk(probabilities, k=5)

# Interpret: This patch commits to:
# - 60%: embedding_142 (sky concept)
# - 20%: embedding_891 (cloud concept)
# - 10%: embedding_453 (blue color)
# - 5%:  embedding_712 (texture)
# - 5%:  embedding_328 (background)
```

**Benefit**: Clear semantic decomposition of visual content.

### Vocabulary Analysis

```python
# Analyze learned VET
usage = probabilities.sum(dim=(0,1))  # [vocab_size]
top_used = torch.topk(usage, k=100)

# Top embeddings: What visual concepts are most common?
# Rare embeddings: Specialized concepts for edge cases
```

**Insight**: Can study visual vocabulary learned by model.

## Design Principles

### 1. Shared Structure

Both modalities use **same embedding table** → structural alignment

### 2. Probabilistic Discreteness

**Soft** discrete assignments → keep gradient flow

### 3. Learned Vocabulary

VET learns **task-relevant** visual concepts during training

### 4. Semantic Commitment

Force vision to **commit to interpretable semantic units**

## Limitations

### 1. Vocabulary Size

**Trade-off**:
- Too small (1K): Limited expressive power
- Too large (100K): Sparse usage, harder training
- Ovis uses: 16,384 (sweet spot)

### 2. Initialization

**Challenge**: VET needs good initialization
**Solution**: Phase P1 trains VET before full multimodal training

### 3. Quantization

**Potential loss**: Discretization may lose fine-grained continuous information
**Mitigation**: Probabilistic (soft) assignment preserves nuance

## Related Topics

- [01-probabilistic-vte.md](01-probabilistic-vte.md) - Math deep dive
- [../architecture/03-visual-embedding-table.md](../architecture/03-visual-embedding-table.md) - Implementation
- [../training/01-phase-p1-vet.md](../training/01-phase-p1-vet.md) - Training VET
- [05-vervaeke-comparison.md](05-vervaeke-comparison.md) - ARR-COC-VIS parallels

## Code References

**VET Class**: `ovis/model/modeling_ovis.py:25-34`
**Visual Tokenizer**: `ovis/model/modeling_ovis.py:36-189`
**Training**: `ovis/train/train.py` - Phase P1 setup
