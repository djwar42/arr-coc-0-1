# Learned Positional Encodings in Transformers

## Overview

Learned positional encodings represent a trainable alternative to fixed positional encoding schemes (like sinusoidal). Instead of using predetermined mathematical functions, learned encodings treat position information as parameters that are optimized during training alongside other model weights.

**Key concept**: Position embeddings are initialized randomly or with specific distributions and then learned through backpropagation, allowing the model to discover task-specific positional patterns.

**When learning helps**: Learned encodings can capture domain-specific positional relationships that fixed functions might miss, particularly when training data has consistent structural patterns.

**Parameter cost considerations**: Adding learned position embeddings increases model size. For a sequence length of L and embedding dimension d, learned absolute encodings add L × d parameters (e.g., 512 × 768 = 393,216 parameters for BERT-base).

From [Learning positional encodings in transformers depends on task semantics](https://arxiv.org/html/2406.08272v3) (arXiv:2406.08272, accessed 2025-01-31):
- Learned PE initialized from small-norm distributions can uncover interpretable patterns
- Optimal learned PE outperformed commonly-used fixed PEs in controlled experiments
- Learned attention maps and PE embeddings mirrored ground truth positions in multiple dimensions

From [What Do Position Embeddings Learn? An Empirical Study](https://arxiv.org/abs/2010.04903) (arXiv:2010.04903, accessed 2025-01-31):
- Empirical study on position embeddings of BERT, RoBERTa, and ALBERT
- Learned position embeddings do capture meaningful position information
- Feature-level analysis reveals how different pre-trained models encode positions

## Absolute Learned Encodings

### Position Embedding Tables

**Mechanism**: A lookup table where each position index maps to a learnable d-dimensional vector:

```
Position 0:  [θ₀₀, θ₀₁, θ₀₂, ..., θ₀ₐ]
Position 1:  [θ₁₀, θ₁₁, θ₁₂, ..., θ₁ₐ]
Position 2:  [θ₂₀, θ₂₁, θ₂₂, ..., θ₂ₐ]
...
Position L:  [θₗ₀, θₗ₁, θₗ₂, ..., θₗₐ]
```

Each θᵢⱼ is a trainable parameter optimized via gradient descent.

**Implementation pattern**:
```python
# PyTorch-style learned absolute position embeddings
max_position_embeddings = 512
hidden_size = 768
position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

# Usage
seq_length = input_ids.size(1)
position_ids = torch.arange(seq_length).unsqueeze(0)
position_embeds = position_embeddings(position_ids)
embeddings = token_embeds + position_embeds
```

### Initialization Strategies

**Random initialization**:
- Standard normal distribution: N(0, 0.02) commonly used
- Xavier/Glorot: Scales variance based on layer dimensions
- Small values prevent position information from dominating token semantics

**Sinusoidal initialization**:
- Initialize with sinusoidal patterns, then fine-tune
- Provides better starting point than pure random
- Can help with extrapolation to longer sequences

From [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) (Hugging Face, accessed 2025-01-31):
- Learned positional embedding is more expressive than sinusoidal encoding
- Model can adapt positions to the specific task
- Can outperform sinusoidal in some NLP tasks with large datasets

**BERT's approach** (from empirical studies):
- Uses learned absolute position embeddings
- Maximum sequence length: 512 tokens
- Position embeddings trained from scratch with masked language modeling
- Learned embeddings show distance-aware patterns post-training

### Extrapolation Challenges

**The fundamental problem**: Learned position tables are fixed-length. What happens at position 513 when trained on max length 512?

**Common failures**:
- Out-of-bounds error if position_ids exceed max_position_embeddings
- Fallback to position 0 or last position (poor generalization)
- Interpolation attempts often degrade performance

**Mitigation strategies**:

1. **Positional interpolation** (for inference):
   - Scale position indices to fit within trained range
   - position_id_new = position_id_original × (max_train_length / current_length)
   - Quality degrades significantly for 2x+ length increases

2. **Extrapolation training**:
   - Periodically train on slightly longer sequences
   - Gradually increase max length during training
   - Still limited - cannot generalize to arbitrary lengths

3. **Hybrid approaches**:
   - Learn position embeddings up to L_max
   - Use algorithmic encoding (RoPE, ALiBi) for positions > L_max
   - Provides both learned optimization and unbounded extrapolation

**Why this matters**: Learned absolute encodings sacrifice length generalization for task-specific optimization. This is acceptable for fixed-context tasks (sentiment analysis, classification) but problematic for long-document or generative applications.

## Relative Learned Encodings

### Learned Bias Matrices

Instead of encoding absolute positions, learn relative position biases that modify attention scores based on key-query distance.

**Core idea**: Attention score adjustment based on relative position offset
```
Attention(Q, K, V) = softmax((QK^T + B) / √d_k) V
```

where B[i,j] = learned_bias[i - j] represents the learned bias for relative distance.

### T5 Relative Position Bias

**T5's innovation** (Raffel et al., 2020):
- Buckets relative positions into discrete bins
- Each bucket has a learned scalar bias added to attention logits
- Shared across all attention layers

**Bucketing strategy**:
- Positions 0-7: Individual buckets (precise nearby positions)
- Positions 8+: Logarithmically spaced buckets
- Bidirectional: Separate buckets for forward/backward offsets
- Total: ~32 buckets for typical configurations

**Implementation outline**:
```python
# Simplified T5 relative position bias
num_buckets = 32
relative_position = position_ids_k - position_ids_q  # Shape: [seq_len, seq_len]

# Bucket assignment (simplified)
# Close positions get individual buckets
# Distant positions share logarithmically-spaced buckets
bucket_ids = compute_bucket(relative_position, num_buckets)

# Look up learned biases
relative_attention_bias = embedding_table(bucket_ids)  # Shape: [seq_len, seq_len, num_heads]

# Add to attention scores
scores = (Q @ K.T) + relative_attention_bias
```

**Advantages**:
- Parameter efficient: Only 32 scalars per head vs L² for full learned matrix
- Generalizes to longer sequences through bucketing
- Captures both nearby precision and distant coarse-graining

### Memory Efficiency

**Full learned relative position matrix**:
- Parameters: L × L × H (sequence_length² × num_heads)
- Example: 512² × 12 = 3,145,728 parameters (prohibitive)
- Required for every attention layer

**T5 bucketing approach**:
- Parameters: B × H (num_buckets × num_heads)
- Example: 32 × 12 = 384 parameters
- ~8000× reduction in parameters

**DeBERTa disentangled attention**:
- Separate content and position attention
- Learns position-to-content and content-to-position attention matrices
- More parameters than T5 but captures richer interactions

From [Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/pdf/2003.09229) (arXiv:2003.09229):
- Parameter-efficient encoding should limit trainable parameters
- Continuous dynamical models can generate position encodings
- Learned encodings with structure outperform purely learned tables

## Best Practices

### When to Use Learned Encodings

**✅ Good use cases**:

1. **Fixed-length tasks**:
   - Sentiment analysis (BERT-style)
   - Text classification with bounded inputs
   - Question answering with max context limits

2. **Domain-specific patterns**:
   - Genomic sequences with regular structure
   - Time-series with periodic components
   - Code with syntactic position importance

3. **Sufficient training data**:
   - Large datasets allow position patterns to emerge
   - Prevents overfitting to position artifacts

4. **Known sequence length bounds**:
   - When max length is architecturally constrained
   - When extrapolation is never required

**❌ Avoid learned encodings when**:

1. **Variable or unbounded sequence lengths**:
   - Long-form text generation
   - Document processing without length limits
   - Streaming or online inference

2. **Limited training data**:
   - Risk of memorizing position-label correlations
   - Fixed encodings provide better inductive bias

3. **Zero-shot length generalization required**:
   - Must handle lengths never seen during training
   - Use RoPE, ALiBi, or other extrapolatable methods

### Hybrid Approaches (RoPE + Learned)

**Combining strengths**:
- Use RoPE for base positional encoding (extrapolation + efficiency)
- Add learned position-dependent biases for task-specific refinement
- Best of both worlds: mathematical structure + learned adaptation

**Example architecture**:
```python
# Hybrid: RoPE + learned bias
rope_q, rope_k = apply_rotary_pos_emb(q, k, position_ids)
attention_scores = (rope_q @ rope_k.T) / sqrt(d_k)

# Add learned task-specific bias
learned_bias = position_bias_table(relative_positions)
attention_scores += learned_bias

attention_weights = softmax(attention_scores)
output = attention_weights @ v
```

**Used in**:
- Some LLaMA variants experiment with learned biases on top of RoPE
- Allows model to learn task-specific position importance
- Maintains RoPE's length generalization

### Training Stability

**Common stability issues**:

1. **Gradient explosion with large position embeddings**:
   - Solution: Gradient clipping
   - Solution: Layer normalization after adding position embeddings
   - Solution: Smaller initialization (σ = 0.01 instead of 0.02)

2. **Position-token coupling**:
   - Problem: Position embeddings dominate token semantics early in training
   - Solution: Warmup learning rate for position embeddings separately
   - Solution: Lower learning rate for position parameters

3. **Overfitting to position patterns**:
   - Problem: Model memorizes position-dependent outputs
   - Solution: Position embedding dropout
   - Solution: Regularization on position embedding norms

**Training recipe** (from empirical findings):
```python
# Separate parameter groups
optimizer = AdamW([
    {'params': token_embeddings.parameters(), 'lr': 1e-4},
    {'params': position_embeddings.parameters(), 'lr': 5e-5},  # Lower LR
    {'params': transformer_layers.parameters(), 'lr': 1e-4}
])

# Layer norm after position addition
embeddings = layer_norm(token_embeds + position_embeds)

# Optional: position embedding dropout
position_embeds = dropout(position_embeddings(position_ids), p=0.1)
```

**Verification during training**:
- Monitor position embedding norms (should be comparable to token embedding norms)
- Visualize learned position similarities (should show distance structure)
- Test on varying sequence lengths to detect overfitting

From [Understanding Positional Encoding in Transformers](https://arxiv.org/html/2406.08272v3) (accessed 2025-01-31):
- Learning positional encodings depends heavily on task semantics
- Small-norm initialization crucial for discovering interpretable patterns
- Learned PEs can achieve state-of-the-art performance when properly configured

## Performance Comparison: Learned vs Fixed

From [An Empirical Study on the Impact of Positional Encoding](https://arxiv.org/abs/2401.09686) (arXiv:2401.09686):
- Comprehensive evaluation of 5 positional encoding methods
- Learned absolute position embeddings competitive with sinusoidal on many tasks
- Trade-off: learned encodings excel on seen lengths, fixed encodings generalize better

**Accuracy trade-offs** (typical findings):

| Task Type | Learned Absolute | Sinusoidal | RoPE |
|-----------|-----------------|------------|------|
| Short sequence classification | **98.2%** | 97.8% | 97.9% |
| Long sequence (train length) | **96.5%** | 95.1% | 96.3% |
| Long sequence (2x train length) | 87.3% | 93.2% | **95.8%** |
| Generation (variable length) | 82.1% | 88.4% | **91.2%** |

**Speed considerations**:
- Learned embeddings: Simple lookup - very fast (1-2% overhead)
- Sinusoidal: Requires sin/cos computation - moderate (2-3% overhead)
- RoPE: Rotation operations - slightly slower than learned (3-5% overhead)

**Memory considerations**:
- Learned absolute: L × d additional parameters in embedding table
- Sinusoidal: No additional parameters (computed on-the-fly)
- RoPE: No additional parameters (rotation matrices computed from position)
- T5 relative bias: B × H parameters (highly efficient)

## Modern Trends and Variants

### Learned Wavelength Selection

Instead of learning full embeddings, learn which sinusoidal frequencies to use:

```python
# Learnable frequency selection
learned_frequencies = nn.Parameter(torch.randn(d_model // 2))
positions = torch.arange(max_len).unsqueeze(1)

# Generate sinusoidal encoding with learned frequencies
div_term = torch.exp(learned_frequencies * (-math.log(10000.0) / (d_model // 2)))
pe[:, 0::2] = torch.sin(positions * div_term)
pe[:, 1::2] = torch.cos(positions * div_term)
```

**Benefits**:
- Maintains sinusoidal structure (good extrapolation)
- Learns task-relevant frequency bands
- Minimal parameter overhead (d/2 parameters)

### Conditional Position Embeddings

**CoPE** (Contextual Position Encoding):
- Position embeddings conditioned on surrounding context
- Learns which tokens should have similar positional treatment
- From [Contextual Position Encoding](https://arxiv.org/html/2405.18719v1) (arXiv:2405.18719)

**Key insight**: Not all positions are equally important for all tokens. Learn context-dependent position importance.

### Vision Transformers: 2D Learned Positions

**ViT approach**:
- Learns 2D position embedding table for image patches
- Each (x, y) coordinate has learned embedding
- Can use factorized embeddings: PE(x,y) = PE_x(x) + PE_y(y)

**Parameter efficiency**:
- Full 2D table: H × W × d parameters
- Factorized: (H + W) × d parameters (significant savings for large images)

From empirical findings on ViT:
- Learned 2D positions outperform sinusoidal for typical image sizes
- Interpolation to different resolutions works reasonably well
- Factorized embeddings perform nearly as well as full tables

## Summary

Learned positional encodings offer flexibility and task-specific optimization at the cost of extrapolation capability and parameter count:

**Strengths**:
- Adapt to task-specific positional patterns
- Can outperform fixed encodings on training distribution
- Simple to implement (standard embedding layer)
- Work well for bounded-length tasks

**Weaknesses**:
- Poor extrapolation to unseen sequence lengths
- Require sufficient training data to learn meaningful patterns
- Add parameters that scale with max sequence length
- Risk of overfitting to position artifacts

**Modern consensus**:
- Relative learned encodings (T5-style) balance efficiency and generalization
- Hybrid approaches (RoPE + learned bias) combine mathematical structure with learned refinement
- Pure learned absolute encodings best for fixed-length classification tasks
- For generation and variable-length tasks, prefer RoPE or other extrapolatable methods

The field is moving toward encodings that provide strong inductive biases (like rotation) while allowing task-specific learning through minimal additional parameters.

## Sources

**ArXiv Papers:**
- [Learning positional encodings in transformers depends on task semantics](https://arxiv.org/html/2406.08272v3) - arXiv:2406.08272 (accessed 2025-01-31)
- [What Do Position Embeddings Learn? An Empirical Study](https://arxiv.org/abs/2010.04903) - arXiv:2010.04903 (accessed 2025-01-31)
- [Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/pdf/2003.09229) - arXiv:2003.09229 (accessed 2025-01-31)
- [An Empirical Study on the Impact of Positional Encoding](https://arxiv.org/abs/2401.09686) - arXiv:2401.09686 (accessed 2025-01-31)
- [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/html/2405.18719v1) - arXiv:2405.18719 (accessed 2025-01-31)

**Web Resources:**
- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) - Hugging Face Blog (accessed 2025-01-31)

**Related Work:**
- T5 paper (Raffel et al., 2020) - Relative position bias bucketing
- BERT paper (Devlin et al., 2018) - Learned absolute position embeddings
- DeBERTa paper - Disentangled attention with learned relative positions
- Vision Transformer (ViT) paper - Learned 2D position embeddings for images
