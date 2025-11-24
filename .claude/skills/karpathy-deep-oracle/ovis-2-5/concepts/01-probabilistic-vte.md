# Probabilistic Visual Token Embedding

**Category**: Concepts
**Related**: [00-structural-alignment.md](00-structural-alignment.md)

## Math

**Standard (discrete)**:
```
token_id = 42
embedding = VET[42]
```

**Ovis (probabilistic)**:
```
probabilities = [p₀, p₁, ..., p₁₆₃₈₃]
embedding = Σᵢ (pᵢ × VET[i])
```

## Benefits

1. **Differentiable**: Softmax is smooth
2. **Flexible**: Can blend multiple embeddings
3. **Aligned**: Structurally similar to text

## Training

Gradients flow through both:
- VET embeddings: ∂Loss/∂VET[i]
- Probabilities: ∂Loss/∂p via visual_head

Result: Co-optimization of vocabulary and assignments.
