# RoPE: Rotary Position Embeddings

**Category**: Concepts
**Related**: [../architecture/01-navit-vision.md](../architecture/01-navit-vision.md)

## What is RoPE?

Relative position encoding via rotation in complex space.

## Why in Every ViT Block?

**Standard**: Absolute position embeddings added once
**Ovis**: RoPE applied in every attention layer

**Benefits**:
- Better spatial awareness
- Generalizes to unseen sizes
- Improves object localization

## Implementation

```python
def apply_rope(q, k, positions):
    # Compute rotation angles
    freqs = get_freqs(positions)

    # Rotate queries and keys
    q_rot = rotate(q, freqs)
    k_rot = rotate(k, freqs)

    return q_rot, k_rot
```

Applied in every self-attention operation.
