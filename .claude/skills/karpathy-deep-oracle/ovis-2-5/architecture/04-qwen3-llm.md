# Qwen3 Language Model

**Category**: Architecture
**Related**: [00-overview.md](00-overview.md), [05-multimodal-merging.md](05-multimodal-merging.md)
**Code**: LLM integration in `modeling_ovis.py:206-211`

## Overview

Ovis 2.5 uses **Qwen3** as its language model backbone, providing strong reasoning and multimodal understanding capabilities.

## Why Qwen3?

**Advantages over Qwen2.5**:
- Better multimodal understanding
- Improved reasoning (especially for vision tasks)
- Enhanced long-context processing
- Optimized for vision-language integration

## Model Variants

### Ovis2.5-2B
- **LLM**: Qwen3-1.7B
- **Total params**: ~2B
- **Context**: 32K tokens
- **Use case**: Edge, fast inference

### Ovis2.5-9B
- **LLM**: Qwen3-8B
- **Total params**: ~9B
- **Context**: 32K tokens
- **Use case**: Production, best quality

## Architecture

**Decoder-only transformer**:
```python
{
    "num_hidden_layers": 28,  # (Qwen3-8B)
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,  # GQA
    "intermediate_size": 18944,
    "vocab_size": 151936
}
```

**Key Features**:
- GQA (Grouped Query Attention)
- RoPE position embeddings
- SwiGLU activations
- Flash Attention 2 compatible

## Integration with Vision

```python
# Merged sequence
embeddings = [vision_tokens (273), text_tokens (50)]
# Total: 323 tokens

# LLM processes both equally
hidden_states = qwen3(embeddings)
# Output: [323, hidden_size]

# Generate next tokens
logits = lm_head(hidden_states)
# [323, vocab_size]
```

## Generation

**Autoregressive**:
```python
for step in range(max_new_tokens):
    logits = model(input_ids)
    next_token = sample(logits[-1])
    input_ids = cat([input_ids, next_token])
```

## Related Topics

- [05-multimodal-merging.md](05-multimodal-merging.md) - How vision merges with LLM
- [06-thinking-mode.md](06-thinking-mode.md) - Advanced generation
