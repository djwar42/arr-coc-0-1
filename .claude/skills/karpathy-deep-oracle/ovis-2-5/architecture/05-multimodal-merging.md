# Multimodal Merging

**Category**: Architecture
**Related**: [03-visual-embedding-table.md](03-visual-embedding-table.md), [04-qwen3-llm.md](04-qwen3-llm.md)
**Code**: `merge_multimodal()` method in `modeling_ovis.py`

## Overview

Multimodal merging combines visual and text embeddings into a unified sequence for the LLM.

## Process

### 1. Text Tokenization

```python
prompt = "<image>\nDescribe this image."
tokens = tokenizer(prompt)
# [<image>] * 273 + [Describe, this, image, .]
# token_ids: [100000, 100000, ..., 100000, 5791, 420, 2217, 13]
```

### 2. Text Embedding

```python
text_embeds = embedding_table(token_ids)
# [batch, seq_len, hidden_size]
```

### 3. Visual Embedding Replacement

```python
# Find <image> token positions
image_positions = (token_ids == IMAGE_TOKEN_ID)  # 100000

# Replace with visual embeddings
merged_embeds = text_embeds.clone()
merged_embeds[image_positions] = visual_embeddings
```

### 4. Attention Mask Creation

```python
# Create proper attention mask
attention_mask = torch.ones_like(token_ids)
# All positions attend to all positions (causal masking in LLM)
```

### 5. Label Creation (Training)

```python
# For training: mask vision tokens in loss
labels = token_ids.clone()
labels[image_positions] = IGNORE_ID  # -100

# Loss only computed on text tokens
loss = cross_entropy(logits, labels, ignore_index=IGNORE_ID)
```

## Complete Example

```python
# Input
prompt = "<image>\nA cat."
image = [visual features] # 273 tokens

# After merging
embeddings = [
    visual_emb[0],    # Image token 1
    visual_emb[1],    # Image token 2
    ...
    visual_emb[272],  # Image token 273
    text_emb[0],      # "\n"
    text_emb[1],      # "A"
    text_emb[2],      # "cat"
    text_emb[3],      # "."
]
# Shape: [277, hidden_size]

# Labels for training
labels = [
    IGNORE,  # Vision tokens ignored in loss
    IGNORE,
    ...
    IGNORE,
    "\n",    # Text tokens used in loss
    "A",
    "cat",
    "."
]
```

## Related Topics

- [03-visual-embedding-table.md](03-visual-embedding-table.md) - Visual embeddings
- [04-qwen3-llm.md](04-qwen3-llm.md) - LLM processing
- [../codebase/01-modeling-ovis.md](../codebase/01-modeling-ovis.md) - Implementation
