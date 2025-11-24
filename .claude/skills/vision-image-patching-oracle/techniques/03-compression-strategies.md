# Compression Strategies

**Practical token compression techniques for reducing visual token count**

## Overview

This document covers practical implementation strategies for compressing visual tokens after patch encoding, enabling efficient processing of high-resolution images.

## Common Strategies

### 1. Perceiver Resampler

**From [source-documents/09_Efficient Vision-Language Pretraining](../source-documents/09_Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment - BMVC 2022.md)**:

Fixed-size learnable queries attend to all visual tokens:
```python
class PerceiverResampler(nn.Module):
    def __init__(self, num_latents=64, dim=1024, depth=2):
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.cross_attn = nn.ModuleList([
            CrossAttention(dim) for _ in range(depth)
        ])
    
    def forward(self, visual_tokens):
        x = self.latents.expand(visual_tokens.shape[0], -1, -1)
        for layer in self.cross_attn:
            x = layer(x, visual_tokens)
        return x  # Fixed 64 tokens output
```

**Compression**: 576 → 64 tokens (9×)

### 2. Token Merging

Merge similar adjacent tokens based on cosine similarity:
```python
def token_merging(tokens, threshold=0.9):
    similarities = cosine_similarity(tokens)
    merge_pairs = find_similar_pairs(similarities, threshold)
    return merge_tokens(tokens, merge_pairs)
```

**Compression**: 20-40% reduction, training-free

### 3. Attention-Based Pruning

Keep only high-attention tokens:
```python
def attention_pruning(tokens, attention_scores, keep_ratio=0.5):
    k = int(len(tokens) * keep_ratio)
    top_indices = torch.topk(attention_scores, k).indices
    return tokens[top_indices]
```

**Compression**: 2-4×

## When to Use Each Strategy

- **Perceiver**: General purpose, fixed output size
- **Token merging**: Training-free, adaptive compression
- **Attention pruning**: Query-dependent compression

## Primary Sources

- [21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)
- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)
- [03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)

## Related Documents

- [../architecture/04-compression-modules.md](../architecture/04-compression-modules.md)
- [../concepts/02-token-efficiency.md](../concepts/02-token-efficiency.md)
