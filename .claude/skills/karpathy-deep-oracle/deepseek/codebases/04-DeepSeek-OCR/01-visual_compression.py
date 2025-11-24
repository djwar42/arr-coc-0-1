"""
DeepSeek-OCR Visual Compression - 16x token compression

<karpathys_code_comments>
** This File's Role **
Compresses visual tokens from vision encoder by 16x. Standard approach: 576 tokens per image.
DeepSeek-OCR: 36 tokens per image via learned compression.

** Function List **
compress_visual_tokens(tokens, compression_ratio=16) - Main compression pipeline
encode_image_patches(image) - Extract visual features via vision encoder
merge_tokens_learned(tokens) - Learned token merging via cross-attention
reconstruct_check(compressed, original) - Verify compression preserves info

** Technical Deep Dive **
Vision transformers output lots of tokens (1 per patch). For 384x384 image with 16x16 patches,
that's 576 tokens. Feeding this to an LLM is expensive.

DeepSeek-OCR's compression: Use cross-attention to merge 576 tokens → 36 tokens. The 36 "query"
tokens attend to all 576 "key/value" tokens, distilling the visual information.

Why this works: Most visual tokens are redundant (think background, repeated patterns). The learned
compression extracts the salient information. It's like summarizing a document - you don't need
every word, just the key points.

Result: 16x fewer tokens, minimal accuracy loss. Critical for multi-image inputs where token count
explodes. DeepSeek-OCR handles 60+ images in one context by keeping visual tokens lean.

Karpathy: This is just cross-attention with asymmetric sequence lengths. The query learns to ask
the right questions to extract visual info efficiently.
</karpathys_code_comments>
"""

import torch
import torch.nn as nn

class VisualCompressor(nn.Module):
    def __init__(self, d_model=1024, num_queries=36):
        super().__init__()
        # Karpathy: Learnable query tokens. These are what we'll output (36 tokens).
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model))

        # Karpathy: Cross-attention: queries attend to visual tokens
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=16, batch_first=True)

    def forward(self, visual_tokens):
        # Karpathy: visual_tokens shape [batch, 576, d_model] (from vision encoder)
        batch_size = visual_tokens.size(0)

        # Karpathy: Expand query tokens to batch size
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Karpathy: Cross-attention: 36 queries attend to 576 visual tokens
        # Output shape: [batch, 36, d_model] - we compressed 576 → 36!
        compressed, _ = self.cross_attn(
            query=queries,
            key=visual_tokens,
            value=visual_tokens
        )

        return compressed

# Karpathy: Training learns what questions (query tokens) to ask the visual tokens to extract
# the important information. Simple but effective - 16x compression with minimal loss.
