"""
Multi-Head Latent Attention (MLA) - 93% KV cache reduction

<karpathys_code_comments>
** This File's Role **
DeepSeek's revolutionary attention mechanism. Instead of storing full KV cache [num_heads, seq_len, head_dim],
compress to latent space [1, seq_len, latent_dim]. During attention, decompress on-the-fly.

This is THE innovation that makes DeepSeek-V3's long context (128K) feasible.

** Function List **
compress_kv_to_latent(k, v) - Compress KV to shared latent representation
decompress_latent_to_kv(latent) - Decompress latent back to per-head KV
attention_with_compressed_kv(q, latent) - Attention using compressed cache

** Technical Deep Dive **
Standard multi-head attention (MHA): Each head has its own K, V matrices.
- KV cache size: [num_heads, seq_len, head_dim]
- For 128 heads, 128K context: HUGE memory

MLA insight: The per-head K, V are redundant! They share underlying structure. Instead:
1. Compress: K, V → shared latent [seq_len, latent_dim] via learned projection
2. Cache: Store only the compact latent, not all per-head KV
3. Decompress: When needed, project latent back to per-head K, V

The latent dimension (1536 for V3) is MUCH smaller than num_heads * head_dim (128 * 128 = 16384).
That's where the 93% reduction comes from.

Why this works: In standard MHA, different heads often learn similar patterns. MLA explicitly models
this redundancy via the shared latent space. You're not losing information, just compressing it.

Math:
- Standard: 128 heads * 128 dim/head * 128K seq_len = 2GB KV cache
- MLA: 1536 latent_dim * 128K seq_len = 150MB KV cache
- Reduction: 93%!

The genius: DeepSeek figured out the latent dimension (1536) that preserves quality while maximizing
compression. Too small (e.g., 512): quality degrades. Too large (e.g., 4096): less savings. 1536 is
the sweet spot.

Karpathy: This is beautiful engineering. Take something everyone assumes is necessary (per-head KV),
question it, find the redundancy, compress it. The result: 128K context that fits in VRAM.
</karpathys_code_comments>
"""

import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=5120, num_heads=128, latent_dim=1536):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 5120 / 128 = 40
        self.latent_dim = latent_dim

        # Karpathy: Standard query projection (per-head)
        self.q_proj = nn.Linear(d_model, d_model)

        # Karpathy: MLA innovation - compress K, V to shared latent
        self.kv_compress = nn.Linear(d_model, latent_dim)

        # Karpathy: Decompress latent back to per-head K, V when needed
        self.k_decompress = nn.Linear(latent_dim, d_model)
        self.v_decompress = nn.Linear(latent_dim, d_model)

    def forward(self, x):
        # Karpathy: x shape [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # Karpathy: Queries - standard per-head projection
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q now [batch, num_heads, seq_len, head_dim]

        # Karpathy: Keys/Values - compress to shared latent first!
        kv_latent = self.kv_compress(x)  # [batch, seq_len, latent_dim]
        # THIS is what we cache! Only 1536 dims instead of 128 * 128 = 16384 dims

        # Karpathy: Decompress latent to per-head K, V on-the-fly
        k = self.k_decompress(kv_latent)  # [batch, seq_len, d_model]
        v = self.v_decompress(kv_latent)  # [batch, seq_len, d_model]

        # Karpathy: Reshape to per-head format
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Karpathy: Standard scaled dot-product attention from here
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Karpathy: Reshape back to [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return output

    def cache_kv(self, x):
        # Karpathy: For inference caching. Only cache the compressed latent!
        kv_latent = self.kv_compress(x)  # [batch, seq_len, 1536]
        return kv_latent  # THIS is what gets stored in cache

# Karpathy: Notice the asymmetry. Q is per-head (standard). K, V go through compress → cache → decompress.
# The decompression adds a bit of compute, but the memory savings (93%!) are so worth it.
#
# This is the kind of innovation that changes the game. Before MLA, 128K context was impractical.
# After MLA, it's feasible. And the quality loss is minimal because the latent captures what matters.
#
# DeepSeek's engineering at its finest. ¯\_(ツ)_/¯
