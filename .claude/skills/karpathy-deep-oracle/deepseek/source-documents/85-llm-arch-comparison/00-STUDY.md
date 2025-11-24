# LLM Architecture Comparison - Sebastian Raschka - Study

**Source**: Ahead of AI (The Big LLM Architecture Comparison - Ahead of AI - Sebastian Raschka.md)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - Comparative Analysis

---

## 📝 TL;DR

Sebastian Raschka's comprehensive comparison of modern LLM architectures (2024-2025). Covers DeepSeek V3/R1, Llama 4, GPT-oss, and others. Focuses on structural changes: RoPE, GQA, SwiGLU, MLA, MoE. Good high-level overview showing how DeepSeek fits into the broader LLM landscape. Not super technical but useful for context.

---

## 🎯 Key Concepts

### Evolution Since GPT-2 (2019)
- **Positional embeddings**: Absolute → RoPE (Rotational Position Embeddings)
- **Attention**: Multi-Head → GQA (Grouped-Query) → MLA (Multi-Head Latent)
- **Activation**: GELU → SwiGLU (more efficient)
- **Architecture**: Dense → MoE (Mixture of Experts)

### Key Observation
"At first glance... one might be surprised at how structurally similar these models still are."

Most changes are refinements, not revolutions. The Transformer is still the Transformer.

### DeepSeek V3 Highlights (from article)
- MLA for memory efficiency
- MoE for computational efficiency
- These are the "two key architectural techniques" that distinguish V3

---

## 💡 Why This Matters

**Big Picture Context**: Shows that DeepSeek's innovations (MLA, aux-loss-free MoE) are part of a broader trend toward efficient scaling. Everyone's chasing the same goals: better performance per dollar.

**Comparative Analysis**: Puts DeepSeek in context with Llama, GPT, Gemini, etc. Useful for understanding "what makes DeepSeek different?"

---

## 🔧 Karpathy-Style Implementation Notes

This is an overview article, not an implementation guide. Key takeaways for practitioners:

```
Architecture Checklist (2025 Edition):
✓ RoPE instead of absolute positional embeddings
✓ GQA or MLA instead of standard MHA
✓ SwiGLU instead of GELU
✓ MoE if you need efficient scaling
✓ FP8/BF16 mixed precision
```

If you're designing a new LLM from scratch, these are the defaults. DeepSeek V3 checks all these boxes + adds MLA on top.

---

## 🔗 Connections

- **01-deepseek-v3-technical-report**: Detailed V3 architecture
- **06-mla-explained**: Multi-Head Latent Attention deep dive
- **03-deepseek-moe-paper**: MoE architecture details
- **02-deepseek-v2-technical-report**: V2 architecture evolution

---

## 💭 Karpathy Take

Sebastian Raschka writes great technical overviews. This article is like "LLM architectures in 2025 for busy people." He correctly identifies that most models are still fundamentally Transformers with incremental improvements.

The interesting part is WHERE the improvements happen:
- **Attention efficiency**: MLA (DeepSeek), GQA (Llama), FlashAttention (everyone)
- **Compute efficiency**: MoE routing, FP8 training
- **Context length**: RoPE extensions, sparse attention

DeepSeek's contribution is making MLA work well + showing that MoE load balancing doesn't need auxiliary losses. Both are "small" changes that have big practical impacts.

The article mentions DeepSeek V3's impact in Jan 2025 - that's when people started taking it seriously. Before that it was "some Chinese lab." After R1 dropped, suddenly everyone's studying V3's architecture. Timing matters lol.

Good reference article for understanding the LLM landscape, but not deep technical detail. For that, read the actual papers.
