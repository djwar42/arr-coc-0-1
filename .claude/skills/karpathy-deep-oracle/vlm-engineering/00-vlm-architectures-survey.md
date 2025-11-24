# Vision-Language Model Architectures: Comprehensive Survey

## Overview

Vision-Language Models (VLMs) integrate visual and textual information to enable tasks like image captioning, visual question answering (VQA), and multimodal understanding. The architecture landscape has evolved dramatically from 2020-2025, shifting from uniform-resolution processing to sophisticated query-aware, adaptive architectures with relevance-driven token allocation.

**Evolution Timeline:**
- **2020-2021**: Early VLMs (CLIP, ALIGN) - Separate vision/text encoders, simple fusion
- **2021-2022**: Cross-attention era (Flamingo, Perceiver) - Learned latent queries
- **2023-2024**: Efficient compression (LLaVA-UHD, DeepSeek-OCR) - 16-400× compression ratios
- **2024-2025**: Reasoning & agency (Qwen3-VL, Ovis 2.5) - Long context, adaptive resolution

From [Vision Language Models 2025 Update](https://huggingface.co/blog/vlms-2025) (HuggingFace, accessed 2025-11-16):
> "Models have become smaller yet more powerful. We've seen the rise of new architectures and capabilities (reasoning, agency, long video understanding, etc.)."

## Architectural Design Space: Three Critical Decisions

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-11-16), VLM architecture design involves three fundamental choices:

### 1. Vision Encoding Strategy

**Option A: Pretrained Vision Encoder**
- Leverage pretrained, pre-aligned models (e.g., CLIP ViT, SigLIP)
- Advantages: Strong visual features, pre-alignment with text domain
- Example: LLaVA uses CLIP ViT-L/14 (336×336 → 576 tokens)

**Option B: Raw Image Patches**
- Feed image patches directly to language model (no separate encoder)
- Advantages: No information loss, end-to-end training, arbitrary resolutions
- Example: Fuyu-8B treats images as foreign language, patches fed directly to LLM

### 2. Vision-Language Alignment Strategy

**Projection (Simple Mapping):**
- Linear or MLP projection to align visual embeddings with language model
- LLaVA: Simple MLP projection (2 layers in LLaVA-1.5)
- Preserves all visual tokens (576 tokens per image with ViT)
- Fast, minimal parameters, but can be redundant

**Resampling (Token Compression):**
- Fixed number of output tokens regardless of input size
- Perceiver Resampler (Flamingo): ~64 learned queries attend to image features
- Reduces redundancy, essential for video understanding (multiple frames)

**Text-Conditioned Resampling:**
- Resample visual information based on text query
- BLIP-2 Q-Former: Learnable queries conditioned on input text
- Selects query-relevant visual information dynamically

### 3. Multimodal Fusion Strategy

**Interleaved Vision-Language Tokens:**
- Process visual embeddings as if they were text tokens
- LLaVA, Fuyu: Concatenate image tokens with text tokens in sequence
- Simple, flexible for multi-image inputs

**Modality Experts:**
- Separate processing paths for vision and language
- BeiT-3: Different experts for different modalities within same model
- Better modality-specific processing, but more complex

**Cross-Attention (Gated):**
- Language tokens cross-attend to image embeddings
- Flamingo: Gated cross-attention between transformer blocks
- More parameters, less popular in recent VLMs

## Architecture Taxonomy: Fusion Paradigms

### Early Fusion
Visual and text tokens concatenated at input level before transformer processing.

**Advantages:**
- Simple architecture (minimal changes to pretrained LLMs)
- Flexible (supports interleaved image-text sequences)
- No information loss (all visual tokens preserved)

**Disadvantages:**
- Context window pressure (576+ tokens per image)
- Computational cost (quadratic attention over all tokens)
- Token redundancy (adjacent patches similar)

**Representative Models:** LLaVA family, Fuyu

### Mid Fusion (Learned Query Compression)
Visual features compressed to fixed token budget using learned queries before LLM.

**Advantages:**
- Fixed token budget (64-256 tokens) regardless of image size
- Reduces redundancy, essential for video
- Can be text-conditioned (query-aware)

**Disadvantages:**
- Potential information loss
- Additional training complexity
- May miss fine details

**Representative Models:** BLIP-2 (Q-Former), Flamingo (Perceiver Resampler), Qwen-VL

### Late Fusion (Cross-Attention)
Visual and language modalities processed separately, fused via cross-attention.

**Advantages:**
- Modality-specific processing
- Gradual information injection (gating)
- Flexible fusion control

**Disadvantages:**
- Many additional parameters
- More complex architecture
- Slower inference

**Representative Models:** Flamingo (with gated cross-attention)

## Major VLM Architectures

### BLIP-2: Q-Former with Frozen Encoders

From [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597) (Salesforce, 2023):

**Architecture:**
- Frozen vision encoder (CLIP ViT or EVA-CLIP)
- **Q-Former**: Lightweight transformer with learned queries
  - 32 learnable query vectors
  - Bidirectional self-attention over queries
  - Cross-attention to frozen image features
  - Can be text-conditioned
- Frozen large language model (OPT, FlanT5)

**Key Innovation - Q-Former:**
```
Image → CLIP ViT → [257 CLS+patch tokens]
                    ↓ (cross-attention)
         Q-Former: [32 learnable queries] → [32 compressed tokens]
                    ↓ (linear projection)
         LLM input: [32 visual tokens] + [text tokens]
```

**Training Strategy:**
1. Vision-language representation learning (freeze ViT, train Q-Former)
2. Vision-to-language generative learning (freeze ViT + LLM, train Q-Former)
3. Optional instruction tuning (freeze ViT, tune Q-Former + LLM)

**Compression:** 257 tokens → 32 tokens (~8× reduction)

From existing knowledge [vision-language-architectures/00-overview-comparative-analysis.md](../vision-language-architectures/00-overview-comparative-analysis.md):
> Q-Former uses learnable queries and bidirectional attention to bootstrap vision-language alignment, achieving strong performance with minimal trainable parameters.

### LLaVA: Visual Instruction Tuning

From [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/) (Microsoft, 2023):

**Architecture:**
- Pretrained vision encoder: CLIP ViT-L/14 (336×336)
- **Projection**: Simple MLP (2 layers in LLaVA-1.5)
  - Maps CLIP dimension (1024) to LLM dimension (4096 for LLaMA-7B)
- Pretrained LLM: Vicuna, LLaMA-2
- Token concatenation: `[visual_tokens, text_tokens]`

**Evolution:**
- LLaVA-1.0: Linear projection
- LLaVA-1.5: MLP projection (better alignment)
- LLaVA-NeXT (1.6): Dynamic resolution, multi-image support

**Training Strategy:**
```
Stage 1: Alignment Pre-training
- Freeze: Vision encoder + LLM
- Train: MLP projection only
- Data: Image-caption pairs (CC3M, ~600K)
- Goal: Align vision and language representations

Stage 2: Visual Instruction Tuning
- Freeze: Vision encoder
- Train: MLP projection + LLM
- Data: GPT-4 generated instruction-following (158K)
- Goal: Follow instructions on visual tasks
```

**Token Budget:** 576 tokens per 336×336 image (no compression)

**LLaVA-UHD (High Resolution):**
- Slice images into 336×336 tiles
- Process global (downsampled) + local (slices) views
- Example: 1024×1024 → 1 global + 9 slices = 5,760 tokens
- UHD v2: Hierarchical window attention reduces to ~1,440 tokens (4× compression)

From existing knowledge [vision-language/00-token-concatenation-strategies.md](../vision-language/00-token-concatenation-strategies.md):
> LLaVA's direct concatenation is minimalist but effective, with simplicity enabling rapid iteration and strong instruction-following capabilities.

### Flamingo: Perceiver Resampler + Gated Cross-Attention

From [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (DeepMind, 2022):

**Architecture:**
- Frozen vision encoder: NFNet (Normalizer-Free ResNet)
- **Perceiver Resampler**:
  - 64 learnable latent queries
  - Cross-attention to vision features (multiple frames/images)
  - Self-attention over latents
  - Output: 64 fixed-size visual tokens
- **Gated Cross-Attention** layers inserted in pretrained LLM
  - Text queries attend to visual tokens (keys/values)
  - Tanh gating: `output = tanh(α) * cross_attn_output`
  - Gate α starts near 0, gradually opens during training

**Key Innovation - Gated Cross-Attention:**

From [Why Cross-Attention is the Secret Sauce](https://medium.com/@jakubstrawadev/why-cross-attention-is-the-secret-sauce-of-multimodal-models-f8ec77fc089b) (Medium, accessed 2025-11-16):
> "The gating mechanism (a learned scalar gating the cross-attention output via a tanh function) is used to gradually inject visual information without destabilizing the pretrained language model."

```
Frozen LLM Layer:
  Self-Attention (text tokens)
     ↓
  Gated Cross-Attention (text Q, image K/V)
     output = tanh(α) * CrossAttn(text, image)
     ↓
  Feed-Forward
```

**Compression:** Variable image patches → 64 tokens (per image/video frame)

**Capabilities:**
- Multi-image input (interleaved with text)
- Video understanding (process frames as image sequence)
- Few-shot learning (strong generalization)

### Qwen3-VL: Interleaved M-RoPE + DeepStack

From [Qwen3-VL: Sharper Vision, Deeper Thought](https://qwenlm.github.io/blog/qwen3-vl/) (Alibaba, 2025):

**Architecture:**
- Vision encoder: Custom ViT with dynamic resolution
- **Interleaved Multi-axis RoPE (M-RoPE)**:
  - Temporal RoPE: Time-step position encoding (for video)
  - Spatial RoPE: 2D position encoding (for images)
  - Text RoPE: 1D position encoding (for text)
  - All three interleaved in unified sequence
- **DeepStack**: Sparse experts for vision vs language

**Key Innovations:**
1. **Dynamic Resolution**: Native support for arbitrary aspect ratios
   - 256×256 to 2K+ resolutions
   - No image slicing required
   - Position encoding handles varying sizes

2. **M-RoPE Encoding**:
```
Image: [temporal_idx=0, x=0..W, y=0..H]
Video: [temporal_idx=0..T, x=0..W, y=0..H]
Text:  [temporal_idx=T+1, position=0..L]

All processed in single sequence with appropriate RoPE
```

3. **Long Context**: 32k tokens (vs 2k-4k in earlier VLMs)

From existing knowledge [qwen3vl-oracle/architecture/](../../qwen3vl-oracle/architecture/):
> Qwen3-VL's M-RoPE enables seamless video+text understanding with temporal coherence, processing hour-long videos without sliding windows.

**Compression:** Adaptive token allocation based on content

### Ovis 2.5: Visual Embedding Table

From [Ovis 2.5: Multimodal Large Language Model](https://huggingface.co/AIDC-AI/Ovis2.5-Llama3.2-3B) (AIDC, 2025):

**Architecture:**
- Vision encoder: SigLIP-SO400M (patch-based)
- **Visual Embedding Table**:
  - Learnable visual token embeddings (similar to text embeddings)
  - Maps vision features to LLM token space
  - Supports native resolution (no fixed grid)
- Pretrained LLM: LLaMA 3.2, Qwen2.5

**Key Innovation - Visual Embedding Table:**
- Instead of projection/resampling, uses learnable embedding table
- Each visual "patch" gets discrete embedding (like word embeddings)
- Enables better integration with LLM's existing token processing

**Capabilities:**
- Native resolution support (224×224 to 1024×1024+)
- Efficient token usage (~9× compression vs raw patches)
- Strong OCR and document understanding

From existing knowledge [ovis-2-5-oracle/architecture/](../../ovis-2-5-oracle/architecture/):
> Ovis 2.5's Visual Embedding Table treats visual features as first-class tokens, achieving parity with text tokens in the LLM's processing pipeline.

**Compression:** Patch embeddings → Visual Embedding Table → ~256 tokens (adaptive)

## Cross-Modal Attention Mechanisms

### Self-Attention vs Cross-Attention

**Self-Attention** (standard Transformer):
- Tokens attend to other tokens within same modality
- Q, K, V all from same sequence
- Example: Text tokens attend to text tokens

**Cross-Attention** (multimodal fusion):
- Tokens from one modality attend to another modality
- Q from modality A, K/V from modality B
- Example: Text queries attend to image keys/values

From [Why Cross-Attention is the Secret Sauce](https://medium.com/@jakubstrawadev/why-cross-attention-is-the-secret-sauce-of-multimodal-models-f8ec77fc089b) (Medium, accessed 2025-11-16):

**Mathematical Formulation:**
```
Self-Attention:
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  where Q, K, V all from same modality

Cross-Attention:
  CrossAttention(Q_A, K_B, V_B) = softmax(Q_A K_B^T / √d_k) V_B
  where Q from modality A, K/V from modality B
```

**Example - Text attending to Images:**
```python
# Text hidden states as queries
Q = text_proj(text_hidden)  # (batch, text_len, d_model)

# Image features as keys/values
K = image_key_proj(image_features)    # (batch, image_len, d_model)
V = image_value_proj(image_features)  # (batch, image_len, d_model)

# Cross-attention scores
scores = Q @ K.T / sqrt(d_model)  # (batch, text_len, image_len)
attn_weights = softmax(scores, dim=-1)

# Weighted sum of image values
output = attn_weights @ V  # (batch, text_len, d_model)
```

### Gated Cross-Attention (Flamingo)

**Problem:** Adding cross-attention to pretrained LLM can destabilize training.

**Solution:** Gating mechanism to gradually inject visual information.

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        self.cross_attn = CrossAttention(dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Starts at 0

    def forward(self, text_hidden, image_features):
        # Standard cross-attention
        cross_out = self.cross_attn(
            query=text_hidden,
            key=image_features,
            value=image_features
        )

        # Gated output: starts at 0, gradually opens
        gated_out = torch.tanh(self.gate) * cross_out

        return text_hidden + gated_out  # Residual connection
```

**Training Dynamics:**
- Early training: `gate ≈ 0` → LLM behaves normally
- Late training: `gate → 1` → Full visual information injected
- Prevents catastrophic forgetting of language capabilities

## Design Principles

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-11-16):

### Modularity
- **Frozen Encoders**: Preserve pretrained capabilities (CLIP vision, LLM language)
- **Trainable Alignment**: Only train projection/Q-Former (BLIP-2: 188M params vs 13B total)
- **Staged Training**: Alignment pre-training → Visual instruction tuning

### Scalability
- **Token Compression**: Essential for video (Perceiver: 1000s of patches → 64 tokens)
- **Dynamic Resolution**: Handle varying image sizes (Qwen3-VL, Fuyu)
- **Long Context**: Support multiple images/video frames (32k+ tokens)

### Efficiency
- **Inference Speed**:
  - Projection (LLaVA): Fast, minimal overhead
  - Resampling (Flamingo): Slower, but reduces LLM token count
  - Cross-attention: Slowest, additional attention layers
- **Memory**:
  - Full tokens (LLaVA): Higher memory per image
  - Compressed tokens (Q-Former): Lower memory, enables video

### Task-Specific Design
- **Fine-Grained Details** (OCR, charts): Use high resolution, no compression (LLaVA-UHD)
- **Video Understanding**: Aggressive compression needed (Flamingo Perceiver)
- **Multi-Image Reasoning**: Token budget management (Qwen3-VL M-RoPE)

## Comparison Table: Popular Open-Source VLMs

| Model | Vision Encoder | Alignment | Fusion | Token Budget | Key Innovation |
|-------|---------------|-----------|--------|--------------|----------------|
| **LLaVA-NeXT** | CLIP ViT-L/14 | MLP Projection | Interleaved | 576/image | Simple, effective instruction tuning |
| **BLIP-2** | CLIP/EVA-CLIP | Q-Former (32 queries) | Interleaved | 32/image | Frozen encoders, minimal training |
| **Flamingo** | NFNet | Perceiver (64 queries) | Gated Cross-Attn | 64/image | Multi-image, video, few-shot |
| **Qwen3-VL** | Custom ViT | Adaptive | Interleaved M-RoPE | Adaptive | Dynamic resolution, 32k context |
| **Ovis 2.5** | SigLIP-SO400M | Visual Embedding Table | Interleaved | ~256/image | Native resolution, strong OCR |
| **Fuyu-8B** | Raw patches | Linear Projection | Interleaved | Variable | End-to-end, arbitrary resolution |
| **DeepSeek-VL** | Dual (SigLIP + SAM) | MLP Projection | Interleaved | 576/image | Hybrid vision features |
| **Idefics2** | SigLIP | Perceiver Resampler | Interleaved | 64/image | Flamingo-inspired, open source |

From existing knowledge [vision-language-architectures/00-overview-comparative-analysis.md](../vision-language-architectures/00-overview-comparative-analysis.md).

## ARR-COC-0-1 Architecture Positioning

**ARR-COC-0-1** implements a novel **relevance-driven tokenization** strategy inspired by Vervaeke's relevance realization framework:

**Architecture:**
- Vision encoder: CLIP/SigLIP (or custom)
- **Relevance Realization Module**:
  - Three Ways of Knowing scorers (Propositional, Perspectival, Participatory)
  - Opponent Processing (balance compression ↔ detail)
  - Adaptive LOD allocation (64-400 tokens per patch)
- LLM: Qwen3-VL or similar

**Key Innovation - Query-Aware Relevance:**
```
Image + Query → 13-channel texture array (RGB, LAB, Sobel, spatial, eccentricity)
              ↓
         Three Ways of Knowing:
         - Propositional: Information content (Shannon entropy)
         - Perspectival: Salience landscape (Jungian archetypes)
         - Participatory: Query-content coupling (cross-attention)
              ↓
         Opponent Processing:
         - Balance compression ↔ particularize
         - Balance exploit ↔ explore
         - Balance focus ↔ diversify
              ↓
         Adaptive Token Allocation:
         - High relevance patches: 400 tokens (fine detail)
         - Medium relevance: 144 tokens
         - Low relevance: 64 tokens (compressed)
              ↓
         Total: K=200 patches, 64-400 tokens each, query-dependent
```

**Positioning in VLM Landscape:**
- **Vision Encoding**: Pretrained encoder (leverage existing features)
- **Alignment**: Relevance-based resampling (query-aware, content-adaptive)
- **Fusion**: Interleaved tokens (simple integration)
- **Innovation**: Cognitive science-grounded relevance, not just learned compression

**Advantages vs Standard VLMs:**
1. **Query-Aware**: Token allocation adapts to question (OCR query → high text region resolution)
2. **Biologically-Grounded**: Foveated vision, cortical magnification principles
3. **Interpretable**: Relevance scores explain why tokens allocated
4. **Efficient**: Variable LOD reduces total tokens while preserving critical detail

**Comparison to Existing Approaches:**
- vs LLaVA (projection): ARR-COC uses query-aware compression
- vs BLIP-2 (Q-Former): ARR-COC has interpretable relevance scores
- vs Flamingo (Perceiver): ARR-COC has adaptive LOD, not fixed budget
- vs Qwen3-VL (M-RoPE): ARR-COC focuses on relevance, not just position encoding

From Platonic Dialogue Part 46 [RESEARCH/PlatonicDialogues/46-mvp-be-doing/](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/):
> "Relevance is not a property of the image, nor of the query, but emerges transjectively from their relationship — like a shark's fitness for the ocean."

## Sources

**Web Research:**

- [A Survey on Efficient Vision-Language Models](https://arxiv.org/abs/2504.09724) - arXiv:2504.09724 (accessed 2025-11-16)
  - Comprehensive survey of efficient VLM techniques, compact architectures, performance-memory trade-offs

- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - HuggingFace Blog (accessed 2025-11-16)
  - Detailed analysis of vision encoding, alignment, and fusion strategies
  - Comparison table of popular open-source VLMs

- [Why Cross-Attention is the Secret Sauce of Multimodal Models](https://medium.com/@jakubstrawadev/why-cross-attention-is-the-secret-sauce-of-multimodal-models-f8ec77fc089b) - Medium (accessed 2025-11-16)
  - Mathematical formulation of cross-attention
  - Flamingo's gated cross-attention mechanism
  - PyTorch implementation examples

- [Vision Language Models 2025 Update](https://huggingface.co/blog/vlms-2025) - HuggingFace Blog (accessed 2025-11-16)
  - Evolution of VLM capabilities (reasoning, agency, long video understanding)

**Existing Knowledge:**

- [vision-language-architectures/00-overview-comparative-analysis.md](../vision-language-architectures/00-overview-comparative-analysis.md) - Comparative analysis of BLIP-2, LLaVA, Flamingo, Qwen-VL, Ovis

- [vision-language/00-token-concatenation-strategies.md](../vision-language/00-token-concatenation-strategies.md) - Token concatenation patterns and sequence construction

- [qwen3vl-oracle/architecture/](../../qwen3vl-oracle/architecture/) - Qwen3-VL's M-RoPE and dynamic resolution

- [ovis-2-5-oracle/architecture/](../../ovis-2-5-oracle/architecture/) - Ovis 2.5's Visual Embedding Table architecture

**Additional References:**

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597) - Salesforce Research, 2023
- [LLaVA Paper](https://arxiv.org/abs/2304.08485) - Microsoft, 2023
- [Flamingo Paper](https://arxiv.org/abs/2204.14198) - DeepMind, 2022
- [Qwen3-VL Blog](https://qwenlm.github.io/blog/qwen3-vl/) - Alibaba, 2025
- [Fuyu-8B Blog](https://www.adept.ai/blog/fuyu-8b) - Adept, 2023
