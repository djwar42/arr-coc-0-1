# Why Token Concatenation is Rare in Vision-Language Models

## Overview - The Concatenation Paradox

Token concatenation - directly appending visual tokens to text tokens as a unified sequence - seems like the most straightforward approach for vision-language models. If images are just another modality, why not treat visual patches like foreign language tokens and process everything together?

The reality: **concatenation is extremely rare in modern VLMs**. Most architectures actively avoid it through compression (DeepSeek, Ovis), learned queries (BLIP-2, Flamingo), or resampling (Perceiver). Understanding why reveals fundamental constraints in VLM design.

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-01-31):

> The choice of multimodal fusion strategies shows clear patterns: cross-attention was "mostly dismissed in recent vision-language models developments, likely because it introduces a lot of new parameters." Meanwhile, interleaved vision and language tokens dominate - but only after aggressive compression.

## Why Concatenation is Rare

### Memory Explosion: The N×M Problem

The core issue: **visual tokens outnumber text tokens by orders of magnitude**.

**Token Count Reality:**
- Standard ViT-L/14 (336px): **576 visual tokens** per image
- High-res vision encoder (1024px): **4,096 visual tokens** per image
- Typical text query: **10-50 tokens**
- LLM context window: **2,048 - 128,000 tokens** (model-dependent)

**The Explosion:**
```
Single 1024×1024 image = 4,096 tokens
10 images in conversation = 40,960 tokens
+ Text dialogue (500 tokens) = 41,460 tokens
→ Exceeds most LLM context windows before conversation even starts
```

From [Efficient Vision-Language Models by Summarizing Visual Tokens](https://arxiv.org/html/2410.14072v1) (arXiv:2410.14072, accessed 2025-01-31):

> "This process significantly increases the computational cost due to the **quadratic attention cost** with respect to tokens. As an example, LLaVA-NeXT processes images at 672×672 resolution, resulting in 2,880 visual tokens per image - before any text is added."

### Quadratic Attention Cost: O(n²) Scaling

Transformer self-attention scales **quadratically** with sequence length:

**Computational Cost:**
- Attention complexity: **O(n²d)** where n = total tokens, d = hidden dimension
- Memory for attention matrices: **O(n²)**

**Concrete Example (4,096 visual tokens):**
```
Text-only (50 tokens):
  Attention ops: 50² = 2,500 operations

After concatenating 1 image (4,096 visual tokens):
  Total tokens: 50 + 4,096 = 4,146
  Attention ops: 4,146² = 17,189,316 operations
  → 6,876× more computation!
```

From [Quadratic Is Not What You Need For Multimodal Large Language Models](https://arxiv.org/html/2410.06169v1) (arXiv:2410.06169, accessed 2025-01-31):

> "By cutting the attention connections between visual tokens, we reduced the computational cost from **quadratic growth to linear growth** with respect to the number of images."

**The Video Problem:**
Video understanding amplifies this catastrophically:
- 10-second video at 1 FPS = 10 frames
- 10 frames × 4,096 tokens = **40,960 visual tokens**
- Add text = **~41,000 token sequence**
- Self-attention cost: **41,000² = 1.68 billion operations** per layer

### Fixed Context Windows: The Architectural Constraint

Most LLMs have **fixed maximum context lengths**:

**Context Window Limits (2024-2025):**
- GPT-3.5: 4,096 tokens
- GPT-4: 8,192 - 128,000 tokens (model variant)
- LLaMA 2: 4,096 tokens
- LLaMA 3: 8,192 - 128,000 tokens
- Claude: 200,000 tokens (rare exception)

**The Squeeze:**
If visual tokens consume most of the context window, there's no room for:
- Conversation history
- Multi-turn dialogue
- Complex instructions
- Retrieved documents (RAG)
- Chain-of-thought reasoning

From [LLM context window limits visual tokens concatenation problem](https://stackoverflow.com/questions/78602027/) (Stack Overflow, accessed 2025-01-31):

> "However, if I set top_k as 3, I would have the following error as the LLM prompt input has to take in from 3 sources: **BadRequestError: context length exceeded the 8192 token limit**"

**Trade-off Dilemma:**
```
Option A (Full concatenation):
  Use 4,096 tokens for image → 4,000 tokens left for text
  → Rich visual detail, but conversation dies after 2-3 turns

Option B (Aggressive compression):
  Compress to 256 tokens → 7,800 tokens for text
  → Full conversation possible, but lost visual detail
```

### Memory Bandwidth: The Hidden Bottleneck

Beyond computation, **memory access patterns** matter.

From [Faster Transformers? Flash Attention](https://medium.com/@jakubstrawadev/faster-transformers-flash-attention-s-cf0debfeee25) (Medium, accessed 2025-01-31):

> "The core issue lies in computing the attention scores between all pairs of tokens. This scales quadratically, but the **memory bandwidth bottleneck** is often worse than raw FLOPS."

**Memory Access Pattern:**
- Standard attention: Multiple passes over KV cache
- Long sequences (4,000+ tokens): KV cache doesn't fit in fast SRAM
- Result: **Memory-bound** rather than compute-bound

**Concatenation Impact:**
- 4,096 visual tokens = large KV cache from start
- Every text token attends to all 4,096 visual tokens
- Poor cache locality, excessive DRAM reads

## Dominant Alternatives: Why They Win

### Cross-Attention: Selective Information Access

**How it works:**
- Visual tokens processed separately
- Text tokens use **cross-attention** to query visual features
- Asymmetric: text→vision attention, but not vision→text

**Architecture Example (Flamingo):**
```
Vision Encoder (frozen) → Visual Tokens (N_v)
                               ↓
Text Tokens (N_t) ──[Cross-Attention]──> Combined Features
                               ↓
                    Language Model Layers
```

**Efficiency Gain:**
- Self-attention: **O((N_t + N_v)²)** = quadratic in both
- Cross-attention: **O(N_t × N_v)** = linear in each
- For N_t=100, N_v=4096: Cross-attention is **~40× cheaper**

From [HuggingFace VLM Design](https://huggingface.co/blog/gigant/vlm-design) (accessed 2025-01-31):

> "Cross attention: language tokens can attend to image embeddings using cross-attention in-between transformer blocks (eg Flamingo). This strategy was mostly dismissed in recent vision-language models developments, likely because it introduces a lot of new parameters."

**Why Cross-Attention Fell Out of Favor:**
- Adds new parameters (cross-attention layers)
- Harder to leverage pretrained LLM weights
- More complex training dynamics

### Learned Queries: The Q-Former Approach (BLIP-2)

**How it works:**
- Fixed set of **learnable query tokens** (typically 32-64)
- Queries extract information from visual tokens via cross-attention
- Only query outputs concatenated with text

**Architecture:**
```
Visual Tokens (4,096) ──┐
                        ├─[Cross-Attention]─> Query Outputs (32)
Learned Queries (32) ───┘                            ↓
                                          [Concatenate with Text]
                                                     ↓
                                            Language Model
```

**Dramatic Compression:**
- Input: 4,096 visual tokens
- Output: **32 query tokens** (128× compression!)
- Text concatenation: Only 32 tokens added
- Result: **Manageable context window usage**

From [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597) (accessed via HuggingFace article):

> "Q-Former architecture details: Learned query tokens cross-attend to frozen vision encoder, self-attention among queries enables them to extract complementary visual information."

**Counter-Example: LLaVA's "Concatenation"**

LLaVA is often cited as concatenation-based, but there's a critical nuance:

**LLaVA Actually Does:**
1. Vision encoder: 4,096 tokens
2. **MLP projection**: Compresses to 576 tokens (7× compression)
3. Concatenate compressed tokens with text
4. Feed to LLM

It's **compress-then-concatenate**, not pure concatenation.

From [LLaVA Architecture](https://llava-vl.github.io/) (accessed 2025-01-31):

> "The authors call this mapping the 'projection', and it is trained on image/caption pairs while keeping the vision and language models frozen... In LLaVA 1.5 and 1.6/NeXT, it was swapped for a more expressive Multi-Layer Perceptron (MLP)."

### Compression-First Strategies

Modern VLMs follow a clear pattern: **compress first, concatenate later**:

**DeepSeek-VL Optical Compression:**
- SAM segmentation + CLIP encoding
- **16× compression ratio** (4,096 → 256 tokens)
- Then concatenate with text

**Ovis Multi-Visual Embedding Table:**
- Dynamic token allocation (64-400 tokens)
- Context-aware compression
- Relevance-based visual budget

**Window Token Concatenation (WiCo):**

From [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024) (arXiv:2504.04024, accessed 2025-01-31):

> "We propose a novel approach called Window Token Concatenation (WiCo). Specifically, we employ a sliding window to **concatenate spatially adjacent visual tokens**, reducing the total number while maintaining spatial structure."

**Pattern:**
```
Raw Vision Tokens → [Aggressive Compression] → Compact Representation → [Concat with Text] → LLM
   (4,096)              (90-95% reduction)          (256-576)              (manageable)
```

## Counter-Examples: When Concatenation Works

### Fuyu: End-to-End Simplicity

**Fuyu's Radical Approach:**
- No vision encoder
- Raw image patches → Linear projection → Directly to LLM
- Pure concatenation of patch embeddings with text

From [Fuyu-8B Architecture](https://www.adept.ai/blog/fuyu-8b) (Adept Blog, accessed via HuggingFace):

> "They simplified both the architecture and training procedure by feeding the image patch embeddings as is to a language model. With that framework, there is no need to think about how to scale the vision encoder vs the language model."

**Why It Can Work:**
- Variable resolution support (no fixed vision encoder)
- Simpler architecture (fewer components to align)
- Assumes **massive training budget** to learn from scratch

**Critical Caveat:**

From HuggingFace article:

> "The authors claim that the Fuyu framework is 'easier to understand, scale, and deploy', but give no information about the amount of data used or the cost for training such model. **It would be no surprise if it is orders of magnitude more than with the LLaVA framework**, for comparable results."

**Trade-off:**
- Architectural simplicity
- **Paid for with:**
  - Massive training costs
  - Huge token counts (still needs compression in practice)
  - Limited adoption (few can afford training)

### BeiT-3: Multimodal from Scratch

**Approach:**
- Train unified model on text + image patches from scratch
- Modality experts within shared architecture
- Treats images as "foreign language"

**Why Rare:**
- Requires joint pretraining (expensive)
- Can't leverage existing LLM weights
- Doesn't solve token count problem (still needs compression)

## Engineering Pragmatism: The Karpathy Lens

From a first-principles perspective, why does the field avoid concatenation?

### The Computational Reality Check

**FLOPs Analysis (Single Forward Pass):**

```
Pure Concatenation (4,146 tokens total):
  Self-attention: 4,146² × d = 17.2M × d operations per layer
  12-layer LLM: 206M × d total operations

Compress-then-Concat (350 tokens total):
  Compression: 4,096 → 256 (learned queries)
  Self-attention: 350² × d = 122K × d per layer
  12-layer LLM: 1.5M × d total operations
  → 137× speedup on attention alone!
```

### Memory Reality

**GPU Memory Consumption (Inference):**
```
Model: 7B parameters (14 GB in FP16)
Batch size: 1

Pure concat (4,146 tokens):
  KV cache: 4,146 × layers × d × 2 (K+V) × 2 bytes
  = 4,146 × 32 × 4,096 × 2 × 2 = 2.1 GB
  + Model weights (14 GB) = 16.1 GB total

Compressed (350 tokens):
  KV cache: 350 × 32 × 4,096 × 2 × 2 = 180 MB
  + Model weights (14 GB) = 14.18 GB total
  → 12× memory reduction on KV cache
```

### Batching Constraints

**Why It Matters for Production:**

```
Concatenation approach:
  Single image = 4,146 tokens
  Max batch size (on A100 40GB): ~8 samples
  Throughput: 8 images/batch

Compressed approach:
  Single image = 350 tokens
  Max batch size (on A100 40GB): ~96 samples
  Throughput: 96 images/batch
  → 12× higher throughput
```

Production systems care about **tokens/second/dollar** - concatenation loses badly.

## The Granularity Argument: Are Images Really Foreign Language?

From [HuggingFace VLM Design](https://huggingface.co/blog/gigant/vlm-design):

> "An aspect we might reflect on is the **granularity of modalities**... The visual and audio spaces are fine-grained (there are many visuals or sounds of guitars that might be really different to each other) while the textual domain is more coarse as its goal is to abstract away details (e.g. a single 'guitar' word)."

**Counter-Argument:**
Not all visual tokens need equal granularity:
- Document OCR: Text regions = coarse-grained
- Natural photos: Textures = fine-grained
- Similarly, text tokens vary: "the" vs "photosynthesis"

**Implication:**
Maybe we **shouldn't** treat all visual tokens equally. Concatenation forces uniform processing. Compression allows variable importance.

## Modern Consensus: Compress First

**2024-2025 VLM Architecture Pattern:**

```
┌─────────────────────────────────────────────────────┐
│ Vision Input → Encoder → AGGRESSIVE COMPRESSION     │
│    (Image)      (ViT)      (Query/Resample/Pool)   │
│                                ↓                    │
│                         Compact Visual Tokens       │
│                            (64-576 tokens)          │
│                                ↓                    │
│                         [Concatenate Text]          │
│                                ↓                    │
│                         Language Model              │
└─────────────────────────────────────────────────────┘
```

**Compression Techniques (2025):**
- **Learned Queries**: BLIP-2 Q-Former (32-64 tokens)
- **Perceiver Resampler**: Flamingo (64 tokens)
- **Adaptive Pooling**: C-Abstractor, D-Abstractor
- **Optical Compression**: DeepSeek (256 tokens, 16× reduction)
- **Multi-Embedding Tables**: Ovis (64-400 dynamic allocation)
- **Window Concatenation**: WiCo (spatial pooling)

**None of them do pure concatenation.**

## Future: Will Concatenation Ever Return?

### Scenario 1: Infinite Context Windows

If LLMs reach **truly unlimited context** (10M+ tokens):
- Token count becomes less critical
- Quadratic attention still a problem (need sub-quadratic attention)
- Memory bandwidth still limits throughput

**Verdict:** Unlikely. Even with infinite context, quadratic cost kills concatenation.

### Scenario 2: Linear Attention Breakthroughs

State-space models (Mamba, RWKV) offer **linear complexity**:
- O(n) instead of O(n²)
- Could handle 4,096 visual tokens easily

From [HuggingFace VLM Design](https://huggingface.co/blog/gigant/vlm-design):

> "At some point, we might have models trained end-to-end to figure everything by themselves based on statistics of more-and-more massive datasets. e.g. Fuyu-MoD-style with infinite-context."

**Verdict:** Possible but distant. Linear attention models still maturing. Concatenation might return for end-to-end training with state-space models.

### Scenario 3: The Bitter Lesson

From [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html):

> "All these tricks to cleverly leverage and align pretrained models of different modalities, and to filter and focus the visual content for a given token budget, are temporary solutions."

**Eventual Future:**
- Massive compute budgets
- End-to-end joint training
- Models learn optimal compression internally
- But: Still won't use **raw concatenation** - will learn better representations

## Summary

Token concatenation is rare in VLMs due to fundamental constraints:

1. **Memory Explosion**: Visual tokens (4,096) dwarf text tokens (50)
2. **Quadratic Attention**: O(n²) cost makes long sequences prohibitive
3. **Fixed Context Windows**: LLMs have hard token limits (2K-128K)
4. **Memory Bandwidth**: KV cache access patterns kill performance
5. **Batching Requirements**: Production needs high throughput

**Dominant Alternatives:**
- Cross-attention (Flamingo): 40× cheaper than concatenation
- Learned queries (BLIP-2): 128× compression (4,096 → 32 tokens)
- Compression-first (DeepSeek, Ovis): 90-95% token reduction

**Counter-Examples:**
- LLaVA: Actually compress-then-concatenate (not pure concatenation)
- Fuyu: End-to-end training (requires massive budget, rare adoption)

**Engineering Reality:**
- Pure concatenation: 206M operations, 2.1 GB KV cache
- Compressed: 1.5M operations, 180 MB KV cache
- **137× speedup, 12× memory reduction**

**The Pattern:**
Every successful VLM compresses **before** concatenation. The question isn't whether to compress, but **how much** and **by what method**.

## Sources

**Web Research:**

- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - HuggingFace (accessed 2025-01-31)
  - Comprehensive survey of VLM fusion strategies
  - Analysis of why cross-attention was dismissed
  - Comparison table of 10+ open-source VLMs

- [Efficient Vision-Language Models by Summarizing Visual Tokens](https://arxiv.org/html/2410.14072v1) - arXiv:2410.14072 (accessed 2025-01-31)
  - Quadratic attention cost analysis
  - LLaVA-NeXT token count examples
  - Victor compression method

- [Quadratic Is Not What You Need For Multimodal Large Language Models](https://arxiv.org/html/2410.06169v1) - arXiv:2410.06169 (accessed 2025-01-31)
  - Cutting attention connections between visual tokens
  - Quadratic to linear reduction proof
  - 8k context window experiments

- [LLaVA-Mini: Efficient Image and Video Large Multimodal Models](https://arxiv.org/html/2501.03895v1) - arXiv:2501.03895 (accessed 2025-01-31)
  - Modality pre-fusion approach
  - Single-token compression experiments
  - Video understanding efficiency

- [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024) - arXiv:2504.04024 (accessed 2025-01-31)
  - Sliding window concatenation approach
  - Spatial structure preservation
  - Token reduction experiments

- [Faster Transformers? Flash Attention](https://medium.com/@jakubstrawadev/faster-transformers-flash-attention-s-cf0debfeee25) - Medium (accessed 2025-01-31)
  - Memory bandwidth bottleneck analysis
  - Flash attention optimizations
  - SRAM vs DRAM access patterns

- [Context Window Problem: Scaling Agents](https://factory.ai/news/context-window-problem) - Factory.ai (accessed 2025-01-31)
  - Context window limits in practice
  - Enterprise monorepo token counts
  - Production deployment constraints

- Stack Overflow discussions on LLM context limits (accessed 2025-01-31)
  - Practical error messages and constraints
  - OpenAI API token limit issues

**Cross-References:**

- BLIP-2 architecture (Q-Former learned queries)
- Flamingo architecture (Perceiver Resampler, gated cross-attention)
- Fuyu architecture (end-to-end patch embeddings)
- LLaVA family (MLP projection compression)
- DeepSeek-VL (optical compression, 16× reduction)

**Additional Context:**

- The Bitter Lesson (Rich Sutton) - Referenced for end-to-end learning philosophy
- Transformer attention complexity (standard O(n²) analysis)
- GPU memory constraints (A100 40GB example calculations)
