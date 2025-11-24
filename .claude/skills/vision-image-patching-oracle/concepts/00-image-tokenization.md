# Image Tokenization: From Pixels to Visual Vocabulary

**Understanding the transformation from continuous 2D arrays to discrete 1D sequences**

## The Core Challenge

### Text is Discrete, Vision is Continuous

**Text Tokenization (Simple):**
```
"Hello world" → ["Hello", "world"] → [2134, 5678]
Natural word boundaries, finite vocabulary
```

**Image Tokenization (Complex):**
```
1024×1024×3 RGB array (3.1M continuous values)
↓
How to create discrete tokens?
```

**Why This Matters:**
- Transformers expect discrete sequences
- LLMs trained on text tokens
- Vision must speak the "language" of text

---

## What is a Visual Token?

### Definition

**Visual Token**: A compressed representation of a spatial region in an image

**Properties:**
1. **Discrete**: Finite vocabulary of visual patterns
2. **Semantic**: Encodes meaningful visual features
3. **Contextual**: Captures local spatial relationships
4. **Compatible**: Projects into same embedding space as text tokens

### Anatomy of a Visual Token

```
┌─────────────────────────────────┐
│  Visual Token                   │
│                                 │
│  Spatial Info: Position [x,y]   │  ← Where in image
│  Visual Features: [1024 dims]   │  ← What it looks like
│  Semantic Meaning: Learned      │  ← What it represents
└─────────────────────────────────┘
```

**Contrast with Text Token:**
```
Text Token "apple" → [768-dim embedding]
  - Semantic: fruit, red, crispy
  - Syntactic: noun, countable
  - Contextual: Learned from co-occurrence

Visual Token (patch at [10,15]) → [1024-dim embedding]
  - Spatial: Top-left region
  - Visual: Red, round, texture patterns
  - Contextual: Learned from surrounding patches
```

---

## Tokenization Approaches

### 1. Patch-Based Tokenization (Standard)

**Method**: Divide image into grid, each patch = one token

```
336×336 image ÷ 14×14 patches = 576 tokens
Each patch: 14×14×3 = 588 raw values → 1024-dim embedding
```

**Vocabulary**: Continuous (each patch embedding is unique)
**Advantages:**
- Simple, efficient
- Preserves spatial structure
- Direct mapping to positions

**Disadvantages:**
- High token count
- Uniform allocation (sky gets same tokens as faces)

**Used by**: ViT, CLIP, LLaVA, most VLMs

**Reference**: `Vision transformer: To discover the "four secrets" of image patches`

---

### 2. Learned Discrete Codebooks (VQ-VAE)

**Method**: Train discrete vocabulary, assign patches to codes

```
Image → Encoder → Continuous codes
                 ↓
              Quantize to nearest discrete code
                 ↓
              Discrete token IDs (like text!)
```

**Example (VQVAE-2):**
```
256×256 image → 32×32 latent codes → Quantize to 1024 codebook
Result: 1024 discrete tokens, each from finite vocabulary
```

**Vocabulary**: Discrete, finite (512-8192 codes)

**Advantages:**
- True discrete tokens (like text)
- Compact representation
- Enables generative modeling

**Disadvantages:**
- Training complexity (vector quantization)
- Information loss at quantization
- Reconstruction quality limits

**Used by**: DALL-E, VQVAE, discrete diffusion models

**Reference**: Not primary focus for VLMs (more for generation)

---

### 3. Adaptive Tokenization

**Method**: Variable patch sizes based on content complexity

```
Complex region (face) → Small patches → More tokens
Simple region (sky) → Large patches → Fewer tokens
```

**Vocabulary**: Continuous, variable-count

**Advantages:**
- Efficient token allocation
- Preserves detail where needed
- Reduces total token count

**Disadvantages:**
- Irregular structure
- Complex position encoding
- Training overhead

**Used by**: APT, AgentViT, AdaViT

**Reference**: `Accelerating Vision Transformers with Adaptive Patch Sizes`

---

### 4. Hierarchical Tokenization

**Method**: Multi-scale pyramid of tokens

```
Level 1: 8×8 = 64 coarse tokens
Level 2: 16×16 = 256 medium tokens
Level 3: 32×32 = 1024 fine tokens
```

**Vocabulary**: Multi-scale, hierarchical

**Advantages:**
- Captures multiple resolutions
- Natural for hierarchical features
- Efficient coarse-to-fine processing

**Disadvantages:**
- Multiple token sets to manage
- Fusion complexity
- Higher total token count

**Used by**: Swin Transformer, Pyramid ViT

---

## Token Embedding Spaces

### Vision-Only Models (ViT)

```
Image Patches → Visual Encoder → Visual Embeddings
                                 (1024-dim, vision-only)
```

**Self-Contained**: Only process visual information
**Use Case**: Image classification, object detection

### Vision-Language Models (VLM)

```
Image Patches → Visual Encoder → Projection → Unified Embedding
                                             (LLM space)
Text Tokens → Text Embeddings ─────────────→ (Same space!)
```

**Cross-Modal Alignment**: Visual and text tokens in shared space
**Critical**: Enables LLM to "understand" vision

**Projection Methods:**
- **Linear**: Simple, fast (MLP)
- **Cross-Attention**: Q-Former, Perceiver (query-based)
- **Adapter**: Learned transformation layer

**Reference**: `Understanding Multimodal LLMs - by Sebastian Raschka, PhD`

---

## Token Vocabulary Size

### Continuous Embeddings (Standard VLM)

```
Every patch gets unique embedding
Vocabulary = ∞ (continuous space)
```

**Like**: Word embeddings in NLP (each word can have unique vector)
**Not Like**: Discrete token IDs (finite vocabulary)

**Token Count Examples:**
| Model | Resolution | Patch Size | Token Count |
|-------|------------|------------|-------------|
| ViT-B | 224×224 | 16×16 | 196 |
| ViT-L | 336×336 | 14×14 | 576 |
| LLaVA-1.5 | 336×336 | 14×14 | 576 |
| LLaVA-UHD | 672×1008 | 336 slices | ~3456 (6 slices × 576) |

---

## Semantic Richness

### What Information Does a Visual Token Encode?

**Low-Level Features:**
- Colors, edges, textures
- Local patterns, gradients
- Spatial frequency

**Mid-Level Features:**
- Object parts (eye, wheel, corner)
- Repeated structures
- Geometric relationships

**High-Level Features:**
- Object identities (face, car, text)
- Scenes (kitchen, street)
- Semantic concepts

**Learned Through:**
- Self-supervised pretraining (MAE, CLIP)
- Supervised classification (ImageNet)
- Vision-language alignment (contrastive learning)

**Reference**: `Vision-and-Language Pretrained Models: A Survey`

---

## Token Efficiency

### Redundancy in Visual Tokens

**Observation**: Most patches are redundant

```
Sky region: 100 patches of blue → highly redundant
Face region: 50 patches of details → information-rich
Text region: 30 patches of characters → critical

Without compression: All get equal tokens (wasteful!)
```

**Spatial Redundancy**: Neighboring patches are similar
**Semantic Redundancy**: Repeated patterns (sky, grass, walls)

**Compression Strategies:**
1. **Similarity-Based**: Merge similar tokens (ToMe, FastV)
2. **Attention-Based**: Keep high-attention tokens
3. **Query-Based**: Select relevant tokens for task (Q-Former)
4. **Optical Compression**: Serial encoder design (DeepSeek-OCR)

**Typical Ratios:**
- No compression: 576 tokens (336×336, patch=14)
- With pooling (4×): 144 tokens
- With ToMe (8×): 72 tokens
- With optical (16×): 36 tokens

**Reference**: `When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression`

---

## Visual vs Text Tokens

### Key Differences

| Aspect | Text Tokens | Visual Tokens |
|--------|-------------|---------------|
| **Nature** | Discrete, finite vocab | Continuous embeddings |
| **Count** | ~50-500 per input | 196-4000+ per image |
| **Redundancy** | Low (most words matter) | High (spatial correlation) |
| **Position** | 1D sequence | 2D grid (flattened) |
| **Semantics** | Symbolic, compositional | Perceptual, holistic |

### Implications for VLMs

**Challenge 1: Token Budget Imbalance**
```
Text: "Describe this image" → 4 tokens
Image: 336×336 → 576 tokens

Visual input dominates context length!
```

**Solution**: Token compression, adaptive allocation

**Challenge 2: Semantic Granularity**
```
Text: "cat" = 1 token (complete concept)
Visual: Cat requires ~20-50 patches (distributed representation)

Fine-grained visual details are expensive!
```

**Solution**: Hierarchical features, efficient encoders

**Challenge 3: Modality Alignment**
```
Must project visual tokens to same embedding space as text
Requires careful training (contrastive learning, instruction tuning)
```

**Solution**: CLIP-style pretraining, vision-language alignment

---

## Token Generation Process

### End-to-End Pipeline

```
┌──────────────┐
│ Input Image  │ (H×W×3 RGB array)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Patch Divide │ (H/P × W/P patches)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Flatten    │ (P×P×3 values per patch)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Project    │ (Linear: P²×3 → D)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Add Pos     │ (+ Position Embeddings)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Visual Tokens│ (N tokens, D dimensions)
└──────────────┘
```

**Output**: Sequence of N visual tokens, each D-dimensional
**Ready for**: Transformer processing alongside text tokens

---

## Token Quality Metrics

### How to Measure Token Effectiveness?

**1. Reconstruction Quality**
```
Tokens → Decoder → Reconstructed Image
Measure: MSE, PSNR, SSIM
```
**Good tokens preserve visual information**

**2. Downstream Performance**
```
Tokens → VLM → Task Performance
Measure: Accuracy on VQA, captioning, etc.
```
**Good tokens enable task success**

**3. Compression Ratio**
```
Ratio = Input Pixels / Token Count
Example: (336×336) / 576 = 196 pixels/token
```
**Higher ratio = more efficient** (if quality maintained)

**4. Semantic Alignment**
```
Visual Tokens ↔ Text Tokens
Measure: CLIP score, cross-modal retrieval
```
**Good tokens align with language**

---

## Future Directions

### Learned Visual Vocabularies

**Goal**: True discrete visual tokens (like text)
**Approach**: Trainable codebooks, efficient quantization
**Benefit**: Unified discrete sequence processing

### Query-Aware Tokenization

**Goal**: Different tokens for different questions
**Example**:
```
Q: "What color is the car?" → Tokenize car region finely
Q: "How many people?" → Tokenize people regions finely
```
**Benefit**: Efficient, task-specific allocation

**Reference**: ARR-COC-VIS relevance realization approach

### Multi-Granularity Tokens

**Goal**: Single image → multiple token resolutions
**Use**: Coarse tokens for global, fine tokens for local
**Benefit**: Flexible detail access

---

## Key Takeaways

### 1. Patches are Visual Words
Tokenization transforms continuous images to discrete-like sequences

### 2. Continuous vs Discrete
Most VLMs use continuous embeddings (not true discrete tokens like text)

### 3. Token Count is Critical
High token counts → computational bottleneck
Drives compression & adaptive strategies

### 4. Alignment Enables Understanding
Projecting visual tokens to LLM space enables cross-modal reasoning

### 5. Redundancy is High
Spatial correlation means compression opportunities

---

## Related Documentation

- **[architecture/01-patch-fundamentals.md](../architecture/01-patch-fundamentals.md)** - Patching mechanics
- **[concepts/01-patch-size-tradeoffs.md](01-patch-size-tradeoffs.md)** - Size selection
- **[concepts/02-token-efficiency.md](02-token-efficiency.md)** - Compression techniques
- **[comparisons/01-token-budgets.md](../comparisons/01-token-budgets.md)** - Token counts across models

---

**Next**: Understand [patch size tradeoffs](01-patch-size-tradeoffs.md) for optimal tokenization
