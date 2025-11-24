# Architecture Overview: Image Patching in Vision-Language Models

**The complete landscape of visual tokenization strategies**

## The Patching Problem

Vision-Language Models must convert images into token sequences that LLMs can process. This transformation—from 2D pixel arrays to 1D token sequences—is the fundamental challenge of visual tokenization.

### Why Patching?

**Challenge**: A 1024×1024 RGB image contains ~3M values. Processing each pixel independently:
- Creates sequences too long for transformers (quadratic attention complexity)
- Loses local spatial structure
- Ignores hierarchical visual patterns

**Solution**: Divide images into patches, treat each patch as a "visual word"
- Reduces sequence length dramatically
- Preserves local spatial coherence
- Enables transfer of NLP transformer architectures

## Three Patching Paradigms

### 1. Fixed Patching (Standard ViT)

**Principle**: Uniform grid division regardless of content

```
336×336 image ÷ 14×14 patches = 576 tokens
```

**Characteristics:**
- Simple, predictable token count
- Shape distortion on non-square images
- Inefficient on varied content (sky gets same tokens as text)

**Used by**: Original ViT, CLIP, early LLaVA models

**Reference**: `A Comprehensive Study of Vision Transformers in Image Classification Tasks`

---

### 2. Adaptive Patching (Content-Aware)

**Principle**: Variable patch sizes based on image content complexity

```
Complex regions (faces, text) → Small patches (more tokens)
Simple regions (sky, walls) → Large patches (fewer tokens)
```

**Characteristics:**
- Dynamic token allocation
- Computationally efficient
- Requires saliency/attention scoring
- Variable token counts per image

**Key Models:**
- **APT (Adaptive Patch Transformer)**: 2-4× speedup with minimal accuracy loss
- **AgentViT**: RL-guided patch selection
- **AdaViT**: Dynamic usage policies on patches, heads, layers

**Reference**: `Accelerating Vision Transformers with Adaptive Patch Sizes` (arXiv 2510.18091)

---

### 3. Native-Resolution Processing

**Principle**: Preserve aspect ratio and resolution through flexible slicing

```
Original: 672×1008 image
LLaVA-UHD: Divides into variable-sized 336×336 slices
Ovis: Processes natively with Visual Embedding Table
```

**Characteristics:**
- No shape distortion
- Handles any aspect ratio
- Multi-slice coordination
- Spatial schemas for slice positions

**Key Models:**
- **LLaVA-UHD**: Variable-sized slices + compression module
- **Ovis 2.5**: Native resolution with VET structural alignment
- **DeepSeek-OCR**: 16× compression with Gundam tiling

**Reference**: `LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images`

---

## The Token Budget Challenge

### Computation Bottleneck

Transformer self-attention: **O(n²)** complexity where n = token count

**Example Token Counts:**

| Resolution | Standard ViT (16×16) | With Compression (4×) | DeepSeek-OCR (16×) |
|------------|---------------------|----------------------|-------------------|
| 224×224 | 196 tokens | 49 tokens | 12-14 tokens |
| 336×336 | 441 tokens | 110 tokens | 27-73 tokens |
| 672×672 | 1764 tokens | 441 tokens | 110 tokens |
| 1008×1008 | 3969 tokens | 992 tokens | 248 tokens |

**Reference**: `When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression`

---

## Core Components

### 1. Patch Division Layer

**Function**: Split image into grid or adaptive regions

```python
# Fixed patching (ViT)
patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                   p1=patch_size, p2=patch_size)

# Adaptive patching (APT)
patches = adaptive_partition(image, saliency_map, patch_sizes=[8,16,32])
```

### 2. Patch Embedding

**Function**: Project flattened patches to embedding dimension

```python
# Linear projection
patch_embed = Linear(patch_dim, embed_dim)  # (patch_h × patch_w × 3) → D

# With learnable position encoding
embedded = patch_embed(patches) + pos_embed
```

### 3. Compression Module (Optional)

**Function**: Reduce token count before LLM

**Methods:**
- **Pooling**: Max/average over spatial dimensions (2-4× reduction)
- **Token Merging**: Similarity-based clustering (ToMe, FastV)
- **Cross-Attention**: Query-based selection (Q-Former, Perceiver)
- **Optical Compression**: Serial encoder design (DeepSeek-OCR, 16×)

**Reference**: `Generic Token Compression in Multimodal Large Language Models`

### 4. Spatial Encoding

**Function**: Preserve patch position information

**Approaches:**
- **Learned 2D Positional Embeddings**: ViT standard
- **RoPE (Rotary Position Embedding)**: Relative positions
- **Spatial Schema**: Explicit slice organization (LLaVA-UHD)

---

## Design Tradeoffs

### The Patching Trilemma

```
    Fine-grained Detail
           /\
          /  \
         /    \
        /______\
Computational    Context
    Cost         Length
```

**Cannot simultaneously maximize all three:**
- Small patches = fine detail BUT high token count
- Large patches = efficient BUT lose detail
- Variable patches = balanced BUT complex implementation

### Resolution vs Efficiency

**Observation**: 2× resolution increase = 4× token count

```
224×224 (ViT-B/16) → 196 tokens
448×448 (ViT-L/16) → 784 tokens (4× increase)
672×672 (LLaVA-HD) → 1764 tokens (9× increase!)
```

**Solution Strategies:**
1. **Fixed Budget**: Downsample high-res images (loses detail)
2. **Compression**: Reduce tokens post-encoding (adds compute)
3. **Adaptive**: Allocate tokens by importance (complex training)
4. **Modularization**: Process slices separately (coordination overhead)

---

## Evolution Timeline

### 2020: ViT Foundation
- 16×16 fixed patches
- 224×224 standard resolution
- Established transformer viability for vision

### 2021-2022: Scaling & Efficiency
- Larger models, higher resolutions
- Initial token compression (ToMe, FastV)
- Hybrid CNN-Transformer designs

### 2023: Multi-Resolution Era
- LLaVA-HD: Slice-based processing
- Token budgets become critical
- Native resolution awareness

### 2024-2025: Adaptive & Compressed
- **APT**: Content-aware patch sizing
- **LLaVA-UHD**: Variable slice modularization
- **DeepSeek-OCR**: 16× optical compression
- **Ovis 2.5**: Native resolution with VET

---

## Key Principles

### 1. Spatial Locality Matters
**Insight**: Nearby pixels are highly correlated
**Implementation**: Patches preserve local structure

### 2. Content Redundancy is High
**Insight**: Sky/walls need fewer tokens than faces/text
**Implementation**: Adaptive allocation, compression modules

### 3. Aspect Ratio Preservation Improves Accuracy
**Insight**: Shape distortion degrades fine-grained tasks (OCR, counting)
**Implementation**: Native resolution, variable slicing

### 4. Position Information is Critical
**Insight**: "Where" a patch is matters for understanding
**Implementation**: 2D positional embeddings, spatial schemas, RoPE

---

## Architecture Patterns

### Pattern 1: Encode → Compress → LLM
```
Image → ViT Encoder → Compression Layer → LLM
       (576 tokens)   (144 tokens)       (process)
```
**Examples**: LLaVA-1.5, MiniGPT-4

### Pattern 2: Slice → Encode → Organize → LLM
```
Image → Variable Slices → Encode Each → Spatial Schema → LLM
       (6 slices)         (576 tok/slice) (organize)     (process)
```
**Examples**: LLaVA-UHD, GPT-4V

### Pattern 3: Adaptive → Encode → LLM
```
Image → Saliency Map → Adaptive Patches → Variable Encoder → LLM
       (analyze)       (mixed sizes)      (multi-scale)      (process)
```
**Examples**: APT, AgentViT, AdaViT

### Pattern 4: Serial Compression → LLM
```
Image → Deep Encoder → Optical Compression → Minimal Tokens → LLM
       (SAM+CLIP)     (16× reduction)        (73-421 tok)    (process)
```
**Examples**: DeepSeek-OCR

---

## Implementation Considerations

### Memory Constraints
- **Training**: Batch size limited by peak token count
- **Inference**: KV cache grows with visual tokens
- **Trade-off**: More tokens = better quality BUT slower, costlier

### Positional Out-of-Distribution
- **Problem**: Encoders trained on fixed resolution
- **Risk**: High-res images exceed training positions
- **Solutions**: Interpolation, RoPE, native training

### Slice Coordination
- **Challenge**: Multi-slice images need spatial organization
- **Methods**: Explicit schemas, learned aggregation, hierarchical pooling

---

## Related Documentation

- **[01-patch-fundamentals.md](01-patch-fundamentals.md)** - Detailed patch mechanics
- **[02-adaptive-patching.md](02-adaptive-patching.md)** - Content-aware strategies
- **[03-native-resolution.md](03-native-resolution.md)** - LLaVA-UHD modularization
- **[04-compression-modules.md](04-compression-modules.md)** - Token reduction techniques
- **[concepts/01-patch-size-tradeoffs.md](../concepts/01-patch-size-tradeoffs.md)** - Resolution vs efficiency
- **[comparisons/00-approaches-compared.md](../comparisons/00-approaches-compared.md)** - Strategy comparison

---

**Next**: Dive into [patch fundamentals](01-patch-fundamentals.md) for implementation details
