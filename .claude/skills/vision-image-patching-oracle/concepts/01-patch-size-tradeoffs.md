# Patch Size Tradeoffs

**The fundamental tradeoff between resolution detail and computational efficiency**

## The Core Tradeoff

Patch size determines how image is divided into tokens, creating a fundamental tension:

**Smaller patches** (e.g., 8×8):
- ✅ More detail, better fine-grained understanding
- ✅ Higher effective resolution
- ❌ More tokens → quadratic compute increase
- ❌ Higher memory usage

**Larger patches** (e.g., 32×32):
- ✅ Fewer tokens → efficient computation
- ✅ Lower memory footprint
- ❌ Less detail, missed fine structures
- ❌ Poor for text recognition, small objects

## Mathematical Analysis

### Token Count

For image of size H×W with patch size P×P:
```
num_tokens = (H/P) × (W/P) = (H×W) / P²
```

**Example**: 336×336 image
- 16×16 patches: (336/16)² = 21×21 = **441 tokens**
- 14×14 patches: (336/14)² = 24×24 = **576 tokens**
- 8×8 patches: (336/8)² = 42×42 = **1,764 tokens** (4× more!)

### Computational Cost

Self-attention complexity: O(n²) where n = number of tokens

**Example compute comparison** for 336×336:
- 16×16 patches: 441² = **194,481** operations
- 14×14 patches: 576² = **331,776** operations (1.7×)
- 8×8 patches: 1,764² = **3,111,696** operations (16×!)

**Key insight**: Halving patch size quadruples compute cost

## Resolution vs Efficiency Sweet Spots

### Standard Choices in Practice

**From [source-documents/18_Vision Transformer](../source-documents/18_Vision transformer - Wikipedia.md)** and model literature:

| Patch Size | Image Size | Tokens | Use Case | Models |
|-----------|------------|--------|----------|--------|
| 32×32 | 224×224 | 49 | Fast inference, low-res | Efficient ViT variants |
| 16×16 | 224×224 | 196 | Balanced (most common) | ViT-B, ViT-L, CLIP |
| 16×16 | 336×336 | 441 | Higher quality | LLaVA-1.5 |
| 14×14 | 336×336 | 576 | Fine detail | OpenCLIP, LLaVA-UHD |
| 14×14 | 672×672 | 2,304 | Very high-res | LLaVA-UHD slices |
| 8×8 | 224×224 | 784 | Research/specialized | Dense prediction tasks |

### The 14×14 and 16×16 Standard

**Why these sizes dominate**:
1. **Pretraining heritage**: CLIP and ImageNet models used these
2. **Balanced tradeoff**: Good detail without excessive tokens
3. **GPU efficiency**: Token counts align well with batch sizes
4. **Proven performance**: Extensive validation in literature

## Tradeoff Dimensions

### 1. Semantic Understanding vs Pixel Detail

**Coarse patches** (32×32):
- Capture global semantics well
- Good for: scene classification, general object recognition
- Poor for: text, small objects, fine textures

**Fine patches** (8×8):
- Capture pixel-level details
- Good for: OCR, dense prediction, texture analysis
- Poor for: efficiency, may overfit to noise

**From [source-documents/13_Patch Embedding as Local Features](../source-documents/13_Patch Embedding as Local Features_ Unifying Deep Local and Global Features Via Vision Transformer for Image Retrieval.md)**:
- Patches act as local features at different granularities
- Larger patches = global features
- Smaller patches = local features

### 2. Training Efficiency vs Inference Quality

**Training considerations**:
- Smaller patches → longer training (more FLOPs per image)
- Larger patches → faster training, more iterations possible
- **Tradeoff**: 2× smaller patches ≈ 4× training time

**Inference considerations**:
- Deployment constraints (edge devices) favor larger patches
- Cloud inference can afford smaller patches
- Real-time applications need larger patches

### 3. Context Window vs Resolution

**Fixed context budget** (e.g., 2048 tokens total for LLM):

**Strategy A**: Large patches, single high-res image
- 8×8 patches on 1024×1024 image = 16,384 tokens ❌ Exceeds budget

**Strategy B**: Medium patches, moderate-res
- 16×16 patches on 672×672 image = 1,764 tokens ✅ Within budget

**Strategy C**: Compression + small patches
- 8×8 patches → 16,384 tokens → compress 10× → 1,638 tokens ✅ Feasible

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:
Compression enables smaller patches within same token budget

## Multi-Scale Approaches

### Hybrid Patch Sizes

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:

**Concept**: Use different patch sizes for different purposes
- **Large patches** (32×32): Global context path
- **Small patches** (16×16): Detail path
- **Fusion**: Combine both representations

**Benefits**:
- Get both global and local information
- Optimize compute/quality tradeoff
- Better than single fixed patch size

### Adaptive Patch Sizing

**Dynamic approach**: Vary patch size by image region

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:
- High-detail regions: Small patches (8×8)
- Low-detail regions: Large patches (32×32)
- Effective resolution varies by content

**Result**: 30-40% token reduction with similar quality

## Practical Guidelines

### Patch Size Selection Framework

**Consider these factors**:

1. **Task requirements**:
   - OCR/text-heavy: Prefer 14×14 or smaller
   - Scene understanding: 16×16 sufficient
   - General vision-language: 14×14 or 16×16

2. **Computational budget**:
   - Unlimited: Go smaller (14×14)
   - Constrained: Go larger (16×16 or 32×32)
   - Mixed: Use compression or adaptive patching

3. **Pretrained model**:
   - Using CLIP: Match its patch size (typically 14×14 or 16×16)
   - Training from scratch: Experiment, but 16×16 is safe default

4. **Image resolution**:
   - Low-res (≤224×224): Larger patches acceptable (16×16, 32×32)
   - High-res (≥672×672): Prefer smaller patches (14×14) or use slicing

### Common Mistakes

❌ **Too small patches without compression**:
   - Problem: Token explosion, out-of-memory
   - Solution: Add compression module or use larger patches

❌ **Mismatched patch size from pretraining**:
   - Problem: Position embedding interpolation errors
   - Solution: Keep patch size same, vary image resolution instead

❌ **One-size-fits-all**:
   - Problem: Suboptimal for varied use cases
   - Solution: Use adaptive patching or multi-scale approaches

## Empirical Results

### Patch Size Ablation Studies

**From [source-documents/00_Comprehensive Study of Vision Transformers](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)**:

**ImageNet accuracy** (ViT-Base, 224×224):
- 32×32 patches: 75.3%
- 16×16 patches: **81.8%** (standard)
- 8×8 patches: 82.1% (+0.3%, but 4× compute)

**Key finding**: Diminishing returns below 16×16

**TextVQA accuracy** (from LLaVA papers):
- 16×16 patches @ 336×336: 58.2%
- 14×14 patches @ 336×336: **61.5%** (+3.3%)
- 14×14 patches @ 672×672 (sliced): **63.9%** (+5.7%)

**Key finding**: Smaller patches help text-heavy tasks significantly

## Future Directions

**From recent research (2024-2025)**:

1. **Learned patch sizing**: Models that learn optimal patch size per image/region
2. **Continuous patching**: Move beyond discrete patch grids
3. **Query-dependent patches**: Different patch sizes for different queries
4. **Neural architecture search**: Automatically find best patch configurations

## Primary Sources

- [00_Comprehensive Study of ViT](../source-documents/00_A Comprehensive Study of Vision Transformers in Image Classification Tasks - arXiv.md)
- [18_Vision Transformer Wikipedia](../source-documents/18_Vision transformer - Wikipedia.md)
- [21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)
- [14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)

## Related Documents

- [00-image-tokenization.md](00-image-tokenization.md) - How patches become tokens
- [02-token-efficiency.md](02-token-efficiency.md) - Token usage optimization
- [03-resolution-scaling.md](03-resolution-scaling.md) - Multi-resolution strategies
- [../architecture/01-patch-fundamentals.md](../architecture/01-patch-fundamentals.md) - Basic patching mechanics
