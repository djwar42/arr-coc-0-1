# Token Efficiency

**Optimizing token usage in vision-language models through redundancy reduction and intelligent compression**

## The Efficiency Problem

Vision transformers convert images to token sequences, but most images contain significant redundancy:
- Uniform backgrounds (sky, walls)
- Repeated patterns (textures, grids)
- Low-frequency regions (gradients, shadows)

**Challenge**: Standard patching treats all regions equally, wasting tokens on redundant content.

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:
Visual tokens can contain 60-80% redundancy depending on image content.

## Token Redundancy Analysis

### Spatial Redundancy

**Principle**: Nearby patches are highly correlated

**Example**: Sky region in 672×672 image
```
Standard patching (14×14):
- Sky occupies ~40% of image
- Generates ~900 tokens
- Information content: ~50 tokens worth (18× redundancy!)
```

**From [source-documents/06_Computer Vision](../source-documents/06_Computer Vision_ Algorithms and Applications.md)**:
Natural images exhibit strong local correlation (neighboring pixels differ by <5% on average)

### Semantic Redundancy

**Principle**: Many patches encode similar semantic information

**Example**: Texture analysis
```
Brick wall image:
- 441 patches (16×16 from 336×336)
- Semantic content: "brick pattern"
- Unique information: ~20 patches
- Redundancy ratio: 22:1
```

### Temporal Redundancy (Video)

**Principle**: Adjacent frames are nearly identical

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:
- Video frames at 30fps: ~95% similarity between adjacent frames
- Standard approach: Process each frame independently (massive redundancy)
- Efficient approach: Differential encoding (5-10× compression)

## Compression Ratio Metrics

### Defining Compression Ratio

```
Compression Ratio (CR) = Original Tokens / Compressed Tokens
```

**Example calculations**:
```
336×336 image with 14×14 patches:
- Original: 576 tokens
- 4× compression: 144 tokens (CR = 4.0)
- 16× compression: 36 tokens (CR = 16.0)
```

### Quality-Aware Metrics

**Problem**: Raw compression ratio ignores quality loss

**Solution**: Quality-adjusted efficiency score

```
Efficiency Score = Compression Ratio × Task Performance
```

**Example**:
```
Model A: 8× compression, 85% accuracy → Score: 6.8
Model B: 4× compression, 92% accuracy → Score: 3.68
Model C: 16× compression, 78% accuracy → Score: 12.5

Best: Model C (highest efficiency despite lower accuracy)
```

### Practical Compression Ratios

**From empirical literature**:

| Approach | Compression Ratio | Quality Impact | Use Case |
|----------|------------------|----------------|----------|
| Pooling (2×2) | 4× | Minimal (<2%) | General vision |
| Token Merging | 2-8× | Low (2-5%) | Balanced efficiency |
| Cross-Attention | 8-16× | Moderate (5-10%) | Query-focused tasks |
| DeepSeek-OCR | 16× | Low (3-7%) | Document understanding |
| Aggressive Pooling | 32× | High (15-25%) | Rough understanding |

**Reference**: [source-documents/07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)

## Token Budget Optimization

### Fixed Budget Allocation

**Scenario**: LLM context limit = 2048 tokens, reserve 1500 for text

**Visual token budget**: 548 tokens

**Options for 672×672 image**:

**Option A: Downsample**
```
Resize 672×672 → 336×336
Patch 14×14 → 576 tokens ❌ Exceeds budget
Resize 672×672 → 308×308
Patch 14×14 → 484 tokens ✅ Within budget (loses 46% resolution)
```

**Option B: Compress**
```
Keep 672×672 native resolution
Patch 14×14 → 2,304 tokens
Compress 5× → 461 tokens ✅ Within budget (keeps full resolution)
```

**Option C: Adaptive Patching**
```
High-detail regions: 14×14 patches (200 tokens)
Low-detail regions: 28×28 patches (300 tokens)
Total: 500 tokens ✅ Within budget (variable resolution)
```

### Dynamic Budget Allocation

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:

**Principle**: Allocate tokens based on query complexity

**Simple query** ("What color is the sky?"):
- Low detail needed → 144 tokens (4× compression)

**Complex query** ("Read all text in this document"):
- High detail needed → 576 tokens (1× compression)

**Implementation**:
```python
def allocate_budget(query_complexity: float, max_tokens: int):
    # complexity ∈ [0, 1]
    min_tokens = max_tokens // 4  # 4× compression
    budget = int(min_tokens + complexity * (max_tokens - min_tokens))
    return budget

# Examples:
allocate_budget(0.2, 576)  # → 173 tokens (simple query)
allocate_budget(0.8, 576)  # → 489 tokens (complex query)
```

## Compression Techniques

### 1. Spatial Pooling

**Method**: Aggregate neighboring tokens via pooling

```python
# Max pooling (2×2 → 4× compression)
pooled = F.max_pool2d(tokens.reshape(h, w, d), kernel_size=2)

# Average pooling (preserves more information)
pooled = F.avg_pool2d(tokens.reshape(h, w, d), kernel_size=2)
```

**Pros**:
- Simple, fast, deterministic
- No additional parameters

**Cons**:
- Fixed compression ratio
- Doesn't consider semantic importance

**Typical CR**: 4× (2×2 pooling)

### 2. Token Merging (ToMe)

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:

**Method**: Merge similar tokens based on cosine similarity

```python
def merge_tokens(tokens, target_count):
    # Compute pairwise similarities
    similarities = cosine_similarity(tokens, tokens)

    # Greedily merge most similar pairs
    while len(tokens) > target_count:
        i, j = find_most_similar_pair(similarities)
        tokens[i] = (tokens[i] + tokens[j]) / 2  # Average merge
        tokens = remove_index(tokens, j)

    return tokens
```

**Pros**:
- Content-aware merging
- Preserves semantic information
- Flexible compression ratio

**Cons**:
- O(n²) similarity computation
- May merge semantically distinct but visually similar tokens

**Typical CR**: 2-8× (tunable)

### 3. Cross-Attention Compression

**Method**: Use queries to select relevant tokens

**From [source-documents/09_Efficient Vision-Language Pretraining](../source-documents/09_Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment - BMVC 2022.md)**:

```python
def compress_via_attention(visual_tokens, query_embed, target_count):
    # Learn compressed token queries
    queries = nn.Parameter(torch.randn(target_count, embed_dim))

    # Cross-attention: queries attend to visual tokens
    compressed = cross_attention(queries, visual_tokens, visual_tokens)

    return compressed  # Shape: [target_count, embed_dim]
```

**Examples**: Q-Former (BLIP-2), Perceiver Resampler

**Pros**:
- Query-aware compression
- Learnable (optimized via training)
- High compression ratios possible

**Cons**:
- Requires training
- Additional parameters
- Query-independent tokens may lose information

**Typical CR**: 8-16× (Q-Former uses 32 queries for 576 tokens → 18× CR)

### 4. Optical Compression (Serial Encoding)

**From [source-documents/07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)**:

**Method**: Serial encoder architecture (SAM → CLIP) with deep compression

```python
# Conceptual architecture
image → SAM encoder (1024d features)
      → CLIP encoder (process SAM features as "visual words")
      → Deep cross-attention (16 layers)
      → Compressed output (73-421 tokens, 16× CR)
```

**Key innovation**: Treat encoder outputs as compressible signals

**Pros**:
- Extreme compression (16×) with minimal quality loss
- Preserves fine details (OCR accuracy)
- Learns optimal compression representation

**Cons**:
- Computationally expensive (serial encoding)
- Requires large-scale training
- Complex architecture

**Typical CR**: 16× (DeepSeek-OCR standard)

## Efficiency Metrics

### Tokens Per Megapixel (TPM)

**Definition**: Token count per million pixels

```
TPM = (num_tokens × 1,000,000) / (image_width × image_height)
```

**Comparisons**:
```
Standard ViT (336×336, 14×14):
TPM = (576 × 1,000,000) / (336 × 336) = 5,102 TPM

DeepSeek-OCR (336×336, 16× compression):
TPM = (73 × 1,000,000) / (336 × 336) = 647 TPM (7.9× more efficient)

LLaVA-UHD (672×672, sliced):
TPM = (2,304 × 1,000,000) / (672 × 672) = 5,102 TPM (same as standard)
```

### FLOPs Per Token

**Definition**: Compute operations per generated token

**Calculation**:
```
FLOPs = Encoder FLOPs + Compression FLOPs
```

**Example**:
```
ViT-L encoder: 190 GFLOPs (336×336)
No compression: 190 GFLOPs / 576 tokens = 330 MFLOPs/token

ViT-L + Q-Former: 190 + 15 = 205 GFLOPs
With compression: 205 GFLOPs / 32 tokens = 6,406 MFLOPs/token

Tradeoff: 18× fewer tokens, but 19× compute per token (net: slight savings)
```

### Memory Efficiency

**KV cache growth**:
```
Memory per token = 2 × num_layers × hidden_dim × precision_bytes
```

**Example** (LLaMA-7B: 32 layers, 4096 hidden dim, FP16):
```
Standard patching: 576 tokens
Memory: 576 × 2 × 32 × 4096 × 2 = 302 MB

16× compression: 36 tokens
Memory: 36 × 2 × 32 × 4096 × 2 = 18.9 MB (16× reduction)
```

**Impact**: Compression directly reduces memory usage linearly

## Practical Guidelines

### Compression Strategy Selection

**Task-based recommendations**:

**Document understanding (OCR-heavy)**:
- High detail required
- Use: Optical compression (DeepSeek-OCR style, 16×)
- Rationale: Preserves text while compressing backgrounds

**General VQA**:
- Moderate detail sufficient
- Use: Token merging (4-8×) or cross-attention (8×)
- Rationale: Balanced efficiency and quality

**Scene classification**:
- Low detail needed
- Use: Spatial pooling (4-8×) or aggressive merging (16×)
- Rationale: Global semantics preserved despite heavy compression

**Chart/diagram understanding**:
- Moderate-high detail
- Use: Adaptive patching or cross-attention (4-8×)
- Rationale: Focus tokens on chart elements, compress whitespace

### Compression Rate Tuning

**Guidelines for selecting compression ratio**:

**Light compression (2-4×)**:
- When: Plenty of compute budget, accuracy critical
- Methods: Spatial pooling, light token merging
- Quality impact: <2%

**Medium compression (4-8×)**:
- When: Balanced efficiency/quality needs (most common)
- Methods: Token merging, learned compression
- Quality impact: 2-5%

**Heavy compression (8-16×)**:
- When: Strict token budgets, efficiency critical
- Methods: Cross-attention, optical compression
- Quality impact: 5-10%

**Extreme compression (16×+)**:
- When: Ultra-low latency, coarse understanding acceptable
- Methods: Aggressive pooling, minimal query sets
- Quality impact: 10-25%

## Measuring Redundancy

### Entropy-Based Analysis

**From information theory**:

```python
def token_entropy(tokens):
    """Measure information content via entropy"""
    # Quantize token embeddings
    quantized = kmeans_quantize(tokens, n_clusters=256)

    # Compute histogram
    hist = np.bincount(quantized) / len(quantized)

    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy  # bits per token
```

**Interpretation**:
- High entropy (7-8 bits): Low redundancy, hard to compress
- Medium entropy (4-6 bits): Moderate redundancy, compress 2-4×
- Low entropy (1-3 bits): High redundancy, compress 8-16×

### Compression Potential Estimation

**Method**: Test multiple compression ratios, measure quality degradation

```python
def find_optimal_compression(image, task, quality_threshold=0.95):
    results = []
    for ratio in [1, 2, 4, 8, 16, 32]:
        compressed_tokens = compress(image, ratio)
        quality = evaluate(compressed_tokens, task)
        results.append((ratio, quality))

    # Find highest compression maintaining quality
    optimal = max([r for r, q in results if q >= quality_threshold])
    return optimal
```

**Example output**:
```
Ratio: 1×  → Quality: 1.00 (baseline)
Ratio: 2×  → Quality: 0.99 (negligible loss)
Ratio: 4×  → Quality: 0.97 (minimal loss)
Ratio: 8×  → Quality: 0.93 (moderate loss)
Ratio: 16× → Quality: 0.85 (significant loss)

Optimal: 4× (highest ratio above 0.95 threshold)
```

## Future Directions

**From recent research (2024-2025)**:

1. **Learned compression policies**: Models that predict optimal compression per image/region
2. **Multi-stage compression**: Coarse compression initially, selective decompression on demand
3. **Neural codecs**: End-to-end learned compression (like image/video codecs)
4. **Query-progressive refinement**: Start with heavy compression, add tokens as needed

## Primary Sources

- [21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)
- [07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)
- [09_Efficient Vision-Language Pretraining](../source-documents/09_Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment - BMVC 2022.md)
- [12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)

## Related Documents

- [01-patch-size-tradeoffs.md](01-patch-size-tradeoffs.md) - Resolution vs efficiency fundamentals
- [03-resolution-scaling.md](03-resolution-scaling.md) - Multi-resolution efficiency
- [../architecture/04-compression-modules.md](../architecture/04-compression-modules.md) - Compression implementations
- [../techniques/03-compression-strategies.md](../techniques/03-compression-strategies.md) - Practical compression methods
