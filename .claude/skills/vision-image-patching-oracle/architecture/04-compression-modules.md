# Token Compression Modules

**Reducing visual token count while preserving semantic information**

## Overview

Token compression modules address the computational bottleneck in vision-language models by reducing the number of visual tokens that must be processed by the LLM, while maintaining the semantic richness needed for accurate understanding.

## The Token Explosion Problem

### Scale of the Challenge

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:

**Image tokens**:
- Standard ViT-L/14 at 336×336: **576 tokens**
- High-resolution 896×896: **4,096 tokens**
- LLaVA-UHD 672×1008 (6 slices): **3,456 tokens** (before compression)

**Computational impact**:
- LLM attention: O(n²) complexity
- 4,096 tokens: **16× more compute** than 1,024 tokens
- Memory: Scales quadratically with token count

**Key insight**: Visual tokens often contain significant redundancy

## Compression Approaches

### 1. Transformation-Based Compression

**Principle**: Use learned transformations to merge/reduce tokens

#### Perceiver Resampler

**From [source-documents/09_Efficient Vision-Language Pretraining](../source-documents/09_Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment - BMVC 2022.md)**:

**Architecture**:
```python
class PerceiverResampler(nn.Module):
    def __init__(self, dim, depth, num_latents):
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(dim) for _ in range(depth)
        ])

    def forward(self, visual_tokens):
        # visual_tokens: (batch, 576, dim)
        # latents: (batch, 64, dim) - learnable query tokens

        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.cross_attention_layers:
            latents = layer(latents, visual_tokens)  # Q=latents, K=V=visual_tokens

        return latents  # (batch, 64, dim) - 9× compression
```

**Compression ratio**: 576 tokens → 64 tokens (9×)

**Benefits**:
- Fixed output size (predictable compute)
- Learnable queries (task-adaptive)
- Preserves most semantic information

#### C-Abstractor (Compression Abstractor)

**Used in**: LLaVA-UHD, Qwen-VL

**Architecture**: Multi-layer transformer with query-based pooling

**Process**:
1. Input: Visual tokens from ViT encoder
2. Compression: Transformer layers with learned query tokens
3. Output: Compressed representation

**Typical configuration**:
- Input: 576 tokens (336×336 image)
- Output: 64-144 tokens
- Compression: 4-9×

### 2. Similarity-Based Compression

**Principle**: Merge similar tokens to reduce redundancy

#### Token Merging (ToMe)

**From [source-documents/16_Token Pooling](../source-documents/16_Token Pooling in Vision Transformers for Image Classification.md)**:

**Algorithm**:
```python
def token_merging(tokens, similarity_threshold=0.9):
    """
    Merge tokens based on similarity

    Args:
        tokens: Visual tokens (N, dim)
        similarity_threshold: Threshold for merging

    Returns:
        merged_tokens: Compressed tokens (M, dim) where M < N
    """
    # Compute pairwise similarities
    similarities = cosine_similarity(tokens, tokens)

    # Identify similar pairs
    merge_pairs = find_pairs_above_threshold(similarities, similarity_threshold)

    # Merge tokens
    merged_tokens = []
    merged_set = set()

    for i in range(len(tokens)):
        if i in merged_set:
            continue

        # Find all tokens to merge with token i
        merge_group = [i] + [j for j in merge_pairs[i] if j not in merged_set]

        # Average tokens in group
        merged_token = tokens[merge_group].mean(dim=0)
        merged_tokens.append(merged_token)

        merged_set.update(merge_group)

    return torch.stack(merged_tokens)
```

**Characteristics**:
- **Training-free**: No additional parameters
- **Adaptive**: Compression ratio varies by image
- **Efficient**: Low computational overhead

**Typical results**: 20-40% token reduction with minimal accuracy loss

#### Hierarchical Patch Compression

**From [source-documents/10_Hierarchical Patch Compression](../source-documents/10_Hierarchical Patch Compression for ColPali_ Efficient Multi-Vector Document Retrieval with Dynamic Pruning and Quantization - SciTePress.md)**:

**Multi-level approach**:
1. **Layer 1**: Fine-grained patches (full resolution)
2. **Layer 2**: Merge similar adjacent patches
3. **Layer 3**: Further compression based on importance

**Benefits**: Preserves details where needed, compresses elsewhere

### 3. Attention-Based Compression

**Principle**: Use attention weights to identify important tokens

#### Attention-Weighted Token Pruning

**From [source-documents/03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)**:

**Algorithm**:
```python
def attention_based_pruning(tokens, attention_scores, keep_ratio=0.5):
    """
    Prune tokens based on attention importance

    Args:
        tokens: Visual tokens (N, dim)
        attention_scores: Cumulative attention from model (N,)
        keep_ratio: Fraction of tokens to keep

    Returns:
        pruned_tokens: Important tokens (N*keep_ratio, dim)
    """
    # Sort tokens by attention score
    sorted_indices = torch.argsort(attention_scores, descending=True)

    # Keep top-k tokens
    k = int(len(tokens) * keep_ratio)
    keep_indices = sorted_indices[:k]

    return tokens[keep_indices]
```

**Key insight**: Tokens with low attention scores contribute minimally

**Compression ratio**: Configurable (typically 2-4×)

**Advantage**: Preserves most-attended (important) information

### 4. Query-Based Compression

**Principle**: Compress differently based on user query

#### Query-Aware Token Selection

**From [source-documents/21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md)**:

**Concept**: Different queries need different visual information

**Example**:
- Query: "What color is the car?" → Focus on car region, compress background
- Query: "Describe the scene" → Balanced compression across image

**Implementation**:
```python
def query_aware_compression(visual_tokens, query_embedding, compression_net):
    """
    Compress visual tokens based on query

    Args:
        visual_tokens: Image tokens (N, dim)
        query_embedding: Text query embedding (dim,)
        compression_net: Learned compression network

    Returns:
        compressed_tokens: Query-relevant compressed tokens
    """
    # Compute query-token relevance
    relevance_scores = torch.matmul(visual_tokens, query_embedding)

    # Attention-weighted compression
    attention_weights = torch.softmax(relevance_scores, dim=0)

    # Learned compression with query guidance
    compressed = compression_net(visual_tokens, attention_weights, query_embedding)

    return compressed
```

**Benefits**: Maximally relevant compression for each query

## Specialized Compression Techniques

### For High-Resolution Images

#### Multi-Scale Compression

**Strategy**: Compress different scales differently

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:

- **Global features** (low-res): Minimal compression (important for context)
- **Local details** (high-res): Aggressive compression (redundant fine details)

#### Slice-Wise Compression

**Strategy**: Compress each image slice independently

**Used in**: LLaVA-UHD

**Process**:
1. Divide image into slices (e.g., 3×2 grid)
2. Encode each slice: 336×336 → 576 tokens
3. Compress each slice: 576 → 64 tokens
4. Result: 6 slices × 64 tokens = 384 tokens (vs 3,456 without compression)

### For Document Images

#### OCR-Aware Compression

**From [source-documents/07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)**:

**Key insight**: Text regions need preservation, backgrounds can be heavily compressed

**DeepSeek-OCR approach**:
- **16× optical compression** through specialized architecture
- **SAM (Segment Anything Model)**: Extract visual content
- **CLIP encoding**: Compress to semantic representations
- **Serial design**: Process sequentially for efficiency

**Compression ratio**: 576 tokens → 36 tokens (16×)

**Application**: Long documents with mixed text/images

## Compression Quality Metrics

### 1. Reconstruction Error

**Measure**: Can we reconstruct original tokens from compressed tokens?

**Formula**: `MSE = ||original - reconstructed||²`

**Limitation**: Low MSE doesn't guarantee semantic preservation

### 2. Task Performance

**Measure**: Accuracy on downstream tasks (VQA, captioning, etc.)

**Ideal**: Minimal performance drop with high compression

**Benchmark results** (from survey):
- 4× compression: <1% accuracy drop
- 9× compression: 1-3% accuracy drop
- 16× compression: 3-8% accuracy drop (DeepSeek-OCR maintains performance through specialized design)

### 3. Computational Savings

**Measure**: FLOPs reduction, latency improvement

**Formula**: `Speedup = (original_tokens / compressed_tokens)²` (for attention)

**Example**:
- Original: 4,096 tokens
- Compressed: 512 tokens (8× compression)
- Attention speedup: ~64×
- Overall speedup: ~20-30× (accounting for compression overhead)

## Design Tradeoffs

### Compression Ratio vs Quality

**Low compression** (2-4×):
- Pros: Minimal information loss, safe
- Cons: Limited computational savings

**Medium compression** (4-9×):
- Pros: Good balance, widely used
- Cons: Requires careful design

**High compression** (10-16×):
- Pros: Maximal efficiency
- Cons: Significant information loss (except specialized designs like DeepSeek-OCR)

### Fixed vs Adaptive Compression

**Fixed**:
- Pros: Predictable compute, simple batching
- Cons: Suboptimal for varied content

**Adaptive**:
- Pros: Optimal compression per image
- Cons: Variable compute, complex batching

### Training Cost

**Training-free** (ToMe, similarity-based):
- Pros: No additional training
- Cons: Limited compression ratio, may miss learned patterns

**Learned compression** (Perceiver, C-Abstractor):
- Pros: Task-optimized, higher quality
- Cons: Requires training data and compute

## Implementation Example: LLaVA-UHD Compression Module

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

```python
class CompressionModule(nn.Module):
    """
    LLaVA-UHD compression module
    Reduces visual tokens from 576 to 64-144 tokens
    """
    def __init__(self, input_dim=1024, num_queries=64, num_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, input_dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=16)
            for _ in range(num_layers)
        ])

    def forward(self, visual_tokens):
        # visual_tokens: (batch, 576, 1024) from ViT
        # queries: (batch, 64, 1024) learned

        batch_size = visual_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: Q=queries, K=V=visual_tokens
        for layer in self.layers:
            queries = layer(queries, visual_tokens)

        return queries  # (batch, 64, 1024) - 9× compression
```

## Practical Guidelines

**When to use compression**:
- High-resolution images (>672×672)
- Multi-image inputs
- Long sequences (video frames)
- Limited computational budget

**Compression ratio selection**:
- **4×**: Conservative, use for critical tasks
- **6-8×**: Balanced, general purpose (LLaVA-UHD default: 9×)
- **16×**: Specialized (documents with DeepSeek-OCR design)

**Architecture choice**:
- **Perceiver/C-Abstractor**: General purpose, fixed output
- **Token merging**: Training-free, adaptive
- **Attention pruning**: Query-dependent, moderate compression
- **Specialized (DeepSeek-OCR)**: Domain-specific (documents)

## Primary Sources

- [21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md) - Comprehensive compression overview
- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md) - Compression module implementation
- [07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md) - 16× optical compression
- [03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md) - Attention-based pruning
- [16_Token Pooling](../source-documents/16_Token Pooling in Vision Transformers for Image Classification.md) - Token merging strategies

## Related Documents

- [02-adaptive-patching.md](02-adaptive-patching.md) - Adaptive patch sizing (pre-compression)
- [03-native-resolution.md](03-native-resolution.md) - Native resolution with compression
- [../techniques/03-compression-strategies.md](../techniques/03-compression-strategies.md) - Practical compression techniques
- [../concepts/02-token-efficiency.md](../concepts/02-token-efficiency.md) - Token efficiency principles
