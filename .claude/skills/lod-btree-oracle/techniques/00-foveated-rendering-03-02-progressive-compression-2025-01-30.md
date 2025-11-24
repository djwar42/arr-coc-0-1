# Progressive Visual Token Compression

**Dynamic knowledge addition**: 2025-01-30 (Level 2 Deep Dive)
**Source**: PVC (CVPR 2025), FastVLM (CVPR 2025, 27 citations), Token Compensator (arXiv 2408.06798)
**Parent**: [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md)

---

## Overview

Progressive compression reduces visual tokens **incrementally across multiple stages** rather than in a single step. By compressing gradually (encoder → projector → LLM layers), the model can adapt compression based on accumulated context, achieving better accuracy-efficiency trade-offs than one-shot compression.

**Key Insight**: Early layers need many tokens to understand the image. Later layers can work with fewer tokens as understanding consolidates.

---

## PVC: Progressive Visual Token Compression

**Paper**: CVPR 2025, Yang et al.
**Innovation**: Unified image+video processing with temporal redundancy exploitation

### Core Concept

**Problem**: Images and videos processed separately → inefficient
**Solution**: Treat images as 1-frame videos, compress using temporal redundancy

```python
class ProgressiveVisualCompressor:
    """PVC: Progressive compression for images and videos"""

    def forward(self, visual_input):
        # Unify: Image → 1-frame video, Video → T-frame sequence
        if is_image(visual_input):
            frames = visual_input.unsqueeze(0)  # [H,W,C] → [1,H,W,C]
        else:
            frames = visual_input  # Already [T,H,W,C]

        T, H, W, C = frames.shape

        # Stage 1: Spatial compression (per-frame)
        spatial_tokens = []
        for frame in frames:
            tokens = vit_encoder(frame)  # [196, D]
            compressed = spatial_compressor(tokens)  # 196 → 98
            spatial_tokens.append(compressed)

        spatial_tokens = torch.stack(spatial_tokens)  # [T, 98, D]

        # Stage 2: Temporal compression (across frames)
        if T > 1:
            # Exploit temporal redundancy
            temporal_tokens = temporal_compressor(spatial_tokens)
            # [T, 98, D] → [T', 98, D] where T' < T
        else:
            # Single frame (image): no temporal compression
            temporal_tokens = spatial_tokens

        # Stage 3: Progressive layer-wise compression
        tokens = temporal_tokens.flatten(0, 1)  # [T'*98, D]

        for layer_idx, layer in enumerate(llm_layers):
            # Compress more aggressively in later layers
            compression_ratio = 1 - (layer_idx / len(llm_layers)) * 0.6
            num_keep = int(len(tokens) * compression_ratio)

            # Importance-based selection
            importance = layer.compute_importance(tokens)
            tokens = tokens[top_k_indices(importance, num_keep)]

            # Process through layer
            tokens = layer(tokens)

        return llm_head(tokens)
```

### Temporal Redundancy Compression

**Key Innovation**: Adjacent video frames are highly similar

```python
def temporal_compressor(frame_tokens):
    """Compress across time dimension"""
    T, N, D = frame_tokens.shape  # T frames, N tokens/frame, D dims

    # Compute inter-frame similarity
    similarities = []
    for t in range(1, T):
        # Similarity between frame t and t-1
        sim = cosine_similarity(frame_tokens[t], frame_tokens[t-1])
        similarities.append(sim)

    # Merge similar frames
    merged_frames = [frame_tokens[0]]  # Always keep first frame

    for t in range(1, T):
        if similarities[t-1].mean() > threshold:  # High similarity
            # Merge with previous frame (weighted by similarity)
            alpha = similarities[t-1].mean()
            merged = alpha * merged_frames[-1] + (1 - alpha) * frame_tokens[t]
            merged_frames[-1] = merged  # Update last frame
        else:
            # Low similarity, add as new frame
            merged_frames.append(frame_tokens[t])

    return torch.stack(merged_frames)  # [T', N, D] where T' <= T
```

**Result**: 30-second video (900 frames) → ~45 merged frames (95% reduction)

### Progressive Compression Schedule

| Stage | Input Tokens | Output Tokens | Method | Reduction |
|-------|--------------|---------------|--------|-----------|
| Spatial | 196/frame | 98/frame | ViT pooling | 50% |
| Temporal | T × 98 | T' × 98 | Frame merging | 60-90% |
| Layer 0 | T' × 98 | 100% kept | - | 0% |
| Layer 8 | T' × 98 | 70% kept | Importance | 30% |
| Layer 16 | 70% | 50% kept | Importance | 50% |
| Layer 24 | 50% | 40% kept | Importance | 60% |

**Total**: 176,400 tokens (900 frames) → **~1,800 tokens** (99% reduction!)

### Results

| Task | Baseline | PVC | Reduction | Acc Change |
|------|----------|-----|-----------|------------|
| Image VQA | 196 tok | 78 tok | 60% | -0.8% |
| Video QA | 900×196 | 1800 tok | 99% | -2.1% |
| Image Caption | 196 tok | 98 tok | 50% | +0.3% |
| Video Caption | 900×196 | 2400 tok | 98.6% | -1.5% |

**Surprising finding**: Image captioning IMPROVES with compression (removes noise)

---

## FastVLM: Efficient Vision Encoding

**Paper**: CVPR 2025, 27 citations
**Authors**: PKA Vasu et al. (Apple ML Research)
**Innovation**: Hybrid architecture with learned compression

### FastViTHD Architecture

**Two-Path Design**:
```python
class FastViTHD:
    """Fast Vision Transformer for High Definition images"""

    def __init__(self):
        # Low-resolution path (fast, global context)
        self.fast_path = MobileViT(resolution=224)  # Efficient backbone

        # High-resolution path (slow, local details)
        self.detail_path = ViT(resolution=672)  # Standard ViT

        # Hybrid fusion
        self.fusion = HybridTokenFusion()

    def forward(self, image):
        # Path 1: Low-res global (fast)
        low_res = resize(image, 224)
        global_tokens = self.fast_path(low_res)  # 49 tokens

        # Path 2: High-res details (adaptive)
        if requires_details(image, query):  # Query-aware decision
            high_res = resize(image, 672)
            detail_tokens = self.detail_path(high_res)  # 196 tokens

            # Compress detail tokens (keep only novel information)
            novel_details = remove_redundant_with_global(
                detail_tokens,
                global_tokens
            )  # 196 → ~40 tokens

            # Fuse
            fused = self.fusion(global_tokens, novel_details)  # 49 + 40 = 89
        else:
            # Simple query: use only global
            fused = global_tokens  # 49 tokens

        return fused
```

**Key Innovation**: Adaptive detail path activation

**Decision Criterion**:
```python
def requires_details(image, query):
    """Decide if high-res path needed"""

    # Check query complexity
    query_words = len(query.split())
    has_spatial_terms = any(word in query.lower()
                           for word in ['small', 'text', 'corner', 'background'])

    # Check image complexity
    edge_density = compute_edges(image).mean()
    text_likelihood = ocr_detector.score(image)

    # Decision logic
    if query_words > 15 or has_spatial_terms:
        return True  # Complex query needs details
    if edge_density > 0.3 or text_likelihood > 0.5:
        return True  # Complex image needs details
    return False  # Simple case: global sufficient
```

### Learned Token Compression

Unlike rule-based compression, FastVLM **learns** how to compress:

```python
class LearnedCompressor(nn.Module):
    """Neural network that learns optimal compression"""

    def __init__(self):
        self.importance_network = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.merger_network = nn.Sequential(
            nn.Linear(768 * 2, 768),  # Input: concat of two tokens
            nn.GELU(),
            nn.Linear(768, 768)  # Output: merged token
        )

    def forward(self, tokens, target_count):
        # Compute importance scores
        importance = self.importance_network(tokens).squeeze(-1)

        # Select top-k most important
        kept_indices = top_k_indices(importance, target_count)
        kept_tokens = tokens[kept_indices]

        # Merge remaining into kept tokens
        discarded_indices = set(range(len(tokens))) - set(kept_indices.tolist())

        for disc_idx in discarded_indices:
            # Find most similar kept token
            similarities = cosine_similarity(
                tokens[disc_idx].unsqueeze(0),
                kept_tokens
            )
            most_similar_idx = similarities.argmax()

            # Learned merging (not simple average!)
            concat = torch.cat([
                kept_tokens[most_similar_idx],
                tokens[disc_idx]
            ])
            merged = self.merger_network(concat)

            kept_tokens[most_similar_idx] = merged

        return kept_tokens
```

**Training**: End-to-end with VQA task loss
- Learns what information to preserve
- Learns how to merge effectively
- Task-specific optimization

### Results

| Method | Avg Tokens | MME | POPE | TextVQA | Latency |
|--------|------------|-----|------|---------|---------|
| LLaVA-1.5 | 576 | 1511 | 85.9 | 58.3 | 183ms |
| FastVLM-Small | 89 | 1505 | 85.1 | 57.8 | **67ms** |
| FastVLM-Adaptive | 49-196 | 1509 | 85.6 | 58.1 | **85ms** |

**FastVLM achieves 2.7× speedup with <1% accuracy loss!**

---

## Token Compensator: Altering Inference Cost

**Paper**: arXiv 2408.06798 (Aug 2024), 9 citations
**Innovation**: Dynamic token budget allocation during inference

### Core Idea

**Problem**: Different inputs need different token budgets
**Solution**: Allocate tokens based on estimated difficulty

```python
class TokenCompensator:
    """Dynamically adjust token count per input"""

    def __init__(self):
        self.difficulty_estimator = DifficultyEstimator()
        self.token_budgets = {
            'easy': 64,
            'medium': 144,
            'hard': 256,
            'very_hard': 400
        }

    def forward(self, image, query):
        # Estimate difficulty
        difficulty = self.difficulty_estimator(image, query)

        # Allocate budget
        if difficulty < 0.3:
            budget = self.token_budgets['easy']
        elif difficulty < 0.6:
            budget = self.token_budgets['medium']
        elif difficulty < 0.85:
            budget = self.token_budgets['hard']
        else:
            budget = self.token_budgets['very_hard']

        # Extract tokens with budget constraint
        visual_tokens = self.encoder(image)
        compressed = self.compress_to_budget(visual_tokens, budget)

        return compressed

class DifficultyEstimator(nn.Module):
    """Estimate how hard a VQA sample is"""

    def forward(self, image, query):
        # Image complexity features
        edge_density = compute_edges(image).mean()
        color_variance = compute_color_variance(image)
        object_count = count_objects(image)  # Rough estimate

        # Query complexity features
        query_length = len(query.split())
        has_negation = 'not' in query.lower() or "n't" in query.lower()
        has_counting = any(word in query.lower()
                          for word in ['how many', 'count', 'number'])
        has_reasoning = any(word in query.lower()
                           for word in ['why', 'because', 'reason'])

        # Combine features
        image_difficulty = (
            0.3 * edge_density +
            0.2 * color_variance +
            0.5 * (object_count / 10)  # Normalize
        )

        query_difficulty = (
            0.2 * (query_length / 20) +  # Normalize
            0.3 * float(has_negation) +
            0.3 * float(has_counting) +
            0.2 * float(has_reasoning)
        )

        total_difficulty = 0.6 * image_difficulty + 0.4 * query_difficulty
        return total_difficulty.clip(0, 1)
```

### Adaptive Compression

```python
def compress_to_budget(visual_tokens, budget):
    """Compress tokens to meet budget"""

    if len(visual_tokens) <= budget:
        return visual_tokens  # Already under budget

    # Progressive compression
    current = visual_tokens

    while len(current) > budget:
        # How much to compress?
        compression_needed = len(current) / budget

        if compression_needed > 2.0:
            # Aggressive: merge similar tokens first
            current = token_merging(current, keep_ratio=0.7)
        elif compression_needed > 1.5:
            # Moderate: prune low-importance
            current = token_pruning(current, keep_ratio=0.8)
        else:
            # Gentle: remove only most redundant
            current = remove_duplicates(current, threshold=0.95)

    return current
```

### Results

| Difficulty | Avg Budget | Accuracy | Latency | Savings |
|------------|------------|----------|---------|---------|
| Easy (30%) | 64 tok | 89.2% | 45ms | 88% |
| Medium (40%) | 144 tok | 84.5% | 78ms | 75% |
| Hard (20%) | 256 tok | 79.1% | 121ms | 56% |
| Very Hard (10%) | 400 tok | 73.8% | 178ms | 31% |
| **Weighted Avg** | **134 tok** | **84.2%** | **74ms** | **77%** |

**Compared to fixed 400 tokens**:
- Token Compensator: 134 avg (77% savings)
- Accuracy: 84.2% vs 82.5% (BETTER with adaptive!)
- Latency: 74ms vs 178ms (2.4× faster)

**Why better accuracy?** Easy samples don't need many tokens → cleaner signal, less noise

---

## Comparison: Progressive vs One-Shot

| Compression | Stages | Adaptiveness | Reduction | Acc Loss | Latency |
|-------------|--------|--------------|-----------|----------|---------|
| **One-Shot** | 1 | Static | 60% | 2.5% | Low |
| **PVC** | 3 | Per-stage | 60-99% | 0.8-2.1% | Low |
| **FastVLM** | 2 | Query-aware | 50-85% | <1% | Lowest |
| **Token Comp** | Progressive | Difficulty-aware | 77% avg | <1% | Low |

**Takeaway**: Progressive compression consistently beats one-shot

---

## Integration Strategies

### With ARR-COC-VIS

```python
def arr_coc_with_progressive_compression(image, query):
    """Combine progressive compression with relevance realization"""

    # Stage 1: Initial encoding
    visual_tokens = vit_encoder(image)  # 576 tokens

    # Stage 2: Difficulty-aware compression (Token Compensator)
    difficulty = estimate_difficulty(image, query)
    initial_budget = difficulty_to_budget(difficulty)  # 64-400
    compressed = compress_to_budget(visual_tokens, initial_budget)

    # Stage 3: ARR-COC relevance-based allocation
    relevance = compute_relevance(compressed, query)
    lod_allocation = homunculus_protocol(
        relevance,
        tiers=[(20, 8), (30, 3), (23, 1)]  # 273 effective
    )

    # Stage 4: Progressive LLM processing
    for layer in llm_layers:
        lod_allocation = progressive_compression(lod_allocation, layer.depth)

    return lod_allocation
```

**Result**: Multi-stage efficiency!

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md)
- [00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md](00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md)
- [integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md)

---

## Key Citations

1. **PVC** - CVPR 2025
   - Yang, C., et al. "Progressive Visual Token Compression for Unified Image and Video Processing"

2. **FastVLM** - CVPR 2025 (27 citations)
   - Vasu, P.K.A., et al. "FastVLM: Efficient Vision Encoding for Vision Language Models"

3. **Token Compensator** - arXiv:2408.06798 (Aug 2024)
   - Jie, S., et al. "Token Compensator: Altering Inference Cost of Vision Transformers"

---

**Last Updated**: 2025-01-30
**Status**: CVPR 2025 cutting-edge methods
**Relevance**: ★★★★★ (Essential for efficient VLMs)
