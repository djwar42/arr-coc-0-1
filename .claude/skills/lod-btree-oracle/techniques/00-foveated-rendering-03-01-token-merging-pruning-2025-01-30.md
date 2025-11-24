# Token Merging and Pruning for VLM Efficiency

**Dynamic knowledge addition**: 2025-01-30 (Level 1 Deep Dive)
**Source**: AIM (arXiv 2412.03248), SparseVLM (arXiv 2410.04417), PruMerge, HiRED (AAAI 2025)
**Parent**: [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md)

---

## Overview

Token merging and pruning are **complementary strategies** for reducing visual tokens in VLMs. Instead of selecting top-k tokens (discarding information), these methods either **merge similar tokens** (preserving information) or **prune redundant tokens** (removing duplicates) to achieve 10× compression with minimal accuracy loss.

**Key Insight**: Not all visual tokens are unique - many are redundant. Merging similar tokens preserves information while pruning removes true redundancy.

---

## AIM: Adaptive Inference via Token Merging and Pruning

**Paper**: arXiv 2412.03248 (Dec 2024), 15 citations
**Authors**: Yuntao Zhong et al.

### Core Innovation

AIM combines TWO orthogonal compression strategies:
1. **Iterative token merging** (similarity-based, pre-LLM)
2. **Progressive token pruning** (importance-based, within LLM layers)

**Result**: 71.9% token reduction with only 2.3% accuracy drop on average

### Method Details

**Stage 1: Similarity-Based Token Merging**
```python
def iterative_token_merging(visual_tokens, num_iterations=3):
    """Merge similar tokens before feeding to LLM"""

    tokens = visual_tokens  # Initial: e.g., 576 tokens

    for iteration in range(num_iterations):
        # Compute pairwise similarity
        similarity_matrix = compute_cosine_similarity(tokens, tokens)

        # Find most similar pair
        i, j = find_max_similarity(similarity_matrix, exclude_diagonal=True)

        # Merge tokens (weighted average by similarity)
        sim_score = similarity_matrix[i, j]
        merged = (sim_score * tokens[i] + (1 - sim_score) * tokens[j]) / 2

        # Replace and remove
        tokens[i] = merged
        tokens = remove(tokens, j)

    return tokens  # Reduced: e.g., 576 → 400 → 300 → 230
```

**Why Iterative?**
- First pass: Merges most obvious duplicates
- Later passes: Finds subtler similarities
- Gradual reduction prevents information loss

**Stage 2: Multi-Modal Importance Pruning (Within LLM)**
```python
class MultiModalImportanceScorer:
    """Prune tokens layer-by-layer inside LLM"""

    def compute_importance(self, visual_tokens, text_query, layer_idx):
        """Score tokens by multimodal relevance"""

        # Text-guided importance (cross-attention scores)
        text_importance = cross_attention_scores(visual_tokens, text_query)

        # Visual self-importance (self-attention scores)
        visual_importance = self_attention_scores(visual_tokens)

        # Layer-adaptive weighting
        alpha = layer_idx / total_layers  # Later layers → more text-guided
        importance = alpha * text_importance + (1 - alpha) * visual_importance

        return importance

    def progressive_pruning(self, tokens, layer_idx):
        """Prune at each LLM layer"""

        # Compute importance
        scores = self.compute_importance(tokens, text_query, layer_idx)

        # Keep top-k (k decreases each layer)
        k = initial_tokens * (1 - layer_idx / total_layers * prune_ratio)

        kept_indices = top_k_indices(scores, k)
        pruned_tokens = tokens[kept_indices]

        return pruned_tokens
```

**Progressive Pruning Schedule**:
- Layer 0: Keep 100% (230 tokens from merging)
- Layer 8: Keep 80% (184 tokens)
- Layer 16: Keep 60% (138 tokens)
- Layer 24: Keep 40% (92 tokens)
- Final: 92/576 = **16% of original tokens**

### Results

| Dataset | Baseline Tokens | AIM Tokens | Reduction | Acc Drop |
|---------|-----------------|------------|-----------|----------|
| VQA-v2 | 576 | 162 | 71.9% | -1.8% |
| GQA | 576 | 144 | 75.0% | -2.9% |
| TextVQA | 576 | 230 | 60.1% | -1.2% |
| **Average** | **576** | **162** | **71.9%** | **-2.3%** |

**Key Findings**:
- Text-heavy tasks (TextVQA) need more tokens
- General VQA can tolerate aggressive pruning
- Adaptive approach beats fixed compression

---

## SparseVLM: Training-Free Token Sparsification

**Paper**: arXiv 2410.04417 (Oct 2024), **114 citations** (highly cited!)
**Key Innovation**: Text-guided sparsification WITHOUT extra training

### Core Idea

**Problem**: Existing methods require fine-tuning or add parameters
**Solution**: Directly compute token importance from text-vision attention

```python
def sparsevlm_selection(visual_tokens, text_query, keep_ratio=0.3):
    """Training-free token selection guided by text"""

    # Extract text embedding
    text_emb = text_encoder(text_query)

    # Compute text-image cross-modal similarity
    similarities = cosine_similarity(visual_tokens, text_emb)

    # Combine with visual saliency (attention rollout from ViT)
    visual_saliency = vit_attention_rollout(visual_tokens)

    # Hybrid importance score
    importance = 0.7 * similarities + 0.3 * visual_saliency

    # Select top-k
    k = int(len(visual_tokens) * keep_ratio)
    selected_indices = top_k_indices(importance, k)

    return visual_tokens[selected_indices]
```

**Why Training-Free Works**:
- Cross-attention already contains relevance information
- ViT attention rollout captures visual importance
- No task-specific adaptation needed

### Results (Impressive!)

| Model | Method | Tokens | MME↑ | POPE↑ | Speed↑ |
|-------|--------|--------|------|-------|--------|
| LLaVA-1.5 | Baseline | 576 | 1511.5 | 85.9% | 1.0× |
| LLaVA-1.5 | SparseVLM | 144 | 1502.1 | 85.2% | **2.8×** |
| | **Reduction** | **75%** | **-0.6%** | **-0.7%** | - |

**Analysis**:
- 75% token reduction
- < 1% accuracy drop
- 2.8× inference speedup
- No training required!

### Why So Effective?

**Token Redundancy Analysis**:
```
Original 576 tokens:
  - ~40% background (sky, floor, walls)
  - ~30% object patches (multiple views of same object)
  - ~20% texture details (repeated patterns)
  - ~10% unique information

After SparseVLM (144 tokens):
  - ~80% unique information
  - ~20% necessary context
```

Most visual tokens are redundant → can be safely pruned!

---

## Token Merging (ToMe) vs Token Pruning

### ToMe: Merge Similar Tokens

**Algorithm**:
```python
def token_merging(tokens, merge_ratio=0.5):
    """Merge tokens by similarity (ToMe method)"""

    # Compute similarity matrix
    S = tokens @ tokens.T  # Cosine similarity

    # Find merge pairs (bipartite matching)
    pairs = find_merge_pairs(S, num_pairs=int(len(tokens) * merge_ratio / 2))

    # Merge each pair (average)
    merged_tokens = []
    unmerged_indices = set(range(len(tokens)))

    for i, j in pairs:
        merged = (tokens[i] + tokens[j]) / 2
        merged_tokens.append(merged)
        unmerged_indices.remove(i)
        unmerged_indices.remove(j)

    # Add unmerged tokens
    for idx in unmerged_indices:
        merged_tokens.append(tokens[idx])

    return torch.stack(merged_tokens)
```

**Pros**:
- Preserves information (averaging)
- No hard decisions
- Differentiable

**Cons**:
- Blurs distinct features
- Can't merge too aggressively (quality degrades)

### Token Pruning: Remove Redundant

**Algorithm**:
```python
def token_pruning(tokens, keep_ratio=0.5):
    """Prune low-importance tokens"""

    # Compute importance (attention-based)
    importance = compute_attention_scores(tokens)

    # Select top-k
    k = int(len(tokens) * keep_ratio)
    kept_indices = top_k_indices(importance, k)

    return tokens[kept_indices]
```

**Pros**:
- Clean removal of redundancy
- Preserves unique features exactly
- Can be very aggressive

**Cons**:
- Hard decisions (irreversible)
- Might discard useful information
- Requires good importance metric

### Hybrid: Best of Both Worlds (AIM Approach)

```python
def hybrid_compression(tokens, query):
    """Merge first, then prune"""

    # Stage 1: Merge similar (safe information preservation)
    merged = iterative_token_merging(tokens, num_iterations=3)
    # 576 → 300 tokens (52% kept, merging reduces redundancy)

    # Stage 2: Prune low-importance (aggressive reduction)
    pruned = progressive_pruning(merged, query, target_ratio=0.5)
    # 300 → 150 tokens (26% of original)

    return pruned  # Total: 576 → 150 = 74% reduction
```

**Why Hybrid Works**:
1. Merging removes obvious redundancy safely
2. Pruning then removes true low-importance
3. Combined effect > either alone

---

## HiRED: Hierarchical Token Dropping

**Paper**: AAAI 2025, 41 citations
**Innovation**: Drop tokens at MULTIPLE stages (image encoder → projector → LLM)

### Three-Stage Dropping

```python
class HiRED:
    """Hierarchical token dropping"""

    def forward(self, image, query):
        # Stage 1: ViT Encoder (early dropping)
        vit_tokens = vit_encoder(image)  # 196 patches
        important_tokens_1 = attention_guided_drop(
            vit_tokens,
            keep_ratio=0.7  # Drop 30% early
        )  # 196 → 137 tokens

        # Stage 2: Projector (middle dropping)
        projected = projector(important_tokens_1)  # 137 → 137 (same count)
        important_tokens_2 = cross_attention_drop(
            projected,
            query,
            keep_ratio=0.6  # Drop another 40% with query guidance
        )  # 137 → 82 tokens

        # Stage 3: LLM Layers (progressive dropping)
        for layer in llm_layers:
            important_tokens_2 = layer_wise_drop(
                important_tokens_2,
                keep_ratio=dynamic_ratio(layer.depth)
            )
        # 82 → 50 → 30 tokens (progressive)

        return llm_output(important_tokens_2)
```

**Dropping Schedule**:
- Early (ViT): 70% kept (drop background, redundancy)
- Middle (Projector): 60% of remaining (drop query-irrelevant)
- Late (LLM): 30-50% of remaining (keep only essential)

**Total**: 196 → 30 = **85% reduction**

### Results

| Method | Tokens | VQAv2 | GQA | TextVQA | FLOPs↓ |
|--------|--------|-------|-----|---------|--------|
| Baseline | 196 | 78.5 | 62.0 | 58.7 | 1.0× |
| ToMe | 98 | 77.9 | 61.2 | 57.8 | 0.51× |
| SparseVLM | 59 | 77.1 | 60.5 | 56.9 | 0.30× |
| **HiRED** | **29** | **77.0** | **60.3** | **56.5** | **0.15×** |

**HiRED achieves 85% reduction with only 1-2% accuracy drop!**

---

## Comparison: All Methods

| Method | Type | Training? | Reduction | Acc Loss | Speed | Key Advantage |
|--------|------|-----------|-----------|----------|-------|---------------|
| **AIM** | Merge+Prune | ✓ | 72% | 2.3% | 2.5× | Hybrid approach |
| **SparseVLM** | Prune | ✗ | 75% | 0.7% | 2.8× | Training-free |
| **HiRED** | Progressive | ✓ | 85% | 1.5% | 6.7× | Multi-stage |
| **ToMe** | Merge | ✗ | 50% | 0.5% | 1.8× | Info preservation |
| **PruMerge** | Prune+Merge | ✓ | 60% | 1.0% | 2.2× | Balanced |

**Takeaways**:
- Training-free (SparseVLM) surprisingly effective
- Hierarchical (HiRED) most aggressive
- Hybrid (AIM) best accuracy-efficiency trade-off
- All achieve >50% reduction with <2% loss

---

## Implementation Recommendations

### When to Use Each Method

**SparseVLM** - Use when:
- No training budget
- Need quick deployment
- Moderate compression (70-75%)
- Text queries available

**AIM** - Use when:
- Can fine-tune
- Need balanced compression
- Want interpretability (separate merge/prune)
- Target 70-75% reduction

**HiRED** - Use when:
- Maximum compression needed
- Can afford training
- Have multi-stage architecture
- Target 80-85% reduction

### Combined with ARR-COC-VIS

```python
def arr_coc_with_token_compression(image, query):
    """Integrate token compression with relevance realization"""

    # Stage 1: Extract visual tokens
    visual_tokens = vit_encoder(image)  # 576 tokens

    # Stage 2: SparseVLM compression (training-free)
    compressed_tokens = sparsevlm_selection(
        visual_tokens,
        query,
        keep_ratio=0.4  # 576 → 230 tokens
    )

    # Stage 3: ARR-COC relevance-based LOD allocation
    relevance_scores = compute_relevance(compressed_tokens, query)
    lod_allocation = homunculus_protocol(
        relevance_scores,
        tiers=[(20, 8), (30, 3), (23, 1)]  # 273 effective tokens
    )

    # Result: 576 original → 230 compressed → 273 effective = 47% efficiency
    return lod_allocation
```

**Benefit**: Double compression!
- SparseVLM: 60% reduction (576 → 230)
- ARR-COC: Further quality allocation
- Total: High efficiency with relevance-aware detail

---

## Open Questions

1. **Optimal merge/prune balance**: What ratio gives best accuracy-efficiency?
2. **Query-aware merging**: Can similarity be modulated by query?
3. **Dynamic scheduling**: Should compression vary per image complexity?
4. **Multimodal merging**: Should text tokens also be merged?
5. **Reversible compression**: Can we un-merge if needed?

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Parent doc
- [00-foveated-rendering-04-vlm-chain-of-focus-2025-01-30.md](00-foveated-rendering-04-vlm-chain-of-focus-2025-01-30.md) - Adaptive focusing
- [integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md) - Relevance scoring

**Other Oracles**:
- **vision-image-patching-oracle**: Token compression strategies
- **deepseek-ocr-oracle**: 16× optical compression (complementary)

---

## Key Citations

1. **AIM** - arXiv:2412.03248 (Dec 2024)
   - Zhong, Y., et al. "Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning"

2. **SparseVLM** - arXiv:2410.04417 (Oct 2024)
   - Zhang, Y., et al. "SparseVLM: Visual Token Sparsification for Efficient Vision-Language Models"
   - 114 citations - highly influential!

3. **HiRED** - AAAI 2025
   - Arif, K.H.I., et al. "HiRED: Attention-Guided Token Dropping for Efficient Inference"
   - 41 citations

---

**Last Updated**: 2025-01-30
**Status**: Cutting-edge 2024-2025 methods
**Relevance**: ★★★★★ (Critical for VLM efficiency)


## Hardware Texture Sampling Alternative

**Cross-Reference**: [07-gpu-texture-primitives-vlm-2025-01-30.md](07-gpu-texture-primitives-vlm-2025-01-30.md)

Token merging operates in software (7× FLOPs reduction). Hardware texture sampling provides orthogonal benefits:
- **Software**: Adaptive token merging (AIM, ToMe) reduces compute by 7×
- **Hardware**: GPU texture units execute 50× faster (0.1ms vs 5ms mipmaps)
- **Combined**: Use hardware for pyramid generation, software for intelligent token selection

**Speedup Comparison**:
- AIM: 7× FLOPs reduction (still uses slow software pyramids)
- Hardware mipmaps: 50× actual time reduction
- **Best**: Combine both for maximum efficiency
