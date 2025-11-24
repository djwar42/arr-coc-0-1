# Attention-Driven Pyramid Pruning for VLMs

Query-aware mipmap level selection, sparse pyramid sampling, and dynamic token budget allocation for vision-language models. This document explores how attention mechanisms guide efficient pyramid traversal and token pruning strategies.

---

## Section 1: Query-Aware Mipmap Level Selection (~80 lines)

### Cross-Attention Scores → LOD Allocation

Modern VLMs use cross-attention between text queries and visual pyramid levels to determine which resolution layers are most relevant for a given task. The key insight: **not all queries require all pyramid levels**.

**Dynamic Level Selection Mechanism:**

```python
# Simplified query-aware level selector
def select_pyramid_levels(query_embed, pyramid_features):
    """
    Args:
        query_embed: [B, D] - text query embedding
        pyramid_features: List of [B, H_i, W_i, C] - features at each level
    Returns:
        selected_levels: List of level indices to process
        token_budgets: Tokens allocated per level
    """
    level_scores = []
    for level_idx, feats in enumerate(pyramid_features):
        # Compute query-feature alignment
        pooled_feat = global_pool(feats)  # [B, C]
        score = cosine_similarity(query_embed, pooled_feat)
        level_scores.append(score)

    # Select top-K levels based on scores
    selected_levels = torch.topk(level_scores, k=3).indices
    return selected_levels
```

**From Research (Pyramid Token Pruning, arXiv:2509.15704v2):**

The **Pyramid Token Pruning (PTP)** method introduces hierarchical pruning that combines:
- **Region-level importance**: Assign budgets to sub-images based on visual saliency
- **Token-level bottom-up scoring**: Use CLS-to-patch attention from vision encoder
- **Instruction-guided top-down scoring**: Leverage text-to-vision attention in LLM

PTP achieves 50% token reduction with 99.8% performance retention on InternVL2-2B across 13 benchmarks.

**Key Finding**: Query embedding guides which pyramid levels receive higher token budgets. Fine-grained queries (e.g., "count the red apples") allocate more tokens to high-resolution levels, while coarse queries (e.g., "describe the scene") use mid-level representations.

### Fine-Grained vs Coarse-Grained Queries

**Coarse queries** benefit from mid-level pyramid features:
- "What is in this image?" → Levels 2-3 (downsampled 4-8×)
- "Describe the overall scene" → Level 2-4
- Global semantic understanding requires less spatial detail

**Fine-grained queries** require high-resolution levels:
- "Read the text on the sign" → Level 0-1 (original or 2× downsampled)
- "Count objects smaller than X" → Levels 0-2
- OCR, small object detection, texture analysis

**From SparseVLM Research (arXiv:2410.04417):**

SparseVLM achieves **54% FLOPs reduction, 37% latency decrease** while maintaining 97% accuracy on LLaVA by using text-guided token optimization. The method:
- Uses self-attention matrices to rate visual token importance
- Adaptively determines sparsification ratio per layer
- Recycles pruned tokens into compressed representations

**Trade-off**: Purely bottom-up pruning (vision-only) misses task-relevant tokens that are visually salient but semantically critical for specific queries.

### Learned Level Selection Networks

**Approach**: Train a small MLP to predict which pyramid levels are necessary given query embeddings.

```python
class PyramidLevelPredictor(nn.Module):
    def __init__(self, query_dim=512, num_levels=5):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_levels),
            nn.Sigmoid()  # Per-level relevance score
        )

    def forward(self, query_embed):
        level_probs = self.predictor(query_embed)  # [B, num_levels]
        # Threshold: keep levels with prob > 0.5
        selected_mask = level_probs > 0.5
        return selected_mask
```

**Training Signal**: Supervise level selection with task performance. Levels that improve downstream accuracy receive higher weights.

**From TokenFLEX Research (arXiv:2504.03154):**

TokenFLEX presents a unified training paradigm that enables VLMs to operate with variable token counts (64, 144, 256 tokens). Key innovations:
- **Stochastic token modulation** during training
- **Adaptive pooling layer** for flexible downsampling
- Achieves **1.6% gain at 64 tokens, 1.0% at 144, 0.4% at 256** over fixed-token baselines

The takeaway: Training with variable token budgets generalizes better than fixed-resolution training.

---

## Section 2: Sparse Pyramid Sampling (Skip Levels) (~80 lines)

### Not All Levels Needed for Every Query

Traditional pyramid processing encodes all levels (0 to L), but this is wasteful. **Sparse sampling** skips intermediate levels when they don't add value.

**Example Scenario:**
- Input: 2048×2048 image → 5 pyramid levels (0, 1, 2, 3, 4)
- Query: "What color is the car?"
- **Full pyramid**: Encode all 5 levels → 10,000 tokens
- **Sparse pyramid**: Skip levels 0, 1, 4 → Encode only levels 2, 3 → 3,000 tokens

**Skip-Level Strategy:**

```python
def sparse_pyramid_sampling(query, pyramid_levels):
    """
    Skip levels that don't contribute to query.

    Heuristic:
    - For global semantic queries: Skip fine levels (0, 1)
    - For local detail queries: Skip coarse levels (3, 4)
    - For mid-level tasks: Use levels 1-3
    """
    query_type = classify_query_type(query)

    if query_type == "global_semantic":
        return [2, 3, 4]  # Skip fine levels
    elif query_type == "local_detail":
        return [0, 1, 2]  # Skip coarse levels
    elif query_type == "mid_level":
        return [1, 2, 3]
    else:
        return list(range(5))  # Use all levels for complex queries
```

### Efficiency Gains (Reduce Tokens by 30-50%)

**From Pyramid Token Pruning Results:**

At 50% pruning ratio (r=0.5), PTP achieves:
- **Memory reduction**: 24.6 GB → 20.9 GB (15% decrease)
- **Latency per token**: 34.2ms → 19.7ms (42% decrease)
- **Total inference time**: 325.7s → 187.4s (42% faster)
- **FLOPs**: 6.40 TFLOPs → 3.04 TFLOPs (52% reduction)
- **KV-cache**: 336 MB → 168 MB (50% reduction)

**Performance retention**: 99.8% of original accuracy across 13 benchmarks (MME, MMB, TextVQA, etc.)

### When to Skip: Low-Relevance Regions, Simple Textures

**Skip Criteria:**

1. **Low visual saliency**: If CLS-to-patch attention is uniformly low across a region, skip that level
2. **Homogeneous textures**: Sky, walls, uniform backgrounds → coarse levels sufficient
3. **Out-of-query regions**: If query mentions "left side," skip tokens from right side at all levels
4. **Redundant levels**: Adjacent levels with high feature correlation → keep one, skip other

**Adaptive Skip Decision:**

```python
def should_skip_level(level_features, query_embed, threshold=0.3):
    """
    Decide whether to skip a pyramid level.

    Returns:
        skip: bool - True if level should be skipped
    """
    # Measure query-feature alignment
    alignment = cross_attention_score(query_embed, level_features)

    # Measure intra-level variance (detect homogeneous regions)
    variance = feature_variance(level_features)

    skip = (alignment < threshold) or (variance < 0.1)
    return skip
```

**Real-World Example** (from PTP case study):
- **Query**: "What is written on the sign?"
- **Image**: Street scene with small text on sign in upper-right corner
- **Sparse sampling decision**:
  - **Keep**: Level 0 (full resolution) for text recognition
  - **Keep**: Level 1 (2× downsampled) for context around sign
  - **Skip**: Levels 2-4 (no additional text info)
- **Result**: 60% token reduction, text correctly recognized

### Training Strategies for Skip Prediction

**Differentiable Skip Gates:**

Use Gumbel-Softmax to make skip decisions differentiable during training:

```python
def gumbel_skip_gate(level_score, temperature=1.0):
    """
    Soft skip gate using Gumbel-Softmax.
    During training: Stochastic sampling
    During inference: Deterministic (threshold)
    """
    if training:
        logits = torch.stack([level_score, 1 - level_score], dim=-1)
        sample = gumbel_softmax(logits, tau=temperature, hard=True)
        skip = sample[..., 1]  # 1 = skip, 0 = keep
    else:
        skip = (level_score < 0.5).float()
    return skip
```

**Reinforcement Learning Approach:**

Treat level selection as a sequential decision problem:
- **State**: Query embedding + current pyramid level
- **Action**: Keep or skip level
- **Reward**: Downstream task accuracy + efficiency bonus (fewer tokens = higher reward)

Train with policy gradient (REINFORCE) or actor-critic methods.

---

## Section 3: ARR-COC Relevance → Pyramid LOD Mapping (~90 lines)

### CRITICAL: Direct Connection to ARR-COC Project

The [ARR-COC-VIS project](https://github.com/djwar42/arr-coc-0-1) (Adaptive Relevance Realization - Contexts Optical Compression - Vision) implements John Vervaeke's **Relevance Realization** framework for vision-language models. This section bridges pyramid LOD allocation with ARR-COC's cognitive principles.

**Core Principle**: Relevance is **transjective** — it emerges from the relationship between agent (query) and arena (visual content). Like a shark's fitness for the ocean, relevance isn't in the query alone or the image alone, but in their coupling.

### Propositional Knowing → Information Content → Pyramid Level

**Propositional knowing** (knowing THAT) corresponds to statistical information content. Measured via Shannon entropy.

**Mapping to Pyramid LOD:**

High-entropy regions (complex textures, edges, text) → **High-resolution levels (0-1)**
- More information to encode → more tokens needed
- Example: Dense text, intricate patterns, object boundaries

Low-entropy regions (smooth gradients, uniform backgrounds) → **Low-resolution levels (3-4)**
- Less information → fewer tokens suffice
- Example: Clear sky, blank walls, water surfaces

**Implementation in ARR-COC ([knowing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/knowing.py)):**

```python
class InformationScorer(nn.Module):
    """
    Propositional knowing: Information density (Shannon entropy).
    Measures statistical complexity of visual patches.
    """
    def forward(self, x):
        # Compute entropy of patch features
        # High entropy = high information = needs high-res level
        return entropy_score(x)
```

**Pyramid Allocation Rule:**
```
if information_score > threshold_high:
    use_pyramid_level = 0  # Full resolution
elif information_score > threshold_mid:
    use_pyramid_level = 1  # 2× downsampled
else:
    use_pyramid_level = 2  # 4× downsampled
```

### Perspectival Knowing → Salience → Foveal Allocation

**Perspectival knowing** (knowing WHAT IT'S LIKE) captures subjective salience — what "stands out" in the visual field.

**Mapping to Pyramid LOD:**

Salient regions (objects, faces, motion) → **High-resolution levels (0-1)**
- Perceptually important → allocate more detail
- Example: Human faces, moving objects, bright contrasts

Non-salient regions (backgrounds, shadows) → **Low-resolution levels (2-4)**
- Perceptually less important → coarse representation sufficient
- Example: Out-of-focus backgrounds, shadows, peripheral areas

**Foveated Vision Analogy:**
Human fovea provides high-resolution detail (2° visual angle), while periphery is coarse (60° visual angle). Pyramid LOD allocation mimics this: **foveal regions = high pyramid levels, peripheral = low pyramid levels**.

**Implementation in ARR-COC ([knowing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/knowing.py)):**

```python
class SalienceScorer(nn.Module):
    """
    Perspectival knowing: Visual salience (Jungian archetypes).
    Measures perceptual 'pull' of visual elements.
    """
    def forward(self, x):
        # Compute salience using learned archetypes
        # High salience = foveal region = high-res level
        return salience_score(x)
```

**Pyramid Allocation with Salience:**
```
salience_map = compute_salience(image)
for region in image_patches:
    if salience_map[region] > high_threshold:
        allocate_pyramid_level(region, level=0)  # Foveal
    elif salience_map[region] > mid_threshold:
        allocate_pyramid_level(region, level=1)  # Para-foveal
    else:
        allocate_pyramid_level(region, level=3)  # Peripheral
```

### Participatory Knowing → Query-Image Coupling → Adaptive LOD

**Participatory knowing** (knowing BY BEING) reflects agent-arena coupling. The query "participates" in determining what visual content is relevant.

**Mapping to Pyramid LOD:**

Query-relevant regions → **High-resolution levels (0-1)**
- Query: "Read the license plate" → Text region gets level 0
- Query: "What breed is the dog?" → Dog gets levels 0-1, background gets level 3

Query-irrelevant regions → **Low-resolution levels (2-4)**
- Query: "Describe the building" → Sky and ground get level 3

**This is Transjective Relevance:**
- Object alone doesn't determine LOD
- Query alone doesn't determine LOD
- **Query-object relationship** determines LOD

**Implementation in ARR-COC ([knowing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/knowing.py)):**

```python
class CouplingScorer(nn.Module):
    """
    Participatory knowing: Query-content coupling.
    Measures how query 'grabs' visual content.
    """
    def forward(self, visual_feats, query_embed):
        # Cross-attention: query attends to visual patches
        coupling = cross_attention(query_embed, visual_feats)
        return coupling  # High coupling = high-res level
```

**Adaptive LOD Strategy:**
```python
def allocate_pyramid_levels_arr_coc(image, query):
    """
    ARR-COC-inspired pyramid LOD allocation.
    """
    # Three ways of knowing
    information = compute_entropy(image)      # Propositional
    salience = compute_salience(image)        # Perspectival
    coupling = compute_coupling(image, query) # Participatory

    # Fuse scores (weighted sum)
    relevance = 0.3 * information + 0.3 * salience + 0.4 * coupling

    # Allocate pyramid levels based on relevance
    for patch in image_patches:
        if relevance[patch] > 0.8:
            pyramid_level[patch] = 0  # Full resolution
        elif relevance[patch] > 0.5:
            pyramid_level[patch] = 1  # 2× downsampled
        elif relevance[patch] > 0.3:
            pyramid_level[patch] = 2  # 4× downsampled
        else:
            pyramid_level[patch] = 3  # 8× downsampled

    return pyramid_level
```

### Opponent Processing: Compress ↔ Particularize at Pyramid Levels

**Opponent processing** navigates cognitive tensions. For pyramid LOD:

**Tension 1: Compress ↔ Particularize**
- **Compress**: Use coarse pyramid levels (fewer tokens, abstract features)
- **Particularize**: Use fine pyramid levels (more tokens, detailed features)
- **Balance**: Allocate tokens where compression and particularization trade-off optimally

**Tension 2: Exploit ↔ Explore**
- **Exploit**: Use known query-relevant levels (e.g., if query mentions text, use level 0)
- **Explore**: Sample additional levels to discover unexpected relevance
- **Balance**: Mostly exploit (focus on query-relevant levels), occasionally explore (sample nearby levels)

**Tension 3: Focus ↔ Diversify**
- **Focus**: Allocate tokens to single high-relevance region
- **Diversify**: Spread tokens across multiple regions
- **Balance**: Focus on primary object, diversify to capture context

**Implementation in ARR-COC ([balancing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/balancing.py)):**

```python
class TensionBalancer(nn.Module):
    """
    Navigate opponent processing tensions.
    Balances compress/particularize, exploit/explore, focus/diversify.
    """
    def forward(self, prop_score, pers_score, part_score):
        # Combine three ways of knowing with learned weights
        balanced = self.weights[0] * prop_score + \
                   self.weights[1] * pers_score + \
                   self.weights[2] * part_score
        return balanced
```

**ARR-COC Token Budget Allocation ([attending.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/attending.py)):**

ARR-COC allocates **64-400 tokens per patch** based on balanced relevance scores:
- High relevance → 400 tokens (full detail)
- Mid relevance → 160-256 tokens (moderate detail)
- Low relevance → 64-100 tokens (coarse representation)

This maps naturally to pyramid levels:
- **400 tokens** = Level 0 (original resolution)
- **256 tokens** = Level 1 (2× downsampled)
- **160 tokens** = Level 2 (4× downsampled)
- **100 tokens** = Level 3 (8× downsampled)
- **64 tokens** = Level 4 (16× downsampled)

**Cross-Reference**: See [ARR-COC-VIS README](https://github.com/djwar42/arr-coc-0-1/blob/main/README.md) for full architecture details.

---

## Section 4: Dynamic Token Budgets Across Scales (~70 lines)

### Allocate 64-400 Tokens Per Pyramid Level (Not Per Patch)

**Traditional VLM approach**: Fixed tokens per image (e.g., 576 tokens total).

**Pyramid LOD approach**: Variable tokens per pyramid level based on query needs.

**Token Budget Distribution:**

```python
def distribute_token_budget(total_budget=1000, num_levels=5, relevance_scores):
    """
    Distribute total token budget across pyramid levels.

    Args:
        total_budget: Total tokens available (e.g., 1000)
        num_levels: Number of pyramid levels (e.g., 5)
        relevance_scores: [num_levels] - relevance of each level

    Returns:
        level_budgets: [num_levels] - tokens allocated to each level
    """
    # Normalize relevance scores to sum to 1
    weights = relevance_scores / relevance_scores.sum()

    # Distribute budget proportionally
    level_budgets = (weights * total_budget).int()

    # Ensure minimum token budget per level (e.g., 64 tokens)
    level_budgets = torch.clamp(level_budgets, min=64, max=400)

    return level_budgets
```

**Example Allocation:**
- **Query**: "Read the text on the sign"
- **Relevance scores**: [0.5, 0.3, 0.1, 0.05, 0.05] (levels 0-4)
- **Budget**: 1000 tokens total
- **Allocation**:
  - Level 0: 500 tokens (high-res text region)
  - Level 1: 300 tokens (context around sign)
  - Level 2: 100 tokens (background)
  - Level 3: 50 tokens (far periphery)
  - Level 4: 50 tokens (minimal global context)

### Coarse Levels: Fewer Tokens (Global Context)

**Coarse pyramid levels** (3-4) capture global scene structure with minimal tokens:
- **Purpose**: Scene layout, object positions, overall composition
- **Token count**: 64-100 tokens per level
- **Use cases**: "What is the overall mood?" "Where is this photo taken?"

**Why fewer tokens suffice:**
- Low spatial frequency information (edges, contours)
- High compression tolerance (JPEG-like artifacts not critical)
- Semantic abstraction (category-level recognition)

**Implementation:**
```python
# Coarse level processing (level 3-4)
coarse_features = pyramid[3]  # 1/8 resolution
coarse_tokens = adaptive_pool(coarse_features, target_tokens=64)
# Result: 64 tokens capturing global layout
```

### Fine Levels: More Tokens (Local Detail)

**Fine pyramid levels** (0-1) capture local textures and fine details:
- **Purpose**: Text recognition, small object detection, texture classification
- **Token count**: 256-400 tokens per level
- **Use cases**: "What does the text say?" "Count the number of screws"

**Why more tokens needed:**
- High spatial frequency information (edges, text, fine textures)
- Low compression tolerance (lossy compression degrades critical details)
- Pixel-level precision required

**Implementation:**
```python
# Fine level processing (level 0-1)
fine_features = pyramid[0]  # Full resolution
fine_tokens = adaptive_pool(fine_features, target_tokens=400)
# Result: 400 tokens capturing fine-grained detail
```

### Total Budget Allocation Strategy

**Hierarchical Budget Allocation:**

1. **Query analysis**: Determine task type (coarse vs. fine-grained)
2. **Level scoring**: Assign relevance score to each pyramid level
3. **Budget distribution**: Allocate tokens proportionally to relevance
4. **Minimum guarantees**: Ensure each level gets at least 64 tokens (maintain diversity)
5. **Maximum caps**: Cap fine levels at 400 tokens (diminishing returns beyond this)

**Example Strategies:**

**Strategy A: Text-Heavy Task**
```
Query: "Transcribe all visible text"
Level 0: 400 tokens (full-res text)
Level 1: 300 tokens (verify text boundaries)
Level 2: 150 tokens (paragraph structure)
Level 3: 100 tokens (page layout)
Level 4: 50 tokens (document type)
Total: 1000 tokens
```

**Strategy B: Scene Understanding**
```
Query: "Describe the scene"
Level 0: 100 tokens (key objects)
Level 1: 150 tokens (object relationships)
Level 2: 250 tokens (mid-level composition)
Level 3: 300 tokens (overall layout)
Level 4: 200 tokens (scene category)
Total: 1000 tokens
```

**Strategy C: Object Counting**
```
Query: "How many cars are in the parking lot?"
Level 0: 350 tokens (individual cars)
Level 1: 300 tokens (car clusters)
Level 2: 200 tokens (parking lot structure)
Level 3: 100 tokens (lot boundaries)
Level 4: 50 tokens (aerial view)
Total: 1000 tokens
```

**From PTP Research:**

At different pruning ratios, PTP maintains performance with varying token counts:
- **10% pruning**: 1613 tokens → 1792.4 MME score
- **30% pruning**: 1076 tokens → 1835.9 MME score
- **50% pruning**: 896 tokens → 1879.7 MME score (best performance!)

The counter-intuitive result: **pruning improves performance** by removing noise tokens that distract the LLM.

**Key Insight**: More tokens ≠ better performance. **Optimal token budget** depends on task complexity and query type.

---

## Sources

### Source Documents
- None (pure web research expansion)

### Web Research (Accessed 2025-01-31)

**Primary Papers:**
- [Pyramid Token Pruning for High-Resolution Large Vision-Language Models](https://arxiv.org/html/2509.15704v2) - arXiv:2509.15704v2
  - Hierarchical pruning combining region-level, token-level, and instruction-guided scoring
  - 50% token reduction with 99.8% performance retention on InternVL2
  - Key method: Coarse-to-fine pruning pipeline inspired by human visual cognition

- [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2410.04417) - arXiv:2410.04417
  - Training-free token optimization using text guidance
  - 54% FLOPs reduction, 37% CUDA latency decrease on LLaVA
  - Adaptive sparsification ratio per layer with token recycling

- [TokenFLEX: Unified VLM Training for Flexible Visual Tokens Inference](https://arxiv.org/html/2504.03154v1) - arXiv:2504.03154
  - Variable token count training paradigm (64, 144, 256 tokens)
  - Adaptive pooling layer with SwiGLU for flexible downsampling
  - Performance gains across different token budgets

**Search Results:**
- Google Scholar search: "attention-driven pyramid pruning vision language model"
- Google Scholar search: "sparse pyramid sampling skip levels VLM 2024"
- arXiv search: "dynamic token budget multi-scale attention VLM"

**ARR-COC Project Links:**
- [ARR-COC-VIS Repository](https://github.com/djwar42/arr-coc-0-1) - Relevance Realization framework implementation
- [ARR-COC README](https://github.com/djwar42/arr-coc-0-1/blob/main/README.md) - Full architecture documentation
- [knowing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/knowing.py) - Three ways of knowing scorers
- [attending.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/attending.py) - Token allocation (64-400 tokens/patch)
- [balancing.py](https://github.com/djwar42/arr-coc-0-1/blob/main/arr_coc/balancing.py) - Opponent processing (tension balancing)

**Additional References:**
- John Vervaeke's Relevance Realization framework - cognitive science foundations
- Foveated rendering techniques - biological vision inspiration
- CLIP, ViT, LLaVA architectures - VLM baseline models
