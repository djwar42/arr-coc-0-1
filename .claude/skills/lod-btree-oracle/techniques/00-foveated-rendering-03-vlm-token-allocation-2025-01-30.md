# VLM Token Allocation: Foveation for Vision-Language Models

**Dynamic knowledge addition**: 2025-01-30
**Source**: ARR-COC-VIS Platonic Dialogues 12-18, The Homunculus Protocol
**Parent**: [00-foveated-rendering.md](00-foveated-rendering.md)

---

## Overview

This document extends foveated rendering principles from graphics (pixels) to **vision-language models** (tokens). VLMs must process images efficiently under fixed computational budgets. Instead of reducing pixel rendering cost, we reduce **token count** for LLM processing.

**Core Challenge**: Convert 4096 image patches → 273 tokens intelligently.

**Key Insight**: Foveation principles (cortical magnification, gaze-contingent allocation, log-polar sampling) apply directly to token allocation in VLMs.

---

## 1. The Fixed Token Budget: Biological Grounding

### 273 Tokens ≈ 273 V1 Hypercolumn Clusters

**Biological Foundation**:
- Human visual cortex V1: ~273 distinct processing clusters (hypercolumns)
- Fixed cortical budget (genetically determined brain size)
- Variable allocation (learned through experience, task-driven)

**VLM Parallel**:
- Fixed token budget: 273 tokens (matches cortical constraint)
- Variable allocation: Grid vs atlas vs hybrid (learned strategy)
- Query-aware: Like task-driven saccades and attention

**Cortical Magnification Analogy**:
```
Fovea (center): 20-25% of V1 cortex → 90 tokens (33% of 273)
Periphery: Remaining 75% of V1 → 183 tokens (67% of 273)
Exponential falloff: M(e) = M₀/(e+e₀)
```

**Why 273 specifically?**
- Computational: Small enough for efficient LLM processing
- Biological: Matches V1 functional organization
- Practical: Balances detail (20×8=160 high-res) vs coverage (grid background)

---

## 2. Token Allocation Strategies

### 2.1 Grid Sampling (Baseline)

**Method**: Uniform grid over image, score patches, select top-K.

```python
# Simple grid top-K
patches = encode_grid(image, size=64x64)  # 4096 patches
scores = score_patches(patches, query)
tokens = topk(patches, scores, k=273)
```

**Characteristics**:
- ✅ **Fast**: 50ms processing time
- ✅ **Simple**: No segmentation needed
- ✅ **Batchable**: Always 273 tokens
- ❌ **Fragments objects**: Arbitrary boundaries cut through semantic regions
- ❌ **Ignores structure**: Doesn't respect perceptual boundaries

**When to use**: Baseline, speed-critical applications, unstructured images (textures, noise)

---

### 2.2 Semantic Atlas (SAM-Based)

**Method**: Segment image into regions (SAM), allocate tokens to whole regions.

```python
# Semantic atlas
regions = sam.generate(image)  # ~50-300 regions
scores = score_regions(regions, query)
top_regions = sorted(regions, key=scores)[:91]  # Select 91 regions
tokens = [encode_region(r, num_tokens=3) for r in top_regions]  # 91×3=273
```

**Canonical Structure**:
- 91 regions × 3 tokens = 273 total
- Each region encoded with 3 tokens (whole object visible)
- Fixed structure → batchable

**Characteristics**:
- ✅ **Semantic coherence**: Whole objects, not fragments
- ✅ **Respects boundaries**: Perceptual edges (SAM detects)
- ✅ **Linguistically aligned**: Regions match concepts ("car", "person")
- ❌ **Slower**: 300ms (SAM segmentation overhead)
- ❌ **Complex**: More moving parts than grid

**When to use**: Documents, structured images, object-centric queries ("What is the person holding?")

---

### 2.3 Tiered Semantic Atlas (Advanced)

**Method**: Variable tokens per region based on importance, fixed tier structure.

```python
class TieredSemanticAtlas:
    def __init__(self):
        self.tiers = [
            {'name': 'high', 'num_regions': 20, 'tokens_per': 8},  # 160 tokens (59%)
            {'name': 'mid', 'num_regions': 30, 'tokens_per': 3},   # 90 tokens (33%)
            {'name': 'low', 'num_regions': 23, 'tokens_per': 1},   # 23 tokens (8%)
        ]
        # Total: 20*8 + 30*3 + 23*1 = 273 ✓

    def forward(self, image, query):
        all_regions = sam.generate(image)
        scores = score_regions(all_regions, query)
        sorted_regions = sorted(zip(all_regions, scores), key=lambda x: x[1], reverse=True)

        tokens = []
        idx = 0
        for tier in self.tiers:
            tier_regions = sorted_regions[idx:idx + tier['num_regions']]
            idx += tier['num_regions']

            for region, score in tier_regions:
                region_tokens = encode_region(region, num_tokens=tier['tokens_per'])
                tokens.extend(region_tokens)

        return torch.stack(tokens)  # [273, D] - always!
```

**Pareto Allocation**:
- Top 20 regions (22%) get 160 tokens (59%) - high importance
- Next 30 regions (33%) get 90 tokens (33%) - medium
- Last 23 regions (25%) get 23 tokens (8%) - low (context only)

**Characteristics**:
- ✅ **Expressive**: More detail where it matters
- ✅ **Fixed structure**: Always 273 tokens (batchable!)
- ✅ **Interpretable**: Clear tiers, easy to debug
- ⚠️ **More complex**: Additional tier management

**When to use**: Complex documents (formulas need 8 tokens, background needs 1), fine-grained detail requirements

---

### 2.4 Hybrid Grid-Atlas (Best of Both Worlds)

**Method**: Atlas for foreground (salient objects), grid for background (texture/context).

```python
def hybrid_grid_atlas(image, query):
    # Stage 1: Identify foreground (query-independent)
    foreground_score = (
        0.4 * visual_saliency(image) +      # Itti-Koch model
        0.4 * objectness(image) +            # Object detector
        0.2 * high_frequency(image)          # Edge/texture density
    )

    # Stage 2: SAM on foreground
    sam_regions = sam.generate(image)
    foreground_regions = [r for r in sam_regions
                         if foreground_score[r.bbox] > threshold]

    # Stage 3: Score foreground regions (NOW query-aware)
    region_importance = [score_region(r, query) for r in foreground_regions]

    # Stage 4: Adaptive budget based on foreground coverage
    foreground_coverage = foreground_mask.sum() / image.size

    if foreground_coverage < 0.1:
        atlas_budget = int(0.2 * 273)  # 55 tokens (sparse foreground)
        grid_budget = int(0.8 * 273)   # 218 tokens
    elif foreground_coverage > 0.8:
        atlas_budget = int(0.8 * 273)  # 218 tokens (dense foreground)
        grid_budget = int(0.2 * 273)   # 55 tokens
    else:
        atlas_budget = int(0.6 * 273)  # 164 tokens (balanced)
        grid_budget = int(0.4 * 273)   # 109 tokens

    # Stage 5: Allocate foreground tokens
    foreground_tokens = tiered_allocation(
        foreground_regions,
        region_importance,
        budget=atlas_budget
    )

    # Stage 6: Grid sample background
    background_mask = create_mask(exclude=foreground_regions)
    background_patches = grid_sample(image, mask=background_mask)
    background_importance = [score_patch(p, query) for p in background_patches]
    background_tokens = top_k_allocation(
        background_patches,
        background_importance,
        budget=grid_budget
    )

    return foreground_tokens + background_tokens  # 273 total
```

**Foreground Detection** (query-independent, avoids circularity):
- **Visual saliency**: Bottom-up attention (Itti-Koch model)
- **Objectness score**: Discrete object detector (independent of query)
- **Spatial frequency**: High-frequency regions = edges/details (foreground)

**Adaptive Budget**:
- Adjusts atlas/grid split based on image content
- Sparse foreground → mostly grid
- Dense foreground → mostly atlas
- Balanced → 60/40 split

**Characteristics**:
- ✅ **Robust**: Adapts to image structure
- ✅ **Best of both**: Semantic coherence + full coverage
- ✅ **Graceful degradation**: Falls back to grid if no foreground
- ⚠️ **Most complex**: Two allocation systems

**When to use**: Mixed content (photos with text overlay, documents with diagrams), unknown image types

---

### 2.5 Vortex and Spiral Sampling (Exploratory)

**Method**: Sample along logarithmic spirals centered on importance peaks.

```python
def vortex_sampling(image, query, num_vortices=5):
    # Identify importance centers
    importance_map = compute_importance(image, query)
    vortex_centers = find_peaks(importance_map, n=num_vortices)

    tokens = []
    for center in vortex_centers:
        # Logarithmic spiral (golden ratio)
        spiral_points = generate_log_spiral(
            center,
            a=0.1,  # Initial radius
            b=0.3,  # Growth rate (golden angle ≈ 137.5°)
            num_points=273 // num_vortices
        )

        for point in spiral_points:
            tokens.append(encode_patch(image, point))

    return tokens[:273]  # Truncate to budget
```

**Characteristics**:
- ✅ **Exploratory**: Covers space around importance centers
- ✅ **Biologically inspired**: Saccadic search patterns
- ❌ **Complex**: Spiral generation, vortex placement
- ❌ **Batching hard**: Variable point counts per vortex

**When to use**: Research, active vision experiments, uncertainty-driven exploration

---

## 3. Relevance Scoring Framework

### 3.1 Four Ways of Knowing (Vervaeke Framework)

**Applied to visual token allocation:**

```python
# Propositional (WHAT exists in the patch)
propositional_score = shannon_entropy(patch) + edge_density(patch)

# Perspectival (WHERE patch stands out)
perspectival_score = saliency_relative_to_neighbors(patch, context)

# Participatory (HOW patch couples to query)
participatory_score = cross_attention(patch_features, query_embedding)

# Procedural (LEARNED importance)
procedural_score = learned_importance_network(patch)

# Combined relevance (adaptive weights!)
total_relevance = policy_network([prop, pers, part, proc], state)
```

**Key Insight**: Weights are NOT fixed (e.g., 0.2, 0.2, 0.4, 0.2). Instead, learn a **policy** that adjusts weights based on:
- Query specificity (specific → high participatory, vague → high perspectival)
- Exploration phase (early → explore, late → exploit)
- Confidence (low → diversify, high → focus)

### 3.2 Opponent Processing

**Three tensions to navigate:**

1. **Compress ↔ Particularize**
   - Compress: Coarse background (1 token per region)
   - Particularize: Fine foreground (8 tokens per region)

2. **Exploit ↔ Explore**
   - Exploit: Known-important regions (high query-relevance)
   - Explore: Uncertain regions (might be relevant)

3. **Focus ↔ Diversify**
   - Focus: Deep detail on few regions
   - Diversify: Broad context across many regions

**Not fixed ratios, but dynamic policies that navigate based on context.**

---

## 4. Multi-Fixation Processing

### 4.1 Sequential Allocation Strategy

**Analogy**: Human saccades (3-4 per second, 200-300ms each) → VLM fixations (3-5 per query, 190ms each)

```python
def multi_fixation_with_hybrid_confidence(image, query, max_fix=5):
    fixations = []

    for i in range(max_fix):
        # Allocate tokens (context-aware)
        tokens = allocate(image, query, context=fixations)
        output = llm.generate(tokens, query, return_confidence=True)
        fixations.append(output)

        # Hybrid confidence
        self_confidence = output.confidence  # Model's self-report

        if i > 0:
            # Measure consistency with previous fixation
            consistency = measure_agreement(fixations[-1], fixations[-2])
            confidence = 0.6 * self_confidence + 0.4 * consistency
        else:
            confidence = self_confidence

        # Stopping criteria
        if confidence > 0.92:
            print(f"High confidence after {i+1} fixations")
            break

        if i > 0 and confidence < fixations[-2].combined_confidence:
            print(f"Confidence decreasing, stopping at {i+1}")
            break

    # Ensemble: weight fixations by confidence
    final_answer = weighted_ensemble(
        fixations,
        weights=[f.combined_confidence for f in fixations]
    )
    return final_answer
```

**Fixation Evolution**:
1. **Fixation 1**: Exploratory (broad coverage, low query-bias)
2. **Fixation 2**: Exploitative (concentrate on candidates, high query-bias)
3. **Fixation 3+**: Adaptive (confidence-driven, stop when confident)

### 4.2 Confidence Estimation

**Hybrid metric** (combines two approaches):

**Option 1: Self-reported probability**
- Easy to implement (LLMs support this)
- Single fixation sufficient
- ❌ Poorly calibrated (overconfident or underconfident)
- ❌ Doesn't detect hallucinations

**Option 2: Consistency across fixations**
- Detects uncertainty (inconsistent answers → low confidence)
- Robust to overconfidence
- ❌ Requires 2+ fixations
- ❌ Slow convergence

**Hybrid (best of both)**:
```python
if first_fixation:
    confidence = self_reported_probability
else:
    confidence = 0.6 * self_reported + 0.4 * consistency_with_previous
```

**Stopping Criteria**:
1. High confidence (> 0.92)
2. Decreasing confidence (confidence_i < confidence_{i-1})
3. Max fixations reached (budget limit, e.g., 5)

### 4.3 Ensemble Integration

**Key Insight**: Later fixations aren't always better! Weight by confidence.

```python
final_answer = weighted_ensemble(
    [fix.answer for fix in fixations],
    weights=[fix.combined_confidence for fix in fixations]
)
```

**Why?** If fixation 2 has confidence 0.95 but fixation 3 has 0.80, trust fixation 2 more!

---

## 5. Edge Cases and Robustness

### 5.1 Uniform Images (No Structure)

**Scenario**: Blank wall, uniform texture, solid color

**Behavior**:
- foreground_score uniformly low
- SAM finds no regions (or very few)
- **Graceful fallback to grid**: Sample 273 patches uniformly
- Result: Degraded to baseline (which is fine—no structure to respect!)

### 5.2 Highly Complex Images (Everything is Foreground)

**Scenario**: Dense text document, complex fractal, detailed artwork

**Behavior**:
- foreground_score uniformly high
- SAM returns 200+ regions, all "foreground"
- **Adaptive budget**: 80% atlas (218 tokens), 20% grid (55 tokens)
- Take top 72 regions (72×3=216 ≈ 218 tokens)
- Grid provides minimal context for remaining space

### 5.3 Confidence Degeneracy

**Problem 1**: Model overconfident but wrong
- **Solution**: Consistency check catches it (inconsistent across fixations → low confidence)

**Problem 2**: Model underconfident but correct
- **Solution**: Self-reported maintains it (high self-confidence contributes 60%)

**Hybrid prevents both failure modes**

---

## 6. Computational Trade-offs

### Speed vs Quality

| Strategy | Speed | Semantic Coherence | Batching | Complexity |
|----------|-------|-------------------|----------|------------|
| Grid top-K | 50ms | ❌ Fragments | ✅ Always 273 | Low |
| Semantic atlas (91×3) | 300ms | ✅ Whole objects | ✅ Always 273 | Medium |
| Tiered atlas (20/30/23) | 320ms | ✅✅ Variable detail | ✅ Always 273 | Medium-High |
| Hybrid grid-atlas | 350ms | ✅ Adaptive | ✅ Always 273 | High |
| Vortex sampling | 80ms | ⚠️ Exploratory | ⚠️ Variable | High |
| Multi-fixation (3×) | 3× base | ✅✅ Iterative refinement | ✅ Per fixation | High |

**Recommended progression**:
1. Start with **grid top-K** (baseline)
2. Test **semantic atlas** (91×3) to measure coherence benefit
3. Implement **tiered atlas** (20/30/23) for detail where needed
4. Deploy **hybrid** for production (robustness)
5. Experiment with **multi-fixation** for hard queries

---

## 7. Training Paradigms

### 7.1 Supervised Learning

**Data**: (image, query, answer) triplets

```python
# Learn to predict which patches are relevant
loss = cross_entropy(predicted_selection, ground_truth_selection)
```

**Pros**: Straightforward, interpretable
**Cons**: Requires labeled data (which patches were important)

### 7.2 Reinforcement Learning

**Reward**: Task accuracy (e.g., VQA correct answer)

```python
# Policy gradient: allocation strategy as policy
reward = 1.0 if answer_correct else 0.0
policy_loss = -reward * log_prob(action)
```

**Pros**: End-to-end optimization, no labels needed
**Cons**: Sparse rewards, slow convergence, harder to debug

### 7.3 Interpretable Rules + Learned Weights

**Hybrid approach** (recommended):
- Hand-crafted scoring functions (entropy, saliency, cross-attention)
- **Learn weights** for combining them (policy network)

```python
# Fixed scoring functions
scores = {
    'propositional': shannon_entropy(patch),
    'perspectival': saliency(patch, context),
    'participatory': cross_attention(patch, query),
    'procedural': learned_net(patch)
}

# Learned policy for weighting
weights = policy_network(query_features, image_features)
final_score = sum(weights[k] * scores[k] for k in scores)
```

**Pros**: Interpretable components, adaptive combination
**Cons**: Still requires training policy network

---

## 8. Open Research Questions

From ARR-COC-VIS Dialogue 17 (Socratic examination):

1. **Semantic coherence hypothesis**: Does seeing whole objects (atlas) improve understanding vs fragments (grid)?
   - **Test**: Measure accuracy vs object fragmentation degree
   - **Prediction**: Atlas wins when objects span 10+ patches

2. **Boundary quality hypothesis**: Do semantic boundaries reduce noise vs arbitrary grid?
   - **Test**: Correlate atlas advantage with grid edge-cutting score
   - **Prediction**: Atlas helps when grid cuts high-contrast edges

3. **Linguistic alignment hypothesis**: Do regions align with linguistic concepts?
   - **Test**: Stratify by query type (object vs relational)
   - **Prediction**: Atlas wins on object queries ("What is X?")

4. **Multi-fixation efficiency**: Fixed (3) vs adaptive (confidence-based) stopping?
   - **Test**: Compare accuracy vs computational cost
   - **Prediction**: Adaptive stops early (2) on simple queries, late (5) on complex

5. **Conceptual vs perceptual segmentation**: Does topic-based text segmentation help?
   - **Test**: Atlas with OCR + topic segmentation vs SAM alone
   - **Prediction**: Conceptual wins on multi-topic documents

6. **Tier boundary optimization**: Are (20, 30, 23) optimal or domain-specific?
   - **Test**: Grid search over (N_h, N_m, N_l) for documents vs photos
   - **Prediction**: Documents need (15, 40, 18), photos need (30, 20, 23)

7. **End-to-end RL vs interpretable rules**: Can RL discover better allocation?
   - **Test**: Train policy network end-to-end vs hand-crafted + learned weights
   - **Prediction**: RL wins on task accuracy, interpretable wins on debugging

8. **Universal principle vs domain-specific**: One allocation strategy or many?
   - **Test**: Compare domain-agnostic model vs specialist models
   - **Prediction**: Hybrid meta-policy (learns WHEN to use which strategy)

9. **Full differentiability**: Can we make atlas allocation fully differentiable?
   - **Challenge**: SAM segmentation is discrete
   - **Solution**: Soft region assignments (weighted blending)

10. **Vervaekean validation**: Do we understand relevance or merely mimic it?
    - **Philosophical**: When is allocation "meaningful" vs "effective"?
    - **Test**: Measure alignment with human judgments of importance

---

## 9. The Homunculus Protocol

**Named in Dialogue 18**: A unified framework for VLM token allocation.

### Core Principles

1. **Fixed computational budget**: 273 tokens (like fixed cortical budget)
2. **Semantic-aware allocation**: Atlas > grid when structure matters
3. **Query-driven relevance**: Participatory knowing (transjective)
4. **Multi-fixation processing**: Active vision, not passive reception
5. **Opponent process balancing**: Navigate compress↔particularize, exploit↔explore, focus↔diversify

### Implementations

**Simple (baseline)**:
```python
patches = encode_vit(image)
scores = cosine_similarity(patches, query_embedding)
tokens = topk(patches, scores, k=273)
```

**Intermediate (tiered atlas)**:
```python
regions = sam.generate(image)
scores = score_regions(regions, query)
# 20 high (8 tok) + 30 mid (3 tok) + 23 low (1 tok) = 273
tokens = tiered_allocation(regions, scores, tiers=[20,30,23], tokens_per=[8,3,1])
```

**Advanced (hybrid grid-atlas)**:
```python
foreground_score = 0.4*saliency + 0.4*objectness + 0.2*high_freq
foreground_regions = sam_regions[foreground_score > threshold]
atlas_tokens = allocate_to_regions(foreground_regions, budget=164)
grid_tokens = grid_sample(image, exclude=foreground, budget=109)
return atlas_tokens + grid_tokens
```

**Expert (multi-fixation)**:
```python
for i in range(max_fixations):
    tokens = allocate(image, query, context=prev_fixations)
    output = llm(tokens, query)
    confidence = 0.6*output.prob + 0.4*consistency(output, prev)
    if confidence > 0.92: break
```

### Status

- **Designed and specified**: Dialogues 12-18 (ARR-COC-VIS)
- **Next**: Implementation and empirical validation
- **Benchmarks**: DocVQA, TextVQA, ChartQA, general VQA

---

## 10. Connections to Graphics Foveation

### Pixel Rendering → Token Allocation

| Graphics Foveation | VLM Token Allocation |
|-------------------|---------------------|
| Reduce pixel shading cost | Reduce token count for LLM |
| Gaze point (eye tracking) | Query embedding (relevance) |
| Eccentricity (distance from gaze) | Query-relevance score |
| Cortical magnification M(e) | Token budget allocation |
| Fovea (high res) | High-detail regions (8 tokens) |
| Periphery (low res) | Low-detail regions (1 token) |
| Saccades (eye movements) | Multi-fixation (sequential allocation) |
| 60-80% rendering cost reduction | 15× token reduction (4096→273) |

### Log-Polar Transform

**Graphics**: Map retinal coordinates to cortical coordinates
```
(x, y) → (ρ, θ)  where ρ = log(r), θ = arctan(y/x)
```

**VLMs**: Map image patches to token allocation density
```
distance_from_query_focus → token_budget_per_patch
```

**Both use logarithmic scaling**: Dense near center, sparse far away.

---

## 11. References and Source Material

**Primary Sources**:
- ARR-COC-VIS Platonic Dialogues 12-18 (`RESEARCH/PlatonicDialogues/`)
- Dialogue 18: Knowledge Expansion (The Homunculus Protocol)

**Cross-References within LOD Oracle**:
- [00-foveated-rendering.md](00-foveated-rendering.md) - Graphics foveation overview
- [00-foveated-rendering-01-logpolar-mapping-2025-01-30.md](00-foveated-rendering-01-logpolar-mapping-2025-01-30.md) - Log-polar transforms
- [00-foveated-rendering-02-biological-foundations-2025-01-30.md](00-foveated-rendering-02-biological-foundations-2025-01-30.md) - Cortical magnification
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Gaze-content coupling
- [integration/02-multidimensional-queries.md](../integration/02-multidimensional-queries.md) - Query-aware optimization
- [integration/07-metadata-texture-arrays-2025-01-30.md](../integration/07-metadata-texture-arrays-2025-01-30.md) - GPU texture array metadata storage (40-channel architecture, 33× image speedup, 280× video speedup)
- [optimization/01-spatial-locality-cache-2025-01-30.md](../optimization/01-spatial-locality-cache-2025-01-30.md) - Spatial locality optimization (5× fewer cache misses via co-located metadata)

**Related Oracles**:
- `john-vervaeke-oracle` - Relevance realization framework (4Ps, opponent processing)
- `ovis-2-5-oracle` - VLM architectures (VET, native resolution)
- `qwen3vl-oracle` - M-RoPE, dynamic resolution handling
- `computer-vision-foundation-oracle` - SAM, vision transformers

**VLM Research Papers** (related work):
- DeepSeek-OCR (SAM+CLIP serial architecture, 16× compression)
- Ovis 2.5 (Visual Embedding Table, native resolution)
- Qwen3-VL (M-RoPE, DeepStack injection)
- LLaVA-UHD (multi-crop, adaptive patching)

---

## 12. Summary

**VLM token allocation is foveation for language models:**

- Fixed budget (273 tokens) ≈ fixed cortex (V1 hypercolumns)
- Variable allocation (grid/atlas/hybrid) ≈ cortical magnification
- Query-aware relevance ≈ task-driven saccades
- Multi-fixation ≈ active vision

**The Homunculus Protocol** provides a unified framework with implementations from simple (grid) to expert (multi-fixation with hybrid confidence).

**Next steps**: Implement, test on benchmarks, validate open hypotheses.

---

**Last Updated**: 2025-01-30
**Status**: Comprehensive VLM token allocation knowledge from Dialogues 12-18
