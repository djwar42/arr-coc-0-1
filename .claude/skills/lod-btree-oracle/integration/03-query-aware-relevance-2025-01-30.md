# Query-Aware Relevance Realization for LOD Allocation

**Dynamic knowledge addition**: 2025-01-30
**Source**: Vervaeke RR framework, 2024-2025 VLM research, ARR-COC-VIS Dialogues
**Parent**: [02-multidimensional-queries.md](02-multidimensional-queries.md), [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md)

---

## Overview

Query-aware relevance realization applies John Vervaeke's cognitive framework to vision systems, enabling **dynamic LOD allocation based on the relationship between query and visual content**. Unlike static or content-only foveation, query-aware systems realize relevance through **transjective coupling** - relevance emerges from the agent-arena interaction.

**Key Principle**: Relevance is neither objective (in the image alone) nor subjective (in the query alone), but **transjective** - it arises from the coupling between query and content.

---

## Four Ways of Knowing for Visual Relevance

### 1. Propositional Knowing (WHAT exists)

**Measures**: Information content independent of query

```python
def propositional_score(patch):
    """What information does this patch contain?"""
    entropy = shannon_entropy(patch.pixels)  # Complexity
    edges = edge_density(patch)               # Structure
    texture = texture_variance(patch)         # Detail

    return 0.4 * entropy + 0.4 * edges + 0.2 * texture
```

**Applications**:
- Text regions: High entropy (many unique characters)
- Diagrams: High edge density (lines, shapes)
- Uniform backgrounds: Low on all metrics

**When Dominant**: Early exploration when query is vague

### 2. Perspectival Knowing (WHERE stands out)

**Measures**: Salience relative to surrounding context

```python
def perspectival_score(patch, context):
    """How much does this patch stand out?"""
    # Contrast with neighbors
    neighbor_diff = mean_difference(patch, get_neighbors(patch))

    # Visual pop-out (Itti-Koch saliency)
    saliency = compute_saliency_map(context)[patch.location]

    # Center bias (human gaze patterns)
    distance_from_center = euclidean_distance(patch.center, image.center)
    center_weight = exp(-distance_from_center / image.width)

    return 0.5 * neighbor_diff + 0.3 * saliency + 0.2 * center_weight
```

**Applications**:
- Foreground objects: High contrast with background
- Anomalies: Unusual patterns that pop out
- Central regions: Natural gaze bias

**When Dominant**: Query asks "What stands out?" or "Find unusual..."

### 3. Participatory Knowing (HOW couples to query)

**Measures**: Query-content alignment (transjective relevance)

```python
def participatory_score(patch, query):
    """How relevant is this patch to the specific query?"""
    # CLIP-style cross-modal embedding
    patch_embedding = visual_encoder(patch)
    query_embedding = text_encoder(query)

    # Cosine similarity
    similarity = cosine_similarity(patch_embedding, query_embedding)

    # Cross-attention (deeper coupling)
    attention_weights = cross_attention(
        query=query_embedding,
        key=patch_embedding,
        value=patch_embedding
    )

    return 0.6 * similarity + 0.4 * attention_weights
```

**Applications**:
- "Find the cat" → high score for cat regions
- "Read the sign" → high score for text
- "What color is X?" → high score for object X

**When Dominant**: Specific queries with clear intent

### 4. Procedural Knowing (LEARNED efficiency)

**Measures**: Learned importance patterns

```python
class ProceduralScorer(nn.Module):
    """Learns what regions tend to be important for tasks"""

    def __init__(self):
        self.importance_network = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, patch_features):
        """Learned importance from experience"""
        return self.importance_network(patch_features)
```

**Training**:
- Supervised: Label important regions in training data
- Reinforcement: Reward correct answers using selected regions
- Meta-learning: Learn to learn importance across domains

**Applications**:
- Domain-specific patterns (medical: lesions, documents: headers)
- Efficiency shortcuts (skip processing blank regions)
- Adaptive to task distribution

**When Dominant**: Well-defined task with training data

---

## Opponent Processing: Navigating Tensions

### Tension 1: Compress ↔ Particularize

**The Trade-off**: Broad coverage vs fine detail

**Compress (Cognitive Scope Narrowing)**:
- Coarse resolution across whole image
- Many regions, few tokens each
- Context preservation

**Particularize (Cognitive Scope Focusing)**:
- Fine resolution on key regions
- Few regions, many tokens each
- Detail extraction

**Dynamic Navigation**:
```python
def navigate_compress_particularize(query_specificity):
    """Adjust compression based on query type"""
    if query_specificity < 0.3:  # Vague query
        # Broad coverage
        return {'strategy': 'compress', 'num_regions': 200, 'tokens_per': 1.5}
    elif query_specificity > 0.7:  # Specific query
        # Deep focus
        return {'strategy': 'particularize', 'num_regions': 30, 'tokens_per': 9}
    else:
        # Balanced
        return {'strategy': 'balanced', 'num_regions': 91, 'tokens_per': 3}
```

**Examples**:
- "Describe this image" → Compress (broad coverage)
- "What's written on the small sign?" → Particularize (zoom in)

### Tension 2: Exploit ↔ Explore

**The Trade-off**: Use known information vs discover new

**Exploit (Use Current Knowledge)**:
- Allocate to regions already identified as relevant
- Refine understanding of known objects
- High confidence, focused allocation

**Explore (Search for New Information)**:
- Allocate to uncertain or unexamined regions
- Discover unexpected elements
- Diversity in sampling

**Dynamic Navigation**:
```python
def navigate_exploit_explore(confidence, iteration):
    """Balance exploitation and exploration"""
    # Early iterations: more exploration
    # High confidence: more exploitation

    explore_weight = (1 - confidence) * exp(-iteration / 3)
    exploit_weight = 1 - explore_weight

    return {
        'exploit': exploit_weight,  # Known-important regions
        'explore': explore_weight   # Uncertain regions
    }
```

**Examples**:
- First fixation: 70% explore, 30% exploit
- High confidence answer: 90% exploit, 10% explore
- Low confidence answer: 50-50 split → continue exploring

### Tension 3: Focus ↔ Diversify

**The Trade-off**: Single region vs multiple regions

**Focus (Concentrated Attention)**:
- Allocate most tokens to single best region
- Deep understanding of one thing
- Risk: Miss context, tunnel vision

**Diversify (Distributed Attention)**:
- Spread tokens across many regions
- Broad understanding
- Risk: Insufficient detail anywhere

**Dynamic Navigation**:
```python
def navigate_focus_diversify(query_type, image_complexity):
    """Adjust focus based on task requirements"""
    if query_type == 'single_object':
        # "What's in the red box?"
        return {'focus': 0.8, 'diversify': 0.2}
    elif query_type == 'relationship':
        # "How many cats are near the tree?"
        return {'focus': 0.4, 'diversify': 0.6}
    elif query_type == 'scene_understanding':
        # "Describe this scene"
        return {'focus': 0.3, 'diversify': 0.7}
```

---

## Adaptive Weight Policies

**Core Insight**: Weights between the four ways of knowing should NOT be fixed, but dynamically adjusted based on context.

### Policy Network Approach

```python
class AdaptiveWeightPolicy(nn.Module):
    """Learn when to emphasize which way of knowing"""

    def __init__(self):
        self.policy_network = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )

    def forward(self, query, image_stats, iteration):
        """
        Input context:
        - query: embedding of user query
        - image_stats: complexity, object count, text density
        - iteration: which fixation (0, 1, 2, ...)

        Output: [w_prop, w_pers, w_part, w_proc]
        """
        context = torch.cat([
            query_embedding(query),
            image_stats,
            torch.tensor([iteration])
        ])

        weights = self.policy_network(context)
        return weights

def compute_relevance(patch, query, state):
    """Combine all four ways with adaptive weights"""
    scores = {
        'propositional': propositional_score(patch),
        'perspectival': perspectival_score(patch, state.context),
        'participatory': participatory_score(patch, query),
        'procedural': procedural_score(patch)
    }

    # Get adaptive weights
    weights = policy_network(query, state.image_stats, state.iteration)

    # Weighted combination
    total_relevance = (
        weights[0] * scores['propositional'] +
        weights[1] * scores['perspectival'] +
        weights[2] * scores['participatory'] +
        weights[3] * scores['procedural']
    )

    return total_relevance, weights
```

### Training the Policy

**Supervised Approach**:
```python
# Training data: (image, query, ground_truth_answer, important_regions)
for sample in training_data:
    # Forward pass with learned policy
    weights = policy_network(sample.query, sample.image_stats, iteration=0)
    relevance_scores = compute_all_relevances(sample.patches, weights)
    selected_regions = top_k(relevance_scores, k=91)

    # Answer using selected regions
    answer = vlm(selected_regions, sample.query)

    # Loss: Did we select the right regions?
    loss = cross_entropy(answer, sample.ground_truth)

    # Backprop through policy
    loss.backward()
```

**Reinforcement Learning Approach**:
```python
# Treat region selection as RL action
for episode in episodes:
    # Policy outputs weights
    weights = policy_network(state)

    # Select regions using weighted scores
    relevance = compute_relevance(patches, query, weights)
    selected = top_k(relevance, k=91)

    # Get answer
    answer = vlm(selected, query)

    # Reward: Accuracy + efficiency
    reward = accuracy(answer, ground_truth) - 0.1 * num_tokens_used

    # REINFORCE update
    policy_gradient_update(policy_network, reward)
```

---

## 2024-2025 Research Integration

### CARES: Context-Aware Resolution Selector

**Paper**: arXiv 2510.19496 (Oct 2025)

**Key Contribution**: Adaptive pixel allocation per query

**Integration**:
```python
def cares_resolution_selection(image, query):
    """Select resolution adaptively based on query complexity"""
    # Estimate query complexity
    query_complexity = estimate_complexity(query)

    if query_complexity < 0.3:
        # Simple query: "What color is the sky?"
        resolution = (224, 224)
        num_tokens = 49  # 7×7 grid
    elif query_complexity > 0.7:
        # Complex query: "Read all text in this document"
        resolution = (1792, 1792)
        num_tokens = 196  # 14×14 grid
    else:
        # Medium complexity
        resolution = (672, 672)
        num_tokens = 144  # 12×12 grid

    # Encode at selected resolution
    visual_tokens = encode_image(image, resolution)

    # Apply relevance realization to further filter
    relevance_scores = compute_relevance(visual_tokens, query)
    selected_tokens = top_k(relevance_scores, k=min(num_tokens, 273))

    return selected_tokens
```

**Benefit**: Don't waste pixels on simple queries, allocate more for complex ones

### Question-Aware Vision Transformer (QA-ViT)

**Paper**: CVPR 2024 (Ganz et al., 38 citations)

**Key Contribution**: Embed question awareness directly in ViT layers

**Integration**:
```python
class QuestionAwareViT(nn.Module):
    """ViT with query conditioning at each layer"""

    def forward(self, image_patches, query):
        # Encode query
        query_emb = self.query_encoder(query)

        # ViT layers with query modulation
        x = self.patch_embedding(image_patches)

        for layer in self.transformer_layers:
            # Standard self-attention
            x_attn = layer.self_attention(x)

            # Cross-attention with query
            x_cross = layer.cross_attention(
                query=x_attn,
                key=query_emb,
                value=query_emb
            )

            # Query-modulated feedforward
            x = layer.ffn(x_cross)

        return x  # Query-aware visual features
```

**Benefit**: Every layer knows the query, not just final fusion

### Context-Aware Token Selection (SPA)

**Paper**: arXiv 2410.23608 (Oct 2024)

**Key Contribution**: Supervised gating block for informative tokens

**Integration**:
```python
class SupervisedGatingBlock(nn.Module):
    """Learn which tokens are informative for the query"""

    def __init__(self, dim):
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Concat patch + query
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, patch_features, query_embedding):
        # Concatenate patch and query
        combined = torch.cat([
            patch_features,
            query_embedding.expand(patch_features.shape[0], -1)
        ], dim=-1)

        # Gate score: 0 (irrelevant) to 1 (informative)
        gate_scores = self.gate(combined)

        # Multiply patch features by gate
        gated_features = patch_features * gate_scores

        return gated_features, gate_scores
```

**Training**:
- Supervise gate_scores with human annotations of important regions
- Or train end-to-end with task loss (backprop through gate)

**Benefit**: Explicit learned importance, interpretable

---

## Practical Implementation

### Complete Query-Aware LOD Pipeline

```python
class QueryAwareRelevanceAllocator:
    """Full pipeline with all four ways of knowing"""

    def __init__(self):
        self.propositional_scorer = ProposItional_scorer()
        self.perspectival_scorer = PerspectivalScorer()
        self.participatory_scorer = ParticipatoryScorer()
        self.procedural_scorer = ProceduralScorer()
        self.policy_network = AdaptiveWeightPolicy()

    def allocate(self, image, query, budget=273):
        # Extract visual features
        patches = extract_patches(image)
        patch_features = encode_patches(patches)

        # Compute image statistics
        image_stats = {
            'num_objects': count_objects(image),
            'text_density': measure_text_density(image),
            'complexity': shannon_entropy(image)
        }

        # Get adaptive weights from policy
        weights = self.policy_network(query, image_stats, iteration=0)

        # Score each patch using all four ways
        relevance_scores = []
        for patch in patches:
            prop = self.propositional_scorer(patch)
            pers = self.perspectival_scorer(patch, patches)
            part = self.participatory_scorer(patch, query)
            proc = self.procedural_scorer(patch_features)

            # Weighted combination
            total_relevance = (
                weights[0] * prop +
                weights[1] * pers +
                weights[2] * part +
                weights[3] * proc
            )
            relevance_scores.append(total_relevance)

        # Select top-k patches
        selected_indices = top_k_indices(relevance_scores, k=budget)
        selected_patches = [patches[i] for i in selected_indices]

        # Allocate LOD based on relevance scores
        lod_per_patch = self.allocate_lod(
            relevance_scores[selected_indices],
            total_budget=budget * tokens_per_patch
        )

        return selected_patches, lod_per_patch
```

---

## Connection to LOD Systems

### Gaze-Contingent Displays

**Traditional** ([integration/01-gaze-tracking.md](01-gaze-tracking.md)):
- Track eye gaze
- Allocate detail to foveal region
- Degrade periphery

**Query-Aware Extension**:
- Track eye gaze AND query intent
- Allocate detail to gaze × query relevance
- Periphery degradation modulated by query need

**Example**:
- Query: "Count the people"
- Gaze at person #1
- But also maintain medium detail on other people (to count them)
- Traditional: Only high detail at gaze point
- Query-aware: High at gaze, medium at other people, low elsewhere

### Adaptive LOD for Terrain

**Traditional** ([algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md)):
- LOD based on distance from camera
- Screen-space error metric

**Query-Aware Extension**:
- LOD based on distance × query relevance
- "Find the red building" → allocate more to red structures regardless of distance

---

## Open Research Questions

1. **Optimal weight learning**: Supervised vs RL vs meta-learning?
2. **Generalization**: Do learned policies transfer across domains?
3. **Interpretability**: Can we explain WHY a region got high relevance?
4. **Temporal consistency**: How to maintain relevance across multi-fixation?
5. **Computational cost**: Is adaptive weighting worth the overhead?

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Foundational concept
- [integration/01-gaze-tracking.md](01-gaze-tracking.md) - Gaze-aware LOD
- [integration/02-multidimensional-queries.md](02-multidimensional-queries.md) - Database queries
- [techniques/00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](../techniques/00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Homunculus Protocol
- [techniques/00-foveated-rendering-04-vlm-chain-of-focus-2025-01-30.md](../techniques/00-foveated-rendering-04-vlm-chain-of-focus-2025-01-30.md) - Adaptive search

**Other Oracles**:
- **john-vervaeke-oracle**: Complete RR framework source
- **vision-image-patching-oracle**: Patching strategies
- **deepseek-ocr-oracle**: Optical compression
- **ovis-2-5-oracle**: Native resolution processing
- **qwen3vl-oracle**: Interleaved M-RoPE, DeepStack

---

**Last Updated**: 2025-01-30
**Status**: Synthesis of Vervaeke RR + 2024-2025 VLM research
**Relevance**: ★★★★★ (Core theoretical framework for adaptive LOD)
