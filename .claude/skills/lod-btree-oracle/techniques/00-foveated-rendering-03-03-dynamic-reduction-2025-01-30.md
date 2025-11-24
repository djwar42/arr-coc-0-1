# Dynamic Token Reduction During Generation (Level 3 Deep-Dive)

**Date**: 2025-01-30
**Parent**: [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md)
**Level**: 3 of 4 (Runtime Adaptive Compression)

---

## Overview

Dynamic token reduction during generation represents a paradigm shift from static pre-processing compression to **runtime-adaptive token management** that responds to the changing importance of visual tokens as VLM generation progresses.

**Key Insight**: Visual token importance **decreases progressively** during autoregressive generation as the model shifts attention from visual context to generated text tokens.

---

## Core Concept: Generation-Aware Compression

### The Problem with Static Compression

Traditional approaches (FastV, VTW, SparseVLM) apply **fixed compression rates** before or at the start of generation:

```
Visual Tokens (576) → Prune 50% → 288 tokens (constant throughout generation)
```

**Limitation**: This ignores the autoregressive nature of VLM decoders where token importance evolves over time.

### The Dynamic Solution

**DyRate (arXiv 2501.14204, Jan 2025)** introduces compression that adapts **during each iteration** of text generation:

```
Iteration 1: Visual tokens at 100% importance → Keep 288 tokens (50% compression)
Iteration 5: Visual tokens at 70% importance → Keep 144 tokens (75% compression)
Iteration 10: Visual tokens at 40% importance → Keep 72 tokens (87.5% compression)
```

**Result**: More aggressive pruning as generation progresses, matching actual token importance.

---

## DyRate: Dynamic Rate Adjustment

**Authors**: Liang et al., Zhejiang University
**Published**: arXiv 2501.14204, January 24, 2025
**Citations**: 2 (very recent)

### Key Findings

**Empirical Observation**:
> "Our analysis reveals that the importance of visual tokens gradually decreases as the VLM progresses."

**Evidence from Attention Analysis**:
- Early generation (tokens 1-5): Visual tokens receive 60-70% of attention
- Mid generation (tokens 6-15): Visual tokens receive 40-50% of attention
- Late generation (tokens 16+): Visual tokens receive 20-30% of attention

### Architecture

**Three-Component System**:

1. **Attention Distribution Analyzer**
   - Tracks attention across 4 token types: system, image, instruction, response
   - Computes per-head attention distributions at each generation step
   - Feeds vector `v_t` to predictor

2. **Lightweight Predictor (Classifier C)**
   - Input: Attention distribution vector `v_t`
   - Output: Probability distribution over K discrete compression rates
   - **Trainable** via Gumbel-Softmax for end-to-end learning

3. **Differentiable Rate Selection**
   - Uses Gumbel-Softmax trick to sample compression rate R
   - Maintains differentiability for backpropagation
   - Applies token masks based on selected rate

### Compression Rate Formula

Discretize rates into K levels:

```
R = {r_k} for k=1 to K
r_k = (k-1)/K

For K=5:
r_1 = 0.0  (keep all tokens)
r_2 = 0.25 (prune 25%)
r_3 = 0.50 (prune 50%)
r_4 = 0.75 (prune 75%)
r_5 = 1.0  (prune all - extreme case)
```

**Predictor learns to map**: Attention distribution → Optimal r_k for current iteration

### Token Selection (Inherited from FastV)

Once rate R is determined, tokens are sorted by "attention-score" rule:

```python
# Compute importance scores
importance = sum(attention_weights_from_all_heads)

# Sort tokens by importance
sorted_tokens = sort(tokens, key=importance, descending=True)

# Apply dynamic mask
num_keep = int(num_tokens * (1 - R))
mask[sorted_tokens[:num_keep]] = 1  # Keep
mask[sorted_tokens[num_keep:]] = 0  # Prune
```

### Gumbel-Softmax Training

**Challenge**: Discrete sampling is non-differentiable
**Solution**: Gumbel-Softmax provides differentiable approximation

```python
# Forward pass: Sample discrete rate
R = Gumbel_Softmax(π_R)  # One-hot vector in {0,1}^K

# Backward pass: Straight-through estimator
gradient flows through π_R (continuous probabilities)
```

This enables **end-to-end training** of the predictor without pretraining.

---

## Performance Results

### Benchmark Performance (LLaVA-1.5-7B)

**Short Response Tasks**:

| Benchmark | Original | FastV (R=0.5) | DyRate | Improvement |
|-----------|----------|---------------|--------|-------------|
| GQA | 62.0 | 60.3 | **61.9** | +1.6 vs FastV |
| VisWiz | 50.0 | 54.4 | **54.2** | -0.2 (comparable) |
| SQA^I | 66.8 | 69.0 | **69.2** | +0.2 |
| VQA^T | 58.2 | 45.4 | **45.7** | +0.3 |
| POPE | 85.9 | 82.5 | **86.8** | +4.3 (significant) |

**Long Response Tasks** (CIDEr scores):

| Dataset | Original | FastV (K=3,R=0.5) | VTW (K=16,R=1) | DyRate | Improvement |
|---------|----------|-------------------|----------------|--------|-------------|
| Nocaps | 74.89 | 74.75 | 44.54 | **75.00** | +0.11 |
| Flickr30k | 105.57 | 105.00 | 58.00 | **108.41** | +3.41 |
| COCO2017 | 110.43 | 110.80 | 67.20 | **110.54** | -0.26 (comparable) |

### Efficiency Gains

**Computational Savings** (vs LLaVA-1.5-7B baseline):

| Model | TFLOPs (%) | Latency (ms) | Nocaps CIDEr | Auto R? |
|-------|-----------|--------------|--------------|---------|
| Original | 100.0 | 70.80 | 74.89 | ✗ |
| FastV (K=3,R=0.5) | 57.90 | 42.36 | 74.75 | ✗ |
| **DyRate** | **33.33** | **40.13** | **75.00** | ✓ |

**Key Wins**:
- **42% fewer FLOPs** than static FastV
- **43% faster** than original (70.80ms → 40.13ms)
- **Automatic rate selection** - no manual tuning required
- **Better accuracy** despite more aggressive compression

---

## Layer-Adaptive Compression

### Depth-Based Pruning (DP)

Alternative strategy: Adjust compression rate based on **layer depth** rather than generation iteration.

**Intuition**: Deeper layers may require fewer visual tokens as information gets more abstract.

**Formula**:
```
C_retain = 1 - H(L_index - 4) * P_prune_4th - H(L_index - 4) * R'

Where:
- C_retain: Proportion of tokens kept at current layer
- L_index: Current layer index
- P_prune_4th: Pruning ratio at 4th layer
- R': Modified pruning ratio adapting to layer
- H(): Heaviside step function
```

**Results** (LLaVA-1.5-7B, Nocaps):

| Strategy | Nocaps CIDEr | FLOPs (%) |
|----------|--------------|-----------|
| Fixed Prune (FP) | 107.00 | 28.80 |
| Depth-Based Prune (DP) | 105.00 | 58.10 |
| **DyRate (Attention-Based)** | **109.57** | **33.33** |

**Conclusion**: Attention-based dynamic pruning outperforms layer-based heuristics.

---

## Attention-Guided Dropping

### Multi-Head Importance Scoring

**Observation**: Different attention heads focus on different aspects of visual input.

**Strategy**: Weight visual token importance by **aggregating scores across all heads**:

```python
for head in attention_heads:
    importance[token] += attention_weights[head, token, :]

# Normalize across tokens
importance = importance / sum(importance)

# Prune tokens below threshold
threshold = percentile(importance, R * 100)
keep_tokens = [t for t in tokens if importance[t] > threshold]
```

**Advantage**: More nuanced than simple spatial or temporal heuristics.

---

## Related Work: Generation-Stage Dropping

### HiRED (AAAI 2025, 41 citations)

**"Attention-Guided Token Dropping for Efficient Video Large Multimodal Models"**

**Three-Stage Dropping**:
1. **Encoder stage**: Prune redundant patches in visual encoder
2. **Projection stage**: Reduce tokens in cross-modal projector
3. **LLM stage**: Adaptive dropping during text generation ← **Related to DyRate**

**Mechanism**:
- Tracks attention scores from current generated token to visual tokens
- Drops visual tokens with attention below dynamic threshold
- **85% reduction** in visual tokens with minimal accuracy loss

**Difference from DyRate**:
- HiRED: Hard dropping based on attention threshold
- DyRate: Soft probabilistic rate selection via learned predictor

### AIM (arXiv 2412.03248)

**"Adaptive Importance-based Merging"**

**Hybrid Approach**: Combines merging (like ToMe) with pruning

**Generation-Aware Component**:
- Monitors KL divergence between distributions with/without token reduction
- Adjusts merge/prune aggressiveness if divergence exceeds threshold
- **Not fully dynamic** - still uses fixed schedule with safety checks

---

## Integration with Levels 1-2

### Complementary to Token Merging/Pruning (Level 1)

**Combine Static + Dynamic**:

```
Pre-generation (Level 1 - AIM):
  576 tokens → 288 tokens (50% via hybrid merge+prune)

During generation (Level 3 - DyRate):
  Iteration 1: 288 tokens → 216 tokens (25% dynamic prune)
  Iteration 5: 288 tokens → 144 tokens (50% dynamic prune)
  Iteration 10: 288 tokens → 72 tokens (75% dynamic prune)
```

**Total compression**: Up to **87.5%** (576 → 72 tokens) by late generation

### Complementary to Progressive Compression (Level 2)

**Multi-Scale + Multi-Iteration**:

Level 2 (PVC): Compress across spatial/temporal dimensions before generation
Level 3 (DyRate): Further compress during generation iterations

**Example**:
```
Video input: 8 frames × 576 tokens = 4608 tokens

PVC (Level 2): 4608 → 1152 tokens (75% reduction across frames)
DyRate (Level 3): 1152 → 288 tokens (dynamic, by iteration 10)

Combined: 4608 → 288 = 93.75% reduction
```

---

## When to Use Dynamic Reduction

### Ideal Scenarios

✅ **Long text generation** (>16 tokens)
- Visual importance decreases significantly
- More aggressive pruning justified in later iterations

✅ **Conversational VLMs**
- Multi-turn dialogues where visual context becomes less critical
- Can prune aggressively after initial grounding

✅ **Deployment constraints**
- Need to maintain quality while hitting strict latency budgets
- Automatic rate selection avoids manual tuning

### Not Ideal Scenarios

❌ **Short captions** (<8 tokens)
- Insufficient time for adaptive benefit
- Static compression simpler and comparable

❌ **Dense visual reasoning** (OCR, charts, diagrams)
- Visual tokens remain important throughout generation
- Dynamic pruning may discard needed information

❌ **Offline processing**
- Overhead of predictor not justified if latency doesn't matter
- Static high-quality compression (Level 1-2) sufficient

---

## Code Example: DyRate Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DyRatePredictor(nn.Module):
    """Lightweight classifier for dynamic compression rate"""
    def __init__(self, num_heads=32, num_token_types=4, num_rates=5):
        super().__init__()
        input_dim = num_heads * num_token_types  # Attention dist per head
        self.classifier = nn.Linear(input_dim, num_rates)
        self.num_rates = num_rates

    def forward(self, attention_dist, temperature=1.0):
        """
        Args:
            attention_dist: [batch, num_heads, num_token_types]
            temperature: Gumbel-Softmax temperature
        Returns:
            rate_logits: [batch, num_rates]
        """
        # Flatten attention distribution
        batch_size = attention_dist.shape[0]
        v_t = attention_dist.view(batch_size, -1)

        # Predict rate probabilities
        logits = self.classifier(v_t)  # [batch, num_rates]
        return logits

class DyRatePruner:
    """Dynamic token pruning during VLM generation"""
    def __init__(self, predictor, num_rates=5):
        self.predictor = predictor
        self.num_rates = num_rates
        self.rate_values = torch.linspace(0, 1, num_rates)  # [0, 0.25, 0.5, 0.75, 1.0]

    def compute_token_importance(self, attention_weights):
        """Aggregate attention scores (FastV-style)"""
        # attention_weights: [batch, num_heads, seq_len, num_visual_tokens]
        importance = attention_weights.sum(dim=1)  # Sum across heads
        importance = importance.mean(dim=1)  # Mean across sequence
        return importance  # [batch, num_visual_tokens]

    def prune_tokens(self, visual_tokens, attention_dist, training=False, temperature=1.0):
        """
        Args:
            visual_tokens: [batch, num_tokens, dim]
            attention_dist: [batch, num_heads, 4] (sys, img, ins, res)
            training: Whether in training mode
            temperature: Gumbel-Softmax temperature
        Returns:
            pruned_tokens: [batch, num_keep, dim]
            mask: [batch, num_tokens] (binary)
        """
        batch_size, num_tokens, dim = visual_tokens.shape

        # Predict compression rate
        rate_logits = self.predictor(attention_dist, temperature)

        if training:
            # Gumbel-Softmax: differentiable sampling
            rate_probs = F.gumbel_softmax(rate_logits, tau=temperature, hard=True)
        else:
            # Inference: argmax
            rate_idx = rate_logits.argmax(dim=-1)
            rate_probs = F.one_hot(rate_idx, num_classes=self.num_rates).float()

        # Convert to compression rate
        rates = (rate_probs @ self.rate_values.to(rate_probs.device)).unsqueeze(-1)

        # Compute token importance
        importance = self.compute_token_importance(attention_dist)  # [batch, num_tokens]

        # Create masks for each rate
        masks = []
        for k in range(self.num_rates):
            r_k = self.rate_values[k]
            num_keep = int(num_tokens * (1 - r_k))

            # Sort by importance, keep top-k
            _, indices = importance.topk(num_keep, dim=-1)
            mask_k = torch.zeros(batch_size, num_tokens, device=visual_tokens.device)
            mask_k.scatter_(1, indices, 1.0)
            masks.append(mask_k)

        # Weighted combination of masks (differentiable)
        masks = torch.stack(masks, dim=1)  # [batch, num_rates, num_tokens]
        final_mask = (rate_probs.unsqueeze(-1) * masks).sum(dim=1)  # [batch, num_tokens]

        # Apply mask
        pruned_tokens = visual_tokens * final_mask.unsqueeze(-1)

        return pruned_tokens, final_mask

# Usage in VLM generation loop
predictor = DyRatePredictor(num_heads=32, num_token_types=4, num_rates=5)
pruner = DyRatePruner(predictor)

for iteration in range(max_iterations):
    # Get current attention distribution
    attention_dist = model.get_attention_dist()  # [batch, num_heads, 4]

    # Dynamically prune visual tokens
    visual_tokens, mask = pruner.prune_tokens(
        visual_tokens,
        attention_dist,
        training=True,
        temperature=0.5
    )

    # Continue generation with pruned tokens
    next_token = model.generate_next_token(visual_tokens, text_tokens)
    text_tokens = torch.cat([text_tokens, next_token], dim=1)
```

---

## Future Directions

### Multi-Modal Rate Prediction

Current DyRate: Uses only attention distribution
**Future**: Incorporate text semantics

```python
# Predict rate based on both attention and text content
rate = predictor(attention_dist, current_text_embeddings)

# Example: Keep more visual tokens when text mentions "image", "picture", "see"
```

### Layer-Wise Dynamic Rates

Current: Same rate across all layers
**Future**: Different rates per layer

```
Layer 1-4: Conservative pruning (R=0.25)
Layer 5-8: Moderate pruning (R=0.5, dynamic)
Layer 9-12: Aggressive pruning (R=0.75, dynamic)
```

### Cross-Attention Recycling

Instead of discarding pruned tokens, **cache for potential reuse**:

```python
pruned_cache = []  # Store pruned tokens

if generation_stalls:  # Model uncertain
    # Retrieve relevant pruned tokens
    relevant = find_relevant(pruned_cache, current_context)
    visual_tokens = concat(visual_tokens, relevant)
```

---

## Cross-References

**Related LOD Oracle Files**:
- [00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md](00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md) - Level 1: Static compression
- [00-foveated-rendering-03-02-progressive-compression-2025-01-30.md](00-foveated-rendering-03-02-progressive-compression-2025-01-30.md) - Level 2: Multi-scale compression
- [00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Parent: VLM token allocation overview

**Related Techniques**:
- [00-foveated-rendering.md](00-foveated-rendering.md) - Biological inspiration for adaptive allocation
- [00-foveated-rendering-01-logpolar-mapping-2025-01-30.md](00-foveated-rendering-01-logpolar-mapping-2025-01-30.md) - Log-polar transforms for spatial compression

---

## References

1. **DyRate** - Liang et al., "Dynamic Token Reduction during Generation for Vision Language Models," arXiv:2501.14204, Jan 2025. [2 citations]

2. **HiRED** - Arif et al., "Attention-Guided Token Dropping for Efficient Video Large Multimodal Models," AAAI 2025. [41 citations]

3. **AIM** - "Adaptive Importance-based Merging for Vision-Language Models," arXiv:2412.03248, Dec 2024.

4. **FastV** - Chen et al., "An image is worth 1/2 tokens after layer 2," arXiv:2403.06764, 2024.

5. **VTW** - Lin et al., "Boosting multimodal large language models with visual tokens withdrawal," arXiv:2405.05803, 2024.

---

**Status**: Research-grounded (5 papers cited)
**Next Level**: [00-foveated-rendering-03-04-training-free-methods-2025-01-30.md](00-foveated-rendering-03-04-training-free-methods-2025-01-30.md) - Level 4: Zero-shot deployment
