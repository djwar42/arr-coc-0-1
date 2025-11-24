# BATCH 3: Token Budget Allocation Research

## Token Merging (ToMe)

**Paper:** "Token Merging: Your ViT But Faster" - Bolya et al. 2022
**Citations:** 783+

### Core Algorithm

ToMe uses **bipartite soft matching** to merge similar tokens:

```python
def token_merge(tokens, r):
    """Merge r tokens per layer using bipartite matching"""
    # Split tokens into two sets
    set_a = tokens[::2]   # Even indices
    set_b = tokens[1::2]  # Odd indices

    # Compute similarity scores
    scores = cosine_similarity(set_a, set_b)

    # Find best matches using Hungarian algorithm
    matches = bipartite_match(scores, r)

    # Merge matched tokens (average or weighted)
    merged = (set_a[matches[:, 0]] + set_b[matches[:, 1]]) / 2

    return merged
```

### Performance

- **2-3x speedup** on existing ViT models
- **No training required** - apply to pretrained models
- Minimal accuracy loss (often <1%)
- Works on ViT, DeiT, MAE, etc.

### Key Insight

Similar tokens are redundant! Merging them:
1. Reduces computation
2. Preserves information
3. Acts like learned pooling

## Token Fusion & Pruning

### Dynamic Token Sparsification (DynamicViT)

- Learns which tokens to keep vs discard
- Different sparsity patterns per image
- Gumbel-softmax for differentiable selection

### Token Fusion (Bridge between Pruning & Merging)

- Pruning: Discard tokens entirely (loses information)
- Merging: Combine tokens (preserves information)
- Fusion: Learnable combination of both strategies

## Variable Length Vision Tokens

### Challenge

Different images need different numbers of tokens:
- Simple image: 16 tokens sufficient
- Complex scene: Need 256+ tokens

### Solutions

**1. Adaptive Patching:**
```python
def adaptive_patch(image, complexity_score):
    if complexity_score < 0.3:
        return patch_size_32(image)  # 49 tokens
    elif complexity_score < 0.7:
        return patch_size_16(image)  # 196 tokens
    else:
        return patch_size_8(image)   # 784 tokens
```

**2. Early Exit:**
- Stop processing after N layers for simple inputs
- Continue full depth for complex inputs

**3. Token Budget Prediction:**
```python
class TokenBudgetPredictor(nn.Module):
    def forward(self, image_features):
        # Predict optimal token count
        budget = self.predictor(global_pool(image_features))
        return softmax(budget)  # Distribution over budgets
```

## Visual Token Compression

### LLaVA Token Efficiency

**Challenge:** Context window limits total tokens
- 8K context = need room for text + visual tokens
- Too many visual tokens → no room for reasoning

**Approaches:**

1. **Pooling:** Average/max pool spatial tokens
2. **Resampling:** Perceiver-style learned queries
3. **Compression:** Autoencoder for visual features

### Query-Aware Token Selection

Select tokens based on the question:
```python
def query_aware_selection(visual_tokens, query_embedding, k):
    # Compute relevance scores
    scores = visual_tokens @ query_embedding.T

    # Select top-k relevant tokens
    top_k_indices = torch.topk(scores, k).indices

    return visual_tokens[top_k_indices]
```

## Integration with Spicy Lentil

### Per-Slot Token Budget

Each object slot can have different token allocations:
```python
# Slot complexity determines token budget
for slot in object_slots:
    complexity = estimate_complexity(slot)
    budget = allocate_budget(complexity, total_budget)
    slot_tokens = merge_to_budget(slot.tokens, budget)
```

### Saccade-Based Token Allocation

- First pass: Coarse tokens (few per slot)
- Saccade detected: Allocate more tokens to relevant slots
- Progressive refinement

### 27.34% Token Entropy

Maintain token diversity:
```python
# Ensure sufficient token entropy
token_entropy = compute_entropy(merged_tokens)
if token_entropy < 0.2734 * max_entropy:
    # Inject diversity - don't over-merge!
    keep_more_tokens()
```

## Performance Numbers

### ToMe Results

| Model | Original FPS | ToMe FPS | Accuracy Drop |
|-------|-------------|----------|---------------|
| ViT-B/16 | 85 | 210 | -0.3% |
| ViT-L/16 | 28 | 78 | -0.4% |
| DeiT-S | 152 | 367 | -0.2% |

### Token Budget Trade-offs

| Tokens | Accuracy | Speed | Memory |
|--------|----------|-------|--------|
| 196 | 100% | 1.0x | 1.0x |
| 98 | 99.2% | 1.8x | 0.5x |
| 49 | 97.5% | 3.2x | 0.25x |

## Key Formulas

### Bipartite Matching Score
```
score(a, b) = cos(a, b) = (a · b) / (||a|| ||b||)
```

### Token Merge Operation
```
merged = (token_a + token_b) / 2
# Or weighted by importance:
merged = w_a * token_a + w_b * token_b
```

### Budget Allocation Loss
```
L_budget = λ * (actual_tokens - target_budget)²
```

## Implementation Recommendations

1. **Start with ToMe:** Zero training, immediate speedup
2. **Add query-awareness:** Select relevant tokens for VQA
3. **Learn budgets:** Predict optimal token count per image
4. **Entropy regularization:** Prevent over-merging (27.34%!)

---

**Sources:**
- "Token Merging: Your ViT But Faster" - ICLR 2023
- "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification"
- "Token Fusion: Bridging the Gap between Token Pruning and Token Merging"
- "Efficient Vision Transformer by Information Redundancy Reduction"
