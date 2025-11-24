# Adaptive Patching

**Content-aware patch sizing for efficient token allocation**

## Overview

Adaptive patching dynamically adjusts patch sizes based on image content complexity, allocating more tokens to important regions and fewer to redundant areas.

**From [source-documents/03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)** and empirical research:

**Core principle**: Not all image regions deserve equal representation

## Motivation

**Problem with fixed patching**:
```
Sky region: 500 tokens (mostly uniform blue)
Text region: 500 tokens (dense information)

Information density mismatch: 100:1 ratio!
```

**Adaptive solution**:
```
Sky region: 50 tokens (large patches)
Text region: 500 tokens (small patches)

Result: Same quality, 45% fewer tokens overall
```

## Adaptive Patching Strategies

### 1. Saliency-Based Patch Selection

**Principle**: Compute saliency map, allocate smaller patches to salient regions

```python
def saliency_based_patching(image, patch_sizes=[8, 16, 32], target_tokens=512):
    """
    Adaptive patching based on saliency

    Args:
        image: [C, H, W]
        patch_sizes: Available patch sizes (small to large)
        target_tokens: Desired total token count

    Returns:
        patches: List of patches with variable sizes
        positions: Position info for each patch
    """
    # Compute saliency map
    saliency = compute_saliency(image)  # [H, W]

    # Initialize patch allocation
    patches = []
    positions = []
    remaining_budget = target_tokens

    # Greedy allocation
    h, w = image.shape[1:]
    covered = np.zeros((h, w), dtype=bool)

    while remaining_budget > 0 and not covered.all():
        # Find most salient uncovered region
        salient_region = find_most_salient_uncovered(saliency, covered)

        # Choose patch size based on local saliency variance
        local_variance = compute_local_variance(saliency, salient_region)
        if local_variance > 0.5:
            patch_size = patch_sizes[0]  # Small (high detail)
        elif local_variance > 0.2:
            patch_size = patch_sizes[1]  # Medium
        else:
            patch_size = patch_sizes[2]  # Large (low detail)

        # Extract patch
        patch = extract_patch(image, salient_region, patch_size)
        patches.append(patch)
        positions.append(salient_region)

        # Mark as covered
        mark_covered(covered, salient_region, patch_size)
        remaining_budget -= 1

    return patches, positions
```

**Saliency computation methods**:

**Edge-based saliency**:
```python
def compute_saliency_edges(image):
    """Saliency via edge density"""
    gray = rgb_to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150)

    # Smooth with Gaussian
    saliency = cv2.GaussianBlur(edges.astype(float), (15, 15), 0)

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency
```

**Attention-based saliency**:
```python
def compute_saliency_attention(image, model):
    """Saliency via self-attention"""
    # Forward pass through vision encoder
    with torch.no_grad():
        attn_weights = model.get_attention_maps(image)  # [L, H, N, N]

    # Aggregate attention across layers and heads
    # High attention = high importance
    saliency = attn_weights.mean(dim=(0, 1))  # [N, N]

    # Average attention received by each patch
    saliency = saliency.mean(dim=1)  # [N]

    # Reshape to spatial
    grid_size = int(saliency.shape[0] ** 0.5)
    saliency = saliency.reshape(grid_size, grid_size)

    return saliency
```

### 2. Hierarchical Adaptive Patching

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:

**Principle**: Multi-scale analysis, refine high-importance regions

```python
class HierarchicalAdaptivePatcher:
    def __init__(self, base_patch_size=32, min_patch_size=8):
        self.base_size = base_patch_size
        self.min_size = min_patch_size

    def patch_hierarchical(self, image, importance_threshold=0.7):
        """
        Hierarchical refinement of patches

        1. Start with coarse patches (e.g., 32×32)
        2. Compute importance for each patch
        3. Subdivide high-importance patches (→ 16×16)
        4. Repeat until min patch size or budget exhausted
        """
        patches = []
        queue = [(0, 0, image.shape[1], image.shape[2], self.base_size)]

        while queue:
            y, x, h, w, patch_size = queue.pop(0)

            # Extract region
            region = image[:, y:y+h, x:x+w]

            # Compute importance
            importance = self.compute_importance(region)

            if importance > importance_threshold and patch_size > self.min_size:
                # Subdivide into 4 quadrants
                new_size = patch_size // 2
                for dy in [0, new_size]:
                    for dx in [0, new_size]:
                        queue.append((y+dy, x+dx, new_size, new_size, new_size))
            else:
                # Keep as single patch
                patches.append({
                    'position': (y, x),
                    'size': patch_size,
                    'data': region
                })

        return patches

    def compute_importance(self, region):
        """Score region importance (entropy, variance, etc.)"""
        # Simple: use variance as proxy for complexity
        return region.var().item()
```

**Example execution**:
```
Initial: 8×8 grid of 32×32 patches (64 total)

Iteration 1:
- 2 patches exceed threshold → subdivide to 16×16
- Result: 6×8 + 2×4 = 56 patches

Iteration 2:
- 3 of the 16×16 patches subdivide to 8×8
- Result: 53 + 3×4 = 65 patches

Final: 65 patches (mix of 32×32, 16×16, 8×8)
Compared to uniform 8×8: 256 patches (4× reduction!)
```

### 3. Learned Adaptive Patch Selection

**Principle**: Train model to predict optimal patch configuration

```python
class LearnedPatchSelector(nn.Module):
    """Neural network that predicts patch configuration"""
    def __init__(self, img_size=224, embed_dim=256):
        super().__init__()

        # Lightweight encoder for patch selection
        self.selector_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

        # Predict importance scores for spatial grid
        self.importance_head = nn.Linear(embed_dim, (img_size // 16) ** 2)

    def forward(self, image, target_tokens=256):
        # Encode image
        features = self.selector_encoder(image)  # [B, D, 1, 1]
        features = features.flatten(1)  # [B, D]

        # Predict importance scores
        importance = self.importance_head(features)  # [B, N]
        importance = importance.reshape(-1, 14, 14)  # [B, H, W]

        # Select top-k patches
        patches_to_select = select_patches_by_importance(
            image, importance, target_tokens
        )

        return patches_to_select

def select_patches_by_importance(image, importance_map, k):
    """Select k patches with highest importance"""
    # Flatten importance
    importance_flat = importance_map.flatten(1)  # [B, H*W]

    # Get top-k indices
    top_k_values, top_k_indices = torch.topk(importance_flat, k, dim=1)

    # Extract corresponding patches
    patches = extract_patches_at_indices(image, top_k_indices)

    return patches, top_k_indices
```

**Training objective**:
```python
def train_patch_selector(model, dataloader, task_model):
    """Train selector to maximize task performance with minimal tokens"""
    for images, labels in dataloader:
        # Select patches
        patches, indices = model(images, target_tokens=256)

        # Task model forward
        task_output = task_model(patches)

        # Loss = task loss + sparsity penalty
        task_loss = criterion(task_output, labels)
        sparsity_penalty = 0.01 * patches.shape[1]  # Encourage fewer tokens

        loss = task_loss + sparsity_penalty
        loss.backward()
```

### 4. Token Pruning (Post-Encoding)

**Alternative approach**: Start with fixed patching, prune unimportant tokens

**From [source-documents/03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)**:

```python
class TokenPruner(nn.Module):
    """Prune tokens after initial encoding"""
    def __init__(self, embed_dim, prune_ratio=0.5):
        super().__init__()
        self.prune_ratio = prune_ratio

        # Learnable importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, tokens):
        """
        Args:
            tokens: [B, N, D] visual tokens from encoder

        Returns:
            pruned_tokens: [B, K, D] where K = N * (1 - prune_ratio)
        """
        B, N, D = tokens.shape

        # Compute importance scores
        importance = self.scorer(tokens).squeeze(-1)  # [B, N]

        # Keep top-k tokens
        k = int(N * (1 - self.prune_ratio))
        top_k_values, top_k_indices = torch.topk(importance, k, dim=1)

        # Gather selected tokens
        # Expand indices for gathering
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_tokens = torch.gather(tokens, 1, indices_expanded)

        return pruned_tokens, top_k_indices
```

**Usage in pipeline**:
```python
# Standard ViT encoding
image = preprocess(raw_image)  # [3, 224, 224]
tokens = vit_encoder(image)  # [196, 768]

# Adaptive pruning
pruner = TokenPruner(embed_dim=768, prune_ratio=0.5)
pruned_tokens, indices = pruner(tokens)  # [98, 768] (50% reduction)

# Continue with pruned tokens
llm_input = projection(pruned_tokens)
output = llm(llm_input)
```

## Implementation Details

### Patch Extraction with Variable Sizes

```python
def extract_variable_patches(image, patch_configs):
    """
    Extract patches with different sizes

    Args:
        image: [C, H, W]
        patch_configs: List of (y, x, size) tuples

    Returns:
        patches: List of [C, size, size] tensors
        embeddings: [N, D] after projection
    """
    patches = []

    for y, x, size in patch_configs:
        # Extract patch
        patch = image[:, y:y+size, x:x+size]

        # Resize to standard size if needed (for uniform embedding)
        if size != 16:  # Standard size
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(16, 16),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        patches.append(patch)

    # Stack and embed
    patches = torch.stack(patches)  # [N, C, 16, 16]
    embeddings = patch_embedding_layer(patches)  # [N, D]

    return embeddings
```

### Position Encoding for Variable Patches

```python
def create_adaptive_position_encoding(patch_configs, embed_dim):
    """Position encoding for variable-sized patches"""
    positions = []

    for y, x, size in patch_configs:
        # Normalize position to [0, 1]
        pos_y = y / image_height
        pos_x = x / image_width

        # Include size information
        size_normalized = size / max_patch_size

        # Create position vector
        pos_vec = torch.tensor([pos_y, pos_x, size_normalized])

        # Project to embedding dim
        pos_embed = position_mlp(pos_vec)  # [D]
        positions.append(pos_embed)

    return torch.stack(positions)  # [N, D]
```

## Efficiency Gains

### Theoretical Analysis

**Token reduction** for typical images:

**Natural scene** (30% salient regions):
```
Fixed patching (16×16): 256 tokens
Adaptive (70% use 32×32, 30% use 16×16):
  - 70% → 64 tokens (32×32)
  - 30% → 77 tokens (16×16)
  - Total: 141 tokens (45% reduction)
```

**Document image** (80% text regions):
```
Fixed patching: 256 tokens
Adaptive (80% use 8×8, 20% use 32×32):
  - 80% → 819 tokens (8×8)
  - 20% → 16 tokens (32×32)
  - Total: 835 tokens (3.3× increase, but better quality!)
```

**Key insight**: Adaptive patching adjusts to content needs

### Empirical Results

**From [source-documents/03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)**:

| Method | Tokens | ImageNet Acc | Speedup |
|--------|--------|--------------|---------|
| Baseline ViT | 196 | 81.8% | 1× |
| Token pruning (50%) | 98 | 81.2% | 1.6× |
| Token pruning (70%) | 59 | 79.8% | 2.4× |
| Adaptive patching | 120 | 81.5% | 1.4× |

**Key findings**:
- 50% pruning: <1% accuracy loss
- Adaptive patching: Better accuracy-efficiency tradeoff than fixed pruning

## Practical Considerations

### When to Use Adaptive Patching

**✅ Good for**:
- Varied content (some complex, some simple)
- Strict token budgets
- Heterogeneous datasets (documents, scenes, diagrams)
- Real-time applications (can reduce latency)

**❌ Not ideal for**:
- Uniformly complex images (all regions important)
- When pretrained fixed-patch models are sufficient
- Simple implementation requirements
- When training cost outweighs inference savings

### Hyperparameter Tuning

**Patch size range**:
- **Narrow range** (e.g., [12, 16, 20]): Subtle adaptation, easier training
- **Wide range** (e.g., [8, 16, 32]): More flexibility, harder to train

**Target token count**:
- Too low: May miss important details
- Too high: Minimal efficiency gains
- **Recommendation**: Start with 50-70% of fixed patching token count

**Importance threshold**:
- Low threshold: More aggressive pruning
- High threshold: Conservative (keeps more tokens)
- **Recommendation**: Tune on validation set for task-specific sweet spot

### Integration with Existing Models

**Option 1: Replace patch embedding**
```python
# Before: Fixed patching
tokens = fixed_patch_embed(image)

# After: Adaptive patching
tokens = adaptive_patch_embed(image, target_tokens=256)

# Rest of model unchanged
output = transformer_encoder(tokens)
```

**Option 2: Add pruning layer**
```python
# Encode with fixed patching
tokens = vit_encoder(image)  # [B, 196, D]

# Prune tokens
tokens_pruned = token_pruner(tokens)  # [B, 98, D]

# Project to LLM
llm_input = projector(tokens_pruned)
```

## Challenges and Solutions

### Challenge 1: Position Encoding

**Problem**: Variable patch positions/sizes complicate position encoding

**Solution**: Learned position embeddings that include size info
```python
pos_embed = MLP([y_pos, x_pos, patch_size]) → [D]
```

### Challenge 2: Training Instability

**Problem**: Discrete patch selection is non-differentiable

**Solutions**:
- **Gumbel-Softmax**: Differentiable approximation
- **REINFORCE**: Policy gradient
- **Straight-through estimator**: Gradient approximation

```python
# Gumbel-Softmax for patch selection
def select_patches_differentiable(importance_scores, k, temperature=1.0):
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_scores)))
    perturbed_scores = (importance_scores + gumbel_noise) / temperature

    # Soft top-k (differentiable approximation)
    soft_mask = torch.softmax(perturbed_scores, dim=-1)

    return soft_mask
```

### Challenge 3: Batch Heterogeneity

**Problem**: Different images → different token counts → hard to batch

**Solutions**:
- **Padding**: Pad to max length (wastes computation)
- **Dynamic batching**: Group similar token counts
- **Nested tensors**: PyTorch 2.0+ supports variable-length sequences

```python
# Dynamic batching
def create_adaptive_batches(images, batch_size=32):
    # Estimate token counts
    token_counts = [estimate_tokens(img) for img in images]

    # Sort by token count
    sorted_indices = np.argsort(token_counts)

    # Create batches with similar counts
    batches = []
    for i in range(0, len(images), batch_size):
        batch_indices = sorted_indices[i:i+batch_size]
        batches.append([images[idx] for idx in batch_indices])

    return batches
```

## Future Directions

1. **Query-aware adaptive patching**: Patch selection based on user query
2. **Online adaptation**: Refine patches during forward pass (like progressive rendering)
3. **Neural architecture search**: Find optimal patch configurations automatically
4. **Multimodal importance**: Use text query to guide visual patch selection

## Primary Sources

- [03_Balanced Token Pruning](../source-documents/03_Balanced Token Pruning_ Accelerating Vision Language Models Beyond Local Optimization.md)
- [14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)
- [12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)

## Related Documents

- [00-fixed-patching.md](00-fixed-patching.md) - Standard baseline approach
- [03-compression-strategies.md](03-compression-strategies.md) - Post-encoding compression
- [../concepts/02-token-efficiency.md](../concepts/02-token-efficiency.md) - Efficiency principles
- [../examples/01-adaptive-implementation.md](../examples/01-adaptive-implementation.md) - Code examples
