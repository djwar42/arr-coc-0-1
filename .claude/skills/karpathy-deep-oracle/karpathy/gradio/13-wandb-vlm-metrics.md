# W&B VLM-Specific Metrics

Vision-Language Models (VLMs) require specialized metrics beyond standard ML tracking. This guide covers VLM training metrics, ARR-COC relevance tracking, and debugging visualizations for multimodal models.

---

## Section 1: VLM Training Metrics (Standard)

### Core Loss Metrics

**Vision-Language Alignment Loss**
- Contrastive loss between image and text embeddings
- Cross-entropy for caption generation tasks
- Triplet loss for retrieval tasks

From [Neptune.ai Multimodal LLMs Guide](https://neptune.ai/blog/multimodal-large-language-models) (accessed 2025-01-31):
- Alignment metrics measure cross-modal connections between visual and text modalities
- Contrastive learning matches representations of same concepts (e.g., word "car" with car image)
- Similarity scores (cosine distance) quantify alignment quality

**Language Generation Quality**
```python
import wandb

# Log core losses
wandb.log({
    "train/vision_text_loss": vision_text_loss,
    "train/caption_loss": caption_loss,
    "train/perplexity": torch.exp(caption_loss),
    "step": global_step
})
```

**Key metrics to track:**
- Cross-entropy loss (caption generation)
- Perplexity (language quality)
- Contrastive loss (CLIP-style models)
- KL divergence (distribution matching)

### Visual Encoder Metrics

**Image Feature Quality**
- Visual embedding norm
- Attention entropy (how focused are attention heads)
- Patch-level feature statistics

```python
# Track visual encoder health
wandb.log({
    "vision/embedding_norm": vision_embeddings.norm(dim=-1).mean(),
    "vision/attention_entropy": attention_entropy,
    "vision/patch_variance": patch_features.var(dim=0).mean(),
})
```

**Why this matters:**
- Embedding norm collapse → model degradation
- Low attention entropy → model ignoring visual info
- Low patch variance → encoder not discriminating features

### Token Budget Metrics

**Vision Token Usage**
- Number of visual tokens per image
- Token compression ratio
- Computational efficiency (FLOPs per token)

From [Multimodal LLM Architecture](https://neptune.ai/blog/multimodal-large-language-models) (accessed 2025-01-31):
- MLLMs use specialized encoders (e.g., CLIP ViT-L/14) to extract visual features
- Token count directly impacts computational cost and context window usage
- Typical range: 64-400 tokens per image patch depending on compression

```python
# Track token efficiency
num_vision_tokens = vision_features.shape[1]
num_text_tokens = input_ids.shape[1]
compression_ratio = original_patch_count / num_vision_tokens

wandb.log({
    "tokens/vision_tokens": num_vision_tokens,
    "tokens/text_tokens": num_text_tokens,
    "tokens/total_tokens": num_vision_tokens + num_text_tokens,
    "tokens/compression_ratio": compression_ratio,
    "tokens/vision_percentage": num_vision_tokens / (num_vision_tokens + num_text_tokens),
})
```

### Cross-Modal Fusion Metrics

**Modality Balance**
- Text vs vision contribution to predictions
- Cross-attention weights distribution
- Modality dropout sensitivity

```python
# Track modality balance
text_only_logits = model(text_only=True)
vision_only_logits = model(vision_only=True)
full_logits = model(text_and_vision=True)

wandb.log({
    "fusion/text_contribution": (full_logits - vision_only_logits).abs().mean(),
    "fusion/vision_contribution": (full_logits - text_only_logits).abs().mean(),
    "fusion/cross_attn_mean": cross_attention_weights.mean(),
    "fusion/cross_attn_max": cross_attention_weights.max(),
})
```

### Evaluation Metrics

**VQA (Visual Question Answering)**
- Accuracy (exact match)
- BLEU score (n-gram overlap)
- CIDEr (consensus-based evaluation)

**Image Captioning**
- BLEU-1, BLEU-4
- METEOR (synonym matching)
- CIDEr (human judgment correlation)
- SPICE (scene graph matching)

**Image-Text Retrieval**
- Recall@K (K=1, 5, 10)
- Mean reciprocal rank (MRR)
- Normalized discounted cumulative gain (NDCG)

```python
# Log evaluation metrics
wandb.log({
    "eval/vqa_accuracy": vqa_acc,
    "eval/caption_bleu4": bleu4,
    "eval/caption_cider": cider,
    "eval/retrieval_recall@5": recall_at_5,
})
```

---

## Section 2: ARR-COC Specific Metrics

### Relevance Score Tracking

**Three Ways of Knowing**
Track propositional, perspectival, and participatory relevance scores per image patch.

From ARR-COC validation requirements:
- Propositional knowing: Statistical information content (Shannon entropy)
- Perspectival knowing: Salience landscapes (Jungian archetypes)
- Participatory knowing: Query-content coupling (cross-attention)

```python
# Log relevance scores
wandb.log({
    "relevance/propositional_mean": propositional_scores.mean(),
    "relevance/propositional_std": propositional_scores.std(),
    "relevance/perspectival_mean": perspectival_scores.mean(),
    "relevance/perspectival_std": perspectival_scores.std(),
    "relevance/participatory_mean": participatory_scores.mean(),
    "relevance/participatory_std": participatory_scores.std(),
    "step": global_step
})

# Log score distributions as histograms
wandb.log({
    "relevance/propositional_hist": wandb.Histogram(propositional_scores),
    "relevance/perspectival_hist": wandb.Histogram(perspectival_scores),
    "relevance/participatory_hist": wandb.Histogram(participatory_scores),
})
```

### Token Allocation Metrics

**Per-Patch Token Budgets**
Track how many tokens each patch receives (64-400 range).

```python
# Track token allocation distribution
token_allocations = []  # Per patch: [64, 128, 256, 400, ...]

wandb.log({
    "allocation/min_tokens": min(token_allocations),
    "allocation/max_tokens": max(token_allocations),
    "allocation/mean_tokens": np.mean(token_allocations),
    "allocation/median_tokens": np.median(token_allocations),
    "allocation/token_budget_hist": wandb.Histogram(token_allocations),
})

# Track budget distribution by relevance
high_relevance_patches = token_allocations[relevance_scores > threshold]
low_relevance_patches = token_allocations[relevance_scores <= threshold]

wandb.log({
    "allocation/high_relevance_mean": np.mean(high_relevance_patches),
    "allocation/low_relevance_mean": np.mean(low_relevance_patches),
    "allocation/relevance_correlation": np.corrcoef(relevance_scores, token_allocations)[0, 1],
})
```

### Compression Ratios

**LOD (Level of Detail) Efficiency**
Track compression effectiveness per relevance level.

```python
# Compression tracking
original_tokens = 1024  # Full resolution ViT tokens
compressed_tokens = sum(token_allocations)
compression_ratio = original_tokens / compressed_tokens

wandb.log({
    "compression/total_ratio": compression_ratio,
    "compression/tokens_saved": original_tokens - compressed_tokens,
    "compression/efficiency": performance_metric / compressed_tokens,  # Performance per token
})

# Track compression per relevance quartile
for i, quartile in enumerate(["Q1_low", "Q2", "Q3", "Q4_high"]):
    quartile_patches = get_quartile_patches(relevance_scores, i)
    quartile_compression = original_patch_tokens / quartile_patches.tokens.mean()

    wandb.log({
        f"compression/{quartile}_ratio": quartile_compression,
        f"compression/{quartile}_tokens": quartile_patches.tokens.mean(),
    })
```

### Opponent Processing Metrics

**Tension Navigation**
From ARR-COC architecture: Track how model balances opposing forces.

```python
# Three key tensions in relevance realization
tensions = {
    "compress_vs_particularize": (compression_score, particularization_score),
    "exploit_vs_explore": (exploitation_score, exploration_score),
    "focus_vs_diversify": (focus_score, diversity_score),
}

for tension_name, (score_a, score_b) in tensions.items():
    wandb.log({
        f"tension/{tension_name}_balance": score_a / (score_a + score_b),
        f"tension/{tension_name}_magnitude": abs(score_a - score_b),
    })
```

### Transjective Relevance

**Query-Content Coupling**
Measure how relevance emerges from query-image relationship (not objective or subjective).

```python
# Track transjective metrics
baseline_relevance = compute_relevance(image_only=True)  # Objective
query_relevance = compute_relevance(query_only=True)     # Subjective
coupled_relevance = compute_relevance(query_and_image=True)  # Transjective

wandb.log({
    "transjective/coupling_gain": coupled_relevance - (baseline_relevance + query_relevance) / 2,
    "transjective/interaction_strength": coupled_relevance / baseline_relevance,
    "transjective/query_sensitivity": (coupled_relevance - baseline_relevance).abs().mean(),
})
```

---

## Section 3: Debugging Visualizations

### Attention Heatmaps

**Cross-Modal Attention Patterns**
Visualize which image patches attend to which query tokens.

From [Multimodal Model Debugging Research](https://arxiv.org/abs/2207.00056) (accessed 2025-01-31):
- MultiViz framework enables visualization of cross-modal attention patterns
- Debugging MLLMs requires understanding modality interactions
- Attention heatmaps reveal which visual regions influence text generation

```python
import matplotlib.pyplot as plt
import numpy as np

# Get cross-attention weights [num_heads, query_len, image_patches]
attn_weights = model.get_cross_attention()

# Average over heads
avg_attn = attn_weights.mean(dim=0)  # [query_len, image_patches]

# Reshape to 2D grid (assuming 14x14 patches)
patch_grid = avg_attn.reshape(query_len, 14, 14)

# Log for each query token
for i, token in enumerate(query_tokens):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(patch_grid[i].cpu(), cmap='hot', interpolation='nearest')
    ax.set_title(f"Attention for token: {token}")
    plt.colorbar(im, ax=ax)

    wandb.log({
        f"attention/token_{i}_{token}": wandb.Image(fig),
        "step": global_step
    })
    plt.close(fig)
```

### Patch Selection Visualization

**Show which patches get high/low token budgets**

```python
# Visualize token allocation overlaid on image
def visualize_token_allocation(image, patch_coords, token_budgets, relevance_scores):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Token budget heatmap
    budget_grid = create_grid_from_patches(patch_coords, token_budgets)
    im1 = axes[1].imshow(budget_grid, cmap='viridis', alpha=0.7)
    axes[1].imshow(image, alpha=0.3)
    axes[1].set_title("Token Budget (64-400)")
    plt.colorbar(im1, ax=axes[1])

    # Relevance heatmap
    relevance_grid = create_grid_from_patches(patch_coords, relevance_scores)
    im2 = axes[2].imshow(relevance_grid, cmap='plasma', alpha=0.7)
    axes[2].imshow(image, alpha=0.3)
    axes[2].set_title("Relevance Score")
    plt.colorbar(im2, ax=axes[2])

    return fig

# Log visualization
fig = visualize_token_allocation(image, patches, budgets, relevance)
wandb.log({
    "debug/token_allocation": wandb.Image(fig),
    "step": global_step
})
plt.close(fig)
```

### Token Budget Histograms

**Distribution Analysis**

```python
# Create comprehensive budget distribution plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Overall distribution
axes[0, 0].hist(token_allocations, bins=20, edgecolor='black')
axes[0, 0].set_title("Token Budget Distribution")
axes[0, 0].set_xlabel("Tokens per Patch")
axes[0, 0].set_ylabel("Count")

# Distribution by relevance quartile
for i, (ax, quartile) in enumerate(zip(axes.flat[1:], ["Q1", "Q2", "Q3", "Q4"])):
    quartile_budgets = get_quartile_budgets(relevance_scores, token_allocations, i)
    ax.hist(quartile_budgets, bins=10, alpha=0.7, label=quartile)
    ax.set_title(f"Relevance {quartile}")
    ax.set_xlabel("Tokens")

wandb.log({
    "debug/budget_distributions": wandb.Image(fig),
    "step": global_step
})
plt.close(fig)
```

### Failure Case Analysis

**Track and visualize problematic examples**

From [VLM debugging best practices](https://neptune.ai/blog/multimodal-large-language-models) (accessed 2025-01-31):
- MLLMs inherit hallucination issues from pre-trained LLMs
- Models may fail to ground text in visual content
- Tracking failure modes helps identify systematic biases

```python
# Detect failure cases
failures = []
for idx, (pred, target, image, query) in enumerate(validation_set):
    if is_failure(pred, target):
        failures.append({
            "idx": idx,
            "prediction": pred,
            "target": target,
            "image": image,
            "query": query,
            "error_type": classify_error(pred, target),
        })

# Log failure examples
failure_table = wandb.Table(
    columns=["image", "query", "prediction", "target", "error_type"],
    data=[[
        wandb.Image(f["image"]),
        f["query"],
        f["prediction"],
        f["target"],
        f["error_type"]
    ] for f in failures[:20]]  # Log top 20 failures
)

wandb.log({
    "debug/failure_cases": failure_table,
    "debug/failure_rate": len(failures) / len(validation_set),
    "epoch": epoch
})

# Track error type distribution
error_types = [f["error_type"] for f in failures]
error_counts = {etype: error_types.count(etype) for etype in set(error_types)}

wandb.log({
    "debug/hallucination_rate": error_counts.get("hallucination", 0) / len(failures),
    "debug/visual_grounding_errors": error_counts.get("visual_grounding", 0) / len(failures),
    "debug/language_errors": error_counts.get("language", 0) / len(failures),
})
```

### Relevance Score Evolution

**Track how relevance changes during training**

```python
# Store relevance scores for same validation images over time
validation_images = load_validation_set()
relevance_history = []

# During training loop
for epoch in range(num_epochs):
    for img_id, image in validation_images:
        relevance = compute_relevance(image, query)
        relevance_history.append({
            "epoch": epoch,
            "img_id": img_id,
            "relevance": relevance.cpu().numpy()
        })

# Visualize evolution
fig, ax = plt.subplots(figsize=(10, 6))
for img_id in range(5):  # Track 5 example images
    img_scores = [r["relevance"].mean() for r in relevance_history if r["img_id"] == img_id]
    epochs = [r["epoch"] for r in relevance_history if r["img_id"] == img_id]
    ax.plot(epochs, img_scores, label=f"Image {img_id}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Relevance Score")
ax.set_title("Relevance Score Evolution")
ax.legend()

wandb.log({
    "debug/relevance_evolution": wandb.Image(fig),
})
plt.close(fig)
```

### Modality Balance Debugging

**Detect when model over-relies on one modality**

```python
# Ablation study: drop modalities
text_only_acc = evaluate(model, text_only=True)
vision_only_acc = evaluate(model, vision_only=True)
full_acc = evaluate(model, both=True)

wandb.log({
    "debug/text_only_accuracy": text_only_acc,
    "debug/vision_only_accuracy": vision_only_acc,
    "debug/full_accuracy": full_acc,
    "debug/multimodal_gain": full_acc - max(text_only_acc, vision_only_acc),
})

# Red flag: if text_only_acc ≈ full_acc → model ignoring vision
if abs(text_only_acc - full_acc) < 0.05:
    wandb.log({
        "debug/warning": "Model may be ignoring visual input",
        "debug/vision_contribution": (full_acc - text_only_acc),
    })
```

---

## Sources

**Web Research:**
- [Neptune.ai: Multimodal Large Language Models](https://neptune.ai/blog/multimodal-large-language-models) (accessed 2025-01-31)
  - MLLM architecture (input module, fusion module, output module)
  - Token budget considerations for vision encoders
  - Alignment metrics and contrastive learning
  - Challenges: representation, alignment, reasoning, generation, transference

- [MultiViz: Visualizing Multimodal Models (arXiv:2207.00056)](https://arxiv.org/abs/2207.00056) (accessed 2025-01-31)
  - Debugging visualization frameworks for MLLMs
  - Cross-modal attention pattern analysis
  - Understanding modality interactions

- W&B search results (accessed 2025-01-31):
  - VLM training monitoring patterns
  - Image captioning metrics (BLEU, CIDEr, METEOR)
  - VQA evaluation standards

**ARR-COC Architecture References:**
- ARR-COC validation requirements (VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md)
- Three ways of knowing: propositional, perspectival, participatory
- Opponent processing: compress↔particularize, exploit↔explore, focus↔diversify
- Token allocation range: 64-400 per patch based on relevance
- Transjective relevance: query-content coupling

**Additional References:**
- VQA evaluation metrics: accuracy, BLEU, CIDEr
- Image-text retrieval: Recall@K, MRR, NDCG
- Cross-modal fusion debugging techniques
- Failure case classification for MLLMs
