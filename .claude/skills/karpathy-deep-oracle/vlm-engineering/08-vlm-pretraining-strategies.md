# VLM Pre-training Strategies

**From contrastive learning to multi-task objectives: the foundation of vision-language models**

---

## Overview

Vision-language model (VLM) pre-training establishes the foundation for models to understand the relationship between images and text. Unlike supervised fine-tuning which adapts models to specific tasks, pre-training creates general-purpose representations that bridge visual and linguistic modalities.

**Key insight from VILA research** ([VILA: On Pre-training for Visual Language Models, Lin et al., CVPR 2024](https://arxiv.org/abs/2312.07533)):
> "Freezing LLMs during pre-training can achieve decent zero-shot performance, but lack in-context learning capability"

Pre-training objectives must balance:
- **Modality alignment**: Learning shared representations across vision and text
- **Computational efficiency**: Training on millions to billions of image-text pairs
- **Generalization**: Supporting diverse downstream tasks without catastrophic forgetting

---

## Section 1: Pre-training Objectives Taxonomy

### The Four Core Objectives

From [karpathy/training-llms/00-overview.md](../../training-llms/00-overview.md) (LLM pre-training baseline):
- **Language modeling**: Predict next token given context
- **Training scale**: 99% of compute happens in pre-training
- **Data mixture**: Web scrapes, code, books, academic papers

**VLM-specific objectives extend this with cross-modal tasks:**

1. **Image-Text Contrastive (ITC)** - Learn aligned embeddings
2. **Image-Text Matching (ITM)** - Binary classification of image-text pairs
3. **Masked Language Modeling with Vision (MLM)** - Predict masked tokens using image context
4. **Image Captioning** - Generate text descriptions from images

### Objective Comparison

| Objective | Granularity | Supervision | Use Case |
|-----------|-------------|-------------|----------|
| **ITC** | Global (full image/caption) | Weakly supervised | Zero-shot retrieval, embedding alignment |
| **ITM** | Global (binary match) | Weakly supervised | Fine-grained matching, hard negatives |
| **MLM** | Token-level (masked words) | Self-supervised | Language understanding with visual grounding |
| **Captioning** | Sequence-level (full text) | Supervised | Generative tasks, dense descriptions |

From [Analytics Vidhya VLM Overview](https://www.analyticsvidhya.com/blog/2024/07/vision-language-models/) (accessed 2025-11-16):
> "Learning Objectives: VLMs differ from solely computer vision-based models through various VLM-based pre-training objectives including contrastive learning, masked language modeling, and image-text matching."

---

## Section 2: Contrastive Learning (CLIP-style, InfoNCE Loss)

### Algorithm: Learning Through Contrast

From [OpenAI CLIP](https://openai.com/index/clip/) (accessed 2025-11-16):
> "CLIP (Contrastive Language-Image Pre-training) builds on a large body of work on zero-shot transfer, natural language supervision, and multimodal learning."

**Core mechanism:**
```python
# Conceptual CLIP training
image_features = vision_encoder(images)  # [B, D]
text_features = text_encoder(captions)   # [B, D]

# L2 normalize
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# Cosine similarity matrix
logits = image_features @ text_features.T  # [B, B]
logits = logits * temperature  # Learnable temperature (τ ≈ 0.07)

# Symmetric cross-entropy loss
labels = torch.arange(B)  # Diagonal = positive pairs
loss_i2t = F.cross_entropy(logits, labels)  # Image → Text
loss_t2i = F.cross_entropy(logits.T, labels)  # Text → Image
loss = (loss_i2t + loss_t2i) / 2
```

### InfoNCE Loss Formulation

For a batch of N image-text pairs, the contrastive loss maximizes similarity of matched pairs while minimizing similarity of mismatched pairs:

**Image-to-Text direction:**
```
L_i2t = -log( exp(sim(i, t+) / τ) / Σ_j exp(sim(i, tj) / τ) )
```

Where:
- `i` = image embedding
- `t+` = matching text embedding
- `tj` = all text embeddings in batch (negatives)
- `τ` = temperature parameter (typically 0.07)

**Key properties:**
- **Symmetric**: Both I→T and T→I directions trained
- **In-batch negatives**: Other batch samples serve as hard negatives
- **Temperature scaling**: Controls sharpness of similarity distribution

From [OpenCLIP Reproducible Scaling Laws](https://arxiv.org/abs/2212.07143) (Cherti et al., 2022, 1,276 citations):
> "We investigate scaling laws for contrastive language-image pre-training (CLIP) with the public LAION dataset and the open-source OpenCLIP repository."

### Data Scale Requirements

**CLIP (original, OpenAI 2021):**
- 400M image-text pairs (WIT dataset)
- 32 epochs
- 592 V100 GPUs for ~2 weeks

**OpenCLIP (open-source):**
- LAION-400M: 400M pairs
- LAION-2B: 2 billion pairs
- LAION-5B: 5.85 billion pairs (multilingual)

**Performance scaling:**
- CLIP ViT-B/32: 400M pairs → 63.2% ImageNet zero-shot
- CLIP ViT-L/14: 400M pairs → 75.5% ImageNet zero-shot
- OpenCLIP ViT-g/14: 2B pairs → 80.1% ImageNet zero-shot

**Training costs:**
- ViT-B/32 on LAION-400M: ~$2,000 (8×A100, 2 days)
- ViT-L/14 on LAION-2B: ~$50,000 (256×A100, 9 days)
- ViT-g/14 on LAION-2B: ~$200,000 (512×A100, 14 days)

### Temperature Scaling

The temperature parameter τ controls loss sharpness:

```python
# Low temperature (τ = 0.01): Very sharp, hard negatives dominate
logits = similarity / 0.01  # Multiplies similarities by 100

# Medium temperature (τ = 0.07): Standard CLIP setting
logits = similarity / 0.07  # Multiplies similarities by ~14

# High temperature (τ = 0.5): Softer, easier optimization
logits = similarity / 0.5   # Multiplies similarities by 2
```

**Learnable temperature** (modern practice):
```python
self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
temperature = self.temperature.exp()  # Ensures τ > 0
```

From [karpathy/training-llms/01-four-stage-pipeline.md](../../training-llms/01-four-stage-pipeline.md) (pre-training philosophy):
> "Pre-training stage is where all of the computational work basically happens this is 99% of the training compute time"

---

## Section 3: Image-Text Matching (ITM)

### Binary Classification Objective

ITM treats vision-language alignment as a binary classification problem: given an image-text pair, predict whether they match.

**Architecture:**
```python
# After cross-modal fusion (e.g., Q-Former or cross-attention)
fusion_features = cross_modal_encoder(image_features, text_features)

# Binary classification head
itm_logits = nn.Linear(hidden_dim, 2)(fusion_features[:, 0])  # [B, 2]

# Binary cross-entropy loss
itm_loss = F.cross_entropy(itm_logits, labels)  # labels: 1 (match), 0 (mismatch)
```

### Hard Negative Mining

From [Selectively Hard Negative Mining for ITM](https://arxiv.org/abs/2303.00181) (Li et al., 2023, 11 citations):
> "Most existing ITM models suffer from gradient vanishing issue caused by excessive easy negative samples"

**Problem**: Random negative sampling creates mostly easy negatives
- 99% of random pairs are obviously mismatched
- Model quickly learns to ignore easy negatives
- Gradients become very small, slowing learning

**Solution: Hard negative mining**
```python
# 1. Compute image-text similarities using contrastive model
with torch.no_grad():
    similarities = image_features @ text_features.T  # [B, B]

# 2. Sample hard negatives (high similarity but wrong match)
hard_neg_indices = torch.topk(similarities, k=5, dim=1).indices
hard_neg_indices = hard_neg_indices[:, 1:]  # Exclude positive (index 0)

# 3. Create ITM training batch
positive_pairs = (images, matching_texts, labels=1)
hard_negative_pairs = (images, hard_negative_texts, labels=0)

# 4. Train ITM head on mixed batch
itm_loss = train_itm(positive_pairs + hard_negative_pairs)
```

**Strategies:**
- **In-batch hard negatives**: Use most similar non-matching pairs within batch
- **Cross-batch hard negatives**: Cache hard negatives across batches
- **Curriculum learning**: Start with random negatives, increase difficulty over time

From [Integrating Language Guidance Into ITM](https://ieeexplore.ieee.org/document/10081045) (Li et al., 2023, 32 citations):
> "We propose an ITM framework integrating Language Guidance (LG) for correcting false negatives"

### ITM Loss Formulation

```python
def itm_loss(image_features, text_features, labels):
    """
    Args:
        image_features: [B, D] vision features
        text_features: [B, D] text features
        labels: [B] binary labels (1 = match, 0 = mismatch)

    Returns:
        Binary cross-entropy loss
    """
    # Cross-modal fusion (e.g., concatenate + MLP)
    fusion = torch.cat([image_features, text_features], dim=-1)
    logits = classification_head(fusion)  # [B, 2]

    return F.cross_entropy(logits, labels)
```

**Typical batch composition:**
- 50% positive pairs (matched image-text)
- 25% hard negative images (similar but wrong image)
- 25% hard negative texts (similar but wrong caption)

From [Bi-directional Image-Text Matching](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011573) (Tuerhong et al., 2024, 3 citations):
> "We propose a novel end-to-end image-text matching approach considering semantic uncertainty (SU-ITM), aiming to deal with the one-to-many semantic diversity"

---

## Section 4: Masked Language Modeling with Vision (MLM)

### Extending BERT MLM to Vision-Language

From [Data Efficient Masked Language Modeling for Vision and Language](https://arxiv.org/abs/2109.02040) (Bitton et al., 2021, 27 citations):
> "Masked language modeling (MLM) is one of the key sub-tasks in vision-language pretraining. In the cross-modal setting, tokens in the sentence are masked at random, and the model learns to predict them based on the image and text context."

**Standard MLM (BERT-style):**
```python
# Mask 15% of text tokens
text = "A dog playing in the park"
masked_text = "A [MASK] playing in the [MASK]"

# Predict masked tokens using text context only
predicted = bert(masked_text)  # → "dog", "park"
```

**Vision-grounded MLM:**
```python
# Mask tokens but use BOTH image and text context
image_features = vision_encoder(image)
text_features = text_encoder(masked_text)
fused_features = cross_modal_fusion(image_features, text_features)

# Predict masked tokens using multimodal context
predictions = mlm_head(fused_features)  # Uses visual evidence!
```

### Masking Strategies

From [Bitton et al. (2021)](https://aclanthology.org/2021.findings-emnlp.259/) research:

**Random masking (baseline):**
- Mask 15% of tokens uniformly
- ~50% are stop-words or punctuation
- Model learns little from masked stop-words

**Object-aware masking:**
- Detect objects in image (e.g., "dog", "frisbee", "grass")
- Preferentially mask object-related words
- Forces model to use visual grounding

**Hard masking:**
- Mask words that are difficult to predict from text alone
- Example: "The [MASK] is red" → Could be anything
- But with image of apple: Must predict "apple"

### MLM Loss Formulation

```python
def masked_lm_loss(image, text, mask_indices):
    """
    Args:
        image: Input image
        text: Token sequence with [MASK] tokens
        mask_indices: Indices of masked positions

    Returns:
        Cross-entropy loss on masked tokens
    """
    # Encode image and text
    image_features = vision_encoder(image)
    text_features = text_encoder(text)

    # Cross-modal fusion
    fused = cross_attention(text_features, image_features)

    # Predict only masked positions
    predictions = mlm_head(fused[mask_indices])
    targets = original_tokens[mask_indices]

    return F.cross_entropy(predictions, targets)
```

**Typical masking probabilities:**
- 80% → Replace with [MASK] token
- 10% → Replace with random token
- 10% → Keep original token

From [Masked Vision and Language Modeling](https://arxiv.org/abs/2208.02131) (Kwon et al., 2022, 95 citations):
> "In this paper, we study how to use masked signal modeling in vision and language (V+L) representation learning"

### Benefits of Vision-Grounded MLM

1. **Forces visual grounding**: Model must attend to image to predict masked words
2. **Learns fine-grained alignment**: Word-level vision-text correspondence
3. **Complements contrastive learning**: ITC learns global alignment, MLM learns local
4. **Improves downstream VQA**: Better at answering specific questions about images

**Example:**
```
Image: [Photo of red apple on wooden table]
Text: "A fresh [MASK] sits on the [MASK] table"

Without image: Could predict "fruit" / "wooden"
With image: Must predict "apple" / "wooden" (visual evidence for red apple)
```

---

## Section 5: Multi-Task Pre-training

### Combining Objectives for Better Representations

From [karpathy/practical-implementation/46-frozen-backbone-adapter-training.md](../practical-implementation/46-frozen-backbone-adapter-training.md) (multi-stage training):
> "Multi-stage training philosophy: Projector pre-training → SFT → Alignment due to limited availability of high-quality data, memory constraints, and stability issues"

**Standard multi-task pre-training:**
```python
def vlm_pretraining_loss(batch):
    images, captions = batch

    # Encode both modalities
    image_features = vision_encoder(images)
    text_features = text_encoder(captions)

    # 1. Contrastive loss (ITC)
    loss_itc = contrastive_loss(image_features, text_features)

    # 2. Image-text matching (ITM)
    # Generate hard negatives
    hard_neg_images, hard_neg_texts = sample_hard_negatives(
        image_features, text_features
    )
    loss_itm = itm_loss(
        images, captions, hard_neg_images, hard_neg_texts
    )

    # 3. Masked language modeling (MLM)
    masked_captions, targets = mask_tokens(captions)
    loss_mlm = masked_lm_loss(images, masked_captions, targets)

    # 4. Image captioning (optional)
    loss_cap = captioning_loss(images, captions)

    # Weighted combination
    total_loss = (
        w_itc * loss_itc +
        w_itm * loss_itm +
        w_mlm * loss_mlm +
        w_cap * loss_cap
    )

    return total_loss
```

### Loss Weighting Strategies

**BLIP-2 approach** ([Li et al., ICML 2023](https://arxiv.org/abs/2301.12597)):
- Stage 1 (vision-language alignment): ITC + ITM + MLM
  - w_itc = 1.0, w_itm = 1.0, w_mlm = 1.0
- Stage 2 (generative pre-training): Captioning only
  - w_cap = 1.0

**ALBEF approach** (adaptive loss balancing):
```python
# Dynamic weighting based on loss magnitudes
weights = {
    'itc': 1.0 / loss_itc.detach(),
    'itm': 1.0 / loss_itm.detach(),
    'mlm': 1.0 / loss_mlm.detach()
}
# Normalize
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}
```

**Curriculum weighting** (easier → harder):
```python
# Early training: Focus on contrastive learning
if epoch < 5:
    w_itc, w_itm, w_mlm = 1.0, 0.1, 0.1
# Mid training: Add ITM
elif epoch < 15:
    w_itc, w_itm, w_mlm = 1.0, 1.0, 0.5
# Late training: Full multi-task
else:
    w_itc, w_itm, w_mlm = 1.0, 1.0, 1.0
```

### Data Scale and Mixture

From [karpathy/training-llms/00-overview.md](../../training-llms/00-overview.md) (data mixture strategy):
> "Data mixture strategy (LLaMA example): CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Books 4.5%, ArXiv 2.5%, StackExchange 2%"

**VLM pre-training data mixture:**

| Source | Scale | Quality | Use Case |
|--------|-------|---------|----------|
| **LAION-400M** | 400M pairs | Medium | General contrastive learning |
| **COYO-700M** | 700M pairs | Medium | Diverse web images |
| **CC12M** | 12M pairs | High | Curated web images |
| **SBU Captions** | 1M pairs | High | Detailed descriptions |
| **Visual Genome** | 100K images | Very high | Dense annotations, relationships |
| **COCO Captions** | 120K images | Very high | High-quality human captions |

**Training schedule:**
```python
# Stage 1: Large-scale noisy data (80% of steps)
# Dataset: LAION-400M
# Objectives: ITC (contrastive learning)
# Goal: Learn general vision-language alignment

# Stage 2: High-quality data (15% of steps)
# Dataset: CC12M + COCO + Visual Genome
# Objectives: ITC + ITM + MLM
# Goal: Refine alignment, learn fine-grained matching

# Stage 3: Task-specific data (5% of steps)
# Dataset: VQA, captioning, grounding datasets
# Objectives: All objectives + task-specific heads
# Goal: Prepare for downstream fine-tuning
```

---

## Section 6: Computational Efficiency Techniques

### Frozen Encoders Strategy

From [karpathy/practical-implementation/46-frozen-backbone-adapter-training.md](../practical-implementation/46-frozen-backbone-adapter-training.md):
> "Freezing pre-trained backbone models during vision-language model (VLM) training has become the dominant approach in modern multimodal AI"

**Memory savings (7B VLM example):**
```
Frozen backbones:
- Vision encoder (CLIP ViT-L): 304M params → No gradients, no optimizer states
- Language model (LLaMA-7B): 7B params → No gradients, no optimizer states
- Trainable projector: 21M params → Gradients + Adam states

Memory usage:
- Frozen: 14.6 GB (FP16 weights only)
- Trainable projector: 21M × 12 bytes (FP16 weight + FP32 grad + 2× optimizer) = 252 MB
- Total: ~15 GB vs. ~64 GB (fully trainable)

Speedup: 3-5× faster training
```

### Gradient Checkpointing

```python
# Trade compute for memory
vision_encoder.gradient_checkpointing_enable()
text_encoder.gradient_checkpointing_enable()

# Recomputes activations during backward pass
# Memory: 40-50% reduction
# Speed: 20-30% slower
```

### Mixed Precision Training

From [karpathy/training-llms/07-mixed-precision-2025-best-practices.md](../../training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md) (FP16/BF16 training):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward in FP16/BF16
with autocast(dtype=torch.bfloat16):
    loss = vlm_pretraining_loss(batch)

# Backward with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2× memory reduction (FP32 → FP16)
- 2-3× speedup on modern GPUs (A100, H100)
- BF16 preferred for stability (same range as FP32)

### Distributed Training

From [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../../distributed-training/00-deepspeed-zero-optimizer.md) (ZeRO optimization):

**ZeRO-2 for VLM pre-training:**
```json
{
  "train_batch_size": 2048,
  "gradient_accumulation_steps": 16,
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

**Enables:**
- 13B VLM on 8×A100 (80GB)
- Effective batch size: 2048 samples
- Gradient accumulation: 16 steps per optimizer update

---

## Section 7: Data Quality and Filtering

### Noisy Web Data Challenge

**LAION-400M statistics** (web-scraped alt-text):
- 30% completely mismatched (image and text unrelated)
- 40% partially matched (general relationship but not specific)
- 30% well-matched (accurate description)

**Impact on model quality:**
- Training on 100% noisy data → 60% zero-shot ImageNet accuracy
- Training on 50% filtered data → 70% zero-shot ImageNet accuracy
- Training on high-quality only → 75% accuracy (but 10× less data)

### Filtering Strategies

**1. CLIP-based filtering:**
```python
# Use pre-trained CLIP to score image-text similarity
clip_model = load_pretrained_clip()
similarity = clip_model.compute_similarity(image, text)

# Keep only high-similarity pairs
if similarity > threshold:  # e.g., 0.25
    keep_sample = True
```

**2. Text quality filtering:**
```python
def filter_text(caption):
    # Remove too short
    if len(caption.split()) < 5:
        return False

    # Remove non-English (if English-only model)
    if detect_language(caption) != 'en':
        return False

    # Remove spam patterns
    if contains_url(caption) or is_advertisement(caption):
        return False

    return True
```

**3. Image quality filtering:**
```python
def filter_image(image):
    # Remove too small
    if min(image.size) < 224:
        return False

    # Remove duplicates (perceptual hashing)
    if is_duplicate(image, seen_hashes):
        return False

    # Remove NSFW/inappropriate
    if nsfw_detector(image) > threshold:
        return False

    return True
```

**DataComp filtering** (state-of-the-art):
- Combine CLIP similarity + text quality + image quality
- Train small CLIP model on filtered data
- Use it to filter larger dataset iteratively
- Results: 40% accuracy improvement with 50% less data

---

## Section 8: ARR-COC-0-1 Pre-training Strategy

### Relevance-Aware Pre-training Objectives

**ARR-COC-0-1 specific requirements:**

From the arr-coc-0-1 MVP architecture:
- 13-channel texture array (RGB, LAB, Sobel, spatial, eccentricity)
- Adaptive token allocation (64-400 tokens per patch)
- Three ways of knowing (Propositional, Perspectival, Participatory)
- Opponent processing for balancing tensions

**Custom pre-training objectives:**

**1. Relevance-weighted contrastive loss:**
```python
def relevance_weighted_itc(image, text, relevance_scores):
    """
    Weight contrastive loss by relevance realization scores

    Args:
        image: Image features [B, D]
        text: Text features [B, D]
        relevance_scores: Per-sample relevance [B]
    """
    # Standard contrastive logits
    logits = image @ text.T / temperature

    # Weight by relevance (high relevance = more important)
    weighted_logits = logits * relevance_scores.unsqueeze(1)

    labels = torch.arange(B)
    loss = F.cross_entropy(weighted_logits, labels)

    return loss
```

**2. Multi-scale ITM (patch-level matching):**
```python
def multiscale_itm(image_patches, text, patch_relevance):
    """
    Match text to relevant image patches, not just full image

    Args:
        image_patches: [B, num_patches, D]
        text: [B, D]
        patch_relevance: [B, num_patches] relevance scores
    """
    # Compute similarity of each patch to text
    similarities = image_patches @ text.unsqueeze(2)  # [B, num_patches, 1]

    # Weight by relevance scores
    weighted_sim = similarities.squeeze(-1) * patch_relevance

    # Aggregate: max-pool high-relevance patches
    match_score = weighted_sim.max(dim=1).values

    return match_score
```

**3. Vervaekean MLM (three ways of knowing):**
```python
def vervaekean_mlm(image_texture_array, masked_text):
    """
    Predict masked tokens using three ways of knowing:
    - Propositional: Statistical content (entropy)
    - Perspectival: Salience (edge strength)
    - Participatory: Query-image coupling (cross-attention)
    """
    # Extract multi-channel features
    rgb = texture_array[:, :3]      # Color content
    lab = texture_array[:, 3:6]     # Perceptual color
    sobel = texture_array[:, 6:8]   # Edge information
    spatial = texture_array[:, 8:10]  # Position
    ecc = texture_array[:, 10:13]   # Eccentricity

    # Three ways of knowing
    propositional = entropy_scorer(rgb, lab)
    perspectival = salience_scorer(sobel, ecc)
    participatory = query_coupling(masked_text, texture_array)

    # Opponent processing: balance tensions
    relevance = opponent_process(
        propositional, perspectival, participatory
    )

    # Predict masked tokens weighted by relevance
    predictions = mlm_head(relevance)

    return predictions
```

### Training Schedule

**Stage 1: Texture feature learning (30% of steps)**
- Dataset: LAION-400M (web-scale)
- Objectives: ITC only
- Frozen: Language model
- Trainable: Vision encoder (texture extraction) + projector
- Goal: Learn 13-channel texture representations

**Stage 2: Relevance realization (50% of steps)**
- Dataset: CC12M + COCO (high-quality)
- Objectives: ITC + Relevance-weighted ITM + Vervaekean MLM
- Frozen: Vision encoder (texture extractor)
- Trainable: Relevance scorers + opponent processor + projector + LLM (LoRA)
- Goal: Learn to allocate attention based on relevance

**Stage 3: VQA fine-tuning (20% of steps)**
- Dataset: VQAv2 + GQA
- Objectives: All objectives + VQA task loss
- Trainable: Full model (or LoRA)
- Goal: Optimize for question answering with adaptive LOD

**Expected computational cost:**
- 8-node, 64-GPU setup (A100 80GB)
- Total training time: 7-10 days
- Cost estimate: $150,000-$200,000

### Key Innovations

1. **Texture-aware pre-training**: 13-channel input forces model to learn multi-scale features
2. **Relevance weighting**: Pre-training loss weighted by Vervaekean relevance scores
3. **Adaptive token allocation**: Model learns to allocate 64-400 tokens based on query
4. **Opponent processing**: Three ways of knowing balanced through tension navigation

---

## Sources

**Primary Research:**
- [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533) - Lin et al., CVPR 2024 (680 citations)
- [OpenAI CLIP](https://openai.com/index/clip/) - Radford et al., 2021 (accessed 2025-11-16)
- [Reproducible Scaling Laws for Contrastive Language-Image Learning](https://arxiv.org/abs/2212.07143) - Cherti et al., 2022 (1,276 citations)
- [Data Efficient Masked Language Modeling for Vision and Language](https://arxiv.org/abs/2109.02040) - Bitton et al., 2021 (27 citations)

**ITM Research:**
- [Selectively Hard Negative Mining for ITM](https://arxiv.org/abs/2303.00181) - Li et al., 2023 (11 citations)
- [Integrating Language Guidance Into ITM](https://ieeexplore.ieee.org/document/10081045) - Li et al., IEEE 2023 (32 citations)
- [Bi-directional Image-Text Matching](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011573) - Tuerhong et al., Neurocomputing 2024 (3 citations)

**MLM Research:**
- [Masked Vision and Language Modeling](https://arxiv.org/abs/2208.02131) - Kwon et al., 2022 (95 citations)

**Source Documents:**
- [training-llms/00-overview.md](../../training-llms/00-overview.md) - LLM pre-training fundamentals
- [training-llms/01-four-stage-pipeline.md](../../training-llms/01-four-stage-pipeline.md) - Pre-training compute requirements
- [practical-implementation/46-frozen-backbone-adapter-training.md](../practical-implementation/46-frozen-backbone-adapter-training.md) - Frozen encoder strategies
- [distributed-training/00-deepspeed-zero-optimizer.md](../../distributed-training/00-deepspeed-zero-optimizer.md) - ZeRO optimization for large-scale training

**Web Resources:**
- [Analytics Vidhya: Vision-Language Models](https://www.analyticsvidhya.com/blog/2024/07/vision-language-models/) - VLM pre-training overview (accessed 2025-11-16)
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip) - Open-source CLIP implementation
- [Hugging Face VLM Guide](https://huggingface.co/blog/vlms) - Practical VLM training (accessed 2025-11-16)

---

*Knowledge domain: Vision-language pre-training, contrastive learning, multi-task objectives, VLM training at scale*
*Created: 2025-11-16*
*For: ARR-COC-0-1 relevance-aware VLM development*
