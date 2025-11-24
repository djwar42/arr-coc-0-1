# Multimodal Transformer Training Pipeline

Complete guide to training vision-language models through multi-stage pipelines, covering pre-training, vision-language alignment, and instruction tuning strategies used by BLIP-2, Flamingo, LLaVA, and modern VLMs.

## Overview

Modern VLM training follows a **staged approach** to efficiently leverage pretrained models while aligning visual and language modalities. The typical pipeline consists of 2-3 stages, each with specific objectives, frozen/unfrozen components, and training data.

**Key Insight**: Staged training is more efficient than end-to-end training from scratch, leveraging billions of pre-training examples already captured in CLIP encoders and LLMs.

## Three-Stage Training Framework

### Stage 1: Vision-Language Pre-Alignment (Optional for some models)

**Goal**: Learn basic cross-modal correspondence between vision and text

**What's trained**: Visual abstractor/projection layer only
**What's frozen**: Vision encoder, language model
**Duration**: 1-3 days (8 GPUs)
**Data**: Image-caption pairs (LAION-400M, CC3M, CC12M)

**Models that use this**:
- BLIP-2 (Q-Former pre-training)
- Flamingo (Perceiver Resampler)
- LLaVA 1.5 (MLP projection warmup)

**Training recipe**:
```yaml
# Stage 1: Feature Alignment
frozen:
  - vision_encoder: true
  - language_model: true
trainable:
  - projection_layer: true  # or Q-Former, Perceiver

data:
  - dataset: "laion2b-en"
    samples: 600k
    format: "image-caption pairs"

hyperparameters:
  epochs: 1
  batch_size: 256  # per GPU
  learning_rate: 1e-3
  optimizer: AdamW
  warmup_steps: 2000
  gradient_accumulation: 4
```

**Why this stage**:
- Cheap compute (only projection layer trained)
- Aligns CLIP image embeddings → LLM token space
- Prevents catastrophic forgetting in later stages

**Loss function**:
- Image-text matching (ITM): Binary classification (match/no-match)
- Image-text contrastive (ITC): Contrastive loss (InfoNCE)
- Image-grounded text generation (ITG): Language modeling loss

### Stage 2: Vision-Language Alignment / Multimodal Pre-Training

**Goal**: Align vision encoder outputs with language model embeddings for downstream tasks

**What's trained**: Projection/resampler + optionally vision encoder
**What's frozen**: Language model (usually)
**Duration**: 3-7 days (8-32 GPUs)
**Data**: Large-scale image-text pairs (millions)

**Models that use this**:
- BLIP-2 Stage 1: Q-Former + frozen vision/LLM
- LLaVA Stage 1: Projection layer + frozen vision/LLM
- InternVL Stage 1: Vision encoder + projection

**Training recipe**:
```yaml
# Stage 2: Multimodal Pre-Training
frozen:
  - language_model: true
  - vision_encoder: false  # Can be trainable for some models
trainable:
  - visual_abstractor: true
  - vision_encoder: false  # Usually frozen

data:
  - laion2b: 600M image-text pairs
  - coyo-700m: 700M pairs
  - datacomp-1b: Filtered web pairs

tasks:
  - image_captioning
  - visual_question_answering_generation
  - image_text_matching

hyperparameters:
  epochs: 1
  batch_size: 2048  # Total across GPUs
  learning_rate: 1e-4
  optimizer: AdamW
  weight_decay: 0.05
  warmup_ratio: 0.03
  gradient_checkpointing: true
```

**Loss functions**:
```python
# BLIP-2 uses 3 objectives
total_loss = (
    lambda_itm * itm_loss +      # Image-text matching
    lambda_itc * itc_loss +      # Image-text contrastive
    lambda_itg * itg_loss        # Image-grounded text generation
)

# LLaVA uses pure next-token prediction
total_loss = language_modeling_loss(image_tokens + text_tokens)
```

**Data format**:
```json
{
  "image": "path/to/image.jpg",
  "caption": "A photo of a cat sitting on a couch"
}
```

### Stage 3: Instruction Tuning / Visual Instruction Following

**Goal**: Teach model to follow human instructions on visual tasks

**What's trained**: Visual abstractor + LLM (QLoRA/LoRA/Full FT)
**What's frozen**: Vision encoder
**Duration**: 1-3 days (8 GPUs typical)
**Data**: Instruction-following datasets (100k-1M samples)

**Training recipe**:
```yaml
# Stage 3: Instruction Tuning
frozen:
  - vision_encoder: true
trainable:
  - visual_abstractor: true
  - language_model: true  # Via LoRA/QLoRA

data:
  - llava_instruct_150k: Conversation-style QA
  - sharegpt4v: High-quality multi-turn dialogues
  - docvqa: Document understanding
  - chartqa: Chart reasoning
  - textvqa: OCR-dependent QA

format: "conversational"

hyperparameters:
  epochs: 1-3
  batch_size: 128  # Total
  learning_rate: 2e-5  # Lower for LLM tuning
  optimizer: AdamW
  lora_r: 16
  lora_alpha: 16
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Data format (conversational)**:
```json
{
  "image": "image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is shown in this image?"
    },
    {
      "from": "gpt",
      "value": "The image shows a modern kitchen with..."
    }
  ]
}
```

**Loss masking**: Only compute loss on assistant responses
```python
# Mask human prompts from loss
loss_mask = (labels != IGNORE_TOKEN_ID)
loss = F.cross_entropy(logits, labels, reduction='none')
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

## Model-Specific Training Strategies

### BLIP-2: Bootstrapping from Frozen Models

**Philosophy**: Leverage frozen pretrained models, train only Q-Former

**Architecture**:
```
CLIP ViT-L/14 (frozen) → Q-Former (32 queries) → OPT-2.7B (frozen)
```

**Stage 1: Vision-Language Representation Learning**
- Train Q-Former with frozen vision encoder
- 3 objectives: ITC, ITM, ITG
- 129M samples from filtered CapFilt
- Duration: ~6 days on 16 A100s

**Stage 2: Vision-to-Language Generative Pre-Training**
- Connect Q-Former to frozen LLM
- Language modeling objective only
- 129M samples (same dataset)
- Duration: ~5 days on 16 A100s

**Key insight**: Q-Former learns to extract text-relevant visual features with only 188M trainable parameters

### Flamingo: Few-Shot Learning Focus

**Philosophy**: Gated cross-attention for flexible image conditioning

**Architecture**:
```
NFNet (frozen) → Perceiver Resampler → Chinchilla LLM (frozen + gated cross-attn)
```

**Training stages**:
1. **Vision encoder pre-training**: Contrastive learning (CLIP-style)
2. **Multimodal pre-training**: Freeze vision, train Perceiver + cross-attention layers
   - Datasets: Interleaved image-text from web (M3W), image-text pairs, video-text
   - 15B samples total
   - Duration: Weeks on 1536 TPUv4 chips

**Key innovation**: Gated cross-attention layers inserted between LLM layers
```python
# Gated cross-attention
cross_attn_output = CrossAttention(text_hidden, image_features)
gated_output = tanh(gate) * cross_attn_output
output = text_hidden + gated_output
```

### LLaVA: Simplicity and Efficiency

**Philosophy**: Minimal architecture, maximum instruction-following

**Architecture**:
```
CLIP ViT-L/14 (frozen) → MLP projection → Vicuna-7B (trainable)
```

**Stage 1: Pre-training for Feature Alignment**
- Train 2-layer MLP projection only
- 595k image-caption pairs (filtered CC3M)
- Duration: ~4 hours on 8 A100s
- Learning rate: 1e-3

**Stage 2: Visual Instruction Tuning**
- Fine-tune MLP + LLM (LoRA or full)
- 158k unique language-image instruction samples
- Generated with GPT-4 (multimodal)
- Duration: ~1 day on 8 A100s
- Learning rate: 2e-5

**Key insight**: Simple linear projection works when leveraging strong pretrained encoders

### InternVL: Joint Training for Better Alignment

**Philosophy**: Train vision encoder end-to-end for better VLM alignment

**Architecture**:
```
InternViT-6B → Dynamic pixel resampler → InternLM-20B
```

**Training stages**:
1. **Vision-Language contrastive pre-training**: Train ViT + text encoder together
2. **Multimodal pre-training**: Unfreeze vision encoder, train with LLM
3. **Supervised fine-tuning**: Instruction datasets

**Key innovation**: Co-training vision and language improves cross-modal alignment

## Data Requirements by Stage

### Stage 1: Pre-Alignment (Optional)

| Dataset | Size | Type | Purpose |
|---------|------|------|---------|
| LAION-400M | 400M | Image-caption | General alignment |
| COYO-700M | 700M | Image-caption | Web-scale pairs |
| CC3M/CC12M | 3-12M | Image-caption | Filtered quality |

**Quality matters**: Filter with CLIP similarity > 0.28

### Stage 2: Multimodal Pre-Training

| Dataset | Size | Format | Use Case |
|---------|------|--------|----------|
| LAION-2B | 2B | Pairs | General vision-language |
| DataComp-1B | 1B | Pairs | High-quality filtered |
| COYO | 700M | Pairs | Korean + English |
| M3W (Flamingo) | Web-scale | Interleaved | Multi-image contexts |

### Stage 3: Instruction Tuning

| Dataset | Size | Task Type |
|---------|------|-----------|
| LLaVA-Instruct-150K | 150k | Conversation, detailed descriptions, reasoning |
| ShareGPT4V | 100k | High-quality multi-turn dialogues |
| A-OKVQA | 17k | Knowledge-based VQA |
| VQAv2 | 83k | General VQA |
| GQA | 72k | Compositional reasoning |
| OCR-VQA | 80k | OCR-based questions |
| TextVQA | 22k | Reading text in images |
| DocVQA | 39k | Document understanding |
| ChartQA | 18k | Chart interpretation |
| DVQA | 200k | Bar chart QA |
| AI2D | 3k | Diagram understanding |

**Synthetic data generation** (LLaVA approach):
```python
# Use GPT-4V to generate instructions from COCO images
prompt = """
Given this image, generate:
1. A detailed description
2. 3 complex reasoning questions
3. Conversational questions
Format as instruction-response pairs.
"""
# Results in 158k high-quality samples from 80k images
```

## Training Efficiency Strategies

### Progressive Unfreezing

Start with minimal trainable parameters, gradually unfreeze:

```python
# Stage 1
freeze(vision_encoder)
freeze(language_model)
train(projection_layer)

# Stage 2
freeze(vision_encoder)  # Keep frozen
freeze(language_model)  # Or use LoRA
train(projection_layer)
train(lora_adapters)  # If using PEFT

# Stage 3 (optional)
unfreeze(language_model)  # Full fine-tuning
```

### Mixed Batch Training

Combine different data types in single batch:

```python
batch = {
    'image_captions': 40%,      # Stage 1 data
    'instruction_following': 40%,  # Stage 3 data
    'visual_reasoning': 20%     # Specialized tasks
}
```

### Curriculum Learning

Order data by difficulty:

1. **Week 1**: Simple captions, single objects
2. **Week 2**: Multiple objects, spatial relations
3. **Week 3**: Complex scenes, reasoning
4. **Week 4**: Multi-turn conversations, edge cases

## Common Training Issues

### Issue 1: Catastrophic Forgetting

**Symptoms**: Model forgets vision capabilities during LLM tuning

**Solutions**:
- Keep vision encoder frozen after Stage 2
- Use lower learning rates for LLM (2e-5 vs 1e-3 for projection)
- Add replay buffer with Stage 2 data during Stage 3

### Issue 2: Hallucination

**Symptoms**: Model generates plausible but incorrect visual details

**Solutions**:
- Add negative examples ("What's NOT in this image?")
- Use POPE (Polling-based Object Probing) during training
- Lower temperature during inference (0.2-0.5)
- Add uncertainty tokens ("I'm not sure but...")

### Issue 3: Poor Cross-Modal Alignment

**Symptoms**: Model ignores image, generates generic text

**Solutions**:
- Increase Stage 1 duration (more alignment pre-training)
- Use harder negatives in contrastive learning
- Add image-conditioned prefix tokens
- Check CLIP similarity of training data (filter < 0.25)

### Issue 4: Slow Convergence

**Symptoms**: Loss plateaus early, poor downstream performance

**Solutions**:
- Use cosine learning rate schedule with warmup
- Enable gradient checkpointing for larger batches
- Verify data diversity (check for duplicates)
- Use AdamW with correct weight decay (0.01-0.1)

## Hyperparameter Guidelines

### Stage 1: Pre-Alignment

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-3 | Higher OK, only projection trained |
| Batch size | 256-2048 | Large batches help contrastive learning |
| Warmup steps | 2000-5000 | Critical for stability |
| Weight decay | 0.05 | Prevent overfitting |
| Epochs | 1 | One pass usually sufficient |
| Gradient clip | 1.0 | Prevent exploding gradients |

### Stage 2: Multimodal Pre-Training

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-4 to 5e-5 | Lower if vision encoder unfrozen |
| Batch size | 1024-4096 | As large as GPU memory allows |
| Warmup ratio | 0.03 | 3% of total steps |
| Weight decay | 0.05-0.1 | Higher for larger models |
| Epochs | 1 | Multiple epochs risk overfitting |
| Gradient accumulation | 4-16 | For effective larger batch size |

### Stage 3: Instruction Tuning

| Hyperparameter | Typical Value | Notes |
|----------------|---------------|-------|
| Learning rate | 2e-5 (full FT), 1e-4 (LoRA) | Much lower than Stage 1/2 |
| Batch size | 128-256 | Smaller OK for instruction data |
| Warmup ratio | 0.1 | Longer warmup for stability |
| Epochs | 1-3 | Monitor for overfitting |
| LoRA rank | 16-64 | Higher for complex tasks |
| LoRA alpha | rank * 1 | Standard setting |

## Data Pipeline Best Practices

### Image Preprocessing

```python
# Standard VLM preprocessing
transform = Compose([
    Resize(336, interpolation=BICUBIC),  # CLIP resolution
    CenterCrop(336),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])
])
```

### Data Augmentation (Stage 2 only)

```python
# Light augmentation, avoid destroying semantic content
augment = Compose([
    RandomResizedCrop(336, scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1),
    # NO: heavy rotation, cropping that removes key objects
])
```

### Tokenization Strategy

```python
# For LLaVA-style models
def tokenize_conversation(image, conversations):
    # Special image token
    IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_ID = 32000

    # Build prompt
    prompt = f"{IMAGE_TOKEN}\n{conversations[0]['value']}"
    response = conversations[1]['value']

    # Tokenize
    input_ids = tokenizer(prompt + response).input_ids
    labels = input_ids.copy()

    # Mask prompt tokens (don't compute loss)
    prompt_len = len(tokenizer(prompt).input_ids)
    labels[:prompt_len] = IGNORE_INDEX

    return input_ids, labels
```

## Evaluation During Training

### Stage 1: Feature Alignment Metrics

- **CLIP similarity**: Measure alignment quality (target > 0.30)
- **Image-text retrieval**: R@1, R@5 on Flickr30K
- **Visual grounding**: RefCOCO pointing accuracy

### Stage 2: Zero-Shot Capabilities

- **VQAv2**: Zero-shot QA accuracy
- **COCO Captioning**: CIDEr score
- **TextVQA**: OCR-dependent QA
- **Visual reasoning**: NLVR2 accuracy

### Stage 3: Instruction Following

- **MMBench**: Comprehensive VLM benchmark
- **SEED-Bench**: Generative evaluation
- **LLaVA-Bench**: Human-eval conversations
- **MM-Vet**: Complex integrated reasoning

## Checkpoint Strategy

```python
# Save frequency
save_policy = {
    'stage_1': {
        'frequency': 'every_epoch',  # Fast, save often
        'keep_last': 3
    },
    'stage_2': {
        'frequency': 'every_5000_steps',  # Expensive, save milestones
        'keep_last': 5,
        'keep_best': 3  # Based on validation metric
    },
    'stage_3': {
        'frequency': 'every_epoch',
        'keep_last': 3,
        'keep_best': 2  # Best on instruction-following eval
    }
}
```

## Advanced: End-to-End Training

Some recent models (Fuyu, PaliGemma) skip staged training:

**Advantages**:
- Simpler training pipeline
- Better cross-modal gradient flow
- Potentially better alignment

**Disadvantages**:
- Requires massive compute (100x+ more GPU hours)
- Needs much more data (billions of samples)
- Risk of catastrophic forgetting
- Harder to debug issues

**When to use**: Only if you have Meta/Google-scale compute and data

## Cost Estimates

### BLIP-2 Training (Reference)

- **Stage 1**: 6 days × 16 A100 GPUs = 2,304 GPU-hours (~$5k)
- **Stage 2**: 5 days × 16 A100 GPUs = 1,920 GPU-hours (~$4k)
- **Total**: ~$9k for base model pre-training

### LLaVA Training (More Accessible)

- **Stage 1**: 4 hours × 8 A100 GPUs = 32 GPU-hours (~$60)
- **Stage 2**: 20 hours × 8 A100 GPUs = 160 GPU-hours (~$350)
- **Total**: ~$410 for complete VLM

**Note**: Stage 3 (instruction tuning) can be done on consumer GPUs with QLoRA (~$50 on cloud)

## Key Takeaways

1. **Staged training is essential** for efficient VLM development
2. **Stage 1 (pre-alignment)** is cheap and prevents catastrophic forgetting
3. **Stage 2 (multimodal pre-training)** requires most compute but is one-time cost
4. **Stage 3 (instruction tuning)** is where task-specific performance emerges
5. **Keep vision frozen** after Stage 2 to preserve learned features
6. **Use PEFT (LoRA/QLoRA)** in Stage 3 to reduce memory and cost
7. **Data quality > quantity** for instruction tuning

---

**Sources:**
- Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training"
- Alayrac, J. et al. (2022). "Flamingo: a Visual Language Model for Few-Shot Learning"
- Liu, H. et al. (2024). "Visual Instruction Tuning" (LLaVA)
- Chen, Z. et al. (2024). "InternVL: Scaling up Vision Foundation Models"
- Dai, W. et al. (2023). "InstructBLIP: General-purpose Vision-Language Models"
