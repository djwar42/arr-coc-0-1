# BLIP-2 Q-Former Training Recipe

Complete hyperparameter recipe for training BLIP-2's Querying Transformer (Q-Former) with frozen vision encoders and language models.

## Overview

BLIP-2 achieves state-of-the-art vision-language performance through a two-stage bootstrapping strategy using a lightweight Q-Former (188M parameters) to bridge frozen image encoders and frozen LLMs. The Q-Former uses 32 learnable queries to extract visual features most relevant to text.

**Key Innovation**: Freeze expensive pre-trained models (ViT, LLM) and only train the lightweight Q-Former bridge.

**Source**: Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (2023)

## Q-Former Architecture

### Model Configuration

```
Q-Former Parameters: 188M
Number of Queries: 32
Query Dimension: 768 (hidden dimension)
Base Architecture: BERT-base initialization
Cross-Attention: Inserted every other transformer block
Frozen Components: Vision encoder (ViT-L/g), Language model (OPT/FlanT5)
```

**Design Philosophy**: Queries are model parameters (not input tokens), learned to extract task-relevant visual features.

## Two-Stage Training Strategy

### Stage 1: Vision-Language Representation Learning

**Objective**: Bootstrap vision-language alignment from frozen image encoder

**Training Steps**: 250,000 steps
**Batch Size**:
- ViT-L: 2,320
- ViT-g: 1,680

**Optimizer**: AdamW
- Weight decay: 0.05
- Peak learning rate: 1e-4
- Warmup steps: 2,000 (linear)
- Schedule: Cosine decay

**Loss Functions**: Combined objective with three losses:

1. **ITC (Image-Text Contrastive)**: InfoNCE loss for global alignment
2. **ITM (Image-Text Matching)**: Binary classification (match/no-match)
3. **ITG (Image-Grounded Text Generation)**: Language modeling loss

**Datasets**:
- COCO Captions
- Visual Genome
- CC3M (Conceptual Captions 3M)
- SBU Captions

**Computational Requirements**:
- 8x A100 GPUs (typical)
- ~10 days training time
- Mixed precision (FP16/BF16)

### Stage 2: Vision-to-Language Generative Learning

**Objective**: Bootstrap vision-to-language generation from frozen LLM

**Training Steps**: 80,000 steps
**Batch Size**:
- OPT models: 1,920
- FlanT5 models: 1,520

**Optimizer**: AdamW
- Weight decay: 0.05
- Peak learning rate: 1e-4
- Warmup steps: 2,000 (linear)
- Schedule: Cosine decay
- Minimum learning rate: 5e-5

**Loss Functions**: Language modeling loss only

**Key Change**: Q-Former output fed as soft visual prompts to frozen LLM

**Connection Strategy**:
- Q-Former outputs → Fully connected layer → LLM input dimension
- 32 query outputs become 32 visual tokens for LLM

## Detailed Hyperparameters

### Learning Rate Schedule

```python
# Stage 1
peak_lr = 1e-4
warmup_steps = 2000
total_steps = 250000
schedule = "cosine"
min_lr = 0  # Decay to zero

# Stage 2
peak_lr = 1e-4
warmup_steps = 2000
total_steps = 80000
schedule = "cosine"
min_lr = 5e-5  # Don't decay below this
```

**Rationale**: Linear warmup prevents instability, cosine decay smoothly reduces LR.

### Optimizer Configuration

```python
optimizer = "AdamW"
weight_decay = 0.05
betas = (0.9, 0.999)  # Adam defaults
eps = 1e-8
gradient_clipping = 1.0  # Max gradient norm
```

**Why AdamW**: Decoupled weight decay improves generalization vs Adam.

### Batch Size Selection

**Stage 1 (Vision-Language Representation)**:
- ViT-L encoder: 2,320 batch size (smaller encoder)
- ViT-g encoder: 1,680 batch size (larger encoder, less fits in memory)

**Stage 2 (Generative Learning)**:
- OPT LLM: 1,920 batch size
- FlanT5 LLM: 1,520 batch size

**Gradient Accumulation**: Use if needed to achieve effective batch size on limited GPUs.

### Mixed Precision Training

```python
precision = "fp16"  # or "bf16" on A100/H100
loss_scaling = True  # For FP16
gradient_scaling = "dynamic"
```

**BF16 vs FP16**: BF16 preferred on A100+ for better training stability (wider dynamic range).

## Q-Former Query Initialization

**Strategy**: Random initialization from BERT-base weights

```python
# Pseudocode
q_former = BERTModel.from_pretrained("bert-base-uncased")
learned_queries = nn.Parameter(torch.randn(32, 768))  # 32 queries, 768 dim
cross_attention_layers = nn.ModuleList([
    CrossAttentionBlock() for _ in range(num_blocks // 2)
])
```

**32 Queries Rationale**: Balances expressiveness (enough queries to capture diverse visual concepts) with efficiency (not too many parameters).

## Training Loss Weights

### Stage 1 Multi-Task Loss

```python
# Loss weighting (defaults from paper)
loss_itc_weight = 1.0   # Image-text contrastive
loss_itm_weight = 1.0   # Image-text matching
loss_itg_weight = 1.0   # Image-grounded text generation

total_loss = (
    loss_itc_weight * loss_itc +
    loss_itm_weight * loss_itm +
    loss_itg_weight * loss_itg
)
```

**Equal Weighting**: Paper uses 1.0 for all three losses (no tuning needed in most cases).

### Stage 2 Generation Loss

```python
# Only language modeling loss
loss = cross_entropy_loss(
    lm_logits,
    target_tokens,
    ignore_index=pad_token_id
)
```

## Hard Negative Mining (ITM Loss)

**Strategy**: For image-text matching, use in-batch negatives + hard negatives from ITC similarity

```python
# Hard negative selection
itc_scores = compute_itc_similarity(image_features, text_features)
hard_negatives = select_top_k_hardest(itc_scores, k=128)

# ITM binary classification
itm_loss = binary_cross_entropy(
    match_scores(image, hard_negatives),
    labels  # 0 or 1
)
```

**Why Hard Negatives**: Improves discrimination by focusing on difficult examples.

## Datasets and Data Loading

### Stage 1 Datasets

```
COCO Captions: 118K images, 5 captions each = 590K pairs
Visual Genome: 100K images, ~5M region descriptions
CC3M: 3M image-caption pairs
SBU Captions: 1M image-caption pairs

Total: ~9-10M training pairs
```

**Data Augmentation**:
- Random resized crop (224x224 or 384x384 depending on ViT)
- Horizontal flip (50% prob)
- Color jitter (brightness, contrast, saturation, hue)

### Stage 2 Datasets

**Same as Stage 1** but now training for generation task.

## Inference Configuration

### Caption Generation

```python
# Hyperparameters from paper
num_beams = 5  # Beam search width
max_length = 30  # Max caption length
min_length = 5
length_penalty = 1.0
repetition_penalty = 1.5  # Discourage repeats
```

**Nucleus Sampling Alternative**:
```python
use_nucleus_sampling = True
top_p = 0.9
temperature = 1.0
num_captions = 3  # Generate multiple diverse captions
```

## Training Stability Tricks

1. **Gradient Clipping**: Clip to max norm 1.0 to prevent exploding gradients in cross-attention
2. **Warmup**: Essential for first 2k steps to avoid early instability
3. **Frozen Components**: Vision encoder and LLM stay frozen (no gradient flow)
4. **Mixed Precision**: Use automatic mixed precision (AMP) with GradScaler
5. **Checkpointing**: Save every 10k steps in case of crashes

## Common Issues and Solutions

### Issue: Training Loss Explodes

**Solution**:
- Check gradient clipping is enabled (max_norm=1.0)
- Reduce learning rate peak (try 5e-5)
- Increase warmup steps (try 5000)

### Issue: Poor Vision-Language Alignment

**Solution**:
- Verify ITC loss is decreasing (should drop from ~6 to ~2)
- Check ITM accuracy (should reach 85-90%)
- Ensure hard negative mining is working

### Issue: OOM (Out of Memory)

**Solution**:
- Reduce batch size, use gradient accumulation
- Enable activation checkpointing
- Use BF16 instead of FP32
- Reduce image resolution (224 vs 384)

## Evaluation Metrics

**Track during training**:

Stage 1:
- ITC loss (contrastive)
- ITM loss (matching)
- ITG loss (generation)
- ITM accuracy

Stage 2:
- Language modeling loss
- Perplexity
- BLEU score (on val set)

**Final Evaluation**:
- VQAv2 accuracy (zero-shot)
- COCO caption metrics (BLEU, METEOR, CIDEr, SPICE)
- Image-text retrieval (recall@1, @5, @10)

## Computational Budget

### Stage 1 (250k steps)

- **Hardware**: 8x A100 80GB
- **Time**: ~10 days
- **Cost**: ~$4,000 on cloud (at $3.20/hr per GPU)

### Stage 2 (80k steps)

- **Hardware**: 8x A100 80GB
- **Time**: ~3 days
- **Cost**: ~$1,200 on cloud

**Total**: ~$5,200 for full BLIP-2 training from scratch (stage 1 + 2)

## Scaling to Smaller Budgets

### Budget Option 1: Use Pretrained Q-Former

**Skip Stage 1**, start from Salesforce's pretrained Q-Former checkpoint:

```python
from lavis.models import load_model_and_preprocess
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=False  # Training mode
)
# Fine-tune only on your dataset
```

**Cost**: ~$1,200 (only Stage 2 fine-tuning)

### Budget Option 2: Reduce Steps

```python
# Minimum viable training
stage_1_steps = 50000  # Down from 250k (5x reduction)
stage_2_steps = 20000  # Down from 80k (4x reduction)

# Total time: ~2 days on 8x A100
# Total cost: ~$1,000
```

**Trade-off**: Lower performance but still competitive.

### Budget Option 3: Smaller Batch Size + Grad Accumulation

```python
# Run on 4x A100 instead of 8x
per_gpu_batch = 16
accumulation_steps = 36  # Effective batch = 16 * 4 * 36 = 2304
```

**Cost**: $500-800 depending on cloud provider.

## Pre-Trained Checkpoints

**Salesforce LAVIS provides**:

```
blip2_opt2.7b:
  - pretrain_opt2.7b: Q-Former + OPT 2.7B (after Stage 1)
  - caption_coco_opt2.7b: Fine-tuned for captioning (after Stage 2)

blip2_opt6.7b:
  - pretrain_opt6.7b
  - caption_coco_opt6.7b

blip2_t5:
  - pretrain_flant5xl: Q-Former + Flan-T5-XL
  - pretrain_flant5xxl: Q-Former + Flan-T5-XXL
  - caption_coco_flant5xl
```

**Usage**: Fine-tune from these checkpoints for your specific task.

## References

**Primary Source**:
- Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" ICML 2023
  - arXiv: https://arxiv.org/abs/2301.12597
  - GitHub: https://github.com/salesforce/LAVIS

**Key Insights From**:
- BLIP-2 MLRC paper (https://proceedings.mlr.press/v202/li23q/li23q.pdf)
- HuggingFace implementation (https://huggingface.co/docs/transformers/en/model_doc/blip-2)
- Medium tutorials on Q-Former architecture

**Training Implementation**:
- Salesforce LAVIS library: https://github.com/salesforce/LAVIS
- HuggingFace Transformers: Blip2ForConditionalGeneration

## Summary

**BLIP-2 Q-Former Recipe** (TL;DR):

```yaml
Q-Former:
  queries: 32
  parameters: 188M
  base: BERT-base

Stage 1 (Representation Learning):
  steps: 250k
  batch: 2320 (ViT-L) / 1680 (ViT-g)
  optimizer: AdamW (wd=0.05)
  lr: 1e-4 peak, 2k warmup, cosine decay
  losses: ITC + ITM + ITG (equal weights)

Stage 2 (Generative Learning):
  steps: 80k
  batch: 1920 (OPT) / 1520 (FlanT5)
  optimizer: AdamW (wd=0.05)
  lr: 1e-4 peak, 2k warmup, cosine to 5e-5
  losses: Language modeling only

Frozen:
  - Vision encoder (ViT-L/g)
  - Language model (OPT/FlanT5)

Budget: ~$5k cloud cost for full training, 2 weeks
```

Congratulations! You now have the complete recipe to train BLIP-2 Q-Former. 🚀
