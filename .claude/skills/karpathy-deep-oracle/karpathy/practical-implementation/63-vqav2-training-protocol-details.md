# VQAv2 Training Protocol Details

Practical training protocols for Visual Question Answering on VQAv2 dataset, including data preparation, hyperparameters, augmentation strategies, and achieving SOTA performance.

## VQAv2 Dataset Overview

**VQAv2**: 1.1M questions on 204k COCO images (train+val+test)
- **Train**: 443k questions, 82k images
- **Val**: 214k questions, 40k images
- **Test-dev**: 214k questions, 40k images
- **Test-std**: 447k questions, 81k images

**Task**: Given image + natural language question → predict answer
**Evaluation**: Soft accuracy (answer may have multiple valid forms)

**Answer Distribution**:
- Yes/No questions: 38%
- Number questions: 12%
- Other (open-ended): 50%

## Standard Training Pipeline

### Data Preparation

**Answer vocabulary** (critical for VQAv2):
```python
# Build answer vocabulary from training set
# Standard: top 3129 most frequent answers (covers ~90% of train answers)
def build_answer_vocab(annotations, min_freq=9):
    answer_counter = Counter()
    for ann in annotations:
        # VQA soft accuracy: multiple valid answers per question
        for answer in ann['answers']:
            answer_counter[answer['answer']] += 1

    # Filter by frequency
    vocab = {ans: idx for idx, (ans, count) in enumerate(answer_counter.items())
             if count >= min_freq}
    return vocab  # Typically 3129 answers

# Map rare answers to <unk>
answer_to_idx = build_answer_vocab(train_annotations, min_freq=9)
```

**Data format**:
```json
{
  "image_id": 262145,
  "question": "What color is the cat?",
  "question_id": 262145000,
  "answers": [
    {"answer": "gray", "answer_confidence": "yes"},
    {"answer": "grey", "answer_confidence": "yes"},
    {"answer": "gray and white", "answer_confidence": "maybe"}
  ]
}
```

### Model Architecture (Baseline)

```python
class VQAModel(nn.Module):
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=512, num_answers=3129):
        super().__init__()
        # Vision encoder (frozen CLIP)
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Question encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Answer classifier
        self.classifier = nn.Linear(hidden_dim, num_answers)

    def forward(self, images, question_ids):
        # Extract features
        vision_feats = self.vision_encoder(images).pooler_output  # [B, 768]
        text_feats = self.text_encoder(question_ids).pooler_output  # [B, 768]

        # Fuse modalities
        combined = torch.cat([vision_feats, text_feats], dim=-1)  # [B, 1536]
        fused = self.fusion(combined)  # [B, 512]

        # Classify answer
        logits = self.classifier(fused)  # [B, 3129]
        return logits
```

### Training Hyperparameters

**Baseline configuration** (competitive performance ~72% accuracy):
```yaml
model:
  vision_encoder: "openai/clip-vit-large-patch14"
  text_encoder: "bert-base-uncased"
  hidden_dim: 512
  num_answers: 3129
  dropout: 0.5

training:
  epochs: 15
  batch_size: 512  # Across GPUs
  learning_rate: 1e-4
  optimizer: Adam
  weight_decay: 0
  lr_schedule: "constant"  # or "cosine"

  # Critical for VQA
  label_smoothing: 0.1
  gradient_clip: 1.0

data:
  image_size: 336  # CLIP resolution
  max_question_length: 20  # Tokens
  augmentation: "basic"  # Flip, color jitter
```

**SOTA configuration** (>78% accuracy):
```yaml
model:
  vision_encoder: "openai/clip-vit-large-patch14-336"  # Higher res
  text_encoder: "roberta-large"  # Better than BERT
  hidden_dim: 1024  # Larger
  num_answers: 3129
  dropout: 0.3  # Lower for larger model
  use_attention_fusion: true  # Instead of concatenation

training:
  epochs: 20
  batch_size: 1024  # Larger batch
  learning_rate: 3e-5  # Lower for large model
  optimizer: AdamW
  weight_decay: 0.05
  lr_schedule: "cosine_with_warmup"
  warmup_ratio: 0.1

  label_smoothing: 0.1
  gradient_clip: 0.5  # Tighter for stability
```

### Loss Function

**Soft target cross-entropy** (VQA-specific):
```python
def vqa_loss(logits, answer_ids, answer_scores):
    """
    Args:
        logits: [B, num_answers] predicted scores
        answer_ids: [B, 10] answer indices (up to 10 annotators)
        answer_scores: [B, 10] soft scores (confidence weights)
    """
    # Build soft target distribution
    targets = torch.zeros_like(logits)
    for i in range(10):  # Max 10 annotators
        valid = answer_ids[:, i] != -1
        targets[valid, answer_ids[valid, i]] += answer_scores[valid, i]

    # Normalize to sum to 1
    targets = targets / targets.sum(dim=-1, keepdim=True)

    # Soft cross-entropy
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss
```

### Data Augmentation

```python
# Image augmentation (train only)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

# Val/test: no augmentation
val_transform = transforms.Compose([
    transforms.Resize(336),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
```

**Question augmentation** (optional):
- Paraphrasing with GPT-4
- Back-translation (En → De → En)
- Synonym replacement (WordNet)

## Advanced Techniques

### 1. Attention-Based Fusion

Instead of simple concatenation:

```python
class AttentionFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=12, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, vision_feats, text_feats):
        # vision_feats: [B, 197, 768] (ViT patches)
        # text_feats: [B, L, 768] (BERT tokens)

        # Text attends to vision
        attn_out, _ = self.cross_attn(text_feats, vision_feats, vision_feats)
        fused = self.norm(text_feats + attn_out)  # Residual
        return fused.mean(dim=1)  # Pool to [B, 768]
```

### 2. Pre-training on VQA-Related Datasets

**Recommended pre-training order**:
1. Visual Genome QA (1.7M QA pairs) - 5 epochs
2. GQA (22M QA pairs) - 1 epoch
3. VQAv2 (443k questions) - 15 epochs

**Transfer protocol**:
```python
# Stage 1: Pre-train on VG+GQA
model.train_on_datasets(['visual_genome', 'gqa'], epochs=5)

# Stage 2: Fine-tune on VQAv2
model.load_checkpoint('vg_gqa_pretrained.pt')
model.train_on_dataset('vqav2', epochs=15, lr=1e-4)
```

### 3. Ensemble Methods

**Simple ensemble** (70% → 72% accuracy boost):
```python
models = [
    VQAModel(vision='clip-vit-L', text='bert-base'),
    VQAModel(vision='clip-vit-L', text='roberta-base'),
    VQAModel(vision='clip-vit-L', text='deberta-base'),
]

# Average logits
ensemble_logits = torch.stack([m(image, question) for m in models]).mean(dim=0)
```

**Weighted ensemble** (best):
```python
# Learn ensemble weights on validation set
weights = [0.4, 0.35, 0.25]  # Optimized
ensemble_logits = sum(w * m(x) for w, m in zip(weights, models))
```

## Evaluation Protocol

### VQA Accuracy Metric

```python
def vqa_accuracy(predicted_answer, ground_truth_answers):
    """
    VQA soft accuracy: min(#humans_said_answer / 3, 1)
    """
    # Count how many annotators gave this answer
    count = sum(1 for gt in ground_truth_answers
                if normalize_answer(predicted_answer) == normalize_answer(gt))

    # Soft accuracy
    return min(count / 3.0, 1.0)

def normalize_answer(answer):
    # Normalize: lowercase, remove articles, punctuation
    answer = answer.lower().strip()
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer.strip()
```

### Inference

```python
@torch.no_grad()
def predict(model, image, question):
    model.eval()

    # Encode
    image_input = preprocess(image).unsqueeze(0).to(device)
    question_input = tokenizer(question, return_tensors='pt').to(device)

    # Forward
    logits = model(image_input, question_input['input_ids'])

    # Get top-k answers
    probs = F.softmax(logits, dim=-1)
    top_k_prob, top_k_idx = probs.topk(k=5, dim=-1)

    # Decode
    answers = [idx_to_answer[idx.item()] for idx in top_k_idx[0]]
    return answers[0]  # Return top-1
```

## Performance Benchmarks

### Baseline Models (Test-dev)

| Model | Vision | Text | Accuracy |
|-------|--------|------|----------|
| Simple Concat | CLIP ViT-L | BERT-base | 68.5% |
| + Better fusion | CLIP ViT-L | BERT-base | 71.2% |
| + Larger text | CLIP ViT-L | RoBERTa-large | 73.8% |
| + Pre-training | CLIP ViT-L | RoBERTa-large | 75.1% |

### SOTA Models (Test-dev)

| Model | Accuracy | Year |
|-------|----------|------|
| BLIP-2 (FlanT5-XXL) | 82.19% | 2023 |
| BEiT-3 | 84.19% | 2022 |
| LLaVA-1.5 (13B) | 80.0% | 2024 |
| Qwen-VL (7B) | 79.5% | 2023 |

## Common Issues

### Issue 1: Low Accuracy on Number Questions

**Problem**: Model guesses common numbers (1, 2, 3)

**Solution**:
```python
# Separate head for number prediction
class VQAModelWithNumHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = VQAModel()
        self.question_type_classifier = nn.Linear(768, 3)  # yes/no, number, other
        self.number_head = nn.Linear(512, 20)  # Predict 0-19

    def forward(self, image, question):
        # Detect question type
        q_type_logits = self.question_type_classifier(question_feats)
        q_type = q_type_logits.argmax(dim=-1)

        # Route to appropriate head
        if q_type == 1:  # Number question
            return self.number_head(fused_feats)
        else:
            return self.base_model(image, question)
```

### Issue 2: Bias Toward Common Answers

**Problem**: Model always predicts "yes", "2", "white"

**Solution**:
- Class-balanced sampling during training
- Focal loss for rare answers
- Data augmentation with counter-examples

```python
# Focal loss for handling class imbalance
def focal_loss(logits, targets, gamma=2.0):
    probs = F.softmax(logits, dim=-1)
    ce_loss = F.cross_entropy(logits, targets, reduction='none')

    # Down-weight easy examples
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - pt) ** gamma

    loss = (focal_weight * ce_loss).mean()
    return loss
```

### Issue 3: Overfitting

**Symptoms**: Train acc 95%, val acc 72%

**Solutions**:
- Increase dropout (0.3 → 0.5)
- Add weight decay (0 → 0.05)
- Reduce model capacity
- More data augmentation
- Early stopping based on val acc

---

**Sources:**
- Goyal, Y. et al. (2017). "Making the V in VQA Matter: Elevating the Role of Image Understanding"
- Anderson, P. et al. (2018). "Bottom-Up and Top-Down Attention for Image Captioning and VQA"
- Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training"
- VQAv2 Dataset: https://visualqa.org/
