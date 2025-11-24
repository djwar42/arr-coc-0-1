# VQAv2 Training Protocols

## Overview

VQAv2 (Visual Question Answering version 2.0) is a large-scale benchmark dataset for training and evaluating vision-language models on open-ended visual question answering tasks. The dataset addresses visual priming biases present in v1.0 by ensuring answer distributions are balanced across complementary image pairs.

**Dataset Scale:**
- Training: 82,783 MS COCO images, 443,757 questions, 4,437,570 answers (10 per question)
- Validation: 40,504 MS COCO images, 214,354 questions, 2,143,540 answers (10 per question)
- Test: 81,434 MS COCO images, 447,793 questions (answers withheld for evaluation server)

**Key Characteristics:**
- Open-ended task (no multiple choice)
- Multi-label classification problem (multiple valid answers per question)
- Soft labels with weights based on annotator agreement
- Addresses visual priming bias from v1.0

## VQAv2 Dataset Structure

### Data Organization

From [VQA GitHub API](https://github.com/GT-Vision-Lab/VQA):

**Directory Structure:**
```
./Questions/
  - v2_OpenEnded_mscoco_train2014_questions.json
  - v2_OpenEnded_mscoco_val2014_questions.json
  - v2_OpenEnded_mscoco_test2015_questions.json

./Annotations/
  - v2_mscoco_train2014_annotations.json
  - v2_mscoco_val2014_annotations.json

./Images/
  mscoco/
    - train2014/ (from MS COCO)
    - val2014/ (from MS COCO)
    - test2015/ (from MS COCO)
```

### Question File Format

Questions JSON structure:
```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "Where is he looking?",
  "question_type": "none of the above"
}
```

### Annotation File Format

Annotations JSON structure:
```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question_type": "none of the above",
  "answer_type": "other",
  "answers": [
    {"answer": "down", "answer_confidence": "yes", "answer_id": 1},
    {"answer": "at table", "answer_confidence": "maybe", "answer_id": 2},
    {"answer": "skateboard", "answer_confidence": "yes", "answer_id": 3},
    ...
  ],
  "multiple_choice_answer": "down"
}
```

### Answer Encoding Strategy

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

**Multi-label soft encoding:**
- VQA is inherently ambiguous (subjective answers)
- Each answer gets a weight/score based on annotator agreement
- Most common answer gets weight 1.0, others scaled proportionally

**Implementation:**
```python
# Create label mappings
labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Convert to soft labels
def create_soft_labels(labels, scores, num_classes):
    target = torch.zeros(num_classes)
    for label, score in zip(labels, scores):
        target[label] = score
    return target
```

**Example encoding:**
- Question: "Where is he looking?"
- Answers: {"down": 1.0, "at table": 0.3, "skateboard": 0.3}
- Target vector: [0, 0, ..., 1.0, ..., 0.3, ..., 0.3, ..., 0] (size = vocabulary)

## Data Preprocessing Pipeline

### Image Preprocessing

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

**Standard preprocessing for ViT-based models:**
```python
from transformers import ViltProcessor

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Image processing includes:
# 1. Resize shortest edge to 384px
# 2. Center crop to 384x384
# 3. Normalize with ImageNet stats
# 4. Convert to tensor

images = [Image.open(path) for path in image_paths]
encoding = processor(images, texts, padding="max_length",
                     truncation=True, return_tensors="pt")
```

**Output tensors:**
- `pixel_values`: Shape [B, 3, 384, 384] - normalized image pixels
- `pixel_mask`: Shape [B, 384, 384] - attention mask for images

### Question Preprocessing

**Text tokenization:**
```python
# Questions tokenized with BERT tokenizer (for ViLT)
texts = ["What color is the car?", "How many people?"]

encoding = processor(images, texts,
                     padding="max_length",
                     truncation=True,
                     max_length=40,
                     return_tensors="pt")
```

**Output tensors:**
- `input_ids`: Token IDs from vocabulary
- `attention_mask`: 1 for real tokens, 0 for padding
- `token_type_ids`: Segment IDs (all 0 for single sentence)

### Batch Preprocessing Function

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

```python
import torch

def preprocess_data(examples):
    image_paths = examples['image_id']
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples['question']

    # Process images and questions
    encoding = processor(images, texts,
                         padding="max_length",
                         truncation=True,
                         return_tensors="pt")

    # Squeeze batch dimension for single examples
    for k, v in encoding.items():
        encoding[k] = v.squeeze()

    # Create soft label targets
    targets = []
    for labels, scores in zip(examples['label.ids'],
                              examples['label.weights']):
        target = torch.zeros(len(id2label))
        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)

    encoding["labels"] = targets
    return encoding

# Apply to dataset
processed_dataset = dataset.map(preprocess_data,
                                batched=True,
                                remove_columns=['question', 'image_id', ...])
```

## Training Hyperparameters and Protocols

### ViLT Fine-tuning Configuration

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

**Model architecture:**
- Vision: ViT-B/32 (Vision Transformer)
- Text: BERT tokenizer + embeddings
- Fusion: Cross-attention every 4 layers
- Head: Linear classifier on [CLS] token

**Training hyperparameters:**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="vilt_finetuned_vqav2",
    per_device_train_batch_size=4,      # Small batch due to memory
    per_device_eval_batch_size=8,       # Can be larger for inference
    num_train_epochs=20,                # Longer training for small datasets
    learning_rate=5e-5,                 # Standard for fine-tuning
    warmup_steps=500,                   # Warmup for stability
    weight_decay=0.01,                  # L2 regularization
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,                 # Keep only 2 checkpoints
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    remove_unused_columns=False,        # Keep all columns for VQA
    push_to_hub=True,
)
```

**Batch size considerations:**
- GPU memory constrains batch size (4-8 on single GPU)
- Use gradient accumulation for effective larger batches
- Larger batches (16-32) improve stability if memory allows

**Learning rate schedule:**
- Warmup: Linear warmup for first 500-1000 steps
- Decay: Cosine decay or linear decay to 0
- Peak LR: 5e-5 for ViLT, 1e-4 to 3e-4 for larger models

### OpenFlamingo Training Configuration

From [OFv2 ICL VQA GitHub](https://github.com/GaryJiajia/OFv2_ICL_VQA):

**Model setup:**
```python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)

# Load checkpoint
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "checkpoint.pt"
)
model.load_state_dict(torch.load(checkpoint_path), strict=False)
```

**VQAv2 evaluation parameters:**
```bash
python evaluate_vqa.py \
    --lm_path "path/to/mpt-7b" \
    --checkpoint_path "path/to/checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --device 0 \
    --vqav2_train_image_dir_path "mscoco2014/train2014/" \
    --vqav2_train_questions_json_path "vqav2/v2_OpenEnded_train2014_questions.json" \
    --vqav2_train_annotations_json_path "vqav2/v2_mscoco_train2014_annotations.json" \
    --num_samples 5000 \
    --shots 4 8 16 32 \      # Few-shot in-context learning
    --batch_size 1 \
    --precision fp16 \
    --seed 5
```

**Key differences from ViLT:**
- Uses frozen vision encoder (OpenCLIP ViT-L-14)
- Language model: MPT-7B (7 billion parameters)
- In-context learning paradigm (few-shot, no gradient updates)
- FP16 precision for memory efficiency

### Gradient Accumulation Strategy

For large models on limited hardware:

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,       # Physical batch size
    gradient_accumulation_steps=8,       # Effective batch = 4 * 8 = 32
    ...
)
```

**Memory vs compute tradeoff:**
- Small physical batch (4): Fits in GPU memory
- Large effective batch (32): Better gradient estimates
- Accumulation steps: Trade compute time for memory

## Evaluation Metrics and Protocols

### VQA Accuracy Metric

From [VQA Evaluation Server](https://visualqa.org/evaluation):

**Official VQA accuracy formula:**
```
VQA Accuracy = min(# humans that said answer / 3, 1.0)
```

**Rationale:**
- At least 3 humans must agree for 100% accuracy
- Partial credit for 1-2 agreements (33%, 67%)
- Accounts for answer ambiguity and subjectivity

**Implementation:**
```python
def vqa_accuracy(prediction, ground_truth_answers):
    """
    prediction: single predicted answer string
    ground_truth_answers: list of 10 human answers
    """
    # Count how many humans gave this answer
    matches = sum([1 for gt in ground_truth_answers
                   if gt == prediction])
    # Apply VQA accuracy formula
    return min(matches / 3.0, 1.0)

# Average over all questions
total_accuracy = sum([vqa_accuracy(pred, gt)
                     for pred, gt in zip(predictions, ground_truths)])
average_accuracy = total_accuracy / len(predictions)
```

### Test-Dev vs Test-Standard

From [VQA Evaluation Page](https://visualqa.org/evaluation):

**Test-dev split:**
- Subset of test set for development
- Limited submissions per day
- Leaderboard with live results
- Use for hyperparameter tuning

**Test-standard split:**
- Full test set for final evaluation
- Very limited submissions
- Official benchmark results
- Use only for final model

**Submission format:**
```json
[
  {
    "question_id": 262148000,
    "answer": "down"
  },
  {
    "question_id": 262148001,
    "answer": "blue"
  }
]
```

### Evaluation Best Practices

From [VQA Evaluation Tools](https://github.com/GT-Vision-Lab/VQA):

**1. Answer normalization:**
```python
import re

def normalize_answer(answer):
    # Convert to lowercase
    answer = answer.lower()
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer
```

**2. Number handling:**
- "5" and "five" should match
- Implement number-to-word and word-to-number conversion
- Handle approximate numbers ("about 10" vs "10")

**3. Synonym matching:**
- "car" vs "automobile"
- "couch" vs "sofa"
- Use WordNet or custom synonym dictionary

**4. Answer type analysis:**
```python
# Evaluate separately by answer type
answer_types = ["yes/no", "number", "other"]
for ans_type in answer_types:
    subset = [ex for ex in examples if ex['answer_type'] == ans_type]
    accuracy = compute_vqa_accuracy(subset)
    print(f"{ans_type}: {accuracy:.2f}%")
```

## Common Training Challenges and Solutions

### Challenge 1: Answer Vocabulary Size

**Problem:** VQAv2 has 3,129 unique answers in common split

**Solutions:**
- Use top-K answers (K=3129 for VQAv2)
- Treat as classification problem (not generation)
- For generation models: Use constrained decoding

### Challenge 2: Class Imbalance

**Problem:** Answer distribution is highly skewed

From web research analysis:

**Distribution characteristics:**
- Top 10 answers cover ~30% of dataset
- Long tail of rare answers
- "Yes/no" questions dominate certain categories

**Mitigation strategies:**
```python
# Weighted loss function
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

# Apply in loss
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights)
)
```

### Challenge 3: Visual Priming Bias

**Problem:** Models may answer based on image statistics alone

**VQAv2 solution:**
- Complementary image pairs with different answers
- Forces model to read both image AND question
- Reduces language priors

**Validation:**
```python
# Measure blind model accuracy (image-only, no question)
blind_accuracy = evaluate_image_only_baseline()
# Should be close to random (< 50% on VQAv2)
```

### Challenge 4: Memory Constraints

**Problem:** Large images + long questions + big models = OOM

**Solutions:**

**1. Gradient checkpointing:**
```python
from transformers import ViltForQuestionAnswering

model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-mlm",
    gradient_checkpointing=True
)
```

**2. Mixed precision training:**
```python
training_args = TrainingArguments(
    fp16=True,  # Use FP16 instead of FP32
    ...
)
```

**3. Smaller image resolution:**
```python
# ViLT default: 384x384
# Can reduce to 224x224 for memory savings
# Trade: ~10% accuracy loss, ~40% memory savings
```

**4. Dynamic batching:**
```python
# Group similar-length questions together
# Reduces padding waste
from transformers import DataCollatorWithPadding
collator = DataCollatorWithPadding(tokenizer)
```

### Challenge 5: Overfitting on Small Datasets

**Problem:** Full VQAv2 is large, but custom splits may overfit

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

**Regularization techniques:**
```python
training_args = TrainingArguments(
    weight_decay=0.01,              # L2 regularization
    dropout=0.1,                    # Dropout in classifier head
    warmup_ratio=0.1,               # Warmup prevents early overfitting
    early_stopping_patience=3,      # Stop if val loss doesn't improve
)
```

**Data augmentation:**
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

## Answer Encoding Strategies

### Classification vs Generation

**Classification approach (ViLT, LXMERT):**
- Fixed vocabulary of N answers (N=3129 for VQAv2)
- Softmax over answer candidates
- Fast inference
- Cannot produce novel answers

**Generative approach (BLIP-2, InstructBLIP):**
- Sequence-to-sequence generation
- Can produce any answer
- Slower inference (autoregressive)
- More flexible, handles open vocabulary

### Top-K Answer Selection

From web research and HuggingFace docs:

**Choosing K:**
- K=100: Fast, covers ~60% of validation set
- K=1000: Moderate speed, covers ~85%
- K=3129: Full coverage, slower training

**Creating answer vocabulary:**
```python
from collections import Counter

# Count answer frequencies in training set
answer_counts = Counter()
for annotation in train_annotations:
    for answer in annotation['answers']:
        answer_counts[answer['answer']] += 1

# Select top-K
top_k_answers = [ans for ans, _ in answer_counts.most_common(k=3129)]
answer_to_id = {ans: idx for idx, ans in enumerate(top_k_answers)}
```

### Soft Label Construction

From [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering):

**Why soft labels:**
- Multiple annotators provide different answers
- Captures answer uncertainty and diversity
- Better than arbitrary tie-breaking

**Construction algorithm:**
```python
def create_soft_label(answers_list, answer_vocab):
    """
    answers_list: List of 10 human answers for one question
    answer_vocab: Dictionary mapping answer -> id
    """
    label_vector = np.zeros(len(answer_vocab))

    # Count occurrences
    answer_counts = Counter(answers_list)

    # Normalize by VQA accuracy formula
    for answer, count in answer_counts.items():
        if answer in answer_vocab:
            idx = answer_vocab[answer]
            # VQA accuracy: min(count/3, 1.0)
            label_vector[idx] = min(count / 3.0, 1.0)

    return label_vector
```

**Loss function:**
```python
# Use BCEWithLogitsLoss for multi-label soft targets
criterion = nn.BCEWithLogitsLoss()

# Or KL divergence if treating as probability distribution
criterion = nn.KLDivLoss(reduction='batchmean')
predictions = F.log_softmax(logits, dim=-1)
loss = criterion(predictions, soft_labels)
```

## Sources

**Source Documents:**
- None (web research only)

**Web Research:**

- [HuggingFace VQA Tutorial](https://huggingface.co/docs/transformers/en/tasks/visual_question_answering) - Complete ViLT fine-tuning guide (accessed 2025-01-31)
- [VQA GitHub - GT-Vision-Lab](https://github.com/GT-Vision-Lab/VQA) - Official VQA dataset API and evaluation tools (accessed 2025-01-31)
- [OFv2 ICL VQA GitHub](https://github.com/GaryJiajia/OFv2_ICL_VQA) - OpenFlamingo in-context learning for VQA (accessed 2025-01-31)
- [VQA Official Website](https://visualqa.org/download.html) - Dataset download and evaluation information
- [VQA Evaluation Page](https://visualqa.org/evaluation) - Official evaluation protocol and metrics

**Additional References:**
- HuggingFace Transformers documentation on VQA task
- MS COCO dataset structure and organization
- OpenFlamingo model architecture and training
