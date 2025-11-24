# Embedding Fine-Tuning for Domain Adaptation

## Overview

Fine-tuning embedding models for specific domains dramatically improves retrieval quality and semantic understanding. While general-purpose embeddings work well for broad tasks, domain-specific fine-tuning adapts models to specialized vocabularies, relationships, and semantic patterns in fields like medicine, law, finance, or technical documentation.

**Key benefits:**
- **Domain gap reduction**: Adapt models to specialized terminology and concepts
- **Improved retrieval accuracy**: Better performance on domain-specific queries
- **Semantic alignment**: Learn domain-specific relationships between concepts
- **Resource efficiency**: Smaller fine-tuned models can outperform larger general models

From [Voyage AI Legal Edition](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/) (accessed 2025-02-02):
- Domain-specific embeddings show significant retrieval quality improvements
- Legal domain example: 15-20% improvement in retrieval accuracy vs general models
- Specialized models capture domain relationships general models miss

From [Do we need domain-specific embedding models?](https://arxiv.org/html/2409.18511v3) (arXiv:2409.18511, accessed 2025-02-02):
- State-of-the-art embeddings struggle with domain-specific linguistic patterns
- Domain fine-tuning provides compelling performance improvements
- Medical and legal domains show largest gaps between general and specialized models

## Why Fine-Tune Embeddings

### Domain Gap Problem

**General embeddings limitations:**
```
Medical query: "acute MI with ST elevation"
General model embedding: Groups with "heart attack" (correct)
                        Also groups with "myocardial ischemia" (related but distinct)
                        Misses: STEMI vs NSTEMI distinction (critical clinical difference)

Domain-tuned model: Correctly distinguishes STEMI vs NSTEMI
                   Understands "acute MI" = "myocardial infarction"
                   Preserves clinical precision while capturing synonyms
```

**Domain-specific challenges:**
- **Specialized vocabulary**: Terms with different meanings (e.g., "cell" in biology vs. law)
- **Acronym ambiguity**: "AI" = Artificial Intelligence vs. Aortic Insufficiency
- **Relationship patterns**: Domain-specific concept hierarchies
- **Semantic nuances**: Subtle distinctions critical in specialized fields

### Performance Improvements

From [Fine-tuning Embeddings for Specific Domains](https://blog.gopenai.com/fine-tuning-embeddings-for-specific-domains-a-comprehensive-guide-5e4298b42185) (accessed 2025-02-02):

**Typical improvements after domain fine-tuning:**
- **Retrieval accuracy**: 15-30% improvement in domain-specific recall@10
- **Semantic precision**: Better distinction between similar domain concepts
- **Query understanding**: Improved handling of domain jargon and abbreviations
- **Cross-lingual**: Better performance on domain-specific multilingual tasks

**Use cases:**
- Medical literature search and diagnosis support
- Legal document retrieval and case law research
- Scientific paper recommendation
- Technical documentation search
- Customer support knowledge bases

## Training Objectives

### 1. Contrastive Loss

**Core principle**: Pull similar embeddings together, push dissimilar apart

From [SetFit Conceptual Guide](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit) (accessed 2025-02-02):

**How contrastive learning works:**
```python
# Positive pairs: Same class/similar meaning
pair_1 = ("The movie was awesome", "I loved it")  # label=1
pair_2 = ("Acute myocardial infarction", "Heart attack")  # label=1

# Negative pairs: Different classes/meanings
pair_3 = ("The movie was awesome", "It was disappointing")  # label=0
pair_4 = ("Acute MI", "Chronic heart failure")  # label=0

# Loss function
if label == 1:
    loss = max(0, distance(emb_a, emb_b) - margin)  # Minimize distance
else:
    loss = max(0, margin - distance(emb_a, emb_b))  # Maximize distance
```

**Contrastive loss formula:**
```
L_contrastive = (1 - y) * (1/2) * D²
              + y * (1/2) * max(0, margin - D)²

where:
  y = label (1 for similar, 0 for dissimilar)
  D = distance between embeddings
  margin = minimum distance for negative pairs
```

**Advantages:**
- Works with pairs (easier to create training data)
- Scales exponentially: 8 samples per class → 92 unique pairs
- Effective for few-shot learning scenarios

From [Sentence Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) (accessed 2025-02-02):
- Contrastive learning is the foundation of SetFit fine-tuning
- Creates positive pairs from same-class sentences
- Negative pairs from different classes
- Enables training with minimal labeled data (8-16 examples per class)

### 2. Triplet Loss

**Core principle**: Anchor closer to positive than to negative by margin

From [Sentence Transformers Loss Functions](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) (accessed 2025-02-02):

**Triplet structure:**
```python
# Triplet: (anchor, positive, negative)
triplet_1 = (
    "Patient presents with chest pain",  # anchor
    "Substernal chest discomfort",       # positive (similar)
    "Headache and dizziness"             # negative (dissimilar)
)

# Loss function
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

**Triplet loss formula:**
```
L_triplet = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + α)

where:
  f(x) = embedding function
  a = anchor sample
  p = positive sample (same class as anchor)
  n = negative sample (different class)
  α = margin (e.g., 0.2)
```

**Triplet mining strategies:**
- **Batch-all triplets**: Use all valid triplets in batch
- **Hard negatives**: Select most challenging negatives
- **Semi-hard negatives**: Negatives closer than positive but beyond margin

**Advantages over contrastive:**
- Faster convergence (direct margin optimization)
- More informative gradients
- Better for ranking tasks

### 3. Multiple Negatives Ranking Loss

**Core principle**: Treat batch as in-batch negatives for efficiency

From [Sentence Transformers Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) (accessed 2025-02-02):

**In-batch negatives approach:**
```python
# Batch of query-document pairs
batch = [
    (query_1, doc_1_positive),  # Positive pair
    (query_2, doc_2_positive),  # Positive pair
    (query_3, doc_3_positive),  # Positive pair
]

# For query_1:
#   - Positive: doc_1
#   - Negatives: doc_2, doc_3 (in-batch negatives)

# Loss: query_1 should be closer to doc_1 than to doc_2, doc_3
```

**MNR Loss formula:**
```
L_MNR = -log( exp(sim(q, d⁺) / τ) / Σ exp(sim(q, dⁱ) / τ) )

where:
  q = query embedding
  d⁺ = positive document embedding
  dⁱ = all documents in batch (positive + negatives)
  τ = temperature parameter
  sim = similarity function (cosine, dot product)
```

**Advantages:**
- Very efficient: Uses batch structure as negatives
- Scales with batch size (larger batch = more negatives)
- State-of-the-art results on retrieval tasks
- No need to explicitly mine negatives

**From research:**
- Used in DPR (Dense Passage Retrieval)
- Foundation of modern bi-encoder training
- Effective with batch sizes 32-256

## Dataset Preparation

### Creating Training Pairs

**Positive pair sources:**
- **Paraphrases**: Different phrasings of same concept
- **Synonymous terms**: Domain-specific synonym pairs
- **Query-document**: Relevant query-answer pairs
- **Duplicate detection**: Semantically equivalent texts
- **Translation pairs**: Multilingual domain content

**Example medical pairs:**
```python
positive_pairs = [
    # Synonym pairs
    ("myocardial infarction", "heart attack"),
    ("cerebrovascular accident", "stroke"),

    # Paraphrase pairs
    ("patient presents with acute chest pain",
     "individual experiencing sudden thoracic discomfort"),

    # Abbreviation-expansion
    ("COPD exacerbation", "chronic obstructive pulmonary disease flare-up"),

    # Query-answer
    ("What is the treatment for pneumonia?",
     "Antibiotic therapy is the primary treatment for bacterial pneumonia")
]
```

### Hard Negatives Mining

**Critical for effective training**: Easy negatives provide weak signal

From [What is hard negative mining?](https://zilliz.com/ai-faq/what-is-hard-negative-mining-and-how-does-it-improve-embeddings) (accessed 2025-02-02):

**Hard negative definition:**
- Samples that are semantically similar but belong to different classes
- Difficult for model to distinguish from positives
- Provide stronger training signal than random negatives

**Mining strategies:**

**1. BM25-based mining:**
```python
# Retrieve top-K candidates using BM25
candidates = bm25.retrieve(query, k=100)

# Filter out ground truth positive
negatives = [c for c in candidates if c != positive_doc]

# Select top-N as hard negatives (most lexically similar)
hard_negatives = negatives[:5]
```

**2. Model-based mining:**
```python
# Use current model to find most confusing negatives
embeddings = model.encode(candidate_pool)
query_emb = model.encode(query)

# Find most similar non-relevant documents
similarities = cosine_similarity(query_emb, embeddings)
sorted_idx = np.argsort(similarities)[::-1]

# Select high-similarity but incorrect documents
hard_negatives = [candidates[i] for i in sorted_idx[:5]
                  if candidates[i] not in positives]
```

**3. Cross-encoder reranking:**
```python
# Use cross-encoder to score candidates
scores = cross_encoder.predict([(query, cand) for cand in candidates])

# Select high-scoring but incorrect candidates
hard_negatives = select_top_scoring_negatives(candidates, scores, n=5)
```

From [NV-Retriever: Improving text embeddings with effective hard-negative mining](https://arxiv.org/pdf/2407.15831) (accessed 2025-02-02):
- **Positive-aware hard negatives**: Mine negatives that are close to positive examples
- **Dynamic mining**: Update hard negatives during training (not static)
- **Balanced sampling**: Mix hard negatives with medium-difficulty negatives
- **Result**: 10-15% improvement in retrieval metrics

**Hard negative quality:**
```python
# Good hard negative characteristics
hard_negative = {
    "lexical_similarity": "high",  # Share keywords with query
    "semantic_similarity": "medium-high",  # Topically related
    "relevance": "low",  # Not actually relevant
    "difficulty": "challenging but learnable"
}

# Example
query = "treatment for bacterial pneumonia"
hard_negative = "diagnosis of viral pneumonia"  # Similar topic, different intent
easy_negative = "car maintenance schedule"  # Too easy, weak signal
```

### Data Augmentation

**Augmentation techniques:**

**1. Back-translation:**
```python
# Translate to intermediate language and back
original = "Acute myocardial infarction with ST elevation"
translated = translate(original, "en" -> "fr" -> "en")
augmented = "Acute heart attack with ST segment elevation"
# Creates paraphrase pair
```

**2. Synonym replacement:**
```python
# Replace domain terms with synonyms
original = "Patient exhibits hypertension"
augmented = "Patient shows high blood pressure"
```

**3. Contextual word substitution:**
```python
# Use BERT/LLM to suggest contextual replacements
original = "Administer 5mg morphine for pain control"
augmented = "Give 5mg morphine for analgesia"
```

**4. Query generation:**
```python
# Generate synthetic queries for documents
document = "Treatment guidelines for Type 2 diabetes..."
generated_queries = [
    "How to treat type 2 diabetes?",
    "What are the treatment options for T2DM?",
    "Type 2 diabetes management protocols"
]
```

### Dataset Format

**Sentence Transformers format:**
```python
from sentence_transformers import InputExample

# Contrastive learning pairs
train_examples = [
    InputExample(texts=["text1", "text2"], label=1.0),  # Similar
    InputExample(texts=["text1", "text3"], label=0.0),  # Dissimilar
]

# Triplet format
train_examples = [
    InputExample(texts=["anchor", "positive", "negative"])
]

# MNR Loss format (query, positive document)
train_examples = [
    InputExample(texts=["query", "relevant_doc"])
]
```

**Dataset size recommendations:**
- **Minimum**: 1,000-5,000 pairs for domain adaptation
- **Optimal**: 10,000-100,000 pairs for robust fine-tuning
- **Quality over quantity**: 5,000 high-quality pairs > 50,000 noisy pairs

## Fine-Tuning Strategies

### Full Fine-Tuning

**Approach**: Update all model parameters

From [Sentence Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) (accessed 2025-02-02):

**Implementation:**
```python
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Prepare training data
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune all parameters
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5}
)
```

**Pros:**
- Maximum adaptation to domain
- Best final performance
- Learns domain-specific representations fully

**Cons:**
- Requires more training data
- Risk of catastrophic forgetting
- Longer training time
- Higher compute requirements

### Adapter Layers

**Approach**: Add trainable layers, freeze base model

From [Embedding Compression for Teacher-to-Student Knowledge Transfer](https://arxiv.org/abs/2402.06761) (arXiv:2402.06761, accessed 2025-02-02):

**Architecture:**
```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Bottleneck adapter
        adapter_out = self.down_project(x)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.up_project(adapter_out)
        # Residual connection
        return x + adapter_out

# Add adapters to frozen model
model = SentenceTransformer('base-model')
for layer in model._modules['0'].auto_model.encoder.layer:
    layer.output = AdapterWrapper(layer.output, adapter_size=64)
```

**Pros:**
- Parameter efficient: Train 1-5% of parameters
- Preserves general knowledge
- Faster training
- Easy to swap adapters for different domains

**Cons:**
- Slightly lower performance than full fine-tuning
- Additional inference latency (small)

### LoRA for Embeddings

**Approach**: Low-rank adaptation of weight matrices

**Implementation:**
```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["query", "value"],  # Attention matrices
    lora_dropout=0.1,
    bias="none"
)

# Apply LoRA to sentence transformer
base_model = SentenceTransformer('base-model')
model = get_peft_model(base_model, lora_config)

# Train only LoRA parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
# Typically 1-2% of base model size
```

**LoRA benefits:**
- Trains 1-2% of parameters
- Minimal performance degradation vs full fine-tuning
- Multiple LoRA modules for different domains
- Memory efficient training

### Distillation-Based Fine-Tuning

**Approach**: Transfer knowledge from larger teacher to smaller student

From [Embedding Compression for Teacher-to-Student Knowledge Transfer](https://arxiv.org/abs/2402.06761) (accessed 2025-02-02):

**Distillation strategy:**
```python
# Teacher model (large, general-purpose)
teacher = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Student model (smaller, domain-specific)
student = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Distillation loss
def distillation_loss(student_emb, teacher_emb, temperature=2.0):
    # Soften embeddings with temperature
    student_logits = student_emb / temperature
    teacher_logits = teacher_emb / temperature

    # MSE or cosine loss
    loss = F.mse_loss(student_logits, teacher_logits)
    return loss

# Train student to mimic teacher on domain data
for batch in domain_dataloader:
    teacher_emb = teacher.encode(batch, convert_to_tensor=True)
    student_emb = student.encode(batch, convert_to_tensor=True)
    loss = distillation_loss(student_emb, teacher_emb)
    loss.backward()
```

**Benefits:**
- Compress large model knowledge into smaller model
- Student optimized for specific domain
- Faster inference with student model

## Implementation Guide

### Using Sentence Transformers

From [Sentence Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) (accessed 2025-02-02):

**Complete training pipeline:**
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# 1. Prepare training data
train_examples = [
    InputExample(texts=['query1', 'doc1']),
    InputExample(texts=['query2', 'doc2']),
    # ... more examples
]

# 2. Prepare evaluation data
eval_examples = [
    InputExample(texts=['eval_q1', 'eval_d1'], label=1.0),
    InputExample(texts=['eval_q2', 'eval_d2'], label=0.0),
]

# 3. Load base model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 4. Create data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 5. Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# 6. Create evaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,
    name='eval'
)

# 7. Fine-tune model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=500,
    warmup_steps=100,
    output_path='./fine-tuned-model',
    optimizer_params={'lr': 2e-5},
    save_best_model=True
)

# 8. Load best model
model = SentenceTransformer('./fine-tuned-model')
```

### Using HuggingFace Transformers

**Fine-tuning with Trainer:**
```python
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch
import torch.nn.functional as F

class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
)

# Initialize model
model = EmbeddingModel('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Hyperparameter Tuning

**Critical hyperparameters:**

**Learning rate:**
```python
# Recommended ranges
full_finetuning_lr = 2e-5  # Conservative, preserves pre-training
adapter_lr = 1e-4          # Higher for adapter layers only
lora_lr = 3e-4            # Highest for low-rank updates

# Learning rate schedule
warmup_ratio = 0.1  # 10% of training steps
scheduler = 'linear'  # or 'cosine'
```

**Batch size:**
```python
# For MNR Loss, larger batch = more negatives = better training
batch_size_recommendations = {
    "small_gpu_8gb": 16,
    "medium_gpu_16gb": 32,
    "large_gpu_24gb": 64,
    "multi_gpu": 128  # With gradient accumulation
}

# Gradient accumulation for effective large batch
effective_batch_size = batch_size * gradient_accumulation_steps
```

**Training epochs:**
```python
# Domain adaptation typically requires fewer epochs
epochs_by_scenario = {
    "light_adaptation": 1-2,  # Similar domain
    "moderate_adaptation": 3-5,  # Different domain
    "strong_adaptation": 5-10,  # Very different domain
}

# Monitor validation loss to avoid overfitting
```

**Temperature for contrastive:**
```python
# Controls softness of similarity distribution
temperature = 0.05  # Lower = harder negatives emphasized
# Range: 0.01 (very hard) to 0.1 (softer)
```

## Evaluation Metrics

### Retrieval Quality

**Recall@K:**
```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    """
    Fraction of relevant docs in top-K retrievals
    """
    top_k = retrieved_docs[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_docs))
    return relevant_retrieved / len(relevant_docs)

# Typical evaluation
recall_1 = recall_at_k(results, ground_truth, k=1)
recall_5 = recall_at_k(results, ground_truth, k=5)
recall_10 = recall_at_k(results, ground_truth, k=10)
```

**Mean Reciprocal Rank (MRR):**
```python
def mean_reciprocal_rank(queries_results):
    """
    Average of reciprocal ranks of first relevant result
    """
    reciprocal_ranks = []
    for query, (retrieved, relevant) in queries_results.items():
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**NDCG (Normalized Discounted Cumulative Gain):**
```python
def ndcg_at_k(retrieved_docs, relevance_scores, k):
    """
    Accounts for ranking quality with graded relevance
    """
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
    return dcg / idcg if idcg > 0 else 0.0
```

### Embedding Quality

**Embedding similarity distribution:**
```python
def evaluate_embedding_quality(model, positive_pairs, negative_pairs):
    """
    Check separation between positive and negative similarities
    """
    # Compute similarities
    pos_sims = [cosine_similarity(
        model.encode(p1), model.encode(p2)
    ) for p1, p2 in positive_pairs]

    neg_sims = [cosine_similarity(
        model.encode(n1), model.encode(n2)
    ) for n1, n2 in negative_pairs]

    # Ideal: high positive similarity, low negative similarity
    metrics = {
        'mean_pos_sim': np.mean(pos_sims),
        'mean_neg_sim': np.mean(neg_sims),
        'separation': np.mean(pos_sims) - np.mean(neg_sims),
        'overlap': len([s for s in neg_sims if s > np.mean(pos_sims)]) / len(neg_sims)
    }
    return metrics
```

**Embedding space visualization:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings(model, texts, labels):
    """
    Project embeddings to 2D for visualization
    """
    embeddings = model.encode(texts)

    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot by label
    for label in set(labels):
        mask = [l == label for l in labels]
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label)

    plt.legend()
    plt.title('Embedding Space Visualization')
    plt.show()
```

### Domain Adaptation Metrics

**Before/after comparison:**
```python
# Evaluate on domain-specific test set
def compare_models(base_model, finetuned_model, test_queries):
    results = {
        'base': evaluate_retrieval(base_model, test_queries),
        'finetuned': evaluate_retrieval(finetuned_model, test_queries)
    }

    improvement = {
        metric: (results['finetuned'][metric] - results['base'][metric]) / results['base'][metric] * 100
        for metric in results['base']
    }

    print(f"Recall@10 improvement: {improvement['recall@10']:.1f}%")
    print(f"MRR improvement: {improvement['mrr']:.1f}%")

    return results, improvement
```

## Production Considerations

### Model Deployment

**Save fine-tuned model:**
```python
# Sentence Transformers format
model.save('fine-tuned-medical-embeddings')

# HuggingFace Hub
model.push_to_hub('username/medical-embeddings-v1')

# ONNX export for production
from optimum.onnxruntime import ORTModelForFeatureExtraction

ort_model = ORTModelForFeatureExtraction.from_pretrained(
    'fine-tuned-medical-embeddings',
    export=True
)
ort_model.save_pretrained('medical-embeddings-onnx')
```

**Quantization for deployment:**
```python
# 8-bit quantization (2-4x smaller, minimal accuracy loss)
from sentence_transformers.quantization import quantize_embeddings

embeddings = model.encode(documents)
quantized = quantize_embeddings(embeddings, precision='int8')

# Store quantized embeddings
np.save('embeddings_int8.npy', quantized)
```

### Monitoring and Maintenance

**Embedding drift detection:**
```python
def detect_embedding_drift(model, reference_queries, current_queries):
    """
    Monitor if embedding distribution shifts over time
    """
    ref_embeddings = model.encode(reference_queries)
    curr_embeddings = model.encode(current_queries)

    # Compare distributions
    from scipy.stats import ks_2samp

    drift_scores = []
    for dim in range(ref_embeddings.shape[1]):
        stat, pvalue = ks_2samp(ref_embeddings[:, dim], curr_embeddings[:, dim])
        drift_scores.append(pvalue)

    # Alert if significant drift
    significant_drift = sum(p < 0.05 for p in drift_scores) / len(drift_scores)
    if significant_drift > 0.1:  # More than 10% dimensions drifting
        print("Warning: Significant embedding drift detected")
        print(f"Percentage of drifting dimensions: {significant_drift*100:.1f}%")

    return drift_scores
```

**Continuous evaluation:**
```python
# Track retrieval quality over time
def monitor_retrieval_quality(model, test_set, window_size=100):
    """
    Rolling window evaluation for production monitoring
    """
    metrics_history = []

    for i in range(0, len(test_set), window_size):
        window = test_set[i:i+window_size]
        metrics = evaluate_retrieval(model, window)
        metrics['timestamp'] = datetime.now()
        metrics_history.append(metrics)

    # Alert on degradation
    recent_avg = np.mean([m['recall@10'] for m in metrics_history[-5:]])
    baseline_avg = np.mean([m['recall@10'] for m in metrics_history[:5]])

    if recent_avg < baseline_avg * 0.9:  # 10% degradation
        print(f"Alert: Recall@10 degraded from {baseline_avg:.3f} to {recent_avg:.3f}")

    return metrics_history
```

### Re-training Strategy

**When to re-train:**
- New domain vocabulary emerges (medical: new drug names, procedures)
- Query patterns shift significantly
- Retrieval quality degrades beyond threshold
- New data sources added to corpus

**Incremental fine-tuning:**
```python
# Start from previous fine-tuned model
model = SentenceTransformer('previous-finetuned-model')

# Fine-tune on new domain data
new_train_data = load_new_domain_data()
model.fit(
    train_objectives=[(DataLoader(new_train_data), loss)],
    epochs=2,  # Fewer epochs for incremental update
    warmup_steps=50,
    optimizer_params={'lr': 1e-5}  # Lower LR to preserve existing knowledge
)
```

## Sources

**Web Research:**
- [Domain-Specific Embeddings: Legal Edition - Voyage AI](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/) - Domain-specific embedding benefits and case studies (accessed 2025-02-02)
- [Do we need domain-specific embedding models?](https://arxiv.org/html/2409.18511v3) - arXiv:2409.18511, Analysis of domain-specific vs general embeddings (accessed 2025-02-02)
- [Fine-tuning Embeddings for Specific Domains](https://blog.gopenai.com/fine-tuning-embeddings-for-specific-domains-a-comprehensive-guide-5e4298b42185) - Comprehensive domain adaptation guide (accessed 2025-02-02)
- [SetFit Conceptual Guide](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit) - Contrastive learning and few-shot fine-tuning (accessed 2025-02-02)
- [Sentence Transformers Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) - Training objectives and best practices (accessed 2025-02-02)
- [Sentence Transformers Loss Functions](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) - Detailed loss function documentation (accessed 2025-02-02)
- [Embedding Compression for Teacher-to-Student Knowledge Transfer](https://arxiv.org/abs/2402.06761) - arXiv:2402.06761, Distillation and compression techniques (accessed 2025-02-02)
- [NV-Retriever: Improving text embeddings with effective hard-negative mining](https://arxiv.org/pdf/2407.15831) - Hard negative mining strategies (accessed 2025-02-02)
- [What is hard negative mining?](https://zilliz.com/ai-faq/what-is-hard-negative-mining-and-how-does-it-improve-embeddings) - Hard negative mining fundamentals (accessed 2025-02-02)

**Additional References:**
- [Medical Embedding Models - Hugging Face](https://huggingface.co/mejurix/medical-legal-embedder) - Domain-specific model example
- [Top embedding models on the MTEB leaderboard](https://modal.com/blog/mteb-leaderboard-article) - Embedding model benchmarks
