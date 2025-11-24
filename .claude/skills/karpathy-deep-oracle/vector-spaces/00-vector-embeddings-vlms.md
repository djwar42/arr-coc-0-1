# Vector Embeddings for Vision-Language Models

## Overview

Vector embeddings are numerical representations of data that capture semantic meaning in a high-dimensional space. For Vision-Language Models (VLMs), embeddings enable the representation of both visual and textual information in a unified vector space, allowing cross-modal understanding and retrieval.

This guide covers multimodal embeddings, focusing on CLIP and similar architectures, embedding dimensions, alignment techniques, and practical implementation.

## Fundamentals of Embeddings

### What are Embeddings?

**Embeddings** are learned numerical representations of data that encode semantic information. Unlike hand-crafted features, embeddings are learned implicitly through model training.

**Key Properties:**
- **Dense vectors**: Typically 256-1024 dimensions
- **Semantic similarity**: Similar concepts have similar embeddings
- **Learned representations**: Derived from training objectives, not manual engineering
- **Task-agnostic**: Can be reused for multiple downstream tasks

**Historical Context:**
- **Word embeddings** (Word2Vec, GloVe): Text-only, ~100-300 dimensions
- **BERT embeddings**: Contextualized text, 768 dimensions
- **Vision embeddings** (ResNet, ViT): Image-only, 512-2048 dimensions
- **Multimodal embeddings** (CLIP, ALIGN): Unified text+image space, 512-768 dimensions

From [Multimodal Embeddings: An Introduction](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/) (accessed 2025-02-02):
> "Embeddings are (useful) numerical representations of data learned implicitly through model training. For example, through learning how to predict text, BERT learned representations of text, which are helpful for many NLP tasks."

## Multimodal Embedding Spaces

### Vision vs Text Embeddings

**Single-Modality Limitations:**
- Text embeddings (BERT): Only understand language
- Image embeddings (ResNet, ViT): Only understand visual content
- Cannot perform cross-modal tasks (image captioning, text-to-image search)

**Multimodal Solution:**
Both text and images are represented as vectors (mathematical objects), making unified representation possible.

From [Towards Data Science](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/):
> "Although text and images may look very different to us, in a neural network, these are represented via the same mathematical object, i.e., a vector. Therefore, in principle, text, images, or any other data modality can be processed by a single model."

### The Shared Embedding Space

**Multimodal embeddings** represent multiple data modalities in the same vector space where similar concepts are co-located regardless of original representation.

**Example:**
- Text: "two dogs running across a frosty field"
- Image: Photo of two dogs running across a frosty field
- Result: Both encoded to similar vectors (high cosine similarity)

**Benefits:**
1. **Cross-modal retrieval**: Search images with text, text with images
2. **Zero-shot transfer**: No task-specific training needed
3. **Unified representation**: Same vector space for all modalities
4. **Semantic understanding**: Captures meaning beyond surface features

### Alignment Mechanism

Multimodal models learn to "speak the same language" by projecting different modalities into a shared vector space.

**Alignment Process:**
1. Separate encoders for each modality (text encoder, image encoder)
2. Project outputs to same dimensionality (e.g., 512D)
3. Train with contrastive loss to align related pairs
4. Result: Semantically similar content has similar embeddings

## CLIP: Contrastive Language-Image Pre-training

### Architecture Overview

**CLIP** (OpenAI, 2021) is the foundational multimodal embedding model.

From [HuggingFace CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip) (accessed 2025-02-02):
> "CLIP uses an image encoder and text encoder to get visual features and text features. Both features are projected to a latent space with the same number of dimensions."

**Components:**
1. **Text Encoder**: 12-layer Transformer
   - Input: Text tokens (max 77 tokens)
   - Output: 512D or 768D embeddings
   - Architecture: GPT-style autoregressive transformer

2. **Image Encoder**: ResNet or Vision Transformer (ViT)
   - **ResNet variants**: ResNet-50, ResNet-101
   - **ViT variants**: ViT-B/32, ViT-B/16, ViT-L/14
   - Input: 224×224 RGB images
   - Output: 512D or 768D embeddings

3. **Projection Layers**: Map encodings to shared embedding space
   - Linear projection to final dimension
   - L2 normalization for cosine similarity

**Model Variants:**

| Model | Image Encoder | Embed Dim | Params | Patch Size |
|-------|---------------|-----------|--------|------------|
| CLIP-ResNet-50 | ResNet-50 | 1024 | 102M | - |
| CLIP-ViT-B/32 | ViT-Base | 512 | 151M | 32×32 |
| CLIP-ViT-B/16 | ViT-Base | 512 | 149M | 16×16 |
| CLIP-ViT-L/14 | ViT-Large | 768 | 428M | 14×14 |

### Contrastive Learning

**Core Principle:** Maximize similarity between positive pairs, minimize similarity between negative pairs.

From [Pinecone CLIP Guide](https://www.pinecone.io/learn/series/image-search/clip/) (accessed 2025-02-02):
> "Contrastive pretraining works by taking a (text, image) pair – where the text describes the image – and learning to encode the pairs as closely as possible in vector space."

**Training Process:**
1. **Batch Construction**: N (image, text) pairs
2. **Encode All**: Create N image embeddings and N text embeddings
3. **Similarity Matrix**: Compute N×N similarities (cosine or dot product)
4. **Positive Pairs**: Diagonal elements (matched pairs)
5. **Negative Pairs**: Off-diagonal elements (mismatched pairs)
6. **Loss Function**: Symmetric cross-entropy across rows and columns

**Mathematical Formulation:**

For batch of N pairs:
- Image embeddings: I = {i₁, i₂, ..., iₙ}
- Text embeddings: T = {t₁, t₂, ..., tₙ}
- Similarity matrix: S[i,j] = cos_sim(iᵢ, tⱼ)

**Contrastive Loss:**
```
L = -1/N Σᵢ log(exp(S[i,i]/τ) / Σⱼ exp(S[i,j]/τ))
```
where τ is temperature parameter (typically 0.07)

**Key Insights:**
- **Symmetric loss**: Applied to both image-to-text and text-to-image
- **In-batch negatives**: All other pairs in batch serve as negatives
- **Large batches**: CLIP trained with 32,768 batch size for diverse negatives
- **No manual labeling**: Pairs curated from image-caption data from web

### Training Data

**Dataset Scale:**
- 400 million (image, text) pairs
- Collected from internet (web scraping)
- Diverse: Multiple languages, domains, concepts
- Natural supervision: Captions are naturally occurring alt-text/descriptions

**Data Quality:**
- No manual annotation required
- Web-scale diversity
- Noisy but effective (model learns to handle noise)
- Covers broad visual and linguistic concepts

## Embedding Dimensions Explained

### Why 512, 768, 1024?

From [Stack Overflow discussion](https://stackoverflow.com/questions/75693493/why-the-text-embedding-or-image-embedding-generated-by-clip-model-is-768) (accessed 2025-02-02):
> "768 comes from the embedding of ViT used by CLIP. In ViT, it transforms the input image of 224×224."

From [LinkedIn discussion](https://www.linkedin.com/posts/tarang-balani_came-across-this-tweet-today-that-said-activity-7368747704908193792) (accessed 2025-02-02):
> "Aligned memory access → faster throughput. That's why you see dimensions like 512, 768, 1024, 2048; all clean multiples of 32 and often close to powers of 2."

**Common Dimensions:**
- **256**: Lightweight models, mobile deployment
- **384**: Smaller ViT models (ViT-Small)
- **512**: CLIP-ViT-B/32, CLIP-ViT-B/16, standard choice
- **768**: BERT-Base, ViT-Large, GPT-2, balanced performance
- **1024**: CLIP-ResNet variants, larger models
- **1536**: OpenAI text-embedding-3-large
- **2048**: Very large vision models

**Design Considerations:**

1. **Hardware Efficiency:**
   - Powers of 2 (256, 512, 1024): Optimal GPU memory alignment
   - Multiples of 32/64: CUDA warp size optimization
   - Faster matrix operations on aligned dimensions

2. **Model Capacity:**
   - Higher dimensions: More expressive representations
   - Diminishing returns: 512-1024 sufficient for most tasks
   - Trade-off: Dimension vs computational cost

3. **Historical Precedent:**
   - BERT established 768 as standard for transformers
   - Vision models adapted similar scales
   - Community convergence on proven dimensions

4. **Downstream Task Performance:**
   - Retrieval: 384-512 often sufficient
   - Fine-grained classification: 768-1024 better
   - Zero-shot tasks: Larger dimensions help

### Dimension Trade-offs

**Lower Dimensions (256-384):**
- ✓ Faster inference
- ✓ Lower memory footprint
- ✓ Better for large-scale retrieval (millions of vectors)
- ✗ Less nuanced representations
- ✗ Lower accuracy on complex tasks

**Medium Dimensions (512-768):**
- ✓ Balanced performance/efficiency
- ✓ Standard in production systems
- ✓ Good for most VLM tasks
- ✓ Wide ecosystem support

**Higher Dimensions (1024-2048):**
- ✓ More expressive representations
- ✓ Better fine-grained understanding
- ✓ State-of-the-art accuracy
- ✗ Slower inference
- ✗ Higher memory/compute cost

## Normalization and Similarity Metrics

### L2 Normalization

**Why Normalize?**

From [Pinecone guide](https://www.pinecone.io/learn/series/image-search/clip/):
> "One important thing to note here is that these embeddings are not normalized. If we plan on using a similarity metric like the dot product, we must normalize the embeddings."

**Unnormalized embeddings:**
- Magnitude varies: min=-1.19, max=4.80 (from CLIP text encoder)
- Different scales for different inputs
- Dot product conflates similarity with magnitude

**L2 Normalization Process:**
```python
import numpy as np

# Calculate L2 norm for each embedding
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Divide each embedding by its norm
normalized = embeddings / norm

# Result: All embeddings have unit length (||v|| = 1)
```

**Properties After Normalization:**
- All vectors have length 1.0
- Lie on unit hypersphere
- Dot product = Cosine similarity
- Range: [-1, 1]

### Similarity Metrics

**1. Cosine Similarity**

Most common for embeddings:
```
cos_sim(u, v) = (u · v) / (||u|| × ||v||)
```

**Properties:**
- Range: [-1, 1]
- Measures angular similarity
- Ignores magnitude
- 1 = identical direction, 0 = orthogonal, -1 = opposite

**Use when:**
- Comparing semantic similarity
- Embeddings not normalized
- Want angle-based similarity

**2. Dot Product Similarity**

For normalized embeddings:
```
dot_sim(u, v) = u · v
```

**Properties:**
- For normalized vectors: dot_sim = cos_sim
- Computationally efficient (no division)
- Range: [-1, 1] (for normalized vectors)

**Use when:**
- Embeddings are L2-normalized
- Maximum computational efficiency needed
- Large-scale retrieval (millions of comparisons)

**3. Euclidean Distance**

L2 distance:
```
euclidean(u, v) = ||u - v|| = sqrt(Σ(uᵢ - vᵢ)²)
```

**Properties:**
- Range: [0, ∞]
- Measures spatial distance
- Considers magnitude
- 0 = identical vectors

**Use when:**
- Magnitude matters
- Clustering algorithms (K-means)
- Anomaly detection

**Equivalence for Normalized Vectors:**

For unit-length vectors:
```
euclidean² = 2(1 - cosine_similarity)
```
So Euclidean distance and cosine similarity are monotonically related.

## Practical Implementation with HuggingFace

### Setup and Installation

```bash
pip install transformers torch pillow datasets
```

### Basic Usage: Text and Image Encoding

From [HuggingFace documentation](https://huggingface.co/docs/transformers/model_doc/clip):

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model and processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load image
image = Image.open("cat.jpg")

# Define text
text = ["a photo of a cat", "a photo of a dog"]

# Process inputs
inputs = processor(
    text=text,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

# Get embeddings
outputs = model(**inputs)
image_embeds = outputs.image_embeds  # Shape: (1, 512)
text_embeds = outputs.text_embeds    # Shape: (2, 512)

# Calculate similarity
logits_per_image = outputs.logits_per_image  # Image-to-text similarity
probs = logits_per_image.softmax(dim=1)      # Convert to probabilities

# Get prediction
predicted_class = text[probs.argmax()]
print(f"Prediction: {predicted_class}, Probability: {probs.max():.4f}")
```

### Zero-Shot Image Classification

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("animal.jpg")

# Define possible classes
classes = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a fish"
]

# Process
inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Get prediction
probs = outputs.logits_per_image.softmax(dim=1)
predicted_idx = probs.argmax()
predicted_class = classes[predicted_idx]

print(f"Predicted: {predicted_class}")
print(f"Confidence: {probs[0][predicted_idx]:.2%}")
```

### Text-to-Image Search

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load multiple images
images = [
    Image.open("cat.jpg"),
    Image.open("dog.jpg"),
    Image.open("bird.jpg")
]

# Search query
query = "a cute puppy"

# Process
inputs = processor(text=query, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Find best match
probs = outputs.logits_per_text.softmax(dim=1)
best_match_idx = probs.argmax()

print(f"Best match: Image {best_match_idx}")
print(f"Similarity score: {probs[0][best_match_idx]:.2%}")

# Display result
images[best_match_idx].show()
```

### Extracting Raw Embeddings

```python
import torch
import numpy as np

# Get text embeddings only
text = ["a cat", "a dog", "a bird"]
text_inputs = processor(text=text, return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)

# Normalize embeddings
text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)

# Get image embeddings only
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
image_inputs = processor(images=images, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)

# Normalize
image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)

# Compute similarity matrix
similarity = (text_features_normalized @ image_features_normalized.T)
# Shape: (3, 10) - 3 texts vs 10 images
```

### Batching for Large Datasets

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imagenet-1k", split="train", streaming=True)

# Create dataloader
def collate_fn(batch):
    images = [item["image"] for item in batch]
    return processor(images=images, return_tensors="pt", padding=True)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Process in batches
all_embeddings = []
for batch in dataloader:
    with torch.no_grad():
        embeddings = model.get_image_features(**batch.to(device))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu())

# Concatenate all embeddings
final_embeddings = torch.cat(all_embeddings, dim=0)
```

## Advanced Techniques

### Temperature Scaling

CLIP uses temperature parameter τ in contrastive loss:
```
similarity = cos_sim(image, text) / τ
```

**Effect of Temperature:**
- **Low τ (0.01)**: Sharp distribution, confident predictions
- **Medium τ (0.07)**: CLIP's default, balanced
- **High τ (0.5)**: Smooth distribution, uncertain predictions

```python
# Adjust temperature at inference
temperature = 0.07
logits = outputs.logits_per_image / temperature
probs = logits.softmax(dim=1)
```

### Ensemble Prompts

Use multiple text prompts per class for robustness:

```python
# Instead of single prompt per class
classes = ["a photo of a cat", "a photo of a dog"]

# Use multiple prompt templates
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "{}"
]

# Generate all combinations
all_prompts = []
class_names = ["cat", "dog"]
for cls in class_names:
    for template in templates:
        all_prompts.append(template.format(cls))

# Process all prompts
inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Average predictions per class
probs = outputs.logits_per_image.softmax(dim=1)
probs_per_class = probs.reshape(len(class_names), len(templates)).mean(dim=1)
```

### Embedding Dimensionality Reduction

For storage/speed optimization:

```python
from sklearn.decomposition import PCA

# Original embeddings: (N, 512)
embeddings = ...  # Your CLIP embeddings

# Reduce to 256 dimensions
pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(embeddings)

# Explains variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

## Other Multimodal Embedding Models

### ALIGN (Google, 2021)

**Key Differences from CLIP:**
- Trained on 1.8 billion noisy image-text pairs (vs CLIP's 400M)
- Uses EfficientNet for image encoder
- Similar contrastive learning approach
- Better performance on some benchmarks due to scale

**Architecture:**
- Text: BERT-Large (24 layers, 1024D)
- Image: EfficientNet-L2
- Embedding: 640D

### BLIP (Salesforce, 2022)

**Innovations:**
- **Unified architecture**: Single model for multiple tasks
- **Captioning and understanding**: Not just retrieval
- **Bootstrap training**: Uses synthetic captions
- **ITC + ITM + LM**: Image-text contrastive, matching, language modeling

**Use Cases:**
- Image captioning
- Visual question answering
- Image-text retrieval

### BLIP-2 (2023)

**Improvements:**
- Q-Former architecture: Bridges frozen image and language models
- More efficient: Trains only Q-Former, freezes vision and LLM
- Better zero-shot performance
- Supports larger language models (OPT, FlanT5)

### ImageBind (Meta, 2023)

**Multimodal Extension:**
- **6 modalities**: Image, text, audio, depth, thermal, IMU
- All aligned to shared embedding space
- Enables cross-modal retrieval across any pair
- 1024D embeddings

### OpenCLIP (Community)

**Open-source Alternative:**
- Reproducible CLIP training
- Multiple model sizes
- Trained on LAION datasets (LAION-400M, LAION-2B)
- Better than original CLIP in many cases

**Models:**
```python
# HuggingFace usage
from transformers import CLIPModel

model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
```

## Embedding Quality and Validation

### Intrinsic Evaluation

**1. Embedding Similarity Distribution**

Check if similar items have high similarity:
```python
import matplotlib.pyplot as plt

# Compute pairwise similarities
similarities = embeddings @ embeddings.T

# Plot distribution
plt.hist(similarities.flatten(), bins=50)
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.title("Embedding Similarity Distribution")
```

**Good distribution:**
- Clear separation between related and unrelated pairs
- Most similarities < 0.5 (dissimilar items)
- Few high similarities (only truly related items)

**2. Embedding Norm Analysis**

```python
norms = np.linalg.norm(embeddings, axis=1)
print(f"Mean norm: {norms.mean():.3f}")
print(f"Std norm: {norms.std():.3f}")

# After normalization, all norms should be 1.0
```

**3. Dimensionality Check**

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(embeddings)

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

# Check intrinsic dimensionality
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95)
print(f"95% variance explained by {n_components_95} components")
```

### Extrinsic Evaluation

**1. Retrieval Accuracy**

```python
# For text-to-image retrieval
def evaluate_retrieval(text_queries, image_embeddings, ground_truth_indices, k=5):
    """
    Args:
        text_queries: Query text embeddings (N, D)
        image_embeddings: Database image embeddings (M, D)
        ground_truth_indices: True matches for each query (N,)
        k: Number of top results to consider
    """
    similarities = text_queries @ image_embeddings.T
    top_k_indices = similarities.argsort(axis=1)[:, -k:][:, ::-1]

    # Calculate recall@k
    recall_at_k = 0
    for i, gt_idx in enumerate(ground_truth_indices):
        if gt_idx in top_k_indices[i]:
            recall_at_k += 1

    recall_at_k /= len(ground_truth_indices)
    return recall_at_k

# Usage
recall_at_5 = evaluate_retrieval(text_emb, image_emb, gt_indices, k=5)
print(f"Recall@5: {recall_at_5:.2%}")
```

**2. Zero-Shot Classification**

```python
from sklearn.metrics import accuracy_score

def zero_shot_accuracy(image_embeddings, text_embeddings, labels):
    """
    Args:
        image_embeddings: Image embeddings (N, D)
        text_embeddings: Class text embeddings (C, D)
        labels: True class indices (N,)
    """
    similarities = image_embeddings @ text_embeddings.T
    predictions = similarities.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    return accuracy
```

## Best Practices

### 1. Always Normalize Embeddings

```python
# CRITICAL: Normalize before similarity computation
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# OR in PyTorch
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
```

### 2. Use Appropriate Batch Sizes

**For inference:**
- GPU memory limited: 32-128 images
- Large models (ViT-L): 16-32 images
- Small models (ViT-B): 64-256 images

```python
# Good practice: Process in batches
def encode_images_batched(images, model, processor, batch_size=32):
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            batch_emb = model.get_image_features(**inputs)
        embeddings.append(batch_emb)
    return torch.cat(embeddings, dim=0)
```

### 3. Cache Embeddings

```python
import pickle

# Compute once
image_embeddings = model.get_image_features(...)

# Save
with open("image_embeddings.pkl", "wb") as f:
    pickle.dump(image_embeddings.cpu().numpy(), f)

# Load
with open("image_embeddings.pkl", "rb") as f:
    cached_embeddings = pickle.load(f)
```

### 4. Prompt Engineering for Zero-Shot

```python
# Bad: Single word
text = ["cat"]

# Better: Natural sentence
text = ["a photo of a cat"]

# Best: Multiple templates
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image showing a {}",
    "a {} in the photo"
]
```

### 5. Monitor Embedding Statistics

```python
def check_embedding_quality(embeddings):
    """Validate embedding quality."""
    # Check shape
    print(f"Shape: {embeddings.shape}")

    # Check for NaN/Inf
    assert not np.isnan(embeddings).any(), "Found NaN values"
    assert not np.isinf(embeddings).any(), "Found Inf values"

    # Check norm (should be ~1 after normalization)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm mean: {norms.mean():.3f}, std: {norms.std():.3f}")

    # Check value range
    print(f"Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
```

## Sources

**Web Research:**
- [CLIP Model Documentation - HuggingFace](https://huggingface.co/docs/transformers/model_doc/clip) (accessed 2025-02-02)
- [Multimodal Embeddings: An Introduction - Towards Data Science](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/) (accessed 2025-02-02)
- [Multi-modal ML with OpenAI's CLIP - Pinecone](https://www.pinecone.io/learn/series/image-search/clip/) (accessed 2025-02-02)
- [Why embedding dimensions come in neat sizes - LinkedIn](https://www.linkedin.com/posts/tarang-balani_came-across-this-tweet-today-that-said-activity-7368747704908193792) (accessed 2025-02-02)
- [Understanding CLIP embedding dimensions - Stack Overflow](https://stackoverflow.com/questions/75693493/why-the-text-embedding-or-image-embedding-generated-by-clip-model-is-768) (accessed 2025-02-02)

**Academic Papers:**
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - OpenAI, 2021
- [ALIGN: Scaling Up Visual and Vision-Language Representation Learning](https://arxiv.org/abs/2102.05918) - Google, 2021
- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) - Salesforce, 2022
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders](https://arxiv.org/abs/2301.12597) - Salesforce, 2023

**Additional References:**
- OpenAI CLIP GitHub: https://github.com/openai/CLIP
- OpenCLIP: https://github.com/mlfoundations/open_clip
- HuggingFace Transformers: https://huggingface.co/docs/transformers
