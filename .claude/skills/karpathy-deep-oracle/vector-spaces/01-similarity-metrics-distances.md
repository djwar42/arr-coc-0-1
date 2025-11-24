# Similarity Metrics & Distance Functions for Vector Embeddings

## Overview

Similarity metrics and distance functions are fundamental to vector-based machine learning systems. They determine how we measure relationships between embeddings, influencing everything from semantic search to clustering algorithms. Choosing the right metric can dramatically impact model performance, especially in high-dimensional spaces where intuition from 2D/3D geometry breaks down.

**Core Principle**: Match your similarity metric to the one used during model training. For example, if a sentence-transformer model was trained with cosine similarity, use cosine similarity for retrieval and comparison tasks.

## Section 1: Cosine Similarity

### Definition

Cosine similarity measures the **angle** between two vectors, not their magnitude. It computes the cosine of the angle by taking the dot product divided by the product of vector magnitudes:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Range: -1 (opposite directions) to +1 (same direction), with 0 indicating orthogonality.

### Mathematical Properties

From [Pinecone Vector Similarity](https://www.pinecone.io/learn/vector-similarity/) (accessed 2025-02-02):

- **Direction-only**: Magnitude is normalized away, only orientation matters
- **Bounded**: Always returns values in [-1, 1]
- **Scale invariant**: `cos(2A, 2B) = cos(A, B)`

**Relationship to dot product**: For normalized vectors (length = 1), cosine similarity equals the dot product.

### When to Use Cosine Similarity

**Best for**:
- **High-dimensional embeddings** (100+ dimensions)
- **Text analysis** with word counts or TF-IDF vectors
- **Document similarity** where length differences are irrelevant
- **Sentence embeddings** from models like CLIP, BERT, Sentence-BERT

From [Baeldung: Euclidean vs Cosine](https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity) (accessed 2025-02-02):

> "Vectors with a high cosine similarity are located in the same general direction from the origin... Cosine similarity is proportional to the dot product of two vectors and inversely proportional to the product of their magnitudes."

**Example use case**: Comparing documents where one is 10,000 words and another is 500 words. Cosine similarity treats them as equally comparable if their word distributions (directions) are similar, ignoring that one is 20x longer.

### Advantages

1. **Dimension-robust**: Works well even in 1000+ dimensional spaces
2. **Intuitive for text**: Natural interpretation as "thematic similarity"
3. **Normalized comparison**: Different-sized documents compared fairly
4. **CLIP/BERT standard**: Most vision-language models use this metric

### Disadvantages

1. **Magnitude ignored**: Cannot distinguish `[1, 2]` from `[100, 200]`
2. **Rating scales lost**: In recommendation systems, magnitude differences between user ratings are discarded
3. **Not a true distance metric**: Violates triangle inequality

### Implementation

**NumPy**:
```python
import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Batch computation
def cosine_similarity_matrix(X):
    """Pairwise cosine similarities for matrix X (n_samples, n_features)."""
    # Normalize rows
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Compute similarities
    return X_normalized @ X_normalized.T
```

**PyTorch**:
```python
import torch
import torch.nn.functional as F

# Single pair
cosine_sim = F.cosine_similarity(tensor_a, tensor_b, dim=0)

# Batch pairwise
def cosine_similarity_torch(X):
    """Pairwise cosine similarities (GPU-optimized)."""
    X_normalized = F.normalize(X, p=2, dim=1)
    return X_normalized @ X_normalized.T
```

**Scikit-learn**:
```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings_matrix)
```

## Section 2: Euclidean Distance (L2 Distance)

### Definition

Euclidean distance is the straight-line distance between two points in space:

```
d(A, B) = √(Σᵢ (aᵢ - bᵢ)²)
```

This is the L2-norm of the difference vector `(A - B)`.

### Geometric Interpretation

From [Baeldung: Euclidean vs Cosine](https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity):

> "Vectors with a small Euclidean distance from one another are located in the same region of a vector space."

Euclidean distance measures **absolute positional difference**, considering both direction and magnitude.

### When to Use Euclidean Distance

**Best for**:
- **Low-dimensional data** (2D-50D typically)
- **Physical measurements** (height, weight, coordinates)
- **Magnitude matters**: When vector length encodes important information
- **K-means clustering** (designed for Euclidean distance)
- **LSH (Locality Sensitive Hashing)** encodings

### The Curse of Dimensionality

From [Pinecone Vector Similarity](https://www.pinecone.io/learn/vector-similarity/):

> "As the dimensionality increases of your data, the less useful Euclidean distance becomes... Higher-dimensional space does not act as we would, intuitively, expect from 2- or 3-dimensional space."

In high dimensions (1000+), all points become approximately equidistant, making comparisons less meaningful.

### Advantages

1. **Intuitive**: Matches human spatial understanding
2. **Complete information**: Uses both direction and magnitude
3. **True metric**: Satisfies triangle inequality
4. **Geometric interpretability**: Clear physical meaning

### Disadvantages

1. **Not scale-invariant**: Features with larger ranges dominate
2. **Requires normalization**: Must standardize features to same scale
3. **Poor in high dimensions**: Distances become uniform (curse of dimensionality)
4. **Sensitive to outliers**: Squared differences amplify large deviations

### Implementation

**NumPy**:
```python
import numpy as np

def euclidean_distance(a, b):
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

# Batch pairwise
from scipy.spatial.distance import cdist
distances = cdist(X, X, metric='euclidean')
```

**PyTorch**:
```python
import torch

# Single pair
dist = torch.dist(tensor_a, tensor_b, p=2)

# Batch pairwise (efficient)
def euclidean_distances_torch(X):
    """Pairwise Euclidean distances (GPU-optimized)."""
    # X: (n_samples, n_features)
    dot_product = X @ X.T
    squared_norms = torch.diag(dot_product).unsqueeze(0)
    distances_squared = squared_norms + squared_norms.T - 2 * dot_product
    return torch.sqrt(torch.clamp(distances_squared, min=0))
```

**Scikit-learn**:
```python
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(embeddings_matrix)
```

## Section 3: Dot Product (Inner Product)

### Definition

The dot product is the sum of element-wise multiplications:

```
A · B = Σᵢ (aᵢ × bᵢ) = ||A|| × ||B|| × cos(θ)
```

Unlike cosine similarity, it **includes magnitude information**.

### Relationship to Cosine Similarity

From [Pinecone Vector Similarity](https://www.pinecone.io/learn/vector-similarity/):

> "The dot product can be expressed as the product of the magnitudes of the vectors and the cosine of the angle between them."

**For normalized vectors**: `dot(A, B) = cosine_similarity(A, B)`

This is why many systems normalize embeddings and use dot product—it's computationally cheaper than cosine similarity but mathematically equivalent for unit vectors.

### When to Use Dot Product

**Best for**:
- **Matrix factorization** (collaborative filtering)
- **Recommendation systems**: User-item embeddings
- **Transformer attention**: Query-key similarity
- **Normalized embeddings**: When vectors are pre-normalized to unit length

From [Pinecone](https://www.pinecone.io/learn/vector-similarity/):

> "You're likely to run into many Large Language Models (LLMs) that use dot product for training... For example, the msmarco-bert-base-dot-v5 model specifies the 'suitable scoring functions' to be only dot product."

### Advantages

1. **Computationally efficient**: No square roots or divisions
2. **LLM standard**: Many models (BERT variants) trained with dot product
3. **Magnitude encoding**: Can represent "strength" or "popularity"
4. **Hardware optimized**: GPUs excel at matrix multiplication

### Disadvantages

1. **Unbounded**: No fixed range, makes thresholding harder
2. **Scale dependent**: Large vectors dominate scores
3. **Requires normalization**: For fair comparisons, normalize first

### Implementation

**NumPy**:
```python
import numpy as np

# Single pair
similarity = np.dot(a, b)

# Batch pairwise
similarities = X @ X.T
```

**PyTorch**:
```python
import torch

# Single pair
similarity = torch.dot(a, b)  # 1D tensors
similarity = (a @ b)  # Works for any dimensions

# Batch pairwise
similarities = X @ X.T
```

## Section 4: Manhattan Distance (L1 Distance)

### Definition

Manhattan distance (also called Taxicab or City Block distance) is the sum of absolute differences:

```
d(A, B) = Σᵢ |aᵢ - bᵢ|
```

Named after Manhattan street grids where you can only move along axes (no diagonal shortcuts).

From [Maarten Grootendorst: 9 Distance Measures](https://maartengrootendorst.com/blog/distances/) (accessed 2025-02-02):

> "Manhattan distance then refers to the distance between two vectors if they could only move right angles. There is no diagonal movement involved."

### When to Use Manhattan Distance

**Best for**:
- **Discrete/binary attributes**: When features are categorical
- **Grid-based problems**: City navigation, warehouse logistics
- **Robust to outliers**: Less sensitive than Euclidean (no squaring)
- **High-dimensional data**: Can work better than Euclidean in some cases

### Advantages

1. **Outlier robust**: Linear scaling vs. quadratic (Euclidean)
2. **Interpretable**: Sum of component-wise differences
3. **Grid-realistic**: Models real-world constrained movement

### Disadvantages

1. **Overestimates distance**: Always ≥ Euclidean distance
2. **Less intuitive**: Not the "shortest path"
3. **Direction insensitive**: Same distance for different paths

### Implementation

**NumPy**:
```python
import numpy as np

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# Batch
from scipy.spatial.distance import cdist
distances = cdist(X, X, metric='cityblock')
```

**PyTorch**:
```python
import torch

dist = torch.dist(a, b, p=1)  # L1 norm

# Batch
distances = torch.cdist(X, X, p=1)
```

## Section 5: Chebyshev Distance (L∞ Distance)

### Definition

Chebyshev distance is the **maximum** absolute difference along any dimension:

```
d(A, B) = maxᵢ |aᵢ - bᵢ|
```

From [Maarten Grootendorst](https://maartengrootendorst.com/blog/distances/):

> "Chebyshev distance is defined as the greatest of difference between two vectors along any coordinate dimension... It is simply the maximum distance along one axis."

### When to Use Chebyshev Distance

**Best for**:
- **Chess/game AI**: King movement on chessboard
- **Warehouse logistics**: Overhead crane movement
- **8-way movement**: Games with unrestricted diagonal movement
- **Worst-case scenarios**: When maximum deviation matters

### Advantages

1. **Computationally cheap**: Single max operation
2. **Worst-case focus**: Highlights largest discrepancy

### Disadvantages

1. **Very specific use-cases**: Not general-purpose
2. **Information loss**: Ignores all but one dimension

### Implementation

**NumPy**:
```python
import numpy as np

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

# Batch
from scipy.spatial.distance import cdist
distances = cdist(X, X, metric='chebyshev')
```

**PyTorch**:
```python
import torch

dist = torch.max(torch.abs(a - b))
```

## Section 6: Other Important Metrics

### Hamming Distance

**Definition**: Number of positions at which vectors differ.

**Use case**: Binary vectors, error detection, categorical similarity.

```python
from scipy.spatial.distance import hamming
dist = hamming(a, b)  # Returns proportion of differing elements
```

### Jaccard Index (Intersection over Union)

**Definition**: Size of intersection divided by size of union.

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

**Use case**: Set similarity, image segmentation, text overlap.

```python
from sklearn.metrics import jaccard_score
similarity = jaccard_score(y_true, y_pred)
```

## Section 7: Metric Selection Guide

### Decision Matrix

| Metric | Best For | Avoid When | Computational Cost |
|--------|----------|------------|-------------------|
| **Cosine Similarity** | High-dim text/embeddings, direction matters | Magnitude is important | Medium (normalization) |
| **Euclidean Distance** | Low-dim physical data, complete info needed | High dimensions (>100D) | Low |
| **Dot Product** | Normalized embeddings, LLM outputs | Unnormalized vectors | Very Low |
| **Manhattan (L1)** | Discrete attributes, outlier-prone data | Need shortest path | Low |
| **Chebyshev (L∞)** | Grid movement, worst-case analysis | General-purpose tasks | Very Low |

### Training Alignment Principle

From [Pinecone](https://www.pinecone.io/learn/vector-similarity/):

> "The basic rule of thumb in selecting the best similarity metric for your Pinecone index is to **match it to the one used to train your embedding model**."

**Examples**:
- `all-MiniLM-L6-v2`: Trained with **cosine similarity**
- `msmarco-bert-base-dot-v5`: Trained with **dot product**
- Custom models: Check training loss function

### High-Dimensional Considerations

From search results on [Data Science Stack Exchange](https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity) (accessed 2025-02-02):

> "Cosine similarity looks at the angle between two vectors, euclidian similarity at the distance between two points."

**In practice**:
- **Dimensions < 50**: Euclidean often works well
- **Dimensions 50-500**: Cosine similarity generally preferred
- **Dimensions > 500**: Cosine similarity strongly recommended

## Section 8: Performance Optimization

### GPU-Accelerated Computation (PyTorch)

**Batch cosine similarity** (most efficient):
```python
import torch
import torch.nn.functional as F

def fast_cosine_similarity(embeddings):
    """
    Compute pairwise cosine similarities for matrix of embeddings.

    Args:
        embeddings: Tensor of shape (n_samples, embedding_dim)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize once
    normalized = F.normalize(embeddings, p=2, dim=1)
    # Single matrix multiplication
    return normalized @ normalized.T

# Example: 10,000 embeddings of dimension 768
embeddings = torch.randn(10000, 768, device='cuda')
similarities = fast_cosine_similarity(embeddings)
# Result: (10000, 10000) similarity matrix in <100ms on modern GPU
```

### Memory-Efficient Batch Processing

For large datasets that don't fit in memory:

```python
def batched_similarities(embeddings, batch_size=1000):
    """Compute similarities in batches to save memory."""
    n = embeddings.shape[0]
    similarities = torch.zeros(n, n, device=embeddings.device)

    normalized = F.normalize(embeddings, p=2, dim=1)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch = normalized[i:end_i]

        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            similarities[i:end_i, j:end_j] = batch @ normalized[j:end_j].T

    return similarities
```

### Top-K Similarity Search

When you only need the K most similar items:

```python
def top_k_similar(query_embedding, embeddings, k=10, metric='cosine'):
    """
    Find K most similar embeddings efficiently.

    Args:
        query_embedding: Single query vector (embedding_dim,)
        embeddings: Database of vectors (n_samples, embedding_dim)
        k: Number of top results to return
        metric: 'cosine', 'euclidean', or 'dot'

    Returns:
        indices: Indices of top K similar items
        similarities: Similarity scores
    """
    if metric == 'cosine':
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        similarities = (query_norm @ emb_norm.T).squeeze(0)
    elif metric == 'dot':
        similarities = (query_embedding @ embeddings.T)
    elif metric == 'euclidean':
        # Negative distance for "similarity" sorting
        similarities = -torch.cdist(query_embedding.unsqueeze(0), embeddings).squeeze(0)

    # Get top K
    top_k_scores, top_k_indices = torch.topk(similarities, k)

    return top_k_indices, top_k_scores
```

## Section 9: Practical Examples

### Example 1: Text Similarity with Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Load model (trained with cosine similarity)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sits on the mat.",
    "A feline rests on a rug.",
    "Dogs play in the park."
]
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarities (matches training metric)
cosine_sim = F.cosine_similarity(
    embeddings.unsqueeze(1),  # (3, 1, 384)
    embeddings.unsqueeze(0),  # (1, 3, 384)
    dim=2
)

print(cosine_sim)
# Output:
# tensor([[1.0000, 0.7821, 0.2156],  # cat-cat, cat-feline, cat-dogs
#         [0.7821, 1.0000, 0.2398],  # feline-cat, feline-feline, feline-dogs
#         [0.2156, 0.2398, 1.0000]]) # dogs-cat, dogs-feline, dogs-dogs
```

### Example 2: Image Embedding Comparison

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Simulated CLIP image embeddings (512-dimensional)
image_embeddings = np.random.randn(1000, 512)

# Normalize for consistent comparison
from sklearn.preprocessing import normalize
image_embeddings_norm = normalize(image_embeddings, norm='l2')

# Cosine similarity (preferred for high-dim)
cosine_sims = cosine_similarity(image_embeddings_norm)

# Euclidean distance (will be large in 512D space)
euclidean_dists = euclidean_distances(image_embeddings_norm)

# Dot product on normalized = cosine similarity
dot_products = image_embeddings_norm @ image_embeddings_norm.T

# Verify equivalence
print(f"Cosine vs Dot Product (normalized): {np.allclose(cosine_sims, dot_products)}")
# True
```

### Example 3: Choosing Metrics by Task

```python
def recommend_metric(task_type, dimensionality, normalized=False):
    """
    Recommend similarity metric based on task characteristics.

    Args:
        task_type: 'text', 'image', 'tabular', 'collaborative_filtering'
        dimensionality: Number of features
        normalized: Whether embeddings are pre-normalized

    Returns:
        Recommended metric name and reasoning
    """
    recommendations = {
        'text': {
            'metric': 'cosine',
            'reason': 'Text embeddings are high-dimensional and direction matters more than magnitude'
        },
        'image': {
            'metric': 'cosine' if dimensionality > 100 else 'euclidean',
            'reason': f'High-dim ({dimensionality}D) favors cosine; low-dim can use euclidean'
        },
        'tabular': {
            'metric': 'euclidean' if dimensionality < 50 else 'cosine',
            'reason': 'Low-dim tabular data works with euclidean; high-dim needs cosine'
        },
        'collaborative_filtering': {
            'metric': 'dot',
            'reason': 'Matrix factorization models trained with dot product'
        }
    }

    rec = recommendations.get(task_type, recommendations['text'])

    if normalized and rec['metric'] == 'cosine':
        rec['metric'] = 'dot'
        rec['reason'] += ' (use dot product for speed since vectors are normalized)'

    return rec

# Examples
print(recommend_metric('text', 768))
# {'metric': 'cosine', 'reason': 'Text embeddings are high-dimensional...'}

print(recommend_metric('tabular', 12))
# {'metric': 'euclidean', 'reason': 'Low-dim tabular data works with euclidean...'}
```

## Sources

**Web Research** (accessed 2025-02-02):

**Core similarity metrics**:
- [Vector Similarity Explained | Pinecone](https://www.pinecone.io/learn/vector-similarity/) - Comprehensive guide to Euclidean, cosine, and dot product similarities
- [Euclidean Distance vs Cosine Similarity | Baeldung](https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity) - Formal mathematical definitions and comparisons
- [9 Distance Measures in Data Science | Maarten Grootendorst](https://maartengrootendorst.com/blog/distances/) - Detailed coverage of 9 distance metrics with use cases

**Practical comparisons**:
- [When to use cosine similarity over Euclidean similarity | Data Science Stack Exchange](https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity)
- [Cosine Distance vs Dot Product vs Euclidean in vector similarity search | Medium](https://medium.com/data-science-collective/cosine-distance-vs-dot-product-vs-euclidean-in-vector-similarity-search-227a6db32edb)

**Implementation resources**:
- [PyTorch Metric Learning Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Distance Metrics in Vector Search | Weaviate](https://weaviate.io/blog/distance-metrics-in-vector-search)
- [Similarity Metrics for Vector Search | Zilliz](https://zilliz.com/blog/similarity-metrics-for-vector-search)

**Additional references**:
- [Vector Similarity - Pinecone Documentation](https://docs.pinecone.io/)
- [Choosing Between Cosine Similarity, Dot Product, and Euclidean Distance for RAG Applications | Ragwalla](https://ragwalla.com/blog/choosing-between-cosine-similarity-dot-product-and-euclidean-distance-for-rag-applications)
