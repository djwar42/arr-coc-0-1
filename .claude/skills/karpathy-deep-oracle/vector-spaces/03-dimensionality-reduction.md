# Dimensionality Reduction for Embeddings

## Overview

Dimensionality reduction transforms high-dimensional embeddings (512, 768, 1024+ dimensions) into lower-dimensional representations (typically 2D or 3D) for visualization and analysis. This addresses the curse of dimensionality while preserving essential data structure.

**Why Reduce Dimensionality?**

- **Visualization**: Human perception limited to 2D/3D spaces
- **Computational efficiency**: Faster processing with fewer dimensions
- **Noise reduction**: Remove redundant or noisy dimensions
- **Pattern discovery**: Reveal hidden structure in data
- **Clustering**: Improve clustering algorithm performance

**Key Challenge**: Preserving meaningful relationships while drastically reducing dimensions (e.g., 768D → 2D represents 99.7% information loss).

## PCA (Principal Component Analysis)

### Algorithm Overview

PCA is a linear dimensionality reduction technique that finds orthogonal axes (principal components) capturing maximum variance in data.

**Mathematical Foundation:**

1. Center data by subtracting mean
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Select top k eigenvectors (principal components)
5. Project data onto new basis

**Key Properties:**

- **Linear transformation**: Preserves global structure
- **Variance maximization**: First PC captures most variance
- **Orthogonal components**: No correlation between PCs
- **Deterministic**: Same input always produces same output
- **Fast**: O(min(n²p, np²)) complexity

### When to Use PCA

**Best for:**

- Linear relationships in data
- Quick exploratory analysis
- Preprocessing for other algorithms
- Large datasets (scales well)
- Understanding variance distribution

**Not suitable for:**

- Non-linear manifolds
- Complex cluster structures
- Preserving local neighborhoods
- Data with categorical features

### Implementation (scikit-learn)

```python
from sklearn.decomposition import PCA
import numpy as np

# Basic PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Check explained variance
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total variance: {sum(pca.explained_variance_ratio_):.3f}")

# Determine optimal components
pca_full = PCA()
pca_full.fit(embeddings)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 95% variance
print(f"Components for 95% variance: {n_components}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar()
plt.show()
```

### Parameters

**n_components**: Number of components to keep
- Integer: Exact number (e.g., 2 for 2D visualization)
- Float (0-1): Variance to preserve (e.g., 0.95 for 95%)
- 'mle': Automatic selection via maximum likelihood

**whiten**: Whether to normalize components (default: False)
- True: Unit variance per component
- Useful for downstream algorithms assuming normalized features

**Performance**: PCA is fastest - handles 70,000 samples in ~1 second.

From [UMAP performance comparison](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) (accessed 2025-02-02):
- PCA: Linear scaling, excellent for large datasets
- 10-100x faster than non-linear methods

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Algorithm Overview

t-SNE is a non-linear technique that preserves local neighborhood structure by modeling pairwise similarities.

**How it Works:**

1. Compute pairwise similarities in high-dimensional space (Gaussian)
2. Initialize random low-dimensional embedding
3. Compute pairwise similarities in low-dimensional space (t-distribution)
4. Minimize KL divergence between distributions via gradient descent
5. Iterate until convergence

**Key Properties:**

- **Non-linear**: Captures complex manifolds
- **Local preservation**: Excels at preserving clusters
- **Stochastic**: Different runs produce different results
- **Non-parametric**: Cannot transform new data directly
- **Slower**: O(n²) complexity, struggles with large datasets

### Critical Parameters

**perplexity** (default: 30, range: 5-50)

Controls balance between local and global structure. Roughly corresponds to number of nearest neighbors.

From [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne) (accessed 2025-02-02):

- **Low perplexity (5-15)**: Emphasizes local structure, creates many small clusters
- **Medium perplexity (30-50)**: Balanced view, good default
- **High perplexity (50-100)**: Captures more global structure, fewer clusters

**WARNING**: Cluster size/distance can be misleading! Always try multiple perplexity values.

**learning_rate** (default: 200, range: 10-1000)

- Too low: Stuck in local minima, compressed ball
- Too high: Unstable optimization, scattered points
- Rule of thumb: n_samples / 12 to n_samples / 4

**n_iter** (default: 1000, minimum: 250)

- At least 250 iterations for convergence
- Complex datasets may need 1000-5000
- Monitor: Should reach stable state

### When to Use t-SNE

**Best for:**

- Visualizing high-dimensional clusters
- Discovering local structure
- Small to medium datasets (<10,000 samples)
- Final visualization (not preprocessing)

**Limitations:**

- Cannot embed new points
- Distances between clusters are meaningless
- Cluster sizes can be misleading
- Very slow on large datasets
- Different runs produce different layouts

From [t-SNE visualization comparison](https://distill.pub/2016/misread-tsne) (accessed 2025-02-02):
- Perplexity dramatically affects results
- Global distances not preserved
- Must try multiple parameter settings

### Implementation (scikit-learn)

```python
from sklearn.manifold import TSNE

# Basic t-SNE
tsne = TSNE(n_components=2, perplexity=30,
            learning_rate=200, n_iter=1000,
            random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Try multiple perplexities
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

for perp, ax in zip(perplexities, axes.flat):
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
               c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(f'Perplexity = {perp}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# For large datasets: use PCA preprocessing
from sklearn.decomposition import PCA

# Reduce to 50D first
pca = PCA(n_components=50)
embeddings_reduced = pca.fit_transform(embeddings)

# Then apply t-SNE
tsne = TSNE(n_components=2, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_reduced)
```

### Performance Notes

From [UMAP benchmarking](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) (accessed 2025-02-02):

- **Dataset size scaling**: O(n²) - very poor
- **12,800 samples**: ~100 seconds
- **70,000 samples**: 10+ minutes (MulticoreTSNE)
- scikit-learn implementation even slower
- **Recommendation**: Use PCA preprocessing for >10k samples

## UMAP (Uniform Manifold Approximation and Projection)

### Algorithm Overview

UMAP is a modern non-linear technique based on manifold learning and topological data analysis. Faster than t-SNE with better global structure preservation.

**How it Works:**

1. Construct high-dimensional fuzzy topological representation
2. Optimize low-dimensional representation to match topology
3. Use stochastic gradient descent with negative sampling
4. Leverage mathematical theory (Riemannian geometry)

**Key Properties:**

- **Non-linear**: Handles complex manifolds
- **Faster than t-SNE**: O(n log n) complexity
- **Better scalability**: Handles 100k+ samples
- **Preserves both local and global structure**
- **Can transform new data** (with caveats)
- **Theoretically grounded**: Mathematical foundations

From [Understanding UMAP](https://pair-code.github.io/understanding-umap/) (accessed 2025-02-02):
- Increased speed over t-SNE
- Better preservation of global structure
- More stable across parameter changes

### Critical Parameters

**n_neighbors** (default: 15, range: 2-200)

Controls local vs global structure balance. Corresponds to neighborhood size.

From [UMAP parameters guide](https://umap-learn.readthedocs.io/en/latest/parameters.html) (accessed 2025-02-02):

- **Low (2-10)**: Focus on local structure, fine detail, more clusters
- **Medium (15-50)**: Balanced view, good default
- **High (50-200)**: Emphasize global structure, fewer clusters

**min_dist** (default: 0.1, range: 0.0-0.99)

Controls how tightly points can be packed in low-dimensional space.

- **0.0**: Allows tight packing, dense clusters
- **0.1**: Default balance
- **0.3-0.5**: Spread out for better separation
- **0.99**: Maximum spread, loose clusters

**metric** (default: 'euclidean')

Distance metric for high-dimensional space:
- 'euclidean': Standard for normalized embeddings
- 'cosine': For CLIP/text embeddings (direction matters)
- 'manhattan', 'chebyshev': Alternative distances
- Custom callable functions supported

**n_epochs** (default: None - auto-determined)

Number of optimization iterations:
- Auto: Based on dataset size
- Small datasets: 200-500
- Large datasets: 100-200 (sufficient due to negative sampling)

### When to Use UMAP

**Best for:**

- Large datasets (10k-1M+ samples)
- Balanced local/global structure
- Faster visualization needs
- Embedding new test data
- Production pipelines

**Advantages over t-SNE:**

- 10-50x faster on large datasets
- Better global structure preservation
- More stable across parameter choices
- Can embed new points
- Scales to millions of samples

From [UMAP vs t-SNE comparison](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) (accessed 2025-02-02):

**Performance scaling:**
- 12,800 samples: ~5 seconds (vs t-SNE 100s)
- 70,000 samples: ~30 seconds (vs t-SNE 10+ minutes)
- Dramatically better scaling for larger datasets

### Implementation (umap-learn)

```python
import umap

# Basic UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                     n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# For CLIP/cosine embeddings
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                     metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(clip_embeddings)

# Parameter exploration
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
params = [
    (5, 0.0), (15, 0.1), (30, 0.1),
    (15, 0.0), (15, 0.5), (50, 0.3)
]

for (n_neigh, min_d), ax in zip(params, axes.flat):
    reducer = umap.UMAP(n_neighbors=n_neigh, min_dist=min_d,
                         random_state=42)
    emb_2d = reducer.fit_transform(embeddings)
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
               c=labels, cmap='tab10', alpha=0.6, s=5)
    ax.set_title(f'neighbors={n_neigh}, min_dist={min_d}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Transform new data
reducer.fit(train_embeddings)
new_embeddings_2d = reducer.transform(test_embeddings)
```

### UMAP for Production

```python
# Save model for later use
import pickle

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
reducer.fit(embeddings)

# Save
with open('umap_model.pkl', 'wb') as f:
    pickle.dump(reducer, f)

# Load and transform new data
with open('umap_model.pkl', 'rb') as f:
    reducer = pickle.load(f)

new_embeddings_2d = reducer.transform(new_embeddings)
```

## Comparison Matrix

### PCA vs t-SNE vs UMAP

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Type** | Linear | Non-linear | Non-linear |
| **Speed** | Fastest (1s) | Slow (100s) | Fast (5s) |
| **Scalability** | Excellent (millions) | Poor (thousands) | Excellent (millions) |
| **Local Structure** | Poor | Excellent | Excellent |
| **Global Structure** | Excellent | Poor | Good |
| **Deterministic** | Yes | No | No |
| **New Data** | Yes (fast) | No | Yes (slower) |
| **Parameters** | Simple (n_components) | Sensitive (perplexity) | Moderate (n_neighbors, min_dist) |
| **Complexity** | O(np²) | O(n²) | O(n log n) |
| **Best Dataset Size** | Any | <10k | Any |

*Timing for 12,800 MNIST samples on standard CPU*

From [performance benchmarks](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) (accessed 2025-02-02)

### When to Use Each Method

**Use PCA when:**
- Need fast results
- Working with very large datasets
- Linear relationships expected
- Interpretable components desired
- Preprocessing for other methods
- Variance analysis needed

**Use t-SNE when:**
- Dataset <10,000 samples
- Local cluster structure most important
- Final visualization only
- Can afford computation time
- Don't need to embed new points

**Use UMAP when:**
- Large datasets (10k+)
- Need both local and global structure
- Production systems requiring new data
- Faster results than t-SNE desired
- Working with CLIP/cosine embeddings

## Visualization Best Practices

### CLIP Embeddings Visualization

From [CLIP embedding visualization](https://blog.roboflow.com/embeddings-clustering-computer-vision-clip-umap/) (accessed 2025-02-02):

```python
import umap
from sklearn.preprocessing import normalize

# CLIP embeddings are already normalized, but verify
clip_embeddings = normalize(clip_embeddings, norm='l2')

# Use cosine metric for CLIP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',  # Important for CLIP
    n_components=2,
    random_state=42
)

embeddings_2d = reducer.fit_transform(clip_embeddings)

# Interactive visualization with labels
import plotly.express as px

fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=labels,
    hover_data={'image_path': image_paths},
    title='CLIP Embeddings - UMAP Projection'
)
fig.show()
```

### Multi-Scale Visualization

```python
# Compare all three methods
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)
axes[0].scatter(pca_result[:, 0], pca_result[:, 1],
                c=labels, cmap='tab10', alpha=0.6)
axes[0].set_title(f'PCA\nVariance: {sum(pca.explained_variance_ratio_):.2%}')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(embeddings)
axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                c=labels, cmap='tab10', alpha=0.6)
axes[1].set_title('t-SNE\nperplexity=30')

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_result = reducer.fit_transform(embeddings)
axes[2].scatter(umap_result[:, 0], umap_result[:, 1],
                c=labels, cmap='tab10', alpha=0.6)
axes[2].set_title('UMAP\nn_neighbors=15')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 3D Visualizations

```python
import plotly.graph_objects as go

# UMAP 3D
reducer = umap.UMAP(n_components=3, n_neighbors=15,
                     min_dist=0.1, random_state=42)
embeddings_3d = reducer.fit_transform(embeddings)

fig = go.Figure(data=[go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=labels,
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    title='UMAP 3D Projection',
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    )
)
fig.show()
```

## Common Pitfalls

### t-SNE Misinterpretations

From [Misreading t-SNE](http://distill.pub/2016/misread-tsne) (accessed 2025-02-02):

**Don't assume:**
- Cluster sizes are meaningful (they're not!)
- Distances between clusters are meaningful (they're not!)
- All structure is captured (try different perplexities)
- Single run is representative (use multiple random seeds)

**Do:**
- Try 3-5 different perplexity values
- Run multiple times with different seeds
- Use for exploration, not definitive conclusions
- Validate findings with quantitative metrics

### UMAP Parameter Selection

**Common mistakes:**
- Using default parameters without exploration
- n_neighbors too small (<5): Over-fragmented clusters
- n_neighbors too large (>100): Loss of local detail
- min_dist = 0 on noisy data: Overfitting to noise
- Wrong metric: euclidean for cosine-normalized embeddings

### PCA Limitations

**When PCA fails:**
- Non-linear manifolds (e.g., Swiss roll)
- Data in concentric circles
- XOR-like patterns
- Categorical relationships

**Solution:** Use as preprocessing step before non-linear methods.

## Performance Optimization

### Large Dataset Strategies

```python
# Strategy 1: PCA + UMAP pipeline
from sklearn.decomposition import PCA
import umap

# Reduce 768D → 50D with PCA (fast)
pca = PCA(n_components=50)
embeddings_50d = pca.fit_transform(embeddings)

# Then UMAP 50D → 2D (faster on fewer dims)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
embeddings_2d = reducer.fit_transform(embeddings_50d)

# Strategy 2: Subsample for parameter tuning
import numpy as np

# Tune on subset
sample_idx = np.random.choice(len(embeddings), 5000, replace=False)
sample_embeddings = embeddings[sample_idx]

# Find best parameters
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
reducer.fit(sample_embeddings)

# Apply to full dataset
embeddings_2d = reducer.fit_transform(embeddings)
```

### Memory Optimization

```python
# For very large datasets
import umap
from sklearn.utils import gen_batches

# Process in batches
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, low_memory=True)

# Fit on full data
reducer.fit(embeddings)

# Transform in batches
batch_size = 10000
results = []
for batch in gen_batches(len(test_embeddings), batch_size):
    batch_result = reducer.transform(test_embeddings[batch])
    results.append(batch_result)

embeddings_2d = np.vstack(results)
```

## Quality Assessment

### Quantitative Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette score (higher is better, range: -1 to 1)
sil_score = silhouette_score(embeddings_2d, labels)
print(f"Silhouette score: {sil_score:.3f}")

# Davies-Bouldin score (lower is better)
db_score = davies_bouldin_score(embeddings_2d, labels)
print(f"Davies-Bouldin score: {db_score:.3f}")

# Preservation of k-nearest neighbors
from sklearn.neighbors import NearestNeighbors

def knn_preservation(X_high, X_low, k=10):
    """Measure how well k-NN are preserved."""
    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k).fit(X_low)

    _, indices_high = nn_high.kneighbors(X_high)
    _, indices_low = nn_low.kneighbors(X_low)

    # Compute overlap
    overlap = []
    for i in range(len(X_high)):
        overlap.append(len(set(indices_high[i]) & set(indices_low[i])) / k)

    return np.mean(overlap)

preservation = knn_preservation(embeddings, embeddings_2d, k=10)
print(f"10-NN preservation: {preservation:.2%}")
```

## Sources

**Research Papers:**
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) - van der Maaten & Hinton, 2008 (accessed 2025-02-02)
- [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) - McInnes et al., 2018 (accessed 2025-02-02)

**Documentation:**
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) (accessed 2025-02-02)
- [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) (accessed 2025-02-02)
- [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/) (accessed 2025-02-02)

**Tutorials & Guides:**
- [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne) - Distill.pub (accessed 2025-02-02)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) - Google PAIR (accessed 2025-02-02)
- [UMAP Performance Benchmarking](https://umap-learn.readthedocs.io/en/latest/benchmarking.html) (accessed 2025-02-02)

**Applications:**
- [CLIP Embeddings Visualization with UMAP](https://blog.roboflow.com/embeddings-clustering-computer-vision-clip-umap/) - Roboflow Blog (accessed 2025-02-02)
- [Dimension Reduction Techniques Comparison](https://voxel51.com/blog/how-to-visualize-your-data-with-dimension-reduction-techniques) - Voxel51 (accessed 2025-02-02)

**Implementation Examples:**
- [Plotly t-SNE and UMAP](https://plotly.com/python/t-sne-and-umap-projections/) (accessed 2025-02-02)
- [DataCamp t-SNE Tutorial](https://www.datacamp.com/tutorial/introduction-t-sne) (accessed 2025-02-02)
