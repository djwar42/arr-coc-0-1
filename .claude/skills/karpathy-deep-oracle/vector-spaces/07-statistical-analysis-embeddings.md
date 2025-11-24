# Statistical Analysis of Embeddings

## Overview

Statistical analysis of embeddings is essential for understanding embedding quality, detecting issues, and validating representation learning systems. This guide covers distribution analysis, clustering validation, dimensionality assessment, and quality metrics for vision-language model embeddings.

## 1. Distribution Analysis

### 1.1 Isotropy and Anisotropy

**Isotropy** refers to uniform distribution of embeddings across all directions in the embedding space. Isotropic embeddings occupy the full dimensionality of the space, while **anisotropic** embeddings are concentrated in specific directions.

**Why It Matters:**
- Isotropic spaces provide better utilization of embedding dimensions
- Anisotropic embeddings waste capacity and reduce expressiveness
- Affects similarity measurements and retrieval quality

**Measuring Isotropy:**

From [An Isotropy Analysis in the Multilingual BERT Embedding](https://arxiv.org/abs/2110.04504) (Rajaee et al., 2021):

1. **Covariance Matrix Analysis**: Compute covariance of embedding vectors. Fully isotropic distribution has covariance proportional to identity matrix.

2. **Explained Variance Ratio**: Measure variance explained by principal components. Uniform distribution across components indicates isotropy.

3. **Angular Distribution**: Analyze pairwise cosine similarities. Isotropic embeddings have centered distribution around zero.

From [On Isotropy of Multimodal Embeddings](https://www.mdpi.com/2078-2489/14/7/392) (Tyshchuk et al., 2023):
- CLIP multimodal embeddings often exhibit anisotropic behavior
- Image and text modalities may have different isotropy characteristics
- Temperature parameter in contrastive learning affects isotropy

**Implementation (NumPy):**

```python
import numpy as np
from scipy.stats import norm

def measure_isotropy(embeddings):
    """
    Measure isotropy of embeddings using covariance analysis.

    Args:
        embeddings: np.ndarray of shape (n_samples, embed_dim)

    Returns:
        dict with isotropy metrics
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending order

    # Isotropy metrics
    explained_var_ratio = eigenvalues / eigenvalues.sum()
    cumulative_var = np.cumsum(explained_var_ratio)

    # Isotropy score: uniformity of eigenvalue distribution
    # Closer to 1 = more isotropic
    isotropy_score = 1 - np.std(eigenvalues) / np.mean(eigenvalues)

    return {
        'isotropy_score': isotropy_score,
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance': cumulative_var,
        'top_10_components_variance': cumulative_var[9]
    }
```

### 1.2 Mean and Variance Analysis

**Distribution Statistics:**

```python
def analyze_distribution(embeddings):
    """Analyze basic distribution statistics of embeddings."""
    return {
        'mean': embeddings.mean(axis=0),
        'std': embeddings.std(axis=0),
        'global_mean_norm': np.linalg.norm(embeddings.mean(axis=0)),
        'mean_norm_per_sample': np.linalg.norm(embeddings, axis=1).mean(),
        'std_norm_per_sample': np.linalg.norm(embeddings, axis=1).std()
    }
```

**Outlier Detection:**

```python
def detect_outliers(embeddings, threshold=3.0):
    """Detect embedding outliers using z-score method."""
    norms = np.linalg.norm(embeddings, axis=1)
    z_scores = (norms - norms.mean()) / norms.std()
    outlier_mask = np.abs(z_scores) > threshold

    return {
        'outlier_indices': np.where(outlier_mask)[0],
        'outlier_count': outlier_mask.sum(),
        'outlier_percentage': 100 * outlier_mask.sum() / len(embeddings)
    }
```

### 1.3 Embedding Collapse Detection

From [Understanding Embedding Dimensional Collapse](https://mlfrontiers.substack.com/p/understanding-embedding-dimensional) and [On the Embedding Collapse](https://arxiv.org/abs/2310.04400):

**Embedding collapse** occurs when embeddings occupy a low-dimensional subspace instead of utilizing full dimensionality.

```python
def detect_collapse(embeddings, threshold_variance=0.95):
    """
    Detect embedding collapse by measuring effective dimensionality.

    Returns:
        dict with collapse metrics
    """
    # PCA analysis
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(embeddings)

    # Find number of components explaining threshold variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.argmax(cumsum >= threshold_variance) + 1

    # Collapse severity
    full_dim = embeddings.shape[1]
    collapse_ratio = effective_dim / full_dim

    return {
        'effective_dimensionality': effective_dim,
        'full_dimensionality': full_dim,
        'collapse_ratio': collapse_ratio,
        'is_collapsed': collapse_ratio < 0.5,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }
```

**Signs of Embedding Collapse:**
- Effective dimensionality << embedding dimensionality
- Most variance concentrated in few principal components
- High correlation between embedding dimensions
- Poor performance on downstream tasks

## 2. Clustering Validation

### 2.1 Silhouette Score

From [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html):

The silhouette score measures how well-matched an object is to its own cluster versus other clusters. Range: [-1, +1]

- +1: Well-matched to cluster, poorly matched to neighbors
- 0: On decision boundary between clusters
- -1: Likely assigned to wrong cluster

```python
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

def compute_silhouette(embeddings, labels=None, n_clusters=10):
    """
    Compute silhouette score for embeddings.

    Args:
        embeddings: Embedding vectors
        labels: Cluster labels (if None, performs KMeans)
        n_clusters: Number of clusters if labels not provided
    """
    if labels is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    # Overall silhouette score
    overall_score = silhouette_score(embeddings, labels)

    # Per-sample silhouette scores
    sample_scores = silhouette_samples(embeddings, labels)

    # Per-cluster average scores
    unique_labels = np.unique(labels)
    cluster_scores = {}
    for label in unique_labels:
        mask = labels == label
        cluster_scores[int(label)] = sample_scores[mask].mean()

    return {
        'silhouette_score': overall_score,
        'per_sample_scores': sample_scores,
        'per_cluster_scores': cluster_scores,
        'labels': labels
    }
```

**Interpretation:**
- Score > 0.5: Strong cluster structure
- Score 0.25-0.5: Weak cluster structure
- Score < 0.25: No substantial cluster structure

### 2.2 Davies-Bouldin Index

The Davies-Bouldin Index measures cluster separation and compactness. Lower values indicate better clustering.

```python
from sklearn.metrics import davies_bouldin_score

def compute_davies_bouldin(embeddings, labels=None, n_clusters=10):
    """
    Compute Davies-Bouldin Index for clustering quality.

    Lower scores indicate better clustering (more separated, compact clusters).
    """
    if labels is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    db_score = davies_bouldin_score(embeddings, labels)

    return {
        'davies_bouldin_index': db_score,
        'labels': labels
    }
```

**Advantages over Silhouette:**
- Faster to compute
- Only requires cluster centroids and dispersions
- No pairwise distance computations

### 2.3 Calinski-Harabasz Index

Also known as Variance Ratio Criterion. Higher values indicate better-defined clusters.

```python
from sklearn.metrics import calinski_harabasz_score

def compute_calinski_harabasz(embeddings, labels=None, n_clusters=10):
    """
    Compute Calinski-Harabasz Index (Variance Ratio Criterion).

    Higher scores indicate better clustering.
    """
    if labels is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    ch_score = calinski_harabasz_score(embeddings, labels)

    return {
        'calinski_harabasz_index': ch_score,
        'labels': labels
    }
```

### 2.4 Optimal Cluster Number Selection

```python
def find_optimal_clusters(embeddings, max_clusters=20):
    """
    Find optimal number of clusters using multiple metrics.

    Returns cluster counts and corresponding quality scores.
    """
    from sklearn.cluster import KMeans

    results = {
        'n_clusters': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []
    }

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        results['n_clusters'].append(k)
        results['silhouette'].append(silhouette_score(embeddings, labels))
        results['davies_bouldin'].append(davies_bouldin_score(embeddings, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(embeddings, labels))
        results['inertia'].append(kmeans.inertia_)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Find optimal k (highest silhouette, lowest DB, highest CH)
    optimal_k_silhouette = results['n_clusters'][np.argmax(results['silhouette'])]
    optimal_k_db = results['n_clusters'][np.argmin(results['davies_bouldin'])]
    optimal_k_ch = results['n_clusters'][np.argmax(results['calinski_harabasz'])]

    results['optimal_k_silhouette'] = optimal_k_silhouette
    results['optimal_k_davies_bouldin'] = optimal_k_db
    results['optimal_k_calinski_harabasz'] = optimal_k_ch

    return results
```

## 3. Dimensionality Analysis

### 3.1 Intrinsic Dimensionality

**Intrinsic dimensionality** is the minimum number of features needed to represent data without significant information loss.

From [Redundancy, Isotropy, and Intrinsic Dimensionality](https://arxiv.org/abs/2506.01435):

```python
def estimate_intrinsic_dimension(embeddings, k_neighbors=10):
    """
    Estimate intrinsic dimensionality using PCA and k-NN methods.

    Args:
        embeddings: Embedding vectors
        k_neighbors: Number of neighbors for local ID estimation

    Returns:
        dict with ID estimates
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    # Global ID via PCA (95% variance threshold)
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(embeddings)
    global_id_95 = pca.n_components_

    # Global ID via PCA (99% variance threshold)
    pca_99 = PCA(n_components=0.99, svd_solver='full')
    pca_99.fit(embeddings)
    global_id_99 = pca_99.n_components_

    # Local ID via k-NN distance ratios (MLE estimator)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # MLE estimate: mean of log distance ratios
    # ID ≈ (k-1) / sum(log(d_k / d_i)) for i in 1..k-1
    local_ids = []
    for dists in distances:
        dists = dists[1:]  # exclude self
        if dists[-1] > 0 and np.all(dists > 0):
            log_ratios = np.log(dists[-1] / dists[:-1])
            local_id = (k_neighbors - 1) / np.sum(log_ratios)
            local_ids.append(local_id)

    avg_local_id = np.mean(local_ids) if local_ids else None

    return {
        'global_id_95pct': global_id_95,
        'global_id_99pct': global_id_99,
        'avg_local_id': avg_local_id,
        'embedding_dim': embeddings.shape[1],
        'compression_ratio_95': global_id_95 / embeddings.shape[1],
        'compression_ratio_99': global_id_99 / embeddings.shape[1]
    }
```

### 3.2 Explained Variance Analysis

```python
def explained_variance_analysis(embeddings):
    """Detailed analysis of explained variance across dimensions."""
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(embeddings)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)

    # Find dimensions for various thresholds
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    dims_for_threshold = {}
    for thresh in thresholds:
        dims_for_threshold[f'{int(thresh*100)}pct'] = np.argmax(cumulative_var >= thresh) + 1

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_var,
        'singular_values': pca.singular_values_,
        'dimensions_for_threshold': dims_for_threshold,
        'effective_rank': np.sum(pca.explained_variance_ratio_ > 0.01)
    }
```

### 3.3 Rank Analysis

```python
def matrix_rank_analysis(embeddings, tolerance=1e-10):
    """Analyze effective rank of embedding matrix."""
    # Numerical rank
    numerical_rank = np.linalg.matrix_rank(embeddings, tol=tolerance)

    # Stable rank (Frobenius norm / spectral norm)^2
    U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)
    frobenius_norm = np.linalg.norm(embeddings, 'fro')
    spectral_norm = s[0]  # largest singular value
    stable_rank = (frobenius_norm / spectral_norm) ** 2

    return {
        'numerical_rank': numerical_rank,
        'stable_rank': stable_rank,
        'full_rank': embeddings.shape[1],
        'rank_ratio': numerical_rank / embeddings.shape[1]
    }
```

## 4. Similarity Distribution Analysis

### 4.1 Pairwise Similarity Histograms

```python
def analyze_similarity_distribution(embeddings, sample_size=10000):
    """
    Analyze distribution of pairwise similarities.

    Args:
        embeddings: Embedding vectors (will be normalized)
        sample_size: Number of pairs to sample (for efficiency)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Sample pairs if dataset is large
    n_samples = len(normalized)
    if n_samples > 1000:
        indices = np.random.choice(n_samples, size=min(1000, n_samples), replace=False)
        sample_embeds = normalized[indices]
    else:
        sample_embeds = normalized

    # Compute pairwise cosine similarities
    sim_matrix = cosine_similarity(sample_embeds)

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]

    return {
        'mean_similarity': similarities.mean(),
        'std_similarity': similarities.std(),
        'min_similarity': similarities.min(),
        'max_similarity': similarities.max(),
        'median_similarity': np.median(similarities),
        'similarity_histogram': np.histogram(similarities, bins=50),
        'similarities': similarities
    }
```

### 4.2 Hubness Problem Detection

From [Hubness in High-Dimensional Spaces](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) (Radovanović et al., 2010):

**Hubness** occurs in high dimensions when certain points appear frequently in k-nearest neighbors of other points.

```python
def detect_hubness(embeddings, k=10):
    """
    Detect hubness problem in embeddings.

    Hubness: some points appear in many k-NN lists (hubs),
    others appear in few or none (anti-hubs).
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Count how many times each point appears in k-NN lists
    n_samples = len(embeddings)
    k_occurrence = np.zeros(n_samples)

    for idx_list in indices:
        for idx in idx_list[1:]:  # exclude self
            k_occurrence[idx] += 1

    # Hubness statistics
    mean_occurrence = k * (n_samples - 1) / n_samples  # expected value
    hubness_skewness = np.mean((k_occurrence - mean_occurrence) ** 3) / (np.std(k_occurrence) ** 3)

    # Identify hubs and anti-hubs
    threshold_hub = mean_occurrence + 2 * np.std(k_occurrence)
    threshold_antihub = mean_occurrence - 2 * np.std(k_occurrence)

    hubs = np.where(k_occurrence > threshold_hub)[0]
    antihubs = np.where(k_occurrence < threshold_antihub)[0]

    return {
        'k_occurrence': k_occurrence,
        'mean_k_occurrence': k_occurrence.mean(),
        'std_k_occurrence': k_occurrence.std(),
        'hubness_skewness': hubness_skewness,
        'n_hubs': len(hubs),
        'n_antihubs': len(antihubs),
        'hub_indices': hubs,
        'antihub_indices': antihubs,
        'has_hubness_problem': abs(hubness_skewness) > 1.0
    }
```

## 5. Embedding Quality Metrics

### 5.1 Alignment Quality (Multimodal)

For vision-language embeddings (e.g., CLIP):

```python
def measure_alignment_quality(image_embeddings, text_embeddings):
    """
    Measure alignment quality between image and text embeddings.

    Assumes paired embeddings: image_embeddings[i] corresponds to text_embeddings[i]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize embeddings
    image_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(image_norm, text_norm)

    # Diagonal contains correct pairs
    correct_similarities = np.diag(sim_matrix)

    # For each image, check if correct text is in top-k
    top_k = [1, 5, 10]
    image_to_text_recall = {}
    for k in top_k:
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        correct_in_top_k = np.any(top_k_indices == np.arange(len(image_embeddings))[:, None], axis=1)
        image_to_text_recall[f'R@{k}'] = correct_in_top_k.mean()

    # For each text, check if correct image is in top-k
    text_to_image_recall = {}
    for k in top_k:
        top_k_indices = np.argsort(-sim_matrix.T, axis=1)[:, :k]
        correct_in_top_k = np.any(top_k_indices == np.arange(len(text_embeddings))[:, None], axis=1)
        text_to_image_recall[f'R@{k}'] = correct_in_top_k.mean()

    return {
        'mean_correct_similarity': correct_similarities.mean(),
        'std_correct_similarity': correct_similarities.std(),
        'image_to_text_recall': image_to_text_recall,
        'text_to_image_recall': text_to_image_recall,
        'mean_retrieval_recall_at_1': (image_to_text_recall['R@1'] + text_to_image_recall['R@1']) / 2
    }
```

### 5.2 Uniformity and Tolerance

Measures from contrastive learning literature:

```python
def measure_uniformity_tolerance(embeddings, t=2.0):
    """
    Measure uniformity (how evenly distributed) and tolerance of embeddings.

    From "Understanding Contrastive Representation Learning through Alignment and Uniformity"
    (Wang & Isola, 2020)

    Args:
        embeddings: Normalized embedding vectors
        t: Temperature parameter for uniformity
    """
    # Normalize
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Uniformity: how uniformly distributed are embeddings on hypersphere
    # Lower is better (more uniform)
    n = len(normalized)
    indices = np.random.choice(n, size=min(10000, n * (n-1) // 2), replace=True)
    i = indices // n
    j = indices % n

    pairwise_dot = (normalized[i] * normalized[j]).sum(axis=1)
    uniformity = np.log(np.exp(t * pairwise_dot).mean())

    return {
        'uniformity': uniformity,
        'avg_pairwise_similarity': pairwise_dot.mean()
    }
```

## 6. Comprehensive Analysis Pipeline

```python
def comprehensive_embedding_analysis(embeddings, labels=None, modality_pairs=None):
    """
    Run complete statistical analysis pipeline on embeddings.

    Args:
        embeddings: Embedding vectors or dict {'image': ..., 'text': ...} for multimodal
        labels: Optional cluster labels
        modality_pairs: For multimodal, dict with 'image' and 'text' keys

    Returns:
        Complete analysis report
    """
    results = {}

    # Handle multimodal input
    if isinstance(embeddings, dict):
        is_multimodal = True
        image_embeds = embeddings['image']
        text_embeds = embeddings['text']
        embeds = image_embeds  # use image for single-modality analyses
    else:
        is_multimodal = False
        embeds = embeddings

    # 1. Distribution Analysis
    results['distribution'] = analyze_distribution(embeds)
    results['isotropy'] = measure_isotropy(embeds)
    results['outliers'] = detect_outliers(embeds)
    results['collapse'] = detect_collapse(embeds)

    # 2. Clustering Validation
    if labels is not None:
        results['silhouette'] = compute_silhouette(embeds, labels)
        results['davies_bouldin'] = compute_davies_bouldin(embeds, labels)
        results['calinski_harabasz'] = compute_calinski_harabasz(embeds, labels)

    # 3. Dimensionality Analysis
    results['intrinsic_dimensionality'] = estimate_intrinsic_dimension(embeds)
    results['explained_variance'] = explained_variance_analysis(embeds)
    results['rank'] = matrix_rank_analysis(embeds)

    # 4. Similarity Distribution
    results['similarity_distribution'] = analyze_similarity_distribution(embeds)
    results['hubness'] = detect_hubness(embeds)

    # 5. Quality Metrics
    results['uniformity'] = measure_uniformity_tolerance(embeds)

    # 6. Multimodal Alignment (if applicable)
    if is_multimodal:
        results['alignment'] = measure_alignment_quality(image_embeds, text_embeds)

    return results
```

## 7. Visualization Helpers

```python
def plot_analysis_results(results, save_path=None):
    """
    Create visualization plots for embedding analysis results.

    Requires matplotlib and seaborn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Explained Variance
    ax = axes[0, 0]
    cumvar = results['explained_variance']['cumulative_variance']
    ax.plot(cumvar[:50], marker='o')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance Curve')
    ax.legend()
    ax.grid(True)

    # Plot 2: Similarity Distribution
    ax = axes[0, 1]
    sims = results['similarity_distribution']['similarities']
    ax.hist(sims, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=sims.mean(), color='r', linestyle='--', label=f'Mean: {sims.mean():.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Pairwise Similarity Distribution')
    ax.legend()

    # Plot 3: Hubness k-occurrence
    ax = axes[0, 2]
    k_occ = results['hubness']['k_occurrence']
    ax.hist(k_occ, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=k_occ.mean(), color='r', linestyle='--', label=f'Mean: {k_occ.mean():.1f}')
    ax.set_xlabel('k-Occurrence')
    ax.set_ylabel('Frequency')
    ax.set_title('Hubness Distribution')
    ax.legend()

    # Plot 4: Eigenvalue Spectrum
    ax = axes[1, 0]
    eigenvalues = results['isotropy']['eigenvalues'][:50]
    ax.plot(eigenvalues, marker='o')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectrum (Top 50)')
    ax.set_yscale('log')
    ax.grid(True)

    # Plot 5: Dimensionality Summary
    ax = axes[1, 1]
    dims = [
        results['intrinsic_dimensionality']['global_id_95pct'],
        results['intrinsic_dimensionality']['global_id_99pct'],
        results['rank']['numerical_rank'],
        results['intrinsic_dimensionality']['embedding_dim']
    ]
    labels_dims = ['ID (95%)', 'ID (99%)', 'Rank', 'Embed Dim']
    ax.bar(labels_dims, dims, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Dimensionality')
    ax.set_title('Dimensionality Comparison')
    ax.grid(True, axis='y')

    # Plot 6: Quality Summary
    ax = axes[1, 2]
    metrics_text = f"""
    Isotropy Score: {results['isotropy']['isotropy_score']:.3f}
    Collapse Ratio: {results['collapse']['collapse_ratio']:.3f}
    Mean Similarity: {results['similarity_distribution']['mean_similarity']:.3f}
    Uniformity: {results['uniformity']['uniformity']:.3f}
    Hubness Skew: {results['hubness']['hubness_skewness']:.3f}
    """
    if 'alignment' in results:
        metrics_text += f"\n    R@1 (I→T): {results['alignment']['image_to_text_recall']['R@1']:.3f}"
        metrics_text += f"\n    R@1 (T→I): {results['alignment']['text_to_image_recall']['R@1']:.3f}"

    ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax.axis('off')
    ax.set_title('Quality Metrics Summary')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

## Sources

**Web Research:**
- [An Isotropy Analysis in the Multilingual BERT Embedding](https://arxiv.org/abs/2110.04504) - Rajaee et al., 2021 (accessed 2025-02-02)
- [On Isotropy of Multimodal Embeddings](https://www.mdpi.com/2078-2489/14/7/392) - Tyshchuk et al., 2023 (accessed 2025-02-02)
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html) (accessed 2025-02-02)
- [Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) - Radovanović et al., 2010 (accessed 2025-02-02)
- [Redundancy, Isotropy, and Intrinsic Dimensionality of Embeddings](https://arxiv.org/abs/2506.01435) (accessed 2025-02-02)
- [Understanding Embedding Dimensional Collapse](https://mlfrontiers.substack.com/p/understanding-embedding-dimensional) - Machine Learning Frontiers (accessed 2025-02-02)
- [On the Embedding Collapse When Scaling Up Recommendation Models](https://arxiv.org/abs/2310.04400) - Guo et al., 2024 (accessed 2025-02-02)

**Implementation Libraries:**
- scikit-learn metrics and clustering modules
- NumPy for numerical operations
- SciPy for statistical functions
