# Multivariate Analysis & Dimensionality Reduction

## Overview

Multivariate analysis examines relationships among multiple variables simultaneously, revealing patterns invisible to univariate methods. Dimensionality reduction techniques transform high-dimensional data into interpretable low-dimensional representations while preserving essential structure. These methods are fundamental to modern cognitive science research, enabling visualization of neural representations, behavioral data, and computational models.

**Core principle**: High-dimensional data often lies on lower-dimensional manifolds. Dimensionality reduction recovers these intrinsic structures, making complex relationships visible and computationally tractable.

## Principal Component Analysis (PCA)

### Mathematical Foundation

PCA identifies orthogonal directions of maximum variance in data:

**Algorithm**:
1. Center data: subtract mean from each feature
2. Compute covariance matrix: C = (1/n)X^T X
3. Eigenvalue decomposition: C = VΛV^T
4. Project onto top k eigenvectors: Z = XV_k

**Key properties**:
- Linear transformation preserving global variance
- Orthogonal components (uncorrelated)
- Optimal for Gaussian data
- Fast computation: O(min(n²p, np²))

From [Principal Component Analysis (PCA): Explained Step-by-Step](https://builtin.com/data-science/step-step-explanation-principal-component-analysis) (Built In, accessed 2025-11-16):
- PCA simplifies large datasets into smaller sets while retaining variance
- Each principal component is a linear combination of original variables
- Scree plots show variance explained by each component

From [Data Dimensionality Reduction Using Principal Component Analysis](https://ieeexplore.ieee.org/document/10593421/) (IEEE Xplore, Ramasubramanian et al., 2024):
- PCA reduces training dataset size for machine learning
- Preserves 95-99% of variance with 10-20% of original dimensions
- Critical for high-dimensional neural data analysis

### Applications in Cognitive Science

**Neural population analysis**:
- Reduce thousands of neurons to 3-10 latent dimensions
- Reveals computational subspaces (motor planning, working memory)
- Identifies neural trajectories during behavior

**Behavioral data**:
- Extract underlying constructs from questionnaires
- Reduce redundancy in correlated measurements
- Identify cognitive factors (attention, memory, processing speed)

**Computational models**:
- Analyze high-dimensional state spaces
- Visualize learning dynamics
- Compare model representations to neural data

### Limitations

**Linear assumptions fail for**:
- Nonlinear manifolds (Swiss roll, S-curves)
- Local neighborhood structure
- Clustered or branching data

**Solution**: Nonlinear dimensionality reduction (t-SNE, UMAP, autoencoders)

## Factor Analysis

### Core Concepts

Factor analysis models observed variables as linear combinations of latent factors:

**Model**: X = ΛF + ε

Where:
- X: observed variables (p-dimensional)
- Λ: factor loadings matrix
- F: latent factors (k-dimensional, k << p)
- ε: unique variance (measurement error)

From [Bayesian Multivariate Factor Analysis Model for Causal Inference](https://academic.oup.com/biostatistics/article/25/3/867/7459857) (Oxford Academic, Samartsidis et al., 2024):
- Bayesian factor analysis enables causal intervention effect estimation
- Handles multiple outcomes with shared latent structure
- Markov chain Monte Carlo for posterior inference

### Exploratory vs Confirmatory

**Exploratory Factor Analysis (EFA)**:
- Discover latent structure from data
- No prior hypotheses about factor structure
- Rotation methods (varimax, promax) for interpretability

**Confirmatory Factor Analysis (CFA)**:
- Test specific hypothesized factor structures
- Model fit indices (CFI, RMSEA, χ²)
- Theory-driven cognitive research

From [Clustered Factor Analysis for Multivariate Spatial Data](https://www.sciencedirect.com/science/article/pii/S2211675325000119) (Jin et al., 2025):
- Factor analysis reveals dependence structures in multivariate data
- Spatial extensions handle geographically clustered observations
- Applications: environmental monitoring, epidemiology

### Cognitive Science Applications

**Psychometric testing**:
- Intelligence factors (g-factor, fluid vs crystallized)
- Personality dimensions (Big Five)
- Cognitive abilities batteries

**Neural manifold analysis**:
- Shared variance across brain regions
- Latent cognitive states from fMRI/EEG
- Factor scores as neural signatures

**Behavioral experiments**:
- Response time decomposition
- Error variance analysis
- Individual difference factors

## Multidimensional Scaling (MDS)

### Distance Preservation

MDS visualizes dissimilarity matrices in low dimensions:

**Goal**: Find configuration Y where d(y_i, y_j) ≈ δ(x_i, x_j)

**Stress function**: σ = √[Σ(d_ij - δ_ij)² / Σδ_ij²]

From [Multidimensional Scaling Method and Practical Applications](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4934816) (Prem S., 2024):
- MDS identifies clusters by representing distances between points
- Visualizes similarity structures in complex datasets
- Applications: market research, social network analysis, cognitive mapping

**Variants**:
- **Classical MDS**: metric, preserves actual distances
- **Non-metric MDS**: preserves only rank order
- **Weighted MDS**: different dimensions have different importance

### Cognitive Similarity Spaces

**Semantic representations**:
- Word embeddings visualization
- Conceptual similarity judgments
- Category structure in memory

**Perceptual spaces**:
- Color similarity
- Sound timbre spaces
- Object recognition confusion matrices

From [Cluster-based Multidimensional Scaling for Visualization](https://iopscience.iop.org/article/10.1088/1402-4896/ad432e) (Hernández-León et al., 2024):
- Cluster-MDS (cl-MDS) addresses dimensionality reduction challenges
- Hierarchical clustering before MDS improves structure preservation
- Better visualization of high-dimensional scientific data

### Comparison to PCA

| Aspect | PCA | MDS |
|--------|-----|-----|
| Input | Raw data matrix | Distance/dissimilarity matrix |
| Assumption | Linear variance preservation | Distance preservation |
| Computation | Eigenvalue decomposition | Iterative optimization |
| Interpretability | Components = weighted features | Dimensions = abstract similarity |

## t-SNE: t-Distributed Stochastic Neighbor Embedding

### Probabilistic Framework

t-SNE preserves local neighborhood structure through probability matching:

**High-dimensional similarities**: p_ij = exp(-||x_i - x_j||²/2σ²) / Σ exp(-||x_k - x_l||²)

**Low-dimensional similarities**: q_ij = (1 + ||y_i - y_j||²)^(-1) / Σ (1 + ||y_k - y_l||²)^(-1)

**Cost function**: KL divergence KL(P||Q) = Σ p_ij log(p_ij/q_ij)

From [Comparative Analysis of Manifold Learning-Based Dimension Reduction](https://www.mdpi.com/2227-7390/12/15/2388) (Yi et al., 2024):
- t-SNE outperforms PCA for nonlinear manifolds
- Preserves local clusters and global separation
- Student's t-distribution prevents crowding in low dimensions

### Strengths & Weaknesses

**Advantages**:
- Excellent cluster visualization
- Preserves local neighborhoods
- Handles nonlinear manifolds
- Widely used in single-cell genomics, neural data

**Limitations**:
- Stochastic: different runs produce different embeddings
- Hyperparameter sensitive (perplexity)
- Computationally expensive: O(n²)
- Distances between clusters not meaningful
- Cannot embed new points without retraining

From [Dimensionality Reduction: PCA & t-SNE Explained](https://www.youtube.com/watch?v=75K6a1KzF8g) (The Data Key, 2024):
- t-SNE visualizes clusters and local structure
- Perplexity parameter controls local neighborhood size
- Typical values: 5-50 for small datasets, 30-100 for large

### Cognitive Science Applications

**Neural population visualization**:
- Single-neuron response patterns
- Population state trajectories
- Stimulus representation manifolds

**Behavioral clustering**:
- Task strategy identification
- Individual difference patterns
- Response pattern typologies

## UMAP: Uniform Manifold Approximation and Projection

### Topological Data Analysis Foundation

UMAP builds on Riemannian geometry and algebraic topology:

**Key insight**: Data lies on a locally connected Riemannian manifold

**Algorithm**:
1. Construct fuzzy topological representation of data
2. Build weighted k-neighbor graph
3. Optimize low-dimensional layout preserving fuzzy topology

From [How UMAP Works - Detailed Comparison with t-SNE](https://www.reddit.com/r/MachineLearning/comments/di7u52/r_how_umap_works_a_detailed_comparison_with_tsne/) (Reddit r/MachineLearning, 2019):
- UMAP uses category theory and Riemannian geometry
- Cross-entropy loss instead of KL divergence
- Better global structure preservation than t-SNE
- 10-100x faster for large datasets

### Advantages Over t-SNE

From [Nonlinear Manifold Learning with t-SNE and UMAP](https://www.youtube.com/watch?v=Chng7-nfDOo) (Bing Wen Brunton, 2024):

**Speed**: O(n^1.14) vs O(n²) for t-SNE
- UMAP: minutes for 100k points
- t-SNE: hours for 100k points

**Global structure**: UMAP preserves both local and global relationships
- Inter-cluster distances more meaningful
- Better topology preservation

**Deterministic**: Same random seed → same embedding
- Reproducible visualizations
- Consistent interpretations

**Embedding new data**: Can transform new points without retraining

### Hyperparameters

**n_neighbors**: Local neighborhood size (default 15)
- Small: emphasize local structure
- Large: emphasize global structure

**min_dist**: Minimum distance between points (default 0.1)
- Small: tight clusters
- Large: loose clusters

**metric**: Distance function (Euclidean, cosine, correlation)

From [Comparison of Dimensionality Reduction for Chemoinformatics](https://www.sciencedirect.com/science/article/pii/S2949747724000137) (Villares et al., 2024):
- UMAP superior for chemical compound visualization
- Better separation of functional groups
- Preserves pharmacological similarity

## Manifold Learning: Theoretical Framework

### Manifold Hypothesis

**Core assumption**: High-dimensional data concentrates near low-dimensional manifolds

**Mathematical formulation**:
- Data X ⊂ ℝ^D lies on manifold M
- Intrinsic dimension d << D
- Goal: Find embedding f: M → ℝ^d

From [The Wacky World of Non-Linear Manifold Learning](https://sites.gatech.edu/omscs7641/2024/03/10/no-straight-lines-here-the-wacky-world-of-non-linear-manifold-learning/) (Sites@GeorgiaTech, 2024):

**Algorithms overview**:
- **Isomap**: Preserves geodesic distances via graph shortest paths
- **LLE (Locally Linear Embedding)**: Preserves local linear reconstructions
- **Laplacian Eigenmaps**: Graph Laplacian spectral decomposition
- **t-SNE**: Preserves probabilistic neighborhoods
- **UMAP**: Preserves fuzzy topological structure

### Global vs Local Structure

**Local methods** (t-SNE, LLE):
- Focus on nearby point relationships
- Excellent cluster separation
- May distort global layout

**Global methods** (Isomap, PCA):
- Preserve large-scale structure
- Maintain inter-cluster distances
- May miss fine-grained clusters

**Hybrid approach** (UMAP):
- Balance local and global preservation
- n_neighbors parameter controls tradeoff

From [Comparison of Manifold Learning Algorithms](https://www.nature.com/articles/s41598-025-23301-7) (Min et al., 2025):
- Systematic assessment of Isomap, LLE, t-SNE, UMAP vs PCA
- UMAP best preserves both local and global structure
- Isomap better for geodesic distance preservation
- t-SNE superior for cluster visualization

### Computational Complexity

| Method | Complexity | Large-scale feasible? |
|--------|------------|----------------------|
| PCA | O(min(n²p, np²)) | Yes |
| MDS | O(n²) | No |
| Isomap | O(n² log n) | No |
| t-SNE | O(n²) | With approximations (Barnes-Hut) |
| UMAP | O(n^1.14) | Yes |

## Distributed Training for Dimensionality Reduction (File 4: FSDP)

From [distributed-training/03-fsdp-vs-deepspeed.md](../distributed-training/03-fsdp-vs-deepspeed.md):

**Large-scale PCA**:
- Distribute covariance matrix computation across GPUs
- FSDP shards eigenvector computation
- Incremental SVD for streaming data

**Neural dimensionality reduction**:
- Train autoencoders with FSDP for billion-parameter encoders
- Distributed t-SNE approximations (Barnes-Hut on GPUs)
- Parallel UMAP graph construction

**Cognitive neuroscience applications**:
- Whole-brain fMRI dimensionality reduction (millions of voxels)
- Large-scale neural recording analysis (10,000+ neurons)
- Population-level behavioral data (100,000+ participants)

## ML Workload Patterns for Analysis (File 12: K8s Workloads)

From [orchestration/03-ml-workload-patterns-k8s.md](../orchestration/03-ml-workload-patterns-k8s.md):

**Batch dimensionality reduction**:
- Kubernetes jobs for PCA across datasets
- Distributed t-SNE/UMAP on Spark clusters
- Scheduled factor analysis pipelines

**Interactive exploration**:
- JupyterHub for real-time UMAP parameter tuning
- Streamlit dashboards with dimensionality reduction
- MLflow tracking of embedding quality metrics

**Production pipelines**:
- Kubeflow for end-to-end analysis workflows
- Cached PCA transformations for new data
- UMAP models served via KServe

## TPU Optimization for Matrix Operations (File 16: TPU Fundamentals)

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md):

**PCA on TPUs**:
- Matrix multiplication acceleration (128x128 systolic array)
- Fast eigenvalue decomposition (batch operations)
- Bfloat16 for covariance computation (2x speedup)

**t-SNE/UMAP approximations**:
- k-NN graph construction on TPU
- Gradient descent optimization (attractive/repulsive forces)
- Batch embedding updates

**Cognitive data analysis**:
- Whole-brain connectivity matrices (TPU matrix ops)
- Similarity matrix computations (pairwise distances)
- Large-scale behavioral correlation analysis

**TPU advantages**:
- 10-100x faster eigenvalue decomposition vs CPU
- Memory bandwidth for large covariance matrices
- Pod slices for distributed dimensionality reduction

## ARR-COC-0-1: Relevance Realization as Learned Dimensionality Reduction

### Adaptive Compression as Manifold Learning

**Core insight**: Token allocation IS learned dimensionality reduction

From ARR-COC-0-1 implementation (arr-coc-0-1/arr_coc/):

**Quality adapter as nonlinear encoder**:
- Input: 13-channel texture array (D = 13 × patch_tokens)
- Latent: Relevance scores (d = 3 ways of knowing)
- Output: Token budgets (64-400 per patch)

**Three scorers = three PCA-like components**:
1. **Propositional** (InformationScorer): Shannon entropy → variance preservation
2. **Perspectival** (SaliencyScorer): Jungian archetypes → cluster separation
3. **Participatory** (QueryScorer): Cross-attention → supervised projection

**Opponent processing = balanced embedding**:
- Compress ↔ Particularize: Control information loss
- Exploit ↔ Explore: Balance certainty vs novelty
- Focus ↔ Diversify: Local vs global structure preservation

### Comparison to Classical Methods

| Method | ARR-COC Analog | Dimension |
|--------|----------------|-----------|
| PCA | Propositional knowing | Global variance |
| Factor Analysis | Three ways of knowing | Latent cognitive factors |
| t-SNE | Query-aware allocation | Local relevance clusters |
| UMAP | Relevance realization | Balanced local/global |

**Key difference**: ARR-COC is query-conditional
- PCA/Factor: Fixed projection for all queries
- ARR-COC: Different "projection" per query
- Enables contextual relevance realization

### Learned vs Fixed Dimensionality Reduction

**Classical methods** (PCA, MDS):
- Fixed transformation learned once
- Same embedding for all inputs
- Linear or fixed-kernel nonlinear

**Deep learning** (Autoencoders):
- Learned nonlinear transformation
- Generalizes to new data
- Black-box representations

**ARR-COC quality adapter**:
- Learned query-conditional reduction
- Interpretable dimensions (3 ways of knowing)
- Differentiable token allocation

**Training objective**:
- Minimize cross-entropy loss (VQA accuracy)
- Implicitly learns relevance manifold
- Balances compression vs task performance

### Visualization of Relevance Space

**Hypothetical analysis**:

```python
# Extract relevance scores for 1000 image patches
scores = []  # Shape: (1000, 3) - propositional, perspectival, participatory

# Apply UMAP to visualize relevance manifold
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(scores)

# Color by allocated tokens
plt.scatter(embedding[:, 0], embedding[:, 1], c=token_budgets, cmap='viridis')
plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
plt.title('Relevance Realization Manifold')
```

**Expected findings**:
- High-propositional patches cluster (informative regions)
- High-perspectival patches separate (salient objects)
- High-participatory patches group by query type
- Token allocation correlates with distance from manifold center

### Future: Learned Manifold Regularization

**Idea**: Regularize quality adapter to preserve known manifold structure

**Implementation**:
1. Compute pairwise patch similarities (perceptual hash)
2. Build k-NN graph in texture space
3. Add loss term: Preserve local neighborhoods in relevance space

**Benefit**: Smooth relevance allocation
- Similar patches → similar token budgets
- Avoid noisy allocation
- Improve generalization

**Connection to UMAP loss**:
- UMAP minimizes fuzzy set cross-entropy
- ARR-COC could add similar topological preservation term
- Balance task loss vs manifold smoothness

## Practical Guidelines for Method Selection

### Decision Tree

**Data characteristics**:

1. **Linear relationships?**
   - Yes → PCA (fast, interpretable)
   - No → Proceed to 2

2. **Need interpretable factors?**
   - Yes → Factor Analysis (psychometrics, theory testing)
   - No → Proceed to 3

3. **Have distance matrix only?**
   - Yes → MDS (dissimilarity data)
   - No → Proceed to 4

4. **Priority: speed or quality?**
   - Speed → UMAP (fast, good global structure)
   - Quality visualization → t-SNE (best clusters)

5. **Need to embed new points?**
   - Yes → UMAP or parametric t-SNE
   - No → Standard t-SNE OK

### Hyperparameter Tuning

**PCA**:
- n_components: Scree plot (elbow method) or cumulative variance ≥ 0.95
- Scaling: StandardScaler for mixed units

**t-SNE**:
- perplexity: 5-50 (small data), 30-100 (large data)
- learning_rate: 10-1000, tune if poor convergence
- n_iter: 1000+ (watch convergence plot)

**UMAP**:
- n_neighbors: 5-15 (local), 50-100 (global)
- min_dist: 0.0 (tight), 0.5 (loose)
- metric: Euclidean (default), cosine (text/sparse), correlation (gene expression)

From [Federated t-SNE and UMAP for Distributed Data](https://arxiv.org/html/2412.13495v1) (arXiv, December 2024):
- Federated learning enables privacy-preserving dimensionality reduction
- Fed-tSNE and Fed-UMAP learn distribution information without sharing data
- Applications: multi-site clinical studies, collaborative research

### Quality Metrics

**Cluster separation**:
- Silhouette score: (-1, 1), higher = better
- Davies-Bouldin index: Lower = better

**Neighborhood preservation**:
- Trustworthiness: Are k-nearest neighbors preserved?
- Continuity: Are embedded neighbors true neighbors?

**Computational cost**:
- Runtime: PCA < UMAP << t-SNE < MDS
- Memory: PCA (O(np)) < UMAP (O(n)) < t-SNE (O(n²))

## Integration with Cognitive Research Workflows

### Experimental pipeline

1. **Data collection**: fMRI, EEG, behavioral responses
2. **Preprocessing**: Artifact removal, normalization
3. **Dimensionality reduction**: PCA (noise reduction), then UMAP (visualization)
4. **Statistical analysis**: Cluster-based permutation tests
5. **Interpretation**: Map low-D structure to cognitive theory

### Reproducibility

**Version control**:
- Track random seeds (t-SNE, UMAP)
- Log hyperparameters (perplexity, n_neighbors)
- Save trained transformers (PCA, UMAP models)

**Validation**:
- Cross-validation for factor stability
- Multiple random initializations (t-SNE)
- Stability analysis across parameter ranges

### Reporting standards

**Method description**:
- Algorithm name and version (scikit-learn 1.3.0)
- Hyperparameters (perplexity=30, n_neighbors=15)
- Preprocessing steps (standardization, outlier removal)

**Visualization**:
- Color code by known labels (stimulus type, brain region)
- Include variance explained (PCA) or stress (MDS)
- Provide interactive plots (Plotly) for supplementary materials

## Sources

**Web Research**:
- [Principal Component Analysis (PCA): Explained Step-by-Step](https://builtin.com/data-science/step-step-explanation-principal-component-analysis) - Built In (accessed 2025-11-16)
- [Data Dimensionality Reduction Using Principal Component Analysis](https://ieeexplore.ieee.org/document/10593421/) - IEEE Xplore, Ramasubramanian et al., 2024
- [Bayesian Multivariate Factor Analysis Model for Causal Inference](https://academic.oup.com/biostatistics/article/25/3/867/7459857) - Oxford Academic, Samartsidis et al., 2024
- [Clustered Factor Analysis for Multivariate Spatial Data](https://www.sciencedirect.com/science/article/pii/S2211675325000119) - ScienceDirect, Jin et al., 2025
- [Multidimensional Scaling Method and Practical Applications](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4934816) - SSRN, Prem S., 2024
- [Cluster-based Multidimensional Scaling for Visualization](https://iopscience.iop.org/article/10.1088/1402-4896/ad432e) - IOPscience, Hernández-León et al., 2024
- [Comparative Analysis of Manifold Learning-Based Dimension Reduction](https://www.mdpi.com/2227-7390/12/15/2388) - MDPI, Yi et al., 2024
- [Dimensionality Reduction: PCA & t-SNE Explained](https://www.youtube.com/watch?v=75K6a1KzF8g) - The Data Key, 2024
- [How UMAP Works - Detailed Comparison with t-SNE](https://www.reddit.com/r/MachineLearning/comments/di7u52/r_how_umap_works_a_detailed_comparison_with_tsne/) - Reddit r/MachineLearning
- [Nonlinear Manifold Learning with t-SNE and UMAP](https://www.youtube.com/watch?v=Chng7-nfDOo) - Bing Wen Brunton, 2024
- [Comparison of Dimensionality Reduction for Chemoinformatics](https://www.sciencedirect.com/science/article/pii/S2949747724000137) - Villares et al., 2024
- [The Wacky World of Non-Linear Manifold Learning](https://sites.gatech.edu/omscs7641/2024/03/10/no-straight-lines-here-the-wacky-world-of-non-linear-manifold-learning/) - Sites@GeorgiaTech, 2024
- [Comparison of Manifold Learning Algorithms](https://www.nature.com/articles/s41598-025-23301-7) - Nature, Min et al., 2025
- [Federated t-SNE and UMAP for Distributed Data](https://arxiv.org/html/2412.13495v1) - arXiv, December 2024

**Referenced Influence Files**:
- File 4: [distributed-training/03-fsdp-vs-deepspeed.md](../distributed-training/03-fsdp-vs-deepspeed.md) - Distributed framework comparison
- File 12: [orchestration/03-ml-workload-patterns-k8s.md](../orchestration/03-ml-workload-patterns-k8s.md) - Production ML patterns
- File 16: [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - TPU architecture

**ARR-COC-0-1 Implementation**:
- Quality adapter architecture (arr-coc-0-1/arr_coc/adapter.py)
- Three ways of knowing scorers (arr-coc-0-1/arr_coc/knowing.py)
- Relevance realization pipeline (arr-coc-0-1/arr_coc/realizing.py)
