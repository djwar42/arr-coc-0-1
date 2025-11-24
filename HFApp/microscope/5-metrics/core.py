"""
Metrics Visualization - Summary statistics and performance metrics

Display key metrics for understanding system behavior:
- Patch selection statistics (coverage, diversity)
- Score distributions (histograms, quartiles)
- Query-awareness metrics (variance across queries)
- Efficiency metrics (tokens used, compression ratio)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def compute_summary_metrics(
    selected_indices,
    scores,
    query="",
    grid_size=32,
    total_patches=1024
):
    """
    Compute summary statistics for patch selection.

    Args:
        selected_indices: [K] Selected patch indices
        scores: [N] Relevance scores
        query: Query string
        grid_size: Grid dimension
        total_patches: Total number of patches

    Returns:
        Dict of metrics
    """
    K = len(selected_indices)
    N = len(scores)

    # Compute clustering metrics
    clustering = _compute_clustering_metrics(selected_indices, grid_size)

    metrics = {
        # Selection stats
        'num_selected': K,
        'selection_ratio': K / total_patches,
        'compression_ratio': total_patches / K if K > 0 else 0,

        # Score distribution
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_q25': float(np.percentile(scores, 25)),
        'score_q50': float(np.percentile(scores, 50)),
        'score_q75': float(np.percentile(scores, 75)),

        # Selected patch stats
        'selected_score_mean': float(scores[selected_indices].mean()) if K > 0 else 0,
        'selected_score_std': float(scores[selected_indices].std()) if K > 0 else 0,

        # Spatial distribution
        'spatial_coverage': _compute_spatial_coverage(selected_indices, grid_size),
        'spatial_entropy': _compute_spatial_entropy(selected_indices, grid_size),

        # Clustering metrics (NEW!)
        'mean_nn_distance': clustering['mean_nn_distance'],
        'clustering_coef': clustering['clustering_coef'],
        'num_clusters': clustering['num_clusters'],
    }

    return metrics


def _compute_spatial_coverage(selected_indices, grid_size):
    """Compute what fraction of grid regions are covered."""
    if len(selected_indices) == 0:
        return 0.0

    # Divide into 4×4 regions
    region_size = grid_size // 4
    regions_covered = set()

    for idx in selected_indices:
        y = idx // grid_size
        x = idx % grid_size
        region_y = y // region_size
        region_x = x // region_size
        regions_covered.add((region_y, region_x))

    return len(regions_covered) / 16  # 4×4 = 16 regions


def _compute_spatial_entropy(selected_indices, grid_size):
    """Compute entropy of spatial distribution."""
    if len(selected_indices) == 0:
        return 0.0

    # Histogram of patches across 8×8 regions
    region_size = grid_size // 8
    hist = np.zeros((8, 8))

    for idx in selected_indices:
        y = idx // grid_size
        x = idx % grid_size
        region_y = min(y // region_size, 7)
        region_x = min(x // region_size, 7)
        hist[region_y, region_x] += 1

    # Normalize
    hist = hist / hist.sum()

    # Compute entropy
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def _compute_clustering_metrics(selected_indices, grid_size):
    """
    Compute spatial clustering metrics.

    Returns:
        Dict with:
        - mean_nn_distance: Average distance to nearest neighbor
        - clustering_coef: How clustered patches are (0=scattered, 1=clustered)
        - num_clusters: Approximate number of spatial clusters
    """
    if len(selected_indices) < 2:
        return {
            'mean_nn_distance': 0.0,
            'clustering_coef': 0.0,
            'num_clusters': len(selected_indices)
        }

    # Convert indices to (y, x) coordinates
    coords = np.array([
        [idx // grid_size, idx % grid_size]
        for idx in selected_indices
    ])

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(coords, metric='euclidean'))

    # Mean nearest neighbor distance
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self
    nn_distances = dist_matrix.min(axis=1)
    mean_nn_dist = float(nn_distances.mean())

    # Clustering coefficient (inverse of normalized NN distance)
    # Max possible NN distance = diagonal of grid
    max_dist = np.sqrt(2) * grid_size
    clustering_coef = 1.0 - (mean_nn_dist / max_dist)
    clustering_coef = max(0.0, min(1.0, clustering_coef))

    # Estimate number of clusters (simple heuristic: threshold at 2× mean NN distance)
    threshold = 2.0 * mean_nn_dist
    adjacency = dist_matrix < threshold

    # Count connected components (simple DFS)
    visited = set()
    num_clusters = 0

    def dfs(node):
        visited.add(node)
        for neighbor in range(len(selected_indices)):
            if neighbor not in visited and adjacency[node, neighbor]:
                dfs(neighbor)

    for i in range(len(selected_indices)):
        if i not in visited:
            dfs(i)
            num_clusters += 1

    return {
        'mean_nn_distance': mean_nn_dist,
        'clustering_coef': clustering_coef,
        'num_clusters': num_clusters
    }


def visualize_metrics(
    metrics,
    scores,
    selected_indices,
    query="",
    grid_size=32
):
    """
    Create comprehensive metrics visualization.

    Args:
        metrics: Dict from compute_summary_metrics()
        scores: [N] Relevance scores
        selected_indices: [K] Selected indices
        query: Query string
        grid_size: Grid size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    title = "ARR-COC Performance Metrics"
    if query:
        title += f'\nQuery: "{query}"'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Score Distribution (all patches)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(metrics['score_mean'], color='red', linestyle='--', label=f"Mean: {metrics['score_mean']:.3f}")
    ax1.axvline(metrics['score_q50'], color='green', linestyle='--', label=f"Median: {metrics['score_q50']:.3f}")
    ax1.set_xlabel('Relevance Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution (All Patches)', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Selected vs Rejected Scores
    ax2 = fig.add_subplot(gs[0, 1])
    rejected_indices = np.setdiff1d(np.arange(len(scores)), selected_indices)
    selected_scores = scores[selected_indices]
    rejected_scores = scores[rejected_indices]

    box_data = [selected_scores, rejected_scores]
    bp = ax2.boxplot(box_data, labels=['Selected', 'Rejected'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Relevance Score')
    ax2.set_title('Selected vs Rejected Patches', fontweight='bold')
    ax2.grid(alpha=0.3)

    # 3. Spatial Coverage Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    coverage_grid = np.zeros((grid_size, grid_size))
    if len(selected_indices) > 0:
        selected_y = selected_indices // grid_size
        selected_x = selected_indices % grid_size
        coverage_grid[selected_y, selected_x] = 1

    ax3.imshow(coverage_grid, cmap='RdYlGn', interpolation='nearest')
    ax3.set_title(f'Spatial Coverage: {metrics["spatial_coverage"]*100:.1f}%', fontweight='bold')
    ax3.axis('off')

    # 4. Selection Summary Table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    summary_text = f"""
    SELECTION STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━
    Patches Selected: {metrics['num_selected']}
    Selection Ratio: {metrics['selection_ratio']*100:.1f}%
    Compression: {metrics['compression_ratio']:.1f}×

    SCORE STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━
    Mean: {metrics['score_mean']:.3f}
    Std Dev: {metrics['score_std']:.3f}
    Range: [{metrics['score_min']:.3f}, {metrics['score_max']:.3f}]
    Q25/Q50/Q75: {metrics['score_q25']:.3f}/{metrics['score_q50']:.3f}/{metrics['score_q75']:.3f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    # 5. Spatial & Clustering Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    # Interpret clustering coefficient
    if metrics['clustering_coef'] > 0.7:
        cluster_desc = "HIGHLY CLUSTERED"
    elif metrics['clustering_coef'] > 0.4:
        cluster_desc = "MODERATELY CLUSTERED"
    else:
        cluster_desc = "SCATTERED"

    spatial_text = f"""
    SPATIAL DISTRIBUTION
    ━━━━━━━━━━━━━━━━━━━━━━
    Coverage: {metrics['spatial_coverage']*100:.1f}%
    (% of 4×4 regions with patches)

    Entropy: {metrics['spatial_entropy']:.2f} bits
    (uniformity, max=6.0)

    CLUSTERING ANALYSIS
    ━━━━━━━━━━━━━━━━━━━━━━
    Coefficient: {metrics['clustering_coef']:.3f}
    ({cluster_desc})

    Mean NN Distance: {metrics['mean_nn_distance']:.2f} patches
    Num Clusters: ~{metrics['num_clusters']}

    Selected Score Mean: {metrics['selected_score_mean']:.3f}
    """

    ax5.text(0.1, 0.5, spatial_text, fontsize=10, family='monospace',
            verticalalignment='center')

    # 6. Top-K Score Curve
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_scores = np.sort(scores)[::-1]  # Descending
    K_range = np.arange(1, len(sorted_scores) + 1)

    ax6.plot(K_range, sorted_scores, color='blue', linewidth=2)
    ax6.axvline(metrics['num_selected'], color='red', linestyle='--',
               label=f"K={metrics['num_selected']}")
    ax6.axhline(metrics['selected_score_mean'], color='green', linestyle='--',
               label=f"Selected Mean: {metrics['selected_score_mean']:.3f}")
    ax6.set_xlabel('Top-K Patches')
    ax6.set_ylabel('Relevance Score')
    ax6.set_title('Score Decay Curve', fontweight='bold')
    ax6.set_xlim(0, min(500, len(sorted_scores)))
    ax6.legend()
    ax6.grid(alpha=0.3)

    return fig


def test_metrics():
    """Test metrics visualization with synthetic data."""
    print("🧪 Testing 5-metrics microscope...")

    grid_size = 32
    N = grid_size * grid_size

    # Generate synthetic scores (with query-aware pattern)
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    scores = np.exp(-(X**2 + Y**2) / 0.5).flatten()
    scores += np.random.rand(N) * 0.2  # Add noise

    # Select top-200
    K = 200
    selected_indices = np.argsort(scores)[::-1][:K]

    # Compute metrics
    metrics = compute_summary_metrics(
        selected_indices,
        scores,
        query="Where is the object?",
        grid_size=grid_size,
        total_patches=N
    )

    # Visualize
    fig = visualize_metrics(
        metrics,
        scores,
        selected_indices,
        query="Where is the object?",
        grid_size=grid_size
    )

    fig.savefig('test_metrics.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ 5-metrics tests passed!")
    print("   Output saved to: test_metrics.png")
    print(f"   Computed {len(metrics)} metrics")


if __name__ == '__main__':
    test_metrics()
