"""
Comparison Visualization - Side-by-side model/variant comparison

Compare multiple models, checkpoints, or configurations:
- ARR-COC vs Baseline (uniform random)
- Different tension weights
- Fixed vs adaptive balancing
- Ablation studies (turn off scorers)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def compare_patch_selections(
    selections_dict,
    image,
    query="",
    grid_size=32
):
    """
    Compare patch selections from multiple models/variants side-by-side.

    Args:
        selections_dict: Dict of {name: selected_indices} where selected_indices is [K] array
        image: PIL Image or numpy array
        query: Query string
        grid_size: Grid dimension (32 for 32×32 patches)

    Returns:
        matplotlib Figure
    """
    n_variants = len(selections_dict)

    # Create figure with n_variants columns
    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, 6))
    if n_variants == 1:
        axes = [axes]

    # Title
    title = "Patch Selection Comparison"
    if query:
        title += f'\nQuery: "{query}"'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Convert image to numpy if needed
    if hasattr(image, 'convert'):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image

    for idx, (name, selected_indices) in enumerate(selections_dict.items()):
        ax = axes[idx]

        # Create selection grid
        selection_grid = np.zeros((grid_size, grid_size))
        if len(selected_indices) > 0:
            selected_y = selected_indices // grid_size
            selected_x = selected_indices % grid_size
            selection_grid[selected_y, selected_x] = 1

        # Overlay on image
        overlay = img_array.copy().astype(float)
        patch_h = img_array.shape[0] // grid_size
        patch_w = img_array.shape[1] // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                if selection_grid[i, j] == 0:
                    # Dim rejected patches
                    overlay[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] *= 0.3

        # Draw
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f'{name}\n({len(selected_indices)} patches)', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    return fig


def compare_relevance_heatmaps(
    scores_dict,
    image,
    query="",
    grid_size=32
):
    """
    Compare relevance score heatmaps from multiple variants.

    Args:
        scores_dict: Dict of {name: scores} where scores is [N] array
        image: PIL Image or numpy array
        query: Query string
        grid_size: Grid dimension

    Returns:
        matplotlib Figure
    """
    n_variants = len(scores_dict)

    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, 6))
    if n_variants == 1:
        axes = [axes]

    title = "Relevance Score Comparison"
    if query:
        title += f'\nQuery: "{query}"'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Find global min/max for consistent colorbar
    all_scores = np.concatenate([s for s in scores_dict.values()])
    vmin, vmax = all_scores.min(), all_scores.max()

    for idx, (name, scores) in enumerate(scores_dict.items()):
        ax = axes[idx]

        # Reshape to grid
        score_grid = scores.reshape(grid_size, grid_size)

        # Show heatmap
        im = ax.imshow(score_grid, cmap='hot', interpolation='bilinear', vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_ablation_comparison(
    full_scores,
    ablation_scores_dict,
    image,
    query="",
    grid_size=32
):
    """
    Show ablation study: full model vs variants with components removed.

    Args:
        full_scores: [N] Full model scores
        ablation_scores_dict: Dict of {ablation_name: scores}
        image: Image
        query: Query
        grid_size: Grid size

    Returns:
        matplotlib Figure
    """
    # Add full model to dict
    all_scores = {'Full Model': full_scores}
    all_scores.update(ablation_scores_dict)

    # Create comparison
    fig = compare_relevance_heatmaps(all_scores, image, query, grid_size)

    # Update title to indicate ablation study
    fig.suptitle(f'Ablation Study\nQuery: "{query}"', fontsize=14, fontweight='bold')

    # Add performance metrics
    metrics_text = "Relative Performance:\n"
    full_mean = full_scores.mean()

    for name, scores in ablation_scores_dict.items():
        change = ((scores.mean() - full_mean) / full_mean) * 100
        metrics_text += f"{name}: {change:+.1f}%\n"

    fig.text(0.98, 0.02, metrics_text,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


def compare_query_impact(
    info_scores,
    persp_scores,
    partic_scores,
    balanced_scores,
    image,
    query="",
    grid_size=32,
    K=200
):
    """
    Show query impact: how much does participatory knowing (query coupling) shift selection?

    Args:
        info_scores: [N] Information (Propositional) scores
        persp_scores: [N] Perspectival (Salience) scores
        partic_scores: [N] Participatory (Query) scores
        balanced_scores: [N] Final balanced scores (with query)
        image: PIL Image
        query: Query string
        grid_size: Grid size
        K: Number of patches to select

    Returns:
        matplotlib Figure showing:
        - No Query baseline (info + persp only)
        - With Query (full ARR-COC)
        - Difference map
        - Metrics on query impact
    """
    # Compute "no query" baseline (equal weight info + persp, no partic)
    no_query_scores = 0.5 * info_scores + 0.5 * persp_scores

    # Select top-K for both
    no_query_selected = np.argsort(no_query_scores)[::-1][:K]
    with_query_selected = np.argsort(balanced_scores)[::-1][:K]

    # Compute metrics
    overlap = len(np.intersect1d(no_query_selected, with_query_selected))
    overlap_pct = (overlap / K) * 100
    shift_pct = 100 - overlap_pct

    # Create figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[3, 1])

    fig.suptitle(f'Query Impact Analysis\nQuery: "{query}"', fontsize=14, fontweight='bold')

    # Convert image to numpy if needed
    if hasattr(image, 'convert'):
        img_array = np.array(image.resize((256, 256)).convert('RGB'))
    else:
        img_array = image

    patch_h = 256 // grid_size
    patch_w = 256 // grid_size

    # 1. No Query Selection
    ax1 = fig.add_subplot(gs[0, 0])
    overlay1 = img_array.copy().astype(float)
    no_query_grid = np.zeros((grid_size, grid_size))
    no_query_y = no_query_selected // grid_size
    no_query_x = no_query_selected % grid_size
    no_query_grid[no_query_y, no_query_x] = 1

    for i in range(grid_size):
        for j in range(grid_size):
            if no_query_grid[i, j] == 0:
                overlay1[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] *= 0.3

    ax1.imshow(overlay1.astype(np.uint8))
    ax1.set_title('NO QUERY\n(Info + Persp only)', fontweight='bold')
    ax1.axis('off')

    # 2. With Query Selection
    ax2 = fig.add_subplot(gs[0, 1])
    overlay2 = img_array.copy().astype(float)
    with_query_grid = np.zeros((grid_size, grid_size))
    with_query_y = with_query_selected // grid_size
    with_query_x = with_query_selected % grid_size
    with_query_grid[with_query_y, with_query_x] = 1

    for i in range(grid_size):
        for j in range(grid_size):
            if with_query_grid[i, j] == 0:
                overlay2[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] *= 0.3

    ax2.imshow(overlay2.astype(np.uint8))
    ax2.set_title('WITH QUERY\n(Full ARR-COC)', fontweight='bold')
    ax2.axis('off')

    # 3. Difference Map (what changed?)
    ax3 = fig.add_subplot(gs[0, 2])
    diff_map = np.zeros((grid_size, grid_size))
    # Red = removed by query, Green = added by query, White = same
    for idx in no_query_selected:
        y, x = idx // grid_size, idx % grid_size
        if idx not in with_query_selected:
            diff_map[y, x] = -1  # Removed (show red)
    for idx in with_query_selected:
        y, x = idx // grid_size, idx % grid_size
        if idx not in no_query_selected:
            diff_map[y, x] = 1  # Added (show green)

    im3 = ax3.imshow(diff_map, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
    ax3.set_title('DIFFERENCE\n(Red=removed, Green=added)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 4. Score Heatmap Difference
    ax4 = fig.add_subplot(gs[0, 3])
    score_diff = (balanced_scores - no_query_scores).reshape(grid_size, grid_size)
    im4 = ax4.imshow(score_diff, cmap='RdBu_r', interpolation='bilinear')
    ax4.set_title('SCORE SHIFT\n(Query effect)', fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # 5. Metrics Panel
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')

    metrics_text = f"""
    QUERY IMPACT METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Overlap: {overlap}/{K} patches ({overlap_pct:.1f}%)  |  Selection Shift: {shift_pct:.1f}%  |  Query Changed: {K - overlap} patches

    Score Statistics:
      No Query Mean: {no_query_scores.mean():.3f}  |  With Query Mean: {balanced_scores.mean():.3f}  |  Mean Shift: {(balanced_scores - no_query_scores).mean():+.3f}
      No Query Std: {no_query_scores.std():.3f}   |  With Query Std: {balanced_scores.std():.3f}   |  Max Shift: {np.abs(balanced_scores - no_query_scores).max():.3f}

    Participatory Contribution: {partic_scores.mean():.3f} mean score  |  Query pulls attention to query-relevant regions
    """

    ax5.text(0.5, 0.5, metrics_text, fontsize=10, family='monospace', ha='center', va='center')

    return fig


def test_comparison():
    """Test the comparison visualization with synthetic data."""
    print("🧪 Testing 4-comparison microscope...")

    grid_size = 32
    N = grid_size * grid_size

    # Create synthetic variants
    # Baseline: uniform random
    baseline_scores = np.random.rand(N)

    # ARR-COC: query-aware (center focus)
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    arr_coc_scores = np.exp(-(X**2 + Y**2) / 0.3).flatten()

    # Adaptive: combination
    adaptive_scores = 0.6 * arr_coc_scores + 0.4 * baseline_scores

    # Compare relevance heatmaps
    scores_dict = {
        'Baseline (Uniform)': baseline_scores,
        'ARR-COC': arr_coc_scores,
        'Adaptive': adaptive_scores
    }

    # Create synthetic image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    fig = compare_relevance_heatmaps(
        scores_dict,
        img,
        query="Where is the object?",
        grid_size=grid_size
    )

    fig.savefig('test_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ 4-comparison tests passed!")
    print("   Output saved to: test_comparison.png")


if __name__ == '__main__':
    test_comparison()
