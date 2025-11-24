"""
Three Ways of Knowing Visualization - Vervaekean scorer breakdown

Shows how the three ways of knowing contribute to relevance realization:
- Propositional (Information - entropy, complexity)
- Perspectival (Salience - edges, what stands out)
- Participatory (Query coupling - what matches the question)

The balanced combination shows how opponent processing navigates tensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_three_ways(
    info_scores,
    persp_scores,
    partic_scores,
    balanced_scores,
    image=None,
    query="",
    grid_size=32
):
    """
    Visualize the three ways of knowing breakdown.

    Args:
        info_scores: [N] Information scores (Propositional)
        persp_scores: [N] Perspectival scores (Salience)
        partic_scores: [N] Participatory scores (Query coupling)
        balanced_scores: [N] Balanced combined scores
        image: Optional PIL Image or numpy array
        query: Query string
        grid_size: Grid dimension (32 for 32×32 patches)

    Returns:
        matplotlib Figure
    """
    # Reshape scores to grid
    info_grid = info_scores.reshape(grid_size, grid_size)
    persp_grid = persp_scores.reshape(grid_size, grid_size)
    partic_grid = partic_scores.reshape(grid_size, grid_size)
    balanced_grid = balanced_scores.reshape(grid_size, grid_size)

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Add title
    title = "Three Ways of Knowing Breakdown"
    if query:
        title += f'\nQuery: "{query}"'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Propositional (Information)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(info_grid, cmap='hot', interpolation='nearest')
    ax1.set_title('Propositional Knowing\n(Information Content)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Complexity, Entropy, Structure')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Perspectival (Salience)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(persp_grid, cmap='viridis', interpolation='nearest')
    ax2.set_title('Perspectival Knowing\n(Salience)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Edges, Contrast, What Stands Out')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Participatory (Query Coupling)
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(partic_grid, cmap='plasma', interpolation='nearest')
    ax3.set_title('Participatory Knowing\n(Query Coupling)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Agent-Arena Coupling, Relevance')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Balanced (Opponent Processing)
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(balanced_grid, cmap='inferno', interpolation='nearest')
    ax4.set_title('Balanced Relevance\n(Opponent Processing)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Adaptive Tension Navigation')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    return fig


def create_three_ways_figure(
    info_scores,
    persp_scores,
    partic_scores,
    balanced_scores,
    image=None,
    query="",
    show_weights=True
):
    """
    Create comprehensive three ways visualization with optional weight display.

    Args:
        info_scores: [N] Information scores
        persp_scores: [N] Perspectival scores
        partic_scores: [N] Participatory scores
        balanced_scores: [N] Balanced scores
        image: Optional image
        query: Query string
        show_weights: Show learned weights if available

    Returns:
        matplotlib Figure
    """
    fig = visualize_three_ways(
        info_scores,
        persp_scores,
        partic_scores,
        balanced_scores,
        image,
        query
    )

    # Add weight indicators if requested
    if show_weights:
        # Calculate effective weights from contribution analysis
        info_contrib = np.mean(info_scores / (balanced_scores + 1e-8))
        persp_contrib = np.mean(persp_scores / (balanced_scores + 1e-8))
        partic_contrib = np.mean(partic_scores / (balanced_scores + 1e-8))

        total = info_contrib + persp_contrib + partic_contrib
        if total > 0:
            info_pct = 100 * info_contrib / total
            persp_pct = 100 * persp_contrib / total
            partic_pct = 100 * partic_contrib / total

            weights_text = (
                f"Effective Contribution:\n"
                f"Information: {info_pct:.1f}%\n"
                f"Perspectival: {persp_pct:.1f}%\n"
                f"Participatory: {partic_pct:.1f}%"
            )

            fig.text(0.98, 0.02, weights_text,
                    fontsize=10, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


def test_three_ways():
    """Test the three ways visualization with synthetic data."""
    print("🧪 Testing 3-three-ways microscope...")

    # Create synthetic scores
    grid_size = 32
    N = grid_size * grid_size

    # Information: random complexity
    info_scores = np.random.rand(N)

    # Perspectival: edge pattern
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    persp_scores = (np.abs(X) + np.abs(Y)).flatten()
    persp_scores = persp_scores / persp_scores.max()

    # Participatory: center focus
    partic_scores = np.exp(-(X**2 + Y**2) / 0.3).flatten()

    # Balanced: combination
    balanced_scores = 0.3 * info_scores + 0.3 * persp_scores + 0.4 * partic_scores

    # Create visualization
    fig = create_three_ways_figure(
        info_scores,
        persp_scores,
        partic_scores,
        balanced_scores,
        query="Where is the object?",
        show_weights=True
    )

    # Save test output
    fig.savefig('test_three_ways.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ 3-three-ways tests passed!")
    print("   Output saved to: test_three_ways.png")


if __name__ == '__main__':
    test_three_ways()
