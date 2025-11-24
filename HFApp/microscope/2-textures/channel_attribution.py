"""
Per-Channel Attribution Analysis - Which texture channels drive relevance?

Compute contribution of each of the 13 texture channels to final relevance scores.

Key analyses:
1. Channel-Relevance Correlation (Pearson/Spearman)
2. Channel Importance Ranking
3. Semantic Group Contribution
4. Per-Patch Channel Activation vs Relevance

Usage:
    attr = compute_channel_attribution(textures, relevance_scores)
    fig = visualize_attribution(attr, textures)
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .semantic_view import SEMANTIC_GROUPS


def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if hasattr(x, 'cpu'):  # PyTorch tensor
        return x.cpu().numpy()
    return np.asarray(x)


CHANNEL_NAMES = [
    'Red (RGB-R)',
    'Green (RGB-G)',
    'Blue (RGB-B)',
    'Lightness (L*)',
    'Green-Red (a*)',
    'Sobel-X (∂x)',
    'Sobel-Y (∂y)',
    'Edge Magnitude',
    'Position-Y',
    'Position-X',
    'Eccentricity',
    'Saliency',
    'Luminance'
]


def compute_channel_attribution(
    textures,  # [1, 13, 32, 32] or numpy [13, 32, 32]
    relevance_scores,  # [1024] or [1, 32, 32]
    method='pearson'
):
    """
    Compute how much each texture channel correlates with relevance scores.

    Args:
        textures: Texture array (torch or numpy)
        relevance_scores: Final relevance scores (torch or numpy)
        method: 'pearson' or 'spearman' correlation

    Returns:
        Dict with:
        - correlations: [13] correlation coefficients
        - p_values: [13] p-values
        - rankings: [13] importance ranking (0=most important)
        - semantic_contributions: Dict of group-level contributions
    """
    # Convert to numpy and flatten
    textures_np = _to_numpy(textures)
    relevance_np = _to_numpy(relevance_scores)

    # Ensure correct shapes
    if textures_np.ndim == 4:
        textures_np = textures_np[0]  # Remove batch dim → [13, H, W]
    if relevance_np.ndim == 2:
        relevance_np = relevance_np.flatten()  # [H, W] → [H*W]
    elif relevance_np.ndim == 3:
        relevance_np = relevance_np[0].flatten()  # [1, H, W] → [H*W]

    num_channels = textures_np.shape[0]
    num_patches = relevance_np.shape[0]

    # Compute correlations for each channel
    correlations = []
    p_values = []

    for ch_idx in range(num_channels):
        channel_flat = textures_np[ch_idx].flatten()

        if method == 'pearson':
            corr, p_val = pearsonr(channel_flat, relevance_np)
        elif method == 'spearman':
            corr, p_val = spearmanr(channel_flat, relevance_np)
        else:
            raise ValueError(f"Unknown method: {method}")

        correlations.append(corr)
        p_values.append(p_val)

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    # Rank by absolute correlation (most important = rank 0)
    rankings = np.argsort(np.abs(correlations))[::-1]

    # Compute semantic group contributions
    semantic_contrib = {}
    for group_name, group_info in SEMANTIC_GROUPS.items():
        ch_indices = group_info['channels']
        group_corrs = correlations[ch_indices]
        # Mean absolute correlation for this group
        semantic_contrib[group_name] = {
            'mean_abs_corr': np.abs(group_corrs).mean(),
            'max_abs_corr': np.abs(group_corrs).max(),
            'channels': ch_indices,
            'correlations': group_corrs
        }

    return {
        'correlations': correlations,
        'p_values': p_values,
        'rankings': rankings,
        'semantic_contributions': semantic_contrib,
        'method': method
    }


def visualize_channel_attribution(
    attribution,
    textures,
    relevance_scores,
    image=None,
    query=""
):
    """
    Visualize per-channel attribution analysis.

    Creates 3-row figure:
    - Row 1: Top-5 channels by importance (heatmaps + correlation)
    - Row 2: Correlation bar chart + semantic group contributions
    - Row 3: Scatter plots (channel value vs relevance) for top-3 channels

    Args:
        attribution: Output from compute_channel_attribution()
        textures: Texture array
        relevance_scores: Relevance scores
        image: Optional original image
        query: Query string

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.4)

    title = "Per-Channel Attribution Analysis"
    if query:
        title += f'\nQuery: "{query}"'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Convert to numpy
    textures_np = _to_numpy(textures)
    if textures_np.ndim == 4:
        textures_np = textures_np[0]

    relevance_np = _to_numpy(relevance_scores)
    if relevance_np.ndim == 2:
        relevance_grid = relevance_np
        relevance_flat = relevance_np.flatten()
    elif relevance_np.ndim == 3:
        relevance_grid = relevance_np[0]
        relevance_flat = relevance_np[0].flatten()
    else:
        H = W = int(np.sqrt(len(relevance_np)))
        relevance_grid = relevance_np.reshape(H, W)
        relevance_flat = relevance_np

    corrs = attribution['correlations']
    rankings = attribution['rankings']

    # === ROW 1: Top-5 Channels ===
    for i in range(5):
        ch_idx = rankings[i]
        ch_name = CHANNEL_NAMES[ch_idx]
        corr = corrs[ch_idx]

        ax = fig.add_subplot(gs[0, i])
        channel_img = textures_np[ch_idx]
        im = ax.imshow(channel_img, cmap='gray', interpolation='bilinear')
        ax.set_title(f"#{i+1}: {ch_name}\nCorr={corr:.3f}", fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Relevance map for reference
    ax_rel = fig.add_subplot(gs[0, 5])
    im_rel = ax_rel.imshow(relevance_grid, cmap='hot', interpolation='bilinear')
    ax_rel.set_title('Relevance Map\n(Reference)', fontsize=9, fontweight='bold')
    ax_rel.axis('off')
    plt.colorbar(im_rel, ax=ax_rel, fraction=0.046)

    # === ROW 2: Bar Chart + Semantic Contributions ===

    # Channel correlation bar chart
    ax_bar = fig.add_subplot(gs[1, :3])
    colors = ['green' if c > 0 else 'red' for c in corrs]
    bars = ax_bar.barh(range(13), corrs, color=colors, alpha=0.7)
    ax_bar.set_yticks(range(13))
    ax_bar.set_yticklabels([CHANNEL_NAMES[i] for i in range(13)], fontsize=8)
    ax_bar.set_xlabel('Correlation with Relevance', fontweight='bold')
    ax_bar.set_title('Channel-Relevance Correlations', fontweight='bold')
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.grid(alpha=0.3, axis='x')

    # Semantic group contributions
    ax_sem = fig.add_subplot(gs[1, 3:])
    sem_contrib = attribution['semantic_contributions']
    group_names = list(sem_contrib.keys())
    group_contribs = [sem_contrib[g]['mean_abs_corr'] for g in group_names]
    group_colors = [SEMANTIC_GROUPS[g]['color'] for g in group_names]

    bars_sem = ax_sem.bar(group_names, group_contribs, color=group_colors, alpha=0.6, edgecolor='black')
    ax_sem.set_ylabel('Mean Absolute Correlation', fontweight='bold')
    ax_sem.set_title('Semantic Group Contributions', fontweight='bold')
    ax_sem.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars_sem, group_contribs):
        height = bar.get_height()
        ax_sem.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # === ROW 3: Scatter Plots (top-3 channels) ===
    for i in range(3):
        ch_idx = rankings[i]
        ch_name = CHANNEL_NAMES[ch_idx]
        corr = corrs[ch_idx]

        ax = fig.add_subplot(gs[2, i])
        channel_flat = textures_np[ch_idx].flatten()

        # Scatter plot: channel value vs relevance
        ax.scatter(channel_flat, relevance_flat, alpha=0.3, s=5, c='blue')

        # Trend line
        z = np.polyfit(channel_flat, relevance_flat, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(channel_flat.min(), channel_flat.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Fit: r={corr:.3f}')

        ax.set_xlabel(f'{ch_name} Value', fontsize=9)
        ax.set_ylabel('Relevance Score', fontsize=9)
        ax.set_title(f'#{i+1}: {ch_name}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Attribution summary (bottom right)
    ax_summary = fig.add_subplot(gs[2, 3:])
    ax_summary.axis('off')

    top3_names = [CHANNEL_NAMES[rankings[i]] for i in range(3)]
    top3_corrs = [corrs[rankings[i]] for i in range(3)]

    summary_text = f"""
    ATTRIBUTION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Method: {attribution['method'].capitalize()} correlation

    Top-3 Channels:
      1. {top3_names[0]}: {top3_corrs[0]:+.3f}
      2. {top3_names[1]}: {top3_corrs[1]:+.3f}
      3. {top3_names[2]}: {top3_corrs[2]:+.3f}

    Most Influential Semantic Group:
      {max(sem_contrib.items(), key=lambda x: x[1]['mean_abs_corr'])[0]}
      (Mean |corr| = {max(group_contribs):.3f})

    Interpretation:
    • Positive correlation → higher value = higher relevance
    • Negative correlation → lower value = higher relevance
    • |corr| > 0.3 → moderate influence
    • |corr| > 0.5 → strong influence
    """

    ax_summary.text(0.5, 0.5, summary_text, fontsize=9, family='monospace',
                   ha='center', va='center')

    return fig


def test_attribution():
    """Test attribution analysis with synthetic data."""
    print("🧪 Testing channel attribution...")

    # Create synthetic textures
    textures = torch.rand(1, 13, 32, 32)

    # Create synthetic relevance (correlate with channel 7 - edge magnitude)
    relevance = textures[0, 7] + 0.3 * torch.rand(32, 32)
    relevance = relevance.flatten()

    # Compute attribution
    attr = compute_channel_attribution(textures, relevance, method='pearson')

    print(f"  Top channel: #{attr['rankings'][0]} (corr={attr['correlations'][attr['rankings'][0]]:.3f})")
    print(f"  Expected: #7 (Edge Magnitude) with high correlation")

    assert attr['correlations'][7] > 0.5, "Edge magnitude should have high correlation"

    # Visualize
    fig = visualize_channel_attribution(attr, textures, relevance, query="Test query")
    fig.savefig('test_attribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ Channel attribution tests passed!")
    print("   Output saved to: test_attribution.png")


if __name__ == '__main__':
    test_attribution()
