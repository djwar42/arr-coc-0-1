"""Comparison microscope - Side-by-side model comparisons."""

from .core import (
    compare_patch_selections,
    compare_relevance_heatmaps,
    create_ablation_comparison,
    compare_query_impact,  # NEW: Query impact analysis
)

__all__ = [
    'compare_patch_selections',
    'compare_relevance_heatmaps',
    'create_ablation_comparison',
    'compare_query_impact',
]
