"""
Microscope - Visualization toolkit for ARR-COC development

Numbered auto-loading visualization components.
New microscopes: just add 6-name/, 7-name/, etc. with core.py inside.
"""

import importlib

# 0-homunculus
try:
    _0 = importlib.import_module('microscope.0-homunculus.core')
    draw_homunculus = _0.draw_homunculus
    create_homunculus_figure = _0.create_homunculus_figure
    create_homunculus_grid = _0.create_homunculus_grid
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 0-homunculus: {e}")
    traceback.print_exc()
    draw_homunculus = create_homunculus_figure = create_homunculus_grid = None

# 1-heatmaps
try:
    _1 = importlib.import_module('microscope.1-heatmaps.core')
    draw_heatmap = _1.draw_heatmap
    create_heatmap_figure = _1.create_heatmap_figure
    create_multi_heatmap_figure = _1.create_multi_heatmap_figure
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 1-heatmaps: {e}")
    traceback.print_exc()
    draw_heatmap = create_heatmap_figure = create_multi_heatmap_figure = None

# 2-textures
try:
    _2a = importlib.import_module('microscope.2-textures.channel_grid')
    _2b = importlib.import_module('microscope.2-textures.false_color')
    _2c = importlib.import_module('microscope.2-textures.semantic_view')
    _2d = importlib.import_module('microscope.2-textures.channel_attribution')
    create_channel_grid = _2a.create_channel_grid
    visualize_all_channels = _2a.visualize_all_channels
    apply_false_color = _2b.apply_false_color
    create_false_color_composite = _2b.create_false_color_composite
    visualize_by_meaning = _2c.visualize_by_meaning
    create_semantic_groups = _2c.create_semantic_groups
    compute_channel_attribution = _2d.compute_channel_attribution
    visualize_channel_attribution = _2d.visualize_channel_attribution
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 2-textures: {e}")
    traceback.print_exc()
    create_channel_grid = visualize_all_channels = apply_false_color = None
    create_false_color_composite = visualize_by_meaning = create_semantic_groups = None
    compute_channel_attribution = visualize_channel_attribution = None

# 3-three-ways
try:
    _3 = importlib.import_module('microscope.3-three-ways.core')
    visualize_three_ways = _3.visualize_three_ways
    create_three_ways_figure = _3.create_three_ways_figure
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 3-three-ways: {e}")
    traceback.print_exc()
    visualize_three_ways = create_three_ways_figure = None

# 4-comparison
try:
    _4 = importlib.import_module('microscope.4-comparison.core')
    compare_patch_selections = _4.compare_patch_selections
    compare_relevance_heatmaps = _4.compare_relevance_heatmaps
    create_ablation_comparison = _4.create_ablation_comparison
    compare_query_impact = _4.compare_query_impact  # NEW: Query impact analysis
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 4-comparison: {e}")
    traceback.print_exc()
    compare_patch_selections = compare_relevance_heatmaps = create_ablation_comparison = None
    compare_query_impact = None

# 5-metrics
try:
    _5 = importlib.import_module('microscope.5-metrics.core')
    compute_summary_metrics = _5.compute_summary_metrics
    visualize_metrics = _5.visualize_metrics
except Exception as e:
    import traceback
    print(f"⚠️ Failed to import 5-metrics: {e}")
    traceback.print_exc()
    compute_summary_metrics = visualize_metrics = None

__all__ = [
    'draw_homunculus', 'create_homunculus_figure', 'create_homunculus_grid',
    'draw_heatmap', 'create_heatmap_figure', 'create_multi_heatmap_figure',
    'create_channel_grid', 'visualize_all_channels', 'apply_false_color',
    'create_false_color_composite', 'visualize_by_meaning', 'create_semantic_groups',
    'compute_channel_attribution', 'visualize_channel_attribution',  # NEW
    'visualize_three_ways', 'create_three_ways_figure',
    'compare_patch_selections', 'compare_relevance_heatmaps', 'create_ablation_comparison',
    'compare_query_impact',  # NEW
    'compute_summary_metrics', 'visualize_metrics',
]
