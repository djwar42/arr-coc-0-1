# SA-1B Annotation Fields: IoU, Stability, Prompts

## Overview

Each SA-1B mask has rich annotation fields beyond the mask itself: predicted IoU, stability score, prompt coordinates, and quality metrics. These enable selective loading, quality filtering, and understanding SAM's confidence.

**Key fields:**
- `predicted_iou`: SAM's confidence in mask quality (0-1)
- `stability_score`: Mask consistency under perturbation (0-1)
- `point_coords`: Input prompt coordinates `[[x, y]]`
- `bbox`: Bounding box `[x, y, w, h]`
- `area`: Mask area in pixels

## Field Descriptions

**predicted_iou**: 0-1 confidence
- Predicted by SAM's IoU head
- High values (> 0.9) indicate confident predictions
- Use for quality filtering

**stability_score**: 0-1 robustness
- Measures mask stability under threshold variations
- Computed by SAM during inference
- High values indicate well-defined boundaries

**point_coords**: `[[x, y], ...]`
- Input point prompts used to generate mask
- Multiple points for iterative refinement
- Coordinate system: (0, 0) = top-left

## Python Access

```python
import json

with open("sa_000000.json") as f:
    annotations = json.load(f)

for ann in annotations['annotations']:
    iou = ann['predicted_iou']  # Confidence
    stability = ann['stability_score']  # Robustness
    points = ann['point_coords']  # Prompt

    if iou > 0.9 and stability > 0.95:
        # High-quality mask
        process_mask(ann)
```

## ARR-COC Application

**Quality-aware spatial grounding:**
- Filter masks by confidence for clean training data
- Use stability scores for hard negative mining
- Track prompt coords for multi-round refinement

**Sources**: SA-1B Dataset Card, SAM Paper Appendix
