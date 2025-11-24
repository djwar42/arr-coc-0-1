# SAM 3 Annotation Format & Tools

## Overview

SAM 3 introduces the SA-Co (Segment Anything with Concepts) dataset with a specialized annotation format derived from COCO JSON. The format supports both positive prompts (with masks) and negative prompts (no matching objects), along with comprehensive tools for visualization and evaluation.

---

## JSON Annotation Format

### Format Structure

The SA-Co annotation format is derived from [COCO format](https://cocodataset.org/#format-data) with extensions for concept-based segmentation:

From [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md):

### Core Data Fields

**`images`** - List of image-NP (Noun Phrase) pairs:
- `id`: Unique identifier for the image-NP pair
- `text_input`: The noun phrase for the image-NP pair
- `file_name`: Relative image path in the data folder
- `height`/`width`: Image dimensions
- `is_instance_exhaustive`: Boolean (0 or 1) - If 1, all instances are correctly annotated
- `is_pixel_exhaustive`: Boolean (0 or 1) - If 1, union of all masks covers all pixels for the prompt
- `queried_category`: Category identifier (for tracking)

**`annotations`** - List of all mask annotations:
- `image_id`: Maps to the identifier for the image-NP pair
- `bbox`: Bounding box in [x, y, w, h] format, normalized by image dimensions
- `segmentation`: Segmentation mask in RLE (Run-Length Encoding) format
- `category_id`: Always 1 (unused in open-vocabulary detection)
- `iscrowd`: Boolean (0 or 1) - If 1, segment overlaps several instances
- `area`: Mask area as fraction of image
- `source`: Annotation source (e.g., "manual")

**`categories`** - Category list for COCO compatibility (unused in open-vocabulary setting)

---

## Example Annotation

### Images Array

```json
[
  {
    "id": 10000000,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "chili",
    "width": 600,
    "height": 600,
    "queried_category": "0",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  },
  {
    "id": 10000001,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "the fish ball",
    "width": 600,
    "height": 600,
    "queried_category": "2001",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  }
]
```

### Annotations Array

```json
[
  {
    "id": 1,
    "image_id": 10000000,
    "source": "manual",
    "area": 0.002477777777777778,
    "bbox": [
      0.44333332777023315,
      0.0,
      0.10833333432674408,
      0.05833333358168602
    ],
    "segmentation": {
      "counts": "`kk42fb01O1O1O1O001O1O1O001O1O00001O1O001O001O0000000000O1001000O010O02O001N10001N0100000O10O1000O10O010O100O1O1O1O1O0000001O0O2O1N2N2Nobm4",
      "size": [600, 600]
    },
    "category_id": 1,
    "iscrowd": 0
  }
]
```

---

## Positive vs Negative Prompts

### Understanding Positive and Negative NPs

From [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md):

**Positive NPs**: Image-NP pairs where `id` in `images` has corresponding annotations (exists as `image_id` in `annotations`)
- Objects matching the noun phrase exist in the image
- Have associated segmentation masks

**Negative NPs**: Image-NP pairs where `id` in `images` has NO corresponding annotations
- Objects matching the noun phrase do NOT exist in the image
- No masks (empty annotation set)
- **Shown in red font in documentation figures**

### Negative Prompt Purpose

Negative prompts serve critical functions:
1. **Train discrimination ability** - Model learns when objects are NOT present
2. **Reduce false positives** - Prevents hallucinating objects
3. **Benchmark hard cases** - Tests model's ability to correctly reject prompts
4. **Real-world scenarios** - Many queries will have no matches

### Code Example for Identifying Positive/Negative

From [visualization notebook](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb):

```python
# Group GT annotations by image_id
gt_image_np_map = {img["id"]: img for _, img in gt_image_np_pairs.iterrows()}
gt_image_np_ann_map = defaultdict(list)
for _, ann in gt_annotations.iterrows():
    image_id = ann["image_id"]
    if image_id not in gt_image_np_ann_map:
        gt_image_np_ann_map[image_id] = []
    gt_image_np_ann_map[image_id].append(ann)

# Separate positive and negative NPs
positiveNPs = [img_id for img_id in gt_image_np_map.keys()
               if img_id in gt_image_np_ann_map and gt_image_np_ann_map[img_id]]
negativeNPs = [img_id for img_id in gt_image_np_map.keys()
               if img_id not in gt_image_np_ann_map or not gt_image_np_ann_map[img_id]]
```

---

## Visualization Tools

### Official Visualization Notebooks

SAM 3 provides several visualization tools in the `examples/` directory:

1. **`saco_gold_silver_vis_example.ipynb`** - Visualize GT annotations
2. **`sam3_data_and_predictions_visualization.ipynb`** - GT annotations and predictions side-by-side
3. **`saco_veval_vis_example.ipynb`** - Video evaluation set visualization

### Core Visualization Utilities

From [visualization notebook](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb):

```python
import sam3.visualization_utils as utils
from pycocotools import mask as mask_util
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Color palette
COLORS = utils.pascal_color_map()[1:]

# Load annotations
annot_file_list = glob(os.path.join(ANNOT_DIR, "*gold*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)

# Decode RLE masks
annot_masks = [mask_util.decode(ann["segmentation"]) for ann in gt_annotation]

# Draw masks on white background
all_masks = utils.draw_masks_to_frame(
    frame=np.ones_like(img)*255,
    masks=annot_masks,
    colors=COLORS[:len(annot_masks)]
)

# Draw masks overlaid on image
masked_frame = utils.draw_masks_to_frame(
    frame=img,
    masks=annot_masks,
    colors=COLORS[:len(annot_masks)]
)
```

### Visualization Functions

**`utils.get_annot_dfs(file_list)`** - Load annotation JSON files into pandas DataFrames

**`utils.draw_masks_to_frame(frame, masks, colors)`** - Draw colored masks on image/background

**`utils.pascal_color_map()`** - Get Pascal VOC color palette for consistent visualization

**`mask_util.decode(rle)`** - Decode RLE segmentation to binary mask (from pycocotools)

---

## Annotation Guidelines

### Quality Tiers

**Gold Dataset** - Multi-reviewed by 3 independent human annotators:
- Allows measurement of human agreement (upper bound for models)
- Annotators may disagree on:
  - Precise mask borders
  - Number of instances
  - Whether phrase exists

**Silver Dataset** - Single annotator with AI assistance

### Special Annotations

**Crowd Segments** (dashed borders in figures):
- Cover more than a single instance
- Used when separating instances is too difficult
- Marked with `iscrowd: 1`

**Exhaustive Annotation Flags**:
- `is_instance_exhaustive: 1` - All instances correctly annotated (use for instance segmentation)
- `is_pixel_exhaustive: 1` - Union of masks covers all relevant pixels (use for semantic segmentation)

### Annotation Domains (7 subsets)

| Domain | Media Source | # Image-NPs | # Masks |
|--------|-------------|-------------|---------|
| MetaCLIP captioner NPs | MetaCLIP | 33,393 | 20,144 |
| SA-1B captioner NPs | SA-1B | 13,258 | 30,306 |
| Attributes | MetaCLIP | 9,245 | 3,663 |
| Crowded Scenes | MetaCLIP | 20,687 | 50,417 |
| Wiki-Common1K | MetaCLIP | 65,502 | 6,448 |
| Wiki-Food/Drink | MetaCLIP | 13,951 | 9,825 |
| Wiki-Sports Equipment | MetaCLIP | 12,166 | 5,075 |

---

## Evaluation Tools

### Official Evaluator

The SAM 3 evaluator inherits from COCO evaluator with modifications:
- Evaluates against all 3 annotations (oracle setting - picks most favorable)
- Minimal dependencies: `pycocotools`, `numpy`, `scipy`

**Primary Metric**: cgF1 (concept-grounded F1)

### Running Evaluation

**Standalone evaluation script**:
```bash
python scripts/eval/standalone_cgf1.py \
  --pred_file /path/to/coco_predictions_segm.json \
  --gt_files /path/to/annotations/gold_metaclip_merged_a_release_test.json \
            /path/to/annotations/gold_metaclip_merged_b_release_test.json \
            /path/to/annotations/gold_metaclip_merged_c_release_test.json
```

**Example notebook**: `saco_gold_silver_eval_example.ipynb`

### Prediction Format

Predictions must be in [COCO result format](https://cocodataset.org/#format-results):
```json
[
  {
    "image_id": 10000000,
    "category_id": 1,
    "segmentation": {"counts": "...", "size": [600, 600]},
    "score": 0.95
  }
]
```

---

## Data Access

### Download Locations

**HuggingFace**:
- [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)
- [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver)
- [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)

**Roboflow**:
- [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold)
- [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver)
- [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

### Image Sources

**MetaCLIP images** (6 subsets):
- Download from [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-metaclip-merged-a-release-test/)

**SA-1B images** (1 subset):
- Download from [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-sa-1b-merged-a-release-test/)
- Or from [Meta SA-1B downloads](https://ai.meta.com/datasets/segment-anything-downloads/)

---

## Key Differences from Standard COCO

| Feature | Standard COCO | SA-Co Format |
|---------|---------------|--------------|
| Text prompts | Not included | `text_input` per image |
| Category usage | Fixed categories | Ignored (open-vocabulary) |
| Negative samples | Not explicit | No annotations = negative |
| Exhaustive flags | Not present | `is_instance_exhaustive`, `is_pixel_exhaustive` |
| Image ID meaning | Single image | Image-NP pair (same image can have multiple IDs) |
| Multi-annotator | Single GT | 3 independent annotations (Gold) |

---

## Best Practices

### Working with SA-Co Annotations

1. **Check exhaustive flags** before evaluation:
   - Instance segmentation: Use only `is_instance_exhaustive: 1`
   - Semantic segmentation: Can use `is_pixel_exhaustive: 1`

2. **Handle negative prompts**:
   - Don't expect all image IDs to have annotations
   - Model should return empty results for negatives

3. **Multiple annotators** (Gold):
   - Evaluate against all 3 and use oracle/average
   - Accounts for annotation disagreement

4. **Crowd segments**:
   - Handle `iscrowd: 1` appropriately in evaluation
   - These are valid annotations but cover multiple instances

5. **Bounding box format**:
   - Normalized [x, y, w, h] format
   - Multiply by image dimensions to get pixel coordinates

---

## Sources

**GitHub Repository:**
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Main repository
- [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md) - Annotation format details
- [Visualization notebook](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb) - Code examples

**Web Research:**
- [Roboflow SAM 3 Guide](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23)
- [Meta AI SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/) (accessed 2025-11-23)

**Dataset Hosting:**
- [HuggingFace SA-Co Datasets](https://huggingface.co/datasets/facebook/SACo-Gold)
- [Roboflow SA-Co Universe](https://universe.roboflow.com/sa-co-gold)
