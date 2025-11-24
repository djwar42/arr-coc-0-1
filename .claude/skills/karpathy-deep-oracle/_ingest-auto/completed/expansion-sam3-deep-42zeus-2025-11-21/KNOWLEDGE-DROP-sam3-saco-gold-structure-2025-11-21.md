# SA-Co/Gold Dataset Structure

## Overview

SA-Co/Gold is the highest-quality tier of the SA-Co (Segment Anything with Concepts) benchmark for promptable concept segmentation (PCS) in images. It features triple-annotator verification for each image-noun phrase pair, ensuring benchmark-quality annotations for evaluating open-vocabulary segmentation models.

## Dataset Composition

### Image Sources
SA-Co/Gold covers 2 primary image sources:
- **MetaCLIP**: Images from MetaCLIP dataset
- **SA-1B**: Images from the original Segment Anything 1 Billion images dataset

### Seven Annotation Domains

| Domain | Media Source | # Image-NPs | # Image-NP-Masks |
|--------|-------------|-------------|------------------|
| MetaCLIP captioner NPs | MetaCLIP | 33,393 | 20,144 |
| SA-1B captioner NPs | SA-1B | 13,258 | 30,306 |
| Attributes | MetaCLIP | 9,245 | 3,663 |
| Crowded Scenes | MetaCLIP | 20,687 | 50,417 |
| Wiki-Common1K | MetaCLIP | 65,502 | 6,448 |
| Wiki-Food&Drink | MetaCLIP | 13,951 | 9,825 |
| Wiki-Sports Equipment | MetaCLIP | 12,166 | 5,075 |

**Total**: 168,202 unique image-NP pairs with 125,878 annotated masks

### Quality Assurance
- Each image-NP pair is **annotated by 3 different annotators**
- Multi-reviewed and agreed upon annotations
- Used to measure human performance baseline (72.8 cgF1 on SA-Co/Gold)
- Contains both positive NPs (with masks) and negative NPs (no matching objects)

## Annotation Schema

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data) with extensions for open-vocabulary detection.

### Core Data Fields

#### `images` Field
A list of dict features containing all image-NP pairs:

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

**Key fields**:
- `id`: Unique identifier for the image-NP pair (string)
- `text_input`: The noun phrase/text prompt for this pair
- `file_name`: Relative path to the image file
- `width`, `height`: Image dimensions
- `queried_category`: Category identifier (for COCO compatibility)
- `is_instance_exhaustive`: Whether all instances are annotated (1 = yes)
- `is_pixel_exhaustive`: Whether masks are pixel-exhaustive (1 = yes)

#### `annotations` Field
A list of dict features containing bounding boxes and segmentation masks:

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

**Key fields**:
- `id`: Unique annotation identifier
- `image_id`: Maps to `id` in images (the image-NP pair)
- `source`: "manual" for human annotations
- `area`: Normalized area of the mask
- `bbox`: Bounding box in [x, y, w, h] format (normalized)
- `segmentation`: Mask in RLE (Run-Length Encoding) format
  - `counts`: RLE-encoded mask string
  - `size`: [height, width] of the mask
- `category_id`: Category identifier
- `iscrowd`: 0 for instance segmentation

#### `categories` Field
A list of categories for COCO format compatibility:

```json
[
  {
    "id": 1,
    "name": "object"
  }
]
```

**Note**: In open-vocabulary detection, the text prompt is stored directly in each image entry (`text_input`), not in categories. A unique `id` in images corresponds to an (image, text prompt) combination.

### Positive vs Negative Noun Phrases

- **Positive NPs**: Have corresponding annotations (exist as `image_id` in annotations)
  - Objects matching the text prompt are present in the image
  - Contains segmentation masks for all matching instances

- **Negative NPs**: No annotations (do not exist as `image_id` in annotations)
  - No objects match the text prompt in the image
  - Displayed in red font in visualizations
  - Critical for testing false positive rejection

## Hosting Locations

### HuggingFace
**URL**: [https://huggingface.co/datasets/facebook/SACo-Gold](https://huggingface.co/datasets/facebook/SACo-Gold)

**Access Requirements**:
- Must agree to share contact information
- Information collected per Meta Privacy Policy
- Requires HuggingFace login

**Format**: JSON files with images organized in folder structure

**Libraries Supported**:
- HuggingFace Datasets
- pandas
- Croissant

**License**: Other (Meta-specific license)

### Roboflow Universe
**URL**: [https://universe.roboflow.com/sa-co-gold](https://universe.roboflow.com/sa-co-gold)

**Features**:
- Visual dataset exploration
- Can be used with Roboflow annotation tools
- Integration with SAM 3 API endpoint
- Fine-tuning capabilities

## Download Instructions

### Method 1: HuggingFace CLI (Recommended)

```bash
# 1. Install HuggingFace Hub
pip install huggingface_hub

# 2. Login to HuggingFace
huggingface-cli login
# Or: hf auth login

# 3. Accept dataset terms at:
# https://huggingface.co/datasets/facebook/SACo-Gold

# 4. Download using datasets library
from datasets import load_dataset
dataset = load_dataset("facebook/SACo-Gold")
```

### Method 2: Direct Download via Python

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download entire dataset
snapshot_download(
    repo_id="facebook/SACo-Gold",
    repo_type="dataset",
    local_dir="./saco_gold"
)

# Or download specific files
annotation_file = hf_hub_download(
    repo_id="facebook/SACo-Gold",
    filename="annotations.json",
    repo_type="dataset"
)
```

### Method 3: Roboflow Universe

1. Visit [https://universe.roboflow.com/sa-co-gold](https://universe.roboflow.com/sa-co-gold)
2. Create/login to Roboflow account
3. Export dataset in desired format (COCO, YOLO, etc.)

## Usage with SAM 3

### Visualization Example

SAM 3 provides example notebooks for visualizing SA-Co datasets:
- [`saco_gold_silver_vis_example.ipynb`](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)

### Evaluation Scripts

Full evaluation documentation at: [https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/)

```python
# Basic loading pattern
import json

# Load annotations
with open("saco_gold_annotations.json", "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# Create image_id to annotations mapping
from collections import defaultdict
ann_by_image = defaultdict(list)
for ann in annotations:
    ann_by_image[ann["image_id"]].append(ann)

# Process each image-NP pair
for img in images:
    image_id = img["id"]
    text_prompt = img["text_input"]
    file_path = img["file_name"]

    # Get annotations (empty for negative NPs)
    img_anns = ann_by_image.get(image_id, [])

    if img_anns:
        # Positive NP - has matching objects
        for ann in img_anns:
            bbox = ann["bbox"]
            rle_mask = ann["segmentation"]
            # Process mask...
    else:
        # Negative NP - no matching objects
        pass
```

### Decoding RLE Masks

```python
import numpy as np
from pycocotools import mask as mask_utils

def decode_rle(rle_dict):
    """Decode RLE mask to binary numpy array"""
    return mask_utils.decode(rle_dict)

# Example usage
for ann in annotations:
    mask = decode_rle(ann["segmentation"])
    # mask is now a numpy array of shape (height, width)
```

## Evaluation Metrics

SA-Co/Gold is evaluated using:
- **cgF1** (Concept-Grounded F1): Primary metric for PCS
- **AP** (Average Precision): Box detection metric
- **mAP**: Mean Average Precision

### Human Baseline on SA-Co/Gold
- Human cgF1: **72.8** (image segmentation)
- Human cgF1: **74.0** (with box detection)

### SAM 3 Performance
- SAM 3 cgF1: **54.1** (image segmentation)
- SAM 3 cgF1: **55.7** (with box detection)
- Achieves ~75-80% of human performance

## Comparison with Other SA-Co Splits

| Split | Annotators | Domains | Purpose |
|-------|------------|---------|---------|
| **SA-Co/Gold** | 3 per pair | 7 | Benchmark evaluation, human baseline |
| SA-Co/Silver | 1 per pair | 10 | Larger scale evaluation |
| SA-Co/Bronze | Existing | 9 | Pre-existing datasets |
| SA-Co/VEval | Video | Video | Temporal evaluation |

## Key Characteristics

1. **Exhaustive Annotation**: All instances of the queried concept are annotated
2. **Negative Prompts**: Critical test cases where concept doesn't appear
3. **Multi-Review**: Triple annotator agreement ensures quality
4. **Open-Vocabulary**: 270K unique concepts tested (50x more than existing benchmarks)
5. **Instance-Level**: Individual masks for each object instance

## Sources

**Primary Sources**:
- [HuggingFace SA-Co/Gold Dataset](https://huggingface.co/datasets/facebook/SACo-Gold) (accessed 2025-11-23)
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) (accessed 2025-11-23)
- [Roboflow SAM 3 Integration Blog](https://blog.roboflow.com/sam3/) (accessed 2025-11-23)

**Additional References**:
- [SAM 3 Research Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Meta AI SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/)
- [Roboflow Universe SA-Co/Gold](https://universe.roboflow.com/sa-co-gold)
- [COCO Format Documentation](https://cocodataset.org/#format-data)
- [SA-Co Evaluation Scripts](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/)
