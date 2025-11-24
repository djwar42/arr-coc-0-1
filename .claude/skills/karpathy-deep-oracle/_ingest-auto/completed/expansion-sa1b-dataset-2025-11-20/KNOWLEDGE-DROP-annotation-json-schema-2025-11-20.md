# SA-1B Annotation JSON Schema: Complete Structure Guide

## Overview

SA-1B uses a COCO-compatible JSON annotation format that provides structured metadata for each of the 1.1 billion segmentation masks. Understanding this schema is essential for loading, processing, and training on SA-1B data. Each annotation file corresponds to a single image and contains all mask annotations with quality metrics, prompts, and RLE-encoded segmentations.

---

## 1. Top-Level JSON Structure

### File Organization

Each SA-1B image has a corresponding JSON annotation file:
- **Image file**: `sa_XXXXXX.jpg` (e.g., `sa_000001.jpg`)
- **Annotation file**: `sa_XXXXXX.json` (e.g., `sa_000001.json`)

### Root Structure

```json
{
    "image": {
        "image_id": 1,
        "width": 2250,
        "height": 1500,
        "file_name": "sa_000001.jpg"
    },
    "annotations": [
        {
            "id": 1,
            "segmentation": {
                "size": [1500, 2250],
                "counts": "encoded_rle_string"
            },
            "bbox": [x, y, width, height],
            "area": 12345,
            "predicted_iou": 0.95,
            "stability_score": 0.98,
            "point_coords": [[x1, y1]],
            "crop_box": [0, 0, 2250, 1500]
        }
    ]
}
```

---

## 2. Image Metadata Fields

### image_id (Integer)

Unique identifier for the image across the entire dataset:
- Range: 1 to ~11,000,000
- Globally unique across all 1,000 tar files
- Used to link annotations to images

```python
# Example usage
image_id = annotation_data["image"]["image_id"]
print(f"Processing image: {image_id}")
```

### width and height (Integers)

Image dimensions in pixels:
- Variable resolution (not fixed 1500x2250)
- Average resolution: ~1500x2250
- Important for mask decoding

```python
width = annotation_data["image"]["width"]
height = annotation_data["image"]["height"]
# Used for RLE decoding
mask_shape = (height, width)
```

### file_name (String)

Filename of the corresponding image:
- Format: `sa_XXXXXX.jpg`
- 6-digit zero-padded number
- Matches annotation filename (different extension)

```python
image_path = os.path.join(
    image_dir,
    annotation_data["image"]["file_name"]
)
```

---

## 3. Annotation Fields (Per-Mask)

Each image contains multiple annotations (average ~100 per image).

### id (Integer)

Unique identifier for this specific mask annotation:
- Unique within the annotation file
- Sequential numbering starting from 1
- Used for tracking and referencing masks

```python
for ann in annotation_data["annotations"]:
    mask_id = ann["id"]
    print(f"Processing mask {mask_id}")
```

### segmentation (Object)

COCO Run-Length Encoding (RLE) of the binary mask:

```json
"segmentation": {
    "size": [height, width],
    "counts": "RLE_encoded_string"
}
```

**Fields:**
- `size`: [height, width] tuple for mask dimensions
- `counts`: RLE-encoded string (compressed mask data)

**Decoding with pycocotools:**

```python
from pycocotools import mask as mask_utils

# Decode RLE to binary mask
rle = ann["segmentation"]
binary_mask = mask_utils.decode(rle)
# binary_mask shape: (height, width) with 0/1 values
```

### bbox (List of 4 Floats)

Bounding box coordinates in COCO format [x, y, width, height]:
- `x`: Left coordinate (pixels from left edge)
- `y`: Top coordinate (pixels from top edge)
- `width`: Box width in pixels
- `height`: Box height in pixels

```python
x, y, w, h = ann["bbox"]
# Convert to corner format if needed
x1, y1, x2, y2 = x, y, x + w, y + h
```

**Note:** This is NOT [x1, y1, x2, y2] format - it's [x, y, width, height].

### area (Float)

Total pixel area of the segmentation mask:
- Computed as sum of all foreground pixels
- Can be used for filtering by size
- Important for mask granularity analysis

```python
# Filter small masks
MIN_AREA = 100
large_masks = [
    ann for ann in annotations
    if ann["area"] >= MIN_AREA
]
```

### predicted_iou (Float)

Model's confidence in mask quality (0.0 to 1.0):
- SAM's self-assessed IoU prediction
- Higher values indicate higher confidence
- Useful for quality filtering

```python
# Filter high-quality masks
HIGH_QUALITY_THRESHOLD = 0.88
quality_masks = [
    ann for ann in annotations
    if ann["predicted_iou"] >= HIGH_QUALITY_THRESHOLD
]
```

### stability_score (Float)

Mask consistency under prompt perturbation (0.0 to 1.0):
- Measures robustness to input variations
- Higher scores indicate more stable masks
- Critical for filtering ambiguous regions

```python
# Filter stable masks
STABILITY_THRESHOLD = 0.95
stable_masks = [
    ann for ann in annotations
    if ann["stability_score"] >= STABILITY_THRESHOLD
]
```

### point_coords (List of Lists)

Original point prompts used to generate the mask:
- Format: [[x1, y1], [x2, y2], ...]
- Usually single point for SA-1B automatic generation
- Coordinates in pixel space

```python
points = ann["point_coords"]
for x, y in points:
    print(f"Prompt point at ({x}, {y})")
```

### crop_box (List of 4 Floats)

Region used for mask generation [x, y, width, height]:
- Usually full image: [0, 0, width, height]
- May be smaller for cropped generation
- Same format as bbox

```python
crop_x, crop_y, crop_w, crop_h = ann["crop_box"]
# Check if full image was used
is_full_image = (
    crop_x == 0 and crop_y == 0 and
    crop_w == image_width and crop_h == image_height
)
```

---

## 4. COCO Format Compatibility

SA-1B annotations are COCO-compatible but simplified:

### Differences from Standard COCO

| Feature | COCO | SA-1B |
|---------|------|-------|
| Class labels | 80+ categories | None (class-agnostic) |
| category_id | Required | Not present |
| iscrowd | 0 or 1 | Not present |
| Captions | Available | Not present |
| Keypoints | Available | Not present |

### Converting to Standard COCO Format

```python
def sa1b_to_coco(sa1b_annotation, category_id=1):
    """Convert SA-1B annotation to standard COCO format."""
    return {
        "id": sa1b_annotation["id"],
        "image_id": sa1b_annotation.get("image_id"),
        "category_id": category_id,  # Add default category
        "segmentation": sa1b_annotation["segmentation"],
        "bbox": sa1b_annotation["bbox"],
        "area": sa1b_annotation["area"],
        "iscrowd": 0  # SA-1B masks are never crowd
    }
```

### Building COCO Dataset Structure

```python
def build_coco_dataset(sa1b_files):
    """Build complete COCO-format dataset from SA-1B files."""
    coco_dataset = {
        "info": {
            "description": "SA-1B Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Meta AI"
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "none"}
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1
    for sa1b_file in sa1b_files:
        with open(sa1b_file) as f:
            data = json.load(f)

        # Add image info
        coco_dataset["images"].append({
            "id": data["image"]["image_id"],
            "file_name": data["image"]["file_name"],
            "width": data["image"]["width"],
            "height": data["image"]["height"]
        })

        # Add annotations
        for ann in data["annotations"]:
            coco_ann = {
                "id": annotation_id,
                "image_id": data["image"]["image_id"],
                "category_id": 1,
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(coco_ann)
            annotation_id += 1

    return coco_dataset
```

---

## 5. Complete Loading Example

### Basic JSON Loading

```python
import json
import os
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image

def load_sa1b_sample(image_path, annotation_path):
    """Load SA-1B image and annotations."""
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Load annotations
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # Verify dimensions match
    img_info = annotation_data["image"]
    assert image_array.shape[0] == img_info["height"]
    assert image_array.shape[1] == img_info["width"]

    return image_array, annotation_data

# Example usage
image_dir = "sa_000000"
ann_dir = "sa_000000"

image_path = os.path.join(image_dir, "sa_000001.jpg")
ann_path = os.path.join(ann_dir, "sa_000001.json")

image, annotations = load_sa1b_sample(image_path, ann_path)
print(f"Image shape: {image.shape}")
print(f"Number of masks: {len(annotations['annotations'])}")
```

### Decoding All Masks

```python
def decode_all_masks(annotation_data):
    """Decode all RLE masks to binary arrays."""
    masks = []
    for ann in annotation_data["annotations"]:
        rle = ann["segmentation"]
        binary_mask = mask_utils.decode(rle)
        masks.append({
            "id": ann["id"],
            "mask": binary_mask,
            "bbox": ann["bbox"],
            "area": ann["area"],
            "predicted_iou": ann["predicted_iou"],
            "stability_score": ann["stability_score"]
        })
    return masks

masks = decode_all_masks(annotations)
print(f"Decoded {len(masks)} masks")
```

### Quality Filtering

```python
def filter_quality_masks(annotations,
                         min_iou=0.88,
                         min_stability=0.95,
                         min_area=100):
    """Filter masks by quality metrics."""
    filtered = []
    for ann in annotations["annotations"]:
        if (ann["predicted_iou"] >= min_iou and
            ann["stability_score"] >= min_stability and
            ann["area"] >= min_area):
            filtered.append(ann)
    return filtered

quality_masks = filter_quality_masks(annotations)
print(f"Quality masks: {len(quality_masks)}/{len(annotations['annotations'])}")
```

---

## 6. Validation and Error Handling

### Schema Validation

```python
def validate_sa1b_annotation(data):
    """Validate SA-1B annotation structure."""
    errors = []

    # Check required top-level keys
    if "image" not in data:
        errors.append("Missing 'image' key")
    if "annotations" not in data:
        errors.append("Missing 'annotations' key")

    # Validate image metadata
    if "image" in data:
        img = data["image"]
        required_img = ["image_id", "width", "height", "file_name"]
        for field in required_img:
            if field not in img:
                errors.append(f"Missing image.{field}")

    # Validate annotations
    if "annotations" in data:
        for i, ann in enumerate(data["annotations"]):
            required_ann = [
                "id", "segmentation", "bbox", "area",
                "predicted_iou", "stability_score"
            ]
            for field in required_ann:
                if field not in ann:
                    errors.append(f"Annotation {i}: missing {field}")

            # Validate bbox format
            if "bbox" in ann:
                if len(ann["bbox"]) != 4:
                    errors.append(f"Annotation {i}: invalid bbox length")

            # Validate segmentation
            if "segmentation" in ann:
                seg = ann["segmentation"]
                if "size" not in seg or "counts" not in seg:
                    errors.append(f"Annotation {i}: invalid segmentation")

    return errors

# Usage
errors = validate_sa1b_annotation(annotation_data)
if errors:
    print("Validation errors:", errors)
else:
    print("Annotation is valid")
```

### Robust Loading

```python
def safe_load_annotation(annotation_path):
    """Safely load SA-1B annotation with error handling."""
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        # Validate structure
        errors = validate_sa1b_annotation(data)
        if errors:
            return None, errors

        return data, None

    except json.JSONDecodeError as e:
        return None, [f"JSON decode error: {e}"]
    except FileNotFoundError:
        return None, [f"File not found: {annotation_path}"]
    except Exception as e:
        return None, [f"Unexpected error: {e}"]
```

---

## 7. Batch Processing Patterns

### Processing Multiple Files

```python
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_annotation_file(ann_path):
    """Process single annotation file."""
    data, errors = safe_load_annotation(ann_path)
    if errors:
        return {"path": ann_path, "error": errors}

    # Extract statistics
    num_masks = len(data["annotations"])
    avg_iou = np.mean([
        a["predicted_iou"] for a in data["annotations"]
    ])
    avg_stability = np.mean([
        a["stability_score"] for a in data["annotations"]
    ])

    return {
        "path": ann_path,
        "image_id": data["image"]["image_id"],
        "num_masks": num_masks,
        "avg_iou": avg_iou,
        "avg_stability": avg_stability
    }

def batch_process(annotation_dir, num_workers=8):
    """Process all annotations in directory."""
    ann_files = glob.glob(os.path.join(annotation_dir, "*.json"))

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_annotation_file, f)
            for f in ann_files
        ]

        for future in tqdm(futures, desc="Processing"):
            results.append(future.result())

    return results
```

### Streaming from Tar Files

```python
import tarfile
import io

def stream_annotations_from_tar(tar_path):
    """Stream annotations directly from tar file."""
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.json'):
                f = tar.extractfile(member)
                if f:
                    data = json.load(io.TextIOWrapper(f, encoding='utf-8'))
                    yield data

# Usage
for annotation in stream_annotations_from_tar("sa_000000.tar"):
    image_id = annotation["image"]["image_id"]
    num_masks = len(annotation["annotations"])
    # Process without extracting to disk
```

---

## 8. ARR-COC-0-1 Integration: Structured Annotations for Spatial Grounding

### Relevance for VLM Training

SA-1B's structured JSON annotations provide essential components for ARR-COC spatial relevance training:

1. **Precise Spatial Information**: bbox and segmentation fields enable exact object localization
2. **Quality Metrics**: predicted_iou and stability_score allow filtering for high-confidence masks
3. **Prompt Tracking**: point_coords shows where attention was directed
4. **Hierarchical Granularity**: Multiple masks per image at different granularity levels

### Integration Patterns

```python
def prepare_spatial_grounding_sample(annotation_data, relevance_threshold=0.9):
    """Prepare SA-1B sample for ARR-COC spatial grounding training."""

    # Filter high-relevance masks
    quality_masks = [
        ann for ann in annotation_data["annotations"]
        if ann["predicted_iou"] >= relevance_threshold
        and ann["stability_score"] >= relevance_threshold
    ]

    # Convert to spatial grounding format
    spatial_annotations = []
    for mask in quality_masks:
        x, y, w, h = mask["bbox"]
        spatial_annotations.append({
            "bbox_normalized": [
                x / annotation_data["image"]["width"],
                y / annotation_data["image"]["height"],
                w / annotation_data["image"]["width"],
                h / annotation_data["image"]["height"]
            ],
            "area_normalized": mask["area"] / (
                annotation_data["image"]["width"] *
                annotation_data["image"]["height"]
            ),
            "confidence": mask["predicted_iou"],
            "stability": mask["stability_score"],
            "prompt_location": mask.get("point_coords", [])
        })

    return spatial_annotations
```

---

## Sources

**Web Research:**
- [V7 Labs COCO Dataset Guide](https://www.v7labs.com/blog/coco-dataset-guide) - Comprehensive COCO JSON format documentation (accessed 2025-11-20)
- [Roboflow COCO JSON Format](https://roboflow.com/formats/coco-json) - COCO format description and structure (accessed 2025-11-20)
- [GitHub segment-anything Issue #291](https://github.com/facebookresearch/segment-anything/issues/291) - SA-1B dataset usage discussion (accessed 2025-11-20)
- [Meta AI SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Official SA-1B documentation

**Technical References:**
- COCO Dataset Format: Microsoft COCO 2015
- pycocotools: Official COCO API for Python
- SA-1B Paper: "Segment Anything" (Kirillov et al., 2023)

**ARR-COC Integration:**
- Structured annotations enable precise spatial grounding for relevance realization
- Quality metrics (predicted_iou, stability_score) provide confidence filtering
- COCO compatibility allows integration with existing computer vision pipelines
