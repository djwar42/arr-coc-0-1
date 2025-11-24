# SA-1B Annotation Fields: Quality Metrics and Prompt Tracking

## Overview

SA-1B annotations include several specialized fields beyond standard COCO format that provide quality metrics and provenance information for each mask. Understanding `predicted_iou`, `stability_score`, `point_coords`, and other fields is essential for quality filtering, mask selection, and effective training on SA-1B data. These metrics enable users to filter the 1.1 billion masks to extract the highest-quality subsets for their specific applications.

---

## 1. predicted_iou: Model Confidence Metric

### What is predicted_iou?

`predicted_iou` is SAM's self-assessed prediction of the Intersection-over-Union (IoU) between its generated mask and the "true" mask it's attempting to segment.

**Range:** 0.0 to 1.0
**Higher = Better:** Higher values indicate SAM's confidence that the mask accurately captures the intended object.

### How It's Calculated

During SAM's training, an IoU prediction head learns to estimate mask quality:

```python
# Simplified SAM architecture component
class IoUPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, mask_tokens):
        return self.mlp(mask_tokens)
```

The model learns to predict IoU by comparing:
- Predicted mask vs. ground truth during training
- Multiple mask proposals to select the best one

### Interpreting Values

| predicted_iou | Quality Level | Use Case |
|---------------|---------------|----------|
| 0.95+ | Excellent | High-precision applications |
| 0.88-0.95 | Good | General training |
| 0.75-0.88 | Moderate | May need filtering |
| <0.75 | Low | Often ambiguous or partial |

### Quality Filtering by predicted_iou

```python
def filter_by_predicted_iou(annotations, threshold=0.88):
    """Filter masks by SAM's IoU prediction confidence."""
    return [
        ann for ann in annotations
        if ann["predicted_iou"] >= threshold
    ]

# Example: Progressive filtering
all_masks = annotation_data["annotations"]
print(f"Total masks: {len(all_masks)}")

thresholds = [0.75, 0.88, 0.95]
for thresh in thresholds:
    filtered = filter_by_predicted_iou(all_masks, thresh)
    percentage = len(filtered) / len(all_masks) * 100
    print(f"IoU >= {thresh}: {len(filtered)} ({percentage:.1f}%)")
```

### Distribution Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_iou_distribution(annotations):
    """Analyze predicted_iou distribution across masks."""
    ious = [ann["predicted_iou"] for ann in annotations]

    stats = {
        "mean": np.mean(ious),
        "median": np.median(ious),
        "std": np.std(ious),
        "min": np.min(ious),
        "max": np.max(ious),
        "percentile_25": np.percentile(ious, 25),
        "percentile_75": np.percentile(ious, 75),
        "percentile_90": np.percentile(ious, 90)
    }

    return stats, ious

# Visualization
stats, ious = analyze_iou_distribution(annotations["annotations"])
plt.hist(ious, bins=50, edgecolor='black')
plt.xlabel("predicted_iou")
plt.ylabel("Count")
plt.title("Distribution of predicted_iou")
plt.axvline(x=0.88, color='r', linestyle='--', label='Common threshold (0.88)')
plt.legend()
plt.show()
```

---

## 2. stability_score: Robustness to Prompt Perturbation

### What is stability_score?

`stability_score` measures how consistent a mask is when the input prompt (point) is slightly perturbed. It quantifies mask robustness - whether small changes in prompt location produce similar or different masks.

**Range:** 0.0 to 1.0
**Higher = More Stable:** The mask is consistent under prompt variations.

### How It's Calculated

SAM computes stability by:
1. Generating mask with original prompt
2. Generating masks with perturbed prompts (small shifts)
3. Computing IoU between original and perturbed masks
4. Averaging IoU across perturbations

```python
# Conceptual stability calculation
def compute_stability_score(sam_model, image, original_point, num_perturbations=16):
    """Compute stability score for a mask."""
    # Generate original mask
    original_mask = sam_model.predict(image, original_point)

    ious = []
    for _ in range(num_perturbations):
        # Perturb point by small random offset
        perturbed_point = original_point + np.random.uniform(-5, 5, size=2)

        # Generate perturbed mask
        perturbed_mask = sam_model.predict(image, perturbed_point)

        # Compute IoU
        intersection = (original_mask & perturbed_mask).sum()
        union = (original_mask | perturbed_mask).sum()
        iou = intersection / union if union > 0 else 0

        ious.append(iou)

    return np.mean(ious)
```

### Interpreting Values

| stability_score | Stability Level | Typical Scenario |
|-----------------|-----------------|------------------|
| 0.95+ | Very Stable | Well-defined objects |
| 0.90-0.95 | Stable | Clear boundaries |
| 0.80-0.90 | Moderate | Some ambiguity |
| <0.80 | Unstable | Ambiguous regions |

### Why Stability Matters

**High stability masks** (>0.95):
- Represent unambiguous objects
- Have clear boundaries
- Consistent semantic interpretation
- Better for training

**Low stability masks** (<0.80):
- May be at object boundaries
- Ambiguous granularity (part vs whole)
- Multiple valid interpretations
- Can confuse models during training

### Combined Quality Filtering

```python
def filter_high_quality_masks(annotations,
                               min_iou=0.88,
                               min_stability=0.95,
                               min_area=100):
    """
    Filter for high-quality masks using multiple criteria.

    Args:
        annotations: List of SA-1B annotations
        min_iou: Minimum predicted IoU
        min_stability: Minimum stability score
        min_area: Minimum mask area in pixels

    Returns:
        Filtered list of annotations
    """
    filtered = []
    for ann in annotations:
        if (ann["predicted_iou"] >= min_iou and
            ann["stability_score"] >= min_stability and
            ann["area"] >= min_area):
            filtered.append(ann)

    return filtered

# Usage
high_quality = filter_high_quality_masks(
    annotation_data["annotations"],
    min_iou=0.88,
    min_stability=0.95,
    min_area=100
)
print(f"High quality masks: {len(high_quality)}")
```

---

## 3. point_coords: Prompt Tracking

### What is point_coords?

`point_coords` records the original point prompt(s) used to generate the mask. This is the input SAM received to produce this specific segmentation.

**Format:** List of [x, y] coordinates in pixel space

### Structure

```json
{
    "point_coords": [[1125.5, 750.0]]
}
```

Most SA-1B masks use a single point prompt, but the format supports multiple points:
```json
{
    "point_coords": [[100, 200], [150, 250]]
}
```

### Understanding Point Placement

SA-1B uses automatic mask generation with a grid of points:

```python
def generate_point_grid(image_width, image_height, points_per_side=32):
    """Generate grid of points for automatic mask generation."""
    offset_x = image_width / points_per_side / 2
    offset_y = image_height / points_per_side / 2

    points = []
    for i in range(points_per_side):
        for j in range(points_per_side):
            x = offset_x + i * (image_width / points_per_side)
            y = offset_y + j * (image_height / points_per_side)
            points.append([x, y])

    return points

# For a 2250x1500 image with 32x32 grid
points = generate_point_grid(2250, 1500, 32)
print(f"Total prompt points: {len(points)}")  # 1024
```

### Using point_coords

```python
def visualize_prompts(image, annotation):
    """Visualize prompt points on image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw prompt points
    for x, y in annotation["point_coords"]:
        ax.plot(x, y, 'r*', markersize=15)

    # Draw bounding box
    bbox = annotation["bbox"]
    rect = plt.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        fill=False, edgecolor='green', linewidth=2
    )
    ax.add_patch(rect)

    plt.title(f"Prompt at {annotation['point_coords'][0]}")
    plt.show()
```

### Point-Mask Relationship Analysis

```python
def analyze_point_position(annotation):
    """Analyze where prompt point lies within the mask."""
    from pycocotools import mask as mask_utils

    # Decode mask
    mask = mask_utils.decode(annotation["segmentation"])

    point = annotation["point_coords"][0]
    x, y = int(point[0]), int(point[1])

    # Check if point is inside mask
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        is_inside = mask[y, x] == 1
    else:
        is_inside = False

    # Distance to mask centroid
    if mask.sum() > 0:
        ys, xs = np.where(mask == 1)
        centroid_x = xs.mean()
        centroid_y = ys.mean()
        distance_to_centroid = np.sqrt(
            (point[0] - centroid_x)**2 +
            (point[1] - centroid_y)**2
        )
    else:
        distance_to_centroid = float('inf')

    return {
        "point_inside_mask": is_inside,
        "distance_to_centroid": distance_to_centroid
    }
```

---

## 4. crop_box: Generation Region

### What is crop_box?

`crop_box` specifies the region of the image that was used during mask generation. For most SA-1B masks, this is the full image.

**Format:** [x, y, width, height] in pixels

### Structure

```json
{
    "crop_box": [0, 0, 2250, 1500]
}
```

This means the full image (2250x1500) was used for generation.

### When crop_box Differs

SAM can use cropped regions for higher-resolution processing:

```json
{
    "crop_box": [500, 300, 800, 600]
}
```

This indicates a 800x600 crop starting at (500, 300) was used.

### Working with crop_box

```python
def is_full_image_crop(annotation, image_width, image_height):
    """Check if mask was generated on full image."""
    crop = annotation["crop_box"]
    return (
        crop[0] == 0 and
        crop[1] == 0 and
        crop[2] == image_width and
        crop[3] == image_height
    )

def get_crop_region(image, annotation):
    """Extract the crop region from image."""
    x, y, w, h = annotation["crop_box"]
    return image[y:y+h, x:x+w]
```

---

## 5. bbox: Bounding Box

### Format and Calculation

`bbox` is the axis-aligned bounding box containing the mask.

**Format:** [x, y, width, height]
- `x`: Left edge (pixels from left)
- `y`: Top edge (pixels from top)
- `width`: Box width
- `height`: Box height

```python
def bbox_to_corners(bbox):
    """Convert COCO bbox to corner format."""
    x, y, w, h = bbox
    return {
        "x1": x,
        "y1": y,
        "x2": x + w,
        "y2": y + h
    }

def bbox_to_center(bbox):
    """Get bounding box center."""
    x, y, w, h = bbox
    return {
        "cx": x + w / 2,
        "cy": y + h / 2
    }

def normalize_bbox(bbox, image_width, image_height):
    """Normalize bbox to [0, 1] range."""
    x, y, w, h = bbox
    return [
        x / image_width,
        y / image_height,
        w / image_width,
        h / image_height
    ]
```

### Bbox vs Area Analysis

```python
def compute_bbox_fill_ratio(annotation):
    """Compute how much of bbox is filled by mask."""
    bbox = annotation["bbox"]
    bbox_area = bbox[2] * bbox[3]
    mask_area = annotation["area"]

    fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
    return fill_ratio

# High fill ratio (~1.0): rectangular objects
# Low fill ratio (<0.5): irregular shapes, holes
```

---

## 6. area: Mask Size

### Understanding Area

`area` is the total number of foreground pixels in the mask.

```python
# Verify area matches decoded mask
from pycocotools import mask as mask_utils

mask = mask_utils.decode(annotation["segmentation"])
computed_area = mask.sum()
stored_area = annotation["area"]

assert abs(computed_area - stored_area) < 1, "Area mismatch"
```

### Size-Based Filtering

```python
def filter_by_size(annotations, min_area=100, max_area=None,
                   image_width=None, image_height=None,
                   min_relative=None, max_relative=None):
    """
    Filter masks by absolute or relative size.

    Args:
        annotations: List of annotations
        min_area: Minimum absolute area (pixels)
        max_area: Maximum absolute area (pixels)
        image_width/height: For relative filtering
        min_relative: Minimum relative area (0-1)
        max_relative: Maximum relative area (0-1)
    """
    filtered = []
    image_area = None
    if image_width and image_height:
        image_area = image_width * image_height

    for ann in annotations:
        area = ann["area"]

        # Absolute filtering
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue

        # Relative filtering
        if image_area:
            relative_area = area / image_area
            if min_relative and relative_area < min_relative:
                continue
            if max_relative and relative_area > max_relative:
                continue

        filtered.append(ann)

    return filtered

# Examples
# Small objects only (< 1% of image)
small_masks = filter_by_size(
    annotations,
    max_relative=0.01,
    image_width=2250,
    image_height=1500
)

# Medium objects (1-10% of image)
medium_masks = filter_by_size(
    annotations,
    min_relative=0.01,
    max_relative=0.10,
    image_width=2250,
    image_height=1500
)
```

---

## 7. Multi-Criteria Quality Scoring

### Composite Quality Score

```python
def compute_quality_score(annotation,
                          iou_weight=0.4,
                          stability_weight=0.4,
                          area_weight=0.2,
                          target_area_range=(1000, 100000)):
    """
    Compute composite quality score for mask selection.

    Args:
        annotation: SA-1B annotation
        iou_weight: Weight for predicted_iou
        stability_weight: Weight for stability_score
        area_weight: Weight for area appropriateness
        target_area_range: Ideal area range (min, max)

    Returns:
        Quality score in [0, 1]
    """
    # IoU component
    iou_score = annotation["predicted_iou"]

    # Stability component
    stability_score = annotation["stability_score"]

    # Area appropriateness (penalize too small or too large)
    area = annotation["area"]
    min_area, max_area = target_area_range

    if area < min_area:
        area_score = area / min_area
    elif area > max_area:
        area_score = max_area / area
    else:
        area_score = 1.0

    # Weighted combination
    total = (
        iou_weight * iou_score +
        stability_weight * stability_score +
        area_weight * area_score
    )

    return total

def select_top_masks(annotations, n=10):
    """Select top n masks by quality score."""
    scored = [
        (ann, compute_quality_score(ann))
        for ann in annotations
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [ann for ann, score in scored[:n]]
```

### Stratified Sampling by Quality

```python
def stratified_quality_sample(annotations, n_samples,
                               quality_buckets=[0.75, 0.88, 0.95, 1.0]):
    """
    Sample masks stratified by quality levels.

    Args:
        annotations: List of annotations
        n_samples: Total samples to return
        quality_buckets: Boundaries for quality strata

    Returns:
        Stratified sample of annotations
    """
    import random

    # Assign to buckets
    buckets = {i: [] for i in range(len(quality_buckets))}
    for ann in annotations:
        iou = ann["predicted_iou"]
        for i, threshold in enumerate(quality_buckets):
            if iou <= threshold:
                buckets[i].append(ann)
                break

    # Sample from each bucket
    samples_per_bucket = n_samples // len(quality_buckets)
    sampled = []

    for bucket_anns in buckets.values():
        if len(bucket_anns) >= samples_per_bucket:
            sampled.extend(random.sample(bucket_anns, samples_per_bucket))
        else:
            sampled.extend(bucket_anns)

    return sampled
```

---

## 8. Field Relationships and Correlations

### Analyzing Correlations

```python
import pandas as pd
import seaborn as sns

def analyze_field_correlations(annotations):
    """Analyze correlations between annotation fields."""
    data = []
    for ann in annotations:
        data.append({
            "predicted_iou": ann["predicted_iou"],
            "stability_score": ann["stability_score"],
            "area": ann["area"],
            "bbox_area": ann["bbox"][2] * ann["bbox"][3],
            "fill_ratio": ann["area"] / (ann["bbox"][2] * ann["bbox"][3])
        })

    df = pd.DataFrame(data)

    # Compute correlations
    correlations = df.corr()

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title("Field Correlations")
    plt.show()

    return correlations
```

### Common Patterns

**High IoU + Low Stability:**
- Ambiguous boundaries
- Multiple valid interpretations
- May be at granularity boundaries

**Low IoU + High Stability:**
- Consistently wrong predictions
- Possibly background confusion
- May indicate model limitations

**Small Area + Low Stability:**
- Very small objects are often ambiguous
- Single-pixel shifts have large relative effect

---

## 9. Practical Quality Filtering Guidelines

### Recommended Thresholds by Application

| Application | predicted_iou | stability_score | min_area |
|-------------|---------------|-----------------|----------|
| High-precision training | >= 0.95 | >= 0.98 | 500 |
| General training | >= 0.88 | >= 0.95 | 100 |
| Exploratory analysis | >= 0.75 | >= 0.85 | 50 |
| Maximum coverage | >= 0.50 | >= 0.50 | 10 |

### Complete Filtering Pipeline

```python
class SA1BQualityFilter:
    """Comprehensive quality filtering for SA-1B annotations."""

    def __init__(self,
                 min_predicted_iou=0.88,
                 min_stability_score=0.95,
                 min_area=100,
                 max_area=None,
                 min_relative_area=None,
                 max_relative_area=None):
        self.min_predicted_iou = min_predicted_iou
        self.min_stability_score = min_stability_score
        self.min_area = min_area
        self.max_area = max_area
        self.min_relative_area = min_relative_area
        self.max_relative_area = max_relative_area

    def filter(self, annotation_data):
        """Filter annotations from SA-1B data."""
        image_area = (
            annotation_data["image"]["width"] *
            annotation_data["image"]["height"]
        )

        filtered = []
        for ann in annotation_data["annotations"]:
            # Quality thresholds
            if ann["predicted_iou"] < self.min_predicted_iou:
                continue
            if ann["stability_score"] < self.min_stability_score:
                continue

            # Absolute area
            if ann["area"] < self.min_area:
                continue
            if self.max_area and ann["area"] > self.max_area:
                continue

            # Relative area
            relative = ann["area"] / image_area
            if self.min_relative_area and relative < self.min_relative_area:
                continue
            if self.max_relative_area and relative > self.max_relative_area:
                continue

            filtered.append(ann)

        return filtered

    def get_stats(self, original, filtered):
        """Get filtering statistics."""
        return {
            "original_count": len(original),
            "filtered_count": len(filtered),
            "retention_rate": len(filtered) / len(original) if original else 0,
            "avg_iou_original": np.mean([a["predicted_iou"] for a in original]),
            "avg_iou_filtered": np.mean([a["predicted_iou"] for a in filtered]) if filtered else 0
        }

# Usage
filter = SA1BQualityFilter(
    min_predicted_iou=0.88,
    min_stability_score=0.95,
    min_area=100
)
filtered = filter.filter(annotation_data)
stats = filter.get_stats(annotation_data["annotations"], filtered)
print(f"Retained {stats['retention_rate']*100:.1f}% of masks")
```

---

## 10. ARR-COC-0-1 Integration: Quality Metrics for Relevance Scoring

### Relevance-Quality Mapping

SA-1B quality metrics directly map to spatial relevance concepts in ARR-COC:

| SA-1B Metric | ARR-COC Relevance Aspect |
|--------------|--------------------------|
| predicted_iou | Spatial grounding confidence |
| stability_score | Robustness of spatial reference |
| area | Spatial importance/salience |
| point_coords | Attention focus location |

### Integration Patterns

```python
class ARRCOCSpatialRelevanceExtractor:
    """Extract spatial relevance features from SA-1B for ARR-COC training."""

    def __init__(self, confidence_threshold=0.88, stability_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold

    def extract_relevance_features(self, annotation_data):
        """
        Extract spatial relevance features for VLM training.

        Returns features suitable for spatial grounding objectives.
        """
        image_info = annotation_data["image"]
        w, h = image_info["width"], image_info["height"]

        # Filter high-confidence masks
        quality_masks = [
            ann for ann in annotation_data["annotations"]
            if ann["predicted_iou"] >= self.confidence_threshold
            and ann["stability_score"] >= self.stability_threshold
        ]

        # Extract spatial relevance features
        spatial_features = []
        for ann in quality_masks:
            bbox = ann["bbox"]
            spatial_features.append({
                # Normalized spatial coordinates
                "center_x": (bbox[0] + bbox[2]/2) / w,
                "center_y": (bbox[1] + bbox[3]/2) / h,
                "width": bbox[2] / w,
                "height": bbox[3] / h,

                # Relevance confidence
                "spatial_confidence": ann["predicted_iou"],
                "spatial_stability": ann["stability_score"],

                # Size-based salience (larger = more salient)
                "salience_score": min(1.0, ann["area"] / (w * h * 0.1)),

                # Prompt location (attention focus)
                "attention_focus": [
                    ann["point_coords"][0][0] / w,
                    ann["point_coords"][0][1] / h
                ] if ann.get("point_coords") else None
            })

        return spatial_features

    def compute_relevance_ranking(self, spatial_features):
        """
        Rank regions by spatial relevance for ARR-COC training.
        """
        # Composite relevance score
        for feat in spatial_features:
            feat["relevance_score"] = (
                0.4 * feat["spatial_confidence"] +
                0.3 * feat["spatial_stability"] +
                0.3 * feat["salience_score"]
            )

        # Sort by relevance
        spatial_features.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        return spatial_features
```

---

## Sources

**Web Research:**
- [arXiv Stable Segment Anything](https://arxiv.org/html/2311.15776v2) - Analysis of SAM stability metrics (accessed 2025-11-20)
- [Meta AI SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Official SA-1B documentation (accessed 2025-11-20)
- [Stanford CRFM SA-1B](https://crfm.stanford.edu/ecosystem-graphs/index.html?asset=SA-1B) - SA-1B ecosystem information (accessed 2025-11-20)
- [Medium SAM Annotations](https://medium.com/@sumithearra/segment-anything-model-sam-to-generate-masks-with-annotations-5e2fa8bcb961) - SAM mask generation details

**Technical References:**
- "Segment Anything" (Kirillov et al., 2023) - Original SAM paper with metric definitions
- "Uncertainty-aware Fine-tuning of Segmentation" (Liu et al., 2024) - Quality metrics for fine-tuning

**ARR-COC Integration:**
- Quality metrics enable relevance-based filtering for spatial grounding
- predicted_iou and stability_score map directly to spatial confidence
- Point coordinates track attention focus for relevance realization
