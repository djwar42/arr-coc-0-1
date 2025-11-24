# SA-1B Benchmark Evaluation & Model Comparison

## Overview

The SA-1B dataset enables comprehensive benchmark evaluation through zero-shot transfer across 23 diverse segmentation datasets. This evaluation methodology demonstrates SAM's ability to generalize without task-specific fine-tuning, establishing new standards for segmentation model comparison.

---

## Section 1: Zero-Shot Benchmark Suite (23 Datasets)

### The SA-23 Evaluation Protocol

SAM's zero-shot evaluation spans 23 carefully selected datasets covering diverse domains:

**Image Segmentation Datasets:**
- COCO (Common Objects in Context)
- LVIS (Large Vocabulary Instance Segmentation)
- ADE20K (Scene Parsing)
- Cityscapes (Urban Scenes)
- PASCAL VOC
- SBD (Semantic Boundaries Dataset)

**Video Segmentation Datasets:**
- DAVIS 2017
- YouTube-VOS
- MOSE (Complex Scenes)
- VOST (Extreme Transformations)
- BURST (Diverse Scenes)

**Specialized Domains:**
- Medical imaging datasets
- Remote sensing datasets
- Camouflaged object datasets
- Part segmentation datasets

### Dataset Selection Criteria

From [SAM Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf):
- "We use all 23 datasets for mIoU evaluation"
- Datasets chosen to test generalization across:
  - Object scales (tiny to large)
  - Domain distributions (natural, medical, satellite)
  - Annotation granularity (whole objects to parts)
  - Visual complexity (simple to cluttered)

---

## Section 2: mIoU Comparison Tables

### SAM vs Prior State-of-the-Art

**1-Click mIoU Results (SA-23 Average):**

| Model | SA-23 All | SA-23 Image | SA-23 Video | Speed (FPS) |
|-------|-----------|-------------|-------------|-------------|
| SAM (ViT-H) | 58.1 | 60.8 | 54.5 | 21.7 |
| SAM 2 (Hiera-B+) | 58.9 | 60.8 | 56.4 | 130.1 |
| SAM 2 (Full Mix) | 61.4 | 63.1 | 59.1 | 130.1 |

**5-Click mIoU Results:**

| Model | SA-23 All | SA-23 Image | SA-23 Video |
|-------|-----------|-------------|-------------|
| SAM | 81.3 | 82.1 | 80.3 |
| SAM 2 | 81.7 | 82.1 | 81.2 |
| SAM 2 (Full Mix) | 83.7 | 83.9 | 83.3 |

### Video Object Segmentation (J&F Metric)

From [SAM 2 Paper](https://arxiv.org/html/2408.00714v1):

| Method | MOSE val | DAVIS 2017 val | SA-V val | SA-V test |
|--------|----------|----------------|----------|-----------|
| XMem | 59.6 | 86.0 | 60.1 | 62.3 |
| Cutie-base | 69.9 | 87.9 | 60.7 | 62.7 |
| SAM 2 (Hiera-B+) | 75.8 | 90.9 | 73.6 | 74.1 |
| SAM 2 (Hiera-L) | 77.2 | 91.6 | 75.6 | 77.6 |

---

## Section 3: Model Size vs Performance

### Architecture Scaling Analysis

**Image Encoder Size Impact:**

| Encoder | Parameters | MOSE dev | SA-V val | 9 Zero-shot | Speed |
|---------|------------|----------|----------|-------------|-------|
| Hiera-S | ~22M | 70.9 | 65.5 | 69.4 | 1.33x |
| Hiera-B+ | ~80M | 73.0 | 68.3 | 70.7 | 1.00x |
| Hiera-L | ~308M | 75.0 | 66.3 | 71.9 | 0.60x |

**Key Findings:**
- Larger encoders improve accuracy but reduce speed
- B+ provides best balance for real-time applications
- L variant best for accuracy-critical tasks

### Memory Architecture Scaling

| #Memories | MOSE dev | SA-V val | 9 Zero-shot | Speed |
|-----------|----------|----------|-------------|-------|
| 4 | 73.5 | 68.6 | 70.5 | 1.01x |
| 6 | 73.0 | 68.3 | 70.7 | 1.00x |
| 8 | 73.2 | 69.0 | 70.7 | 0.93x |

---

## Section 4: Inference Speed Benchmarks

### Real-Time Performance Metrics

**Frames Per Second on A100 GPU:**

| Model Variant | Image FPS | Video FPS | Batch Size |
|--------------|-----------|-----------|------------|
| SAM (ViT-H) | 21.7 | N/A | 1 |
| SAM 2 (Hiera-B+) | 130.1 | 43.8 | 1 |
| SAM 2 (Hiera-L) | 77.4 | 30.2 | 1 |

**Speed Improvement: SAM 2 is 6x faster than SAM on images**

### Resolution Impact on Speed

| Resolution | MOSE dev | SA-V val | 9 Zero-shot | Relative Speed |
|------------|----------|----------|-------------|----------------|
| 512 | 73.0 | 68.3 | 70.7 | 1.00x |
| 768 | 76.1 | 71.1 | 72.5 | 0.43x |
| 1024 | 77.0 | 70.1 | 72.3 | 0.22x |

---

## Section 5: SAM vs Prior SOTA

### Comparison with Previous Methods

**Instance Segmentation (COCO):**

| Method | AP | AP50 | AP75 |
|--------|-----|------|------|
| Mask R-CNN | 37.1 | 59.0 | 39.4 |
| PointRend | 38.3 | 60.3 | 40.8 |
| SAM (Zero-shot) | ~40+ | ~65+ | ~42+ |

**Interactive Segmentation:**

| Method | NoC@85 | NoC@90 | NFO@90 |
|--------|--------|--------|--------|
| RITM | 1.68 | 2.18 | 2.51 |
| SimpleClick | 1.50 | 1.92 | 2.18 |
| SAM | **1.38** | **1.76** | **2.03** |

### Video Object Segmentation

From SAM 2 evaluation:

**Promptable Video Segmentation (3-click, offline):**

| Method | Average J&F (9 datasets) |
|--------|--------------------------|
| SAM+XMem++ | ~68 |
| SAM+Cutie | ~70 |
| SAM 2 | **73.2** |

**Key Result:** SAM 2 achieves better accuracy with 3x fewer interactions

---

## Section 6: Ablation Studies

### Data Ablations

**Training Data Mix Impact:**

| Training Data | SA-V val | 9 Zero-shot | SA-23 mIoU |
|---------------|----------|-------------|------------|
| VOS only | 48.1 | 59.7 | 45.4 |
| SA-V only | 63.0 | 69.7 | 53.0 |
| SA-V + SA-1B | 62.9 | 69.7 | 58.6 |
| VOS + SA-V + SA-1B | **63.1** | **71.6** | **58.9** |

**Key Finding:** Combining all data sources yields best results

### Data Quantity Scaling

Power law relationship observed:
- Performance scales logarithmically with data size
- Consistent across SA-V val, zero-shot, and MOSE benchmarks
- No saturation observed at current data scales

### Architecture Ablations

**Memory Attention Layers:**

| (Self-Attn, Cross-Attn) | MOSE dev | SA-V val | Speed |
|-------------------------|----------|----------|-------|
| (2, 2) | 73.3 | 67.3 | 1.13x |
| (3, 2) | 72.7 | 64.1 | 1.08x |
| (4, 4) | 73.0 | 68.3 | 1.00x |

---

## Section 7: Benchmark Reproduction Guide

### Setting Up Evaluation Environment

```python
# Install dependencies
pip install segment-anything
pip install pycocotools
pip install scipy scikit-image

# Clone SAM repository
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
```

### Loading Models for Evaluation

```python
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
model_type = "vit_h"
checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
```

### Evaluation Protocol Implementation

```python
import numpy as np
from pycocotools import mask as mask_utils

def evaluate_single_image(predictor, image, gt_mask, prompt_type="point"):
    """
    Evaluate SAM on a single image with ground truth mask.

    Args:
        predictor: SAM predictor instance
        image: Input image (H, W, 3)
        gt_mask: Ground truth binary mask (H, W)
        prompt_type: "point", "box", or "mask"

    Returns:
        iou: Intersection over Union score
    """
    predictor.set_image(image)

    if prompt_type == "point":
        # Sample point from ground truth mask
        ys, xs = np.where(gt_mask > 0)
        if len(ys) == 0:
            return 0.0

        # Use center of mass as prompt
        center_y = int(np.mean(ys))
        center_x = int(np.mean(xs))
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Select best mask by predicted IoU
        best_mask = masks[np.argmax(scores)]

    elif prompt_type == "box":
        # Get bounding box from ground truth
        ys, xs = np.where(gt_mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        input_box = np.array([x_min, y_min, x_max, y_max])

        masks, _, _ = predictor.predict(
            box=input_box,
            multimask_output=False
        )
        best_mask = masks[0]

    # Calculate IoU
    intersection = np.logical_and(best_mask, gt_mask).sum()
    union = np.logical_or(best_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    return iou

def evaluate_dataset(predictor, dataset, num_clicks=1):
    """
    Evaluate SAM on entire dataset with iterative clicking.

    Args:
        predictor: SAM predictor instance
        dataset: List of (image, gt_mask) tuples
        num_clicks: Number of iterative clicks (1, 3, or 5)

    Returns:
        mean_iou: Average IoU across dataset
    """
    ious = []

    for image, gt_mask in dataset:
        predictor.set_image(image)

        # Initial point from mask center
        ys, xs = np.where(gt_mask > 0)
        if len(ys) == 0:
            continue

        center_y, center_x = int(np.mean(ys)), int(np.mean(xs))
        points = [[center_x, center_y]]
        labels = [1]

        for click in range(num_clicks):
            input_points = np.array(points)
            input_labels = np.array(labels)

            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=(click == 0)
            )

            if click == 0:
                best_mask = masks[np.argmax(scores)]
            else:
                best_mask = masks[0]

            if click < num_clicks - 1:
                # Find largest error region for next click
                error = np.logical_xor(best_mask, gt_mask)
                false_neg = np.logical_and(error, gt_mask)
                false_pos = np.logical_and(error, ~gt_mask)

                if false_neg.sum() > false_pos.sum():
                    # Click on false negative (positive click)
                    ys, xs = np.where(false_neg)
                    idx = np.random.randint(len(ys))
                    points.append([xs[idx], ys[idx]])
                    labels.append(1)
                else:
                    # Click on false positive (negative click)
                    ys, xs = np.where(false_pos)
                    idx = np.random.randint(len(ys))
                    points.append([xs[idx], ys[idx]])
                    labels.append(0)

        # Calculate final IoU
        intersection = np.logical_and(best_mask, gt_mask).sum()
        union = np.logical_or(best_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    return np.mean(ious)
```

### Running Full Benchmark Suite

```python
import json
from pathlib import Path

def run_sa23_benchmark(model_checkpoint, datasets_root, output_dir):
    """
    Run complete SA-23 benchmark evaluation.

    Args:
        model_checkpoint: Path to SAM checkpoint
        datasets_root: Root directory containing all 23 datasets
        output_dir: Directory for saving results
    """
    # Initialize model
    sam = sam_model_registry["vit_h"](checkpoint=model_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    results = {}

    # Dataset configurations
    datasets = {
        "coco": {"type": "coco", "split": "val2017"},
        "lvis": {"type": "lvis", "split": "val"},
        "ade20k": {"type": "ade20k", "split": "validation"},
        # ... add all 23 datasets
    }

    for name, config in datasets.items():
        print(f"Evaluating on {name}...")

        # Load dataset
        dataset = load_dataset(datasets_root, name, config)

        # Evaluate with different click counts
        for num_clicks in [1, 3, 5]:
            miou = evaluate_dataset(predictor, dataset, num_clicks)
            results[f"{name}_{num_clicks}click"] = miou
            print(f"  {num_clicks}-click mIoU: {miou:.3f}")

    # Save results
    output_path = Path(output_dir) / "sa23_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Calculate averages
    for num_clicks in [1, 3, 5]:
        avg = np.mean([v for k, v in results.items()
                       if f"{num_clicks}click" in k])
        print(f"Average {num_clicks}-click mIoU: {avg:.3f}")

    return results
```

### Video Evaluation (J&F Metric)

```python
def calculate_j_and_f(pred_masks, gt_masks):
    """
    Calculate J&F metric for video object segmentation.

    J = Jaccard index (IoU)
    F = Boundary F-measure

    Args:
        pred_masks: List of predicted masks per frame
        gt_masks: List of ground truth masks per frame

    Returns:
        j_mean: Mean Jaccard index
        f_mean: Mean boundary F-measure
        jf_mean: Mean of J and F
    """
    j_scores = []
    f_scores = []

    for pred, gt in zip(pred_masks, gt_masks):
        # Jaccard index
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        j = intersection / union if union > 0 else 0.0
        j_scores.append(j)

        # Boundary F-measure
        pred_boundary = get_boundary(pred)
        gt_boundary = get_boundary(gt)

        precision = boundary_precision(pred_boundary, gt_boundary)
        recall = boundary_recall(pred_boundary, gt_boundary)
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f_scores.append(f)

    j_mean = np.mean(j_scores)
    f_mean = np.mean(f_scores)
    jf_mean = (j_mean + f_mean) / 2

    return j_mean, f_mean, jf_mean

def get_boundary(mask, dilation=5):
    """Extract boundary from binary mask."""
    from scipy.ndimage import binary_dilation, binary_erosion

    dilated = binary_dilation(mask, iterations=dilation)
    eroded = binary_erosion(mask, iterations=dilation)
    boundary = dilated ^ eroded

    return boundary
```

---

## Section 8: ARR-COC-0-1 Integration (Relevance Realization)

### Benchmarking Relevance Realization Models

The SA-1B benchmark methodology provides a template for evaluating spatial relevance models:

**Relevance-Aware Evaluation Metrics:**

```python
class RelevanceEvaluator:
    """Evaluate spatial relevance realization models."""

    def __init__(self, model, benchmark_datasets):
        self.model = model
        self.datasets = benchmark_datasets

    def evaluate_spatial_grounding(self, image, text_query, gt_regions):
        """
        Evaluate model's ability to ground text to spatial regions.

        Args:
            image: Input image
            text_query: Natural language description
            gt_regions: Ground truth relevant regions

        Returns:
            relevance_score: How well model identifies relevant regions
        """
        # Get model predictions
        predicted_relevance = self.model.predict_relevance(image, text_query)

        # Calculate relevance alignment
        alignment = self.calculate_alignment(predicted_relevance, gt_regions)

        return alignment

    def benchmark_zero_shot_transfer(self):
        """
        Test zero-shot transfer across diverse domains.

        Following SA-1B methodology:
        - Multiple datasets without fine-tuning
        - Various prompt types (text, region, hybrid)
        - Different granularity levels
        """
        results = {}

        for dataset_name, dataset in self.datasets.items():
            scores = []

            for sample in dataset:
                image = sample['image']
                queries = sample['relevance_queries']
                gt_regions = sample['relevant_regions']

                for query, gt in zip(queries, gt_regions):
                    score = self.evaluate_spatial_grounding(image, query, gt)
                    scores.append(score)

            results[dataset_name] = {
                'mean_relevance': np.mean(scores),
                'std': np.std(scores),
                'samples': len(scores)
            }

        return results
```

**Multi-Granular Relevance Testing:**

```python
def evaluate_hierarchical_relevance(model, image, hierarchy):
    """
    Test model's ability to handle multi-level relevance.

    Example hierarchy:
    - "person" (whole object)
    - "person's face" (part)
    - "person's left eye" (sub-part)

    Args:
        model: Relevance realization model
        image: Input image
        hierarchy: List of (query, gt_mask) at different granularities

    Returns:
        granularity_scores: Performance at each level
    """
    scores = {}

    for level, (query, gt_mask) in enumerate(hierarchy):
        pred_mask = model.ground_relevance(image, query)

        iou = calculate_iou(pred_mask, gt_mask)
        scores[f"level_{level}"] = iou

    # Test consistency across levels
    scores["hierarchical_consistency"] = check_containment(hierarchy)

    return scores
```

**Integration with ARR-COC Training:**

The benchmark evaluation patterns inform:
1. **Evaluation protocol** - Zero-shot testing across diverse relevance scenarios
2. **Metric design** - mIoU-style metrics for relevance alignment
3. **Ablation studies** - Data mix, model size, architecture choices
4. **Comparison baselines** - Prior attention/grounding methods

---

## Sources

**Primary Source:**
- [Segment Anything (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf) - Kirillov et al., Cited by 15,632

**SAM 2 Source:**
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/html/2408.00714v1) - Meta FAIR, arXiv:2408.00714

**Benchmark References:**
- [DAVIS 2017 Benchmark](https://davischallenge.org/)
- [YouTube-VOS Dataset](https://youtube-vos.org/)
- [MOSE: Complex Video Object Segmentation](https://henghuiding.github.io/MOSE/)

**Evaluation Tools:**
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

**Additional Web Sources:**
- [Medium: SAM Prompts Not Labels](https://medium.com/@kdk199604/sam-segment-anything-with-prompts-not-labels-7a85e6ec4d09) - SAM overview and evaluation
- [Lightly AI: SAM and Friends](https://www.lightly.ai/blog/segment-anything-model-and-friends) - Model comparison analysis

---

*Document created: 2025-11-20*
*Part of SA-1B Dataset Mastery Expansion - PART 34*
*~700 lines - Benchmark evaluation and model comparison guide*
