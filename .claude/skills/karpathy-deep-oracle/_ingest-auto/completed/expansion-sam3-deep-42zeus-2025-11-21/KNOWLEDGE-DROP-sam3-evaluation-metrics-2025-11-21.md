# SAM 3 Evaluation Metrics: cgF1, pHOTA, mAP, and HOTA

## Overview

SAM 3 uses a combination of evaluation metrics designed to measure different aspects of concept-guided segmentation and tracking performance. These metrics evaluate detection accuracy, association accuracy across time, localization quality, and open-vocabulary concept matching.

---

## Core Metrics

### 1. cgF1 (Concept-Guided F1 Score)

**What it measures**: cgF1 is a variant of the F1 score specifically designed for concept-guided segmentation tasks. It evaluates how well a model can segment objects based on text concept prompts.

**Definition**:
- F1 Score is the harmonic mean of precision and recall
- Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- Or equivalently: `F1 = 2 * TP / (2 * TP + FP + FN)`

**SAM 3 Context**:
- Measures segmentation accuracy for specific text prompts
- Evaluates whether the model correctly identifies and segments ALL instances of a given concept
- Handles open-vocabulary evaluation with 270K unique concepts
- Critical for evaluating presence token effectiveness (distinguishing "player in white" vs "player in red")

**Why F1 for Segmentation**:
- Balances finding all instances (recall) with avoiding false positives (precision)
- Harmonic mean penalizes extreme imbalances between precision and recall
- Better than accuracy for imbalanced cases (many concepts have few instances per image)

From [Wikipedia F-score](https://en.wikipedia.org/wiki/F-score) and [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

---

### 2. pHOTA (Prompt-guided HOTA)

**What it measures**: pHOTA extends the HOTA metric for prompt-guided video segmentation and tracking, evaluating how well models maintain consistent segmentation of prompted concepts across video frames.

**Relationship to HOTA**:
- Based on HOTA (Higher Order Tracking Accuracy) framework
- Adapted for text/concept-prompted segmentation in video
- Measures alignment between prompted concept and tracked object across time

**Key Components** (inherited from HOTA):
1. **Detection Accuracy (DetA)**: How well the model finds all prompted objects
2. **Association Accuracy (AssA)**: How well objects are tracked over time
3. **Localization Accuracy (LocA)**: Spatial alignment of predictions with ground truth

**SAM 3 Usage**:
- Evaluates video segmentation on SA-Co/VEval benchmark
- Measures temporal consistency of concept-guided tracking
- Assesses whether "player in white" maintains identity across frames

---

### 3. HOTA (Higher Order Tracking Accuracy)

**What it measures**: HOTA is a comprehensive metric for multi-object tracking that balances detection, association, and localization into a single score.

**Mathematical Foundation**:

HOTA combines three IoU-based scores:

**Localization IoU (Loc-IoU)**:
```
Loc-IoU = Intersection / Union (between predicted and ground-truth detection)
```

**Detection IoU (Det-IoU)**:
```
Det-IoU = TP / (TP + FP + FN)
```
Where:
- TP (True Positives) = matched prediction-ground truth pairs
- FP (False Positives) = unmatched predictions
- FN (False Negatives) = unmatched ground truths

**Association IoU (Ass-IoU)**:
```
Ass-IoU = TPA / (TPA + FPA + FNA)
```
Where:
- TPA (True Positive Associations) = matched detections between two tracks
- FPA (False Positive Associations) = remaining detections in predicted track
- FNA (False Negative Associations) = remaining detections in ground-truth track

**Final HOTA Score**:
```
HOTA = sqrt(DetA * AssA)
```
Integrated over IoU thresholds from 0.05 to 0.95 to include localization accuracy.

**Sub-metrics**:
- **DetA (Detection Accuracy)**: Average Det-IoU over dataset
- **AssA (Association Accuracy)**: Average Ass-IoU over all TP pairs
- **LocA (Localization Accuracy)**: Average Loc-IoU over all TP pairs

**Why HOTA over MOTA/IDF1**:

| Metric | Detection Weight | Association Weight | Issue |
|--------|-----------------|-------------------|-------|
| MOTA | High | Low | Ignores association quality |
| IDF1 | Low | High | Ignores detection quality |
| HOTA | Equal | Equal | Balanced evaluation |

From [HOTA paper (Luiten et al., IJCV 2020)](https://link.springer.com/article/10.1007/s11263-020-01375-2) and [TrackEval GitHub](https://github.com/JonathonLuiten/TrackEval)

---

### 4. mAP (Mean Average Precision)

**What it measures**: mAP evaluates object detection/segmentation by measuring precision-recall trade-offs across confidence thresholds and IoU thresholds.

**Calculation Steps**:

1. **IoU Calculation**: For each prediction, compute IoU with ground truths
2. **Matching**: Use Hungarian algorithm for one-to-one matching above IoU threshold
3. **Precision-Recall Curve**: Plot precision vs recall at different confidence thresholds
4. **Average Precision (AP)**: Area under the precision-recall curve for one class
5. **Mean AP**: Average AP across all classes

**COCO-style mAP**:
- **AP (primary)**: Average over IoU thresholds [0.50:0.05:0.95]
- **AP50**: AP at IoU = 0.50
- **AP75**: AP at IoU = 0.75
- **APs**: AP for small objects (area < 32^2)
- **APm**: AP for medium objects (32^2 < area < 96^2)
- **APl**: AP for large objects (area > 96^2)

**For Instance Segmentation**:
- IoU computed between predicted and ground-truth masks (not bounding boxes)
- Uses pycocotools for standardized evaluation

**SAM 3 Usage**:
- Evaluates image segmentation on LVIS and SA-Co/Gold benchmarks
- Measures quality across different IoU strictness levels
- Provides breakdown by object size

From [COCO evaluation metrics](https://www.picsellia.com/post/coco-evaluation-metrics-explained) and [V7 Labs mAP guide](https://www.v7labs.com/blog/mean-average-precision)

---

## Metric Comparison

### When to Use Each Metric

| Metric | Best For | Measures |
|--------|----------|----------|
| cgF1 | Concept-guided image segmentation | Balanced precision/recall for prompted concepts |
| pHOTA | Concept-guided video segmentation | Detection + association + localization |
| HOTA | Multi-object tracking | Balanced detection and association |
| mAP | Object detection/instance segmentation | Precision-recall across IoU thresholds |

### Key Differences

**cgF1 vs mAP**:
- cgF1: Single threshold, concept-specific, emphasizes finding all instances
- mAP: Multiple thresholds, class-averaged, emphasizes confidence calibration

**pHOTA vs HOTA**:
- pHOTA: Adapted for text-prompted segmentation
- HOTA: General multi-object tracking metric

**HOTA vs MOTA**:
- HOTA: Equally weighs detection and association
- MOTA: Detection-focused, can be high even with poor association

---

## Implementation Details

### HOTA Implementation

Reference implementation: [TrackEval GitHub](https://github.com/JonathonLuiten/TrackEval)

Key implementation aspects:
1. Compute one-to-one matching using Hungarian algorithm
2. Calculate TP, FP, FN for detection
3. For each TP pair, compute track-level TPA, FPA, FNA
4. Average over all TPs and integrate over alpha thresholds

### mAP Implementation

Reference: [pycocotools](https://github.com/cocodataset/cocoapi)

Key implementation aspects:
1. Sort predictions by confidence
2. Compute IoU matrix between predictions and ground truths
3. Use Hungarian matching at each IoU threshold
4. Interpolate precision-recall curve with 101 recall points
5. Compute area under curve

### F1 Score Implementation

```python
# Standard F1 calculation
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# For segmentation, TP is determined by IoU threshold
# Typically IoU > 0.5 for a match
```

---

## SAM 3 Benchmark Results

### Image Segmentation (SA-Co/Gold)
- Primary metric: cgF1
- Measures concept-guided segmentation accuracy
- SAM 3 achieves 75-80% of human performance

### Video Segmentation (SA-Co/VEval)
- Primary metric: pHOTA
- Evaluates temporal consistency of prompted concepts
- Assesses tracking quality alongside segmentation

### Comparison Benchmarks
- LVIS: Uses mAP (standard detection/segmentation metric)
- SA-V, YT-Temporal-1B: Use pHOTA/HOTA variants

---

## Key Insights

### Why Multiple Metrics?

SAM 3 uses multiple metrics because different aspects require different evaluation:

1. **cgF1**: Does it find ALL instances of a concept?
2. **pHOTA**: Does it maintain identity over time?
3. **mAP**: How well calibrated are confidence scores?
4. **HOTA**: Balanced view of detection + association

### Metric Selection Rationale

- **cgF1 for concept matching**: F1's harmonic mean penalizes models that either miss instances (low recall) or produce false positives (low precision)
- **pHOTA for video**: HOTA's balanced weighting prevents optimizing detection at the expense of association
- **mAP for comparison**: Standard metric allows comparison with other methods on established benchmarks

### Limitations

- **cgF1**: Doesn't capture localization quality (IoU)
- **pHOTA/HOTA**: Requires temporal annotations, computationally expensive
- **mAP**: Sensitive to IoU threshold choice, can be dominated by easy categories

---

## Sources

**Papers and Documentation**:
- [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2) - Luiten et al., IJCV 2020
- [SAM 3 Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) - OpenReview, November 2025
- [COCO Evaluation Metrics](https://www.picsellia.com/post/coco-evaluation-metrics-explained) - Picsellia blog

**Implementation References**:
- [TrackEval GitHub](https://github.com/JonathonLuiten/TrackEval) - Official HOTA implementation
- [pycocotools](https://github.com/cocodataset/cocoapi) - Standard mAP evaluation
- [scikit-learn F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

**Web Resources**:
- [How to evaluate tracking with HOTA metrics](https://autonomousvision.github.io/hota-metrics/) - Autonomous Vision Blog (accessed 2025-11-23)
- [Mean Average Precision Explained](https://www.v7labs.com/blog/mean-average-precision) - V7 Labs (accessed 2025-11-23)
- [Instance Segmentation Evaluation Criteria](https://softwaremill.com/instance-segmentation-evaluation-criteria/) - SoftwareMill (accessed 2025-11-23)
- [COCO AP metric](https://www.intel.com/content/www/us/en/docs/programmable/768977/2024-1/the-map-and-coco-ap-metrics.html) - Intel Documentation

---

## Summary

SAM 3's evaluation framework uses complementary metrics that together provide a comprehensive assessment:

- **cgF1**: Concept-guided segmentation accuracy (find all instances)
- **pHOTA**: Video segmentation with temporal association (track over time)
- **mAP**: Standard detection/segmentation metric (compare with other methods)
- **HOTA**: Balanced multi-object tracking (detect + associate + localize)

The choice of multiple metrics reflects SAM 3's dual role as both an image segmenter and video tracker, requiring evaluation of static segmentation quality, temporal consistency, and comparison with existing benchmarks.
