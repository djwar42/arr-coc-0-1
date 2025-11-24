# SAM 3: Benchmark Metrics and Evaluation

**PART 28/42 - SA-Co Benchmark Metrics (CGF1, IL-MCC, pmF1)**

**Date**: 2025-11-20
**Source**: SAM 3 paper evaluation methodology

---

## SA-Co Evaluation Metrics

SAM 3 introduces **new metrics** specifically designed for Promptable Concept Segmentation (PCS).

### Why New Metrics?

**Traditional metrics** (AP, mAP, IoU):
- Designed for closed-set detection (80 COCO classes)
- Don't measure calibration (confidence reliability)
- Don't capture open-vocabulary challenges

**PCS requirements**:
- Open vocabulary (214K concepts)
- Binary presence decision ("Is concept present?")
- Calibrated confidence (predictions > 0.5 should be reliable)
- Fine-grained discrimination (hard negatives)

**Solution**: SA-Co metrics measure BOTH localization AND recognition.

---

## The Three Core Metrics

### 1. CGF1 (Classification-Gated F1)

**Primary metric**: Combines localization quality and recognition accuracy

**Formula**:
```
CGF1 = 100 × pmF1 × IL-MCC
```

**Range**: 0-100 (higher is better)

**Interpretation**:
- CGF1 = 65: Model has 65% of human-level performance
- CGF1 = 100: Perfect performance (all positives found, no false positives)

**Why multiplicative?**
- BOTH localization (pmF1) AND recognition (IL-MCC) must be good
- Poor recognition (IL-MCC low) → CGF1 tanks (even with perfect localization)
- Poor localization (pmF1 low) → CGF1 tanks (even with perfect recognition)

### 2. IL-MCC (Image-Level Matthews Correlation Coefficient)

**Measures**: Binary classification accuracy ("Is concept present?")

**Formula**:
```
IL-MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range**: -1 to +1 (higher is better)
- +1: Perfect classification (all concepts correctly present/absent)
- 0: Random guessing
- -1: Perfectly wrong (all concepts inverted)

**Why MCC?**
- Balanced metric (works with class imbalance)
- SA-Co has 30 hard negatives per image (imbalanced toward negatives)
- Traditional accuracy would be misleading (99% accuracy by always predicting negative!)

**Example**:
```
Image with "yellow school bus"
Model predictions:
✓ "yellow school bus" present (TP)
✓ "yellow taxi" absent (TN - hard negative)
✓ "red bus" absent (TN - hard negative)
✗ "school van" present (FP - false positive)

IL-MCC = weighted combination of TP, TN, FP, FN
```

### 3. pmF1 (Positive Macro F1)

**Measures**: Localization quality on positive examples only

**Formula**:
```
For each concept with ≥1 positive instance:
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)

pmF1 = Average F1 across all positive concepts
```

**Range**: 0-100 (higher is better)

**Why "positive macro"?**
- Only evaluates concepts that ARE present (ignores negatives)
- Macro averaging: Each concept weighted equally (rare concepts not ignored)
- Measures spatial localization quality (IoU thresholds)

**IoU threshold**: Prediction matches ground truth if IoU > 0.5

**Example**:
```
Concept: "person"
Ground truth: 5 instances
Model predicts: 6 instances

Matching:
- 4 predictions match ground truth (IoU > 0.5) → TP = 4
- 2 predictions don't match (IoU < 0.5) → FP = 2
- 1 ground truth missed → FN = 1

Precision = 4 / (4+2) = 0.67
Recall = 4 / (4+1) = 0.80
F1 = 2 × (0.67 × 0.80) / (0.67 + 0.80) = 0.73

pmF1 = Average across all positive concepts
```

---

## Metric Relationships

### CGF1 Decomposition

**CGF1 = pmF1 × IL-MCC**

**Example 1**: Good localization, poor recognition
- pmF1 = 75 (good spatial localization)
- IL-MCC = 0.50 (poor concept recognition, many false positives)
- **CGF1 = 75 × 0.50 = 37.5** (mediocre overall)

**Example 2**: Good recognition, poor localization
- pmF1 = 50 (poor spatial localization, many missed instances)
- IL-MCC = 0.85 (good concept recognition, few false positives)
- **CGF1 = 50 × 0.85 = 42.5** (mediocre overall)

**Example 3**: Good both
- pmF1 = 75 (good localization)
- IL-MCC = 0.85 (good recognition)
- **CGF1 = 75 × 0.85 = 63.8** (strong performance)

**SAM 3 performance**: CGF1 = 65.0 (pmF1 = 77.1, IL-MCC = 0.82)

---

## Comparison to Traditional Metrics

### AP (Average Precision)

**Traditional metric** for object detection (COCO, LVIS)

**Formula**:
```
AP = Integral of Precision-Recall curve
```

**Limitations for PCS**:
- Requires confidence thresholding (no single threshold works for all concepts)
- Doesn't measure calibration directly
- Doesn't penalize false negatives on absent concepts (hard negatives)

**CGF1 vs AP**:

| Aspect | AP | CGF1 |
|--------|----|----|
| **Calibration** | Indirect | Direct (via IL-MCC) |
| **Hard negatives** | Not evaluated | Core component |
| **Open vocabulary** | Closed set | Open vocabulary |
| **Threshold** | All thresholds | Single threshold (0.5) |

**When to use AP**: Closed-set detection (80 COCO classes)
**When to use CGF1**: Open-vocabulary concept segmentation (214K concepts)

### mIoU (Mean Intersection over Union)

**Traditional metric** for semantic segmentation

**Formula**:
```
For each class:
  IoU = Intersection / Union
mIoU = Average IoU across all classes
```

**Limitations for PCS**:
- Assumes all pixels belong to some class
- Doesn't handle open vocabulary (new concepts at test time)
- Doesn't measure instance-level segmentation

**pmF1 vs mIoU**:

| Aspect | mIoU | pmF1 |
|--------|------|------|
| **Granularity** | Pixel-level | Instance-level |
| **Vocabulary** | Closed set | Open vocabulary |
| **Hard negatives** | N/A | Handled via IL-MCC |

---

## Calibration and Thresholding

### Why Calibration Matters

**Uncalibrated model**:
- Predicts "elephant" at 0.75 confidence
- Actually correct only 40% of the time
- User can't trust confidence scores

**Calibrated model** (SAM 3 with presence token):
- Predicts "elephant" at 0.75 confidence
- Actually correct ~75% of the time
- User can use single threshold (0.5) for all concepts

### SA-Co Evaluation Protocol

**Fixed threshold**: 0.5 confidence

**Predictions above 0.5**:
- Counted as positive predictions
- Evaluated for localization quality (pmF1)
- Evaluated for recognition accuracy (IL-MCC)

**Predictions below 0.5**:
- Ignored (concept considered absent)

**Why 0.5?**
- Reflects real-world usage (users want reliable predictions)
- Tests calibration (good calibration → high performance at 0.5)
- Simpler than threshold sweeping (no per-concept tuning)

---

## Human Performance Bounds

### SA-Co/Gold Triple Annotation

**Setup**: 3 independent human annotators per image

**Human variability**:
- Annotator A: Conservative (few predictions, high precision)
- Annotator B: Balanced
- Annotator C: Liberal (many predictions, high recall)

### Human Performance Metrics

**Lower bound** (most conservative annotator):
- CGF1 = 74.2
- IL-MCC = 0.87
- pmF1 = 85.3

**Upper bound** (most liberal annotator):
- CGF1 = 81.4
- IL-MCC = 0.92
- pmF1 = 88.5

**Average** (median annotator):
- CGF1 = 77.8
- IL-MCC = 0.89
- pmF1 = 87.4

### SAM 3 vs Human

**SAM 3 performance**:
- CGF1 = 65.0
- IL-MCC = 0.82
- pmF1 = 77.1

**SAM 3 achievement**:
- **88% of human lower bound** (65.0 / 74.2)
- **79% of human average** (65.0 / 77.8)

**Gap analysis**:
- Largest gap: pmF1 (77.1 vs 85.3) → localization quality
- Smaller gap: IL-MCC (0.82 vs 0.87) → recognition accuracy

**Conclusion**: SAM 3 approaching human-level performance, especially on recognition (presence detection).

---

## Metric Sensitivity Analysis

### Impact of Presence Head

**Without presence head**:
- CGF1 = 57.6
- IL-MCC = 0.77 (poor recognition)
- pmF1 = 74.7 (decent localization)

**With presence head**:
- CGF1 = 63.3 (+9.9%)
- IL-MCC = 0.82 (+6.5%)
- pmF1 = 77.1 (+3.2%)

**Key insight**: Presence head improves BOTH recognition (IL-MCC) and localization (pmF1).

### Impact of Hard Negatives

**0 hard negatives**:
- CGF1 = 31.8
- IL-MCC = 0.44 (terrible recognition!)
- pmF1 = 70.2 (localization okay)

**30 hard negatives**:
- CGF1 = 49.2 (+54.8%)
- IL-MCC = 0.68 (+54.5%)
- pmF1 = 72.3 (+3.0%)

**Key insight**: Hard negatives almost entirely improve recognition (IL-MCC), not localization.

---

## ARR-COC Connection (10%)

### Propositional Knowing (IL-MCC)

**IL-MCC measures propositional knowing**: "Is concept X present?" (true/false)

**Vervaeke's propositional knowing**:
- Declarative statements ("This IS a cat")
- Truth-value bearing (true or false)
- Explicit representation

**SAM 3 presence token**:
- Explicit prediction: P(concept present) = 0.82
- Binary decision at threshold: 0.82 > 0.5 → TRUE
- IL-MCC measures accuracy of these propositional judgments

### Perspectival Knowing (pmF1)

**pmF1 measures perspectival knowing**: Spatial localization from viewer perspective

**Vervaeke's perspectival knowing**:
- Embodied understanding ("How it appears to ME")
- Agent-relative (depends on viewpoint)
- Phenomenological (experiential)

**SAM 3 localization**:
- Bounding boxes/masks = phenomenological appearance from camera viewpoint
- pmF1 measures how well model captures viewer's spatial perspective
- IoU threshold (0.5) = perceptual adequacy criterion

### CGF1 as Coupled Knowing

**CGF1 = pmF1 × IL-MCC** = Perspectival × Propositional

**Vervaeke's reciprocal narrowing**:
- Propositional (what) constrains Perspectival (where)
- Perspectival (where) validates Propositional (what)

**SAM 3 architecture mirrors this**:
- Presence token (propositional) gates proposal queries (perspectival)
- Proposal localizations (perspectival) confirm presence prediction (propositional)

**Insight**: CGF1's multiplicative structure captures reciprocal narrowing—BOTH framings must be accurate for successful concept segmentation.

---

## Summary

**SA-Co Benchmark Metrics**:

✅ **CGF1** (Primary): Combines localization × recognition (SAM 3: 65.0)
✅ **IL-MCC** (Recognition): Binary presence accuracy (SAM 3: 0.82)
✅ **pmF1** (Localization): Spatial quality on positives (SAM 3: 77.1)
✅ **Calibration**: Single threshold (0.5) across all concepts
✅ **Human comparison**: 88% of human lower bound
✅ **Sensitivity**: Presence head (+9.9% CGF1), hard negatives (+54.8% CGF1)

**Key Innovation**: CGF1 = pmF1 × IL-MCC captures reciprocal narrowing between propositional knowing (recognition) and perspectival knowing (localization), mirroring Vervaeke's coupled cognitive processes.

---

**Status**: PART 28/42 complete (66.7%)
**Next**: PART 29 - Continuing expansion...
