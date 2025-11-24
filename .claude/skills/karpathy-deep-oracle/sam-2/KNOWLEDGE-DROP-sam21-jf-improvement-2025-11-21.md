# SAM 2.1: J&F Metric Improvement (+2.8 Points)

**Change**: 78.2% (SAM 2) → 81.0% (SAM 2.1) on SA-V dataset
**Release**: October 18, 2024 (11 weeks after SAM 2)
**Impact**: State-of-the-art video segmentation performance
**Model Family**: 4 checkpoints (tiny, small, base+, large)

---

## What is J&F Metric?

**J&F = Combined Video Object Segmentation Metric**

The J&F metric is the mean of two complementary measures for evaluating video object segmentation quality:

### J (Jaccard Index)
- **What it measures**: Region similarity and area accuracy
- **Formula**: Intersection over Union (IoU) of predicted mask vs ground truth
- **Focus**: How well the pixels of predicted and ground truth masks match
- **Strength**: Evaluates overall mask coverage

### F (F-measure / Boundary F-score)
- **What it measures**: Contour accuracy and boundary localization
- **Focus**: Precision of object boundaries
- **Strength**: Evaluates edge/boundary quality

### Combined J&F Score
```
J&F = (J + F) / 2
```

This aggregated metric provides comprehensive evaluation by measuring both:
- **Region overlap** (are we segmenting the right area?)
- **Boundary precision** (are the edges accurate?)

**Why it matters**: Video segmentation requires both spatial accuracy (J) and temporal consistency across frames, which J&F captures effectively.

---

## The +2.8 Point Improvement

### Performance Gains

**SA-V Dataset (Segment Anything Video)**:
- SAM 2 (July 2024): **78.2% J&F**
- SAM 2.1 (October 2024): **81.0% J&F**
- **Improvement: +2.8 points** (3.6% relative gain)

**MOSE Validation Set**:
- SAM 2: 74.6% J&F
- SAM 2.1: 74.6% J&F (maintained performance)

**LVOS v2 Benchmark**:
- SAM 2: 79.8% J&F
- SAM 2.1: 80.6% J&F (+0.8 points)

### How It Was Achieved

According to Meta AI's research paper (OpenReview, arXiv:2408.00714):

**Primary Method: Enhanced Training Strategy**

From the SAM 2 paper:
> "Adopting such a strategy further improves the performance, e.g., we find SAM 2.1's performance improves by +2.8 J&F from 78.2 to 81.0 on SA-V"

**Key Improvements** (from Meta AI blog, October 18, 2024):

1. **Additional Training Data**
   - Expanded dataset with more diverse video scenarios
   - Focus on challenging cases (occlusions, similar objects, small objects)

2. **New Data Augmentation Techniques**
   - Better handling of visually similar objects
   - Improved small object detection
   - Enhanced robustness to appearance variations

3. **Training-Free Improvements**
   - No architectural changes required
   - Same model architecture as SAM 2
   - Backwards compatible with existing code

**What Did NOT Change**:
- Model architecture (same Hiera backbone + memory attention)
- Inference speed (maintains real-time performance)
- Promptable interface (same interaction paradigm)
- Memory mechanism (streaming memory for video)

---

## Performance Comparison

### SAM 2.1 Model Family (Released September 29, 2024)

| Model | Size (M) | Speed (FPS) | SA-V (J&F) | MOSE (J&F) | LVOS v2 (J&F) |
|-------|----------|-------------|------------|------------|---------------|
| sam2.1_hiera_tiny | 38.9 | 91.2 | **76.5** | 71.8 | 77.3 |
| sam2.1_hiera_small | 46 | 84.8 | **76.6** | 73.5 | 78.3 |
| sam2.1_hiera_base_plus | 80.8 | 64.1 | **78.2** | 73.7 | 78.2 |
| sam2.1_hiera_large | 224.4 | 39.5 | **79.5** | 74.6 | 80.6 |

### SAM 2 Original (Released July 29, 2024)

| Model | Size (M) | Speed (FPS) | SA-V (J&F) | MOSE (J&F) | LVOS v2 (J&F) |
|-------|----------|-------------|------------|------------|---------------|
| sam2_hiera_tiny | 38.9 | 91.5 | 75.0 | 70.9 | 75.3 |
| sam2_hiera_small | 46 | 85.6 | 74.9 | 71.5 | 76.4 |
| sam2_hiera_base_plus | 80.8 | 64.8 | 74.7 | 72.8 | 75.8 |
| sam2_hiera_large | 224.4 | 39.7 | 76.0 | 74.6 | 79.8 |

**Key Observations**:
- All model sizes improved on SA-V dataset
- Improvements consistent across tiny → large models
- Speed maintained (no performance degradation)
- Most dramatic gains: base+ (+3.5 pts) and large (+3.5 pts)

---

## Technical Details

### Training Strategy Evolution

**SAM 2 Training** (July 2024):
- Model-in-the-loop data engine
- SA-V dataset (51K videos, 642K masklets)
- Standard augmentation techniques

**SAM 2.1 Training** (October 2024):
- **Enhanced data augmentation**:
  - Copy-paste augmentations for small objects
  - Color jitter for appearance robustness
  - Spatial augmentations for geometric invariance

- **Improved data diversity**:
  - Additional edge cases (occlusions, clutter)
  - More visually similar object pairs
  - Extended long-video scenarios

- **Training refinements**:
  - Better handling of object boundaries
  - Improved temporal consistency signals
  - Enhanced memory attention training

### No Architectural Changes

**Important**: SAM 2.1 uses the **exact same architecture** as SAM 2:
- Same Hiera image encoder
- Same memory attention mechanism
- Same mask decoder
- Same prompt encoder

**This means**:
- Drop-in replacement for SAM 2 checkpoints
- No code changes required
- Same inference API
- Training code released (September 30, 2024)

### Inference Remains Real-Time

Speed measurements on A100 GPU:
- **Tiny**: 91.2 FPS (real-time video processing)
- **Small**: 84.8 FPS (still real-time)
- **Base+**: 64.1 FPS (high quality, real-time)
- **Large**: 39.5 FPS (highest quality)

---

## Comparison with Other Methods

### Video Object Segmentation Benchmarks

**SA-V Test Set** (primary benchmark):
- **SAM 2.1 Large**: 79.5% J&F ← **Best**
- SAM 2 Large: 76.0% J&F
- XMem: ~65% J&F (estimated)
- AOT: ~60% J&F (estimated)

**MOSE Validation** (complex scenes):
- **SAM 2.1 Large**: 74.6% J&F ← **State-of-the-art**
- SAM 2 Large: 74.6% J&F (tied)
- Previous best: ~70% J&F

**LVOS v2** (long videos):
- **SAM 2.1 Large**: 80.6% J&F ← **Best**
- SAM 2 Large: 79.8% J&F
- XMem: ~75% J&F

### Key Advantages Over SAM 2

1. **Better small object handling** (+data augmentation)
2. **Improved similar object discrimination** (+training diversity)
3. **Enhanced boundary accuracy** (+refined training)
4. **Maintained speed** (no architectural overhead)
5. **Backwards compatible** (same API, drop-in replacement)

---

## Sources

**Primary Research Paper**:
- Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714, 2024
- OpenReview: https://openreview.net/forum?id=Ha6RTeWMd0 (accessed 2025-11-21)

**Official Announcements**:
- Meta AI Blog: "Sharing new research, models, and datasets from Meta FAIR" (October 18, 2024)
  - https://ai.meta.com/blog/fair-news-segment-anything-2-1 (accessed 2025-11-21)

**GitHub Repository**:
- facebookresearch/sam2: SAM 2.1 checkpoints and release notes
  - https://github.com/facebookresearch/sam2 (accessed 2025-11-21)
  - Release notes: September 29, 2024 checkpoint release

**Technical Documentation**:
- J&F Metric explained: BayernCollab DLMA - Video Object Segmentation
  - https://collab.dvb.bayern/spaces/TUMdlma/pages/309631382/Video+Object+Segmentation
- Disney Research: "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation"
  - Perazzi et al., arXiv (accessed 2025-11-21)

**Additional Analysis**:
- Encord Blog: "Meta's SAM 2.1 Explained: Improved Performance & Usability" (October 22, 2024)
- Ultralytics Blog: "AI Research Updates from Meta FAIR: SAM 2.1 and CoTracker3"
- Medium: "Why Meta's SAM 2.1 Will Change Video Segmentation Forever" (Mirza Samad)

---

**Research Date**: 2025-11-21
**SAM 2 Release**: July 29, 2024
**SAM 2.1 Release**: October 18, 2024 (checkpoints September 29, 2024)
**Improvement Timeline**: 11 weeks between releases
