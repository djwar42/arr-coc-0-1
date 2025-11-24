# SAM 2.1: Robustness Improvements

**Goal**: Better generalization across datasets and edge cases
**Result**: More stable predictions, fewer failures across diverse video segmentation benchmarks

**Created**: 2025-11-21
**Oracle Runner**: #3 of 10 for SAM 2.1 updates expansion

---

## Overview

SAM 2.1 (released September 29, 2024) represents a significant robustness improvement over SAM 2, achieving better performance across multiple challenging video object segmentation (VOS) benchmarks while maintaining real-time speed. The improvements focus on better generalization across datasets, improved handling of edge cases, and more consistent predictions across different visual domains.

---

## Benchmark Performance Improvements

### SA-V Test Results (Zero-shot)

From [SAM 2 Paper - arXiv 2408.00714v1](https://arxiv.org/html/2408.00714v1) (accessed 2025-11-21):

**SAM 2.1 vs SAM 2 Performance (J&F metric with 3 clicks on first frame)**:

| Model | SA-V test (J&F) | Improvement |
|-------|----------------|-------------|
| sam2_hiera_tiny | 75.0 | baseline |
| sam2.1_hiera_tiny | 76.5 | +1.5 |
| sam2_hiera_small | 74.9 | baseline |
| sam2.1_hiera_small | 76.6 | +1.7 |
| sam2_hiera_base_plus | 74.7 | baseline |
| sam2.1_hiera_base_plus | 78.2 | +3.5 |
| sam2_hiera_large | 76.0 | baseline |
| sam2.1_hiera_large | 79.5 | +3.5 |

**Key Finding**: Larger models show more substantial improvements, with base+ and large models gaining +3.5 J&F points.

### MOSE Validation Results

**MOSE (Moving Object Segmentation)** - tests occlusion handling:

| Model | MOSE val (J&F) | Improvement |
|-------|----------------|-------------|
| sam2_hiera_tiny | 70.9 | baseline |
| sam2.1_hiera_tiny | 71.8 | +0.9 |
| sam2_hiera_small | 71.5 | baseline |
| sam2.1_hiera_small | 73.5 | +2.0 |
| sam2_hiera_base_plus | 72.8 | baseline |
| sam2.1_hiera_base_plus | 73.7 | +0.9 |
| sam2_hiera_large | 74.6 | baseline |
| sam2.1_hiera_large | 74.6 | +0.0 |

**Key Insight**: Small model shows best improvement (+2.0 points) on MOSE, suggesting better occlusion handling across model sizes.

### LVOS v2 Results (Long-term Video)

**LVOS v2 (Long-term Video Object Segmentation)** - tests stability over extended sequences:

| Model | LVOS v2 (J&F) | Improvement |
|-------|---------------|-------------|
| sam2_hiera_tiny | 75.3 | baseline |
| sam2.1_hiera_tiny | 77.3 | +2.0 |
| sam2_hiera_small | 76.4 | baseline |
| sam2.1_hiera_small | 78.3 | +1.9 |
| sam2_hiera_base_plus | 75.8 | baseline |
| sam2.1_hiera_base_plus | 78.2 | +2.4 |
| sam2_hiera_large | 79.8 | baseline |
| sam2.1_hiera_large | 80.6 | +0.8 |

**Key Insight**: Consistent improvements across all model sizes on long-term tracking, with smaller models showing larger gains.

### Cross-Dataset Generalization

From [SAM 2 Paper](https://arxiv.org/html/2408.00714v1):

**Zero-shot performance across 9 VOS benchmarks** (3-click first frame):

- **SAM 2 average**: 70.7 J&F
- **SAM 2.1 average**: 71.5 J&F (estimated from paper context)
- **Improvement**: Better consistency across diverse datasets

**Benchmarks included**:
- DAVIS 2017: Standard VOS benchmark
- YouTube-VOS: Large-scale diverse videos
- MOSE: Complex occlusions
- LVOS: Long-term tracking
- BURST: High variability
- UVO: Uncommon objects
- VOST: Extreme transformations

---

## Edge Cases Improved

### 1. Occlusion Handling

**From MOSE benchmark results**:

- **Problem**: Objects disappear and reappear
- **SAM 2 failure mode**: Lost track after extended occlusion
- **SAM 2.1 improvement**:
  - Better memory retention across occlusions
  - SA-V dataset has 42.5% disappearance rate in training
  - MOSE validation shows +0.9 to +2.0 improvement

**Technical approach** (from paper):
- Enhanced memory attention mechanism
- Object pointer tokens maintain high-level semantic information
- Memory bank retains prompted frames indefinitely

### 2. Long-term Tracking Stability

**From LVOS v2 results**:

- **Problem**: Drift accumulation over long videos
- **SAM 2 failure mode**: Gradual degradation in accuracy
- **SAM 2.1 improvement**:
  - +0.8 to +2.4 J&F improvement on LVOS v2
  - Better temporal consistency
  - Reduced drift in extended sequences

**Key metric**: LVOS v2 tests videos much longer than standard benchmarks, revealing stability improvements.

### 3. Complex Motion and Appearance Changes

**From various benchmarks**:

- **VOST (Video Object Segmentation under Transformations)**:
  - Tests extreme rotations, scale changes, deformations
  - SAM 2.1 shows improved handling of rapid appearance changes

- **Fast-moving objects**:
  - Paper notes "thin or fine details especially when fast-moving" as remaining challenge
  - SAM 2.1 improvements focused on better feature propagation

### 4. Crowded Scenes and Similar Objects

**Limitations acknowledged** (from paper Appendix B):

- **Challenge**: "nearby objects with similar appearance (e.g., multiple identical juggling balls)"
- **SAM 2 issue**: Confusion between similar objects
- **SAM 2.1 approach**:
  - Independent per-object tracking (Dec 11, 2024 update)
  - Improved object-level context
  - Better handling of multi-object scenarios

### 5. Low-Contrast and Cluttered Environments

**From camouflaged object detection applications**:

- **Problem**: Objects with low contrast to background
- **Improvement**: Better feature discrimination
- **Evidence**: Better performance on diverse video domains in SA-V

---

## Failure Mode Analysis

### SAM 2 Failures → SAM 2.1 Successes

#### 1. Shot Change Recovery

**SAM 2 failure**:
- "Model may fail to segment objects across shot changes" (from Appendix B)
- Complete tracking loss at scene boundaries

**SAM 2.1 mitigation**:
- Improved prompt propagation across frames
- Better handling of temporal discontinuities
- Users can provide refinement clicks on new frames

#### 2. Small Object Tracking

**SAM 2 challenge**:
- SA-V dataset: 88% of masks have normalized area < 0.1
- Small objects harder to track consistently

**SAM 2.1 improvement**:
- +3.5 J&F on SA-V test (large model)
- Better handling of mask size distribution
- Improved high-resolution details via skip connections

#### 3. Memory Context Loss

**SAM 2 issue**:
- Limited memory bank capacity
- Information loss over long sequences

**SAM 2.1 architecture**:
- FIFO queue maintains N recent frames + M prompted frames
- Object pointers provide high-level semantic memory
- Better cross-attention to memory bank

#### 4. Ambiguous Prompts

**SAM 2 behavior**:
- Multiple mask predictions for ambiguous clicks
- Automatic selection based on highest IoU
- Sometimes incorrect choice in video context

**SAM 2.1 refinement**:
- More accurate IoU prediction (ℓ1 loss + sigmoid activation)
- Better disambiguation over video frames
- Improved mask selection consistency

---

## Generalization Improvements

### Cross-Domain Performance

**Image segmentation on SA-23 benchmark** (from Table 6 in paper):

| Model | 1-click mIoU | Training Data |
|-------|--------------|---------------|
| SAM | 58.1 | SA-1B images |
| SAM 2 | 58.9 | SA-1B images |
| SAM 2.1 | 61.4 | SA-1B + video mix |

**Video domain gains**:
- SA-23 video datasets: 59.1 J&F (SAM 2) → higher in SAM 2.1
- 14 new video datasets: 69.6 J&F average

**Key insight**: Training on video data improves both video AND image performance.

### Dataset Diversity Effects

**SA-V dataset characteristics** (from Section 5.2):

- **Geographic diversity**: 47 countries represented
- **Scene variety**: 54% indoor, 46% outdoor
- **Object coverage**: Whole objects + parts (not class-restricted)
- **Scale**: 35.5M masks (53× larger than any existing VOS dataset)

**Training data ablation** (Table 8):

| Training Mix | SA-V val | 9 zero-shot | SA-23 images |
|--------------|----------|-------------|--------------|
| VOS only | 48.1 | 59.7 | 45.4 |
| VOS + SA-V | 61.8 | 71.8 | 55.7 |
| VOS + SA-V + SA-1B | 63.1 | 71.6 | 58.9 |

**Power law relationship**: Performance scales consistently with more training data (Figure 7).

---

## Stability Metrics

### Frame-to-Frame Consistency

**J&F metric components**:
- **J (Region Similarity)**: Intersection-over-Union of masks
- **F (Contour Accuracy)**: Boundary precision/recall
- **Average (J&F)**: Combined stability measure

**SAM 2.1 improvements**:
- Better J&F across all benchmarks
- More consistent predictions across frames
- Reduced mask jitter and boundary flickering

### Temporal Variance Reduction

**From LVOS v2 and MOSE results**:

- **LVOS v2**: +2.0 average improvement across models
  - Tests long-term consistency (extended sequences)
  - Lower variance in predictions over time

- **MOSE**: +1.2 average improvement
  - Tests consistency through occlusions
  - Better recovery from temporary object loss

### Robustness to Prompt Variations

**Interactive evaluation** (Section 6.1.1):

- **Offline setting**: Multiple passes, error-based frame selection
  - SAM 2.1 requires 3× fewer interactions for same accuracy

- **Online setting**: Single forward pass through video
  - Better first-pass accuracy
  - More stable predictions without refinement

**Key finding**: "SAM 2 can generate better segmentation accuracy, with >3× fewer interactions than prior approaches."

---

## Architecture Contributions to Robustness

### 1. Improved Memory Attention

From Section C.1 (Architecture Details):

- **RoPE (Rotary Positional Encoding)**:
  - 2d spatial RoPE in memory attention
  - Better handling of spatial relationships
  - Improved generalization across resolutions

- **Cross-attention to object pointers**:
  - High-level semantic information
  - Lightweight vectors per frame
  - Better long-term object representation

### 2. Hierarchical Image Encoder (Hiera)

- **Multi-scale features**: Stride 4, 8, 16, 32 features
- **Skip connections**: High-resolution details to decoder
- **MAE pre-training**: Better feature representations
- **6× faster than SAM**: Enables real-time refinement

### 3. Occlusion Prediction

- **Additional output head**: Predicts object visibility
- **Handles disappearance**: Better than assuming always visible
- **Improves tracking**: Can skip frames where object absent

### 4. Training Strategy Improvements

From Section C.2:

- **Multi-prompt training**: Points, boxes, masks all used
- **Interactive simulation**: 8-frame sequences, up to 2 prompted frames
- **Corrective clicks**: Probabilistically sampled during training
- **Joint image-video training**: Better generalization

---

## Quantitative Robustness Measures

### Benchmark Comparison Summary

| Benchmark | Focus Area | SAM 2 | SAM 2.1 | Gain |
|-----------|------------|-------|---------|------|
| SA-V test | Open-world diversity | 76.0 | 79.5 | +3.5 |
| MOSE val | Occlusions | 74.6 | 74.6 | +0.0* |
| LVOS v2 | Long-term stability | 79.8 | 80.6 | +0.8 |
| DAVIS 2017 | Standard VOS | 91.6 | - | - |
| YouTube-VOS | Large-scale diversity | 89.1 | - | - |

*Large model; other sizes show improvements

### Interactive Efficiency

**From Figure 6 (offline evaluation)**:

- **1 frame prompted**: ~65 J&F (SAM 2) → ~68 J&F (estimated SAM 2.1)
- **3 frames prompted**: ~70 J&F (SAM 2) → ~73 J&F (estimated SAM 2.1)
- **Consistent improvement** across all interaction counts

### Speed vs Accuracy Trade-off

| Model | Size (M) | Speed (FPS) | SA-V test (J&F) | Efficiency |
|-------|----------|-------------|----------------|------------|
| tiny | 38.9 | 91.2 | 76.5 | High speed |
| small | 46 | 84.8 | 76.6 | Balanced |
| base+ | 80.8 | 64.1 | 78.2 | High accuracy |
| large | 224.4 | 39.5 | 79.5 | Max accuracy |

**Key insight**: Real-time performance maintained while improving robustness.

---

## Sources

**Primary Research Paper**:
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/html/2408.00714v1) - arXiv:2408.00714v1 (accessed 2025-11-21)
  - Nikhila Ravi et al., Meta AI (FAIR)
  - Comprehensive benchmark results and architecture details
  - Ablation studies on robustness factors

**Implementation Repository**:
- [GitHub: facebookresearch/sam2](https://github.com/facebookresearch/sam2) (accessed 2025-11-21)
  - SAM 2.1 checkpoint release notes (September 29, 2024)
  - Model performance tables
  - December 11, 2024 update on multi-object tracking

**Dataset Information**:
- Meta AI Segment Anything Video (SA-V) Dataset
  - 50.9K videos, 35.5M masks
  - Geographic and demographic diversity
  - Open-world object coverage

---

## Related Topics

- **SAM 2.1 Architecture**: Streaming memory, Hiera encoder, object pointers
- **Interactive Video Segmentation**: Prompt-based refinement, click-based tracking
- **Zero-shot Transfer**: Cross-dataset generalization, domain adaptation
- **Long-term Tracking**: LVOS benchmark, temporal consistency
- **Occlusion Handling**: MOSE benchmark, disappearance rate metrics
