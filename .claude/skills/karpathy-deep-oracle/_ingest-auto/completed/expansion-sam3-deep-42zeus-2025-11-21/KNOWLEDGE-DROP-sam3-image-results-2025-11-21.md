# SAM 3 Image Segmentation Results

## Overview

SAM 3 achieves state-of-the-art performance on image segmentation benchmarks, delivering approximately **2x improvement** over existing systems on the new SA-Co benchmark while setting new records on established benchmarks like LVIS and COCO. The model demonstrates particularly strong results in open-vocabulary and zero-shot settings where previous models struggled.

## Key Metrics

### Primary Evaluation Metric: CGF1

**Classification-Gated F1 (CGF1)** is SAM 3's primary evaluation metric, combining localization quality with recognition accuracy:

```
CGF1 = 100 x pmF1 x IL_MCC
```

Where:
- **pmF1** (Positive Macro F1): Measures localization quality on positive examples
- **IL_MCC** (Image-Level Matthews Correlation Coefficient): Measures binary classification accuracy ("is the concept present?")

This metric enforces good calibration by only evaluating predictions above 0.5 confidence, mimicking real-world usage patterns.

## Image Segmentation Benchmarks

### LVIS Zero-Shot Instance Segmentation

| Metric | SAM 3 | Previous Best | Improvement |
|--------|-------|---------------|-------------|
| **Mask AP** | **47.0** | 38.5 | +22.1% |

**Key insight**: SAM 3 achieves this without any training on LVIS categories, demonstrating strong zero-shot generalization from its open-vocabulary training on SA-Co.

**Why SAM 2 doesn't compare**: SAM 2 lacks open-vocabulary detection capability - it requires visual prompts (points, boxes, masks) to segment objects and cannot perform zero-shot detection from text prompts. SAM 3's LVIS performance represents a fundamentally new capability.

### SA-Co/Gold Benchmark (Primary)

| Model | CGF1 | IL_MCC | pmF1 |
|-------|------|--------|------|
| **SAM 3** | **65.0** | **0.82** | **77.1** |
| OWLv2 (best baseline) | 34.3 | - | - |
| Human lower bound | 74.2 | - | - |
| Human upper bound | 81.4 | - | - |

**Key results**:
- **89.5% improvement** over OWLv2 (nearly 2x better)
- Achieves **88% of human lower bound** performance
- 75-80% of overall human performance

### COCO Zero-Shot Detection

| Metric | SAM 3 | Previous Best (T-Rex2) | Improvement |
|--------|-------|------------------------|-------------|
| **Box AP** | **53.5** | 52.2 | +2.5% |

### Semantic Segmentation Benchmarks

| Benchmark | Metric | SAM 3 | Previous Best (APE-D) | Improvement |
|-----------|--------|-------|----------------------|-------------|
| **ADE-847** | mIoU | **14.7** | 9.2 | +59.8% |
| **PascalConcept-59** | mIoU | **59.4** | 58.5 | +1.5% |
| **Cityscapes** | mIoU | **65.1** | 44.2 | +47.3% |

## Comparison with Competitors

### Open-Vocabulary Detection Models

| Model | SA-Co/Gold CGF1 | LVIS Mask AP | Notes |
|-------|-----------------|--------------|-------|
| **SAM 3** | **65.0** | **47.0** | State-of-the-art |
| OWLv2 | 34.3 | 38.5 | Previous best on SA-Co |
| DINO-X | - | 35.2 | Strong zero-shot detector |
| T-Rex2 | - | 38.5 | Competitive on LVIS |
| Gemini 2.5 | - | - | MLLM-based approach |

### Why SAM 3 Outperforms

1. **Larger training vocabulary**: 270K unique concepts (50x more than existing benchmarks like LVIS's ~4K)
2. **High-quality data**: SA-Co/HQ with 5.2M images and 4M unique noun phrases
3. **Presence token mechanism**: Decouples recognition from localization
4. **Hard negative training**: 30 hard negatives per image improves IL_MCC by 54.5%

## SAM 3 vs SAM 2: Key Differences

| Capability | SAM 3 | SAM 2 |
|------------|-------|-------|
| **Open-vocabulary detection** | Yes (text prompts) | **No** |
| **Zero-shot LVIS** | 47.0 AP | N/A (requires visual prompts) |
| **Concept segmentation** | All instances from text | Not supported |
| **Visual prompting** | Yes (backward compatible) | Yes |
| **Detection capability** | Built-in DETR detector | Requires external detector |

**Critical distinction**: SAM 2 is fundamentally a **promptable visual segmentation** model that segments individual objects given geometric prompts (points, boxes, masks). It cannot:
- Detect objects from text descriptions
- Find all instances of a concept automatically
- Perform zero-shot open-vocabulary detection

SAM 3 adds these capabilities while maintaining SAM 2's visual prompting features.

## Few-Shot Adaptation

SAM 3 excels at adapting to new domains with minimal examples:

| Benchmark | 0-shot AP | 10-shot AP | Previous Best (10-shot) |
|-----------|-----------|------------|-------------------------|
| **ODinW13** | 59.9 | **71.6** | 67.9 (gDino1.5-Pro) |
| **RF100-VL** | 14.3 | **35.7** | 33.7 (gDino-T) |

## Interactive Refinement Performance

SAM 3's exemplar-based refinement shows rapid convergence:

| Prompts Added | CGF1 Score | Gain vs Text-Only |
|---------------|------------|-------------------|
| Text only | 46.4 | baseline |
| +1 exemplar | 57.6 | +11.2 |
| +2 exemplars | 62.2 | +15.8 |
| +3 exemplars | **65.0** | **+18.6** |
| +4 exemplars | 65.7 | +19.3 (plateau) |

**Key insight**: Just 3 exemplar prompts achieve near-optimal performance, making interactive refinement highly efficient.

## Object Counting Accuracy

SAM 3 provides accurate counting by segmenting all instances:

| Benchmark | Accuracy | MAE | vs Best MLLM |
|-----------|----------|-----|--------------|
| **CountBench** | **95.6%** | 0.11 | 92.4% (Gemini 2.5) |
| **PixMo-Count** | **87.3%** | 0.22 | 88.8% (Molmo-72B) |

## Inference Performance

| Metric | SAM 3 | Notes |
|--------|-------|-------|
| **Per-image latency** | **30 ms** | H200 GPU, 100+ detected objects |
| **Model size** | ~400+ MB (estimated) | Larger than SAM 2 |

## Human Performance Comparison

On SA-Co/Gold with triple human annotation:

| Annotator Type | CGF1 Score |
|----------------|------------|
| Human lower bound (conservative) | 74.2 |
| **SAM 3** | **65.0** |
| Human upper bound (liberal) | 81.4 |

**SAM 3 achievement**: 88% of estimated human lower bound

The remaining gap primarily consists of:
- Ambiguous concepts ("small window", "cozy room")
- Subjective interpretations
- Fine-grained distinctions

## Ablation Studies

### Impact of Presence Head

| Configuration | CGF1 | IL_MCC | pmF1 |
|---------------|------|--------|------|
| Without presence | 57.6 | 0.77 | 74.7 |
| **With presence** | **63.3** | **0.82** | **77.1** |

**Result**: +5.7 CGF1 boost (+9.9%), primarily improving recognition (IL_MCC +6.5%)

### Effect of Hard Negatives

| Hard Negatives/Image | CGF1 | IL_MCC | pmF1 |
|----------------------|------|--------|------|
| 0 | 31.8 | 0.44 | 70.2 |
| 5 | 44.8 | 0.62 | 71.9 |
| **30** | **49.2** | **0.68** | **72.3** |

**Result**: Hard negatives improve IL_MCC by 54.5% (0.44 -> 0.68)

### Training Data Scaling

| Data Sources | CGF1 | IL_MCC | pmF1 |
|--------------|------|--------|------|
| External only | 30.9 | 0.46 | 66.3 |
| External + Synthetic | 39.7 | 0.57 | 70.6 |
| External + HQ | 51.8 | 0.71 | 73.2 |
| **All three** | **54.3** | **0.74** | **73.5** |

**Result**: High-quality human annotations (SA-Co/HQ) provide the largest gains

## Summary

SAM 3 represents a significant advancement in image segmentation:

1. **2x improvement** on SA-Co benchmark over previous best (OWLv2)
2. **22% improvement** on LVIS zero-shot instance segmentation
3. **88% of human performance** on concept segmentation
4. First SAM model with **open-vocabulary detection** capability
5. Strong **zero-shot generalization** across diverse benchmarks

The key innovation enabling these results is the combination of:
- Decoupled detector-tracker architecture with presence token
- Massive-scale training data (270K concepts, 4M unique phrases)
- Hard negative mining for improved recognition
- Interactive exemplar-based refinement

---

## Sources

**Web Research:**
- [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) - Comprehensive benchmark tables and architecture details (accessed 2025-11-23)
- [Meta AI Blog: Segment Anything Model 3](https://ai.meta.com/blog/segment-anything-model-3/) - Official announcement with 2x performance claims (accessed 2025-11-23)
- [SAM 3 OpenReview Paper](https://openreview.net/forum?id=r35clVtGzw) - ICLR 2026 submission with full benchmark results (accessed 2025-11-23)
- [GitHub: facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository with performance metrics (accessed 2025-11-23)
- [Roboflow Blog: What is SAM3](https://blog.roboflow.com/what-is-sam3/) - Release date and benchmark summary (accessed 2025-11-23)
- [Medium: Meta SAM 3 Analysis](https://medium.com/@harsh.vardhan7695/meta-sam-3-the-ai-that-understands-find-every-red-hat-b489d341977b) - SA-Co/Gold CGF1 comparison details (accessed 2025-11-23)

**Additional References:**
- LVIS Dataset: https://www.lvisdataset.org/
- COCO Dataset: https://cocodataset.org/
