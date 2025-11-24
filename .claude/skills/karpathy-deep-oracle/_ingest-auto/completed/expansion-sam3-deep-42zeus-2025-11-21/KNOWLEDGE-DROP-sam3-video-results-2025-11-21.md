# SAM 3: Video Segmentation Results

## Overview

SAM 3 achieves state-of-the-art video segmentation performance across multiple benchmarks, including SA-V, YT-Temporal-1B, SmartGlasses, LVVIS, and BURST. The model unifies promptable concept segmentation with temporal tracking, achieving 75-80% of human performance on video tasks.

## Video Benchmark Results

### SA-Co/VEval Video Benchmarks

SAM 3 is evaluated on three primary video domains within the SA-Co/VEval benchmark:

| Benchmark | cgF1 | pHOTA | Notes |
|-----------|------|-------|-------|
| **SA-V test** | 30.3 | 58.0 | Segment Anything Video dataset |
| **YT-Temporal-1B test** | 50.8 | 69.9 | YouTube Temporal 1B dataset |
| **SmartGlasses test** | 36.4 | 63.6 | Egocentric smart glasses footage |

### Human Performance Comparison

| Benchmark | Human cgF1 | Human pHOTA | SAM 3 vs Human |
|-----------|------------|-------------|----------------|
| SA-V test | 53.1 | 70.5 | ~57% cgF1, ~82% pHOTA |
| YT-Temporal-1B test | 71.2 | 78.4 | ~71% cgF1, ~89% pHOTA |
| SmartGlasses test | 58.5 | 72.3 | ~62% cgF1, ~88% pHOTA |

From [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) and [MarkTechPost Article](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) (accessed 2025-11-23)

### Additional Video Benchmarks

| Benchmark | Metric | Score | Description |
|-----------|--------|-------|-------------|
| **LVVIS test** | mAP | 36.3 | Large Vocabulary Video Instance Segmentation |
| **BURST test** | HOTA | 44.5 | Benchmark for Unifying Recognition, Segmentation, and Tracking |

## Understanding the Metrics

### cgF1 (Concept-Grounded F1)

**What it measures**: How well the model recognizes AND localizes concepts in video

**Components**:
- **Precision**: Of all segments predicted as concept X, how many are correct?
- **Recall**: Of all actual instances of concept X, how many were found?
- **F1**: Harmonic mean of precision and recall

**Why it matters for SAM 3**: cgF1 evaluates the core PCS task - finding all instances of a concept described by text prompt. A score of 30.3 on SA-V means the model correctly finds and segments about 30% of all concept instances with correct localization.

From [Meta AI SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) (accessed 2025-11-23)

### pHOTA (Promptable Higher Order Tracking Accuracy)

**What it measures**: Combined tracking accuracy across detection, association, and localization

**Based on HOTA (Higher Order Tracking Accuracy)**:
- **Detection Accuracy (DetA)**: How well are objects detected?
- **Association Accuracy (AssA)**: How well are tracks maintained over time?
- **Localization Accuracy (LocA)**: How accurate are the masks/boxes?

**pHOTA Formula**: Geometric mean of sub-metrics, adapted for promptable segmentation

**Why it matters for SAM 3**: pHOTA captures temporal consistency - whether the model maintains correct identity associations across frames. A pHOTA of 58.0 on SA-V indicates strong but imperfect tracking continuity.

From [HOTA Paper by Luiten et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881978/) - Cited by 1176+ papers

### HOTA (Higher Order Tracking Accuracy)

**Standard video tracking metric** used on BURST benchmark.

**Key properties**:
- Balances detection and association errors equally
- Decomposes into sub-metrics for error analysis
- More informative than legacy MOTA metric

SAM 3 achieves **44.5 HOTA** on BURST test.

### mAP (Mean Average Precision)

**Standard detection metric** used on LVVIS benchmark.

SAM 3 achieves **36.3 mAP** on LVVIS test.

## Video Benchmark Descriptions

### SA-V (Segment Anything Video)

**Description**: High-quality video segmentation benchmark from Meta's SA-V dataset

**Characteristics**:
- Diverse video content
- Dense instance annotations
- Open-vocabulary concept prompts
- Both active and completed tracking

**Challenge**: General video object segmentation with text prompts

### YT-Temporal-1B (YouTube Temporal 1B)

**Description**: Large-scale YouTube video dataset for temporal understanding

**Characteristics**:
- Web-crawled diverse content
- Temporal grounding tasks
- 1B+ video samples in full dataset
- Various durations and qualities

**SAM 3 Performance**: 50.8 cgF1, 69.9 pHOTA - best performance among video benchmarks

### SmartGlasses

**Description**: Egocentric video from smart glasses devices

**Characteristics**:
- First-person viewpoint
- Rapid camera motion
- Object occlusions
- Real-world dynamic scenes

**Challenge**: Egocentric understanding with motion blur and perspective shifts

### LVVIS (Large Vocabulary Video Instance Segmentation)

**Description**: Video instance segmentation with large vocabulary of categories

**Characteristics**:
- 1,000+ categories
- Instance-level masks
- Video object tracking
- Open-world evaluation

**Metric**: mAP (Mean Average Precision) = 36.3

### BURST (Benchmark for Unifying Recognition, Segmentation, and Tracking)

**Description**: Unified benchmark combining multiple video understanding tasks

**Characteristics**:
- Object recognition
- Instance segmentation
- Multi-object tracking
- Open-world evaluation

**Metric**: HOTA = 44.5

From [BURST Paper](https://www.researchgate.net/publication/368311615_BURST_A_Benchmark_for_Unifying_Object_Recognition_Segmentation_and_Tracking_in_Video) (accessed 2025-11-23)

## Temporal Consistency Analysis

### What Makes Video Segmentation Challenging

**Temporal challenges** that SAM 3 addresses:

1. **Identity Maintenance**: Same object must have consistent ID across frames
2. **Occlusion Handling**: Objects may disappear and reappear
3. **Appearance Changes**: Lighting, viewpoint, and deformation changes
4. **Scale Variations**: Objects move closer/farther from camera
5. **Motion Blur**: Fast-moving objects or camera motion

### SAM 3 Temporal Architecture

**Inherited from SAM 2**:
- Streaming memory attention
- Temporal propagation mechanisms
- Interactive point-based refinement

**New in SAM 3**:
- Decoupled detector-tracker design
- Presence token for concept discrimination
- Text-conditioned detection

### Performance Patterns

**Best performance**: YT-Temporal-1B (69.9 pHOTA)
- Possibly more diverse training overlap
- Longer clips allow better temporal modeling

**Most challenging**: SA-V (58.0 pHOTA)
- Strictest annotation quality
- Novel concepts less seen in training

**Egocentric challenge**: SmartGlasses (63.6 pHOTA)
- First-person viewpoint is harder than third-person
- Rapid motion and occlusions

## Comparison with Human Performance

### Overall Video Performance

SAM 3 achieves approximately **75-80% of human pHOTA** across video benchmarks:

| Benchmark | SAM 3 pHOTA | Human pHOTA | Ratio |
|-----------|-------------|-------------|-------|
| SA-V | 58.0 | 70.5 | 82.3% |
| YT-Temporal-1B | 69.9 | 78.4 | 89.2% |
| SmartGlasses | 63.6 | 72.3 | 87.9% |

### Remaining Gap Analysis

**Where SAM 3 struggles vs humans**:

1. **Fine-grained concept discrimination**: "player in white" vs "player in slightly off-white"
2. **Novel concepts**: Concepts not well-represented in training data
3. **Severe occlusions**: Objects hidden for many frames
4. **Ambiguous prompts**: Text that could match multiple interpretations

## 2x Improvement Over Prior Systems

**Key claim**: SAM 3 delivers 2x gain over existing systems in video PCS

**Comparison context**:
- Prior systems (OWLv2, DINO-X, Gemini 2.5) primarily designed for images
- Video tracking typically done with separate tracker
- SAM 3 unifies detection + tracking with concept understanding

**Why 2x improvement**:
1. Unified architecture (no handoff between detector and tracker)
2. Presence token for precise concept matching
3. 4M concept training data (vs ~1K-10K for prior work)
4. SA-Co benchmark designed for fair evaluation

## Implementation Details

### Video Predictor API

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

# Start video session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="<VIDEO_PATH>",  # JPEG folder or MP4
    )
)

# Add text prompt
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,
        text="<CONCEPT_PROMPT>",
    )
)

# Get tracked outputs
outputs = response["outputs"]
```

From [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) (accessed 2025-11-23)

### Output Format

Video predictions include:
- Instance masks per frame
- Stable instance IDs across frames
- Confidence scores
- Bounding boxes (optional)

## Key Takeaways

### Performance Summary

1. **Best overall pHOTA**: 69.9 on YT-Temporal-1B (89% of human)
2. **Most challenging**: SA-V at 30.3 cgF1, 58.0 pHOTA
3. **Egocentric**: SmartGlasses shows strong egocentric understanding
4. **Open-vocabulary**: LVVIS (36.3 mAP) and BURST (44.5 HOTA) confirm generalization

### Architectural Contributions to Video Performance

1. **Decoupled design**: Detector and tracker share encoder but have separate heads
2. **Presence token**: Discriminates closely related prompts across frames
3. **Memory attention**: Inherited from SAM 2 for temporal propagation
4. **Text conditioning**: Enables open-vocabulary video tracking

### Remaining Challenges

1. **cgF1 gap**: 30-50 cgF1 vs 50-70 human shows room for improvement in concept recognition
2. **Temporal consistency**: ~80-90% of human pHOTA indicates tracking improvements needed
3. **Egocentric scenarios**: SmartGlasses still challenging with motion blur
4. **Long-horizon tracking**: Objects over very long videos (hours) not fully evaluated

## Sources

**Primary Sources**:
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) - Official code and results tables
- [Meta AI SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) - Technical details
- [Meta AI Blog Post](https://ai.meta.com/blog/segment-anything-model-3/) - Overview and motivation

**News Coverage**:
- [MarkTechPost Article](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) - Detailed benchmark breakdown

**Metric References**:
- [HOTA Paper (Luiten et al., 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881978/) - HOTA metric definition
- [BURST Benchmark Paper](https://www.researchgate.net/publication/368311615_BURST_A_Benchmark_for_Unifying_Object_Recognition_Segmentation_and_Tracking_in_Video) - BURST dataset

All sources accessed 2025-11-23.
