# SA-Co/VEval Video Benchmark Structure

## Overview

SA-Co/VEval is the video evaluation component of the SA-Co (Segment Anything with Concepts) benchmark, specifically designed to evaluate SAM 3's video segmentation and tracking capabilities with concept prompts. It represents a significant advancement in video object segmentation benchmarks by incorporating open-vocabulary concept understanding.

## Dataset Composition

### Three Video Domains

SA-Co/VEval comprises videos from three distinct domains to test model robustness:

**1. SA-V Domain**
- Source: [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/) from Meta
- Content: General video segmentation scenes
- License: CC-BY-NC 4.0
- Characteristics: Diverse real-world videos with multiple objects

**2. YT-Temporal-1B Domain**
- Source: [YT-Temporal-1B dataset](https://cove.thecvf.com/datasets/704)
- Content: YouTube videos with temporal annotations
- License: CC-BY-NC 4.0
- Additional file: `yt1b_start_end_time.json` specifies video IDs and temporal segments used

**3. SmartGlasses Domain**
- Source: Egocentric video from smart glasses recordings
- Content: First-person viewpoint videos
- License: CC-BY-4.0
- Media: Preprocessed JPEGImages available in `saco_sg.tar.gz`
- Unique perspective: Tests model performance on egocentric vision

### Data Splits

Each domain includes:
- **Validation split**: For model development and hyperparameter tuning
- **Test split**: For final evaluation and benchmarking

## Annotation Format

### YTVIS-Compatible Format

SA-Co/VEval follows a format similar to [YouTube-VIS](https://youtube-vos.org/dataset/vis/) with five main fields:

**1. info**
```json
{
  "version": "v1",
  "date": "2025-09-24",
  "description": "SA-Co/VEval SA-V Test"
}
```

**2. videos**
List of video metadata:
```json
{
  "id": int,
  "video_name": str,      // e.g., "sav_000000"
  "file_names": [str],    // Frame file paths
  "height": int,
  "width": int,
  "length": int           // Number of frames
}
```

**3. annotations**
List of **positive** masklets (temporal mask sequences):
```json
{
  "id": int,
  "segmentations": [RLE],        // RLE-encoded masks per frame
  "bboxes": [[x, y, w, h]],      // Bounding boxes per frame
  "areas": [int],                 // Mask areas per frame
  "iscrowd": int,
  "video_id": int,                // Links to videos.id
  "height": int,
  "width": int,
  "category_id": int,             // Links to categories.id
  "noun_phrase": str              // Text concept
}
```

**4. categories**
Global noun phrase ID mapping (consistent across all domains):
```json
{
  "id": int,
  "name": str                     // The noun phrase
}
```

**5. video_np_pairs**
Complete list of video-noun phrase pairs (positive AND negative):
```json
{
  "id": int,
  "video_id": int,
  "category_id": int,
  "noun_phrase": str,
  "num_masklets": int             // 0 = negative, >0 = positive
}
```

### Positive vs Negative Prompts

- **Positive prompts**: `num_masklets > 0` - concept appears in video
- **Negative prompts**: `num_masklets = 0` - concept NOT in video (hard negatives)

Hard negatives are critical for testing model discrimination capabilities.

## Temporal Annotation Details

### Masklet Structure

Masklets are temporal sequences of masks tracking the same object instance:
- Each masklet has consistent identity across frames
- RLE (Run-Length Encoding) used for efficient mask storage
- Bounding boxes provided for each frame
- Areas computed for each frame's mask

### Instance Identity Preservation

Unlike single-image annotation:
- Same `annotation.id` links masks across all frames
- Tracks object through occlusions, motion, appearance changes
- Enables evaluation of temporal consistency

## Evaluation Metrics

### Primary Video Metrics

**cgF1 (Classification-Gated F1)**
- Combines localization quality with binary classification
- Formula: `cgF1 = 100 × pmF1 × IL_MCC`
- Measures both "what" and "where" accuracy

**pHOTA (Promptable Higher Order Tracking Accuracy)**
- Extension of HOTA for concept segmentation
- Balances detection, association, and localization
- Key metric for video tracking evaluation

### SAM 3 Video Results on SA-Co/VEval

| Domain | Human cgF1 | Human pHOTA | SAM 3 cgF1 | SAM 3 pHOTA |
|--------|------------|-------------|------------|-------------|
| SA-V test | 53.1 | 70.5 | 30.3 | 58.0 |
| YT-Temporal-1B test | 71.2 | 78.4 | 50.8 | 69.9 |
| SmartGlasses test | 58.5 | 72.3 | 36.4 | 63.6 |

SAM 3 achieves approximately 75-80% of human performance (measured in pHOTA).

### Additional Video Benchmarks

SA-Co/VEval complements other video benchmarks:
- **LVVIS test**: SAM 3 achieves 36.3 mAP
- **BURST test**: SAM 3 achieves 44.5 HOTA

## Download and Usage

### HuggingFace Access

Dataset hosted at: [facebook/SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval)

**Note**: Requires agreeing to Meta's terms to access.

### Roboflow Access

Alternative hosting at: [SA-Co VEval on Roboflow](https://universe.roboflow.com/sa-co-veval)

### Repository Structure

```
datasets/facebook/SACo-VEval/tree/main/
├── annotation/
│   ├── saco_veval_sav_test.json
│   ├── saco_veval_sav_val.json
│   ├── saco_veval_smartglasses_test.json
│   ├── saco_veval_smartglasses_val.json
│   ├── saco_veval_yt1b_test.json
│   └── saco_veval_yt1b_val.json
└── media/
    ├── saco_sg.tar.gz              # SmartGlasses frames
    └── yt1b_start_end_time.json    # YouTube video timestamps
```

### Preparation Instructions

Full preparation details in [SAM 3 GitHub VEval README](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/veval).

### Example Notebook

Visualization examples available:
- [`saco_veval_vis_example.ipynb`](https://github.com/facebookresearch/sam3/blob/main/examples/saco_veval_vis_example.ipynb)

## Key Features for Evaluation

### Temporal Tracking Challenges

SA-Co/VEval tests:
1. **Object persistence**: Tracking through full video duration
2. **Occlusion handling**: Maintaining identity when hidden
3. **Re-identification**: Recovering lost tracks
4. **Multiple instances**: Tracking all objects matching concept
5. **Appearance variation**: Handling viewpoint/lighting changes

### Open-Vocabulary Video Understanding

Unlike fixed-category video benchmarks:
- Tests arbitrary noun phrase understanding
- Includes hard negative prompts
- Requires semantic understanding + temporal tracking
- Bridges gap between VOS and open-vocabulary detection

### Domain Diversity

Three domains ensure robustness:
- **General scenes** (SA-V)
- **Web videos** (YT-Temporal-1B)
- **Egocentric** (SmartGlasses)

## Comparison with Related Benchmarks

| Benchmark | Type | Vocabulary | Temporal | Negatives |
|-----------|------|------------|----------|-----------|
| SA-Co/VEval | Video | Open (270K concepts) | Full video | Yes |
| LVVIS | Video | Fixed (1K classes) | Full video | No |
| BURST | Video | Fixed | Full video | No |
| YouTube-VIS | Video | Fixed (40 classes) | Full video | No |
| SA-V | Video | None (visual prompts) | Full video | No |

SA-Co/VEval uniquely combines open-vocabulary concepts with temporal annotation and hard negatives.

## Technical Implementation Notes

### RLE Encoding

Masks use COCO-style RLE encoding via [pycocotools](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py):
- Efficient storage for binary masks
- Compatible with standard evaluation tools
- Each frame has separate RLE in segmentations list

### Evaluation Protocol

1. Load annotation JSON for target domain/split
2. Process video frames sequentially
3. For each video-noun phrase pair:
   - If positive: generate masklets, compare with GT
   - If negative: should produce no masks
4. Compute cgF1 and pHOTA across all pairs

## Sources

**GitHub Repository:**
- [SAM 3 GitHub - VEval](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/veval)
- [Example notebook](https://github.com/facebookresearch/sam3/blob/main/examples/saco_veval_vis_example.ipynb)

**HuggingFace:**
- [facebook/SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval)

**Documentation:**
- [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/)
- [Meta AI Blog](https://ai.meta.com/blog/segment-anything-model-3/)

**Research Paper:**
- [SAM 3: Segment Anything with Concepts](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) (OpenReview, ICLR 2026 submission)

**Related Datasets:**
- [SA-V Dataset](https://ai.meta.com/datasets/segment-anything-video/)
- [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
- [YouTube-VIS](https://youtube-vos.org/dataset/vis/)
