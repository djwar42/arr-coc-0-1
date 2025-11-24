# SAM 3: SA-Co Dataset Creation

## Overview

The SA-Co (Segment Anything with Concepts) dataset is the largest high-quality open-vocabulary segmentation dataset to date, created to train and benchmark SAM 3's Promptable Concept Segmentation (PCS) capabilities. It contains images and videos paired with text labels (noun phrases), each exhaustively annotated with masks for all matching object instances.

## Dataset Tiers and Quality Levels

### SA-Co/Gold - Highest Quality Evaluation

**Purpose**: Primary benchmark for measuring model performance with human agreement baseline

**Key Characteristics**:
- **Triple annotation**: Each image-NP pair annotated by 3 independent human annotators
- **Human performance measurement**: Multiple annotations allow measuring inter-annotator agreement as upper bound
- **7 annotation domains** covering diverse visual concepts:
  1. MetaCLIP captioner NPs
  2. SA-1B captioner NPs
  3. Attributes
  4. Crowded Scenes
  5. Wiki-Common1K
  6. Wiki-Food/Drink
  7. Wiki-Sports Equipment

**Data Statistics**:
| Domain | Image-NPs | Image-NP-Masks |
|--------|-----------|----------------|
| MetaCLIP captioner NPs | 33,393 | 20,144 |
| SA-1B captioner NPs | 13,258 | 30,306 |
| Attributes | 9,245 | 3,663 |
| Crowded Scenes | 20,687 | 50,417 |
| Wiki-Common1K | 65,502 | 6,448 |
| Wiki-Food/Drink | 13,951 | 9,825 |
| Wiki-Sports Equipment | 12,166 | 5,075 |

**Image Sources**:
- MetaCLIP images (6 out of 7 subsets)
- SA-1B images (1 subset)

**Evaluation Approach**: Oracle setting - evaluate against each of 3 annotations and pick most favorable result, accounting for valid interpretation differences.

From [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md):
- Dashed borders indicate "group masks" covering multiple instances when separation is too difficult
- Annotators may disagree on mask borders, instance counts, and phrase existence
- Three independent annotations enable measuring human agreement ceiling

---

### SA-Co/Silver - Large-Scale Single Annotation

**Purpose**: Broader coverage evaluation with more domains but single ground-truth

**Key Characteristics**:
- **Single annotation**: One human annotator per image-NP pair
- **10 diverse domains** including specialized areas:
  1. BDD100k (driving)
  2. DROID (robotics)
  3. Ego4D (egocentric video)
  4. MyFoodRepo-273 (food)
  5. GeoDE (geographic diversity)
  6. iNaturalist-2017 (nature/wildlife)
  7. National Gallery of Art (art)
  8. SA-V (video frames)
  9. YT-Temporal-1B (YouTube video frames)
  10. Fathomnet (underwater/marine)

**Data Statistics**:
| Domain | Image-NPs | Image-NP-Masks |
|--------|-----------|----------------|
| BDD100k | 5,546 | 13,210 |
| DROID | 9,445 | 11,098 |
| Ego4D | 12,608 | 24,049 |
| MyFoodRepo-273 | 20,985 | 28,347 |
| GeoDE | 14,850 | 7,570 |
| iNaturalist-2017 | 1,439,051 | 48,899 |
| National Gallery of Art | 22,294 | 18,991 |
| SA-V | 18,337 | 39,683 |
| YT-Temporal-1B | 7,816 | 12,221 |
| Fathomnet | 287,193 | 14,174 |

**Important Caveat**: Results may have more variance and tend to underestimate model performance since they don't account for valid alternative interpretations.

---

### SA-Co/VEval - Video Evaluation Benchmark

**Purpose**: Evaluate temporal consistency and tracking in video segmentation

**Key Characteristics**:
- **3 video domains** with val/test splits:
  1. **SA-V**: Videos from Segment Anything Video dataset (24fps)
  2. **YT-Temporal-1B**: YouTube videos (6fps)
  3. **SmartGlasses**: Egocentric videos (6fps)

**Licenses**:
- SA-V: CC-BY-NC 4.0
- YT-Temporal-1B: CC-BY-NC 4.0
- SmartGlasses: CC-BY-4.0

**Annotation Format**: Similar to YTVIS (YouTube Video Instance Segmentation) format with:
- `videos`: Video metadata (id, name, file_names, dimensions, length)
- `annotations`: Positive masklets with temporal segmentations
- `categories`: Global noun phrase ID mapping
- `video_np_pairs`: Both positive and negative video-NP pairs

**Video-NP Pair Structure**:
- `num_masklets > 0`: Positive pair (object present)
- `num_masklets = 0`: Negative pair (object not present)

---

## Annotation Methodology

### Multi-Phase Data Engine

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

Meta built a four-phase data engine combining humans, SAM models, and fine-tuned LLMs:

**Phases 1-3: Image Annotation**
- Progressive automation increase
- AI annotators propose candidate noun phrases
- AI verifiers (fine-tuned Llama 3.2) assess mask quality and exhaustivity
- Human effort concentrated on failure cases
- **Result**: 2x throughput compared to human-only pipelines

**Phase 4: Video Extension**
- Extended methodology to video annotation
- Temporal consistency requirements
- Masklet tracking across frames

### Annotation Format (COCO-Derived)

**Images Field**:
```json
{
  "id": 10000000,
  "file_name": "path/to/image.jpeg",
  "text_input": "chili",
  "width": 600,
  "height": 600,
  "is_instance_exhaustive": 1,
  "is_pixel_exhaustive": 1
}
```

**Key Fields**:
- `text_input`: The noun phrase for the image-NP pair
- `is_instance_exhaustive`: 1 = all instances correctly annotated (for instance segmentation)
- `is_pixel_exhaustive`: 1 = all pixels covered (weaker, allows crowd segments for semantic segmentation)

**Annotations Field**:
```json
{
  "id": 1,
  "image_id": 10000000,
  "bbox": [x, y, w, h],
  "segmentation": {"counts": "RLE_STRING", "size": [h, w]},
  "category_id": 1,
  "iscrowd": 0
}
```

**Key Fields**:
- `iscrowd`: 1 = segment covers multiple instances (when separation infeasible)
- Bounding boxes normalized by image dimensions
- Segmentation in RLE (Run-Length Encoding) format

### Positive vs Negative NPs

- **Positive NP**: `id` in images has corresponding annotations (object present)
- **Negative NP**: `id` in images has no annotations (object not present)
- Negative prompts shown in red font in visualizations

---

## Quality Control Mechanisms

### Gold Dataset Triple Review

From [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md):

Each Gold datapoint receives:
1. **Three independent human annotators**
2. **No inter-annotator communication** during annotation
3. **Disagreement tolerance** for:
   - Precise mask borders
   - Number of instances
   - Whether phrase exists in image

This enables measuring human agreement as performance ceiling (75-80% for SAM 3).

### Group Masks for Ambiguity

When separating instances is too difficult (e.g., overlapping objects, poor image quality):
- **Group masks** (dashed borders) cover multiple instances
- Marked with `iscrowd: 1`
- Used when instances are not separable

### AI-Assisted Quality Verification

From data engine documentation:
- Fine-tuned Llama 3.2 models assess:
  - Mask quality
  - Annotation exhaustivity
  - Phrase-mask alignment
- Human review focused on AI-flagged failure cases

---

## Dataset Scale

### Overall Statistics

From [SAM 3 GitHub README](https://github.com/facebookresearch/sam3):

- **Total unique concepts**: 270K (50x more than existing benchmarks)
- **Total unique phrases**: 4M+
- **Total masks**: ~1.4B
- **Images**: ~5.2M high-quality images
- **Videos**: 52.5K videos

### SA-Co Ontology

- **22 million entities**
- **17 top-level categories**
- **72 sub-categories**
- Coverage from common objects to long-tail concepts

---

## Hosting and Access

### HuggingFace
- [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)
- [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver)
- [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)

### Roboflow Universe
- [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold)
- [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver)
- [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

---

## Evaluation Metrics

### Primary Metric: cgF1

The official metric for SA-Co/Gold and SA-Co/Silver is **cgF1** (concept-grounded F1).

From [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md):
- Evaluator inherits from official COCO evaluator with modifications
- For Gold: Evaluate against each of 3 annotations, pick most favorable (oracle setting)
- Minimal dependencies: pycocotools, numpy, scipy

### Video Metrics

For SA-Co/VEval:
- **pHOTA** (panoptic Higher Order Tracking Accuracy)
- Standard video instance segmentation metrics

---

## Key Insights

### Why Triple Annotation Matters

1. **Ambiguity handling**: Natural language prompts have inherent ambiguity
2. **Human baseline**: Measures realistic upper bound (not 100%)
3. **Fair evaluation**: Models not penalized for valid alternative interpretations

### Gold vs Silver Trade-offs

| Aspect | Gold | Silver |
|--------|------|--------|
| Annotation depth | Triple | Single |
| Domain count | 7 | 10 |
| Variance | Lower | Higher |
| Performance estimate | More accurate | May underestimate |
| Use case | Primary benchmark | Broader coverage |

### Video Annotation Challenges

From [VEval README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/veval/README.md):

**YT-Temporal-1B Caveats**:
- YouTube videos may become unavailable (deleted/private)
- Video specs may differ from annotation time
- Frame extraction inconsistencies across environments
- Frame shifting alignment issues possible

---

## Sources

**GitHub Documentation**:
- [SA-Co/Gold README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md)
- [SA-Co/Silver README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/silver/README.md)
- [SA-Co/VEval README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/veval/README.md)
- [SAM 3 Main README](https://github.com/facebookresearch/sam3)

**Web Research**:
- [Roboflow Blog: What is SAM 3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23)
- [Ultralytics YOLO Docs: SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23)

**Dataset Hosting**:
- [HuggingFace SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)
- [HuggingFace SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver)
- [HuggingFace SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
- [Roboflow Universe SA-Co](https://universe.roboflow.com/sa-co-gold)
