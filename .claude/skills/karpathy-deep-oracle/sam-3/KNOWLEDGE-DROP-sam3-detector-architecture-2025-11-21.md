# SAM 3 Text-Conditioned Detector Architecture

## Overview

SAM 3's detector is a **DETR-based transformer architecture** that performs open-vocabulary concept detection in images. Unlike traditional object detectors with fixed label sets, the detector is conditioned on text prompts, image exemplars, or both to find all instances of specified concepts.

The detector is part of SAM 3's unified architecture where it **shares a vision encoder** (Perception Encoder) with the tracker component. This design achieves a key innovation: decoupling detection ("what and where") from tracking ("maintain identity across time").

## Core Architecture Components

### 1. Perception Encoder (PE) - Shared Vision Backbone

The Perception Encoder provides visual features for both detector and tracker:

- **Architecture**: ViT-based (Vision Transformer) or Hiera-based encoder
- **Function**: Extracts multi-scale visual features from input images
- **Shared**: Same encoder serves both detector and tracker, enabling memory efficiency
- **Feature Alignment**: Aligns visual features with text and exemplar embeddings in a joint embedding space

From [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):
> "SAM 3 consists of a detector and a tracker that share a vision encoder. It has 848M parameters."

### 2. Text Encoder

Processes noun phrase prompts into concept embeddings:

- **Input**: Simple noun phrases (e.g., "yellow school bus", "striped cat", "person wearing a hat")
- **Architecture**: Text encoder from Meta's MetaCLIP or similar vision-language model
- **Output**: Text embeddings that represent the target concept
- **Alignment**: Trained to align with visual features in shared embedding space

From [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):
> "Text and image encoders come from Meta's [vision-language models]"

### 3. Exemplar Encoder

Processes image-based prompts (bounding boxes around example objects):

- **Input**: Bounding boxes with positive/negative labels
- **Function**: Encodes visual appearance of example objects
- **Positive Exemplars**: "Find objects that look like this"
- **Negative Exemplars**: "Do NOT include objects that look like this"
- **Generalization**: Model generalizes to similar objects, not just individual corrections

### 4. Fusion Encoder

Conditions image features on concept prompts:

- **Function**: Fuses text embeddings, exemplar embeddings, and visual features
- **Output**: Concept-conditioned feature maps
- **Design**: Cross-attention between prompt embeddings and visual features
- **Purpose**: Creates unified representation for detection

### 5. Presence Head (Key Innovation)

A global token that predicts whether the concept exists in the image:

From [Ultralytics Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):
> "Novel **presence head** that decouples recognition ('what') from localization ('where')"

**Why This Matters**:
- **Recognition vs Localization**: Separates "does this concept exist?" from "where is it?"
- **Performance Gain**: +5.7 CGF1 improvement (+9.9%)
- **IL_MCC Improvement**: Recognition ability (IL_MCC) improved by 6.5%
- **Hard Negative Handling**: Critical for distinguishing similar concepts

**Ablation Results**:

| Configuration | CGF1 | IL_MCC | pmF1 |
|--------------|------|--------|------|
| Without presence | 57.6 | 0.77 | 74.7 |
| **With presence** | **63.3** | **0.82** | **77.1** |

### 6. Mask Head

Generates instance segmentation masks:

- **Input**: Detected object proposals with concept conditioning
- **Output**: Binary masks for each detected instance
- **Architecture**: Lightweight mask decoder similar to DETR mask head
- **Per-Instance**: Separate mask for each detected object instance

## DETR-Based Detection Design

The detector follows the DETR (DEtection TRansformer) paradigm:

### Object Queries

- **Learnable Queries**: Set of learnable query embeddings
- **Concept Queries**: Queries conditioned on concept embeddings
- **Bipartite Matching**: Hungarian algorithm matches queries to objects
- **Set Prediction**: Predicts all instances in parallel (no NMS needed)

### Transformer Decoder

From [Ultralytics Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):
> "Detector: DETR-based architecture for image-level concept detection"

- **Self-Attention**: Queries attend to each other
- **Cross-Attention**: Queries attend to concept-conditioned visual features
- **Multi-Scale Features**: Processes features at multiple resolutions
- **Output**: Bounding boxes, class scores, and mask predictions

### Multi-Scale Feature Processing

- **Feature Pyramid**: Multi-scale features from Perception Encoder
- **Deformable Attention**: Efficient attention over multi-scale features (likely following Deformable DETR)
- **High Resolution**: Preserves detail for precise segmentation masks

## Text Conditioning Mechanism

### How Text Prompts Work

1. **Text Tokenization**: Noun phrase tokenized and embedded
2. **Text Encoding**: Transformer encoder produces text embeddings
3. **Feature Conditioning**: Text embeddings condition visual features via cross-attention
4. **Query Initialization**: Object queries incorporate concept information
5. **Detection**: Decoder outputs instances matching the text concept

### Supported Prompt Types

From [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):
> "The detector is a DETR-based model conditioned on text, geometry, and image exemplars."

| Prompt Type | Example | Function |
|-------------|---------|----------|
| Text (noun phrase) | "yellow school bus" | Find all instances matching description |
| Text (attribute) | "striped cat" | Find instances with specific attributes |
| Text (wearing/holding) | "person wearing red hat" | Find instances with relationships |
| Image exemplar (+) | Box around one dog | Find all similar dogs |
| Image exemplar (-) | Box around not-target | Exclude similar objects |
| Combined | "dog" + exemplar box | Precise concept matching |

### Open-Vocabulary Capability

Key advancement over SAM 1/2:

- **No Fixed Labels**: Can segment any concept describable by text
- **270K Concepts**: Trained on 270K unique concepts (50x more than existing benchmarks)
- **Zero-Shot**: Works on novel concepts without fine-tuning
- **Calibrated**: Predictions above 0.5 confidence are reliable

## Integration with Vision Encoder

### Shared Encoder Benefits

1. **Memory Efficiency**: Single encoder for both detector and tracker
2. **Feature Consistency**: Same features used for detection and tracking
3. **Training Efficiency**: Joint optimization of shared parameters
4. **Inference Speed**: ~30ms per image with 100+ detected objects on H200 GPU

### Feature Flow

```
Image → Perception Encoder → Visual Features
                                    ↓
Text → Text Encoder → Text Embeddings → Fusion Encoder → Conditioned Features
                                    ↓
                            DETR Decoder → Object Queries
                                    ↓
                    Presence Head → Does concept exist?
                            ↓              ↓
                    Yes: Boxes, Masks, Scores
                    No: Empty output
```

## Query Embeddings for Concepts

### Concept Query Design

- **Base Queries**: Learnable position embeddings (standard DETR)
- **Concept Conditioning**: Queries modulated by concept embeddings
- **Dynamic Queries**: Different concepts activate different query patterns
- **Shared Queries**: Same queries handle any concept (open-vocabulary)

### Attention Mechanism

From [Ultralytics Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):
> "Fusion encoder to condition image features on prompts"

1. **Query Self-Attention**: Queries communicate to avoid duplicate detections
2. **Query-to-Image**: Queries attend to visual features
3. **Concept Modulation**: Text/exemplar embeddings modulate attention
4. **Output Projection**: Queries decode into boxes and masks

## Decoupled Recognition and Localization

### Why Decoupling Matters

Traditional detectors conflate recognition and localization in single queries:
- Query must both identify "is this the target?" AND locate it
- Conflicting objectives during training
- Harder optimization landscape

SAM 3's approach:
- **Presence Head**: Handles recognition globally ("is concept in image?")
- **Object Queries**: Focus only on localization ("where are instances?")
- **Result**: Better performance on both tasks

### Implementation

```
Concept-Conditioned Features
        ↓
   [Presence Token] → Binary: Concept present? (global decision)
        ↓
   [Object Queries] → If present: Locate all instances
        ↓
   Boxes + Masks + Scores
```

## Performance Characteristics

### Inference Speed

From [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):
> "SAM 3 runs at ~30 ms per image on an H200 GPU, handling 100+ objects"

### Model Size

- **Total Parameters**: 848M
- **Encoder**: Shared Perception Encoder (largest component)
- **Detector**: DETR decoder + heads
- **Tracker**: SAM 2-inherited components

### Benchmark Results

From [Ultralytics Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

| Benchmark | Metric | SAM 3 | Previous Best | Improvement |
|-----------|--------|-------|---------------|-------------|
| LVIS (zero-shot) | Mask AP | **47.0** | 38.5 | +22.1% |
| SA-Co/Gold | CGF1 | **65.0** | 34.3 (OWLv2) | +89.5% |
| COCO (zero-shot) | Box AP | **53.5** | 52.2 (T-Rex2) | +2.5% |

## Training Details

### Training Stages

From [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):
> "Training occurs in four stages: (1) Perception Encoder pre-training, (2) detector pre-training on synthetic and HQ data, (3) detector fine-tuning on SA-Co HQ, and (4) tracker training with a frozen backbone"

1. **Stage 1**: Perception Encoder pre-training
2. **Stage 2**: Detector pre-training on synthetic + HQ data
3. **Stage 3**: Detector fine-tuning on SA-Co high-quality data
4. **Stage 4**: Tracker training (frozen backbone)

### Hard Negatives Importance

From [Ultralytics Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

Hard negatives are crucial for open-vocabulary recognition:

| Hard Negatives/Image | CGF1 | IL_MCC | pmF1 |
|---------------------|------|--------|------|
| 0 | 31.8 | 0.44 | 70.2 |
| 5 | 44.8 | 0.62 | 71.9 |
| **30** | **49.2** | **0.68** | **72.3** |

IL_MCC improved by **54.5%** with hard negatives.

## Comparison with Prior Work

### vs OWLv2 / DINO-X

- **SAM 3**: Integrated detector-segmenter with presence head
- **OWLv2**: Open-vocabulary detector without segmentation
- **DINO-X**: Strong detection but lower on SA-Co benchmark

SAM 3 outperforms by +89.5% on SA-Co/Gold CGF1 metric.

### vs SAM 2

| Feature | SAM 2 | SAM 3 |
|---------|-------|-------|
| Text Prompts | No | Yes |
| Open-Vocabulary | No (visual prompts only) | Yes (270K concepts) |
| Detection | No (requires external) | Yes (built-in DETR) |
| All Instances | No (single per prompt) | Yes (all matching) |

## Sources

**Primary Sources:**
- [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository (accessed 2025-11-23)
- [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) - Technical overview (accessed 2025-11-23)
- [Ultralytics YOLO Docs - SAM 3](https://docs.ultralytics.com/models/sam-3/) - Architecture documentation (accessed 2025-11-23)

**Paper Reference:**
- SAM 3: Segment Anything with Concepts - Submitted to ICLR 2026 (Paper ID: 4183)
- [OpenReview](https://openreview.net/forum?id=r35clVtGzw)

**Related Resources:**
- [Meta AI Blog - SAM 3](https://ai.meta.com/blog/segment-anything-model-3/)
- [Meta AI Research - SAM 3 Publication](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Segment Anything Demo](https://segment-anything.com/)
