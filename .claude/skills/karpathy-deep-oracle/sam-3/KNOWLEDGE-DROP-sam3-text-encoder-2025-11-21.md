# SAM 3 Text Encoder Integration

## Overview

SAM 3 introduces text-based prompting to the Segment Anything family through a dedicated text encoder integrated into its DETR-based detector architecture. This enables open-vocabulary Promptable Concept Segmentation (PCS) where users can specify concepts using natural language phrases like "yellow school bus" or "person wearing a red hat".

## Text Encoder Architecture

### Integration Within Detector

From [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) and [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

The text encoder is a core component of SAM 3's **detector** module (not the tracker). The detector architecture consists of:

1. **Text Encoder** - Processes noun phrase prompts into embeddings
2. **Exemplar Encoder** - Processes image-based prompts (boxes around example objects)
3. **Fusion Encoder** - Conditions image features on the prompt embeddings
4. **Presence Head** - Novel component that decouples recognition from localization
5. **Mask Head** - Generates instance segmentation masks

### Text Encoding Process

**Input Format:**
- Simple noun phrases (e.g., "red apple", "striped cat")
- Descriptive phrases with attributes (e.g., "person wearing a hat")
- Short text prompts focused on atomic visual concepts

**Processing Flow:**
```
Text Prompt → Text Encoder → Text Embeddings → Fusion Encoder → Conditioned Features
```

The text embeddings are used to condition the detector's features, enabling the model to search for all instances of the specified concept in the image or video.

## Text Embedding Generation

### Embedding Characteristics

Based on the architecture described in the research:

- Text embeddings represent semantic concepts that can be matched against visual features
- The system handles **270K unique concepts** in the SA-Co benchmark
- Supports **4M unique noun phrases** in training data (SA-Co/HQ)
- Embeddings must enable discrimination between closely related concepts

### Concept Vocabulary Scale

From [HuggingFace - facebook/sam3](https://huggingface.co/facebook/sam3) (accessed 2025-11-23):

SAM 3's text encoder was trained on an unprecedented scale:
- **4M unique noun phrases** in SA-Co/HQ (high-quality human annotations)
- **38M noun phrases** in SA-Co/SYN (synthetic dataset)
- **270K unique concepts** in evaluation benchmark
- **50x more concepts** than existing benchmarks like LVIS (~4K concepts)

## Alignment with Vision Features

### Fusion Mechanism

The text encoder output is aligned with vision features through the **Fusion Encoder**:

1. **Vision Encoder** (shared between detector and tracker) extracts image/video features
2. **Text Encoder** generates text embeddings for the concept prompt
3. **Fusion Encoder** combines these to produce concept-conditioned visual features
4. These conditioned features feed into detection queries

### Presence Token Innovation

A key architectural innovation for text-vision alignment is the **Presence Token**:

From [Ultralytics Documentation](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

> "A learned global token that predicts whether the target concept is present in the image/frame, improving detection by separating recognition from localization."

**Why This Matters for Text Encoding:**
- Traditional approaches conflate "is this concept here?" with "where is it?"
- The presence token handles recognition (text-image matching) separately
- Proposal queries focus only on localization
- This decoupling improves discrimination between similar text prompts

**Performance Impact:**
- Without presence head: 57.6 CGF1, 0.77 IL_MCC
- With presence head: **63.3 CGF1**, **0.82 IL_MCC**
- **+9.9% improvement** primarily in recognition ability

## Text Prompt Discrimination

### Handling Similar Concepts

SAM 3's text encoder excels at discriminating closely related prompts:

From [GitHub README](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):

> "SAM 3 introduces a new model architecture featuring a presence token that improves discrimination between closely related text prompts (e.g., 'a player in white' vs. 'a player in red')"

### Hard Negatives Training

The text encoder's discrimination ability comes from training with hard negatives:

| Hard Negatives/Image | CGF1 | IL_MCC | pmF1 |
|---------------------|------|--------|------|
| 0 | 31.8 | 0.44 | 70.2 |
| 5 | 44.8 | 0.62 | 71.9 |
| **30** | **49.2** | **0.68** | **72.3** |

Hard negatives improve IL_MCC (classification accuracy) by **54.5%**, indicating the text encoder learns fine-grained concept distinctions.

## Combined Prompting: Text + Exemplars

### Multimodal Concept Specification

SAM 3 supports combining text with visual exemplars:

```python
# Text + positive image exemplar
results = model(
    "path/to/image.jpg",
    prompt="dog",
    bboxes=[100, 150, 300, 400],
    labels=[1]  # Positive exemplar
)

# Text + negative exemplar to exclude certain instances
results = model(
    "path/to/image.jpg",
    prompt="handle",
    bboxes=[oven_handle_box],
    labels=[0]  # Negative - exclude oven handle
)
```

This allows:
- Text to specify the general concept
- Positive exemplars to refine what "counts" as that concept
- Negative exemplars to exclude unwanted instances

## API Usage

### Basic Text Prompting

From [HuggingFace Model Card](https://huggingface.co/facebook/sam3) (accessed 2025-11-23):

```python
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image = Image.open("image.jpg").convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
```

### Native SAM3 API

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("image.jpg")
inference_state = processor.set_image(image)

# Text prompt
output = processor.set_text_prompt(
    state=inference_state,
    prompt="yellow school bus"
)

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

### Video Text Prompting

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="video.mp4",
    )
)

response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,
        text="person wearing blue shirt",  # Text prompt for video
    )
)
```

## Technical Specifications

### Model Parameters

From [MarkTechPost](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) (accessed 2025-11-23):

- **Total model size**: 848M parameters
- Includes detector (with text encoder) and tracker sharing vision encoder

### Supported Prompt Types

| Prompt Type | Description | Use Case |
|------------|-------------|----------|
| Text (noun phrases) | "red apple", "striped cat" | Find all instances of a concept |
| Combined text + exemplar | Text + positive/negative boxes | Refine concept definition |
| Batched text | Different text per image in batch | Efficient multi-image processing |

### Performance Benchmarks

**Zero-Shot Text-Based Segmentation:**
- LVIS Mask AP: **47.0** (vs 38.5 previous best, +22% improvement)
- SA-Co/Gold CGF1: **65.0** (vs 34.3 OWLv2, +89.5% improvement)

**Inference Speed:**
- 30ms per image with 100+ detected objects (H200 GPU)

## Comparison: Text Encoder vs CLIP

### Key Differences from CLIP-Style Approaches

While SAM 3 uses text encoding for vision-language alignment, it differs from standard CLIP approaches:

1. **Task-Specific Design**: Optimized for segmentation, not just classification
2. **Instance-Level Output**: Returns all matching instances, not just image-level scores
3. **Presence Token**: Explicit recognition/localization decoupling
4. **Exemplar Fusion**: Can combine text with visual examples
5. **Scale**: Trained on 4M+ unique noun phrases for segmentation

### Training Data Advantage

The text encoder benefits from SAM 3's massive data engine:
- **SA-Co/HQ**: 5.2M images, 4M unique noun phrases (human-annotated)
- **SA-Co/SYN**: 1.4B masks, 38M noun phrases (synthetic)
- Hard negatives mined to improve discrimination

## Limitations

### Text Prompt Constraints

From [Ultralytics Documentation](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

1. **Simple Phrases Only**: Best suited for simple noun phrases
2. **No Complex Reasoning**: Long referring expressions require MLLM integration
3. **Ambiguity Challenges**: Some concepts remain inherently ambiguous (e.g., "small window")
4. **Vocabulary Scope**: Focused on atomic visual concepts

### Recommended Query Complexity

**Native SAM 3 (works well):**
- "yellow school bus"
- "striped cat"
- "person wearing red hat"

**Requires MLLM (SAM 3 Agent):**
- "People sitting down but not holding a gift box"
- "The dog closest to the camera without a collar"
- "Red objects larger than the person's hand"

## Sources

**Primary Sources:**
- [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository (accessed 2025-11-23)
- [HuggingFace - facebook/sam3](https://huggingface.co/facebook/sam3) - Model card and API documentation (accessed 2025-11-23)
- [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) - Comprehensive technical documentation (accessed 2025-11-23)

**Additional References:**
- [MarkTechPost SAM 3 Article](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) - Model parameter details (accessed 2025-11-23)
- Meta AI Blog (requires login) - Official announcement

## Key Takeaways

1. **Text encoder is part of detector module**, not a standalone component
2. **Presence token is the key innovation** for text-vision alignment, separating recognition from localization
3. **Trained on 4M+ unique noun phrases** - largest open-vocabulary segmentation training set
4. **+9.9% improvement** from presence token, +54.5% from hard negatives training
5. **Supports combined prompting** - text + positive/negative exemplars for refinement
6. **Simple noun phrases only** - complex reasoning requires MLLM integration
