# Text Prompts and Visual Exemplars in SAM

## Overview

Text prompts and visual exemplars represent a paradigm shift in image segmentation, enabling **open-vocabulary concept detection** without predefined label sets. SAM 3 introduces this capability, accepting natural language noun phrases or image-based examples to segment all matching instances.

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md):
- SAM 3 released November 19, 2025
- Detects, segments, and tracks objects via text prompts
- Supports 270K unique concepts (50x more than existing benchmarks)
- Achieves 75-80% of human performance on SA-Co benchmark

### Key Innovations

**Promptable Concept Segmentation (PCS)**:
- New task type introduced with SAM 3
- Find ALL instances of a concept in image/video
- Not limited to fixed label vocabularies
- Natural language interface eliminates training requirements

**Text Prompt Examples**:
```python
# Natural language prompts
"Find all red apples"
"Segment the person wearing a hat"
"Detect all stop signs"
"shipping container"
"yellow school bus"
```

**Visual Exemplar Examples**:
```python
# Use visual example instead of text
output = processor.set_visual_prompt(
    state=inference_state,
    exemplar_box=[100, 100, 200, 200]  # BBox of example object
)
# Finds all visually similar objects
```

### Comparison: SAM 1/2 vs SAM 3

| Feature | SAM 1/2 | SAM 3 |
|---------|---------|-------|
| **Prompt Types** | Points, boxes, masks | + Text, visual exemplars |
| **Output** | Single object per prompt | ALL matching instances |
| **Vocabulary** | N/A (geometric) | 270K concepts |
| **Use Case** | Interactive annotation | Concept-level detection |

### Performance Characteristics

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-20):
- ~30 ms per image on H200 GPU
- Handles 100+ objects per image
- 848M total parameters (~3.4 GB)
- Server-scale model (not edge-deployable)

---

## Text Encoding

Text prompts are encoded through a sophisticated text encoder that aligns language with visual features, enabling semantic understanding of natural language descriptions.

### CLIP Integration

**Architecture Overview**:
```
Text Prompt ("red apple")
    ↓
Text Encoder (CLIP-based)
    ↓
512-dim text embedding
    ↓
Linear Projection
    ↓
Prompt embedding dimension
    ↓
Cross-modal fusion with image features
```

**Text Encoder Details**:
- Based on CLIP text encoder architecture
- Produces 512-dimensional text features
- Projected to match prompt embedding dimension
- Enables semantic similarity matching

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 600-615:
```python
# Text prompt encoding (CLIP-based)
def encode_text(text_prompt):
    text_features = clip_encoder(text_prompt)  # 512-dim
    prompt_embedding = linear_projection(text_features)
    return prompt_embedding
```

### Presence Token Innovation

SAM 3 introduces a novel **presence token** that improves discrimination between similar concepts:

```python
class SAM3Detector(nn.Module):
    def __init__(self):
        self.vision_encoder = HieraEncoder()
        self.text_encoder = CLIPTextEncoder()
        self.presence_token = nn.Embedding(1, 512)  # Novel!
        self.detection_head = DetectionHead()

    def forward(self, image, text_prompt):
        # Encode image
        image_features = self.vision_encoder(image)

        # Encode text
        text_features = self.text_encoder(text_prompt)

        # Add presence token for better discrimination
        text_features = text_features + self.presence_token.weight

        # Cross-modal fusion
        fused = cross_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )

        # Detect all instances
        detections = self.detection_head(fused)
        return detections
```

**Why Presence Token Matters**:
- Improves discrimination: "red apple" vs "green apple"
- Better handling of fine-grained distinctions
- Reduces false positives on similar concepts

### Open-Vocabulary Capability

**270K Concept Vocabulary**:
- 50x larger than LVIS benchmark
- Derived from web-scale image-text pairs
- Covers common objects to long-tail concepts
- 22 million entities across 17 top-level categories

**Vocabulary Categories**:
- Objects (vehicles, furniture, electronics)
- Animals (species, breeds)
- Food (ingredients, dishes)
- Clothing (garments, accessories)
- Natural elements (plants, geological features)
- And many more specialized domains

### Text Encoding Implementation

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("street_scene.jpg")
state = processor.set_image(image)

# Text prompt - finds ALL matching instances
output = processor.set_text_prompt(
    state=state,
    prompt="car"  # Natural language noun phrase
)

masks = output["masks"]    # All masks matching "car"
boxes = output["boxes"]    # Bounding boxes
scores = output["scores"]  # Confidence scores

print(f"Found {len(masks)} cars")
```

---

## Visual Exemplars

Visual exemplars provide an alternative to text prompts, allowing users to specify targets by pointing to example objects in images rather than describing them with words.

### Geometry and Exemplar Encoder

When prompts include visual examples (cropped objects, points, or boxes), SAM 3 uses a **geometry and exemplar encoder** to convert them into embeddings:

```python
# Visual exemplar prompting
output = processor.set_visual_prompt(
    state=inference_state,
    exemplar_box=[100, 100, 200, 200]  # Bounding box of example
)

# Finds all visually similar objects
similar_masks = output["masks"]
```

**Encoding Process**:
```
Visual Exemplar (cropped region)
    ↓
Geometry Encoder (spatial features)
    ↓
Exemplar Encoder (appearance features)
    ↓
Combined embedding
    ↓
Similarity matching against image features
```

### Use Cases for Visual Exemplars

**When to Use Visual Exemplars**:
- Concept is hard to describe in words
- Need visual similarity matching
- Domain-specific objects without common names
- Want to find "objects like this one"

**Example Applications**:
```python
# Find all objects visually similar to selection
# Useful when:
# - You don't know the exact term
# - Multiple terms could apply
# - Visual appearance matters more than category
```

### Combining Text and Exemplars

SAM 3 supports hybrid prompting:
- Start with text prompt
- Refine with positive/negative clicks
- Use exemplars for similar object discovery

**Interactive Hybrid Mode**:
```python
# Text prompt to find category
output = processor.set_text_prompt(state, "car")

# Then refine with visual clicks
# Positive clicks to add regions
# Negative clicks to remove regions
```

---

## Multi-Modal Fusion

Multi-modal fusion combines visual features from images with text/exemplar embeddings to create a unified representation for concept detection.

### Perception Encoder Architecture

SAM 3 uses a unified **Perception Encoder (PE)** that aligns visual and language features:

```
Image                    Text/Exemplar Prompt
   ↓                            ↓
Vision Encoder              Prompt Encoder
(Hiera-based)              (Text/Exemplar)
   ↓                            ↓
Visual Features          Prompt Embeddings
   ↓                            ↓
       Perception Encoder (PE)
              ↓
    Joint Embedding Space
              ↓
   Cross-Modal Attention
              ↓
      Fused Features
              ↓
    Detection/Tracking
```

### Cross-Modal Attention Mechanism

```python
def cross_modal_fusion(image_features, text_features):
    """
    Fuse visual and text features through attention
    """
    # Query: image features seek relevant text
    # Key/Value: text features provide semantic context
    fused = cross_attention(
        query=image_features,
        key=text_features,
        value=text_features
    )
    return fused
```

### Detector-Tracker Coupling

SAM 3's architecture tightly couples detector and tracker:

**Shared Components**:
- Both use same Perception Encoder
- Shared visual feature extraction
- Consistent embedding space

**Decoupled Functions**:
- **Detector**: Finds all instances in single frame
- **Tracker**: Propagates masklets across video frames

### Global Presence Head

A key architectural innovation for open-vocabulary detection:

```python
class PresenceHead(nn.Module):
    """
    Determines IF concept exists before WHERE
    """
    def forward(self, fused_features):
        # First: Does this concept exist in image?
        presence_score = self.presence_classifier(fused_features)

        # Only if present, localize instances
        if presence_score > threshold:
            instances = self.localization_head(fused_features)
            return instances
        return []
```

**Benefits**:
- Separates recognition (what) from localization (where)
- Improves accuracy on unseen concepts
- Better handling of hard negatives
- Reduces false positives

---

## Zero-Shot Text Segmentation

Zero-shot text segmentation enables segmenting objects from novel categories without any task-specific training, using only natural language descriptions.

### Zero-Shot Capabilities

**What Zero-Shot Means**:
- No training on specific target categories
- Generalizes to unseen concepts
- Uses language understanding from pre-training
- Works immediately with any noun phrase

**Performance on Benchmarks**:
- SA-Co/Gold: 54.1 cgF1 (human baseline: 72.8)
- LVIS: 37.2 cgF1 (state-of-the-art open-vocabulary)
- Achieves 75-80% of human performance

### Open-Vocabulary Instance Detection

Unlike traditional instance segmentation:
- No fixed label set
- Natural language interface
- All matching instances found
- Unique masks and IDs per object

```python
# Traditional: Limited to trained classes
# SAM 3: Any noun phrase works
output = processor.set_text_prompt(state, "ergonomic office chair")
# Works even though "ergonomic office chair" wasn't in training
```

### Lazy Visual Grounding

From research on zero-shot open-vocabulary segmentation:
- First discovers distinguishable visual features
- Then grounds them to language
- Late interaction manner improves accuracy
- Better generalization to unseen concepts

### Limitations of Zero-Shot

**Current Challenges**:
- Performance gap vs human (75-80%)
- Long-tail concepts less accurate
- Ambiguous prompts cause confusion
- Requires clear, specific noun phrases

---

## Use Cases

### Dataset Labeling and Annotation

**High-Throughput Labeling**:
```python
# Label entire dataset with text prompts
for image in dataset:
    state = processor.set_image(image)

    # Multi-class labeling
    for class_name in ["car", "person", "bicycle"]:
        output = processor.set_text_prompt(state, class_name)
        save_annotations(output["masks"], class_name)
```

**Benefits**:
- Eliminates manual clicking per object
- Consistent labeling across images
- Can label 100+ objects per image
- Reduces annotation time significantly

### Training Smaller Supervised Models

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-20):

SAM 3 is computationally expensive (~3.4 GB, server-scale). Use SAM 3 to generate training data for smaller, edge-deployable models:

```python
# 1. Use SAM 3 to label data
sam3_labels = sam3.segment_all_cars(images)

# 2. Train smaller model (RF-DETR, YOLOv8)
smaller_model.train(images, sam3_labels)

# 3. Deploy smaller model at edge
# - Faster inference
# - Lower memory
# - Specialized for your task
```

### Video Object Tracking

**Text-Prompted Video Tracking**:
```python
# Start session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="video.mp4"
    )
)

# Add text prompt for frame 0
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="person wearing red shirt"
    )
)

# Automatically tracks through all frames
```

### Domain-Specific Applications

**Medical Imaging**:
- "Tumor" / "lesion" segmentation
- Organ delineation
- Cell counting

**Autonomous Driving**:
- "Pedestrian" / "vehicle" detection
- Road sign identification
- Lane detection

**Agriculture**:
- "Tomato plant" detection
- Crop health monitoring
- Yield estimation

**Satellite/Aerial**:
- "Building" detection
- "Solar panel" identification
- Urban density analysis

**E-Commerce**:
- Product segmentation
- Background removal
- Virtual try-on support

### Interactive Refinement

**Combining Text with Visual Feedback**:
```python
# Initial text detection
output = processor.set_text_prompt(state, "car")

# User provides feedback
# Positive clicks to add missed objects
# Negative clicks to remove false positives

# SAM 3 refines without restarting inference
refined_output = processor.refine_with_clicks(
    positive_clicks=[[x1, y1], [x2, y2]],
    negative_clicks=[[x3, y3]]
)
```

---

## ARR-COC Integration

### Text-Prompted Segmentation for Training Data

SAM 3's text prompting enables efficient generation of segmentation training data for ARR-COC fine-tuning:

```python
# Generate masks for specific concepts
output = processor.set_text_prompt(state, "attention_region")

# Use masks as training targets
# Train transformer attention to focus on these regions
```

### Integration Architecture

**SAM 3 in Training Pipeline**:
```
Raw Images
    ↓
SAM 3 Text Prompts ("focus_region", "context_background")
    ↓
High-Quality Segmentation Masks
    ↓
Attention Supervision Targets
    ↓
ARR-COC Training
```

### Concept-Guided Attention

Use SAM 3's concept understanding to guide attention mechanisms:
- Segment regions by semantic concept
- Weight attention based on concept relevance
- Improve feature extraction for specific tasks

### Implementation Considerations

**Computational Requirements**:
- SAM 3 requires ~3.4 GB GPU memory
- ~30 ms per image inference
- Best for batch preprocessing, not online inference

**Workflow Integration**:
```python
# Preprocessing phase
def generate_attention_targets(images, concepts):
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    targets = []
    for image in images:
        state = processor.set_image(image)

        # Multi-concept attention regions
        masks = {}
        for concept in concepts:
            output = processor.set_text_prompt(state, concept)
            masks[concept] = output["masks"]

        targets.append(masks)

    return targets

# Training phase
attention_targets = generate_attention_targets(
    images,
    ["primary_object", "context", "background"]
)
```

### Fine-Tuning for Domain Tasks

SAM 3 can be fine-tuned for ARR-COC specific concepts:

```python
# Fine-tune SAM 3 on domain-specific data
# - Custom concept vocabulary
# - Domain-specific visual features
# - Improved accuracy on target task
```

**Benefits for ARR-COC**:
- Semantic understanding of attention targets
- Automated mask generation
- Consistent annotation quality
- Reduced manual labeling effort

---

## Sources

**Source Documents**:
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Complete SAM research study

**Web Research**:
- [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) - SAM 3 overview and capabilities (accessed 2025-11-20)
- [Meta AI SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/) - Official announcement
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/) - Technical documentation
- [OpenReview SAM 3 Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) - Research paper

**GitHub Repositories**:
- [SAM 3 GitHub](https://github.com/facebookresearch/sam3) - Official implementation
- [Awesome Multimodal Referring Segmentation](https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation) - Related research

**Related Research**:
- Promptable Concept Segmentation (PCS) task definition
- Open-vocabulary instance detection
- Vision-language model fusion
- Zero-shot visual grounding

**Datasets**:
- SA-Co (Segment Anything with Concepts) - 270K concepts, 5.2M images, 52.5K videos
- SA-Co/Gold - High-quality evaluation set
- SA-Co/Silver - Larger training set
- SA-Co/VEval - Video evaluation set
