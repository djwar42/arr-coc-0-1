# SAM 3: Geometry and Exemplar Prompts

## Overview

SAM 3 introduces a unified prompting system that supports multiple modalities beyond text. Unlike SAM 2 which was primarily geometry-focused (points, boxes, masks), SAM 3 combines traditional visual prompts with concept-based prompts (text and image exemplars) in a single model. This creates a versatile system where users can prompt using whatever method best suits their task.

## Prompt Modalities in SAM 3

### 1. Visual/Geometry Prompts (Inherited from SAM 2)

SAM 3 maintains full backward compatibility with SAM 2's visual prompting system:

**Points:**
- Positive points (label=1): Indicate object to segment
- Negative points (label=0): Indicate regions to exclude
- Multiple points can be combined for refinement

**Boxes:**
- Bounding boxes around target objects
- Define region of interest for segmentation
- Can be used alone or combined with points

**Masks:**
- Initial mask input for refinement
- Useful for iterative improvement
- Supports coarse-to-fine segmentation

### 2. Concept Prompts (NEW in SAM 3)

**Text Prompts:**
- Simple noun phrases like "yellow school bus" or "striped cat"
- Descriptive phrases like "person wearing a red hat"
- Open-vocabulary - no fixed label set

**Image Exemplars:**
- Bounding boxes around example objects in reference images
- Positive exemplars (label=1): "Find objects like this"
- Negative exemplars (label=0): "Don't include objects like this"
- Model generalizes from examples to find all similar objects

### 3. Combined Prompts

SAM 3's most powerful capability is combining prompt types:

**Text + Exemplars:**
- Use text for concept specification
- Add exemplars for precision refinement
- Example: "dog" + positive box around specific dog type

**Exemplars + Points:**
- Use exemplar for concept matching
- Add points for spatial refinement
- Useful when exemplar finds correct objects but needs boundary adjustment

**All Three Combined:**
- Text for broad concept
- Exemplars for appearance matching
- Points for fine-grained boundary control

## Technical Implementation

### Prompt Encoding Architecture

From [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):

The detector component includes:
- **Text encoder**: Processes noun phrase prompts into embeddings
- **Exemplar encoder**: Processes image-based exemplar prompts
- **Fusion encoder**: Conditions image features on combined prompts
- **Presence head**: Decouples recognition ("what") from localization ("where")

Each prompt type is encoded into a shared embedding space, allowing seamless combination.

### Interactive Refinement System

From [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-23):

SAM 3's exemplar-based refinement converges much faster than pure visual prompting:

| Prompts Added | CGF1 Score | Gain vs Text-Only | Gain vs PVS Baseline |
|---------------|------------|-------------------|----------------------|
| Text only | 46.4 | baseline | baseline |
| +1 exemplar | 57.6 | +11.2 | +6.7 |
| +2 exemplars | 62.2 | +15.8 | +9.7 |
| +3 exemplars | 65.0 | +18.6 | +11.2 |
| +4 exemplars | 65.7 | +19.3 | +11.5 (plateau) |

Key insight: Adding just 3 exemplars provides an **18.6 point CGF1 improvement** over text-only prompting.

### Exemplar Prompting Mechanism

From [Roboflow - What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

The exemplar system works by:
1. User provides bounding box around example object(s)
2. Exemplar encoder extracts visual features
3. Features are fused with any text prompt
4. Detector finds all objects matching the combined representation
5. Unlike point prompts, exemplars **generalize** to similar objects, not just correct individual instances

## Usage Patterns

### Pattern 1: Pure Visual Prompting (SAM 2 Style)

```python
from ultralytics import SAM

model = SAM("sam3.pt")

# Single point prompt
results = model(points=[900, 370], labels=[1])

# Multiple points (positive and negative)
results = model(points=[[400, 370], [900, 370]], labels=[1, 0])

# Box prompt
results = model(bboxes=[100, 150, 300, 400])
```

**Use case:** When you can precisely indicate the specific object you want to segment.

### Pattern 2: Exemplar-Based Prompting

```python
# Provide positive example - finds all similar objects
results = model("image.jpg", bboxes=[100, 150, 300, 400], labels=[1])

# Add negative examples to exclude certain instances
results = model(
    "image.jpg",
    bboxes=[[100, 150, 300, 400], [500, 200, 600, 350]],
    labels=[1, 0],  # First positive, second negative
)
```

**Use case:** When you have an example of what you want but need to find all similar objects.

### Pattern 3: Combined Text + Exemplar

```python
# Text concept with exemplar refinement
results = model(
    "image.jpg",
    prompt="dog",
    bboxes=[100, 150, 300, 400],
    labels=[1]
)
```

**Use case:** When text alone is ambiguous but you have examples to clarify intent.

### Pattern 4: Interactive Refinement Workflow

```python
# Step 1: Initial text-based segmentation
results = model("image.jpg", prompt="car")

# Step 2: Add positive exemplar for missed instances
results = model(
    "image.jpg",
    prompt="car",
    bboxes=[missed_car_box],
    labels=[1],
)

# Step 3: Add negative exemplar for false positives
results = model(
    "image.jpg",
    prompt="car",
    bboxes=[false_positive_box],
    labels=[0],
)
```

**Use case:** Production annotation workflows requiring high precision.

## Key Differences from SAM 2

### Prompting Paradigm Shift

| Feature | SAM 2 | SAM 3 |
|---------|-------|-------|
| **Primary task** | Single object per prompt | All instances of a concept |
| **Prompt types** | Points, boxes, masks | + Text phrases, image exemplars |
| **Detection** | Requires external detector | Built-in open-vocabulary detector |
| **Recognition** | Geometry-based only | Text and visual recognition |
| **Refinement** | Clicks only | Clicks + exemplar generalization |

### Exemplar vs Point Behavior

**SAM 2 Points:**
- Indicate specific pixels to include/exclude
- Correct individual mask boundaries
- Local, instance-specific effect

**SAM 3 Exemplars:**
- Indicate example objects
- Find all similar objects in image
- Global, concept-level effect
- Model generalizes from example to class

## Video Prompting

SAM 3 extends all prompt types to video:

```python
from ultralytics.models.sam import SAM3VideoPredictor

predictor = SAM3VideoPredictor(model="sam3.pt")

# Text prompt for video tracking
results = predictor(source="video.mp4", prompt="person wearing blue shirt")

# Combined text + exemplar for precision
results = predictor(
    source="video.mp4",
    prompt="kangaroo",
    bboxes=[initial_box],  # Exemplar from first frame
    labels=[1],
)
```

The tracker inherits SAM 2's memory-based architecture:
- Memory bank stores object appearance across frames
- Temporal disambiguation handles occlusions
- Periodic re-prompting prevents drift

## Best Practices for Combined Prompting

### When to Use Text vs Exemplars

**Use Text When:**
- Object concept is clearly describable ("red apple", "traffic sign")
- You want to find all instances without bias
- Starting exploration of image content

**Use Exemplars When:**
- Text alone is ambiguous ("that specific type of chair")
- You have clear positive/negative examples
- Refining initial text-based results
- Appearance matters more than category

**Combine When:**
- Maximum precision required
- Category + specific appearance both matter
- Interactive annotation workflows

### Refinement Strategy

1. **Start broad**: Use text prompt for initial segmentation
2. **Identify errors**: Note false positives and false negatives
3. **Add positive exemplars**: For missed objects
4. **Add negative exemplars**: For false positives
5. **Use points**: For final boundary refinement

From research: This workflow typically reaches plateau accuracy after 3-4 exemplars.

### Prompt Design Guidelines

**Text Prompts:**
- Keep noun phrases simple and specific
- Include distinguishing attributes ("striped cat" vs "cat")
- Avoid complex compositional language

**Exemplar Selection:**
- Choose representative examples
- Include variety if objects vary in appearance
- Negative exemplars should be confusable cases

## Performance Characteristics

### Computational Considerations

From [Roboflow - What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

- Inference: ~30 ms per image on H200 GPU
- Handles 100+ objects per image
- Model size: ~848M parameters (~3.4 GB)
- Server-scale model, not edge deployment

### Prompt Modality Speed

Different prompt types have similar computational cost since all are encoded into the same embedding space before detection. The fusion encoder processes combined prompts efficiently.

## Applications by Prompt Strategy

### Text Prompts Ideal For:
- Content moderation (find all instances of specific content)
- E-commerce cataloging (segment all products of type X)
- Dataset annotation (label all occurrences of category)

### Exemplar Prompts Ideal For:
- Few-shot adaptation to new domains
- Finding specific object variants
- When appearance matters more than category label

### Combined Prompts Ideal For:
- High-precision annotation pipelines
- Medical imaging (specific tissue types)
- Quality control (specific defect types)

## Comparison with Other Systems

### vs. Traditional Object Detection (YOLO, Faster R-CNN)
- SAM 3: Open vocabulary, any concept via prompts
- Traditional: Fixed categories from training

### vs. Open-Vocabulary Detectors (OWLv2, DINO-X)
- SAM 3: Adds exemplar prompting and instance segmentation
- Others: Primarily text-only, box detection focus

### vs. Referring Segmentation
- SAM 3: Noun phrases, all instances
- Referring: Complex expressions, single instance

## Sources

**Primary Sources:**
- [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository (accessed 2025-11-23)
- [Roboflow Blog - What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) - Comprehensive overview (accessed 2025-11-23)
- [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) - Detailed technical docs (accessed 2025-11-23)

**Additional References:**
- [HuggingFace - facebook/sam3](https://huggingface.co/facebook/sam3) - Model repository
- SAM 3: Segment Anything with Concepts - Research paper (ICLR 2026 submission)

## Summary

SAM 3's prompting system represents a significant evolution from SAM 2:

1. **Unified prompting**: Text, exemplars, and geometry in single model
2. **Exemplar generalization**: Examples find all similar objects, not just correct single instances
3. **Combined strategies**: Mixing prompt types for precision
4. **Interactive refinement**: 3 exemplars typically sufficient for plateau accuracy
5. **Backward compatibility**: All SAM 2 visual prompts still work

The ability to combine prompt modalities - using text for broad concept specification, exemplars for appearance matching, and points for boundary refinement - makes SAM 3 the most flexible segmentation foundation model to date.
