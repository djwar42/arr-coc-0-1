# SAM 3 Text Prompt Examples

## Overview

SAM 3 introduces text prompting as a primary input modality for segmentation, enabling users to describe objects using natural language noun phrases instead of clicking or drawing boxes. This represents a fundamental shift from visual-only prompts in SAM 1 and SAM 2 to open-vocabulary concept segmentation.

From [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3):
- SAM 3 accepts "text prompts, open-vocabulary short noun phrases, and image exemplar prompts"
- Supports 270K unique concepts (50x more than existing benchmarks)
- Achieves 75-80% of human performance on SA-Co benchmark

---

## The Canonical Example: "Player in White" vs "Player in Red"

From [SAM 3 GitHub README](https://github.com/facebookresearch/sam3/blob/main/README.md):

> SAM 3 introduces a new model architecture featuring a **presence token** that improves discrimination between closely related text prompts (e.g., "a player in white" vs. "a player in red")

This example is central to understanding SAM 3's advancement:

**Why This Is Challenging:**
- Both prompts describe visually similar objects (players)
- The only distinguishing feature is clothing color
- Previous models would struggle to separate these semantically related concepts
- Requires both visual recognition AND text understanding

**How SAM 3 Solves It:**
The presence token mechanism:
1. First determines IF the target concept exists in the scene
2. Then localizes instances that match
3. Separates recognition (what) from localization (where)
4. Reduces confusion between closely related prompts

From [MarkTechPost Analysis](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/):
> "The presence token predicts whether each candidate box or mask actually corresponds to the requested concept. It is especially important when the text prompts describe related entities."

---

## Categories of Text Prompts

### 1. Simple Object Prompts

Basic noun phrases identifying common objects:

**Examples from Documentation:**
- `"solar panels"` - Identifies all solar panels in an image
- `"vehicle"` - Finds all cars, motorbikes, trucks
- `"box"` - Detects dozens of boxes in warehouse imagery
- `"striped cat"` - Finds cats with specific visual attributes

From [Roboflow SAM 3 Guide](https://blog.roboflow.com/what-is-sam3/):
- `"shipping container"` - Generates precise masks for all containers
- `"trailers"` - Identifies trailers in aerial imagery
- `"stall numbers"` - Detects text/numbers in intermodal yards

### 2. Descriptive/Attributed Prompts

Prompts that include visual attributes:

**Examples:**
- `"yellow school bus"` - Color + object type
- `"player in red"` - Object + clothing attribute
- `"player in white"` - Distinguishing by color attribute
- `"red baseball cap"` - Color + specific object type
- `"striped cat"` - Pattern + animal

**Key Insight:** SAM 3 can parse these compound descriptions and match both the object category AND the specified attributes.

### 3. Domain-Specific Prompts

From [Roboflow Examples](https://blog.roboflow.com/what-is-sam3/):

**Aerial/Satellite Imagery:**
- `"solar panels"` - Clean identification avoiding occluded areas
- `"trailers"` - In intermodal yard imagery
- `"stall numbers"` - Text detection in aerial views

**Industrial/Warehouse:**
- `"box"` - Large-scale box detection in warehouse scenes
- `"vehicle"` - Multi-type vehicle detection on streets

**Sports/Video:**
- `"player in white"` vs `"player in red"` - Team differentiation
- Track specific players through video frames

### 4. Complex/Compound Prompts (SAM 3 Agent)

From [SAM 3 GitHub](https://github.com/facebookresearch/sam3/blob/main/README.md):
> `sam3_agent.ipynb`: Demonstrates the use of SAM 3 Agent to segment complex text prompt on images.

SAM 3 Agent handles:
- Multi-step reasoning for complex prompts
- Integration with multimodal large language models (MLLMs)
- Distilling longer referring expressions into concept prompts

**Use Case:** An MLLM generates a detailed description, which is then distilled into an appropriate SAM 3 concept prompt for segmentation.

---

## Prompt Format Guidelines

### What Works Well

**1. Short Noun Phrases**
- Keep prompts concise and specific
- Focus on the core concept name
- Add 1-2 key distinguishing attributes

**Good Examples:**
- `"yellow school bus"` (3 words: color + type + category)
- `"player in white"` (3 words: role + attribute)
- `"shipping container"` (2 words: material/function + category)
- `"solar panels"` (2 words: type + category)

**2. Visual Attributes**
- Color: `"red"`, `"white"`, `"yellow"`
- Pattern: `"striped"`
- Material: implied in object names

**3. Specific Object Categories**
- Use precise nouns: `"container"` not `"thing"`
- Include category when helpful: `"school bus"` vs just `"bus"`

### What to Avoid

**1. Long Complex Sentences**
- SAM 3 expects "short noun phrases"
- Not designed for full sentence descriptions
- Use SAM 3 Agent for complex language

**2. Abstract Concepts**
- Focus on visually groundable concepts
- Avoid purely abstract terms

**3. Ambiguous Pronouns**
- Be specific about what you want to segment
- Don't use `"it"`, `"that"`, `"the one"`

---

## Negative Prompts (Red Font Examples)

From [SAM 3 GitHub](https://github.com/facebookresearch/sam3/blob/main/README.md):
> "Phrases that have no matching objects (negative prompts) have no masks, shown in red font in the figure."

**How Negative Prompts Work:**
- When a concept doesn't exist in the image, SAM 3 returns no masks
- The presence token determines absence
- Critical for precision - avoiding false positives

**Training Data Includes:**
- Hard negative mining: phrases visually similar but semantically distinct
- Improves discrimination between related concepts
- Part of SA-Co dataset quality control

---

## Text + Visual Prompt Combinations

### Interactive Refinement

From [Roboflow](https://blog.roboflow.com/what-is-sam3/):
> "SAM 3 keeps the same interactive refinement loop -- positive clicks to add, negative to remove regions -- but these can now be applied to concept-wide detections"

**Workflow:**
1. Start with text prompt: `"vehicle"`
2. Model finds all vehicles
3. Use positive/negative clicks to refine specific instances
4. Clicks add/remove regions from text-detected masks

### Exemplar Prompts

From [MarkTechPost](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/):
> "Concept prompts combine short noun phrases with visual exemplars. The model supports detailed phrases such as 'yellow school bus' or 'player in red' and can also use exemplar crops as positive or negative examples."

**Use Cases:**
- Text describes the concept
- Exemplar crops disambiguate fine-grained visual differences
- Combine both for maximum precision

---

## Code Examples

### Basic Text Prompt Usage

From [SAM 3 GitHub](https://github.com/facebookresearch/sam3/blob/main/README.md):

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(
    state=inference_state,
    prompt="<YOUR_TEXT_PROMPT>"  # e.g., "yellow school bus"
)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

### Video Text Prompt

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>"  # JPEG folder or MP4 file

# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)

# Add text prompt
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,  # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",  # e.g., "player in red"
    )
)
output = response["outputs"]
```

---

## Prompt Engineering Best Practices

### 1. Start Simple, Add Specificity

**Iterative Approach:**
1. Try generic prompt: `"person"`
2. If too many results, add attribute: `"person in red"`
3. If still ambiguous, be more specific: `"player in red jersey"`

### 2. Use Visual Attributes for Discrimination

**When multiple similar objects exist:**
- Add color: `"white car"` vs `"black car"`
- Add pattern: `"striped shirt"`
- Add material: `"glass bottle"` vs `"plastic bottle"`

### 3. Match Training Data Vocabulary

**SAM 3 was trained on:**
- 270K unique concepts from SA-Co benchmark
- 4M+ automatically annotated concepts
- Common noun phrases extracted from image descriptions

**Tip:** Use natural, commonly-used object names rather than technical jargon.

### 4. Consider the Presence Token

**For closely related concepts:**
- Be explicit about distinguishing features
- The presence token helps, but clear prompts improve accuracy
- Think about what makes your target visually distinct

### 5. Leverage Interactive Refinement

**When text alone isn't enough:**
1. Start with best text prompt
2. Use positive clicks on missed instances
3. Use negative clicks on false positives
4. Combine text + click for precision

### 6. Use SAM 3 Agent for Complex Descriptions

**When you need:**
- Multi-step reasoning
- Complex spatial relationships
- Longer referring expressions
- MLLM integration

---

## Performance Expectations

### Text Prompt Accuracy

From [MarkTechPost](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/):

**Image Segmentation (SA-Co/Gold cgF1):**
- SAM 3: **54.1** (instance segmentation)
- SAM 3: **55.7** (box detection)
- Human: **72.8** (instance), **74.0** (box)
- SAM 3 achieves 75-80% of human performance

**Comparison with Competitors:**
- OWLv2: 24.6 cgF1
- DINO-X: 21.3 cgF1
- Gemini 2.5: 13.0 cgF1

**SAM 3 is 2x+ better than any competing system on text-prompted segmentation.**

### Inference Speed

From [Roboflow](https://blog.roboflow.com/what-is-sam3/):
> "SAM 3 runs at ~30 ms per image on an H200 GPU, handling 100+ objects"

- 848M parameters (~3.4 GB)
- Server-scale model
- Real-time video tracking capability

---

## Real-World Applications

### 1. Content Creation (Instagram Edits)

From [Medium](https://medium.com/@harsh.vardhan7695/meta-sam-3-the-ai-that-understands-find-every-red-hat-b489d341977b):
> "Within 24 hours, it powered Instagram Edits' most-used feature, enabling hundreds of thousands of creators to apply professional-grade effects by simply describing what they want to segment."

**Use Cases:**
- Spotlighting specific objects
- Motion trails on tracked subjects
- Face pixelation
- Professional masking without After Effects

### 2. Data Annotation

From [Roboflow](https://blog.roboflow.com/what-is-sam3/):
- Auto-labeling with text prompts
- Training smaller, faster supervised models
- Dataset creation at scale

### 3. Video Analysis

- Sports analytics: Track specific players by team color
- Security: Find all instances of specific objects
- E-commerce: Product identification and tracking

---

## Sources

**Primary Sources:**
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3/blob/main/README.md) - Official documentation and code examples
- [Meta AI SAM 3 Blog](https://ai.meta.com/blog/segment-anything-model-3/) - Official announcement (accessed 2025-11-23)

**Technical Analysis:**
- [Roboflow: What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) - Comprehensive guide with examples (accessed 2025-11-23)
- [MarkTechPost: SAM 3 Analysis](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) - Technical deep dive (accessed 2025-11-23)

**Additional References:**
- [Medium: Meta SAM 3 Overview](https://medium.com/@harsh.vardhan7695/meta-sam-3-the-ai-that-understands-find-every-red-hat-b489d341977b) - Application examples (accessed 2025-11-23)
- [PyPI SAM3 Package](https://pypi.org/project/sam3/) - Python package documentation

**Example Notebooks:**
- `sam3_image_predictor_example.ipynb` - Text and visual box prompts on images
- `sam3_video_predictor_example.ipynb` - Text prompts on videos with refinement
- `sam3_agent.ipynb` - Complex text prompt handling

---

## Key Takeaways

1. **Text prompts are short noun phrases** - Keep them concise with 1-2 key attributes
2. **Presence token enables fine discrimination** - "player in white" vs "player in red" now works
3. **270K+ concepts supported** - 50x more than previous benchmarks
4. **75-80% of human performance** - 2x better than competitors
5. **Combine text + clicks for precision** - Interactive refinement after text detection
6. **Use SAM 3 Agent for complex prompts** - Multi-step reasoning capability
7. **30ms per image** - Fast enough for real-time applications on H200 GPU
