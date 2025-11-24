# SAM 3 Integration with Multimodal LLMs (MLLMs)

## Overview

SAM 3 (Segment Anything Model 3) can serve as a powerful visual grounding tool for Multimodal Large Language Models (MLLMs). This integration enables vision-language systems to perform precise pixel-level segmentation based on natural language understanding and reasoning.

## SAM 3 as a Tool for MLLMs

### Official Support

From the SAM 3 GitHub repository README:
> "There are additional notebooks in the examples directory that demonstrate how to use SAM 3 for interactive instance segmentation in images and videos (SAM 1/2 tasks), **or as a tool for an MLLM**, and how to run evaluations on the SA-Co dataset."

This confirms Meta's official intention for SAM 3 to be used as a vision tool within MLLM pipelines.

### Native Text-Prompting Capability

Unlike SAM 1/2 which required external detectors for text-based prompting, SAM 3 has **built-in text-to-segmentation**:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

# Direct text prompting - no external detector needed!
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="a red car")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

This native capability simplifies MLLM integration significantly.

## Integration Patterns

### Pattern 1: MLLM Reasoning + SAM 3 Execution

**Pipeline Flow:**
1. User provides image + complex query to MLLM (GPT-4V, Gemini, Claude)
2. MLLM reasons about the query and generates noun phrase(s)
3. SAM 3 segments all instances matching the noun phrase(s)
4. Results returned to MLLM for further analysis or to user

**Example Use Case:**
- User: "What's causing the traffic jam in this image?"
- MLLM reasons: "I see multiple vehicles stopped at an intersection"
- SAM 3 prompt: "vehicles", "traffic light", "road"
- Result: Segmented masks for all relevant objects
- MLLM: "The traffic jam appears to be caused by the red traffic light at the intersection, where 12 vehicles are currently stopped."

### Pattern 2: SAM 3 Agent for Complex Prompts

SAM 3 includes a built-in agent capability (see `sam3_agent.ipynb` notebook) that can:
- Handle complex text prompts requiring multi-step reasoning
- Break down compound queries into individual concepts
- Aggregate results from multiple segmentation passes

```python
# SAM 3 Agent handles complex prompts like:
# "all people wearing hats except the one holding an umbrella"
```

### Pattern 3: Visual Question Answering with Grounding

**Integration for VQA:**
1. MLLM receives question about image regions
2. MLLM identifies objects of interest
3. SAM 3 provides precise segmentation masks
4. Masks used for:
   - Region-specific analysis
   - Object counting
   - Spatial relationship verification
   - Attribute extraction

### Pattern 4: Referring Expression Segmentation

When MLLM generates referring expressions, SAM 3 can ground them:
- MLLM output: "the person on the left wearing blue"
- SAM 3 handles discriminative prompts with presence tokens
- Presence token mechanism distinguishes "player in white" vs "player in red"

## API Patterns for MLLM Integration

### Image-Based Integration

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Initialize SAM 3
model = build_sam3_image_model()
processor = Sam3Processor(model)

def segment_from_mllm_output(image_path, concept_list):
    """
    Takes MLLM-generated concepts and returns segmentation masks.

    Args:
        image_path: Path to image
        concept_list: List of noun phrases from MLLM reasoning

    Returns:
        Dictionary mapping concepts to their masks
    """
    image = Image.open(image_path)
    inference_state = processor.set_image(image)

    results = {}
    for concept in concept_list:
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=concept
        )
        results[concept] = {
            "masks": output["masks"],
            "boxes": output["boxes"],
            "scores": output["scores"]
        }

    return results
```

### Video-Based Integration

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

def track_mllm_concepts_in_video(video_path, concepts, start_frame=0):
    """
    Track MLLM-identified concepts through video.
    """
    # Start session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    # Add prompts for each concept
    for concept in concepts:
        video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=start_frame,
                text=concept,
            )
        )

    # Propagate through video
    return video_predictor.handle_request(
        request=dict(
            type="propagate",
            session_id=session_id,
        )
    )
```

### Batched Inference for Efficiency

For processing multiple MLLM queries efficiently:

```python
# See sam3_image_batched_inference.ipynb
# Process multiple images/prompts in parallel for better throughput
```

## Comparison with Previous Approaches

### Before SAM 3: Grounded-SAM Pipelines

Previous integrations required multiple models:
1. **Grounding DINO** - Open-vocabulary detector
2. **SAM 1/2** - Segment from bounding boxes
3. External text encoder alignment

**Grounded-SAM workflow:**
```
Text prompt -> Grounding DINO -> Boxes -> SAM -> Masks
```

### With SAM 3: Unified Pipeline

SAM 3 consolidates this into one model:
```
Text prompt -> SAM 3 -> Masks + Boxes
```

**Advantages:**
- Single model to deploy
- Faster inference (no cascading)
- Better alignment (end-to-end training)
- Handles 270K concepts (vs limited detector vocabularies)
- Native presence token for discriminative prompts

## Example MLLM Integrations

### With GPT-4V / GPT-4o

```python
import openai
from sam3_integration import segment_from_mllm_output

def visual_qa_with_segmentation(image_path, question):
    # Step 1: GPT-4V analyzes image and question
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": f"{question}\n\nList the key objects to segment as noun phrases."}
                ]
            }
        ]
    )

    # Step 2: Extract concepts from GPT-4V response
    concepts = parse_concepts(response.choices[0].message.content)

    # Step 3: SAM 3 segments each concept
    segmentation_results = segment_from_mllm_output(image_path, concepts)

    return segmentation_results
```

### With Gemini

```python
import google.generativeai as genai

def gemini_grounded_segmentation(image_path, instruction):
    # Gemini analyzes and generates concepts
    model = genai.GenerativeModel('gemini-2.5-flash')

    image = genai.upload_file(image_path)
    response = model.generate_content([
        image,
        f"{instruction}\n\nProvide a list of specific noun phrases to segment."
    ])

    # SAM 3 grounds the concepts
    concepts = parse_concepts(response.text)
    return segment_from_mllm_output(image_path, concepts)
```

### With Claude

```python
import anthropic

def claude_visual_analysis(image_path, task):
    client = anthropic.Anthropic()

    # Claude reasons about the image
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", ...}},
                    {"type": "text", "text": task}
                ]
            }
        ]
    )

    # Extract concepts and segment with SAM 3
    concepts = extract_noun_phrases(message.content)
    return segment_from_mllm_output(image_path, concepts)
```

## Advanced Integration: LISA-Style Reasoning Segmentation

### LISA (Language Instructed Segmentation Assistant)

LISA pioneered the integration of LLMs with segmentation models:
- Uses LLM for reasoning about complex queries
- Special `<SEG>` token triggers segmentation
- LLM hidden states converted to SAM prompts

From [LISA paper](https://arxiv.org/abs/2308.00692) (CVPR 2024, 874 citations):
> "We introduce LISA: a large Language Instructed Segmentation Assistant, a multimodal LLM capable of producing segmentation masks."

### SAM 3 Advantages for LISA-Style Systems

1. **No need for embedding alignment** - SAM 3 natively understands text
2. **Richer concept vocabulary** - 270K concepts vs fixed vocabularies
3. **Presence tokens** - Better discrimination between similar concepts
4. **Video support** - Track segmented objects through time

### SAM4MLLM Approach

From [SAM4MLLM paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10570.pdf) (ECCV 2024):
> "SAM4MLLM integrates the Segment Anything Model (SAM) with Multi-Modal Large Language Models (MLLMs) to achieve precise object localization."

SAM 3's unified architecture simplifies this integration significantly.

## Use Cases

### 1. Visual Content Moderation
- MLLM identifies policy-violating content
- SAM 3 segments specific regions for review
- Enables targeted blurring/removal

### 2. Automated Data Annotation
- MLLM generates object descriptions
- SAM 3 creates instance masks
- Scales annotation pipelines

### 3. Interactive Image Editing
- User describes edits in natural language
- MLLM interprets and identifies regions
- SAM 3 provides precise edit masks

### 4. Robotics and Embodied AI
- MLLM processes natural language instructions
- SAM 3 grounds objects in scene
- Robot acts on segmented regions

### 5. Medical Image Analysis
- MLLM interprets clinical queries
- SAM 3 segments anatomical structures
- Enables region-specific diagnosis

## Performance Considerations

### Latency
- SAM 3 single-model inference faster than multi-stage pipelines
- Video predictor enables real-time tracking applications

### Throughput
- Batched inference support for processing multiple queries
- Shared vision encoder between detector and tracker

### Quality
- 75-80% of human performance on SA-Co benchmark
- Presence token improves discrimination accuracy
- Open-vocabulary coverage (270K concepts)

## Comparison: SAM 3 vs Competitors for MLLM Integration

| Feature | SAM 3 | Grounded-SAM | LISA |
|---------|-------|--------------|------|
| Native text prompts | Yes | No (needs GDINO) | Via LLM |
| Concept vocabulary | 270K | Detector-limited | LLM-generated |
| Video support | Yes | Limited | No |
| Presence tokens | Yes | No | No |
| Single model | Yes | No (2+ models) | No (LLM + SAM) |
| End-to-end trained | Yes | No | Partially |

## Sources

**Official SAM 3 Resources:**
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3)
- [SAM 3 Project Page](https://ai.meta.com/sam3)
- [SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)

**Related Work:**
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) - Multi-model pipeline (17.2k stars)
- [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692) - CVPR 2024
- [SAM4MLLM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10570.pdf) - ECCV 2024
- [LLM-Seg](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Wang_LLM-Seg_Bridging_Image_Segmentation_and_Large_Language_Model_Reasoning_CVPRW_2024_paper.pdf) - CVPR 2024 Workshop

**Web Research (accessed 2025-11-23):**
- Roboflow Blog: SAM 3 Integration Guide
- Ultralytics YOLO Docs: SAM 3 Model Documentation
- Hugging Face: SAM 3 Model Card

## Summary

SAM 3 represents a significant advancement for MLLM integration by providing:

1. **Native text-to-segmentation** - No external detector needed
2. **Unified architecture** - Simpler deployment and maintenance
3. **Massive concept vocabulary** - 270K concepts for open-vocabulary understanding
4. **Presence tokens** - Better discrimination for complex prompts
5. **Video support** - Track objects through time
6. **Official MLLM tool support** - Documented example notebooks

The combination of MLLMs for reasoning and SAM 3 for pixel-precise grounding creates powerful vision-language systems capable of understanding and segmenting virtually any visual concept described in natural language.
