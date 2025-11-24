# SAM 3 Interactive Refinement

## Overview

SAM 3 introduces a powerful interactive refinement system that combines text-based concept prompts with geometric visual prompts (points, boxes, masks) for precise segmentation control. This hybrid approach enables users to start with high-level semantic prompts and iteratively refine results using spatial cues.

## Core Concept: Hybrid Prompting

SAM 3's refinement workflow bridges two prompting paradigms:
- **Concept Prompts**: Text descriptions or image exemplars that specify WHAT to segment
- **Visual Prompts**: Points, boxes, and masks that specify WHERE to segment

The key innovation is that these can be **combined in a single inference pass**, allowing text to constrain the semantic category while visual prompts refine spatial boundaries.

## Points After Text: The Refinement Loop

### How It Works

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

> "SAM 3 keeps the same interactive refinement loop - positive clicks to add, negative to remove regions - but these can now be applied to concept-wide detections or to individual masklets within videos."

The workflow:
1. **Initial text prompt** - User provides text like "person" or "red car"
2. **Model detection** - SAM 3 finds all matching instances
3. **User refinement** - Add positive/negative points to correct errors
4. **Mask update** - Model refines based on combined prompts

### Technical Implementation

From [HuggingFace SAM 3 Documentation](https://huggingface.co/facebook/sam3) (accessed 2025-11-23):

```python
# Combined Prompts (Text + Negative Box)
# Segment "handle" but exclude the oven handle using a negative box

text = "handle"
# Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
oven_handle_box = [40, 183, 318, 204]
input_boxes = [[oven_handle_box]]

inputs = processor(
    images=kitchen_image,
    text=text,
    input_boxes=input_boxes,
    input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
    return_tensors="pt"
).to(device)

# This will segment pot handles but exclude the oven handle
```

## Combining Text + Geometry Prompts

### Prompt Types and Labels

SAM 3 supports sophisticated prompt combinations:

**Positive prompts (label=1)**:
- Add regions that SHOULD be segmented
- Reinforce the text-specified concept

**Negative prompts (label=0)**:
- Exclude regions that should NOT be segmented
- Filter out false positives from text prompt

### Use Cases

From [aifilms.ai](https://studio.aifilms.ai/blog/meta-sam3-text-segmentation-tracking) (accessed 2025-11-23):

> "Combine text and exemplar prompts for refined control. Text constrains the general category while exemplars specify visual characteristics."

**Example scenarios**:

1. **Disambiguation**: Text "car" + negative box on parked car = only moving cars
2. **Precision**: Text "person" + positive point on specific individual = that person only
3. **Filtering**: Text "laptop" + negative boxes on closed laptops = only open laptops

### Multiple Visual Prompts

From [HuggingFace Documentation](https://huggingface.co/facebook/sam3) (accessed 2025-11-23):

```python
# Multiple Box Prompts (Positive and Negative)
# Define two positive boxes (e.g., dial and button on oven)

box1_xyxy = [59, 144, 76, 163]  # Dial box
box2_xyxy = [87, 148, 104, 159]  # Button box
input_boxes = [[box1_xyxy, box2_xyxy]]
input_boxes_labels = [[1, 1]]  # Both positive

inputs = processor(
    images=kitchen_image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt"
).to(device)
```

Multiple positive boxes act as exemplars, teaching the model what visual pattern to look for.

## Refinement Workflow

### Image Refinement

**Step-by-step process**:

1. **Load image and initialize**:
```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open("image.jpg")
inference_state = processor.set_image(image)
```

2. **Start with text prompt**:
```python
output = processor.set_text_prompt(
    state=inference_state,
    prompt="shipping container"
)
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

3. **Add refinement points** (using tracker for point prompts):
```python
# Multiple points for refinement
input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points
input_labels = [[[1, 1]]]  # Both positive clicks

inputs = processor(
    images=raw_image,
    input_points=input_points,
    input_labels=input_labels,
    return_tensors="pt"
).to(device)
```

### Video Refinement

From [GitHub SAM 3 Repository](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):

The `sam3_video_predictor_example.ipynb` notebook demonstrates:
> "how to prompt SAM 3 with text prompts on videos, and doing further interactive refinements with points."

**Video refinement workflow**:

1. **Initialize video session**:
```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
```

2. **Add text prompt on initial frame**:
```python
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,
        text="person",
    )
)
```

3. **Refine with points on any frame**:
```python
# Add point refinement after initial text detection
processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=[[[[210, 350]]]],
    input_labels=[[[1]]],
)
```

4. **Propagate refined masks through video**:
```python
for sam3_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam3_output.pred_masks],
        original_sizes=[[session.video_height, session.video_width]]
    )[0]
    video_segments[sam3_output.frame_idx] = video_res_masks
```

## User Experience Design

### Interactive Annotation Workflow

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

SAM 3's refinement system enables efficient dataset annotation:

1. **Bulk detection** - Text prompt finds all instances
2. **Quick review** - User scans for errors
3. **Targeted correction** - Click to add missed objects or remove false positives
4. **Export** - High-quality annotations with minimal effort

### Key UX Principles

**Positive clicks (label=1)**:
- Click on object centers to add
- Click on missed instances
- Each click is encoded as a positional token

**Negative clicks (label=0)**:
- Click on false positives to remove
- Click on background mistakenly included
- Removes regions from the mask

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/):
> "Internally, each click is encoded as a positional token and fused into the prompt embedding space before being re-decoded into refined masks. This design allows incremental correction without restarting inference, enabling fluid dataset annotation and fine-tuning workflows."

### Real-Time Feedback

The refinement system provides:
- **Immediate mask updates** after each click
- **No re-inference needed** for entire image
- **Cumulative refinement** - each click builds on previous
- **Undo capability** through prompt removal

## Comparison with SAM 2 Refinement

### What's Inherited

From SAM 2, SAM 3 maintains:
- Point and box prompts for spatial guidance
- Positive/negative click paradigm
- Mask input for further refinement
- Video tracking with memory attention

### What's New in SAM 3

1. **Text + Visual combination**: Can start with text and refine with points
2. **Concept-wide refinement**: Points can affect all instances of a concept
3. **Presence token**: Better discrimination of similar prompts before refinement
4. **Batched refinement**: Multiple prompt types in single forward pass

From [GitHub README](https://github.com/facebookresearch/sam3):
> "The tracker inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement."

## Advanced Patterns

### Batched Mixed Prompts

Process different images with different prompt types simultaneously:

```python
# Image 1: text prompt "laptop"
# Image 2: visual prompt (dial box)
box2_xyxy = [59, 144, 76, 163]

inputs = processor(
    images=images,
    text=["laptop", None],  # Only first image has text
    input_boxes=[None, [box2_xyxy]],  # Only second image has box
    input_boxes_labels=[None, [1]],  # Positive box for second image
    return_tensors="pt"
).to(device)

# Both images processed in single forward pass
```

### Semantic + Instance Refinement

SAM 3 provides both output types:

```python
# Instance segmentation masks
instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

# Semantic segmentation (single channel)
semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]
```

This enables refinement at either the instance or semantic level.

### Streaming Video Refinement

For real-time applications:

```python
# Process frames one by one (streaming mode)
for frame_idx, frame in enumerate(video_frames):
    inputs = processor(images=frame, device=device, return_tensors="pt")

    if frame_idx == 0:
        # Add initial prompt with refinement points
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
            original_size=inputs.original_sizes[0],
        )

    # Process current frame
    output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
```

## Performance Considerations

### Speed
- ~30 ms per image on H200 GPU
- Handles 100+ objects simultaneously
- Incremental refinement without full re-inference

### Model Size
- ~840M parameters (3.4 GB)
- Server-scale model, not edge-deployable
- Can be used to label data for smaller models

### Streaming Limitations

From [HuggingFace Documentation](https://huggingface.co/facebook/sam3):
> "Streaming inference disables hotstart heuristics that remove unmatched and duplicate objects, as these require access to future frames to make informed decisions. This may result in more false positive detections and duplicate object tracks compared to pre-loaded video inference."

## Best Practices

### For Dataset Annotation

1. **Start broad** - Use text prompts to find all candidates
2. **Review systematically** - Scan for false positives/negatives
3. **Refine efficiently** - Single clicks for each correction
4. **Use negative prompts** - Exclude entire regions via boxes

### For Production Applications

1. **Pre-loaded video** - Better quality than streaming
2. **Threshold tuning** - Adjust `threshold` and `mask_threshold` for use case
3. **Batch processing** - Combine multiple images when possible
4. **Cache inference state** - Reuse for multiple prompts on same image

### For Interactive Tools

1. **Show immediate feedback** - Update masks after each click
2. **Support undo** - Allow removing previous refinements
3. **Guide users** - Indicate where refinement would help most
4. **Save refinement history** - Enable iterative improvement

## Sources

**Primary Documentation:**
- [HuggingFace SAM 3 Model Card](https://huggingface.co/facebook/sam3) - Complete API reference with code examples (accessed 2025-11-23)
- [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository with notebooks (accessed 2025-11-23)

**Technical Articles:**
- [Roboflow Blog - What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) - Comprehensive overview with examples (accessed 2025-11-23)
- [aifilms.ai - Meta SAM 3](https://studio.aifilms.ai/blog/meta-sam3-text-segmentation-tracking) - Prompt combination strategies (accessed 2025-11-23)

**Example Notebooks:**
- `sam3_video_predictor_example.ipynb` - Video refinement with points
- `sam3_image_predictor_example.ipynb` - Image refinement workflow
