# SAM 3 API Reference

## Overview

SAM 3 provides a clean, well-designed API for promptable segmentation in images and videos. The API follows modern Python design patterns with clear separation between model building, image processing, and video prediction. This document covers the complete API surface including `Sam3Processor`, model builders, and the request-based video predictor interface.

## Core API Components

### 1. Model Builder Functions

#### build_sam3_image_model()

The primary entry point for creating a SAM 3 image segmentation model.

```python
from sam3.model_builder import build_sam3_image_model

# Load the model (downloads checkpoint from HuggingFace)
model = build_sam3_image_model()
```

**Returns**: SAM 3 model instance (848M parameters)

**Features**:
- Automatic checkpoint download from HuggingFace
- Requires `huggingface-cli login` authentication
- Model includes detector, tracker, and shared vision encoder

**Source**: [sam3/__init__.py](https://github.com/facebookresearch/sam3/blob/main/sam3/__init__.py) exports this as the main API

#### build_sam3_video_predictor()

Creates a video predictor for temporal segmentation and tracking.

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
```

**Returns**: Video predictor instance with session-based API

**Features**:
- Session management for video processing
- Request-response pattern for operations
- Supports JPEG folders or MP4 video files

---

## 2. Sam3Processor Class

The `Sam3Processor` is the main high-level API for image segmentation tasks.

**Location**: `sam3/model/sam3_image_processor.py`

### Constructor

```python
from sam3.model.sam3_image_processor import Sam3Processor

processor = Sam3Processor(
    model,                      # SAM 3 model instance
    resolution=1008,            # Input image resolution
    device="cuda",              # Computation device
    confidence_threshold=0.5    # Detection threshold
)
```

**Parameters**:
- `model`: SAM 3 model from `build_sam3_image_model()`
- `resolution`: Image resize resolution (default: 1008x1008)
- `device`: PyTorch device ("cuda" or "cpu")
- `confidence_threshold`: Minimum confidence for mask output (0.0-1.0)

### Core Methods

#### set_image()

Sets the image for inference and computes backbone features.

```python
from PIL import Image

image = Image.open("path/to/image.jpg")
inference_state = processor.set_image(image)
```

**Parameters**:
- `image`: PIL Image, torch.Tensor, or np.ndarray
- `state` (optional): Existing state dict to update

**Returns**: State dictionary containing:
- `original_height`: Input image height
- `original_width`: Input image width
- `backbone_out`: Encoded image features

**Implementation Details**:
- Transforms image to tensor with normalization (mean=0.5, std=0.5)
- Resizes to configured resolution
- Runs backbone feature extraction
- Handles SAM 2 interactive predictor features if available

#### set_image_batch()

Process multiple images for batched inference.

```python
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
batch_state = processor.set_image_batch(images)
```

**Parameters**:
- `images`: List of PIL Images

**Returns**: Batch state with:
- `original_heights`: List of heights
- `original_widths`: List of widths
- `backbone_out`: Batched features

**Use Case**: High-throughput annotation pipelines

#### set_text_prompt()

Run inference with a text prompt.

```python
output = processor.set_text_prompt(
    prompt="a dog",
    state=inference_state
)

# Access results
masks = output["masks"]           # Boolean masks [N, 1, H, W]
boxes = output["boxes"]           # Bounding boxes [N, 4] in xyxy format
scores = output["scores"]         # Confidence scores [N]
masks_logits = output["masks_logits"]  # Raw mask logits
```

**Parameters**:
- `prompt`: Text description of object to segment
- `state`: State from `set_image()`

**Returns**: Updated state with segmentation results

**Key Features**:
- Uses presence token for discrimination
- Filters outputs by confidence threshold
- Converts boxes to original image coordinates

#### add_geometric_prompt()

Add box prompts for geometric guidance.

```python
# Box format: [center_x, center_y, width, height] normalized [0, 1]
output = processor.add_geometric_prompt(
    box=[0.5, 0.5, 0.3, 0.3],  # Center box
    label=True,                 # Positive prompt
    state=inference_state
)
```

**Parameters**:
- `box`: [cx, cy, w, h] normalized coordinates
- `label`: True for positive, False for negative box
- `state`: Inference state

**Special Behavior**:
- If no text prompt set, uses "visual" as placeholder
- Allows combining text + geometry prompts

#### reset_all_prompts()

Clear all prompts and results while keeping image features.

```python
processor.reset_all_prompts(state)
```

**Clears**:
- Text embeddings
- Geometric prompts
- Previous results (masks, boxes, scores)

**Preserves**:
- Backbone features (expensive to recompute)

#### set_confidence_threshold()

Dynamically adjust detection threshold.

```python
# Lower threshold for more detections
processor.set_confidence_threshold(0.3, state=inference_state)
```

**Parameters**:
- `threshold`: New confidence threshold (0.0-1.0)
- `state` (optional): Re-filter existing results

---

## 3. Video Predictor API

The video predictor uses a **request-response pattern** for session management.

### Basic Video Workflow

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

# 1. Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="/path/to/video.mp4",  # or JPEG folder
    )
)
session_id = response["session_id"]

# 2. Add text prompt at specific frame
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,           # Frame to prompt
        text="a person running",
    )
)
output = response["outputs"]
```

### Request Types

#### start_session

Initialize a video processing session.

```python
request = {
    "type": "start_session",
    "resource_path": "/path/to/video"  # MP4 or JPEG folder
}
```

**Response**:
```python
{
    "session_id": "uuid-string",
    "num_frames": 150,
    # ... session metadata
}
```

#### add_prompt

Add segmentation prompts to a frame.

```python
# Text prompt
request = {
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "object description"
}

# Point prompt (for refinement)
request = {
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 10,
    "points": [[x, y]],
    "labels": [1]  # 1=positive, 0=negative
}

# Box prompt
request = {
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 5,
    "box": [x1, y1, x2, y2]
}
```

**Response**: Segmentation outputs for the frame and propagated tracks

### Interactive Refinement Pattern

SAM 3 inherits SAM 2's interactive refinement capabilities:

```python
# Initial text prompt
response = video_predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "player in white jersey"
})

# Refine with positive point
response = video_predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "points": [[150, 200]],
    "labels": [1]  # Include this region
})

# Refine with negative point
response = video_predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "points": [[300, 400]],
    "labels": [0]  # Exclude this region
})
```

---

## 4. API Design Patterns

### State Management Pattern

SAM 3 uses an explicit state dictionary pattern for efficient multi-prompt workflows:

```python
# Expensive: Compute once
state = processor.set_image(image)

# Cheap: Reuse features
output1 = processor.set_text_prompt("dog", state)
processor.reset_all_prompts(state)
output2 = processor.set_text_prompt("cat", state)
processor.reset_all_prompts(state)
output3 = processor.add_geometric_prompt(box, True, state)
```

**Benefits**:
- Backbone features computed once
- Multiple prompts without re-encoding
- Memory efficient for interactive use

### Transform Pipeline

Internal image preprocessing pipeline:

```python
transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(1008, 1008)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

**Key Points**:
- Fixed resolution (1008x1008 default)
- Normalization to [-1, 1] range
- Uses torchvision v2 transforms

### Inference Mode Decorator

All inference methods use PyTorch's inference mode:

```python
@torch.inference_mode()
def set_text_prompt(self, prompt: str, state: Dict):
    # No gradients computed
    # Memory optimized
```

### Output Format

Standardized output structure:

```python
output = {
    "masks": tensor,        # [N, 1, H, W] boolean masks
    "masks_logits": tensor, # [N, 1, H, W] raw logits
    "boxes": tensor,        # [N, 4] in [x1, y1, x2, y2] format
    "scores": tensor,       # [N] confidence scores
}
```

**Coordinate Systems**:
- Boxes: Original image coordinates (not normalized)
- Masks: Original image resolution
- Internal: Normalized [0, 1] for geometric prompts

---

## 5. Complete Usage Examples

### Image Segmentation

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Setup
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load and process image
image = Image.open("scene.jpg")
state = processor.set_image(image)

# Text prompt
output = processor.set_text_prompt("a car", state)

# Visualize results
for i, (mask, box, score) in enumerate(zip(
    output["masks"], output["boxes"], output["scores"]
)):
    print(f"Object {i}: score={score:.3f}, box={box.tolist()}")
```

### Batched Image Inference

```python
# Process multiple images efficiently
images = [Image.open(f"image_{i}.jpg") for i in range(batch_size)]
state = processor.set_image_batch(images)

# Run inference (handles batch dimension internally)
output = processor.set_text_prompt("target object", state)
```

### Video Tracking

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()

# Start session
resp = predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4"
})

# Track object through video
resp = predictor.handle_request({
    "type": "add_prompt",
    "session_id": resp["session_id"],
    "frame_index": 0,
    "text": "the dog"
})

# Access tracked masks across frames
outputs = resp["outputs"]
```

### Combined Text + Geometry

```python
# Set image
state = processor.set_image(image)

# Add text prompt
state = processor.set_text_prompt("bird", state)

# Refine with box
state = processor.add_geometric_prompt(
    box=[0.3, 0.4, 0.2, 0.3],  # Center region
    label=True,
    state=state
)
```

---

## 6. API Best Practices

### Performance Optimization

1. **Reuse state**: Call `set_image()` once, prompt multiple times
2. **Batch when possible**: Use `set_image_batch()` for throughput
3. **Adjust threshold**: Lower threshold for recall, higher for precision

### Memory Management

1. **Clear prompts**: Use `reset_all_prompts()` between queries
2. **Delete state**: Remove state dict when done with image
3. **Device placement**: Use CUDA for GPU acceleration

### Error Handling

```python
# Check state before prompting
if "backbone_out" not in state:
    raise ValueError("Must call set_image before set_text_prompt")
```

### Threshold Tuning

```python
# More detections (lower precision)
processor.set_confidence_threshold(0.3)

# Fewer detections (higher precision)
processor.set_confidence_threshold(0.7)
```

---

## Sources

**GitHub Repository:**
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository
- [sam3/__init__.py](https://github.com/facebookresearch/sam3/blob/main/sam3/__init__.py) - API exports
- [sam3/model/sam3_image_processor.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py) - Complete Sam3Processor implementation (222 lines)

**Official Resources:**
- [SAM 3 Project Page](https://ai.meta.com/sam3) - Meta AI official page
- [HuggingFace Model](https://huggingface.co/facebook/sam3) - Checkpoint downloads
- [SAM 3 Demo](https://segment-anything.com/) - Interactive demo

**Example Notebooks:**
- [sam3_image_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb) - Image API usage
- [sam3_video_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb) - Video API usage
- [sam3_image_batched_inference.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_batched_inference.ipynb) - Batched processing

**Model Architecture:**
- 848M parameters total
- DETR-based detector with text conditioning
- SAM 2 tracker architecture with streaming memory
- Shared vision encoder (Hiera)

---

*Worker 41 - API Reference - Created 2025-11-23*
