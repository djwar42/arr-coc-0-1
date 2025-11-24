# SAM 3 Example Notebooks Walkthrough

## Overview

SAM 3 provides 6 official example notebooks in the `examples/` directory demonstrating core capabilities for image segmentation, video tracking, batched inference, agent-based complex prompting, and SA-Co dataset visualization. This walkthrough covers each notebook's purpose, key code patterns, and practical usage.

---

## 1. Image Predictor Example

**File**: `sam3_image_predictor_example.ipynb`
**Size**: 2.76 MB (includes output visualizations)
**Purpose**: Demonstrates text and visual box prompts on single images

### Key Capabilities

- **Text prompts**: Segment objects using natural language descriptions
- **Box prompts**: Provide bounding boxes for precise object selection
- **Combined prompting**: Use text with geometric refinement

### Core Code Pattern

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("your_image.jpg")
inference_state = processor.set_image(image)

# Text prompt
output = processor.set_text_prompt(
    state=inference_state,
    prompt="your text prompt"
)

# Get results
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

### Key Features

- Single image processing
- Returns masks, bounding boxes, and confidence scores
- Supports confidence threshold configuration
- Works with standard PIL Image format

---

## 2. Video Predictor Example

**File**: `sam3_video_predictor_example.ipynb`
**Size**: 54.2 KB
**Purpose**: Interactive video segmentation and dense tracking

### Key Capabilities

- **Text prompts**: Natural language descriptions for object detection
- **Point prompts**: Positive/negative clicks for refinement
- **Session-based inference**: Stateful tracking across video frames
- **Object management**: Add, remove, and refine tracked objects
- **Multi-GPU support**: Scale inference across available GPUs

### Video Input Formats

- JPEG folder with frames named `<frame_index>.jpg`
- MP4 video files
- Extract frames with ffmpeg: `ffmpeg -i video.mp4 -q:v 2 -start_number 0 output/'%05d.jpg'`

### Core Code Pattern

```python
from sam3.model_builder import build_sam3_video_predictor

# Initialize predictor (supports multi-GPU)
gpus_to_use = range(torch.cuda.device_count())
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

# Start session
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,  # JPEG folder or MP4
    )
)
session_id = response["session_id"]

# Add text prompt
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="person",
    )
)
```

### Session Operations

**Reset Session** (clear all objects):
```python
predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)
```

**Propagate in Video** (track across frames):
```python
for response in predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]
```

**Remove Object**:
```python
predictor.handle_request(
    request=dict(
        type="remove_object",
        session_id=session_id,
        obj_id=2,
    )
)
```

**Add Point Prompts** (positive/negative clicks):
```python
# Coordinates in relative format (0-1 range)
points_tensor = torch.tensor([[x/width, y/height]], dtype=torch.float32)
labels_tensor = torch.tensor([1], dtype=torch.int32)  # 1=positive, 0=negative

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        points=points_tensor,
        point_labels=labels_tensor,
        obj_id=obj_id,
    )
)
```

### Coordinate Conversion

```python
def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative (0-1 range)"""
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [[x/IMG_WIDTH, y/IMG_HEIGHT, w/IMG_WIDTH, h/IMG_HEIGHT]
                for x, y, w, h in coords]
```

### Session Lifecycle

```python
# Close session (free GPU resources)
predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)

# Shutdown predictor (free multi-GPU process group)
predictor.shutdown()
```

---

## 3. Batched Image Inference

**File**: `sam3_image_batched_inference.ipynb`
**Size**: 4.42 MB (includes output visualizations)
**Purpose**: Process multiple images simultaneously for throughput optimization

### Key Capabilities

- Batch processing multiple images
- Memory-efficient inference
- Optimized for large-scale annotation tasks
- GPU utilization maximization

### Use Cases

- Large-scale dataset annotation
- Production batch processing
- High-throughput inference pipelines
- Automated labeling systems

### Performance Considerations

- Batch size depends on GPU memory
- Images should be similar resolution for efficiency
- Use with memory management best practices
- Consider async processing for very large batches

---

## 4. SAM 3 Agent

**File**: `sam3_agent.ipynb`
**Size**: 6.83 KB
**Purpose**: Use MLLM (Multimodal LLM) with SAM 3 for complex text queries

### Key Capabilities

- Complex text prompt handling (e.g., "the leftmost child wearing blue vest")
- Multi-step reasoning with LLM
- SAM 3 as tool for MLLM
- Agentic behavior for nuanced segmentation

### Architecture

The SAM 3 Agent combines:
1. **MLLM** (e.g., Qwen3-VL-8B-Thinking) for understanding complex queries
2. **SAM 3** as segmentation tool called by the LLM

### LLM Configuration

```python
LLM_CONFIGS = {
    "qwen3_vl_8b_thinking": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
    },
    # Can add external API models (GPT-4V, Gemini, etc.)
}
```

### vLLM Server Setup

```bash
# Create separate conda env (avoid dependency conflicts)
conda create -n vllm python=3.12
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Start vLLM server
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --tensor-parallel-size 4 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001
```

### Core Code Pattern

```python
from functools import partial
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

# Setup model
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)

# Configure LLM endpoint
send_generate_request = partial(
    send_generate_request_orig,
    server_url=LLM_SERVER_URL,
    model=llm_config["model"],
    api_key=llm_config["api_key"]
)
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

# Run inference
output_image_path = run_single_image_inference(
    image="path/to/image.jpg",
    prompt="the leftmost child wearing blue vest",
    llm_config=llm_config,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    debug=True,
    output_dir="agent_output"
)
```

### Prompt Complexity Examples

Simple SAM 3 prompts:
- "person"
- "car"
- "dog"

Complex Agent prompts (require MLLM reasoning):
- "the leftmost child wearing blue vest"
- "the person holding a red umbrella"
- "the largest building in the background"

---

## 5. SA-Co Gold/Silver Visualization

**File**: `saco_gold_silver_vis_example.ipynb`
**Size**: 7.8 KB
**Purpose**: Visualize examples from SA-Co image evaluation datasets

### Key Capabilities

- Load and explore SA-Co/Gold and SA-Co/Silver annotations
- Visualize masks with different display modes
- Understand annotation format and structure
- Separate positive (with masks) and negative (no masks) noun phrases

### Data Structure

```python
annot_dfs = {
    "gold_fg_sports_equipment_merged_a_release_test": {
        "info": {...},           # Dataset metadata
        "images": DataFrame,      # Image information with text_input (noun phrases)
        "annotations": DataFrame  # Segmentation masks in RLE format
    }
}
```

### Loading Annotations

```python
from glob import glob
import sam3.visualization_utils as utils

ANNOT_DIR = "/path/to/annotations"
IMG_DIR = "/path/to/images"

# Load Gold annotations
annot_file_list = glob(os.path.join(ANNOT_DIR, "*gold*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)
```

### Annotation Access

```python
# Get images dataframe
images_df = annot_dfs["gold_fg_food_merged_a_release_test"]["images"]

# Get annotations dataframe
annotations_df = annot_dfs["gold_fg_food_merged_a_release_test"]["annotations"]

# Group annotations by image_id
gt_image_np_map = {img["id"]: img for _, img in images_df.iterrows()}
gt_image_np_ann_map = defaultdict(list)
for _, ann in annotations_df.iterrows():
    gt_image_np_ann_map[ann["image_id"]].append(ann)
```

### Visualization

```python
from pycocotools import mask as mask_util

# Decode RLE masks
annot_masks = [mask_util.decode(ann["segmentation"]) for ann in annotations]

# Draw masks on white background
COLORS = utils.pascal_color_map()[1:]
all_masks = utils.draw_masks_to_frame(
    frame=np.ones_like(img)*255,
    masks=annot_masks,
    colors=COLORS[:len(annot_masks)]
)

# Draw masks overlaid on image
masked_frame = utils.draw_masks_to_frame(
    frame=img,
    masks=annot_masks,
    colors=COLORS[:len(annot_masks)]
)
```

### Positive vs Negative Examples

```python
# Positive NPs have matching annotations
positiveNPs = [img_id for img_id in gt_image_np_map.keys()
               if img_id in gt_image_np_ann_map and gt_image_np_ann_map[img_id]]

# Negative NPs have no matching objects (shown in red in paper figures)
negativeNPs = [img_id for img_id in gt_image_np_map.keys()
               if img_id not in gt_image_np_ann_map or not gt_image_np_ann_map[img_id]]
```

---

## 6. SA-Co VEval Visualization

**File**: `saco_veval_vis_example.ipynb`
**Size**: 7.23 KB
**Purpose**: Visualize examples from SA-Co video evaluation benchmark

### Key Capabilities

- Load video annotations with temporal tracking
- Visualize masks across multiple frames
- Understand video-noun phrase pair structure
- Track objects with consistent IDs across frames

### Data Structure

```python
annot_dfs = {
    "saco_veval_yt1b_val": {
        "info": {...},              # Dataset metadata
        "videos": DataFrame,         # Video information
        "annotations": DataFrame,    # Frame-level segmentation masks
        "categories": DataFrame,     # Category/noun phrase definitions
        "video_np_pairs": DataFrame  # Video-noun phrase mappings
    }
}
```

### Loading Video Annotations

```python
DATA_DIR = "./sam3_saco_veval_data"
ANNOT_DIR = os.path.join(DATA_DIR, "annotation")

annot_file_list = glob(os.path.join(ANNOT_DIR, "*veval*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)
```

### Video-NP Pair Selection

```python
df_pairs = annot_dfs["saco_veval_yt1b_val"]["video_np_pairs"]

# Filter positive pairs (have masklets)
df_positive_pairs = df_pairs[df_pairs.num_masklets > 0]

# Select random pair
pair_row = df_positive_pairs.iloc[np.random.randint(len(df_positive_pairs))]
video_id = pair_row.video_id
noun_phrase = pair_row.noun_phrase
```

### Frame Annotation Retrieval

```python
# Get all annotations for specific frame
frame, annot_masks, annot_noun_phrases = utils.get_all_annotations_for_frame(
    annot_dfs["saco_veval_yt1b_val"],
    video_id=video_id,
    frame_idx=frame_idx,
    data_dir=DATA_DIR,
    dataset="saco_veval_yt1b_val"
)

# Filter by noun phrase
annot_masks = [m for m, np in zip(annot_masks, annot_noun_phrases)
               if np == noun_phrase]
```

### Temporal Visualization

```python
num_frames_to_show = 5
every_n_frames = 4  # Frame sampling interval

for idx in range(num_frames_to_show):
    sampled_frame_idx = idx * every_n_frames
    frame, annot_masks, _ = utils.get_all_annotations_for_frame(...)

    # Visualize frame, masks only, and overlay
```

---

## Additional Notebooks

The README mentions additional notebooks for:
- Interactive instance segmentation (SAM 1/2 tasks)
- SAM 3 as tool for MLLM integration
- SA-Co dataset evaluation scripts

These extend the core functionality with specialized use cases.

---

## Common Setup Code

### Environment Check

```python
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
```

### GPU Configuration

```python
# TF32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# BFloat16 autocast
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Inference mode
torch.inference_mode().__enter__()
```

### Google Colab Setup

```python
if using_colab:
    !pip install opencv-python matplotlib scikit-learn
    !pip install 'git+https://github.com/facebookresearch/sam3.git'
```

---

## Visualization Utilities

SAM 3 provides `sam3.visualization_utils` with:

- `pascal_color_map()`: Standard color palette for masks
- `draw_masks_to_frame()`: Overlay masks on images
- `load_frame()`: Load video frames
- `prepare_masks_for_visualization()`: Format outputs for display
- `visualize_formatted_frame_output()`: Complete visualization pipeline
- `get_annot_dfs()`: Load annotation JSON files into DataFrames
- `get_all_annotations_for_frame()`: Retrieve frame-specific annotations

---

## Best Practices

### Memory Management

1. Close video sessions when done
2. Shutdown predictor to free GPU resources
3. Use batched inference for multiple images
4. Consider memory when setting batch sizes

### Prompt Engineering

1. Start with simple prompts (single nouns)
2. Use specific attributes for disambiguation ("player in white")
3. For complex queries, use SAM 3 Agent with MLLM
4. Combine text with point/box prompts for refinement

### Video Processing

1. Reset session when switching prompts
2. Use relative coordinates (0-1 range)
3. First inference may be slower (buffer setup)
4. Consider frame sampling for visualization

### Data Handling

1. SA-Co uses RLE format for masks (pycocotools)
2. Separate positive/negative noun phrases
3. Video annotations include temporal tracking IDs
4. Support both Gold (high-quality) and Silver (larger scale) datasets

---

## Sources

**GitHub Repository:**
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Main repository (accessed 2025-11-23)
- [examples/sam3_image_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb)
- [examples/sam3_video_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb)
- [examples/sam3_image_batched_inference.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_batched_inference.ipynb)
- [examples/sam3_agent.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb)
- [examples/saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)
- [examples/saco_veval_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_veval_vis_example.ipynb)

**SA-Co Datasets:**
- [HuggingFace SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)
- [HuggingFace SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver)
- [HuggingFace SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
