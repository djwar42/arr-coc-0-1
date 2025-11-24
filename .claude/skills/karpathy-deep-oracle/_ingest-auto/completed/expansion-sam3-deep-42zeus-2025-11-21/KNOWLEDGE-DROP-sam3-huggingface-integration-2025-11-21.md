# SAM 3 HuggingFace Integration

## Overview

SAM 3 (Segment Anything Model 3) provides comprehensive HuggingFace integration for checkpoint access, model loading, and inference. The model is hosted at `facebook/sam3` on HuggingFace Hub and requires authentication to access checkpoints due to licensing requirements.

## Checkpoint Access Requirements

### Access Request Process

**Step 1: Request Model Access**
- Navigate to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- Click "Agree to share contact information"
- Information is collected per Meta Privacy Policy
- Wait for access approval (typically quick for standard requests)

**Step 2: Authentication Setup**

From [GitHub README](https://github.com/facebookresearch/sam3/blob/main/README.md):

```bash
# Generate access token at https://huggingface.co/settings/tokens
# Then authenticate via CLI
huggingface-cli login
# Or use: hf auth login

# Alternative: Set environment variable
export HF_TOKEN=your_token_here
```

**Step 3: Verify Access**

```python
from huggingface_hub import HfApi
api = HfApi()
# Will raise error if not authenticated
api.model_info("facebook/sam3")
```

## Model Download Methods

### Method 1: Transformers AutoModel (Recommended)

```python
from transformers import Sam3Model, Sam3Processor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Downloads automatically on first use
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
```

### Method 2: Native SAM 3 API

From [HuggingFace Model Card](https://huggingface.co/facebook/sam3):

```python
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor

# For image segmentation
model = build_sam3_image_model()  # Auto-downloads checkpoint
processor = Sam3Processor(model)

# For video segmentation
video_predictor = build_sam3_video_predictor()
```

### Method 3: Manual Download with huggingface-cli

```bash
# Download model files
huggingface-cli download facebook/sam3

# Download to specific directory
huggingface-cli download facebook/sam3 --local-dir ./sam3_model

# Download specific files only
huggingface-cli download facebook/sam3 model.safetensors config.json
```

### Method 4: Python huggingface_hub

```python
from huggingface_hub import snapshot_download, hf_hub_download

# Download entire model
model_path = snapshot_download(repo_id="facebook/sam3")

# Download specific file
config_path = hf_hub_download(
    repo_id="facebook/sam3",
    filename="config.json"
)
```

## Model Architecture

**Model Size**: 848M parameters

**Format**: Safetensors (efficient and safe tensor serialization)

**Components Available**:
- Vision encoder (ViT backbone)
- Text encoder
- DETR-based detector
- Mask decoder
- Tracker (inherited from SAM 2)

## Transformers API Reference

### Core Classes

From [Transformers Documentation](https://huggingface.co/docs/transformers/main/model_doc/sam3):

**Sam3Model** - Main model for image segmentation
```python
from transformers import Sam3Model
model = Sam3Model.from_pretrained("facebook/sam3")
```

**Sam3Processor** - Handles image preprocessing and postprocessing
```python
from transformers import Sam3Processor
processor = Sam3Processor.from_pretrained("facebook/sam3")
```

**Sam3VideoModel** - For video segmentation
```python
from transformers import Sam3VideoModel, Sam3VideoProcessor
model = Sam3VideoModel.from_pretrained("facebook/sam3")
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
```

**Sam3TrackerModel** - For interactive instance segmentation (SAM 2 style)
```python
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
model = Sam3TrackerModel.from_pretrained("facebook/sam3")
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
```

### Basic Usage Examples

**Text Prompt Segmentation**:
```python
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

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

print(f"Found {len(results['masks'])} objects")
# Results contain: masks, boxes, scores
```

**Batched Inference**:
```python
images = [image1, image2]
text_prompts = ["ear", "dial"]

inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)
```

## Fine-Tuning Considerations

### Current Status

As of November 2025, SAM 3 is released for **inference only**. Training code is not publicly released.

### What's Available

- Pre-trained checkpoints (facebook/sam3)
- Inference code (native + Transformers)
- Example notebooks for various use cases
- SA-Co benchmark datasets for evaluation

### Future Fine-Tuning (When Available)

For when training code is released:

```python
# Expected pattern (not yet available)
from transformers import Sam3Model, Sam3Processor, Trainer, TrainingArguments

model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Custom dataset
# training_args = TrainingArguments(...)
# trainer = Trainer(model=model, args=training_args, ...)
# trainer.train()
```

### Development Environment Setup

From [GitHub README](https://github.com/facebookresearch/sam3/blob/main/README.md):

```bash
# Install with training dependencies (for future use)
pip install -e ".[train,dev]"
```

## HuggingFace Spaces and Demos

**Official Spaces Using SAM 3**:
- [akhaliq/sam3](https://huggingface.co/spaces/akhaliq/sam3) - General demo
- [merve/SAM3-video-segmentation](https://huggingface.co/spaces/merve/SAM3-video-segmentation) - Video demo
- [webml-community/SAM3-Tracker-WebGPU](https://huggingface.co/spaces/webml-community/SAM3-Tracker-WebGPU) - WebGPU implementation

## SA-Co Datasets on HuggingFace

SAM 3's benchmark datasets are also hosted on HuggingFace:

- [facebook/SACo-Gold](https://huggingface.co/datasets/facebook/SACo-Gold) - High-quality evaluation set
- [facebook/SACo-Silver](https://huggingface.co/datasets/facebook/SACo-Silver) - Larger training set
- [facebook/SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval) - Video evaluation set

```python
from datasets import load_dataset

# Load SA-Co Gold dataset
dataset = load_dataset("facebook/SACo-Gold")
```

## Common Issues and Solutions

### Authentication Errors

**Error**: "You need to agree to share your contact information to access this model"

**Solution**: Visit https://huggingface.co/facebook/sam3 and accept terms

---

**Error**: "Repository not found" or "401 Unauthorized"

**Solution**:
```bash
# Verify login
huggingface-cli whoami

# Re-authenticate
huggingface-cli login
```

### Download Issues

**Issue**: Slow download speeds

**Solution**: Use `hf_transfer` for faster downloads:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download facebook/sam3
```

**Issue**: Incomplete downloads

**Solution**: Resume with:
```bash
huggingface-cli download facebook/sam3 --resume-download
```

### Memory Issues

**Issue**: CUDA out of memory

**Solution**: Use bfloat16 precision:
```python
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
```

## Pipeline Integration

For simple use cases, use the pipeline API:

```python
from transformers import pipeline

# Mask generation pipeline
generator = pipeline("mask-generation", model="facebook/sam3", device=0)
outputs = generator("image.jpg", points_per_batch=64)
```

## Model Collection

SAM 3 is part of the official Facebook SAM3 collection on HuggingFace:
- Collection: [facebook/SAM3](https://huggingface.co/collections/facebook/sam3)
- Contains 5 items including model variants and related resources

## Prompt Label Conventions

**For points and boxes**:
- `1`: Positive prompt (include this region/object)
- `0`: Negative prompt (exclude this region/object)
- `-10`: Padding value for batched inputs

**Coordinate formats**:
- Input boxes: `[x1, y1, x2, y2]` (xyxy format) in pixel coordinates
- Output boxes (raw): normalized to [0, 1]
- Output boxes (post-processed): absolute pixel coordinates

## Summary

SAM 3's HuggingFace integration provides:
1. **Easy checkpoint access** via `from_pretrained()` after authentication
2. **Multiple model variants** (Sam3Model, Sam3VideoModel, Sam3TrackerModel)
3. **Comprehensive processing** with Sam3Processor for all input/output handling
4. **Benchmark datasets** hosted alongside the model
5. **Demo Spaces** for quick experimentation

The integration is production-ready for inference, with training capabilities expected to be released in future updates.

---

## Sources

**Primary Sources**:
- [facebook/sam3 HuggingFace Model Card](https://huggingface.co/facebook/sam3) (accessed 2025-11-23)
- [Transformers SAM3 Documentation](https://huggingface.co/docs/transformers/main/model_doc/sam3) (accessed 2025-11-23)
- [SAM 3 GitHub Repository README](https://github.com/facebookresearch/sam3/blob/main/README.md) (accessed 2025-11-23)

**Additional References**:
- [HuggingFace Hub Authentication Guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
- [SAM 3 Research Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
