# Qwen3-VL Quickstart Guide

**Category**: Usage
**Related**: [examples/00-basic-inference.md](../examples/00-basic-inference.md)

## Installation

### Basic Installation

```bash
# Install transformers (requires >= 4.57.0)
pip install "transformers>=4.57.0"

# Install qwen-vl-utils for preprocessing
pip install qwen-vl-utils==0.0.14

# Optional: Install decord for faster video loading
pip install qwen-vl-utils[decord]

# Optional: Flash Attention 2 for speed
pip install flash-attn --no-build-isolation
```

### Check Installation

```python
import transformers
print(f"transformers version: {transformers.__version__}")  # Should be >= 4.57.0

from qwen_vl_utils import process_vision_info
print("qwen-vl-utils installed successfully!")
```

## Basic Image Inference

### Load Model

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

# Load model (default: auto device placement)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

print(f"Model loaded: {model.config.model_type}")
```

**Model Sizes**:
- `Qwen3-VL-2B-Instruct` - Edge devices (~8GB VRAM)
- `Qwen3-VL-4B-Instruct` - Balanced (~12GB VRAM)
- `Qwen3-VL-8B-Instruct` - Production (~20GB VRAM)
- `Qwen3-VL-32B-Instruct` - Best quality (~80GB VRAM)

### Simple Inference

```python
# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Prepare inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode output
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text[0])
```

## Multi-Image Inference

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the differences between these images?"},
        ],
    }
]

# Same processing as single image
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

## Video Understanding

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Same processing
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

## Dynamic Resolution Control

### Per-Image Budget

```python
from qwen_vl_utils import process_vision_info

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "path/to/image.jpg",
                "min_pixels": 256 * 32 * 32,    # 256 tokens minimum
                "max_pixels": 1280 * 32 * 32,   # 1280 tokens maximum
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Process with qwen-vl-utils
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True)

# Process (do_resize=False to avoid duplicate resizing)
inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")
inputs = inputs.to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

### Video FPS Control

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "path/to/video.mp4",
                "fps": 4,  # Sample at 4 frames per second
            },
            {"type": "text", "text": "What happens in this video?"},
        ],
    }
]

# Process video with fps
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

# Split videos and metadata (Qwen3-VL specific)
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

# Process
inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, do_resize=False, return_tensors="pt", **video_kwargs)
inputs = inputs.to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

## Flash Attention 2

```python
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    device_map="auto"
)

# 20-30% speedup on supported hardware
```

## Thinking Mode

### Enable Thinking

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/chart.jpg"},
            {"type": "text", "text": "/think\nAnalyze this chart in detail."},
        ],
    }
]

# Process normally
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

# Generate with larger budget for thinking
generated_ids = model.generate(**inputs, max_new_tokens=2048)  # Larger for thinking
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
# Output includes <think> tags with reasoning process
```

## Common Parameters

### Generation Parameters

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,        # Maximum tokens to generate
    temperature=0.7,           # Sampling temperature (0.0 = greedy)
    top_p=0.8,                # Nucleus sampling
    top_k=20,                 # Top-k sampling
    do_sample=True,           # Enable sampling (vs greedy)
    repetition_penalty=1.0,   # Penalize repetition
)
```

### Processor Parameters

```python
# Token budget control (via processor)
processor.image_processor.size = {
    "longest_edge": 1280 * 32 * 32,  # Max pixels per image
    "shortest_edge": 256 * 32 * 32,   # Min pixels per image
}

processor.video_processor.size = {
    "longest_edge": 16384 * 32 * 32,  # Max total pixels for video
    "shortest_edge": 256 * 32 * 32,    # Min total pixels for video
}
```

## Supported Input Formats

### Images

```python
# Local file path
{"type": "image", "image": "file:///path/to/image.jpg"}
{"type": "image", "image": "/path/to/image.jpg"}

# HTTP(S) URL
{"type": "image", "image": "https://example.com/image.jpg"}

# Base64 encoded
{"type": "image", "image": "data:image/jpeg;base64,/9j/4AAQ..."}

# PIL Image object
{"type": "image", "image": PIL.Image.open("path.jpg")}
```

### Videos

```python
# Local file path
{"type": "video", "video": "file:///path/to/video.mp4"}

# HTTP(S) URL
{"type": "video", "video": "https://example.com/video.mp4"}

# List of frames (images)
{"type": "video", "video": ["frame1.jpg", "frame2.jpg", ...], "sample_fps": 1}
```

## Troubleshooting

### Out of Memory

```python
# Use smaller model
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",  # Smaller
    device_map="auto"
)

# Or reduce image resolution
processor.image_processor.size = {"longest_edge": 512 * 32 * 32}
```

### Slow Inference

```python
# Enable Flash Attention 2
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Use FP8 checkpoint (H100+)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct-FP8",
    device_map="auto"
)
```

## Next Steps

- [01-huggingface-integration.md](01-huggingface-integration.md) - Detailed HF usage
- [02-dynamic-resolution.md](02-dynamic-resolution.md) - Resolution control
- [03-vllm-deployment.md](03-vllm-deployment.md) - Production serving
- [examples/00-basic-inference.md](../examples/00-basic-inference.md) - More examples

## Related Topics

- [architecture/00-overview.md](../architecture/00-overview.md) - System architecture
- [concepts/03-dynamic-resolution.md](../concepts/03-dynamic-resolution.md) - Smart resize
- [codebase/01-vision-process.md](../codebase/01-vision-process.md) - Preprocessing code
