# Ovis 2.5 Quickstart Guide

**Category**: Usage
**Related**: [01-huggingface-integration.md](01-huggingface-integration.md), [../examples/00-basic-inference.md](../examples/00-basic-inference.md)
**Time**: 5 minutes to first inference

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- 8GB+ GPU memory (CPU also supported)

### Install from HuggingFace

```bash
pip install transformers torch pillow
```

That's it! Ovis uses `trust_remote_code=True` to load directly from HuggingFace Hub.

## Basic Inference (3 Lines)

```python
from transformers import AutoModelForCausalLM
from PIL import Image

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    torch_dtype="auto",
    trust_remote_code=True
).cuda()

# Load image
image = Image.open("your_image.jpg")

# Generate
response, _, _ = model.chat(
    prompt="<image>\nDescribe this image in detail.",
    images=[image],
    max_new_tokens=1024
)

print(response)
```

**Output**: Detailed image description with high accuracy.

## Model Variants

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| **Ovis2.5-2B** | 2B | ~8GB | Fast | Edge, mobile, quick tests |
| **Ovis2.5-9B** | 9B | ~20GB | Moderate | Production, best quality |

```python
# For 2B model (faster, less memory)
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-2B",
    torch_dtype="auto",
    trust_remote_code=True
).cuda()
```

## Common Use Cases

### 1. Image Description

```python
response, _, _ = model.chat(
    prompt="<image>\nDescribe this image.",
    images=[image]
)
```

### 2. Document OCR

```python
response, _, _ = model.chat(
    prompt="<image>\nExtract all text from this document.",
    images=[document_image]
)
```

### 3. Chart Analysis

```python
response, _, _ = model.chat(
    prompt="<image>\nAnalyze the trends in this chart.",
    images=[chart_image]
)
```

### 4. Visual Question Answering

```python
response, _, _ = model.chat(
    prompt="<image>\nHow many people are in this photo?",
    images=[photo]
)
```

### 5. Table Extraction

```python
response, _, _ = model.chat(
    prompt="<image>\nConvert this table to markdown format.",
    images=[table_image]
)
```

## Key Parameters

### `chat()` Method

```python
response, history, generation_info = model.chat(
    prompt: str,              # Text prompt with <image> token
    images: List[Image],      # List of PIL Images
    videos: List = None,      # Optional: video frames
    max_new_tokens: int = 1024,        # Max output length
    temperature: float = 0.7,          # Sampling temperature (0=greedy)
    top_p: float = 0.9,                # Nucleus sampling
    do_sample: bool = True,            # Enable sampling
    enable_thinking: bool = False,     # Enable thinking mode
    thinking_budget: int = 2048        # Tokens for thinking phase
)
```

### Resolution Control

```python
# Default: Auto-select resolution
response, _, _ = model.chat(prompt, images=[image])

# Explicit resolution (for very large images)
response, _, _ = model.chat(
    prompt=prompt,
    images=[image],
    # Note: Resolution handled automatically by smart_resize
)
```

## CPU Inference (No GPU)

```python
# Load on CPU
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-2B",  # Use 2B for CPU
    torch_dtype=torch.float32,
    trust_remote_code=True
)  # Don't call .cuda()

# Inference is slower but works
response, _, _ = model.chat(
    prompt="<image>\nWhat's in this image?",
    images=[image],
    max_new_tokens=512  # Reduce for speed
)
```

**Performance**: ~1-2 tokens/sec on CPU (vs ~20-30 on GPU)

## Mixed Precision (Recommended)

```python
# Use bfloat16 for faster inference (Ampere+ GPUs)
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

# Or float16 for older GPUs
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
```

**Benefits**:
- 2× faster inference
- 2× less VRAM
- Minimal accuracy loss

## Batch Inference

```python
# Multiple images, single prompt
images = [Image.open(f"image_{i}.jpg") for i in range(5)]

for image in images:
    response, _, _ = model.chat(
        prompt="<image>\nDescribe this.",
        images=[image]
    )
    print(response)
    print("---")
```

**Note**: Ovis processes one image at a time. For true batching, use vLLM (see advanced usage).

## Troubleshooting

### ImportError: trust_remote_code

**Error**: `Can't load model with trust_remote_code=False`

**Solution**: Always use `trust_remote_code=True`:
```python
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True  # Required!
)
```

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use 2B model instead of 9B
2. Enable mixed precision (bfloat16/float16)
3. Reduce max_new_tokens
4. Clear cache: `torch.cuda.empty_cache()`

```python
# Memory-efficient setup
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-2B",          # Smaller model
    torch_dtype=torch.bfloat16,    # Half precision
    trust_remote_code=True
).cuda()

response, _, _ = model.chat(
    prompt=prompt,
    images=[image],
    max_new_tokens=512              # Limit output
)
```

### Slow Inference

**Issue**: <1 token/sec

**Solutions**:
1. Use mixed precision (bfloat16)
2. Ensure GPU is actually used: Check `model.device`
3. Use smaller model (2B)
4. For production: Use vLLM backend

```python
# Fast setup
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-2B",
    torch_dtype=torch.bfloat16,  # Faster
    trust_remote_code=True
).cuda()

# Verify GPU usage
print(f"Model device: {next(model.parameters()).device}")  # Should show cuda:0
```

## Next Steps

### Learn More
- [01-huggingface-integration.md](01-huggingface-integration.md) - Full HF Hub integration
- [02-advanced-features.md](02-advanced-features.md) - Multi-image, video, thinking mode
- [03-fine-tuning.md](03-fine-tuning.md) - Custom training

### Examples
- [../examples/00-basic-inference.md](../examples/00-basic-inference.md) - More examples
- [../examples/01-thinking-mode.md](../examples/01-thinking-mode.md) - Deep reasoning
- [../examples/02-multi-image.md](../examples/02-multi-image.md) - Multiple images

### Architecture
- [../architecture/00-overview.md](../architecture/00-overview.md) - How Ovis works
- [../architecture/03-visual-embedding-table.md](../architecture/03-visual-embedding-table.md) - VET explained
- [../concepts/00-structural-alignment.md](../concepts/00-structural-alignment.md) - Core innovation

## Complete Minimal Example

Save as `ovis_demo.py`:

```python
from transformers import AutoModelForCausalLM
from PIL import Image
import torch

def main():
    # Load model
    print("Loading Ovis2.5-2B...")
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis2.5-2B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    print("Model loaded!")

    # Load image
    image = Image.open("test_image.jpg")

    # Generate
    print("\nGenerating response...")
    response, _, _ = model.chat(
        prompt="<image>\nDescribe this image in detail.",
        images=[image],
        max_new_tokens=512
    )

    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()
```

Run:
```bash
python ovis_demo.py
```

**You're now ready to use Ovis 2.5!** 🎉

## Quick Reference Card

```
┌──────────────────────────────────────┐
│ OVIS 2.5 QUICK REFERENCE             │
├──────────────────────────────────────┤
│ Load:                                │
│   model = AutoModelForCausalLM...    │
│   trust_remote_code=True             │
│                                      │
│ Inference:                           │
│   response, _, _ = model.chat(       │
│     prompt="<image>\n...",           │
│     images=[image]                   │
│   )                                  │
│                                      │
│ Speed: bfloat16                      │
│ Memory: Use 2B model                 │
│ CPU: float32, no .cuda()             │
└──────────────────────────────────────┘
```

## Related Topics

- [../architecture/00-overview.md](../architecture/00-overview.md) - System overview
- [../training/00-overview.md](../training/00-overview.md) - How Ovis is trained
- [../concepts/00-structural-alignment.md](../concepts/00-structural-alignment.md) - Why VET matters

## Code References

**Model Class**: `ovis/model/modeling_ovis.py` - `Ovis` class
**Chat Method**: `ovis/model/modeling_ovis.py` - `chat()` method
**Demo Scripts**: `ovis/serve/infer_basic_demo.py`
