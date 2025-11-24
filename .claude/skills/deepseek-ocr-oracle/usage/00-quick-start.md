# Quick Start Guide

## 3-Line Inference

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR")

result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["document.jpg"])
print(result[0])
```

**That's it!** DeepSeek-OCR will automatically:
- Choose optimal resolution
- Process the image
- Extract text
- Return OCR result

## Installation

```bash
pip install transformers torch pillow
```

**Requirements**:
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (recommended) or CPU
- 16GB+ RAM for base mode

## Basic Usage Patterns

### 1. Free OCR (Extract All Text)

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR.",
    images=["document.jpg"],
    base_size=1024,      # Base mode (273 tokens)
    image_size=1024,
    crop_mode=False
)
```

### 2. Selective OCR (Specific Region)

```python
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Extract the table in the upper left.",
    images=["document.jpg"],
    base_size=1024,
    image_size=1024,
    crop_mode=False
)
```

### 3. Markdown Conversion

```python
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    images=["document.jpg"],
    base_size=1024,
    image_size=1024,
    crop_mode=False
)
```

### 4. Chart/Figure Parsing

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nChart Parsing.",
    images=["chart.png"],
    base_size=640,       # Small mode (111 tokens) - sufficient for charts
    image_size=640,
    crop_mode=False
)
```

### 5. Mathematical Formula Extraction

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nExtract all formulas in LaTeX format.",
    images=["textbook.jpg"],
    base_size=1280,      # Large mode (421 tokens) - for complex math
    image_size=1280,
    crop_mode=False
)
```

## Resolution Mode Selection

**Quick Guide**:
- **Tiny** (512×512, ~73 tokens): Slides, simple images
- **Small** (640×640, ~111 tokens): Books, reports, charts
- **Base** (1024×1024, ~273 tokens): Standard documents, default choice
- **Large** (1280×1280, ~421 tokens): High-detail content, dense text
- **Gundam** (dynamic, variable): Ultra-high-resolution, multi-page

**Rule of thumb**: Start with Base, go up if quality insufficient, down if too slow.

## Common Parameters

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR.",
    images=["document.jpg"],

    # Resolution settings
    base_size=1024,              # Global view resolution
    image_size=1024,             # Local view resolution
    crop_mode=False,             # Enable tiling (Gundam mode)

    # Generation settings
    max_new_tokens=2048,         # Max output length
    temperature=0.0,             # Deterministic (set >0 for creative)
    top_p=1.0,                   # Nucleus sampling
    repetition_penalty=1.05,     # Prevent repetition

    # Device
    device="cuda"                # or "cpu"
)
```

## GPU Memory Requirements

| Mode | Min VRAM | Recommended | Batch Size |
|------|----------|-------------|------------|
| Tiny | 8GB | 12GB | 4 |
| Small | 10GB | 16GB | 4 |
| Base | 12GB | 20GB | 2 |
| Large | 16GB | 24GB | 1 |
| Gundam | 20GB+ | 32GB+ | 1 |

**OOM errors?** → Reduce batch size, enable gradient checkpointing, or use smaller mode.

## Batch Processing

```python
images = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]

results = model.infer(
    tokenizer,
    prompt=["<image>\nFree OCR."] * len(images),  # One prompt per image
    images=images,
    base_size=1024,
    image_size=1024,
    crop_mode=False
)

for i, result in enumerate(results):
    print(f"Document {i+1}:")
    print(result)
    print("-" * 50)
```

## Performance Tips

**1. Use vLLM for Production**
```python
# 10-20× faster than HuggingFace Transformers
# See usage/vllm-deployment.md for setup
```

**2. Enable Flash Attention**
```python
# Automatically enabled if flash-attn installed
pip install flash-attn
```

**3. Mixed Precision**
```python
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # or torch.float16
)
```

**4. Choose Appropriate Mode**
- Don't use Large for simple documents
- Tiny is 8× faster than Large
- Quality vs speed tradeoff

## Error Handling

```python
try:
    result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"])
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM! Try smaller resolution mode")
    elif "CUDA" in str(e):
        print("CUDA error! Check GPU availability")
    else:
        raise

except FileNotFoundError:
    print("Image file not found")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Output Processing

```python
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"])

# Result is a list (one item per image)
ocr_text = result[0]

# Clean up
cleaned = ocr_text.strip()

# Split into lines
lines = cleaned.split('\n')

# Remove empty lines
lines = [line for line in lines if line.strip()]

# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)
```

## Next Steps

- **Resolution selection** → [resolution-selection.md](resolution-selection.md)
- **Fine-tuning** → [fine-tuning.md](fine-tuning.md)
- **Production deployment** → [vllm-deployment.md](vllm-deployment.md)
- **Complete HF guide** → [huggingface.md](huggingface.md)

## File References

**Implementation**:
- Model loading: HuggingFace AutoModel API
- Inference: `deepseek_ocr.py:infer()` method
- Preprocessing: `process/image_process.py`
- Token calculation: `deepseek_ocr.py:get_num_image_tokens()`

**See Also**:
- [../architecture/overview.md](../architecture/overview.md) - How it works
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - Mode details
- [../code-reference/inference-flow.md](../code-reference/inference-flow.md) - Complete execution trace
