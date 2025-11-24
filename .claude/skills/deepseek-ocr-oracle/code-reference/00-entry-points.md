# Entry Points

## 1. CLI Interface

**File**: `run_dpsk_ocr_image.py`

```bash
python run_dpsk_ocr_image.py --image doc.jpg --mode base
```

**Main function** (lines 260-304):
```python
def main():
    # Parse args
    args = parse_args()

    # Load model
    model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR")

    # Infer
    result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=[args.image],
                        base_size=args.base_size, image_size=args.image_size)

    print(result[0])
```

## 2. Python API

**File**: `deepseek_ocr.py:409-503`

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR")

result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"])
```

**infer() method** handles:
- Image loading
- Preprocessing
- Forward pass
- Generation
- Post-processing

## 3. Direct forward()

**File**: `deepseek_ocr.py:321-407`

```python
# For custom pipelines
vision_tokens = model.forward(images, ...)
```

**See Also**:
- [inference-flow.md](inference-flow.md) - Complete execution trace
- [../usage/quick-start.md](../usage/quick-start.md) - Usage examples
