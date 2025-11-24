# HuggingFace Integration

**Full documentation**: `RESEARCH/DeepSeekOCR/HF.md` (1377 lines)

## Quick Install

```bash
pip install transformers torch pillow
```

## Basic Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR")

result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"])
```

## Model Hub

**Location**: `huggingface.co/deepseek-ai/DeepSeek-OCR`

**Statistics**:
- 1.96k likes
- 841k downloads
- 13 files (23.6GB)

## Key Files

- `pytorch_model.bin` - Model weights
- `config.json` - Architecture config
- `tokenizer.json` - Tokenizer
- `preprocessing_config.json` - Image preprocessing

## Custom Code

`modeling_deepseek_ocr.py` in the repo - uses `trust_remote_code=True`

## Advanced Options

**Mixed Precision**:
```python
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR",
                                 trust_remote_code=True,
                                 torch_dtype=torch.bfloat16)
```

**Device Selection**:
```python
model = model.to("cuda")  # or "cpu"
```

**See HF.md** for:
- Fine-tuning guide
- vLLM deployment
- Gradio demo
- Batch processing
- Quantization

**File Reference**: `RESEARCH/DeepSeekOCR/HF.md`
