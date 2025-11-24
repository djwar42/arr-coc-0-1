# DeepSeek-OCR Complete File Index

**Codebase Location**: `RESEARCH/DeepSeekOCR/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`

## Core Model Files

### `deepseek_ocr.py`
**The main model class** - Contains complete DeepSeek-OCR implementation

**Key Components**:
- Lines 61-106: `get_num_image_tokens()` - Dynamic token budget calculation
- Lines 158-201: `__init__()` - Model initialization
- Lines 321-407: `forward()` - Complete forward pass
  - Line 394-397: SAM processing
  - Line 399-402: CLIP processing
  - Line 404-407: Feature fusion
- Lines 409-503: `infer()` - High-level inference API

**What it does**: Orchestrates entire pipeline from image to text output

---

### `config.py`
**Configuration management**

**Key Settings**:
- Lines 5-15: Model paths and checkpoint locations
- Lines 17-30: Resolution mode definitions
- Lines 32-43: Generation parameters

**What it does**: Centralizes all configurable parameters

---

## Image Processing

### `process/image_process.py`
**Image preprocessing and tokenization**

**Key Functions**:
- Lines 45-120: `find_best_resize()` - Dynamic tiling algorithm
- Lines 122-180: `process_images()` - Main preprocessing pipeline
- Lines 182-250: `create_vision_text_mask()` - Token mask generation
- Lines 252-340: `pad_images()` - Padding and normalization
- Lines 342-420: `calculate_aspect_ratio()` - Aspect ratio matching

**What it does**: Converts raw images into model-ready tensor sequences

**Key insight**: Not uniform grid - finds best aspect ratio match!

---

## DeepEncoder Components

### `deepencoder/sam_vary_sdpa.py`
**SAM (Segment Anything Model) implementation**

**Key Components**:
- Lines 166-183: Compression layers (neck + net_2 + net_3)
  ```python
  self.neck = nn.Sequential(
      nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),  # 768→256
      LayerNorm2d(out_chans),
      nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
  )
  self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)  # spatial /2
  self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False) # spatial /2
  ```
- Lines 300-350: `forward()` - SAM forward pass
- Lines 400-450: Window attention implementation

**What it does**: Processes high-res patches (4096 tokens) → compresses to 256 patches (16×)

**Key design**: Window attention (cheap) + compression layers

---

### `deepencoder/clip_sdpa.py`
**CLIP vision encoder implementation**

**Key Components**:
- Lines 200-250: `__init__()` - CLIP initialization
- Lines 300-380: `forward()` - CLIP forward pass
- Lines 400-450: Global attention layers

**What it does**: Takes SAM's compressed output, adds semantic understanding

**Key design**: Global attention (expensive but on compressed 256 patches)

---

### `deepencoder/build_linear.py`
**Feature fusion and projection**

**Key Components**:
- Lines 40-80: `MlpProjector` class
  ```python
  class MlpProjector(nn.Module):
      def __init__(self, in_features=2048, out_features=1280):
          # Linear projection from concatenated SAM+CLIP to language space
  ```
- Lines 100-150: `forward()` - Concatenate SAM + CLIP → project

**What it does**: Fuses SAM (fine-grained) + CLIP (semantic) → language space

**Formula**: [B, 256, 1024] + [B, 256, 1024] → [B, 256, 2048] → [B, 256, 1280]

---

## Utility Files

### `run_dpsk_ocr_image.py`
**Command-line inference script**

**Key Components**:
- Lines 50-100: Argument parsing
- Lines 120-180: Model loading
- Lines 200-250: Image processing
- Lines 260-304: Inference loop and output

**What it does**: Provides CLI interface for batch inference

**Usage**: `python run_dpsk_ocr_image.py --image doc.jpg --mode base`

---

### `process/image_utils.py`
**Image manipulation utilities**

**Key Functions**:
- `resize_image()` - Smart resizing
- `pad_to_square()` - Padding
- `normalize()` - ImageNet normalization
- `tile_image()` - Gundam mode tiling

**What it does**: Low-level image operations

---

## Configuration Files

### `deepseek_ocr_config.json`
**Model configuration metadata**

```json
{
  "model_type": "deepseek_ocr",
  "vision_encoder": {
    "sam": { "embed_dim": 768, ... },
    "clip": { "embed_dim": 1024, ... }
  },
  "projector": { "in_features": 2048, "out_features": 1280 },
  "language_model": { "type": "deepseek_moe", ... }
}
```

---

## Documentation Files

### `RESEARCH/DeepSeekOCR/ARCHITECTURE.md`
Complete architecture documentation (723 lines)
- System design
- Component breakdown
- Compression mechanism
- Resolution modes

### `RESEARCH/DeepSeekOCR/TRAINING.md`
Training pipeline documentation (1321 lines)
- 3-stage training process
- Data engineering
- Hyperparameters
- Infrastructure

### `RESEARCH/DeepSeekOCR/HF.md`
HuggingFace integration guide (1377 lines)
- Quick start
- Fine-tuning
- vLLM deployment
- Gradio demo

### `RESEARCH/DeepSeekOCR/CODE_FLOW.md`
Complete execution trace (1526 lines)
- Step-by-step code flow
- Tensor shapes at each stage
- Performance characteristics

---

## Execution Flow Summary

```
Entry Point: run_dpsk_ocr_image.py or model.infer()
    ↓
Image Preprocessing: process/image_process.py
    ↓
Model Forward Pass: deepseek_ocr.py:forward()
    ├── SAM: deepencoder/sam_vary_sdpa.py
    ├── CLIP: deepencoder/clip_sdpa.py
    └── Projector: deepencoder/build_linear.py
    ↓
Language Model Decoding: (proprietary, not in OSS)
    ↓
Output: Text result
```

---

## File Size Reference

| File | Lines | Purpose |
|------|-------|---------|
| `deepseek_ocr.py` | 583 | Main model class |
| `sam_vary_sdpa.py` | 528 | SAM implementation |
| `clip_sdpa.py` | 505 | CLIP implementation |
| `image_process.py` | 503 | Image preprocessing |
| `run_dpsk_ocr_image.py` | 304 | CLI interface |
| `build_linear.py` | 175 | Projector |
| `config.py` | 43 | Configuration |

---

## Quick File Lookup

**Want to understand...**

**Architecture?**
→ `deepseek_ocr.py` (main class)
→ `deepencoder/sam_vary_sdpa.py` (SAM)
→ `deepencoder/clip_sdpa.py` (CLIP)

**Preprocessing?**
→ `process/image_process.py` (complete pipeline)

**Token calculation?**
→ `deepseek_ocr.py:61-106`

**Forward pass?**
→ `deepseek_ocr.py:321-407`

**Compression mechanism?**
→ `deepencoder/sam_vary_sdpa.py:166-183`

**Inference API?**
→ `deepseek_ocr.py:409-503`

**CLI usage?**
→ `run_dpsk_ocr_image.py`

---

## Directory Structure

```
DeepSeek-OCR-vllm/
├── deepseek_ocr.py              # Main model
├── config.py                     # Configuration
├── run_dpsk_ocr_image.py        # CLI interface
│
├── deepencoder/                  # Vision encoder
│   ├── sam_vary_sdpa.py         # SAM implementation
│   ├── clip_sdpa.py             # CLIP implementation
│   └── build_linear.py          # Projector
│
├── process/                      # Image processing
│   ├── image_process.py         # Main preprocessing
│   └── image_utils.py           # Utilities
│
└── configs/                      # Configuration files
    └── deepseek_ocr_config.json
```

---

## See Also

- [architecture/overview.md](architecture/overview.md) - System architecture
- [code-reference/inference-flow.md](code-reference/inference-flow.md) - Complete execution trace
- [usage/quick-start.md](usage/quick-start.md) - How to use
