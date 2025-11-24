# DeepSeek-OCR - Source Codebase Index

**Repository**: https://github.com/deepseek-ai/DeepSeek-OCR
**Purpose**: Vision-language OCR with SAM+CLIP serial design
**Key Innovation**: 16× optical compression ratio

## Directory Structure

```
06-DeepSeek-OCR/
├── INDEX.md                 # This file
├── README.md                # Original documentation
├── DeepSeek_OCR_paper.pdf   # Technical paper ⭐
├── LICENSE
├── requirements.txt
│
├── DeepSeek-OCR-master/     # Main implementation ⭐
│   ├── models/                 # Model definitions
│   │   ├── sam_encoder.py         # SAM visual encoder
│   │   ├── clip_encoder.py        # CLIP semantic encoder
│   │   └── ocr_decoder.py         # Text decoder
│   ├── data/                   # Data processing
│   ├── configs/                # Training configs
│   └── scripts/                # Training/eval scripts
│
└── assets/                  # Demo images
```

## Key Concepts

### SAM+CLIP Serial Architecture
1. **SAM encoder**: Segment Anything Model for visual features
2. **CLIP encoder**: Semantic understanding
3. **Serial fusion**: SAM → CLIP (not parallel)
4. **16× compression**: Efficient token reduction

### Performance
- **State-of-the-art OCR**: Multi-language support
- **Document understanding**: Tables, charts, handwriting
- **Efficient inference**: Reduced visual tokens

## Key Files

| File | Description | Keywords |
|------|-------------|----------|
| `models/sam_encoder.py` | Visual feature extraction | SAM, ViT |
| `models/clip_encoder.py` | Semantic encoding | CLIP, contrastive |
| `models/ocr_decoder.py` | Text generation | autoregressive |
| `DeepSeek_OCR_paper.pdf` | Technical report | architecture details |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Inference
python inference.py --image test.png --model deepseek-ocr
```

## Cross-References

**DeepSeek efficiency**: `karpathy/codebases/02-karpathy-on-deepseek-efficiency.md`
**Related VLMs**: `08-DeepSeek-VL2`, `14-Ovis-2-5`
**Dedicated oracle**: `deepseek-ocr-oracle` (detailed analysis)
