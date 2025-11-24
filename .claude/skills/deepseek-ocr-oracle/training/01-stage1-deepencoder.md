# Stage 1: DeepEncoder Pre-training

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 26-178

## Objective

Train DeepEncoder (380M params) to extract vision features decodable by a language model.

## Architecture

```
DeepEncoder (SAM + CLIP + Projector)
    ↓
OPT-IML (compact LM for training)
    ↓
Next Token Prediction
```

**Why compact LM?**: Faster training, vision encoder is the focus

## Data (130M samples × 2 epochs)

1. **OCR 1.0** (53M):
   - 30M document pages
   - 20M scene images
   - 3M Word documents

2. **OCR 2.0** (16M):
   - 10M charts
   - 5M chemical formulas
   - 1M geometry figures

3. **General Vision** (100M):
   - LAION image-caption pairs

## Training Details

- **Duration**: ~1 week on 160 A100 GPUs
- **Batch size**: 1280
- **Learning rate**: 5e-5 with cosine annealing
- **Precision**: bfloat16
- **Throughput**: ~70B tokens/day

## Multi-Resolution Training

All modes trained together with weighted sampling:
- Tiny: 15%
- Small: 25%
- Base: 35%
- Large: 20%
- Gundam: 5%

**Result**: Single encoder handles all resolutions!
