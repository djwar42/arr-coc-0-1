# Training Overview

**Full documentation**: `RESEARCH/DeepSeekOCR/TRAINING.md` (1321 lines)

## 3-Stage Pipeline

```
Stage 1: DeepEncoder Pre-training (1 week, 160 A100s)
    ↓
Stage 2: Full VLM Training (1 week, 160 A100s)
    ↓
Stage 3: Gundam-Master Fine-tuning (3 days, 160 A100s)
```

**Total**: ~17 days, $260k compute cost

## Stage 1: DeepEncoder Pre-training

**Goal**: Train vision encoder independently

**Components**:
- DeepEncoder (SAM + CLIP + Projector)
- Compact LM (OPT-IML)

**Data**: 130M samples × 2 epochs
- OCR 1.0: Document + scene OCR
- OCR 2.0: Charts, formulas, geometry
- General: LAION image-caption pairs

**Hyperparameters**:
- Batch size: 1280
- Learning rate: 5e-5
- Optimizer: AdamW
- Precision: bfloat16

## Stage 2: Full VLM Training

**Goal**: Train complete system with production LM

**Components**:
- DeepEncoder (from Stage 1)
- DeepSeek-3B-MoE

**Multi-resolution training**: All modes (Tiny/Small/Base/Large/Gundam) trained simultaneously

**Pipeline parallelism**: 4-stage split across GPUs

## Stage 3: Gundam-Master

**Goal**: High-resolution fine-tuning

**Focus**: Ultra-high-res documents (Gundam mode)

**Duration**: 3 days (shorter than main training)

## Key Training Insights

1. **Multi-resolution from start**: Single model handles all modes
2. **Progressive training**: Vision encoder → Full VLM → High-res
3. **Data diversity**: OCR + charts + general vision
4. **Efficiency**: Pipeline parallelism + mixed precision

**See TRAINING.md** for complete details on data, hyperparameters, and infrastructure!

**File Reference**: `RESEARCH/DeepSeekOCR/TRAINING.md`
