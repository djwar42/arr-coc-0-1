# Stage 2: Full VLM Training

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 180-450

## Objective

Train complete DeepSeek-OCR with production-grade MoE decoder.

## Architecture

```
DeepEncoder (380M, initialized from Stage 1)
    ↓
DeepSeek-3B-MoE (570M active, 3B total)
    ↓
Vision-Language Understanding
```

## Pipeline Parallelism (4-stage)

```
Stage 1: DeepEncoder (SAM + CLIP)
Stage 2: Projector
Stage 3: DeepSeek-MoE (layers 0-5)
Stage 4: DeepSeek-MoE (layers 6-11 + head)
```

**Benefit**: Distribute memory across 4 GPUs per pipeline

## Training

- **Duration**: ~1 week on 160 A100 GPUs
- **Data**: Same as Stage 1 (130M × 2 epochs)
- **Batch size**: 1024 (slightly smaller due to larger model)
- **Learning rate**: 2e-5 (lower than Stage 1)

## Freezing Strategy

**DeepEncoder**: Fine-tuned (not frozen)
**LLM**: Fully trained

**Rationale**: Vision encoder needs adaptation to MoE decoder

## Result

Production-ready DeepSeek-OCR model handling all resolution modes!
