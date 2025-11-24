# Phase P2: Multimodal Pre-training

**Category**: Training
**Related**: [00-overview.md](00-overview.md)

## Goal

Core visual understanding: OCR, grounding, spatial reasoning.

## Configuration

**Trainable**: ALL modules (full-parameter)
**Frozen**: Nothing

## Data

**Composition**:
- OCR: 70% (document + scene text)
- Grounding: 15% (object detection)
- Captions: 15% (general)

**Size**: ~500M examples

## Resolution

**Range**: 448²-1792² (2× expansion from P1)
**RoPE**: Enabled

## Training

```yaml
epochs: 1
learning_rate: 3e-5
batch_size: 640
```

## Output

Strong base multimodal model with OCR and spatial understanding.
