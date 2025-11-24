# Knowledge Drop: Hiera Image Encoder

**Date**: 2025-11-20 14:45 UTC
**PART**: 14
**File Created**: sam-general/13-hiera-image-encoder.md
**Lines**: 502

## Summary

Created comprehensive documentation on the Hiera (Hierarchical) image encoder used in SAM 2.

## Key Concepts Documented

### Core Innovation
- Hiera replaces ViT-H from SAM 1 with a simplified hierarchical transformer
- Achieves **6× speedup** while maintaining/improving accuracy
- Key insight: MAE pre-training eliminates need for complex architectural components

### Architecture Details
- 4-stage hierarchical design (high res → low res)
- Standard attention (no bells-and-whistles like shifted windows, relative position)
- Token reordering for memory efficiency
- Model sizes from Tiny (6M params) to Huge (672M params)

### Speed Improvements
- SAM 2 with Hiera: 6× faster than SAM 1 with ViT-H
- Enables real-time video at 44 FPS on A100
- Per-frame encoding: ~17ms vs ~100ms for ViT-H

### Multi-Scale Features
- Natural feature pyramid from hierarchical design
- Stages produce 256×256 → 128×128 → 64×64 → 32×32 features
- Integrates with FPN neck for segmentation

### MAE Pre-Training
- Critical enabler for architectural simplification
- 60% mask ratio for images, 90% for video
- Normalized pixel targets for loss computation

### Memory System Integration
- Hiera's speed enables streaming memory for video
- Multi-scale features used in memory attention
- FIFO memory bank (6 recent frames)

## Sources Used

- arXiv:2306.00989 (Hiera paper, ICML 2023 Oral)
- arXiv:2408.00714 (SAM 2 paper)
- GitHub facebookresearch/hiera
- GitHub facebookresearch/sam2
- Hugging Face documentation
- SAM_STUDY_GENERAL.md (source document)

## ARR-COC Integration

- Feature extraction for training
- Fine-tuning strategies (freeze early stages)
- Gradient checkpointing for memory efficiency
- Speed benefits for training and inference
