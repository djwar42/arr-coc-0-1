# Knowledge Drop: SAM 2 Overview for Video Segmentation

**Date**: 2025-11-20 21:50
**Type**: Knowledge Acquisition
**Part**: 13

## Summary

Created comprehensive overview of SAM 2 (Segment Anything Model 2) covering Meta AI's unified image and video segmentation model released August 2024.

## File Created

**Path**: `sam-general/12-sam2-overview-video.md`
**Lines**: 809

## Sections Covered

1. **SAM 2 Introduction** (~130 lines) - Unified image+video architecture, streaming memory design
2. **Key Improvements over SAM 1** (~140 lines) - 6x faster, Hiera encoder, memory attention
3. **Video Segmentation Paradigm** (~120 lines) - Stateful inference, propagation, multi-object tracking
4. **Real-Time Performance** (~100 lines) - 44 FPS on A100, model size trade-offs
5. **SA-V Dataset** (~90 lines) - Largest video segmentation dataset, data engine
6. **Applications** (~140 lines) - Video editing, AR/VR, autonomous vehicles, annotation
7. **Code Examples** (~120 lines) - Video/image segmentation, visualization
8. **ARR-COC Integration** (~70 lines) - Temporal attention training, feature extraction

## Key Technical Details

### Performance
- **Video**: 44 FPS on A100 GPU (real-time)
- **Image**: 6x faster than SAM 1
- **Interactions**: 3x fewer than prior methods

### Architecture Innovations
- Hiera (Hierarchical) Vision Transformer replaces ViT-H
- Streaming memory architecture for real-time processing
- Memory attention module for temporal consistency

### Model Variants
| Model | Parameters | Speed | Size |
|-------|------------|-------|------|
| Tiny | 38.9M | ~47 FPS | 154 MB |
| Small | ~50M | ~43 FPS | 185 MB |
| Base+ | ~80M | ~35 FPS | 310 MB |
| Large | 224.4M | ~30 FPS | 900 MB |

## Sources Used

### Primary Sources
- SAM_STUDY_GENERAL.md - Source document with comprehensive SAM research
- arXiv:2408.00714 - Original SAM 2 paper

### Web Research
- Meta AI SAM 2 announcement (about.fb.com)
- Roboflow SAM 2 video segmentation guide
- GitHub repository documentation

## ARR-COC Integration Opportunities

1. **Temporal attention training** - Use SAM 2's streaming memory pattern
2. **Video dataset annotation** - Semi-automatic labeling
3. **Feature extraction** - Object-centric video representations
4. **Training curriculum** - Progressive difficulty with video understanding

## Citation Format

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## Notes

- SAM 2 is a foundation model extending SAM 1's image capabilities to video
- Streaming architecture enables real-time processing without full video in memory
- Known limitations: shot changes, crowded scenes, long occlusions, fine details
- SA-V dataset is the largest video segmentation dataset to date
- Memory attention module is key innovation for temporal consistency
