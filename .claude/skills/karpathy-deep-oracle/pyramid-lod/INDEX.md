# Pyramid LOD & Hierarchical Vision - Overview

**Total**: 11 files covering pyramid-based vision processing

## What This Covers

Pyramid and Level-of-Detail (LOD) systems for efficient vision processing:
- Foveated attention (eye-tracking, VR/AR)
- Multi-scale feature hierarchies
- Token budget optimization
- ARR-COC integration

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `01-foveated-gaze-pyramids.md` | Eye-tracking, VR/AR foveation | gaze, fovea, peripheral |
| `02-neural-texture-compression-pyramids.md` | Learned pyramids, compression | neural codec, learned LOD |
| `03-attention-driven-pyramid-pruning.md` | ARR-COC integration ⭐ | relevance, token pruning |
| `04-gigapixel-tiled-pyramids.md` | HIPT, satellite imagery | gigapixel, pathology |
| `05-3d-volumetric-pyramids-video.md` | Spatiotemporal pyramids | video, 3D |
| `06-differentiable-pyramid-operators.md` | End-to-end learning | differentiable, gradients |
| `07-hybrid-cpu-gpu-pyramid.md` | Heterogeneous computing | edge, mobile |
| `08-super-resolution-pyramid-guidance.md` | Coarse-to-fine upsampling | super-res, Laplacian |
| `09-cross-modal-pyramids.md` | Text-image-audio hierarchies | multimodal |
| `10-quantization-aware-pyramid-storage.md` | INT8/FP16 mixed-precision | quantization |

## Quick Start

**For ARR-COC project:**
→ Start with `03-attention-driven-pyramid-pruning.md` (direct integration guide)

**For foveated vision:**
→ See `01-foveated-gaze-pyramids.md` + `../karpathy/biological-vision/`

**For video understanding:**
→ See `05-3d-volumetric-pyramids-video.md`

## ARR-COC Connection

File `03-attention-driven-pyramid-pruning.md` provides:
- Vervaekean framework mapping to pyramid levels
- Dynamic token budget allocation (64-400 tokens)
- Propositional/perspectival/participatory knowing at different scales

## Cross-References

- Biological vision: `../karpathy/biological-vision/`
- Multiscale transformers: `../karpathy/pyramid-multiscale-vision/`
- Vision token budgets: `../karpathy/practical-implementation/51-vision-token-budgets.md`
