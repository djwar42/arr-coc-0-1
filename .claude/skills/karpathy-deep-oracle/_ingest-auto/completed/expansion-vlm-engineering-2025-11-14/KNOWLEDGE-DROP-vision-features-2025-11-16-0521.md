# KNOWLEDGE DROP: Vision Feature Extractors

**Date**: 2025-11-16 05:21
**PART**: 6 of 20
**File Created**: vlm-engineering/05-vision-feature-extractors.md
**Lines**: ~700

---

## What Was Created

Comprehensive knowledge file covering vision feature extraction for VLMs, from low-level edges to high-level semantic features.

**8 Major Sections**:
1. Low-Level Features (edges, colors, textures) - 150 lines
2. Mid-Level Features (SIFT, SURF, HOG, corners) - 150 lines
3. High-Level Features (DINO, CLIP, EVA-CLIP) - 150 lines
4. Multi-Scale Feature Extraction (FPN, PyramidViT) - 100 lines
5. Learned vs Handcrafted Features - 100 lines
6. Feature Fusion Strategies - 100 lines
7. Feature Extraction Efficiency - 100 lines
8. ARR-COC-0-1 Texture Array (13-channel) - 150 lines

---

## Key Knowledge Acquired

### Edge Detection Methods

**Sobel Edge Detection**:
- Gradient-based with 3×3 kernels
- Horizontal Gx, Vertical Gy
- Magnitude G = √(Gx² + Gy²)
- Fast, simple, good for real-time

**Canny Edge Detection**:
- Multi-stage: Gaussian blur → gradient → non-max suppression → hysteresis
- Optimal thin connected edges
- Low/high threshold parameters
- More robust but slower

**Laplacian**:
- Second derivative method
- Zero-crossing detection
- Sensitive to noise
- Common kernel: [[0,1,0],[1,-4,1],[0,1,0]]

### Self-Supervised Dense Features

**DINOv3** (Meta AI):
- Trained on 142M images without labels
- Provides global (CLS) and dense (patch) features
- Automatic part discovery
- Strong zero-shot performance

**DenseDINO**:
- Boosts dense self-supervised learning
- Better than supervised for dense prediction
- Segmentation, depth, surface normals

### Multi-Scale Extraction

**Feature Pyramid Networks (FPN)**:
- Bottom-up: Standard ConvNet forward
- Top-down: Upsample semantic features
- Lateral connections: Merge at each level
- Unified 256-channel representation

**Pyramid Vision Transformers**:
- Multi-scale hierarchies for transformers
- Pooling attention at different stages
- Pyramid Sparse Transformer (PST): coarse-to-fine token selection

### ARR-COC-0-1 Integration

**13-Channel Texture Array**:
- RGB (3): Raw color
- LAB (3): Perceptually uniform
- Sobel (2): Edge magnitude + orientation
- Spatial (2): X, Y coordinates
- Eccentricity (1): Distance from center
- Reserved (2): Future learned features

**Usage by Scorers**:
- Propositional: RGB statistics (Shannon entropy)
- Perspectival: LAB + Sobel + eccentricity (salience)
- Participatory: Spatial + all channels (query coupling)

---

## Sources Used

**Existing Knowledge** (3 directories):
- gpu-texture-optimization/ - 20 files on texture compression, mipmaps
- pyramid-multiscale-vision/ - 10 files on multi-scale features
- biological-vision/ - 7 files on foveal vision, attention

**Web Research** (4 searches, 2 scrapes):
- Edge detection methods (Sobel, Canny, Laplacian)
- DINO features for dense prediction
- Multi-scale pyramids in VLMs
- Vision feature extraction 2024

**Key URLs**:
- https://blog.roboflow.com/edge-detection/ - Comprehensive edge detection guide
- Meta AI DINOv3 blog (login required, used search results)
- arXiv papers on DenseDINO, MViT

---

## Quality Metrics

✓ **All 8 sections completed** with proper citations
✓ **Low-level to high-level progression** (edges → textures → semantics)
✓ **Handcrafted + learned features** balanced coverage
✓ **ARR-COC-0-1 integration** detailed (13-channel array)
✓ **Multi-scale extraction** (FPN, pyramids, wavelets)
✓ **Efficiency considerations** (compute, memory, caching)
✓ **Code examples** included for texture array extraction
✓ **Source citations** throughout (existing + web)

---

## Integration Notes

**Connects to existing knowledge**:
- Extends gpu-texture-optimization with VLM-specific features
- Builds on pyramid-multiscale-vision for multi-scale extraction
- Uses biological-vision for foveal bias and eccentricity

**New concepts introduced**:
- DINO/DenseDINO for dense self-supervised features
- Edge detection methods (Sobel, Canny, Laplacian, Prewitt, Roberts, Scharr)
- Feature fusion strategies (concatenation, gating, hierarchical)
- 13-channel texture array design for ARR-COC-0-1

**Ready for next PART**:
- PART 7: Foveated Vision & Adaptive Resolution
- Will use feature extractors for adaptive LOD
- Eccentricity channel enables foveal processing
