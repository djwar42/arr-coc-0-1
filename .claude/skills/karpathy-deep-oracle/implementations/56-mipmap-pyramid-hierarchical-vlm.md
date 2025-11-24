# Mipmap Pyramid & Hierarchical Vision Transformers

## Overview

Mipmap pyramids are GPU texture structures that store multiple resolution levels of the same image, enabling efficient Level-of-Detail (LOD) sampling. When integrated with Vision Transformers (ViTs), they enable hierarchical multi-scale feature extraction that mirrors biological vision and improves computational efficiency.

**Core Concept**: Hierarchical VLMs process visual information at multiple scales simultaneously - high-resolution details in early layers, coarse semantic features in deeper layers - using pyramid structures inspired by GPU texture mipmapping.

## Mipmap Fundamentals

### What Are Mipmaps?

Mipmaps are pre-computed image pyramid structures used in GPU texture mapping:

- **Pyramid Structure**: Original image + downsampled versions (1/2, 1/4, 1/8, etc.)
- **GPU Hardware**: Texture units automatically select appropriate mipmap level
- **Memory Overhead**: +33% total memory (1 + 1/4 + 1/16 + 1/64 + ... ≈ 1.33×)
- **Purpose**: Efficient texture sampling, anti-aliasing, LOD rendering

**Mipmap Levels**:
```
Level 0: 1024×1024 (full resolution)
Level 1:  512×512  (1/2 scale)
Level 2:  256×256  (1/4 scale)
Level 3:  128×128  (1/8 scale)
Level 4:   64×64   (1/16 scale)
...
Level N:    1×1    (single pixel)
```

### GPU Texture Sampling

Hardware texture units support:
- **Trilinear filtering**: Blend between mipmap levels
- **Anisotropic filtering**: Sample multiple mipmap levels for oblique viewing
- **Automatic LOD**: Hardware calculates texture gradients to select mipmap level

## Hierarchical VLM Feature Extraction

### Multiscale Vision Transformers (MViT)

From [Multiscale Vision Transformers (Fan et al., 2021, arXiv:2104.11227)](https://arxiv.org/abs/2104.11227):

**Architecture**: Channel-resolution scale stages that create feature pyramids
- **Early Layers**: High spatial resolution (e.g., 224×224), small channel count (96 dim)
- **Mid Layers**: Medium resolution (56×56), moderate channels (192 dim)
- **Deep Layers**: Low resolution (14×14), high channel capacity (768 dim)

**Key Innovation**: Unlike standard ViT (uniform resolution), MViT progressively reduces spatial resolution while expanding channel capacity - creating a "visual pyramid" inside the transformer.

### Coarse-to-Fine Vision Understanding

**Multi-Scale Processing Benefits**:

1. **Computational Efficiency**: Process high-res details only where needed
2. **Receptive Field Growth**: Coarse layers capture global context efficiently
3. **Feature Hierarchy**: Low-level edges → mid-level textures → high-level semantics
4. **Biological Correspondence**: Mirrors V1 → V2 → V4 → IT cortical hierarchy

**Example Pipeline**:
```
Input Image (224×224×3)
    ↓
Stage 1: 56×56 patches, 96 channels  (fine details)
    ↓
Stage 2: 28×28 patches, 192 channels (textures)
    ↓
Stage 3: 14×14 patches, 384 channels (objects)
    ↓
Stage 4: 7×7 patches, 768 channels   (scenes)
```

### Hierarchical Image Pyramid Transformer (HIPT)

From [HIPT (Mahmood Lab, GitHub)](https://github.com/mahmoodlab/HIPT):

**Use Case**: Gigapixel image analysis (medical imaging, satellite imagery)

**Three-Level Hierarchy**:
1. **256×256 patches**: Extract local features (cells, textures)
2. **4096×4096 regions**: Aggregate patch features (tissue structures)
3. **Full image**: Global context integration (entire slide)

**Key Technique**: Nested ViT encoders - each level uses separate transformer, features propagate upward through pyramid.

## VLM Architecture Integration

### Dynamic Resolution with Mipmaps

**Query-Aware Mipmap Selection**:
- Text query determines required visual detail level
- GPU samples appropriate mipmap level per image region
- "Find the cat" → High-res mipmap (Level 0-1)
- "What's the overall scene?" → Coarse mipmap (Level 3-4)

**ARR-COC Relevance Mapping**:
- High relevance regions: Sample fine mipmap levels (64-400 tokens)
- Low relevance regions: Sample coarse mipmap levels (16-64 tokens)
- Smooth transitions: GPU trilinear filtering between levels

### Feature Pyramid Networks (FPN) + ViT

**Integration Pattern**:
1. Build mipmap pyramid from input image (GPU texture units)
2. Apply ViT patch embedding to each pyramid level
3. Fuse multi-scale features with lateral connections
4. Attend across scales based on query relevance

**Computational Benefits**:
- Avoid processing full-res image for all features
- Leverage GPU hardware acceleration (texture cache, filtering)
- Dynamic LOD based on attention patterns

### Real-World VLM Examples

**Ovis Visual Embedding Table (VET)**:
- Fixed embedding table acts as "learned mipmap"
- Interpolate between discrete visual tokens (like mipmap sampling)
- 16×16 patches compressed to variable token budgets

**DeepSeek-OCR Optical Compression**:
- 16× compression ratio mirrors mipmap Level 4 (1/16 scale)
- Multi-scale features before final compression
- Preserve high-frequency details where needed (text regions)

## Implementation Considerations

**GPU Texture Pipeline**:
```python
# Pseudo-code: Mipmap-based VLM sampling
mipmap_pyramid = build_mipmap_pyramid(image)  # GPU texture units
relevance_map = compute_relevance(query, image)  # ARR-COC

for region in image_regions:
    lod_level = map_relevance_to_lod(relevance_map[region])
    features = sample_mipmap(mipmap_pyramid, region, lod_level)
    tokens.append(vit_embed(features))
```

**Hardware Acceleration**:
- Use CUDA texture objects for mipmap storage
- Leverage L2 texture cache for repeated sampling
- Trilinear filtering for smooth LOD transitions

**Memory Trade-offs**:
- Mipmap storage: +33% memory overhead
- Computation savings: 2-4× faster for multi-scale processing
- Quality: Better anti-aliasing, smooth scale transitions

## Sources

**Research Papers**:
- [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) - Fan et al., ICCV 2021 (arXiv:2104.11227, accessed 2025-01-31)
- [HIPT: Hierarchical Image Pyramid Transformer](https://github.com/mahmoodlab/HIPT) - Mahmood Lab (accessed 2025-01-31)

**Web Resources**:
- [AI at Meta: Multiscale Vision Transformers](https://ai.meta.com/blog/multiscale-vision-transformers-an-architecture-for-modeling-visual-data/) (accessed 2025-01-31)
- Feature Pyramid Networks for object detection (FPN architecture)

**Related Oracle Knowledge**:
- Biological vision hierarchies (V1-IT cortical processing)
- ARR-COC relevance-aware token allocation
- GPU texture memory optimization
