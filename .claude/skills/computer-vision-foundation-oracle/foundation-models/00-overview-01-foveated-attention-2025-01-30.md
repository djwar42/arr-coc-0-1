# Foveated Attention in Vision Transformers

**Dynamic Knowledge Addition** - Created 2025-01-30
**Parent**: [foundation-models/00-overview.md](00-overview.md)
**Cross-Reference**: See **lod-btree-oracle** for deep foveation knowledge

## Quick Overview

**Foveated sampling** and **log-polar mapping** are emerging techniques for efficient vision transformers (2024-2025 trend). They apply biological visual system principles to token allocation.

## Core Concept

**Traditional ViT**:
- Uniform 16×16 patches everywhere
- All tokens treated equally
- High computational cost

**Foveated ViT**:
- Dense sampling at "fovea" (regions of interest)
- Sparse sampling in "periphery" (background)
- 40-60% token reduction
- Inspired by human retinal structure

## Key Research

### 1. TransNeXt (CVPR 2024)

**Paper**: Shi et al., "TransNeXt: Robust Foveal Visual Perception for Vision Transformers"

**Innovation**: **Aggregated Attention** module simulating foveal vision

**Method**:
- Focus tokens at semantically important regions
- Peripheral tokens with reduced resolution
- Convolutional GLU for channel mixing

**Results**: State-of-the-art performance with lower compute

### 2. Foveated Dynamic Transformer

**Paper**: Akkaya et al., "Foveated Dynamic Transformer" (OpenReview)

**Innovation**: Fixation + foveation modules inspired by human visual system

**Method**:
- Dynamic fixation point selection
- Log-polar-like token allocation around fixation
- Efficient and accurate

### 3. FoveaTer (2021)

**Paper**: Jonnalagadda et al., "Foveated Transformer for Image Classification"

**Method**:
- Pooling regions based on visual eccentricity
- Eye movement-inspired token selection
- Dynamic resource allocation

## Log-Polar Transformation

**Mathematical basis**: Converts Cartesian coordinates → log-polar (retinotopic) space

```
ρ = log(r)  where r = √(x² + y²)
θ = atan2(y, x)
```

**Why this matters for VLMs**:
- **Dense center**: High-resolution tokens at focal point
- **Sparse periphery**: Low-resolution tokens away from focus
- **Biologically grounded**: Matches human retina-to-cortex mapping
- **Computationally efficient**: Exponential compression with eccentricity

**Token allocation analogy**:
```
Foveal region (0-2°)   →  400 tokens per patch
Parafoveal (2-10°)     →  200 tokens per patch
Peripheral (>10°)      →  64 tokens per patch
```

## Connection to Your Work (ARR-COC-VIS)

**Your 64-400 token allocation = computational foveation!**

Comparison:
```
Gaze-contingent rendering     →  Query-aware token allocation
Log-polar image sampling      →  Relevance-based patch density
Foveal detail (human: 2°)     →  High relevance (400 tokens)
Peripheral compression (>10°) →  Low relevance (64 tokens)
Visual acuity function        →  Relevance realization scores
```

**Key insight**: You're doing **log-polar sampling in attention space**, not pixel space!

## For Deep Dive: See LOD Oracle

This oracle provides a **brief overview**. For comprehensive knowledge on foveated rendering and log-polar mapping, consult:

**lod-btree-oracle** → `techniques/00-foveated-rendering.md`

The LOD oracle contains:
- Full mathematical treatment of log-polar transforms
- Biological foundations (retinal structure, cortical magnification)
- Implementation strategies (GPU optimization, numerical stability)
- Performance metrics (60-90% cost reduction)
- Recent research (2024-2025 papers)
- Application to VR/AR/VLMs

**Why LOD oracle has this knowledge**:
- Foveated rendering is a core LOD technique
- Extensively studied in computer graphics (20+ years)
- Direct biological grounding in human vision
- Performance optimization for real-time graphics

## Quick References

**Key papers** (ask LOD oracle for details):
- Zhang et al. (2024): "Retinotopic Foveated Rendering"
- Zhang et al. (2025): "Visual Acuity Consistent Foveated Rendering"
- Fan et al. (2024): "Scene-aware Foveated Rendering"

**Vision Transformer papers** (this oracle's domain):
- Shi et al. (2024): TransNeXt (CVPR)
- Akkaya et al.: Foveated Dynamic Transformer
- Jonnalagadda et al. (2021): FoveaTer

## Integration Pattern

**When to use which oracle**:

**Ask LOD-BTree Oracle** when you need:
- Mathematics of log-polar transforms
- Biological foundations (retinal sampling)
- Graphics rendering implementations
- Performance optimization strategies
- Gaze-contingent systems

**Ask CV Foundation Oracle** (this oracle) when you need:
- Vision Transformer architectures
- Self-attention mechanisms
- VLM-specific implementations
- Comparison with standard ViT
- Benchmark results on image classification

**Ask both** when exploring:
- Foveated VLMs (intersection of both domains)
- Token allocation strategies
- Perceptual AI systems
- Biological grounding for transformers

## Key Takeaway

**Foveated attention is emerging** (2024-2025) as a way to make Vision Transformers more efficient by mimicking human visual sampling. The core math comes from computer graphics (log-polar transforms), the architecture comes from deep learning (transformers), and the inspiration comes from biology (human fovea).

**Your ARR-COC-VIS project is at this intersection!**

---

**Cross-reference**: For full treatment, see **lod-btree-oracle/techniques/00-foveated-rendering.md** and its dynamic knowledge additions.
