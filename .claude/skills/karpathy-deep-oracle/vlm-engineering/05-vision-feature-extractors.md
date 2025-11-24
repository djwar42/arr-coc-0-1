# Vision Feature Extractors for VLMs

**Knowledge Base**: VLM Engineering
**Date**: 2025-11-16
**Focus**: Feature extraction techniques for vision-language models

---

## Overview

Vision feature extraction is the process of identifying and representing meaningful visual patterns in images for use in vision-language models. Unlike raw pixel values, extracted features capture semantic information at multiple levels - from low-level edges and textures to high-level semantic concepts. Modern VLMs combine handcrafted features (edges, gradients, color spaces) with learned features (from self-supervised models like DINO) to create rich visual representations.

**Key Insight**: The quality of visual features directly impacts VLM performance. Multi-level feature extraction (low, mid, high) provides the model with both fine-grained details and semantic understanding, enabling better cross-modal reasoning.

From existing knowledge in [gpu-texture-optimization/00-neural-texture-compression-vlm.md](../karpathy/gpu-texture-optimization/00-neural-texture-compression-vlm.md):
- VLM visual tokens are learned features, not raw pixels
- Token compression ratios: DeepSeek-OCR 16×, Ovis VET variable, Qwen3-VL dynamic
- Each token: ~2-4 KB (hidden dimension 768-1536, float16)

---

## Section 1: Low-Level Features (~150 lines)

### Edge Detection

Edge detection identifies boundaries where pixel intensity changes abruptly. Edges correspond to object boundaries, surface discontinuities, and texture changes.

**Common Edge Detection Methods**:

**Sobel Edge Detection**:
- Gradient-based method using 3×3 convolution kernels
- Horizontal kernel (Gx) detects vertical edges, vertical kernel (Gy) detects horizontal edges
- Gradient magnitude: G = √(Gx² + Gy²)
- Fast, simple, good for real-time applications

**Canny Edge Detection**:
- Multi-stage algorithm (Gaussian blur → gradient → non-maximum suppression → hysteresis thresholding)
- Optimal edge detection with thin, connected edges
- Parameters: low/high thresholds for edge classification
- More robust than Sobel, but computationally expensive

**Prewitt and Scharr Operators**:
- Similar to Sobel but different kernel weights
- Scharr provides better rotational symmetry
- Prewitt uses simpler 3×3 kernels

From [Edge Detection in Image Processing (Roboflow Blog)](https://blog.roboflow.com/edge-detection/) (accessed 2025-11-16):
- Edge models: Step (abrupt change), Ramp (gradual transition), Roof (intensity peak)
- First derivative detects rapid intensity changes
- Second derivative (Laplacian) detects zero-crossings at edges
- Sobel uses horizontal Gx = [[-1,0,1],[-1,0,1],[-1,0,1]] and vertical Gy = [[-1,-1,-1],[0,0,0],[1,1,1]]

**Laplacian Edge Detection**:
- Second-order derivative method
- Detects edges via zero-crossings
- More sensitive to noise than gradient methods
- Common kernel: [[0,1,0],[1,-4,1],[0,1,0]]

### Color Space Features

**RGB (Red, Green, Blue)**:
- Native sensor representation
- Highly correlated channels
- Sensitive to illumination changes

**LAB Color Space**:
- L: Lightness (0-100)
- A: Green-Red axis
- B: Blue-Yellow axis
- Perceptually uniform
- Separates luminance from chrominance

**HSV/HSL**:
- Hue, Saturation, Value/Lightness
- More intuitive for color-based segmentation
- Rotation-invariant to illumination (Hue)

### Texture Features

**Gradient Features**:
- Sobel magnitude and orientation
- Captures edge strength and direction
- Used in HOG (Histogram of Oriented Gradients)

**Local Binary Patterns (LBP)**:
- Compares each pixel with neighbors
- Creates binary pattern descriptor
- Rotation-invariant variants available
- Efficient for texture classification

From [pyramid-multiscale-vision/02-laplacian-gaussian-pyramids.md](../karpathy/pyramid-multiscale-vision/02-laplacian-gaussian-pyramids.md):
- Laplacian pyramids capture texture details at multiple scales
- Band-pass filtering preserves high-frequency texture information
- Gaussian pyramids for multi-resolution texture analysis

---

## Section 2: Mid-Level Features (~150 lines)

### Gradient-Based Descriptors

**SIFT (Scale-Invariant Feature Transform)**:
- Detects keypoints at different scales
- Creates 128-dimensional descriptors
- Invariant to scale, rotation, illumination
- Patent-encumbered (now expired)

**SURF (Speeded-Up Robust Features)**:
- Faster alternative to SIFT
- Uses Haar wavelets and integral images
- 64 or 128-dimensional descriptors
- Good balance of speed and accuracy

**HOG (Histogram of Oriented Gradients)**:
- Divides image into cells
- Computes gradient histograms per cell
- Normalizes over blocks
- Popular for pedestrian detection

### Corner Detection

**Harris Corner Detector**:
- Identifies corners via intensity variation
- Eigenvalue analysis of structure tensor
- Rotation-invariant
- Not scale-invariant

**FAST (Features from Accelerated Segment Test)**:
- Extremely fast corner detection
- Compares pixel intensity with circular neighbors
- Used in real-time applications (SLAM, tracking)

**Shi-Tomasi Corner Detector**:
- Improvement over Harris
- Better corner selection criterion
- Used in optical flow tracking

### Multi-Scale Representations

From [pyramid-multiscale-vision/01-fpn-feature-pyramids.md](../karpathy/pyramid-multiscale-vision/01-fpn-feature-pyramids.md):
- FPN (Feature Pyramid Networks) create multi-scale features with minimal cost
- Top-down pathway propagates semantic features to high-resolution maps
- Lateral connections merge bottom-up (spatial) with top-down (semantic)
- Pyramid levels {P2, P3, P4, P5} with stride {4, 8, 16, 32}

**Gaussian Pyramids**:
- Successive blurring and downsampling
- Each level is half resolution of previous
- Captures features at different scales

**Laplacian Pyramids**:
- Difference between Gaussian levels
- Captures band-pass filtered details
- Efficient for multi-resolution analysis

---

## Section 3: High-Level Features (Dense Self-Supervised) (~150 lines)

### DINO Features

DINO (Self-Distillation with No Labels) produces rich, dense features without supervised training.

From search results on "DINO features dense prediction self-supervised":

**DINOv3 Architecture** (Meta AI, accessed 2025-11-16):
- Self-supervised learning at unprecedented scale
- Universal vision backbones for dense prediction tasks
- Provides both global (CLS token) and dense (patch) features
- Trained on 142M images without labels

**Key Properties**:
- Dense features: Every patch embedding captures local semantics
- Part discovery: Automatically segments objects into parts
- Scene understanding: Captures spatial relationships
- Transfer learning: Strong zero-shot performance on downstream tasks

**Applications**:
- Semantic segmentation (per-pixel classification)
- Depth estimation (monocular depth prediction)
- Object detection (without bounding box supervision)
- Visual grounding (linking text to image regions)

**DenseDINO** (from arXiv search results, accessed 2025-11-16):
- Boosts dense self-supervised learning for transformers
- Creates dense visual representations without labels
- Better than supervised methods for dense prediction
- Used for segmentation, depth, surface normal estimation

### CLIP Features

**Contrastive Learning**:
- Trained on 400M image-text pairs
- Learns joint embedding space
- Zero-shot classification capability
- Good semantic features but weaker spatial detail

**Vision Encoder**:
- ViT-based (Vision Transformer)
- CLS token for global features
- Patch tokens for dense features
- Multiple scales: ViT-B/32, ViT-B/16, ViT-L/14

### EVA-CLIP

- Scaling CLIP to 1B parameters
- Billion-scale pre-training
- Better performance on downstream tasks
- Stronger dense features than original CLIP

---

## Section 4: Multi-Scale Feature Extraction (~100 lines)

### Feature Pyramid Networks (FPN)

From [pyramid-multiscale-vision/01-fpn-feature-pyramids.md](../karpathy/pyramid-multiscale-vision/01-fpn-feature-pyramids.md):

**Architecture**:
1. **Bottom-Up Pathway**: Standard ConvNet forward pass (ResNet stages)
2. **Top-Down Pathway**: Upsample and propagate semantic features
3. **Lateral Connections**: 1×1 conv + element-wise addition at each level

**Properties**:
- All pyramid levels have 256 channels (unified representation)
- P2 (stride 4): Highest resolution, finest details, small objects
- P3 (stride 8): Medium objects
- P4 (stride 16): Large objects
- P5 (stride 32): Strongest semantics, very large objects

### Pyramid Vision Transformers

From search results on "multi-scale feature pyramids VLM vision transformers":

**Multiscale Vision Transformers (MViT)**:
- Creates multi-scale feature hierarchies for transformers
- Effective modeling of dense visual input
- Pooling attention at different stages
- Reduces computational cost while maintaining accuracy

**Pyramid Sparse Transformer (PST)**:
- Coarse-to-fine token selection
- Shared attention parameters across scales
- Reduces computation while preserving features
- Plug-and-play module for existing architectures

### Wavelet Features

From [pyramid-multiscale-vision/05-wavelet-multiresolution.md](../karpathy/pyramid-multiscale-vision/05-wavelet-multiresolution.md):
- Wavelet transforms decompose images into frequency bands
- Captures both spatial and frequency information
- Useful for texture analysis across scales
- Efficient compression and denoising

---

## Section 5: Learned Features vs Handcrafted Features (~100 lines)

### Handcrafted Features

**Advantages**:
- Interpretable (known what they measure)
- Fast to compute
- No training required
- Domain knowledge encoded
- Stable across datasets

**Disadvantages**:
- Limited expressiveness
- May miss complex patterns
- Manual design required
- Not adaptive to task

**Examples**:
- Sobel edges
- LAB color channels
- HOG descriptors
- SIFT keypoints

### Learned Features

**Advantages**:
- Adaptive to task and data
- Hierarchical representations
- Capture complex patterns
- End-to-end optimization
- Better performance (usually)

**Disadvantages**:
- Require training data
- Computationally expensive
- Less interpretable
- May overfit
- Domain transfer challenges

**Examples**:
- DINO patch embeddings
- CLIP vision features
- ResNet activations
- ViT patch tokens

### Hybrid Approaches

**Best Practice**: Combine both for robustness

**Example**: ARR-COC-0-1 texture array uses:
- Handcrafted: RGB, LAB, Sobel edges, spatial coordinates, eccentricity
- Learned: Vision encoder embeddings (from frozen ViT)

**Rationale**:
- Handcrafted provides interpretable low-level cues
- Learned captures semantic high-level patterns
- Together: Better generalization and robustness

---

## Section 6: Feature Fusion Strategies (~100 lines)

### Concatenation

**Simple Feature Stacking**:
```
Concatenated = [RGB, LAB, Sobel, Spatial, Learned]
```

**Properties**:
- Preserves all information
- Linear combination possible
- Higher dimensionality
- May have redundancy

### Channel-wise Addition

**Weighted Sum**:
```
Fused = α·F1 + β·F2 + γ·F3
```

**Properties**:
- Reduces dimensionality
- Requires aligned features
- Learned or fixed weights
- Information loss possible

### Gated Fusion

**Attention-based Weighting**:
```
Gate = σ(W·[F1, F2])
Fused = Gate ⊙ F1 + (1-Gate) ⊙ F2
```

**Properties**:
- Dynamic weighting
- Adaptive to input
- Learned gate function
- Selective information flow

### Hierarchical Fusion

**Multi-Level Combination**:
- Low-level features → edges, color
- Mid-level features → corners, textures
- High-level features → semantics, objects
- Fuse at appropriate stages

From [pyramid-multiscale-vision/01-fpn-feature-pyramids.md](../karpathy/pyramid-multiscale-vision/01-fpn-feature-pyramids.md):
- FPN fuses features with semantic meaning AND spatial accuracy
- Bottom-up provides spatial precision
- Top-down provides semantic strength
- Lateral connections enable effective fusion

---

## Section 7: Feature Extraction Efficiency (~100 lines)

### Computational Considerations

**Handcrafted Features**:
- Sobel: ~1ms per image (GPU)
- LAB conversion: <1ms
- Spatial grids: Negligible
- Total handcrafted: ~2-3ms

**Learned Features**:
- ViT-B/16 forward pass: ~10-20ms (GPU)
- DINO features: ~15-25ms
- CLIP encoding: ~10-15ms
- Total learned: ~10-25ms

**Tradeoff**:
- Handcrafted: 10× faster but less expressive
- Learned: Slower but better semantic understanding
- Hybrid: Best balance for VLMs

### Memory Footprint

**Feature Storage**:
- RGB (3 channels): H×W×3 bytes
- LAB (3 channels): H×W×3 bytes
- Sobel (2 channels): H×W×2 bytes
- Spatial (2 channels): H×W×2 bytes
- Total handcrafted: H×W×10 bytes

**Learned Features**:
- ViT patch embeddings: (H/16)×(W/16)×768 floats
- At 224×224: 196×768×4 = ~600KB
- At 672×672: 1764×768×4 = ~5.4MB

**Compression Strategy**:
- Extract features densely
- Apply relevance-based selection
- Store only selected tokens
- 3-10× memory reduction

### Caching Strategies

From [gpu-texture-optimization/08-texture-cache-coherency.md](../karpathy/gpu-texture-optimization/08-texture-cache-coherency.md):
- Spatial locality: Adjacent pixels often accessed together
- Temporal locality: Same regions queried multiple times
- Cache extracted features for reuse
- Mipmap-like hierarchies for multi-scale access

---

## Section 8: ARR-COC-0-1 Texture Array (13-Channel Feature Extraction) (~150 lines)

### Architecture Overview

ARR-COC-0-1 uses a **13-channel texture array** combining handcrafted and learned features for adaptive relevance realization.

**Channel Breakdown**:
1-3. **RGB**: Raw color (3 channels)
4-6. **LAB**: Perceptually uniform color (3 channels)
7-8. **Sobel**: Edge magnitude and orientation (2 channels)
9-10. **Spatial**: Normalized x, y coordinates (2 channels)
11. **Eccentricity**: Distance from image center (1 channel)
12-13. **Reserved**: Future learned features (2 channels)

### Design Rationale

**Why 13 Channels?**

From biological vision knowledge [biological-vision/03-foveated-rendering-peripheral.md](../karpathy/biological-vision/03-foveated-rendering-peripheral.md):
- Human retina has ~6M cones (color) + 120M rods (luminance)
- Multiple retinal ganglion cell types encode different features
- Parallel pathways for color, edges, motion
- ARR-COC-0-1 mimics multi-channel encoding

**RGB (Channels 1-3)**:
- Direct sensor data
- Color object recognition
- Illumination information
- Used by: Propositional scorer (statistics)

**LAB (Channels 4-6)**:
- Perceptually uniform
- L: Lightness (luminance)
- A: Green-red (opponent)
- B: Blue-yellow (opponent)
- Used by: Perspectival scorer (salience)

**Sobel (Channels 7-8)**:
- Magnitude: Edge strength
- Orientation: Edge direction (0-360°)
- Detects object boundaries
- Used by: All scorers (structure)

**Spatial (Channels 9-10)**:
- X-coordinate (0-1, left to right)
- Y-coordinate (0-1, top to bottom)
- Absolute position in image
- Used by: Participatory scorer (query localization)

**Eccentricity (Channel 11)**:
- Distance from image center
- Radial coordinate: r = √((x-0.5)² + (y-0.5)²)
- Foveal bias (center important)
- Used by: Perspectival scorer (attention)

### Extraction Pipeline

```python
def extract_texture_array(image: torch.Tensor) -> torch.Tensor:
    """
    Extract 13-channel texture array from image.

    Args:
        image: [B, 3, H, W] RGB image (0-1 normalized)

    Returns:
        texture: [B, 13, H, W] multi-channel features
    """
    B, C, H, W = image.shape

    # 1. RGB (channels 0-2)
    rgb = image  # Already in [0,1]

    # 2. LAB conversion (channels 3-5)
    lab = rgb_to_lab(image)  # Custom conversion

    # 3. Sobel edges (channels 6-7)
    sobel_mag, sobel_orient = compute_sobel(image)

    # 4. Spatial coordinates (channels 8-9)
    x_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
    y_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)

    # 5. Eccentricity (channel 10)
    eccentricity = torch.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)

    # 6. Concatenate all channels
    texture = torch.cat([
        rgb,                # [B, 3, H, W]
        lab,                # [B, 3, H, W]
        sobel_mag,          # [B, 1, H, W]
        sobel_orient,       # [B, 1, H, W]
        x_coords,           # [B, 1, H, W]
        y_coords,           # [B, 1, H, W]
        eccentricity,       # [B, 1, H, W]
        torch.zeros(B, 2, H, W)  # Reserved
    ], dim=1)  # [B, 13, H, W]

    return texture
```

### Integration with Relevance Realization

**Three Ways of Knowing Use Features**:

1. **Propositional Knowing** (Information Scorer):
   - Uses RGB channels for statistical content
   - Shannon entropy: H = -Σ p(x) log p(x)
   - Histogram of pixel intensities
   - High entropy = high information

2. **Perspectival Knowing** (Salience Scorer):
   - Uses LAB for perceptual salience
   - Sobel for edge-based attention
   - Eccentricity for foveal bias
   - Jungian archetypes (light/dark, center/periphery)

3. **Participatory Knowing** (Coupling Scorer):
   - Uses spatial coordinates for localization
   - RGB/LAB for query-content matching
   - Sobel for query-relevant edges
   - Cross-attention between query and features

**Adaptive LOD Based on Features**:
- High Sobel magnitude → allocate more tokens
- High eccentricity → reduce tokens (peripheral)
- High coupling score → increase resolution
- Balanced by opponent processing

---

## Sources

**Existing Knowledge**:
- [gpu-texture-optimization/00-neural-texture-compression-vlm.md](../karpathy/gpu-texture-optimization/00-neural-texture-compression-vlm.md) - VLM token representations
- [pyramid-multiscale-vision/01-fpn-feature-pyramids.md](../karpathy/pyramid-multiscale-vision/01-fpn-feature-pyramids.md) - Multi-scale feature fusion
- [pyramid-multiscale-vision/02-laplacian-gaussian-pyramids.md](../karpathy/pyramid-multiscale-vision/02-laplacian-gaussian-pyramids.md) - Multi-resolution texture
- [pyramid-multiscale-vision/05-wavelet-multiresolution.md](../karpathy/pyramid-multiscale-vision/05-wavelet-multiresolution.md) - Wavelet decomposition
- [biological-vision/00-gestalt-visual-attention.md](../karpathy/biological-vision/00-gestalt-visual-attention.md) - Perceptual grouping
- [biological-vision/03-foveated-rendering-peripheral.md](../karpathy/biological-vision/03-foveated-rendering-peripheral.md) - Foveal vision
- [gpu-texture-optimization/08-texture-cache-coherency.md](../karpathy/gpu-texture-optimization/08-texture-cache-coherency.md) - Cache strategies

**Web Research**:
- [Edge Detection in Image Processing (Roboflow Blog)](https://blog.roboflow.com/edge-detection/) - Sobel, Canny, Laplacian edge detection (accessed 2025-11-16)
- [DINOv3: Self-supervised learning for vision at unprecedented scale (Meta AI)](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) - Dense self-supervised features (accessed 2025-11-16)
- Google Search: "DINO features dense prediction self-supervised" - DenseDINO, emergent properties (accessed 2025-11-16)
- Google Search: "multi-scale feature pyramids VLM vision transformers" - Pyramid ViT, MViT (accessed 2025-11-16)
- Google Search: "vision feature extraction VLM 2024" - VLM feature overview (accessed 2025-11-16)
