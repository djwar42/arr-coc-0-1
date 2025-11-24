# Super-Resolution with Pyramid Guidance

## Overview

Super-resolution (SR) networks reconstruct high-resolution images from low-resolution inputs using learned transformations. Pyramid-guided architectures leverage multi-scale hierarchical structures to progressively refine upsampling, providing coarse-to-fine guidance that improves perceptual quality and reduces artifacts. This document explores pyramid-based SR architectures, multi-scale loss functions, GAN discriminators, and progressive training strategies.

**Key insight**: Pyramids provide natural scaffolding for super-resolution — coarse levels establish global structure while fine levels add local detail, mirroring human visual processing.

---

## Section 1: Coarse-to-Fine Upsampling Networks

### Progressive Upsampling Strategy

Traditional SR networks perform single-stage upsampling (e.g., bicubic → 4× upsampled output), which struggles to preserve both global structure and local detail. Pyramid-based approaches decompose upsampling into progressive stages:

```
LR input (64×64)
    ↓ 2× upsampling stage
Intermediate (128×128)
    ↓ 2× upsampling stage
SR output (256×256)
```

**Benefits**:
- **Gradual refinement**: Each stage focuses on appropriate frequency bands
- **Reduced aliasing**: Smaller upsampling factors per stage reduce high-frequency artifacts
- **Feature reuse**: Intermediate representations guide subsequent stages

From [Laplacian Pyramid Super-Resolution Networks (LapSRN)](https://www.wslai.net/publications/lapsrn/papers/cvpr17_LapSRN.pdf) (CVPR 2017):
> "We progressively reconstruct the sub-band residuals of high-resolution images at multiple pyramid levels, enabling deep networks to learn coarse-to-fine super-resolution mappings."

### Laplacian Pyramid Architecture

LapSRN predicts residual images at each pyramid level rather than full reconstructions:

```python
# Conceptual Laplacian pyramid SR
def laplacian_pyramid_sr(lr_image, target_scale=4):
    levels = []
    current = lr_image

    # Upsampling stages (log2(target_scale) stages)
    for stage in range(int(np.log2(target_scale))):
        # Feature extraction at current resolution
        features = feature_extractor(current)

        # Predict residual (high-frequency details)
        residual = residual_predictor(features)

        # Upsampling + add residual
        upsampled = upsample_2x(current)
        current = upsampled + residual
        levels.append(current)

    return current, levels  # SR image + pyramid levels
```

**Key components**:
1. **Feature extraction branch**: Learns hierarchical features at each scale
2. **Residual prediction**: Predicts Laplacian residuals (high-frequency details only)
3. **Transposed convolution**: Learnable 2× upsampling operators
4. **Skip connections**: Propagate low-frequency information across stages

### EDSR and Enhanced Residual Networks

Enhanced Deep Super-Resolution (EDSR) removes batch normalization from residual blocks, enabling larger models and better gradient flow. While not explicitly pyramid-structured, EDSR's deep residual architecture creates implicit multi-scale feature hierarchies.

From [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://link.springer.com/article/10.1007/s00521-024-09702-1) (2024):
> "SRResNet and EDSR methods suggest removing redundant modules and optimizing them, simplifying the network architecture. As the model becomes deeper, residual connections enable effective gradient propagation across multiple scales."

**Architecture comparison**:

| Model | Pyramid Structure | Upsampling | Key Innovation |
|-------|------------------|------------|----------------|
| **LapSRN** | Explicit Laplacian pyramid | Progressive 2× stages | Residual prediction per level |
| **EDSR** | Implicit feature hierarchy | Single-stage (or progressive) | Removed batch norm, wider channels |
| **RCAN** | Channel attention pyramid | Single or multi-stage | Residual-in-residual + attention |

### Skip Connections from Coarse to Fine

Pyramid networks use skip connections to propagate coarse-level information to fine levels:

```
Coarse level (64×64) ───────┐
    ↓                        │
Mid level (128×128) ────────┤ (skip connections)
    ↓                        │
Fine level (256×256) ←───────┘
```

**Benefits**:
- **Global context preservation**: Fine levels retain global structure from coarse predictions
- **Gradient flow**: Enables backpropagation through deep progressive stages
- **Semantic guidance**: High-level semantic features guide low-level texture synthesis

---

## Section 2: Pyramid Loss Functions (Perceptual, Laplacian)

### Multi-Scale Reconstruction Loss

Standard pixel-wise Mean Squared Error (MSE) loss operates on the final SR output only. Multi-scale pyramid loss computes reconstruction error at each pyramid level:

```python
def multi_scale_loss(sr_pyramid, hr_pyramid):
    """
    Compute reconstruction loss at each pyramid level.

    Args:
        sr_pyramid: List of SR images at [64×64, 128×128, 256×256]
        hr_pyramid: List of downsampled HR targets at matching scales
    """
    loss = 0.0
    weights = [0.25, 0.5, 1.0]  # Weight fine levels more heavily

    for sr_level, hr_level, weight in zip(sr_pyramid, hr_pyramid, weights):
        loss += weight * F.mse_loss(sr_level, hr_level)

    return loss
```

**Advantages**:
- **Multi-scale supervision**: Guides learning at intermediate stages
- **Prevents gradient vanishing**: Earlier layers receive direct supervision
- **Better convergence**: Faster training compared to single-scale loss

From [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://ar5iv.labs.arxiv.org/html/1710.01992) (CVPR 2017):
> "Our method achieves fast and accurate super-resolution using a Laplacian pyramid structure with shared-weight feature extraction and progressive reconstruction at multiple scales."

### Perceptual Loss (VGG Feature Matching)

Perceptual loss measures distance in feature space rather than pixel space, producing more natural-looking results. SRGAN introduced VGG-based perceptual loss for SR:

```python
def perceptual_loss(sr_image, hr_image, vgg_model, layer='conv5_4'):
    """
    Compute MSE of VGG features (perceptual loss).

    Args:
        vgg_model: Pre-trained VGG-19 (frozen weights)
        layer: Which VGG layer to extract features from
    """
    sr_features = vgg_model.extract_features(sr_image, layer)
    hr_features = vgg_model.extract_features(hr_image, layer)

    return F.mse_loss(sr_features, hr_features)
```

From [SRGAN - Super Resolution GAN](https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec) (Medium, 2018):
> "SRGAN uses a perceptual loss measuring the MSE of features extracted by a VGG-19 network. For a specific layer within VGG-19, we want their features to be matched (Minimum MSE for features)."

**Why perceptual loss works**:
- **Semantic similarity**: VGG features encode high-level semantics (edges, textures, objects)
- **Texture fidelity**: Better at preserving perceptual quality than pixel-wise MSE
- **Human alignment**: Correlates better with human perceptual judgments

**Multi-scale perceptual loss**: Combine perceptual loss at multiple VGG layers:

```python
# Extract features from multiple VGG layers
layers = ['conv3_4', 'conv4_4', 'conv5_4']  # shallow → deep
perceptual_loss = sum([
    F.mse_loss(vgg(sr, layer), vgg(hr, layer))
    for layer in layers
])
```

### Laplacian Pyramid Loss

Laplacian pyramid loss explicitly penalizes errors in frequency-domain decomposition:

```python
def laplacian_pyramid_decompose(image, levels=3):
    """Build Laplacian pyramid."""
    pyramid = []
    current = image

    for i in range(levels):
        # Downsample
        downsampled = F.avg_pool2d(current, 2)
        # Upsample back to original size
        upsampled = F.interpolate(downsampled, scale_factor=2, mode='bilinear')
        # Laplacian (high-frequency residual)
        laplacian = current - upsampled
        pyramid.append(laplacian)
        current = downsampled

    pyramid.append(current)  # Coarsest level (low-frequency base)
    return pyramid

def laplacian_pyramid_loss(sr_image, hr_image):
    """Compare Laplacian pyramids of SR and HR."""
    sr_pyramid = laplacian_pyramid_decompose(sr_image)
    hr_pyramid = laplacian_pyramid_decompose(hr_image)

    loss = 0.0
    for sr_level, hr_level in zip(sr_pyramid, hr_pyramid):
        loss += F.l1_loss(sr_level, hr_level)  # L1 for sparsity

    return loss
```

**Benefits**:
- **Frequency-aware**: Penalizes errors in specific frequency bands
- **Edge preservation**: High-frequency Laplacian levels capture edges/textures
- **Artifact reduction**: Prevents over-smoothing common with MSE-only loss

From [GAN-Based Super-Resolution with Enhanced Multi-Scale Laplacian Pyramid](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70028) (IET 2025):
> "An enhanced multi-scale Laplacian pyramid structure is designed to capture and process image details at different scales, combined with frequency domain loss for artifact reduction."

---

## Section 3: Multi-Scale Discriminators (GANs)

### SRGAN: Single-Scale Discriminator

SRGAN (Super-Resolution GAN) uses adversarial training to produce photo-realistic SR images. The discriminator operates on full-resolution images only:

```
Generator: LR (64×64) → SR (256×256)
Discriminator: SR/HR (256×256) → Real/Fake probability
```

**SRGAN architecture**:
- **Generator**: 16 residual blocks + sub-pixel convolution for upsampling
- **Discriminator**: VGG-style CNN with strided convolutions
- **Loss**: Perceptual loss (VGG features) + adversarial loss

From [SRGAN paper](https://arxiv.org/pdf/1609.04802.pdf) (CVPR 2017):
> "We propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network trained to differentiate between super-resolved images and original photo-realistic images."

**Generator loss**:
```python
# SRGAN generator loss
generator_loss = (
    perceptual_loss(sr, hr, vgg_model) * lambda_percept +  # Content loss
    adversarial_loss(discriminator(sr)) * lambda_adv        # Adversarial loss
)

# Adversarial loss: fool discriminator
adversarial_loss = -torch.log(discriminator(sr))  # Want D(SR) → 1
```

### Multi-Scale Discriminators

Multi-scale discriminators evaluate image realism at multiple resolutions, providing richer adversarial supervision:

```
SR image (256×256) ──→ Discriminator 1 (fine details)
    ↓ downsample 2×
(128×128) ──→ Discriminator 2 (medium details)
    ↓ downsample 2×
(64×64) ──→ Discriminator 3 (coarse structure)
```

**Architecture**:
```python
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        self.disc_fine = Discriminator(channels=3)     # 256×256
        self.disc_medium = Discriminator(channels=3)   # 128×128
        self.disc_coarse = Discriminator(channels=3)   # 64×64

    def forward(self, image):
        # Evaluate at multiple scales
        fine_score = self.disc_fine(image)
        medium_score = self.disc_medium(F.avg_pool2d(image, 2))
        coarse_score = self.disc_coarse(F.avg_pool2d(image, 4))

        return fine_score, medium_score, coarse_score

# Multi-scale adversarial loss
adv_loss = (
    bce_loss(disc_fine(sr), 1.0) +
    bce_loss(disc_medium(sr_downsampled_2x), 1.0) +
    bce_loss(disc_coarse(sr_downsampled_4x), 1.0)
) / 3
```

**Benefits**:
- **Hierarchical realism**: Ensures realistic appearance at multiple scales
- **Reduced artifacts**: Fine discriminator catches local artifacts, coarse discriminator ensures global coherence
- **Training stability**: Multi-scale feedback provides richer gradient signal

From [Progressive Super-Resolution](https://www.researchgate.net/scientific-contributions/Byungkon-Kang-2140412846):
> "The multi-scale discriminator approach evaluates realism at different resolutions, providing comprehensive adversarial feedback. LapSRN achieves progressive super-resolution through image pyramids with hierarchical supervision."

### ESRGAN: Enhanced Multi-Scale Architecture

Enhanced SRGAN (ESRGAN) improves upon SRGAN with:
1. **Residual-in-Residual Dense Block (RRDB)**: Deeper, more expressive generator
2. **Relativistic discriminator**: Predicts whether real images are more realistic than fake
3. **Network interpolation**: Blend PSNR-optimized and GAN-optimized models

From [ESRGAN paper](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) (ECCV 2018):
> "To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss - leading to an improved ESRGAN model."

**RealESRGAN** extends ESRGAN for real-world degradation:
- **High-order degradation model**: Blur → downsample → noise → JPEG compression
- **Spectral normalization**: Stabilizes discriminator training
- **U-Net discriminator with skip connections**: Better gradient flow

---

## Section 4: Progressive Super-Resolution

### ProGAN-Style Progressive Training

Progressive Growing of GANs (ProGAN) gradually increases resolution during training, starting from 4×4 and progressively adding layers to reach 1024×1024. This strategy applies to SR:

```
Stage 1: Train 64×64 → 128×128 SR (10k iterations)
Stage 2: Add layers, train 128×128 → 256×256 SR (10k iterations)
Stage 3: Add layers, train 256×256 → 512×512 SR (10k iterations)
```

**Benefits**:
- **Training stability**: Easier to learn coarse structure first, then refine details
- **Faster convergence**: Early stages converge quickly on low-resolution data
- **Memory efficiency**: Can use larger batch sizes for low-resolution stages

**Progressive upsampling implementation**:
```python
class ProgressiveSRGenerator(nn.Module):
    def __init__(self, max_scale=8):
        self.stages = nn.ModuleList([
            UpsampleStage(in_ch=64, out_ch=64) for _ in range(int(np.log2(max_scale)))
        ])
        self.alpha = 1.0  # Blending factor for new layers

    def forward(self, lr_image, target_stage):
        x = lr_image
        for i, stage in enumerate(self.stages[:target_stage]):
            x = stage(x)
            if i == target_stage - 1:
                # Blend with skip connection (fade in new layer)
                x_skip = F.interpolate(lr_image, scale_factor=2**target_stage)
                x = self.alpha * x + (1 - self.alpha) * x_skip
        return x

    def grow(self):
        """Add next stage and reset alpha for smooth transition."""
        self.alpha = 0.0  # Fade in new layer
```

**Training schedule**:
1. Start with stage 1 (2× SR), train to convergence
2. Add stage 2 (4× SR), set alpha=0.0 (initially rely on skip connections)
3. Linearly increase alpha from 0→1 over 5k iterations (fade in new layers)
4. Continue training to convergence
5. Repeat for subsequent stages

### Adaptive Quality Deployment

Progressive SR models enable adaptive quality serving:

```python
def adaptive_sr_inference(lr_image, user_bandwidth='high'):
    """Serve appropriate SR quality based on bandwidth."""
    if user_bandwidth == 'low':
        return progressive_model.forward(lr_image, target_stage=1)  # 2× SR
    elif user_bandwidth == 'medium':
        return progressive_model.forward(lr_image, target_stage=2)  # 4× SR
    else:  # high bandwidth
        return progressive_model.forward(lr_image, target_stage=3)  # 8× SR
```

**Deployment advantages**:
- **Single model, multiple outputs**: No need to train separate 2×, 4×, 8× models
- **Adaptive streaming**: Serve low-res SR first, progressively refine as bandwidth allows
- **Edge deployment**: Coarse stages (smaller models) can run on edge devices

### Training Stability Benefits

Progressive training improves stability compared to direct high-scale SR:

**Gradient flow**:
- Early stages establish coarse structure → provides good initialization for later stages
- Avoids mode collapse common in direct 8× or 16× SR training

**Discriminator capacity**:
- Low-resolution discriminators are easier to train (fewer pixels to evaluate)
- Progressively increasing resolution keeps discriminator challenged but not overwhelmed

**Loss landscape smoothness**:
- Coarse stages optimize smoother loss landscapes (fewer local minima)
- Fine stages refine within already-good solution space

From [Hierarchical Frequency-Based Upsampling and Refining](https://dl.acm.org/doi/10.1109/TCSVT.2024.3517840) (IEEE TCSVT 2024):
> "Coarse-to-fine transfer through progressive pyramid structures enables stable training and fast convergence. HIR framework introduces hierarchical refinement for improved video super-resolution with enhanced propagation."

---

## Connections to VLM and ARR-COC

### Pyramid SR for VLM Image Preprocessing

Vision-language models benefit from high-quality image inputs. Pyramid-guided SR can:

1. **Upscale low-resolution images**: Improve OCR accuracy, object detection in VLMs
2. **Restore degraded images**: Remove compression artifacts, blur from web scraping
3. **Adaptive resolution**: Allocate SR compute to relevant image regions (ARR-COC relevance-guided SR)

**Integration example**:
```python
# VLM pipeline with pyramid SR preprocessing
def vlm_inference_with_sr(image_low_res, query):
    # 1. Relevance-guided SR (allocate SR budget to query-relevant regions)
    relevance_map = arr_coc_relevance_scorer(image_low_res, query)
    sr_image = pyramid_sr_selective(image_low_res, relevance_map)

    # 2. VLM encoding
    image_tokens = vlm_vision_encoder(sr_image)
    response = vlm_language_model(query, image_tokens)

    return response
```

### Pyramid LOD + Super-Resolution

Combine pyramid LOD (Section 3: Attention-Driven Pyramid Pruning) with SR:

```
Low-res pyramid:
    64×64 (coarse context)
    128×128 (medium detail)
    256×256 (not computed yet)

Query: "What is the small text in the bottom-right?"

ARR-COC relevance realization:
    → Bottom-right region has high participatory relevance
    → Apply pyramid SR to upsample 128×128 → 256×256 ONLY in bottom-right
    → Saves compute: 75% of image stays at 128×128, only 25% upsampled

Result: 512×512 SR in bottom-right, 128×128 elsewhere
```

**Efficiency gains**:
- **Selective SR**: Only upsample query-relevant regions (50-80% compute savings)
- **Quality where it matters**: High perceptual quality in regions needed for query
- **Multi-scale token budgets**: Coarse levels = fewer tokens, fine SR levels = more tokens

**See also**:
- [pyramid-lod/03-attention-driven-pyramid-pruning.md](03-attention-driven-pyramid-pruning.md) - ARR-COC relevance → pyramid LOD mapping
- [practical-implementation/51-vision-token-budgets.md](../practical-implementation/51-vision-token-budgets.md) - Token allocation strategies
- [karpathy/gpu-texture-optimization/09-neural-block-compression-vlm.md](../karpathy/gpu-texture-optimization/09-neural-block-compression-vlm.md) - Neural compression + VLMs

---

## Implementation Notes

### Training Recipe for Pyramid-Guided SR

**1. Multi-stage training**:
```python
# Stage 1: Train with MSE loss only (PSNR optimization)
for epoch in range(100):
    sr = generator(lr_images)
    loss = F.mse_loss(sr, hr_images)
    loss.backward()

# Stage 2: Add perceptual loss
for epoch in range(100, 200):
    sr = generator(lr_images)
    loss = 0.01 * F.mse_loss(sr, hr_images) + perceptual_loss(sr, hr_images, vgg)
    loss.backward()

# Stage 3: Add adversarial loss (GAN training)
for epoch in range(200, 300):
    sr = generator(lr_images)
    loss = (
        0.01 * F.mse_loss(sr, hr_images) +
        perceptual_loss(sr, hr_images, vgg) +
        0.001 * adversarial_loss(discriminator(sr))
    )
    loss.backward()
```

**2. Pyramid loss scheduling**:
```python
# Weight schedule: gradually increase fine-level weight
def pyramid_loss_weight(level, epoch):
    base_weight = 0.5 ** (2 - level)  # [0.25, 0.5, 1.0] for 3 levels
    schedule = min(1.0, epoch / 50)   # Ramp up over 50 epochs
    return base_weight * schedule
```

**3. Hyperparameters (based on LapSRN/SRGAN)**:
- Learning rate: 1e-4 (generator), 1e-4 (discriminator)
- Batch size: 16 (256×256 patches)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Loss weights: λ_percept=1.0, λ_adv=0.001, λ_mse=0.01

### Code Example: Minimal Pyramid SR

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidSRNet(nn.Module):
    """Minimal pyramid super-resolution network."""

    def __init__(self, scale=4, num_features=64):
        super().__init__()
        self.scale = scale
        self.num_stages = int(np.log2(scale))

        # Initial feature extraction
        self.feature_extractor = nn.Conv2d(3, num_features, 3, padding=1)

        # Progressive upsampling stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.PReLU(),
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.PReLU(),
                nn.ConvTranspose2d(num_features, num_features, 4, stride=2, padding=1)
            ) for _ in range(self.num_stages)
        ])

        # Residual predictors (Laplacian pyramid)
        self.residual_predictors = nn.ModuleList([
            nn.Conv2d(num_features, 3, 3, padding=1)
            for _ in range(self.num_stages)
        ])

    def forward(self, lr_image):
        """
        Args:
            lr_image: (B, 3, H, W) low-resolution input

        Returns:
            sr_image: (B, 3, H*scale, W*scale) super-resolved output
            pyramid: List of intermediate SR images at each scale
        """
        # Initial features
        features = self.feature_extractor(lr_image)
        current_image = lr_image
        pyramid = []

        # Progressive upsampling
        for stage, residual_pred in zip(self.stages, self.residual_predictors):
            # Upsampling + feature refinement
            features = stage(features)

            # Predict residual (high-frequency details)
            residual = residual_pred(features)

            # Upsample current image + add residual
            current_image = F.interpolate(current_image, scale_factor=2, mode='bilinear')
            current_image = current_image + residual

            pyramid.append(current_image)

        return current_image, pyramid

# Training example
model = PyramidSRNet(scale=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for lr_batch, hr_batch in dataloader:
    sr, pyramid = model(lr_batch)

    # Multi-scale loss
    loss = 0.0
    for i, sr_level in enumerate(pyramid):
        hr_level = F.interpolate(hr_batch, scale_factor=2**(i+1) / 4)
        loss += F.mse_loss(sr_level, hr_level) * (0.5 ** (2 - i))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Sources

**Web Research**:
- [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://ar5iv.labs.arxiv.org/html/1710.01992) - LapSRN architecture (CVPR 2017)
- [Laplacian Pyramid Super-Resolution Network Paper](https://www.wslai.net/publications/lapsrn/papers/cvpr17_LapSRN.pdf) - Original LapSRN paper (accessed 2025-01-31)
- [SRGAN - Super Resolution GAN](https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec) - SRGAN explanation (Medium, 2018)
- [GAN-Based Super-Resolution with Enhanced Multi-Scale Laplacian Pyramid](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70028) - Recent Laplacian pyramid methods (IET 2025)
- [Enhanced Pyramidal Residual Networks](https://link.springer.com/article/10.1007/s00521-024-09702-1) - EDSR variants (Springer 2024)
- [Hierarchical Frequency-Based Upsampling and Refining](https://dl.acm.org/doi/10.1109/TCSVT.2024.3517840) - Coarse-to-fine SR (IEEE TCSVT 2024)
- [Progressive Super-Resolution](https://www.researchgate.net/scientific-contributions/Byungkon-Kang-2140412846) - Multi-scale discriminator approaches (ResearchGate)

**Oracle Knowledge Base**:
- [pyramid-lod/03-attention-driven-pyramid-pruning.md](03-attention-driven-pyramid-pruning.md) - ARR-COC relevance → pyramid LOD
- [pyramid-lod/06-differentiable-pyramid-operators.md](06-differentiable-pyramid-operators.md) - Gradient flow through pyramids
- [practical-implementation/51-vision-token-budgets.md](../practical-implementation/51-vision-token-budgets.md) - Token allocation strategies
- [karpathy/biological-vision/03-foveated-rendering-peripheral.md](../karpathy/biological-vision/03-foveated-rendering-peripheral.md) - Biological vision inspiration
- [vision-language/02-image-encoder-architectures.md](../vision-language/02-image-encoder-architectures.md) - ViT architectures for VLMs

**Additional References**:
- SRGAN paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf) (CVPR 2017)
- ESRGAN paper: [Enhanced Super-Resolution Generative Adversarial Networks](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) (ECCV 2018)
- RealESRGAN: Real-world degradation models for practical SR applications

---

**Document Status**: Complete (300 lines)
**Last Updated**: 2025-01-31
**Part of**: Pyramid LOD & Hierarchical Vision expansion (PART 8 of 10)
