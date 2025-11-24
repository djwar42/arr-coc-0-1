# Laplacian and Gaussian Pyramids in Deep Learning

## Overview

Laplacian and Gaussian pyramids are classical multi-resolution image representations that decompose images into hierarchical frequency bands. Originally developed by Burt and Adelson in 1983 for image compression and blending, these pyramids have found renewed relevance in modern deep learning for edge-preserving upsampling, progressive refinement networks, and multi-scale feature extraction.

**Classical pyramid construction:**
- **Gaussian pyramid**: Repeated downsampling with Gaussian blur (low-pass filtering)
- **Laplacian pyramid**: Band-pass representation (difference of Gaussians between pyramid levels)

**Modern neural network integration:**
- Laplacian pyramid decoders for super-resolution and image synthesis
- Progressive refinement networks (coarse-to-fine generation)
- Edge-preserving upsampling in encoder-decoder architectures
- Multi-scale loss functions for texture recovery

From [Foundations of Computer Vision (MIT)](https://visionbook.mit.edu/pyramids_new_notation.html) (accessed 2025-01-31):
- The Laplacian pyramid represents what is present in a Gaussian pyramid image of one level but not present at the level below
- Efficient storage: Only 1/3 additional storage cost for entire pyramid
- Enables fast coarse-to-fine search strategies

**Key advantage**: Unlike standard bilinear upsampling, Laplacian pyramid reconstruction preserves high-frequency edge details while progressively refining coarse predictions.

## Gaussian Pyramid: Downsampling with Blur

The Gaussian pyramid creates a multi-resolution image hierarchy through iterative blur-and-downsample operations:

**Construction algorithm:**
```python
def gaussian_pyramid(image, levels=4):
    """
    Build Gaussian pyramid by repeated Gaussian blur + downsampling
    """
    pyramid = [image]
    current = image

    for i in range(levels - 1):
        # Gaussian blur (5x5 kernel typical)
        blurred = gaussian_blur(current, kernel_size=5, sigma=1.0)

        # Downsample by factor of 2
        downsampled = downsample(blurred, factor=2)
        pyramid.append(downsampled)
        current = downsampled

    return pyramid
```

**Properties:**
- Each level reduces resolution by factor of 2 (1/4 area)
- Gaussian blur removes high frequencies before downsampling (anti-aliasing)
- Level k has dimensions: (H/2^k, W/2^k)
- Storage cost: 1 + 1/4 + 1/16 + ... ≈ 4/3 of original image

From [Lei Mao's Image Pyramids in Deep Learning](https://leimao.github.io/blog/Image-Pyramids-In-Deep-Learning/) (accessed 2025-01-31):
- **Coarse-to-fine search advantage**: Looking for an object (e.g., hippo's eye) in low-resolution first, then refining in high-resolution significantly reduces search translations
- Storage cost increases by only 1/3 at most with 1/2 dimension reduction per level

**Applications in neural encoders:**

**1. Multi-scale feature extraction:**
```python
class GaussianPyramidEncoder(nn.Module):
    def __init__(self, in_channels, pyramid_levels=4):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_channels, 64 * (2**i))
            for i in range(pyramid_levels)
        ])

    def forward(self, x):
        pyramid = []
        current = x

        for i, conv_block in enumerate(self.conv_blocks):
            # Process current resolution
            features = conv_block(current)
            pyramid.append(features)

            # Downsample for next level
            if i < self.pyramid_levels - 1:
                current = F.avg_pool2d(current, kernel_size=2)

        return pyramid
```

**2. Input pyramid augmentation:**
- Train network on multiple resolutions simultaneously
- Improves robustness to scale variations
- Used in detection networks (e.g., SSD, RetinaNet)

**3. Spatial pyramid pooling (SPP):**
From [Lei Mao's blog](https://leimao.github.io/blog/Image-Pyramids-In-Deep-Learning/):
- Kaiming He's SPP: Parallel max pooling (4×4, 2×2, 1×1) → linearize & concatenate
- Creates fixed-length feature vector independent of input size
- Rich spatial information improves vision task accuracy

## Laplacian Pyramid: Band-Pass Edge Representation

The Laplacian pyramid captures high-frequency edge details lost during Gaussian downsampling:

**Construction via difference-of-Gaussians:**
```python
def laplacian_pyramid(image, levels=4):
    """
    Build Laplacian pyramid from Gaussian pyramid differences
    """
    gaussian_pyr = gaussian_pyramid(image, levels)
    laplacian_pyr = []

    for i in range(levels - 1):
        # Upsample lower resolution to match higher resolution
        upsampled = upsample(gaussian_pyr[i + 1], size=gaussian_pyr[i].shape)

        # Difference captures high-frequency details
        laplacian = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(laplacian)

    # Highest pyramid level (coarsest) is residual
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr
```

**Reconstruction (perfect reconstruction property):**
```python
def reconstruct_from_laplacian(laplacian_pyramid):
    """
    Reconstruct original image from Laplacian pyramid
    """
    # Start from coarsest level
    reconstructed = laplacian_pyramid[-1]

    # Progressively add high-frequency details
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = upsample(reconstructed, size=laplacian_pyramid[i].shape)
        reconstructed = upsampled + laplacian_pyramid[i]

    return reconstructed
```

**Key properties:**
- Band-pass filter: Each level contains specific frequency range
- Perfect reconstruction: Sum of pyramid recovers original image
- Sparse representation: Most coefficients near zero except at edges
- Edge-preserving: High-frequency details explicitly stored

**Laplacian pyramid layers in PyTorch:**

From research on [Deep Laplacian Pyramid Networks (Lai et al., CVPR 2017)](https://arxiv.org/abs/1704.03915) and [SWDL paper](https://arxiv.org/abs/2506.10325):

```python
class LaplacianPyramidLayer(nn.Module):
    """
    Laplacian pyramid layer for edge-preserving upsampling
    """
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, low_res, high_res_residual):
        """
        low_res: Coarse prediction from lower pyramid level
        high_res_residual: High-frequency details at this level
        """
        # Upsample coarse prediction
        upsampled = self.upsample(low_res)

        # Concatenate with residual details
        combined = torch.cat([upsampled, high_res_residual], dim=1)

        # Refine with learned filter
        refined = self.refine(combined) + upsampled

        return refined
```

**Applications in CNNs (2018-2024):**

**1. Deep Laplacian Pyramid Networks (LapSRN, 2017):**
From [Lai et al., CVPR 2017](https://arxiv.org/abs/1704.03915):
- Progressive super-resolution: 2× → 4× → 8× upsampling
- Multiple intermediate SR predictions in one feedforward pass
- Laplacian pyramid reconstruction at each level
- Charbonnier loss at multiple scales

**2. Laplacian Pyramid GANs (LAPGAN, Denton et al., NIPS 2015):**
From [Deep Generative Image Models using Laplacian Pyramid](http://papers.neurips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf):
- Cascade of convolutional networks within adversarial framework
- Generate images in coarse-to-fine manner
- Each level generates residual to refine lower level
- High-quality natural image generation

**3. Medical image fusion (Fu et al., 2020):**
From research on [Multimodal medical image fusion (Computers in Biology and Medicine)](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303796):
- Laplacian pyramid decomposes source images
- CNN reconstruction fuses pyramid coefficients
- Local gradient energy guides fusion
- Preserves both anatomical and functional information

**4. Lightweight image deraining (LPNet, Fu et al., 2018):**
From [Lightweight Pyramid Networks (arXiv)](https://arxiv.org/abs/1805.06173):
- Lightweight pyramid architecture for single image deraining
- Progressive refinement from coarse to fine
- 465 citations, widely adopted for degradation removal

**PyTorch implementation example:**

```python
class LaplacianPyramidDecoder(nn.Module):
    """
    Laplacian pyramid decoder for progressive upsampling
    """
    def __init__(self, channels=[512, 256, 128, 64], num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        self.residual_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i], channels[i], 3, padding=1)
            )
            for i in range(num_levels)
        ])

        self.laplacian_layers = nn.ModuleList([
            LaplacianPyramidLayer(channels[i])
            for i in range(num_levels - 1)
        ])

    def forward(self, encoder_features):
        """
        encoder_features: List of features from encoder (coarse to fine)
        Returns: Progressively refined outputs
        """
        outputs = []

        # Start from coarsest level
        x = self.residual_predictors[-1](encoder_features[-1])
        outputs.append(x)

        # Progressive refinement
        for i in range(self.num_levels - 2, -1, -1):
            residual = self.residual_predictors[i](encoder_features[i])
            x = self.laplacian_layers[i](x, residual)
            outputs.append(x)

        return outputs[::-1]  # Fine to coarse
```

**Hyperparameters:**
- **Pyramid levels**: Typically 3-5 levels (balance detail vs. computation)
- **Blur kernel**: Gaussian 5×5, σ=1.0 standard
- **Upsampling**: Bilinear or learned transposed convolution
- **Loss weighting**: Higher weight for finer levels (preserve detail)

## Progressive Refinement Networks

Progressive refinement networks use Laplacian pyramid structure for coarse-to-fine generation:

**Architecture paradigm:**
```
Input → Coarse Prediction → Refine 2× → Refine 4× → Refine 8× → Final Output
          (64×64)              (128×128)   (256×256)   (512×512)
```

**Key papers and applications:**

**1. ProGAN (Progressive Growing of GANs, Karras et al., ICLR 2018):**
From research on [progressive refinement networks](https://www.semanticscholar.org/paper/Deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks):
- Progressive training: Start 4×4, gradually add layers to 1024×1024
- Fade-in layers for smooth transitions
- Training stability at high resolutions
- StyleGAN built on this foundation

**2. Cascaded Refinement Networks (Chen & Koltun, ICCV 2017):**
From [Photographic Image Synthesis With Cascaded Refinement Networks](https://arxiv.org/abs/1707.09405):
- 1206 citations, foundational work
- Synthesize photographic images from semantic layouts
- Coarse-to-fine refinement modules
- Scales to 2048×1024 resolution

**3. LP-Net for depth completion (Wang et al., 2025):**
From [Learning Inverse Laplacian Pyramid (arXiv:2502.07289)](https://arxiv.org/abs/2502.07289):
- Multi-scale progressive prediction paradigm
- Inverse Laplacian pyramid for depth completion
- Cited 6 times (recent work)
- Addresses sparse depth input challenges

**4. LLF-LUT++ for photo enhancement (2025):**
From [High-Resolution Photo Enhancement in Real-time](https://arxiv.org/html/2510.11613v1):
- Pyramid network integrating global and local operators
- Closed-form Laplacian pyramid decomposition
- Real-time high-resolution enhancement
- Look-Up Table (LUT) optimization

**Training strategies:**

**Multi-scale loss function:**
```python
def progressive_refinement_loss(predictions, target):
    """
    Multi-scale loss for progressive refinement
    predictions: List of outputs at different scales
    target: Ground truth high-resolution image
    """
    losses = []
    weights = [0.1, 0.2, 0.3, 0.4]  # Increasing weight for finer scales

    for i, pred in enumerate(predictions):
        # Downsample target to match prediction scale
        target_scaled = F.interpolate(target, size=pred.shape[-2:], mode='bilinear')

        # Charbonnier loss (robust to outliers)
        loss = torch.sqrt((pred - target_scaled)**2 + 1e-6).mean()
        losses.append(weights[i] * loss)

    return sum(losses)
```

**Progressive training schedule:**
- Stage 1: Train coarsest level only (fast convergence)
- Stage 2: Add next refinement level, fine-tune
- Stage 3: Add next level, fine-tune
- Final: End-to-end fine-tuning of all levels

**Applications:**

**Super-resolution:**
- LapSRN: 2×→4×→8× progressive upsampling
- MS-LapSRN: Multi-scale training with shared features
- Fast inference: Early exit for lower resolutions

**Image synthesis:**
- Semantic → photographic conversion
- Coarse layout → refined details
- Style transfer with detail preservation

**Inpainting:**
- Coarse structure completion
- Progressive texture refinement
- Edge-preserving hole filling

**Deraining/denoising:**
- Coarse rain removal
- Fine detail recovery
- Multi-scale degradation handling

**Advantages over single-scale:**
- **Training stability**: Easier to learn coarse structure first
- **Faster convergence**: Curriculum learning effect
- **Better quality**: Explicit high-frequency modeling
- **Interpretability**: Visualize intermediate predictions

## Implementation: PyTorch Laplacian Pyramid Module

Complete working implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    """
    Gaussian blur for pyramid construction
    """
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.repeat(channels, 1, 1, 1)

        self.register_buffer('weight', kernel)
        self.padding = kernel_size // 2

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Generate 2D Gaussian kernel"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)

class LaplacianPyramid(nn.Module):
    """
    Differentiable Laplacian pyramid for deep learning
    """
    def __init__(self, num_levels=4, channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.blur = GaussianBlur(channels)

    def build_gaussian_pyramid(self, x):
        """Build Gaussian pyramid"""
        pyramid = [x]
        for _ in range(self.num_levels - 1):
            x = self.blur(x)
            x = F.avg_pool2d(x, kernel_size=2)
            pyramid.append(x)
        return pyramid

    def build_laplacian_pyramid(self, x):
        """Build Laplacian pyramid"""
        gaussian_pyr = self.build_gaussian_pyramid(x)
        laplacian_pyr = []

        for i in range(self.num_levels - 1):
            upsampled = F.interpolate(
                gaussian_pyr[i + 1],
                size=gaussian_pyr[i].shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            laplacian = gaussian_pyr[i] - upsampled
            laplacian_pyr.append(laplacian)

        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    def reconstruct(self, laplacian_pyramid):
        """Reconstruct image from Laplacian pyramid"""
        x = laplacian_pyramid[-1]

        for i in range(self.num_levels - 2, -1, -1):
            x = F.interpolate(
                x,
                size=laplacian_pyramid[i].shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            x = x + laplacian_pyramid[i]

        return x

    def forward(self, x):
        """Build and reconstruct (for testing)"""
        lap_pyr = self.build_laplacian_pyramid(x)
        reconstructed = self.reconstruct(lap_pyr)
        return lap_pyr, reconstructed

# Example usage
if __name__ == "__main__":
    # Create pyramid
    lap_pyramid = LaplacianPyramid(num_levels=4, channels=3)

    # Test image
    image = torch.randn(1, 3, 256, 256)

    # Build pyramid
    pyramid, reconstructed = lap_pyramid(image)

    # Check reconstruction error
    error = (image - reconstructed).abs().mean()
    print(f"Reconstruction error: {error.item():.6f}")

    # Check pyramid sizes
    for i, level in enumerate(pyramid):
        print(f"Level {i}: {level.shape}")
```

**Key hyperparameters:**
- `num_levels`: Typically 3-5 (balance detail vs. memory)
- `kernel_size`: 5×5 Gaussian standard
- `sigma`: 1.0 typical, higher = more blur
- `upsample_mode`: 'bilinear' standard, 'nearest' faster

**Integration with encoder-decoder:**
```python
class LapPyramidUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNetwork()
        self.decoder = LaplacianPyramidDecoder()
        self.lap_pyramid = LaplacianPyramid(num_levels=4)

    def forward(self, x):
        # Extract multi-scale features
        features = self.encoder(x)

        # Progressive refinement
        predictions = self.decoder(features)

        # Optional: Laplacian loss at each scale
        losses = []
        for pred in predictions:
            target_pyramid = self.lap_pyramid.build_laplacian_pyramid(x)
            pred_pyramid = self.lap_pyramid.build_laplacian_pyramid(pred)

            # L1 loss on pyramid coefficients
            loss = sum([
                (t - p).abs().mean()
                for t, p in zip(target_pyramid, pred_pyramid)
            ])
            losses.append(loss)

        return predictions, losses
```

## Sources

**Source Documents:**
- None (pure web research task)

**Web Research:**

**Classical foundations:**
- [MIT Foundations of Computer Vision: Image Pyramids](https://visionbook.mit.edu/pyramids_new_notation.html) - Mathematical foundation, Laplacian pyramid definition (accessed 2025-01-31)
- [Lei Mao: Image Pyramids and Its Applications in Deep Learning](https://leimao.github.io/blog/Image-Pyramids-In-Deep-Learning/) - Comprehensive tutorial on pyramid methods (accessed 2025-01-31)

**Key papers (via arXiv/Google Scholar):**
- [Deep Generative Image Models using Laplacian Pyramid (Denton et al., NIPS 2015)](http://papers.neurips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf) - LAPGAN foundational work (3221 citations)
- [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution (Lai et al., CVPR 2017)](https://arxiv.org/abs/1704.03915) - LapSRN progressive refinement (3549 citations)
- [Photographic Image Synthesis With Cascaded Refinement Networks (Chen & Koltun, ICCV 2017)](https://arxiv.org/abs/1707.09405) - Cascaded refinement (1206 citations)
- [Lightweight Pyramid Networks for Image Deraining (Fu et al., 2018)](https://arxiv.org/abs/1805.06173) - LPNet deraining application (465 citations)

**Recent work (2020-2025):**
- [Multimodal medical image fusion via laplacian pyramid (Fu et al., 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010482520303796) - Medical imaging application (64 citations)
- [Learning Inverse Laplacian Pyramid for Progressive Depth (Wang et al., 2025)](https://arxiv.org/abs/2502.07289) - Recent depth completion (6 citations)
- [SWDL: Stratum-Wise Difference Learning (2025)](https://arxiv.org/abs/2506.10325) - Deep Laplacian upsampling
- [High-Resolution Photo Enhancement in Real-time (2025)](https://arxiv.org/html/2510.11613v1) - LLF-LUT++ pyramid network

**Additional references:**
- [Laplacian Pyramid Reconstruction and Refinement (ar5iv)](https://ar5iv.labs.arxiv.org/html/1605.02264) - Multi-resolution reconstruction architecture
- [Laplacian pyramid-based complex neural network learning (PMLR)](http://proceedings.mlr.press/v121/liang20a/liang20a.pdf) - CLP-Net for MRI (7 citations)
- [Progressive Refinement Imaging (Wiley, 2020)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13808) - Adaptive Laplacian pyramid for 3D

**Historical context:**
- E.H. Adelson and C.H. Anderson et al., "Pyramid methods in image processing" (1984) - Original pyramid formulation
- P.J. Burt and E.H. Adelson, "The Laplacian pyramid as a compact image code" (1983) - Foundational paper
