# Coarse-to-Fine Architectures in Neural Networks

## Overview

Coarse-to-fine architectures represent a fundamental paradigm in neural network design where models progressively refine their predictions or training from low-resolution/coarse representations toward high-resolution/fine-grained outputs. This approach mirrors classical computer vision techniques while leveraging modern deep learning capabilities for improved training stability, faster convergence, and superior final performance.

**Core Principle**: Start simple, add complexity gradually. Train on easy tasks first (low resolution, coarse features), then progressively introduce harder challenges (high resolution, fine details).

**Key Benefits**:
- **Training Stability**: Gradual complexity prevents mode collapse in GANs and gradient instabilities
- **Faster Convergence**: Lower-resolution training requires fewer FLOPs, enabling faster initial learning
- **Better Generalization**: Curriculum from easy→hard improves model robustness
- **Resource Efficiency**: Saves compute during early training stages

**Primary Application Domains**:
- Image synthesis (ProGAN, StyleGAN)
- Optical flow estimation (PWC-Net, RAFT)
- Depth estimation (multi-scale pyramids)
- Video generation and prediction
- Object detection (feature pyramids with refinement)

---

## Progressive GAN Training (ProGAN)

### Architecture and Method

From [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) (Karras et al., ICLR 2018):

**Core Innovation**: Grow both generator and discriminator progressively, starting from 4×4 resolution and doubling until reaching 1024×1024.

**Training Schedule**:
```
Stage 1: 4×4    (G: 4×4 → 4×4,   D: 4×4 → score)
Stage 2: 8×8    (G: 4×4 → 8×8,   D: 8×8 → score)
Stage 3: 16×16  (G: 4×4 → 16×16, D: 16×16 → score)
Stage 4: 32×32  (continued...)
Stage 5: 64×64
Stage 6: 128×128
Stage 7: 256×256
Stage 8: 512×512
Stage 9: 1024×1024
```

**Smooth Transition (Fade-In Layers)**:
When adding new resolution layer, use linear interpolation parameter α ∈ [0,1]:
```
Output = (1-α) * Upsample(previous_layer) + α * new_layer
```
- α=0: Only use upsampled previous layer (no new layer yet)
- α increases linearly over K iterations
- α=1: Fully use new layer (transition complete)

This smooth blending prevents sudden training shocks and maintains stability.

### Training Procedure

**Phase 1: Stabilization** (at current resolution)
- Train both G and D at current resolution until convergence
- Typical: 600k-800k images shown
- Monitor FID (Fréchet Inception Distance) for quality

**Phase 2: Transition** (add next resolution)
- Introduce new layers with fade-in (α: 0→1)
- Typical: 600k-800k images with gradual α increase
- Both networks grow simultaneously

**Phase 3: Repeat**
- Once new resolution stabilized, add next layer
- Continue until target resolution reached

**Implementation Detail (PyTorch-style)**:
```python
class ProgressiveGenerator(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            Block_4x4(),   # Base: always active
            Block_8x8(),   # Stage 2
            Block_16x16(), # Stage 3
            # ... up to 1024x1024
        ])

    def forward(self, z, stage, alpha):
        x = self.blocks[0](z)  # 4x4 base

        for i in range(1, stage):
            x = self.blocks[i](x)  # Grow through stages

        if stage < len(self.blocks):
            # Transition: blend old (upsampled) with new
            x_old = F.interpolate(x, scale_factor=2)
            x_new = self.blocks[stage](x)
            x = (1 - alpha) * x_old + alpha * x_new

        return to_rgb(x)
```

### Results and Impact

**CelebA-HQ Dataset** (1024×1024 faces):
- ProGAN: First to generate photorealistic 1024² faces
- Training time: ~2 weeks on 8 Tesla V100 GPUs
- vs. Standard GAN: Would diverge or mode-collapse at high resolutions

**CIFAR-10** (32×32 natural images):
- Inception Score: 8.80 (record at publication)
- Previous best: ~7.90

**Key Insight**: Progressive growing is not just about resolution—it's about curriculum learning for adversarial training. The discriminator learns coarse features first (face shape, color palette) before fine details (individual hairs, skin texture).

From [ProGAN paper](https://arxiv.org/abs/1710.10196) (accessed 2025-01-31):
> "The key observation is that by starting with low-resolution images, the generator and discriminator are presented with a drastically simpler task. Adding new layers then allows them to progressively refine the results."

---

## StyleGAN: Progressive Training Evolution

### Progressive Training in StyleGAN v1

From [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) (Karras et al., CVPR 2019):

StyleGAN v1 retained ProGAN's progressive training but added:
- **Style-based generator**: Latent code modulates each resolution via AdaIN
- **Progressive training schedule**: Identical to ProGAN (4×4 → 1024×1024)
- **Improved architecture**: Removed traditional input layer, used learned constant

**Training modifications**:
- Still uses fade-in for smooth transitions
- Each resolution stage: ~600k real images shown
- Total training: ~25M images for FFHQ-1024

### StyleGAN v2: Moving Beyond Progressive Training

From [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) (Karras et al., CVPR 2020):

**Major Change**: StyleGAN v2 **removed progressive growing** in favor of:
- **MSG-GAN approach**: Multi-scale gradient flow from all resolutions
- **Skip connections**: RGB outputs at every resolution, all supervised
- **Residual discriminator**: Processes all scales simultaneously

**Why abandon progressive training?**
- ProGAN creates resolution-dependent artifacts (blob-like features at specific scales)
- Phase-dependent training complicates the learning dynamics
- Modern architectures (residual connections, normalization) can handle direct high-res training

**StyleGAN v2 Training**:
```python
# No progressive growing - train at target resolution from start
# But use multi-scale discrimination:
def discriminator_forward(img_1024):
    scores = []
    x = img_1024
    for block in blocks:
        x = block(x)  # Process and downsample
        scores.append(classify(x))  # Score at this scale

    return sum(scores)  # Multi-scale loss
```

**Results**: StyleGAN v2 eliminates characteristic artifacts while maintaining quality, proving progressive training is not always necessary with better architectures.

From [StyleGAN v2 paper](https://arxiv.org/abs/1912.04958) (accessed 2025-01-31):
> "Progressive growing has been key to StyleGAN's success. However, we show that it is the cause of characteristic blob-like artifacts... We redesign the generator to remove progressive growing while retaining its training benefits."

---

## Curriculum Learning with Resolution

### Theoretical Foundation

**Curriculum Learning Principle**: Train on easier examples first, gradually increase difficulty. In vision, "easy" often means low-resolution.

**Why Resolution Curriculum Works**:
- **Fewer pixels** = smaller search space = easier optimization
- **Coarse features** learned first provide strong prior for fine details
- **Prevents overfitting** to high-frequency noise early in training

### Progressive Resolution Training (Modern DNNs)

From [Progressive learning: A deep learning framework for continual learning](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301817) (Fayek et al., Neural Networks 2020):

**Training Schedule for Classification/Recognition**:
```
Epochs   1-10:  Train on 64×64 images
Epochs  11-20:  Train on 128×128 images
Epochs  21-30:  Train on 256×256 images
Epochs  31-40:  Train on 512×512 images (target)
```

**Implementation Strategy**:
```python
def progressive_resolution_training(model, data, resolution_schedule):
    for epoch, target_res in resolution_schedule:
        # Resize dataset to target resolution
        data_res = resize_dataset(data, target_res)

        # Train for specified epochs
        for batch in data_res:
            loss = model(batch)
            loss.backward()
            optimizer.step()
```

**Benefits**:
- **Faster initial convergence**: 64×64 images are 64× fewer pixels than 512×512
- **Better feature learning**: Coarse features (object shape) before textures
- **Reduced overfitting**: Regularization effect from resolution curriculum

**Empirical Results** (ImageNet classification):
- Standard training (224×224 only): 75.2% top-1 accuracy, 90 epochs
- Progressive (96→128→160→224): 75.8% accuracy, 70 epochs
- 20% faster training + 0.6% accuracy improvement

### Multi-Scale Curriculum for Object Detection

From [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) (Lin et al., CVPR 2017) + curriculum extensions:

**Training Strategy**:
- **Stage 1**: Train RPN (Region Proposal Network) on single scale (e.g., 800px)
- **Stage 2**: Add multi-scale training (400px-1200px random)
- **Stage 3**: Fine-tune with full pyramid {P2, P3, P4, P5, P6}

**Benefits for Detection**:
- Small objects learned in early stages (appear larger at lower resolution)
- Large objects refined in later stages
- Prevents imbalance where detector ignores small objects

---

## Coarse-to-Fine Optimization: Optical Flow

### PWC-Net: Pyramid Warping and Cost Volume

From [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) (Sun et al., CVPR 2018):

**Architecture**: Classical coarse-to-fine pyramid with neural networks

**Pyramid Levels** (L6 → L1):
```
L6: 1/64 resolution (coarsest)
L5: 1/32 resolution
L4: 1/16 resolution
L3: 1/8 resolution
L2: 1/4 resolution
L1: 1/1 resolution (finest, original)
```

**Coarse-to-Fine Flow Estimation**:
```python
def pwc_net_forward(img1, img2):
    # Extract feature pyramids
    pyramid1 = feature_pyramid(img1)  # [L6, L5, ..., L1]
    pyramid2 = feature_pyramid(img2)

    flow = None
    for level in [6, 5, 4, 3, 2, 1]:  # Coarse to fine
        # 1. Warp img2 features using upsampled flow from previous level
        if flow is not None:
            flow_up = upsample(flow, scale=2)
            feat2_warped = warp(pyramid2[level], flow_up)
        else:
            feat2_warped = pyramid2[level]

        # 2. Compute cost volume (correlation between features)
        cost = correlation(pyramid1[level], feat2_warped)

        # 3. Estimate flow at this level
        flow_delta = flow_estimator(cost, pyramid1[level], flow_up)
        flow = flow_up + flow_delta  # Refine previous estimate

    return flow  # Final dense optical flow field
```

**Key Concepts**:
- **Warping**: Align img2 to img1 using current flow estimate before computing next level
- **Cost Volume**: 4D correlation tensor measuring feature similarity
- **Residual Refinement**: Each level predicts Δflow, not absolute flow

**Performance**:
- Sintel Final: 2.54 EPE (end-point error)
- KITTI 2015: 9.60% outlier rate
- 35 FPS on 1024×436 images (Titan X)

From [PWC-Net paper](https://arxiv.org/abs/1709.02371) (accessed 2025-01-31):
> "The coarse-to-fine approach is a classic strategy for optical flow. By estimating flow at multiple scales, we handle both large and small motions effectively."

### RAFT: Replacing Pyramids with Iterative Refinement

From [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) (Teed & Deng, ECCV 2020):

**Major Innovation**: RAFT operates on a **single high-resolution flow field** instead of coarse-to-fine pyramid.

**Why This Works**:
- **All-pairs correlation**: 4D correlation volume at full resolution (not downsampled)
- **Iterative refinement**: Update flow field through recurrent GRU 12-32 times
- **Multi-scale correlation lookup**: Index correlation volume at multiple resolutions simultaneously

**Architecture**:
```python
def raft_forward(img1, img2, iters=12):
    # Extract dense features at 1/8 resolution
    fmap1 = feature_encoder(img1)  # H/8 × W/8 × 256
    fmap2 = feature_encoder(img2)

    # Build 4D correlation pyramid (single compute, multi-scale lookup)
    corr_pyramid = all_pairs_correlation(fmap1, fmap2)
    # corr_pyramid = [full_res, 1/2, 1/4, 1/8] correlation volumes

    # Initialize flow field (zero)
    flow = torch.zeros(B, 2, H, W)

    # Iteratively refine flow
    for i in range(iters):
        # Lookup correlation features at current flow estimate
        corr_features = lookup_correlation(corr_pyramid, flow)
        # corr_features: multi-scale lookups at different radii

        # Update flow using GRU
        flow = flow + gru_update(corr_features, flow)

    return flow
```

**Coarse-to-Fine Elements in RAFT**:
- **Multi-scale correlation lookup**: Equivalent to pyramid (coarse features from large radius)
- **Iterative refinement**: Analogous to pyramid levels (12+ iterations = 12+ refinement stages)
- **Hierarchical lookups**: Sample correlation volume at radii {1, 2, 4, 8} = 4 scales

**Performance** (State-of-the-art):
- Sintel Clean: 1.43 EPE
- Sintel Final: 2.71 EPE
- KITTI 2015: 5.04% F1-all
- 10 FPS on 1024×436 (slower than PWC-Net, but far more accurate)

**Key Insight**: RAFT shows that explicit coarse-to-fine pyramids are not necessary if you have:
1. All-pairs correlation (captures all possible matches simultaneously)
2. Iterative refinement (implicit multi-scale processing)
3. Multi-scale lookup (hierarchical feature extraction)

From [RAFT paper](https://arxiv.org/abs/2003.12039) (accessed 2025-01-31):
> "By operating on a single high-resolution flow field, RAFT overcomes several limitations of a coarse-to-fine cascade: the difficulty of recovering from errors at coarse resolutions, and the inability to model small, fast-moving objects."

---

## Coarse-to-Fine Depth Estimation

### Pyramid-Based Depth Networks

**Classical Approach**: U-Net with pyramid decoder
```
Encoder: 256×256 → 128×128 → 64×64 → 32×32 → 16×16
Decoder: 16×16 → 32×32 → 64×64 → 128×128 → 256×256
                  ↓       ↓       ↓        ↓
          Depth: 1/16    1/8     1/4      1/2    1/1
```

**Multi-Scale Supervision**:
```python
def depth_loss(pred_pyramid, gt_depth):
    loss = 0
    for scale in [1/16, 1/8, 1/4, 1/2, 1/1]:
        pred = pred_pyramid[scale]
        gt = downsample(gt_depth, scale)
        loss += L1(pred, gt) + SSIM(pred, gt)
    return loss
```

**Benefits**:
- **Early depth estimates**: Coarse depth at 1/16 resolution converges faster
- **Pyramid consistency**: Encourages consistency across scales
- **Gradient flow**: Multiple supervision points improve gradients

### Coarse-to-Fine Stereo Matching

**PSMNet** (Pyramid Stereo Matching Network):
```
Stage 1: Coarse disparity (1/4 resolution)
         - Build cost volume at 1/4 scale
         - Regression: softargmin over disparity hypotheses

Stage 2: Refinement (1/2 resolution)
         - Upsample coarse disparity
         - Residual refinement module

Stage 3: Final (full resolution)
         - Upsample + final refinement
```

**Disparity Regression**:
- Coarse stage: Search range [0, D_max/4], step size = 1
- Refined stages: Search around upsampled coarse, smaller range

From [Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669) (Chang & Chen, CVPR 2018):
> "The coarse-to-fine strategy significantly reduces computational cost while maintaining accuracy. Coarse stages handle large disparities; fine stages correct details."

---

## Implementation Patterns

### Progressive Training Loop (General)

```python
def progressive_train(model, dataset, schedule):
    """
    schedule: [(resolution, epochs), ...]
    Example: [(64, 10), (128, 10), (256, 10), (512, 10)]
    """
    for resolution, epochs in schedule:
        print(f"Training at {resolution}×{resolution}")

        # Resize dataset to current resolution
        data_loader = get_dataloader(dataset, resolution)

        # Train for specified epochs
        for epoch in range(epochs):
            for batch in data_loader:
                loss = model(batch)
                loss.backward()
                optimizer.step()

            # Validate at target resolution periodically
            if epoch % 5 == 0:
                val_metric = validate(model, val_data, target_resolution=512)
                print(f"Epoch {epoch}, Val metric: {val_metric}")
```

### Smooth Transition (ProGAN-style)

```python
def train_with_fadeIn(G, D, data, current_stage, transition_imgs=800_000):
    """
    current_stage: 0=4x4, 1=8x8, 2=16x16, etc.
    transition_imgs: number of images for fade-in (α: 0→1)
    """
    alpha = 0.0
    alpha_step = 1.0 / transition_imgs

    for img_real in data:
        # Update alpha for smooth transition
        alpha = min(alpha + alpha_step, 1.0)

        # Generate fake image with current stage and alpha
        z = torch.randn(batch_size, latent_dim)
        img_fake = G(z, stage=current_stage, alpha=alpha)

        # Train discriminator
        score_real = D(img_real, stage=current_stage, alpha=alpha)
        score_fake = D(img_fake.detach(), stage=current_stage, alpha=alpha)
        loss_D = -torch.mean(score_real) + torch.mean(score_fake)
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        score_fake = D(img_fake, stage=current_stage, alpha=alpha)
        loss_G = -torch.mean(score_fake)
        loss_G.backward()
        optimizer_G.step()
```

### Pyramid Flow Estimation

```python
def pyramid_flow_estimation(pyramid1, pyramid2, num_levels=5):
    """
    pyramid1, pyramid2: List of feature maps [coarse, ..., fine]
    Returns: Dense optical flow at finest level
    """
    flow = None

    for level in range(num_levels):
        feat1 = pyramid1[level]
        feat2 = pyramid2[level]

        if flow is not None:
            # Upsample previous flow estimate (coarse → fine)
            flow = F.interpolate(flow, scale_factor=2, mode='bilinear')

            # Warp feat2 using current flow
            feat2_warped = warp(feat2, flow)
        else:
            feat2_warped = feat2

        # Compute cost volume (correlation)
        cost = compute_cost_volume(feat1, feat2_warped)

        # Estimate flow residual at this level
        flow_delta = flow_decoder(cost)

        # Update flow: previous + residual
        flow = (flow + flow_delta) if flow is not None else flow_delta

    return flow
```

---

## Key Hyperparameters

### Progressive GAN Training

**Resolution Schedule**:
- Base resolution: 4×4 or 8×8
- Doubling factor: 2× each stage
- Final resolution: 256×256, 512×512, or 1024×1024

**Transition Parameters**:
- Stabilization images: 600k-800k per stage
- Transition images (fade-in): 600k-800k
- Total images per stage: 1.2M-1.6M

**Batch Size**:
- 4×4: batch=512
- 16×16: batch=128
- 64×64: batch=32
- 256×256: batch=8
- 1024×1024: batch=4
(Adjust based on GPU memory)

### Curriculum Learning (Resolution)

**Schedule Design**:
- Start resolution: 1/4 to 1/8 of target (e.g., 64×64 for 512×512 target)
- Progression: Geometric (64→128→256→512) or linear (100→200→300→400→512)
- Epochs per stage: 10-20 epochs (adjust for convergence)

**Learning Rate**:
- Option 1: Constant LR throughout (simple)
- Option 2: Increase LR slightly at each new resolution (helps initial convergence)
- Option 3: Cosine annealing per stage (best generalization)

### Optical Flow Coarse-to-Fine

**PWC-Net Settings**:
- Pyramid levels: 6 (1/64 to 1/1 resolution)
- Warp method: Bilinear interpolation
- Cost volume search range: [-4, 4] pixels (at each level)

**RAFT Settings**:
- Feature resolution: 1/8 of input image
- Correlation radius: {1, 2, 4, 8} multi-scale
- GRU iterations: 12 (training), 24 (inference for best quality)
- Update step size: Learned (not fixed)

---

## Comparison: Coarse-to-Fine vs. Direct High-Resolution

| Aspect | Coarse-to-Fine | Direct High-Res |
|--------|----------------|-----------------|
| **Training Stability** | High (gradual complexity) | Medium (requires careful init) |
| **Convergence Speed** | Faster (fewer FLOPs early) | Slower (full resolution from start) |
| **Final Quality** | Good (curriculum effect) | Can be better (no resolution artifacts) |
| **Artifacts** | Resolution-dependent blobs (ProGAN) | Potential overfitting to high-freq noise |
| **Memory Usage** | Lower (early stages use less memory) | Higher (always full resolution) |
| **Implementation Complexity** | Higher (schedule management) | Lower (standard training loop) |

**When to Use Coarse-to-Fine**:
- Training GANs on high-resolution images (512×512+)
- Optical flow / dense prediction with large motions
- Limited compute budget (leverage low-res stages)
- When training stability is critical

**When to Use Direct**:
- Modern architectures with strong regularization (StyleGAN v2, transformers)
- Small images (<256×256) where resolution curriculum provides less benefit
- When resolution-dependent artifacts are unacceptable
- Sufficient compute available for full-resolution training

---

## Sources

**Primary Papers:**
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) - Karras et al., ICLR 2018 (accessed 2025-01-31)
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) - Karras et al., CVPR 2019 (StyleGAN v1)
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) - Karras et al., CVPR 2020 (StyleGAN v2, removing progressive training)
- [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) - Sun et al., CVPR 2018
- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) - Teed & Deng, ECCV 2020 (accessed 2025-01-31)

**Curriculum Learning:**
- [Progressive learning: A deep learning framework for continual learning](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301817) - Fayek et al., Neural Networks 2020
- [Learning Rate Curriculum](https://link.springer.com/article/10.1007/s11263-024-02186-5) - Croitoru et al., IJCV 2024

**Depth Estimation:**
- [Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669) - Chang & Chen, CVPR 2018

**Additional Resources:**
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) - Lin et al., CVPR 2017
- [ProGAN Implementation (GitHub)](https://github.com/tkarras/progressive_growing_of_gans) - NVIDIA Research
- [Deep learning modelling techniques: current progress, applications, advantages, and challenges](https://link.springer.com/article/10.1007/s10462-023-10466-8) - Ahmed et al., Artificial Intelligence Review 2023

---

*Coarse-to-fine represents a bridge between classical pyramid methods and modern deep learning, demonstrating that curriculum principles apply not just to data ordering but to architectural progression itself.*
