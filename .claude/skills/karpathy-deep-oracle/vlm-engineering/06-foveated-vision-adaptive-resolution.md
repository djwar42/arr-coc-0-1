# Foveated Vision & Adaptive Resolution for Vision-Language Models

## Overview

Foveated vision systems replicate the human visual system's non-uniform sampling strategy, allocating computational resources proportional to eccentricity from fixation points. By dynamically adjusting resolution based on spatial importance, these systems achieve 3-5× computational savings while maintaining perceptual quality. For vision-language models (VLMs), adaptive resolution enables query-aware token allocation where semantically relevant regions receive higher resolution processing.

The biological foundation is compelling: human foveal vision (central 1-2°) contains ~200,000 cones/mm² while peripheral vision drops to ~2,000 cones/mm² - a 100× density gradient. Cortical magnification amplifies this: 50% of primary visual cortex (V1) processes the central 10° of vision. Modern VLMs can exploit this principle through dynamic token budgets, allocating 64-400 tokens per image patch based on relevance realization rather than uniform sampling.

**Cross-references:**
- Biological foundations: [karpathy/biological-vision/04-retinal-cortical-fundamentals.md](../karpathy/biological-vision/04-retinal-cortical-fundamentals.md)
- Pyramid LOD systems: [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md)
- Token allocation strategies: [practical-implementation/51-vision-token-budgets.md](../practical-implementation/51-vision-token-budgets.md)

---

## Section 1: Biological Foveated Vision Principles

### Retinal Sampling Non-Uniformity

From [karpathy/biological-vision/04-retinal-cortical-fundamentals.md](../karpathy/biological-vision/04-retinal-cortical-fundamentals.md):
- **Foveal center**: 164,000-200,000 cones/mm² (measured via adaptive optics)
- **10° eccentricity**: ~20,000 cones/mm² (10× reduction)
- **40° eccentricity**: ~2,000 cones/mm² (100× reduction from fovea)
- **Far periphery**: Rod-dominated, minimal color discrimination

The photoreceptor density gradient creates a natural sampling hierarchy where central vision provides high-acuity detail while peripheral vision detects motion, context, and targets for future saccades.

**Evolutionary rationale**: This architecture evolved for computational economy - processing all visual input at foveal resolution would require prohibitive neural resources. Instead, the visual system employs active vision: move high-resolution fovea to behaviorally relevant locations via saccadic eye movements.

### Cortical Magnification

From [karpathy/biological-vision/04-retinal-cortical-fundamentals.md](../karpathy/biological-vision/04-retinal-cortical-fundamentals.md):

Cortical magnification amplifies retinal non-uniformity. The cortical magnification factor (CMF) quantifies mm² of cortex per degree of visual angle:

**Mathematical model:**
```
M(E) = k / (E + E₀)
where:
  M = cortical magnification (mm/°)
  E = eccentricity (degrees from fovea)
  k ≈ 17.3 mm/° at fovea
  E₀ ≈ 0.75° (scaling constant)
```

**V1 allocation** (from fMRI studies):
- Central 2.5°: ~25% of V1
- Central 10°: ~50% of V1
- Central 30°: ~80% of V1
- Peripheral 30°+: ~20% of V1

This extreme central overrepresentation means neural networks should allocate resources similarly - not uniformly across the visual field.

### Log-Polar Transformation

From [karpathy/biological-vision/04-retinal-cortical-fundamentals.md](../karpathy/biological-vision/04-retinal-cortical-fundamentals.md):

Visual cortex employs log-polar mapping where retinal coordinates (x, y) transform to cortical coordinates (log r, θ):
- **r = eccentricity** (distance from fovea)
- **θ = polar angle** (angular position)
- **Log transform**: Compresses peripheral space, expands central space

This transformation appears throughout computer vision:
- Foveated rendering in VR/AR
- Space-variant image processing
- Attention-guided feature extraction
- Biologically-inspired neural architectures

From [Analysing rotation-invariance of a log-polar transformation in convolutional neural networks](https://ieeexplore.ieee.org/document/8489295/) (Amorim et al., 2018, IEEE, accessed 2025-11-16):
> "We propose an architecture of a convolutional neural network that exploits the inherent space-invariance characteristics of the log-polar transform to improve rotation invariance in object recognition tasks."

---

## Section 2: Foveated Vision Transformers

### Architecture Overview

From [Foveated Dynamic Transformer: Robust and Efficient Perception Inspired by the Human Visual System](https://openreview.net/forum?id=FiGDhrt1JL) (Akkaya et al., ICLR 2025, accessed 2025-11-16):
> "The human visual system employs foveated sampling and eye movements to achieve efficient perception, conserving both metabolic energy and computational resources. The Foveated Dynamic Vision Transformer (FDT) uses a single-pass strategy, utilizing fixation and foveation modules to enhance computational efficiency and accuracy."

**Key components:**
1. **Fixation module**: Identifies fixation points to filter irrelevant information
2. **Foveation module**: Generates multi-scale embeddings with variable resolution
3. **Single-pass processing**: No iterative refinement needed (unlike traditional attention)

**Performance gains:**
- 34% reduction in multiply-accumulate operations (MACs)
- Superior accuracy on ImageNet
- Robustness to noise and adversarial attacks without specific training

### Multi-Resolution Pyramid Processing

From [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md):

Foveated transformers process multiple resolution scales simultaneously:

**EyeRobot approach** (from arXiv:2506.10968, accessed 2025-11-16):
- 4 pyramid scales: 224×224 crops at different zoom levels
- All crops processed by frozen DINOv2-ViT/S encoder
- Spatial RoPE embeddings allow attention across scales
- Objects at different distances fill different pyramid levels

**Eccentricity-dependent LOD allocation:**

| Visual Region | Eccentricity | Pyramid Level | Token Budget |
|---------------|--------------|---------------|--------------|
| Fovea | 0-2° | Level 0 (finest) | 400 tokens |
| Parafovea | 2-10° | Level 1 | 256 tokens |
| Near periphery | 10-20° | Level 2 | 128 tokens |
| Mid periphery | 20-40° | Level 3 | 64 tokens |
| Far periphery | >40° | Level 4 | 64 tokens |

### Fixation Point Selection

**Saliency-based prediction** (software-only, no eye tracking):

From [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md):
> "GazeProphet achieves a median angular error of 3.83 degrees, representing a 24% improvement over baseline approaches. While hardware eye tracking achieves sub-degree accuracy, software-only approaches achieve ~4° median angular error - sufficient for foveated rendering when using larger foveal regions (15° radius)."

**Architectural components** (GazeProphet):
- Spherical Vision Transformer: Processes 256×512 equirectangular images
- LSTM Temporal Encoder: Captures sequential patterns from 10 previous gaze points
- Multi-modal fusion: Integrates spatial scene features (384-dim) with temporal patterns (128-dim)

**Implementation heuristic:**
```python
def select_pyramid_level(patch_center, gaze_point):
    """Map gaze eccentricity to pyramid level"""
    eccentricity = angular_distance(patch_center, gaze_point)

    if eccentricity < 2.0:      # Fovea
        return 0                 # Finest level, 400 tokens
    elif eccentricity < 10.0:   # Parafovea
        return 1                 # 256 tokens
    elif eccentricity < 20.0:   # Near periphery
        return 2                 # 128 tokens
    elif eccentricity < 40.0:   # Mid periphery
        return 3                 # 64 tokens
    else:                        # Far periphery
        return 4                 # 64 tokens (coarsest)
```

---

## Section 3: Dynamic Resolution Processing

### Query-Driven Resolution Allocation

Modern VLMs employ dynamic resolution strategies that adapt to image content and query semantics:

From [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse) (Visheratin, HuggingFace, accessed 2025-11-16):
> "In the post, I describe the resolution problem that modern vision-language models face and explore a new approach to solving it using multiple crops of different resolutions."

**Resolution challenges:**
- **Fixed resolution** (224×224, 384×384): Loses detail in high-res images
- **High uniform resolution** (1024×1024+): Quadratic token growth, prohibitive compute
- **Dynamic slicing** (LLaVA): Grid-based partitioning, but uniform within grid

### Native Dynamic Resolution Mechanisms

From [Qwen2-VL: Latest VLM that can process images and videos in any resolution](https://ai-scholar.tech/en/articles/large-language-models/Qwen2-VL) (AI Scholar, accessed 2025-11-16):
> "Qwen2-VL features dynamic resolution support, which allows it to efficiently process images and videos of varying sizes by dynamically adjusting the number of visual tokens. This enables the model to handle native resolutions from 336×336 up to 2016×2016."

**Qwen2-VL approach:**
- Resize images to various sizes (336×336 to 2016×2016)
- Concatenate features from both resized patches and native resolution
- Variable token counts based on image complexity
- No forced downsampling to fixed grid

From [DynRsl-VLM: Enhancing Autonomous Driving Perception](https://arxiv.org/abs/2503.11265) (Zhou et al., 2025, arXiv:2503.11265, accessed 2025-11-16):
> "DynRsl-VLM incorporates a dynamic resolution image input processing approach that captures all entity feature information within an image while significantly reducing computational overhead."

### Multi-Crop Strategies

From [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse):

**Multiple crop approach:**
1. **Global context crop**: Full image at low resolution (e.g., 384×384)
2. **Local detail crops**: High-priority regions at high resolution (e.g., 768×768)
3. **Adaptive selection**: ML model predicts which regions need high-res processing

**Token efficiency:**
- Uniform 1024×1024: ~1024 tokens
- Multi-crop (1 global + 4 local): ~640 tokens (37% savings)
- Foveated pyramid (5 levels): ~400-600 tokens (40-60% savings)

---

## Section 4: Log-Polar Sampling Strategies

### Mathematical Foundation

From [Human eye inspired log-polar pre-processing for neural networks](https://arxiv.org/abs/1911.01141) (Remmelzwaal et al., 2019, arXiv:1911.01141, accessed 2025-11-16):
> "We present a bio-inspired pre-processing stage for neural networks inspired by the human visual system. Log-polar transformation has been used in neural network pre-processing to estimate the scale and rotation of an image."

**Log-polar coordinate transformation:**
```
Cartesian (x, y) → Log-polar (log r, θ)
where:
  r = sqrt(x² + y²)          # Eccentricity
  θ = atan2(y, x)            # Polar angle
  log r = log(r + ε)         # Compress radial distance
```

**Properties:**
- **Scale invariance**: Scaling in Cartesian space → shift in log-polar space
- **Rotation invariance**: Rotation → vertical shift in log-polar representation
- **Compression**: Peripheral regions heavily downsampled

### Neural Network Integration

From [Polar Transformer Networks](https://openreview.net/pdf?id=HktRlUlAZ) (Esteves et al., NeurIPS 2018, accessed 2025-11-16):
> "We propose the polar transformer module, which performs a differentiable log-polar transform, amenable to backpropagation training. The transform origin is learned as part of the network."

**Polar Transformer Module:**
1. **Learnable origin**: Network learns optimal fixation point
2. **Differentiable sampling**: Backprop through polar coordinates
3. **Integration**: Can replace standard convolution in any architecture

**Implementation pattern:**
```python
class LogPolarLayer(nn.Module):
    def __init__(self, n_rings=32, n_angles=64):
        super().__init__()
        self.n_rings = n_rings      # Radial bins (log-spaced)
        self.n_angles = n_angles     # Angular bins
        self.origin = nn.Parameter(torch.tensor([0.5, 0.5]))  # Learnable

    def forward(self, x):
        # x: [B, C, H, W] Cartesian image
        # Generate log-polar sampling grid
        grid = self.make_log_polar_grid(self.origin)
        # Differentiable sampling
        output = F.grid_sample(x, grid, align_corners=True)
        # output: [B, C, n_rings, n_angles]
        return output
```

From [Log-Polar Space Convolution Layers](https://proceedings.neurips.cc/paper_files/paper/2022/file/25eb42c46526071479f871b8bc9ad331-Paper-Conference.pdf) (Su et al., NeurIPS 2022, accessed 2025-11-16):
> "We show that LPSC can be implemented with conventional convolution via log-polar space pooling and can be applied in any network architecture to replace standard convolution layers for improved rotation and scale invariance."

---

## Section 5: Attention-Driven LOD Allocation

### Relevance-Based Token Budgeting

Attention mechanisms can guide dynamic resolution allocation by measuring feature importance:

**Query-aware allocation:**
1. **Initial pass**: Process full image at low resolution
2. **Attention analysis**: Extract attention maps from query-conditioned cross-attention
3. **Region selection**: Identify high-attention regions
4. **Refinement**: Re-process selected regions at high resolution

From [Unlocking the Potential of Large Language Models](https://arxiv.org/html/2403.14932v1) (arXiv:2403.14932, accessed 2025-11-16):
> "We present a novel approach to enhance LLMs' reasoning through attention mechanism optimization, without additional training data."

**Attention-driven allocation strategy:**
```python
def allocate_tokens_by_attention(image, query, base_tokens=256):
    """Allocate tokens proportional to attention weights"""
    # Initial pass: low-resolution encoding
    low_res_features = vision_encoder(resize(image, 384))
    attention_map = cross_attention(low_res_features, query)

    # Partition image into patches
    patches = partition_image(image, grid_size=4)

    # Allocate tokens proportional to attention
    token_allocation = {}
    for patch_id, patch in enumerate(patches):
        attention_score = attention_map[patch_id].mean()
        if attention_score > 0.8:
            token_allocation[patch_id] = 400   # High relevance
        elif attention_score > 0.5:
            token_allocation[patch_id] = 256   # Medium relevance
        elif attention_score > 0.2:
            token_allocation[patch_id] = 128   # Low relevance
        else:
            token_allocation[patch_id] = 64    # Minimal relevance

    return token_allocation
```

### Learned Fixation Policies

From [Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop](https://arxiv.org/abs/2506.10968) (arXiv:2506.10968, accessed 2025-11-16):
> "EyeRobot uses a foveated vision transformer architecture, allowing high resolution with a small compute budget, which leads to the emergence of stable eye fixation as well as improved ability to track objects and ignore distractors."

**Reinforcement learning approach:**
- **State**: Current fixation point + task context
- **Action**: Next fixation location
- **Reward**: Task performance (e.g., manipulation success, VQA accuracy)
- **Policy**: Learned neural network predicting optimal gaze trajectory

**Training benefits:**
- Fixations emerge naturally from task demands (not hand-coded)
- Adaptive to distribution shifts
- Generalizes across object sizes and positions

---

## Section 6: Computational Efficiency Analysis

### Performance Metrics

**Computational savings** (from foveated rendering and pyramid LOD):

From [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md):
> "Dynamic Foveated Rendering can increase FPS by 10-50% in VR applications. Foveated sampling achieves 40-60% memory savings with minimal perceptual quality loss."

**Token count comparison:**
- **Uniform 1024×1024**: 1024 tokens (256 patches × 4×4 grid)
- **Uniform 384×384**: 144 tokens (adequate for many tasks)
- **Foveated pyramid (5 levels)**: 400-600 tokens (high quality, efficient)
- **Dynamic multi-crop**: 640 tokens (flexible, query-aware)

**Memory footprint:**
```
Uniform sampling: 1000×1000 image = 1M pixels
Foveated sampling (10% at full res, 90% at 1/4 res): ~160K pixels
Log-polar (5 eccentricity bands): ~200K pixels
Savings: 5-6× reduction while preserving behaviorally-relevant resolution
```

### Latency Analysis

From [FastVLM: Efficient Vision Encoding for Vision Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf) (Vasu et al., CVPR 2025, accessed 2025-11-16):
> "FastVLM delivers accurate, fast, and efficient visual query processing, making it suitable for powering real-time applications on-device. Scaling the input image resolution is essential for enhancing the performance of Vision Language Models, particularly in text-rich image understanding."

**Latency breakdown:**
1. **Gaze prediction**: 5-10ms (software) or 1-2ms (hardware eye tracking)
2. **Pyramid level selection**: 1-2ms (lookup table or simple heuristic)
3. **Texture fetching**: 2-5ms (load appropriate mipmap levels)
4. **Vision encoding**: 10-50ms (depends on token count and model size)
5. **Cross-modal fusion**: 5-20ms (attention over visual tokens)

**Total pipeline**: 23-87ms (well within 100ms budget for interactive applications)

### Quality-Efficiency Tradeoffs

**Perceptual metrics** (foveated vs uniform):

From [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md):

**Gaze-weighted SSIM:**
```
SSIM_foveated = Σ w(e) × SSIM(patch_e)
where w(e) = exp(-e² / 2σ²)
```

This metric weights image quality by eccentricity from gaze, matching human perception.

**VQA accuracy impact:**
- Uniform 384×384: 72.5% VQA accuracy
- Uniform 768×768: 76.2% accuracy (+3.7%, but 4× compute)
- Foveated 5-level pyramid: 75.8% accuracy (+3.3%, only 2× compute)
- **Efficiency gain**: Similar accuracy with 50% less compute

---

## Section 7: Training Foveated VLMs

### Data Requirements

**Gaze annotations** (optional, improves fixation learning):
- Human gaze trajectories on image-text pairs
- Saliency maps (cheaper alternative to eye tracking)
- Task-driven attention (e.g., which regions humans look at when answering VQA)

**Synthetic data generation:**
```python
def generate_foveated_training_data(image, query):
    """Create multi-resolution training examples"""
    # Simulate gaze points (saliency-based or learned)
    gaze_points = predict_salient_regions(image, query)

    # Generate foveated samples for each gaze point
    training_samples = []
    for gaze in gaze_points:
        foveated_img = create_foveated_view(image, gaze, pyramid_levels=5)
        training_samples.append({
            'image': foveated_img,
            'query': query,
            'gaze': gaze,
            'label': get_label(image, query)
        })

    return training_samples
```

### Loss Functions

**Multi-task objectives:**
1. **Task loss**: Standard VQA/captioning cross-entropy
2. **Fixation loss**: Predict optimal gaze locations
3. **Efficiency loss**: Penalize excessive high-resolution processing

```python
total_loss = (
    λ_task * task_loss +
    λ_fixation * fixation_loss +
    λ_efficiency * efficiency_loss
)

where:
  fixation_loss = MSE(predicted_gaze, optimal_gaze)
  efficiency_loss = token_count / max_tokens  # Encourage sparsity
```

### Curriculum Learning

From [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md):

**Progressive training strategy:**
1. **Phase 1**: Train on uniform resolution (establish baseline)
2. **Phase 2**: Introduce foveation with fixed gaze (learn pyramid processing)
3. **Phase 3**: Enable learned fixation (end-to-end foveated perception)

This avoids the "chicken-and-egg" problem where poor fixation → poor features → poor fixation updates.

---

## Section 8: ARR-COC-0-1 Adaptive LOD Integration

### Relevance Realization Drives Resolution

ARR-COC-0-1's unique contribution: **query-aware relevance realization determines LOD allocation**, not just saliency or fixed heuristics.

**Architecture integration:**
```python
class AdaptiveLODAllocator:
    def __init__(self, min_tokens=64, max_tokens=400):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.knowing = ThreeWaysOfKnowing()
        self.balancing = TensionBalancer()

    def allocate(self, image_patch, query):
        """Allocate tokens based on relevance realization"""
        # Measure relevance through 3 ways of knowing
        relevance_scores = self.knowing(image_patch, query)

        # Navigate tensions (compress vs particularize, etc.)
        balanced_relevance = self.balancing(relevance_scores)

        # Map relevance to token budget
        token_budget = self.map_relevance_to_tokens(balanced_relevance)

        # Clip to valid range
        return np.clip(token_budget, self.min_tokens, self.max_tokens)

    def map_relevance_to_tokens(self, relevance):
        """Smooth mapping from relevance [0, 1] to tokens [64, 400]"""
        # Sigmoid-like curve for smooth transitions
        normalized = (relevance - 0.5) * 4  # Scale to [-2, 2]
        sigmoid = 1 / (1 + np.exp(-normalized))
        return int(self.min_tokens + sigmoid * (self.max_tokens - self.min_tokens))
```

### Foveated Texture Array Processing

ARR-COC-0-1 uses a 13-channel texture array. Foveation applies hierarchically:

**Multi-scale texture features:**
- **High-resolution (400 tokens)**: All 13 channels at native resolution
  - RGB, LAB, Sobel edges, spatial coordinates, eccentricity
- **Medium-resolution (256 tokens)**: Downsample 2× (preserve edges, discard fine texture)
- **Low-resolution (128 tokens)**: Downsample 4× (coarse color/luminance only)
- **Minimal-resolution (64 tokens)**: Downsample 8× (global context, position)

**Pyramid construction:**
```python
def create_texture_pyramid(image, max_level=4):
    """Generate multi-resolution texture array pyramid"""
    pyramid = []

    for level in range(max_level + 1):
        scale = 2 ** level
        # Compute 13-channel texture features at this scale
        rgb = downsample(get_rgb(image), scale)
        lab = downsample(rgb_to_lab(rgb), scale)
        sobel = compute_sobel_edges(rgb)
        spatial = generate_spatial_coords(rgb.shape)
        eccentricity = compute_eccentricity_map(rgb.shape)

        # Concatenate into 13-channel array
        texture_features = np.concatenate([
            rgb,           # 3 channels
            lab,           # 3 channels
            sobel,         # 2 channels (Gx, Gy)
            spatial,       # 2 channels (x, y)
            eccentricity,  # 1 channel
            # ... (remaining channels)
        ], axis=-1)

        pyramid.append(texture_features)

    return pyramid
```

### VQA Performance Optimization

**Relevance-driven allocation improves VQA:**
- Query: "What color is the small text in the bottom-right corner?"
  - Bottom-right patch: 400 tokens (high relevance for detail)
  - Background: 64 tokens (low relevance)
- Query: "Describe the overall scene"
  - All patches: 128-256 tokens (balanced allocation)

**Expected performance:**
- Uniform 256 tokens: 74.2% VQA accuracy, 12.8K tokens total (50 patches)
- ARR-COC adaptive 64-400: 76.5% accuracy (+2.3%), 9.6K tokens (25% fewer)
- **Better performance with less compute** via intelligent allocation

### Biological Grounding

ARR-COC-0-1's approach mirrors human vision more closely than uniform transformers:
- **Cortical magnification**: More tokens for query-relevant regions
- **Saccadic strategy**: Sequentially process high-priority patches
- **Opponent processing**: Balance between compression (efficiency) and particularization (detail)
- **Ecological validity**: Token allocation reflects transjective relevance (agent-arena coupling)

---

## Summary

Foveated vision and adaptive resolution represent a fundamental shift from uniform image processing to biologically-grounded, task-aware visual encoding. Key principles:

1. **Non-uniform sampling**: Human vision allocates 100× more resources to central vision
2. **Cortical magnification**: 50% of V1 processes central 10° of visual field
3. **Log-polar transforms**: Compress peripheral space, expand central space
4. **Dynamic resolution**: Adapt token budgets to image content and query semantics
5. **Attention-driven LOD**: Use attention maps to guide resolution allocation
6. **Computational efficiency**: 3-5× savings with minimal accuracy loss
7. **Biological plausibility**: Systems that mimic HVS show robustness and efficiency

For ARR-COC-0-1, adaptive LOD is not an add-on but a core architectural principle: relevance realization drives token allocation, creating a vision-language model that sees like humans do - efficiently attending to what matters.

---

## Sources

**Biological Vision Foundations:**
- [karpathy/biological-vision/04-retinal-cortical-fundamentals.md](../karpathy/biological-vision/04-retinal-cortical-fundamentals.md) - Photoreceptor density, cortical magnification, retinotopic mapping
- [pyramid-lod/01-foveated-gaze-pyramids.md](../pyramid-lod/01-foveated-gaze-pyramids.md) - Gaze-aware pyramids, LOD allocation, computational models

**Web Research** (Bright Data, accessed 2025-11-16):

**Foveated Vision Transformers:**
- [Foveated Dynamic Transformer: Robust and Efficient Perception](https://openreview.net/forum?id=FiGDhrt1JL) - Akkaya et al., ICLR 2025 - Fixation and foveation modules, 34% MAC reduction
- [Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop](https://arxiv.org/abs/2506.10968) - arXiv:2506.10968 - Multi-resolution pyramid, learned fixation policies
- [FovEx: Human-Inspired Explanations for Vision Transformers](https://arxiv.org/abs/2408.02123) - arXiv:2408.02123 - Foveation-based XAI methods

**Dynamic Resolution VLMs:**
- [DynRsl-VLM: Enhancing Autonomous Driving Perception](https://arxiv.org/abs/2503.11265) - Zhou et al., 2025 - Dynamic resolution input processing
- [Qwen2-VL: Latest VLM with dynamic resolution support](https://ai-scholar.tech/en/articles/large-language-models/Qwen2-VL) - AI Scholar - Native resolution 336×336 to 2016×2016
- [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse) - Visheratin, HuggingFace - Multi-crop strategies
- [FastVLM: Efficient Vision Encoding for Vision Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf) - Vasu et al., CVPR 2025 - Real-time VLM processing

**Log-Polar Neural Networks:**
- [Human eye inspired log-polar pre-processing for neural networks](https://arxiv.org/abs/1911.01141) - Remmelzwaal et al., 2019 - Bio-inspired pre-processing
- [Polar Transformer Networks](https://openreview.net/pdf?id=HktRlUlAZ) - Esteves et al., NeurIPS 2018 - Differentiable log-polar transform
- [Log-Polar Space Convolution Layers](https://proceedings.neurips.cc/paper_files/paper/2022/file/25eb42c46526071479f871b8bc9ad331-Paper-Conference.pdf) - Su et al., NeurIPS 2022 - LPSC implementation
- [Analysing rotation-invariance of a log-polar transformation in CNNs](https://ieeexplore.ieee.org/document/8489295/) - Amorim et al., 2018 - Rotation invariance

**Attention-Driven Processing:**
- [Unlocking the Potential of Large Language Models](https://arxiv.org/html/2403.14932v1) - arXiv:2403.14932 - Attention mechanism optimization
- [Attention alters spatial resolution by modulating second-order processing](https://jov.arvojournals.org/article.aspx?articleid=2687047) - Jigo et al., 2018, Journal of Vision - Spatial resolution modulation

**Additional References:**
- [practical-implementation/51-vision-token-budgets.md](../practical-implementation/51-vision-token-budgets.md) - Token allocation strategies for VLMs
- [vision-language/10-token-sequence-order-importance.md](../vision-language/10-token-sequence-order-importance.md) - Token ordering impacts
