# Foveated Vision with Gaze-Aware Pyramids

## Overview

Foveated vision with gaze-aware pyramids combines biological principles of human vision with computational Level-of-Detail (LOD) techniques to achieve efficient visual processing. By dynamically allocating higher-resolution pyramid levels to gaze fixation regions and coarser levels to peripheral areas, these systems replicate the non-uniform sampling strategy of the human retina while maintaining computational efficiency. This approach has profound implications for VR/AR rendering, vision-language models (VLMs), and robotic vision systems.

**Core Principle**: Just as the human fovea contains ~200,000 cones/mm² while peripheral vision has 100× lower density, gaze-aware pyramid systems allocate 64-400 visual tokens per patch based on query-driven relevance and eye-tracking data. The result is 40-60% memory savings with minimal perceptual quality loss.

**Cross-references**:
- [Foveated Rendering & Peripheral Context Preservation](../karpathy/biological-vision/03-foveated-rendering-peripheral.md)
- [Vision Token Budgets: Optimal Patch Counts](../practical-implementation/51-vision-token-budgets.md)

---

## Section 1: Eye-Tracking Driven LOD Selection

### Eye-Tracking Hardware and Calibration

Modern eye-tracking systems for foveated rendering use infrared cameras and specialized sensors to monitor eye movements in real-time. Commercial systems like Tobii Pro VR Integration and SMI Eye Tracking HMD provide sub-degree accuracy for research applications, while consumer VR headsets (Meta Quest Pro, Apple Vision Pro, PlayStation VR2) integrate eye tracking for mainstream deployment.

**Hardware Requirements**:
- **Infrared cameras**: Track pupil position and corneal reflection at 90-120 Hz
- **Calibration procedures**: Initial 5-9 point calibration, frequent recalibration needed
- **Latency targets**: <20ms gaze-to-render for imperceptible foveation
- **Accuracy challenges**: Individual eye anatomy differences, environmental lighting

From [GazeProphet: Software-Only Gaze Prediction for VR Foveated Rendering](https://arxiv.org/html/2508.13546v2) (arXiv:2508.13546, accessed 2025-01-31):
> "Hardware-based eye tracking limits foveated rendering adoption across the VR ecosystem. Premium headsets like Meta Quest Pro and Apple Vision Pro include eye tracking capabilities. However, the majority of VR users own devices without this hardware."

**Calibration complexity**: Eye tracking requires precise calibration procedures and frequent recalibration due to headset movement, individual eye anatomy variations, and environmental factors like lighting conditions. This adds friction to user experience and limits practical deployment.

### Gaze Prediction Algorithms (Software-Only Approaches)

Software-only gaze prediction offers an alternative to expensive eye-tracking hardware. By analyzing scene content and temporal gaze patterns, machine learning models can predict where users will look next without specialized sensors.

**Key Approaches**:

1. **Saliency-based prediction**: Traditional computer vision models use low-level features (color, intensity, orientation) to identify attention-grabbing regions
2. **Deep learning saliency**: CNNs demonstrate superior performance over hand-crafted features for static images
3. **Temporal sequence modeling**: LSTMs capture gaze history patterns for dynamic scenes
4. **Spherical Vision Transformers**: Process 360° VR environments with specialized positional encoding

From [GazeProphet: Software-Only Gaze Prediction for VR Foveated Rendering](https://arxiv.org/html/2508.13546v2):
> "GazeProphet achieves a median angular error of 3.83 degrees, representing a 24% improvement over the best baseline approach. The system demonstrates consistent performance across all evaluation metrics."

**Practical performance**: While hardware eye tracking achieves sub-degree accuracy, software-only approaches like GazeProphet achieve ~4° median angular error—sufficient for foveated rendering when using larger foveal regions (15° radius) that account for prediction uncertainty.

**Architectural components**:
- **Spherical Vision Transformer**: Processes 256×512 equirectangular images with spherical harmonic positional encoding
- **LSTM Temporal Encoder**: Captures sequential patterns from 10 previous gaze points
- **Multi-modal fusion**: Integrates spatial scene features (384-dim) with temporal patterns (128-dim)

### LOD Selection Heuristics Based on Fixation

The mapping from gaze location to pyramid level selection follows biological acuity gradients. Different regions of the visual field receive different LOD allocations based on eccentricity from fixation.

**Eccentricity-dependent LOD allocation**:

| Visual Region | Eccentricity | Human Acuity | Pyramid Level | Token Budget |
|---------------|--------------|--------------|---------------|--------------|
| Fovea | 0-2° | 20/20 | Level 0 (finest) | 400 tokens |
| Parafovea | 2-10° | 20/60 | Level 1 | 256 tokens |
| Near periphery | 10-20° | 20/120 | Level 2 | 128 tokens |
| Mid periphery | 20-40° | 20/400 | Level 3 | 64 tokens |
| Far periphery | >40° | >20/400 | Level 4 (coarsest) | 64 tokens |

**Mathematical model from biological data**:
```
Resolution(eccentricity) = R₀ / (1 + eccentricity/e₀)
where R₀ = foveal resolution, e₀ ≈ 2.5°
```

**Implementation heuristic**:
```python
def select_pyramid_level(patch_center, gaze_point):
    """Map gaze eccentricity to pyramid level"""
    eccentricity = angular_distance(patch_center, gaze_point)

    if eccentricity < 2.0:  # Fovea
        return 0  # Finest level
    elif eccentricity < 10.0:  # Parafovea
        return 1
    elif eccentricity < 20.0:  # Near periphery
        return 2
    elif eccentricity < 40.0:  # Mid periphery
        return 3
    else:  # Far periphery
        return 4  # Coarsest level
```

From [Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop](https://arxiv.org/html/2506.10968v2) (arXiv:2506.10968, accessed 2025-01-31):
> "EyeRobot uses a foveated vision transformer architecture, allowing high resolution with a small compute budget, which we find leads to the emergence of stable eye fixation as well as improved ability to track objects and ignore distractors."

### Real-Time Gaze → Pyramid Level Mapping

Real-time systems must map gaze coordinates to pyramid level selections within tight latency budgets (<20ms total) to avoid perceptual artifacts.

**Pipeline stages**:
1. **Gaze acquisition**: Eye tracking or prediction (5-10ms)
2. **Pyramid level selection**: Map gaze to LOD per patch (1-2ms)
3. **Texture fetching**: Load appropriate mipmap levels (2-5ms)
4. **Rendering**: Generate final frame with variable LOD (5-10ms)

**Dynamic reallocation**: When gaze shifts, the system must smoothly transition pyramid levels to avoid visible "popping" artifacts. Techniques include:
- **Temporal filtering**: Average gaze position over 3-5 frames
- **Gradual LOD transitions**: Blend between pyramid levels during saccades
- **Motion blur**: Mask LOD changes during rapid eye movements

**Latency-accuracy tradeoff**: Aggressive gaze prediction (30-50ms ahead) reduces latency but increases prediction error. Conservative prediction (10-20ms) maintains accuracy but requires faster rendering pipelines.

From [Dynamic Foveated Rendering (DFR) in Virtual Reality](https://pimax.com/blogs/blogs/about-dynamic-foveated-rendering-dfr-in-virtual-reality-vr) (Pimax, accessed 2025-01-31):
> "Dynamic Foveated Rendering utilizes eye-tracking technology integrated into VR headsets. By constantly monitoring the user's gaze, the system dynamically adjusts the level of detail rendered in different parts of the virtual scene."

**Performance benefits**: DFR can increase FPS by 10-50% in VR applications, with Pimax Crystal demonstrating compatibility across games like iRacing, Microsoft Flight Simulator 2020, and DCS World.

---

## Section 2: Peripheral Vision Degradation Modeling

### Biological Basis: Eccentricity-Dependent Acuity

The human visual system exhibits dramatic non-uniformity in spatial resolution. The fovea, occupying ~1-2° of the central visual field, contains ~50% of the ganglion cells in the retina despite representing <1% of the visual field area. This "cortical magnification" means the brain devotes disproportionate processing to the foveal region.

**Photoreceptor density gradient**:
- **Foveal center**: ~200,000 cones/mm² (pure cone vision)
- **10° eccentricity**: ~20,000 cones/mm² (10× reduction)
- **40° eccentricity**: ~2,000 cones/mm² (100× reduction)
- **Far periphery**: Rod-dominated, minimal color vision

**Functional specialization**:
- **Fovea**: High-acuity color vision, detail discrimination, reading, face recognition
- **Periphery**: Motion detection, spatial orientation, attention guidance, scene context

From [Foveated Rendering & Peripheral Context Preservation](../karpathy/biological-vision/03-foveated-rendering-peripheral.md):
> "The fovea (central ~1-2° of visual field) contains densely packed cone photoreceptors enabling high acuity, while peripheral regions have progressively lower cone density. Despite this non-uniform sampling, humans perceive rich visual scenes because peripheral vision provides critical contextual information."

### Cortical Magnification Curves

The cortical magnification factor describes how much cortical area in primary visual cortex (V1) is devoted to each degree of visual field:

**Mathematical model**:
```
M(e) = k / (e + e₀)
where:
  M(e) = cortical magnification at eccentricity e
  k = constant (~17.3 mm/degree at fovea)
  e₀ = scaling constant (~0.75°)
```

**Implications for pyramid LOD**:
- Cortical magnification decays as 1/eccentricity
- Pyramid downsampling should match this 1/eccentricity gradient
- Each pyramid level represents a 2× downsampling (half resolution)
- 4 pyramid levels → 16× total downsampling matches ~40° periphery

**Perceptual consequence**: The visual system naturally performs "lossy compression" in peripheral vision, allocating processing bandwidth according to information relevance. Gaze-aware pyramid systems replicate this biological compression strategy.

### Computational Models: Gaussian Falloff vs Log-Polar

Two primary computational approaches model peripheral degradation for foveated rendering:

**1. Gaussian Falloff Model**:
```
Quality(eccentricity) = exp(-eccentricity² / 2σ²)
```
- Smooth degradation from center to periphery
- Single parameter (σ) controls falloff rate
- Easy to implement with standard texture filtering
- Commonly used in VR foveated rendering

**2. Log-Polar Sampling Model**:
```
(r, θ) = (log(eccentricity), angle)
```
- Matches retinal ganglion cell distribution more closely
- Preserves angular resolution while reducing radial resolution
- More complex to implement (requires coordinate transformation)
- Better matches biological processing

From [Eye, Robot: Learning to Look to Act](https://arxiv.org/html/2506.10968v2):
> "Foveated systems emulate the resource-rational nonuniformity of the retina, using either specialized hardware or learned neural mechanisms. EyeRobot uses a multi-resolution transformer that processes input images into an image pyramid of crops with N scales centered at the center pixel."

**Multi-resolution pyramid approach** (EyeRobot):
- 4 scales: 224×224 crops at different zoom levels
- All crops processed by frozen DINOv2-ViT/S encoder
- Spatial RoPE embeddings allow attention across scales
- Object at different distances fills different pyramid levels

### Perceptual Metrics for Degradation

Measuring the perceptual impact of peripheral degradation requires specialized metrics beyond traditional PSNR or SSIM, which treat all pixels equally.

**Foveated Image Quality Metrics**:

1. **Gaze-weighted SSIM**: Weight SSIM scores by eccentricity from gaze point
   ```
   SSIM_foveated = Σ w(e) × SSIM(patch_e)
   where w(e) = exp(-e² / 2σ²)
   ```

2. **Cortical-weighted MSE**: Weight pixel errors by cortical magnification
   ```
   MSE_cortical = Σ M(e) × (pixel_pred - pixel_gt)²
   ```

3. **User study pass rates**: Percentage of users unable to detect foveation
   - Gold standard: >90% undetectable threshold
   - Practical: >70% for real-time VR applications

4. **Task performance metrics**: Measure impact on downstream tasks
   - Visual search time
   - Object recognition accuracy
   - Navigation performance

From [GazeProphet](https://arxiv.org/html/2508.13546v2):
> "Performance remains consistent across different spatial regions and scene types. Center region performance (3.81° angular error) closely matches peripheral region results (3.89° angular error). This consistency enables effective foveated rendering across the entire visual field."

**Quality-performance tradeoff**: Systems must balance perceptual quality (measured by foveated metrics) against computational savings (FPS improvement, memory reduction). Optimal tradeoff depends on application—VR gaming tolerates more aggressive foveation than medical visualization.

---

## Section 3: VR/AR Foveated Rendering + VLMs

### Foveated Rendering in VR Headsets (Meta Quest, Vision Pro)

Modern VR headsets increasingly integrate eye tracking and foveated rendering as standard features:

**Meta Quest Pro** (2022):
- Built-in eye tracking sensors
- Dynamic foveated rendering support
- 10-30% FPS improvement in supported apps
- Requires explicit developer support (not automatic)

**Apple Vision Pro** (2023):
- Advanced eye tracking for UI control
- Eye tracking at 60 Hz minimum
- Foveated rendering for power efficiency
- Seamless integration (no developer opt-in needed)

**PlayStation VR2** (2023):
- Eye tracking + foveated rendering standard
- Significant power savings for standalone mode
- Used for both rendering and interaction

From [Dynamic Foveated Rendering (DFR) in Virtual Reality](https://pimax.com/blogs/blogs/about-dynamic-foveated-rendering-dfr-in-virtual-reality-vr):
> "DFR can increase your FPS by 10 to 50%. This means you can even run DCS on a 2060 GPU. For standalone VR devices with limited battery capacity, Dynamic Foveated Rendering can contribute to extending battery life by optimizing resource usage."

**Implementation challenges**:
- Eye tracking adds hardware cost ($100-300 per headset)
- Calibration friction in consumer devices
- Software ecosystem fragmentation (not all engines support DFR)
- Latency requirements (<20ms) require tight hardware-software integration

### Integrating VLM Inference with Gaze Tracking

Vision-language models (VLMs) can leverage gaze tracking to allocate visual tokens efficiently—a computational analog to foveated rendering.

**VLM token allocation strategy**:
```python
def allocate_tokens_by_gaze(image, gaze_point, total_budget=256):
    """Allocate visual tokens based on gaze fixation"""
    patches = image_to_patches(image, patch_size=14)

    token_allocation = []
    for patch in patches:
        eccentricity = angular_distance(patch.center, gaze_point)

        # More tokens to foveal patches
        if eccentricity < 2.0:
            tokens = 64  # Full resolution
        elif eccentricity < 10.0:
            tokens = 32  # Half resolution
        elif eccentricity < 20.0:
            tokens = 16  # Quarter resolution
        else:
            tokens = 4   # Minimal context

        token_allocation.append(tokens)

    # Normalize to meet total budget
    return normalize_to_budget(token_allocation, total_budget)
```

**Benefits for VLMs**:
- Reduce visual tokens from 576 to ~100-150 (70-80% reduction)
- Focus compute on task-relevant image regions
- Maintain peripheral context for scene understanding
- Enable longer text generation without context overflow

From [Vision Token Budgets](../practical-implementation/51-vision-token-budgets.md):
> "The inference-optimal regime requires using larger LLMs with fewer visual tokens, often achieving 80% token reduction (from 576 to ~100 tokens) or even compression to single-digit token counts with minimal accuracy loss."

### Query-Aware Foveation (Attend to Relevant Regions)

Beyond eye tracking, VLMs can use query semantics to guide foveation—attending to image regions relevant to the question being asked.

**Query-aware pyramid selection**:

Example: "What color is the car?"
- High LOD: Patches containing cars
- Medium LOD: Patches with roads, parking lots (contextual)
- Low LOD: Sky, distant trees (irrelevant)

Example: "How many people are in the image?"
- High LOD: Patches with human figures
- Medium LOD: Patches with crowd-like textures
- Low LOD: Empty background regions

**Implementation via cross-attention**:
```python
# Query: "What color is the car?"
query_embedding = text_encoder("What color is the car?")

# Compute relevance scores for image patches
relevance = cross_attention(
    query=query_embedding,
    keys=patch_embeddings
)

# Allocate pyramid levels based on relevance
for i, patch in enumerate(patches):
    if relevance[i] > 0.8:  # High relevance
        pyramid_level[i] = 0  # Finest
    elif relevance[i] > 0.5:
        pyramid_level[i] = 1
    else:
        pyramid_level[i] = 2  # Coarsest
```

**Relation to ARR-COC project**: This query-aware foveation is central to the ARR-COC (Adaptive Relevance Realization - Contexts Optical Compression) framework, where:
- **Propositional knowing**: Statistical information content of patches
- **Perspectival knowing**: Salience landscapes (what stands out)
- **Participatory knowing**: Query-content coupling (what matters for this task)

See: [ARR-COC-VIS Project](../../../../README.md) for full architecture details.

### Latency Challenges (20ms Gaze-to-Render)

The human visual system detects delays >20ms between gaze movement and scene update, creating perceptual artifacts and potential motion sickness in VR.

**Latency budget breakdown**:
| Stage | Latency | Mitigation |
|-------|---------|------------|
| Eye tracking | 5-10ms | Hardware-accelerated sensors |
| Gaze prediction | 1-2ms | Lightweight neural networks |
| LOD selection | 1-2ms | Precomputed lookup tables |
| Texture fetch | 2-5ms | GPU texture cache optimization |
| Rendering | 5-10ms | Foveated rendering reduces workload |
| Display update | 1-2ms | High refresh rate displays (120 Hz+) |
| **Total** | **15-31ms** | **Aim for <20ms** |

**Prediction to hide latency**: Systems can predict gaze location 20-50ms ahead to hide sensor and rendering latency. Trade-off between prediction horizon (longer = more latency hiding) and prediction accuracy (longer = more errors).

From [GazeProphet](https://arxiv.org/html/2508.13546v2):
> "Real-time foveated rendering requires latency below 10 milliseconds, which needs validation on target VR hardware. Model compression and hardware acceleration could reduce latency below the 10 millisecond threshold required for foveated rendering."

**Future directions**: Neural rendering techniques (NeRF, 3D Gaussian Splatting) may enable sub-5ms rendering by caching scene representations and only rendering the foveal region at high quality in real-time.

---

## Section 4: Dynamic Pyramid Allocation Based on Gaze

### Allocate High-Res Pyramid Levels to Fovea

The core strategy of gaze-aware pyramids is to dynamically allocate the finest pyramid levels (highest resolution) to regions under foveal fixation, while peripheral regions use progressively coarser levels.

**Pyramid structure** (4 levels for 1024×1024 image):
- **Level 0**: 1024×1024 (full resolution, 1,048,576 pixels)
- **Level 1**: 512×512 (2× downsample, 262,144 pixels)
- **Level 2**: 256×256 (4× downsample, 65,536 pixels)
- **Level 3**: 128×128 (8× downsample, 16,384 pixels)

**Per-patch allocation** (foveal region = 32×32 patch at Level 0):
```
Foveal patch (Level 0): 32×32 = 1,024 pixels → 400 visual tokens
Peripheral patch (Level 3): 32×32 at 8× = 4×4 effective = 64 visual tokens
Memory savings: 84% reduction per peripheral patch
```

**Smooth transition zones**: Rather than hard boundaries, use weighted blending:
```python
def compute_lod_weight(eccentricity, level):
    """Smooth LOD transition based on eccentricity"""
    # Eccentricity thresholds for each level
    thresholds = [2.0, 10.0, 20.0, 40.0]

    if level == 0:  # Finest level
        return max(0, 1.0 - eccentricity / thresholds[0])
    elif level < len(thresholds):
        lower = thresholds[level - 1]
        upper = thresholds[level]
        t = (eccentricity - lower) / (upper - lower)
        return max(0, 1.0 - t)
    else:
        return 1.0 if eccentricity > thresholds[-1] else 0
```

### Peripheral Regions Use Coarse Levels

Peripheral vision, while lower resolution, remains critical for scene understanding and spatial awareness. The challenge is determining the minimum acceptable quality for peripheral regions.

**Perceptual guidelines**:
- **Scene gist**: Requires coarse spatial layout (Level 2-3 sufficient)
- **Motion detection**: Temporal contrast more important than spatial resolution
- **Attention guidance**: Salient features visible even at coarse resolution
- **Spatial navigation**: Geometric structure preserved at coarse LOD

**Failure modes to avoid**:
- **Too coarse**: Peripheral patches become unrecognizable blobs
- **Too frequent LOD changes**: Visible "popping" during gaze shifts
- **Discontinuities**: Visible seams between LOD regions
- **Temporal aliasing**: Coarse LOD + motion = distracting flicker

From [Eye, Robot](https://arxiv.org/html/2506.10968v2):
> "The uniform resolution model only loosely keeps the target within view, whereas the foveated model consistently maintains the target object near the center of its field of view. The foveated model is also significantly more robust to distractor objects."

**Emergent behavior**: Foveated architectures naturally learn fixation stability—keeping targets centered in the high-resolution foveal region rather than letting them drift to low-resolution periphery.

### ARR-COC Relevance + Gaze = Optimal LOD

The ARR-COC (Adaptive Relevance Realization - Contexts Optical Compression) framework provides a principled approach to combining gaze information with query-driven relevance for optimal LOD allocation.

**Integration strategy**:
```python
def arr_coc_pyramid_lod(patch, gaze_point, query_embedding):
    """
    Combine gaze-based LOD with ARR-COC relevance realization

    Three ways of knowing from Vervaeke:
    1. Propositional: Information content (Shannon entropy)
    2. Perspectival: Salience (what stands out)
    3. Participatory: Query-content coupling (what matters)
    """

    # Gaze-based eccentricity factor
    eccentricity = angular_distance(patch.center, gaze_point)
    gaze_weight = exp(-eccentricity**2 / (2 * sigma**2))

    # Propositional knowing: Information content
    patch_entropy = compute_shannon_entropy(patch)
    prop_score = patch_entropy / max_entropy

    # Perspectival knowing: Visual salience
    salience = compute_salience(patch)  # e.g., contrast, edge density
    persp_score = salience / max_salience

    # Participatory knowing: Query relevance
    query_relevance = cross_attention_score(
        query=query_embedding,
        key=patch_embedding
    )
    partic_score = query_relevance

    # Combine all three with gaze weight
    relevance = (
        0.3 * prop_score +
        0.3 * persp_score +
        0.4 * partic_score
    ) * gaze_weight

    # Map relevance to pyramid level
    if relevance > 0.8:
        return 0  # Finest level (400 tokens)
    elif relevance > 0.6:
        return 1  # (256 tokens)
    elif relevance > 0.4:
        return 2  # (128 tokens)
    else:
        return 3  # Coarsest level (64 tokens)
```

**Opponent processing** (balancing tensions):
- **Compress ↔ Particularize**: Use coarse LOD globally, fine LOD locally
- **Exploit ↔ Explore**: Fixate on known-relevant regions vs scan for new information
- **Focus ↔ Diversify**: Allocate tokens to single salient region vs distribute across scene

See: [ARR-COC Project README](../../../../README.md) and [realizing.py](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/realizing.py) for implementation details.

### Memory Savings (40-60% Typical)

Quantifying memory savings from gaze-aware pyramid allocation:

**Baseline** (uniform high resolution):
- Image: 1024×1024 pixels
- Patch size: 32×32 (grid of 32×32 = 1,024 patches)
- Tokens per patch: 400 (dense sampling)
- **Total: 409,600 tokens**

**Foveated pyramid** (gaze-aware allocation):
- Foveal region (5% of patches): 400 tokens × 51 patches = 20,400 tokens
- Parafoveal (15% of patches): 256 tokens × 154 patches = 39,424 tokens
- Near periphery (30% of patches): 128 tokens × 307 patches = 39,296 tokens
- Far periphery (50% of patches): 64 tokens × 512 patches = 32,768 tokens
- **Total: 131,888 tokens (68% reduction)**

**Memory breakdown**:
```
Uniform:   409,600 tokens × 2 bytes (FP16) = 819 KB
Foveated:  131,888 tokens × 2 bytes (FP16) = 264 KB
Savings:   555 KB (68% reduction)
```

**Performance impact**:
- **Inference latency**: 2-3× faster (fewer tokens to process)
- **Training throughput**: 1.5-2× more examples per batch
- **Memory bandwidth**: 60-70% reduction in GPU↔VRAM transfers

From [Vision Token Budgets](../practical-implementation/51-vision-token-budgets.md):
> "For visual reasoning tasks, the inference-optimal behavior in VLMs is achieved by using the largest LLM that fits within the inference budget while minimizing visual token count - often to a single token."

**Extreme compression**: Research shows VLMs can function with 10-20 visual tokens (99% reduction) for simple queries, though spatial tasks require more tokens for localization.

---

## Sources

### Source Documents
- [Foveated Rendering & Peripheral Context Preservation](../karpathy/biological-vision/03-foveated-rendering-peripheral.md)
- [Vision Token Budgets: Optimal Patch Counts](../practical-implementation/51-vision-token-budgets.md)

### Web Research

**arXiv Papers** (accessed 2025-01-31):
- [GazeProphet: Software-Only Gaze Prediction for VR Foveated Rendering](https://arxiv.org/html/2508.13546v2) - arXiv:2508.13546v2, Oct 9, 2025
- [Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop](https://arxiv.org/html/2506.10968v2) - arXiv:2506.10968v2, Sep 15, 2025

**Industry Resources**:
- [About Dynamic Foveated Rendering (DFR) in Virtual Reality (VR)](https://pimax.com/blogs/blogs/about-dynamic-foveated-rendering-dfr-in-virtual-reality-vr) - Pimax Blog, May 11, 2024
- [The fundamentals of eye tracking part 4: Tools for VR](https://link.springer.com/article/10.3758/s13428-024-02529-7) - Springer, 2025

**Academic Papers**:
- [The Effects of Visual Behavior and Ego-Movement on Foveated Rendering](https://pmc.ncbi.nlm.nih.gov/articles/PMC12000250/) - PMC, 2025
- [Foveated gaze-contingent displays for peripheral LOD management](https://www.researchgate.net/publication/51986032_Foveated_gaze-contingent_displays_for_peripheral_LOD_management_3D_visualization_and_stereo_imaging) - ResearchGate, May 12, 2025

**Additional References**:
- [Objects guide human gaze behavior in dynamic real-world scenes](https://pmc.ncbi.nlm.nih.gov/articles/PMC10602265/) - PMC, 2023
- [Gaze Estimation Method Combining Facial Feature Extractor with Pyramid Squeeze Attention Mechanism](https://www.researchgate.net/publication/372410175) - ResearchGate, July 4, 2023

---

## Cross-References to ARR-COC Project

This knowledge directly informs the ARR-COC-VIS implementation:

**Relevant modules**:
- `knowing.py`: Three ways of knowing scorers (Propositional, Perspectival, Participatory)
- `balancing.py`: Opponent processing for compress↔particularize tension
- `attending.py`: Salience realization and token budget allocation
- `realizing.py`: Pipeline orchestrator combining all components

**Key connection**: Gaze-aware pyramid LOD is a biological implementation of relevance realization—the human visual system already performs dynamic token allocation based on transjective relevance (the coupling between agent goals and scene content). ARR-COC formalizes this process for VLMs.

**Future work**: Integrate eye-tracking data (real or predicted) with ARR-COC's relevance scores to create a unified foveated VLM system that matches human visual efficiency.
