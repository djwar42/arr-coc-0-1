# Foveated Rendering

## Overview

Foveated rendering exploits the human visual system's non-uniform spatial resolution to allocate rendering resources proportionally to visual acuity. High detail at gaze center (fovea), reduced detail in periphery.

## Primary Sources

From [10-Gaze-aware Displays](../source-documents/10-Gaze-aware%20Displays%20and%20Interaction%20-%20SURREAL%20TEAM.md):
- Foveated rendering techniques for VR
- Eye tracking integration
- Gaze prediction and latency compensation
- Quality metrics and user studies

From [17-Virtual Reality Annual International Symposium](../source-documents/17-Virtual%20Reality%20Annual%20International%20Symposium%2C%20March%201-5%2C%201997%2C%20Albuquerque%2C%20New%20Mexico.%20-%20DTIC.md):
- Early foveated rendering systems
- Hardware implementations
- Performance analysis

## Biological Foundation

**Human visual acuity**:
- Fovea (0-2°): ~150,000 cones/mm², highest acuity
- Parafovea (2-10°): Moderate acuity, 10-20% of foveal
- Periphery (>10°): ~5,000 cones/mm², motion detection only

**Visual field**:
- Foveal vision: ~2° (thumbnail at arm's length)
- Detailed vision: ~10°
- Total visual field: ~200° horizontal, ~130° vertical

**Implications**: 99% of visual field has dramatically reduced acuity → enormous LOD opportunity.

## Core Technique

### Eccentricity-Based LOD Allocation

**Acuity function**:
```
Acuity(e) = 1 / (1 + α × e)
```
Where e = eccentricity (degrees from gaze), α = falloff rate (typ. 0.3-0.5)

**LOD allocation**:
```
LOD_level(pixel) = max_LOD × (1 - Acuity(eccentricity(pixel)))
```

**Example** (4 LOD levels):
- 0-2°: LOD 0 (full quality)
- 2-5°: LOD 1 (75% quality)
- 5-10°: LOD 2 (50% quality)
- 10-20°: LOD 3 (25% quality)
- >20°: LOD 4 (minimal, 10% quality)

### Three-Region Model

**Region 1: Foveal (0-2°)**
- Full resolution rendering
- All visual features preserved
- No compression

**Region 2: Parafoveal (2-10°)**
- Moderate reduction (50-75% quality)
- Preserve salient features
- Smooth transitions

**Region 3: Peripheral (>10°)**
- Aggressive reduction (10-25% quality)
- Motion and coarse structure only
- Maximum LOD benefit

## Implementation Approaches

### Fixed Foveated Rendering (FFR)

**No eye tracking required**:
- Assume gaze at screen center
- Fixed eccentricity map
- Works for static viewing

**Limitations**:
- Breaks when gaze moves off-center
- Less effective than eye-tracked
- Best for constrained viewing (HMDs with fixed gaze)

**Use cases**:
- VR applications (gaze typically centered)
- Cinematic content
- Predictable viewing patterns

### Gaze-Contingent Foveated Rendering (GCFR)

**Requires eye tracking**:
- Real-time gaze position (90-120 Hz)
- Dynamic eccentricity map
- Follows user attention

**Pipeline**:
```
1. Capture gaze position (eye tracker)
2. Predict gaze at render time (compensate latency)
3. Generate eccentricity map
4. Allocate LOD per pixel/tile
5. Render with variable quality
6. Display result
```

**Critical: Latency**
- Eye movement to photon: <20ms target
- Eye tracker: 8-16ms
- Rendering: <11ms (90 fps)
- Total budget: ≤20ms for imperceptible

### Foveated Ray Tracing

**Ray count allocation**:
```
Rays_per_pixel(e) = max_rays × Acuity(e)
```

**Example**:
- Fovea: 1024 samples per pixel (path tracing quality)
- Parafovea: 256 samples per pixel
- Periphery: 16-64 samples per pixel

**Benefits**:
- Natural integration with ray tracing
- Per-pixel control
- Smooth quality gradient

**From source 10** (Andrew Polychronakis thesis):
- Ray-tracing foveated pipeline
- Reduced ray generation in periphery
- Path tracing acceleration

### Foveated Rasterization

**Geometry LOD**:
- Fovea: High-poly models
- Periphery: Reduced tessellation, simplified meshes

**Texture LOD**:
- Fovea: High-res textures (mip level 0)
- Periphery: Low-res textures (mip level 3-4)

**Shading LOD**:
- Fovea: Full PBR shading
- Periphery: Simplified shading (flat, ambient only)

**Tile-based approach**:
- Divide screen into tiles (e.g., 64×64 pixels)
- Assign LOD per tile based on average eccentricity
- Render each tile with allocated quality

## Gaze Prediction

**Challenge**: Rendering latency causes gaze-LOD mismatch

**Solution**: Predict gaze position at display time

**Methods**:

1. **Velocity-based**:
   ```
   gaze_predicted = gaze_current + velocity × latency
   ```
   Simple, works for smooth pursuit

2. **Saccade detection**:
   - Detect saccade onset (velocity threshold)
   - Predict destination (saccade typically 3-15°)
   - Preload high LOD at destination

3. **Task-driven**:
   - Learn typical gaze patterns for task
   - Predict based on task context
   - Proactive LOD loading

4. **Neural networks**:
   - Train on user gaze history
   - Predict next fixation target
   - Personalized, accurate

## Performance Benefits

**Rendering cost reduction**:
- Foveal region: 1-5% of screen area
- Quality maintained: Only in foveal region
- Savings: 60-90% rendering cost

**Example** (1920×1080 display, 2° fovea):
- Foveal pixels: ~67×67 = 4,489 pixels (0.2%)
- Full quality: 4,489 pixels
- Reduced quality: 2,073,600 - 4,489 = 2,069,111 pixels (99.8%)
- Typical savings: 70-80% GPU time

**Concrete numbers**:
- Non-foveated: 11ms per frame (90 fps)
- Foveated: 3-4ms per frame (can reach 240+ fps)
- Enables higher quality foveal rendering

## Quality Considerations

### Transition Artifacts

**Popping**: Abrupt LOD transitions visible as gaze moves

**Solutions**:
- Smooth LOD gradients
- Hysteresis in LOD selection
- Alpha blending between LOD levels
- Transition during saccades (exploit saccadic masking)

### Aliasing

**Problem**: Undersampling in periphery causes temporal aliasing during eye movement

**Solutions**:
- Temporal anti-aliasing (TAA)
- Motion blur in periphery
- Spatial anti-aliasing at LOD boundaries

### Calibration

**Individual differences**:
- Foveal extent varies (1-3°)
- Acuity falloff rate varies
- Perceptual thresholds vary

**Calibration procedure**:
1. Measure user acuity at eccentricities
2. Determine JND thresholds
3. Fit acuity function parameters
4. Generate personalized LOD map

## Application Domains

### Virtual Reality (VR)

**High benefits**:
- High resolution displays (4K+ per eye)
- Constrained viewing (HMD)
- Eye tracking increasingly common (HTC Vive Pro Eye, Meta Quest Pro)

**Challenges**:
- Low latency critical (<20ms)
- Stereo rendering (2× cost)
- Barrel distortion complicates eccentricity map

### Augmented Reality (AR)

**Benefits**:
- Transparent displays (lower fill rate)
- Battery savings crucial

**Challenges**:
- Outdoor viewing (bright, variable lighting)
- Eye tracking harder in AR
- Registration errors

### Desktop/Mobile

**Less effective**:
- Gaze not constrained
- Viewing distance varies
- Fixed foveated rendering only (center assumption often wrong)

**Potential**:
- Low-power devices (smartphones, tablets)
- Cloud gaming (reduce bandwidth)

## Implementation Best Practices

### LOD Transition Design

**Smooth gradients**:
- 3-5 LOD levels
- Wide transition zones (5-10° visual angle)
- Gaussian or sigmoid falloff

**Update strategy**:
- Update eccentricity map at eye tracking rate (90-120 Hz)
- Render at display rate (90-120 Hz)
- Synchronize updates with saccades when possible

### Multi-Resolution Rendering

**Render targets**:
- Foveal layer: Full resolution
- Parafoveal layer: 50% resolution
- Peripheral layer: 25% resolution
- Composite final image

**Memory savings**:
- Foveal: 1.0 relative memory
- Parafoveal: 0.25 relative
- Peripheral: 0.0625 relative
- Total: ~1.3 (vs 1.0 non-foveated per layer, but covers full screen)

### Integration with Other Techniques

**Foveated + ASW (Async Spacewarp)**:
- Reduce foveal rendering load
- Maintain 90 fps with spacewarp
- Smooth experience

**Foveated + Dynamic Resolution**:
- Scale overall resolution based on load
- Maintain foveal quality
- Reduce peripheral further

## Cross-References

- [concepts/02-visual-perception.md](../concepts/02-visual-perception.md) - Biological foundation
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Gaze-content coupling
- [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Eye tracking integration
- [applications/01-vr-ar.md](../applications/01-vr-ar.md) - VR/AR applications
- [integration/07-metadata-texture-arrays-2025-01-30.md](../integration/07-metadata-texture-arrays-2025-01-30.md) - GPU texture array metadata storage (40-channel architecture, 33× speedup)
- [optimization/01-spatial-locality-cache-2025-01-30.md](../optimization/01-spatial-locality-cache-2025-01-30.md) - Spatial locality and cache optimization (5× fewer cache misses)

## Hardware Acceleration for Foveated VLMs

### GPU Texture Primitives

**Cross-Reference**: See [techniques/07-gpu-texture-primitives-vlm-2025-01-30.md](07-gpu-texture-primitives-vlm-2025-01-30.md) for comprehensive hardware acceleration techniques.

**Key Innovation**: Map foveated rendering to GPU texture hardware
- Hardware mipmap generation: glGenerateMipmap() 50× faster than PyTorch (0.1ms vs 5ms)
- Texture sampling with cortical magnification: M(e) = M₀/(e+e₀) determines mip level
- Compute shaders for parallel foveated sampling: tex2DLod() samples appropriate pyramid level
- Result: 6.7× overall vision encoding speedup (67ms → 10ms)

### Production Implementation

**Meta Quest 3**: Foveated rendering in production VR (2023)
- Uses cortical magnification formula in hardware
- Eye tracking integrated with rendering pipeline
- Proves commercial viability of foveated approaches

**VLM Applications**: See [applications/02-real-time-video-vlms-2025-01-30.md](../applications/02-real-time-video-vlms-2025-01-30.md)
- Real-time video understanding at 60+ FPS
- AR/VR spatial assistants with gaze-aware processing
- Hardware acceleration enables entirely new application categories

## Recent VLM Applications (2024-2025)

### Overview: Foveation Wave in Vision-Language Models

**Discovery**: Independent research groups converged on foveated approaches for VLM efficiency in 2024-2025.

**Cross-Reference**: See [research-landscape/00-vlm-token-allocation-2024-2025.md](../research-landscape/00-vlm-token-allocation-2024-2025.md) for comprehensive competitive analysis.

### PyramidDrop (ICLR 2025, 90 citations)

**Paper**: "Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models"

**Approach**: Pyramid-based token pruning with bottom-up saliency
```python
# Conceptual: PyramidDrop uses visual saliency, not query-awareness
pyramid = build_gaussian_pyramid(image, levels=4)
for level in pyramid:
    saliency_scores = compute_saliency(level)  # Bottom-up
    selected_tokens = prune_by_saliency(level, saliency_scores)
```

**Results**:
- 65-75% token reduction
- <3% accuracy drop on VQA benchmarks
- 2-3× speedup, training-free

**Limitation**: Saliency-driven (perspectival knowing only), not query-aware

### FastVLM (Apple, July 2025)

**Paper**: "FastVLM: Efficient Vision-Language Models via Difficulty-Aware Pyramid Sampling"

**Approach**: Difficulty-aware pyramid sampling
```python
# FastVLM: difficulty-based token budget
difficulty = classify_difficulty(image, query)  # Statistics-based
if difficulty == "easy":
    tokens = 150  # 45% reduction
elif difficulty == "medium":
    tokens = 273  # Baseline
else:  # hard
    tokens = 450  # More tokens for complex images
```

**Results**:
- **Production deployment** in iOS/macOS
- 2.5× average speedup
- <1% accuracy drop in practice

**Limitation**: Difficulty estimation, not fixation-based foveation

### Foveated Retinotopy (October 2025)

**Paper**: "Foveated Retinotopy Improves Classification in CNNs" (arXiv 2402.15480)

**Approach**: Biological cortical magnification M(e) = M₀/(e+e₀)
```python
def cortical_magnification(eccentricity, M0=1.0, e0=0.5):
    """Daniel & Whitteridge 1961 primate data"""
    return M0 / (eccentricity + e0)

# Sample with M(e) weighting
for token in tokens:
    eccentricity = distance(token_pos, fixation)
    weight = cortical_magnification(eccentricity)
    # Token density proportional to M(e)
```

**Results**:
- **+3-5% accuracy on ImageNet** (not just efficiency!)
- Biology improves performance via spatial regularization
- Validates foveated approaches for deep learning

**Limitation**: CNNs only, not VLMs; center fixation only (not query-driven)

### ARR-COC-VIS Integration

**Our approach combines all three**:
1. **Pyramids** (PyramidDrop) → multi-scale processing
2. **Difficulty-awareness** (FastVLM) → adaptive budgets
3. **Biological M(e)** (Foveated Retinotopy) → cortical magnification

**Plus unique contributions**:
- Query-driven fixation (participatory knowing)
- Vervaeke's relevance realization (four ways of knowing)
- VLM-specific implementation (extends Foveated Retinotopy beyond CNNs)

### Research Wave Phenomenon (2024-2025)

**Timeline of convergence**:
- **January 2025**: PyramidDrop (ICLR 2025) - pyramid pruning
- **March 2025**: DPN-LLaVA - dynamic pyramid depth
- **July 2025**: FastVLM (Apple) - difficulty-aware pyramids
- **October 2025**: Foveated Retinotopy - cortical magnification

**Common themes**:
- Multi-scale is universal (all use pyramids)
- Query-awareness emerging (FastVLM, DPN-LLaVA)
- Training-free methods work (PyramidDrop validates)
- Biological grounding is rare (only Foveated Retinotopy + us)

**Our positioning**: First biologically-grounded VLM token allocation with query-driven fixation.

### Comparison Table: VLM Foveation Approaches

| Method | Pyramids | Biology | Query-Aware | VLMs | Production |
|--------|----------|---------|-------------|------|------------|
| PyramidDrop | ✅ | ❌ | ❌ | ✅ | Research |
| FastVLM | ✅ | ❌ | ✅ (difficulty) | ✅ | **Apple iOS** |
| Foveated Retinotopy | ❌ | ✅ (M(e)) | ❌ | ❌ | Research |
| ARR-COC-VIS | ✅ | ✅ (M(e)) | ✅ (fixation) | ✅ | Research |

**Unique corner**: We're the only approach combining ALL FOUR (pyramids + biology + query + VLMs).

## Key Takeaways

1. **Biological grounding**: Fovea = 2°, acuity drops exponentially with eccentricity
2. **Massive savings**: 60-90% rendering cost reduction
3. **Eye tracking essential**: GCFR far superior to FFR
4. **Latency critical**: <20ms gaze-to-photon for imperceptibility
5. **Smooth transitions**: Prevent popping, maintain perceptual quality
6. **VR is killer app**: Constrained viewing, high resolution, eye tracking available
7. **Hardware acceleration**: GPU texture primitives provide 50× speedup for VLM foveation
8. **2024-2025 VLM wave**: Multiple groups converged on foveated/pyramid approaches
9. **Biological grounding improves accuracy**: Foveated Retinotopy showed +3-5% (not just efficiency)
10. **Query-awareness is differentiator**: Our unique contribution in VLM foveation landscape

**Design principle**: Track gaze, predict position at render time, allocate quality proportional to visual acuity, smooth transitions, measure and minimize latency, leverage GPU texture hardware for production performance, integrate biological cortical magnification with query-driven fixation for VLMs.
