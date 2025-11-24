# Perceptual Masking for LOD Optimization

## Overview

Perceptual masking exploits limitations of human vision to reduce rendering quality without visible degradation. Contrast masking and spatial masking allow aggressive LOD reduction in specific image regions while maintaining perceived quality.

## Primary Sources

From [01-Interactive Perceptual Rendering Pipeline](../source-documents/01-An%20Interactive%20Perceptual%20Rendering%20Pipeline%20using%20Contrast%20and%20Spatial%20Masking%20-%20Inria.md):
- Contrast sensitivity function (CSF)
- Spatial frequency masking
- Perceptual rendering pipeline
- Quality metrics (VDP, HDR-VDP)

## Key Concepts

### Contrast Sensitivity Function (CSF)

**What it is**: Human ability to detect contrast varies with spatial frequency and luminance.

**Key properties**:
- **Peak sensitivity**: 3-5 cycles per degree (cpd)
- **Low frequency rolloff**: Reduced sensitivity to large, uniform gradients
- **High frequency rolloff**: Reduced sensitivity to fine texture detail
- **Luminance dependence**: Better contrast detection at higher luminance

**LOD implications**:
- Reduce high-frequency detail (> 20 cpd) first
- Preserve mid-frequency structure (3-5 cpd)
- Large uniform regions tolerate aggressive simplification

### Contrast Masking

**Definition**: High contrast features reduce visibility of nearby lower contrast features.

**Mechanism**:
- Strong edge "masks" nearby subtle edges
- Texture detail invisible near high-contrast boundaries
- Allows LOD reduction near salient features

**Example**:
- Detailed brick wall next to plain concrete → reduce concrete detail
- High-contrast character in scene → reduce background LOD
- Sharp shadow edges → reduce texture detail in shadow

**Masking strength**:
```
Visibility_threshold(target) = base_threshold × (1 + k × mask_contrast^α)
```
- Stronger mask → higher threshold → less visible target
- Enables aggressive LOD near high-contrast regions

### Spatial Masking

**Definition**: Visual sensitivity varies with spatial location and content complexity.

**Masking effects**:

1. **Texture masking**
   - Fine texture hides LOD artifacts
   - Coarse texture more revealing
   - Highly textured regions tolerate compression

2. **Edge masking**
   - Nearby strong edges reduce sensitivity
   - Smooth gradients require careful LOD
   - Edge-based LOD allocation

3. **Pattern masking**
   - Regular patterns hide distortions
   - Irregular patterns reveal artifacts
   - Exploit repetitive structure

**LOD strategy**:
- Aggressive reduction in highly textured areas
- Preserve quality in smooth gradients
- Maintain strong edges, simplify interiors

## Perceptual Rendering Pipeline

### Stage 1: Image Analysis

**Input**: Scene to be rendered

**Analysis steps**:
1. **Luminance map**: Compute pixel luminance values
2. **Spatial frequency analysis**: Decompose into frequency bands
3. **Contrast map**: Compute local contrast at multiple scales
4. **Masking map**: Identify masking regions (high contrast, texture)

**Output**: Per-region masking strength

### Stage 2: LOD Allocation

**Masking-guided LOD**:
```
LOD_budget(region) = base_budget × masking_factor(region)
```

**Masking factors**:
- High texture region: 0.3-0.5 (aggressive LOD)
- Near strong edges: 0.4-0.6 (moderate LOD)
- Smooth gradients: 0.8-1.0 (preserve quality)
- Focal regions: 1.0 (full quality)

**Frequency-specific LOD**:
- Reduce high frequencies first (> 20 cpd)
- Preserve mid frequencies (3-5 cpd)
- Smooth low frequencies acceptable

### Stage 3: Perceptual Quality Assessment

**Metrics**:

1. **Visual Difference Predictor (VDP)**
   - Models contrast sensitivity
   - Spatial frequency decomposition
   - Probability of detection map

2. **HDR-VDP** (High Dynamic Range)
   - Handles HDR images
   - Luminance adaptation
   - Advanced CSF modeling

3. **JND (Just Noticeable Difference)**
   - Threshold for visible difference
   - Guide LOD selection
   - User-calibrated thresholds

**Quality validation**:
- Render with allocated LOD
- Compare to reference (VDP)
- Adjust LOD if differences exceed threshold
- Iterate until perceptually equivalent

## Practical Applications

### Texture LOD with Masking

**Standard texture LOD**: Distance-based mipmapping

**Perceptual texture LOD**:
- Analyze texture complexity
- High-frequency textures → aggressive mipmap bias
- Low-frequency textures → conservative bias
- Contrast masking near edges

**Example**:
- Rough stone wall: Use mip level +2 (more blur)
- Smooth metal surface: Use mip level +0 (preserve)
- Near high-contrast edge: Use mip level +1

### Geometry LOD with Masking

**Standard geometry LOD**: Distance-based polygon reduction

**Perceptual geometry LOD**:
- Textured meshes tolerate higher reduction
- Smooth meshes require more polygons
- Near high-contrast features → reduce polygons
- In focal region → preserve detail

**Silhouette preservation**:
- Maintain silhouette edges (high visibility)
- Simplify interior polygons (masked by texture/lighting)
- Contrast-adaptive tessellation

### Lighting LOD with Masking

**Shadow detail**:
- High-frequency shadows masked by nearby contrast
- Soft shadows tolerate aggressive filtering
- Sharp shadow edges require precision

**Indirect lighting**:
- Diffuse inter-reflections: Low frequency, aggressive LOD
- Glossy reflections: Mid-high frequency, preserve quality
- Caustics: High frequency, but often masked by texture

**Ambient occlusion**:
- Fine-scale AO masked by texture
- Large-scale AO perceptually important
- LOD: Reduce sample count in masked regions

## Integration with Gaze-Contingent LOD

**Combined approach**:
```
Final_LOD(region) = gaze_LOD(eccentricity) × masking_LOD(contrast, texture)
```

**Synergy**:
- Peripheral regions: Low sensitivity (gaze) + masking → very aggressive LOD
- Foveal textured regions: High sensitivity (gaze) + masking → moderate LOD
- Foveal smooth regions: High sensitivity (gaze) + no masking → full quality

**Example allocation**:
- Peripheral + textured: LOD level 6 (very coarse)
- Peripheral + smooth: LOD level 4 (moderate)
- Foveal + textured: LOD level 2 (detailed)
- Foveal + smooth: LOD level 0 (full quality)

## Temporal Masking

**Saccadic masking**:
- Vision suppressed during saccades (200-300ms)
- Tolerate dramatic LOD changes
- Pre-saccade: Reduce quality
- Post-saccade: Restore quality

**Motion blur masking**:
- Fast-moving objects: High-frequency detail invisible
- Camera motion: Peripheral detail masked
- Predictive: Reduce LOD before motion, restore after

**Flicker fusion**:
- Temporal alternation < 50 Hz invisible
- Alternate LOD levels between frames
- Time-multiplexed detail rendering

## Quality Metrics and Validation

### Objective Metrics

**VDP-based**:
- Probability of detection (PoD) maps
- Target: PoD < 75% for invisible differences
- Per-region quality assessment

**SSIM (Structural Similarity)**:
- Luminance, contrast, structure comparison
- Target: SSIM > 0.95 for high quality
- Faster than VDP, less accurate

### Subjective Validation

**User studies**:
- A/B testing: Reference vs perceptual LOD
- Detection threshold experiments
- Task performance measures

**Calibration**:
- Individual CSF differences
- Display calibration (luminance, gamma)
- Viewing distance and resolution

## Cross-References

- [02-visual-perception.md](02-visual-perception.md) - Perceptual foundations
- [03-transjective-relevance.md](03-transjective-relevance.md) - Gaze-content coupling
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md) - Gaze-contingent techniques
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - LOD calculation

## Key Takeaways

1. **Contrast sensitivity varies**: Peak at 3-5 cpd, rolloff at low and high frequencies
2. **Masking enables LOD**: High contrast and texture hide LOD artifacts
3. **Perceptual metrics guide LOD**: VDP, HDR-VDP predict visibility
4. **Combine with gaze**: Multiplicative benefit from gaze + masking LOD
5. **Temporal masking**: Saccades and motion enable aggressive transitions

**Design principle**: Analyze scene for masking opportunities, allocate LOD based on perceptual thresholds, validate with VDP-based metrics, combine with gaze-contingent strategies for optimal efficiency.

---

**See also**: [references/00-glossary.md](../references/00-glossary.md) for definitions of CSF, contrast masking, spatial masking, and VDP.
