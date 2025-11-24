# Focus Maps and Partial Schematization

## Overview

Focus maps create visual hierarchy through selective detail and schematization, accentuating regions of interest while maintaining spatial context. This technique guides attention and enables efficient LOD allocation.

## Primary Sources

From [00-Accentuating focus maps via partial schematization](../source-documents/00-Accentuating%20focus%20maps%20via%20partial%20schematization%20-%20Pure.md):
- Focus map generation techniques
- Schematization strategies
- User studies on effectiveness
- Design guidelines

From [09-Focus Guided Light Field Saliency](../source-documents/09-Focus%20Guided%20Light%20Field%20Saliency%20Estimation.md):
- Focus cues for saliency
- Light field rendering
- Depth-of-field simulation

## Key Concepts

### Focus Maps

**Definition**: Spatial distribution indicating importance/attention for each region.

**Properties**:
- **Values**: 0.0 (context) to 1.0 (full focus)
- **Spatial**: 2D map over image/scene
- **Smooth**: Gradual transitions between focus and context
- **Dynamic**: Can update over time with changing attention

**Generation methods**:
1. **Gaze-based**: Eye tracking → fixation positions → focus map
2. **Task-based**: Task-relevant regions → high focus values
3. **Salience-based**: Visual salience → focus map
4. **User-specified**: Manual painting of focus regions
5. **Automatic**: Machine learning from task/context

### Schematization Spectrum

**Levels of abstraction**:
```
Photorealistic (focus) ← → Symbolic (context)
```

**Intermediate levels**:
1. **Photorealistic**: Full detail, realistic materials, lighting
2. **Simplified**: Reduced polygons, simplified shading
3. **Cel-shaded**: Cartoon-like, flat colors, outline
4. **Line art**: Black lines on white, minimal shading
5. **Iconic**: Simplified symbols, essential features only
6. **Symbolic**: Abstract symbols or text labels

**Selection**: Based on focus map value
```
schematization_level = function(focus_value)
```

## Focus Map Generation

### Gaze-Based Focus Maps

**From eye tracking**:
```
focus_map(x,y) = Σ gaussian(x,y, fixation_i)
```
Sum of Gaussians centered at fixation points

**Parameters**:
- **σ (spread)**: Width of focus region (typ. 2-5° visual angle)
- **Decay**: Temporal decay of past fixations
- **Weight**: Recent fixations weighted higher

**Example**:
```python
def generate_gaze_focus_map(fixations, image_size, sigma=50):
    focus_map = zeros(image_size)
    for fix in fixations:
        weight = exp(-fix.age / decay_time)
        gaussian_blob = gaussian_2d(fix.position, sigma)
        focus_map += weight * gaussian_blob
    focus_map = normalize(focus_map, 0, 1)
    return focus_map
```

### Salience-Based Focus Maps

**Bottom-up salience**:
```
salience = combine(color_contrast, intensity_contrast,
                  orientation_contrast, motion)
```

**Itti-Koch model**:
- Feature maps (color, intensity, orientation)
- Center-surround differences
- Normalization and combination
- Winner-take-all for fixation prediction

**Top-down task salience**:
- Task-relevant features enhanced
- Learned from task history
- Combined with bottom-up

**Final focus map**:
```
focus = α × bottom_up + β × top_down
```

### Task-Based Focus Maps

**Known task requirements**:
- Reading: Text regions → high focus
- Navigation: Path ahead + landmarks → high focus
- Search: Target features → high focus

**Example (reading)**:
```
focus_map(x,y) = {
  1.0 if (x,y) in current_line
  0.7 if (x,y) in adjacent_lines
  0.3 if (x,y) in same_paragraph
  0.1 elsewhere
}
```

### Hybrid Approaches

**Combine multiple sources**:
```
focus_final = w1×gaze + w2×salience + w3×task
```

**Adaptive weighting**:
- Eye tracking available: w1=0.6, w2=0.2, w3=0.2
- No eye tracking: w1=0.0, w2=0.5, w3=0.5

## Schematization Techniques

### Geometry Schematization

**Photorealistic → Simplified**:
- Progressive mesh collapse
- Quadric error minimization
- LOD reduction

**Simplified → Line art**:
- Extract silhouette edges
- Remove internal edges
- Render as black lines

**Line art → Iconic**:
- Replace with billboard
- Show representative icon
- Minimal geometry

### Material/Shading Schematization

**Photorealistic**:
- PBR materials (metallic, roughness, etc.)
- Complex lighting (GI, reflections, shadows)
- High-frequency detail

**Simplified**:
- Phong/Blinn shading
- Direct lighting only
- Baked ambient occlusion

**Cel-shaded**:
- Quantized lighting (2-3 levels)
- Strong outline
- Flat colors

**Flat**:
- Single color per object
- No shading variation
- Ambient lighting only

### Texture Schematization

**Levels**:
1. **Full texture**: High-res (2048×2048+)
2. **Reduced**: Medium-res (512×512)
3. **Simplified**: Low-res or solid color (64×64 or flat)
4. **Removed**: Pure geometry color

**Adaptive**: Focus regions maintain full texture, context reduces progressively

## Rendering Pipeline

### Multi-Pass Focus-Based Rendering

**Pass 1: Focus region (photorealistic)**
```
for object in focus_region:
  render_full_quality(object)
```

**Pass 2: Context region (schematized)**
```
for object in context_region:
  schematization = get_schematization_level(focus_map, object)
  render_at_level(object, schematization)
```

**Pass 3: Composition**
```
final = blend(focus_pass, context_pass, focus_map)
```

### Single-Pass Shader-Based

**Vertex shader**:
```glsl
attribute vec3 position;
attribute vec3 normal;
uniform mat4 mvp;
varying float focus_value;

void main() {
  gl_Position = mvp * vec4(position, 1.0);

  // Sample focus map at vertex position
  focus_value = texture(focus_map, screen_position).r;
}
```

**Fragment shader**:
```glsl
varying float focus_value;
uniform sampler2D texture;

void main() {
  vec4 photo = photorealistic_shading();
  vec4 schem = schematized_shading();

  // Blend based on focus value
  gl_FragColor = mix(schem, photo, focus_value);
}
```

## Transition Design

### Smooth Focus Gradients

**Gaussian blur focus map**:
```
focus_smooth = gaussian_blur(focus_raw, kernel_size)
```

**Prevents harsh boundaries**:
- Smooth transitions imperceptible
- Gradual quality change
- Reduced artifacts

### Temporal Coherence

**Filter focus map over time**:
```
focus_current = α × focus_new + (1-α) × focus_previous
```
Where α = 0.1-0.3 for smooth temporal changes

**Hysteresis**:
- Threshold for entering focus > threshold for leaving
- Prevents flickering
- Stable focus regions

### Schematization Interpolation

**Blend between levels**:
```
render = mix(level_n, level_n+1, fractional_level)
```

**Smooth LOD transitions**:
- No popping between schematization levels
- Continuous quality spectrum

## Perceptual Considerations

### Attention Guidance

**High contrast draws attention**:
- Focus region: Detailed, colorful
- Context region: Simplified, desaturated
- Natural attention capture

**Maintaining context**:
- Context not invisible, just simplified
- Spatial relationships preserved
- Scene understanding maintained

### Cognitive Load Reduction

**Simplified context**:
- Less competing detail
- Easier to focus on task
- Reduced visual clutter

**Studies show** (from source 00):
- Faster task completion with focus maps
- Improved search performance
- Higher subjective satisfaction

## Advanced Techniques

### Depth-of-Field Integration

**Simulate camera focus** (from source 09):
```
blur_amount = function(depth_difference_from_focus)
focus_map *= (1 - blur_amount)
```

**Combined cues**:
- Depth-of-field + schematization
- Reinforces focus region
- Natural and intuitive

### Light Field Saliency

**From source 09**:
- Analyze light field structure
- Extract focus cues from 4D light field
- Enhanced saliency estimation
- Applications to VR/AR

### Animated Focus

**Guided attention**:
- Animate focus map to guide user
- Tutorials, storytelling
- Smooth transitions (2-3 seconds)

**Implementation**:
```python
def animate_focus(start_pos, end_pos, duration):
    for t in range(0, duration, frame_time):
        alpha = t / duration
        current_pos = lerp(start_pos, end_pos, alpha)
        focus_map = generate_focus_at(current_pos)
        render_with_focus(focus_map)
```

## Application Domains

### Information Visualization

**Large datasets**:
- Focus: Selected data points (detailed)
- Context: Overview (simplified)
- Interactive exploration

**Network graphs**:
- Focus: Selected nodes and edges (detailed)
- Context: Distant nodes (iconic or removed)

### Medical Imaging

**Focus on pathology**:
- Tumor/lesion: Full detail
- Surrounding anatomy: Context
- Efficient review

### Geographic Visualization

**Maps**:
- Focus: Current location (street-level detail)
- Context: Schematic roads and regions
- Like Google Maps interaction

### Video Games

**Performance optimization**:
- Player character: Full detail
- Nearby NPCs: Moderate detail
- Background: Schematic
- Maintains visual quality + performance

## Integration with LOD Systems

### Focus-Driven LOD Allocation

**LOD budget allocation**:
```
LOD_budget(region) = total_budget × focus_map(region)
```

**Dynamic reallocation**:
- Focus shifts → reallocate LOD budget
- High focus regions get more polygons/textures
- Context regions simplified

### Combined with Foveated Rendering

**Multiplicative benefit**:
```
Final_quality = gaze_LOD × focus_map × distance_LOD
```

**Example**:
- Foveal + high focus + near: Full quality
- Peripheral + low focus + far: Minimal quality

## Cross-References

- [concepts/05-focus-schematization.md](../concepts/05-focus-schematization.md) - Concepts
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Gaze-content coupling
- [techniques/00-foveated-rendering.md](00-foveated-rendering.md) - Foveated integration
- [applications/03-multiresolution-viz.md](../applications/03-multiresolution-viz.md) - Visualization apps

## Key Takeaways

1. **Focus maps guide LOD**: Spatial importance distribution drives quality allocation
2. **Schematization creates hierarchy**: Photorealistic focus, abstract context
3. **Smooth transitions essential**: Prevent artifacts, maintain quality perception
4. **Multiple sources**: Gaze, task, salience all contribute to focus maps
5. **Attention guidance**: High contrast naturally directs user attention
6. **Cognitive benefits**: Simplified context reduces load, improves performance

**Design principle**: Generate focus map from gaze/task/salience, apply schematization based on focus value, smooth transitions between levels, update dynamically as focus shifts, combine with other LOD techniques for maximum benefit.
