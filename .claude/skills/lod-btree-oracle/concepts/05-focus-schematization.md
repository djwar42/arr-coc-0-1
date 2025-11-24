# Focus and Schematization Techniques

## Overview

Focus+context visualization uses partial schematization to accentuate regions of interest while preserving context. This technique selectively abstracts less important content, creating visual hierarchy that guides attention and enables effective LOD strategies.

## Primary Sources

From [00-Accentuating focus maps via partial schematization](../source-documents/00-Accentuating%20focus%20maps%20via%20partial%20schematization%20-%20Pure.md):
- Focus+context design principles
- Schematization techniques
- Selective detail vs abstraction
- Visual hierarchy creation

## Key Concepts

### Focus+Context Paradigm

**Definition**: Presentation technique showing detailed focus region in context of simplified overview.

**Three regions**:
1. **Focus**: Area of primary interest, full detail
2. **Context**: Surrounding area, abstracted but recognizable
3. **Transition**: Smooth gradient between focus and context

**Benefits**:
- Maintain spatial relationships
- Reduce cognitive load
- Guide visual attention
- Enable efficient rendering

### Schematization

**What it is**: Abstraction that preserves essential structure while removing detail.

**Schematization spectrum**:
```
Photorealistic → Simplified → Abstract → Iconic → Symbolic
```

**Examples**:
- **Photorealistic**: High-resolution texture, detailed geometry
- **Simplified**: Flat shading, reduced polygons
- **Abstract**: Line drawing, basic shapes
- **Iconic**: Symbols representing objects
- **Symbolic**: Pure text or abstract representation

**Partial schematization**: Different regions at different abstraction levels.

### Accentuation Strategies

**Techniques to emphasize focus**:

1. **Detail contrast**
   - Focus: Full detail
   - Context: Simplified geometry, low-res textures
   - Sharp or gradual transition

2. **Color/saturation**
   - Focus: Full color saturation
   - Context: Desaturated or grayscale
   - Highlights importance visually

3. **Lighting**
   - Focus: Realistic lighting, shadows
   - Context: Flat shading, ambient only
   - Directs attention via contrast

4. **Stylization**
   - Focus: Photorealistic rendering
   - Context: Sketch, cartoon, or line art
   - Clear visual distinction

5. **Opacity/blur**
   - Focus: Sharp, fully opaque
   - Context: Blurred or semi-transparent
   - Depth-of-field analogy

## Implementation Strategies

### Focus Map Generation

**Input sources**:
- Gaze tracking: Eye position → focus region
- Task analysis: Task-relevant objects → focus
- User interaction: Recently manipulated → focus
- Salience: Visually salient regions → potential focus

**Focus map properties**:
- Spatial distribution: 2D map over image/scene
- Values: 0.0 (full context) to 1.0 (full focus)
- Smooth gradients: Prevent abrupt transitions
- Temporal coherence: Smooth changes over time

**Mathematical representation**:
```
Focus(x,y) = f(distance_from_gaze, salience, task_relevance)
```

**Example**:
```
Focus(x,y) = exp(-dist²/2σ²) × salience_weight × task_weight
```

### Schematization Levels

**Level 0 (Full detail)**:
- Photorealistic rendering
- High-poly geometry
- High-resolution textures
- Complex materials (PBR)
- Dynamic lighting and shadows

**Level 1 (Simplified)**:
- Moderate poly reduction (50%)
- Mid-resolution textures
- Simplified materials
- Baked lighting

**Level 2 (Abstract)**:
- Aggressive poly reduction (10-20% orig)
- Flat colors or simple textures
- Cel shading or flat shading
- No shadows

**Level 3 (Iconic)**:
- Billboard representations
- Symbolic shapes
- Solid colors
- No texture detail

**Level 4 (Symbolic)**:
- Pure symbols or text
- Minimal geometry
- Highest abstraction

### Transition Design

**Smooth blending**:
```
Final_render(x,y) = focus_render × Focus(x,y) + context_render × (1 - Focus(x,y))
```

**Gradient types**:
- **Gaussian**: Smooth, natural falloff
- **Linear**: Simple, predictable
- **Sigmoid**: Sharper focus region, smooth periphery
- **Custom**: Task-specific, learned from data

**Transition zone size**:
- **Narrow** (< 5° visual angle): Sharp focus, clear boundary
- **Wide** (10-20°): Gradual, less noticeable
- **Adaptive**: Based on content complexity

## Application Domains

### Geographic Visualization

**Focus+context maps**:
- Focus: Detailed street-level view
- Context: Schematic road network
- Examples: Google Maps, transit maps

**LOD strategy**:
- Focus region: Full geographic detail
- Near context: Major roads, landmarks
- Far context: Schematic, labeled regions

### Scientific Visualization

**Volume rendering**:
- Focus: Isosurface with texture
- Context: Transparent volume, low sampling
- Guides exploration of 3D data

**Flow visualization**:
- Focus: Dense streamlines, detailed flow
- Context: Sparse streamlines, directional arrows
- Reduces visual clutter

### Architectural Visualization

**Building models**:
- Focus: Detailed facade, realistic materials
- Context: Simple blocks, flat colors
- Enables large scene navigation

**Urban planning**:
- Focus: Proposed building (photorealistic)
- Context: Existing structures (schematic)
- Highlights changes clearly

### Video Games

**NPC rendering**:
- Focus: Main character (high detail)
- Near NPCs: Moderate detail
- Distant NPCs: Billboards or low-poly

**Environment**:
- Focus: Interactive objects (detailed)
- Context: Background scenery (simplified)
- Optimizes performance

## Perceptual Benefits

### Attention Guidance

**Focus region attracts attention**:
- Contrast with simplified context
- Perceptual pop-out effect
- Guides visual search

**Reduces cognitive load**:
- Less competing detail
- Clear visual hierarchy
- Faster information extraction

### Change Detection

**Focus region changes visible**:
- High detail preserves subtle changes
- Context changes less noticeable
- Supports dynamic content

**Context provides stability**:
- Simplified context changes less
- Maintains spatial orientation
- Smooth camera motion

## Integration with LOD Systems

### Dynamic Focus Reallocation

**Gaze-driven focus**:
```
1. Track gaze position (90-120 Hz)
2. Update focus map (smooth update)
3. Reallocate LOD based on focus map
4. Render with focus+context LOD
```

**Smooth transitions**:
- Hysteresis: Focus region "sticky"
- Temporal filtering: Smooth focus map over time
- Predictive: Anticipate gaze shifts

### Rendering Pipeline

**Multi-pass rendering**:

**Pass 1**: Focus region (full quality)
- High LOD geometry
- Detailed textures
- Full lighting

**Pass 2**: Context region (simplified)
- Low LOD geometry
- Schematized appearance
- Simplified lighting

**Pass 3**: Composition
- Blend based on focus map
- Apply transition zone
- Final output

**Single-pass alternative**:
- Focus map as shader input
- LOD selection in vertex shader
- Appearance blending in fragment shader

## Temporal Considerations

### Focus Persistence

**Working memory support**:
- Maintain focus for recently attended objects (3-4 objects)
- Gradual fade-out (not abrupt)
- Support brief look-aways

**Saccade handling**:
- Maintain previous focus during saccade
- Update focus post-saccade
- Smooth transition (200-300ms)

### Animated Focus

**Use cases**:
- Guided tutorials (animated focus)
- Storytelling (directorial control)
- Alert systems (draw attention to events)

**Implementation**:
- Animated focus map keyframes
- Smooth interpolation
- Override gaze during animation

## Cross-References

- [02-visual-perception.md](02-visual-perception.md) - Perceptual foundations
- [03-transjective-relevance.md](03-transjective-relevance.md) - Gaze-content coupling
- [techniques/04-focus-maps.md](../techniques/04-focus-maps.md) - Implementation details
- [applications/03-multiresolution-viz.md](../applications/03-multiresolution-viz.md) - Visualization applications

## Key Takeaways

1. **Focus+context**: Shows detail in context without losing overview
2. **Schematization spectrum**: Photorealistic → abstract, choose level per region
3. **Accentuation techniques**: Detail contrast, color, lighting, stylization
4. **Smooth transitions**: Critical for perceptual quality
5. **Dynamic focus**: Gaze or task-driven reallocation
6. **Attention guidance**: Focus region naturally attracts attention

**Design principle**: Identify focus regions (gaze, task, or salience), apply appropriate schematization levels to context, smooth transitions prevent artifacts, dynamically reallocate as focus shifts.

---

**See also**: [references/00-glossary.md](../references/00-glossary.md) for definitions of focus+context, schematization, and accentuation.
