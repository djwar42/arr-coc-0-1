# Visual Perception in LOD Systems

## Overview

Visual perception research reveals how human vision processes images through preattentive processing, visual memory, and attention mechanisms. Understanding these processes is fundamental to designing effective Level of Detail (LOD) systems that match human perceptual capabilities.

## Primary Sources

From [05-Attention and Visual Memory](../source-documents/05-Attention%20and%20Visual%20Memory%20in%20Visualization%20and%20Computer%20Graphics.md):
- Preattentive processing and target detection (< 200-250ms)
- Five theories: Feature Integration, Textons, Similarity, Guided Search, Boolean Maps
- Change blindness and inattentional blindness
- Visual memory limitations and their impact on graphics

## Key Concepts

### Preattentive Processing

**What it is**: Visual features detected rapidly (< 200-250ms) in a single glance, without eye movements.

**Preattentive visual features**:
- **Color**: Hue, intensity, saturation
- **Form**: Orientation, line length, width, curvature, size, collinearity
- **Motion**: Flicker, direction of motion
- **Spatial position**: 2D position, depth cues (stereopsis, shadows, perspective)

**Target detection patterns**:
- **Pop-out**: Target with unique feature is instantly visible (e.g., red circle among blue circles)
- **Conjunction search**: Target defined by multiple features requires serial search (slower)

### Visual Attention Mechanisms

**Fixation-saccade cycle**:
- Eyes fixate 3-4 times per second for detailed vision
- Detailed vision only possible in small region (thumbnail size at arm's length)
- **Saccade**: Rapid eye movement between fixations (200ms+ to initiate)
- Bottom-up: Visual features guide attention
- Top-down: Goals and tasks guide eye movements

**Selective attention theories**:
1. **Feature Integration Theory** (Treisman)
   - Features processed in parallel
   - Conjunction requires serial attention
   - Illusory conjunctions when attention divided

2. **Texton Theory** (Julesz)
   - Textons: Fundamental texture primitives
   - Boundary detection through texton differences

3. **Guided Search** (Wolfe)
   - Top-down + bottom-up guidance
   - Activation map combines feature salience with task goals

### Visual Memory

**Three memory systems**:

1. **Iconic memory**
   - Ultra-brief (< 500ms)
   - High capacity, sensory-level storage
   - Rapidly decays unless attended

2. **Visual working memory**
   - Limited capacity (3-4 complex objects)
   - Active maintenance through attention
   - Critical for change detection

3. **Visual long-term memory**
   - Vast capacity for gist and scene statistics
   - Supports rapid scene categorization
   - Ensemble coding of feature distributions

### Change Blindness

**What it is**: Failure to detect significant visual changes when change coincides with visual disruption.

**Causes**:
- Lack of focused attention at moment of change
- Limited visual working memory capacity
- Overwriting of previous scene representation

**Implications for LOD**:
- Changes during saccades or blinks go unnoticed
- LOD transitions should occur during viewer motion or attention shifts
- Maintain visual continuity for attended regions

### Inattentional Blindness

**What it is**: Failure to see visible objects when attention focused elsewhere.

**Classic example**: Gorilla experiment - viewers counting basketball passes miss person in gorilla suit walking through scene.

**Implications for LOD**:
- Unattended regions can have significantly reduced detail
- Focus computational resources on task-relevant areas
- Peripheral vision tolerates much lower quality

## Application to LOD Systems

### Perceptual LOD Allocation

**Foveal region** (center of gaze):
- Full detail required
- All preattentive features preserved
- No change blindness tolerance

**Parafoveal region** (near center):
- Moderate detail
- Preserve salient features for attention guidance
- Gradual LOD transitions acceptable

**Peripheral region** (far from center):
- Minimal detail sufficient
- Motion and coarse features only
- Aggressive compression tolerated

### Feature Salience for LOD

**High salience** (demands attention):
- Unique colors in scene
- Motion or flicker
- High contrast edges
- Orientation discontinuities

**Low salience** (can degrade):
- Texture detail in uniform regions
- Subtle color variations
- Static, homogeneous areas

### Transition Timing

**Good times for LOD transitions**:
- During saccades (200-300ms)
- During viewer head motion
- In peripheral vision
- When viewer attention on task (inattentional blindness window)

**Bad times for LOD transitions**:
- During fixation on target
- In foveal region
- When feature is task-relevant
- Immediate post-saccade (change detection active)

## Research Foundations

### Preattentive Vision Studies

**Treisman's Feature Integration**:
- Systematic identification of preattentive features
- Conjunction vs feature search timing
- 200-250ms threshold for preattentive tasks

**Julesz Texton Theory**:
- Texture boundary detection
- Spatial frequency and orientation as primitives
- Parallel processing of texture elements

### Visual Memory Research

**Iconic memory** (Sperling, 1960):
- Partial report superiority
- Brief sensory storage
- Attention-based selection

**Change detection** (Rensink, O'Regan, Simons):
- Mudsplash paradigm
- Flicker paradigm
- Real-world change blindness

### Attention Guidance

**Itti-Koch salience model**:
- Feature maps (color, intensity, orientation)
- Center-surround differences
- Winner-take-all competition

**Task-driven attention** (Wolfe):
- Top-down modulation of feature maps
- Goal-directed search strategies
- Attention history effects

## LOD System Design Principles

### 1. Match Perceptual Capabilities

- Allocate detail based on visual acuity falloff
- Preserve preattentive features in task-relevant areas
- Exploit change blindness for smooth transitions

### 2. Respect Visual Attention

- Track gaze or infer from task
- Maintain high detail at attention focus
- Aggressive LOD in unattended regions

### 3. Leverage Visual Memory Limitations

- Limited working memory (3-4 objects) → many objects can transition simultaneously
- Gist preservation sufficient for scene recognition
- Statistical summaries (ensemble coding) replace individual detail

### 4. Design for Change Blindness

- Transition during saccades or motion
- Gradual transitions (popping avoided)
- Preserve attended object identity

### 5. Exploit Inattentional Blindness

- Task-irrelevant areas tolerate dramatic LOD reduction
- Background simplification during focused tasks
- Dynamic reallocation based on task demands

## Cross-References

- [00-lod-fundamentals.md](00-lod-fundamentals.md) - LOD basics and motivation
- [03-transjective-relevance.md](03-transjective-relevance.md) - Gaze-content coupling
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md) - Practical implementation
- [techniques/01-peripheral-degradation.md](../techniques/01-peripheral-degradation.md) - Head-tracked LOD

## Key Takeaways

1. **Human vision is hierarchical**: Preattentive → attentive → working memory → long-term memory
2. **Attention is selective**: Detailed vision limited to small region, 3-4 times per second
3. **Visual memory is limited**: Only 3-4 complex objects in working memory
4. **Change blindness is pervasive**: Changes go unnoticed without focused attention
5. **Inattentional blindness enables LOD**: Unattended areas can have minimal detail without detection

**Design implication**: LOD systems should mirror human visual processing - high fidelity at attention focus, graceful degradation in periphery, transitions during attention shifts.

---

**See also**: [references/00-glossary.md](../references/00-glossary.md) for definitions of preattentive processing, change blindness, inattentional blindness, and visual working memory.
