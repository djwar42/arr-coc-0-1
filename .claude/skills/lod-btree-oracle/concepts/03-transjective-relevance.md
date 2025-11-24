# Transjective Relevance in LOD Systems

## Overview

Transject

ive relevance describes LOD selection that emerges from the *relationship* between viewer and content, rather than objective scene properties or subjective preferences alone. It's the perceptual coupling between gaze, task, and visual content that determines optimal detail allocation.

## Primary Sources

From [10-Gaze-aware Displays](../source-documents/10-Gaze-aware%20Displays%20and%20Interaction%20-%20SURREAL%20TEAM.md):
- Gaze-contingent rendering and interaction
- Foveated rendering optimization
- Eye tracking in VR/AR displays
- Gaze direction and prediction

From [13-Managing LOD through Head-Tracked Peripheral Degradation](../source-documents/13-Managing%20Level%20of%20Detail%20through%20Head-Tracked%20Peripheral%20Degradation_%20A%20Model%20and%20Resulting%20Design%20Principles%20-%20arXiv.md):
- Head-tracked LOD management
- Peripheral degradation principles
- User-centered LOD strategies

## Key Concept: Transjective

**Not objective** (in content alone):
- LOD not determined solely by polygon count, texture resolution, or geometric complexity
- Scene properties insufficient to determine rendering quality

**Not subjective** (in viewer alone):
- LOD not based only on user preferences or individual differences
- Viewer characteristics alone don't determine optimal detail

**Transjective** (in viewer-content relationship):
- LOD emerges from coupling between gaze and scene
- Like shark's fitness for ocean - neither shark nor ocean alone, but relationship
- Task, attention, and content interact to determine relevance

## Gaze-Content Coupling

### Foveal-Peripheral Gradient

**Biological grounding**:
- Human retina: 150,000 cones/mm² in fovea, 5,000/mm² at 20° eccentricity
- Visual acuity drops exponentially with eccentricity
- Detail perception limited to ~2° central vision

**LOD allocation**:
- **Fovea** (0-2°): Full detail required
- **Parafovea** (2-10°): Moderate detail, preserve salient features
- **Periphery** (>10°): Minimal detail, motion and coarse structure only

**Gaze-dependent rendering**:
```
LOD_level(region) = f(eccentricity_from_gaze, content_salience, task_relevance)
```

### Eye Tracking for LOD

**Real-time gaze tracking**:
- Modern HMDs: 90-120 Hz eye tracking
- Latency: 10-20ms gaze-to-photon
- Accuracy: 0.5-1.0° visual angle

**Gaze prediction**:
- Saccade destination prediction during saccade
- Task-driven attention models
- History-based prediction (repeated fixations)

**Gaze-contingent updates**:
- Update LOD during saccades (200-300ms)
- Preload high-detail for predicted gaze targets
- Maintain quality at current fixation

## Task-Relevance Coupling

### Task-Driven LOD

**Explicit task knowledge**:
- Navigation task → preserve landmarks, paths, horizon
- Search task → enhance target features, suppress distractors
- Inspection task → full detail at task-relevant objects

**Implicit task inference**:
- Gaze dwell time → prolonged fixation indicates relevance
- Saccade patterns → systematic search vs random exploration
- Interaction history → objects manipulated recently

**Example**: Reading task
- High LOD: Text being read
- Medium LOD: Adjacent lines (saccade targets)
- Low LOD: Distant text, background imagery

### User-Aware Personalization

**Individual differences**:
- Visual acuity variations
- Attention span differences
- Task experience levels

**Adaptive LOD strategies**:
- Calibrate foveal extent per user
- Learn user-specific attention patterns
- Adjust LOD aggressiveness based on tolerance

**From [07-Exploiting LOD-Based Similarity](../source-documents/07-Exploiting%20Lod-Based%20Similarity%20Personalization%20Strategies%20for%20Recommender%20Systems.md)**:
- User modeling for personalized LOD
- Preference learning over time
- Balancing fidelity and performance per user

## Head-Tracked LOD

**Head motion as attention proxy**:
- Head orientation indicates general attention direction
- When eye tracking unavailable, use head tracking
- Coarser but still effective for LOD allocation

**Design principles (from source 13)**:

1. **Peripheral degradation during head motion**
   - Reduce peripheral LOD when head moving
   - Exploit motion blur tolerance
   - Restore detail when head stationary

2. **Latency compensation**
   - Predict head orientation 50-100ms ahead
   - Preload LOD for predicted view
   - Smooth transitions prevent disorientation

3. **Graceful degradation**
   - LOD falloff matches visual acuity
   - No abrupt transitions
   - Imperceptible to user when properly calibrated

## Salience-Guided LOD

**Bottom-up salience**:
- Color contrast, unique hues
- Motion and flicker
- High-frequency edges
- Orientation discontinuities

**Top-down task salience**:
- Task-relevant objects
- Goal-related features
- Learned importance from interaction

**Combined salience map**:
```
Salience(x,y) = α × bottom_up(x,y) + β × top_down(x,y)
LOD(x,y) = threshold(Salience(x,y), gaze_eccentricity(x,y))
```

**From [09-Focus Guided Light Field Saliency](../source-documents/09-Focus%20Guided%20Light%20Field%20Saliency%20Estimation.md)**:
- Focus cues enhance salience estimation
- Depth-of-field simulates foveal attention
- Light field rendering supports gaze-contingent focus

## Temporal Coupling

### Smooth LOD Transitions

**Change blindness windows**:
- Saccades (200-300ms): Excellent transition window
- Head motion: Peripheral changes unnoticed
- Blinks (~150ms): Brief but usable

**Hysteresis in LOD selection**:
- Prevent "thrashing" between LOD levels
- Threshold for increasing detail < threshold for decreasing
- Smooth interpolation during transitions

**Predictive LOD loading**:
- Track gaze velocity and direction
- Preload detail for predicted fixation targets
- Maintain detail at recent fixations (working memory support)

### Attention History

**Working memory support**:
- Maintain high LOD for recently attended objects (3-4 objects)
- Support attentional blink (<500ms)
- Prevent quality degradation during brief look-aways

**Sustained attention detection**:
- Prolonged fixation → critical object, maintain quality
- Repeated refixations → importance signal
- Task-relevant object set → always high LOD

## Implementation Strategies

### Gaze-Contingent Pipeline

**1. Gaze tracking**:
- Capture eye position at 90-120 Hz
- Predict gaze position at render time (compensate latency)
- Map gaze to screen/world coordinates

**2. Eccentricity map generation**:
- Compute distance from gaze for each pixel/object
- Apply visual acuity function
- Generate LOD allocation map

**3. Salience modulation**:
- Enhance LOD for salient regions
- Apply task-relevance weights
- Combine with eccentricity map

**4. LOD selection and rendering**:
- Select appropriate LOD level per object/region
- Render with allocated detail budget
- Smooth transitions during saccades

### Quality Metrics

**Perceptual quality**:
- JND (Just Noticeable Difference) threshold
- User studies on LOD detectability
- Task performance preservation

**Performance metrics**:
- Frame rate maintained (90+ fps for VR)
- Polygon/pixel budget utilization
- Memory footprint

**Coupling quality**:
- Gaze-LOD latency (< 20ms desirable)
- Transition smoothness
- Prediction accuracy (gaze destination)

## Applications to Vision-Language Models

**Relevance to ARR-COC-VIS**:

**Query-visual coupling**:
- Text query analogous to "task"
- Visual content analogous to "scene"
- Token budget allocation = LOD selection

**Transjective token allocation**:
- Not objective (all patches equal detail)
- Not subjective (uniform compression)
- **Transjective**: Query-patch relevance determines token budget

**Vervaeke's participatory knowing**:
- Viewer participates through task/query
- Content participates through features/patches
- Relevance realized through coupling

**Dynamic realization**:
- Like gaze shifting, query context shifts
- LOD updates → token reallocation
- Smooth transitions → graceful degradation

## Cross-References

- [00-lod-fundamentals.md](00-lod-fundamentals.md) - LOD basics
- [02-visual-perception.md](02-visual-perception.md) - Perceptual foundations
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md) - Implementation
- [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Gaze integration

## Key Takeaways

1. **Transjective relevance**: Emerges from viewer-content coupling, not isolated properties
2. **Gaze is key proxy**: Indicates attention focus for LOD allocation
3. **Task matters**: Same scene, different task → different LOD allocation
4. **Biological grounding**: Foveal-peripheral gradient mirrors human vision
5. **Temporal dynamics**: Saccades, blinks, and motion enable smooth transitions
6. **Personalization**: User differences require adaptive strategies

**Design principle**: LOD systems should track or infer user attention, allocate detail based on gaze-content-task coupling, and exploit perceptual windows for smooth transitions.

---

**See also**: [references/00-glossary.md](../references/00-glossary.md) for definitions of transjective, eccentricity, salience, and gaze-contingent rendering.
