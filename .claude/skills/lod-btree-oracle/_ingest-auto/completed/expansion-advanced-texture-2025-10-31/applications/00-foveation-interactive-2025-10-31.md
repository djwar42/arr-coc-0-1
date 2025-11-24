# Computational Foveation: Interactive Visualization with Semantic Gaze

## Overview

Computational foveation is a rendering optimization technique that concentrates visual processing resources on regions of interest while reducing quality in peripheral areas. This mirrors human vision, where the fovea (central retina) processes high-resolution detail while peripheral vision detects motion and context with lower acuity.

In the context of ARR-COC-VIS, the query becomes a "semantic gaze"—directing which visual regions deserve maximum token allocation, transforming static foveation into query-aware relevance realization.

## Foveation Fundamentals

### Human Vision Model

The human visual system operates with extreme resolution inequality:

- **Fovea**: ~2 degrees of visual angle, 100% resolution (30-40 pixels per degree acuity)
- **Parafovea**: ~5 degrees, ~60% resolution
- **Periphery**: >5 degrees, ~10% resolution
- **Total coverage**: ~180 degrees horizontal field of view

From [Tobii Blog - What is Foveated Rendering](https://www.tobii.com/blog/what-is-foveated-rendering) (accessed 2025-10-31):
- Our brains blend high-resolution central focus with medium and low-resolution peripheral context
- Perceptual degradation across field of view is non-linear and exploitable
- Users typically don't notice quality differences in rendered periphery

### Two Foveation Strategies

**Fixed Foveated Rendering (FFR)**:
- Assumes user attention at display center
- Hardcoded resolution zones (center 100%, middle 60%, periphery 10%)
- Implementable on any device without eye tracking
- Suboptimal user experience when attention drifts

**Dynamic Foveated Rendering (DFR)**:
- Uses real-time eye tracking to locate gaze point
- Renders only gazed region at full resolution
- Remaining areas scale progressively from medium to low quality
- Requires accurate, low-latency eye tracking
- Benchmark results: 60-72% GPU shading load reduction (Tobii)

From [Tobii DFR Research](https://www.tobii.com/blog/what-is-foveated-rendering) (accessed 2025-10-31):
- DFR on Meta Quest achieves 60% average GPU load reduction
- Frame rates remain stable (90+ FPS minimum)
- Enables realistic shading on resource-constrained hardware
- Reduces heat, power consumption, cooling noise

## Gaze-Contingent Processing

### Eye Tracking Technology

Modern eye tracking for VR/AR achieves:

- **Temporal Resolution**: 250 Hz to 1000 Hz sampling (Tobii, SMI systems)
- **Spatial Accuracy**: 0.5-1.5 degrees visual angle
- **Latency**: <20 milliseconds eye-to-screen
- **Robustness**: Works across diverse eye shapes, lighting conditions

From [Ruofei Du - Kernel Foveated Rendering](https://duruofei.com/projects/foveatedrendering/) (accessed 2025-10-31):
- Log-polar mapping transforms cartesian screen coordinates to eye-centric gaze-contingent space
- Saccadic suppression: eye movements happen too fast to see (0-200ms), allowing quality transitions
- Perceptual fusion at ring boundaries makes foveation transitions invisible

### Gaze-Contingent Applications

**VR/XR Content Delivery**:
- Apple Vision Pro (2023): Dynamic foveated rendering in visionOS
- Meta Quest Pro (2022): Eye-tracked foveated rendering
- PlayStation VR2 (2023): 3.6x performance improvement with foveation
- HTC Vive Pro Eye (2019): First consumer headset with eye tracking + foveation

From [Wang, Shi, Liu - Foveated Rendering State-of-Art Survey](https://research.google/pubs/3d-kernel-foveated-rendering-for-light-fields/) (Computational Visual Media, 2023):
- 68+ recent papers on foveated rendering techniques
- Applications: VR gaming, scientific visualization, 360° video streaming, volumetric rendering

**Remote Rendering & Streaming**:
- Gaze-contingent transport reduces network bandwidth by only transmitting high-detail foveal region
- Edge-compute systems render at server, stream foveated output to lightweight headsets
- 40x compression achieved for hologram delivery with foveated compression

## ARR-COC Query as Semantic Gaze

### Conceptual Bridge

In ARR-COC-VIS, the query provides semantic coordinates analogous to eye position in visual foveation:

| Visual Foveation | ARR-COC Relevance |
|---|---|
| Eye gaze point | Query content/semantics |
| Foveal high-res region | High LOD patches (400 tokens) |
| Parafoveal medium-res | Medium LOD patches (200 tokens) |
| Peripheral low-res | Low LOD patches (64 tokens) |
| Eye tracking latency | Query processing latency |
| Visual saccades (eye jumps) | Query reformulation/refinement |

### Semantic Gaze Implementation

**Query Embedding as Gaze**:
- Query text/image mapped to embedding space
- Cosine similarity = semantic "gaze direction"
- Patches scoring high similarity = foveal region
- Patches with moderate similarity = parafoveal
- Remaining patches = periphery

**Multi-Query Gaze Fusion**:
- Complex queries split into sub-queries (like eye attention switching)
- Each sub-query creates semantic salience map
- Integrate maps with opponent processing (compress vs. particularize tension)
- Allocate tokens based on combined relevance landscape

**Saccadic Attention Dynamics**:
- Quick shifts between different query aspects = semantic saccades
- Buffer patches from previous gaze points (temporal coherence)
- Manage token budgets across sequential queries (query chains)

### Computational Advantages

**Relevance-Aware Compression**:
- Only allocate high tokens to semantically relevant regions
- Reduces wasted computation on irrelevant patches
- Aligns token usage with actual information need

**Interactive Refinement**:
- User reformulates query = semantic gaze redirects
- Patch LOD adjusts dynamically based on new semantic focus
- Fast response: ~100ms query-to-render pipeline

**Perceptual Alignment**:
- Humans intuively understand query-driven compression
- Foveation feels natural when semantic gaze matches visual salience
- Reduces artifacts from arbitrary LOD decisions

## Technical Implementation Patterns

### Log-Polar Mapping in Semantic Space

Adapt visual log-polar transform to feature space:

```
Semantic log-polar(patch, query_embedding):
  // Angular component: patch semantic direction
  angle = arctan2(patch.salience_y, patch.salience_x)

  // Radial component: relevance distance in embedding space
  relevance_distance = 1 - cosine_similarity(patch.embedding, query)

  // Log scale emphasizes foveal region (low distance)
  log_radius = log(1 + relevance_distance)

  return (angle, log_radius)
```

### Ring-Based LOD Allocation

Structure patches in semantic rings:

- **Ring 0 (Foveal)**: Top-K most relevant patches → 400 tokens each
- **Ring 1 (Parafoveal)**: Next-M relevant patches → 200 tokens each
- **Ring 2 (Peripheral)**: Remaining patches → 64 tokens each

Ring boundaries determined by:
- Query importance distribution (how peaked?)
- Available token budget (total tokens to allocate)
- Content structure (how many "important" regions?)

### Perceptual Masking During Transitions

Exploit semantic saccadic suppression:

- When query changes, allow brief transition period (100ms)
- During transition, prioritize temporal coherence over perfect relevance
- Reduces visual glitches in token reallocation
- Mimics human perception of smooth gaze transitions

## Applications in Interactive Visualization

### Multi-Query Image Exploration

User explores image with successive queries:

1. Query 1: "Find people"
   - Patches containing people → Foveal (high tokens)
   - Patches with backgrounds → Parafoveal/Peripheral

2. Query 2: "What is everyone wearing?"
   - Clothing regions → Foveal
   - Faces, scenery → Parafoveal

3. Query 3: "Describe expressions"
   - Face/eye regions → Foveal
   - Bodies, environment → Parafoveal

Each query creates fresh semantic gaze, reallocating tokens dynamically.

### Scientific Visualization

For multi-variate volumetric data:

- **Query**: "Show high-pressure regions"
- Semantic gaze directs tokens to pressure-dense areas
- Temperature, velocity data in parafoveal regions (context)
- Geometry/skeleton in peripheral regions (awareness only)

Reduces tokens needed for comprehensive understanding by 3-4x.

### 360-Degree Video Streaming

User watches immersive video; head gaze + query semantics combine:

- Head tracking = visual gaze (standard foveation)
- Current scene description = semantic gaze
- When user looks left and scene says "action happening on right":
  - Visual gaze → left (head position)
  - Semantic gaze → right (scene context)
  - Network stream includes both foveal regions
  - Enables anticipatory streaming for saliency awareness

## Challenges and Open Problems

### Temporal Coherence

Challenge: Overly aggressive token reallocation causes visible flickering.

Solutions (from research):

- Smooth LOD transitions over 2-3 frames
- Maintain hysteresis (patches don't downgrade immediately)
- Temporal-aware opponent processing (exploit vs. explore tension)

### Query Interpretation

Challenge: Not all queries map cleanly to visual salience.

Example: Query "What's the mood?" doesn't highlight specific patches—distributed judgment.

Solutions:

- Use attention maps from multi-modal CLIP/LLaVA models
- Allow soft (probabilistic) gaze patterns, not just discrete rings
- Combine multiple relevance scorers (propositional + perspectival + participatory)

### Latency Sensitivity

Challenge: Eye tracking latency >50ms causes prediction errors.

Solutions:

- Predictive gaze: forecast eye movement (saccade target prediction)
- Latency-aware streaming: buffer redundant high-detail patches
- Grace period: maintain high tokens while gaze uncertain

### Privacy and Gaze Data

Concern: Eye-tracking data reveals attention patterns (sensitive information).

Mitigation:

- On-device processing: compute gaze locally, never transmit raw eye coordinates
- Differential privacy: add noise to gaze maps before using for adaptive rendering
- User controls: let user disable gaze-contingent features

## Recent Research (2024-2025)

From research landscape:

- **FovealNet** (2025): AI-driven gaze tracking with event-based cropping
- **Scene-aware Foveation** (2024): Foveation adapts to image content, not just gaze
- **Neural Foveated Super-Resolution** (2023): Deep learning reconstructs peripheral detail from low tokens
- **Locomotion-aware Foveation** (2023): Accounts for user movement speed affecting gaze stability

Convergence: Foveation increasingly integrates *semantic*, *motion*, and *learning-based* components.

## ARR-COC Integration Points

### Knowing Component
Propositional knowing: Query embedding measures relevance mathematically (cosine similarity)

Perspectival knowing: Visual salience maps reveal what "stands out" to humans about patches

Participatory knowing: Query-content coupling creates transjective relevance (not in query alone, not in image alone)

### Balancing Component
Foveation naturally balances four opponent tensions:

- **Compress ↔ Particularize**: Foveal detail vs. peripheral awareness
- **Exploit ↔ Explore**: Deep attention on query-relevant regions vs. broad context
- **Focus ↔ Diversify**: Concentration on fovea vs. distributed processing across rings
- **Fast ↔ Accurate**: Quick LOD transitions vs. stable relevance estimates

### Attending Component
Relevance realization maps query-aware salience to token budgets, creating adaptive visual hierarchies.

### Realizing Component
Execute foveated pipeline: validate patches, compress appropriately, allocate tokens, render features.

## Sources

**Research Papers & Surveys:**
- [Wang, Shi, Liu - Foveated Rendering: A State-of-the-Art Survey](https://www.sciencedirect.com/science/article/pii/S0097849321002211) - Computational Visual Media (2023), cited by 68+ papers
- [Meng, Du, Zwicker, Varshney - Kernel Foveated Rendering](https://duruofei.com/projects/foveatedrendering/) - ACM I3D 2018, most-read in PACMCGIT
- [Du, Meng, et al. - 3D-Kernel Foveated Rendering for Light Fields](https://research.google/pubs/3d-kernel-foveated-rendering-for-light-fields/) - Google Research, TVCG 2020
- [Guenter, Finch, Drucker, Tan, Snyder - Foveated 3D Graphics](https://www.microsoft.com/en-us/research/publication/foveated-3d-graphics/) - Microsoft Research, ACM TOG (cited 602 times)

**Industry Resources:**
- [Tobii - What is Foveated Rendering](https://www.tobii.com/blog/what-is-foveated-rendering) - Blog post on DFR technology (accessed 2025-10-31)
- [Tobii - Eye Tracking and Dynamic Foveated Rendering e-book](https://www.tobii.com/resource-center/reports-and-papers/eye-tracking-and-dynamic-foveated-rendering)
- [Meta Store - Eye Tracked Foveated Rendering Documentation](https://developers.meta.com/horizon/documentation/unreal/unreal-eye-tracked-foveated-rendering/) (accessed 2025-10-31)
- [Unity - XR Foveated Rendering Manual](https://docs.unity3d.com/6000.2/Documentation/Manual/xr-foveated-rendering.html) (accessed 2025-10-31)

**Wikipedia Reference:**
- [Foveated Rendering - Wikipedia](https://en.wikipedia.org/wiki/Foveated_rendering) - Overview, history, applications (accessed 2025-10-31)

**Product Implementations:**
- Apple Vision Pro (2023): visionOS dynamic foveated rendering
- Meta Quest Pro (2022): Eye-tracked foveated rendering
- PlayStation VR2 (2023): 3.6x performance improvement
- HTC Vive Pro Eye (2019): First consumer foveated HMD

**Related Concepts:**
- Gaze-contingency paradigm (psychology of attention)
- Log-polar coordinate transforms (mathematics)
- Saccadic suppression (neuroscience of eye movement)
- Deep learning-based gaze prediction (2024-2025 frontier)

## Implementation Roadmap

1. **Integrate Query-to-Relevance Mapping**: Map query embeddings to semantic salience scores
2. **Implement Ring-Based LOD Allocation**: Cluster patches into foveal/parafoveal/peripheral rings
3. **Add Temporal Smoothing**: Reduce flickering during LOD transitions via hysteresis
4. **Validate Against Baselines**: Compare uniform LOD vs. semantic gaze LOD on accuracy/efficiency
5. **User Study**: Test whether semantic foveation improves exploration experience vs. fixed LOD
