# Gaze-Tracking Integration with LOD

## Overview

Eye-tracking enables gaze-contingent rendering where LOD adapts to user attention in real-time. By tracking where users look, systems can allocate maximum detail to foveal regions while aggressively reducing peripheral LOD.

## Primary Sources

From [10-Gaze-aware Displays and Interaction](../source-documents/10-Gaze-aware%20Displays%20and%20Interaction%20-%20SURREAL%20TEAM.md):
- Gaze-tracking hardware and algorithms
- Gaze-contingent rendering techniques
- Foveated displays
- Interaction design for gaze

From [13-Managing LOD through Head-Tracked Peripheral Degradation](../source-documents/13-Managing%20Level%20of%20Detail%20through%20Head-Tracked%20Peripheral%20Degradation_%20A%20Model%20and%20Resulting%20Design%20Principles%20-%20arXiv.md):
- Head motion + gaze combined
- Peripheral degradation during motion
- Perceptual validation

From [18-Visual Attention in 3D Video Games](../source-documents/18-Visual%20Attention%20in%203D%20Video%20Games%20-%20SFU%20Summit.md):
- Attention patterns in interactive 3D
- Task-dependent gaze behavior

## Gaze-Tracking Fundamentals

### Hardware Technologies
From [10-Gaze-aware Displays]:

**Video-based eye tracking**:
- **Camera**: Infrared camera captures eye images
- **Illumination**: IR LEDs create corneal reflections
- **Detection**: Pupil center + corneal reflection = gaze vector
- **Sampling rate**: 60-1000 Hz

**Commercial systems**:
- Tobii Eye Tracker (90-120 Hz, 0.5° accuracy)
- SMI (250 Hz, 0.4° accuracy)
- Built-in VR (Meta Quest Pro, Pico 4 Pro): 90 Hz, 1° accuracy

### Gaze Estimation
**Input**: Eye image from camera
**Output**: 3D gaze vector in world coordinates

**Algorithm**:
1. Detect pupil center (image processing or CNN)
2. Detect corneal reflection (glint)
3. Compute pupil-glint vector
4. Map to gaze angle via calibration
5. Ray-cast gaze vector into 3D scene
6. Determine fixation point on geometry

**Accuracy**: 0.5-1° visual angle (typical)
**Latency**: 10-20ms (hardware + processing)

### Calibration
**Required**: Map eye features to gaze angles (varies per user)

**Process**:
1. Show calibration points on screen (5-9 points)
2. User fixates each point
3. Record eye features (pupil-glint vector)
4. Compute mapping function (polynomial or neural network)

**Frequency**: Once per session, or continuous drift correction.

## Foveated Rendering with Gaze

### Biological Motivation
**Human visual acuity**:
- Fovea (0-2° from fixation): 100% acuity
- Parafovea (2-5°): 50% acuity
- Near periphery (5-30°): 25% acuity
- Far periphery (>30°): <10% acuity

**Implication**: Can render 100% quality in 2° circle, <10% quality beyond 30°.

### Gaze-Contingent LOD Zones
From [10-Gaze-aware Displays]:

**Zone definition**:
```python
def compute_lod_zones(gaze_point, fov):
    """Define LOD zones centered on gaze point."""
    return {
        'foveal': CircularRegion(gaze_point, radius=2°),   # LOD0 (100% quality)
        'parafoveal': AnnulusRegion(gaze_point, 2°, 5°),   # LOD1 (75% quality)
        'near_peripheral': AnnulusRegion(gaze_point, 5°, 15°),  # LOD2 (50% quality)
        'mid_peripheral': AnnulusRegion(gaze_point, 15°, 30°),  # LOD3 (25% quality)
        'far_peripheral': OuterRegion(gaze_point, >30°),   # LOD4 (10% quality)
    }
```

**LOD selection**:
```python
def select_gaze_lod(object, gaze_point):
    """Determine LOD based on angular distance from gaze."""
    angle = compute_angle(object.center, gaze_point, camera.position)

    if angle < 2:
        return LOD0  # Foveal
    elif angle < 5:
        return LOD1  # Parafoveal
    elif angle < 15:
        return LOD2  # Near peripheral
    elif angle < 30:
        return LOD3  # Mid peripheral
    else:
        return LOD4  # Far peripheral
```

**Performance gain**: 3-10x polygon reduction with imperceptible quality loss.

### Gaze Prediction
**Problem**: Eye tracking has 10-20ms latency, eye can move 1-2° in that time.

**Solution**: Predict future gaze position.

**Algorithm**:
```python
def predict_gaze(gaze_history, prediction_time):
    """Predict gaze position at future time using velocity."""
    # Compute gaze velocity from recent history
    velocity = (gaze_history[-1] - gaze_history[-5]) / (5 * frame_time)

    # Linear extrapolation
    predicted_gaze = gaze_history[-1] + velocity * prediction_time

    return predicted_gaze
```

**Improvement**: Reduces perceptible lag from 20ms to <5ms.

## Head Motion + Gaze Integration

### Combined Tracking
From [13-Managing LOD through Head-Tracked Peripheral Degradation]:

**Concept**: During head rotation, peripheral vision is motion-blurred. Combine head velocity with gaze eccentricity for aggressive LOD.

**Model**:
```python
def combined_lod_factor(gaze_eccentricity, head_velocity):
    """Compute LOD reduction based on gaze and head motion."""
    # Base reduction from gaze eccentricity
    gaze_reduction = min(gaze_eccentricity / 30.0, 1.0)

    # Motion boost during head rotation
    motion_boost = min(head_velocity / 180.0, 1.0)  # 180°/s = max

    # Combined: More reduction in periphery during motion
    combined = gaze_reduction * (1.0 + 3.0 * motion_boost)

    return min(combined, 1.0)
```

**Result**: 2-5x additional LOD reduction during head motion without detection.

### IMU + Eye Tracking
**Hardware**: VR headsets with IMU (1000 Hz) + eye tracking (90-120 Hz)

**Data fusion**:
1. Track head orientation (IMU) → 1000 Hz
2. Track gaze direction (eye tracker) → 90 Hz
3. Combine: Gaze in world = Head transform × Eye gaze
4. Compute head angular velocity from IMU
5. Adjust LOD based on gaze eccentricity + head velocity

**Benefit**: Smooth, responsive gaze-contingent LOD even during rapid motion.

## Attention-Driven LOD

### Fixation vs. Saccades
**Fixation**: Eyes stationary (>150ms) → processing visual information
**Saccade**: Rapid eye movement (20-80ms) → repositioning gaze

**LOD strategy**:
- **During fixation**: Full gaze-contingent LOD (foveal = high detail)
- **During saccade**: Reduce global LOD (saccadic suppression = blind during motion)

**Implementation**:
```python
def saccade_lod_adjustment(eye_state):
    if eye_state == FIXATION:
        return 1.0  # Full quality
    elif eye_state == SACCADE:
        return 0.3  # 70% reduction (user is blind during saccade)
    elif eye_state == SMOOTH_PURSUIT:
        return 0.8  # 20% reduction (tracking moving object)
```

**Performance**: Saccade LOD reduction saves 50-70% rendering during eye movements.

### Task-Dependent Gaze Patterns
From [18-Visual Attention in 3D Video Games]:

**Gaze behavior varies by task**:

**Combat**: Tight focus on enemies (small fixation radius, few objects)
→ Boost LOD for central 5°, aggressive reduction elsewhere

**Exploration**: Wide attention (distributed fixations, many objects)
→ Moderate LOD globally, slight boost for fixation

**Navigation**: Path-focused (fixate ground ahead, waypoints)
→ Boost LOD for ground and path, reduce for scenery

**Implementation**: Detect task from gameplay state, adjust LOD zones accordingly.

## Perceptual Validation

### User Studies
From [13-Managing LOD]:

**Methodology**: A/B testing with real users, vary LOD based on gaze.

**Results**:
- <3% detection rate for LOD changes in periphery (>15°)
- <1% detection during saccades
- <10% detection during head motion (>60°/s)

**Conclusion**: Gaze-contingent LOD is perceptually safe when properly tuned.

### Quality Thresholds
**Foveal region (0-2°)**: Zero tolerance (users notice any reduction)
**Parafoveal (2-5°)**: <10% detection rate at 50% LOD reduction
**Peripheral (>15°)**: <5% detection rate at 90% LOD reduction

**Design guideline**: Conservative foveal, aggressive peripheral.

## Smooth Transitions

### Gradual LOD Shifts
**Problem**: Abrupt LOD changes at zone boundaries are noticeable.

**Solution**: Smooth LOD interpolation across boundaries.

**Algorithm**:
```glsl
float compute_smooth_lod(float angle_from_gaze) {
    // LOD ranges
    const float foveal_end = 2.0;
    const float parafoveal_end = 5.0;
    const float near_periph_end = 15.0;
    const float mid_periph_end = 30.0;

    // Smooth interpolation (smoothstep)
    if (angle < foveal_end) {
        return 0.0;  // LOD0
    } else if (angle < parafoveal_end) {
        return smoothstep(foveal_end, parafoveal_end, angle) * 1.0;  // LOD0 → LOD1
    } else if (angle < near_periph_end) {
        return 1.0 + smoothstep(parafoveal_end, near_periph_end, angle) * 1.0;  // LOD1 → LOD2
    } else if (angle < mid_periph_end) {
        return 2.0 + smoothstep(near_periph_end, mid_periph_end, angle) * 1.0;  // LOD2 → LOD3
    } else {
        return 3.0 + min((angle - mid_periph_end) / 30.0, 1.0);  // LOD3 → LOD4
    }
}
```

**Result**: Imperceptible LOD transitions.

### Gaze Stabilization
**Problem**: Eye tracking is noisy (jitter), causing LOD flickering.

**Solution**: Temporal filtering + hysteresis.

**Algorithm**:
```python
def stabilize_gaze(raw_gaze, previous_stable_gaze, history):
    # Exponential moving average
    alpha = 0.3  # Smoothing factor
    smoothed_gaze = alpha * raw_gaze + (1 - alpha) * previous_stable_gaze

    # Hysteresis: Require significant change to update
    distance = length(smoothed_gaze - previous_stable_gaze)
    if distance > 0.5°:  # Threshold
        return smoothed_gaze
    else:
        return previous_stable_gaze  # Keep previous to avoid jitter
```

**Result**: Stable LOD zones despite tracking noise.

## System Design

### Real-Time Pipeline
**Per-frame processing**:
1. **Capture gaze** (eye tracker): 1ms
2. **Predict gaze** (15ms ahead): 0.1ms
3. **Compute LOD zones** (centered on predicted gaze): 0.2ms
4. **Classify objects** (assign LOD per object): 1ms
5. **Render** (variable LOD): 8ms

**Total**: 10.3ms (97 FPS) with gaze-contingent LOD.

### Fallback Strategy
**Problem**: Eye tracking can fail (blinks, poor lighting, calibration drift).

**Fallback**:
- Assume center gaze (common in many tasks)
- Use fixed foveation zones (center = high LOD)
- Revert to distance-only LOD

**Implementation**: Detect tracking confidence, switch seamlessly to fallback.

## Case Studies

### Varifocal VR (Meta Research)
**System**: VR headset with eye tracking + adjustable focus

**LOD integration**:
- Foveated rendering (3-5x speedup)
- Depth-of-field LOD (defocused regions reduced)
- Combined: 5-10x rendering reduction

**Result**: 4K per eye at 90 FPS on mobile GPU.

### FOVE VR Headset
**First consumer eye-tracked VR**

**LOD approach**:
- Circular foveation (5 zones)
- Gaze-contingent asset loading
- Interaction via gaze (look to select)

**Performance**: 2-3x rendering speedup over uniform resolution.

### Microsoft Flight Simulator (future)
**Proposed**: Eye tracking + terrain LOD

**Strategy**:
- Boost terrain LOD where user looks
- Reduce LOD for peripheral scenery
- Gaze-driven asset streaming

**Expected benefit**: 2-4x terrain rendering speedup.

## Interaction Design

### Gaze as Input
From [10-Gaze-aware Displays]:

**Dwell-time selection**: Fixate object >500ms → select
**Gaze + gesture**: Look + hand gesture to confirm
**Implicit interaction**: UI appears where user looks

**LOD implication**: Gazed objects get LOD boost automatically (since fixated).

### Multi-User Gaze
**Challenge**: VR/AR collaboration, multiple users, each with own gaze.

**LOD strategy**:
- Union of all gaze regions → high LOD
- Regions outside all gazes → low LOD

**Result**: Shared experience maintains quality for all users.

## Connection to ARR-COC-VIS

Gaze-tracking + LOD demonstrates attention-driven resource allocation:

**Shared principles**:
- **Attention = Resource**: Gaze center gets detail ↔ Query-relevant patches get tokens
- **Peripheral reduction**: Eccentricity-based LOD ↔ Relevance-based compression
- **Dynamic adaptation**: Gaze moves, LOD follows ↔ Query changes, budgets update
- **Perceptual validation**: Below JND threshold ↔ Below task performance threshold

**Key difference**: Spatial attention (gaze location) vs. semantic attention (query relevance).

**Adaptation**: Foveated LOD zones (2°, 5°, 15°, 30°) map to our relevance-based token budgets (400, 256, 128, 64). Both allocate based on transjective relationship (viewer + content → relevance).

## Summary

Gaze-tracking enables 3-10x rendering reduction by allocating LOD based on visual attention. Our relevance realization framework extends this biological grounding to semantic visual tokens—allocating compression based on query-aware relevance rather than spatial fixation.

**Core insight**: Humans attend selectively (gaze). VLMs should "attend" selectively (relevance). Foveated rendering proves aggressive resource reduction is imperceptible when aligned with attention. We apply this to tokens.

---

*This document synthesizes gaze-tracking research with perceptual LOD techniques, demonstrating effective attention-driven resource allocation for interactive 3D systems.*
