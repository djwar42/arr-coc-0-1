# Peripheral Degradation Techniques

## Overview

Head-tracked peripheral degradation reduces LOD in peripheral vision during head motion, exploiting motion blur tolerance and change blindness to enable aggressive quality reduction without detection.

## Primary Sources

From [13-Managing LOD through Head-Tracked Peripheral Degradation](../source-documents/13-Managing%20Level%20of%20Detail%20through%20Head-Tracked%20Peripheral%20Degradation_%20A%20Model%20and%20Resulting%20Design%20Principles%20-%20arXiv.md):
- Head-tracking LOD model
- Design principles for peripheral degradation
- User study results
- Performance metrics

## Key Concepts

### Head Motion as Attention Proxy

**When eye tracking unavailable**:
- Head orientation indicates general attention direction
- Coarser than gaze but still effective
- Widely available (VR headsets, phones with gyroscopes)

**Head vs eye tracking**:
- **Eye**: Precise (0.5-1°), fast (3-4 fixations/sec)
- **Head**: Coarse (5-10°), slower (< 1 movement/sec)
- **Combined**: Optimal (head direction + eye offset)

### Motion Blur Tolerance

**During head motion**:
- Natural motion blur reduces perceived detail
- High-frequency information lost
- Peripheral vision further degraded

**LOD opportunity**:
- Reduce quality during motion
- Restore quality when stationary
- Change blindness during motion masks transitions

## Design Principles

From source 13, key principles for peripheral degradation:

### Principle 1: Gradual Degradation

**Smooth eccentricity falloff**:
```
Quality(angle) = 1.0 - k × angle^α
```
Where angle = angle from head direction, k = scaling factor, α = falloff exponent (typ. 1.5-2.0)

**Not step functions**:
- Abrupt transitions visible
- Creates artifacts at boundaries
- Use smooth Gaussian or sigmoid curves

### Principle 2: Motion-Contingent Reduction

**Reduce during motion**:
- Detect head velocity above threshold (>30°/sec)
- Reduce peripheral LOD (50-75% reduction)
- Aggressive degradation tolerated

**Restore when stationary**:
- Head velocity < 10°/sec → increase LOD
- Gradual restoration (1-2 seconds)
- Hysteresis prevents oscillation

### Principle 3: Preserve Central Region

**Maintain center quality**:
- Central 20-30° always high quality
- Peripheral degradation only beyond this
- Protects task-relevant content

**Adaptive center size**:
- Task-dependent (reading: wider center)
- User-calibrated
- Performance-adaptive

### Principle 4: Exploit Saccadic Masking

**Coordinate with eye movements**:
- Head motion often accompanies saccades
- Combined motion + saccade → maximal tolerance
- Update LOD during these windows

### Principle 5: Temporal Hysteresis

**Prevent thrashing**:
- Threshold for LOD increase > threshold for decrease
- Time delays (100-200ms) before LOD changes
- Smooth interpolation between levels

**Example**:
- Increase LOD: velocity < 10°/sec for 200ms
- Decrease LOD: velocity > 30°/sec immediately

## Implementation Strategies

### Head Tracking LOD Pipeline

**1. Track head orientation**:
- IMU data (gyroscope, accelerometer)
- Compute head velocity and acceleration
- Predict orientation at render time

**2. Generate degradation map**:
```
For each screen pixel:
  angle = angleBetween(pixel_direction, head_forward)
  eccentricity = angle
  quality(pixel) = qualityFunction(eccentricity, head_velocity)
```

**3. Allocate LOD**:
- High quality: Central region
- Medium: Parafoveal (20-45°)
- Low: Peripheral (>45°)
- During motion: Shift all down 1-2 levels

**4. Render and composite**:
- Multi-resolution rendering
- Blend LOD levels
- Final output

### Quality Function

**Stationary head**:
```
Q_static(e) = 1.0 - 0.3 × (e / 45°)^2
```
Where e = eccentricity in degrees

**Moving head** (velocity v in °/sec):
```
motion_factor = min(v / 100°, 1.0)
Q_motion(e) = Q_static(e) × (1.0 - 0.6 × motion_factor)
```

**Result**: Up to 60% quality reduction during fast motion

### Latency Compensation

**Head tracking latency**:
- IMU reading: 1-2ms
- Processing: 1-2ms
- Rendering: 8-11ms
- Display: 5-8ms
- Total: 15-23ms

**Prediction**:
```
head_predicted = head_current + angular_velocity × latency
```

**Benefits**:
- Reduces perceived lag
- LOD map accurate at display time
- Smoother experience

## Performance Benefits

**Rendering cost savings**:

**Stationary head**:
- Central 30°: Full quality (25% of screen)
- Peripheral: 40-60% quality (75% of screen)
- Savings: 30-40% overall

**Moving head**:
- Central 30°: 80% quality
- Peripheral: 20-40% quality
- Savings: 50-70% overall

**Example** (VR headset, 100° FOV):
- Non-degraded: 11ms per frame
- Stationary degradation: 7-8ms per frame
- Motion degradation: 3-5ms per frame

## Quality Validation

### User Studies (from source 13)

**Detection thresholds**:
- Stationary: 40-50% quality reduction detectable in periphery
- Moving (30°/sec): 70-80% reduction tolerated
- Moving (60°/sec): 90% reduction tolerated

**Task performance**:
- Search task: No degradation with peripheral LOD
- Navigation task: Improved (higher frame rate)
- Reading task: Requires larger central region

### Subjective Quality

**Questionnaires**:
- Presence: No significant difference
- Comfort: Slightly improved (less judder)
- Immersion: No change or improved

**Simulator sickness**:
- Reduced with peripheral degradation (higher frame rate)
- Smoother motion
- Less perceptual conflict

## Advanced Techniques

### Predictive LOD Loading

**Anticipate head motion**:
- Detect motion onset (acceleration > threshold)
- Predict motion direction
- Preload LOD for predicted view
- Reduce latency perceived

**Saccade detection** (with eye tracking):
- Head motion + saccade often coupled
- Maximal LOD reduction opportunity
- Combined prediction

### Adaptive Degradation

**Performance-based**:
- Monitor frame rate
- If < target (e.g., 90 fps): Increase degradation
- If > target: Restore quality
- Dynamic quality adjustment

**Content-based**:
- Complex scenes: More aggressive
- Simple scenes: Less degradation
- Per-region analysis

### Stereo Considerations

**VR stereo rendering**:
- Independent degradation per eye
- Convergence point = high quality
- Peripheral can differ between eyes

**Binocular summation**:
- Both eyes degrade similarly
- No perceptual conflict
- Doubled rendering savings

## Integration with Other Techniques

### Head + Eye Tracking

**Combined approach**:
```
eccentricity = angleBetween(pixel, head_forward + eye_offset)
```

**Benefits**:
- Eye offset provides precision
- Head provides stability
- Best of both

### Head + Gaze Prediction

**Task-driven prediction**:
- Learn typical gaze patterns
- Predict fixation during head turn
- Preload LOD for predicted gaze target

### Head + Foveated Rendering

**Two-stage degradation**:
1. Head-based peripheral degradation (coarse)
2. Eye-based foveated rendering (fine)
3. Combined savings multiplicative

## Practical Applications

### Mobile VR

**Limited GPU power**:
- Aggressive peripheral degradation essential
- Motion common (untethered)
- Battery savings significant

### Desktop VR

**High-end systems**:
- Enable higher fidelity in center
- Super-sampling in foveal region
- Maintain performance

### AR Applications

**Outdoor use**:
- Head tracking robust (IMU-based)
- Eye tracking challenging (bright light)
- Peripheral degradation still effective

## Cross-References

- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Head-gaze coupling
- [techniques/00-foveated-rendering.md](00-foveated-rendering.md) - Eye-based approach
- [applications/01-vr-ar.md](../applications/01-vr-ar.md) - VR/AR applications
- [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Combined head+eye

## Key Takeaways

1. **Head tracking enables LOD**: Even without eye tracking, effective peripheral degradation
2. **Motion is opportunity**: 70-90% quality reduction tolerated during head motion
3. **Smooth gradients essential**: Abrupt transitions visible and distracting
4. **Hysteresis prevents thrashing**: Time delays and threshold differences
5. **Combine with eye tracking**: Optimal when both available

**Design principle**: Track head orientation and velocity, reduce peripheral quality during motion, smooth transitions with hysteresis, restore quality when stationary, combine with eye tracking when available for maximum effectiveness.
