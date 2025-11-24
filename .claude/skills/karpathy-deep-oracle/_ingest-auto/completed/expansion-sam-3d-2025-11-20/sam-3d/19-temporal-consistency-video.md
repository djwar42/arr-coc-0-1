# Temporal Consistency in Video for SAM 3D Body

**Smooth 3D human tracking across frames: temporal coherence, jitter reduction, and trajectory smoothing**

---

## 1. Temporal Consistency Challenge

**Per-Frame vs Temporal Reconstruction:**

**Per-Frame HMR:**
- Process each frame independently
- No temporal information
- Result: **Jittery** reconstruction (pose jumps between frames)

**Temporal HMR:**
- Use previous/next frames
- Enforce smooth transitions
- Result: **Smooth** trajectories (consistent pose over time)

**Why Temporal Matters:**
- Videos show continuous motion
- Human poses change gradually (not abruptly)
- Jitter is visually distracting and unrealistic

---

## 2. Sources of Jitter

**Why Per-Frame Reconstructions Are Jittery:**

1. **Depth Ambiguity**: 2D→3D is ambiguous, different depths per frame
2. **2D Keypoint Noise**: Detector jitters (±2-5 pixels)
3. **Left-Right Flips**: Arm swaps (left/right confusion)
4. **Pose Multimodality**: Multiple 3D poses fit same 2D keypoints

**Example:**
- Frame t: Right arm detected at (x, y)
- Frame t+1: Right arm detected at (x+3, y-2) (detector noise)
- Result: 3D arm position jumps (jitter)

---

## 3. Temporal Smoothing Techniques

**Post-Processing Smoothing:**

**Moving Average Filter:**
- Average joint positions over window (e.g., 5 frames)
- Reduces high-frequency jitter
- Latency: Half window size (2-3 frames)

**Kalman Filtering:**
- Predict joint trajectory (physics-based motion model)
- Correct prediction with observation (2D keypoints)
- Optimal smoothing (minimizes error + jitter)

**Savitzky-Golay Filter:**
- Polynomial smoothing (preserves sharp motions)
- Better than moving average (less lag)

**Limitation:**
- Post-processing doesn't fix depth ambiguity
- Can over-smooth (blur fast motions)

---

## 4. Temporal Networks

**End-to-End Temporal Modeling:**

**RNN/LSTM:**
- Recurrent network processes sequence
- Hidden state encodes temporal context
- Output: Smooth pose sequence

**Temporal Convolutional Networks (TCN):**
- 1D convolution over time
- Receptive field spans multiple frames
- Parallelizable (faster than RNN)

**Transformer (Temporal Attention):**
- Self-attention over frame sequence
- Captures long-range dependencies
- State-of-the-art temporal modeling

**Example Architecture:**
- Input: T frames of 2D keypoints
- TCN: 1D conv over time dimension
- Output: T frames of smooth 3D poses

---

## 5. Optical Flow for Consistency

**Tracking Body Parts Across Frames:**

Optical flow tracks pixel motion:
- Input: Frame t and Frame t+1
- Output: Displacement vectors (per pixel)

**Usage in HMR:**
- Track 2D keypoints using flow (reduce detector noise)
- Warp previous 3D mesh to current frame
- Refine warped mesh with current 2D keypoints

**Benefit:**
- Enforces pixel-level correspondence
- Reduces jitter from keypoint detector

**Example:**
- Frame t: Detect right hand at (x, y)
- Optical flow: Right hand moved (+5, -3) pixels
- Frame t+1: Expect right hand at (x+5, y-3)
- If detector says (x+10, y+2) → likely noise, trust flow

---

## 6. Physics-Based Constraints

**Enforcing Physical Realism:**

Human motion follows physics:
- **Inertia**: Body parts don't teleport (momentum)
- **Gravity**: Falling objects accelerate downward
- **Biomechanics**: Joints have limited acceleration

**Physics Losses:**
- **Velocity Constraint**: Limit per-frame velocity
- **Acceleration Constraint**: Smooth acceleration
- **Ground Contact**: Foot velocity = 0 when on ground

**Example:**
- Frame t: Foot at height h=0.1m
- Frame t+1: Foot at height h=0.8m
- Physics: Impossible jump (0.7m in 1/30 sec)
- Solution: Constrain foot to ground (h=0)

---

## 7. Multi-Frame Optimization

**Batch Optimization Over Time:**

Instead of per-frame inference, optimize entire sequence:
- Input: T frames (e.g., T=30 for 1 second video)
- Optimize: Pose parameters for all T frames simultaneously
- Constraints: Temporal smoothness + 2D reprojection

**Loss Function:**
- L_reproj: 2D keypoint reprojection error
- L_smooth: Temporal smoothness (adjacent frames similar)
- L_physics: Physics constraints (velocity, acceleration)
- L_total = L_reproj + λ_smooth * L_smooth + λ_physics * L_physics

**Benefit:**
- Globally optimal trajectory (not greedy per-frame)
- Enforces smoothness explicitly

**Limitation:**
- Slow (batch optimization is expensive)
- Not real-time (requires full sequence)

---

## 8. Tracking & Re-Identification

**Handling Occlusions & Exits:**

In video, people can:
- **Occlude**: Temporarily disappear (behind object)
- **Exit**: Leave frame entirely
- **Re-Enter**: Come back into view

**Tracking Strategy:**
- **Re-ID Network**: Match person across frames (by appearance)
- **Trajectory Association**: Link detections across frames (Hungarian algorithm)
- **Occlusion Handling**: Interpolate during brief occlusions

**Example:**
- Frame 10: Person A exits frame
- Frames 11-20: Person A absent
- Frame 21: Person A re-enters
- Re-ID: Match frame 21 to frame 10 (same person)
- Interpolate frames 11-20 (smooth trajectory)

---

## 9. Temporal Consistency Metrics

**Measuring Smoothness:**

**Acceleration Error:**
- Measure joint acceleration (second derivative of position)
- Lower = smoother
- Metric: Mean Acceleration Error (MAE)

**Jitter Metric:**
- Per-joint position variance over time
- Lower = less jitter

**MPJPE Temporal:**
- Standard MPJPE (position error) + temporal consistency term
- Penalizes accurate but jittery poses

**Example:**
- Method A: MPJPE=45mm, Acceleration=8.2 m/s²
- Method B: MPJPE=50mm, Acceleration=3.1 m/s²
- Method B is smoother (preferred for video)

---

## 10. ARR-COC-0-1 Integration (10%)

**Temporal Spatial Grounding for Relevance Realization:**

Temporal consistency enables dynamic relevance:

1. **Motion Attention**: Relevance follows motion (moving objects salient)
2. **Predictive Grounding**: Anticipate future positions (smooth trajectories)
3. **Action Understanding**: Actions unfold over time (temporal context needed)
4. **Persistent Relevance**: Track relevant objects across occlusions

**Use Cases:**
- VQA: "What did the person do?" → Requires temporal aggregation
- Action recognition: "Are they dancing?" → Dance is temporal pattern
- Tracking: "Follow the person in red" → Maintain identity over time

**Training Integration:**
- Temporal transformer for sequence modeling
- Relevance realization tracks salient objects over time
- Occlusion-robust tracking (interpolate during brief occlusions)

---

**Sources:**
- Temporal smoothing (Kalman, Savitzky-Golay filters)
- RNN/LSTM/TCN temporal networks
- Optical flow for HMR consistency
- Physics-based motion constraints
- Multi-frame batch optimization
- Tracking and re-identification
- ARR-COC-0-1 project spatial grounding concepts
