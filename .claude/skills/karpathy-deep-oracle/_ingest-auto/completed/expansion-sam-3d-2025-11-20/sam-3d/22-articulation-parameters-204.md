# 204 Articulation Parameters for Full-Body Pose

**MHR pose space: 23 body + 40 hand + 51 face joints with axis-angle rotations**

---

## 1. Articulation Parameter Breakdown

**204 Pose Parameters:**

**Body (69 params):**
- 23 joints × 3 rotations (axis-angle)
- Shoulders, elbows, hips, knees, spine, neck, etc.

**Hands (120 params):**
- Left hand: 20 joints × 3 = 60 params
- Right hand: 20 joints × 3 = 60 params

**Face (15 params, simplified):**
- Jaw, eyebrows, eyelids (simplified from 51 face joints)

---

## 2. Kinematic Tree

**Body Skeleton:**
- Root: Pelvis (6 DOF: 3 rotation + 3 translation)
- Spine: 3 joints (lower, mid, upper)
- Arms: Shoulder, elbow, wrist (per side)
- Legs: Hip, knee, ankle (per side)
- Head: Neck, head

**Hand Skeleton (per hand):**
- Thumb: 4 joints (CMC, MCP, IP, Tip)
- Fingers (4): 4 joints each (MCP, PIP, DIP, Tip)

---

## 3. Axis-Angle Representation

**3D Rotation per Joint:**

Each joint has 3 rotation parameters:
- Axis-angle: [θₓ, θᵧ, θ_z]
- Direction: Rotation axis
- Magnitude: Rotation angle

**Example:**
- Shoulder rotation: [0.5, -0.3, 0.1]
- Means: Rotate 0.6 radians around axis (0.5, -0.3, 0.1)

---

## 4. Pose Priors

**Biomechanical Constraints:**

Not all 204-dimensional poses are valid:
- Joint limits (elbow can't bend backwards)
- Pose distribution (some poses common, others rare)

**Learned Priors:**
- Trained on motion capture data (AMASS, CMU Mocap)
- Gaussian mixture model (GMM) over pose space
- Prevents impossible poses

---

## 5. ARR-COC-0-1 Integration (10%)

**Fine-Grained Pose Understanding:**

204 parameters enable:
- Hand gestures (sign language, pointing)
- Facial expressions (emotion, speech)
- Full-body actions (sports, dance)

Relevance realization at joint-level granularity.

---

**Sources:**
- MHR articulation parameterization
- Axis-angle rotations
- AMASS motion capture database
