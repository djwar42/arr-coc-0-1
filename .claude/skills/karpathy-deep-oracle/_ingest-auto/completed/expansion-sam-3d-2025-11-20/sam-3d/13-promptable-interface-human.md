# Promptable Interface for Human 3D Reconstruction

**Interactive prompting for SAM 3D Body: clicks, keypoints, and segmentation guidance**

---

## 1. Promptable Human Reconstruction Overview

**What is Promptable Human 3D Reconstruction?**

SAM 3D Body extends SAM's promptable interface to 3D human mesh recovery:
- **Click Prompts**: User clicks on body parts → guide reconstruction
- **Keypoint Prompts**: Provide 2D joint locations → refine 3D pose
- **Mask Prompts**: Segment person region → improve reconstruction accuracy
- **Box Prompts**: Bounding box around person → localization hint

**Why Promptable?**

Traditional HMR is fully automatic - no user control. Promptable interface enables:
- **Correction**: Fix automatic errors interactively
- **Disambiguation**: Resolve ambiguous poses (is the arm up or down?)
- **Prioritization**: Focus computation on specific body parts
- **Zero-Shot Adaptation**: Adapt to unusual poses/viewpoints via prompts

---

## 2. Click-Based Prompts

**Point Prompts for Body Parts:**

User clicks on image → guide 3D reconstruction:
- Click on shoulder → Refine shoulder joint position
- Click on hand → Focus on hand pose estimation
- Click on foot → Improve ground contact

**Click Types:**
- **Positive Click**: "This is the body part I care about"
- **Negative Click**: "This is NOT part of the person" (background suppression)
- **Multiple Clicks**: Provide multiple constraints simultaneously

**Example Workflow:**
1. Automatic HMR produces initial 3D mesh
2. User clicks on incorrectly positioned elbow
3. System refines mesh to match clicked 2D location
4. Updated 3D mesh rendered

---

## 3. 2D Keypoint Prompts

**Providing Joint Locations:**

User provides 2D keypoint annotations:
- **Full Skeleton**: 17 COCO keypoints (shoulders, elbows, wrists, etc.)
- **Partial Skeleton**: Only ambiguous joints (e.g., just wrists)
- **Critical Points**: High-impact joints (hips, shoulders)

**Keypoint Prompt Format:**
- [x, y] pixel coordinates
- Optional confidence scores [0-1]
- Joint type identifier (COCO joint IDs)

**Integration:**
- Keypoints treated as **hard constraints** in optimization
- 2D reprojection loss weighted heavily on prompted joints
- SMPL pose parameters adjusted to match keypoints

---

## 4. Segmentation Mask Prompts

**Binary Person Masks:**

SAM generates instance segmentation → guide HMR:
- **Foreground Mask**: Which pixels belong to person
- **Background Mask**: Which pixels are NOT person
- **Part Masks**: Segment individual body parts (arms, legs, torso)

**Mask-Guided HMR:**
- Silhouette loss: Rendered mesh must match foreground mask
- Part-level guidance: Segment left arm → refine left arm pose
- Occlusion handling: Masked regions can be ignored

**Workflow:**
1. SAM segments person from background
2. Binary mask fed to HMR network
3. Reconstruction constrained to match mask boundary
4. Improved accuracy on occluded/ambiguous regions

---

## 5. Bounding Box Prompts

**Coarse Localization:**

Simple rectangular box around person:
- **Tight Box**: Minimal padding (faster inference)
- **Loose Box**: Extra context (better accuracy)
- **Multiple Boxes**: Multi-person scenes

**Box Prompt Uses:**
- Person detection (where is the person?)
- Scale estimation (how big is the person?)
- Crop-and-center pre-processing

**SAM Integration:**
- Box prompt → SAM generates mask → Mask guides HMR
- Chain: Box → Mask → 3D Mesh

---

## 6. Interactive Refinement Loop

**Iterative Prompt-and-Refine:**

User-in-the-loop workflow:
1. **Automatic Prediction**: Initial 3D mesh from image
2. **User Review**: Identify errors visually
3. **Provide Prompt**: Click/keypoint/mask to guide correction
4. **Refinement**: System updates 3D mesh based on prompt
5. **Repeat**: Iterate until satisfactory

**Convergence:**
- Typically 1-3 iterations sufficient
- Diminishing returns after 5 iterations
- Each iteration: ~100ms overhead

---

## 7. Prompt Types Comparison

**Effectiveness by Prompt Type:**

| Prompt Type | Speed | Accuracy Gain | Use Case |
|-------------|-------|---------------|----------|
| Click       | Fast  | +2-5% MPJPE   | Quick corrections |
| Keypoint    | Medium| +5-10% MPJPE  | Pose refinement |
| Mask        | Slow  | +10-15% MPJPE | Occlusion handling |
| Box         | Fastest| +1-3% MPJPE | Localization only |

**Best Practices:**
- Start with Box (localization)
- Add Mask if occlusion present
- Use Clicks for quick fixes
- Use Keypoints for precision refinement

---

## 8. Multi-Prompt Fusion

**Combining Multiple Prompts:**

Users can provide multiple prompts simultaneously:
- Box + Mask (localize + segment)
- Mask + Clicks (segment + refine specific joints)
- Keypoints + Mask (pose + silhouette constraints)

**Fusion Strategy:**
- Weighted loss: L_total = w1*L_box + w2*L_mask + w3*L_keypoint
- Adaptive weighting: More confident prompts get higher weight
- Conflict resolution: Prioritize keypoints > mask > box

---

## 9. Prompt Ambiguity & Multi-Hypothesis

**Handling Ambiguous Prompts:**

Some prompts allow multiple valid 3D interpretations:
- **Example**: Click on wrist → Arm up or down?
- **Solution**: Generate multiple hypotheses (top-3 poses)
- **User Selection**: Show 3 options, user picks best

**Multi-Hypothesis Generation:**
- Diverse pose sampling from SMPL prior
- All hypotheses match prompt constraints
- Ranked by likelihood (learned prior)

---

## 10. ARR-COC-0-1 Integration (10%)

**Promptable Spatial Grounding for Relevance Realization:**

Promptable interfaces enable dynamic relevance allocation:

1. **User-Guided Attention**: Clicks signal "this body part matters"
2. **Disambiguation**: Prompts resolve relevance ambiguity (which interpretation?)
3. **Interactive Relevance**: Refine spatial grounding iteratively
4. **Zero-Shot Adaptation**: Prompts adapt to novel poses/contexts

**Use Cases:**
- VQA: "Where is the person's left hand?" → Click prompt guides attention
- Action recognition: "Are they waving?" → Hand keypoints focus relevance
- Gesture understanding: Click-based prompts for fine-grained gestures

**Training Integration:**
- Prompt-conditioned relevance realization (clicks → spatial focus)
- Multi-hypothesis selection as relevance ranking
- Interactive refinement loop trains adaptability

---

**Sources:**
- SAM promptable interface (clicks, boxes, masks)
- 2D keypoint-guided 3D pose estimation literature
- Mask-guided HMR architectures
- Interactive 3D reconstruction user studies
- ARR-COC-0-1 project spatial grounding concepts
