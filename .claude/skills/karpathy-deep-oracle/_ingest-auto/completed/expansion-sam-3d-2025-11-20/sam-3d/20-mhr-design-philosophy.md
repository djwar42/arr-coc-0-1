# MHR Overview & Design Philosophy

**Meta Human Reconstruction: Unified parametric model for full-body 3D humans with 321 parameters**

---

## 1. What is MHR?

**Meta Human Reconstruction** (MHR) is Meta's unified parametric human model combining:
- **Body Shape**: 45 shape parameters (height, weight, proportions)
- **Body Pose**: 204 articulation parameters (full-body pose)
- **Facial Expression**: 72 expression parameters (face, mouth, eyes)

**Total**: 321 parameters → Complete 3D human

**Contrast with SMPL:**
- SMPL: Body-only (10 shape + 69 pose = 79 params)
- MHR: Body + Hands + Face (321 params)

---

## 2. Design Philosophy

**Unified Representation:**

MHR unifies previously separate models:
- SMPL (body)
- MANO (hands)
- FLAME (face)

**Benefits:**
- Single model for entire human (no stitching)
- Consistent global coordinate frame
- Simultaneous optimization (body + hands + face)

**Use Cases:**
- Full-body avatars (gaming, VR)
- Expressive characters (animation, film)
- Social VR (body language + facial expressions)

---

## 3. Parameter Breakdown

**45 Shape Parameters:**
- Height, weight, body proportions
- Learned from 3D scan datasets
- Linear blend shape model

**204 Articulation Parameters:**
- 23 body joints × 3 (rotation) = 69 params
- 20 hand joints × 3 (per hand) = 120 params
- 51 face joints × 3 = 15 params (simplified)

**72 Expression Parameters:**
- Facial action units (AU)
- Mouth shapes (visemes for speech)
- Eye gaze, blink, brow movements

---

## 4. Training & Data

**Massive Scale:**
- ~1M training images
- 3.14M 3D meshes
- Diverse poses, shapes, expressions

**Data Sources:**
- Body: COCO, 3DPW, Human3.6M
- Hands: FreiHAND, InterHand2.6M
- Face: 300W, CelebA, FFHQ

---

## 5. ARR-COC-0-1 Integration (10%)

**Unified Human Model for Spatial Grounding:**

MHR enables complete human understanding:
- Body language + gestures + facial expressions
- Relevance realization considers full human context
- Expressive avatars for interactive VR/AR

**Use Cases:**
- Social VR: Reconstruct meeting participants with expressions
- Action understanding: Full-body + hand gestures + facial emotion
- Avatar creation: Personalized 3D avatars from single image

---

**Sources:**
- MHR technical report (Meta AI)
- Unified body+hands+face parametric models
- SMPL, MANO, FLAME comparisons
