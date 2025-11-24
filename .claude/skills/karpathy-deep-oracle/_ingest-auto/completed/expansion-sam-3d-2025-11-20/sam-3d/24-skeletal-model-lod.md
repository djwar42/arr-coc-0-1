# Skeletal Model & LOD System

**MHR hierarchical skeleton and level-of-detail optimization for real-time performance**

---

## 1. Skeletal Hierarchy

**Kinematic Chain:**

MHR uses forward kinematics:
- Root joint (pelvis) transforms first
- Child joints inherit parent transforms
- Leaf joints (fingertips, toes) at end

**Hierarchy Example:**
```
Pelvis (root)
├─ Spine1
│  ├─ Spine2
│  │  └─ Spine3
│  │     ├─ Neck → Head
│  │     ├─ Left Shoulder → Elbow → Wrist → Hand
│  │     └─ Right Shoulder → Elbow → Wrist → Hand
├─ Left Hip → Knee → Ankle → Foot
└─ Right Hip → Knee → Ankle → Foot
```

---

## 2. Level of Detail (LOD)

**Adaptive Complexity:**

Reduce joints for distant/background humans:
- **LOD 0** (High): All 321 params (close-up)
- **LOD 1** (Medium): Body + simplified hands (120 params)
- **LOD 2** (Low): Body only, no hands/face (69 params)
- **LOD 3** (Minimal): Rigid body (6 params: position + rotation)

**Use Case:**
- VR scene with 20 people
- Nearby person: LOD 0 (full detail)
- Background people: LOD 2-3 (low detail)
- **Performance**: 10× faster

---

## 3. Skinning

**Linear Blend Skinning (LBS):**

Mesh vertices deform with skeleton:
- Each vertex has bone weights
- Vertex position = Σ(wᵢ × Tᵢ × vᵢ)
- wᵢ: Weight for bone i
- Tᵢ: Bone i transformation matrix

**Example:**
- Elbow vertex: 50% upper arm, 50% forearm
- When elbow bends: Vertex smoothly interpolates

---

## 4. Pose Correctives

**Corrective Blendshapes:**

LBS artifacts (e.g., "candy wrapper" effect at elbow):
- Add pose-dependent correctives
- Activate when joint bends >45°
- Smooth deformation

---

## 5. ARR-COC-0-1 Integration (10%)

**Efficient Multi-Person Reconstruction:**

LOD system enables:
- Real-time VR with many people
- Adaptive detail based on relevance
- Focus computation on salient individuals

---

**Sources:**
- MHR skeletal rigging
- LOD hierarchies for real-time rendering
- Linear blend skinning (LBS)
