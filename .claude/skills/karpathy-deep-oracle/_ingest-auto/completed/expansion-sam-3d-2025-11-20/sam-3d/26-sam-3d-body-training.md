# SAM 3D Body Training: ~8M Human Images

**Human-specific training data for body mesh recovery**

---

## 1. Dataset Scale

**~8M Human Images:**
- Body-focused datasets
- Diverse poses, shapes, clothing
- Indoor + outdoor environments

**Ground Truth:**
- SMPL/SMPL-X parameters (pseudo-GT)
- 2D keypoints (detected)
- 3D keypoints (triangulated from multi-view)

---

## 2. Data Sources

**COCO (150K):**
- General-purpose object detection
- Human instance segmentation
- 17 keypoints per person

**3DPW (60 sequences):**
- 3D Poses in the Wild
- Outdoor activities (walking, sitting)
- IMU + video → 3D ground truth

**Human3.6M (3.6M frames):**
- Indoor mocap studio
- High-quality 3D pose annotations
- 11 actors, 17 activities

**AMASS (40+ hours motion):**
- Aggregated mocap databases
- Diverse activities (dance, sports, everyday)
- SMPL pose parameters

**Additional:**
- FreiHAND (hands): 130K images
- 300W (face): 600 images
- InterHand2.6M: 2.6M hand images

---

## 3. Training Strategy

**Multi-Dataset Training:**
- Sample from all datasets each batch
- Weighted by dataset size
- Ensures diverse pose coverage

**Data Augmentation:**
- Random crops (simulate different distances)
- Horizontal flip (left/right symmetry)
- Color jitter (lighting robustness)
- Synthetic occlusion (random rectangles)

---

## 4. Pseudo Ground Truth

**SMPL Fitting:**
- Many datasets lack 3D mesh GT
- Fit SMPL to 2D keypoints → pseudo-GT
- Noisy but sufficient for training

**Iterative Refinement:**
- Train model on pseudo-GT
- Use trained model to generate better pseudo-GT
- Repeat (self-improvement loop)

---

## 5. ARR-COC-0-1 Integration (10%)

**Human-Centric Spatial Grounding:**

8M diverse human images enable:
- Robust pose estimation (all poses)
- Zero-shot generalization (unusual poses)
- Action understanding (activities in context)

---

**Sources:**
- COCO, 3DPW, Human3.6M, AMASS datasets
- SMPL pseudo-ground-truth fitting
- Multi-dataset training strategies
