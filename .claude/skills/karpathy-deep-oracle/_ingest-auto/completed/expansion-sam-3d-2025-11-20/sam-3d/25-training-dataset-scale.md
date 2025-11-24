# Training Dataset Scale: ~1M Images, 3.14M Meshes

**Massive-scale 3D training data for SAM 3D Objects**

---

## 1. Dataset Overview

**Scale:**
- ~1M training images
- 3.14M 3D meshes
- 200+ object categories
- Indoor + outdoor scenes

**Purpose:**
Train foundation model for zero-shot 3D reconstruction from single images.

---

## 2. Data Sources

**Synthetic Data (~70%):**
- Objaverse (800K+ 3D models)
- Rendered from multiple viewpoints
- Perfect ground truth (known 3D mesh)

**Real-World Data (~30%):**
- CO3D dataset (Common Objects in 3D)
- Multi-view captures of real objects
- SfM-reconstructed meshes

---

## 3. Data Diversity

**Object Categories:**
- Furniture (chairs, tables, sofas)
- Vehicles (cars, bikes, planes)
- Electronics (phones, laptops)
- Food (fruits, dishes)
- Animals (stylized 3D models)

**Viewpoints:**
- 360° coverage (all angles)
- Top-down, side, oblique views

---

## 4. Mesh Quality

**3.14M High-Quality Meshes:**
- Average: 10K-50K vertices per mesh
- Textured (materials, colors)
- Watertight (no holes)
- Manifold (valid topology)

**Quality Control:**
- Automated filtering (degenerate meshes removed)
- Manual inspection (top 10% of data)

---

## 5. Training Strategy

**Curriculum Learning:**
1. Simple objects (cubes, spheres) - early training
2. Complex objects (chairs, cars) - mid training
3. Articulated objects (humans) - late training

**Data Augmentation:**
- Random viewpoints
- Lighting variations
- Background compositing

---

## 6. ARR-COC-0-1 Integration (10%)

**Dataset Diversity for Spatial Grounding:**

Diverse training enables zero-shot relevance:
- Novel object categories (not in training)
- Unusual viewpoints (oblique, top-down)
- Complex scenes (clutter, occlusion)

---

**Sources:**
- Objaverse 3D model repository
- CO3D (Common Objects in 3D) dataset
- SAM 3D technical report training details
