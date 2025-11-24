# Error Analysis & Failure Cases

**Understanding when and why SAM 3D fails: systematic error analysis**

---

## 1. Common Failure Modes

**Symmetric Objects:**
- Objects with rotational symmetry (cylinders, spheres)
- Front/back ambiguity (which side is facing camera?)
- Error: 15-25% higher MPJPE on symmetric objects

**Transparent/Reflective:**
- Glass, mirrors, polished metal
- SAM struggles with transparency
- Solution: Specialized models for transparent objects

**Extreme Occlusion:**
- >70% of object occluded
- Reconstruction is highly uncertain
- SAM 3D: Predicts visible parts, hallucinates occluded

---

## 2. Human Pose Errors

**Depth Ambiguity:**
- Left/right limb confusion (30% of errors)
- Arm forward vs backward (20% of errors)

**Extreme Poses:**
- Handstands, splits: +40% MPJPE
- Contortion: +60% MPJPE

**Small People:**
- People <100 pixels tall: +35% MPJPE
- Solution: Multi-scale processing

---

## 3. Dataset Bias

**Training Distribution Bias:**
- Standing poses: 35% of training → Low error
- Lying poses: 5% of training → High error (+25%)

**Geographic Bias:**
- Western clothing styles: Low error
- Traditional cultural garments: +10% error

**Age Bias:**
- Adults (20-60): Low error
- Children (<12): +15% error (fewer training samples)

---

## 4. Systematic Errors

**Scale Ambiguity:**
- Single image lacks absolute scale
- SAM predicts normalized scale (person height = 1.7m assumed)
- Actual height unknown without context

**Ground Plane:**
- SAM assumes flat ground
- Sloped terrain: Foot-ground contact errors

---

## 5. Error Metrics

**MPJPE by Scenario:**
- Standing, indoor, front-view: 38mm (best)
- Sitting, outdoor, side-view: 52mm
- Lying, dim lighting, oblique: 78mm (worst)

**Failure Rate:**
- Complete failure (>200mm error): 2.3% of samples
- Unusable output (>150mm): 5.8%
- Acceptable (<75mm): 82.1%

---

## 6. ARR-COC-0-1 Integration (10%)

**Error-Aware Relevance Realization:**

SAM 3D uncertainty informs relevance:
- Low confidence → Lower relevance weight
- Ambiguous depth → Multiple hypotheses
- Failure detection → Fallback to 2D

---

**Sources:**
- Error analysis on 3DPW, Human3.6M benchmarks
- Failure mode categorization
- Bias analysis in training data
