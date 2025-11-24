# Quality Control & Human Verification

**Ensuring high-quality 3D training data through automated checks and manual review**

---

## 1. Quality Control Pipeline

**Automated Checks:**
1. Mesh validation (watertight, manifold)
2. Keypoint detection confidence > 0.7
3. Reprojection error < 50mm
4. SMPL fit error < 100mm

**Manual Review:**
- Top 10% of data manually inspected
- Annotators verify 3D mesh quality
- Flag problematic samples for removal

---

## 2. Mesh Quality Metrics

**Watertight:**
- No holes in mesh
- All edges belong to exactly 2 faces

**Manifold:**
- Valid topology (no self-intersections)
- Orientable surface

**Vertex Count:**
- Range: 5K-100K vertices
- Too few: Blocky appearance
- Too many: Slow rendering

**Texture Quality:**
- Resolution: 512×512 minimum
- UV unwrapping (no seams)

---

## 3. Annotation Quality

**2D Keypoint Quality:**
- Confidence scores per joint
- Visibility flags (occluded joints)
- Reprojection test (3D→2D consistency)

**3D Mesh Quality:**
- Fit error (SMPL vs GT scan)
- Penetration check (body parts intersect?)
- Ground contact (feet on floor?)

**Inter-Annotator Agreement:**
- Multiple annotators label same samples
- Cohen's kappa > 0.8 (high agreement)

---

## 4. Outlier Detection

**Automated Outliers:**
- Statistical outliers (3σ from mean)
- Extreme poses (unlikely SMPL parameters)
- Corrupted images (artifacts, blur)

**Manual Outliers:**
- Incorrect gender labels
- Wrong activity annotations
- Mislabeled keypoints

**Action:**
- Remove outliers (<1% of data)
- Re-annotate borderline cases

---

## 5. Data Audits

**Regular Audits:**
- Quarterly data quality reviews
- Check for labeling drift over time
- Verify demographic balance

**Audit Findings:**
- 2023 Q1: 0.8% corrupt meshes removed
- 2023 Q2: Improved keypoint detector (5% accuracy gain)
- 2023 Q3: Added 200K diverse samples (age, ethnicity)

---

## 6. ARR-COC-0-1 Integration (10%)

**Quality Data for Reliable Relevance:**

High-quality training data ensures:
- Accurate spatial grounding (low error)
- Trustworthy relevance scores (no outliers)
- Robust zero-shot performance

---

**Sources:**
- Mesh quality metrics (manifold, watertight)
- Annotation verification protocols
- Inter-annotator agreement statistics
