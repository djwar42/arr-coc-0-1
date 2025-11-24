# 45 Shape Parameters for Body Identity

**MHR shape space: height, weight, proportions, and individual body identity**

---

## 1. Shape Parameter Overview

**What Do Shape Parameters Control?**

45 PCA (Principal Component Analysis) dimensions:
- **β₁**: Overall body size (height, weight)
- **β₂**: Body mass index (thin vs heavy)
- **β₃**: Shoulder width
- **β₄-β₄₅**: Fine-grained proportions

**Linear Blend Shape Model:**
- Neutral mesh: M₀ (average body)
- Shape deformation: ΔM = Σᵢ(βᵢ × Sᵢ)
- Final mesh: M = M₀ + ΔM

---

## 2. Key Shape Dimensions

**Top 10 Shape Parameters (by variance explained):**

1. **Height** (~25% variance)
2. **Weight/BMI** (~18% variance)
3. **Shoulder Width** (~12% variance)
4. **Hip Width** (~8% variance)
5. **Torso Length** (~6% variance)
6-10. **Limb proportions, chest/waist ratio** (~15% combined)

**Cumulative:**
- First 10 params: ~85% of shape variance
- First 20 params: ~95% of shape variance
- All 45 params: ~99.5% of shape variance

---

## 3. Shape Identity

**Personalizing 3D Bodies:**

Shape parameters encode individual identity:
- **Example Person A**: β = [0.8, -0.5, 0.3, ...] (tall, thin, broad shoulders)
- **Example Person B**: β = [-0.6, 0.9, -0.2, ...] (short, heavy, narrow shoulders)

**Reconstruction from Image:**
- Input: Photo of person
- Output: 45 shape parameters β
- Generate mesh: M = M₀ + Σᵢ(βᵢ × Sᵢ)

---

## 4. Shape Priors

**Learned from 3D Scans:**

Shape space trained on diverse scans:
- CAESAR (civilian body scans, ~4,000 people)
- 3dMD (high-res scans)
- Commercial databases

**Distribution:**
- Shape params follow Gaussian: β ~ N(0, σ²)
- Most people near origin (average body)
- Outliers: Very tall/short, very thin/heavy

---

## 5. ARR-COC-0-1 Integration (10%)

**Body Identity for Personalized Relevance:**

Shape parameters enable identity recognition:
- Track individuals by body shape (re-identification)
- Personalized avatars (user-specific 3D models)
- Body type understanding (fitness, clothing size)

---

**Sources:**
- MHR shape parameterization
- PCA dimensionality reduction
- CAESAR body scan database
