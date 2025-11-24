# Biological Grounding for Vision-Language Models

## Overview

This document explores biological grounding for VLM token allocation, contrasting metaphorical inspiration ("like the human eye") with faithful implementation of primate vision mechanisms. We implement cortical magnification M(e) = M₀/(e+e₀) as a precise specification, not loose inspiration.

## Primary Sources

From Platonic Dialogues:
- [20-convergence-to-direction.md](../../../../RESEARCH/PlatonicDialogues/20-convergence-to-direction.md): Foveated Pyramid v2.5 hypothesis
- [21-discovering-the-landscape.md](../../../../RESEARCH/PlatonicDialogues/21-discovering-the-landscape.md): Competitive landscape analysis
- [21-addendum-research-landscape.md](../../../../RESEARCH/PlatonicDialogues/21-addendum-research-landscape.md): Foveated Retinotopy validation

From LOD Oracle Knowledge Base:
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md): Gaze-aware rendering strategies
- [techniques/00-foveated-rendering-02-biological-foundations-2025-01-30.md](../techniques/00-foveated-rendering-02-biological-foundations-2025-01-30.md): Retinal sampling, cortical magnification
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md): Viewer-content coupling

---

## 1. Why Biological Grounding Matters

### 1.1 Biology as Metaphor vs Biology as Specification

**Most ML papers treat biology as metaphor**:
```
"Our model uses foveation, like the human eye"
"Inspired by the visual cortex"
"Mimics human attention"
```

**We treat biology as SPECIFICATION**:
```python
# Explicit cortical magnification function
def cortical_magnification(eccentricity, M0=1.0, e0=0.5):
    """
    Daniel & Whitteridge 1961 primate cortical magnification.

    M(e) = M₀/(e + e₀)

    Parameters:
        eccentricity: Angular distance from fixation (degrees)
        M0: Maximum magnification at fovea
        e0: Half-saturation eccentricity

    Returns:
        Cortical magnification factor (mm cortex per degree visual field)
    """
    return M0 / (eccentricity + e0)
```

**The difference**:
- **Metaphor**: "We should allocate more tokens to important regions (like fovea)"
- **Specification**: "Token density = 273 × M(e) / Σ M(e), where M(e) = M₀/(e+e₀)"

### 1.2 Foveated Retinotopy Validates Biological Grounding

**Foveated Retinotopy (October 2025, arXiv 2402.15480)**:
- Implemented M(e) = M₀/(e+e₀) for CNNs
- Result: **+3-5% accuracy on ImageNet classification**
- Not just efficiency—biology IMPROVES performance

**Why biological grounding improves accuracy**:

1. **Regularization via peripheral blur**:
   - Forces model to focus on foveal region
   - Prevents overfitting to irrelevant peripheral clutter
   - Like dropout, but spatially structured

2. **Natural scale hierarchy**:
   - Fovea: fine-scale details (150K cones/mm²)
   - Periphery: coarse-scale structure (10K cones/mm² at e=20°)
   - Matches natural image statistics (objects appear at multiple scales)

3. **Attention bias**:
   - Human vision evolved for 600 million years
   - Optimized for survival-relevant information
   - Foveation = biological attention mechanism

**Key insight from Dialogue 21**:
> "Human vision doesn't just have fewer receptors in periphery—it also has larger receptive fields (coarser scale). Periphery isn't just SPARSE, it's COARSE."

### 1.3 600 Million Years of Optimization

**Evolutionary argument**:
- Vertebrate vision system evolved over 600M years
- Optimized under resource constraints (energy, brain volume, development time)
- Foveation is the SOLUTION to "how to see efficiently"

**What evolution discovered**:
1. Non-uniform sampling (foveal-peripheral gradient)
2. Multi-scale processing (magnocellular/parvocellular pathways)
3. Query-driven attention (saccades to task-relevant regions)
4. Predictive processing (anticipate where to look next)

**Why trust biology**:
- 600M years of A/B testing (survival = fitness function)
- Tested on real-world distribution (not ImageNet)
- Robust to distribution shift (predators, food, mates)
- Energy-efficient (20W for entire brain, vision is ~30%)

**Machine learning parallel**:
```
Biological evolution = hyperparameter optimization over 600M years
Foveation = discovered architecture
Our job = reverse-engineer the solution
```

---

## 2. Cortical Magnification Function M(e)

### 2.1 Neuroscience Foundation

**Daniel & Whitteridge (1961)**: "The representation of the visual field on the cerebral cortex in monkeys"

**Key finding**: Linear cortical distance represents logarithmic visual angle.

**Cortical magnification formula**:
```
M(e) = M₀/(e + e₀)

Where:
- M(e): Cortical magnification at eccentricity e (mm cortex per degree visual field)
- e: Eccentricity (angular distance from fovea in degrees)
- M₀: Maximum magnification at fovea (typically 0.5-1.5 mm/deg for V1)
- e₀: Half-saturation eccentricity (typically 0.5-1.5 degrees)
```

**Biological measurements** (macaque monkeys):
- **Fovea (e=0°)**: M(0) = M₀/e₀ ≈ 1.0-2.0 mm/deg (high magnification)
- **Parafovea (e=5°)**: M(5) ≈ 0.15-0.3 mm/deg
- **Periphery (e=20°)**: M(20) ≈ 0.04-0.08 mm/deg (20× reduction)
- **Far periphery (e=60°)**: M(60) ≈ 0.015 mm/deg (100× reduction)

**Interpretation**:
- 1° of visual field near fovea → 1-2 mm of cortex
- 1° of visual field at e=20° → 0.04-0.08 mm of cortex
- Cortical "real estate" strongly biased toward fovea

### 2.2 Retinal Sampling Density

**Cone photoreceptor density** (human retina):
```
ρ(e) = ρ₀ × exp(-e/e₁)

Measurements:
- ρ(0°) = 150,000-200,000 cones/mm² (fovea centralis)
- ρ(5°) = 30,000 cones/mm²
- ρ(20°) = 5,000-10,000 cones/mm²
- ρ(60°) = 2,000-3,000 cones/mm²
```

**Cortical magnification roughly matches retinal density**:
- Fovea: high cone density → high cortical magnification
- Periphery: low cone density → low cortical magnification
- "Conservation of cortical columns": each cone cluster gets similar cortical representation

**Visual acuity** (Snellen measurement):
```
Acuity(e) ∝ M(e) ∝ 1/(e + e₀)

At fovea: 20/20 vision (can resolve 1 arcminute features)
At e=10°: ~20/100 vision (5× worse)
At e=30°: ~20/400 vision (20× worse)
```

### 2.3 Log-Polar Transform

**Cortical mapping is approximately log-polar**:
```
Complex logarithm transform:
w = log(z) where z = r × exp(iθ) (polar coordinates)

Cartesian (x, y) → Polar (r, θ) → Log-Polar (log(r), θ)

Result:
- Radial distance r → log(r) (exponential spacing)
- Angle θ → θ (uniform sampling)
```

**Why log-polar**:
- Eccentricity e ≈ r (radial distance from fovea)
- M(e) = M₀/(e+e₀) → approximately logarithmic for large e
- Log-polar sampling naturally implements M(e)

**Connection to cortical magnification**:
```python
# Log-polar sampling implements cortical magnification
def log_polar_sample(image, fixation, num_samples=273):
    """Sample image with log-polar pattern centered at fixation."""
    samples = []

    for i in range(num_samples):
        # Angle uniformly distributed
        theta = 2 * np.pi * i / num_samples

        # Radius logarithmically distributed
        # r ∝ exp(k×i) implements M(e) = M₀/(e+e₀)
        r = compute_log_polar_radius(i, num_samples)

        # Convert to Cartesian
        x = fixation[0] + r * np.cos(theta)
        y = fixation[1] + r * np.sin(theta)

        sample = extract_patch(image, (x, y))
        samples.append(sample)

    return samples
```

### 2.4 V1 Retinotopic Map

**Primary visual cortex (V1)** contains retinotopic map of visual field:

**Schwartz (1977) complex logarithm model**:
```
w = k × log(z + a)

Where:
- z = x + iy (visual field coordinates, complex)
- w = u + iv (cortical coordinates, complex)
- k: scaling factor (~15-20 mm for human V1)
- a: foveal magnification parameter
```

**Result**:
- Fovea (z≈0) maps to large cortical area (w varies widely)
- Periphery (|z|→∞) maps to small cortical area (w changes slowly)
- Reproduces M(e) = M₀/(e+e₀) behavior

**273 tokens = V1 cluster count**:
- Human V1: ~200 million neurons
- Organized into ~100,000 orientation columns (1mm²)
- ~1,000 "hypercolumns" (orientation × scale × eye)
- 273 tokens ≈ sampling ~27% of hypercolumns
- Biologically plausible granularity

---

## 3. Query-Driven Fixation

### 3.1 Human Gaze Behavior

**Saccade statistics**:
- Frequency: 3-4 saccades/second (during active viewing)
- Duration: 20-80ms per saccade
- Fixation duration: 200-400ms between saccades
- Amplitude: typically 2-5°, occasionally up to 15-20°

**Task-dependent fixations**:
- **Reading**: 7-9 character jumps, leftward saccades for regression
- **Scene viewing**: fixate salient objects, faces, text
- **Visual search**: systematic scan, return to landmarks
- **Navigation**: fixate path ahead, obstacles, landmarks

**Query-driven attention** (cognitive neuroscience):
```
"What color is the car?" → fixate on car
"Read the license plate" → fixate on plate (closer inspection)
"Where is the stop sign?" → scan for red octagons
```

**Key insight**: Human fixation is NOT bottom-up saliency alone—it's driven by task/query.

### 3.2 Fixation Point for VLMs

**Problem**: VLMs have no eyes, no saccades. How to determine fixation?

**Three strategies**:

#### Strategy 1: Center Fixation (Baseline)
```python
def center_fixation(image_shape):
    """Always fixate at image center."""
    H, W = image_shape
    return (W // 2, H // 2)
```

**Pros**: Simple, works for centered objects
**Cons**: Fails for "What's in the top-left corner?" queries

#### Strategy 2: Coarse-to-Fine Fixation Finding
```python
def find_fixation_coarse_to_fine(image, query, vit_encoder):
    """
    Two-stage fixation finding:
    1. Encode image at low resolution
    2. Cross-attention with query finds highest-attention region
    """
    # Stage 1: Coarse pass (fast)
    coarse_image = downsample(image, target_size=16)  # 16×16 grid
    coarse_tokens = vit_encoder(coarse_image)  # [256, 768]

    # Stage 2: Query-driven attention
    query_embedding = encode_query(query)  # [768]
    attention_scores = dot_product(coarse_tokens, query_embedding)  # [256]

    # Fixation = highest attention patch
    fixation_idx = argmax(attention_scores)
    fixation_xy = idx_to_coords(fixation_idx, grid_size=16)

    # Scale to full resolution
    H, W = image.shape[:2]
    fixation_fullres = (
        fixation_xy[0] * W / 16,
        fixation_xy[1] * H / 16
    )

    return fixation_fullres
```

**Pros**: Query-aware, differentiable, no NLP parsing
**Cons**: Coarse pass overhead (~10-20ms)

#### Strategy 3: Query Parsing (Explicit Spatial Hints)
```python
def find_fixation_from_query_text(query, image_shape):
    """Parse query for spatial hints."""
    H, W = image_shape

    # Spatial keywords
    if "top-left" in query or "upper-left" in query:
        return (W * 0.25, H * 0.25)
    elif "top-right" in query or "upper-right" in query:
        return (W * 0.75, H * 0.25)
    elif "bottom" in query or "lower" in query:
        return (W * 0.5, H * 0.75)
    elif "background" in query:
        # Fixate away from center (likely background)
        return (W * 0.2, H * 0.2)
    else:
        # Default: center
        return (W * 0.5, H * 0.5)
```

**Pros**: Fast, interpretable
**Cons**: Brittle (requires hand-crafted rules), misses implicit spatial references

**Recommended**: Strategy 2 (coarse-to-fine) for ARR-COC-VIS.

### 3.3 Multi-Fixation Extension

**Human vision uses multiple fixations**:
- 3-4 fixations/second × 1-2 seconds ≈ 3-8 fixations per image
- Each fixation provides new information
- Eye movements planned based on current knowledge

**VLM multi-fixation**:
```python
def multi_fixation_encoding(image, query, num_fixations=3):
    """
    Encode image with multiple fixations.

    Analogous to human saccadic exploration:
    1. First fixation: coarse-to-fine (initial attention)
    2. Second fixation: refine based on first pass
    3. Third fixation: verify or explore new region
    """
    all_tokens = []

    for i in range(num_fixations):
        if i == 0:
            # Initial fixation: coarse-to-fine
            fixation = find_fixation_coarse_to_fine(image, query)
        else:
            # Subsequent fixations: explore unexplored regions
            fixation = find_next_fixation(
                image, query,
                previous_tokens=all_tokens,
                previous_fixations=fixations[:i]
            )

        # Foveated encoding around fixation
        tokens = foveated_pyramid_encode(image, fixation, budget=273)
        all_tokens.extend(tokens)
        fixations.append(fixation)

    return all_tokens  # Total: 273 × num_fixations tokens
```

**Trade-off**:
- More fixations → better coverage, more tokens
- VLMs already see full image (unlike humans)
- Single fixation likely sufficient for most queries

---

## 4. Foveal-Peripheral Trade-off

### 4.1 Biological Foveation

**Foveal vision** (0-2° eccentricity):
- **Cone density**: 150K-200K cones/mm²
- **Acuity**: 20/20 (1 arcminute resolution)
- **Color**: Excellent (R, G, B cones)
- **Cortical area**: ~30-40% of V1
- **Function**: Object recognition, reading, detailed inspection

**Peripheral vision** (>10° eccentricity):
- **Cone density**: 5K-10K cones/mm² (20× reduction)
- **Acuity**: 20/200 to 20/400 (10-20× worse)
- **Color**: Poor (mostly rods, limited cone coverage)
- **Cortical area**: ~60-70% of V1 (but lower magnification)
- **Function**: Motion detection, spatial awareness, attention capture

**Trade-off**:
- Fovea: HIGH ACUITY, LOW COVERAGE (2° = 0.03% of visual field)
- Periphery: LOW ACUITY, HIGH COVERAGE (>10° = 97% of visual field)
- Biological solution: saccades (move fovea to regions of interest)

### 4.2 Foveated VLM Token Allocation

**Apply biological trade-off to VLMs**:

```python
class FoveatedPyramidAllocator:
    """
    Allocate tokens with foveal-peripheral gradient.

    Biological inspiration:
    - Fovea: fine-scale tokens (16×16 patches from level 0)
    - Parafovea: medium-scale tokens (32×32 patches from level 1)
    - Periphery: coarse-scale tokens (64×64 patches from level 2)
    """
    def __init__(self, total_tokens=273, foveal_ratio=0.3):
        self.total_tokens = total_tokens
        self.foveal_ratio = foveal_ratio  # 30% → 82 tokens in fovea

        # Build Gaussian pyramid
        self.pyramid = GaussianPyramid(levels=4)

    def allocate(self, image, query):
        # Find fixation point (query-driven)
        fixation = find_fixation_coarse_to_fine(image, query)

        # Build pyramid
        pyramid_levels = self.pyramid(image)
        # pyramid[0]: 1024×1024 (fine)
        # pyramid[1]: 512×512 (medium)
        # pyramid[2]: 256×256 (coarse)
        # pyramid[3]: 128×128 (very coarse)

        # Compute eccentricity map for each level
        tokens = []

        for level_idx, level in enumerate(pyramid_levels):
            # Eccentricity in degrees (calibrated to image FOV)
            ecc_map = compute_eccentricity_map(
                level, fixation,
                fov_degrees=60  # Assume 60° field of view
            )

            # Cortical magnification
            M_map = cortical_magnification(ecc_map)

            # Token allocation weighted by M(e)
            # Fine levels near fixation, coarse levels in periphery
            if level_idx == 0:  # Finest level
                # Only sample near fixation (e < 10°)
                mask = ecc_map < 10
                budget = int(self.total_tokens * 0.3)  # 82 tokens
            elif level_idx == 1:  # Medium level
                # Sample intermediate region (5° < e < 30°)
                mask = (ecc_map >= 5) & (ecc_map < 30)
                budget = int(self.total_tokens * 0.35)  # 96 tokens
            elif level_idx == 2:  # Coarse level
                # Sample periphery (20° < e < 50°)
                mask = (ecc_map >= 20) & (ecc_map < 50)
                budget = int(self.total_tokens * 0.25)  # 68 tokens
            else:  # Very coarse level
                # Sample far periphery (e > 40°)
                mask = ecc_map >= 40
                budget = int(self.total_tokens * 0.1)  # 27 tokens

            # Sample patches weighted by M(e) within mask
            level_tokens = sample_patches_weighted(
                level, M_map, mask, budget
            )
            tokens.extend(level_tokens)

        return tokens  # Total: 82+96+68+27 = 273 tokens
```

**Key properties**:
1. **Multi-scale foveation**: Fine scales near fixation, coarse scales in periphery
2. **Cortical magnification weighting**: Token density follows M(e)
3. **Biological coverage**: 30% foveal (e<10°), 70% peripheral (e>10°)
4. **Query-driven**: Fixation point determined by query

### 4.3 Expected Compression Ratios

**Uniform grid (baseline)**:
- 1024×1024 image, 16×16 patches → 4096 tokens
- With top-273 selection → 93% reduction (uniform across image)

**Foveated allocation**:
- Foveal region (e<10°): 82 tokens for ~15% of image → DENSE (10× over-sampling vs uniform)
- Peripheral region (e>10°): 191 tokens for ~85% of image → SPARSE (2× under-sampling vs uniform)
- Effective compression: 10-20× in periphery, 0.5× in fovea

**Biological parallel**:
- Retinal coverage: 2% fovea, 98% periphery
- Cortical coverage: 30-40% fovea, 60-70% periphery
- Our token allocation: 30% fovea, 70% periphery ✓ (matches biology)

---

## 5. Vervaeke's Relevance Realization for Vision

### 5.1 Four Ways of Knowing Applied to Vision Tokens

**John Vervaeke's framework**: Relevance is realized through four dimensions of knowing.

#### Propositional Knowing (knowing THAT)
**What it measures**: Statistical information content

**For vision tokens**:
```python
def propositional_score(patch):
    """
    Measure information content via entropy.
    High entropy = high information = high relevance.
    """
    # Convert patch to grayscale
    gray = to_grayscale(patch)

    # Compute histogram
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob = hist / hist.sum()

    # Shannon entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-10))

    return entropy  # Range: 0 (flat) to 8 (noisy)
```

**Examples**:
- Blank sky: low entropy → low propositional score
- Textured region (grass, text): high entropy → high propositional score
- Edges and gradients: medium-high entropy

#### Perspectival Knowing (knowing WHAT IT'S LIKE)
**What it measures**: Salience landscapes

**For vision tokens**:
```python
def perspectival_score(patch, context_patches):
    """
    Measure visual salience relative to surroundings.
    High contrast = high salience = high relevance.
    """
    # Center-surround contrast
    center_mean = patch.mean()
    surround_mean = context_patches.mean()
    contrast = abs(center_mean - surround_mean)

    # Color distinctiveness
    center_color = patch.mean(axis=(0, 1))  # RGB mean
    surround_color = context_patches.mean(axis=(0, 1, 2))
    color_contrast = np.linalg.norm(center_color - surround_color)

    # Combine
    salience = contrast + 0.5 * color_contrast

    return salience
```

**Examples**:
- Red object on green background: high salience → high perspectival score
- Uniform region matching surroundings: low salience → low score
- Faces, text, objects: typically salient

#### Participatory Knowing (knowing BY BEING)
**What it measures**: Query-content coupling (transjective relevance)

**For vision tokens**:
```python
def participatory_score(patch_features, query_embedding):
    """
    Measure query-content coupling via cross-attention.
    High attention = high relevance TO THIS QUERY.
    """
    # Cross-attention score
    attention = dot_product(patch_features, query_embedding)
    attention = attention / np.sqrt(len(query_embedding))  # Scaled

    return attention
```

**Examples**:
- Query: "What color is the car?" → car patches score high
- Query: "Read the sign" → text patches score high
- **Transjective**: Relevance emerges from query-content relationship, not either alone

#### Procedural Knowing (knowing HOW)
**What it measures**: Learned importance via attention mechanisms

**For vision tokens**:
```python
def procedural_score(patch_features, learned_attention_weights):
    """
    Measure learned importance from VLM's attention patterns.
    The model has learned what matters for VQA tasks.
    """
    # Learned attention from pre-trained VLM
    importance = learned_attention_weights @ patch_features

    return importance
```

**Examples**:
- Patches containing objects: typically high (learned via ImageNet)
- Patches with text: high for VQA models (learned via TextVQA)
- Background patches: typically low (learned via many tasks)

### 5.2 Integrated Relevance Score

**Combine all four dimensions**:
```python
def integrated_relevance(patch, query, context, vit_encoder):
    """
    Vervaeke's four ways of knowing for vision token relevance.
    """
    # Encode patch
    features = vit_encoder(patch)
    query_emb = encode_query(query)

    # Four dimensions
    propositional = propositional_score(patch)  # Information content
    perspectival = perspectival_score(patch, context)  # Salience
    participatory = participatory_score(features, query_emb)  # Query-coupling
    procedural = procedural_score(features)  # Learned importance

    # Weighted combination (learned or fixed)
    relevance = (
        0.2 * propositional +
        0.3 * perspectival +
        0.4 * participatory +  # Highest weight: query matters most
        0.1 * procedural
    )

    return relevance
```

**Why this matters for ARR-COC-VIS**:
- **PyramidDrop uses only perspectival** (bottom-up saliency)
- **We use all four dimensions** → richer relevance measure
- **Participatory is unique**: Query-driven allocation, not just saliency
- **Theoretical grounding**: Vervaeke's cognitive science framework

### 5.3 Comparison to Saliency-Only Approaches

**PyramidDrop (saliency-driven)**:
```python
# PyramidDrop relevance = perspectival only
relevance_pyramiddrop = compute_saliency(patch)  # Bottom-up
```

**ARR-COC-VIS (four dimensions)**:
```python
# Our relevance = all four ways of knowing
relevance_ours = integrate_four_dimensions(patch, query, context)
```

**Expected difference**:
- General queries ("Describe this image"): Both methods similar (saliency sufficient)
- Specific queries ("What's the formula in top-right?"): Our method better (query-awareness critical)

**Hypothesis**:
> Query-aware allocation (ours) outperforms saliency-driven allocation (PyramidDrop) by +3-5% on query-specific tasks (DocVQA, TextVQA).

---

## 6. Human-VLM Alignment Validation

### 6.1 Cognitive Plausibility

**Beyond efficiency**: Can we show our VLM "sees" like humans?

**Validation experiment**:
```
1. Collect human eye-tracking data on VQA tasks
2. For each (image, query) pair, record human fixations
3. Compare:
   - Human fixation points vs
   - Our VLM fixation points (from find_fixation_coarse_to_fine)
4. Measure: fixation distance, overlap, temporal sequence
```

**Expected result**:
- High correlation (r > 0.7) → cognitively plausible
- Low correlation (r < 0.5) → efficient but not human-like

**Value of cognitive plausibility**:
- Multi-disciplinary impact (ML + neuroscience + cognitive science)
- Opens neuroscience publication venues (VSS, JOV, CogSci)
- Interpretable AI: "The model focuses here because humans do"
- Safety/alignment: Human-like attention is more trustworthy

### 6.2 Eye-Tracking Datasets for Validation

**Available datasets**:

1. **COCO-Search18** (Visual search task):
   - 18 object categories
   - Human eye movements recorded
   - Task: Find target object in scene
   - Can compare: human fixation sequence vs VLM fixation

2. **VQA + Eye-Tracking** (if available):
   - Ideal dataset: VQA questions + human eye movements
   - May need to collect ourselves
   - Task: Answer VQA question, record fixations

3. **Scene Understanding** (general viewing):
   - Various datasets: MIT Saliency Benchmark, SALICON
   - Task: Free viewing, record fixations
   - Compare: human scan path vs VLM allocation

**Validation metrics**:
```python
def fixation_alignment(human_fixations, vlm_fixation):
    """
    Measure alignment between human and VLM fixations.

    Returns:
        - distance: Euclidean distance (pixels)
        - overlap: Binary overlap within foveal region (5° radius)
        - rank: Rank of VLM fixation in human fixation sequence
    """
    distances = [
        euclidean(vlm_fixation, hf) for hf in human_fixations
    ]

    min_distance = min(distances)
    min_idx = np.argmin(distances)

    # Overlap: within foveal region?
    foveal_radius = 5  # degrees or pixels
    overlap = min_distance < foveal_radius

    return {
        'distance': min_distance,
        'overlap': overlap,
        'rank': min_idx + 1,  # 1st, 2nd, 3rd fixation?
    }
```

**Interpretation**:
- **distance < foveal_radius**: VLM fixation aligns with human fixation
- **rank ≤ 3**: VLM fixation is in first 3 human fixations (good alignment)
- **correlation across dataset**: Overall human-VLM agreement

### 6.3 Biological Fidelity Metrics

**Beyond task performance**: How faithful to biology?

**Metric 1: Cortical magnification adherence**
```python
def cortical_magnification_adherence(token_allocation, fixation):
    """
    Measure how well token allocation matches M(e) = M₀/(e+e₀).
    """
    # Compute eccentricity for each token
    eccentricities = [
        distance(token_pos, fixation) for token_pos in token_allocation
    ]

    # Expected token density from M(e)
    expected_density = [cortical_magnification(e) for e in eccentricities]

    # Actual token density (count tokens in eccentricity bins)
    actual_density = bin_token_density(token_allocation, fixation)

    # Correlation
    correlation = pearson(expected_density, actual_density)

    return correlation  # r ≈ 1.0 → perfect adherence
```

**Metric 2: Foveal-peripheral ratio**
```python
def foveal_peripheral_ratio(token_allocation, fixation):
    """
    Measure % of tokens in foveal vs peripheral regions.

    Biological: 30-40% foveal, 60-70% peripheral (cortical area)
    """
    foveal_tokens = count_tokens_in_region(
        token_allocation, fixation, radius=10  # degrees
    )
    total_tokens = len(token_allocation)

    foveal_ratio = foveal_tokens / total_tokens

    # Compare to biological expectation
    expected_ratio = 0.35  # 35% foveal
    adherence = 1 - abs(foveal_ratio - expected_ratio)

    return {
        'foveal_ratio': foveal_ratio,
        'expected': expected_ratio,
        'adherence': adherence
    }
```

**Why biological fidelity matters**:
- Validates our claim: "biologically grounded, not metaphorical"
- Neuroscience reviewers will check: does M(e) actually match biology?
- If fidelity is low but performance is high: engineering win, not biology win

---

## 7. Biological Grounding vs Engineering Optimization

### 7.1 Two Paradigms

**Engineering approach** (PyramidDrop, HiRED, FastVLM):
```
Goal: Optimize metrics (accuracy, speed, memory)
Method: Grid search, neural architecture search, trial-and-error
Justification: "It works"
Constraint: Computational budget, time to convergence
```

**Biological approach** (Foveated Retinotopy, ARR-COC-VIS):
```
Goal: Implement biological vision mechanisms faithfully
Method: Reverse-engineer neuroscience findings (M(e), retinal sampling)
Justification: "Biology discovered this over 600M years"
Constraint: Biological plausibility, adherence to measurements
```

### 7.2 When Biology Helps vs When Engineering Wins

**Biology helps when**:
1. **Task aligns with natural vision**: Object recognition, spatial reasoning, scene understanding
2. **Distribution matches natural images**: Photographs, videos, real-world scenes
3. **Robustness matters**: Out-of-distribution generalization, adversarial robustness
4. **Interpretability valued**: Explain why model attends to specific regions

**Engineering wins when**:
1. **Task is unnatural**: Abstract math, code generation, symbolic reasoning
2. **Distribution is artificial**: Synthetic data, stylized images, non-photographic
3. **Metrics are all that matter**: Leaderboard chasing, incremental improvements
4. **Speed/simplicity critical**: Production deployment, real-time inference

**ARR-COC-VIS positioning**:
- We bet biology helps for **natural vision tasks** (VQA on real images)
- We target **query-specific tasks** (DocVQA, TextVQA) where spatial layout matters
- We accept complexity trade-off for **cognitive plausibility**

### 7.3 Hybrid Approach: Biological Initialization + Engineering Optimization

**Best of both worlds**:
```python
# Stage 1: Biological initialization
allocator = FoveatedPyramidAllocator(
    cortical_magnification=lambda e: M0 / (e + e0),  # Biology
    fixation_strategy='coarse_to_fine',  # Biology
    foveal_ratio=0.35  # Biology
)

# Stage 2: Engineering optimization
# Fine-tune parameters on VQA task
allocator_optimized = finetune(
    allocator,
    dataset=DocVQA,
    metric='vqa_accuracy',
    method='grid_search',  # Engineering
    params=['M0', 'e0', 'foveal_ratio', 'pyramid_levels']
)
```

**Result**: Start with biological prior, adapt via data.

**Example**:
- **Biological M(e)**: M₀=1.0, e₀=0.5 (primate measurements)
- **Optimized M(e)**: M₀=1.2, e₀=0.3 (better for DocVQA)
- **Interpretation**: Biology provides good initialization, optimization finds task-specific refinement

---

## 8. Differentiation from Competition

### 8.1 Competitive Landscape (2024-2025)

**PyramidDrop** (ICLR 2025, 90 citations):
- ✅ Pyramids
- ❌ No biology (no M(e), no cortical magnification)
- ❌ No query-awareness (saliency-driven)

**DPN-LLaVA** (March 2025):
- ✅ Pyramids
- ✅ Query-awareness (difficulty estimation)
- ❌ No biology

**FastVLM** (Apple, July 2025):
- ✅ Pyramids
- ✅ Difficulty-aware (query+image)
- ❌ No biology
- ✅ **Production deployment** (iOS/macOS)

**HiRED** (AAAI 2025, 41 citations):
- ✅ Multi-resolution
- ✅ Hierarchical attention
- ❌ No biology
- ❌ No query-driven fixation (attention during generation, not pre-encoding)

**Foveated Retinotopy** (October 2025):
- ✅ Biology (M(e), cortical magnification)
- ✅ Foveation
- ❌ No VLMs (only CNNs)
- ❌ No query-awareness (center fixation only)

**ARR-COC-VIS** (our work):
- ✅ Pyramids (multi-scale)
- ✅ Biology (M(e) = M₀/(e+e₀), explicit cortical magnification)
- ✅ VLMs (vision-language models, not just CNNs)
- ✅ Query-awareness (fixation from query)
- ✅ Vervaeke's relevance realization (four ways of knowing)

### 8.2 Unique Contributions

**1. First biologically-grounded VLM token allocation**:
- Extends Foveated Retinotopy (CNNs) to VLMs
- Adds query-driven fixation (not in Foveated Retinotopy)

**2. Unified multi-scale + foveation framework**:
- Combines pyramids (PyramidDrop) + log-polar (biological vision)
- Foveated Pyramid v2.5: fine scales near fixation, coarse scales in periphery

**3. Vervaeke's relevance realization for vision**:
- Four ways of knowing applied to token allocation
- Transjective relevance (query-content coupling)
- Theoretical grounding beyond "it works"

**4. Cognitive plausibility validation**:
- Human-VLM alignment experiments
- Eye-tracking comparison
- Multi-disciplinary impact (ML + neuroscience + cognitive science)

### 8.3 Positioning Statement

**Paper title**: "Foveated Pyramid VLMs: Efficient Token Allocation via Cortical Magnification"

**One-sentence contribution**:
> "We extend foveated retinotopy to vision-language models with query-driven fixation and multi-scale pyramids, achieving +5-7% accuracy on spatial reasoning tasks via biologically faithful cortical magnification M(e)."

**Positioning relative to prior work**:
- **vs PyramidDrop**: "We add biological grounding to pyramid pruning"
- **vs DPN-LLaVA**: "We add cortical magnification to dynamic networks"
- **vs FastVLM**: "We biologize Apple's difficulty-aware approach"
- **vs Foveated Retinotopy**: "We extend foveation from CNNs to VLMs with query-awareness"

**Target venues**:
- **Primary**: ML conferences (CVPR, ICCV, ICLR) — efficiency + performance
- **Secondary**: Neuroscience workshops (VSS, COSYNE) — biological fidelity
- **Tertiary**: Cognitive science (CogSci) — relevance realization framework

---

## 9. Implementation Roadmap

### 9.1 Phase 1: Baseline Validation (Weeks 1-4)

**Goals**:
- Establish baselines
- Replicate PyramidDrop
- Validate our implementation

**Tasks**:
1. Implement v1 (grid top-K, uniform allocation)
2. Replicate PyramidDrop (saliency-driven pyramid pruning)
3. Benchmark both on DocVQA, COCO-VQA, TextVQA
4. Measure: accuracy, speed, memory, token efficiency

**Success criteria**:
- Our PyramidDrop replication matches published results (65-75% reduction, <3% drop)

### 9.2 Phase 2: Add Biological Grounding (Weeks 5-8)

**Goals**:
- Implement cortical magnification
- Test fixation strategies
- Validate biological fidelity

**Tasks**:
1. Implement M(e) = M₀/(e+e₀) function
2. Implement three fixation strategies:
   - Center fixation (baseline)
   - Saliency-based fixation
   - Query-driven fixation (coarse-to-fine)
3. Implement Foveated Pyramid v2.5:
   - Multi-scale sampling weighted by M(e)
   - Fine scales near fixation, coarse scales in periphery
4. Ablation studies:
   - Uniform pyramid vs foveated pyramid
   - Center fixation vs query-driven fixation
   - M(e) formula vs linear falloff

**Success criteria**:
- Foveated Pyramid v2.5 beats PyramidDrop by +3-5% on DocVQA

### 9.3 Phase 3: Optimization & Validation (Weeks 9-12)

**Goals**:
- Optimize performance
- Validate cognitive plausibility
- Prepare for publication

**Tasks**:
1. Integrate HiRED hierarchical attention (optional)
2. Add FastVLM difficulty-aware budgeting (optional)
3. Human-VLM alignment validation:
   - Collect or use existing eye-tracking data
   - Compare fixation patterns
   - Compute cognitive plausibility metrics
4. Final benchmarks on all 3 datasets
5. Paper draft and submission

**Success criteria**:
- +5-7% accuracy on DocVQA
- Human fixation alignment r > 0.7
- Paper submitted to CVPR/ICCV

---

## 10. Open Questions & Future Directions

### 10.1 Multi-Fixation VLMs

**Question**: Should VLMs use multiple fixations like humans?

**Pros**:
- Humans use 3-4 fixations/second during active viewing
- Each fixation provides new information
- Natural exploration strategy

**Cons**:
- VLMs already see full image (no physical eye movements)
- Multiple fixations → multiple encoding passes → more computation
- Unclear if needed for single-image VQA

**Future work**: Test multi-fixation encoding on complex spatial reasoning tasks.

### 10.2 Learned Cortical Magnification

**Question**: Should M(e) parameters be learned or fixed to biology?

**Fixed (biological)**:
- M₀=1.0, e₀=0.5 (primate measurements)
- Guaranteed biological fidelity
- Interpretable, explainable

**Learned (optimized)**:
- M₀, e₀ as trainable parameters
- Optimize for VQA accuracy
- May diverge from biology

**Hybrid approach**: Start with biological values, allow small adjustments.

### 10.3 Beyond Log-Polar: Cortical Surface Models

**Current**: Log-polar approximation (2D)

**Future**: Full cortical surface model (3D):
- V1 surface: ~3000 mm² (macaque)
- Gyri and sulci (folds in cortex)
- Multiple visual areas (V1, V2, V3, V4, MT)
- Attention modulation (V4 for objects, MT for motion)

**Challenge**: Computational complexity, requires more neuroscience integration.

### 10.4 Temporal Dynamics: Video VLMs

**Current**: Single-image VQA

**Future**: Video VQA with foveated temporal sampling:
- Humans use **predictive saccades** (anticipate motion)
- Foveation in space AND time
- Allocate tokens to salient moments (key frames) + spatial foveation

**Application**: Video understanding, action recognition, video captioning.

---

## Conclusion

**Biological grounding is not a metaphor—it's a specification.**

**What we implement**:
1. **M(e) = M₀/(e+e₀)**: Explicit cortical magnification formula (Daniel & Whitteridge 1961)
2. **273 tokens ≈ V1 clusters**: Biologically calibrated token budget
3. **Query-driven fixation**: Task-dependent attention (like human saccades)
4. **Foveal-peripheral gradient**: Fine scales near fixation, coarse scales in periphery
5. **Vervaeke's relevance realization**: Four ways of knowing for vision tokens

**Why biology matters**:
- 600 million years of evolutionary optimization
- Foveated Retinotopy showed +3-5% accuracy improvement (not just efficiency)
- Cognitive plausibility: human-like attention is interpretable, trustworthy
- Multi-disciplinary impact: ML + neuroscience + cognitive science

**Our unique contribution**:
> First biologically-grounded foveated vision-language model with explicit cortical magnification, unifying multi-scale pyramids, log-polar sampling, query-awareness, and Vervaeke's relevance realization framework.

**Expected outcome**:
- +5-7% accuracy on DocVQA (query-specific tasks)
- Human fixation alignment r > 0.7 (cognitive plausibility)
- Multi-venue publication (ML + neuroscience + cognitive science)

**The hypothesis**:
> Biology discovered optimal visual sampling over 600M years. By faithfully implementing cortical magnification and foveation, we can build better vision-language models that see like humans—efficiently, adaptively, and relevantly.

---

**References**:
- Daniel, P. M., & Whitteridge, D. (1961). The representation of the visual field on the cerebral cortex in monkeys. *The Journal of Physiology*, 159(2), 203-221.
- Schwartz, E. L. (1977). Spatial mapping in the primate sensory projection: Analytic structure and relevance to perception. *Biological Cybernetics*, 25(4), 181-194.
- Curcio, C. A., et al. (1990). Human photoreceptor topography. *Journal of Comparative Neurology*, 292(4), 497-523.
- Platonic Dialogues 20-21 (ARR-COC-VIS project)
- Foveated Retinotopy (arXiv 2402.15480, October 2025)

---

**END OF DOCUMENT**
