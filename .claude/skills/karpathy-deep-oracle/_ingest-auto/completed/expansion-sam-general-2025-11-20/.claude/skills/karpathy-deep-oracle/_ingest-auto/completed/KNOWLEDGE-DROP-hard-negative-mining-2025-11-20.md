# SAM 3: Hard Negative Mining Strategies

**PART 27/42 - Hard Negative Mining and Selection**

**Date**: 2025-11-20
**Source**: SAM 3 paper, SA-Co dataset methodology

---

## What are Hard Negatives?

**Hard negatives** = Visually similar concepts that are **NOT present** in the image

### Contrast with Easy Negatives

**Easy negative**: "airplane" when image shows cats
- Visually completely different
- Model easily distinguishes

**Hard negative**: "yellow taxi" when image shows "yellow school bus"
- Same color (yellow)
- Similar shape (vehicle)
- Model struggles to distinguish

**SA-Co uses 30 hard negatives per image** (on average)

---

## Why Hard Negatives Matter

### Performance Impact

| Hard Negatives/Image | CGF1 | IL-MCC | pmF1 | Improvement |
|----------------------|------|--------|------|-------------|
| 0 | 31.8 | 0.44 | 70.2 | baseline |
| 5 | 44.8 | 0.62 | 71.9 | +41.0% |
| 10 | 46.9 | 0.65 | 72.5 | +47.5% |
| **30** | **49.2** | **0.68** | **72.3** | **+54.5%** |

**Largest gain**: IL-MCC (image-level recognition) improves 54.5%!

**Key insight**: Hard negatives teach model "what is NOT the concept" → better concept discrimination.

---

## Hard Negative Generation Strategies

### Strategy 1: Ontology-Driven (Wikidata)

**Method**: Use Wikidata knowledge graph to find related concepts

**Example: "yellow school bus"**

```
Wikidata graph:
yellow school bus
    ├─ is-a: vehicle
    ├─ is-a: bus
    ├─ has-color: yellow
    └─ used-for: student transportation

Related concepts (potential hard negatives):
    ├─ Same type, different color: "red bus", "blue bus"
    ├─ Same color, different type: "yellow taxi", "yellow car"
    ├─ Same use, different type: "school van", "school train"
    └─ Similar appearance: "tour bus", "city bus"
```

**Selection criteria**:
- Visual similarity score > 0.7 (CLIP embeddings)
- Conceptual overlap ≥ 1 attribute (color, type, or use)

**Ontology categories**:
- **Color variants**: "red apple" → "green apple", "yellow apple"
- **Size variants**: "large dog" → "small dog", "medium dog"
- **Pattern variants**: "striped cat" → "spotted cat", "solid cat"
- **Type variants**: "sports car" → "sedan", "SUV", "truck"

### Strategy 2: Llama-Based Proposal

**Method**: Multimodal LLM analyzes image and proposes hard negatives

**Prompt to Llama**:
```
Image contains: yellow school bus
Propose 30 visually similar concepts that are NOT in this image.
Prioritize concepts that:
1. Share attributes with "yellow school bus" (color, shape, type)
2. Could plausibly appear in similar scenes
3. Are challenging to distinguish visually
```

**Llama output**:
```
Hard negatives:
1. yellow taxi (same color, similar size)
2. yellow delivery truck (same color, vehicle type)
3. red school bus (same type, different color)
4. yellow fire truck (same color, emergency vehicle)
5. school van (same use, smaller)
...
30. yellow sports car (same color, different type)
```

**Advantages**:
- Contextual awareness (scene-appropriate negatives)
- Diverse proposals (not just ontology siblings)
- Novel combinations ("yellow fire truck" may not be in ontology)

### Strategy 3: Model-Based Mining

**Method**: Use current SAM 3 model to find **failure cases** (false positives)

**Process**:
1. Run SAM 3 on validation set
2. Collect false positives (model predicted concept when absent)
3. Add false positives as hard negatives to training data
4. Retrain model

**Example**:
- Model predicts "elephant" (0.45 confidence) when only "rhinoceros" present
- Add "elephant" as hard negative to rhinoceros images
- Next iteration: Model learns to distinguish elephants from rhinos

**Iterative improvement**:
- Round 1: Collect 50K false positives → add as hard negatives
- Round 2: Model improves, new 30K false positives → add as hard negatives
- Round 3: Diminishing returns (~10K new false positives)

---

## Hard Negative Selection Criteria

### Visual Similarity Threshold

**CLIP embedding similarity**:
- Compute CLIP embeddings for positive and negative concepts
- Hard negative similarity > 0.7 (highly similar)
- Easy negative similarity < 0.5 (visually distinct)

**Example**:
```
Positive: "yellow school bus"
Negative candidates:
- "yellow taxi": similarity = 0.82 → HARD negative ✓
- "red bus": similarity = 0.74 → HARD negative ✓
- "airplane": similarity = 0.31 → easy negative ✗ (exclude)
```

### Attribute Overlap

**At least ONE shared attribute**:

| Positive | Hard Negative | Shared Attribute |
|----------|---------------|------------------|
| yellow school bus | yellow taxi | Color (yellow) |
| yellow school bus | red bus | Type (bus) |
| striped cat | spotted cat | Animal (cat) |
| large dog | small dog | Animal (dog) |

**Rule**: Hard negative must share ≥1 attribute to be confusing.

### Scene Plausibility

**Good hard negative**: Could plausibly appear in same scene

**Example**: Image of "yellow school bus" on street
- ✓ "yellow taxi" (often on streets)
- ✓ "red bus" (buses share routes)
- ✗ "yellow submarine" (implausible scene, despite color match)

**Llama filters implausible negatives** during proposal stage.

---

## Hard Negative Categories

### 1. Color Variants (25% of hard negatives)

**Same object, different color**:
- "red apple" vs "green apple"
- "blue car" vs "red car"
- "white cat" vs "black cat"

**Challenge**: Model must attend to color, not just shape.

### 2. Size/Scale Variants (15%)

**Same object, different size**:
- "large dog" vs "small dog"
- "tall building" vs "short building"
- "adult person" vs "child"

**Challenge**: Model must reason about scale and proportions.

### 3. Pattern/Texture Variants (10%)

**Same object, different texture**:
- "striped shirt" vs "solid shirt"
- "polka dot dress" vs "floral dress"
- "wooden chair" vs "metal chair"

**Challenge**: Model must discriminate fine-grained textures.

### 4. Type/Category Variants (30%)

**Related categories**:
- "sedan" vs "SUV" vs "truck" (all vehicles)
- "cat" vs "dog" (both pets)
- "apple" vs "orange" (both fruit)

**Challenge**: Model must distinguish within-category instances.

### 5. Contextual Variants (20%)

**Same object in different context**:
- "person wearing hat" vs "person not wearing hat"
- "car on road" vs "parked car"
- "dog sitting" vs "dog running"

**Challenge**: Model must reason about object states and actions.

---

## Hard Negative Validation

### Human Verification (Stage 2)

**Annotator checks**:
1. Is negative truly absent from image? (✓ or ✗)
2. Is negative visually similar to positive? (similarity score 1-5)
3. Is negative plausible in this scene? (yes/no)

**Rejection criteria**:
- Negative is actually present (false negative)
- Negative is not visually similar (easy negative)
- Negative is implausible (unrealistic)

**Rejection rate**: ~15% of Llama proposals rejected

### AI Verification (Stage 3)

**Fine-tuned MLLM**:
- Trained on human-verified hard negatives (Stage 2)
- Predicts: "Is this a valid hard negative?" (0-1 score)

**Verification accuracy**: 0.89 correlation with human judgment

**Threshold**: Score > 0.8 = accept, score < 0.8 = reject (send to human)

---

## Impact on Model Behavior

### Without Hard Negatives

**Model behavior**:
- Overly broad predictions ("any yellow vehicle" → yellow school bus)
- High false positive rate (50% precision on fine-grained concepts)
- Poor calibration (confidence scores unreliable)

**Example failure**:
- Prompt: "yellow school bus"
- Image: Yellow taxi
- Prediction: "yellow school bus" (0.65 confidence) ❌

### With Hard Negatives (30/image)

**Model behavior**:
- Fine-grained predictions ("yellow school bus" only when truly present)
- Low false positive rate (95% precision on fine-grained concepts)
- Good calibration (confidence > 0.5 reliable)

**Example success**:
- Prompt: "yellow school bus"
- Image: Yellow taxi
- Presence token: 0.12 confidence → NO prediction ✓

---

## Hard Negative Distribution in SA-Co

### Per-Image Statistics

**Average SA-Co image**:
- Positive concepts: 8.5 (concepts present)
- Hard negatives: 30 (concepts similar but absent)
- Easy negatives: Not stored (too many possibilities)

**Ratio**: 3.5 hard negatives per positive concept

### Dataset-Wide Statistics

**SA-Co/HQ** (5.2M images):
- Total hard negatives: 156M (30 × 5.2M)
- Unique hard negative concepts: 8M
- Overlap with positives: 4M (50% of hard negatives also appear as positives in other images)

**Key insight**: 50% overlap ensures model sees concepts in both positive and negative contexts.

---

## Hard Negative Mining for Long-Tail Concepts

### The Long-Tail Problem

**Challenge**: Rare concepts (<10 training examples)
- Model overfits to few examples
- False positives on visually similar common concepts

**Example**:
- Rare concept: "snow leopard" (5 training images)
- Common concept: "spotted cat" (5,000 training images)
- Model mis-predicts "spotted cat" as "snow leopard" (false positive)

### Solution: Targeted Hard Negatives

**For each rare concept, mine hard negatives from common concepts**:
- "snow leopard" → hard negatives: "spotted cat", "white cat", "leopard"
- Ratio: 10 hard negatives per 1 positive (for rare concepts)

**Result**: Model learns fine-grained distinctions even with few examples.

---

## ARR-COC Connection (10%)

### Opponent Processing (Vervaeke)

**Hard negatives = Opponent framings**

**Example opponent pair**:
- "Yellow school bus" (framing A)
- "Yellow taxi" (framing B)

**Shared features**: Yellow, vehicle, similar size
**Distinguishing features**: Bus type vs taxi type, school use vs transport use

**Relevance realization**: Model must BALANCE salience of shared vs distinguishing features.

### Recursive Opponent Processing

**30 hard negatives per image** = 30 opponent framings!

**Example**: "Yellow school bus" with 30 hard negatives
- Level 1 opponents: "yellow taxi" (same color)
- Level 2 opponents: "red bus" (same type)
- Level 3 opponents: "school van" (same use)
- ...
- Level 30: "yellow sports car" (distant but visually similar)

**Vervaeke insight**: Recursively resolving opponents builds sophisticated relevance realization.

**SAM 3**: 30-level opponent hierarchy → superior concept discrimination.

### Participatory Knowing Through Negative Examples

**Traditional learning**: Positive examples only (what IS the concept)

**SAM 3 learning**: Positive + Hard negatives (what IS and what IS NOT)

**Participatory aspect**:
- Model PARTICIPATES in defining concept boundaries
- Hard negatives co-create understanding through contrast
- Negative space (what is NOT) as important as positive space (what IS)

**Insight**: Knowing "yellow school bus" = knowing what is NOT (yellow taxi, red bus, ...). Hard negatives enable participatory boundary definition.

---

## Summary

**Hard Negative Mining**:

✅ **30 hard negatives/image**: +54.5% IL-MCC improvement
✅ **3 strategies**: Ontology (Wikidata), Llama proposal, model-based mining
✅ **5 categories**: Color, size, pattern, type, context variants
✅ **Selection criteria**: Visual similarity > 0.7, attribute overlap ≥1, scene plausibility
✅ **Validation**: Human (Stage 2) + AI (Stage 3) verification
✅ **Impact**: Fine-grained concept discrimination, low false positives, good calibration

**Key Innovation**: Hard negatives as opponent framings enable sophisticated relevance realization through recursive balancing of shared vs distinguishing features.

---

**Status**: PART 27/42 complete (64.3%)
**Next**: PART 28 - Continuing expansion...
