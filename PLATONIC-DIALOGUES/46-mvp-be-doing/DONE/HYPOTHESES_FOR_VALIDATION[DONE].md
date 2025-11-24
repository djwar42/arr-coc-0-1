# ARR-COC MVP Hypotheses for Validation

**Extracted from Platonic Dialogues Parts 37-45**

## 🎯 THE CORE HYPOTHESIS (Part 41)

**Does query-aware visual token allocation based on relevance realization actually work?**

This is THE question. Everything else is secondary.

---

## Primary Hypotheses (MVP Testable)

### H1: Query-Aware Beats Uniform (The Fundamental Test)

**Claim**: ARR-COC's query-aware relevance realization will allocate tokens more effectively than uniform allocation.

**What to test**:
- ARR-COC (200 patches, relevance-weighted) vs Baseline (200 patches, uniform)
- Same total tokens, different allocation strategy

**Expected behavior**:
- Query: "What color is the car?" → ARR-COC focuses on car region
- Query: "How many people?" → ARR-COC distributes across people
- Baseline: Always uniform distribution regardless of query

**Validation interface needs**:
- Side-by-side patch visualization
- Heatmap showing token allocation
- Query input to test different questions
- Qualitative: "Which makes more sense?"

---

### H2: Three Ways of Knowing Provide Complementary Information

**Claim**: Combining propositional (info), perspectival (salience), and participatory (query-coupling) gives better relevance than any single way.

**What to test**:
- ARR-COC full (all 3 ways) vs ablations:
  - Information only
  - Perspectival only
  - Participatory only
  - Info + Persp (no query)

**Expected behavior**:
- Info-only: Prioritizes high-entropy regions (edges, text)
- Persp-only: Prioritizes salient regions (edges, contrasts)
- Partic-only: Prioritizes query-relevant regions (may miss context)
- **Full system**: Balances all three → best results

**Validation interface needs**:
- Toggle between different "ways of knowing"
- Show individual scores (info, persp, partic) before balancing
- Show final balanced score
- Heatmaps for each way + combined

---

### H3: Adaptive Balancing Outperforms Fixed Weights (Part 37, 39)

**Claim**: Learning to balance the three ways (via MLP) beats fixed weighting.

**What to test**:
- ARR-COC adaptive (learned balancer) vs fixed weights (e.g., 0.33/0.33/0.33)

**Expected behavior**:
- Spatial variation: Different weights for different patches
- Query variation: Different weights for different queries
- Fixed weights: Same everywhere, suboptimal

**Validation interface needs**:
- Show learned weights per patch (compress/exploit/focus tensions)
- Color-code patches by dominant "way of knowing"
- Compare fixed vs adaptive side-by-side

---

### H4: Top-K Selection Captures Relevant Regions

**Claim**: Selecting top-200 most relevant patches (out of 1024) preserves query-critical information while reducing computation.

**What to test**:
- K=200 (MVP) vs K=512 (more tokens) vs K=100 (fewer tokens)
- Does K=200 find the right patches?

**Expected behavior**:
- Homunculus visualization shows semantic clustering
- Selected patches cover query-relevant objects/regions
- Not just edges or random distribution

**Validation interface needs**:
- **THE HOMUNCULUS**: Visual overlay showing selected vs rejected patches
- Interactive K slider (100, 200, 400, 800)
- Visual feedback: "Did it pick the right stuff?"

---

## Secondary Hypotheses (Post-MVP)

### H5: 13-Channel Texture Outperforms RGB-Only

**Claim**: Texture array (RGB + LAB + edges + spatial) provides richer signal than raw RGB.

**Test**: Full texture (13 ch) vs RGB-only (3 ch)

**Validation**: Compare relevance maps, check if edges help

---

### H6: LOD Compression Would Improve Efficiency

**Claim**: Variable token budgets (64-400) per patch based on relevance would beat fixed 200.

**Test**: Post-MVP (not in 0.1)

---

## 🎨 MVP Validation Interface Requirements

Based on hypotheses H1-H4, our Gradio app_local.py needs:

### Must-Have Visualizations:

1. **The Homunculus** (H4)
   - Original image with overlay
   - Green boxes: Selected patches (top-200)
   - Red/dim: Rejected patches
   - Shows WHAT the system chose

2. **Relevance Heatmap** (H1, H2)
   - Color-coded by relevance score
   - Shows WHY patches were chosen
   - Graduated colors (blue=low, red=high)

3. **Three Ways Breakdown** (H2)
   - 3 separate heatmaps: Info, Persp, Partic
   - Combined/balanced heatmap
   - Allows ablation comparison

4. **Adaptive Weights Visualization** (H3)
   - Show learned tension values per patch
   - Compress/Exploit/Focus sliders
   - Spatial variation visible

5. **Query Input**
   - Text box for different queries
   - Pre-set examples:
     - "Where is the cat?"
     - "How many people?"
     - "What color is the car?"
     - "Read the sign"
   - Real-time re-computation

6. **Side-by-Side Comparison** (H1)
   - ARR-COC vs Baseline (uniform)
   - Same image, same query, different allocation
   - Qualitative validation

### Interaction Flow:

```
User uploads image
  ↓
User enters query OR selects example
  ↓
System computes:
  - 13-channel texture
  - Info/Persp/Partic scores
  - Adaptive balancing
  - Top-K selection
  ↓
Display:
  ├─ Original image
  ├─ Homunculus (selected patches)
  ├─ Relevance heatmap
  ├─ Three ways breakdown
  └─ Learned weights
  ↓
User can:
  - Try different queries
  - Toggle visualizations
  - Compare to baseline
  - Export results
```

---

## Validation Metrics (Quantitative)

For each hypothesis, we need:

**Automatic metrics**:
- Entropy of patch distribution (uniform vs concentrated)
- Coverage of query-relevant objects (via bounding boxes if available)
- Computational cost (FLOPs, memory)

**Manual metrics** (qualitative):
- Likert scale: "Does the allocation make sense?" (1-5)
- A/B preference: "Which allocation is better?"
- Binary: "Did it find the object?"

---

## Test Dataset for MVP

**100 image-query pairs** covering:
- Object localization: "Where is X?"
- Counting: "How many X?"
- Attribute: "What color is X?"
- Spatial: "What is to the left of X?"
- Text: "Read the sign"

Diverse images:
- Simple (1-2 objects)
- Complex (cluttered scenes)
- Text-heavy (documents, signs)
- Natural scenes
- Indoor/outdoor

---

## Decision Criteria

After validation, we can answer:

**✅ Hypothesis CONFIRMED if**:
- Qualitative: Users prefer ARR-COC allocation >70% of time
- Quantitative: ARR-COC achieves better coverage with same K
- Visual: Homunculus shows semantic selectivity (not random/edges)

**❌ Hypothesis REJECTED if**:
- ARR-COC performs same as uniform baseline
- Homunculus shows no query-awareness
- Three ways don't add complementary information

**🤔 Hypothesis UNCLEAR if**:
- Results are mixed or marginal
- Need more data or refinement
- Back to dialogue mode!

---

## Summary: What We're Really Testing

The MVP validation interface is NOT about perfect VQA accuracy.

It's about answering ONE question:

**"Does relevance realization produce semantically meaningful token allocation that responds to queries?"**

If YES: We have proof-of-concept. Build infrastructure, train on VQAv2, optimize.

If NO: Back to the drawing board. The Vervaekean framework doesn't translate to vision-language, or our implementation is wrong.

---

**Next Step**: Build these visualizations into `app_local.py` and `app.py`.

Let the validation begin! 🎯
