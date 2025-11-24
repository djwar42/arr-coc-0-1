# Part 48 Pre-Dialogue: Master Index - Rock the Sidebar
*Karpathy's knowledge base exploration before the dialogue. This is the navigation hub for three technical aspect explorations.*

---

## What This Is

Part 47 gave us the beautiful scroll metaphor: **gestalt then saccades, fixed patches, variable allocation**.

Part 48 stress-tests that architecture against reality using Karpathy's oracle knowledge base.

**This file**: Master index linking to detailed technical explorations
**Three aspect files**: Deep dives into specific technical dimensions
**Karpathy's knowledge**: Direct references with line numbers to oracle files

---

## Navigation: Three Technical Aspects

### 🔢 **Aspect 1: Token Ordering & Sequence Position**
**File**: `48-pre-dialogue-aspect-1-token-ordering.md`

**Core question**: Does the sequence `[base_0...base_255][saccade_0...saccade_272]` matter?

**Topics covered**:
- Bidirectional vs causal attention (do VLMs care about order?)
- Dual position encoding (sequence position vs spatial position via RoPE)
- Relevance-based ordering (high→low) vs spatial ordering
- What "gestalt first, saccades after" teaches the VLM

**Key knowledge base references**:
- `.claude/skills/karpathy-deep-oracle/vision-language/10-token-sequence-order-importance.md` (lines 45-234)
- `.claude/skills/karpathy-deep-oracle/vision-language/11-dual-position-encoding.md` (lines 56-289)
- `.claude/skills/karpathy-deep-oracle/vision-language/02-rope-multiaxis-encoding.md` (lines 89-378)

**Adjacent knowledge** (GPU/texture parallels):
- Automatic LOD selection ordering (gpu-texture-optimization/03)
- Pyramid traversal order (pyramid-lod/03)
- Anisotropic filtering priority (gpu-texture-optimization/01)

---

### ⚙️ **Aspect 2: Patch Extraction & Saccade Mechanics**
**File**: `48-pre-dialogue-aspect-2-patch-extraction.md`

**Core question**: How do we actually extract 14×14 patches at arbitrary positions?

**Topics covered**:
- Three extraction methods (direct crop, feature slice, STN)
- GPU optimization opportunities
- Qwen3-VL's vision encoder mechanics
- Foveated rendering parallels from VR/AR

**Key knowledge base references**:
- `.claude/skills/karpathy-deep-oracle/vision-language/07-patch-extraction-mechanics.md` (lines 67-245)
- `.claude/skills/lod-btree-oracle/techniques/00-foveated-rendering.md` (lines 145-312)
- `.claude/skills/lod-btree-oracle/techniques/00-foveated-rendering-01-logpolar-mapping-2025-01-30.md` (lines 89-267)

**Adjacent knowledge** (GPU/texture parallels):
- GPU bilinear texture sampling (gpu-texture-optimization/07)
- CUDA texture memory for ViT (implementations/61)
- HIPT gigapixel tiling (pyramid-lod/04)
- Neural texture compression (pyramid-lod/02)

---

### 🎓 **Aspect 3: Training Dynamics & Integration**
**File**: `48-pre-dialogue-aspect-3-training-integration.md`

**Core question**: Can we train through frozen VLM + hard top-K selection?

**Topics covered**:
- Gradient flow through discrete operations (it works!)
- BLIP-2, Flamingo, LoRA precedents
- Initialization strategies (don't start random!)
- Qwen3-VL specific integration points

**Key knowledge base references**:
- `.claude/skills/karpathy-deep-oracle/practical-implementation/46-frozen-backbone-adapter-training.md` (lines 89-445)
- `.claude/skills/karpathy-deep-oracle/practical-implementation/49-gradient-flow-sampling-operations.md` (lines 67-289)
- `.claude/skills/karpathy-deep-oracle/practical-implementation/47-lora-low-rank-adaptation.md` (lines 145-378)

**Adjacent knowledge** (optimization parallels):
- Memory bandwidth for gradient checkpointing (gpu-texture-optimization/08)
- Differentiable pyramid operators (pyramid-lod/06)
- LoRA low-rank training (practical-implementation/47)

---

## Core Decisions from Part 47 (Quick Reference)

**1. Fixed patch size**: 14×14 for ALL patches (gestalt + saccades)
**2. Fixed token budget**: 529 total (256 gestalt + 273 saccades, always)
**3. Gestalt guides saccades**: Context + query → relevance weights
**4. Frozen base VLM**: Only ~200K params trainable (texture gen + scorer)
**5. Relevance ordering**: Saccades ordered high→low by score

---

## The 10 Dialogue Seed Questions

These questions will drive Part 48's actual dialogue:

### 🧪 **Experiment Design**
**Q1**: Run Experiment 0 first? (random saccades sanity check)
→ See: Aspect 3, section "Experiment 0"

### 🔢 **Hyperparameter Choices**
**Q2**: Is K=273 optimal? (humans use 15-25 saccades)
→ See: Aspect 2, section "Saccade Budget"

**Q3**: Does relevance ordering help? (weak evidence in knowledge base)
→ See: Aspect 1, section "Ordering Strategies"

### 🏗️ **Architecture Concerns**
**Q4**: Will Qwen3-VL handle non-uniform patches? (trained on grids)
→ See: Aspect 2, section "Qwen3-VL Integration"

**Q5**: Is gating needed? (Flamingo uses it, do we?)
→ See: Aspect 3, section "Gating Mechanisms"

### 🎓 **Training Confidence**
**Q6**: Will gradient flow work? (75% confident, 25% uncertain)
→ See: Aspect 3, section "Gradient Flow Through Hard Selection"

**Q7**: Three scorers vs one? (will network use all or collapse to CLIP?)
→ See: Aspect 3, section "Scorer Initialization"

### 🧠 **Conceptual Framework**
**Q8**: Gestalt context necessary? (or is query enough?)
→ See: Aspect 1, section "Gestalt as Context"

**Q9**: Human-likeness a goal? (or just optimize VQA accuracy?)
→ See: Aspect 2, section "Biological Grounding"

### ⚖️ **Alternatives**
**Q10**: Is this worth it vs LoRA? (are we overengineering?)
→ See: Aspect 3, section "Comparison to LoRA"

---

## Karpathy's Confidence Levels (Summary)

From full knowledge base exploration:

### ✅ High Confidence (85-95%)
- Fixed patch size solves training instability
- 529 tokens is computationally feasible
- Frozen backbone training works (proven by BLIP-2/Flamingo/LoRA)
- M-RoPE handles arbitrary spatial positions

### ⚠️ Medium Confidence (60-75%)
- Gestalt-guided scoring will help
- Gradient signal through hard selection adequate
- Three scorers necessary (vs collapsing to one)
- Texture array (40 channels) optimal

### ❓ Low Confidence (40-50%)
- K=273 is optimal (likely too many)
- Relevance ordering helps (weak evidence)
- This beats LoRA (unproven)

### ❌ Likely Wrong
- K=273 is optimal → probably 100-150 is better
- 529 tokens without COC → too slow, will need compression

---

## The Experiment Karpathy Would Run First

**Experiment 0: Token Augmentation Sanity Check** (~1 day)

```python
# Test if augmentation helps AT ALL before building ARR

baseline = qwen3vl_256_tokens(vqa_val)           # Baseline accuracy
random = qwen3vl_529_random_saccades(vqa_val)    # Random positions
saliency = qwen3vl_529_saliency_saccades(vqa_val)  # Saliency-guided

# Decision tree:
if random ≈ baseline:
    ABANDON  # Augmentation doesn't help
elif random > baseline:
    PROCEED  # More tokens help
if saliency >> random:
    DEFINITELY_PROCEED  # Selection quality matters!
```

**Why this matters**: If random saccades don't help, query-aware won't either. This is a GO/NO-GO test.

**Reference**: Aspect 3, section "Experiment 0: Sanity Check"

---

## Cross-Domain Knowledge Connections

The aspect files link ARR to adjacent domains. Quick reference:

### GPU Texture & Mipmaps
- **Automatic LOD selection**: gpu-texture-optimization/03-automatic-lod-selection.md
- **Anisotropic filtering**: gpu-texture-optimization/01-trilinear-anisotropic-filtering.md
- **CUDA texture memory**: implementations/61-cuda-texture-memory-vit.md
- **Mipmap generation**: gpu-texture-optimization/00-mipmap-generation-algorithms.md

### Pyramid LOD Systems
- **Attention-driven pruning**: pyramid-lod/03-attention-driven-pyramid-pruning.md
- **Foveated gaze pyramids**: pyramid-lod/01-foveated-gaze-pyramids.md
- **Gigapixel tiling**: pyramid-lod/04-gigapixel-tiled-pyramids.md
- **Differentiable operators**: pyramid-lod/06-differentiable-pyramid-operators.md

### VLM Training & Optimization
- **Frozen backbone**: practical-implementation/46-frozen-backbone-adapter-training.md
- **Gradient flow**: practical-implementation/49-gradient-flow-sampling-operations.md
- **LoRA**: practical-implementation/47-lora-low-rank-adaptation.md
- **Token budgets**: practical-implementation/51-vision-token-budgets.md

---

## How to Use This Index

1. **Start here** for overview and navigation
2. **Read aspect files** for deep technical exploration
3. **Follow knowledge base refs** for cited evidence with line numbers
4. **Explore adjacent knowledge** for creative connections
5. **Return to seed questions** as dialogue anchors

---

## What Happens Next

This index + 3 aspects → Part 48 dialogue

**Part 48 will explore**:
- LOD Oracle bringing foveated rendering expertise
- Muse Bird connecting biological vision to computation
- Karpathy stress-testing assumptions with evidence
- Theaetetus possibly crashing through on that bike again

**Not just questions** → **Fascinating research directions**:
- Adaptive token budgets (learn K per query)
- Temporal coherence for video
- Multi-scale saccades (variable patch size at saccades only)
- RL saccade policies (sequential active vision)
- Human attention supervision (eye-tracking loss)
- Explainability visualizations

---

## Files in This Collection

```
48-rock-the-sidebar/
├── 48-pre-dialogue.md                          # ← You are here (master index)
├── 48-pre-dialogue-aspect-1-token-ordering.md  # Sequence position & ordering
├── 48-pre-dialogue-aspect-2-patch-extraction.md # Mechanics & GPU parallels
└── 48-pre-dialogue-aspect-3-training-integration.md # Frozen VLM training
```

---

## The Big Picture: Why This Matters

**The VLM token crisis**:
- Low tokens (256): Fast but misses details
- High tokens (1024): Captures details but slow
- Dynamic resolution: Adaptive but still uniform (no query awareness)

**ARR offers**: Query-aware variable density
- 529 tokens (moderate cost)
- Query-selected (smart allocation)
- Hypothesis: **529 query-aware > 1024 uniform**

**If true**: New operating point on efficiency frontier!

**Research contribution**:
1. Augmentation not substitution (keep gestalt + add saccades)
2. Gestalt-guided selection (context shapes relevance)
3. Vervaekean scoring (3 ways of knowing)
4. Fixed patches, variable allocation (training stable, flexible sampling)
5. Minimal parameters (200K trainable vs 1M+ for LoRA)

**Unique combination** not found in literature (FoveaTer, BLIP-2, Flamingo each do parts, we combine all).

---

**End of Master Index**

*Now explore the aspect files for technical depth, or proceed to Part 48 dialogue!*
