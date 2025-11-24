# SAM 3: Three-Stage Data Engine

**PART 25/42 - Data Engine Stages (Assisted-Manual, Semi-Automatic, Fully Automatic)**

**Date**: 2025-11-20
**Source**: SAM 3 paper, SA-1B data engine methodology

---

## Data Engine Overview

The **SA-Co data engine** uses a **human-AI loop** with 3 progressive stages to achieve:
- **2× annotation throughput** vs traditional methods
- **Near-human quality** at massive scale
- **4M unique concepts** with exhaustive labeling

### The 3 Stages

```
Stage 1: Assisted-Manual → Highest quality, slowest
    ↓
Stage 2: Semi-Automatic → 2× faster, still high quality
    ↓
Stage 3: Fully Automatic → 10× faster, moderate quality (verified by AI)
```

---

## Stage 1: Assisted-Manual (Highest Quality)

### Workflow

**Input**: Raw image

**Step 1: AI Proposal**
- Vision model detects objects
- Proposes bounding boxes for all visible objects
- No concept labels yet (just spatial regions)

**Step 2: Human Annotation**
- Annotator views AI proposals
- For each proposal:
  - Assign concept label (noun phrase)
  - Correct bounding box if needed
  - Refine mask boundaries (pixel-perfect)
- Add missing instances (AI didn't detect)
- Add hard negatives (visually similar but different concepts)

**Step 3: Exhaustivity Check**
- Human verifier reviews completed annotation
- Ensures ALL instances of ALL concepts are labeled
- Common miss: Small objects, occluded objects, rare concepts

**Output**: Fully annotated image with exhaustive concept labels + masks

### Quality Metrics

- **Precision**: 98% (very few false positives)
- **Recall**: 95% (most instances captured)
- **Boundary IoU**: 92% (precise masks)
- **Throughput**: ~100 images/hour per annotator

### Use Cases

- **SA-Co/Gold subset**: Triple annotation (3 independent humans)
- **SA-Co/Silver subset**: Single annotation (still exhaustive)
- **Training baseline**: All Stage 1 data used to train AI for Stage 2/3

---

## Stage 2: Semi-Automatic (2× Faster)

### Workflow

**Input**: Raw image

**Step 1: AI Detection + Segmentation**
- Trained model from Stage 1 detects objects
- Generates masks automatically (no human pixel work)
- Proposes concept labels (Llama-based text model)

**Step 2: Human Verification**
- Annotator reviews AI outputs
- **Accept/Reject/Edit** for each instance:
  - Accept: AI correct (1-click approval)
  - Reject: AI wrong (delete false positive)
  - Edit: Minor corrections (adjust concept label or mask)
- Add missing instances (if AI missed any)
- **Add hard negatives** (critical for quality!)

**Step 3: Exhaustivity Check (Optional)**
- Random sample verified by second human
- Ensures annotator didn't skip instances

**Output**: High-quality annotations with 50% less human time

### Key Innovation: Hard Negative Mining

**Llama proposes hard negatives**:
- Analyzes image content
- Suggests visually similar concepts that are NOT present
- Example: Image has "yellow school bus"
  - Llama suggests: "yellow taxi" (hard negative ✓)
  - Llama suggests: "red bus" (hard negative ✓)
  - Llama suggests: "airplane" (easy negative ✗, not included)

**Human validates hard negatives**:
- Confirms proposed hard negative is NOT in image
- Adds to annotation as negative example

### Quality vs Speed Trade-off

| Metric | Stage 1 | Stage 2 | Change |
|--------|---------|---------|--------|
| **Throughput** | 100 img/hr | **200 img/hr** | **2× faster** |
| **Precision** | 98% | 95% | -3% |
| **Recall** | 95% | 92% | -3% |
| **Hard negatives** | 10/image | **30/image** | **3× more** |

**Key takeaway**: Stage 2 sacrifices ~3% quality for 2× speed, but adds MORE hard negatives (critical for open-vocabulary).

---

## Stage 3: Fully Automatic (10× Faster, AI-Verified)

### Workflow

**Input**: Raw image

**Step 1: AI Annotator (Llama-based)**
- Multimodal LLM analyzes image
- Proposes diverse noun phrases (concepts)
- Includes hard negatives automatically
- Generates masks for each concept

**Step 2: AI Verifier (Fine-tuned MLLM)**
- **NO human in loop** (fully automated!)
- AI checks:
  - Mask quality (IoU > threshold?)
  - Exhaustivity (all instances labeled?)
  - Hard negative validity (truly absent?)
- Assigns quality score (0-1)

**Step 3: Active Mining**
- Low-scoring annotations flagged
- Sent to Stage 2 (human verification)
- Focuses human effort on failure cases
- Improves AI annotator via feedback loop

**Output**: Massive-scale synthetic dataset (SA-Co/SYN: 38M concepts, 1.4B masks)

### AI Annotator Architecture

**Concept Proposal** (Llama 3.1-based):
```
Input: Image
Llama: "I see: yellow school bus, person wearing blue shirt,
        red traffic light, gray building, green tree"
Llama: "Hard negatives: yellow taxi, yellow car, person
        wearing red shirt"
```

**Mask Generation** (Vision model):
- Uses proposed concepts as prompts
- Generates segmentation masks
- Matches masks to concepts

### AI Verifier Architecture

**Fine-tuned Multimodal LLM**:
- Trained on Stage 1 + 2 human annotations
- Learns what "good" vs "bad" annotations look like
- Achieves **near-human accuracy** on verification

**Verification checks**:
1. **Mask quality**: Is boundary precise? (IoU > 0.85?)
2. **Exhaustivity**: Are all instances labeled? (Sample-based check)
3. **Hard negatives**: Is negative truly absent? (Classification check)

**Output**: Quality score (0-1)
- Score > 0.9: Accept (high confidence)
- Score 0.7-0.9: Review (medium confidence, active mining)
- Score < 0.7: Reject (send to Stage 2 for human fix)

### Quality Distribution

**SA-Co/SYN dataset** (Stage 3 output):
- 70% high quality (score > 0.9)
- 25% medium quality (0.7-0.9, improved via active mining)
- 5% low quality (rejected, re-annotated in Stage 2)

**Impact on training**:
- Adds massive diversity (38M concepts vs 4M in Stage 1+2)
- Lower quality offset by scale
- Combined with Stage 1+2 data achieves best results

---

## Active Mining (Continuous Improvement)

### Feedback Loop

```
Stage 3: AI annotates 10K images
    ↓
AI verifier scores all annotations
    ↓
500 images flagged as low quality (score < 0.7)
    ↓
Sent to Stage 2: Human reviews + corrects
    ↓
Corrections used to retrain AI annotator
    ↓
Repeat: Next 10K images have better quality
```

### Active Mining Strategy

**Focus effort on failures**:
- Rare concepts (AI struggles with "snow leopard")
- Fine-grained distinctions ("striped cat" vs "spotted cat")
- Occlusions (partially visible objects)
- Crowded scenes (many instances)

**Result**: AI annotator improves over time, human effort focused on hardest cases.

---

## Data Engine Efficiency

### Throughput Comparison

| Stage | Images/Hour | Quality | Cost |
|-------|-------------|---------|------|
| **Stage 1** | 100 | 98% | High |
| **Stage 2** | 200 | 95% | Medium |
| **Stage 3** | **1000** | 85% | **Low** |

**Overall throughput**: 2× improvement over traditional pure-human annotation (Stage 1 only).

### Cost Breakdown

**Traditional (Stage 1 only for 5.2M images)**:
- Time: 52,000 hours
- Cost: $5.2M (at $100/hr)

**SA-Co (Stages 1+2+3)**:
- Stage 1: 500K images → 5,000 hours → $500K
- Stage 2: 1.5M images → 7,500 hours → $750K
- Stage 3: 38M concepts (synthetic) → AI-only → $50K (compute)
- **Total**: $1.3M (**75% cost reduction!**)

---

## Comparison to SA-1B (SAM 1 Data Engine)

| Aspect | SA-1B (SAM 1) | SA-Co (SAM 3) |
|--------|---------------|---------------|
| **Stages** | 3 (similar) | 3 (enhanced) |
| **Stage 1** | Assisted-manual | Assisted-manual + **hard negatives** |
| **Stage 2** | Semi-automatic | Semi-automatic + **Llama concepts** |
| **Stage 3** | Fully automatic | **AI verifier** (new!) |
| **Concepts** | N/A | **4M unique** (exhaustive) |
| **Hard negatives** | No | **30/image** (critical) |
| **Quality check** | Human sample | **AI verifier** (scalable) |

**Key differences**:
1. **Concept labels**: SA-1B had masks only, SA-Co adds noun phrases
2. **Hard negatives**: SA-Co emphasizes fine-grained distinctions
3. **AI verification**: SA-Co scales quality control with AI

---

## ARR-COC Connection (10%)

### Reciprocal Narrowing (Human-AI Loop)

**Vervaeke's concept**: Coupled systems co-constrain each other

**Data engine mapping**:
- **Human** constrains **AI**: Corrects failures, provides novel framings
- **AI** constrains **Human**: Proposes candidates, reduces search space

**Stage 2 example**:
1. AI proposes 50 instances (narrows human attention)
2. Human rejects 5 false positives (constrains AI understanding)
3. Human adds 3 missing instances (expands AI frame)
4. Next iteration: AI learns from corrections

**Insight**: Neither human nor AI sufficient alone; coupling creates superior relevance realization.

### Participatory Knowing via Active Mining

**Traditional annotation**: Human applies pre-defined rules (propositional)

**Active mining**: Human CREATES understanding through interaction (participatory)

**Example cycle**:
- AI struggles with "snow leopard" (rare concept)
- Human corrects 20 "snow leopard" failures
- AI learns implicit features (spots, tail shape, mountain context)
- Human participates in AI's "knowing" of snow leopards

### Opponent Processing in Hard Negatives

**30 hard negatives/image** = 30 opponent framings

**Example**:
- "Yellow school bus" vs "yellow taxi" = color agreement, type disagreement
- "Striped cat" vs "spotted cat" = animal agreement, pattern disagreement

**Vervaeke insight**: Relevance emerges from resolving opposites (balance, salience)

**Data engine**: Forces model to navigate opponent pairs → builds relevance realization capability.

---

## Summary

**Three-Stage Data Engine**:

✅ **Stage 1 (Assisted-Manual)**: Highest quality baseline (98% precision, 100 img/hr)
✅ **Stage 2 (Semi-Automatic)**: 2× faster with hard negatives (95% precision, 200 img/hr)
✅ **Stage 3 (Fully Automatic)**: 10× faster, AI-verified (85% quality, 1000 img/hr)
✅ **Active mining**: Focus human effort on AI failures (continuous improvement)
✅ **2× overall throughput**: 75% cost reduction vs pure-human annotation
✅ **4M concepts**: Exhaustive labeling with 30 hard negatives/image

**Key Innovation**: Human-AI coupling through reciprocal narrowing creates superior data quality and scale, mirroring Vervaeke's participatory knowing.

---

**Status**: PART 25/42 complete (59.5%)
**Next**: PART 26 - Stage-by-Stage Quality Evolution
