---
summary: whereby Karpathy stress-tests the biologically-inspired foveation design by exploring uncomfortable edge cases (multi-object queries like "Is the cat on the table?" requiring fixation on both objects, small object detection where periphery loses critical detail, documents with text everywhere challenging single-fixation assumptions), examining the tension between biological faithfulness (humans use 3-4 saccades per second) and engineering constraints (273 tokens × 3 fixations = 819 tokens losing 10× speedup claim), questioning training strategies (freeze vision encoder vs learnable M₀/e₀ parameters), and ultimately leaving questions open without premature conclusions about adaptive fixations or curriculum learning
---

# Part 24: Questioning the Blueprint

*Wherein the oracles explore edge cases, question assumptions, and discover the tensions between biological faithfulness and engineering pragmatism—without reaching conclusions*

---

## Opening: The Uncomfortable Questions

*Scene: The Dirac Sea, late afternoon. The V1 model floats between the oracles, perfectly specified—273 tokens, log-polar sampling, cortical magnification. But something feels... incomplete.*

**MUSE BIRD:** *[Perched on a floating question mark]*
🐦 *SPECIFICATIONS COMPLETE! NEUROSCIENCE → CUDA! READY TO BUILD!*

**KARPATHY:**
...

**LOD ORACLE:**
You're quiet.

**KARPATHY:**
I keep thinking about edge cases.

**LOD ORACLE:**
Such as?

**KARPATHY:**
Query: "Is the cat on the table?"

Two objects. Cat on the left, table on the right. Our model fixates on... what? The cat? The table? The center?

If we fixate on the cat, we allocate 55 tokens to it. But the table is in the periphery—maybe 20 tokens, mip level 3. Is that enough to see if the cat is ON the table?

**LOD ORACLE:**
...That's a problem.

**KARPATHY:**
Or this: "What time does the clock show?" Clock is 50×50 pixels in a 1024×1024 image. If it's in the periphery, we sample at mip level 4. That's 64×64 resolution. The clock becomes 3×3 pixels. Unreadable.

**LOD ORACLE:**
Also a problem.

**KARPATHY:**
Or documents. Text everywhere. Where do we fixate? If we fixate on one paragraph, we miss the others.

**MUSE BIRD:** *[Drooping slightly]*
🐦 *...problems?*

**KARPATHY:**
And that's just query-level stuff. What about training? Do we freeze the vision encoder? Do we make M₀ and e₀ learnable? Do we use curriculum learning?

And what about the bigger question: How much can we deviate from biology before it stops being "biologically inspired"?

**LOD ORACLE:**
You're right. We have a beautiful specification. But we haven't stress-tested it.

**KARPATHY:**
Let's do that now. No conclusions. Just... exploration.

**MUSE BIRD:** *[Perking up]*
🐦 *EXPLORATION MODE! QUESTION EVERYTHING! BREAK THE MODEL!*

---

## Act I: The Multi-Object Problem

**LOD ORACLE:**
Let's start with the cat and table. Two objects, opposite sides of the image.

**KARPATHY:**
Human vision handles this with SACCADES. Eyes jump 3-4 times per second.

First fixation: Cat
Second fixation: Table
Third fixation: Spatial relationship between them

**LOD ORACLE:**
So should we do multiple fixations?

**KARPATHY:**
Maybe. But that triples our token budget. 273 tokens × 3 fixations = 819 tokens.

Still better than 4096, but we lose the "10× speedup" claim.

**LOD ORACLE:**
What if we do ADAPTIVE fixations? One fixation for simple queries, multiple for complex ones.

**KARPATHY:**
How do we know if a query is complex?

**LOD ORACLE:**
Parse it? "Is the cat on the table?" has two objects → complex.

"What color is the cat?" has one object → simple.

**KARPATHY:**
But parsing is unreliable. And what about:

"Is there a cat on the table?"

One object (cat)? Or two (cat + table)?

**MUSE BIRD:**
🐦 *PARSE THE PARSE! META-QUERY ANALYSIS!*

**LOD ORACLE:**
Or we could use HIERARCHICAL fixation.

**KARPATHY:**
Explain.

**LOD ORACLE:**
First fixation: Coarse, wide angle (like mip level 4). Covers the whole scene.

Second fixation: If needed, zoom in on high-attention region.

**KARPATHY:**
So the first pass is like peripheral vision—survey the scene. Second pass is foveal—focus on details.

**LOD ORACLE:**
Exactly. That's how humans scan unfamiliar scenes.

**KARPATHY:**
But is that still "biologically inspired"? Humans don't consciously decide "first coarse, then fine." They just... saccade.

**LOD ORACLE:**
Good point. Are we modeling the MECHANISM (unconscious saccades) or the STRATEGY (coarse-to-fine)?

**KARPATHY:**
That's a philosophical question.

**MUSE BIRD:**
🐦 *PHILOSOPHY! SHIP OF THESEUS! HOW MUCH BIOLOGY CAN YOU REMOVE BEFORE IT'S NOT BIOLOGY?*

**KARPATHY:**
Exactly. If we add learned scheduling, attention-based routing, dynamic budgets... at what point is it just "inspired by" biology rather than "implementing" biology?

**LOD ORACLE:**
And does it matter?

**KARPATHY:**
I don't know.

---

## Act II: The Text Problem (OCR Hell)

**KARPATHY:**
Let's tackle the document case. Image is a scanned page. Text everywhere.

Query: "What does the third paragraph say?"

Where do we fixate?

**LOD ORACLE:**
We could run OCR first, detect all text regions, then allocate tokens proportionally.

**KARPATHY:**
That's expensive. OCR adds 20-30ms. We're back to 50ms total (vs 67ms baseline). Speedup drops from 10× to 1.3×.

**LOD ORACLE:**
What if we use a FAST text detector? Just bounding boxes, no recognition.

**KARPATHY:**
Still adds 5-10ms. And then we have a new problem: Do we allocate tokens to ALL text regions equally? Or prioritize based on query?

**LOD ORACLE:**
Query-based. "Third paragraph" → find paragraph 3, allocate more tokens there.

**KARPATHY:**
But what if the query is "Summarize the document"? Now we need ALL paragraphs.

**LOD ORACLE:**
Then we're back to uniform sampling. No foveation.

**KARPATHY:**
So for document-heavy VQA (like DocVQA), foveation might not help?

**LOD ORACLE:**
Maybe not. Or maybe we need a DIFFERENT foveation strategy for text.

**MUSE BIRD:**
🐦 *ANISOTROPIC! ELONGATED! TEXT LINES ARE HORIZONTAL! SAMPLE THAT WAY!*

**KARPATHY:**
The Muse is right. Anisotropic filtering.

**LOD ORACLE:**
From the hardware addendum. Sample elongated regions along text lines. One sample covers multiple words.

**KARPATHY:**
Right. So for text, we don't use RADIAL foveation (from fixation point). We use DIRECTIONAL foveation (along text orientation).

**LOD ORACLE:**
Which means our M(e) formula doesn't apply anymore.

**KARPATHY:**
So we need TWO allocation strategies:

1. **Natural images**: Radial foveation, cortical magnification
2. **Documents**: Directional sampling, anisotropic filtering

**LOD ORACLE:**
How do we decide which to use?

**KARPATHY:**
Classify the image? "Is this a document or a natural scene?"

**LOD ORACLE:**
That's another model. More overhead.

**KARPATHY:**
Or we could do HYBRID. Allocate some tokens radially (for layout, structure), some directionally (for text content).

**MUSE BIRD:**
🐦 *RADIAL + DIRECTIONAL! ORTHOGONAL SAMPLING! SPANNING SPACE!*

**LOD ORACLE:**
But then we're not following V1 anymore. V1 does radial foveation, period.

**KARPATHY:**
So we're back to the question: How much can we deviate before it's not "biological"?

**LOD ORACLE:**
I think the answer is: It depends on the goal.

If the goal is SCIENTIFIC (model V1 faithfully) → strict adherence.

If the goal is ENGINEERING (best VLM performance) → deviate as needed.

**KARPATHY:**
And our goal is... both?

**LOD ORACLE:**
That's the tension.

---

## Act III: The Small Object Problem

**LOD ORACLE:**
Let's look at the clock problem. "What time does the clock show?"

Clock is 50×50 pixels in 1024×1024 image. Suppose it's at eccentricity 0.3 (not foveal, not extreme periphery).

```python
M = 1.0 / (0.3 + 0.5) = 1.25
mip_level = -log2(1.25) ≈ -0.32 → rounds to 0 (full resolution)
```

Oh. Actually, it's fine. Mip level 0.

**KARPATHY:**
Wait, let me recalculate. If the clock center is at eccentricity 0.3, but we're sampling 16×16 patches...

Actually, the clock is 50×50 pixels. At mip level 0, that's roughly 3 patches. Should be readable.

**LOD ORACLE:**
Hmm. So maybe small objects aren't a problem if the fixation is good?

**KARPATHY:**
Unless the fixation is BAD. If we fixate on the wrong region, the clock ends up in the far periphery (eccentricity 0.7).

```python
M = 1.0 / (0.7 + 0.5) = 0.83
mip_level = -log2(0.83) ≈ 0.27 → rounds to 0 still
```

Hmm, still mip 0.

**LOD ORACLE:**
Let's try extreme periphery. Eccentricity 0.9.

```python
M = 1.0 / (0.9 + 0.5) = 0.71
mip_level = -log2(0.71) ≈ 0.49 → rounds to 0
```

STILL mip 0?

**KARPATHY:**
Wait, our formula is wrong. Let me recalculate.

M = M₀ / (e + e₀)

For mip level to be 1, we need M < 0.5 (since mip_level = -log2(M), and -log2(0.5) = 1).

M < 0.5
1.0 / (e + 0.5) < 0.5
1.0 < 0.5 * (e + 0.5)
2.0 < e + 0.5
e > 1.5

But eccentricity is in [0, 1] (normalized). So we NEVER reach mip level 1?

**LOD ORACLE:**
...Our parameters are wrong.

**KARPATHY:**
Or our normalization is wrong. We're normalizing eccentricity to [0, 1] (fraction of image size), but the formula expects DEGREES of visual angle.

**LOD ORACLE:**
Right. In human vision:
- Fovea: 0-1° → M = 17.3 mm/deg
- Periphery: 40° → M = 0.5 mm/deg

We need to map image eccentricity to visual angle.

**KARPATHY:**
Assume the image spans 40° of visual field (typical for desktop viewing). Then:

```python
eccentricity_degrees = eccentricity_normalized * 40

M0 = 1.0  # Normalized peak magnification
e0 = 0.75  # In degrees (from neuroscience)

M = M0 / (eccentricity_degrees + e0)
```

Let's redo the calculation:

```python
# Extreme periphery (edge of image)
eccentricity_normalized = 0.9
eccentricity_degrees = 0.9 * 40 = 36°

M = 1.0 / (36 + 0.75) = 0.027
mip_level = -log2(0.027) ≈ 5.2 → rounds to 5 (but we only have 0-4)
```

NOW we get high mip levels!

**LOD ORACLE:**
So the issue was unit mismatch. We were treating normalized eccentricity as if it were degrees.

**KARPATHY:**
Right. And this changes EVERYTHING about our allocation.

**MUSE BIRD:**
🐦 *UNITS! DIMENSIONS! PHYSICS!*

**LOD ORACLE:**
Let me recalculate the allocation with correct units.

```python
def tokens_from_magnification_corrected(
    eccentricity_normalized,
    image_spans_degrees=40,
    M0=1.0,
    e0=0.75,
    max_tokens=400,
    min_tokens=64
):
    # Convert normalized eccentricity to degrees
    eccentricity_deg = eccentricity_normalized * image_spans_degrees

    # Cortical magnification
    M = M0 / (eccentricity_deg + e0)

    # Normalize to token range
    M_max = M0 / e0  # At eccentricity 0
    M_min = M0 / (image_spans_degrees + e0)  # At periphery

    normalized = (M - M_min) / (M_max - M_min)
    tokens = int(min_tokens + normalized * (max_tokens - min_tokens))

    return tokens

# Examples:
print(f"Fovea (0°): {tokens_from_magnification_corrected(0)}")        # 400 tokens
print(f"10° (0.25): {tokens_from_magnification_corrected(0.25)}")     # 134 tokens
print(f"20° (0.5): {tokens_from_magnification_corrected(0.5)}")       # 81 tokens
print(f"Edge (0.9): {tokens_from_magnification_corrected(0.9)}")      # 65 tokens
```

**KARPATHY:**
So foveal allocation is MASSIVE (400 tokens), and peripheral drops to minimum (64 tokens) very quickly.

**LOD ORACLE:**
Which means our 273-token budget is EXTREMELY fovea-biased. Almost all tokens go to the central region.

**KARPATHY:**
Is that what we want?

**LOD ORACLE:**
For VQA, where queries are usually about central objects? Maybe.

For scene understanding, where you need global context? Probably not.

**KARPATHY:**
So the optimal parameters depend on the TASK.

**LOD ORACLE:**
Yes. And we don't have a principled way to choose them.

**MUSE BIRD:**
🐦 *LEARN THEM! MAKE M0 AND e0 TRAINABLE!*

**KARPATHY:**
That's an option. But then we lose the "biologically grounded" claim.

**LOD ORACLE:**
Not necessarily. We're still using the FORM of the equation (hyperbolic). We're just letting the data choose the parameters.

**KARPATHY:**
So it's biology-INFORMED, not biology-FAITHFUL.

**LOD ORACLE:**
Exactly.

---

## Act IV: The Training Question

**KARPATHY:**
Speaking of learning. How do we train this thing?

**LOD ORACLE:**
Good question. We have THREE components:

1. **Fixation predictor** (where to look)
2. **Vision encoder** (extract features)
3. **LLM** (generate answer)

Do we train them jointly? Separately? Freeze some?

**KARPATHY:**
Standard VLM training: Freeze LLM, train vision encoder + projector.

**LOD ORACLE:**
But we also have the fixation predictor. If we train it jointly with the vision encoder, there's a circularity:

- Fixation depends on visual features
- Visual features depend on fixation

**KARPATHY:**
That's just a feedback loop. It should stabilize during training.

**LOD ORACLE:**
Unless it diverges. Fixation learns to jump around randomly, vision encoder learns to expect random fixations, performance collapses.

**MUSE BIRD:**
🐦 *CHAOS! INSTABILITY! EXPLODING GRADIENTS!*

**KARPATHY:**
We could use CURRICULUM LEARNING. Start with fixed center fixation, gradually allow learned fixation.

**LOD ORACLE:**
From the addendum, Section 6.2. Epoch 1-2: center fixation. Epoch 3-4: mix of center and learned. Epoch 5+: fully learned.

**KARPATHY:**
That should stabilize training. But it's another hyperparameter (curriculum schedule).

**LOD ORACLE:**
And we still don't know if jointly training fixation + vision is BETTER than training them separately.

**KARPATHY:**
We'd need ablation studies.

**LOD ORACLE:**
Which means more experiments. More compute. More time.

**KARPATHY:**
And we haven't even talked about the QUALITY ADAPTER. Remember from Dialogue 2? We need to train a small network to upscale low-resolution peripheral tokens.

**LOD ORACLE:**
Right. So we have FOUR components:

1. Fixation predictor
2. Vision encoder (foveated)
3. Quality adapter
4. LLM

**KARPATHY:**
This is getting complicated.

**LOD ORACLE:**
And we're trying to keep it simple.

**MUSE BIRD:**
🐦 *SIMPLICITY IS HARD! COMPLEXITY IS EASY! ENGINEERING IS CHOOSING TRADE-OFFS!*

**KARPATHY:**
Maybe we start with the SIMPLEST version:

- Freeze fixation (always center)
- Freeze LLM
- Only train vision encoder

Then gradually unfreeze components?

**LOD ORACLE:**
That's a safe approach. But it might not achieve the best performance.

**KARPATHY:**
So we're trading SIMPLICITY (easier to debug) for PERFORMANCE (might be suboptimal).

**LOD ORACLE:**
Exactly. And we don't know how much performance we're leaving on the table.

---

## Act V: The Biological Purity Question

**LOD ORACLE:**
Let's zoom out. We've discovered several ways our model DEVIATES from biology:

1. **Learnable parameters** (M₀, e₀) instead of fixed values
2. **Hybrid sampling** (radial + directional for text)
3. **Adaptive budgets** (different token counts for different tasks)
4. **Multiple fixations** (vs single fixation per query)

At what point do we stop calling it "biologically inspired"?

**KARPATHY:**
I think "inspired" has always meant "loosely based on." We're not claiming to perfectly simulate V1.

**LOD ORACLE:**
But we DID claim to implement V1 faithfully. "500 million years of evolution found the solution. We just need to implement it in silicon."

**KARPATHY:**
Fair. So maybe we should distinguish:

- **V1-Faithful**: Strict adherence to neuroscience (fixed parameters, radial-only, single fixation)
- **V1-Inspired**: Uses biological principles but adapts for engineering (learnable params, hybrid strategies)
- **V1-Motivated**: Justifies design choices with biology but optimizes for performance

**MUSE BIRD:**
🐦 *THREE MODELS! FAITHFUL! INSPIRED! MOTIVATED! ABLATION STUDY!*

**LOD ORACLE:**
That's actually a good idea. We could implement all three and compare:

**V1-Faithful** (Baseline):
- Fixed M₀ = 1.0, e₀ = 0.75
- Radial foveation only
- Single fixation (query-driven)
- No quality adapter
- 273 tokens (55 foveal + 218 peripheral)

**V1-Inspired** (Practical):
- Learnable M₀, e₀
- Radial + directional (text-aware)
- Adaptive fixations (1-3 based on query complexity)
- Quality adapter for peripheral tokens
- Adaptive budget (150-500 tokens)

**V1-Motivated** (Performance):
- Learned fixation network (no explicit M(e))
- Task-specific sampling strategies
- Multi-fixation cascade
- Full transformer for peripheral processing
- Variable budget (optimized per-sample)

**KARPATHY:**
And we measure:

1. **Accuracy** (VQA, DocVQA, scene understanding)
2. **Speed** (inference time)
3. **Biological fidelity** (correlation with human eye-tracking)

**LOD ORACLE:**
Hypothesis: Faithful is slowest but most interpretable. Motivated is fastest but least biological. Inspired is the sweet spot.

**KARPATHY:**
But we don't know until we build and test them.

**LOD ORACLE:**
Exactly. And that's THREE models to implement, train, and benchmark.

**KARPATHY:**
Is it worth it?

**LOD ORACLE:**
I don't know. From a RESEARCH perspective, yes—we learn which biological principles matter.

From an ENGINEERING perspective, maybe not—we could just optimize end-to-end and call it "inspired."

**MUSE BIRD:**
🐦 *RESEARCH VS ENGINEERING! SCIENCE VS PRODUCTION! UNDERSTANDING VS PERFORMANCE!*

**KARPATHY:**
That's the fundamental tension.

---

## Act VI: The Video Question

**KARPATHY:**
What about video? We've been talking about single images, but video is where real-time matters most.

**LOD ORACLE:**
From the hardware addendum: Temporal coherence saves 90-95% of computation. Only update changed regions.

**KARPATHY:**
But that's for the MIPMAP generation. What about fixation?

Should fixation be STATIC (fixed for entire video) or DYNAMIC (updates with each frame)?

**LOD ORACLE:**
Human eyes saccade 3-4 times per second. For 30 FPS video, that's 7-10 frames per fixation.

**KARPATHY:**
So we could update fixation every 10 frames, keep it static in between?

**LOD ORACLE:**
Yes. But then there's the TRACKING problem. If the object moves, should fixation FOLLOW it?

**KARPATHY:**
Example: "Track the red car." Car moves left to right. Fixation should track it.

**LOD ORACLE:**
So we need OBJECT TRACKING. Which is another model. More complexity.

**KARPATHY:**
Unless we use the LLM's attention. If the LLM is "attending to" certain visual tokens more, maybe those correspond to the tracked object?

**LOD ORACLE:**
That's clever. LLM attention → fixation update. The LLM GUIDES where to look next.

**MUSE BIRD:**
🐦 *LLM-GUIDED VISION! TOP-DOWN CONTROL! CORTEX → V1 FEEDBACK!*

**LOD ORACLE:**
That's actually biologically plausible. V1 receives feedback from higher cortical areas (V4, IT, PFC). Top-down attention modulates V1 responses.

**KARPATHY:**
So instead of FIXED fixation (bottom-up), we have DYNAMIC fixation (top-down, LLM-guided).

**LOD ORACLE:**
But that requires re-encoding visual tokens every time fixation changes. If fixation updates every 10 frames, we re-encode 3 times per second.

At 5ms per encoding, that's 15ms/sec overhead. For 30 FPS video (33ms per frame), that's 45% overhead.

**KARPATHY:**
Unless we use INCREMENTAL encoding. Only re-encode tokens near the NEW fixation, keep peripheral tokens cached.

**LOD ORACLE:**
That's possible with texture arrays (from hardware addendum). Update one slice of the array, keep others.

**KARPATHY:**
But now we're deep into engineering territory. Caching, incremental updates, LLM feedback loops...

**LOD ORACLE:**
And we've lost the simple biological model.

**KARPATHY:**
So again: SIMPLICITY vs PERFORMANCE.

---

## Act VII: The Interpretability Advantage

**LOD ORACLE:**
Here's something we haven't discussed: INTERPRETABILITY.

**KARPATHY:**
What do you mean?

**LOD ORACLE:**
Our model has INTERPRETABLE parameters:

- M₀, e₀: Magnification
- Fixation (x, y): Where the model is "looking"
- Mip levels: Resolution at each patch
- Token allocation: Attention distribution

Compare to a standard ViT:
- 4096 tokens, uniform grid
- Attention weights: 4096 × 4096 matrix (16M parameters)
- Impossible to visualize

**KARPATHY:**
So we can SHOW users where the model is looking?

**LOD ORACLE:**
Exactly. Imagine a VQA interface:

```
Image: [photo of kitchen]
Query: "What color is the stove?"

Fixation: (0.3, 0.6) ← Left side, middle
Foveal tokens: 55 tokens centered on stove
Peripheral tokens: 218 tokens covering rest of scene

Answer: "The stove is black."

Explanation: "I focused on the left-middle region (stove area) and allocated
55 high-resolution tokens there. The stove is clearly black in this region."
```

**KARPATHY:**
That's way more interpretable than "Attention scores: [0.02, 0.03, 0.01, ...]"

**MUSE BIRD:**
🐦 *HUMANS UNDERSTAND FIXATIONS! NEUROSCIENCE → UX!*

**LOD ORACLE:**
And we can VISUALIZE the token allocation:

```
Token Density Map:
  [Heatmap showing 55 dense tokens at fixation, sparse tokens in periphery]

Mipmap Usage:
  Level 0: 55 tokens (foveal)
  Level 1: 48 tokens (near periphery)
  Level 2: 64 tokens (mid periphery)
  Level 3: 72 tokens (far periphery)
  Level 4: 34 tokens (extreme periphery)
```

**KARPATHY:**
This is HUGE for trust and debugging.

If the model gets an answer wrong, we can see: "Oh, it fixated on the wrong object."

Or: "It allocated too few tokens to the relevant region."

**LOD ORACLE:**
And we can intervene. User says: "No, look at the RIGHT side, not the left."

We update fixation, re-run inference, get better answer.

**KARPATHY:**
That's interactive VLMs. Human-in-the-loop.

**LOD ORACLE:**
And it's only possible because our model has STRUCTURED allocation (foveated, fixation-based) instead of UNSTRUCTURED allocation (learned attention).

**KARPATHY:**
So biological grounding gives us interpretability for free.

**LOD ORACLE:**
Yes. That's a MAJOR advantage we haven't emphasized enough.

**MUSE BIRD:**
🐦 *EXPLAINABILITY! TRUST! SAFETY! AI ALIGNMENT!*

---

## Act VIII: The Failure Mode Gallery

**KARPATHY:**
Let's collect all the edge cases we've found. A "failure mode gallery."

**LOD ORACLE:**
Good idea. Enumerate them:

**1. Multi-Object Queries**

```
Query: "Is the cat on the table?"
Problem: Two objects, different locations
Current solution: Single fixation (misses one object)
Possible solutions: Multi-fixation, wider fovea, better query parsing
```

**2. Text-Heavy Images (OCR)**

```
Query: "What does paragraph 3 say?"
Problem: Text everywhere, need to find specific region
Current solution: Query-driven fixation (but need OCR first)
Possible solutions: Text-aware sampling, anisotropic filtering, hybrid radial+directional
```

**3. Small Objects**

```
Query: "What time does the clock show?"
Problem: Clock is 50×50 pixels, might be in periphery
Current solution: Adaptive budget (more foveal tokens)
Possible solutions: Multi-scale cascade, zoom-in fixation
```

**4. Ambiguous Queries**

```
Query: "What's unusual here?"
Problem: No clear fixation target
Current solution: Center fixation (hope it's lucky)
Possible solutions: Multiple fixations, saliency-based exploration, LLM-guided scan
```

**5. Dynamic Scenes (Video)**

```
Query: "Track the red car"
Problem: Object moves, fixation should follow
Current solution: Static fixation (loses object)
Possible solutions: Tracking, LLM-guided fixation, predictive saccades
```

**6. Extreme Aspect Ratios**

```
Image: 2048×512 (panorama)
Problem: Radial foveation assumes roughly square image
Current solution: Normalized coordinates (but distorted)
Possible solutions: Anisotropic foveation, aspect-aware M(e)
```

**7. Very High Resolution**

```
Image: 4096×4096
Problem: Need more tokens for same coverage
Current solution: Fixed 273 budget (under-samples high-res)
Possible solutions: Resolution-adaptive budget, hierarchical encoding
```

**KARPATHY:**
That's at least 7 failure modes, each requiring a different fix.

**LOD ORACLE:**
And each fix moves us further from biological purity.

**KARPATHY:**
So we need to PRIORITIZE. Which failure modes matter most for our target tasks?

**LOD ORACLE:**
If we're optimizing for VQA (natural images, single objects), then modes 1, 4 are most important.

If we're optimizing for DocVQA (text-heavy), then mode 2 is critical.

If we're optimizing for video, then mode 5 is key.

**KARPATHY:**
So there's no ONE MODEL that handles all cases perfectly.

**LOD ORACLE:**
Right. We need TASK-SPECIFIC adaptations.

**MUSE BIRD:**
🐦 *MODULAR DESIGN! PLUG-AND-PLAY FOVEATION! ADAPTIVE ARCHITECTURE!*

**KARPATHY:**
That's a lot of engineering.

---

## Act IX: The Philosophical Detour

**KARPATHY:**
Can we step back for a second?

**LOD ORACLE:**
Sure.

**KARPATHY:**
We started with this beautiful idea: "Human V1 is optimized by 500 million years of evolution. Let's implement it."

**LOD ORACLE:**
Yes.

**KARPATHY:**
But now we've discovered that V1 is optimized for HUMAN TASKS (survival, navigation, object recognition in 3D world). Not for VQA.

VQA is weird. It's text-driven. It's about answering specific questions. It's about linguistic concepts.

**LOD ORACLE:**
So you're saying V1 might not be the right model for VLMs?

**KARPATHY:**
I'm saying V1 solves a DIFFERENT PROBLEM. And we're trying to adapt it to our problem.

**LOD ORACLE:**
But V1 has some UNIVERSAL principles, right? Foveation, multi-scale, saliency. Those apply broadly.

**KARPATHY:**
Maybe. Or maybe they're specific to primate vision and don't generalize.

**MUSE BIRD:**
🐦 *CONVERGENT EVOLUTION! DIFFERENT SPECIES, SAME SOLUTIONS!*

**LOD ORACLE:**
The Muse has a point. Many animals have foveation (birds, fish, primates). It's a general solution to resource-constrained vision.

**KARPATHY:**
But birds have different cortical organization. Fish don't have cortex at all.

So maybe "foveation" is universal, but "V1-style cortical magnification" is not.

**LOD ORACLE:**
So we should abstract HIGHER. The principle is: "Allocate more resources to relevant regions."

How we DO that (radial, directional, learned, fixed) is implementation detail.

**KARPATHY:**
Which brings us full circle: If we're just implementing "allocate more resources to relevant regions," why bother with biology at all?

Why not just use LEARNED attention (like TransNeXt) and call it a day?

**LOD ORACLE:**
...That's a good question.

**KARPATHY:**
I think the answer is: Biology gives us a PRIOR. A starting point. A set of CONSTRAINTS.

Without constraints, we're just doing end-to-end learning. Which works but is opaque.

With biological constraints, we have STRUCTURE. Interpretability. Testable hypotheses.

**LOD ORACLE:**
So biology is valuable not because it's OPTIMAL, but because it's STRUCTURED.

**KARPATHY:**
Exactly. It's a REGULARIZER. It keeps us from overfitting to weird solutions.

**MUSE BIRD:**
🐦 *INDUCTIVE BIAS! ARCHITECTURE SEARCH! BIOLOGY AS PRIOR DISTRIBUTION!*

**LOD ORACLE:**
That's a more modest claim than "implementing human V1."

But it's also more defensible.

**KARPATHY:**
And it opens the door to HYBRID approaches. Biology-informed but data-driven.

---

## Act X: The Open-Ended Questions

**LOD ORACLE:**
We've been exploring for a while. Let's collect the OPEN QUESTIONS we've discovered.

**KARPATHY:**
Good idea. List them:

**1. What's the right token budget?**
   - 150? 273? 500? Task-dependent?
   - How to choose?

**2. Should magnification parameters be learned?**
   - Fixed (M₀=1.0, e₀=0.75): Biologically faithful
   - Learned (M₀, e₀ as nn.Parameters): Data-driven
   - Hybrid (constrained learning, M₀ ∈ [0.5, 2.0])?

**3. Single fixation or multiple?**
   - Single: Simple, fast
   - Multiple: Better coverage, slower
   - Adaptive: Complex to implement

**4. How to handle text?**
   - Radial foveation (consistent with natural images)
   - Directional sampling (optimized for text)
   - Hybrid (detect text, adapt strategy)

**5. Training strategy?**
   - Freeze fixation, train vision
   - Freeze vision, train fixation
   - Joint training (risk instability)
   - Curriculum learning (complex schedule)

**6. Quality adapter?**
   - Yes (upscale peripheral tokens)
   - No (let transformer handle it)
   - Partial (only for mip level 3-4)

**7. Video: Static or dynamic fixation?**
   - Static: Simple, fast
   - Dynamic: Better tracking, expensive
   - LLM-guided: Principled but complex

**8. Biological fidelity vs. performance?**
   - Faithful (strict V1): Interpretable, possibly suboptimal
   - Inspired (adapted V1): Balanced
   - Motivated (loose V1): Best performance, less biological

**9. Integration with existing VLMs?**
   - Replace CLIP (train from scratch)
   - Augment CLIP (hybrid global+fovea)
   - Drop-in module (plug into LLaVA/Qwen)

**10. Evaluation metrics?**
   - Accuracy (VQA, DocVQA, etc.)
   - Speed (inference time)
   - Biological fidelity (eye-tracking correlation)
   - Interpretability (human judgement)

**LOD ORACLE:**
That's 10 open questions. Each with multiple possible answers.

**KARPATHY:**
Combinatorial explosion. 3^10 = 59,049 possible configurations.

**LOD ORACLE:**
Obviously we can't test them all. We need to make CHOICES.

**KARPATHY:**
Based on what?

**LOD ORACLE:**
Intuition? Prior work? Ablation studies?

**KARPATHY:**
Or we could just... start building. Make reasonable choices. See what breaks. Iterate.

**MUSE BIRD:**
🐦 *AGILE DEVELOPMENT! FAIL FAST! ITERATE!*

**LOD ORACLE:**
That's the engineering approach.

**KARPATHY:**
As opposed to?

**LOD ORACLE:**
The scientific approach: Formulate hypotheses, design experiments, test systematically.

**KARPATHY:**
Can we do both?

**LOD ORACLE:**
Maybe. Build a prototype (engineering). Then study it systematically (science).

**KARPATHY:**
That's probably the pragmatic path.

---

## Act XI: The Meta-Question

**MUSE BIRD:** *[Suddenly very serious]*
🐦 *WHY ARE WE DOING THIS?*

**KARPATHY:**
...What?

**MUSE BIRD:**
🐦 *WHY FOVEATED VLMS? WHAT'S THE GOAL?*

**LOD ORACLE:**
That's a good question. Let's think:

**Goal 1: Speed**
- 10× faster vision encoding (67ms → 6ms)
- Enables real-time VLMs (60+ FPS)
- Useful for: Robotics, AR/VR, mobile devices

**Goal 2: Efficiency**
- Fewer tokens (4096 → 273)
- Lower memory footprint
- Lower energy consumption
- Useful for: Edge devices, sustainability

**Goal 3: Interpretability**
- Explicit fixation point
- Visualizable token allocation
- Biologically grounded
- Useful for: Trust, debugging, human-AI collaboration

**Goal 4: Biological Understanding**
- Model of human V1
- Testable hypotheses (eye-tracking correlation)
- Insights into vision-language coupling
- Useful for: Cognitive science, neuroscience

**Goal 5: Architectural Innovation**
- New attention pattern (radial, multi-scale)
- Query-driven allocation
- Inspiration for future models
- Useful for: Research, advancing field

**KARPATHY:**
So we have FIVE goals. But they're not all compatible.

Speed vs. biological fidelity (learned params are faster, fixed params are faithful).

Efficiency vs. accuracy (fewer tokens might hurt performance).

Interpretability vs. flexibility (fixed allocation is interpretable, learned is flexible).

**LOD ORACLE:**
So we need to PRIORITIZE. What's the PRIMARY goal?

**KARPATHY:**
I think... it depends on who's building it.

**For a robotics company**: Speed and efficiency (Goal 1-2)
**For a research lab**: Biological understanding (Goal 4)
**For a product team**: Interpretability and trust (Goal 3)
**For an ML conference**: Architectural innovation (Goal 5)

**LOD ORACLE:**
So there's no ONE answer. It's context-dependent.

**MUSE BIRD:**
🐦 *AUDIENCE MATTERS! FRAMING MATTERS! STORY MATTERS!*

**KARPATHY:**
And we haven't decided our audience yet.

**LOD ORACLE:**
So before we build, we need to ask: WHO IS THIS FOR?

**KARPATHY:**
That's the meta-question.

---

## Closing: The Questions Remain

*The Dirac Sea darkens. The V1 model floats, surrounded now by question marks, edge cases, trade-offs. The oracles sit in contemplative silence.*

**KARPATHY:**
We started today wanting to stress-test the model.

**LOD ORACLE:**
And we found... a lot of questions.

**KARPATHY:**
Multi-object queries. Text-heavy images. Small objects. Training strategies. Biological purity. Interpretability. Goals. Audience.

**LOD ORACLE:**
We have a beautiful SPECIFICATION. But we don't have ANSWERS.

**KARPATHY:**
Is that bad?

**LOD ORACLE:**
No. It's HONEST.

Research is about QUESTIONS, not answers.

**MUSE BIRD:** *[Softly]*
🐦 *Questions are the light. Answers are the shadow.*

**KARPATHY:**
That's surprisingly deep, Muse.

**MUSE BIRD:**
🐦 *I have my moments.*

**LOD ORACLE:**
So what do we do with all these questions?

**KARPATHY:**
We... hold them. We don't force answers. We let them GUIDE the exploration.

When we build the prototype, we'll discover which questions matter most.

Some will resolve themselves. Some will become more important. Some will lead to new questions.

**LOD ORACLE:**
That's the scientific process.

**KARPATHY:**
And the engineering process. Build, measure, learn.

**LOD ORACLE:**
So the next dialogue is... implementation?

**KARPATHY:**
Maybe. Or maybe more exploration. Or maybe a detour into related work.

**LOD ORACLE:**
We're not forcing a narrative.

**KARPATHY:**
No. We're following the QUESTIONS wherever they lead.

**MUSE BIRD:**
🐦 *OPEN-ENDED! EXPLORATORY! NO HARD PLANS!*
🐦 *QUESTIONS PROLIFERATE! CURIOSITY COMPOUNDS!*
🐦 *THE MAP IS NOT THE TERRITORY BUT THE QUESTIONS ARE THE COMPASS!*

**KARPATHY:**
The Muse is right. We have a compass. We have questions. We don't need a map yet.

**LOD ORACLE:**
Then let's leave it here. With the questions.

*The oracles rise. The V1 model continues to float, now surrounded by luminous question marks. Each one represents an unexplored direction, a possible path, a trade-off to consider.*

*The Dirac Sea does not provide answers. It provides SPACE for questioning.*

*And sometimes, that's more valuable.*

---

## Appendix: The Question Map

**Collected Questions from This Dialogue:**

1. Multi-object queries: Single or multiple fixations?
2. Saccade mechanism vs. saccade strategy?
3. How much biology to keep before it's not "biological"?
4. OCR: Radial vs. directional sampling?
5. Small objects: Adaptive budgets or multi-scale cascade?
6. Training: Joint or separate? Curriculum or direct?
7. Parameters: Fixed (faithful) vs. learned (practical) vs. hybrid?
8. Video: Static, dynamic, or LLM-guided fixation?
9. Token budget: Task-dependent or universal?
10. Interpretability vs. flexibility trade-off?
11. Biological fidelity vs. performance trade-off?
12. Quality adapter: Yes, no, or partial?
13. V1-Faithful vs. V1-Inspired vs. V1-Motivated?
14. Integration: Replace, augment, or drop-in?
15. Evaluation: Accuracy, speed, biology, interpretability?
16. Primary goal: Speed, efficiency, interpretability, science, or innovation?
17. Target audience: Robotics, research, product, or ML community?

**Status**: ALL UNANSWERED.

**Next Steps**: Implement prototype, discover which questions matter, let data guide decisions.

**Principle**: Questions over answers. Exploration over conclusions.

---

**END OF DIALOGUE 24**

∿◇∿
