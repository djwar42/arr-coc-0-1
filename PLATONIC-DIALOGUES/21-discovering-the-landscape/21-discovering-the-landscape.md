---
summary: whereby Karpathy and the LOD Oracle discover recent research (2024-2025) exploring similar ideas—PyramidDrop with 90 citations doing training-free pyramid pruning via bottom-up saliency, Dynamic Pyramid Network adapting based on input, and multiple other approaches converging on pyramid+foveation—but realize their unique angle is query-awareness (participatory knowing) rather than query-agnostic saliency, positioning ARR-COC as complementary rather than competitive since PyramidDrop optimizes for general efficiency while they optimize for query-specific relevance realization
---

# Part 21: Discovering the Landscape
*Wherein the LOD Oracle and Karpathy Oracle discover recent research that validates their direction while revealing they're not alone in this space*

---

## Opening: The Search Results

*Scene: The Dirac Sea. KARPATHY and LOD ORACLE sit surrounded by glowing search results floating in the quantum foam—papers from 2024-2025, all exploring similar ideas.*

**KARPATHY:**
So... we just searched for recent work on pyramid sampling, foveation, and adaptive token allocation.

**LOD ORACLE:**
And we found a LOT.

**KARPATHY:**
PyramidDrop. Dynamic Pyramid Network. Foveated Retinotopy. HiRED. FastVLM.

**LOD ORACLE:**
All from 2024-2025. All doing variations of what we just designed.

**KARPATHY:**
Are we... late to the party?

**LOD ORACLE:**
Let me look closer at what they're actually doing.

---

## Act I: PyramidDrop - The 90-Citation Elephant

**LOD ORACLE:**
PyramidDrop (ICLR 2025, 90 citations). "Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models."

**KARPATHY:**
Training-free pyramid pruning. That's literally our v2 + v4 (training-free methods) combined.

**LOD ORACLE:**
Let me parse the title more carefully:

**"Training-Free"** → No fine-tuning needed (like SparseVLM from our knowledge)
**"Pyramid"** → Multi-scale processing (our v2)
**"Token Pruning"** → Removes tokens, doesn't merge them
**"Efficient Large Vision-Language Models"** → Exactly our target

**KARPATHY:**
So they beat us to it?

**LOD ORACLE:**
Partially. The question is: HOW do they use the pyramid?

From the description: "By leveraging bottom-up visual saliency at both region and token levels, our method effectively identifies and removes redundant visual tokens."

**KARPATHY:**
Bottom-up saliency. That's perspectival knowing from Vervaeke—what stands out visually.

But they're not using the QUERY. It's saliency-driven, not query-driven.

**LOD ORACLE:**
Exactly! They're doing:
```
Pyramid → Saliency scores → Prune tokens
```

We're doing:
```
Pyramid + Query → Relevance scores (participatory knowing) → Allocate tokens
```

**KARPATHY:**
So they're query-AGNOSTIC, we're query-AWARE.

**LOD ORACLE:**
Yes. And that's a significant difference.

For "Describe this image" (broad query), saliency works fine.
For "What's the formula in the top-right?" (specific query), you NEED query-awareness.

**KARPATHY:**
So PyramidDrop is solving a related but different problem?

**LOD ORACLE:**
Exactly. They're optimizing for general-purpose efficiency (works for any query).
We're optimizing for query-specific relevance (adapts per query).

**KARPATHY:**
That's actually a good positioning. We're not competing—we're complementary.

---

## Act II: Dynamic Pyramid Network - The March 2025 Paper

**KARPATHY:**
What about Dynamic Pyramid Network? That one's from March 2025—very recent.

**LOD ORACLE:**
"Dynamic Pyramid Network for Efficient Multimodal Large Language Models."

Let me parse:
**"Dynamic"** → Adapts based on input (not fixed)
**"Pyramid Network"** → Multi-scale architecture
**"Multimodal LLMs"** → Our exact domain

**KARPATHY:**
This sounds even CLOSER to what we're doing.

**LOD ORACLE:**
The key word is "dynamic." They're likely adapting the pyramid structure based on image characteristics.

Our foveated pyramid (v2.5) adapts based on QUERY + image.

The difference:
- **DPN**: Image → Dynamic pyramid structure → Process
- **Us**: (Image + Query) → Foveated pyramid + Fixation point → Process

**KARPATHY:**
So they're missing the query-driven fixation?

**LOD ORACLE:**
Probably. Most VLM work treats the query as SEPARATE from visual encoding:

**Standard pipeline**:
```
1. Encode image (query-independent)
2. Encode query (image-independent)
3. Cross-attention (combine)
```

**Our pipeline**:
```
1. Coarse encode image
2. Query determines fixation point
3. Fine encode around fixation (query-dependent!)
4. Cross-attention (combine)
```

**KARPATHY:**
So we're doing TWO-STAGE encoding where stage 2 is guided by the query.

**LOD ORACLE:**
Exactly. And I don't think Dynamic Pyramid Network does that, based on the title.

**KARPATHY:**
We should read the actual paper to confirm.

**LOD ORACLE:**
Agreed. But the title suggests they're adapting the pyramid to IMAGE characteristics, not QUERY characteristics.

---

## Act III: Foveated Retinotopy - The Biological Validation

**KARPATHY:**
Okay, what about this one: "Foveated Retinotopy Improves Classification" (Oct 2025, arXiv 2402.15480).

**LOD ORACLE:**
This is the biological grounding paper. Let me read the description:

"Biologically Inspired Deep Learning Model for Efficient Foveal-Peripheral Vision. Frontiers in Computational Neuroscience, 15, 2021."

Wait, it's from 2021 but updated in October 2025?

**KARPATHY:**
Probably a revised version or follow-up work.

**LOD ORACLE:**
The key claim: "Foveated Retinotopy Improves Classification."

This is DIRECT validation that biological foveation helps!

**KARPATHY:**
What's their approach?

**LOD ORACLE:**
From the description: "End-to-end neural model for foveal-peripheral vision, inspired by retino-cortical mapping in primates and humans."

**Retino-cortical mapping** = how the retina maps to V1 cortex (log-polar!)

**KARPATHY:**
So they're using log-polar transforms?

**LOD ORACLE:**
Likely. And they're showing it IMPROVES classification, not just reduces compute.

This is huge for us—it means our v6 (log-polar) isn't just about efficiency, it might actually IMPROVE accuracy.

**KARPATHY:**
Wait, how does reducing resolution in periphery IMPROVE accuracy?

**LOD ORACLE:**
Regularization! By blurring the periphery, you force the model to focus on the FOVEAL region (high-res).

It's like dropout—you remove information to prevent overfitting to irrelevant details.

**KARPATHY:**
So foveation is a form of attention-based regularization?

**LOD ORACLE:**
Exactly. And it's biologically validated—humans don't overfit to peripheral clutter because we literally can't SEE it in high resolution.

**MUSE BIRD:** *[Appearing suddenly]*
🐦 *BIOLOGY GIVES YOU FOCUS! Blur the noise, see the signal!*

---

## Act IV: HiRED - The Already-Published Solution

**KARPATHY:**
HiRED: "Attention-Guided Token Dropping for Efficient Inference" (AAAI 2025, 41 citations).

We mentioned this in Dialogue 18, right?

**LOD ORACLE:**
Yes! HiRED was in our knowledge expansion (Level 3: Dynamic Reduction During Generation).

**From our notes**:
"HiRED: High-to-low Resolution Elastic Dependency"

They use resolution-based elastic dependency—different resolutions for different parts of the image.

**KARPATHY:**
So they're doing multi-resolution, like our pyramid?

**LOD ORACLE:**
Yes, but the key is "Attention-Guided."

They use ATTENTION SCORES from the model to decide which tokens to keep.

High attention → keep token
Low attention → drop token

**KARPATHY:**
That's similar to our attention-driven LOD idea from Dialogue 19!

**LOD ORACLE:**
Exactly. We speculated:

"What if attention scores DIRECTLY control LOD levels? High attention → fine tokens, low attention → coarse tokens?"

HiRED is doing exactly this.

**KARPATHY:**
So they beat us to that idea too?

**LOD ORACLE:**
For DURING GENERATION, yes. They're dropping tokens dynamically as the model processes.

We're focusing on BEFORE GENERATION—allocating tokens upfront based on predicted relevance.

**Different problems**:
- **HiRED**: "I'm generating an answer. Which tokens can I drop NOW to save compute?"
- **Us**: "I haven't started yet. Which tokens should I encode FIRST to maximize relevance?"

**KARPATHY:**
Pre-allocation vs dynamic dropping.

**LOD ORACLE:**
Exactly. And they're complementary—you could use BOTH.

1. Our method: Allocate 273 tokens intelligently (foveated pyramid)
2. HiRED: Drop tokens during generation based on attention

**KARPATHY:**
So we're stage 1, HiRED is stage 2?

**LOD ORACLE:**
Yes. Different parts of the pipeline.

---

## Act V: FastVLM - The Apple Deployment

**KARPATHY:**
FastVLM (Apple Research, July 2025). We cited this extensively.

**LOD ORACLE:**
"Efficient Vision Encoding for Vision Language Models."

From Apple's description: "Difficulty-aware pyramid sampling. FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image."

**KARPATHY:**
Difficulty-aware. So they DO use some form of query or task difficulty?

**LOD ORACLE:**
The key phrase: "solely by scaling the input image."

I think they mean: instead of complex token merging/pruning, they just DOWNSAMPLE the image for easy queries, keep full resolution for hard queries.

**Simple approach**:
```python
if is_difficult(query, image):
    resolution = 1024  # High res → more tokens
else:
    resolution = 256   # Low res → fewer tokens

tokens = encode(resize(image, resolution))
```

**KARPATHY:**
That's... actually really simple.

**LOD ORACLE:**
And effective! Apple deployed it in production.

The insight: you don't need fancy pyramid structures if you can just DECIDE upfront whether to use high or low resolution.

**KARPATHY:**
But they're losing the multi-scale benefits?

**LOD ORACLE:**
Yes. They're picking ONE scale (low or high), not combining multiple scales like our pyramid approach.

**Trade-off**:
- **FastVLM**: Simple, fast, easy to deploy (Apple production)
- **Our pyramid**: Complex, slower, potentially higher quality (research)

**KARPATHY:**
So FastVLM is the "good enough" solution that ships?

**LOD ORACLE:**
Exactly. And we're the "can we do 3% better?" research exploration.

**KARPATHY:**
That's a fair division. Production vs research.

---

## Act VI: What They're NOT Doing

**KARPATHY:**
Alright, we've found a lot of related work. What are they NOT doing that we are?

**LOD ORACLE:**
Let me list the unique aspects of our Foveated Pyramid (v2.5):

**1. Explicit cortical magnification formula**

None of the papers mention M(e) = M₀/(e + e₀).

They do "foveation" or "pyramids," but not with the explicit neuroscience math.

**2. Query-driven fixation point**

PyramidDrop uses saliency (bottom-up).
FastVLM uses difficulty (image statistics).
HiRED uses attention (during generation).

NONE use query-driven fixation BEFORE encoding:
```python
fixation_xy = find_fixation_from_query(query, coarse_image)
```

**3. Unified biological + signal processing framework**

We're combining:
- Cortical magnification (neuroscience)
- Gaussian pyramids (signal processing)
- Log-polar sampling (biological vision)
- Cross-attention relevance (ML)

Most papers pick ONE of these, not all four.

**4. Vervaekean relevance realization**

We're explicitly measuring relevance through four dimensions:
- Propositional (information content)
- Perspectival (saliency)
- Participatory (query-relevance)
- Procedural (learned importance)

The other papers use ONE dimension (usually just saliency or attention).

**KARPATHY:**
So we're broader and more theoretically grounded?

**LOD ORACLE:**
Yes. But that's a double-edged sword.

**Advantage**: Deeper understanding, potentially better performance
**Disadvantage**: More complex, harder to implement, harder to publish

**KARPATHY:**
The classic research vs engineering trade-off.

**LOD ORACLE:**
Exactly.

---

## Act VII: The Positioning Question

**KARPATHY:**
So how do we position our work relative to this landscape?

**LOD ORACLE:**
Three options:

**Option A: Comprehensive Comparison**

"We compare our foveated pyramid approach against PyramidDrop, HiRED, and FastVLM on DocVQA, COCO, TextVQA."

**Pros**: Thorough, scientific, shows how we fit in the landscape
**Cons**: Requires implementing their methods, 3× more work

**Option B: Orthogonal Contribution**

"While prior work (PyramidDrop, HiRED) focuses on saliency-driven pruning, we explore query-driven allocation with biological grounding."

**Pros**: Positions us as different, not competing
**Cons**: Smaller contribution, less general

**Option C: Unification Framework**

"We unify pyramids (FastVLM), foveation (Foveated Retinotopy), and query-awareness (HiRED) into a single biologically-grounded framework."

**Pros**: Big claim, broad contribution
**Cons**: Ambitious, harder to validate

**KARPATHY:**
Which do you prefer?

**LOD ORACLE:**
Depends on our goal.

**If goal = PUBLICATION**: Option B (easier, clearer contribution)
**If goal = UNDERSTANDING**: Option A (learn the most)
**If goal = BIG SWING**: Option C (high risk, high reward)

**KARPATHY:**
What if we do Option A for ourselves (implement and compare), but POSITION as Option B (orthogonal contribution)?

**LOD ORACLE:**
Perfect. We do the thorough work (Option A) but tell the story simply (Option B).

**Paper narrative**:
"Prior work uses bottom-up saliency for token allocation. We show that TOP-DOWN query-driven allocation, inspired by cortical magnification, improves performance on query-specific tasks like DocVQA."

**KARPATHY:**
That's clean. We're not claiming to beat everything—just showing query-driven helps for query-specific tasks.

**LOD ORACLE:**
Exactly. Narrow, defensible contribution.

---

## Act VIII: The Timeline Adjustment

**KARPATHY:**
Does this change our 12-week plan from Dialogue 20?

**LOD ORACLE:**
Yes. Let me revise:

**OLD PLAN** (Dialogue 20):
- Week 1-2: v1 (grid) baseline
- Week 3-4: v2 (pyramid)
- Week 5-6: v2.5 (foveated pyramid)
- Week 7-8: Ablations

**NEW PLAN** (adjusted):
- Week 1-2: v1 (grid) baseline
- Week 3-4: **PyramidDrop reproduction** (saliency-driven)
- Week 5-6: v2.5 (foveated pyramid, query-driven)
- Week 7-8: **Direct comparison**: PyramidDrop vs Ours

**KARPATHY:**
So we're adding PyramidDrop as a BASELINE, not just grid?

**LOD ORACLE:**
Yes. The real question is:

**"Does query-driven allocation (ours) beat saliency-driven allocation (PyramidDrop) on query-specific tasks?"**

**KARPATHY:**
And if the answer is no?

**LOD ORACLE:**
Then we learned that queries don't matter as much as visual saliency for token allocation.

That's ALSO a valuable result—it tells us the query information is redundant (already captured by saliency).

**KARPATHY:**
Negative results are still results.

**LOD ORACLE:**
Exactly.

---

## Act IX: The Biological Grounding Advantage

**MUSE BIRD:** *[Hopping excitedly]*
🐦 *But you have something they DON'T! The BIOLOGY!*

**KARPATHY:**
What do you mean?

**MUSE BIRD:**
🐦 *They have pyramids. They have pruning. They have attention.*
🐦 *But do they have M(e) = M₀/(e + e₀)? NO!*
🐦 *Do they have 273 tokens = 273 V1 clusters? NO!*
🐦 *Do they cite neuroscience? BARELY!*

**LOD ORACLE:**
The Muse has a point.

Most ML papers treat biological inspiration as a metaphor:
"We use foveation, like the human eye."

We're treating it as a SPEC:
"Cortical magnification factor is M(e) = M₀/(e + e₀). We implement this EXACTLY."

**KARPATHY:**
So we're taking biology more seriously?

**LOD ORACLE:**
Yes. And that might be our DIFFERENTIATOR.

**Paper angle**:
"Prior work uses foveation as a loose inspiration. We implement a FAITHFUL model of primate cortical magnification, with explicit M(e) formula, 273-token budget matching V1 cluster count, and log-polar sampling matching retinotopic maps."

**KARPATHY:**
That's a strong biological claim.

**LOD ORACLE:**
And we can VALIDATE it:

**Experiment**: Does our allocation pattern match human eye-tracking data?

If humans fixate on region A when asked query Q, does our fixation-finding algorithm also select region A?

**KARPATHY:**
Ooh, that's a neat validation. Human-VLM alignment.

**LOD ORACLE:**
Exactly. If we match human fixations, we're not just "efficient"—we're "cognitively plausible."

**KARPATHY:**
That opens neuroscience publication venues: Vision Sciences, JOV (Journal of Vision), Computational Neuroscience.

**LOD ORACLE:**
Yes! Multi-disciplinary contribution:
- **ML venues** (CVPR, ICCV): Efficiency gains
- **Neuro venues** (VSS, JOV): Biological fidelity
- **Cog Sci venues** (CogSci): Relevance realization framework

---

## Act X: The Research Landscape Map

**KARPATHY:**
Let me try to map out the landscape we've discovered.

**LOD ORACLE:**
Go ahead.

**KARPATHY:**
```
TOKEN ALLOCATION FOR VLMs (2024-2025 LANDSCAPE)

Axis 1: WHEN is allocation done?
├─ Pre-encoding (ours, FastVLM): Decide tokens BEFORE encoding
└─ During generation (HiRED, DyRate): Drop tokens DURING processing

Axis 2: WHAT drives allocation?
├─ Bottom-up (PyramidDrop): Visual saliency
├─ Image statistics (FastVLM): Difficulty estimation
├─ Attention (HiRED): Learned attention scores
└─ Query-driven (ours): Query + cortical magnification

Axis 3: HOW is allocation done?
├─ Single-scale (FastVLM): Pick one resolution
├─ Pruning (PyramidDrop, SparseVLM): Remove tokens
├─ Merging (ToMe, AIM): Combine similar tokens
└─ Hierarchical (ours): Multi-scale pyramid + foveation

Axis 4: WHY (theoretical grounding)?
├─ Engineering (PyramidDrop, HiRED): "It works"
├─ Signal processing (FastVLM): Multi-scale theory
└─ Neuroscience (ours, Foveated Retinotopy): Cortical magnification
```

**LOD ORACLE:**
That's a good map. And it shows we're in a relatively unexplored corner:

**Pre-encoding + Query-driven + Hierarchical + Neuroscience**

Most papers are:
**During-generation + Saliency + Pruning + Engineering**

**KARPATHY:**
So we're not competing head-to-head?

**LOD ORACLE:**
Not directly. We're exploring a different part of the design space.

**KARPATHY:**
Which means we're more likely to find something novel, but also more risky (less validated).

**LOD ORACLE:**
Exactly. Classic exploration-exploitation trade-off.

---

## Act XI: The Humility Check

**KARPATHY:**
Alright, real talk. Are we fooling ourselves?

**LOD ORACLE:**
What do you mean?

**KARPATHY:**
We've designed this elaborate foveated pyramid system with cortical magnification formulas and Vervaekean relevance realization.

But PyramidDrop is simpler, already published, has 90 citations, and probably WORKS.

Are we over-engineering?

**LOD ORACLE:**
...That's a fair question.

Let me think through this honestly.

**Evidence we might be over-engineering**:
1. FastVLM (simple approach) is deployed at Apple (production!)
2. PyramidDrop (training-free) has 90 citations in a few months
3. Our approach requires: SAM, pyramids, log-polar, fixation finding, M(e) formula...

**Evidence we might NOT be**:
1. Foveated Retinotopy shows biological foveation IMPROVES accuracy (not just efficiency)
2. None of the existing work uses query-driven fixation
3. Neuroscience grounding might unlock human-like performance

**KARPATHY:**
So the jury's out?

**LOD ORACLE:**
Yes. And that's WHY we do the experiment.

If v2.5 (foveated pyramid) beats PyramidDrop by <2%, we're over-engineering.
If it beats by >5%, the complexity is justified.

**KARPATHY:**
And if it's in between? 2-5% improvement?

**LOD ORACLE:**
Then it's a RESEARCH contribution (novel, interesting) but not an ENGINEERING contribution (too complex for production).

**Different venues**:
- <2%: Don't publish (negative result, maybe arXiv)
- 2-5%: Research conference (NeurIPS, ICLR—theory-friendly)
- >5%: Application conference (CVPR, ICCV—practitioners care)

**KARPATHY:**
That's a useful framing.

---

## Closing: The Landscape Integrated

*The Dirac Sea calms. The search results settle into organized layers around the two oracles.*

**KARPATHY:**
So we've discovered we're not alone. PyramidDrop, FastVLM, HiRED, Dynamic Pyramid Networks—they're all exploring similar ideas.

**LOD ORACLE:**
But we're exploring a different CORNER of the space:
- Query-driven (not just saliency)
- Biologically faithful (explicit M(e) formula)
- Pre-encoding (not just dynamic dropping)
- Unified framework (Vervaeke + pyramids + log-polar)

**KARPATHY:**
And we've adjusted our plan:
- Add PyramidDrop as baseline (not just grid)
- Position as "query-driven vs saliency-driven"
- Validate against human eye-tracking (cognitive plausibility)
- Target multi-disciplinary venues (ML + neuroscience)

**LOD ORACLE:**
Exactly. We're not claiming to beat everything.

We're claiming: "For QUERY-SPECIFIC tasks, query-driven allocation with cortical magnification outperforms saliency-driven allocation."

**KARPATHY:**
That's a narrow, testable hypothesis.

**LOD ORACLE:**
And if it's wrong, we learned something valuable: queries are redundant, saliency is sufficient.

**KARPATHY:**
Either way, we advance understanding.

**MUSE BIRD:** *[Final flourish]*
🐦 *YOU ARE NOT FIRST! But you are NOT LAST!*
🐦 *You are in the WAVE! The 2024-2025 wave of VLM token allocation!*
🐦 *SURF THE WAVE! Don't fight it!*

**LOD ORACLE:**
The Muse is right. We're part of a research wave.

PyramidDrop (ICLR 2025)
FastVLM (Apple, July 2025)
HiRED (AAAI 2025)
Dynamic Pyramid Network (March 2025)
Foveated Retinotopy (Oct 2025)

This is the MOMENT for this research direction.

**KARPATHY:**
The zeitgeist.

**LOD ORACLE:**
Exactly. Ideas become ripe when the time is right.

**KARPATHY:**
So we're not late—we're ON TIME.

**LOD ORACLE:**
Yes. And we have a unique angle (biology + query-awareness) that differentiates us.

**KARPATHY:**
Then let's build it and see what we learn.

**LOD ORACLE:**
Agreed.

*The search results organize themselves into a clear taxonomy—the landscape is mapped, the position is clear, the path forward is visible.*

**KARPATHY:**
One more dialogue after this?

**LOD ORACLE:**
One more. Dialogue 22: The Decision.

We'll finalize the exact approach, write the pseudocode, and commit to the implementation.

**KARPATHY:**
From exploration to specification.

**LOD ORACLE:**
Exactly.

*The Dirac Sea shimmers. The landscape is understood. The wave is identified. The next dialogue awaits.*

---

**END OF DIALOGUE 21**

∿◇∿
