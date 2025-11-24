# Part 35: The Failure Modes - When Relevance Breaks Down
*Wherein the oracles explore edge cases and catastrophic failures, discovering that intelligent attention allocation can fail in surprisingly specific ways, each revealing assumptions baked into the system*

---

## Opening: The Overconfidence Problem

**KARPATHY:**
We've built it. We've trained it. We've evaluated it. Now let's talk about how it FAILS.

**LOD ORACLE:**
Every system has failure modes. The question is: can we predict them?

**THEAETETUS:**
Why focus on failures? Shouldn't we celebrate successes?

**KARPATHY:**
Failures teach us more than successes. When the system breaks, we see its assumptions.

**SOCRATES:**
*Appearing*

Well said. To know what something IS, study where it ISN'T. Boundaries define essence.

---

## Act I: The Adversarial Text Problem

**KARPATHY:**
Failure Mode #1: Camouflaged text.

*He shows an image:*

```
Image: Busy magazine page
- Background: Colorful patterns, gradients
- Foreground: Important text in LOW contrast (gray-on-gray)
- Distractor: Bright red box with irrelevant text

Query: "What does the gray text say?"

ARR-COC allocation:
  Red box: 250 tokens (HIGH saliency, bright color)
  Gray text: 50 tokens (LOW saliency, blends in)
  Background: 150 tokens

Result: FAILURE - misses the query-relevant text
```

**THEAETETUS:**
But didn't we handle this with inverted edges (Channel 7)?

**KARPATHY:**
Yes, Channel 7 catches LOW-contrast text. But here's the problem:

```python
# Scoring pipeline
info_score = 0.4 * edges + 0.3 * highpass + 0.3 * distance
persp_score = 0.6 * saliency + 0.4 * motion
partic_score = clip_similarity(patch, query)

balanced = 0.2 * info + 0.2 * persp + 0.6 * partic
```

**Even if info_score detects low-contrast text (via inverted edges), perspectival score is LOW (gray blends in). The red box dominates saliency.**

**If participatory score (CLIP) doesn't recognize the gray text, balanced score stays low.**

**SOCRATES:**
So the failure is a CASCADE. One scorer fails (persp), and participatory doesn't compensate enough.

**LOD ORACLE:**
**Root cause:** Over-reliance on saliency. The opponent process (Exploit ↔ Explore) is stuck in Exploit mode (focus on salient regions).

**Fix:**
```python
# Adaptive weighting
if persp_scores.max() < 0.3:  # No highly salient regions
    # Rebalance: reduce perspectival weight, increase propositional
    weights = [0.4, 0.1, 0.5]  # info, persp, partic
else:
    weights = [0.2, 0.2, 0.6]  # Normal
```

**KARPATHY:**
So when saliency is weak, rely more on information content (edges, structure).

---

## Act II: The Semantic Void

**KARPATHY:**
Failure Mode #2: Semantically empty images.

*Shows example:*

```
Image: Abstract art (Jackson Pollock-style)
- No recognizable objects
- High entropy (lots of edges everywhere)
- No semantic segmentation possible (SAM fails)

Query: "What does this painting represent?"

ARR-COC allocation:
  Uniform distribution (every region scores similarly)
  No concentration (all regions have edges)

Result: FAILURE - allocates tokens uniformly, no intelligent focus
```

**THEAETETUS:**
But isn't uniform allocation correct here? There's no "important" region.

**KARPATHY:**
Maybe. But then we're no better than a dumb grid. The whole point of ARR-COC is to be SMART about allocation.

**SOCRATES:**
Perhaps the failure is not in allocation, but in RECOGNITION of futility?

**KARPATHY:**
Meaning?

**SOCRATES:**
The system should recognize: "This image has no focal structure. I will allocate uniformly." versus blindly applying the same algorithm.

**Meta-awareness:** Knowing when your strategy won't work.

**LOD ORACLE:**
That requires a SECOND layer: allocation strategy selection.

```python
def allocate_with_strategy_selection(image, query):
    """
    First: Decide WHICH allocation strategy
    Then: Apply it
    """
    # Analyze image properties
    semantic_structure = measure_semantic_structure(image)  # 0-1
    # High = clear objects, low = uniform/abstract

    if semantic_structure > 0.7:
        # Clear structure → cluster-first cascade
        return cluster_based_allocation(image, query)

    elif semantic_structure < 0.3:
        # Abstract/uniform → grid sampling with diversity
        return grid_with_diversity(image, query)

    else:
        # Medium structure → standard ARR-COC
        return arr_coc_allocation(image, query)
```

**KARPATHY:**
So instead of ONE strategy always, ADAPT the strategy to image characteristics.

---

## Act III: The Temporal Incoherence

**KARPATHY:**
Failure Mode #3: Video frame jitter.

*Shows video sequence:*

```
Video: Camera panning across a document
Frame 1: Camera focuses on title → ARR-COC allocates 300 tokens to title
Frame 2: Camera moves slightly → New regions appear at edges
  → ARR-COC reallocates: 200 tokens to title, 100 tokens to new text
Frame 3: Camera moves more → Composition shifts
  → ARR-COC reallocates completely: 50 tokens to title, 250 tokens to body

Result: JITTER - allocation jumps around frame-to-frame
User experience: Confusing, answers change as camera moves
```

**THEAETETUS:**
But didn't we handle this with temporal cache (Channels 34-36)?

**KARPATHY:**
We tried. But the temporal cache only SMOOTHS allocation. It doesn't STABILIZE across large motions.

**Problem:**
```python
# Temporal blending (from Part 32)
blended_relevance = (
    0.7 * cached_relevance +  # Previous frame
    0.3 * current_relevance    # Current frame
)
```

**This works for SMALL motions (object moves 10 pixels). But for LARGE motions (camera pans, scene cuts), cached relevance is INVALID (warped to wrong location).**

**LOD ORACLE:**
We need scene change detection.

```python
def temporal_allocation_with_scene_detection(current_frame, prev_frame, query):
    """
    Detect scene changes, reset allocation when needed.
    """
    if prev_frame is None:
        # First frame
        return fresh_allocation(current_frame, query)

    # Detect scene change
    motion_magnitude = compute_optical_flow(prev_frame, current_frame).abs().mean()

    if motion_magnitude > SCENE_CHANGE_THRESHOLD:
        # Large motion → scene changed → reset
        return fresh_allocation(current_frame, query)

    else:
        # Small motion → smooth transition
        cached = warp_relevance(prev_relevance, optical_flow)
        current = score_relevance(current_frame, query)
        return blend(cached, current, alpha=0.7)
```

**KARPATHY:**
So smooth for continuity, but BREAK continuity when scene changes.

**SOCRATES:**
The system must recognize CONTEXT BOUNDARIES. Within a context, smooth. Across contexts, reset.

---

## Act IV: The Query Ambiguity Trap

**KARPATHY:**
Failure Mode #4: Vague queries.

*Example:*

```
Image: Kitchen with table, stove, refrigerator, window
Query: "Describe this."

ARR-COC allocation:
  Attempts to score query-relevance (participatory)
  But "describe this" doesn't specify WHAT is relevant
  Defaults to saliency (perspectival) → allocates to visually prominent objects

Result: Semi-success? Focuses on salient items, but misses context
```

**THEAETETUS:**
Why is that a failure? Salient items are reasonable to describe first.

**KARPATHY:**
Because the system doesn't RECOGNIZE the ambiguity. It confidently allocates based on saliency, without knowing the query is underspecified.

**Ideal behavior:**

```python
def handle_vague_query(image, query):
    """
    Detect vague queries, request clarification or diversify allocation.
    """
    query_specificity = measure_specificity(query)
    # "What is the red object?" → high (0.9)
    # "Describe this" → low (0.1)

    if query_specificity < 0.3:
        # Vague query → diversify allocation (broad coverage)
        # Allocate tokens spatially uniformly
        return diversified_allocation(image, num_tokens=273)

    else:
        # Specific query → focused allocation
        return focused_allocation(image, query, num_tokens=273)
```

**SOCRATES:**
Again, meta-awareness. The system must know WHAT KIND of situation it's in.

**KARPATHY:**
And adjust its strategy accordingly.

---

## Act V: The Overallocation Cascade

**KARPATHY:**
Failure Mode #5: Runaway token allocation.

*Example:*

```
Image: Dense textbook page (equations, diagrams, captions)
Query: "Explain the main equation."

ARR-COC allocation:
  Main equation: 400 tokens (max budget)
  Related diagram: 350 tokens (high relevance)
  Caption: 300 tokens (explains equation)
  Context paragraph: 250 tokens

  Total: 1300 tokens requested
  Budget: 273 tokens × ~300 avg = 82K tokens
  Available: 100K tokens (VLM limit)
```

**THEAETETUS:**
Everything is relevant! How do you prioritize?

**KARPATHY:**
Current system just scales down:

```python
# TokenAllocator (from Part 32)
if total_allocated > self.total_budget:
    # Scale down to fit budget
    scale_factor = self.total_budget / total_allocated
    final_budgets = initial_budgets * scale_factor
```

**Problem:** UNIFORM scaling. The 400-token region and the 250-token region both get scaled by 0.8×. But maybe the main equation deserves FULL budget and others should be dropped entirely?

**Better approach:**

```python
def allocate_with_hard_constraints(positions, scores, budgets, total_budget):
    """
    Greedy allocation: fund highest-priority regions fully, drop others.
    """
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)

    allocated = []
    remaining_budget = total_budget

    for idx in sorted_indices:
        budget_needed = budgets[idx]

        if budget_needed <= remaining_budget:
            # Fund fully
            allocated.append((positions[idx], budget_needed))
            remaining_budget -= budget_needed
        else:
            # Can't afford this region → skip
            continue

        if remaining_budget < MIN_BUDGET:
            # Budget exhausted
            break

    return allocated
```

**SOCRATES:**
Greedy allocation. Fund the best, ignore the rest.

**KARPATHY:**
Yeah, but is that right? Maybe we SHOULD allocate a little to everything for context?

**Tension: Concentrate (deep understanding of few) vs Spread (shallow understanding of many)**

---

## Act VI: The Out-of-Distribution Collapse

**KARPATHY:**
Failure Mode #6: Novel image types.

*Example:*

```
Training data: Natural images (photos, documents, paintings)
Test image: Medical X-ray

ARR-COC allocation:
  CLIP embeddings: Meaningless (CLIP not trained on X-rays)
  Saliency: Detects bright spots (artifacts, not relevant anatomy)
  Clusters: Segments random noise patterns

Result: CATASTROPHIC FAILURE - allocation is nonsense
```

**LOD ORACLE:**
This is the classic out-of-distribution problem. The system assumes images look LIKE training data.

**SOCRATES:**
Can the system detect its own confusion?

**KARPATHY:**
Maybe. Measure UNCERTAINTY.

```python
def allocate_with_uncertainty_aware(image, query):
    """
    Detect OOD images, fall back to safe strategy.
    """
    # Measure uncertainty across scorers
    info_score_variance = compute_variance(info_scorer(all_positions))
    persp_score_variance = compute_variance(persp_scorer(all_positions))
    partic_score_variance = compute_variance(partic_scorer(all_positions))

    total_variance = info_score_variance + persp_score_variance + partic_score_variance

    if total_variance < LOW_VARIANCE_THRESHOLD:
        # All regions score similarly → model is confused
        # Fall back to uniform grid
        return grid_allocation(image, num_tokens=273)

    else:
        # Normal operation
        return arr_coc_allocation(image, query)
```

**THEAETETUS:**
So if the model is uncertain, it admits defeat and uses a safe baseline?

**KARPATHY:**
Better than confidently allocating to nonsense.

---

## Act VII: The Feedback Loop Trap

**KARPATHY:**
Failure Mode #7: Self-fulfilling allocation.

*Example:*

```
Training scenario:
  Model learns: "Faces are important" (because training data has many face-centric queries)

  Result: Model ALWAYS allocates heavily to faces

Test scenario:
  Query: "What color is the wall?"
  Image: Person standing in front of wall

  Model allocation: 60% to person's face, 40% to wall

  Result: FAILURE - over-allocated to face due to learned bias
```

**SOCRATES:**
The model learned a HEURISTIC (faces matter), but applies it UNIVERSALLY, even when inappropriate.

**KARPATHY:**
This is a training data bias. If most queries in training involved faces, the model generalizes incorrectly.

**LOD ORACLE:**
Fix: Adversarial training.

```python
def adversarial_training(model, dataset):
    """
    Generate queries that INTENTIONALLY avoid salient regions.
    """
    for image in dataset:
        # Detect salient regions
        salient_regions = detect_salient(image)  # e.g., faces, bright objects

        # Generate anti-salient queries
        other_regions = image_regions - salient_regions
        query = generate_query_about_region(other_regions)
        # e.g., "What color is the wall?" (NOT "Who is the person?")

        # Train model to allocate to non-salient regions
        loss = train_step(model, image, query, target_region=other_regions)
```

**THEAETETUS:**
So you explicitly teach the model: "Don't always look at faces"?

**KARPATHY:**
Right. Balance the training distribution. Otherwise, learned heuristics become harmful biases.

---

## Act VIII: The Recursive Failure

**KARPATHY:**
Failure Mode #8: Multi-fixation loops.

*Example:*

```
Multi-fixation protocol (from Part 17):
  Fixation 1: Allocate to region A, process, low confidence
  Fixation 2: Allocate to region B, process, low confidence
  Fixation 3: Allocate to region A again (forgot we already looked there!)
  Fixation 4: Allocate to region B again...
  ...
  Loop indefinitely, never confident
```

**THEAETETUS:**
The system forgets what it already examined?

**KARPATHY:**
If we don't track fixation history properly, yes.

**Fix:**

```python
def multi_fixation_with_memory(image, query, max_fixations=5):
    """
    Track which regions have been examined, avoid revisiting.
    """
    examined_regions = set()
    context = []

    for i in range(max_fixations):
        # Allocate tokens
        allocation = allocate_tokens(image, query, context)

        # Mark regions as examined
        for pos in allocation.positions:
            examined_regions.add(pos)

        # Process
        output = vlm_process(allocation, query)
        context.append(output)

        # Check confidence
        if output.confidence > 0.9:
            break

        # NEXT fixation: Penalize examined regions
        # Allocate to unexplored areas
        relevance_scores = score_all_positions(image, query, context)

        for pos in examined_regions:
            relevance_scores[pos] *= 0.1  # Heavy penalty

    return integrate_context(context, query)
```

**SOCRATES:**
Memory prevents repetition. The system must remember its own history.

---

## Act IX: The Catastrophic Forgetting

**KARPATHY:**
Failure Mode #9: Training destroys pretrained knowledge.

*Scenario:*

```
Stage 1: Pretrain CLIP on 400M images (general vision understanding)
Stage 2: Train ARR-COC on VQA (document-heavy dataset)

Result after Stage 2:
  VQA accuracy: 68% (good!)
  General image understanding: 45% (was 72% before training!)

CATASTROPHIC FORGETTING - model lost general knowledge, overfit to documents
```

**THEAETETUS:**
Can't you just... not overtrain?

**KARPATHY:**
It's not about duration. It's about DISTRIBUTION SHIFT.

If training data is 80% documents, 20% photos, the model forgets how to handle photos.

**Solutions:**

```python
# Solution 1: Freeze CLIP
# Don't fine-tune CLIP embeddings, only train ARR-COC components

# Solution 2: Regularization
loss = vqa_loss + 0.1 * regularization_loss
# Regularization: Keep CLIP weights close to pretrained values

# Solution 3: Continual learning
# Periodically sample from general dataset during training
for epoch in range(100):
    vqa_batch = sample_vqa_data()
    general_batch = sample_imagenet_data()

    loss = vqa_loss(vqa_batch) + 0.2 * general_loss(general_batch)
```

**SOCRATES:**
The system must maintain balance between SPECIALIZATION (VQA) and GENERALIZATION (all images).

---

## Act X: The Failure Taxonomy

**KARPATHY:**
Let me organize what we've found.

```
ARR-COC FAILURE MODES:

1. PERCEPTUAL FAILURES (sensing)
   ├─ Camouflaged text (low-contrast missed)
   ├─ Semantic void (abstract art, no structure)
   └─ Out-of-distribution (novel image types)

2. STRATEGIC FAILURES (allocation logic)
   ├─ Over-reliance on saliency (exploit bias)
   ├─ Query ambiguity (vague questions)
   └─ Overallocation cascade (budget exceeded)

3. TEMPORAL FAILURES (video/sequences)
   ├─ Frame jitter (allocation jumps)
   ├─ Scene change missed (context boundary)
   └─ Recursive loops (multi-fixation repeats)

4. LEARNING FAILURES (training)
   ├─ Feedback loop bias (learned heuristics)
   ├─ Catastrophic forgetting (lost generalization)
   └─ Distribution mismatch (training ≠ test)
```

**LOD ORACLE:**
Each failure reveals an ASSUMPTION:

1. Perceptual: "Saliency indicates importance" (FALSE for camouflaged content)
2. Strategic: "One strategy fits all" (FALSE for diverse image types)
3. Temporal: "Smooth transitions always valid" (FALSE for scene cuts)
4. Learning: "Training data represents all cases" (FALSE for OOD)

**SOCRATES:**
And each assumption, when violated, produces failure. To fix failures, recognize and relax assumptions.

**THEAETETUS:**
But can we predict NEW failures? Ones we haven't seen?

**KARPATHY:**
That's the hard part. We can only fix failures we ENCOUNTER. Unknown unknowns remain.

**SOCRATES:**
Then test broadly. Probe edges. Seek adversarial examples. The more you explore failure space, the more you understand success space.

**MUSE BIRD:**
*Swooping down*

🐦 *FAILURE IS FEEDBACK! Every break reveals a boundary! Love your bugs, they teach you limits!*

---

## Closing: The Robustness Checklist

**KARPATHY:**
Based on these failure modes, here's a robustness checklist:

```python
class RobustnessChecklist:
    """
    Pre-deployment validation checklist.
    """

    def validate_robustness(self, model):
        tests = []

        # Test 1: Low-contrast text
        tests.append(self.test_camouflaged_text(model))

        # Test 2: Abstract images
        tests.append(self.test_semantic_void(model))

        # Test 3: Video jitter
        tests.append(self.test_temporal_stability(model))

        # Test 4: Vague queries
        tests.append(self.test_query_ambiguity(model))

        # Test 5: Overallocation
        tests.append(self.test_budget_constraints(model))

        # Test 6: Out-of-distribution
        tests.append(self.test_ood_detection(model))

        # Test 7: Learned biases
        tests.append(self.test_allocation_fairness(model))

        # Test 8: Multi-fixation memory
        tests.append(self.test_fixation_loops(model))

        # Test 9: Catastrophic forgetting
        tests.append(self.test_general_knowledge_retention(model))

        # Report
        passing = sum(tests)
        total = len(tests)

        if passing == total:
            print(f"✓ All {total} robustness tests passed!")
            return True
        else:
            print(f"✗ {total - passing} / {total} tests failed")
            print("Review failures before deployment")
            return False
```

**LOD ORACLE:**
And for each failure mode, a mitigation strategy:

```python
MITIGATION_STRATEGIES = {
    'camouflaged_text': 'Adaptive weighting when saliency is low',
    'semantic_void': 'Strategy selection based on image structure',
    'video_jitter': 'Scene change detection + reset policy',
    'query_ambiguity': 'Specificity measurement + diversification',
    'overallocation': 'Greedy allocation with hard budget constraints',
    'ood_collapse': 'Uncertainty detection + fallback to grid',
    'feedback_bias': 'Adversarial training + balanced datasets',
    'fixation_loops': 'Memory tracking + exploration penalty',
    'catastrophic_forgetting': 'Freeze CLIP + continual learning'
}
```

**SOCRATES:**
You cannot prevent all failures. But you can DESIGN FOR graceful degradation.

**When the system fails, fail safely.**

---

**END OF PART 35**

∿◇∿

**KEY INSIGHT:** Every failure mode reveals a hidden assumption. Robust systems recognize their assumptions, detect violations, and degrade gracefully rather than catastrophically. Testing failure modes is as important as optimizing success cases.
