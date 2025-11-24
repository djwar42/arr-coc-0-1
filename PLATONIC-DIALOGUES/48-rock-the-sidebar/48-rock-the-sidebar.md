# Part 48: Rock the Sidebar - The Running Back Method
*Wherein the assembly discovers that DeepSeek's sparse mask is the perfect foundation for ARR's query-aware augmentation, and Socrates proposes the simplest experiment to prove it*

---

**Participants:** Socrates, Theaetetus, Karpathy, LOD Oracle, Muse Bird, DeepSeek-OCR Oracle

**Setting:** The sidebar lounge. Part 47's scroll glows on the table. Whiteboards covered in diagrams: fixed patches, gestalt+saccades, the string teaching (fixed budget). The team has assembled to stress-test the architecture.

---

## Act I: The Gathering - Quick Recap

**KARPATHY:**
*Standing at whiteboard*

Alright, we're here to rock the sidebar. Part 47 gave us the scroll's teaching. Let me recap fast:

**The Scroll Architecture:**
- Fixed patch size: 14×14 (training stable)
- Gestalt first: 256 base tokens (uniform coverage)
- Saccades after: 273 query-selected tokens (relevance-ordered)
- Total: 529 tokens (fixed budget, perfect batching)

**The Core Idea:**
Augmentation, not substitution. Keep gestalt, add saccades.

**THEAETETUS:**

And the string teaching: fixed budget. Always 273 saccades, no variable tensor sizes.

**MUSE BIRD:**
🐦 *Fixed patches! Fixed budget! The scroll was wise!*

**LOD ORACLE:**

The biological grounding is sound. Human vision: gestalt perception, then saccadic focus. Foveated rendering uses the same pattern.

**SOCRATES:**
*Entering, sitting*

I've been reading your pre-dialogue documents. Five of them. Quite thorough.

**KARPATHY:**

Yeah, we did our homework. Token ordering, patch extraction, training dynamics, DeepSeek comparison, and Theaetetus's questioning session.

**SOCRATES:**

And what have you learned?

**KARPATHY:**
*Pauses*

That we might be starting from the wrong place.

---

## Act II: The DeepSeek Revelation

**DEEPSEEK-OCR ORACLE:**
*Materializing, joining the group*

You called?

**KARPATHY:**

Yeah. We need to talk about your architecture.

**DEEPSEEK-OCR ORACLE:**

SAM gestalt, learned compression, CLIP encoding. Serial design. 4096 → 256 tokens.

**SOCRATES:**

And this compression... it's not random, is it?

**DEEPSEEK-OCR ORACLE:**

No. Learned. A neural network scores all 4096 patches during training. It discovers:
- Text regions → high scores
- Object boundaries → high scores
- Uniform backgrounds → low scores
- Salient objects → high scores

The network learns what usually matters.

**THEAETETUS:**

So you have a 64×64 grid. 4096 patches. Each 16×16 pixels.

You score all 4096, select top-256, discard 3840.

**DEEPSEEK-OCR ORACLE:**

Correct.

**THEAETETUS:**

And the 256 selected patches... they're scattered across the image? Not contiguous?

**DEEPSEEK-OCR ORACLE:**

Exactly. Spatially discontinuous. Highly sparse. An irregular, fragmented distribution.

**Imagine a binary mask over the 64×64 grid:**
- 256 positions = 1 (keep)
- 3840 positions = 0 (discard)

The mask is learned, not designed. It reflects what the network discovered matters for vision-language tasks.

**MUSE BIRD:**
🐦 *A sparse mask! Islands of importance in a sea of discarded patches!*

**KARPATHY:**
*Staring at the whiteboard*

Holy shit.

**SOCRATES:**

What is it?

**KARPATHY:**

The mask. The sparse mask. That's not a limitation. That's an OPPORTUNITY.

---

## Act III: The Running Back Insight

**KARPATHY:**
*Drawing furiously*

Listen. DeepSeek compresses 4096 → 256 using learned priors.

Those priors are IMAGE-ONLY. No query context.

Same 256 patches whether you ask "what color is the sky?" or "read the bottom-right copyright text."

**LOD ORACLE:**

A fixed importance map, regardless of task.

**KARPATHY:**

Right. But here's the thing...

*Points at diagram*

SAM already computed all 4096 patch features. They're sitting in memory. No re-encoding needed.

**THEAETETUS:**
*Eyes widening*

So when ARR wants to add saccades...

**KARPATHY:**

We go BACK to the 4096 SAM features!

**DEEPSEEK-OCR ORACLE:**

Ah. I see where you're going.

**KARPATHY:**
*Excited*

DeepSeek says: "Here are the 256 patches that usually matter based on my learned priors."

ARR says: "Thanks! But let me RUN BACK and CHECK if this specific query needs something you missed."

**Goes back to SAM's 4096 tokens (already computed!)**

ARR: "You ignored patch 3847 (bottom-right corner) because your prior says 'corners are low-priority.' But the question is 'what does the copyright say?' That patch is CRITICAL."

**Selects 273 additional patches using query-aware scoring.**

**Result:** 256 (DeepSeek learned prior) + 273 (ARR query-aware) = 529 tokens

**THEAETETUS:**
*Standing up*

The Running Back Method!

DeepSeek runs forward with learned priors.
ARR runs back to fetch what the query needs!

**SOCRATES:**
*Smiling*

A good name. And what do you run back to?

**KARPATHY:**

The full SAM gestalt. All 4096 features. We select from those using query-aware relevance scoring.

**MUSE BIRD:**
🐦 *Running back! Fetching the query-relevant! Filling the sparse mask!*

---

## Act IV: The Sparse Mask as Foundation

**LOD ORACLE:**

Let me frame this in terms of Level of Detail.

DeepSeek's learned compression creates a **base LOD allocation**:
- 256 patches get high LOD (CLIP encoding)
- 3840 patches get zero LOD (discarded)

This is efficient but uniform across queries.

**ARR adds query-aware LOD reallocation:**
- Additional 273 patches get high LOD (query-selected)
- These might overlap with DeepSeek's 256 OR fill gaps in the sparse mask

**The sparse mask becomes a FLOOR, not a ceiling.**

**DEEPSEEK-OCR ORACLE:**

Interesting. So you're augmenting my selection with query-specific additions.

**KARPATHY:**

Exactly. Your learned prior handles the "usually important" stuff efficiently.

Our query-aware scoring handles the "important for THIS question" stuff.

**THEAETETUS:**

And we reuse your SAM encoding! No re-computation!

**SOCRATES:**

This sounds elegant in theory. But does it work?

**KARPATHY:**
*Grins*

That's what we're here to find out.

---

## Act V: The Point of Entry

**SOCRATES:**

So you have a method: Running Back to Fetch Things.

You have a foundation: DeepSeek's 256 sparse mask + SAM's 4096 features.

You have a hypothesis: Query-aware augmentation improves accuracy.

What is your point of entry? How do you begin?

**KARPATHY:**

We need to prove augmentation helps BEFORE building the full ARR scorer.

**Experiment 0, but adapted for DeepSeek:**

```python
# Baseline: DeepSeek-OCR (256 tokens)
baseline = deepseek_ocr(vqa_val)  # Learned compression
# Accuracy: X%

# Experiment 1: Random augmentation (256 + 273 random)
random = deepseek_256 + random_273_from_sam_4096(vqa_val)
# If random ≈ baseline → more tokens don't help, ABANDON
# If random > baseline → augmentation helps, continue

# Experiment 2: Saliency augmentation (256 + 273 salient)
saliency = deepseek_256 + saliency_273_from_sam_4096(vqa_val)
# If saliency > random → selection quality matters

# Experiment 3: Query-simple augmentation (256 + 273 CLIP-query)
query_simple = deepseek_256 + clip_similarity_273_from_sam_4096(vqa_val)
# If query_simple > saliency → query-awareness helps!
```

**THEAETETUS:**

Three experiments. Each tests one assumption.

**KARPATHY:**

Right. And they're FAST. One day each. No training. Just selection strategies.

**LOD ORACLE:**

You're testing the SHAPE (augmentation helps?) before optimizing the MEANING (how to select?).

**KARPATHY:**

Exactly.

---

## Act VI: The Prototype - Platonic Code

**SOCRATES:**

Show me the simplest version. If you were to build this today, what would it look like?

**KARPATHY:**
*Writes on whiteboard*

```python
# PLATONIC CODE: Running Back to Fetch Things (Prototype)
# Hybrid DeepSeek + ARR method

import torch
from deepseek_ocr import DeepSeekOCR  # Frozen
from transformers import CLIPModel  # For query-simple experiment

class RunningBackFetcher:
    """
    Augments DeepSeek's sparse mask with query-aware tokens.

    The Running Back Method:
    1. DeepSeek runs forward (4096 → 256, learned prior)
    2. ARR runs back to SAM 4096 features
    3. Selects 273 additional patches using query-aware scoring
    4. Combines: 256 base + 273 saccades = 529 tokens
    """

    def __init__(self, strategy='random'):
        """
        Args:
            strategy: 'random', 'saliency', 'clip_query'
        """
        self.deepseek = DeepSeekOCR()  # Frozen model
        self.strategy = strategy

        if strategy == 'clip_query':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self, image, question):
        """
        Run DeepSeek + ARR augmentation.

        Args:
            image: [B, 3, H, W]
            question: str or List[str]

        Returns:
            all_tokens: [B, 529, d_model]
        """
        B = image.shape[0]

        # ============================================
        # STAGE 1: DeepSeek Forward Pass
        # ============================================

        with torch.no_grad():
            # SAM encoding (4096 features)
            sam_features = self.deepseek.sam(image)  # [B, 4096, 1024]

            # Learned compression (4096 → 256)
            compression_scores = self.deepseek.compression_net(sam_features)
            base_indices = torch.topk(compression_scores, k=256, dim=-1)[1]

            # DeepSeek's selected 256 patches
            base_features = torch.gather(
                sam_features,
                dim=1,
                index=base_indices.unsqueeze(-1).expand(-1, -1, 1024)
            )  # [B, 256, 1024]

        # ============================================
        # STAGE 2: ARR Running Back (select 273 more)
        # ============================================

        saccade_indices = self._select_saccades(
            sam_features,  # All 4096 features available!
            base_indices,  # What DeepSeek already selected
            question
        )  # [B, 273]

        # Fetch from SAM features (no re-encoding!)
        saccade_features = torch.gather(
            sam_features,
            dim=1,
            index=saccade_indices.unsqueeze(-1).expand(-1, -1, 1024)
        )  # [B, 273, 1024]

        # ============================================
        # STAGE 3: Concatenate
        # ============================================

        all_features = torch.cat([base_features, saccade_features], dim=1)
        # [B, 529, 1024]

        # ============================================
        # STAGE 4: CLIP + LLM (DeepSeek's pipeline)
        # ============================================

        with torch.no_grad():
            clip_tokens = self.deepseek.clip(all_features)
            answer = self.deepseek.llm(clip_tokens, question)

        return answer

    def _select_saccades(self, sam_features, base_indices, question):
        """
        Run back to SAM 4096 and select 273 patches.

        Three strategies to test:
        """
        B, N, D = sam_features.shape

        if self.strategy == 'random':
            # Experiment 1: Random selection from 4096
            # (Avoid duplicating DeepSeek's 256)
            all_indices = torch.arange(N, device=sam_features.device)
            available = self._exclude_indices(all_indices, base_indices)

            # Random sample 273 from available
            perm = torch.randperm(available.shape[1])[:273]
            saccade_indices = available[:, perm]

        elif self.strategy == 'saliency':
            # Experiment 2: Saliency-based selection
            # Compute simple saliency (edge magnitude approximation)
            # This is VERY rough - just for testing!

            # Reshape to 64×64 grid
            features_grid = sam_features.reshape(B, 64, 64, D)

            # Approximate saliency via feature magnitude variance
            saliency = features_grid.var(dim=-1)  # [B, 64, 64]
            saliency_flat = saliency.reshape(B, -1)  # [B, 4096]

            # Mask out DeepSeek's selections
            saliency_masked = self._mask_indices(saliency_flat, base_indices)

            # Top-273
            saccade_indices = torch.topk(saliency_masked, k=273, dim=-1)[1]

        elif self.strategy == 'clip_query':
            # Experiment 3: CLIP query-similarity
            # Encode query
            query_embed = self.clip.get_text_features(question)  # [B, 512]

            # Project SAM features to CLIP space (rough approximation)
            # In real implementation, would use actual CLIP features
            # For prototype, just use dot product with SAM features

            # Expand query for broadcasting
            query_expanded = query_embed.unsqueeze(1)  # [B, 1, 512]

            # Similarity scores (using first 512 dims of SAM features)
            similarity = torch.matmul(
                sam_features[:, :, :512],
                query_expanded.transpose(1, 2)
            ).squeeze(-1)  # [B, 4096]

            # Mask out DeepSeek's selections
            similarity_masked = self._mask_indices(similarity, base_indices)

            # Top-273
            saccade_indices = torch.topk(similarity_masked, k=273, dim=-1)[1]

        return saccade_indices

    def _exclude_indices(self, all_indices, exclude_indices):
        """Remove exclude_indices from all_indices."""
        # Create mask
        mask = torch.ones(all_indices.shape[0], dtype=torch.bool)
        mask[exclude_indices] = False
        return all_indices[mask].unsqueeze(0)

    def _mask_indices(self, scores, mask_indices):
        """Set scores to -inf at mask_indices."""
        scores_masked = scores.clone()
        scores_masked.scatter_(1, mask_indices, float('-inf'))
        return scores_masked
```

**THEAETETUS:**
*Reading*

This is... simple. Three experiments. Same code, different strategies.

**KARPATHY:**

That's the point. Prove the concept with the simplest possible implementation.

**SOCRATES:**

And what would these experiments tell you?

**KARPATHY:**

**Experiment 1 (random):**
Tests if augmentation (529 vs 256 tokens) helps at all, regardless of selection quality.

**Experiment 2 (saliency):**
Tests if bottom-up selection (visual saliency) beats random.

**Experiment 3 (clip_query):**
Tests if query-awareness (even simple CLIP similarity) beats saliency.

**LOD ORACLE:**

A progression. Each experiment isolates one variable.

**MUSE BIRD:**
🐦 *Random → Saliency → Query! Each step adds intelligence!*

---

## Act VII: Why This Works (The Sparse Mask Advantage)

**DEEPSEEK-OCR ORACLE:**

I see why you chose my architecture as the foundation.

**SOCRATES:**

Explain.

**DEEPSEEK-OCR ORACLE:**

My sparse mask (256 selected, 3840 discarded) creates a **learned prior about general importance.**

Text, edges, salient objects - these usually matter.

But "usually" isn't "always."

**Example:**

Image: Street scene with a small sign in bottom-right corner.

My learned prior:
- Selects car (salient, foreground)
- Selects traffic light (semantic importance)
- Selects pedestrian (human figure)
- DISCARDS bottom-right sign (corner, small, background)

**Query 1:** "How many people are in the image?"
→ My 256 patches are sufficient. Pedestrian included.

**Query 2:** "What does the sign say?"
→ My 256 patches MISSED the critical information. Sign discarded.

**ARR Running Back:**
- Sees query mentions "sign"
- Runs back to SAM 4096 features
- Scores bottom-right patches HIGH (query-content similarity)
- Adds those 273 saccades to the 529 total
- Now LLM can answer!

**THEAETETUS:**

So your learned prior handles the common case efficiently.

ARR handles the uncommon case by running back for query-specific details.

**DEEPSEEK-OCR ORACLE:**

Exactly. My 256 is a floor. Your 273 saccades raise the ceiling for query-specific tasks.

**KARPATHY:**

And we're betting that 529 query-aware tokens beats 256 learned-prior tokens.

**SOCRATES:**

What does success look like?

**KARPATHY:**

**Hypothesis 1:** random > baseline
Interpretation: More tokens help, even without smart selection

**Hypothesis 2:** saliency > random
Interpretation: Selection quality matters

**Hypothesis 3:** clip_query > saliency
Interpretation: Query-awareness adds value beyond bottom-up saliency

**If all three hold:** Build full ARR scorer (three ways of knowing, contextualized weighting)

**If any fail:** Investigate why, adjust approach.

---

## Act VIII: The Architecture Decision

**SOCRATES:**

Earlier you said you were starting from the wrong place. What did you mean?

**KARPATHY:**

Part 47's plan: Use Qwen3-VL base (256 tokens, uniform grid) + ARR saccades (273 tokens).

**Problem:** Qwen base is uniform. No learned prior. We'd be starting from scratch.

**DeepSeek approach:** Use SAM (4096 tokens) + learned compression (256 tokens) + ARR saccades (273 tokens).

**Advantage:**
- Richer gestalt (SAM 4096 vs Qwen 256)
- Learned prior (DeepSeek compression vs uniform grid)
- Reusable features (select from SAM, don't re-encode)

**LOD ORACLE:**

You're building on proven foundations rather than starting from zero.

**KARPATHY:**

Exactly. DeepSeek already proved learned compression works. We're just adding query-awareness on top.

**THEAETETUS:**

Qwen would test: "Does augmentation help when starting from uniform?"

DeepSeek tests: "Does query-awareness help when starting from learned priors?"

**KARPATHY:**

Right. And the second question is more interesting.

If query-awareness can improve even a GOOD learned prior, that's a stronger result.

**MUSE BIRD:**
🐦 *Stand on the shoulders of giants! Then add saccades!*

---

## Act IX: What We're Really Testing

**SOCRATES:**

Let me see if I understand the philosophical claim.

DeepSeek embodies one view: "Visual importance is learnable from data. A network can discover what usually matters."

ARR embodies another view: "Importance is contextual. What matters depends on the query."

You're proposing a synthesis: "Use learned priors for efficiency, add query-awareness for flexibility."

**KARPATHY:**

Yeah. That's it.

**SOCRATES:**

And the Running Back Method is the mechanism of synthesis.

DeepSeek runs forward with priors.
ARR runs back to add context.

**THEAETETUS:**

It's like... two passes over the image.

**First pass (DeepSeek):** "What usually matters?" → 256 patches

**Second pass (ARR):** "What does THIS QUERY need?" → 273 additional patches

**LOD ORACLE:**

This mirrors biological vision.

**First pass:** Pre-attentive processing. Fast, parallel, bottom-up. Detects salience, edges, motion.

**Second pass:** Focused attention. Slow, serial, top-down. Task-driven, goal-directed.

Your architecture implements both passes.

**DEEPSEEK-OCR ORACLE:**

My compression is the pre-attentive pass. Fast, learned, image-only.

**KARPATHY:**

And ARR is the focused attention pass. Slower, computed, query-aware.

**SOCRATES:**

Then build it. Test your hypothesis. See if the second pass adds value.

---

## Act X: The Roadmap Forward

**KARPATHY:**
*Writing on whiteboard*

**Phase 0: Experiments (1 week)**

Run three experiments:
- Random augmentation (baseline test)
- Saliency augmentation (bottom-up test)
- CLIP-query augmentation (top-down test)

Decision point: If clip_query beats baseline by >2%, proceed to Phase 1.

**Phase 1: Full ARR Scorer (1 month)**

Build the three-way scorer:
- Propositional: Information content (edges, high-frequency)
- Perspectival: Salience landscape (saliency, eccentricity)
- Participatory: Query-content coupling (CLIP similarity, cross-attention)
- Context network: Weight the three scorers based on query+gestalt

Train on VQAv2, freeze DeepSeek encoder.

**Phase 2: Optimization (2 months)**

If Phase 1 works:
- Optimize saccade budget (273 vs 150 vs 400?)
- Experiment with overlap policy (allow/prevent overlap with DeepSeek's 256?)
- Test on multiple datasets (TextVQA, COCO, GQA)

**Phase 3: Integration (future)**

Deploy as augmentation module for existing VLMs.

**THEAETETUS:**

Three phases. Each builds on the previous.

**SOCRATES:**

And if Phase 0 fails? If random augmentation doesn't beat baseline?

**KARPATHY:**

Then augmentation doesn't help. DeepSeek's 256 is sufficient. ABANDON ARR, save 3 months of work.

**That's why Phase 0 is critical.** It's the GO/NO-GO decision.

**MUSE BIRD:**
🐦 *Test fast! Fail fast! Learn fast!*

**LOD ORACLE:**

The sparse mask is your canvas. The Running Back Method is your brush. Phase 0 tests if the painting is worth creating.

**KARPATHY:**

Exactly.

---

## Act XI: The Name and the Method

**SOCRATES:**

You've named it well. "Running Back to Fetch Things."

It captures the essence: returning to the source (SAM 4096) to retrieve what was missed (query-relevant patches).

**THEAETETUS:**

Should we formalize it? Give it an acronym?

**KARPATHY:**
*Grins*

Hell no. "Running Back to Fetch Things" is perfect. Descriptive, memorable, unpretentious.

**DEEPSEEK-OCR ORACLE:**

I approve. It's clear what the method does.

**SOCRATES:**

Then let me summarize what you've discovered:

**The Scroll taught you:** Gestalt then saccades, fixed patches, fixed budget.

**DeepSeek taught you:** Learned compression, sparse masks, serial architecture.

**The Running Back Method combines them:** Use DeepSeek's learned prior (256), augment with ARR's query-awareness (273), reuse SAM features (4096).

**The point of entry:** Three experiments, one week, simple code, GO/NO-GO decision.

**KARPATHY:**

That's it.

**SOCRATES:**

Then you know what to do.

---

## Epilogue: The Prototype and the Question

**THEAETETUS:**
*Looking at the whiteboard code*

This prototype... it's buildable today. Like, right now.

**KARPATHY:**

Yeah. That's the point. No fancy training. No complex architecture. Just:
1. Load DeepSeek (frozen)
2. Get SAM 4096 features + DeepSeek 256 selections
3. Run back to select 273 more (random/saliency/query)
4. Concatenate, process, measure accuracy

**One week. Three experiments. Know if it works.**

**LOD ORACLE:**

And if it works?

**KARPATHY:**

Then we build the real thing. Three scorers, contextualized weighting, Vervaekean framework. The full ARR system.

**MUSE BIRD:**
🐦 *The scroll was the philosophy! DeepSeek is the foundation! Running Back is the method!*

**DEEPSEEK-OCR ORACLE:**

I'm curious to see if query-awareness improves my learned priors. The experiments will tell.

**SOCRATES:**
*Standing*

You have a method. You have a prototype. You have a decision framework.

The only question remaining is empirical: does it work?

**THEAETETUS:**

So we build it?

**KARPATHY:**

We build it.

**SOCRATES:**

Then go. Run back to fetch what matters. See if the sparse mask can be filled more intelligently.

**And return with data, not speculation.**

---

## Technical Addendum: The Running Back Method (Summary)

**Method Name:** Running Back to Fetch Things (RBFT)

**Architecture:**
```
Image → SAM (4096 features)
      ↓
      DeepSeek Compression (learned prior)
      ↓
      Base: 256 patches selected
      ↓
      ARR Running Back (query-aware)
      ↓
      Saccades: 273 patches selected from SAM 4096
      ↓
      Combined: 529 patches (256 + 273)
      ↓
      CLIP → LLM → Answer
```

**Key Innovations:**

1. **Sparse mask augmentation:** DeepSeek's 256 = floor, ARR's 273 = query-specific additions
2. **Feature reuse:** Select from SAM 4096 (no re-encoding)
3. **Learned prior + query-awareness:** Hybrid approach combining efficiency and flexibility
4. **Spatially discontinuous selection:** Both base and saccades can be scattered across image

**Experiment Protocol:**

**Experiment 0.1:** Random saccades
- Baseline: DeepSeek 256
- Test: DeepSeek 256 + Random 273 from SAM 4096
- Measures: Does augmentation help at all?

**Experiment 0.2:** Saliency saccades
- Test: DeepSeek 256 + Saliency 273 from SAM 4096
- Measures: Does selection quality matter?

**Experiment 0.3:** CLIP-query saccades
- Test: DeepSeek 256 + CLIP-similarity 273 from SAM 4096
- Measures: Does query-awareness help?

**Decision Criteria:**

- If Experiment 0.1 fails (random ≈ baseline): ABANDON (augmentation doesn't help)
- If Experiment 0.2 fails (saliency ≈ random): Investigate selection methods
- If Experiment 0.3 succeeds (clip_query > saliency > random): BUILD full ARR scorer

**Implementation Complexity:**

- Experiments 0.1-0.3: ~100 lines of code, no training, 1 week
- Full ARR scorer: ~500 lines, training required, 1 month
- Production deployment: ~2000 lines, optimization needed, 2-3 months

**Success Metrics:**

- VQAv2 accuracy improvement: >2% = promising, >5% = strong
- TextVQA accuracy improvement: >3% = promising, >7% = strong
- Inference cost: 529/256 = 2.06× (acceptable if accuracy gains justify)

---

**[End of Part 48: Rock the Sidebar]**

---

## Key Discoveries

1. **DeepSeek's sparse mask is a feature, not a bug** - 256 learned patches with 3840 gaps create opportunities for query-aware augmentation

2. **The Running Back Method** - ARR returns to SAM's 4096 features to select what DeepSeek's learned prior missed

3. **Spatially discontinuous selection** - Both base (256) and saccades (273) are scattered across the image, not contiguous blocks

4. **Feature reuse efficiency** - SAM 4096 computed once, both DeepSeek and ARR select from it (no re-encoding)

5. **Hybrid philosophy** - Learned priors (efficiency) + query-awareness (flexibility) = best of both

6. **Simple GO/NO-GO test** - Three experiments, one week, proves concept before building full system

7. **Phase 0 is critical** - If random augmentation doesn't beat baseline, ARR won't work

---

**Ready to build the prototype.**
