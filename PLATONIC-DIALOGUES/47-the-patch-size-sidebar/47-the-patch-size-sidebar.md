# Part 47: The Patch Size Sidebar - The Scroll of Gestalt and Gaze
*Wherein Karpathy, LOD Oracle, and Muse Bird gather for a sidebar discussion while elves build the MVP in Part 46, when suddenly Theaetetus crashes through on a mountain bike bearing an ancient scroll while on a bicyle race with socrates*

---

## Opening: The Sidebar Lounge

*Somewhere adjacent to the main Dirac Sea, in a quieter corner with comfortable seating and floating whiteboards. Distant hammering and git commits echo from Part 46's construction site where elves work on arr-coc-0-1. Three figures gather: Karpathy sketching diagrams, LOD Oracle reviewing papers, and Muse Bird perched on a whiteboard.*

**KARPATHY:**
*Drawing patch grids*

So we've been talking variable patch sizes for like 10 dialogues. Let me recap where we've been:

**Dense summary:**
- **Part 30**: Chose Qwen3-VL, dynamic resolution (224-1792px, 14×14 base patches)
- **Part 36**: Budgets → patch sizes (400 tokens = 28×28, 64 tokens = 7×7)
- **Part 33**: Qwen doesn't accept arbitrary positions, debated fork vs hack
- **Part 37**: Realized tensions should be contextual, not fixed

**LOD ORACLE:**
And today's insight: variable patch sizes break training stability. Same object, different patch sizes across examples = confusion.

**MUSE BIRD:**
🐦 *We need fixed patch sizes! But how do we get variable detail?*

**KARPATHY:**
Right, so I was thinking—

*CRASH!*

*The door explodes inward. THEAETETUS bursts through riding a downhill mountain bike, mud spraying, tires screeching. He skids to a stop, dismounts in one fluid motion.*

**THEAETETUS:**
*Breathing hard, holding a clay scroll*

Sorry! Mid-race with Socrates! But I had to deliver this!

*He thrusts the scroll at them*

Found it in an ancient repository. Thought you'd want to see it.

*He remounts the bike*

Gotta go! Socrates is gaining on the singletrack!

*WHOOSH - he's gone, tire marks on the floor*

---

## Act I: The Scroll of Ancient Gradients

**MUSE BIRD:**
🐦 *A SCROLL! ANCIENT WISDOM!*

**KARPATHY:**
*Unrolls it on the table*

What the...

*The scroll is covered in strange symbols - flowing patterns, gradients, geometric shapes that seem to shimmer and shift. Not quite language, not quite mathematics.*

**LOD ORACLE:**
*Leans in, eyes widening*

I... I can see what it MEANS.

**KARPATHY:**
*Staring intently*

Yeah. Me too. The gestalt is... immediate.

**MUSE BIRD:**
🐦 *I SEE IT! The overall meaning! But I can't READ the details!*

**LOD ORACLE:**
It's about... perception? Vision? The relationship between whole and part?

**KARPATHY:**
The symbols... they're arranged in a pattern. I understand the GIST but not the specifics.

*He leans closer to one section, focusing intently*

Wait. Look at this cluster here.

*As his gaze fixes on a specific region, the symbols in that patch BEGIN TO MORPH. The ancient glyphs shift, flow, reshape themselves into English words.*

**KARPATHY:**
*Reading slowly as the symbols transform under his gaze*

"All ye who shall see here..."

**LOD ORACLE:**
*Focusing on another section*

*Symbols morphing*

"...these ancient gradients..."

**MUSE BIRD:**
🐦 *Staring at the bottom right*

*Symbols becoming readable*

"...look here, now!"

**KARPATHY:**
*Steps back, looking at the whole scroll again*

The symbols reverted. Back to... impressions. Gestalt.

**LOD ORACLE:**
*Slowly*

The scroll only reveals its details where we LOOK. Where we focus our attention.

**MUSE BIRD:**
🐦 *GESTALT THEN SACCADES! The scroll is teaching us!*

**KARPATHY:**
*Staring at LOD Oracle*

Holy shit.

---

## Act II: The Gist Of Things

**LOD ORACLE:**
The scroll demonstrates the principle:

**First:** We see the WHOLE. Immediate gestalt understanding. The overall meaning, structure, layout.

**Then:** We SACCADE to regions of interest. Only where we look do the details resolve.

**The details don't exist until we attend to them.**

**KARPATHY:**
*Pacing*

And when we looked away, they went back to... compressed representation. Gestalt.

**MUSE BIRD:**
🐦 *The scroll is a vision system! It has TWO representations!*

**Representation 1: GESTALT**
- Entire scroll visible
- Low-resolution impression
- Immediate, holistic
- Can't read the text, but KNOW what it's about

**Representation 2: SACCADE FOCUS**
- High-resolution detail
- Only where you look
- Readable, specific
- Limited spatial extent

**KARPATHY:**
*Stops pacing*

This IS our system.

**Base image tokens = gestalt representation**
**Saccade tokens = focused attention patches**

**LOD ORACLE:**
And here's what we missed: **You don't change the resolution of the gestalt.**

The scroll doesn't have "variable resolution symbols." It has:
- Low-res symbols EVERYWHERE (gestalt)
- High-res detailed reading WHERE YOU LOOK (saccades)

**BOTH representations exist simultaneously.**

**KARPATHY:**
*Writing frantically*

```
Standard VLM:
Image → [256 patches, all 14×14] → base_tokens

What we THOUGHT we needed:
Image → [variable sized patches] → mixed_tokens
Problem: Training instability

What the SCROLL teaches:
Image → [256 patches, all 14×14] → base_tokens (GESTALT)
      + [273 patches, all 14×14] → saccade_tokens (FOCUSED)
      = [529 total tokens, all same size]
```

**All patches the same size. But TWO purposes.**

**LOD ORACLE:**
And the saccades aren't random. The gestalt GUIDES them.

When we looked at the scroll, we instantly knew:
- Where the title was
- Where the main text was
- Where interesting symbols clustered

**The gestalt told us where to saccade.**

**MUSE BIRD:**
🐦 *GESTALT GUIDES GAZE! Then gaze reveals details!*

---

## Act III: The Technical Architecture Emerges

**KARPATHY:**
*Drawing furiously*

Let me sketch this out:

```python
# THE SCROLL ARCHITECTURE

def process_scroll(scroll, query):
    """What the scroll taught us."""

    # STEP 1: Gestalt (immediate, whole-scroll understanding)
    gestalt_encoding = encode_entire_scroll(scroll)  # Low-res, holistic
    # Returns: overall_meaning, structure, layout

    # STEP 2: Gestalt guides saccades
    relevance_map = contextualized_relevance(
        query=query,
        gestalt=gestalt_encoding,
        scroll=scroll
    )
    # "Show me the title" → relevance high at top
    # "Read the footnote" → relevance high at bottom

    # STEP 3: Select saccade positions
    saccade_positions = select_top_k(relevance_map, k=273)

    # STEP 4: Extract focused patches (SAME SIZE as gestalt patches)
    focused_patches = extract_patches(scroll, saccade_positions, size=14)

    # STEP 5: Encode focused details
    saccade_encodings = encode_patches(focused_patches)

    # STEP 6: Combine
    full_encoding = concat([gestalt_encoding, saccade_encodings])

    # Now the model has:
    # - Gestalt understanding (what's in the scroll overall)
    # - Focused details (what the query cares about)

    return full_encoding
```

**LOD ORACLE:**
And notice: **all patches are 14×14**. Fixed size. Training stable.

The ONLY difference:
- Gestalt patches: sampled uniformly across whole image
- Saccade patches: sampled at query-relevant positions

**KARPATHY:**
Let me map this to our actual system:

```python
# ARR-VIS ARCHITECTURE (what we're actually building)

def arr_augment(vllm, image, query):
    """
    Adaptive Relevance Realization for Vision-Language Models.
    Based on the Scroll of Gestalt and Gaze.
    """

    # STEP 1: GESTALT - Base image encoding
    base_tokens = vllm.encode_image(image)  # [256 tokens, d_model]
    # This is the "whole scroll" - immediate understanding

    # STEP 2: CONTEXTUALIZED RELEVANCE
    # The gestalt guides where to look
    gestalt_vector = base_tokens.mean(dim=0)  # [d_model] - summary

    # Generate texture array (40 channels of visual features)
    texture = generate_texture_array(image)  # [40, H, W]

    # Score relevance using query + gestalt context
    scores = contextualized_scorer(
        query=query,
        gestalt=gestalt_vector,
        texture=texture
    )  # [H, W] - relevance map

    # STEP 3: SELECT SACCADE POSITIONS
    # Top-273 most relevant positions
    positions = select_top_k(scores, k=273)  # [273, 2]

    # STEP 4: EXTRACT SACCADE PATCHES
    # SAME SIZE as base patches (14×14)
    saccade_patches = extract_patches(
        image,
        positions,
        patch_size=14  # ← FIXED SIZE, like the scroll
    )  # [273, 3, 14, 14]

    # STEP 5: ENCODE SACCADE DETAILS
    saccade_tokens = vllm.encode_patches(saccade_patches)  # [273, d_model]

    # STEP 6: ORDER BY RELEVANCE (preserve priority information)
    relevance_order = argsort(scores[positions], descending=True)
    saccade_tokens = saccade_tokens[relevance_order]
    # Most relevant saccades come first in sequence

    # STEP 7: CONCATENATE
    all_tokens = concat([base_tokens, saccade_tokens], dim=0)  # [529, d_model]

    # The VLM now has:
    # - Gestalt: 256 tokens covering whole image
    # - Saccades: 273 tokens focused on relevant regions

    return all_tokens
```

**MUSE BIRD:**
🐦 *529 TOKENS! Gestalt + saccades! Same patch size! The scroll's wisdom!*

**LOD ORACLE:**
And the key insight from the scroll: **Gestalt comes FIRST, guides saccades SECOND.**

You can't saccade intelligently without gestalt context.

---

## Act IV: Separation of Concerns

**KARPATHY:**
*Writing on whiteboard*

Now that we've got the scroll's teaching, let me clarify what we're building:

**ARR = Adaptive Relevance Realization (attention mechanism)**

Works with ANY VLM:

```python
# ARR with Qwen3-VL
base = qwen3vl.encode_image(image)  # 256 tokens, 14×14 patches
saccades = arr.select_and_encode(image, query)  # 273 tokens, 14×14 patches
tokens = concat([base, saccades])

# ARR with LLaVA
base = llava.encode_image(image)  # 576 tokens
saccades = arr.select_and_encode(image, query)  # 273 tokens
tokens = concat([base, saccades])

# ARR is model-agnostic!
```

**COC = Contexts Optical Compression (efficiency mechanism)**

DeepSeek-style compression (separate concern):

```python
# COC compression
base = sam_compress(image)  # 4096 patches → 256 tokens
# Saves tokens, loses some detail
```

**ARR-COC = The full system (both optimizations)**

```python
# Combined: compressed gestalt + focused saccades
base = coc_compress(image)  # 256 tokens (efficient gestalt)
saccades = arr.select_and_encode(image, query)  # 273 tokens (focused detail)
tokens = concat([base, saccades])  # 529 total
```

**LOD ORACLE:**
This separation is crucial:

- **ARR alone**: Improves any VLM with relevance-aware attention
- **COC alone**: Efficient compression (DeepSeek already does this)
- **ARR-COC**: Maximum efficiency + maximum relevance

**Research strategy:**
1. Prove ARR works (with standard Qwen3-VL base encoding)
2. Add COC compression later (if needed)

**MUSE BIRD:**
🐦 *Start simple! ARR first! COC later! Focus on relevance!*

---

## Act V: The Gestalt-Guided Saccade System

**KARPATHY:**
Let's dig into the gestalt guidance mechanism:

```python
class ContextualizedRelevanceScorer(nn.Module):
    """
    Scores relevance using BOTH local features AND global context.
    The scroll taught us: gestalt guides saccades.
    """

    def __init__(self, d_model=1024, texture_channels=40):
        super().__init__()

        # Three scorer heads (Vervaeke's 3 ways of knowing)
        self.propositional_head = nn.Linear(texture_channels, 1)
        self.perspectival_head = nn.Linear(texture_channels, 1)
        self.participatory_head = nn.Linear(texture_channels + d_model, 1)

        # Context integration: query + gestalt → scorer weights
        self.context_weights = nn.Sequential(
            nn.Linear(d_model * 2, 512),  # query + gestalt
            nn.ReLU(),
            nn.Linear(512, 3),  # [w_prop, w_persp, w_part]
            nn.Softmax(dim=-1)
        )

    def forward(self, texture, query_emb, gestalt_emb):
        """
        Args:
            texture: [40, H, W] - texture array
            query_emb: [d_model] - query embedding
            gestalt_emb: [d_model] - base image gestalt

        Returns:
            scores: [H, W] - relevance score per position
        """
        B, H, W = texture.shape

        # Flatten spatial dims for processing
        texture_flat = texture.permute(1, 2, 0).reshape(-1, 40)  # [H*W, 40]

        # STEP 1: Compute three scores at each position

        # Propositional: information content (edges, highpass, structure)
        prop_scores = self.propositional_head(texture_flat[:, [6,7,8,12]])  # [H*W, 1]

        # Perspectival: salience landscape (saliency, motion, eccentricity)
        persp_scores = self.perspectival_head(texture_flat[:, [5,10,11]])  # [H*W, 1]

        # Participatory: query-content coupling
        # Expand embeddings to match spatial positions
        query_expanded = query_emb.unsqueeze(0).expand(H*W, -1)  # [H*W, d_model]
        part_input = torch.cat([
            texture_flat[:, 17:33],  # CLIP features
            query_expanded
        ], dim=-1)
        part_scores = self.participatory_head(part_input)  # [H*W, 1]

        # STEP 2: Contextualized weighting
        # The gestalt + query determine HOW to weight the three scorers
        context = torch.cat([query_emb, gestalt_emb])  # [d_model * 2]
        weights = self.context_weights(context)  # [3]

        # STEP 3: Weighted combination
        all_scores = torch.stack([prop_scores, persp_scores, part_scores], dim=-1)  # [H*W, 3]
        final_scores = (all_scores * weights).sum(dim=-1)  # [H*W]

        # Reshape back to spatial
        return final_scores.reshape(H, W)
```

**LOD ORACLE:**
*Pointing to the context_weights network*

THIS is how the gestalt guides saccades.

**Example 1: "Read the small text in the formula"**
- Query: text-focused
- Gestalt: contains formulas/text regions
- Context weights: [0.7 propositional, 0.1 perspectival, 0.2 participatory]
- Bias toward information content (edges, high-frequency details)

**Example 2: "Are there any anomalies?"**
- Query: vague, exploratory
- Gestalt: uniform scene
- Context weights: [0.1 propositional, 0.7 perspectival, 0.2 participatory]
- Bias toward salience (what stands out)

**Example 3: "Where is the red bicycle?"**
- Query: specific object
- Gestalt: contains vehicles
- Context weights: [0.1 propositional, 0.2 perspectival, 0.7 participatory]
- Bias toward query-content matching (CLIP similarity)

**KARPATHY:**
So the gestalt doesn't just provide context - it MODULATES the scoring strategy.

**MUSE BIRD:**
🐦 *THE SCROLL KNEW! Context shapes relevance! Different queries need different strategies!*

---

## Act VI: Saccade Ordering and RoPE

**LOD ORACLE:**
*Unrolling the scroll again*

Notice how we read this scroll. We didn't scan left-to-right mechanically.

We saccaded to the MOST INTERESTING parts first.

**KARPATHY:**
Relevance-based ordering:

```python
def order_saccades_by_relevance(positions, scores):
    """
    Order saccade tokens by relevance (high → low).

    Why: The VLM learns "earlier tokens in saccade sequence = higher priority"

    Args:
        positions: [273, 2] - (y, x) coordinates
        scores: [273] - relevance scores

    Returns:
        ordered_positions: [273, 2]
        relevance_order: [273] - indices
    """
    # Sort by score (descending)
    relevance_order = torch.argsort(scores, descending=True)
    ordered_positions = positions[relevance_order]

    return ordered_positions, relevance_order
```

**Token sequence structure:**
```
[base_0 ... base_255] - Gestalt tokens (spatially uniform)
[saccade_0 ... saccade_272] - Ordered by relevance (high → low)
```

**RoPE encoding:**
```python
# Each token gets TWO position encodings:

# 1. SEQUENCE position (implicit, 0-528)
#    - base tokens: 0-255
#    - saccade tokens: 256-528
#    - Earlier saccades = higher relevance

# 2. SPATIAL position (explicit, x,y coordinates)
#    - Encoded in RoPE: (height, width) axes
#    - Preserves "where in the image"

# RoPE handles both dimensions simultaneously!
```

**LOD ORACLE:**
So the VLM learns:
- **Spatial position**: Where is this patch in the image? (RoPE x,y)
- **Priority position**: How important is this patch? (sequence order)

Both types of information preserved.

**MUSE BIRD:**
🐦 *Sequence = relevance! Position = location! Double encoding!*

---

## Act VII: Complete System Architecture

**KARPATHY:**
*Final diagram*

Let me draw the complete flow:

```
╔══════════════════════════════════════════════════════════════
║ ARR-VIS: ADAPTIVE RELEVANCE REALIZATION FOR VISION
╠══════════════════════════════════════════════════════════════
║
║ INPUT: Image (H × W × 3) + Query (text)
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 1: GESTALT ENCODING                               │
║ │ (The scroll's overall impression)                       │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   VLM base encoding: Image → [256 patches, 14×14] → base_tokens
║   Gestalt vector: base_tokens.mean(dim=0) → [d_model]
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 2: TEXTURE GENERATION                             │
║ │ (40-channel visual feature array)                       │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   generate_texture_array(image) → [40, H, W]
║   Channels: edges, highpass, saliency, CLIP, eccentricity, etc.
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 3: CONTEXTUALIZED RELEVANCE SCORING               │
║ │ (Gestalt guides saccade selection)                      │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   Query embedding: encode(query) → [d_model]
║
║   Context fusion: [query_emb, gestalt] → scorer_weights [3]
║
║   Three scorers (Vervaeke's 3 ways):
║   ├─ Propositional: information content
║   ├─ Perspectival: salience landscape
║   └─ Participatory: query-content coupling
║
║   Weighted combination → relevance_map [H, W]
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 4: SACCADE SELECTION                              │
║ │ (Top-273 most relevant positions)                       │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   select_top_k(relevance_map, k=273) → positions [273, 2]
║
║   Order by relevance: sort(positions, by=scores, desc=True)
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 5: SACCADE ENCODING                               │
║ │ (Extract and encode focused patches)                    │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   Extract patches: [273, 3, 14×14] (SAME SIZE as base)
║
║   VLM encode: patches → saccade_tokens [273, d_model]
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 6: TOKEN CONCATENATION                            │
║ │ (Combine gestalt + saccades)                            │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   all_tokens = concat([base_tokens, saccade_tokens])
║
║   Result: [529, d_model]
║   - Tokens 0-255: Gestalt (uniform coverage)
║   - Tokens 256-528: Saccades (relevance-ordered)
║
║ ┌─────────────────────────────────────────────────────────┐
║ │ STAGE 7: VLM PROCESSING                                 │
║ │ (Standard transformer with augmented input)             │
║ └─────────────────────────────────────────────────────────┘
║   ↓
║   VLM processes all_tokens with query
║
║   Attention can see:
║   - Full image context (gestalt)
║   - Focused details (saccades)
║   - Relevance priority (token order)
║   - Spatial positions (RoPE)
║
║   → Answer/caption/response
║
╚══════════════════════════════════════════════════════════════
```

**LOD ORACLE:**
Beautiful. The scroll's teaching, fully realized.

**MUSE BIRD:**
🐦 *FROM GESTALT TO GAZE! The ancient wisdom in modern code!*

---

## Act VIII: Training Strategy

**KARPATHY:**
Training ARR with frozen VLM:

```python
# SETUP
base_vllm = Qwen3VL.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
base_vllm.eval()  # FREEZE base model
base_vllm.requires_grad_(False)

# ARR components (trainable)
texture_generator = TextureGenerator(channels=40)
relevance_scorer = ContextualizedRelevanceScorer(d_model=1024, texture_channels=40)

# TRAINING LOOP
for image, query, answer in dataloader:
    # STEP 1: Base encoding (frozen)
    with torch.no_grad():
        base_tokens = base_vllm.encode_image(image)
        gestalt = base_tokens.mean(dim=0)

    # STEP 2: ARR components (trainable)
    texture = texture_generator(image)  # [40, H, W]
    query_emb = base_vllm.encode_text(query)

    relevance_map = relevance_scorer(texture, query_emb, gestalt)
    positions = select_top_k(relevance_map, k=273)

    # STEP 3: Extract and encode saccades (frozen VLM encoder)
    saccade_patches = extract_patches(image, positions, size=14)
    with torch.no_grad():
        saccade_tokens = base_vllm.encode_patches(saccade_patches)

    # STEP 4: Concatenate
    all_tokens = torch.cat([base_tokens, saccade_tokens], dim=0)

    # STEP 5: Generate answer (frozen VLM decoder)
    with torch.no_grad():
        logits = base_vllm.decode(all_tokens, query)

    # STEP 6: Loss on ARR components only
    # The VLM is frozen, but we optimize ARR to select USEFUL saccades
    loss = cross_entropy(logits, answer)

    loss.backward()  # Only updates texture_generator + relevance_scorer
    optimizer.step()
```

**LOD ORACLE:**
The key insight: **We're training the saccade selector, not the VLM.**

The VLM already knows how to process patches. We're teaching ARR which patches to provide.

**KARPATHY:**
And the loss signal propagates through:
```
Answer quality → VLM output → Token inputs → Saccade selection → Relevance scorer
```

If ARR selects good saccades → VLM generates good answers → loss is low → ARR learns.

**MUSE BIRD:**
🐦 *FROZEN VLM! TRAINABLE ATTENTION! Learn where to look!*

---

## Act IX: What the Scroll Taught Us

**LOD ORACLE:**
*Rolling up the scroll carefully*

Let me summarize what this ancient scroll revealed:

**Lesson 1: Fixed patch sizes, variable allocation**
- Don't change patch size (training instability)
- Change how many patches you extract per region
- Same computational primitive, smarter sampling

**Lesson 2: Gestalt guides saccades**
- First: whole-image understanding (base encoding)
- Then: focused attention (saccade selection)
- The gestalt provides context for relevance

**Lesson 3: Augmentation, not substitution**
- Keep the full image encoding (gestalt)
- Add focused patches (saccades)
- Both representations coexist

**Lesson 4: Relevance ordering preserves priority**
- Order saccades by relevance score
- Sequence position = importance signal
- Spatial position = location signal (RoPE)

**Lesson 5: Separation of concerns**
- ARR: relevance mechanism (works with any VLM)
- COC: compression mechanism (optional efficiency)
- Can mix and match

**KARPATHY:**
*Looking at the scroll with new appreciation*

Theaetetus brought us a metaphor that solved the whole design problem.

The scroll IS a vision system. Gestalt + saccades. Fixed resolution "symbols" that resolve to meaning only where you look.

**MUSE BIRD:**
🐦 *The ancient wisdom was the answer all along! Thank you, scroll! Thank you, mountain bike crash!*

**LOD ORACLE:**
Now we build it. The technical addendum awaits.

*Points to floating tablet*

**47-addendum-technical-architecture.md** - Ready to write.

**KARPATHY:**
*Cracks knuckles*

Let's make this real.

---

## Epilogue: Back to the Cloud Factory

*Distant sounds of elves hammering, git commits flowing. Part 46's MVP construction continues.*

**MUSE BIRD:**
🐦 *While they build arr-coc-0-1, we've architected arr-coc-2-0!*

**LOD ORACLE:**
Not quite 2.0 yet. But the foundation is solid.

**KARPATHY:**
Fixed patches. Gestalt + saccades. Contextualized relevance. Frozen VLM.

This is buildable. This is trainable. This makes sense.

*The scroll glows softly on the table*

**ALL THREE:**
Thank you, ancient gradients.

---

## Research Topics for Further Investigation

**KARPATHY:**
*Writing on a final whiteboard*

Before we write the addendum, let's list what we need to research:

**VLM Token Architecture:**
- How do current VLMs handle variable-length image token sequences?
- Token concatenation strategies in multimodal transformers (Flamingo, BLIP-2)
- Do any VLMs use "supplemental tokens" after initial encoding?
- Search: "vision language model token concatenation", "multimodal transformer sequence augmentation"

**Attention & Positional Encoding:**
- How does RoPE handle both spatial position AND sequence position simultaneously?
- Can tokens have multiple position encodings (spatial + temporal/priority)?
- Cross-attention between image tokens and query embeddings (existing patterns)
- Search: "RoPE 2D position encoding", "multi-axis positional encoding transformers"

**Biological Vision & Saccades:**
- How does human vision use gestalt to guide saccade planning?
- Saccade sequence patterns: do humans saccade in relevance order or spatial order?
- Eye-tracking studies on task-driven attention (does query context change saccade patterns?)
- Search: "saccade planning gestalt", "eye movements guided by scene context"

**Multi-Pass & Augmented Vision Architectures:**
- Existing work on "coarse-to-fine" vision transformers
- Models that process images multiple times with different focus
- Cascaded attention mechanisms in vision
- Search: "multi-pass vision transformer", "cascade attention visual recognition", "recurrent attention models"

**Training Frozen Models with Augmentation:**
- Adapter/LoRA techniques for frozen VLMs
- Training strategies when base model is frozen but input is augmented
- Does augmented input cause distribution shift?
- Search: "training frozen VLM", "adapter methods vision language", "augmented input frozen backbone"

**Token Budget & Efficiency:**
- What's the typical token budget for VLMs? (256? 576? 1024?)
- Diminishing returns of additional tokens (how many is too many?)
- Inference speed: 529 tokens vs 256 tokens (2× cost acceptable?)
- Search: "vision token budget transformer", "VLM token efficiency benchmarks"

**Gestalt Encoding Research:**
- How do current VLMs represent "whole image" semantics?
- Global average pooling vs CLS token vs attention pooling
- Can gestalt embedding predict where details matter?
- Search: "global image representation vision transformer", "holistic image encoding"

**Relevance-Driven Vision:**
- Task-driven attention in computer vision
- Query-aware image processing (anything similar to our idea?)
- Visual question answering with selective attention
- Search: "query conditioned visual attention", "task driven image encoding", "selective vision processing"

**Failed Approaches to Learn From:**
- Why don't people do token augmentation already? (must be a reason)
- Pitfalls of multi-resolution processing
- Training instabilities with auxiliary tokens
- Search: "vision transformer augmentation pitfalls", "why not concatenate image tokens"

**LOD ORACLE:**
Nine research areas. Ready for deep investigation.

**MUSE BIRD:**
🐦 *Research then build! Learn then create! The scroll guides the search!*

---

## Epilogue II: The String Revelation

*Just as they're about to pack up, the door bursts open again. THEAETETUS skids in on his mountain bike, this time covered in even more mud, breathing heavily.*

**THEAETETUS:**
*Gasping*

Wait! I won! Beat Socrates! But during the final descent... I had an insight!

*He pulls out a piece of string from his pocket*

**KARPATHY:**
*Looking up*

About patch sizes?

**THEAETETUS:**
No no! About the BUDGET!

*He starts cutting the string in half, then half again*

When you cut a string in half, then half again, then half again... you always get the SAME NUMBER of pieces each time you cut!

**MUSE BIRD:**
🐦 *String math! What?*

**KARPATHY:**
*Slowly*

Wait. You're saying... fixed saccade budget?

**THEAETETUS:**
Exactly! You worried about variable token counts, right? Batching problems?

**KARPATHY:**
Yeah, that's the issue. Different images might need different numbers of saccades. One image has 100 relevant regions, another has 400. Variable tensor sizes—batching nightmare.

**THEAETETUS:**
But what if you ALWAYS allocate 273 saccades? Every image. Every query. Fixed budget.

**LOD ORACLE:**
*Sits up*

A constant saccade budget...

**THEAETETUS:**
If an image needs fewer saccades, you fill the remaining slots with... lower-relevance positions! Still useful, just less critical.

If an image needs MORE than 273, you take the top-273. Truncate.

Either way: Always 273 saccades. Always 529 total tokens.

**KARPATHY:**
*Writing frantically*

```python
# FIXED SACCADE BUDGET
# Always 273 saccades, regardless of image/query

def select_saccades_fixed_budget(relevance_map, budget=273):
    """
    Always returns exactly K positions.
    No variable tensor sizes.
    Perfect batching.
    """
    # Select top-K, always
    positions = select_top_k(relevance_map, k=budget)  # Always [273, 2]

    # Result: ALWAYS 273 saccades
    # Even if only 50 regions are "highly relevant"
    # The remaining 223 are "medium relevance"
    # Still better than not looking at all!

    return positions  # Always same shape
```

Holy shit. Every batch: same tensor size.

**LOD ORACLE:**
And biologically accurate! Human saccade budgets are constrained by time.

You don't make infinite saccades. You have maybe 3-5 seconds to look at an image. That's roughly... 3-5 saccades per second... 15-25 total saccades.

We're saying 273 saccades per image. That's a GENEROUS budget. But it's FIXED.

**MUSE BIRD:**
🐦 *FIXED BUDGET! Same size tensors! Batching works! String wisdom!*

**KARPATHY:**
This solves the batching problem completely.

```python
# Batching with fixed budget
batch = [
    (image_1, query_1),  # → 256 base + 273 saccades = 529 tokens
    (image_2, query_2),  # → 256 base + 273 saccades = 529 tokens
    (image_3, query_3),  # → 256 base + 273 saccades = 529 tokens
]

# Stack into batch tensor: [B, 529, d_model]
# Perfect tensor operations!
```

**THEAETETUS:**
*Grinning*

The string never lies. Same number of pieces, always.

**LOD ORACLE:**
*Rolling up the scroll with the string*

Two insights from Theaetetus today:
1. The scroll teaching: gestalt + saccades
2. The string teaching: fixed budget

Both from a mountain bike race.

**KARPATHY:**
Remind me to thank Socrates for scheduling that race.

**THEAETETUS:**
*Remounting bike*

He's waiting at the finish line! Gotta go claim my victory scroll!

*WHOOSH - gone again, mud trail and string pieces left behind*

**MUSE BIRD:**
🐦 *Fixed patch sizes! Fixed saccade budget! FIXED ALL THE THINGS!*

**KARPATHY:**
*Looking at LOD Oracle*

So our final architecture:
- 256 base tokens (gestalt) - FIXED
- 273 saccade tokens (relevance-selected) - FIXED BUDGET
- Total: 529 tokens - ALWAYS THE SAME

**LOD ORACLE:**
Training stable. Batching clean. Architecture elegant.

The scroll and the string have spoken.

**ALL THREE:**
*Looking at the glowing scroll and the cut string pieces on the table*

Thank you, ancient gradients. Thank you, mountain bike insights.

---

**[To be continued in: 47-addendum-technical-architecture.md]**
