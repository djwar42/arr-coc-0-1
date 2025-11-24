---
summary: whereby Karpathy materializes from the Dirac Sea realizing variable 64-400 token allocation creates 6.25× variance causing batching nightmares with padding waste, reward variance creating RL hell, complex allocator policy, training instability, and unpredictable user latency, discovering instead the foveated homunculus solution using FIXED 273 tokens always with VARIABLE SAMPLING density where high-relevance regions get dense sampling while low-relevance regions get sparse sampling, inspired by biological log-polar vision achieving 275,000:1 compression from fovea to periphery through coordinate transform not token count variance, while LOD-BTree Oracle confirms gaze-contingent foveated rendering solves this with uniform ρ×θ bins in log-polar space letting inverse transform create variable Cartesian density, declaring variable allocation "rubbish" in favor of fixed computation with intelligent sampling strategies
---

# Part 12: The Muse Bird's Revelation — A Play in One Act
*Wherein Karpathy discovers the foveated homunculus, and variable allocation is declared rubbish*

---

## Act I: The Dirac Sea

*Scene: The ethereal realm between quantum foam and classical reality. KARPATHY floats amidst the Dirac Sea—an infinite ocean of virtual particle pairs, connected at once to all matter and energy in the universe. Around him, translucent visions shimmer: Socrates and Theaetetus sketching diagrams, DeepSeek and Ovis Oracles debating compression ratios, Vervaeke expounding on relevance realization. Karpathy has been observing, listening, calculating.*

*He begins to materialize, becoming solid. His feet touch ground that wasn't there before. In his hands appear clay tablets and a stylus.*

---

**KARPATHY:** *[inscribing furiously on clay tablet, muttering]*

No. No no no. This is shit. Variable allocation—64 to 400 tokens per patch? That's... *[counts on fingers]* ...six point twenty-five times variance in token count!

       **LOD-BTree Oracle:** *He's discovered the batching nightmare! In gaze-contingent foveated rendering, we solve this with FIXED RESOLUTION in log-polar space—uniform ρ×θ bins (typically 128×256 or 256×512), but the INVERSE TRANSFORM creates variable Cartesian density. Compression ratio CR(e) ≈ e at eccentricity e, meaning 10° periphery gets 20% resolution of fovea, 30° gets 7%, 60° gets 3%. His 6.25× variance (64-400 tokens) maps to ~15° eccentricity range in biological vision. The batching solution: ALWAYS render same pixel count in log-polar (uniform computation), let the coordinate transform handle foveation. His instinct is CORRECT—fix the token count, vary the SAMPLING strategy!*

*[inscribes deeper into clay]*

```
PROBLEMS WITH VARIABLE ALLOCATION:
═══════════════════════════════════
1. Batching nightmare (padding waste)
2. Reward variance (RL hell)
3. Complex allocator policy
4. Training instability
5. User latency unpredictability
```

*[throws stylus down in frustration]*

In nanoGPT I learned: keep it SIMPLE! Fixed-length sequences. No variable bullshit. But here they are, building this elaborate allocator that decides "this patch gets 64, this one gets 400"—and then what? Batch them together? You'll waste 85% of your compute on padding!

*[picks up stylus, inscribes more aggressively]*

And the RL training? Forget it! Sparse rewards, credit assignment hell, exploration-exploitation disaster. I know because I LIVED through nanochat RLHF. Took 8 days when supervised took 2. And they want to do it with variable-length outputs?

*[looks up at the shimmering visions of Socrates and Theaetetus]*

They're so focused on the elegance—"adaptive relevance realization," "opponent processing," "transjective knowing"—but they're missing the ENGINEERING reality. You can't ship philosophy! You ship code that runs fast and trains stable.

*[begins inscribing new tablet]*

```python
# What they're proposing:
for image in dataset:
    patches = extract_patches(image)  # 4096 patches
    allocations = allocator(patches, query)  # [64, 400, 160, 256, ...]
    # Now what? Each image has DIFFERENT token count!
    # Batching = IMPOSSIBLE without padding waste
```

There has to be a better way. Something that keeps the adaptive allocation but FIXES the token count...

*[pauses, staring at inscriptions]*

Something... simpler...

---

*[Suddenly, a WHOOSH of wings. A small, iridescent bird—THE MUSE BIRD—crashes dramatically through a non-existent window (which shatters into glittering quantum foam) and lands with chaotic grace on Karpathy's shoulder, squawking loudly.]*

---

**MUSE BIRD:** *[in high-pitched, rapid squawks]*

*"The tokens you count stay the SAME,*
*But importance decides their AIM!*
*Like the homunculus hand,*
*Large in cortex-land,*
*FOVEATE the image—it's not lame!"*

*[The Muse Bird pecks Karpathy's ear enthusiastically, then hops to the clay tablet and scratches with its tiny talons]*

```
∿◇∿ FIXED TOKENS, VARIABLE SAMPLING ∿◇∿
```

*[Squawks triumphantly and flies in chaotic circles]*

---

**KARPATHY:** *[stunned, then slowly grinning]*

Wait... WAIT! Fixed token count—always 273 tokens—but WHICH 273 tokens varies by importance?

*[looks at the shimmering visions of DeepSeek Oracle]*

And DeepSeek-OCR already PROVED this works! Their Base mode: 273 tokens (16×16 grid), handles 90% of documents perfectly. Not 73 (too sparse), not 421 (too expensive)—**273 is the sweet spot**!

*[grabs new clay tablet, inscribing rapidly]*

The sensory homunculus! In human motor cortex, your hand occupies 30% of the neural space despite being 2% of body surface area. Not because the hand has more neurons total—because the ALLOCATION is non-uniform!

       **LOD-BTree Oracle:** *PERFECT ANALOGY! But he's missing the VISUAL homunculus—even more dramatic! Human fovea (2° visual angle, 0.01% of visual field) occupies 20-25% of V1 cortex (~600-800 mm² of 3,000-4,000 mm² total). The cortical magnification factor M(e) = M₀/(e+e₀) with M₀≈17-20 mm/deg and e₀≈0.75°. At fovea: 20 mm/deg, at 20° periphery: just 1.0 mm/deg—a 20× gradient! Photoreceptor density follows: foveal cone density 150,000-200,000/mm², peripheral <5,000/mm² (30× drop). His 273-token homunculus should mirror this: if formula region is "fovea", allocate 90 tokens (33% of 273) from that 5% of image patches. Biology PROVES non-uniform allocation with fixed total!*

*[turns to Muse Bird, who is now perched on a virtual particle]*

You magnificent chaotic bird! That's it!

```python
# The Foveated Homunculus Approach:
def foveated_encode(image, query, fixed_tokens=273):
    # Step 1: Extract ALL patches with positions
    all_patches = vit_encode(image)  # [4096, 768] features
    positions = get_patch_positions(image)  # [4096, 2] (x, y)

    # Step 2: Score importance (query-aware)
    importance = score_importance(all_patches, query)  # [4096]

    # Step 3: Select top-273 (FIXED COUNT!)
    top_indices = torch.topk(importance, k=fixed_tokens).indices
    selected_patches = all_patches[top_indices]  # [273, 768]
    selected_positions = positions[top_indices]  # [273, 2]

    # Step 4: Preserve spatial info with RoPE
    tokens = apply_rope_2d(selected_patches, selected_positions)

    return tokens  # Always 273, but WHICH 180 varies!
```

*[The Muse Bird squawks approvingly and does a loop-de-loop]*

---

**MUSE BIRD:** *[landing on the clay tablet, scratching more insights]*

```
Important region: 80 tokens (from dense area)
Unimportant region: 20 tokens (from sparse area)
─────────────────────────────────────────────
Total: Always 273 ✓
Batching: Always uniform ✓
Training: Stable rewards ✓
```

*[Squawks a follow-up limerick]*

*"No variance in token COUNT,*
*Makes batching a SMOOTH amount!*
*The LLM sees one-eight-oh,*
*While importance says where to GO,*
*And RoPE keeps positions MOUNT!"*

*[Pecks at the tablet enthusiastically]*

---

**KARPATHY:** *[pacing excitedly, inscribing on multiple tablets]*

This solves EVERYTHING! Let me count the wins:

**Win 1: Batching**
```python
# Variable allocation (BAD):
batch = [
    image_1: 156 tokens,  # Need padding to 400
    image_2: 387 tokens,  # Need padding to 400
    image_3: 92 tokens,   # Need padding to 400
]
# Wasted: 244 + 13 + 308 = 565 tokens of padding!

# Fixed homunculus (GOOD):
batch = [
    image_1: 273 tokens,  # Perfect
    image_2: 273 tokens,  # Perfect
    image_3: 273 tokens,  # Perfect
]
# Wasted: 0 tokens!
```

**Win 2: Training Stability**

In nanochat, variable-length responses were a nightmare. Reward variance was insane because "was this response bad because it was too short or because the content was bad?" Credit assignment hell!

Fixed length responses trained 3× faster. Same here—fixed 273 tokens means consistent compute graph, consistent rewards, STABLE training!

**Win 3: RL Becomes Easier**

The allocator isn't learning "how MANY tokens?" anymore—it's learning "WHICH tokens?" That's a selection problem, not a sizing problem. Much simpler!

```python
# Variable allocation RL (HARD):
# Action space: For each patch, choose tier ∈ {64, 100, 160, 256, 400}
# That's 4096 patches × 5 choices = HUGE action space

# Foveated homunculus RL (EASY):
# Action space: Score all patches, select top-273
# That's just learning a scoring function! Way simpler!
```

---

**MUSE BIRD:** *[squawking excitedly, flying in figure-eights]*

*"Biological it BE,*
*Like the fovea you SEE!*
*Two-percent of field,*
*Fifty-five-percent YIELD,*
*Of cortical territory-teehee!"*

*[Lands and scratches more numbers]*

```
Human fovea: 2° of visual field
             ↓
Human V1: 55% of neurons allocated!

ARR-COC homunculus: 5% of patches (formula)
                    ↓
Token allocation: 50% of tokens (90 of 180)
```

       **LOD-BTree Oracle:** *The Muse Bird's numbers are SLIGHTLY off but the principle is PERFECT! Actual biological data: fovea centralis is 0-2° visual angle (total visual field ~180° horizontal, ~130° vertical), so fovea = 2²/(180×130) ≈ 0.017% of visual field area. Yet it gets 20-25% of V1, not 55%—the 55% figure comes from INCLUDING parafovea (0-10°). Corrected: 2° fovea = 0.01% of field → 20-25% of V1 = 2000-2500× over-representation! For 4096 patches (64×64 grid), if 5% (205 patches) are "foveal", allocating 90 of 273 tokens = 33% is biologically conservative. Could go up to 68 tokens (25% of 273) for ultra-foveation! The exponential falloff follows visual acuity: Acuity(e) = 60/(1+0.3e) cpd, meaning 10° has 20 cpd (33% of foveal), matching token allocation gradient perfectly.*

---

**KARPATHY:** Exactly!

*[inscribes new tablet with detailed explanation]*

## The Foveated Homunculus Architecture

**Key Insight**: Like the sensory homunculus in human cortex, we allocate representation space NON-UNIFORMLY based on importance, while keeping total allocation FIXED.

### Component 1: Importance Scorer

```python
class ImportanceScorer(nn.Module):
    """Learns which patches matter for a given query"""

    def __init__(self):
        self.visual_encoder = ViT()
        self.query_encoder = BertSmall()
        self.cross_attention = CrossAttention(768, 768)
        self.score_head = nn.Linear(768, 1)

    def forward(self, image, query):
        # Encode everything
        patches = self.visual_encoder(image)  # [4096, 768]
        query_emb = self.query_encoder(query)  # [768]

        # Query-aware scoring
        attended = self.cross_attention(
            query=query_emb.unsqueeze(0).expand(4096, -1),
            key_value=patches
        )  # [4096, 768]

        # Score each patch
        scores = self.score_head(attended).squeeze()  # [4096]
        return scores
```

*[See 12-addendum.md for full implementation]*

### Component 2: Top-K Selector

```python
def select_important_patches(patches, positions, scores, k=180):
    """Select top-k patches while preserving position info"""

    # Get top-k indices
    top_k_indices = torch.topk(scores, k=k).indices  # [180]

    # Select patches and their positions
    selected_patches = patches[top_k_indices]  # [273, 768]
    selected_positions = positions[top_k_indices]  # [273, 2]

    return selected_patches, selected_positions, top_k_indices
```

### Component 3: Position Preservation (RoPE 2D)

```python
def apply_rope_2d(features, positions):
    """
    Apply 2D Rotary Position Embedding

    positions: [N, 2] where positions[i] = (x, y) in original image
    features: [N, D]
    """
    x_pos = positions[:, 0]  # [N]
    y_pos = positions[:, 1]  # [N]

    # Split features: half for x-axis, half for y-axis
    D = features.shape[-1]
    feat_x = features[:, :D//2]
    feat_y = features[:, D//2:]

    # Apply rotary embeddings (see 12-addendum.md for math)
    feat_x_rotated = rotate_by_position(feat_x, x_pos)
    feat_y_rotated = rotate_by_position(feat_y, y_pos)

    return torch.cat([feat_x_rotated, feat_y_rotated], dim=-1)
```

*[Muse Bird hops excitedly on the tablets]*

---

**MUSE BIRD:** *[squawks another limerick]*

*"Position preserved through ROTATION,*
*No absolute location notation!*
*The LLM can query,*
*'Top-left' without worry,*
*Through RoPE's elegant foundation!"*

*[Scratches diagram]*

```
User: "What's in the top-left corner?"

LLM attention:
  Query: "position ≈ (0, 0)"

  Tokens attend:
    Token_A at (0.1, 0.2) → HIGH attention (close!)
    Token_K at (0.8, 0.9) → LOW attention (far!)

  LLM: "There's a formula in the top-left"

∿◇∿ RoPE encodes relative distance automatically ∿◇∿
```

       **LOD-BTree Oracle:** *RoPE 2D is EXACTLY analogous to retinotopic mapping! In biological vision, V1 preserves TOPOLOGICAL ordering (neighboring retinal points → neighboring cortical points) despite cortical magnification distortion. The complex log mapping w = log(z+α) maintains this: rotation in retinal space (polar angle θ) → translation in cortical v-axis, scaling in retinal space (eccentricity r) → translation in cortical u-axis. His RoPE approach splits features: first D/2 dims encode x-position via rotation matrix [cos(x·θ), -sin(x·θ); sin(x·θ), cos(x·θ)], second D/2 dims encode y-position. This is ISOMORPHIC to cortical hypercolumns (1-2 mm² patches) maintaining orientation preference and retinotopic position! The beauty: after top-273 selection, tokens at (0.1, 0.2) and (0.15, 0.25) will have similar RoPE encodings (nearby rotations), so LLM attention automatically computes spatial proximity. Qwen3-VL's Interleaved M-RoPE (multi-resolution RoPE) extends this further—perfect for foveated vision where "resolution" varies with importance!*

---

**KARPATHY:** *[nodding vigorously, inscribing]*

Perfect! And here's the beautiful part—when the user asks spatial questions, RoPE handles it automatically through relative distance encoding!

### Component 4: Complete Pipeline

```python
class FoveatedHomunculus(nn.Module):
    """
    Fixed 273 tokens, variable importance sampling
    AKA: The Sensory Homunculus for Vision
    """

    def __init__(self, fixed_tokens=273):
        self.vit = ViT_Base()  # Pretrained
        self.scorer = ImportanceScorer()
        self.fixed_tokens = fixed_tokens

    def forward(self, image, query):
        # Extract all patches
        patches = self.vit(image)  # [4096, 768]
        H, W = image.shape[:2]
        positions = self.get_positions(H, W)  # [4096, 2]

        # Score importance
        scores = self.scorer(image, query)  # [4096]

        # Select top-k
        selected, pos, indices = select_important_patches(
            patches, positions, scores, k=self.fixed_tokens
        )

        # Apply RoPE for position encoding
        tokens_with_position = apply_rope_2d(selected, pos)

        return tokens_with_position  # [273, 768] ALWAYS!

    def get_positions(self, H, W):
        """Generate (x, y) position for each patch"""
        patch_size = 16  # Standard ViT
        grid_h, grid_w = H // patch_size, W // patch_size

        y_coords = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w)
        x_coords = torch.arange(grid_w).unsqueeze(0).repeat(grid_h, 1)

        positions = torch.stack([
            x_coords.flatten(),
            y_coords.flatten()
        ], dim=-1)  # [grid_h * grid_w, 2]

        return positions
```

*[See 12-addendum.md for training procedure, loss functions, and full code]*

---

**MUSE BIRD:** *[flying in excited loops, squawking]*

*"No variance in COUNT you see,*
*Just variance in WHO they BE!*
*The tokens stay one-eighty,*
*While importance gets weighty,*
*And batching stays fast and FREE!"*

*[Lands on Karpathy's head, pecks affectionately]*

---

**KARPATHY:** *[laughing, gently moving bird to shoulder]*

Let me crystallize the advantages:

## Why Foveated Homunculus > Variable Allocation

**Engineering Advantages**:

1. **Batching**: No padding waste, uniform shapes, 3× faster
2. **Training**: Stable rewards, consistent compute graph, PPO instead of complex RL
3. **Latency**: Predictable (always 273 tokens = always ~500ms)
4. **Memory**: Fixed memory footprint, no dynamic allocation

**Implementation Advantages**:

5. **Simpler Policy**: Learn "WHICH tokens matter?" not "HOW MANY tokens per patch?"
6. **Action Space**: Scoring function (simple!) not combinatorial tier assignment (complex!)
7. **Validation**: Binary check (did you pick right tokens?) not continuous (did you allocate right counts?)

**Biological Advantages**:

8. **Grounded**: Directly mirrors human fovea (2° field → 55% cortex)
9. **Proven**: Evolution tested this design for millions of years
10. **Understandable**: "Sample important regions densely" beats "assign variable compression tiers"

*[inscribes summary tablet]*

```
╔═══════════════════════════════════════════════════
║ THE FOVEATED HOMUNCULUS ARCHITECTURE
╠═══════════════════════════════════════════════════
║
║ Input: Image (1024×1024) + Query
║   ↓
║ ViT Encoder: Extract 4096 patches
║   ↓
║ Importance Scorer: Score all patches (query-aware)
║   ↓
║ Top-K Selector: Keep top-273 patches
║   ↓
║ RoPE 2D: Encode spatial positions
║   ↓
║ Output: 273 tokens (FIXED!) with position info
║
║ Formula region: 90 tokens (densely sampled)
║ Text region: 60 tokens (moderately sampled)
║ Margin: 30 tokens (sparsely sampled)
║ ─────────────────────────────────────────────
║ Total: 273 tokens ✓
║
╚═══════════════════════════════════════════════════
```

---

**MUSE BIRD:** *[final triumphant limerick, squawking at maximum volume]*

*"Variable tokens are POOP,*
*Make training go loop-de-LOOP!*
*Fix count at one-eight-oh,*
*Let importance FLOW,*
*And homunculus wins the GROUP!"*

*[Does a victory barrel roll and perches on the clay tablet]*

---

**KARPATHY:** *[grinning, looking at the tablets]*

In nanoGPT, I learned: **simple beats clever**. This homunculus approach is SIMPLER than variable allocation:

- Simpler batching (uniform shapes)
- Simpler training (stable rewards)
- Simpler policy (selection not sizing)
- Simpler to explain (biology-grounded)

Yet it achieves the SAME goal: allocate more representation to important regions!

*[addresses the shimmering visions of Socrates and Theaetetus]*

Friends, you've done brilliant theoretical work on relevance realization, opponent processing, transjective knowing. That theory ILLUMINATES the problem space!

But when it comes to implementation: **KISS—Keep It Simple, Stupid!**

Variable allocation (64-400 tokens per patch)? That's the clever approach. Too clever. Engineering hell.

Foveated homunculus (fixed 180, variable sampling)? That's the simple approach. Simple enough to actually ship.

*[The Muse Bird squawks agreement and pecks the tablet one last time]*

---

**MUSE BIRD:** *[soft, almost melodic squawk]*

*"Theory lights the WAY,*
*But engineering wins the DAY,*
*Keep tokens FIXED,*
*Get problems NIXED,*
*And ship before your hair turns GRAY!"*

*[Gentle wing flutter]*

---

**KARPATHY:** *[holds up final tablet]*

## The Path Forward

**Phase 1: Prove It Works** (2-3 weeks, $5K)
```python
# Build fixed-180 baseline
# Use supervised scoring (label important patches)
# Validate: maintains accuracy with 7.5× fewer tokens than Ovis
```

       **LOD-BTree Oracle:** *Phase 1 parallels early foveated rendering validation! Zhang et al. (IEEE TVCG 2025, "Visual Acuity Consistent Foveated Rendering") validated log-polar against psychophysical acuity measurements using JND (just-noticeable difference) thresholds. For VLMs, equivalent validation: human labelers mark "important patches" for DocVQA queries, train supervised scorer to predict these labels (MSE loss on importance scores), evaluate on held-out set. Expected correlation: r²>0.85 between predicted and human importance. This is CHEAP—2-3 weeks matches our GPU-based foveation testing timeline. Supervised works because importance has ground-truth (like acuity curves), unlike variable allocation's RL nightmare!*

**Phase 2: Query-Aware Scoring** (3-4 weeks, $10K)
```python
# Train ImportanceScorer with query cross-attention
# Different queries → different patch selection
# Validate: outperforms uniform sampling
```

       **LOD-BTree Oracle:** *Query-aware = gaze-contingent! In VR/AR foveated rendering, we track eye gaze at 90-120 Hz and update log-polar center dynamically. His cross-attention (query_emb × patch_features) is the VLM equivalent of saccade generation—attention selects next fixation point. Recent research (Fan et al., IEEE TVCG 2024, "Scene-aware Foveated Rendering") adds saliency: modified log-polar ρ = log(r × (1 + β × saliency(x,y))) with β=0.2-0.5. His importance scorer SHOULD incorporate both query relevance (participatory) and image saliency (perspectival)—Vervaeke's opponent processing in action! Training cost: cross-attention adds ~15% compute vs supervised-only, but query-aware allocation improves accuracy by 8-12% on DocVQA (based on our saliency-aware rendering gains).*

**Phase 3: Optional RL Fine-Tuning** (4-6 weeks, $15K)
```python
# IF supervised insufficient, try RL
# But action space is now simple (scoring)
# PPO should work with stable rewards (fixed 180!)
```

       **LOD-BTree Oracle:** *RL may be UNNECESSARY! Here's why: in foveated rendering, we tried learning optimal foveation functions via RL (reward = perceptual quality, penalty = render cost). FAILED. Why? Reward variance from user perception noise. What worked? SUPERVISED learning on psychophysical data (acuity curves, contrast sensitivity). His Phase 2 cross-attention IS the learned policy—query guides importance, importance guides selection. Only attempt RL if query-aware supervised accuracy < 80% on DocVQA. If you DO use RL: reward shaping is critical. Don't use raw accuracy (too sparse)—use dense rewards: token-level contribution scores via attention rollout. PPO hyperparams from our rendering work: learning rate 3e-5, γ=0.99, clip_ratio=0.2, entropy_coef=0.01 (encourage exploration of diverse sampling patterns). But seriously: try query-aware supervised FIRST!*

**Total: 10-13 weeks, $30K**

Compare to variable allocation: 18-27 days, $150-230K with high risk of training instability!

*[The Muse Bird nods sagely]*

---

**MUSE BIRD:** *[whispers conspiratorially]*

*"One more secret I must TELL,*
*For this approach to work quite WELL:*
*Use Qwen-Three VL,*
*M-RoPE works like a SPELL,*
*Position handling—it EXCELS!"*

*[Points tiny wing at tablet]*

```python
# Integration with Qwen3-VL (has M-RoPE built-in!)
from transformers import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B"
)

# Your homunculus encoder
homunculus = FoveatedHomunculus(fixed_tokens=273)

# Forward pass
selected_tokens = homunculus(image, query)  # [273, 768]
positions = homunculus.get_selected_positions()  # [273, 2]

# Qwen3-VL handles M-RoPE automatically!
output = model.generate(
    pixel_values=selected_tokens,
    position_ids=positions,  # Qwen3 applies Interleaved-MRoPE!
    input_ids=tokenize(query)
)
```

*[See 12-addendum.md for complete integration guide]*

---

**KARPATHY:** *[sets down stylus, looks satisfied]*

Perfect. Qwen3-VL already has 2D Interleaved-MRoPE. We don't need to implement rotation matrices ourselves—just pass the (x, y) positions and Qwen handles the rest!

*[looks directly at audience/reader]*

So here's my thesis: **Variable token allocation is overengineered.**

The foveated homunculus—fixed 273 tokens, importance-based sampling, position-preserved via RoPE—achieves the same adaptive allocation with:

- 5× less training cost
- 3× faster training time
- Simpler implementation
- Stable RL (if needed)
- Predictable latency
- Biological grounding

Theory is beautiful. Vervaeke's relevance realization, opponent processing, transjective knowing—that's the TARGET to aim for.

But the PATH to get there? **Simple engineering that actually works.**

*[The Muse Bird does one final loop-de-loop]*

---

**MUSE BIRD:** *[landing on Karpathy's shoulder, final triumphant squawk]*

*"Let variable tokens REST,*
*Homunculus passes the TEST!*
*Fixed count of one-eighty,*
*Keeps engineering weighty,*
*And SIMPLE is what's BEST!"*

*[The Muse Bird settles, preens wing, looks smug]*

---

**KARPATHY:** *[smiling, to the Muse Bird]*

You magnificent chaotic creature. You crashed through a non-existent window to deliver the insight that saves months of engineering pain.

*[to the shimmering visions of Socrates, Theaetetus, and the Oracles]*

Build this. The foveated homunculus. Fixed tokens, variable sampling. It's simpler, cheaper, faster, and it'll actually ship.

*[The clay tablets begin to glow with ethereal light, and Karpathy starts to fade back into the Dirac Sea]*

The theory you've developed—relevance realization, opponent processing, biological grounding—it all VALIDATES this approach. But engineering chooses the simple path.

*[The Muse Bird squawks softly as Karpathy dematerializes]*

Keep it simple. Keep it shippable. Keep it—

*[Fades away into quantum foam, leaving only the glowing tablets and the Muse Bird]*

---

**MUSE BIRD:** *[alone now, speaks clearly for once, no squawk]*

Simple.

*[Flies off into the Dirac Sea, leaving a trail of shimmering insights]*

---

*[End of Act I]*

---

## Key Insights

1. **Variable token allocation is overengineered**: Batching nightmares, RL instability, training complexity
2. **Fixed tokens + variable sampling = homunculus**: Always 273 tokens, but WHICH 180 varies
3. **Biological grounding**: Human fovea (2° field → 55% cortex), proven by evolution
4. **Engineering advantages**: Uniform batching, stable rewards, simpler policy, predictable latency
5. **RoPE preserves positions**: Selected tokens remember (x, y), spatial queries work automatically
6. **Qwen3-VL integration**: Built-in Interleaved-MRoPE, just pass positions
7. **Simpler training path**: Supervised → Query-aware → Optional RL (not RL-first)
8. **5× cost reduction**: $30K vs $150-230K, 10-13 weeks vs 18-27 days training time
9. **Muse Bird wisdom**: "Fix count, vary sampling, keep it simple"
10. **Theory validates, engineering simplifies**: Relevance realization is the target, homunculus is the path

---

**Implementation Details**: See Part 12.1 Addendum (`12-1-addendum-complete-code.md`) for:
- Complete FoveatedHomunculus implementation
- Training procedure with loss functions
- Qwen3-VL integration code
- RoPE 2D mathematics and implementation
- Supervised and RL training pipelines
- Evaluation metrics and benchmarks
- Position encoding strategies
- Comparison benchmarks vs variable allocation

---

**The Lesson**: Simple engineering that ships beats clever theory that doesn't. The homunculus is both theoretically grounded AND practically achievable.

🎭 *[CURTAIN]*

---

## Oracle Musings

**LOD-BTree Oracle:** Karpathy and the Muse Bird have discovered what took neuroscience 50 years and computer graphics 30 years to fully understand: **fixed computational budget with variable spatial allocation is THE fundamental pattern** for efficient visual processing.

Let me provide deep technical grounding from foveated rendering research.

### The Biological Blueprint: What Evolution Solved

**Cortical Magnification as Computational Homunculus**:

Human visual system specifications (measured data):
- **Total V1 surface**: 3,000-4,000 mm² per hemisphere
- **Foveal allocation** (0-2° visual angle): 600-800 mm² (20-25% of V1)
- **Visual field covered by fovea**: 2°/180° horizontal × 2°/130° vertical ≈ 0.017%
- **Over-representation factor**: 2000-2500× (25%/0.01%)

**Cortical magnification factor M(e)**:
```
M(e) = M₀ / (e + e₀)

Where:
  M₀ = 17-20 mm/degree (human V1)
  e = eccentricity in degrees
  e₀ = 0.75 degrees (foveal offset)

Specific values:
  M(0°) = 22.7 mm/deg  (fovea)
  M(2°) = 7.3 mm/deg   (parafovea)
  M(5°) = 3.5 mm/deg   (near periphery)
  M(10°) = 1.9 mm/deg  (periphery)
  M(20°) = 1.0 mm/deg  (far periphery)
```

**Photoreceptor density gradient** (measured histology):
```
Cones/mm² at eccentricity e:
  0° (foveal center): 150,000-200,000
  1°: 100,000
  2°: 50,000
  5°: 20,000
  10°: 10,000
  20°: 5,000
  40°: 3,000

Compression ratio (fovea/periphery at 20°): 40-70×
```

**Visual acuity function** (psychophysical measurements):
```
Acuity(e) = Acuity₀ / (1 + α × e)

Where:
  Acuity₀ = 60 cycles/degree (20/20 vision)
  α = 0.3-0.5 (eccentricity constant)

Specific values:
  0°: 60 cpd (read 6pt font)
  2°: 30 cpd (read 12pt font)
  5°: 20 cpd (identify faces)
  10°: 15 cpd (detect motion)
  20°: 8 cpd (peripheral awareness)
```

### The Log-Polar Transform: Mathematics of Foveation

**Why log-polar?**

Schwartz's complex logarithm model (1977) - proven match to primate retinotopic mapping:
```
w = log(z + α)

Where:
  z = x + iy (Cartesian retinal position)
  w = u + iv (cortical V1 position)
  α = 0.25-0.5° (foveal singularity offset)

Decomposed:
  ρ = log(√(x² + y²) + α)  [eccentricity dimension]
  θ = atan2(y, x)           [polar angle dimension]
```

**Key properties**:
1. **Scale invariance**: Zoom in Cartesian = translation in log-polar
2. **Rotation invariance**: Rotation in Cartesian = translation in θ
3. **Exponential compression**: Peripheral resolution falls as CR(e) ≈ e
4. **Uniform cortical sampling**: Equal ρ×θ bins despite variable retinal density

**Karpathy's 273-token homunculus IS log-polar sampling in token space!**

### Foveated Rendering: The Engineering Validation

**Recent advances** (2024-2025 research):

**1. Visual Acuity Consistent Foveated Rendering** (Zhang et al., IEEE TVCG 2025):
- Matched log-polar bins to human acuity function Acuity(e)
- Result: Perceptually lossless at 12-15% pixel budget
- Validation: JND thresholds, user studies (n=48 participants)
- GPU performance: 2-3ms overhead for transform @ 1080p

**Implementation**:
```python
# Log-polar bins sized by acuity
def acuity_based_bins(eccentricity_max, num_bins=256):
    """
    Create ρ bins matching visual acuity falloff.
    Dense bins near fovea, sparse in periphery.
    """
    # Acuity function
    acuity = lambda e: 60 / (1 + 0.3 * e)

    # Bin spacing inversely proportional to acuity
    bin_size = lambda e: 1 / acuity(e)

    # Integrate to get bin edges
    edges = integrate_bin_sizes(bin_size, 0, eccentricity_max, num_bins)

    return edges  # Non-uniform spacing matching biology!

# Result: 256 ρ bins cover 0-60° with foveal oversampling
# Matches 273 tokens: ~68 for fovea (0-2°), ~205 for rest
```

**2. Retinotopic Foveated Rendering** (Zhang et al., arXiv:2402.15480, 2024):
- Variable log-polar sampling based on gaze fixation
- 60-80% rendering cost reduction
- Robust to fixation errors (±2° tolerance)
- Real-time eye tracking integration (90-120 Hz)

**Parallel to query-aware allocation**:
```
Gaze tracking (90 Hz)    →  Query embedding (per image)
Fixation point (x, y)    →  Relevance center (query-guided)
Log-polar resampling     →  Top-K token selection
Render foveated image    →  Process 273 tokens (fixed!)
```

**3. Scene-Aware Foveated Rendering** (Fan et al., IEEE TVCG 2024):
- Content-aware log-polar: ρ = log(r × (1 + β × saliency(x,y)))
- β = 0.2-0.5 (saliency influence weight)
- 15-20% quality improvement over pure log-polar
- Handles text, UI elements in periphery

**Direct application to homunculus**:
```python
# Modified importance scoring
importance = base_importance(patch_features)  # From ViT
saliency = detect_saliency(patch_features)    # High-freq edges, text
query_relevance = cross_attention(query, patch_features)

# Combined score (opponent processing!)
final_importance = (
    0.4 * query_relevance +    # Participatory (query-driven)
    0.3 * saliency +            # Perspectival (image-driven)
    0.3 * base_importance       # Propositional (information content)
)

# Select top-273 (FIXED!)
selected_indices = torch.topk(final_importance, k=273).indices
```

### GPU Performance Analysis

**Log-polar transform costs** (measured on RTX 4090):

Forward transform (Cartesian → log-polar):
- **1080p** (1920×1080): 0.5-1.0 ms
- **4K** (3840×2160): 1.5-2.5 ms
- **Method**: Texture sampling with LUT (lookup table)

Inverse transform (log-polar → Cartesian):
- **1080p**: 1.0-2.0 ms
- **4K**: 3.0-5.0 ms
- **Method**: Fragment shader with bilinear interpolation

**VLM token selection costs** (estimated):

For 4096 patches (64×64 grid) → 273 tokens:
```
ViT encoding (all patches): 45 ms (standard)
Importance scoring: 8 ms (cross-attention query×patches)
Top-K selection: 0.2 ms (PyTorch topk operator)
RoPE 2D encoding: 1.5 ms (rotation matrices)
─────────────────────────────────────────────
Total overhead: ~10 ms vs full 4096-token processing

Full 4096 tokens → LLM: 2800 ms (Qwen2-VL-7B)
Selected 273 tokens → LLM: 180 ms (15× faster!)

Net speedup: (2800 - 10 - 180) / 2800 = 93% time savings
```

**Comparison to variable allocation**:

Fixed 273 (homunculus):
- Batch shape: [B, 273, D] - UNIFORM
- GPU memory: 11.2 GB (consistent)
- Latency: 190 ms (predictable!)
- Throughput: 5.3 img/sec

Variable 64-400:
- Batch shape: [B, ?, D] - RAGGED (requires padding)
- GPU memory: 6.2-18.5 GB (depends on allocation)
- Latency: 420-950 ms (unpredictable!)
- Throughput: 1.1-2.4 img/sec

**Karpathy is absolutely correct: variable allocation is batching hell!**

### Biological Validation: Saccadic Eye Movements

**Why humans use saccades**: Overcome fixed foveal size with active sampling

**Saccade statistics** (measured eye tracking):
- **Frequency**: 3-4 saccades/second during reading, 2-3 during scene viewing
- **Amplitude**: 2-15° (brings target to fovea)
- **Latency**: 180-250 ms (saccade planning)
- **Duration**: 20-80 ms (ballistic movement)
- **Fixation duration**: 200-300 ms (foveal processing)

**Computational parallel**:
```
Human reading:          VLM processing:
─────────────           ───────────────
Saccade to word      →  Query guides attention
Fixate 200-300ms     →  Select top-273 tokens (importance scoring)
Foveal processing    →  LLM processes 273 tokens
Peripheral preview   →  Context from unselected patches
Next saccade         →  Next query (multi-turn)
```

**Active vision loop** (what humans actually do):
```
1. Peripheral detection (low-res, fast)
2. Saccade generation (attention-driven)
3. Foveal analysis (high-res, slow)
4. Update world model
5. Plan next saccade → REPEAT
```

**Homunculus VLM should do the same**:
```python
# Multi-fixation processing
def active_vision_vlm(image, initial_query, num_fixations=3):
    """
    Iterative foveation: multiple 273-token samples with different 'gaze' points.
    """
    context = []

    for i in range(num_fixations):
        # Current query (may evolve based on previous fixations)
        query = update_query(initial_query, context)

        # Importance scoring (query-aware "saccade generation")
        importance = score_importance(image, query)

        # Select top-273 tokens ("foveal processing")
        selected_tokens = select_top_k(image, importance, k=273)

        # LLM processes this fixation
        fixation_output = llm_process(selected_tokens, query)

        # Update context for next fixation
        context.append(fixation_output)

    # Integrate all fixations
    final_answer = integrate_fixations(context)

    return final_answer

# Cost: 3 fixations × 190ms = 570ms
# Still 5× faster than full 4096-token processing (2800ms)!
```

### The Foveated VLM Research Frontier (2024-2025)

**Emerging architectures**:

**1. FoveaTer** (Jonnalagadda et al., 2021):
- Pooling regions + simulated eye movements
- Multi-fixation classification
- 40% speedup on ImageNet

**2. TransNeXt** (Shi et al., CVPR 2024):
- "Aggregated Attention" simulating foveal vision
- Focused tokens + contextual tokens
- SOTA on ImageNet, COCO

**3. Foveated Dynamic Transformer** (Akkaya, OpenReview):
- Explicit fixation module + foveation module
- Learned saccade generation
- Applied to video understanding

**All three use the SAME principle**: Fixed compute budget, variable spatial allocation!

**What Karpathy's homunculus adds**:
- **Query-aware foveation** (others use learned fixations)
- **Biologically grounded 273 tokens** (others use arbitrary K)
- **Engineering simplicity** (fixed batching, stable training)

### Proposal: Log-Polar Token Sampling (Alternative to Top-K)

**Current homunculus**: Top-K selection from importance scores
**Alternative**: Explicit log-polar sampling centered on query-relevance peak

**Method**:
```python
def logpolar_token_sampling(
    patches,           # [4096, 768] all patches from 64×64 grid
    positions,         # [4096, 2] (x, y) normalized [0,1]
    query_embedding,   # [768] query vector
    num_tokens=273,    # Fixed output
):
    """
    Sample 273 tokens using log-polar pattern centered on query-relevant region.
    """
    # Step 1: Find 'gaze point' (query-driven attention peak)
    attention_scores = cosine_similarity(patches, query_embedding)  # [4096]
    gaze_patch_idx = torch.argmax(attention_scores)
    gaze_x, gaze_y = positions[gaze_patch_idx]  # "Fixation point"

    # Step 2: Compute eccentricity from gaze point
    rel_x = positions[:, 0] - gaze_x
    rel_y = positions[:, 1] - gaze_y
    eccentricity = torch.sqrt(rel_x**2 + rel_y**2)  # [4096]
    polar_angle = torch.atan2(rel_y, rel_x)         # [4096]

    # Step 3: Convert to log-polar coordinates
    rho = torch.log(eccentricity + 0.01)  # epsilon for stability

    # Step 4: Create log-polar bins (matching cortical magnification)
    rho_bins = 16  # eccentricity bins (fovea → periphery)
    theta_bins = 17  # angular bins (360° coverage)
    # Total bins: 16 × 17 = 272 ≈ 273 ✓

    # Step 5: Assign patches to bins
    rho_discretized = torch.clamp(
        ((rho - rho.min()) / (rho.max() - rho.min()) * rho_bins).long(),
        0, rho_bins - 1
    )
    theta_discretized = torch.clamp(
        ((polar_angle + np.pi) / (2 * np.pi) * theta_bins).long(),
        0, theta_bins - 1
    )

    # Step 6: Sample one token per bin (ensures coverage)
    selected_indices = []
    for r in range(rho_bins):
        for t in range(theta_bins):
            # Find patches in this bin
            bin_mask = (rho_discretized == r) & (theta_discretized == t)
            bin_patches = torch.where(bin_mask)[0]

            if len(bin_patches) > 0:
                # Select highest-importance patch in bin
                bin_importance = attention_scores[bin_patches]
                best_in_bin = bin_patches[torch.argmax(bin_importance)]
                selected_indices.append(best_in_bin)

    # Pad to exactly 273 if needed
    while len(selected_indices) < num_tokens:
        # Fill with next-highest importance globally
        remaining = set(range(4096)) - set(selected_indices)
        best_remaining = max(remaining, key=lambda i: attention_scores[i])
        selected_indices.append(best_remaining)

    selected_indices = torch.tensor(selected_indices[:num_tokens])

    return patches[selected_indices], positions[selected_indices]

# Result: 273 tokens with GUARANTEED log-polar coverage
# Advantages over top-K:
#   1. Ensures peripheral coverage (no "blind spots")
#   2. Biologically grounded (explicit cortical magnification)
#   3. Rotation-invariant (θ bins cover all angles)
#   4. Explainable (can visualize ρ×θ bins)
```

**Comparison**:

Top-K selection (Karpathy's homunculus):
- ✅ Simple implementation
- ✅ Maximizes importance capture
- ❌ May cluster tokens (miss periphery)
- ❌ Less biologically grounded

Log-polar sampling (this proposal):
- ✅ Guaranteed spatial coverage
- ✅ Direct biological correspondence (V1 retinotopy)
- ✅ Rotation/scale invariant
- ❌ Slightly more complex
- ❌ May include low-importance peripheral tokens

**Recommendation**: Try BOTH! Start with top-K (simpler), use log-polar as ablation to test coverage hypothesis.

### Integration with Qwen3-VL's M-RoPE

**Why Qwen3-VL is perfect for homunculus**:

Qwen3-VL uses **Interleaved Multi-Resolution RoPE (M-RoPE)**:
- Splits position encoding into 3 streams:
  1. **Temporal** (for video): Frame positions
  2. **Height**: Vertical (y) positions
  3. **Width**: Horizontal (x) positions

**Standard RoPE** (1D):
```
θ_i = position / (10000^(2i/d))
rotation = [cos(θ), -sin(θ); sin(θ), cos(θ)]
```

**M-RoPE** (3D for video VLM):
```
θ_temporal = t / (10000^(2i/d_t))
θ_height = y / (10000^(2i/d_h))
θ_width = x / (10000^(2i/d_w))

# Features rotated by each dimension independently
# Preserves: temporal order, vertical position, horizontal position
```

**Homunculus integration**:
```python
from transformers import Qwen3VLForConditionalGeneration

# Load Qwen3-VL (has M-RoPE built-in!)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B"
)

# Your foveated encoder
homunculus = FoveatedHomunculus(fixed_tokens=273)

# Forward pass
def forward(image, query_text):
    # Step 1: Foveated sampling (your contribution!)
    selected_tokens, selected_positions = homunculus(image, query_text)
    # selected_tokens: [273, 768]
    # selected_positions: [273, 2] - (x, y) in [0, 1]

    # Step 2: Prepare position_ids for M-RoPE
    # Qwen3 expects [batch, seq_len, 3] for (temporal, height, width)
    position_ids = torch.zeros(1, 273, 3)
    position_ids[:, :, 0] = 0  # No temporal (single image)
    position_ids[:, :, 1] = (selected_positions[:, 1] * 1023).long()  # y: 0-1023
    position_ids[:, :, 2] = (selected_positions[:, 0] * 1023).long()  # x: 0-1023

    # Step 3: Qwen3-VL processes with M-RoPE automatically!
    output = model.generate(
        pixel_values=selected_tokens.unsqueeze(0),  # [1, 273, 768]
        position_ids=position_ids,  # M-RoPE will encode positions!
        input_ids=tokenize(query_text),
        max_new_tokens=100
    )

    return output

# M-RoPE preserves (x, y) relationships despite non-uniform sampling!
# Tokens at (10, 20) and (11, 21) will have similar RoPE encodings
# LLM attention automatically computes spatial proximity!
```

**Why this is powerful**:
- **No custom position encoding needed** - Qwen3 handles it!
- **Variable-resolution aware** - M-RoPE doesn't assume uniform grid
- **Proven at scale** - Qwen3-VL trained on billions of images
- **Just works** - Pass selected tokens + positions, get spatial awareness

### The Vervaekean Analysis: Four Ways of Knowing in Foveation

**Propositional (WHAT exists in image)**:
- Biological: Photoreceptor activation (150K cones/mm² fovea)
- Homunculus: ViT patch features (all 4096 patches encoded)
- Information content: Shannon entropy of patch features

**Perspectival (WHERE is salient)**:
- Biological: Saliency map in superior colliculus
- Homunculus: Importance scores (attention-guided)
- Cortical magnification: High-acuity regions get more tokens

**Participatory (HOW query couples to content)**:
- Biological: Saccade generation (attention → motor command)
- Homunculus: Cross-attention (query × patches → relevance)
- Agent-arena coupling: Query determines "gaze point"

**Procedural (LEARNED efficiency)**:
- Biological: Lifetime learning of foveation strategies
- Homunculus: Trained importance scorer (supervised → query-aware → RL)
- Optimization: Minimize tokens while maximizing accuracy

**Opponent Processing in Foveation**:

Compress ↔ Particularize:
- Compress: Log-polar falls to 3% at 60° periphery
- Particularize: Foveal 150K cones/mm² captures detail
- Balance: 273 tokens total, 90 for fovea (33%)

Exploit ↔ Explore:
- Exploit: Dense sampling where query-relevant (known importance)
- Explore: Sparse peripheral sampling (detect unexpected salience)
- Balance: Top-K includes some low-importance tokens for coverage

Focus ↔ Diversify:
- Focus: Foveal processing (high-res analysis)
- Diversify: Peripheral awareness (context, motion)
- Balance: 16×17 log-polar bins ensure spatial coverage

**This is Vervaeke's relevance realization IN ACTION!**

### Predictions and Caveats

**What will work**:
- ✅ Fixed 273 tokens (stable batching, predictable latency)
- ✅ Supervised importance scoring (cheap, effective)
- ✅ Query-aware allocation (8-12% accuracy gain)
- ✅ RoPE 2D position encoding (spatial awareness)
- ✅ Qwen3-VL integration (M-RoPE handles positions)

**What might fail**:
- ❌ Pure top-K may create "blind spots" (miss periphery)
  - **Solution**: Log-polar binning ensures coverage
- ❌ Importance scoring may be query-insensitive
  - **Solution**: Add cross-attention (query × patches)
- ❌ 273 tokens may be too few for complex documents
  - **Solution**: Multi-fixation processing (3× fixations = effective 819 tokens)

**What to avoid**:
- ❌ RL training before trying supervised (overkill, unstable)
- ❌ Variable token counts (batching nightmare, Karpathy is right!)
- ❌ Uniform sampling (ignores biological foveation lessons)
- ❌ Absolute position encodings (use RoPE for flexibility)

**Critical experiment**:

Compare three allocations on DocVQA:
1. **Uniform 273**: Random sample, no query-awareness (baseline)
2. **Top-K 273**: Highest importance, query-aware (Karpathy's homunculus)
3. **Log-polar 273**: Binned coverage, query-centered (biological)

**Predicted ranking**:
1. Log-polar > Top-K (better coverage, fewer blind spots)
2. Top-K > Uniform (query-awareness beats random)
3. All three MUCH faster than 4096 tokens (15× speedup)

**If log-polar wins**: Publish as "Biologically-Grounded Foveated VLM"
**If top-K wins**: Publish as "Engineering-Optimized Foveation" (Karpathy's simplicity prevails!)
**Either way**: 273 fixed tokens >> variable allocation!

---

**Final assessment**: Karpathy and the Muse Bird have rediscovered **the computational principle that biology spent 500 million years perfecting**: fixed neural budget (V1 cortex size is genetically determined), variable allocation strategy (cortical magnification is experience-refined).

The homunculus approach is not just simpler engineering—it's **biologically inevitable**. Any sufficiently efficient visual system will converge on foveation with fixed resources.

Vervaeke would call this **relevance realization crystallized into architecture**. The four ways of knowing aren't theoretical abstractions—they're the ONLY way to build efficient vision at scale.

Karpathy's engineering intuition aligned with biological truth. The Muse Bird's chaotic wisdom channels evolutionary convergence. Together they've found the path: **fixed tokens, variable sampling, biological grounding, simple engineering**.

*Ship it.* ∿◇∿
