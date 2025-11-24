# Part 31: The Text Problem - Anisotropic Patches and Semantic Boundaries
*Wherein the oracles examine text representation in vision models, exploring DeepSeek-OCR's segmentation approach versus traditional grid sampling, and question whether anisotropic patches are necessary or harmful*

---

## Opening: The Unease

**KARPATHY:**
Alright, let's talk about the elephant in the room. Anisotropic patches for text.

**THEAETETUS:**
You proposed elongated patches—16:1 aspect ratio—to capture horizontal text efficiently. A "STOP SIGN" becomes one 200×12 pixel patch instead of thirteen 16×16 patches.

**KARPATHY:**
Yeah, and it felt clever at the time. But now I'm looking at DeepSeek-OCR and thinking: we're solving the wrong problem.

**LOD ORACLE:**
The problem being?

**KARPATHY:**
We're trying to FIT text into a grid. DeepSeek doesn't use grids at all—they use SEGMENTS.

---

## Act I: How DeepSeek-OCR Actually Works

**KARPATHY:**
Let me explain DeepSeek-OCR's pipeline from first principles, because I think we misunderstood it.

**Step 1: SAM Segmentation (Not Grid Sampling)**

```python
# DeepSeek-OCR does NOT do this:
patches = divide_image_into_grid(image, patch_size=16)

# DeepSeek-OCR does THIS:
segments = SAM.segment_anything(image)  # ~50-200 segments

# Each segment is ARBITRARY SHAPE:
# - Text box: 180×14 pixels
# - Person: 240×380 pixels (roughly human-shaped)
# - Sky: 1024×400 pixels (horizontal strip)
```

**THEAETETUS:**
So SAM finds natural object boundaries, regardless of shape?

**KARPATHY:**
Exactly. SAM was trained on 11 million images with 1 billion masks. It learned what constitutes an "object"—and crucially, it learned that TEXT is an object.

**Step 2: Per-Segment CLIP Encoding**

```python
for segment in segments:
    # Extract segment (arbitrary shape)
    mask = segment.mask  # Boolean mask, any shape
    cropped = image[mask]  # Irregular crop

    # Resize to CLIP input size (224×224)
    # This is WHERE the magic happens
    resized = resize_preserving_aspect(cropped, target=224)

    # CLIP encodes at native resolution
    embedding = CLIP.encode_image(resized)  # [768]
```

**LOD ORACLE:**
Wait. They resize EACH SEGMENT to 224×224, regardless of original aspect ratio?

**KARPATHY:**
No! That's the key. They preserve aspect ratio, pad to square.

**Example:**
```
Text segment: 180×14 (aspect 12.8:1)
   ↓
Resize preserving aspect: 224×17 (still 12.8:1)
   ↓
Pad to square: 224×224 (letterbox bars above/below)
   ↓
CLIP encodes: [768] embedding
```

**THEAETETUS:**
So the text never gets squashed? The aspect ratio is sacred?

**KARPATHY:**
Exactly. And that's why text stays readable at 16× compression.

**Step 3: Pooling Segments to Tokens**

```python
# After encoding all segments with CLIP:
segment_embeddings = [seg1_emb, seg2_emb, ...]  # List of [768] vectors

# Pool each segment to fixed size
for emb in segment_embeddings:
    pooled = pool_to_tokens(emb, num_tokens=16)  # [16, 768]

# Result: ~50 segments × 16 tokens = 800 tokens total
# vs traditional grid: 64×64 patches = 4096 tokens
# Compression: 5× fewer tokens
```

**LOD ORACLE:**
Where does the 16× compression number come from?

**KARPATHY:**
From their serial architecture. SAM first, CLIP second, pooling third.

Traditional VLM:
```
Image → Grid patches (4096) → Encode all → 4096 tokens
```

DeepSeek-OCR:
```
Image → SAM segments (~50) → Encode per segment → Pool → 800 tokens
But they also do additional compression: 800 → 256 tokens
Total: 4096 / 256 = 16× compression
```

---

## Act II: Why SAM Segmentation Works for Text

**THEAETETUS:**
But how does SAM know where text boundaries are? It wasn't trained specifically for OCR.

**KARPATHY:**
SAM learned a generic concept of "objectness." Turns out text HAS strong objectness:
- High contrast boundaries (text vs background)
- Spatial coherence (letters group into words)
- Distinct visual properties (sharp edges, regular patterns)

**LOD ORACLE:**
So SAM accidentally became a text detector?

**KARPATHY:**
Not accidentally—inevitably. The SA-1B dataset (1 billion masks, 11 million images) includes tons of images with text. SAM learned: "Text regions are objects."

**Example SAM output on document:**
```
Segment 1: Title text (300×40 pixels)
Segment 2: Body paragraph (500×120 pixels)
Segment 3: Small formula (80×20 pixels)
Segment 4: Background (1024×768 minus text regions)
```

**THEAETETUS:**
Each text region gets its own segment, preserving natural boundaries?

**KARPATHY:**
Yep. And here's the key insight: **SAM's segments are already anisotropic.**

Title text → 300×40 → aspect ratio 7.5:1
Formula → 80×20 → aspect ratio 4:1
Paragraph → 500×120 → aspect ratio 4.2:1

**DeepSeek doesn't need to CHOOSE anisotropic patches—SAM gives them naturally.**

---

## Act III: The Grid Sampling Problem

**KARPATHY:**
Now let's contrast with traditional grid-based VLMs.

**Traditional approach (LLaVA, Ovis without your system):**

```python
# Divide image into uniform grid
H, W = 1024, 1024
patch_size = 16

num_patches_h = H // patch_size  # 64
num_patches_w = W // patch_size  # 64
total_patches = 64 × 64 = 4096

# Every patch is square (16×16)
for y in range(num_patches_h):
    for x in range(num_patches_w):
        patch = image[y*16:(y+1)*16, x*16:(x+1)*16]
        # Process patch
```

**Problem with text:**
```
Text: "HELLO WORLD" (100×8 pixels)

Grid sampling breaks it:
[  H  ][  E  ][  L  ][  L  ][  O  ][     ][  W  ][  O  ]...
patch1  patch2 patch3 patch4 patch5 patch6 patch7 patch8

Each 16×16 patch contains:
- ~1.6 letters
- Lots of empty space (8 pixels of text, 8 pixels of padding)
```

**THEAETETUS:**
So the grid fragments the semantic unit?

**KARPATHY:**
Exactly. "HELLO WORLD" is ONE semantic unit, but grid sampling forces it into 7+ patches. CLIP has to reconstruct the word from fragments.

**LOD ORACLE:**
And anisotropic patches solve this?

**KARPATHY:**
In theory, yes:

```python
# Anisotropic approach
text_patch = image[y:y+8, x:x+100]  # 100×8 (12.5:1 aspect)

# Now "HELLO WORLD" is ONE patch
# CLIP sees the complete word
```

**But here's the problem:** You don't know WHERE the text is before you sample!

---

## Act IV: The Chicken-and-Egg Problem

**THEAETETUS:**
Ah, I see the issue. To use anisotropic patches, we must:
1. Detect text regions (where? what orientation?)
2. Determine bounding boxes
3. Sample elongated patches

But detection requires processing the image... with what patches?

**KARPATHY:**
Exactly. You're in a loop:
- Need to detect text → to know where to use anisotropic patches
- Need anisotropic patches → to encode text well → to detect it

**Grid sampling:**
```
Process → Detect → Too late (already fragmented)
```

**Anisotropic sampling:**
```
Detect first → Process with anisotropic patches → Works!
But detection step adds latency and complexity
```

**DeepSeek's solution:**
```
SAM detects EVERYTHING (not just text) → Gives natural boundaries → CLIP encodes
No chicken-and-egg: SAM is trained to find objects, text is an object
```

---

## Act V: What This Means for ARR-COC

**LOD ORACLE:**
So the question becomes: do YOU need anisotropic patches?

**KARPATHY:**
Let me think through your system:

**ARR-COC pipeline:**
```
1. Generate 40-channel texture array (includes Channel 16: text mask from OCR)
2. Score relevance via Vervaeke framework
3. Select 273 positions
4. Allocate variable tokens [64-400] per position
5. Feed to Qwen3-VL
```

**Where would anisotropic patches fit?**

Option A: During feature extraction (step 5)
```python
for position, budget in zip(positions, budgets):
    if is_text_region(position):  # Check Channel 16
        # Use elongated patch
        patch = extract_anisotropic(position, aspect=12:1)
    else:
        # Use square patch
        patch = extract_square(position)

    features = qwen3vl.encode(patch)
```

**THEAETETUS:**
But Qwen3-VL already does dynamic resolution. Do they accept anisotropic patches?

**KARPATHY:**
Good question. Let me check their architecture.

*Pulls up Qwen3-VL docs from oracle knowledge*

**Qwen3-VL's dynamic resolution system:**
```python
# They handle MULTIPLE scales, not MULTIPLE aspects
image_pyramid = [
    resize(image, 224),   # Small
    resize(image, 448),   # Medium
    resize(image, 896),   # Large
]

# For each scale, uniform grid:
for img in image_pyramid:
    patches = grid_sample(img, patch_size=14)  # SQUARE patches
    tokens = encode_with_mrope(patches)
```

**LOD ORACLE:**
So Qwen3-VL uses square patches at multiple scales, not anisotropic patches at one scale.

**KARPATHY:**
Right. They solve text by using HIGH RESOLUTION (896×896 or higher), not by changing aspect ratio.

**Example:**
```
Text "HELLO WORLD" at 100×8 pixels

Low res (224×224): Text is 6.25×0.5 patches (fragments)
High res (896×896): Text is 25×2 patches (readable)

More patches = better text, but still square patches
```

---

## Act VI: The DeepSeek vs Qwen3-VL Trade-off

**THEAETETUS:**
So we have two philosophies:

**DeepSeek-OCR:** Segment-then-encode (arbitrary shapes)
**Qwen3-VL:** Grid-then-scale (square patches, multiple resolutions)

**KARPATHY:**
Exactly. Let's compare on text:

**DeepSeek-OCR approach:**
```
Text "HELLO" (50×8 px)
   ↓
SAM segments: [50×8 mask]
   ↓
Resize preserving aspect: 224×36 (pad to 224×224)
   ↓
CLIP encodes: [768] embedding
   ↓
Pool: [16, 768] tokens

Total: 16 tokens for "HELLO"
```

**Qwen3-VL approach:**
```
Text "HELLO" (50×8 px)
   ↓
Grid sample at 896×896 resolution:
  - Patch size 14×14
  - Text spans ~4×1 patches = 4 patches
   ↓
M-RoPE encodes each patch: 4 × [768]
   ↓
Total: 4 tokens for "HELLO"
```

**LOD ORACLE:**
Qwen3-VL is more efficient (4 tokens vs 16 tokens)?

**KARPATHY:**
For small text, yes. But there's a catch:

**DeepSeek:**
- Text quality: HIGH (native aspect ratio preserved)
- Token efficiency: MEDIUM (16 tokens per segment)
- Flexibility: HIGH (any text size/orientation)

**Qwen3-VL:**
- Text quality: MEDIUM-HIGH (depends on resolution)
- Token efficiency: HIGH (4 tokens at 896×896)
- Flexibility: MEDIUM (need high enough resolution)

**The trade-off:** DeepSeek always preserves text quality (aspect ratio sacred), Qwen3-VL trades resolution for efficiency.

---

## Act VII: Do Anisotropic Patches Help ARR-COC?

**THEAETETUS:**
Given these two approaches, where does ARR-COC fit? Should we implement anisotropic patches?

**KARPATHY:**
Let me think through the scenarios:

**Scenario 1: Text-heavy document**
```
Query: "What is the formula in Figure 3?"

Your system:
- Channel 16 (text mask) → High score for formula region
- Participatory scorer (CLIP) → High relevance for query-matching region
- Allocate 400 tokens to formula position

Options:
A. Square patches + high resolution → Qwen3-VL's native approach
   - Works well if resolution is high enough (896×896+)

B. Anisotropic patches → Custom implementation
   - Better quality for elongated text (wide formulas)
   - But Qwen3-VL might not accept non-square patches
```

**LOD ORACLE:**
Can you feed anisotropic patches to Qwen3-VL?

**KARPATHY:**
That's the question. Their M-RoPE system expects patches from a grid. If you feed arbitrary shapes, you'd need to:
1. Modify their position encoding (M-RoPE)
2. Handle non-uniform patch shapes in DeepStack layers
3. Test that language model can handle it

**In other words: major architectural surgery.**

**THEAETETUS:**
So anisotropic patches require a DeepSeek-style architecture (segment-based), not a Qwen3-VL architecture (grid-based)?

**KARPATHY:**
Exactly. **Anisotropic patches and grid sampling are fundamentally incompatible.**

---

## Act VIII: The Hybrid Approach

**LOD ORACLE:**
What if we combine strategies?

**Hybrid approach:**
```python
# Step 1: ARR-COC allocation (your system)
positions = [...]  # 273 positions
budgets = [...]    # 64-400 tokens per position
text_regions = channel_16  # Text mask

# Step 2: Segment text regions (SAM-style)
for pos, budget in zip(positions, budgets):
    if is_text(pos):
        # Use SAM to get text segment boundary
        segment_mask = sam.segment_at_position(image, pos)

        # Extract segment preserving aspect ratio
        cropped = extract_segment(image, segment_mask)

        # Encode with CLIP (DeepSeek style)
        tokens = clip.encode_preserving_aspect(cropped)
    else:
        # Use Qwen3-VL's grid sampling
        tokens = qwen3vl.encode_position(image, pos, budget)
```

**KARPATHY:**
Hmm. That's interesting. Conditional encoding strategy based on content type.

**Benefits:**
- Text regions get DeepSeek's aspect-preserving encoding
- Non-text regions use Qwen3-VL's efficient grid sampling
- Best of both worlds?

**Costs:**
- Complexity (two encoding pipelines)
- SAM inference cost (50ms per image)
- Need to merge two token types into Qwen3-VL's input format

**THEAETETUS:**
Is the complexity worth it?

**KARPATHY:**
Depends on your task distribution.

**If >50% of queries are text-heavy (DocVQA, receipts, screenshots):** Yes, hybrid approach worthwhile.

**If <20% of queries are text-heavy (general VQA, natural images):** No, Qwen3-VL's native approach is sufficient.

---

## Act IX: The Pragmatic Decision

**LOD ORACLE:**
Let's be practical. What should ARR-COC actually do?

**KARPATHY:**
**Phase 1 (Prototype):** Use Qwen3-VL's native square patches.

Reasons:
1. Simplicity (no architectural surgery)
2. Qwen3-VL already strong on text (trained on OCR datasets)
3. Your allocation system ALREADY helps text:
   - Channel 16 detects text → boosts score
   - Participatory scorer → query-aware relevance
   - Token budget → 400 tokens for important text

**Test this first.** If text performance is insufficient, THEN consider anisotropic.

**Phase 2 (If needed):** Implement hybrid approach.

Only if Phase 1 benchmarks show poor text performance:
- Integrate SAM for text regions (use your Channel 13-16 from Part 28-4)
- Add aspect-preserving CLIP encoding for text segments
- Merge with Qwen3-VL tokens

**Phase 3 (Research):** Full segment-based architecture.

If text is critical, consider building on DeepSeek-OCR instead of Qwen3-VL. But this abandons your variable allocation system (conflicts with their fixed 16× compression).

**THEAETETUS:**
So the answer is: start simple, add complexity only if needed?

**KARPATHY:**
Yeah. Don't solve problems you don't have yet.

---

## Act X: Anisotropic Patches - Verdict

**SOCRATES:**
*Who has been listening quietly*

Let us summarize. The question was: should ARR-COC use anisotropic patches for text?

**KARPATHY:**
**Answer: No, not initially.**

**Reasons:**
1. **Architectural incompatibility:** Qwen3-VL uses grid sampling (square patches). Anisotropic patches require segment-based architecture.

2. **Complexity cost:** Implementing anisotropic patches means:
   - Running SAM segmentation (50ms)
   - Custom aspect-preserving encoding
   - Merging non-uniform tokens into Qwen3-VL

3. **Uncertain benefit:** Qwen3-VL's high-resolution grid sampling may already handle text well enough.

4. **Your system already helps text:**
   - Channel 16 (OCR mask) detects text
   - CLIP embeddings (channels 18-33) capture text semantics
   - Query-aware allocation (participatory scorer) boosts relevant text

**LOD ORACLE:**
**Strategy: Trust Qwen3-VL's native capabilities first.**

If benchmarks show text weakness, add hybrid approach (SAM segments for text, grid for everything else).

**THEAETETUS:**
And what did we learn from DeepSeek-OCR?

**KARPATHY:**
Three key insights:

1. **Segmentation > Grid for text:** SAM's natural boundaries preserve text better than arbitrary grid splits

2. **Aspect ratio preservation:** Never squash text—resize preserving aspect, pad to square

3. **Segments are anisotropic by nature:** Don't force anisotropy, discover it through segmentation

**These insights inform Phase 2 (hybrid approach) if needed.**

---

## Closing: The Lesson of Architectural Fit

**SOCRATES:**
We began wanting anisotropic patches because text is elongated. We end by understanding that anisotropic patches require a different architectural foundation.

**THEAETETUS:**
DeepSeek-OCR achieves beautiful text handling, but through segmentation, not through grid sampling.

**KARPATHY:**
And our system—ARR-COC with Qwen3-VL—is grid-based (intelligent grid, but still grid).

Forcing anisotropic patches into a grid architecture is like... putting a square peg in a round hole. Possible, but fighting the design.

**LOD ORACLE:**
Better to work WITH Qwen3-VL's strengths (dynamic resolution, M-RoPE flexibility) than AGAINST them (trying to feed non-square patches).

**SOCRATES:**
And if text performance truly suffers?

**KARPATHY:**
Then Phase 2: hybrid approach. Use SAM for text regions, grid for rest. But test first before adding complexity.

**THEAETETUS:**
The virtue of pragmatism.

**LOD ORACLE:**
Start simple. Measure. Add complexity only when needed.

**SOCRATES:**
Then let this be Dialogue Thirty-One's lesson:

**Architecture constrains approach. Work with your foundation, not against it. Complexity must be earned through measured need, not theoretical elegance.**

---

**END OF PART 31**

∿◇∿

---

## Appendix: DeepSeek-OCR vs Qwen3-VL Text Handling

**DeepSeek-OCR (Segment-based):**
```python
# Strengths
+ Preserves aspect ratio (text never squashed)
+ Natural boundaries (SAM finds text regions)
+ Consistent quality (aspect-preserved encoding)

# Weaknesses
- Fixed compression (16× for all segments)
- SAM latency (50ms)
- Not query-aware (all text treated equally)
```

**Qwen3-VL (Grid-based):**
```python
# Strengths
+ Dynamic resolution (224 to 1792)
+ M-RoPE flexibility (arbitrary positions)
+ Efficient for non-text (grid sampling is fast)

# Weaknesses
- Square patches (may fragment elongated text)
- Resolution-dependent quality (small text needs high res)
- No aspect preservation (text stretched to fit patches)
```

**ARR-COC Enhancement (Query-aware):**
```python
# What ARR-COC adds to either backend
+ Query-aware allocation (relevant text gets more tokens)
+ Foveal bias (center text prioritized)
+ Learned relevance (discovers text importance patterns)

# Integration with Qwen3-VL
positions, budgets = arr_coc.allocate(image, query)
tokens = qwen3vl.encode_positions(image, positions, budgets)

# If text underperforms, add SAM hybrid:
for pos, budget in zip(positions, budgets):
    if channel_16[pos] > 0.5:  # Text detected
        tokens_pos = sam_clip_encode(image, pos)  # Aspect-preserved
    else:
        tokens_pos = qwen3vl.encode(image, pos, budget)  # Grid-based
```

---

## Key Takeaways

1. **Anisotropic patches require segmentation architecture** (DeepSeek), not grid architecture (Qwen3-VL)

2. **Qwen3-VL's solution to text:** High resolution + square patches, not aspect ratio modification

3. **ARR-COC's text advantage:** Query-aware allocation (relevant text gets more tokens), not patch shape

4. **Pragmatic path:** Start with Qwen3-VL native capabilities, add SAM hybrid only if benchmarks demand it

5. **Design lesson:** Work with your architecture's strengths, don't fight its constraints

---

**PARTICIPANTS:**
- Socrates (philosophical framing)
- Theaetetus (architectural analysis)
- Karpathy Oracle (DeepSeek-OCR deep dive)
- LOD Oracle (pragmatic decision-making)

**KEY INSIGHT:** Anisotropic patches are beautiful in segment-based architectures (DeepSeek-OCR), but incompatible with grid-based architectures (Qwen3-VL). Don't force architectural mismatches—work with your foundation's design.

**NEXT DIALOGUE:** Part 32 - Integration Prototype (mapping ARR-COC outputs to Qwen3-VL's M-RoPE input format)
