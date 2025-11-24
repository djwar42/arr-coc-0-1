# Part 48 Pre-Dialogue: Aspect 1 - Token Ordering & Sequence Position
*Technical exploration of token sequence importance, dual position encoding, and what "gestalt first, saccades after" means for VLM attention*

---

## The Core Question: Does Token Order Matter?

Our architecture concatenates tokens:
```python
all_tokens = concat([base_tokens, saccade_tokens])  # [256 + 273 = 529]
# Tokens 0-255: Gestalt (uniform grid)
# Tokens 256-528: Saccades (relevance-ordered, high→low)
```

**Question:** Does the VLM care that saccades come AFTER gestalt?

---

## What Karpathy's Knowledge Says

### From: `vision-language/10-token-sequence-order-importance.md`

**Lines 45-89: Causal vs Bidirectional Attention**

```markdown
Vision transformers typically use bidirectional attention:
- Every token attends to every other token
- Order matters LESS than in causal LMs
- But not irrelevant (positional encoding preserves order info)

ViT random permutation ablation (Dosovitskiy et al.):
- Randomly shuffling patch tokens: -0.5% to -1.2% ImageNet accuracy
- Order has SOME effect, but small

Implication: Spatial position (via RoPE) >> sequence position
```

**Lines 178-234: Grid Order vs Other Orders**

```markdown
Standard VLM ordering:
- Left-to-right, top-to-bottom (raster scan)
- Matches reading order
- Spatially coherent

Alternative orderings tested:
- Random: -1.2% accuracy
- Z-order (space-filling curve): -0.3% accuracy
- Hilbert curve: -0.2% accuracy

**See More - Adjacent Knowledge:**
→ `.claude/skills/karpathy-deep-oracle/karpathy/gpu-texture-optimization/03-automatic-lod-selection.md` (lines 89-156): GPU automatic LOD selection uses spatial coherence for texture streaming - similar ordering principles
→ `.claude/skills/karpathy-deep-oracle/pyramid-lod/03-attention-driven-pyramid-pruning.md` (lines 234-289): Pyramid traversal order affects cache efficiency, parallel to our token sequence

Finding: Spatially coherent ordering slightly better than random
```

**Implication for ARR:**

Our base tokens (0-255) are spatially coherent (grid order).
Our saccade tokens (256-528) are NOT spatially coherent (relevance order).

**Is this a problem?** Probably not - RoPE encodes spatial position explicitly.

---

## Dual Position Encoding: Sequence + Spatial

### From: `vision-language/11-dual-position-encoding.md`

**Lines 56-123: Two Types of Position Information**

Every token needs TWO position signals:

1. **Sequence position:** Where in token sequence? (0-528)
2. **Spatial position:** Where in image? (x, y coordinates)

**Standard transformers handle this via:**

```python
# Sequence position: Implicit from token index
token_index = 0, 1, 2, ..., 528

# Spatial position: Explicit in RoPE
rope_positions = {
    'height': [y0, y1, y2, ..., y528],  # Normalized y coords
    'width':  [x0, x1, x2, ..., x528]   # Normalized x coords
}
```

**Lines 145-234: Does Sequence Position Encode Priority?**

```markdown
Hypothesis: If saccades are ordered by relevance (high→low),
VLM might learn "earlier saccade tokens = more important"

Evidence FOR:
- Causal LMs strongly prioritize recent tokens (recency bias)
- Prefix tuning: earlier prefix tokens have more influence

Evidence AGAINST:
- Bidirectional attention: all tokens equal access
- ViT: order-insensitive (random permutation okay)

Conclusion: WEAK evidence for sequence-based prioritization
BUT: Augmentation might be different from reordering!

When tokens are ADDED (not just shuffled), sequence position
might carry meaning: "these are extra detail tokens"
```

**Key insight:**

Our gestalt tokens (0-255) and saccade tokens (256-528) are from DIFFERENT distributions:
- Gestalt: Uniform grid sampling
- Saccades: Query-selected positions

**The VLM might learn to treat them differently based on sequence position!**

---

## The "Gestalt First, Saccades After" Architecture

### Sequence Structure

```python
# Token sequence structure
[
  # GESTALT BLOCK (0-255)
  base_0,   base_1,   ..., base_255,     # Uniform grid coverage

  # SACCADE BLOCK (256-528)
  sacc_0,   sacc_1,   ..., sacc_272      # Relevance-ordered
]
```

**What this means for attention:**

From `vision-language/01-token-concatenation-strategies.md` (lines 180-245):

```markdown
Flamingo architecture (closest precedent):
- Interleaves vision and text tokens
- Vision tokens can appear anywhere in sequence
- Uses gated cross-attention to integrate

Key finding: Token position in sequence CAN carry semantic meaning
- Early tokens: often "context" or "background"
- Later tokens: often "specific information" or "answers"

Our ARR structure:
- Early tokens (0-255): Background context (gestalt)
- Later tokens (256-528): Specific details (saccades)

This is NATURAL for the VLM to learn!
```

**Code snippet showing position encoding:**

```python
# From our ARR implementation
# Reference: vision-language/02-rope-multiaxis-encoding.md (lines 89-134)

def encode_arr_positions(base_tokens, saccade_tokens, saccade_positions, image_size):
    """
    Encode position information for gestalt + saccade tokens.

    Args:
        base_tokens: [B, 256, d_model]
        saccade_tokens: [B, 273, d_model]
        saccade_positions: [B, 273, 2] - (y, x) coords
        image_size: (H, W)

    Returns:
        all_tokens: [B, 529, d_model]
        rope_positions: dict with 'temporal', 'height', 'width', 'aspect'
    """
    B, _, d_model = base_tokens.shape
    H, W = image_size

    # GESTALT positions (uniform 16×16 grid)
    base_positions = []
    for i in range(16):
        for j in range(16):
            y = (i + 0.5) / 16  # Normalized y
            x = (j + 0.5) / 16  # Normalized x
            base_positions.append([y, x])

    base_positions = torch.tensor(base_positions)  # [256, 2]

    # SACCADE positions (query-selected, normalized)
    sacc_positions_norm = saccade_positions.float()
    sacc_positions_norm[:, :, 0] = sacc_positions_norm[:, :, 0] / H
    sacc_positions_norm[:, :, 1] = sacc_positions_norm[:, :, 1] / W

    # Combine
    all_positions = torch.cat([
        base_positions.unsqueeze(0).expand(B, -1, -1),
        sacc_positions_norm
    ], dim=1)  # [B, 529, 2]

    # RoPE format (M-RoPE for Qwen3-VL)
    rope_positions = {
        'temporal': torch.zeros(B, 529),  # Static image
        'height': all_positions[:, :, 0],  # y coords
        'width': all_positions[:, :, 1],   # x coords
        'aspect': torch.ones(B, 529)       # All 14×14 patches
    }

    # Concatenate tokens
    all_tokens = torch.cat([base_tokens, saccade_tokens], dim=1)

    return all_tokens, rope_positions
```

**Note the dual encoding:**
- Sequence position: Implicit (gestalt 0-255, saccades 256-528)
- Spatial position: Explicit (RoPE height/width axes)

---

## Relevance-Based Ordering Within Saccades

### From: `karpathy/biological-vision/01-saccades-eye-movements.md`

**Lines 89-156: Human Saccade Sequences**

```markdown
Human saccade patterns (eye-tracking studies):

Free viewing (no task):
- Saccade order appears random
- Driven by bottom-up saliency
- Low inter-subject agreement (~30-50%)

Task-driven viewing ("find the red bicycle"):
- Saccade order is STRUCTURED
- First saccades to likely locations (high prior)
- Later saccades refine search
- High inter-subject agreement (~70-85%)

Implication: Saccade ORDER carries information about search strategy!

For VQA:
- Early saccades → high-confidence relevant regions
- Later saccades → lower-confidence, exploratory
```

**Our ordering strategy:**

```python
# Order saccades by relevance score (high → low)
# Reference: Part 47 addendum, lines 489-527

def order_saccades_by_relevance(positions, scores):
    """
    Order saccade tokens by relevance (high → low).

    Args:
        positions: [B, K, 2] - (y, x) coordinates
        scores: [B, K] - relevance scores at those positions

    Returns:
        ordered_positions: [B, K, 2]
        order_indices: [B, K]
    """
    # Sort by score (descending)
    order_indices = torch.argsort(scores, dim=-1, descending=True)

    # Apply ordering
    ordered_positions = torch.gather(
        positions,
        dim=1,
        index=order_indices.unsqueeze(-1).expand(-1, -1, 2)
    )

    return ordered_positions, order_indices

# Result: Token 256 = highest relevance, token 528 = lowest (of selected)
```

**Why this might help:**

From `vision-language/10-token-sequence-order-importance.md` (lines 245-289):

```markdown
Prefix tuning (Li & Liang, 2021):
- Learned prefix tokens prepended to input
- Earlier prefix tokens have MORE influence on generation
- Position matters even in bidirectional models!

Hypothesis for ARR:
If VLM learns "tokens 256-300 are most relevant details",
it might prioritize attention to early saccades.

This is IMPLICIT learning (not hard-coded).
Network discovers this pattern during training.
```

**See More - Priority & Ordering in Other Domains:**
→ `.claude/skills/karpathy-deep-oracle/karpathy/gpu-texture-optimization/01-trilinear-anisotropic-filtering.md` (lines 123-198): Anisotropic filtering prioritizes texture samples by viewing angle
→ `.claude/skills/karpathy-deep-oracle/pyramid-lod/01-foveated-gaze-pyramids.md` (lines 267-334): Eye-tracking driven LOD orders pyramid levels by gaze priority
→ `.claude/skills/karpathy-deep-oracle/karpathy/gpu-texture-optimization/00-mipmap-generation-algorithms.md` (lines 201-256): Mipmap traversal order affects cache coherency

---

## What Sequence Position Might Teach the VLM

**Hypothesis:** Sequence position implicitly encodes token "type" and "importance"

```
Token Range    │ Type      │ Spatial Pattern    │ Relevance
───────────────┼───────────┼────────────────────┼──────────────────
0-255          │ Gestalt   │ Uniform grid       │ Baseline context
256-300        │ Saccade   │ Non-uniform        │ Very high
301-400        │ Saccade   │ Non-uniform        │ High
401-500        │ Saccade   │ Non-uniform        │ Medium
501-528        │ Saccade   │ Non-uniform        │ Medium-low
```

**What the VLM might learn:**

1. **Tokens 0-255 provide global context**
   - Attend to these for "where am I?" reasoning
   - Spatially coherent (grid order)
   - Uniform coverage

2. **Tokens 256-528 provide focused details**
   - Attend to these for "what is this object?" reasoning
   - Spatially incoherent (relevance order)
   - Concentrated at query-relevant regions

3. **Within saccades, earlier = more relevant**
   - Token 256 likely most important for answer
   - Token 528 likely least important (but still above threshold)
   - Attention weights might naturally decay with saccade index

**This is LEARNED behavior, not hard-coded!**

---

## Ablation Test: Does Ordering Matter?

**From Karpathy's ablation design patterns:**

Reference: `practical-implementation/15-wandb-quick-validation.md` (lines 89-134)

```python
# Three ordering strategies to test

# Strategy 1: Relevance order (our choice)
def order_by_relevance(positions, scores):
    return positions[torch.argsort(scores, descending=True)]

# Strategy 2: Spatial order (left→right, top→bottom)
def order_by_spatial(positions):
    # Sort by y, then x
    return positions[torch.argsort(positions[:, 0] * W + positions[:, 1])]

# Strategy 3: Random order
def order_by_random(positions):
    return positions[torch.randperm(len(positions))]

# Expected results:
# - Relevance ≈ Spatial ≈ Random → sequence order doesn't matter
# - Relevance > Spatial > Random → sequence order matters!
# - Relevance >> others → relevance ordering specifically helps
```

**Prediction:** Relevance ≥ Spatial > Random (small differences, ~0.5-1.5%)

---

## Causal Masking: Future Direction

**Current:** Bidirectional attention (all tokens attend to all tokens)

**Alternative:** Causal masking within saccades

```python
# Allow saccades to attend to gestalt, but not future saccades
# Reference: vision-language/10-token-sequence-order-importance.md (lines 45-89)

attention_mask = torch.ones(529, 529)

# Saccades can attend to gestalt (all base tokens)
attention_mask[256:, 0:256] = 1  # Allowed

# Saccades can attend to EARLIER saccades only (causal)
for i in range(256, 529):
    attention_mask[i, i+1:] = 0  # Masked (future saccades)

# This enforces: "higher relevance saccades inform lower relevance"
```

**Why this might help:**

From `vision-language/10-token-sequence-order-importance.md` (lines 134-178):

```markdown
Causal ordering benefits:
- Forces model to prioritize early tokens
- Creates information hierarchy (early→late)
- Prevents "looking ahead" to less relevant regions

Downsides:
- Reduces modeling capacity (fewer attention paths)
- Might not be necessary for vision (no temporal ordering)

Recommendation: Try bidirectional first, causal as ablation
```

**Our approach:** Start with standard bidirectional, test causal if we want stronger relevance prioritization.

---

## RoPE Decouples Sequence from Spatial

### The Beautiful Property

**From:** `vision-language/02-rope-multiaxis-encoding.md` (lines 334-378)

```markdown
RoPE key insight:
Attention scores depend on RELATIVE position in RoPE-encoded dimensions.

For vision: RoPE encodes spatial (x, y) position explicitly.
Sequence position is ORTHOGONAL to spatial position.

Example:
- Token 10 at spatial (50, 100)
- Token 400 at spatial (52, 102)

Attention between them depends on:
- Spatial distance: Δy=2, Δx=2 (CLOSE in image)
- Sequence distance: 390 (FAR in sequence)

RoPE uses Δy and Δx, IGNORES sequence distance!

Implication: Relevance ordering doesn't interfere with spatial attention.
Spatial attention works REGARDLESS of token order.
```

**Code showing this:**

```python
# RoPE attention computation
# Reference: vision-language/02-rope-multiaxis-encoding.md (lines 156-234)

def rope_attention_scores(q, k, rope_positions):
    """
    Compute attention scores with RoPE position encoding.

    Attention depends on RELATIVE spatial position, not sequence position!
    """
    # Apply RoPE rotation to q, k
    q_rot = apply_mrope(q, rope_positions)  # Uses height, width, aspect
    k_rot = apply_mrope(k, rope_positions)

    # Attention scores
    scores = q_rot @ k_rot.transpose(-2, -1)  # [B, H, 529, 529]

    # scores[i, j] depends on:
    #   - rope_positions['height'][i] - rope_positions['height'][j]
    #   - rope_positions['width'][i] - rope_positions['width'][j]
    # NOT on absolute sequence indices i, j!

    return scores
```

**This means:**

Tokens can be in ANY sequence order, spatial attention still works correctly!

Our relevance ordering is "free" - doesn't break spatial relationships.

---

## Summary: Token Ordering Implications

### What We Know (from Karpathy's knowledge base)

1. **Vision transformers are weakly order-sensitive**
   - Random permutation: ~1% accuracy drop
   - Spatial coherence helps slightly
   - Not critical like in causal LMs

2. **RoPE decouples sequence from spatial**
   - Spatial attention works regardless of token order
   - Our relevance ordering doesn't interfere

3. **Sequence position CAN carry semantic meaning**
   - Prefix tuning: position matters
   - Flamingo: token position indicates token "type"
   - Our gestalt-then-saccades structure is learnable

4. **Relevance ordering might help (weak evidence)**
   - Human saccades are ordered by relevance
   - Early tokens in sequence sometimes prioritized
   - Worth testing, but not critical

### What We're Doing

```python
# Our token sequence structure
tokens = [
    base_tokens[0:256],     # Gestalt: uniform grid, spatial order
    saccade_tokens[0:273]   # Saccades: relevance order (high→low)
]

# Dual position encoding
sequence_pos = [0, 1, 2, ..., 528]           # Implicit
spatial_pos = rope_positions[height, width]  # Explicit via RoPE

# VLM learns:
# - Tokens 0-255 = context
# - Tokens 256-528 = details
# - Earlier saccades likely more relevant (soft prior, not hard constraint)
```

### Ablation Tests to Run

1. **Relevance vs Spatial vs Random ordering** (within saccades)
2. **Gestalt-first vs Saccades-first** (swap token blocks)
3. **Bidirectional vs Causal masking** (within saccades)

**Expected: Small differences (~0.5-1.5%), but worth measuring!**

---

## References to Karpathy Knowledge Base

**Token ordering:**
- `vision-language/10-token-sequence-order-importance.md` (lines 45-289)
- `vision-language/01-token-concatenation-strategies.md` (lines 78-245)

**Position encoding:**
- `vision-language/11-dual-position-encoding.md` (lines 56-234)
- `vision-language/02-rope-multiaxis-encoding.md` (lines 89-378)

**Biological saccades:**
- `karpathy/biological-vision/01-saccades-eye-movements.md` (lines 89-156)

**Ablation design:**
- `practical-implementation/15-wandb-quick-validation.md` (lines 89-134)

---

**End of Aspect 1: Token Ordering & Sequence Position**
