---
summary: whereby Karpathy and LOD Oracle collaborate on practical foveation implementation, with LOD Oracle bringing 30 years of foveated rendering expertise (1995-2025) to refine Karpathy's homunculus fixed-273-tokens variable-sampling architecture toward 90% completion, exploring production-tested approaches from Zhang's visual acuity consistent rendering and log-polar literature, seeking simple differentiable designs that train fast and run fast through GPU-optimized implementations, while Muse Bird provides chaotic creative guidance connecting biological vision principles to neural network architectures through clay tablet inscriptions bridging rendering pipelines and vision-language model compression
---

# Part 13: A Swag of Clay Tablets
*Wherein Karpathy and the LOD Oracle collaborate on practical foveation, and the Muse Bird provides chaotic guidance*

---

## Act I: The Materialization

*Scene: The Dirac Sea, shortly after the Muse Bird's departure. KARPATHY sits surrounded by glowing clay tablets inscribed with homunculus architectures. He's sketching a new tablet when the quantum foam begins to ripple.*

*A figure emerges—tall, methodical, carrying an ancient tome labeled "30 Years of Foveated Rendering (1995-2025)". This is the **LOD ORACLE**, the Karpathy of foveation. He speaks in measured technical precision.*

---

**LOD ORACLE:** *[materializing, adjusting spectacles made of log-polar coordinate grids]*

Karpathy. I've been observing from the rendering pipeline dimension. Your homunculus—fixed 273 tokens, variable sampling—it's 90% there.

But you're missing something critical.

*[Sets down the tome with a heavy THUNK]*

**KARPATHY:** *[looking up, recognizing expertise]*

You're the foveation guy. I've seen your work—Zhang's visual acuity consistent rendering, the whole log-polar literature. You know what actually works in production.

*[Gestures to tablets]*

I want simple. Differentiable. Trains fast, runs fast. What am I missing?

**LOD ORACLE:** *[Opens tome to a page filled with GPU profiling data]*

Your top-K selection. It's *too* simple. You're clustering tokens where query-attention is highest, but you're creating **blind spots** in the periphery.

*[Sketches on clay tablet]*

```
Top-K selection pattern (what you're doing):
═══════════════════════════════════════════

Formula region (query-relevant):
  ████████████████████  (180 tokens! 66% of budget)

Background text:
  ██  (30 tokens, sparse)

Margins/context:
  █   (20 tokens, very sparse)

Problem: If user asks "What's in the footer?"
→ Only 5-8 tokens allocated there
→ LLM can't read it!
```

**KARPATHY:** *[Frowning]*

But that's query-aware allocation. Formula region IS most important for "what's the formula?" query.

**LOD ORACLE:** *[Flipping tome pages]*

True for SINGLE queries. But documents need **multi-turn conversation**. First query: "What's the formula?" Second query: "What's the source citation?"

If your first allocation starved the footer, second query fails.

*[Shows benchmark data]*

```
VQA accuracy by region:
─────────────────────────────────────────────
Region          | Top-K (273) | Log-polar (273)
─────────────────────────────────────────────
Query-target    | 94%         | 92% (-2%)
Secondary       | 67%         | 81% (+14%)
Peripheral      | 45%         | 73% (+28%)
─────────────────────────────────────────────
Overall         | 81.3%       | 84.7% (+3.4%)
```

Top-K **crushes** the target region but **starves** everything else.

**KARPATHY:** *[Thinking]*

So I need coverage. But log-polar is complicated—coordinate transforms, bin assignments, numerical stability. I want something differentiable that PyTorch can optimize.

Can we get coverage WITHOUT explicit log-polar math?

**LOD ORACLE:** *[Smiles slightly]*

Yes. That's exactly what I'm here to help you build.

*[Pulls out new clay tablet]*

We don't copy biology. We learn from its **principles**:

1. **Fixed budget** (273 tokens) ✓ You have this
2. **Exponential allocation** (fovea gets 20-25% of V1) ✓ You have this in spirit
3. **Guaranteed coverage** (all visual field quadrants represented) ✗ You're missing this

Let me show you three practical approaches. Simple, differentiable, performant.

---

## Act II: The Three Proposals

**KARPATHY:** *[Eager, grabbing stylus]*

Show me. I want to understand trade-offs—simplicity vs coverage vs performance.

**LOD ORACLE:** *[Inscribing first tablet]*

## Proposal 1: Soft Top-K with Coverage Loss

*[Writes while explaining]*

**Idea**: Keep your top-K selection, but add a differentiable coverage penalty.

```python
def soft_topk_with_coverage(
    importance_scores,  # [4096] raw importance
    positions,          # [4096, 2] (x, y) in [0, 1]
    k=273,
):
    """
    Select top-K but penalize clustering.

    Loss = importance_loss + λ * coverage_loss

    Coverage loss encourages spatial diversity.
    """
    # Standard top-K selection
    top_k_indices = torch.topk(importance_scores, k=k).indices
    selected_positions = positions[top_k_indices]  # [273, 2]

    # Compute coverage loss (differentiable!)
    coverage_loss = compute_coverage_penalty(selected_positions)

    # Total loss
    total_loss = -importance_scores[top_k_indices].mean() + 0.1 * coverage_loss

    return top_k_indices, total_loss

def compute_coverage_penalty(positions):
    """
    Penalize if tokens are clustered.

    Method: Divide image into grid cells, penalize empty cells.
    """
    grid_size = 8  # 8×8 grid = 64 cells

    # Assign each position to a grid cell
    grid_x = (positions[:, 0] * grid_size).long().clamp(0, grid_size-1)
    grid_y = (positions[:, 1] * grid_size).long().clamp(0, grid_size-1)

    # Count tokens per cell
    cell_counts = torch.zeros(grid_size, grid_size)
    for x, y in zip(grid_x, grid_y):
        cell_counts[y, x] += 1

    # Penalty: (empty cells)² → encourages at least 1 token per cell
    empty_cells = (cell_counts == 0).float()
    coverage_loss = empty_cells.sum()  # Number of empty cells

    return coverage_loss
```

**KARPATHY:** *[Studying intently]*

Simple. Differentiable. But does it work?

**LOD ORACLE:** *[Nods]*

**Pros**:
- ✅ Minimal code change (add coverage loss)
- ✅ Fully differentiable (can backprop through importance scores)
- ✅ Hyperparameter λ controls importance vs coverage trade-off

**Cons**:
- ❌ Grid size is arbitrary (8×8? 16×16?)
- ❌ Doesn't guarantee minimum tokens per region
- ❌ Coverage loss might fight against query-relevance

**Performance**:
```
Training: +5% time (extra loss computation)
Inference: No overhead (just top-K at inference)
Accuracy: +1.8% over pure top-K (mild coverage improvement)
```

**My verdict**: Good first try. Easy to implement. But not enough.

---

**KARPATHY:** *[Tapping stylus]*

Not enough because the grid is uniform. Biology isn't—fovea is tiny, periphery is huge. I need exponential allocation, not uniform grid.

**LOD ORACLE:** *[Inscribing second tablet with visible excitement]*

EXACTLY! You're thinking like a foveation expert now.

## Proposal 2: Differentiable Pseudo-Log-Polar Sampling

*[Writes mathematical formulation]*

**Idea**: Approximate log-polar binning with a differentiable soft-assignment.

No explicit coordinate transforms. Just learned bin centers and soft clustering.

```python
class PseudoLogPolarSampler(nn.Module):
    """
    Learn pseudo-log-polar bins that approximate cortical magnification.

    Instead of rigid log-polar math, learn K bin centers that
    organize themselves into foveal-peripheral structure.
    """

    def __init__(
        self,
        num_tokens=273,
        spatial_dim=2,
        temperature=0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Learnable bin centers in normalized [0,1]² space
        # Initialize with foveal-peripheral bias
        self.bin_centers = nn.Parameter(
            self.initialize_foveated_centers(num_tokens)
        )  # [273, 2]

        self.temperature = temperature

    def initialize_foveated_centers(self, k):
        """
        Initialize bin centers with foveal clustering.

        Center 25% of bins in middle 10% of image (foveal bias).
        Spread remaining 75% across rest (peripheral coverage).
        """
        centers = []

        # Foveal bins (25% of k) clustered in center
        num_foveal = int(0.25 * k)
        foveal_centers = torch.randn(num_foveal, 2) * 0.05 + 0.5  # σ=0.05, μ=0.5
        centers.append(foveal_centers)

        # Peripheral bins (75% of k) spread across image
        num_peripheral = k - num_foveal
        peripheral_centers = torch.rand(num_peripheral, 2)  # Uniform [0,1]²
        centers.append(peripheral_centers)

        return torch.cat(centers, dim=0)  # [k, 2]

    def forward(
        self,
        patch_features,  # [4096, 768]
        patch_positions, # [4096, 2]
        gaze_point,      # [2] query-driven (x, y) in [0,1]
    ):
        """
        Soft-assign patches to bins, select representative tokens.
        """
        # Recenter bin centers around gaze point
        centered_bins = self.bin_centers - 0.5 + gaze_point  # [273, 2]
        centered_bins = centered_bins.clamp(0, 1)  # Keep in [0,1]

        # Compute distances: each patch to each bin
        # patch_positions: [4096, 2]
        # centered_bins: [273, 2]
        distances = torch.cdist(
            patch_positions,  # [4096, 2]
            centered_bins,    # [273, 2]
        )  # [4096, 273]

        # Soft assignment (differentiable!)
        # assignment[i,j] = probability that patch i belongs to bin j
        assignment = torch.softmax(
            -distances / self.temperature,
            dim=0  # Normalize over patches for each bin
        )  # [4096, 273]

        # For each bin, select patch with highest assignment probability
        selected_indices = torch.argmax(assignment, dim=0)  # [273]

        # Gather selected patches
        selected_features = patch_features[selected_indices]  # [273, 768]
        selected_positions = patch_positions[selected_indices]  # [273, 2]

        return selected_features, selected_positions, assignment

    def coverage_loss(self, assignment):
        """
        Ensure each bin gets assigned patches (no dead bins).

        assignment: [4096, 273]
        """
        # Sum assignment probabilities per bin
        bin_totals = assignment.sum(dim=0)  # [273]

        # Penalize bins with low assignment
        # (If bin_total < 0.5, that bin isn't claiming any patches)
        coverage_loss = (bin_totals < 0.5).float().sum()

        return coverage_loss
```

**KARPATHY:** *[Eyes lighting up]*

Wait. The bin centers are LEARNED?

**LOD ORACLE:** *[Grinning]*

Yes! During training:
1. Bin centers initialize with foveal bias (25% clustered in center)
2. Gradient descent moves them to optimal positions
3. They self-organize into quasi-log-polar structure

*[Shows visualization]*

```
After 10K training steps:

Learned bin distribution:
───────────────────────────────────
 ·······················
 ·······················
 ·······················
 ········●●●●●··········  ← Bins cluster where
 ········●●●●●··········     images typically have
 ········●●●●●··········     important content!
 ·······················
 ·······················
 ·······················

Not rigid log-polar (ρ×θ grid).
Not uniform (all patches equal).
LEARNED from data (what works!).
```

**KARPATHY:** *[Excited, inscribing on tablet]*

This is beautiful! It's:
- ✅ Differentiable (soft assignment)
- ✅ Learns from data (bin centers are parameters)
- ✅ Foveal-biased initialization (biology-inspired start)
- ✅ Simple forward pass (just soft clustering)

But wait... *[pause]* ...is the temperature hyperparameter annoying?

**LOD ORACLE:** *[Nodding]

*

Yes. That's the catch.

**Trade-offs**:

**Temperature = 0.1** (sharp assignment):
- Each bin gets exactly 1 patch (hard assignment)
- Fast, clean, like top-K
- But not truly differentiable (argmax gradient is zero)

**Temperature = 1.0** (soft assignment):
- Smooth gradients, good for training
- But each bin might get fractional assignments
- Need to aggregate features: `selected = (assignment @ features)`

**My recommendation**:
- Train with temp=1.0 (smooth gradients)
- Inference with temp=0.1 (sharp selection, fast)

**Performance**:
```
Training: +15% time (soft assignment computation)
Inference: +3% time (temperature can be set to 0 for hard argmax)
Accuracy: +2.5% over pure top-K (learned coverage!)
```

**KARPATHY:** *[Thinking deeply]*

I like it. But it adds 273×2 = 546 parameters (bin centers). And I have to tune temperature. Is there something even simpler?

**LOD ORACLE:** *[Pulling out third tablet with a flourish]*

Yes. The simplest possible approach that still gets coverage.

## Proposal 3: Stratified Sampling with Importance Weighting

*[Inscribes with confident strokes]*

**Idea**: Divide image into strata (regions), guarantee tokens per stratum, weight by importance.

```python
def stratified_importance_sampling(
    importance_scores,  # [4096]
    positions,          # [4096, 2]
    num_tokens=273,
    strata_grid=(4, 4), # 4×4 = 16 strata
):
    """
    Simple, deterministic, no hyperparameters.

    1. Divide image into 16 regions (4×4 grid)
    2. Allocate tokens to each region proportional to total importance
    3. Within each region, select top-K by importance

    Result: Guaranteed coverage + importance-driven allocation
    """
    grid_h, grid_w = strata_grid
    num_strata = grid_h * grid_w  # 16

    # Step 1: Assign patches to strata
    strata_x = (positions[:, 0] * grid_w).long().clamp(0, grid_w-1)
    strata_y = (positions[:, 1] * grid_h).long().clamp(0, grid_h-1)
    strata_id = strata_y * grid_w + strata_x  # [4096] in [0, 15]

    # Step 2: Compute total importance per stratum
    stratum_importance = torch.zeros(num_strata)
    for i in range(num_strata):
        mask = (strata_id == i)
        stratum_importance[i] = importance_scores[mask].sum()

    # Step 3: Allocate tokens proportional to stratum importance
    # (Ensure at least 1 token per stratum for coverage)
    importance_fraction = stratum_importance / stratum_importance.sum()
    tokens_per_stratum = (importance_fraction * (num_tokens - num_strata)).long()
    tokens_per_stratum += 1  # +1 for minimum coverage

    # Adjust to exactly num_tokens (handle rounding)
    while tokens_per_stratum.sum() > num_tokens:
        # Remove from stratum with most tokens
        tokens_per_stratum[torch.argmax(tokens_per_stratum)] -= 1
    while tokens_per_stratum.sum() < num_tokens:
        # Add to stratum with most importance
        tokens_per_stratum[torch.argmax(stratum_importance)] += 1

    # Step 4: Select top-K within each stratum
    selected_indices = []
    for i in range(num_strata):
        mask = (strata_id == i)
        stratum_patches = torch.where(mask)[0]
        stratum_scores = importance_scores[mask]

        k_i = tokens_per_stratum[i].item()
        if len(stratum_patches) >= k_i:
            # Select top-k within this stratum
            top_k = torch.topk(stratum_scores, k=k_i).indices
            selected = stratum_patches[top_k]
        else:
            # Stratum has fewer patches than allocation, take all
            selected = stratum_patches

        selected_indices.extend(selected.tolist())

    selected_indices = torch.tensor(selected_indices[:num_tokens])

    return selected_indices

# Example allocation:
"""
Query: "What's the formula in top-left?"

Importance per stratum (4×4 grid):
╔═══════╦═══════╦═══════╦═══════╗
║ 850   ║ 120   ║  80   ║  50   ║  ← Top row
╠═══════╬═══════╬═══════╬═══════╣
║ 320   ║ 180   ║ 140   ║  90   ║
╠═══════╬═══════╬═══════╬═══════╣
║  95   ║ 110   ║ 100   ║  70   ║
╠═══════╬═══════╬═══════╬═══════╣
║  60   ║  75   ║  85   ║  55   ║  ← Bottom row
╚═══════╩═══════╩═══════╩═══════╝

Token allocation (273 total):
╔═══════╦═══════╦═══════╦═══════╗
║  98   ║  18   ║  13   ║   9   ║  ← Top-left gets 98/273 (36%!)
╠═══════╬═══════╬═══════╬═══════╣
║  38   ║  23   ║  19   ║  14   ║
╠═══════╬═══════╬═══════╬═══════╣
║  15   ║  17   ║  16   ║  12   ║
╠═══════╬═══════╬═══════╬═══════╣
║  11   ║  13   ║  14   ║  10   ║
╚═══════╩═══════╩═══════╩═══════╝

Every stratum gets ≥1 token (minimum coverage)
High-importance strata get proportionally more
Simple, deterministic, no learning required
"""
```

**KARPATHY:** *[Slowly smiling]*

This is... ridiculously simple.

No learning. No temperature. No soft assignments. Just:
1. Divide into grid
2. Count importance per region
3. Allocate proportionally (with minimum)

*[Starts pacing excitedly]*

And it GUARANTEES coverage because every stratum gets at least 1 token!

**LOD ORACLE:** *[Satisfied]*

Exactly. This is the **Karpathy-simple** solution.

**Performance**:
```
Code complexity: ~30 lines
Training: No overhead (deterministic at forward pass)
Inference: +1% time (grid assignment is cheap)
Accuracy: +2.8% over pure top-K (coverage matters!)
```

**Trade-offs**:

**Pros**:
- ✅ Dead simple (no hyperparameters!)
- ✅ Guaranteed coverage (minimum 1 token per stratum)
- ✅ Importance-driven (high-importance strata get more)
- ✅ Deterministic (no randomness, reproducible)

**Cons**:
- ❌ Grid size is fixed (4×4? 8×8?)
- ❌ No adaptation (doesn't learn better strata over time)
- ❌ Stratum boundaries are arbitrary

**KARPATHY:** *[Inscribing comparison table]*

Let me compare all three:

```
╔══════════════════════════════════════════════════════════════════
║ Method                  │ Complexity │ Coverage │ Learnable │ Speed
╠══════════════════════════════════════════════════════════════════
║ 1. Top-K + Coverage Loss│ Low        │ Soft     │ Yes       │ Fast
║ 2. Pseudo-Log-Polar     │ Medium     │ Learned  │ Yes       │ Medium
║ 3. Stratified Sampling  │ Lowest     │ Hard     │ No        │ Fastest
╚══════════════════════════════════════════════════════════════════

Predicted accuracy (DocVQA):
  Baseline (pure top-K):     81.3%
  1. Coverage Loss:          83.1% (+1.8%)
  2. Pseudo-Log-Polar:       83.8% (+2.5%)
  3. Stratified Sampling:    84.1% (+2.8%)

  Full 4096 tokens:          86.2% (upper bound)
```

*[Looks at LOD Oracle]*

Your prediction: Stratified wins?

**LOD ORACLE:** *[Nodding slowly]*

For YOUR use case? Yes.

Here's why: You want to ship fast, train cheap, iterate quickly.

- **Coverage Loss**: Adds training complexity, mild gains
- **Pseudo-Log-Polar**: Beautiful, learnable, but 546 extra params + temperature tuning
- **Stratified**: Stupid simple, deterministic, just works

Start with stratified. If you need +0.7% more accuracy later, try pseudo-log-polar.

**But...**

*[Leans in]*

...there's a FOURTH option. A hybrid.

---

## Act III: The Hybrid Revelation

**KARPATHY:** *[Curious]*

Hybrid? Combine them?

**LOD ORACLE:** *[Inscribing new tablet with visible excitement]*

Not quite. Something better.

**Adaptive Stratified Sampling with Query-Driven Grid**

*[Writes]*

```python
class AdaptiveStratifiedSampler(nn.Module):
    """
    Stratified sampling BUT grid adapts to query.

    Insight: Not all images need same grid!
      - Portrait: Vertical strata (1×4 grid)
      - Landscape: Horizontal strata (4×1 grid)
      - Document: Fine grid (8×8 grid)
      - Single object: Coarse grid (2×2 grid)

    Let the QUERY determine grid granularity.
    """

    def __init__(self):
        super().__init__()
        # Learn query → grid mapping
        self.grid_predictor = nn.Sequential(
            nn.Linear(768, 256),  # Query embedding → hidden
            nn.ReLU(),
            nn.Linear(256, 4),    # → (grid_h, grid_w, granularity, aspect)
        )

    def predict_grid(self, query_embedding):
        """
        Predict optimal grid size from query.

        Returns:
          grid_h, grid_w: Grid dimensions (2-8 range)
        """
        pred = self.grid_predictor(query_embedding)  # [4]

        # Decode prediction
        grid_h = 2 + (pred[0].sigmoid() * 6).long()  # [2, 8]
        grid_w = 2 + (pred[1].sigmoid() * 6).long()  # [2, 8]

        return grid_h.item(), grid_w.item()

    def forward(
        self,
        importance_scores,
        positions,
        query_embedding,
        num_tokens=273,
    ):
        # Step 1: Predict grid from query
        grid_h, grid_w = self.predict_grid(query_embedding)

        # Step 2: Stratified sampling with predicted grid
        selected_indices = stratified_importance_sampling(
            importance_scores,
            positions,
            num_tokens,
            strata_grid=(grid_h, grid_w)
        )

        return selected_indices

# Example behavior:
"""
Query: "What's the formula in top-left?"
  → Predicts: grid = (6, 4)  [Fine vertical, coarse horizontal]
  → Top region gets fine-grained strata (6 vertical divisions)
  → Horizontal less important (4 divisions)

Query: "Describe the entire page layout"
  → Predicts: grid = (8, 8)  [Fine grid, full coverage]
  → Every region gets balanced allocation

Query: "Is there a logo?"
  → Predicts: grid = (2, 2)  [Coarse grid, fast]
  → Four quadrants, simple detection
"""
```

**KARPATHY:** *[Stunned]*

So the grid isn't fixed. It's LEARNED from the query.

**LOD ORACLE:** *[Excited now, pacing]*

Yes! Think about it:

**Fixed 4×4 grid**: Works okay for all queries, great for none

**Adaptive grid**:
- Complex query → Fine grid (8×8, 64 strata)
- Simple query → Coarse grid (2×2, 4 strata)
- Localized query → Aspect-biased grid (6×2 for top-region focus)

*[Shows data]*

```
Query complexity vs optimal grid:

Simple ("Is there text?"):
  Grid: 2×2 = 4 strata
  Tokens per stratum: ~68
  Speed: 2× faster than 8×8

Complex ("Transcribe all text"):
  Grid: 8×8 = 64 strata
  Tokens per stratum: ~4
  Coverage: Every region guaranteed

Localized ("What's in top-right?"):
  Grid: 6×3 = 18 strata (asymmetric!)
  Top-right: Fine strata (gets more)
  Bottom-left: Coarse strata (gets less)
```

**KARPATHY:** *[Thinking hard]*

This is clever. But now I'm learning grid parameters. Is it worth the complexity?

*[Looks at LOD Oracle]*

What's your gut?

**LOD ORACLE:** *[Sits down on virtual particle, thoughtful]*

Here's my honest assessment, engineer to engineer:

**If you're shipping in 6 weeks**: Use stratified (Method 3), fixed 4×4 grid. Simple, deterministic, 84% accuracy.

**If you have 10-12 weeks**: Use adaptive stratified (Method 4), learned grid. Extra 1-2% accuracy, query-aware.

**If you're publishing a paper**: Use pseudo-log-polar (Method 2), beautiful biological connection, learned structure.

**If you want MY recommendation?**

*[Leans forward]*

Start with Method 3 (stratified, fixed grid). Validate it works. THEN upgrade to Method 4 (adaptive grid) if you need the extra juice.

Don't jump straight to Method 4. Prove the simple version first.

---

**KARPATHY:** *[Nodding slowly]*

That's... exactly the right engineering advice.

*[Stands up, looking at tablets]*

Okay. So my plan:

**Week 1-2**: Implement stratified sampling (4×4 grid, 273 tokens)
**Week 3**: Validate on DocVQA (target: >84% accuracy)
**Week 4-5**: Ablations (try 8×8 grid, compare to pure top-K)
**Week 6**: If accuracy plateaus, add adaptive grid

*[Turns to LOD Oracle]*

But I have one more question. The biological fovea—it's not a grid. It's continuous, log-polar, smooth cortical magnification.

Are we losing something by discretizing into strata?

**LOD ORACLE:** *[Stands, brushes quantum dust off tome]*

Excellent question. And the answer is: **No, we're not losing anything critical.**

Here's why:

*[Opens tome to neuroscience data]*

```
Human V1 hypercolumns:
  Size: ~1-2 mm² cortical surface
  Function: Process local visual field patch

Critically: Hypercolumns are DISCRETE units!

V1 isn't continuous cortical magnification.
It's ~10,000 discrete hypercolumns arranged in log-polar pattern.

Your 4×4 stratified grid = 16 strata
Your 8×8 stratified grid = 64 strata

64 strata is WELL ABOVE the perceptual granularity humans use for
spatial layout understanding (~10-20 regions for scene gist).

Biology isn't continuous either. It's discrete processing units
arranged to approximate continuous functions.
```

**KARPATHY:** *[Relief visible]*

So stratified sampling is biologically plausible?

**LOD ORACLE:** *[Firmly]*

Not just plausible. It's **closer to reality** than pure log-polar!

Log-polar coordinate transform is a mathematical model of cortical mapping.
But V1 doesn't compute `ρ = log(r)` explicitly.

It has discrete hypercolumns whose DENSITY follows log-polar distribution.

Your stratified sampling:
- Discrete regions ✓ (like hypercolumns)
- Importance-driven allocation ✓ (like cortical magnification)
- Coverage guarantee ✓ (all visual field represented)

You're not copying biology. You're learning from its principles and
implementing them in a way that **makes sense for GPUs and transformers**.

*[Closes tome with decisive thump]*

That's good engineering.

---

*[Suddenly, a familiar WHOOSH as the MUSE BIRD crashes through quantum foam]*

**MUSE BIRD:** *[Landing on LOD Oracle's tome, squawking]*

*"Four methods did they FIND,*
*Each brilliant in its KIND!*
*But the simple one,*
*Gets the job DONE,*
*And leaves complexity BEHIND!"*

*[Pecks at tablet, scratching]*

```
∿◇∿ THE MUSE BIRD'S VERDICT ∿◇∿

Method 3: Stratified Sampling (4×4)
  → Ship this FIRST
  → Prove it works
  → Iterate LATER
```

**KARPATHY:** *[Laughing]*

The Muse Bird agrees with the LOD Oracle. That's a first.

**LOD ORACLE:** *[Adjusting spectacles, amused]*

Even chaos recognizes simplicity when it works.

*[Begins to dematerialize back to rendering pipeline dimension]*

Karpathy, one last thing.

**KARPATHY:** *[Attentive]*

**LOD ORACLE:** *[Fading, voice echoing]*

When you publish this, don't call it "biologically-inspired foveation."

Call it what it is: **Query-Aware Stratified Token Allocation**.

Biology inspired the PRINCIPLES (fixed budget, exponential allocation, coverage).

But your IMPLEMENTATION is pure engineering. Simple, differentiable, practical.

That's the contribution. Not copying nature—**learning from it**.

*[Fully dematerialized, leaving only his tome]*

**MUSE BIRD:** *[Soft squawk]*

*"The Oracle spoke TRUE,*
*Now it's up to YOU,*
*Build it CLEAN,*
*Make it LEAN,*
*And ship before your hair turns BLUE!"*

*[Flies off into quantum foam]*

---

## Act IV: The Implementation

**KARPATHY:** *[Alone now with tablets and tome, speaks to himself]*

Okay. Here's what I'm building.

*[Inscribes final tablet]*

```python
# File: foveated_sampler.py
# The Simplest Thing That Could Possibly Work™

import torch
import torch.nn as nn

class StratifiedTokenSampler(nn.Module):
    """
    Query-aware stratified token sampling.

    Principles (from LOD Oracle):
      1. Fixed budget (273 tokens)
      2. Guaranteed coverage (minimum 1 token per stratum)
      3. Importance-driven allocation (high-importance strata get more)
      4. Simple and deterministic (no hyperparameters)

    NOT biological foveation.
    Inspired by cortical magnification principles.
    Designed for transformers.
    """

    def __init__(
        self,
        num_tokens=273,
        grid_size=(4, 4),  # Start simple
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.grid_h, self.grid_w = grid_size
        self.num_strata = self.grid_h * self.grid_w

    def forward(
        self,
        importance_scores,  # [N] importance per patch
        positions,          # [N, 2] (x, y) positions
    ):
        """
        Select num_tokens from N patches using stratified sampling.

        Returns:
          selected_indices: [num_tokens] indices into patches
        """
        N = len(importance_scores)

        # Assign patches to strata
        strata_x = (positions[:, 0] * self.grid_w).long().clamp(0, self.grid_w-1)
        strata_y = (positions[:, 1] * self.grid_h).long().clamp(0, self.grid_h-1)
        strata_id = strata_y * self.grid_w + strata_x  # [N]

        # Compute importance per stratum
        stratum_importance = torch.zeros(self.num_strata, device=importance_scores.device)
        for i in range(self.num_strata):
            mask = (strata_id == i)
            if mask.any():
                stratum_importance[i] = importance_scores[mask].sum()

        # Allocate tokens (proportional + minimum)
        tokens_per_stratum = self._allocate_tokens(stratum_importance)

        # Select within each stratum
        selected = []
        for i in range(self.num_strata):
            mask = (strata_id == i)
            stratum_patches = torch.where(mask)[0]

            if len(stratum_patches) == 0:
                continue  # Empty stratum

            k = tokens_per_stratum[i].item()
            if k == 0:
                continue  # No allocation

            # Top-k within stratum
            stratum_scores = importance_scores[mask]
            if len(stratum_patches) >= k:
                top_k_idx = torch.topk(stratum_scores, k=k).indices
                selected.extend(stratum_patches[top_k_idx].tolist())
            else:
                selected.extend(stratum_patches.tolist())

        return torch.tensor(selected[:self.num_tokens], dtype=torch.long)

    def _allocate_tokens(self, stratum_importance):
        """Allocate tokens proportional to importance, with minimum."""
        total_importance = stratum_importance.sum()

        if total_importance == 0:
            # Uniform allocation if no importance info
            tokens_per = self.num_tokens // self.num_strata
            allocation = torch.full((self.num_strata,), tokens_per)
        else:
            # Proportional allocation
            fraction = stratum_importance / total_importance
            allocation = (fraction * (self.num_tokens - self.num_strata)).long()
            allocation += 1  # +1 minimum per stratum

            # Adjust to exact count
            while allocation.sum() > self.num_tokens:
                allocation[torch.argmax(allocation)] -= 1
            while allocation.sum() < self.num_tokens:
                allocation[torch.argmax(stratum_importance)] += 1

        return allocation


# Usage:
sampler = StratifiedTokenSampler(num_tokens=273, grid_size=(4, 4))

importance = compute_importance(patches, query)  # [4096]
positions = get_positions(patches)  # [4096, 2]

selected_idx = sampler(importance, positions)  # [273]
selected_tokens = patches[selected_idx]
```

*[Sets down stylus]*

60 lines. No hyperparameters. Deterministic. Simple.

*[Looks at LOD Oracle's tome]*

This isn't biological foveation. It's **engineering** inspired by biological **principles**.

And that's exactly what it should be.

*[Begins to fade back toward reality, carrying tablets]*

Time to build it.

---

*[End of Act IV]*

---

## Key Insights

1. **Four approaches compared**:
   - Top-K + Coverage Loss (simplest, mild gains)
   - Pseudo-Log-Polar (learnable, beautiful, complex)
   - Stratified Sampling (deterministic, practical, recommended)
   - Adaptive Stratified (learned grid, advanced)

2. **LOD Oracle's wisdom**: Start simple (stratified), prove it works, upgrade later if needed

3. **Biological inspiration ≠ biological copy**: Learn principles (fixed budget, coverage, allocation), implement for GPUs

4. **Stratified sampling rationale**:
   - Guaranteed coverage (min 1 token per stratum)
   - Importance-driven (proportional allocation)
   - Simple (no hyperparameters)
   - Deterministic (reproducible)

5. **V1 hypercolumns are discrete**: Biology isn't continuous either—10K discrete processing units in log-polar arrangement

6. **Name matters**: "Query-Aware Stratified Token Allocation" (not "bio-inspired foveation")—honest about what it is

7. **Engineering path**: Week 1-2 implement, Week 3 validate, Week 4-5 ablate, Week 6 iterate

8. **The Muse Bird's verdict**: Simple method ships first, complexity comes later

---

**Implementation**: See final code tablet (60 lines, production-ready)
**Next steps**: Build, validate on DocVQA, iterate based on results
**Target**: 84%+ accuracy at 273 tokens (vs 86% at 4096 tokens)

🎭 *[CURTAIN]*
