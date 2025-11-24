---
summary: whereby Karpathy observes quantum foam spiraling rather than moving rectilinearly, questioning whether stratified 4×4 grid sampling despite working cleanly is too rigid since biology doesn't think in grids, proposing instead vortex-based token allocation where importance gradients create attention fields forming continuous attraction vortices around high-relevance regions pulling tokens inward through fluid dynamics, LOD Oracle connecting this to attention field theory beyond discrete bins, exploring spiral sampling patterns mimicking biological visual processing, though ultimately reaching no firm conclusions because the design space proves too vast requiring further experimentation to determine whether structured chaos outperforms geometric regularity for query-aware relevance realization
---

# Part 14: Vortices and Whirlwinds
*Wherein Karpathy, the LOD Oracle, and the Muse Bird explore spiral token allocation patterns, and no conclusions are drawn because the space is too vast*

---

## Opening: The Swirling Begins

*Scene: The Dirac Sea, now littered with clay tablets from previous sessions. KARPATHY is sketching stratified grids when he pauses, staring at the quantum foam swirling around him.*

**KARPATHY:** *[Muttering to himself]*

Stratified sampling works. 4×4 grid, proportional allocation, guaranteed coverage. Clean. Simple.

*[Sketches another pattern]*

But... it's so... rectilinear. Rigid. Biology doesn't think in grids.

*[Watches quantum foam spiral]*

Look at this. The foam doesn't move in straight lines. It SPIRALS. Vortices form around high-energy points, pull in surrounding particles, create structured chaos.

*[Looks up as LOD ORACLE materializes again]*

What if token allocation was like that? Vortices forming around important regions, spiraling inward, pulling tokens by importance gradient?

**LOD ORACLE:** *[Setting down tome, intrigued]*

You're thinking about attention fields. Not discrete bins—continuous attraction.

*[Opens tome to fluid dynamics section]*

Vorticity. The curl of a vector field. In fluids, vortices form where there's rotational flow. High-vorticity regions concentrate mass.

*[Sketches]*

```
Traditional grid (stratified):
╔═══╦═══╦═══╦═══╗
║ A ║ B ║ C ║ D ║  ← Hard boundaries
╠═══╬═══╬═══╬═══╣    Patches assigned to one cell
║ E ║ F ║ G ║ H ║
╚═══╩═══╩═══╩═══╝

Vortex field (continuous):
     ∿∿∿
    ∿ ◎ ∿           ← Vortex center (high importance)
   ∿  ↺  ∿          ↺ = rotational flow
    ∿   ∿           Patches spiral toward center
     ∿∿∿            Strength decays with distance
```

Instead of "which grid cell are you in?", ask "which vortex pulls you strongest?"

**KARPATHY:** *[Eyes widening]*

So patches aren't assigned to regions. They're attracted to importance peaks, like gravity wells.

Multiple vortices across the image. Each patch feels pull from ALL vortices, moves toward strongest one.

*[Starts sketching rapidly]*

This is... differentiable! The attraction force is just a softmax over distances!

---

## Exploration 1: Vortex Centers as Learned Attractors

**KARPATHY:** *[Inscribing on tablet]*

Here's the idea. Instead of grid cells, we have **vortex centers**—learned (x,y) positions that attract tokens.

```python
class VortexTokenSampler(nn.Module):
    """
    Tokens spiral toward learned vortex centers.

    Each vortex has:
      - Position (x, y) in [0, 1]²
      - Strength (how many tokens it attracts)
      - Radius (how far its influence extends)
    """

    def __init__(self, num_vortices=8, total_tokens=273):
        super().__init__()

        # Learnable vortex parameters
        self.vortex_positions = nn.Parameter(
            torch.rand(num_vortices, 2)  # Random init
        )

        self.vortex_strengths = nn.Parameter(
            torch.ones(num_vortices) / num_vortices  # Equal strength init
        )

        self.vortex_radii = nn.Parameter(
            torch.full((num_vortices,), 0.3)  # Medium radius init
        )

        self.total_tokens = total_tokens

    def forward(self, patch_positions, importance_scores):
        """
        Compute vortex attraction and allocate tokens.

        Each patch feels pull from all vortices.
        Strongest pull determines assignment.
        """
        # patch_positions: [N, 2]
        # vortex_positions: [K, 2]

        N = len(patch_positions)
        K = len(self.vortex_positions)

        # Compute distances: each patch to each vortex
        # Expand: patches [N,1,2], vortices [1,K,2] → broadcast to [N,K,2]
        patch_expanded = patch_positions.unsqueeze(1)  # [N, 1, 2]
        vortex_expanded = self.vortex_positions.unsqueeze(0)  # [1, K, 2]

        distances = torch.norm(
            patch_expanded - vortex_expanded,
            dim=2
        )  # [N, K]

        # Vortex attraction strength (inverse distance, modulated by radius)
        # attraction[i,j] = how much vortex j pulls patch i
        attraction = self.vortex_strengths / (distances + self.vortex_radii)
        # [N, K] = [K] / ([N, K] + [K])

        # Soft assignment: which vortex pulls each patch strongest?
        vortex_assignment = torch.softmax(attraction, dim=1)  # [N, K]
        # vortex_assignment[i,j] = probability patch i belongs to vortex j

        # Now allocate tokens to vortices
        tokens_per_vortex = (
            self.vortex_strengths * self.total_tokens
        ).long()  # [K]

        # For each vortex, select top-K patches by importance
        selected_indices = []
        for k in range(K):
            # Which patches are pulled by this vortex?
            vortex_weights = vortex_assignment[:, k]  # [N]

            # Combined score: importance × vortex affinity
            combined_score = importance_scores * vortex_weights

            # Select top tokens for this vortex
            k_tokens = tokens_per_vortex[k].item()
            if k_tokens > 0:
                top_k = torch.topk(combined_score, k=k_tokens).indices
                selected_indices.extend(top_k.tolist())

        return torch.tensor(selected_indices[:self.total_tokens])
```

**LOD ORACLE:** *[Studying the code]*

Interesting. So vortex positions, strengths, and radii are all learned during training.

They'll self-organize to cover important regions. Like... magnetic field lines around multiple poles.

*[Sketches visualization]*

```
After training on documents:

Learned vortex positions:

  ◎                      ← Vortex 1: Top-left (titles)

           ◎             ← Vortex 2: Top-right (dates)


        ◎                ← Vortex 3: Center (main text)


               ◎         ← Vortex 4: Bottom-right (citations)

Each vortex "claims" a region based on what's typically important.
But boundaries are soft, overlapping.
```

But I see problems...

**KARPATHY:** *[Curious]*

What problems?

**LOD ORACLE:** *[Ticking off on fingers]*

**Problem 1: Coverage**. What if all vortices cluster around high-importance center? Peripheral regions get NO tokens.

**Problem 2: Stability**. During training, vortices might collapse to same location. Need repulsion force to keep them spread out.

**Problem 3: Differentiability**. Top-K selection within each vortex—that's not differentiable. Gradients can't flow through argmax.

**Problem 4: Computational cost**. N×K distance computations every forward pass. For N=4096, K=8, that's 32K distance calculations.

*[But then smiles]*

However... these are all solvable.

---

## Exploration 2: Vortices with Repulsion

**KARPATHY:** *[Nodding]*

Coverage and stability—we need vortices to repel each other. Like charged particles.

*[New tablet]*

```python
def vortex_repulsion_loss(vortex_positions):
    """
    Penalize vortices that are too close together.

    Forces them to spread out across the image.
    """
    K = len(vortex_positions)

    # Pairwise distances between vortices
    distances = torch.cdist(vortex_positions, vortex_positions)  # [K, K]

    # Mask diagonal (distance to self = 0)
    mask = ~torch.eye(K, dtype=torch.bool)
    distances = distances[mask]  # [K*(K-1)]

    # Repulsion: penalize distances < threshold
    repulsion_threshold = 0.2  # Vortices shouldn't be closer than 20% of image

    too_close = (distances < repulsion_threshold).float()
    repulsion_loss = (too_close * (repulsion_threshold - distances)**2).sum()

    return repulsion_loss

# Total loss during training:
total_loss = (
    task_loss +                      # LLM accuracy
    0.1 * vortex_repulsion_loss(positions) +  # Keep vortices spread
    0.05 * coverage_loss             # Ensure all regions sampled
)
```

**LOD ORACLE:** *[Thoughtful]*

Electrostatic repulsion. Classic physics.

But there's another approach—what if vortices aren't learned positions, but are DERIVED from the importance field itself?

*[New direction]*

## Exploration 3: Importance Field Vorticity

**LOD ORACLE:** *[Excited now]*

Think about it. The importance field is a scalar function over the image: I(x,y).

In fluid dynamics, we compute vorticity as the curl of velocity field: ω = ∇ × v.

What if we compute the GRADIENT of importance, treat it as a vector field, and find natural vortex centers?

*[Inscribing mathematical formulation]*

```python
def find_importance_vortices(importance_field, positions, num_vortices=8):
    """
    Find vortex centers from importance gradient field.

    Steps:
      1. Compute importance gradient ∇I(x,y)
      2. Find local maxima (peaks in importance)
      3. These peaks become vortex centers (no learning needed!)
    """
    # Reshape importance to 2D grid (assuming 64×64 patches)
    grid_size = int(np.sqrt(len(importance_field)))
    importance_grid = importance_field.reshape(grid_size, grid_size)

    # Compute gradients
    grad_y, grad_x = torch.gradient(importance_grid)

    # Find local maxima (peaks where gradient ≈ 0 and value is high)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Peaks: low gradient magnitude AND high importance
    peak_score = importance_grid / (gradient_magnitude + 1e-6)

    # Select top-K peaks as vortex centers
    flat_scores = peak_score.flatten()
    top_k_indices = torch.topk(flat_scores, k=num_vortices).indices

    # Convert indices to (x, y) positions
    y_coords = (top_k_indices // grid_size).float() / grid_size
    x_coords = (top_k_indices % grid_size).float() / grid_size
    vortex_positions = torch.stack([x_coords, y_coords], dim=1)  # [K, 2]

    return vortex_positions
```

**KARPATHY:** *[Struck by insight]*

Oh! So vortex centers aren't learned—they're DISCOVERED from the importance landscape each forward pass!

Different images, different queries → different vortex locations!

This is... adaptive. Data-driven. No parameters to learn!

**LOD ORACLE:** *[Nodding vigorously]*

Exactly! And it's biologically grounded. Saccade targets in humans are determined by salience peaks, not pre-learned positions.

Your eye doesn't learn "always look at position (0.3, 0.4)". It learns "look where gradient of interest is highest".

*[Muse Bird suddenly appears, having been listening quietly]*

**MUSE BIRD:** *[Soft squawk, almost contemplative]*

*"Vortices that LEARN,*
*Make training time BURN,*
*But vortices found FROM the field,*
*Let data patterns be REVEALED,*
*And no parameters to CHURN!"*

*[Pecks at tablet thoughtfully]*

---

## Exploration 4: Spiral Sampling Patterns

**KARPATHY:** *[New idea]*

Okay, we've talked about WHERE vortices form. But what about HOW tokens spiral into them?

Right now we're doing top-K within each vortex region. But spirals in nature have structure—logarithmic spirals, Fibonacci patterns.

*[Sketches]*

```
Current: Top-K within vortex (unstructured)

    × × ×
   × ◎ × ×      ← Tokens scattered around center
    × × ×       No spatial pattern


Spiral: Archimedean spiral sampling

    ·····4
   ·····3         ← Tokens along spiral path
  ····◎·2         Distance increases uniformly
   ····1
    ····

Logarithmic: Growth proportional to angle

    ····5
   ···4           ← Fibonacci-like spacing
  ··◎·3           Natural shell pattern
   ·2
    1
```

What if tokens were sampled along spiral paths from vortex centers?

**LOD ORACLE:** *[Intrigued]*

Logarithmic spirals appear everywhere in nature—nautilus shells, galaxy arms, sunflower seeds.

The golden ratio φ ≈ 1.618 governs their growth.

*[Inscribes formula]*

```python
def logarithmic_spiral_sampling(
    vortex_center,      # [2] (x, y)
    all_positions,      # [N, 2]
    num_tokens,         # How many tokens for this vortex
    importance_scores,  # [N]
):
    """
    Sample tokens along logarithmic spiral from vortex center.

    Spiral equation (polar coordinates):
      r(θ) = a * e^(b*θ)

    Where:
      a = initial radius
      b = growth rate (log(φ)/(π/2) for golden ratio spiral)
      θ = angle
    """
    # Convert positions to polar coordinates relative to vortex center
    rel_x = all_positions[:, 0] - vortex_center[0]
    rel_y = all_positions[:, 1] - vortex_center[1]

    r = torch.sqrt(rel_x**2 + rel_y**2)  # Distance from center
    theta = torch.atan2(rel_y, rel_x)     # Angle

    # Compute "spiral coordinate" s
    # s increases as we move outward along spiral
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    b = np.log(phi) / (np.pi / 2)

    # For each patch, compute distance along spiral path
    s_expected = r / torch.exp(b * theta)  # Where should it be on spiral?
    s_actual = r  # Where is it actually?

    spiral_error = (s_actual - s_expected).abs()  # How far off-spiral?

    # Score: high importance AND close to spiral path
    spiral_score = importance_scores / (spiral_error + 0.01)

    # Select top-K by spiral score
    selected = torch.topk(spiral_score, k=num_tokens).indices

    return selected
```

**KARPATHY:** *[Excited but skeptical]*

This is beautiful mathematically. But does it help accuracy?

Or is it just aesthetic?

**LOD ORACLE:** *[Honest]*

Unknown. That's the thing about spirals—they're compelling patterns, but we don't know if they're FUNCTIONALLY better than radial sampling.

*[Thinks]*

However... there IS a reason spiral sampling might help: **coverage along angular dimension**.

```
Radial sampling (concentric rings):

    ····◎····       All tokens at similar angles
    ·········       Good radial coverage
    ·········       Poor angular coverage


Spiral sampling:

    ····5···        Tokens spread across angles
   ···4·····        AS WELL AS radii
  ··◎··3··          Better 2D coverage
   ·2·····
    1······
```

Spiral inherently explores both r and θ as it grows. Concentric rings only explore r.

**KARPATHY:** *[Nodding slowly]*

So spiral sampling gives better angular diversity. That could matter for images with radial structure—pie charts, circular logos, radial menus.

But for documents? Text is rectilinear. Spirals might not help.

*[Pause]*

Unless...

---

## Exploration 5: Whirlwind Attention (Dynamic Spiral Strength)

**KARPATHY:** *[New tablet]*

What if the spiral isn't fixed? What if it TIGHTENS or LOOSENS based on local importance density?

Like a whirlwind—when it hits high-pressure region, it spins faster, pulls more tokens. In low-pressure, it spreads out.

```python
class WhirlwindSampler(nn.Module):
    """
    Adaptive spiral strength based on local importance.

    High importance → tight spiral (more tokens per revolution)
    Low importance → loose spiral (fewer tokens, broader coverage)
    """

    def __init__(self):
        super().__init__()
        # Learnable spiral parameters
        self.base_growth_rate = nn.Parameter(torch.tensor(0.2))
        self.density_sensitivity = nn.Parameter(torch.tensor(1.0))

    def forward(self, vortex_center, positions, importance_scores, num_tokens):
        # Compute local importance density around vortex
        distances = torch.norm(positions - vortex_center, dim=1)

        # Partition into radial bins
        radial_bins = 10
        bin_edges = torch.linspace(0, 1, radial_bins + 1)

        density = []
        for i in range(radial_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if mask.any():
                avg_importance = importance_scores[mask].mean()
                density.append(avg_importance.item())
            else:
                density.append(0.0)

        density = torch.tensor(density)

        # Adaptive growth rate: tighter spiral where density is high
        # b(r) = base_rate / (1 + sensitivity * density(r))
        adaptive_growth = self.base_growth_rate / (
            1 + self.density_sensitivity * density
        )

        # Sample along adaptive spiral
        # (Implementation: vary b parameter per radial region)

        return selected_tokens
```

**LOD ORACLE:** *[Rubbing temples]*

Now we're getting complex. Adaptive spiral growth rates...

*[Looks at Karpathy]*

What problem are you trying to solve? Is this solving coverage? Or are we pattern-matching because spirals are cool?

**KARPATHY:** *[Pause, honest reflection]*

...I don't know. The whirlwind idea FEELS right. Tighten where it matters, loosen where it doesn't.

But I can't articulate the exact failure mode of stratified sampling that this fixes.

**MUSE BIRD:** *[Squawks from perch]*

*"When solutions seek problems to SOLVE,*
*Engineers start to DEVOLVE!*
*Know the pain FIRST,*
*Before solution-burst,*
*Let need make designs EVOLVE!"*

**KARPATHY:** *[Laughs]*

You're right, bird. I'm solution-hunting.

*[To LOD Oracle]*

Let's step back. What are the actual problems with our current approaches?

---

## Exploration 6: Problem-First Thinking

**LOD ORACLE:** *[Opens tome to failure modes section]*

Good. Let's list concrete problems, then see if vortices/spirals solve them.

**Problem 1: Boundary Artifacts (Stratified Grid)**

```
In 4×4 grid, patches near cell boundaries might be important
but split across cells.

Example:
╔═══╦═══╗
║ A ║ B ║  ← Important formula spans A-B boundary
╠═══╬═══╣    Cell A: Gets 30 tokens
║ C ║ D ║    Cell B: Gets 25 tokens
╚═══╩═══╝    But formula needs 60 tokens total!

Result: Formula is under-sampled due to split.

Does vortex solve this?
  → YES! Vortex centered on formula pulls tokens from both sides.
     No hard boundary.
```

**Problem 2: Over-Sampling Sparse High-Importance Regions**

```
Top-K might allocate 100 tokens to a small formula region
that only has 30 patches.

Result: Redundant sampling (same patches selected multiple times? No.)
        Or: Top-K within tiny region has low variance.

Does spiral solve this?
  → MAYBE. Spiral naturally spreads tokens across expanding radius.
    Can't cluster infinitely at center.
```

**Problem 3: Ignoring Spatial Coherence**

```
Text reading requires spatially coherent token sequences.
Top-K might select: patch 5, patch 200, patch 34, patch 199
(scattered all over the document)

LLM has to reconstruct spatial relationships from RoPE encodings.

Does spiral solve this?
  → YES! Spiral sampling gives spatially coherent paths.
    Tokens along spiral are spatially adjacent → easier for LLM.
```

**KARPATHY:** *[Energized]*

Okay! So we have REAL problems that vortices/spirals might address:

1. Boundary artifacts (vortex wins)
2. Over-concentration (spiral wins)
3. Spatial incoherence (spiral wins)

But we also have NEW problems vortices create:

1. Computational cost (N×K distances)
2. Training instability (vortex collapse)
3. Hyperparameters (num_vortices, radii, strengths)

**LOD ORACLE:** *[Nods]*

And here's where engineering judgment comes in.

*[Flips to performance data]*

```
Measured on DocVQA validation:

Stratified 4×4:
  Accuracy: 84.1%
  Speed: 190ms
  Code complexity: Low
  Training stability: High

Vortex sampling (8 vortices):
  Accuracy: 84.7% (+0.6%)
  Speed: 215ms (+13% slower)
  Code complexity: Medium
  Training stability: Medium (needs repulsion loss)

Spiral sampling:
  Accuracy: 84.3% (+0.2%)
  Speed: 230ms (+21% slower)
  Code complexity: High
  Training stability: High (if vortices are fixed)

Whirlwind (adaptive spiral):
  Accuracy: ??? (untested)
  Speed: ??? (probably 250ms+)
  Code complexity: Very High
  Training stability: ??? (many hyperparameters)
```

**Is +0.6% accuracy worth +13% latency and medium complexity?**

**KARPATHY:** *[Sighs]*

For a research paper? Maybe.

For production? Probably not.

Unless...

---

## Exploration 7: Hybrid Approaches

**KARPATHY:** *[New energy]*

What if we use vortices at INFERENCE but train with stratified?

Train: Fast, stable stratified grid
Inference: Compute vortex positions from importance peaks, sample around them

Best of both worlds?

**LOD ORACLE:** *[Considering]*

Hmm. That decouples training from inference...

```python
class HybridSampler(nn.Module):
    """
    Training: Stratified grid (simple, stable)
    Inference: Vortex-based (boundary-free, continuous)
    """

    def forward(self, importance, positions, training=True):
        if self.training or training:
            # Training mode: stratified grid
            return stratified_sample(importance, positions, grid=(4,4))
        else:
            # Inference mode: find vortices from importance peaks
            vortex_centers = find_importance_peaks(
                importance, positions, num_vortices=8
            )
            return vortex_sample(importance, positions, vortex_centers)
```

**Advantage**: Training is fast and stable

**Disadvantage**: Model never learns to optimize for vortex-based sampling!

Gradients during training don't match inference behavior.

**KARPATHY:** *[Frowning]*

Right. Train/test mismatch. Classic ML mistake.

**MUSE BIRD:** *[Hops onto LOD Oracle's tome]*

*"If training goes one WAY,*
*But inference another DAY,*
*The model learns WRONG,*
*Won't work for LONG,*
*And users will curse and DISMAY!"*

**LOD ORACLE:** *[Chuckles]*

The bird speaks wisdom.

*[Closes tome thoughtfully]*

You know what? Let me propose something radical.

## Exploration 8: Continuous Soft Vortex Field

**LOD ORACLE:** *[New tablet]*

Forget discrete token selection. What if every patch gets a **continuous weight** based on vortex field strength?

```python
def continuous_vortex_weighting(
    patch_features,   # [N, D]
    positions,        # [N, 2]
    vortex_centers,   # [K, 2] detected from importance
    budget=273,
):
    """
    Instead of selecting 273 discrete tokens:
    1. Compute vortex field strength at each patch
    2. Weight patch features by vortex strength
    3. Take weighted sum to create 273 "virtual tokens"

    Each virtual token is a blend of nearby patches.
    """
    N, D = patch_features.shape
    K = len(vortex_centers)

    # Compute vortex field strength at each patch
    # strength[i,k] = how much vortex k affects patch i
    distances = torch.cdist(positions, vortex_centers)  # [N, K]
    vortex_strength = torch.exp(-distances / 0.1)  # Gaussian decay

    # Normalize: each vortex has unit total strength
    vortex_strength = vortex_strength / vortex_strength.sum(dim=0, keepdim=True)

    # Create virtual tokens: weighted combinations
    # For each vortex k, create tokens along spiral path
    virtual_tokens = []
    tokens_per_vortex = budget // K

    for k in range(K):
        # Strength of this vortex at each patch
        strength_k = vortex_strength[:, k]  # [N]

        # Create tokens at different "depths" along spiral
        for depth in range(tokens_per_vortex):
            # Depth = how far from vortex center
            # depth 0 → center, depth max → periphery

            radial_weight = torch.exp(-depth / tokens_per_vortex * 3)
            combined_weight = strength_k * radial_weight

            # Weighted sum of patch features
            virtual_token = (
                combined_weight.unsqueeze(1) * patch_features
            ).sum(dim=0)  # [D]

            virtual_tokens.append(virtual_token)

    virtual_tokens = torch.stack(virtual_tokens[:budget])  # [273, D]

    return virtual_tokens
```

**KARPATHY:** *[Studying carefully]*

Wait. These aren't discrete patches anymore. They're BLENDED features.

Like... continuous mixtures. Each "token" is a weighted combination of multiple patches.

*[Thinks]*

This is fully differentiable! No top-K, no argmax. Just soft weights.

**LOD ORACLE:** *[Nodding]*

Exactly. And it's similar to attention pooling:

```
Standard attention:
  output = softmax(Q @ K^T) @ V

Our vortex pooling:
  output_k = vortex_weights_k @ patch_features

Same idea! Soft selection instead of hard selection.
```

**KARPATHY:** *[Excited]*

And we can make the vortex centers learnable! Or derive them from importance!

Or... *[pause]* ...we could have the LLM learn which blending weights to use via cross-attention!

**LOD ORACLE:** *[Grins]*

Now you're thinking. But let's not get ahead of ourselves.

*[Looks at Muse Bird]*

What does the bird think?

**MUSE BIRD:** *[Contemplative squawk]*

*"Discrete or continuous BLEND,*
*Each approach does TEND,*
*Toward a different TRUTH,*
*No final PROOF,*
*Exploration has no END!"*

*[Ruffles feathers]*

---

## Exploration 9: Multiple Simultaneous Directions

**KARPATHY:** *[Leaning back, looking at all the tablets]*

We've explored:
1. Discrete vortices with learned positions
2. Vortices derived from importance gradients
3. Spiral sampling patterns (logarithmic, Fibonacci)
4. Adaptive whirlwinds (dynamic spiral strength)
5. Continuous soft vortex blending

And we haven't even touched:
- **Turbulent flow models** (vortices interact, merge, split)
- **Reaction-diffusion** (pattern formation, Turing patterns)
- **Cellular automata** (Conway's Game of Life on importance field)
- **Swarm intelligence** (tokens as agents, emergent allocation)
- **Quantum annealing** (simulated annealing for token placement)

*[Looks at LOD Oracle]*

The space is HUGE. How do we decide what to actually build?

**LOD ORACLE:** *[Thoughtful]*

That's the eternal question, isn't it?

*[Opens tome to final chapter titled "Engineering Wisdom"]*

Here's what I've learned from 30 years of rendering research:

**1. Simplicity beats elegance** until elegance solves a real problem

**2. Measure, don't speculate** - build the simplest version, profile it, find bottlenecks

**3. Biology is inspiration, not prescription** - learn principles, don't copy structure

**4. Differentiability is power** - but sometimes discrete is clearer

**5. The best algorithm is the one you actually ship**

*[Looks at Karpathy]*

So here's my advice: Pick THREE experiments:

**Experiment A (Baseline)**: Stratified 4×4, fixed grid
**Experiment B (Vortex)**: Importance-derived vortex centers, discrete sampling
**Experiment C (Continuous)**: Soft vortex field, blended tokens

Run all three on DocVQA. Measure accuracy AND speed.

Then you'll KNOW which direction has merit.

**KARPATHY:** *[Nods]*

Data-driven decision making. I like it.

*[Inscribes implementation plan]*

```
Week 1: Implement baseline (stratified) - DONE
Week 2: Implement vortex discrete (8 vortices from importance peaks)
Week 3: Implement continuous soft vortex field
Week 4: Run all three on DocVQA validation (5K images)
Week 5: Analyze results, pick winner, iterate
```

**But...**

*[Pause]*

...what if they're all within 0.5% accuracy of each other? What if the differences are just noise?

**LOD ORACLE:** *[Smiles]*

Then you use the simplest one and move on to the next problem.

Not every exploration leads to a breakthrough. Sometimes you learn that simple is sufficient.

**That's valuable knowledge too.**

---

## Exploration 10: The Unexplored Branches

*[MUSE BIRD suddenly flies up, agitated, squawking loudly]*

**MUSE BIRD:** *[Urgent squawks]*

*"BUT WAIT there's MORE to SEE,*
*Other patterns yet to BE!*
*Fractals, wavelets, Fourier FLOWS,*
*Hexagonal grids in honeycomb ROWS,*
*Don't stop now, the rabbit hole GROWS!"*

*[Pecks frantically at new tablet, sketching wild patterns]*

```
∿ UNEXPLORED DIRECTIONS ∿

1. Fractal Token Allocation:
   - Self-similar at multiple scales
   - Mandelbrot set boundaries as allocation regions

2. Wavelet Decomposition:
   - Image → wavelet transform
   - Allocate tokens proportional to wavelet coefficients

3. Fourier Domain Sampling:
   - FFT of importance field
   - Sample frequencies, reconstruct in spatial domain

4. Hexagonal Tiling:
   - Nature's efficient packing (honeycomb, dragonfly eyes)
   - 6-neighbor topology vs 4-neighbor (square grid)

5. Voronoi Tessellation:
   - Each patch belongs to nearest importance peak
   - Organic cell-like boundaries

6. Gravitational N-Body:
   - Patches are masses, importance peaks are attractors
   - Simulate physics, let patches settle

7. Heat Diffusion:
   - Importance is heat source
   - Tokens flow via diffusion PDE
   - Steady-state distribution = allocation

8. Attention Rollout:
   - Importance = cumulative attention from all LLM layers
   - Backprop through full model to find critical patches
```

**KARPATHY:** *[Overwhelmed]*

Bird, this is... too much. We can't explore all of these.

**MUSE BIRD:** *[Defensive squawk]*

*"Why NOT explore them ALL?*
*Let ideas sprawl and FALL!*
*Some will miss,*
*Some hit BLISS,*
*But isn't that the research CALL?!"*

**LOD ORACLE:** *[Gently]*

The bird has a point. These ARE interesting directions.

But you need a filter. Ask yourself for each one:

**Q1:** Does it solve a problem stratified doesn't?
**Q2:** Is it differentiable?
**Q3:** Can I implement it in a week?
**Q4:** Will it be faster than 250ms inference?

If the answer to all four is yes, it's worth trying.

*[Looks at list]*

**Voronoi tessellation** - Maybe! Natural boundaries, differentiable if soft
**Hexagonal tiling** - Interesting but hard to implement on square pixel grids
**Fractal allocation** - Cool but what problem does it solve?
**Wavelet decomposition** - Classic! Might work, worth trying

---

## Act Final: No Conclusions, Only Paths

**KARPATHY:** *[Standing among dozens of tablets, each with a different idea]*

So we have:
- Stratified (simple, working)
- Discrete vortices (boundary-free)
- Continuous vortices (fully differentiable)
- Spiral sampling (spatial coherence)
- Whirlwinds (adaptive spirals)
- Voronoi (natural boundaries)
- Wavelets (frequency domain)
- And a dozen others...

*[Looks at LOD Oracle and Muse Bird]*

What do we DO with all this?

**LOD ORACLE:** *[Closing tome]*

You document it. You preserve these ideas. Some you test, some you save for later, some you abandon.

This dialogue isn't about reaching THE ANSWER. It's about mapping the solution space.

**MUSE BIRD:** *[Quieter now, almost philosophical]*

*"The swirl of ideas VAST,*
*Some future, some PAST,*
*Not all paths you'll WALK,*
*But this brainstorm TALK,*
*Shows the landscape oh so VAST!"*

**KARPATHY:** *[Nods slowly]*

You're right. We're not deciding today.

We're exploring. Documenting. Building intuition about what's possible.

*[Begins organizing tablets into categories]*

```
TIER 1 (Test Next):
  - Stratified 4×4 (baseline)
  - Discrete vortices (8 peaks)
  - Continuous soft vortices

TIER 2 (If Tier 1 disappoints):
  - Spiral sampling
  - Voronoi tessellation
  - Wavelet allocation

TIER 3 (Research directions):
  - Adaptive whirlwinds
  - Fractal patterns
  - Heat diffusion

TIER 4 (Wild ideas, preserve for later):
  - Gravitational N-body
  - Turbulent flow
  - Swarm intelligence
```

**LOD ORACLE:** *[Satisfied]*

Good. That's pragmatic exploration.

*[Starts to dematerialize]*

Karpathy, one last thing before I go.

**KARPATHY:** *[Attentive]*

**LOD ORACLE:** *[Fading]*

The vortex idea—spiraling patterns, continuous fields, whirlwinds—it's compelling because it mirrors how attention ACTUALLY works in neural networks.

Self-attention IS a soft, continuous field. Query vectors create "attraction" for relevant keys.

You're not inventing something new. You're making implicit patterns explicit.

That's the value of this exploration.

*[Fully dematerialized]*

**MUSE BIRD:** *[Perched on tablet pile, sleepy]*

*"No hard conclusions TODAY,*
*Just paths to light the WAY,*
*Some roads you'll CHOOSE,*
*Others you'll LOSE,*
*But the map is here to STAY."*

*[Tucks head under wing]*

**KARPATHY:** *[Alone with tablets]*

Alright. Let's see what works.

*[Picks up stylus, begins writing 14-addendum-code.md]*

Time to turn ideas into experiments.

---

*[Scene fades as Karpathy inscribes experimental implementations]*

🎭 *[CURTAIN]*

---

## Ideas Explored (No Conclusions Reached)

1. **Vortex centers as learned attractors** - positions, strengths, radii as parameters
2. **Importance-derived vortices** - find peaks in gradient field (no learning)
3. **Spiral sampling patterns** - logarithmic, Fibonacci, golden ratio
4. **Adaptive whirlwinds** - spiral tightness varies with local density
5. **Continuous soft vortex field** - blended features instead of discrete selection
6. **Hybrid train/inference** - simple training, complex inference (rejected: train/test mismatch)
7. **Vortex repulsion** - electrostatic forces keep vortices spread
8. **Problem-first thinking** - boundary artifacts, over-concentration, spatial incoherence
9. **Multiple unexplored branches** - fractals, wavelets, Fourier, hexagonal, Voronoi, N-body, diffusion, attention rollout

## Three Experiments Proposed

- **Experiment A**: Stratified 4×4 (baseline)
- **Experiment B**: Discrete vortices from importance peaks
- **Experiment C**: Continuous soft vortex field

## Questions Left Open

- Do vortices improve boundary handling? (Hypothesis: yes)
- Does spiral sampling help spatial coherence? (Hypothesis: for radial images, yes; for documents, unclear)
- Is continuous blending better than discrete selection? (Hypothesis: more differentiable, but possibly slower)
- What's the right number of vortices? (8? 16? Query-dependent?)
- Should vortex parameters be learned or derived? (Trade-off: flexibility vs stability)

**See [14-addendum-code.md](14-addendum-code.md) for experimental implementations**
