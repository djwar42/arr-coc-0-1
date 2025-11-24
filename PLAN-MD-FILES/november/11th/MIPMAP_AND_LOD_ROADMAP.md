# Mipmap & LOD Hierarchical Sampling Roadmap

**A synthesis of discussions from Parts 28, 33, 44, 45, 46 on when and how to implement multi-resolution foveation**

---

## 🎯 The Strategic Decision

**ARR-COC v0.1 (MVP)**: Single-resolution 32×32 grid
**ARR-COC v0.2**: Mipmap pyramids + hierarchical LOD sampling

**Why delay mipmaps to v0.2?** The MVP must validate the CORE HYPOTHESIS first: Does query-aware relevance realization work at all?

Once proven, add multi-scale foveation for true biological vision emulation.

---

## 📚 What the Dialogues Say

### Part 33: "The Mipmap Illusion" - Software is Fast Enough

**Key Insight**: PyTorch doesn't expose hardware mipmaps, but software mipmaps via `F.avg_pool2d()` are **0.5ms** - fast enough!

```python
# Software mipmaps (PyTorch native)
texture = torch.randn(40, 1024, 1024, device='cuda')

mip1 = F.avg_pool2d(texture, 2)  # 512×512
mip2 = F.avg_pool2d(mip1, 2)     # 256×256
mip3 = F.avg_pool2d(mip2, 2)     # 128×128

# Cost: ~0.5ms on A100
# vs Hardware mipmaps: 0.1ms BUT 10-20ms interop overhead
```

**Conclusion**: Skip OpenGL/CUDA interop complexity. Use PyTorch-native downsampling.

**Decision**: Software mipmaps for v0.2 ✓

---

### Part 28-1: Position Channels + Mipmaps - The Magic

**Key Insight**: Position channels (x, y, eccentricity) downsample **intelligently**!

```python
# Level 0: 1024×1024
position[512, 512] = (0.5, 0.5)  # Center pixel

# Level 1: 512×512 (downsampled via avg_pool2d)
# Position at [256, 256] = average of 4 pixels from Level 0
# = CENTER of the 2×2 region!

# Level 4: 64×64
# Each pixel represents 16×16 region
# Position channel AUTOMATICALLY gives you region center
```

**Why This Matters**:
- Coarse scanning at level 4 (64×64) naturally samples region centers
- No manual offset calculation
- Mipmap cascade gets correct coordinates for free

**Cost**: 0.001ms (literally free)

**Decision**: Position channels (channels 3-5) included in MVP ✓

---

### Part 44: LOD Oracle - Hierarchical Sampling is v0.2

**The LOD Oracle speaks** (lines 518-636):

#### Option A: Sparse Sampling (MVP)
```python
# Select 200 patches from 32×32 grid (1024 total)
# Patches can be anywhere
# No spatial structure guaranteed

selected_indices = torch.topk(relevance_scores, k=200).indices
```

**Pros**: Simple, differentiable, validates core hypothesis
**Cons**: No spatial coherence, no multi-resolution

#### Option B: Hierarchical LOD (v0.2)
```python
# Quad-tree: Divide image recursively based on relevance variance

def allocate_quadtree(relevance_scores, budget):
    """
    High variance region → split into 4 quadrants
    Low variance region → keep as single region
    """
    if budget <= 1 or region.variance() < threshold:
        return [region]  # Base case

    # Split into 4 quadrants, distribute budget
    sub_budgets = distribute_budget(budget, 4, relevance_scores)
    return [
        allocate_quadtree(top_left, sub_budgets[0]),
        allocate_quadtree(top_right, sub_budgets[1]),
        allocate_quadtree(bottom_left, sub_budgets[2]),
        allocate_quadtree(bottom_right, sub_budgets[3])
    ]
```

**Benefits**:
1. **Spatial coherence** - Regions stay contiguous (Qwen3VL M-RoPE friendly)
2. **Interpretability** - "This region is important" vs "These 17 random patches"
3. **Biological plausibility** - Mimics foveal cascade (coarse → fine)

**LOD Oracle's verdict** (line 626):
> "For MVP: **Sparse sampling** (your current plan).
> For v0.2: **Hierarchical LOD** (quad-tree).
> Start simple. Add sophistication later."

**LOD Oracle's promise** (line 1438):
> "And when you're ready for hierarchical LOD in v0.2, call upon me. The quad-tree approach will scale naturally from your sparse selection."

---

### Part 45: MVP Specification - Single Resolution Only

**Texture Array Spec** (lines 30-206):
```python
def generate_texture_array(image, target_size=32):
    """
    Generate 13-channel texture array from RGB image.

    Args:
        target_size: Output spatial resolution (32 → [B, 13, 32, 32])

    Returns:
        textures: [B, 13, 32, 32] - SINGLE RESOLUTION
    """
    # Channels 0-2: RGB
    # Channels 3-4: LAB L* and a*
    # Channels 5-7: Sobel edges
    # Channels 8-9: Spatial position (y, x)
    # Channels 10: Eccentricity
    # Channels 11-12: Reused (saliency, luminance)

    # NO mipmaps - just 32×32 single resolution
```

**Why no mipmaps in MVP?**
- Validate core hypothesis first: Does query-aware allocation work?
- Simplicity: 32×32 grid is easy to visualize and debug
- Speed: Single resolution is fastest for initial experiments

**Build checklist** (lines 988-1043): No mention of mipmaps - intentional!

---

### Part 46: MVP Be Doing - Implementation Reality

**Repository structure**:
```
46-mvp-be-doing/
└── code/
    └── arr-coc-0-1/        # v0.1 implementation
        ├── arr_coc/
        │   ├── texture.py  # Single-resolution 32×32
        │   ├── knowing.py
        │   ├── balancing.py
        │   └── attending.py
        └── README.md
```

**No mipmap implementation** - focus on getting the pipeline working first.

---

## 🗺️ The Roadmap: v0.1 → v0.2

### v0.1 (MVP) - VALIDATE CORE HYPOTHESIS

**Scope**:
```
Single-resolution 32×32 grid (1024 patches)
    ↓
Three Ways of Knowing (info, persp, partic)
    ↓
Adaptive Balancing (learned tensions)
    ↓
Top-K=200 selection (sparse sampling)
    ↓
Qwen3-VL integration
```

**Goal**: Prove that query-aware relevance realization produces semantically meaningful token allocation.

**Timeline**: 6 weeks
- Week 1-2: Build core components
- Week 3-4: Train on VQAv2
- Week 5-6: Validate with Gradio microscope

**Success metric**: Homunculus shows query-aware selectivity (not random, not just edges)

---

### v0.2 (Hierarchical Foveation) - SCALE UP

**Add after MVP validation**:

#### 1. Mipmap Pyramid Generation

```python
class TextureArrayWithMipmaps:
    def generate(self, image):
        """Generate texture with 5-level mipmap pyramid"""
        base = self.generate_channels(image)  # [13, 1024, 1024]

        # Generate pyramid
        mipmaps = [base]
        current = base
        for level in range(4):
            current = F.avg_pool2d(current, kernel_size=2)
            mipmaps.append(current)

        # Level 0: 1024×1024 (fine detail)
        # Level 1: 512×512
        # Level 2: 256×256
        # Level 3: 128×128
        # Level 4: 64×64 (coarse scan)

        return mipmaps  # Memory: 1.33× overhead, Time: +0.5ms
```

#### 2. Coarse-to-Fine Cascade

```python
def realize_relevance_cascade(self, image, query):
    """
    Multi-scale relevance realization.

    Mimics biological vision:
    - Coarse scan (peripheral): 64×64, fast, low-res
    - Refinement (parafoveal): 256×256, medium-res around ROI
    - Fine detail (foveal): 1024×1024, high-res where it matters
    """
    mipmaps = self.texture_array.generate(image)

    # Stage 1: Coarse scan (level 4: 64×64)
    coarse_scores = self.score_knowing(mipmaps[4], query)
    top_64_regions = torch.topk(coarse_scores.flatten(), k=64).indices

    # Stage 2: Refine (level 2: 256×256)
    # Sample 4×4 neighborhood around each top region
    candidates_medium = expand_to_neighbors(top_64_regions, scale=4)
    medium_scores = self.score_knowing(
        mipmaps[2][candidates_medium],
        query
    )
    top_256 = torch.topk(medium_scores, k=256).indices

    # Stage 3: Fine detail (level 0: 1024×1024)
    # Sample 4×4 neighborhood around top medium patches
    candidates_fine = expand_to_neighbors(top_256, scale=4)
    fine_scores = self.score_knowing(
        mipmaps[0][candidates_fine],
        query
    )

    # Final selection: Top 200 at full resolution
    selected_indices = torch.topk(fine_scores, k=200).indices
    return selected_indices
```

#### 3. Quad-Tree Hierarchical Allocation (Advanced)

```python
class QuadTreeAllocator:
    """
    Hierarchical LOD allocation via quad-tree.

    Adaptively subdivide regions based on relevance variance:
    - Uniform regions: Large patches, few tokens
    - High-variance regions: Small patches, many tokens
    """

    def allocate(self, relevance_scores, budget=200):
        """
        Build quad-tree and allocate tokens hierarchically.

        Returns:
            regions: List of (y, x, size, num_tokens) tuples
        """
        # Start with full image as single region
        root = Region(y=0, x=0, size=32, scores=relevance_scores)

        # Recursively subdivide
        tree = self.subdivide(root, budget)

        return self.flatten_tree(tree)

    def subdivide(self, region, budget):
        """Recursive quad-tree subdivision"""
        if budget <= 1 or region.variance() < threshold:
            return LeafNode(region, tokens=budget)

        # Split into 4 quadrants
        quads = region.split_into_four()
        sub_budgets = self.distribute_budget(budget, quads)

        return QuadNode([
            self.subdivide(quads[0], sub_budgets[0]),
            self.subdivide(quads[1], sub_budgets[1]),
            self.subdivide(quads[2], sub_budgets[2]),
            self.subdivide(quads[3], sub_budgets[3])
        ])
```

#### 4. Variable Token Budgets (64-400 per patch)

```python
class AdaptiveTokenAllocator:
    """
    Allocate variable token budgets based on relevance.

    High-relevance patches: 400 tokens (16×25 features)
    Medium-relevance patches: 200 tokens (16×12.5)
    Low-relevance patches: 64 tokens (16×4 features)
    """

    def allocate(self, relevance_scores, total_budget=5120):
        """
        Distribute total_budget across patches based on relevance.

        Returns:
            selected_patches: [N] indices
            token_budgets: [N] tokens per patch (64-400 range)
        """
        # Normalize relevance to [0, 1]
        rel_norm = (relevance_scores - relevance_scores.min()) / \
                   (relevance_scores.max() - relevance_scores.min())

        # Map to token budget range [64, 400]
        budgets = 64 + rel_norm * (400 - 64)

        # Select patches until budget exhausted
        sorted_idx = torch.argsort(budgets, descending=True)

        selected = []
        cumulative = 0
        for idx in sorted_idx:
            if cumulative + budgets[idx] <= total_budget:
                selected.append(idx)
                cumulative += budgets[idx]
            else:
                break

        return selected, budgets[selected]
```

---

### v0.2 Performance Impact

**Current (v0.1 - Single Resolution)**:
```
Generate 13-channel texture (32×32):  ~5ms
Score all 1024 patches:               ~1ms
Select top 200:                       <0.1ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                                ~6ms per image
```

**Future (v0.2 - Mipmap Cascade)**:
```
Generate 13-channel texture (1024×1024): ~10ms
Generate 5 mipmap levels:                 ~0.5ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1 - Coarse (64×64):   Score 4096    ~2ms
Stage 2 - Medium (256×256): Score 256     ~0.5ms
Stage 3 - Fine (1024×1024): Score 256     ~0.5ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                                    ~14ms per image
```

**Trade-off**:
- 2.3× slower
- But full 1024×1024 resolution (16× more detail)
- Hierarchical cascade (biologically plausible)
- Better token allocation (focused on relevant regions)

---

## 🎓 Why This Sequence?

### MVP First (v0.1)
**Question**: Does relevance realization work at all?

If ARR-COC v0.1 shows:
✓ Query-aware token allocation
✓ Homunculus highlights semantically relevant patches
✓ Better than uniform baseline

**Then proceed to v0.2.**

If ARR-COC v0.1 shows:
✗ Random/edge-only allocation
✗ No query-awareness
✗ Same as baseline

**Then fix the core algorithm before adding complexity.**

### Mipmaps Second (v0.2)
**Question**: Does multi-scale foveation improve efficiency?

After v0.1 validation, add mipmaps to:
- Increase resolution (32×32 → 1024×1024)
- Add coarse-to-fine cascade (biological vision)
- Enable variable token budgets (64-400 per patch)
- Support hierarchical LOD (quad-trees)

**Don't build v0.2 before v0.1 proves the concept!**

---

## 📊 Hypotheses Timeline

### H1-H4: Test in v0.1 (MVP)
- H1: Query-aware beats uniform
- H2: Three ways provide complementary info
- H3: Adaptive balancing > fixed weights
- H4: Top-K captures relevant regions

### H5-H6: Test in v0.2 (Post-MVP)
- H5: 13-channel texture > RGB-only
- H6: LOD compression improves efficiency ← **Mipmaps!**

From `HYPOTHESES_FOR_VALIDATION.md` (line 112):
> **H6: LOD Compression Would Improve Efficiency**
>
> **Claim**: Variable token budgets (64-400) per patch based on relevance would beat fixed 200.
>
> **Test**: Post-MVP (not in 0.1)

---

## 🔮 v0.3+ Future Directions

After mipmaps are validated:

**Temporal Mipmaps** (Video):
```python
# 3D mipmap pyramid (time, height, width)
mipmaps_temporal = []
for t in range(num_frames):
    frame_pyramid = generate_mipmaps(video[t])
    mipmaps_temporal.append(frame_pyramid)

# Coarse temporal scan: Sample every 4th frame at low-res
# Fine temporal scan: Sample every frame at high-res for ROI
```

**Learned Mipmap Filters** (Beyond avg_pool2d):
```python
class LearnedDownsampler(nn.Module):
    """
    Replace F.avg_pool2d with learned convolution.

    Preserves query-relevant features better than simple averaging.
    """
    def __init__(self, in_channels=13):
        self.downsample = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=2, padding=1
        )
```

**Attention-Weighted Mipmaps**:
```python
# Downsample using query-aware attention weights
# Not just average, but relevance-weighted average
mip1 = downsample_with_attention(texture, query_embeds)
```

---

## 📝 Summary: The Path Forward

**Now (v0.1 MVP)**:
```
✓ Position channels (ch 3-5) with eccentricity
✓ Single-resolution 32×32 grid
✓ Sparse top-K=200 selection
✓ Validate core hypothesis
```

**Next (v0.2 Mipmaps)**:
```
□ Software mipmaps (F.avg_pool2d, +0.5ms)
□ 5-level pyramid (1024 → 512 → 256 → 128 → 64)
□ Coarse-to-fine cascade (3-stage sampling)
□ Variable budgets (64-400 tokens per patch)
```

**Future (v0.3+ Advanced)**:
```
□ Quad-tree hierarchical allocation
□ Temporal mipmaps (video)
□ Learned downsampling filters
□ Attention-weighted mipmaps
```

---

## 🎯 Key Takeaway

**Mipmaps are NOT forgotten** - they're strategically deferred to v0.2!

The dialogues make this explicit:
- **Part 33**: Software mipmaps are fast enough (0.5ms)
- **Part 28**: Position channels downsample intelligently (free!)
- **Part 44**: LOD Oracle endorses hierarchical sampling for v0.2
- **Part 45/46**: MVP intentionally single-resolution

**Build v0.1 → Validate → Then add v0.2 mipmaps.**

That's the roadmap. ∿◇∿
