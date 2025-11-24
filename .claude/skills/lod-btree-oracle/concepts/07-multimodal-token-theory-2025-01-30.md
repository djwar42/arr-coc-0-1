# Multimodal Token Theory: VAR and Laplacian Pyramids

**Date**: 2025-01-30
**Parent**: N/A (Theoretical foundations)
**Cross-Domain**: Deep learning generation models → VLM token allocation

---

## Overview

Recent theoretical work (Hong & Belkadi, arXiv 2510.02826, October 2025) reveals deep connections between **Visual Autoregressive (VAR) models** and **Laplacian pyramid decompositions**. This insight bridges classical computer vision (multi-scale pyramids) with modern generative AI (diffusion + autoregressive models), providing theoretical foundations for VLM token allocation strategies.

**Core insight**: VAR models implicitly construct Laplacian-style pyramids during generation, refining coarse-to-fine in frequency space. This explains why pyramid-based token allocation (PVC, FastVLM) works so well.

---

## VAR as Laplacian Diffusion

### Traditional View of VAR

**Visual Autoregressive (VAR)** models generate images by:
1. Predicting **next-scale** latent codes autoregressively
2. Starting from coarse scales (8×8) → fine scales (512×512)
3. Using discrete codebook (like VQGAN tokens)

### New Framework: Iterative Refinement

**Hong & Belkadi's key contribution**: Reframe VAR as:

```
VAR = Deterministic Forward Process (Laplacian pyramid construction)
    + Learned Backward Process (coarse-to-fine reconstruction)
    ≈ Discrete Diffusion Model in latent space
```

**Forward process** (deterministic):
1. Downsample image to multiple scales: 512×512 → 256×256 → 128×128 → 64×64...
2. Compute **Laplacian residuals** between adjacent scales
3. Quantize residuals into discrete codes (VQ codebook)

**Backward process** (learned):
1. Start with coarsest scale (64×64 code)
2. **Refine** by predicting residuals at next scale
3. Repeat until reaching target resolution

**This is identical to Laplacian pyramid reconstruction!**

---

## Three Design Choices Explaining VAR's Efficiency

### 1. Learned Latent Space (Not Pixel Space)

**Why it matters**:
- Operating in **VQ latent space** (discrete codes) is more compact than pixels
- 512×512 RGB image = 786,432 values
- 32×32 VQ codes = 1,024 tokens (768× reduction)

**Comparison**:
```
Pixel-space diffusion: Denoise 512×512×3 = 786K values
VAR latent diffusion: Predict 32×32 codes = 1K tokens

Speedup: ~768× fewer operations per step
```

**VLM analogy**: Visual tokens (768-dim) are more efficient than raw pixels for reasoning.

### 2. Discrete Classification (Not Continuous Prediction)

**VAR predicts discrete code indices** (classification):
```python
# VAR prediction head
logits = model(current_scale)  # Shape: (batch, vocab_size, H, W)
next_codes = torch.argmax(logits, dim=1)  # Discrete indices
```

**Why discrete wins**:
- **Sharper predictions**: Softmax over discrete codes → confident choices
- **No mode collapse**: Can't average to blurry middle ground
- **Faster convergence**: Categorical cross-entropy is simpler than continuous regression

**Continuous alternative** (diffusion):
```python
# Continuous diffusion (slower, blurrier)
noise_pred = model(noisy_image, timestep)
denoised = noisy_image - noise_pred  # Continuous values
```

**VLM relevance**: Discrete visual tokens (VQGAN, DALL-E) enable faster, sharper reasoning.

### 3. Spatial Frequency Partitioning

**VAR partitions generation by frequency bands**:
- **Low frequencies** (coarse scales): Generate first (fast, high certainty)
- **High frequencies** (fine scales): Generate last (slow, low certainty)

**This matches Laplacian pyramid decomposition**:
```
Level 0 (512×512): High-frequency details
Level 1 (256×256): Mid-high frequencies
Level 2 (128×128): Mid frequencies
Level 3 (64×64):   Low frequencies
```

**Generation order** (coarse-to-fine):
```
Step 1: Generate 64×64 (low freq, easy, fast)
Step 2: Refine to 128×128 (add mid freq details)
Step 3: Refine to 256×256 (add mid-high freq)
Step 4: Refine to 512×512 (add high freq details)
```

**Why this is efficient**:
- **Low frequencies** contain most information (easy to predict)
- **High frequencies** are sparse corrections (hard to predict, but small impact)

---

## Implications for VLM Token Allocation

### Frequency-Aware Token Budgets

**Idea**: Allocate tokens based on **frequency content** of each patch.

**Laplacian pyramid analysis**:
```python
def frequency_aware_allocation(image, budget=4096):
    """Allocate tokens based on Laplacian pyramid energy."""

    # Build 4-level Laplacian pyramid
    pyramid = laplacian_pyramid(image, levels=4)

    patches = split_into_patches(image, patch_size=32)
    allocations = []

    for patch in patches:
        patch_pyramid = laplacian_pyramid(patch, levels=4)

        # Compute energy per frequency band
        energy = [np.sum(level**2) for level in patch_pyramid]

        # High-freq patches get more tokens (contain fine details)
        high_freq_energy = energy[0] + energy[1]  # Levels 0-1
        low_freq_energy = energy[2] + energy[3]   # Levels 2-3

        if high_freq_energy > 2 * low_freq_energy:
            tokens = int(budget * 0.05)  # 5% of budget (high detail)
        else:
            tokens = int(budget * 0.01)  # 1% of budget (low detail)

        allocations.append(tokens)

    # Normalize to total budget
    total = sum(allocations)
    allocations = [int(t * budget / total) for t in allocations]

    return allocations
```

**Performance expectation**:
- **Simple patches** (sky, walls): 64 tokens (low-freq, easy to compress)
- **Complex patches** (text, foliage): 400 tokens (high-freq, hard to compress)

### Progressive Token Loading

**Inspired by VAR's coarse-to-fine generation**:

```python
class ProgressiveVLM:
    def __init__(self):
        self.scales = [64, 128, 256, 512]  # Pyramid levels

    def forward(self, image, query):
        # Level 0: Coarse understanding (64×64 → 256 tokens)
        coarse_tokens = self.encode_coarse(image)  # Fast
        coarse_response = self.llm(query, coarse_tokens)

        # If confident, stop early
        if coarse_response.confidence > 0.9:
            return coarse_response

        # Level 1: Refine with mid-scale (128×128 → 1024 tokens)
        mid_tokens = self.encode_mid(image)
        mid_response = self.llm(query, mid_tokens)

        if mid_response.confidence > 0.8:
            return mid_response

        # Level 2: Full resolution if needed (512×512 → 4096 tokens)
        full_tokens = self.encode_full(image)
        full_response = self.llm(query, full_tokens)

        return full_response
```

**Benefits**:
- **Early exit** for simple queries (64× token reduction)
- **Adaptive depth** based on query difficulty
- **Mirrors VAR's frequency partitioning**

---

## Discrete Latent Codes → Discrete Visual Tokens

### VAR Uses Discrete Codes (VQ-VAE)

**Quantization process**:
```python
# Encode image to continuous latent
z = encoder(image)  # Shape: (batch, 256, 32, 32)

# Quantize to discrete codes
distances = torch.cdist(z, codebook)  # codebook: (vocab_size, 256)
codes = torch.argmin(distances, dim=-1)  # Shape: (batch, 32, 32)

# Lookup quantized vectors
z_quantized = codebook[codes]  # Shape: (batch, 32, 32, 256)
```

**Why discrete wins** (from paper):
1. **Sharper predictions**: Softmax classification → confident choices
2. **Faster training**: Cross-entropy converges faster than MSE
3. **Better compression**: Discrete codes = compact representation

### VLM Visual Tokens Should Be Discrete

**Current VLMs** (continuous):
```python
# CLIP-style continuous tokens
visual_tokens = vision_encoder(image)  # Shape: (batch, 196, 768)
# Tokens are continuous 768-dim vectors
```

**Proposed discrete alternative**:
```python
# VQ-style discrete tokens
z = vision_encoder(image)
codes = vq_quantize(z, codebook_size=8192)  # Discrete indices

# Lookup discrete embeddings
visual_tokens = embedding_table[codes]  # Shape: (batch, 196, 768)
```

**Expected benefits**:
- **Faster inference**: Discrete lookups cheaper than continuous projection
- **Sharper reasoning**: No blurry averaging in latent space
- **Better compression**: 8192 vocab vs infinite continuous space

---

## Theoretical Connections

### VAR ≈ Discrete Diffusion in Latent Space

**Standard diffusion** (continuous, pixel space):
```
x₀ → add noise → x_t → denoise → x₀
```

**VAR reframed** (discrete, latent space):
```
codes₀ → downsample → codes_coarse → upsample+refine → codes₀
```

**Key equivalence**:
- **Downsampling** = Adding noise (removes high frequencies)
- **Upsampling + refining** = Denoising (recovers high frequencies)
- **Discrete codes** = Quantized latent space (sharper than continuous)

### Laplacian Pyramid as Frequency Diffusion

**Laplacian pyramid levels** = **Frequency bands**:
```
L₀ (high-freq) ← Hardest to predict (fine details)
L₁ (mid-freq)
L₂ (low-freq)  ← Easiest to predict (coarse structure)
```

**Diffusion analogy**:
```
High noise    ← Hard to denoise (like high-freq)
Mid noise
Low noise     ← Easy to denoise (like low-freq)
```

**This explains why coarse-to-fine works**:
- **Coarse (low-freq)**: Contains most information, easy to predict
- **Fine (high-freq)**: Sparse corrections, hard but small impact

---

## Practical Applications

### 1. Hierarchical VLM Architecture

**Inspired by VAR's scale hierarchy**:

```python
class HierarchicalVLM(nn.Module):
    def __init__(self):
        self.coarse_encoder = VisionEncoder(resolution=64)
        self.mid_encoder = VisionEncoder(resolution=128)
        self.fine_encoder = VisionEncoder(resolution=256)

    def forward(self, image, query):
        # Level 0: Coarse (64×64 → 256 tokens)
        coarse = self.coarse_encoder(downsample(image, 64))

        # Level 1: Mid residuals (128×128 → 512 tokens)
        mid_full = self.mid_encoder(downsample(image, 128))
        mid_residual = mid_full - upsample(coarse, 128)

        # Level 2: Fine residuals (256×256 → 1024 tokens)
        fine_full = self.fine_encoder(downsample(image, 256))
        fine_residual = fine_full - upsample(mid_full, 256)

        # Adaptive fusion based on query
        if query_needs_detail(query):
            tokens = concat(coarse, mid_residual, fine_residual)
        else:
            tokens = coarse  # Early exit

        return self.llm(query, tokens)
```

### 2. Frequency-Stratified Training

**Train VLM with pyramid-based loss**:

```python
def pyramid_loss(pred, target):
    """Multi-scale perceptual loss (like LapLoss)."""

    pred_pyramid = laplacian_pyramid(pred, levels=4)
    target_pyramid = laplacian_pyramid(target, levels=4)

    losses = []
    weights = [1.0, 0.5, 0.25, 0.125]  # More weight to high-freq

    for pred_level, target_level, w in zip(pred_pyramid, target_pyramid, weights):
        loss = F.l1_loss(pred_level, target_level)
        losses.append(w * loss)

    return sum(losses)
```

**Why this helps**:
- **Multi-scale gradients**: Better training signal at all frequencies
- **Prevents blurring**: High-freq levels penalize loss of detail
- **Aligns with VAR theory**: Matches frequency partitioning

### 3. Discrete Token Vocabulary

**Replace continuous CLIP tokens with discrete VQ tokens**:

```python
class VQVisualEncoder(nn.Module):
    def __init__(self, vocab_size=8192, dim=768):
        self.encoder = VisionTransformer()
        self.codebook = nn.Embedding(vocab_size, dim)

    def forward(self, image):
        z = self.encoder(image)  # (batch, 196, 768)

        # Vector quantization
        distances = torch.cdist(z, self.codebook.weight)
        codes = torch.argmin(distances, dim=-1)  # (batch, 196)

        # Straight-through estimator (for backprop)
        z_q = self.codebook(codes)
        z_q = z + (z_q - z).detach()  # Copy gradients

        return z_q, codes
```

**Expected gains**:
- **Sharper features**: Discrete codes prevent blurry averaging
- **Faster inference**: Embedding lookup cheaper than projection
- **Better compression**: 8K vocab vs continuous space

---

## Future Directions

**VAR-inspired VLM innovations**:

1. **Laplacian token hierarchies**: Encode images as pyramid, allocate tokens per level
2. **Frequency-stratified attention**: Separate attention heads for different frequency bands
3. **Coarse-to-fine inference**: Start with low-res, refine only if needed
4. **Discrete visual vocabularies**: Replace continuous CLIP with VQ tokens
5. **Pyramid-based pre-training**: Use LapLoss during masked auto-encoding

**Open questions**:
- Can VLMs directly predict Laplacian residuals (like VAR)?
- Should visual tokens be discrete (VQ) or continuous (CLIP)?
- How much speedup from coarse-to-fine early-exit strategies?
- Can pyramid-based training improve zero-shot transfer?

---

## Cross-References

**Related oracle files**:
- [../algorithms/06-image-pyramid-multiscale-2025-01-30.md](../algorithms/06-image-pyramid-multiscale-2025-01-30.md) - Laplacian pyramids, steerable pyramids
- [../techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md](../techniques/00-foveated-rendering-03-02-progressive-compression-2025-01-30.md) - PVC and FastVLM (pyramid-based VLMs)

**Integration concepts**:
- [../integration/03-query-aware-relevance-2025-01-30.md](../integration/03-query-aware-relevance-2025-01-30.md) - Adaptive token allocation

---

## References

**Core Paper**:
- Hong & Belkadi (October 2025): "Multi-scale Autoregressive Models are Laplacian, Discrete, and Latent Diffusion Models in Disguise" - arXiv 2510.02826

**Related Work**:
- Burt & Adelson (1983): "The Laplacian Pyramid as a Compact Image Code"
- Van Den Oord et al. (2017): "Neural Discrete Representation Learning" (VQ-VAE)
- Tian et al. (2024): "Visual Autoregressive Modeling" (VAR original paper)

**VLM Applications**:
- CVPR 2025: "Progressive Visual Compression" (PVC) - Pyramid-based token allocation
- CVPR 2025: "FastVLM" (Apple ML) - Difficulty-aware pyramid sampling
- arXiv 2503.05974 (March 2025): "LapLoss: Laplacian Pyramid-based Multiscale loss"

---

## Summary

**Key takeaways**:

1. **VAR = Laplacian diffusion**: VAR models implicitly construct pyramids, explaining their efficiency
2. **Three design choices matter**: Learned latent space, discrete classification, frequency partitioning
3. **VLMs should adopt pyramid thinking**: Hierarchical tokens, frequency-aware budgets, coarse-to-fine inference
4. **Discrete tokens may be superior**: Sharper predictions, faster training, better compression
5. **Theory guides practice**: Understanding VAR→pyramid connection informs VLM architecture design

**Practical impact**: This theoretical framework provides rigorous justification for pyramid-based VLM token allocation strategies (PVC, FastVLM), and suggests concrete improvements (discrete tokens, hierarchical architectures, pyramid loss functions).
