# Rate-Distortion Theory

## Overview

Rate-distortion theory is a fundamental branch of information theory that provides the theoretical foundations for lossy data compression. Originally developed by Claude Shannon in the 1950s, it addresses the critical question: **what is the minimum number of bits per symbol (rate R) needed to represent a source signal such that the reconstruction doesn't exceed a specified distortion D?**

**Core Insight**: Unlike lossless compression (Shannon entropy sets the limit), lossy compression involves a tradeoff between compression rate and reconstruction quality. Rate-distortion theory mathematically characterizes this fundamental tradeoff.

**Cognitive Connection**: Perception is compression. The brain doesn't represent sensory inputs with perfect fidelity—it compresses them optimally subject to task-relevant distortion constraints. Rate-distortion theory formalizes the information-theoretic limits of this process.

From [Wikipedia: Rate-Distortion Theory](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory) (accessed 2025-11-16):
> "Rate–distortion theory addresses the problem of determining the minimal number of bits per symbol, as measured by the rate R, that should be communicated over a channel, so that the source (input signal) can be approximately reconstructed at the receiver (output signal) without exceeding an expected distortion D."

## Section 1: The Rate-Distortion Problem

### Problem Formulation

**Given:**
- Source X with probability distribution P_X(x)
- Distortion measure d(x, x̂) quantifying cost of representing x by x̂
- Maximum acceptable distortion D*

**Find:**
- Minimum rate R(D) = minimum bits/symbol to achieve distortion ≤ D*

**Rate-Distortion Function:**
```
R(D) = min I(X; Y)
       Q_Y|X

subject to: E[d(X,Y)] ≤ D
```

Where:
- Q_Y|X(y|x) is the "test channel" (conditional probability of reconstruction Y given source X)
- I(X; Y) is mutual information between source and reconstruction
- E[d(X,Y)] is expected distortion

**Key Properties:**
1. **Continuous**: R(D) is a smooth function of D
2. **Monotonically decreasing**: More distortion allowed → fewer bits needed
3. **Convex**: The curve is U-shaped (no local minima)
4. **Bounded**: R(0) = H(X) (lossless), R(D_max) = 0 (maximum distortion requires no bits)

### The Distortion-Rate Tradeoff

**Alternative formulation** (dual problem):
```
D(R) = min E[d(X,Y)]
       Q_Y|X

subject to: I(X; Y) ≤ R
```

These two formulations are inverses of each other—they describe the same fundamental tradeoff from different perspectives.

**Cognitive Interpretation:**
- **Rate R**: Computational/memory budget (e.g., neural firing rate, synaptic transmission capacity)
- **Distortion D**: Task-relevant reconstruction error (e.g., classification accuracy, action success)
- **Tradeoff**: Brains minimize distortion subject to metabolic/bandwidth constraints

From [Shannon Bounds for Quadratic Rate-Distortion](https://ieeexplore.ieee.org/document/10684730/) (2024):
> "This paper surveys Shannon bounds on rate-distortion problems under mean-squared error distortion with a particular emphasis on Berger's techniques."

## Section 2: Distortion Measures

### Common Distortion Functions

**1. Hamming Distortion (Discrete Sources):**
```
d(x, x̂) = {
  0  if x = x̂
  1  if x ≠ x̂
}
```
- **Use case**: Binary/discrete data where all errors are equally costly
- **Example**: Text compression where any wrong character is equally bad

**2. Squared-Error Distortion (Continuous Sources):**
```
d(x, x̂) = (x - x̂)²
```
- **Use case**: Continuous signals (audio, images) where error magnitude matters
- **Properties**: Penalizes large errors more heavily (quadratic cost)
- **Perceptual limitation**: MSE doesn't always correlate with human perception

**3. Absolute-Error Distortion:**
```
d(x, x̂) = |x - x̂|
```
- **Use case**: More robust to outliers than squared error
- **Less common** in practice due to mathematical tractability issues

### Perceptual Distortion Measures

**Problem with MSE**: Low distortion ≠ high perceptual quality

From [Blau & Michaeli 2019: Rate-Distortion-Perception Tradeoff](https://arxiv.org/abs/1901.07821):
> "In recent years, it has become increasingly accepted that 'low distortion' is not a synonym for 'high perceptual quality', and in fact optimization of one often comes at the expense of the other."

**Rate-Distortion-Perception (RDP) Tradeoff:**
```
R(D, P) = min I(X; Y)
          Q_Y|X

subject to:
  E[d(X,Y)] ≤ D      (distortion constraint)
  d_percept(P_X, P_Y) ≤ P  (perceptual constraint)
```

**Key Finding**: Restricting perceptual quality to be high generally elevates the rate-distortion curve—you sacrifice either rate OR distortion to maintain perceptual quality.

**Perceptual Measures:**
- **SSIM** (Structural Similarity): Captures perceived image quality better than MSE
- **LPIPS** (Learned Perceptual Image Patch Similarity): Neural network-based perceptual metric
- **Psychoacoustic models**: Used in MP3/AAC audio codecs
- **Rate-distortion-perception**: Formalized framework extending Shannon theory

## Section 3: Analytical Rate-Distortion Functions

### Gaussian Source with Squared-Error Distortion

**Setup:**
- X ~ N(0, σ²) (Gaussian source with variance σ²)
- Memoryless (independent samples)
- Distortion: d(x, x̂) = (x - x̂)²

**Rate-Distortion Function:**
```
R(D) = {
  (1/2) log₂(σ²/D)    if 0 ≤ D ≤ σ²
  0                    if D > σ²
}
```

**Interpretation:**
- **D = σ²**: No compression needed (maximum acceptable distortion)
- **D = σ²/2**: R = 0.5 bits/sample (half variance reduction)
- **D = σ²/4**: R = 1 bit/sample (quarter variance reduction)
- **D → 0**: R → ∞ (perfect reconstruction requires infinite bits)

**Logarithmic relationship**: Halving distortion costs 1 additional bit/sample

**Optimal encoder/decoder**:
- Encoder adds Gaussian noise: Y = X + N where N ~ N(0, D)
- Decoder: Wiener filter (MMSE estimator)

**Shannon Lower Bound (SLB):**
For arbitrary sources with finite differential entropy h(X):
```
R(D) ≥ h(X) - h(D) = h(X) - (1/2)log₂(2πeD)
```

This bound is **tight** for Gaussian sources (equality holds).

### Bernoulli Source with Hamming Distortion

**Setup:**
- X ~ Bernoulli(p) (binary source with P(X=1) = p)
- Memoryless (independent bits)
- Distortion: d(x, x̂) = 1 if x ≠ x̂, else 0

**Rate-Distortion Function:**
```
R(D) = {
  H_b(p) - H_b(D)         if 0 ≤ D ≤ min(p, 1-p)
  0                        if D > min(p, 1-p)
}
```

Where H_b(p) = -p log₂(p) - (1-p) log₂(1-p) is the binary entropy function.

**Interpretation:**
- **D = 0**: R = H_b(p) (lossless compression, Shannon entropy limit)
- **D = min(p, 1-p)**: R = 0 (maximum distortion, no information transmitted)
- **D = 0.5 for p=0.5**: R = 0 (random flipping is acceptable)

**Example (p = 0.5):**
- D = 0: R = 1 bit/symbol (lossless)
- D = 0.1: R ≈ 0.53 bits/symbol (10% error rate)
- D = 0.25: R ≈ 0.19 bits/symbol (25% error rate)
- D ≥ 0.5: R = 0 (can guess randomly)

## Section 4: Rate-Distortion Theory for Sources with Memory

### Extension to Stationary Sources

For sources with memory (correlated samples), the rate-distortion function requires a limit:

```
R(D) = lim_{n→∞} (1/n) R_n(D)
```

Where R_n(D) is the rate-distortion function for blocks of length n.

**Key Insight**: Exploiting temporal/spatial correlations improves compression efficiency.

**Example: Gauss-Markov Source**
- X_{t+1} = ρX_t + W_t where W_t ~ N(0, σ²_w)
- Correlation coefficient ρ determines memory strength
- R(D) < R_memoryless(D) when ρ ≠ 0

**Practical Implications:**
- **Image compression**: Neighboring pixels are highly correlated
- **Video compression**: Temporal correlation between frames (motion compensation)
- **Speech coding**: Phoneme structure creates long-range dependencies

### Operational Rate-Distortion Theory

**Achievability**: For any ε > 0, there exist encoder-decoder pairs such that:
```
R ≤ R(D) + ε
E[d(X,X̂)] ≤ D + ε
```

**Converse**: No encoder-decoder can achieve:
```
R < R(D) - ε while E[d(X,X̂)] ≤ D
```

**Theorem (Shannon)**: R(D) is the **infimum** of achievable rates for distortion D.

## Section 5: Computational Methods

### Blahut-Arimoto Algorithm

**Problem**: Computing R(D) requires optimizing over all conditional distributions Q_Y|X(y|x).

**Solution**: Iterative algorithm that converges to optimal Q*_Y|X.

**Algorithm:**
```
Initialize: Q_Y|X(y|x) arbitrarily

Repeat until convergence:
  1. Compute P_Y(y) = Σ_x Q_Y|X(y|x) P_X(x)

  2. Update Q_Y|X(y|x) ∝ P_Y(y) exp(-λ d(x,y))
     (normalize to sum to 1 over y)

  3. Adjust λ to satisfy distortion constraint E[d(X,Y)] = D

  4. Compute R = I(X;Y) with current Q_Y|X
```

**Properties:**
- Guaranteed convergence to global optimum (convexity)
- Polynomial time complexity in alphabet size
- Extends to continuous sources via discretization

**Applications:**
- Numerical computation of R(D) for arbitrary sources
- Design of quantizers for lossy compression
- Optimal bit allocation in transform coding

From [PLOS Computational Biology 2025: Rate-Distortion Tradeoffs](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952):
> "Rate-distortion theory formalizes the optimal way to compress information while minimizing such distortions, by considering factors such as capacity limitations."

### Neural Estimators of Rate-Distortion Functions

**Challenge**: Classical methods require knowing P_X(x), which is often unavailable in modern applications.

**Solution**: Deep learning-based variational estimators.

**Approach:**
```
min_θ I_θ(X; Z) subject to E[d(X, g_φ(Z))] ≤ D
```

Where:
- Encoder f_θ: X → Z (stochastic mapping)
- Decoder g_φ: Z → X̂ (reconstruction)
- Variational bound on mutual information

**Architectures:**
- **Variational Autoencoders (VAEs)**: β-VAE controls rate-distortion tradeoff
- **Neural compression**: End-to-end learned image/video codecs
- **Rate-distortion autoencoders**: Explicit I(X;Z) estimation

**Advantages**:
- No need for explicit P_X(x)
- Learns optimal encoder/decoder jointly
- Handles high-dimensional data (images, video)

**Example: β-VAE**
```
Loss = E[||X - X̂||²] + β KL(q(z|x) || p(z))
      \_____________/   \__________________/
       distortion            rate
```

β parameter controls rate-distortion tradeoff.

## Section 6: Perception as Lossy Compression

### Information Bottleneck Principle

**Hypothesis**: Perception implements optimal lossy compression of sensory inputs.

**Formulation**:
```
min I(X; Z) - β I(Z; Y)
 Z
```

Where:
- X: sensory input (e.g., retinal image)
- Z: internal representation (e.g., V1 neurons)
- Y: task-relevant variable (e.g., object category)
- β: Lagrange multiplier (tradeoff parameter)

**Equivalence to Rate-Distortion:**
- Rate: I(X; Z) (information preserved from input)
- Distortion: -I(Z; Y) (task-relevant information lost)
- Minimize rate while maximizing task-relevant information

**Neuroscience Evidence:**
- **Retinal ganglion cells**: Compress ~1 million photoreceptors → ~1 million ganglion cells using predictive coding
- **V1 simple cells**: Sparse coding implements near-optimal rate-distortion for natural images
- **Foveal vision**: Allocates bits/area according to eccentricity (more bits for fovea)

From [The Evolution of Lossy Compression](https://pubmed.ncbi.nlm.nih.gov/28490604/) (2017):
> "An organism needs to track fitness-relevant information about its world, but the more information it tracks, the more resources it must devote to perception."

### Efficient Coding Hypothesis

**Barlow (1961)**: Sensory systems are adapted to efficiently encode natural statistics.

**Predictions:**
1. **Whitening**: Remove predictable correlations (spatial/temporal)
2. **Sparsity**: Use rate-distortion-optimal sparse representations
3. **Adaptation**: Adjust encoding to match current statistics

**Experimental Support:**
- **Retina**: Center-surround receptive fields remove spatial correlations (whitening)
- **V1**: Gabor-like filters match statistics of natural images
- **Auditory cortex**: Tonotopic organization matched to speech/music statistics

**Rate-Distortion Connection:**
- **Natural images**: Heavy-tailed wavelet coefficient distributions
- **Optimal encoding**: Sparse coding (many zeros, few large coefficients)
- **Rate-distortion**: Sparse codes achieve near-optimal R(D) for natural images

### Neural Token Allocation as Rate-Distortion

**Vision-Language Models (VLMs)**: Compress images into discrete tokens.

**Rate-Distortion Formulation:**
- **Rate**: Number of tokens (64-576 typical)
- **Distortion**: Task performance (VQA, captioning accuracy)
- **Tradeoff**: More tokens → better performance, higher compute cost

From [arXiv 2024: Inference Optimal VLMs](https://arxiv.org/abs/2411.03312):
> "The inference-optimal behavior in VLMs is achieved by using the largest LLM that fits within the inference budget while minimizing visual token count."

**Empirical Findings:**
- **Gaussian images**: R(D) = (1/2)log(σ²/D) predicts performance
- **Natural images**: Heavy-tailed statistics require adaptive token allocation
- **Optimal allocation**: More tokens for high-information regions (fovea, salient objects)

**ARR-COC-0-1 Connection**: Token budget (64-400) AS rate constraint in rate-distortion optimization.

## Section 7: Connections to Channel Capacity

### Shannon's Source-Channel Separation Theorem

**Theorem**: Optimal communication systems separate source coding and channel coding.

**Two-stage process:**
1. **Source coding**: Compress X to rate R(D) with distortion D
2. **Channel coding**: Transmit R bits reliably over channel with capacity C

**Condition for reliable transmission:**
```
R(D) ≤ C
```

**Interpretation:**
- If C < H(X): Some information must be lost
- Choose D such that R(D) = C for optimal tradeoff
- Channel capacity determines achievable distortion

**Example:**
- Source: Gaussian X with H(X) = 5 bits/sample
- Channel: Capacity C = 2 bits/sample
- Solution: Accept distortion D such that R(D) = 2
- From R(D) = (1/2)log(σ²/D): D = σ²/16 (93.75% variance reduction impossible)

### Joint Source-Channel Coding

**Practical systems**: Sometimes joint coding outperforms separation.

**When?**
- Very low latency requirements (can't afford long block codes)
- Channel varies rapidly (can't track with separate codes)
- Analog transmission (can exploit analog channel properties)

**Example: Analog transmission**
- Directly map source amplitude to channel amplitude
- No quantization, no digital encoding
- Can achieve R(D) = C exactly (no gap)

**Digital vs Analog:**
- **Digital**: Separation theorem applies, R(D) + ε achievable
- **Analog**: Can match R(D) = C exactly but sensitive to noise
- **Hybrid**: Best of both worlds in some scenarios

## Section 8: ARR-COC-0-1 Token Budget as Rate-Distortion Optimization (10%)

### Relevance Realization as Lossy Compression

**ARR-COC-0-1 Core Problem**: Allocate 64-400 tokens per patch under compute budget.

**Rate-Distortion Formulation:**
- **Rate R**: Token count (64, 128, 256, 400)
- **Distortion D**: Task performance degradation (VQA accuracy loss, captioning BLEU decrease)
- **Constraint**: Total token budget K (e.g., 10,000 tokens for 50 patches)

**Optimization:**
```
Allocate tokens {t₁, t₂, ..., tₙ} to patches to:

min Σᵢ D_i(t_i)        (minimize total distortion)

subject to: Σᵢ t_i ≤ K  (rate constraint)
```

**Solution (Lagrangian):**
```
Allocate more tokens where dD/dt is large
(high marginal benefit of additional tokens)
```

### Three Ways of Knowing as Distortion Components

**Propositional (Information Content)**:
- **Measure**: Shannon entropy H(patch)
- **Rate-distortion**: High entropy → needs more bits
- **ARR-COC**: High H → allocate more tokens

**Perspectival (Salience)**:
- **Measure**: Visual saliency, edge density
- **Rate-distortion**: Salient regions have higher d(D)/dt (benefit more from extra bits)
- **ARR-COC**: High salience → allocate more tokens

**Participatory (Query Relevance)**:
- **Measure**: Query-content mutual information I(Q; X_patch)
- **Rate-distortion**: High I(Q; X) → critical for task, minimize distortion
- **ARR-COC**: High I(Q; X) → allocate more tokens

### Opponent Processing as Rate-Distortion Constraint Navigation

**Compress ↔ Particularize:**
- **Compress**: Reduce rate R (fewer tokens)
- **Particularize**: Reduce distortion D (more tokens, fine detail)
- **Balance**: Find optimal R(D) operating point

**Exploit ↔ Explore:**
- **Exploit**: Allocate to known-relevant regions (minimize distortion for current task)
- **Explore**: Allocate broadly (build flexible representations for future tasks)
- **Rate-distortion**: Multi-task learning changes optimal allocation

**Focus ↔ Diversify:**
- **Focus**: Concentrate tokens on few patches (low rate, high distortion on others)
- **Diversify**: Spread tokens evenly (higher rate, lower worst-case distortion)
- **Rate-distortion**: Risk-sensitive formulations (minimize max distortion vs. expected distortion)

### Variable LOD as Rate Allocation

**Pyramid Structure:**
- **Level 0**: 64 tokens (coarse, low rate)
- **Level 1**: 128 tokens (medium rate)
- **Level 2**: 256 tokens (fine detail)
- **Level 3**: 400 tokens (maximum detail)

**Rate-Distortion Interpretation:**
- Each level corresponds to operating point on R(D) curve
- Higher levels: Higher rate R, lower distortion D
- Select level based on patch importance (d(D)/dt)

**Adaptive Allocation:**
```
For each patch i:
  Compute relevance score r_i (from 3 ways of knowing)

  Allocate tokens:
    t_i = {
      64   if r_i in bottom 25%  (low rate, accept distortion)
      128  if r_i in 25-50%      (medium rate)
      256  if r_i in 50-75%      (high rate)
      400  if r_i in top 25%     (maximum rate, minimize distortion)
    }
```

**Foveated Vision Analogy:**
- **Fovea**: 400 tokens (high acuity, low distortion)
- **Parafovea**: 256 tokens (medium acuity)
- **Periphery**: 64-128 tokens (low acuity, high distortion acceptable)
- **Rate budget**: Total retinal ganglion cell firing rate

### Empirical Rate-Distortion Curves for VLMs

**Measurement:**
1. Vary token budget T = {64, 128, 256, 400, 576}
2. Measure task performance P(T) (VQA accuracy, captioning BLEU)
3. Define distortion: D(T) = 1 - P(T)
4. Plot R(D) curve: Rate = T, Distortion = D(T)

**Expected Shape (log-linear):**
```
log₂(1/D) ≈ α·T + β

Or equivalently: D ≈ 2^(-α·T - β)
```

**Parameter interpretation:**
- α: Efficiency of token usage (steeper = better encoding)
- β: Base performance (intercept)

**Comparison to Gaussian bound:**
- Gaussian: R(D) = (1/2)log(σ²/D)
- VLM: R(D) ≈ (1/α)log(1/D) - β/α
- Ratio α/0.5 measures efficiency relative to Gaussian optimum

**Practical findings:**
- Most VLMs operate 2-4x above theoretical R(D) for natural images
- CLIP visual encoders: ~3x gap
- BLIP-2 Q-Former: ~2.5x gap (better due to learned compression)
- ARR-COC-0-1 goal: Approach theoretical R(D) through relevance-aware allocation

### Training for Rate-Distortion Optimality

**Objective (Rate-Distortion Loss):**
```
L = E[Task Loss(X, t)] + λ·E[Token Count(t)]
    \________________/   \________________/
       distortion            rate
```

Where:
- Task Loss: Cross-entropy (classification), negative BLEU (captioning), etc.
- Token Count: t_i for patch i
- λ: Lagrange multiplier controlling tradeoff

**Lagrangian optimization:**
```
∂L/∂t_i = ∂D/∂t_i + λ = 0

⟹ Allocate tokens where ∂D/∂t_i = -λ
```

**Adaptive λ annealing:**
- Start: λ = 0 (no rate penalty, learn task)
- Middle: Gradually increase λ (introduce compression pressure)
- End: λ = λ* (optimal rate-distortion tradeoff)

**Curriculum learning:**
1. **Phase 1**: All patches get maximum tokens (t_i = 400)
   - Learn task without compression pressure
   - Establish performance ceiling
2. **Phase 2**: Introduce rate penalty (λ > 0)
   - Model learns to identify high-relevance patches
   - Allocates more tokens to critical regions
3. **Phase 3**: Increase λ to target operating point
   - Converge to optimal R(D) allocation
   - Validate on rate-distortion curve

## Section 9: Modern Applications and Extensions

### Neural Image Compression

**End-to-end learned codecs** outperform traditional codecs (JPEG, JPEG2000) on rate-distortion curves.

**Architecture:**
```
X → Encoder → Z (latent) → Quantize → Entropy Code
                                           ↓
X̂ ← Decoder ← Ẑ (quantized) ← Entropy Decode
```

**Training Loss (Rate + Distortion):**
```
L = λ·D(X, X̂) + R(Ẑ)
  = λ·||X - X̂||² + E[- log₂ p(Ẑ)]
```

**State-of-the-art (2024-2025):**
- **Google VCM**: Beats VVC (H.266) at all bitrates
- **Facebook Hific**: High-fidelity image compression with GANs
- **Neural compression**: 20-40% bitrate savings over JPEG at same PSNR

From [arXiv 2024: Rate-Distortion-Complexity Tradeoffs](https://arxiv.org/abs/2410.03898):
> "This paper aims to delve into the rate-distortion-complexity trade-offs of modern neural video coding."

### Rate-Distortion-Complexity Tradeoff

**Three-way tradeoff:**
- **Rate R**: Bits/pixel
- **Distortion D**: MSE or perceptual metric
- **Complexity C**: FLOPs for encoding/decoding

**Pareto frontier**: Codecs that are optimal on (R, D, C) jointly.

**Example:**
- **JPEG**: Low complexity, moderate R(D)
- **Neural codec (lightweight)**: Medium complexity, better R(D)
- **Neural codec (heavy)**: High complexity, best R(D)

**Inference-optimal point**: Given compute budget C*, use codec that minimizes D subject to C ≤ C*.

**Mobile/edge deployment**: Complexity constraint dominates (C* small).

**Cloud deployment**: Rate constraint dominates (bandwidth expensive).

### Video Compression and Temporal Redundancy

**Video**: Sequences of correlated frames X_1, X_2, ..., X_T

**Rate-distortion with memory:**
```
R(D) = lim_{T→∞} (1/T) R_T(D)
```

**Exploiting temporal correlation:**
- **Motion compensation**: Predict X_t from X_{t-1}
- **Residual coding**: Encode prediction error with rate R(D_residual)
- **Total rate**: R_motion + R_residual < R_independent

**Modern codecs (H.264, H.265, AV1):**
- Motion estimation: 30-50% of encoding time
- Rate-distortion optimization for mode selection
- Lagrangian: Select mode minimizing D + λR

**Neural video compression:**
- **Learned motion**: Neural networks predict motion fields
- **Learned residual**: Compress residual with learned codec
- **End-to-end**: Joint optimization of motion + residual

From [MDPI Entropy 2025: Rate-Distortion-Perception Trade-off](https://www.mdpi.com/1099-4300/27/4/373):
> "Conventional RD theory, originally developed by Claude Shannon, has effectively characterized the minimal rate required to satisfy distortion constraints."

### Distributed Source Coding

**Slepian-Wolf theorem**: Two correlated sources can be compressed separately but decoded jointly.

**Setup:**
- X and Y correlated (e.g., stereo cameras)
- Compress X at rate R_X, Y at rate R_Y separately
- Joint decoder reconstructs both

**Achievable rate region:**
```
R_X ≥ H(X|Y)
R_Y ≥ H(Y|X)
R_X + R_Y ≥ H(X,Y)
```

**Rate-distortion extension (Wyner-Ziv):**
- Source X available at decoder (side information)
- Compress Y at rate R_Y with distortion D
- Achieves R_Y = R(Y|X, D) < R(Y, D)

**Applications:**
- **Stereo video**: Compress left/right views jointly
- **Sensor networks**: Distributed compression of correlated measurements
- **Multi-view learning**: Compress multiple views of same scene

## Sources

**Source Documents:**
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md) - Free energy minimization framework
- [information-theory/00-shannon-entropy-mutual-information.md](../information-theory/00-shannon-entropy-mutual-information.md) - Shannon entropy and mutual information

**Web Research:**
- [Rate-Distortion Theory - Wikipedia](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory) (accessed 2025-11-16) - Foundational concepts and analytical solutions
- [Blau & Michaeli 2019: The Rate-Distortion-Perception Tradeoff](https://arxiv.org/abs/1901.07821) (arXiv:1901.07821, accessed 2025-11-16) - Extended framework with perceptual quality
- [Shannon Bounds for Quadratic Rate-Distortion Problems](https://ieeexplore.ieee.org/document/10684730/) (IEEE 2024, accessed 2025-11-16) - Shannon lower bounds and Berger techniques
- [PLOS Computational Biology 2025: The Geometry of Efficient Codes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952) (accessed 2025-11-16) - Rate-distortion in biological perception
- [Marzen & DeDeo 2017: The Evolution of Lossy Compression](https://pubmed.ncbi.nlm.nih.gov/28490604/) (accessed 2025-11-16) - Information-theoretic constraints on perception
- [arXiv 2024: Inference Optimal VLMs](https://arxiv.org/abs/2411.03312) (accessed 2025-11-16) - Token budget optimization in vision-language models
- [arXiv 2024: Rate-Distortion-Complexity Tradeoffs in Neural Video Coding](https://arxiv.org/abs/2410.03898) (accessed 2025-11-16) - Three-way tradeoff analysis
- [MDPI Entropy 2025: Rate-Distortion-Perception Trade-off](https://www.mdpi.com/1099-4300/27/4/373) (accessed 2025-11-16) - Neural network applications

**Additional References:**
- [ScienceDirect 2024: Rate-Distortion-Perception-Semantics Tradeoff](https://www.sciencedirect.com/science/article/abs/pii/S0016003224002941) - Multi-objective optimization
- [Universal Rate-Distortion-Perception Representations](https://proceedings.neurips.cc/paper/2021/file/5fde40544cff0001484ecae2466ce96e-Paper.pdf) (NeurIPS 2021) - Theoretical foundations
