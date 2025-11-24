# Perceiver & Perceiver IO: Cross-Attention to Learned Latents

**DeepMind's general-purpose architecture using asymmetric attention to scale transformers**

## Overview - General Architecture

### Core Innovation

Perceiver and Perceiver IO solve a fundamental problem in transformer architectures: scaling attention to very large inputs (millions of elements) without domain-specific assumptions. Instead of applying self-attention directly to all inputs (O(N²) complexity), they use an **asymmetric cross-attention mechanism** that maps inputs to a small learned latent bottleneck, then processes that bottleneck with self-attention.

**Key insight**: By separating the input encoding from the latent processing, Perceivers achieve linear scaling with input size while maintaining the global attention capabilities of transformers.

### Architectural Philosophy

From [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) (Jaegle et al., ICML 2021):

> "Biological systems perceive the world by simultaneously processing high-dimensional inputs from modalities as diverse as vision, audition, touch, proprioception, etc. The perception models used in deep learning on the other hand are designed for individual modalities, often relying on domain-specific assumptions such as the local grid structures exploited by virtually all existing vision models."

**Design goals**:
- General-purpose: Handle any modality (images, audio, video, text, point clouds)
- Scalable: Process hundreds of thousands of inputs efficiently
- Simple: No domain-specific architectural assumptions
- Flexible: Support diverse output types and sizes (Perceiver IO)

### Two Versions

**Perceiver (2021)**: Cross-attention from inputs → learned latent array, limited to simple outputs (classification)

**Perceiver IO (2021)**: Adds output cross-attention, enabling arbitrary structured outputs (language, flow fields, game actions)

---

## Perceiver Architecture

### Learned Latent Queries

**Core mechanism**: Instead of processing all inputs directly, Perceiver uses a **fixed-size learned latent array** as the queries in cross-attention.

**Architecture flow**:
```
Input: x ∈ ℝ^(M × C_input)  (M = 50,000 pixels for ImageNet)
Latent: z ∈ ℝ^(N × D)       (N = 256-1024, D = 512-1024)

1. Cross-attention: z' = CrossAttend(Q=z, KV=x)
2. Self-attention: z'' = SelfAttend(z')  (repeated L layers)
3. Output: Pool(z'') → classification
```

**Key properties**:
- **Input size independence**: Latent processing cost is O(N²L) regardless of M
- **Cross-attention cost**: O(N·M·D) - linear in input size
- **Total scaling**: Linear with input size M, quadratic with latent size N (but N << M)

From [Building architectures that can handle the world's data](https://deepmind.google/discover/blog/building-architectures-that-can-handle-the-worlds-data/) (DeepMind blog, 2021):

> "The Perceiver does this by using attention to first encode the inputs into a small latent array. This latent array can then be processed further at a cost independent of the input's size, enabling the Perceiver's memory and computational needs to grow gracefully as the input grows larger."

### Cross-Attention Mechanism

**Asymmetric attention**: Unlike standard transformer self-attention where Q, K, V all come from the same sequence, Perceiver uses:
- **Queries (Q)**: Learned latent array (small, fixed size)
- **Keys (K) and Values (V)**: Input data (large, variable size)

**Mathematical formulation**:
```
Q = LatentArray · W_Q    (N × D)
K = Input · W_K           (M × D)
V = Input · W_V           (M × D)

Attention(Q, K, V) = softmax(Q·K^T / √D) · V
```

**Why this works**:
- Latents act as **learned feature detectors** that pull relevant information from inputs
- Each latent can attend to all inputs simultaneously (global receptive field)
- No need for local inductive biases (convolutions, positional grids)

### Iterative Refinement

Perceiver applies cross-attention **multiple times** (typically 1-8 iterations):

```python
# Iterative cross-attention
z = learned_latent_array  # Initialize
for _ in range(num_iterations):
    z = cross_attention(queries=z, kv=inputs)  # Update latents
    z = self_attention(z)                       # Process latents
```

**Benefits**:
- Early iterations: Rough feature extraction
- Later iterations: Refinement and integration
- Similar to iterative inference in Bayesian models

**Performance trade-off**:
- More iterations = better performance but higher compute
- ImageNet: 6-8 iterations competitive with ResNet-50
- AudioSet: 2-3 iterations sufficient

---

## Perceiver IO Extensions

### Output Queries

**Major limitation of Perceiver**: Could only produce simple outputs (single classification label) because final latent array was pooled and projected.

**Perceiver IO solution**: Add **output cross-attention** that decodes from latents using learned or structured output queries.

From [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) (Jaegle et al., ICLR 2022):

> "We propose Perceiver IO, a general-purpose architecture that handles data from arbitrary settings while scaling linearly with the size of inputs and outputs."

**Architecture flow**:
```
Input: x ∈ ℝ^(M × C_input)
Latent: z ∈ ℝ^(N × D)
Output queries: q_out ∈ ℝ^(P × D_out)

1. Encode: z' = CrossAttend(Q=z, KV=x)
2. Process: z'' = SelfAttend(z')  (L layers)
3. Decode: out = CrossAttend(Q=q_out, KV=z'')
```

**Output query design** (task-dependent):
- **Classification**: Single learned query → logits
- **Language modeling**: Position-indexed queries (one per character)
- **Optical flow**: Pixel-grid queries (2D spatial structure)
- **Multimodal**: Mixed queries (text positions + image patches)

### Task Flexibility

**Perceiver IO enables diverse tasks with the same architecture**:

**Language (GLUE benchmark)**:
- Input: Raw UTF-8 bytes (no tokenization!)
- Output queries: Per-character positions
- Result: Competitive with BERT despite simpler preprocessing

**Optical flow (Sintel dataset)**:
- Input: Two consecutive video frames (concat along channel dimension)
- Output queries: 2D grid of pixel coordinates
- Result: State-of-the-art performance, no explicit correspondence matching

**StarCraft II**:
- Input: Game state (units, terrain, resources) as unstructured array
- Output queries: Action space queries (move, attack, build)
- Result: Competitive with specialized game architectures

**Multi-task learning**:
- Share encoder latents across tasks
- Use different output queries per task
- Train on multiple datasets simultaneously

### Multimodal Capabilities

**Perceiver IO naturally handles multiple modalities** by concatenating inputs:

```python
# Video understanding (vision + audio)
visual_features = extract_frames(video)  # (T × H × W × 3)
audio_features = extract_spectrogram(audio)  # (T × F)

# Flatten and concatenate
input = concat([
    flatten(visual_features),  # (T·H·W × 3)
    flatten(audio_features)    # (T·F × 1)
])

# Process with Perceiver IO
output = perceiver_io(input, output_queries)
```

**No special fusion modules needed** - cross-attention naturally integrates information across modalities.

**Example results (AudioSet)**:
- Audio-only: 0.363 mAP
- Video-only: 0.344 mAP
- Audio + Video (Perceiver IO): 0.408 mAP (strong fusion without hand-engineering)

---

## Comparison to Standard Transformers

### Computational Complexity

**Standard Transformer**:
```
Input length: M
Self-attention: O(M² · D)
Memory: O(M²)  (attention matrix)

Limitation: Cannot scale to M > 5000 due to quadratic cost
```

**Perceiver**:
```
Input length: M
Latent size: N (typically 256-1024)
Cross-attention: O(M · N · D)  (linear in M)
Self-attention: O(N² · D)      (independent of M)
Total: O(M·N·D + N²·D)

Scales to M = 50,000+ (ImageNet pixels) efficiently
```

**Comparison table**:

| Architecture | ImageNet (224×224) | Sintel (1024×436) | Audio (96kHz, 1min) |
|--------------|-------------------|------------------|---------------------|
| ViT (Transformer) | 150K tokens → 22.5B FLOPs | 446K tokens → 199B FLOPs | 5.7M tokens → 32T FLOPs |
| Perceiver (N=512) | 50K pixels → 2.1B FLOPs | 446K pixels → 3.8B FLOPs | 5.7M samples → 5.2B FLOPs |

**Scaling advantage grows with input size** - Perceiver becomes increasingly efficient for large inputs.

### Inductive Biases

**Standard vision transformers (ViT)**:
- 2D positional embeddings (bakes in spatial structure)
- Patch-based tokenization (local receptive fields initially)
- Often pre-trained on large datasets to overcome lack of CNN inductive bias

**Perceiver**:
- Optional position encodings (Fourier features for images, learned for other modalities)
- Direct pixel/sample processing (no patching required)
- Learns inductive biases from data via cross-attention patterns

**Trade-off**: Perceiver is more data-hungry but more general. ViT is more sample-efficient on vision tasks but less transferable.

### Attention Patterns

**ViT**: Every token attends to every other token (full self-attention)

**Perceiver**:
- Cross-attention phase: Latents attend to all inputs (bottleneck compression)
- Self-attention phase: Latents attend to each other (reasoning and integration)

**Learned specialization**: Different latent queries learn to attend to different input features:
- Some latents focus on local patterns (edges, textures)
- Some latents integrate global context (object categories, scene layout)
- Emerges naturally without hard-coded specialization

---

## Performance Analysis

### ImageNet Classification

**Perceiver results** (from original paper):
- 50,000 raw pixels as input (no patching)
- 79.1% top-1 accuracy (competitive with ResNet-50: 79.3%)
- 1427 latent array size, 8 cross-attention iterations

**Key insight**: Achieves ResNet-level performance **without 2D convolutions** and with **minimal domain assumptions**.

**Comparison**:
```
ResNet-50:      79.3% top-1  (25M params, 4.1B FLOPs)
ViT-B/16:       81.8% top-1  (86M params, 17.6B FLOPs, requires pre-training)
Perceiver:      79.1% top-1  (44M params, 3.2B FLOPs)
```

### AudioSet (Audio Event Detection)

**Multimodal challenge**: Classify audio events in 10-second clips (632 classes).

**Perceiver IO results**:
- Audio-only: 0.363 mAP (competitive with specialized audio CNNs)
- Video-only: 0.344 mAP
- Audio + Video: **0.408 mAP** (SOTA fusion without manual architecture design)

**Advantage**: Same architecture handles audio spectrograms, video frames, and their fusion.

### Optical Flow (Sintel)

**Task**: Estimate per-pixel motion between consecutive frames.

**Perceiver IO results**:
- Clean pass: 1.52 EPE (end-point error)
- Final pass: 2.53 EPE
- **State-of-the-art** among methods without explicit multi-scale correspondence

**Remarkable**: No hand-crafted features for correspondence matching - learns everything from cross-attention.

### Language Understanding (GLUE)

**GLUE benchmark** (9 language tasks: sentiment, entailment, similarity, etc.)

**Perceiver IO results**:
- 72.0 average score (comparable to BERT-base: 71.2)
- Processes **raw UTF-8 bytes** (no WordPiece tokenization)
- 220M parameters vs. BERT-base 110M

**Significance**: Competitive performance with simpler preprocessing, demonstrates generality.

### StarCraft II

**Game AI challenge**: Real-time strategy game with complex action space.

**Perceiver IO as game agent**:
- Input: Game state (units, terrain, resources) as flat array
- Output: Action logits (move, attack, build, etc.)
- Performance: Competitive with specialized architectures

**Demonstrates**: Perceiver IO can handle diverse, unstructured inputs and structured, high-dimensional outputs.

---

## Karpathy Analysis

### Query-Based Compression Insights

Perceiver's cross-attention to learned latents is a **query-based compression mechanism** - similar in spirit to:

**Vector Quantized VAEs (VQ-VAE)**:
- Compress inputs to discrete latent codes
- Perceiver: Continuous latent codes learned via gradient descent

**Bottleneck architectures (autoencoders)**:
- Compress to low-dimensional representation
- Perceiver: Attention-based compression (information-theoretic bottleneck)

**Set-to-set transformations**:
- Perceiver encodes a set (inputs) to another set (latents)
- Similar to [Set Transformer](https://arxiv.org/abs/1810.00825) but with iterative refinement

**Key difference from standard compression**: Latent queries are **learned** to pull task-relevant information, not just minimize reconstruction error.

### Relation to Attention Mechanisms

**Perceiver vs. Cross-Attention in VLMs**:

Many vision-language models use cross-attention (e.g., Flamingo, BLIP-2), but Perceiver's approach is distinct:

**BLIP-2 Q-Former**:
- Fixed number of learned query tokens (32)
- Cross-attend to frozen vision encoder
- Goal: Align vision and language modalities

**Perceiver**:
- Larger learned latent array (256-1024)
- Cross-attend to raw inputs (pixels, samples)
- Goal: General-purpose encoding for any modality

**Similarity**: Both compress high-dimensional vision with learned queries.

**Difference**: Perceiver is input-agnostic (no vision-specific assumptions), Q-Former assumes vision encoder exists.

### Simplicity vs. Performance Trade-Off

**Karpathy principle**: "Prefer simplicity and hackability over marginal performance gains."

**Perceiver strengths** (simple):
- Single architecture for all modalities
- No tokenization, patching, or preprocessing
- Few hyperparameters (latent size, depth, iterations)

**Perceiver weaknesses** (less competitive):
- Lags specialized models by 1-3% on domain-specific benchmarks
- Higher parameter count for equivalent performance
- Requires more data to learn inductive biases

**Educational value**: Excellent architecture for **understanding attention** and **cross-modal learning**. Less ideal for production where specialized models outperform.

**Recommended use cases**:
- Exploratory research on new modalities
- Multi-task learning across diverse datasets
- Teaching/learning about attention mechanisms
- Prototyping before optimizing for specific domain

---

## Implementation Considerations

### Latent Array Initialization

**Critical design choice**: How to initialize learned latent array.

**Options**:
1. **Random initialization**: Sample from N(0, 0.02)
2. **Learned initialization**: Trainable parameters
3. **Input-dependent**: Generated by small encoder network

**Recommendation** (from paper): Learned initialization works best - latents specialize during training.

### Position Encodings

**Perceiver supports flexible position encodings**:

**For images (2D spatial structure)**:
- Fourier features: `sin(2πk·x)`, `cos(2πk·y)` for multiple frequencies k
- Concatenate with RGB channels
- Helps latents learn spatial relationships

**For sequences (1D temporal structure)**:
- Sinusoidal positional encodings (Transformer-style)
- Or learned embeddings per position

**For unstructured data** (point clouds, graphs):
- Coordinate-based encodings
- Or no position encoding (permutation-invariant)

**Key insight**: Position encodings are **optional** - Perceiver can learn from raw data, but encodings help.

### Scaling to Very Large Inputs

**For inputs beyond 1M elements**, use hierarchical processing:

```python
# Hierarchical Perceiver (HiP)
def hierarchical_perceiver(input, num_levels=3):
    latents = learned_latent_array

    for level in range(num_levels):
        # Cross-attend to inputs at this resolution
        latents = cross_attention(queries=latents, kv=input[level])

        # Self-attention among latents
        latents = self_attention(latents)

        # (Optional) Expand latent array for next level
        if level < num_levels - 1:
            latents = expand_latent_array(latents)

    return latents
```

**Hierarchical Perceiver (HiP)** from [Carreira et al., 2022](https://arxiv.org/abs/2202.10890):
- Process inputs at multiple resolutions (coarse-to-fine)
- Increase latent array size at each level
- State-of-the-art on long videos (hours of footage)

### Training Considerations

**Data efficiency**:
- Perceiver requires **more training data** than specialized models
- Learns inductive biases from scratch (no CNN priors)
- Benefit: More general, but slower to train

**Compute requirements**:
- Cross-attention dominates (O(M·N·D))
- Use efficient attention implementations (FlashAttention, xFormers)
- Gradient checkpointing for deep models

**Hyperparameter tuning**:
- Latent size N: Larger = more capacity, higher cost (256-1024 typical)
- Cross-attention iterations: More = better performance, slower (1-8 typical)
- Depth (self-attention layers): 4-12 layers common

---

## Code Availability

**Official implementations**:

**DeepMind Perceiver (JAX)**:
- GitHub: [deepmind/deepmind-research/perceiver](https://github.com/deepmind/deepmind-research/tree/master/perceiver)
- Includes both Perceiver and Perceiver IO
- Example configs for ImageNet, AudioSet, GLUE

**Hugging Face Transformers (PyTorch)**:
- Models: `PerceiverForImageClassification`, `PerceiverForOpticalFlow`, etc.
- Pre-trained weights available
- Example: [deepmind/vision-perceiver-learned](https://huggingface.co/deepmind/vision-perceiver-learned)

**Usage example**:
```python
from transformers import PerceiverForImageClassification

model = PerceiverForImageClassification.from_pretrained(
    "deepmind/vision-perceiver-learned"
)

# Process raw pixels (no patching)
logits = model(pixel_values=images)  # (batch, 224, 224, 3) → (batch, 1000)
```

**Key components to understand**:
1. `PerceiverEncoder`: Cross-attention + self-attention
2. `PerceiverDecoder`: Output cross-attention (Perceiver IO only)
3. `PerceiverResampler`: Variant used in Flamingo (fixed latent size)

---

## Sources

**Original Papers**:
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) - Jaegle et al., ICML 2021 (accessed 2025-01-31)
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) - Jaegle et al., ICLR 2022 (accessed 2025-01-31)

**Web Resources**:
- [Building architectures that can handle the world's data](https://deepmind.google/discover/blog/building-architectures-that-can-handle-the-worlds-data/) - DeepMind blog, August 2021 (accessed 2025-01-31)
- [Perceiver IO: a scalable, fully-attentional model](https://huggingface.co/blog/perceiver) - Hugging Face blog, December 2021 (accessed 2025-01-31)

**GitHub Implementations**:
- [deepmind/deepmind-research/perceiver](https://github.com/deepmind/deepmind-research/tree/master/perceiver) - Official JAX implementation
- [huggingface/transformers](https://huggingface.co/docs/transformers/model_doc/perceiver) - PyTorch implementation with pre-trained weights

**Related Work**:
- [HiP: Hierarchical Perceiver](https://arxiv.org/abs/2202.10890) - Carreira et al., 2022 (hierarchical extension)
- [Perceiver AR: General-purpose, long-context autoregressive modeling](https://arxiv.org/abs/2202.07765) - Hawthorne et al., 2022 (autoregressive variant)

**Additional References**:
- [Set Transformer](https://arxiv.org/abs/1810.00825) - Lee et al., ICML 2019 (set-to-set attention)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., NeurIPS 2017 (original Transformer)
