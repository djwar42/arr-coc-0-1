# Position Encoding for Vision-Language Models

## Overview

Position encoding is critical for Vision-Language Models (VLMs) because the self-attention mechanism in transformers is **permutation-invariant** – without explicit positional information, the model cannot distinguish token order or spatial relationships. While 1D position encoding (text-only) is well-established, VLMs face the unique challenge of encoding **heterogeneous modalities**: 1D text sequences and 2D/3D visual content (images, video frames).

This knowledge file systematically examines position encoding taxonomies, the evolution from absolute to relative methods, and the state-of-the-art approaches used in modern VLMs.

## Position Encoding Taxonomy

### Three Core Categories

**Category**

**What It Encodes**

**Representative Methods**

**Absolute Position Encoding**

Fixed position index for each token

Sinusoidal (Vaswani et al. 2017), Learned embeddings

**Relative Position Encoding**

Distance between token pairs

Shaw et al. 2018, T5 bias, ALiBi, Transformer-XL

**Rotary Position Encoding (RoPE)**

Rotation-based relative encoding

RoFormer (Su et al. 2021), M-RoPE, 2D/3D RoPE variants

From [Revisiting Multimodal Positional Encoding in Vision-Language Models](https://arxiv.org/html/2510.23095v2) (Huang et al., 2025):

> Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding.

## Absolute Position Embeddings (First Generation)

### Sinusoidal Position Encoding

The original Transformer (Vaswani et al., 2017) used **fixed sinusoidal functions** to encode position:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where:
- `pos` = token position index
- `i` = dimension index in the position vector
- `d` = model dimension (d_model)
- `10000` = base wavelength (θ)

**Key Properties:**
- **Parameter-free**: No learnable weights
- **Continuous**: Smooth, differentiable functions
- **Multi-scale**: Each frequency captures different granularities (token-level vs phrase-level)
- **Linear relationship**: Phase difference between positions is proportional to distance

From [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) (Fleetwood, 2024):

> The phase difference between two positions is a **linear function of distance**, enabling the model to decode relative offsets algebraically.

### Learned Absolute Embeddings

An alternative approach is to learn a unique embedding vector for each position during training:

```python
position_embeddings = nn.Embedding(max_seq_len, d_model)
```

**Strengths:**
- Task-specific adaptation
- Fastest to implement

**Weaknesses:**
- Hard upper limit on sequence length
- Cannot extrapolate to unseen lengths
- Requires parameters (max_seq_len × d_model)

### Limitations of Absolute Encodings

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

**Problem 1: Discards 3D Structure**

Vanilla RoPE and absolute encodings treat multimodal input as a **flattened 1D sequence**:
- Position indices assigned incrementally: `[0, 1, 2, 3, 4, ...]`
- For images: spatial relationships (2D grid) are lost
- For video: temporal structure (frame order) is conflated with spatial layout

**Problem 2: Large Position Indices**

In long sequences (e.g., high-resolution images, long videos), position indices grow excessively large:
- Hurts extrapolation performance
- Causes numerical instability in some implementations

**Problem 3: Absolute vs Relative**

For language and vision, **distance** ("how far apart") is more informative than **absolute index** ("position 42"):
- "The dog chased another dog" – the two "dog" tokens refer to different entities
- Their relationship is defined by distance, not absolute position

## Relative Position Embeddings (Second Generation)

### Core Principle

Relative encodings inject a **bias** into the attention score between tokens `i` and `j`:

```
Attention(Q, K, V) = softmax((QK^T + Bias(i-j)) / √d_k) V
```

Where `Bias(i-j)` depends only on the **offset** `(i-j)`, not absolute positions.

### Shaw et al. 2018: Additive Distance Embeddings

Learn a vector per offset, add to query or key:

```
a_ij = (q_i + r_{i-j})^T k_j
```

Where `r_{i-j}` is a learned relative position embedding.

**Benefits:**
- Parameter sharing across positions
- Naturally captures translation invariance

From [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/) (LearnOpenCV, 2025):

> The absolute position of a word rarely matters for meaning – what matters is how words relate to each other.

### Transformer-XL: Segment-Level Recurrence

Transformer-XL (Dai et al., 2019) extends relative encodings to **very long contexts** by:
- Reusing hidden states across segments
- Sharing relative position parameters across segments
- Enabling models to handle thousands of tokens

### T5 Bucketed Relative Bias

T5 (Raffel et al., 2020) groups distances into **logarithmic buckets**:
- Nearby positions get fine-grained bins
- Distant positions share coarser bins
- Reduces parameter count while preserving distance signal

### ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) adds a **slope × distance** term directly to attention scores:

```
Attention_score(q_i, k_j) = q_i · k_j - slope × |i - j|
```

**Advantages:**
- Zero new parameters
- Constant memory footprint
- Smooth extrapolation to longer sequences

**Used in:**
- GPT-NeoX-20B
- BLOOM
- Long-context LLMs

### Limitations of Relative Position Embeddings

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

Early relative position embeddings made **distance** explicit but:
- Required **lookup tables** (memory overhead)
- Used **bucketing** (loses granularity)
- Involved **runtime gather operations** (slower)

**RoPE** delivers the same distance signal through **phase rotation** – analytical, continuous, parameter-free.

## Rotary Position Embedding (RoPE)

### Core Innovation

RoPE (Su et al., 2021) encodes position as a **rotation** in 2D sub-spaces of the query and key vectors:

```
q_rotated = R(θ_p) · q
k_rotated = R(θ_p) · k

where θ_p,i = p / 10000^(2i/d)
```

**Rotation Matrix:**
```
R(θ) = [cos θ   -sin θ]
       [sin θ    cos θ]
```

From [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding):

> RoPE rotates query and key vectors in a shared 2-D sub-space by an angle proportional to their positions. After rotation, their dot product encodes **only the relative distance**.

### Why Rotation Works

**Geometric Interpretation of Dot Product:**
```
a · b = |a| |b| cos θ
```

By rotating vectors, we can modulate the dot product purely by changing the angle between them, **without affecting vector norms** (which encode semantic information).

### Mathematical Formulation

For position `p` and dimension pair `i`:

```
θ_p,i = p / 10000^(2i/d)

[q_{2i}^rotated  ]   [cos θ_p,i   -sin θ_p,i] [q_{2i}  ]
[q_{2i+1}^rotated] = [sin θ_p,i    cos θ_p,i] [q_{2i+1}]
```

Apply the same rotation to keys:

```
[k_{2i}^rotated  ]   [cos θ_p,i   -sin θ_p,i] [k_{2i}  ]
[k_{2i+1}^rotated] = [sin θ_p,i    cos θ_p,i] [k_{2i+1}]
```

**Attention Score:**
```
Attention(q_p, k_q) ∝ q_p^rotated · k_q^rotated
                    ∝ R(θ_p) q · R(θ_q) k
                    ∝ q · R(θ_q - θ_p) k
```

The score depends on `θ_q - θ_p`, which is **proportional to distance** `(q - p)`.

### Efficient Implementation

From [Inside RoPE](https://learnopencv.com/rope-position-embeddings/):

In practice, rotation is computed element-wise (not as matrix multiplication):

```python
q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
k_rotated = k * cos(θ) + rotate_half(k) * sin(θ)

def rotate_half(x):
    # Rotate pairs: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat((-x2, x1), dim=-1)
```

### Multi-Clock Interpretation

RoPE creates **d/2 independent "clocks"**, each with a different frequency:

- **Low i (fast clocks)**: Rotate quickly, encode **local** relationships (n-grams, punctuation)
- **High i (slow clocks)**: Rotate slowly, encode **long-range** relationships (paragraphs, documents)

From [Inside RoPE](https://learnopencv.com/rope-position-embeddings/):

> For `d = 1024`, the slowest clock (pair 511) makes one full revolution after ≈ 6.2 × 10^4 tokens.

**Example (d=64):**
- Pair 0: ~6 tokens per rotation (fast)
- Pair 7: ~47 tokens per rotation (mid-range)
- Pair 31: ~50,000 tokens per rotation (slow)

### Key Properties of RoPE

**Property**

**Benefit**

**Relative by construction**

Attention depends on `(p - q)`, not absolute indices

**Parameter-free**

No additional weights beyond standard Q/K/V projections

**Smooth extrapolation**

Angles extend indefinitely; with scaling (NTK/YaRN), models handle 256k+ tokens

**Streaming-friendly**

Rotation computed once when writing KV cache

**Multi-scale encoding**

Different frequency pairs capture different temporal scales

### RoPE in Production

**Model**

**Usage**

**LLaMA 2/3**

Default positional encoding

**Qwen 2/2.5**

Base text encoding; M-RoPE for vision

**Gemma**

RoPE with NTK scaling

**Mistral**

RoPE with sliding window attention

**CodeLlama**

RoPE with NTK scaling for long context (100k tokens)

## Extending RoPE to Multimodal Inputs

### The Challenge: Heterogeneous Modalities

VLMs must encode:
- **1D text**: Sequential tokens `[0, 1, 2, 3, ...]`
- **2D images**: Grid layout `(height, width)`
- **3D video**: Temporal + spatial `(time, height, width)`

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

> Vision-Language Models (VLMs) also require positional encodings that can handle heterogeneous modalities, including 1D text and 2D/3D visual inputs.

### Multimodal RoPE (M-RoPE)

**Qwen2-VL** (Wang et al., 2024) introduced **M-RoPE** – extending RoPE to 3D:

**Position Tuple:**
```
m_i = (m_i^t, m_i^h, m_i^w)
```

Where:
- `m^t` = temporal position (frame index)
- `m^h` = vertical position (row index)
- `m^w` = horizontal position (column index)

**Frequency Allocation:**

M-RoPE partitions the `d` dimensions into three contiguous blocks:
- Dimensions 0 to d/3: temporal axis (t)
- Dimensions d/3 to 2d/3: height axis (h)
- Dimensions 2d/3 to d: width axis (w)

**Angle Computation:**
```
θ_t = m^t / 10000^(2i_t / (d/3))
θ_h = m^h / 10000^(2i_h / (d/3))
θ_w = m^w / 10000^(2i_w / (d/3))
```

Each axis gets its own set of rotation matrices applied to its dimension slice.

### M-RoPE Limitations

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

**Problem 1: Temporal Bias**

M-RoPE allocates the temporal axis to the **highest-frequency channels**:
- Rapid decay of attention over time
- Detrimental to long-sequence video understanding

**Problem 2: Asymmetric Spatial Decay**

Height and width axes occupy **distinct, non-overlapping frequency ranges**:
- Different long-range decay rates for vertical vs horizontal
- Impairs learning of consistent spatial relationships

**Problem 3: Reduced Frequency Resolution**

Partitioning feature dimensions coarsens the frequency spectrum for each axis:
- Less granular multi-scale modeling per axis

### Advanced M-RoPE Variants

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

**Multi-Head RoPE (MHRoPE):**
- Partitions encoding task among **different attention heads**
- Each head assigned to specific positional axis
- Each axis gets **full frequency spectrum** within its heads
- More scalable as number of axes grows

**MRoPE-Interleave (MRoPE-I):**
- Distributes channels in **round-robin manner**: `[t, h, w, t, h, w, ...]`
- Each axis encoded with full frequency spectrum
- Compatible with extrapolation algorithms (NTK, YaRN)

### Spatial-Reset Mechanism

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

**Observation:** MRoPE exhibits a visual "attention sink" – attention concentrates on top-left corner of images/frames.

**Solution: Spatial-Reset**
- Reset spatial dimensions `(h, w) → (0, 0)` for each new image/video frame
- Temporal dimension continues incrementing
- Aligns visual sink with LLM's bias for small position IDs

**Position Update:**
```
# Without spatial-reset
m_i = (t, t+h, t+w)  # spatial coupled to temporal

# With spatial-reset
m_i = (t, h, w)      # spatial independent
```

**Benefits:**
1. **Decouples motion representation**: Relative position `(t2-t1, h2-h1, w2-w1)` is purely spatio-temporal
2. **Improves visual attention**: More attention allocated to visual content (verified on DocVQA)
3. **Accelerates visual adaptation**: Aligns with LLM's positional bias

### 2D RoPE for Vision Transformers

For pure vision tasks (no text), **2D RoPE** assigns positions based on patch coordinates:

```
Position: (h, w) for patch at row h, column w

θ_h,i = h / 10000^(2i / d_head)
θ_w,i = w / 10000^(2i / d_head)
```

**Frequency Allocation:**
- Half dimensions encode height
- Half dimensions encode width

From [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298) (Heo et al., 2024):

> This study provides a comprehensive analysis of RoPE when applied to ViTs, utilizing practical implementations of RoPE for 2D vision data.

**Applications:**
- Vision Transformers (ViT)
- Masked Autoencoders (MAE)
- DINO self-supervised learning

## Design Principles for Robust Multimodal Position Encoding

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2), three core guidelines emerge:

### Guideline 1: Positional Coherence

**Requirements:**
1. **Preserve 3D structure** of visual content (don't flatten to 1D)
2. **Maintain slow growth rate** of position indices
3. **Avoid modality confusion** in generation (no overlapping position IDs)
4. **Establish appropriate modality interval** (not too large, not zero)

**Example: VideoRoPE's Diagonal Layout Problem**

VideoRoPE centers spatial coordinates, creating a "diagonal layout":
- Visual frames shifted along all three axes
- **Position ID overlap** between visual and text tokens
- Failure mode: endless text repetition ("1111...")

### Guideline 2: Full Frequency Utilization

**Requirements:**
1. Each positional axis should have access to **full frequency spectrum**
2. High frequencies for local/fine-grained relationships
3. Low frequencies for long-range dependencies

**Why it matters:**
- Long-video understanding needs robust low-frequency temporal encoding
- Visual grounding benefits from high-frequency spatial encoding

**Solutions:**
- Multi-Head allocation: Different heads for different axes
- Interleaved allocation: Round-robin channel distribution

### Guideline 3: Preservation of Textual Priors

**Requirements:**
1. Text RoPE must remain **identical** to base LLM
2. No modification to text position design
3. No modification to text frequency allocation

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

> This outcome strongly indicates the critical importance of maintaining full compatibility with the text-only RoPE for effective knowledge transfer from pre-trained LLMs.

**Violations to avoid:**
- Resetting spatial dimensions for text tokens (IL-RoPE, Omni-RoPE)
- Scaling rotary base differently for text
- Breaking compatibility causes poor performance across benchmarks

## Position Encoding Variants for Different VLM Architectures

### Early Fusion VLMs

**Architecture:** Merge vision and language **before** processing (VisualBERT, ViLBERT)

**Position Strategy:**
- Concatenate visual region features + BERT tokens
- Apply standard 1D positional encoding
- **Problem:** Discards native visual geometry

### Mid Fusion VLMs

**Architecture:** Separate encoding, late fusion via cross-attention (BLIP-2, Flamingo)

**Position Strategy for BLIP-2 Q-Former:**
- Learnable query tokens (32-64) compress visual features
- Q-Former uses learned absolute position embeddings
- Text encoder uses original BERT position embeddings

**Position Strategy for Flamingo Perceiver Resampler:**
- Cross-attention between learned queries and visual features
- Queries have learned position embeddings
- Achieves ~200× compression of visual tokens

### Late Fusion VLMs

**Architecture:** Project and concatenate (LLaVA, InstructBLIP)

**Position Strategy for LLaVA:**
1. Vision encoder (CLIP): Absolute position embeddings (learned or sinusoidal)
2. MLP projector: Maps vision tokens to LLM space
3. Concatenation: `[text_prefix, vision_tokens, text_suffix]`
4. LLM: RoPE applied to **entire concatenated sequence**

**Image slicing (dynamic resolution):**
- Grid tokenization: 336×336 crops from high-res image
- Each crop processed independently
- Position encoding challenge: How to encode **grid structure**?

### Hybrid Fusion VLMs

**Architecture:** Multi-stage, multi-layer injection (Ovis 2.5, Qwen3-VL)

**Ovis 2.5 Visual Embedding Table (VET):**
- Native resolution processing (no resizing)
- Structural alignment between vision and language layers
- Uses M-RoPE with spatial-reset

**Qwen3-VL DeepStack:**
- Shallow layer injection: Early fusion signals
- Deep layer injection: Late fusion refinement
- M-RoPE with interleaved frequency allocation

## Scaling RoPE to Long Contexts

### The Extrapolation Challenge

RoPE's base frequency (θ = 10000) was designed for sequences up to ~8k tokens. For longer contexts:
- High-frequency pairs "wrap around" too quickly
- Attention loses fine-grained positional detail
- Perplexity degrades beyond training length

From [Inside RoPE](https://learnopencv.com/rope-position-embeddings/):

**Failure modes at long context:**

**Mode**

**Cause**

**Symptom**

**Phase-shift drift**

Fast pairs spin too quickly

Loss of syntactic coherence after 8k-16k tokens

**Numerical precision**

fp16 rounding for large `p`

Attention logits become noisy

**Training-inference gap**

Model only saw 4k tokens

Quality degrades smoothly at 32k

### NTK-Aware Scaling

**Idea:** Increase the base wavelength proportionally to context length

```
θ_scaled = θ_base * (L_new / L_train)^(d/(d-2))
```

Where:
- `L_train` = training sequence length
- `L_new` = target inference length

**Effect:** Slows down high-frequency rotations, preventing wrap-around

### YaRN (Yet another RoPE extensioN)

YaRN (Peng et al., 2023) applies **non-uniform scaling**:
- Low frequencies (slow clocks): minimal scaling
- High frequencies (fast clocks): aggressive scaling
- Smooth interpolation between

**Benefits:**
- Better preservation of local positional signal
- Improved long-range modeling
- Validated to 128k-256k tokens

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2):

> MRoPE-I requires a smaller YaRN scaling factor than vanilla RoPE due to more efficient position design (advancing by O(max(h,w)) vs O(h×w) for an image).

### Position Interpolation

**Idea:** Linearly interpolate position indices into training range

```
p_interpolated = p * (L_train / L_new)
```

**Example:** For 16k context (trained on 4k):
- Token at position 8000 → interpolated to position 2000
- Maps new positions into familiar range

**Drawback:** Compresses positional signal, potentially losing resolution

## Pipeline Parallelism and Position Encoding

From [DeepSpeed Pipeline Parallelism](../distributed-training/01-deepspeed-pipeline-parallelism.md):

**VLM Pipeline Stages:**
1. Vision encoder (CLIP/ViT) → Stage 1
2. Fusion module (Q-Former/Projector) → Stage 2
3. LLM decoder → Stages 3-N

**Position Encoding Distribution:**

**Stage**

**Position Encoding Type**

**Vision encoder**

Absolute (learned or sinusoidal) applied within ViT

**Fusion module**

None or learned queries

**LLM decoder**

RoPE applied to concatenated sequence (vision + text)

**Pipeline considerations:**
- Pre-compute position embeddings to reduce per-stage computation
- Cache RoPE rotation matrices for KV cache optimization
- Ensure consistent position indexing across pipeline boundaries

## VLM Serving Optimization

From [TensorRT VLM Deployment](../inference-optimization/01-tensorrt-vlm-deployment.md):

**RoPE Optimization Strategies:**

### Pre-computation and Caching

```python
# Pre-compute rotation matrices for all positions
cos_cache = cos(θ) for θ in [θ_0, θ_1, ..., θ_max]
sin_cache = sin(θ) for θ in [θ_0, θ_1, ..., θ_max]

# During inference: lookup instead of compute
cos_pos = cos_cache[position]
sin_pos = sin_cache[position]
```

**Benefits:**
- Eliminates trigonometric computation on critical path
- Reduces per-token latency by 5-10%

### Fused Kernels

TensorRT can fuse RoPE application with attention computation:

```
# Standard (3 kernels)
q_rot = rope(q)
k_rot = rope(k)
attn = scaled_dot_product(q_rot, k_rot, v)

# Fused (1 kernel)
attn = fused_rope_attention(q, k, v, cos_cache, sin_cache)
```

**Benefit:** 20-30% faster attention on A100/H100 GPUs

### Mixed-Precision Considerations

From [Inside RoPE](https://learnopencv.com/rope-position-embeddings/):

> In mixed-precision inference, large angles lead to `sin, cos` values that differ by < ε of fp16, effectively collapsing several high-freq clocks to the same vector ("angle saturation").

**Solution:** Compute RoPE in **FP32**, store in **FP16** after rotation

## Apple Metal Neural Engine Optimization

From [Apple Metal ML](../alternative-hardware/01-apple-metal-ml.md):

**M4 Neural Engine Position Ops:**

Apple Silicon has dedicated hardware for:
- **Trigonometric functions**: sin, cos, atan2
- **Matrix rotations**: 2×2 rotation matrices
- **FP16 precision**: Optimized for mobile inference

**Optimization for M-RoPE on Metal:**
```swift
// Metal Performance Shaders (MPS) graph
let cos_θ = MPSGraph.cos(position / base)
let sin_θ = MPSGraph.sin(position / base)

let q_rot = q * cos_θ + rotate_half(q) * sin_θ
```

**Performance:** 2-3x faster than CPU, approaching discrete GPU efficiency for RoPE

## ARR-COC-0-1: Spatial Position for Relevance Maps

The **ARR-COC-0-1** project (Adaptive Relevance Realization - Contexts Optical Compression) uses spatial position encoding for query-aware visual token allocation.

From [ARR-COC-0-1 texture.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/texture.py):

### 13-Channel Texture Array

Position information is encoded in the **spatial channels** and **eccentricity channel**:

**Channel**

**Position Information**

**x_norm, y_norm**

Normalized spatial coordinates (0-1)

**eccentricity**

Distance from image center: `sqrt((x - 0.5)^2 + (y - 0.5)^2)`

**Use in Relevance Scoring:**

From [ARR-COC-0-1 knowing.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py):

```python
# Perspectival knowing: Salience via eccentricity weighting
eccentricity = texture[:, :, :, 10]  # Channel 10
salience = compute_salience(visual_features) * (1 - eccentricity)
```

**Rationale:** Human foveal vision has higher acuity at center, degrading toward periphery

### Relevance-Driven Token Selection

Position encoding enables **spatially-aware compression**:

1. **Query Embedding**: Text query encoded with RoPE
2. **Visual Patches**: 2D position + eccentricity
3. **Participatory Knowing**: Cross-attention between query and visual patches
4. **Token Budget Allocation**: 64-400 tokens per patch based on **query-position coupling**

From [ARR-COC-0-1 attending.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py):

**Example:** For query "What is the text in the bottom-right corner?"
- High relevance scores for patches at position (h > 0.7, w > 0.7)
- Higher token budget allocated to bottom-right region
- Lower budget for top-left (low query-position coupling)

This demonstrates how **explicit 2D position encoding** combined with query understanding enables **adaptive, relevance-driven** visual token compression – a key innovation beyond standard uniform tokenization.

## Empirical Performance Comparison

From [Revisiting Multimodal Positional Encoding](https://arxiv.org/html/2510.23095v2) benchmark results (Qwen2.5-VL base, 7B LLM):

**Method**

**Image Avg**

**Video Avg**

**Grounding Avg**

Vanilla RoPE

65.69

51.64

73.48

MRoPE (Qwen2-VL)

65.18

51.51

73.69

VideoRoPE

60.64

52.18

72.59

CircleRoPE

62.86

51.09

74.96

**MHRoPE**

**66.40**

**52.98**

74.92

**MRoPE-I**

**66.65**

52.36

**75.85**

**Key Findings:**

1. **Vanilla RoPE is competitive** for images but suffers on grounding tasks requiring fine-grained spatial reasoning

2. **VideoRoPE's diagonal layout** causes severe degradation on document tasks (DocVQA: 60.13 vs 82.94) due to position ID overlap

3. **MHRoPE and MRoPE-I** achieve best overall performance by satisfying all three design guidelines

4. **Spatial-reset mechanism** significantly improves visual attention allocation (verified on DocVQA test set)

## Future Directions

### Context-Aware Position Encoding

**CABLE / GLiN** (2024-25 research):
- Learnable functions of distance
- Context-dependent adaptation
- Target: 100k-token windows

### Hierarchical Position Encoding

Wavelets or multi-scale hierarchical approaches:
- Coarse position for document/paragraph
- Fine position for token
- Better long-context modeling

### Modality-Specific Optimization

**Text:** RoPE with NTK/YaRN scaling
**Images:** 2D RoPE with spatial-reset
**Video:** 3D RoPE with temporal emphasis
**Audio:** 1D RoPE with frequency-domain encoding

### Learned vs Fixed Trade-offs

Hybrid approaches:
- Fixed RoPE for base positional signal
- Learned adapters for task-specific refinement
- Best of both worlds: generalization + specialization

## Sources

**Source Documents:**
- None (external research only)

**Web Research:**

Primary Papers:
- [Revisiting Multimodal Positional Encoding in Vision-Language Models](https://arxiv.org/html/2510.23095v2) – Huang et al., 2025 (accessed 2025-11-16)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) – Su et al., 2021
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) – Vaswani et al., 2017

Technical Guides:
- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) – Fleetwood, HuggingFace Blog, 2024 (accessed 2025-11-16)
- [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/) – LearnOpenCV, 2025 (accessed 2025-11-16)

Additional References:
- Shaw et al., 2018 – Self-Attention with Relative Position Representations
- Dai et al., 2019 – Transformer-XL
- Raffel et al., 2020 – T5
- Press et al., 2022 – ALiBi (Attention with Linear Biases)
- Peng et al., 2023 – YaRN
- Wang et al., 2024 – Qwen2-VL
- Heo et al., 2024 – Rotary Position Embedding for Vision Transformer

**ARR-COC-0-1 References:**
- [texture.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/texture.py) – 13-channel texture array with spatial coordinates
- [knowing.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py) – Perspectival knowing via eccentricity weighting
- [attending.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py) – Query-aware token budget allocation

**Infrastructure References:**
- [DeepSpeed Pipeline Parallelism](../distributed-training/01-deepspeed-pipeline-parallelism.md) – File 2 influence
- [TensorRT VLM Deployment](../inference-optimization/01-tensorrt-vlm-deployment.md) – File 6 influence
- [Apple Metal ML](../alternative-hardware/01-apple-metal-ml.md) – File 14 influence
