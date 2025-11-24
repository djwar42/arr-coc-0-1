# Dual Position Encoding (Spatial + Sequential)

**Knowledge expansion topic**: Combining spatial and sequential position encoding for multimodal vision-language models

**Date**: 2025-01-31

---

## Overview

Dual position encoding refers to the practice of encoding **two types of positional information simultaneously**:

1. **Spatial position**: Where in the image (2D: height, width)
2. **Sequential position**: Where in the token sequence (1D: 0, 1, 2, ..., N)

This is critical for **vision-language models (VLMs)** and **video transformers** where tokens have both spatial structure (image patches arranged in a grid) and sequential structure (interleaved vision-text tokens or temporal frames).

**Key insight**: Spatial and sequential positions capture different aspects of structure:
- **Spatial**: "This patch is in the top-left corner of the image"
- **Sequential**: "This token comes after the text prompt and before the next image patch"

**Use cases**:
- Vision-language models (CLIP, BLIP, LLaVA, Qwen3-VL)
- Video transformers (spatial position + temporal position)
- Multi-image VLMs (spatial position within each image + sequence order across images)

---

## Section 1: What is Dual Position Encoding? (70 lines)

### Motivation: Vision-Language Token Sequences

VLMs process **interleaved vision-text sequences**:

```
Sequence:
[text_0, text_1, text_2,        # Query: "What is in this image?"
 img_patch_0,0, img_patch_0,1, img_patch_0,2, ..., img_patch_H,W,  # Image tokens
 text_3, text_4]                # Response: "A cat on a mat"
```

Each image patch has:
- **Spatial position**: (row, col) in the image grid (e.g., patch_3,5 = row 3, column 5)
- **Sequential position**: Position in the full token sequence (e.g., token #15 overall)

**Problem**: Standard 1D position encoding only captures sequential position, losing spatial relationships within the image.

### Two Position Embedding Spaces

**Spatial position encoding** (2D):
```python
# For image patch at (row=3, col=5):
spatial_pe = get_2d_position_encoding(row=3, col=5, d_model=768)
```

Captures spatial relationships:
- Patch (3, 5) is **near** patch (3, 6) [horizontal neighbor]
- Patch (3, 5) is **below** patch (2, 5) [vertical neighbor]

**Sequential position encoding** (1D):
```python
# For token at position 15 in the sequence:
sequential_pe = get_1d_position_encoding(position=15, d_model=768)
```

Captures temporal/sequential relationships:
- Token 15 comes **after** token 14
- Token 15 comes **before** token 16

### Combining Dual Encodings

**Method 1: Addition**
```python
combined_pe = spatial_pe + sequential_pe
token_embedding = token + combined_pe
```

**Method 2: Concatenation**
```python
combined_pe = concat([spatial_pe, sequential_pe])  # d_model → 2*d_model
token_embedding = concat([token, combined_pe])
```

**Method 3: Interleaved (Qwen3-VL M-RoPE)**
```python
# Apply spatial RoPE to spatial dimensions
# Apply sequential RoPE to sequential dimensions
# Different frequency bands for each type
```

---

## Section 2: Spatial Position Encoding (2D) (70 lines)

### Image Patch Spatial Layout

Images are tokenized into patches arranged in a **2D grid**:

```
Image (224×224) → Patches (16×16) → Grid (14×14)

Spatial positions:
(0,0)  (0,1)  (0,2)  ... (0,13)
(1,0)  (1,1)  (1,2)  ... (1,13)
...
(13,0) (13,1) (13,2) ... (13,13)
```

Each patch needs **2D coordinates** (row, col) to preserve spatial structure.

### 2D Sinusoidal Position Encoding

Extend 1D sinusoidal encoding to 2D:

```python
def get_2d_sinusoidal_pe(row, col, d_model):
    # Split d_model into two halves: one for row, one for col
    d_row = d_model // 2
    d_col = d_model - d_row

    # Row encoding (first half of d_model)
    pe_row = []
    for i in range(d_row // 2):
        angle = row / (10000 ** (2 * i / d_row))
        pe_row.append(sin(angle))
        pe_row.append(cos(angle))

    # Column encoding (second half of d_model)
    pe_col = []
    for i in range(d_col // 2):
        angle = col / (10000 ** (2 * i / d_col))
        pe_col.append(sin(angle))
        pe_col.append(cos(angle))

    # Concatenate row and column encodings
    return concat([pe_row, pe_col])
```

**Result**: Each patch embedding encodes its (row, col) position in the image.

### Learned 2D Position Tables

Alternative: Learn a position table for each (row, col):

```python
position_table = nn.Parameter(torch.randn(max_rows, max_cols, d_model))
spatial_pe = position_table[row, col, :]  # Lookup
```

**Trade-off**:
- **Learned**: Better performance, limited to training resolution
- **Sinusoidal**: Extrapolates to any resolution (zero-shot)

---

## Section 3: Sequential Position Encoding (1D) (70 lines)

### Interleaved Vision-Text Sequences

VLMs process tokens sequentially, mixing text and vision:

```
Token sequence:
[CLS,         # Special token (position 0)
 "What",      # Text token (position 1)
 "is",        # Text token (position 2)
 "in",        # Text token (position 3)
 "image",     # Text token (position 4)
 "?",         # Text token (position 5)
 patch_0,0,   # Vision token (position 6)
 patch_0,1,   # Vision token (position 7)
 ...,
 patch_13,13, # Vision token (position 201)
 "A",         # Text token (position 202)
 "cat"]       # Text token (position 203)
```

Each token needs **1D sequential position** to maintain order.

### Standard 1D Position Encoding

**Sinusoidal** (original Transformer):
```python
def get_1d_sinusoidal_pe(position, d_model):
    pe = []
    for i in range(d_model // 2):
        angle = position / (10000 ** (2 * i / d_model))
        pe.append(sin(angle))
        pe.append(cos(angle))
    return pe
```

**RoPE** (Rotary Position Embedding - GPT-Neo, Llama):
```python
# Rotate query and key vectors by angle proportional to position
theta = position / (10000 ** (2 * i / d_model))
Q_rotated = rotate(Q, theta)
K_rotated = rotate(K, theta)
```

### Causal Masking Preservation

Sequential position encoding must preserve **causal relationships** in autoregressive VLMs:

```
Causal mask:
Token i can attend to tokens [0, 1, 2, ..., i]
Token i CANNOT attend to tokens [i+1, i+2, ..., N]
```

This ensures:
- Text tokens can attend to previous text AND previous image patches
- Image patches can attend to all previous tokens
- Future tokens don't leak information (no time travel!)

---

## Section 4: Combining Dual Encodings (70 lines)

### Method 1: Addition (Most Common)

**Approach**: Add spatial and sequential embeddings element-wise

```python
# Get both encodings
spatial_pe = get_2d_pe(row=patch_row, col=patch_col, d_model=768)
sequential_pe = get_1d_pe(position=token_position, d_model=768)

# Combine by addition
combined_pe = spatial_pe + sequential_pe

# Add to token embedding
token_with_pe = token_embedding + combined_pe
```

**Pros**:
- Simple, no parameter increase
- Both encodings influence the representation equally

**Cons**:
- Addition may "mix" spatial and sequential information
- Hard to disentangle which encoding contributes what

**Used by**: Many VLMs (CLIP, BLIP, LLaVA)

### Method 2: Concatenation

**Approach**: Concatenate spatial and sequential embeddings

```python
# Get both encodings (smaller d_model each)
spatial_pe = get_2d_pe(row, col, d_model=384)        # Half size
sequential_pe = get_1d_pe(position, d_model=384)     # Half size

# Combine by concatenation
combined_pe = concat([spatial_pe, sequential_pe])    # Total: 768

# Add to token embedding
token_with_pe = token_embedding + combined_pe
```

**Pros**:
- Spatial and sequential information are **separate** in embedding space
- Model can learn to attend to spatial or sequential parts independently

**Cons**:
- Doubles the position embedding dimension (or halves effective encoding size)

**Used by**: Some video transformers (spatial + temporal concatenation)

### Method 3: Qwen3-VL M-RoPE (Interleaved Multi-Axis RoPE)

**Approach**: Apply RoPE separately to different frequency bands

```python
# Split d_model into three parts:
# - Spatial (2D: row, col)
# - Temporal (1D: frame index for video)
# - Sequential (1D: token position)

d_spatial = d_model // 3
d_temporal = d_model // 3
d_sequential = d_model - 2 * d_spatial

# Apply different RoPE rotations to each part
Q_rotated = concat([
    rope_2d(Q[:d_spatial], row, col),           # Spatial RoPE
    rope_1d(Q[d_spatial:2*d_spatial], frame),   # Temporal RoPE
    rope_1d(Q[2*d_spatial:], position)          # Sequential RoPE
])
```

**Pros**:
- Completely disentangled spatial/temporal/sequential encoding
- Each axis gets dedicated frequency bands
- Extrapolates better to unseen resolutions and sequence lengths

**Cons**:
- More complex implementation
- Requires careful frequency allocation

**Used by**: Qwen3-VL (see `02-rope-multiaxis-encoding.md` for details)

---

## Video Transformers: Spatial + Temporal Dual Encoding (40 lines)

### Three-Way Position Encoding

Video transformers often use **3D position encoding**:

1. **Spatial (2D)**: (height, width) within each frame
2. **Temporal (1D)**: Frame index in the video
3. **Sequential (1D)**: Token position in the overall sequence (if text is interleaved)

Example: Video with 8 frames, each 14×14 patches, plus text prompt:

```
Token sequence:
[text_0, text_1, ..., text_N,                    # Text query
 frame0_patch_0,0, ..., frame0_patch_13,13,      # Frame 0 (196 patches)
 frame1_patch_0,0, ..., frame1_patch_13,13,      # Frame 1 (196 patches)
 ...,
 frame7_patch_0,0, ..., frame7_patch_13,13]      # Frame 7 (196 patches)
```

Each video patch has:
- **Spatial position**: (row, col) in frame
- **Temporal position**: Frame index (0-7)
- **Sequential position**: Overall token position

### Factorized 3D Position Encoding

```python
def get_3d_pe(row, col, frame, position, d_model):
    d_spatial = d_model // 4
    d_temporal = d_model // 4
    d_sequential = d_model - 2 * d_spatial

    pe_spatial = get_2d_pe(row, col, d_spatial)
    pe_temporal = get_1d_pe(frame, d_temporal)
    pe_sequential = get_1d_pe(position, d_sequential)

    return concat([pe_spatial, pe_temporal, pe_sequential])
```

**Used by**: TimeSformer, ViViT, Video-LLaMA

---

## Best Practices and Trade-offs

### When to Use Dual Encoding

**Strong recommendation**:
- Vision-language models (interleaved vision-text)
- Video transformers (spatial + temporal structure)
- Multi-image VLMs (spatial within images + sequential across images)

**Optional**:
- Pure vision models (ViT) - spatial encoding alone is often sufficient
- Pure text models (GPT) - sequential encoding only

### Choosing Combination Method

**Addition** (default):
- Simplest implementation
- No parameter overhead
- Use when spatial and sequential information should be equally weighted

**Concatenation**:
- Use when spatial and sequential need to be disentangled
- Good for analysis (can measure spatial vs sequential attention)
- Acceptable parameter overhead (~2x position embedding size)

**M-RoPE / Interleaved**:
- Use for maximum extrapolation and flexibility
- Good for variable resolutions, sequence lengths, or frame counts
- More complex but worth it for production VLMs

### Implementation Tips

1. **Normalize encodings separately** before combining (prevents one from dominating)
2. **Use learnable scaling factors** to balance spatial vs sequential influence
3. **Test extrapolation** to longer sequences or higher resolutions early
4. **Monitor attention patterns** to ensure both encodings are used effectively

---

## Key Takeaways

1. **Dual position encoding** combines spatial (2D) and sequential (1D) position information for multimodal models
2. **Spatial encoding** preserves image structure (row, col), **sequential encoding** preserves token order
3. **Three combination methods**: Addition (simple), Concatenation (disentangled), M-RoPE (extrapolation)
4. **Video transformers** extend to 3D: spatial (height, width) + temporal (frame) + sequential (token position)
5. **Qwen3-VL M-RoPE** is state-of-the-art: interleaved multi-axis RoPE with dedicated frequency bands

---

## Sources

**Academic Papers**:
- arXiv:2510.23095v1 - "Revisiting Multimodal Positional Encoding in Vision-Language Models" (2024) - accessed 2025-01-31
  - URL: https://arxiv.org/html/2510.23095v1
  - Key insight: "MRoPE position design with spatial-reset improves model's focus on visual information"

- arXiv:2506.19651v1 - "PEVLM: Parallel Encoding for Vision-Language Models" (June 2024) - accessed 2025-01-31
  - URL: https://arxiv.org/html/2506.19651v1
  - Key insight: Parallel encoding strategies for prefill efficiency in VLMs

- CVPR 2025 - "MASH-VLM: Mitigating Action-Scene Hallucination through Disentangled Spatial-Temporal Representations" (Bae et al., 2025) - accessed 2025-01-31
  - URL: http://openaccess.thecvf.com/content/CVPR2025/papers/Bae_MASH-VLM_Mitigating_Action-Scene_Hallucination_in_Video-LLMs_through_Disentangled_Spatial-Temporal_Representations_CVPR_2025_paper.pdf
  - Key insight: Disentangled spatial-temporal representations for video VLMs

- ACL Anthology: EMNLP 2024.emnlp-main.793 - "Shaking Up VLMs: Comparing Transformers and Structured State Space Models" (Pantazopoulos et al., 2024) - accessed 2025-01-31
  - URL: https://aclanthology.org/2024.emnlp-main.793.pdf
  - Key insight: SSMs vs Transformers for multimodal position encoding

**Technical Articles**:
- Hugging Face Blog - "You could have designed state of the art positional encoding" (November 2024) - accessed 2025-01-31
  - URL: https://huggingface.co/blog/designing-positional-encoding
  - Step-by-step discovery of modern position encoding techniques

- Medium (Deepankar Singh) - "Design Guidelines for Incorporating Positional Encodings into New Models" (2024) - accessed 2025-01-31
  - URL: https://medium.com/ai-enthusiast/design-guidelines-for-incorporating-positional-encodings-into-new-models-d6eec1e9ea32
  - Position encoding design patterns and best practices

- LearnOpenCV - "Inside RoPE: Rotary Magic into Position Embeddings" (July 2024) - accessed 2025-01-31
  - URL: https://learnopencv.com/rope-position-embeddings/
  - Deep dive on RoPE mechanics and multi-axis extensions

**Related Papers**:
- ScienceDirect - "Grounding spatial relations in text-only language models" (Azkune et al., 2024) - accessed 2025-01-31
  - URL: https://www.sciencedirect.com/science/article/pii/S089360802300655X
  - Spatial reasoning in language models without vision

**Search Queries**:
- "dual position encoding spatial sequential VLM 2024" (Google, accessed 2025-01-31)

**Related Knowledge**:
- See `02-rope-multiaxis-encoding.md` for detailed M-RoPE implementation
- See `03-2d-positional-encoding.md` for spatial position encoding methods
- See `10-token-sequence-order-importance.md` for why sequential position matters
- See `04-sequence-vs-spatial-attention.md` for attention mechanism differences
