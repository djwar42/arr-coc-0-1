# Multi-Modal Tokenization Strategies for Vision-Language Models

**Knowledge expansion topic**: Multi-modal tokenization - strategies for combining visual and text tokens in unified sequences

**Date**: 2025-11-14

---

## Overview

Multi-modal tokenization is the process of converting heterogeneous inputs (images and text) into unified token sequences that can be processed by transformer-based vision-language models (VLMs). Unlike text-only models where tokenization is straightforward (BPE, SentencePiece), VLMs must handle fundamentally different data types with vastly different characteristics:

- **Text tokens**: Discrete, sequential, semantic units (typically 1-100 tokens per sentence)
- **Visual tokens**: High-dimensional, spatially structured representations (typically 64-576 tokens per image)

The tokenization strategy determines how these modalities are merged, ordered, and fed to the model, directly impacting:
1. **Computational efficiency**: Quadratic attention cost means every token counts
2. **Context window utilization**: Visual tokens can consume 90%+ of available context
3. **Cross-modal reasoning**: Token arrangement affects how vision and language interact
4. **Task performance**: Different tasks benefit from different tokenization approaches

**Key insight from existing knowledge**: Token budgets are critical - reducing from 576 to 64-144 tokens achieves 80-90% of performance with 4-10× speedup (from [51-vision-token-budgets.md](../../karpathy/practical-implementation/51-vision-token-budgets.md)).

---

## Section 1: Text Tokenization Fundamentals (60 lines)

### Subword Tokenization Methods

Modern VLMs inherit text tokenization from their language model backbones:

**Byte-Pair Encoding (BPE)** (GPT, LLaMA):
```python
# BPE merges most frequent byte pairs iteratively
text = "tokenization"
# Iteration 1: "token" + "ization"
# Iteration 2: "tok" + "en" + "iz" + "ation"
# Final: ["tok", "en", "ization"]  # 3 tokens
```

**Key properties**:
- Vocabulary size: 32k-50k tokens (GPT-3, LLaMA)
- Out-of-vocabulary handling: Any text can be represented (fallback to bytes)
- Compression ratio: ~4 characters per token (English)

**SentencePiece** (T5, BERT, many VLMs):
```python
# Unigram language model or BPE, treats spaces as tokens
text = "Hello world"
# Output: ["▁Hello", "▁world"]  # ▁ = space marker
```

**Advantages over BPE**:
- Language-agnostic (no pre-tokenization needed)
- Reversible (can reconstruct original text exactly)
- Handles whitespace explicitly

**WordPiece** (BERT):
- Similar to BPE but uses likelihood-based merging
- Vocabulary: 30k tokens
- Common in earlier VLMs (VisualBERT, LXMERT)

### Special Tokens for Multi-Modal Sequences

VLMs extend text tokenization with modality-specific markers:

**Image boundary tokens**:
```python
# LLaVA-style
sequence = ["<image>", visual_token_1, ..., visual_token_576, "</image>", text_tokens...]

# Flamingo-style (gated cross-attention)
sequence = [text_tokens..., "<img>", "<img>", ...]  # Repeated markers for multiple images
```

**System tokens**:
```
<s>  # Start of sequence
</s> # End of sequence
<pad> # Padding token
<unk> # Unknown token
<|im_start|> # Image start (Qwen3-VL style)
<|im_end|>   # Image end
```

**Example full sequence** (LLaVA with image + question):
```
<s> <image> [576 visual tokens] </image> What is in this image? </s>
```

### Text Token Characteristics

**Semantic density**: Each text token carries significant meaning
- "dog" = single concept
- Visual token = 16×16 patch, may contain partial objects

**Sequence length variability**:
- Short prompt: 5-20 tokens
- Long prompt: 100-500 tokens
- Document: 1000+ tokens

**Positional importance**:
- Causal models: Order is critical ("dog bites man" ≠ "man bites dog")
- Bidirectional models: Less order-sensitive (see [10-token-sequence-order-importance.md](../../karpathy/vision-language/10-token-sequence-order-importance.md))

From [Token Sequence Order Importance](../../karpathy/vision-language/10-token-sequence-order-importance.md):
> "Transformers are permutation-equivariant without position encoding - reordering inputs reorders outputs identically. Position encoding is what injects order awareness."

---

## Section 2: Vision Tokenization - From Pixels to Tokens (80 lines)

### Patch-Based Tokenization (ViT Standard)

Vision transformers divide images into fixed-size patches:

**Grid tokenization**:
```python
# Image: 224×224, Patch size: 16×16
num_patches_h = 224 // 16  # = 14
num_patches_w = 224 // 16  # = 14
total_tokens = 14 × 14 = 196

# Each patch becomes a token
patch = image[i*16:(i+1)*16, j*16:(j+1)*16]  # 16×16×3 RGB
token = linear_projection(patch.flatten())    # → D-dimensional embedding
```

**Raster scan order** (top-left to bottom-right):
```
Patch sequence:
[patch_0,0, patch_0,1, ..., patch_0,13,   # Row 0
 patch_1,0, patch_1,1, ..., patch_1,13,   # Row 1
 ...
 patch_13,0, ..., patch_13,13]            # Row 13
```

**Common configurations**:

| Resolution | Patch Size | Grid | Tokens | Use Case |
|------------|------------|------|---------|----------|
| 224×224 | 16×16 | 14×14 | 196 | Standard (CLIP ViT-B/16) |
| 336×336 | 14×14 | 24×24 | 576 | High-res (CLIP ViT-L/14) |
| 384×384 | 16×16 | 24×24 | 576 | Alternative high-res |
| 448×448 | 14×14 | 32×32 | 1024 | Ultra-high-res |

**Position encoding for vision tokens**:
```python
# 2D sinusoidal encoding (common)
pos_embed_2d = get_2d_sinusoidal_encoding(height=14, width=14, embed_dim=768)

# Learned 2D encoding (ViT)
pos_embed_learnable = nn.Parameter(torch.randn(1, 196, 768))

# Relative position encoding (Swin Transformer)
# Encodes relative distances between patches
```

### Image Slicing for High-Resolution (LLaVA-style)

To process images larger than the vision encoder's native resolution:

**Dynamic grid slicing**:
```python
# Input: 1280×720 image
# Target: 336×336 patches (CLIP ViT-L/14)

# Slice into grid
grid_h = ceil(1280 / 336)  # = 4 slices
grid_w = ceil(720 / 336)   # = 3 slices

# Creates 4×3 = 12 overlapping patches
# Each patch: 336×336 → 576 tokens
# Total: 12 × 576 = 6,912 tokens!
```

From [LLaVA Architecture: From Frozen ViT to Fine-Tuned LLM](https://learnopencv.com/llava-training-a-visual-assistant/) (accessed 2025-11-14):
> "Image Slicing: The input image is divided into a grid of smaller patches, each sized to match the native input resolution of the CLIP encoder."

**Strategies to manage token explosion**:

1. **Adaptive slicing** (LLaVA-1.5):
   - Simple images: Single 336×336 patch (576 tokens)
   - Complex images: 2×2 grid (4 × 576 = 2,304 tokens)

2. **Downsampling + slicing**:
   - Resize to intermediate resolution first
   - Then slice into encoder-compatible patches

3. **Hierarchical encoding** (HiRes-LLaVA):
   - Low-resolution global view (576 tokens)
   - High-resolution local patches (adaptive)
   - Combine features intelligently

From [HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf) (CVPR 2025, accessed 2025-11-14):
> "A novel framework designed to efficiently process high-resolution inputs of any size without altering the original contextual and geometric information."

### Learned Token Representations

**CLIP vision tokens**:
```python
# Pre-trained CLIP ViT produces aligned embeddings
image_features = clip_vision_encoder(image)  # [196, 1024] for ViT-L/14

# Already aligned to text space via contrastive learning
# No explicit tokenization, features ARE the tokens
```

**Discrete visual tokens** (VQVAE-based):
```python
# Vector-quantized image representation (CM3, DALL-E)
image_tokens = vqvae_encoder(image)  # Discrete codebook indices
# 256×256 image → 32×32 grid → 1024 discrete tokens
# Each token ∈ {0, 1, ..., vocab_size-1}

# Advantage: Can use same vocabulary as text
# Disadvantage: Lossy compression, limited fidelity
```

**Comparison**:
- **Continuous tokens** (CLIP): Rich, high-fidelity, used by most modern VLMs
- **Discrete tokens** (VQVAE): Enables unified generation (text + images), but lower quality

---

## Section 3: Token Concatenation Strategies (90 lines)

### Prefix Concatenation (Most Common)

Visual tokens prepended to text sequence:

```python
# LLaVA, Qwen3-VL, most modern VLMs
sequence = [visual_tokens, text_tokens]

# Example
visual_tokens = vision_encoder(image)  # [576, D]
text_tokens = tokenizer("What is this?")  # [5, D]
full_sequence = concat([visual_tokens, text_tokens], dim=0)  # [581, D]

# LLM processes: [v1, v2, ..., v576, t1, t2, t3, t4, t5]
```

**Advantages**:
- **Simple**: No architectural changes to LLM
- **Causal compatibility**: Text can attend to all visual tokens
- **Natural order**: Image "context" comes before question

**Disadvantages**:
- **Context pressure**: 576 visual tokens consume significant context window
- **Redundancy**: Adjacent patches often similar (sky, grass, backgrounds)

From [Vision-Language Model Token Concatenation Strategies](../../karpathy/vision-language/00-token-concatenation-strategies.md):
> "With merely 8 visual registers - about 1% of the original tokens - Victor shows less than 4% accuracy drop while reducing total training time by 43% and boosting inference throughput by 3.3×."

### Suffix Concatenation (Less Common)

Text tokens followed by visual tokens:

```python
sequence = [text_tokens, visual_tokens]

# Example: "Describe <image>"
# Tokens: ["Describe", "<image_start>", v1, v2, ..., v576]
```

**Used when**:
- Instruction comes before image reference
- Multi-turn dialogues with late image insertion

**Challenge**: Causal masking prevents early text tokens from seeing image
- Only tokens AFTER image can attend to visual information
- Limits usefulness for question answering

### Interleaved Concatenation (Multi-Image)

Arbitrary mixing of text and visual tokens:

```python
# Multiple images with text
sequence = [text1, visual_tokens_img1, text2, visual_tokens_img2, text3]

# Example: "First <img1>, then <img2>. What changed?"
# Tokens: ["First", v1_1, ..., v1_576, "then", v2_1, ..., v2_576, "What", "changed"]
```

**Applications**:
- Document understanding (text + figures + tables)
- Multi-image reasoning (compare two images)
- Video understanding (frames as image sequence)

From [Interleaved Image-Text Generative Modeling](https://arxiv.org/abs/2401.10208) (MM-Interleaved, arXiv:2401.10208, accessed 2025-11-14):
> "This paper presents MM-Interleaved, an end-to-end generative model for interleaved image-text data. It introduces a multi-scale and multi-image feature synchronizer."

**Attention masking for interleaved sequences**:
```python
# Causal mask ensures each token only sees previous tokens
# Text token at position i can attend to:
# - All previous text tokens
# - Visual tokens from previous images only

# Example mask (T=text, I=image):
# Sequence: [T1, I1, I2, T2, I3, T3]
# T1 sees: [T1]
# I1 sees: [T1, I1]
# I2 sees: [T1, I1, I2]
# T2 sees: [T1, I1, I2, T2]
# I3 sees: [T1, I1, I2, T2, I3]
# T3 sees: [T1, I1, I2, T2, I3, T3]
```

**CoMM Dataset** (high-quality interleaved data):

From [A Coherent Interleaved Image-Text Dataset for Multimodal Understanding](https://arxiv.org/abs/2406.10462) (CoMM, arXiv:2406.10462, accessed 2025-11-14):
> "CoMM is a high-quality dataset for interleaved image-text generation, designed to enhance coherence, consistency, and alignment of multimodal content."

### Window-Based Concatenation (Efficiency)

Sliding window approach to reduce token count:

**WiCo (Window Token Concatenation)**:
```python
# Original: 24×24 grid = 576 tokens
# Window size: 2×2
# Output: 12×12 = 144 tokens (4× reduction)

tokens_grid = tokens.reshape(24, 24, D)
windows = []
for i in range(0, 24, 2):
    for j in range(0, 24, 2):
        window = tokens_grid[i:i+2, j:j+2, :]  # 2×2 window
        merged_token = window.mean(dim=(0,1))  # Average pooling
        windows.append(merged_token)

compressed_tokens = stack(windows)  # [144, D]
```

**WiCo+ Enhancement**:
- Early LLM layers: Use 144 compressed tokens (efficient)
- Late LLM layers: Decompress to higher resolution for fine-grained tasks
- Learnable decompression module

From [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024) (WiCo, arXiv:2504.04024, accessed 2025-11-14):
> "We employ a sliding window to concatenate spatially adjacent visual tokens. However, directly concatenating these tokens may group diverse tokens into one, and thus obscure some fine details."

**Performance**:
- VQAv2: 80.1% (WiCo+) vs 78.5% (LLaVA-1.5 baseline)
- 4× token reduction (576 → 144)
- 2.3× inference speedup

---

## Section 4: Sequence Order and Position Encoding (70 lines)

### Order Importance in Multi-Modal Sequences

Unlike pure vision (where spatial position > sequence order), text order is critical:

**Text sequence order**:
```python
# Order matters!
"dog bites man" ≠ "man bites dog"
"What color is the car?" ≠ "color What car the is?"
```

**Visual sequence order** (less critical):
```python
# Raster scan order is conventional but not semantic
# Permuting patches degrades performance but doesn't destroy meaning
# (Unlike permuting words in text)
```

From [Token Sequence Order Importance](../../karpathy/vision-language/10-token-sequence-order-importance.md):
> "Vision transformers show low order sensitivity - spatial position matters more than sequence order. Permuting patches breaks spatial locality but doesn't destroy semantic content as severely as permuting words."

### Multi-Modal Position Encoding

**Absolute position encoding** (standard):
```python
# Separate encodings for vision and text
vision_pos_embed = get_2d_sinusoidal(height=24, width=24, dim=1024)
text_pos_embed = get_1d_sinusoidal(max_len=512, dim=1024)

# Concatenate
full_pos_embed = concat([vision_pos_embed, text_pos_embed], dim=0)
```

**Relative position encoding** (RoPE for text):
```python
# Qwen3-VL uses Interleaved Multi-axis RoPE
# Different axes for temporal, height, width dimensions
# Enables dynamic resolution and video understanding

# Text positions: Standard RoPE
# Visual positions: 2D RoPE (row + column)
```

From [Qwen3-VL architecture](../../qwen3vl-oracle/) (existing knowledge):
> "Interleaved-MRoPE enables Qwen3-VL to handle any image resolution without training - extends position encoding to 2D/3D for vision."

**Learned position embeddings**:
```python
# Vision transformer style
pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

# Added to token embeddings
tokens_with_pos = tokens + pos_embed
```

### Special Tokens as Positional Markers

**Boundary markers** help model distinguish modalities:

```python
# Explicit modality boundaries
sequence = [
    "<text_start>", "What", "is", "this", "<text_end>",
    "<image_start>", v1, v2, ..., v576, "<image_end>",
    "<text_start>", "Describe", "it", "<text_end>"
]

# Enables model to learn modality-specific processing
```

**Segment embeddings** (BERT-style):
```python
# Modality type embedding added to each token
text_segment_embed = nn.Parameter(torch.randn(1, embed_dim))
vision_segment_embed = nn.Parameter(torch.randn(1, embed_dim))

# Applied during concatenation
text_tokens_final = text_tokens + text_segment_embed
vision_tokens_final = vision_tokens + vision_segment_embed
```

---

## Section 5: Dynamic Token Allocation (Query-Aware) (80 lines)

### Relevance-Based Token Selection

Instead of fixed token counts, allocate based on query relevance:

**Attention-driven pruning**:
```python
# Compute query-image attention scores
query_embed = encode_text("What color is the car?")
attention_scores = query_embed @ visual_tokens.T  # [576]

# Keep top-K most relevant tokens
K = 144  # Budget
top_indices = attention_scores.topk(K).indices
selected_tokens = visual_tokens[top_indices]  # [144, D]
```

**SparseVLM approach**:

From [SparseVLM: Visual Token Sparsification](https://arxiv.org/abs/2410.04417) (accessed 2025-11-14):
> "Visual tokens usually bear a significant amount of computational overhead despite sparsity of information in them when compared to text tokens."

**Results**: 54% reduction in FLOPs, 37% decrease in latency, maintaining 97% accuracy.

### Progressive Token Dropping (Layer-by-Layer)

Reduce tokens gradually through network:

**Dynamic-VLM strategy**:
```python
# Layer-wise token reduction schedule
layer_0_6:   576 tokens  # Early: full detail
layer_7_15:  256 tokens  # Mid: prune 55%
layer_16_23: 144 tokens  # Late: prune 75%
layer_24_31: 64 tokens   # Final: prune 89%

# Total compute saved: ~60% vs fixed 576 tokens
```

From [Dynamic Token Reduction during Generation for Vision-Language Models](https://arxiv.org/html/2501.14204v1) (accessed 2025-11-14):
> "We introduce a dynamic pruning strategy tailored for VLMs, named Dynamic Rate (DyRate), which progressively adjusts the token count."

**MustDrop (importance-based)**:

From [Multi-Stage Vision Token Dropping](https://arxiv.org/abs/2411.10803) (accessed 2025-11-14):
> "MustDrop measures the importance of each token from the whole lifecycle, enabling layer-by-layer adaptive pruning with 80% token reduction and <4% accuracy drop."

### Query-Conditioned Budgets (ARR-COC Connection)

Allocate tokens based on task requirements:

**Adaptive budget function**:
```python
def query_aware_budget(query, image_complexity):
    base_budget = 144

    # Spatial reasoning tasks need more tokens
    if "where" in query or "count" in query:
        budget = base_budget * 1.5  # 216 tokens

    # Semantic tasks need fewer
    elif "what" in query or "describe" in query:
        budget = base_budget * 1.0  # 144 tokens

    # Image complexity adjustment
    if image_complexity > 0.8:  # Dense scene
        budget *= 1.3
    elif image_complexity < 0.3:  # Simple scene
        budget *= 0.7

    return clip(budget, min=64, max=400)
```

**Example allocations**:
```
Query: "What color is the car?"
→ Simple semantic query
→ Focus on vehicle region (high relevance)
→ Budget: 144 tokens (standard)

Query: "Count the people in the image"
→ Spatial reasoning required
→ Distribute across whole image
→ Budget: 288 tokens (2× baseline)

Query: "Describe the entire scene"
→ Comprehensive coverage needed
→ Even distribution
→ Budget: 400 tokens (maximum)
```

### TokenFLEX (Flexible Visual Tokens)

From [Unified VLM Training for Flexible Visual Tokens Inference](https://arxiv.org/html/2504.03154v1) (accessed 2025-11-14):
> "TokenFLEX is a novel vision-language framework that liberates VLMs from fixed visual token constraints. Our approach enables dynamic token allocation."

**Training approach**:
- Train with variable token counts (64, 144, 256, 400)
- Model learns to adapt to different budgets
- Inference-time flexibility without retraining

**Benefits**:
- Fast inference: Use 64 tokens for simple queries
- Accurate reasoning: Use 400 tokens for complex queries
- No separate models needed

---

## Section 6: Special Modality Tokens and Embeddings (50 lines)

### Image Boundary Tokens

Different VLMs use different conventions:

**LLaVA style** (explicit boundaries):
```python
tokens = ["<image>"] + visual_tokens + ["</image>"] + text_tokens
# Advantages: Clear modality separation
# Model can learn special processing for content between markers
```

**Flamingo style** (implicit):
```python
# No explicit markers
# Cross-attention layers handle vision-language fusion
# Text tokens attend to visual tokens via gated cross-attention
```

**Qwen3-VL style** (vision markers):
```python
tokens = ["<|im_start|>"] + visual_tokens + ["<|im_end|>"] + text_tokens
# Also supports multi-image: <|im_start|> img1 <|im_end|> text <|im_start|> img2 <|im_end|>
```

### Padding and Masking

**Batch processing** requires consistent sequence lengths:

```python
# Batch with variable image counts
batch = [
    [img1_tokens, text1],  # 576 + 20 = 596 tokens
    [img2_tokens, text2],  # 576 + 50 = 626 tokens
]

# Pad to max length
max_len = 626
batch_padded = [
    pad(batch[0], max_len),  # Add 30 <pad> tokens
    batch[1]  # Already max length
]

# Attention mask (1 = attend, 0 = ignore padding)
mask = [
    [1]*596 + [0]*30,
    [1]*626
]
```

**Efficient masking**:
- Only compute attention over non-padded tokens
- Saves compute in forward pass
- Critical for variable-length sequences

---

## Section 7: ARR-COC-0-1 Tokenization Strategy (60 lines)

### Relevance-Driven Variable LOD

ARR-COC-0-1 implements **query-conditioned adaptive tokenization** based on Vervaekean relevance realization:

**Three Ways of Knowing** guide token selection:

1. **Propositional knowing** (information content):
   ```python
   # Shannon entropy per patch
   entropy = -sum(p * log(p) for p in patch_distribution)
   # High entropy → more information → allocate more tokens
   ```

2. **Perspectival knowing** (salience):
   ```python
   # Sobel edge detection, LAB color variance
   saliency = compute_saliency(patch)
   # High saliency → visually interesting → allocate more tokens
   ```

3. **Participatory knowing** (query-content coupling):
   ```python
   # Query-patch relevance
   relevance = query_embedding @ patch_embedding
   # High relevance → important for query → allocate more tokens
   ```

**Token budget allocation**:
```python
# Per-patch budget: 64-400 tokens based on relevance
for patch in image_patches:
    # Compute relevance score (0-1)
    relevance = (
        0.33 * propositional_score(patch) +
        0.33 * perspectival_score(patch) +
        0.34 * participatory_score(patch, query)
    )

    # Map to token count
    if relevance > 0.8:
        patch_tokens = 400  # High detail
    elif relevance > 0.5:
        patch_tokens = 256  # Medium detail
    elif relevance > 0.3:
        patch_tokens = 144  # Standard detail
    else:
        patch_tokens = 64   # Minimal detail
```

**Example allocation**:
```
Query: "What is the person on the left holding?"

Patch relevance distribution:
┌────────┬────────┬────────┐
│ Sky    │ Sky    │ Sky    │
│ 64 tok │ 64 tok │ 64 tok │
├────────┼────────┼────────┤
│ Person │ Object │ Background │
│ 400tok │ 256tok │ 144 tok │
├────────┼────────┼────────┤
│ Ground │ Ground │ Ground │
│ 64 tok │ 64 tok │ 64 tok │
└────────┴────────┴────────┘

Total: 400 + 256 + 144 + (6 × 64) = 1,184 tokens
Average: 132 tokens/patch (vs 256 fixed)
```

**Efficiency gains**:
- Uniform allocation: 9 patches × 256 tokens = 2,304 tokens
- Relevance-driven: 1,184 tokens (48% reduction)
- Performance: Maintains accuracy on query-relevant regions
- Speedup: ~2× faster inference

**Integration with texture array**:
```python
# 13-channel texture array
texture_channels = [
    "R", "G", "B",           # RGB color
    "L", "A", "B",           # LAB color space
    "Sobel_H", "Sobel_V",    # Edge detection
    "spatial_x", "spatial_y", # Spatial position
    "eccentricity",          # Distance from center
    "semantic_class",        # Object category
    "depth"                  # Depth estimation
]

# Rich features enable better relevance scoring
# Example: Sobel channels → high edge regions get more tokens
```

From [ARR-COC-0-1 architecture](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/README.md):
> "13-channel texture array (RGB, LAB, Sobel, spatial, eccentricity) enables rich visual representations for relevance-aware token allocation."

---

## Section 8: Implementation Considerations (50 lines)

### Tokenizer Training and Alignment

**Vision encoder tokenizer** (pre-trained):
```python
# CLIP ViT already produces aligned embeddings
vision_tokenizer = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
# Output: [num_patches, 1024] continuous embeddings
# No training needed, frozen during VLM training
```

**Text tokenizer** (language model):
```python
# Use backbone's tokenizer
text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# Vocabulary: 32,000 tokens
# Already trained, frozen
```

**Projection layer** (trainable):
```python
# Align vision embeddings to LLM space
vision_projection = nn.Sequential(
    nn.Linear(1024, 4096),  # CLIP → LLaMA dimensions
    nn.GELU(),
    nn.Linear(4096, 4096)
)
# Only component trained during visual instruction tuning
```

### Batch Processing Challenges

**Variable image counts**:
```python
# Batch with different numbers of images
batch = [
    {"images": [img1], "text": "What is this?"},           # 1 image
    {"images": [img2, img3], "text": "Compare these."},    # 2 images
]

# Strategy 1: Pad to max images
max_images = 2
batch_padded = [
    {"images": [img1, dummy_image], "text": "..."},
    {"images": [img2, img3], "text": "..."}
]
# Wasteful: Processes dummy image

# Strategy 2: Pack sequences efficiently
# Concatenate all images + text in single sequence
# Use attention mask to prevent cross-sample attention
```

**Memory considerations**:
```python
# GPU memory for batch_size=8, 576 tokens/image
activations = 8 * 576 * 4096 * 2 bytes  # ~150 MB
KV_cache = 8 * 576 * 32_layers * 4096 * 2 bytes  # ~4.8 GB
total = activations + KV_cache  # ~5 GB just for visual tokens!
```

### Efficiency Optimizations

**FlashAttention-2** for visual tokens:
```python
# Standard attention: O(n²) memory
# FlashAttention-2: O(n) memory, 2-4× speedup

# Example savings for 576 tokens
standard_memory = 576 * 576 * 4 bytes  # ~1.3 MB per head
flash_memory = 576 * 4 bytes           # ~2.3 KB per head
# 560× memory reduction!
```

**Token pruning** (training-free):
```python
# Prune low-attention tokens progressively
# Keep only tokens with cumulative attention > threshold

def prune_tokens(tokens, attention_weights, threshold=0.9):
    importance = attention_weights.sum(dim=0)  # [num_tokens]
    sorted_indices = importance.argsort(descending=True)

    cumsum = 0
    keep_count = 0
    for idx in sorted_indices:
        cumsum += importance[idx]
        keep_count += 1
        if cumsum >= threshold * importance.sum():
            break

    return tokens[sorted_indices[:keep_count]]

# Typical reduction: 576 → 200 tokens (65% reduction)
```

---

## Key Takeaways

1. **Text tokenization is mature** (BPE, SentencePiece) - VLMs inherit from language models
2. **Vision tokenization is patch-based** - fixed grids of 196-1024 tokens per image
3. **Concatenation order matters** - prefix (image first) is most common and effective
4. **Token explosion is real** - multiple images or high-res can exceed context windows
5. **Dynamic allocation is key** - query-aware budgets outperform fixed token counts
6. **Efficiency techniques are critical** - pruning, compression, FlashAttention enable practical deployment
7. **ARR-COC-0-1 connection**: Relevance realization provides principled framework for dynamic tokenization (64-400 tokens based on knowing + balancing)

**Future directions**:
- End-to-end learned tokenization (trainable patch sizes)
- Semantic grouping (object-aware tokens, not fixed grids)
- Cross-modal token generation (diffusion-based visual tokens)
- Ultra-efficient tokenization (10-50 tokens per image with minimal loss)

---

## Sources

**Source Documents**:
- [00-token-concatenation-strategies.md](../../karpathy/vision-language/00-token-concatenation-strategies.md) - Token concatenation strategies for VLMs
- [10-token-sequence-order-importance.md](../../karpathy/vision-language/10-token-sequence-order-importance.md) - Sequence order importance in transformers
- [51-vision-token-budgets.md](../../karpathy/practical-implementation/51-vision-token-budgets.md) - Vision token budgets and performance trade-offs

**Web Research**:
- [Revisiting Visual Token Pruning for Vision-Language Models](https://arxiv.org/html/2412.13180v2) - arXiv (accessed 2025-11-14)
- [LLaVA Architecture: From Frozen ViT to Fine-Tuned LLM](https://learnopencv.com/llava-training-a-visual-assistant/) - LearnOpenCV (accessed 2025-11-14)
- [HiRes-LLaVA: Restoring Fragmentation Input](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf) - CVPR 2025 (accessed 2025-11-14)
- [Window Token Concatenation for Efficient VLMs](https://arxiv.org/abs/2504.04024) - arXiv:2504.04024 (accessed 2025-11-14)
- [Interleaved Image-Text Generative Modeling](https://arxiv.org/abs/2401.10208) - MM-Interleaved, arXiv:2401.10208 (accessed 2025-11-14)
- [CoMM: A Coherent Interleaved Image-Text Dataset](https://arxiv.org/abs/2406.10462) - arXiv:2406.10462 (accessed 2025-11-14)
- [SparseVLM: Visual Token Sparsification](https://arxiv.org/abs/2410.04417) - arXiv:2410.04417 (accessed 2025-11-14)
- [Dynamic Token Reduction for Vision-Language Models](https://arxiv.org/html/2501.14204v1) - arXiv (accessed 2025-11-14)
- [Multi-Stage Vision Token Dropping](https://arxiv.org/abs/2411.10803) - MustDrop, arXiv:2411.10803 (accessed 2025-11-14)
- [Unified VLM Training for Flexible Visual Tokens](https://arxiv.org/html/2504.03154v1) - TokenFLEX, arXiv (accessed 2025-11-14)

**Additional References**:
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - HuggingFace (accessed 2025-11-14)
- [OmniJARVIS Unified Vision-Language-Action Tokenization](https://proceedings.neurips.cc/paper_files/paper/2024/file/85f1225db986e629289f402c46eff1a4-Paper-Conference.pdf) - NeurIPS 2024 (accessed 2025-11-14)
