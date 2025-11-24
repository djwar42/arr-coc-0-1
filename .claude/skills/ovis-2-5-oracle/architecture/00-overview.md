# Ovis 2.5 Architecture Overview

**Category**: Architecture
**Related**: All architecture files
**Code**: `RESEARCH/Ovis25/Ovis/ovis/model/`

## Overview

Ovis 2.5 is a novel Multimodal Large Language Model (MLLM) architecture designed to **structurally align visual and textual embeddings** through the Visual Embedding Table (VET).

**Key Innovation**: Unlike standard vision-language models that use continuous visual features, Ovis introduces a probabilistic discrete embedding lookup that better aligns with the discrete nature of language tokens.

## System Architecture

```
Image Input
    ↓
┌─────────────────────────────────────┐
│ NaViT (Native Vision Transformer)   │
│ - SigLIP 2 backbone                 │
│ - Native resolution (no fixed grid) │
│ - RoPE positional encoding          │
│ Output: Visual features [B, N, D]   │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Visual Tokenizer (VT)               │
│ - ViT encoding                      │
│ - Visual head projection            │
│ - Smart resize algorithm            │
│ Output: Probability dist [B, N, V]  │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Visual Embedding Table (VET)       │
│ - Probabilistic lookup              │
│ - Structural alignment              │
│ - Learnable embeddings              │
│ Output: Discrete embeds [B, N, D]   │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Multimodal Merging                  │
│ - Replace <image> tokens            │
│ - Merge with text embeddings        │
│ - Create attention masks            │
│ Output: Merged sequence [B, L, D]   │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Qwen3 Language Model                │
│ - Decoder-only transformer          │
│ - Autoregressive generation         │
│ - Optional thinking mode            │
│ Output: Text tokens [B, L]          │
└─────────────────────────────────────┘
    ↓
Text Output
```

## Core Components

### 1. NaViT (Native Vision Transformer)

**File**: `ovis/model/vit/modeling_siglip2_navit.py`

**Key Features**:
- **Native resolution**: No fixed tiling, preserves aspect ratios
- **SigLIP 2 backbone**: Pre-trained on image-text pairs
- **RoPE integration**: Rotary position embeddings in every ViT block for spatial awareness
- **Dynamic grid encoding**: Adapts to variable image sizes

**Why Native Resolution?**
- Traditional ViTs use fixed 224×224 or 336×336 grids
- Ovis supports arbitrary resolutions (448²-1792²) without distortion
- Better for documents, wide images, tall images

### 2. Visual Tokenizer (VT)

**File**: `ovis/model/modeling_ovis.py:36-189`

**Purpose**: Convert continuous visual features to probability distributions over visual vocabulary

**Components**:
- **ViT Encoder**: Processes image patches
- **Visual Head**: Linear + LayerNorm projection
- **Smart Resize**: Aspect-ratio preserving resize algorithm

**Output**: Probability distribution `[batch, num_patches, visual_vocab_size]`

### 3. Visual Embedding Table (VET)

**File**: `ovis/model/modeling_ovis.py:25-34`

**Purpose**: Discrete embedding lookup using probability distributions

**Innovation**: Instead of continuous visual features, uses probabilistic weighted sum of discrete embeddings:

```python
# Continuous approach (standard VLMs)
visual_embedding = projection(continuous_features)

# Discrete approach (Ovis)
probabilities = visual_head(continuous_features)  # [B, N, vocab_size]
visual_embedding = probabilities @ embedding_table  # [B, N, hidden_dim]
```

**Why This Matters**:
- **Structural alignment**: Visual tokens become discrete like text tokens
- **Improved cross-modal learning**: LLM sees structurally similar inputs
- **Better training dynamics**: Clearer gradients through discrete space

### 4. Multimodal Merging

**File**: `ovis/model/modeling_ovis.py` - `merge_multimodal()` method

**Purpose**: Combine visual and text embeddings into unified sequence

**Process**:
1. Text tokenization: Convert prompt to token IDs
2. Text embedding: Look up text token embeddings
3. Visual embedding: Replace `<image>` tokens with VET outputs
4. Attention masking: Create proper masks for vision+text
5. Label creation: Mask vision tokens for loss computation (IGNORE_ID)

**Example**:
```python
# Input prompt: "<image>\nDescribe this image."
# Token sequence: [<image>] * 256 + [Describe, this, image, .]

# After merging:
# Embeddings: [visual_emb_1, ..., visual_emb_256, text_emb_1, ..., text_emb_4]
# Shape: [batch, 260, hidden_dim]
```

### 5. Qwen3 Language Model

**File**: LLM integration in `modeling_ovis.py:206-211`

**Why Qwen3?**
- Deep reasoning capabilities
- Strong multimodal understanding
- Efficient architecture
- Better than Qwen2.5 for vision-language tasks

**Sizes Available**:
- Ovis2.5-2B: Qwen3-1.7B
- Ovis2.5-9B: Qwen3-8B

### 6. Thinking Mode (Optional)

**File**: `generate()` method with `enable_thinking=True`

**Purpose**: Enable reflective reasoning before answering

**Process**:
1. **Phase 1**: Generate thinking process with `<think>` tags
   - Budget: Up to `thinking_budget` tokens (e.g., 2048)
   - Model reflects on problem
   - Self-correction and reasoning
2. **Phase 2**: Generate final answer
   - Budget: Remaining tokens (e.g., 1024)
   - Clean answer without reasoning tags

**Example Output**:
```
<think>
Let me analyze this chart step by step:
1. The x-axis shows months from Jan to Dec
2. The y-axis shows sales in thousands
3. There's a peak in July at 450k
4. Lowest point is January at 200k
</think>

The chart shows monthly sales data with a peak of 450,000 in July and a low of 200,000 in January.
```

## Data Flow

### Forward Pass (Inference)

**Input**: Image (PIL) + Query (str)

```
1. Image → smart_resize() → [3, H, W]
2. ViT encode → [N, D_vit]
3. Visual head → [N, vocab_size] probabilities
4. VET lookup → [N, D_hidden] embeddings
5. Merge with text → [L, D_hidden]
6. Qwen3 forward → [L, vocab_size] logits
7. Decode → Text output
```

### Training Forward Pass

**Input**: Image + Query + Ground Truth Answer

```
1-5. Same as inference
6. Compute loss:
   - CrossEntropyLoss on answer tokens
   - Ignore vision tokens (IGNORE_ID)
7. Backward pass:
   - Gradients flow through VET
   - Update visual head, VET, LLM (depending on phase)
```

## Key Design Decisions

### Why Probabilistic VET?

**Problem**: Continuous visual features don't align structurally with discrete text tokens
- LLM sees continuous vectors for vision, discrete embeddings for text
- Creates semantic misalignment
- Harder to learn cross-modal interactions

**Solution**: Make visual tokens discrete (but soft)
- Probability distribution over visual vocabulary
- Weighted sum of discrete embeddings
- Structurally similar to text tokens
- Better cross-modal alignment

### Why Native Resolution?

**Problem**: Fixed tiling distorts images
- 224×224 grid destroys aspect ratios
- Wide images get squashed
- Tall images get stretched

**Solution**: Process at native resolution
- Smart resize preserves aspect ratio
- Dynamic grid encoding
- No information loss from distortion

### Why RoPE in Every ViT Block?

**Problem**: Standard absolute position embeddings don't capture spatial relationships well

**Solution**: Rotary Position Embeddings (RoPE)
- Applied in every ViT block
- Better spatial awareness
- Improved object localization
- Generalizes to unseen sizes

## Model Variants

| Model | ViT | LLM | Params | Use Case |
|-------|-----|-----|--------|----------|
| Ovis2.5-2B | SigLIP 2 | Qwen3-1.7B | 2B | Edge devices, fast inference |
| Ovis2.5-9B | SigLIP 2 | Qwen3-8B | 9B | Production, best performance |

## Performance Characteristics

### Memory
- **Ovis2.5-2B**: ~8GB VRAM (bfloat16)
- **Ovis2.5-9B**: ~20GB VRAM (bfloat16)

### Speed
- **2B**: ~20-30 tokens/sec on RTX 3090
- **9B**: ~10-15 tokens/sec on RTX 3090

### Supported Resolutions
- **Minimum**: 448×448
- **Maximum**: 1792×1792
- **Native**: Any aspect ratio within min/max

## Related Topics

- [01-navit-vision.md](01-navit-vision.md) - Detailed ViT architecture
- [02-visual-tokenizer.md](02-visual-tokenizer.md) - VT implementation
- [03-visual-embedding-table.md](03-visual-embedding-table.md) - VET deep dive
- [04-qwen3-llm.md](04-qwen3-llm.md) - Language model details
- [05-multimodal-merging.md](05-multimodal-merging.md) - Embedding fusion
- [06-thinking-mode.md](06-thinking-mode.md) - Reflective reasoning
- [07-data-flow.md](07-data-flow.md) - Complete execution trace

## Code References

**Main Model**: `ovis/model/modeling_ovis.py`
- Line 25-34: VisualEmbedding (VET)
- Line 36-189: VisualTokenizer (VT)
- Line 196-end: Ovis (main model class)

**ViT**: `ovis/model/vit/modeling_siglip2_navit.py`
**Config**: `ovis/model/configuration_ovis.py`
**Formatter**: `ovis/model/conversation_formatter.py`
