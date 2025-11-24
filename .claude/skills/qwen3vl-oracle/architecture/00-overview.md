# Qwen3-VL Architecture Overview

**Category**: Architecture
**Related**: All architecture files
**Code**: `RESEARCH/Qwen3VL/Qwen3-VL/`

## Overview

Qwen3-VL is a state-of-the-art Multimodal Large Language Model (MLLM) designed with **three key architectural innovations** that advance vision-language understanding:

1. **Interleaved-MRoPE**: Full-frequency positional encoding across time, width, and height
2. **DeepStack**: Multi-layer ViT feature injection for fine-grained visual understanding
3. **Text-Timestamp Alignment**: Precise temporal grounding using learnable timestamp tokens

## System Architecture

```
Image/Video Input
    ↓
┌─────────────────────────────────────┐
│ Vision Transformer (ViT)            │
│ - Pre-trained encoder               │
│ - Multi-layer feature extraction    │
│ - Extract at layers [6,12,18,24]   │
│ Output: Multi-level features        │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Dynamic Resolution Preprocessing    │
│ - smart_resize() algorithm          │
│ - Pixel budget: 64-16384 tokens     │
│ - Factor rounding (28 or 32)        │
│ - Aspect ratio preservation         │
│ Output: Resized images/frames       │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Interleaved-MRoPE                   │
│ - 3D position encoding (t,h,w)      │
│ - Full-frequency allocation         │
│ - Robust long-horizon reasoning     │
│ Output: Position IDs [3, B, L]      │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ DeepStack Multi-Layer Injection     │
│ - ViT layers → LLM layers           │
│ - [6,12,18,24] → [0,8,16,24]       │
│ - Low-level to high-level features  │
│ Output: Enriched LLM states         │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Qwen3 Language Model                │
│ - Decoder-only transformer          │
│ - Thinking/Non-thinking modes       │
│ - Autoregressive generation         │
│ Output: Text tokens                 │
└─────────────────────────────────────┘
    ↓
Text Output
```

## Key Innovations

### 1. Interleaved-MRoPE

**File**: `qwen-vl-finetune/qwenvl/data/rope2d.py`

**Purpose**: Allocate full-frequency positional encoding across three dimensions

**Key Features**:
- **Temporal dimension (t)**: Frame/temporal position
- **Height dimension (h)**: Vertical spatial position
- **Width dimension (w)**: Horizontal spatial position
- **Full-frequency**: Each dimension gets complete frequency range
- **Robust scaling**: Handles long videos and high-resolution images

**Why It Matters**:
```
Traditional RoPE: 1D sequence position
M-RoPE (Qwen2): 3D but shared frequencies
Interleaved-MRoPE (Qwen3): 3D with full frequencies each
→ Better long-horizon video reasoning
→ Improved spatial understanding
→ Scalable to 256K+ context
```

### 2. DeepStack

**Where**: HuggingFace model's `forward()` pass (not in this repo)

**Purpose**: Fuse multi-level ViT features for fine-grained understanding

**Mechanism**:
```python
# Extract features at multiple ViT layers
low_level = vit.layers[6](image)      # Edges, textures
mid_level_1 = vit.layers[12](image)   # Simple shapes
mid_level_2 = vit.layers[18](image)   # Object parts
high_level = vit.layers[24](image)    # Semantic concepts

# Inject into corresponding LLM layers
llm.layers[0].inject(low_level)      # Early LLM: Fine details
llm.layers[8].inject(mid_level_1)    # Mid LLM: Compositional
llm.layers[16].inject(mid_level_2)   # Late-mid: High-level
llm.layers[24].inject(high_level)    # Final: Semantic
```

**Benefits**:
- **Fine-grained details**: Low-level features preserved
- **Hierarchical understanding**: Multi-scale visual reasoning
- **Sharper alignment**: Better image-text correspondence

**ARR-COC Opportunity**:
```python
# Current: Uniform injection
for patch in all_patches:
    inject_at_all_layers(patch)

# ARR-COC: Relevance-aware injection
for patch in all_patches:
    if relevance[patch] > high_threshold:
        inject_at_all_layers(patch)  # 4 layers
    elif relevance[patch] > low_threshold:
        inject_at_mid_layers(patch)   # 2 layers
    else:
        inject_at_final_layer(patch)  # 1 layer

# Compression: 25× (64 tokens × 1 layer vs 400 × 4 layers)
```

### 3. Text-Timestamp Alignment

**File**: `rope2d.py:get_rope_index_3()` (lines 112-235)

**Revolutionary Approach**: Decouple temporal info from position IDs

**Traditional Approach (Qwen2/2.5)**:
```
Frame 1 patches: t=[0,0,0,0], h=[0,0,1,1], w=[0,1,0,1]
Frame 2 patches: t=[1,1,1,1], h=[0,0,1,1], w=[0,1,0,1]
Frame 3 patches: t=[2,2,2,2], h=[0,0,1,1], w=[0,1,0,1]

Temporal info encoded in position IDs
```

**Qwen3 Approach**:
```
<t1> <vision_start> <image_token> [frame1 patches] <vision_end>
<t2> <vision_start> <image_token> [frame2 patches] <vision_end>
<t3> <vision_start> <image_token> [frame3 patches] <vision_end>

Frame 1 patches: t=[0,0,0,0], h=[0,0,1,1], w=[0,1,0,1]
Frame 2 patches: t=[0,0,0,0], h=[0,0,1,1], w=[0,1,0,1]  # Still 0!
Frame 3 patches: t=[0,0,0,0], h=[0,0,1,1], w=[0,1,0,1]  # Still 0!

Temporal info in <t1>, <t2>, <t3> tokens (learnable embeddings)
```

**Why Revolutionary**:
- ✅ **Independent frames**: Each frame treated separately
- ✅ **No position overflow**: Temporal position doesn't grow unbounded
- ✅ **Variable density**: Can have different frame counts without breaking encoding
- ✅ **Precise grounding**: "At timestamp 3:27, the car turns left" → sees `<t_3m27s>` token
- ✅ **ARR-COC compatible**: Perfect for variable compression per frame

**Code Reference**:
```python
# Line 126-129: Split video into individual frames
if video_grid_thw is not None:
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1  # Set temporal dimension to 1

# Line 197: t_index is ALWAYS 0 (comment on line 181!)
# "t_index is always 0 because llm_grid_t is always 1
#  (we use timestamps to encode the temporal information for videos)"
t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
# Result: [0, 0, 0, 0, ...] for all patches in frame
```

## Data Flow

### Image Understanding

```
1. Image (PIL) → smart_resize() → [3, H, W]
2. ViT multi-layer encode → Features at layers [6,12,18,24]
3. DeepStack inject → LLM layers [0,8,16,24]
4. Interleaved-MRoPE → Position IDs [3, B, L]
5. LLM forward → Logits [B, L, vocab]
6. Decode → Text output
```

### Video Understanding

```
1. Video → fetch_video() → Sampled frames [T, 3, H, W]
2. Per-frame smart_resize() → Variable resolutions
3. Split into frames → Each frame independent
4. Timestamp tokens → <t1>, <t2>, ..., <tN>
5. Per-frame ViT encode → Multi-level features
6. DeepStack inject → Hierarchical LLM states
7. M-RoPE with t=0 → Spatial encoding only
8. LLM processes sequence → Text output
```

## Model Variants

| Model | ViT | LLM | Context | Use Case |
|-------|-----|-----|---------|----------|
| Qwen3-VL-2B | ViT | Qwen3-2B | 256K | Edge, fast inference |
| Qwen3-VL-4B | ViT | Qwen3-4B | 256K | Balanced performance |
| Qwen3-VL-8B | ViT | Qwen3-8B | 256K | Production quality |
| Qwen3-VL-32B | ViT | Qwen3-32B | 256K | Best quality |
| Qwen3-VL-235B-A22B | ViT | Qwen3-235B MoE | 256K | Flagship, 22B active |

## Performance Characteristics

### Resolution Range
- **Image**: 4 tokens (56×56) to 16384 tokens (3584×3584)
- **Video frame**: 128 tokens to 768 tokens per frame
- **Total video**: Up to 90M pixels shared across frames

### Context Window
- **Standard**: 256K tokens (extendable with YaRN)
- **Extended**: 1M tokens with rope_scaling config
- **Video**: Allocates 90% of context for visual tokens

### Compression Factor
- **ViT patch**: 32×32 pixels → 1 token (Qwen3-VL-8B+)
- **ViT patch**: 28×28 pixels → 1 token (Qwen3-VL-2B/4B)
- **Spatial merge**: 2×2 patches after ViT

## Related Topics

- [01-positional-encoding.md](01-positional-encoding.md) - Interleaved-MRoPE details
- [02-deepstack.md](02-deepstack.md) - Multi-layer injection
- [03-timestamp-alignment.md](03-timestamp-alignment.md) - Temporal grounding
- [04-vision-encoder.md](04-vision-encoder.md) - ViT architecture
- [05-llm-integration.md](05-llm-integration.md) - Language model
- [06-thinking-mode.md](06-thinking-mode.md) - Reasoning capabilities
- [07-data-flow.md](07-data-flow.md) - Execution trace

## Code References

**Main Preprocessing**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
**M-RoPE**: `qwen-vl-finetune/qwenvl/data/rope2d.py`
**Training**: `qwen-vl-finetune/qwenvl/train/train_qwen.py`
**Model**: HuggingFace `transformers` library - `Qwen3VLForConditionalGeneration`
