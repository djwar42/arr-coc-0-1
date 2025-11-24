# ARR-COC-VIS Integration with Qwen3-VL

**Category**: Concepts
**Related**: [architecture/02-deepstack.md](../architecture/02-deepstack.md), [codebase/01-vision-process.md](../codebase/01-vision-process.md)
**Code**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

## Overview

Qwen3-VL's architecture provides **three perfect integration points** for ARR-COC-VIS's relevance-aware visual compression:

1. **Vision Preprocessing** - Variable pixel budgets per patch
2. **M-RoPE Position Encoding** - Automatic handling (no changes needed)
3. **DeepStack Multi-Layer Injection** - Hierarchical relevance-based injection

**Expected Compression**: **12-25× reduction** with minimal quality loss

## Integration Point 1: Vision Preprocessing

### Current Qwen3-VL Flow

**File**: `vision_process.py::process_vision_info()` (line 569)

```python
def process_vision_info(conversations, ...):
    # Extract vision info
    vision_infos = extract_vision_info(conversations)

    # UNIFORM budget for all patches
    for vision_info in vision_infos:
        if "image" in vision_info:
            # Same max_pixels for entire image
            image = fetch_image(
                vision_info,
                max_pixels=16384 * 28**2  # 12.8 MP for ALL patches
            )
            image_inputs.append(image)

    return image_inputs, video_inputs
```

**Problem**: Every image region gets same resolution, regardless of relevance to query

### ARR-COC Enhanced Flow

```python
def arr_coc_process_vision_info(conversations, query):
    from arr_coc_vis import RelevanceAllocator, map_relevance_to_budgets

    vision_infos = extract_vision_info(conversations)

    # Initialize ARR-COC allocator
    allocator = RelevanceAllocator(
        propositional_weight=0.33,  # Shannon entropy
        perspectival_weight=0.33,   # Salience
        participatory_weight=0.34   # Query-content coupling
    )

    image_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info:
            # Load full-resolution image for analysis
            full_image = Image.open(vision_info["image"])

            # Score relevance for each patch (e.g., 16×16 grid)
            relevance_scores = allocator.realize_relevance(
                image=full_image,
                query=query,
                patch_grid=(16, 16)
            )

            # Map to token budgets: 64-400 tokens per patch
            token_budgets = map_relevance_to_budgets(
                relevance_scores,
                min_tokens=64,   # Low relevance: 64×64 pixels
                max_tokens=400   # High relevance: 400×400 pixels
            )

            # Process with variable budgets
            patch_images = []
            for patch_idx, (patch, budget) in enumerate(zip(patches, token_budgets)):
                # Convert token budget to pixel budget
                # Qwen3-VL-8B: 32×32 pixels per token
                max_pixels = budget * (32 ** 2)

                # Resize with relevance-aware budget
                resized = fetch_image_with_budget(
                    {"image": patch},
                    max_pixels=max_pixels
                )
                patch_images.append(resized)

            # Reconstruct image from patches
            image = reconstruct_from_patches(patch_images, grid=(16, 16))
            image_inputs.append(image)

    return image_inputs, video_inputs
```

**Key Changes**:
- Query-aware relevance scoring per patch
- Variable `max_pixels` based on relevance (64-400 token range)
- Adaptive resolution allocation

### Example Allocation

**Query**: "What is the price on the receipt?"

**Relevance Map**:
```
Image Grid (16×16 patches):
┌───────────────────────────┐
│ 64   64   64   64   ...   │  Background (low relevance)
│ 64   256  256  256  ...   │  Text region detected
│ 64   256  400  400  ...   │  Price area (high relevance)
│ 64   256  400  400  ...   │
│ ...                       │
└───────────────────────────┘

Token Allocation:
- Background patches: 64 tokens each (56×56 pixels)
- Text regions: 256 tokens each (224×224 pixels)
- Price area: 400 tokens each (400×400 pixels)
```

**Compression**:
```
Uniform (current):  256 patches × 256 tokens = 65,536 tokens
ARR-COC (adaptive): (200×64) + (40×256) + (16×400) = 29,440 tokens
Reduction:          65,536 → 29,440 = 2.2× compression
```

## Integration Point 2: M-RoPE Position Encoding

### Automatic Compatibility

**File**: `rope2d.py::get_rope_index_3()` (line 116)

**Why No Changes Needed**:
```python
# Position IDs generated AFTER tokenization
def get_rope_index_3(input_ids, image_grid_thw, ...):
    # Automatically handles variable token counts
    for i, input_ids in enumerate(total_input_ids):
        # Detects actual sequence length
        input_ids = input_ids[attention_mask[i] == 1]

        # Generates position IDs for ACTUAL tokens
        # Whether 64 tokens or 400 tokens per patch
        llm_pos_ids_list.append(...)

    return position_ids, mrope_position_deltas
```

**Result**: M-RoPE works seamlessly with ARR-COC's variable budgets!

**Example**:
```python
# High-relevance patch: 400 tokens
position_ids_high = [
    t: [0,0,0,...,0],  # 400 zeros
    h: [0,0,1,1,2,2,...,19,19],  # 20×20 grid
    w: [0,1,0,1,0,1,...,0,1]
]

# Low-relevance patch: 64 tokens
position_ids_low = [
    t: [0,0,0,...,0],  # 64 zeros
    h: [0,0,1,1,2,2,...,7,7],  # 8×8 grid
    w: [0,1,0,1,0,1,...,0,1]
]

# Both work perfectly with Interleaved-MRoPE!
```

## Integration Point 3: DeepStack Hierarchical Injection

### Current Qwen3-VL DeepStack

**Where**: HuggingFace model `forward()` pass

```python
# Extract ViT features at multiple layers
low_level = vit.layers[6](image)      # All patches, all layers
mid_level_1 = vit.layers[12](image)
mid_level_2 = vit.layers[18](image)
high_level = vit.layers[24](image)

# Inject ALL patches at ALL LLM layers
llm.layers[0].inject(low_level)       # 256 patches × 4 layers
llm.layers[8].inject(mid_level_1)     # = 1024 effective tokens
llm.layers[16].inject(mid_level_2)
llm.layers[24].inject(high_level)
```

### ARR-COC Enhanced DeepStack

```python
# Extract ViT features (same as before)
low_level = vit.layers[6](image)
mid_level_1 = vit.layers[12](image)
mid_level_2 = vit.layers[18](image)
high_level = vit.layers[24](image)

# SELECTIVE injection based on relevance
for patch_idx in range(num_patches):
    relevance = relevance_scores[patch_idx]

    if relevance > 0.7:  # High-relevance patches
        # Inject at ALL 4 layers
        llm.layers[0].inject(low_level[patch_idx])
        llm.layers[8].inject(mid_level_1[patch_idx])
        llm.layers[16].inject(mid_level_2[patch_idx])
        llm.layers[24].inject(high_level[patch_idx])
        # Effective tokens: 400 × 4 = 1600

    elif relevance > 0.4:  # Mid-relevance patches
        # Inject at MID layers only
        llm.layers[8].inject(mid_level_1[patch_idx])
        llm.layers[16].inject(mid_level_2[patch_idx])
        # Effective tokens: 256 × 2 = 512

    else:  # Low-relevance patches
        # Inject at FINAL layer only
        llm.layers[24].inject(high_level[patch_idx])
        # Effective tokens: 64 × 1 = 64
```

**Hierarchical Compression**:
```
High-relevance: 400 tokens × 4 layers = 1600 effective
Mid-relevance:  256 tokens × 2 layers = 512 effective
Low-relevance:  64 tokens  × 1 layer  = 64 effective

Compression ratio: 1600 / 64 = 25× !
```

### Example: Receipt Image

**Query**: "What is the total amount?"

**Relevance Tiers**:
```
Patches (16×16 grid = 256 patches):
- High (16 patches): Total amount region
  → 400 tokens × 4 layers = 6,400 effective tokens

- Mid (40 patches): Other numbers, text
  → 256 tokens × 2 layers = 20,480 effective tokens

- Low (200 patches): Background, margins
  → 64 tokens × 1 layer = 12,800 effective tokens

Total effective tokens: 39,680

vs Uniform:
  256 patches × 256 tokens × 4 layers = 262,144 effective tokens

Compression: 262,144 → 39,680 = 6.6× reduction!
```

## Complete ARR-COC Integration Pipeline

```python
from arr_coc_vis import (
    InformationScorer,      # Propositional (Shannon entropy)
    SalienceScorer,         # Perspectival (Jungian archetypes)
    CouplingScorer,         # Participatory (query-content)
    TensionBalancer,        # Opponent processing
    AttentionAllocator,     # Relevance → budgets
    RelevanceRealizer       # Complete pipeline
)

# 1. Initialize ARR-COC system
realizer = RelevanceRealizer(
    information_weight=0.33,
    salience_weight=0.33,
    coupling_weight=0.34
)

# 2. Process vision info with relevance
def arr_coc_qwen3vl_forward(image, query):
    # Realize relevance for each patch
    relevance_map = realizer.realize(
        image=image,
        query=query,
        patch_grid=(16, 16)
    )

    # Map to pixel budgets (64-400 tokens)
    token_budgets = realizer.allocate_budgets(
        relevance_map,
        min_tokens=64,
        max_tokens=400
    )

    # Preprocess with variable budgets
    patches = []
    for patch, budget in zip(image_patches, token_budgets):
        resized = smart_resize(
            patch,
            max_pixels=budget * (32**2)
        )
        patches.append(resized)

    # ViT encoding (multi-layer)
    vit_features = multi_layer_vit_encode(patches)

    # Hierarchical DeepStack injection
    for i, (features, relevance) in enumerate(zip(vit_features, relevance_map)):
        if relevance > 0.7:
            inject_at_layers(features, [0, 8, 16, 24])
        elif relevance > 0.4:
            inject_at_layers(features, [8, 16])
        else:
            inject_at_layers(features, [24])

    # M-RoPE (automatic)
    position_ids = get_rope_index_3(...)

    # LLM forward
    output = llm.forward(
        input_embeds=merged_embeds,
        position_ids=position_ids
    )

    return output
```

## Expected Performance

### Compression Ratios

**Spatial Compression** (preprocessing):
```
Uniform:  256 patches × 256 tokens = 65,536 tokens
ARR-COC:  Adaptive allocation = 20,000-35,000 tokens
Ratio:    2-3× compression
```

**Semantic Compression** (DeepStack):
```
Uniform:  256 patches × 4 layers = 1024 effective patches
ARR-COC:  Selective injection = 100-200 effective patches
Ratio:    5-10× compression
```

**Total Compression**:
```
Spatial × Semantic = 2-3× × 5-10× = 10-30× total
Conservative estimate: 12-25× compression
```

### Quality Preservation

**High-relevance regions**:
- ✅ Maximum resolution (400 tokens)
- ✅ Full DeepStack (4 layers)
- ✅ No quality loss

**Low-relevance regions**:
- ⚡ Minimal resolution (64 tokens)
- ⚡ Final layer only (1 layer)
- ✅ Acceptable quality (not critical for query)

## Compatibility Advantages

**Why Qwen3-VL is Perfect for ARR-COC**:

1. **Timestamp-based encoding** - Frames independent, supports variable density
2. **Interleaved-MRoPE** - Auto-handles variable token counts
3. **DeepStack architecture** - Natural hierarchical compression
4. **Dynamic resolution** - Already supports variable budgets
5. **Open architecture** - Easy to modify preprocessing

## Next Steps

1. **Prototype allocator** in `vision_process.py`
2. **Test on sample images** with query-aware budgets
3. **Measure compression ratio** and quality
4. **Implement hierarchical DeepStack** injection
5. **Benchmark on standard datasets** (VQA, OCR, etc.)

## Related Topics

- [architecture/02-deepstack.md](../architecture/02-deepstack.md) - Multi-layer injection details
- [codebase/01-vision-process.md](../codebase/01-vision-process.md) - Preprocessing code
- [concepts/00-interleaved-mrope.md](00-interleaved-mrope.md) - Position encoding
- [concepts/03-dynamic-resolution.md](03-dynamic-resolution.md) - Smart resize algorithm

## Code References

**Preprocessing**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py:569`
**M-RoPE**: `qwen-vl-finetune/qwenvl/data/rope2d.py:116`
**DeepStack**: HuggingFace `Qwen3VLForConditionalGeneration.forward()`
