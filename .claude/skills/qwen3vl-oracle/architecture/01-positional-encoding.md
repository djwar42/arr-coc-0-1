# Interleaved-MRoPE: Full-Frequency Positional Encoding

**Qwen3-VL's signature innovation for multimodal position encoding**

## What is Interleaved-MRoPE?

**Interleaved-MRoPE** (Multimodal Rotary Position Embedding) is Qwen3-VL's revolutionary approach to encoding positional information across three dimensions: **time, height, and width**.

Unlike traditional approaches that allocate separate frequency ranges to each dimension, Interleaved-MRoPE **distributes all three dimensions across the full frequency spectrum**, enabling richer positional representations.

## Key Innovation

```
Traditional M-RoPE (Qwen2-VL):
┌─────────────┬─────────────┬─────────────┐
│   Time      │   Height    │   Width     │
│  (freqs 0-7)│  (freqs 8-15)│ (freqs 16-23)│
└─────────────┴─────────────┴─────────────┘
Each dimension uses a separate frequency band

Interleaved-MRoPE (Qwen3-VL):
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│T│H│W│T│H│W│T│H│W│T│H│W│T│H│W│T│H│W│T│H│W│T│H│W│
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
All dimensions interleaved across full frequency spectrum
```

## Architecture Details

### Dimension Allocation

From model configuration:
```python
mrope_section = [24, 20, 20]  # [temporal, height, width]
mrope_interleaved = True       # Enable interleaving
```

**Embedding split**:
- **Total embedding dimensions**: 64 (for head_dim=64)
- **Temporal**: 24 dimensions
- **Height**: 20 dimensions
- **Width**: 20 dimensions

### Interleaving Pattern

Instead of sequential allocation (dims 0-23 temporal, 24-43 height, 44-63 width), Interleaved-MRoPE distributes:

```
Frequency 0:  [t0, h0, w0]
Frequency 1:  [t1, h1, w1]
Frequency 2:  [t2, h2, w2]
...
Frequency 21: [t21, h20, w20]
Frequency 22: [t22, -, -]
Frequency 23: [t23, -, -]
```

Each frequency band contains information from **all three spatial-temporal dimensions**.

## Why Interleaving Matters

### 1. **Full-Frequency Access**
Every dimension benefits from the entire frequency spectrum:
- **Low frequencies** → long-range dependencies (full video duration, image borders)
- **High frequencies** → fine-grained details (frame-to-frame, pixel-level)

### 2. **Better Long-Horizon Video Reasoning**
From the README:
> "Full‑frequency allocation over time, width, and height via robust positional embeddings, enhancing long‑horizon video reasoning."

Traditional M-RoPE limits temporal dimension to low frequencies only. Interleaved-MRoPE gives temporal information access to high frequencies, enabling precise event localization in long videos.

### 3. **Balanced Spatial-Temporal Representation**
No dimension is "starved" of frequency diversity. All three dimensions get equal access to the full representational capacity.

## Implementation

### Code Reference

**File**: `qwen-vl-finetune/qwenvl/data/rope2d.py`

**Function**: `get_rope_index_3()` (Qwen3-VL version)

**Key lines**:
```python
# Lines 99-101: Video grid splitting for timestamp-based encoding
video_grid_thw_tensor = repeat_interleave(
    video_grid_thw_tensor, video_grid_thw_tensor[:, 0], dim=0
).contiguous()
video_grid_thw_tensor[:, 0] = 1  # Each frame gets own vision block

# Lines 169-185: Position ID generation
# Returns (3, batch_size, seq_len) for [temporal, height, width]
t_index = torch.repeat_interleave(
    llm_grid_t_index, llm_grid_h * llm_grid_w, dim=0
)
h_index = torch.repeat_interleave(
    llm_grid_h_index, llm_grid_w, dim=0
).tile(llm_grid_t)
w_index = llm_grid_w_index.tile(llm_grid_t * llm_grid_h)
```

**Returns**: 3D position indices `(3, batch_size, sequence_length)`:
- Index 0: Temporal positions
- Index 1: Height positions
- Index 2: Width positions

These indices are then processed by the model's RoPE layer with `mrope_interleaved=True` to apply interleaved encoding.

## Position ID Structure

For a **2-frame video** with **4×4 spatial patches**:

```python
# Video grid: [[2, 4, 4]] → 2 frames, 4×4 patches each

# After splitting (lines 99-101): [[1,4,4], [1,4,4]]

# Frame 1 position IDs (16 tokens):
t: [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]  # All same temporal
h: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]  # Row index
w: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]  # Column index

# Frame 2 position IDs (16 tokens):
t: [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]  # All same temporal (timestamp handles time)
h: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]  # Row index
w: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]  # Column index
```

**Key insight**: Temporal position stays constant within each frame (llm_grid_t=1). Time is encoded via **timestamp tokens** (see [03-timestamp-alignment.md](03-timestamp-alignment.md)).

## Evolution Across Versions

### Qwen2-VL: Basic 3D M-RoPE
- Sequential allocation: `[temporal, height, width]`
- Simple temporal scaling
- Good for static images

### Qwen2.5-VL: Temporal Scaling
- Added `second_per_grid_t` parameter
- Stretched temporal IDs to match video duration
- Better long-video handling

### Qwen3-VL: Interleaved + Timestamp
- **Full-frequency interleaving** (signature innovation)
- **Timestamp-based temporal encoding** (decoupled from position IDs)
- **Superior long-horizon reasoning**

## Benefits for ARR-COC Integration

Interleaved-MRoPE is **perfectly suited** for variable-token compression:

1. **Flexible Position Encoding**: Handles variable patch token counts naturally
2. **Rich Representations**: Full-frequency access ensures compressed patches retain positional nuance
3. **Timestamp Compatibility**: Works seamlessly with Qwen3-VL's timestamp tokens
4. **No Retraining Needed**: Position IDs generated after tokenization, independent of compression

## Related Documentation

- [03-timestamp-alignment.md](03-timestamp-alignment.md) - How timestamps work with M-RoPE
- [../codebase/02-rope2d.md](../codebase/02-rope2d.md) - Complete rope2d.py code walkthrough
- [../concepts/00-interleaved-mrope.md](../concepts/00-interleaved-mrope.md) - Conceptual deep dive

## Quick Reference

**Model Config**:
```python
config.rope_scaling = {
    "type": "mrope",
    "mrope_section": [24, 20, 20],  # [t, h, w] dimension allocation
    "mrope_interleaved": True        # ENABLE INTERLEAVING
}
```

**Source Files**:
- `qwen-vl-finetune/qwenvl/data/rope2d.py` - Position ID generation
- HuggingFace model config - `mrope_section` and `mrope_interleaved` settings

**Key Papers**: Qwen3-VL Technical Report (coming soon)

---

**Last Updated**: 2025-10-28
**Status**: Core architectural innovation
**Importance**: ⭐⭐⭐⭐⭐ (Critical)
