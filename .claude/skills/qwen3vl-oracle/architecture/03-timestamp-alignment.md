# Text-Timestamp Alignment: Precise Video Temporal Modeling

**Beyond T-RoPE: Explicit timestamp tokens for fine-grained event localization**

## What is Text-Timestamp Alignment?

**Text-Timestamp Alignment** is Qwen3-VL's revolutionary approach to video temporal encoding that uses **explicit timestamp tokens** (`<t1>`, `<t2>`, etc.) instead of relying solely on positional embeddings.

From the README:
> "Moves beyond T‑RoPE to precise, timestamp‑grounded event localization for stronger video temporal modeling."

## The Problem with Traditional Approaches

### Qwen2-VL: Basic Temporal Encoding
```
<vision_start> [frame1_tokens] [frame2_tokens] [frame3_tokens] <vision_end>

Temporal info: Encoded via position IDs only
Position IDs: [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]

❌ Problem: Temporal position is implicit, hard to localize specific moments
```

### Qwen2.5-VL: Temporal Scaling (T-RoPE)
```
Same token sequence, but position IDs scaled by video duration:
Position IDs: [0,0,0,0, 50,50,50,50, 100,100,100,100, ...]
(scaled by second_per_grid_t parameter)

✓ Better: Handles long videos without position overflow
❌ Still limited: Can't reference precise timestamps in generated text
```

## Qwen3-VL Solution: Explicit Timestamp Tokens

### Token Sequence Format
```
<t1> <vision_start> [frame1_tokens] <vision_end>
<t2> <vision_start> [frame2_tokens] <vision_end>
<t3> <vision_start> [frame3_tokens] <vision_end>

Timestamp tokens: <t1>, <t2>, <t3>, ... (learnable embeddings)
Vision blocks: Separated by timestamps, each frame gets own vision block
```

**Key innovation**: Each video frame is preceded by an **explicit timestamp token** that marks its temporal position.

### Position ID Structure

From `rope2d.py` lines 99-101:
```python
# Split video into separate vision blocks
video_grid_thw_tensor = repeat_interleave(
    video_grid_thw_tensor, video_grid_thw_tensor[:, 0], dim=0
)
video_grid_thw_tensor[:, 0] = 1  # Set temporal dimension to 1

# Result: [[3,4,4]] → [[1,4,4], [1,4,4], [1,4,4]]
# Each frame becomes independent vision block
```

**Result**: `llm_grid_t = 1` for every vision block
- Position IDs within each frame: `t_index = [0,0,0,0, ...]` (constant)
- Temporal information: Encoded via timestamp tokens, NOT position IDs

## How It Works

### 1. Video Frame Sampling
```python
# Example: 30-second video at 1 FPS
num_frames = 30
timestamps = [0s, 1s, 2s, ..., 29s]
```

### 2. Timestamp Token Assignment
```python
# Map each frame to a timestamp token
frame_0 → <t0>   (0 seconds)
frame_1 → <t1>   (1 second)
...
frame_29 → <t29> (29 seconds)
```

### 3. Sequence Construction
```python
sequence = [
    "<t0>", "<vision_start>", *frame_0_patches, "<vision_end>",
    "<t1>", "<vision_start>", *frame_1_patches, "<vision_end>",
    ...
    "<t29>", "<vision_start>", *frame_29_patches, "<vision_end>"
]
```

### 4. Position ID Generation
```python
# From rope2d.py get_rope_index_3()

# For each vision block (single frame):
llm_grid_t = 1  # Temporal dimension always 1
llm_grid_h = 4  # Spatial height (example)
llm_grid_w = 4  # Spatial width (example)

# Position IDs for 16 patches (4×4):
t_index: [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]  # Constant!
h_index: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]  # Row position
w_index: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]  # Column position
```

**Critical insight**: Temporal position (t_index) remains **constant within each frame**. Time is tracked by **timestamp tokens**, not position IDs.

## Benefits

### 1. **Precise Event Localization**

Users can ask:
```
Q: "At what timestamp does the car turn left?"
A: "At <t127> (2:07), the car begins turning left."
```

The model can **generate timestamp references** in its output because timestamps are explicit tokens in the vocabulary.

### 2. **Flexible Temporal Granularity**

```python
# Adaptive frame sampling based on video content
static_scene = [<t0>, <t10>, <t20>, ...]  # Sparse sampling (every 10s)
action_scene = [<t0>, <t1>, <t2>, ...]     # Dense sampling (every 1s)
```

The model learns to handle **variable temporal spacing** because timestamps are explicit.

### 3. **Decoupled Temporal and Spatial Encoding**

Traditional M-RoPE couples temporal and spatial position:
```
Position IDs: [t, h, w] all increment together
```

Qwen3-VL decouples them:
```
Temporal: Encoded via timestamp tokens (explicit)
Spatial: Encoded via position IDs (h, w)
```

**Result**: Cleaner separation of concerns, easier to scale to longer videos.

### 4. **Better Long-Video Understanding**

From README:
> "Native 256K context, expandable to 1M; handles books and hours-long video with full recall and second-level indexing."

Timestamp tokens enable **second-level indexing**:
- User: "What happened at 1:23:45?"
- Model: Directly attends to `<t5025>` token (1h 23m 45s = 5025 seconds)

## Implementation Details

### Timestamp Token Vocabulary

Timestamp tokens are **added to the model vocabulary**:

```python
# Special tokens (from rope2d.py)
vision_start_token_id = 151652
vision_end_token_id = 151653
image_token_id = 151655
video_token_id = 151656

# Timestamp tokens (example IDs, actual may vary)
timestamp_token_ids = [151700, 151701, 151702, ...]  # <t0>, <t1>, <t2>, ...
```

### Timestamp Token Embeddings

Each timestamp token has a **learnable embedding**:

```python
class QwenVLModel:
    def __init__(self):
        self.timestamp_embeddings = nn.Embedding(
            num_timestamps=1000,  # Support up to 1000 unique timestamps
            embedding_dim=model_dim
        )
```

**Training**: Model learns to associate timestamp embeddings with temporal positions during video pre-training.

### Inference Example

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Video processing with timestamps
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "path/to/video.mp4"},
        {"type": "text", "text": "Describe what happens at each timestamp."}
    ]
}]

# Model automatically inserts timestamp tokens during preprocessing
# Output: "At <t0>, the scene shows... At <t5>, the person walks... At <t12>, ..."
```

## Comparison: T-RoPE vs Text-Timestamp Alignment

| Feature | T-RoPE (Qwen2.5-VL) | Timestamp Alignment (Qwen3-VL) |
|---------|---------------------|--------------------------------|
| **Temporal encoding** | Position IDs scaled by duration | Explicit timestamp tokens |
| **Event reference** | Implicit (frame index) | Explicit (`<t127>`) |
| **Temporal granularity** | Fixed by position scaling | Adaptive (any spacing) |
| **Long video support** | Limited by position ID range | Scalable (arbitrary timestamps) |
| **Model output** | Can't generate timestamps | Can generate `<t>` tokens |
| **Context decoupling** | Temporal + spatial coupled | Temporal (tokens) + spatial (IDs) |

## ARR-COC Integration

Text-Timestamp Alignment is **perfectly compatible** with ARR-COC compression:

### 1. **Timestamp Tokens Unaffected**
Timestamp tokens remain intact regardless of compression:
```python
# High relevance (many patches)
<t5> <vision_start> [400 patches] <vision_end>

# Low relevance (few patches)
<t12> <vision_start> [64 patches] <vision_end>

# Timestamp tokens <t5>, <t12> unchanged
```

### 2. **Adaptive Temporal Sampling**
ARR-COC can implement **query-aware temporal sampling**:

```python
query = "When does the car appear?"

# Sample densely around car appearance
timestamps = [<t0>, <t5>, <t10>, <t50>, <t51>, <t52>, <t53>, <t54>, <t55>, <t100>]
#                                      ^^^^^ Dense around t=50-55 (car appears)
```

### 3. **Hierarchical Temporal Compression**
Combine timestamp sparsity with patch-level compression:

```python
# Important moments: Dense timestamps + many patches
<t50> <vision_start> [400 patches] <vision_end>  # Car appears
<t51> <vision_start> [400 patches] <vision_end>  # Car moves

# Unimportant moments: Sparse timestamps + few patches
<t10> <vision_start> [64 patches] <vision_end>   # Empty road
<t20> <vision_start> [64 patches] <vision_end>   # Empty road
<t30> <vision_start> [64 patches] <vision_end>   # Empty road
```

## Related Documentation

- [01-positional-encoding.md](01-positional-encoding.md) - How M-RoPE works with timestamps
- [../codebase/02-rope2d.md](../codebase/02-rope2d.md) - `get_rope_index_3()` implementation
- [../concepts/02-temporal-encoding.md](../concepts/02-temporal-encoding.md) - Deep dive on temporal modeling
- [../examples/02-video-understanding.md](../examples/02-video-understanding.md) - Practical video examples

## Quick Reference

**Token Format**: `<t{frame_index}>` (e.g., `<t0>`, `<t1>`, `<t2>`)

**Sequence Structure**:
```
<t0> <vision_start> [frame_0_patches] <vision_end>
<t1> <vision_start> [frame_1_patches] <vision_end>
...
```

**Position IDs**: `t_index = 0` within each frame (temporal info from tokens)

**Key Benefit**: Precise, explicit temporal referencing in both input and output

---

**Last Updated**: 2025-10-28
**Status**: Core architectural innovation
**Importance**: ⭐⭐⭐⭐⭐ (Critical - unique to Qwen3-VL)
