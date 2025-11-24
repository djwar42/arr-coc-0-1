# vision_process.py - Vision Preprocessing Pipeline

**Category**: Codebase
**File**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` (608 lines)
**Related**: [concepts/03-dynamic-resolution.md](../concepts/03-dynamic-resolution.md), [usage/02-dynamic-resolution.md](../usage/02-dynamic-resolution.md)

## Overview

`vision_process.py` contains **ALL preprocessing logic** for Qwen3-VL's dynamic resolution architecture. This is the **primary ARR-COC integration point**.

**Key Functions**:
- `smart_resize()` - Adaptive resolution with pixel budgets
- `fetch_image()` - Load and resize single images
- `fetch_video()` - Sample and resize video frames
- `process_vision_info()` - **Main entry point** (ARR-COC injection here!)

## Constants

**Lines**: 98-111

```python
# Aspect ratio limits
MAX_RATIO = 200                # Max aspect ratio (prevent 1×200 images)

# Patch merging
SPATIAL_MERGE_SIZE = 2         # 2×2 patch merging after ViT

# Image token budgets
IMAGE_MIN_TOKEN_NUM = 4        # Min 4 patches = 56×56 pixels
IMAGE_MAX_TOKEN_NUM = 16384    # Max 16384 patches = 3584×3584 pixels

# Video token budgets (per frame)
VIDEO_MIN_TOKEN_NUM = 128      # Min 128 tokens per frame
VIDEO_MAX_TOKEN_NUM = 768      # Max 768 tokens per frame

# Video sampling
FPS = 2.0                      # Default frame sampling rate (fps)
FRAME_FACTOR = 2               # Temporal stride (2-frame patches)
FPS_MIN_FRAMES = 4             # Minimum 4 frames per video
FPS_MAX_FRAMES = 768           # Maximum 768 frames (6.4 min at 2fps)
MAX_NUM_WORKERS_FETCH_VIDEO = 8  # Parallel frame loading

# Context window
MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
```

**ARR-COC Relevance**:
- `IMAGE_MAX_TOKEN_NUM`: Currently uniform (16384)
  → ARR-COC: Variable per patch (64-400)
- `VIDEO_MAX_TOKEN_NUM`: Currently uniform (768)
  → ARR-COC: Variable per frame based on relevance

## Helper Functions

### round_by_factor()

**Lines**: 127-131

```python
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' divisible by 'factor'."""
    return round(number / factor) * factor
```

**Example**:
```python
round_by_factor(220, 28) → 224  # Nearest multiple of 28
round_by_factor(230, 28) → 224
```

### ceil_by_factor()

**Lines**: 134-136

```python
def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' divisible by 'factor'."""
    return math.ceil(number / factor) * factor
```

**Example**:
```python
ceil_by_factor(220, 28) → 224
ceil_by_factor(200, 28) → 224
```

### floor_by_factor()

**Lines**: 139-141

```python
def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' divisible by 'factor'."""
    return math.floor(number / factor) * factor
```

**Example**:
```python
floor_by_factor(230, 28) → 224
floor_by_factor(250, 28) → 224
```

## Core Functions

### smart_resize()

**Lines**: 144-169

**Purpose**: Resize image to fit pixel budget while preserving aspect ratio

**Signature**:
```python
def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
) -> Tuple[int, int]:
```

**Algorithm**:
```python
# Line 152-153: Set default budgets
max_pixels = max_pixels or (IMAGE_MAX_TOKEN_NUM * factor ** 2)
min_pixels = min_pixels or (IMAGE_MIN_TOKEN_NUM * factor ** 2)
# max_pixels = 16384 * 28² = 12,845,056 (3584×3584)
# min_pixels = 4 * 28² = 3,136 (56×56)

# Line 155-158: Check aspect ratio
if max(height, width) / min(height, width) > MAX_RATIO:
    raise ValueError(f"Aspect ratio must be < {MAX_RATIO}")

# Line 159-160: Round to factor multiples
h_bar = max(factor, round_by_factor(height, factor))
w_bar = max(factor, round_by_factor(width, factor))

# Line 161-164: Scale DOWN if too large
if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = floor_by_factor(height / beta, factor)
    w_bar = floor_by_factor(width / beta, factor)

# Line 165-168: Scale UP if too small
elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = ceil_by_factor(height * beta, factor)
    w_bar = ceil_by_factor(width * beta, factor)

return h_bar, w_bar
```

**Examples**:

**Example 1: Small image (no scaling)**
```python
Input: 1920×1080, factor=28
Step 1: Round to multiples
  h_bar = round_by_factor(1920, 28) = 1932
  w_bar = round_by_factor(1080, 28) = 1092
Step 2: Check budget
  1932 × 1092 = 2,109,744 < 12,845,056 ✓
Result: 1932×1092 (no scaling needed)
```

**Example 2: Large image (scale down)**
```python
Input: 4000×4000, factor=28
Step 1: Round to multiples
  h_bar = round_by_factor(4000, 28) = 4004
  w_bar = round_by_factor(4000, 28) = 4004
Step 2: Check budget
  4004 × 4004 = 16,032,016 > 12,845,056 ✗
Step 3: Scale down
  beta = sqrt(16,032,016 / 12,845,056) = 1.117
  h_bar = floor_by_factor(4000/1.117, 28) = 3584
  w_bar = floor_by_factor(4000/1.117, 28) = 3584
Result: 3584×3584 (exactly at max budget)
```

**Example 3: Tiny image (scale up)**
```python
Input: 40×40, factor=28
Step 1: Round to multiples
  h_bar = round_by_factor(40, 28) = 28
  w_bar = round_by_factor(40, 28) = 28
Step 2: Check budget
  28 × 28 = 784 < 3,136 ✗ (below minimum!)
Step 3: Scale up
  beta = sqrt(3,136 / 1,600) = 1.4
  h_bar = ceil_by_factor(40*1.4, 28) = 56
  w_bar = ceil_by_factor(40*1.4, 28) = 56
Result: 56×56 (at minimum budget)
```

**🎯 ARR-COC Injection Point**:
```python
# Current: Uniform max_pixels for entire image
max_pixels = 16384 * 28**2  # Same for all patches

# ARR-COC: Variable max_pixels per patch
relevance_scores = arr_coc_allocator(image, query)
for patch, relevance in zip(patches, relevance_scores):
    token_budget = map_relevance_to_tokens(relevance)  # 64-400
    max_pixels_patch = token_budget * 28**2  # Variable!
    resized_patch = smart_resize(patch, max_pixels=max_pixels_patch)
```

### fetch_image()

**Lines**: 167-214

**Purpose**: Load image from various sources and apply smart_resize

**Signature**:
```python
def fetch_image(
    ele: dict,
    image_patch_size: int
) -> Image.Image:
```

**Supported Input Formats**:
```python
# 1. PIL Image object
ele = {"image": PIL.Image.open("path.jpg")}

# 2. Local file path
ele = {"image": "file:///path/to/image.jpg"}
ele = {"image": "/path/to/image.jpg"}

# 3. HTTP(S) URL
ele = {"image": "https://example.com/image.jpg"}
ele = {"image": "http://example.com/image.jpg"}

# 4. Base64 encoded
ele = {"image": "data:image/jpeg;base64,/9j/4AAQ..."}

# 5. Alternative key
ele = {"image_url": "https://example.com/image.jpg"}
```

**Algorithm**:
```python
# Line 168-172: Get image source
if "image" in ele:
    image = ele["image"]
else:
    image = ele["image_url"]

# Line 174: Calculate patch factor
patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
# = 14 * 2 = 28 (for Qwen2.5-VL)
# = 16 * 2 = 32 (for Qwen3-VL)

# Line 175-191: Load from source
if isinstance(image, Image.Image):
    image_obj = image
elif image.startswith("http://") or image.startswith("https://"):
    response = requests.get(image, stream=True)
    image_obj = Image.open(BytesIO(response.content))
elif image.startswith("file://"):
    image_obj = Image.open(image[7:])
elif image.startswith("data:image"):
    _, base64_data = image.split("base64,", 1)
    data = base64.b64decode(base64_data)
    image_obj = Image.open(BytesIO(data))
else:
    image_obj = Image.open(image)

# Line 194: Convert to RGB
image = to_rgb(image_obj)

# Line 196-210: Apply smart_resize
height, width = image.size
min_pixels = ele.get("min_pixels", None)
max_pixels = ele.get("max_pixels", None)

resized_height, resized_width = smart_resize(
    height,
    width,
    factor=patch_factor,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

image = image.resize((resized_width, resized_height))
return image
```

**🎯 ARR-COC Enhancement**:
```python
def fetch_image_with_relevance(ele: dict, query: str, allocator):
    # Load full-resolution image
    full_image = load_full_image(ele)

    # Score relevance for patches
    relevance_map = allocator.realize_relevance(
        full_image,
        query,
        patch_grid=(16, 16)
    )

    # Process each patch with variable budget
    patches = split_into_patches(full_image, grid=(16, 16))
    resized_patches = []

    for patch, relevance in zip(patches, relevance_map):
        token_budget = map_relevance_to_tokens(relevance)
        max_pixels = token_budget * (patch_factor ** 2)

        resized_patch = smart_resize(
            patch.height,
            patch.width,
            factor=patch_factor,
            max_pixels=max_pixels
        )
        resized_patches.append(resized_patch)

    # Reconstruct image
    return reconstruct_from_patches(resized_patches)
```

### fetch_video()

**Lines**: 477-554

**Purpose**: Sample and resize video frames with shared pixel budget

**Key Differences from Images**:
1. **Frame sampling**: Extract frames at specified FPS
2. **Shared budget**: Total pixels distributed across ALL frames
3. **Per-frame constraint**: Each frame still limited by VIDEO_FRAME_MAX_PIXELS

**Budget Calculation** (lines 522-529):
```python
nframes, _, height, width = video.shape

# Minimum per-frame budget
min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
# = 128 * 28² = 100,352 pixels per frame

# Total budget shared across frames
total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor² * 0.9)
# = 128000 * 784 * 0.9 = 90,316,800 pixels TOTAL

# Per-frame budget (with cap)
max_pixels = max(
    min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
    int(min_pixels * 1.05)
)

# Example: 100 frames
# Theoretical: 90,316,800 / 100 * 2 = 1,806,336 pixels/frame
# Capped:      min(602,112, 1,806,336) = 602,112 pixels/frame
#              ≈ 775×775 per frame
```

**🎯 ARR-COC for Videos**:
```python
def fetch_video_with_relevance(ele: dict, query: str, allocator):
    # Load full video
    video_frames = load_video(ele)

    # Score relevance for each FRAME
    frame_relevance = []
    for frame in video_frames:
        relevance = allocator.score_frame_relevance(frame, query)
        frame_relevance.append(relevance)

    # Allocate budgets across frames
    frame_budgets = map_relevance_to_budgets(
        frame_relevance,
        total_budget=90_316_800,  # Total pixel budget
        min_per_frame=100_352,
        max_per_frame=602_112
    )

    # Process each frame with variable budget
    processed_frames = []
    for frame, budget in zip(video_frames, frame_budgets):
        resized_frame = smart_resize(
            frame.height,
            frame.width,
            factor=28,
            max_pixels=budget
        )
        processed_frames.append(resized_frame)

    return processed_frames
```

### process_vision_info() ⭐⭐⭐

**Lines**: 569-608

**Purpose**: **MAIN ENTRY POINT** - Process all images and videos in conversation

**This is THE function to modify for ARR-COC integration!**

**Signature**:
```python
def process_vision_info(
    conversations: list,
    image_patch_size: int = 14,
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False
) -> Tuple[Optional[list], Optional[list], Optional[dict]]:
```

**Current Flow**:
```python
# Line 575: Extract vision info from conversations
vision_infos = extract_vision_info(conversations)

# Lines 577-588: Load all images and videos
image_inputs = []
video_inputs = []

for vision_info in vision_infos:
    if "image" in vision_info or "image_url" in vision_info:
        # Process image with UNIFORM budget
        image_inputs.append(fetch_image(vision_info, image_patch_size))

    elif "video" in vision_info:
        # Process video with UNIFORM per-frame budget
        video_input, sample_fps = fetch_video(vision_info, ...)
        video_inputs.append(video_input)

# Lines 591-600: Format outputs
if len(image_inputs) == 0:
    image_inputs = None
if len(video_inputs) == 0:
    video_inputs = None

video_kwargs = {'do_sample_frames': False, 'fps': video_sample_fps_list}

return image_inputs, video_inputs, video_kwargs
```

**🎯 ARR-COC Enhanced Flow**:
```python
def arr_coc_process_vision_info(
    conversations: list,
    query: str,  # NEW: query for relevance
    allocator: RelevanceAllocator,  # NEW: ARR-COC allocator
    **kwargs
):
    from arr_coc_vis import map_relevance_to_budgets

    vision_infos = extract_vision_info(conversations)

    image_inputs = []
    video_inputs = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            # Load full image
            full_image = Image.open(vision_info["image"])

            # ARR-COC: Score relevance
            relevance_scores = allocator.realize_relevance(
                image=full_image,
                query=query,
                patch_grid=(16, 16)
            )

            # Map to token budgets
            token_budgets = map_relevance_to_budgets(
                relevance_scores,
                min_tokens=64,
                max_tokens=400
            )

            # Process with variable budgets
            image = fetch_image_with_variable_budgets(
                vision_info,
                token_budgets
            )
            image_inputs.append(image)

        elif "video" in vision_info:
            # ARR-COC: Score frame relevance
            video = fetch_video_with_relevance(
                vision_info,
                query,
                allocator
            )
            video_inputs.append(video)

    return image_inputs, video_inputs, video_kwargs
```

## Integration Summary

**Three levels of modification**:

1. **Minimal**: Modify `max_pixels` per image
   - Change line 152 to use query-aware budgets
   - Easiest, but uniform within each image

2. **Moderate**: Modify `fetch_image()` to support patch-level budgets
   - Split image into patches
   - Apply `smart_resize()` per patch with variable budgets
   - Reconstruct from patches

3. **Complete**: Modify `process_vision_info()` for full ARR-COC
   - Integrate RelevanceAllocator
   - Per-patch relevance scoring
   - Variable budget allocation
   - Hierarchical DeepStack injection

## Related Topics

- [concepts/03-dynamic-resolution.md](../concepts/03-dynamic-resolution.md) - Smart resize algorithm
- [concepts/05-arr-coc-integration.md](../concepts/05-arr-coc-integration.md) - Full integration
- [usage/02-dynamic-resolution.md](../usage/02-dynamic-resolution.md) - Practical usage

## Full Code Location

**File**: `RESEARCH/Qwen3VL/Qwen3-VL/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
**Lines**: 1-608
**Repository**: https://github.com/QwenLM/Qwen3-VL
