# Image Processing Pipeline

**File**: `process/image_process.py`

## Main Functions

### 1. Dynamic Tiling (lines 45-120)

```python
def find_best_resize(image_width, image_height, base_size, image_size, crop_mode):
    """Find best aspect ratio match for tiling"""
    # Tries different tile configurations
    # Returns: best_resize, best_aspect_ratio
```

**Purpose**: Content-aware tiling (not uniform grid!)

### 2. Image Preprocessing (lines 122-180)

```python
def process_images(images, base_size=1024, image_size=1024, crop_mode=False):
    """Main preprocessing pipeline"""
    # 1. Load images
    # 2. Find best tiling
    # 3. Resize
    # 4. Normalize (ImageNet stats)
    # 5. Create tensor
```

### 3. Vision/Text Masking (lines 182-250)

```python
def create_vision_text_mask(num_patches):
    """Create mask distinguishing vision vs text tokens"""
    # Vision tokens: True
    # Text tokens: False
    # Newlines: True (vision sequence separators)
```

### 4. Padding (lines 252-340)

```python
def pad_images(images, target_size):
    """Pad to square + normalize"""
    # ImageNet normalization
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
```

## Processing Flow

```
Input: PIL Image or path
    ↓
1. Load & decode
    ↓
2. Find best tiling (aspect ratio matching)
    ↓
3. Resize to target resolution
    ↓
4. Pad to square if needed
    ↓
5. Normalize (ImageNet stats)
    ↓
6. Convert to tensor [C, H, W]
    ↓
7. Create vision/text mask
    ↓
Output: Preprocessed tensor + mask
```

**See Also**:
- [token-calculation.md](token-calculation.md) - How token counts are determined
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - Resolution modes
