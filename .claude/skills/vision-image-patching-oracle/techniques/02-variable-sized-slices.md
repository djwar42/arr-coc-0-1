# Variable-Sized Slices

**Image modularization strategy that divides images into adaptive slices preserving aspect ratio**

## Overview

Variable-sized slicing divides images into flexible-dimension regions based on aspect ratio and resolution, avoiding fixed-size constraints that cause padding or distortion.

## Core Concept

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

**Traditional approach**:
- Fixed slices (e.g., always 512×512 grid)
- Padding required for non-divisible dimensions
- Information loss at boundaries

**Variable-sized approach**:
- Adaptive slice dimensions based on image shape
- No padding needed
- Natural boundary preservation

**Analogy**: "Water drops vs ice cubes" - water adapts to container shape, ice cubes force-fit with gaps.

## Implementation

```python
def variable_sized_slicing(image, target_size=336, max_slices=9):
    """Divide image into variable-sized slices"""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    # Determine slice configuration
    if aspect_ratio > 1.5:  # Wide image
        num_w, num_h = calculate_wide_config(w, h, target_size)
    elif aspect_ratio < 0.67:  # Tall image
        num_w, num_h = calculate_tall_config(w, h, target_size)
    else:  # Balanced
        num_w, num_h = calculate_balanced_config(w, h, target_size)
    
    # Create variable-sized slices
    slice_w = w // num_w
    slice_h = h // num_h
    
    slices = []
    for i in range(num_h):
        for j in range(num_w):
            # Handle edge slices (may be slightly larger to avoid remainder)
            w_start, w_end = j * slice_w, (j+1) * slice_w if j < num_w-1 else w
            h_start, h_end = i * slice_h, (i+1) * slice_h if i < num_h-1 else h
            
            slice_img = image[h_start:h_end, w_start:w_end]
            slices.append((slice_img, (i, j)))
    
    return slices
```

## Benefits

1. **No shape distortion**: Preserves original aspect ratio within each slice
2. **No padding waste**: Uses all image pixels efficiently
3. **Flexible resolution**: Adapts to any input size
4. **Better for pretraining**: Slices stay close to encoder's trained distribution

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)

## Related Documents

- [../architecture/03-native-resolution.md](../architecture/03-native-resolution.md)
- [../models/02-llava-uhd.md](../models/02-llava-uhd.md)
