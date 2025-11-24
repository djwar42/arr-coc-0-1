# Native Resolution Processing

**Category**: Concepts
**Related**: [../architecture/01-navit-vision.md](../architecture/01-navit-vision.md)

## Problem

Fixed tiling distorts images:
- 1920×1080 → 336×336 → Stretched
- Information loss from distortion

## Solution

Process at native resolution within budget:
- 1920×1080 → 1792×896 → Preserved
- smart_resize preserves aspect ratio

## Benefits

- No distortion
- Better for documents (extreme aspect ratios)
- Efficient (small images not over-processed)

## Implementation

```python
def smart_resize(h, w, min_pixels, max_pixels):
    pixels = h * w
    if pixels > max_pixels:
        scale = sqrt(max_pixels / pixels)
        return int(h*scale), int(w*scale)
    # ...
```
