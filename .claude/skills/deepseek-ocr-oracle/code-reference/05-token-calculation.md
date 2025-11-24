# Token Calculation

**File**: `deepseek_ocr.py:61-106`

## Implementation

```python
def get_num_image_tokens(self, image_width, image_height, cropping):
    """Calculate actual token count including newlines"""

    if cropping:
        # Gundam mode (dynamic tiling)
        num_width_tiles = ...
        num_height_tiles = ...

        # Global view
        h, w = base_grid_size  # e.g., 16×16
        global_tokens = h * (w + 1)  # +1 for newlines

        # Local views
        h2, w2 = local_grid_size
        local_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)

        return global_tokens + local_tokens + 1

    else:
        # Base modes (Tiny/Small/Base/Large)
        h = image_height // patch_size
        w = image_width // patch_size

        # Formula: (w + 1) * h + 1
        # +1 per row for newline token
        # +1 final separator
        return h * (w + 1) + 1
```

## Examples

**Base mode (1024×1024)**:
- Grid: 16×16
- Formula: 16 * (16 + 1) + 1 = 16 * 17 + 1 = 273

**Tiny mode (512×512)**:
- Grid: 8×8
- Formula: 8 * (8 + 1) + 1 = 8 * 9 + 1 = 73

**Gundam (2×3 tiling at 640×640)**:
- Local: (10*2 + 1) * (10*3) = 21 * 30 = 630
- Global: 273
- Total: 903

**See Also**:
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - All modes
- [../concepts/token-budgets.md](../concepts/token-budgets.md) - Why these numbers?
