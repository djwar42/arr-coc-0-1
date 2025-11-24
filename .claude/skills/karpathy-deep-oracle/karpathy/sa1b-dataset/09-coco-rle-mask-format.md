# COCO RLE Mask Format in SA-1B

## Overview

SA-1B uses **COCO Run-Length Encoding (RLE)** for compact mask storage. RLE compresses binary masks by storing alternating run lengths of 0s and 1s, achieving ~10-20× compression over raw bitmaps.

**Key facts:**
- 1.1B masks stored as RLE (not raw pixels)
- Reduces 11M images × 100 masks × 256×256 from ~7TB to ~340GB
- Decode with `pycocotools.mask.decode()`
- Binary format: alternating counts `[0_count, 1_count, 0_count, ...]`

## COCO RLE Format

**Format**: `{"size": [H, W], "counts": [run1, run2, ...]}`

**Example**:
```json
{
  "size": [10, 10],
  "counts": [10, 5, 35, 10, 40]
}
```
Means: 10 zeros, 5 ones, 35 zeros, 10 ones, 40 zeros

## Decoding RLE

```python
from pycocotools import mask as mask_utils
import numpy as np

# Decode RLE to binary mask
rle_mask = {
    "size": [256, 256],
    "counts": [1234, 567, 8901, ...]
}
binary_mask = mask_utils.decode(rle_mask)  # (256, 256) uint8 array
```

## ARR-COC Application

**Efficient mask storage for spatial grounding training:**
- Store millions of relevance masks compactly
- Stream from disk without loading all masks
- Fast decode during training (< 1ms per mask)

**Sources**: COCO Dataset Paper, SA-1B Technical Report
