# COCO RLE Mask Format: Run-Length Encoding for Efficient Segmentation Storage

## Overview

Run-Length Encoding (RLE) is a lossless compression algorithm used by SA-1B and COCO datasets to efficiently store binary segmentation masks. Instead of storing every pixel (which would require millions of values per mask), RLE stores consecutive runs of identical values, achieving compression ratios of 10-100x for typical segmentation masks. This makes storing 1.1 billion masks in SA-1B practically feasible.

---

## 1. What is Run-Length Encoding?

### Basic Concept

RLE compresses data by replacing consecutive identical values with a single value and count:

```
Original:  0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0
RLE:       [5, 3, 4, 2, 6]  (5 zeros, 3 ones, 4 zeros, 2 ones, 6 zeros)
```

### Why RLE Works Well for Masks

Binary segmentation masks have properties that make RLE highly effective:
- **Spatial coherence**: Object pixels cluster together
- **Binary values**: Only 0 (background) and 1 (foreground)
- **Large contiguous regions**: Most masks have smooth boundaries

**Compression ratios:**
- Small objects: 50-100x compression
- Medium objects: 20-50x compression
- Large complex objects: 10-20x compression

---

## 2. COCO RLE Format Specification

### Structure in JSON

```json
{
    "segmentation": {
        "size": [height, width],
        "counts": "encoded_string_or_list"
    }
}
```

**Fields:**
- `size`: [height, width] of the mask (matches image dimensions)
- `counts`: RLE-encoded data (string for uncompressed, string for compressed)

### Uncompressed vs Compressed RLE

**Uncompressed RLE** (list of integers):
```json
{
    "size": [480, 640],
    "counts": [10000, 500, 20000, 800, 276700]
}
```

**Compressed RLE** (LEB128 encoded string):
```json
{
    "size": [480, 640],
    "counts": "XdP31o0e0O2N1O1N2O1N101N2O0O2O0O1O100O100O10001N100O01O0010O010O0"
}
```

SA-1B uses **compressed RLE** (string format) for maximum storage efficiency.

---

## 3. Column-Major Ordering

### Critical Detail: Fortran-Style Ordering

COCO RLE uses **column-major (Fortran) ordering**, not row-major (C) ordering:

```python
# Image pixels are read column-by-column:
# Column 0: pixels [0,0], [1,0], [2,0], ..., [height-1,0]
# Column 1: pixels [0,1], [1,1], [2,1], ..., [height-1,1]
# etc.
```

This is important when:
- Manually encoding/decoding masks
- Debugging mask alignment issues
- Working with non-COCO formats

**pycocotools handles this automatically** - you don't need to worry about ordering when using the official tools.

---

## 4. Encoding Algorithm

### Step-by-Step Encoding

```python
import numpy as np

def encode_rle_uncompressed(binary_mask):
    """
    Encode binary mask to uncompressed RLE.

    Args:
        binary_mask: numpy array of shape (height, width) with 0/1 values

    Returns:
        dict with 'size' and 'counts' (list of run lengths)
    """
    # Flatten in column-major (Fortran) order
    pixels = binary_mask.flatten(order='F')

    # Find where values change
    runs = []
    current_value = 0  # Always start with background (0)
    run_length = 0

    for pixel in pixels:
        if pixel == current_value:
            run_length += 1
        else:
            runs.append(run_length)
            run_length = 1
            current_value = pixel

    # Append final run
    runs.append(run_length)

    return {
        'size': list(binary_mask.shape),
        'counts': runs
    }

# Example
mask = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 1, 0]
])
rle = encode_rle_uncompressed(mask)
print(rle)
# {'size': [3, 4], 'counts': [3, 2, 1, 4, 1, 1]}
```

### LEB128 Compression (COCO Compressed Format)

COCO uses Little-Endian Base 128 (LEB128) to compress run-length counts:

```python
def encode_leb128(counts):
    """Encode list of counts to LEB128 compressed string."""
    result = []
    for count in counts:
        # Convert to LEB128 bytes
        while count > 0:
            byte = count & 0x7F  # Take 7 bits
            count >>= 7
            if count > 0:
                byte |= 0x80  # Set continuation bit
            result.append(byte)

    # Convert to ASCII string representation
    return ''.join(chr(b + 48) for b in result)  # Offset by 48 for printable chars
```

**Note:** The actual COCO implementation is more complex. Use pycocotools for production code.

---

## 5. Decoding Algorithm

### Using pycocotools (Recommended)

```python
from pycocotools import mask as mask_utils
import numpy as np

def decode_rle(rle):
    """
    Decode COCO RLE to binary mask using pycocotools.

    Args:
        rle: dict with 'size' and 'counts'

    Returns:
        numpy array of shape (height, width) with 0/1 values
    """
    return mask_utils.decode(rle)

# Example usage
rle = {
    "size": [1500, 2250],
    "counts": "XdP31o0e0O2N1O1N2O1N101N2O0O2O0O1O100..."
}
binary_mask = decode_rle(rle)
print(f"Mask shape: {binary_mask.shape}")  # (1500, 2250)
print(f"Foreground pixels: {binary_mask.sum()}")
```

### Manual Decoding (Educational)

```python
def decode_rle_uncompressed(rle):
    """
    Decode uncompressed RLE to binary mask.

    Args:
        rle: dict with 'size' and 'counts' (list)

    Returns:
        numpy array of shape (height, width)
    """
    height, width = rle['size']
    counts = rle['counts']

    # Build flat pixel array
    pixels = []
    current_value = 0  # Start with background

    for count in counts:
        pixels.extend([current_value] * count)
        current_value = 1 - current_value  # Toggle 0/1

    # Reshape in column-major order
    mask = np.array(pixels).reshape((height, width), order='F')
    return mask

# Verify round-trip
original = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
rle = encode_rle_uncompressed(original)
decoded = decode_rle_uncompressed(rle)
assert np.array_equal(original, decoded)
```

---

## 6. Compression Ratios and Efficiency

### Storage Analysis

**Uncompressed mask:**
```
1500 x 2250 pixels = 3,375,000 bytes (3.2 MB per mask)
11 million images x 100 masks x 3.2 MB = 3.52 PB (petabytes!)
```

**RLE compressed:**
```
Average compressed size: ~500 bytes to 5 KB per mask
11 million images x 100 masks x 2.5 KB = ~2.75 TB
```

**Compression ratio: ~1000x to 5000x** for typical masks

### Factors Affecting Compression

**Better compression:**
- Simple shapes (circles, rectangles)
- Small objects (fewer foreground pixels)
- Objects without holes
- Smooth boundaries

**Worse compression:**
- Complex textures
- Many small disconnected regions
- Highly irregular boundaries
- Very large objects

### Measuring Compression

```python
import sys

def analyze_compression(binary_mask):
    """Analyze compression ratio for a mask."""
    from pycocotools import mask as mask_utils

    # Uncompressed size (1 byte per pixel)
    uncompressed_size = binary_mask.size

    # RLE compressed size
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    compressed_size = len(rle['counts'])

    ratio = uncompressed_size / compressed_size

    return {
        'uncompressed_bytes': uncompressed_size,
        'compressed_bytes': compressed_size,
        'compression_ratio': ratio
    }

# Example
mask = np.zeros((1500, 2250), dtype=np.uint8)
mask[500:1000, 800:1400] = 1  # Rectangle

stats = analyze_compression(mask)
print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
```

---

## 7. pycocotools Usage Guide

### Installation

```bash
pip install pycocotools

# Or from source for latest version
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

### Core Functions

```python
from pycocotools import mask as mask_utils

# Decode RLE to binary mask
binary_mask = mask_utils.decode(rle)

# Encode binary mask to RLE
# IMPORTANT: Must be Fortran-contiguous array
mask_fortran = np.asfortranarray(binary_mask)
rle = mask_utils.encode(mask_fortran)

# Calculate mask area
area = mask_utils.area(rle)

# Get bounding box [x, y, width, height]
bbox = mask_utils.toBbox(rle)

# Calculate IoU between masks
iou = mask_utils.iou([rle1], [rle2], [0])

# Merge multiple masks
merged_rle = mask_utils.merge([rle1, rle2, rle3])
```

### Working with Multiple Masks

```python
def process_all_masks(annotation_data):
    """Process all masks in an SA-1B annotation."""
    results = []

    for ann in annotation_data["annotations"]:
        rle = ann["segmentation"]

        # Decode to binary mask
        mask = mask_utils.decode(rle)

        # Compute properties
        area = mask_utils.area(rle)
        bbox = mask_utils.toBbox(rle)

        results.append({
            "id": ann["id"],
            "mask": mask,
            "area": float(area),
            "bbox": bbox.tolist(),
            "predicted_iou": ann["predicted_iou"]
        })

    return results
```

---

## 8. Batch Processing for Efficiency

### Vectorized Operations

```python
def batch_decode_masks(annotations):
    """Efficiently decode multiple RLE masks."""
    rles = [ann["segmentation"] for ann in annotations]

    # pycocotools can decode multiple RLEs efficiently
    masks = mask_utils.decode(rles)  # Returns (H, W, N) array

    return masks

def batch_compute_areas(annotations):
    """Compute areas for multiple masks efficiently."""
    rles = [ann["segmentation"] for ann in annotations]
    areas = mask_utils.area(rles)  # Returns numpy array
    return areas.tolist()
```

### Memory-Efficient Processing

```python
def process_masks_streaming(annotation_data, processor_fn):
    """
    Process masks one at a time to minimize memory usage.

    Args:
        annotation_data: SA-1B annotation dict
        processor_fn: Function to apply to each (mask, annotation) pair
    """
    for ann in annotation_data["annotations"]:
        # Decode single mask
        mask = mask_utils.decode(ann["segmentation"])

        # Process and immediately free memory
        result = processor_fn(mask, ann)

        # Yield result instead of collecting all
        yield result

        # Explicit cleanup for large masks
        del mask

# Usage
def compute_mask_stats(mask, ann):
    return {
        "id": ann["id"],
        "mean": mask.mean(),
        "nonzero": mask.sum()
    }

for stats in process_masks_streaming(data, compute_mask_stats):
    print(stats)
```

---

## 9. Common Operations

### Mask Visualization

```python
import matplotlib.pyplot as plt

def visualize_mask_overlay(image, mask, alpha=0.5, color=[1, 0, 0]):
    """Overlay mask on image."""
    overlay = image.copy().astype(float) / 255.0

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask == 1,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c]
        )

    return overlay

# Usage
image = np.array(Image.open("image.jpg"))
mask = mask_utils.decode(rle)
overlay = visualize_mask_overlay(image, mask)
plt.imshow(overlay)
plt.show()
```

### Mask to Polygon Conversion

```python
from skimage import measure

def mask_to_polygons(binary_mask):
    """Convert binary mask to polygon contours."""
    contours = measure.find_contours(binary_mask, 0.5)

    polygons = []
    for contour in contours:
        # Flip to (x, y) format and flatten
        polygon = contour[:, ::-1].flatten().tolist()
        if len(polygon) >= 6:  # Minimum 3 points
            polygons.append(polygon)

    return polygons
```

### IoU Computation

```python
def compute_mask_iou(rle1, rle2):
    """Compute Intersection over Union between two masks."""
    # Using pycocotools
    iou = mask_utils.iou([rle1], [rle2], [0])[0, 0]
    return float(iou)

def compute_pairwise_iou(rles):
    """Compute IoU matrix for all pairs of masks."""
    n = len(rles)
    iou_matrix = mask_utils.iou(rles, rles, [0] * n)
    return iou_matrix
```

---

## 10. RLE Format Conversion

### From Polygon to RLE

```python
def polygon_to_rle(polygon, height, width):
    """Convert polygon to RLE mask."""
    from pycocotools import mask as mask_utils

    # pycocotools expects specific format
    rle = mask_utils.frPyObjects([polygon], height, width)
    return mask_utils.merge(rle)
```

### From Binary Mask to RLE

```python
def binary_to_rle(binary_mask):
    """Convert binary numpy array to COCO RLE."""
    # MUST be Fortran-contiguous
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)

    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle
```

### Validation

```python
def validate_rle(rle, expected_height, expected_width):
    """Validate RLE format and dimensions."""
    errors = []

    # Check structure
    if 'size' not in rle or 'counts' not in rle:
        errors.append("Missing 'size' or 'counts'")
        return errors

    # Check dimensions
    if rle['size'] != [expected_height, expected_width]:
        errors.append(f"Size mismatch: {rle['size']} vs [{expected_height}, {expected_width}]")

    # Try decoding
    try:
        mask = mask_utils.decode(rle)
        if mask.shape != (expected_height, expected_width):
            errors.append(f"Decoded shape mismatch: {mask.shape}")
    except Exception as e:
        errors.append(f"Decode error: {e}")

    return errors
```

---

## 11. Performance Optimization

### Caching Decoded Masks

```python
from functools import lru_cache

class MaskCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get_mask(self, ann_id, rle):
        if ann_id in self.cache:
            return self.cache[ann_id]

        # Decode and cache
        mask = mask_utils.decode(rle)

        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[ann_id] = mask
        self.access_order.append(ann_id)
        return mask
```

### Parallel Decoding

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def parallel_decode_masks(annotations, num_workers=None):
    """Decode masks in parallel."""
    if num_workers is None:
        num_workers = mp.cpu_count()

    def decode_single(ann):
        return {
            "id": ann["id"],
            "mask": mask_utils.decode(ann["segmentation"])
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(decode_single, annotations))

    return results
```

---

## 12. ARR-COC-0-1 Integration: Efficient Mask Storage for VLM Training

### Why RLE Matters for ARR-COC

RLE compression is essential for ARR-COC spatial relevance training:

1. **Storage Efficiency**: Train on 1.1B masks without petabytes of storage
2. **Memory Management**: Decode only needed masks during batch loading
3. **I/O Performance**: Smaller files = faster loading = better GPU utilization
4. **Quality Preservation**: Lossless compression maintains mask precision

### Integration Patterns

```python
class SA1BDataLoader:
    """Efficient data loader for ARR-COC training on SA-1B."""

    def __init__(self, annotation_dir, cache_size=500):
        self.annotation_dir = annotation_dir
        self.mask_cache = MaskCache(cache_size)

    def load_training_sample(self, annotation_path):
        """Load sample with on-demand RLE decoding."""
        with open(annotation_path) as f:
            data = json.load(f)

        # Don't decode all masks - just quality ones
        quality_masks = [
            ann for ann in data["annotations"]
            if ann["predicted_iou"] >= 0.88
        ]

        # Decode only top-k by stability
        quality_masks.sort(key=lambda x: x["stability_score"], reverse=True)
        top_masks = quality_masks[:10]

        # Decode selected masks
        decoded = []
        for ann in top_masks:
            mask = self.mask_cache.get_mask(ann["id"], ann["segmentation"])
            decoded.append({
                "mask": mask,
                "bbox_normalized": self._normalize_bbox(
                    ann["bbox"],
                    data["image"]["width"],
                    data["image"]["height"]
                ),
                "confidence": ann["predicted_iou"]
            })

        return decoded

    def _normalize_bbox(self, bbox, width, height):
        x, y, w, h = bbox
        return [x/width, y/height, w/width, h/height]
```

---

## Sources

**Web Research:**
- [V7 Labs COCO Dataset Guide](https://www.v7labs.com/blog/coco-dataset-guide) - COCO format and segmentation details (accessed 2025-11-20)
- [Roboflow COCO JSON Format](https://roboflow.com/formats/coco-json) - RLE format description (accessed 2025-11-20)
- [GitHub isarandi/rlemasklib](https://github.com/isarandi/rlemasklib) - Efficient RLE mask operations library
- [Stack Overflow RLE Discussions](https://stackoverflow.com/questions/68838058/convert-coco-run-length-encoding-rle-back-to-segmentation) - RLE decoding Q&A

**Technical References:**
- pycocotools: Official COCO API (github.com/cocodataset/cocoapi)
- COCO Dataset Paper: "Microsoft COCO: Common Objects in Context" (Lin et al., 2014)
- SA-1B Paper: "Segment Anything" (Kirillov et al., 2023)

**ARR-COC Integration:**
- RLE enables efficient storage of spatial grounding masks for relevance training
- On-demand decoding minimizes memory footprint during batch processing
- Lossless compression preserves mask precision for accurate spatial grounding
