# KNOWLEDGE DROP: RLE Mask Decoding with pycocotools

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 22)
**File Created**: `sa1b-dataset/21-rle-decoding-pycocotools.md`

---

## What Was Created

**Knowledge File**: RLE Mask Decoding with pycocotools (~700 lines)

**8 Sections**:
1. Run-Length Encoding (RLE) Overview
2. pycocotools Installation
3. mask_utils.decode() Function
4. Batch Decoding Strategies
5. Vectorized Operations
6. Memory-Efficient Decoding
7. GPU-Accelerated Decoding
8. **ARR-COC-0-1** (10%): Efficient mask processing for spatial relevance

---

## Key Insights

### Understanding COCO RLE Format

**What is RLE?**
- Run-Length Encoding compresses binary masks
- Stores alternating counts of 0s and 1s
- SA-1B uses COCO's compressed RLE variant

**COCO RLE structure**:
```python
{
    'size': [height, width],  # Image dimensions
    'counts': 'encoded_string'  # Compressed RLE data
}
```

From [pycocotools mask.py](https://robotics.cse.unsw.edu.au/gitlab/hri-edu-au/cocoapi/-/blob/727b546dd9fa4e4bb113213c98a3925829fac0bf/PythonAPI/pycocotools/mask.py):
- encode: Encode binary masks using RLE
- decode: Decode binary masks from RLE
- merge: Compute union or intersection
- iou: Compute intersection over union

### pycocotools Installation

```bash
# Standard installation
pip install pycocotools

# For Windows (may need Visual C++ build tools)
pip install pycocotools-windows

# From source (if issues)
pip install cython
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

### Basic mask_utils.decode()

```python
from pycocotools import mask as mask_utils
import numpy as np

def decode_rle_mask(rle_annotation):
    """
    Decode single RLE annotation to binary mask.

    Args:
        rle_annotation: Dict with 'size' and 'counts' keys

    Returns:
        numpy.ndarray: Binary mask of shape (H, W)
    """
    # Decode RLE to binary mask
    mask = mask_utils.decode(rle_annotation)

    return mask  # Shape: (H, W), dtype: uint8, values: 0 or 1


# Example usage
rle = {
    'size': [480, 640],
    'counts': 'Yjj31fc06J5L4M2N2N1O1O2N1O1O1O1O2N1O100O1...'
}

mask = decode_rle_mask(rle)
print(f"Mask shape: {mask.shape}")  # (480, 640)
print(f"Mask dtype: {mask.dtype}")  # uint8
print(f"Unique values: {np.unique(mask)}")  # [0, 1]
```

### Decoding SA-1B Annotations

```python
import json
from pycocotools import mask as mask_utils

def decode_sa1b_annotation(annotation_path):
    """
    Decode all masks from SA-1B annotation file.

    Args:
        annotation_path: Path to JSON annotation file

    Returns:
        List of binary masks, metadata dict
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    image_info = data.get('image', {})
    annotations = data.get('annotations', [])

    masks = []
    metadata = {
        'bboxes': [],
        'areas': [],
        'predicted_ious': [],
        'stability_scores': [],
        'point_coords': []
    }

    for ann in annotations:
        # Decode RLE mask
        segmentation = ann['segmentation']
        mask = mask_utils.decode(segmentation)
        masks.append(mask)

        # Extract metadata
        metadata['bboxes'].append(ann.get('bbox', [0, 0, 0, 0]))
        metadata['areas'].append(ann.get('area', 0))
        metadata['predicted_ious'].append(ann.get('predicted_iou', 0.0))
        metadata['stability_scores'].append(ann.get('stability_score', 0.0))
        metadata['point_coords'].append(ann.get('point_coords', [[0, 0]]))

    return masks, metadata


# Example
masks, metadata = decode_sa1b_annotation('sa_1.json')
print(f"Decoded {len(masks)} masks")
print(f"First mask shape: {masks[0].shape}")
```

### Batch Decoding Strategies

**Challenge**: SA-1B has ~100 masks per image

```python
def batch_decode_masks(annotations: list) -> np.ndarray:
    """
    Efficiently decode multiple RLE masks.

    Args:
        annotations: List of SA-1B annotation dicts

    Returns:
        numpy.ndarray: Stacked masks (N, H, W)
    """
    if not annotations:
        return np.array([])

    # Get dimensions from first annotation
    first_seg = annotations[0]['segmentation']
    h, w = first_seg['size']

    # Pre-allocate output array
    n_masks = len(annotations)
    masks = np.zeros((n_masks, h, w), dtype=np.uint8)

    # Decode each mask
    for i, ann in enumerate(annotations):
        masks[i] = mask_utils.decode(ann['segmentation'])

    return masks


def batch_decode_parallel(annotations: list, num_workers: int = 4):
    """
    Parallel batch decoding using multiprocessing.
    """
    from multiprocessing import Pool

    def decode_single(ann):
        return mask_utils.decode(ann['segmentation'])

    with Pool(num_workers) as pool:
        masks = pool.map(decode_single, annotations)

    return np.stack(masks)
```

### Vectorized Operations

**pycocotools supports batch operations**:

```python
def merge_masks_vectorized(annotations: list) -> np.ndarray:
    """
    Merge multiple masks using vectorized operations.

    Args:
        annotations: List of RLE annotations

    Returns:
        Merged binary mask
    """
    # Extract RLE objects
    rles = [ann['segmentation'] for ann in annotations]

    # Merge all masks (union)
    merged_rle = mask_utils.merge(rles)

    # Decode merged result
    merged_mask = mask_utils.decode(merged_rle)

    return merged_mask


def compute_iou_matrix(annotations: list) -> np.ndarray:
    """
    Compute pairwise IoU between all masks.

    Args:
        annotations: List of RLE annotations

    Returns:
        IoU matrix of shape (N, N)
    """
    rles = [ann['segmentation'] for ann in annotations]

    # Compute IoU matrix (vectorized in C)
    iou_matrix = mask_utils.iou(rles, rles, [0] * len(rles))

    return iou_matrix
```

### Memory-Efficient Decoding

**For very large images or many masks**:

```python
def memory_efficient_decode(
    annotation_path: str,
    max_masks: int = 100,
    chunk_size: int = 20
):
    """
    Decode masks in chunks to limit memory usage.

    Args:
        annotation_path: Path to annotation file
        max_masks: Maximum masks to decode
        chunk_size: Masks per chunk

    Yields:
        Chunks of decoded masks
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])[:max_masks]

    for i in range(0, len(annotations), chunk_size):
        chunk = annotations[i:i+chunk_size]
        masks = batch_decode_masks(chunk)
        yield masks, chunk


def streaming_decode(annotation_path: str):
    """
    Generator that decodes one mask at a time.
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    for ann in data.get('annotations', []):
        mask = mask_utils.decode(ann['segmentation'])
        yield mask, ann
```

### Encoding Binary Masks to RLE

**Convert numpy arrays back to RLE**:

```python
def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """
    Encode binary mask to COCO RLE format.

    Args:
        binary_mask: Binary numpy array (H, W)

    Returns:
        RLE dict with 'size' and 'counts'
    """
    # Ensure correct format
    mask = np.asfortranarray(binary_mask.astype(np.uint8))

    # Encode to RLE
    rle = mask_utils.encode(mask)

    # Convert counts to string if bytes
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')

    return rle


def batch_encode_masks(masks: np.ndarray) -> list:
    """
    Encode multiple masks to RLE.

    Args:
        masks: Array of shape (N, H, W)

    Returns:
        List of RLE dicts
    """
    rles = []
    for mask in masks:
        rle = encode_mask_to_rle(mask)
        rles.append(rle)

    return rles
```

### GPU-Accelerated Decoding

**Using CuPy for GPU acceleration**:

```python
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def gpu_decode_rle(rle_annotation, stream=None):
    """
    Decode RLE mask on GPU using CuPy.

    Note: RLE decoding itself must be on CPU,
    but we can transfer immediately to GPU.
    """
    # Decode on CPU (pycocotools is CPU-only)
    mask = mask_utils.decode(rle_annotation)

    if HAS_CUPY:
        # Transfer to GPU
        with cp.cuda.Stream(stream):
            gpu_mask = cp.asarray(mask)
        return gpu_mask

    return mask


def batch_decode_to_gpu(annotations: list):
    """
    Decode masks and transfer to GPU efficiently.
    """
    # Decode all on CPU
    masks = batch_decode_masks(annotations)

    if HAS_CUPY:
        # Single transfer to GPU
        return cp.asarray(masks)

    return masks
```

**PyTorch GPU transfer**:

```python
import torch

def decode_to_torch(annotations: list, device='cuda'):
    """
    Decode masks and create PyTorch tensor on device.
    """
    # Decode to numpy
    masks = batch_decode_masks(annotations)

    # Convert to PyTorch tensor on device
    mask_tensor = torch.from_numpy(masks).to(device)

    return mask_tensor
```

---

## Complete SA-1B Mask Decoder Class

```python
from pycocotools import mask as mask_utils
import numpy as np
import json
from pathlib import Path

class SA1BMaskDecoder:
    """
    Efficient mask decoder for SA-1B dataset.

    Features:
    - Batch decoding
    - Memory-efficient chunking
    - Optional GPU transfer
    - Caching for repeated access
    """

    def __init__(
        self,
        max_masks: int = 100,
        cache_decoded: bool = False,
        device: str = 'cpu'
    ):
        self.max_masks = max_masks
        self.cache_decoded = cache_decoded
        self.device = device
        self._cache = {}

    def decode(self, annotation_path: str) -> dict:
        """
        Decode all masks from annotation file.

        Returns:
            Dict with 'masks' tensor and metadata
        """
        # Check cache
        if self.cache_decoded and annotation_path in self._cache:
            return self._cache[annotation_path]

        # Load annotation
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        annotations = data.get('annotations', [])

        # Limit masks
        if len(annotations) > self.max_masks:
            # Sort by predicted_iou (quality)
            annotations = sorted(
                annotations,
                key=lambda x: x.get('predicted_iou', 0),
                reverse=True
            )[:self.max_masks]

        # Batch decode
        masks = self._batch_decode(annotations)

        # Extract metadata
        result = {
            'masks': masks,
            'bboxes': np.array([a.get('bbox', [0,0,0,0]) for a in annotations]),
            'areas': np.array([a.get('area', 0) for a in annotations]),
            'predicted_ious': np.array([a.get('predicted_iou', 0) for a in annotations]),
            'stability_scores': np.array([a.get('stability_score', 0) for a in annotations])
        }

        # Transfer to device
        if self.device != 'cpu':
            result = self._to_device(result)

        # Cache
        if self.cache_decoded:
            self._cache[annotation_path] = result

        return result

    def _batch_decode(self, annotations: list) -> np.ndarray:
        """Decode batch of annotations."""
        if not annotations:
            return np.array([])

        masks = []
        for ann in annotations:
            mask = mask_utils.decode(ann['segmentation'])
            masks.append(mask)

        return np.stack(masks)

    def _to_device(self, result: dict) -> dict:
        """Transfer result to device."""
        import torch

        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).to(self.device)

        return result

    def clear_cache(self):
        """Clear decoded mask cache."""
        self._cache.clear()
```

---

## Research Performed

**Web sources consulted**:
1. [GitHub: cocodataset/cocoapi Issue #492](https://github.com/cocodataset/cocoapi/issues/492) - RLE encoding details
2. [StackOverflow: Encode numpy array using RLE](https://stackoverflow.com/questions/49494337) - Encoding/decoding examples
3. [RLEMaskLib Documentation](https://istvansarandi.com/docs/rlemasklib/) - Alternative library
4. [Kaggle: Decoding masks from RLE](https://www.kaggle.com/code/philiphucklesby/decoding-masks-from-the-rle-string) - Practical examples
5. [pycocotools mask.py source](https://robotics.cse.unsw.edu.au/gitlab/hri-edu-au/cocoapi) - API reference

**Source document**:
- SAM_DATASET_SA1B.md (lines 125-145: RLE format specification)

---

## ARR-COC-0-1 Integration (10%)

### Efficient Mask Processing for Spatial Relevance

```python
class ARRCOCMaskProcessor(SA1BMaskDecoder):
    """
    Extended decoder for ARR-COC spatial relevance training.
    """

    def decode_with_spatial(self, annotation_path: str) -> dict:
        """Decode masks with spatial relationship features."""
        result = self.decode(annotation_path)

        # Compute spatial features
        masks = result['masks']

        # Centroid positions
        result['centroids'] = self._compute_centroids(masks)

        # Containment relationships
        result['containment'] = self._compute_containment(masks)

        # IoU matrix for mask relationships
        result['iou_matrix'] = self._compute_iou_matrix(masks)

        return result

    def _compute_centroids(self, masks):
        """Compute mask centroids."""
        centroids = []
        for mask in masks:
            y, x = np.where(mask > 0)
            if len(y) > 0:
                centroids.append([x.mean(), y.mean()])
            else:
                centroids.append([0, 0])
        return np.array(centroids)

    def _compute_containment(self, masks):
        """Compute which masks contain others."""
        n = len(masks)
        containment = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    overlap = (masks[i] * masks[j]).sum()
                    containment[i, j] = overlap / (masks[j].sum() + 1e-6)

        return containment
```

**Benefits**:
- **Efficient decoding**: Batch processing with caching
- **Spatial features**: Pre-computed for training
- **GPU ready**: Direct transfer to device

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: 12+ (decoding, encoding, batch, GPU)
- **Sections**: 8 (7 technical + 1 ARR-COC at 10%)
- **Web sources**: 5 cited with URLs
- **Completion time**: ~45 minutes

---

## Next Steps

**PART 23** will cover: Mask Visualization Techniques
- Matplotlib overlays
- Alpha blending
- Multi-mask display
