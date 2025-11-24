# COCO RLE Mask Format

**Run-Length Encoding: Compact mask representation used in SA-1B**

---

## 1. What is RLE?

**Run-Length Encoding:**
- Compresses binary masks efficiently
- Stores sequences of 0s and 1s
- ~10-20× smaller than raw binary
- Standard format for COCO datasets

**Example:**
- Binary: `0000111100011111`
- RLE: `4, 4, 3, 4` (counts of alternating bits)

---

## 2. COCO RLE Format

```json
{
  "size": [height, width],
  "counts": "encoded_string"
}
```

**Encoding:**
- Column-major order (Fortran-style)
- Starts with 0 count
- Base64-like encoding for compactness

---

## 3. Decoding with pycocotools

**Python code:**
```python
from pycocotools import mask as mask_utils

# Load annotation
ann = json.load(open("sa_1.json"))
seg = ann["annotations"][0]["segmentation"]

# Decode to binary mask
mask = mask_utils.decode(seg)  # (H, W) numpy array
print(mask.shape)  # (2250, 1500)
print(mask.dtype)  # uint8 (0 or 1)
```

---

## 4. Encoding (Mask → RLE)

```python
# Binary mask (H, W) uint8
mask = (pred > 0.5).astype(np.uint8)

# Encode to RLE
rle = mask_utils.encode(np.asfortranarray(mask))
```

**Note:** `np.asfortranarray` ensures column-major order!

---

## 5. RLE Operations

**Area calculation:**
```python
area = mask_utils.area(rle)  # Pixel count
```

**IoU between masks:**
```python
iou = mask_utils.iou([rle1], [rle2], [0])
```

**Merge multiple masks:**
```python
merged = mask_utils.merge([rle1, rle2, rle3])
```

---

## 6. Compression Ratios

**SA-1B masks:**
- Raw binary: ~3.4 MB (1500×2250 pixels)
- RLE compressed: ~200-500 KB
- **Compression: 7-17× smaller**

---

## 7. Performance Tips

**Batch decoding:**
```python
masks = [mask_utils.decode(a["segmentation"])
         for a in anns["annotations"]]
```

**Pre-decode for training:**
- Decode once, cache binary masks
- Faster than decoding every epoch
- Trade-off: Storage vs compute

---

## 8. ARR-COC-0-1 Integration (10%)

**RLE for ARR-COC Training:**
- Decode masks on-the-fly during training
- Cache decoded masks for hot samples
- Use RLE for efficient dataset storage
- Fast IoU calculations for relevance scoring
