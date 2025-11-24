# Annotation Files: JSON Schema

**SA-1B annotation format: COCO-style JSON with image metadata and mask annotations**

---

## 1. JSON File Structure

**Each image has corresponding JSON:**
- Image: `sa_000000/sa_1.jpg`
- Annotation: `sa_000000/sa_1.json`

**Root Fields:**
```json
{
  "image": {...},           // Image metadata
  "annotations": [...]      // Array of mask annotations
}
```

---

## 2. Image Metadata Fields

```json
"image": {
  "image_id": 1,
  "width": 1500,
  "height": 2250,
  "file_name": "sa_1.jpg"
}
```

**Fields:**
- `image_id`: Unique integer ID
- `width`, `height`: Image dimensions (variable)
- `file_name`: Filename (sa_N.jpg)

---

## 3. Annotations Array

**~100 mask annotations per image:**
```json
"annotations": [
  {
    "id": 1850012,
    "segmentation": {...},        // COCO RLE mask
    "area": 26371,               // Pixel count
    "bbox": [943, 463, 180, 177], // [x, y, w, h]
    "predicted_iou": 0.983,      // Quality score
    "stability_score": 0.952,    // Robustness
    "crop_box": [0, 0, 1500, 2250],
    "point_coords": [[1050, 550]] // Prompt used
  },
  ...
]
```

---

## 4. Segmentation Field (COCO RLE)

**Run-Length Encoding:**
```json
"segmentation": {
  "size": [2250, 1500],    // [height, width]
  "counts": "aB3d2..."     // RLE compressed string
}
```

Decoding requires `pycocotools`:
```python
from pycocotools import mask as mask_utils
mask = mask_utils.decode(seg)  # → binary numpy array
```

---

## 5. Quality Metrics

**predicted_iou (0-1):**
- Model's confidence in mask quality
- Higher = better mask
- Typical range: 0.7-0.99

**stability_score (0-1):**
- Robustness to threshold perturbations
- Measures mask consistency
- Typical range: 0.8-0.99

---

## 6. Bounding Box Format

**COCO format [x, y, width, height]:**
- `x`, `y`: Top-left corner
- `width`, `height`: Box dimensions
- Pixel coordinates (0-indexed)

---

## 7. Point Coords (Prompt Tracking)

**[[x, y]] format:**
- Tracks which point prompt generated this mask
- Used during SA-1B annotation process
- Can be used to reproduce mask generation

---

## 8. ARR-COC-0-1 Integration (10%)

**SA-1B JSON for Spatial Grounding:**
- Load masks with quality filtering (IoU > 0.9)
- Use bbox for region proposals
- Point coords for attention grounding
- Hierarchical mask organization for relevance levels
