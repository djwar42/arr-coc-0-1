# Annotation Fields: IoU, Stability, Prompts

**Detailed breakdown of SA-1B annotation metadata fields**

---

## 1. predicted_iou Field

**Model's Quality Prediction:**
- Range: 0.0 - 1.0
- Predicts mask IoU with ground truth
- Higher = better quality

**Typical Distribution:**
- Excellent (>0.95): 40% of masks
- Good (0.85-0.95): 35%
- Acceptable (0.75-0.85): 20%
- Low (<0.75): 5%

**Usage:**
```python
high_quality = [a for a in anns if a["predicted_iou"] > 0.9]
```

---

## 2. stability_score Field

**Robustness Metric:**
- Measures mask consistency under perturbations
- Tests multiple threshold values
- Range: 0.0 - 1.0

**Calculation:**
- Apply thresholds: [0.4, 0.45, 0.5, 0.55, 0.6]
- Measure IoU between resulting masks
- Average IoU = stability_score

**High stability (>0.9):** Mask unchanged by threshold
**Low stability (<0.8):** Sensitive to threshold choice

---

## 3. point_coords Field

**Prompt Tracking:**
```json
"point_coords": [[x, y]]
```

**Purpose:**
- Records which point generated this mask
- Used during SA-1B annotation
- Can reproduce mask with SAM

**Example:**
```python
from segment_anything import SamPredictor
predictor.set_image(image)
point = ann["point_coords"][0]
mask = predictor.predict(point_coords=np.array([point]))
```

---

## 4. crop_box Field

**Image Region:**
```json
"crop_box": [x0, y0, x1, y1]
```

- Full image: `[0, 0, width, height]`
- Used for cropped predictions
- Rarely used in SA-1B (mostly full images)

---

## 5. area Field

**Mask Size:**
- Pixel count in mask
- Sum of binary mask values
- Useful for filtering by object size

**Distribution:**
- Small objects (<1000px): 45%
- Medium (1000-10000px): 40%
- Large (>10000px): 15%

---

## 6. bbox Field

**Bounding Box [x, y, width, height]:**
- Tight box around mask
- COCO format
- Useful for region proposals

**Aspect Ratio Analysis:**
```python
aspect = bbox[2] / bbox[3]  # width/height
```

---

## 7. id Field

**Unique Annotation ID:**
- Integer identifier
- Unique across entire SA-1B
- Used for referencing specific masks

---

## 8. ARR-COC-0-1 Integration (10%)

**Using Quality Metrics:**
- Filter masks: `predicted_iou > 0.9 AND stability_score > 0.85`
- Use point_coords for attention grounding
- Area-based relevance weighting (larger = more salient)
- bbox for region-based attention masks
