# SAM: Automatic Mask Generation

**PART 30/42 - Automatic Mask Generation (Grid-Based Dense Segmentation)**

**Date**: 2025-11-20
**Source**: SAM paper Section 4.3, Automatic Mask Generation API

---

## What is Automatic Mask Generation?

**Definition**: Generate all object masks in an image **without prompts** using a dense grid of query points.

**Use Case**: When you want to segment everything in the image (no prior knowledge of objects).

**Output**: Set of all valid object masks with confidence scores, bounding boxes, and stability metrics.

---

## How It Works

### 1. Grid Sampling Strategy

**Step 1: Generate Point Grid**
```python
# Example: 32×32 grid = 1,024 points
points = []
for x in range(0, image_width, stride):
    for y in range(0, image_height, stride):
        points.append((x, y))
```

**Stride Selection**:
- **Dense grid (stride=16)**: More masks, higher recall, slower
- **Sparse grid (stride=64)**: Fewer masks, faster inference, lower recall

### 2. Per-Point Mask Prediction

**For each grid point**:
1. Feed point as prompt to SAM
2. Get 3 mask predictions (multi-mask output)
3. Select mask with highest IoU confidence
4. Store mask + metadata (bbox, area, stability score)

**Total masks**: 1,024 points × 3 predictions = 3,072 candidate masks

### 3. Non-Maximum Suppression (NMS)

**Why NMS?**: Adjacent grid points produce overlapping masks for the same object.

**Algorithm**:
```python
def nms_masks(masks, iou_threshold=0.7):
    # Sort masks by confidence (IoU prediction)
    masks = sorted(masks, key=lambda m: m['predicted_iou'], reverse=True)

    keep = []
    while masks:
        # Keep highest confidence mask
        best = masks.pop(0)
        keep.append(best)

        # Remove overlapping masks
        masks = [m for m in masks if iou(m['mask'], best['mask']) < iou_threshold]

    return keep
```

**Result**: ~100-300 final masks per image (depends on scene complexity)

### 4. Post-Processing Filters

**Quality Filters**:
- **Minimum area**: Remove tiny masks (<100 pixels)
- **Edge touching**: Discard masks at image borders (likely cropped objects)
- **Stability score**: Keep only stable predictions (consistent across crop variations)

**Stability Score Calculation**:
```python
# Predict mask at slightly different crops
mask_original = sam.predict(point)
mask_perturbed = sam.predict(point + small_shift)

# Stability = IoU between original and perturbed predictions
stability = iou(mask_original, mask_perturbed)

# Keep only stable masks (stability > 0.9)
```

---

## Mask Metadata

**Each mask includes**:
- `segmentation`: Binary mask (H × W)
- `area`: Pixel count
- `bbox`: [x, y, width, height]
- `predicted_iou`: Confidence score (0-1)
- `stability_score`: Robustness metric (0-1)
- `crop_box`: Crop region used for generation

**Example**:
```json
{
  "segmentation": [[0, 0, 1, 1, ...]],
  "area": 15234,
  "bbox": [120, 50, 300, 250],
  "predicted_iou": 0.94,
  "stability_score": 0.87,
  "crop_box": [0, 0, 1024, 1024]
}
```

---

## Implementation Variants

### 1. Single-Scale Generation (Fastest)

**Method**: One grid at native image resolution

**Pros**:
- Fast inference (~5s per image on GPU)
- Good for large objects

**Cons**:
- Misses small objects (<10 pixels)

### 2. Multi-Scale Generation (Most Accurate)

**Method**: Generate masks at multiple resolutions

**Example**:
```python
scales = [0.5, 1.0, 1.5]  # Zoom out, native, zoom in
all_masks = []

for scale in scales:
    resized_image = resize(image, scale)
    masks = sam_auto_mask_generator(resized_image)
    all_masks.extend(masks)

# Combine and NMS across scales
final_masks = nms_masks(all_masks)
```

**Pros**:
- Captures objects at all scales
- Best recall

**Cons**:
- 3× slower (15s per image)

### 3. Crop-Based Generation (For Huge Images)

**Method**: Divide image into overlapping crops, generate masks per crop

**Use Case**: Gigapixel images, satellite imagery

**Algorithm**:
```python
crops = create_overlapping_crops(image, crop_size=1024, overlap=128)

all_masks = []
for crop, offset in crops:
    masks = sam_auto_mask_generator(crop)
    # Adjust mask coordinates to global image space
    for mask in masks:
        mask['bbox'] = translate_bbox(mask['bbox'], offset)
    all_masks.extend(masks)

# NMS across crops
final_masks = nms_masks(all_masks, iou_threshold=0.8)
```

**Pros**:
- Handles unlimited image sizes
- Memory efficient (process one crop at a time)

---

## Performance Benchmarks

### Inference Speed (Single Image, 1024×1024, A100 GPU)

| Method | Grid Density | Masks Generated | NMS Masks | Time |
|--------|--------------|-----------------|-----------|------|
| Sparse | 16×16 (256 points) | 768 | ~80 | 2.1s |
| Medium | 32×32 (1,024 points) | 3,072 | ~150 | 5.3s |
| Dense | 64×64 (4,096 points) | 12,288 | ~250 | 18.7s |

**Insight**: Dense grids provide better recall but 9× slower than sparse grids.

### Recall Analysis (COCO Validation)

| Grid Stride | Recall@50 (IoU>0.5) | Recall@75 (IoU>0.75) |
|-------------|---------------------|----------------------|
| 64 (sparse) | 76.3% | 58.1% |
| 32 (medium) | 89.2% | 71.4% |
| 16 (dense) | 94.7% | 78.3% |

**Trade-off**: 2× denser grid → +5% recall, +4× inference time.

---

## Comparison: Automatic vs. Prompted Segmentation

### Automatic Mask Generation
- **Input**: Image only (no prompts)
- **Output**: All objects (100-300 masks)
- **Use Case**: Exploration, dataset generation, dense annotation
- **Speed**: Slow (5-18s per image)

### Prompted Segmentation
- **Input**: Image + point/box/mask prompt
- **Output**: Single target object (1-3 masks)
- **Use Case**: Interactive annotation, known target
- **Speed**: Fast (0.05s per prompt)

**Key Difference**: Automatic = exhaustive search, Prompted = targeted query.

---

## Practical Applications

### 1. Instance Segmentation Pipeline
- **Step 1**: SAM automatic mask generation
- **Step 2**: CLIP classification (assign labels to masks)
- **Step 3**: Filter by confidence threshold
- **Result**: Open-vocabulary instance segmentation

### 2. Image Editing Workflows
- **Adobe Photoshop**: "Select all objects" → SAM automatic masks
- **GIMP**: Layer extraction (each mask = separate layer)
- **Benefit**: 10× faster than manual selection

### 3. Dataset Annotation
- **Use Case**: Generate pseudo-labels for semi-supervised learning
- **Workflow**: SAM auto-generate → Human review → Export annotations
- **Speed**: 80% reduction in annotation time

### 4. Image Retrieval
- **Method**: Segment objects → Extract features → Index for search
- **Example**: "Find images with cars" → Match car masks
- **Benefit**: Object-level retrieval (not just image-level)

### 5. Robotics Scene Understanding
- **Application**: Segment all objects in robot's view
- **Challenge**: Real-time requirement (30 FPS)
- **Solution**: Sparse grid (16×16) + GPU optimization → 15 FPS

---

## Advanced Techniques

### 1. Adaptive Grid Density

**Idea**: Dense grid for complex regions, sparse for simple regions.

**Algorithm**:
```python
# Generate masks with sparse grid
masks_sparse = sam_auto_mask_generator(image, grid_stride=64)

# Identify complex regions (many overlapping masks)
complex_regions = find_dense_areas(masks_sparse)

# Re-generate with dense grid in complex regions only
masks_dense = sam_auto_mask_generator(complex_regions, grid_stride=16)

# Combine
final_masks = nms_masks(masks_sparse + masks_dense)
```

**Benefit**: 2× faster than uniform dense grid, same recall.

### 2. Mask Hierarchy Construction

**Goal**: Organize masks into part-whole relationships.

**Method**:
- Sort masks by area (largest first)
- Build hierarchy: Large masks contain smaller masks
- Result: Tree structure (image → objects → parts)

**Use Case**: Compositional scene understanding, fine-grained annotation.

### 3. Multi-Modal Fusion

**Combine**: SAM masks + depth maps + semantic labels

**Example**:
- SAM segments object boundaries
- Depth map provides 3D shape
- CLIP assigns semantic labels

**Result**: 3D object-level scene graph.

---

## Limitations and Failure Cases

### 1. Over-Segmentation
- **Problem**: Dense grid generates too many redundant masks
- **Cause**: NMS threshold too low (allows overlapping masks)
- **Solution**: Increase IoU threshold (0.7 → 0.85)

### 2. Under-Segmentation
- **Problem**: Misses small objects
- **Cause**: Sparse grid, stride too large
- **Solution**: Multi-scale generation, denser grid

### 3. Boundary Precision
- **Problem**: Masks may not align perfectly with object edges
- **Cause**: ViT-H downsampling (32× stride)
- **Solution**: Post-processing refinement (GrabCut, CRF)

### 4. Computational Cost
- **Problem**: 10-20s per image on A100 GPU
- **Barrier**: Real-time applications (robotics, video)
- **Solution**: SAM-Fast (distilled model), TensorRT optimization

---

## ARR-COC Integration (5%)

### Automatic Mask Generation as Relevance Discovery

**Connection**: Automatic mask generation = **unsupervised relevance realization**.

**ARR-COC Mapping**:
1. **Grid sampling** = exhaustive salience landscape exploration
2. **Multi-mask output** = perspectival ambiguity (multiple valid interpretations)
3. **Stability score** = relevance robustness (consistent across perturbations)

**Insight**: SAM discovers relevance structures (objects) without prior knowledge → parallels ARR-COC's autonomous relevance realization!

### Participatory Knowing in Mask Generation

**Interactive refinement**:
- Automatic masks = initial relevance hypothesis
- User feedback = participatory adjustment
- Final masks = co-created relevance landscape

**Example**: User removes false positives → SAM learns what's NOT relevant → improves future generations.

---

## Key Takeaways

1. **Automatic Mask Generation**: Grid-based dense segmentation without prompts
2. **Pipeline**: Grid sampling → Per-point prediction → NMS → Filtering
3. **Trade-offs**: Dense grids (better recall) vs. sparse grids (faster inference)
4. **Applications**: Dataset annotation, image editing, instance segmentation, robotics
5. **ARR-COC**: Automatic generation = unsupervised relevance discovery

**Next**: PART 31 - ViT-H Image Encoder Architecture

---

**References**:
- Kirillov et al., "Segment Anything" (ICCV 2023) - Section 4.3
- SAM GitHub: `segment_anything/automatic_mask_generator.py`
- Automatic mask generation API documentation
