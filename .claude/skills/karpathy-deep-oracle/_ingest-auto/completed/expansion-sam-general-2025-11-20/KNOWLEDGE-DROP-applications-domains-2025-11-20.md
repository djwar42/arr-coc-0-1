# SAM: Applications Across Domains

**PART 36/42 - Real-World Applications (Medical, Remote Sensing, Robotics, etc.)**

**Date**: 2025-11-20
**Source**: SAM applications survey, domain-specific papers

---

## Application Categories

### 1. Medical Imaging

**Use Cases**:
- Tumor segmentation (CT, MRI scans)
- Organ boundary detection
- Cell counting (microscopy)
- Surgical planning

**Example: MedSAM** (Fine-tuned SAM for medical imaging):
- Dataset: 1M medical images, 10 modalities (CT, MRI, ultrasound, X-ray, etc.)
- Performance: 90.06% Dice score (vs. 60% for general SAM)
- **Key Insight**: Fine-tuning on medical data bridges domain gap!

**Workflow**:
```python
# 1. Radiologist clicks tumor center
point_prompt = (x=234, y=156, label=foreground)

# 2. SAM predicts tumor boundary
mask = medsam.predict(ct_scan, point_prompt)

# 3. Radiologist refines with correction points
correction = (x=250, y=170, label=background)  # Exclude nearby vessel
refined_mask = medsam.predict(ct_scan, [point_prompt, correction])
```

**Benefits**:
- 10× faster annotation vs. manual tracing
- Consistent boundaries (reduces inter-annotator variability)
- Works across modalities (CT, MRI, ultrasound) without retraining

### 2. Remote Sensing & Satellite Imagery

**Use Cases**:
- Building footprint extraction
- Road network detection
- Crop field segmentation
- Disaster damage assessment

**Example: SAM-GEO** (Fine-tuned for geospatial data):
- Dataset: SpaceNet, xBD (building detection, 100K satellite images)
- Performance: 78.3% IoU (vs. 68.9% for general SAM)

**Workflow**:
```python
# 1. Draw bounding box around building cluster
box_prompt = [x_min=450, y_min=320, x_max=580, y_max=440]

# 2. SAM segments individual buildings
masks = sam_geo.automatic_mask_generation(satellite_image, box=box_prompt)

# 3. Export to GIS format (GeoJSON)
export_geojson(masks, coordinate_system="EPSG:4326")
```

**Benefits**:
- Scalable (millions of km² satellite data)
- Multi-resolution (handles Landsat, Sentinel, high-res commercial imagery)
- Temporal analysis (compare building footprints over years)

### 3. Robotics & Autonomous Systems

**Use Cases**:
- Grasp point selection (robot manipulation)
- Obstacle segmentation (navigation)
- Scene understanding (object affordances)
- Human-robot collaboration (shared workspace)

**Example: SAM-6D** (6D pose estimation + segmentation):
- Task: Segment object → Estimate 3D pose → Plan grasp
- Performance: 89.2% grasp success rate (vs. 73% without SAM)

**Workflow**:
```python
# 1. Robot camera captures scene
rgbd_image = robot.get_camera_view()

# 2. SAM segments objects (automatic mask generation)
object_masks = sam.automatic_mask_generation(rgbd_image)

# 3. Select graspable objects (filter by size, stability)
graspable = [m for m in object_masks if 100 < m['area'] < 5000]

# 4. Estimate 6D pose for grasping
for mask in graspable:
    pose = estimate_6d_pose(rgbd_image, mask)
    grasp_point = compute_grasp(pose, robot.gripper)
```

**Benefits**:
- Real-time (28ms with TensorRT on Jetson)
- Generalizes to novel objects (zero-shot)
- Robustness to clutter/occlusions

### 4. Content Creation & Image Editing

**Use Cases**:
- Object removal (inpainting)
- Background replacement
- Layer extraction (Photoshop-style)
- Virtual try-on (fashion, AR)

**Example: Adobe Photoshop SAM Integration**:
- Feature: "Select Object" tool (one-click object selection)
- Backend: SAM with point prompt at click location
- Performance: 0.3s per selection (vs. 5-10s manual lasso)

**Workflow**:
```python
# 1. User clicks on object (e.g., person in photo)
point_prompt = user_click_location()

# 2. SAM segments person
person_mask = sam.predict(image, point_prompt)

# 3. Replace background (inpainting)
new_background = load_image("beach.jpg")
composite = inpaint(image, person_mask, new_background)
```

**Benefits**:
- Intuitive (single click vs. manual tracing)
- Precise boundaries (hair, clothing edges)
- Fast iteration (try multiple backgrounds instantly)

### 5. Video Object Tracking

**Use Cases**:
- Sports analytics (player tracking)
- Surveillance (person re-identification)
- Wildlife monitoring (animal tracking)
- Video editing (rotoscoping)

**Example: SAM-Track** (SAM + tracking model):
- Method: SAM segments first frame → tracker propagates mask to subsequent frames
- Performance: 82.3% J&F (DAVIS benchmark) vs. 76.1% (SAM alone, no tracking)

**Workflow**:
```python
# 1. User clicks object in first frame
first_frame_mask = sam.predict(video[0], point_prompt)

# 2. Initialize tracker
tracker = DeAOT(first_frame_mask)  # Or SAM 2 memory module

# 3. Track through video
for frame in video[1:]:
    tracked_mask = tracker.update(frame)
    # Optional: Re-segment with SAM for refinement
    refined_mask = sam.predict(frame, mask_prompt=tracked_mask)
```

**Benefits**:
- Temporal consistency (smooth mask transitions)
- Occlusion handling (tracker + SAM re-detection)
- Interactive correction (user can fix mask at any frame)

### 6. Augmented Reality (AR)

**Use Cases**:
- Object occlusion (AR objects behind real objects)
- Surface detection (place virtual objects)
- Hand segmentation (gesture recognition)
- Virtual try-on (glasses, makeup)

**Example: SAM-AR** (Real-time AR segmentation):
- Hardware: iPhone 14 Pro (A16 Bionic Neural Engine)
- Model: SAM-Mobile (INT8 quantized, 48ms latency)
- Use Case: Virtual furniture placement (segment floor/walls)

**Workflow**:
```python
# 1. User points phone at room
camera_frame = arkit.get_camera_feed()

# 2. SAM segments surfaces (floor, walls, furniture)
surfaces = sam_mobile.automatic_mask_generation(camera_frame)

# 3. Filter for floor (largest horizontal surface)
floor_mask = [s for s in surfaces if s['area'] > 50000 and is_horizontal(s)][0]

# 4. Place virtual sofa on floor
virtual_sofa_pose = compute_placement(floor_mask, arkit.camera_pose)
arkit.place_object(virtual_sofa, virtual_sofa_pose)
```

**Benefits**:
- Accurate occlusion (AR objects correctly hidden by real furniture)
- Device-independent (runs on mobile, AR glasses)
- Real-time (60 FPS required, achieved with SAM-Mobile)

### 7. Document Analysis

**Use Cases**:
- Receipt scanning (item extraction)
- Form parsing (field segmentation)
- Handwriting recognition (word/line segmentation)
- Table extraction (cell boundaries)

**Example: SAM-Doc** (Document segmentation):
- Task: Segment text regions in scanned documents
- Dataset: PubLayNet (360K document images)
- Performance: 92.1% mIoU (vs. 85.3% for traditional CV methods)

**Workflow**:
```python
# 1. Scan document
document_image = scanner.get_image()

# 2. SAM automatic mask generation (segment all text blocks)
text_masks = sam_doc.automatic_mask_generation(document_image)

# 3. OCR each text region
for mask in text_masks:
    text_region = crop_image(document_image, mask['bbox'])
    text = ocr_engine.recognize(text_region)
    structured_data[mask['label']] = text  # e.g., "invoice_number": "INV-12345"
```

**Benefits**:
- Robust to layout variations (invoices, receipts, forms)
- Handles skew/rotation (segments at arbitrary angles)
- Multi-language (segmentation agnostic to language)

---

## Cross-Domain Patterns

**Common Success Factors**:
1. **Prompt Engineering**: Domain-specific prompt strategies (e.g., box prompts for buildings, points for tumors)
2. **Fine-Tuning**: Adapt to domain shift (MedSAM, SAM-GEO examples)
3. **Post-Processing**: Domain-specific refinement (morphological ops, CRF for sharp boundaries)
4. **Integration**: Combine with domain models (CLIP for labels, depth for 3D, tracking for video)

**Common Failure Modes**:
1. **Extreme Domain Shift**: Thermal, X-ray (no RGB color cues) → requires fine-tuning
2. **Fine-Grained Boundaries**: Adjacent objects without clear edges → needs iterative refinement
3. **Real-Time Constraints**: Robotics (30 FPS) → requires optimization (TensorRT, distillation)

---

## ARR-COC Integration (5%)

**Domain Applications = Contextualized Relevance Realization**

Each domain has unique relevance criteria:
- **Medical**: Tumor boundaries (propositional: abnormal tissue)
- **Remote Sensing**: Building footprints (perspectival: human-made structures)
- **Robotics**: Graspable objects (participatory: actionable affordances)

**SAM's Flexibility**: Promptable interface adapts to domain-specific relevance landscapes!

---

**Next**: PART 37 - Limitations & Failure Cases

---

**References**:
- MedSAM: Ma et al., "Segment Anything in Medical Images" (arXiv:2304.12306)
- SAM applications survey: Various domain-specific papers
