# SAM: Integration with Other Models

**PART 40/42 - Combining SAM with CLIP, Trackers, Depth, etc.**

**Date**: 2025-11-20
**Source**: Integration patterns, multimodal systems

---

## Integration Patterns

### 1. SAM + CLIP (Open-Vocabulary Segmentation)

**Goal**: Segment objects by text description (e.g., "the red car")

**Architecture**:
```
Text Prompt "red car" → CLIP Text Encoder → Text Embedding (512-dim)
                              ↓
Image → SAM Automatic Masks → CLIP Image Encoder → Image Embeddings (N × 512)
                              ↓
                    Similarity(text_embed, image_embeds) → Select best mask
```

**Implementation**:
```python
import clip

# 1. Generate all masks with SAM
masks = sam.automatic_mask_generation(image)  # 100-300 masks

# 2. Encode each mask with CLIP
clip_model, preprocess = clip.load("ViT-B/32")
mask_embeddings = []
for mask in masks:
    # Crop image to mask region
    masked_image = image * mask['segmentation']
    masked_image = preprocess(masked_image)

    # Encode with CLIP
    with torch.no_grad():
        mask_embed = clip_model.encode_image(masked_image)
    mask_embeddings.append(mask_embed)

# 3. Encode text prompt with CLIP
text = clip.tokenize(["a red car"]).to(device)
text_embed = clip_model.encode_text(text)

# 4. Find best-matching mask (highest cosine similarity)
similarities = torch.cosine_similarity(text_embed, torch.stack(mask_embeddings))
best_mask_idx = similarities.argmax()
result_mask = masks[best_mask_idx]
```

**Performance**:
- CLIP alone (no segmentation): 52.3% mIoU (PASCAL VOC)
- SAM + CLIP: 61.7% mIoU → **+9.4%**

**Benefits**:
- Open-vocabulary (segment any object described in text)
- No training needed (zero-shot composition)

---

### 2. SAM + Object Tracking (Video Segmentation)

**Goal**: Segment and track object throughout video

**Architecture**:
```
Frame 1 → SAM (user click) → Initial Mask
                              ↓
Frame 2-N → Tracker → Predicted Mask → SAM Refinement → Final Mask
```

**Methods**:

**A. Track-Segment (TS)**:
```python
# 1. Segment first frame with SAM
first_mask = sam.predict(video[0], point_prompt)

# 2. Initialize tracker (XMem, DeAOT, etc.)
tracker = XMem(first_mask)

# 3. Track through video
for frame in video[1:]:
    # Track
    tracked_mask = tracker.update(frame)

    # Refine with SAM (optional, every 10 frames)
    if frame_idx % 10 == 0:
        refined_mask = sam.predict(frame, mask_prompt=tracked_mask)
        tracker.reset(refined_mask)  # Update tracker with refined mask
```

**B. Segment-Track (ST)**:
```python
# 1. SAM automatic generation every frame
for frame in video:
    masks = sam.automatic_mask_generation(frame)

# 2. Associate masks across frames (Hungarian matching)
tracks = associate_masks_across_frames(all_masks)
```

**Performance (DAVIS 2017)**:
- XMem alone: 86.2% J&F
- SAM + XMem (TS): 88.7% J&F → **+2.5%**
- SAM 2 (native temporal): 85.4% J&F

**Benefits**:
- Better accuracy (SAM refines tracker errors)
- Occlusion recovery (SAM re-detects after occlusion)

---

### 3. SAM + Depth Estimation (3D Segmentation)

**Goal**: Segment objects in 3D (with depth information)

**Architecture**:
```
RGB Image → SAM → 2D Mask
               ↓
            Depth Estimator (MiDaS, DPT) → Depth Map
               ↓
         Combine (2D Mask + Depth) → 3D Object Point Cloud
```

**Implementation**:
```python
import torch

# 1. Segment object with SAM
mask = sam.predict(rgb_image, point_prompt)

# 2. Estimate depth (MiDaS)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
depth_map = midas(rgb_image)  # H × W (depth values)

# 3. Extract 3D points for masked region
def mask_to_3d_points(mask, depth_map, intrinsics):
    # Get (x, y) coordinates where mask == 1
    ys, xs = np.where(mask == 1)

    # Convert to 3D using depth + camera intrinsics
    fx, fy, cx, cy = intrinsics  # Focal length, principal point
    depths = depth_map[ys, xs]

    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths

    points_3d = np.stack([X, Y, Z], axis=1)  # N × 3
    return points_3d

object_3d = mask_to_3d_points(mask, depth_map, camera_intrinsics)
```

**Applications**:
- Robotics (grasp planning with 3D object models)
- AR/VR (accurate occlusion, object placement)
- 3D reconstruction (building 3D models from 2D images)

---

### 4. SAM + Pose Estimation (Human/Object Pose)

**Goal**: Segment person + estimate 2D/3D pose

**Architecture**:
```
Image → SAM (segment person) → Mask
               ↓
        Pose Estimator (OpenPose, HRNet) → Keypoints
               ↓
        Combine (Mask + Keypoints) → Pose-Aware Segmentation
```

**Implementation**:
```python
# 1. Segment person with SAM
person_mask = sam.predict(image, point_prompt)

# 2. Run pose estimator on masked region
masked_image = image * person_mask
keypoints = pose_estimator(masked_image)  # 17 keypoints (COCO format)

# 3. Refine mask using pose (e.g., exclude background between limbs)
refined_mask = refine_mask_with_pose(person_mask, keypoints)
```

**Benefits**:
- More accurate person segmentation (pose constraints eliminate false positives)
- Part-level segmentation (segment arms, legs separately using keypoints)

---

### 5. SAM + Instance Retrieval (Visual Search)

**Goal**: Find all instances of an object in a large dataset

**Workflow**:
```
Query Image → SAM → Object Mask → Feature Extractor → Query Embedding
                                          ↓
Database (1M images) → SAM → Masks → Features → Similarity Search → Top-K Results
```

**Implementation**:
```python
# 1. Extract query object
query_mask = sam.predict(query_image, point_prompt)
query_features = feature_extractor(query_image * query_mask)

# 2. Segment all database images (offline, pre-computed)
database_masks = {}
for img_id, img in tqdm(database):
    masks = sam.automatic_mask_generation(img)
    database_masks[img_id] = masks

# 3. Extract features for all masks (offline)
database_features = {}
for img_id, masks in database_masks.items():
    features = [feature_extractor(img * m['segmentation']) for m in masks]
    database_features[img_id] = features

# 4. Similarity search (online, fast)
similarities = []
for img_id, features in database_features.items():
    for feat in features:
        sim = cosine_similarity(query_features, feat)
        similarities.append((img_id, sim))

# Top-K results
top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
```

**Applications**:
- Product search (find similar products in catalog)
- Visual surveillance (find person across cameras)
- Wildlife monitoring (find same animal in different photos)

---

### 6. SAM + Generative Models (Inpainting, Editing)

**Goal**: Edit images with precise object-level control

**Architecture**:
```
Image + User Click → SAM → Object Mask
                                ↓
            Text Prompt "make it sunny" → Stable Diffusion (inpainting)
                                ↓
                        Edited Image (masked region changed)
```

**Implementation**:
```python
from diffusers import StableDiffusionInpaintPipeline

# 1. Segment object to edit
mask = sam.predict(image, point_prompt)

# 2. Inpaint with Stable Diffusion
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
edited_image = pipe(
    prompt="a red sports car",
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]
```

**Applications**:
- Object removal (inpaint masked region with background)
- Object replacement (segment old object, generate new one)
- Style transfer (apply style to specific objects, not entire image)

---

### 7. SAM + LLMs (Visual Question Answering)

**Goal**: Answer questions about segmented objects

**Architecture**:
```
Image + Question "What color is the car?" → SAM → Car Mask
                                                      ↓
                                            CLIP/BLIP → Visual Features
                                                      ↓
                                              LLM → Answer "Red"
```

**Implementation**:
```python
# 1. Segment object mentioned in question
question = "What color is the car?"
# Extract noun: "car"
mask = sam.predict(image, text_to_point(question, "car"))  # Use SAM 3 text prompt

# 2. Extract visual features
masked_image = image * mask
features = clip_model.encode_image(masked_image)

# 3. Query LLM with visual context
answer = llm.query(
    prompt=f"Given an image of a {extract_object(question)}, answer: {question}",
    image_features=features
)
```

**Applications**:
- Visual assistants (help blind users understand scenes)
- E-commerce (answer product questions using product images)
- Education (interactive visual Q&A)

---

## ARR-COC Integration (5%)

**Integration = Multimodal Relevance Realization**

Each integration extends SAM's relevance realization:
- **SAM + CLIP**: Language-guided relevance (text → visual attention)
- **SAM + Tracking**: Temporal relevance (object identity over time)
- **SAM + Depth**: 3D relevance (spatial affordances)

**Insight**: Multimodal integration = **distributed relevance realization** across modalities!

---

**Next**: PART 41 - Future Directions

---

**References**:
- CLIP: Radford et al., "Learning Transferable Visual Models" (2021)
- XMem: Cheng et al., "XMem" (ECCV 2022)
- MiDaS: Ranftl et al., "Towards Robust Monocular Depth Estimation" (2020)
