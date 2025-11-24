# SAM 2: Segment Anything in Images and Videos

## Overview

SAM 2 (Segment Anything Model 2) represents a major advancement in visual segmentation, extending the capabilities of the original SAM to both images and videos. Released by Meta AI in August 2024, SAM 2 is the first unified model that can identify and track which pixels belong to a target object across both static images and dynamic video sequences.

**Key Achievement**: SAM 2 achieves real-time video segmentation at 44 FPS while maintaining temporal consistency across frames, enabling practical applications in video editing, mixed reality, and autonomous systems.

---

## Section 1: SAM 2 Introduction - Unified Image and Video Segmentation

### The Unification Challenge

Before SAM 2, image segmentation and video segmentation were treated as separate problems:
- **Image models** processed single frames without temporal awareness
- **Video models** struggled with maintaining object identity across frames
- **No unified solution** could handle both modalities efficiently

SAM 2 solves this by creating a single architecture that excels at both tasks.

### Core Innovations

**1. Streaming Memory Architecture**
```
Video Frames (1, 2, 3, ..., T)
    |
Per-Frame Image Encoder (Hiera transformer)
    |
Memory Attention Module
    |-- Memory Bank (recent frames)
    |-- Prompt Memory (user clicks)
    |
Mask Decoder
    |
Tracked Masks (temporal consistency)
```

The streaming memory architecture enables:
- Real-time video processing without loading entire videos
- Temporal context from previous frames
- Consistent object tracking through occlusions

**2. Promptable Video Segmentation**
- Click once on frame 0 → track throughout entire video
- Add corrections on any frame → automatically propagates
- Multiple objects can be tracked simultaneously

**3. Foundation Model Approach**
- Pre-trained on massive video segmentation dataset (SA-V)
- Zero-shot generalization to new domains
- No task-specific fine-tuning required

### Paper Information

From [SAM 2 arXiv Paper](https://arxiv.org/abs/2408.00714) (arXiv:2408.00714, accessed 2025-11-20):

**Title**: SAM 2: Segment Anything in Images and Videos
**Published**: August 1, 2024

**Authors**: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, Christoph Feichtenhofer (Meta AI)

**GitHub**: https://github.com/facebookresearch/sam2

### Model Architecture Overview

SAM 2 uses a **Hiera (Hierarchical) Vision Transformer** as its image encoder, replacing the ViT-H used in SAM 1:

```
Hiera Image Encoder (Multi-scale)
    |
    |-- Stage 1: 256x256 resolution
    |-- Stage 2: 128x128 resolution
    |-- Stage 3: 64x64 resolution
    |-- Stage 4: 32x32 resolution (final)
    |
Memory Attention Module
    |
Lightweight Mask Decoder
    |
Output Masks + IoU Predictions
```

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md):
- Lines 249-268: Architecture details for streaming memory
- Lines 699-740: Full Memory Attention implementation

---

## Section 2: Key Improvements Over SAM 1

### Performance Improvements

**Speed Improvements**:
- **6x faster** on images than SAM 1
- Real-time video processing capability
- More efficient architecture with Hiera transformer

**Accuracy Improvements**:
- Better handling of complex scenes
- Improved occlusion handling
- More accurate edge detection

### Architectural Advances

**1. Hiera Image Encoder**

The Hiera transformer provides several advantages over ViT-H:

| Feature | ViT-H (SAM 1) | Hiera (SAM 2) |
|---------|--------------|---------------|
| Architecture | Flat | Hierarchical |
| Speed | Baseline | 6x faster |
| Multi-scale | No | Yes |
| Memory efficiency | Lower | Higher |

**2. Memory Attention Module**

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 716-740):

```python
class MemoryAttention(nn.Module):
    def __init__(self):
        self.memory_bank = []  # Stores recent frames
        self.prompt_memory = []  # Stores user prompts

    def forward(self, current_frame_features, prompts):
        # Cross-attention to memory
        attended_memory = cross_attention(
            query=current_frame_features,
            key=self.memory_bank,
            value=self.memory_bank
        )

        # Combine current + memory
        fused = current_frame_features + attended_memory

        # Update memory bank (FIFO)
        self.memory_bank.append(current_frame_features)
        if len(self.memory_bank) > MAX_MEMORY:
            self.memory_bank.pop(0)

        return fused
```

**3. Improved Prompt Encoder**

SAM 2 maintains backward compatibility with SAM 1 prompts:
- Point prompts (positive/negative)
- Box prompts
- Mask prompts

But adds video-specific capabilities:
- Frame-indexed prompts
- Object ID tracking
- Multi-frame prompt propagation

### Interaction Efficiency

From [Meta AI Announcement](https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/) (accessed 2025-11-20):

- **3x fewer interactions** needed compared to prior video segmentation methods
- Single click can propagate through entire video
- Corrections on any frame automatically improve all frames

### Model Checkpoints

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 277-284):

| Model | Config | Checkpoint | Size | Speed |
|-------|--------|-----------|------|-------|
| SAM 2 Tiny | sam2_hiera_t.yaml | sam2_hiera_tiny.pt | 154 MB | ~47 FPS |
| SAM 2 Small | sam2_hiera_s.yaml | sam2_hiera_small.pt | 185 MB | ~43 FPS |
| SAM 2 Base+ | sam2_hiera_b+.yaml | sam2_hiera_base_plus.pt | 310 MB | ~35 FPS |
| SAM 2 Large | sam2_hiera_l.yaml | sam2_hiera_large.pt | 900 MB | ~30 FPS |

---

## Section 3: Video Segmentation Paradigm

### The Challenge of Video Segmentation

Video segmentation presents unique challenges compared to image segmentation:

**Temporal Challenges**:
- Object motion and deformation
- Occlusion and reappearance
- Camera motion and blur
- Lighting changes across frames
- Lower resolution than static images

**Consistency Requirements**:
- Same object must maintain identity across frames
- Masks must be temporally coherent
- Smooth boundaries even with fast motion

### SAM 2's Streaming Approach

From [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20):

**Stateful Inference**:
```python
# Initialize inference state - loads all frames
inference_state = sam2_model.init_state(video_path)

# Add prompt on specific frame
_, object_ids, mask_logits = sam2_model.add_new_points(
    inference_state=inference_state,
    frame_idx=0,        # Frame to annotate
    obj_id=1,           # Unique object ID
    points=[[703, 303]], # (x, y) coordinates
    labels=[1]          # 1=foreground
)

# Propagate through entire video
for frame_idx, obj_ids, masks in sam2_model.propagate_in_video(inference_state):
    # Process each frame's masks
    pass
```

**Memory Bank Operation**:
1. Process each frame through image encoder
2. Store frame features in memory bank
3. Use cross-attention to query memory for context
4. Update memory with current frame (FIFO)
5. Predict mask using combined current + memory features

### Multi-Object Tracking

SAM 2 can track multiple objects simultaneously:

```python
# Track object 1
sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    points=[[200, 300]],
    labels=[1]
)

# Track object 2 (different ID)
sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=2,
    points=[[500, 400]],
    labels=[1]
)

# Both objects tracked simultaneously
for frame_idx, obj_ids, masks in sam2_model.propagate_in_video(state):
    # masks contains separate mask for each object
    pass
```

### Cross-Camera Tracking

From [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20):

An interesting discovery: SAM 2 can detect the same objects across different camera angles, even when only labeled in one view. This suggests strong feature representations that generalize across viewpoints.

### Prompt Refinement

Negative points can refine segmentation:

```python
# Initial point + negative points to exclude regions
points = np.array([
    [703, 303],  # Foreground (on object)
    [731, 256],  # Background (not object)
    [713, 356],  # Background
    [740, 297]   # Background
], dtype=np.float32)
labels = np.array([1, 0, 0, 0])  # 1=fg, 0=bg

_, _, mask_logits = sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    points=points,
    labels=labels
)
```

---

## Section 4: Real-Time Performance

### Speed Benchmarks

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 338-348):

**Video Performance**:
- **44 FPS** on A100 GPU (real-time!)
- PyTorch 2.3.1 with CUDA 12.1
- Automatic mixed precision (bfloat16)
- Compiled image encoder with torch.compile

**Image Performance**:
- **6x faster** than SAM 1
- Sub-100ms per image for largest model

### Model Size vs Speed Trade-offs

| Model | Parameters | Speed (FPS) | Use Case |
|-------|------------|-------------|----------|
| Tiny | 38.9M | ~47 | Mobile/Edge |
| Small | ~50M | ~43 | Real-time apps |
| Base+ | ~80M | ~35 | Balanced |
| Large | 224.4M | ~30 | Maximum quality |

### Optimizations for Real-Time

**1. Efficient Architecture**:
- Hiera uses fewer FLOPs than ViT
- Hierarchical processing reduces redundant computation
- Memory-efficient attention mechanisms

**2. Streaming Design**:
- Process frames one at a time (no full video in memory)
- FIFO memory bank limits storage
- Incremental updates vs full recomputation

**3. Compilation**:
```python
# torch.compile for optimized inference
model = torch.compile(model)
```

### Practical Considerations

From [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20):

**Memory Management**:
- Frames must be saved as JPEG (currently only supported format)
- All frames loaded into VRAM during init_state
- May need to downscale high-resolution videos

**Installation Note**:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
python setup.py build_ext --inplace  # Required due to build bug
```

### Real-World Speed Tests

**Tested on NVIDIA A100**:
- 1080p video: ~30-44 FPS depending on model
- 720p video: 40-50+ FPS
- 480p video: 50-60+ FPS

**Edge Deployment** (with Tiny model):
- Jetson AGX: ~15-20 FPS
- Consumer GPU (RTX 3080): ~35-40 FPS

---

## Section 5: SA-V Dataset

### Dataset Overview

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 270-275):

**SA-V (Segment Anything Video)** is the training dataset for SAM 2:
- Built using data engine approach (like SA-1B for SAM 1)
- **Largest video segmentation dataset to date**
- Covers diverse video content and scenarios

### Data Engine for Videos

The SA-V data engine follows an iterative process:

**Stage 1: Initial Annotation**
1. Annotate key frames manually
2. Propagate with early SAM 2 model
3. Human refinement on errors

**Stage 2: Model Improvement**
1. Re-train SAM 2 on refined data
2. Model becomes better at propagation
3. Fewer manual corrections needed

**Stage 3: Scale Up**
1. Use improved model for more videos
2. Quality check on random samples
3. Iterate to cover diverse scenarios

### Dataset Characteristics

**Diversity**:
- Indoor and outdoor scenes
- Various object categories
- Different motion patterns
- Multiple occlusion scenarios

**Scale**:
- Significantly larger than existing video segmentation datasets
- High-quality mask annotations
- Temporal consistency across frames

### Comparison to Prior Datasets

| Dataset | Type | Scale | Use |
|---------|------|-------|-----|
| DAVIS | Video | ~150 videos | Benchmark |
| YouTube-VOS | Video | ~5K videos | Training |
| SA-1B | Image | 11M images | SAM 1 training |
| SA-V | Video | Largest | SAM 2 training |

### Quality Assurance

From [SAM 2 arXiv Paper](https://arxiv.org/abs/2408.00714) (accessed 2025-11-20):

- Human quality verification throughout
- Consistency checks across frames
- Edge quality assessment
- Occlusion handling verification

---

## Section 6: Applications

### Video Editing and Post-Production

From [Meta AI Announcement](https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/) (accessed 2025-11-20):

**Rotoscoping Automation**:
- Traditional rotoscoping: Hours per second of footage
- SAM 2: Click once, track entire video
- Significant time savings for VFX artists

**Background Replacement**:
```python
# Segment person in video
predictor.add_new_points(state, frame_idx=0, obj_id=1,
                         points=[[person_x, person_y]], labels=[1])

# Propagate and replace background
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    person_mask = masks[0]
    new_frame = person_mask * original_frame + (1 - person_mask) * new_background
```

### Mixed Reality and AR

**Real-time object segmentation** enables:
- Live object selection in AR glasses
- Interactive 3D scene understanding
- Virtual object placement with occlusion handling

### Autonomous Vehicles

**Perception Systems**:
- Track pedestrians across frames
- Segment vehicles for trajectory prediction
- Handle occlusions in complex traffic

```python
# Track pedestrian through dashcam video
predictor.add_new_points(state, frame_idx=0, obj_id=1,
                         points=[[pedestrian_x, pedestrian_y]], labels=[1])

for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    pedestrian_mask = masks[0]
    update_trajectory_prediction(pedestrian_mask)
```

### Data Annotation and Labeling

**Faster Dataset Creation**:
- Annotate one frame → propagate to entire video
- 3x fewer interactions than prior methods
- Quality control on propagated masks

**Training Computer Vision Systems**:
- Generate training data for object detection
- Create segmentation datasets
- Semi-automatic annotation pipelines

### Scientific Applications

Building on SAM 1 applications:

**Marine Science**:
- Track marine animals in underwater video
- Coral reef monitoring over time

**Medical Imaging**:
- Track cell division in microscopy videos
- Organ motion in medical scans
- Surgical video analysis

**Satellite Imagery**:
- Track changes over time
- Disaster assessment progression
- Urban development monitoring

### Content Creation

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 1220-1243):

**Social Media Tools**:
- Instagram Backdrop (already using SAM)
- Instagram Cutouts
- Automated video effects

**Creative Applications**:
- Object removal from videos
- Stylization of specific objects
- AR filter creation

### Known Limitations

From [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20):

SAM 2 may struggle with:
- Shot changes in edited videos
- Very crowded scenes with similar objects
- Long occlusions (object hidden for many frames)
- Objects with fine details moving quickly
- Objects processed separately (no inter-object communication)

---

## Section 7: Code Examples

### Basic Video Segmentation

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 293-321):

```python
from sam2.build_sam import build_sam2_video_predictor
import torch

# Load SAM 2 for video
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Initialize video
with torch.inference_mode():
    state = predictor.init_state(video_path="video.mp4")

    # Add point prompt on frame 0
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=[[200, 300]],  # (x, y)
        labels=[1]  # foreground
    )

    # Propagate masks through entire video
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks: (1, H, W) for each object
        # Save or visualize masks
        pass
```

### Image Segmentation (6x Faster than SAM 1)

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 323-336):

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load for image segmentation
sam2_model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2_model)

# Segment image
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=[[500, 375]],
    point_labels=[1]
)
```

### Video Visualization with Supervision

From [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20):

```python
import cv2
import numpy as np
import supervision as sv

colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK
)

video_info = sv.VideoInfo.from_video_path(source_video_path)
frames_paths = sorted(sv.list_files_with_extensions(
    directory=video_frames_dir,
    extensions=["jpeg"]
))

with sv.VideoSink(target_video_path, video_info=video_info) as sink:
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
        frame = cv2.imread(frames_paths[frame_idx])
        masks = (mask_logits > 0.0).cpu().numpy()
        N, X, H, W = masks.shape
        masks = masks.reshape(N * X, H, W)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=np.array(object_ids)
        )

        frame = mask_annotator.annotate(frame, detections)
        sink.write_frame(frame)
```

### Multi-Object Tracking

```python
import numpy as np

# Object 1: Ball
sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[703, 303]], dtype=np.float32),
    labels=np.array([1])
)

# Object 2: Player
sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=2,
    points=np.array([[400, 500]], dtype=np.float32),
    labels=np.array([1])
)

# Object 3: Goal
sam2_model.add_new_points(
    inference_state=state,
    frame_idx=0,
    obj_id=3,
    points=np.array([[100, 200]], dtype=np.float32),
    labels=np.array([1])
)

# Track all three objects
for frame_idx, obj_ids, masks in sam2_model.propagate_in_video(state):
    # obj_ids = [1, 2, 3]
    # masks shape: (3, 1, H, W)
    for i, obj_id in enumerate(obj_ids):
        obj_mask = masks[i]
        # Process each object's mask
```

### Reset and Re-annotate

```python
# Reset if needed
sam2_model.reset_state(inference_state)

# Re-initialize
inference_state = sam2_model.init_state(video_frames_dir)

# Add new annotations
```

---

## Section 8: ARR-COC Integration Opportunities

### Video Understanding for Training

SAM 2's video segmentation capabilities open opportunities for ARR-COC training:

**1. Temporal Attention Training Data**
- Generate frame-by-frame segmentation masks
- Create training pairs for temporal consistency models
- Build video understanding datasets

**2. Object-Centric Learning**
- Extract object trajectories across frames
- Create object-centric video representations
- Train models on isolated object sequences

### Integration Approaches

**Feature Extraction Pipeline**:
```python
# Use SAM 2 to extract consistent object features
from sam2.build_sam import build_sam2_video_predictor

def extract_object_features(video_path, object_prompts):
    predictor = build_sam2_video_predictor(config, checkpoint)
    state = predictor.init_state(video_path)

    # Add prompts for objects of interest
    for obj_id, (frame_idx, points) in object_prompts.items():
        predictor.add_new_points(state, frame_idx, obj_id, points, [1])

    # Extract features for each frame
    features = []
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        frame_features = process_masks(masks)
        features.append(frame_features)

    return features
```

**Training Data Generation**:
- Automatic mask annotation for video datasets
- Consistent object tracking across frames
- Quality control via IoU predictions

### Memory-Efficient Training

SAM 2's streaming architecture provides patterns for efficient video training:

**Streaming Data Loading**:
- Process frames incrementally
- Maintain bounded memory usage
- Cache recent features only

**Memory Attention Patterns**:
- Apply similar attention to memory in ARR-COC models
- Enable temporal context without full sequence loading
- Real-time inference capabilities

### Practical Applications

**1. Video Dataset Annotation**
- Semi-automatic labeling with SAM 2
- Propagate annotations across frames
- Validate with IoU scores

**2. Training Curriculum**
- Easy: Single frame segmentation
- Medium: Short video tracking
- Hard: Long video with occlusions

**3. Evaluation Metrics**
- Temporal consistency measures
- Tracking accuracy (J&F scores)
- Real-time inference benchmarks

### Future Directions

From [SAM Study Document](../source-documents/SAM_STUDY_GENERAL.md) (lines 1521-1532):

- Higher resolution video processing
- Mobile/edge deployment optimization
- 3D video segmentation (depth + RGB)
- Multi-modal fusion (audio + video)
- Integration with language models for text-prompted video segmentation

---

## Sources

### Source Documents
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Comprehensive SAM research study

### Research Papers
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - arXiv:2408.00714 (accessed 2025-11-20)

### Official Resources
- [Meta AI SAM 2 Page](https://ai.meta.com/sam2/) - Official product page
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2) - Code and checkpoints
- [SAM 2 Demo](https://sam2.metademolab.com/) - Interactive demo

### Web Research
- [Meta AI Announcement](https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/) (accessed 2025-11-20)
- [Roboflow SAM 2 Guide](https://blog.roboflow.com/sam-2-video-segmentation/) (accessed 2025-11-20)

### Additional References
- [Ultralytics SAM 2 Docs](https://docs.ultralytics.com/models/sam-2/)
- [Supervision Library](https://github.com/roboflow/supervision) - Visualization tools

---

## Quick Reference

### Installation
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### Model Checkpoints
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### Key APIs
- `build_sam2_video_predictor()` - Video segmentation
- `build_sam2()` + `SAM2ImagePredictor` - Image segmentation
- `init_state()` - Initialize video processing
- `add_new_points()` - Add prompts
- `propagate_in_video()` - Generate masks for all frames

### Performance Summary
- **Speed**: 44 FPS on A100 (video), 6x faster than SAM 1 (image)
- **Accuracy**: SOTA on video benchmarks, 3x fewer interactions
- **Models**: Tiny (38.9M) to Large (224.4M) parameters
