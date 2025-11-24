# SAM 2: Video Segmentation Architecture

**Meta AI 2024 | Segment Anything in Images AND Videos**

**Source**: arXiv 2024 (2,421 citations already), Meta AI Research
**Paper**: "SAM 2: Segment Anything in Images and Videos"
**Release**: July 2024
**Impact**: First unified model for real-time promptable segmentation in both images and videos

---

## What's New in SAM 2

### Major Breakthrough: Unified Image + Video Model

SAM 2 extends the original SAM to handle **video segmentation** while maintaining image capabilities:

**Original SAM (2023)**:
- Images only
- Single-frame processing
- No temporal understanding
- ViT-H encoder (slow)

**SAM 2 (2024)**:
- Images + Videos unified
- Multi-frame tracking
- Temporal consistency
- Hiera encoder (6× faster!)
- Real-time processing (44 FPS)

---

## Core Architecture Components

### 1. Hiera Image Encoder (NEW!)

**Key Innovation**: Replaced ViT-H with Hiera hierarchical vision transformer

**Why Hiera?**
- **6× faster** than ViT-H
- Hierarchical multi-scale features
- MAE pre-trained
- Simpler design (no "bells and whistles")
- Better for video (multi-scale temporal features)

**Performance**:
- ViT-H: ~0.5 FPS on video
- Hiera-L: ~44 FPS on video (88× faster!)

**Architecture**:
```
Input Frame (H×W×3)
    ↓
Hiera Hierarchical Encoder
    ↓
Multi-scale Features:
    - Stage 1: H/4 × W/4   (high-res, local details)
    - Stage 2: H/8 × W/8   (mid-level features)
    - Stage 3: H/16 × W/16 (semantic features)
    - Stage 4: H/32 × W/32 (global context)
    ↓
Encoded Image Features → Memory Bank
```

**Key Difference from ViT**:
- ViT: Single-scale, flat attention (all tokens attend to all)
- Hiera: Multi-scale, hierarchical (local → global)
- Result: Much faster, better for video

---

### 2. Streaming Memory Architecture (VIDEO!)

**The Big Idea**: Process video frames sequentially, remembering past frames

**Memory System Components**:

**A) Memory Bank**
- Stores features from previous frames
- Fixed capacity (default: 7 recent frames + 7 long-term)
- FIFO queue for recent frames
- Adaptive sampling for long-term memory

**B) Memory Encoder**
- Encodes current frame features into memory format
- Compresses spatial dimensions (reduce memory footprint)
- Adds temporal position encoding

**C) Memory Attention Module**
- Current frame attends to memory bank
- Retrieves relevant past information
- Cross-attention mechanism

**Architecture Flow**:
```
Frame t → Hiera Encoder → Current Features
                            ↓
                    Memory Attention
                            ↓
                    [Memory Bank: frames t-1, t-2, ..., t-N]
                            ↓
                    Attended Features (with temporal context)
                            ↓
                    Prompt Encoder + Mask Decoder
                            ↓
                    Segmentation Mask (frame t)
                            ↓
                    Add to Memory Bank → Process frame t+1
```

---

### 3. Memory Attention Mechanism

**Purpose**: Let current frame "see" past frames for temporal consistency

**How It Works**:

**Cross-Attention**:
- **Query**: Current frame features
- **Key/Value**: Memory bank features
- **Output**: Temporally-aware features

**Attention Weights**:
- Recent frames: Higher weight (more relevant)
- Distant frames: Lower weight (less relevant)
- Adaptive: Learns what to remember

**Code Concept** (simplified):
```python
# Current frame embedding
query = current_frame_features  # [H/16, W/16, C]

# Memory bank
keys = memory_features  # [N_frames, H/16, W/16, C]
values = memory_features

# Cross-attention
attention_weights = softmax(query @ keys.T / sqrt(d))
attended = attention_weights @ values

# Combine with current features
output = current_features + attended
```

**Benefits**:
- Temporal consistency (object doesn't "jump" between frames)
- Occlusion handling (remembers object when hidden)
- Identity preservation (same object gets same ID)

---

### 4. Temporal Consistency Mechanisms

**Challenge**: Keep object identity consistent across 100s-1000s of frames

**SAM 2 Solutions**:

**A) Memory-Based Tracking**
- Object features stored in memory
- Matched across frames
- No separate tracker needed!

**B) Streaming Processing**
- Process frame-by-frame (not batch)
- Low latency (real-time)
- Constant memory usage

**C) Mask Propagation**
- Use frame t mask to guide frame t+1
- Temporal smoothness
- Reduce jitter

**Performance**:
- 44 FPS on NVIDIA A100
- 480p video resolution
- Real-time for robotics/VR applications

---

## Streaming vs. Offline Processing

### SAM 2: Streaming Architecture

**Advantages**:
- Real-time processing (44 FPS)
- Constant memory (doesn't grow with video length)
- Works for infinite-length videos
- Low latency (~22ms per frame)

**How It Works**:
```
Frame 1 → Encode → Segment → Add to Memory
                                ↓
Frame 2 → Encode → Segment (using Frame 1) → Add to Memory
                                                ↓
Frame 3 → Encode → Segment (using Frames 1-2) → ...
```

### Traditional Offline Approaches

**Disadvantages**:
- Must buffer entire video
- Memory grows linearly with video length
- No real-time capability
- High latency (wait for all frames)

**Example**: Track-Anything, XMem, STCN
- Process all frames at once
- Expensive global optimization
- Not suitable for robotics/VR

---

## Real-Time Performance Benchmarks

### Inference Speed (NVIDIA A100)

| Model | Resolution | FPS | Latency |
|-------|-----------|-----|---------|
| SAM (ViT-H) | 1024×1024 | 0.5 | 2000ms |
| SAM 2 (Hiera-B+) | 480p video | 31 FPS | 32ms |
| SAM 2 (Hiera-L) | 480p video | 44 FPS | 23ms |
| SAM 2 (Hiera-L) | 720p video | 18 FPS | 55ms |

**Key Insight**: Hiera encoder + streaming architecture = **88× speedup** for video!

### Memory Footprint

| Model | Single Image | 100-frame Video |
|-------|-------------|-----------------|
| SAM | 4.2 GB | N/A (no video) |
| SAM 2 | 4.8 GB | 6.1 GB (constant!) |
| Traditional (XMem) | N/A | 18.5 GB (grows) |

**SAM 2 Advantage**: Constant memory regardless of video length!

---

## Architecture Improvements Over SAM 1

### Encoder: ViT-H → Hiera-L

**ViT-H (SAM 1)**:
- 632M parameters
- Single-scale features
- Global attention (expensive)
- Slow (0.5 FPS on video)

**Hiera-L (SAM 2)**:
- 212M parameters (3× smaller!)
- Multi-scale hierarchical
- Local + global attention
- Fast (44 FPS on video)

**Speedup**: 88× faster for video processing

### Memory System (NEW in SAM 2)

**Components**:
1. Memory Bank (stores past frames)
2. Memory Encoder (compresses features)
3. Memory Attention (cross-frame attention)

**Result**: Temporal consistency without separate tracker!

### Prompt Support

**SAM 1**: Point, box, mask prompts (single frame)
**SAM 2**: Same prompts + temporal propagation (across frames)

**Example**:
- Click object in frame 1
- SAM 2 tracks it through entire video
- No need to re-prompt each frame!

---

## Streaming Memory Implementation Details

### Memory Bank Structure

**Recent Memory** (FIFO queue):
- Stores last 7 frames
- High temporal resolution
- Captures recent motion

**Long-Term Memory** (adaptive sampling):
- Stores 7 older frames (sampled)
- Low temporal resolution
- Captures long-term context

**Total Memory**: 14 frames stored (constant size)

### Memory Update Strategy

**Every Frame**:
1. Encode current frame → features
2. Attend to memory bank → context
3. Decode mask → segmentation
4. Add current frame to memory
5. Evict oldest frame from recent memory

**Adaptive Sampling** (long-term):
- Keep frames where object changes significantly
- Skip frames with minimal motion
- Result: Efficient long-term memory

---

## Applications Enabled by SAM 2

### 1. Video Object Segmentation (VOS)

**Use Case**: Segment object in frame 1, track through video

**Performance**:
- DAVIS 2017 benchmark: 82.5 J&F (SOTA)
- YouTube-VOS: 81.2 J (SOTA)
- Real-time (44 FPS)

### 2. Interactive Video Editing

**Use Case**: Click object → automatic segmentation → edit/remove

**Examples**:
- Background removal (video)
- Object replacement (video)
- Rotoscoping (automatic!)

### 3. Robotics (Real-Time Object Tracking)

**Use Case**: Robot tracks objects in real-time

**Advantages**:
- 44 FPS (real-time for 30 FPS cameras)
- Low latency (23ms)
- Handles occlusions

### 4. Autonomous Driving

**Use Case**: Track pedestrians, vehicles, obstacles

**Advantages**:
- Multi-object tracking
- Temporal consistency
- Zero-shot (no fine-tuning)

### 5. Medical Imaging (Video Endoscopy)

**Use Case**: Track organs/lesions in video endoscopy

**Advantages**:
- Real-time feedback during surgery
- Consistent across frames
- Handles occlusions (organs moving)

---

## Training Data: SA-V Dataset

### What's New: Video Dataset!

**SA-1B (SAM 1)**: 1.1 billion masks, 11M images (static)
**SA-V (SAM 2)**: 50.9K videos, 642.6K masklets (video)

**Masklet**: Object segmentation tracked across multiple frames

### Video Characteristics

- **Length**: 10-90 seconds per video
- **Objects**: 1-10 objects per video
- **Frames**: 24-30 FPS
- **Domains**: Diverse (indoor, outdoor, drone, handheld, etc.)

### Data Engine (Video)

**3-Phase Process** (similar to SAM 1, but for video):

**Phase 1: Assisted Annotation** (manual)
- Annotators click object in frame 1
- SAM 2 propagates through video
- Annotators correct errors

**Phase 2: Semi-Automatic**
- SAM 2 proposes masks automatically
- Annotators verify/correct

**Phase 3: Fully Automatic**
- SAM 2 segments all objects in video
- No human intervention
- Quality control via automatic checks

**Result**: 642.6K high-quality video object tracks

---

## SAM 2 vs. SAM 1: Quick Comparison

| Feature | SAM 1 (2023) | SAM 2 (2024) |
|---------|-------------|-------------|
| **Media** | Images only | Images + Videos |
| **Encoder** | ViT-H (632M) | Hiera-L (212M) |
| **Speed** | 0.5 FPS (video) | 44 FPS (video) |
| **Memory** | N/A | Streaming (constant) |
| **Temporal** | No | Yes (memory attention) |
| **Dataset** | SA-1B (11M images) | SA-V (51K videos) |
| **Use Cases** | Image segmentation | Image + video tracking |

---

## Code Example: SAM 2 Video Inference

```python
from sam2.build_sam import build_sam2_video_predictor

# Initialize SAM 2
predictor = build_sam2_video_predictor(
    config_file="sam2_hiera_l.yaml",
    checkpoint="sam2_hiera_large.pt"
)

# Initialize inference state for video
inference_state = predictor.init_state(video_path="video.mp4")

# Prompt: Click object in frame 1
frame_idx = 0
point = (x, y)  # Click coordinates
label = 1  # Foreground

# Add prompt to frame 0
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=1,
    points=point,
    labels=label,
)

# Propagate through entire video
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# video_segments now contains masks for ALL frames!
```

**Key Point**: Click once in frame 1 → SAM 2 tracks through entire video!

---

## Limitations & Future Work

### Current Limitations

1. **Fast Motion**: Object moving very fast (> 50% frame-to-frame displacement)
2. **Extreme Occlusions**: Object hidden for >5 seconds
3. **Resolution**: Optimized for 480p-720p (1080p slower)
4. **Memory**: Fixed 14-frame memory (can't remember very long-term)

### Future Directions

1. **Multi-modal prompts**: Text + point + box combined
2. **3D understanding**: Depth-aware segmentation
3. **Longer memory**: Adaptive memory capacity
4. **Mobile deployment**: ONNX/TensorRT optimization

---

## ARR-COC Integration (10%)

### Relevance-Guided Video Segmentation

**Connection**: SAM 2 memory attention = relevance realization over time

**Propositional Knowing** (semantic labels):
- Track object category across frames
- "This is a person" (frame 1) → "Still a person" (frame 100)

**Perspectival Knowing** (spatial relationships):
- Object's motion trajectory
- Spatial relationships preserved temporally

**Participatory Knowing** (interactive refinement):
- User clicks → model tracks
- Feedback loop: correct frame 50 → propagates to future frames

**Memory as Relevance**:
- Memory bank = "what matters from the past"
- Attention weights = relevance scores
- Adaptive sampling = salience-based retention

### Temporal Relevance Realization

**Frame t relevance depends on**:
1. Recent frames (high relevance)
2. Key frames (object state changes)
3. Current task (user prompt)

**SAM 2 memory attention** implements temporal relevance:
- Attend strongly to recent, relevant frames
- Attend weakly to distant, less-relevant frames
- Result: Efficient temporal processing

---

## Key Takeaways

1. **SAM 2 = SAM + Streaming Memory + Temporal Consistency**
2. **Hiera encoder**: 6× faster than ViT-H (88× for video)
3. **Real-time video**: 44 FPS (23ms latency)
4. **Constant memory**: Works for infinite-length videos
5. **Unified model**: Images + videos in one architecture
6. **SOTA performance**: Best on DAVIS, YouTube-VOS benchmarks
7. **Promptable video**: Click once → track through entire video

---

## References

**Paper**: Ravi, N., et al. (2024). "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714
**Citations**: 2,421+ (as of 2025-11-21)
**Code**: https://github.com/facebookresearch/segment-anything-2
**Demo**: https://sam2.metademolab.com/

**Related Papers**:
- Bolya, D., et al. (2023). "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles." ICML 2023.
- Original SAM: Kirillov, A., et al. (2023). "Segment Anything." ICCV 2023.

---

**Last Updated**: 2025-11-21
**Status**: Active Research (Meta AI 2024)
**Production**: Available via Meta AI + Ultralytics integration
