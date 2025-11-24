# 3D Volumetric Pyramids for Video Understanding

## Overview

Video understanding requires processing spatiotemporal data across three dimensions: width (X), height (Y), and time (T). Extending 2D image pyramids into the temporal dimension creates **3D volumetric pyramids** that enable hierarchical multi-scale video processing. These structures are essential for efficient video encoding, action recognition, and vision-language models operating on video data.

This document explores spatiotemporal pyramid architectures, temporal mipmap structures, video Vision Transformers with multi-scale features, and efficient encoding hierarchies for real-time video inference.

**Key Challenge**: Video data introduces massive memory requirements. A 1920×1080 video at 30fps with 5 pyramid levels consumes:
- Spatial pyramid alone: ~12MB per frame
- Temporal extension (100 frames): **1.2GB** for single pyramid
- Full spatiotemporal pyramid: **Memory explosion** without careful design

---

## Section 1: Spatiotemporal Pyramids (X×Y×T)

### Extending 2D Pyramids to 3D Space-Time

Traditional 2D image pyramids downsample spatial dimensions (X, Y) to create multiple resolution levels. **Spatiotemporal pyramids** extend this concept into the temporal dimension (T), creating a 3D hierarchy that captures both spatial structure and temporal dynamics.

**Pyramid Structure**:
```
Level 0: 1920×1080 × 30fps  (full resolution, full frame rate)
Level 1:  960×540  × 15fps  (half spatial, half temporal)
Level 2:  480×270  × 8fps   (quarter spatial, ~quarter temporal)
Level 3:  240×135  × 4fps   (eighth spatial, eighth temporal)
```

Each pyramid level reduces **both** spatial dimensions and temporal sampling rate, maintaining coherence across space-time.

From [A Semantic and Motion-Aware Spatiotemporal Transformer](https://arxiv.org/html/2405.08204v1) (arXiv:2405.08204, accessed 2025-01-31):
- Spatiotemporal pyramid pooling mechanisms extract features at multiple scales
- Coarse levels capture long-range motion patterns (e.g., walking trajectory)
- Fine levels capture local temporal details (e.g., hand gestures, facial expressions)

### 3D Convolution Downsampling

Downsampling spatiotemporal pyramids requires 3D convolution kernels that operate across X, Y, and T simultaneously:

**3D Box Filter** (simple averaging):
```python
# Pseudo-code: 3D downsampling by factor of 2
def downsample_3d(video_tensor):  # Shape: (T, H, W, C)
    # Spatial: 2×2 box filter
    # Temporal: average pairs of frames
    return F.avg_pool3d(video_tensor, kernel_size=(2, 2, 2), stride=(2, 2, 2))
```

**Learned 3D Convolutions** (neural downsampling):
```python
# Learned 3D downsampling layer
conv3d = nn.Conv3d(
    in_channels=256,
    out_channels=256,
    kernel_size=(3, 3, 3),  # Temporal + spatial receptive field
    stride=(2, 2, 2),        # Downsample all dimensions
    padding=(1, 1, 1)
)
```

From [Spatiotemporal Pyramid Network for Video Action Recognition](https://www.researchgate.net/publication/320967245_Spatiotemporal_Pyramid_Network_for_Video_Action_Recognition) (accessed 2025-01-31):
- 3D convolutional filters learn motion-aware downsampling
- Temporal coherence preserved across pyramid levels (smooth transitions between frames)
- Avoids temporal aliasing (e.g., wagon-wheel effect in fast motion)

### Temporal Coherence Across Pyramid Levels

**Critical requirement**: Adjacent frames in the same pyramid level must maintain temporal consistency. Naive downsampling can introduce temporal artifacts.

**Coherence strategies**:
1. **Optical flow-guided downsampling**: Warp frames before averaging to account for motion
2. **Motion-compensated filtering**: Use motion vectors (similar to video codecs) to align frames
3. **Learned temporal consistency**: Neural networks trained to preserve motion smoothness

**Example**: Action recognition benefits from temporally coherent pyramids:
- Level 0 (30fps): Detect rapid hand movements
- Level 2 (8fps): Classify overall action (waving, clapping)
- Temporal jumps or inconsistencies confuse motion understanding

### Memory Explosion Problem

**Fundamental challenge**: Spatiotemporal pyramids scale cubically with video length.

**Memory calculation for 10-second 1080p video (30fps)**:
```
Base resolution: 1920×1080 × 300 frames × 3 channels × 4 bytes (float32)
= 7.5 GB (uncompressed)

With 4-level pyramid (1.33× storage overhead per dimension):
Spatial overhead: 1.33 (standard 2D pyramid)
Temporal overhead: 1.33 (temporal pyramid)
Total: 7.5 GB × 1.33 × 1.33 ≈ 13 GB

For batch processing (16 videos): 208 GB GPU memory!
```

**Mitigation strategies**:
1. **Sparse temporal sampling**: Don't store all frame rates at all levels (see Section 2)
2. **Tile-based processing**: Process video in temporal chunks (10-frame windows)
3. **Low-precision storage**: FP16 or INT8 at coarse levels (see `pyramid-lod/10-quantization-aware-pyramid-storage.md`)
4. **On-the-fly generation**: Compute pyramid levels during forward pass, discard after use

From [Understanding Video Transformers for Segmentation](https://arxiv.org/html/2310.12296) (accessed 2025-01-31):
- Modern video transformers adopt sparse spatiotemporal sampling to avoid memory explosion
- Trade-off: miss high-frequency temporal events (e.g., ball bounce) but capture global motion

---

## Section 2: Temporal Mipmap Structures

### Frame-Rate Pyramids (60fps → 30fps → 15fps)

**Temporal mipmaps** treat frame rate as analogous to spatial resolution in standard mipmaps. Each level halves the temporal sampling rate while maintaining spatial resolution.

**Pyramid structure**:
```
Level 0: Full resolution @ 60fps   (16.67ms per frame)
Level 1: Full resolution @ 30fps   (33.33ms per frame)
Level 2: Full resolution @ 15fps   (66.67ms per frame)
Level 3: Full resolution @ 7.5fps  (133ms per frame)
```

**Use case: Action recognition at multiple temporal scales**:
- **60fps**: Fine-grained hand gesture recognition (finger movements)
- **30fps**: Human action classification (walking, jumping)
- **15fps**: Scene-level activity understanding (crowd behavior)
- **7.5fps**: Long-term context (entire video summary)

From [TPDiff: Temporal Pyramid Video Diffusion Model](https://arxiv.org/abs/2503.09566) (arXiv:2503.09566, accessed 2025-01-31):
- TPDiff framework divides video diffusion into stages with progressive frame rate increase
- Early diffusion stages operate at low frame rates (high entropy reduction)
- Final stage uses full frame rate for detailed temporal synthesis
- **Result**: 50% training cost reduction, 1.5× inference speedup

### Temporal Downsampling Strategies

**Challenge**: How to reduce frame rate while preserving motion information?

**1. Frame Skipping (Simple Decimation)**:
```python
# Keep every Nth frame
def frame_skip(video, factor=2):
    return video[::factor]  # Takes frames 0, 2, 4, 6, ...
```
- **Pros**: Zero computational cost, exact frames preserved
- **Cons**: Temporal aliasing (fast motion becomes jumpy), loses in-between information

**2. Frame Blending (Temporal Averaging)**:
```python
# Average adjacent frames
def frame_blend(video, factor=2):
    blended = []
    for i in range(0, len(video), factor):
        chunk = video[i:i+factor]
        blended.append(chunk.mean(dim=0))
    return blended
```
- **Pros**: Smoother motion, reduces temporal aliasing
- **Cons**: Motion blur, loses sharpness (similar to camera long exposure)

**3. Optical Flow Interpolation**:
```python
# Synthesize intermediate frame using optical flow
def optical_flow_downsample(video, factor=2):
    """Compute flow between frames, downsample in flow space"""
    flows = compute_optical_flow(video)  # Motion vectors
    downsampled = select_keyframes(video, flows, factor)
    return downsampled
```
- **Pros**: Motion-aware sampling, preserves salient motion events
- **Cons**: Computational cost (optical flow estimation), errors on complex motion

From [ResidualViT for Efficient Temporally Dense Video Encoding](https://arxiv.org/html/2509.13255v1) (arXiv:2509.13255, accessed 2025-01-31):
- ResidualViT architecture optimizes for efficient video encoding with sparse frame sampling
- Uses residual connections across temporal scales for dense feature extraction
- Achieves state-of-the-art efficiency on video action recognition benchmarks

### Motion-Aware Temporal Filtering

**Adaptive frame sampling** based on motion content:
- High-motion scenes: Sample more densely (higher frame rate)
- Static scenes: Aggressive downsampling (lower frame rate)

**Algorithm sketch**:
```python
def motion_aware_sampling(video):
    motion_scores = []
    for i in range(1, len(video)):
        # Simple motion metric: frame difference
        diff = np.abs(video[i] - video[i-1]).mean()
        motion_scores.append(diff)

    # Adaptive sampling rate
    sampled_frames = []
    for i, score in enumerate(motion_scores):
        if score > threshold_high:
            sampled_frames.append(video[i])  # Keep frame
        elif score > threshold_low and i % 2 == 0:
            sampled_frames.append(video[i])  # Keep every 2nd
        elif i % 4 == 0:
            sampled_frames.append(video[i])  # Keep every 4th

    return sampled_frames
```

**Applications**:
- Video compression: Allocate more bits to high-motion frames
- VLM inference: Process high-motion segments at full resolution, low-motion at coarse levels

### Video Codec Integration (H.264, VP9 Hierarchical B-Frames)

Modern video codecs already implement hierarchical temporal structures:

**H.264 Hierarchical B-Frames**:
```
I-frame (keyframe) → B-frame → B-frame → B-frame → P-frame
       Level 0           Level 2   Level 1   Level 2    Level 0
```

- **I-frames**: Full independent frames (Level 0 in pyramid)
- **P-frames**: Predicted from previous frames (Level 1)
- **B-frames**: Bidirectionally predicted (Level 2, highest compression)

**Integration with neural networks**:
- Decode I-frames at full resolution → feed to ViT encoder
- Decode P/B-frames at coarse levels → lightweight processing
- **Benefit**: Leverage existing codec infrastructure, avoid re-encoding video

From [Efficient Video Encoding with Pyramid Frame Sampling](https://www.researchgate.net/publication/271920091_Keyframe_extraction_in_endoscopic_video) (accessed 2025-01-31):
- Keyframe extraction for efficient video encoding reduces storage by 60-80%
- Can be combined with neural codecs for end-to-end learned compression

**Challenge**: Video codec DCT/wavelet representations don't align well with neural network features. Emerging research on **neural video codecs** (learned end-to-end) may provide better pyramid structures for VLMs.

---

## Section 3: Video ViT with Multi-Scale Temporal Features

### TimeSformer, ViViT Architectures

**Video Vision Transformers (Video ViT)** extend image ViT to spatiotemporal data by incorporating temporal attention mechanisms.

**TimeSformer** (Facebook AI, ICML 2021):
- **Divided attention**: Separate spatial and temporal attention blocks
- **Architecture**:
  ```
  Input: Video clip (T × H × W × 3)
  ↓
  Patch embedding: Divide into patches (T × P × P)
  ↓
  [Spatial Attention Block] → Attend within each frame
  ↓
  [Temporal Attention Block] → Attend across frames at same spatial position
  ↓
  Classification head
  ```

- **Advantage**: Factorized attention reduces computational cost (O(T·HW + H·W·T) vs O(T·H·W)^2)
- **Pyramid connection**: Can process different temporal resolutions at different layers

**ViViT** (Google Research, ICCV 2021):
- **Tubelet embedding**: 3D patches across space-time
- **Multi-scale feature extraction**: Early layers process coarse spatiotemporal features, late layers refine
- **Pyramid integration**: Can use temporal pyramid as input (feed different frame rates to different transformer layers)

From [Video ViT with Multi-Scale Temporal Features](https://www.researchgate.net/publication/346663492_Gait_Lateral_Network_Learning_Discriminative_and_Compact_Representations_for_Gait_Recognition) (accessed 2025-01-31):
- GaitVViT leverages Video ViT architectures for multi-scale temporal feature extraction
- Extracts temporal context at different granularities for robust gait recognition

### Multi-Scale Temporal Attention

**Hierarchical temporal attention** across pyramid levels:

```python
class MultiScaleTemporalAttention(nn.Module):
    def __init__(self):
        self.attention_coarse = TemporalAttention(num_frames=8)   # 8fps
        self.attention_medium = TemporalAttention(num_frames=16)  # 16fps
        self.attention_fine = TemporalAttention(num_frames=32)    # 32fps

    def forward(self, video_pyramid):
        # Attend at multiple temporal scales
        feat_coarse = self.attention_coarse(video_pyramid['8fps'])
        feat_medium = self.attention_medium(video_pyramid['16fps'])
        feat_fine = self.attention_fine(video_pyramid['32fps'])

        # Fuse multi-scale features
        return feat_coarse + feat_medium + feat_fine
```

**Benefits**:
- **Coarse levels**: Capture long-range temporal dependencies (entire action sequence)
- **Fine levels**: Capture short-term dynamics (sudden motion changes)
- **Fusion**: Combines complementary temporal information

From [A Survey of Video Action Recognition Based on Deep Learning](https://www.sciencedirect.com/science/article/pii/S0950705125006409) (accessed 2025-01-31):
- Attention mechanisms (AMs) are critical for video understanding architectures
- Multi-scale temporal attention provides significant improvements over single-scale approaches

### Coarse: Long-Range Motion, Fine: Frame-Level Detail

**Temporal hierarchy in action recognition**:

| Pyramid Level | Frame Rate | Captures | Example Features |
|--------------|-----------|----------|------------------|
| Level 0 (Fine) | 30fps | Frame-level details | Hand position, facial micro-expressions |
| Level 1 (Medium) | 15fps | Motion segments | Arm swing trajectory, walking cycle |
| Level 2 (Coarse) | 8fps | Overall action | "Waving hand", "throwing ball" |
| Level 3 (Context) | 4fps | Scene context | "Person in kitchen", "outdoor sports" |

**Query-aware temporal allocation** (ARR-COC style):
- Query: "Is the person waving?" → Focus on Level 2 (overall action pattern)
- Query: "What is the exact hand gesture?" → Focus on Level 0 (fine detail)
- Query: "What activity is happening?" → Focus on Level 3 (long-term context)

This mirrors the spatial pyramid allocation in `pyramid-lod/03-attention-driven-pyramid-pruning.md`, extended to the temporal dimension.

### Efficient Video Transformers (Token Merging, FastViT)

**Challenge**: Full spatiotemporal attention on high-resolution video is computationally prohibitive.

**Token Merging for Video** (ToMe-Video):
```python
# Progressively merge similar tokens across frames
def token_merge_video(tokens, similarity_threshold=0.85):
    """Tokens shape: (T, N, D) where T=frames, N=patches, D=features"""
    merged_tokens = []
    for t in range(len(tokens)):
        # Compute similarity between tokens at frame t and t-1
        if t > 0:
            similarity = cosine_similarity(tokens[t], tokens[t-1])
            # Merge highly similar tokens (static regions)
            tokens[t] = merge_similar(tokens[t], tokens[t-1], similarity)
        merged_tokens.append(tokens[t])
    return merged_tokens
```

**Result**: 40-60% token reduction in low-motion video sequences without accuracy loss.

**FastViT for Video**:
- Hierarchical architecture with pyramid feature extraction
- Early layers use coarse temporal resolution (8fps)
- Late layers refine with fine temporal resolution (30fps)
- **Efficiency gain**: 3× faster inference vs uniform temporal sampling

From [ResidualViT for Efficient Video Encoding](https://arxiv.org/html/2509.13255v1) (accessed 2025-01-31):
- ResidualViT architecture requires 50% fewer FLOPs than standard Video ViT
- Achieves comparable accuracy on Kinetics-400 and Something-Something-v2 benchmarks

**Integration with pyramid LOD**:
- Allocate more tokens to high-relevance temporal regions (query-aware)
- Coarse levels: Fewer tokens per frame (global context)
- Fine levels: More tokens per frame (local detail)

---

## Section 4: Efficient Video Encoding Hierarchies

### Learned Video Codecs (Neural Compression)

**Traditional codecs** (H.264, HEVC) use hand-crafted transforms (DCT, motion estimation). **Neural video codecs** learn end-to-end compression with neural networks.

**Architecture sketch**:
```python
class NeuralVideoCodec(nn.Module):
    def __init__(self):
        self.encoder = SpatioTemporalEncoder()  # Compress to latent space
        self.quantizer = VectorQuantizer()       # Discretize latents
        self.decoder = SpatioTemporalDecoder()  # Reconstruct video

    def encode(self, video):
        # Hierarchical encoding across temporal scales
        latents_coarse = self.encoder(downsample_time(video, factor=4))
        latents_fine = self.encoder(video)
        return self.quantizer(latents_coarse), self.quantizer(latents_fine)

    def decode(self, latents_coarse, latents_fine):
        # Progressive decoding: coarse → fine
        video_coarse = self.decoder(latents_coarse)
        video_fine = self.decoder(latents_fine, condition=video_coarse)
        return video_fine
```

**Pyramid structure in neural codecs**:
- Encode video at multiple temporal scales (8fps, 16fps, 30fps)
- Transmit coarse levels first (progressive streaming)
- Decode coarse levels quickly for preview, refine with fine levels

From [TPDiff: Temporal Pyramid Video Diffusion](https://arxiv.org/abs/2503.09566) (accessed 2025-01-31):
- TPDiff uses stage-wise diffusion across temporal pyramid levels
- Partitioned probability flow ODEs enable efficient multi-stage training
- Applicable to various diffusion forms (DDPM, DDIM, flow-matching)

### Hierarchical Motion Prediction

**Motion compensation** is critical for video compression. Hierarchical motion estimation leverages pyramid structure:

**Coarse-to-fine motion estimation**:
1. **Level 3** (8fps): Estimate large-scale motion (e.g., camera pan, person walking)
2. **Level 2** (16fps): Refine motion vectors (e.g., arm movement relative to body)
3. **Level 1** (30fps): Fine-grained motion (e.g., finger gestures, facial expressions)

```python
def hierarchical_motion_estimation(video_pyramid):
    # Start at coarse level
    motion_L3 = estimate_motion(video_pyramid['L3'])

    # Refine at medium level
    motion_L2 = estimate_motion(
        video_pyramid['L2'],
        init=upsample(motion_L3, factor=2)  # Initialize from coarse level
    )

    # Refine at fine level
    motion_L1 = estimate_motion(
        video_pyramid['L1'],
        init=upsample(motion_L2, factor=2)
    )

    return motion_L1  # Full-resolution motion vectors
```

**Benefits**:
- Faster convergence (coarse initialization guides fine search)
- Handles large displacements (coarse level captures large motion)
- Reduces computational cost (small search window at fine levels)

### Pyramid-Based Optical Flow

**Classical optical flow** (Lucas-Kanade, Farnebäck) operates on image pyramids. **Neural optical flow** (RAFT, FlowFormer) can integrate pyramid structures:

**RAFT with Pyramid Features**:
```python
class PyramidRAFT(nn.Module):
    def __init__(self):
        self.feature_encoder = FeaturePyramidNetwork()
        self.correlation_layer = CorrelationPyramid()
        self.refinement = GRURefiner()

    def forward(self, frame1, frame2):
        # Extract feature pyramids from both frames
        feat_pyramid1 = self.feature_encoder(frame1)  # Levels 0-3
        feat_pyramid2 = self.feature_encoder(frame2)

        # Compute correlation at each pyramid level
        corr_pyramid = self.correlation_layer(feat_pyramid1, feat_pyramid2)

        # Iteratively refine flow (coarse to fine)
        flow = None
        for level in reversed(range(4)):  # L3 → L2 → L1 → L0
            flow = self.refinement(corr_pyramid[level], flow_init=flow)

        return flow  # Final optical flow field
```

**Applications in video VLMs**:
- Motion-aware frame sampling (skip static frames, densely sample high-motion)
- Query-aware motion focus: "Where is the ball moving?" → High-res flow in ball region
- Temporal token merging: Merge tokens in static regions (zero optical flow)

### Real-Time Video Inference (30fps+)

**Challenge**: Video VLMs must process 30+ frames per second for real-time applications (live video chat, robotics, AR/VR).

**Pyramid-based optimization for 30fps inference**:

**Strategy 1: Temporal Keyframe Selection**
```python
# Process only keyframes at full resolution
keyframe_interval = 10  # Full inference every 10 frames
for i, frame in enumerate(video_stream):
    if i % keyframe_interval == 0:
        # Full VLM inference on keyframe
        features = vlm.encode(frame, resolution='full')
    else:
        # Lightweight update using optical flow
        features = update_features(features, optical_flow(frame, prev_frame))
```

**Strategy 2: Sliding Temporal Window**
```python
# Maintain pyramid for last N frames
temporal_window = deque(maxlen=30)  # 1 second at 30fps

for frame in video_stream:
    temporal_window.append(frame)

    # Downsample temporal window to pyramid
    pyramid = create_temporal_pyramid(temporal_window)

    # VLM inference on pyramid (not full video)
    output = vlm.process_pyramid(pyramid)
```

**Strategy 3: Multi-Resolution Streaming**
```python
# Process different spatial regions at different temporal rates
def adaptive_streaming(video_stream, attention_map):
    for frame in video_stream:
        # High-attention regions: full frame rate
        roi_high = extract_roi(frame, attention_map > 0.8)
        process(roi_high, fps=30)

        # Medium-attention regions: half frame rate
        roi_med = extract_roi(frame, attention_map > 0.5)
        if frame_count % 2 == 0:
            process(roi_med, fps=15)

        # Low-attention regions: quarter frame rate
        roi_low = extract_roi(frame, attention_map <= 0.5)
        if frame_count % 4 == 0:
            process(roi_low, fps=7.5)
```

From [Understanding Video Transformers for Segmentation](https://arxiv.org/html/2310.12296) (accessed 2025-01-31):
- Video transformer architectures must balance temporal resolution with computational constraints
- Sparse sampling strategies (uniform, random, keyframe) have different trade-offs

**Real-world systems**:
- **Meta Quest 3**: Foveated rendering + 90fps video processing (VR)
- **Tesla FSD**: Multi-camera video at 36fps (8 cameras × 4.5fps effective)
- **Google Meet**: Real-time background blur on 720p @ 30fps

**Pyramid enables real-time inference**:
- Process coarse pyramid levels on CPU (low power)
- Process fine pyramid levels on GPU (high performance)
- Dynamically adjust pyramid depth based on available compute

See `pyramid-lod/07-hybrid-cpu-gpu-pyramid.md` for heterogeneous processing strategies.

---

## Sources

**Source Documents:**
- [practical-implementation/55-3d-volume-texture-spatiotemporal-vit.md](../practical-implementation/55-3d-volume-texture-spatiotemporal-vit.md)

**Web Research:**
- [A Semantic and Motion-Aware Spatiotemporal Transformer](https://arxiv.org/html/2405.08204v1) - arXiv:2405.08204 (accessed 2025-01-31)
- [TPDiff: Temporal Pyramid Video Diffusion Model](https://arxiv.org/abs/2503.09566) - arXiv:2503.09566 (accessed 2025-01-31)
- [ResidualViT for Efficient Temporally Dense Video Encoding](https://arxiv.org/html/2509.13255v1) - arXiv:2509.13255 (accessed 2025-01-31)
- [Spatiotemporal Pyramid Network for Video Action Recognition](https://www.researchgate.net/publication/320967245_Spatiotemporal_Pyramid_Network_for_Video_Action_Recognition) (accessed 2025-01-31)
- [Understanding Video Transformers for Segmentation](https://arxiv.org/html/2310.12296) (accessed 2025-01-31)
- [A Survey of Video Action Recognition Based on Deep Learning](https://www.sciencedirect.com/science/article/pii/S0950705125006409) (accessed 2025-01-31)
- [Gait Lateral Network: Video ViT Multi-Scale Features](https://www.researchgate.net/publication/346663492_Gait_Lateral_Network_Learning_Discriminative_and_Compact_Representations_for_Gait_Recognition) (accessed 2025-01-31)
- [Keyframe Extraction in Endoscopic Video](https://www.researchgate.net/publication/271920091_Keyframe_extraction_in_endoscopic_video) (accessed 2025-01-31)

**Additional References:**
- TimeSformer (Facebook AI, ICML 2021) - Divided spatiotemporal attention
- ViViT (Google Research, ICCV 2021) - Tubelet embedding for video
- RAFT Optical Flow - Feature pyramid-based motion estimation
- H.264/HEVC Video Codecs - Hierarchical B-frame structures
