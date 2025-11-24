# Video Understanding Extensions for Vision-Language Models

## Overview

Extending vision-language models (VLMs) to video understanding represents a critical evolution from single-image reasoning to temporal, multi-frame comprehension. Video VLMs must process spatiotemporal data, capture motion dynamics, reason across time, and handle significantly longer context lengths than their image-based counterparts. This document explores the architectures, sampling strategies, temporal encoding methods, and benchmarks that enable effective video understanding in modern VLMs.

**Key Challenge**: Video introduces a temporal dimension that explodes computational requirements. A 1-minute video at 30fps contains 1,800 frames. At 256 tokens per frame (standard ViT 224×224), this yields 460,800 tokens—far exceeding most LLM context windows. Efficient video VLMs require intelligent frame sampling, temporal compression, and hierarchical processing strategies.

From [CogVLM2: Visual Language Models for Image and Video Understanding](https://arxiv.org/abs/2408.16500) (arXiv:2408.16500, accessed 2025-11-16):
- CogVLM2-Video extends image VLMs to video with temporal attention mechanisms
- Processes multiple frames while maintaining spatial resolution
- Achieves state-of-the-art on video question answering benchmarks
- Balances temporal coverage with computational efficiency

---

## Section 1: Video VLM Architectures

### Architecture Taxonomy

**Video VLMs extend image VLMs with temporal processing capabilities:**

**1. Frame-Independent Processing** (Early Approaches):
```
Architecture:
  For each frame f in video:
    features[f] = vision_encoder(frame[f])

  concatenated = concat(features[0], features[1], ..., features[T])
  answer = LLM(concatenated + question)

Advantages:
- Simple, reuses image VLM architecture
- No architectural changes needed

Disadvantages:
- Ignores temporal relationships
- Massive token explosion (T × 256 tokens)
- No motion understanding
```

**2. Temporal Attention Mechanisms** (Modern Approaches):
```
Architecture:
  # Spatial encoding per frame
  For each frame f:
    spatial_features[f] = vision_encoder(frame[f])

  # Temporal attention across frames
  temporal_features = temporal_attention(spatial_features)

  # Fuse and process
  answer = LLM(temporal_features + question)

Advantages:
- Explicit temporal modeling
- Captures motion and frame relationships
- Learned temporal dependencies

Disadvantages:
- Higher computational cost
- Requires temporal attention layers
```

**3. Hierarchical Spatiotemporal Processing**:
```
Architecture:
  # Multi-level temporal hierarchy
  Level 1: Local temporal features (1-5 frames)
  Level 2: Medium temporal features (5-50 frames)
  Level 3: Global video features (entire video)

  # Hierarchical fusion
  features = fuse(level1, level2, level3)
  answer = LLM(features + question)

Advantages:
- Captures both fine-grained and long-range patterns
- Efficient multi-scale processing
- Better temporal coverage

Disadvantages:
- Complex architecture
- Requires careful pyramid design
```

From [VILA: Visual Language Model Family](https://github.com/NVlabs/VILA) (accessed 2025-11-16):
- VILA family optimized for video understanding and multi-image reasoning
- Efficient temporal processing with frame sampling strategies
- State-of-the-art performance on video benchmarks
- Open-source implementation for video VLMs

### Video-LLaVA and Video-ChatGPT

**Video-LLaVA** extends the LLaVA architecture to video:

**Architecture**:
```
Components:
  1. Vision Encoder: CLIP ViT-L/14 (frozen)
  2. Frame Sampler: Uniform or adaptive sampling
  3. Temporal Aggregator: Temporal attention layers
  4. Projection Layer: Maps visual tokens to LLM space
  5. Language Model: Vicuna-13B or Llama-2

Processing Flow:
  video (T frames)
    → sample N frames (typically 8-16)
    → vision_encoder (N × 256 tokens)
    → temporal_aggregator (compress to M tokens, M < N×256)
    → projection_layer
    → LLM(visual_tokens + text_query)
```

**Video-ChatGPT** (similar architecture, different training):
- Focus on conversational video understanding
- Multi-turn dialogue about video content
- Trained on video instruction-following data

From [Towards Detailed Video Understanding via Large Vision and Language Models](https://aclanthology.org/2024.acl-long.679/) (ACL 2024, cited 1170 times):
- Merges video-adapted visual encoder with LLM
- Capable of understanding and generating detailed video descriptions
- Handles complex temporal reasoning tasks
- Strong performance on video question answering

### VideoLLaMA and CogVLM2-Video

**VideoLLaMA**:
- Audio-visual understanding (video + sound)
- Dual encoders: vision encoder + audio encoder
- Cross-modal fusion for audio-visual reasoning
- Applications: video captioning, QA with sound

**CogVLM2-Video** (State-of-the-art, 2024):

**Architecture Innovations**:
```
1. Temporal Visual Attention:
   - Dedicated temporal attention blocks
   - Separate spatial and temporal processing
   - Efficient factorized attention

2. Dynamic Frame Sampling:
   - Adaptive sampling based on motion
   - More frames for high-motion segments
   - Fewer frames for static scenes

3. Long Context Support:
   - Processes up to 1 hour of video
   - Hierarchical temporal windows
   - Memory-efficient KV cache management
```

From [CogVLM2: Visual Language Models for Image and Video Understanding](https://arxiv.org/abs/2408.16500) (arXiv:2408.16500, 211 citations):
- CogVLM2 family includes CogVLM2, CogVLM2-Video, and GLM-4V
- New generation of visual language models
- Superior performance on both image and video tasks
- Efficient architecture for long video understanding

---

## Section 2: Frame Sampling Strategies

### Uniform Sampling

**Simplest approach**: Sample frames at regular intervals.

```python
def uniform_sampling(video, num_frames=8):
    """
    Sample num_frames evenly spaced from video.

    Args:
        video: List of frames [f0, f1, ..., f_T]
        num_frames: Number of frames to sample

    Returns:
        Sampled frames
    """
    T = len(video)
    indices = np.linspace(0, T-1, num_frames, dtype=int)
    return [video[i] for i in indices]

# Example: 300 frame video → sample 8 frames
# Indices: [0, 42, 85, 128, 171, 214, 257, 299]
```

**Advantages**:
- Simple, deterministic, reproducible
- Even temporal coverage
- No additional computation

**Disadvantages**:
- May miss important events between samples
- Ignores motion content
- Same sampling for all videos (not adaptive)

From [Frame Sampling Strategies Matter: A Benchmark for Small VLMs](https://arxiv.org/html/2509.14769v1) (arXiv:2509.14769, accessed 2025-11-16):
- Frame sampling strategy significantly impacts VLM video understanding
- Uniform sampling creates "frame-sampling bias" in benchmarks
- Different sampling strategies can change accuracy by 10-20%
- First frame-accurate benchmark for small video VLMs

### Adaptive Sampling (Motion-Aware)

**Sample more densely in high-motion regions**:

```python
def motion_aware_sampling(video, num_frames=8):
    """
    Sample frames based on motion content.
    More frames in high-motion segments.
    """
    motion_scores = []
    for i in range(1, len(video)):
        # Compute frame difference (simple motion metric)
        diff = np.abs(video[i] - video[i-1]).mean()
        motion_scores.append(diff)

    # Normalize to [0, 1]
    motion_scores = np.array(motion_scores)
    motion_scores = motion_scores / (motion_scores.max() + 1e-8)

    # Sample with probability proportional to motion
    probs = motion_scores / motion_scores.sum()
    indices = np.random.choice(
        len(video),
        size=num_frames,
        replace=False,
        p=probs
    )
    indices.sort()

    return [video[i] for i in indices]
```

**Advantages**:
- Captures important motion events
- Reduces redundancy in static scenes
- Better temporal coverage of dynamic content

**Disadvantages**:
- Requires motion computation (optical flow or frame differencing)
- Non-deterministic (randomness in sampling)
- May miss static but important frames

From [MGSampler: An Explainable Sampling Strategy for Video Action Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.pdf) (ICCV 2021, cited 105 times):
- Motion-guided sampling improves action recognition
- Explains which frames are most relevant for each action
- Reduces temporal redundancy while preserving key information
- Outperforms uniform sampling on action recognition benchmarks

### Keyframe Extraction

**Select frames at scene boundaries or semantic transitions**:

```python
def keyframe_extraction(video, num_frames=8, method='scene_change'):
    """
    Extract keyframes based on scene changes or semantic content.
    """
    if method == 'scene_change':
        # Detect scene boundaries
        scene_scores = []
        for i in range(1, len(video)):
            # Histogram difference (scene change detector)
            hist_diff = compute_histogram_difference(video[i], video[i-1])
            scene_scores.append(hist_diff)

        # Select peaks (scene boundaries)
        keyframe_indices = find_peaks(scene_scores, num_frames)

    elif method == 'semantic_diversity':
        # Cluster frames by visual similarity
        # Select cluster centroids as keyframes
        features = [extract_features(frame) for frame in video]
        keyframe_indices = kmeans_centroids(features, num_frames)

    return [video[i] for i in sorted(keyframe_indices)]
```

**Advantages**:
- Captures semantic transitions
- Reduces redundancy (one frame per scene)
- Good for long videos with multiple scenes

**Disadvantages**:
- May miss within-scene dynamics
- Requires scene detection algorithms
- Not suitable for continuous action videos

### Query-Dependent Sampling

**ARR-COC-style relevance-driven sampling**:

```python
def query_aware_sampling(video, query, num_frames=8):
    """
    Sample frames most relevant to the query.

    This is the ARR-COC approach: relevance realization
    determines which frames to attend to.
    """
    # Encode query
    query_embedding = text_encoder(query)

    # Score each frame for relevance to query
    relevance_scores = []
    for frame in video:
        frame_embedding = vision_encoder(frame)
        relevance = cosine_similarity(query_embedding, frame_embedding)
        relevance_scores.append(relevance)

    # Sample top-k relevant frames
    top_indices = np.argsort(relevance_scores)[-num_frames:]
    top_indices.sort()  # Maintain temporal order

    return [video[i] for i in top_indices]
```

**Advantages**:
- Maximizes relevance to specific query
- Ignores irrelevant frames
- Better accuracy on focused questions

**Disadvantages**:
- Requires encoding all frames initially
- Different sampling for different queries (less efficient)
- May miss temporal context

From [Generative Frame Sampler for Long Video Understanding](https://aclanthology.org/2025.findings-acl.921.pdf) (ACL 2025 Findings, cited 6 times):
- GenS: generative frame sampling based on query
- Evaluates various frame sampling approaches
- Query-aware sampling outperforms uniform sampling
- Reduces frames needed while maintaining accuracy

---

## Section 3: Spatiotemporal Attention

### Factorized Spatial-Temporal Attention

**Separate spatial and temporal attention to reduce complexity**:

**Standard 3D Attention** (expensive):
```
Cost: O((T × H × W)²) where T=frames, H×W=spatial dimensions

For 8 frames of 224×224 with 14×14 patches:
  Tokens: 8 × 196 = 1,568 tokens
  Attention: 1,568² ≈ 2.5M operations per head
```

**Factorized Attention** (efficient):
```
Spatial Attention (within each frame):
  Cost: O(T × (H × W)²)
  For 8 frames: 8 × 196² = 307K operations per head

Temporal Attention (across frames at same spatial position):
  Cost: O(H × W × T²)
  For 8 frames: 196 × 8² = 12.5K operations per head

Total: 307K + 12.5K = 319.5K (vs 2.5M for full 3D)
  → 8× reduction in complexity
```

**Implementation**:
```python
class FactorizedSpatioTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        self.spatial_attention = MultiHeadAttention(dim, num_heads)
        self.temporal_attention = MultiHeadAttention(dim, num_heads)

    def forward(self, x):
        # x shape: (batch, time, height, width, channels)
        B, T, H, W, C = x.shape

        # Spatial attention: attend within each frame
        x_spatial = x.view(B * T, H * W, C)
        x_spatial = self.spatial_attention(x_spatial)
        x_spatial = x_spatial.view(B, T, H, W, C)

        # Temporal attention: attend across frames
        x_temporal = x_spatial.view(B, T, H * W, C)
        x_temporal = x_temporal.transpose(1, 2)  # (B, H*W, T, C)
        x_temporal = x_temporal.reshape(B * H * W, T, C)
        x_temporal = self.temporal_attention(x_temporal)
        x_temporal = x_temporal.view(B, H * W, T, C)
        x_temporal = x_temporal.transpose(1, 2)  # (B, T, H*W, C)

        return x_temporal.view(B, T, H, W, C)
```

From [MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs](https://arxiv.org/abs/2503.15871) (arXiv:2503.15871, cited 6 times):
- Introduces disentangled spatial-temporal representations
- Mitigates hallucination in video understanding
- Separate processing reduces confusion between action and scene
- Improves temporal reasoning accuracy

### Divided Attention (TimeSformer)

**TimeSformer**: Facebook's approach to video transformers.

**Architecture**:
```
Input: Video clip (T × H × W × 3)

1. Patch Embedding:
   Divide each frame into patches (P × P)
   Total tokens: T × (H/P) × (W/P)

2. Divided Attention Blocks:
   For each block:
     a. Temporal Attention:
        Query: patch at (t, h, w)
        Keys/Values: same spatial position (*, h, w) across all frames

     b. Spatial Attention:
        Query: patch at (t, h, w)
        Keys/Values: all positions (t, *, *) in same frame

3. Classification Head
```

**Advantages**:
- Linear complexity in both spatial and temporal dimensions
- Parallelizable (spatial and temporal attention independent)
- Scalable to long videos

**Disadvantages**:
- Misses some spatiotemporal correlations
- Two separate attention passes needed

From [Video Understanding with Temporal Attention](https://arxiv.org/html/2509.13255v1) (arXiv:2509.13255, accessed 2025-11-16):
- ResidualViT architecture for efficient temporally dense video encoding
- Residual connections across temporal scales
- State-of-the-art efficiency on video action recognition
- 50% fewer FLOPs than standard Video ViT

### Joint Spatiotemporal Attention (ViViT)

**ViViT** (Google): Tubelet embeddings for joint spatiotemporal modeling.

**Tubelet Embedding**:
```
Standard ViT: 2D patches (P × P) from single image
ViViT: 3D tubelets (T × P × P) from video

Tubelet: stack of patches across T consecutive frames
Embedding: 3D convolution over (T × P × P) region

Example:
  Input: 8 frames of 224×224
  Tubelet size: 2 × 16 × 16
  Result: 4 temporal × 14 spatial × 14 spatial = 784 tubelets
```

**Attention**:
- Single attention operation over all tubelets
- Captures spatiotemporal correlations directly
- More expensive but potentially more expressive

From [Long-Short Temporal Contrastive Learning of Video Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Long-Short_Temporal_Contrastive_Learning_of_Video_Transformers_CVPR_2022_paper.pdf) (CVPR 2022):
- Self-supervised pretraining of video transformers
- Factorized (2D+1D) and joint (3D) spatial-temporal operations
- Long-term and short-term temporal contrastive learning
- Strong action recognition results

---

## Section 4: Temporal Encoding

### Temporal Positional Encoding

**Encoding frame position in the sequence**:

**Absolute Temporal Encoding**:
```python
def temporal_position_encoding(frame_index, total_frames, d_model):
    """
    Encode temporal position similar to Transformer position encoding.

    Args:
        frame_index: Index of current frame (0 to T-1)
        total_frames: Total frames in video
        d_model: Embedding dimension

    Returns:
        Temporal position embedding
    """
    # Normalize to [0, 1]
    t = frame_index / total_frames

    # Sinusoidal encoding
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = np.sin(t / (10000 ** (i / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = np.cos(t / (10000 ** (i / d_model)))

    return pe
```

**Relative Temporal Encoding**:
```python
# Encode temporal distance between frames
def relative_temporal_bias(frame_i, frame_j, max_distance=100):
    """
    Bias based on temporal distance between frames.

    Frames close in time should attend to each other more.
    """
    distance = abs(frame_i - frame_j)
    # Learnable bias table indexed by distance
    return learned_bias[min(distance, max_distance)]
```

**Hierarchical Temporal Encoding**:
```python
def hierarchical_temporal_encoding(frame_index, clip_index, scene_index):
    """
    Multi-level temporal encoding:
      Level 1: Frame within clip
      Level 2: Clip within scene
      Level 3: Scene within video
    """
    frame_enc = encode_position(frame_index, max_frames_per_clip)
    clip_enc = encode_position(clip_index, max_clips_per_scene)
    scene_enc = encode_position(scene_index, max_scenes_per_video)

    # Concatenate or sum
    return concat(frame_enc, clip_enc, scene_enc)
```

From [Enhancing Video-Language Representations with Structural Spatio-Temporal Alignment](https://arxiv.org/html/2406.19255v1) (arXiv:2406.19255, accessed 2025-11-16):
- Fine-grained structural spatio-temporal alignment (Finsta)
- Multi-granularity temporal encoding
- Enhances VLMs with better temporal understanding
- Improves video-text alignment

### Rotary Position Embeddings (RoPE) for Video

**RoPE** can be extended to multi-dimensional sequences:

**2D Spatial RoPE** (for images):
```
Encodes 2D spatial position (x, y) using rotation matrices
```

**3D Spatiotemporal RoPE** (for video):
```
Encodes 3D position (t, x, y) using rotation matrices
  - t: temporal dimension
  - x, y: spatial dimensions

Advantage:
  - Generalizes to longer sequences
  - Relative position encoding built-in
  - No learned parameters
```

From [Qwen3-VL (oracle knowledge)](../qwen3vl-oracle/architecture/01-multimodal-rotary-position-embedding.md):
- M-RoPE (Multimodal RoPE) for vision-language models
- Extends to temporal dimension for video
- DeepStack architecture for efficient processing

---

## Section 5: Action Recognition and Temporal Reasoning

### Video Question Answering Benchmarks

**Major VQA Benchmarks**:

| Benchmark | Videos | Questions | Focus | Avg Length |
|-----------|--------|-----------|-------|------------|
| **MSRVTT-QA** | 10K | 243K | General video QA | 15s |
| **MSVD-QA** | 1.9K | 50K | Short clips | 10s |
| **ActivityNet-QA** | 5.8K | 58K | Human activities | 180s |
| **NExT-QA** | 5.4K | 47K | Causal reasoning | 44s |
| **CinePile** | 3.7K | 305K | Long-form movies | 2h+ |
| **CG-Bench** | 1.2K | 30K | Clue-grounded reasoning | 4.5min |

From [A Long Video Question Answering Dataset and Benchmark: CinePile](https://arxiv.org/abs/2405.08813) (arXiv:2405.08813, cited 81 times):
- Authentic long-form video understanding benchmark
- 3,763 web-collected videos averaging 2+ hours
- Tests temporal reasoning over extended narratives
- Challenges current VLMs on long-context video

From [CG-Bench: Clue-grounded Question Answering for Long Videos](https://openreview.net/forum?id=le4IoZZHy1) (cited 30 times):
- Emphasizes retrieving relevant clues from video
- 1,219 manually curated videos
- Enhances evaluation credibility through clue grounding
- Tests fine-grained temporal localization

### ActivityNet-QA and Action Understanding

**ActivityNet**: Human activity recognition dataset.

**Challenge**: Understanding complex human actions over time.

**Example Questions**:
- "What activity is the person performing?"
- "What happens after the person picks up the ball?"
- "Why did the person stop running?"

**Temporal Reasoning Requirements**:
1. **Action Recognition**: Identify specific actions (running, jumping, throwing)
2. **Temporal Ordering**: Understand sequence of actions (first X, then Y)
3. **Causal Reasoning**: Infer why actions occur (stopped because of obstacle)

From [LifeQA: A Real-Life Dataset for Video Question Answering](https://aclanthology.org/2020.lrec-1.536/) (LREC 2020, cited 47 times):
- Focus on real-life situations
- 275 video clips with 2.3K+ multiple-choice questions
- Tests practical understanding of everyday activities
- Benchmarks VLM performance on realistic scenarios

### Multi-Hop Temporal Reasoning

**Complex reasoning requiring evidence from multiple frames**:

**Example**:
```
Video: Person enters kitchen → opens fridge → takes milk → closes fridge → pours milk → drinks

Question: "What did the person drink?"

Required Reasoning:
  1. Frame 50: Identify milk being taken from fridge
  2. Frame 120: Connect milk to pouring action
  3. Frame 150: Connect poured liquid to drinking
  Answer: "Milk"

Evidence spans 100 frames → multi-hop reasoning required
```

**Architecture Requirements**:
- Long-range temporal dependencies
- Cross-frame information flow
- Memory/context aggregation

From [Video-CoT: Spatiotemporal Reasoning in Video QA](https://dl.acm.org/doi/abs/10.1145/3746027.3758313) (ACM 2025, cited 6 times):
- Video-CoT dataset for spatiotemporal reasoning
- Measures VLM generalization on rare/unseen answers
- Significant challenges for current VLMs
- Requires multi-hop reasoning across frames

---

## Section 6: Efficient Video Processing

### Token Reduction Strategies

**Challenge**: 8 frames × 256 tokens = 2,048 tokens (just for vision!)

**Strategy 1: Spatial Pooling per Frame**:
```python
# Reduce tokens per frame from 256 → 32
def spatial_pool(frame_tokens):  # (196, dim)
    # Average pooling in spatial dimension
    pooled = frame_tokens.view(14, 14, dim).mean(dim=[0, 1])
    return pooled  # (dim,) → 1 token per frame
```

**Strategy 2: Temporal Pooling Across Frames**:
```python
# Group frames and pool temporally
def temporal_pool(frames, group_size=2):
    pooled_frames = []
    for i in range(0, len(frames), group_size):
        group = frames[i:i+group_size]
        pooled = torch.stack(group).mean(dim=0)
        pooled_frames.append(pooled)
    return pooled_frames
```

**Strategy 3: Learned Token Merging**:
```python
# Merge similar tokens across space and time
def token_merging_video(tokens, similarity_threshold=0.85):
    """
    Merge highly similar tokens to reduce count.

    Tokens: (T, N, D) where T=frames, N=patches, D=features
    """
    merged = []
    for t in range(len(tokens)):
        if t > 0:
            # Compute similarity with previous frame
            sim = cosine_similarity(tokens[t], tokens[t-1])
            # Merge similar tokens (static regions)
            tokens[t] = merge_similar_tokens(tokens[t], tokens[t-1], sim)
        merged.append(tokens[t])

    return merged  # Reduced token count
```

From [STORM: Token-Efficient Long Video Understanding](https://arxiv.org/abs/2503.04130) (arXiv:2503.04130):
- Mamba State Space Model for temporal encoding
- Up to 8× computation reduction through token compression
- Test-time sampling + training-based pooling
- 5%+ improvement on MLVU and LongVideoBench

### Memory-Efficient Inference

**KV Cache Management for Long Videos**:

```python
class VideoKVCache:
    def __init__(self, max_cache_size=4096):
        self.max_cache_size = max_cache_size
        self.cache = deque(maxlen=max_cache_size)

    def update(self, new_kv):
        """Add new key-value pairs, evict oldest if needed."""
        if len(self.cache) >= self.max_cache_size:
            # Sliding window: evict oldest
            self.cache.popleft()
        self.cache.append(new_kv)

    def hierarchical_compression(self):
        """
        Compress old cache entries at lower resolution.

        Recent: full resolution (last 100 tokens)
        Medium: 2× compressed (100-500 tokens ago)
        Distant: 4× compressed (500+ tokens ago)
        """
        compressed = []
        cache_list = list(self.cache)

        # Recent: no compression
        compressed.extend(cache_list[-100:])

        # Medium: 2× compression
        medium = cache_list[-500:-100:2]  # every 2nd token
        compressed.extend(medium)

        # Distant: 4× compression
        distant = cache_list[:-500:4]  # every 4th token
        compressed.extend(distant)

        self.cache = deque(compressed, maxlen=self.max_cache_size)
```

From [Memory Consolidation for Long-Context Video Understanding](https://arxiv.org/html/2402.05861v2) (arXiv:2402.05861v2):
- Hierarchical Temporal Window Attention (HTWA)
- Enables scaling beyond training budget to 128K tokens
- Memory consolidation for long-range dependencies
- Efficient attention approximations and masking

### Streaming Video Processing

**Real-time processing for live video**:

```python
class StreamingVideoVLM:
    def __init__(self, window_size=30):
        self.window = deque(maxlen=window_size)  # 1 second at 30fps
        self.global_context = None

    def process_frame(self, frame, query):
        """
        Process incoming frame in real-time.
        """
        # Add to sliding window
        self.window.append(frame)

        # Update global context (every 10 frames)
        if len(self.window) % 10 == 0:
            self.global_context = self.encode_window(self.window)

        # Quick inference on local window
        local_features = self.encode_local(list(self.window)[-5:])

        # Combine local + global
        answer = self.llm(local_features, self.global_context, query)

        return answer

    def encode_window(self, frames):
        """Encode full window into compressed global context."""
        features = [self.vision_encoder(f) for f in frames]
        return self.temporal_aggregator(features)

    def encode_local(self, frames):
        """Quick encoding of recent frames."""
        return [self.vision_encoder(f) for f in frames]
```

---

## Section 7: Multi-Modal Extensions

### Audio-Visual Understanding

**VideoLLaMA**: Combines vision and audio.

**Architecture**:
```
Input: Video (visual frames) + Audio waveform

Visual Path:
  frames → vision_encoder → visual_features

Audio Path:
  waveform → audio_encoder → audio_features

Fusion:
  combined = fusion_layer(visual_features, audio_features)

LLM:
  answer = llm(combined + text_query)
```

**Use Cases**:
- "What instrument is playing?" (audio required)
- "Is the person speaking?" (audio-visual sync)
- "Describe the music in the video" (audio focus)

### Video Captioning

**Dense Video Captioning**: Generate captions for multiple events in video.

**Architecture**:
```
Input: Long video (5 minutes)

1. Temporal Segmentation:
   - Detect event boundaries
   - Segment video into clips (10-30 seconds each)

2. Caption Generation per Clip:
   For each clip:
     features = vision_encoder(clip_frames)
     caption = llm.generate(features)

3. Temporal Grounding:
   - Timestamp each caption
   - "0:15-0:30: Person enters room"
   - "0:30-1:00: Person cooks dinner"
```

From [Distilling Vision-Language Models on Millions of Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Distilling_Vision-Language_Models_on_Millions_of_Videos_CVPR_2024_paper.pdf) (CVPR 2024, cited 29 times):
- Distills VLMs on millions of videos
- Generates high-quality pseudo-captions
- Motion-focused output for dynamic understanding
- Scales video-language model training

---

## Section 8: ARR-COC-0-1 Video Extension

### Temporal Relevance Realization

**Extending ARR-COC's relevance realization to video**:

**Core Concept**: Not all frames are equally relevant to a query. Apply Vervaekean relevance realization across temporal dimension.

**Architecture**:
```python
class ARRCOC_VideoVLM:
    def __init__(self):
        self.texture_extractor = TextureExtractor()  # 13-channel features
        self.knowing = ThreeWaysOfKnowing()  # Propositional, Perspectival, Participatory
        self.balancing = TensionBalancer()  # Opponent processing
        self.attending = TemporalAttentionAllocator()  # Frame-wise LOD

    def process_video(self, video, query):
        """
        Apply relevance realization to video understanding.

        Steps:
          1. Extract texture features per frame
          2. Score relevance per frame (3 ways of knowing)
          3. Balance temporal tensions
          4. Allocate LOD to each frame
          5. Process high-relevance frames densely, low-relevance sparsely
        """
        # Extract features per frame
        frame_features = []
        for frame in video:
            texture = self.texture_extractor(frame)  # 13 channels
            frame_features.append(texture)

        # Score relevance per frame
        frame_relevance = []
        for features in frame_features:
            scores = self.knowing.score_all(features, query)
            relevance = self.balancing.navigate_tensions(scores)
            frame_relevance.append(relevance)

        # Allocate tokens per frame based on relevance
        frame_budgets = self.attending.allocate_temporal_lod(frame_relevance)

        # Process each frame at allocated LOD
        processed_frames = []
        for features, budget in zip(frame_features, frame_budgets):
            # High-relevance frames: 400 tokens
            # Low-relevance frames: 64 tokens
            tokens = self.compress_to_budget(features, budget)
            processed_frames.append(tokens)

        # Temporal fusion
        video_representation = self.fuse_temporal(processed_frames)

        # LLM reasoning
        answer = self.llm(video_representation, query)

        return answer
```

**Temporal Opponent Processing**:

**Tensions**:
1. **Temporal Coverage vs Detail**: Sample many frames (coverage) or few frames deeply (detail)?
2. **Motion Focus vs Scene Context**: Attend to motion or static context?
3. **Local Dynamics vs Global Narrative**: Short-term actions or long-term story?

**Balance**:
```python
def temporal_tension_balancing(frames, query):
    """
    Navigate temporal tensions based on query.

    Query: "What is the person doing?"
      → Focus on motion (local dynamics)
      → High LOD on motion frames

    Query: "What is the overall story?"
      → Focus on scene context (global narrative)
      → Even LOD across key frames
    """
    if is_action_query(query):
        # Motion focus
        motion_frames = detect_high_motion(frames)
        return allocate_lod(motion_frames, budget=400)
    else:
        # Coverage focus
        keyframes = extract_keyframes(frames)
        return allocate_lod(keyframes, budget=200)
```

**Evaluation on Video Benchmarks**:

**Hypothesis**: ARR-COC's relevance-driven approach should:
1. Reduce token count (fewer frames at lower LOD)
2. Maintain accuracy (high LOD on relevant frames)
3. Improve efficiency (skip irrelevant temporal regions)

**Experimental Setup**:
- Benchmark: ActivityNet-QA, MSVD-QA, CinePile
- Baseline: Uniform 8-frame sampling, 256 tokens per frame (2,048 tokens total)
- ARR-COC: Relevance-driven sampling, 64-400 tokens per frame (adaptive)

**Expected Results**:
- 30-50% token reduction
- Comparable or better VQA accuracy
- 2-3× inference speedup

---

## Sources

**Source Documents:**
- [pyramid-lod/05-3d-volumetric-pyramids-video.md](../pyramid-lod/05-3d-volumetric-pyramids-video.md) - Spatiotemporal pyramids, temporal mipmaps, video ViT
- [vision-language/14-video-understanding-temporal-128k.md](../karpathy/vision-language/14-video-understanding-temporal-128k.md) - Long-context video, temporal attention, 128K tokens

**Web Research:**

**Video VLM Architectures:**
- [CogVLM2: Visual Language Models for Image and Video Understanding](https://arxiv.org/abs/2408.16500) - arXiv:2408.16500 (accessed 2025-11-16, 211 citations)
- [Towards Detailed Video Understanding via Large Vision and Language Models](https://aclanthology.org/2024.acl-long.679/) - ACL 2024 (accessed 2025-11-16, 1170 citations)
- [VILA: Visual Language Model Family](https://github.com/NVlabs/VILA) - NVIDIA Labs (accessed 2025-11-16)
- [Distilling Vision-Language Models on Millions of Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Distilling_Vision-Language_Models_on_Millions_of_Videos_CVPR_2024_paper.pdf) - CVPR 2024 (accessed 2025-11-16, 29 citations)

**Spatiotemporal Attention:**
- [MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs](https://arxiv.org/abs/2503.15871) - arXiv:2503.15871 (accessed 2025-11-16, 6 citations)
- [Temporal Attention for Video Understanding](https://medium.com/biased-algorithms/temporal-attention-for-video-understanding-ca6fa7c09409) - Medium (accessed 2025-11-16)
- [Long-Short Temporal Contrastive Learning of Video Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Long-Short_Temporal_Contrastive_Learning_of_Video_Transformers_CVPR_2022_paper.pdf) - CVPR 2022 (accessed 2025-11-16)
- [Enhancing Video-Language Representations with Structural Spatio-Temporal Alignment](https://arxiv.org/html/2406.19255v1) - arXiv:2406.19255 (accessed 2025-11-16)

**Frame Sampling Strategies:**
- [Frame Sampling Strategies Matter: A Benchmark for Small VLMs](https://arxiv.org/html/2509.14769v1) - arXiv:2509.14769 (accessed 2025-11-16)
- [MGSampler: An Explainable Sampling Strategy for Video Action Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.pdf) - ICCV 2021 (accessed 2025-11-16, 105 citations)
- [Generative Frame Sampler for Long Video Understanding](https://aclanthology.org/2025.findings-acl.921.pdf) - ACL 2025 Findings (accessed 2025-11-16, 6 citations)

**Video Question Answering Benchmarks:**
- [A Long Video Question Answering Dataset and Benchmark: CinePile](https://arxiv.org/abs/2405.08813) - arXiv:2405.08813 (accessed 2025-11-16, 81 citations)
- [CG-Bench: Clue-grounded Question Answering for Long Videos](https://openreview.net/forum?id=le4IoZZHy1) - OpenReview (accessed 2025-11-16, 30 citations)
- [LifeQA: A Real-Life Dataset for Video Question Answering](https://aclanthology.org/2020.lrec-1.536/) - LREC 2020 (accessed 2025-11-16, 47 citations)
- [Video-CoT: Spatiotemporal Reasoning in Video QA](https://dl.acm.org/doi/abs/10.1145/3746027.3758313) - ACM 2025 (accessed 2025-11-16, 6 citations)

**Efficient Video Processing:**
- [STORM: Token-Efficient Long Video Understanding](https://arxiv.org/abs/2503.04130) - arXiv:2503.04130 (accessed 2025-11-16)
- [Memory Consolidation for Long-Context Video Understanding](https://arxiv.org/html/2402.05861v2) - arXiv:2402.05861v2 (accessed 2025-11-16)
- [ResidualViT for Efficient Temporally Dense Video Encoding](https://arxiv.org/html/2509.13255v1) - arXiv:2509.13255 (accessed 2025-11-16)

**Additional References:**
- [First Workshop on Video-Language Models 2024](https://video-and-language-workshop-2024.webflow.io/) - NeurIPS 2024 (accessed 2025-11-16)
- [Video Understanding with Qwen2-VL](https://medium.com/@tenyks_blogger/qwen2-vl-expert-vision-language-model-for-video-understanding-db5da45560f3) - Medium (accessed 2025-11-16)
