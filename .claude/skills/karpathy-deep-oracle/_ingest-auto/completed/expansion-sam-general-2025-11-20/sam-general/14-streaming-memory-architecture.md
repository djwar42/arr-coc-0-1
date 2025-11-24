# SAM 2 Streaming Memory Architecture

## Overview

SAM 2's streaming memory architecture is the core innovation enabling real-time video segmentation with temporal consistency. Unlike traditional approaches that require access to all video frames simultaneously, SAM 2 processes frames sequentially while maintaining a memory of past predictions and user prompts.

**Key Innovation**: Memory-conditioned frame processing that enables:
- Real-time streaming video segmentation (44 FPS on A100)
- Temporal consistency across frames
- Occlusion handling and re-identification
- Interactive refinement at any frame

**Architecture Philosophy**: The streaming approach treats video as a sequence of frames processed one-at-a-time, with each frame's segmentation conditioned on:
1. Current frame features (from image encoder)
2. Memories of past frames and predictions
3. User prompts (current and historical)
4. Object pointer vectors for semantic consistency

---

## Section 1: Memory Architecture Overview

### The Memory-Conditioned Pipeline

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714) (arXiv:2408.00714):

```
Video Frame t
    |
    v
Image Encoder (Hiera) --> Frame Embedding
    |
    v
Memory Attention Module
    |-- Cross-attention to Memory Bank
    |-- Cross-attention to Object Pointers
    |
    v
Conditioned Frame Embedding
    |
    v
Mask Decoder (+ optional prompts)
    |
    v
Segmentation Mask
    |
    v
Memory Encoder --> Memory Bank (for frame t+1)
```

### Key Architectural Components

**1. Image Encoder (Hiera)**
- Hierarchical Vision Transformer (MAE pre-trained)
- Produces unconditioned frame embeddings
- Multi-scale features for high-resolution decoding
- Run once per frame (streaming efficiency)

**2. Memory Attention Module**
- L=4 transformer blocks (default)
- Self-attention on current frame features
- Cross-attention to memory bank contents
- Cross-attention to object pointers
- Uses 2D spatial RoPE for positional encoding

**3. Memory Encoder**
- Fuses predicted mask with frame embedding
- Downsamples mask via convolutional module
- Creates spatial memory features (64-dim projection)

**4. Memory Bank**
- FIFO queue of recent frame memories (N frames)
- FIFO queue of prompted frame memories (M frames)
- Object pointer storage for semantic information

### Why Streaming Matters

**Traditional Approaches (Non-Streaming)**:
- Require all video frames in memory
- O(T^2) attention complexity for T frames
- Cannot handle arbitrarily long videos
- High latency before first prediction

**SAM 2 Streaming Approach**:
- Process frames as they arrive
- O(N) memory complexity (fixed memory bank size)
- Supports arbitrarily long videos
- Immediate response on each frame

### Memory Attention Mechanism

The memory attention conditions current frame features on temporal context:

```python
class MemoryAttention(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=256):
        self.layers = nn.ModuleList([
            MemoryAttentionBlock(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, frame_embedding, memory_bank, object_pointers):
        x = frame_embedding

        for layer in self.layers:
            # Self-attention on current frame
            x = layer.self_attention(x)

            # Cross-attention to spatial memories
            x = layer.cross_attention(
                query=x,
                key=memory_bank.spatial_features,
                value=memory_bank.spatial_features
            )

            # Cross-attention to object pointers
            x = layer.cross_attention(
                query=x,
                key=object_pointers,
                value=object_pointers
            )

            # MLP
            x = layer.mlp(x)

        return x  # Conditioned frame embedding
```

### Positional Encoding Strategy

**Absolute Positional Encoding**:
- Windowed absolute positional embeddings in image encoder
- Global positional embedding interpolated across windows
- No relative positional biases (RPB) - enables FlashAttention-2

**2D Spatial RoPE**:
- Applied in memory attention self- and cross-attention
- Captures spatial relationships across frames
- Object pointers excluded (no specific spatial correspondence)

**Temporal Encoding**:
- Embedded into recent frame memories
- Represents short-term object motion
- NOT applied to prompted frames (generalization concerns)

---

## Section 2: Memory Bank Design

### Memory Bank Structure

The memory bank is the central data structure for temporal reasoning:

```python
class MemoryBank:
    def __init__(self, max_recent=6, max_prompted=8):
        # Recent unprompted frames (FIFO)
        self.recent_memories = deque(maxlen=max_recent)

        # Prompted frames (FIFO)
        self.prompted_memories = deque(maxlen=max_prompted)

        # Object pointers for semantic consistency
        self.object_pointers = []

    def add_memory(self, memory_features, object_pointer, is_prompted):
        if is_prompted:
            self.prompted_memories.append(memory_features)
        else:
            self.recent_memories.append(memory_features)

        self.object_pointers.append(object_pointer)

    def get_all_memories(self):
        return list(self.recent_memories) + list(self.prompted_memories)
```

### Memory Types

**1. Recent Frame Memories (N=6 default)**
- Unprompted frames from recent processing
- FIFO queue with automatic replacement
- Contains temporal position information
- Represents short-term object appearance/motion

**2. Prompted Frame Memories (M=8 default)**
- Frames where user provided prompts
- Always retained (until queue full)
- NO temporal position information (generalization)
- Anchors for object identity

**3. Object Pointers**
- Lightweight vectors (256-dim split to 4x64-dim)
- Based on mask decoder output tokens
- High-level semantic information
- Cross-attended alongside spatial features

### Memory Feature Generation

The memory encoder creates features by fusing predictions with frame context:

```python
class MemoryEncoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=64):
        # Mask downsampling
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, 1)
        )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )

    def forward(self, frame_embedding, predicted_mask):
        # Downsample mask to match frame embedding resolution
        mask_embedding = self.mask_conv(predicted_mask)

        # Element-wise sum with frame embedding
        fused = frame_embedding + mask_embedding

        # Apply fusion convolutions
        memory = self.fusion(fused)

        return memory  # (B, 64, H/16, W/16)
```

### Memory Capacity Trade-offs

From ablation studies in the SAM 2 paper:

| # Memories | MOSE dev | SA-V val | Speed |
|------------|----------|----------|-------|
| 4 | 73.5 | 68.6 | 1.01x |
| 6 | 73.0 | 68.3 | 1.00x |
| 8 | 73.2 | 69.0 | 0.93x |

**Findings**:
- More memories generally help but with variance
- N=6 provides good balance of context and compute
- Diminishing returns beyond 8 memories

### Memory Projection Efficiency

Memory features are projected to lower dimension for efficiency:
- Full feature dimension: 256
- Memory projection: 64
- Object pointer split: 256-dim → 4 tokens of 64-dim

This 4x compression reduces memory attention cost while preserving important information.

### VOS Task Memory Configuration

For semi-supervised VOS (mask on first frame):
- First frame memory always retained (prompted)
- Up to N=6 recent frames in FIFO queue
- First frame provides object identity anchor
- Recent frames provide temporal context

---

## Section 3: Temporal Propagation

### Propagation Mechanism

SAM 2 propagates segmentation through video using memory-conditioned inference:

```python
def propagate_in_video(predictor, video_path, initial_prompts):
    """Propagate object segmentation through entire video."""

    # Initialize video state
    state = predictor.init_state(video_path)

    # Add initial prompts (e.g., clicks on frame 0)
    for prompt in initial_prompts:
        predictor.add_prompt(
            state,
            frame_idx=prompt.frame,
            obj_id=prompt.obj_id,
            points=prompt.points,
            labels=prompt.labels
        )

    # Propagate through all frames
    video_segments = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        video_segments[frame_idx] = {
            obj_id: masks[i]
            for i, obj_id in enumerate(obj_ids)
        }

    return video_segments
```

### Forward and Backward Propagation

SAM 2 supports bi-directional propagation:

**Forward Propagation** (default):
- Process frames 0 → T
- Memory contains past frames
- Natural for streaming video

**Backward Propagation**:
- Process frames T → 0
- Useful when prompt is on later frame
- Training uses 50% temporal reversal for robustness

### Multi-Object Tracking

SAM 2 processes multiple objects independently:

```python
# Track multiple objects simultaneously
state = predictor.init_state(video_path)

# Add prompts for different objects
predictor.add_new_points(state, frame_idx=0, obj_id=1,
                         points=[[100, 200]], labels=[1])
predictor.add_new_points(state, frame_idx=0, obj_id=2,
                         points=[[300, 400]], labels=[1])

# Each object has its own memory bank
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # masks[0] = object 1 mask
    # masks[1] = object 2 mask
    pass
```

**Current Limitation**: Objects processed separately without inter-object communication. Future work could add object-level context.

### Ambiguity Resolution in Video

When a single click could refer to multiple valid objects:

1. **Initial Prediction**: Output multiple mask candidates
2. **Propagation**: Select highest predicted IoU mask
3. **Refinement**: User can add prompts to resolve ambiguity

```python
# Ambiguous prompt (single click)
masks, scores, _ = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1],
    multimask_output=True  # Returns 3 candidates
)

# For propagation, use best scoring mask
best_idx = np.argmax(scores)
selected_mask = masks[best_idx]
```

### Temporal Consistency Mechanisms

**1. Memory Attention**:
- Cross-attention to past predictions
- Maintains object appearance model
- Smooths predictions across frames

**2. Object Pointers**:
- High-level semantic anchors
- Help re-identify objects after occlusion
- Provide identity consistency

**3. Prompted Frame Retention**:
- Always keep initial prompt frame
- User corrections become permanent anchors
- Prevents drift from intended object

---

## Section 4: Memory Encoding

### Frame Embedding Creation

The Hiera image encoder produces hierarchical features:

```python
class HieraImageEncoder(nn.Module):
    def __init__(self):
        # Four stages with different resolutions
        self.stage1 = HieraBlock(...)  # stride 4
        self.stage2 = HieraBlock(...)  # stride 8
        self.stage3 = HieraBlock(...)  # stride 16
        self.stage4 = HieraBlock(...)  # stride 32

        # FPN to fuse stage 3 and 4
        self.fpn = FeaturePyramidNetwork()

    def forward(self, image):
        # Hierarchical features
        f1 = self.stage1(image)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        # FPN fusion for memory attention
        frame_embedding = self.fpn(f3, f4)  # stride 16

        # Return all for skip connections
        return {
            'embedding': frame_embedding,
            'skip_4': f1,
            'skip_8': f2
        }
```

### Memory Encoding Process

After mask prediction, create memory for future frames:

```python
def create_memory(frame_embedding, predicted_mask, memory_encoder):
    """Create memory features from prediction."""

    # Ensure mask is correct resolution
    mask_resized = F.interpolate(
        predicted_mask,
        size=frame_embedding.shape[-2:],
        mode='bilinear'
    )

    # Encode into memory
    memory_features = memory_encoder(frame_embedding, mask_resized)

    return memory_features  # (B, 64, H/16, W/16)
```

### Object Pointer Extraction

Object pointers come from mask decoder output tokens:

```python
class MaskDecoder(nn.Module):
    def forward(self, frame_embedding, prompts):
        # ... transformer decoding ...

        # Get mask and IoU predictions
        masks = self.mask_head(mask_tokens, frame_embedding)
        ious = self.iou_head(iou_token)

        # Object pointer from mask token
        object_pointer = mask_tokens[best_mask_idx]  # 256-dim

        return masks, ious, object_pointer
```

### Memory Feature Dimensions

| Component | Dimension | Purpose |
|-----------|-----------|---------|
| Frame embedding | 256 x H/16 x W/16 | Spatial features |
| Memory features | 64 x H/16 x W/16 | Compressed spatial |
| Object pointer | 256 (split to 4x64) | Semantic anchor |

The 4x compression (256 → 64) significantly reduces memory attention cost while experiments show minimal quality loss.

### Skip Connections for Detail

High-resolution details bypass memory attention:

```python
# In mask decoder
def decode_with_skips(self, conditioned_embedding, skip_4, skip_8):
    # Upsample from stride 16 to stride 8
    x = self.upsample_1(conditioned_embedding)
    x = x + skip_8  # Add stride 8 features

    # Upsample from stride 8 to stride 4
    x = self.upsample_2(x)
    x = x + skip_4  # Add stride 4 features

    # Final upsample to original resolution
    mask = self.final_conv(x)
    return mask
```

This preserves fine details (edges, thin structures) that might be lost in memory attention.

---

## Section 5: Occlusion Handling

### Occlusion Prediction Head

SAM 2 explicitly predicts whether the object is visible:

```python
class MaskDecoder(nn.Module):
    def __init__(self):
        # ... existing components ...

        # Occlusion prediction token and head
        self.occlusion_token = nn.Embedding(1, 256)
        self.occlusion_head = MLP(256, 256, 1)

    def forward(self, frame_embedding, prompts):
        # Add occlusion token to decoder
        tokens = torch.cat([
            self.mask_tokens.weight,
            self.iou_token.weight,
            self.occlusion_token.weight,  # NEW
            prompt_embeddings
        ], dim=0)

        # Transformer decoding
        tokens, _ = self.transformer(tokens, frame_embedding)

        # Extract predictions
        masks = self.mask_head(tokens[:3], frame_embedding)
        ious = self.iou_head(tokens[3])
        occlusion_score = self.occlusion_head(tokens[4])

        return masks, ious, occlusion_score
```

### Handling Occluded Frames

When object is predicted as occluded:

1. **No mask output** for that frame
2. **Memory still updated** (empty/occluded state)
3. **Re-identification** when object reappears

```python
def process_frame(predictor, frame, memory_bank):
    masks, ious, occlusion = predictor.predict(frame)

    if occlusion > 0.5:  # Object likely occluded
        # Skip mask but track occlusion state
        memory_bank.add_occlusion_marker()
        return None
    else:
        # Normal processing
        memory = create_memory(frame, masks[0])
        memory_bank.add_memory(memory)
        return masks[0]
```

### Re-identification After Occlusion

Object pointers enable re-identification:

```python
# During inference after occlusion
def find_reappearing_object(current_features, object_pointers):
    """Use object pointers to re-identify object."""

    # Cross-attention to historical object pointers
    attended = cross_attention(
        query=current_features,
        key=object_pointers,
        value=object_pointers
    )

    # Object pointer provides semantic anchor
    # Even if appearance changed, semantic identity persists
    return attended
```

### Disappearance Rate in Training Data

SA-V dataset has high disappearance rate (42.5%):
- Objects frequently leave and re-enter frame
- Model trained on realistic occlusion patterns
- Contrast: YouTube-VOS only 13.0%

This training diversity enables robust occlusion handling.

### Occlusion vs Lost Track

Important distinction:

**Occlusion** (temporary):
- Object physically blocked
- Occlusion score high
- Memory maintains object state
- Automatic re-identification

**Lost Track** (error):
- Model failure
- Requires user intervention
- Refinement prompt on new frame

---

## Section 6: Long Video Support

### Streaming Architecture Benefits

The streaming design naturally supports arbitrarily long videos:

```python
# Process video of any length
def stream_video(predictor, video_generator, prompts):
    """Stream processing for unlimited length videos."""

    state = predictor.init_state()

    # Add initial prompts
    for p in prompts:
        predictor.add_prompt(state, p)

    # Process frames as they arrive
    for frame_idx, frame in enumerate(video_generator):
        # Memory bank automatically manages capacity
        mask = predictor.process_frame(state, frame_idx, frame)

        # Yield immediately (real-time)
        yield frame_idx, mask

        # Memory bank FIFO removes old frames
        # No memory growth with video length
```

### Memory Bank FIFO Management

Fixed memory capacity prevents unbounded growth:

```python
class MemoryBank:
    def add_memory(self, memory, is_prompted):
        if is_prompted:
            if len(self.prompted_memories) >= self.max_prompted:
                self.prompted_memories.popleft()  # FIFO
            self.prompted_memories.append(memory)
        else:
            if len(self.recent_memories) >= self.max_recent:
                self.recent_memories.popleft()  # FIFO
            self.recent_memories.append(memory)
```

### Long-Term Video Object Segmentation

Performance on LVOS (Long Video Object Segmentation) benchmark:

| Method | LVOS val J&F |
|--------|--------------|
| Cutie-base | 66.0 |
| DEVA | 55.9 |
| SAM 2 (Hiera-B+) | 74.9 |
| SAM 2 (Hiera-L) | 76.1 |

SAM 2 shows **+10% improvement** on long video benchmark.

### Challenges in Long Videos

**1. Appearance Drift**:
- Object appearance changes over time
- Solution: Object pointers maintain semantic identity

**2. Similar Objects**:
- Multiple similar objects over time
- Solution: Prompted frame retention as anchors

**3. Memory Limitations**:
- Recent 6 frames may miss important context
- Solution: Prompted frames always retained

### SAM2Long Enhancement

Research extension for even longer videos (from [SAM2Long paper](https://mark12ding.github.io/project/SAM2Long/)):

```python
# SAM2Long adds constrained memory selection
def select_memories(memory_bank, current_features):
    """Select most relevant memories for long videos."""

    # Compute relevance scores
    scores = []
    for mem in memory_bank.all_memories:
        score = compute_similarity(current_features, mem)
        scores.append(score)

    # Select top-k most relevant (not just recent)
    selected = select_top_k(memory_bank.all_memories, scores, k=N)

    return selected
```

This allows selective memory use instead of strict FIFO for better long-term tracking.

---

## Section 7: ARR-COC Integration

### Implementing Memory-Aware Video Models

For ARR-COC training pipelines, key considerations:

**1. Memory-Efficient Training**:
```python
# Training with limited memory
class MemoryEfficientVideoTrainer:
    def __init__(self, max_frames=8, memory_size=6):
        self.max_frames = max_frames
        self.memory_size = memory_size

    def train_step(self, video_batch):
        # Sample fixed-length sequences
        sequences = self.sample_sequences(video_batch, self.max_frames)

        # Gradient checkpointing for memory efficiency
        with torch.cuda.amp.autocast():
            for seq in sequences:
                loss = self.model.forward_with_memory(seq)
                loss.backward()
```

**2. Streaming Inference Implementation**:
```python
# Real-time streaming inference
class StreamingPredictor:
    def __init__(self, model, memory_config):
        self.model = model
        self.memory_bank = MemoryBank(**memory_config)
        self.frame_buffer = []

    @torch.inference_mode()
    def process_stream(self, frame_generator):
        for frame in frame_generator:
            # Encode frame
            embedding = self.model.image_encoder(frame)

            # Condition on memory
            conditioned = self.model.memory_attention(
                embedding,
                self.memory_bank
            )

            # Predict mask
            mask = self.model.mask_decoder(conditioned)

            # Update memory
            memory = self.model.memory_encoder(embedding, mask)
            self.memory_bank.add_memory(memory)

            yield mask
```

### Training Configuration

Key hyperparameters from SAM 2 paper:

```python
training_config = {
    # Sequence sampling
    'num_frames': 8,
    'max_prompted_frames': 2,

    # Memory configuration
    'num_memories': 6,
    'memory_dim': 64,
    'pointer_dim': 256,

    # Interactive training
    'corrective_click_prob': 0.1,
    'initial_prompt_type': {
        'mask': 0.5,
        'click': 0.25,
        'box': 0.25
    },

    # Data augmentation
    'temporal_reversal_prob': 0.5,
    'max_masklets_per_sequence': 3,
}
```

### Multi-GPU Considerations

For distributed training with memory states:

```python
# Each GPU maintains its own memory bank
class DistributedVideoModel(nn.Module):
    def __init__(self):
        self.model = SAM2Model()
        # Memory bank is per-GPU, not synchronized
        self.memory_bank = MemoryBank()

    def forward(self, sequence):
        # Reset memory at sequence start
        self.memory_bank.clear()

        losses = []
        for frame_idx, (frame, gt_mask) in enumerate(sequence):
            pred_mask = self.model(frame, self.memory_bank)
            loss = self.compute_loss(pred_mask, gt_mask)
            losses.append(loss)

            # Update memory (local to this GPU)
            self.update_memory(frame, pred_mask)

        return sum(losses) / len(losses)
```

### Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Inference Speed | 44 FPS | A100 GPU, Hiera-B+ |
| Memory per Frame | ~64 KB | 64-dim features at H/16 x W/16 |
| Max Video Length | Unlimited | Streaming architecture |
| Interaction Reduction | 3x fewer | vs prior interactive methods |

---

## Sources

### Primary Sources

**SAM 2 Paper**:
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - arXiv:2408.00714 (August 2024)
- Authors: Nikhila Ravi, Valentin Gabeur, et al. (Meta AI)

**GitHub Repository**:
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) - Official implementation

### Web Research

- [Papers Explained 239: SAM 2](https://ritvik19.medium.com/papers-explained-239-sam-2-6ffb7f187281) - Ritvik Rastogi (Medium, accessed 2025-11-20)
- [Ultralytics SAM 2 Documentation](https://docs.ultralytics.com/models/sam-2/) - Technical reference
- [Emergent Mind SAM-2 Topic](https://www.emergentmind.com/topics/segment-anything-model-2-sam-2) - Research overview (accessed 2025-11-20)

### Additional References

- [Encord SAM 2 Guide](https://encord.com/blog/segment-anything-model-2-sam-2/) - Technical deep dive
- [SAM2Long Paper](https://mark12ding.github.io/project/SAM2Long/) - Long video extension
- [Meta AI SAM 2](https://ai.meta.com/sam2/) - Official project page

### Related SAM Research

- [Segment Anything (SAM 1)](https://arxiv.org/abs/2304.02643) - Original foundation
- [Hiera: A Hierarchical Vision Transformer](https://arxiv.org/abs/2306.00989) - Image encoder architecture
