# SAM 2 Memory Encoder: Streaming Attention Architecture

**"How SAM 2 Remembers and Tracks Objects Across Video Frames"**
**Innovation**: Streaming memory attention with temporal propagation
**Performance**: 30-44 FPS real-time video segmentation
**Key Insight**: Remember recent frames, forget distant past

---

## Overview

The **memory encoder** is SAM 2's secret weapon for video segmentation. It allows the model to:
- Remember objects across frames
- Handle occlusions and reappearances
- Stream video in real-time (no full-video processing)
- Maintain temporal consistency

### What Problem Does It Solve?

**Challenge**: Video segmentation requires tracking objects across time
- Object moves, changes shape, gets occluded
- Can't process entire video at once (memory constraints)
- Need real-time performance (30+ FPS)

**Solution**: Streaming memory with attention
- Store compressed representations of recent frames
- Attend to memory when segmenting current frame
- Forget old frames (sliding window)

---

## Architecture Components

### 1. Memory Bank

**Stores encoded representations of past frames**

```
Memory Bank = [frame_t-N, frame_t-N+1, ..., frame_t-1]
              ↑                                     ↑
              oldest frame                     most recent
```

**Properties:**
- **Capacity**: Stores N recent frames (typically 8-16)
- **Compression**: Each frame encoded to low-dim representation
- **FIFO**: Oldest frame dropped when adding new frame
- **Selective**: Only stores frames with user interaction OR important events

### 2. Memory Encoder Network

**Architecture:**
```
Current Frame → Hiera Encoder → Frame Features
                                       ↓
                                 Fuse with Mask
                                       ↓
                              Memory Representation
                                       ↓
                              Add to Memory Bank
```

**Key operations:**
1. **Encode frame**: Hiera extracts visual features
2. **Fuse with mask**: Combine features + predicted mask
3. **Compress**: Reduce dimensionality for storage
4. **Store**: Add to memory bank (FIFO)

### 3. Memory Attention Module

**Cross-attention between current frame and memory:**

```
Query:   Current frame features [H×W×D]
Keys:    Memory bank frames [N×H×W×D]
Values:  Memory bank frames [N×H×W×D]

Output:  Attended features [H×W×D]
         (current frame enhanced with memory)
```

**Attention mechanism:**
- **Spatial**: Attend to spatial locations in past frames
- **Temporal**: Weight recent frames higher than old frames
- **Object-centric**: Focus on regions with same object

---

## How Memory Attention Works

### Step-by-Step Process

**Frame t=0 (first frame):**
1. User clicks object → Prompt encoder processes click
2. Image encoder (Hiera) encodes frame
3. Mask decoder predicts mask
4. Memory encoder stores: frame + mask representation
5. Memory bank = [frame_0]

**Frame t=1:**
1. Encode frame_1 with Hiera
2. Cross-attend to memory bank [frame_0]
3. Mask decoder predicts mask (informed by frame_0)
4. Store frame_1 + mask
5. Memory bank = [frame_0, frame_1]

**Frame t=N (memory full):**
1. Encode frame_N
2. Cross-attend to memory bank [frame_0, ..., frame_N-1]
3. Predict mask
4. Drop frame_0 (oldest), add frame_N
5. Memory bank = [frame_1, ..., frame_N] (FIFO)

### Temporal Weighting

**Recent frames are more important:**

```
Attention weights = softmax(Q·K^T / sqrt(d))

With temporal decay:
Attention weights *= exp(-λ·Δt)
                     ↑      ↑
                     │      time since frame
                     decay rate
```

**Example:**
- frame_t-1: weight = 1.0× (most recent)
- frame_t-2: weight = 0.9×
- frame_t-3: weight = 0.8×
- ...
- frame_t-8: weight = 0.3× (oldest in memory)

---

## Handling Occlusions

### Problem

**Object disappears behind another object:**
- frame_t: Object visible
- frame_t+1 to frame_t+10: Object occluded (not visible)
- frame_t+11: Object reappears

### SAM 2 Solution

**Memory persistence:**

1. **Before occlusion** (frame_t):
   - Object segmented correctly
   - Stored in memory bank

2. **During occlusion** (frame_t+1 to frame_t+10):
   - Object not visible in current frame
   - But memory bank still contains frame_t representation
   - Model attends to memory → "remembers" object

3. **After occlusion** (frame_t+11):
   - Object reappears
   - Memory attention → high similarity to frame_t
   - Model recognizes: "This is the same object!"

**Key insight**: Memory acts as object identity storage during occlusions

---

## Memory Attention vs Standard Attention

### Standard Self-Attention (ViT)

```
Query:  Current frame patches
Keys:   Current frame patches
Values: Current frame patches

→ Only sees current frame (no temporal info)
```

### SAM 2 Memory Attention

```
Query:  Current frame features
Keys:   Memory bank (past frames)
Values: Memory bank (past frames)

→ Sees current + past frames (temporal context)
```

**Advantage**: Object tracking, occlusion handling, temporal consistency

---

## Streaming Video Processing

### How SAM 2 Achieves Real-Time Performance

**Key design choice: Streaming (not batch)**

**Batch processing (prior methods):**
```
Load entire video → Process all frames → Output masks
Problem: Requires full video in memory (slow, memory-hungry)
```

**Streaming processing (SAM 2):**
```
Process frame 1 → Store in memory
Process frame 2 → Attend to memory → Store
Process frame 3 → Attend to memory → Store
...

→ Only current frame + memory bank in memory
→ Can process indefinitely long videos
→ Real-time (30-44 FPS)
```

### Memory Budget

**Typical memory bank size:**
- **Small**: 8 frames (low memory, ~2GB VRAM)
- **Medium**: 12 frames (balanced, ~4GB VRAM)
- **Large**: 16 frames (best quality, ~6GB VRAM)

**Trade-offs:**
- More frames → Better long-term consistency
- Fewer frames → Lower memory, faster inference

---

## Technical Details

### Memory Encoder Architecture

**Network layers:**
```
Input: Frame features [H×W×D] + Mask [H×W×1]

1. Concatenate: [H×W×(D+1)]
2. Conv 3×3: [H×W×D']
3. LayerNorm
4. Self-attention (within frame)
5. FFN (feedforward network)
6. Downsample: [H/2×W/2×D'']  ← Compression!

Output: Memory representation [H/2×W/2×D'']
```

**Compression ratio**: 4× spatial reduction (H×W → H/2×W/2)

### Memory Attention Module

**Cross-attention implementation:**
```python
# Pseudocode
class MemoryAttention:
    def forward(self, current_features, memory_bank):
        # current_features: [B, H, W, D]
        # memory_bank: [B, N, H', W', D']

        # Reshape for attention
        Q = current_features.reshape(B, H*W, D)
        K = memory_bank.reshape(B, N*H'*W', D')
        V = memory_bank.reshape(B, N*H'*W', D')

        # Attention
        attn_weights = softmax(Q @ K.T / sqrt(D))
        attended = attn_weights @ V

        # Reshape back
        output = attended.reshape(B, H, W, D)
        return output
```

**Efficiency:**
- **Linear attention**: O(N·H·W) instead of O((N·H·W)²)
- **Sparse attention**: Only attend to top-k memory frames
- **Flash attention**: Fused kernels for speed

---

## Comparison with Other Temporal Models

### RNNs/LSTMs

**Traditional sequential models:**
```
h_t = RNN(h_{t-1}, x_t)
      ↑           ↑
      hidden state   current input
```

**Problems:**
- **Sequential**: Can't parallelize (slow)
- **Forgetting**: Long-term dependencies decay
- **Fixed capacity**: Hidden state size fixed

### SAM 2 Memory Attention

**Advantages:**
- **Parallel**: All memory frames attended in parallel
- **Explicit memory**: Store actual frames (no compression to hidden state)
- **Flexible**: Add/remove memory frames dynamically

---

## Training the Memory Encoder

### Training Objectives

**Multi-task learning:**

1. **Mask prediction loss**:
   - Focal + Dice loss
   - Predict accurate mask for current frame

2. **Temporal consistency loss**:
   - Penalize sudden mask changes
   - Smooth tracking across frames

3. **Memory reconstruction loss** (optional):
   - Reconstruct past frame from memory
   - Ensures memory stores useful info

### Training Data

**SA-V dataset (50.9k videos, 642k masklets):**
- Sample temporal windows (8-16 frames)
- Simulate occlusions (drop frames randomly)
- Train memory attention to track objects

### Training Strategy

**Curriculum learning:**
1. **Stage 1**: Short videos (4 frames) → Learn basic tracking
2. **Stage 2**: Medium videos (8 frames) → Learn occlusions
3. **Stage 3**: Long videos (16 frames) → Learn long-term memory

---

## Performance Metrics

### Inference Speed

**Memory attention overhead:**
- **Hiera encoder**: 15ms per frame (H100)
- **Memory attention**: +3ms per frame
- **Mask decoder**: 5ms per frame
- **Total**: 23ms → 43.5 FPS

**Memory bank size impact:**
- 8 frames: 44 FPS
- 12 frames: 38 FPS
- 16 frames: 30 FPS

### Memory Usage

**VRAM consumption:**
- **Hiera-B+ (256M params)**: 1.2GB (model weights)
- **Memory bank (8 frames)**: +1.8GB
- **Memory bank (16 frames)**: +3.6GB

**Total**: 3-5GB VRAM (fits on consumer GPUs!)

---

## Key Innovations

### 1. Streaming Memory

**First real-time video segmentation model:**
- Prior work: Batch processing (slow)
- SAM 2: Streaming (real-time)

### 2. Occlusion Handling

**Explicit memory-based occlusion reasoning:**
- Prior work: Optical flow (fails on occlusions)
- SAM 2: Memory attention (handles occlusions)

### 3. Flexible Memory Bank

**Dynamic memory management:**
- Add frames when needed (user interaction)
- Drop frames when memory full (FIFO)
- No fixed video length requirement

---

## Future Directions

### Potential Improvements

**1. Hierarchical memory:**
- Store recent frames at high resolution
- Store distant frames at low resolution
- Multi-scale temporal attention

**2. Memory compression:**
- Vector quantization (VQ-VAE)
- Reduce memory footprint further

**3. Long-term memory:**
- Store key frames from entire video
- Not just recent N frames

**4. Associative memory:**
- Neural Turing Machine-style content-based addressing
- Retrieve most relevant frames (not just recent)

---

## Key Takeaways

1. **Streaming memory**: SAM 2 processes video frame-by-frame with FIFO memory bank
2. **Cross-attention**: Current frame attends to past frames for temporal consistency
3. **Occlusion handling**: Memory persists object representation during occlusions
4. **Real-time**: 30-44 FPS on H100/A100 (memory overhead only +3ms per frame)
5. **Flexible**: Works on indefinitely long videos (no full-video requirement)

**The memory encoder is what makes SAM 2 a video segmentation foundation model (not just image segmentation).**

---

## References

- SAM 2 Paper: "SAM 2: Segment Anything in Images and Videos" (arXiv 2024)
- Memory attention: Section 3.2 of paper
- Streaming architecture: Figure 3 of paper
- Meta AI Blog: https://ai.meta.com/sam2/
- GitHub: https://github.com/facebookresearch/sam2
