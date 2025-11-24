# SAM 3 Tracker Architecture (Inherited from SAM 2)

## Overview

SAM 3's tracker component directly inherits the memory-based video segmentation architecture from SAM 2. The tracker is responsible for temporal propagation of segmentation masks across video frames, using streaming memory attention to maintain object identity over time. In SAM 3's decoupled detector-tracker design, the tracker inherits the full SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement.

## Core Architecture Components

### 1. Streaming Memory Architecture

SAM 2's streaming architecture processes video frames one at a time as they become available, making it suitable for real-time video processing:

**Key Design Principles:**
- Frames processed sequentially in a streaming fashion
- Image encoder runs once per frame to produce unconditioned tokens
- Memory attention conditions current frame on past frames and predictions
- Natural generalization of SAM's image architecture to video domain

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714) (arXiv:2408.00714, accessed 2025-11-23):
- "We adopt a streaming architecture, which is a natural generalization of SAM to the video domain, processing video frames one at a time"
- "SAM 2 is equipped with a memory that stores information about the object and previous interactions"

### 2. Memory Attention Mechanism

The memory attention module is the key innovation that enables temporal propagation:

**Architecture:**
- Stack of L transformer blocks (default L=4)
- Each block performs:
  1. Self-attention on current frame features
  2. Cross-attention to memories from previous frames
  3. Cross-attention to object pointers
  4. MLP processing

**Positional Encoding:**
- Sinusoidal absolute positional embeddings
- 2D spatial Rotary Positional Embedding (RoPE) in attention layers
- Object pointer tokens excluded from RoPE (no spatial correspondence)

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714):
- "The memory attention operation takes the per-frame embedding from the image encoder and conditions it on the memory bank to produce an embedding that is then passed to the mask decoder"

### 3. Memory Bank System

The memory bank maintains temporal context for tracking:

**Components:**
1. **Recent Frame Memories** - FIFO queue of up to N recent frames (default N=6)
2. **Prompted Frame Memories** - FIFO queue of up to M prompted frames
3. **Object Pointers** - Lightweight vectors for high-level semantic information

**Memory Features:**
- Stored as spatial feature maps
- Projected to 64-dim for efficient cross-attention
- Object pointers (256-dim) split into 4 tokens of 64-dim

**Temporal Encoding:**
- Temporal position embedded in recent frame memories
- Enables modeling short-term object motion
- Prompted frames do NOT have temporal encoding (sparser training signal)

### 4. Memory Encoder

Generates memories from predictions:

**Process:**
1. Downsample output mask using convolutional module
2. Sum element-wise with unconditioned frame embedding from image encoder
3. Apply light-weight convolutional layers to fuse information
4. Store resulting memory features in memory bank

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714):
- "The memory encoder generates a memory by downsampling the output mask using a convolutional module and summing it element-wise with the unconditioned frame embedding"

## Temporal Propagation

### How Objects Are Tracked Across Frames

1. **Initial Prompt** - User provides click/box/mask on any frame
2. **Mask Generation** - Decoder produces segmentation mask for prompted frame
3. **Memory Creation** - Memory encoder creates memory from prediction
4. **Forward Propagation** - Process subsequent frames:
   - Image encoder produces frame embeddings
   - Memory attention conditions on stored memories
   - Decoder predicts mask for current frame
   - New memory stored in bank
5. **Backward Propagation** - Same process for earlier frames
6. **Refinement** - Additional prompts can correct errors

### Handling Occlusions and Reappearance

SAM 2 includes mechanisms for challenging video scenarios:

**Occlusion Prediction:**
- Additional output head predicts if object is visible in current frame
- Occlusion token processed by MLP to produce visibility score
- Allows model to "skip" frames where object is occluded

**Object Reappearance:**
- Memory bank retains object information during occlusion
- When object reappears, memory attention can re-identify it
- Object pointers provide high-level semantic matching

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714):
- "In the PVS task it is possible for no valid object to exist on some frames (e.g. due to occlusion). To account for this new output mode, we add an additional head that predicts whether the object of interest is present on the current frame"

## Interactive Refinement with Points

### Refinement Workflow

SAM 3 inherits SAM 2's powerful interactive refinement capability:

1. **Initial Segmentation** - Prompt model on any frame
2. **Automatic Propagation** - Model generates masklet for entire video
3. **Error Detection** - User identifies frames with errors
4. **Refinement Prompts** - Single click often sufficient to correct
5. **Memory-Aware Update** - Refinement uses existing object memory
6. **Re-propagation** - Corrected mask propagates to other frames

**Key Advantage Over Decoupled SAM+Tracker:**
- SAM 2/3 refinement uses memory context
- Single click can recover lost object
- Decoupled approach requires re-annotating from scratch

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714), Figure 2 example:
- "Step 2 (refinement): a single click in frame 3 is sufficient to recover the object and propagate it to obtain the correct masklet"
- "A decoupled SAM + video tracker approach would require several clicks in frame 3 (as in frame 1) to correctly re-annotate the object"

### Multi-Mask Ambiguity Handling

When prompts are ambiguous (e.g., single click):

1. **Multiple Mask Prediction** - Model outputs multiple valid masks per frame
2. **Ambiguity Propagation** - Ambiguity can extend across video frames
3. **IoU Selection** - If unresolved, select mask with highest predicted IoU
4. **Refinement Resolution** - Additional prompts can resolve ambiguity

## Image Encoder Details

The shared vision encoder for SAM 3's tracker:

**Architecture:**
- MAE pre-trained Hiera (Hierarchical Vision Transformer)
- Feature Pyramid Network (FPN) for multi-scale features
- Fuses stride 16 and 32 features from Stages 3 and 4

**Multi-Scale Features for Decoding:**
- Stride 4 features (Stage 1) - High-resolution details
- Stride 8 features (Stage 2) - Mid-level features
- Added to upsampling layers in mask decoder

**Encoder Sizes:**
- Tiny (T), Small (S), Base+ (B+), Large (L)
- Global attention in subset of layers for efficiency

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714):
- "We use an MAE pre-trained Hiera image encoder, which is hierarchical, allowing us to use multiscale features during decoding"

## Prompt Encoder and Mask Decoder

### Prompt Encoder

Identical to SAM's design:
- Sparse prompts: Positional encodings + learned embeddings per prompt type
- Mask prompts: Convolutions + sum with frame embedding
- Supports clicks (positive/negative), bounding boxes, masks

### Mask Decoder

Modified from SAM for video:
- "Two-way" transformer blocks update prompt and frame embeddings
- Skip connections from hierarchical image encoder
- Multi-mask output for ambiguous prompts
- Occlusion prediction head (new for video)
- Object pointer output for memory bank

## Performance Characteristics

### Speed Benchmarks

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714):
- Hiera-B+: 43.8 FPS on A100 GPU (real-time)
- Hiera-L: 30.2 FPS on A100 GPU (real-time)
- 6x faster than SAM on images

### Memory Efficiency

Design choices for efficiency:
- Memory features projected to 64-dim
- Configurable memory bank size (default N=6)
- No recurrent GRU needed (simpler, equally effective)
- FlashAttention-2 compatible (no RPB in image encoder)

### Accuracy

Semi-supervised VOS results (SAM 2 Hiera-L):
- MOSE val: 77.2 J&F
- DAVIS 2017 val: 91.6 J&F
- SA-V test: 77.6 J&F

## What SAM 3 Inherits vs. Adds

### Inherited from SAM 2 (Tracker)

1. **Full streaming memory architecture**
2. **Memory attention mechanism (L=4 transformer blocks)**
3. **Memory bank system (recent + prompted + object pointers)**
4. **Memory encoder design**
5. **Temporal propagation logic**
6. **Interactive refinement workflow**
7. **Occlusion handling**
8. **Multi-mask ambiguity resolution**
9. **Prompt encoder (identical to SAM)**
10. **Mask decoder (with video modifications)**

### SAM 3 Additions (Detector)

1. **Text-conditioned DETR detector** - NEW
2. **Presence token mechanism** - NEW
3. **Open-vocabulary prompting** - NEW
4. **270K concept support** - NEW
5. **Decoupled detector-tracker design** - NEW ARCHITECTURE

From [SAM 3 OpenReview Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf):
- "The tracker inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement"

## Training Details for Tracker

### Training Strategy

- Joint training on image and video data
- Simulated interactive prompting
- Sequences of 8 frames sampled
- Up to 2 frames prompted per sequence
- Corrective clicks sampled during training

### Prompt Simulation

Initial prompts probability:
- Ground-truth mask: 50%
- Positive click: 25%
- Bounding box: 25%

### Losses

- Segmentation loss (focal + dice)
- IoU prediction loss (L1 with sigmoid activation)
- Occlusion prediction loss

## Key Ablation Insights

From SAM 2 paper ablations:

### Memory Design

- **Object pointers help** - Significant boost on SA-V val and long videos
- **GRU not needed** - Direct memory storage equally effective, simpler
- **6 memory frames optimal** - Balance between context and compute

### Capacity

- **More memory attention layers help video** (not image)
- **Larger image encoder helps both** image and video
- **Higher resolution improves all benchmarks**

### Positional Encoding

- **2D RoPE in memory attention helps**
- **Removing RPB from image encoder** - No regression, speed boost

## Sources

**Primary Sources:**
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - arXiv:2408.00714 (accessed 2025-11-23)
- [SAM 3 OpenReview Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) (accessed 2025-11-23)

**Additional References:**
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/) - Architecture overview
- [Meta AI SAM 2 Page](https://ai.meta.com/sam2/) - Official project page
- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2) - Implementation reference

**Key Papers Referenced:**
- Hiera: MAE pre-trained hierarchical vision transformer (Ryali et al., 2023)
- RoPE: Rotary Position Embedding (Su et al., 2021)
- FlashAttention-2 (Dao, 2023)
