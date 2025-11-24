# Video VLM Architectures: Spatiotemporal Transformers

## Overview

Video understanding presents unique challenges compared to static images: modeling temporal dynamics, handling long sequences, and capturing motion patterns across frames. This document examines four influential video transformer architectures that address these challenges through different design philosophies.

**Key Architectures:**
- **ViViT** (Google Research, 2021): Pure transformer for video with factorized spatiotemporal attention
- **TimeSformer** (Meta AI, 2021): Divided space-time attention for efficient video modeling
- **Video Swin Transformer** (Microsoft Research, 2021): Locality-based hierarchical architecture
- **VideoMAE** (Nanjing University, 2022): Self-supervised masked autoencoding for video
- **CogVideoX** (Tsinghua/Zhipu AI, 2024): Diffusion transformer for text-to-video generation

These architectures represent a fundamental shift from 3D CNNs to pure transformer-based video understanding.

---

## ViViT: Video Vision Transformer

### Architecture

**Paper**: "ViViT: A Video Vision Transformer" (Arnab et al., 2021)
**Source**: [arXiv:2103.15691](https://arxiv.org/abs/2103.15691) (accessed 2025-02-02)

ViViT extends the Vision Transformer (ViT) from images to videos by extracting spatiotemporal tokens from input video and encoding them through transformer layers.

**Core Innovation**: Factorized spatiotemporal attention that handles long video sequences efficiently.

**Architecture Variants:**

1. **Model 1 - Spatio-Temporal Attention**
   - Joint space-time attention across all tokens
   - Tokens: H×W×T (height × width × temporal)
   - Computational complexity: O((HWT)²)

2. **Model 2 - Factorised Encoder** (Most efficient)
   - Spatial transformer followed by temporal transformer
   - First encodes each frame independently
   - Then models temporal relationships
   - Complexity: O(HW·T + T²·HW)

3. **Model 3 - Factorised Self-Attention**
   - Alternating spatial and temporal attention within same layer
   - More parameter sharing than Model 2

4. **Model 4 - Factorised Dot-Product**
   - Factorizes attention computation itself
   - Computes spatial and temporal attention separately then combines

**Key Technical Details:**
- **Tubelet Embedding**: Extracts non-overlapping spatiotemporal "tubes" from video
  - Tube size: t×h×w (e.g., 2×16×16)
  - Projects to embedding dimension
- **Positional Encoding**:
  - Factorized: separate spatial + temporal embeddings
  - Or joint spatiotemporal embeddings
- **Regularization**: Strong regularization required for small datasets
  - Can leverage pretrained ViT image models

**Benchmarks** (from paper):
- **Kinetics-400**: 84.9% top-1 accuracy (Model 2)
- **Kinetics-600**: 85.8% top-1 accuracy
- **Epic Kitchens-100**: State-of-the-art on action recognition
- **Something-Something v2**: 68.3% top-1 (strong temporal reasoning)

**Implementation**: [GitHub - google-research/scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)

### Strengths & Limitations

**Strengths:**
- Multiple architectural variants for different compute budgets
- Leverages pretrained image models (ViT)
- State-of-the-art accuracy on multiple benchmarks
- Handles variable-length videos

**Limitations:**
- Requires large training datasets or strong regularization
- Quadratic complexity in sequence length (even with factorization)
- Model 1 (full attention) impractical for long videos

---

## TimeSformer: Divided Space-Time Attention

### Architecture

**Paper**: "Is Space-Time Attention All You Need for Video Understanding?" (Bertasius et al., 2021)
**Source**: Meta AI Research blog post (accessed 2025-02-02)
**Implementation**: [GitHub - facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer)

TimeSformer proposes a pure-transformer architecture for video that divides spatial and temporal attention operations.

**Core Innovation**: Divided space-time attention that separately models spatial and temporal dimensions within each block.

**Attention Schemes Explored:**

1. **Space-only Attention** (baseline)
   - Independent per-frame attention
   - No temporal modeling

2. **Joint Space-Time Attention**
   - Attention over all patches across all frames
   - Expensive: O(N²) where N = num_frames × patches_per_frame

3. **Divided Space-Time Attention** (chosen design)
   - Temporal attention: attend to same spatial location across frames
   - Spatial attention: attend to all spatial locations in same frame
   - Cost: O(NF + NT) vs O(N²) where F = patches/frame, T = frames

4. **Sparse Local Global Attention**
   - Local attention + global attention at intervals

5. **Axial Attention**
   - Attention along height, width, time axes independently

**Architecture Details:**
- **Input**: Video clips (8-96 frames at 224×224)
- **Patch embedding**: 16×16 patches per frame
- **Sequence**: Flatten spatial patches, add temporal dimension
- **Layers**: 12 transformer blocks with divided attention
- **Parameters**: 121M (TimeSformer-B), 636M (TimeSformer-L)

**Divided Attention Block:**
```
Input patches: [CLS] + [patches from frame 1] + [patches from frame 2] + ...

1. Temporal Attention:
   - For each spatial position p:
     - Attend across all frames at position p

2. Spatial Attention:
   - For each frame t:
     - Attend across all spatial positions in frame t

3. FFN (feed-forward network)
```

**Training Details:**
- Can initialize from pretrained ViT image models
- Faster training than 3D CNNs: 14 hours on 32 GPUs vs 30+ hours
- Works well with supervised learning (unlike VideoMAE which needs self-supervised pretraining)

**Benchmarks**:
- **Kinetics-400**: 80.7% top-1 accuracy
- **Kinetics-600**: 82.0% top-1 accuracy
- **Something-Something v2**: 62.5% top-1
- **Epic Kitchens-100**: 42.1% action recognition

**Key Insight**: Divided attention achieves better speed-accuracy tradeoff than joint space-time attention, making it practical for longer videos.

### Strengths & Limitations

**Strengths:**
- Linear complexity in sequence length (vs quadratic for joint attention)
- Fast training compared to 3D CNNs
- Can use pretrained image transformers
- Handles long videos (96+ frames)

**Limitations:**
- Still computes full attention (just factorized)
- No local inductive bias (unlike CNNs or Swin)
- Accuracy slightly lower than Video Swin on some benchmarks

---

## Video Swin Transformer: Locality-Based Video Modeling

### Architecture

**Paper**: "Video Swin Transformer" (Liu et al., 2021)
**Source**: [arXiv:2106.13230](https://arxiv.org/abs/2106.13230) (accessed 2025-02-02)
**Implementation**: [GitHub - SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

Video Swin Transformer advocates for **locality inductive bias** in video transformers, adapting the hierarchical Swin Transformer from images to video.

**Core Innovation**: 3D shifted windows for local spatiotemporal attention with hierarchical feature maps.

**Architecture Components:**

1. **3D Patch Partition**
   - Input: H×W×T video
   - Divide into 3D patches: (2×4×4) non-overlapping cubes
   - Linear embedding to C dimensions

2. **3D Shifted Window Attention**
   - **Local window**: Attention within M×M×M spatiotemporal windows
   - **Shifted windows**: Shift windows by (M/2, M/2, M/2) between layers
   - Enables cross-window connections while maintaining linear complexity

3. **Hierarchical Architecture** (4 stages)
   - Stage 1: 96 dims, T×H/4×W/4 resolution
   - Stage 2: 192 dims, T/2×H/8×W/8 (patch merging)
   - Stage 3: 384 dims, T/4×H/16×W/16
   - Stage 4: 768 dims, T/8×H/32×W/32

4. **3D Relative Position Bias**
   - Learned bias based on relative spatiotemporal positions
   - Improves generalization across different video lengths

**Window Attention:**
```
Window size: M×M×M (typically 8×7×7)
Complexity per window: O(M²·T·M)
Total complexity: O(T·H·W) linear in video volume!

Compare to global attention: O((T·H·W)²) quadratic
```

**Model Variants:**
- **Swin-T**: 28M parameters
- **Swin-S**: 50M parameters
- **Swin-B**: 88M parameters
- **Swin-L**: 200M parameters

**Benchmarks** (from paper):
- **Kinetics-400**: 84.9% top-1 (Swin-L, ~20× less pretraining data than ViViT)
- **Kinetics-600**: 86.1% top-1 (state-of-the-art)
- **Something-Something v2**: 69.6% top-1 (best temporal reasoning)
- **AVA action detection**: 31.4% mAP

**Key Achievement**: Better speed-accuracy tradeoff than global attention methods while using significantly less pretraining data.

### Strengths & Limitations

**Strengths:**
- Linear complexity in video dimensions
- Hierarchical features (useful for downstream tasks)
- Strong inductive bias from locality
- Excellent temporal modeling (Something-Something v2)
- Can use ImageNet pretrained Swin weights

**Limitations:**
- Fixed window size less flexible than global attention
- Shifted windows add implementation complexity
- Hierarchical design less straightforward than flat transformers

---

## VideoMAE: Masked Autoencoding for Video

### Architecture

**Paper**: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" (Tong et al., 2022)
**Source**: [arXiv:2203.12602](https://arxiv.org/abs/2203.12602) (accessed 2025-02-02)
**Implementation**: [GitHub - MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)

VideoMAE applies masked autoencoding (inspired by image MAE) to video with extreme masking ratios and tube masking.

**Core Innovation**:
- **Extremely high masking ratio**: 90-95% of video tokens masked (vs 75% in image MAE)
- **Tube masking**: Mask same spatial location across all frames (preserves temporal continuity)

**Architecture:**

1. **Encoder** (ViT-based)
   - Only processes visible tokens (5-10% of input)
   - Spatiotemporal transformer
   - Input: 16 frames, 224×224, patch size 16×16×2
   - Tokens per video: ~1,568 → encode only ~157 visible

2. **Decoder** (lightweight)
   - Reconstructs all tokens (visible + masked)
   - Adds mask tokens with learned positional embeddings
   - Predicts normalized pixel values for masked regions

3. **Tube Masking Strategy**
   - Temporal consistency: masking entire tubes across time
   - Random sampling: different tubes each iteration
   - Mask ratio: 90% (much higher than image MAE's 75%)

**Why High Masking Works for Video:**
- Video has high temporal redundancy
- Harder reconstruction task → better representations
- Forces model to understand temporal dynamics
- More data-efficient training

**Training Protocol:**
- **Pretraining**: Self-supervised on video (e.g., Kinetics-400)
- **Fine-tuning**: Supervised on downstream tasks
- Asymmetric encoder-decoder (encoder 12 layers, decoder 4 layers)
- No data augmentation during pretraining (unlike contrastive methods)

**Benchmarks**:

**Data Efficiency** (Kinetics-400 pretraining):
- 800 epochs: 81.1% accuracy on K400 validation
- 1600 epochs: 83.2% accuracy
- 3200 epochs: 84.6% accuracy

**Comparison** (K400 fine-tuning):
- VideoMAE (ViT-B): 81.5% with 3.5k pretraining videos
- CVRL (prior SOTA): 77.6% with same data
- BEVT: 76.7%

**Something-Something v2**: 68.9% top-1 (strong temporal understanding)

**VideoMAE V2 Improvements** (Wang et al., 2023):
- Dual masking: encoder uses high masking, decoder sees full video
- Scalable to billion-parameter models
- **Kinetics-400**: 86.6% top-1 (ViT-g, 1B params)

### Strengths & Limitations

**Strengths:**
- Extremely data-efficient (works with small datasets)
- Simple architecture (standard ViT + decoder)
- No complex data augmentation needed
- Self-supervised pretraining transfers well
- Scalable to large models

**Limitations:**
- Requires two-stage training (pretrain + finetune)
- Decoder discarded after pretraining
- High masking may not capture fine-grained details
- Less effective for tasks requiring spatial precision

---

## CogVideoX: Diffusion Transformer for Text-to-Video

### Architecture

**Paper**: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (Yang et al., 2024)
**Source**: [arXiv:2408.06072](https://arxiv.org/abs/2408.06072) (accessed 2025-02-02)
**Accepted**: ICLR 2025
**Implementation**: [GitHub - THUDM/CogVideo](https://github.com/THUDM/CogVideo)

CogVideoX generates 10-second, high-resolution videos (768×1360 at 16fps) from text prompts using diffusion transformers.

**Core Innovation**:
- **3D Causal VAE**: Compresses video along spatial AND temporal dimensions
- **Expert Transformer**: Deep text-video fusion via expert adaptive LayerNorm
- **Progressive training**: Enables long, coherent video generation

**Architecture Components:**

1. **3D Variational Autoencoder (VAE)**
   - Spatial compression: 8× downsampling
   - Temporal compression: 4× downsampling
   - Latent space: (T/4, H/8, W/8, C)
   - Causal: preserves temporal causality for generation
   - **Compression ratio**: 128× total (4×8×8)

2. **Expert Transformer Blocks**
   - Standard components: Self-attention, cross-attention, FFN
   - **Expert Adaptive LayerNorm**:
     - Multiple expert LayerNorms specialized for text-video fusion
     - Gating mechanism selects appropriate experts
     - Enables deep multimodal fusion

3. **Text Encoder**
   - T5-XXL (4.7B parameters) for text encoding
   - Rich semantic understanding of prompts

4. **Diffusion Process**
   - Latent diffusion in compressed 3D space
   - V-prediction objective (predicts velocity)
   - Noise schedule optimized for video

**Training Strategy:**

1. **Progressive Training**
   - Stage 1: Low resolution, short videos
   - Stage 2: Higher resolution
   - Stage 3: Longer duration
   - Final: 10-second, 768×1360 videos

2. **Multi-Resolution Frame Packing**
   - Different aspect ratios during training
   - Different video lengths (6s, 8s, 10s)
   - Improves generalization

**Model Variants:**
- **CogVideoX-2B**: 2 billion parameters
- **CogVideoX-5B**: 5 billion parameters (best quality)

**Key Results**:
- **VBench benchmark**: State-of-the-art on 16 dimensions
- **Human evaluation**: Preferred over Pika, Gen-2, VideoCrafter
- **Text alignment**: Superior prompt following vs baselines
- **Motion quality**: Coherent, significant motion over 10 seconds

**Data Pipeline Innovation**:
- Video captioning model generates detailed descriptions
- Data filtering: quality, aesthetics, text relevance
- Preprocessing: scene detection, resolution handling

### Strengths & Limitations

**Strengths:**
- Long video generation (10 seconds continuous)
- High resolution (768×1360)
- Strong text-video alignment
- Coherent motion and narratives
- Open-source weights available

**Limitations:**
- Computationally expensive (5B parameters)
- Still struggles with complex physical interactions
- 10-second limit (longer videos require stitching)
- Diffusion sampling is slow (vs GANs)

---

## Architectural Comparison

### Attention Mechanisms

| Architecture | Attention Type | Complexity | Spatiotemporal Handling |
|-------------|----------------|------------|------------------------|
| **ViViT** | Factorized (spatial → temporal) | O(HW·T + T²·HW) | Sequential modeling |
| **TimeSformer** | Divided (spatial ‖ temporal) | O(NF + NT) | Parallel, independent |
| **Video Swin** | Local windowed 3D | O(THW) linear | Hierarchical, shifted windows |
| **VideoMAE** | Standard ViT (masked) | O(N²) on visible tokens only | Joint spatiotemporal |
| **CogVideoX** | Diffusion transformer | O(N²) | 3D latent space diffusion |

### Design Philosophies

**Global Attention** (ViViT, TimeSformer):
- Models long-range dependencies
- Factorization for efficiency
- Better for temporal reasoning tasks

**Local Attention** (Video Swin):
- Inductive bias from locality
- Hierarchical features
- Better speed-accuracy tradeoff

**Self-Supervised** (VideoMAE):
- Data-efficient learning
- Extreme masking forces temporal understanding
- Transfer learning focused

**Generative** (CogVideoX):
- Text-conditional generation
- Latent space diffusion
- Focus on quality and coherence

### Benchmark Performance Summary

**Kinetics-400** (action recognition):
- Video Swin-L: 84.9% (least pretraining data)
- VideoMAE V2: 86.6% (with 1B params)
- ViViT-L: 84.9%
- TimeSformer-L: 80.7%

**Something-Something v2** (temporal reasoning):
- Video Swin-B: 69.6% ⭐ (best)
- VideoMAE: 68.9%
- ViViT: 68.3%
- TimeSformer: 62.5%

**Key Insight**: Local attention (Video Swin) excels at temporal reasoning, while global attention works well for appearance-based tasks.

---

## Implementation Considerations

### When to Use Each Architecture

**ViViT**:
- ✅ When you have large training datasets
- ✅ Multiple compute budgets (model variants 1-4)
- ✅ Need to leverage pretrained ViT models
- ❌ Limited compute for long videos

**TimeSformer**:
- ✅ Long video sequences (96+ frames)
- ✅ Fast training required
- ✅ Pretrained models available
- ❌ Need absolute best accuracy

**Video Swin**:
- ✅ Best speed-accuracy tradeoff
- ✅ Downstream tasks need hierarchical features (detection, segmentation)
- ✅ Temporal reasoning critical
- ✅ Limited pretraining data
- ❌ Implementation complexity acceptable

**VideoMAE**:
- ✅ Limited labeled training data
- ✅ Self-supervised pretraining feasible
- ✅ Transfer learning focus
- ❌ Two-stage training acceptable

**CogVideoX**:
- ✅ Text-to-video generation
- ✅ High-quality, long video synthesis
- ✅ Generative tasks
- ❌ Discriminative video understanding

### Computational Requirements

**Training** (approximate, single GPU memory):
- ViViT-B: 16-32 GB (8-16 frames, batch size 8)
- TimeSformer-B: 24-40 GB (8-32 frames)
- Video Swin-B: 16-24 GB (32 frames, efficient)
- VideoMAE-B: 12-20 GB (16 frames, only 10% visible)
- CogVideoX-5B: 80+ GB (requires multi-GPU)

**Inference** (FLOPs per video):
- Video Swin: Most efficient (linear complexity)
- TimeSformer: Moderate (divided attention)
- ViViT: Higher (full factorized attention)
- VideoMAE: Similar to ViViT
- CogVideoX: Highest (iterative diffusion sampling)

---

## Recent Developments & Future Directions

### Hybrid Architectures

**Combining strengths**:
- Local attention (Swin) + global attention (ViViT) = best of both worlds
- Masked pretraining (VideoMAE) + any architecture = data efficiency
- Hierarchical features + factorized attention

### Scaling Laws

**VideoMAE V2 findings**:
- Scaling to 1B+ parameters improves accuracy
- Dual masking enables stable large-scale training
- Pretraining data quality > quantity

### Multimodal Integration

**Text-Video alignment** (CogVideoX direction):
- Joint training on video-text pairs
- Diffusion for generation
- Discriminative models benefit from generative pretraining

### Efficiency Improvements

**Active research areas**:
- Dynamic token selection (adaptive computation)
- Mixed precision training (FP8, INT8)
- Knowledge distillation (large → small models)
- Flash Attention for transformers

---

## Code Examples

### ViViT (via HuggingFace Transformers)

```python
from transformers import VivitForVideoClassification, VivitImageProcessor
import torch

# Load pretrained model
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

# Process video (list of frames)
inputs = processor(video, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
```

### TimeSformer (via Facebook Research repo)

```python
# From facebookresearch/TimeSformer
import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type='divided_space_time',
    pretrained_model='path/to/checkpoint.pyth'
)

# Input: (batch, channels, frames, height, width)
video = torch.rand(1, 3, 8, 224, 224)
output = model(video)  # (batch, num_classes)
```

### Video Swin (via MMAction2 or Transformers)

```python
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

inputs = feature_extractor(video, return_tensors="pt")
outputs = model(**inputs)
```

### VideoMAE (via HuggingFace)

```python
from transformers import VideoMAEForPreTraining, VideoMAEFeatureExtractor
import torch

# Self-supervised pretraining model
model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

# Masked video reconstruction
inputs = extractor(video, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss  # Reconstruction loss
```

---

## Sources

**Primary Papers:**
- [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) - Arnab et al., ICCV 2021 (accessed 2025-02-02)
- [Video Swin Transformer](https://arxiv.org/abs/2106.13230) - Liu et al., CVPR 2022 (accessed 2025-02-02)
- [VideoMAE: Masked Autoencoders are Data-Efficient Learners](https://arxiv.org/abs/2203.12602) - Tong et al., NeurIPS 2022 (accessed 2025-02-02)
- [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) - Yang et al., ICLR 2025 (accessed 2025-02-02)

**Implementation References:**
- [google-research/scenic (ViViT)](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
- [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer)
- [SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
- [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [THUDM/CogVideo](https://github.com/THUDM/CogVideo)

**Blog Posts & Technical Reports:**
- [TimeSformer: A new architecture for video understanding - Meta AI](https://ai.meta.com/blog/timesformer-a-new-architecture-for-video-understanding/) (accessed 2025-02-02)

**HuggingFace Documentation:**
- [ViViT Model Documentation](https://huggingface.co/docs/transformers/model_doc/vivit)
- [VideoMAE Model Documentation](https://huggingface.co/docs/transformers/model_doc/videomae)
- [TimeSformer Model Documentation](https://huggingface.co/docs/transformers/model_doc/timesformer)
