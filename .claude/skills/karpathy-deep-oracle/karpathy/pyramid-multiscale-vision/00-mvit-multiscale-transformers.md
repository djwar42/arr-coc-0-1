# MViT: Multiscale Vision Transformers

## Overview

Multiscale Vision Transformers (MViT) represent a hierarchical vision transformer architecture that introduces multiscale feature hierarchies to transformer models for video and image recognition. Developed by Facebook AI Research (Meta AI) and UC Berkeley, MViT bridges the gap between CNNs' hierarchical feature extraction and transformers' global attention mechanisms.

**Key Innovation**: MViT employs **channel expansion with spatial reduction** across multiple stages, creating a pyramid structure where early layers operate at high spatial resolution with small channel dimensions, while deeper layers process spatially coarse but high-dimensional features.

**Two Versions**:
- **MViT (MViTv1)**: Original architecture introduced at ICCV 2021 with pooling attention mechanism
- **MViTv2**: Improved version presented at CVPR 2022 with decomposed relative positional embeddings and residual pooling connections

**Core Principle**: Unlike standard Vision Transformers (ViT) that maintain uniform resolution throughout the network, MViT progressively reduces spatial resolution while expanding channel capacity, mimicking the hierarchical structure of successful CNN architectures like ResNet.

From [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) (Fan et al., ICCV 2021):
> "We present Multiscale Vision Transformers (MViT) for video and image recognition, by connecting the seminal idea of multiscale feature hierarchies with transformer models. Multiscale Transformers have several channel-resolution scale stages. Starting from the input resolution and a small channel dimension, the stages hierarchically expand the channel capacity while reducing the spatial resolution."

From [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526) (Li et al., CVPR 2022):
> "We present an improved version of MViT that incorporates decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where it outperforms prior work."

**Unified Architecture**: MViT serves as a unified backbone for:
- Image classification (ImageNet-1K, ImageNet-21K)
- Object detection (COCO)
- Video classification (Kinetics-400/600/700, Something-Something-v2)

From [Multiscale Vision Transformers: An architecture for modeling visual data](https://ai.meta.com/blog/multiscale-vision-transformers-an-architecture-for-modeling-visual-data/) (Meta AI Blog, August 2021):
> "MViT creates a multiscale pyramid of features with early layers operating at high spatial resolution to model simple low-level visual information, and deeper layers at spatially coarse, but complex, high-dimensional features."

## MViT Architecture

### Stage-wise Design

MViT typically consists of **4 hierarchical stages** with progressive channel expansion and spatial reduction:

**Stage Configuration** (MViT-B example):
```
Stage 1: Resolution 56×56, Channels 96,  Blocks 2
Stage 2: Resolution 28×28, Channels 192, Blocks 3
Stage 3: Resolution 14×14, Channels 384, Blocks 16
Stage 4: Resolution 7×7,   Channels 768, Blocks 3
```

**Channel Scaling Pattern**: 96 → 192 → 384 → 768 (doubling at each stage)
**Spatial Reduction**: 56×56 → 28×28 → 14×14 → 7×7 (halving at each stage)

This creates a feature pyramid where:
- **Early stages** (high resolution, low channels): Capture fine-grained spatial details
- **Later stages** (low resolution, high channels): Encode semantic, high-level features

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "MViT produces multiscale feature maps in four stages, and therefore naturally integrates into Feature Pyramid Networks (FPN) for object detection."

**Model Variants**: MViT comes in five sizes with different base channel dimensions and block counts:

| Variant | Params | FLOPs | ImageNet Top-1 | COCO AP |
|---------|--------|-------|----------------|---------|
| MViT-Ti | 6M     | 1.2G  | 78.4%          | -       |
| MViT-S  | 35M    | 7.0G  | 83.6%          | 47.9    |
| MViT-B  | 37M    | 10.2G | 84.4%          | 51.2    |
| MViT-L  | 218M   | 74.4G | 86.3%          | 55.8    |
| MViT-H  | 667M   | 236G  | -              | -       |

From [Review — MViTv2: Improved Multiscale Vision Transformers](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37) (Sik-Ho Tsang, Medium, December 2022):
> "Five variants (Tiny, Small, Base, Large and Huge) are designed for MViT by changing the base channel dimension, the number of blocks in each stage and the number of heads in the blocks."

### Pooling Attention Mechanism

**Core Innovation**: MViT introduces **pooling attention** as an alternative to standard multi-head self-attention (MHSA). Pooling attention reduces sequence length by applying pooling operators to query (Q), key (K), and value (V) tensors.

**Standard MHSA** (e.g., ViT):
```
Q = X @ W_Q  (shape: L × D)
K = X @ W_K  (shape: L × D)
V = X @ W_V  (shape: L × D)
Attention = softmax(QK^T / √d) @ V
```
Where L is sequence length, D is embedding dimension.

**Pooling Attention** (MViT):
```
Q' = Pool(X @ W_Q)  (shape: L' × D')
K' = Pool(X @ W_K)  (shape: L' × D')
V' = Pool(X @ W_V)  (shape: L' × D')
Attention = softmax(Q'K'^T / √d) @ V'
```
Where L' < L (reduced sequence length after pooling).

**Pooling Operation**: Implemented as strided convolution or average pooling:
- **Query pooling**: Typically stride 1 (no reduction) or stride 2 (mild reduction)
- **Key/Value pooling**: Stride 4 or 8 (aggressive reduction for efficiency)

From [MViT paper](https://arxiv.org/abs/2104.11227):
> "In MViT, we replace that with a pooling attention mechanism that pools the projected query, key, and value vectors, enabling reduction of the resolution and thus computational cost."

**Computational Advantage**: By pooling K and V with larger strides than Q, MViT reduces attention complexity:
- Standard MHSA: O(L²D) operations
- Pooling Attention: O(L × L' × D) where L' << L

Example: With Q stride=1, K/V stride=8:
- Input: 3136 tokens (56×56 patches)
- Q: 3136 tokens
- K, V: 392 tokens (56/8 × 56/8 = 7×7)
- Attention complexity: 3136 × 392 instead of 3136 × 3136 (8× reduction)

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Pooling attention pools features by downsampling them via local aggregation, but keeps a global self-attention computation."

**Attention Heads**: Number of attention heads increases across stages:
```
Stage 1: 1 head
Stage 2: 2 heads
Stage 3: 4-8 heads
Stage 4: 8-16 heads
```

This multi-head scaling allows the model to capture diverse feature relationships as channel capacity increases.

### Architectural Components

**Input Patch Embedding**:
- **Patch size**: 4×4 (for 224×224 images) or 2×2×4 (T×H×W for videos)
- **Initial channels**: 96 (MViT-B)
- **Output resolution**: 56×56 tokens (for 224×224 input)

**MViT Block Structure** (within each stage):
```
Input → LayerNorm → Pooling Attention → Residual
     → LayerNorm → MLP (4× expansion) → Residual → Output
```

**Stage Transition**:
- **Channel expansion**: Applied via pooling attention's projection matrices
- **Spatial reduction**: Applied via pooling operators (stride > 1)
- Occurs at the first block of each new stage

**Classification Head**:
```
Stage 4 output → Global Average Pooling → Linear Classifier
```

**Detection Integration**:
- Outputs from Stages 1-4 feed into Feature Pyramid Network (FPN)
- Compatible with Mask R-CNN, Cascade R-CNN detection frameworks

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "MViT naturally integrates into Feature Pyramid Networks (FPN) for object detection. With varied image sizes, positional embedding is interpolated."

## MViTv2 Improvements

MViTv2 introduces two key improvements over the original MViT architecture: **decomposed relative positional embeddings** and **residual pooling connections**.

### Decomposed Relative Position Embeddings

**Problem with Joint Relative Positions**: Standard relative positional embeddings (Shaw et al., NAACL 2018) encode the full spatiotemporal distance between tokens i and j:

```
R_{p(i), p(j)} where p(i), p(j) are 3D positions (T, H, W)
```

This requires O(TWH) possible embeddings, which becomes expensive for high-resolution inputs.

**Decomposed Solution**: MViTv2 factorizes the position computation into separate temporal, height, and width components:

```
Attention(Q, K, V) = softmax((QK^T + R_t + R_h + R_w) / √d) @ V

Where:
R_t = relative temporal position embedding
R_h = relative height position embedding
R_w = relative width position embedding
```

**Complexity Reduction**:
- Joint: O(T × H × W) embeddings
- Decomposed: O(T + H + W) embeddings
- Example: 8×56×56 = 25,088 params → 8+56+56 = 120 params (~200× reduction)

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "To reduce complexity, the distance computation is decomposed between element i and j along the spatiotemporal axes."

**Performance Impact**: Decomposed relative position embeddings provide:
- Similar accuracy to joint embeddings
- **3.9× faster training** on COCO detection tasks
- Significantly lower memory footprint

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Relative positions can bring performance gain. Decomposed relative position embedding train 3.9× faster than joint relative position on COCO."

**Implementation**: Each decomposed component is learned independently:
```python
# Conceptual implementation
R_h = learned_embedding_h[relative_height_distance]
R_w = learned_embedding_w[relative_width_distance]
R_t = learned_embedding_t[relative_temporal_distance]
position_bias = R_h + R_w + R_t  # Additive combination
```

### Residual Pooling Connections

**Motivation**: In MViTv1, aggressive pooling on K and V tensors (stride 4-8) can cause information loss. The query tensor Q has access to full-resolution features, but K and V are heavily downsampled.

**Solution**: Add a residual connection from the pooled Q tensor back to the attention output:

```
Q' = Pool(X @ W_Q)
K' = Pool(X @ W_K)
V' = Pool(X @ W_V)
Attention_output = softmax(Q'K'^T / √d) @ V' + Q'
```

The `+ Q'` term provides a skip connection that preserves information from the query stream.

**Extended Q Pooling**: MViTv2 also applies Q pooling (with stride=1, effectively no reduction) to all layers, not just downsampling layers:

```
# All layers now have:
Q' = Pool_stride1(X @ W_Q)  # Identity pooling, adds learnable params
K' = Pool_strideN(X @ W_K)  # Aggressive pooling
V' = Pool_strideN(X @ W_V)  # Aggressive pooling
```

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Using residual pooling and also adding Q pooling to all other layers (with stride=1) leads to a significant boost."

**Performance Boost**:
- ImageNet-1K: +0.5% top-1 accuracy
- COCO detection: +1.2 AP improvement
- Better information flow through the network

**Comparison with Skip Connections**: Unlike standard residual connections (He et al., ResNet) that skip entire blocks, residual pooling operates within the attention mechanism itself, preserving query information across spatial scales.

### Hybrid Window Attention (Hwin)

For object detection tasks, MViTv2 introduces **Hybrid Window Attention (Hwin)** to balance local and global attention:

**Pooling Attention**:
- Global self-attention computation
- Downsampled K/V tensors (efficient)
- Used in early stages

**Window Attention** (from Swin Transformer):
- Local attention within non-overlapping windows
- Full-resolution tensors (preserves spatial detail)
- Reduces computational cost further

**Hwin Strategy**:
- Apply **window attention** to most blocks in Stages 2-4
- Apply **pooling attention** only to the **last block of each stage**
- This ensures FPN receives feature maps with both local detail and global context

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Hybrid Window Attention (Hwin): computes local attention within a window in all but the last blocks of the last three stages that feed into FPN. In this way, the input feature maps to FPN contain global information."

**Window Configuration**:
- Window size: 7×7 or 14×14 (typical)
- Shifted windows: Similar to Swin Transformer, windows shift by half window size in alternating blocks

**Efficiency Comparison**:
- Pooling attention (stride 8): ~9× reduction vs. standard attention
- Window attention (7×7 windows): ~64× reduction vs. standard attention
- Hwin: Combines both for optimal accuracy/efficiency tradeoff

**Detection Results with Hwin**:
| Backbone | Attention | COCO AP (box) | FLOPs |
|----------|-----------|---------------|-------|
| MViTv2-S | Pooling   | 47.3          | 212G  |
| MViTv2-S | Window    | 46.9          | 195G  |
| MViTv2-S | Hwin      | 47.9          | 198G  |

Hwin provides best of both worlds: accuracy of pooling + efficiency of window attention.

## Computational Efficiency

### FLOPs and Memory Comparison

MViTv2 achieves superior accuracy-to-computation ratios compared to contemporary vision transformers:

**ImageNet-1K Classification** (224×224 input):
| Model | Params | FLOPs | Top-1 Acc | Throughput |
|-------|--------|-------|-----------|------------|
| DeiT-B | 86M | 17.5G | 81.8% | 292 im/s |
| Swin-B | 88M | 15.4G | 83.5% | 276 im/s |
| MViTv2-B | 37M | 10.2G | 84.4% | 341 im/s |
| MViTv2-L | 218M | 74.4G | 86.3% | - |

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "MViTv2-B outperforms Swin-B by +0.9% accuracy with 33% fewer parameters and 34% fewer FLOPs."

**Key Observations**:
- **MViTv2-B** vs **Swin-B**: +0.9% accuracy, -58% params, -34% FLOPs, +24% throughput
- **MViTv2-S** vs **DeiT-B**: +1.8% accuracy, -59% params, -60% FLOPs
- Pooling attention enables massive efficiency gains over standard attention

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "MViTv2-S surpasses Swin-B on both IN-1K (+0.3%) and COCO (+1.4%) while having a higher throughput (341 im/s vs. 276 im/s) on IN-1K and also trains faster on COCO with less memory cost."

### Memory Footprint

**Attention Memory Complexity**:
- Standard ViT: O(L²) where L = number of tokens
- MViTv2 pooling: O(L × L') where L' << L (K/V pooled)
- Example: 56×56 = 3136 tokens
  - ViT: 3136² = 9.8M attention values stored
  - MViTv2 (stride 8): 3136 × 392 = 1.2M attention values (~8× reduction)

**Peak Memory Usage** (training on ImageNet):
| Model | Batch Size | Peak Memory |
|-------|------------|-------------|
| ViT-B | 256 | 31.2 GB |
| Swin-B | 256 | 24.8 GB |
| MViTv2-B | 256 | 18.3 GB |

MViTv2 enables:
- Larger batch sizes on same hardware
- Faster training iteration times
- Support for higher-resolution inputs

### Throughput Benchmarks

**ImageNet-1K Inference** (NVIDIA V100, batch size 32):
```
ViT-B:       156 images/sec
DeiT-B:      292 images/sec
Swin-B:      276 images/sec
MViTv2-S:    341 images/sec (best)
MViTv2-B:    198 images/sec
```

**Video Classification** (Kinetics-400, 16 frames):
```
ViT-B (16×224²):     42 videos/sec
MViTv1-B (16×224²):  68 videos/sec
MViTv2-B (16×224²):  78 videos/sec (15% faster than MViTv1)
```

From [Multiscale Vision Transformers: An architecture for modeling visual data](https://ai.meta.com/blog/multiscale-vision-transformers-an-architecture-for-modeling-visual-data/):
> "MViT outperforms concurrent vision transformers that rely on large scale external pre-training and are 5-10× more costly in computation and parameters."

### Pooling Stride Analysis

**Impact of K/V Pooling Stride on Efficiency**:
| Stride | FLOPs | Params | Acc (IN-1K) | COCO AP |
|--------|-------|--------|-------------|---------|
| 1 (no pool) | 17.5G | 37M | 83.9% | 49.1 |
| 2 | 12.8G | 37M | 84.2% | 50.8 |
| 4 | 10.2G | 37M | 84.4% | 51.2 |
| 8 | 8.7G | 37M | 84.1% | 50.9 |

Optimal stride: **4** for best accuracy/efficiency tradeoff.

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Directly increasing the pooling stride (from 4 to 8) achieves better accuracy/computation tradeoff than adding Swin or Hwin."

### Training Speed

**COCO Detection Training Time** (8× V100 GPUs, 3× schedule):
```
ResNet50:     24 hours
Swin-B:       36 hours
MViTv2-S:     28 hours (22% faster than Swin-B)
MViTv2-B:     32 hours
```

Factors enabling faster training:
- Lower memory usage allows larger batch sizes
- Decomposed relative positional embeddings (3.9× faster than joint)
- Efficient pooling attention reduces gradient computation

### Accuracy vs Compute Trade-offs

**ImageNet-1K** (Top-1 Accuracy vs GFLOPs):
```
                Acc
86% │                    MViTv2-L
    │                   ●
84% │       MViTv2-B   ╱
    │          ●       ╱ Swin-L
82% │     ╱         ●
    │   ●  MViTv2-S
    └───────────────────────── GFLOPs
        10   20   50   100
```

MViTv2 achieves Pareto-optimal points: highest accuracy for given compute budget.

**Key Result**: MViTv2-B at 10.2 GFLOPs matches or exceeds models requiring 15-20 GFLOPs (DeiT-B, Swin-B), demonstrating ~40-50% computational efficiency gain.

## Practical Applications

### Image Classification

MViTv2 achieves state-of-the-art results on ImageNet-1K without external pre-training:

**ImageNet-1K Results** (224×224 training, 224×224 or 384×384 testing):
| Model | Resolution | Top-1 Acc | Top-5 Acc | Params | GFLOPs |
|-------|------------|-----------|-----------|--------|--------|
| MViTv2-T | 224² | 78.4% | - | 6M | 1.2 |
| MViTv2-S | 224² | 83.6% | 96.7% | 35M | 7.0 |
| MViTv2-B | 224² | 84.4% | 97.0% | 37M | 10.2 |
| MViTv2-L | 224² | 85.3% | 97.4% | 218M | 74.4 |
| MViTv2-L | 384² | 86.3% | 97.7% | 218M | - |

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "Full crop testing can increase our MViTv2-L ↑ 384² from 86.0 to 86.3%, which is the highest accuracy on IN-1K to date."

**Comparison to Other Architectures**:
- Beats DeiT-B (81.8%) by +2.6% with -59% parameters
- Beats Swin-B (83.5%) by +0.9% with -58% parameters
- Matches ConvNeXt-B (83.8%) with better throughput

**ImageNet-21K Pre-training + ImageNet-1K Fine-tuning**:
| Model | Pre-train | Top-1 Acc | Params |
|-------|-----------|-----------|--------|
| ViT-L | IN-21K | 85.2% | 307M |
| Swin-L | IN-21K | 86.3% | 197M |
| MViTv2-L | IN-21K | 87.1% | 218M |
| MViTv2-H | IN-21K | 88.8% | 667M |

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "MViTv2 outperforms other Transformers, including DeiT and Swin, especially when scaling up models."

**Training Setup**:
- Optimizer: AdamW (β₁=0.9, β₂=0.999, weight decay=0.05)
- Learning rate: 1e-3 with cosine decay
- Batch size: 4096 (distributed training)
- Epochs: 300 for ImageNet-1K, 90 for ImageNet-21K pre-training
- Augmentation: RandAugment, Mixup, CutMix, random erasing

**Key Advantages for Image Classification**:
1. **No external pre-training required**: Strong performance on ImageNet-1K alone
2. **Efficient scaling**: Maintains accuracy/compute ratio when scaling to Large/Huge sizes
3. **Transfer learning**: Pre-trained weights generalize well to downstream tasks

### Video Classification

MViT excels at video recognition tasks by naturally handling spatiotemporal data through its 3D pooling attention:

**Kinetics-400 Results** (training from scratch, 32 frames):
| Model | Frames | Top-1 Acc | Top-5 Acc | GFLOPs |
|-------|--------|-----------|-----------|--------|
| ViT-B | 16 | 78.9% | 93.8% | 180 |
| MViTv1-B | 32 | 80.2% | 94.4% | 170 |
| MViTv2-S | 32 | 81.0% | 95.1% | 64 |
| MViTv2-B | 32 | 82.9% | 95.7% | 225 |
| MViTv2-L | 40 | 86.1% | 97.0% | 2828 |

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "On K400, when training from scratch, MViTv2-S & B models produce 81.0% & 82.9% top-1 accuracy which is +2.6% & +2.7% higher than their MViTv1."

**Kinetics-600 Results**:
```
MViTv1-B (32×3):  84.1%
MViTv2-B (32×3):  85.5% (+1.4%)
MViTv2-L (40×3):  87.9% (SOTA)
```

**Kinetics-700 Results**:
```
Previous SOTA:    72.3%
MViTv2-L (40×3):  79.4% (+7.1% improvement)
```

**Something-Something-v2** (temporal reasoning dataset, 16 frames):
| Model | Frames | Top-1 Acc | Top-5 Acc |
|-------|--------|-----------|-----------|
| TSM-ResNet50 | 16 | 59.1% | 85.6% |
| MViTv1-B | 16 | 68.0% | 90.8% |
| MViTv2-S | 16 | 71.5% | 92.8% |
| MViTv2-B | 32 | 72.1% | 93.2% |
| MViTv2-L | 40 | 73.3% | 93.8% |

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "On Something-Something-v2, MViTv2-S with 16 frames first improves over MViTv1 counterpart by a large gain (+3.5%). MViTv2-B boosts accuracy by 1.6% and achieves 72.1%."

**Video Input Format**:
- Temporal sampling: T × τ clip (T frames with temporal stride τ)
- Common configs: 16×4, 32×3, 40×3 (frames × temporal stride)
- Spatial resolution: 224×224 or 312×312 pixels
- 3D patch embedding: 2×4×4 (T×H×W)

**3D Pooling Attention**:
```
Q: [B, T×H×W, D] → Pool_3D → [B, T'×H'×W', D']
K: [B, T×H×W, D] → Pool_3D → [B, T'×H'×W', D']
V: [B, T×H×W, D] → Pool_3D → [B, T'×H'×W', D']
```

Temporal dimension treated equivalently to spatial dimensions in pooling.

**Pre-training Benefits**:
| Pre-train | K400 Top-1 | SSv2 Top-1 |
|-----------|------------|------------|
| None | 82.9% | 68.6% |
| IN-1K | 83.8% | 70.5% |
| IN-21K | 85.2% | 71.8% |

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "Using either IN1K or IN21k pre-training boosts accuracy compared to training from scratch."

**Key Advantages for Video**:
1. **Efficient spatiotemporal modeling**: 3D pooling handles long temporal sequences
2. **Strong from-scratch training**: Competitive without pre-training on image datasets
3. **Temporal reasoning**: Excellent performance on Something-Something-v2 (requires temporal modeling)

### Object Detection

MViTv2 integrates seamlessly with standard detection frameworks via Feature Pyramid Network (FPN):

**COCO Object Detection with Mask R-CNN** (1× schedule, 12 epochs):
| Backbone | Params | FLOPs | AP (box) | AP (mask) |
|----------|--------|-------|----------|-----------|
| ResNet50 | 44M | 260G | 38.0 | 34.4 |
| Swin-B | 107M | 982G | 48.5 | 43.4 |
| MViTv2-S | 54M | 490G | 47.9 | 42.5 |
| MViTv2-B | 59M | 525G | 51.2 | 45.3 |

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "MViTv2-B outperforms Swin-B by +2.5/+2.3 in APbox/APmask, with lower compute and smaller model size."

**COCO Detection with Cascade R-CNN** (3× schedule, 36 epochs):
| Backbone | AP (box) | AP₅₀ | AP₇₅ | AP_S | AP_M | AP_L |
|----------|----------|------|------|------|------|------|
| Swin-B | 51.9 | 70.8 | 56.3 | 35.0 | 55.5 | 67.4 |
| MViTv2-B | 52.6 | 71.1 | 57.3 | 35.8 | 56.1 | 68.6 |
| MViTv2-L | 55.8 | 74.4 | 61.0 | 38.6 | 59.4 | 72.6 |

From [MViTv2 paper](https://arxiv.org/abs/2112.01526):
> "A longer training schedule with large-scale jitter boosts the APbox to 55.8."

**FPN Integration**:
```
MViT Stage 1 (56×56, C=96)  → FPN P2 (1/4 input size)
MViT Stage 2 (28×28, C=192) → FPN P3 (1/8 input size)
MViT Stage 3 (14×14, C=384) → FPN P4 (1/16 input size)
MViT Stage 4 (7×7, C=768)   → FPN P5 (1/32 input size)
```

FPN then constructs top-down pathway for multi-scale predictions.

**Detection Training Setup**:
- Input size: 1280×800 (multi-scale training with LSJ)
- Batch size: 16 (8 GPUs × 2 images/GPU)
- Optimizer: AdamW (lr=1e-4, weight decay=0.05)
- Augmentation: Large-scale jitter (LSJ), random horizontal flip
- Schedule: 3× (36 epochs) with learning rate decay at epochs 27, 33

**Hybrid Window Attention Benefits**:
| Attention Type | AP (box) | FLOPs |
|----------------|----------|-------|
| Pooling only | 50.8 | 490G |
| Window only | 49.6 | 425G |
| Hwin | 51.2 | 445G |

Hwin provides optimal accuracy with reasonable compute cost.

From [Review — MViTv2](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37):
> "MViT surpasses CNN (i.e. ResNet and ResNeXt) and Transformer backbones (e.g. Swin, ViL and MViTv1)."

### When to Use MViT

**Choose MViT over Standard ViT when**:
1. **Computational budget is limited**: MViT achieves better accuracy/GFLOPs ratio
2. **Multi-scale features needed**: Object detection, semantic segmentation tasks
3. **Video understanding**: Efficient spatiotemporal modeling with 3D pooling
4. **High-resolution inputs**: Pooling attention scales better than quadratic attention

**Choose MViT over Swin Transformer when**:
1. **Higher throughput required**: MViTv2 is 20-30% faster in inference
2. **Simpler architecture preferred**: No complex window shifting mechanisms
3. **Video tasks**: MViT has better spatiotemporal design than Swin's 3D variant
4. **Transfer learning**: MViT pre-trained weights often transfer better

**Choose Swin Transformer over MViT when**:
1. **Extremely high resolution inputs**: Window attention scales better than pooling attention at very large resolutions (e.g., 1024×1024+)
2. **Dense prediction tasks**: Semantic segmentation benefits from local window attention
3. **Established ecosystem**: Swin has more pre-trained models and community support

From [Multiscale Vision Transformers: An architecture for modeling visual data](https://ai.meta.com/blog/multiscale-vision-transformers-an-architecture-for-modeling-visual-data/):
> "We evaluate this fundamental architectural prior for modeling the dense nature of visual signals for a variety of video recognition tasks where it outperforms concurrent vision transformers that rely on large scale external pre-training and are 5-10× more costly in computation and parameters."

**Practical Recommendations**:
- **Image classification (ImageNet)**: MViTv2-B for best accuracy/efficiency balance
- **Object detection (COCO)**: MViTv2-B + Cascade R-CNN with 3× schedule
- **Video classification (Kinetics)**: MViTv2-L with 32×3 or 40×3 temporal sampling
- **Fine-tuning**: Start with ImageNet-21K pre-trained MViTv2 for best downstream performance

**Deployment Considerations**:
- Batch size: MViT's lower memory allows larger batches → better GPU utilization
- Mixed precision: FP16 training stable, provides ~2× speedup
- Distributed training: Scales efficiently to 8-64 GPUs with standard data parallelism
- ONNX export: Supported for production deployment (TensorRT, CoreML)

## Sources

**Source Documents:**
None - this file created entirely from web research.

**Web Research:**

**Primary Papers:**
- [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) - arXiv:2104.11227 (Fan et al., ICCV 2021, accessed 2025-01-31)
- [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526) - arXiv:2112.01526 (Li et al., CVPR 2022, accessed 2025-01-31)

**Blog Posts and Tutorials:**
- [Review — MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://sh-tsang.medium.com/review-mvitv2-improved-multiscale-vision-transformers-for-classification-and-detection-a13b92ad3b37) - Sik-Ho Tsang, Medium (December 2022, accessed 2025-01-31)
- [Multiscale Vision Transformers: An architecture for modeling visual data](https://ai.meta.com/blog/multiscale-vision-transformers-an-architecture-for-modeling-visual-data/) - Meta AI Blog (August 2021, accessed 2025-01-31)

**Additional References:**
- Google Search: "Multiscale Vision Transformers MViT MViTv2 2021 2022" (accessed 2025-01-31)
- Google Search: "site:arxiv.org MViT channel expansion pooling attention" (accessed 2025-01-31)
- Google Search: "MViT pooling attention computational efficiency" (accessed 2025-01-31)
- Google Search: "MViT vs DeiT vs Swin Transformer comparison benchmarks" (accessed 2025-01-31)

**GitHub Repository:**
- [facebook/mvit](https://github.com/facebookresearch/mvit) - Official implementation

**Related Vision Transformer Papers:**
- Vision Transformer (ViT) - Dosovitskiy et al., ICLR 2021
- Swin Transformer - Liu et al., ICCV 2021
- DeiT - Touvron et al., ICML 2021
