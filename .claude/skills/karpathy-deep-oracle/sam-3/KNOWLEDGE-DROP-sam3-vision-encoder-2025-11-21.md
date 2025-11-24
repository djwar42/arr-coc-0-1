# SAM 3 Shared Vision Encoder

## Overview

SAM 3 uses a **shared vision encoder** (also called the Perception Encoder or PE) that serves as the backbone for both the detector and tracker components. This unified encoder is a critical architectural decision that enables memory efficiency and consistent feature extraction across both image detection and video tracking tasks.

## Architecture Details

### Encoder Type: Hierarchical Vision Transformer

SAM 3 builds on the **Hiera (Hierarchical) architecture** that was introduced in SAM 2. The Perception Encoder is a state-of-the-art vision encoder trained via vision-language learning that produces features suitable for both detection and tracking.

**Key characteristics:**
- **Hierarchical design**: Produces multi-scale features at different resolutions
- **Transformer-based**: Uses attention mechanisms for global context
- **Shared weights**: Same encoder parameters used by both detector and tracker
- **Pre-trained**: Leverages large-scale pre-training for robust visual representations

### Model Size

From [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3):
- SAM 3 total model size: **848M parameters**
- The vision encoder is shared between detector and tracker, contributing to efficient parameter usage

### Multi-Scale Feature Extraction

The shared encoder produces features at multiple scales that are used differently by the detector and tracker:

**For the Detector (DETR-based):**
- Multi-scale features fed to transformer encoder-decoder
- Enables detection of objects at various sizes
- Features conditioned on text/exemplar prompts via fusion encoder

**For the Tracker (SAM 2 architecture):**
- Features used for temporal propagation
- Memory attention operates on these features
- Enables consistent object tracking across video frames

## How Sharing Works

### Unified Feature Extraction Pipeline

From [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/):

> "SAM 3 consists of a detector and tracker that share a Perception Encoder (PE) vision backbone. This decoupled design avoids task conflicts while enabling both image-level detection and video-level tracking."

**Pipeline Flow:**
1. Input image/frame passes through shared encoder
2. Encoder produces hierarchical feature maps
3. **For Detection**: Features go to fusion encoder -> DETR decoder -> mask head
4. **For Tracking**: Features go to memory attention -> mask decoder -> memory encoder

### Decoupled Task Heads

While the encoder is shared, the task-specific heads are decoupled:
- **Detector**: DETR-based with text encoder, exemplar encoder, fusion encoder, presence head, mask head
- **Tracker**: Memory-based with prompt encoder, mask decoder, memory encoder, memory bank

This decoupling avoids task conflicts where detection and tracking objectives might interfere with each other.

## Memory Efficiency Benefits

### Parameter Efficiency

From [OpenReview SAM 3 Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf):

**Single backbone advantages:**
- No need to run separate encoders for detection and tracking
- Reduced total parameter count
- Consistent visual representations across tasks

### Computational Efficiency

**Inference benefits:**
- Encode image once, use features for both tasks
- Enables efficient pipeline: detect objects -> track through video
- 30ms per image with 100+ detected objects on H200 GPU

**Memory footprint:**
- Shared feature maps reduce GPU memory during inference
- Enables processing longer videos without memory overflow
- Efficient for real-time or near-real-time applications

### Unified Processing Pipeline

The shared encoder enables a seamless workflow:
1. **Image Mode**: Detector uses encoded features for all concept instances
2. **Video Mode**: Same features used by tracker for temporal propagation
3. **Interactive Mode**: Features reused when adding refinement prompts

## Connection to Perception Encoder (PE)

From [arXiv - Perception Encoder](https://arxiv.org/abs/2504.13181):

The Perception Encoder is Meta's state-of-the-art vision encoder for image and video understanding. Key aspects:

- **Training**: Vision-language learning approach
- **Architecture**: Hierarchical ViT design
- **Features**: Best visual embeddings extracted from intermediate layers (not output)
- **Applications**: Used as backbone in SAM 3 and other Meta vision models

## Comparison with SAM 2

### SAM 2 Architecture
- Used **Hiera pre-trained MAE encoder**
- Primarily designed for video object segmentation
- Required visual prompts (points, boxes, masks)

### SAM 3 Enhancements
- Same Hiera-style hierarchical architecture
- Enhanced to support concept-based detection
- Shared between detector AND tracker (dual purpose)
- Supports text and exemplar prompts in addition to visual prompts

## Technical Specifications

### Feature Maps

The shared encoder produces:
- **Multi-resolution features**: Different scales for multi-scale detection
- **Rich semantic features**: Capture object appearance and context
- **Temporally consistent features**: Enable reliable video tracking

### Integration Points

**Detector Integration:**
- Fusion encoder conditions features on text/exemplar prompts
- DETR transformer decoder uses features for proposal queries
- Presence token operates on global pooled features

**Tracker Integration:**
- Memory attention fuses features with object memory
- Mask decoder produces segmentation from encoded features
- Memory encoder creates compact representations for memory bank

## Why Shared Encoder Design?

### Advantages

1. **Consistency**: Same visual representations for detection and tracking
2. **Efficiency**: Single forward pass through encoder for both tasks
3. **Reduced parameters**: No duplication of encoder weights
4. **Better generalization**: Encoder learns features useful for multiple tasks
5. **Simplified pipeline**: Natural transition from detection to tracking

### Task Conflict Mitigation

Despite sharing the encoder, SAM 3 avoids task conflicts through:
- **Decoupled task heads**: Separate specialized components
- **Presence token**: Separates recognition from localization
- **Independent training objectives**: Detection and tracking can scale independently

## Performance Impact

From benchmark results:

**Inference Speed:**
- 30ms per image with 100+ objects (H200 GPU)
- Near real-time for ~5 concurrent video objects

**Quality:**
- LVIS zero-shot mask AP: 47.0
- MOSEv2 J&F: 60.1 (+25.5% over SAM 2.1)

The shared encoder architecture contributes to both speed and quality by providing:
- Efficient computation (single encoding)
- Strong features from large-scale pre-training
- Consistent representations across tasks

## Sources

**Web Research:**
- [GitHub - facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official repository with model description (accessed 2025-11-23)
- [Ultralytics SAM 3 Documentation](https://docs.ultralytics.com/models/sam-3/) - Comprehensive architecture overview (accessed 2025-11-23)
- [OpenReview SAM 3 Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) - Technical paper submitted to ICLR 2026 (accessed 2025-11-23)
- [arXiv - Perception Encoder](https://arxiv.org/abs/2504.13181) - PE encoder paper by Bolya et al. 2025 (accessed 2025-11-23)
- [AI at Meta - SAM 3 Research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) - Official publication page (accessed 2025-11-23)

**Additional References:**
- [HuggingFace SAM3 Tracker Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_tracker) - Tracker component details
- [Lightly AI - Segment Anything Model and Friends](https://www.lightly.ai/blog/segment-anything-model-and-friends) - SAM family comparison with Hiera details
