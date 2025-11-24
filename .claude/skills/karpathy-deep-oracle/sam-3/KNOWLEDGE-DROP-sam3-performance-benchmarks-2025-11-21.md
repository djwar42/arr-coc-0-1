# SAM 3 Performance Benchmarks

## Overview

SAM 3 delivers fast inference with excellent performance-efficiency tradeoffs, achieving 30 ms per image on high-end GPUs while handling 100+ detected objects simultaneously. The model supports real-time video processing at 30 FPS with multi-GPU parallelization.

---

## Image Inference Performance

### Single Image Latency

From [Meta AI Blog](https://ai.meta.com/blog/segment-anything-model-3/) (accessed 2025-11-23):

**30 ms per single image** with 100+ detected objects on H200 GPU

This enables:
- ~33 FPS theoretical throughput for single images
- Processing of over 100 concept instances per inference call
- Practical batch annotation workflows

From [OpenReview Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf):
- "On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects"

---

## Video Inference Performance

### Real-Time Video Requirements

From the [SAM 3 paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) and [GitHub Issue #155](https://github.com/facebookresearch/sam3/issues/155):

**30 FPS target for real-time video** (practical applications like web demos)

### Multi-GPU Scaling for Video

To achieve 30 FPS on video with multiple tracked objects:

| GPU Configuration | Max Tracked Objects | FPS |
|-------------------|---------------------|-----|
| 2x H200 | Up to 10 objects | 30 FPS |
| 4x H200 | Up to 28 objects | 30 FPS |
| 8x H200 | Up to 64 objects | 30 FPS |

### Single GPU Video Performance

From the paper:
- "The inference latency scales with the number of objects"
- "Sustaining near real-time performance for ~5 concurrent objects" on single GPU
- Linear scaling: more objects = more latency

**Estimated single H200 performance:**
- 5 objects: ~30 FPS (real-time)
- 10 objects: ~15 FPS
- 20 objects: ~7-8 FPS

---

## Model Size and Memory Requirements

### Model Parameters

From [Roboflow Blog](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23):

**~840 million parameters** (~3.4 GB model size)

This makes SAM 3 a "server-scale model" - not suitable for edge deployment without distillation.

### GPU VRAM Requirements

From [stable-learn.com](https://stable-learn.com/en/sam3-segment-anything-model-tutorial/):

**Recommended: 16GB+ VRAM** for the 848M parameter model

Minimum requirements:
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB+ system memory
- Storage: 50GB+ available space

### Model Size Comparison

| Model | Parameters | Size | Type |
|-------|------------|------|------|
| SAM 3 | ~840M | ~3.4 GB | Server-scale |
| SAM 2 Large | 224M | ~1 GB | Server-scale |
| SAM 2 Base | 80M | ~350 MB | Desktop |
| MobileSAM | 10M | ~40 MB | Mobile |

---

## Batch Size Scaling

### Batch Processing Characteristics

SAM 3 supports batched inference for throughput optimization:

**Key considerations:**
- VRAM scales linearly with batch size
- Optimal batch size depends on:
  - Available GPU memory
  - Number of objects per image
  - Resolution of input images

### Throughput Optimization

For dataset annotation workflows:
- Batch multiple images together
- Use larger batches when objects per image are few
- Reduce batch size for complex scenes with many objects

Estimated throughput (single H200, 1024x1024 images):
- Batch 1: ~33 images/second
- Batch 4: ~50-60 images/second (memory permitting)
- Batch 8: ~70-80 images/second (memory permitting)

*Note: Actual throughput varies based on number of detected objects and prompt complexity*

---

## Hardware Recommendations

### For Production Deployment

**Recommended GPUs:**
- NVIDIA H200 (80GB) - Best performance
- NVIDIA H100 (80GB) - Excellent performance
- NVIDIA A100 (80GB) - Good performance

**For real-time video (30 FPS):**
- 2-8x H200 GPUs depending on tracked object count
- High-bandwidth NVLink for multi-GPU setups

### For Development/Testing

**Minimum viable:**
- NVIDIA RTX 4090 (24GB)
- NVIDIA A10 (24GB)
- NVIDIA L40 (48GB)

*May require reduced batch sizes or fewer concurrent objects*

### Not Recommended

- GPUs with <16GB VRAM
- Edge devices without distilled models
- CPUs (extremely slow)

---

## EfficientSAM3: Lighter Alternatives

From [arXiv:2511.15833](https://arxiv.org/abs/2511.15833) (EfficientSAM3 paper):

For on-device/edge deployment, EfficientSAM3 provides distilled alternatives:

**Progressive Hierarchical Distillation (PHD)** produces:
- RepViT backbone variants
- TinyViT backbone variants
- EfficientViT backbone variants

These enable:
- On-device concept segmentation
- Lower VRAM requirements
- Maintained performance fidelity to SAM 3 teacher

---

## Performance Comparison vs Competitors

### SAM 3 vs Other Open-Vocabulary Models

From Meta benchmarks:
- **2x gain over existing systems** in both image and video PCS

SAM 3 significantly outperforms:
- OWLv2
- DINO-X
- Gemini 2.5

### SAM 3 vs SAM 2

| Aspect | SAM 2 | SAM 3 |
|--------|-------|-------|
| Text prompts | No | Yes |
| Open-vocabulary | No | Yes (270K concepts) |
| Instance detection | Single object | Multiple objects |
| Model size | ~224M | ~840M |
| Inference speed | Faster | Slightly slower |

---

## Benchmark Metrics

### Standard Evaluation Metrics

For image segmentation:
- **mAP** (mean Average Precision)
- **cgF1** (concept-grounded F1)

For video segmentation:
- **pHOTA** (prompt-conditioned HOTA)
- **HOTA** (Higher Order Tracking Accuracy)

### Key Benchmark Results

On SA-Co/Gold (image):
- SAM 3 achieves state-of-the-art on cgF1

On SA-V (video):
- Near-human performance on video concept segmentation

---

## Practical Deployment Considerations

### Latency Breakdown (Estimated)

For single image inference:
- Image encoding: ~10 ms
- Text/prompt encoding: ~2 ms
- Detection: ~10 ms
- Mask decoding: ~8 ms
- **Total: ~30 ms**

### Scaling Guidelines

**For annotation pipelines:**
- Use batched inference
- Pre-encode text prompts
- Cache image embeddings for multi-prompt scenarios

**For video applications:**
- Plan GPU requirements based on max tracked objects
- Consider multi-GPU parallelization
- Use streaming memory efficiently

**For API services:**
- Target <100ms total latency including network
- Use async processing for batches
- Consider request queuing for high load

---

## Sources

**Primary Sources:**
- [Meta AI Blog - SAM 3](https://ai.meta.com/blog/segment-anything-model-3/) (accessed 2025-11-23)
- [OpenReview Paper](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf)
- [GitHub Issue #155 - Performance Benchmark](https://github.com/facebookresearch/sam3/issues/155)

**Secondary Sources:**
- [Roboflow Blog - What is SAM 3](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-23)
- [EfficientSAM3 arXiv:2511.15833](https://arxiv.org/abs/2511.15833)
- [stable-learn SAM3 Tutorial](https://stable-learn.com/en/sam3-segment-anything-model-tutorial/)

**GitHub Repository:**
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
