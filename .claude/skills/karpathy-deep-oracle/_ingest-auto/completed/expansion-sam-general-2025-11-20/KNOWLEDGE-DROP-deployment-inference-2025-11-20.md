# SAM: Deployment & Inference Optimization

**PART 35/42 - Production Deployment Patterns**

**Date**: 2025-11-20
**Source**: SAM deployment guide, inference optimization techniques

---

## Inference Pipeline Overview

**Standard SAM Inference**:
```python
# 1. Load model (one-time)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
sam.eval()

# 2. Encode image (expensive, ~180ms on A100)
image_embedding = sam.image_encoder(image)  # 64×64×1,280 features

# 3. Generate masks for prompt (cheap, ~5ms per prompt)
masks = sam.mask_decoder(
    image_embeddings=image_embedding,
    prompt_embeddings=prompt_encoder(point)
)
```

**Key Insight**: Encoder runs once, decoder reused for multiple prompts!

---

## Deployment Options

### 1. Cloud API (Hosted Service)

**Architecture**: REST API on GCP/AWS

**Endpoint**:
```bash
POST /v1/segment
{
  "image_url": "https://example.com/image.jpg",
  "prompts": [
    {"type": "point", "x": 120, "y": 80, "label": 1},
    {"type": "box", "bbox": [100, 50, 200, 150]}
  ]
}
```

**Response**:
```json
{
  "masks": [
    {"mask": "base64_encoded_mask", "score": 0.94, "bbox": [98, 48, 203, 152]},
    {"mask": "base64_encoded_mask", "score": 0.88, "bbox": [100, 50, 200, 150]},
    {"mask": "base64_encoded_mask", "score": 0.76, "bbox": [95, 45, 210, 160]}
  ],
  "inference_time_ms": 185
}
```

**Pros**:
- No local GPU required
- Managed scaling (auto-scale on demand)
- Easy integration (HTTP API)

**Cons**:
- Latency (network round-trip + inference)
- Cost (per-request pricing)
- Privacy (image data sent to cloud)

### 2. On-Premise GPU Server

**Setup**: Single A100 GPU server

**Serving Framework**: TorchServe or FastAPI

**Example (FastAPI)**:
```python
from fastapi import FastAPI, UploadFile
from segment_anything import sam_model_registry

app = FastAPI()
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").cuda()

@app.post("/segment")
async def segment(image: UploadFile, prompts: list):
    image_tensor = preprocess(image)
    embedding = sam.image_encoder(image_tensor)  # Cache this!

    results = []
    for prompt in prompts:
        masks = sam.mask_decoder(embedding, encode_prompt(prompt))
        results.append(masks)

    return {"masks": results}
```

**Pros**:
- Low latency (~185ms end-to-end)
- Data privacy (no cloud upload)
- Full control over model/infrastructure

**Cons**:
- Hardware cost (A100 = $10K)
- Manual scaling (limited throughput)

### 3. Edge Device (Mobile/Embedded)

**Models**: SAM-Mobile, MobileSAM (distilled variants)

**Hardware**: NVIDIA Jetson, Apple Neural Engine, Qualcomm Snapdragon

**Example (ONNX Runtime)**:
```python
import onnxruntime as ort

# Load quantized INT8 model
session = ort.InferenceSession("sam_mobile_int8.onnx")

# Run inference (50ms on Jetson AGX Orin)
masks = session.run(None, {"image": image, "prompts": prompts})[0]
```

**Pros**:
- Ultra-low latency (<50ms)
- No network dependency (offline capable)
- Scales to millions of devices

**Cons**:
- Accuracy loss (distillation + quantization)
- Memory constraints (embedded GPUs have 4-8GB)

---

## Optimization Techniques

### 1. TensorRT Optimization

**What**: NVIDIA's inference optimizer (kernel fusion, precision tuning)

**Steps**:
```bash
# Export PyTorch to ONNX
python export_onnx.py --checkpoint sam_vit_h.pth --output sam.onnx

# Optimize with TensorRT
trtexec --onnx=sam.onnx \
        --saveEngine=sam.trt \
        --fp16 \  # Mixed precision
        --workspace=4096  # 4GB scratch memory
```

**Performance**:
- PyTorch: 185ms (A100)
- TensorRT FP16: 92ms (A100) → **2× speedup**
- TensorRT INT8: 58ms (A100) → **3.2× speedup, -1.2% mIoU**

### 2. Model Distillation (SAM-Fast)

**Approach**: Train smaller student model to mimic SAM (teacher)

**Student Architecture**:
- Encoder: ViT-B (86M params) instead of ViT-H (630M)
- Decoder: Same (6M params)
- **Total**: 92M params vs. 636M (7× smaller)

**Training**:
```python
# Distillation loss
teacher_masks = sam_teacher(image, prompts)
student_masks = sam_student(image, prompts)

loss_distill = MSE(student_masks, teacher_masks.detach())
loss_gt = focal_loss(student_masks, gt_masks) + dice_loss(student_masks, gt_masks)

loss_total = 0.7 * loss_distill + 0.3 * loss_gt
```

**Performance**:
- SAM (ViT-H): 185ms, 50.3 mIoU (COCO)
- SAM-Fast (ViT-B): 52ms, 47.1 mIoU → **3.6× faster, -3.2 mIoU**

### 3. INT8 Quantization

**Method**: Reduce weight precision (FP32 → INT8)

**Quantization-Aware Training (QAT)**:
```python
# Insert fake quantization ops during training
model = torch.quantization.prepare_qat(sam, inplace=True)

# Train for 10 epochs (fine-tune)
for epoch in range(10):
    for batch in dataloader:
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()

# Convert to INT8
model_int8 = torch.quantization.convert(model, inplace=True)
```

**Performance**:
- FP32: 185ms, 4.8GB memory
- INT8: 58ms, 1.2GB memory → **3.2× faster, 4× less memory, -1.2% mIoU**

### 4. Batch Inference

**Scenario**: Process 100 images with 1 prompt each

**Sequential**:
```python
# Total: 100 × 185ms = 18,500ms (18.5 seconds)
for image in images:
    embedding = sam.image_encoder(image)  # 180ms
    mask = sam.mask_decoder(embedding, prompt)  # 5ms
```

**Batched**:
```python
# Batch encode images (amortized cost)
embeddings = sam.image_encoder(torch.stack(images))  # 4,200ms for 100 images

# Batch decode masks
masks = sam.mask_decoder(embeddings, prompt)  # 500ms for 100 images

# Total: 4,200 + 500 = 4,700ms (4.7 seconds) → 4× faster!
```

**Benefit**: GPU parallelism across batch dimension

### 5. Cached Embeddings

**Use Case**: Interactive annotation (user adds multiple prompts to same image)

**Pattern**:
```python
# First prompt: Full inference (180ms encoder + 5ms decoder)
embedding = sam.image_encoder(image)  # 180ms
mask1 = sam.mask_decoder(embedding, prompt1)  # 5ms

# Store embedding in cache
cache[image_id] = embedding

# Subsequent prompts: Skip encoder! (5ms decoder only)
mask2 = sam.mask_decoder(cache[image_id], prompt2)  # 5ms
mask3 = sam.mask_decoder(cache[image_id], prompt3)  # 5ms

# Total for 3 prompts: 180 + 5 + 5 + 5 = 195ms
# vs. 185 × 3 = 555ms (no cache) → 2.8× faster!
```

**Cache Invalidation**: Clear cache when image changes or after N minutes

---

## Throughput Benchmarks

**Single A100 GPU**:

| Optimization | Latency (ms) | Throughput (images/sec) | Accuracy (COCO mIoU) |
|--------------|--------------|-------------------------|----------------------|
| PyTorch FP32 | 185 | 5.4 | 50.3 |
| TensorRT FP16 | 92 | 10.9 | 50.1 (-0.2) |
| TensorRT INT8 | 58 | 17.2 | 49.1 (-1.2) |
| SAM-Fast (ViT-B) | 52 | 19.2 | 47.1 (-3.2) |
| SAM-Fast + TensorRT INT8 | 28 | 35.7 | 45.8 (-4.5) |

**Insight**: Trade-off between speed and accuracy (TensorRT INT8 = sweet spot for most applications)

---

## Memory Optimization

### 1. Gradient Checkpointing (Training Only)

**Benefit**: 60% less memory during training

**Trade-off**: 30% slower training

### 2. Mixed Precision (Inference)

**FP16 vs. FP32**:
- FP32: 4.8GB VRAM (encoder + decoder)
- FP16: 2.4GB VRAM → **50% less memory**

**Benefit**: Can run on lower-end GPUs (RTX 3090, RTX 4080)

### 3. Model Pruning

**Method**: Remove less important weights (structured pruning)

**Example**:
```python
# Prune 30% of attention heads
from torch.nn.utils import prune

for layer in sam.image_encoder.transformer_blocks:
    prune.l1_unstructured(layer.attn.qkv, amount=0.3)
```

**Performance**:
- 30% pruning: 3.3GB VRAM, 48.7 mIoU (COCO) → -1.6 mIoU
- 50% pruning: 2.4GB VRAM, 45.2 mIoU → -5.1 mIoU

---

## Deployment Patterns

### 1. Interactive Annotation Tool

**Workflow**:
1. User uploads image → Encode once (cache embedding)
2. User clicks points → Decode masks (5ms per prompt)
3. User refines → Add correction points → Re-decode

**Optimization**: Cache embeddings (critical for responsiveness)

### 2. Batch Dataset Annotation

**Workflow**:
1. Automatic mask generation (grid-based inference)
2. Human review (accept/reject/refine)
3. Export annotations (COCO format)

**Optimization**: Batch inference (process 100 images in parallel)

### 3. Real-Time Video Segmentation

**Challenge**: 30 FPS = 33ms per frame (SAM baseline = 185ms)

**Solution**:
- SAM 2 (temporal memory) → 42ms per frame on A100
- TensorRT FP16 → 22ms per frame → **45 FPS!**

### 4. Robotic Vision

**Challenge**: Embedded GPU (Jetson AGX Orin)

**Solution**:
- SAM-Mobile (distilled + quantized) → 48ms per frame
- 20 FPS segmentation for grasping/navigation

---

## Cost Analysis (Cloud Deployment)

**A100 GPU Instance** (AWS p4d.24xlarge):
- Cost: $32.77/hour
- Throughput: 10.9 images/sec (TensorRT FP16)
- **Cost per 1M images**: $836

**Optimization Impact**:
- PyTorch FP32: $1,683 per 1M images
- TensorRT FP16: $836 → **50% cost reduction**
- TensorRT INT8: $528 → **69% cost reduction**

**Break-even**: Distillation + quantization pays for itself after ~100K images

---

## Monitoring and Logging

**Metrics to Track**:
- **Latency**: P50, P95, P99 (ms per inference)
- **Throughput**: Images/sec, prompts/sec
- **GPU Utilization**: % busy (should be >80%)
- **Memory Usage**: VRAM (should be <90% to avoid OOM)

**Tools**:
- Prometheus + Grafana (metrics dashboard)
- NVIDIA Nsight Systems (GPU profiling)
- TensorBoard (model performance)

**Alerts**:
- P95 latency > 200ms → Scale up
- GPU util < 50% → Scale down
- Memory > 90% → Risk of OOM

---

## ARR-COC Integration (5%)

### Deployment as Relevance Accessibility

**Connection**: Inference optimization = **making relevance realization accessible** (latency, cost, scalability).

**ARR-COC Mapping**:
1. **Propositional knowing**: Model distillation preserves core knowledge (what is object vs. background)
2. **Perspectival knowing**: Quantization trades precision for speed (perspectival trade-off)
3. **Participatory knowing**: Interactive annotation = real-time co-creation (requires low latency!)

**Insight**: Deployment optimization enables participatory knowing at scale (humans + AI co-realize relevance efficiently).

---

## Key Takeaways

1. **Inference Pipeline**: Encode once (180ms) → Decode many (5ms per prompt)
2. **Optimization**: TensorRT FP16 (2× faster), INT8 (3.2× faster, -1.2% mIoU)
3. **Distillation**: SAM-Fast (ViT-B) → 3.6× faster, -3.2% mIoU
4. **Deployment**: Cloud API (scalable), on-prem (low latency), edge (offline)
5. **ARR-COC**: Optimization enables accessible relevance realization (participatory knowing at scale)

**Next**: PART 36-42 (Final 7 PARTs - SAM Applications, Limitations, Future Directions)

---

**References**:
- SAM deployment guide (Meta AI)
- TensorRT optimization: NVIDIA docs
- Model distillation: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- INT8 quantization: NVIDIA Quantization Toolkit
