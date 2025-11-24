# SAM 2 Training Pipeline: End-to-End

**"How to Train a Video Segmentation Foundation Model"**
**Training data**: SA-1B (images) + SA-V (videos)
**Training time**: ~7 days on 256× A100 GPUs
**Cost**: ~$50k (estimated)
**Result**: SOTA video segmentation, zero-shot generalization

---

## Overview

SAM 2 training is a **3-stage process**:
1. **Image pre-training** (SA-1B dataset)
2. **Video fine-tuning** (SA-V dataset)
3. **Joint training** (SA-1B + SA-V mixed)

### Why Multi-Stage?

**Stage 1 (Images)**: Learn spatial segmentation
- Cheaper to annotate images than videos
- SA-1B has 1.1B masks (massive scale)
- Initializes encoder with strong visual features

**Stage 2 (Videos)**: Learn temporal tracking
- Add memory encoder/attention
- Train on SA-V (50.9k videos, 642k masklets)
- Learn occlusion handling, temporal consistency

**Stage 3 (Joint)**: Maintain image quality + improve video
- Mix image + video data
- Prevents catastrophic forgetting (image skills)
- Final stage produces best model

---

## Stage 1: Image Pre-Training (SA-1B)

### Dataset

**SA-1B** (Segment Anything 1 Billion):
- 11M images
- 1.1B masks
- Collected using SAM 1's annotation loop
- Diverse: web images, permissive licenses

### Training Configuration

**Model components (Stage 1):**
```
┌──────────────────────────────────────┐
│  Hiera Encoder (image features)      │
│  Prompt Encoder (clicks, boxes)      │
│  Mask Decoder (segmentation)         │
│  ❌ NO Memory Encoder (images only)  │
└──────────────────────────────────────┘
```

**Hyperparameters:**
- **Model**: Hiera-B+ (256M params), Hiera-L (1.08B params)
- **Batch size**: 256 images × 256 GPUs = 65,536 images/batch (!)
- **Optimizer**: AdamW (lr=1e-4, weight decay=0.1)
- **Epochs**: ~40 epochs over SA-1B
- **Duration**: ~5 days on 256× A100 GPUs

### Training Procedure

**1. Data sampling:**
```python
# Per training step:
1. Sample 256 images from SA-1B
2. For each image, sample 1-3 masks randomly
3. For each mask, sample prompt type (50% click, 30% box, 20% mask)
4. Generate prompt embeddings
5. Forward pass → Predict masks
6. Compute loss, backprop
```

**2. Loss functions:**
```
Total Loss = λ₁·Focal Loss + λ₂·Dice Loss + λ₃·IoU Loss

Where:
  Focal Loss: Handles class imbalance (fg/bg)
  Dice Loss: Boundary accuracy
  IoU Loss: Confidence calibration

Weights: λ₁=1.0, λ₂=1.0, λ₃=1.0
```

**3. Data augmentation:**
- Random crop/resize (0.8-1.2×)
- Horizontal flip (50%)
- Color jitter (brightness, contrast, saturation)
- Gaussian blur (10%)

### Stage 1 Output

**Checkpoint after Stage 1:**
- Hiera encoder: Strong visual features
- Mask decoder: Accurate image segmentation
- Performance: 91.1% on COCO (image segmentation)

**What's missing?**
- No temporal understanding
- No memory encoder
- Can't track objects across frames

---

## Stage 2: Video Fine-Tuning (SA-V)

### Dataset

**SA-V** (Segment Anything Video):
- 50,900 videos (~140 hours)
- 642,036 masklets (temporal tracks)
- Diverse: 47 countries, various scenes

### Training Configuration

**Model components (Stage 2):**
```
┌──────────────────────────────────────┐
│  Hiera Encoder (frozen or fine-tuned)│
│  Prompt Encoder (frozen)             │
│  Mask Decoder (fine-tuned)           │
│  ✅ Memory Encoder (NEW! trained)    │
│  ✅ Memory Attention (NEW! trained)  │
└──────────────────────────────────────┘
```

**Key change**: Add memory encoder/attention, train on videos!

**Hyperparameters:**
- **Batch size**: 64 videos × 256 GPUs = 16,384 videos/batch
- **Temporal window**: 8-16 frames per video
- **Optimizer**: AdamW (lr=5e-5, lower than Stage 1)
- **Epochs**: ~10 epochs over SA-V
- **Duration**: ~2 days on 256× A100 GPUs

### Training Procedure

**1. Temporal window sampling:**
```python
# Per training step:
1. Sample video from SA-V
2. Sample starting frame t
3. Sample temporal window: [t, t+1, ..., t+T] where T=8-16
4. Sample masklets within window (objects to track)
5. For frame t: Sample prompt (click/box)
6. For frames t+1...t+T: Propagate via memory
7. Compute temporal loss (consistency)
```

**2. Memory bank simulation:**
```
Frame t:
  → Encode frame → Predict mask → Store in memory

Frame t+1:
  → Encode frame → Attend to memory [frame_t]
  → Predict mask → Store in memory

Frame t+2:
  → Encode frame → Attend to memory [frame_t, frame_t+1]
  → Predict mask → Store in memory

...

Frame t+8:
  → Encode frame → Attend to full memory bank
  → Predict mask
```

**3. Loss functions:**
```
Total Loss = λ₁·Mask Loss + λ₂·Temporal Consistency Loss

Where:
  Mask Loss: Focal + Dice + IoU (same as Stage 1)
  Temporal Consistency Loss: Penalize sudden mask changes

Temporal Loss:
  L_temporal = Σ ||mask_t - mask_{t-1}||²
  (Encourages smooth tracking)

Weights: λ₁=1.0, λ₂=0.5
```

**4. Occlusion training:**
```
Randomly drop frames (simulate occlusions):
  Frame t: Visible
  Frame t+1 to t+5: ❌ Dropped (occluded)
  Frame t+6: Visible again

Model must:
  → Remember object during occlusion
  → Re-identify when reappears
```

### Stage 2 Output

**Checkpoint after Stage 2:**
- Memory encoder trained
- Mask decoder adapted for video
- Performance: 78.2% J&F on YouTube-VOS (video segmentation)

**What's improved?**
- Temporal tracking
- Occlusion handling
- Video understanding

**What's degraded?**
- Image segmentation: 89.3% on COCO (dropped 1.8% from Stage 1!)
- **Problem**: Catastrophic forgetting (model forgets image-only tasks)

---

## Stage 3: Joint Training (SA-1B + SA-V)

### Motivation

**Problem**: Video fine-tuning degrades image performance
- Stage 1: 91.1% (image)
- Stage 2: 89.3% (image), 78.2% (video)
- **Catastrophic forgetting** of image-only skills

**Solution**: Joint training with mixed data

### Training Configuration

**Data mixing:**
```
Training batch (256 samples):
  → 128 images from SA-1B
  → 128 videos from SA-V

Alternating steps:
  Step 1: Image batch (no memory)
  Step 2: Video batch (with memory)
  Step 3: Image batch
  Step 4: Video batch
  ...
```

**Hyperparameters:**
- **Batch size**: 256 samples (128 images + 128 videos)
- **Optimizer**: AdamW (lr=2e-5, even lower)
- **Epochs**: ~5 epochs over mixed data
- **Duration**: ~1 day on 256× A100 GPUs

### Training Procedure

**Dynamic memory enabling:**
```python
for batch in mixed_data:
    if batch_type == "image":
        # Disable memory encoder/attention
        model.memory_enabled = False
        loss = compute_image_loss(batch)

    elif batch_type == "video":
        # Enable memory encoder/attention
        model.memory_enabled = True
        loss = compute_video_loss(batch)

    loss.backward()
    optimizer.step()
```

**Why this works:**
- Model learns: "Memory for videos, no memory for images"
- Maintains image skills (SA-1B refresher)
- Improves video understanding (more SA-V training)

### Stage 3 Output

**Final checkpoint:**
- **Image**: 91.5% on COCO (recovered + improved!)
- **Video**: 82.5% J&F on YouTube-VOS (4.3% gain over Stage 2)

**Best of both worlds!**

---

## Full Training Timeline

### Hardware Requirements

**GPUs**: 256× NVIDIA A100 80GB
- Total VRAM: 20TB (!)
- Total compute: ~5,120 A100-hours per day

**Storage**: 10TB+
- SA-1B: 5TB (compressed)
- SA-V: 2TB (videos)
- Checkpoints: 3TB (model states)

### Timeline Breakdown

| Stage | Duration | Dataset | Result |
|-------|----------|---------|--------|
| **Stage 1** | 5 days | SA-1B (images) | Image segmentation: 91.1% |
| **Stage 2** | 2 days | SA-V (videos) | Video: 78.2%, Image: 89.3% |
| **Stage 3** | 1 day | Mixed (SA-1B + SA-V) | Video: 82.5%, Image: 91.5% |
| **Total** | **8 days** | **1.1B + 642k masks** | **SOTA both tasks** |

**Total cost**: ~$50k USD (estimated 256× A100 @ $2.50/hr/GPU)

---

## Training Tricks & Optimizations

### 1. Mixed Precision Training (FP16/BF16)

**PyTorch AMP (Automatic Mixed Precision):**
```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2× faster training
- 2× less VRAM usage
- No accuracy loss (tested on SAM 2)

### 2. Gradient Accumulation

**Small GPU memory? Accumulate gradients:**
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Simulates larger batch size without more VRAM**

### 3. Distributed Data Parallel (DDP)

**Multi-GPU training:**
```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False  # Memory optimization
)
```

**256× A100 scaling**:
- Linear scaling up to ~128 GPUs
- Diminishing returns beyond 128 (communication overhead)
- SAM 2 used 256 GPUs (8 nodes × 32 GPUs/node)

### 4. Checkpoint Sharding

**Large model (1.08B params) → shard checkpoints:**
```python
# Save checkpoint across multiple files
torch.save(model.encoder.state_dict(), "encoder.pth")
torch.save(model.decoder.state_dict(), "decoder.pth")
torch.save(model.memory.state_dict(), "memory.pth")
```

**Benefits:**
- Faster save/load (parallel I/O)
- Resume training from partial failure

---

## Evaluation During Training

### Validation Metrics

**Every 1000 steps:**
- **Image**: COCO val (mAP, IoU)
- **Video**: YouTube-VOS val (J&F score)
- **Occlusion**: MOSE val (challenging occlusions)

**Early stopping criteria:**
- Video J&F plateaus for 3 checkpoints
- Image mAP drops >2% (catastrophic forgetting detected)

### Tensorboard Logging

**Logged metrics:**
- Train loss (mask, temporal, IoU)
- Val metrics (COCO, YouTube-VOS)
- Learning rate schedule
- GPU utilization, memory usage

**Example log:**
```
Step 10000:
  train/mask_loss: 0.15
  train/temporal_loss: 0.08
  val/coco_map: 0.911
  val/ytvos_jf: 0.825
  gpu/memory_gb: 72.3
```

---

## Reproducing SAM 2 Training

### Minimal Reproduction (Single GPU)

**Can't afford 256× A100?**

**Scaled-down version:**
- Model: Hiera-Tiny (50M params)
- Data: 1% of SA-1B + SA-V (random sample)
- GPU: 1× A100 (80GB)
- Duration: ~7 days
- **Result**: ~70% of full SAM 2 quality

**Training script (pseudocode):**
```python
# Stage 1: Image pre-training
train_on_sa1b(model, epochs=40, batch_size=256)

# Stage 2: Video fine-tuning
train_on_sav(model, epochs=10, batch_size=64)

# Stage 3: Joint training
train_mixed(model, epochs=5, batch_size=128)
```

### Full Reproduction

**Requirements:**
- GPUs: 256× A100 80GB ($50k cost)
- Data: SA-1B + SA-V (~7TB)
- Code: Official SAM 2 training repo (GitHub)
- Time: 8 days

**Meta AI released:**
- ✅ Pre-trained checkpoints (Hiera-B+, Hiera-L)
- ✅ Datasets (SA-1B, SA-V)
- ❌ Training code (NOT released yet, as of 2024)

**Why no training code?**
- Competitive advantage (Meta AI)
- Complexity (256-GPU distributed training)
- Expected to release later (research timeline)

---

## Key Takeaways

1. **3-stage training**: Images → Videos → Mixed (prevents catastrophic forgetting)
2. **Massive scale**: 256× A100 GPUs, 8 days, $50k cost
3. **Mixed precision**: FP16/BF16 for 2× speedup
4. **Temporal loss**: Smooth tracking across frames
5. **Occlusion training**: Drop frames to simulate occlusions

**Training SAM 2 is expensive but achieves SOTA on both image and video segmentation!**

---

## Future Improvements

### 1. Efficient Training

**Current**: 256× A100 (expensive!)
**Future**: Distillation to smaller models
- Teacher: SAM 2 (1.08B params)
- Student: SAM 2-Tiny (50M params)
- Same quality, 10× cheaper training

### 2. Continual Learning

**Current**: Fixed dataset (SA-1B + SA-V)
**Future**: Online learning from user corrections
- Deploy SAM 2 → Collect user feedback
- Retrain incrementally (no catastrophic forgetting)

### 3. Multi-Task Training

**Current**: Segmentation only
**Future**: Segmentation + depth + pose + semantic labels
- Share encoder across tasks
- Joint optimization

---

## References

- SAM 2 Paper: "SAM 2: Segment Anything in Images and Videos" (arXiv 2024)
- Training details: Section 4 of paper
- SA-1B dataset: SAM 1 paper (2023)
- SA-V dataset: Released with SAM 2 (2024)
- Meta AI Blog: https://ai.meta.com/sam2/
- GitHub: https://github.com/facebookresearch/sam2
