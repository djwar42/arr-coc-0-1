# SAM 2.1 Updates: Complete Changelog
**Release**: October 18, 2024 (11 weeks after SAM 2)
**Key improvement**: +2.8 J&F (78.2% → 81.0% on SA-V)
**Architecture**: UNCHANGED (same model, better training)
**Compatibility**: 100% backward compatible
**Developer suite**: Training code + web demo released

---

## Executive Summary

SAM 2.1 is an incremental update to SAM 2 that achieves significant performance improvements through enhanced training techniques **without changing the model architecture**. The key insight: same Hiera encoder, same streaming memory attention, same mask decoder—but trained better.

**What changed**: Training recipe (data augmentation, training strategy)
**What stayed the same**: Model architecture, API, inference code

---

## 1. Performance Improvements

### 1.1 J&F Metric Improvement (+2.8 Points)

**J&F Metric Explained**:
- **J (Jaccard)**: Region similarity/area accuracy (Intersection over Union)
- **F (F-measure)**: Boundary accuracy/contour precision
- **J&F**: Mean of both metrics (comprehensive video segmentation evaluation)

**The +2.8 Improvement**:
- **SAM 2** (July 2024): 78.2% J&F on SA-V dataset
- **SAM 2.1** (October 2024): 81.0% J&F on SA-V dataset
- **+2.8 points** = 3.6% relative improvement

**How it was achieved**:
1. Additional training data (more diverse video scenarios)
2. New data augmentation techniques (better for small objects, similar objects)
3. Enhanced training strategy (improved boundary and temporal consistency)

**Performance across all model sizes**:
| Model | SAM 2 J&F | SAM 2.1 J&F | Improvement |
|-------|-----------|-------------|-------------|
| Tiny  | 70.9%     | 73.3%       | +2.4        |
| Small | 73.0%     | 75.7%       | +2.7        |
| Base+ | 76.0%     | 79.5%       | +3.5        |
| Large | 76.0%     | 79.5%       | +3.5        |

**Key insight**: Speed maintained! Real-time inference preserved (39.5-91.2 FPS depending on model size).

### 1.2 Occlusion Handling Enhancements

**Challenge**: Objects disappearing behind obstacles and reappearing later in video

**SAM 2 approach**:
- Streaming memory attention mechanism
- Object pointers for tracking
- Memory persistence across frames

**SAM 2.1 improvements**:
1. **Longer frame sequences**: Extended temporal context for better occlusion reasoning
2. **Enhanced memory persistence**: Objects remembered longer across occlusion events
3. **Improved positional encoding**: Better spatial tracking when objects reappear
4. **Data augmentation**: Training includes synthetic occlusion scenarios

**Performance gains**:
- +3.5 J&F improvement on long-term occlusion benchmarks
- Better re-identification when objects reappear
- More robust to extended occlusions (>2 seconds)

**Real-world impact**:
- Autonomous driving: Track vehicles behind obstacles
- Sports: Track players behind crowd/other players
- Medical imaging: Track organs during occlusion by instruments
- Surveillance: Maintain ID across temporary occlusions

### 1.3 General Robustness Improvements

**Benchmark performance**:

**SA-V test** (video segmentation):
| Model | SAM 2 | SAM 2.1 | Improvement |
|-------|-------|---------|-------------|
| Large | 76.0  | 79.5    | +3.5        |
| Base+ | 76.0  | 79.5    | +3.5        |
| Small | 73.0  | 75.7    | +2.7        |
| Tiny  | 70.9  | 73.3    | +2.4        |

**MOSE** (multi-object segmentation):
- Improvement: +0.9 to +2.0 across models
- Better handling of crowded scenes
- Improved small object tracking

**LVOS v2** (long-term video object segmentation):
- Improvement: +0.8 to +2.4 across models
- Better temporal stability
- Fewer identity switches

**Edge cases improved**:
1. **Occlusions**: Objects going behind obstacles
2. **Long-term tracking**: Maintaining identity over >100 frames
3. **Crowded scenes**: Multiple similar objects
4. **Low contrast**: Objects similar to background
5. **Deformable objects**: Non-rigid transformations

**Efficiency gains**:
- **3× fewer interactions** required for same accuracy vs prior approaches
- Real-time speed maintained: 39.5-91.2 FPS (depending on model)
- Memory efficiency: Same memory footprint as SAM 2

---

## 2. Training & Data Improvements

### 2.1 Data Augmentation Techniques

**New augmentations added in SAM 2.1**:

**Spatial augmentations**:
- Enhanced random crops (focus on small objects)
- Multi-scale training (better scale invariance)
- Color jittering (robustness to lighting)
- Geometric transforms (rotation, shear, affine)

**Temporal augmentations**:
- **Longer frame sequences**: Extended from short clips to longer sequences
- Variable frame sampling rates
- Temporal consistency constraints

**Positional encoding adjustments**:
- Improved encoding for object tracking across frames
- Better handling of camera motion
- Enhanced spatial-temporal alignment

**Impact**:
- +2.8 J&F improvement (78.2% → 81.0%)
- Better generalization across domains
- Robust to diverse environments

**Comparison SAM 2 vs SAM 2.1**:
- **SAM 2**: Standard augmentations (crops, flips, color jitter)
- **SAM 2.1**: Enhanced + longer sequences + positional improvements

### 2.2 Training Code Release (Developer Suite)

**Location**: `github.com/facebookresearch/sam2/training/`

**What's included**:
1. Complete training scripts (multi-GPU, multi-node)
2. Dataset loaders (SA-V, custom datasets)
3. Training configurations (Hydra configs)
4. Fine-tuning guides (MOSE example provided)
5. Evaluation scripts
6. Documentation

**Repository structure**:
```
training/
├── train.py                # Main training script
├── configs/
│   ├── sam2.1/
│   │   ├── sam2.1_hiera_l.yaml     # Large model config
│   │   ├── sam2.1_hiera_b_plus.yaml
│   │   ├── sam2.1_hiera_s.yaml
│   │   └── sam2.1_hiera_t.yaml
│   └── datasets/
│       └── sa_v.yaml       # SA-V dataset config
├── dataset/
│   └── sam_dataset.py      # Dataset implementation
├── modeling/               # Model definitions
└── README.md               # Getting started guide
```

**Getting started** (fine-tuning example):
```bash
# 1. Prepare MOSE dataset
cd sam2/training
bash scripts/download_mose.sh

# 2. Fine-tune on MOSE (8× A100 80GB GPUs)
python train.py \
  --config-path configs/sam2.1 \
  --config-name sam2.1_hiera_l.yaml \
  dataset=mose

# 3. Evaluate
python train.py \
  --config-path configs/sam2.1 \
  --config-name sam2.1_hiera_l.yaml \
  mode=eval
```

**Hardware requirements**:
- Fine-tuning: 8× A100 80GB GPUs (single node)
- Training from scratch: Multiple nodes, weeks of compute (infeasible for most)
- Inference: Single GPU (T4, A100, H100, etc.)

**Key finding**: MOSE fine-tuning achieves **79.4 J&F** (state-of-the-art).

### 2.3 Training Improvements

**Optimizer configuration** (likely, based on code analysis):
- **Optimizer**: AdamW (standard for Vision Transformers)
- **Learning rate schedule**: Cosine decay with warmup
- **Weight decay**: Applied to non-bias parameters
- **Gradient clipping**: For training stability

**Mixed precision training**:
- **Format**: bfloat16 (evidence from inference examples + A100 optimization)
- **Benefits**: 2× speed, 50% memory reduction
- **Precision**: Maintains accuracy vs fp32

**Training infrastructure**:
- Multi-node SLURM support
- Distributed data parallel (DDP)
- Gradient accumulation for effective large batch sizes
- ZeRO optimizer (for memory efficiency)

**Training duration** (estimated):
- Fine-tuning (MOSE): ~1-2 days on 8× A100
- Full training (SA-V): Weeks on large clusters (not public)

**Loss function**: `MultiStepMultiMasksAndIous`
- Multi-scale supervision
- IoU prediction head
- Temporal consistency losses

**Dataset mixing**:
- SA-V (video) + SA-1B (images) mixed training
- Frame sampling strategies for video
- Augmentation applied per-sample

**What changed from SAM 2 → SAM 2.1**:
- Enhanced data augmentation (spatial + temporal)
- Longer training sequences (better occlusion handling)
- Improved training recipe (not architecture)
- **Result**: +2.8 J&F improvement!

---

## 3. Developer Suite

### 3.1 Overview

**Release date**: September 30, 2024 (same as SAM 2.1 checkpoints)

**What is the Developer Suite**:
Meta's complete development toolkit for SAM 2, enabling developers to:
- Train SAM 2 from scratch
- Fine-tune on custom datasets
- Deploy interactive web demos
- Build production applications

**Components**:
1. **Training code**: Complete training infrastructure (Hydra configs, multi-GPU support)
2. **Web demo code**: Interactive browser-based segmentation (React + Flask + GraphQL)

**Target audience**:
- ML researchers (custom training, ablation studies)
- Computer vision engineers (domain adaptation, fine-tuning)
- App developers (web demos, production APIs)
- Domain specialists (medical, autonomous vehicles, etc.)

**Community impact**:
- 700K+ downloads
- 17.8K GitHub stars
- Widespread adoption across industries

### 3.2 Web Demo Code Release

**Location**: `github.com/facebookresearch/sam2/demo/`

**Tech stack**:
- **Frontend**: React + TypeScript + Vite
- **Backend**: Python Flask + Strawberry GraphQL
- **Model serving**: ONNX Runtime (browser-based) or PyTorch (server-side)

**Live demo**: https://sam2.metademolab.com
- Real-time video segmentation
- Interactive refinement (points, boxes, masks)
- Multi-object tracking

**Local deployment**:
```bash
# Backend
cd sam2/demo/backend
pip install -r requirements.txt
python app.py

# Frontend
cd sam2/demo/frontend
npm install
npm run dev
```

**ONNX Runtime Web support**:
- Client-side inference (no server needed!)
- Works in modern browsers
- Model loaded once, runs offline
- Community implementations available

**Comparison SAM 1 vs SAM 2.1 demo**:
- **SAM 1**: Image-only segmentation
- **SAM 2/2.1**: Video object tracking + segmentation
- **Major upgrade**: Temporal tracking, memory attention

---

## 4. Technical Details

### 4.1 Model Architecture (NO CHANGES!)

**CRITICAL**: SAM 2.1 has **ZERO** architectural changes from SAM 2.

**Architecture components** (identical in SAM 2 and 2.1):
- **Hiera encoder**: Hierarchical vision transformer (4 stages)
- **Streaming memory attention**: Temporal propagation with FIFO memory
- **Mask decoder**: Lightweight decoder (4M params)
- **Prompt encoder**: Points, boxes, masks, text (future)

**Parameter counts** (identical):
| Model | Parameters |
|-------|-----------|
| Tiny  | 38.9M     |
| Small | 46.0M     |
| Base+ | 80.8M     |
| Large | 224.4M    |

**What improved**: Training recipe (data augmentation, longer sequences, better training strategy)
**What stayed same**: Every layer, every weight initialization, every architectural choice

**Checkpoint sizes** (identical):
- Tiny: ~156 MB
- Small: ~184 MB
- Base+: ~323 MB
- Large: ~898 MB

**Performance implications**:
- Same inference speed (30-44 FPS)
- Same memory footprint
- Same hardware requirements
- **But**: +2-3% accuracy improvement!

### 4.2 API & Backward Compatibility

**Breaking changes**: NONE (100% backward compatible)

**Migration required**: Trivial (just swap checkpoint files)

**API changes**: ZERO
- `sam2_video_predictor` - identical API
- `sam2_image_predictor` - identical API
- `sam2.build_sam2()` - identical function signature

**Checkpoint compatibility**:
- SAM 2 code can load SAM 2.1 weights ✅
- SAM 2.1 code can load SAM 2 weights ✅
- Fully interchangeable

**Migration guide** (trivial):
```python
# OLD (SAM 2)
from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor(
    "sam2_hiera_large.yaml",
    "sam2_hiera_large.pt"  # ← Old checkpoint
)

# NEW (SAM 2.1) - just change checkpoint path
predictor = build_sam2_video_predictor(
    "sam2_hiera_large.yaml",  # ← Same config!
    "sam2.1_hiera_large.pt"   # ← New checkpoint
)
# That's it! No code changes needed.
```

**Training code** (optional, not required for inference):
- Released September 30, 2024
- Located in `training/` folder
- Not needed unless fine-tuning

**Web demo** (optional):
- Released September 30, 2024
- Located in `demo/` folder
- Not needed unless deploying web interface

**Deprecations**: None

---

## 5. Comparison Tables

### 5.1 SAM 2 vs SAM 2.1 Feature Comparison

| Feature | SAM 2 (July 2024) | SAM 2.1 (October 2024) | Change |
|---------|-------------------|------------------------|--------|
| **Performance** |
| J&F (SA-V) | 78.2% | 81.0% | +2.8 |
| J&F (MOSE) | 77.3% | 79.4% | +2.1 |
| J&F (LVOS v2) | 75.0% | 77.4% | +2.4 |
| Occlusion handling | Good | Enhanced | Memory improvements |
| **Training** |
| Training code | Not released | Released | Developer suite |
| Data augmentation | Standard | Enhanced | Spatial + temporal |
| Training sequences | Short | Longer | Better occlusions |
| **Architecture** |
| Hiera encoder | ✓ | ✓ | Unchanged |
| Memory attention | ✓ | ✓ | Unchanged |
| Mask decoder | ✓ | ✓ | Unchanged |
| Parameters | 38.9M - 224.4M | 38.9M - 224.4M | Identical |
| **API** |
| sam2_video_predictor | ✓ | ✓ | Identical API |
| sam2_image_predictor | ✓ | ✓ | Identical API |
| Checkpoint format | v1.0 | v2.1 | Compatible |
| **Developer Tools** |
| Web demo code | Not released | Released | Developer suite |
| Training configs | Not available | Available | Hydra configs |
| Fine-tuning guide | No | Yes (MOSE example) | Documentation |

### 5.2 Performance Benchmarks (All Models)

**SA-V test** (primary benchmark):
| Model | SAM 2 J&F | SAM 2.1 J&F | Δ J&F | FPS (H100) |
|-------|-----------|-------------|-------|------------|
| Large | 76.0      | 79.5        | +3.5  | 39.5       |
| Base+ | 76.0      | 79.5        | +3.5  | 47.2       |
| Small | 73.0      | 75.7        | +2.7  | 58.8       |
| Tiny  | 70.9      | 73.3        | +2.4  | 91.2       |

**MOSE** (multi-object):
| Model | SAM 2 | SAM 2.1 | Δ |
|-------|-------|---------|---|
| Large | 77.3  | 79.4    | +2.1 |
| Base+ | 76.5  | 78.5    | +2.0 |

**LVOS v2** (long-term):
| Model | SAM 2 | SAM 2.1 | Δ |
|-------|-------|---------|---|
| Large | 75.0  | 77.4    | +2.4 |
| Base+ | 74.0  | 76.3    | +2.3 |

---

## 6. Key Takeaways

### What SAM 2.1 Is

✅ **SAM 2 with better training**
- Same architecture, enhanced training recipe
- +2.8 J&F improvement (78.2% → 81.0%)
- 100% backward compatible

✅ **Developer-friendly release**
- Training code released (fine-tune on custom data)
- Web demo code released (deploy interactive UI)
- Complete documentation

✅ **Drop-in replacement for SAM 2**
- Just swap checkpoint files
- No code changes needed
- Immediate performance boost

### What SAM 2.1 Is NOT

❌ **Not a new architecture**
- Zero architectural changes
- Same Hiera encoder, memory attention, decoder
- Improvements from training only

❌ **Not a breaking change**
- 100% API compatible
- Checkpoint compatible
- No migration needed

❌ **Not requiring training code**
- Training code is optional
- Only needed for fine-tuning
- Inference works out-of-box

### When to Use SAM 2.1

**Use SAM 2.1 if**:
- Starting new project (always use latest)
- Need best accuracy (+2.8 J&F improvement)
- Want to fine-tune on custom data (training code available)
- Building production system (SOTA performance)

**Stay on SAM 2 if**:
- Existing production system works fine
- No need for +2.8% improvement
- Avoiding revalidation costs
- Using custom fine-tuned SAM 2 checkpoints

### How to Upgrade

**One-line upgrade**:
```python
# Change this:
checkpoint = "sam2_hiera_large.pt"

# To this:
checkpoint = "sam2.1_hiera_large.pt"

# Done!
```

---

## 7. Sources & References

**Official sources**:
1. Meta AI Blog: [Segment Anything 2.1 announcement](https://ai.meta.com/blog/fair-news-segment-anything-2-1)
2. GitHub: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
3. Paper: [Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
4. HuggingFace: [facebook/sam2 model hub](https://huggingface.co/facebook/sam2)

**Community resources**:
5. Roboflow: [SAM 2.1 guide](https://blog.roboflow.com/sam-2-1/)
6. Encord: [SAM 2.1 technical analysis](https://encord.com/blog/sam-2-1/)
7. Ultralytics: [SAM 2.1 tutorial](https://docs.ultralytics.com/models/sam-2/)

**Benchmarks**:
8. SA-V dataset: [Official evaluation](https://ai.meta.com/datasets/segment-anything-video)
9. MOSE: [Multi-object segmentation benchmark](https://henghuiding.github.io/MOSE/)
10. LVOS v2: [Long-term video object segmentation](https://lingyihongfd.github.io/lvos.github.io/)

**All sources accessed**: 2025-11-21

---

## Changelog Summary

**Version**: SAM 2.1
**Release**: October 18, 2024
**Improvements**: Training-only (no architecture changes)

**Performance**:
- ✅ +2.8 J&F on SA-V (78.2% → 81.0%)
- ✅ +3.5 J&F on long-term occlusion benchmarks
- ✅ 3× fewer interactions required

**Training**:
- ✅ Enhanced data augmentation (spatial + temporal)
- ✅ Longer frame sequences (better occlusions)
- ✅ Training code released (fine-tuning guide)

**Developer tools**:
- ✅ Complete training infrastructure
- ✅ Web demo code (React + Flask)
- ✅ 700K+ downloads, 17.8K stars

**Compatibility**:
- ✅ 100% backward compatible
- ✅ Zero API changes
- ✅ Drop-in replacement

**Migration**:
- ✅ Trivial (swap checkpoint files)
- ✅ No code changes needed

🎉 **SAM 2.1 = SAM 2 architecture + better training!** 🎉
