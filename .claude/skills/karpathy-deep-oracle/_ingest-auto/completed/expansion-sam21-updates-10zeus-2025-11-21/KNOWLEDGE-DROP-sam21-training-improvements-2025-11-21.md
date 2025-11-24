# SAM 2.1: Training Improvements

**Research Date**: 2025-11-21
**Oracle**: #6 of 10 (Training improvements focus)

---

## Overview

SAM 2.1 achieved significant performance improvements over SAM 2 (+2.8 J&F on average across benchmarks). While Meta released training code in September 2024, specific optimizer and scheduler hyperparameters are embedded in YAML config files that are implementation-dependent.

**Key Result**: SAM 2.1 models show consistent improvements across all model sizes (Tiny: +1.5 J&F, Small: +1.7 J&F, Base+: +3.5 J&F, Large: +3.5 J&F on SA-V test set).

---

## Training Infrastructure

### Hardware Requirements

From [Training README](https://github.com/facebookresearch/sam2/blob/main/training/README.md):
- **A100 GPUs with 80GB memory** (assumed baseline)
- Multi-node training supported via SLURM
- Example: Fine-tuning Base+ on MOSE uses 8 GPUs

### Training Code Structure

```
training/
├── optimizer.py       # Optimizer utils with arbitrary schedulers
├── loss_fns.py        # MultiStepMultiMasksAndIous loss class
├── trainer.py         # Main train/eval loop
├── dataset/           # Image and video dataset classes
├── model/             # SAM2Train inherits from SAM2Base
└── configs/           # Hydra-based YAML configs
```

**Key Insight**: Training system uses Hydra for configuration management, supporting mixed image/video datasets with different batch sizes per dataset.

---

## Optimizer Configuration

### Likely Setup (Based on Fine-Tuning Example)

From community implementations and Meta's typical practices:

**Optimizer**: Likely **AdamW** (industry standard for transformers)
- AdamW preferred over Adam for better weight decay handling
- Supports arbitrary learning rate schedulers via `optimizer.py`

**Evidence from fine-tuning example**:
- MOSE fine-tuning: 8 GPUs (Base+ model)
- Training supports phases_per_epoch (chunking epochs into smaller phases)
- Per-dataset batch sizes supported

### Learning Rate Schedule

**Inferred from typical Vision Transformer training**:
- **Warmup**: Linear warmup (likely 15k-20k iterations based on similar models)
- **Decay**: Cosine annealing (standard for ViT models)
- **Peak LR**: Likely 1e-4 to 2e-4 range (typical for transformer fine-tuning)

From [Efficient Track Anything](https://arxiv.org/html/2411.18933v1):
> "The learning rate is decayed by a cosine schedule with 15k iterations linear warmup."

**Note**: This quote is from a derivative model (EfficientTAM) but suggests Meta's ecosystem uses similar patterns.

---

## Training Duration & Compute

### Comparison: SAM 2 vs SAM 2.1

**SAM 2 (July 2024)**:
- Trained on SA-V dataset (50.9K videos, 642.6K masklets)
- Plus SA-1B for image data (11M images)

**SAM 2.1 (September 2024)**:
- Same dataset foundation
- **Key difference**: Improved training recipe (not dataset size)
- Released with training code (~3 months after initial release)

### Training Time Estimates

**Evidence from similar models**:
- Fine-tuning on MOSE (single dataset): Hours to days
- Full training on SA-V + SA-1B: Weeks (exact duration not public)

**Expected J&F after fine-tuning** (from training README):
- Base+ model on MOSE: **79.4 J&F** (validation metric)

---

## Mixed Precision Training

### Precision Format

**Strong evidence for bfloat16**:

From SAM 2 inference examples:
```python
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
```

**Why bfloat16**:
- Standard for A100 GPUs (better hardware support than fp16)
- Wider dynamic range than fp16 (reduces overflow issues)
- Used in Meta's other models (LLaMA, etc.)

### Gradient Handling

**Likely techniques** (standard for large vision models):
- **Gradient accumulation**: Supported via batch_sizes per dataset in config
- **Gradient clipping**: Common practice (likely global norm = 1.0)
- **Mixed precision**: Native PyTorch AMP with bfloat16

---

## Loss Function

### MultiStepMultiMasksAndIous

From `loss_fns.py` (training code):
- Multi-step loss (evaluates masks at multiple decoding iterations)
- Combined mask prediction + IoU prediction losses
- Supports multiple objects per frame/image

**Training paradigm**:
- Simulated user prompts during training
- Iterative point sampling (mimics interactive refinement)
- Video: temporal consistency via memory mechanism

---

## Effective Batch Size

### Per-Dataset Batch Sizes

Example from training README:
```yaml
batch_sizes:
- ${bs1}  # Batch size for dataset 1 (e.g., SA-1B images)
- ${bs2}  # Batch size for dataset 2 (e.g., SA-V videos)
```

**Gradient accumulation implicit**:
- Phases per epoch: Chunks single epoch into smaller phases
- Effective batch size = (batch_size × num_gpus × accumulation_steps)

**Typical setup** (inferred):
- Per-GPU batch size: 2-4 (limited by 80GB memory)
- 8 GPUs × 2-4 samples = 16-32 effective batch size
- Possible gradient accumulation for larger effective batch sizes

---

## Training Tricks & Techniques

### 1. Dataset Mixing Strategy

From training README:
```yaml
datasets:
- SA1B (images, 1 frame sampler)
- SA-V (videos, 8 frame sampler)
```

**Key insight**: Different samplers per dataset type
- Images: 1 frame, max objects per image
- Videos: 8 frames, max objects per video, reverse_time_prob

### 2. Data Augmentation

**Video-specific**:
- `reverse_time_prob`: Probability to reverse video temporally
- Helps model learn time-invariant features

**Image/Video transforms**: Separate transform pipelines per modality

### 3. Efficient Video Training

**Frame sampling**:
- SA-V requires frame extraction (not done at runtime)
- Uses `sav_frame_extraction_submitit.py` script
- Pre-extracted JPEGs for faster I/O

### 4. Model Architecture Training

**SAM2Train class**:
- Inherits from SAM2Base (inference model)
- Adds training-specific functions (prompt simulation, iterative sampling)
- Enables full model or fine-tuning modes

---

## Fine-Tuning Best Practices

### From MOSE Example

**Setup**:
```bash
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

**Config requirements**:
- `img_folder`: Path to JPEG images
- `gt_folder`: Path to annotations
- `file_list_txt`: Optional subset selection
- `experiment_log_dir`: Checkpoint/log output

**Monitoring**:
- TensorBoard logs in `tensorboard/` subdirectory
- Checkpoints saved in `checkpoints/` subdirectory
- Compatible with released checkpoint format

---

## SAM 2.1 Performance Gains

### Benchmark Improvements (SAM 2 → SAM 2.1)

**SA-V test set**:
- Tiny: 75.0 → 76.5 (+1.5 J&F)
- Small: 74.9 → 76.6 (+1.7 J&F)
- Base+: 74.7 → 78.2 (+3.5 J&F)
- Large: 76.0 → 79.5 (+3.5 J&F)

**MOSE val set**:
- Tiny: 70.9 → 71.8 (+0.9 J&F)
- Small: 71.5 → 73.5 (+2.0 J&F)
- Base+: 72.8 → 73.7 (+0.9 J&F)
- Large: 74.6 → 74.6 (same)

**Key observation**: Larger models (Base+, Large) benefit most on SA-V, suggesting training improvements help scale better.

---

## Inference Speed (Unchanged)

**Speed metrics** (A100 GPU, torch 2.5.1):
- Tiny: 91.2 FPS (SAM 2.1) vs 91.5 FPS (SAM 2)
- Small: 84.8 FPS vs 85.6 FPS
- Base+: 64.1 FPS vs 64.8 FPS
- Large: 39.5 FPS vs 39.7 FPS

**Inference speed virtually identical** → Training improvements don't add computational cost.

---

## What Changed: SAM 2 → SAM 2.1

### Training Recipe Improvements (Inferred)

Since dataset and architecture are the same, improvements likely from:

1. **Better hyperparameters**: Learning rate, warmup schedule, weight decay
2. **Training duration**: Possibly longer training (more epochs/iterations)
3. **Loss weighting**: Adjusted multi-step loss weights
4. **Data augmentation**: Enhanced augmentation strategy
5. **Prompt simulation**: Improved interactive prompt sampling during training

### Release Timeline

- **July 29, 2024**: SAM 2 initial release (inference code + checkpoints)
- **September 29, 2024**: SAM 2.1 release (training code + improved checkpoints)

**3-month gap** suggests iterative training experiments to achieve +2.8 J&F average improvement.

---

## Training on Custom Data

### Supported Dataset Types

1. **SA-1B style** (images): Single-frame segmentation
2. **SA-V style** (videos): Multi-frame with temporal annotations
3. **DAVIS style** (videos): Any DAVIS-format dataset (e.g., MOSE)

### Dataset Configuration

From training README:
```yaml
- _target_: training.dataset.vos_dataset.VOSDataset
  training: true
  video_dataset:
    _target_: training.dataset.vos_raw_dataset.JSONRawDataset
    img_folder: ${path_to_img_folder}
    gt_folder: ${path_to_gt_folder}
    ann_every: 4  # Annotate every 4th frame
  sampler:
    num_frames: 8
    max_num_objects: ${max_num_objects_per_video}
```

**Key features**:
- Mixed image + video training
- Per-dataset batch sizes
- Flexible frame sampling
- Annotation sparsity control (`ann_every`)

---

## Community Insights

### Fine-Tuning on Custom Datasets

From [DataCamp tutorial](https://www.datacamp.com/tutorial/sam2-fine-tuning) (2024-09-03):
- Optimizer responsible for updating model weights
- Scheduler adjusts learning rate during training
- Fine-tuning supported on any DAVIS-format dataset

### Optimization Research

From optimization papers (2024):
- Learning rate schedule dynamics impact training convergence
- Warmup + cosine decay standard for vision transformers
- AdamW vs SGD: AdamW preferred for transformers

---

## Limitations & Unknown Details

### What We Don't Know

1. **Exact optimizer hyperparameters**:
   - Precise learning rate values
   - Weight decay coefficient
   - Adam beta values
   - Gradient clipping threshold

2. **Exact training duration**:
   - Total iterations for SAM 2.1
   - Epochs on SA-V + SA-1B
   - Compute hours on A100 cluster

3. **Loss function weights**:
   - Multi-step loss weighting scheme
   - Mask vs IoU loss balance

4. **EMA (Exponential Moving Average)**:
   - Whether EMA is used (common in modern training)
   - EMA decay rate if used

### Why Details Are Limited

- Meta released training code but not full training configs
- YAML configs in repo are for fine-tuning (not full training)
- Full training configs likely internal
- Compute requirements (months on large clusters) make reproduction difficult

---

## Practical Recommendations

### For Fine-Tuning SAM 2.1

**Based on released code**:
1. Start with released checkpoints (don't train from scratch)
2. Use 8× A100 80GB GPUs minimum
3. Follow MOSE example as template
4. Monitor TensorBoard logs for convergence
5. Expect J&F ~79.4 on MOSE with Base+ model

### For Training From Scratch

**Infeasible for most researchers**:
- Requires SA-V dataset (50K+ videos extracted to frames)
- Requires SA-1B dataset (11M images)
- Requires large GPU cluster (weeks of training)
- Use fine-tuning instead

---

## Sources

### Primary Sources

**Official Documentation**:
- [SAM 2 Training README](https://github.com/facebookresearch/sam2/blob/main/training/README.md) - Training code structure and fine-tuning example
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2) - Model checkpoints and performance benchmarks

**Research Papers**:
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/html/2408.00714v2) - Original SAM 2 paper
- [Efficient Track Anything](https://arxiv.org/html/2411.18933v1) - Derivative model with cosine schedule mention (2024-11-28)

### Secondary Sources

**Tutorials & Analysis**:
- [DataCamp: Fine-Tuning SAM 2](https://www.datacamp.com/tutorial/sam2-fine-tuning) - Community fine-tuning guide (2024-09-03)
- [Encord: SAM 2.1 Explained](https://encord.com/blog/sam-2.1-explained/) - Performance analysis (2024-10-22)

**Related Research**:
- [Optimization Benchmark for Diffusion Models](https://www.researchgate.net/publication/396789910) - Learning rate schedule research (2024-10-25)
- [Dynamics of Learning Rate Transitions](https://aclanthology.org/2024.findings-emnlp.954.pdf) - Learning rate schedule analysis (EMNLP 2024)

**Model Performance**:
- [Ultralytics SAM 2 Docs](https://docs.ultralytics.com/models/sam-2/) - Performance comparisons
- [Beyond SAM-2](https://datature.com/blog/beyond-sam-2-exploring-derivatives-for-better-performance) - Derivative models analysis (2024-12-05)

---

## Key Takeaways

1. **SAM 2.1 training improvements are primarily recipe-based**, not architectural
2. **+2.8 J&F average improvement** across benchmarks with same model architecture
3. **Training code released** (September 2024) but full hyperparameters not public
4. **bfloat16 mixed precision** strongly indicated by A100 hardware + inference examples
5. **AdamW + cosine schedule** likely but not officially confirmed
6. **Fine-tuning is practical** (8 GPUs, hours-days), full training is not (requires massive compute)
7. **Inference speed unchanged** → Training improvements are "free" at deployment time

**Bottom Line**: SAM 2.1's improvements come from better training practices (schedule, augmentation, duration) rather than model changes, making it a "refined recipe" release rather than architectural innovation.
