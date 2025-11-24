# SAM 2.1: Training Code Release (Developer Suite)

**Release Date**: September 30, 2024
**Repository**: https://github.com/facebookresearch/sam2/tree/main/training
**Purpose**: Enable fine-tuning and custom training of SAM 2 on user datasets (images, videos, or both)

---

## Overview

As part of the SAM 2.1 Developer Suite announcement, Meta released complete training and fine-tuning code for SAM 2. This enables researchers and developers to:
- Fine-tune SAM 2.1 on custom datasets
- Train from scratch on proprietary data
- Mix image and video datasets for training
- Adapt the model for domain-specific segmentation tasks

From [GitHub README](https://github.com/facebookresearch/sam2):
> "The training (and fine-tuning) code has been released. See training/README.md on how to get started."

---

## Repository Structure

The training code is organized into well-defined components:

```
training/
├── dataset/          # Image and video dataset classes, dataloaders, transforms
├── model/            # SAM2Train class (inherits from SAM2Base)
├── utils/            # Training utilities (loggers, distributed training)
├── scripts/          # SA-V frame extraction scripts
├── loss_fns.py       # MultiStepMultiMasksAndIous loss class
├── optimizer.py      # Optimizer utils with arbitrary schedulers
├── trainer.py        # Main Trainer class (train/eval loop)
├── train.py          # Launch script (single/multi-node)
└── configs/          # Training configuration files
```

### Key Components

**1. SAM2Train Model** (`model/`)
- Inherits from `SAM2Base`
- Adds training-specific functionality
- Supports simulating user prompts (iterative point sampling)
- Configurable via Hydra

**2. Dataset Classes** (`dataset/`)
- **VOSDataset**: Generic video object segmentation dataset
- **SA1BRawDataset**: SA-1B image dataset loader
- **JSONRawDataset**: SA-V video dataset loader (JSON annotations)
- DAVIS-style dataset support (e.g., MOSE)
- Custom transforms for images and videos

**3. Training Infrastructure**
- Multi-GPU support (single node)
- Multi-node training via SLURM
- TensorBoard logging
- Checkpoint management

---

## Getting Started: Fine-Tuning on MOSE

Meta provides a complete example for fine-tuning on the [MOSE dataset](https://henghuiding.github.io/MOSE/).

### Requirements

- **Hardware**: A100 GPUs with 80 GB memory (assumed baseline)
- **Dataset**: MOSE dataset ([download links](https://github.com/henghuiding/MOSE-api?tab=readme-ov-file#download))
- **Python packages**: Install with `pip install -e ".[dev]"`

### Configuration Setup

Edit `configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`:

```yaml
dataset:
    # PATHS to Dataset
    img_folder: null # PATH to MOSE JPEGImages folder
    gt_folder: null # PATH to MOSE Annotations folder
    file_list_txt: null # Optional PATH to filelist for subset
```

### Single-Node Training (8 GPUs)

```bash
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

### Multi-Node Training (SLURM)

```bash
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 1 \
    --num-gpus 8 \
    --num-nodes 2 \
    --partition $PARTITION \
    --qos $QOS \
    --account $ACCOUNT
```

**Note**: Partition, QOS, and account depend on your SLURM configuration.

### Output and Monitoring

**Default log directory**: `sam2_logs/` (configurable via `experiment_log_dir`)

**Structure**:
```
sam2_logs/${config_name}/
├── checkpoints/        # Saved model checkpoints
├── tensorboard/        # TensorBoard logs
└── training.log        # Text logs
```

**Monitoring**:
- Use TensorBoard to track training losses
- Checkpoints saved automatically in `checkpoints/`

**Expected Results** (MOSE fine-tuning):
- Base Plus model: **79.4 J&F** on MOSE validation

---

## Training Configurations

### Available Configs

SAM 2.1 includes pre-configured training setups:

```
configs/sam2.1_training/
├── sam2.1_hiera_tiny_MOSE_finetune.yaml
├── sam2.1_hiera_small_MOSE_finetune.yaml
├── sam2.1_hiera_b+_MOSE_finetune.yaml
└── sam2.1_hiera_large_MOSE_finetune.yaml
```

### Key Hyperparameters

From training configs (typical values):

**Model parameters**:
- Hiera backbone variants (tiny, small, base+, large)
- Image encoder vs memory encoder
- Prompt encoder configuration

**Training parameters**:
- Batch size per GPU
- Learning rate schedules
- Number of frames per video
- Max objects per image/video
- Reverse time probability (data augmentation)

**Loss configuration**:
- Multi-step multi-mask loss
- IoU prediction loss
- Object score loss

---

## Training on Mixed Datasets

SAM 2 supports training on **images and videos simultaneously** (how SAM 2 was originally trained).

### Dataset Configuration Example

```yaml
data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset
    phases_per_epoch: ${phases_per_epoch} # Chunk epochs into phases
    batch_sizes: # List of batch sizes per dataset
    - ${bs1} # Batch size for dataset 1
    - ${bs2} # Batch size for dataset 2
    datasets:
    # Image dataset (SA-1B example)
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist}
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 1
        max_num_objects: ${max_num_objects_per_image}
      transforms: ${image_transforms}

    # Video dataset (SA-V example)
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.JSONRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist}
        ann_every: 4 # Annotation frequency
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 8
        max_num_objects: ${max_num_objects_per_video}
        reverse_time_prob: ${reverse_time_prob}
      transforms: ${video_transforms}
```

### Supported Dataset Types

**1. Image Datasets**:
- SA-1B (Segment Anything 1 Billion)
- Custom image datasets with masks
- Single-frame sampling

**2. Video Datasets**:
- SA-V (Segment Anything Video)
- DAVIS-style datasets (MOSE, YouTube-VOS, etc.)
- Multi-frame sampling with temporal consistency

**3. Mixed Training**:
- Combine multiple image and video datasets
- Different batch sizes per dataset
- Unified training loop

---

## Hardware Requirements

### Minimum Requirements

From README and configurations:

**GPU Memory**:
- **80 GB A100 GPUs** (assumed baseline for training)
- Multi-GPU training recommended (8 GPUs standard)
- Multi-node for large-scale training

**Model Size vs Hardware**:
- **Tiny** (38.9M params): Lower memory requirements
- **Small** (46M params): Moderate requirements
- **Base+** (80.8M params): 80 GB GPUs recommended
- **Large** (224.4M params): 80 GB GPUs required

### Training Time Estimates

Not explicitly documented, but based on typical foundation model fine-tuning:
- MOSE fine-tuning (base+): Hours to days depending on GPUs
- Full training from scratch: Weeks to months (SA-V scale)

---

## Fine-Tuning vs Training from Scratch

### Fine-Tuning (Recommended)

**What**: Start from SAM 2.1 pretrained checkpoints

**When to use**:
- Limited training data
- Domain adaptation
- Faster convergence
- Better generalization

**Example**: MOSE fine-tuning (provided in docs)

### Training from Scratch

**What**: Train SAM 2 architecture from random initialization

**When to use**:
- Very large custom datasets (SA-V scale)
- Completely different domains
- Full control over training process

**Requires**:
- Massive datasets (billions of masks)
- Significant compute (multi-node clusters)
- Weeks of training time

---

## SA-V Dataset Preparation

To train on SA-V, frames must be extracted from videos.

### Frame Extraction

Script provided: `scripts/sav_frame_extraction_submitit.py`

**Purpose**: Extract SA-V videos to JPEG frames for training

**Process**:
1. Download SA-V dataset
2. Run extraction script
3. Configure paths in training config

**Format**: DAVIS-style structure (JPEGImages + Annotations)

---

## Validation and Evaluation

### Validation Split

Meta provides sample validation split: `training/assets/MOSE_sample_val_list.txt`

### Generating Predictions

Use `tools/vos_inference.py` script:
```bash
python tools/vos_inference.py \
    --checkpoint checkpoints/your_model.pt \
    --config configs/your_config.yaml \
    --dataset_path /path/to/validation
```

See [tools/README.md](../tools/README.md) for details.

### Evaluation Metrics

Run `sav_evaluator.py` (detailed in [sav_dataset/README.md](../sav_dataset/README.md)):
```bash
python sav_dataset/sav_evaluator.py \
    --gt_path /path/to/ground_truth \
    --pred_path /path/to/predictions
```

**Metrics**:
- **J&F** (Jaccard and F-measure)
- Expected MOSE result: 79.4 J&F (base+ model)

---

## Using Fine-Tuned Checkpoints

After training, use your checkpoint like SAM 2.1 released models:

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load your fine-tuned model
checkpoint = "sam2_logs/experiment/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Use as normal
with torch.inference_mode():
    predictor.set_image(your_image)
    masks, _, _ = predictor.predict(your_prompts)
```

---

## Developer Suite Components

The SAM 2.1 Developer Suite includes:

1. **Training Code** (this document)
   - Fine-tuning scripts
   - Dataset loaders
   - Multi-GPU support

2. **Web Demo Code** ([demo/README.md](../demo/README.md))
   - Frontend + backend
   - Locally deployable
   - Similar to https://sam2.metademolab.com/demo

3. **Improved Checkpoints** (SAM 2.1)
   - Better performance
   - Released September 29, 2024

---

## Community Resources

From search results:

**Third-Party Guides**:
- [Train/Fine-Tune SAM 2 in 60 Lines of Code](https://medium.com/data-science/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3) (Sagi Eppel, Medium)
- [Fine-Tune SAM-2.1 on Custom Dataset](https://blog.roboflow.com/fine-tune-sam-2-1/) (Roboflow)
- [Google Colab Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/add-fine-tune-sam-2.1/notebooks/fine-tune-sam-2.1.ipynb) (Roboflow)

**Community Implementations**:
- Simplified training scripts
- Custom dataset examples
- Docker containers

---

## Key Takeaways

**What's Included**:
✅ Complete training infrastructure
✅ Multi-GPU and multi-node support
✅ Image + video dataset loaders
✅ MOSE fine-tuning example
✅ Hydra configuration system
✅ TensorBoard monitoring

**What's Required**:
- 80 GB A100 GPUs (recommended)
- PyTorch 2.5.1+
- Custom datasets in DAVIS format
- SLURM for multi-node (optional)

**Best Practices**:
- Start with fine-tuning (faster, better results)
- Use mixed image+video training for robustness
- Monitor with TensorBoard
- Validate on held-out data
- Expected MOSE result: 79.4 J&F (baseline)

---

## Sources

**Official Documentation**:
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2) (accessed 2025-11-21)
- [training/README.md](https://github.com/facebookresearch/sam2/blob/main/training/README.md) (accessed 2025-11-21)
- [SAM 2.1 Release Notes](https://github.com/facebookresearch/sam2/blob/main/RELEASE_NOTES.md) (Sept 30, 2024)

**Web Research**:
- Search: "SAM 2.1 training code release github facebookresearch" (2025-11-21)
- Search: "SAM 2 training developer suite fine-tuning guide" (2025-11-21)

**Community Resources**:
- Medium: Train/Fine-Tune SAM 2 in 60 Lines (Sagi Eppel)
- Roboflow: Fine-Tune SAM-2.1 on Custom Dataset
- Encord: Meta's SAM-2.1 Explained
