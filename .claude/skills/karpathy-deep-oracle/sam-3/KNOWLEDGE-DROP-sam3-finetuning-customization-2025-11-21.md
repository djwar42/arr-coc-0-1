# SAM 3 Fine-Tuning & Customization

## Overview

SAM 3 provides official fine-tuning support for adapting the model to custom domains and specific use cases. Meta has released training code, configuration files, and documentation to enable fine-tuning on custom datasets. This represents a significant advancement over SAM 2, which did not have official fine-tuning functionality.

From [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) (accessed 2025-11-23):
- Official training code in `sam3/train.py`
- Uses Hydra configuration management
- Supports multi-node distributed training and local execution
- Model checkpoints available on HuggingFace

## Installation for Training

From [README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md) (accessed 2025-11-23):

```bash
cd sam3
pip install -e ".[train]"
```

**Prerequisites:**
- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA 12.6 or higher

## Training Script Usage

### Basic Training Commands

```bash
# Train on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml

# Train on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```

### Command Line Arguments

```bash
python sam3/train/train.py \
    -c CONFIG_NAME \
    [--use-cluster 0|1] \
    [--partition PARTITION_NAME] \
    [--account ACCOUNT_NAME] \
    [--qos QOS_NAME] \
    [--num-gpus NUM_GPUS] \
    [--num-nodes NUM_NODES]
```

**Arguments:**
- `-c, --config`: **Required.** Path to configuration file
- `--use-cluster`: 0 for local, 1 for cluster (default: uses config setting)
- `--partition`: SLURM partition name
- `--account`: SLURM account name
- `--qos`: SLURM Quality of Service setting
- `--num-gpus`: Number of GPUs per node
- `--num-nodes`: Number of nodes for distributed training

### Local Training Examples

```bash
# Single GPU training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Multi-GPU training on single node
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4
```

### Cluster Training Examples

```bash
# Basic cluster training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1

# With specific SLURM settings
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1 \
    --partition gpu_partition \
    --account my_account \
    --qos high_priority \
    --num-gpus 8 \
    --num-nodes 2
```

## Configuration System

Training configurations are stored in `sam3/train/configs/` using Hydra YAML format.

### Key Configuration Sections

```yaml
# Paths to datasets and checkpoints
paths:
  bpe_path: /path/to/bpe/file
  dataset_root: /path/to/dataset
  experiment_log_dir: /path/to/logs

# Launcher settings for local/cluster execution
launcher:
  num_nodes: 1
  gpus_per_node: 2
  experiment_log_dir: ${paths.experiment_log_dir}

# Cluster execution settings
submitit:
  use_cluster: True
  timeout_hour: 72
  cpus_per_task: 10
  partition: null
  account: null
```

### Configuration Capabilities

- **Dataset Configuration**: Data paths, transforms, loading parameters
- **Model Configuration**: Architecture settings, checkpoint paths, model parameters
- **Training Configuration**: Batch sizes, learning rates, optimization settings
- **Launcher Configuration**: Distributed training and cluster settings
- **Logging Configuration**: TensorBoard, experiment tracking, output directories

## Supported Datasets

### Roboflow 100-VL

From [Roboflow Fine-Tune Guide](https://blog.roboflow.com/fine-tune-sam3/) (accessed 2025-11-23):

**Dataset structure:**
```
roboflow_vl_100_root:
  13-lkc01/
    train/
    valid/
    test/
  2024-frc/
  actions/
  ...
```

Download from [Roboflow 100-VL](https://github.com/roboflow/rf100-vl/)

### ODinW (Object Detection in the Wild)

**Dataset structure:**
```
odinw_data_root:
  AerialMaritimeDrone/
    large/
      train/
      valid/
      test/
  Aquarium/
  ...
```

Download from [GLIP Repository](https://github.com/microsoft/GLIP)

## Fine-Tuning on Custom Datasets

### Dataset Requirements

From [Roboflow Blog](https://blog.roboflow.com/fine-tune-sam3/) (accessed 2025-11-23):

To fine-tune SAM 3, you need an **instance segmentation dataset** with:
- Images on which to draw segmentations
- Precise segmentation ground truth masks
- Prompts to feed into the model (text or bounding boxes)

### Dataset Preparation Steps

1. **Create/Annotate Dataset**: Use tools like Roboflow with SAM 3-powered Label Assist
2. **Apply Preprocessing**: Resize images to 1008x1008 (expected input size)
3. **Generate Dataset Version**: Create frozen snapshot for training

### Using Roboflow for Fine-Tuning

Roboflow provides hosted SAM 3 fine-tuning with:
- Custom dataset upload
- Automatic hyperparameter setting
- Training monitoring with graphs
- One-click procedure in web app

**Workflow:**
1. Upload instance segmentation dataset
2. Create dataset version with 1008x1008 preprocessing
3. Select SAM 3 for Custom Train
4. Choose base checkpoint
5. Start training job
6. Deploy via Roboflow Workflows

## Domain Adaptation Strategies

### Why Fine-Tune SAM 3?

From [Encord Blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/) (accessed 2025-11-23):

**Key reasons:**
- Improve performance on data not seen in pre-training
- Adapt to specific visual domains (e.g., medical imaging, satellite imagery, industrial inspection)
- Handle edge cases specific to your application
- Reduce computational cost vs training from scratch

### Architecture Considerations

SAM 3 consists of:
- **Image Encoder**: Large, computationally expensive (848M parameters total)
- **Prompt Encoder**: Handles text, points, boxes, masks
- **Mask Decoder**: Lightweight, efficient to fine-tune

**Recommended approach**: Fine-tune only the mask decoder for:
- Faster training
- Lower memory requirements
- Preservation of general segmentation capabilities

### Fine-Tuning Best Practices

1. **Start with pre-trained weights**: Leverage knowledge from 4M+ concepts
2. **Focus on mask decoder**: Freeze image encoder for efficiency
3. **Use domain-specific data**: Target the visual patterns in your domain
4. **Apply appropriate augmentations**: Start without augmentations, add based on performance
5. **Monitor for overfitting**: Validate on held-out test set

## Training Output Structure

```
experiment_log_dir/
├── config.yaml              # Original configuration
├── config_resolved.yaml     # Resolved configuration
├── checkpoints/             # Model checkpoints
├── tensorboard/             # TensorBoard logs
├── logs/                    # Text logs
└── submitit_logs/           # Cluster job logs
```

### Monitoring Training

```bash
tensorboard --logdir /path/to/experiment_log_dir/tensorboard
```

## Job Arrays for Dataset Sweeps

For training multiple models on different datasets:

```yaml
submitit:
  job_array:
    num_tasks: 100
    task_index: 0
```

The `task_index` automatically selects which dataset to use from the complete list of Roboflow supercategories.

## Evaluation After Fine-Tuning

### Running Evaluation

Set `trainer.mode = val` in config:

```bash
# Evaluate on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml

# Evaluate on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml
```

### Reproducing ODinW13 10-shot Results

```bash
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```

Change `odinw_train.train_file` to test different seeds:
- `fewshot_train_shot10_seed300`
- `fewshot_train_shot10_seed30`
- `fewshot_train_shot10_seed3`

Final results are aggregated from all three seeds.

## Custom Concept Training

### Adding New Concepts

SAM 3's architecture supports open-vocabulary prompting with 270K concepts. For custom concepts:

1. **Create annotated dataset** with your custom concepts
2. **Use text prompts** that describe your concepts
3. **Fine-tune with appropriate data volume** (10-100+ images per concept recommended)
4. **Evaluate on held-out examples** to verify generalization

### Concept Hierarchy

SAM 3 can handle:
- **Fine-grained concepts**: Specific object types (e.g., "Siamese cat")
- **Coarse concepts**: General categories (e.g., "cat")
- **Attribute-based concepts**: Objects with specific properties (e.g., "red car")
- **Contextual concepts**: Objects in specific situations (e.g., "player in white jersey")

## Deployment After Fine-Tuning

### Saving Checkpoints

```python
torch.save(model.state_dict(), PATH)
```

### Loading Fine-Tuned Model

```python
model = build_sam3_image_model(checkpoint=PATH)
```

### Deployment Options

From [Roboflow](https://blog.roboflow.com/fine-tune-sam3/) (accessed 2025-11-23):

1. **Cloud API (Serverless)**: Roboflow hosted inference
2. **On-device**: Using Roboflow Inference server
3. **Custom deployment**: Direct PyTorch inference

## Technical Considerations

### Memory Requirements

- Full model: 848M parameters
- Fine-tuning mask decoder only: Significantly reduced memory
- Image encoder embedding: Can be pre-computed and cached

### Training Time Estimates

- Depends on dataset size, GPU count, and epochs
- Roboflow provides time estimates for hosted training
- Local training times vary with hardware

### Handling Training Divergence

From [README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md):
> "A small number of jobs may diverge during training, in which case we just use the last checkpoint's result before it diverges."

## Sources

**Source Documents:**
- SAM 3 GitHub README and training documentation

**Web Research:**
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) - Official Meta repository (accessed 2025-11-23)
- [README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md) - Official training documentation (accessed 2025-11-23)
- [Roboflow Fine-Tune Guide](https://blog.roboflow.com/fine-tune-sam3/) - Hosted fine-tuning walkthrough (accessed 2025-11-23)
- [Encord Fine-Tuning Tutorial](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/) - SAM fine-tuning best practices (accessed 2025-11-23)
- [Meta AI Blog](https://ai.meta.com/blog/segment-anything-model-3/) - Official SAM 3 announcement (accessed 2025-11-23)

**Additional References:**
- [Roboflow 100-VL Dataset](https://github.com/roboflow/rf100-vl/) - Training dataset
- [GLIP ODinW Datasets](https://github.com/microsoft/GLIP) - Object Detection in the Wild
- [SAM 3 HuggingFace](https://huggingface.co/facebook/sam3) - Model checkpoints
