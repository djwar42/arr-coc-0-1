# SAM 2.1: Developer Suite Overview

**Release**: September 30, 2024
**Components**: Training code + Web demo code
**Purpose**: Enable developers to build, train, and deploy SAM 2 applications
**Repository**: https://github.com/facebookresearch/sam2

---

## What is the Developer Suite?

The **SAM 2.1 Developer Suite** is Meta's complete set of development tools released alongside the SAM 2.1 model checkpoints. It represents Meta's commitment to making SAM 2 accessible and customizable for the developer community.

From Meta's announcement:
> "09/30/2024 -- SAM 2.1 Developer Suite (new checkpoints, training code, web demo) is released. The training (and fine-tuning) code has been released. The frontend + backend code for the SAM 2 web demo has been released."

**Key Philosophy**: Open-source, community-driven development enabling researchers and developers to adapt SAM 2 to their specific use cases.

---

## Components

### 1. Training Code

**Location**: `training/` directory in GitHub repository

**What It Provides**:
- Complete training pipeline for SAM 2
- Fine-tuning capabilities on custom datasets
- Support for image datasets, video datasets, or both
- Pre-configured examples (e.g., MOSE dataset fine-tuning)
- Multi-GPU and multi-node distributed training support

**Key Files**:
- `train.py` - Main training script
- `trainer.py` - Training loop implementation
- `loss_fns.py` - Loss functions for training
- `optimizer.py` - Optimizer utilities with schedulers
- `dataset/` - Dataset and dataloader classes
- `model/` - SAM2Train class for training/fine-tuning

**Capabilities**:
- Train SAM 2 from scratch on custom data
- Fine-tune existing checkpoints for domain-specific tasks
- Mix image and video datasets in training
- Configure data augmentation techniques
- Simulate user prompts during training (iterative point sampling)

### 2. Web Demo Code

**Location**: `demo/` directory in GitHub repository

**What It Provides**:
- Complete frontend (React TypeScript + Vite)
- Complete backend (Python Flask + Strawberry GraphQL)
- Docker deployment configurations
- Local deployment options (with MPS support on macOS)

**Architecture**:
- **Frontend**: Interactive web interface for segmentation tasks
- **Backend**: Model inference service with GraphQL API
- **Docker Support**: Containerized deployment for easy setup
- **MPS Support**: Metal Performance Shaders for Apple Silicon

**Features**:
- Video upload and processing
- Interactive segmentation with click/box prompts
- Real-time mask generation and tracking
- Multi-object segmentation in videos
- Gallery of example videos

---

## Target Audience

### ML Researchers
**Use Cases**:
- Experiment with segmentation architectures
- Develop new training techniques
- Benchmark on custom datasets
- Publish reproducible research

**Provided Tools**: Full training code, model architecture access, dataset loaders

### Computer Vision Engineers
**Use Cases**:
- Fine-tune SAM 2 for specific domains (medical, autonomous driving, etc.)
- Integrate segmentation into existing pipelines
- Optimize for production deployment
- Build custom annotation tools

**Provided Tools**: Pre-trained checkpoints, fine-tuning scripts, inference APIs

### Application Developers
**Use Cases**:
- Build web applications with segmentation features
- Create video editing tools
- Develop annotation platforms
- Integrate SAM 2 into products

**Provided Tools**: Web demo code (frontend + backend), Docker deployment, API examples

### Domain Specialists
**Use Cases**:
- Medical imaging (MRI, CT scan segmentation)
- Meteorology (weather pattern analysis)
- Robotics (object detection and tracking)
- Autonomous vehicles (obstacle segmentation)

**Provided Tools**: Fine-tuning on domain-specific datasets, pre-configured training examples

---

## Getting Started

### Quick Start: Using Pre-trained Models

**1. Installation**:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

**Requirements**:
- Python ≥ 3.10
- PyTorch ≥ 2.5.1
- TorchVision ≥ 0.20.1

**2. Download Checkpoints**:
```bash
cd checkpoints
./download_ckpts.sh
```

**Available Models**:
- `sam2.1_hiera_tiny.pt` (38.9M parameters)
- `sam2.1_hiera_small.pt` (46M parameters)
- `sam2.1_hiera_base_plus.pt` (80.8M parameters)
- `sam2.1_hiera_large.pt` (224.4M parameters)

**3. Run Inference**:
```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(your_image)
    masks, _, _ = predictor.predict(input_prompts)
```

### Training/Fine-tuning Workflow

**1. Prepare Your Dataset**:
- Image datasets: SA-1B format or custom
- Video datasets: DAVIS-style (JPEG frames + annotations)
- Extract videos to frames using provided scripts

**2. Configure Training**:
```yaml
# configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml
dataset:
  img_folder: /path/to/JPEGImages
  gt_folder: /path/to/Annotations
  file_list_txt: /path/to/train_list.txt  # Optional
```

**3. Launch Training**:

**Single GPU**:
```bash
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

**Multi-GPU (8 GPUs)**:
```bash
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 0 \
  --num-gpus 8
```

**Multi-Node with SLURM**:
```bash
python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
  --use-cluster 1 \
  --num-gpus 8 \
  --num-nodes 2 \
  --partition $PARTITION
```

**4. Monitor Training**:
- Logs saved to: `sam2_logs/${config_name}/`
- TensorBoard: `sam2_logs/${config_name}/tensorboard/`
- Checkpoints: `sam2_logs/${config_name}/checkpoints/`

**5. Use Fine-tuned Model**:
```python
# Load your fine-tuned checkpoint
checkpoint = "sam2_logs/my_experiment/checkpoints/checkpoint.pt"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
```

### Web Demo Deployment

**Quick Start (Docker)**:
```bash
docker compose up --build
```

Access:
- Frontend: http://localhost:7262
- Backend: http://localhost:7263/graphql

**Local Deployment with MPS (Apple Silicon)**:

**1. Setup Environment**:
```bash
conda create --name sam2-demo python=3.10 --yes
conda activate sam2-demo
conda install -c conda-forge ffmpeg
pip install -e '.[interactive-demo]'
```

**2. Download Checkpoints**:
```bash
cd checkpoints && ./download_ckpts.sh
```

**3. Start Backend**:
```bash
cd demo/backend/server/

PYTORCH_ENABLE_MPS_FALLBACK=1 \
APP_ROOT="$(pwd)/../../../" \
API_URL=http://localhost:7263 \
MODEL_SIZE=base_plus \
DATA_PATH="$(pwd)/../../data" \
gunicorn --worker-class gthread app:app \
  --workers 1 --threads 2 \
  --bind 0.0.0.0:7263 --timeout 60
```

**Model Size Options**: "tiny", "small", "base_plus", "large"

**4. Start Frontend**:
```bash
cd demo/frontend
yarn install
yarn dev --port 7262
```

---

## Documentation

### Official Resources

**Primary Documentation**:
- Main README: https://github.com/facebookresearch/sam2/blob/main/README.md
- Training Guide: https://github.com/facebookresearch/sam2/blob/main/training/README.md
- Demo Guide: https://github.com/facebookresearch/sam2/blob/main/demo/README.md
- Installation Guide: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md

**Example Notebooks**:
- Image Prediction: `notebooks/image_predictor_example.ipynb`
- Video Prediction: `notebooks/video_predictor_example.ipynb`
- Automatic Masks: `notebooks/automatic_mask_generator_example.ipynb`
- Available on Google Colab (links in notebooks)

**Configuration Files**:
- Model configs: `sam2/configs/sam2.1/`
- Training configs: `configs/sam2.1_training/`

### Community Resources

**GitHub Repository**:
- 17.8k+ stars
- 700,000+ downloads since SAM 2 launch
- Active issue tracker (404 issues, 53 pull requests as of Nov 2024)
- Community contributions and discussions

**HuggingFace Integration**:
```python
# Load from HuggingFace Hub
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
```

**Research Paper**:
- Title: "SAM 2: Segment Anything in Images and Videos"
- Authors: Nikhila Ravi, Valentin Gabeur, et al. (FAIR, Meta AI)
- arXiv: https://arxiv.org/abs/2408.00714
- Citation: 2,421+ citations

---

## Developer Features

### Training Capabilities

**Dataset Support**:
- SA-1B (image dataset)
- SA-V (video dataset)
- DAVIS-style datasets (MOSE, etc.)
- Custom image datasets
- Custom video datasets
- Mixed image + video training

**Training Features**:
- Multi-step multi-mask loss functions
- Iterative point sampling simulation
- Data augmentation pipelines
- Reverse time probability for videos
- Frame sequence sampling
- Object count limits per image/video

**Distributed Training**:
- Single GPU
- Multi-GPU (data parallel)
- Multi-node (SLURM integration)
- Gradient accumulation support
- Mixed precision training (bfloat16)

### Inference APIs

**Image Prediction**:
```python
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    box=box
)
```

**Video Prediction**:
```python
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
state = predictor.init_state(video_path)

# Add prompts
frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    state, prompts
)

# Propagate through video
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    # Process masks
    pass
```

**Automatic Mask Generation**:
```python
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
mask_generator = SAM2AutomaticMaskGenerator(model)
masks = mask_generator.generate(image)
```

### Web Demo Architecture

**Frontend Stack**:
- React TypeScript
- Vite build system
- Interactive canvas for annotations
- Video player integration
- Real-time mask visualization

**Backend Stack**:
- Python Flask server
- Strawberry GraphQL API
- PyTorch model inference
- Video frame extraction
- Multi-object tracking state management

**API Features**:
- GraphQL queries for model inference
- WebSocket support (optional)
- Session state management
- Video upload and processing
- Mask export (PNG, JSON)

---

## Use Cases & Examples

### Medical Imaging
**Scenario**: Segment tumors in MRI scans

**Approach**:
1. Fine-tune SAM 2.1 on medical imaging dataset (DICOM converted to JPEG)
2. Train with domain-specific augmentations
3. Deploy inference API for radiologists
4. Integrate with PACS systems

**Benefits**: Improved accuracy on small objects, occlusion handling for overlapping tissues

### Autonomous Vehicles
**Scenario**: Segment pedestrians and obstacles in video streams

**Approach**:
1. Fine-tune on autonomous driving datasets (nuScenes, Waymo)
2. Train on longer video sequences (8+ frames)
3. Deploy real-time inference with GPU optimization
4. Track objects across frames with video predictor

**Benefits**: Handles occlusions (partially hidden pedestrians), tracks moving objects

### Video Editing Tools
**Scenario**: Build web-based video object removal tool

**Approach**:
1. Deploy SAM 2.1 web demo as starting point
2. Customize frontend for video editing UI
3. Add mask-based video inpainting backend
4. Export edited videos with removed objects

**Benefits**: Ready-to-deploy frontend/backend code, interactive segmentation

### Meteorology
**Scenario**: Analyze weather patterns in satellite imagery

**Approach**:
1. Fine-tune on satellite image sequences
2. Train with data augmentation for small cloud features
3. Deploy batch inference for historical data analysis
4. Integrate with forecasting models

**Benefits**: Better segmentation of visually similar weather features

---

## Community Impact

### Adoption Metrics
- **700,000+ downloads** since SAM 2 launch (July 2024)
- **17.8k GitHub stars** (November 2024)
- **2.2k forks** on GitHub
- **2,421+ citations** of research paper

### Industry Applications
- **Medical Imaging**: Tumor segmentation, organ delineation
- **Meteorology**: Weather pattern analysis, satellite imagery
- **Autonomous Driving**: Obstacle detection, pedestrian tracking
- **Robotics**: Object manipulation, scene understanding
- **AR/VR**: Spatial tracking, object interaction
- **Video Production**: Object removal, rotoscoping, effects

### Research Contributions
- Open-source training code enables reproducible research
- Community feedback shapes model improvements
- SA-V dataset (50.9K videos, 642.6K masklets) available for research
- Benchmark comparisons on standard datasets (MOSE, LVOS v2)

---

## Sources

**Official Meta Resources**:
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2) - Main repository with code and documentation
- [Meta AI Blog: SAM 2.1 Announcement](https://ai.meta.com/blog/fair-news-segment-anything-2-1/) - Official release announcement (October 18, 2024)
- [SAM 2 Research Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) - arXiv:2408.00714
- [SAM 2 Project Page](https://ai.meta.com/sam2) - Official project page
- [SAM 2 Demo](https://sam2.metademolab.com/) - Live web demo

**Training Documentation**:
- [Training README](https://github.com/facebookresearch/sam2/blob/main/training/README.md) - Complete training guide
- [Demo README](https://github.com/facebookresearch/sam2/blob/main/demo/README.md) - Web demo deployment guide
- [Installation Guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) - Setup instructions

**Community & Analysis**:
- [Encord: SAM 2.1 Explained](https://encord.com/blog/sam-2.1-explained/) - Technical analysis (October 22, 2024)
- [GitHub Issues & Pull Requests](https://github.com/facebookresearch/sam2/issues) - Community discussions and contributions

**Model Checkpoints**:
- [HuggingFace Model Hub](https://huggingface.co/models?search=facebook/sam2) - Pre-trained checkpoints
- [Direct Download Links](https://dl.fbaipublicfiles.com/segment_anything_2/092824/) - SAM 2.1 checkpoints (September 2024)

**Research Context**:
- Original SAM paper (2023): Foundation for SAM 2
- SA-V dataset paper: Training data methodology
- SAM 2 technical report: Architecture and improvements

**Accessed**: November 21, 2024
