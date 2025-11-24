# SAM 3 Source Code

**Repository**: https://github.com/facebookresearch/sam3
**Release**: November 2025 (Meta AI)
**Size**: 69M (155 Python files)
**Cloned**: November 21, 2025

---

## Overview

Segment Anything Model 3 (SAM 3) - Multimodal segmentation with **text + visual prompts**. Combines detector-tracker architecture with 4M open-vocabulary concepts.

### Key Features

- **Text prompts**: Segment objects by natural language descriptions
- **Visual prompts**: Click, box, mask (like SAM 1/2)
- **Detector-tracker**: Decoupled architecture for robustness
- **4M concepts**: Open-vocabulary via data engine
- **SA-Co benchmark**: Gold, Silver, VEval evaluation sets

---

## Directory Structure

```
sam3/
├── sam3/                    # Core library (155 Python files)
│   ├── __init__.py
│   ├── agent/               # SAM 3 Agent system (26 files, ~200KB)
│   │   ├── __init__.py
│   │   ├── agent_core.py    # Main orchestration (26KB)
│   │   ├── system_prompt.py # Default prompt (66KB)
│   │   ├── system_prompt_iterative.py  # Iterative mode (5.8KB)
│   │   ├── helpers/         # Agent utilities (13 files)
│   │   │   ├── visualizer.py         # Visualization (63KB)
│   │   │   ├── masks.py              # Mask utilities
│   │   │   ├── boxes.py              # Box utilities
│   │   │   ├── rotated_boxes.py      # Rotated boxes
│   │   │   ├── clip_image_encoder.py # CLIP encoder
│   │   │   ├── sam3_to_video_result.py
│   │   │   └── ...
│   │   └── tools/           # LLM tools (7 files)
│   │       ├── detect_boxes.py
│   │       ├── track_objects.py
│   │       ├── segment_objects.py
│   │       └── ...
│   ├── eval/                # Evaluation tools (37 files, ~400KB)
│   │   ├── saco_evaluation.py  # SA-Co benchmark
│   │   ├── cgF1.py             # Compositional F1 metric
│   │   ├── hota/               # HOTA tracking metric (10 files)
│   │   ├── teta/               # TETA tracking metric (8 files)
│   │   ├── coco_eval/          # COCO evaluation (12 files)
│   │   ├── youtube_vis_eval/   # YouTube-VIS (7 files)
│   │   └── ...
│   ├── model/               # Core detector-tracker (30+ files)
│   │   ├── detector/        # Detection module
│   │   │   ├── detr.py      # DETR-based detector
│   │   │   ├── backbone.py  # ResNet backbone
│   │   │   ├── transformer.py
│   │   │   └── ...
│   │   ├── tracker/         # Tracking module (SAM 2-style)
│   │   │   ├── sam2_tracker.py
│   │   │   ├── memory_attention.py
│   │   │   └── ...
│   │   ├── text_encoder/    # Text embedding
│   │   │   ├── clip_text.py
│   │   │   └── ...
│   │   └── fusion/          # Multimodal fusion
│   │       └── ...
│   ├── data_engine/         # 4M concept annotation
│   │   ├── annotation_pipeline.py
│   │   ├── concept_bank.py  # 4M concepts
│   │   └── ...
│   └── utils/               # Shared utilities
│       ├── transforms.py
│       ├── visualization.py
│       └── ...
├── examples/                # Jupyter notebooks (11)
│   ├── sam3_text_prompts.ipynb
│   ├── sam3_interactive.ipynb
│   ├── sam3_agent_demo.ipynb
│   ├── sam3_video_tracking.ipynb
│   └── ...
├── scripts/                 # CLI tools
│   ├── train_detector.py
│   ├── train_tracker.py
│   ├── inference.py
│   └── evaluate.py
├── assets/                  # Checkpoints, configs, images
├── README.md                # Main documentation
├── README_TRAIN.md          # Training guide
└── LICENSE                  # Apache 2.0
```

---

## Core Architecture Files

### Main Entry Points

**`agent/agent_core.py`** - SAM 3 Agent System
- LLM-powered agent for complex segmentation tasks
- Natural language understanding
- Multi-step reasoning (detect → track → refine)
- Integration with GPT-4, Claude, etc.

**`model/detector/detr.py`** - Object Detector
- DETR-based detection head
- Open-vocabulary via text encoder
- Decoupled from tracker (robust to missed detections)

**`model/tracker/sam2_tracker.py`** - Object Tracker
- SAM 2-style tracking with memory attention
- Independent per-object tracking
- Handles occlusions and re-identifications

**`model/text_encoder/clip_text.py`** - Text Encoder
- CLIP text embeddings for open-vocabulary
- 4M concepts from data engine
- Zero-shot generalization

### Architecture Components

**Detector-Tracker Decoupling**:
- **Detector**: Finds objects in key frames (text or visual prompts)
- **Tracker**: Propagates masks through video (SAM 2 memory)
- **Advantage**: Robust to detector failures, better temporal consistency

**Presence Token**:
- Fine-grained object discrimination
- Resolves ambiguity when multiple objects match text prompt
- Example: "red car" when multiple red cars present

**Data Engine**:
- Automated annotation pipeline
- 4M open-vocabulary concepts
- SA-Co dataset (composition of objects)

---

## Model Variants

Download from: https://github.com/facebookresearch/sam3#model-checkpoints

| Model | Detector Backbone | Tracker | Concepts | Checkpoint |
|-------|-------------------|---------|----------|------------|
| SAM 3-Base | ResNet-50 | SAM 2-Tiny | 4M | `sam3_base.pth` |
| SAM 3-Large | ResNet-101 | SAM 2-Large | 4M | `sam3_large.pth` |

---

## Usage Patterns

### Pattern 1: Text Prompts

```python
from sam3 import SAM3Agent

# Load agent
agent = SAM3Agent(checkpoint="sam3_large.pth")

# Segment objects by text description
results = agent.segment(
    image=image,
    text_prompt="all red cars"
)

# Returns: masks, boxes, scores
```

### Pattern 2: Interactive Segmentation (SAM 1/2-compatible)

```python
from sam3.model import SAM3Predictor

predictor = SAM3Predictor(checkpoint="sam3_large.pth")
predictor.set_image(image)

# Click prompts (like SAM 1/2)
masks, scores = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1]
)
```

### Pattern 3: Video Tracking with Text

```python
from sam3 import SAM3VideoPredictor

predictor = SAM3VideoPredictor(checkpoint="sam3_large.pth")

# Initialize with text prompt on first frame
predictor.init_video(video_path)
predictor.add_text_object(
    frame_idx=0,
    text="person wearing blue shirt"
)

# Track through video
for frame_idx, masks in predictor.propagate():
    # Process masks
    ...
```

### Pattern 4: SAM 3 Agent (LLM-powered)

```python
from sam3.agent import SAM3Agent

agent = SAM3Agent(
    checkpoint="sam3_large.pth",
    llm="gpt-4"  # or "claude-3", etc.
)

# Natural language task
result = agent.execute(
    image=image,
    instruction="Find all animals in the image and count them"
)

# Agent uses tools: detect_boxes, track_objects, segment_objects
# Returns: Natural language response + visualizations
```

---

## SA-Co Benchmark

**Compositional segmentation evaluation**:

**SA-Co Gold** (high-quality, ~1K images):
- Hand-annotated compositions
- Challenging object combinations
- Metric: cgF1 (compositional F1)

**SA-Co Silver** (~10K images):
- Model-assisted annotations
- Broader coverage
- Metric: cgF1

**SA-Co VEval** (video evaluation):
- Compositional tracking
- Temporal consistency
- Metrics: cgF1, HOTA, TETA

**Download**: https://github.com/facebookresearch/sam3#saco-dataset

---

## Data Engine

**4M concept annotation pipeline**:
1. **Concept Bank**: 4M open-vocabulary concepts
2. **Auto-labeling**: Text-to-image matching
3. **Quality Control**: Human verification
4. **Refinement**: Iterative improvement

**Output**: SA-Co dataset with compositional annotations

---

## Training

**Detector training**:
```bash
python scripts/train_detector.py \
  --config configs/sam3_detector.yaml \
  --gpus 8
```

**Tracker fine-tuning**:
```bash
python scripts/train_tracker.py \
  --config configs/sam3_tracker.yaml \
  --gpus 8
```

See: `README_TRAIN.md` for full details

---

## Installation

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

**Requirements**:
- Python ≥ 3.10
- PyTorch ≥ 2.1
- transformers ≥ 4.30 (for text encoder)
- OpenAI/Anthropic API key (for agent)

---

## Citation

```bibtex
@article{sam3_2025,
  title={SAM 3: Segment Anything with Text and Visual Prompts},
  author={Meta AI},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Cross-References

- **SAM 1**: `../sam1/` - Image segmentation
- **SAM 2**: `../sam2/` - Video segmentation
- **Research**: `.claude/skills/karpathy-deep-oracle/sam-3/` - SAM 3 concepts (when created)

---

**Last Updated**: November 21, 2025
**Source**: GitHub clone (--depth 1, main branch)
**Files**: 155 Python files
**Key Innovation**: Text + visual prompts via detector-tracker + 4M concepts
