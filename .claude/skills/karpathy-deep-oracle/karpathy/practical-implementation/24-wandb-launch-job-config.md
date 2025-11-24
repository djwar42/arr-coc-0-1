# W&B Launch Job Configuration and Reproducibility

## Overview

W&B Launch job configuration enables reproducible, automated ML training through declarative YAML/JSON specifications. Launch jobs encapsulate complete training environments - code, dependencies, resources, secrets, and execution parameters - ensuring experiments can be reliably reproduced across different compute environments and over time.

**Key capabilities:**
- Declarative job specifications (YAML/JSON)
- Resource allocation (GPU, CPU, memory)
- Environment variable and secret management
- Docker image specification and customization
- Artifact dependencies (datasets, models, checkpoints)
- Configuration templating and inheritance
- Version control integration
- Reproducibility guarantees through containerization

**Common use cases:**
- Reproducing published experiments
- Team collaboration with shared compute resources
- Multi-environment deployment (dev → staging → prod)
- Automated hyperparameter sweeps at scale
- Cost-optimized training on cloud resources

From [W&B Launch Documentation](https://docs.wandb.ai/platform/launch/set-up-launch) (accessed 2025-01-31):
> "Launch abstracts away complexity. Launch also provides click-button reproducibility of models by automatically containerizing jobs."

---

## Job Configuration Structure

### Launch Config YAML Anatomy

Launch job configurations define how training runs execute on target resources. Core structure:

```yaml
# Queue and resource targeting
queue: <queue-name>
entity: <wandb-entity>
project: <wandb-project>

# Code source (git, artifact, or image)
job: <job-name-or-uri>
git:
  repository: https://github.com/user/repo
  branch: main
  commit: abc123

# Resource specifications
resource_args:
  kubernetes:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "2"
    limits:
      memory: "64Gi"
      nvidia.com/gpu: "2"
    node_selector:
      accelerator: "nvidia-tesla-v100"

  sagemaker:
    InstanceType: ml.p3.8xlarge
    VolumeSizeInGB: 100
    MaxRuntimeInSeconds: 86400

# Entry point and command
entry_point: train.py
command: ["python", "train.py", "--epochs", "100"]

# Environment variables
env:
  WANDB_PROJECT: llm-training
  CUDA_VISIBLE_DEVICES: "0,1"
  NCCL_DEBUG: INFO

# Artifact dependencies
input_artifacts:
  - dataset:latest
  - pretrained-model:v1.2
```

**Configuration hierarchy:**
1. **Queue config** - Default settings for all jobs on queue
2. **Job config** - Job-specific overrides
3. **Runtime overrides** - CLI arguments when submitting

From [W&B Launch Queue Configuration](https://docs.wandb.ai/platform/launch/setup-queue-advanced) (accessed 2025-01-31):
> "When an agent receives a job from a queue, it also receives the queue configuration. When the agent submits the job to the target resource, it includes the queue configuration along with any overrides from the job itself."

### Resource Specifications

**Compute resources:**

```yaml
resource_args:
  # Kubernetes
  kubernetes:
    requests:  # Minimum guaranteed resources
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
    limits:    # Maximum allowed resources
      memory: "32Gi"
      nvidia.com/gpu: "1"

    # Node selection
    node_selector:
      node.kubernetes.io/instance-type: p3.8xlarge
      topology.kubernetes.io/zone: us-west-2a

    # Tolerations for tainted nodes
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"

  # AWS SageMaker
  sagemaker:
    RoleArn: arn:aws:iam::123456789:role/SageMakerRole
    ResourceConfig:
      InstanceType: ml.p3.16xlarge
      InstanceCount: 1
      VolumeSizeInGB: 500
    StoppingCondition:
      MaxRuntimeInSeconds: 259200  # 72 hours
```

**GPU allocation:**

```yaml
# Docker
resource_args:
  docker:
    gpus: "all"  # All available GPUs
    # OR specific GPUs
    gpus: "device=0,1"  # GPUs 0 and 1

    # GPU memory limits (NVIDIA)
    runtime: nvidia
    env:
      - NVIDIA_VISIBLE_DEVICES=0,1
      - CUDA_VISIBLE_DEVICES=0,1
```

### Entry Point and Command Configuration

**Script execution:**

```yaml
# Python script entry point
entry_point: train.py
args:
  - --model=gpt2-large
  - --batch-size=32
  - --learning-rate=3e-4

# Full command override
command:
  - python
  - -m
  - torch.distributed.launch
  - --nproc_per_node=8
  - train.py
  - --config=config/llm-training.yaml
```

**Multi-stage training:**

```yaml
# Stage 1: Data preprocessing
job: preprocess-data
entry_point: preprocess.py
output_artifacts:
  - processed-dataset

# Stage 2: Training (depends on Stage 1)
job: train-model
entry_point: train.py
input_artifacts:
  - processed-dataset:latest
```

### Environment Variables and Secrets

**Environment variable configuration:**

```yaml
env:
  # Public configuration
  WANDB_PROJECT: vlm-training
  WANDB_ENTITY: research-team
  MODEL_TYPE: multimodal-transformer

  # Reference to agent environment
  CUDA_HOME:  # Inherits from agent environment

  # Secrets (not directly visible)
  WANDB_API_KEY:  # Set via queue config or agent
  HF_TOKEN:       # HuggingFace API token
  AWS_ACCESS_KEY_ID:
  AWS_SECRET_ACCESS_KEY:
```

From [W&B Launch Docker Setup](https://docs.wandb.ai/platform/launch/setup-launch-docker) (accessed 2025-01-31):
> "Docker automatically passes environment variables, that are not assigned a value, from the launch agent environment. This means that, if the launch agent has an environment variable MY_EXISTING_ENV_VAR, that environment variable is available in the container."

**Secret management best practices:**

```yaml
# Queue config (admins only)
env:
  WANDB_API_KEY: ${WANDB_SERVICE_ACCOUNT_KEY}  # From agent env
  DATABASE_URL: ${DB_CONNECTION_STRING}

# Job config (users)
env:
  MODEL_NAME: llama-2-7b  # Public config only

# Secrets never in job config - always queue/agent level
```

### Git Repository and Code Versioning

**Git-based jobs:**

```yaml
job:
  git:
    repository: https://github.com/myorg/llm-training
    branch: main
    commit: a1b2c3d4  # Pin to specific commit for reproducibility

    # Private repo authentication
    # (credentials set in agent environment)
```

**Code artifact jobs:**

```yaml
job:
  artifact: wandb-artifact://entity/project/code:v1.2.3

# Automatically captures:
# - All Python files
# - requirements.txt / environment.yaml
# - Training scripts
# - Config files
```

**Reproducibility guarantee:**
- Git commit hash locks exact code version
- Docker image captures dependencies
- Artifacts version datasets/models
- Config captures hyperparameters

### Docker Image Specification

**Pre-built images:**

```yaml
# Use existing Docker image
job:
  docker:
    image: nvcr.io/nvidia/pytorch:23.12-py3
    # OR
    image: myregistry.azurecr.io/vlm-training:v1.2
```

**Custom base images for building:**

```yaml
# When Launch builds from git/artifact
builder:
  accelerator:
    base_image: tensorflow/tensorflow:latest-gpu
    # OR
    base_image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

From [W&B Launch Queue Configuration](https://docs.wandb.ai/platform/launch/setup-queue-advanced) (accessed 2025-01-31):
> "You might need to specify an accelerator base image if you use launch to build images that are executed in an accelerator environment. This accelerator base image must satisfy the following requirements:
> - Debian compatibility (the Launch Dockerfile uses apt-get to fetch python)
> - Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
> - Compatibility between the accelerator version you provide and the packages installed in your ML algorithm"

**TensorFlow GPU example:**

```yaml
# Queue config for TensorFlow training
resource_args:
  builder:
    accelerator:
      base_image: "tensorflow/tensorflow:latest-gpu"
  docker:
    gpus: "all"
```

---

## Environment Management and Reproducibility

### Docker Images for Launch Jobs

**Launch image build process:**

1. **Base image selection** - CUDA-compatible base (PyTorch, TensorFlow, custom)
2. **Dependency installation** - From requirements.txt, environment.yaml, or pyproject.toml
3. **Code injection** - Git clone or artifact extraction
4. **W&B integration** - wandb CLI and Python SDK
5. **Entry point configuration** - Training script execution

**Automatic image building:**

```yaml
# Agent config (launch-config.yaml)
builder:
  type: docker  # Build images locally
  # OR
  type: kaniko  # Build in Kubernetes without Docker daemon
  # OR
  type: noop    # Don't build, use pre-built images only
```

**Custom Docker builds:**

```dockerfile
# Custom Dockerfile for Launch
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Python environment
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# W&B integration
RUN pip install wandb

# Training code
WORKDIR /workspace
COPY . .

ENTRYPOINT ["python", "train.py"]
```

### Dependency Management

**Requirements.txt pattern:**

```txt
# Core ML frameworks
torch==2.1.0
transformers==4.35.0
datasets==2.14.0

# W&B integration
wandb>=0.16.0

# Training utilities
accelerate==0.24.0
deepspeed==0.12.0
flash-attn==2.3.0

# Vision dependencies (for VLMs)
timm==0.9.10
Pillow==10.1.0

# Pinned for reproducibility
numpy==1.24.3
scipy==1.11.3
```

**Conda environment.yaml:**

```yaml
name: llm-training
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision=0.16.0
  - cudatoolkit=12.1
  - pip
  - pip:
      - transformers==4.35.0
      - wandb>=0.16.0
      - accelerate==0.24.0
```

### Artifact Dependencies

**Dataset artifacts:**

```yaml
# Job config
input_artifacts:
  - type: dataset
    name: imagenet-1k:latest
    path: /datasets/imagenet

  - type: dataset
    name: coco-captions:v2023.1
    path: /datasets/coco
```

**Model artifacts (warm start):**

```yaml
input_artifacts:
  - type: model
    name: pretrained-clip:v1.0
    path: /models/clip-base

env:
  PRETRAINED_MODEL_PATH: /models/clip-base
```

**Checkpoint resumption:**

```yaml
# Automatic checkpoint loading
input_artifacts:
  - type: model
    name: training-checkpoint:latest  # Latest checkpoint from previous run
    path: /checkpoints/resume

command:
  - python
  - train.py
  - --resume-from=/checkpoints/resume/model.pt
```

### Secret Management

**Queue-level secrets (admin configured):**

```yaml
# Queue config (not visible to job submitters)
env:
  WANDB_API_KEY: ${SERVICE_ACCOUNT_KEY}
  AWS_ACCESS_KEY_ID: ${AWS_KEY_ID}
  AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_KEY}
  DATABASE_PASSWORD: ${DB_PASS}
  HUGGINGFACE_TOKEN: ${HF_TOKEN}
```

**Kubernetes secrets integration:**

```yaml
# Queue config
resource_args:
  kubernetes:
    env_from:
      - secret_ref:
          name: wandb-credentials
      - secret_ref:
          name: cloud-provider-keys
```

**Best practices:**
- Never put secrets in job configs
- Use agent environment variables
- Leverage cloud provider secret managers (AWS Secrets Manager, GCP Secret Manager)
- Rotate credentials regularly
- Use service accounts, not personal API keys

### Environment Reproducibility

**Complete reproducibility checklist:**

```yaml
# 1. Code version
git:
  commit: abc123def456  # Exact commit hash

# 2. Dependencies
# requirements.txt with pinned versions
# torch==2.1.0 (not torch>=2.0)

# 3. Base image
builder:
  accelerator:
    base_image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 4. Random seeds
env:
  PYTHONHASHSEED: "42"
command:
  - python
  - train.py
  - --seed=42
  - --deterministic

# 5. Data version
input_artifacts:
  - dataset:v2023.12.01  # Exact version, not :latest

# 6. Hardware specification
resource_args:
  kubernetes:
    node_selector:
      accelerator: nvidia-tesla-v100  # Specific GPU type
```

**Reproducibility features Launch provides:**
- Git commit tracking
- Docker image versioning
- Artifact lineage
- Config snapshots
- Environment capture
- Dependency locking

---

## Configuration Templates and Reuse

### Template Creation for Common Patterns

**Queue config templates** allow admins to define guardrails and defaults:

```yaml
# Queue config with templates
resource_args:
  sagemaker:
    RoleArn: arn:aws:iam::123456789:role/SageMakerRole
    ResourceConfig:
      InstanceType: "{{instance_type}}"  # Template variable
      InstanceCount: 1
      VolumeSizeInGB: "{{volume_size}}"
    StoppingCondition:
      MaxRuntimeInSeconds: "{{max_runtime}}"
```

From [W&B Launch Queue Configuration](https://docs.wandb.ai/platform/launch/setup-queue-advanced) (accessed 2025-01-31):
> "Administer and manage guardrails on compute consumption with Queue Config Templates. Set defaults, minimums, and maximum values for fields such as memory consumption, GPU, and runtime duration."

**Template field configuration:**

After parsing `{{variable-name}}` templates in queue config:

**String templates** (for instance types):
```yaml
instance_type:
  type: string
  default: ml.p3.2xlarge
  options:
    - ml.p3.2xlarge   # 1 V100 GPU
    - ml.p3.8xlarge   # 4 V100 GPUs
    - ml.p3.16xlarge  # 8 V100 GPUs
```

**Integer templates** (for volume size):
```yaml
volume_size:
  type: integer
  default: 100
  min: 50
  max: 500
```

**Float templates** (for runtime limits):
```yaml
max_runtime:
  type: float
  default: 86400    # 24 hours
  min: 3600         # 1 hour minimum
  max: 259200       # 72 hours maximum
```

### Parameterized Configs

**Job config with parameters:**

```yaml
# Base training config
job: llm-training:v1
hyperparameters:
  model_size: "{{model_size}}"
  batch_size: "{{batch_size}}"
  learning_rate: "{{learning_rate}}"

resource_args:
  kubernetes:
    requests:
      memory: "{{memory}}"
      nvidia.com/gpu: "{{num_gpus}}"
```

**Sweeps integration:**

```yaml
# Sweep config using Launch
method: bayes
metric:
  name: val_loss
  goal: minimize

parameters:
  model_size:
    values: [small, medium, large]
  batch_size:
    values: [16, 32, 64]
  learning_rate:
    min: 1e-5
    max: 1e-3

# Launch queue to execute sweep runs
queue: gpu-training-queue
```

### Config Inheritance and Overrides

**Inheritance hierarchy:**

```yaml
# 1. Queue config (base defaults)
# Defined by admin, applies to all jobs

# 2. Job config (job-level overrides)
resource_args:
  kubernetes:
    requests:
      memory: "64Gi"  # Override queue default

# 3. Runtime overrides (CLI submission)
wandb launch --queue=gpu-queue \
  --override-resource-args='{"kubernetes": {"requests": {"nvidia.com/gpu": "4"}}}'
```

**Override examples:**

```python
# Python API job submission with overrides
import wandb

wandb.launch(
    job="vlm-training:latest",
    queue="gpu-queue",
    config={
        "hyperparameters": {
            "batch_size": 64,  # Override default
            "learning_rate": 5e-4
        }
    },
    resource_args={
        "kubernetes": {
            "requests": {
                "nvidia.com/gpu": "8"  # Override to 8 GPUs
            }
        }
    }
)
```

### Version Control for Configs

**Config versioning strategies:**

```yaml
# configs/base-training.yaml (v1.0)
job: llm-training
resource_args:
  kubernetes:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "2"

# configs/base-training-v1.1.yaml
# Inherit from v1.0, add improvements
job: llm-training
resource_args:
  kubernetes:
    requests:
      memory: "64Gi"  # Increased
      cpu: "16"       # Increased
      nvidia.com/gpu: "4"  # Increased
    node_selector:
      accelerator: nvidia-tesla-a100  # Better GPUs
```

**Git-based config management:**

```bash
# Store configs in git repo
configs/
├── queues/
│   ├── cpu-training.yaml
│   ├── gpu-training.yaml
│   └── multi-node-training.yaml
├── jobs/
│   ├── data-preprocessing.yaml
│   ├── model-training.yaml
│   └── evaluation.yaml
└── README.md

# Submit with specific config version
wandb launch --config configs/jobs/model-training.yaml \
  --queue gpu-training
```

### Team Config Sharing

**Shared team configurations:**

```yaml
# Team standard: LLM training config
# File: team-configs/llm-base.yaml

entity: research-team
project: llm-experiments

resource_args:
  kubernetes:
    requests:
      memory: "128Gi"
      cpu: "32"
      nvidia.com/gpu: "8"
    node_selector:
      node.kubernetes.io/instance-type: p4d.24xlarge
    tolerations:
      - key: "gpu-workload"
        operator: "Exists"

env:
  WANDB_PROJECT: llm-experiments
  NCCL_DEBUG: INFO
  CUDA_LAUNCH_BLOCKING: "0"

input_artifacts:
  - type: dataset
    name: the-pile:latest
    path: /datasets/the-pile
```

**Personal overrides:**

```yaml
# researcher-a/my-experiment.yaml
# Inherits team config, adds experiment-specific settings

base_config: team-configs/llm-base.yaml

# Personal overrides
project: llm-experiments-researcher-a

hyperparameters:
  model_architecture: transformer-xl
  num_layers: 24
  hidden_size: 2048

resource_args:
  kubernetes:
    requests:
      nvidia.com/gpu: "4"  # Use fewer GPUs for this experiment
```

### ARR-COC Training Job Template Example

**ARR-COC VLM training configuration:**

```yaml
# arr-coc-training.yaml
# Adaptive Relevance Realization - Contexts Optical Compression training

entity: research-team
project: arr-coc-vlm
job: arr-coc-training:latest

# Code and dependencies
git:
  repository: https://github.com/myorg/arr-coc-ovis
  branch: main
  commit: def456abc789

# Resource allocation for multi-GPU VLM training
resource_args:
  kubernetes:
    requests:
      memory: "256Gi"      # Large for vision + language
      cpu: "64"
      nvidia.com/gpu: "8"  # Multi-GPU training
    limits:
      memory: "512Gi"
      nvidia.com/gpu: "8"
    node_selector:
      accelerator: nvidia-a100-80gb  # High VRAM for VLMs

# Training configuration
hyperparameters:
  # ARR-COC architecture
  model_architecture: arr-coc-ovis
  vision_encoder: siglip-so400m
  llm_backbone: llama-3.1-8b

  # Relevance realization settings
  min_visual_tokens: 64
  max_visual_tokens: 400
  lod_levels: 4

  # Three ways of knowing weights
  propositional_weight: 0.33
  perspectival_weight: 0.33
  participatory_weight: 0.34

  # Training hyperparameters
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 1000
  max_steps: 100000
  gradient_accumulation_steps: 4

  # Mixed precision
  fp16: true
  bf16: false

# Environment
env:
  WANDB_PROJECT: arr-coc-vlm
  WANDB_RUN_NAME: arr-coc-8gpu-${run_id}

  # Distributed training
  MASTER_ADDR: localhost
  MASTER_PORT: "29500"
  NCCL_DEBUG: WARN

  # ARR-COC specific
  ARR_COC_LOG_RELEVANCE_MAPS: "true"
  ARR_COC_SAVE_FREQUENCY: "1000"

# Dataset artifacts
input_artifacts:
  - type: dataset
    name: vqa-v2-train:latest
    path: /datasets/vqa
  - type: dataset
    name: coco-captions:latest
    path: /datasets/coco
  - type: model
    name: siglip-pretrained:v1.0
    path: /models/siglip-base

# Checkpoint outputs
output_artifacts:
  - type: model
    name: arr-coc-checkpoint-${run_id}
    path: /workspace/checkpoints

# Training script
command:
  - torchrun
  - --nproc_per_node=8
  - --nnodes=1
  - train.py
  - --config=configs/arr_coc_training.yaml
  - --output-dir=/workspace/checkpoints
  - --log-with=wandb
```

**Key ARR-COC configuration features:**
- Multi-GPU training (8x A100 80GB)
- Vision-language model resource requirements
- Relevance realization hyperparameters (3 ways of knowing)
- Dynamic visual token allocation (64-400 tokens)
- Distributed training environment variables
- VQA and captioning dataset artifacts
- Checkpoint artifact outputs with run ID tracking

---

## Sources

**W&B Launch Documentation:**
- [Set up Launch](https://docs.wandb.ai/platform/launch/set-up-launch) (accessed 2025-01-31)
- [Configure launch queue](https://docs.wandb.ai/platform/launch/setup-queue-advanced) (accessed 2025-01-31)
- [Tutorial: Set up W&B Launch with Docker](https://docs.wandb.ai/platform/launch/setup-launch-docker) (accessed 2025-01-31)

**Web Research:**
- Google Search: "W&B Launch job configuration yaml" (2025-01-31)
- Google Search: "wandb launch config environment variables" (2025-01-31)
- Google Search: "W&B Launch docker image configuration" (2025-01-31)
- Google Search: "wandb launch reproducibility" (2025-01-31)

**Additional References:**
- [W&B Launch GitHub Repository](https://github.com/wandb/launch-jobs) - Job examples and templates
- [W&B Launch FAQ](https://docs.wandb.ai/platform/launch/launch-faq) - Common configuration questions
- [wandb launch CLI reference](https://docs.wandb.ai/models/ref/cli/wandb-launch) - Command-line interface
