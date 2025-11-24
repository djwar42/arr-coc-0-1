# HuggingFace Hub: Models, Datasets, and Spaces

## Overview

The HuggingFace Hub is a collaborative platform hosting over 2M models, 500k datasets, and 1M Spaces (interactive demos). It serves as the central model registry, dataset repository, and application deployment platform for the ML community. The Hub uses Git repositories optimized for ML artifacts with a custom storage backend (Xet) that handles large binary files up to terabyte scale.

**Core Value Propositions**:
- Version-controlled ML artifacts (models, datasets, demos)
- Collaborative development with pull requests and discussions
- Model cards and dataset cards for documentation
- Integrated inference API for hosted models
- Private repositories for teams and organizations
- Git LFS optimization for large file handling

From [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/en/index) (accessed 2025-11-15):
> "The Hugging Face Hub is a platform with over 2M models, 500k datasets, and 1M demos in which people can easily collaborate in their ML workflows."

## Section 1: Hub Repository Structure

### Repository Types

The Hub hosts three types of Git repositories:

**1. Model Repositories**
- Store model weights (safetensors, PyTorch, TensorFlow formats)
- Configuration files (config.json)
- Tokenizer files
- README.md as model card
- Example: `bert-base-uncased`, `gpt2`

**2. Dataset Repositories**
- Data files (Parquet, CSV, JSON, images, audio)
- Dataset scripts (for complex loading)
- README.md as dataset card
- Metadata (dataset_infos.json)
- Example: `squad`, `imagenet-1k`

**3. Space Repositories**
- Application code (Gradio, Streamlit, Docker)
- README.md configuration
- Dependencies (requirements.txt)
- Example: `stabilityai/stable-diffusion`

From [Repositories Documentation](https://huggingface.co/docs/hub/en/repositories) (accessed 2025-11-15):
> "Models, Spaces, and Datasets are hosted on the Hugging Face Hub as Git repositories, which means that version control and collaboration are core elements of the Hub."

### Git-Based Version Control

All repositories use Git for version control with Hub-specific optimizations:

**Standard Git Operations**:
```bash
# Clone repository
git clone https://huggingface.co/username/repo-name

# Add files
git add model.safetensors

# Commit changes
git commit -m "Update model weights"

# Push to Hub
git push
```

**Hub-Specific Features**:
- Git LFS automatic for files >10MB
- Branch-based versioning (main, v1.0, experimental)
- Tags for releases
- Commit history viewable in web UI
- Diff visualization for supported formats

**Xet Storage Backend**: The Hub uses a custom storage system optimized for ML artifacts, enabling chunk-level deduplication and faster transfers for large files.

## Section 2: Model Cards (README.md)

Model cards provide essential documentation for models, displayed as README.md in repositories.

### Required Metadata (YAML Front Matter)

```yaml
---
language:
- en
- fr
license: apache-2.0
tags:
- text-generation
- transformers
datasets:
- squad
metrics:
- accuracy
- f1
library_name: transformers
pipeline_tag: text-classification
---
```

From [Model Cards Documentation](https://huggingface.co/docs/hub/en/model-cards) (accessed 2025-11-15):
> "Model cards are files that accompany the models and provide handy information. Under the hood, model cards are simple Markdown files with additional metadata."

### Best Practice Structure

From [Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook) (accessed 2025-11-15):

**1. Model Details**
- Architecture description
- Training procedure
- Intended use cases
- Out-of-scope uses

**2. Training Data**
- Dataset sources
- Preprocessing steps
- Data splits

**3. Evaluation Results**
- Metrics on test sets
- Comparison to baselines
- Limitations

**4. Ethical Considerations**
- Potential biases
- Intended applications
- Misuse potential

**5. Citation Information**
```bibtex
@misc{model-name,
  author = {Author Name},
  title = {Model Name},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/username/model}
}
```

### Auto-Generated Model Cards

Libraries like `transformers` and `setfit` can auto-generate model cards:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model-name")
tokenizer = AutoTokenizer.from_pretrained("model-name")

# Push with auto-generated card
model.push_to_hub("username/model-name",
                   use_auth_token=True,
                   commit_message="Initial model upload")
```

From [Model Card Generator](https://huggingface.co/blog/mitalipo/model-card-generator-interface) (accessed 2025-11-15):
> "The Model Card Generator Interface enables users to effortlessly create interactive HTML reports or static Markdown reports that showcase detailed insights."

## Section 3: Repository Management

### Creating Repositories

**Via Web UI**:
1. Navigate to huggingface.co
2. Click "New Model" / "New Dataset" / "New Space"
3. Fill in repository name, license, visibility
4. Initialize with README

**Via Python API**:
```python
from huggingface_hub import create_repo

# Create model repository
create_repo(
    repo_id="username/model-name",
    repo_type="model",  # or "dataset", "space"
    private=False
)
```

**Via CLI**:
```bash
huggingface-cli repo create username/model-name --type model
```

From [Repository Management Documentation](https://huggingface.co/docs/huggingface_hub/en/guides/repository) (accessed 2025-11-15):
> "This guide will show you how to interact with the repositories on the Hub, especially: Create and delete a repository. Manage branches and tags. Rename your repository."

### Cloning and Pushing

**Clone with Git**:
```bash
git clone https://huggingface.co/username/repo-name
cd repo-name

# Configure Git LFS for large files
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
```

**Push Files**:
```bash
# Add model files
git add model.safetensors config.json tokenizer.json

# Commit
git commit -m "Add model weights and config"

# Push (requires HF token)
git push
```

**Authentication**: Use HuggingFace access token as Git password:
```bash
# Store token
huggingface-cli login

# Or set environment variable
export HUGGING_FACE_HUB_TOKEN=hf_...
```

### Branching and Tagging

**Create Version Branches**:
```bash
git checkout -b v1.0
git push origin v1.0

# Create tag
git tag v1.0.0
git push origin v1.0.0
```

**Use Case**: Maintain stable releases while developing experimental features.

## Section 4: Private Repositories and Team Collaboration

### Private Repository Options

From [Enterprise Hub Documentation](https://huggingface.co/docs/hub/en/enterprise-hub) and [Private Hub Blog](https://huggingface.co/blog/introducing-private-hub) (accessed 2025-11-15):

**Individual PRO Plan** ($9/month):
- Unlimited private repos
- Personal use
- Standard compute resources

**Team Plan** ($20/user/month):
- Shared private repos
- Team collaboration
- Role-based access control
- Resource groups

**Enterprise Hub Plan** (Custom pricing):
- SSO integration
- Audit logs
- Dedicated compute
- Private Hub instance
- Advanced security features

### Access Control

**Repository-Level Permissions**:
- **Admin**: Full control (settings, delete, manage access)
- **Write**: Push commits, create branches
- **Read**: Clone, view files, download

**Organization Roles**:
```python
from huggingface_hub import add_space_collaborator

# Add team member to private repo
add_space_collaborator(
    repo_id="org/private-model",
    user="username",
    permission="write"
)
```

From [Organizations Documentation](https://huggingface.co/docs/hub/en/organizations) (accessed 2025-11-15):
> "The Hub also allows admins to set user roles to control access to repositories and manage their organization's payment method and billing info."

### Team Collaboration Features

**Pull Requests**:
- Propose changes to repositories
- Code review and discussion
- Approval workflows
- Merge strategies (squash, rebase, merge commit)

**Discussions**:
- Community Q&A on repo page
- Issue tracking
- Feature requests
- Threaded conversations

**Resource Groups (Enterprise)**:
From [Resource Groups Documentation](https://huggingface.co/docs/hub/en/enterprise-hub-resource-groups) (accessed 2025-11-15):
> "Keep private repositories visible only to authorized group members; Enable multiple teams to work independently within the same organization."

## Section 5: Hub API (huggingface_hub Python Library)

### Installation and Authentication

```bash
pip install huggingface_hub
```

**Login**:
```python
from huggingface_hub import login

# Interactive login (stores token)
login()

# Or programmatic
login(token="hf_...")
```

From [Quickstart Documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start) (accessed 2025-11-15):
> "The Hugging Face Hub is the go-to place for sharing machine learning models, demos, datasets, and metrics."

### Downloading Files

**Download Model Files**:
```python
from huggingface_hub import hf_hub_download

# Download single file
model_path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="pytorch_model.bin"
)

# Download with specific revision
config_path = hf_hub_download(
    repo_id="gpt2",
    filename="config.json",
    revision="v1.0"  # branch or tag
)
```

**Download Entire Repository**:
```python
from huggingface_hub import snapshot_download

# Download all files
repo_path = snapshot_download(
    repo_id="username/model-name",
    cache_dir="./models"
)
```

### Uploading Files

**Upload Single Files**:
```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="username/model-name",
    repo_type="model"
)
```

**Upload Folder**:
```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./model_dir",
    repo_id="username/model-name",
    repo_type="model",
    commit_message="Upload complete model"
)
```

From [Ultimate Guide to huggingface_hub](https://deepnote.com/blog/ultimate-guide-to-huggingfacehub-library-in-python) (accessed 2025-11-15):
> "It provides programmatic access to Hugging Face Hub: downloading models/datasets, uploading and versioning files, searching and listing repositories."

### Repository Operations

**List Files in Repository**:
```python
from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="username/model-name",
    repo_type="model"
)
print(files)  # ['config.json', 'model.safetensors', 'README.md']
```

**Search Hub**:
```python
from huggingface_hub import HfApi

api = HfApi()

# Search models
models = api.list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=10
)

# Search datasets
datasets = api.list_datasets(
    filter="sentiment-analysis",
    limit=5
)
```

**Create Repository**:
```python
from huggingface_hub import create_repo, HfApi

# Create new repo
repo_url = create_repo(
    repo_id="username/new-model",
    private=True,
    exist_ok=False
)

# Delete repository
api = HfApi()
api.delete_repo(repo_id="username/old-model", repo_type="model")
```

## Section 6: Versioning and Tags

### Branch-Based Versioning

**Main Branch**: Default development branch, points to latest version

**Release Branches**:
```bash
# Create stable release
git checkout -b v1.0
git push origin v1.0

# Continue development on main
git checkout main
```

**Access Specific Versions**:
```python
from transformers import AutoModel

# Load from specific branch
model = AutoModel.from_pretrained(
    "username/model-name",
    revision="v1.0"
)

# Load from main (latest)
model = AutoModel.from_pretrained(
    "username/model-name",
    revision="main"
)
```

### Git Tags for Releases

**Create Tags**:
```bash
# Annotated tag
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0

# Lightweight tag
git tag v1.0.1
git push origin v1.0.1
```

**Semantic Versioning**:
- `v1.0.0`: Major release (breaking changes)
- `v1.1.0`: Minor release (new features, backward compatible)
- `v1.1.1`: Patch release (bug fixes)

### Commit Hashes

Every commit has unique SHA hash:
```python
# Download from specific commit
from huggingface_hub import hf_hub_download

file = hf_hub_download(
    repo_id="username/model",
    filename="model.safetensors",
    revision="abc123def456"  # commit hash
)
```

## Section 7: Hub Search and Discovery

### Filtering Models

**By Task**:
- text-classification
- text-generation
- image-classification
- object-detection
- translation
- summarization

**By Framework**:
- PyTorch
- TensorFlow
- JAX
- Safetensors

**By License**:
- apache-2.0
- mit
- gpl-3.0
- cc-by-4.0

**Web UI Filters**: Use sidebar on huggingface.co/models

**API Search**:
```python
from huggingface_hub import HfApi

api = HfApi()

# Search with multiple filters
models = api.list_models(
    task="text-generation",
    library="transformers",
    language="en",
    sort="downloads",
    direction=-1
)

for model in models:
    print(f"{model.modelId}: {model.downloads} downloads")
```

### Trending and Popular Models

**Sort Options**:
- **Downloads**: Most downloaded models
- **Likes**: Community favorites
- **Recently Updated**: Latest changes
- **Created**: Newest models

**Featured Models**: Curated by HuggingFace team for quality and usefulness

### Model Tags and Metadata

Models can have custom tags for discovery:
```yaml
---
tags:
- bert
- sentiment-analysis
- finance
- custom-tag
---
```

Search by tags:
```python
models = api.list_models(filter="sentiment-analysis")
```

## Section 8: arr-coc-0-1 Hub Deployment

### Project Context

The arr-coc-0-1 project (Adaptive Relevance Realization - Contexts Optical Compression, version 0.1) is a vision-language model implementing Vervaekean relevance realization with variable-LOD visual compression.

**Repository Structure**:
```
arr-coc-0-1/  (GitHub + HuggingFace)
├── .git/                    # Separate git repo
├── arr_coc/                 # Python package
│   ├── texture.py          # 13-channel texture array
│   ├── knowing.py          # 3 ways of knowing scorers
│   ├── balancing.py        # Opponent processing
│   ├── attending.py        # Relevance-to-budget mapping
│   └── realizing.py        # Pipeline orchestrator
├── training/
│   ├── cli.py              # Launch training on Vertex AI
│   ├── tui.py              # Textual TUI for monitoring
│   └── config/
├── app.py                  # Gradio demo
├── README.md               # Model card
├── requirements.txt
└── pyproject.toml
```

From project documentation (arr-coc-0-1/CLAUDE.md):
> "The arr-coc-0-1 MVP implementation lives in: RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/"

### Dual Hosting Strategy

**1. GitHub Repository** (Code + Version Control)
```bash
# Primary development repo
git remote -v
# origin  https://github.com/djwar42/arr-coc-0-1
```

**Purpose**:
- Version control for code
- Issue tracking
- Pull requests
- CI/CD pipelines

**2. HuggingFace Space** (Model + Demo)
```bash
# HF Space for demo hosting
git remote -v
# hf  https://huggingface.co/spaces/NorthHead/arr-coc-0-1
```

**Purpose**:
- Interactive Gradio demo
- Model weights hosting
- Inference API
- Community discovery

### Model Card for arr-coc-0-1

**README.md Structure**:
```markdown
---
title: ARR-COC Vision-Language Model
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
- vision-language
- relevance-realization
- variable-lod
- vervaeke
---

# ARR-COC: Adaptive Relevance Realization

Query-aware visual compression using Vervaekean cognitive framework.

## Model Description

- **Architecture**: 13-channel texture → 3 ways of knowing → opponent processing → variable LOD (64-400 tokens)
- **Training**: Vertex AI custom jobs, W&B tracking
- **Dataset**: VQA, image-text pairs
- **Use Cases**: Efficient vision-language inference

## Quick Start

```python
from arr_coc import ARRCOCModel

model = ARRCOCModel.from_pretrained("NorthHead/arr-coc-0-1")
result = model(image, query="What is in this image?")
```

## Training

See `training/cli.py` for Vertex AI launch scripts.

## Citation

```bibtex
@misc{arr-coc-0-1,
  title={ARR-COC: Adaptive Relevance Realization for Vision},
  author={North Head},
  year={2025},
  url={https://huggingface.co/NorthHead/arr-coc-0-1}
}
```
```

### Deployment Workflow

**1. Model Training** (Vertex AI):
```bash
cd arr-coc-0-1
python training/cli.py launch --config training/config/default.yaml
```

**2. Push to GitHub**:
```bash
git add arr_coc/ training/ README.md
git commit -m "Update model architecture"
git push origin main
```

**3. Sync to HuggingFace**:
```bash
# Push to HF Space (triggers rebuild)
git push hf main
```

**4. Gradio App Auto-Deploys**: HF Space automatically rebuilds and redeploys `app.py`

### Private Model Development

During development, use private HF repo:

```python
from huggingface_hub import create_repo, upload_folder

# Create private model repo
create_repo(
    repo_id="NorthHead/arr-coc-0-1",
    repo_type="space",
    private=True,  # Keep private during dev
    space_sdk="gradio"
)

# Upload development version
upload_folder(
    folder_path="./arr-coc-0-1",
    repo_id="NorthHead/arr-coc-0-1",
    repo_type="space"
)
```

**Make Public on Release**:
```python
from huggingface_hub import update_repo_visibility

update_repo_visibility(
    repo_id="NorthHead/arr-coc-0-1",
    private=False,
    repo_type="space"
)
```

### Model Weights Hosting

**Upload Checkpoints to HF Hub**:
```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model checkpoint
api.upload_file(
    path_or_fileobj="./checkpoints/arr-coc-epoch-10.safetensors",
    path_in_repo="model.safetensors",
    repo_id="NorthHead/arr-coc-0-1",
    repo_type="model"
)

# Upload config
api.upload_file(
    path_or_fileobj="./arr_coc/config.json",
    path_in_repo="config.json",
    repo_id="NorthHead/arr-coc-0-1",
    repo_type="model"
)
```

**Load from Hub in Production**:
```python
from arr_coc import ARRCOCModel

# Auto-downloads from HF Hub
model = ARRCOCModel.from_pretrained(
    "NorthHead/arr-coc-0-1",
    cache_dir="./models"
)
```

### Integration with W&B Launch

The arr-coc-0-1 project uses W&B Launch for training orchestration on Vertex AI, with model artifacts synced to HF Hub:

**Training Flow**:
1. Launch job via W&B Launch
2. Train on Vertex AI GPUs
3. Log metrics to W&B
4. Save checkpoint to GCS
5. Auto-upload best checkpoint to HF Hub

**Configuration** (from training/config/wandb.yaml):
```yaml
wandb:
  project: arr-coc-training
  entity: north-head
  sync_to_hf: true
  hf_repo: NorthHead/arr-coc-0-1
```

This enables seamless model versioning across training (W&B), code (GitHub), and deployment (HF Hub).

## Sources

**HuggingFace Documentation**:
- [Hub Overview](https://huggingface.co/docs/hub/en/index) - Platform introduction
- [Repositories](https://huggingface.co/docs/hub/en/repositories) - Git-based repo structure
- [Repository Management](https://huggingface.co/docs/huggingface_hub/en/guides/repository) - Create, delete, manage repos
- [Model Cards](https://huggingface.co/docs/hub/en/model-cards) - Model documentation
- [Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook) - Best practices
- [Enterprise Hub](https://huggingface.co/docs/hub/en/enterprise-hub) - Team plans and features
- [Organizations](https://huggingface.co/docs/hub/en/organizations) - Team collaboration
- [Resource Groups](https://huggingface.co/docs/hub/en/enterprise-hub-resource-groups) - Access control
- [Quickstart](https://huggingface.co/docs/huggingface_hub/en/quick-start) - Python API guide

**Web Research** (accessed 2025-11-15):
- [Ultimate Guide to huggingface_hub Library](https://deepnote.com/blog/ultimate-guide-to-huggingfacehub-library-in-python) - Programmatic access patterns
- [Introducing Private Hub](https://huggingface.co/blog/introducing-private-hub) - Team collaboration features
- [Model Card Generator](https://huggingface.co/blog/mitalipo/model-card-generator-interface) - Auto-generation tools

**Project Documentation**:
- arr-coc-0-1/CLAUDE.md - Project-specific Hub deployment patterns
- arr-coc-0-1/README.md - Model card example
- RESEARCH/PlatonicDialogues/46-mvp-be-doing/ - Platonic Coding nested repo pattern

**Existing Knowledge**:
- huggingface-hub/ skill - Core Hub documentation (models, datasets, Spaces)
- mlops-production/00-monitoring-cicd-cost-optimization.md - Model registry patterns (referenced in plan)
