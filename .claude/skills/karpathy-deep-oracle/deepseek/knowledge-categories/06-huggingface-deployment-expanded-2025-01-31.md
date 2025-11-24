# HuggingFace Deployment: Complete 2025 Guide (Spaces, Models, Datasets)

**Source**: Bright Data research (2025-01-31) + HuggingFace official docs
**Context**: Production deployment, Spaces hardware, model cards, dataset repos
**Philosophy**: Complete reference for HF Hub deployment ecosystem

---

## Overview

This document provides comprehensive 2025 guidance on deploying to HuggingFace Hub:
- **Spaces**: GPU hardware, secrets, Docker, pricing
- **Model repos**: Model cards, metadata, discoverability
- **Dataset repos**: Parquet format, Data Studio, large-scale uploads

**Related**: Complements existing knowledge in `05-huggingface-deployment-vlm-2025-01-30.md`

---

## Part 1: HuggingFace Spaces - Complete Reference

### Hardware Options & Pricing (2025)

**From HuggingFace Pricing Page + Official Docs**:

```
CPU Options:
- CPU Basic (Free):     2 vCPU, 16GB RAM, 50GB disk      → $0/hour
- CPU Upgrade:          8 vCPU, 32GB RAM, 50GB disk      → $0.03/hour

GPU Options (sorted by cost):
- T4 Small:      16GB VRAM, 4 vCPU, 15GB RAM, 50GB       → $0.60/hour
- T4 Medium:     16GB VRAM, 8 vCPU, 30GB RAM, 100GB      → $0.90/hour
- L4 (1x):       24GB VRAM, 8 vCPU, 30GB RAM, 100GB      → $1.05/hour (newer than T4)
- A10G Small:    24GB VRAM, 4 vCPU, 15GB RAM, 110GB      → $1.05/hour
- A10G Large:    24GB VRAM, 12 vCPU, 46GB RAM, 200GB     → $3.15/hour
- A10G (2x):     48GB VRAM, 24 vCPU, 92GB RAM, 1000GB    → $5.70/hour
- A10G (4x):     96GB VRAM, 48 vCPU, 184GB RAM, 2000GB   → $10.80/hour
- A100 Large:    40GB VRAM, 12 vCPU, 142GB RAM, 1000GB   → $4.13/hour

Storage Tiers (Persistent):
- Ephemeral (default):  50GB, not persistent              → Free
- Small:                +20GB persistent                  → $5/month
- Medium:               +150GB persistent                 → $25/month
- Large:                +1TB persistent                   → $100/month
```

**Key Insights**:
- **T4 Small ($0.60/hr)** is the sweet spot for 2B VLM demos (16GB VRAM, sufficient for inference)
- **A10G Small ($1.05/hr)** for 7B models or batch inference (24GB VRAM)
- **Free CPU Basic** for lightweight demos (non-ML apps, small models on CPU)
- **Community GPU grants** available for innovative Spaces (apply in Settings tab)

**Cost Comparison** (from TutorialsWithAI):
```
HuggingFace T4: $0.60/hour
AWS SageMaker T4 equivalent: $1.26/hour

Savings: ~50% cheaper than AWS for same hardware
```

### Creating a Space

**Step-by-Step** (from HuggingFace Spaces docs):

**1. Create New Space**:
```
Navigate to: https://huggingface.co/spaces
Click: "Create new Space"
Configure:
- Name: your-space-name
- SDK: Gradio / Streamlit / Docker / Static HTML
- Visibility: Public / Private
- License: Choose appropriate license
- Hardware: CPU Basic (free) or upgrade later
```

**2. Space Structure**:
```
your-space/
├── README.md           # Space card (with YAML metadata)
├── app.py              # Main application (Gradio/Streamlit)
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages (apt-get)
└── .env                # NOT committed (use Secrets instead)
```

**3. README.md Metadata Example**:
```yaml
---
title: ARR-COC VLM Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
models:
  - Qwen/Qwen3-VL-2B-Instruct
datasets:
  - your-username/your-dataset
tags:
  - vision
  - multimodal
  - computer-vision
---
```

**Key Metadata Fields**:
- `sdk`: gradio, streamlit, docker, static
- `sdk_version`: Pin for reproducibility
- `app_file`: Entry point (default: app.py)
- `models`: Links to model repos (improves discoverability)
- `datasets`: Links to dataset repos
- `tags`: For search and filtering

### Secrets & Environment Variables

**From HuggingFace Spaces Settings docs**:

**Two Types**:
1. **Variables**: Non-sensitive, publicly visible, copied on duplication
2. **Secrets**: Sensitive, hidden values, NOT copied on duplication

**When to Use Each**:
```
Variables (Public):
✅ Model repo IDs (e.g., MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Instruct)
✅ Configuration settings (e.g., MAX_TOKENS=512)
✅ Feature flags (e.g., ENABLE_DEBUG=false)

Secrets (Private):
✅ API keys (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
✅ Access tokens (e.g., HF_TOKEN for private repos)
✅ Database credentials
✅ OAuth client secrets
```

**Setting Secrets/Variables**:
```
1. Go to Space → Settings
2. Scroll to "Repository secrets" or "Variables"
3. Add key-value pairs
4. Save (Space will rebuild automatically)
```

**Accessing in Code**:
```python
import os

# Access secrets or variables (same method)
api_key = os.getenv('OPENAI_API_KEY')
model_id = os.getenv('MODEL_REPO_ID', 'default-value')

# For Static Spaces (JavaScript)
// Available in window.huggingface.variables
const modelId = window.huggingface.variables.MODEL_REPO_ID;
```

**Security Best Practices**:
```
❌ NEVER hardcode secrets in code:
   api_key = "sk-1234567890abcdef"  # BAD!

✅ ALWAYS use environment variables:
   api_key = os.getenv('OPENAI_API_KEY')  # GOOD!

❌ NEVER commit .env files to Git
✅ ALWAYS add .env to .gitignore
✅ ALWAYS use Spaces Secrets for deployment
```

### Docker Spaces

**From HuggingFace Docker Spaces docs + Docker.com tutorial**:

**When to Use Docker**:
- Custom runtime requirements (specific Python version, system libraries)
- Non-Python frameworks (Node.js, Rust, C++)
- Complex multi-service apps (FastAPI + frontend + database)
- Control over entire environment

**Basic Dockerfile Example**:
```dockerfile
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Spaces expects port 7860)
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run application
CMD ["python", "app.py"]
```

**HuggingFace Docker Requirements**:
```
CRITICAL:
- Must expose port 7860 (Spaces default)
- Must bind to 0.0.0.0 (not localhost)
- Health check endpoint recommended (GET /)

Example Gradio app.py for Docker:
```

```python
import gradio as gr

def process(input_text):
    return f"Processed: {input_text}"

demo = gr.Interface(fn=process, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # CRITICAL: Not "localhost"
        server_port=7860,        # CRITICAL: Port 7860
        show_error=True
    )
```

**Accessing Secrets in Docker**:
```dockerfile
# Secrets/variables are injected as environment variables
# No special configuration needed in Dockerfile

# In your Python code:
import os
api_key = os.getenv('API_KEY')

# In Dockerfile, you can set defaults:
ENV MODEL_NAME="default-model"

# Space secrets override Dockerfile ENV values
```

**Docker Build Process**:
```
1. Push Dockerfile + code to Space repo
2. Space automatically builds Docker image
3. Image cached (rebuilt only on changes)
4. Container runs on selected hardware
5. Logs visible in Space settings
```

**Debugging Docker Spaces**:
```bash
# Check build logs in Space settings
# Common issues:
❌ Port not 7860 → Space won't start
❌ Binding to localhost → External access fails
❌ Missing requirements.txt → Build fails
❌ Large base image → Slow builds (use -slim variants)

# Test locally before deploying:
docker build -t my-space .
docker run -p 7860:7860 my-space
# Visit http://localhost:7860
```

### Networking & Security

**From HuggingFace Spaces Networking docs**:

**Allowed Outbound Ports**:
```
✅ HTTP:  Port 80
✅ HTTPS: Port 443
✅ Custom: Port 8080
❌ All other ports: BLOCKED
```

**Use Cases**:
```
✅ API calls to external services (OpenAI, Anthropic, etc.)
✅ Downloading models from HuggingFace Hub
✅ Fetching data from public APIs
❌ Custom databases on non-standard ports (use hosted services)
❌ SSH connections
❌ P2P protocols
```

**Rate Limiting**:
```
Free Spaces:
- No explicit rate limit, but "sleep" after inactivity
- Fair use policy (no cryptocurrency mining, abuse)

Paid Hardware:
- No sleep, runs 24/7
- Standard network bandwidth limits apply
```

### Lifecycle Management

**From HuggingFace Spaces docs**:

**Free CPU Spaces**:
```
Behavior:
- Sleep after ~48 hours of inactivity
- Wake up on first request (cold start: 10-30 seconds)
- Sleep again after inactivity

Solution for 24/7 availability:
- Upgrade to paid hardware (even T4 Small: $0.60/hr = $14.40/day)
- Or accept cold starts for research demos
```

**Manual Pause**:
```
Location: Space → Settings → "Pause Space"

When to use:
- Pause paid Space when not in use (save costs)
- Development/testing phases
- Temporary maintenance

Important:
- Paused time is NOT billed
- Only owner can unpause
- Useful for cost control
```

**Automatic Rebuilds**:
```
Triggered by:
✅ Git push to repo (any file change)
✅ Secrets/variables update
✅ Hardware upgrade/downgrade
❌ NOT triggered by: Code edits in web UI without commit

Rebuild time:
- Gradio/Streamlit: 1-3 minutes
- Docker: 5-15 minutes (depends on image size)
- Cached builds: 30 seconds - 2 minutes
```

### Helper Environment Variables

**From HuggingFace Spaces docs**:

**Auto-injected Variables**:
```python
import os

# Space metadata
author = os.getenv('SPACE_AUTHOR_NAME')           # "osanseviero"
repo_name = os.getenv('SPACE_REPO_NAME')          # "i-like-flan"
title = os.getenv('SPACE_TITLE')                  # "I Like Flan"
space_id = os.getenv('SPACE_ID')                  # "osanseviero/i-like-flan"
host = os.getenv('SPACE_HOST')                    # "osanseviero-i-like-flan.hf.space"

# Hardware info
cpu_cores = os.getenv('CPU_CORES')                # "4"
memory = os.getenv('MEMORY')                      # "15Gi"

# Creator info (useful for org Spaces)
creator_id = os.getenv('SPACE_CREATOR_USER_ID')   # "6032802e1f993496bc14d9e3"
# Get creator info: https://huggingface.co/api/users/{creator_id}/overview

# OAuth (if enabled)
if os.getenv('OAUTH_CLIENT_ID'):
    oauth_client_id = os.getenv('OAUTH_CLIENT_ID')
    oauth_secret = os.getenv('OAUTH_CLIENT_SECRET')
    oauth_scopes = os.getenv('OAUTH_SCOPES')              # "openid profile"
    openid_url = os.getenv('OPENID_PROVIDER_URL')        # HF OpenID endpoint
```

**Use Cases**:
```python
# Example: Dynamic Space info display
def get_space_info():
    return f"""
    Running on: {os.getenv('SPACE_HOST')}
    Created by: {os.getenv('SPACE_AUTHOR_NAME')}
    Hardware: {os.getenv('CPU_CORES')} CPU cores, {os.getenv('MEMORY')} RAM
    """

# Example: Conditional features based on Space
if os.getenv('SPACE_AUTHOR_NAME') == 'your-username':
    # Enable debug mode for your own Spaces
    DEBUG = True
else:
    DEBUG = False

# Example: Tracking duplications
original_space = "original-author/original-space"
current_space = os.getenv('SPACE_ID')
if current_space != original_space:
    print(f"This Space was duplicated from {original_space}")
```

---

## Part 2: Model Cards - Best Practices

### Model Card Metadata (YAML)

**From HuggingFace Model Cards docs + GeeksforGeeks guide**:

**Essential Metadata** (top of README.md):
```yaml
---
# Required for proper discovery
language:
  - en
  - zh
license: mit
library_name: transformers
tags:
  - vision
  - multimodal
  - image-to-text
  - visual-question-answering
datasets:
  - liuhaotian/LLaVA-Instruct-150K
base_model: Qwen/Qwen3-VL-2B-Instruct
pipeline_tag: image-to-text

# Model card metadata
model-index:
  - name: ARR-COC-VIS
    results:
      - task:
          type: image-to-text
        dataset:
          name: VQAv2
          type: vqa-v2
        metrics:
          - name: Accuracy
            type: accuracy
            value: 82.3
          - name: Visual Tokens Used (avg)
            type: efficiency
            value: 186
        source:
          name: Internal Evaluation
          url: https://huggingface.co/spaces/your-username/arr-coc-evaluation

# Optional but recommended
model_name: ARR-COC-VIS
model_type: vision-language-model
inference: true
widget:
  - src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat.jpg
    text: "What is in this image?"
---
```

**Critical Metadata Fields** (from Friendli.ai best practices):

**1. library_name** (REQUIRED):
```yaml
library_name: transformers  # Auto-enables model download button
# Other options: timm, diffusers, sentence-transformers, etc.
```

**2. pipeline_tag** (REQUIRED for widgets):
```yaml
pipeline_tag: image-to-text
# Determines which widget to show
# Options: text-generation, image-to-text, text-to-image, etc.
```

**3. base_model** (REQUIRED for fine-tunes):
```yaml
base_model: Qwen/Qwen3-VL-2B-Instruct
# Shows "Fine-tuned from" badge
# Enables filtering by base model
```

**4. datasets** (improves discoverability):
```yaml
datasets:
  - liuhaotian/LLaVA-Instruct-150K
  - coco-2017
# Links to dataset pages
# Shows "Trained on" section
```

**5. license** (REQUIRED for legal clarity):
```yaml
license: mit
# Standard licenses: mit, apache-2.0, gpl-3.0, etc.
# Custom license:
license: other
license_name: custom-license-name
license_link: https://example.com/license.txt
```

**Common Mistakes** (from Friendli.ai pitfalls article):

```yaml
# ❌ BAD: Missing library_name
tags:
  - transformers  # Tag is not enough!

# ✅ GOOD: Explicit library_name
library_name: transformers

# ❌ BAD: Wrong pipeline_tag for model type
pipeline_tag: text-generation  # For a VLM!

# ✅ GOOD: Correct pipeline_tag
pipeline_tag: image-to-text

# ❌ BAD: Incomplete base_model for adapters
base_model: meta-llama/Llama-2-7b
# Missing adapter-specific metadata

# ✅ GOOD: Complete adapter metadata
base_model: meta-llama/Llama-2-7b
library_name: peft
tags:
  - lora
  - adapter
```

### Using the Metadata UI

**From HuggingFace docs + screenshot analysis**:

**Step-by-Step**:
```
1. Navigate to model page
2. Click "Edit model card" (top right)
3. Metadata UI appears on right side
4. Fill fields (autocomplete suggestions appear)
5. Click sections to expand/collapse
6. Changes auto-added to YAML section
7. Preview card on left side
8. Commit changes
```

**UI Sections**:
```
- Model Name & Description
- Languages (multi-select with autocomplete)
- License (dropdown of common licenses)
- Datasets (search HF datasets)
- Metrics (standard ML metrics)
- Tags (freeform + suggestions)
- Library (dropdown)
- Pipeline Tag (dropdown based on library)
- Base Model (search HF models)
```

**UI Limitations**:
```
❌ Doesn't support all metadata fields
❌ Can't edit complex model-index structures
❌ No support for custom license links

For these, edit YAML directly in README.md
```

### Evaluation Results in Model Cards

**From HuggingFace model-index spec**:

**Purpose**:
- Display results on model page
- Auto-submit to Papers with Code leaderboards
- Enable filtering by performance

**Format**:
```yaml
model-index:
  - name: ARR-COC-VIS-2B          # Model name
    results:
      - task:                       # Task definition
          type: image-to-text       # Task type (standardized)
          name: Visual Question Answering  # Human-readable name
        dataset:
          name: VQAv2               # Dataset name
          type: vqa-v2              # Dataset identifier
          split: validation         # Which split
          revision: main            # Dataset version
        metrics:
          - name: Accuracy          # Metric name
            type: accuracy          # Metric type
            value: 82.3             # Score
            verified: false         # Verified by HF?
          - name: Tokens/Image (avg)
            type: efficiency
            value: 186
          - name: Latency (ms)
            type: latency
            value: 450
        source:                     # Where results came from
          name: Internal Evaluation
          url: https://huggingface.co/spaces/user/evaluation-space

      - task:                       # Second benchmark
          type: image-to-text
          name: Image Captioning
        dataset:
          name: COCO Captions
          type: coco-captions
        metrics:
          - name: CIDEr
            type: cider
            value: 125.4
```

**Rendered UI**:
```
Model Card → "Model Evaluation" section appears
- Table with benchmark results
- Links to datasets
- Links to evaluation source
- Compare button (if multiple models on same benchmark)
```

**Best Practices**:
```
✅ Use standardized task types (image-to-text, not "VQA")
✅ Link to evaluation Space or code
✅ Include multiple metrics (accuracy, latency, efficiency)
✅ Specify dataset split and version
❌ Don't claim "verified: true" (HF sets this)
❌ Don't use custom metric names without description
```

### CO2 Emissions Reporting

**From HuggingFace CO2 docs**:

**Why Report**:
- Environmental transparency
- Compare carbon footprint across models
- Track sustainability progress

**Format**:
```yaml
---
# ... other metadata ...
co2_eq_emissions:
  emissions: 12.5              # kg CO2
  source: "CodeCarbon"         # Tracking tool
  training_type: "pre-training"
  geographical_location: "USA, Virginia"
  hardware_used: "8x NVIDIA A100"
---
```

**Tools for Tracking**:
```python
# Using CodeCarbon
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# ... training code ...

emissions = tracker.stop()
print(f"CO2 emissions: {emissions} kg")

# Automatically saves to emissions.csv
# Add to model card metadata
```

**Model Card Section**:
```markdown
## Environmental Impact

This model was trained using the following hardware and runtime:

- **Hardware**: 8x NVIDIA A100 (40GB)
- **Training time**: 48 hours
- **Cloud provider**: AWS (us-east-1)
- **Carbon footprint**: 12.5 kg CO2 eq.
- **Tracking tool**: [CodeCarbon](https://codecarbon.io)

Estimated carbon emissions calculated using [Code Carbon](https://github.com/mlco2/codecarbon).
```

### LaTeX in Model Cards

**From HuggingFace Model Cards FAQ**:

**Supported** (KaTeX rendering):
```markdown
## Math Formulas

**Display mode** (centered):
$$
\text{Relevance}(p) = \alpha \cdot H(p) + \beta \cdot S(p) + \gamma \cdot C(p, q)
$$

**Inline mode**: The loss function \\( \mathcal{L} = -\log P(y|x) \\) is minimized.

**Important**:
- Display mode: $$ ... $$
- Inline mode: \\( ... \\) (no space between slashes and parentheses)
```

**Example Formulas**:
```latex
# Display mode
$$
\text{Token Budget}(p) = \text{MIN\_TOKENS} + \frac{\text{Relevance}(p)}{\sum_i \text{Relevance}(p_i)} \cdot (\text{MAX\_TOKENS} - \text{MIN\_TOKENS})
$$

# Inline mode
The attention mechanism computes \\( \text{softmax}(QK^T / \sqrt{d_k})V \\).

# Matrix notation
$$
\begin{bmatrix}
Q_1 \\ Q_2 \\ Q_3
\end{bmatrix}
\cdot
\begin{bmatrix}
K_1 & K_2 & K_3
\end{bmatrix}
$$
```

---

## Part 3: Dataset Repositories

### Uploading Datasets - File Formats

**From HuggingFace Datasets docs**:

**Supported Formats**:
```
Recommended (best support):
✅ Parquet (.parquet)          - Best for tabular data, analytics
✅ CSV (.csv, .tsv)            - Easy to create, widely supported
✅ JSON Lines (.jsonl)         - Good for nested data
✅ WebDataset (.tar)           - Best for large-scale image/audio

Also supported:
✅ JSON (.json)
✅ Arrow (.arrow)
✅ Text (.txt)
✅ Images (.png, .jpg, .jpeg, .gif, .bmp, .tiff)
✅ Audio (.wav, .mp3, .flac, .ogg, .m4a)
✅ Video (.mp4, .avi, .mov, .mkv)
✅ PDF (.pdf)

Compression:
✅ .zip, .gz, .zst, .bz2, .lz4, .xz
```

**Which Format to Choose**:

```
Use Case → Format

Tabular data (small, <1GB):
→ CSV or JSON Lines
   Pros: Easy to create, human-readable
   Cons: Slower than Parquet, no rich typing

Tabular data (large, >1GB):
→ Parquet
   Pros: Fast, compressed, rich typing, columnar
   Cons: Binary format (need tools to inspect)

Nested/hierarchical data:
→ JSON Lines
   Pros: Supports nested structures
   Cons: Slower than Parquet for large files

Image datasets (small-medium, <10K images):
→ Raw image files (.jpg, .png)
   Pros: Easy to access individual files
   Cons: Slow for large datasets (file system overhead)

Image datasets (large, >10K images):
→ WebDataset (.tar) or Parquet (with image bytes)
   Pros: Fast streaming, reduces file system overhead
   Cons: More complex to create

Audio datasets (similar logic as images):
→ Raw audio files (<10K) or WebDataset (>10K)

Mixed multimodal (images + text + audio):
→ Parquet with metadata + raw files
   or WebDataset with multiple modalities
```

**Parquet Best Practices**:
```python
# Create Parquet from pandas
import pandas as pd

df = pd.DataFrame({
    'text': ['Hello', 'World'],
    'label': [0, 1],
    'score': [0.9, 0.8]
})

df.to_parquet('data.parquet', engine='pyarrow', compression='snappy')

# Create Parquet from datasets library
from datasets import Dataset

dataset = Dataset.from_pandas(df)
dataset.save_to_disk('dataset_cache')
dataset.push_to_hub('your-username/your-dataset')  # Auto-converts to Parquet
```

**CSV/TSV Tips**:
```python
# Ensure UTF-8 encoding
df.to_csv('data.csv', encoding='utf-8', index=False)

# For large CSVs, split into chunks
for i, chunk in enumerate(pd.read_csv('large.csv', chunksize=10000)):
    chunk.to_csv(f'data_{i:04d}.csv', index=False)
```

### Uploading via push_to_hub

**From HuggingFace Datasets library docs**:

**Method 1: From Datasets Library**:
```python
from datasets import load_dataset, Dataset
import pandas as pd

# Option A: Load existing dataset and push
dataset = load_dataset('csv', data_files='data.csv')
dataset.push_to_hub('your-username/dataset-name', private=True)

# Option B: Create from scratch
data = {
    'text': ['Example 1', 'Example 2'],
    'label': [0, 1]
}
dataset = Dataset.from_dict(data)
dataset.push_to_hub('your-username/dataset-name')

# Option C: From pandas
df = pd.read_csv('data.csv')
dataset = Dataset.from_pandas(df)
dataset.push_to_hub('your-username/dataset-name')
```

**Method 2: Split Datasets**:
```python
from datasets import DatasetDict, Dataset

# Create train/val/test splits
train_data = Dataset.from_dict({...})
val_data = Dataset.from_dict({...})
test_data = Dataset.from_dict({...})

dataset_dict = DatasetDict({
    'train': train_data,
    'validation': val_data,
    'test': test_data
})

dataset_dict.push_to_hub('your-username/dataset-name')
```

**Method 3: Large Datasets (Streaming)**:
```python
from huggingface_hub import HfApi
import os

api = HfApi()

# Create repo
api.create_repo(
    repo_id='your-username/large-dataset',
    repo_type='dataset',
    private=False
)

# Upload folder (automatically chunks large files)
api.upload_folder(
    folder_path='./local_dataset',
    repo_id='your-username/large-dataset',
    repo_type='dataset',
    commit_message='Upload large dataset'
)
```

### Dataset Cards

**From HuggingFace Dataset Cards docs**:

**Essential Metadata**:
```yaml
---
# Dataset card metadata (at top of README.md)
annotations_creators:
  - crowdsourced
  - expert-generated
language_creators:
  - found
  - crowdsourced
language:
  - en
  - zh
license: cc-by-4.0
multilinguality:
  - multilingual
size_categories:
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - image-to-text
  - visual-question-answering
task_ids:
  - image-captioning
  - visual-question-answering
paperswithcode_id: vqa-v2
pretty_name: Visual Question Answering v2
tags:
  - vision
  - multimodal
configs:
  - config_name: default
    data_files:
      - split: train
        path: "train/*.parquet"
      - split: validation
        path: "val/*.parquet"
      - split: test
        path: "test/*.parquet"
---
```

**Dataset Card Structure**:
```markdown
# Dataset Card for VQA v2

## Dataset Description

- **Homepage**: https://visualqa.org/
- **Repository**: https://github.com/GT-Vision-Lab/VQA
- **Paper**: [Link to paper]
- **Point of Contact**: researcher@university.edu

### Dataset Summary

Brief 2-3 sentence summary of what the dataset contains and its purpose.

### Supported Tasks and Leaderboards

- `visual-question-answering`: The dataset can be used to train models for VQA.
- Leaderboard: https://eval.ai/web/challenges/challenge-page/830/overview

### Languages

English (en)

## Dataset Structure

### Data Instances

Example of one data instance:

```json
{
  "image": <PIL.Image>,
  "question": "What color is the car?",
  "answers": ["red", "red", "red", "red", "red", "red", "red", "red", "red", "red"],
  "question_type": "what color",
  "answer_type": "other"
}
```

### Data Fields

- `image`: PIL Image object
- `question`: string, the question about the image
- `answers`: list of 10 string answers from different annotators
- `question_type`: string, category of question
- `answer_type`: string, type of answer expected

### Data Splits

| Split | Size |
|-------|------|
| train | 443,757 |
| validation | 214,354 |
| test | 447,793 |

## Dataset Creation

### Curation Rationale

Why was this dataset created? What problem does it solve?

### Source Data

Where did the data come from?

#### Initial Data Collection and Normalization

How was the data collected? Was it cleaned or processed?

#### Who are the source language producers?

Who created the original text/images?

### Annotations

#### Annotation process

How were labels/annotations added?

#### Who are the annotators?

Who added the annotations? (Crowdworkers, experts, automatic)

### Personal and Sensitive Information

Does the dataset contain personal information? How is it handled?

## Considerations for Using the Data

### Social Impact of Dataset

Potential positive and negative impacts.

### Discussion of Biases

Known biases in the dataset.

### Other Known Limitations

Other limitations (coverage, quality, etc.)

## Additional Information

### Dataset Curators

Who maintains this dataset?

### Licensing Information

CC-BY-4.0

### Citation Information

```bibtex
@inproceedings{goyal2017making,
  title={Making the V in VQA matter...},
  author={Goyal, Yash and ...},
  booktitle={CVPR},
  year={2017}
}
```

### Contributions

Thanks to [@username] for adding this dataset.
```

### Data Studio (Dataset Viewer)

**From HuggingFace Data Studio docs**:

**What is Data Studio**:
- Automatic preview of dataset contents
- No code needed to explore data
- Enabled by default for public datasets
- Available for PRO user private datasets

**Requirements**:
```
✅ Supported format (Parquet, CSV, JSON Lines, images, audio)
✅ Files in standard locations (train.csv, data/*.parquet, etc.)
✅ Dataset card with basic metadata
✅ Files < 5GB each (for automatic processing)

Optional:
- configs in dataset card for custom file locations
```

**Configuration** (if auto-detection fails):
```yaml
---
# In dataset card metadata
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train-*.parquet"
      - split: test
        path: "data/test-*.parquet"
  - config_name: subset_a
    data_files:
      - split: train
        path: "subset_a/train.csv"
---
```

**Features**:
```
- Browse rows interactively
- Filter and search
- View images/audio inline
- Download samples
- Check schema and statistics
- SQL queries (DuckDB integration)
```

**Troubleshooting**:
```
Dataset Viewer not working?

1. Check file formats (must be supported)
2. Check file locations (must match configs)
3. Check file sizes (<5GB recommended)
4. Check dataset card has metadata
5. Check repo is public (or PRO user for private)

Force refresh:
- Edit dataset card (triggers rebuild)
- Contact HF support if issues persist
```

### Large-Scale Datasets

**From HuggingFace Storage docs**:

**Upload Strategies**:

**1. Small datasets (<1GB)**:
```python
# Direct push_to_hub
from datasets import Dataset

dataset = Dataset.from_dict({...})
dataset.push_to_hub('username/dataset')
```

**2. Medium datasets (1-10GB)**:
```python
# Upload with huggingface_hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='./data',
    repo_id='username/dataset',
    repo_type='dataset'
)
```

**3. Large datasets (>10GB)**:
```python
# Upload by chunks (for very large files)
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj='large_file.parquet',
    path_in_repo='data/large_file.parquet',
    repo_id='username/dataset',
    repo_type='dataset',
    commit_message='Upload large file'
)

# Or upload folder with automatic chunking
api.upload_folder(
    folder_path='./data',
    repo_id='username/dataset',
    repo_type='dataset',
    multi_commits=True,              # Enable chunked commits
    multi_commits_verbose=True        # Show progress
)
```

**Storage Limits** (2025):
```
Free accounts:
- No hard limit on public datasets
- Fair use policy applies
- Large uploads may be rate-limited

PRO accounts ($9/month):
- Higher rate limits
- Priority support
- Private datasets with Data Studio

Enterprise:
- Dedicated support
- Custom storage arrangements
- SLA guarantees
```

**Best Practices for Large Datasets**:
```
✅ Use Parquet format (efficient compression)
✅ Split into multiple files (easier to resume uploads)
✅ Use multi_commits for >10GB uploads
✅ Test with small subset first
❌ Don't upload as single huge file
❌ Don't upload uncompressed data
❌ Don't include unnecessary files (.DS_Store, __pycache__, etc.)
```

**Parquet Content-Defined Chunking** (July 2025 feature):
```
From: HuggingFace Blog (Jul 25, 2025)

New Xet storage layer:
- Deduplicates Parquet chunks
- Faster uploads for large datasets
- Faster downloads with caching
- Reduces storage costs

Automatically enabled for new uploads.
No code changes needed.
```

---

## Part 4: Advanced Topics

### OAuth Integration

**From HuggingFace OAuth docs**:

**When to Use**:
- Need to know user identity
- Access user's private repos
- Save user preferences

**Setup**:
```yaml
# In Space README.md metadata
---
oauth:
  enabled: true
  scopes: "openid profile"
---
```

**Environment Variables** (auto-injected):
```python
import os

oauth_client_id = os.getenv('OAUTH_CLIENT_ID')
oauth_client_secret = os.getenv('OAUTH_CLIENT_SECRET')
oauth_scopes = os.getenv('OAUTH_SCOPES')  # "openid profile"
openid_url = os.getenv('OPENID_PROVIDER_URL')

# User info from OAuth flow (requires implementation)
# See: https://huggingface.co/docs/hub/spaces-oauth
```

### Duplicating Spaces

**From HuggingFace Spaces docs**:

**Use Cases**:
- Fork demo for experimentation
- Create personal copy with upgraded hardware
- Test changes without affecting original

**Process**:
```
1. Click three dots (top right of Space)
2. Select "Duplicate this Space"
3. Configure:
   - Owner: Your account or organization
   - Name: New Space name
   - Visibility: Public or Private
   - Hardware: Choose tier
   - Storage: Choose persistent storage
   - Secrets: Re-enter (not copied automatically)
4. Click "Duplicate Space"
```

**What Gets Copied**:
```
✅ All code files
✅ README.md and metadata
✅ Public variables (from original Space)
✅ Hardware settings (can override)
❌ Secrets (must re-enter)
❌ Persistent storage data
❌ Build cache
```

### Space Lifecycle - Free vs Paid

**Summary Table**:

| Feature | Free CPU | Paid Hardware |
|---------|----------|---------------|
| Sleep after inactivity | ✅ Yes (~48h) | ❌ No, runs 24/7 |
| Cold start time | 10-30 sec | N/A |
| Can pause manually | ✅ Yes | ✅ Yes (no billing when paused) |
| Persistent storage | ❌ No (ephemeral only) | ✅ Yes (paid tiers) |
| Community GPU grants | ✅ Available | N/A |
| Build caching | ✅ Yes | ✅ Yes (faster) |

---

## Summary: HuggingFace Deployment Quick Reference

### Spaces Checklist

**MVP Deployment (Free CPU)**:
- [ ] Create Space on Hub
- [ ] Add README.md with metadata
- [ ] Write app.py (Gradio/Streamlit)
- [ ] Add requirements.txt
- [ ] Set secrets/variables if needed
- [ ] Push to repo → auto-builds
- [ ] Test on CPU Basic (free)
- [ ] Accept cold starts for research demo

**Production Deployment (Paid Hardware)**:
- [ ] Upgrade to T4 Small ($0.60/hr minimum)
- [ ] Add persistent storage if needed
- [ ] Configure secrets properly
- [ ] Test load and latency
- [ ] Monitor costs in Billing page
- [ ] Set up pause schedule if needed

### Model Card Checklist

**Essential Metadata**:
- [ ] library_name (transformers, timm, etc.)
- [ ] pipeline_tag (for widgets)
- [ ] license (legal requirement)
- [ ] language (for discoverability)
- [ ] base_model (if fine-tuned/adapted)
- [ ] datasets (training data)
- [ ] tags (search keywords)

**Optional but Recommended**:
- [ ] model-index with evaluation results
- [ ] CO2 emissions data
- [ ] Link to paper (if published)
- [ ] Usage examples in model card body
- [ ] Limitations and biases section

### Dataset Upload Checklist

**Before Upload**:
- [ ] Choose format (Parquet recommended for >1GB)
- [ ] Split into train/val/test if applicable
- [ ] Compress if needed (.gz, .zip)
- [ ] Remove unnecessary files
- [ ] Test with small sample first

**During Upload**:
- [ ] Use push_to_hub() or upload_folder()
- [ ] Add dataset card with metadata
- [ ] Configure Data Studio if needed
- [ ] Set visibility (public/private)
- [ ] Add license information

**After Upload**:
- [ ] Verify Data Studio preview works
- [ ] Test loading with datasets library
- [ ] Link from model cards if applicable
- [ ] Share with community if public

---

**Related Oracle Files**:
- [05-huggingface-deployment-vlm-2025-01-30.md](05-huggingface-deployment-vlm-2025-01-30.md) - VLM-specific deployment
- [08-gpu-memory-debugging-vlm-2025-01-30.md](../../karpathy/practical-implementation/08-gpu-memory-debugging-vlm-2025-01-30.md) - T4 memory management
- [09-gradio-testing-patterns-2025-01-30.md](../../karpathy/practical-implementation/09-gradio-testing-patterns-2025-01-30.md) - Gradio app patterns

**Primary Sources**:
- HuggingFace Spaces Overview docs (official)
- HuggingFace Model Cards docs (official)
- HuggingFace Datasets Adding docs (official)
- HuggingFace Pricing page (2025-01-31)
- GeeksforGeeks: Hugging Face Model Card (Aug 7, 2025)
- Friendli.ai: Common Pitfalls in Sharing Models (Jul 1, 2025)
- Docker.com: Build ML Apps with HF Docker Spaces (Mar 23, 2023)
- TutorialsWithAI: HuggingFace Spaces Review (2025)
- HuggingFace Blog: Parquet Content-Defined Chunking (Jul 25, 2025)

**Last Updated**: 2025-01-31
**Version**: 1.0 - Comprehensive expansion from Bright Data research + official docs
