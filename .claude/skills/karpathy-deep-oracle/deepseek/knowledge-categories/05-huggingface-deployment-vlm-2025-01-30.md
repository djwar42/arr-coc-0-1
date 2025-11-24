# HuggingFace Deployment for Vision-Language Models (2025)

**Engineering guide for deploying VLMs on HuggingFace Spaces with Gradio demos, model repos, and private development workflows**

**Source**: Platonic Dialogues 38, 38 Addendum, 41
**Focus**: Production deployment architecture, not MVP development
**Validated**: 2025 HuggingFace Spaces documentation, Gradio integration patterns

---

## When to Use This Guide

**Use when you have:**
- ✅ A working VLM implementation
- ✅ Validated hypothesis (accuracy metrics)
- ✅ Results worth sharing publicly

**Don't use if:**
- ❌ Still building MVP (see `karpathy/practical-implementation/11-mvp-first-philosophy-2025-01-30.md`)
- ❌ Testing locally only
- ❌ No public deployment planned

**Key principle from Part 41**: Infrastructure follows validation, not the reverse.

---

## Section 1: HuggingFace Spaces Architecture (Hardware, GPU Options, Configuration)

### Overview: What Are Spaces?

HuggingFace Spaces are **Git repositories** that run ML demos with optional GPU acceleration. Under the hood, Spaces:
- Store code in Git (same tools as model repos)
- Auto-rebuild on each commit
- Provide public URLs (shareable demos)
- Support Gradio, Streamlit, Docker SDKs

**From Part 38**: "Spaces make it easy to deploy ML-powered demos in minutes" (HF docs). Perfect for VLM comparison demos.

### Hardware Options (T4 Small = Sweet Spot for 2B VLMs)

**Official HuggingFace Spaces GPU tiers** (from `mcp__bright-data__scrape_as_markdown` research):

| Hardware | GPU Memory | CPU | RAM | Disk | Hourly Price |
|----------|-----------|-----|-----|------|--------------|
| **CPU Basic** | - | 2 vCPU | 16GB | 50GB | **Free!** |
| CPU Upgrade | - | 8 vCPU | 32GB | 50GB | $0.03 |
| **T4 Small** | **16GB** | 4 vCPU | 15GB | 50GB | **$0.40** ⭐ |
| T4 Medium | 16GB | 8 vCPU | 30GB | 100GB | $0.60 |
| L4 (1x) | 24GB | 8 vCPU | 30GB | 400GB | $0.80 |
| A10G Small | 24GB | 4 vCPU | 14GB | 110GB | $1.00 |
| A10G Large | 24GB | 12 vCPU | 46GB | 200GB | $1.50 |
| A100 Large | 80GB | 12 vCPU | 142GB | 1000GB | $2.50 |
| H100 | 80GB | 23 vCPU | 240GB | 3000GB | $4.50 |

**⭐ Recommended for ARR-COC-VIS**: **T4 Small** ($0.40/hour)

**Why T4 Small?**
- **Sufficient VRAM**: 2B model (4GB weights) + 3GB activations = 7GB (fits in 16GB with headroom)
- **Cost-effective**: $288/month if running 24/7 ($0.40 × 24 × 30)
- **Free tier available**: Community GPU grants for innovative demos
- **Handles inference**: T4 optimized for bfloat16 inference (Turing architecture)

**From Part 41**: T4 (16GB VRAM) can run ONE 2B VLM model comfortably. Multi-model comparison (ARR-COC vs baseline) requires local workstation or A10G.

### Configuration: README.md Header (Space Metadata)

Spaces use YAML frontmatter in `README.md` to configure hardware and settings:

**Example: ARR-COC-VIS Demo Space**

```yaml
---
title: ARR-COC-VIS Demo
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: t4-small  # Free T4 GPU
suggested_storage: standard
---
```

**Key fields** (from Part 38 + HF docs):
- `sdk`: Choose `gradio`, `streamlit`, `docker`, or `static`
- `sdk_version`: Pin to specific Gradio version (e.g., `5.0.0`)
- `app_file`: Entry point (usually `app.py`)
- `hardware`: `cpu-basic` (free), `t4-small` (paid), `zerogpu` (dynamic), etc.
- `suggested_storage`: `standard` (ephemeral 50GB) vs `small`/`medium`/`large` (persistent)

**From HF docs**: "Each time a new commit is pushed, the Space will automatically rebuild and restart."

### Environment Variables & Secrets Management

**Two types** (from HF Spaces docs):

1. **Variables** (public, visible):
   - Non-sensitive config (e.g., `MODEL_REPO_ID`)
   - Auto-copied when Space is duplicated
   - Accessible via `os.getenv('VARIABLE_NAME')`

2. **Secrets** (private, hidden):
   - API keys, tokens, credentials
   - NOT copied when Space is duplicated
   - Value cannot be read after setting
   - Accessible via `os.getenv('SECRET_NAME')`

**Example: Load model from private repo**

```python
import os
from huggingface_hub import hf_hub_download

# Secret: HF_TOKEN (set in Space settings)
hf_token = os.getenv('HF_TOKEN')

# Download from private repo
arr_coc_weights = hf_hub_download(
    repo_id="YOUR-ORG/arr-coc-vis-private",
    filename="arr_coc_components.safetensors",
    token=hf_token
)
```

**From Part 40 Addendum**: Never hard-code tokens. Use Spaces secrets.

### Sleep Time Settings (Save Costs)

**Free CPU Spaces**: Auto-sleep after 48 hours of inactivity
**Paid GPU Spaces**: Never sleep by default (billed 24/7)

**Custom sleep time** (paid hardware only):
- Set in Space settings
- Options: Never, 15min, 1hr, 3hr, 1 day
- Space "wakes up" when visited (no billing while asleep)
- **Use case**: Development demos (not production)

**From HF docs**: "You are not going to be charged for the upgraded hardware while it is asleep."

### ZeroGPU: Dynamic Allocation (Advanced)

**What is ZeroGPU?**
- GPU allocated **only when function runs** (on-demand)
- Billed per second used (not 24/7)
- Auto-scales to demand
- Max 120 seconds per request

**Example: Compare models with ZeroGPU**

```python
import spaces
import gradio as gr

@spaces.GPU(duration=120)  # Allocate GPU for 120 seconds max
def compare_models(image, query):
    """GPU allocated ONLY when this function runs"""
    baseline_result = baseline_model.generate(image, query)
    arr_coc_result = arr_coc_model.generate(image, query)
    return baseline_result, arr_coc_result

demo = gr.Interface(
    fn=compare_models,
    inputs=[gr.Image(), gr.Textbox()],
    outputs=[gr.Textbox(), gr.Textbox()]
)
```

**When to use ZeroGPU vs T4 Small?**
- **ZeroGPU**: Bursty traffic, sporadic use, multiple models (costs scale with usage)
- **T4 Small**: Steady traffic, always-on demo (fixed cost)

**From Part 38**: ARR-COC demo = T4 Small (simpler, predictable cost). ZeroGPU for multi-model comparison.

### Helper Environment Variables (Auto-Available)

HuggingFace Spaces expose these **automatically** (from HF docs):

```python
import os

# Hardware specs
CPU_CORES = os.getenv('CPU_CORES')  # e.g., "4"
MEMORY = os.getenv('MEMORY')        # e.g., "15Gi"

# Space metadata
SPACE_AUTHOR_NAME = os.getenv('SPACE_AUTHOR_NAME')  # "YOUR-ORG"
SPACE_REPO_NAME = os.getenv('SPACE_REPO_NAME')      # "arr-coc-demo"
SPACE_TITLE = os.getenv('SPACE_TITLE')              # "ARR-COC-VIS Demo"
SPACE_ID = os.getenv('SPACE_ID')                    # "YOUR-ORG/arr-coc-demo"
SPACE_HOST = os.getenv('SPACE_HOST')                # "your-org-arr-coc-demo.hf.space"
```

**Use case**: Dynamic model loading based on Space author:

```python
# Load model dynamically based on duplicated Space owner
space_author = os.getenv('SPACE_AUTHOR_NAME', 'default-org')
model_repo = f"{space_author}/arr-coc-vis"
```

**From Part 38**: Useful when users duplicate your Space (they can use their own model repo).

---

## Section 2: Model Repository Structure (Model Cards, Config.json, Safetensors)

### Overview: Model Repo vs Code Repo

**Two separate repositories** (from Part 38):

1. **Model Repository** (HuggingFace Hub):
   - Trained weights (`model.safetensors` or `.bin`)
   - Model card (`README.md` with YAML metadata)
   - Architecture config (`config.json`)
   - Python modules (if needed)

2. **Code Repository** (GitHub):
   - Full development code
   - Tests, evaluation scripts
   - Training pipelines
   - Research docs (Platonic Dialogues)

**Why separate?**
- **Model repo**: Optimized for inference, lightweight, HF Hub features (widgets, downloads)
- **Code repo**: Full development, version control, collaboration

**From Part 38**: "GitHub for development, HuggingFace for deployment."

### Model Card Format (README.md with YAML Frontmatter)

Model cards use **Markdown + YAML** (CommonMark spec):

**File: YOUR-ORG/arr-coc-vis/README.md**

```markdown
---
license: apache-2.0
base_model: Qwen/Qwen3-VL-2B-Instruct
tags:
- vision-language
- relevance-realization
- vervaeke
- foveated-vision
- adaptive-attention
library_name: transformers
pipeline_tag: image-text-to-text
datasets:
- coco
- vqav2
metrics:
- accuracy
---

# ARR-COC-VIS: Adaptive Relevance Realization for Vision-Language Models

**Adaptive Relevance Realization - Context Optimized Compression - Vision**

ARR-COC-VIS implements John Vervaeke's relevance realization framework for vision-language models,
enabling query-aware, context-adaptive visual token allocation.

## 🎯 Key Innovation

Traditional VLMs process images uniformly. ARR-COC-VIS realizes relevance dynamically:
- **Variable token allocation:** 64-400 tokens per region based on query relevance
- **Adaptive tensions:** Context-dependent strategy selection
- **Vervaekean framework:** Three ways of knowing + opponent processing

## 📊 Performance

| Metric | Standard Qwen3-VL | ARR-COC-VIS | Improvement |
|--------|------------------|-------------|-------------|
| Inference Time | 60ms | 45ms | **25% faster** ⚡ |
| Memory Usage | 2.8GB | 2.1GB | **25% reduction** 💾 |
| VQA Accuracy | 67.8% | 68.2% | **+0.4%** ✓ |
| Diverse Queries | 64.5% | 69.8% | **+5.3%** 🎯 |

*Tested on Qwen3-VL-2B-Instruct base model*

## 🏗️ Architecture

... (architecture details, code examples, citation)
```

**Key YAML fields** (from HF model cards docs + Part 38):

- **`base_model`**: Specifies fine-tuned base (displays on model page as "Fine-tuned from Qwen/Qwen3-VL-2B-Instruct")
- **`tags`**: Discovery keywords (users can filter models by tags)
- **`library_name`**: Which library loads this model (`transformers`, `flair`, etc.)
- **`pipeline_tag`**: Task type (`image-text-to-text`, `text-generation`, etc.)
- **`datasets`**: Links to datasets used for training (shows "Trained on:" message)
- **`metrics`**: Evaluation metrics used

**From HF docs**: "The Hub will infer the type of relationship from the current model to the base model (`adapter`, `merge`, `quantized`, `finetune`)."

### Config.json (Architecture Metadata)

**Purpose**: Stores model architecture details for auto-loading.

**Example: ARR-COC-VIS config.json**

```json
{
  "architectures": ["ARR_COC_Qwen3VL"],
  "model_type": "qwen3_vl",
  "base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
  "arr_coc_components": {
    "texture_channels": 40,
    "knowing_scorers": ["propositional", "perspectival", "participatory"],
    "tension_count": 3,
    "token_allocation_range": [64, 400]
  },
  "transformers_version": "4.48.0",
  "torch_dtype": "bfloat16",
  "vocab_size": 151936,
  "hidden_size": 3584,
  "num_hidden_layers": 28,
  "num_attention_heads": 28
}
```

**Key fields** (from Part 38):
- **`architectures`**: Custom model class name (for `from_pretrained()`)
- **`model_type`**: Base model type (`qwen3_vl`)
- **`arr_coc_components`**: Custom metadata (your architecture specifics)
- **`torch_dtype`**: Precision (`bfloat16` for inference)

**From Part 41**: Always include architecture metadata to prevent checkpoint corruption errors.

### Model Weights: Safetensors vs PyTorch .bin

**Two formats** (from `mcp__bright-data__search_engine` research):

1. **Safetensors** (`.safetensors`):
   - **76.6× faster on CPU**, 2× faster on GPU (vs PyTorch .bin)
   - Safe format (prevents arbitrary code execution)
   - Zero-copy loading (memory-efficient)
   - **Recommended for 2025**

2. **PyTorch** (`.bin`):
   - Legacy format (uses Python pickle)
   - Slower loading
   - Security risk (can execute arbitrary code)
   - **Deprecated for Hub uploads**

**File naming conventions**:
- Single file: `model.safetensors` or `pytorch_model.bin`
- Sharded (large models): `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, etc.

**Example: Upload safetensors**

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model weights
api.upload_file(
    path_or_fileobj="arr_coc_components.safetensors",
    path_in_repo="model.safetensors",
    repo_id="YOUR-ORG/arr-coc-vis",
    token=hf_token
)
```

**From HF docs**: "safetensors is a safe and fast file format for storing and loading tensors."

**Conversion (if needed)**:

```python
from safetensors.torch import save_file
import torch

# Load PyTorch .bin
state_dict = torch.load("pytorch_model.bin")

# Save as safetensors
save_file(state_dict, "model.safetensors")
```

**From Reddit/Stack Overflow research**: Always prefer safetensors for new uploads (faster + safer).

### Requirements.txt (Dependency Management)

**Pin all versions** (from Part 38):

```txt
# Base dependencies
transformers==4.48.0
torch==2.5.0
torchvision==0.20.0
pillow==11.0.0

# Qwen3-VL specific
qwen-vl-utils==0.0.8

# ARR-COC components
numpy==2.2.2
accelerate==1.2.1

# Inference optimization
bitsandbytes==0.45.0  # For 8-bit/4-bit quantization (optional)
```

**Why pin versions?**
- Reproducibility (model works in 6 months)
- Avoid breaking changes (transformers API changes)
- Easier debugging (known versions)

**From Part 41**: "Pin everything. Future you will thank present you."

---

## Section 3: Dataset Repository Structure (Dataset Cards, Parquet Files, Data Organization)

### Overview: Dataset Repos on HuggingFace Hub

**Purpose**: Host evaluation data, test images, benchmark results.

**Example repo**: `YOUR-ORG/arr-coc-benchmarks`

**From Part 38**: "Dataset repos are Git repos with special data file handling (Parquet, images, etc.)."

### Dataset Card Format (README.md with YAML)

**File: YOUR-ORG/arr-coc-benchmarks/README.md**

```markdown
---
license: apache-2.0
task_categories:
- visual-question-answering
- image-to-text
language:
- en
tags:
- benchmark
- vlm
- relevance-realization
pretty_name: ARR-COC-VIS Benchmark Results
size_categories:
- 1K<n<10K
---

# ARR-COC-VIS Benchmark Results

Evaluation results for ARR-COC-VIS adaptive relevance realization system.

## Dataset Structure

```
arr-coc-benchmarks/
├── README.md (this file)
├── test_images/
│   ├── coco_val_*.jpg (500 images)
│   └── vqa_val_*.jpg (500 images)
├── results/
│   ├── vqa_results.json (VQA accuracy per image)
│   ├── efficiency_metrics.csv (time, memory, tokens)
│   └── ablation_studies.json (component removal effects)
└── annotations/
    ├── coco_annotations.json
    └── vqa_annotations.json
```

## Benchmark Splits

| Split | Images | Queries | Purpose |
|-------|--------|---------|---------|
| test_standard | 500 | 500 | Standard VQA (uniform complexity) |
| test_diverse | 500 | 500 | Diverse queries (relevance varies) |
| test_efficiency | 100 | 100 | Speed/memory benchmarks |

## Usage

```python
from datasets import load_dataset

# Load benchmark data
dataset = load_dataset("YOUR-ORG/arr-coc-benchmarks")

# Access test images
test_images = dataset['test_standard']['image']

# Access results
vqa_results = dataset['test_standard']['vqa_results']
```

## Metrics

- **VQA Accuracy**: Exact match + partial match
- **Inference Time**: Per-image processing time (ms)
- **Memory Usage**: Peak VRAM during inference (GB)
- **Token Allocation**: Tokens used per image region

## Citation

```bibtex
@dataset{arr_coc_benchmarks,
  title={ARR-COC-VIS Benchmark Results},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/datasets/YOUR-ORG/arr-coc-benchmarks}
}
```
```

**Key YAML fields** (from HF datasets docs):
- **`task_categories`**: Type of task (`visual-question-answering`, `image-to-text`, etc.)
- **`size_categories`**: Dataset size (`n<1K`, `1K<n<10K`, etc.)
- **`language`**: Language codes (`en`, `zh`, etc.)
- **`pretty_name`**: Display name on Hub

### Data File Organization (Parquet for Large Datasets)

**Recommended structure** (from HF docs):

```
your-dataset/
├── README.md
├── data/
│   ├── train-00000-of-00003.parquet
│   ├── train-00001-of-00003.parquet
│   ├── train-00002-of-00003.parquet
│   ├── test-00000-of-00001.parquet
│   └── validation-00000-of-00001.parquet
└── images/
    ├── train/
    │   ├── img_0001.jpg
    │   └── ...
    └── test/
        ├── img_0001.jpg
        └── ...
```

**Why Parquet?**
- **Columnar format**: Fast filtering, efficient storage
- **Streaming**: Can load subsets without downloading entire dataset
- **Type safety**: Schema enforced
- **HF Hub native**: Auto-detected, viewer-friendly

**Convert JSON to Parquet**:

```python
import pandas as pd

# Load JSON results
with open('vqa_results.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as Parquet
df.to_parquet('vqa_results.parquet', index=False)
```

**From Part 38**: Use Parquet for large datasets (>1K samples). JSON is fine for small test sets.

### Dataset Viewer (Automatic)

HuggingFace Hub **automatically renders** datasets with:
- Image previews
- Table views (for Parquet/CSV)
- Filtering UI
- Download links

**Example viewer features** (from HF docs):
- Preview first 100 rows
- Filter by column values
- Sort by columns
- Export subsets

**No configuration needed** – just upload Parquet files to `data/` directory.

### Licensing Dataset (CRITICAL)

**From HF datasets cards docs**: Always specify license.

**Common licenses for benchmarks**:
- **CC-BY-4.0**: Attribution required, commercial use OK
- **CC-BY-NC-4.0**: Attribution required, non-commercial only
- **MIT**: Permissive, commercial use OK
- **Apache-2.0**: Permissive, patent grant

**Example (COCO images)**:

```yaml
---
license: cc-by-4.0  # COCO license
```

**From Part 38**: "Respect original dataset licenses. COCO = CC-BY-4.0."

---

## What's Next (PART 2 will cover...)

Remaining sections for this file:
- **Section 4**: Gradio App Integration (app.py structure, requirements.txt, Dockerfile)
- **Section 5**: Training with HuggingFace Trainer (TrainingArguments, callbacks)
- **Section 6**: Private Development Workflow (private repos, W&B, stealth launch)
- **Section 7**: Deployment Pipeline (local → testing → private Space → public)
- **Section 8**: Memory Constraints on Spaces (T4 limits, multi-model reality)
- **Section 9**: Debugging Spaces Deployment (logs, build failures, OOM)
- **Section 10**: FastAPI + Gradio for Production (when to use each)

**Research needed for PART 2**: Queries 7-12 (Trainer API, private repos, Gradio + FastAPI patterns)

---

## Sources & References

**Platonic Dialogues**:
- Part 38: The Implementation Structure (HuggingFace integration strategy, repository structure, model cards)
- Part 38 Addendum: Implementation code guide (complete examples, deployment code)
- Part 41: The Reality Check (deployment reality, T4 limits, single-model demos)

**Web Research (2025)**:
- HuggingFace Spaces GPU documentation: Hardware specs, pricing, configuration
- HuggingFace Spaces overview: SDK options, deployment flow, environment variables
- Model cards documentation: YAML frontmatter, metadata fields, discovery features
- Dataset cards documentation: Task categories, Parquet format, licensing
- Safetensors vs PyTorch .bin: Performance benchmarks (76.6× faster), security, best practices

**Key principle from Part 41**: "Parts 38-40 describe the DESTINATION. Part 41 describes the PATH. Build infrastructure after MVP validates hypothesis."

---

## 11. Advanced GPU Optimization

### 11.1 ZeroGPU Dynamic Allocation

**H200 Hardware with Automatic Scaling**:

```python
# Using ZeroGPU on HuggingFace Spaces (Sept 2025 feature)
import spaces

@spaces.GPU  # Decorator for dynamic GPU allocation
def inference(image, query):
    """
    ZeroGPU automatically allocates H200 GPU only during function execution.
    GPU is released when function completes (no idle cost).
    """
    model = load_model()  # Model loaded on-demand
    result = model.generate(image, query)
    return result  # GPU released here

# Advanced: AOTI (Ahead-of-Time Compilation) optimization
@spaces.GPU(aoti=True)  # Enable torch.compile speedup
def fast_inference(image, query):
    """
    AOTI compiles model graph ahead of time.
    Result: 2-3× faster inference on H200.
    """
    model = load_model()
    result = model.generate(image, query)
    return result
```

**How ZeroGPU Works**:
- **Dynamic allocation**: GPU allocated only when @spaces.GPU function called
- **Automatic release**: GPU freed immediately after function returns
- **Cost savings**: Pay only for actual GPU usage (not idle time)
- **H200 hardware**: 141GB HBM3 memory, 4.8TB/s bandwidth
- **AOTI speedup**: torch.compile optimization (2-3× faster)

**When to use ZeroGPU**:
- ✅ Inference demos with sporadic traffic
- ✅ Public Spaces with unpredictable usage
- ❌ NOT for training (training needs persistent GPU)
- ❌ NOT for high-concurrency (use T4/A10G persistent instead)

**Source**: [HuggingFace ZeroGPU AOTI](https://huggingface.co/blog/zerogpu-aoti), [Bright Data Query 13]

---

### 11.2 GPU Memory Profiling Tools

**PyTorch Native Profiling**:

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Method 1: PyTorch Profiler (built-in)
def profile_training_step(model, batch):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,  # Track tensor shapes
        profile_memory=True,  # Track memory allocations
        with_stack=True      # Include stack traces
    ) as prof:
        # Training step
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

    # Print memory summary
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=10
    ))

    # Export Chrome trace (visualize in chrome://tracing)
    prof.export_chrome_trace("trace.json")

# Method 2: Simple memory tracking
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"[{tag}] GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {max_allocated:.2f} GB")
        print(f"  Free: {(reserved - allocated):.2f} GB")

# Usage in training loop
log_gpu_memory("Before forward")
outputs = model(**batch)
log_gpu_memory("After forward")
loss = outputs.loss
loss.backward()
log_gpu_memory("After backward")
optimizer.step()
log_gpu_memory("After optimizer")
```

**NVIDIA Nsight Systems (Advanced)**:

```bash
# Profile entire training run
nsys profile --trace=cuda,nvtx,osrt --output=training_profile python train.py

# Analyze profile
nsys stats training_profile.nsys-rep

# Generate report
nsys stats --report cuda_gpu_trace training_profile.nsys-rep
```

**What to profile**:
- **Forward pass**: Model weights + activations memory
- **Backward pass**: Gradients memory (typically 2× forward)
- **Optimizer state**: Momentum + variance (2× model size for Adam)
- **Peak usage**: Max memory during training (OOM prevention)

**Source**: [PyTorch GPU memory visualization](https://huggingface.co/blog/train_memory), [Bright Data Query 13]

---

### 11.3 Memory Optimization Techniques

**Gradient Checkpointing (50% memory savings)**:

```python
from transformers import AutoModelForCausalLM

# Enable gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16
)
model.gradient_checkpointing_enable()

# TrainingArguments integration
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...,
    gradient_checkpointing=True,  # ← Enable checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False}  # PyTorch 2.0+
)
```

**How it works**:
- **Normal**: Store all activations for backward pass (high memory)
- **Checkpointing**: Store only subset of activations, recompute on backward (50% less memory, 30% slower)
- **Trade-off**: Memory for compute

**Gradient Accumulation (simulate larger batch)**:

```python
# Instead of batch_size=32 (OOM)
# Use batch_size=4 with gradient_accumulation_steps=8
# Effective batch size = 4 × 8 = 32

training_args = TrainingArguments(
    per_device_train_batch_size=4,      # Fits in 16GB GPU
    gradient_accumulation_steps=8,      # Accumulate 8 steps
    # Effective batch size = 32
)
```

**CPU Offloading (handle models > GPU memory)**:

```python
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM

# Offload layers to CPU when GPU full
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "large-model",
        torch_dtype=torch.float16
    )

device_map = infer_auto_device_map(
    model,
    max_memory={0: "16GB", "cpu": "32GB"},  # GPU + CPU limits
    no_split_module_classes=["Block"]       # Don't split transformer blocks
)

model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map=device_map,
    torch_dtype=torch.float16,
    offload_folder="offload"  # Disk offload if needed
)
```

**Source**: [HuggingFace GPU optimization](https://huggingface.co/docs/transformers/perf_infer_gpu_one), [Bright Data Query 13]

---

### 11.4 Debugging OOM Errors

**Common OOM causes and fixes**:

```python
# Issue 1: Accumulating computation graphs
# ❌ Bad
for batch in dataloader:
    loss = model(**batch).loss
    total_loss += loss  # ← Accumulates graph!

# ✅ Good
for batch in dataloader:
    loss = model(**batch).loss
    total_loss += loss.item()  # ← Detach from graph

# Issue 2: Not clearing cache
# ✅ Good
import torch
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Clear unused memory
    del outputs, loss
    torch.cuda.empty_cache()

# Issue 3: Eval mode not set
# ❌ Bad (stores gradients in eval)
for batch in val_dataloader:
    loss = model(**batch).loss

# ✅ Good
model.eval()
with torch.no_grad():  # ← Disable gradient computation
    for batch in val_dataloader:
        loss = model(**batch).loss
model.train()
```

**Emergency OOM recovery**:

```python
import gc

def clear_memory():
    """Nuclear option: clear all GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Clear IPC memory

    # Print memory status
    if torch.cuda.is_available():
        print(f"Memory freed. Current usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Use when OOM occurs
try:
    outputs = model(**batch)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM detected. Clearing memory...")
        clear_memory()
        # Reduce batch size and retry
        batch = {k: v[:len(v)//2] for k, v in batch.items()}
        outputs = model(**batch)
```

**Source**: [GPU memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management), [Bright Data Query 13]

---

## 12. LoRA Adapter Merging Strategies

### 12.1 Basic Adapter Merging

**merge_and_unload() - Standard workflow**:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Load LoRA adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/lora_adapter",  # Checkpoint directory
    torch_dtype=torch.bfloat16
)

# 3. Merge adapter into base model
merged_model = peft_model.merge_and_unload()

# 4. Save merged model (standalone, no adapter needed)
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# 5. Upload to HuggingFace Hub
merged_model.push_to_hub("username/arr-coc-vis-2b-merged")
tokenizer.push_to_hub("username/arr-coc-vis-2b-merged")
```

**What merge_and_unload() does**:
- Adds LoRA weights (A, B matrices) to base model weights
- Formula: `W_merged = W_base + (LoRA_B @ LoRA_A) * scaling`
- Result: Single model (no adapter dependency)
- Use case: Deployment, sharing, inference optimization

**Source**: [PEFT merge_and_unload docs](https://huggingface.co/docs/peft), [Bright Data Query 14]

---

### 12.2 Quantized Adapter Merging

**Merging LoRA adapters with quantized models**:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Issue: Can't merge adapter onto quantized model directly
# Solution: Dequantize → merge → re-quantize

# Step 1: Load base model in fp16 (NOT quantized yet)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,  # ← fp16, not quantized
    device_map="auto"
)

# Step 2: Load LoRA adapter (trained on quantized model)
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/lora_adapter",
    torch_dtype=torch.float16
)

# Step 3: Merge adapter into fp16 base
merged_model = peft_model.merge_and_unload()

# Step 4: Save merged model (fp16)
merged_model.save_pretrained("./merged_fp16")

# Step 5 (optional): Re-quantize merged model for deployment
quantized_merged = AutoModelForCausalLM.from_pretrained(
    "./merged_fp16",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    ),
    device_map="auto"
)
```

**Why this pattern?**
- LoRA trained on quantized model (QLoRA) stores fp16 adapters
- Can't add fp16 adapter to 4-bit weights directly
- Must dequantize → merge → re-quantize

**Source**: [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization), [Bright Data Query 14]

---

### 12.3 Advanced Merging Methods

**New PEFT merging algorithms (2024+)**:

```python
from peft import PeftModel

# Method 1: TIES (Trim, Elect, and Merge)
# Resolves conflicts when merging multiple adapters
merged_model = peft_model.merge_and_unload(
    merge_method="ties",
    ties_k=0.2  # Keep top 20% of weights
)

# Method 2: DARE (Drop And REscale)
# Randomly drops adapter weights, rescales survivors
merged_model = peft_model.merge_and_unload(
    merge_method="dare",
    dare_p=0.5  # Drop 50% of weights
)

# Method 3: Task Arithmetic
# Merge multiple task-specific adapters
from peft import load_peft_weights

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("base")

# Load multiple adapters
adapter1 = load_peft_weights("adapter_task1")
adapter2 = load_peft_weights("adapter_task2")

# Merge with weighted combination
merged_weights = {
    k: 0.5 * adapter1[k] + 0.5 * adapter2[k]
    for k in adapter1.keys()
}

# Apply merged weights
peft_model = PeftModel(base_model, merged_weights)
final_model = peft_model.merge_and_unload()
```

**When to use each method**:
- **Standard merge**: Single adapter, no conflicts
- **TIES**: Multiple adapters with overlapping parameters
- **DARE**: Reduce merged model size (sparse merging)
- **Task Arithmetic**: Multi-task model (combine task-specific adapters)

**Source**: [PEFT merging methods](https://huggingface.co/blog/peft_merging), [Bright Data Query 14]

---

### 12.4 Merging Best Practices

**Workflow for production deployment**:

```python
# 1. Train LoRA adapter
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(base_model, lora_config)
trainer = Trainer(model=peft_model, ...)
trainer.train()

# 2. Save adapter checkpoint
peft_model.save_pretrained("./lora_checkpoint")

# 3. Test adapter before merging
test_adapter_quality(peft_model)  # Custom validation

# 4. Merge if quality passes
if validation_passed:
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("./merged_model")

# 5. Test merged model (should match adapter performance)
test_merged_quality(merged_model)

# 6. Compare sizes
import os
adapter_size = os.path.getsize("./lora_checkpoint/adapter_model.bin") / 1e6
merged_size = os.path.getsize("./merged_model/model.safetensors") / 1e6

print(f"Adapter size: {adapter_size:.1f} MB")  # ~10-50 MB
print(f"Merged size: {merged_size:.1f} MB")    # ~4000 MB (full model)
print(f"Compression: {merged_size / adapter_size:.0f}× larger")
```

**When NOT to merge**:
- ✅ **Keep adapter separate** if:
  - Serving multiple adapters on same base (LoRA switching)
  - Frequent adapter updates (don't re-upload 4GB each time)
  - Storage constrained (adapter = 10MB, merged = 4GB)
- ✅ **Merge** if:
  - Single production model (no switching needed)
  - Simplify deployment (one model file)
  - Inference speed (merged = faster than adapter)

**Source**: [PEFT methods guide](https://huggingface.co/blog/samuellimabraz/peft-methods), [Bright Data Query 14]

---

## 13. Multi-GPU Training with Accelerate

### 13.1 device_map='auto' - Inference Only!

**Critical distinction: Inference vs Training**:

```python
from transformers import AutoModelForCausalLM

# ✅ CORRECT: device_map='auto' for INFERENCE
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",  # ← Automatic model parallelism
    torch_dtype=torch.bfloat16
)

# Inference works (model split across GPUs if needed)
outputs = model.generate(**inputs)

# ❌ WRONG: device_map='auto' for TRAINING
# This will ERROR during backward pass!
# Reason: device_map wraps model in torch.no_grad contexts

# ✅ CORRECT: Use DDP/FSDP for training
from accelerate import Accelerator

accelerator = Accelerator()  # Auto-detects multi-GPU
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16
)
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# Training works (DDP handles multi-GPU)
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)  # ← Gradients work!
    optimizer.step()
```

**Why device_map fails for training**:
- device_map splits model layers across GPUs
- Uses `torch.no_grad()` contexts for efficiency
- Backward pass requires gradients → crashes
- **Solution**: Use DDP (DistributedDataParallel) or FSDP instead

**Source**: [HuggingFace Accelerate docs](https://huggingface.co/docs/transformers/en/accelerate), [Bright Data Query 15]

---

### 13.2 Multi-GPU Training Setup

**4-line integration with Accelerate**:

```python
# Original single-GPU code
model = AutoModelForCausalLM.from_pretrained("model")
optimizer = AdamW(model.parameters())

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# ↓ Add 4 lines for multi-GPU ↓

from accelerate import Accelerator

accelerator = Accelerator()  # Line 1: Create accelerator

model = AutoModelForCausalLM.from_pretrained("model")
optimizer = AdamW(model.parameters())

# Line 2: Prepare model, optimizer, dataloader
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)  # Line 3: Use accelerator.backward
    optimizer.step()
    optimizer.zero_grad()

# Line 4: Save only from main process
if accelerator.is_main_process:
    model.save_pretrained("./model")
```

**What Accelerate handles automatically**:
- ✅ Detects GPU count (DDP if multi-GPU)
- ✅ Distributes batches across GPUs
- ✅ Synchronizes gradients
- ✅ Mixed precision (fp16/bf16) if configured
- ✅ Gradient accumulation
- ✅ DeepSpeed/FSDP integration

**Source**: [Accelerate multi-GPU guide](https://github.com/huggingface/accelerate), [Bright Data Query 15]

---

### 13.3 Mixed Precision Configuration

**Accelerate config controls precision** (not TrainingArguments):

```bash
# Run once to configure Accelerate
accelerate config

# Select options:
# - Compute environment: This machine
# - Number of machines: 1
# - Number of GPUs: 2 (or auto-detect)
# - Mixed precision: bf16 (or fp16, or no)
# - DeepSpeed: No (or Yes for >2 GPUs)
```

**Generated config file** (`~/.cache/huggingface/accelerate/default_config.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
machine_rank: 0
num_machines: 1
mixed_precision: bf16  # ← Controls precision
use_cpu: false
```

**Using in code**:

```python
from accelerate import Accelerator

# Precision comes from config file
accelerator = Accelerator()  # Reads ~/.cache/huggingface/accelerate/default_config.yaml

# Check what precision is used
print(f"Mixed precision: {accelerator.mixed_precision}")  # bf16

# Model automatically uses bf16 after prepare()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# Forward/backward automatically in bf16
for batch in dataloader:
    outputs = model(**batch)  # ← Runs in bf16
    loss = outputs.loss
    accelerator.backward(loss)  # ← Gradients in bf16
    optimizer.step()
```

**Override config in code**:

```python
# Ignore config file, specify precision directly
accelerator = Accelerator(mixed_precision="bf16")

# Or disable mixed precision
accelerator = Accelerator(mixed_precision="no")
```

**Source**: [Accelerate mixed precision](https://huggingface.co/docs/accelerate), [Bright Data Query 15]

---

### 13.4 DDP vs FSDP vs DeepSpeed

**Choosing distributed strategy**:

```python
# Option 1: DDP (DistributedDataParallel) - Default
# Use when: Model fits in single GPU
# Behavior: Replicates model on each GPU, sync gradients

accelerator = Accelerator()  # Defaults to DDP if multi-GPU

# Option 2: FSDP (Fully Sharded Data Parallel)
# Use when: Model too large for single GPU (7B+)
# Behavior: Shards model + optimizer across GPUs

from accelerate import FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",  # Shard everything
    cpu_offload=False,               # Keep on GPU
    auto_wrap_policy=lambda module, recurse, nonwrapped_numel: True
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Option 3: DeepSpeed ZeRO (Stage 2 or 3)
# Use when: Massive models (70B+), need maximum efficiency
# Behavior: Shards optimizer (Stage 2) or model+optimizer (Stage 3)

from accelerate.utils import DeepSpeedPlugin

ds_plugin = DeepSpeedPlugin(
    zero_stage=2,  # ZeRO-2 (optimizer sharding)
    gradient_accumulation_steps=4,
    gradient_clipping=1.0,
    offload_optimizer_device="cpu",  # Offload optimizer to CPU
    offload_param_device="none"
)

accelerator = Accelerator(deepspeed_plugin=ds_plugin)
```

**Memory comparison (2B model, 2× A100)**:

| Strategy | GPU Memory | Speed | Use Case |
|----------|-----------|-------|----------|
| DDP | 8GB/GPU | Fastest | Model fits single GPU |
| FSDP | 4GB/GPU | Medium | Model doesn't fit single GPU |
| DeepSpeed ZeRO-2 | 3GB/GPU | Slower | Optimizer state too large |
| DeepSpeed ZeRO-3 | 2GB/GPU | Slowest | Massive models (70B+) |

**Source**: [Accelerate distributed training](https://huggingface.co/docs/lerobot/en/multi_gpu_training), [Bright Data Query 15]

---

### 13.5 Multi-GPU Training Example

**Complete training script**:

```python
# train_multi_gpu.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from torch.utils.data import DataLoader

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="bf16",  # Enable bf16
    gradient_accumulation_steps=4  # Accumulate 4 steps
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Prepare optimizer and dataloader
optimizer = AdamW(model.parameters(), lr=2e-5)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Prepare for distributed training
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Training loop
model.train()
for epoch in range(3):
    for step, batch in enumerate(train_dataloader):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass (handles gradient accumulation)
        accelerator.backward(loss)

        # Optimizer step (syncs gradients across GPUs)
        if (step + 1) % accelerator.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Log from main process only
        if accelerator.is_main_process and step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save checkpoint (main process only)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(f"./checkpoint-epoch-{epoch}")
        tokenizer.save_pretrained(f"./checkpoint-epoch-{epoch}")

# Wait for all processes to finish
accelerator.wait_for_everyone()

print("Training complete!")
```

**Run with accelerate launch**:

```bash
# Single command works for 1, 2, 4, or 8 GPUs
accelerate launch train_multi_gpu.py

# Or specify GPU count manually
accelerate launch --num_processes=2 train_multi_gpu.py

# With DeepSpeed
accelerate launch --config_file ds_config.yaml train_multi_gpu.py
```

**Source**: [Multi-GPU tutorial](https://www.digitalocean.com/community/tutorials/multi-gpu-on-raw-pytorch-with-hugging-faces-accelerate-library), [Bright Data Query 15]

---

## Sources & References (Updated - 15 Queries)

### Bright Data Research (All Queries)
- Query 1: https://huggingface.co/docs/hub/spaces-gpus - Hardware specs, ZeroGPU
- Query 2: https://huggingface.co/docs/hub/spaces-overview - Deployment flow
- Query 3: Search "HuggingFace Spaces deployment best practices 2025 Gradio GPU"
- Query 4: https://huggingface.co/docs/hub/model-cards - YAML format
- Query 5: https://huggingface.co/docs/hub/datasets-cards - Parquet format
- Query 6: Search "safetensors vs pytorch_model.bin HuggingFace 2025" - Performance benchmarks
- Query 7: https://huggingface.co/docs/transformers/main_classes/trainer - TrainingArguments
- Query 8: Search "HuggingFace TrainerCallback custom validation checkpointing"
- Query 9: Search "HuggingFace Trainer fp16 bf16 mixed precision configuration"
- Query 10: Search "HuggingFace private model repository create upload"
- Query 11: Search "HuggingFace private Space deployment authentication"
- Query 12: Search "Gradio FastAPI production deployment authentication 2024 2025"
- Query 13: Search "HuggingFace Spaces GPU optimization memory profiling 2024 2025" - ZeroGPU, PyTorch Profiler
- Query 14: Search "HuggingFace Transformers LoRA PEFT adapter merging quantization 2024 2025" - TIES, DARE, Task Arithmetic
- Query 15: Search "HuggingFace accelerate device_map mixed precision multi-GPU training 2024 2025" - DDP, FSDP, DeepSpeed

---

**Document Status**: ✅ SECTIONS 1-3 + 11-13 COMPLETE (1,430 lines)
**Remaining**: Sections 4-10 (from PART 2 worker)
**Last Updated**: 2025-01-30
