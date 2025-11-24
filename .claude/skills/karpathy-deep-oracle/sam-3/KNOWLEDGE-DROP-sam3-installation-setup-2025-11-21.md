# SAM 3 Installation & Setup

## Overview

SAM 3 (Segment Anything Model 3) requires specific Python, PyTorch, and CUDA versions for optimal performance. This guide covers complete installation procedures, dependency management, and common troubleshooting solutions.

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA 12.6 or higher (for GPU support)
- CUDA-compatible NVIDIA GPU

**Hardware Recommendations:**
- GPU with sufficient VRAM (SAM 3 has 848M parameters)
- Minimum 16GB system RAM
- SSD for faster model loading

**Driver Requirements:**
- NVIDIA Driver 560 or later (for CUDA 12.6)

---

## Installation Steps

### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.12
conda create -n sam3 python=3.12

# Deactivate current environment (if any)
conda deactivate

# Activate sam3 environment
conda activate sam3
```

**Why Python 3.12?**
- SAM 3 officially requires Python 3.12+
- While pyproject.toml shows support for Python 3.8-3.12, the README specifies 3.12 as minimum for optimal compatibility

### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.7+ with CUDA 12.6 support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Important Notes:**
- The `--index-url` flag ensures you get CUDA 12.6 compatible builds
- PyTorch 2.7 requires CUDA 12.6 wheels from the PyTorch index
- Standard pip install without index URL may install CPU-only version

### Step 3: Clone Repository and Install Package

```bash
# Clone the SAM 3 repository
git clone https://github.com/facebookresearch/sam3.git

# Navigate to repository
cd sam3

# Install in editable mode
pip install -e .
```

### Step 4: Install Optional Dependencies

**For running example notebooks:**
```bash
pip install -e ".[notebooks]"
```

**For development and training:**
```bash
pip install -e ".[train,dev]"
```

**For full installation with all features:**
```bash
pip install -e ".[notebooks,train,dev]"
```

---

## Dependencies

### Core Dependencies (from pyproject.toml)

```python
dependencies = [
    "timm>=1.0.17",        # PyTorch Image Models
    "numpy==1.26",          # Numerical computing (pinned version!)
    "tqdm",                 # Progress bars
    "ftfy==6.1.1",         # Text fixing (for text encoder)
    "regex",               # Regular expressions
    "iopath>=0.1.10",      # IO path handling
    "typing_extensions",    # Type hints
    "huggingface_hub",      # Model checkpoint downloads
]
```

### Notebook Dependencies

```python
notebooks = [
    "matplotlib",           # Plotting
    "jupyter",             # Jupyter notebooks
    "notebook",            # Notebook interface
    "ipywidgets",          # Interactive widgets
    "ipycanvas",           # Canvas rendering
    "ipympl",              # Matplotlib integration
    "pycocotools",         # COCO evaluation
    "decord",              # Video decoding
    "opencv-python",       # Image/video processing
    "einops",              # Tensor operations
    "scikit-image",        # Image processing
    "scikit-learn",        # Machine learning utilities
]
```

### Training Dependencies

```python
train = [
    "hydra-core",          # Configuration management
    "submitit",            # Job submission
    "tensorboard",         # Training visualization
    "zstandard",           # Compression
    "scipy",               # Scientific computing
    "torchmetrics",        # Metrics
    "fvcore",              # Facebook core utilities
    "fairscale",           # Distributed training
    "scikit-image",        # Image processing
    "scikit-learn",        # ML utilities
]
```

### Development Dependencies

```python
dev = [
    "pytest",              # Testing
    "pytest-cov",          # Coverage
    "black==24.2.0",       # Formatting
    "ufmt==2.8.0",         # Unified formatting
    "ruff-api==0.1.0",     # Linting
    "usort==1.0.2",        # Import sorting
    "gitpython==3.1.31",   # Git integration
    "yt-dlp",              # Video downloading
    "pandas",              # Data analysis
    "opencv-python",       # Image processing
    "pycocotools",         # COCO tools
    "numba",               # JIT compilation
    "python-rapidjson",    # Fast JSON
]
```

---

## HuggingFace Authentication

### Why Authentication is Required

SAM 3 checkpoints are hosted on HuggingFace and require access approval before download. This is a gated repository requiring explicit access request.

### Step 1: Request Access

1. Visit the SAM 3 HuggingFace repository: https://huggingface.co/facebook/sam3
2. Sign in or create a HuggingFace account
3. Read and accept the license terms
4. Wait for access approval (usually immediate)

### Step 2: Generate Access Token

1. Go to HuggingFace Settings: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name your token (e.g., "sam3-access")
4. Select "Read" permission (minimum required)
5. Click "Generate token"
6. Copy the token (starts with `hf_`)

### Step 3: Authenticate

**Method 1: CLI Login (Interactive)**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Method 2: CLI Login (Non-interactive)**
```bash
huggingface-cli login --token hf_your_token_here
```

**Method 3: Environment Variable**
```bash
export HF_TOKEN=hf_your_token_here
```

**Method 4: Python (Programmatic)**
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Verify Authentication

```python
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f"Logged in as: {user['name']}")
```

---

## Verification

### Check Installation

```python
# Verify PyTorch and CUDA
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Verify SAM 3 import
import sam3
print(f"SAM 3 version: {sam3.__version__}")

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Test Model Loading

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# This will download checkpoints on first run
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("SAM 3 loaded successfully!")
```

---

## Common Installation Issues

### Issue 1: CUDA Version Mismatch

**Symptoms:**
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`
- Model runs on CPU instead of GPU

**Solutions:**
```bash
# Check your CUDA version
nvidia-smi  # Shows driver CUDA version

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Issue 2: PyTorch Not Compiled with CUDA

**Symptoms:**
- `RuntimeError: Torch not compiled with CUDA enabled`
- `torch.cuda.is_available()` returns `False`

**Solutions:**
```bash
# Check if CUDA build was installed
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with explicit CUDA index
pip uninstall torch torchvision torchaudio
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Issue 3: HuggingFace Authentication Failure

**Symptoms:**
- `401 Client Error: Unauthorized`
- `Access to model facebook/sam3 is restricted`

**Solutions:**
1. Verify you've requested and received access at https://huggingface.co/facebook/sam3
2. Re-login with token:
```bash
huggingface-cli login --token hf_your_token_here
```
3. Check token permissions (needs at least "Read" access)

### Issue 4: NumPy Version Conflict

**Symptoms:**
- `ImportError: cannot import name 'xxx' from 'numpy'`
- Version compatibility errors

**Solutions:**
```bash
# SAM 3 requires numpy 1.26 specifically
pip install numpy==1.26
```

### Issue 5: decord Import Error

**Symptoms:**
- `ModuleNotFoundError: No module named 'decord'`
- Video processing fails

**Solutions:**
```bash
# Install decord for video support
pip install decord

# On some systems, may need conda
conda install -c conda-forge decord
```

### Issue 6: Missing CUDA Libraries

**Symptoms:**
- `libcudart.so.12: cannot open shared object file`
- CUDA library not found

**Solutions:**
```bash
# Install CUDA toolkit 12.6
# On Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6

# Add to PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

### Issue 7: Memory Issues (Out of Memory)

**Symptoms:**
- `CUDA out of memory`
- Process killed during model loading

**Solutions:**
```python
# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Use smaller batch sizes
# Reduce image resolution
# Enable gradient checkpointing for training
```

### Issue 8: CPU/MPS/TPU Backend Issues

**Current Status (as of Nov 2025):**
- SAM 3 currently requires CUDA GPU
- CPU/MPS/TPU support is a feature request (GitHub issue #164)
- Importing video utilities throws "Torch not compiled with CUDA enabled" on non-CUDA systems

**Workaround:**
- Use a CUDA-compatible system for SAM 3
- For development/testing without GPU, wait for backend-agnostic support

---

## Alternative Installation Methods

### Using pip directly (without clone)

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

### Docker Container

```dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN pip install git+https://github.com/facebookresearch/sam3.git
RUN pip install huggingface_hub

# Authenticate (pass token at runtime)
ENV HF_TOKEN=${HF_TOKEN}
```

### Conda Environment File

```yaml
# sam3_environment.yml
name: sam3
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - pip:
    - torch==2.7.0
    - torchvision
    - torchaudio
    - git+https://github.com/facebookresearch/sam3.git
```

Install with:
```bash
conda env create -f sam3_environment.yml
```

---

## Quick Start After Installation

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("your_image.jpg")
inference_state = processor.set_image(image)

# Segment with text prompt
output = processor.set_text_prompt(
    state=inference_state,
    prompt="dog"
)

# Get results
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print(f"Found {len(masks)} instances")
```

---

## Version Compatibility Matrix

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| Python | >= 3.12 | Recommended, supports 3.8-3.12 |
| PyTorch | >= 2.7.0 | CUDA 12.6 build required |
| CUDA | >= 12.6 | With driver 560+ |
| NumPy | == 1.26 | Pinned version required |
| timm | >= 1.0.17 | PyTorch Image Models |

---

## Sources

**GitHub Repository:**
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - Official SAM 3 repository (accessed 2025-11-23)
- [pyproject.toml](https://github.com/facebookresearch/sam3/blob/main/pyproject.toml) - Dependencies specification

**HuggingFace:**
- [facebook/sam3](https://huggingface.co/facebook/sam3) - Model checkpoints
- [HuggingFace CLI Documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli) - Authentication guide

**PyTorch:**
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) - Installation guide
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) - Version history

**GitHub Issues:**
- [Issue #164](https://github.com/facebookresearch/sam3/issues/164) - Backend-agnostic inference request
- [Issue #191](https://github.com/facebookresearch/sam3/issues/191) - decord dependency issue

**NVIDIA:**
- [NVIDIA PyTorch Container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html) - Container releases
