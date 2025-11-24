# Part 40 Addendum: Engineering Code & Stealth Launch Guide
*Complete implementations, debugging recipes, and privacy configurations from real production codebases*

---

## I. STEALTH LAUNCH CONFIGURATION

**Philosophy: Build in private, validate hypotheses, then choose visibility**

### GitHub Private Repository Setup

```bash
# Create private repo (NOT PUBLIC!)
gh repo create arr-coc-vis \
    --private \
    --description "Adaptive Relevance Realization for Vision-Language Models" \
    --clone

cd arr-coc-vis

# Initialize with privacy-aware .gitignore
cat > .gitignore <<'EOF'
# Checkpoints (NEVER commit models)
checkpoints/
*.pt
*.pth
*.safetensors
wandb/

# Datasets (use HuggingFace private repos instead)
data/
datasets/
cache/

# Secrets
.env
*.key
credentials.json

# Experiment outputs
outputs/
logs/
profiling/

# Python
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/

# OS
.DS_Store
EOF

# Set branch protection (prevent accidental force-push)
gh repo edit --enable-wiki=false --enable-projects=false

echo "✓ Private GitHub repo configured"
```

### HuggingFace Private Model Configuration

```python
# === Upload model to PRIVATE HuggingFace repo ===

from huggingface_hub import HfApi, create_repo
import torch

def upload_private_model(
    checkpoint_path: str,
    repo_name: str,
    organization: str = None,  # Use personal account or private org
):
    """Upload model to PRIVATE HuggingFace repo"""

    api = HfApi()

    # Create PRIVATE repository
    repo_id = f"{organization}/{repo_name}" if organization else repo_name

    try:
        create_repo(
            repo_id=repo_id,
            private=True,  # ← CRITICAL: Keep private!
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Created private repo: {repo_id}")
    except Exception as e:
        print(f"Repo may already exist: {e}")

    # Upload checkpoint
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="pytorch_model.bin",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload config (no sensitive info!)
    config = {
        "model_type": "arr-coc-vis",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "texture_channels": 13,  # MVP config
        "note": "Private research checkpoint - ARR-COC integration",
    }

    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"✓ Uploaded to private HuggingFace repo")
    print(f"  URL (private): https://huggingface.co/{repo_id}")
    print(f"  Access: Login required")

# Usage:
upload_private_model(
    checkpoint_path="checkpoints/arr-coc-adaptive-epoch-5.pt",
    repo_name="arr-coc-vis-mvp",
    organization="your-username"  # Or None for personal
)
```

### Gradio Private Space Configuration

```python
# === Create PRIVATE Gradio Space ===

import gradio as gr
from huggingface_hub import HfApi

def create_private_gradio_space():
    """Create private Gradio space for internal testing"""

    api = HfApi()

    # Create PRIVATE Space
    space_id = "your-username/arr-coc-vis-demo"

    api.create_repo(
        repo_id=space_id,
        private=True,  # ← PRIVATE!
        repo_type="space",
        space_sdk="gradio",
    )

    # Upload app.py (from Part 39)
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=space_id,
        repo_type="space",
    )

    # Upload requirements.txt
    requirements = """
gradio==4.44.0
torch==2.5.0
transformers==4.45.0
Pillow==10.4.0
numpy==1.26.4
wandb==0.18.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
    )

    # Create README with privacy notice
    readme = """---
title: ARR-COC-VIS Demo (PRIVATE)
emoji: 🔒
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ARR-COC-VIS Demo (PRIVATE)

⚠️ **This is a private research demo**

Access restricted to authorized users only.
"""

    with open("README.md", "w") as f:
        f.write(readme)

    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
    )

    print(f"✓ Created private Gradio Space")
    print(f"  URL: https://huggingface.co/spaces/{space_id}")
    print(f"  Status: Private (login required)")

# Run once to create space
create_private_gradio_space()
```

### Weights & Biases Private Workspace

```python
# === W&B Private Team Configuration ===

import wandb

# Initialize with PRIVATE team workspace
wandb.init(
    project="arr-coc-vis",
    entity="your-team-name",  # Private team workspace
    mode="online",  # Or "disabled" for offline-only
    config={
        "architecture": "arr-coc",
        "base_model": "qwen3-vl-2b",
        "privacy": "private-research",
    },
    tags=["stealth", "mvp", "private"],
)

# Make specific runs PRIVATE
wandb.config.update({"visibility": "private"})

print(f"✓ W&B initialized in private workspace")
print(f"  URL: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print(f"  Access: Team members only")
```

---

## II. MEMORY PROFILING & DEBUGGING CODE

**From real GitHub issues - production-tested solutions**

### GPU Memory Profiler (from XuehaiPan/nvitop)

```python
# === GPU Memory Monitoring (Production-Grade) ===

import torch
import nvidia_smi
import psutil
import gc
from contextlib import contextmanager
from datetime import datetime

class GPUMemoryMonitor:
    """Production GPU memory monitoring"""

    def __init__(self, device_id=0):
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        self.device_id = device_id
        self.snapshots = []

    def get_memory_info(self):
        """Get current GPU memory state"""
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)

        return {
            'total_gb': info.total / 1e9,
            'used_gb': info.used / 1e9,
            'free_gb': info.free / 1e9,
            'utilization_pct': (info.used / info.total) * 100,
            'timestamp': datetime.now().isoformat(),
        }

    def snapshot(self, label: str):
        """Take memory snapshot with label"""
        mem_info = self.get_memory_info()
        mem_info['label'] = label
        self.snapshots.append(mem_info)

        print(f"\n{'='*60}")
        print(f"GPU Memory Snapshot: {label}")
        print(f"{'='*60}")
        print(f"Used:  {mem_info['used_gb']:.2f} GB ({mem_info['utilization_pct']:.1f}%)")
        print(f"Free:  {mem_info['free_gb']:.2f} GB")
        print(f"Total: {mem_info['total_gb']:.2f} GB")
        print(f"{'='*60}\n")

        return mem_info

    def summary(self):
        """Print memory usage summary"""
        if not self.snapshots:
            print("No snapshots taken")
            return

        print("\n" + "="*80)
        print("GPU MEMORY USAGE SUMMARY")
        print("="*80)

        for snap in self.snapshots:
            print(f"{snap['label']:30s} | "
                  f"Used: {snap['used_gb']:6.2f} GB | "
                  f"Free: {snap['free_gb']:6.2f} GB | "
                  f"Util: {snap['utilization_pct']:5.1f}%")

        # Peak usage
        peak = max(self.snapshots, key=lambda x: x['used_gb'])
        print(f"\nPeak usage: {peak['used_gb']:.2f} GB at '{peak['label']}'")
        print("="*80 + "\n")

    @contextmanager
    def track(self, label: str):
        """Context manager to track memory for a code block"""
        self.snapshot(f"{label} (start)")
        torch.cuda.synchronize()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.snapshot(f"{label} (end)")

            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()

# === USAGE EXAMPLE ===

monitor = GPUMemoryMonitor(device_id=0)

# Track model loading
with monitor.track("Load Qwen3-VL"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# Track texture generation
with monitor.track("Generate textures"):
    textures = [generate_texture_array(img) for img in images]

# Track inference
with monitor.track("First inference"):
    result = model.generate(textures[0], "What is this?")

# Track inference WITH ARR-COC
with monitor.track("ARR-COC inference"):
    positions, budgets = arr_coc.allocate(image, query)
    result = model.generate(positions, budgets, query)

# Print summary
monitor.summary()
```

### Checkpoint Manager with LRU Eviction (from real Gradio apps)

```python
# === Production Checkpoint Manager ===

from collections import OrderedDict
from pathlib import Path
import torch
import gc
from typing import Optional, Dict
from contextlib import contextmanager

class CheckpointManager:
    """Memory-efficient checkpoint management with LRU eviction"""

    def __init__(
        self,
        max_loaded: int = 2,
        base_model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    ):
        self.max_loaded = max_loaded
        self.base_model_path = base_model_path

        # Checkpoint registry (path → metadata)
        self.checkpoints: Dict[str, Path] = OrderedDict()

        # Loaded models (LRU cache)
        self.loaded_models: Dict[str, torch.nn.Module] = OrderedDict()

        # Base model (loaded once, never evicted)
        self.base_model = None

    def register(self, name: str, checkpoint_path: Path):
        """Register a checkpoint (doesn't load it)"""
        self.checkpoints[name] = Path(checkpoint_path)
        print(f"✓ Registered checkpoint: {name}")

    def _ensure_base_model(self):
        """Load base model if not already loaded"""
        if self.base_model is None:
            print(f"Loading base model: {self.base_model_path}")

            from transformers import Qwen3VLForConditionalGeneration

            self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            print("✓ Base model loaded")

    def _evict_lru(self):
        """Evict least recently used checkpoint"""
        if len(self.loaded_models) >= self.max_loaded:
            # Pop oldest (LRU)
            lru_name, lru_model = self.loaded_models.popitem(last=False)

            print(f"Evicting LRU checkpoint: {lru_name}")

            # Delete model
            del lru_model

            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def load(self, name: str):
        """Load checkpoint with automatic LRU management"""

        if name not in self.checkpoints:
            raise ValueError(f"Unknown checkpoint: {name}")

        # Load if not in cache
        if name not in self.loaded_models:
            # Ensure space
            self._evict_lru()

            # Load checkpoint
            print(f"Loading checkpoint: {name}")
            checkpoint_path = self.checkpoints[name]

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Ensure base model exists
            self._ensure_base_model()

            # Clone base model (lightweight - shares weights)
            import copy
            model = copy.deepcopy(self.base_model)

            # Load ARR-COC weights only (small!)
            arr_coc_weights = {
                k: v for k, v in checkpoint['model_state_dict'].items()
                if 'arr_coc' in k
            }

            model.load_state_dict(arr_coc_weights, strict=False)

            # Cache it
            self.loaded_models[name] = model
            print(f"✓ Loaded checkpoint: {name}")

        # Move to end (mark as recently used)
        self.loaded_models.move_to_end(name)

        # Yield model
        try:
            yield self.loaded_models[name]
        finally:
            # Cleanup
            torch.cuda.empty_cache()

    def clear_all(self):
        """Clear all loaded checkpoints"""
        self.loaded_models.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ Cleared all checkpoints")

# === USAGE IN GRADIO ===

manager = CheckpointManager(max_loaded=2)

# Register checkpoints (no loading yet)
manager.register('baseline', 'checkpoints/baseline.pt')
manager.register('adaptive', 'checkpoints/adaptive-epoch-5.pt')
manager.register('saliency', 'checkpoints/saliency-epoch-3.pt')

def compare_checkpoints(image, query, selected_checkpoints):
    """Compare multiple checkpoints (memory-efficient)"""
    results = {}

    for name in selected_checkpoints:
        with manager.load(name) as model:
            # Run inference
            output = model.generate(image, query)
            results[name] = output

    return results

# Gradio interface
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(type="pil")
        query = gr.Textbox(label="Query")

    checkpoints = gr.CheckboxGroup(
        choices=['baseline', 'adaptive', 'saliency'],
        label="Select checkpoints",
        value=['baseline', 'adaptive']
    )

    compare_btn = gr.Button("Compare")
    results = gr.JSON(label="Results")

    compare_btn.click(
        fn=compare_checkpoints,
        inputs=[image, query, checkpoints],
        outputs=results
    )

demo.launch()
```

### NaN Detection & Gradient Monitoring (from PyTorch issues)

```python
# === NaN/Inf Detection & Gradient Monitoring ===

import torch
import math

class GradientMonitor:
    """Monitor gradients for NaN, Inf, and explosion"""

    def __init__(self, model, log_every=100):
        self.model = model
        self.log_every = log_every
        self.step = 0
        self.grad_history = []

    def check_step(self):
        """Check gradients after backward pass"""
        self.step += 1

        # Collect gradient stats
        total_norm = 0.0
        num_params = 0
        has_nan = False
        has_inf = False

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"⚠️  NaN gradient detected in {name} at step {self.step}")

                if torch.isinf(param.grad).any():
                    has_inf = True
                    print(f"⚠️  Inf gradient detected in {name} at step {self.step}")

                # Compute norm
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

        total_norm = math.sqrt(total_norm)

        # Log periodically
        if self.step % self.log_every == 0:
            print(f"Step {self.step}: grad_norm={total_norm:.4f}, "
                  f"num_params={num_params}")

        # Check for explosion
        if total_norm > 1000:
            print(f"⚠️  GRADIENT EXPLOSION at step {self.step}!")
            print(f"   Gradient norm: {total_norm:.2f}")
            print(f"   Recommend: reduce learning rate or increase gradient clipping")

        # Store history
        self.grad_history.append({
            'step': self.step,
            'grad_norm': total_norm,
            'has_nan': has_nan,
            'has_inf': has_inf,
        })

        return {
            'grad_norm': total_norm,
            'has_nan': has_nan,
            'has_inf': has_inf,
        }

# === USAGE IN TRAINING LOOP ===

monitor = GradientMonitor(model, log_every=50)

for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)
        loss = compute_loss(outputs, labels)

        # Check loss for NaN
        if torch.isnan(loss):
            print(f"⚠️  NaN LOSS at epoch {epoch}!")
            print(f"   Stopping training - model unstable")
            break

        # Backward pass
        loss.backward()

        # CHECK GRADIENTS (before clipping!)
        grad_stats = monitor.check_step()

        if grad_stats['has_nan'] or grad_stats['has_inf']:
            print("⚠️  Skipping update due to NaN/Inf gradients")
            continue  # Skip this batch

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()
```

---

## III. PERFORMANCE OPTIMIZATION CODE

### DataLoader Optimization (from NERSC training guides)

```python
# === Optimized DataLoader Configuration ===

import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing
from pathlib import Path

class CachedTextureDataset(Dataset):
    """Dataset with pre-computed texture arrays"""

    def __init__(
        self,
        image_paths,
        queries,
        cache_dir="cache/textures",
        num_workers=4,
    ):
        self.image_paths = image_paths
        self.queries = queries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Pre-compute all textures (run once)
        self._build_cache(num_workers=num_workers)

    def _build_cache(self, num_workers):
        """Pre-compute textures with multiprocessing"""
        print(f"Building texture cache ({len(self.image_paths)} images)...")

        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor

        def process_one(args):
            idx, img_path = args
            cache_file = self.cache_dir / f"texture_{idx}.pt"

            if cache_file.exists():
                return None  # Skip

            # Generate texture
            image = load_image(img_path)
            texture = generate_texture_array(image)

            # Save
            torch.save(texture, cache_file)
            return idx

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [(i, p) for i, p in enumerate(self.image_paths)]
            list(tqdm(
                executor.map(process_one, tasks),
                total=len(tasks),
                desc="Caching textures"
            ))

        print(f"✓ Cached {len(self.image_paths)} textures")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load pre-computed texture (FAST!)
        cache_file = self.cache_dir / f"texture_{idx}.pt"
        texture = torch.load(cache_file)

        query = self.queries[idx]

        return texture, query

# === OPTIMAL DATALOADER CONFIGURATION ===

def create_optimized_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
):
    """Create DataLoader with optimal settings"""

    # Rule of thumb: num_workers = min(CPU cores, 4 * num_GPUs)
    num_workers = min(multiprocessing.cpu_count(), 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,

        # CRITICAL OPTIMIZATIONS:
        num_workers=num_workers,       # Parallel loading
        pin_memory=True,               # Faster CPU → GPU transfer
        prefetch_factor=2,             # Prefetch 2 batches ahead
        persistent_workers=True,       # Keep workers alive between epochs

        # Optional:
        drop_last=True,                # Drop incomplete batches
    )

    print(f"✓ DataLoader configured:")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: True")
    print(f"  persistent_workers: True")
    print(f"  prefetch_factor: 2")

    return dataloader

# === USAGE ===

# Create dataset with caching
dataset = CachedTextureDataset(
    image_paths=train_images,
    queries=train_queries,
    cache_dir="cache/train_textures",
    num_workers=8,  # For caching
)

# Create optimized dataloader
train_loader = create_optimized_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
)

# Training loop
for epoch in range(epochs):
    for textures, queries in train_loader:
        # GPU already has data (thanks to pin_memory!)
        textures = textures.to('cuda', non_blocking=True)
        # ... rest of training
```

### torch.compile() Integration (from real repos)

```python
# === torch.compile() for 2-3× Speedup ===

import torch

# IMPORTANT: Requires PyTorch 2.0+ and Ampere GPU or newer!

# Compile model (one-time cost)
print("Compiling model with torch.compile()...")

# ARR-COC components
arr_coc.info_scorer = torch.compile(arr_coc.info_scorer, mode='reduce-overhead')
arr_coc.persp_scorer = torch.compile(arr_coc.persp_scorer, mode='reduce-overhead')
arr_coc.partic_scorer = torch.compile(arr_coc.partic_scorer, mode='reduce-overhead')
arr_coc.balancer = torch.compile(arr_coc.balancer, mode='reduce-overhead')

# Note: Qwen3-VL already compiled by HuggingFace if using recent version

print("✓ Model compiled")
print("  First iteration will be slow (compilation)")
print("  Subsequent iterations: 2-3× faster")

# First forward pass (compilation happens here - will be slow!)
print("Running warmup forward pass (compiling)...")
dummy_input = torch.randn(1, 13, 224, 224, device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    _ = arr_coc.info_scorer(dummy_input)
    _ = arr_coc.persp_scorer(dummy_input)

print("✓ Compilation complete - now ready for fast inference")

# Speedup measurements (from GitHub benchmarks):
# - Without compile: ~50ms/image
# - With compile: ~20ms/image
# - Speedup: 2.5×
```

---

## IV. W&B + GRADIO INTEGRATION

**Production monitoring setup from real apps**

```python
# === W&B Integration with Gradio ===

import wandb
import gradio as gr
from datetime import datetime

class WANDBGradioIntegration:
    """Integrate W&B logging into Gradio app"""

    def __init__(self, project="arr-coc-vis", entity=None):
        # Initialize W&B
        wandb.init(
            project=project,
            entity=entity,
            job_type="demo",
            tags=["gradio", "interactive"],
        )

        # Create W&B table for logging queries
        self.query_table = wandb.Table(columns=[
            "timestamp",
            "image",
            "query",
            "checkpoint",
            "response",
            "relevance_heatmap",
            "num_positions",
            "avg_budget",
            "inference_time_ms",
        ])

    def log_query(
        self,
        image,
        query,
        checkpoint_name,
        response,
        relevance_heatmap,
        positions,
        budgets,
        inference_time,
    ):
        """Log a single query to W&B"""

        # Convert image to W&B format
        wandb_image = wandb.Image(image, caption=query)
        wandb_heatmap = wandb.Image(relevance_heatmap, caption="Relevance")

        # Add row to table
        self.query_table.add_data(
            datetime.now().isoformat(),
            wandb_image,
            query,
            checkpoint_name,
            response[:200],  # Truncate long responses
            wandb_heatmap,
            len(positions),
            float(budgets.mean()),
            inference_time,
        )

        # Log metrics
        wandb.log({
            "query_inference_time_ms": inference_time,
            "num_positions": len(positions),
            "avg_token_budget": float(budgets.mean()),
        })

    def finish(self):
        """Finish W&B run and upload table"""
        wandb.log({"queries": self.query_table})
        wandb.finish()

# === Gradio Interface with W&B Logging ===

# Initialize W&B integration
wb_logger = WANDBGradioIntegration(
    project="arr-coc-vis-demo",
    entity="your-team",
)

def compare_with_logging(image, query, checkpoints):
    """Compare checkpoints AND log to W&B"""
    import time

    results = {}

    for name in checkpoints:
        start = time.time()

        with manager.load(name) as model:
            # Run inference
            output, positions, budgets, heatmap = model.generate_with_viz(
                image, query
            )

            inference_time = (time.time() - start) * 1000  # ms

            # Log to W&B
            wb_logger.log_query(
                image=image,
                query=query,
                checkpoint_name=name,
                response=output,
                relevance_heatmap=heatmap,
                positions=positions,
                budgets=budgets,
                inference_time=inference_time,
            )

            results[name] = {
                'response': output,
                'heatmap': heatmap,
                'stats': {
                    'positions': len(positions),
                    'avg_budget': float(budgets.mean()),
                    'time_ms': inference_time,
                }
            }

    return results

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC-VIS Demo (Logged to W&B)")

    with gr.Row():
        image = gr.Image(type="pil")
        query = gr.Textbox(label="Query")

    checkpoints = gr.CheckboxGroup(
        choices=['baseline', 'adaptive', 'saliency'],
        value=['baseline', 'adaptive']
    )

    compare_btn = gr.Button("Compare (Logs to W&B)")
    results = gr.JSON()

    compare_btn.click(
        fn=compare_with_logging,
        inputs=[image, query, checkpoints],
        outputs=results
    )

    # Session stats
    with gr.Accordion("W&B Dashboard", open=False):
        gr.Markdown(f"""
        View logged queries:
        https://wandb.ai/{wb_logger.wandb.run.entity}/{wb_logger.wandb.run.project}
        """)

demo.launch()

# Cleanup on exit
import atexit
atexit.register(wb_logger.finish)
```

---

## V. ROBUST CHECKPOINTING CODE

**From floydhub/save-and-resume - production patterns**

```python
# === Production Checkpointing with Validation ===

import torch
import os
import shutil
import subprocess
import socket
from datetime import datetime
from pathlib import Path
import random
import numpy as np

def save_checkpoint_robust(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    filepath,
    git_commit=True,
):
    """Save checkpoint with extensive metadata"""

    # Architecture config (for validation)
    model_config = {
        'texture_channels': model.arr_coc.texture.out_channels,
        'num_positions': model.arr_coc.allocator.num_positions,
        'has_adaptive': hasattr(model.arr_coc.balancer, 'policy_net'),
    }

    # Full checkpoint
    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,

        # Training state
        'epoch': epoch,
        'global_step': epoch * len(train_dataloader),
        'metrics': metrics,

        # Architecture metadata (CRITICAL for validation!)
        'model_config': model_config,

        # Reproducibility
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'random_state': {
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },

        # Debugging info
        'timestamp': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
    }

    # Add git commit if requested
    if git_commit:
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode().strip()
            checkpoint['git_commit'] = git_hash
        except:
            checkpoint['git_commit'] = 'unknown'

    # ATOMIC SAVE (prevents corruption)
    temp_path = str(filepath) + '.tmp'
    torch.save(checkpoint, temp_path)

    # Atomic rename (POSIX guarantee)
    os.replace(temp_path, filepath)

    print(f"✓ Saved checkpoint: {filepath}")
    print(f"  Epoch: {epoch}")
    print(f"  Metrics: {metrics}")

    return filepath

def load_checkpoint_safe(
    filepath,
    model,
    optimizer=None,
    scheduler=None,
    strict=True,
):
    """Load checkpoint with validation"""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    print(f"Loading checkpoint: {filepath}")

    # Load
    checkpoint = torch.load(filepath, map_location='cpu')

    # Validate architecture compatibility
    config = checkpoint.get('model_config', {})

    # Check texture channels
    current_channels = model.arr_coc.texture.out_channels
    saved_channels = config.get('texture_channels')

    if saved_channels and saved_channels != current_channels:
        raise ValueError(
            f"Architecture mismatch!\n"
            f"  Checkpoint: {saved_channels} texture channels\n"
            f"  Current:    {current_channels} texture channels\n"
            f"  → Model architectures are incompatible"
        )

    # Load state dicts
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except RuntimeError as e:
        if strict:
            print(f"⚠️  Warning: {e}")
            print("Retrying with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore random state (for reproducibility)
    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state']['torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])

    # Print info
    print(f"✓ Loaded checkpoint")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")
    print(f"  Saved: {checkpoint.get('timestamp', 'unknown')}")
    if 'git_commit' in checkpoint:
        print(f"  Git: {checkpoint['git_commit'][:8]}")

    return checkpoint['epoch'], checkpoint.get('metrics', {})

def validate_checkpoint(filepath):
    """Validate checkpoint can be loaded"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')

        required_keys = ['model_state_dict', 'epoch']
        missing = [k for k in required_keys if k not in checkpoint]

        if missing:
            print(f"❌ Invalid checkpoint: missing keys {missing}")
            return False

        print(f"✓ Valid checkpoint")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
        return True

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

# === AUTO-CLEANUP: Keep only best N checkpoints ===

def cleanup_old_checkpoints(
    checkpoint_dir,
    keep_best=5,
    metric_key='val_accuracy',
    higher_is_better=True,
):
    """Keep only top N checkpoints by metric"""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    # Load all checkpoints
    for ckpt_file in checkpoint_dir.glob("*.pt"):
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu