# W&B Checkpoint Management

Comprehensive guide to managing model checkpoints with Weights & Biases Artifacts for ML training workflows.

## Overview

W&B Artifacts provide a powerful system for versioning, tracking, and managing model checkpoints throughout the training lifecycle. Unlike simple file saving, artifacts offer automatic versioning, metadata tracking, and seamless integration with the W&B ecosystem.

**Core Concept**: Artifacts track data as inputs and outputs of W&B runs. For checkpointing, you log model weights as artifacts with automatic versioning (v0, v1, v2...) and customizable aliases (latest, best, production).

## W&B Artifacts Basics

### What Are Artifacts?

Artifacts are versioned data objects that W&B tracks as inputs/outputs of runs:
- **Datasets**: Training/validation/test data
- **Models**: Checkpoints, trained models
- **Evaluation Results**: Tables, predictions
- **Any File**: Configuration, results, visualizations

From [W&B Artifacts Documentation](https://docs.wandb.ai/models/artifacts) (accessed 2025-01-31):
> Use W&B Artifacts to track and version data as the inputs and outputs of your W&B Runs. For example, a model training run might take in a dataset as input and produce a trained model as output.

### Creating an Artifact

Basic pattern for logging a checkpoint:

```python
import wandb

# Initialize run
run = wandb.init(project="my-project", job_type="train")

# Create artifact
artifact = wandb.Artifact(name="model-checkpoint", type="model")

# Add checkpoint file(s)
artifact.add_file("checkpoint.pth")

# Log artifact (automatically versioned)
run.log_artifact(artifact)
```

**Key Points**:
- First log creates `v0`
- Subsequent logs with same name create `v1`, `v2`, etc. (if content changed)
- W&B checksums content - unchanged files don't create new versions
- `type` parameter organizes artifacts in UI (dataset, model, evaluation)

### Artifact Versioning

From [W&B Artifact Versioning Documentation](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version) (accessed 2025-01-31):

**Automatic Versioning**:
```python
# First run - creates v0
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pth")
run.log_artifact(artifact)  # Saved as "model:v0"

# Second run - creates v1 (if content changed)
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pth")  # Different file
run.log_artifact(artifact)  # Saved as "model:v1"

# Third run - no new version (if content identical to v1)
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pth")  # Same as v1
run.log_artifact(artifact)  # Still points to "model:v1"
```

**Version Aliases**:
- `latest`: Always points to most recent version
- `v0`, `v1`, `v2`: Explicit version numbers
- Custom aliases: `best`, `production`, `staging`

### Artifact Aliases

From [W&B Artifact Alias Documentation](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias/) (accessed 2025-01-31):

Use aliases as pointers to specific versions:

```python
# Log with custom aliases
artifact = wandb.Artifact("model-checkpoint", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-accuracy"])

# Download using alias
run = wandb.init()
artifact = run.use_artifact("model-checkpoint:best-accuracy")
artifact.download()
```

**Common Alias Patterns**:
- `best`: Best performing checkpoint
- `best-val-loss`: Best validation loss
- `best-accuracy`: Highest accuracy
- `epoch-50`: Checkpoint at specific epoch
- `production`: Currently deployed model
- `candidate`: Model being evaluated for deployment

## Checkpoint Saving Patterns

### Pattern 1: Save Every N Steps

Save checkpoints at regular intervals during training:

```python
import wandb
import torch

run = wandb.init(project="training", config={"save_every": 500})

for step in range(total_steps):
    # Training logic
    loss = train_step(model, batch)
    wandb.log({"loss": loss}, step=step)

    # Save checkpoint every N steps
    if step > 0 and step % config.save_every == 0:
        checkpoint_path = f"checkpoint_step_{step}.pth"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        # Log to W&B
        artifact = wandb.Artifact(
            name="model-checkpoint",
            type="model",
            metadata={"step": step, "loss": float(loss)}
        )
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
```

**Pros**:
- Regular snapshots of training progress
- Can resume from any checkpoint
- Good for debugging training instabilities

**Cons**:
- Storage grows quickly
- May save many redundant checkpoints

### Pattern 2: Save Best Model Only

Track validation metrics and save only when performance improves:

```python
import wandb
import torch

run = wandb.init(project="training")
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    train_loss = train_epoch(model, train_loader)

    # Validation
    val_loss = validate(model, val_loader)

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

    # Save if best
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        checkpoint_path = "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

        # Log with "best" alias
        artifact = wandb.Artifact(
            name="model-checkpoint",
            type="model",
            metadata={
                "epoch": epoch,
                "val_loss": float(val_loss),
                "train_loss": float(train_loss)
            }
        )
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact, aliases=["best", "latest"])

        print(f"New best model at epoch {epoch}, val_loss: {val_loss:.4f}")
```

**Pros**:
- Minimal storage usage
- Always have best performing model
- Clean, simple logic

**Cons**:
- No intermediate checkpoints for analysis
- Can't recover from overfitting after the fact

### Pattern 3: Save Best + Latest

Combination strategy: keep best model and latest checkpoint:

```python
import wandb
import torch

run = wandb.init(project="training")
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training and validation
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

    # Always save latest
    latest_path = "latest_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, latest_path)

    artifact_latest = wandb.Artifact(
        name="model-checkpoint",
        type="model",
        metadata={"epoch": epoch, "val_loss": float(val_loss)}
    )
    artifact_latest.add_file(latest_path)
    run.log_artifact(artifact_latest, aliases=["latest"])

    # Save as best if improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_path = "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, best_path)

        artifact_best = wandb.Artifact(
            name="model-checkpoint",
            type="model",
            metadata={"epoch": epoch, "val_loss": float(val_loss), "is_best": True}
        )
        artifact_best.add_file(best_path)
        run.log_artifact(artifact_best, aliases=["best"])
```

**Pros**:
- Best of both worlds
- Can analyze latest training state
- Always have best model

**Cons**:
- Slightly more storage than "best only"
- More logging overhead

### Pattern 4: Save Multiple Metric-Based Checkpoints

Track different metrics for different use cases:

```python
import wandb
import torch

run = wandb.init(project="training")

best_metrics = {
    "val_loss": float('inf'),
    "val_accuracy": 0.0,
    "val_f1": 0.0
}

for epoch in range(num_epochs):
    # Training and validation
    metrics = {
        "train_loss": train_epoch(model, train_loader),
        **validate_all_metrics(model, val_loader)  # Returns dict with val_loss, val_accuracy, val_f1
    }

    wandb.log({**metrics, "epoch": epoch})

    # Check each metric
    for metric_name, metric_value in metrics.items():
        if metric_name.startswith("val_"):
            # For loss: lower is better
            is_better = (metric_name == "val_loss" and metric_value < best_metrics[metric_name]) or \
                       (metric_name != "val_loss" and metric_value > best_metrics[metric_name])

            if is_better:
                best_metrics[metric_name] = metric_value

                checkpoint_path = f"best_{metric_name}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, checkpoint_path)

                artifact = wandb.Artifact(
                    name="model-checkpoint",
                    type="model",
                    metadata={"epoch": epoch, **metrics}
                )
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact, aliases=[f"best-{metric_name}"])

                print(f"New best {metric_name}: {metric_value:.4f} at epoch {epoch}")
```

**Use Case**: When different deployment scenarios need different optimizations (e.g., accuracy vs inference speed).

## Checkpoint Loading Strategies

### Loading Latest Checkpoint

```python
import wandb

run = wandb.init(project="training")

# Download latest checkpoint
artifact = run.use_artifact("model-checkpoint:latest")
artifact_dir = artifact.download()

# Load PyTorch model
checkpoint = torch.load(f"{artifact_dir}/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint.get('epoch', 0) + 1
```

### Loading Specific Version

```python
# Load version v5
artifact = run.use_artifact("model-checkpoint:v5")
artifact_dir = artifact.download()

# Or load by custom alias
artifact = run.use_artifact("model-checkpoint:best-accuracy")
artifact_dir = artifact.download()
```

### Loading from Different Project

```python
# Full artifact path: entity/project/artifact:alias
artifact = run.use_artifact("my-team/my-project/model-checkpoint:best")
artifact_dir = artifact.download()
```

### Loading with Fallback

```python
def load_checkpoint_with_fallback(run, artifact_name, aliases=["best", "latest"]):
    """Try multiple aliases in order"""
    for alias in aliases:
        try:
            artifact = run.use_artifact(f"{artifact_name}:{alias}")
            artifact_dir = artifact.download()
            checkpoint = torch.load(f"{artifact_dir}/checkpoint.pth")
            print(f"Loaded checkpoint from alias: {alias}")
            return checkpoint
        except Exception as e:
            print(f"Failed to load {alias}: {e}")

    raise ValueError(f"Could not load checkpoint with any alias: {aliases}")

# Usage
checkpoint = load_checkpoint_with_fallback(run, "model-checkpoint")
model.load_state_dict(checkpoint['model_state_dict'])
```

## Cleanup and Storage Management

### Problem: Artifact Storage Growth

From [W&B Community Discussion](https://community.wandb.ai/t/how-to-keep-only-last-checkpoint-artifact/3014) (accessed 2025-01-31):

> If I keep multiple checkpoint artifact versions on wandb, they get big really quickly. However, I can't just checkpoint at the end of training.

### Solution 1: Delete Old Versions Programmatically

```python
import wandb

# Get API handle
api = wandb.Api()

# Get artifact
artifact_collection = api.artifact("my-project/model-checkpoint")

# List all versions
versions = artifact_collection.versions()

# Keep only last N versions
keep_count = 5
for i, version in enumerate(versions):
    if i >= keep_count:
        version.delete()
        print(f"Deleted version: {version.name}")
```

### Solution 2: Keep Only Best and Latest

Strategy: Delete intermediate versions, keep only `best` and `latest` aliases:

```python
import wandb

def cleanup_checkpoints(project, artifact_name, keep_aliases=["best", "latest"]):
    """Delete all versions except those with specified aliases"""
    api = wandb.Api()
    artifact_collection = api.artifact(f"{project}/{artifact_name}")

    # Get all versions
    versions = list(artifact_collection.versions())

    for version in versions:
        # Check if version has any protected alias
        has_protected_alias = any(alias in version.aliases for alias in keep_aliases)

        if not has_protected_alias:
            try:
                version.delete()
                print(f"Deleted {version.name}")
            except Exception as e:
                print(f"Failed to delete {version.name}: {e}")

# Usage at end of training
cleanup_checkpoints("my-project", "model-checkpoint", keep_aliases=["best", "latest"])
```

### Solution 3: Automatic Cleanup in Training Loop

```python
import wandb

run = wandb.init(project="training")
api = wandb.Api()

max_checkpoints = 3
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training logic
    val_loss = validate(model, val_loader)

    # Save checkpoint
    artifact = wandb.Artifact("model-checkpoint", type="model")
    artifact.add_file("checkpoint.pth")

    aliases = ["latest"]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        aliases.append("best")

    run.log_artifact(artifact, aliases=aliases)

    # Cleanup old versions (keep only max_checkpoints + best)
    try:
        artifact_collection = api.artifact(f"{run.project}/model-checkpoint")
        versions = list(artifact_collection.versions())

        # Sort by creation time
        versions.sort(key=lambda v: v.created_at)

        # Delete old versions (keep latest max_checkpoints)
        for version in versions[:-max_checkpoints]:
            if "best" not in version.aliases:  # Never delete best
                version.delete()
    except Exception as e:
        print(f"Cleanup failed: {e}")
```

**Note**: Deletion requires API permissions. Set `WANDB_API_KEY` environment variable.

## Integration with HuggingFace Hub

W&B artifacts can complement HuggingFace Hub for model distribution:

```python
import wandb
from transformers import AutoModelForCausalLM

# Save to W&B during training
run = wandb.init(project="llm-training")
artifact = wandb.Artifact("llm-checkpoint", type="model")
model.save_pretrained("./checkpoint")
artifact.add_dir("./checkpoint")
run.log_artifact(artifact, aliases=["best"])

# Later: Download from W&B and push to HF Hub
artifact = run.use_artifact("llm-checkpoint:best")
artifact_dir = artifact.download()

model = AutoModelForCausalLM.from_pretrained(artifact_dir)
model.push_to_hub("my-username/my-llm")
```

**Strategy**:
- Use W&B for training checkpoints (frequent saves)
- Use HF Hub for release models (curated, documented)
- W&B provides versioning during iteration
- HF Hub provides public distribution

## Advanced Patterns

### Distributed Training Checkpoints

From [W&B Distributed Artifacts Documentation](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version) (accessed 2025-01-31):

For multi-GPU or multi-node training:

```python
import wandb
import torch

# Each process has same distributed_id (e.g., run group)
run = wandb.init(project="distributed-training", group="experiment-1")

# Process 0 saves model
if torch.distributed.get_rank() == 0:
    artifact = wandb.Artifact("model-checkpoint", type="model")
    torch.save(model.state_dict(), "model.pth")
    artifact.add_file("model.pth")

    # Use upsert_artifact for collaborative saving
    run.upsert_artifact(artifact, distributed_id=run.group)

# Final process commits the artifact
if is_final_process:
    artifact = wandb.Artifact("model-checkpoint", type="model")
    run.finish_artifact(artifact, distributed_id=run.group)
```

### Checkpoint with Rich Metadata

```python
import wandb
import torch
import json

# Save comprehensive metadata
metadata = {
    "epoch": epoch,
    "step": step,
    "metrics": {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc)
    },
    "hyperparameters": dict(config),
    "model_architecture": str(model),
    "num_parameters": sum(p.numel() for p in model.parameters()),
    "training_time_hours": elapsed_hours,
    "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
}

# Save checkpoint
artifact = wandb.Artifact(
    name="model-checkpoint",
    type="model",
    metadata=metadata,
    description=f"Checkpoint at epoch {epoch} with val_loss {val_loss:.4f}"
)

# Add model file
artifact.add_file("checkpoint.pth")

# Add metadata as JSON
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
artifact.add_file("metadata.json")

# Add training config
artifact.add_file("config.yaml")

run.log_artifact(artifact, aliases=["latest"])
```

### Incremental Checkpointing

For large models where you want to update only changed layers:

```python
import wandb

# Load previous checkpoint artifact
run = wandb.init()
previous_artifact = run.use_artifact("model-checkpoint:latest")

# Create incremental version
draft_artifact = previous_artifact.new_draft()

# Update only changed files
draft_artifact.add_file("updated_layer.pth")  # New file
draft_artifact.remove("old_layer.pth")  # Remove old file

# Log incremental update
run.log_artifact(draft_artifact)
```

**Use Case**: Continual learning, fine-tuning, or updating specific model components without re-uploading entire checkpoint.

## Best Practices

### 1. Use Descriptive Metadata

```python
# Bad: No metadata
artifact = wandb.Artifact("model", type="model")

# Good: Rich metadata
artifact = wandb.Artifact(
    name="resnet50-imagenet",
    type="model",
    metadata={
        "architecture": "ResNet50",
        "dataset": "ImageNet",
        "val_accuracy": 76.5,
        "epoch": 90,
        "training_date": "2025-01-31"
    },
    description="ResNet50 trained on ImageNet, 90 epochs, best val accuracy 76.5%"
)
```

### 2. Use Meaningful Aliases

```python
# Bad: Generic aliases
run.log_artifact(artifact, aliases=["v1", "latest"])

# Good: Semantic aliases
run.log_artifact(artifact, aliases=["best-accuracy", "epoch-90", "latest"])
```

### 3. Include Resume Information

```python
checkpoint = {
    'epoch': epoch,
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict() if use_amp else None,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
    'best_val_loss': best_val_loss,
    'config': dict(config)
}
torch.save(checkpoint, "checkpoint.pth")
```

### 4. Clean Up Local Files

```python
import os

# After logging to W&B, remove local checkpoint
artifact = wandb.Artifact("model-checkpoint", type="model")
artifact.add_file("checkpoint.pth")
run.log_artifact(artifact)

# Clean up local storage
os.remove("checkpoint.pth")
```

### 5. Test Checkpoint Loading

```python
def test_checkpoint_save_load(model, checkpoint_path):
    """Verify checkpoint can be saved and loaded correctly"""
    # Save
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

    # Load into new model
    new_model = create_model()
    checkpoint = torch.load(checkpoint_path)
    new_model.load_state_dict(checkpoint['model_state_dict'])

    # Verify weights match
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), new_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2)

    print("Checkpoint save/load verified successfully")

# Run test before training
test_checkpoint_save_load(model, "test_checkpoint.pth")
os.remove("test_checkpoint.pth")
```

## Common Issues and Solutions

### Issue 1: "Artifact already exists"

**Problem**: Trying to create artifact with existing name.

**Solution**: W&B will version automatically. Use same name for versioning:

```python
# This is correct - W&B handles versioning
artifact = wandb.Artifact("model-checkpoint", type="model")  # Creates v0, v1, v2...
```

### Issue 2: Checkpoints Not Updating

**Problem**: New checkpoints don't create new versions.

**Cause**: File content hasn't changed (W&B checksums files).

**Solution**: Ensure checkpoint actually contains new state:

```python
# Verify different content
import hashlib

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

old_hash = file_hash("checkpoint.pth")
torch.save(model.state_dict(), "checkpoint.pth")
new_hash = file_hash("checkpoint.pth")

if old_hash != new_hash:
    artifact = wandb.Artifact("model-checkpoint", type="model")
    artifact.add_file("checkpoint.pth")
    run.log_artifact(artifact)
else:
    print("Checkpoint content unchanged, skipping upload")
```

### Issue 3: Storage Quota Exceeded

**Problem**: Too many checkpoint versions consuming storage.

**Solutions**:
1. Implement automatic cleanup (see Cleanup section above)
2. Use reference artifacts for large files (link to external storage)
3. Save only essential checkpoints (best + latest)

```python
# Reference artifacts (no upload, just metadata)
artifact = wandb.Artifact("model-checkpoint", type="model")
artifact.add_reference(f"s3://my-bucket/checkpoints/epoch_{epoch}.pth")
run.log_artifact(artifact)
```

## Sources

**W&B Official Documentation:**
- [W&B Artifacts Overview](https://docs.wandb.ai/models/artifacts) - Main artifacts documentation (accessed 2025-01-31)
- [Create Artifact Version](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version) - Versioning and incremental artifacts (accessed 2025-01-31)
- [Create Artifact Alias](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias/) - Alias management (accessed 2025-01-31)
- [Track Experiments Tutorial](https://docs.wandb.ai/models/tutorials/experiments) - PyTorch integration examples (accessed 2025-01-31)

**Community Resources:**
- [W&B Community: Keep Only Last Checkpoint](https://community.wandb.ai/t/how-to-keep-only-last-checkpoint-artifact/3014) - Cleanup strategies (accessed 2025-01-31)
- [W&B Community: Best Model Loading](https://community.wandb.ai/t/easiest-way-to-load-the-best-model-checkpoint-after-training-w-pytorch-lightning/3365) - Loading patterns (accessed 2025-01-31)
- [GitHub Discussion: Keep Latest and Best](https://github.com/Lightning-AI/pytorch-lightning/discussions/9342) - PyTorch Lightning patterns (accessed 2025-01-31)

**Additional References:**
- [W&B Checkpoint Tutorial](https://wandb.ai/morg/llama2-fine-tune/reports/How-to-Checkpoint-Models-on-Weights-Biases--Vmlldzo1NDY5MjUy) - End-to-end checkpoint example (accessed 2025-01-31)
- [PyTorch Lightning W&B Integration](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html) - Lightning-specific patterns (accessed 2025-01-31)
