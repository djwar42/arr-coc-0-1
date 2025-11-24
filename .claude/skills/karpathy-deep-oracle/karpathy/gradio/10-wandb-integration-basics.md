# Weights & Biases Integration Basics

## Overview

Weights & Biases (W&B) is an MLOps platform for tracking, visualizing, and managing machine learning experiments. This guide covers the fundamental concepts and patterns for integrating W&B into your ML workflows.

**Core Philosophy:** "Don't tell me it works, show me the loss curve" - Track everything, visualize immediately, share results.

---

## Section 1: W&B Quickstart (~80 lines)

### Installation and Authentication

From [W&B Quickstart](https://docs.wandb.ai/models/quickstart) (accessed 2025-01-31):

**Install the library:**
```bash
pip install wandb
```

**Authentication methods:**

1. **Command Line (recommended for servers):**
```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

2. **Python:**
```python
import wandb
wandb.login()  # Interactive prompt for API key
```

3. **Notebook:**
```python
import wandb
wandb.login()  # Will prompt in cell output
```

**Get your API key:**
- Visit https://wandb.ai/authorize
- Copy your API key
- Store it securely (environment variable or keychain)

### Basic wandb.init() and wandb.log()

From [W&B Quickstart](https://docs.wandb.ai/models/quickstart) (accessed 2025-01-31):

**Initialize a run:**
```python
import wandb

run = wandb.init(
    project="my-awesome-project",  # Project name
    config={                        # Hyperparameters
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 32,
    },
)
```

**Log metrics during training:**
```python
# Inside training loop
for epoch in range(config['epochs']):
    # ... training code ...
    loss = train_one_epoch()
    acc = evaluate()

    # Log metrics to W&B
    wandb.log({
        "loss": loss,
        "accuracy": acc,
        "epoch": epoch
    })

# Finish the run
wandb.finish()
```

**Complete minimal example:**
```python
import wandb
import random

wandb.login()

config = {'epochs': 10, 'lr': 0.01}

with wandb.init(project="my-project", config=config) as run:
    offset = random.random() / 5

    for epoch in range(2, config['epochs']):
        # Simulate training
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        run.log({"accuracy": acc, "loss": loss})
```

### Project Structure

From [W&B Projects Documentation](https://docs.wandb.ai/models/track/project-page) (accessed 2025-01-31):

**Projects organize related experiments:**
- Compare different model architectures
- Track hyperparameter variations
- Share results with team members
- Create reports for findings

**Project creation:**
```python
# Automatic: project created on first run
wandb.init(project="new-project-name")

# Or create via W&B App UI
# Navigate to: https://wandb.ai/<team-name>
# Click "Create new project"
```

**Project visibility options:**
- **Team** (default): Only team members can access
- **Restricted**: Invite-only access
- **Open**: Anyone can submit runs (classrooms/competitions)
- **Public**: Anyone can view, only team can edit

---

## Section 2: Core Concepts (~100 lines)

### Projects, Runs, and Organization

From [W&B Projects Documentation](https://docs.wandb.ai/models/track/project-page) (accessed 2025-01-31):

**Hierarchy:**
```
Organization
  └── Team
       └── Project
            └── Run (individual experiment)
                 └── Metrics, artifacts, logs
```

**Project tabs:**
- **Overview**: Project metadata, stats, contributors
- **Workspace**: Interactive visualization sandbox
- **Runs**: Table of all experiments
- **Sweeps**: Hyperparameter optimization
- **Artifacts**: Datasets, models, results
- **Reports**: Saved snapshots with notes

**Run naming:**
```python
# Auto-generated name (e.g., "ancient-plasma-42")
wandb.init(project="my-project")

# Custom name
wandb.init(project="my-project", name="baseline-resnet50")

# With tags
wandb.init(
    project="my-project",
    tags=["baseline", "resnet", "imagenet"]
)
```

### Config vs Logs vs Summary

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**wandb.config - Hyperparameters and settings:**
```python
# Set at initialization
wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.001,
        "architecture": "ResNet50",
        "dataset": "ImageNet",
        "epochs": 100,
    }
)

# Update during run
wandb.config.update({"dropout": 0.2})

# Access values
lr = wandb.config.learning_rate
```

**wandb.log() - Time-series metrics:**
```python
# Log at each step/epoch
for step in range(1000):
    wandb.log({
        "loss": loss_value,
        "accuracy": acc_value,
        "step": step  # Optional explicit step
    })

# W&B tracks history automatically
# Creates line plots over time
```

**wandb.summary - Final metrics:**
```python
# Automatically set to last logged value
wandb.log({"val_accuracy": 0.95})  # Becomes summary value

# Manually set summary
wandb.summary["best_accuracy"] = 0.97
wandb.summary["final_loss"] = 0.03
```

**Key differences:**
- **config**: Static, set once, defines experiment
- **log**: Dynamic, called repeatedly, tracks progress
- **summary**: Final results, appears in runs table

### Tags and Filtering

From [W&B Projects Documentation](https://docs.wandb.ai/models/track/project-page) (accessed 2025-01-31):

**Using tags for organization:**
```python
# Add tags at initialization
wandb.init(
    project="my-project",
    tags=["experiment-v1", "baseline", "debugging"]
)

# Search/filter in UI:
# - Click "Filter" in runs sidebar
# - Use tag:<tag-name> in search
# - Group by tags in workspace
```

**Common tagging strategies:**
- Experiment phase: `["baseline", "tuning", "final"]`
- Model type: `["resnet", "transformer", "custom"]`
- Data version: `["data-v1", "augmented", "clean"]`
- Status: `["debugging", "production", "ablation"]`

---

## Section 3: Common Patterns (~120 lines)

### Training Loop Integration

From [W&B Quickstart](https://docs.wandb.ai/models/quickstart) and [Best Practices](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**Basic PyTorch pattern:**
```python
import wandb
import torch

# Initialize
wandb.init(
    project="pytorch-training",
    config={
        "learning_rate": 1e-3,
        "epochs": 10,
        "batch_size": 32,
    }
)

# Training loop
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

for epoch in range(wandb.config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log every N batches
        if batch_idx % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
                "train/batch": batch_idx,
            })

    # Validation at end of epoch
    val_loss, val_acc = validate(model, val_loader)
    wandb.log({
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
    })

wandb.finish()
```

**Nested metric naming (recommended):**
```python
# Use "/" for hierarchical organization
wandb.log({
    "train/loss": 0.5,
    "train/accuracy": 0.85,
    "val/loss": 0.6,
    "val/accuracy": 0.83,
    "lr": current_lr,
})

# Creates grouped panels in UI:
# - train/* metrics together
# - val/* metrics together
```

### Epoch vs Step Logging

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**Step-based logging (default):**
```python
# W&B auto-increments step counter
for step in range(1000):
    loss = train_step()
    wandb.log({"loss": loss})  # Step 0, 1, 2, ...
```

**Explicit step control:**
```python
# Specify step manually
global_step = 0
for epoch in range(10):
    for batch in dataloader:
        loss = train_batch(batch)
        wandb.log({"loss": loss}, step=global_step)
        global_step += 1
```

**Commit parameter (advanced):**
```python
# Log multiple metrics at same step
wandb.log({"train_loss": 0.5}, commit=False)
wandb.log({"train_acc": 0.85}, commit=False)
wandb.log({"val_loss": 0.6}, commit=True)  # All logged together
```

**Best practice - separate train/val steps:**
```python
train_step = 0
for epoch in range(epochs):
    # Training
    for batch in train_loader:
        loss = train(batch)
        wandb.log({"train/loss": loss}, step=train_step)
        train_step += 1

    # Validation (once per epoch)
    val_metrics = validate()
    wandb.log({
        "val/loss": val_metrics['loss'],
        "val/accuracy": val_metrics['acc'],
        "epoch": epoch,
    })
```

### wandb.watch() for Gradients

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**Track model gradients and parameters:**
```python
import wandb

wandb.init(project="gradient-tracking")
model = MyModel()

# Watch model (call once after init)
wandb.watch(
    model,
    log="all",        # "gradients", "parameters", "all", or None
    log_freq=100,     # Log every N batches
)

# Training loop - gradients logged automatically
for batch in dataloader:
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()
    wandb.log({"loss": loss.item()})
```

**Watch options:**
- `log="gradients"`: Track gradient distributions
- `log="parameters"`: Track weight distributions
- `log="all"`: Track both (can be expensive)
- `log_freq=N`: How often to log (balance detail vs performance)

**When to use wandb.watch():**
- Debugging gradient issues (vanishing/exploding)
- Monitoring layer-wise learning
- Detecting dead neurons
- **Warning**: High memory overhead for large models

### Best Practices

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**1. Use context managers:**
```python
# Automatically calls wandb.finish()
with wandb.init(project="my-project", config=config) as run:
    train(run)
    # No need to call wandb.finish()
```

**2. Log hyperparameters in config:**
```python
# Good - searchable, comparable
wandb.init(config={"lr": 0.01, "batch_size": 32})

# Bad - not tracked
lr = 0.01  # Lost information
```

**3. Meaningful project/run names:**
```python
# Good
wandb.init(
    project="image-classification-resnet",
    name="resnet50-imagenet-baseline",
    tags=["baseline", "resnet50"]
)

# Bad
wandb.init(project="test123", name="run1")
```

**4. Group related metrics:**
```python
# Use prefixes for organization
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "system/gpu_memory": gpu_mem,
})
```

**5. Log at appropriate frequency:**
```python
# Too frequent - slow, cluttered
for step in range(100000):
    wandb.log({"loss": loss})  # 100k points

# Good - every N steps
if step % 10 == 0:
    wandb.log({"loss": loss})  # 10k points
```

**6. Always call wandb.finish():**
```python
# Ensures data is uploaded
try:
    wandb.init(...)
    train()
finally:
    wandb.finish()  # Even if training crashes
```

**7. Use wandb offline mode for debugging:**
```bash
# Set before running script
export WANDB_MODE=offline

# Or in Python
wandb.init(mode="offline")
```

---

## Sources

**Official Documentation:**
- [W&B Quickstart](https://docs.wandb.ai/models/quickstart) - Installation, basic usage, minimal examples (accessed 2025-01-31)
- [W&B Projects Documentation](https://docs.wandb.ai/models/track/project-page) - Project organization, tabs, visibility (accessed 2025-01-31)
- [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) - Training patterns, logging strategies (accessed 2025-01-31)

**Additional References:**
- [PyPI wandb package](https://pypi.org/project/wandb/) - Installation and versioning
- [W&B GitHub repository](https://github.com/wandb/wandb) - Source code and examples

**Next Steps:**
- See [11-wandb-huggingface-trainer.md](11-wandb-huggingface-trainer.md) for HuggingFace integration
- See [12-wandb-pytorch-manual.md](12-wandb-pytorch-manual.md) for advanced PyTorch patterns
- See [13-wandb-vlm-metrics.md](13-wandb-vlm-metrics.md) for vision-language model specific metrics
