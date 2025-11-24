# Manual PyTorch W&B Logging

**Manual training loop integration for Weights & Biases with PyTorch**

Manual W&B logging gives you complete control over what gets tracked and when. Unlike HuggingFace Trainer's automatic logging, you write explicit `wandb.log()` calls in your training loop. This is essential when building custom architectures or when you need precise control over metrics.

## Section 1: Training Loop Integration

### Basic Logging Pattern

The core pattern: initialize a run, log metrics in your training loop, and let W&B handle the rest.

From [PyTorch - Weights & Biases Documentation](https://docs.wandb.ai/models/tutorials/pytorch) (accessed 2025-01-31):

**Simple training loop structure:**
```python
import wandb

# Initialize run with config
config = {
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.005,
    "architecture": "CNN"
}

with wandb.init(project="my-project", config=config) as run:
    # Access config through run.config to ensure logged values match execution
    config = run.config

    # Training loop
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            if batch_idx % log_interval == 0:
                run.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
```

**Key principles:**
- Use `with wandb.init()` context manager for automatic cleanup
- Access hyperparameters through `run.config` to ensure consistency
- Log metrics as dictionaries with string keys
- W&B automatically increments step counter on each `log()` call

### Step vs Epoch Tracking

From [Overview - Weights & Biases Documentation](https://docs.wandb.ai/models/track/log) (accessed 2025-01-31):

**Default behavior: automatic step increment**
```python
# Each wandb.log() call creates a new step (0, 1, 2, ...)
run.log({"loss": 0.5})  # step=0
run.log({"loss": 0.4})  # step=1
run.log({"loss": 0.3})  # step=2
```

**Manual step control:**
```python
# Use 'step' parameter for explicit control
global_step = 0
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)

        run.log({"loss": loss}, step=global_step)
        global_step += 1
```

**Epoch-based logging:**
```python
# Log once per epoch
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        loss = train_step(batch)
        epoch_loss += loss

    avg_loss = epoch_loss / len(train_loader)
    run.log({"epoch_loss": avg_loss, "epoch": epoch}, step=epoch)
```

From [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31):

**Best practice: use example count as step**
```python
example_ct = 0  # number of examples seen
batch_ct = 0

for epoch in range(config.epochs):
    for images, labels in train_loader:
        loss = train_batch(images, labels, model, optimizer, criterion)
        example_ct += len(images)
        batch_ct += 1

        # Log every 25 batches
        if (batch_ct % 25) == 0:
            run.log({
                "loss": loss,
                "epoch": epoch,
                "examples_seen": example_ct
            }, step=example_ct)
```

**Why example count?** Makes comparisons across different batch sizes meaningful. A model trained with batch_size=32 and batch_size=128 will have different numbers of steps, but the same number of examples at equivalent points in training.

### Learning Rate Schedules

**Logging learning rate changes:**
```python
from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)
        optimizer.step()

    # Log learning rate after scheduler step
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    run.log({
        "learning_rate": current_lr,
        "epoch": epoch
    })
```

**Multiple parameter groups:**
```python
# Log different learning rates for different parts of model
for epoch in range(num_epochs):
    # ... training ...

    run.log({
        "lr_backbone": optimizer.param_groups[0]['lr'],
        "lr_classifier": optimizer.param_groups[1]['lr'],
        "epoch": epoch
    })
```

### Gradient Clipping

**Log gradient norms before and after clipping:**
```python
import torch.nn.utils as nn_utils

for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # Compute gradient norm before clipping
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5

    # Clip gradients
    max_norm = 1.0
    nn_utils.clip_grad_norm_(model.parameters(), max_norm)

    # Compute gradient norm after clipping
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5

    optimizer.step()

    run.log({
        "grad_norm_before_clip": total_norm_before,
        "grad_norm_after_clip": total_norm_after,
        "grad_clipped": total_norm_before > max_norm
    })
```

## Section 2: Advanced Logging

### wandb.watch() for Model Weights

From [PyTorch - Weights & Biases Documentation](https://docs.wandb.ai/models/tutorials/pytorch) (accessed 2025-01-31):

**Track gradients and parameters automatically:**
```python
model = ConvNet()
criterion = nn.CrossEntropyLoss()

with wandb.init(project="my-project") as run:
    # Watch model: logs gradients and parameters
    run.watch(model, criterion, log="all", log_freq=10)

    # Training loop - gradients logged automatically
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = train_step(batch)
```

**log parameter options:**
- `"gradients"`: Log only gradients
- `"parameters"`: Log only parameter values
- `"all"`: Log both gradients and parameters
- `None`: Don't log anything (useful to disable temporarily)

**log_freq parameter:** How often to log (every N steps)
```python
# Log every 100 steps (reduces overhead for large models)
run.watch(model, log="all", log_freq=100)
```

### Custom Metrics Computation

**Per-class accuracy:**
```python
def compute_per_class_accuracy(outputs, targets, num_classes):
    _, preds = torch.max(outputs, 1)

    per_class_acc = {}
    for cls in range(num_classes):
        mask = (targets == cls)
        if mask.sum() > 0:
            correct = (preds[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
            per_class_acc[f"accuracy_class_{cls}"] = correct / total

    return per_class_acc

# In training loop
for batch_idx, (data, target) in enumerate(val_loader):
    output = model(data)

    # Compute per-class metrics
    metrics = compute_per_class_accuracy(output, target, num_classes=10)

    run.log({
        "val_step": batch_idx,
        **metrics  # Unpack dictionary
    })
```

**Running averages:**
```python
from collections import defaultdict

class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metric_dict):
        for k, v in metric_dict.items():
            self.metrics[k].append(v)

    def get_averages(self):
        return {k: sum(v) / len(v) for k, v in self.metrics.items()}

    def reset(self):
        self.metrics.clear()

# Usage
tracker = MetricTracker()

for epoch in range(num_epochs):
    tracker.reset()

    for batch in train_loader:
        loss = train_step(batch)
        tracker.update({"batch_loss": loss})

    # Log epoch averages
    run.log({
        "epoch": epoch,
        **tracker.get_averages()
    })
```

### Multiple Validation Sets

**Logging different validation splits:**
```python
# Separate validation sets
val_loaders = {
    "val_seen": seen_val_loader,
    "val_unseen": unseen_val_loader,
    "val_ood": out_of_distribution_loader
}

for epoch in range(num_epochs):
    # Training
    train_epoch(model, train_loader, optimizer)

    # Validation on multiple sets
    val_metrics = {}
    for split_name, val_loader in val_loaders.items():
        acc, loss = evaluate(model, val_loader)
        val_metrics[f"{split_name}/accuracy"] = acc
        val_metrics[f"{split_name}/loss"] = loss

    run.log({
        "epoch": epoch,
        **val_metrics
    })
```

**Result in W&B:** Each validation set gets its own line in the dashboard, grouped by prefix (`val_seen/`, `val_unseen/`, etc.)

### System Metrics (GPU, Memory)

From [W&B Integration Best Practices](https://wandb.ai/wandb-smle/integration_best_practices/reports/W-B-Integration-Best-Practices--VmlldzoyMzc5MTI2) (accessed 2025-01-31):

**W&B automatically logs:**
- CPU utilization
- GPU utilization (via nvidia-smi)
- System memory
- GPU memory
- Disk I/O
- Network I/O

**Manual logging for custom system metrics:**
```python
import psutil
import torch

def get_system_metrics():
    metrics = {}

    # CPU
    metrics["cpu_percent"] = psutil.cpu_percent()

    # Memory
    mem = psutil.virtual_memory()
    metrics["memory_used_gb"] = mem.used / (1024**3)
    metrics["memory_percent"] = mem.percent

    # GPU (if available)
    if torch.cuda.is_available():
        metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

    return metrics

# Log periodically
if global_step % 100 == 0:
    run.log({
        **training_metrics,
        **get_system_metrics()
    })
```

## Section 3: Visualization

### Custom Charts and Plots

From [Custom charts overview - Weights & Biases Documentation](https://docs.wandb.ai/models/app/features/custom-charts) (accessed 2025-01-31):

**wandb.plot for built-in visualizations:**
```python
import wandb

# Line plot
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
run.log({
    "custom_line_plot": wandb.plot.line(
        table, "x", "y",
        title="Custom Y vs X Line Plot"
    )
})

# Scatter plot
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
run.log({
    "custom_scatter": wandb.plot.scatter(
        table, "x", "y",
        title="Height vs Weight"
    )
})

# Bar chart
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
run.log({
    "custom_bar_chart": wandb.plot.bar(
        table, "label", "value",
        title="Metric by Category"
    )
})

# Histogram
data = [[val] for val in values]
table = wandb.Table(data=data, columns=["value"])
run.log({
    "custom_histogram": wandb.plot.histogram(
        table, "value",
        title="Distribution of Values"
    )
})
```

### Image Logging (wandb.Image)

**Single images:**
```python
import wandb
from PIL import Image

# From PIL Image
img = Image.open("example.png")
run.log({"example": wandb.Image(img, caption="Example image")})

# From numpy array
import numpy as np
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
run.log({"random_image": wandb.Image(img_array)})

# From PyTorch tensor
import torch
img_tensor = torch.rand(3, 224, 224)  # CHW format
run.log({"tensor_image": wandb.Image(img_tensor)})
```

**Multiple images:**
```python
# Log multiple images in a grid
images = []
for i in range(10):
    img = generate_image(i)
    images.append(wandb.Image(img, caption=f"Image {i}"))

run.log({"examples": images})
```

**Images with bounding boxes (object detection):**
```python
# Bounding box format: {"position": {x, y, width, height}, "class_id": int}
boxes = [
    {
        "position": {"minX": 10, "minY": 20, "maxX": 100, "maxY": 80},
        "class_id": 1,
        "box_caption": "dog",
        "scores": {"confidence": 0.95}
    }
]

class_labels = {1: "dog", 2: "cat", 3: "bird"}

run.log({
    "predictions": wandb.Image(
        img,
        boxes=boxes,
        classes=class_labels
    )
})
```

**Segmentation masks:**
```python
# Mask should be HxW array with integer class IDs
mask = np.zeros((224, 224), dtype=np.uint8)
mask[50:150, 50:150] = 1  # Class 1
mask[100:200, 100:200] = 2  # Class 2

class_labels = {0: "background", 1: "dog", 2: "cat"}

run.log({
    "segmentation": wandb.Image(
        img,
        masks={
            "predictions": {
                "mask_data": mask,
                "class_labels": class_labels
            }
        }
    )
})
```

### Tables for Structured Data

From [Custom charts overview](https://docs.wandb.ai/models/app/features/custom-charts) (accessed 2025-01-31):

**Basic tables:**
```python
# Create table with columns
columns = ["id", "prediction", "target", "loss"]
data = [
    [0, 7, 7, 0.0],
    [1, 3, 8, 2.5],
    [2, 5, 5, 0.0]
]

table = wandb.Table(data=data, columns=columns)
run.log({"predictions_table": table})
```

**Tables with images:**
```python
# Table combining images and metrics
columns = ["image", "prediction", "target", "correct"]
data = []

for img, pred, target in zip(images, predictions, targets):
    data.append([
        wandb.Image(img),
        pred,
        target,
        pred == target
    ])

table = wandb.Table(data=data, columns=columns)
run.log({"validation_results": table})
```

**Incremental table logging:**
```python
# Add rows one at a time (useful for large tables)
table = wandb.Table(columns=["step", "metric_a", "metric_b"])

for step in range(1000):
    metric_a = compute_metric_a(step)
    metric_b = compute_metric_b(step)
    table.add_data(step, metric_a, metric_b)

run.log({"metrics_table": table})
```

**Note:** Maximum table size is 10,000 rows. For larger datasets, log incrementally or sample.

### Media Logging (Audio, Video)

**Audio logging:**
```python
import numpy as np

# Generate audio (sample_rate, numpy array)
sample_rate = 44100
audio_data = np.random.randn(sample_rate * 2)  # 2 seconds

run.log({
    "audio_sample": wandb.Audio(
        audio_data,
        sample_rate=sample_rate,
        caption="Generated audio"
    )
})

# From file
run.log({
    "audio_file": wandb.Audio("example.wav")
})
```

**Video logging:**
```python
import numpy as np

# Video from numpy array (T, H, W, C)
video_data = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)

run.log({
    "video": wandb.Video(
        video_data,
        fps=30,
        caption="Training visualization"
    )
})

# From file
run.log({
    "video_file": wandb.Video("example.mp4")
})
```

## Sources

**Web Research:**
- [PyTorch - Weights & Biases Documentation](https://docs.wandb.ai/models/tutorials/pytorch) (accessed 2025-01-31) - Official PyTorch integration guide with training loop examples
- [Overview - Weights & Biases Documentation](https://docs.wandb.ai/models/track/log) (accessed 2025-01-31) - Core logging concepts, step management, and best practices
- [W&B Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) (accessed 2025-01-31) - Step parameter usage and logging frequency recommendations
- [Custom charts overview - Weights & Biases Documentation](https://docs.wandb.ai/models/app/features/custom-charts) (accessed 2025-01-31) - Custom visualizations, tables, and Vega charts
- [W&B Integration Best Practices](https://wandb.ai/wandb-smle/integration_best_practices/reports/W-B-Integration-Best-Practices--VmlldzoyMzc5MTI2) (accessed 2025-01-31) - Step management and integration patterns

**Additional References:**
- [Foundation Model Builder Tips & Best Practices](https://wandb.ai/wandb/report/reports/Foundation-Model-Builder-Tips-Best-Practices--Vmlldzo5MTA3MDI2) - Long training run monitoring
- [How To Write Efficient Training Loops in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-To-Write-Efficient-Training-Loops-in-PyTorch--VmlldzoyMjg4OTk5) - Memory and compute efficiency
