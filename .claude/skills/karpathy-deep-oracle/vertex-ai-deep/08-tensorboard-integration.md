# Vertex AI TensorBoard Integration

## Overview

Vertex AI TensorBoard is Google Cloud's managed, enterprise-ready version of the open-source TensorBoard visualization toolkit. It provides scalable, team-shareable experiment tracking and visualization for machine learning workflows on Vertex AI.

**Key Capabilities:**
- Managed infrastructure (no self-hosting required)
- Automatic experiment organization and tracking
- Multi-run comparison across experiments
- Cloud Profiler integration for performance analysis
- Team collaboration with IAM-based access control
- Persistent storage in Cloud Storage
- Integration with Vertex AI Pipelines and Custom Training

**From** [Introduction to Vertex AI TensorBoard](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) (accessed 2025-02-03):
> "Vertex AI TensorBoard is an enterprise-ready managed version of Open Source TensorBoard, which is a Google Open Source project for machine learning experiment tracking and visualization."

**From** [Ray on Vertex AI Blog](https://cloud.google.com/blog/products/ai-machine-learning/ray-on-vertex-ai) (accessed 2025-02-03):
> "Vertex AI TensorBoard provides a managed TensorBoard service that enables you to track, visualize, compare your tuning jobs, and collaborate with your team."

---

## 1. Managed TensorBoard Setup

### Architecture

Vertex AI TensorBoard operates as a fully managed service with:
- **TensorBoard Instance**: Regional endpoint for accessing UI
- **Experiment Storage**: GCS buckets storing event files
- **Metadata Service**: Vertex AI Metadata tracks relationships
- **Access Control**: IAM permissions for team collaboration
- **Automatic Scaling**: Handles concurrent users seamlessly

### Creating TensorBoard Instances

**Using gcloud CLI:**
```bash
# Create TensorBoard instance
gcloud ai tensorboards create \
  --display-name="my-training-experiments" \
  --region=us-central1 \
  --description="PyTorch model training experiments"

# Get TensorBoard ID
TENSORBOARD_ID=$(gcloud ai tensorboards list \
  --region=us-central1 \
  --filter="display_name:my-training-experiments" \
  --format="value(name)")

echo $TENSORBOARD_ID
# Output: projects/123456/locations/us-central1/tensorboards/987654
```

**Using Python SDK:**
```python
from google.cloud import aiplatform

aiplatform.init(
    project="my-project",
    location="us-central1"
)

# Create TensorBoard
tensorboard = aiplatform.Tensorboard.create(
    display_name="my-training-experiments",
    description="PyTorch model training experiments",
    labels={"team": "ml-research", "project": "vision-transformer"}
)

print(f"TensorBoard resource name: {tensorboard.resource_name}")
print(f"TensorBoard web URL: {tensorboard.web_url}")
```

### Service Account Setup for TensorBoard

**Critical**: Custom training jobs require service accounts with TensorBoard permissions.

**From** [Use Vertex AI TensorBoard with Custom Training](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-training) (accessed 2025-02-03):

```bash
# Create service account
SA_NAME="tensorboard-sa"
gcloud iam service-accounts create $SA_NAME \
  --display-name="Vertex AI TensorBoard Service Account"

# Grant TensorBoard write permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Grant Cloud Storage permissions (for event files)
gsutil iam ch \
  serviceAccount:${SA_NAME}@PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://TENSORBOARD_BUCKET
```

### Automatic Upload from Custom Training

**Configure training job with TensorBoard:**
```python
from google.cloud import aiplatform

# Define custom job with TensorBoard
job = aiplatform.CustomTrainingJob(
    display_name="pytorch-vision-transformer",
    container_uri="gcr.io/my-project/pytorch-trainer:latest",
    tensorboard=tensorboard.resource_name  # Link to TensorBoard instance
)

# Launch job (logs automatically upload to TensorBoard)
job.run(
    service_account=f"{SA_NAME}@PROJECT_ID.iam.gserviceaccount.com",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--model=vit-base",
        "--batch-size=64",
        "--learning-rate=1e-4"
    ]
)
```

**Training script (PyTorch example):**
```python
from torch.utils.tensorboard import SummaryWriter
import os

# Vertex AI automatically sets AIP_TENSORBOARD_LOG_DIR
log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "./logs")
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        loss = train_step(data, target)

        # Log to TensorBoard (automatically uploaded to Vertex AI)
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("loss/train", loss, global_step)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step)

    # Validation metrics
    val_acc, val_loss = validate()
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("accuracy/val", val_acc, epoch)

writer.close()
```

### TensorBoard Web Access

**From** [View Vertex AI TensorBoard Data](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-view) (accessed 2025-02-03):

Access TensorBoard UI via Cloud Console or direct URL:
```python
# Get TensorBoard web URL
tensorboard = aiplatform.Tensorboard(tensorboard_id)
print(tensorboard.web_url)

# Output: https://us-central1-tensorboard.googleusercontent.com/experiment/...
```

**IAM Permissions for Access:**
- `roles/aiplatform.viewer` - Read-only access to TensorBoard
- `roles/aiplatform.user` - Read/write access for uploading data
- `roles/aiplatform.admin` - Full management permissions

---

## 2. Custom Metrics Visualization

### Scalars Dashboard

The Scalars dashboard tracks numeric metrics over training steps/epochs.

**Standard Patterns:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir)

# Training metrics (logged per batch)
writer.add_scalar("loss/train", train_loss, global_step)
writer.add_scalar("loss/val", val_loss, epoch)

# Performance metrics
writer.add_scalar("accuracy/train", train_acc, global_step)
writer.add_scalar("accuracy/val", val_acc, epoch)
writer.add_scalar("accuracy/test", test_acc, epoch)

# Learning dynamics
writer.add_scalar("learning_rate", lr, global_step)
writer.add_scalar("gradient_norm", grad_norm, global_step)

# Hardware utilization
writer.add_scalar("gpu/memory_used_mb", gpu_mem_mb, global_step)
writer.add_scalar("gpu/utilization_pct", gpu_util, global_step)
```

**Custom Scalar Layouts:**

**From** [TensorBoard Scalars and Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras) (accessed 2025-02-03):

Create custom layouts for organized metric views:
```python
from tensorboard.plugins.custom_scalar import layout_pb2
from google.protobuf import json_format

# Define custom layout
layout_summary = layout_pb2.Layout(
    category=[
        layout_pb2.Category(
            title='Loss Metrics',
            chart=[
                layout_pb2.Chart(
                    title='Training vs Validation Loss',
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r'loss/train', r'loss/val']
                    )
                )
            ]
        ),
        layout_pb2.Category(
            title='Performance',
            chart=[
                layout_pb2.Chart(
                    title='Accuracy Comparison',
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r'accuracy/.*']  # Regex pattern
                    )
                )
            ]
        )
    ]
)

# Write layout to TensorBoard
writer.add_custom_scalar_layout(layout_summary)
```

### Histograms and Distributions

Track parameter and activation distributions:
```python
# Model parameter distributions
for name, param in model.named_parameters():
    writer.add_histogram(f"params/{name}", param, epoch)
    if param.grad is not None:
        writer.add_histogram(f"grads/{name}", param.grad, epoch)

# Activation distributions
def hook_fn(module, input, output):
    writer.add_histogram(f"activations/{module.__class__.__name__}",
                         output, global_step)

# Register hooks
for name, layer in model.named_modules():
    if isinstance(layer, (torch.nn.ReLU, torch.nn.GELU)):
        layer.register_forward_hook(hook_fn)
```

### Images and Media

Visualize input data, predictions, and model outputs:
```python
# Input images
writer.add_images("inputs/batch", images, global_step)

# Predictions vs ground truth
writer.add_image("predictions/sample_0", pred_image, global_step)
writer.add_image("ground_truth/sample_0", gt_image, global_step)

# Attention maps (for vision transformers)
writer.add_image("attention/head_0_layer_5", attention_map, global_step)

# Video logging (for video models)
writer.add_video("predictions/video_sample", video_tensor, global_step, fps=30)

# Text logging
writer.add_text("hyperparameters", str(hparams), 0)
writer.add_text("model_architecture", str(model), 0)
```

### Custom Plots with Matplotlib

Create complex visualizations:
```python
import matplotlib.pyplot as plt
import io
from PIL import Image

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax)

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)

    return transforms.ToTensor()(image)

# Log custom plot
cm_image = plot_confusion_matrix(confusion_matrix, class_names)
writer.add_image("eval/confusion_matrix", cm_image, epoch)
```

---

## 3. Cloud Profiler Integration

### Enabling Profiler in Training

**From** [Enable Cloud Profiler for Debugging Model Training Performance](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-02-03):

Cloud Profiler analyzes training performance bottlenecks.

**Setup in Training Script (PyTorch):**
```python
import torch.profiler

# Enable profiling with TensorBoard integration
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,      # Skip first batch
        warmup=1,    # Warmup for 1 batch
        active=3,    # Profile 3 batches
        repeat=2     # Repeat profiling cycle
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    record_shapes=True,        # Record tensor shapes
    profile_memory=True,       # Track memory usage
    with_stack=True            # Record call stacks
) as prof:
    for step, (data, target) in enumerate(train_loader):
        # Training step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()  # Signal profiler to advance

        if step >= (1 + 1 + 3) * 2:  # Profile complete
            break
```

**TensorFlow Profiler:**
```python
import tensorflow as tf

# Callback for profiling
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    profile_batch='10,20',  # Profile batches 10-20
    update_freq='epoch'
)

model.fit(
    train_dataset,
    epochs=num_epochs,
    callbacks=[tensorboard_callback],
    validation_data=val_dataset
)
```

### Analyzing Profiler Data

**From** [5 Ways to Optimize Training Performance with TensorFlow Profiler](https://cloud.google.com/blog/topics/developers-practitioners/how-optimize-training-performance-tensorflow-profiler-vertex-ai) (accessed 2025-02-03):

**Key Profiler Views:**

1. **Overview Page**: High-level performance summary
   - GPU utilization %
   - Step time breakdown
   - Top TensorFlow operations

2. **Trace Viewer**: Timeline of operations
   - CPU/GPU activity over time
   - Kernel launch delays
   - Data transfer bottlenecks

3. **Memory Profile**: Memory usage patterns
   - Peak memory consumption
   - Memory allocation timeline
   - OOM prediction

4. **Kernel Stats**: GPU kernel performance
   - Execution time per kernel
   - Occupancy metrics
   - Roofline analysis

5. **TensorFlow Stats**: Op-level statistics
   - Time spent per operation
   - Host vs device time
   - Input pipeline analysis

**Common Performance Issues Detected:**

**GPU Underutilization:**
```
Issue: GPU only 30% utilized
Root Cause: CPU preprocessing bottleneck
Solution: Optimize data pipeline with tf.data/DataLoader prefetching
```

**Memory Bottleneck:**
```
Issue: High GPU memory usage (95%+)
Root Cause: Batch size too large
Solution: Reduce batch size or use gradient accumulation
```

**Input Pipeline Starvation:**
```
Issue: GPU waiting for data 40% of time
Root Cause: Slow data loading from GCS
Solution: Use larger prefetch buffer, enable parallel reads
```

### Profiling Best Practices

**Selective Profiling:**
- Profile only representative batches (not entire training)
- Skip initial batches (warmup period)
- Profile both training and validation steps

**Resource Considerations:**
- Profiling adds 5-10% overhead
- Generates large trace files (100MB-1GB)
- Only profile on representative hardware

**Interpretation:**
- Focus on "critical path" operations
- Look for unexpected CPU/GPU idle time
- Check data loading parallelism

---

## 4. Multi-Run Comparison

### Comparing Experiments

Vertex AI TensorBoard excels at comparing multiple training runs side-by-side.

**From** [Track, Compare, Manage Experiments with Vertex AI Experiments](https://cloud.google.com/blog/topics/developers-practitioners/track-compare-manage-experiments-vertex-ai-experiments) (accessed 2025-02-03):

**Organizing Runs:**
```python
# Create experiment runs with descriptive names
for lr in [1e-3, 5e-4, 1e-4]:
    run_name = f"vit-base-lr{lr:.0e}"

    job = aiplatform.CustomTrainingJob(
        display_name=run_name,
        container_uri=container_uri,
        tensorboard=tensorboard.resource_name
    )

    job.run(
        args=[f"--learning-rate={lr}"],
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1
    )
```

**UI Comparison:**
1. Navigate to TensorBoard web interface
2. Select multiple runs from sidebar
3. Scalars dashboard shows overlaid plots
4. Compare metrics across hyperparameter variations

**Filtering and Selection:**
- Regex filtering: `/.*lr1e-3.*/`
- Tag-based filtering by experiment tags
- Time-based filtering (last 7 days)

### Hyperparameter Analysis

**Tag Runs with Hyperparameters:**
```python
# Log hyperparameters as text
hparams = {
    "learning_rate": 1e-4,
    "batch_size": 64,
    "model": "vit-base-patch16",
    "optimizer": "adamw",
    "weight_decay": 0.01
}

writer.add_hparams(
    hparam_dict=hparams,
    metric_dict={
        "accuracy/val_final": 0.87,
        "loss/val_final": 0.42
    }
)
```

**Parallel Coordinates Plot:**
- Visualize relationship between hyperparameters and metrics
- Identify optimal hyperparameter ranges
- Filter runs by performance threshold

**Table View:**
- Sort runs by metric (e.g., best validation accuracy)
- Export run metadata and metrics to CSV
- Link to training job details in Vertex AI

### Statistical Comparison

**Smoothing and Aggregation:**
```python
# TensorBoard UI supports:
# - Exponential smoothing (adjustable)
# - Running average window
# - Outlier detection
# - Relative/absolute view
```

**Run Grouping:**
```
Group runs by shared prefix:
- vit-base-* (all VIT base model runs)
- vit-large-* (all VIT large model runs)
- Compare aggregated statistics
```

---

## 5. Embeddings Visualization

### Embedding Projector

**From** [Visualizing Data using the Embedding Projector in TensorBoard](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) (accessed 2025-02-03):

The Embedding Projector visualizes high-dimensional embeddings using dimensionality reduction.

**Logging Embeddings:**
```python
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter(log_dir)

# Extract embeddings (e.g., from final layer)
model.eval()
embeddings = []
labels = []
images_list = []

with torch.no_grad():
    for data, target in test_loader:
        output = model.get_embedding(data)  # Custom method
        embeddings.append(output)
        labels.append(target)
        images_list.append(data)

embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)
images = torch.cat(images_list, dim=0)

# Log to TensorBoard
writer.add_embedding(
    mat=embeddings,                    # (N, D) tensor
    metadata=labels.tolist(),          # Class labels
    label_img=images,                  # Thumbnail images
    global_step=epoch,
    tag='embedding/test_set'
)
```

### Dimensionality Reduction Methods

**Available in Embedding Projector:**

1. **PCA (Principal Component Analysis)**
   - Linear projection
   - Fast computation
   - Good for visualizing variance

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
   - Non-linear projection
   - Preserves local structure
   - Best for cluster visualization
   - Configurable perplexity (5-50)

3. **UMAP (Uniform Manifold Approximation and Projection)**
   - Non-linear projection
   - Faster than t-SNE
   - Preserves global structure better
   - Configurable n_neighbors

### Interactive Features

**Projection Controls:**
- Rotate/zoom 3D visualization
- Select reduction algorithm
- Adjust algorithm parameters (perplexity, learning rate)

**Point Inspection:**
- Click point to see label and metadata
- View thumbnail image (if provided)
- Find nearest neighbors in embedding space

**Filtering:**
- Search by metadata (e.g., class label)
- Isolate clusters visually
- Color points by metadata field

**Use Cases:**
- Visualize learned representations
- Identify class overlap/confusion
- Debug model clustering behavior
- Explore transfer learning features

---

## 6. Advanced TensorBoard Features

### Custom Plugins

TensorBoard supports custom plugins for specialized visualizations.

**Example: ROC Curve Plugin**
```python
from tensorboard.plugins.pr_curve import summary as pr_summary

# Log precision-recall curve
pr_summary.op(
    name='pr_curve',
    labels=labels,
    predictions=predictions,
    num_thresholds=200,
    display_name='PR Curve'
)
```

### What-If Tool Integration

**From** [What-If Tool](https://pair-code.github.io/what-if-tool/) (Google PAIR):

Interactive model exploration for:
- Individual example inspection
- Fairness analysis across groups
- Performance slicing by attributes
- Counterfactual analysis

**Setup:**
```python
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Configure What-If Tool
config_builder = WitConfigBuilder(
    test_examples=test_data,
    model_name='my-model'
).set_model_type('classification')

WitWidget(config_builder, height=800)
```

### Debugging with TensorBoard Debugger

**TF 2.x Debugger v2:**
```python
import tensorflow as tf

# Enable debugging
tf.debugging.experimental.enable_dump_debug_info(
    dump_root=log_dir,
    tensor_debug_mode="FULL_HEALTH",  # Check for NaN/Inf
    circular_buffer_size=-1
)

# Training runs with automatic tensor health checks
# Alerts on NaN/Inf in gradients or activations
```

### Integration with Vertex AI Pipelines

**From** [Vertex AI TensorBoard Integration with Vertex AI Pipelines](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/tensorboard/tensorboard_vertex_ai_pipelines_integration.ipynb) (accessed 2025-02-03):

**Pipeline Component with TensorBoard:**
```python
from kfp.v2 import dsl
from kfp.v2.dsl import component

@component(
    base_image="gcr.io/my-project/pytorch-trainer:latest",
    packages_to_install=["tensorboard"]
)
def train_model(
    tensorboard_resource_name: str,
    learning_rate: float,
    batch_size: int
) -> str:
    from google.cloud import aiplatform
    from torch.utils.tensorboard import SummaryWriter
    import os

    # Initialize Vertex AI with TensorBoard
    aiplatform.init(
        tensorboard=tensorboard_resource_name
    )

    # TensorBoard writer with managed backend
    writer = SummaryWriter(
        log_dir=os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    )

    # Training loop logs automatically to Vertex AI
    for epoch in range(num_epochs):
        train_loss = train_one_epoch()
        writer.add_scalar("loss/train", train_loss, epoch)

    writer.close()
    return "training_complete"

@dsl.pipeline(name="training-pipeline")
def pipeline(
    tensorboard_resource_name: str,
    learning_rate: float = 1e-4
):
    train_task = train_model(
        tensorboard_resource_name=tensorboard_resource_name,
        learning_rate=learning_rate,
        batch_size=64
    )
```

**Benefits:**
- Each pipeline run creates separate TensorBoard experiment
- Compare pipeline executions side-by-side
- Track metrics across pipeline stages
- Unified view of distributed training runs

---

## 7. Performance and Scaling

### TensorBoard Backend Optimization

**Event File Writing:**
- Batch writes every N steps (reduce I/O)
- Use `flush()` judiciously
- Async writing to GCS

```python
writer = SummaryWriter(
    log_dir=log_dir,
    flush_secs=120  # Flush every 2 minutes
)

# Log metrics
writer.add_scalar("loss", loss, step)

# Explicit flush after validation
writer.flush()
```

**Storage Costs:**
- Event files compressed automatically
- Typical size: 1-10 MB per 1000 steps
- Retention: Managed by TensorBoard instance settings

### Large-Scale Experiments

**Handling 100+ Runs:**
- Use experiment tagging for organization
- Archive old experiments (export to BigQuery)
- Use Vertex AI Experiments for metadata queries

**Distributed Training:**
```python
# Only rank 0 writes to TensorBoard (avoid conflicts)
if torch.distributed.get_rank() == 0:
    writer = SummaryWriter(log_dir)
    writer.add_scalar("loss", loss, global_step)
```

**Multi-Node Profiling:**
```python
# Profile each worker separately
worker_id = os.environ.get("CLOUD_ML_WORKER_ID", "0")
log_dir = f"{base_log_dir}/worker_{worker_id}"

# Aggregate profiling data offline
```

### Quotas and Limits

**Vertex AI TensorBoard Limits:**
- Max TensorBoard instances per project: 100
- Max experiments per TensorBoard: 10,000
- Max runs per experiment: 10,000
- Max event file size: 100 MB (automatic splitting)

**Performance Tips:**
- Log scalars every 10-100 steps (not every batch)
- Log images/histograms every epoch
- Use downsampling for long-running experiments

---

## 8. Team Collaboration

### Access Control

**IAM Roles:**
```bash
# Grant viewer access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:teammate@example.com" \
  --role="roles/aiplatform.viewer"

# Grant contributor access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:ml-engineer@example.com" \
  --role="roles/aiplatform.user"
```

### Sharing Experiments

**Shareable URLs:**
```python
# Get shareable TensorBoard URL
tensorboard = aiplatform.Tensorboard(tensorboard_id)
print(f"Share this URL: {tensorboard.web_url}")

# URL format:
# https://REGION-tensorboard.googleusercontent.com/experiment/EXPERIMENT_ID
```

**Screenshot and Export:**
- Download charts as SVG/PNG
- Export run data to CSV
- Share specific views via URL parameters

### Experiment Documentation

**Annotations and Notes:**
```python
# Add text descriptions
writer.add_text(
    "experiment_description",
    """
    # Vision Transformer Training

    ## Hypothesis
    Larger patch size improves efficiency with minimal accuracy loss.

    ## Configuration
    - Model: ViT-B/32 (patch size 32x32)
    - Dataset: ImageNet-1K
    - Training: 300 epochs, cosine LR schedule

    ## Results
    See accuracy/val metric for validation performance.
    """,
    global_step=0
)
```

---

## Sources

**Official Google Cloud Documentation:**
- [Introduction to Vertex AI TensorBoard](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) (accessed 2025-02-03)
- [Use Vertex AI TensorBoard with Custom Training](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-training) (accessed 2025-02-03)
- [Enable Cloud Profiler for Debugging Model Training Performance](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-02-03)
- [View Vertex AI TensorBoard Data](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-view) (accessed 2025-02-03)

**Google Cloud Blogs:**
- [Ray on Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/ray-on-vertex-ai) (accessed 2025-02-03)
- [Track, Compare, Manage Experiments with Vertex AI Experiments](https://cloud.google.com/blog/topics/developers-practitioners/track-compare-manage-experiments-vertex-ai-experiments) (accessed 2025-02-03)
- [5 Ways to Optimize Training Performance with TensorFlow Profiler on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/how-optimize-training-performance-tensorflow-profiler-vertex-ai) (accessed 2025-02-03)

**TensorFlow Documentation:**
- [TensorBoard Scalars and Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras) (accessed 2025-02-03)
- [Visualizing Data using the Embedding Projector in TensorBoard](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) (accessed 2025-02-03)
- [Get Started with TensorBoard](https://www.tensorflow.org/tensorboard/get_started) (accessed 2025-02-03)

**Code Samples:**
- [Vertex AI TensorBoard Integration with Vertex AI Pipelines](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/tensorboard/tensorboard_vertex_ai_pipelines_integration.ipynb) (Colab notebook, accessed 2025-02-03)
- [Profile Model Training Performance using Cloud Profiler](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/tensorboard/tensorboard_profiler_custom_training.ipynb) (Colab notebook, accessed 2025-02-03)

**Additional Resources:**
- [Deep Dive Into TensorBoard: Tutorial With Examples](https://neptune.ai/blog/tensorboard-tutorial) - neptune.ai (accessed 2025-02-03)
- [What to Expect from Vertex TensorBoard](https://medium.com/@j.berzborn/what-you-can-expect-from-vertex-tensorboard-53f5995bbac3) - Medium (accessed 2025-02-03)
