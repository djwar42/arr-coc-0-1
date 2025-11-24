# Vertex AI Workbench & Experiments: Experiment Tracking and Metadata

## Overview

Vertex AI Workbench provides managed Jupyter notebook environments integrated with Vertex AI Experiments for tracking ML experiments, and Vertex AI Metadata Store for lineage tracking. This document covers experiment tracking, TensorBoard integration, metadata management, and collaborative development workflows.

**Core Philosophy**: Track everything, visualize comprehensively, share transparently - from notebook experiments to production pipelines.

From [Vertex AI Experiments Documentation](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) (accessed 2025-11-16):
> "Vertex AI Experiments helps you track and analyze different model architectures, hyperparameters, and training environments, enabling you to compare results and reproduce experiments."

---

## Section 1: Vertex AI Workbench Instances (~120 lines)

### 1.1 Workbench Overview

**Two deployment options (as of 2024):**

From [Vertex AI Workbench Release Notes](https://docs.cloud.google.com/vertex-ai/docs/workbench/release-notes) (accessed 2025-11-16):
- **Managed Notebooks** (deprecated April 14, 2025) → Migrate to **Workbench Instances**
- **User-Managed Notebooks** (deprecated April 14, 2025) → Migrate to **Workbench Instances**
- **Workbench Instances** (current recommended option)

**Workbench Instances features:**
```python
# Create Workbench instance via Python SDK
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Workbench instance creation
instance = aiplatform.NotebookRuntimeTemplate(
    display_name="ml-workbench-instance",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    network="projects/my-project/global/networks/my-vpc",
    idle_shutdown_timeout=1800,  # 30 minutes
)
```

**Key capabilities:**
- **Managed infrastructure**: Google-managed Jupyter environments
- **Custom containers**: Bring your own Docker images
- **GPU/TPU support**: Attach accelerators for training
- **VPC integration**: Private networking with Shared VPC
- **Idle shutdown**: Automatic cost optimization
- **Git integration**: GitHub/GitLab repository sync
- **Service account authentication**: IAM-based access control

### 1.2 Custom Container Support

**Using custom Docker images:**

```python
# Workbench with custom container
from google.cloud.aiplatform_v1 import NotebookServiceClient
from google.cloud.aiplatform_v1.types import NotebookRuntimeTemplate

client = NotebookServiceClient()

runtime_template = NotebookRuntimeTemplate(
    display_name="custom-pytorch-workbench",
    machine_spec={
        "machine_type": "n1-standard-8",
    },
    data_persistent_disk_spec={
        "disk_type": "PD_STANDARD",
        "disk_size_gb": 100,
    },
    network_spec={
        "enable_internet_access": True,
        "network": "projects/my-project/global/networks/my-vpc",
    },
    # Custom container image
    container_image_uri="gcr.io/my-project/pytorch-cuda:latest",
    # Environment variables
    environment_variables={
        "WANDB_API_KEY": "secret-manager://projects/my-project/secrets/wandb-key",
        "HF_TOKEN": "secret-manager://projects/my-project/secrets/hf-token",
    },
)

parent = f"projects/{project_id}/locations/{location}"
response = client.create_notebook_runtime_template(
    parent=parent,
    notebook_runtime_template=runtime_template,
    notebook_runtime_template_id="custom-pytorch-template",
)
```

**Pre-built container images:**
- `gcr.io/deeplearning-platform-release/pytorch-gpu`: PyTorch + CUDA
- `gcr.io/deeplearning-platform-release/tf2-gpu`: TensorFlow 2 + CUDA
- `gcr.io/deeplearning-platform-release/base-cpu`: Minimal Python environment

### 1.3 Workbench vs Colab Enterprise

From [Vertex AI Documentation](https://cloud.google.com/vertex-ai-notebooks) (accessed 2025-11-16):

| Feature | Workbench Instances | Colab Enterprise |
|---------|-------------------|------------------|
| **Use Case** | Individual development, experimentation | Team collaboration, production notebooks |
| **Compute** | Dedicated VM per user | Serverless, ephemeral kernels |
| **Persistence** | Persistent disk attached | BigQuery, GCS for data |
| **Custom Containers** | ✅ Supported | ✅ Supported |
| **Cost Model** | Pay per VM uptime | Pay per execution time |
| **Sharing** | Git-based sharing | Native notebook sharing UI |
| **Scheduling** | Vertex AI Executor service | Colab schedules |

---

## Section 2: Vertex AI Experiments API (~150 lines)

### 2.1 Experiments Fundamentals

**Hierarchy:**
```
Project
  └── Experiment (e.g., "arr-coc-v1-training")
       └── Run (individual training attempt)
            ├── Parameters (hyperparameters)
            ├── Metrics (loss, accuracy over time)
            └── Artifacts (models, datasets, plots)
```

From [Vertex AI Experiments Introduction](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) (accessed 2025-11-16):
> "Track steps, inputs, and outputs of your ML workflow. Compare different model architectures, hyperparameters, and training environments."

**Initialize experiment:**

```python
from google.cloud import aiplatform

aiplatform.init(
    project="my-project",
    location="us-central1",
    experiment="arr-coc-v1-training",
    experiment_description="ARR-COC MVP training experiments",
)

# Start experiment run
aiplatform.start_run(
    run="baseline-resnet50",
    tensorboard="projects/my-project/locations/us-central1/tensorboards/my-tb",
)
```

### 2.2 Logging Parameters and Metrics

**Log hyperparameters:**

```python
# Log configuration parameters
aiplatform.log_params({
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "AdamW",
    "model_architecture": "ResNet50",
    "token_budget_min": 64,
    "token_budget_max": 400,
    "lod_levels": 5,
})
```

**Log metrics over time:**

```python
# Training loop with metric logging
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # Log metrics (step auto-incremented)
    aiplatform.log_metrics({
        "train/loss": train_loss,
        "train/epoch": epoch,
    })

    aiplatform.log_metrics({
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
    })

    # Log custom metrics
    aiplatform.log_metrics({
        "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
    })
```

**Log time-series with explicit steps:**

```python
# Log at specific step numbers
global_step = 0
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        loss = train_step(model, batch)

        # Log every 10 batches
        if global_step % 10 == 0:
            aiplatform.log_metrics(
                {"batch_loss": loss.item()},
                step=global_step,
            )

        global_step += 1
```

### 2.3 Logging Artifacts

**Save and log models:**

```python
# Save model checkpoint
checkpoint_path = "gs://my-bucket/checkpoints/epoch_10.pt"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
}, checkpoint_path)

# Log artifact to experiment
aiplatform.log_model(
    model=checkpoint_path,
    artifact_id=f"arr-coc-checkpoint-epoch-{epoch}",
)
```

**Log plots and visualizations:**

```python
import matplotlib.pyplot as plt

# Create confusion matrix plot
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(y_true, y_pred, ax=ax)
plt.savefig("/tmp/confusion_matrix.png")

# Log plot as artifact
aiplatform.log_metrics({
    "confusion_matrix": "/tmp/confusion_matrix.png"
})
```

**Log datasets:**

```python
# Log dataset version used for training
aiplatform.log_params({
    "dataset_path": "gs://my-bucket/datasets/coco-2017-train",
    "dataset_version": "v2.0",
    "num_samples": 118287,
})
```

### 2.4 Comparing Experiment Runs

**Retrieve and compare runs:**

```python
from google.cloud.aiplatform import Experiment

# Get experiment
experiment = Experiment(
    experiment_name="arr-coc-v1-training",
    project="my-project",
    location="us-central1",
)

# List all runs
runs_df = experiment.get_data_frame()
print(runs_df[["run_name", "metric.val/accuracy", "param.learning_rate"]])

# Filter runs by metric
best_runs = runs_df[runs_df["metric.val/accuracy"] > 0.90]
print(f"Runs with >90% accuracy: {len(best_runs)}")

# Get specific run details
run = experiment.get_run("baseline-resnet50")
metrics = run.get_metrics()
params = run.get_params()
artifacts = run.get_artifacts()
```

---

## Section 3: TensorBoard Integration (~140 lines)

### 3.1 Vertex AI TensorBoard Setup

From [TensorBoard Vertex AI Integration](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) (accessed 2025-11-16):

**Create TensorBoard instance:**

```bash
# Create via gcloud
gcloud ai tensorboards create \
    --display-name="arr-coc-tensorboard" \
    --project=my-project \
    --region=us-central1 \
    --description="ARR-COC training monitoring"
```

**Python SDK:**

```python
from google.cloud import aiplatform

tensorboard = aiplatform.Tensorboard.create(
    display_name="arr-coc-tensorboard",
    project="my-project",
    location="us-central1",
)

print(f"TensorBoard created: {tensorboard.resource_name}")
print(f"Web UI: {tensorboard.web_app_uri}")
```

### 3.2 TensorBoard Logging from Training

**PyTorch integration:**

```python
from torch.utils.tensorboard import SummaryWriter
from google.cloud import aiplatform

# Initialize TensorBoard writer
log_dir = "gs://my-bucket/tensorboard-logs/run-001"
writer = SummaryWriter(log_dir=log_dir)

# Start Vertex AI experiment run with TensorBoard
aiplatform.init(
    project="my-project",
    location="us-central1",
    experiment="arr-coc-v1-training",
)

aiplatform.start_run(
    run="baseline-resnet50",
    tensorboard=tensorboard.resource_name,
)

# Training loop with TensorBoard logging
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log scalar metrics
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/learning_rate',
                         optimizer.param_groups[0]['lr'], global_step)

    # Validation metrics
    val_loss, val_acc = validate(model, val_loader)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/accuracy', val_acc, epoch)

    # Log histograms of model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

writer.close()
```

### 3.3 TensorBoard Custom Scalars and Embeddings

**Custom scalar layouts:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir)

# Define custom layout for TensorBoard UI
layout = {
    "Training Metrics": {
        "loss": ["Multiline", ["train/loss", "val/loss"]],
        "accuracy": ["Multiline", ["train/accuracy", "val/accuracy"]],
    },
    "ARR-COC Specific": {
        "token_allocation": ["Multiline", [
            "avg_tokens_per_patch",
            "max_tokens_per_patch",
            "min_tokens_per_patch",
        ]],
        "relevance_scores": ["Multiline", [
            "propositional_score",
            "perspectival_score",
            "participatory_score",
        ]],
    },
}

writer.add_custom_scalars(layout)
```

**Log embeddings for visualization:**

```python
import torch
from torch.utils.tensorboard import SummaryWriter

# Generate patch embeddings
patch_embeddings = model.get_patch_embeddings(images)  # (N, D)
labels = [f"patch_{i}" for i in range(len(patch_embeddings))]

# Log to TensorBoard (creates t-SNE/PCA visualizations)
writer.add_embedding(
    mat=patch_embeddings,
    metadata=labels,
    tag="patch_embeddings",
    global_step=epoch,
)
```

### 3.4 TensorBoard Profiling

From [TensorBoard Profiler Documentation](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-11-16):

**Enable profiling:**

```python
import torch.profiler

# TensorBoard profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:  # wait + warmup + active × repeat
            break

        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prof.step()  # Signal profiler to move to next step

# Profiling results automatically uploaded to TensorBoard
```

---

## Section 4: Vertex AI Metadata Store (~130 lines)

### 4.1 Metadata Store Architecture

From [Vertex ML Metadata Introduction](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/introduction) (accessed 2025-11-16):

**Core entities:**
- **Artifacts**: Datasets, models, metrics (immutable data)
- **Executions**: Training runs, preprocessing steps (processes)
- **Contexts**: Experiments, pipelines (grouping mechanism)
- **Events**: Relationships between artifacts and executions

**Automatic metadata tracking:**
- Vertex AI Pipelines automatically log metadata
- Vertex AI Training jobs create execution records
- Model Registry uploads create artifact records

### 4.2 Lineage Tracking

**Querying lineage programmatically:**

```python
from google.cloud.aiplatform import Artifact, Execution

# Get artifact (e.g., trained model)
model_artifact = Artifact(
    artifact_name="projects/123/locations/us-central1/metadataStores/default/artifacts/456",
)

# Get upstream lineage (what created this model?)
upstream_artifacts = model_artifact.get_upstream_artifacts()
for artifact in upstream_artifacts:
    print(f"Upstream: {artifact.display_name} (type: {artifact.schema_title})")

# Get downstream lineage (what uses this model?)
downstream_artifacts = model_artifact.get_downstream_artifacts()
for artifact in downstream_artifacts:
    print(f"Downstream: {artifact.display_name}")

# Get execution that created this artifact
executions = model_artifact.get_executions()
for execution in executions:
    print(f"Created by: {execution.display_name}")
    print(f"State: {execution.state}")
```

**Visualizing lineage in UI:**
- Navigate to Vertex AI > Metadata
- Select artifact or execution
- View lineage graph (interactive DAG visualization)

### 4.3 Custom Metadata Tracking

**Manually create metadata records:**

```python
from google.cloud.aiplatform import Artifact, Execution, Context

# Create context for experiment
experiment_context = Context.create(
    schema_title="system.Experiment",
    display_name="arr-coc-v1-baseline",
    description="Baseline ARR-COC training with ResNet50 backbone",
    metadata={
        "start_time": "2025-11-16T10:00:00Z",
        "owner": "ml-team@example.com",
    },
)

# Create dataset artifact
dataset_artifact = Artifact.create(
    schema_title="system.Dataset",
    display_name="coco-2017-train-processed",
    uri="gs://my-bucket/datasets/coco-2017-processed/",
    metadata={
        "num_samples": 118287,
        "num_classes": 80,
        "preprocessing_version": "v2.0",
    },
)

# Create training execution
training_execution = Execution.create(
    schema_title="system.ContainerExecution",
    display_name="arr-coc-training-run-001",
    metadata={
        "image": "gcr.io/my-project/arr-coc-trainer:v1",
        "machine_type": "a2-highgpu-1g",
        "hyperparameters": {
            "learning_rate": 3e-4,
            "batch_size": 32,
        },
    },
)

# Link execution to context
training_execution.assign_to_context(experiment_context)

# Record input artifact to execution
training_execution.assign_input_artifacts([dataset_artifact])

# Create model artifact as output
model_artifact = Artifact.create(
    schema_title="system.Model",
    display_name="arr-coc-model-epoch-50",
    uri="gs://my-bucket/models/arr-coc-epoch-50.pt",
    metadata={
        "val_accuracy": 0.92,
        "val_loss": 0.24,
        "framework": "PyTorch",
        "framework_version": "2.1.0",
    },
)

# Record output artifact from execution
training_execution.assign_output_artifacts([model_artifact])
```

### 4.4 Metadata Store Queries

**Search for artifacts:**

```python
from google.cloud.aiplatform import Artifact

# Find all models with accuracy > 90%
models = Artifact.list(
    filter='schema_title="system.Model" AND metadata.val_accuracy>0.90',
    order_by="create_time desc",
)

for model in models:
    print(f"Model: {model.display_name}")
    print(f"  Accuracy: {model.metadata['val_accuracy']}")
    print(f"  URI: {model.uri}")
```

---

## Section 5: Git Integration and Collaborative Development (~100 lines)

### 5.1 Git Repository Sync

**Connect Workbench to GitHub:**

```bash
# In Workbench terminal
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone repository
git clone https://github.com/your-org/arr-coc.git
cd arr-coc

# Work with notebooks
jupyter notebook
```

**Automatic git sync (via Workbench UI):**
- Workbench > Create Instance > Git Repository URL
- Automatically clones repo on instance creation
- Pull latest changes on instance restart

### 5.2 Notebook Versioning Best Practices

From [W&B Integration Basics](../karpathy/gradio/10-wandb-integration-basics.md):

**Clear outputs before committing:**

```bash
# Install nbstripout (removes notebook outputs)
pip install nbstripout

# Setup git filter
nbstripout --install

# Now git commits won't include cell outputs
git add notebook.ipynb
git commit -m "Add experiment notebook"
```

**Track experiments, not outputs:**
- Use Vertex AI Experiments to log results
- Keep notebooks focused on code/documentation
- Store outputs (plots, models) in GCS/TensorBoard

### 5.3 Shared Workbench Access

**Service account mode (team sharing):**

```python
# Create Workbench instance with service account
from google.cloud.aiplatform_v1 import NotebookServiceClient
from google.cloud.aiplatform_v1.types import NotebookRuntimeTemplate

runtime_template = NotebookRuntimeTemplate(
    display_name="shared-team-workbench",
    # Service account for shared access
    service_account="ml-team@my-project.iam.gserviceaccount.com",
    # Shareable link enabled
    enable_shareable_link=True,
)
```

**Grant access via IAM:**

```bash
# Grant user access to Workbench instance
gcloud notebooks instances add-iam-policy-binding my-instance \
    --member="user:teammate@example.com" \
    --role="roles/notebooks.runner" \
    --location=us-central1
```

---

## Section 6: Notebook Scheduling (Executor Service) (~80 lines)

### 6.1 Scheduled Notebook Execution

From [Vertex AI Workbench Documentation](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/migrate-to-instances) (accessed 2025-11-16):

**Create scheduled execution:**

```python
from google.cloud import aiplatform

# Define notebook execution
execution = aiplatform.NotebookExecution.create(
    display_name="daily-training-run",
    notebook_file="gs://my-bucket/notebooks/train_arr_coc.ipynb",
    execution_schedule="0 2 * * *",  # Daily at 2 AM UTC (cron format)
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    service_account="ml-training@my-project.iam.gserviceaccount.com",
    # Output notebook saved to GCS
    output_notebook_folder="gs://my-bucket/notebook-outputs/",
)

print(f"Scheduled execution: {execution.resource_name}")
```

**Parameterized notebook execution:**

```python
# Notebook with parameters (using papermill tags)
# Cell 1 (tagged as "parameters"):
# learning_rate = 3e-4
# batch_size = 32
# epochs = 100

# Execute notebook with custom parameters
execution = aiplatform.NotebookExecution.create(
    display_name="arr-coc-hyperparameter-sweep",
    notebook_file="gs://my-bucket/notebooks/train_arr_coc.ipynb",
    parameters={
        "learning_rate": 1e-4,
        "batch_size": 64,
        "epochs": 200,
    },
)
```

### 6.2 Execution Monitoring

**Track execution status:**

```python
# Get execution
execution = aiplatform.NotebookExecution("projects/.../notebookExecutions/123")

# Check status
print(f"State: {execution.state}")  # RUNNING, SUCCEEDED, FAILED

# Get output notebook
if execution.state == "SUCCEEDED":
    output_uri = execution.output_notebook_file
    print(f"Output notebook: {output_uri}")

# Get execution logs
logs = execution.get_logs()
```

---

## Section 7: arr-coc-0-1 Experiment Tracking Example (~80 lines)

### 7.1 Complete Training Integration

**arr-coc-0-1 training with full experiment tracking:**

```python
from google.cloud import aiplatform
from torch.utils.tensorboard import SummaryWriter
import torch

# Initialize Vertex AI
aiplatform.init(
    project="arr-coc-project",
    location="us-west2",
    experiment="arr-coc-v1-training",
    staging_bucket="gs://arr-coc-training",
)

# Create TensorBoard instance (one-time setup)
tensorboard = aiplatform.Tensorboard.create(
    display_name="arr-coc-tensorboard",
)

# Start experiment run
run_name = f"run-lr{learning_rate}-bs{batch_size}"
aiplatform.start_run(
    run=run_name,
    tensorboard=tensorboard.resource_name,
)

# Log hyperparameters
aiplatform.log_params({
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "token_budget_min": 64,
    "token_budget_max": 400,
    "lod_levels": 5,
    "relevance_threshold": 0.3,
    "model_backbone": "resnet50",
    "optimizer": "AdamW",
    "weight_decay": 0.01,
})

# TensorBoard writer
log_dir = f"gs://arr-coc-training/tensorboard/{run_name}"
writer = SummaryWriter(log_dir=log_dir)

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        loss, metrics = train_step(model, batch)
        train_loss += loss.item()

        # Log batch metrics to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), global_step)
        writer.add_scalar('train/avg_tokens_allocated',
                         metrics['avg_tokens'], global_step)

    # Validation phase
    val_loss, val_metrics = validate(model, val_loader)

    # Log epoch metrics to Vertex AI Experiments
    aiplatform.log_metrics({
        "train/epoch_loss": train_loss / len(train_loader),
        "val/loss": val_loss,
        "val/accuracy": val_metrics['accuracy'],
        "val/avg_tokens_per_patch": val_metrics['avg_tokens'],
        "epoch": epoch,
    })

    # Log to TensorBoard
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)

    # Save checkpoint
    if epoch % 10 == 0:
        checkpoint_path = f"gs://arr-coc-training/checkpoints/epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Log model artifact
        aiplatform.log_model(
            model=checkpoint_path,
            artifact_id=f"arr-coc-epoch-{epoch}",
        )

# Final model
final_model_path = "gs://arr-coc-training/models/arr-coc-final.pt"
torch.save(model.state_dict(), final_model_path)

aiplatform.log_model(
    model=final_model_path,
    artifact_id="arr-coc-final-model",
)

writer.close()
aiplatform.end_run()
```

### 7.2 Comparing ARR-COC Experiments

**Analyze multiple runs:**

```python
from google.cloud.aiplatform import Experiment
import pandas as pd

# Get experiment
experiment = Experiment(
    experiment_name="arr-coc-v1-training",
    project="arr-coc-project",
    location="us-west2",
)

# Get all runs as DataFrame
runs_df = experiment.get_data_frame()

# Filter best runs
best_runs = runs_df[runs_df["metric.val/accuracy"] > 0.90]

# Analyze token allocation efficiency
print("Token allocation vs accuracy:")
print(runs_df[["run_name",
               "metric.val/accuracy",
               "metric.val/avg_tokens_per_patch",
               "param.token_budget_max"]].sort_values("metric.val/accuracy", ascending=False))

# Find optimal hyperparameters
optimal_lr = best_runs["param.learning_rate"].mode()[0]
optimal_bs = best_runs["param.batch_size"].mode()[0]

print(f"Optimal learning rate: {optimal_lr}")
print(f"Optimal batch size: {optimal_bs}")
```

---

## Sources

**Official Documentation:**
- [Vertex AI Experiments Introduction](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) - Experiment tracking fundamentals (accessed 2025-11-16)
- [Vertex AI Workbench Release Notes](https://docs.cloud.google.com/vertex-ai/docs/workbench/release-notes) - Workbench deprecation and migration timeline (accessed 2025-11-16)
- [TensorBoard Vertex AI Integration](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) - TensorBoard setup and integration (accessed 2025-11-16)
- [Vertex ML Metadata Introduction](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/introduction) - Metadata Store and lineage tracking (accessed 2025-11-16)
- [TensorBoard Profiler](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) - Performance profiling integration (accessed 2025-11-16)
- [Vertex AI Notebooks](https://cloud.google.com/vertex-ai-notebooks) - Workbench managed infrastructure (accessed 2025-11-16)

**Source Documents:**
- [10-wandb-integration-basics.md](../karpathy/gradio/10-wandb-integration-basics.md) - W&B experiment tracking patterns
- [15-wandb-quick-validation.md](../karpathy/practical-implementation/15-wandb-quick-validation.md) - Smoke testing and validation workflows

**Web Research:**
- Google Search: "Vertex AI Workbench managed notebooks 2024" (accessed 2025-11-16)
- Google Search: "Vertex AI Experiments API tracking" (accessed 2025-11-16)
- Google Search: "TensorBoard Vertex AI integration" (accessed 2025-11-16)
- Google Search: "Vertex AI Metadata Store ML lineage" (accessed 2025-11-16)

**Additional Resources:**
- [Vertex AI Workbench Migration Guide](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/migrate-to-instances) - Migrating from user-managed notebooks
- [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) - Medium article on experiment tracking patterns (accessed 2025-11-16)
- [Manage Machine Learning Experiments with Vertex AI](https://codelabs.developers.google.com/vertex_experiments_pipelines_intro) - Google Codelabs tutorial (accessed 2025-11-16)
