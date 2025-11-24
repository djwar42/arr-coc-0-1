# Advanced W&B Artifacts Patterns

## Overview

This guide covers advanced patterns for W&B Artifacts including collections, partitioning, incremental updates, dataset versioning strategies, lineage tracking, cross-project sharing, and storage optimization for production ML workflows.

**Key capabilities:**
- Artifact collections for organizing related versions
- Incremental artifact updates (append/modify/remove patterns)
- Dataset versioning strategies (full vs incremental)
- Artifact lineage and provenance tracking
- Cross-project artifact references
- Storage optimization and caching
- Distributed artifact creation
- Reference artifacts for external storage

From [W&B Artifacts Documentation](https://docs.wandb.ai/models/artifacts) (accessed 2025-01-31):
- Core artifact system architecture
- Version control and lineage tracking
- Storage and memory management

---

## Section 1: Advanced Artifact Patterns (150 lines)

### Collections and Partitioning

**Artifact Collections:**

Collections group related artifact versions for a specific task or use case. Each collection represents a distinct workflow stage.

```python
import wandb

# Create artifact collection for dataset splits
run = wandb.init(project="dataset-pipeline", job_type="prepare-splits")

# Training split artifact
train_artifact = wandb.Artifact(
    name="imagenet_train",
    type="dataset",
    description="ImageNet training split"
)
train_artifact.add_dir("data/train/")
run.log_artifact(train_artifact)

# Validation split artifact
val_artifact = wandb.Artifact(
    name="imagenet_val",
    type="dataset",
    description="ImageNet validation split"
)
val_artifact.add_dir("data/val/")
run.log_artifact(val_artifact)

# Test split artifact
test_artifact = wandb.Artifact(
    name="imagenet_test",
    type="dataset",
    description="ImageNet test split"
)
test_artifact.add_dir("data/test/")
run.log_artifact(test_artifact)

run.finish()
```

**When to use collections:**
- Dataset splits (train/val/test)
- Model families (ResNet-50, ResNet-101, ResNet-152)
- Evaluation suites (accuracy, robustness, fairness)
- Processing stages (raw → cleaned → augmented)

From [W&B Artifacts Overview](https://docs.wandb.ai/models/artifacts) (accessed 2025-01-31):
- Artifacts track inputs/outputs of runs
- Type parameter affects W&B UI organization
- Collections enable workflow stage management

### Partitioning Large Datasets

For datasets too large to manage as single artifacts, partition into logical chunks:

```python
import wandb
import os

# Partition by year/month for time-series data
run = wandb.init(project="logs-archive", job_type="partition-logs")

for year in range(2020, 2025):
    for month in range(1, 13):
        partition_artifact = wandb.Artifact(
            name=f"server_logs_{year}_{month:02d}",
            type="logs",
            metadata={
                "year": year,
                "month": month,
                "partition_key": f"{year}-{month:02d}"
            }
        )

        partition_path = f"logs/{year}/{month:02d}/"
        if os.path.exists(partition_path):
            partition_artifact.add_dir(partition_path)
            run.log_artifact(partition_artifact)

run.finish()
```

**Partitioning strategies:**
- Temporal: Year/month/day splits
- Spatial: Geographic regions
- Categorical: Class labels, data sources
- Size-based: Fixed chunk sizes (e.g., 1GB per partition)

**Benefits:**
- Faster downloads (only needed partitions)
- Parallel processing across partitions
- Easier data management and retention policies
- Reduced memory footprint

### Incremental Artifact Updates

From [W&B Create Artifact Version](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version) (accessed 2025-01-31):

Incremental artifacts modify a subset of files without re-uploading unchanged data:

```python
import wandb

# Start with existing artifact
run = wandb.init(project="dataset-curation", job_type="incremental-update")

# Fetch current artifact version
saved_artifact = run.use_artifact("curated_dataset:latest")

# Create draft for incremental changes
draft_artifact = saved_artifact.new_draft()

# Add new batch of annotated images
draft_artifact.add_file("annotations/batch_42.json")
draft_artifact.add_dir("images/batch_42/")

# Remove duplicate files discovered
draft_artifact.remove("images/batch_15/duplicate_001.jpg")
draft_artifact.remove("images/batch_15/duplicate_002.jpg")

# Update corrected annotations
draft_artifact.remove("annotations/batch_10.json")
draft_artifact.add_file("annotations/batch_10_corrected.json")

# Log new version (only changed files uploaded)
run.log_artifact(draft_artifact)
run.finish()
```

**Three incremental patterns:**

1. **Add**: Periodic new batches (active learning, continuous data collection)
2. **Remove**: Duplicates, corrupted data, privacy compliance
3. **Modify**: Annotation corrections, data quality improvements

**When to use incremental updates:**
- Large existing artifacts (>10GB)
- Small subset of changes (<10% of files)
- Continuous data pipelines
- Annotation refinement workflows

### Reference Artifacts (External Storage)

Track external data without duplicating storage:

```python
import wandb

run = wandb.init(project="s3-datasets", job_type="reference-external")

# Create artifact referencing S3 bucket
s3_artifact = wandb.Artifact(
    name="imagenet_s3",
    type="dataset",
    description="ImageNet stored in S3"
)

# Add S3 references (no data copied to W&B)
s3_artifact.add_reference(
    uri="s3://my-bucket/imagenet/train/",
    name="train"
)
s3_artifact.add_reference(
    uri="s3://my-bucket/imagenet/val/",
    name="val"
)

run.log_artifact(s3_artifact)
run.finish()
```

**Supported storage backends:**
- Amazon S3 (`s3://`)
- Google Cloud Storage (`gs://`)
- Azure Blob Storage (`azure://`)
- HTTP/HTTPS URLs
- Local file systems (absolute paths)

**Use cases:**
- Datasets too large for W&B storage limits
- Data in corporate data lakes
- Shared storage across teams
- Regulatory data residency requirements

From [W&B Track External Files](https://docs.wandb.ai/models/artifacts/track-external-files) (accessed 2025-01-31):
- Reference artifacts maintain lineage without copying data
- Checksums verify external file integrity
- Access credentials managed separately

---

## Section 2: Dataset Versioning Strategies (130 lines)

### Full vs Incremental Versioning

**Full versioning** re-creates entire artifact each version:

```python
import wandb

# Full dataset versioning
run = wandb.init(project="full-versioning", job_type="create-v2")

# Create new artifact from scratch
dataset_v2 = wandb.Artifact(
    name="processed_dataset",
    type="dataset",
    description="Fully reprocessed dataset v2"
)

# Add all files (even if unchanged from v1)
dataset_v2.add_dir("data/processed/v2/")
run.log_artifact(dataset_v2)
run.finish()
```

**Incremental versioning** only updates changed files:

```python
import wandb

# Incremental dataset versioning
run = wandb.init(project="incremental-versioning", job_type="update-to-v2")

# Start from v1
dataset_v1 = run.use_artifact("processed_dataset:v1")
dataset_v2 = dataset_v1.new_draft()

# Only add new/changed files
dataset_v2.add_file("data/new_batch_500.parquet")
dataset_v2.remove("data/old_batch_001.parquet")  # Remove outdated

run.log_artifact(dataset_v2)
run.finish()
```

**Decision matrix:**

| Scenario | Full Versioning | Incremental Versioning |
|----------|----------------|----------------------|
| Dataset size | <5GB | >10GB |
| Change frequency | Monthly | Daily/weekly |
| Change magnitude | >50% files | <10% files |
| Reproducibility | Critical | Important |
| Storage cost | Low priority | High priority |

### Data Preprocessing Pipelines

Track preprocessing transformations as artifacts:

```python
import wandb
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

run = wandb.init(project="preprocessing-pipeline", job_type="fit-scaler")

# Input: raw dataset artifact
raw_data_artifact = run.use_artifact("raw_sensor_data:latest")
raw_data_path = raw_data_artifact.download()

# Preprocessing
df = pd.read_csv(f"{raw_data_path}/sensors.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['temperature', 'pressure', 'humidity']])

# Output 1: Fitted scaler artifact
scaler_artifact = wandb.Artifact(
    name="sensor_scaler",
    type="preprocessor",
    metadata={
        "features": ['temperature', 'pressure', 'humidity'],
        "n_samples": len(df)
    }
)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
scaler_artifact.add_file("scaler.pkl")
run.log_artifact(scaler_artifact)

# Output 2: Processed dataset artifact
processed_artifact = wandb.Artifact(
    name="processed_sensor_data",
    type="dataset",
    metadata={
        "preprocessing": "StandardScaler",
        "scaler_version": "sensor_scaler:v0"
    }
)
pd.DataFrame(scaled_data, columns=['temp_scaled', 'press_scaled', 'humid_scaled']).to_csv("processed.csv")
processed_artifact.add_file("processed.csv")
run.log_artifact(processed_artifact)

run.finish()
```

**Preprocessing artifact types:**
- `preprocessor`: Fitted scalers, encoders, tokenizers
- `dataset`: Transformed data
- `config`: Preprocessing parameters
- `code`: Processing scripts (via `artifact.add_file()`)

### Train/Val/Test Split Management

Ensure consistent splits across experiments:

```python
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

run = wandb.init(project="split-management", job_type="create-splits")

# Input dataset
dataset_artifact = run.use_artifact("full_dataset:latest")
dataset_path = dataset_artifact.download()

# Load and split
data = np.load(f"{dataset_path}/data.npy")
labels = np.load(f"{dataset_path}/labels.npy")

# Create reproducible splits
train_idx, test_idx = train_test_split(
    np.arange(len(data)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.125,  # 10% of total (0.8 * 0.125 = 0.1)
    random_state=42,
    stratify=labels[train_idx]
)

# Save split indices as artifact
split_artifact = wandb.Artifact(
    name="dataset_splits",
    type="split_indices",
    metadata={
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "random_seed": 42
    }
)
np.save("train_indices.npy", train_idx)
np.save("val_indices.npy", val_idx)
np.save("test_indices.npy", test_idx)
split_artifact.add_file("train_indices.npy")
split_artifact.add_file("val_indices.npy")
split_artifact.add_file("test_indices.npy")

run.log_artifact(split_artifact)
run.finish()
```

**Using split indices:**

```python
# In training script
run = wandb.init(project="model-training", job_type="train")

split_artifact = run.use_artifact("dataset_splits:latest")
split_path = split_artifact.download()

train_idx = np.load(f"{split_path}/train_indices.npy")
val_idx = np.load(f"{split_path}/val_indices.npy")

# Load data and apply splits
dataset_artifact = run.use_artifact("full_dataset:latest")
# ... use train_idx, val_idx for consistent splits
```

### Data Augmentation Tracking

Track augmented datasets with lineage to source:

```python
import wandb
from torchvision import transforms

run = wandb.init(project="augmentation-tracking", job_type="augment-dataset")

# Source dataset
source_artifact = run.use_artifact("original_images:latest")
source_path = source_artifact.download()

# Define augmentation pipeline
augmentation_config = {
    "rotation": 15,
    "horizontal_flip": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2},
    "crop_scale": (0.8, 1.0)
}

# Create augmented dataset artifact
augmented_artifact = wandb.Artifact(
    name="augmented_images",
    type="dataset",
    metadata={
        "source": "original_images:latest",
        "augmentations": augmentation_config,
        "multiplier": 5  # 5x data from augmentation
    }
)

# Apply augmentations and add to artifact
# (implementation details omitted)
augmented_artifact.add_dir("data/augmented/")

run.log_artifact(augmented_artifact)
run.finish()
```

From [W&B Artifacts Walkthrough](https://docs.wandb.ai/models/artifacts/artifacts-walkthrough) (accessed 2025-01-31):
- Artifacts enable dataset provenance tracking
- Metadata stores preprocessing parameters
- Lineage connects augmented data to source

---

## Section 3: Production Artifact Patterns (120 lines)

### Model + Dataset + Config Bundles

Package complete training contexts for reproducibility:

```python
import wandb
import json
import torch

run = wandb.init(project="training-bundles", job_type="create-bundle")

# Create bundle artifact
bundle_artifact = wandb.Artifact(
    name="training_bundle_v1",
    type="bundle",
    description="Complete training context for ResNet-50"
)

# 1. Add model checkpoint
model_artifact = run.use_artifact("resnet50_trained:best")
model_path = model_artifact.download()
bundle_artifact.add_file(f"{model_path}/model.pth", name="model.pth")

# 2. Add dataset reference
dataset_artifact = run.use_artifact("imagenet_processed:v5")
bundle_artifact.add_reference(
    uri=f"wandb-artifact://{dataset_artifact.id}",
    name="dataset"
)

# 3. Add configuration
config = {
    "model": "ResNet-50",
    "dataset": "ImageNet",
    "dataset_version": "v5",
    "epochs": 90,
    "batch_size": 256,
    "learning_rate": 0.1,
    "optimizer": "SGD",
    "scheduler": "StepLR",
    "preprocessing": {
        "resize": 256,
        "crop": 224,
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
}
with open("training_config.json", "w") as f:
    json.dump(config, f, indent=2)
bundle_artifact.add_file("training_config.json")

# 4. Add training script
bundle_artifact.add_file("train.py")
bundle_artifact.add_file("model_architecture.py")

run.log_artifact(bundle_artifact)
run.finish()
```

**Reproducibility guarantees:**
- Exact model weights
- Exact dataset version
- Exact hyperparameters
- Exact training code

### Distributed Artifact Creation

From [W&B Create Artifact Version - Distributed Runs](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version#distributed-runs) (accessed 2025-01-31):

Multiple runs collaboratively create a single artifact version:

```python
import wandb
import argparse

# Run on worker 1
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int)
args = parser.parse_args()

run = wandb.init(
    project="distributed-processing",
    job_type="process-shard",
    group="batch_42"  # Shared distributed_id
)

# Each worker processes a shard
shard_artifact = wandb.Artifact(
    name="processed_results",
    type="dataset"
)

# Worker-specific processing
shard_file = f"results_worker_{args.worker_id}.parquet"
# ... process data shard ...
shard_artifact.add_file(shard_file)

# Upsert to collaborative artifact
run.upsert_artifact(
    shard_artifact,
    distributed_id="batch_42"  # Same across all workers
)

run.finish()
```

```python
# Final coordinator run (runs after all workers complete)
import wandb

run = wandb.init(
    project="distributed-processing",
    job_type="finalize",
    group="batch_42"
)

# Finalize the artifact (locks version)
final_artifact = wandb.Artifact(
    name="processed_results",
    type="dataset",
    metadata={"workers": 8, "batch_id": 42}
)

run.finish_artifact(
    final_artifact,
    distributed_id="batch_42"
)

run.finish()
```

**Distributed artifact workflow:**
1. Workers share same `distributed_id` (often run `group`)
2. Each worker calls `upsert_artifact()` with partial data
3. Final run calls `finish_artifact()` to commit version
4. All worker files combined into single artifact

### Artifact Caching Strategies

From [W&B Storage Management](https://docs.wandb.ai/models/artifacts/storage) (accessed 2025-01-31):

**Local cache configuration:**

```bash
# Set cache directory
export WANDB_CACHE_DIR="/mnt/fast-ssd/wandb-cache"

# Set artifact download directory
export WANDB_ARTIFACT_DIR="./artifacts"

# Clean up cache (keep 10GB)
wandb artifact cache cleanup 10GB
```

**Smart caching in code:**

```python
import wandb
import os

run = wandb.init(project="efficient-loading", job_type="train")

# Check if artifact already cached
artifact = run.use_artifact("large_dataset:latest")

cache_path = os.path.join(
    os.getenv("WANDB_CACHE_DIR", "~/.cache/wandb"),
    artifact.digest
)

if os.path.exists(cache_path):
    print(f"Using cached artifact at {cache_path}")
    dataset_path = cache_path
else:
    print("Downloading artifact...")
    dataset_path = artifact.download()

# Use dataset_path for training
run.finish()
```

**Performance optimizations:**
- Cache on fast local SSD for repeated access
- Share cache directory across team members
- Regular cleanup to manage disk usage
- Parallel downloads for large artifacts

### Storage Cost Optimization

**Deduplication across versions:**

W&B automatically deduplicates files across artifact versions (same content hash = stored once):

```python
import wandb

# Version 1: 1000 images
run = wandb.init(project="cost-optimization", job_type="v1")
v1_artifact = wandb.Artifact("dataset", type="dataset")
v1_artifact.add_dir("images/batch_1-10/")  # 1000 images
run.log_artifact(v1_artifact)
run.finish()

# Version 2: Add 100 new images
run = wandb.init(project="cost-optimization", job_type="v2")
v2_artifact = wandb.Artifact("dataset", type="dataset")
v2_artifact.add_dir("images/batch_1-10/")  # Same 1000 images (not re-uploaded!)
v2_artifact.add_dir("images/batch_11/")    # Only 100 new images uploaded
run.log_artifact(v2_artifact)
run.finish()
```

**Storage usage:** ~1000 images (not 2000) due to deduplication.

**Retention policies:**

```python
import wandb

# Set TTL (time-to-live) on temporary artifacts
run = wandb.init(project="retention-policies", job_type="temp-results")

temp_artifact = wandb.Artifact(
    name="experiment_outputs",
    type="results",
    metadata={"ttl_days": 30}  # Custom metadata for tracking
)
temp_artifact.add_dir("outputs/")

run.log_artifact(temp_artifact)
run.finish()
```

**Best practices:**
- Use aliases for important versions (`latest`, `production`, `best`)
- Delete old experiment artifacts regularly
- Use reference artifacts for large external datasets
- Leverage incremental updates for large datasets
- Archive to cheaper storage after N days

### Cross-Project Artifact Sharing

Reference artifacts across projects:

```python
import wandb

# Project A: Create shared artifact
run_a = wandb.init(project="project-a", job_type="create-shared")
shared_artifact = wandb.Artifact("shared_preprocessor", type="model")
shared_artifact.add_file("tokenizer.pkl")
run_a.log_artifact(shared_artifact)
run_a.finish()

# Project B: Use artifact from Project A
run_b = wandb.init(project="project-b", job_type="use-shared")
artifact_ref = run_b.use_artifact("project-a/shared_preprocessor:latest")
artifact_path = artifact_ref.download()
# Use shared preprocessor in Project B
run_b.finish()
```

**Cross-project syntax:**
- Same entity: `project-name/artifact-name:version`
- Different entity: `entity/project/artifact-name:version`
- Public artifacts: Include full path

**Use cases:**
- Shared preprocessing models across teams
- Company-wide foundational datasets
- Certified model checkpoints
- Standard evaluation benchmarks

### Cleanup and Retention Policies

**Automated cleanup script:**

```python
import wandb
from datetime import datetime, timedelta

api = wandb.Api()

# Define retention policy
KEEP_ALIASES = ["latest", "production", "best", "baseline"]
RETENTION_DAYS = 90

# Get all artifacts in project
artifacts = api.artifacts(
    type_name="dataset",
    project="my-project"
)

for artifact in artifacts:
    # Keep if has important alias
    if any(alias in artifact.aliases for alias in KEEP_ALIASES):
        continue

    # Keep if recent
    created_at = datetime.fromisoformat(artifact.created_at.replace('Z', '+00:00'))
    age_days = (datetime.now(created_at.tzinfo) - created_at).days

    if age_days < RETENTION_DAYS:
        continue

    # Delete old artifact
    print(f"Deleting {artifact.name}:{artifact.version} (age: {age_days} days)")
    artifact.delete()
```

**Retention policy matrix:**

| Artifact Type | Retention | Reason |
|--------------|-----------|--------|
| Raw datasets | 1 year | Source of truth |
| Processed datasets | 90 days | Reproducible from raw |
| Model checkpoints | Keep best 5 | Disk space |
| Experiment outputs | 30 days | Temporary results |
| Production models | Indefinite | Critical assets |

From [W&B Artifact TTL](https://docs.wandb.ai/models/artifacts/ttl) (accessed 2025-01-31):
- Artifacts support time-to-live policies
- Automatic cleanup reduces storage costs
- Important versions protected via aliases

---

## Sources

**W&B Official Documentation:**
- [Artifacts Overview](https://docs.wandb.ai/models/artifacts) - Core artifact concepts and API
- [Create Artifact Version](https://docs.wandb.ai/models/artifacts/create-a-new-artifact-version) - Incremental updates and distributed creation
- [Artifact Storage Management](https://docs.wandb.ai/models/artifacts/storage) - Cache configuration and cleanup
- [Explore Artifact Graphs](https://docs.wandb.ai/models/artifacts/explore-and-traverse-an-artifact-graph) - Lineage tracking and visualization
- [Track External Files](https://docs.wandb.ai/models/artifacts/track-external-files) - Reference artifacts for external storage
- [Artifact TTL](https://docs.wandb.ai/models/artifacts/ttl) - Data retention policies

**Web Research:**
- [W&B Artifacts Site](https://wandb.ai/site/artifacts) - Lightweight dataset and model versioning (accessed 2025-01-31)
- [Medium: Managing ML Workflow with W&B Artifacts](https://medium.com/the-owl/managing-your-machine-learning-workflow-with-weights-biases-artifacts-1577b6a58319) - Practical workflow patterns
- [Reddit: Model Performance Tracking & Versioning](https://www.reddit.com/r/datascience/comments/1dfk0hn/model_performance_tracking_versioning/) - W&B vs MLflow artifact comparison

**Additional References:**
- [W&B Artifact Fundamentals Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb) - Interactive tutorial
- [GitHub wandb/wandb](https://github.com/wandb/wandb) - Official SDK repository
- [Kaggle: Experiment Tracking with W&B](https://www.kaggle.com/code/ayuraj/experiment-tracking-with-weights-and-biases) - Dataset versioning examples
