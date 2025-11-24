# Vertex AI ML Metadata

## Overview

Vertex ML Metadata is a managed service for tracking and analyzing the metadata produced by machine learning workflows. It provides automatic lineage tracking, artifact versioning, and experiment analysis capabilities across the entire ML lifecycle.

**Core Purpose**: Track artifacts, executions, and contexts to enable ML workflow reproducibility, debugging, and governance.

From [Introduction to Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/introduction) (accessed 2025-02-03):
- Automatically captures metadata from Vertex AI Pipelines, Custom Jobs, and Experiments
- Based on ML Metadata (MLMD) library from TensorFlow Serving
- Integrated with Vertex AI Experiments for comprehensive tracking

## Architecture

### Core Metadata Entities

**Three fundamental entity types** form the metadata graph:

**1. Artifacts**
- Represent data objects (datasets, models, metrics)
- Immutable once created
- Typed (Dataset, Model, Metrics, etc.)
- Store URIs, properties, and custom metadata

From [ML metadata artifact types](https://cloud.google.com/vertex-ai/docs/pipelines/artifact-types) (accessed 2025-02-03):
- Standard first-party types defined by Google Cloud Pipeline Components
- Custom artifact types supported
- Type schema defines expected properties

**2. Executions**
- Represent workflow steps or runs
- Capture runtime information
- Link input/output artifacts
- Store state, parameters, and logs

**3. Contexts**
- Logical grouping mechanism
- Represent pipelines, experiments, or projects
- Create subgraphs of related artifacts and executions
- Enable hierarchical organization

### Metadata Store

From [REST Resource: MetadataStore](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.metadataStores) (accessed 2025-02-03):
- Regional resource (must match region of Vertex AI services)
- Default store auto-created on first use
- Custom stores supported for isolation
- Pricing: $10 per GiB stored

**Store Structure**:
```
MetadataStore (default)
├── Contexts (pipelines, experiments)
├── Artifacts (datasets, models)
├── Executions (pipeline runs, training jobs)
└── Events (linkages between entities)
```

## Artifact Tracking

### Artifact Types

From [The nuts and bolts of VertexAI](https://leftasexercise.com/2024/02/18/the-nuts-and-bolts-of-vertexai-metadata-logging-and-metrics/) (accessed 2025-02-03):
- **Dataset**: Training/validation/test data
- **Model**: Trained model artifacts
- **Metrics**: Performance measurements
- **ClassificationMetrics**: Classification-specific metrics
- **HTML**: Visualization artifacts
- **Markdown**: Documentation artifacts
- **UnmanagedContainerModel**: Externally managed models
- **VertexModel**: Models registered in Vertex AI Model Registry

### Creating Artifacts

**Via Vertex AI Pipelines** (automatic):
```python
from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp

dataset_op = TabularDatasetCreateOp(
    display_name="my-dataset",
    gcs_source="gs://bucket/data.csv",
    project=PROJECT_ID,
    location=REGION
)
# Artifact automatically created in ML Metadata
```

**Via Python SDK** (manual):

From [Track Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/tracking) (accessed 2025-02-03):
```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

# Create artifact
artifact = aiplatform.Artifact.create(
    schema_title="system.Dataset",
    display_name="training-dataset-v1",
    uri="gs://bucket/training-data",
    metadata={
        "num_rows": 10000,
        "num_features": 50,
        "created_by": "data-pipeline"
    }
)

print(f"Created artifact: {artifact.resource_name}")
```

### Artifact Properties

**Immutable Properties**:
- `schema_title`: Artifact type (e.g., "system.Dataset")
- `uri`: Location of artifact
- `create_time`: Creation timestamp

**Mutable Properties**:
- `display_name`: Human-readable name
- `description`: Artifact description
- `metadata`: Custom key-value pairs
- `labels`: For filtering/organization
- `state`: Lifecycle state (LIVE, PENDING, DELETED)

### Querying Artifacts

From [Analyze Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/analyzing) (accessed 2025-02-03):

**List artifacts by type**:
```python
artifacts = aiplatform.Artifact.list(
    filter='schema_title="system.Model"',
    order_by="create_time desc"
)

for artifact in artifacts:
    print(f"{artifact.display_name}: {artifact.uri}")
```

**Query by metadata**:
```python
# Find artifacts with specific metadata
artifacts = aiplatform.Artifact.list(
    filter='metadata.framework="pytorch" AND metadata.version="2.0"'
)
```

**Retrieve single artifact**:
```python
artifact = aiplatform.Artifact(artifact_name=ARTIFACT_RESOURCE_NAME)
print(f"URI: {artifact.uri}")
print(f"Metadata: {artifact.metadata}")
```

## Execution Tracking

### Execution Lifecycle

**Execution states**:
- `NEW`: Just created
- `RUNNING`: Currently executing
- `COMPLETE`: Finished successfully
- `FAILED`: Terminated with error
- `CACHED`: Result reused from cache
- `CANCELLED`: Manually stopped

### Creating Executions

From [Manage Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/managing-metadata) (accessed 2025-02-03):

```python
from google.cloud import aiplatform

# Create execution
execution = aiplatform.Execution.create(
    schema_title="system.ContainerExecution",
    display_name="training-run-42",
    metadata={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    }
)

# Update state during run
execution.update(state=aiplatform.gapic.Execution.State.RUNNING)

# Record completion
execution.update(
    state=aiplatform.gapic.Execution.State.COMPLETE,
    metadata={
        "final_loss": 0.042,
        "training_time_seconds": 3600
    }
)
```

### Linking Artifacts to Executions

**Input/Output relationships**:
```python
# Create event linking input artifact to execution
execution.assign_input_artifacts([dataset_artifact])

# Create event linking execution to output artifacts
execution.assign_output_artifacts([model_artifact, metrics_artifact])
```

**Event types**:
- `INPUT`: Artifact consumed by execution
- `OUTPUT`: Artifact produced by execution
- `DECLARED_INPUT`: Planned input (before execution)
- `DECLARED_OUTPUT`: Planned output (before execution)

## Lineage Tracking

### Lineage Graph

From [Track the lineage of pipeline artifacts](https://docs.cloud.google.com/vertex-ai/docs/pipelines/lineage) (accessed 2025-02-03):
- Directed acyclic graph (DAG) of metadata entities
- Nodes: Artifacts, Executions, Contexts
- Edges: Events (input/output relationships)
- Automatically captured for Vertex AI Pipelines

**Lineage visualization** in Vertex AI console:
- Click any artifact → View lineage
- Shows upstream (inputs) and downstream (outputs)
- Traces data provenance across pipeline runs

### Querying Lineage

**Get artifact lineage**:
```python
from google.cloud.aiplatform import Artifact

# Get all upstream artifacts (inputs)
artifact = Artifact(artifact_name=MODEL_ARTIFACT_NAME)
lineage = artifact.get_upstream_artifacts()

for parent_artifact in lineage:
    print(f"Input: {parent_artifact.display_name}")
```

**Get execution lineage**:
```python
from google.cloud.aiplatform import Execution

execution = Execution(execution_name=EXECUTION_NAME)

# Get input artifacts
inputs = execution.get_input_artifacts()
for inp in inputs:
    print(f"Consumed: {inp.display_name}")

# Get output artifacts
outputs = execution.get_output_artifacts()
for out in outputs:
    print(f"Produced: {out.display_name}")
```

### Lineage API Examples

From [Build Vertex AI Experiment lineage](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/build_model_experimentation_lineage_with_prebuild_code.ipynb) (accessed 2025-02-03):

**Complete lineage tracking workflow**:
```python
from google.cloud import aiplatform

# 1. Create dataset artifact
dataset = aiplatform.Artifact.create(
    schema_title="system.Dataset",
    uri="gs://bucket/train.csv",
    display_name="training-data-v3",
    metadata={"rows": 50000, "version": "v3"}
)

# 2. Create training execution
training_run = aiplatform.Execution.create(
    schema_title="system.ContainerExecution",
    display_name="train-resnet-50",
    metadata={"model_arch": "resnet50", "lr": 0.001}
)

# 3. Link dataset as input
training_run.assign_input_artifacts([dataset])

# 4. Create model artifact
model = aiplatform.Artifact.create(
    schema_title="system.Model",
    uri="gs://bucket/model.pkl",
    display_name="resnet-50-trained",
    metadata={"accuracy": 0.94, "f1": 0.92}
)

# 5. Link model as output
training_run.assign_output_artifacts([model])

# 6. Query lineage
print("Model lineage:")
for input_artifact in model.get_upstream_artifacts():
    print(f"  Trained on: {input_artifact.display_name}")
```

## Context Management

### Context Hierarchy

From [Manage Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/managing-metadata) (accessed 2025-02-03):
- Contexts group related artifacts and executions
- Enable logical organization (by project, pipeline, experiment)
- Support nested hierarchies

**Common context patterns**:
```
Experiment Context
├── Pipeline Context (run 1)
│   ├── Dataset Artifact
│   ├── Training Execution
│   └── Model Artifact
└── Pipeline Context (run 2)
    ├── Dataset Artifact
    ├── Training Execution
    └── Model Artifact
```

### Creating Contexts

```python
from google.cloud import aiplatform

# Create experiment context
experiment_context = aiplatform.Context.create(
    schema_title="system.Experiment",
    display_name="resnet-hyperparameter-tuning",
    description="HPO for ResNet50 on ImageNet",
    metadata={"framework": "pytorch", "dataset": "imagenet"}
)

# Create pipeline run context
pipeline_context = aiplatform.Context.create(
    schema_title="system.PipelineRun",
    display_name="training-run-42",
    metadata={"run_id": "42", "triggered_by": "scheduler"}
)

# Add artifacts to context
experiment_context.add_artifacts_and_executions(
    artifact_resource_names=[dataset.resource_name, model.resource_name],
    execution_resource_names=[training_run.resource_name]
)
```

### Querying by Context

```python
# List all contexts of a type
experiments = aiplatform.Context.list(
    filter='schema_title="system.Experiment"'
)

# Get context and query its contents
context = aiplatform.Context(context_name=CONTEXT_NAME)
artifacts = context.query_artifacts()
executions = context.query_executions()

print(f"Context {context.display_name} contains:")
print(f"  {len(artifacts)} artifacts")
print(f"  {len(executions)} executions")
```

## Integration with Vertex AI Services

### Vertex AI Pipelines Integration

From [vertex-pipelines-ml-metadata.ipynb](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/ml_metadata/vertex-pipelines-ml-metadata.ipynb) (accessed 2025-02-03):
- **Automatic metadata capture** for all pipeline components
- Each component execution creates Execution entity
- Component inputs/outputs create Artifact entities
- Pipeline run creates Context entity

**Metadata automatically captured**:
- Component parameters
- Input/output URIs
- Runtime metrics
- Error messages (on failure)
- Caching information

### Vertex AI Experiments Integration

From [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) (accessed 2025-02-03):
- Experiments stored in Vertex ML Metadata
- Parameters logged as Execution metadata
- Metrics logged as Artifact metadata
- Auto-linking between experiments and models

**Track experiment with metadata**:
```python
from google.cloud import aiplatform

aiplatform.init(
    experiment="hyperparameter-tuning",
    experiment_tensorboard=TENSORBOARD_RESOURCE_NAME
)

aiplatform.start_run(run="trial-42")

# Log parameters (stored as Execution metadata)
aiplatform.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam"
})

# Train model...

# Log metrics (stored as Artifact metadata)
aiplatform.log_metrics({
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.90
})

aiplatform.end_run()
```

### Model Registry Integration

From [Vertex AI Best Practices 2025](https://skywork.ai/blog/vertex-ai-best-practices-governance-quotas-collaboration/) (accessed 2025-02-03):
- Models registered in Model Registry automatically linked to ML Metadata
- Lineage from training data → training run → model version
- Model cards enriched with metadata pointers

**Query models with lineage**:
```python
# Get model from registry
model = aiplatform.Model(model_name=MODEL_RESOURCE_NAME)

# Query training lineage
training_executions = model.get_upstream_executions()
for execution in training_executions:
    print(f"Trained by: {execution.display_name}")
    print(f"Parameters: {execution.metadata}")

    # Get training datasets
    datasets = execution.get_input_artifacts()
    for dataset in datasets:
        print(f"  Trained on: {dataset.uri}")
```

## Advanced Query Patterns

### REST API Queries

From [REST Resource: artifacts.list](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.metadataStores.artifacts/list) (accessed 2025-02-03):

**Filter syntax**:
- Equality: `schema_title="system.Model"`
- Comparison: `create_time > "2024-01-01T00:00:00Z"`
- Logical: `AND`, `OR`
- Metadata: `metadata.key="value"`
- Labels: `labels.env="prod"`

**Pagination**:
```python
page_token = None
all_artifacts = []

while True:
    response = aiplatform.Artifact.list(
        filter='schema_title="system.Model"',
        page_size=100,
        page_token=page_token
    )
    all_artifacts.extend(response.artifacts)

    if not response.next_page_token:
        break
    page_token = response.next_page_token
```

### Complex Lineage Queries

**Find all models trained on specific dataset**:
```python
# Get dataset artifact
dataset = aiplatform.Artifact(artifact_name=DATASET_NAME)

# Get executions that consumed this dataset
executions = dataset.get_downstream_executions()

# Get models produced by those executions
models = []
for execution in executions:
    output_artifacts = execution.get_output_artifacts()
    models.extend([a for a in output_artifacts if a.schema_title == "system.Model"])

print(f"Found {len(models)} models trained on {dataset.display_name}")
```

**Trace model back to original data sources**:
```python
def trace_lineage(artifact, depth=0):
    """Recursively trace artifact lineage."""
    indent = "  " * depth
    print(f"{indent}{artifact.display_name} ({artifact.schema_title})")

    if depth < 5:  # Limit recursion
        for parent in artifact.get_upstream_artifacts():
            trace_lineage(parent, depth + 1)

model = aiplatform.Artifact(artifact_name=MODEL_NAME)
print("Model lineage:")
trace_lineage(model)
```

## Best Practices

### Metadata Organization

From [Vertex AI Best Practices 2025](https://skywork.ai/blog/vertex-ai-best-practices-governance-quotas-collaboration/) (accessed 2025-02-03):

**1. Use consistent naming conventions**:
```python
# Good: Semantic, versioned names
dataset = aiplatform.Artifact.create(
    display_name="imagenet-train-v3",  # Clear, versioned
    schema_title="system.Dataset"
)

# Bad: Opaque, unversioned names
dataset = aiplatform.Artifact.create(
    display_name="data123",  # Unclear
    schema_title="system.Dataset"
)
```

**2. Add rich metadata**:
```python
model = aiplatform.Artifact.create(
    schema_title="system.Model",
    display_name="resnet-50-v1",
    metadata={
        # Model details
        "architecture": "resnet50",
        "framework": "pytorch",
        "framework_version": "2.0.1",

        # Training details
        "training_job_id": "12345",
        "training_duration_minutes": 120,
        "gpu_type": "a100",

        # Performance
        "accuracy": 0.94,
        "f1_score": 0.92,
        "auc": 0.96,

        # Governance
        "owner": "ml-team@company.com",
        "approval_status": "pending",
        "production_ready": False
    }
)
```

**3. Use contexts for organization**:
```python
# Organize by project
project_context = aiplatform.Context.create(
    schema_title="system.Project",
    display_name="recommendation-system",
    metadata={"team": "ml-recommendations", "priority": "high"}
)

# Organize by experiment
experiment_context = aiplatform.Context.create(
    schema_title="system.Experiment",
    display_name="transformer-vs-lstm",
    metadata={"hypothesis": "Transformers outperform LSTMs on long sequences"}
)
```

### Performance Optimization

**Batch operations when possible**:
```python
# Instead of multiple single queries
artifacts = []
for name in artifact_names:
    artifacts.append(aiplatform.Artifact(artifact_name=name))

# Use filter to batch query
artifacts = aiplatform.Artifact.list(
    filter=f'display_name IN ({",".join(artifact_names)})'
)
```

**Use pagination for large result sets**:
```python
# Avoid loading all results at once
artifacts = aiplatform.Artifact.list(
    filter='schema_title="system.Model"',
    page_size=100  # Process in chunks
)
```

### Governance and Compliance

From [Build responsible models with Model Cards](https://medium.com/google-cloud/build-responsible-models-with-model-cards-and-vertex-ai-pipelines-8cbf451e7632) (accessed 2025-02-03):

**Track model approval workflows**:
```python
model = aiplatform.Artifact.create(
    schema_title="system.Model",
    display_name="credit-scoring-v2",
    metadata={
        "approval_status": "pending_review",
        "reviewer": "data-science-lead@company.com",
        "compliance_checklist": {
            "bias_audit": False,
            "fairness_metrics": False,
            "explainability": False,
            "documentation": False
        }
    }
)

# Update after review
model.update(metadata={
    "approval_status": "approved",
    "reviewed_by": "data-science-lead@company.com",
    "reviewed_date": "2025-02-03",
    "compliance_checklist": {
        "bias_audit": True,
        "fairness_metrics": True,
        "explainability": True,
        "documentation": True
    }
})
```

## Cost Optimization

From [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) (accessed 2025-02-03):

**Pricing**: $10 per GiB of metadata stored

**Optimization strategies**:
1. **Limit metadata size**: Store URIs to large objects, not the objects themselves
2. **Delete old artifacts**: Clean up experiment artifacts after analysis
3. **Use separate stores**: Isolate production from experimentation metadata
4. **Compress metadata**: Use concise keys and values

```python
# Good: Lightweight metadata
model = aiplatform.Artifact.create(
    uri="gs://bucket/model.pkl",  # URI only
    metadata={"acc": 0.94}  # Concise keys
)

# Bad: Heavy metadata
model = aiplatform.Artifact.create(
    metadata={
        "training_data": "<entire dataset>",  # Don't embed data!
        "model_weights": "<entire model>",  # Use URI instead!
        "very_long_description_key_name": "..."  # Use short keys
    }
)
```

## Sources

**Google Cloud Documentation:**
- [Introduction to Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/introduction) (accessed 2025-02-03)
- [Track Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/tracking) (accessed 2025-02-03)
- [Analyze Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/analyzing) (accessed 2025-02-03)
- [Manage Vertex ML Metadata](https://docs.cloud.google.com/vertex-ai/docs/ml-metadata/managing-metadata) (accessed 2025-02-03)
- [Track the lineage of pipeline artifacts](https://docs.cloud.google.com/vertex-ai/docs/pipelines/lineage) (accessed 2025-02-03)
- [ML metadata artifact types](https://cloud.google.com/vertex-ai/docs/pipelines/artifact-types) (accessed 2025-02-03)
- [REST Resource: MetadataStore](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.metadataStores) (accessed 2025-02-03)
- [REST Resource: artifacts](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.metadataStores.artifacts) (accessed 2025-02-03)

**Community Resources:**
- [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) - Sascha Heyer (accessed 2025-02-03)
- [Build responsible models with Model Cards](https://medium.com/google-cloud/build-responsible-models-with-model-cards-and-vertex-ai-pipelines-8cbf451e7632) - Ivan Nardini (accessed 2025-02-03)
- [The nuts and bolts of VertexAI](https://leftasexercise.com/2024/02/18/the-nuts-and-bolts-of-vertexai-metadata-logging-and-metrics/) - LeftAsExercise (accessed 2025-02-03)
- [Vertex AI Best Practices 2025](https://skywork.ai/blog/vertex-ai-best-practices-governance-quotas-collaboration/) - Skywork.ai (accessed 2025-02-03)

**Code Labs & Samples:**
- [Using Vertex ML Metadata with Pipelines](https://codelabs.developers.google.com/vertex-mlmd-pipelines) (accessed 2025-02-03)
- [vertex-pipelines-ml-metadata.ipynb](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/ml_metadata/vertex-pipelines-ml-metadata.ipynb) (accessed 2025-02-03)
- [Build Vertex AI Experiment lineage](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/experiments/build_model_experimentation_lineage_with_prebuild_code.ipynb) (accessed 2025-02-03)
- [googleapis/python-aiplatform](https://github.com/googleapis/python-aiplatform) - GitHub (accessed 2025-02-03)

**Additional References:**
- [Better ML Engineering with ML Metadata](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial) - TensorFlow (accessed 2025-02-03)
