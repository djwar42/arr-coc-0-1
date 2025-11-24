# Vertex AI Pipelines with Kubeflow (KFP)

Comprehensive guide to building ML workflows with Vertex AI Pipelines using Kubeflow Pipelines SDK.

---

## Overview

**Vertex AI Pipelines** is Google Cloud's managed service for orchestrating ML workflows. It supports two frameworks:

- **Kubeflow Pipelines (KFP)** - Framework-agnostic, flexible orchestration (recommended for most use cases)
- **TensorFlow Extended (TFX)** - TensorFlow-specific, standardized ML components

This guide focuses on **KFP** due to its flexibility and support for diverse ML libraries (PyTorch, TensorFlow, Scikit-Learn, etc.).

**Key Benefits:**
- Serverless execution (no cluster management)
- Automatic lineage tracking and metadata storage
- Built-in caching to avoid redundant computations
- Integration with Vertex AI services (Training, Model Registry, Endpoints)
- Visual DAG representation in Vertex AI console

From [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) (accessed 2025-02-03):
> "With KFP you can author components and pipelines using the KFP Python SDK, compile them, and run them on Vertex AI Pipelines or other KFP-conformant backends."

---

## Section 1: Pipeline Components Overview (~100 lines)

### What are Pipeline Components?

Components are **remote function definitions** - they specify inputs, contain user logic, and produce outputs. When instantiated with parameters, they become **tasks**.

**Component Types:**

1. **Lightweight Python Components** - Pure Python functions (easiest to author)
2. **Containerized Python Components** - Python functions with complex dependencies
3. **Container Components** - Arbitrary container definitions (maximum flexibility)
4. **Importer Components** - Special pre-built component for importing external artifacts

From [Create Components](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/) (accessed 2025-02-03):
> "Components are the building blocks of KFP pipelines. A component is a remote function definition; it specifies inputs, has user-defined logic in its body, and can create outputs."

### Lightweight Python Component Example

```python
from kfp import dsl

@dsl.component
def preprocess_data(
    data_path: str,
    output_data: dsl.Output[dsl.Dataset]
):
    """Preprocess raw data."""
    import pandas as pd

    # Load and preprocess
    df = pd.read_csv(data_path)
    df_clean = df.dropna()

    # Save output
    df_clean.to_csv(output_data.path, index=False)
```

**Key Features:**
- Decorated with `@dsl.component`
- Self-contained (all imports inside function)
- Inputs: Python primitives (str, int, float, bool)
- Outputs: `dsl.Output[Type]` for artifacts (Dataset, Model, Metrics)

### Containerized Python Component

For complex dependencies (specific package versions, GPU libraries):

```python
from kfp import dsl

@dsl.component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310",
    packages_to_install=["scikit-learn==1.3.0"]
)
def train_model(
    train_data: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model]
):
    """Train model with specific TensorFlow version."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    # Training logic here
    pass
```

**Use Cases:**
- Specific framework versions (TensorFlow 2.17, PyTorch 2.1, etc.)
- GPU-accelerated containers
- Pre-installed system libraries
- Custom build environments

### Container Component

Maximum flexibility - define any container:

```python
from kfp import dsl

@dsl.container_component
def custom_processing():
    return dsl.ContainerSpec(
        image="gcr.io/my-project/custom-processor:v1",
        command=["python", "process.py"],
        args=["--input", dsl.InputPath("dataset"), "--output", dsl.OutputPath("results")]
    )
```

**Use Cases:**
- Non-Python workloads (R, Julia, Scala)
- Legacy container images
- Complex multi-step shell scripts
- Third-party tools (MLflow, DVC)

---

## Section 2: Building Pipelines - Composing Components (~150 lines)

### Pipeline Decorator

Pipelines are Python functions decorated with `@dsl.pipeline`:

```python
from kfp import dsl

@dsl.pipeline(
    name="fashion-mnist-classification",
    pipeline_root="gs://my-bucket/pipelines"
)
def ml_training_pipeline(
    data_path: str,
    learning_rate: float = 0.001,
    epochs: int = 10
):
    """Complete ML training pipeline."""

    # 1. Preprocess
    preprocess_task = preprocess_data(data_path=data_path)

    # 2. Train (depends on preprocess)
    train_task = train_model(
        train_data=preprocess_task.outputs['processed_data'],
        learning_rate=learning_rate,
        epochs=epochs
    )

    # 3. Evaluate (depends on train)
    eval_task = evaluate_model(
        model=train_task.outputs['model'],
        test_data=preprocess_task.outputs['test_data']
    )

    # 4. Register model if accuracy > threshold
    with dsl.Condition(eval_task.outputs['accuracy'] > 0.9):
        register_model(
            model=train_task.outputs['model'],
            metrics=eval_task.outputs['metrics']
        )
```

**Key Concepts:**

1. **Data Passing**: Outputs from one task become inputs to next
2. **Task Dependencies**: Implicit from data flow (train depends on preprocess)
3. **Pipeline Parameters**: Input arguments passed at runtime
4. **Conditional Execution**: `dsl.Condition()` for branching logic

From [Compose Components into Pipelines](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/compose-components-into-pipelines/) (accessed 2025-02-03):
> "Data passing creates implicit task dependencies. When you pass an output from one task as an input to another, KFP automatically understands that the second task must wait for the first task to complete."

### Task Configuration Methods

Configure resources, retries, caching per task:

```python
@dsl.pipeline
def resource_optimized_pipeline():

    # GPU training
    train_task = train_model()
    train_task.set_accelerator_type('NVIDIA_TESLA_T4')
    train_task.set_accelerator_limit(1)
    train_task.set_memory_limit('16G')
    train_task.set_cpu_limit('4')

    # Disable caching for experiment
    train_task.set_caching_options(enable_caching=False)

    # Retry on failure
    train_task.set_retry(num_retries=3, backoff_duration='60s')
```

**Available Task Methods:**
- `.set_accelerator_type()` - GPU type (T4, V100, A100)
- `.set_accelerator_limit()` - Number of GPUs
- `.set_memory_limit()` - RAM allocation
- `.set_cpu_limit()` - CPU cores
- `.set_caching_options()` - Enable/disable caching
- `.set_retry()` - Retry policy
- `.set_display_name()` - Custom UI name

### Parameter Passing Patterns

**1. Direct parameter passing:**
```python
train_task = train_model(learning_rate=0.001)
```

**2. Pipeline parameter to component:**
```python
@dsl.pipeline
def pipeline(lr: float):
    train_task = train_model(learning_rate=lr)
```

**3. Artifact passing (Dataset, Model, Metrics):**
```python
eval_task = evaluate(
    model=train_task.outputs['model'],  # Model artifact
    data=preprocess_task.outputs['dataset']  # Dataset artifact
)
```

**4. Chaining outputs:**
```python
result = component_c(
    input_a=component_a.output,
    input_b=component_b.output
)
```

### Control Flow

**Conditional execution:**
```python
with dsl.Condition(eval_task.outputs['accuracy'] > 0.9, name='high-accuracy'):
    deploy_task = deploy_model(model=train_task.outputs['model'])
```

**Parallel loops (ParallelFor):**
```python
from kfp import dsl

@dsl.pipeline
def parallel_training():
    hyperparams = [
        {'lr': 0.001, 'batch': 32},
        {'lr': 0.01, 'batch': 64},
        {'lr': 0.1, 'batch': 128}
    ]

    with dsl.ParallelFor(hyperparams) as hp:
        train_task = train_model(
            learning_rate=hp.lr,
            batch_size=hp.batch
        )
```

From [Kubeflow Pipelines KFP 2024](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) (accessed 2025-02-03):
> "The output of previous steps in the pipelines (all or a part) can be used as input for next step in the pipeline. The actual output is saved, upon each step execution, on a dedicated bucket on GCP, specified as pipeline_root."

---

## Section 3: Execution Caching Strategies (~100 lines)

### How Caching Works

Vertex AI Pipelines automatically caches component execution results. When a component runs with:
- Same inputs
- Same parameters
- Same component code

The cached result is reused instead of re-executing.

From [Use Caching](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/caching/) (accessed 2025-02-03):
> "Caching in KFP is a feature that allows you to cache the results of a component execution and reuse them in subsequent runs. When caching is enabled for a component, KFP will reuse the component's outputs if the component is executed again with the same inputs and parameters."

**Visual Indicator:**
Cached tasks show a green "arrow from cloud" icon in Vertex AI UI.

**Cache Storage:**
- Results stored in Vertex ML Metadata
- No TTL (time to live) - persists until manually deleted
- Shared across pipeline runs

### Controlling Caching

**Default Behavior: Caching ENABLED**

All components cache by default. Disable per-task:

```python
@dsl.pipeline
def pipeline_with_caching():
    # Cached (default)
    preprocess_task = preprocess_data()

    # NOT cached (disabled)
    train_task = train_model()
    train_task.set_caching_options(enable_caching=False)

    # Cached again (default)
    eval_task = evaluate_model()
```

**Disable caching for entire pipeline run:**

```python
from kfp.client import Client

client = Client()
client.create_run_from_pipeline_func(
    pipeline_func=ml_pipeline,
    arguments={'data_path': 'gs://...'},
    enable_caching=False  # Override all component settings
)
```

**Disable caching globally at compile time:**

```bash
# Via CLI flag
kfp dsl compile \
  --py pipeline.py \
  --output pipeline.yaml \
  --disable-execution-caching-by-default

# Via environment variable
export KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT=true
python compile_pipeline.py
```

**Environment Variable Behavior:**
- Must be set BEFORE importing KFP components
- Values: `true`, `1`, `yes` (disable), `false` or absent (enable)
- Works with both CLI and `Compiler().compile()`

### When to Disable Caching

**Always disable for:**
- **Experiment tracking** - Want fresh training runs with different hyperparameters
- **Non-deterministic components** - Random data augmentation, sampling
- **Time-sensitive operations** - Fetching latest data from APIs
- **External state changes** - Database queries, model endpoints

**Keep enabled for:**
- **Expensive preprocessing** - Hours-long data transformations
- **Deterministic computations** - Feature engineering, validation
- **Development/debugging** - Iterate quickly without re-running everything

**Example: Selective caching in training pipeline**

```python
@dsl.pipeline
def experiment_pipeline():
    # Cache expensive preprocessing (deterministic)
    preprocess_task = preprocess_data()  # Caching ON

    # Don't cache training (want fresh experiments)
    train_task = train_model()
    train_task.set_caching_options(enable_caching=False)

    # Cache evaluation (deterministic given same model)
    eval_task = evaluate_model()  # Caching ON
```

### Cache Invalidation

Cache is invalidated when:
- Component code changes
- Input parameters change
- Input artifact URIs change
- Component definition changes

**Manual cache clearing:**
Cache entries stored in Vertex ML Metadata can be deleted via:
- Vertex AI console (ML Metadata section)
- `gcloud` commands
- Python SDK (`aiplatform.metadata` API)

---

## Section 4: Vertex AI vs Standalone KFP (~100 lines)

### Key Differences

| Feature | Vertex AI Pipelines | Standalone KFP |
|---------|-------------------|---------------|
| **Infrastructure** | Fully managed, serverless | Self-managed Kubernetes cluster |
| **Setup** | No setup required | Install KFP on GKE |
| **Scaling** | Automatic | Manual cluster configuration |
| **Cost** | Pay-per-execution | Pay for cluster uptime |
| **Integration** | Native Vertex AI services | Manual integration |
| **Metadata** | Vertex ML Metadata | KFP Metadata Store |
| **IAM** | GCP IAM | Kubernetes RBAC |

### Vertex AI Integration Benefits

**1. Vertex AI Custom Training:**
```python
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

train_op = create_custom_training_job_from_component(
    component_spec=train_model,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

**2. Vertex AI Model Registry:**
```python
from google_cloud_pipeline_components.v1.model import ModelUploadOp

upload_model = ModelUploadOp(
    project=project_id,
    display_name="my-classifier",
    artifact_uri=train_task.outputs['model'],
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest"
)
```

**3. Vertex AI Endpoints (deployment):**
```python
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

create_endpoint = EndpointCreateOp(
    project=project_id,
    display_name="classifier-endpoint"
)

deploy_model = ModelDeployOp(
    model=upload_model.outputs['model'],
    endpoint=create_endpoint.outputs['endpoint'],
    machine_type="n1-standard-2"
)
```

### Google Cloud Pipeline Components (GCPC)

Pre-built components for Vertex AI services:

```python
from google_cloud_pipeline_components.v1 import aiplatform

# AutoML training
automl_task = aiplatform.AutoMLTabularTrainingJobRunOp(
    project=project_id,
    display_name="automl-tabular",
    optimization_prediction_type="classification",
    dataset=dataset_task.outputs['dataset']
)

# Batch prediction
batch_pred = aiplatform.ModelBatchPredictOp(
    project=project_id,
    model=model_task.outputs['model'],
    job_display_name="batch-prediction",
    gcs_source_uris=["gs://my-bucket/input.csv"],
    gcs_destination_output_uri_prefix="gs://my-bucket/output/"
)
```

From [Introduction to Google Cloud Pipeline Components](https://cloud.google.com/vertex-ai/docs/pipelines/components-introduction) (accessed 2025-02-03):
> "The Google Cloud Pipeline Components SDK provides a set of prebuilt Kubeflow Pipelines components that are production quality, performant, and easy to use."

### Portability Considerations

**KFP pipelines are portable across:**
- Vertex AI Pipelines
- Standalone KFP on GKE
- Other KFP-conformant backends

**To maintain portability:**
- Use generic KFP SDK components (not GCPC) for core logic
- Abstract cloud-specific operations (storage, logging)
- Use environment variables for cloud-specific configs

**Example: Portable component**
```python
import os

@dsl.component
def portable_preprocess(input_path: str, output: dsl.Output[dsl.Dataset]):
    """Works on Vertex AI and standalone KFP."""
    # Use environment variable for cloud storage
    storage_backend = os.getenv('STORAGE_BACKEND', 'gcs')

    if storage_backend == 'gcs':
        from google.cloud import storage
        # GCS logic
    elif storage_backend == 's3':
        import boto3
        # S3 logic
```

### When to Use Vertex AI vs Standalone

**Choose Vertex AI Pipelines when:**
- Building on GCP
- Want zero infrastructure management
- Need tight Vertex AI integration
- Prefer pay-per-execution pricing
- Small to medium team

**Choose Standalone KFP when:**
- Multi-cloud or on-prem deployment
- Need full Kubernetes control
- High-volume workloads (cluster more cost-effective)
- Custom backend integrations
- Large team with Kubernetes expertise

---

## Section 5: CI/CD Integration and Best Practices (~50 lines)

### Pipeline Compilation

Compile Python pipeline to IR YAML:

```bash
# CLI
kfp dsl compile \
  --py pipeline.py \
  --output pipeline.yaml

# Python SDK
from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path='pipeline.yaml'
)
```

**IR YAML** (Intermediate Representation) is portable across KFP backends.

### CI/CD Pipeline Pattern

**1. Version control pipelines:**
```bash
my-ml-project/
├── pipelines/
│   ├── training_pipeline.py
│   ├── batch_prediction_pipeline.py
│   └── __init__.py
├── components/
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluation.py
├── tests/
│   ├── test_components.py
│   └── test_pipelines.py
├── requirements.txt
└── cloudbuild.yaml
```

**2. Cloud Build automation:**
```yaml
# cloudbuild.yaml
steps:
  - name: 'python:3.10'
    entrypoint: 'pip'
    args: ['install', '-r', 'requirements.txt']

  - name: 'python:3.10'
    entrypoint: 'pytest'
    args: ['tests/']

  - name: 'python:3.10'
    entrypoint: 'python'
    args: ['-m', 'kfp.compiler', 'pipelines/training_pipeline.py', 'pipeline.yaml']

  - name: 'gcr.io/cloud-builders/gsutil'
    args: ['cp', 'pipeline.yaml', 'gs://my-bucket/pipelines/training/latest.yaml']
```

**3. Trigger pipeline from Cloud Build:**
```python
from kfp.client import Client

client = Client(host='https://[PROJECT_ID].pipelines.googleusercontent.com')
run = client.create_run_from_pipeline_package(
    pipeline_file='gs://my-bucket/pipelines/training/latest.yaml',
    arguments={'data_path': 'gs://my-bucket/data/latest'},
    experiment_name='training-prod'
)
```

### Best Practices

**Component design:**
- Keep components single-purpose and reusable
- Use type hints for all inputs/outputs
- Include docstrings with Args/Returns
- Handle errors gracefully (don't fail silently)

**Pipeline structure:**
- Separate data prep, training, evaluation, deployment
- Use meaningful task names (`.set_display_name()`)
- Add conditional deployment based on metrics
- Keep pipeline logic in Python (not YAML)

**Resource management:**
- Right-size machine types (don't over-provision)
- Use preemptible VMs for fault-tolerant tasks
- Set appropriate timeouts
- Monitor costs per pipeline run

**Metadata and lineage:**
- Log hyperparameters as pipeline parameters
- Track model versions in Model Registry
- Use consistent naming conventions
- Tag experiments for organization

From [Vertex AI Pipelines Introduction](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-02-03):
> "When you run a pipeline using Vertex AI Pipelines, all parameters and artifact metadata consumed and generated by the pipeline are stored in Vertex ML Metadata."

---

## Sources

**Kubeflow Documentation:**
- [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) (accessed 2025-02-03)
- [Create Components](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/) (accessed 2025-02-03)
- [Compose Components into Pipelines](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/compose-components-into-pipelines/) (accessed 2025-02-03)
- [Use Caching](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/caching/) (accessed 2025-02-03)

**Google Cloud Documentation:**
- [Vertex AI Pipelines Build Pipeline](https://docs.cloud.google.com/vertex-ai/docs/pipelines/build-pipeline) (accessed 2025-02-03)
- [Introduction to Google Cloud Pipeline Components](https://docs.cloud.google.com/vertex-ai/docs/pipelines/components-introduction) (accessed 2025-02-03)
- [Configure Execution Caching](https://docs.cloud.google.com/vertex-ai/docs/pipelines/configure-caching) (accessed 2025-02-03)
- [Vertex AI Pipelines Introduction](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-02-03)

**Web Research:**
- [Building ML Pipelines with Vertex AI and Kubeflow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) by Gabriel Preda (accessed 2025-02-03)

**Additional References:**
- [Google Cloud Pipeline Components Documentation](https://google-cloud-pipeline-components.readthedocs.io/)
- [KFP SDK Reference](https://kubeflow-pipelines.readthedocs.io/)
- [Vertex AI Release Notes](https://docs.cloud.google.com/vertex-ai/docs/core-release-notes)
