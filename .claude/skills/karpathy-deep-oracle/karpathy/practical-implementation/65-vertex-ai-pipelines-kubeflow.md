# Vertex AI Pipelines + Kubeflow: Production ML Pipeline Orchestration

**Comprehensive guide to building, deploying, and orchestrating ML pipelines on Vertex AI using Kubeflow Pipelines SDK v2**

---

## Overview

Vertex AI Pipelines is Google Cloud's managed ML pipeline orchestration service that integrates Kubeflow Pipelines (KFP) for defining and executing multi-step ML workflows. It provides serverless pipeline execution with automatic infrastructure management, enabling teams to focus on ML logic rather than DevOps.

**Core Value Proposition:**
- Serverless execution (no cluster management required)
- Component-based architecture for reusability
- Integration with Vertex AI services (Model Registry, Feature Store, Endpoints)
- DAG-based orchestration with automatic dependency resolution
- Built-in artifact tracking and lineage
- Production-ready autoscaling and monitoring

From [Vertex AI Pipelines Documentation](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-01-13):
> "Vertex AI Pipelines lets you automate, monitor, and govern your machine learning (ML) systems in a serverless manner by using ML pipelines to orchestrate your ML workflows."

---

## Section 1: Architecture & Core Concepts (~100 lines)

### 1.1 Vertex AI Pipelines vs Kubeflow Pipelines

**Vertex AI Pipelines** is a managed service that runs Kubeflow Pipelines on Google Cloud infrastructure. The relationship:

**Kubeflow Pipelines (KFP):**
- Open-source ML workflow orchestration framework
- Originally designed for Kubernetes clusters
- Provides Python SDK for pipeline definition
- Framework-agnostic (supports TensorFlow, PyTorch, scikit-learn, etc.)

**Vertex AI Pipelines:**
- Managed execution of KFP pipelines
- No Kubernetes cluster management required
- Serverless infrastructure (auto-scaling, auto-provisioning)
- Native integration with Vertex AI services
- GCS-based artifact storage (no persistent volumes)

From [Migrate from Kubeflow Pipelines to Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/migrate-kfp) (accessed 2025-01-13):
> "Kubeflow Pipelines and Vertex AI Pipelines handle storage differently. In Kubeflow Pipelines you can make use of Kubernetes resources such as persistent volumes. In Vertex AI Pipelines, use Cloud Storage as your data storage solution."

**Key Differences:**

| Aspect | Kubeflow (Self-Managed) | Vertex AI Pipelines |
|--------|------------------------|---------------------|
| Infrastructure | Manual GKE cluster setup | Serverless (fully managed) |
| Storage | Persistent Volumes | GCS (Cloud Storage) |
| Cost | Cluster + compute | Pay-per-pipeline-run |
| Maintenance | Manual upgrades | Automatic |
| Integration | Custom | Native Vertex AI services |
| Portability | High (any Kubernetes) | GCP-specific |

From [Kubeflow vs Vertex AI Pipelines](https://traceroute.net/2025/09/25/kubeflow-vs-vertex-ai-pipelines/) (accessed 2025-01-13):
> "Kubeflow may be cost-effective for organizations already running Kubernetes, but the hidden cost of management is high."

### 1.2 Component-Based Architecture

Vertex AI Pipelines uses a component-based design where each step in the pipeline is a self-contained, reusable unit:

**Component Anatomy:**
```python
from kfp.v2.dsl import component, Output, Dataset

@component(base_image="python:3.10")
def preprocess_data(
    input_path: str,
    processed_data: Output[Dataset]
):
    """
    Standalone preprocessing component.

    Args:
        input_path: GCS path to raw data
        processed_data: Output artifact (Dataset)
    """
    import pandas as pd
    from google.cloud import storage

    # Load data
    df = pd.read_csv(input_path)

    # Preprocess
    df_processed = df.dropna()

    # Save output
    df_processed.to_csv(processed_data.path, index=False)
```

**Component Types:**

1. **Python Function-Based Components:**
   - Decorated with `@component`
   - Self-contained functions
   - Dependencies installed via `packages_to_install` parameter
   - Fast iteration (no container build required)

2. **Container-Based Components:**
   - Custom Docker images
   - Complex dependencies
   - Pre-built binaries/libraries
   - Slower iteration (requires container rebuild)

From [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) (accessed 2025-01-13):
> "Container based components are more suitable for complex tasks where there are many code dependencies in comparison to function based components... The latter runs more quickly as we do not need to build and deploy an image every time we edit our code."

### 1.3 Pipeline Execution Model

**DAG (Directed Acyclic Graph):**
- Components are nodes
- Data dependencies are edges
- Automatic parallel execution where possible
- Sequential execution when dependencies exist

**Pipeline Root:**
- GCS bucket for storing artifacts
- Persists all component inputs/outputs
- Enables caching and reuse
- Specified via `pipeline_root` parameter

**Artifact Passing:**
- Outputs saved to pipeline_root GCS bucket
- Paths passed to downstream components
- Automatic serialization/deserialization
- Supports typed artifacts (Dataset, Model, Metrics, etc.)

From [All you need to know to get started with Vertex AI Pipelines](https://www.artefact.com/blog/all-you-need-to-know-to-get-started-with-vertex-ai-pipelines/) (accessed 2025-01-13):
> "With kfp.dsl.condition, you can easily deploy a trained model and prepare to reuse it later with some code logic... Mix it up with a great CICD, you will operate your ML model lifecycle without a hitch."

---

## Section 2: Python SDK & Component Development (~120 lines)

### 2.1 Kubeflow Pipelines SDK v2

**Installation:**
```bash
pip install kfp google-cloud-aiplatform google-cloud-pipeline-components
```

**Version Requirements:**
- KFP SDK: v2.x (latest: 2.9+)
- Python: 3.7-3.11
- Vertex AI SDK: 1.x

From [googleapis/python-aiplatform](https://github.com/googleapis/python-aiplatform) (accessed 2025-01-13):
> "The Vertex AI SDK for Python allows you train Custom and AutoML Models. You can train custom models using a custom Python script, custom Python package, or container."

### 2.2 Component Decorator & Parameters

**Basic Component:**
```python
from kfp.v2.dsl import component

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_model(
    train_data_path: str,
    model_output: Output[Model],
    learning_rate: float = 0.01,
    epochs: int = 10
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load data
    df = pd.read_csv(train_data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_output.path)
```

**Component Parameters:**

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `base_image` | Docker image to run component | `"python:3.10"` |
| `packages_to_install` | Pip packages | `["pandas", "numpy"]` |
| `output_component_file` | Save compiled component | `"component.yaml"` |
| `install_kfp_package` | Include KFP in environment | `False` (default: True) |

**Using Custom Base Images:**
```python
@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310")
def tensorflow_component(...):
    import tensorflow as tf
    # TensorFlow already installed in base image
```

From [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) (accessed 2025-01-13):
> "Our solution to run both complex and simple components the same way is to work with an overwritten version of the default base image. Inside this altered base image we installed all our codes as a package."

### 2.3 Input/Output Artifacts

**Artifact Types:**
```python
from kfp.v2.dsl import (
    Input, Output,
    Dataset, Model, Metrics, Artifact
)

@component
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import classification_report

    # Load inputs
    df = pd.read_csv(test_data.path)
    trained_model = joblib.load(model.path)

    # Evaluate
    predictions = trained_model.predict(df.drop('target', axis=1))
    report = classification_report(df['target'], predictions, output_dict=True)

    # Save metrics
    with open(metrics.path, 'w') as f:
        json.dump(report, f)
```

**Artifact Semantics:**
- `Dataset`: Tabular/structured data
- `Model`: Trained model files
- `Metrics`: Evaluation results
- `Artifact`: Generic artifact type
- `HTML`: HTML visualization
- `Markdown`: Markdown documentation

**Path Handling:**
- Input artifacts: Access via `.path` attribute
- Output artifacts: Write to `.path` location
- Automatic GCS upload/download
- Metadata tracked by Vertex AI

### 2.4 Container-Based Components

**When to Use:**
- Complex dependencies (C++ libraries, CUDA)
- Pre-built binaries
- Large model files bundled in image
- Consistent environment across runs

**Example:**
```python
from kfp.v2.dsl import container_component

@container_component
def custom_training_component(
    data_path: str,
    model_output: Output[Model]
):
    return ContainerSpec(
        image="gcr.io/my-project/custom-trainer:v1.0",
        command=["python", "train.py"],
        args=[
            "--data", data_path,
            "--output", model_output.path
        ]
    )
```

**Building Custom Images:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python", "train.py"]
```

---

## Section 3: Pipeline Orchestration (~100 lines)

### 3.1 Pipeline Definition

**Complete Pipeline Example:**
```python
from kfp.v2.dsl import pipeline

@pipeline(
    name="fashion-mnist-classifier",
    pipeline_root="gs://my-bucket/pipelines/fashion-mnist"
)
def training_pipeline(
    project: str,
    location: str,
    data_bucket: str,
    train_file: str,
    test_file: str,
    learning_rate: float = 0.001,
    epochs: int = 20
):
    """
    End-to-end Fashion MNIST training pipeline.

    Args:
        project: GCP project ID
        location: GCP region (e.g., 'us-central1')
        data_bucket: GCS bucket containing data
        train_file: Training data filename
        test_file: Test data filename
        learning_rate: Model learning rate
        epochs: Training epochs
    """

    # Step 1: Preprocess data
    preprocess_task = preprocess_data(
        data_bucket=data_bucket,
        train_file=train_file,
        test_file=test_file
    )

    # Step 2: Train model
    train_task = train_model(
        processed_train_data=preprocess_task.outputs['processed_train_data'],
        train_labels=preprocess_task.outputs['train_labels'],
        learning_rate=learning_rate,
        epochs=epochs
    )

    # Step 3: Evaluate model
    eval_task = evaluate_model(
        processed_test_data=preprocess_task.outputs['processed_test_data'],
        test_labels=preprocess_task.outputs['test_labels'],
        model=train_task.outputs['model']
    )

    # Step 4: Register model (conditional on accuracy threshold)
    with dsl.Condition(eval_task.outputs['accuracy'] >= 0.90):
        register_model(
            project=project,
            location=location,
            model=train_task.outputs['model'],
            metrics=eval_task.outputs['metrics']
        )
```

From [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) (accessed 2025-01-13):
> "Each step is implemented by a Kubeflow pipeline component. The output of previous steps in the pipelines (all or a part) can be used as input for next step in the pipeline."

### 3.2 Pipeline Compilation

**Compile to JSON:**
```python
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path="pipeline.json"
)
```

**Compilation Process:**
1. Validates pipeline DAG (no cycles)
2. Resolves component dependencies
3. Generates JSON specification
4. Includes component definitions
5. Embeds artifact schemas

**Compiled Pipeline Structure:**
- Components section (all component definitions)
- DAG section (task dependencies)
- Runtime parameters (pipeline inputs)
- Deployment config (resource requests)

### 3.3 Pipeline Execution

**Submit via Vertex AI SDK:**
```python
from google.cloud import aiplatform

aiplatform.init(
    project="my-gcp-project",
    location="us-central1"
)

job = aiplatform.PipelineJob(
    display_name="fashion-mnist-training",
    template_path="pipeline.json",
    pipeline_root="gs://my-bucket/pipelines/fashion-mnist",
    parameter_values={
        "project": "my-gcp-project",
        "location": "us-central1",
        "data_bucket": "my-data-bucket",
        "train_file": "fashion_mnist_train.csv",
        "test_file": "fashion_mnist_test.csv",
        "learning_rate": 0.001,
        "epochs": 20
    },
    enable_caching=True
)

job.submit()
```

**Execution Options:**

| Option | Purpose | Default |
|--------|---------|---------|
| `enable_caching` | Reuse cached component outputs | `True` |
| `labels` | Metadata labels for tracking | `{}` |
| `encryption_spec_key_name` | CMEK encryption | `None` |
| `service_account` | Custom service account | Default compute SA |

**Monitoring Execution:**
```python
# Wait for completion
job.wait()

# Check status
print(f"Pipeline state: {job.state}")

# Get output artifacts
artifacts = job.get_output_artifacts()
```

### 3.4 Parallel Execution & Conditional Logic

**Parallel Component Execution:**
```python
@pipeline(name="parallel-training")
def multi_model_pipeline(categories: list):
    """Train separate models for each category in parallel."""

    for category in categories:
        # Each iteration runs in parallel
        train_task = train_category_model(category=category)
        eval_task = evaluate_category_model(
            model=train_task.outputs['model'],
            category=category
        )
```

From [All you need to know to get started with Vertex AI Pipelines](https://www.artefact.com/blog/all-you-need-to-know-to-get-started-with-vertex-ai-pipelines/) (accessed 2025-01-13):
> "Doing this optimally would mean parallelizing the different training workflows to gain time and optimize resources. With Vertex Pipelines and Kubeflow the effort is minimal by design; it will only cost you to write a for loop."

**Conditional Deployment:**
```python
from kfp.v2 import dsl

@pipeline(name="conditional-deploy")
def pipeline_with_conditions(accuracy_threshold: float = 0.90):

    eval_task = evaluate_model(...)

    # Only deploy if accuracy meets threshold
    with dsl.Condition(eval_task.outputs['accuracy'] >= accuracy_threshold):
        deploy_task = deploy_to_endpoint(
            model=eval_task.outputs['model']
        )
```

---

## Section 4: ARR-COC Production Integration (~80 lines)

### 4.1 Multi-Stage VLM Training Pipelines

**ARR-COC Training Workflow:**
```python
@pipeline(
    name="arr-coc-vlm-training",
    pipeline_root="gs://arr-coc-bucket/pipelines"
)
def arr_coc_pipeline(
    project: str,
    base_model: str = "Qwen/Qwen2-VL-7B",
    texture_channels: int = 13,
    num_patches: int = 200,
    lod_min: int = 64,
    lod_max: int = 400
):
    """
    Complete ARR-COC VLM training pipeline.

    Stages:
    1. Texture array preprocessing (RGB, LAB, Sobel, etc.)
    2. Relevance scorer training (propositional/perspectival/participatory)
    3. Quality adapter training (procedural knowing)
    4. End-to-end integration training
    5. Evaluation on VQA benchmarks
    6. Model registry and deployment
    """

    # Stage 1: Prepare texture arrays
    texture_task = prepare_texture_arrays(
        num_channels=texture_channels,
        patch_grid_size=14  # For 196 patches (14x14)
    )

    # Stage 2: Train relevance scorers
    propositional_task = train_propositional_scorer(
        texture_data=texture_task.outputs['texture_arrays']
    )

    perspectival_task = train_perspectival_scorer(
        texture_data=texture_task.outputs['texture_arrays']
    )

    participatory_task = train_participatory_scorer(
        texture_data=texture_task.outputs['texture_arrays'],
        base_model=base_model
    )

    # Stage 3: Train quality adapter
    adapter_task = train_quality_adapter(
        propositional_scorer=propositional_task.outputs['model'],
        perspectival_scorer=perspectival_task.outputs['model'],
        participatory_scorer=participatory_task.outputs['model'],
        lod_range=(lod_min, lod_max)
    )

    # Stage 4: Integrate and fine-tune end-to-end
    integration_task = train_integrated_model(
        base_model=base_model,
        quality_adapter=adapter_task.outputs['model'],
        num_patches=num_patches
    )

    # Stage 5: Evaluate on VQA benchmarks
    eval_task = evaluate_vqa_performance(
        model=integration_task.outputs['model'],
        benchmarks=["VQAv2", "GQA", "TextVQA"]
    )

    # Stage 6: Conditional registration (accuracy > 0.75)
    with dsl.Condition(eval_task.outputs['vqav2_accuracy'] >= 0.75):
        register_task = register_arr_coc_model(
            project=project,
            model=integration_task.outputs['model'],
            metrics=eval_task.outputs['metrics']
        )
```

### 4.2 Hyperparameter Sweeps with Pipelines

**Parallel Hyperparameter Search:**
```python
@pipeline(name="arr-coc-hp-sweep")
def hyperparameter_sweep_pipeline(
    lod_ranges: list,  # [(64, 200), (64, 300), (64, 400)]
    num_patches_options: list  # [100, 150, 200]
):
    """
    Test multiple ARR-COC configurations in parallel.
    """

    for lod_min, lod_max in lod_ranges:
        for num_patches in num_patches_options:
            # Each combination trains in parallel
            train_task = train_arr_coc_variant(
                lod_min=lod_min,
                lod_max=lod_max,
                num_patches=num_patches
            )

            eval_task = evaluate_variant(
                model=train_task.outputs['model'],
                config_name=f"lod{lod_min}-{lod_max}_patches{num_patches}"
            )

    # Aggregate results across all runs
    aggregate_task = aggregate_sweep_results()
```

### 4.3 Model Registry Integration

**Registering ARR-COC Models:**
```python
@component(base_image="python:3.10")
def register_arr_coc_model(
    project: str,
    location: str,
    model: Input[Model],
    metrics: Input[Metrics]
):
    """Register ARR-COC model in Vertex AI Model Registry."""
    from google.cloud import aiplatform
    import json

    aiplatform.init(project=project, location=location)

    # Load metrics
    with open(metrics.path, 'r') as f:
        eval_metrics = json.load(f)

    # Register with metadata
    uploaded_model = aiplatform.Model.upload(
        artifact_uri=f"{model.uri}/saved_model",
        display_name="arr-coc-vlm",
        labels={
            "framework": "pytorch",
            "model_type": "vlm",
            "architecture": "arr-coc"
        },
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-2:latest"
    )

    # Add evaluation metrics to description
    uploaded_model.update(
        description=f"ARR-COC VLM Model\n\nMetrics:\n{json.dumps(eval_metrics, indent=2)}"
    )

    print(f"Model registered: {uploaded_model.resource_name}")
```

From [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) (accessed 2025-01-13):
> "The Vertex AI Model Registry serves as a centralized place for model management, allowing collaboration (multiple teams can use a model, once properly registered), scalability (supporting multiple models across various environments) and compliance (maintains a clear audit trail for the models)."

### 4.4 Production Deployment Automation

**Automated Model Deployment:**
```python
@pipeline(name="arr-coc-deploy")
def deployment_pipeline(
    model_version: str,
    endpoint_name: str = "arr-coc-production",
    traffic_split: dict = {"new": 10, "old": 90}  # Canary deployment
):
    """
    Deploy ARR-COC model to Vertex AI Endpoint with canary strategy.
    """

    # Validate model performance
    validation_task = validate_model_metrics(
        model_version=model_version,
        min_vqa_accuracy=0.75
    )

    # Deploy with traffic split (conditional on validation)
    with dsl.Condition(validation_task.outputs['passed'] == True):
        deploy_task = deploy_to_endpoint(
            model_version=model_version,
            endpoint_name=endpoint_name,
            traffic_split=traffic_split,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1
        )

        # Monitor deployment health
        monitor_task = monitor_endpoint_health(
            endpoint=deploy_task.outputs['endpoint'],
            duration_minutes=30
        )
```

---

## Sources

**Google Cloud Documentation:**
- [Vertex AI Pipelines Introduction](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) - accessed 2025-01-13
- [Build a pipeline](https://docs.cloud.google.com/vertex-ai/docs/pipelines/build-pipeline) - accessed 2025-01-13
- [Migrate from Kubeflow Pipelines to Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/migrate-kfp) - accessed 2025-01-13

**Technical Articles:**
- [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) - Gabriel Preda, Medium, December 2024 (accessed 2025-01-13)
- [All you need to know to get started with Vertex AI Pipelines](https://www.artefact.com/blog/all-you-need-to-know-to-get-started-with-vertex-ai-pipelines/) - Artefact, accessed 2025-01-13
- [Kubeflow vs Vertex AI Pipelines](https://traceroute.net/2025/09/25/kubeflow-vs-vertex-ai-pipelines/) - Traceroute.net, September 2025 (accessed 2025-01-13)

**GitHub Resources:**
- [googleapis/python-aiplatform](https://github.com/googleapis/python-aiplatform) - Official Vertex AI Python SDK (accessed 2025-01-13)

**Related Knowledge Files:**
- [30-vertex-ai-fundamentals.md](30-vertex-ai-fundamentals.md) - Vertex AI architecture and custom jobs
- [37-vertex-ai-complete-examples.md](37-vertex-ai-complete-examples.md) - End-to-end training examples
- [gcloud-cicd/00-pipeline-integration.md](../../gcloud-cicd/00-pipeline-integration.md) - CI/CD with Vertex AI Pipelines
