# Kubeflow ML Pipelines: Production ML Workflow Orchestration

**Complete guide to Kubeflow Pipelines for ML workflow automation, training operators, and production deployment**

---

## Overview

Kubeflow Pipelines is an open-source platform for implementing MLOps, providing a framework for building, deploying, and managing machine learning workflows in a scalable, repeatable, secure, and cloud-oriented manner on Kubernetes.

From [Implement MLOps with Kubeflow Pipelines](https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines) (Red Hat Developer, January 2024):

> "Kubeflow Pipelines is an open source platform for implementing MLOps, providing a framework for building, deploying, and managing machine learning workflows in a scalable, repeatable, secure, and cloud-oriented manner on Kubernetes. With the ability to drive agility and efficiency in the development and deployment of machine learning models, MLOps with Kubeflow Pipelines can also improve collaboration between data scientists and machine learning engineers."

**Key capabilities:**
- End-to-end ML pipeline orchestration
- Distributed training with Training Operators (PyTorchJob, TFJob)
- Hyperparameter tuning with Katib
- Model serving integration
- Cloud-native Kubernetes execution

---

## 1. Kubeflow Pipelines Architecture

### 1.1 Core Components

**Pipeline Definition** - Directed Acyclic Graph (DAG) of components:
```python
@kfp.dsl.pipeline(
    name='ml-training-pipeline',
    description='End-to-end ML training workflow'
)
def training_pipeline(
    data_path: str,
    model_name: str,
    learning_rate: float = 0.001
):
    # Component 1: Data preprocessing
    preprocess_task = preprocess_op(data_path=data_path)

    # Component 2: Model training
    train_task = train_op(
        processed_data=preprocess_task.output,
        learning_rate=learning_rate
    )

    # Component 3: Model evaluation
    eval_task = evaluate_op(
        model=train_task.outputs['model'],
        test_data=preprocess_task.outputs['test_data']
    )
```

**Pipeline Components** - Reusable building blocks:
- **Lightweight Python components** - Functions decorated with `@component`
- **Container components** - Custom Docker images
- **Prebuilt components** - Community-contributed (e.g., from Google Cloud AI)

**Pipeline Backend** - Execution infrastructure:
- **Argo Workflows** (legacy v1) - DAG execution engine
- **Tekton** (alternative) - Cloud-native CI/CD pipelines
- **ML Metadata** - Tracks artifacts, executions, lineage

From [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) (accessed 2025-11-13):

> "Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning workflows based on Docker containers. The Kubeflow Pipelines platform consists of a user interface for managing and tracking experiments, jobs, and runs; an engine for scheduling multi-step ML workflows; an SDK for defining and manipulating pipelines and components; and notebooks for interacting with the system using the SDK."

### 1.2 Kubeflow vs MLflow Comparison

From [Kubeflow vs MLflow: An In-Depth Comparison](https://skphd.medium.com/kubeflow-vs-mlflow-an-in-depth-comparison-for-mlops-pipelines-e2f0c0496a36) (Medium, January 2024):

| Aspect | Kubeflow | MLflow |
|--------|----------|--------|
| **Primary Focus** | Orchestration & pipeline automation | Experiment tracking & model registry |
| **Architecture** | Container-based (Kubernetes) | Python library (framework-agnostic) |
| **Deployment** | Full platform (requires K8s cluster) | Lightweight server (pip install) |
| **Distributed Training** | Native (Training Operators) | Requires external orchestration |
| **UI** | Full web UI for pipelines, experiments | Simple UI for tracking, artifacts |
| **Learning Curve** | Steep (Kubernetes knowledge required) | Gentle (Python-first) |
| **Best For** | Large-scale production ML at scale | Experiment tracking, small-to-medium teams |

From [MLflow vs Kubeflow Reddit Discussion](https://www.reddit.com/r/mlops/comments/1evza42/mlflow_vs_kubeflow/) (r/mlops, 2024):

> "MLFlow is an experiment and artifact registry (mainly good for experiment tracking). KubeFlow (usually people think of kubeflow pipelines when they talk about it) is for AI at scale and workflow automation."

**When to use Kubeflow:**
- Multi-GPU/multi-node distributed training required
- Complex ML pipelines with dependencies
- Kubernetes infrastructure already in place
- Need for training operator abstractions (PyTorchJob, TFJob)

**When to use MLflow:**
- Primarily tracking experiments and models
- Simpler deployment requirements
- Team prefers Python-centric tooling
- Integration with Databricks ecosystem

---

## 2. Kubeflow Pipelines SDK

### 2.1 Installation and Setup

**Install KFP SDK v2 (latest):**
```bash
pip install kfp==2.8.0
```

**Install KFP SDK for Tekton (Red Hat OpenShift AI):**
```bash
pip install kfp-tekton==1.5.9
```

From [Kubeflow Pipelines SDK Overview](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/sdk/sdk-overview/) (accessed 2025-11-13):

> "The Kubeflow Pipelines SDK provides a set of Python packages that you can use to specify and run your machine learning (ML) workflows. A pipeline is a description of an ML workflow, including all of the components that make up the steps in the workflow and how the components interact with each other."

### 2.2 Component Development Patterns

**Pattern 1: Lightweight Python Components**

From [Red Hat Developer - Kubeflow Pipelines Tutorial](https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines):

```python
def create_hello_world_message(name: str) -> str:
    """
    Creates a personalized greeting message for the given name.

    Parameters:
        - name (str): The name for which the message is created.

    Returns:
        - hello_world_message (str): A personalized greeting message.

    Raises:
        - ValueError: If the given name is empty or None.
    """
    if not name:
        raise ValueError("Name cannot be empty")

    hello_world_message = f'Hello World, {name}!'
    print(f'name                : {name}')
    print(f'hello_world_message : {hello_world_message}')

    return hello_world_message

# Convert function to pipeline component
task_base_image = 'registry.access.redhat.com/ubi9/python-311'

create_hello_world_message_op = kfp.components.create_component_from_func(
    func=create_hello_world_message,
    base_image=task_base_image
)
```

**Pattern 2: Container Components with Dependencies**

```python
@component(
    base_image='python:3.11',
    packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
)
def preprocess_data(
    input_path: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    test_split: float = 0.2
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data
    df = pd.read_csv(input_path.path)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_split, random_state=42
    )

    # Save outputs
    train_df.to_csv(output_train.path, index=False)
    test_df.to_csv(output_test.path, index=False)
```

**Pattern 3: Multi-Output Components**

```python
from typing import NamedTuple

@component
def train_and_evaluate(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model_output: Output[Model]
) -> NamedTuple('Outputs', [('accuracy', float), ('loss', float)]):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Training logic here...

    # Save model
    torch.save(model.state_dict(), model_output.path)

    # Return metrics
    Outputs = NamedTuple('Outputs', [('accuracy', float), ('loss', float)])
    return Outputs(accuracy=0.95, loss=0.05)
```

### 2.3 Pipeline Compilation and Execution

**Compile pipeline to YAML:**
```python
from kfp import compiler

pipeline_name = 'ml_training_pipeline'
pipeline_package_path = f'{pipeline_name}.yaml'

compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path=pipeline_package_path
)
```

**Execute pipeline programmatically:**
```python
from kfp import Client

kubeflow_endpoint = 'https://kubeflow.example.com'
client = Client(host=kubeflow_endpoint)

# Create experiment
experiment = client.create_experiment(name='ml-experiments')

# Run pipeline
run = client.create_run_from_pipeline_package(
    pipeline_file=pipeline_package_path,
    arguments={
        'data_path': 'gs://my-bucket/data.csv',
        'learning_rate': 0.001,
        'batch_size': 32
    },
    experiment_name='ml-experiments'
)

print(f'Run created: {run.run_id}')
```

---

## 3. Kubeflow Training Operators

### 3.1 PyTorchJob for Distributed Training

From [PyTorch Training (PyTorchJob)](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/) (Kubeflow Documentation):

> "PyTorchJob is a Kubernetes custom resource for running PyTorch training jobs. It is created by defining a config file and deployed to start training."

**PyTorchJob Manifest Example:**
```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed-training
  namespace: kubeflow-user
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/my-project/pytorch-trainer:v1.0
            command:
              - python
              - -m
              - torch.distributed.launch
              - --nproc_per_node=1
              - train.py
              - --epochs=100
              - --batch-size=64
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/my-project/pytorch-trainer:v1.0
            command:
              - python
              - -m
              - torch.distributed.launch
              - --nproc_per_node=1
              - train.py
              - --epochs=100
              - --batch-size=64
            resources:
              limits:
                nvidia.com/gpu: 1
```

**Programmatic PyTorchJob Creation:**
```python
from kubernetes import client
from kubeflow.training import V1PyTorchJob, V1PyTorchJobSpec
from kubeflow.training import V1ReplicaSpec, TrainingClient

# Define training client
training_client = TrainingClient()

# Define PyTorchJob
pytorch_job = V1PyTorchJob(
    api_version="kubeflow.org/v1",
    kind="PyTorchJob",
    metadata=client.V1ObjectMeta(
        name="pytorch-dist-training",
        namespace="kubeflow-user"
    ),
    spec=V1PyTorchJobSpec(
        pytorch_replica_specs={
            "Master": V1ReplicaSpec(
                replicas=1,
                restart_policy="OnFailure",
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="pytorch",
                                image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                                command=["python", "train.py"],
                                resources=client.V1ResourceRequirements(
                                    limits={"nvidia.com/gpu": "1"}
                                )
                            )
                        ]
                    )
                )
            ),
            "Worker": V1ReplicaSpec(
                replicas=2,
                restart_policy="OnFailure",
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="pytorch",
                                image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                                command=["python", "train.py"],
                                resources=client.V1ResourceRequirements(
                                    limits={"nvidia.com/gpu": "1"}
                                )
                            )
                        ]
                    )
                )
            )
        }
    )
)

# Create job
training_client.create_pytorchjob(pytorch_job, namespace="kubeflow-user")
```

### 3.2 TFJob for TensorFlow Distributed Training

From [Distributed Training with the Training Operator](https://www.kubeflow.org/docs/components/trainer/legacy-v1/reference/distributed-training/):

> "The Training Operator creates Kubernetes pods for distributed training, using PyTorch and TensorFlow, with the user creating the job and the operator handling pod management, service discovery, and environment setup."

**TFJob Example:**
```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: tensorflow-distributed-training
spec:
  tfReplicaSpecs:
    PS:  # Parameter Server
      replicas: 2
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=ps
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=worker
            resources:
              limits:
                nvidia.com/gpu: 1
    Chief:  # Coordinator
      replicas: 1
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=chief
            resources:
              limits:
                nvidia.com/gpu: 1
```

### 3.3 Training Operator Benefits

From [PyTorch Joins the Kubeflow Ecosystem](https://pytorch.org/blog/pytorch-on-kubernetes-kubeflow-trainer-joins-the-pytorch-ecosystem/) (PyTorch Blog, July 2025):

> "This project was originally started as a distributed training operator for TensorFlow (e.g. TFJob), and later we merged efforts from other frameworks to support PyTorch, XGBoost, MPI, PaddlePaddle, and others under a unified Training Operator."

**Key benefits:**
1. **Automatic environment setup** - Handles distributed training env vars
2. **Service discovery** - Manages pod-to-pod communication
3. **Fault tolerance** - Restarts failed pods according to policy
4. **Gang scheduling** - Ensures all replicas start together
5. **Integration with Katib** - Hyperparameter tuning out-of-the-box

**Environment variables automatically set:**
- `MASTER_ADDR` - Master node address
- `MASTER_PORT` - Master node port
- `WORLD_SIZE` - Total number of processes
- `RANK` - Process rank
- `LOCAL_RANK` - GPU device ID

---

## 4. Hyperparameter Tuning with Katib

### 4.1 Katib Overview

Katib is Kubeflow's native hyperparameter tuning and neural architecture search (NAS) component.

**Katib Experiment Example:**
```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: pytorch-hp-tuning
spec:
  algorithm:
    algorithmName: random  # or bayesianoptimization, grid, tpe
  parallelTrialCount: 3
  maxTrialCount: 12
  maxFailedTrialCount: 3
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.0001"
        max: "0.01"
    - name: batch_size
      parameterType: int
      feasibleSpace:
        min: "16"
        max: "128"
        step: "16"
    - name: optimizer
      parameterType: categorical
      feasibleSpace:
        list:
          - adam
          - sgd
          - rmsprop
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: learningRate
        description: Learning rate for the training model
        reference: learning_rate
      - name: batchSize
        description: Batch size for training
        reference: batch_size
      - name: optimizer
        description: Optimizer type
        reference: optimizer
    trialSpec:
      apiVersion: kubeflow.org/v1
      kind: PyTorchJob
      spec:
        pytorchReplicaSpecs:
          Master:
            replicas: 1
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                  - name: training-container
                    image: gcr.io/my-project/pytorch-trainer:v1.0
                    command:
                      - python
                      - train.py
                      - --lr=${trialParameters.learningRate}
                      - --batch-size=${trialParameters.batchSize}
                      - --optimizer=${trialParameters.optimizer}
```

### 4.2 Supported Algorithms

**Hyperparameter Tuning Algorithms:**
- **Random Search** - Random sampling within feasible space
- **Grid Search** - Exhaustive search over discrete grid
- **Bayesian Optimization** - Gaussian process-based optimization
- **TPE (Tree-structured Parzen Estimator)** - Sequential model-based optimization
- **CMA-ES** - Covariance Matrix Adaptation Evolution Strategy

**Neural Architecture Search (NAS) Algorithms:**
- **ENAS** - Efficient Neural Architecture Search
- **DARTS** - Differentiable Architecture Search

### 4.3 Metrics Collection

**StdOut Metrics Collector (default):**
```python
# Training script must print metrics in specific format
print(f'accuracy={accuracy:.4f}')
print(f'loss={loss:.4f}')
```

**File Metrics Collector:**
```yaml
metricsCollectorSpec:
  collector:
    kind: File
  source:
    fileSystemPath:
      path: /tmp/metrics.txt
      kind: File
    filter:
      metricsFormat:
        - "accuracy: {accuracy}"
        - "loss: {loss}"
```

---

## 5. Pipeline Orchestration Patterns

### 5.1 Conditional Execution

```python
from kfp.dsl import If

@pipeline(name='conditional-training-pipeline')
def conditional_pipeline(
    data_quality_threshold: float = 0.95
):
    # Step 1: Validate data
    validation_task = validate_data_op()

    # Step 2: Conditional training
    with If(validation_task.outputs['quality_score'] > data_quality_threshold):
        # Only train if data quality is good
        train_task = train_model_op(
            data=validation_task.outputs['data']
        )

        # Deploy if training succeeds
        deploy_task = deploy_model_op(
            model=train_task.outputs['model']
        )
```

### 5.2 Parallel Execution

```python
from kfp.dsl import ParallelFor

@pipeline(name='multi-model-training')
def multi_model_pipeline(
    model_configs: list
):
    # Preprocess data once
    preprocess_task = preprocess_data_op()

    # Train multiple models in parallel
    with ParallelFor(items=model_configs) as config:
        train_task = train_model_op(
            data=preprocess_task.output,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters
        )

        # Evaluate each model
        eval_task = evaluate_model_op(
            model=train_task.outputs['model']
        )
```

### 5.3 Caching for Efficiency

From [Kubeflow Pipelines Caching](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/overview/caching/):

> "Kubeflow Pipelines supports caching to help reduce execution time and cost. When caching is enabled, Kubeflow Pipelines checks whether the execution results of a step already exist from a previous run. If so, it reuses the cached results instead of re-executing the step."

```python
@component(enable_caching=True)  # Enable caching
def expensive_preprocessing(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    # This expensive operation will be cached
    # Subsequent runs with same inputs reuse results
    pass

@pipeline(name='cached-pipeline')
def pipeline_with_caching():
    # First run: Executes preprocessing
    # Second run: Reuses cached result if inputs unchanged
    preprocess_task = expensive_preprocessing()
```

**Disable caching for specific runs:**
```python
client.create_run_from_pipeline_func(
    pipeline_func=pipeline_with_caching,
    arguments={},
    enable_caching=False  # Force re-execution
)
```

---

## 6. Integration with Vertex AI Pipelines

### 6.1 Vertex AI vs Self-Hosted Kubeflow

From [Vertex AI Pipelines + Kubeflow Guide](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/sdk/connect-api/) (Kubeflow Docs):

| Aspect | Self-Hosted Kubeflow | Vertex AI Pipelines |
|--------|---------------------|-------------------|
| **Infrastructure** | Self-managed K8s cluster | Fully managed by Google |
| **Cost** | Compute + management overhead | Pay-per-pipeline-run |
| **Maintenance** | User responsible | Google handles updates |
| **Storage** | Any object storage (S3, GCS, MinIO) | Cloud Storage only |
| **Customization** | Full control | Limited to Vertex AI features |
| **GPU Access** | Depends on cluster | Easy GPU/TPU access |

### 6.2 Compiling for Vertex AI

```python
from kfp import compiler
from google.cloud import aiplatform

# Compile for Vertex AI
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='pipeline.yaml',
    pipeline_name='vertex-ai-pipeline'
)

# Initialize Vertex AI
aiplatform.init(
    project='my-gcp-project',
    location='us-central1'
)

# Submit to Vertex AI
job = aiplatform.PipelineJob(
    display_name='ml-training-job',
    template_path='pipeline.yaml',
    parameter_values={
        'data_path': 'gs://my-bucket/data.csv',
        'learning_rate': 0.001
    }
)

job.submit()
```

### 6.3 Vertex AI-Specific Features

**Managed datasets:**
```python
from google.cloud import aiplatform

dataset = aiplatform.TabularDataset.create(
    display_name='training-dataset',
    gcs_source='gs://my-bucket/data.csv'
)

# Use in pipeline
@component
def train_on_vertex_dataset(
    dataset_id: str,
    model_output: Output[Model]
):
    dataset = aiplatform.TabularDataset(dataset_id)
    # Training logic...
```

**Model Registry integration:**
```python
@component
def register_model_to_vertex(
    model_path: Input[Model],
    model_display_name: str
) -> str:
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path.uri,
        serving_container_image_uri='gcr.io/my-project/serving:latest'
    )
    return model.resource_name
```

---

## 7. Production Deployment Patterns

### 7.1 CI/CD Integration

**GitHub Actions + Kubeflow:**
```yaml
# .github/workflows/ml-pipeline.yml
name: Deploy ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'pipelines/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install kfp==2.8.0

      - name: Compile pipeline
        run: |
          python pipelines/compile.py

      - name: Deploy to Kubeflow
        env:
          KUBEFLOW_ENDPOINT: ${{ secrets.KUBEFLOW_ENDPOINT }}
          KUBEFLOW_TOKEN: ${{ secrets.KUBEFLOW_TOKEN }}
        run: |
          python pipelines/deploy.py
```

### 7.2 Monitoring and Alerting

**Prometheus metrics for pipelines:**
```python
from prometheus_client import Counter, Histogram

# Define metrics
pipeline_runs = Counter(
    'pipeline_runs_total',
    'Total number of pipeline runs',
    ['pipeline_name', 'status']
)

pipeline_duration = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_name']
)

# Instrument pipeline
@pipeline(name='monitored-pipeline')
def monitored_pipeline():
    with pipeline_duration.labels(pipeline_name='monitored-pipeline').time():
        try:
            # Pipeline steps...
            pipeline_runs.labels(
                pipeline_name='monitored-pipeline',
                status='success'
            ).inc()
        except Exception as e:
            pipeline_runs.labels(
                pipeline_name='monitored-pipeline',
                status='failure'
            ).inc()
            raise
```

### 7.3 Multi-Tenancy and Resource Quotas

**Namespace isolation:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team-namespace
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 500Gi
    requests.nvidia.com/gpu: "10"
    limits.cpu: "200"
    limits.memory: 1000Gi
    limits.nvidia.com/gpu: "10"
    persistentvolumeclaims: "10"
```

**Pipeline resource limits:**
```python
@component
def training_component_with_limits():
    from kubernetes import client

    return client.V1ResourceRequirements(
        requests={
            "cpu": "4",
            "memory": "16Gi",
            "nvidia.com/gpu": "1"
        },
        limits={
            "cpu": "8",
            "memory": "32Gi",
            "nvidia.com/gpu": "1"
        }
    )
```

---

## 8. arr-coc-0-1 Use Cases

### 8.1 VLM Training Pipeline

```python
@pipeline(
    name='arr-coc-vlm-training',
    description='ARR-COC VLM training with relevance realization'
)
def arr_coc_training_pipeline(
    dataset_path: str,
    num_gpus: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 10
):
    # Step 1: Prepare texture arrays
    texture_prep = prepare_texture_arrays_op(
        dataset_path=dataset_path
    )

    # Step 2: Distributed training with PyTorchJob
    training_job = create_pytorch_job_op(
        texture_data=texture_prep.outputs['texture_data'],
        num_gpus=num_gpus,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # Step 3: Evaluate relevance scorers
    eval_task = evaluate_scorers_op(
        model=training_job.outputs['model'],
        test_data=texture_prep.outputs['test_data']
    )

    # Step 4: Deploy to Cloud Run
    with If(eval_task.outputs['accuracy'] > 0.90):
        deploy_to_cloud_run_op(
            model=training_job.outputs['model'],
            region='us-west2'
        )
```

### 8.2 Hyperparameter Tuning for Opponent Processing

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: arr-coc-opponent-processing-tuning
spec:
  algorithm:
    algorithmName: bayesianoptimization
  parallelTrialCount: 4
  maxTrialCount: 20
  objective:
    type: maximize
    objectiveMetricName: relevance_score
  parameters:
    - name: tension_compression_explore
      parameterType: double
      feasibleSpace:
        min: "0.0"
        max: "1.0"
    - name: tension_exploit_explore
      parameterType: double
      feasibleSpace:
        min: "0.0"
        max: "1.0"
    - name: tension_focus_diversify
      parameterType: double
      feasibleSpace:
        min: "0.0"
        max: "1.0"
  trialTemplate:
    trialSpec:
      apiVersion: kubeflow.org/v1
      kind: PyTorchJob
      spec:
        pytorchReplicaSpecs:
          Master:
            replicas: 1
            template:
              spec:
                containers:
                  - name: pytorch
                    image: gcr.io/arr-coc-0-1/trainer:latest
                    command:
                      - python
                      - train.py
                      - --tension-compress=${trialParameters.tension_compression_explore}
                      - --tension-exploit=${trialParameters.tension_exploit_explore}
                      - --tension-focus=${trialParameters.tension_focus_diversify}
```

### 8.3 Multi-Model Evaluation Pipeline

```python
@pipeline(name='arr-coc-multi-scorer-eval')
def evaluate_all_scorers(
    test_dataset: str,
    scorer_configs: list
):
    # Prepare test data
    prep_task = prepare_test_data_op(dataset_path=test_dataset)

    # Evaluate each scorer in parallel
    with ParallelFor(items=scorer_configs) as config:
        # Train scorer
        train_task = train_scorer_op(
            data=prep_task.outputs['data'],
            scorer_type=config.scorer_type,  # propositional/perspectival/participatory
            config=config.hyperparams
        )

        # Evaluate
        eval_task = evaluate_scorer_op(
            scorer=train_task.outputs['model'],
            test_data=prep_task.outputs['test_data']
        )

        # Register best performing
        with If(eval_task.outputs['score'] > 0.85):
            register_scorer_op(
                scorer=train_task.outputs['model'],
                scorer_type=config.scorer_type,
                score=eval_task.outputs['score']
            )
```

---

## 9. Best Practices

### 9.1 Component Design

**Do:**
- Keep components small and focused (single responsibility)
- Use typed inputs/outputs for validation
- Include comprehensive logging
- Make components idempotent
- Version component images

**Don't:**
- Mix data preparation and training in one component
- Hard-code paths or credentials
- Skip error handling
- Use latest tags for production

### 9.2 Pipeline Organization

```
ml-pipelines/
├── components/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── training/
│   │   ├── pytorch_trainer.py
│   │   └── tensorflow_trainer.py
│   └── deployment/
│       └── cloud_run_deploy.py
├── pipelines/
│   ├── training_pipeline.py
│   ├── evaluation_pipeline.py
│   └── deployment_pipeline.py
├── configs/
│   ├── dev_config.yaml
│   └── prod_config.yaml
└── tests/
    ├── test_components.py
    └── test_pipelines.py
```

### 9.3 Cost Optimization

**Spot instances for training:**
```python
@component
def train_on_spot():
    from kubernetes import client

    return client.V1PodSpec(
        node_selector={
            "cloud.google.com/gke-spot": "true"  # GKE
            # OR
            "eks.amazonaws.com/capacityType": "SPOT"  # EKS
        },
        tolerations=[
            client.V1Toleration(
                key="cloud.google.com/gke-spot",
                operator="Equal",
                value="true",
                effect="NoSchedule"
            )
        ]
    )
```

**Automatic cleanup:**
```python
@pipeline(name='training-with-cleanup')
def pipeline_with_cleanup():
    train_task = train_model_op()

    # Always run cleanup, even if training fails
    with ExitHandler(cleanup_op()):
        eval_task = evaluate_model_op(
            model=train_task.outputs['model']
        )
```

---

## Sources

**Official Documentation:**
- [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) - Kubeflow.org (accessed 2025-11-13)
- [Kubeflow Pipelines Legacy v1](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/) - Kubeflow.org (accessed 2025-11-13)
- [Kubeflow Training Operators](https://www.kubeflow.org/docs/components/trainer/) - Kubeflow.org (accessed 2025-11-13)
- [PyTorch Training (PyTorchJob)](https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/) - Kubeflow.org (accessed 2025-11-13)

**Tutorials and Guides:**
- [Implement MLOps with Kubeflow Pipelines](https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines) - Red Hat Developer, January 2024
- [Kubeflow Pipelines Tutorial 2024](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/tutorials/) - Kubeflow.org (accessed 2025-11-13)

**Comparisons:**
- [Kubeflow vs MLflow: An In-Depth Comparison](https://skphd.medium.com/kubeflow-vs-mlflow-an-in-depth-comparison-for-mlops-pipelines-e2f0c0496a36) - Medium, January 2024
- [MLflow vs Kubeflow Reddit Discussion](https://www.reddit.com/r/mlops/comments/1evza42/mlflow_vs_kubeflow/) - r/mlops, 2024
- [Kubeflow vs MLflow vs ZenML](https://www.zenml.io/blog/kubeflow-vs-mlflow) - ZenML Blog, May 2025

**PyTorch Integration:**
- [PyTorch Joins the Kubeflow Ecosystem](https://pytorch.org/blog/pytorch-on-kubernetes-kubeflow-trainer-joins-the-pytorch-ecosystem/) - PyTorch Blog, July 2025

**Additional Resources:**
- [Distributed Training with Training Operator](https://www.kubeflow.org/docs/components/trainer/legacy-v1/reference/distributed-training/) - Kubeflow.org (accessed 2025-11-13)
- [Building Machine Learning Pipelines with Vertex AI and Kubeflow](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d) - Medium, Gabriel Preda, 10 months ago
