# Cloud Composer for ML Orchestration

**What**: Google's managed Apache Airflow service for workflow orchestration
**Why**: Coordinate complex ML pipelines with robust scheduling, monitoring, and dependencies
**When**: Multi-stage training pipelines, data engineering + ML workflows, teams already using Airflow

---

## Overview

Cloud Composer is Google Cloud's fully managed workflow orchestration service built on Apache Airflow. Originally designed for data engineering, it has evolved to handle machine learning workflows with sophisticated dependencies, scheduling, and monitoring capabilities.

**Key Distinction**: Cloud Composer excels at **complex, multi-stage workflows** where tasks have intricate dependencies. For simpler ML-specific pipelines, consider Vertex AI Pipelines as a serverless alternative.

From [ZenML comparison](https://www.zenml.io/blog/cloud-composer-airflow-vs-vertex-ai-kubeflow) (accessed 2025-01-13):
> "Cloud Composer builds upon the most widely adopted open-source tool for data pipeline orchestration. Originally designed with data engineers in mind, Airflow has evolved to accommodate a wide range of use cases, including machine learning workflows."

---

## Section 1: Cloud Composer Architecture (90 lines)

### What is Cloud Composer?

Cloud Composer is a managed Apache Airflow environment that runs on Google Kubernetes Engine (GKE). It provides:

**Core Components**:
- **Airflow Scheduler**: Triggers tasks based on dependencies and schedules
- **Airflow Webserver**: UI for monitoring and managing workflows
- **Airflow Workers**: Execute individual tasks (CPU or container-based)
- **Cloud SQL**: Stores Airflow metadata (DAG runs, task states)
- **GCS Bucket**: Stores DAG files, logs, and plugins

**Managed Infrastructure**:
From [Google Cloud Composer overview](https://cloud.google.com/composer/docs/composer-3/composer-overview) (accessed 2025-01-13):
> "Cloud Composer is a fully managed workflow orchestration service, enabling you to create, schedule, monitor, and manage workflow pipelines that span across clouds and on-premises data centers."

The service automatically handles:
- Airflow version upgrades
- Security patches
- Auto-scaling of workers
- High availability setup
- Integration with IAM

**Cloud Composer 2 vs 3**:
- **Composer 2**: Airflow 2.x, improved performance, auto-scaling workers
- **Composer 3**: Airflow 3.x (released December 2024), DAG versioning, enhanced scheduling

### Apache Airflow Foundation

**DAG (Directed Acyclic Graph)**:
Airflow workflows are defined as DAGs - Python code that describes task dependencies:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    'ml_training_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_function
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_function
    )

    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_function
    )

    # Define dependencies
    preprocess >> train >> evaluate
```

**Key Airflow Concepts**:
- **Operators**: Define individual tasks (PythonOperator, BashOperator, CustomOperator)
- **Sensors**: Wait for external events (FileSensor, TimeSensor)
- **XComs**: Share small data between tasks (< 48KB)
- **Pools**: Limit concurrent task execution
- **Connections**: Store credentials for external systems

### When to Use Cloud Composer

From [Towards Data Science comparison](https://towardsdatascience.com/google-cloud-alternatives-to-cloud-composer-972836388a3f) (accessed 2025-01-13):

**Use Cloud Composer when**:
1. **Large-scale orchestration**: Hundreds or thousands of workflows
2. **Complex dependencies**: Dynamic task generation, conditional branching
3. **Non-trivial trigger rules**: "Run if ANY upstream task fails"
4. **Hybrid workflows**: Mix data engineering + ML + API calls
5. **Existing Airflow expertise**: Team already familiar with Airflow

**Skip Cloud Composer for**:
- Simple sequential pipelines → Use Vertex AI Pipelines
- Microservice orchestration → Use Cloud Workflows
- Single training job → Use Vertex AI Custom Training directly

---

## Section 2: ML Pipeline DAGs (130 lines)

### DAG Definition for ML Workflows

**Basic ML Pipeline Structure**:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.vertex_ai import (
    CreateCustomTrainingJobOperator,
    CreateBatchPredictionJobOperator
)
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryInsertJobOperator
)
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import (
    GCSToBigQueryOperator
)
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ml-alerts@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'vlm_training_pipeline',
    default_args=default_args,
    description='ARR-COC VLM training pipeline',
    schedule='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'vision', 'training'],
) as dag:

    # Task 1: Extract training data from BigQuery
    extract_data = BigQueryInsertJobOperator(
        task_id='extract_training_data',
        configuration={
            'query': {
                'query': 'SELECT * FROM `project.dataset.training_data` WHERE date >= "{{ ds }}"',
                'useLegacySql': False,
                'destinationTable': {
                    'projectId': 'project-id',
                    'datasetId': 'staging',
                    'tableId': 'train_batch_{{ ds_nodash }}'
                }
            }
        }
    )

    # Task 2: Preprocess data with Dataflow
    preprocess = DataflowTemplatedJobStartOperator(
        task_id='preprocess_images',
        template='gs://project/templates/image-preprocessing',
        parameters={
            'input_table': 'project.staging.train_batch_{{ ds_nodash }}',
            'output_gcs': 'gs://project/processed/{{ ds }}/'
        }
    )

    # Task 3: Train model on Vertex AI
    train_model = CreateCustomTrainingJobOperator(
        task_id='train_vlm_model',
        region='us-central1',
        display_name='arr-coc-training-{{ ds }}',
        container_uri='gcr.io/project/arr-coc-trainer:latest',
        machine_type='n1-highmem-16',
        accelerator_type='NVIDIA_TESLA_V100',
        accelerator_count=4,
        replica_count=1,
        args=[
            '--data-path', 'gs://project/processed/{{ ds }}/',
            '--output-path', 'gs://project/models/{{ ds }}/',
            '--epochs', '50',
            '--batch-size', '32',
        ],
        environment_variables={
            'WANDB_API_KEY': '{{ var.value.wandb_api_key }}',
            'WANDB_PROJECT': 'arr-coc-production'
        }
    )

    # Task 4: Evaluate model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_function,
        op_kwargs={
            'model_path': 'gs://project/models/{{ ds }}/',
            'test_data': 'gs://project/test-data/'
        }
    )

    # Task 5: Deploy if accuracy threshold met
    deploy = CreateCustomTrainingJobOperator(
        task_id='deploy_to_endpoint',
        region='us-central1',
        # ... deployment config
        trigger_rule='all_success'  # Only if evaluate succeeds
    )

    # Define workflow
    extract_data >> preprocess >> train_model >> evaluate >> deploy
```

### Operators for ML Tasks

**Vertex AI Operators** (from `airflow.providers.google`):
- `CreateCustomTrainingJobOperator`: Run custom training jobs with GPUs
- `CreateHyperparameterTuningJobOperator`: Hyperparameter search
- `CreateBatchPredictionJobOperator`: Batch inference
- `CreateModelOperator`: Upload model to Vertex AI Model Registry
- `DeployModelOperator`: Deploy to Vertex AI Endpoint

From [Medium article by Tim Swena](https://medium.com/google-cloud/creating-a-production-ready-data-pipeline-with-apache-airflow-and-bigframes-bead7d7d164b) (accessed 2025-01-13):
> "Cloud Composer integrates seamlessly with Vertex AI operators, enabling ML teams to orchestrate training, evaluation, and deployment within a single DAG."

**BigQuery Operators**:
- `BigQueryInsertJobOperator`: Run SQL queries (feature engineering)
- `BigQueryCreateExternalTableOperator`: Create external tables
- `BigQueryToGCSOperator`: Export query results to GCS

**GCS Operators**:
- `GCSToGCSOperator`: Copy datasets between buckets
- `GCSDeleteObjectsOperator`: Cleanup old artifacts

**Custom Python Operators**:
```python
from airflow.operators.python import PythonOperator

def custom_preprocessing(**context):
    """Custom preprocessing logic"""
    import pandas as pd
    from google.cloud import storage

    # Access previous task output via XCom
    input_path = context['ti'].xcom_pull(task_ids='extract_data')

    # Process data
    df = pd.read_csv(input_path)
    processed = preprocess_images(df)

    # Upload to GCS
    output_path = f"gs://bucket/processed/{context['ds']}.csv"
    processed.to_csv(output_path)

    # Return path for next task
    return output_path

preprocess = PythonOperator(
    task_id='custom_preprocess',
    python_callable=custom_preprocessing,
    provide_context=True
)
```

### Task Dependencies and Branching

**Dependency Operators**:
```python
# Sequential
task1 >> task2 >> task3

# Parallel
task1 >> [task2, task3, task4] >> task5

# Conditional branching
from airflow.operators.python import BranchPythonOperator

def decide_branch(**context):
    accuracy = context['ti'].xcom_pull(task_ids='evaluate')
    if accuracy > 0.95:
        return 'deploy_production'
    else:
        return 'retrain_with_more_data'

branch = BranchPythonOperator(
    task_id='decide_deployment',
    python_callable=decide_branch
)

evaluate >> branch >> [deploy_production, retrain_with_more_data]
```

**Trigger Rules**:
- `all_success`: All upstream tasks succeeded (default)
- `all_failed`: All upstream tasks failed
- `one_success`: At least one upstream task succeeded
- `one_failed`: At least one upstream task failed
- `none_failed`: No upstream tasks failed (some may have skipped)

### Dynamic DAG Generation

Generate tasks programmatically for hyperparameter sweeps:

```python
# Generate training tasks for different hyperparameters
hyperparameters = [
    {'learning_rate': 0.001, 'batch_size': 32},
    {'learning_rate': 0.0001, 'batch_size': 64},
    {'learning_rate': 0.01, 'batch_size': 16},
]

training_tasks = []
for idx, params in enumerate(hyperparameters):
    task = CreateCustomTrainingJobOperator(
        task_id=f'train_model_{idx}',
        display_name=f'arr-coc-sweep-{idx}',
        args=[
            '--learning-rate', str(params['learning_rate']),
            '--batch-size', str(params['batch_size']),
        ]
    )
    training_tasks.append(task)

# Run all training tasks in parallel
preprocess >> training_tasks >> select_best_model
```

From [ZenML ECB pipeline example](https://github.com/zenml-io/zenml-projects/tree/main/airflow-cloud-composer-etl-feature-train) (accessed 2025-01-13):
> "ZenML pipelines can run entirely on Airflow, leveraging its robust orchestration without implementing ML-specific features, while selectively outsourcing GPU-intensive tasks to Vertex AI."

---

## Section 3: Production Patterns (100 lines)

### Monitoring and Alerting

**Airflow UI Monitoring**:
- DAG run status (success, failed, running)
- Task duration charts
- Gantt chart for parallelism visualization
- Log viewer for debugging

**Email Alerts**:
```python
default_args = {
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ml-team@company.com'],
}

# Custom alert callback
def slack_alert_on_failure(context):
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

    task_instance = context['task_instance']
    exception = context['exception']

    SlackWebhookOperator(
        task_id='slack_alert',
        http_conn_id='slack_webhook',
        message=f"Task {task_instance.task_id} failed: {exception}"
    ).execute(context=context)

with DAG(..., default_args=default_args, on_failure_callback=slack_alert_on_failure):
    ...
```

**Cloud Monitoring Integration**:
Cloud Composer automatically exports metrics to Cloud Monitoring:
- DAG success rate
- Task duration (P50, P95, P99)
- Worker CPU/memory usage
- Queue length

Create alerts on critical metrics:
```yaml
# Example alert policy
displayName: "High DAG Failure Rate"
conditions:
  - displayName: "Failure rate > 20%"
    conditionThreshold:
      filter: 'resource.type="cloud_composer_environment" AND metric.type="composer.googleapis.com/environment/dag_run/failed"'
      comparison: COMPARISON_GT
      thresholdValue: 0.2
      duration: 300s
```

### Error Handling and Retries

**Automatic Retries**:
```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}
```

**Custom Retry Logic**:
```python
from airflow.exceptions import AirflowException

def train_with_retry(**context):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = train_model()
            return result
        except ResourceExhaustedError as e:
            if attempt < max_attempts - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(60 * (attempt + 1))  # Exponential backoff
            else:
                raise AirflowException(f"Training failed after {max_attempts} attempts")
```

**Task-level error handling**:
```python
evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_function,
    trigger_rule='none_failed',  # Run even if upstream task skipped
    retries=5,
    retry_delay=timedelta(minutes=10),
)
```

### Secrets Management

**Airflow Variables** (for non-sensitive config):
```python
from airflow.models import Variable

wandb_project = Variable.get("wandb_project")
model_bucket = Variable.get("model_artifacts_bucket")
```

**Secret Manager Integration** (for credentials):
```python
from airflow.providers.google.cloud.secrets.secret_manager import CloudSecretManagerBackend

# Configure in airflow.cfg or environment:
# [secrets]
# backend = airflow.providers.google.cloud.secrets.secret_manager.CloudSecretManagerBackend
# backend_kwargs = {"project_id": "my-project"}

# Access secrets in DAG
from airflow.hooks.base import BaseHook

connection = BaseHook.get_connection('wandb_api_key')
wandb_key = connection.password
```

**Best Practices**:
- Store API keys in Secret Manager, not Airflow Variables
- Use IAM service accounts for GCP resource access
- Rotate secrets regularly
- Audit secret access via Cloud Logging

From [Google Cloud Composer documentation](https://cloud.google.com/composer/docs) (accessed 2025-01-13):
> "Cloud Composer integrates with Secret Manager to securely manage credentials and API keys without storing them in DAG files or Airflow variables."

### Environment Configuration

**Airflow Environment Variables**:
Set via Composer environment configuration:
```bash
# Via gcloud CLI
gcloud composer environments update ENVIRONMENT_NAME \
    --location LOCATION \
    --update-env-variables KEY1=VALUE1,KEY2=VALUE2

# Via Terraform
resource "google_composer_environment" "ml_pipeline_env" {
  name   = "ml-pipelines"
  region = "us-central1"

  config {
    software_config {
      env_variables = {
        WANDB_PROJECT = "arr-coc"
        MODEL_BUCKET  = "gs://project-models"
      }
    }
  }
}
```

**Python Package Dependencies**:
```bash
# Install via PyPI
gcloud composer environments update ENVIRONMENT_NAME \
    --update-pypi-packages-from-file requirements.txt

# Custom packages via GCS
gcloud composer environments storage plugins import \
    --environment ENVIRONMENT_NAME \
    --source custom_operators.py
```

**Resource Allocation**:
```python
# Task-level resource requirements
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_function,
    pool='gpu_pool',  # Limit concurrent GPU tasks
    queue='high_memory_queue',  # Route to high-memory workers
)
```

---

## Section 4: ARR-COC Training Pipeline (70 lines)

### End-to-End VLM Training Orchestration

**Multi-Stage Pipeline**:

```python
with DAG(
    'arr_coc_training_pipeline',
    schedule='0 3 * * 1',  # Weekly on Monday 3 AM
    default_args=default_args,
    catchup=False,
) as dag:

    # Stage 1: Data Preprocessing
    extract_images = BigQueryToGCSOperator(
        task_id='extract_vqa_images',
        source_project_dataset_table='project.vqa_dataset.images',
        destination_cloud_storage_uris=['gs://project/raw/images/{{ ds }}/'],
    )

    compute_textures = CreateCustomTrainingJobOperator(
        task_id='compute_texture_arrays',
        display_name='texture-computation-{{ ds }}',
        container_uri='gcr.io/project/texture-processor:latest',
        machine_type='n1-standard-32',
        args=['--input', 'gs://project/raw/images/{{ ds }}/',
              '--output', 'gs://project/textures/{{ ds }}/'],
    )

    # Stage 2: Training
    train_arr_coc = CreateCustomTrainingJobOperator(
        task_id='train_arr_coc_model',
        display_name='arr-coc-training-{{ ds }}',
        container_uri='gcr.io/project/arr-coc-trainer:v2',
        machine_type='a2-highgpu-4g',  # 4x A100 GPUs
        args=[
            '--texture-path', 'gs://project/textures/{{ ds }}/',
            '--checkpoint', 'gs://project/checkpoints/latest.pt',
            '--epochs', '10',
            '--relevance-scorers', 'propositional,perspectival,participatory',
        ],
        environment_variables={
            'WANDB_PROJECT': 'arr-coc-production',
            'NCCL_DEBUG': 'INFO',  # Multi-GPU debugging
        }
    )

    # Stage 3: Evaluation
    evaluate_vqa = PythonOperator(
        task_id='evaluate_vqa_accuracy',
        python_callable=run_vqa_evaluation,
        op_kwargs={
            'model_path': 'gs://project/models/{{ ds }}/',
            'test_dataset': 'vqav2-test',
            'metrics': ['accuracy', 'relevance_distribution']
        }
    )

    validate_relevance = PythonOperator(
        task_id='validate_relevance_realization',
        python_callable=validate_opponent_processing,
        op_kwargs={
            'model_path': 'gs://project/models/{{ ds }}/',
            'test_cases': 'gs://project/test-cases/relevance.json'
        }
    )

    # Stage 4: Checkpoint Management
    save_checkpoint = GCSToBigQueryOperator(
        task_id='save_checkpoint_metadata',
        bucket='project',
        source_objects=['models/{{ ds }}/checkpoint.pt'],
        destination_project_dataset_table='project.models.checkpoints',
        schema_fields=[
            {'name': 'date', 'type': 'DATE'},
            {'name': 'vqa_accuracy', 'type': 'FLOAT'},
            {'name': 'model_path', 'type': 'STRING'},
        ],
    )

    # Stage 5: Deployment Decision
    def decide_deployment(**context):
        accuracy = context['ti'].xcom_pull(task_ids='evaluate_vqa_accuracy')
        if accuracy > context['var']['value']['deployment_threshold']:
            return 'deploy_to_staging'
        else:
            return 'log_experiment_only'

    deployment_decision = BranchPythonOperator(
        task_id='decide_deployment',
        python_callable=decide_deployment
    )

    deploy_staging = DeployModelOperator(
        task_id='deploy_to_staging',
        model='gs://project/models/{{ ds }}/',
        endpoint_id='arr-coc-staging-endpoint',
    )

    # Workflow
    extract_images >> compute_textures >> train_arr_coc
    train_arr_coc >> [evaluate_vqa, validate_relevance] >> save_checkpoint
    save_checkpoint >> deployment_decision >> [deploy_staging, log_experiment_only]
```

### Multi-Stage Checkpoint Validation

**Checkpoint Quality Gates**:

```python
def validate_checkpoint(checkpoint_path, **context):
    """Validate checkpoint before proceeding to next stage"""
    import torch
    from arr_coc.model import ARRCOCModel

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Validation checks
    checks = {
        'has_relevance_scorers': all(k in checkpoint for k in ['propositional', 'perspectival', 'participatory']),
        'has_balancer_weights': 'tension_balancer' in checkpoint,
        'valid_loss': checkpoint['loss'] < 10.0,
        'convergence': checkpoint['epoch'] >= 10,
    }

    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Checkpoint validation failed: {failed}")

    return checkpoint_path

checkpoint_validation = PythonOperator(
    task_id='validate_checkpoint',
    python_callable=validate_checkpoint,
    op_kwargs={'checkpoint_path': 'gs://project/checkpoints/{{ ds }}/'},
)

train_stage1 >> checkpoint_validation >> train_stage2
```

From [ZenML blog comparison](https://www.zenml.io/blog/cloud-composer-airflow-vs-vertex-ai-kubeflow) (accessed 2025-01-13):
> "By using ZenML with Cloud Composer, teams can maintain existing Airflow-based workflows while easily incorporating Vertex AI's powerful ML capabilities when needed for GPU-intensive tasks."

### Cost Optimization Patterns

**Spot Instance Training**:
```python
train_on_spot = CreateCustomTrainingJobOperator(
    task_id='train_with_spot_instances',
    machine_type='n1-highmem-16',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=4,
    base_output_directory='gs://project/spot-training/',
    enable_web_access=False,
    # Use Spot VMs (preemptible) for 60-70% cost savings
    scheduling={
        'timeout': '86400s',  # 24 hours max
        'restart_job_on_worker_restart': True
    }
)
```

**Incremental Training**:
```python
# Only retrain if new data available
def check_new_data(**context):
    from google.cloud import bigquery

    client = bigquery.Client()
    query = f"""
        SELECT COUNT(*) as new_rows
        FROM `project.dataset.training_data`
        WHERE date >= '{context['ds']}'
    """
    result = list(client.query(query))[0]

    if result.new_rows > 1000:  # Threshold for retraining
        return 'train_model'
    else:
        return 'skip_training'

check_data = BranchPythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data
)

check_data >> [train_model, skip_training]
```

---

## Cost Comparison: Cloud Composer vs Vertex AI Pipelines

From [ZenML cost analysis](https://www.zenml.io/blog/cloud-composer-airflow-vs-vertex-ai-kubeflow) (accessed 2025-01-13):

**Monthly Cost Example** (30 daily ML pipeline runs):

| Component | Cloud Composer | Vertex AI Pipelines |
|-----------|----------------|---------------------|
| Environment Fee | $396.00 | $1.80 |
| CPU Resources (2hr/day) | Included | $13.11 |
| GPU Resources (4hr/day, T4) | $42.00 | $50.62 |
| **Total** | **$438.00** | **$65.53** |

**Key Insight**: Vertex AI Pipelines is significantly cheaper for ML-specific workloads due to serverless architecture. However, if your team already uses Airflow for data engineering, the base Composer cost is amortized across all workflows, making per-pipeline costs comparable.

**When Cloud Composer is Cost-Effective**:
- Running 10+ different workflows (amortize base cost)
- Mix of data engineering + ML pipelines
- Need for advanced Airflow features (backfill, dynamic DAGs)

**When Vertex AI Pipelines is Cheaper**:
- Pure ML workloads (training, tuning, deployment)
- Intermittent pipeline execution (pay-per-run)
- Serverless preference (no infrastructure management)

From [Medium comparison by Sascha Heyer](https://medium.com/google-cloud/vertex-ai-pipelines-vs-cloud-composer-for-orchestration-4bba129759de) (accessed 2025-01-13):
> "Vertex AI Pipeline reduces the costs significantly, you only pay for what you use. In comparison to Cloud Composer, you need at least 2 nodes with minimum configuration to have it run properly."

---

## Cloud Composer vs Vertex AI Pipelines Decision Matrix

From [Towards Data Science alternatives article](https://towardsdatascience.com/google-cloud-alternatives-to-cloud-composer-972836388a3f) (accessed 2025-01-13):

| Criterion | Cloud Composer | Vertex AI Pipelines |
|-----------|----------------|---------------------|
| **Simplicity** | 2/5 (steep learning curve) | 4/5 (ML-focused) |
| **Maintainability** | 5/5 (mature ecosystem) | 4/5 (specialized for ML) |
| **Scalability** | 5/5 (infinite) | 4/5 (serverless) |
| **Cost** | 2/5 (base cluster cost) | 5/5 (pay-per-run) |
| **ML-Specific Features** | 3/5 (via operators) | 5/5 (native) |
| **Community Support** | 5/5 (Apache Airflow) | 3/5 (Kubeflow/Vertex) |

**Use Cloud Composer if**:
- Large-scale orchestration (100+ workflows)
- Complex dependencies (dynamic DAGs, conditional branching)
- Team has Airflow expertise
- Mix of data + ML workflows
- Need advanced scheduling (backfill, catch-up, SLA monitoring)

**Use Vertex AI Pipelines if**:
- Pure ML workflows (training → evaluation → deployment)
- Serverless preference (no infrastructure management)
- GPU-intensive workloads with intermittent execution
- Tighter integration with Vertex AI services
- Cost-sensitive ML-specific workloads

From [ZenML hybrid approach](https://www.zenml.io/blog/cloud-composer-airflow-vs-vertex-ai-kubeflow):
> "With ZenML, you can run all steps within a pipeline on Airflow, while selectively outsourcing GPU-intensive workloads to Vertex AI using the ZenML step operator. This hybrid strategy enables teams to maintain their existing Airflow-based workflows while easily incorporating Vertex AI's powerful ML capabilities."

---

## Production Best Practices

**1. Environment Separation**:
- Dev: Small environment (small-composer), cheap testing
- Staging: Production-like (medium-composer), pre-deployment validation
- Production: Large environment (large-composer), high availability

**2. DAG Design Patterns**:
- Keep DAGs modular (reusable tasks)
- Use SubDAGs sparingly (performance overhead)
- Leverage dynamic task mapping for parallel execution
- Implement idempotency (tasks can be safely retried)

**3. Monitoring Strategy**:
- Set up email/Slack alerts for failures
- Monitor task duration trends (detect performance degradation)
- Track DAG success rate (SLO: >95%)
- Use Cloud Logging for detailed debugging

**4. Version Control**:
- Store DAGs in Git repository
- Use CI/CD to deploy DAGs to Composer (GCS bucket sync)
- Tag releases with semantic versioning
- Test DAGs in dev environment before production

**5. Resource Management**:
- Use pools to limit concurrent GPU tasks
- Configure worker auto-scaling based on queue depth
- Set task timeouts to prevent hung tasks
- Clean up old logs and DAG runs (retention policy)

---

## Sources

**Web Research:**
- [ZenML: Cloud Composer vs Vertex AI](https://www.zenml.io/blog/cloud-composer-airflow-vs-vertex-ai-kubeflow) (accessed 2025-01-13)
- [Google Cloud Composer Overview](https://cloud.google.com/composer/docs/composer-3/composer-overview) (accessed 2025-01-13)
- [Medium: Vertex AI Pipelines vs Cloud Composer](https://medium.com/google-cloud/vertex-ai-pipelines-vs-cloud-composer-for-orchestration-4bba129759de) by Sascha Heyer (accessed 2025-01-13)
- [Towards Data Science: Cloud Composer Alternatives](https://towardsdatascience.com/google-cloud-alternatives-to-cloud-composer-972836388a3f) by Marc Djohossou (accessed 2025-01-13)
- [Medium: Production-Ready Data Pipeline with Airflow](https://medium.com/google-cloud/creating-a-production-ready-data-pipeline-with-apache-airflow-and-bigframes-bead7d7d164b) by Tim Swena (accessed 2025-01-13)
- [ZenML ECB Pipeline Example](https://github.com/zenml-io/zenml-projects/tree/main/airflow-cloud-composer-etl-feature-train) (accessed 2025-01-13)

**Additional References:**
- [Google Cloud Composer Documentation](https://cloud.google.com/composer/docs)
- [Apache Airflow Official Docs](https://airflow.apache.org/docs/)
- [Airflow Providers for GCP](https://airflow.apache.org/docs/apache-airflow-providers-google/)
