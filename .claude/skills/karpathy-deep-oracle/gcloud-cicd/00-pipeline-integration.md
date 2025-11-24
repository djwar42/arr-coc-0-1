# CI/CD Pipeline Integration for ML Training

Complete guide to integrating Cloud Build, Vertex AI Pipelines, W&B Launch, and GitOps for production ML training workflows. Covers continuous training, automated deployment, and model lifecycle management.

## Overview

Modern ML production systems require seamless integration of CI/CD pipelines with training infrastructure. This integration enables:

- **Continuous Training**: Automated retraining on code/data changes
- **GitOps Workflows**: Infrastructure and model config as code
- **Automated Testing**: Pre-deployment model validation
- **Progressive Rollouts**: Canary and blue-green deployments
- **Observability**: End-to-end pipeline monitoring

From [Google Cloud MLOps Guide](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (accessed 2025-01-31):
> MLOps level 2 focuses on CI/CD pipeline automation. For rapid and reliable model updates, you need automated CI/CD systems to build, test, and deploy ML pipeline components.

From [karpathy-deep-oracle/karpathy/practical-implementation/28-wandb-launch-cicd.md](../karpathy/practical-implementation/28-wandb-launch-cicd.md):
> W&B Launch enables seamless integration with CI/CD systems to create fully automated ML pipelines. By combining Launch with GitHub Actions, GitLab CI, and W&B Automations, you can trigger training jobs, evaluate models, and deploy to production automatically.

---

## Section 1: Pipeline Architecture Patterns

### Three-Layer CI/CD Architecture

**Layer 1: Code CI/CD (GitHub Actions / Cloud Build)**
- Lint and test Python code
- Build Docker containers
- Push to Artifact Registry
- Trigger training pipelines

**Layer 2: Training Orchestration (Vertex AI Pipelines / W&B Launch)**
- Data preprocessing pipelines
- Model training jobs
- Hyperparameter tuning
- Model evaluation

**Layer 3: Deployment CI/CD (Cloud Build / W&B Launch)**
- Model validation gates
- Endpoint deployment
- Traffic routing
- Rollback mechanisms

From [Google Cloud GitOps Tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/gitops-cloud-build) (accessed 2025-01-31):
> GitOps methodology uses Git as single source of truth for declarative infrastructure and applications. Changes are made via pull requests, automated testing validates changes, and approved changes are automatically applied.

**Complete Pipeline Flow:**

```
Code Commit → CI Tests → Container Build → Training Pipeline Trigger
     ↓            ↓              ↓                    ↓
  GitHub      Cloud Build   Artifact Registry   Vertex AI Pipelines
                                                      ↓
                                              W&B Launch Queue
                                                      ↓
                                              Model Training
                                                      ↓
                                              Evaluation Gate
                                                      ↓
                                              Model Registry
                                                      ↓
                                         Deployment Pipeline (Cloud Build)
                                                      ↓
                                              Vertex AI Endpoint
```

### Cloud Build + Vertex AI Integration

**Cloud Build Configuration for ML Training:**

```yaml
# cloudbuild.yaml - Complete ML CI/CD pipeline
steps:
  # Step 1: Build training container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:$COMMIT_SHA'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:latest'
      - '-f'
      - 'Dockerfile.training'
      - '.'

  # Step 2: Push container to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:$COMMIT_SHA'

  # Step 3: Run unit tests
  - name: 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:$COMMIT_SHA'
    entrypoint: 'pytest'
    args: ['tests/unit/', '-v']

  # Step 4: Compile Vertex AI pipeline
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - 'pipelines/compile_pipeline.py'
      - '--output-path'
      - 'pipeline.json'

  # Step 5: Submit pipeline to Vertex AI
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'ai'
      - 'custom-jobs'
      - 'create'
      - '--region=us-central1'
      - '--display-name=training-$SHORT_SHA'
      - '--config=pipeline.json'

  # Step 6: Notify W&B Launch
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install wandb
        wandb launch \
          --uri . \
          --git-hash $COMMIT_SHA \
          --queue vertex-training \
          --project ml-training \
          --entity $WANDB_ENTITY

options:
  machineType: 'N1_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY

substitutions:
  _WANDB_ENTITY: 'your-team'

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:$COMMIT_SHA'
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/ml-images/training:latest'
```

From [CI/CD for Machine Learning 2024](https://medium.com/infer-qwak/ci-cd-for-machine-learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2) (accessed 2025-01-31):
> Modern ML CI/CD extends traditional software practices with model-specific concerns: data validation, model testing, performance monitoring, and gradual rollouts.

### Vertex AI Pipelines + Kubeflow Integration

**Vertex AI Pipeline with Cloud Build Trigger:**

```python
# pipelines/training_pipeline.py
from kfp.v2 import dsl
from kfp.v2.dsl import component, Dataset, Model, Metrics
from google.cloud import aiplatform

@component(
    base_image='python:3.10',
    packages_to_install=['scikit-learn', 'pandas', 'google-cloud-aiplatform']
)
def preprocess_data(
    input_dataset: Dataset,
    output_dataset: Output[Dataset]
):
    """Preprocess training data"""
    import pandas as pd

    # Load data
    df = pd.read_csv(input_dataset.path)

    # Preprocessing logic
    df_processed = df.dropna()

    # Save processed data
    df_processed.to_csv(output_dataset.path, index=False)

@component(
    base_image='us-central1-docker.pkg.dev/project/ml-images/training:latest'
)
def train_model(
    training_data: Dataset,
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Train ML model"""
    import torch
    from model import MyModel

    # Load data
    data = load_data(training_data.path)

    # Train model
    model_obj = MyModel()
    loss, accuracy = model_obj.fit(data)

    # Save model
    torch.save(model_obj.state_dict(), model.path)

    # Log metrics
    metrics.log_metric('loss', loss)
    metrics.log_metric('accuracy', accuracy)

@component(
    base_image='python:3.10',
    packages_to_install=['google-cloud-aiplatform']
)
def evaluate_model(
    model: Input[Model],
    test_data: Dataset,
    evaluation_metrics: Output[Metrics]
) -> bool:
    """Evaluate model performance"""
    # Load model and test data
    model_obj = load_model(model.path)
    test_df = pd.read_csv(test_data.path)

    # Evaluate
    eval_results = model_obj.evaluate(test_df)

    # Log metrics
    evaluation_metrics.log_metric('test_accuracy', eval_results['accuracy'])
    evaluation_metrics.log_metric('test_f1', eval_results['f1'])

    # Gate: Only deploy if accuracy > 0.85
    return eval_results['accuracy'] > 0.85

@component(
    base_image='gcr.io/google.com/cloudsdktool/cloud-sdk'
)
def deploy_model(
    model: Input[Model],
    endpoint_name: str,
    project_id: str,
    region: str
):
    """Deploy model to Vertex AI Endpoint"""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Upload model to Model Registry
    uploaded_model = aiplatform.Model.upload(
        display_name=f"model-{endpoint_name}",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )

    # Create or get endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )

    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    # Deploy model
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=f"deployed-{endpoint_name}",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100
    )

@dsl.pipeline(
    name='ml-training-pipeline',
    description='Complete ML training and deployment pipeline'
)
def training_pipeline(
    input_dataset_path: str,
    test_dataset_path: str,
    endpoint_name: str,
    project_id: str,
    region: str = 'us-central1'
):
    """Complete training pipeline with deployment gate"""

    # Step 1: Preprocess data
    preprocess_task = preprocess_data(
        input_dataset=input_dataset_path
    )

    # Step 2: Train model
    train_task = train_model(
        training_data=preprocess_task.outputs['output_dataset']
    )

    # Step 3: Evaluate model
    eval_task = evaluate_model(
        model=train_task.outputs['model'],
        test_data=test_dataset_path
    )

    # Step 4: Deploy only if evaluation passes
    with dsl.Condition(eval_task.output == True, name='deploy-gate'):
        deploy_task = deploy_model(
            model=train_task.outputs['model'],
            endpoint_name=endpoint_name,
            project_id=project_id,
            region=region
        )

# Compile pipeline
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='pipeline.json'
)
```

From [Vertex AI Continuous Training Tutorial](https://docs.cloud.google.com/vertex-ai/docs/pipelines/continuous-training-tutorial) (accessed 2025-01-31):
> Build pipelines that automatically train custom models either on a periodic schedule or triggered by Cloud Functions when new data arrives in Cloud Storage.

### GitOps Configuration Management

**Infrastructure as Code for ML Pipelines:**

```yaml
# gitops/training-config.yaml
apiVersion: ml.google.com/v1
kind: TrainingPipeline
metadata:
  name: arr-coc-training
  namespace: production
spec:
  schedule: "0 2 * * 0"  # Weekly Sunday 2 AM

  dataset:
    gcs_path: "gs://arr-coc-data/training/"
    version: "v2.3"

  model:
    architecture: "arr-coc-base"
    config:
      learning_rate: 0.0001
      batch_size: 32
      max_epochs: 100
      early_stopping_patience: 5

  compute:
    machine_type: "n1-standard-32"
    accelerator_type: "NVIDIA_TESLA_V100"
    accelerator_count: 8
    spot_instances: true

  checkpointing:
    frequency: "300s"
    max_checkpoints: 10
    gcs_path: "gs://arr-coc-checkpoints/"

  monitoring:
    wandb_project: "arr-coc-training"
    wandb_entity: "northhead"
    enable_profiling: true

  deployment:
    auto_deploy: true
    evaluation_threshold:
      accuracy: 0.85
      f1_score: 0.80
    traffic_split:
      strategy: "canary"
      initial_percentage: 10
      increment_percentage: 30
      evaluation_period: "1h"
```

**Cloud Build Trigger for GitOps Updates:**

```yaml
# cloudbuild-gitops.yaml
steps:
  # Validate config changes
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - 'scripts/validate_config.py'
      - 'gitops/training-config.yaml'

  # Update Vertex AI Pipeline
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Apply updated pipeline configuration
        gcloud ai pipelines update \
          --region=us-central1 \
          --pipeline-name=arr-coc-training \
          --config=gitops/training-config.yaml

        # Trigger immediate run if requested
        if [ "$_IMMEDIATE_RUN" = "true" ]; then
          gcloud ai pipelines run \
            --region=us-central1 \
            --pipeline-name=arr-coc-training
        fi

substitutions:
  _IMMEDIATE_RUN: 'false'

options:
  logging: CLOUD_LOGGING_ONLY
```

From [GitOps for Cloud-Native Applications 2025](https://advansappz.com/how-gitops-is-transforming-ci-cd-for-cloud-native-applications-in-2025/) (accessed 2025-01-31):
> GitOps transforms CI/CD by treating infrastructure configuration as code, enabling automated deployments, version control for infrastructure, and simplified rollbacks through Git history.

---

## Section 2: Continuous Training Automation

### Event-Driven Training Triggers

**Cloud Function to Trigger Training on New Data:**

```python
# functions/trigger_training.py
import json
from google.cloud import aiplatform
from google.cloud import storage
import wandb

def trigger_training_on_new_data(event, context):
    """
    Triggered by Cloud Storage when new training data uploaded.

    Args:
        event: Cloud Storage event data
        context: Event metadata
    """
    file_name = event['name']
    bucket_name = event['bucket']

    print(f"New file detected: gs://{bucket_name}/{file_name}")

    # Check if file is in training data directory
    if not file_name.startswith('training-data/'):
        print("File not in training-data directory, ignoring")
        return

    # Initialize Vertex AI
    aiplatform.init(
        project='your-project-id',
        location='us-central1'
    )

    # Submit training pipeline
    pipeline = aiplatform.PipelineJob(
        display_name=f"training-triggered-by-{file_name}",
        template_path="gs://your-bucket/pipelines/training_pipeline.json",
        pipeline_root="gs://your-bucket/pipeline-runs/",
        parameter_values={
            'input_dataset_path': f'gs://{bucket_name}/{file_name}',
            'test_dataset_path': 'gs://your-bucket/test-data/test.csv',
            'endpoint_name': 'arr-coc-production',
            'project_id': 'your-project-id'
        }
    )

    pipeline.submit()

    print(f"Training pipeline submitted: {pipeline.resource_name}")

    # Also trigger W&B Launch job
    wandb_api = wandb.Api()
    queue = wandb_api.create_run_queue_item(
        queue_name='vertex-training',
        run_spec={
            'uri': 'https://github.com/your-org/ml-training',
            'entry_point': 'train.py',
            'parameters': {
                'data_path': f'gs://{bucket_name}/{file_name}'
            }
        }
    )

    print(f"W&B Launch job queued: {queue.id}")
```

**Terraform Configuration for Event Trigger:**

```hcl
# terraform/training-triggers.tf
resource "google_storage_bucket" "training_data" {
  name          = "arr-coc-training-data"
  location      = "US-CENTRAL1"
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_cloudfunctions_function" "trigger_training" {
  name        = "trigger-training-on-new-data"
  runtime     = "python310"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.function_source.name
  source_archive_object = google_storage_bucket_object.function_zip.name

  event_trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.training_data.name
  }

  entry_point = "trigger_training_on_new_data"

  environment_variables = {
    PROJECT_ID = var.project_id
    REGION     = var.region
    WANDB_API_KEY = var.wandb_api_key
  }
}

resource "google_cloud_scheduler_job" "periodic_training" {
  name             = "periodic-training-trigger"
  description      = "Trigger training pipeline weekly"
  schedule         = "0 2 * * 0"  # Sunday 2 AM
  time_zone        = "America/Los_Angeles"
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = "https://us-central1-aiplatform.googleapis.com/v1/projects/${var.project_id}/locations/us-central1/pipelineJobs"

    body = base64encode(jsonencode({
      displayName = "scheduled-training-${formatdate("YYYY-MM-DD", timestamp())}"
      templatePath = "gs://${google_storage_bucket.pipelines.name}/training_pipeline.json"
      pipelineRoot = "gs://${google_storage_bucket.pipeline_runs.name}/"
    }))

    oauth_token {
      service_account_email = google_service_account.pipeline_runner.email
    }
  }
}
```

### Data Drift Detection and Retraining

**Automated Data Drift Monitoring:**

```python
# monitoring/data_drift_detector.py
import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import aiplatform, storage
import wandb

class DataDriftDetector:
    """Detect data drift and trigger retraining"""

    def __init__(self, project_id, reference_data_path):
        self.project_id = project_id
        self.reference_data = self.load_data(reference_data_path)

        # Initialize W&B for logging
        wandb.init(project='ml-monitoring', job_type='drift-detection')

    def load_data(self, gcs_path):
        """Load data from GCS"""
        storage_client = storage.Client()

        # Parse GCS path
        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to DataFrame
        return pd.read_csv(blob.open('r'))

    def detect_drift(self, current_data_path, threshold=0.05):
        """
        Detect statistical drift using Kolmogorov-Smirnov test

        Args:
            current_data_path: Path to current production data
            threshold: P-value threshold for drift detection

        Returns:
            bool: True if drift detected
        """
        current_data = self.load_data(current_data_path)

        drift_scores = {}

        # Compare distributions for each feature
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['float64', 'int64']:
                # KS test for numerical features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[column],
                    current_data[column]
                )
                drift_scores[column] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drifted': p_value < threshold
                }

        # Log to W&B
        wandb.log({'drift_scores': drift_scores})

        # Check if any feature drifted
        drift_detected = any(
            score['drifted'] for score in drift_scores.values()
        )

        if drift_detected:
            print("⚠️ Data drift detected!")
            drifted_features = [
                col for col, score in drift_scores.items()
                if score['drifted']
            ]
            print(f"Drifted features: {drifted_features}")

        return drift_detected

    def trigger_retraining(self):
        """Trigger Vertex AI training pipeline"""
        aiplatform.init(project=self.project_id, location='us-central1')

        pipeline = aiplatform.PipelineJob(
            display_name="retraining-data-drift",
            template_path="gs://your-bucket/pipelines/training_pipeline.json",
            pipeline_root="gs://your-bucket/pipeline-runs/",
            parameter_values={
                'reason': 'data_drift_detected',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        )

        pipeline.submit()

        print(f"✓ Retraining pipeline submitted: {pipeline.resource_name}")

        # Log to W&B
        wandb.log({
            'retraining_triggered': True,
            'pipeline_id': pipeline.resource_name
        })

# Cloud Function handler
def check_drift_and_retrain(request):
    """Cloud Function to check drift and trigger retraining"""

    detector = DataDriftDetector(
        project_id='your-project-id',
        reference_data_path='gs://your-bucket/reference-data/train.csv'
    )

    drift_detected = detector.detect_drift(
        current_data_path='gs://your-bucket/production-data/current.csv',
        threshold=0.05
    )

    if drift_detected:
        detector.trigger_retraining()
        return {'status': 'retraining_triggered', 'drift_detected': True}
    else:
        return {'status': 'no_drift', 'drift_detected': False}
```

**Cloud Scheduler Configuration:**

```yaml
# cloud-scheduler-drift-check.yaml
steps:
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'scheduler'
      - 'jobs'
      - 'create'
      - 'http'
      - 'drift-detection-check'
      - '--schedule=0 */6 * * *'  # Every 6 hours
      - '--uri=https://us-central1-PROJECT_ID.cloudfunctions.net/check-drift-and-retrain'
      - '--http-method=POST'
      - '--oidc-service-account-email=drift-checker@PROJECT_ID.iam.gserviceaccount.com'
```

From [Build Pipeline for Continuous Training](https://docs.cloud.google.com/vertex-ai/docs/pipelines/continuous-training-tutorial) (accessed 2025-01-31):
> Vertex AI Pipelines can be triggered automatically when new data arrives, enabling continuous learning systems that adapt to changing data distributions.

### Model Performance Monitoring and Auto-Retraining

**Performance Degradation Detector:**

```python
# monitoring/performance_monitor.py
from google.cloud import aiplatform, monitoring_v3
import wandb
from datetime import datetime, timedelta

class PerformanceMonitor:
    """Monitor deployed model performance and trigger retraining"""

    def __init__(self, endpoint_name, project_id, region='us-central1'):
        self.endpoint_name = endpoint_name
        self.project_id = project_id
        self.region = region

        aiplatform.init(project=project_id, location=region)

        # Get endpoint
        self.endpoint = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )[0]

    def get_recent_predictions(self, hours=24):
        """Fetch recent prediction logs from Cloud Logging"""
        from google.cloud import logging

        client = logging.Client()

        # Query prediction logs
        filter_str = f'''
        resource.type="aiplatform.googleapis.com/Endpoint"
        resource.labels.endpoint_id="{self.endpoint.name.split("/")[-1]}"
        timestamp >= "{(datetime.utcnow() - timedelta(hours=hours)).isoformat()}Z"
        '''

        entries = client.list_entries(filter_=filter_str)

        return list(entries)

    def calculate_performance_metrics(self):
        """Calculate current performance metrics"""
        predictions = self.get_recent_predictions(hours=24)

        # Extract predictions and ground truth
        correct = 0
        total = 0

        for entry in predictions:
            if 'ground_truth' in entry.payload:
                prediction = entry.payload['prediction']
                ground_truth = entry.payload['ground_truth']

                if prediction == ground_truth:
                    correct += 1
                total += 1

        if total == 0:
            return None

        accuracy = correct / total

        return {
            'accuracy': accuracy,
            'total_predictions': total,
            'timestamp': datetime.utcnow().isoformat()
        }

    def check_performance_degradation(self, baseline_accuracy=0.85, threshold=0.05):
        """
        Check if performance degraded below threshold

        Args:
            baseline_accuracy: Expected baseline accuracy
            threshold: Acceptable degradation (e.g., 0.05 = 5% drop)

        Returns:
            bool: True if degradation detected
        """
        current_metrics = self.calculate_performance_metrics()

        if current_metrics is None:
            print("⚠️ No ground truth data available")
            return False

        current_accuracy = current_metrics['accuracy']

        # Log to W&B
        wandb.log({
            'current_accuracy': current_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'degradation': baseline_accuracy - current_accuracy
        })

        degradation = baseline_accuracy - current_accuracy

        if degradation > threshold:
            print(f"⚠️ Performance degradation detected!")
            print(f"Current: {current_accuracy:.3f}, Baseline: {baseline_accuracy:.3f}")
            print(f"Degradation: {degradation:.3f} (threshold: {threshold:.3f})")
            return True

        return False

    def trigger_retraining_pipeline(self):
        """Trigger retraining with current production data"""
        pipeline = aiplatform.PipelineJob(
            display_name="retraining-performance-degradation",
            template_path="gs://your-bucket/pipelines/training_pipeline.json",
            pipeline_root="gs://your-bucket/pipeline-runs/",
            parameter_values={
                'reason': 'performance_degradation',
                'current_endpoint': self.endpoint.name,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

        pipeline.submit()

        print(f"✓ Retraining triggered: {pipeline.resource_name}")

        # Create incident in Cloud Monitoring
        self.create_monitoring_incident()

    def create_monitoring_incident(self):
        """Create incident in Cloud Monitoring"""
        from google.cloud import monitoring_v3

        client = monitoring_v3.NotificationChannelServiceClient()

        # Log incident
        print("⚠️ Performance degradation incident created")

        wandb.log({
            'incident_created': True,
            'incident_type': 'performance_degradation'
        })

# Cloud Function handler
def monitor_performance(request):
    """Scheduled Cloud Function to monitor performance"""

    wandb.init(project='ml-monitoring', job_type='performance-check')

    monitor = PerformanceMonitor(
        endpoint_name='arr-coc-production',
        project_id='your-project-id'
    )

    degradation_detected = monitor.check_performance_degradation(
        baseline_accuracy=0.85,
        threshold=0.05
    )

    if degradation_detected:
        monitor.trigger_retraining_pipeline()
        return {
            'status': 'retraining_triggered',
            'reason': 'performance_degradation'
        }
    else:
        return {
            'status': 'performance_ok',
            'degradation_detected': False
        }
```

---

## Section 3: Deployment Automation and Rollout Strategies

### Canary Deployment with Gradual Traffic Shift

**Automated Canary Deployment:**

```python
# deployment/canary_deployment.py
from google.cloud import aiplatform
import time
import wandb

class CanaryDeployment:
    """Manage canary deployments with automated traffic shifting"""

    def __init__(self, endpoint_name, project_id, region='us-central1'):
        aiplatform.init(project=project_id, location=region)

        self.endpoint = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )[0]

        wandb.init(project='ml-deployments', job_type='canary-deployment')

    def deploy_canary_model(self, model_artifact_uri, canary_percentage=10):
        """
        Deploy new model with initial canary traffic

        Args:
            model_artifact_uri: GCS path to model artifacts
            canary_percentage: Initial traffic percentage (default 10%)
        """
        # Upload new model
        model = aiplatform.Model.upload(
            display_name=f"canary-model-{int(time.time())}",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        )

        print(f"✓ Model uploaded: {model.resource_name}")

        # Get current deployments
        deployed_models = self.endpoint.list_models()

        # Calculate traffic split
        traffic_split = {}

        # Existing model gets (100 - canary_percentage)%
        for deployed_model in deployed_models:
            traffic_split[deployed_model.id] = 100 - canary_percentage

        # Deploy canary with initial traffic
        deployed_model = self.endpoint.deploy(
            model=model,
            deployed_model_display_name=f"canary-{int(time.time())}",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
            traffic_percentage=canary_percentage,
            traffic_split=traffic_split
        )

        print(f"✓ Canary deployed with {canary_percentage}% traffic")

        wandb.log({
            'canary_deployed': True,
            'initial_traffic': canary_percentage,
            'model_id': deployed_model.id
        })

        return deployed_model.id

    def monitor_canary_metrics(self, canary_id, duration_minutes=60):
        """
        Monitor canary performance

        Args:
            canary_id: Deployed model ID
            duration_minutes: Monitoring duration

        Returns:
            dict: Performance metrics
        """
        import time
        from datetime import datetime, timedelta

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        print(f"Monitoring canary for {duration_minutes} minutes...")

        while datetime.utcnow() < end_time:
            # Get prediction metrics from Cloud Monitoring
            metrics = self.get_deployment_metrics(canary_id)

            wandb.log({
                'canary_latency': metrics['latency'],
                'canary_error_rate': metrics['error_rate'],
                'canary_qps': metrics['qps']
            })

            # Check for anomalies
            if metrics['error_rate'] > 0.05:  # 5% error threshold
                print(f"⚠️ High error rate detected: {metrics['error_rate']:.2%}")
                return {'status': 'failed', 'reason': 'high_error_rate'}

            time.sleep(60)  # Check every minute

        return {'status': 'passed', 'metrics': metrics}

    def shift_traffic(self, canary_id, target_percentage):
        """Gradually shift traffic to canary"""
        deployed_models = self.endpoint.list_models()

        traffic_split = {}
        for model in deployed_models:
            if model.id == canary_id:
                traffic_split[model.id] = target_percentage
            else:
                traffic_split[model.id] = 100 - target_percentage

        # Update traffic split
        self.endpoint.update(
            traffic_split=traffic_split
        )

        print(f"✓ Traffic shifted to {target_percentage}% for canary")

        wandb.log({
            'traffic_shift': target_percentage,
            'timestamp': datetime.utcnow().isoformat()
        })

    def promote_canary(self, canary_id):
        """Promote canary to production (100% traffic)"""
        self.shift_traffic(canary_id, 100)

        # Undeploy old models
        deployed_models = self.endpoint.list_models()
        for model in deployed_models:
            if model.id != canary_id:
                self.endpoint.undeploy(deployed_model_id=model.id)
                print(f"✓ Undeployed old model: {model.id}")

        print(f"✓ Canary promoted to production")

        wandb.log({
            'canary_promoted': True,
            'production_model_id': canary_id
        })

    def rollback_canary(self, canary_id):
        """Rollback canary deployment"""
        # Remove canary
        self.endpoint.undeploy(deployed_model_id=canary_id)

        # Restore 100% traffic to previous model
        deployed_models = self.endpoint.list_models()
        if deployed_models:
            self.shift_traffic(deployed_models[0].id, 100)

        print(f"✓ Canary rolled back")

        wandb.log({
            'canary_rolled_back': True,
            'reason': 'performance_issues'
        })

    def get_deployment_metrics(self, deployed_model_id):
        """Fetch metrics from Cloud Monitoring"""
        # Placeholder - integrate with Cloud Monitoring API
        return {
            'latency': 0.15,  # seconds
            'error_rate': 0.01,  # 1%
            'qps': 100  # queries per second
        }

# Complete canary deployment workflow
def automated_canary_deployment(model_artifact_uri):
    """
    Fully automated canary deployment with monitoring and rollout
    """
    deployment = CanaryDeployment(
        endpoint_name='arr-coc-production',
        project_id='your-project-id'
    )

    # Phase 1: Deploy canary with 10% traffic
    canary_id = deployment.deploy_canary_model(
        model_artifact_uri=model_artifact_uri,
        canary_percentage=10
    )

    # Phase 2: Monitor for 1 hour
    result = deployment.monitor_canary_metrics(
        canary_id=canary_id,
        duration_minutes=60
    )

    if result['status'] == 'failed':
        print("⚠️ Canary failed monitoring, rolling back")
        deployment.rollback_canary(canary_id)
        return {'status': 'failed', 'reason': result['reason']}

    # Phase 3: Gradual traffic shift (10% -> 30% -> 50% -> 100%)
    for percentage in [30, 50, 100]:
        deployment.shift_traffic(canary_id, percentage)

        # Monitor after each shift
        result = deployment.monitor_canary_metrics(
            canary_id=canary_id,
            duration_minutes=30
        )

        if result['status'] == 'failed':
            print(f"⚠️ Canary failed at {percentage}% traffic, rolling back")
            deployment.rollback_canary(canary_id)
            return {'status': 'failed', 'stage': f'{percentage}%'}

        time.sleep(60)  # Wait between shifts

    # Phase 4: Promote canary to production
    deployment.promote_canary(canary_id)

    return {'status': 'success', 'production_model_id': canary_id}
```

From [karpathy-deep-oracle/karpathy/practical-implementation/28-wandb-launch-cicd.md](../karpathy/practical-implementation/28-wandb-launch-cicd.md):
> Blue-green deployment strategy enables zero-downtime deployments by maintaining two production environments and switching traffic after validation.

### Blue-Green Deployment

**Cloud Build Pipeline for Blue-Green Deployment:**

```yaml
# cloudbuild-blue-green-deploy.yaml
steps:
  # Step 1: Build new model container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/models/arr-coc:green-$SHORT_SHA'
      - '.'

  # Step 2: Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/models/arr-coc:green-$SHORT_SHA'

  # Step 3: Deploy to "green" environment
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - '-c'
      - |
        from google.cloud import aiplatform

        aiplatform.init(project='$PROJECT_ID', location='us-central1')

        # Deploy to green endpoint
        model = aiplatform.Model.upload(
            display_name='arr-coc-green',
            artifact_uri='us-central1-docker.pkg.dev/$PROJECT_ID/models/arr-coc:green-$SHORT_SHA',
            serving_container_image_uri='us-central1-docker.pkg.dev/$PROJECT_ID/models/arr-coc:green-$SHORT_SHA'
        )

        endpoint = aiplatform.Endpoint.create(display_name='arr-coc-green')
        endpoint.deploy(model=model, machine_type='n1-standard-4')

  # Step 4: Run smoke tests against green
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'pytest'
    args:
      - 'tests/smoke/'
      - '--endpoint=arr-coc-green'
      - '-v'

  # Step 5: Switch traffic from blue to green
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - 'deployment/switch_traffic.py'
      - '--from=blue'
      - '--to=green'
      - '--strategy=immediate'

  # Step 6: Monitor green for issues
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - 'deployment/monitor_deployment.py'
      - '--endpoint=arr-coc-green'
      - '--duration=15'  # 15 minutes

  # Step 7: Teardown old blue environment (if green stable)
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'python'
    args:
      - 'deployment/cleanup_blue.py'
      - '--keep-as-backup=true'

timeout: '3600s'  # 1 hour timeout
```

### Model Registry and Version Management

**Automated Model Registry Updates:**

```python
# model_registry/registry_manager.py
from google.cloud import aiplatform
from datetime import datetime
import wandb

class ModelRegistryManager:
    """Manage model versions in Vertex AI Model Registry"""

    def __init__(self, project_id, region='us-central1'):
        aiplatform.init(project=project_id, location=region)
        self.project_id = project_id
        self.region = region

    def register_model(
        self,
        model_artifact_uri,
        model_name,
        version,
        metrics,
        metadata
    ):
        """
        Register new model version in Model Registry

        Args:
            model_artifact_uri: GCS path to model
            model_name: Base model name
            version: Version string (e.g., "1.2.0")
            metrics: Training metrics dict
            metadata: Additional metadata
        """
        # Upload model
        model = aiplatform.Model.upload(
            display_name=f"{model_name}-v{version}",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            labels={
                'version': version.replace('.', '-'),
                'stage': 'candidate',
                'trained_date': datetime.utcnow().strftime('%Y%m%d')
            }
        )

        # Add version alias
        model.add_version_aliases([f"v{version}", "latest-candidate"])

        # Log to Model Registry
        model.update(
            labels={
                **model.labels,
                'accuracy': str(metrics.get('accuracy', 0.0)),
                'f1_score': str(metrics.get('f1_score', 0.0))
            }
        )

        print(f"✓ Model registered: {model.resource_name}")
        print(f"  Version: {version}")
        print(f"  Metrics: {metrics}")

        # Log to W&B
        wandb.init(project='model-registry')
        wandb.log({
            'model_registered': True,
            'version': version,
            'model_id': model.name,
            **metrics
        })

        return model

    def promote_to_staging(self, model_name, version):
        """Promote model from candidate to staging"""
        models = aiplatform.Model.list(
            filter=f'display_name="{model_name}-v{version}"'
        )

        if not models:
            raise ValueError(f"Model {model_name}-v{version} not found")

        model = models[0]

        # Update stage label
        model.update(labels={
            **model.labels,
            'stage': 'staging'
        })

        # Update aliases
        model.remove_version_aliases(['latest-candidate'])
        model.add_version_aliases(['latest-staging', f'staging-v{version}'])

        print(f"✓ Model promoted to staging: {model_name}-v{version}")

        wandb.log({
            'model_promoted': True,
            'from_stage': 'candidate',
            'to_stage': 'staging',
            'version': version
        })

    def promote_to_production(self, model_name, version):
        """Promote model from staging to production"""
        models = aiplatform.Model.list(
            filter=f'display_name="{model_name}-v{version}"'
        )

        if not models:
            raise ValueError(f"Model {model_name}-v{version} not found")

        model = models[0]

        # Update stage label
        model.update(labels={
            **model.labels,
            'stage': 'production',
            'promoted_to_production': datetime.utcnow().isoformat()
        })

        # Update aliases
        model.remove_version_aliases(['latest-staging', f'staging-v{version}'])
        model.add_version_aliases(['production', f'production-v{version}'])

        print(f"✓ Model promoted to production: {model_name}-v{version}")

        wandb.log({
            'model_promoted': True,
            'from_stage': 'staging',
            'to_stage': 'production',
            'version': version
        })

    def get_production_model(self, model_name):
        """Get current production model"""
        models = aiplatform.Model.list(
            filter=f'labels.stage="production"'
        )

        production_models = [
            m for m in models
            if m.display_name.startswith(model_name)
        ]

        if not production_models:
            return None

        # Return most recently promoted
        return sorted(
            production_models,
            key=lambda m: m.labels.get('promoted_to_production', ''),
            reverse=True
        )[0]
```

---

## Section 4: Complete CI/CD Example for ARR-COC

### End-to-End Pipeline Configuration

**Complete GitHub Actions Workflow:**

```yaml
# .github/workflows/arr-coc-cicd.yaml
name: ARR-COC CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'arr_coc/**'
      - 'configs/**'
      - 'Dockerfile'

  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2 AM

  workflow_dispatch:
    inputs:
      deployment_strategy:
        description: 'Deployment strategy'
        required: true
        type: choice
        options:
          - canary
          - blue-green
          - immediate
      run_ablations:
        description: 'Run ablation studies'
        type: boolean
        default: false

env:
  PROJECT_ID: 'your-project-id'
  REGION: 'us-central1'
  ARTIFACT_REGISTRY: 'us-central1-docker.pkg.dev'
  WANDB_ENTITY: 'northhead'
  WANDB_PROJECT: 'arr-coc-training'

jobs:
  # ═══════════════════════════════════════════════════════════
  #  Stage 1: Code Quality and Testing
  # ═══════════════════════════════════════════════════════════

  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install black mypy pytest pytest-cov

      - name: Code quality checks
        run: |
          black --check arr_coc/
          mypy arr_coc/ --ignore-missing-imports

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=arr_coc --cov-report=term-missing

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # ═══════════════════════════════════════════════════════════
  #  Stage 2: Build and Push Container
  # ═══════════════════════════════════════════════════════════

  build-container:
    needs: lint-and-test
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Configure Docker
        run: gcloud auth configure-docker ${{ env.ARTIFACT_REGISTRY }}

      - name: Docker metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.ARTIFACT_REGISTRY }}/${{ env.PROJECT_ID }}/arr-coc/training
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: Dockerfile.training
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=registry,ref=${{ env.ARTIFACT_REGISTRY }}/${{ env.PROJECT_ID }}/arr-coc/training:cache
          cache-to: type=registry,ref=${{ env.ARTIFACT_REGISTRY }}/${{ env.PROJECT_ID }}/arr-coc/training:cache,mode=max

  # ═══════════════════════════════════════════════════════════
  #  Stage 3: Trigger Training Pipeline
  # ═══════════════════════════════════════════════════════════

  train-model:
    needs: build-container
    runs-on: ubuntu-latest
    outputs:
      run_id: ${{ steps.train.outputs.run_id }}
      model_uri: ${{ steps.train.outputs.model_uri }}
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Install dependencies
        run: |
          pip install google-cloud-aiplatform wandb kfp

      - name: Compile Vertex AI pipeline
        run: |
          python pipelines/compile_training_pipeline.py \
            --output-path pipeline.json \
            --image-tag ${{ needs.build-container.outputs.image_tag }}

      - name: Submit training job
        id: train
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python -c "
          from google.cloud import aiplatform
          import wandb

          aiplatform.init(project='${{ env.PROJECT_ID }}', location='${{ env.REGION }}')

          # Submit Vertex AI Pipeline
          pipeline = aiplatform.PipelineJob(
              display_name='arr-coc-training-${{ github.sha }}',
              template_path='pipeline.json',
              pipeline_root='gs://arr-coc-pipelines/runs/',
              parameter_values={
                  'image_uri': '${{ needs.build-container.outputs.image_tag }}',
                  'dataset_version': 'latest',
                  'git_commit': '${{ github.sha }}'
              }
          )

          pipeline.submit()

          # Also submit to W&B Launch
          wandb.login(key='${{ secrets.WANDB_API_KEY }}')
          run = wandb.launch(
              uri='.',
              git_hash='${{ github.sha }}',
              queue='vertex-gpu-training',
              project='${{ env.WANDB_PROJECT }}',
              entity='${{ env.WANDB_ENTITY }}'
          )

          print(f'::set-output name=run_id::{run.id}')
          print(f'::set-output name=pipeline_id::{pipeline.resource_name}')
          "

      - name: Comment on commit
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.repos.createCommitComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              commit_sha: context.sha,
              body: `🚀 Training pipeline submitted

              **Vertex AI Pipeline**: ${{ steps.train.outputs.pipeline_id }}
              **W&B Run**: https://wandb.ai/${{ env.WANDB_ENTITY }}/${{ env.WANDB_PROJECT }}/runs/${{ steps.train.outputs.run_id }}

              Monitor at: https://console.cloud.google.com/vertex-ai/pipelines`
            })

  # ═══════════════════════════════════════════════════════════
  #  Stage 4: Ablation Studies (Optional)
  # ═══════════════════════════════════════════════════════════

  ablation-studies:
    needs: train-model
    if: github.event.inputs.run_ablations == 'true'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        ablation:
          - no-propositional
          - no-perspectival
          - no-participatory
          - uniform-lod
          - no-opponent-processing
    steps:
      - uses: actions/checkout@v3

      - name: Run ablation study
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python pipelines/run_ablation.py \
            --ablation ${{ matrix.ablation }} \
            --baseline-run ${{ needs.train-model.outputs.run_id }}

  # ═══════════════════════════════════════════════════════════
  #  Stage 5: Model Evaluation
  # ═══════════════════════════════════════════════════════════

  evaluate-model:
    needs: train-model
    runs-on: ubuntu-latest
    outputs:
      evaluation_passed: ${{ steps.eval.outputs.passed }}
      metrics: ${{ steps.eval.outputs.metrics }}
    steps:
      - uses: actions/checkout@v3

      - name: Run evaluation suite
        id: eval
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python evaluation/run_benchmarks.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --benchmarks vqav2,gqa,textvqa,vizwiz \
            --output-file eval_results.json

          # Check if evaluation passed thresholds
          python -c "
          import json

          with open('eval_results.json') as f:
              results = json.load(f)

          passed = (
              results['vqav2']['accuracy'] > 0.70 and
              results['gqa']['accuracy'] > 0.65
          )

          print(f'::set-output name=passed::{passed}')
          print(f'::set-output name=metrics::{json.dumps(results)}')
          "

      - name: Upload evaluation report
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: eval_results.json

  # ═══════════════════════════════════════════════════════════
  #  Stage 6: Deploy to Staging
  # ═══════════════════════════════════════════════════════════

  deploy-staging:
    needs: [train-model, evaluate-model]
    if: needs.evaluate-model.outputs.evaluation_passed == 'true'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging endpoint
        run: |
          python deployment/deploy_model.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --endpoint arr-coc-staging \
            --traffic-percentage 100

      - name: Run regression tests
        run: |
          pytest tests/regression/ \
            --endpoint-url https://staging-arr-coc.example.com \
            -v

  # ═══════════════════════════════════════════════════════════
  #  Stage 7: Deploy to Production
  # ═══════════════════════════════════════════════════════════

  deploy-production:
    needs: [train-model, evaluate-model, deploy-staging]
    if: |
      needs.evaluate-model.outputs.evaluation_passed == 'true' &&
      github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v3

      - name: Canary deployment
        if: github.event.inputs.deployment_strategy == 'canary'
        run: |
          python deployment/canary_deployment.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --endpoint arr-coc-production \
            --initial-percentage 10 \
            --monitoring-duration 60

      - name: Blue-green deployment
        if: github.event.inputs.deployment_strategy == 'blue-green'
        run: |
          python deployment/blue_green_deployment.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --endpoint arr-coc-production

      - name: Immediate deployment
        if: github.event.inputs.deployment_strategy == 'immediate'
        run: |
          python deployment/deploy_model.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --endpoint arr-coc-production \
            --traffic-percentage 100

      - name: Update model registry
        run: |
          python model_registry/promote_to_production.py \
            --model-uri ${{ needs.train-model.outputs.model_uri }} \
            --version $(date +%Y.%m.%d)

      - name: Create release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v$(date +%Y.%m.%d)-${{ github.sha }}
          release_name: ARR-COC Model Release $(date +%Y-%m-%d)
          body: |
            ## Model Metrics
            ${{ needs.evaluate-model.outputs.metrics }}

            ## Deployment
            - Strategy: ${{ github.event.inputs.deployment_strategy }}
            - Endpoint: arr-coc-production
            - Model URI: ${{ needs.train-model.outputs.model_uri }}
```

From [karpathy-deep-oracle/karpathy/practical-implementation/28-wandb-launch-cicd.md](../karpathy/practical-implementation/28-wandb-launch-cicd.md):
> Complete ARR-COC CI/CD pipeline integrates code validation, container builds, training orchestration, evaluation gates, and progressive deployment strategies.

---

## Sources

**Source Documents:**
- [28-wandb-launch-cicd.md](../karpathy/practical-implementation/28-wandb-launch-cicd.md) - W&B Launch CI/CD integration patterns
- [35-vertex-ai-production-patterns.md](../karpathy/practical-implementation/35-vertex-ai-production-patterns.md) - Vertex AI production deployment strategies

**Web Research (accessed 2025-01-31):**
- [Google Cloud MLOps Guide](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) - MLOps continuous delivery and automation
- [GitOps with Cloud Build](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/gitops-cloud-build) - GitOps-style continuous delivery tutorial
- [Vertex AI Continuous Training](https://docs.cloud.google.com/vertex-ai/docs/pipelines/continuous-training-tutorial) - Build pipeline for continuous model training
- [CI/CD for Machine Learning 2024](https://medium.com/infer-qwak/ci-cd-for-machine-learning-in-2024-best-practices-to-build-test-and-deploy-c4ad869824d2) - Best practices for ML CI/CD
- [GitOps Transforming CI/CD 2025](https://advansappz.com/how-gitops-is-transforming-ci-cd-for-cloud-native-applications-in-2025/) - GitOps trends and patterns

**Related Documentation:**
- [30-vertex-ai-fundamentals.md](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md) - Vertex AI basics
- [31-wandb-launch-vertex-agent.md](../karpathy/practical-implementation/31-wandb-launch-vertex-agent.md) - W&B Launch agent setup
- [27-wandb-launch-cloud.md](../karpathy/practical-implementation/27-wandb-launch-cloud.md) - Cloud-based training orchestration
