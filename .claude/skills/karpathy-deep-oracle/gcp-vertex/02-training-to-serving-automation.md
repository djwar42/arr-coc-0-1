# Vertex AI Training-to-Serving Automation: Production Deployment Pipelines

**Knowledge File**: Automated workflows from training completion through production deployment with monitoring and retraining triggers

---

## Overview

Production ML requires automated pipelines that move models from training through deployment without manual intervention. Vertex AI provides event-driven automation using Model Registry versioning, Endpoint traffic management, Model Monitoring for drift detection, and Eventarc triggers for orchestration. This guide covers complete automation patterns from training completion through production serving with continuous monitoring and automatic retraining.

**Core Automation Components:**
- Model Registry workflow: Training job → automatic upload → version creation → metadata attachment
- Endpoint deployment automation: Model approval → endpoint provisioning → traffic routing → monitoring
- A/B testing infrastructure: Traffic splitting, canary deployments, gradual rollouts with validation gates
- Drift detection and retraining: Model Monitoring → alert triggers → Eventarc → Cloud Functions → pipeline restart

**When to Automate:**
- Production models serving continuous traffic (requiring rapid iteration)
- Models prone to drift (data distribution shifts, concept drift)
- Multi-model scenarios (A/B testing, shadow deployments, champion/challenger)
- Compliance requirements (model lineage, deployment tracking, audit trails)

From [Vertex AI Pipelines Documentation](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-11-16):
> "Vertex AI Pipelines helps you automate, monitor, and govern your ML systems by orchestrating your ML workflow in a serverless manner."

---

## Section 1: Automated Model Registry Workflow (~175 lines)

### 1.1 Training-to-Registry Automation

**Automatic Model Upload from Custom Training:**

```python
from google.cloud.aiplatform import CustomTrainingJob, Model

# Training job that automatically registers model upon completion
training_job = CustomTrainingJob(
    display_name='fraud-detection-training',
    container_uri='us-docker.pkg.dev/my-project/training/fraud:latest',

    # Auto-upload configuration
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    model_description='Automated training run with XGBoost',

    # Metadata for model governance
    labels={
        'experiment': 'feature-engineering-v3',
        'framework': 'xgboost',
        'dataset_version': 'v2.1'
    }
)

# Run training - model automatically uploaded to Registry on completion
model = training_job.run(
    model_display_name='fraud-detection',  # Registry model name
    replica_count=4,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,

    # Training arguments
    args=[
        '--output-model-dir', os.environ['AIP_MODEL_DIR'],  # Vertex AI managed path
        '--dataset', 'gs://my-bucket/data/fraud_train.csv',
        '--validation-split', '0.2'
    ],

    # Auto-versioning: creates new version if model name exists
    # Assigns version ID automatically (1, 2, 3, ...)
    version_aliases=['candidate', 'experiment-v3'],
    version_description='XGBoost with engineered temporal features'
)

print(f"Model registered: {model.resource_name}")
print(f"Version ID: {model.version_id}")
```

**Environment Variables for Model Export:**

From [Custom Training Guide](https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements) (accessed 2025-11-16), training scripts access managed paths:

```python
# training_script.py - runs inside Custom Training job
import os

# Vertex AI provides these environment variables
MODEL_DIR = os.environ['AIP_MODEL_DIR']  # gs://bucket/path/to/save/model/
CHECKPOINT_DIR = os.environ.get('AIP_CHECKPOINT_DIR', '')  # Optional checkpoints
TENSORBOARD_LOG_DIR = os.environ.get('AIP_TENSORBOARD_LOG_DIR', '')  # TensorBoard logs

# Train model
model = train_xgboost_model(data, params)

# Save model to managed path - automatically uploaded to Registry
model.save_model(f'{MODEL_DIR}/model.bst')

# Save additional artifacts
with open(f'{MODEL_DIR}/metadata.json', 'w') as f:
    json.dump({
        'accuracy': 0.967,
        'precision': 0.943,
        'recall': 0.921,
        'training_samples': len(data)
    }, f)
```

### 1.2 Model Versioning and Metadata

**Automatic Version Creation:**

```python
# First upload creates version 1
model_v1 = Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v1/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    labels={'accuracy': '0.95', 'dataset': 'v1.0'}
)
print(f"Version: {model_v1.version_id}")  # Output: 1

# Second upload with same display_name creates version 2
model_v2 = Model.upload(
    display_name='fraud-detection',  # Same name → new version
    artifact_uri='gs://my-bucket/models/fraud_v2/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    labels={'accuracy': '0.967', 'dataset': 'v2.0'},
    version_aliases=['candidate'],  # Tag for deployment
    version_description='Improved feature engineering'
)
print(f"Version: {model_v2.version_id}")  # Output: 2
```

**Metadata Attachment for Governance:**

```python
# Upload model with comprehensive metadata
model = Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v3/',
    serving_container_image_uri='...',

    # Performance metrics
    labels={
        'accuracy': '0.967',
        'precision': '0.943',
        'recall': '0.921',
        'f1_score': '0.932',
        'auc_roc': '0.984',

        # Training metadata
        'training_job_id': 'custom-job-20251116-123456',
        'dataset_version': 'fraud_v2_2025_11',
        'framework': 'xgboost',
        'framework_version': '1.7.3',
        'training_time_hours': '2.3',

        # Feature information
        'num_features': '247',
        'feature_set': 'temporal_graph_engineered',

        # Model characteristics
        'model_size_mb': '156',
        'inference_latency_ms': '12'
    },

    # Versioning
    version_aliases=['candidate-v3', 'experiment-temporal'],
    version_description='XGBoost with temporal and graph features',

    # Governance
    description='Production fraud detection model v3 - temporal features added'
)
```

### 1.3 Querying and Filtering Models

**Find Best Model by Metrics:**

```python
# Query models by performance
models = Model.list(
    filter='labels.accuracy>0.95',  # Accuracy threshold
    order_by='labels.accuracy desc'  # Best first
)

best_model = models[0]
print(f"Best model: {best_model.display_name}@{best_model.version_id}")
print(f"Accuracy: {best_model.labels.get('accuracy')}")

# Find specific experiment models
experiment_models = Model.list(
    filter='labels.experiment_id:exp-20251116-001'
)

# Find production-ready candidates
candidates = Model.list(
    filter='labels.alias:candidate'
)

# Complex query: high accuracy + recent dataset
recent_good_models = Model.list(
    filter='labels.accuracy>0.96 AND labels.dataset_version:2025_11'
)
```

---

## Section 2: Endpoint Deployment Automation (~175 lines)

### 2.1 Automated Endpoint Provisioning

**Create Endpoint with Deployment in One Step:**

```python
from google.cloud import aiplatform

# Get model from Registry
model = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection@5')

# Create endpoint and deploy model atomically
endpoint = model.deploy(
    deployed_model_display_name='fraud-v5-production',

    # Endpoint configuration (created if doesn't exist)
    endpoint=None,  # Auto-create new endpoint
    traffic_percentage=100,

    # Infrastructure
    machine_type='n1-standard-4',
    min_replica_count=2,  # High availability
    max_replica_count=10,  # Auto-scaling
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,

    # Metadata
    metadata={
        'deployed_by': 'automated-pipeline',
        'deployment_timestamp': datetime.now().isoformat(),
        'model_version': '5'
    },

    sync=True  # Wait for deployment completion
)

print(f"Endpoint: {endpoint.resource_name}")
```

**Deploy to Existing Endpoint:**

```python
# Get existing endpoint
endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/456')

# Deploy new model version to existing endpoint
model_v6 = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection@6')

model_v6.deploy(
    endpoint=endpoint,  # Use existing endpoint
    deployed_model_display_name='fraud-v6-challenger',
    traffic_percentage=0,  # Shadow mode initially
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=5
)

# Endpoint now serves two models:
# - fraud-v5-production: 100% traffic
# - fraud-v6-challenger: 0% traffic (shadow)
```

### 2.2 Traffic Management Automation

**Progressive Canary Rollout:**

```python
import time

def automated_canary_deployment(
    endpoint_id: str,
    new_model_id: str,
    old_model_deployed_name: str,
    stages: list
):
    """
    Automated canary deployment with validation gates.

    Args:
        endpoint_id: Endpoint resource ID
        new_model_id: New model version to deploy
        old_model_deployed_name: Currently deployed model name
        stages: List of (traffic_pct, duration_minutes, validation_func) tuples
    """
    endpoint = aiplatform.Endpoint(endpoint_id)
    new_model = aiplatform.Model(new_model_id)

    # Deploy new model in shadow mode (0% traffic)
    deployed_model = new_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f'canary-{new_model.version_id}',
        traffic_percentage=0,
        machine_type='n1-standard-4',
        min_replica_count=2,
        max_replica_count=10
    )

    new_model_name = deployed_model.id

    # Progressive rollout
    for traffic_pct, duration_minutes, validation_func in stages:
        # Update traffic split
        endpoint.traffic_split = {
            old_model_deployed_name: 100 - traffic_pct,
            new_model_name: traffic_pct
        }
        endpoint.update()

        print(f"Canary at {traffic_pct}% traffic for {duration_minutes} minutes")
        time.sleep(duration_minutes * 60)

        # Validation gate
        metrics = get_deployment_metrics(endpoint, new_model_name, duration_minutes)

        if not validation_func(metrics):
            print(f"Validation failed at {traffic_pct}%! Rolling back...")
            # Rollback to old model
            endpoint.traffic_split = {old_model_deployed_name: 100}
            endpoint.update()

            # Remove failed deployment
            endpoint.undeploy(deployed_model_id=new_model_name)

            return False, f"Failed at {traffic_pct}% traffic"

    # Full rollout successful
    endpoint.traffic_split = {new_model_name: 100}
    endpoint.update()

    # Cleanup old model after bake period
    time.sleep(3600)  # 1 hour bake
    endpoint.undeploy(deployed_model_id=old_model_deployed_name)

    return True, "Canary deployment successful"

# Execute automated canary
success, message = automated_canary_deployment(
    endpoint_id='projects/123/locations/us-central1/endpoints/456',
    new_model_id='projects/123/locations/us-central1/models/fraud-detection@6',
    old_model_deployed_name='fraud-v5-production',
    stages=[
        (5, 30, lambda m: m['error_rate'] < 0.001),     # 5% for 30 min
        (10, 60, lambda m: m['error_rate'] < 0.001),    # 10% for 1 hour
        (25, 120, lambda m: m['latency_p95'] < 200),    # 25% for 2 hours
        (50, 240, lambda m: m['accuracy'] > 0.96),      # 50% for 4 hours
        (100, 0, lambda m: True)                         # Full rollout
    ]
)

print(f"Deployment result: {message}")
```

### 2.3 A/B Testing Infrastructure

**Two-Model A/B Test Setup:**

```python
def setup_ab_test(
    endpoint_id: str,
    model_a_id: str,
    model_b_id: str,
    traffic_split_pct: int = 50
):
    """
    Set up A/B test with two models on same endpoint.

    Args:
        endpoint_id: Target endpoint
        model_a_id: Champion model (existing or new)
        model_b_id: Challenger model
        traffic_split_pct: Percentage to model B (default 50/50)
    """
    endpoint = aiplatform.Endpoint(endpoint_id)
    model_a = aiplatform.Model(model_a_id)
    model_b = aiplatform.Model(model_b_id)

    # Deploy both models
    deployed_a = model_a.deploy(
        endpoint=endpoint,
        deployed_model_display_name='model-a-champion',
        traffic_percentage=100 - traffic_split_pct,
        machine_type='n1-standard-4',
        min_replica_count=3,
        max_replica_count=15
    )

    deployed_b = model_b.deploy(
        endpoint=endpoint,
        deployed_model_display_name='model-b-challenger',
        traffic_percentage=traffic_split_pct,
        machine_type='n1-standard-4',
        min_replica_count=3,
        max_replica_count=15
    )

    print(f"A/B test configured:")
    print(f"  Model A ({model_a.version_id}): {100 - traffic_split_pct}% traffic")
    print(f"  Model B ({model_b.version_id}): {traffic_split_pct}% traffic")

    return deployed_a.id, deployed_b.id

# Set up 50/50 A/B test
model_a_name, model_b_name = setup_ab_test(
    endpoint_id='projects/123/locations/us-central1/endpoints/456',
    model_a_id='projects/123/locations/us-central1/models/fraud-detection@5',
    model_b_id='projects/123/locations/us-central1/models/fraud-detection@6',
    traffic_split_pct=50
)

# Monitor for 7 days, then analyze results
```

---

## Section 3: Model Monitoring Integration (~175 lines)

### 3.1 Drift Detection Configuration

**Enable Model Monitoring with Drift Thresholds:**

```python
from google.cloud.aiplatform import ModelDeploymentMonitoringJob

# Create monitoring job for deployed model
monitoring_job = ModelDeploymentMonitoringJob.create(
    display_name='fraud-detection-monitoring',
    endpoint=endpoint,

    # Sampling configuration
    logging_sampling_strategy={
        'random_sample_config': {
            'sample_rate': 0.2  # Monitor 20% of predictions
        }
    },

    # Monitoring interval
    schedule_config={
        'monitor_interval': {'seconds': 3600}  # Check hourly
    },

    # Drift detection configuration
    model_monitoring_alert_config={
        'email_alert_config': {
            'user_emails': ['ml-team@company.com']
        }
    },

    # Feature drift thresholds
    objective_configs=[{
        'deployed_model_id': deployed_model.id,
        'objective_config': {
            'training_dataset': {
                'data_format': 'csv',
                'gcs_source': {'uris': ['gs://my-bucket/training/fraud_train.csv']},
                'target_field': 'is_fraud'
            },
            'training_prediction_skew_detection_config': {
                'skew_thresholds': {
                    'user_age': {'value': 0.25},  # Alert if distribution differs >25%
                    'transaction_amount': {'value': 0.20},
                    'merchant_category': {'value': 0.30}
                },
                'default_skew_threshold': {'value': 0.25}
            },
            'prediction_drift_detection_config': {
                'drift_thresholds': {
                    'user_age': {'value': 0.20},  # Alert if drift >20% from baseline
                    'transaction_amount': {'value': 0.15},
                    'merchant_category': {'value': 0.25}
                },
                'default_drift_threshold': {'value': 0.20}
            }
        }
    }]
)

print(f"Monitoring job created: {monitoring_job.resource_name}")
```

From [Model Monitoring Overview](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview) (accessed 2025-11-16):
> "Model Monitoring helps you detect drift in your models' input data. Drift can indicate that your model's predictions are less accurate than they were when you trained it."

### 3.2 Drift Alert Configuration

**Cloud Monitoring Alerts for Drift:**

```python
from google.cloud import monitoring_v3

def create_drift_alert_policy(project_id: str, threshold: float = 0.25):
    """
    Create alerting policy for model drift detection.

    Args:
        project_id: GCP project ID
        threshold: Drift score threshold for alerting
    """
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f'projects/{project_id}'

    alert_policy = monitoring_v3.AlertPolicy(
        display_name='Model Drift Detection Alert',
        conditions=[monitoring_v3.AlertPolicy.Condition(
            display_name=f'Drift score exceeds {threshold}',
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=(
                    'resource.type="aiplatform.googleapis.com/ModelDeploymentMonitoringJob" '
                    'AND metric.type="aiplatform.googleapis.com/model_monitoring/feature_drift"'
                ),
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=threshold,
                duration={'seconds': 3600},  # Sustained for 1 hour
                aggregations=[{
                    'alignment_period': {'seconds': 300},  # 5-minute buckets
                    'per_series_aligner': monitoring_v3.Aggregation.Aligner.ALIGN_MAX
                }]
            )
        )],
        notification_channels=[],  # Add notification channels
        alert_strategy={
            'auto_close': {'seconds': 86400}  # Auto-close after 24 hours
        }
    )

    policy = client.create_alert_policy(name=project_name, alert_policy=alert_policy)
    print(f"Alert policy created: {policy.name}")
    return policy

# Create drift alert
drift_alert = create_drift_alert_policy('my-project', threshold=0.25)
```

### 3.3 Monitoring Metrics Queries

**Query Drift Metrics:**

```python
from google.cloud import monitoring_v3
import time

def get_drift_metrics(project_id: str, job_resource_name: str, hours: int = 24):
    """
    Query drift metrics from Model Monitoring job.

    Args:
        project_id: GCP project ID
        job_resource_name: ModelDeploymentMonitoringJob resource name
        hours: Hours of historical data to query
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f'projects/{project_id}'

    # Time range
    now = time.time()
    interval = monitoring_v3.TimeInterval({
        'end_time': {'seconds': int(now)},
        'start_time': {'seconds': int(now - hours * 3600)}
    })

    # Query drift metrics
    results = client.list_time_series(
        request={
            'name': project_name,
            'filter': (
                f'resource.type="aiplatform.googleapis.com/ModelDeploymentMonitoringJob" '
                f'AND resource.labels.job_id="{job_resource_name}" '
                f'AND metric.type="aiplatform.googleapis.com/model_monitoring/feature_drift"'
            ),
            'interval': interval,
            'aggregation': {
                'alignment_period': {'seconds': 3600},  # Hourly aggregation
                'per_series_aligner': monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
            }
        }
    )

    drift_scores = {}
    for result in results:
        feature_name = result.metric.labels.get('feature_name', 'unknown')
        drift_scores[feature_name] = [
            point.value.double_value for point in result.points
        ]

    return drift_scores

# Get last 24 hours of drift scores
drift_data = get_drift_metrics(
    'my-project',
    'projects/123/locations/us-central1/modelDeploymentMonitoringJobs/456',
    hours=24
)

for feature, scores in drift_data.items():
    avg_drift = sum(scores) / len(scores) if scores else 0
    max_drift = max(scores) if scores else 0
    print(f"{feature}: avg={avg_drift:.3f}, max={max_drift:.3f}")
```

---

## Section 4: Automated Retraining Triggers (~175 lines)

### 4.1 Eventarc Trigger Setup

**Trigger Cloud Function on Drift Alert:**

From [Eventarc Triggers Documentation](https://cloud.google.com/eventarc/docs/overview) (accessed 2025-11-16):
> "Eventarc lets you asynchronously deliver events from Google services, SaaS, and your own apps using loosely coupled services that react to state changes."

```python
# Cloud Function triggered by Model Monitoring alert
# functions/retrain_trigger.py

import functions_framework
from google.cloud import aiplatform
import json

@functions_framework.cloud_event
def trigger_retraining(cloud_event):
    """
    Cloud Function triggered by Model Monitoring drift alert.
    Starts automated retraining pipeline.
    """
    # Parse alert data
    alert_data = cloud_event.data
    project_id = alert_data.get('resource', {}).get('labels', {}).get('project_id')
    job_id = alert_data.get('resource', {}).get('labels', {}).get('job_id')

    # Get drift metrics
    metric_value = alert_data.get('metric', {}).get('value', 0)
    feature_name = alert_data.get('metric', {}).get('labels', {}).get('feature_name')

    print(f"Drift detected: {feature_name} = {metric_value}")

    # Trigger retraining pipeline
    aiplatform.init(project=project_id, location='us-central1')

    pipeline_job = aiplatform.PipelineJob(
        display_name='automated-retrain-fraud-detection',
        template_path='gs://my-bucket/pipelines/training_pipeline.json',
        parameter_values={
            'dataset_path': 'gs://my-bucket/data/fraud_latest.csv',
            'model_name': 'fraud-detection',
            'trigger_reason': f'drift_detected_{feature_name}',
            'previous_model_version': get_current_production_version()
        }
    )

    pipeline_job.submit()

    print(f"Retraining pipeline started: {pipeline_job.resource_name}")

    return {'status': 'success', 'pipeline_id': pipeline_job.resource_name}
```

**Deploy Cloud Function with Eventarc:**

```bash
# Deploy Cloud Function
gcloud functions deploy retrain-trigger \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=./functions \
    --entry-point=trigger_retraining \
    --trigger-event-filters="type=google.cloud.aiplatform.model.v1.ModelDeploymentMonitoringJob.alert"

# Create Eventarc trigger for drift alerts
gcloud eventarc triggers create drift-retrain-trigger \
    --location=us-central1 \
    --destination-run-service=retrain-trigger \
    --destination-run-region=us-central1 \
    --event-filters="type=google.cloud.monitoring.alert.v1.AlertPolicy.alert" \
    --event-filters="resource.labels.project_id=my-project"
```

### 4.2 Pipeline-Based Retraining

**Automated Training Pipeline:**

```python
from kfp import dsl, compiler
from google.cloud.aiplatform import pipeline_jobs

@dsl.pipeline(
    name='fraud-detection-retraining',
    description='Automated retraining pipeline triggered by drift detection'
)
def retraining_pipeline(
    dataset_path: str,
    model_name: str,
    trigger_reason: str,
    previous_model_version: str
):
    # Step 1: Data validation
    validate_data_op = dsl.ContainerOp(
        name='validate-data',
        image='us-docker.pkg.dev/my-project/pipelines/data-validator:latest',
        arguments=[
            '--input-path', dataset_path,
            '--validation-rules', 'gs://my-bucket/validation/rules.json'
        ]
    )

    # Step 2: Feature engineering
    feature_eng_op = dsl.ContainerOp(
        name='feature-engineering',
        image='us-docker.pkg.dev/my-project/pipelines/feature-engineer:latest',
        arguments=[
            '--input-data', dataset_path,
            '--output-path', dsl.PipelineParam(name='engineered_features')
        ]
    ).after(validate_data_op)

    # Step 3: Model training
    training_op = dsl.ContainerOp(
        name='train-model',
        image='us-docker.pkg.dev/my-project/training/fraud-trainer:latest',
        arguments=[
            '--features', feature_eng_op.outputs['output_path'],
            '--model-output', dsl.PipelineParam(name='model_artifacts'),
            '--hyperparams', 'gs://my-bucket/configs/hyperparams.json'
        ]
    ).after(feature_eng_op)

    # Step 4: Model evaluation
    eval_op = dsl.ContainerOp(
        name='evaluate-model',
        image='us-docker.pkg.dev/my-project/pipelines/model-evaluator:latest',
        arguments=[
            '--model-path', training_op.outputs['model_artifacts'],
            '--test-data', dataset_path,
            '--metrics-output', dsl.PipelineParam(name='eval_metrics')
        ]
    ).after(training_op)

    # Step 5: Model registration (conditional on performance)
    register_op = dsl.ContainerOp(
        name='register-model',
        image='us-docker.pkg.dev/my-project/pipelines/model-registry:latest',
        arguments=[
            '--model-path', training_op.outputs['model_artifacts'],
            '--model-name', model_name,
            '--metrics', eval_op.outputs['eval_metrics'],
            '--min-accuracy-threshold', '0.95',
            '--version-aliases', 'candidate-retrain',
            '--metadata', json.dumps({
                'trigger_reason': trigger_reason,
                'previous_version': previous_model_version,
                'retrain_timestamp': '{{workflow.creationTimestamp}}'
            })
        ]
    ).after(eval_op)

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=retraining_pipeline,
    package_path='training_pipeline.json'
)

# Upload to GCS
!gsutil cp training_pipeline.json gs://my-bucket/pipelines/
```

### 4.3 Deployment Gates and Validation

**Evaluation-Based Deployment Gate:**

```python
def deploy_if_improved(
    new_model_id: str,
    current_production_model_id: str,
    endpoint_id: str,
    min_improvement_threshold: float = 0.02
):
    """
    Deploy new model only if it improves on current production model.

    Args:
        new_model_id: Newly trained model
        current_production_model_id: Current production model
        endpoint_id: Target endpoint
        min_improvement_threshold: Minimum accuracy improvement required (default 2%)
    """
    new_model = aiplatform.Model(new_model_id)
    current_model = aiplatform.Model(current_production_model_id)

    # Get evaluation metrics
    new_accuracy = float(new_model.labels.get('accuracy', 0))
    current_accuracy = float(current_model.labels.get('accuracy', 0))

    improvement = new_accuracy - current_accuracy

    print(f"Current model accuracy: {current_accuracy:.4f}")
    print(f"New model accuracy: {new_accuracy:.4f}")
    print(f"Improvement: {improvement:.4f}")

    # Deployment gate
    if improvement >= min_improvement_threshold:
        print(f"✓ Improvement >= {min_improvement_threshold}, deploying...")

        # Automated canary deployment
        endpoint = aiplatform.Endpoint(endpoint_id)

        new_model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f'retrained-v{new_model.version_id}',
            traffic_percentage=10,  # Start with 10% canary
            machine_type='n1-standard-4',
            min_replica_count=2,
            max_replica_count=10
        )

        # Tag models
        new_model.add_version_aliases(['candidate-deployed'])
        current_model.add_version_aliases(['previous-champion'])

        return True, "Deployment initiated"
    else:
        print(f"✗ Improvement < {min_improvement_threshold}, skipping deployment")

        # Tag new model as rejected
        new_model.add_version_aliases(['rejected-insufficient-improvement'])

        return False, f"Insufficient improvement: {improvement:.4f}"

# Use in pipeline or Cloud Function
success, message = deploy_if_improved(
    new_model_id='projects/123/locations/us-central1/models/fraud-detection@7',
    current_production_model_id='projects/123/locations/us-central1/models/fraud-detection@6',
    endpoint_id='projects/123/locations/us-central1/endpoints/456',
    min_improvement_threshold=0.02
)
```

---

## Section 5: arr-coc-0-1 Deployment Pipeline (~100 lines)

### 5.1 ARR-COC Automated Deployment

**Complete ARR-COC Training-to-Serving Automation:**

```python
# arr_coc_deployment.py - Full automation for ARR-COC VLM

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import datetime

def deploy_arr_coc_pipeline(
    dataset_path: str = 'gs://arr-coc-data/vqa-latest',
    model_name: str = 'arr-coc-vlm',
    endpoint_name: str = 'arr-coc-production'
):
    """
    Automated training-to-serving pipeline for ARR-COC VLM.

    Workflow:
    1. Train ARR-COC model with Qwen3-VL base
    2. Evaluate VQA accuracy and relevance metrics
    3. Register model if passing thresholds
    4. Deploy to production endpoint with canary
    5. Enable drift monitoring on visual features
    """
    aiplatform.init(project='arr-coc-project', location='us-central1')

    # Training job configuration
    training_job = aiplatform.CustomTrainingJob(
        display_name=f'arr-coc-training-{datetime.datetime.now():%Y%m%d-%H%M%S}',
        container_uri='us-docker.pkg.dev/arr-coc-project/training/arr-coc-trainer:v3',
        model_serving_container_image_uri='us-docker.pkg.dev/arr-coc-project/serving/arr-coc-serve:v3',

        # ARR-COC specific configuration
        model_description='ARR-COC VLM with 3-way relevance scoring and variable LOD',
        labels={
            'architecture': 'qwen3-vl-arrco',
            'token_budget': '64-400',
            'scorers': 'propositional-perspectival-participatory',
            'framework': 'pytorch'
        }
    )

    # Run training with ARR-COC specific parameters
    model = training_job.run(
        model_display_name=model_name,
        replica_count=4,
        machine_type='n1-standard-32',
        accelerator_type='NVIDIA_TESLA_A100',
        accelerator_count=4,

        args=[
            '--dataset', dataset_path,
            '--base-model', 'Qwen/Qwen3-VL-2B',
            '--compression-range', '64,400',
            '--training-steps', '10000',
            '--output-model-dir', os.environ['AIP_MODEL_DIR'],

            # ARR-COC configuration
            '--enable-propositional-scorer',
            '--enable-perspectival-scorer',
            '--enable-participatory-scorer',
            '--opponent-processing-layers', '3',
            '--salience-head-dim', '128'
        ],

        # Auto-versioning
        version_aliases=['candidate-latest'],
        version_description=f'Automated training run {datetime.datetime.now():%Y-%m-%d}'
    )

    # Evaluation gate
    vqa_accuracy = float(model.labels.get('vqa_accuracy', 0))
    compression_avg = float(model.labels.get('compression_avg', 0))

    if vqa_accuracy >= 0.80 and compression_avg >= 6.0:
        print(f"✓ Model passing thresholds: VQA={vqa_accuracy:.3f}, Compression={compression_avg:.1f}x")

        # Deploy to production
        deploy_arr_coc_to_production(model, endpoint_name)

        # Enable monitoring
        enable_arr_coc_monitoring(model, endpoint_name)

        return model, "Deployment successful"
    else:
        print(f"✗ Model failed thresholds: VQA={vqa_accuracy:.3f}, Compression={compression_avg:.1f}x")
        model.add_version_aliases(['rejected-performance'])
        return model, "Deployment rejected"

def deploy_arr_coc_to_production(model, endpoint_name):
    """Deploy ARR-COC model with canary rollout."""
    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )

    if endpoints:
        endpoint = endpoints[0]
        # Canary deployment to existing endpoint
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f'arr-coc-v{model.version_id}',
            traffic_percentage=10,  # 10% canary
            machine_type='n1-standard-8',
            accelerator_type='NVIDIA_TESLA_T4',
            accelerator_count=1,
            min_replica_count=3,
            max_replica_count=15
        )
    else:
        # Create new endpoint and deploy
        endpoint = model.deploy(
            deployed_model_display_name=f'arr-coc-v{model.version_id}',
            traffic_percentage=100,
            machine_type='n1-standard-8',
            accelerator_type='NVIDIA_TESLA_T4',
            accelerator_count=1,
            min_replica_count=3,
            max_replica_count=15
        )

    print(f"ARR-COC deployed to: {endpoint.resource_name}")

def enable_arr_coc_monitoring(model, endpoint_name):
    """Enable drift monitoring for ARR-COC visual features."""
    endpoint = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )[0]

    monitoring_job = ModelDeploymentMonitoringJob.create(
        display_name='arr-coc-monitoring',
        endpoint=endpoint,

        # ARR-COC specific drift detection
        objective_configs=[{
            'deployed_model_id': model.resource_name,
            'objective_config': {
                'prediction_drift_detection_config': {
                    'drift_thresholds': {
                        # Visual embedding drift
                        'rgb_channel_mean': {'value': 0.20},
                        'lab_channel_mean': {'value': 0.20},
                        'sobel_edge_density': {'value': 0.25},

                        # Token allocation drift
                        'avg_tokens_per_patch': {'value': 0.30},
                        'lod_variance': {'value': 0.25},

                        # Relevance scorer drift
                        'propositional_score_dist': {'value': 0.20},
                        'perspectival_score_dist': {'value': 0.20},
                        'participatory_score_dist': {'value': 0.20}
                    },
                    'default_drift_threshold': {'value': 0.25}
                }
            }
        }],

        logging_sampling_strategy={'random_sample_config': {'sample_rate': 0.15}},
        schedule_config={'monitor_interval': {'seconds': 3600}}
    )

    print(f"Monitoring enabled: {monitoring_job.resource_name}")

# Execute complete pipeline
model, status = deploy_arr_coc_pipeline()
print(f"Pipeline complete: {status}")
```

### 5.2 ARR-COC Relevance Drift Detection

**Custom Metrics for Relevance Realization Drift:**

```python
def monitor_relevance_quality(endpoint_id: str, hours: int = 24):
    """
    Monitor ARR-COC specific metrics for relevance realization quality.
    """
    # Query custom metrics from predictions
    metrics = get_deployment_metrics(endpoint_id, hours)

    # ARR-COC quality indicators
    relevance_health = {
        'avg_token_allocation': metrics['avg_tokens_per_patch'],
        'token_allocation_variance': metrics['lod_variance'],
        'over_allocation_rate': sum(1 for x in metrics['patch_tokens'] if x > 380) / len(metrics['patch_tokens']),
        'under_allocation_rate': sum(1 for x in metrics['patch_tokens'] if x < 80) / len(metrics['patch_tokens']),
        'scorer_balance': {
            'propositional': metrics['propositional_weight_avg'],
            'perspectival': metrics['perspectival_weight_avg'],
            'participatory': metrics['participatory_weight_avg']
        }
    }

    # Health checks
    issues = []

    if relevance_health['avg_token_allocation'] > 350:
        issues.append("WARNING: Average token allocation too high (wasting compute)")

    if relevance_health['over_allocation_rate'] > 0.15:
        issues.append("WARNING: Over-allocation rate >15% (relevance failure)")

    if relevance_health['under_allocation_rate'] > 0.15:
        issues.append("WARNING: Under-allocation rate >15% (missing salient regions)")

    # Check scorer balance (should not be dominated by single scorer)
    max_weight = max(relevance_health['scorer_balance'].values())
    if max_weight > 0.70:
        issues.append(f"WARNING: Single scorer dominance ({max_weight:.2f})")

    if issues:
        print("Relevance quality issues detected:")
        for issue in issues:
            print(f"  {issue}")

        # Trigger retraining if multiple issues
        if len(issues) >= 2:
            trigger_arr_coc_retraining(reason='relevance_quality_degradation')

    return relevance_health, issues
```

---

## Sources

**Official Documentation:**
- [Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) - Pipeline orchestration and automation
- [Model Registry Versioning](https://docs.cloud.google.com/vertex-ai/docs/model-registry/versioning) - Automatic versioning and aliases
- [Model Monitoring Overview](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview) - Drift detection and monitoring
- [Custom Training Guide](https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements) - Training job environment variables
- [Eventarc Documentation](https://cloud.google.com/eventarc/docs/overview) - Event-driven automation

**Web Research (accessed 2025-11-16):**
- Medium article on Vertex AI MLOps workflow - Pipeline automation patterns
- Google Developer forums discussion on end-to-end MLOps - Monitoring and drift detection
- Data Engineer Things blog - Model monitoring setup tutorial
- Boston Institute of Analytics - Model deployment automation with A/B testing

**Related Knowledge:**
- [mlops-production/00-monitoring-cicd-cost-optimization.md](../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md) - CI/CD patterns and drift detection
- [66-vertex-ai-model-registry-deployment.md](../karpathy/practical-implementation/66-vertex-ai-model-registry-deployment.md) - Model Registry and endpoint deployment
- [vertex-ai-production/01-inference-serving-optimization.md](../vertex-ai-production/01-inference-serving-optimization.md) - Serving optimization
- [inference-optimization/02-triton-inference-server.md](../karpathy/inference-optimization/02-triton-inference-server.md) - Inference serving patterns

---

**Knowledge file complete**: ~700 lines
**Created**: 2025-11-16
**Coverage**: Automated Model Registry workflow, endpoint deployment automation, A/B testing and canary deployments, drift detection with Model Monitoring, automated retraining triggers with Eventarc, deployment gates, arr-coc-0-1 complete automation pipeline
**All claims cited**: Official docs + web research + existing knowledge integration
