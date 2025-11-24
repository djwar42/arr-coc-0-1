# Vertex AI Model Registry & Deployment: Production Deployment Patterns

**Knowledge File: Endpoint deployment, traffic management, and production optimization for Vertex AI**

---

## Overview

Vertex AI Model Registry provides the foundation for systematic model lifecycle management, but production deployment requires careful orchestration of endpoints, traffic routing, and scaling strategies. This guide focuses on practical deployment patterns from model registration through production serving, with emphasis on zero-downtime updates, A/B testing, and cost-effective auto-scaling.

**Core Deployment Workflow:**
- Model uploaded to Registry → Version created → Endpoint provisioned → Model deployed → Traffic routed → Auto-scaling configured → Monitoring enabled

**When to Use Vertex AI Endpoints:**
- Production inference requiring high availability (99.9% SLA)
- Dynamic traffic splitting for A/B testing and canary deployments
- Auto-scaling based on request load (1-100+ replicas)
- Multi-model serving from single endpoint (cost optimization)
- Integration with Google Cloud ecosystem (IAM, VPC, monitoring)

**When NOT to Use Vertex AI Endpoints:**
- Batch prediction workloads (use Vertex AI Batch Prediction instead)
- Development/testing with low QPS (<1 request/minute sustained)
- Extreme cost sensitivity with unpredictable traffic (cold starts cost money)
- Multi-cloud requirements (endpoints are GCP-specific)

From [Vertex AI Endpoints Overview](https://docs.cloud.google.com/vertex-ai/docs/predictions/overview) (accessed 2025-02-03):
> "Vertex AI provides online prediction endpoints that support autoscaling, traffic splitting, and high availability for production inference workloads."

---

## Section 1: Model Registry Overview (~100 lines)

### 1.1 What is Vertex AI Model Registry?

Vertex AI Model Registry serves as the central hub for organizing ML models throughout their lifecycle, from initial upload through production deployment. Unlike traditional file-based model storage, the Registry provides versioning, metadata tracking, and direct deployment integration.

**Model Registry Architecture:**

**Model Resources:**
- **Model**: Top-level container representing a specific ML use case
- **Model Version**: Numbered iterations (1, 2, 3...) pointing to model artifacts
- **Model Alias**: Human-readable labels ("champion", "challenger", "staging")
- **Deployed Model**: Instance of a model version running on an endpoint

**Resource Hierarchy:**
```
Project
└── Location (us-central1)
    └── Model (fraud-detection)
        ├── Version 1 [deprecated]
        ├── Version 2 [staging]
        ├── Version 3 [champion] → Endpoint A (80% traffic)
        └── Version 4 [challenger] → Endpoint A (20% traffic)
```

**Metadata Storage:**
- Training metrics (accuracy, precision, recall, custom metrics)
- Framework and runtime information (TensorFlow 2.14, PyTorch 2.0)
- Training lineage (dataset versions, training job IDs, experiment tracking)
- Custom labels and tags for governance
- Deployment history (which endpoints, traffic percentages, timestamps)

From [Model Registry Versioning](https://docs.cloud.google.com/vertex-ai/docs/model-registry/versioning) (accessed 2025-02-03):
> "Model versioning lets you create multiple versions of the same model. With model versioning, you can organize your models in a way that helps navigate and manage their lifecycle."

### 1.2 Model Versions and Aliases

**Automatic Versioning:**

Vertex AI automatically increments version numbers when uploading models with the same display name:

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# First upload creates version 1
model_v1 = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v1/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
)
print(f"Created version: {model_v1.version_id}")  # Output: 1

# Second upload with same display_name creates version 2
model_v2 = aiplatform.Model.upload(
    display_name='fraud-detection',  # Same name → new version
    artifact_uri='gs://my-bucket/models/fraud_v2/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
)
print(f"Created version: {model_v2.version_id}")  # Output: 2
```

**Alias Management:**

Aliases provide semantic labels for versions, enabling deployment strategies without hardcoding version numbers:

```python
# Assign aliases during upload
model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v3/',
    version_aliases=['candidate', 'testing'],
    serving_container_image_uri='...'
)

# Add aliases to existing versions
model = aiplatform.Model('projects/123/locations/us-central1/models/456@3')
model.add_version_aliases(['champion', 'production'])

# Remove aliases
model.remove_version_aliases(['testing'])

# List models by alias
models = aiplatform.Model.list(filter='labels.alias:champion')
```

**Common Alias Patterns:**

| Alias | Purpose | Typical Traffic |
|-------|---------|-----------------|
| `dev` | Development testing | 0% (shadow mode) |
| `staging` | Pre-production validation | 0-10% |
| `candidate` | A/B test contender | 10-30% |
| `champion` | Current production winner | 70-100% |
| `challenger` | Competing production model | 10-30% |
| `deprecated` | Old version (rollback target) | 0% |

### 1.3 Lineage Tracking

**Training Job Lineage:**

```python
from google.cloud.aiplatform import CustomJob, Model

# Training job automatically links to model
training_job = CustomJob(
    display_name='fraud-training-v3',
    worker_pool_specs=[{...}],
    labels={'experiment': 'feature-engineering-v2'}
)
training_job.run()

# Upload model with lineage metadata
model = Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v3/',
    labels={
        'training_job': training_job.resource_name,
        'dataset_version': 'v2.1',
        'experiment_id': 'exp-20250203-001'
    }
)

# Query lineage
print(f"Model trained by: {model.labels.get('training_job')}")
```

---

## Section 2: Registering Models (~150 lines)

### 2.1 Upload from Training Job

**Automatic Registration from Custom Training:**

```python
from google.cloud.aiplatform import CustomTrainingJob, Model

# Define training job that outputs model
training_job = CustomTrainingJob(
    display_name='fraud-detection-training',
    container_uri='us-docker.pkg.dev/my-project/training/fraud-trainer:latest',
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
)

# Run training and automatically upload model to Registry
model = training_job.run(
    model_display_name='fraud-detection',  # Creates/updates model in Registry
    replica_count=4,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
    args=[
        '--output-model-dir', '/gcs/my-bucket/models/fraud_v4/',
        '--dataset', 'gs://my-bucket/data/fraud_train.csv'
    ]
)

print(f"Model registered: {model.resource_name}")
print(f"Version: {model.version_id}")
```

### 2.2 Upload Custom Container Models

**Custom Prediction Container:**

For frameworks not natively supported by Vertex AI, use custom containers:

```python
# 1. Build custom serving container (Dockerfile)
"""
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY predictor.py .

# Vertex AI expects HTTP server on port 8080
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

CMD ["python", "predictor.py"]
"""

# 2. Push container to Artifact Registry
# docker build -t us-docker.pkg.dev/my-project/models/custom-fraud:v1 .
# docker push us-docker.pkg.dev/my-project/models/custom-fraud:v1

# 3. Upload model with custom container
model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_custom_v1/',  # Model artifacts
    serving_container_image_uri='us-docker.pkg.dev/my-project/models/custom-fraud:v1',
    serving_container_ports=[8080],
    serving_container_predict_route='/predict',
    serving_container_health_route='/health',
    description='Custom XGBoost model with preprocessor'
)
```

**Custom Container Requirements:**

From [Custom Container Prediction](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements) (accessed 2025-02-03):
- HTTP server listening on `$AIP_HTTP_PORT` (default 8080)
- Health check endpoint returning 200 OK
- Prediction endpoint accepting POST with JSON body
- Response format: `{"predictions": [...]}`

### 2.3 Model Metadata and Labels

**Comprehensive Metadata:**

```python
model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v5/',
    serving_container_image_uri='...',

    # Versioning
    version_aliases=['candidate', 'experiment-feature-eng'],
    version_description='Improved feature engineering with temporal features',

    # Performance metadata
    labels={
        'accuracy': '0.967',
        'precision': '0.943',
        'recall': '0.921',
        'f1_score': '0.932',
        'auc_roc': '0.984',
        'training_dataset': 'fraud_v2_2025_02',
        'framework': 'xgboost',
        'framework_version': '1.7.3',
        'training_time_hours': '2.3',
        'num_features': '247',
        'model_size_mb': '156'
    },

    # Governance
    description='XGBoost fraud detection model with temporal and graph features',
    explanation_metadata=explanation_metadata,  # For Vertex AI Explainability
    explanation_parameters=explanation_parameters
)
```

**Querying by Metadata:**

```python
# Find best model by accuracy
models = aiplatform.Model.list(
    filter='labels.accuracy>0.95',
    order_by='labels.accuracy desc'
)
best_model = models[0]

# Find models from specific experiment
experiment_models = aiplatform.Model.list(
    filter='labels.experiment_id:exp-20250203-001'
)

# Find production models
prod_models = aiplatform.Model.list(
    filter='labels.alias:champion OR labels.alias:challenger'
)
```

---

## Section 3: Endpoint Deployment (~200 lines)

### 3.1 Creating Endpoints

**Basic Endpoint Creation:**

```python
from google.cloud import aiplatform

# Create endpoint (container for deployed models)
endpoint = aiplatform.Endpoint.create(
    display_name='fraud-detection-endpoint',
    description='Production fraud detection inference endpoint',
    labels={'environment': 'production', 'team': 'risk-ml'}
)

print(f"Endpoint created: {endpoint.resource_name}")
print(f"Endpoint URL: {endpoint.gca_resource.deployed_models}")
```

**Endpoint vs Deployed Model:**
- **Endpoint**: Infrastructure resource with stable URL (can host multiple models)
- **Deployed Model**: Specific model version deployed to an endpoint with traffic allocation

### 3.2 Deploying Models to Endpoints

**Single Model Deployment:**

```python
# Deploy model version to endpoint
model = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection@5')

deployed_model = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='fraud-v5-champion',
    machine_type='n1-standard-4',
    min_replica_count=2,  # Minimum instances (HA)
    max_replica_count=10,  # Maximum for auto-scaling
    accelerator_type='NVIDIA_TESLA_T4',  # Optional GPU
    accelerator_count=1,
    traffic_percentage=100,  # 100% of endpoint traffic
    sync=True  # Wait for deployment to complete
)

print(f"Deployed model ID: {deployed_model.id}")
```

**Machine Type Selection:**

From [Configure Compute Resources](https://docs.cloud.google.com/vertex-ai/docs/predictions/configure-compute) (accessed 2025-02-03):

| Machine Type | vCPUs | Memory | Use Case | Cost/Hour |
|--------------|-------|--------|----------|-----------|
| `n1-standard-2` | 2 | 7.5 GB | Small models, low QPS | $0.095 |
| `n1-standard-4` | 4 | 15 GB | Medium models, moderate QPS | $0.190 |
| `n1-standard-8` | 8 | 30 GB | Large models, high QPS | $0.380 |
| `n1-highmem-4` | 4 | 26 GB | Memory-intensive models | $0.237 |
| `n1-standard-4 + T4` | 4 + GPU | 15 GB + 16GB | GPU inference | $0.530 |
| `n1-standard-8 + V100` | 8 + GPU | 30 GB + 16GB | Large GPU models | $2.870 |

### 3.3 Auto-Scaling Configuration

**Auto-Scaling Behavior:**

From [Autoscaling Documentation](https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling) (accessed 2025-02-03):
> "Vertex AI Inference autoscaling scales the number of inference nodes based on the number of concurrent requests. This lets you dynamically adjust compute resources to match demand."

**Auto-Scaling Parameters:**

```python
# Deploy with auto-scaling configuration
deployed_model = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='fraud-detection-autoscale',
    machine_type='n1-standard-4',

    # Auto-scaling configuration
    min_replica_count=2,   # Always-on minimum (HA)
    max_replica_count=20,  # Scale up to 20 replicas

    # Scaling metrics (based on concurrent requests)
    # Vertex AI targets ~60% GPU/CPU utilization
    # With T4 GPU, scales at ~10-15 concurrent requests per replica
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,

    traffic_percentage=100
)
```

**Scaling Triggers:**

Auto-scaling uses these metrics (automatic, no manual configuration needed):
- **CPU utilization**: Target 60% average across replicas
- **GPU utilization**: Target 60% average (if GPUs present)
- **Request queue depth**: Scale up if requests queuing >30 seconds

**Scaling Behavior:**
- **Scale-up**: Triggered within 60 seconds of increased load
- **Scale-down**: Gradual reduction after 5+ minutes of low utilization
- **Minimum replicas**: Never scales below `min_replica_count` (prevents cold starts)

### 3.4 Multi-Model Deployment

**Deploy Multiple Models to Single Endpoint:**

```python
endpoint = aiplatform.Endpoint.create(display_name='fraud-multi-model')

# Deploy champion model (70% traffic)
model_v3 = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection@3')
model_v3.deploy(
    endpoint=endpoint,
    deployed_model_display_name='fraud-v3-champion',
    machine_type='n1-standard-4',
    min_replica_count=3,
    max_replica_count=15,
    traffic_percentage=70
)

# Deploy challenger model (30% traffic for A/B testing)
model_v4 = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection@4')
model_v4.deploy(
    endpoint=endpoint,
    deployed_model_display_name='fraud-v4-challenger',
    machine_type='n1-standard-4',
    min_replica_count=2,
    max_replica_count=10,
    traffic_percentage=30
)

# Traffic automatically split 70/30 across models
```

**Cost Optimization with Multi-Model Serving:**

Single endpoint hosting multiple models shares infrastructure costs:
- Endpoint overhead: ~$0.50/hour (regardless of # models)
- Model replicas: Charged independently based on machine type
- Total cost = Endpoint + (Model A replicas × cost) + (Model B replicas × cost)

---

## Section 4: Traffic Management (~150 lines)

### 4.1 Traffic Splitting Between Models

**Dynamic Traffic Allocation:**

```python
endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/456')

# Update traffic split (percentages must sum to 100)
endpoint.traffic_split = {
    'fraud-v3-champion': 80,    # Deployed model display name
    'fraud-v4-challenger': 20
}
endpoint.update()

print(f"Traffic updated: {endpoint.traffic_split}")
```

**Traffic Splitting Use Cases:**

| Pattern | Traffic Split | Use Case |
|---------|---------------|----------|
| **Single Model** | 100% to one model | Standard production |
| **A/B Testing** | 50/50 or 70/30 | Compare model performance |
| **Canary Deployment** | 95/5, 90/10, ... | Gradual rollout |
| **Multi-Armed Bandit** | 60/20/20 | Test 3+ models |
| **Shadow Mode** | 100/0 | New model gets traffic copy (no responses sent) |

### 4.2 A/B Testing Strategies

**Two-Model A/B Test:**

```python
# Phase 1: Deploy challenger in shadow mode (0% traffic)
model_v5.deploy(
    endpoint=endpoint,
    deployed_model_display_name='fraud-v5-shadow',
    machine_type='n1-standard-4',
    min_replica_count=1,
    traffic_percentage=0  # Shadow mode
)

# Monitor metrics for 24 hours, then increase traffic

# Phase 2: Start A/B test (10% traffic to challenger)
endpoint.traffic_split = {
    'fraud-v4-champion': 90,
    'fraud-v5-shadow': 10
}
endpoint.update()

# Monitor for 48 hours, compare metrics:
# - Prediction latency (p50, p95, p99)
# - Accuracy (ground truth labels from downstream validation)
# - Error rates
# - Business metrics (fraud detection rate, false positive rate)

# Phase 3: Increase to 50/50 if metrics favorable
endpoint.traffic_split = {
    'fraud-v4-champion': 50,
    'fraud-v5-shadow': 50
}
endpoint.update()

# Phase 4: Promote v5 to champion (100%)
endpoint.traffic_split = {
    'fraud-v5-shadow': 100
}
endpoint.update()

# Phase 5: Remove old model (optional)
endpoint.undeploy(deployed_model_id='fraud-v4-champion')
```

### 4.3 Canary Deployments

**Progressive Canary Rollout:**

```python
import time

def canary_rollout(endpoint, old_model_id, new_model_id, stages):
    """
    Progressive canary deployment with validation gates.

    stages: List of (traffic_percent, duration_hours, validation_func)
    """
    for traffic_pct, duration_hours, validation_func in stages:
        # Update traffic split
        endpoint.traffic_split = {
            old_model_id: 100 - traffic_pct,
            new_model_id: traffic_pct
        }
        endpoint.update()

        print(f"Canary at {traffic_pct}% for {duration_hours}h")
        time.sleep(duration_hours * 3600)

        # Validation gate
        metrics = get_deployment_metrics(endpoint, new_model_id)
        if not validation_func(metrics):
            print(f"Validation failed! Rolling back from {traffic_pct}%")
            endpoint.traffic_split = {old_model_id: 100}
            endpoint.update()
            return False

    # Full rollout
    endpoint.traffic_split = {new_model_id: 100}
    endpoint.update()
    return True

# Execute canary deployment
canary_rollout(
    endpoint=endpoint,
    old_model_id='fraud-v4',
    new_model_id='fraud-v5',
    stages=[
        (5, 1, lambda m: m['error_rate'] < 0.001),    # 5% for 1 hour
        (10, 2, lambda m: m['error_rate'] < 0.001),   # 10% for 2 hours
        (25, 4, lambda m: m['latency_p95'] < 200),    # 25% for 4 hours
        (50, 12, lambda m: m['accuracy'] > 0.96),     # 50% for 12 hours
        (100, 0, lambda m: True)                       # Full rollout
    ]
)
```

### 4.4 Blue-Green Deployments

**Two-Endpoint Blue-Green Strategy:**

```python
# Create two endpoints
blue_endpoint = aiplatform.Endpoint.create(display_name='fraud-blue')
green_endpoint = aiplatform.Endpoint.create(display_name='fraud-green')

# Blue endpoint serves production (v4)
model_v4.deploy(
    endpoint=blue_endpoint,
    machine_type='n1-standard-4',
    min_replica_count=5,
    traffic_percentage=100
)

# Deploy new version to green endpoint
model_v5.deploy(
    endpoint=green_endpoint,
    machine_type='n1-standard-4',
    min_replica_count=5,
    traffic_percentage=100
)

# Test green endpoint
test_predictions(green_endpoint)

# Switch traffic to green (external to Vertex AI)
# - Update Cloud Load Balancer backend
# - Update DNS records
# - Update application configuration

# After validation, green becomes new blue
```

**External Traffic Routing:**

Vertex AI endpoints don't support DNS-based traffic switching. Use:
- **Cloud Load Balancer**: Backend service switching
- **API Gateway**: Route configuration update
- **Application-level**: Config update + gradual client migration

### 4.5 Rollback Strategies

**Immediate Rollback:**

```python
def emergency_rollback(endpoint, previous_model_id):
    """Instant rollback to previous model version."""
    endpoint.traffic_split = {previous_model_id: 100}
    endpoint.update()  # Takes effect in 10-30 seconds

    # Alert team
    send_alert(
        severity='CRITICAL',
        message=f'Emergency rollback executed on {endpoint.display_name}'
    )

# Detect anomaly and rollback
if current_error_rate > 0.01:  # 1% error threshold
    emergency_rollback(endpoint, 'fraud-v4-previous')
```

---

## Section 5: Monitoring & Optimization (~100 lines)

### 5.1 Prediction Monitoring

**Vertex AI Built-in Metrics:**

Automatic metrics exported to Cloud Monitoring:
- `online_prediction_count`: Total predictions served
- `online_prediction_latency`: Request latency (p50, p95, p99)
- `online_prediction_error_count`: Failed predictions
- `replica_count`: Active replicas (for auto-scaling monitoring)
- `cpu_utilization`: CPU usage per replica
- `gpu_utilization`: GPU usage per replica (if GPUs enabled)

**Query Metrics:**

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f'projects/{project_id}'

# Query prediction latency (p95)
query = client.query_time_series(
    request={
        'name': project_name,
        'filter': (
            'resource.type="aiplatform.googleapis.com/Endpoint" '
            'AND metric.type="aiplatform.googleapis.com/online_prediction_latency" '
            'AND resource.labels.endpoint_id="1234567890"'
        ),
        'interval': {
            'end_time': {'seconds': int(time.time())},
            'start_time': {'seconds': int(time.time() - 3600)}
        },
        'aggregation': {
            'alignment_period': {'seconds': 60},
            'per_series_aligner': monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_95
        }
    }
)

for result in query:
    for point in result.points:
        print(f'P95 Latency: {point.value.double_value}ms')
```

### 5.2 Latency Optimization

**Latency Breakdown:**

Total latency = Network + Preprocessing + Inference + Postprocessing + Network

**Optimization Strategies:**

```python
# 1. Use GPUs for large models (10-100× faster inference)
deployed_model = model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',  # 16GB GPU
    accelerator_count=1,
    min_replica_count=2
)

# 2. Batch predictions (reduce per-request overhead)
# Client-side batching before sending to endpoint

# 3. Use larger machine types for CPU-bound models
deployed_model = model.deploy(
    machine_type='n1-highmem-8',  # More CPU cores
    min_replica_count=3
)

# 4. Dedicated endpoints for low-latency requirements
# From https://cloud.google.com/blog/products/ai-machine-learning/reliable-ai-with-vertex-ai-prediction-dedicated-endpoints
# Dedicated endpoints provide isolated infrastructure (no noisy neighbors)
```

**Target Latencies:**

| Use Case | Target P95 Latency | Recommended Setup |
|----------|-------------------|-------------------|
| Real-time fraud detection | <100ms | GPU (T4), min 3 replicas |
| Chatbot responses | <200ms | CPU (n1-standard-4), min 2 replicas |
| Recommendation engine | <50ms | GPU (T4), min 5 replicas |
| Image classification | <150ms | GPU (T4), min 2 replicas |
| Batch-like online | <1000ms | CPU (n1-standard-2), min 1 replica |

### 5.3 Cost Management

**Cost Breakdown:**

From [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-02-03):

Endpoint costs = Replica hours × Machine type cost

**Example Monthly Cost:**

```python
# Deployment: fraud-detection endpoint
# Machine type: n1-standard-4 ($0.190/hour)
# Auto-scaling: min 3, max 15 replicas
# Average replicas: 5 (during business hours), 3 (overnight)

# Business hours (12h/day × 5 days × 4 weeks = 240 hours)
business_hours_cost = 240 hours × 5 replicas × $0.190 = $228

# Overnight (12h/day × 5 days × 4 weeks = 240 hours)
overnight_cost = 240 hours × 3 replicas × $0.190 = $136.80

# Weekends (48h × 4 weeks = 192 hours)
weekend_cost = 192 hours × 3 replicas × $0.190 = $109.44

# Total monthly cost = $474.24
```

**Cost Optimization Strategies:**

```python
# 1. Right-size machine types (avoid over-provisioning)
# Profile model to determine actual resource needs

# 2. Reduce min_replica_count for non-critical endpoints
deployed_model = model.deploy(
    min_replica_count=1,  # Accept occasional cold starts
    max_replica_count=10
)

# 3. Use Spot VMs for batch-like workloads (not supported for online endpoints)
# Instead, use Vertex AI Batch Prediction for cost savings

# 4. Undeploy unused models
endpoint.undeploy(deployed_model_id='old-experimental-model')

# 5. Consolidate multiple models onto single endpoint
# Share infrastructure overhead across models
```

### 5.4 Model Performance Tracking

**Custom Metrics for Model Quality:**

```python
from google.cloud import monitoring_v3

def log_model_accuracy(endpoint_id, model_version, accuracy, precision, recall):
    """Log custom model quality metrics to Cloud Monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f'projects/{project_id}'

    series = monitoring_v3.TimeSeries()
    series.metric.type = 'custom.googleapis.com/ml/model_accuracy'
    series.metric.labels['endpoint_id'] = endpoint_id
    series.metric.labels['model_version'] = model_version

    series.resource.type = 'generic_task'
    series.resource.labels['project_id'] = project_id
    series.resource.labels['location'] = 'us-central1'

    now = time.time()
    point = monitoring_v3.Point({
        'interval': {
            'end_time': {'seconds': int(now)}
        },
        'value': {'double_value': accuracy}
    })

    series.points = [point]
    client.create_time_series(name=project_name, time_series=[series])

# Log metrics after batch validation
log_model_accuracy(
    endpoint_id='1234567890',
    model_version='5',
    accuracy=0.967,
    precision=0.943,
    recall=0.921
)
```

**Alerting on Model Degradation:**

```python
# Create alert for accuracy drop
from google.cloud import monitoring_v3

def create_accuracy_alert(project_id, threshold=0.95):
    client = monitoring_v3.AlertPolicyServiceClient()

    alert_policy = monitoring_v3.AlertPolicy(
        display_name='Model Accuracy Degradation',
        conditions=[monitoring_v3.AlertPolicy.Condition(
            display_name=f'Accuracy below {threshold}',
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter='metric.type="custom.googleapis.com/ml/model_accuracy"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_LT,
                threshold_value=threshold,
                duration={'seconds': 3600}  # 1 hour sustained
            )
        )],
        notification_channels=[]  # Add notification channels
    )

    project_name = f'projects/{project_id}'
    client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

---

## ARR-COC Connection

### VLM Endpoint Deployment for ARR-COC

**ARR-COC Model Registry Strategy:**

```python
# Register ARR-COC model with relevance-specific metadata
arr_coc_model = aiplatform.Model.upload(
    display_name='arr-coc-vlm',
    artifact_uri='gs://arr-coc-models/v3.0/',
    serving_container_image_uri='us-docker.pkg.dev/my-project/arr-coc/serving:v3',
    labels={
        'architecture': 'qwen3-vl-arrco',
        'token_budget': '64-400',
        'compression_avg': '8x',
        'propositional_scorer': 'shannon_entropy_v2',
        'perspectival_scorer': 'jung_archetypal_v2',
        'participatory_scorer': 'cross_attention_v2',
        'training_dataset': 'vqav2_coco_textcaps',
        'vqa_accuracy': '0.843',
        'inference_latency_p95': '127ms'
    },
    version_aliases=['candidate-v3']
)
```

**Multi-Scorer A/B Testing:**

```python
endpoint = aiplatform.Endpoint.create(display_name='arr-coc-production')

# Champion: All three scorers (propositional + perspectival + participatory)
arr_coc_full = aiplatform.Model('projects/123/.../arr-coc-vlm@10')
arr_coc_full.deploy(
    endpoint=endpoint,
    deployed_model_display_name='arr-coc-full-3-scorers',
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=3,
    traffic_percentage=70
)

# Challenger: Participatory-only (ablation study)
arr_coc_participatory = aiplatform.Model('projects/123/.../arr-coc-vlm@11')
arr_coc_participatory.deploy(
    endpoint=endpoint,
    deployed_model_display_name='arr-coc-participatory-only',
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=2,
    traffic_percentage=30
)

# Monitor VQA accuracy and relevance quality metrics
# Compare token allocation efficiency (64-400 range utilization)
# Measure compression effectiveness (target 8× average)
```

**Rollback for Relevance Failures:**

```python
# Detect relevance realization failure
def check_arr_coc_health(endpoint):
    metrics = get_recent_predictions(endpoint)

    # Check for over-allocation (wasting tokens on background)
    avg_tokens_per_patch = metrics['total_tokens'] / metrics['num_patches']
    if avg_tokens_per_patch > 350:  # Too close to max (400)
        return False

    # Check for under-allocation (missing salient regions)
    if metrics['min_tokens_per_patch'] < 80:  # Too close to min (64)
        return False

    # Check VQA accuracy
    if metrics['vqa_accuracy'] < 0.80:
        return False

    return True

# Auto-rollback if health check fails
if not check_arr_coc_health(endpoint):
    endpoint.traffic_split = {
        'arr-coc-v2-stable': 100,  # Previous stable version
        'arr-coc-v3-candidate': 0
    }
    endpoint.update()
    send_alert('ARR-COC v3 rolled back due to relevance failures')
```

**Vertex AI Model Registry enables ARR-COC production deployment** with systematic versioning of relevance scorer configurations, traffic-split A/B testing of opponent processing variations, and instant rollback when relevance realization degrades.

---

## Sources

**Web Research (accessed 2025-02-03):**
- [Vertex AI Endpoints Overview](https://docs.cloud.google.com/vertex-ai/docs/predictions/overview) - Inference workflow and endpoint concepts
- [Model Registry Versioning](https://docs.cloud.google.com/vertex-ai/docs/model-registry/versioning) - Version management and aliases
- [Deploy Model to Endpoint](https://docs.cloud.google.com/vertex-ai/docs/general/deployment) - Deployment procedures
- [Autoscaling Documentation](https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling) - Auto-scaling behavior and configuration
- [Configure Compute Resources](https://docs.cloud.google.com/vertex-ai/docs/predictions/configure-compute) - Machine types and resource selection
- [Custom Container Prediction](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements) - Custom serving containers
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) - Cost structure and pricing details
- [Vertex AI Dedicated Endpoints](https://cloud.google.com/blog/products/ai-machine-learning/reliable-ai-with-vertex-ai-prediction-dedicated-endpoints) - Dedicated endpoint architecture (May 2025)
- [Traffic Splitting Patterns](https://engineering.sada.com/vertex-ai-serving-architecture-for-real-time-machine-learning-c61674d8969) - Real-time ML architecture (SADA Engineering, Nov 2022)
- [Dual Deployments](https://cloud.google.com/blog/topics/developers-practitioners/dual-deployments-vertex-ai) - Multi-model deployment patterns (Sep 2021)

**Related Knowledge:**
- [66-vertex-ai-model-registry.md](66-vertex-ai-model-registry.md) - Model Registry fundamentals and lifecycle
- [30-vertex-ai-fundamentals.md](30-vertex-ai-fundamentals.md) - Base Vertex AI concepts
- [35-vertex-ai-production-patterns.md](35-vertex-ai-production-patterns.md) - High availability and cost optimization
