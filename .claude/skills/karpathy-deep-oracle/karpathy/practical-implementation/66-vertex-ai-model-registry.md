# Vertex AI Model Registry: Lifecycle Management and Production Deployment

**Knowledge File: Model versioning, deployment, and governance for production ML systems**

---

## Overview

Vertex AI Model Registry is Google Cloud's centralized repository for managing the complete lifecycle of machine learning models. Unlike traditional model storage approaches where each training run creates a separate model entry, the Model Registry provides systematic versioning, metadata tracking, and deployment orchestration from a single organized platform.

**Core Value Proposition:**
- Unified model lifecycle management from training to production
- Automatic versioning and lineage tracking without manual bookkeeping
- Direct deployment to endpoints with traffic splitting and A/B testing
- Integration with BigQuery ML, AutoML, and custom training workflows
- Governance and compliance through access controls and audit trails

**When to Use Vertex AI Model Registry:**
- Managing multiple versions of production models (A/B testing, rollback)
- Teams requiring centralized model discovery and reuse
- Organizations with regulatory compliance and audit requirements
- Projects needing automated deployment pipelines with approval gates
- Scenarios requiring model performance monitoring across versions

**When NOT to Use Vertex AI Model Registry:**
- Simple proof-of-concept experiments (use notebooks directly)
- Single-model projects with no versioning needs
- Multi-cloud requirements (prefer MLflow or cloud-agnostic solutions)
- Extreme cost sensitivity with infrequent model updates

From [Optimizing MLOps on Vertex AI](https://promevo.com/blog/optimizing-mlops-on-vertex-ai) (accessed 2025-01-13):
> "The Model Registry helps produce models more efficiently by establishing a single organized platform for model lineage, discovery, and lifecycle management after training."

---

## Section 1: Model Registry Architecture (~90 lines)

### 1.1 What is Vertex AI Model Registry?

Vertex AI Model Registry serves as the central repository for organizing, tracking, and managing ML models throughout their production lifecycle. It provides a structured environment where models are grouped by purpose with multiple versions tracked automatically.

**Key Architectural Components:**

**Model Groups:**
- Logical groupings of related model versions (e.g., "fraud-detection", "image-classifier-v2")
- Single namespace for all iterations of a specific model
- Enables version comparison and lineage tracking within the group

**Model Versions:**
- Automatically incremented version numbers (1, 2, 3, etc.)
- Each version points to specific model artifacts in Cloud Storage
- Immutable once registered (cannot modify artifact, only metadata)

**Model Aliases:**
- Human-readable labels assigned to versions ("champion", "challenger", "staging", "prod")
- Multiple aliases can point to the same version
- Enables deployment strategies without hardcoding version numbers
- Updated dynamically as models are promoted through stages

**Metadata Storage:**
- Performance metrics (accuracy, precision, recall, F1)
- Training parameters and hyperparameters
- Framework versions and dependencies
- Lineage information (training job, dataset versions, experiment tracking)
- Custom labels and annotations for governance

From [Google Vertex AI Model Registry and Versioning](https://medium.com/google-cloud/google-vertex-ai-model-versioning-72696bccd0d2) (accessed 2025-01-13):
> "Until recently, most of us were confronted with an overly cluttered list of models... Each version we train was uploaded as a separate model. Luckily this has changed, as Google introduced features to manage model versions properly."

### 1.2 Model Registry vs Traditional Storage

**Traditional Approach (Pre-Registry):**
```
fraud_model_2024_01_15/
fraud_model_2024_01_22_experiment/
fraud_model_final/
fraud_model_final_v2/
fraud_model_production/  # Which version? Who knows!
```

**Model Registry Approach:**
```
Model: fraud-detection
├── Version 1 (2024-01-15) [deprecated]
├── Version 2 (2024-01-22) [staging]
├── Version 3 (2024-02-01) [champion] → prod endpoint
└── Version 4 (2024-02-10) [challenger] → 10% traffic
```

**Benefits of Registry Architecture:**
- **Version Control**: Automatic numbering prevents naming chaos
- **Lineage Tracking**: Links models to training jobs, datasets, and experiments
- **Deployment Safety**: Aliases enable safe rollback without URL changes
- **Collaboration**: Centralized discovery prevents duplicate work
- **Governance**: Access controls and audit logs for compliance

### 1.3 Supported Model Types

**Custom Models:**
- TensorFlow SavedModel format
- PyTorch models (via TorchServe or custom containers)
- scikit-learn models (pickled or joblib)
- XGBoost, LightGBM models
- Custom prediction containers (any framework)

**AutoML Models:**
- AutoML Tables (tabular data)
- AutoML Vision (image classification, object detection)
- AutoML Text (sentiment analysis, entity extraction)
- AutoML Video (action recognition, classification)

**BigQuery ML Models:**
- Linear/logistic regression, DNN classifiers
- Boosted trees, K-means clustering
- Matrix factorization, ARIMA time series
- Direct registration from BigQuery to Vertex Registry

**Model Garden Models:**
- Pre-trained foundation models (PaLM, Gemini)
- Open-source models from HuggingFace, PyTorch Hub
- Fine-tuned versions of base models

From [Building Effective Model Registry](https://www.projectpro.io/article/model-registry/874) (accessed 2025-01-13):
> "The Vertex AI Model Registry is a central repository for managing ML models. It offers an organized overview of models, allowing for better organization, tracking, and training of new versions."

---

## Section 2: Model Lifecycle Management (~110 lines)

### 2.1 Model Upload and Registration

**Initial Model Registration:**

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Upload and register model as new model group
model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_model_v1/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    description='XGBoost fraud detection model - baseline',
    labels={'team': 'risk', 'use_case': 'fraud'}
)

print(f"Registered model: {model.resource_name}")
# Output: projects/123/locations/us-central1/models/456@1
```

**Registering New Versions:**

```python
# Add new version to existing model group
model_v2 = aiplatform.Model.upload(
    display_name='fraud-detection',  # Same name = new version
    artifact_uri='gs://my-bucket/models/fraud_model_v2/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    description='XGBoost fraud detection - improved feature engineering',
    parent_model='projects/123/locations/us-central1/models/456',  # Links to model group
    model_id='456',  # Same model_id creates new version
    version_aliases=['challenger', 'staging']  # Assign aliases immediately
)

print(f"Registered version: {model_v2.version_id}")
# Output: 2
```

**From Training Pipelines:**

```python
from google.cloud.aiplatform import pipeline_jobs

# Pipeline automatically registers model to registry
pipeline_job = pipeline_jobs.PipelineJob(
    display_name='fraud-detection-training',
    template_path='training_pipeline.yaml',
    parameter_values={
        'model_display_name': 'fraud-detection',  # Target model group
        'register_model': True,
        'version_aliases': ['candidate']
    }
)

pipeline_job.run()
```

### 2.2 Versioning Strategies

**Semantic Versioning Pattern:**

While Vertex uses numeric versions (1, 2, 3), teams often map to semantic versions via labels:

```python
# Label-based semantic versioning
model.update(
    labels={
        'semantic_version': 'v2.1.3',  # Major.Minor.Patch
        'breaking_changes': 'false',
        'feature_additions': 'true'
    }
)
```

**Stage-Based Versioning:**

```python
# Typical promotion workflow
dev_model = upload_model(version_aliases=['dev'])
# After validation
dev_model.add_version_alias('staging')
# After A/B testing
dev_model.remove_version_alias('dev')
dev_model.add_version_alias('champion')
```

**Experimental Branch Versioning:**

```python
# Multiple concurrent experiments
model_baseline = upload_model(
    display_name='fraud-detection',
    version_aliases=['baseline', 'experiment-control']
)

model_feature_eng = upload_model(
    display_name='fraud-detection',
    version_aliases=['experiment-feature-engineering']
)

model_hyperopt = upload_model(
    display_name='fraud-detection',
    version_aliases=['experiment-hyperparameter-tuning']
)

# Compare all experiments, promote winner
```

### 2.3 Promotion Workflows (dev → staging → prod)

**Manual Promotion:**

```python
# Development to staging
dev_version = aiplatform.Model('projects/123/locations/us-central1/models/456@3')
dev_version.add_version_alias('staging')
dev_version.remove_version_alias('dev')

# Staging to production (after validation)
staging_version = aiplatform.Model('projects/123/locations/us-central1/models/456@3')
staging_version.add_version_alias('champion')
staging_version.remove_version_alias('staging')

# Deprecate old production version
old_prod = aiplatform.Model('projects/123/locations/us-central1/models/456@2')
old_prod.remove_version_alias('champion')
old_prod.add_version_alias('deprecated')
```

**Automated CI/CD Promotion:**

```python
# Cloud Build pipeline with approval gates
def promote_to_production(model_resource_name, validation_metrics):
    model = aiplatform.Model(model_resource_name)

    # Validation checks
    if validation_metrics['accuracy'] < 0.95:
        raise ValueError("Model accuracy below threshold")

    if validation_metrics['data_drift_score'] > 0.3:
        raise ValueError("Significant data drift detected")

    # Automated promotion
    model.add_version_alias('champion')

    # Notify team
    send_notification(f"Model {model.version_id} promoted to production")

    return model
```

**Rollback Strategy:**

```python
# Quick rollback to previous version
current_prod = aiplatform.Model.list(
    filter='labels.version_alias:champion',
    order_by='create_time desc'
)[0]

# Rollback: remove champion from current, add to previous
current_prod.remove_version_alias('champion')

previous_version_id = int(current_prod.version_id) - 1
previous_model = aiplatform.Model(
    f'projects/123/locations/us-central1/models/456@{previous_version_id}'
)
previous_model.add_version_alias('champion')
```

From [Optimizing MLOps on Vertex AI](https://promevo.com/blog/optimizing-mlops-on-vertex-ai) (accessed 2025-01-13):
> "Assigning stages to model versions: Categorize model versions into stages (e.g., dev, shadow, prod) to manage their lifecycle and promote them based on performance."

---

## Section 3: Deployment from Registry (~90 lines)

### 3.1 Deploy to Vertex AI Endpoints

**Single Model Deployment:**

```python
from google.cloud import aiplatform

# Deploy model version by alias
model = aiplatform.Model('projects/123/locations/us-central1/models/fraud-detection')
endpoint = model.deploy(
    deployed_model_display_name='fraud-detection-champion',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,
    accelerator_type=None,  # Use 'NVIDIA_TESLA_T4' for GPU
    accelerator_count=0,
    version_aliases=['champion']  # Deploy specific alias
)

print(f"Deployed to endpoint: {endpoint.resource_name}")
```

**Deploy Specific Version:**

```python
# Deploy version 5 explicitly (bypassing aliases)
model_v5 = aiplatform.Model('projects/123/locations/us-central1/models/456@5')
endpoint = model_v5.deploy(
    endpoint=existing_endpoint,  # Add to existing endpoint
    deployed_model_display_name='fraud-v5-shadow',
    traffic_percentage=0,  # Shadow mode (no production traffic)
    machine_type='n1-standard-4'
)
```

### 3.2 A/B Testing with Traffic Splitting

**Two-Version A/B Test:**

```python
endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/789')

# 90% to champion, 10% to challenger
endpoint.traffic_split = {
    'fraud-detection-champion': 90,  # Deployed model ID
    'fraud-detection-challenger': 10
}
endpoint.update()
```

**Multi-Armed Bandit Deployment:**

```python
# Three concurrent experiments
endpoint.traffic_split = {
    'baseline-model': 60,      # Current champion
    'feature-eng-model': 20,   # Experiment 1
    'deep-learning-model': 20  # Experiment 2
}
endpoint.update()

# Monitor metrics, gradually shift traffic to winner
# After 7 days, winning model gets 100%
```

**Canary Deployment Pattern:**

```python
# Start with 5% traffic
endpoint.deploy_model(
    model='projects/123/locations/us-central1/models/456@6',
    traffic_percentage=5,
    deployed_model_display_name='fraud-v6-canary'
)

# Monitor for 24 hours, then increase to 25%
endpoint.update_deployed_model(
    deployed_model_id='fraud-v6-canary',
    traffic_percentage=25
)

# After validation, promote to 100%
endpoint.traffic_split = {'fraud-v6-canary': 100}
endpoint.undeploy_all()  # Remove old versions
```

### 3.3 Rollback Strategies

**Immediate Rollback (Zero-Downtime):**

```python
# Incident detected! Rollback from v6 to v5 instantly
endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/789')

# Shift all traffic back to previous version
endpoint.traffic_split = {
    'fraud-detection-v5': 100,  # Previous champion
    'fraud-detection-v6': 0     # Faulty version
}
endpoint.update()  # Takes effect in seconds

# Optional: Undeploy faulty version entirely
endpoint.undeploy(deployed_model_id='fraud-detection-v6')
```

**Gradual Rollback (Staged):**

```python
# Rollback in stages to monitor impact
traffic_stages = [
    {'v6': 80, 'v5': 20},  # Start rollback
    {'v6': 50, 'v5': 50},  # Equal split
    {'v6': 20, 'v5': 80},  # Mostly v5
    {'v6': 0, 'v5': 100}   # Complete rollback
]

for stage in traffic_stages:
    endpoint.traffic_split = stage
    endpoint.update()
    time.sleep(600)  # 10 minutes between stages

    # Check metrics before proceeding
    if check_error_rate() > threshold:
        # Immediate full rollback
        endpoint.traffic_split = {'v5': 100}
        break
```

**Blue-Green Deployment (Two Endpoints):**

```python
# Maintain two endpoints for instant switchover
blue_endpoint = aiplatform.Endpoint('projects/123/.../endpoints/blue')
green_endpoint = aiplatform.Endpoint('projects/123/.../endpoints/green')

# Deploy new version to green (currently inactive)
model_v6.deploy(endpoint=green_endpoint)

# Validate green endpoint
test_predictions(green_endpoint)

# Switch DNS/load balancer to green endpoint (external to Vertex)
# If issues detected, instant switch back to blue

# After validation period, green becomes blue
```

From [Google Vertex AI Model Registry and Versioning](https://medium.com/google-cloud/google-vertex-ai-model-versioning-72696bccd0d2) (accessed 2025-01-13):
> "The pre-build Vertex AI Components that take care of uploading the model do not support Model versioning yet" - highlighting the need for manual versioning workflows in pipelines.

---

## Section 4: Comparison & Integration (~60 lines)

### 4.1 Vertex AI Registry vs W&B Model Registry

| Feature | Vertex AI Model Registry | W&B Model Registry |
|---------|-------------------------|-------------------|
| **Versioning** | Automatic numeric (1, 2, 3) | Semantic versioning (v1.2.3) |
| **Deployment** | Direct to Vertex endpoints | External deployment tools |
| **Cloud Integration** | Native GCP (BigQuery, GCS, Vertex) | Cloud-agnostic |
| **Pricing** | Included in Vertex AI | Separate W&B subscription |
| **Experiment Tracking** | Vertex AI Experiments | Native W&B tracking |
| **Model Monitoring** | Vertex AI Model Monitoring | W&B Monitoring |
| **Metadata** | GCP-centric (training jobs, datasets) | Framework-agnostic |

**When to Use Vertex AI Model Registry:**
- All infrastructure on Google Cloud
- Need direct deployment to Vertex endpoints
- Leveraging BigQuery ML or AutoML
- Require GCP-native governance and compliance
- Cost optimization through unified billing

**When to Use W&B Model Registry:**
- Multi-cloud or hybrid deployments
- Existing investment in W&B ecosystem
- Framework-agnostic tracking (PyTorch, TensorFlow, JAX)
- Need portable model artifacts across clouds
- Advanced experiment comparison and visualization

### 4.2 Integration Patterns (W&B artifacts → Vertex Registry)

**Hybrid Workflow: W&B for Experiments, Vertex for Production:**

```python
import wandb
from google.cloud import aiplatform

# 1. Track experiments in W&B
wandb.init(project='fraud-detection')
# ... train model ...
wandb.log({'accuracy': 0.96, 'f1': 0.94})

# 2. Log model artifact to W&B
wandb.log_model(path='./model', name='fraud-xgboost')

# 3. Promote to Vertex Registry for production
run = wandb.Api().run(f'{wandb.config.entity}/{wandb.config.project}/{run_id}')
model_artifact = run.use_artifact('fraud-xgboost:latest')

# Download W&B artifact
artifact_dir = model_artifact.download()

# Upload to GCS
upload_to_gcs(artifact_dir, 'gs://my-bucket/models/fraud_v7/')

# Register in Vertex
vertex_model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri='gs://my-bucket/models/fraud_v7/',
    serving_container_image_uri='...',
    labels={
        'wandb_run_id': run_id,
        'wandb_artifact': model_artifact.name
    }
)
```

**Automated Sync Pipeline:**

```python
# Cloud Function triggered on W&B webhook
def sync_wandb_to_vertex(request):
    payload = request.get_json()

    # W&B notifies when model tagged 'production'
    if payload['tag'] == 'production':
        artifact_name = payload['artifact_name']

        # Download from W&B, upload to Vertex
        sync_artifact(artifact_name, vertex_model_name='fraud-detection')

        # Maintain lineage
        store_lineage({
            'wandb_artifact': artifact_name,
            'vertex_model': vertex_model_resource_name,
            'sync_time': datetime.now()
        })
```

### 4.3 Vertex AI Registry + MLflow Integration

```python
import mlflow
from google.cloud import aiplatform

# Register model in MLflow (local/remote tracking)
mlflow.sklearn.log_model(model, 'fraud-model')
mlflow.register_model('runs:/{run_id}/fraud-model', 'fraud-detection')

# Sync to Vertex Registry for GCP deployment
mlflow_model_uri = f'models:/fraud-detection/production'
local_path = mlflow.artifacts.download_artifacts(mlflow_model_uri)

# Upload to Vertex
vertex_model = aiplatform.Model.upload(
    display_name='fraud-detection',
    artifact_uri=f'gs://my-bucket/mlflow-models/{run_id}/',
    labels={'mlflow_model': 'fraud-detection', 'mlflow_stage': 'production'}
)
```

From [Building Effective Model Registry](https://www.projectpro.io/article/model-registry/874) (accessed 2025-01-13):
> "Having a registry makes it easier to: Keep track of model versions as they get retrained, Find and reuse the best-performing models, Promote models from experiments to production deployment, Govern and control access to model artifacts."

---

## ARR-COC Connection

### VLM Model Lifecycle Management

**ARR-COC Model Registry Strategy:**

```python
# Register ARR-COC model versions with relevance-specific metadata
arr_coc_model = aiplatform.Model.upload(
    display_name='arr-coc-vlm',
    artifact_uri='gs://arr-coc-models/v2.1_compressed/',
    description='ARR-COC VLM with 13-channel texture arrays and opponent processing',
    labels={
        'token_budget': '64-400',
        'propositional_scorer': 'shannon_entropy',
        'perspectival_scorer': 'jung_archetypal',
        'participatory_scorer': 'qwen3_cross_attention',
        'training_dataset': 'vqav2_filtered',
        'compression_ratio': '8x_avg'
    }
)

# Version-specific metadata for A/B testing relevance scorers
arr_coc_v2_1 = aiplatform.Model.upload(
    display_name='arr-coc-vlm',
    labels={
        'propositional_version': 'v2.1.0',
        'changes': 'optimized_entropy_calculation_5x_speedup'
    },
    version_aliases=['challenger-propositional']
)
```

**Deployment Strategy for Multi-Scorer Experiments:**

```python
# Deploy multiple ARR-COC variants for relevance scorer ablations
endpoint = aiplatform.Endpoint.create(display_name='arr-coc-inference')

# Baseline: All three scorers
baseline_model.deploy(endpoint, traffic_percentage=70, deployed_model_display_name='arr-coc-full')

# Ablation 1: Propositional + Perspectival only
ablation_1.deploy(endpoint, traffic_percentage=15, deployed_model_display_name='arr-coc-no-participatory')

# Ablation 2: Participatory only
ablation_2.deploy(endpoint, traffic_percentage=15, deployed_model_display_name='arr-coc-participatory-only')

# Monitor which relevance configuration produces best VQA results
```

**Rollback for Relevance Failures:**

If a new ARR-COC version produces poor relevance realization (token over-allocation, missed salient regions), instant rollback:

```python
# Detected: New version allocates 400 tokens to background, ignores foreground objects
# Rollback to previous version immediately
endpoint.traffic_split = {
    'arr-coc-v2.0': 100,  # Previous stable version
    'arr-coc-v2.1': 0      # Faulty relevance scorer
}
```

**Vertex AI Model Registry is essential for ARR-COC production deployment**: systematic version control ensures reproducibility of relevance realization experiments, while traffic splitting enables safe A/B testing of opponent processing variations without disrupting production inference.

---

## Sources

**Web Research:**
- [Optimizing MLOps on Vertex AI: Streamline Your ML Workflow with Google](https://promevo.com/blog/optimizing-mlops-on-vertex-ai) (accessed 2025-01-13)
- [Google Vertex AI Model Registry and Versioning](https://medium.com/google-cloud/google-vertex-ai-model-versioning-72696bccd0d2) (accessed 2025-01-13)
- [Building Effective Model Registry for your Machine Learning Projects](https://www.projectpro.io/article/model-registry/874) (accessed 2025-01-13)

**Official Documentation:**
- Google Cloud Vertex AI Model Registry documentation
- Vertex AI Python SDK reference

**Related Knowledge:**
- [30-vertex-ai-fundamentals.md](30-vertex-ai-fundamentals.md) - Base Vertex AI concepts
- [35-vertex-ai-production-patterns.md](35-vertex-ai-production-patterns.md) - High availability strategies
- [31-wandb-launch-vertex-agent.md](31-wandb-launch-vertex-agent.md) - W&B + Vertex integration
