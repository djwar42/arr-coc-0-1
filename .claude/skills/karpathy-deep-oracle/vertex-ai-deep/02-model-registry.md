# Vertex AI Model Registry

## Overview

Vertex AI Model Registry is a centralized repository for managing the complete lifecycle of machine learning models in production. It provides version control, lineage tracking, deployment management, and A/B testing capabilities for models across AutoML, custom training, and BigQuery ML.

From [Vertex AI Model Registry Introduction](https://docs.cloud.google.com/vertex-ai/docs/model-registry/introduction) (accessed 2025-02-03):
- Central repository for ML model lifecycle management
- Supports custom models, AutoML, and BigQuery ML models
- Provides model lineage, discovery, and lifecycle management
- Enables A/B testing and gradual rollouts

## Registry Overview

### What is Model Registry?

Model Registry serves as the central hub for:
- **Model versioning**: Track multiple versions with metadata
- **Lineage tracking**: Understand model creation, datasets, hyperparameters
- **Deployment management**: Control which versions are deployed where
- **Governance**: Maintain audit trails and compliance

From [Building Effective Model Registry](https://www.projectpro.io/article/model-registry/874) (accessed 2025-02-03):
- Organizes and manages all trained models
- Tracks versions, metadata, and approvals
- Promotes models across stages (dev → staging → production)
- Simplifies lifecycle management from training to retirement

### Architecture

Model Registry integrates with:
- **Vertex AI Training**: Automatic registration of trained models
- **Vertex AI Pipelines**: Pipeline-generated models auto-register
- **Vertex AI Endpoints**: Direct deployment from registry
- **Vertex ML Metadata**: Stores artifacts and lineage
- **BigQuery ML**: Import BigQuery-trained models

## Model Versioning

### Version Management

From [Model versioning with Model Registry](https://docs.cloud.google.com/vertex-ai/docs/model-registry/versioning) (accessed 2025-02-03):

**Import new model version**:
1. Navigate to Model Registry in Google Cloud Console
2. Select "Import" → "Import as new version"
3. Choose parent model for version lineage
4. Specify model artifact location
5. Add version-specific metadata

**Version naming conventions**:
```bash
# Format: model-name@version-number
my-classification-model@1
my-classification-model@2
my-classification-model@3

# Version aliases for clarity
my-classification-model@production  # Points to version 3
my-classification-model@staging     # Points to version 4
my-classification-model@latest      # Always newest version
```

### Version Metadata

Each version stores:
- **Model artifacts**: Saved model files, weights, architecture
- **Training metadata**: Hyperparameters, dataset references, training job ID
- **Evaluation metrics**: Accuracy, precision, recall, custom metrics
- **Timestamps**: Created, last modified, deployed dates
- **Labels**: Custom tags for organization (e.g., "experiment-42", "approved")

From [Vertex AI Best Practices 2025](https://skywork.ai/blog/vertex-ai-best-practices-governance-quotas-collaboration/) (accessed 2025-02-03):
- Use Model Registry for versions, lineage, and model cards
- Track versions and attach evaluation summaries
- Maintain lineage across datasets and training runs

### Version Comparison

Compare versions to determine:
- **Performance differences**: Which version has best metrics?
- **Resource usage**: Memory, latency, throughput differences
- **Training changes**: Different hyperparameters or datasets
- **Deployment impact**: Production vs. staging performance

```bash
# List model versions
gcloud ai models list \
  --region=us-central1 \
  --filter="displayName:my-model"

# Get version details
gcloud ai models describe MODEL_ID@VERSION \
  --region=us-central1
```

## Deployment Pipelines

### Pipeline Architecture

From [Best Practices for managing Vertex Pipelines code](https://cloud.google.com/blog/topics/developers-practitioners/best-practices-managing-vertex-pipelines-code/) (accessed 2025-02-03):

**Modular pipeline design**:
1. **Training component**: Trains model, outputs to registry
2. **Evaluation component**: Validates model performance
3. **Registration component**: Uploads model with metadata
4. **Deployment component**: Deploys to endpoint if metrics pass
5. **Monitoring component**: Sets up drift detection

**Pipeline best practices**:
- Structure pipelines into modular, reusable components
- Chain components with domain-specific language (DSL)
- Store pipeline metadata for lineage tracking
- Use Vertex Pipelines for serverless orchestration

From [Vertex AI pipeline best practices](https://eavelardev.github.io/gcp_courses/ml_in_the_enterprise/best_practices_for_ml_develo/vertex_ai_pipeline_best_practices.html) (accessed 2025-02-03):
- Pipelines automate training and deployment
- Components are modular and chainable
- Vertex AI Pipelines provide automation and orchestration
- Use Kubeflow Pipelines SDK or TensorFlow Extended (TFX)

### Automated Deployment Workflows

**Continuous deployment pattern**:
```yaml
# Kubeflow Pipeline example
- name: train-model
  container: gcr.io/project/trainer:latest
  outputs:
    - {name: model, type: Model}

- name: evaluate-model
  container: gcr.io/project/evaluator:latest
  inputs:
    - {name: model, type: Model}
  outputs:
    - {name: metrics, type: Metrics}

- name: deploy-if-better
  container: gcr.io/project/deployer:latest
  inputs:
    - {name: model, type: Model}
    - {name: metrics, type: Metrics}
  condition: metrics.accuracy > 0.95
```

### Deployment Strategies

From [Dual deployments on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/dual-deployments-vertex-ai) (accessed 2025-02-03):

**Deployment patterns**:
1. **Blue-Green deployment**: Deploy new version alongside old, switch traffic instantly
2. **Canary deployment**: Gradually increase traffic to new version (e.g., 5% → 50% → 100%)
3. **Shadow deployment**: Route traffic to new version without affecting responses
4. **Multi-model endpoint**: Deploy multiple models to same endpoint for A/B testing

**Traffic splitting example**:
```bash
# Deploy new version with 10% traffic
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID@2 \
  --traffic-split=0=90,1=10  # 90% to version 1, 10% to version 2
```

## A/B Testing

### A/B Testing Architecture

From [General guidance on conducting A/B experiments](https://docs.cloud.google.com/retail/docs/a-b-testing) (accessed 2025-02-03):

**A/B experiment setup**:
1. **Control group (A)**: Current production model
2. **Treatment group (B)**: New model version
3. **Traffic allocation**: Random assignment (e.g., 50/50 split)
4. **Metrics tracking**: Compare performance across groups
5. **Statistical significance**: Determine winner with confidence

**Key metrics to track**:
- **Model accuracy**: Precision, recall, F1 score
- **Business metrics**: Conversion rate, revenue, user engagement
- **Latency**: Response time differences
- **Resource usage**: CPU/memory/cost per prediction

### Implementing A/B Tests

From [Monitor and analyze A/B experiments](https://docs.cloud.google.com/retail/docs/a-b-monitor) (accessed 2025-02-03):

**Console setup**:
1. Input experiment name and time range
2. Define experiment arms (A, B, optional C)
3. Configure traffic split percentages
4. Set success metrics and thresholds
5. Monitor real-time results

**Traffic split strategies**:
- **50/50 split**: Equal comparison, fastest results
- **90/10 split**: Low-risk canary test
- **Multi-armed bandit**: Dynamic allocation to best performer

**Example deployment with A/B test**:
```bash
# Deploy two models to same endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID@1 \
  --display-name=baseline \
  --traffic-split=0=50,1=50

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID@2 \
  --display-name=experimental \
  --traffic-split=0=50,1=50
```

### A/B Testing Best Practices

From [A/B Testing for Machine Learning Models](https://medium.com/@deolesopan/a-b-testing-for-machine-learning-models-how-to-compare-models-with-confidence-4de49150a220) (accessed 2025-02-03):

**Statistical rigor**:
- Calculate required sample size upfront
- Use statistical tests (t-test, chi-square) for significance
- Account for multiple testing corrections
- Set minimum detectable effect (MDE) before experiment

**Experiment duration**:
- Run long enough for statistical power (typically 1-4 weeks)
- Account for day-of-week and seasonal effects
- Monitor daily to detect issues early
- Don't stop too early (avoid peeking problem)

**Guardrail metrics**:
- Monitor latency doesn't increase significantly
- Check for unexpected errors or failures
- Verify resource costs stay within budget
- Track user experience metrics (bounce rate, session length)

## Monitoring and Drift

### Drift Detection

From [Mastering Data Drift Detection with Google Vertex AI](https://blog.dataengineerthings.org/mastering-data-drift-detection-with-google-vertex-ai-a-step-by-step-guide-for-beginners-c45f624bffe3) (accessed 2025-02-03):

**Types of drift**:
1. **Data drift (feature drift)**: Input feature distributions change
2. **Concept drift**: Relationship between features and target changes
3. **Prediction drift**: Model output distribution changes

**Vertex AI Model Monitoring**:
- Automatic drift detection for deployed models
- Compares training data vs. production data
- Calculates statistical distance (KL divergence, L-infinity)
- Alerts when drift exceeds thresholds

**Setup monitoring**:
```bash
# Enable model monitoring
gcloud ai model-monitoring-jobs create \
  --region=us-central1 \
  --endpoint=ENDPOINT_ID \
  --display-name=drift-monitor \
  --monitoring-frequency=3600  # Check every hour \
  --feature-thresholds=feature1:0.05,feature2:0.10
```

### Monitoring Best Practices

From [Monitor feature skew and drift](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring) (accessed 2025-02-03):

**Skew detection**:
- Compares training data to production serving data
- Detects training-serving skew immediately
- Useful for catching data preprocessing bugs

**Drift detection**:
- Compares recent production data to baseline
- Tracks changes over time
- Indicates when retraining is needed

**Alert configuration**:
- Set different thresholds per feature based on importance
- Configure email/Pub/Sub notifications
- Create incident response runbooks
- Automate retraining triggers for severe drift

From [Best practices for model monitoring](https://eavelardev.github.io/gcp_courses/ml_in_the_enterprise/best_practices_for_ml_develo/best_practices_for_model_monitoring.html) (accessed 2025-02-03):

**Two monitoring approaches**:
1. **Skew detection**: Degree of distortion from training baseline
2. **Drift detection**: Changes in production data over time

**Best practices**:
- Monitor both features and predictions
- Set appropriate thresholds (not too sensitive)
- Review false positives regularly
- Combine automated alerts with manual reviews

### Model Performance Tracking

From [Monitoring feature attributions](https://cloud.google.com/blog/topics/developers-practitioners/monitoring-feature-attributions-how-google-saved-one-largest-ml-services-trouble) (accessed 2025-02-03):

**Feature attribution monitoring**:
- Track which features most influence predictions
- Detect when feature importance changes
- Identify subtle drift not caught by distribution monitoring
- Use Explainable AI with Model Monitoring

**Performance metrics**:
- Accuracy, precision, recall on holdout set
- Business KPIs (revenue, conversion rate)
- Latency percentiles (p50, p95, p99)
- Error rate and error types

## Production Best Practices

### Model Lineage

From [Introduction to Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-02-03):

**Track lineage to answer**:
- Which dataset was used for training?
- What hyperparameters produced this model?
- Which pipeline run generated this artifact?
- What upstream dependencies does this model have?

**Vertex ML Metadata stores**:
- Artifacts (models, datasets, metrics)
- Executions (training jobs, pipeline runs)
- Contexts (experiments, pipelines)
- Events (relationships between artifacts and executions)

### Governance and Compliance

**Audit trail requirements**:
- Who trained this model? (User, service account)
- When was it deployed? (Timestamp, pipeline ID)
- What data was used? (Dataset versions, preprocessing)
- Why was it promoted? (Evaluation metrics, approval)

**Model cards**:
- Document model purpose and limitations
- Include fairness and bias considerations
- Specify intended use cases
- List known failure modes

### Deployment Hygiene

**Pre-deployment checklist**:
- ✅ Model passes evaluation thresholds
- ✅ Latency tested under production load
- ✅ Resource limits configured (CPU, memory)
- ✅ Monitoring and alerts enabled
- ✅ Rollback plan documented
- ✅ Traffic split gradual (canary)

**Post-deployment monitoring**:
- Monitor for first 24 hours continuously
- Compare metrics to baseline model
- Watch for unexpected error types
- Check resource utilization

## Integration Patterns

### CI/CD for Models

From [Best practices for implementing machine learning on Google Cloud](https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices) (accessed 2025-02-03):

**Continuous training pipeline**:
1. Data validation → Schema check → Training → Evaluation → Registration → Deployment
2. Trigger on: Schedule, data arrival, performance drop
3. Automated quality gates at each step
4. Manual approval before production deployment

**Example GitLab/GitHub Actions workflow**:
```yaml
# .github/workflows/ml-deploy.yml
name: ML Model Deployment
on:
  push:
    branches: [main]
    paths: ['models/**', 'pipelines/**']

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python pipelines/train.py

      - name: Evaluate model
        run: python pipelines/evaluate.py

      - name: Register to Vertex AI
        if: metrics.accuracy > 0.95
        run: |
          gcloud ai models upload \
            --region=us-central1 \
            --artifact-uri=gs://bucket/model

      - name: Deploy with canary
        run: |
          gcloud ai endpoints deploy-model \
            --traffic-split=0=95,1=5
```

### Multi-Cloud and Hybrid

**Exporting models from registry**:
```bash
# Export model for deployment elsewhere
gcloud ai models export \
  --model=MODEL_ID \
  --output-uri=gs://bucket/exported-model \
  --export-format=tf-saved-model
```

**Supported formats**:
- TensorFlow SavedModel
- PyTorch (TorchScript)
- Scikit-learn (pickle)
- XGBoost
- Custom containers (any format)

## Command Reference

### Model Registration

```bash
# Upload custom model to registry
gcloud ai models upload \
  --region=us-central1 \
  --display-name=my-model \
  --artifact-uri=gs://bucket/model \
  --container-image-uri=gcr.io/project/serving-image

# Import existing model
gcloud ai models copy \
  --source-model=projects/PROJECT/locations/REGION/models/MODEL_ID \
  --destination-model=my-model-copy

# Add version to existing model
gcloud ai models upload \
  --parent-model=MODEL_ID \
  --version-aliases=production \
  --artifact-uri=gs://bucket/model-v2
```

### Deployment Management

```bash
# List models in registry
gcloud ai models list --region=us-central1

# Get model details
gcloud ai models describe MODEL_ID --region=us-central1

# Deploy to endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=deployment-name \
  --traffic-split=0=100

# Update traffic split (A/B test)
gcloud ai endpoints update ENDPOINT_ID \
  --region=us-central1 \
  --traffic-split=0=70,1=30

# Undeploy model version
gcloud ai endpoints undeploy-model ENDPOINT_ID \
  --region=us-central1 \
  --deployed-model-id=DEPLOYED_MODEL_ID
```

### Monitoring Configuration

```bash
# Create monitoring job
gcloud ai model-monitoring-jobs create \
  --region=us-central1 \
  --display-name=monitor-job \
  --endpoint=ENDPOINT_ID \
  --emails=alerts@example.com \
  --enable-monitoring

# List monitoring jobs
gcloud ai model-monitoring-jobs list \
  --region=us-central1 \
  --filter="state:JOB_STATE_RUNNING"

# Pause monitoring
gcloud ai model-monitoring-jobs pause MONITORING_JOB_ID \
  --region=us-central1
```

## Sources

**Google Cloud Documentation:**
- [Model versioning with Model Registry](https://docs.cloud.google.com/vertex-ai/docs/model-registry/versioning) (accessed 2025-02-03)
- [Introduction to Vertex AI Model Registry](https://docs.cloud.google.com/vertex-ai/docs/model-registry/introduction) (accessed 2025-02-03)
- [Monitor feature skew and drift](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring) (accessed 2025-02-03)
- [Introduction to Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction) (accessed 2025-02-03)
- [Best practices for implementing machine learning](https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices) (accessed 2025-02-03)

**Google Cloud Blog:**
- [Best Practices for managing Vertex Pipelines code](https://cloud.google.com/blog/topics/developers-practitioners/best-practices-managing-vertex-pipelines-code/) (accessed 2025-02-03)
- [Dual deployments on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/dual-deployments-vertex-ai) (accessed 2025-02-03)
- [Monitoring feature attributions](https://cloud.google.com/blog/topics/developers-practitioners/monitoring-feature-attributions-how-google-saved-one-largest-ml-services-trouble) (accessed 2025-02-03)

**Community Resources:**
- [Vertex AI pipeline best practices — Eduardo Avelar](https://eavelardev.github.io/gcp_courses/ml_in_the_enterprise/best_practices_for_ml_develo/vertex_ai_pipeline_best_practices.html) (accessed 2025-02-03)
- [Best practices for model monitoring — Eduardo Avelar](https://eavelardev.github.io/gcp_courses/ml_in_the_enterprise/best_practices_for_ml_develo/best_practices_for_model_monitoring.html) (accessed 2025-02-03)
- [Mastering Data Drift Detection with Google Vertex AI](https://blog.dataengineerthings.org/mastering-data-drift-detection-with-google-vertex-ai-a-step-by-step-guide-for-beginners-c45f624bffe3) (accessed 2025-02-03)
- [Building Effective Model Registry](https://www.projectpro.io/article/model-registry/874) (accessed 2025-02-03)
- [A/B Testing for Machine Learning Models](https://medium.com/@deolesopan/a-b-testing-for-machine-learning-models-how-to-compare-models-with-confidence-4de49150a220) (accessed 2025-02-03)
- [Vertex AI Best Practices 2025](https://skywork.ai/blog/vertex-ai-best-practices-governance-quotas-collaboration/) (accessed 2025-02-03)

**Retail/Commerce:**
- [General guidance on conducting A/B experiments](https://docs.cloud.google.com/retail/docs/a-b-testing) (accessed 2025-02-03)
- [Monitor and analyze A/B experiments](https://docs.cloud.google.com/retail/docs/a-b-monitor) (accessed 2025-02-03)
