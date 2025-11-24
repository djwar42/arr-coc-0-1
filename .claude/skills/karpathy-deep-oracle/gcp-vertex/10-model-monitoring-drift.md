# Vertex AI Model Monitoring & Drift Detection

**Knowledge File**: Comprehensive guide to Vertex AI Model Monitoring, drift detection algorithms, training-serving skew detection, Cloud Monitoring integration, alerting policies, and automated retraining triggers

---

## Overview

Machine learning models degrade over time in production due to data drift, concept drift, and distribution shifts. Vertex AI Model Monitoring provides automated detection of prediction drift, feature skew, and feature drift to maintain model quality in production deployments.

**Critical Production Challenge:**

Models can fail silently - unlike traditional software errors, ML model degradation happens gradually without visible crashes or exceptions. A model that achieved 95% accuracy during training may drop to 70% in production without triggering any alerts unless monitoring is configured.

From [Vertex AI Model Monitoring capabilities](https://cloud.google.com/blog/products/ai-machine-learning/get-to-know-vertex-ai-model-monitoring) (Google Cloud Blog, accessed 2025-11-16):
> "The new Vertex AI Model Monitoring aims to centralize the management of model monitoring capabilities and help customers continuously monitor their model performance in production."

From [karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md](../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md):
> "Models degrade without visible errors (silent failures). Ground truth labels arrive with delays (feedback lag). Data pipeline failures cascade to model quality."

---

## Section 1: Vertex AI Model Monitoring Architecture (~100 lines)

### 1.1 Monitoring Job Components

**ModelMonitoringJob Resource Structure:**

Vertex AI Model Monitoring V2 (latest version as of 2024) uses a centralized job architecture:

```python
from google.cloud import aiplatform

# Create monitoring job for deployed model
monitoring_job = aiplatform.ModelMonitoringJob.create(
    display_name="arr-coc-monitoring",
    endpoint=endpoint_id,
    model_monitoring_spec={
        # Feature drift monitoring
        "feature_drift_spec": {
            "categorical_thresholds": {"threshold_value": 0.3},
            "numerical_thresholds": {"threshold_value": 0.3}
        },
        # Prediction drift monitoring
        "prediction_drift_spec": {
            "drift_thresholds": {"threshold_value": 0.3}
        },
        # Training-serving skew detection
        "training_serving_skew_spec": {
            "skew_thresholds": {"threshold_value": 0.3},
            "training_dataset": "gs://bucket/training_data.csv"
        }
    },
    # Sampling configuration
    sampling_strategy={
        "random_sample_config": {
            "sample_rate": 0.1  # Sample 10% of traffic
        }
    },
    # Alert configuration
    alert_config={
        "email_alert_config": {
            "user_emails": ["ml-team@company.com"]
        },
        "enable_logging": True
    },
    # Monitoring schedule
    schedule={
        "cron": "0 */6 * * *"  # Every 6 hours
    }
)
```

**Key Configuration Parameters:**

- **Sampling rate** - Balance between monitoring cost and detection accuracy (typical: 0.05-0.20 or 5%-20%)
- **Thresholds** - Statistical significance levels for drift detection (typical: 0.3 for L-infinity distance)
- **Schedule** - Monitoring frequency (typical: hourly for high-traffic, daily for low-traffic models)
- **Training baseline** - Reference dataset statistics for skew detection

### 1.2 Supported Model Types

**Vertex AI Model Monitoring supports:**

1. **AutoML models** - Out-of-the-box monitoring for AutoML Vision, Tables, NLP
2. **Custom-trained models** - Models uploaded to Model Registry from any framework
3. **Pre-built containers** - TensorFlow, PyTorch, Scikit-learn serving containers
4. **Custom containers** - User-defined serving containers with standardized prediction format

**Input/Output Format Requirements:**

```python
# Prediction request format for monitoring
{
    "instances": [
        {
            "feature_1": 0.5,
            "feature_2": "category_a",
            "image_bytes": "base64_encoded_string"  # For vision models
        }
    ]
}

# Prediction response format
{
    "predictions": [
        {
            "score": 0.85,
            "class": "positive"
        }
    ]
}
```

Vertex AI automatically logs prediction inputs/outputs to Cloud Logging, which Model Monitoring analyzes for drift detection.

---

## Section 2: Drift Detection Algorithms (~150 lines)

### 2.1 Statistical Drift Detection Methods

**Vertex AI implements multiple drift detection algorithms:**

#### L-infinity Distance (Categorical Features)

Measures maximum difference in category distributions:

```
L_inf = max|P_production(x_i) - P_training(x_i)|
```

**Example:**
- Training: {"cat": 0.6, "dog": 0.3, "bird": 0.1}
- Production: {"cat": 0.4, "dog": 0.5, "bird": 0.1}
- L_inf = max(|0.6-0.4|, |0.3-0.5|, |0.1-0.1|) = 0.2

**Threshold interpretation:**
- L_inf < 0.1: No drift
- L_inf 0.1-0.3: Minor drift (monitor)
- L_inf > 0.3: Significant drift (alert triggered)

#### Jensen-Shannon Divergence (Categorical Features)

Symmetric measure of distribution similarity:

```
JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
where M = 0.5 * (P + Q)
```

**Properties:**
- Bounded: 0 ≤ JS ≤ 1
- Symmetric: JS(P||Q) = JS(Q||P)
- More sensitive to small distribution changes than L-infinity

#### Kolmogorov-Smirnov Test (Numerical Features)

Non-parametric test comparing cumulative distribution functions:

```
D = sup|F_production(x) - F_training(x)|
```

**Example use case:**
```python
# Vertex AI automatically computes K-S statistic
# User configures threshold for alerting
numerical_thresholds = {
    "feature_name": "pixel_intensity",
    "threshold_value": 0.3,  # Alert if D > 0.3
}
```

**Interpretation:**
- D < 0.1: Distributions are similar
- D 0.1-0.3: Moderate drift
- D > 0.3: Significant distributional shift

### 2.2 Chi-Squared Test for Categorical Drift

**Statistical hypothesis testing:**

```
H0: Production distribution = Training distribution
H1: Distributions differ significantly

Chi-squared statistic:
χ² = Σ((O_i - E_i)² / E_i)

where:
O_i = observed count in production
E_i = expected count from training
```

**Vertex AI Configuration:**

```python
feature_drift_spec = {
    "categorical_thresholds": {
        "threshold_value": 0.05  # p-value threshold
    }
}
```

If p-value < 0.05, reject H0 → drift detected.

### 2.3 Population Stability Index (PSI)

**Industry-standard drift metric:**

```
PSI = Σ((P_production - P_training) * ln(P_production / P_training))
```

**PSI Interpretation Guidelines:**
- PSI < 0.1: No significant change
- PSI 0.1-0.25: Moderate change (investigate)
- PSI > 0.25: Significant drift (retrain model)

**Example Calculation:**

| Bin | Training % | Production % | Difference | ln(Prod/Train) | PSI Component |
|-----|-----------|-------------|-----------|---------------|--------------|
| Low | 30% | 20% | -10% | -0.405 | 0.0405 |
| Med | 50% | 60% | +10% | +0.182 | 0.0182 |
| High | 20% | 20% | 0% | 0 | 0 |
| **Total PSI** | | | | | **0.0587** |

PSI = 0.0587 → No significant drift

**Vertex AI PSI Implementation:**

Vertex AI doesn't directly expose PSI as a configurable metric, but you can compute it from logged predictions:

```python
from google.cloud import bigquery
import numpy as np

# Query prediction logs
client = bigquery.Client()
query = """
SELECT feature_name, feature_value, COUNT(*) as count
FROM `project.dataset.prediction_logs`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY feature_name, feature_value
"""

# Compute PSI from training vs production distributions
def calculate_psi(training_dist, production_dist):
    psi = 0
    for category in training_dist.keys():
        p_train = training_dist.get(category, 0.001)  # Avoid log(0)
        p_prod = production_dist.get(category, 0.001)
        psi += (p_prod - p_train) * np.log(p_prod / p_train)
    return psi
```

### 2.4 KL Divergence (Kullback-Leibler)

**Measures information loss when using production distribution to approximate training:**

```
D_KL(P_train || P_prod) = Σ P_train(x) * log(P_train(x) / P_prod(x))
```

**Properties:**
- Non-symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
- Unbounded: 0 ≤ D_KL ≤ ∞
- Sensitive to small probability events

**Use case in Vertex AI:**

Monitor distribution shifts in high-dimensional embeddings or output probabilities:

```python
# For classification models, monitor prediction distribution drift
prediction_drift_spec = {
    "drift_thresholds": {
        "threshold_value": 0.5  # KL divergence threshold
    }
}
```

---

## Section 3: Training-Serving Skew Detection (~100 lines)

### 3.1 What is Training-Serving Skew?

**Definition:**

Training-serving skew occurs when there are differences between:
- Feature engineering code used during **training**
- Feature engineering code used during **inference** (serving)

From [Qwak - What is training-serving skew in Machine Learning?](https://www.qwak.com/post/training-serving-skew-in-machine-learning) (accessed 2025-11-16):
> "Training-serving skew, a problem that arises due to the most minor discrepancies in feature engineering code between training and deployment environments."

**Common Causes:**

1. **Code duplication** - Training notebook uses pandas, serving uses numpy
2. **Library version mismatch** - Scikit-learn 1.0 vs 1.2
3. **Data preprocessing differences** - Different normalization constants
4. **Feature calculation bugs** - Timezone handling in date features

**Example Skew Scenario:**

```python
# Training code (Python notebook)
def preprocess_age(age):
    return (age - 30) / 15  # Hardcoded mean=30, std=15

# Serving code (Go microservice)
func preprocessAge(age float64) float64 {
    return (age - 32) / 14  // Different constants!
}
```

Result: Model receives different feature distributions at serving time, causing silent performance degradation.

### 3.2 Vertex AI Skew Detection Configuration

**Enable training-serving skew monitoring:**

```python
# Upload training dataset statistics
training_data_uri = "gs://bucket/training_stats.csv"

# Configure skew detection
skew_spec = {
    "training_dataset": training_data_uri,
    "skew_thresholds": {
        "age": {"threshold_value": 0.3},
        "income": {"threshold_value": 0.3},
        "category": {"threshold_value": 0.2}
    },
    "attribution_score_skew_thresholds": {
        "threshold_value": 0.3
    }
}

monitoring_job = aiplatform.ModelMonitoringJob.create(
    display_name="skew-detection-job",
    endpoint=endpoint_id,
    model_monitoring_spec={
        "training_serving_skew_spec": skew_spec
    }
)
```

**Training Data Format:**

```csv
age,income,category,label
25,50000,A,0
32,75000,B,1
45,60000,A,0
```

Vertex AI computes statistical summary (mean, std, distribution) and compares against live serving data.

### 3.3 Skew Detection Metrics

**Vertex AI compares:**

1. **Feature distributions** - Same statistical tests as drift detection (K-S, Chi-squared)
2. **Feature attribution scores** - If using Explainable AI, compare feature importance between training and serving
3. **Prediction distributions** - Compare output distributions

**Skew vs Drift:**

| Aspect | Training-Serving Skew | Data Drift |
|--------|---------------------|-----------|
| **Comparison** | Training data vs serving data | Past serving data vs current serving data |
| **Cause** | Code/pipeline differences | Real-world distribution changes |
| **Fix** | Fix preprocessing code | Retrain model with new data |
| **Detection** | Immediate (at deployment) | Gradual (over time) |

### 3.4 Preventing Training-Serving Skew

**Best Practices:**

1. **Use TensorFlow Transform (tf.Transform):**

```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Preprocessing function applied consistently in training and serving."""
    outputs = {}
    # Normalization using statistics computed from training data
    outputs['age_normalized'] = tft.scale_to_z_score(inputs['age'])
    outputs['income_bucketized'] = tft.bucketize(inputs['income'], num_buckets=10)
    return outputs
```

tf.Transform ensures identical preprocessing by:
- Computing statistics during training (analyze phase)
- Embedding statistics in TensorFlow graph
- Applying same transformations at serving time

2. **Feature Store for consistency:**

```python
from google.cloud import aiplatform

# Write features to Feature Store during training
feature_store = aiplatform.Featurestore("my-featurestore")
entity_type = feature_store.get_entity_type("user")

# Serve features at inference time (identical code path)
features = entity_type.read(entity_ids=["user_123"])
```

3. **Shared preprocessing libraries:**

Package preprocessing code in a library used by both training and serving:

```python
# shared_preprocessing.py
def normalize_age(age, mean=30.5, std=14.2):
    """Normalize age using training statistics."""
    return (age - mean) / std

# Training code
from shared_preprocessing import normalize_age
X_train['age_norm'] = normalize_age(X_train['age'])

# Serving code (same library)
from shared_preprocessing import normalize_age
feature_vector['age_norm'] = normalize_age(request.age)
```

---

## Section 4: Cloud Monitoring Integration (~100 lines)

### 4.1 Cloud Monitoring Metrics

**Vertex AI Model Monitoring automatically exports metrics to Cloud Monitoring:**

**Available Metrics:**

```
# Feature drift metrics
aiplatform.googleapis.com/prediction/feature_drift/l_infinity
aiplatform.googleapis.com/prediction/feature_drift/jensen_shannon_divergence

# Prediction drift metrics
aiplatform.googleapis.com/prediction/prediction_drift/mean
aiplatform.googleapis.com/prediction/prediction_drift/quantiles

# Training-serving skew metrics
aiplatform.googleapis.com/prediction/skew/feature_skew
aiplatform.googleapis.com/prediction/skew/attribution_skew

# Model quality (if ground truth available)
aiplatform.googleapis.com/prediction/online_evaluation/accuracy
aiplatform.googleapis.com/prediction/online_evaluation/precision
aiplatform.googleapis.com/prediction/online_evaluation/recall
```

**Query metrics using Cloud Monitoring API:**

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Query feature drift over last 7 days
interval = monitoring_v3.TimeInterval({
    "end_time": {"seconds": int(time.time())},
    "start_time": {"seconds": int(time.time()) - 7*24*3600}
})

results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="aiplatform.googleapis.com/prediction/feature_drift/l_infinity"',
        "interval": interval,
    }
)

for result in results:
    print(f"Feature: {result.metric.labels['feature_name']}")
    print(f"Drift score: {result.points[0].value.double_value}")
```

### 4.2 Custom Dashboards

**Create monitoring dashboard in Cloud Console:**

```yaml
# dashboard.yaml
displayName: "ARR-COC Model Monitoring"
dashboardFilters:
  - filterType: RESOURCE_LABEL
    labelKey: endpoint_id
    stringValue: "arr-coc-endpoint"

widgets:
  # Feature drift chart
  - title: "Feature Drift - L-infinity Distance"
    xyChart:
      dataSets:
        - timeSeriesQuery:
            timeSeriesFilter:
              filter: 'metric.type="aiplatform.googleapis.com/prediction/feature_drift/l_infinity"'
              aggregation:
                alignmentPeriod: 3600s
                perSeriesAligner: ALIGN_MEAN
          plotType: LINE

  # Prediction volume
  - title: "Prediction Request Rate"
    xyChart:
      dataSets:
        - timeSeriesQuery:
            timeSeriesFilter:
              filter: 'metric.type="aiplatform.googleapis.com/prediction/prediction_count"'
              aggregation:
                alignmentPeriod: 60s
                perSeriesAligner: ALIGN_RATE

  # Error rate
  - title: "Prediction Error Rate"
    scorecard:
      timeSeriesQuery:
        timeSeriesFilter:
          filter: 'metric.type="aiplatform.googleapis.com/prediction/error_count"'
          aggregation:
            alignmentPeriod: 300s
            perSeriesAligner: ALIGN_RATE
```

**Deploy dashboard:**

```bash
gcloud monitoring dashboards create --config-from-file=dashboard.yaml
```

### 4.3 Metrics-Based SLIs and SLOs

**Define Service Level Indicators (SLIs) for model quality:**

```python
# SLI: 95% of predictions should have drift score < 0.3
sli_config = {
    "service": "vertex-ai-model",
    "sli": {
        "type": "request_based",
        "request_based_sli": {
            "good_total_ratio": {
                "good_service_filter": 'metric.type="aiplatform.googleapis.com/prediction/feature_drift/l_infinity" AND metric.drift_score < 0.3',
                "total_service_filter": 'metric.type="aiplatform.googleapis.com/prediction/feature_drift/l_infinity"'
            }
        }
    }
}

# SLO: 99.5% of predictions meet drift SLI over 30-day window
slo_config = {
    "goal": 0.995,
    "rolling_period": {"seconds": 30 * 24 * 3600}
}
```

**Monitor SLO burn rate:**

Fast burn rate (SLO violated quickly) triggers immediate escalation, while slow burn allows investigation time.

---

## Section 5: Alerting Policies (~120 lines)

### 5.1 Email Alerting

**Configure email alerts in Model Monitoring job:**

```python
alert_config = {
    "email_alert_config": {
        "user_emails": [
            "ml-team@company.com",
            "on-call@company.com"
        ]
    },
    "enable_logging": True  # Log alerts to Cloud Logging
}

monitoring_job = aiplatform.ModelMonitoringJob.create(
    display_name="production-monitoring",
    endpoint=endpoint_id,
    alert_config=alert_config,
    # ... other config
)
```

**Alert Email Format:**

```
Subject: [Vertex AI] Feature Drift Detected - arr-coc-endpoint

Feature drift detected for endpoint arr-coc-endpoint.

Affected features:
- pixel_intensity: L-infinity = 0.42 (threshold: 0.3)
- aspect_ratio: L-infinity = 0.38 (threshold: 0.3)

Time detected: 2025-11-16T14:30:00Z
Monitoring job: projects/123/locations/us-west2/modelMonitoringJobs/456

View details: https://console.cloud.google.com/vertex-ai/monitoring/...
```

### 5.2 Cloud Pub/Sub Notifications

**Route alerts to Pub/Sub for custom workflows:**

```python
from google.cloud import pubsub_v1

# Create Pub/Sub topic for drift alerts
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, "model-drift-alerts")

# Configure Cloud Monitoring alert policy with Pub/Sub channel
alert_policy = {
    "display_name": "Feature Drift Alert",
    "conditions": [{
        "display_name": "Feature drift > 0.3",
        "condition_threshold": {
            "filter": 'metric.type="aiplatform.googleapis.com/prediction/feature_drift/l_infinity"',
            "comparison": "COMPARISON_GT",
            "threshold_value": 0.3,
            "duration": {"seconds": 300},  # 5-minute window
            "aggregations": [{
                "alignment_period": {"seconds": 60},
                "per_series_aligner": "ALIGN_MAX"
            }]
        }
    }],
    "notification_channels": [pubsub_channel_id],
    "alert_strategy": {
        "auto_close": {"seconds": 3600}  # Auto-close after 1 hour
    }
}
```

**Consume Pub/Sub messages for automated workflows:**

```python
from google.cloud import pubsub_v1
import json

def drift_alert_callback(message):
    """Handle drift alert from Pub/Sub."""
    alert_data = json.loads(message.data)

    print(f"Drift detected: {alert_data['metric']['labels']['feature_name']}")
    print(f"Drift score: {alert_data['value']}")

    # Trigger automated response
    if alert_data['value'] > 0.5:
        # Critical drift - trigger immediate retraining
        trigger_retraining_pipeline(model_id=alert_data['endpoint'])
    else:
        # Moderate drift - create investigation ticket
        create_jira_ticket(
            title=f"Model drift detected: {alert_data['endpoint']}",
            description=f"Investigate drift in feature {alert_data['metric']['labels']['feature_name']}"
        )

    message.ack()

# Subscribe to drift alerts
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, "drift-alert-subscription")
subscriber.subscribe(subscription_path, callback=drift_alert_callback)
```

### 5.3 Webhook Notifications

**Send alerts to Slack, PagerDuty, or custom endpoints:**

```python
# Cloud Function triggered by Pub/Sub drift alert
import requests
import json

def send_slack_alert(request):
    """Send drift alert to Slack webhook."""
    alert_data = request.get_json()

    slack_message = {
        "text": "🚨 Model Drift Alert",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Feature Drift Detected"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Endpoint:*\n{alert_data['endpoint']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Feature:*\n{alert_data['feature']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Drift Score:*\n{alert_data['drift_score']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:*\n{alert_data['threshold']}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Dashboard"},
                        "url": alert_data['dashboard_url']
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Trigger Retraining"},
                        "url": alert_data['retrain_url'],
                        "style": "danger"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(slack_message),
        headers={'Content-Type': 'application/json'}
    )

    return f"Slack notification sent: {response.status_code}"
```

### 5.4 Multi-Channel Alert Routing

**Route different severity alerts to different channels:**

```python
# Low severity: Email only
low_severity_policy = {
    "conditions": [{
        "condition_threshold": {
            "filter": 'metric.drift_score',
            "comparison": "COMPARISON_GT",
            "threshold_value": 0.2  # Minor drift
        }
    }],
    "notification_channels": [email_channel],
    "severity": "WARNING"
}

# Medium severity: Email + Slack
medium_severity_policy = {
    "conditions": [{
        "condition_threshold": {
            "threshold_value": 0.3  # Moderate drift
        }
    }],
    "notification_channels": [email_channel, slack_channel],
    "severity": "ERROR"
}

# High severity: Email + Slack + PagerDuty (on-call)
high_severity_policy = {
    "conditions": [{
        "condition_threshold": {
            "threshold_value": 0.5  # Critical drift
        }
    }],
    "notification_channels": [email_channel, slack_channel, pagerduty_channel],
    "severity": "CRITICAL",
    "alert_strategy": {
        "notification_rate_limit": {
            "period": {"seconds": 300}  # Max 1 alert per 5 minutes
        }
    }
}
```

---

## Section 6: Automated Retraining Pipeline Triggers (~150 lines)

### 6.1 Eventarc-Based Retraining Triggers

**Trigger Vertex AI Pipeline when drift detected:**

From [Automated retraining triggers drift detection 2024](https://enhancedmlops.com/automatic-model-retraining-when-and-how-to-do-it/) (accessed 2025-11-16):
> "Key indicators for triggering model retraining include declining performance metrics, detection of data or concept drift, significant business changes, and scheduled intervals."

**Architecture:**

```
Vertex AI Monitoring → Cloud Monitoring Alert → Pub/Sub Topic → Cloud Function → Vertex AI Pipeline
```

**Step 1: Create Pub/Sub trigger topic**

```bash
gcloud pubsub topics create drift-retraining-trigger
```

**Step 2: Create Cloud Function to trigger pipeline**

```python
# main.py
from google.cloud import aiplatform
import json
import base64

def trigger_retraining_pipeline(event, context):
    """Triggered by Pub/Sub drift alert. Starts Vertex AI retraining pipeline."""

    # Decode Pub/Sub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    alert_data = json.loads(pubsub_message)

    # Extract drift information
    drift_score = alert_data['incident']['observed_value']
    feature_name = alert_data['incident']['resource']['labels']['feature_name']
    endpoint_id = alert_data['incident']['resource']['labels']['endpoint_id']

    print(f"Drift detected: {feature_name} = {drift_score}")

    # Check if drift exceeds retraining threshold
    RETRAINING_THRESHOLD = 0.4
    if drift_score < RETRAINING_THRESHOLD:
        print(f"Drift {drift_score} below threshold {RETRAINING_THRESHOLD}. Skipping retraining.")
        return

    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}/staging"
    )

    # Create pipeline job with drift context
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"retraining-drift-{feature_name}",
        template_path="gs://bucket/retraining_pipeline.yaml",
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_runs",
        parameter_values={
            "model_name": "arr-coc-vision",
            "training_data_path": "gs://bucket/latest_training_data",
            "drift_feature": feature_name,
            "drift_score": drift_score,
            "previous_model_uri": get_current_model_uri(endpoint_id),
            # Automatically fetch fresh data from last 30 days
            "data_start_date": get_date_30_days_ago(),
            "data_end_date": get_current_date()
        },
        enable_caching=False  # Force fresh training
    )

    # Submit pipeline
    pipeline_job.submit()

    print(f"Retraining pipeline submitted: {pipeline_job.resource_name}")

    # Notify team
    send_slack_notification(
        f"🔄 Automated retraining triggered\n"
        f"Feature: {feature_name}\n"
        f"Drift: {drift_score}\n"
        f"Pipeline: {pipeline_job.resource_name}"
    )

def get_current_model_uri(endpoint_id):
    """Get currently deployed model URI for comparison."""
    endpoint = aiplatform.Endpoint(endpoint_id)
    deployed_models = endpoint.list_models()
    return deployed_models[0].model_version_id

def get_date_30_days_ago():
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

def get_current_date():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
```

**Step 3: Deploy Cloud Function**

```bash
gcloud functions deploy trigger-retraining \
    --runtime python39 \
    --trigger-topic drift-retraining-trigger \
    --entry-point trigger_retraining_pipeline \
    --timeout 540s \
    --memory 512MB \
    --set-env-vars PROJECT_ID=my-project,REGION=us-west2,BUCKET_NAME=my-bucket
```

### 6.2 Retraining Pipeline Definition

**Vertex AI Pipeline for automated retraining:**

```python
# retraining_pipeline.py
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="gcr.io/deeplearning-platform-release/pytorch-gpu",
    packages_to_install=["google-cloud-aiplatform", "wandb"]
)
def fetch_training_data(
    data_start_date: str,
    data_end_date: str,
    output_dataset: Output[Dataset]
):
    """Fetch latest training data from BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client()
    query = f"""
    SELECT * FROM `project.dataset.predictions_table`
    WHERE timestamp BETWEEN '{data_start_date}' AND '{data_end_date}'
    AND label IS NOT NULL  # Only use labeled data
    """

    df = client.query(query).to_dataframe()
    df.to_csv(output_dataset.path, index=False)
    print(f"Fetched {len(df)} training examples")

@component(base_image="gcr.io/deeplearning-platform-release/pytorch-gpu")
def train_model(
    training_data: Input[Dataset],
    previous_model_uri: str,
    drift_feature: str,
    output_model: Output[Model],
    metrics: Output[Metrics]
):
    """Retrain model with emphasis on drifted feature."""
    import torch
    import pandas as pd
    from arr_coc.model import ARRCOCVisionModel

    # Load training data
    df = pd.read_csv(training_data.path)

    # Load previous model for warm start
    model = ARRCOCVisionModel.from_pretrained(previous_model_uri)

    # Apply data augmentation to drift-affected feature
    if drift_feature in df.columns:
        print(f"Applying augmentation to drifted feature: {drift_feature}")
        df = augment_drifted_feature(df, drift_feature)

    # Retrain
    model.train()
    trainer = create_trainer(model, df)
    trainer.fit()

    # Save model
    model.save_pretrained(output_model.path)

    # Log metrics
    metrics.log_metric("accuracy", trainer.evaluate()['accuracy'])
    metrics.log_metric("drift_feature", drift_feature)

@component
def evaluate_model(
    model: Input[Model],
    validation_data: Input[Dataset],
    previous_model_uri: str,
    metrics: Output[Metrics]
) -> str:
    """Compare new model against previous model."""
    # Load both models
    new_model = load_model(model.path)
    old_model = load_model(previous_model_uri)

    # Evaluate on validation set
    val_df = pd.read_csv(validation_data.path)

    new_accuracy = evaluate(new_model, val_df)
    old_accuracy = evaluate(old_model, val_df)

    metrics.log_metric("new_model_accuracy", new_accuracy)
    metrics.log_metric("old_model_accuracy", old_accuracy)
    metrics.log_metric("improvement", new_accuracy - old_accuracy)

    # Decide whether to deploy
    if new_accuracy > old_accuracy:
        return "DEPLOY"
    else:
        return "SKIP"

@component
def deploy_model(
    model: Input[Model],
    endpoint_id: str
):
    """Deploy new model to endpoint with traffic split."""
    from google.cloud import aiplatform

    # Upload model to Model Registry
    uploaded_model = aiplatform.Model.upload(
        display_name="arr-coc-retrained",
        artifact_uri=model.path,
        serving_container_image_uri="gcr.io/project/arr-coc-serving"
    )

    # Deploy with 10% traffic (canary)
    endpoint = aiplatform.Endpoint(endpoint_id)
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name="retrained-canary",
        traffic_percentage=10,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1
    )

    print(f"Model deployed with 10% traffic for A/B testing")

@dsl.pipeline(
    name="automated-retraining-pipeline",
    description="Triggered by drift detection to retrain and deploy model"
)
def retraining_pipeline(
    model_name: str,
    training_data_path: str,
    drift_feature: str,
    drift_score: float,
    previous_model_uri: str,
    data_start_date: str,
    data_end_date: str,
    endpoint_id: str
):
    # Fetch fresh training data
    fetch_task = fetch_training_data(
        data_start_date=data_start_date,
        data_end_date=data_end_date
    )

    # Train model
    train_task = train_model(
        training_data=fetch_task.outputs['output_dataset'],
        previous_model_uri=previous_model_uri,
        drift_feature=drift_feature
    )

    # Evaluate new vs old model
    eval_task = evaluate_model(
        model=train_task.outputs['output_model'],
        validation_data=fetch_task.outputs['output_dataset'],
        previous_model_uri=previous_model_uri
    )

    # Conditional deployment
    with dsl.Condition(eval_task.output == "DEPLOY"):
        deploy_task = deploy_model(
            model=train_task.outputs['output_model'],
            endpoint_id=endpoint_id
        )
```

**Compile and upload pipeline:**

```bash
python retraining_pipeline.py
gsutil cp retraining_pipeline.yaml gs://bucket/
```

### 6.3 Retraining Decision Logic

**Criteria for automated retraining:**

```python
def should_retrain(drift_metrics, performance_metrics):
    """Decision logic for automated retraining."""

    # Criterion 1: Significant drift detected
    drift_threshold_exceeded = any(
        metric['value'] > 0.4
        for metric in drift_metrics
    )

    # Criterion 2: Performance degradation (if ground truth available)
    performance_degraded = (
        performance_metrics.get('accuracy_drop', 0) > 0.05  # 5% drop
    )

    # Criterion 3: Multiple features drifting simultaneously
    num_drifted_features = sum(
        1 for metric in drift_metrics
        if metric['value'] > 0.3
    )
    multi_feature_drift = num_drifted_features >= 3

    # Criterion 4: Time since last retraining
    days_since_retrain = get_days_since_last_retrain()
    time_based_retrain = days_since_retrain > 30  # Max 30 days

    # Decision rules
    if performance_degraded:
        return True, "Performance degradation detected"

    if drift_threshold_exceeded and multi_feature_drift:
        return True, "Multiple features showing significant drift"

    if time_based_retrain and drift_threshold_exceeded:
        return True, "Scheduled retraining + drift detected"

    return False, "No retraining needed"
```

### 6.4 Retraining Safeguards

**Prevent runaway retraining loops:**

```python
# Cloud Function with safeguards
def trigger_retraining_with_safeguards(event, context):
    """Trigger retraining with safety checks."""

    # Safeguard 1: Rate limiting
    last_retrain_time = get_last_retrain_timestamp()
    MIN_HOURS_BETWEEN_RETRAINS = 6

    if (time.time() - last_retrain_time) < (MIN_HOURS_BETWEEN_RETRAINS * 3600):
        print(f"Skipping retrain: only {(time.time() - last_retrain_time)/3600:.1f} hours since last retrain")
        return

    # Safeguard 2: Maximum retrains per day
    retrains_today = count_retrains_last_24h()
    MAX_RETRAINS_PER_DAY = 4

    if retrains_today >= MAX_RETRAINS_PER_DAY:
        print(f"Skipping retrain: already {retrains_today} retrains today")
        send_alert_to_oncall("Excessive retraining attempts - investigate drift root cause")
        return

    # Safeguard 3: Cost estimate
    estimated_cost = estimate_retraining_cost()
    MAX_COST_PER_RETRAIN = 100  # $100

    if estimated_cost > MAX_COST_PER_RETRAIN:
        print(f"Skipping retrain: estimated cost ${estimated_cost} exceeds ${MAX_COST_PER_RETRAIN}")
        send_approval_request(f"Retraining requires ${estimated_cost} - approve?")
        return

    # All safeguards passed - proceed with retraining
    trigger_retraining_pipeline(event, context)
```

---

## Section 7: ARR-COC Visual Drift Monitoring (~100 lines)

### 7.1 Vision-Specific Drift Detection

**Monitoring visual input drift for ARR-COC-VIS model:**

```python
# Configure monitoring for vision model
vision_monitoring_spec = {
    "feature_drift_spec": {
        # Monitor image statistics
        "numerical_thresholds": {
            "mean_pixel_intensity": {"threshold_value": 0.3},
            "contrast_ratio": {"threshold_value": 0.3},
            "image_size_ratio": {"threshold_value": 0.2}
        },
        "categorical_thresholds": {
            "image_format": {"threshold_value": 0.1},  # PNG vs JPEG distribution
            "color_space": {"threshold_value": 0.1}     # RGB vs grayscale
        }
    },
    "prediction_drift_spec": {
        # Monitor output distribution drift
        "drift_thresholds": {
            # For classification: class distribution
            "class_distribution": {"threshold_value": 0.3},
            # For arr-coc: token allocation distribution
            "token_allocation_distribution": {"threshold_value": 0.3}
        }
    }
}
```

### 7.2 Custom Metrics for ARR-COC

**Log custom metrics specific to ARR-COC architecture:**

```python
# In arr-coc serving code
from google.cloud import monitoring_v3
import time

def log_arr_coc_metrics(request, response):
    """Log ARR-COC-specific metrics for drift monitoring."""

    client = monitoring_v3.MetricServiceClient()
    project_path = f"projects/{PROJECT_ID}"

    # Extract ARR-COC-specific features
    token_allocation = response['token_allocation']  # [64, 128, 256, 400] per patch
    relevance_scores = response['relevance_scores']
    num_patches = response['num_patches']

    # Custom metric: Average LOD per image
    avg_lod = sum(token_allocation) / num_patches

    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/arr_coc/average_lod"
    series.resource.type = "global"

    point = monitoring_v3.Point()
    point.value.double_value = avg_lod
    point.interval.end_time.seconds = int(time.time())

    series.points = [point]
    client.create_time_series(name=project_path, time_series=[series])

    # Custom metric: Relevance score distribution drift
    # Compare current relevance scores against training baseline
    relevance_drift = calculate_distribution_drift(
        relevance_scores,
        training_relevance_baseline
    )

    series2 = monitoring_v3.TimeSeries()
    series2.metric.type = "custom.googleapis.com/arr_coc/relevance_drift"
    series2.resource.type = "global"

    point2 = monitoring_v3.Point()
    point2.value.double_value = relevance_drift
    point2.interval.end_time.seconds = int(time.time())

    series2.points = [point2]
    client.create_time_series(name=project_path, time_series=[series2])
```

### 7.3 Visual Embedding Drift

**Monitor drift in visual embeddings produced by ARR-COC:**

```python
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingDriftDetector:
    """Detect drift in high-dimensional visual embeddings."""

    def __init__(self, training_embeddings):
        """Initialize with training set embeddings."""
        self.training_embeddings = training_embeddings

        # Compute PCA for dimensionality reduction
        self.pca = PCA(n_components=50)
        self.training_pca = self.pca.fit_transform(training_embeddings)

        # Compute training statistics
        self.training_mean = np.mean(self.training_pca, axis=0)
        self.training_cov = np.cov(self.training_pca.T)

    def compute_drift(self, production_embeddings):
        """Compute Mahalanobis distance as drift metric."""
        # Project to PCA space
        production_pca = self.pca.transform(production_embeddings)
        production_mean = np.mean(production_pca, axis=0)

        # Mahalanobis distance between distributions
        diff = production_mean - self.training_mean
        inv_cov = np.linalg.inv(self.training_cov)

        mahalanobis = np.sqrt(diff @ inv_cov @ diff.T)

        return mahalanobis

# Usage in monitoring
detector = EmbeddingDriftDetector(training_embeddings)

# Periodically compute drift on production embeddings
production_embeddings = collect_embeddings_from_logs(last_hour=True)
drift_score = detector.compute_drift(production_embeddings)

if drift_score > THRESHOLD:
    send_alert(f"Embedding drift detected: {drift_score:.3f}")
```

### 7.4 Query Distribution Drift

**Monitor changes in user query patterns:**

```python
from collections import Counter

class QueryDriftMonitor:
    """Monitor drift in text query distribution for arr-coc."""

    def __init__(self, training_queries):
        """Initialize with training query distribution."""
        # Extract query features
        self.training_features = self._extract_features(training_queries)

    def _extract_features(self, queries):
        """Extract features from queries."""
        features = {
            'avg_length': np.mean([len(q.split()) for q in queries]),
            'question_ratio': sum(1 for q in queries if '?' in q) / len(queries),
            'contains_object': sum(1 for q in queries if any(obj in q.lower() for obj in ['person', 'car', 'dog', 'cat'])) / len(queries),
            'contains_color': sum(1 for q in queries if any(color in q.lower() for color in ['red', 'blue', 'green'])) / len(queries),
            'avg_complexity': np.mean([len(q.split()) for q in queries])
        }
        return features

    def detect_drift(self, production_queries):
        """Detect drift in production queries."""
        prod_features = self._extract_features(production_queries)

        # Compute feature-wise drift
        drift_scores = {}
        for feature_name in self.training_features:
            train_val = self.training_features[feature_name]
            prod_val = prod_features[feature_name]

            # Normalized difference
            drift = abs(prod_val - train_val) / (train_val + 1e-6)
            drift_scores[feature_name] = drift

        return drift_scores

# Monitor query drift
monitor = QueryDriftMonitor(training_queries)
recent_queries = get_queries_from_logs(last_24_hours=True)
drift = monitor.detect_drift(recent_queries)

for feature, score in drift.items():
    if score > 0.3:
        log_metric(f"query_drift/{feature}", score)
```

**Example drift scenario for ARR-COC:**

**Training queries:**
- "Show me red cars"
- "Find people wearing hats"
- "Detect dogs in the image"

**Production queries (6 months later):**
- "Identify manufacturing defects"
- "Count inventory items"
- "Measure component dimensions"

→ **Query distribution drift detected** → User behavior shifted from general object detection to industrial QA → Model may need retraining on industrial images

---

## Summary

Vertex AI Model Monitoring provides production-grade drift detection for deployed ML models:

**Key Components:**
1. **Monitoring Jobs** - Automated drift detection with configurable sampling and thresholds
2. **Drift Algorithms** - L-infinity, Jensen-Shannon, KL divergence, Chi-squared, PSI
3. **Skew Detection** - Training-serving consistency validation
4. **Cloud Monitoring** - Centralized metrics, dashboards, and SLIs
5. **Alerting** - Multi-channel notifications (email, Pub/Sub, webhooks)
6. **Auto-Retraining** - Eventarc-triggered pipelines for automated model updates
7. **Vision-Specific** - Custom metrics for visual models like ARR-COC

**Best Practices:**
- Sample 5-20% of production traffic for cost-effective monitoring
- Set drift thresholds based on model type (0.2-0.4 for most cases)
- Use multiple drift metrics for comprehensive coverage
- Implement safeguards to prevent retraining loops
- Monitor domain-specific metrics (token allocation, relevance scores for ARR-COC)

**Integration with ARR-COC:**
- Monitor visual input statistics (pixel intensity, contrast, aspect ratios)
- Track token allocation distribution drift
- Detect relevance score distribution changes
- Monitor query pattern shifts
- Automated retraining triggers when visual domain shifts

---

## Sources

**Source Documents:**
- [karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md](../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md) - MLOps monitoring strategies and drift detection fundamentals

**Web Research:**
- [Vertex AI Model Monitoring capabilities](https://cloud.google.com/blog/products/ai-machine-learning/get-to-know-vertex-ai-model-monitoring) - Google Cloud Blog, June 11, 2024 (accessed 2025-11-16)
- [Monitor feature skew and drift | Vertex AI](https://cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring) - Google Cloud Documentation (accessed 2025-11-16)
- [Qwak - What is training-serving skew in Machine Learning?](https://www.qwak.com/post/training-serving-skew-in-machine-learning) - February 10, 2022 (accessed 2025-11-16)
- [Dealing with Train-serve Skew in Real-time ML Models](https://building.nubank.com/dealing-with-train-serve-skew-in-real-time-ml-models-a-short-guide/) - Nubank Engineering, June 27, 2023 (accessed 2025-11-16)
- [Machine learning model monitoring: Best practices](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/) - Datadog, April 26, 2024 (accessed 2025-11-16)
- [Automatic Model Retraining: When and How to Do It?](https://enhancedmlops.com/automatic-model-retraining-when-and-how-to-do-it/) - Enhanced MLOps, July 12, 2025 (accessed 2025-11-16)
- [Data Drift Is Not the Actual Problem: Your Monitoring Strategy Is](https://towardsdatascience.com/data-drift-is-not-the-actual-problem-your-monitoring-strategy-is/) - Towards Data Science, June 3, 2025 (accessed 2025-11-16)

**Additional References:**
- [Google Cloud Monitoring Documentation](https://cloud.google.com/monitoring/docs) - Cloud Monitoring API and metrics
- [Eventarc Documentation](https://cloud.google.com/eventarc/docs) - Event-driven triggers for automation
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines) - ML pipeline orchestration
