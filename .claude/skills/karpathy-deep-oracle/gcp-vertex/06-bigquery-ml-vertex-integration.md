# BigQuery ML + Vertex AI Integration: Complete Production Guide

**Comprehensive guide to BigQuery ML model training, export workflows, Vertex AI Model Registry integration, batch prediction at scale, federated queries, and cost optimization strategies**

This document covers the complete integration between BigQuery ML and Vertex AI, enabling SQL-based machine learning with enterprise-scale deployment, from CREATE MODEL statements through production serving and cost optimization.

---

## Section 1: CREATE MODEL in BigQuery (~100 lines)

### BigQuery ML Model Types

**Supported model types for SQL-based training:**

From [BigQuery ML Introduction](https://cloud.google.com/bigquery/docs/bqml-introduction) (accessed 2025-11-16):

**Classification & Regression:**
- Linear Regression: `model_type='LINEAR_REG'`
- Logistic Regression: `model_type='LOGISTIC_REG'`
- Boosted Tree (XGBoost): `model_type='BOOSTED_TREE_CLASSIFIER'` or `'BOOSTED_TREE_REGRESSOR'`
- Deep Neural Networks: `model_type='DNN_CLASSIFIER'` or `'DNN_REGRESSOR'`
- AutoML Tables: `model_type='AUTOML_CLASSIFIER'` or `'AUTOML_REGRESSOR'`

**Advanced Models:**
- K-Means Clustering: `model_type='KMEANS'`
- Matrix Factorization: `model_type='MATRIX_FACTORIZATION'`
- Time Series (ARIMA): `model_type='ARIMA_PLUS'`
- Random Forest: `model_type='RANDOM_FOREST_CLASSIFIER'`

### CREATE MODEL Syntax

**Basic XGBoost classification example:**
```sql
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  auto_class_weights=TRUE,
  max_iterations=50,
  learn_rate=0.1,
  min_tree_child_weight=10,
  subsample=0.85,
  max_tree_depth=10,
  tree_method='hist',
  early_stop=TRUE,
  min_rel_progress=0.01
) AS
SELECT
  user_id,
  total_purchases,
  avg_purchase_amount,
  days_since_last_purchase,
  category_diversity,
  churned
FROM `project.dataset.training_data`;
```

**Deep Neural Network with TRANSFORM clause:**
```sql
CREATE OR REPLACE MODEL `project.dataset.dnn_model`
TRANSFORM(
  -- Feature preprocessing (applied at training AND prediction)
  ML.STANDARD_SCALER(age) OVER() AS age_normalized,
  ML.MIN_MAX_SCALER(income) OVER() AS income_scaled,
  ML.ONE_HOT_ENCODER(country) OVER() AS country_encoded,
  ML.BUCKETIZE(price, [0, 10, 50, 100, 500]) AS price_bucket,
  label
)
OPTIONS(
  model_type='DNN_CLASSIFIER',
  hidden_units=[128, 64, 32],
  dropout=0.2,
  batch_size=64,
  max_iterations=100,
  learn_rate=0.001,
  optimizer='adam',
  activation_fn='relu',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.features`;
```

### AutoML Integration

**Automatic hyperparameter tuning:**
```sql
CREATE OR REPLACE MODEL `project.dataset.automl_model`
OPTIONS(
  model_type='AUTOML_CLASSIFIER',
  input_label_cols=['target'],
  budget_hours=1.0,  -- Training time budget
  optimization_objective='MAXIMIZE_AU_ROC'
) AS
SELECT * FROM `project.dataset.training_data`;
```

From [OWOX BI BigQuery ML Guide](https://www.owox.com/blog/articles/bigquery-ml) (May 2025):
- AutoML automatically selects best model architecture
- Performs feature engineering and hyperparameter optimization
- Budget controls training time (1-72 hours)
- Outputs model ensemble for better accuracy

### Model Training Best Practices

**1. Data splitting:**
```sql
-- Create training/validation/test split
CREATE OR REPLACE TABLE `project.dataset.data_split` AS
SELECT
  *,
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(user_id AS STRING))), 10) < 7 THEN 'train'
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(user_id AS STRING))), 10) < 9 THEN 'val'
    ELSE 'test'
  END AS split
FROM `project.dataset.raw_data`;

-- Train on split
CREATE OR REPLACE MODEL `project.dataset.model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  data_split_method='custom',
  data_split_col='split'
) AS
SELECT * EXCEPT(split)
FROM `project.dataset.data_split`;
```

**2. Hyperparameter tuning:**
```sql
-- Use num_trials for hyperparameter search
CREATE OR REPLACE MODEL `project.dataset.tuned_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  num_trials=20,  -- Try 20 different hyperparameter combinations
  max_parallel_trials=5,
  hparam_tuning_objectives=['roc_auc']
) AS
SELECT * FROM `project.dataset.training_data`;
```

**3. Model versioning:**
```sql
-- Version models with naming convention
CREATE OR REPLACE MODEL `project.dataset.churn_v1_20251116`
OPTIONS(...) AS SELECT * FROM training_data;

-- Register with Vertex AI Model Registry
CREATE OR REPLACE MODEL `project.dataset.production_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  model_registry='vertex_ai',  -- Auto-register to Vertex AI
  vertex_ai_model_id='churn-predictor',
  vertex_ai_model_version_aliases=['prod', 'v1']
) AS
SELECT * FROM training_data;
```

---

## Section 2: ML.PREDICT for Batch Inference at Scale (~100 lines)

### ML.PREDICT Syntax

**Basic prediction query:**
```sql
SELECT
  user_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability
FROM
  ML.PREDICT(
    MODEL `project.dataset.churn_model`,
    (SELECT * FROM `project.dataset.current_users`)
  );
```

From [BigQuery ML Inference Overview](https://docs.cloud.google.com/bigquery/docs/inference-overview) (accessed 2025-11-16):
- Built-in inference optimized for large-scale batch prediction
- Low latency for smaller batches (<10,000 rows)
- Automatically parallelized across BigQuery slots
- No data movement required (query data in place)

### Batch Prediction Performance Optimization

**1. Partition pruning for large tables:**
```sql
-- Only predict on recent data (reduces cost)
SELECT
  user_id,
  prediction_date,
  predicted_label,
  confidence
FROM
  ML.PREDICT(
    MODEL `project.dataset.model`,
    (
      SELECT *
      FROM `project.dataset.users_partitioned`
      WHERE DATE(updated_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    )
  );
```

**2. Materialized results for repeated access:**
```sql
-- Save predictions to table for downstream use
CREATE OR REPLACE TABLE `project.dataset.predictions_20251116` AS
SELECT
  *,
  CURRENT_TIMESTAMP() AS prediction_timestamp
FROM
  ML.PREDICT(
    MODEL `project.dataset.model`,
    (SELECT * FROM `project.dataset.input_data`)
  );
```

**3. Streaming predictions (real-time):**
```sql
-- For low-latency use cases, combine with streaming inserts
INSERT INTO `project.dataset.predictions`
SELECT
  user_id,
  predicted_value,
  CURRENT_TIMESTAMP() AS prediction_time
FROM
  ML.PREDICT(
    MODEL `project.dataset.model`,
    (
      SELECT *
      FROM `project.dataset.streaming_buffer`
      WHERE _PARTITIONTIME = CURRENT_DATE()
    )
  );
```

### Prediction Explainability

**Use ML.EXPLAIN_PREDICT for feature importance:**
```sql
SELECT
  user_id,
  predicted_churned,
  feature_attributions
FROM
  ML.EXPLAIN_PREDICT(
    MODEL `project.dataset.churn_model`,
    (SELECT * FROM `project.dataset.users` WHERE user_id = 12345),
    STRUCT(3 AS top_k_features)  -- Show top 3 important features
  );
```

**Output includes Shapley values:**
```json
{
  "feature_attributions": [
    {"feature": "days_since_last_purchase", "attribution": 0.23},
    {"feature": "total_purchases", "attribution": -0.15},
    {"feature": "avg_purchase_amount", "attribution": 0.08}
  ]
}
```

### Batch Prediction Cost Optimization

From [BigQuery ML Inference Engine](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-ml-inference-engine) (March 2023):

**Slot usage patterns:**
- ML.PREDICT uses analysis slots (not storage)
- Scales automatically with batch size
- Charged same as query processing ($5/TB on-demand)

**Cost reduction strategies:**
```sql
-- 1. Cluster tables for prediction filtering
CREATE OR REPLACE TABLE `project.dataset.users_clustered`
CLUSTER BY user_segment, country
AS SELECT * FROM users;

-- 2. Only predict on filtered subset
SELECT * FROM ML.PREDICT(
  MODEL `project.dataset.model`,
  (
    SELECT * FROM `project.dataset.users_clustered`
    WHERE user_segment = 'high_value'  -- Cluster key (minimal scan)
  )
);
```

**Monitoring prediction costs:**
```sql
-- Query INFORMATION_SCHEMA for ML.PREDICT costs
SELECT
  user_email,
  query,
  total_bytes_processed / POW(10, 12) AS tb_processed,
  total_slot_ms / 1000 / 60 AS slot_minutes,
  (total_bytes_processed / POW(10, 12)) * 5 AS estimated_cost_usd
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_USER
WHERE
  query LIKE '%ML.PREDICT%'
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY total_bytes_processed DESC;
```

---

## Section 3: EXPORT MODEL to GCS (TensorFlow SavedModel Format) (~80 lines)

### Export Model Syntax

**Export to Google Cloud Storage:**
```sql
EXPORT MODEL `project.dataset.churn_model`
OPTIONS(
  uri='gs://my-bucket/models/churn_model/20251116',
  format='TENSORFLOW_SAVEDMODEL'  -- TensorFlow 2.x SavedModel format
);
```

From [Export BigQuery ML Models](https://docs.cloud.google.com/bigquery/docs/exporting-models) (accessed 2025-11-16):
- Exports to TensorFlow SavedModel format (compatible with TensorFlow Serving, Vertex AI)
- ONNX format available for some models (`format='ONNX'`)
- XGBoost models export as Booster format

**Supported export formats by model type:**

| Model Type | TensorFlow SavedModel | ONNX | XGBoost Booster |
|------------|----------------------|------|-----------------|
| Linear Regression | ✓ | ✓ | - |
| Logistic Regression | ✓ | ✓ | - |
| DNN | ✓ | ✓ | - |
| Boosted Tree (XGBoost) | ✓ | - | ✓ |
| AutoML | ✓ | - | - |
| K-Means | ✓ | ✓ | - |

### Export with TRANSFORM Clause

**Models with preprocessing export preprocessing logic:**
```sql
-- Model with TRANSFORM clause
CREATE OR REPLACE MODEL `project.dataset.model_with_transform`
TRANSFORM(
  ML.STANDARD_SCALER(age) OVER() AS age_normalized,
  ML.ONE_HOT_ENCODER(country) OVER() AS country_encoded,
  label
)
OPTIONS(model_type='DNN_CLASSIFIER', input_label_cols=['label'])
AS SELECT * FROM training_data;

-- Export includes preprocessing in SavedModel
EXPORT MODEL `project.dataset.model_with_transform`
OPTIONS(uri='gs://bucket/model');
```

**Exported SavedModel structure:**
```
gs://bucket/model/
├── saved_model.pb          # Model graph
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
    └── transform_fn/       # TRANSFORM clause preprocessing
        ├── saved_model.pb
        └── variables/
```

### Verify Exported Model

**Check export job status:**
```sql
SELECT
  job_id,
  creation_time,
  end_time,
  state,
  error_result
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  job_type = 'EXPORT'
  AND destination_uris LIKE '%gs://my-bucket/models%'
ORDER BY creation_time DESC
LIMIT 10;
```

**Test exported model locally (Python):**
```python
import tensorflow as tf

# Load exported SavedModel
model = tf.saved_model.load('gs://my-bucket/models/churn_model/20251116')

# Inspect signatures
print(list(model.signatures.keys()))  # ['serving_default']

# Make prediction
serving_fn = model.signatures['serving_default']
predictions = serving_fn(
    age=tf.constant([35.0]),
    income=tf.constant([75000.0]),
    country=tf.constant(['US'])
)
print(predictions)
```

---

## Section 4: Import to Vertex AI Model Registry (~80 lines)

### Register BigQuery ML Model in Vertex AI

**Option 1: Automatic registration during CREATE MODEL:**
```sql
CREATE OR REPLACE MODEL `project.dataset.production_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  model_registry='vertex_ai',  -- Auto-register
  vertex_ai_model_id='churn-predictor',
  vertex_ai_model_version_aliases=['prod', 'latest']
) AS
SELECT * FROM training_data;
```

From [Manage BigQuery ML Models in Vertex AI](https://docs.cloud.google.com/bigquery/docs/managing-models-vertex) (accessed 2025-11-16):
- Register BigQuery ML models with Vertex AI Model Registry
- Manage alongside custom Vertex AI models
- No export required (models stay in BigQuery)
- Enables unified model governance

**Option 2: Register existing model:**
```sql
-- Register after model creation
ALTER MODEL `project.dataset.existing_model`
SET OPTIONS(
  model_registry='vertex_ai',
  vertex_ai_model_id='existing-model',
  vertex_ai_model_version_aliases=['v1', 'staging']
);
```

### Upload Exported Model to Vertex AI

**After exporting to GCS, upload to Model Registry:**

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Upload exported SavedModel to Vertex AI Model Registry
model = aiplatform.Model.upload(
    display_name='churn-predictor',
    artifact_uri='gs://my-bucket/models/churn_model/20251116',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest',
    model_id='churn-predictor-v1',
    version_aliases=['prod', 'v1'],
    version_description='XGBoost churn model trained in BigQuery ML'
)

print(f"Model uploaded: {model.resource_name}")
print(f"Model ID: {model.name}")
```

### Model Versioning in Vertex AI

**Create versioned model deployment:**
```python
# Upload new version of existing model
model_v2 = aiplatform.Model.upload(
    display_name='churn-predictor',  # Same display name
    artifact_uri='gs://my-bucket/models/churn_model/20251117',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest',
    parent_model='projects/123/locations/us-central1/models/456',  # Parent model ID
    version_aliases=['staging', 'v2'],
    is_default_version=False  # Don't make default yet
)

# Promote to production after validation
model_v2.update(version_aliases=['prod', 'v2'], is_default_version=True)
```

---

## Section 5: Vertex AI Batch Prediction from BigQuery Tables (~100 lines)

### Batch Prediction Job Configuration

**Submit batch prediction job (Python):**

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Create batch prediction job with BigQuery input/output
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name='churn-batch-prediction-20251116',
    model_name='projects/123/locations/us-central1/models/456',

    # BigQuery input
    bigquery_source='bq://my-project.dataset.users_to_score',

    # BigQuery output
    bigquery_destination_prefix='bq://my-project.dataset',

    # Configuration
    instances_format='bigquery',
    predictions_format='bigquery',

    # Machine configuration
    machine_type='n1-standard-4',
    accelerator_count=0,
    starting_replica_count=10,
    max_replica_count=50,

    # Batch size tuning
    batch_size=64,

    sync=False  # Async execution
)

print(f"Batch job created: {batch_prediction_job.resource_name}")
```

From [Vertex AI Batch Prediction from BigQuery](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions) (accessed 2025-11-16):
- Input/output directly from/to BigQuery tables
- Automatically parallelized across workers
- No data movement required (serverless)
- Supports up to millions of predictions

### BigQuery Input Schema

**Input table format:**
```sql
-- Input table schema (matches model training features)
CREATE OR REPLACE TABLE `project.dataset.users_to_score` AS
SELECT
  user_id,
  age,
  income,
  country,
  total_purchases,
  days_since_last_purchase
FROM `project.dataset.current_users`;
```

**Output predictions table:**
```sql
-- Vertex AI creates output table automatically
SELECT * FROM `project.dataset.predictions_2025_11_16_batch_prediction`
LIMIT 10;

-- Output schema:
-- user_id, age, income, country, ..., predicted_label, predicted_label_probs
```

### Monitor Batch Job Status

**Check job progress (Python):**
```python
# Get job status
job = aiplatform.BatchPredictionJob(
    batch_prediction_job_name='projects/123/locations/us-central1/batchPredictionJobs/789'
)

print(f"Job state: {job.state}")
print(f"Completed: {job.completion_stats.successful_count} rows")
print(f"Failed: {job.completion_stats.failed_count} rows")

# Wait for completion
job.wait()
print(f"Final state: {job.state}")
```

**Query job logs (SQL):**
```sql
-- Check Vertex AI batch prediction jobs
SELECT
  display_name,
  state,
  create_time,
  end_time,
  TIMESTAMP_DIFF(end_time, create_time, SECOND) AS duration_seconds,
  completion_stats.successful_count,
  completion_stats.failed_count
FROM `project.region-us-central1`.INFORMATION_SCHEMA.BATCH_PREDICTION_JOBS
WHERE create_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY create_time DESC;
```

### Batch Prediction Cost Optimization

**1. Right-size machine types:**
```python
# For CPU-only models (most BigQuery ML exports)
machine_type='n1-standard-4'  # 4 vCPUs, 15GB RAM

# For large batches
machine_type='n1-standard-16'  # 16 vCPUs, 60GB RAM

# Use Spot VMs for cost savings (60-91% cheaper)
# Set max_replica_count high to handle preemptions
```

**2. Optimize batch size:**
```python
# Smaller batches: Lower latency, higher throughput
batch_size=64

# Larger batches: Higher latency, lower cost per prediction
batch_size=512
```

**3. Cost monitoring:**
```sql
-- Estimate batch prediction cost
-- Pricing: ~$0.48/hour for n1-standard-4 (on-demand)
SELECT
  display_name,
  TIMESTAMP_DIFF(end_time, create_time, HOUR) AS runtime_hours,
  TIMESTAMP_DIFF(end_time, create_time, HOUR) * 0.48 AS estimated_cost_usd,
  completion_stats.successful_count AS predictions,
  (TIMESTAMP_DIFF(end_time, create_time, HOUR) * 0.48) / completion_stats.successful_count AS cost_per_prediction
FROM `project.region-us-central1`.INFORMATION_SCHEMA.BATCH_PREDICTION_JOBS
WHERE display_name LIKE 'churn-batch%';
```

---

## Section 6: Federated Queries (BigQuery → Cloud SQL/Sheets) (~80 lines)

### Cloud SQL Federated Queries

**Create external connection to Cloud SQL:**

```sql
-- Step 1: Create connection (one-time setup, use bq command-line)
-- bq mk --connection --location=us --project_id=my-project \
--     --connection_type=CLOUD_SQL \
--     --properties='{"instanceId":"my-project:us-central1:my-instance","database":"mydb","type":"POSTGRES"}' \
--     --connection_credential='{"username":"bq-user","password":"SECRET"}' \
--     my_connection

-- Step 2: Query Cloud SQL data from BigQuery
SELECT
  bq_users.user_id,
  bq_users.total_purchases,
  sql_orders.order_count,
  sql_orders.total_amount
FROM
  `project.dataset.bigquery_users` AS bq_users
JOIN
  EXTERNAL_QUERY(
    'my-project.us.my_connection',
    '''SELECT user_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
       FROM orders
       WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
       GROUP BY user_id'''
  ) AS sql_orders
ON bq_users.user_id = sql_orders.user_id;
```

From [Cloud SQL Federated Queries](https://docs.cloud.google.com/bigquery/docs/cloud-sql-federated-queries) (accessed 2025-11-16):
- Query Cloud SQL (PostgreSQL/MySQL) data from BigQuery
- No data movement required (query in place)
- Supports joins with BigQuery tables
- Performance: Network latency overhead for large queries

### Google Sheets Federated Queries

**Query Google Sheets as external data source:**

```sql
-- Create external table pointing to Google Sheets
CREATE OR REPLACE EXTERNAL TABLE `project.dataset.campaign_budget`
OPTIONS (
  format = 'GOOGLE_SHEETS',
  uris = ['https://docs.google.com/spreadsheets/d/SPREADSHEET_ID'],
  sheet_range = 'Sheet1!A1:D1000',
  skip_leading_rows = 1
);

-- Join with BigQuery data
SELECT
  campaigns.campaign_id,
  campaigns.impressions,
  campaigns.clicks,
  budget.allocated_budget,
  campaigns.spend,
  (budget.allocated_budget - campaigns.spend) AS remaining_budget
FROM
  `project.dataset.ad_campaigns` AS campaigns
JOIN
  `project.dataset.campaign_budget` AS budget
ON campaigns.campaign_id = budget.campaign_id;
```

### Federated Query Performance Optimization

**1. Push down filters to external source:**
```sql
-- Good: Filter pushdown reduces data transfer
SELECT *
FROM EXTERNAL_QUERY(
  'my-project.us.my_connection',
  '''SELECT * FROM large_table
     WHERE created_at >= '2025-11-01'  -- Filter at source
     AND user_id IN (SELECT user_id FROM active_users)'''
);

-- Bad: Full table scan, then filter in BigQuery
SELECT *
FROM EXTERNAL_QUERY(
  'my-project.us.my_connection',
  'SELECT * FROM large_table'
)
WHERE created_at >= '2025-11-01';
```

**2. Limit data transfer:**
```sql
-- Aggregate at source before joining
SELECT
  bq_table.region,
  SUM(bq_table.sales) AS bq_sales,
  SUM(sql_agg.cloud_sql_sales) AS sql_sales
FROM
  `project.dataset.bigquery_sales` AS bq_table
JOIN
  EXTERNAL_QUERY(
    'my-connection',
    '''SELECT region, SUM(amount) AS cloud_sql_sales
       FROM sales_table
       GROUP BY region'''  -- Aggregate before transfer
  ) AS sql_agg
ON bq_table.region = sql_agg.region
GROUP BY bq_table.region;
```

**3. Cache federated query results:**
```sql
-- Materialize federated query results for repeated access
CREATE OR REPLACE TABLE `project.dataset.cloud_sql_snapshot` AS
SELECT * FROM EXTERNAL_QUERY(
  'my-connection',
  'SELECT * FROM frequently_accessed_table'
);

-- Query snapshot instead of live Cloud SQL
SELECT * FROM `project.dataset.cloud_sql_snapshot`;
```

---

## Section 7: Cost Optimization (Slot Reservations vs On-Demand) (~120 lines)

### Pricing Models Comparison

From [BigQuery Pricing](https://cloud.google.com/bigquery/pricing) (accessed 2025-11-16):

**On-Demand Pricing:**
- **Cost**: $5 per TB of data processed (analysis)
- **Slots**: Up to 2,000 concurrent slots (shared across project)
- **Commitment**: None (pay-per-query)
- **Best for**: Variable workloads, development, ad-hoc analytics

**Capacity-Based Pricing (Slot Reservations):**
- **Cost**: Baseline commitments ($2,000/month for 100 slots, autoscaling available)
- **Slots**: Reserved capacity (100-3,000+ slots)
- **Commitment**: Monthly or yearly (yearly = 34% discount)
- **Best for**: Predictable workloads, >$10k/month spend, production ML

**BigQuery Editions (New Pricing Model - 2024):**
- **Standard**: $0.04/slot-hour (baseline 100 slots, max 1,600 slots per reservation)
- **Enterprise**: $0.06/slot-hour (baseline 100 slots, higher autoscaling limits)
- **Enterprise Plus**: $0.10/slot-hour (highest performance, data governance features)

### When to Use Slot Reservations

From [BigQuery CUDs Guide](https://www.economize.cloud/blog/bigquery-commited-use-discounts/) (July 2025):

**Break-even analysis:**
```
On-Demand Cost: $5 per TB processed
Slot Reservation: $2,000/month for 100 slots

Break-even: ~400 TB/month processed
If you process > 400 TB/month → Reservations are cheaper
If you process < 400 TB/month → On-demand is cheaper
```

**Decision matrix:**

| Monthly TB Processed | Recommendation | Estimated Savings |
|---------------------|----------------|------------------|
| < 200 TB | On-demand | N/A |
| 200-400 TB | Analyze workload patterns | 0-20% |
| 400-1,000 TB | Baseline reservation (100 slots) | 20-40% |
| 1,000-5,000 TB | Reservation with autoscaling | 40-60% |
| > 5,000 TB | Enterprise Edition | 50-70% |

### Creating Slot Reservations

**Purchase slot commitment:**
```bash
# Create baseline reservation (100 slots)
bq mk --reservation \
  --project_id=my-project \
  --location=us-central1 \
  --slots=100 \
  my_reservation

# Assign project to reservation
bq mk --reservation_assignment \
  --reservation_id=my-project:us-central1.my_reservation \
  --job_type=QUERY \
  --assignee_type=PROJECT \
  --assignee_id=my-project
```

**Autoscaling configuration:**
```bash
# Enable autoscaling (Standard Edition: max 1,600 slots)
bq update --reservation \
  --location=us-central1 \
  --autoscale_max_slots=500 \
  my_reservation
```

From [Transitioning to Capacity-Based Pricing](https://techblog.rtbhouse.com/transitioning-to-capacity-based-pricing-in-google-bigquery/) (March 2025):
- Baseline slots: Always available
- Autoscale slots: Burst capacity (charged per-second)
- Autoscaling prevents slot exhaustion (eliminates "exceeded rate limits" errors)
- Cost: Baseline + autoscale usage

### Cost Monitoring and Optimization

**1. Track slot usage:**
```sql
-- Query INFORMATION_SCHEMA for slot utilization
SELECT
  TIMESTAMP_TRUNC(creation_time, HOUR) AS hour,
  reservation_id,
  SUM(total_slot_ms) / 1000 / 60 / 60 AS slot_hours,
  COUNT(*) AS query_count,
  AVG(total_slot_ms / 1000 / (TIMESTAMP_DIFF(end_time, start_time, SECOND) + 0.001)) AS avg_slots
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND job_type = 'QUERY'
  AND reservation_id IS NOT NULL
GROUP BY hour, reservation_id
ORDER BY hour DESC;
```

**2. Identify cost optimization opportunities:**
```sql
-- Find expensive queries for optimization
SELECT
  user_email,
  query,
  total_bytes_processed / POW(10, 12) AS tb_processed,
  total_slot_ms / 1000 / 60 AS slot_minutes,
  CASE
    WHEN reservation_id IS NULL THEN
      (total_bytes_processed / POW(10, 12)) * 5  -- On-demand cost
    ELSE
      0  -- Reservation (already paid)
  END AS estimated_cost_usd,

  -- Optimization flags
  CASE
    WHEN total_bytes_processed > 100 * POW(10, 12) THEN 'Consider partitioning'
    WHEN total_slot_ms > 1000 * 1000 * 60 THEN 'High slot usage - optimize query'
    ELSE 'OK'
  END AS optimization_flag
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_USER
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  AND job_type = 'QUERY'
ORDER BY total_bytes_processed DESC
LIMIT 20;
```

**3. Partition and cluster for cost reduction:**
```sql
-- Before: Full table scan (expensive)
SELECT COUNT(*)
FROM `project.dataset.events`
WHERE DATE(event_timestamp) = '2025-11-16';

-- Scans entire table (e.g., 1 TB) → $5 on-demand


-- After: Partitioned + clustered table
CREATE OR REPLACE TABLE `project.dataset.events_optimized`
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, event_type
AS SELECT * FROM `project.dataset.events`;

SELECT COUNT(*)
FROM `project.dataset.events_optimized`
WHERE DATE(event_timestamp) = '2025-11-16';

-- Scans only 1 partition (e.g., 3 GB) → $0.015 on-demand (99.7% reduction)
```

**4. Cost allocation by team:**
```sql
-- Tag queries with labels for chargeback
CREATE OR REPLACE TABLE `project.dataset.labeled_query` AS
SELECT * FROM source_table;
-- Add label: team=data-science

-- Query cost by label
SELECT
  label.value AS team,
  SUM(total_bytes_processed) / POW(10, 12) AS tb_processed,
  SUM(total_bytes_processed) / POW(10, 12) * 5 AS cost_usd
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT,
  UNNEST(labels) AS label
WHERE
  label.key = 'team'
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY team
ORDER BY cost_usd DESC;
```

---

## Section 8: arr-coc-0-1 BigQuery ML Preprocessing Example (~100 lines)

### Visual Feature Preprocessing for arr-coc-0-1

**Use case: Train relevance scorer on visual feature embeddings**

**Step 1: Create BigQuery table with embeddings:**
```sql
-- Import Qwen-VL embeddings from arr-coc-0-1 into BigQuery
CREATE OR REPLACE TABLE `arr-coc-project.dataset.patch_embeddings` AS
SELECT
  image_id,
  patch_id,
  patch_position_x,
  patch_position_y,
  embedding_vector,  -- 1024-dim Qwen-VL embedding (ARRAY<FLOAT64>)
  query_text,
  relevance_score,  -- Ground truth from manual annotation
  eccentricity,  -- Spatial feature
  sobel_magnitude  -- Edge feature
FROM EXTERNAL_QUERY(
  'arr-coc-project.us-central1.cloud-sql-connection',
  '''SELECT * FROM qwen_embeddings
     WHERE dataset_split = 'train' '''
);
```

**Step 2: Feature engineering in BigQuery:**
```sql
-- Aggregate patch embeddings to image-level features
CREATE OR REPLACE TABLE `arr-coc-project.dataset.image_features` AS
SELECT
  image_id,
  query_text,

  -- Aggregate embeddings (mean pooling)
  ARRAY_AGG(embedding_vector) AS patch_embeddings,

  -- Statistical features
  AVG(relevance_score) AS avg_relevance,
  STDDEV(relevance_score) AS relevance_variance,
  COUNT(*) AS num_patches,

  -- Spatial distribution
  STDDEV(patch_position_x) AS spatial_spread_x,
  STDDEV(patch_position_y) AS spatial_spread_y,

  -- Edge density
  AVG(sobel_magnitude) AS avg_edge_density,

  -- Target (high-relevance patch count)
  COUNTIF(relevance_score > 0.7) AS high_relevance_patches
FROM `arr-coc-project.dataset.patch_embeddings`
GROUP BY image_id, query_text;
```

**Step 3: Train XGBoost model for patch selection:**
```sql
-- Train model to predict number of high-relevance patches
CREATE OR REPLACE MODEL `arr-coc-project.dataset.patch_selector_xgboost`
TRANSFORM(
  -- Feature scaling
  ML.STANDARD_SCALER(avg_relevance) OVER() AS avg_relevance_scaled,
  ML.STANDARD_SCALER(relevance_variance) OVER() AS relevance_variance_scaled,
  ML.MIN_MAX_SCALER(num_patches) OVER() AS num_patches_scaled,
  ML.STANDARD_SCALER(spatial_spread_x) OVER() AS spatial_spread_x_scaled,
  ML.STANDARD_SCALER(spatial_spread_y) OVER() AS spatial_spread_y_scaled,
  ML.STANDARD_SCALER(avg_edge_density) OVER() AS avg_edge_density_scaled,

  -- Query text embedding (TF-IDF)
  ML.NGRAMS(SPLIT(LOWER(query_text), ' '), [1, 2], ' ') AS query_ngrams,

  high_relevance_patches  -- Target
)
OPTIONS(
  model_type='BOOSTED_TREE_REGRESSOR',
  l1_reg=0.1,
  l2_reg=0.1,
  max_iterations=100,
  max_tree_depth=8,
  subsample=0.8,
  input_label_cols=['high_relevance_patches']
) AS
SELECT * FROM `arr-coc-project.dataset.image_features`;
```

**Step 4: Evaluate model:**
```sql
-- Compute evaluation metrics
SELECT
  mean_squared_error,
  mean_absolute_error,
  r2_score
FROM
  ML.EVALUATE(
    MODEL `arr-coc-project.dataset.patch_selector_xgboost`,
    (
      SELECT * FROM `arr-coc-project.dataset.image_features`
      WHERE MOD(ABS(FARM_FINGERPRINT(image_id)), 10) = 0  -- 10% test set
    )
  );
```

**Step 5: Batch prediction for new images:**
```sql
-- Predict relevance for new image batch
CREATE OR REPLACE TABLE `arr-coc-project.dataset.relevance_predictions` AS
SELECT
  image_id,
  query_text,
  predicted_high_relevance_patches,

  -- Determine token allocation (64-400 range)
  CASE
    WHEN predicted_high_relevance_patches >= 10 THEN 400  -- High relevance: 400 tokens
    WHEN predicted_high_relevance_patches >= 5 THEN 256   -- Medium: 256 tokens
    WHEN predicted_high_relevance_patches >= 2 THEN 128   -- Low: 128 tokens
    ELSE 64  -- Minimal relevance: 64 tokens
  END AS allocated_tokens
FROM
  ML.PREDICT(
    MODEL `arr-coc-project.dataset.patch_selector_xgboost`,
    (
      SELECT * FROM `arr-coc-project.dataset.new_images_features`
    )
  );
```

**Step 6: Export predictions to Cloud SQL for arr-coc-0-1:**
```sql
-- Write predictions back to Cloud SQL for integration with arr-coc-0-1
EXPORT DATA OPTIONS(
  uri='gs://arr-coc-bucket/predictions/batch_*.csv',
  format='CSV',
  overwrite=true,
  header=true
) AS
SELECT * FROM `arr-coc-project.dataset.relevance_predictions`;

-- Or use federated query to write directly to Cloud SQL
CALL EXTERNAL_QUERY(
  'arr-coc-project.us-central1.cloud-sql-connection',
  '''INSERT INTO relevance_predictions
     (image_id, query_text, allocated_tokens)
     VALUES %s''',
  (SELECT image_id, query_text, allocated_tokens
   FROM `arr-coc-project.dataset.relevance_predictions`)
);
```

### Integration with arr-coc-0-1 Pipeline

**arr-coc-0-1 workflow:**

1. **Extract patch embeddings** (Qwen-VL) → Cloud SQL
2. **Aggregate features** → BigQuery (federated query)
3. **Train XGBoost relevance model** → BigQuery ML
4. **Batch predict token allocation** → BigQuery ML.PREDICT
5. **Export predictions** → Cloud SQL
6. **Dynamic LOD allocation** → arr-coc-0-1 Python (read from Cloud SQL)

**Benefits:**
- Leverage BigQuery for large-scale feature engineering (billions of patches)
- SQL-based ML (no Python training code needed)
- Automatic hyperparameter tuning (AutoML integration)
- Seamless integration with Cloud SQL (federated queries)
- Cost-effective batch inference ($5/TB vs Vertex AI batch pricing)

---

## Sources

**Google Cloud Documentation:**
- [BigQuery ML Introduction](https://cloud.google.com/bigquery/docs/bqml-introduction) - Model types and capabilities (accessed 2025-11-16)
- [BigQuery Pricing](https://cloud.google.com/bigquery/pricing) - On-demand and capacity pricing (accessed 2025-11-16)
- [BigQuery ML Inference Overview](https://docs.cloud.google.com/bigquery/docs/inference-overview) - ML.PREDICT performance (accessed 2025-11-16)
- [Export BigQuery ML Models](https://docs.cloud.google.com/bigquery/docs/exporting-models) - TensorFlow SavedModel format (accessed 2025-11-16)
- [Manage BigQuery ML Models in Vertex AI](https://docs.cloud.google.com/bigquery/docs/managing-models-vertex) - Model Registry integration (accessed 2025-11-16)
- [Vertex AI Batch Prediction from BigQuery](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions) - BigQuery input/output (accessed 2025-11-16)
- [Cloud SQL Federated Queries](https://docs.cloud.google.com/bigquery/docs/cloud-sql-federated-queries) - EXTERNAL_QUERY syntax (accessed 2025-11-16)

**Source Documents:**
- [gcloud-data/00-storage-bigquery-ml-data.md](../gcloud-data/00-storage-bigquery-ml-data.md) - BigQuery ML feature engineering patterns

**Web Research (accessed 2025-11-16):**
- [OWOX BI: A Guide to BigQuery ML](https://www.owox.com/blog/articles/bigquery-ml) - Model types and AutoML (May 2025)
- [BigQuery ML Inference Engine](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-ml-inference-engine) - Scaling inference (March 2023)
- [BigQuery CUDs Guide](https://www.economize.cloud/blog/bigquery-commited-use-discounts/) - Cost optimization strategies (July 2025)
- [Transitioning to Capacity-Based Pricing](https://techblog.rtbhouse.com/transitioning-to-capacity-based-pricing-in-google-bigquery/) - RTB House case study (March 2025)

**Additional References:**
- Search results for "BigQuery ML CREATE MODEL XGBoost DNN AutoML 2024" - Model training patterns
- Search results for "BigQuery federated queries Cloud SQL Sheets 2024" - External data source integration
- Search results for "BigQuery cost optimization slot reservations on-demand 2024" - Pricing model comparison

---

*This document provides comprehensive coverage of BigQuery ML and Vertex AI integration, enabling SQL-based machine learning with enterprise-scale deployment, from training through production serving and cost optimization.*
