# Cloud Storage & BigQuery for ML Data: Complete Production Guide

**Comprehensive guide to data management for machine learning on GCP - storage organization, BigQuery feature engineering, Feature Store patterns, and preprocessing pipelines**

This document covers production-ready data management strategies for ML workflows on Google Cloud Platform, focusing on Cloud Storage bucket organization, BigQuery for feature engineering and ML, Vertex AI Feature Store integration, and scalable ETL patterns.

---

## Section 1: Cloud Storage Organization for ML (~150 lines)

### ML-Optimized Bucket Structure

**Recommended folder hierarchy for ML projects:**
```bash
gs://ml-project-data/
├── raw/                          # Original, immutable datasets
│   ├── images/
│   ├── text/
│   └── metadata/
├── processed/                    # Cleaned, transformed data
│   ├── train/
│   ├── val/
│   └── test/
├── features/                     # Engineered features
│   ├── embeddings/
│   ├── aggregations/
│   └── temporal/
├── tfrecords/                    # Training-ready format
│   ├── sharded-train-*.tfrecord
│   ├── sharded-val-*.tfrecord
│   └── sharded-test-*.tfrecord
├── checkpoints/                  # Training checkpoints
│   ├── experiment-1/
│   └── experiment-2/
├── models/                       # Exported models
│   ├── production/
│   └── staging/
├── artifacts/                    # Pipeline artifacts
│   └── metadata/
└── snapshots/                    # Versioned dataset snapshots
    ├── 2024-01-15/
    └── 2024-02-01/
```

From [Cloud Storage documentation](https://cloud.google.com/storage/docs) (accessed 2025-02-03):
- Standard storage class for active training data
- Nearline/Coldline for archived experiments
- Lifecycle policies for automatic transitions

**Benefits of this structure:**
- Clear separation of concerns (raw vs processed)
- Easy rollback (snapshots folder)
- Training-ready formats (tfrecords)
- Experiment isolation (checkpoints per experiment)

### Data Versioning Strategies

**Immutable snapshots for reproducibility:**
```bash
# Create timestamped snapshot
SNAPSHOT_DATE=$(date +%Y-%m-%d)
gsutil -m cp -r gs://ml-data/processed/ \
    gs://ml-data/snapshots/$SNAPSHOT_DATE/

# Train using specific snapshot
TRAIN_DATA="gs://ml-data/snapshots/2024-01-15/processed/train/"
```

**Metadata tracking:**
```json
{
  "snapshot_id": "2024-01-15",
  "source": "gs://ml-data/raw/",
  "preprocessing_version": "v2.1.0",
  "total_samples": 1000000,
  "train_samples": 800000,
  "val_samples": 100000,
  "test_samples": 100000,
  "created_by": "data-pipeline-job-123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Bucket Organization Best Practices

**Multi-environment setup:**
```bash
# Development
gs://ml-data-dev/

# Staging
gs://ml-data-staging/

# Production
gs://ml-data-prod/
```

**Regional co-location:**
```bash
# Create bucket in same region as Vertex AI training
gsutil mb -p PROJECT_ID \
    -c STANDARD \
    -l us-central1 \
    gs://ml-training-data
```

From [Cloud Storage best practices](https://cloud.google.com/storage/docs/best-practices) (accessed 2025-02-03):
- Co-locate buckets with compute resources (reduces latency and egress costs)
- Use Standard storage for frequently accessed data (<30 days)
- Use Nearline for monthly access, Coldline for quarterly

### Access Control Patterns

**Service account permissions (least privilege):**
```bash
# Training job: read-only access to data
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Pipeline job: read-write for processing
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:pipeline-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

**Bucket-level granular permissions:**
```bash
# Read-only for training data
gsutil iam ch serviceAccount:training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://ml-data/processed/

# Write access for checkpoints only
gsutil iam ch serviceAccount:training-sa@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
    gs://ml-data/checkpoints/
```

### Lifecycle Management for Cost Optimization

**Automated transitions based on age:**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["raw/", "processed/"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["checkpoints/", "snapshots/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesPrefix": ["checkpoints/experiment-"]
        }
      }
    ]
  }
}
```

Apply lifecycle policy:
```bash
gsutil lifecycle set lifecycle.json gs://ml-data
```

**Cost savings example:**
- Standard storage: $0.020/GB/month
- Nearline (after 30 days): $0.010/GB/month (50% savings)
- Coldline (after 90 days): $0.004/GB/month (80% savings)

### Data Transfer Optimization

**Parallel composite uploads for large files:**
```bash
# Enable for files >150MB (10x speedup)
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M \
    cp -r local_dataset/ gs://ml-data/raw/

# Parallel download
gsutil -m cp -r gs://ml-data/processed/ ./local_data/
```

**Resumable transfers:**
```bash
# Resume interrupted uploads
gsutil -m cp -r -c local_dataset/ gs://ml-data/raw/

# Synchronize directories (only transfer changed files)
gsutil -m rsync -r -d local_dataset/ gs://ml-data/raw/
```

From [Cloud Storage performance guide](https://cloud.google.com/storage/docs/cloud-storage-fuse/performance) (accessed 2025-02-03):
- Parallel composite uploads: up to 10x faster for large files
- Use `-m` flag for multi-threading across multiple files
- Request rate limits: 5,000 ops/second per bucket

---

## Section 2: BigQuery for ML Feature Engineering (~200 lines)

### BigQuery ML Overview

**What is BigQuery ML?**

BigQuery ML enables data analysts and scientists to build and deploy machine learning models using SQL, directly within BigQuery. No need to export data to separate ML frameworks.

From [How BigQuery ML does feature preprocessing](https://cloud.google.com/blog/products/data-analytics/how-bigquery-ml-does-feature-preprocessing/) (accessed 2025-02-03):
> "In machine learning, transforming raw data into meaningful features—a preprocessing step known as feature engineering—is a critical step. BigQuery ML's new reusable and modular feature engineering functions make it easier to build and maintain machine learning pipelines."

**Key capabilities:**
- SQL-based model training (logistic regression, DNN, XGBoost, AutoML)
- Built-in feature preprocessing (normalization, encoding, transformations)
- Integration with Vertex AI for deployment
- Automatic hyperparameter tuning
- Model evaluation and explainability

### Feature Engineering with SQL

**Feature transformations in BigQuery ML:**

**1. Numerical transformations:**
```sql
-- Normalization and scaling
CREATE OR REPLACE MODEL `project.dataset.model_name`
TRANSFORM(
  -- Z-score normalization
  ML.STANDARD_SCALER(age) OVER() AS age_normalized,

  -- Min-max scaling
  ML.MIN_MAX_SCALER(income) OVER() AS income_scaled,

  -- Log transformation
  LOG(revenue + 1) AS revenue_log,

  -- Bucketization
  ML.BUCKETIZE(price, [0, 10, 50, 100, 500]) AS price_bucket,

  -- Features passed through
  category,
  is_premium
)
OPTIONS(model_type='logistic_reg', input_label_cols=['is_premium'])
AS
SELECT * FROM `project.dataset.training_data`;
```

From [BigQuery ML feature preprocessing](https://cloud.google.com/blog/products/data-analytics/how-bigquery-ml-does-feature-preprocessing/) (January 2024):
- TRANSFORM clause: reusable, modular preprocessing
- Automatically applied during training AND prediction (prevents training-serving skew)
- Saves transformed features with model artifact

**2. Categorical encoding:**
```sql
-- One-hot encoding and feature crosses
CREATE OR REPLACE MODEL `project.dataset.churn_model`
TRANSFORM(
  -- One-hot encoding
  ML.ONE_HOT_ENCODER(country) OVER() AS country_encoded,

  -- Feature hashing (for high cardinality)
  ML.FEATURE_CROSS([product_id, user_id], 1000) AS product_user_cross,

  -- Label encoding
  ML.LABEL_ENCODER(device_type) OVER() AS device_encoded,

  -- Polynomial features
  ML.POLYNOMIAL_EXPAND(age, 2) AS age_polynomial,

  -- Target variable
  churned
)
OPTIONS(model_type='boosted_tree_classifier', input_label_cols=['churned'])
AS
SELECT * FROM `project.dataset.customer_data`;
```

**3. Temporal features:**
```sql
-- Time-based feature engineering
CREATE OR REPLACE MODEL `project.dataset.demand_forecast`
TRANSFORM(
  -- Extract date components
  EXTRACT(DAYOFWEEK FROM timestamp) AS day_of_week,
  EXTRACT(HOUR FROM timestamp) AS hour_of_day,
  EXTRACT(MONTH FROM timestamp) AS month,

  -- Rolling aggregations (7-day window)
  AVG(sales) OVER(
    ORDER BY timestamp
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS sales_7day_avg,

  -- Lag features
  LAG(sales, 1) OVER(ORDER BY timestamp) AS sales_prev_day,
  LAG(sales, 7) OVER(ORDER BY timestamp) AS sales_prev_week,

  -- Target
  sales AS target_sales
)
OPTIONS(model_type='arima_plus', time_series_timestamp_col='timestamp')
AS
SELECT * FROM `project.dataset.sales_data`;
```

### BigQuery ML Training Workflows

**Complete training pipeline:**
```sql
-- Step 1: Create training dataset with features
CREATE OR REPLACE TABLE `project.dataset.features_train` AS
SELECT
  user_id,
  -- Aggregations
  COUNT(*) AS total_purchases,
  SUM(amount) AS total_spent,
  AVG(amount) AS avg_purchase,
  MAX(purchase_date) AS last_purchase_date,

  -- Recency, Frequency, Monetary (RFM)
  DATE_DIFF(CURRENT_DATE(), MAX(purchase_date), DAY) AS recency,
  COUNT(DISTINCT purchase_date) AS frequency,
  SUM(amount) AS monetary,

  -- Behavioral features
  COUNT(DISTINCT product_category) AS category_diversity,
  AVG(CASE WHEN is_discount = true THEN 1 ELSE 0 END) AS discount_rate,

  -- Target
  MAX(CASE WHEN churned = true THEN 1 ELSE 0 END) AS churned
FROM `project.dataset.user_events`
WHERE event_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY user_id;

-- Step 2: Train model
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type='boosted_tree_classifier',
  input_label_cols=['churned'],
  auto_class_weights=true,
  max_iterations=50,
  l1_reg=0.1,
  l2_reg=0.1
) AS
SELECT * EXCEPT(user_id)
FROM `project.dataset.features_train`;

-- Step 3: Evaluate model
SELECT
  *
FROM
  ML.EVALUATE(MODEL `project.dataset.churn_model`,
    (SELECT * EXCEPT(user_id) FROM `project.dataset.features_test`));

-- Step 4: Make predictions
SELECT
  user_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability
FROM
  ML.PREDICT(MODEL `project.dataset.churn_model`,
    (SELECT * FROM `project.dataset.features_current`));
```

### Exporting Features for External Training

**BigQuery to Cloud Storage:**
```sql
-- Export features as CSV
EXPORT DATA OPTIONS(
  uri='gs://ml-data/features/user_features_*.csv',
  format='CSV',
  overwrite=true,
  header=true
) AS
SELECT * FROM `project.dataset.features_train`;

-- Export as Parquet (better for large datasets)
EXPORT DATA OPTIONS(
  uri='gs://ml-data/features/user_features_*.parquet',
  format='PARQUET',
  overwrite=true
) AS
SELECT * FROM `project.dataset.features_train`;

-- Export as TFRecord (for TensorFlow)
EXPORT DATA OPTIONS(
  uri='gs://ml-data/features/user_features_*.tfrecord',
  format='TFRECORD',
  overwrite=true
) AS
SELECT * FROM `project.dataset.features_train`;
```

### Query Optimization for ML Workloads

**Best practices for feature engineering queries:**

**1. Partitioning and clustering:**
```sql
-- Create partitioned table
CREATE OR REPLACE TABLE `project.dataset.events_partitioned`
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, event_type
AS
SELECT * FROM `project.dataset.events_raw`;

-- Query with partition filter (reduces cost)
SELECT
  user_id,
  COUNT(*) AS event_count
FROM `project.dataset.events_partitioned`
WHERE DATE(event_timestamp) BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY user_id;
```

From [Best practices for implementing ML on GCP](https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices) (accessed 2025-02-03):
> "If you're building your model using BigQuery ML, use the transformations built into BigQuery for preprocessing data. If you're using AutoML or custom training, use Dataflow for preprocessing."

**2. Approximate aggregations for speed:**
```sql
-- Use APPROX functions for large datasets
SELECT
  category,
  APPROX_COUNT_DISTINCT(user_id) AS unique_users,
  APPROX_QUANTILES(price, 100)[OFFSET(50)] AS median_price,
  APPROX_TOP_COUNT(product_id, 10) AS top_products
FROM `project.dataset.sales`
GROUP BY category;
```

**3. Materialized views for repeated features:**
```sql
-- Create materialized view (auto-refreshed)
CREATE MATERIALIZED VIEW `project.dataset.user_features_mv` AS
SELECT
  user_id,
  COUNT(*) AS total_events,
  MAX(event_timestamp) AS last_event,
  ARRAY_AGG(DISTINCT event_type) AS event_types
FROM `project.dataset.events_partitioned`
GROUP BY user_id;

-- Query materialized view (faster than recomputing)
SELECT * FROM `project.dataset.user_features_mv`;
```

### Cost Optimization for BigQuery ML

**Query cost reduction strategies:**

**1. Use partition pruning:**
```sql
-- Scans 1 day (cheap)
SELECT * FROM `project.dataset.events_partitioned`
WHERE DATE(event_timestamp) = '2024-01-15';

-- Scans entire table (expensive)
SELECT * FROM `project.dataset.events_partitioned`
WHERE user_id = 'user123';
```

**2. Select only needed columns:**
```sql
-- Good: Select specific columns
SELECT user_id, amount, category
FROM `project.dataset.sales`;

-- Bad: SELECT * scans all columns
SELECT * FROM `project.dataset.sales`;
```

**3. Use clustering for filtered queries:**
```sql
-- Cluster by commonly filtered columns
CREATE OR REPLACE TABLE `project.dataset.sales_clustered`
CLUSTER BY category, user_segment
AS SELECT * FROM `project.dataset.sales`;

-- Queries filtering on category/user_segment run faster and cheaper
SELECT SUM(amount)
FROM `project.dataset.sales_clustered`
WHERE category = 'electronics' AND user_segment = 'premium';
```

**Cost monitoring:**
```sql
-- Check query cost before running
-- Use BigQuery UI "Query Validator" to see bytes processed
-- Estimated cost: $5 per TB scanned

-- Monthly cost tracking
SELECT
  user_email,
  SUM(total_bytes_processed) / POW(10, 12) AS tb_processed,
  SUM(total_bytes_processed) / POW(10, 12) * 5 AS estimated_cost_usd
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_USER
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY user_email
ORDER BY tb_processed DESC;
```

From [BigQuery pricing](https://cloud.google.com/bigquery/pricing) (accessed 2025-02-03):
- On-demand: $5 per TB scanned
- Flat-rate: $2,000/month for 100 slots (predictable costs)
- Storage: $0.020/GB/month (active), $0.010/GB/month (long-term)

---

## Section 3: Vertex AI Feature Store Integration (~150 lines)

### Feature Store Architecture

**Three-tier hierarchy:**
```
Feature Store
├── Entity Type: User
│   ├── Feature: age (INT64)
│   ├── Feature: location (STRING)
│   └── Feature: lifetime_value (DOUBLE)
├── Entity Type: Product
│   ├── Feature: category (STRING)
│   ├── Feature: price (DOUBLE)
│   └── Feature: inventory_count (INT64)
└── Entity Type: Transaction
    ├── Feature: amount (DOUBLE)
    ├── Feature: timestamp (TIMESTAMP)
    └── Feature: is_fraud (BOOL)
```

From [Vertex AI Feature Store documentation](https://cloud.google.com/vertex-ai/docs/featurestore/overview) (accessed 2025-02-03):
- Centralized feature repository
- Online serving (<10ms latency)
- Offline serving (batch training)
- Point-in-time correctness (avoid data leakage)

### Creating Feature Stores from BigQuery

**Python API integration:**
```python
from google.cloud import aiplatform
from google.cloud import bigquery

# Initialize clients
aiplatform.init(project='PROJECT_ID', location='us-central1')
bq_client = bigquery.Client()

# Step 1: Create feature store
feature_store = aiplatform.Featurestore.create(
    featurestore_id='ml_features',
    online_serving_config={
        'fixed_node_count': 2  # Bigtable nodes for online serving
    }
)

# Step 2: Create entity type
entity_type = feature_store.create_entity_type(
    entity_type_id='user',
    description='User features'
)

# Step 3: Create features
features = entity_type.batch_create_features({
    'age': {'value_type': 'INT64', 'description': 'User age'},
    'location': {'value_type': 'STRING', 'description': 'User location'},
    'lifetime_value': {'value_type': 'DOUBLE', 'description': 'Total user spend'},
    'last_purchase_days': {'value_type': 'INT64', 'description': 'Days since last purchase'}
})

# Step 4: Import features from BigQuery
entity_type.ingest_from_bq(
    feature_ids=['age', 'location', 'lifetime_value', 'last_purchase_days'],
    bq_source_uri='bq://PROJECT_ID.dataset.user_features',
    entity_id_field='user_id',
    feature_time_field='timestamp'
)
```

### Online vs Offline Serving

**Online serving (low-latency predictions):**
```python
# Fetch features for real-time prediction
feature_values = entity_type.read(
    entity_ids=['user_123', 'user_456'],
    feature_ids=['age', 'lifetime_value']
)

# Use in model serving
prediction_input = {
    'age': feature_values['user_123']['age'],
    'lifetime_value': feature_values['user_123']['lifetime_value']
}
model.predict(prediction_input)
```

**Offline serving (batch training):**
```python
# Export features for training
training_dataset = entity_type.batch_serve_to_bq(
    bq_destination_output_uri='bq://PROJECT_ID.dataset.training_features',
    read_instances_uri='bq://PROJECT_ID.dataset.entity_ids',
    feature_destination_fields={
        'age': 'feature_age',
        'lifetime_value': 'feature_ltv'
    }
)

# Point-in-time correctness (critical for training)
# Features as of specific timestamp (avoids data leakage)
training_dataset = entity_type.batch_serve_to_bq(
    bq_destination_output_uri='bq://PROJECT_ID.dataset.training_features',
    read_instances_uri='bq://PROJECT_ID.dataset.entity_ids_with_timestamps',
    feature_destination_fields={'age': 'feature_age'},
    start_time='2023-01-01T00:00:00Z',
    end_time='2023-12-31T23:59:59Z'
)
```

### Feature Store Best Practices

From [Best practices for Vertex AI Feature Store](https://docs.cloud.google.com/vertex-ai/docs/featurestore/best-practices) (accessed 2025-02-03):

**1. Use IAM for team access control:**
```bash
# Data scientists: read-only
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="group:data-scientists@company.com" \
    --role="roles/aiplatform.featurestoreDataViewer"

# Data engineers: full access
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="group:data-engineers@company.com" \
    --role="roles/aiplatform.featurestoreAdmin"
```

**2. Monitor and tune resources:**
```python
# Check feature store metrics
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"

# Query online serving latency
results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="aiplatform.googleapis.com/featurestore/online_serving/request_latency"',
        "interval": {"end_time": {"seconds": int(time.time())}},
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
    }
)

for result in results:
    print(f"Latency: {result.points[0].value.distribution_value}")
```

**3. Use autoscaling for cost reduction:**
```python
# Enable autoscaling for online serving
feature_store.update(
    online_serving_config={
        'scaling': {
            'min_node_count': 1,
            'max_node_count': 5,
            'cpu_utilization_target': 60  # Scale when CPU > 60%
        }
    }
)
```

### Feature Store vs BigQuery Trade-offs

| Aspect | Feature Store | BigQuery |
|--------|---------------|----------|
| **Online serving** | <10ms latency | Not designed for this |
| **Offline serving** | Batch export | Native (SQL queries) |
| **Point-in-time correctness** | Built-in | Manual SQL logic |
| **Cost** | Storage + serving nodes | Query + storage |
| **Best for** | Real-time predictions | Feature engineering |

**Recommended pattern:**
1. Engineer features in BigQuery (SQL is powerful for transformations)
2. Import to Feature Store for serving
3. Use Feature Store for online predictions
4. Use BigQuery for batch training data generation

---

## Section 4: Data Pipeline Patterns (~150 lines)

### ETL Patterns for ML

**Dataflow for large-scale preprocessing:**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Define preprocessing pipeline
def preprocess_example(element):
    # Parse raw data
    user_id, features = element.split(',', 1)

    # Feature transformations
    processed = {
        'user_id': user_id,
        'features_normalized': normalize(features),
        'features_encoded': encode_categorical(features)
    }

    return processed

# Run Dataflow pipeline
options = PipelineOptions(
    project='PROJECT_ID',
    region='us-central1',
    runner='DataflowRunner',
    temp_location='gs://ml-data/temp/',
    staging_location='gs://ml-data/staging/'
)

with beam.Pipeline(options=options) as pipeline:
    (pipeline
     | 'Read from GCS' >> beam.io.ReadFromText('gs://ml-data/raw/*.csv')
     | 'Preprocess' >> beam.Map(preprocess_example)
     | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
         'PROJECT_ID:dataset.features',
         schema='user_id:STRING,features_normalized:STRING,features_encoded:STRING',
         write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
     ))
```

### Data Validation

**TensorFlow Data Validation (TFDV):**
```python
import tensorflow_data_validation as tfdv

# Generate statistics
stats = tfdv.generate_statistics_from_csv('gs://ml-data/processed/train.csv')

# Infer schema
schema = tfdv.infer_schema(stats)

# Validate new data
new_stats = tfdv.generate_statistics_from_csv('gs://ml-data/processed/new_batch.csv')
anomalies = tfdv.validate_statistics(new_stats, schema)

# Check for anomalies
if anomalies.anomaly_info:
    print("Data anomalies detected!")
    for feature, info in anomalies.anomaly_info.items():
        print(f"{feature}: {info.description}")
else:
    print("Data validation passed")
```

### Preprocessing at Scale

**Apache Beam preprocessing for TFRecord generation:**
```python
import apache_beam as beam
import tensorflow as tf

def create_tfrecord_example(row):
    """Convert row to TFRecord Example."""
    features = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['image_bytes']])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['label']])),
        'metadata': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['metadata'].encode()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

# Preprocessing pipeline
with beam.Pipeline(options=options) as p:
    (p
     | 'Read from BigQuery' >> beam.io.ReadFromBigQuery(
         query='SELECT * FROM `project.dataset.training_data`',
         use_standard_sql=True
     )
     | 'Preprocess' >> beam.Map(preprocess_function)
     | 'Create TFRecords' >> beam.Map(create_tfrecord_example)
     | 'Write TFRecords' >> beam.io.WriteToTFRecord(
         'gs://ml-data/tfrecords/train',
         num_shards=100  # Sharding for parallel training
     ))
```

### Orchestration with Vertex AI Pipelines

**Complete ML pipeline with KFP:**
```python
from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform

@dsl.component(base_image='python:3.9')
def extract_features(
    input_table: str,
    output_uri: dsl.Output[dsl.Dataset]
):
    from google.cloud import bigquery

    client = bigquery.Client()

    # Extract features
    query = f"""
    SELECT
      user_id,
      COUNT(*) AS event_count,
      AVG(value) AS avg_value
    FROM `{input_table}`
    GROUP BY user_id
    """

    df = client.query(query).to_dataframe()
    df.to_csv(output_uri.path, index=False)

@dsl.component(base_image='gcr.io/project-id/trainer:latest')
def train_model(
    training_data: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    epochs: int = 10
):
    import pandas as pd
    import torch

    # Load data
    df = pd.read_csv(training_data.path)

    # Train model
    trained_model = train(df, epochs)

    # Save model
    torch.save(trained_model, model.path)

@dsl.pipeline(name='ml-data-pipeline')
def ml_pipeline(
    input_table: str = 'project.dataset.events',
    epochs: int = 10
):
    extract_op = extract_features(input_table=input_table)

    train_op = train_model(
        training_data=extract_op.outputs['output_uri'],
        epochs=epochs
    )

# Compile and run
compiler.Compiler().compile(ml_pipeline, 'pipeline.json')

job = aiplatform.PipelineJob(
    display_name='ml-data-pipeline',
    template_path='pipeline.json',
    parameter_values={'input_table': 'project.dataset.events', 'epochs': 20}
)
job.run()
```

---

## Section 5: Best Practices (~100 lines)

### Data Organization Conventions

**Naming conventions:**
```
# Buckets
gs://PROJECT_ID-ENV-PURPOSE/
gs://mycompany-prod-ml-data/
gs://mycompany-dev-training-data/

# Tables
project.dataset_purpose.table_version
project.ml_features.user_features_v2
project.training.events_2024_01

# Models
project.models.churn_predictor_v3
```

**File formats:**
- **Raw data**: CSV, JSON (human-readable, easy to debug)
- **Processed features**: Parquet (columnar, efficient for analytics)
- **Training data**: TFRecord (optimized for TensorFlow), Parquet (PyTorch)
- **Checkpoints**: Native framework format (PyTorch .pt, TensorFlow SavedModel)

### Versioning Strategies

**Semantic versioning for datasets:**
```
v1.0.0 - Initial dataset
v1.1.0 - Added new features (backward compatible)
v2.0.0 - Breaking change (different schema)
```

**Git-like workflow:**
```bash
# Tag dataset versions
gsutil label ch -l version:v1.0.0 gs://ml-data/snapshots/2024-01-15/

# Track lineage in metadata
{
  "dataset_version": "v1.0.0",
  "parent_version": null,
  "schema_version": "1.0",
  "git_commit": "abc123",
  "pipeline_version": "v2.1.0"
}
```

### Cost Optimization Summary

**Storage costs:**
1. Use lifecycle policies (Standard → Nearline → Coldline)
2. Delete old checkpoints (keep only best/final)
3. Compress data (gzip, Parquet with Snappy)

**BigQuery costs:**
1. Partition tables by date
2. Cluster by commonly filtered columns
3. Use SELECT only needed columns (avoid SELECT *)
4. Use approximate functions for exploratory analysis
5. Consider flat-rate pricing for heavy usage

**Compute costs:**
1. Use Spot VMs for preprocessing (60-91% cheaper)
2. Shut down idle Feature Store nodes
3. Use Dataflow autoscaling
4. Co-locate data and compute (same region)

**Monitoring costs:**
```sql
-- BigQuery cost tracking query
SELECT
  EXTRACT(DATE FROM creation_time) AS date,
  SUM(total_bytes_processed) / POW(10, 12) AS tb_processed,
  SUM(total_bytes_processed) / POW(10, 12) * 5 AS cost_usd
FROM `project.region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY date
ORDER BY date DESC;
```

### Security and Compliance

**Encryption:**
- Data at rest: Google-managed or customer-managed keys (CMEK)
- Data in transit: TLS 1.2+
- Sensitive data: Use Cloud Data Loss Prevention (DLP) API

**Access auditing:**
```bash
# Enable Cloud Audit Logs
gcloud logging read "resource.type=gcs_bucket" \
    --limit 10 \
    --format json
```

**Data governance:**
- Use Data Catalog for metadata management
- Tag sensitive columns (PII, PHI)
- Implement data retention policies
- Regular access reviews

---

## Sources

**Google Cloud Documentation:**
- [Cloud Storage documentation](https://cloud.google.com/storage/docs) - Bucket management, lifecycle policies (accessed 2025-02-03)
- [Cloud Storage best practices](https://cloud.google.com/storage/docs/best-practices) - Performance and cost optimization (accessed 2025-02-03)
- [Cloud Storage FUSE performance](https://cloud.google.com/storage/docs/cloud-storage-fuse/performance) - File caching and parallel downloads (accessed 2025-02-03)
- [BigQuery documentation](https://cloud.google.com/bigquery/docs) - SQL reference and ML capabilities (accessed 2025-02-03)
- [BigQuery pricing](https://cloud.google.com/bigquery/pricing) - On-demand and flat-rate pricing (accessed 2025-02-03)
- [Best practices for implementing ML on GCP](https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices) - Preprocessing recommendations (accessed 2025-02-03)
- [Vertex AI Feature Store overview](https://cloud.google.com/vertex-ai/docs/featurestore/overview) - Architecture and capabilities (accessed 2025-02-03)
- [Vertex AI Feature Store best practices](https://docs.cloud.google.com/vertex-ai/docs/featurestore/best-practices) - IAM, scaling, monitoring (accessed 2025-02-03)

**Web Research (accessed 2025-02-03):**
- [How BigQuery ML does feature preprocessing](https://cloud.google.com/blog/products/data-analytics/how-bigquery-ml-does-feature-preprocessing/) - Google Cloud Blog, January 2024
- [BigQuery and Document AI Layout Parser for document preprocessing](https://cloud.google.com/blog/products/data-analytics/bigquery-and-document-ai-layout-parser-for-document-preprocessing/) - ML.PROCESS_DOCUMENT function, November 2024
- [RAG with BigQuery and Langchain](https://cloud.google.com/blog/products/ai-machine-learning/rag-with-bigquery-and-langchain-in-cloud) - Vector search integration, June 2024
- [NL2SQL with BigQuery and Gemini](https://cloud.google.com/blog/products/data-analytics/nl2sql-with-bigquery-and-gemini) - Natural language queries, November 2024
- [Synthetic data generation with Gretel and BigQuery DataFrames](https://cloud.google.com/blog/products/data-analytics/synthetic-data-generation-with-gretel-and-bigquery-dataframes) - DataFrames API, November 2024

**Additional References:**
- Search results for "Cloud Storage machine learning data lakes 2024 2025" - Data lake architecture patterns
- Search results for "BigQuery ML feature engineering patterns 2024" - Feature transformation techniques
- Search results for "Vertex AI Feature Store best practices 2025" - Production serving patterns

---

*This document provides production-ready patterns for ML data management on GCP, covering Cloud Storage organization, BigQuery feature engineering, Feature Store integration, and scalable ETL pipelines for training enterprise-scale ML models.*
