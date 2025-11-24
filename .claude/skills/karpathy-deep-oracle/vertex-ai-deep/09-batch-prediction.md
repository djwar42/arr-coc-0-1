# Vertex AI Batch Prediction

## Overview

Vertex AI Batch Prediction enables cost-effective, large-scale inference for ML models by processing data asynchronously. Unlike online prediction endpoints that serve real-time requests, batch prediction is optimized for scenarios where you need to process large datasets without immediate response requirements.

**Key characteristics:**
- **50% cost reduction** compared to online prediction (Gemini models)
- Asynchronous processing of large datasets
- Seamless integration with BigQuery and Cloud Storage
- Support for custom models, AutoML, and Generative AI models
- Scalable infrastructure managed by Vertex AI

**Use cases:**
- Periodic model evaluation and scoring
- Processing historical data for analytics
- Bulk inference for recommendation systems
- Feature generation for downstream ML pipelines
- Cost-sensitive workloads with flexible timing

From [Vertex AI Batch Prediction documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) (accessed 2025-02-03):
- Batch prediction offers 50% discount compared to real-time inference
- Ideal for large-scale, non-urgent inference tasks
- Processes data from BigQuery or Cloud Storage

## Batch Prediction Architecture

### Workflow Components

**Input Sources:**
1. **BigQuery Tables**
   - Native integration with data warehouse
   - Direct SQL query results as input
   - Supports complex data transformations
   - Ideal for structured tabular data

2. **Cloud Storage (GCS)**
   - JSONL, CSV, TFRecord formats
   - Supports unstructured data (images, text)
   - Batch file organization patterns
   - Large file handling (multi-GB datasets)

**Processing Pipeline:**
```
Input Data (BigQuery/GCS)
    ↓
Vertex AI Batch Job Creation
    ↓
Resource Allocation (machines, GPUs)
    ↓
Parallel Inference Execution
    ↓
Output Writing (BigQuery/GCS)
    ↓
Job Completion & Results
```

**Output Destinations:**
1. **BigQuery Tables**
   - Structured prediction results
   - Direct integration with BI tools
   - SQL query access to predictions
   - Automatic schema generation

2. **Cloud Storage**
   - JSONL output format
   - Organized by batch job ID
   - Supports downstream processing
   - Archive-friendly storage

From [Understanding Vertex AI Batch Prediction](https://www.gcpstudyhub.com/pages/blog/understanding-vertex-ai-batch-prediction-for-the-professional-ml-engineer-exam) (accessed 2025-02-03):
- Batch prediction follows straightforward pattern handling large-scale processing
- Input data preparation, job submission, resource allocation, parallel execution
- Results written to specified output location

### Resource Management

**Machine Allocation:**
- Vertex AI automatically determines optimal machine count
- Based on input data size and model requirements
- Default: Up to 10 machines (configurable)
- Example: 40 n1-highmem-8 machines for large jobs

**Scaling Behavior:**
```python
# Vertex AI determines replica count dynamically
# Factors considered:
# - Input data volume
# - Model complexity
# - Machine type specifications
# - max_replica_count parameter (if set)
```

**Performance Characteristics:**
- Parallel processing across multiple machines
- Automatic load balancing
- Fault tolerance with retries
- Progress tracking via API

From [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-02-03):
- Batch prediction uses 40 n1-highmem-8 machines (example configuration)
- Machine type and count affect cost and performance

## BigQuery Integration

### Input Configuration

**BigQuery as Input Source:**

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='my-project', location='us-central1')

# Create batch prediction job with BigQuery input
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name='bigquery-batch-prediction',
    model_name='projects/PROJECT_ID/locations/REGION/models/MODEL_ID',

    # BigQuery input configuration
    bigquery_source='bq://my-project.dataset.input_table',

    # BigQuery output configuration
    bigquery_destination_prefix='bq://my-project.dataset',

    # Machine configuration
    machine_type='n1-standard-4',
    max_replica_count=10,

    # Optional: Batch size optimization
    batch_size=64
)
```

**Input Table Requirements:**
- Each row represents one prediction request
- Column names must match model input schema
- Support for nested/repeated fields (STRUCT, ARRAY)
- No explicit ordering required

**Schema Mapping:**
```sql
-- Example input table structure
CREATE TABLE `project.dataset.input_table` AS
SELECT
  user_id,
  feature_1,
  feature_2,
  feature_3,
  STRUCT(lat, lon) AS location,
  ARRAY_AGG(interaction) AS history
FROM source_data
```

From [Batch inference for BigQuery](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-bigquery) (accessed 2025-02-03):
- Native BigQuery integration for seamless workflow
- Direct table access without data export
- Column-based input mapping

### Output Schema

**BigQuery Output Structure:**

Vertex AI creates output tables with the following schema:
```
Original Input Columns (preserved)
    +
Prediction Columns:
    - predicted_<target>
    - predicted_<target>_probs (classification)
    - prediction_scores
    - prediction_errors (if any)
```

**Example Output:**
```sql
SELECT
  user_id,                    -- Original input
  feature_1, feature_2,       -- Original input
  predicted_churn,            -- Prediction result
  predicted_churn_probs,      -- Confidence scores
  prediction_error            -- Error message (if failed)
FROM `project.dataset.predictions_<job_id>`
```

**Handling Predictions:**
```sql
-- Query successful predictions
SELECT *
FROM `project.dataset.predictions_<job_id>`
WHERE prediction_error IS NULL

-- Analyze prediction distribution
SELECT
  predicted_label,
  COUNT(*) as count,
  AVG(confidence_score) as avg_confidence
FROM `project.dataset.predictions_<job_id>`
WHERE prediction_error IS NULL
GROUP BY predicted_label
```

From [Vertex AI batch prediction BigQuery output schema](https://stackoverflow.com/questions/79642358/vertex-ai-batch-prediction-bigquery-output-schema) (accessed 2025-02-03):
- Output schema customization not natively supported
- Schema automatically generated based on model output
- Includes original input columns plus prediction columns

### BigQuery Best Practices

**Data Preparation:**
1. **Optimize Table Size**
   - Partition large tables by date/key
   - Use clustering for query performance
   - Avoid scanning unnecessary columns

2. **Schema Design**
   - Match model input requirements exactly
   - Use appropriate data types
   - Handle NULL values explicitly

3. **Query Patterns**
   ```sql
   -- Efficient input table creation
   CREATE TABLE `project.dataset.batch_input`
   PARTITION BY DATE(created_at)
   CLUSTER BY user_segment
   AS
   SELECT
     id,
     feature_array,
     metadata
   FROM source_table
   WHERE created_at >= '2025-01-01'
   ```

**Cost Optimization:**
- Use BigQuery's flat-rate pricing for large batch jobs
- Pre-filter data to reduce row count
- Leverage table partitioning to minimize scans
- Monitor BigQuery slot usage during batch prediction

From [Five integrations between Vertex AI and BigQuery](https://cloud.google.com/blog/products/ai-machine-learning/five-integrations-between-vertex-ai-and-bigquery) (accessed 2025-02-03):
- Specify BigQuery table as source and destination for prediction job
- Seamless integration reduces data movement
- Direct analytics on prediction results

## GCS Input/Output Patterns

### Input Data Formats

**Supported Formats:**

1. **JSONL (JSON Lines)**
   ```jsonl
   {"feature_1": 1.0, "feature_2": "text", "id": "001"}
   {"feature_1": 2.0, "feature_2": "more text", "id": "002"}
   {"feature_1": 3.0, "feature_2": "another", "id": "003"}
   ```

2. **CSV**
   ```csv
   feature_1,feature_2,id
   1.0,text,001
   2.0,more text,002
   3.0,another,003
   ```

3. **TFRecord**
   - Binary format for TensorFlow models
   - Efficient serialization
   - Supports complex nested structures

**File Organization Patterns:**

```
gs://bucket-name/batch-input/
├── batch_001.jsonl          # Single file
├── batch_002.jsonl
└── partitioned/              # Directory with multiple files
    ├── part-00000.jsonl
    ├── part-00001.jsonl
    └── part-00002.jsonl
```

**GCS Input Configuration:**
```python
# Single file input
gcs_source='gs://bucket/input/data.jsonl'

# Multiple files (wildcard pattern)
gcs_source='gs://bucket/input/*.jsonl'

# Directory of files
gcs_source='gs://bucket/input/'
```

From [Get batch inferences for AutoML models](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions) (accessed 2025-02-03):
- Supports JSONL, CSV, TFRecord formats
- File organization affects parallelization
- Wildcards for multiple file processing

### Output Organization

**GCS Output Structure:**

```
gs://bucket/output/prediction_<job_id>/
├── prediction.results-00000-of-00010
├── prediction.results-00001-of-00010
├── ...
└── prediction.results-00009-of-00010
```

**Output File Format (JSONL):**
```jsonl
{
  "instance": {"feature_1": 1.0, "feature_2": "text"},
  "prediction": 0.85,
  "scores": [0.15, 0.85]
}
```

**Reading Output:**
```python
import json
from google.cloud import storage

def read_batch_predictions(bucket_name, job_id):
    """Read batch prediction results from GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all prediction result files
    prefix = f'prediction_{job_id}/'
    blobs = bucket.list_blobs(prefix=prefix)

    predictions = []
    for blob in blobs:
        if 'prediction.results-' in blob.name:
            content = blob.download_as_text()
            for line in content.split('\n'):
                if line.strip():
                    predictions.append(json.loads(line))

    return predictions
```

### File Size Optimization

**Best Practices:**

1. **File Size Guidelines**
   - Target: 100MB - 1GB per file
   - Too small: Overhead from many files
   - Too large: Reduced parallelization

2. **Sharding Strategy**
   ```python
   # Split large dataset into optimal chunks
   import math

   def calculate_optimal_shards(total_size_gb, target_size_gb=0.5):
       return math.ceil(total_size_gb / target_size_gb)

   # Example: 10GB dataset → 20 files of ~500MB each
   num_shards = calculate_optimal_shards(10)
   ```

3. **Compression**
   - Use gzip compression for JSONL/CSV
   - Reduces storage costs
   - Vertex AI automatically decompresses
   ```
   gs://bucket/input/data.jsonl.gz  # Supported
   ```

From [Understanding Vertex AI Batch Prediction](https://www.gcpstudyhub.com/pages/blog/understanding-vertex-ai-batch-prediction-for-the-professional-ml-engineer-exam) (accessed 2025-02-03):
- File organization patterns affect processing efficiency
- Sharding enables parallel processing
- Optimal file sizes balance overhead and parallelism

## Performance Optimization

### Throughput Tuning

**Replica Count Configuration:**

```python
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    # ... other parameters ...

    # Performance tuning
    starting_replica_count=5,    # Initial parallelization
    max_replica_count=20,        # Maximum scale-out

    # Machine type selection
    machine_type='n1-highmem-8',  # For memory-intensive models
    # machine_type='n1-standard-4',  # For standard models
    # machine_type='a2-highgpu-1g',  # For GPU inference

    # Batch size per replica
    batch_size=64                # Predictions per request
)
```

**Machine Type Selection:**

| Model Type | Recommended Machine | Use Case |
|------------|-------------------|----------|
| Small models (<100MB) | n1-standard-4 | Cost-effective CPU inference |
| Large models (>1GB) | n1-highmem-8 | Memory-intensive models |
| Vision models | n1-standard-8 | Balanced CPU/memory |
| GPU models | a2-highgpu-1g | GPU-accelerated inference |

**Batch Size Optimization:**
```python
# Small models: Larger batches
batch_size=128  # Fast inference, minimize overhead

# Large models: Smaller batches
batch_size=16   # Avoid OOM, stable performance

# GPU models: Tuned to GPU memory
batch_size=32   # Maximize GPU utilization
```

From [Vertex AI Batch Prediction optimization](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) (accessed 2025-02-03):
- Machine type affects throughput and cost
- Replica count enables horizontal scaling
- Batch size tuning critical for performance

### Monitoring & Debugging

**Job Status Tracking:**

```python
# Monitor job progress
batch_prediction_job = aiplatform.BatchPredictionJob.get(
    resource_name='projects/.../batchPredictionJobs/12345'
)

print(f"State: {batch_prediction_job.state}")
print(f"Start time: {batch_prediction_job.start_time}")
print(f"Update time: {batch_prediction_job.update_time}")

# Check for errors
if batch_prediction_job.error:
    print(f"Error: {batch_prediction_job.error.message}")
```

**Performance Metrics:**
```python
# Calculate throughput
import time

start_time = batch_prediction_job.start_time
end_time = batch_prediction_job.end_time
duration_seconds = (end_time - start_time).total_seconds()

# Assuming you know input size
num_predictions = 1_000_000
throughput = num_predictions / duration_seconds

print(f"Throughput: {throughput:.2f} predictions/second")
print(f"Duration: {duration_seconds / 60:.2f} minutes")
```

**Common Issues & Solutions:**

1. **Slow Processing**
   - Increase `max_replica_count`
   - Use larger machine types
   - Optimize batch_size
   - Check input data format (CSV slower than JSONL)

2. **Out of Memory Errors**
   - Reduce batch_size
   - Use n1-highmem machines
   - Split large input files

3. **Timeout Issues**
   - Set explicit `max_replica_count`
   - Verify model serving latency
   - Check for data quality issues

From [Improving Debugging in Vertex AI Batch Prediction Jobs](https://datatonic.com/insights/vertex-ai-improving-debugging-batch-prediction/) (accessed 2025-02-03):
- Debugging reduces manual effort in maintaining ML models
- Monitor job status and resource utilization
- Error handling and retry strategies

### Error Handling

**Prediction Error Patterns:**

```python
# Read predictions with error handling
def process_predictions_with_errors(output_table):
    from google.cloud import bigquery

    client = bigquery.Client()

    # Query with error handling
    query = f"""
    SELECT
      *,
      CASE
        WHEN prediction_error IS NULL THEN 'success'
        WHEN prediction_error LIKE '%timeout%' THEN 'timeout'
        WHEN prediction_error LIKE '%invalid%' THEN 'invalid_input'
        ELSE 'unknown_error'
      END AS error_category
    FROM `{output_table}`
    """

    results = client.query(query).result()

    # Aggregate error statistics
    error_stats = {}
    for row in results:
        category = row.error_category
        error_stats[category] = error_stats.get(category, 0) + 1

    return error_stats
```

**Retry Strategy:**
```python
# Identify failed predictions for retry
failed_query = f"""
CREATE TABLE `project.dataset.retry_input` AS
SELECT * EXCEPT(prediction_error)
FROM `project.dataset.predictions_{job_id}`
WHERE prediction_error IS NOT NULL
"""

# Submit retry job
retry_job = aiplatform.BatchPredictionJob.create(
    job_display_name='batch-prediction-retry',
    model_name=model_name,
    bigquery_source=f'bq://project.dataset.retry_input',
    bigquery_destination_prefix=f'bq://project.dataset.retry_output'
)
```

## Cost Optimization

### Pricing Model

**Cost Components:**

1. **Compute Costs**
   - Machine hours × machine type rate
   - Charged per second (60-second minimum)
   - Different rates for CPU vs GPU machines

2. **Prediction Costs**
   - Per-prediction fees (AutoML models)
   - Volume discounts for large batches
   - Gemini: 50% discount vs online prediction

3. **Storage Costs**
   - BigQuery storage for input/output
   - GCS storage for files
   - Network egress (if cross-region)

**Pricing Examples (US regions):**
```
CPU Batch Prediction:
- n1-standard-4: ~$0.19/hour
- n1-highmem-8: ~$0.47/hour

GPU Batch Prediction:
- a2-highgpu-1g: ~$3.67/hour (includes 1 A100 GPU)

AutoML Models:
- Batch prediction: $3.15 per 1000 predictions (tabular)

Gemini Batch Prediction:
- 50% discount compared to online prediction
- Example: Gemini 1.5 Pro batch = $0.00125/1k input tokens
```

From [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-02-03):
- Batch prediction pricing varies by model type and machine configuration
- Cost optimization strategies available
- 50% discount for Gemini batch prediction

### Cost Reduction Strategies

**1. Batch Size Optimization**
```python
# Larger batches reduce per-prediction overhead
batch_size=128  # vs batch_size=16

# Cost impact:
# - Fewer API calls
# - Better machine utilization
# - Reduced total machine hours
```

**2. Machine Type Selection**
```python
# Choose appropriate machine type
# Over-provisioning wastes money
# Under-provisioning causes OOM

# For 500MB model:
machine_type='n1-standard-4'  # Sufficient, cost-effective

# NOT:
machine_type='n1-highmem-16'  # 3x more expensive, unnecessary
```

**3. Scheduling Strategy**
```python
# Run during off-peak hours (if applicable)
# Use committed use discounts for predictable workloads
# Leverage sustained use discounts (automatic)

# Example: Weekly batch prediction
# - Schedule for weekends
# - Combine multiple smaller batches
# - Reduce total machine hours
```

**4. Data Preprocessing**
```sql
-- Pre-filter data to reduce volume
CREATE TABLE `project.dataset.filtered_input` AS
SELECT *
FROM `project.dataset.raw_data`
WHERE needs_prediction = TRUE
  AND last_prediction_date < CURRENT_DATE() - 7
```

**5. Spot VM Usage (Future)**
```python
# Note: Spot VMs not yet supported for batch prediction
# but expected in future releases
# Potential: 60-90% cost savings
```

From [Cost Optimization Strategies for AI Workloads](https://www.infracloud.io/blogs/ai-workload-cost-optimization/) (accessed 2025-02-03):
- Model optimization reduces computational overhead
- Lower costs without compromising reliability
- Infrastructure-level and model-level optimization

### Cost Monitoring

**Tracking Batch Prediction Costs:**

```python
from google.cloud import billing_v1

def get_batch_prediction_costs(project_id, start_date, end_date):
    """Query Cloud Billing for batch prediction costs"""

    client = billing_v1.CloudBillingClient()

    # Query billing data
    # Note: Requires billing export to BigQuery

    query = f"""
    SELECT
      service.description AS service,
      sku.description AS sku,
      SUM(cost) AS total_cost,
      SUM(usage.amount) AS total_usage
    FROM `project.billing_export.gcp_billing_export`
    WHERE service.description = 'Vertex AI'
      AND sku.description LIKE '%Batch Prediction%'
      AND DATE(usage_start_time) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY service, sku
    ORDER BY total_cost DESC
    """

    return query
```

**Cost Analysis:**
```python
# Calculate cost per prediction
def analyze_prediction_costs(job_id, num_predictions):
    """
    Analyze cost breakdown for a batch prediction job
    """

    # Get job details
    job = aiplatform.BatchPredictionJob.get(job_id)

    # Calculate duration
    duration_hours = (job.end_time - job.start_time).total_seconds() / 3600

    # Machine costs (example: n1-standard-4)
    machine_cost_per_hour = 0.19
    num_machines = 10  # From job configuration

    total_machine_cost = duration_hours * machine_cost_per_hour * num_machines
    cost_per_prediction = total_machine_cost / num_predictions

    print(f"Total Cost: ${total_machine_cost:.2f}")
    print(f"Cost per Prediction: ${cost_per_prediction:.6f}")
    print(f"Throughput: {num_predictions / duration_hours:.0f} predictions/hour")

    return {
        'total_cost': total_machine_cost,
        'cost_per_prediction': cost_per_prediction,
        'duration_hours': duration_hours
    }
```

## Best Practices

### Data Preparation

**Input Data Quality:**
1. **Validation Before Submission**
   ```python
   def validate_input_data(input_table):
       """Validate input data meets model requirements"""

       query = f"""
       SELECT
         COUNT(*) as total_rows,
         COUNTIF(feature_1 IS NULL) as null_feature_1,
         COUNTIF(feature_2 IS NULL) as null_feature_2,
         MIN(feature_1) as min_f1,
         MAX(feature_1) as max_f1
       FROM `{input_table}`
       """

       # Check for issues
       # - NULL values
       # - Out-of-range values
       # - Schema mismatches
   ```

2. **Handle Missing Values**
   ```sql
   -- Impute missing values before batch prediction
   CREATE TABLE `project.dataset.clean_input` AS
   SELECT
     id,
     IFNULL(feature_1, 0.0) AS feature_1,
     IFNULL(feature_2, 'unknown') AS feature_2
   FROM `project.dataset.raw_input`
   ```

3. **Deduplication**
   ```sql
   -- Remove duplicates to avoid wasted predictions
   CREATE TABLE `project.dataset.deduped_input` AS
   SELECT DISTINCT *
   FROM `project.dataset.raw_input`
   ```

### Job Configuration

**Optimal Settings:**

```python
# Production-ready batch prediction configuration
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name=f'batch-prediction-{timestamp}',
    model_name=model_resource_name,

    # Input/Output
    bigquery_source=input_table,
    bigquery_destination_prefix=output_dataset,

    # Performance
    machine_type='n1-standard-4',
    starting_replica_count=5,
    max_replica_count=20,
    batch_size=64,

    # Reliability
    generate_explanation=False,  # Disable if not needed (faster)

    # Monitoring
    labels={
        'environment': 'production',
        'team': 'ml-ops',
        'cost-center': 'analytics'
    }
)
```

**Job Naming Conventions:**
```python
import datetime

# Descriptive job names for tracking
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_version = 'v1.2.3'
dataset_name = 'customer-churn'

job_name = f'batch-pred-{model_version}-{dataset_name}-{timestamp}'
```

### Monitoring & Alerting

**Job Status Checks:**
```python
def monitor_batch_job(job_resource_name, check_interval=60):
    """
    Monitor batch prediction job until completion
    """
    import time

    job = aiplatform.BatchPredictionJob.get(job_resource_name)

    while job.state not in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
        print(f"Job state: {job.state}")
        time.sleep(check_interval)
        job.refresh()

    if job.state == 'JOB_STATE_SUCCEEDED':
        print(f"✓ Job completed successfully")
        print(f"Duration: {(job.end_time - job.start_time).total_seconds() / 60:.1f} minutes")
    else:
        print(f"✗ Job failed: {job.error}")

    return job
```

**Cloud Monitoring Integration:**
```python
from google.cloud import monitoring_v3

def create_batch_prediction_alert(project_id):
    """
    Create alert for batch prediction failures
    """
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Batch Prediction Failure Alert",
        conditions=[{
            "display_name": "Job Failed",
            "condition_threshold": {
                "filter": 'resource.type="aiplatform.googleapis.com/BatchPredictionJob"'
                          ' AND metric.type="aiplatform.googleapis.com/prediction/error_count"',
                "comparison": "COMPARISON_GT",
                "threshold_value": 0,
                "duration": {"seconds": 60}
            }
        }],
        notification_channels=[],  # Add notification channels
        alert_strategy={
            "auto_close": {"seconds": 3600}
        }
    )

    return client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

### Testing & Validation

**Pre-Production Testing:**

1. **Small Sample Test**
   ```python
   # Test with 1% of data first
   test_query = f"""
   CREATE TABLE `project.dataset.test_input` AS
   SELECT *
   FROM `project.dataset.full_input`
   WHERE RAND() < 0.01
   LIMIT 1000
   """

   # Run test batch prediction
   test_job = aiplatform.BatchPredictionJob.create(
       job_display_name='test-batch-prediction',
       model_name=model_name,
       bigquery_source='bq://project.dataset.test_input',
       bigquery_destination_prefix='bq://project.dataset.test_output'
   )
   ```

2. **Output Validation**
   ```python
   def validate_predictions(output_table):
       """Validate prediction output quality"""

       query = f"""
       SELECT
         COUNT(*) as total_predictions,
         COUNTIF(prediction_error IS NOT NULL) as error_count,
         AVG(confidence_score) as avg_confidence,
         MIN(confidence_score) as min_confidence,
         MAX(confidence_score) as max_confidence
       FROM `{output_table}`
       """

       # Check:
       # - Error rate < 1%
       # - Confidence scores in expected range
       # - No NULL predictions
   ```

3. **Performance Benchmarking**
   ```python
   # Compare against previous runs
   def benchmark_job_performance(job_id, baseline_job_id):
       """Compare job performance against baseline"""

       current_job = aiplatform.BatchPredictionJob.get(job_id)
       baseline_job = aiplatform.BatchPredictionJob.get(baseline_job_id)

       current_duration = (current_job.end_time - current_job.start_time).total_seconds()
       baseline_duration = (baseline_job.end_time - baseline_job.start_time).total_seconds()

       speedup = baseline_duration / current_duration

       print(f"Current duration: {current_duration / 60:.1f} min")
       print(f"Baseline duration: {baseline_duration / 60:.1f} min")
       print(f"Speedup: {speedup:.2f}x")
   ```

## Complete Example

### End-to-End Batch Prediction Workflow

```python
"""
Complete batch prediction workflow with BigQuery integration
"""

from google.cloud import aiplatform, bigquery
import datetime

class BatchPredictionPipeline:
    def __init__(self, project_id, location, model_name):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        # Initialize clients
        aiplatform.init(project=project_id, location=location)
        self.bq_client = bigquery.Client(project=project_id)

    def prepare_input_data(self, source_table, filter_condition=None):
        """Prepare and validate input data"""

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        input_table = f'{self.project_id}.batch_predictions.input_{timestamp}'

        # Create input table with filtering
        query = f"""
        CREATE TABLE `{input_table}` AS
        SELECT *
        FROM `{source_table}`
        """

        if filter_condition:
            query += f" WHERE {filter_condition}"

        # Execute query
        job = self.bq_client.query(query)
        job.result()

        print(f"✓ Input table created: {input_table}")
        return input_table

    def run_batch_prediction(self, input_table, output_dataset):
        """Execute batch prediction job"""

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create batch prediction job
        job = aiplatform.BatchPredictionJob.create(
            job_display_name=f'batch-pred-{timestamp}',
            model_name=self.model_name,

            # BigQuery I/O
            bigquery_source=f'bq://{input_table}',
            bigquery_destination_prefix=f'bq://{self.project_id}.{output_dataset}',

            # Performance tuning
            machine_type='n1-standard-4',
            starting_replica_count=5,
            max_replica_count=20,
            batch_size=64,

            # Metadata
            labels={
                'pipeline': 'batch-prediction',
                'timestamp': timestamp
            }
        )

        print(f"✓ Batch prediction job created: {job.resource_name}")
        print(f"  Job name: {job.display_name}")

        # Wait for completion
        job.wait()

        if job.state == 'JOB_STATE_SUCCEEDED':
            print(f"✓ Job completed successfully")
            output_table = job.output_info.bigquery_output_table
            return output_table
        else:
            raise Exception(f"Job failed: {job.error}")

    def analyze_results(self, output_table):
        """Analyze prediction results"""

        query = f"""
        SELECT
          COUNT(*) as total_predictions,
          COUNTIF(prediction_error IS NULL) as successful_predictions,
          COUNTIF(prediction_error IS NOT NULL) as failed_predictions,
          AVG(predicted_score) as avg_predicted_score
        FROM `{output_table}`
        """

        results = self.bq_client.query(query).result()
        stats = list(results)[0]

        print("\nPrediction Results:")
        print(f"  Total: {stats.total_predictions}")
        print(f"  Successful: {stats.successful_predictions}")
        print(f"  Failed: {stats.failed_predictions}")
        print(f"  Avg Score: {stats.avg_predicted_score:.4f}")

        return stats

    def run_pipeline(self, source_table, output_dataset, filter_condition=None):
        """Execute complete pipeline"""

        print("=== Batch Prediction Pipeline ===\n")

        # Step 1: Prepare input
        print("Step 1: Preparing input data...")
        input_table = self.prepare_input_data(source_table, filter_condition)

        # Step 2: Run batch prediction
        print("\nStep 2: Running batch prediction...")
        output_table = self.run_batch_prediction(input_table, output_dataset)

        # Step 3: Analyze results
        print("\nStep 3: Analyzing results...")
        stats = self.analyze_results(output_table)

        print("\n=== Pipeline Complete ===")
        return output_table, stats

# Usage
if __name__ == '__main__':
    pipeline = BatchPredictionPipeline(
        project_id='my-project',
        location='us-central1',
        model_name='projects/my-project/locations/us-central1/models/123456'
    )

    output_table, stats = pipeline.run_pipeline(
        source_table='my-project.raw_data.customers',
        output_dataset='predictions',
        filter_condition='created_date >= "2025-01-01"'
    )
```

## Sources

**Official Documentation:**
- [Vertex AI Batch Prediction - Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) - Batch inference with 50% cost reduction (accessed 2025-02-03)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) - Cost structure and optimization strategies (accessed 2025-02-03)
- [Get batch inferences for AutoML models](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions) - Input data requirements and workflow (accessed 2025-02-03)
- [Batch inference for BigQuery](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-bigquery) - Native BigQuery integration (accessed 2025-02-03)

**Blog Posts & Guides:**
- [Five integrations between Vertex AI and BigQuery](https://cloud.google.com/blog/products/ai-machine-learning/five-integrations-between-vertex-ai-and-bigquery) - Cloud Blog, Feb 10, 2022 (accessed 2025-02-03)
- [Understanding Vertex AI Batch Prediction for the Professional ML Engineer Exam](https://www.gcpstudyhub.com/pages/blog/understanding-vertex-ai-batch-prediction-for-the-professional-ml-engineer-exam) - GCP Study Hub, Aug 24, 2025 (accessed 2025-02-03)
- [A Complete Guide to Batch Predictions with Gemini, Vertex AI and BigQuery](https://medium.com/@ikaromoribayashi/scaling-artificial-intelligence-a-complete-guide-to-batch-predictions-with-gemini-vertex-ai-and-51e8c92a9578) - Medium, 5 months ago (accessed 2025-02-03)
- [Improving Debugging in Vertex AI Batch Prediction Jobs](https://datatonic.com/insights/vertex-ai-improving-debugging-batch-prediction/) - Datatonic (accessed 2025-02-03)

**Cost Optimization:**
- [Cost Optimization Strategies for AI Workloads](https://www.infracloud.io/blogs/ai-workload-cost-optimization/) - InfraCloud, Aug 11, 2025 (accessed 2025-02-03)
- [Google Vertex AI Pricing - Cost Breakdown & Savings Guide](https://www.pump.co/blog/google-vertex-ai-pricing) - Pump.co, May 21, 2025 (accessed 2025-02-03)

**Community Discussions:**
- [Vertex AI batch prediction BigQuery output schema](https://stackoverflow.com/questions/79642358/vertex-ai-batch-prediction-bigquery-output-schema) - Stack Overflow, 5 months ago (accessed 2025-02-03)
- [Performance degradation when using Batch prediction](https://discuss.google.dev/t/performance-degradation-when-using-batch-prediction/184716) - Google Developer Forums, Mar 21, 2025 (accessed 2025-02-03)

**API Reference:**
- [REST Resource: projects.locations.batchPredictionJobs](https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.batchPredictionJobs) - Sep 22, 2025 (accessed 2025-02-03)
