# Batch Prediction & Feature Store

**Large-scale inference workflows and centralized feature management in Vertex AI**

From [Vertex AI Batch Predictions Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction) (accessed 2025-11-16):
> "Batch predictions let you send a large number of multimodal prompts in a single batch request. Make a batch prediction against a model by using input from BigQuery or Cloud Storage."

---

## Overview

Vertex AI provides two critical production capabilities: **Batch Prediction** for large-scale offline inference and **Feature Store** for centralized feature management. Batch Prediction enables cost-effective processing of millions of predictions, while Feature Store ensures consistent feature computation across training and serving.

**Key use cases:**
- **Batch Prediction**: Dataset evaluation, A/B testing, recommendation generation, bulk scoring
- **Feature Store**: Real-time serving (<10ms), offline training data, point-in-time correctness

**Performance characteristics:**
- Batch Prediction throughput: Thousands of predictions per minute (model-dependent)
- Feature Store online serving: <10ms p99 latency
- Feature Store offline serving: BigQuery-scale analytics

From [Medium - Scaling Batch Predictions with Gemini](https://medium.com/@ikaromoribayashi/scaling-artificial-intelligence-a-complete-guide-to-batch-predictions-with-gemini-vertex-ai-and-51e8c92a9578) (accessed 2025-11-16):
> "This guide has demonstrated a powerful and scalable workflow for performing batch predictions with Gemini models, using Python, Pandas, BigQuery and Cloud Storage."

---

## Section 1: Batch Prediction Architecture (~120 lines)

### BigQuery → Vertex AI → GCS Workflow

**Standard batch prediction pipeline:**
```
[BigQuery Table]          [Cloud Storage Bucket]
   |                           |
   | Input data                | Input JSONL files
   ↓                           ↓
[Vertex AI Batch Prediction Job]
   |
   | Model inference (distributed across workers)
   ↓
[Output Location]
   ├─ BigQuery table (structured results)
   └─ GCS bucket (JSONL predictions)
```

**Workflow components:**

**1. Input sources:**
- **BigQuery**: Structured data, automatic schema detection, SQL filtering
- **Cloud Storage**: JSONL files, unstructured data, custom formats
- **Hybrid**: BigQuery for metadata + GCS for binary data (images, videos)

**2. Job configuration:**
```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="my-project", location="us-central1")

# Create batch prediction job
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name="fraud-detection-batch",
    model_name="projects/123/locations/us-central1/models/456",

    # Input from BigQuery
    bigquery_source="bq://my-project.fraud_data.transactions",

    # Output to BigQuery
    bigquery_destination_prefix="bq://my-project.predictions",

    # Configuration
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,

    # Batch settings
    starting_replica_count=5,
    max_replica_count=10,
    batch_size=64,

    # Advanced options
    generate_explanation=True,
    explanation_metadata={...},
    model_parameters={"confidence_threshold": 0.7}
)

# Monitor job
batch_prediction_job.wait()
print(f"Job state: {batch_prediction_job.state}")
print(f"Predictions written to: {batch_prediction_job.output_info}")
```

**3. Output formats:**

**BigQuery output schema:**
```sql
-- Automatic schema (predictions appended to input)
SELECT
    original_col1,
    original_col2,
    predicted_label,
    prediction_confidence,
    prediction_timestamp
FROM `my-project.predictions.batch_job_123`
```

**GCS JSONL output:**
```json
{"instance": {...}, "prediction": [0.92, 0.08], "deployed_model_id": "456"}
{"instance": {...}, "prediction": [0.15, 0.85], "deployed_model_id": "456"}
```

From [Vertex AI Get Batch Predictions Documentation](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) (accessed 2025-11-16):
> "To make a batch inference request, you specify an input source and an output location, either Cloud Storage or BigQuery, where Vertex AI stores the batch prediction results."

### Distributed Execution

**Worker pool architecture:**
- **Job orchestrator**: Splits input into shards, assigns to workers
- **Worker instances**: Execute model inference in parallel
- **Result aggregator**: Combines outputs, writes to destination

**Scaling considerations:**
- Starting replicas: Initial worker count (fast job start)
- Max replicas: Maximum autoscale limit (cost control)
- Batch size: Predictions per worker iteration (throughput vs memory)

**Cost optimization patterns:**
```python
# Development: Small dataset, minimal replicas
dev_config = {
    "machine_type": "n1-standard-2",
    "starting_replica_count": 1,
    "max_replica_count": 2,
    "batch_size": 16
}

# Production: Large dataset, autoscaling
prod_config = {
    "machine_type": "n1-highmem-8",
    "accelerator_type": "NVIDIA_TESLA_T4",
    "starting_replica_count": 10,
    "max_replica_count": 50,
    "batch_size": 128
}
```

---

## Section 2: Feature Store Architecture (~140 lines)

### Entity Types, Features, and FeatureViews

From [Vertex AI Feature Store Data Model](https://docs.cloud.google.com/vertex-ai/docs/featurestore/concepts) (accessed 2025-11-16):
> "A featurestore is the top-level container for entity types, features, and feature values. Typically, an organization creates one shared featurestore to centralize feature management."

**Hierarchical data model:**
```
Featurestore (project-level container)
├─ EntityType: User
│  ├─ Feature: age (INT64)
│  ├─ Feature: lifetime_value (DOUBLE)
│  └─ Feature: country (STRING)
├─ EntityType: Product
│  ├─ Feature: category (STRING)
│  ├─ Feature: price (DOUBLE)
│  └─ Feature: inventory_count (INT64)
└─ EntityType: Transaction
   ├─ Feature: amount (DOUBLE)
   ├─ Feature: merchant_id (STRING)
   └─ Feature: fraud_score (DOUBLE)
```

**Key concepts:**

**1. Entity Type:**
- Represents a business object (user, product, session)
- Has unique entity ID (user_id, product_id, session_id)
- Contains related features

**2. Feature:**
- Individual measurable property
- Typed: INT64, DOUBLE, STRING, BOOL, INT64_ARRAY, etc.
- Versioned: Historical values stored for point-in-time correctness

**3. FeatureView (Vertex AI Feature Store new version):**
- Query-optimized view of features
- Supports BigQuery source tables
- Automatic sync and refresh

From [Medium - Vertex AI Feature Store Features and Advantages](https://medium.com/@ajayverma23/exploring-vertex-ai-feature-store-features-and-advantages-12014ead55d3) (accessed 2025-11-16):
> "This blog will delve into the key features and advantages of Vertex AI Feature Store, explaining its attributes such as Entity Type, Entity, Feature, Featurestore, and Feature Value."

### Creating a Feature Store

**Python API example:**
```python
from google.cloud import aiplatform_v1

# Create featurestore
featurestore_service_client = aiplatform_v1.FeaturestoreServiceClient()

featurestore = aiplatform_v1.Featurestore(
    online_serving_config=aiplatform_v1.Featurestore.OnlineServingConfig(
        fixed_node_count=2  # Dedicated serving nodes
    )
)

create_featurestore_request = aiplatform_v1.CreateFeaturestoreRequest(
    parent="projects/my-project/locations/us-central1",
    featurestore_id="production_features",
    featurestore=featurestore
)

featurestore_operation = featurestore_service_client.create_featurestore(
    request=create_featurestore_request
)
created_featurestore = featurestore_operation.result()

# Create entity type
entity_type = aiplatform_v1.EntityType(
    description="User demographic and behavioral features"
)

create_entity_type_request = aiplatform_v1.CreateEntityTypeRequest(
    parent=created_featurestore.name,
    entity_type_id="user",
    entity_type=entity_type
)

entity_type_operation = featurestore_service_client.create_entity_type(
    request=create_entity_type_request
)
created_entity_type = entity_type_operation.result()

# Create features
features = [
    aiplatform_v1.Feature(
        value_type=aiplatform_v1.Feature.ValueType.INT64,
        description="User age in years"
    ),
    aiplatform_v1.Feature(
        value_type=aiplatform_v1.Feature.ValueType.DOUBLE,
        description="Total lifetime value in USD"
    )
]

feature_ids = ["age", "lifetime_value"]

batch_create_features_request = aiplatform_v1.BatchCreateFeaturesRequest(
    parent=created_entity_type.name,
    requests=[
        aiplatform_v1.CreateFeatureRequest(
            feature=feature,
            feature_id=feature_id
        )
        for feature, feature_id in zip(features, feature_ids)
    ]
)

batch_create_features_response = featurestore_service_client.batch_create_features(
    request=batch_create_features_request
)
```

### Storage Architecture

**Offline storage (BigQuery):**
- Historical feature values
- Training dataset generation
- Batch analytics
- Point-in-time queries

**Online storage (Bigtable):**
- Low-latency serving (<10ms)
- Latest feature values
- Real-time predictions
- High throughput (thousands of QPS)

From [Google Cloud - Vertex AI Feature Store Documentation](https://docs.cloud.google.com/vertex-ai/docs/featurestore) (accessed 2025-11-16):
> "Vertex AI Feature Store is optimized for ultra-low latency serving and lets you do the following: Store and maintain your offline feature data in BigQuery, and serve features from an online store with low latency."

---

## Section 3: Online vs Offline Serving (~100 lines)

### Online Serving (<10ms Latency)

**Architecture:**
```
Client Request (gRPC/REST)
    ↓
[Vertex AI Endpoint]
    ↓
[Bigtable Online Store]
    ├─ Row key: entity_id
    ├─ Column families: features
    └─ Cell: latest feature value
    ↓
Response (JSON with feature values)
```

**Online serving API:**
```python
from google.cloud import aiplatform

# Read features for real-time prediction
def get_user_features(user_id: str) -> dict:
    """Fetch features for online serving (<10ms)"""

    featurestore_online_service_client = (
        aiplatform.gapic.FeaturestoreOnlineServingServiceClient()
    )

    read_feature_values_request = aiplatform.gapic.ReadFeatureValuesRequest(
        entity_type="projects/123/locations/us-central1/featurestores/prod/entityTypes/user",
        entity_id=user_id,
        feature_selector=aiplatform.gapic.FeatureSelector(
            id_matcher=aiplatform.gapic.IdMatcher(
                ids=["age", "lifetime_value", "country", "fraud_risk"]
            )
        )
    )

    response = featurestore_online_service_client.read_feature_values(
        request=read_feature_values_request
    )

    # Extract feature values
    features = {}
    for feature_value in response.entity_view.data:
        feature_id = feature_value.feature_id
        value = feature_value.value

        # Handle different value types
        if value.HasField("int64_value"):
            features[feature_id] = value.int64_value
        elif value.HasField("double_value"):
            features[feature_id] = value.double_value
        elif value.HasField("string_value"):
            features[feature_id] = value.string_value

    return features

# Example usage
user_features = get_user_features("user_12345")
# Returns: {"age": 35, "lifetime_value": 1250.50, "country": "US", "fraud_risk": 0.02}
```

**Performance characteristics:**
- **p50 latency**: 2-5ms
- **p99 latency**: <10ms
- **Throughput**: 10,000+ QPS per node
- **Consistency**: Eventually consistent (streaming ingestion)

From [Medium - Centralizing ML Features through Feature Store](https://medium.com/google-cloud/centralizing-ml-features-through-feature-store-in-google-cloud-vertex-ai-300f5b37b5d8) (accessed 2025-11-16):
> "While batch serving is optimized for throughput and scale, online serving is tailored for speed and low latency, ensuring that time-sensitive predictions receive the most up-to-date feature values."

### Offline Serving (BigQuery Analytics)

**Batch feature retrieval:**
```python
from google.cloud import bigquery

def create_training_dataset(
    start_date: str,
    end_date: str,
    output_table: str
) -> None:
    """Generate point-in-time correct training dataset"""

    client = bigquery.Client()

    query = f"""
    SELECT
        u.user_id,
        u.feature_timestamp,

        -- Features at point in time
        u.age,
        u.lifetime_value,
        u.country,

        -- Label (target variable)
        l.churned AS label

    FROM `project.featurestore.user_features` u
    INNER JOIN `project.labels.churn_labels` l
        ON u.user_id = l.user_id

    -- Point-in-time correctness: Only use features available before label
    WHERE u.feature_timestamp < l.label_timestamp
        AND u.feature_timestamp >= '{start_date}'
        AND u.feature_timestamp < '{end_date}'

    -- Get latest features before each label
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY u.user_id, l.label_timestamp
        ORDER BY u.feature_timestamp DESC
    ) = 1
    """

    job_config = bigquery.QueryJobConfig(
        destination=output_table,
        write_disposition="WRITE_TRUNCATE"
    )

    query_job = client.query(query, job_config=job_config)
    query_job.result()  # Wait for job to complete

    print(f"Training dataset created: {output_table}")

# Generate dataset for model training
create_training_dataset(
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_table="project.training_data.churn_features_2024"
)
```

**Comparison matrix:**

| Aspect | Online Serving | Offline Serving |
|--------|---------------|-----------------|
| **Storage** | Bigtable | BigQuery |
| **Latency** | <10ms (p99) | Seconds to minutes |
| **Throughput** | 10K+ QPS | Millions of rows |
| **Use case** | Real-time predictions | Training data, batch analytics |
| **Consistency** | Latest values | Historical point-in-time |
| **Query pattern** | Single entity lookup | Bulk SQL queries |
| **Cost** | Node-hours + requests | BigQuery query pricing |

---

## Section 4: Feature Streaming from Pub/Sub (~90 lines)

### Real-Time Feature Updates

From [Vertex AI Streaming Import Documentation](https://docs.cloud.google.com/vertex-ai/docs/featurestore/ingesting-stream) (accessed 2025-11-16):
> "Streaming import lets you make real-time updates to feature values. This method is useful when having the latest available data for online serving is a priority."

**Architecture:**
```
[Event Source]
    ├─ User activity
    ├─ Transaction data
    └─ IoT sensor readings
    ↓
[Pub/Sub Topic: feature-updates]
    ↓
[Dataflow Pipeline]
    ├─ Feature transformation
    ├─ Validation
    └─ Enrichment
    ↓
[Vertex AI Feature Store Streaming API]
    ↓
[Bigtable Online Store]
    ↓
Available for online serving (<10ms latency)
```

**Streaming ingestion example:**
```python
from google.cloud import aiplatform
from google.cloud import pubsub_v1
import json

def stream_feature_update(
    entity_id: str,
    features: dict
) -> None:
    """Stream single feature update to online store"""

    featurestore_online_service_client = (
        aiplatform.gapic.FeaturestoreOnlineServingServiceClient()
    )

    # Prepare feature values
    feature_values = []
    for feature_id, value in features.items():
        if isinstance(value, int):
            feature_value = aiplatform.gapic.FeatureValue(
                int64_value=value
            )
        elif isinstance(value, float):
            feature_value = aiplatform.gapic.FeatureValue(
                double_value=value
            )
        elif isinstance(value, str):
            feature_value = aiplatform.gapic.FeatureValue(
                string_value=value
            )

        feature_values.append({
            "feature_id": feature_id,
            "value": feature_value
        })

    # Write to online store
    write_feature_values_request = aiplatform.gapic.WriteFeatureValuesRequest(
        entity_type="projects/123/locations/us-central1/featurestores/prod/entityTypes/user",
        payloads=[
            aiplatform.gapic.WriteFeatureValuesPayload(
                entity_id=entity_id,
                feature_values=feature_values
            )
        ]
    )

    response = featurestore_online_service_client.write_feature_values(
        request=write_feature_values_request
    )

    print(f"Streamed features for entity {entity_id}")

# Pub/Sub consumer
def pubsub_callback(message):
    """Process feature update from Pub/Sub"""
    data = json.loads(message.data.decode('utf-8'))

    stream_feature_update(
        entity_id=data['entity_id'],
        features=data['features']
    )

    message.ack()

# Subscribe to feature update topic
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(
    'my-project', 'feature-updates-sub'
)

streaming_pull_future = subscriber.subscribe(
    subscription_path, callback=pubsub_callback
)

print("Listening for feature updates on Pub/Sub...")
streaming_pull_future.result()  # Block indefinitely
```

**Dataflow streaming pipeline (Apache Beam):**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class TransformFeatures(beam.DoFn):
    """Transform raw events into feature format"""

    def process(self, element):
        event_data = json.loads(element.decode('utf-8'))

        # Extract entity ID
        entity_id = event_data['user_id']

        # Compute features
        features = {
            'session_count': event_data.get('session_count', 0),
            'last_activity_timestamp': event_data['timestamp'],
            'device_type': event_data.get('device', 'unknown')
        }

        yield {
            'entity_id': entity_id,
            'features': features
        }

def run_streaming_pipeline():
    options = PipelineOptions([
        '--project=my-project',
        '--region=us-central1',
        '--runner=DataflowRunner',
        '--streaming',
        '--temp_location=gs://my-bucket/temp'
    ])

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(
                topic='projects/my-project/topics/user-events'
            )
            | 'Transform Features' >> beam.ParDo(TransformFeatures())
            | 'Write to Feature Store' >> beam.ParDo(WriteToFeatureStore())
        )

if __name__ == '__main__':
    run_streaming_pipeline()
```

**Use cases for streaming ingestion:**
- Real-time fraud detection (transaction features updated immediately)
- Recommendation systems (user interaction features)
- IoT anomaly detection (sensor readings)
- Gaming leaderboards (player statistics)

---

## Section 5: Point-in-Time Correctness (~100 lines)

### Training-Serving Skew Prevention

From [Towards Data Science - Point-in-Time Correctness](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1) (accessed 2025-11-16):
> "Feature Stores were built in large part to solve the point-in-time correctness problem. Given a timestamp for each prediction, a feature store can retrieve feature values as they existed at that specific point in time."

**The problem:**
```
Training time (January 1):
    User made purchase → Label = 1 (churned)
    Features used: lifetime_value = $500 (computed on January 1)

Prediction time (December 31 - looking back):
    Using latest features: lifetime_value = $2000 (computed on December 31)

❌ DATA LEAKAGE: Model trained on future information!
```

**Point-in-time correct solution:**
```sql
-- Feature Store automatically handles this
SELECT
    label_timestamp,
    user_id,

    -- Features AS OF the label timestamp
    (
        SELECT lifetime_value
        FROM user_features
        WHERE user_id = labels.user_id
            AND feature_timestamp <= labels.label_timestamp
        ORDER BY feature_timestamp DESC
        LIMIT 1
    ) AS lifetime_value_at_label_time

FROM labels
```

**Vertex AI Feature Store implementation:**

Feature Store automatically maintains:
1. **Feature timestamps**: When feature was computed
2. **Entity versions**: Multiple versions of same entity
3. **Temporal queries**: Retrieve features "as of" specific time

```python
def generate_point_in_time_dataset(
    entity_ids: list,
    label_timestamps: list,
    feature_ids: list
) -> pd.DataFrame:
    """
    Generate training dataset with point-in-time correctness

    For each (entity_id, label_timestamp) pair:
    - Retrieve feature values as they existed at label_timestamp
    - Prevent data leakage from future information
    """

    featurestore_client = aiplatform.gapic.FeaturestoreServiceClient()

    # Build batch read request
    entity_type_specs = [
        aiplatform.gapic.BatchReadFeatureValuesRequest.EntityTypeSpec(
            entity_type_id="user",
            feature_selector=aiplatform.gapic.FeatureSelector(
                id_matcher=aiplatform.gapic.IdMatcher(
                    ids=feature_ids
                )
            )
        )
    ]

    # CSV with entity_id and timestamp
    csv_read_instances = aiplatform.gapic.CsvSource(
        gcs_source=aiplatform.gapic.GcsSource(
            uris=["gs://my-bucket/entity_timestamps.csv"]
        )
    )

    batch_read_request = aiplatform.gapic.BatchReadFeatureValuesRequest(
        featurestore="projects/123/locations/us-central1/featurestores/prod",
        csv_read_instances=csv_read_instances,
        entity_type_specs=entity_type_specs,
        destination=aiplatform.gapic.FeatureValueDestination(
            bigquery_destination=aiplatform.gapic.BigQueryDestination(
                output_uri="bq://my-project.training_data.point_in_time_features"
            )
        )
    )

    # Execute batch read (point-in-time correct)
    operation = featurestore_client.batch_read_feature_values(
        request=batch_read_request
    )

    result = operation.result()  # Wait for completion

    print(f"Point-in-time correct dataset generated")
    return result
```

**Validation example:**
```python
# Check for data leakage
def validate_no_future_leakage(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> bool:
    """Ensure no features computed after label timestamp"""

    merged = features_df.merge(
        labels_df,
        on='entity_id'
    )

    # Check: feature_timestamp <= label_timestamp
    leakage_count = (
        merged['feature_timestamp'] > merged['label_timestamp']
    ).sum()

    if leakage_count > 0:
        print(f"⚠️  Data leakage detected: {leakage_count} rows")
        return False

    print("✓ Point-in-time correctness validated")
    return True
```

From [Databricks - Point-in-Time Feature Joins](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series) (accessed 2025-11-16):
> "This article describes how to use point-in-time correctness to create a training dataset that accurately reflects feature values as of the time a label was recorded, preventing data leakage."

---

## Section 6: Cost Analysis (~100 lines)

### Batch vs Online Inference Pricing

**Batch Prediction costs:**
```
Total Cost = (Compute cost × Duration) + (Storage I/O) + (Network egress)

Components:
- Machine type: n1-standard-4 = $0.19/hour
- GPU (optional): T4 = $0.35/hour
- Execution time: 2 hours
- Input data: 1 TB from BigQuery = $5.00
- Output data: 100 GB to GCS = $0.12

Example calculation:
Compute: ($0.19 + $0.35) × 2 hours × 10 workers = $10.80
Storage I/O: $5.00 + $0.12 = $5.12
Total: $15.92 for 1M predictions = $0.000016/prediction
```

**Online Prediction costs:**
```
Total Cost = (Endpoint deployment) + (Prediction requests) + (Autoscaling)

Components:
- Dedicated endpoint: n1-standard-4 = $0.19/hour × 24 hours = $4.56/day
- Prediction requests: $0.50 per 1000 predictions
- Autoscaling (if enabled): Additional replicas × hourly rate

Example calculation:
Base deployment: $4.56/day
Requests (100K/day): 100 × $0.50 = $50.00/day
Total: $54.56/day = $0.000546/prediction

Comparison:
Batch: $0.000016/prediction (342× cheaper)
Online: $0.000546/prediction (required for real-time)
```

**When to use each:**

**Batch Prediction:**
- ✓ Large datasets (>100K predictions)
- ✓ Non-time-sensitive (can wait hours)
- ✓ Periodic scoring (daily/weekly)
- ✓ Cost optimization priority

**Online Prediction:**
- ✓ Real-time requirements (<100ms)
- ✓ Interactive applications
- ✓ Unpredictable request patterns
- ✓ Latency priority over cost

### Feature Store Pricing

From [Stack Overflow - Vertex AI Feature Store vs BigQuery](https://stackoverflow.com/questions/74807026/vertex-ai-feature-store-vs-bigquery) (accessed 2025-11-16):
> "Vertex AI Feature Store and BigQuery, both can be used to store the features... Vertex AI Feature Store is designed for feature management, batch/online storage, and sharing, while BigQuery is for analysis and not storage."

**Feature Store cost components:**
```
Online Store Costs:
- Node hours: $0.35/hour per node
- Storage: $0.25/GB-month (Bigtable)
- Read requests: $0.10 per million reads
- Write requests: $0.50 per million writes

Offline Store Costs:
- BigQuery storage: $0.02/GB-month (active), $0.01/GB-month (long-term)
- BigQuery analysis: $5.00 per TB scanned
- BigQuery streaming inserts: $0.05 per 200 MB

Example monthly cost (medium deployment):
Online nodes: 2 nodes × $0.35/hour × 730 hours = $511
Online storage: 50 GB × $0.25 = $12.50
Read requests: 100M × $0.10 = $10.00
Write requests: 10M × $0.50 = $5.00

Offline storage: 500 GB × $0.02 = $10.00
Offline queries: 2 TB scanned × $5.00 = $10.00

Total: $558.50/month
```

**Cost optimization strategies:**

**1. Right-size online serving nodes:**
```python
# Minimal deployment (dev/staging)
online_serving_config = aiplatform_v1.Featurestore.OnlineServingConfig(
    fixed_node_count=1  # $255/month
)

# Production deployment (autoscaling)
online_serving_config = aiplatform_v1.Featurestore.OnlineServingConfig(
    scaling=aiplatform_v1.Featurestore.OnlineServingConfig.Scaling(
        min_node_count=2,
        max_node_count=10,
        cpu_utilization_target=70  # Scale at 70% CPU
    )
)
```

**2. Use BigQuery for batch-only features:**
```python
# Don't sync to online store if only used for training
feature_config = {
    "disable_online_serving": True,  # Save online storage costs
    "monitoring_config": None  # Disable monitoring if not needed
}
```

**3. Implement data retention policies:**
```sql
-- Delete old feature values (reduce storage costs)
DELETE FROM `project.featurestore.user_features`
WHERE feature_timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
```

**4. Optimize BigQuery queries:**
```sql
-- ❌ Expensive: Scan entire table
SELECT * FROM `project.featurestore.user_features`

-- ✓ Optimized: Partition and filter
SELECT user_id, age, lifetime_value
FROM `project.featurestore.user_features`
WHERE DATE(feature_timestamp) = '2024-11-16'  -- Partition filter
    AND user_id IN (SELECT user_id FROM active_users)  -- Reduce scan
```

From [Hopsworks - Feature Store vs Data Warehouse](https://www.hopsworks.ai/post/feature-store-vs-data-warehouse) (accessed 2025-11-16):
> "The offline feature store is typically required to efficiently serve and store large amounts of feature data, while the online feature store is optimized for low-latency access to the latest feature values."

---

## Section 7: arr-coc-0-1 Batch Inference for Dataset Evaluation (~100 lines)

### Production Workflow Integration

From [arr-coc-0-1 Vertex AI Setup](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/vertex_setup.py) (project codebase):

**Batch evaluation pipeline:**
```python
# arr-coc-0-1 batch prediction workflow
from google.cloud import aiplatform
from google.cloud import bigquery
import pandas as pd

def evaluate_relevance_allocation_batch(
    model_name: str,
    test_dataset_bq_table: str,
    output_table: str
) -> dict:
    """
    Batch evaluation of ARR-COC relevance allocation strategies

    Workflow:
    1. Load test images from BigQuery (GCS URIs + metadata)
    2. Run batch prediction with arr-coc model
    3. Analyze token allocation patterns
    4. Compare with baseline (uniform LOD)
    """

    # Step 1: Create batch prediction job
    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name="arr-coc-batch-eval",
        model_name=model_name,

        # Input: BigQuery table with image URIs and queries
        bigquery_source=f"bq://{test_dataset_bq_table}",

        # Output: Predictions back to BigQuery
        bigquery_destination_prefix=f"bq://{output_table}",

        # Configuration for vision model
        machine_type="n1-highmem-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,

        # arr-coc specific parameters
        model_parameters={
            "token_budget": 200,  # K=200 patches
            "min_lod": 64,  # Minimum tokens per patch
            "max_lod": 400,  # Maximum tokens per patch
            "relevance_mode": "vervaekean"  # Use 3 ways of knowing
        },

        # Batch settings
        batch_size=8,  # Process 8 images per batch
        starting_replica_count=5,
        max_replica_count=20
    )

    # Wait for completion
    batch_job.wait()

    # Step 2: Analyze results
    metrics = analyze_batch_predictions(output_table)

    return metrics

def analyze_batch_predictions(
    predictions_table: str
) -> dict:
    """Analyze token allocation patterns from batch predictions"""

    client = bigquery.Client()

    query = f"""
    WITH predictions AS (
        SELECT
            image_id,
            query_text,

            -- Extract token allocation from predictions
            JSON_EXTRACT_SCALAR(prediction, '$.total_tokens') AS total_tokens,
            JSON_EXTRACT_SCALAR(prediction, '$.num_patches') AS num_patches,
            JSON_EXTRACT(prediction, '$.patch_lods') AS patch_lod_distribution,

            -- Ground truth
            relevance_label,
            attention_map

        FROM `{predictions_table}`
    ),

    allocation_stats AS (
        SELECT
            AVG(CAST(total_tokens AS INT64)) AS avg_tokens_used,
            AVG(CAST(num_patches AS INT64)) AS avg_patches,

            -- Token efficiency: Correct predictions / tokens used
            SUM(CASE WHEN relevance_label = 'high' THEN 1 ELSE 0 END) /
                SUM(CAST(total_tokens AS INT64)) AS token_efficiency,

            -- LOD variance (measure of dynamic allocation)
            STDDEV(CAST(total_tokens AS INT64) / CAST(num_patches AS INT64))
                AS lod_variance

        FROM predictions
    )

    SELECT * FROM allocation_stats
    """

    results = client.query(query).to_dataframe()

    metrics = {
        'avg_tokens_used': results['avg_tokens_used'][0],
        'avg_patches': results['avg_patches'][0],
        'token_efficiency': results['token_efficiency'][0],
        'lod_variance': results['lod_variance'][0]
    }

    return metrics

def compare_with_baseline(
    arr_coc_predictions: str,
    uniform_lod_predictions: str
) -> pd.DataFrame:
    """Compare ARR-COC dynamic allocation vs uniform baseline"""

    client = bigquery.Client()

    query = f"""
    WITH arr_coc AS (
        SELECT
            image_id,
            CAST(JSON_EXTRACT_SCALAR(prediction, '$.accuracy') AS FLOAT64) AS accuracy,
            CAST(JSON_EXTRACT_SCALAR(prediction, '$.total_tokens') AS INT64) AS tokens
        FROM `{arr_coc_predictions}`
    ),

    baseline AS (
        SELECT
            image_id,
            CAST(JSON_EXTRACT_SCALAR(prediction, '$.accuracy') AS FLOAT64) AS accuracy,
            CAST(JSON_EXTRACT_SCALAR(prediction, '$.total_tokens') AS INT64) AS tokens
        FROM `{uniform_lod_predictions}`
    )

    SELECT
        'ARR-COC' AS model,
        AVG(accuracy) AS avg_accuracy,
        AVG(tokens) AS avg_tokens,
        AVG(accuracy) / AVG(tokens) * 1000 AS accuracy_per_1k_tokens
    FROM arr_coc

    UNION ALL

    SELECT
        'Uniform LOD' AS model,
        AVG(accuracy) AS avg_accuracy,
        AVG(tokens) AS avg_tokens,
        AVG(accuracy) / AVG(tokens) * 1000 AS accuracy_per_1k_tokens
    FROM baseline

    ORDER BY accuracy_per_1k_tokens DESC
    """

    comparison_df = client.query(query).to_dataframe()

    print("\n🔬 ARR-COC Batch Evaluation Results:")
    print(comparison_df.to_string(index=False))

    return comparison_df
```

**Feature Store integration for arr-coc:**
```python
def create_arr_coc_features():
    """Define features for relevance allocation experiments"""

    # Entity: Image
    # Features: Query-aware visual statistics

    features = {
        "image_entropy": "Information content (propositional knowing)",
        "saliency_map": "Attention landscape (perspectival knowing)",
        "query_similarity": "Query-content coupling (participatory knowing)",
        "compression_quality": "Learned LOD mapping (procedural knowing)"
    }

    featurestore_client = aiplatform.gapic.FeaturestoreServiceClient()

    # Create features for online serving
    for feature_id, description in features.items():
        feature = aiplatform_v1.Feature(
            value_type=aiplatform_v1.Feature.ValueType.DOUBLE,
            description=description
        )

        # Enable streaming updates (real-time relevance scores)
        create_feature_request = aiplatform_v1.CreateFeatureRequest(
            parent="projects/arr-coc/locations/us-central1/featurestores/prod/entityTypes/image",
            feature_id=feature_id,
            feature=feature
        )

        featurestore_client.create_feature(request=create_feature_request)

    print("ARR-COC features created in Feature Store")
```

**Cost estimate for arr-coc batch evaluation:**
```
Dataset: 10,000 test images
Model: arr-coc-0-1 (vision-language with dynamic LOD)

Batch Prediction:
- Machine: n1-highmem-8 + T4 GPU = $0.84/hour
- Duration: ~5 hours (10K images, batch_size=8)
- Workers: 5 starting, autoscale to 20
- Avg workers: ~10

Compute cost: $0.84/hour × 5 hours × 10 workers = $42.00
Storage I/O: BigQuery scan (1 GB) + write (500 MB) = $0.10
Total: $42.10 for 10K evaluations = $0.0042/image

Online inference equivalent (for comparison):
- Dedicated endpoint: $4.56/day minimum
- 10K predictions would take ~2.8 hours at 1 req/s
- Cost: $4.56 + ($0.50 × 10) = $9.56

Batch is 4.4× cheaper for this evaluation workload!
```

---

## Sources

**Source Documents:**
- [inference-optimization/02-triton-inference-server.md](../karpathy/inference-optimization/02-triton-inference-server.md) - Batch optimization patterns
- [vertex-ai-production/01-inference-serving-optimization.md](../karpathy/vertex-ai-production/01-inference-serving-optimization.md) - Vertex AI serving architecture

**Web Research:**

**Batch Prediction:**
- [Vertex AI Batch Predictions Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction) - Official Google Cloud documentation (accessed 2025-11-16)
- [Medium - Scaling Batch Predictions with Gemini](https://medium.com/@ikaromoribayashi/scaling-artificial-intelligence-a-complete-guide-to-batch-predictions-with-gemini-vertex-ai-and-51e8c92a9578) - Complete workflow guide (accessed 2025-11-16)
- [Vertex AI Get Batch Predictions](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) - API reference (accessed 2025-11-16)

**Feature Store:**
- [Vertex AI Feature Store Overview](https://docs.cloud.google.com/vertex-ai/docs/featurestore/latest/overview) - Architecture documentation (accessed 2025-11-16)
- [Vertex AI Feature Store Data Model](https://docs.cloud.google.com/vertex-ai/docs/featurestore/concepts) - Entity types and features (accessed 2025-11-16)
- [Medium - Centralizing ML Features](https://medium.com/google-cloud/centralizing-ml-features-through-feature-store-in-google-cloud-vertex-ai-300f5b37b5d8) - Production patterns (accessed 2025-11-16)
- [Medium - Vertex AI Feature Store Features and Advantages](https://medium.com/@ajayverma23/exploring-vertex-ai-feature-store-features-and-advantages-12014ead55d3) - Feature explanation (accessed 2025-11-16)

**Online vs Offline Serving:**
- [Stack Overflow - Vertex AI Feature Store vs BigQuery](https://stackoverflow.com/questions/74807026/vertex-ai-feature-store-vs-bigquery) - Use case comparison (accessed 2025-11-16)
- [Hopsworks - Feature Store vs Data Warehouse](https://www.hopsworks.ai/post/feature-store-vs-data-warehouse) - Architecture patterns (accessed 2025-11-16)
- [DragonflyDB - Feature Store Architecture](https://www.dragonflydb.io/blog/feature-store-architecture-and-storage) - Storage layer deep dive (accessed 2025-11-16)

**Streaming:**
- [Vertex AI Streaming Import](https://docs.cloud.google.com/vertex-ai/docs/featurestore/ingesting-stream) - Real-time updates (accessed 2025-11-16)
- [Google Cloud - Real-Time AI with Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai) - Streaming ingestion announcement (accessed 2025-11-16)

**Point-in-Time Correctness:**
- [Towards Data Science - Point-in-Time Correctness](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1) - Concept explanation (accessed 2025-11-16)
- [Databricks - Point-in-Time Feature Joins](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series) - Implementation patterns (accessed 2025-11-16)

**Additional References:**
- [arr-coc-0-1 Training CLI](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/cli.py) - Project integration examples
- [Google Cloud Skills Boost - Vertex AI Feature Store](https://www.cloudskillsboost.google/paths/17/course_templates/584/video/457053) - Video tutorials
