# Vertex AI Data Integration: Cloud Storage, GCS FUSE, and Datasets

**Complete guide to data management for production ML training on Vertex AI**

This document covers Cloud Storage integration, GCS FUSE for file-like access, Vertex AI Managed Datasets, and data pipeline patterns for scalable ML training.

---

## Section 1: Cloud Storage Integration (~170 lines)

### GCS Bucket Creation and Organization

**Bucket structure for ML workflows:**
```bash
# Create bucket for ML project
gsutil mb -p PROJECT_ID -c STANDARD -l us-central1 gs://ml-training-data

# Recommended folder structure
gs://ml-training-data/
├── raw/                    # Original datasets
├── processed/              # Preprocessed training data
├── tfrecords/              # TFRecord format
├── checkpoints/            # Training checkpoints
├── models/                 # Exported models
└── artifacts/              # Pipeline artifacts
```

**Storage class selection:**
- **Standard**: Frequently accessed data (active training datasets)
- **Nearline**: Monthly access (archived experiments)
- **Coldline**: Quarterly access (compliance/audit data)
- **Archive**: Yearly access (long-term retention)

From [Cloud Storage documentation](https://cloud.google.com/storage/docs) (accessed 2025-01-31):
- Standard storage: Best for hot data and compute workloads
- Automatic class transitions via lifecycle policies

### IAM Permissions for Bucket Access

**Service account permissions for training jobs:**
```bash
# Grant storage access to Vertex AI service account
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Minimum permissions for read-only training data
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Write permissions for checkpoints and outputs
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator"
```

**Bucket-level permissions (more granular):**
```bash
# Grant access to specific bucket only
gsutil iam ch serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://ml-training-data

# Fine-grained folder permissions
gsutil iam ch serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com:objectAdmin \
    gs://ml-training-data/checkpoints/
```

### gsutil for Data Transfer

**Efficient data upload:**
```bash
# Parallel composite uploads for large files (>150MB)
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M \
    cp -r local_dataset/ gs://ml-training-data/raw/

# Resume interrupted uploads
gsutil -m cp -r -c local_dataset/ gs://ml-training-data/raw/

# Synchronize directories
gsutil -m rsync -r -d local_dataset/ gs://ml-training-data/raw/

# Copy with checksums (ensure data integrity)
gsutil cp -c file.tar gs://ml-training-data/raw/
```

From [Cloud Storage best practices](https://cloud.google.com/storage/docs/cloud-storage-fuse/performance) (accessed 2025-01-31):
- Parallel composite uploads can provide up to 10x speedup for large files
- Use `-m` flag for parallel operations across multiple files

**Download optimization:**
```bash
# Parallel downloads
gsutil -m cp -r gs://ml-training-data/processed/ ./local_data/

# Sliced object downloads (for very large files)
gsutil -o GSUtil:sliced_object_download_threshold=150M \
    cp gs://ml-training-data/large_file.tar ./
```

### Performance Optimization

**Parallel composite uploads:**
- Enabled automatically for files >150MB
- Splits file into 32 components for parallel upload
- Requires `storage.objects.compose` permission

**Request rate limits:**
- 5000 write requests/second per bucket
- 5000 read requests/second per bucket
- Use exponential backoff for rate limit errors

**Network optimization:**
```python
# Python: Use storage client with connection pooling
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('ml-training-data')

# Batch operations
blobs = bucket.list_blobs(prefix='processed/')
for blob in blobs:
    blob.download_to_filename(f'./data/{blob.name}')
```

### Cost Optimization

**Lifecycle policies for automatic transitions:**
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
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesPrefix": ["checkpoints/"]
        }
      }
    ]
  }
}
```

Apply lifecycle policy:
```bash
gsutil lifecycle set lifecycle.json gs://ml-training-data
```

**Storage analytics:**
```bash
# Enable storage insights
gcloud storage insights create-report \
    --location=us-central1 \
    --bucket=ml-training-data \
    --report-name=ml-storage-analysis
```

### Data Encryption

**Customer-managed encryption keys (CMEK):**
```bash
# Create Cloud KMS key
gcloud kms keyrings create ml-keyring --location=us-central1
gcloud kms keys create ml-encryption-key \
    --keyring=ml-keyring \
    --location=us-central1 \
    --purpose=encryption

# Upload with CMEK
gsutil -o "GSUtil:encryption_key=projects/PROJECT_ID/locations/us-central1/keyRings/ml-keyring/cryptoKeys/ml-encryption-key" \
    cp dataset.tar gs://ml-training-data/encrypted/
```

**Customer-supplied encryption keys (CSEK):**
```python
# Python: Upload with CSEK
from google.cloud import storage
import base64

# Generate 256-bit key
encryption_key = base64.b64encode(os.urandom(32))

bucket = storage.Client().bucket('ml-training-data')
blob = bucket.blob('encrypted/data.tar')
blob.upload_from_filename('data.tar', encryption_key=encryption_key)
```

---

## Section 2: GCS FUSE for File-Like Access (~170 lines)

### GCS FUSE Overview

From [GCS FUSE optimization guide](https://cloud.google.com/architecture/optimize-ai-ml-workloads-cloud-storage-fuse) (accessed 2025-01-31):

**What is GCS FUSE?**
- User-space file system that mounts GCS buckets as local directories
- Allows standard POSIX file operations (read, write, list)
- Optimized for AI/ML workloads with file caching and parallel downloads

**Key features for ML:**
- File cache with parallel downloads (up to 9x faster)
- Streaming writes for checkpoints
- Memory-mapped file support
- Read-ahead prefetching

### Mounting GCS Buckets in Vertex AI

**Automatic mounting in Custom Jobs:**

From [Vertex AI release notes](https://docs.cloud.google.com/vertex-ai/docs/core-release-notes) (accessed 2025-01-31):
- Vertex AI Custom Jobs now support automatic GCS FUSE mounting
- Access buckets via `/gcs/bucket-name/` path
- No manual mounting required

**Manual GCS FUSE installation:**
```bash
# Install GCS FUSE in container
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y gcsfuse

# Mount bucket
mkdir -p /mnt/training-data
gcsfuse ml-training-data /mnt/training-data
```

**Configuration file for optimized ML workloads:**
```yaml
# config.yaml
file-cache:
  enable-cfs: true  # Enable file cache
  max-size-mb: 100000  # 100GB cache
  cache-file-for-range-read: true
  enable-parallel-downloads: true
  parallel-downloads-per-file: 16
  download-chunk-size-mb: 50

metadata-cache:
  ttl-secs: 3600
  type-cache-max-size-mb: 32
  stat-cache-max-size-mb: 128

file-system:
  kernel-list-cache-ttl-secs: 3600

logging:
  severity: warning
  log-rotate:
    max-file-size-mb: 512
```

Mount with config:
```bash
gcsfuse --config-file config.yaml ml-training-data /mnt/training-data
```

### File Caching and Parallel Downloads

From [GCS FUSE file caching documentation](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/file-caching) (accessed 2025-01-31):

**File cache benefits:**
- Reduces repeat downloads from GCS
- Essential for multi-epoch training
- Automatically manages cache size with LRU eviction
- Up to 9x faster read performance

**Parallel downloads:**
- Splits large files into chunks
- Downloads chunks simultaneously (default: 16 workers)
- Reduces time-to-first-byte for large models
- Optimal for files >50MB

**Python usage example:**
```python
import os
import torch

# Data cached automatically on first read
cache_dir = '/gcs/ml-training-data/processed/'
dataset = torch.load(f'{cache_dir}/imagenet_train.pt')

# Second epoch reads from cache (instant)
for epoch in range(10):
    for batch in dataset:
        # Training loop
        pass
```

**Cache performance tuning:**
```yaml
file-cache:
  # Increase cache size for large datasets
  max-size-mb: 500000  # 500GB

  # Adjust for file size distribution
  download-chunk-size-mb: 100  # Larger chunks for big files
  parallel-downloads-per-file: 32  # More workers for bandwidth
```

### Performance Best Practices

**Directory-specific mounts (recommended):**
```bash
# Mount specific prefix instead of entire bucket
gcsfuse --only-dir processed ml-training-data /mnt/training-data
```

From [GCS FUSE performance tuning](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/performance) (accessed 2025-01-31):
- Directory-specific mounts reduce metadata operations
- Improves listing performance by 10-100x
- Reduces memory usage

**Read-ahead optimization:**
```yaml
file-cache:
  enable-cfs: true
  # Enable read-ahead for sequential access
  read-ahead-size-mb: 100
```

**Memory optimization:**
```yaml
# For memory-constrained environments
file-cache:
  max-size-mb: 50000
metadata-cache:
  stat-cache-max-size-mb: 64
```

### GCS FUSE in Kubernetes (GKE)

**Using GCS FUSE CSI driver:**

From [GKE GCS FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-perf) (accessed 2025-01-31):

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: gcs-fuse-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: ml-training-data
    volumeAttributes:
      bucketName: ml-training-data
      mountOptions: "implicit-dirs,file-cache-max-size-mb=100000"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gcs-fuse-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  volumeName: gcs-fuse-pv
```

**Pod configuration:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  containers:
  - name: trainer
    image: gcr.io/project-id/trainer:latest
    volumeMounts:
    - name: gcs-data
      mountPath: /data
      readOnly: false
  volumes:
  - name: gcs-data
    persistentVolumeClaim:
      claimName: gcs-fuse-pvc
```

### Common Pitfalls and Solutions

**Issue: Slow writes**
- **Cause**: GCS FUSE writes entire objects
- **Solution**: Use streaming writes or buffer locally

```python
# Buffer checkpoints locally, then upload
checkpoint_dir = '/tmp/checkpoints'  # Local
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), f'{checkpoint_dir}/checkpoint_{step}.pt')

# Periodic upload to GCS
if step % 1000 == 0:
    os.system(f'gsutil -m cp -r {checkpoint_dir}/* gs://ml-training-data/checkpoints/')
```

**Issue: High latency for small files**
- **Solution**: Consolidate into TFRecords or tar archives

**Issue: Out of memory with large cache**
- **Solution**: Reduce cache size or use streaming

```yaml
file-cache:
  max-size-mb: 10000  # Reduce if OOM
```

---

## Section 3: Vertex AI Managed Datasets (~170 lines)

### Managed Dataset Types

From [Vertex AI Datasets overview](https://docs.cloud.google.com/vertex-ai/docs/datasets/overview) (accessed 2025-01-31):

**Supported dataset types:**
1. **Tabular**: CSV, BigQuery
2. **Image**: Classification, object detection, segmentation
3. **Video**: Classification, action recognition, tracking
4. **Text**: Classification, entity extraction, sentiment

**Benefits of managed datasets:**
- Centralized data catalog
- Automatic versioning and lineage
- Data labeling integration
- Feature Store compatibility
- W&B Artifact integration

### Creating Managed Datasets

**Tabular dataset from CSV:**
```python
from google.cloud import aiplatform

aiplatform.init(project='PROJECT_ID', location='us-central1')

dataset = aiplatform.TabularDataset.create(
    display_name='customer_churn_data',
    gcs_source='gs://ml-training-data/processed/churn_data.csv',
    labels={'env': 'production', 'version': 'v1'}
)

print(f'Dataset resource name: {dataset.resource_name}')
```

**Image dataset with annotations:**
```python
# Import format: gs://bucket/images/*.jpg with CSV annotations
image_dataset = aiplatform.ImageDataset.create(
    display_name='imagenet_subset',
    gcs_source='gs://ml-training-data/images/imagenet_import.csv',
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
)
```

**CSV format for image classification:**
```csv
gs://ml-training-data/images/cat_001.jpg,cat
gs://ml-training-data/images/dog_001.jpg,dog
gs://ml-training-data/images/cat_002.jpg,cat
```

### Dataset Versioning and Lineage

From [Manage dataset versions](https://docs.cloud.google.com/vertex-ai/docs/datasets/manage-dataset-versions) (accessed 2025-01-31):

**Create dataset versions:**
```python
# Create initial version
dataset_v1 = aiplatform.TabularDataset.create(
    display_name='customer_data_v1',
    gcs_source='gs://ml-training-data/customer_jan2024.csv'
)

# Create new version with updated data
dataset_v2 = dataset_v1.create_version(
    gcs_source='gs://ml-training-data/customer_feb2024.csv',
    version_description='February 2024 data update'
)

# List all versions
versions = dataset_v1.list_versions()
for version in versions:
    print(f'{version.version_id}: {version.version_description}')
```

**Track dataset lineage:**
```python
# Vertex AI automatically tracks:
# - Source GCS paths
# - Creation timestamps
# - Models trained on each version
# - Pipeline runs using dataset

# Query lineage
lineage_client = aiplatform.gapic.MetadataServiceClient()
lineage = lineage_client.query_execution_inputs_and_outputs(
    execution=model.resource_name
)

for artifact in lineage.artifacts:
    if artifact.schema_title == 'Dataset':
        print(f'Trained on dataset: {artifact.display_name}')
```

### Data Labeling Integration

**Create labeling job:**
```python
from google.cloud import aiplatform

# Unlabeled image dataset
dataset = aiplatform.ImageDataset.create(
    display_name='unlabeled_images',
    gcs_source='gs://ml-training-data/raw/images/*.jpg'
)

# Create labeling job
labeling_job = aiplatform.DataLabelingJob.create(
    display_name='image_classification_labeling',
    datasets=[dataset],
    instruction_uri='gs://ml-training-data/labeling_instructions.pdf',
    inputs_schema_uri=aiplatform.schema.datalabelingjob.inputs.image_classification,
    annotation_labels=['cat', 'dog', 'bird'],
    specialist_pools=['projects/PROJECT_ID/locations/us-central1/specialistPools/POOL_ID']
)

# Monitor labeling progress
labeling_job.wait()
labeled_dataset = labeling_job.get_labeled_dataset()
```

### Feature Store Integration

**Register dataset with Feature Store:**
```python
from google.cloud.aiplatform import featurestore

# Create feature store
fs = featurestore.Featurestore.create(
    featurestore_id='ml_features',
    online_serving_config={'fixed_node_count': 1}
)

# Create entity type (e.g., 'user')
entity_type = fs.create_entity_type(
    entity_type_id='user',
    description='User features'
)

# Import features from dataset
entity_type.ingest_from_bq(
    feature_ids=['age', 'country', 'activity_score'],
    bq_source_uri='bq://PROJECT_ID.dataset.user_features',
    entity_id_field='user_id'
)
```

### Dataset Access in Custom Jobs

**Using managed datasets in training:**

From [Using managed datasets](https://docs.cloud.google.com/vertex-ai/docs/training/using-managed-datasets) (accessed 2025-01-31):

```python
# training_script.py
import os
from google.cloud import aiplatform

# Access via environment variable
dataset_uri = os.environ['AIP_TRAINING_DATA_URI']

# For tabular datasets, Vertex AI provides CSV path
import pandas as pd
df = pd.read_csv(dataset_uri)

# For image datasets, get annotation JSON
import json
with open(dataset_uri, 'r') as f:
    annotations = json.load(f)

for annotation in annotations['annotations']:
    image_path = annotation['imageGcsUri']
    label = annotation['displayName']
    # Load and process image
```

**Custom Job with managed dataset:**
```python
from google.cloud import aiplatform

# Reference managed dataset
dataset = aiplatform.TabularDataset('projects/PROJECT_ID/locations/us-central1/datasets/DATASET_ID')

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='training_with_managed_dataset',
    container_uri='gcr.io/project-id/trainer:latest'
)

# Run with dataset
model = job.run(
    dataset=dataset,
    model_display_name='trained_model',
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1
)
```

### W&B Artifact Integration

**Log Vertex AI dataset to W&B:**
```python
import wandb
from google.cloud import aiplatform

# Initialize W&B run
run = wandb.init(project='ml-training', job_type='train')

# Get Vertex AI dataset
dataset = aiplatform.TabularDataset('projects/PROJECT_ID/locations/us-central1/datasets/DATASET_ID')

# Create W&B artifact
artifact = wandb.Artifact(
    name=f'vertex_dataset_{dataset.display_name}',
    type='dataset',
    metadata={
        'vertex_ai_id': dataset.resource_name,
        'gcs_source': dataset.gcs_source,
        'version': dataset.version_id
    }
)

# Add reference to GCS data
artifact.add_reference(dataset.gcs_source, name='training_data')
run.log_artifact(artifact)
```

---

## Section 4: Data Pipeline Patterns (~160 lines)

### Training Data Preparation Jobs

**Multi-stage preprocessing pipeline:**
```python
from google.cloud import aiplatform
from kfp import dsl
from kfp.v2 import compiler

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-storage', 'pandas']
)
def preprocess_raw_data(
    input_uri: str,
    output_uri: str
):
    import pandas as pd
    from google.cloud import storage

    # Download from GCS
    df = pd.read_csv(input_uri)

    # Preprocessing steps
    df = df.dropna()
    df = df[df['value'] > 0]

    # Upload processed data
    df.to_csv(output_uri, index=False)

@dsl.component(
    base_image='gcr.io/project-id/tfrecord-converter:latest'
)
def convert_to_tfrecords(
    input_uri: str,
    output_uri: str
):
    import tensorflow as tf
    import pandas as pd

    df = pd.read_csv(input_uri)

    with tf.io.TFRecordWriter(output_uri) as writer:
        for _, row in df.iterrows():
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['image']])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['label']]))
                    }
                )
            )
            writer.write(example.SerializeToString())

@dsl.pipeline(name='data-preparation-pipeline')
def data_prep_pipeline(
    raw_data: str = 'gs://ml-training-data/raw/data.csv',
    processed_data: str = 'gs://ml-training-data/processed/data.csv',
    tfrecords: str = 'gs://ml-training-data/tfrecords/data.tfrecord'
):
    preprocess_task = preprocess_raw_data(
        input_uri=raw_data,
        output_uri=processed_data
    )

    convert_task = convert_to_tfrecords(
        input_uri=preprocess_task.outputs['output_uri'],
        output_uri=tfrecords
    )

# Compile and run
compiler.Compiler().compile(data_prep_pipeline, 'pipeline.json')

job = aiplatform.PipelineJob(
    display_name='data-preparation',
    template_path='pipeline.json',
    enable_caching=True
)
job.run()
```

### Checkpoints and Intermediate Artifacts

**Checkpoint strategy for fault tolerance:**
```python
# training_script.py
import torch
from google.cloud import storage

class GCSCheckpointManager:
    def __init__(self, bucket_name, checkpoint_dir):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.checkpoint_dir = checkpoint_dir
        self.local_dir = '/tmp/checkpoints'
        os.makedirs(self.local_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step):
        # Save locally first (fast)
        local_path = f'{self.local_dir}/checkpoint_epoch{epoch}_step{step}.pt'
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, local_path)

        # Upload to GCS asynchronously
        gcs_path = f'{self.checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt'
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

        print(f'Checkpoint saved: gs://{self.bucket.name}/{gcs_path}')

    def load_latest_checkpoint(self, model, optimizer):
        # List checkpoints
        blobs = self.bucket.list_blobs(prefix=self.checkpoint_dir)
        checkpoints = sorted([b for b in blobs if 'checkpoint' in b.name])

        if not checkpoints:
            return 0, 0

        # Download latest
        latest = checkpoints[-1]
        local_path = f'{self.local_dir}/latest.pt'
        latest.download_to_filename(local_path)

        # Load state
        checkpoint = torch.load(local_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['step']

# Usage in training loop
checkpoint_mgr = GCSCheckpointManager('ml-training-data', 'checkpoints/experiment_1')

start_epoch, start_step = checkpoint_mgr.load_latest_checkpoint(model, optimizer)

for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(dataloader, start=start_step):
        # Training step
        loss = train_step(model, batch)

        # Save checkpoint every 1000 steps
        if step % 1000 == 0:
            checkpoint_mgr.save_checkpoint(model, optimizer, epoch, step)
```

### Dataset Snapshots for Reproducibility

**Create immutable dataset snapshots:**
```python
from datetime import datetime
from google.cloud import storage

def create_dataset_snapshot(
    source_bucket: str,
    source_prefix: str,
    snapshot_bucket: str,
    experiment_id: str
):
    """
    Create immutable snapshot of training data
    """
    client = storage.Client()
    source = client.bucket(source_bucket)
    target = client.bucket(snapshot_bucket)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_prefix = f'snapshots/{experiment_id}_{timestamp}/'

    # Copy all files to snapshot location
    blobs = source.list_blobs(prefix=source_prefix)
    for blob in blobs:
        target_name = snapshot_prefix + blob.name[len(source_prefix):]
        source.copy_blob(blob, target, target_name)

    snapshot_uri = f'gs://{snapshot_bucket}/{snapshot_prefix}'
    print(f'Dataset snapshot created: {snapshot_uri}')
    return snapshot_uri

# Use in training pipeline
snapshot_uri = create_dataset_snapshot(
    source_bucket='ml-training-data',
    source_prefix='processed/imagenet/',
    snapshot_bucket='ml-snapshots',
    experiment_id='resnet50_v2'
)

# Train using snapshot (immutable)
train_model(data_uri=snapshot_uri)
```

### Large-Scale Data Loading

**TFRecord with sharding:**
```python
import tensorflow as tf

def create_tfrecord_dataset(
    gcs_pattern: str,
    batch_size: int = 32,
    shuffle_buffer: int = 10000
):
    """
    Efficient loading from sharded TFRecords in GCS
    """
    # Auto-discover all shards
    files = tf.io.gfile.glob(gcs_pattern)

    # Create dataset with parallel reads
    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=tf.data.AUTOTUNE
    )

    # Parse and preprocess
    def parse_example(serialized):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(serialized, features)
        image = tf.io.decode_jpeg(example['image'])
        label = example['label']
        return image, label

    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Usage
dataset = create_tfrecord_dataset(
    gcs_pattern='gs://ml-training-data/tfrecords/imagenet_*.tfrecord',
    batch_size=256
)

for images, labels in dataset:
    # Training step
    pass
```

**Parquet for tabular data:**
```python
import pyarrow.parquet as pq
import pyarrow.fs as fs

def load_parquet_dataset(
    gcs_path: str,
    columns: list = None,
    filters: list = None
):
    """
    Efficient Parquet loading from GCS
    """
    gcs_fs = fs.GcsFileSystem()

    # Read with column pruning and predicate pushdown
    table = pq.read_table(
        gcs_path,
        columns=columns,
        filters=filters,
        filesystem=gcs_fs,
        use_threads=True
    )

    return table.to_pandas()

# Usage
df = load_parquet_dataset(
    gcs_path='gs://ml-training-data/processed/user_features.parquet',
    columns=['user_id', 'age', 'country'],
    filters=[('age', '>', 18)]
)
```

### Streaming Data for Real-Time Training

**Pub/Sub to GCS pipeline:**
```python
from google.cloud import pubsub_v1, storage
import json

def stream_to_gcs(
    project_id: str,
    subscription_id: str,
    bucket_name: str,
    buffer_size: int = 1000
):
    """
    Stream data from Pub/Sub to GCS in batches
    """
    subscriber = pubsub_v1.SubscriberClient()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    buffer = []

    def callback(message):
        nonlocal buffer
        data = json.loads(message.data)
        buffer.append(data)
        message.ack()

        # Flush buffer to GCS
        if len(buffer) >= buffer_size:
            timestamp = int(time.time())
            blob = bucket.blob(f'streaming/batch_{timestamp}.json')
            blob.upload_from_string(json.dumps(buffer))
            buffer = []

    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()

# Usage in training pipeline
stream_to_gcs(
    project_id='PROJECT_ID',
    subscription_id='training-data-stream',
    bucket_name='ml-training-data'
)
```

### Complete Data Pipeline Example

**End-to-end pipeline with Vertex AI Pipelines:**
```python
from kfp.v2 import dsl, compiler
from google.cloud import aiplatform

@dsl.component(base_image='python:3.9')
def download_raw_data(output_path: dsl.Output[dsl.Dataset]):
    # Download from external source
    import urllib.request
    urllib.request.urlretrieve('https://example.com/data.csv', output_path.path)

@dsl.component(base_image='gcr.io/project-id/preprocessor:latest')
def preprocess_data(
    input_data: dsl.Input[dsl.Dataset],
    output_data: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    df = pd.read_csv(input_data.path)
    # Preprocessing
    df = df.dropna()
    df.to_csv(output_data.path, index=False)

@dsl.component(base_image='gcr.io/project-id/trainer:latest')
def train_model(
    training_data: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    epochs: int = 10
):
    import torch
    # Training code
    model_artifact = train(training_data.path, epochs)
    torch.save(model_artifact, model.path)

@dsl.pipeline(name='complete-ml-pipeline')
def ml_pipeline(epochs: int = 10):
    download_op = download_raw_data()
    preprocess_op = preprocess_data(input_data=download_op.outputs['output_path'])
    train_op = train_model(
        training_data=preprocess_op.outputs['output_data'],
        epochs=epochs
    )

# Compile and run
compiler.Compiler().compile(ml_pipeline, 'pipeline.json')

job = aiplatform.PipelineJob(
    display_name='complete-ml-pipeline',
    template_path='pipeline.json',
    parameter_values={'epochs': 20}
)
job.run()
```

---

## Sources

**Google Cloud Documentation:**
- [Cloud Storage documentation](https://cloud.google.com/storage/docs) - Bucket management and access patterns
- [GCS FUSE performance tuning](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/performance) - File caching and optimization
- [GCS FUSE optimization for AI/ML](https://cloud.google.com/architecture/optimize-ai-ml-workloads-cloud-storage-fuse) - Parallel downloads and caching strategies
- [GCS FUSE file caching](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/file-caching) - Cache configuration and performance
- [Vertex AI Datasets overview](https://docs.cloud.google.com/vertex-ai/docs/datasets/overview) - Managed datasets capabilities
- [Using managed datasets](https://docs.cloud.google.com/vertex-ai/docs/training/using-managed-datasets) - Dataset access in training jobs
- [Manage dataset versions](https://docs.cloud.google.com/vertex-ai/docs/datasets/manage-dataset-versions) - Versioning and lineage tracking
- [Vertex AI release notes](https://docs.cloud.google.com/vertex-ai/docs/core-release-notes) - GCS FUSE automatic mounting
- [GKE GCS FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-perf) - Kubernetes integration

**Web Research:**
- [Managing ML Datasets with Vertex AI](https://medium.com/@devashish_m/managing-machine-learning-datasets-with-vertex-ai-a-complete-guide-4e0bfef4d6c6) - Medium article (accessed 2025-01-31)
- [Using GCS FUSE in Read-Heavy Workloads](https://medium.com/@saifaliunity/using-gcs-fuse-in-read-heavy-model-workloads-what-breaks-and-what-works-545a6c3ec1ea) - Medium article (accessed 2025-01-31)
- [GCS FUSE GitHub Repository](https://github.com/GoogleCloudPlatform/gcsfuse) - Official implementation and features

**Community Resources:**
- Reddit discussions on GCS dataset versioning best practices (accessed 2025-01-31)
- Stack Overflow: Dataset versioning in Vertex AI (accessed 2025-01-31)
- Google Cloud Community forums on data access patterns (accessed 2025-01-31)

---

*This document provides production-ready patterns for data integration on Vertex AI, covering Cloud Storage optimization, GCS FUSE for efficient file access, managed datasets for centralized management, and scalable data pipeline patterns for enterprise ML training workflows.*
