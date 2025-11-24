# Vertex AI Datasets, Data Labeling, and AutoML Integration

**Complete guide to managed datasets, human labeling services, and seamless AutoML training on Vertex AI**

This document covers Vertex AI Managed Datasets, Data Labeling Service for human annotation, AutoML integration workflows, dataset versioning, cost optimization strategies, and arr-coc-0-1 dataset preparation patterns for production ML pipelines.

---

## Section 1: Vertex AI Dataset Types (~100 lines)

### Overview of Managed Datasets

From [Managing Machine Learning Datasets with Vertex AI](https://medium.com/@devashish_m/managing-machine-learning-datasets-with-vertex-ai-a-complete-guide-4e0bfef4d6c6) (accessed 2025-11-16):

**Managed Datasets** are centralized repositories within Vertex AI that provide:
- Unified data catalog for all ML projects
- Automatic versioning and lineage tracking
- Integrated labeling and annotation tools
- Train/validation/test splitting automation
- Direct integration with AutoML and Custom Training
- Cross-team collaboration with IAM controls

**Key benefits:**
- Centralized data management across models
- Compare model performance using same dataset
- Track data lineage for governance and compliance
- Human labeling task management
- Automatic statistics generation

### ImageDataset

**Supported formats:**
- PNG, JPEG, GIF, BMP, ICO
- Output: Same formats + TIFF, WEBP
- Base64 encoding required

**Use cases:**
1. **Single-label classification** - One category per image (e.g., flower species)
2. **Multi-label classification** - Multiple categories per image (e.g., image tags)
3. **Object detection** - Bounding boxes around objects
4. **Image segmentation** - Pixel-level classification

**JSON schema for classification:**
```json
{
  "imageGcsUri": "gs://bucket/filename.ext",
  "classificationAnnotation": {
    "displayName": "LABEL",
    "annotationResourceLabels": {
      "aiplatform.googleapis.com/annotation_set_name": "displayName",
      "env": "prod"
    }
  },
  "dataItemResourceLabels": {
    "aiplatform.googleapis.com/ml_use": "training"
  }
}
```

**Object detection schema:**
```json
{
  "imageGcsUri": "gs://bucket/filename.ext",
  "objectAnnotations": [
    {
      "boundingBox": {
        "normalizedVertices": [
          {"x": 0.1, "y": 0.2},
          {"x": 0.3, "y": 0.2},
          {"x": 0.3, "y": 0.4},
          {"x": 0.1, "y": 0.4}
        ]
      },
      "displayName": "LABEL"
    }
  ]
}
```

**Best practices:**
- Minimum 1,000 images for training
- Optimal image size: 512×512 to 1024×1024 pixels
- Balance dataset: 100x ratio between most/least frequent labels
- Include "None_of_the_above" class for edge cases
- Use real-world images matching production use case

### TabularDataset

**Supported sources:**
- CSV files (max 10GB per file, 100GB total)
- BigQuery tables: `bq://PROJECT_ID.DATASET_ID.TABLE_ID`

**Use cases:**
1. **Classification** - Predict categorical outcomes (e.g., customer churn)
2. **Regression** - Predict continuous values (e.g., house prices)
3. **Forecasting** - Time series predictions (e.g., sales forecasting)

**Requirements:**
- **Maximum size:** 100 GB
- **Columns:** 2-1,000 (including target + features)
- **Rows:** 1,000 to 100,000,000
- **Target column:** No null values allowed
- **Column names:** Alphanumeric + underscore, cannot start with `_`

**Data types:**
- **Categorical:** 2-500 distinct values
- **Numerical:** No restrictions

**Data split:**
- Default: 80% train, 10% validation, 10% test (random)
- Custom: Add `ml_use` column with values: `training`, `validation`, `test`
- Time series: Chronological split recommended

**CSV format:**
```csv
feature1,feature2,feature3,target,ml_use
value1,value2,value3,label1,training
value4,value5,value6,label2,validation
```

**Weights column:**
- Range: [0, 10000]
- Emphasize important samples
- Default: All samples weighted equally

### TextDataset

From [Managing Machine Learning Datasets with Vertex AI](https://medium.com/@devashish_m/managing-machine-learning-datasets-with-vertex-ai-a-complete-guide-4e0bfef4d6c6) (accessed 2025-11-16):

**Use cases:**
1. **Classification** - Categorize documents (e.g., genre, topic)
2. **Entity extraction** - Extract named entities (e.g., names, locations)
3. **Sentiment analysis** - Determine sentiment (positive, neutral, negative)

**Classification requirements:**
- **Minimum documents:** 20
- **Maximum documents:** 1,000,000
- **Unique labels:** 2-5,000
- **Documents per label:** At least 10

**JSON schema for classification:**
```json
{
  "textContent": "inline text here",
  "classificationAnnotation": {
    "displayName": "LABEL"
  },
  "dataItemResourceLabels": {
    "aiplatform.googleapis.com/ml_use": "training"
  }
}
```

**Alternative with GCS reference:**
```json
{
  "textGcsUri": "gs://bucket/document.txt",
  "classificationAnnotation": {
    "displayName": "LABEL"
  }
}
```

**Entity extraction requirements:**
- **Minimum documents:** 50
- **Maximum documents:** 100,000
- **Unique labels:** 1-100
- **Annotations per label:** At least 200 occurrences

**Sentiment analysis requirements:**
- **Minimum documents:** 10
- **Maximum documents:** 100,000
- **Sentiment values:** Integer range (e.g., 0=negative, 1=neutral, 2=positive)
- **Documents per sentiment:** At least 10

**Best practices:**
- Use diverse data (different lengths, authors, styles)
- Ensure human categorizability (if humans can't label it, ML can't either)
- Balance dataset across labels
- Include edge cases and outliers

### VideoDataset

**Supported formats:**
- .MOV, .MPEG4, .MP4, .AVI
- Recommended: MPEG4 or .MP4 for browser compatibility

**File size:**
- Maximum: 50 GB (up to 3 hours)
- Avoid malformed or empty timestamps

**Use cases:**
1. **Action recognition** - Identify human actions in videos
2. **Classification** - Categorize entire videos
3. **Object tracking** - Track objects across frames

**Action recognition requirements:**
- **Maximum labels:** 1,000 per dataset
- **Training frames per label:** At least 100
- Label all actions in video using VAR console

**Object tracking requirements:**
- **Maximum labeled frames:** 150,000
- **Maximum bounding boxes:** 1,000,000
- **Minimum bounding box size:** 10px × 10px
- **Label frequency:** Each label in at least 3 frames with 10+ annotations

**JSON schema for action recognition:**
```json
{
  "videoGcsUri": "gs://bucket/video.mp4",
  "actionAnnotations": [
    {
      "displayName": "LABEL",
      "timeSegment": {
        "startTimeOffset": "0s",
        "endTimeOffset": "5s"
      }
    }
  ]
}
```

**Best practices:**
- Use training data similar to production use case
- Ensure human categorizability (1-2 second label assignment)
- Balance: 100x ratio between most/least common labels
- Consider image quality for high-resolution videos (>1024×1024)

---

## Section 2: Importing Datasets from Multiple Sources (~120 lines)

### Import from Cloud Storage (GCS)

**CSV import format:**
```python
from google.cloud import aiplatform

aiplatform.init(project='PROJECT_ID', location='us-central1')

# Import from CSV
dataset = aiplatform.TabularDataset.create(
    display_name='customer_churn_data',
    gcs_source='gs://ml-training-data/processed/churn_data.csv',
    labels={'env': 'production', 'version': 'v1'}
)
```

**Import file format (JSONL):**
```python
# For image classification
dataset = aiplatform.ImageDataset.create(
    display_name='imagenet_subset',
    gcs_source='gs://ml-training-data/images/imagenet_import.jsonl',
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
)
```

From [Vertex AI Data Integration](../practical-implementation/34-vertex-ai-data-integration.md):

**Import file requirements:**
- One data item per line (JSONL format)
- Max file size: 10 GB per file
- Multiple files supported (total <100 GB for tabular)
- UTF-8 encoding required

**Image classification import file:**
```jsonl
{"imageGcsUri": "gs://bucket/cat_001.jpg", "classificationAnnotation": {"displayName": "cat"}}
{"imageGcsUri": "gs://bucket/dog_001.jpg", "classificationAnnotation": {"displayName": "dog"}}
```

**Batch import with multiple files:**
```python
# Import from directory
dataset = aiplatform.ImageDataset.create(
    display_name='large_image_dataset',
    gcs_source=['gs://bucket/images/*.jsonl'],
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
)
```

### Import from BigQuery

**Direct BigQuery import:**
```python
# Tabular dataset from BigQuery
dataset = aiplatform.TabularDataset.create(
    display_name='bigquery_customer_data',
    bq_source='bq://PROJECT_ID.dataset_id.customer_table',
    labels={'source': 'bigquery', 'version': 'v2'}
)
```

**BigQuery URI format:**
```
bq://PROJECT_ID.DATASET_ID.TABLE_ID
```

**Requirements:**
- Table must exist in same project or have cross-project permissions
- Schema must match dataset type requirements
- Consider BigQuery quotas for large imports

**Query-based import:**
```python
# Import from BigQuery query result
from google.cloud import bigquery

# Create temporary table from query
client = bigquery.Client()
query = """
    SELECT
        customer_id,
        age,
        country,
        activity_score,
        churned
    FROM `PROJECT_ID.dataset.users`
    WHERE created_date >= '2024-01-01'
"""

query_job = client.query(query)
result = query_job.result()

# Save to table
table_id = f"{PROJECT_ID}.temp_dataset.filtered_users"
result.to_dataframe().to_gbq(table_id, if_exists='replace')

# Import from temp table
dataset = aiplatform.TabularDataset.create(
    display_name='filtered_users',
    bq_source=f'bq://{table_id}'
)
```

### Import from Local Files

**Upload via Console:**
1. Navigate to Vertex AI → Datasets
2. Click "Create"
3. Select dataset type and objective
4. Choose "Upload files from your computer"
5. Select files (max 10GB per file)
6. Click "Continue" to start upload

**Programmatic upload:**
```python
from google.cloud import storage
from google.cloud import aiplatform

# Step 1: Upload files to GCS
storage_client = storage.Client()
bucket = storage_client.bucket('ml-training-data')

# Upload local files
local_files = ['data/train.csv', 'data/val.csv', 'data/test.csv']
gcs_paths = []

for local_file in local_files:
    blob = bucket.blob(f'uploads/{os.path.basename(local_file)}')
    blob.upload_from_filename(local_file)
    gcs_paths.append(f'gs://{bucket.name}/{blob.name}')

# Step 2: Create dataset from uploaded files
dataset = aiplatform.TabularDataset.create(
    display_name='uploaded_dataset',
    gcs_source=gcs_paths
)
```

**Streaming upload for large files:**
```python
def upload_large_file_resumable(bucket_name, source_file, destination_blob):
    """Upload large file with resumable upload."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    # Enable resumable upload (>5MB)
    blob.chunk_size = 256 * 1024 * 1024  # 256 MB chunks

    with open(source_file, 'rb') as f:
        blob.upload_from_file(f)

    return f'gs://{bucket_name}/{destination_blob}'
```

### Schema Validation During Import

From [Vertex AI dataset schema validation](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.datasets/import) (accessed 2025-11-16):

**Automatic validation:**
- Vertex AI validates data against schema during import
- Invalid rows logged but don't block import
- Validation errors available in dataset details

**Manual validation before import:**
```python
import jsonschema
import json

# Load schema
schema = {
    "type": "object",
    "properties": {
        "imageGcsUri": {"type": "string"},
        "classificationAnnotation": {
            "type": "object",
            "properties": {
                "displayName": {"type": "string"}
            },
            "required": ["displayName"]
        }
    },
    "required": ["imageGcsUri", "classificationAnnotation"]
}

# Validate JSONL file
with open('import_file.jsonl', 'r') as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            print(f"Line {line_num}: Validation error - {e.message}")
```

**Common validation errors:**
- Missing required fields
- Incorrect data types
- Invalid GCS URIs (file doesn't exist)
- Malformed timestamps
- Out-of-range values

---

## Section 3: Data Split Strategies (~100 lines)

### Automatic Random Split

**Default behavior:**
- 80% training
- 10% validation
- 10% test
- Stratified sampling for classification (preserves label distribution)

**When to use random split:**
- IID (independent and identically distributed) data
- No temporal dependencies
- Sufficient data in all classes

**Python API automatic split:**
```python
# Default random split
dataset = aiplatform.TabularDataset.create(
    display_name='auto_split_dataset',
    gcs_source='gs://bucket/data.csv'
)
# Vertex AI automatically creates 80/10/10 split
```

### Manual Split with ml_use Column

**CSV with manual split:**
```csv
feature1,feature2,target,ml_use
value1,value2,label1,training
value3,value4,label2,training
value5,value6,label3,validation
value7,value8,label4,test
```

**JSONL with manual split:**
```json
{
  "imageGcsUri": "gs://bucket/img1.jpg",
  "classificationAnnotation": {"displayName": "cat"},
  "dataItemResourceLabels": {
    "aiplatform.googleapis.com/ml_use": "training"
  }
}
```

**When to use manual split:**
- Specific data requirements (e.g., certain users in test set)
- Ensure rare classes in all splits
- Reproduce previous experiments
- Cross-validation folds

### Stratified Sampling for Imbalanced Data

From [Vertex AI Data Integration](../practical-implementation/34-vertex-ai-data-integration.md):

**Automatic stratification:**
- Vertex AI uses stratified sampling for classification by default
- Preserves class distribution across splits
- Particularly important for imbalanced datasets

**Manual stratified split:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')

# Stratified split
train_val, test = train_test_split(
    df,
    test_size=0.1,
    stratify=df['target'],
    random_state=42
)

train, val = train_test_split(
    train_val,
    test_size=0.111,  # 10% of total (0.1/0.9)
    stratify=train_val['target'],
    random_state=42
)

# Add ml_use column
train['ml_use'] = 'training'
val['ml_use'] = 'validation'
test['ml_use'] = 'test'

# Combine and save
final_df = pd.concat([train, val, test])
final_df.to_csv('data_with_splits.csv', index=False)
```

**Verify stratification:**
```python
# Check class distribution
print("Training set:")
print(train['target'].value_counts(normalize=True))
print("\nValidation set:")
print(val['target'].value_counts(normalize=True))
print("\nTest set:")
print(test['target'].value_counts(normalize=True))
```

### Temporal Split for Time Series

**Time-based splitting:**
```python
# Sort by time
df = df.sort_values('timestamp')

# Split chronologically
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

df.iloc[:train_size, df.columns.get_loc('ml_use')] = 'training'
df.iloc[train_size:train_size+val_size, df.columns.get_loc('ml_use')] = 'validation'
df.iloc[train_size+val_size:, df.columns.get_loc('ml_use')] = 'test'
```

**Forecasting-specific split:**
```python
# For forecasting, use latest data for validation/test
# Training: 2020-2023
# Validation: Q1 2024
# Test: Q2 2024

df.loc[df['date'] < '2024-01-01', 'ml_use'] = 'training'
df.loc[(df['date'] >= '2024-01-01') & (df['date'] < '2024-04-01'), 'ml_use'] = 'validation'
df.loc[df['date'] >= '2024-04-01', 'ml_use'] = 'test'
```

**Best practices for temporal splits:**
- No data leakage from future to past
- Validation/test sets represent future scenarios
- Consider seasonality in split boundaries
- Leave enough data in validation for early stopping

### Cross-Validation Splits

**K-fold cross-validation:**
```python
from sklearn.model_selection import KFold

# Create 5-fold splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    fold_df = df.copy()
    fold_df['ml_use'] = 'training'
    fold_df.iloc[val_idx, fold_df.columns.get_loc('ml_use')] = 'validation'

    # Save fold
    fold_df.to_csv(f'fold_{fold}_data.csv', index=False)

    # Create dataset for fold
    dataset = aiplatform.TabularDataset.create(
        display_name=f'cv_fold_{fold}',
        gcs_source=f'gs://bucket/fold_{fold}_data.csv'
    )
```

**Stratified K-fold:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
    # Same as above, but stratified by target
    pass
```

---

## Section 4: Data Labeling Service (~150 lines)

### Overview of Data Labeling Service

From [Vertex AI Data Labeling Service](https://docs.cloud.google.com/vertex-ai/docs/samples/aiplatform-create-data-labeling-job-active-learning-sample) (accessed 2025-11-16):

**Data Labeling Service provides:**
- Human labelers for annotation tasks
- Active learning to optimize labeling efficiency
- Quality control with multiple reviewers
- Specialist pools for domain-specific tasks
- Instruction templates for consistent labeling

**Supported task types:**
1. Image classification (single/multi-label)
2. Object detection (bounding boxes)
3. Text classification
4. Text entity extraction
5. Video classification
6. Video action recognition
7. Video object tracking

### Creating a Labeling Job

**Basic labeling job:**
```python
from google.cloud import aiplatform

# Unlabeled dataset
dataset = aiplatform.ImageDataset.create(
    display_name='unlabeled_images',
    gcs_source='gs://bucket/images/*.jpg'
)

# Create labeling job
labeling_job = aiplatform.DataLabelingJob.create(
    display_name='image_classification_labeling',
    datasets=[dataset],
    instruction_uri='gs://bucket/labeling_instructions.pdf',
    inputs_schema_uri=aiplatform.schema.datalabelingjob.inputs.image_classification,
    annotation_labels=['cat', 'dog', 'bird', 'other'],
    specialist_pools=['projects/PROJECT_ID/locations/us-central1/specialistPools/POOL_ID']
)

# Monitor progress
labeling_job.wait()
labeled_dataset = labeling_job.get_labeled_dataset()
```

**Active learning configuration:**
```python
# Active learning reduces labeling costs by intelligently selecting samples
labeling_job = aiplatform.DataLabelingJob.create(
    display_name='active_learning_labeling',
    datasets=[dataset],
    instruction_uri='gs://bucket/instructions.pdf',
    inputs_schema_uri=aiplatform.schema.datalabelingjob.inputs.image_classification,
    annotation_labels=['label1', 'label2', 'label3'],
    specialist_pools=[pool_id],
    active_learning_config={
        'max_data_item_count': 10000,  # Total items to label
        'sample_size': 1000,  # Initial sample size
        'training_config': {
            'timeout_training_milli_hours': 8000  # 8 hours
        }
    }
)
```

**How active learning works:**
1. Label initial sample (e.g., 1,000 images)
2. Train preliminary model
3. Model identifies most uncertain samples
4. Label those high-value samples
5. Repeat until performance plateaus or budget exhausted

### Labeling Instructions and Quality

**Instruction document structure:**
```markdown
# Image Classification Instructions

## Overview
Classify images into one of the following categories:
- Cat: Domestic cats of any breed
- Dog: Domestic dogs of any breed
- Bird: Any bird species
- Other: Animals that don't fit above categories

## Examples

### Cat Examples
[Image 1: Tabby cat sitting]
[Image 2: Persian cat lying down]

### Dog Examples
[Image 1: Golden retriever playing]
[Image 2: Bulldog standing]

## Edge Cases
- If multiple animals, choose primary subject
- If unclear/blurry, select "Other"
- If no animal visible, select "Other"

## Quality Checks
- Review each image carefully (minimum 3 seconds)
- If unsure, mark for review
- Maintain 95%+ accuracy in calibration sets
```

**Upload instructions:**
```bash
# Upload to GCS
gsutil cp labeling_instructions.pdf gs://bucket/instructions/
```

**Quality control configuration:**
```python
labeling_job = aiplatform.DataLabelingJob.create(
    display_name='high_quality_labeling',
    datasets=[dataset],
    instruction_uri='gs://bucket/instructions.pdf',
    inputs_schema_uri=schema_uri,
    annotation_labels=labels,
    specialist_pools=[pool_id],
    # Quality settings
    annotation_spec_set_config={
        'contributor_emails': ['labeler1@company.com', 'labeler2@company.com'],
        'enable_label_verification': True,  # Multiple reviewers
        'verification_threshold': 0.95  # 95% agreement required
    }
)
```

### Specialist Pools

**Create specialist pool:**
```python
# Create pool of domain experts
from google.cloud.aiplatform_v1 import SpecialistPoolServiceClient
from google.cloud.aiplatform_v1.types import SpecialistPool

client = SpecialistPoolServiceClient()

specialist_pool = SpecialistPool(
    display_name='medical_imaging_specialists',
    specialist_manager_emails=['manager@company.com'],
    specialist_worker_emails=[
        'radiologist1@company.com',
        'radiologist2@company.com',
        'radiologist3@company.com'
    ]
)

parent = f'projects/{PROJECT_ID}/locations/us-central1'
response = client.create_specialist_pool(parent=parent, specialist_pool=specialist_pool)
```

**When to use specialist pools:**
- Medical imaging (requires radiologists)
- Legal documents (requires lawyers)
- Scientific data (requires domain PhDs)
- Niche domains (specific expertise needed)

**Cost considerations:**
- Specialist pools: $0.10-$1.00 per label (domain-dependent)
- General labelers: $0.01-$0.08 per label
- Active learning: 30-70% cost reduction
- Quality verification: +20-50% cost

### Pricing and Cost Optimization

From [Vertex AI Data Labeling pricing](https://cloud.google.com/vertex-ai/pricing) (accessed 2025-11-16):

**Pricing structure:**
```
Base labeling costs:
- Image classification: $0.05-$0.08 per image
- Object detection: $0.10-$0.15 per image (depends on box count)
- Text classification: $0.01-$0.03 per document
- Video labeling: $0.50-$2.00 per minute

Specialist pool markup: 2-10x base rate
Quality verification: +25% per additional reviewer
Active learning: Included (no extra charge)
```

**Cost optimization strategies:**

**1. Use active learning:**
```python
# Reduce labeling by 50% with active learning
active_learning_config = {
    'max_data_item_count': 10000,
    'sample_size': 1000,  # Start with 10% of data
    'training_config': {
        'timeout_training_milli_hours': 8000
    }
}
```

**2. Pre-filter data:**
```python
# Remove duplicates and low-quality images before labeling
from PIL import Image
import imagehash

def deduplicate_images(image_paths):
    """Remove duplicate images using perceptual hashing."""
    hashes = {}
    unique_images = []

    for path in image_paths:
        img = Image.open(path)
        h = str(imagehash.average_hash(img))

        if h not in hashes:
            hashes[h] = path
            unique_images.append(path)

    return unique_images

# Filter blurry images
def filter_blurry_images(image_paths, threshold=100):
    """Remove blurry images using Laplacian variance."""
    import cv2
    import numpy as np

    clear_images = []
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance > threshold:  # Clear image
            clear_images.append(path)

    return clear_images
```

**3. Batch labeling jobs:**
```python
# Submit large batches (1000+ items) for volume discounts
# Vertex AI provides pricing tiers
labeling_job = aiplatform.DataLabelingJob.create(
    display_name='batch_labeling',
    datasets=[large_dataset],  # 10,000+ items
    # ... other params
)
```

**4. Use consensus labeling only where needed:**
```python
# Use single labeler for easy cases, consensus for hard cases
# Implement custom logic to route difficult samples

def create_adaptive_labeling_job(dataset, easy_samples, hard_samples):
    # Easy samples: single labeler
    easy_job = aiplatform.DataLabelingJob.create(
        display_name='easy_samples',
        datasets=[create_subset(dataset, easy_samples)],
        annotation_spec_set_config={
            'enable_label_verification': False  # Single labeler
        }
    )

    # Hard samples: consensus labeling
    hard_job = aiplatform.DataLabelingJob.create(
        display_name='hard_samples',
        datasets=[create_subset(dataset, hard_samples)],
        annotation_spec_set_config={
            'enable_label_verification': True,  # Multiple labelers
            'verification_threshold': 0.90
        }
    )

    return easy_job, hard_job
```

---

## Section 5: AutoML Integration (~120 lines)

### Seamless Training After Labeling

**Direct AutoML training from labeled dataset:**
```python
# Step 1: Create and label dataset
dataset = aiplatform.ImageDataset.create(
    display_name='product_images',
    gcs_source='gs://bucket/products/*.jpg'
)

labeling_job = aiplatform.DataLabelingJob.create(
    display_name='product_classification',
    datasets=[dataset],
    instruction_uri='gs://bucket/instructions.pdf',
    inputs_schema_uri=aiplatform.schema.datalabelingjob.inputs.image_classification,
    annotation_labels=['electronics', 'clothing', 'furniture', 'other']
)

labeling_job.wait()
labeled_dataset = labeling_job.get_labeled_dataset()

# Step 2: Train AutoML model directly
training_job = aiplatform.AutoMLImageTrainingJob(
    display_name='product_classifier',
    prediction_type='classification'
)

model = training_job.run(
    dataset=labeled_dataset,  # Use labeled dataset directly
    model_display_name='product_classifier_v1',
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=8000  # 8 node hours
)
```

**AutoML training options:**
```python
# Image classification
image_training = aiplatform.AutoMLImageTrainingJob(
    display_name='image_classifier',
    prediction_type='classification',
    multi_label=False,  # Single-label vs multi-label
    model_type='CLOUD',  # CLOUD vs MOBILE_TF_LOW_LATENCY vs MOBILE_TF_HIGH_ACCURACY
    base_model=None  # Optional: Use pre-trained base model
)

# Tabular classification
tabular_training = aiplatform.AutoMLTabularTrainingJob(
    display_name='tabular_classifier',
    optimization_prediction_type='classification',
    optimization_objective='maximize-au-prc'  # or 'minimize-log-loss'
)

# Text classification
text_training = aiplatform.AutoMLTextTrainingJob(
    display_name='text_classifier',
    prediction_type='classification',
    multi_label=False
)
```

### Dataset Versioning for AutoML

From [Vertex AI dataset versioning](https://docs.cloud.google.com/vertex-ai/docs/datasets/manage-dataset-versions) (accessed 2025-11-16):

**Create dataset versions:**
```python
# Initial version
dataset_v1 = aiplatform.TabularDataset.create(
    display_name='customer_data_v1',
    gcs_source='gs://bucket/customer_jan2024.csv'
)

# Train model on v1
model_v1 = train_automl_model(dataset_v1)

# Create new version with updated data
dataset_v2 = dataset_v1.create_version(
    gcs_source='gs://bucket/customer_feb2024.csv',
    version_description='February 2024 data update - added 5000 samples'
)

# Train model on v2
model_v2 = train_automl_model(dataset_v2)

# Compare performance
print(f"Model v1 accuracy: {model_v1.evaluate()['accuracy']}")
print(f"Model v2 accuracy: {model_v2.evaluate()['accuracy']}")
```

**List and manage versions:**
```python
# List all versions
versions = dataset_v1.list_versions()
for version in versions:
    print(f"Version: {version.version_id}")
    print(f"Description: {version.version_description}")
    print(f"Created: {version.create_time}")
    print(f"Source: {version.gcs_source}")
    print("---")

# Restore previous version
dataset_restored = dataset_v1.get_version(version_id='v1')
model_restored = train_automl_model(dataset_restored)
```

**Versioning best practices:**
- Version on significant data changes (>5% new data)
- Include descriptive version messages
- Track model performance per dataset version
- Archive old versions after 6-12 months

### Monitoring AutoML Training

**Track training progress:**
```python
# Submit training job
training_job = aiplatform.AutoMLImageTrainingJob(
    display_name='product_classifier',
    prediction_type='classification'
)

# Non-blocking submission
model_future = training_job.submit(
    dataset=labeled_dataset,
    model_display_name='product_classifier_v1',
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1
)

# Poll for completion
import time
while not training_job.has_ended:
    state = training_job.state
    print(f"Training state: {state}")
    time.sleep(60)  # Check every minute

# Get trained model
model = model_future.result()
```

**Access training metrics:**
```python
# Get model evaluation metrics
eval_metrics = model.get_model_evaluation()

print(f"AU-PRC: {eval_metrics.metrics['auPrc']}")
print(f"AU-ROC: {eval_metrics.metrics['auRoc']}")
print(f"Log Loss: {eval_metrics.metrics['logLoss']}")

# Confusion matrix
confusion_matrix = eval_metrics.metrics['confusionMatrix']
for row in confusion_matrix['rows']:
    print(row)

# Per-label metrics
for label_metrics in eval_metrics.slice_metrics:
    print(f"Label: {label_metrics['slice']['value']}")
    print(f"  Precision: {label_metrics['metrics']['precision']}")
    print(f"  Recall: {label_metrics['metrics']['recall']}")
    print(f"  F1: {label_metrics['metrics']['f1Score']}")
```

### AutoML Budget and Cost Control

**Budget configuration:**
```python
# Set training budget in milli-node-hours (1 hour = 1000 milli-node-hours)
model = training_job.run(
    dataset=dataset,
    model_display_name='budget_controlled_model',
    budget_milli_node_hours=8000,  # 8 node hours maximum
    disable_early_stopping=False  # Allow early stopping if performance plateaus
)
```

**Cost estimation:**
```
AutoML training costs (approximate):
- Image classification: $3.15/node hour
- Tabular: $19.32/node hour
- Text: $3.15/node hour
- Video: $6.30/node hour

Typical budgets:
- Small dataset (<1000 samples): 1-4 node hours
- Medium dataset (1000-10000): 4-20 node hours
- Large dataset (>10000): 20-100 node hours
```

**Early stopping:**
```python
# AutoML automatically stops if:
# - Validation metric stops improving
# - Budget exhausted
# - Maximum training time reached

# Force minimum training time
model = training_job.run(
    dataset=dataset,
    budget_milli_node_hours=20000,  # 20 hours max
    disable_early_stopping=True  # Use full budget
)
```

### Export and Deploy AutoML Models

**Export for custom serving:**
```python
# Export AutoML model to GCS
model.export_model(
    export_format_id='tf-saved-model',  # or 'tflite', 'edgetpu-tflite', 'tf-js'
    artifact_destination='gs://bucket/exported_models/'
)

# Export for edge deployment
model.export_model(
    export_format_id='edgetpu-tflite',
    artifact_destination='gs://bucket/edge_models/'
)
```

**Deploy to Vertex AI Endpoint:**
```python
# Create endpoint
endpoint = aiplatform.Endpoint.create(display_name='product_classifier_endpoint')

# Deploy model
endpoint.deploy(
    model=model,
    deployed_model_display_name='product_classifier_v1',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=3,
    traffic_percentage=100
)

# Make predictions
predictions = endpoint.predict(instances=[{
    'content': base64_image_content
}])
```

---

## Section 6: Cost Optimization for Data and Labeling (~80 lines)

### Data Storage Costs

From [Vertex AI Data Integration](../practical-implementation/34-vertex-ai-data-integration.md):

**GCS storage pricing:**
```
Standard storage: $0.020/GB/month
Nearline (30-day): $0.010/GB/month
Coldline (90-day): $0.004/GB/month
Archive (365-day): $0.0012/GB/month

Retrieval costs:
Nearline: $0.01/GB
Coldline: $0.02/GB
Archive: $0.05/GB
```

**Lifecycle policy for datasets:**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 30,
          "matchesPrefix": ["datasets/raw/", "datasets/processed/"]
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90,
          "matchesPrefix": ["datasets/archive/"]
        }
      },
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 365,
          "matchesPrefix": ["datasets/temp/"]
        }
      }
    ]
  }
}
```

**Apply lifecycle policy:**
```bash
gsutil lifecycle set lifecycle.json gs://ml-training-data
```

### Data Labeling Cost Optimization

**Strategy 1: Pre-label with cheap models**
```python
# Use free/cheap model for initial predictions
from transformers import pipeline

classifier = pipeline('image-classification', model='google/vit-base-patch16-224')

# Pre-label images
prelabeled_data = []
for image_path in unlabeled_images:
    prediction = classifier(image_path)
    top_label = prediction[0]['label']
    confidence = prediction[0]['score']

    # Only send low-confidence samples to human labelers
    if confidence < 0.8:
        prelabeled_data.append({
            'image': image_path,
            'suggested_label': top_label,
            'needs_review': True
        })

# Human labeling only for uncertain cases (20-30% of data)
labeling_job = create_labeling_job(prelabeled_data)
```

**Strategy 2: Progressive labeling**
```python
# Label in batches, train incrementally
def progressive_labeling(unlabeled_data, batch_size=1000):
    labeled_data = []

    for batch_num in range(0, len(unlabeled_data), batch_size):
        batch = unlabeled_data[batch_num:batch_num+batch_size]

        # Label batch
        batch_labeled = label_batch(batch)
        labeled_data.extend(batch_labeled)

        # Train preliminary model
        temp_model = train_automl_model(labeled_data)

        # Evaluate on held-out set
        accuracy = evaluate_model(temp_model)
        print(f"Batch {batch_num//batch_size}: Accuracy = {accuracy}")

        # Stop if accuracy plateaus
        if accuracy > 0.95:  # Target threshold
            break

    return labeled_data
```

**Strategy 3: Hybrid human-AI labeling**
```python
# Combine AutoML predictions with human review
def hybrid_labeling(dataset, confidence_threshold=0.9):
    # Step 1: Train initial model on small labeled set
    initial_model = train_automl_model(small_labeled_dataset)

    # Step 2: Predict on unlabeled data
    predictions = initial_model.batch_predict(dataset)

    # Step 3: Route based on confidence
    high_confidence = []  # Auto-label
    low_confidence = []   # Human review

    for item, pred in zip(dataset, predictions):
        if pred.confidence > confidence_threshold:
            high_confidence.append({
                'item': item,
                'label': pred.label,
                'source': 'automl'
            })
        else:
            low_confidence.append(item)

    # Step 4: Human labeling only for low-confidence
    human_labeled = create_labeling_job(low_confidence)

    # Combine
    final_dataset = high_confidence + human_labeled
    return final_dataset
```

### AutoML Training Cost Optimization

**Use model export instead of hosted deployment:**
```python
# Export model for self-hosted serving (cheaper than Vertex AI Endpoints)
model.export_model(
    export_format_id='tf-saved-model',
    artifact_destination='gs://bucket/models/'
)

# Deploy on GKE or Cloud Run (lower cost)
# Vertex AI Endpoint: $0.10/hour + prediction costs
# GKE: $0.04/hour (preemptible) + prediction costs
```

**Preemptible training (when available):**
```python
# Note: AutoML doesn't support preemptible instances directly
# For custom training, use preemptible workers
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='preemptible_training',
    container_uri='gcr.io/project/trainer:latest',
    model_serving_container_uri='gcr.io/project/predictor:latest'
)

model = job.run(
    dataset=dataset,
    replica_count=4,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    boot_disk_type='pd-ssd',
    boot_disk_size_gb=100,
    reduction_server_replica_count=1,
    reduction_server_machine_type='n1-highcpu-16',
    # Use preemptible instances (70% cost savings)
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1
)
```

---

## Section 7: arr-coc-0-1 Dataset Preparation Workflow (~130 lines)

### Visual Texture Dataset for ARR-COC Training

**Dataset structure for texture analysis:**
```python
# arr-coc-0-1 requires 13-channel texture arrays
# Dataset must include:
# - RGB (3 channels)
# - LAB (3 channels)
# - Sobel edges (2 channels: x, y gradients)
# - Spatial coordinates (2 channels: x, y)
# - Eccentricity map (1 channel)
# - LOD ground truth (1 channel)
# - Query relevance scores (1 channel)

class ArrCocDatasetPreparation:
    """Prepare dataset for arr-coc-0-1 training."""

    def __init__(self, image_dir, query_embeddings_path):
        self.image_dir = image_dir
        self.query_embeddings = load_query_embeddings(query_embeddings_path)

    def prepare_training_sample(self, image_path, query_text):
        """
        Convert image to 13-channel texture array.

        Returns:
            texture_array: (H, W, 13) tensor
            lod_target: (num_patches,) LOD ground truth
            relevance_target: (num_patches,) relevance scores
        """
        import cv2
        import numpy as np
        from skimage import color

        # Load image
        img_rgb = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Channel 0-2: RGB
        channels = [img_rgb]

        # Channel 3-5: LAB
        img_lab = color.rgb2lab(img_rgb / 255.0)
        channels.append(img_lab)

        # Channel 6-7: Sobel edges
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        channels.append(np.stack([sobel_x, sobel_y], axis=-1))

        # Channel 8-9: Spatial coordinates
        h, w = img_rgb.shape[:2]
        x_coords = np.linspace(0, 1, w)
        y_coords = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x_coords, y_coords)
        channels.append(np.stack([xx, yy], axis=-1))

        # Channel 10: Eccentricity map
        eccentricity = self.compute_eccentricity_map(h, w)
        channels.append(eccentricity[..., None])

        # Channel 11: LOD ground truth (human-annotated importance)
        lod_map = self.compute_lod_ground_truth(image_path, query_text)
        channels.append(lod_map[..., None])

        # Channel 12: Query relevance scores
        relevance_map = self.compute_query_relevance(img_rgb, query_text)
        channels.append(relevance_map[..., None])

        # Concatenate all channels
        texture_array = np.concatenate(channels, axis=-1)

        return texture_array

    def compute_eccentricity_map(self, h, w):
        """Compute visual eccentricity from center."""
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        eccentricity = distances / max_distance
        return eccentricity

    def compute_lod_ground_truth(self, image_path, query_text):
        """
        Generate LOD ground truth from human annotations.

        For arr-coc-0-1, LOD ranges from 64-400 tokens per patch.
        Ground truth based on human assessment of query-relevant regions.
        """
        # Load human annotations (saliency maps, attention maps)
        annotation_path = image_path.replace('.jpg', '_lod_annotation.npy')
        if os.path.exists(annotation_path):
            lod_map = np.load(annotation_path)
        else:
            # Fallback: Use saliency detection
            lod_map = self.estimate_lod_from_saliency(image_path)

        return lod_map

    def compute_query_relevance(self, img_rgb, query_text):
        """
        Compute pixel-level relevance to query using CLIP.

        This provides supervision for the participatory knowing component.
        """
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Encode query
        text_tokens = clip.tokenize([query_text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)

        # Compute patch-level relevance
        h, w = img_rgb.shape[:2]
        patch_size = 32
        relevance_map = np.zeros((h, w))

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img_rgb[y:y+patch_size, x:x+patch_size]

                # Encode patch
                patch_pil = Image.fromarray(patch)
                patch_tensor = preprocess(patch_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    patch_features = model.encode_image(patch_tensor)

                # Compute similarity
                similarity = torch.cosine_similarity(
                    text_features, patch_features
                ).item()

                relevance_map[y:y+patch_size, x:x+patch_size] = similarity

        return relevance_map
```

### Creating Vertex AI Dataset for arr-coc-0-1

**Prepare dataset in Vertex AI format:**
```python
def create_arrcoc_vertex_dataset(images_dir, annotations_dir, output_jsonl):
    """
    Create JSONL import file for arr-coc-0-1 training dataset.

    Each sample includes:
    - Image GCS URI
    - 13-channel texture array URI
    - LOD ground truth URI
    - Query text
    - Metadata
    """
    import json

    samples = []

    for image_file in os.listdir(images_dir):
        if not image_file.endswith('.jpg'):
            continue

        image_id = image_file.replace('.jpg', '')

        # Prepare training sample
        sample = {
            'imageGcsUri': f'gs://arr-coc-training/images/{image_file}',
            'textureArrayUri': f'gs://arr-coc-training/textures/{image_id}_texture.npy',
            'lodGroundTruthUri': f'gs://arr-coc-training/lod/{image_id}_lod.npy',
            'relevanceGroundTruthUri': f'gs://arr-coc-training/relevance/{image_id}_relevance.npy',
            'queryText': load_query_for_image(image_id),
            'dataItemResourceLabels': {
                'aiplatform.googleapis.com/ml_use': assign_split(image_id),
                'dataset_version': 'v1',
                'contains_human_annotations': 'true'
            }
        }

        samples.append(sample)

    # Write JSONL
    with open(output_jsonl, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created dataset with {len(samples)} samples")
    return output_jsonl

# Upload to GCS
upload_to_gcs(output_jsonl, 'gs://arr-coc-training/dataset_import.jsonl')

# Create Vertex AI dataset
dataset = aiplatform.CustomDataset.create(
    display_name='arr_coc_texture_dataset_v1',
    gcs_source='gs://arr-coc-training/dataset_import.jsonl',
    labels={
        'project': 'arr-coc-ovis',
        'model_type': 'relevance_realization',
        'texture_channels': '13'
    }
)
```

### Custom Training Pipeline for arr-coc-0-1

From [arr-coc-0-1 implementation](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):

**Training pipeline with Vertex AI:**
```python
from google.cloud import aiplatform

# Define custom training job
training_job = aiplatform.CustomContainerTrainingJob(
    display_name='arr_coc_training_v1',
    container_uri='gcr.io/project-id/arr-coc-trainer:latest',
    model_serving_container_image_uri='gcr.io/project-id/arr-coc-server:latest',
    model_serving_container_predict_route='/predict',
    model_serving_container_health_route='/health'
)

# Run training
model = training_job.run(
    dataset=dataset,
    model_display_name='arr_coc_relevance_allocator_v1',
    args=[
        '--texture_channels=13',
        '--lod_range_min=64',
        '--lod_range_max=400',
        '--num_patches=200',
        '--opponent_process_mode=dynamic',
        '--learning_rate=0.001',
        '--batch_size=16',
        '--num_epochs=100'
    ],
    replica_count=1,
    machine_type='n1-highmem-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    sync=False  # Async training
)

# Monitor via W&B (runs inside container)
# See: arr-coc-0-1/training/train.py for W&B integration
```

**Dataset versioning for iterations:**
```python
# Version 1: Initial dataset (1000 samples)
dataset_v1 = create_arrcoc_vertex_dataset(
    images_dir='gs://arr-coc/images_v1/',
    annotations_dir='gs://arr-coc/annotations_v1/'
)

# Train baseline model
model_v1 = train_arr_coc_model(dataset_v1)
baseline_metrics = evaluate_model(model_v1)

# Version 2: Expanded dataset (5000 samples)
dataset_v2 = dataset_v1.create_version(
    gcs_source='gs://arr-coc-training/dataset_v2_import.jsonl',
    version_description='Expanded dataset: added 4000 samples with diverse queries'
)

# Train improved model
model_v2 = train_arr_coc_model(dataset_v2)
improved_metrics = evaluate_model(model_v2)

# Compare
print(f"Baseline relevance accuracy: {baseline_metrics['relevance_acc']}")
print(f"Improved relevance accuracy: {improved_metrics['relevance_acc']}")
```

---

## Sources

**Google Cloud Documentation:**
- [Vertex AI Datasets overview](https://docs.cloud.google.com/vertex-ai/docs/datasets/overview) - Managed dataset types and capabilities
- [Use managed datasets](https://docs.cloud.google.com/vertex-ai/docs/training/using-managed-datasets) - Dataset access in training jobs
- [Manage dataset versions](https://docs.cloud.google.com/vertex-ai/docs/datasets/manage-dataset-versions) - Versioning and lineage tracking
- [Create data labeling job](https://docs.cloud.google.com/vertex-ai/docs/samples/aiplatform-create-data-labeling-job-active-learning-sample) - Active learning for labeling
- [Prepare training data](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/prepare-data) - Data structure requirements
- [Dataset import API](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.datasets/import) - Import methods and schema validation

**Web Research:**
- [Managing Machine Learning Datasets with Vertex AI](https://medium.com/@devashish_m/managing-machine-learning-datasets-with-vertex-ai-a-complete-guide-4e0bfef4d6c6) - Medium article by Devashish Datt Mamgain (accessed 2025-11-16)
- [Dataset Management in Vertex AI](https://promevo.com/blog/dataset-management-vertex-ai) - Promevo guide (accessed 2025-11-16)
- [Import & Export Datasets in Vertex AI](https://promevo.com/blog/import-and-export-data-sets-in-vertex-ai) - Promevo technical guide (accessed 2025-11-16)

**Internal Documentation:**
- [Vertex AI Data Integration](../practical-implementation/34-vertex-ai-data-integration.md) - GCS FUSE, managed datasets, data pipelines
- [arr-coc-0-1 implementation](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - 13-channel texture arrays, relevance scoring

---

*This document provides production-ready patterns for dataset management on Vertex AI, covering managed datasets for all data types, Data Labeling Service with active learning, seamless AutoML integration workflows, versioning strategies, cost optimization techniques, and arr-coc-0-1 specific dataset preparation for visual relevance realization training.*
