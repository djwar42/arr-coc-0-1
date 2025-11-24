# Dataflow for ML Preprocessing

**Master cloud-scale data preprocessing for machine learning with Apache Beam and Dataflow**

## Overview

Google Cloud Dataflow provides serverless, scalable data preprocessing for ML workflows. Built on Apache Beam, it handles both batch and streaming data transformation pipelines with automatic scaling, unified programming model, and tight integration with Vertex AI Pipelines.

**Core value proposition:**
- **Unified model**: Same code for batch and streaming pipelines
- **Serverless scaling**: Automatic worker management (1-1000+ workers)
- **Training-serving consistency**: Prevent skew with TensorFlow Transform
- **Vertex AI integration**: Native pipeline component for end-to-end MLOps

From [Apache Beam ML preprocessing documentation](https://beam.apache.org/documentation/ml/preprocess-data/) (accessed 2025-11-16):
> "MLTransform can do a full pass on the dataset, which is useful when you need to transform a single element only after analyzing the entire dataset."

From [TensorFlow Transform guide](https://www.tensorflow.org/tfx/guide/transform) (accessed 2025-11-16):
> "By emitting a TensorFlow graph, not just statistics, TensorFlow Transform simplifies the process of authoring your preprocessing pipeline... This consistency eliminates one source of training/serving skew."

---

## Section 1: Apache Beam Python SDK for ML

### Core Transforms

Apache Beam provides distributed data processing primitives optimized for ML preprocessing at scale.

**ParDo (Parallel Do)**
Element-wise transformation with side outputs and state management:

```python
import apache_beam as beam

class NormalizeImageFn(beam.DoFn):
    """Normalize image pixels to [0, 1] range."""

    def process(self, element):
        import numpy as np

        image_bytes = element['image']
        label = element['label']

        # Decode and normalize
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = image.reshape(224, 224, 3)
        normalized = image.astype(np.float32) / 255.0

        yield {
            'image_normalized': normalized.tobytes(),
            'label': label,
            'shape': (224, 224, 3)
        }

# Pipeline usage
with beam.Pipeline() as p:
    images = (
        p
        | 'ReadImages' >> beam.io.ReadFromTFRecord('gs://data/images.tfrecord')
        | 'ParseProto' >> beam.Map(parse_tf_example)
        | 'Normalize' >> beam.ParDo(NormalizeImageFn())
    )
```

**ParDo with side inputs** for global context (vocabulary, statistics):

```python
class ApplyVocabularyFn(beam.DoFn):
    """Map tokens to IDs using vocabulary side input."""

    def process(self, element, vocabulary):
        """
        Args:
            element: Text data
            vocabulary: Side input (singleton or dict)
        """
        tokens = element['tokens']
        vocab_dict = vocabulary  # Materialized as dict in memory

        token_ids = [
            vocab_dict.get(token, vocab_dict['<UNK>'])
            for token in tokens
        ]

        yield {**element, 'token_ids': token_ids}

# Build vocabulary as side input
vocab = (
    data
    | 'ExtractTokens' >> beam.FlatMap(lambda x: x['tokens'])
    | 'CountTokens' >> beam.combiners.Count.PerElement()
    | 'TopK' >> beam.combiners.Top.Of(10000, key=lambda x: x[1])
    | 'CreateVocabDict' >> beam.Map(
        lambda items: {token: idx for idx, (token, _) in enumerate(items)}
    )
)

# Use as side input
encoded = (
    data
    | beam.ParDo(ApplyVocabularyFn(), vocabulary=beam.pvalue.AsSingleton(vocab))
)
```

**GroupByKey**
Shuffle and group data for aggregations (expensive operation):

```python
# Group images by class for stratified sampling
class_samples = (
    images
    | 'AddKey' >> beam.Map(lambda x: (x['label'], x))
    | 'GroupByClass' >> beam.GroupByKey()
    | 'SamplePerClass' >> beam.Map(
        lambda kv: (kv[0], random.sample(list(kv[1]), k=100))
    )
)
```

**Combine**
Efficient global and per-key aggregations with combiner optimization:

```python
class ComputePixelMeanStd(beam.CombineFn):
    """Compute mean and std across all images for normalization."""

    def create_accumulator(self):
        return {
            'sum': np.zeros(3, dtype=np.float64),
            'sum_sq': np.zeros(3, dtype=np.float64),
            'count': 0
        }

    def add_input(self, accumulator, element):
        """Add one image's pixel statistics."""
        pixels = np.frombuffer(element['image_normalized'], dtype=np.float32)
        pixels = pixels.reshape(-1, 3)  # (H*W, 3)

        accumulator['sum'] += pixels.sum(axis=0)
        accumulator['sum_sq'] += (pixels ** 2).sum(axis=0)
        accumulator['count'] += pixels.shape[0]
        return accumulator

    def merge_accumulators(self, accumulators):
        """Merge partial statistics from different workers."""
        merged = self.create_accumulator()
        for acc in accumulators:
            merged['sum'] += acc['sum']
            merged['sum_sq'] += acc['sum_sq']
            merged['count'] += acc['count']
        return merged

    def extract_output(self, accumulator):
        """Compute final mean and std."""
        mean = accumulator['sum'] / accumulator['count']
        variance = (accumulator['sum_sq'] / accumulator['count']) - (mean ** 2)
        std = np.sqrt(variance)
        return {'mean': mean.tolist(), 'std': std.tolist()}

# Global combine
stats = (
    images
    | 'ComputeStats' >> beam.CombineGlobally(ComputePixelMeanStd())
)

# Per-key combine (e.g., per class)
class_stats = (
    images
    | 'KeyByClass' >> beam.Map(lambda x: (x['label'], x))
    | 'PerClassStats' >> beam.CombinePerKey(ComputePixelMeanStd())
)
```

### Pipeline Patterns for ML

**Windowing for streaming data:**

```python
from apache_beam import window

# Fixed windows (batch incoming data every 10 minutes)
windowed = (
    streaming_images
    | 'Window' >> beam.WindowInto(window.FixedWindows(10 * 60))  # 10 min
    | 'ComputeBatchStats' >> beam.CombineGlobally(ComputePixelMeanStd())
        .without_defaults()  # Don't emit empty windows
)

# Sliding windows for continuous monitoring
sliding = (
    predictions
    | 'SlidingWindow' >> beam.WindowInto(
        window.SlidingWindows(size=3600, period=300)  # 1hr window, 5min slides
    )
    | 'ComputeAccuracy' >> beam.CombineGlobally(AccuracyMetric())
)

# Session windows (group by user activity bursts)
sessions = (
    user_events
    | 'SessionWindow' >> beam.WindowInto(window.Sessions(gap_size=30 * 60))  # 30min gap
    | 'GroupByUser' >> beam.GroupByKey()
    | 'ExtractFeatures' >> beam.Map(extract_session_features)
)
```

**Data splitting:**

```python
def split_train_val_test(element, num_partitions=10):
    """Deterministic split based on hash."""
    import hashlib

    # Use ID for consistent splitting
    partition = int(hashlib.md5(
        element['id'].encode()
    ).hexdigest(), 16) % num_partitions

    if partition < 7:  # 70%
        return 0  # train
    elif partition < 9:  # 20%
        return 1  # val
    else:  # 10%
        return 2  # test

train, val, test = (
    data
    | 'Split' >> beam.Partition(split_train_val_test, 3)
)

# Write to separate outputs
train | 'WriteTrain' >> beam.io.WriteToTFRecord('gs://bucket/train')
val | 'WriteVal' >> beam.io.WriteToTFRecord('gs://bucket/val')
test | 'WriteTest' >> beam.io.WriteToTFRecord('gs://bucket/test')
```

From [Apache Beam documentation](https://beam.apache.org/documentation/ml/preprocess-data/) (accessed 2025-11-16):
> "MLTransform wraps the various transforms in one class, simplifying your workflow... You can use the same preprocessing steps for both training and inference, which ensures consistent results."

---

## Section 2: TensorFlow Transform (tf.Transform)

### Two-Phase Processing

TensorFlow Transform (TFT) provides training-serving consistency by embedding preprocessing logic in the TensorFlow graph.

**Analyze phase** (full-pass statistics):

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """
    Preprocessing function with two execution phases:

    ANALYZE PHASE (training only):
    - Full pass over training data
    - Compute global statistics (mean, vocab, quantiles)
    - Save as constants in TF graph

    TRANSFORM PHASE (training + serving):
    - Apply transformations using saved constants
    - Same graph used in training and serving
    """

    outputs = {}

    # NUMERIC FEATURES: Z-score normalization
    # Analyze: Compute mean, variance across all training examples
    # Transform: (x - mean) / std using saved constants
    outputs['normalized_pixels'] = tft.scale_to_z_score(
        inputs['pixels']
    )

    # Min-max scaling to [0, 1]
    # Analyze: Find global min, max
    # Transform: (x - min) / (max - min)
    outputs['scaled_intensity'] = tft.scale_to_0_1(
        inputs['intensity']
    )

    # CATEGORICAL FEATURES: Vocabulary generation
    # Analyze: Count all unique values, create vocab file
    # Transform: Map strings to integer IDs
    outputs['category_id'] = tft.compute_and_apply_vocabulary(
        inputs['category'],
        top_k=10000,
        num_oov_buckets=1,  # Out-of-vocabulary bucket
        vocab_filename='category_vocab'
    )

    # BUCKETIZATION: Quantile-based binning
    # Analyze: Compute quantile boundaries
    # Transform: Assign to buckets
    outputs['age_bucket'] = tft.bucketize(
        inputs['age'],
        num_buckets=10
    )

    # FEATURE CROSSING
    # Combine multiple features after vocabulary encoding
    category_x_region = tft.hash(
        tf.strings.join([
            tf.strings.as_string(outputs['category_id']),
            tf.strings.as_string(inputs['region_id'])
        ], separator='_'),
        hash_bucket_size=1000
    )
    outputs['category_x_region'] = category_x_region

    # TEXT FEATURES: TF-IDF
    # Analyze: Build vocabulary, count document frequencies
    # Transform: Compute TF-IDF weights
    tokens = tf.strings.split(inputs['text'])
    outputs['tfidf'] = tft.tfidf(
        tokens,
        vocab_size=10000
    )

    return outputs
```

**Transform phase** (apply transformations):

The same `preprocessing_fn` runs during:
1. **Training**: Analyze phase computes statistics, transform phase applies them
2. **Serving**: Only transform phase runs, using saved statistics

```python
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

# Define input schema
raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'pixels': tf.io.FixedLenFeature([224, 224, 3], tf.float32),
        'intensity': tf.io.FixedLenFeature([], tf.float32),
        'category': tf.io.FixedLenFeature([], tf.string),
        'age': tf.io.FixedLenFeature([], tf.float32),
        'region_id': tf.io.FixedLenFeature([], tf.int64),
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    })
)

# Beam pipeline with TFT
with beam.Pipeline() as p:
    with tft_beam.Context(temp_dir='gs://bucket/tft_tmp'):
        # Read raw data
        raw_data = (
            p
            | 'ReadTrainData' >> beam.io.ReadFromTFRecord(
                'gs://bucket/train/*.tfrecord'
            )
            | 'DecodeExamples' >> beam.Map(
                lambda x: serialized_to_dict(x, raw_data_metadata.schema)
            )
        )

        # AnalyzeAndTransformDataset: Two-phase execution
        # Phase 1 (Analyze): Full pass to compute statistics
        # Phase 2 (Transform): Apply transformations using statistics
        transformed_dataset, transform_fn = (
            (raw_data, raw_data_metadata)
            | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn
            )
        )

        transformed_data, transformed_metadata = transformed_dataset

        # Write transformed data
        _ = (
            transformed_data
            | 'EncodeTransformed' >> beam.Map(
                lambda x: dict_to_example(x, transformed_metadata.schema)
            )
            | 'WriteTransformed' >> beam.io.WriteToTFRecord(
                'gs://bucket/transformed/train',
                coder=beam.coders.ProtoCoder(tf.train.Example)
            )
        )

        # Write transform_fn as SavedModel (contains analyze artifacts)
        _ = (
            transform_fn
            | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
                'gs://bucket/transform_fn'
            )
        )
```

**Using saved transform_fn for serving:**

```python
# Load transform_fn SavedModel
transform_fn = tft.TFTransformOutput('gs://bucket/transform_fn')

# Create serving signature that includes preprocessing
@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serve_tf_examples_fn(serialized_examples):
    """Serving function with embedded preprocessing."""

    # Parse raw examples
    raw_features = tf.io.parse_example(
        serialized_examples,
        raw_data_metadata.schema.as_feature_spec()
    )

    # Apply SAME preprocessing as training
    # Uses saved statistics (mean, vocab, etc.)
    transformed_features = transform_fn.transform_raw_features(
        raw_features
    )

    # Run model inference
    predictions = model(transformed_features)

    return {'predictions': predictions}

# Export with preprocessing embedded
model.save(
    'gs://bucket/saved_model',
    signatures={'serving_default': serve_tf_examples_fn}
)
```

From [TensorFlow Transform documentation](https://www.tensorflow.org/tfx/guide/transform) (accessed 2025-11-16):
> "If you do it within TensorFlow Transform, transforms become part of the TensorFlow graph. This approach helps avoid training/serving skew."

### Training-Serving Skew Prevention

**Common skew sources and TFT solutions:**

| Skew Source | Without TFT | With TFT |
|-------------|-------------|----------|
| **Different vocabularies** | Training uses vocab v1, serving rebuilds vocab v2 | Vocab saved in transform_fn, exact same mapping |
| **Different normalization** | Training normalizes with stats from old data | Mean/std saved, same normalization always |
| **Different preprocessing code** | Training uses Python, serving uses different lang | Same TF graph for both |
| **Different TF versions** | Ops behave differently | Locked to TF version at export |

**Example: Vocabulary consistency**

```python
# Without TFT (SKEW RISK)
# Training:
vocab = build_vocab_from_data(training_data)  # ['cat', 'dog', 'bird']
save_vocab(vocab)

# Serving (6 months later):
vocab = load_vocab()  # What if file corrupted? Order changed?
# 'cat' -> 0 or 'dog' -> 0?

# With TFT (NO SKEW)
# Training:
def preprocessing_fn(inputs):
    return {
        'category_id': tft.compute_and_apply_vocabulary(
            inputs['category'],
            vocab_filename='category'
        )
    }
# Vocabulary embedded in TF graph as lookup table

# Serving:
# Same lookup table loaded from SavedModel
# 'cat' ALWAYS maps to same ID
```

**Example: Normalization consistency**

```python
# Without TFT (SKEW RISK)
# Training:
mean = compute_mean(training_data)  # 128.5
std = compute_std(training_data)    # 45.2
normalized = (x - mean) / std

# Serving (data distribution changed):
# Do we recompute mean/std? Use old values? Different preprocessing!

# With TFT (NO SKEW)
def preprocessing_fn(inputs):
    # Mean/std computed ONCE during training
    # Saved as constants in graph
    return {'normalized': tft.scale_to_z_score(inputs['pixels'])}

# Serving uses EXACT same mean/std from training
```

---

## Section 3: Dataflow Pipeline Deployment

### Autoscaling Workers

Dataflow automatically scales workers based on pipeline backlog and resource utilization.

**Horizontal autoscaling configuration:**

```python
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import WorkerOptions

# Pipeline options for autoscaling
options = PipelineOptions()

# Google Cloud configuration
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'my-project'
google_cloud_options.region = 'us-central1'
google_cloud_options.staging_location = 'gs://bucket/staging'
google_cloud_options.temp_location = 'gs://bucket/temp'

# Worker autoscaling
worker_options = options.view_as(WorkerOptions)
worker_options.machine_type = 'n1-standard-4'
worker_options.disk_size_gb = 100
worker_options.num_workers = 5  # Initial workers
worker_options.max_num_workers = 50  # Scale up to 50
worker_options.autoscaling_algorithm = 'THROUGHPUT_BASED'  # or 'NONE'

# Setup options
setup_options = options.view_as(SetupOptions)
setup_options.save_main_session = True  # Pickle global context

# Create pipeline
with beam.Pipeline(options=options) as p:
    # Your transforms here
    pass
```

**Autoscaling modes:**

| Algorithm | Behavior | Use Case |
|-----------|----------|----------|
| `THROUGHPUT_BASED` | Scale based on pipeline backlog and worker CPU | Default, works for most pipelines |
| `NONE` | Fixed worker count | Predictable costs, known workload |

**Monitoring autoscaling:**

```python
# Metrics to watch (Cloud Monitoring)
# - dataflow.job/current_num_vcpus
# - dataflow.job/backlog_seconds (how far behind)
# - dataflow.job/system_lag (processing delay)
# - dataflow.job/data_watermark_age

# Autoscaling triggers when:
# 1. Backlog > threshold (workers falling behind)
# 2. CPU utilization > 80% (workers saturated)
# 3. Available quotas allow more workers
```

### Shuffle Service

Dataflow Shuffle offloads GroupByKey operations to a managed service, improving performance and reducing worker memory pressure.

**Enable Shuffle service:**

```python
from apache_beam.options.pipeline_options import StandardOptions

# Batch pipelines
standard_options = options.view_as(StandardOptions)
standard_options.dataflow_service_options = ['shuffle_mode=service']

# Streaming pipelines (Streaming Engine includes shuffle)
standard_options.streaming = True
standard_options.enable_streaming_engine = True  # Includes shuffle
```

**Benefits:**

- **Reduced worker memory**: Shuffle data stored in service, not worker RAM
- **Faster recovery**: Failed workers don't lose shuffle data
- **Better autoscaling**: Workers can scale down without losing intermediate data
- **Cost**: Additional cost, but often offset by smaller/fewer workers

**When to use:**

```python
# Use Shuffle service when:
# - GroupByKey on large datasets (>100GB)
# - High cardinality keys (millions of unique keys)
# - Worker OOM errors during shuffle
# - Need fast recovery from worker failures

# Example: Large-scale vocabulary building
vocab = (
    text_data  # Billions of documents
    | 'Tokenize' >> beam.FlatMap(tokenize)
    | 'CountTokens' >> beam.combiners.Count.PerElement()  # LARGE shuffle
    | 'FilterRare' >> beam.Filter(lambda x: x[1] > 10)
    | 'TopK' >> beam.combiners.Top.Of(50000, key=lambda x: x[1])
)
# Without shuffle service: Workers OOM
# With shuffle service: Scales smoothly
```

### Streaming vs Batch Pipelines

**Batch pipeline** (bounded data):

```python
# Process historical data, finite dataset
batch_options = PipelineOptions()
batch_options.view_as(StandardOptions).runner = 'DataflowRunner'

with beam.Pipeline(options=batch_options) as p:
    images = (
        p
        | 'Read' >> beam.io.ReadFromTFRecord('gs://bucket/images/*.tfrecord')
        | 'Preprocess' >> beam.ParDo(PreprocessImageFn())
        | 'Write' >> beam.io.WriteToTFRecord('gs://bucket/preprocessed/')
    )
# Pipeline completes when all data processed
```

**Streaming pipeline** (unbounded data):

```python
# Process real-time data, infinite stream
streaming_options = PipelineOptions()
streaming_options.view_as(StandardOptions).streaming = True
streaming_options.view_as(StandardOptions).enable_streaming_engine = True

with beam.Pipeline(options=streaming_options) as p:
    realtime_images = (
        p
        | 'ReadPubSub' >> beam.io.ReadFromPubSub(
            topic='projects/my-project/topics/images'
        )
        | 'Parse' >> beam.Map(parse_message)
        | 'Window' >> beam.WindowInto(window.FixedWindows(60))  # 1min windows
        | 'Preprocess' >> beam.ParDo(PreprocessImageFn())
        | 'WriteBQ' >> beam.io.WriteToBigQuery(
            'my-project:dataset.preprocessed_images',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
        )
    )
# Pipeline runs indefinitely, processing incoming messages
```

**Streaming Engine benefits:**

- **Lower latency**: Sub-second processing for real-time inference
- **Automatic state management**: Worker state stored in service
- **Better resource utilization**: CPU/memory managed by service
- **Horizontal autoscaling**: Scale workers based on message rate

From [Dataflow documentation](https://cloud.google.com/dataflow/docs/guides/tune-horizontal-autoscaling) (accessed 2025-11-16):
> "Dataflow uses the average CPU utilization as a signal for when to apply Horizontal Autoscaling. By default, Dataflow sets a target CPU utilization of 0.8."

---

## Section 4: Windowing for Streaming Data

### Window Types

**Fixed windows** (non-overlapping time intervals):

```python
# Batch incoming images every 10 minutes
fixed = (
    streaming_images
    | 'FixedWindow' >> beam.WindowInto(
        window.FixedWindows(size=10 * 60)  # 10 minutes
    )
    | 'ComputeStats' >> beam.CombineGlobally(ImageStats())
        .without_defaults()  # Don't emit for empty windows
)

# Use case: Batch predictions every N minutes
# - Accumulate images for 10 minutes
# - Run batch inference
# - Write results to database
```

**Sliding windows** (overlapping intervals):

```python
# 1-hour window, sliding every 5 minutes
# Allows monitoring metrics with overlap
sliding = (
    predictions
    | 'SlidingWindow' >> beam.WindowInto(
        window.SlidingWindows(
            size=3600,    # 1 hour window
            period=300    # Slide every 5 minutes
        )
    )
    | 'ComputeAccuracy' >> beam.CombineGlobally(AccuracyMetric())
)

# At time T:
# Window 1: [T-60min, T]
# Window 2: [T-55min, T+5min]  (5min later)
# Window 3: [T-50min, T+10min] (10min later)
# Each datapoint appears in 12 windows (60min / 5min)

# Use case: Rolling accuracy monitoring
# - Continuously updated metrics
# - Detect accuracy degradation
# - Trigger retraining alerts
```

**Session windows** (activity-based):

```python
# Group events by user sessions (30min inactivity gap)
sessions = (
    user_events
    | 'SessionWindow' >> beam.WindowInto(
        window.Sessions(gap_size=30 * 60)  # 30-minute gap
    )
    | 'GroupByUser' >> beam.GroupByKey()
    | 'ExtractSessionFeatures' >> beam.Map(extract_session_features)
)

# Example: User session at 10:00, 10:05, 10:10, 10:45
# Session 1: [10:00-10:40] (last event + 30min)
# Gap of 35 minutes
# Session 2: [10:45-11:15] (new session)

# Use case: User behavior analysis
# - Session duration
# - Actions per session
# - Session-based recommendations
```

**Global window with triggers** (custom emission):

```python
# Emit early results, then final result when window closes
from apache_beam.transforms import trigger

global_with_trigger = (
    streaming_data
    | 'GlobalWindow' >> beam.WindowInto(
        window.GlobalWindows(),
        trigger=trigger.AfterWatermark(
            early=trigger.AfterProcessingTime(60),  # Emit every 60 sec
            late=trigger.AfterCount(1)  # Handle late data
        ),
        accumulation_mode=trigger.AccumulationMode.DISCARDING,
        allowed_lateness=300  # Accept data up to 5min late
    )
    | 'Aggregate' >> beam.CombineGlobally(RunningMeanVar())
)

# Use case: Real-time model monitoring
# - Early metrics every minute
# - Final metrics when watermark passes
# - Handle late-arriving predictions
```

### Watermarks and Late Data

**Watermark**: Estimate of event time progress
- **Event time**: When data was generated (e.g., image timestamp)
- **Processing time**: When data arrives at pipeline

```python
class ComputeAccuracyWithLateness(beam.DoFn):
    """Track on-time vs late data for monitoring."""

    def process(self, element, timestamp=beam.DoFn.TimestampParam,
                window=beam.DoFn.WindowParam):
        """
        Args:
            timestamp: Event time of element
            window: Window element belongs to
        """
        import time

        processing_time = time.time()
        event_time = timestamp.micros / 1e6

        lateness = processing_time - event_time

        yield {
            **element,
            'lateness_seconds': lateness,
            'window_start': window.start.micros / 1e6,
            'window_end': window.end.micros / 1e6,
            'is_late': lateness > 60  # Late if >1min
        }

# Configure window to handle late data
results = (
    predictions
    | 'Window' >> beam.WindowInto(
        window.FixedWindows(60),
        allowed_lateness=300,  # Accept data up to 5min late
        accumulation_mode=trigger.AccumulationMode.ACCUMULATING
    )
    | 'TrackLateness' >> beam.ParDo(ComputeAccuracyWithLateness())
    | 'ComputeMetrics' >> beam.CombinePerKey(MetricsCombineFn())
)

# Late data handling:
# - allowed_lateness=0: Drop late data (strict)
# - allowed_lateness=300: Accept data up to 5min late
# - accumulation_mode:
#   - DISCARDING: Each pane independent
#   - ACCUMULATING: Include previous pane data
```

**Example: Image preprocessing with late arrivals**

```python
preprocessed = (
    raw_images
    | 'AssignTimestamps' >> beam.Map(
        lambda x: beam.window.TimestampedValue(
            x,
            x['capture_timestamp']  # Use image capture time as event time
        )
    )
    | 'Window' >> beam.WindowInto(
        window.FixedWindows(600),  # 10-minute batches
        trigger=trigger.AfterWatermark(
            early=trigger.AfterProcessingTime(60),  # Preview every 1min
            late=trigger.AfterCount(10)  # Re-emit after 10 late arrivals
        ),
        allowed_lateness=1800,  # Accept images up to 30min late
        accumulation_mode=trigger.AccumulationMode.DISCARDING
    )
    | 'Preprocess' >> beam.ParDo(PreprocessImageFn())
    | 'Write' >> beam.io.WriteToTFRecord(
        'gs://bucket/preprocessed',
        file_name_suffix='.tfrecord'
    )
)

# Behavior:
# Time 10:00: Window [10:00-10:10] starts
# Time 10:01: Emit early pane (preview, incomplete data)
# Time 10:10: Watermark passes, emit on-time pane (most data)
# Time 10:15: Late image arrives from 10:05, re-emit pane
# Time 10:40: allowed_lateness expires, drop new late data
```

---

## Section 5: Cost Optimization

### Flex Templates

Reusable pipeline templates that can be launched without recompilation.

**Create Flex template:**

```python
# pipeline.py - Template code
def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transform_fn', required=True)
    args, pipeline_args = parser.parse_known_args(argv)

    options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | 'Read' >> beam.io.ReadFromTFRecord(args.input)
            | 'Transform' >> ApplyTransformFn(args.transform_fn)
            | 'Write' >> beam.io.WriteToTFRecord(args.output)
        )

if __name__ == '__main__':
    run()
```

**Build template:**

```bash
# Dockerfile for Flex template
FROM gcr.io/dataflow-templates-base/python39-template-launcher-base

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pipeline.py .

ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/template/pipeline.py"
ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE="/template/requirements.txt"
```

```bash
# Build and upload template
gcloud builds submit --tag gcr.io/PROJECT/ml-preprocessing:latest

# Create template metadata
gcloud dataflow flex-template build gs://bucket/templates/ml-preprocessing.json \
  --image gcr.io/PROJECT/ml-preprocessing:latest \
  --sdk-language PYTHON \
  --metadata-file metadata.json
```

**Launch from template:**

```python
from google.cloud import dataflow_v1beta3

client = dataflow_v1beta3.FlexTemplatesServiceClient()

request = dataflow_v1beta3.LaunchFlexTemplateRequest(
    project_id='my-project',
    location='us-central1',
    launch_parameter=dataflow_v1beta3.LaunchFlexTemplateParameter(
        job_name='ml-preprocessing-run-1',
        container_spec_gcs_path='gs://bucket/templates/ml-preprocessing.json',
        parameters={
            'input': 'gs://bucket/raw/*.tfrecord',
            'output': 'gs://bucket/preprocessed/',
            'transform_fn': 'gs://bucket/transform_fn'
        }
    )
)

response = client.launch_flex_template(request=request)
print(f'Job launched: {response.job.name}')
```

**Benefits:**
- **No recompilation**: Change parameters without rebuilding
- **Version control**: Pin template versions for reproducibility
- **Self-service**: Data scientists launch jobs without code access
- **Cost**: Template storage minimal (~MB)

### Streaming Engine

Offload worker state and shuffle to managed service for cost optimization.

```python
# Enable Streaming Engine
streaming_options = PipelineOptions()
streaming_options.view_as(StandardOptions).streaming = True
streaming_options.view_as(StandardOptions).enable_streaming_engine = True

# Cost comparison
# Without Streaming Engine:
# - n1-standard-4 workers: $0.19/hour × 10 workers = $1.90/hour
# - 100GB persistent disks: $0.04/GB/month × 10 = ~$0.05/hour
# Total: ~$1.95/hour

# With Streaming Engine:
# - n1-standard-2 workers: $0.095/hour × 10 workers = $0.95/hour (smaller!)
# - No persistent disks needed
# - Streaming Engine: $0.05/GB shuffled (~$0.30/hour for typical workload)
# Total: ~$1.25/hour (36% savings)
```

**When Streaming Engine saves money:**
- **High shuffle volume**: Offloading reduces worker memory needs
- **Stateful processing**: External state management allows smaller workers
- **Variable load**: Autoscale more aggressively with state managed externally

### Cost Optimization Strategies

**1. Right-size workers:**

```python
# Machine type selection based on workload
# CPU-intensive (image preprocessing): n1-highmem
# I/O-intensive (data parsing): n1-standard
# Mixed: n1-standard with autoscaling

# Cost per hour (us-central1):
# n1-standard-1: $0.0475
# n1-standard-4: $0.19
# n1-highmem-4: $0.237
# n1-highcpu-4: $0.142

worker_options.machine_type = 'n1-standard-4'  # Balance CPU/memory
```

**2. Use preemptible workers:**

```python
# 80% cost savings, but workers can be terminated
worker_options.use_public_ips = False  # Save egress costs
worker_options.num_workers = 10
worker_options.max_num_workers = 50

# Mix preemptible (cheap) and regular (reliable)
worker_options.num_workers = 2  # Regular workers (minimum capacity)
worker_options.max_num_workers = 50  # Scale with preemptible

# Cost: 2 regular + 48 preemptible vs 50 regular
# Regular: $0.19/hour × 50 = $9.50/hour
# Mixed: ($0.19 × 2) + ($0.038 × 48) = $2.20/hour (77% savings)
```

**3. Optimize shuffles:**

```python
# Avoid expensive GroupByKey when possible
# Bad: Full shuffle
word_counts = (
    words
    | 'AddKey' >> beam.Map(lambda w: (w, 1))
    | 'GroupByKey' >> beam.GroupByKey()  # EXPENSIVE SHUFFLE
    | 'Sum' >> beam.Map(lambda kv: (kv[0], sum(kv[1])))
)

# Good: Use Combine (optimized with combiner lifting)
word_counts = (
    words
    | 'Count' >> beam.combiners.Count.PerElement()  # Partial sums before shuffle
)
# 10x faster, 5x cheaper for large datasets
```

**4. Batch write operations:**

```python
# Bad: Write each element (expensive API calls)
_ = (
    data
    | 'Write' >> beam.Map(lambda x: bq_client.insert_row(x))  # 1M API calls
)

# Good: Batch writes
_ = (
    data
    | 'WriteBQ' >> beam.io.WriteToBigQuery(
        'dataset.table',
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        batch_size=1000  # Batch 1000 rows per API call (1000x fewer calls)
    )
)
```

From [Economize Cloud](https://www.economize.cloud/blog/google-dataflow/) (accessed 2025-11-16):
> "Dataflow can run your workload in parallel. When running Dataflow in Streaming mode, Dataflow runs one DoFn per thread... The ability to process data in batch and real-time streaming modes is provided by Dataflow."

---

## Section 6: Integration with Vertex AI Pipelines

### Dataflow Component

Vertex AI Pipelines provides a managed component for launching Dataflow jobs.

**DataflowPythonJobOp component:**

```python
from google_cloud_pipeline_components.v1.dataflow import DataflowPythonJobOp
from kfp import dsl

@dsl.pipeline(
    name='ml-preprocessing-pipeline',
    description='Preprocess images with Dataflow and train model'
)
def preprocessing_pipeline(
    project: str,
    region: str,
    input_data: str,
    output_data: str
):
    # Step 1: Dataflow preprocessing job
    dataflow_task = DataflowPythonJobOp(
        project=project,
        location=region,
        python_module_path='preprocessing/pipeline.py',
        temp_location=f'gs://{project}-dataflow/temp',
        requirements=[
            'tensorflow==2.15.0',
            'apache-beam[gcp]==2.53.0',
            'tensorflow-transform==1.15.0'
        ],
        args=[
            f'--input={input_data}',
            f'--output={output_data}',
            f'--project={project}',
            f'--region={region}',
            '--runner=DataflowRunner',
            '--max_num_workers=50',
            '--autoscaling_algorithm=THROUGHPUT_BASED'
        ]
    )

    # Step 2: Train model using preprocessed data
    from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

    training_task = CustomTrainingJobOp(
        project=project,
        location=region,
        display_name='train-model',
        worker_pool_specs=[{
            'machine_spec': {'machine_type': 'n1-standard-4'},
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'gcr.io/my-project/trainer:latest',
                'args': [
                    f'--training_data={output_data}',
                    '--epochs=100'
                ]
            }
        }]
    ).after(dataflow_task)  # Wait for preprocessing to complete

    return training_task

# Compile and run
from kfp import compiler
compiler.Compiler().compile(
    pipeline_func=preprocessing_pipeline,
    package_path='pipeline.json'
)

from google.cloud import aiplatform
aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.PipelineJob(
    display_name='ml-preprocessing-run',
    template_path='pipeline.json',
    parameter_values={
        'project': 'my-project',
        'region': 'us-central1',
        'input_data': 'gs://bucket/raw/images',
        'output_data': 'gs://bucket/preprocessed/images'
    }
)

job.run()
```

**WaitGcpResourcesOp for long-running jobs:**

```python
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp

# Dataflow job may run for hours
# WaitGcpResourcesOp polls job status, prevents pipeline timeout

dataflow_task = DataflowPythonJobOp(...)

wait_task = WaitGcpResourcesOp(
    gcp_resources=dataflow_task.outputs['gcp_resources']
).set_display_name('Wait for Dataflow job')

# Next steps only run after Dataflow completes
training_task = CustomTrainingJobOp(...).after(wait_task)
```

### End-to-End Pipeline Example

```python
@dsl.pipeline(name='image-classification-pipeline')
def full_ml_pipeline(
    project: str,
    region: str,
    raw_images: str,
    model_output: str
):
    """
    Complete ML pipeline:
    1. Preprocess images with Dataflow (TFT)
    2. Train model on Vertex AI
    3. Evaluate model
    4. Deploy to endpoint
    """

    # STEP 1: Dataflow preprocessing
    preprocess = DataflowPythonJobOp(
        project=project,
        location=region,
        python_module_path='preprocessing/tft_pipeline.py',
        temp_location=f'gs://{project}-tmp/dataflow',
        args=[
            f'--input={raw_images}',
            f'--output=gs://{project}-data/preprocessed',
            f'--transform_fn_output=gs://{project}-data/transform_fn',
            '--runner=DataflowRunner',
            '--max_num_workers=100',
            '--machine_type=n1-highmem-4',
            '--enable_streaming_engine'
        ]
    )

    wait_preprocess = WaitGcpResourcesOp(
        gcp_resources=preprocess.outputs['gcp_resources']
    )

    # STEP 2: Train model
    from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

    train = CustomTrainingJobOp(
        project=project,
        location=region,
        display_name='train-image-classifier',
        worker_pool_specs=[{
            'machine_spec': {
                'machine_type': 'n1-standard-8',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': 1
            },
            'replica_count': 1,
            'container_spec': {
                'image_uri': f'gcr.io/{project}/trainer:latest',
                'args': [
                    f'--training_data=gs://{project}-data/preprocessed/train',
                    f'--validation_data=gs://{project}-data/preprocessed/val',
                    f'--transform_fn=gs://{project}-data/transform_fn',
                    f'--model_output={model_output}',
                    '--epochs=50',
                    '--batch_size=64'
                ]
            }
        }]
    ).after(wait_preprocess)

    # STEP 3: Evaluate model
    from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationOp

    evaluate = ModelEvaluationOp(
        project=project,
        location=region,
        model=train.outputs['model'],
        test_data=f'gs://{project}-data/preprocessed/test',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    ).after(train)

    # STEP 4: Deploy model
    from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp

    deploy = ModelDeployOp(
        project=project,
        location=region,
        model=train.outputs['model'],
        deployed_model_display_name='image-classifier-v1',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=10,
        traffic_percentage=100
    ).after(evaluate)

    return deploy
```

From [Google Cloud Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/pipelines/dataflow-component) (accessed 2025-11-16):
> "The DataflowPythonJobOp operator lets you create a Vertex AI Pipelines component that prepares data by submitting a Python-based Apache Beam job to Dataflow."

---

## Section 7: arr-coc-0-1 Image Preprocessing Pipeline

### Vision Texture Preprocessing

Implementation of image preprocessing pipeline for arr-coc-0-1's 13-channel texture representation.

**Preprocessing function:**

```python
import tensorflow as tf
import tensorflow_transform as tft
import apache_beam as beam
import numpy as np

def preprocessing_fn(inputs):
    """
    Preprocess images for arr-coc-0-1 vision system.

    13-channel texture array:
    - RGB (3 channels)
    - LAB (3 channels)
    - Sobel edges (2 channels: horizontal, vertical)
    - Spatial coordinates (2 channels: x, y)
    - Eccentricity map (1 channel)
    - Frequency content (2 channels: low, high)

    Args:
        inputs: Dict with 'image' (raw bytes) and 'query' (text)

    Returns:
        Dict with preprocessed 13-channel texture array
    """
    outputs = {}

    # Decode image
    image = tf.io.decode_image(inputs['image'], channels=3)
    image = tf.cast(image, tf.float32)

    # Resize to standard size
    image = tf.image.resize(image, [224, 224])

    # CHANNEL 0-2: RGB normalized to [0, 1]
    rgb = image / 255.0
    outputs['rgb'] = rgb

    # CHANNEL 3-5: LAB color space
    # Convert RGB to LAB for perceptual uniformity
    lab = rgb_to_lab(rgb)  # Custom TF op
    outputs['lab'] = lab

    # CHANNEL 6-7: Sobel edge detection
    gray = tf.image.rgb_to_grayscale(image)
    sobel_x = tf.image.sobel_edges(tf.expand_dims(gray, 0))[0, :, :, 0, 0]
    sobel_y = tf.image.sobel_edges(tf.expand_dims(gray, 0))[0, :, :, 0, 1]
    outputs['sobel_x'] = tf.expand_dims(sobel_x, -1)
    outputs['sobel_y'] = tf.expand_dims(sobel_y, -1)

    # CHANNEL 8-9: Spatial coordinates (normalized)
    height, width = 224, 224
    x_coords = tf.range(width, dtype=tf.float32) / width
    y_coords = tf.range(height, dtype=tf.float32) / height
    x_grid = tf.tile(tf.reshape(x_coords, [1, -1, 1]), [height, 1, 1])
    y_grid = tf.tile(tf.reshape(y_coords, [-1, 1, 1]), [1, width, 1])
    outputs['spatial_x'] = x_grid
    outputs['spatial_y'] = y_grid

    # CHANNEL 10: Eccentricity map (distance from center)
    center_x, center_y = 112, 112
    x_dist = (tf.range(width, dtype=tf.float32) - center_x) / width
    y_dist = (tf.range(height, dtype=tf.float32) - center_y) / height
    x_dist_grid = tf.tile(tf.reshape(x_dist, [1, -1, 1]), [height, 1, 1])
    y_dist_grid = tf.tile(tf.reshape(y_dist, [-1, 1, 1]), [1, width, 1])
    eccentricity = tf.sqrt(x_dist_grid**2 + y_dist_grid**2)
    outputs['eccentricity'] = eccentricity

    # CHANNEL 11-12: Frequency content (FFT-based)
    # Low frequency: 0-10 cycles
    # High frequency: 10+ cycles
    freq_low, freq_high = compute_frequency_channels(gray)
    outputs['freq_low'] = tf.expand_dims(freq_low, -1)
    outputs['freq_high'] = tf.expand_dims(freq_high, -1)

    # Stack into 13-channel tensor
    texture_array = tf.concat([
        outputs['rgb'],           # 3 channels
        outputs['lab'],           # 3 channels
        outputs['sobel_x'],       # 1 channel
        outputs['sobel_y'],       # 1 channel
        outputs['spatial_x'],     # 1 channel
        outputs['spatial_y'],     # 1 channel
        outputs['eccentricity'],  # 1 channel
        outputs['freq_low'],      # 1 channel
        outputs['freq_high']      # 1 channel
    ], axis=-1)  # [224, 224, 13]

    # Normalize texture array per-channel
    # Use TFT to compute global statistics
    normalized_texture = tft.scale_to_z_score(texture_array)

    outputs['texture_array'] = normalized_texture

    # Process query text
    # Tokenize and encode query
    query_tokens = tf.strings.split(inputs['query'])
    outputs['query_encoded'] = tft.compute_and_apply_vocabulary(
        query_tokens,
        top_k=10000,
        num_oov_buckets=1,
        vocab_filename='query_vocab'
    )

    return outputs

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space."""
    # Simplified conversion (use proper color science)
    # L: Lightness
    # A: Green-Red
    # B: Blue-Yellow

    # This is a placeholder - implement proper RGB→XYZ→LAB
    l = tf.reduce_mean(rgb, axis=-1, keepdims=True)
    a = rgb[:, :, 0:1] - rgb[:, :, 1:2]  # R - G
    b = rgb[:, :, 2:3] - tf.reduce_mean(rgb[:, :, 0:2], axis=-1, keepdims=True)

    return tf.concat([l, a, b], axis=-1)

def compute_frequency_channels(gray):
    """Compute low and high frequency content using FFT."""
    # FFT to frequency domain
    gray_complex = tf.cast(gray, tf.complex64)
    fft = tf.signal.fft2d(gray_complex)
    fft_shifted = tf.signal.fftshift(fft)

    # Magnitude spectrum
    magnitude = tf.abs(fft_shifted)

    # Low-pass filter (center region)
    height, width = 224, 224
    center_y, center_x = height // 2, width // 2
    y_grid, x_grid = tf.meshgrid(
        tf.range(height, dtype=tf.float32),
        tf.range(width, dtype=tf.float32),
        indexing='ij'
    )
    distance = tf.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)

    low_mask = tf.cast(distance < 10, tf.float32)  # Low freq: radius < 10
    high_mask = 1.0 - low_mask

    freq_low = tf.reduce_mean(magnitude * low_mask)
    freq_high = tf.reduce_mean(magnitude * high_mask)

    # Normalize to image size
    freq_low = tf.fill([height, width], freq_low / (height * width))
    freq_high = tf.fill([height, width], freq_high / (height * width))

    return freq_low, freq_high
```

**Dataflow pipeline:**

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow_transform.beam as tft_beam

def run_preprocessing_pipeline():
    """
    Dataflow pipeline for arr-coc-0-1 image preprocessing.

    Workflow:
    1. Read raw images from GCS
    2. Apply TensorFlow Transform preprocessing
    3. Write 13-channel texture arrays
    4. Save transform artifacts for serving
    """

    options = PipelineOptions([
        '--project=arr-coc-vertex',
        '--region=us-west2',
        '--runner=DataflowRunner',
        '--staging_location=gs://arr-coc-data/staging',
        '--temp_location=gs://arr-coc-data/temp',
        '--max_num_workers=100',
        '--machine_type=n1-highmem-8',  # Memory for image processing
        '--disk_size_gb=200',
        '--autoscaling_algorithm=THROUGHPUT_BASED',
        '--enable_streaming_engine'
    ])

    # Define input schema
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import schema_utils

    raw_schema = schema_utils.schema_from_feature_spec({
        'image': tf.io.FixedLenFeature([], tf.string),
        'query': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
    })

    raw_metadata = dataset_metadata.DatasetMetadata(raw_schema)

    with beam.Pipeline(options=options) as p:
        with tft_beam.Context(temp_dir='gs://arr-coc-data/tft_tmp'):
            # Read training data
            raw_train = (
                p
                | 'ReadTrain' >> beam.io.ReadFromTFRecord(
                    'gs://arr-coc-data/raw/train/*.tfrecord'
                )
                | 'DecodeExamplesTrain' >> beam.Map(
                    tf.train.Example.FromString
                )
                | 'ParseTrain' >> beam.Map(
                    lambda x: parse_example(x, raw_metadata.schema)
                )
            )

            # Analyze and transform training data
            transformed_train, transform_fn = (
                (raw_train, raw_metadata)
                | 'AnalyzeAndTransformTrain' >> tft_beam.AnalyzeAndTransformDataset(
                    preprocessing_fn
                )
            )

            train_data, train_metadata = transformed_train

            # Write transformed training data
            _ = (
                train_data
                | 'EncodeTrain' >> beam.Map(
                    lambda x: encode_example(x, train_metadata.schema)
                )
                | 'WriteTrain' >> beam.io.WriteToTFRecord(
                    'gs://arr-coc-data/preprocessed/train',
                    file_name_suffix='.tfrecord',
                    num_shards=100  # Parallelize writes
                )
            )

            # Transform validation data (using saved statistics)
            raw_val = (
                p
                | 'ReadVal' >> beam.io.ReadFromTFRecord(
                    'gs://arr-coc-data/raw/val/*.tfrecord'
                )
                | 'DecodeExamplesVal' >> beam.Map(tf.train.Example.FromString)
                | 'ParseVal' >> beam.Map(
                    lambda x: parse_example(x, raw_metadata.schema)
                )
            )

            transformed_val = (
                (raw_val, raw_metadata), transform_fn
            ) | 'TransformVal' >> tft_beam.TransformDataset()

            val_data, val_metadata = transformed_val

            _ = (
                val_data
                | 'EncodeVal' >> beam.Map(
                    lambda x: encode_example(x, val_metadata.schema)
                )
                | 'WriteVal' >> beam.io.WriteToTFRecord(
                    'gs://arr-coc-data/preprocessed/val',
                    file_name_suffix='.tfrecord',
                    num_shards=20
                )
            )

            # Save transform function for serving
            _ = (
                transform_fn
                | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
                    'gs://arr-coc-data/transform_fn'
                )
            )

            # Write transform metadata
            _ = (
                train_metadata
                | 'WriteMetadata' >> tft_beam.WriteMetadata(
                    'gs://arr-coc-data/preprocessed/train_metadata',
                    pipeline=p
                )
            )

def parse_example(example, schema):
    """Parse TF Example using schema."""
    features = tf.io.parse_single_example(
        example.SerializeToString(),
        schema.as_feature_spec()
    )
    return features

def encode_example(features, schema):
    """Encode features back to TF Example."""
    example = tf.train.Example()
    for key, value in features.items():
        feature_spec = schema.as_feature_spec()[key]
        if isinstance(feature_spec, tf.io.FixedLenFeature):
            if feature_spec.dtype == tf.float32:
                example.features.feature[key].float_list.value.extend(
                    value.numpy().flatten()
                )
            elif feature_spec.dtype == tf.int64:
                example.features.feature[key].int64_list.value.extend(
                    value.numpy().flatten()
                )
    return example.SerializeToString()

if __name__ == '__main__':
    run_preprocessing_pipeline()
```

**Cost estimation:**

```
Pipeline: 100,000 images (224×224×3)
Workers: 100 n1-highmem-8 (autoscale from 10)
Duration: ~2 hours
Processing rate: ~800 images/second at peak

Costs:
- Compute: 100 workers × $0.474/hour × 2 hours = $94.80
- Streaming Engine: ~$0.05/GB shuffled × 50GB = $2.50
- Storage (temp): ~$0.02/GB × 100GB × 1 day = $0.0055
Total: ~$97.30 for 100K images = $0.97 per 1000 images

Optimization:
- Use preemptible workers: 77% savings → $22.42 total
- Batch smaller (10K images): More cost-effective amortization
- Reuse transform_fn: Only analyze once, transform many times
```

From arr-coc-0-1 project context:
> The 13-channel texture array provides rich perceptual information for relevance realization. Preprocessing with Dataflow ensures consistent feature extraction at both training and inference time, preventing subtle skews that could degrade query-aware compression.

---

## Sources

**Web Research:**

- [Apache Beam ML Preprocessing](https://beam.apache.org/documentation/ml/preprocess-data/) - Apache Beam documentation (accessed 2025-11-16)
- [TensorFlow Transform Guide](https://www.tensorflow.org/tfx/guide/transform) - TensorFlow documentation (accessed 2025-11-16)
- [Dataflow Horizontal Autoscaling](https://cloud.google.com/dataflow/docs/guides/tune-horizontal-autoscaling) - Google Cloud documentation (accessed 2025-11-16)
- [Vertex AI Dataflow Component](https://cloud.google.com/vertex-ai/docs/pipelines/dataflow-component) - Google Cloud documentation (accessed 2025-11-16)
- [Economize Cloud Dataflow Guide](https://www.economize.cloud/blog/google-dataflow/) - Economize Cloud blog (accessed 2025-11-16)

**Additional References:**

- Apache Beam Python SDK documentation
- TensorFlow Transform API reference
- Google Cloud Dataflow pricing documentation
- Vertex AI Pipelines component reference
