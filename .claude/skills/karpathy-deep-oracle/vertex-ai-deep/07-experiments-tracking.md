# Vertex AI Experiments - Tracking and Comparison

## Overview

Vertex AI Experiments is a comprehensive experiment tracking system that helps data scientists and ML engineers track, analyze, and compare different model architectures, hyperparameters, and training configurations. It provides a centralized platform for managing the entire ML experimentation lifecycle.

**Core Purpose**: Capture parameters, metrics, and artifacts from training runs to enable systematic comparison and reproducible ML development.

**Key Capabilities**:
- Parameter tracking (hyperparameters, configurations)
- Metrics logging (training/validation loss, accuracy, custom metrics)
- Artifact versioning (models, datasets, plots)
- Experiment comparison across runs
- TensorBoard integration for visualization
- Lineage tracking with ML Metadata

From [Introduction to Vertex AI Experiments](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) (accessed 2025-02-03):
> "Vertex AI Experiments is a tool that helps you track and analyze different model architectures, hyperparameters, and training environments."

---

## Section 1: Experiments API Architecture

### Core Concepts

**Experiment Hierarchy**:
```
Experiment (Project-level container)
  └── Experiment Run (Individual training execution)
      ├── Parameters (Configuration settings)
      ├── Metrics (Performance measurements)
      ├── Artifacts (Models, plots, datasets)
      └── Time Series Metrics (Training curves)
```

**Terminology** (from official docs):

- **Experiment**: A collection of related runs for systematic comparison
- **Experiment Run**: Single execution of a training script with specific parameters
- **Parameters**: Input configuration values (learning rate, batch size, model architecture)
- **Metrics**: Output performance measurements (loss, accuracy, custom scores)
- **Artifacts**: Files produced during training (model checkpoints, visualizations, datasets)

### Python SDK Integration

**Basic Setup**:
```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='your-project-id',
    location='us-central1'
)

# Create or get experiment
experiment = aiplatform.Experiment('my-experiment')

# Create experiment run
with aiplatform.start_run('run-001') as run:
    # Log parameters
    run.log_params({
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    })

    # Log metrics
    run.log_metrics({
        'final_accuracy': 0.95,
        'final_loss': 0.15
    })
```

**Context Manager Pattern** (recommended):
```python
with aiplatform.start_run('training-run-1') as run:
    # All logging happens within this context
    # Run automatically closes and saves metadata
    pass
```

From [Vertex AI SDK Documentation](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) (accessed 2025-02-03):
> "The Vertex AI SDK for Python provides APIs to consume experiments, experiment runs, experiment run parameters, metrics, and artifacts."

---

## Section 2: Parameter Tracking

### Parameter Types

**1. Scalar Parameters** (single values):
```python
run.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam',
    'dropout_rate': 0.3
})
```

**2. Structured Parameters** (nested configurations):
```python
run.log_params({
    'model_config': {
        'architecture': 'resnet50',
        'num_layers': 50,
        'pretrained': True
    },
    'training_config': {
        'lr_schedule': 'cosine',
        'warmup_epochs': 10
    }
})
```

**3. List/Array Parameters**:
```python
run.log_params({
    'layer_sizes': [512, 256, 128],
    'augmentation_ops': ['rotate', 'flip', 'brightness']
})
```

### Automatic Parameter Capture

**Autologging** (one-line setup):
```python
from google.cloud import aiplatform

# Enable autologging for supported frameworks
aiplatform.autolog()

# Now training code automatically logs:
# - Framework-specific hyperparameters
# - Model architecture details
# - Training configuration
```

From [Effortless tracking of your Vertex AI model training](https://cloud.google.com/blog/products/ai-machine-learning/effortless-tracking-of-your-vertex-ai-model-training) (accessed 2025-02-03):
> "With Vertex AI Experiments autologging, you can now log parameters, performance metrics and lineage artifacts by adding one line of code to your training script."

**Supported Frameworks**:
- TensorFlow/Keras
- PyTorch Lightning
- Scikit-learn
- XGBoost

### Parameter Search Integration

**Grid Search Tracking**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7]
}

for params in param_combinations:
    with aiplatform.start_run(f'run-{idx}') as run:
        run.log_params(params)
        # Train model with params
        score = train_and_evaluate(params)
        run.log_metrics({'cv_score': score})
```

**Hyperparameter Tuning Integration**:
- Experiments automatically capture Vertex AI Hyperparameter Tuning results
- Each trial becomes a separate experiment run
- Optimal parameters automatically identified

From [Track, compare, manage experiments with Vertex AI Experiments](https://cloud.google.com/blog/topics/developers-practitioners/track-compare-manage-experiments-vertex-ai-experiments) (accessed 2025-02-03):
> "The service enables you to track parameters, visualize and compare the performance metrics of your model and pipeline experiments."

---

## Section 3: Metrics Logging

### Metric Types

**1. Summary Metrics** (final values):
```python
run.log_metrics({
    'final_accuracy': 0.945,
    'final_loss': 0.082,
    'training_time_seconds': 3600,
    'validation_f1': 0.92
})
```

**2. Time Series Metrics** (training curves):
```python
# Log metrics at each epoch/step
for epoch in range(num_epochs):
    train_loss, val_loss = train_epoch()

    run.log_time_series_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': current_lr
    }, step=epoch)
```

**3. Custom Metrics** (domain-specific):
```python
run.log_metrics({
    'mean_average_precision': 0.87,
    'inference_latency_ms': 15.3,
    'memory_usage_mb': 2048,
    'data_augmentation_ratio': 0.5
})
```

### Automatic Metrics Logging

**Framework Integration**:
```python
# TensorFlow/Keras
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    callbacks=[
        aiplatform.callbacks.VertexAICallback()
    ]
)
# Automatically logs: loss, accuracy, val_loss, val_accuracy
```

**PyTorch Lightning**:
```python
from pytorch_lightning.loggers import VertexAILogger

trainer = pl.Trainer(
    logger=VertexAILogger(
        experiment_name='my-experiment',
        run_name='lightning-run-1'
    )
)
# Automatically logs training/validation metrics
```

### Metrics Comparison

**Comparing Across Runs**:
```python
# Get all runs in experiment
runs = aiplatform.Experiment('my-experiment').list_runs()

# Compare metrics
for run in runs:
    print(f"Run: {run.name}")
    print(f"Accuracy: {run.get_metrics()['final_accuracy']}")
    print(f"Loss: {run.get_metrics()['final_loss']}")
```

**UI Comparison** (Cloud Console):
- Select multiple runs in experiment view
- Compare side-by-side in table format
- Visualize metric trends across runs
- Filter/sort by any metric

From [Manually log data to an experiment run](https://docs.cloud.google.com/vertex-ai/docs/experiments/log-data) (accessed 2025-02-03):
> "To manually log data to an experiment run, you can use the Vertex AI SDK for Python. Supported metrics and parameters include summary metrics, time series metrics, and custom metrics."

---

## Section 4: Artifact Versioning

### Artifact Types

**1. Model Artifacts**:
```python
# Save and log model
model.save('model.h5')
run.log_artifact('model.h5', artifact_type='model')

# Or use ExperimentModel
from google.cloud.aiplatform import ExperimentModel

exp_model = ExperimentModel(
    model=model,
    parameters={'learning_rate': 0.001}
)
run.track_model(exp_model)
```

**2. Dataset Artifacts**:
```python
# Log training dataset metadata
run.log_artifact(
    'gs://bucket/train_data.csv',
    artifact_type='dataset',
    description='Training data v1.2'
)
```

**3. Visualization Artifacts**:
```python
import matplotlib.pyplot as plt

# Create plot
plt.plot(train_losses)
plt.savefig('loss_curve.png')

# Log plot
run.log_artifact('loss_curve.png', artifact_type='plot')
```

**4. Custom Artifacts**:
```python
# Log any file type
run.log_artifact(
    'config.yaml',
    artifact_type='configuration'
)

# Log HTML reports
run.log_artifact(
    'report.html',
    artifact_type='report'
)
```

### Model Registry Integration

**Register Best Model**:
```python
from google.cloud.aiplatform import ExperimentModel

# After training completes
with aiplatform.start_run('best-run') as run:
    run.log_metrics({'accuracy': 0.96})

    # Register model to Model Registry
    registered_model = run.register_experiment_model(
        model_name='my-classifier',
        model=model_artifact,
        description='Best performing model from experiment'
    )
```

**Version Management**:
```python
# Models automatically versioned in registry
# Version 1: my-classifier@1
# Version 2: my-classifier@2
# etc.

# Query versions
model_versions = aiplatform.Model.list(
    filter='display_name="my-classifier"'
)
```

From [Log models to an experiment run](https://docs.cloud.google.com/vertex-ai/docs/experiments/log-models-exp-run) (accessed 2025-02-03):
> "The register_experiment_model API enables registering the model that was deemed the best, in Vertex AI Model Registry with a minimum amount of configuration."

### Artifact Lineage

**ML Metadata Integration**:
```python
# Artifacts automatically tracked in ML Metadata
# Provides lineage graph:
# Dataset → Training Run → Model → Deployment

# Query lineage
from google.cloud import aiplatform_v1

metadata_client = aiplatform_v1.MetadataServiceClient()

# Get artifact lineage
lineage = metadata_client.query_artifact_lineage_subgraph(
    artifact='projects/.../artifacts/model-123'
)
```

**Lineage Benefits**:
- Track which dataset produced which model
- Identify models trained with specific hyperparameters
- Audit trail for compliance
- Reproducibility (reconstruct exact training conditions)

---

## Section 5: TensorBoard Integration

### Automatic TensorBoard Logging

**Setup with Custom Training**:
```python
import tensorflow as tf
from google.cloud import aiplatform

# Initialize experiment
aiplatform.init(
    experiment='my-experiment',
    experiment_tensorboard='projects/.../locations/.../tensorboards/123'
)

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='gs://bucket/logs',
    histogram_freq=1
)

# Train with automatic TensorBoard logging
with aiplatform.start_run('tb-run') as run:
    model.fit(
        x_train, y_train,
        callbacks=[tensorboard_callback]
    )
```

### Vertex AI TensorBoard Features

**Managed TensorBoard Instance**:
```bash
# Create TensorBoard instance
gcloud ai tensorboards create \
  --display-name="experiment-board" \
  --region=us-central1
```

**Integration Benefits**:
- Persistent TensorBoard hosting
- No local server required
- Shareable URLs for team collaboration
- Automatic experiment association

**Visualizations Available**:
- Scalars (loss, accuracy curves)
- Images (model predictions, feature maps)
- Histograms (weight distributions)
- Graphs (model architecture)
- Embeddings (high-dimensional data)
- Profiler (training performance)

From [Introduction to Vertex AI TensorBoard](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) (accessed 2025-02-03):
> "Integration with Vertex AI Experiments lets you: use a searchable and compare list of all experiments in a project, view time series metrics in the Google Cloud console."

### Multi-Run Comparison in TensorBoard

**Comparing Experiments**:
```python
# Each run logs to different TensorBoard directory
for run_name in ['run-1', 'run-2', 'run-3']:
    with aiplatform.start_run(run_name) as run:
        log_dir = f'gs://bucket/logs/{run_name}'
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

        model.fit(x, y, callbacks=[tb_callback])
```

**TensorBoard UI**:
- Overlay loss curves from multiple runs
- Compare hyperparameter effects visually
- Identify best performing configurations
- Detect overfitting across experiments

### Custom Scalars

**Log Custom Metrics to TensorBoard**:
```python
import tensorflow as tf

# Create summary writer
writer = tf.summary.create_file_writer('gs://bucket/logs')

with writer.as_default():
    for step in range(1000):
        # Log custom metrics
        tf.summary.scalar('custom_metric', value, step=step)
        tf.summary.scalar('learning_rate', lr, step=step)
```

---

## Python SDK Examples

### Complete Training Script with Experiments

```python
from google.cloud import aiplatform
import tensorflow as tf

def train_model(params):
    """Complete training with experiment tracking."""

    # Initialize Vertex AI
    aiplatform.init(
        project='my-project',
        location='us-central1',
        experiment='classifier-experiments',
        experiment_tensorboard='projects/.../tensorboards/123'
    )

    # Start experiment run
    with aiplatform.start_run(f'run-{params["run_id"]}') as run:

        # 1. Log parameters
        run.log_params(params)

        # 2. Build model
        model = create_model(params)

        # 3. Setup callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=f'gs://bucket/logs/{params["run_id"]}'
            ),
            aiplatform.callbacks.VertexAICallback()
        ]

        # 4. Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params['epochs'],
            callbacks=callbacks
        )

        # 5. Log final metrics
        run.log_metrics({
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1]
        })

        # 6. Save and log model artifact
        model.save('model.keras')
        run.log_artifact('model.keras', artifact_type='model')

        # 7. Log evaluation plots
        plot_confusion_matrix(model, val_dataset)
        run.log_artifact('confusion_matrix.png', artifact_type='plot')

        # 8. Register best model
        if history.history['val_accuracy'][-1] > 0.95:
            run.register_experiment_model(
                model_name='production-classifier',
                model='model.keras',
                description=f'High accuracy model from {params["run_id"]}'
            )

    return model

# Run multiple experiments
param_grid = [
    {'run_id': 'exp1', 'learning_rate': 0.001, 'epochs': 50},
    {'run_id': 'exp2', 'learning_rate': 0.01, 'epochs': 50},
    {'run_id': 'exp3', 'learning_rate': 0.001, 'epochs': 100}
]

for params in param_grid:
    train_model(params)
```

### Query and Compare Experiments

```python
from google.cloud import aiplatform

# Get experiment
experiment = aiplatform.Experiment('classifier-experiments')

# List all runs
runs = experiment.list_runs()

# Find best run by accuracy
best_run = max(
    runs,
    key=lambda r: r.get_metrics().get('final_val_acc', 0)
)

print(f"Best run: {best_run.name}")
print(f"Parameters: {best_run.get_params()}")
print(f"Accuracy: {best_run.get_metrics()['final_val_acc']}")

# Compare multiple runs
comparison_df = experiment.get_data_frame()
print(comparison_df[['run_name', 'learning_rate', 'final_val_acc']])
```

### Integration with Custom Training Jobs

```python
from google.cloud import aiplatform

# Submit custom training job with experiment tracking
job = aiplatform.CustomTrainingJob(
    display_name='experiment-training-job',
    script_path='train.py',
    container_uri='gcr.io/project/trainer:latest'
)

# Run with experiment context
with aiplatform.start_run('custom-job-run') as run:
    model = job.run(
        replica_count=1,
        machine_type='n1-highmem-8',
        accelerator_type='NVIDIA_TESLA_V100',
        accelerator_count=1,
        environment_variables={
            'EXPERIMENT_RUN': run.name
        }
    )

    # Metrics logged from training script automatically captured
```

From [Run training job with experiment tracking](https://docs.cloud.google.com/vertex-ai/docs/experiments/run-training-job-experiments) (accessed 2025-02-03):
> "Vertex AI SDK for Python enables experiment tracking, which captures parameters and performance metrics when you submit a custom training job."

---

## Best Practices

### 1. Naming Conventions

**Experiments**:
- Use descriptive names: `bert-finetuning-experiments`
- Include model type and goal
- Avoid generic names like `experiment-1`

**Runs**:
- Include timestamp: `run-2025-02-03-14-30`
- Include key parameter: `lr-0.001-batch-32`
- Be consistent across project

### 2. Parameter Organization

**Structured Logging**:
```python
run.log_params({
    'model': {
        'architecture': 'resnet50',
        'pretrained': True
    },
    'training': {
        'optimizer': 'adam',
        'learning_rate': 0.001
    },
    'data': {
        'batch_size': 32,
        'augmentation': True
    }
})
```

### 3. Metric Selection

**Choose Meaningful Metrics**:
- Primary metric (e.g., validation accuracy)
- Secondary metrics (training loss, inference time)
- Domain-specific metrics (mAP, F1, BLEU)

**Avoid Metric Overload**:
- Log 5-10 key metrics per run
- Use time series for training curves
- Use summary metrics for final values

### 4. Artifact Management

**Storage Best Practices**:
- Save large artifacts to GCS, log GCS path
- Use versioned artifact names: `model-v1.keras`
- Clean up old artifacts periodically

### 5. Experiment Lifecycle

**Development Phase**:
- Many experiments, rapid iteration
- Log all parameters, even if they don't change

**Production Phase**:
- Fewer experiments, careful validation
- Register best models to Model Registry
- Maintain experiment history for audit

---

## Common Use Cases

### 1. Hyperparameter Tuning

Track systematic parameter searches:
```python
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        with aiplatform.start_run(f'lr-{lr}-bs-{bs}') as run:
            run.log_params({'learning_rate': lr, 'batch_size': bs})
            score = train_and_evaluate(lr, bs)
            run.log_metrics({'accuracy': score})
```

### 2. Model Architecture Search

Compare different architectures:
```python
architectures = ['resnet50', 'efficientnet-b0', 'mobilenet-v2']

for arch in architectures:
    with aiplatform.start_run(f'arch-{arch}') as run:
        run.log_params({'architecture': arch})
        model = build_model(arch)
        results = train_and_evaluate(model)
        run.log_metrics(results)
```

### 3. Data Ablation Studies

Test impact of data changes:
```python
data_versions = ['v1-baseline', 'v2-augmented', 'v3-filtered']

for version in data_versions:
    with aiplatform.start_run(f'data-{version}') as run:
        run.log_params({'data_version': version})
        dataset = load_dataset(version)
        results = train_and_evaluate(dataset)
        run.log_metrics(results)
```

### 4. Reproducibility

Capture complete training environment:
```python
import sys

run.log_params({
    'python_version': sys.version,
    'framework_versions': {
        'tensorflow': tf.__version__,
        'numpy': np.__version__
    },
    'random_seed': 42,
    'data_hash': hash_dataset(train_data)
})
```

From [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) (Google Cloud Blog, accessed 2025-02-03):
> "Vertex AI Pipelines has a deep integration of experiments. You can track metrics, parameters, and artifacts for each of the components of your pipeline runs."

---

## Troubleshooting

### Common Issues

**1. Experiment Not Found**:
```python
# Initialize project first
aiplatform.init(project='my-project', location='us-central1')

# Then create/get experiment
experiment = aiplatform.Experiment('my-experiment')
```

**2. Metrics Not Appearing**:
```python
# Ensure run is properly closed
with aiplatform.start_run('my-run') as run:
    run.log_metrics({'accuracy': 0.95})
# Metrics saved when context exits

# Or manually end run
run = aiplatform.start_run('my-run')
run.log_metrics({'accuracy': 0.95})
run.end_run()  # Don't forget this!
```

**3. Artifact Upload Failures**:
```python
# Ensure file exists before logging
import os
assert os.path.exists('model.h5'), "Model file not found"

# Use absolute paths
run.log_artifact(os.path.abspath('model.h5'))
```

### Debugging Tips

**Check Experiment State**:
```python
# List all experiments
experiments = aiplatform.Experiment.list()

# List runs in experiment
runs = experiment.list_runs()
print(f"Found {len(runs)} runs")

# Check run metadata
run = runs[0]
print(f"Parameters: {run.get_params()}")
print(f"Metrics: {run.get_metrics()}")
```

---

## Cost Considerations

**Experiments Pricing**:
- Experiment tracking: **Free**
- TensorBoard hosting: **$0.30/hour** (when active)
- Artifact storage: Standard GCS pricing
- No charge for API calls

**Cost Optimization**:
- Use TensorBoard only when needed (stop instances)
- Clean up old experiments/artifacts
- Compress large artifacts before logging
- Use GCS lifecycle policies for artifact cleanup

---

## Sources

**Official Documentation:**
- [Introduction to Vertex AI Experiments](https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) (accessed 2025-02-03)
- [Manually log data to an experiment run](https://docs.cloud.google.com/vertex-ai/docs/experiments/log-data) (accessed 2025-02-03)
- [Run training job with experiment tracking](https://docs.cloud.google.com/vertex-ai/docs/experiments/run-training-job-experiments) (accessed 2025-02-03)
- [Log models to an experiment run](https://docs.cloud.google.com/vertex-ai/docs/experiments/log-models-exp-run) (accessed 2025-02-03)
- [Introduction to Vertex AI TensorBoard](https://docs.cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) (accessed 2025-02-03)

**Google Cloud Blog Posts:**
- [Track, compare, manage experiments with Vertex AI Experiments](https://cloud.google.com/blog/topics/developers-practitioners/track-compare-manage-experiments-vertex-ai-experiments) (July 13, 2022, accessed 2025-02-03)
- [Effortless tracking of your Vertex AI model training](https://cloud.google.com/blog/products/ai-machine-learning/effortless-tracking-of-your-vertex-ai-model-training) (April 4, 2023, accessed 2025-02-03)

**Community Resources:**
- [ML Experiment Tracking with Vertex AI](https://medium.com/google-cloud/ml-experiment-tracking-with-vertex-ai-8406f8d44376) - Medium article by Sascha Heyer (accessed 2025-02-03)
- [Manage Machine Learning Experiments with Vertex AI](https://codelabs.developers.google.com/vertex_experiments_pipelines_intro) - Google Codelab (January 20, 2023, accessed 2025-02-03)

**SDK Reference:**
- [google-cloud-aiplatform PyPI Package](https://pypi.org/project/google-cloud-aiplatform/) (accessed 2025-02-03)
