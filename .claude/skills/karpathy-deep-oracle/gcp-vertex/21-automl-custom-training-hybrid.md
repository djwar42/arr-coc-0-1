# AutoML & Custom Training Hybrid Strategies

## Overview

AutoML and custom training represent two distinct approaches to machine learning on Vertex AI, each with specific strengths, limitations, and appropriate use cases. Understanding when to use AutoML, when to graduate to custom training, and how to leverage both in a hybrid pipeline is critical for efficient ML development and cost optimization.

This guide covers AutoML capabilities across data types, model export strategies, warm-starting custom training from AutoML baselines, decision frameworks for budget allocation, and graduation criteria for moving beyond AutoML to full custom architectures.

From [Frozen Backbone + Adapter Training](../practical-implementation/46-frozen-backbone-adapter-training.md):
> "Building on unimodal pre-trained backbones rather than training entirely new models from scratch" - this principle applies equally to AutoML → custom training transitions where AutoML provides the baseline.

---

## Section 1: AutoML Capabilities Across Data Types

### 1.1 AutoML Tables (Tabular Data)

**Supported Tasks:**
- Regression (predict continuous values)
- Classification (binary and multi-class)
- Forecasting (time series prediction)

**Key Features:**
- Automatic feature engineering and selection
- Handles missing data and outliers
- Supports 1,000+ features and millions of rows
- Built-in categorical encoding
- Automatic data split (80% train / 10% validation / 10% test by default)

From [Hands-Off Machine Learning with Google AutoML](https://towardsdatascience.com/hands-off-machine-learning-with-google-automl-e63b079f09d1) (accessed 2025-11-16):
> "AutoML Tabular enables developers (with limited ML expertise) to train high-quality models specific to their needs"

**Optimization Objectives:**
- Regression: RMSE, MAE, RMSLE
- Classification: AUC-ROC, AUC-PR, Log Loss, Accuracy
- Forecasting: RMSE, MAE, MAPE

**Example Use Case:**
```python
# California housing price prediction
# Dataset: 20,640 rows, 9 features
# Target: median_house_value
# AutoML Result: RMSE ~48,000 (competitive with manual models)
```

### 1.2 AutoML Vision (Image Data)

**Supported Tasks:**
- Image classification (single-label and multi-label)
- Object detection (bounding boxes)
- Image segmentation (pixel-level masks)

**Requirements:**
- Minimum: 100 images per label for classification
- Recommended: 1,000+ images per label
- Supported formats: JPEG, PNG, GIF, BMP, ICO
- Max file size: 30 MB per image
- Max resolution: Limited by training budget

From [AutoML in Vertex AI: Understanding the Relationship](https://promevo.com/blog/automl-in-vertex-ai) (accessed 2025-11-16):
> "AutoML can handle different data types, like images, text, and tabular data, allowing you to build models for diverse tasks"

**Automatic Augmentation:**
- Random crops and flips
- Color distortion
- Rotation (configurable)
- Brightness/contrast adjustments

**Pre-trained Backbones:**
- EfficientNet family (B0-B7)
- ResNet variants
- Vision Transformer (ViT) for newer models

### 1.3 AutoML Natural Language (Text Data)

**Supported Tasks:**
- Text classification (sentiment, topic, intent)
- Entity extraction (NER)
- Sentiment analysis (document-level)

**Input Formats:**
- Plain text (UTF-8)
- CSV with text column
- JSON Lines
- Max document length: 128,000 characters

**Language Support:**
- 100+ languages
- Multilingual models available
- Language-specific embeddings

**Pre-trained Models:**
- BERT and BERT variants
- mBERT (multilingual)
- DistilBERT for faster inference

### 1.4 AutoML Video Intelligence

**Supported Tasks:**
- Video classification (action recognition)
- Object tracking
- Shot change detection

**Requirements:**
- Minimum: 10 hours of labeled video
- Recommended: 50+ hours for production quality
- Supported formats: MP4, AVI, MOV, FLV
- Max file size: 5 GB per video
- Frame rate: 10-30 FPS recommended

**Temporal Features:**
- Automatic frame sampling
- Temporal convolution
- Optical flow integration

---

## Section 2: Exporting AutoML Models

### 2.1 TensorFlow SavedModel Export

**Export Process:**

From [Hands-Off Machine Learning with Google AutoML](https://towardsdatascience.com/hands-off-machine-learning-with-google-automl-e63b079f09d1) (accessed 2025-11-16):
> "The trained model that is produced by AutoML can then be deployed in two ways; we can export the model as a saved TensorFlow (TF) model which we can then serve ourselves in a Docker container"

**Export via Console:**
1. Navigate to Vertex AI > Models
2. Select trained AutoML model
3. Click "Export Model"
4. Choose "TensorFlow SavedModel" format
5. Specify GCS destination bucket

**Export via gcloud CLI:**
```bash
# Export AutoML Tables model to TensorFlow SavedModel
gcloud ai models export \
  --model=projects/PROJECT_ID/locations/REGION/models/MODEL_ID \
  --export-format-id=tf-saved-model \
  --output-gcs-uri=gs://BUCKET/automl_export/
```

**Exported Artifacts:**
```
gs://BUCKET/automl_export/
├── saved_model.pb          # Model graph
├── variables/
│   ├── variables.data-*    # Model weights
│   └── variables.index
└── assets/                 # Vocabulary, metadata
    └── feature_metadata.json
```

**Loading Exported Model:**
```python
import tensorflow as tf

# Load exported SavedModel
model = tf.saved_model.load('gs://BUCKET/automl_export/')

# Serving signature
serving_fn = model.signatures['serving_default']

# Inference
predictions = serving_fn(input_tensor)
```

### 2.2 ONNX Format Export

**Conversion via tf2onnx:**

From web research on AutoML model export (accessed 2025-11-16):
- AutoML models can be exported to ONNX via TensorFlow SavedModel intermediate format
- Use `tf2onnx` tool for conversion

**Conversion Process:**
```bash
# Step 1: Export to TensorFlow SavedModel (via Console or gcloud)

# Step 2: Convert to ONNX
python -m tf2onnx.convert \
  --saved-model gs://BUCKET/automl_export/ \
  --output automl_model.onnx \
  --opset 13
```

**ONNX Benefits:**
- Cross-framework compatibility (PyTorch, TensorRT, ONNX Runtime)
- Optimized inference on edge devices
- Simplified deployment pipelines
- Mobile and embedded support

**ONNX Runtime Inference:**
```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("automl_model.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
outputs = session.run([output_name], {input_name: input_data})
```

### 2.3 Container Export for Custom Serving

**Vertex AI Container Format:**
- Includes TensorFlow Serving pre-configured
- Pre-built Docker image with model
- Auto-scaling support
- Health checks included

**Export to Container:**
```bash
# Export AutoML model to container
gcloud ai models export \
  --model=MODEL_ID \
  --export-format-id=tf-serving \
  --artifact-destination=gs://BUCKET/container_export/

# Artifact includes:
# - Model files
# - Dockerfile
# - serving_config.yaml
```

**Custom Serving Deployment:**
```yaml
# serving_config.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: automl-predictor
spec:
  predictor:
    tensorflow:
      storageUri: gs://BUCKET/container_export/
      resources:
        limits:
          memory: 4Gi
          cpu: "2"
```

### 2.4 Edge TPU Export (Vision Models Only)

**Edge TPU Compilation:**
- Available for AutoML Vision models
- Optimized for Coral Edge TPU devices
- Quantized INT8 models
- Latency: <10ms per image

**Export Process:**
```bash
# Export AutoML Vision to Edge TPU
gcloud ai models export \
  --model=VISION_MODEL_ID \
  --export-format-id=edgetpu-tflite \
  --output-gcs-uri=gs://BUCKET/edgetpu_export/
```

---

## Section 3: Warm-Starting Custom Training from AutoML

### 3.1 Transfer Learning from AutoML Checkpoints

**Concept:**

From [Transfer Learning with AutoML warm start](https://github.com/automl/transfer-hpo-framework) (accessed 2025-11-16):
> "Warm start uses previous tuning job results to inform new hyperparameter searches, making the search more efficient"

**AutoML as Baseline:**
1. Train AutoML model to convergence
2. Export model weights and architecture
3. Load in custom training code
4. Fine-tune with custom objectives/data

**Example: AutoML Vision → Custom PyTorch Training:**
```python
import tensorflow as tf
import torch

# Step 1: Load AutoML exported SavedModel
automl_model = tf.saved_model.load('gs://BUCKET/automl_export/')

# Step 2: Extract weights from TensorFlow
tf_weights = {}
for var in automl_model.trainable_variables:
    tf_weights[var.name] = var.numpy()

# Step 3: Convert to PyTorch
def tf_to_pytorch_weights(tf_weights):
    """Convert TensorFlow weights to PyTorch format"""
    pt_weights = {}
    for name, value in tf_weights.items():
        # Handle transposed conv weights
        if 'kernel' in name and len(value.shape) == 4:
            value = value.transpose(3, 2, 0, 1)  # HWIO → OIHW
        pt_name = name.replace('/', '.').replace(':0', '')
        pt_weights[pt_name] = torch.from_numpy(value)
    return pt_weights

pt_weights = tf_to_pytorch_weights(tf_weights)

# Step 4: Load into custom PyTorch model
import torchvision.models as models

custom_model = models.efficientnet_b0(pretrained=False)
custom_model.load_state_dict(pt_weights, strict=False)

# Step 5: Fine-tune with custom loss/data
optimizer = torch.optim.AdamW(custom_model.parameters(), lr=1e-4)
custom_loss = CustomContrastiveLoss()

for epoch in range(10):
    for batch in custom_dataloader:
        loss = custom_loss(custom_model(batch['input']), batch['target'])
        loss.backward()
        optimizer.step()
```

### 3.2 Hyperparameter Warm-Starting

From [AWS warm-start hyperparameter tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html) (accessed 2025-11-16):
> "Transfer learning warm starts a tuning job with the evaluations from similar tasks, allowing both search space, algorithm image and dataset change"

**AutoML Hyperparameters as Priors:**

**Extract AutoML Config:**
```python
from google.cloud import aiplatform

# Get AutoML model metadata
model = aiplatform.Model('projects/PROJECT/locations/REGION/models/MODEL_ID')
metadata = model.gca_resource.metadata

# Extract training config
automl_config = {
    'learning_rate': metadata.get('learningRate', 0.001),
    'batch_size': metadata.get('batchSize', 32),
    'epochs': metadata.get('epochs', 100),
    'optimizer': metadata.get('optimizer', 'adam'),
    'weight_decay': metadata.get('weightDecay', 0.0001)
}

# Use as starting point for Vertex AI Vizier
from google.cloud import aiplatform_v1

study_config = aiplatform_v1.StudySpec(
    metrics=[aiplatform_v1.StudySpec.MetricSpec(
        metric_id='accuracy',
        goal=aiplatform_v1.StudySpec.MetricSpec.GoalType.MAXIMIZE
    )],
    parameters=[
        aiplatform_v1.StudySpec.ParameterSpec(
            parameter_id='learning_rate',
            double_value_spec=aiplatform_v1.StudySpec.ParameterSpec.DoubleValueSpec(
                min_value=automl_config['learning_rate'] * 0.1,
                max_value=automl_config['learning_rate'] * 10.0
            ),
            scale_type=aiplatform_v1.StudySpec.ParameterSpec.ScaleType.LOG
        ),
        # ... other parameters based on AutoML config
    ]
)
```

### 3.3 Feature Engineering Transfer

**AutoML Feature Transformations:**

AutoML automatically applies:
- Categorical encoding (one-hot, target encoding, embeddings)
- Numerical normalization (z-score, min-max)
- Missing value imputation
- Feature crosses (limited)

**Exporting Feature Transform Logic:**
```python
# AutoML Tables feature metadata
feature_metadata = model.gca_resource.metadata['feature_metadata']

# Example feature transform spec
for feature in feature_metadata:
    print(f"Feature: {feature['name']}")
    print(f"  Type: {feature['type']}")  # CATEGORICAL, NUMERICAL, TIMESTAMP
    print(f"  Transform: {feature['transform']}")  # ONE_HOT, Z_SCORE, etc.
    print(f"  Stats: {feature.get('stats', {})}")  # mean, std, vocab
```

**Reusing in Custom Preprocessing:**
```python
import tensorflow as tf

def create_preprocessing_layer_from_automl(feature_metadata):
    """Create tf.keras preprocessing layers matching AutoML transforms"""

    preprocessing_layers = {}

    for feature in feature_metadata:
        name = feature['name']
        feature_type = feature['type']
        transform = feature['transform']

        if feature_type == 'NUMERICAL' and transform == 'Z_SCORE':
            # Use AutoML's computed mean and std
            mean = feature['stats']['mean']
            std = feature['stats']['std']
            preprocessing_layers[name] = tf.keras.layers.Normalization(
                mean=mean, variance=std**2
            )

        elif feature_type == 'CATEGORICAL' and transform == 'ONE_HOT':
            vocab = feature['stats']['vocabulary']
            preprocessing_layers[name] = tf.keras.layers.StringLookup(
                vocabulary=vocab, output_mode='one_hot'
            )

    return preprocessing_layers
```

### 3.4 Architecture Transfer for Vision Models

From [Frozen Backbone + Adapter Training](../practical-implementation/46-frozen-backbone-adapter-training.md):
> "Freezing pre-trained backbone models during vision-language model (VLM) training has become the dominant approach in modern multimodal AI"

**AutoML Vision Backbone Extraction:**
```python
# AutoML Vision typically uses EfficientNet or ResNet

# Export full model
# Then extract backbone only (without classification head)

def extract_automl_backbone(saved_model_path):
    """Extract feature extractor from AutoML Vision model"""

    model = tf.saved_model.load(saved_model_path)

    # Get intermediate layer outputs
    # AutoML Vision models have predictable structure
    backbone = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('efficientnet_b0').output  # Feature layer
    )

    return backbone

# Use in custom model
automl_backbone = extract_automl_backbone('gs://BUCKET/automl_export/')

# Freeze AutoML backbone
for layer in automl_backbone.layers:
    layer.trainable = False

# Add custom head
custom_model = tf.keras.Sequential([
    automl_backbone,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Train only custom head initially
custom_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Section 4: Budget Allocation Decision Tree (AutoML vs Custom)

### 4.1 Cost Comparison Framework

**AutoML Pricing (Vertex AI):**
- Tables: $19.32 per node hour (training), $0.0648 per node hour (prediction)
- Vision: $3.15 per node hour (training), $1.51 per node hour (prediction)
- NLP: $3.15 per node hour (training), $0.50 per node hour (prediction)
- Video: $3.465 per node hour (training), $1.71 per node hour (prediction)

**Custom Training Pricing:**
- n1-standard-4 (4 vCPUs, 15 GB): $0.19/hour
- n1-standard-16 (16 vCPUs, 60 GB): $0.76/hour
- NVIDIA T4 GPU: +$0.35/hour
- NVIDIA V100 GPU: +$2.48/hour
- NVIDIA A100 GPU: +$3.67/hour

**Training Budget Example (Image Classification):**

Scenario: 10,000 images, 100 classes, target 95% accuracy

**AutoML Vision:**
- Recommended training budget: 8 node hours
- Cost: 8 × $3.15 = $25.20
- Development time: ~30 minutes (setup + training)
- Total cost (including data scientist time): ~$100

**Custom Training:**
- Instance: n1-standard-16 + V100 GPU
- Estimated hours: 20 hours (including experimentation)
- Compute cost: 20 × ($0.76 + $2.48) = $64.80
- Development time: ~40 hours (data prep + model dev + tuning)
- Total cost (including data scientist time @ $100/hr): $4,064.80

**Decision Criteria:**
```
Use AutoML if:
  - Time to deployment < 1 week
  - Development budget < $500
  - Team ML expertise < 2 years average
  - Data preprocessing is standard
  - Model architecture flexibility not required

Use Custom Training if:
  - Novel architecture needed
  - Specific optimization constraints (latency, model size)
  - Integration with existing custom pipeline
  - Fine-grained control over training process required
  - Long-term iterative development expected
```

### 4.2 Decision Tree Flowchart

```
                      Start ML Project
                            |
                            v
                   Data Type Standard?
                   (Image/Text/Tabular/Video)
                     /              \
                   Yes               No → Custom Training
                    |
                    v
            Data Volume Sufficient?
            (>1000 samples per class)
               /           \
             Yes            No → Data Augmentation
              |                  or Custom Semi-Supervised
              v
      Standard ML Objective?
      (Classification/Regression)
         /              \
       Yes               No → Custom Training
        |                    (Multi-task, Custom Loss)
        v
  Time to Deploy < 2 weeks?
       /          \
     Yes           No → Consider Custom
      |                 (if iteration needed)
      v
  **Use AutoML**
      |
      v
  AutoML Performance Acceptable?
  (Meets accuracy/latency targets)
       /              \
     Yes               No
      |                 |
      v                 v
  **Deploy AutoML**  **Graduate to Custom**
                     (See Section 5)
```

### 4.3 ROI Calculation Template

```python
def calculate_ml_roi(
    automl_training_hours,
    automl_hourly_rate,
    custom_training_hours,
    custom_compute_rate,
    dev_time_automl_hours,
    dev_time_custom_hours,
    data_scientist_hourly_rate=100,
    expected_model_lifetime_months=12,
    prediction_requests_per_month=1_000_000
):
    """
    Calculate ROI for AutoML vs Custom Training decision

    Returns: dict with cost breakdown and recommendation
    """

    # AutoML costs
    automl_training_cost = automl_training_hours * automl_hourly_rate
    automl_dev_cost = dev_time_automl_hours * data_scientist_hourly_rate
    automl_total_dev = automl_training_cost + automl_dev_cost

    # Custom training costs
    custom_training_cost = custom_training_hours * custom_compute_rate
    custom_dev_cost = dev_time_custom_hours * data_scientist_hourly_rate
    custom_total_dev = custom_training_cost + custom_dev_cost

    # Ongoing serving costs (simplified)
    automl_serving_monthly = prediction_requests_per_month * 0.00005  # Estimate
    custom_serving_monthly = prediction_requests_per_month * 0.00002  # Cheaper

    # Total cost of ownership
    automl_tco = automl_total_dev + (automl_serving_monthly * expected_model_lifetime_months)
    custom_tco = custom_total_dev + (custom_serving_monthly * expected_model_lifetime_months)

    return {
        'automl_dev_cost': automl_total_dev,
        'custom_dev_cost': custom_total_dev,
        'automl_tco': automl_tco,
        'custom_tco': custom_tco,
        'cost_difference': custom_tco - automl_tco,
        'recommendation': 'AutoML' if automl_tco < custom_tco else 'Custom Training',
        'break_even_months': (custom_total_dev - automl_total_dev) /
                             (automl_serving_monthly - custom_serving_monthly)
                             if automl_serving_monthly > custom_serving_monthly else None
    }

# Example usage
roi = calculate_ml_roi(
    automl_training_hours=8,
    automl_hourly_rate=3.15,
    custom_training_hours=20,
    custom_compute_rate=3.24,  # n1-standard-16 + V100
    dev_time_automl_hours=4,
    dev_time_custom_hours=40
)

print(f"Recommendation: {roi['recommendation']}")
print(f"Cost difference: ${roi['cost_difference']:.2f}")
```

---

## Section 5: When to Graduate from AutoML (Custom Architecture Needs)

### 5.1 Graduation Triggers

**Technical Limitations:**

1. **Custom Loss Functions Required**
   - AutoML supports standard losses only (cross-entropy, MSE, MAE)
   - Need for contrastive loss, triplet loss, focal loss, etc.
   - Multi-task learning with weighted losses

2. **Novel Architecture Requirements**
   - Attention mechanisms not available in AutoML
   - Graph neural networks
   - Transformer variants beyond BERT
   - Custom layer types

3. **Fine-Grained Training Control**
   - Learning rate schedules beyond cosine/exponential
   - Gradient clipping strategies
   - Mixed precision training optimization
   - Custom regularization techniques

4. **Integration Requirements**
   - Need to integrate with existing custom pipeline
   - Specific framework requirements (PyTorch vs TensorFlow)
   - On-premise deployment constraints
   - Real-time serving with custom preprocessing

From web research on "when to graduate from AutoML" (accessed 2025-11-16):
> "Most AutoML users have 1-2 years or less than a year of machine learning experience" - as teams mature, custom training becomes more valuable

**Performance Plateaus:**

Graduated when:
- AutoML achieves 85% of target accuracy but can't improve further
- Manual experimentation shows 5-10% accuracy gains possible
- Inference latency 2-3x too slow for production requirements
- Model size exceeds deployment constraints (edge devices)

### 5.2 Graduation Strategies

**Strategy 1: Incremental Migration**

Phase 1: AutoML baseline (Week 1-2)
- Train AutoML model
- Establish baseline metrics
- Deploy to production (if acceptable)
- Use as benchmark

Phase 2: Export + Reproduce (Week 3-4)
- Export AutoML model
- Reproduce results in custom training code
- Validate metric parity
- Test serving infrastructure

Phase 3: Targeted Improvements (Week 5-8)
- Identify specific bottlenecks
- Implement custom improvements incrementally
- A/B test against AutoML baseline
- Iterate based on results

Phase 4: Full Custom (Week 9+)
- Complete custom training pipeline
- Advanced hyperparameter tuning
- Production deployment
- Continuous monitoring

**Strategy 2: Hybrid Pipeline**

From Section 3 (warm-starting):
- Use AutoML for initial exploration and baseline
- Export best model as feature extractor
- Train custom head/decoder on top
- Gradually unfreeze and fine-tune

**Example: AutoML → Custom Hybrid for arr-coc-0-1:**

```python
"""
Hybrid approach: AutoML Vision baseline → Custom ARR-COC optimization

Use AutoML to establish visual token quality baseline,
then migrate to custom training for relevance realization
"""

# Phase 1: AutoML Vision baseline
# Train on same images arr-coc-0-1 uses
# Establishes visual understanding baseline

# Phase 2: Export AutoML as frozen backbone
automl_backbone = load_automl_vision_backbone('gs://BUCKET/automl_export/')

# Phase 3: Add ARR-COC custom components
from arr_coc.knowing import (
    InformationScorer,
    SalienceScorer,
    QueryCouplingScorer
)
from arr_coc.balancing import TensionBalancer
from arr_coc.attending import TokenAllocator

class HybridARRCOC(tf.keras.Model):
    def __init__(self, automl_backbone):
        super().__init__()

        # AutoML frozen backbone (pre-trained visual understanding)
        self.visual_encoder = automl_backbone
        for layer in self.visual_encoder.layers:
            layer.trainable = False

        # Custom ARR-COC components (trainable)
        self.info_scorer = InformationScorer()
        self.salience_scorer = SalienceScorer()
        self.coupling_scorer = QueryCouplingScorer()
        self.balancer = TensionBalancer()
        self.allocator = TokenAllocator()

    def call(self, images, query, training=False):
        # AutoML extracts visual features
        visual_features = self.visual_encoder(images, training=False)

        # Custom relevance realization
        info_scores = self.info_scorer(visual_features)
        salience_scores = self.salience_scorer(visual_features)
        coupling_scores = self.coupling_scorer(visual_features, query)

        # Balance tensions
        relevance = self.balancer(info_scores, salience_scores, coupling_scores)

        # Allocate tokens (64-400 per patch based on relevance)
        token_allocation = self.allocator(relevance)

        return token_allocation

# Train custom components while keeping AutoML frozen
model = HybridARRCOC(automl_backbone)

# Only optimize custom relevance components
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)

for epoch in range(50):
    for batch in dataset:
        with tf.GradientTape() as tape:
            allocations = model(batch['images'], batch['query'], training=True)
            loss = custom_relevance_loss(allocations, batch['ground_truth'])

        # Gradients only for custom components (backbone frozen)
        trainable_vars = [v for v in model.trainable_variables
                          if 'visual_encoder' not in v.name]
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
```

### 5.3 Graduation Readiness Checklist

**Before graduating from AutoML, ensure:**

**Team Readiness:**
- [ ] At least one team member with 2+ years ML experience
- [ ] Familiarity with chosen framework (TensorFlow/PyTorch)
- [ ] Understanding of model training fundamentals
- [ ] Experience with distributed training (if needed)
- [ ] GPU infrastructure management knowledge

**Technical Requirements:**
- [ ] Custom training pipeline infrastructure ready
- [ ] Experiment tracking system (W&B, MLflow, TensorBoard)
- [ ] Hyperparameter tuning framework (Vertex AI Vizier, Optuna)
- [ ] CI/CD pipeline for model deployment
- [ ] Monitoring and alerting for model performance

**Business Justification:**
- [ ] AutoML baseline performance documented
- [ ] Clear target for custom training improvements
- [ ] ROI calculation showing positive return
- [ ] Budget allocated for development time
- [ ] Timeline acceptable for custom development

**Fallback Plan:**
- [ ] AutoML model exported and backed up
- [ ] Ability to revert to AutoML if custom fails
- [ ] A/B testing infrastructure to compare approaches
- [ ] Documented criteria for success/failure

---

## Section 6: Hybrid Pipeline Patterns (AutoML Baseline → Custom Optimization)

### 6.1 Pattern 1: AutoML as Data Quality Gate

**Use Case:** Validate data quality before investing in custom training

**Process:**
1. Train quick AutoML model (2-4 hours)
2. Review performance and feature importance
3. If performance <50% target → fix data issues
4. If performance 50-80% target → proceed with custom
5. If performance >80% target → consider deploying AutoML

**Example:**
```python
def data_quality_gate(dataset_path, target_accuracy=0.8):
    """
    Use AutoML to validate data quality before custom training

    Returns: (proceed_to_custom: bool, issues: list)
    """
    from google.cloud import aiplatform

    # Quick AutoML training (low budget)
    automl_job = aiplatform.AutoMLTabularTrainingJob(
        display_name='data-quality-gate',
        optimization_prediction_type='classification'
    )

    model = automl_job.run(
        dataset=dataset_path,
        target_column='label',
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=2000  # 2 hours only
    )

    # Analyze results
    evaluation = model.evaluate()
    accuracy = evaluation['auPrc']  # Use AUC-PR for imbalanced data

    # Check feature importance for data issues
    feature_importance = model.gca_resource.metadata['feature_importance']

    issues = []

    if accuracy < 0.5:
        issues.append("Accuracy below random: check data labels")

    # Check for single-feature dominance
    top_feature_importance = feature_importance[0]['importance']
    if top_feature_importance > 0.9:
        issues.append(f"Single feature dominates: {feature_importance[0]['name']}")

    # Check for low-importance features (potential noise)
    low_importance_features = [
        f['name'] for f in feature_importance
        if f['importance'] < 0.01
    ]
    if len(low_importance_features) > len(feature_importance) * 0.5:
        issues.append(f"50%+ features have low importance: possible noise")

    proceed = (accuracy >= 0.5 * target_accuracy) and (len(issues) == 0)

    return proceed, issues, accuracy
```

### 6.2 Pattern 2: AutoML for Hyperparameter Initialization

**Use Case:** Use AutoML's hyperparameter search as starting point for custom Vizier study

**Process:**
```python
from google.cloud import aiplatform

# Step 1: Train AutoML model
automl_model = train_automl_baseline(dataset)

# Step 2: Extract best hyperparameters
automl_hparams = extract_automl_hyperparameters(automl_model)

# Step 3: Create Vizier study with AutoML config as prior
study_config = {
    'metrics': [{'metric': 'accuracy', 'goal': 'MAXIMIZE'}],
    'parameters': [
        {
            'parameter': 'learning_rate',
            'type': 'DOUBLE',
            'double_value_spec': {
                'min_value': automl_hparams['learning_rate'] * 0.1,
                'max_value': automl_hparams['learning_rate'] * 10,
            },
            'scale_type': 'LOG'
        },
        # Initialize search around AutoML's best values
    ]
}

# Step 4: Run custom training with warm-started search
job = aiplatform.CustomTrainingJob(
    display_name='custom-training-warmstart',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest'
)

hpt_job = job.run(
    args=['--use_vizier'],
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

### 6.3 Pattern 3: AutoML Ensemble with Custom Models

**Use Case:** Combine AutoML baseline with custom models for ensemble

**Architecture:**
```
Input Data
    |
    ├──> AutoML Model 1 (exported) ──┐
    |                                 |
    ├──> AutoML Model 2 (exported) ──┤
    |                                 ├──> Ensemble Combiner ──> Final Prediction
    ├──> Custom Model 1 ──────────────┤
    |                                 |
    └──> Custom Model 2 ──────────────┘
```

**Implementation:**
```python
class AutoMLCustomEnsemble:
    def __init__(self, automl_model_paths, custom_models):
        # Load exported AutoML models
        self.automl_models = [
            tf.saved_model.load(path) for path in automl_model_paths
        ]

        # Custom trained models
        self.custom_models = custom_models

        # Learnable ensemble weights
        self.ensemble_weights = tf.Variable(
            tf.ones(len(automl_model_paths) + len(custom_models)) /
            (len(automl_model_paths) + len(custom_models)),
            trainable=True
        )

    def predict(self, inputs):
        predictions = []

        # AutoML predictions (frozen)
        for automl_model in self.automl_models:
            pred = automl_model.signatures['serving_default'](inputs)
            predictions.append(pred)

        # Custom predictions
        for custom_model in self.custom_models:
            pred = custom_model(inputs)
            predictions.append(pred)

        # Weighted ensemble
        predictions = tf.stack(predictions, axis=0)
        weights = tf.nn.softmax(self.ensemble_weights)

        final_prediction = tf.reduce_sum(
            predictions * weights[:, None, None], axis=0
        )

        return final_prediction
```

### 6.4 Pattern 4: AutoML for Baseline, Custom for Production

**Use Case:** Fast iteration with AutoML, deploy custom for cost/performance

**Workflow:**
```
Development Phase (Weeks 1-2):
  - Train multiple AutoML models for different approaches
  - Rapid A/B testing
  - Feature importance analysis
  - Select best performing approach

Migration Phase (Weeks 3-4):
  - Export best AutoML model
  - Replicate in custom training code
  - Validate metric parity
  - Optimize for production constraints

Production Phase (Week 5+):
  - Deploy optimized custom model
  - Monitor performance vs AutoML baseline
  - Iterate on custom implementation
  - Keep AutoML as fallback
```

**Cost Savings Example:**
- AutoML serving: $0.50/1000 predictions
- Custom TensorFlow Serving: $0.05/1000 predictions
- At 10M predictions/month: $5,000 vs $500/month = $4,500 savings

---

## Section 7: arr-coc-0-1 AutoML Vision Baseline for Comparison

### 7.1 Baseline Experiment Setup

**Objective:** Establish AutoML Vision baseline for visual token quality before implementing custom ARR-COC relevance realization

**Dataset:**
- Same images used in arr-coc-0-1 training
- Labels: High-relevance vs Low-relevance regions (based on query)
- Format: ImageDataset with bounding boxes

**AutoML Configuration:**
```python
from google.cloud import aiplatform

# Create dataset
dataset = aiplatform.ImageDataset.create(
    display_name='arr-coc-baseline-images',
    gcs_source='gs://arr-coc-data/labeled_images.csv',
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box
)

# Train AutoML Vision Object Detection
automl_job = aiplatform.AutoMLImageTrainingJob(
    display_name='arr-coc-automl-baseline',
    prediction_type='object_detection',
    model_type='CLOUD_HIGH_ACCURACY_1',  # Best quality
    budget_milli_node_hours=20000  # 20 hours
)

model = automl_job.run(
    dataset=dataset,
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    model_display_name='arr-coc-automl-v1'
)
```

### 7.2 Baseline Metrics

**AutoML Vision Performance (Expected):**
- mAP@0.5 IOU: 0.65-0.75 (detection accuracy)
- Precision: 0.70-0.80
- Recall: 0.60-0.70
- Inference latency: 50-100ms per image
- Model size: 100-200 MB

**Custom ARR-COC Target Improvements:**
- Query-aware relevance: 20-30% improvement in precision
- Variable LOD allocation: 40-60% reduction in token count
- Inference latency: <30ms (optimized attention)
- Model size: <50 MB (efficient architecture)

### 7.3 Comparison Framework

**Evaluation Metrics:**
```python
def compare_automl_vs_arrcoc(automl_model, arrcoc_model, test_dataset):
    """
    Compare AutoML baseline with custom ARR-COC model
    """
    results = {
        'automl': {},
        'arrcoc': {},
        'improvements': {}
    }

    for batch in test_dataset:
        images = batch['images']
        queries = batch['queries']
        ground_truth = batch['relevance_labels']

        # AutoML predictions (query-agnostic)
        automl_pred = automl_model.predict(images)
        automl_boxes = automl_pred['detection_boxes']

        # ARR-COC predictions (query-aware)
        arrcoc_relevance = arrcoc_model(images, queries)
        arrcoc_tokens = arrcoc_relevance['token_allocations']

        # Metrics
        results['automl']['precision'] = compute_precision(automl_boxes, ground_truth)
        results['automl']['recall'] = compute_recall(automl_boxes, ground_truth)
        results['automl']['token_count'] = len(automl_boxes) * 196  # Fixed tokens

        results['arrcoc']['precision'] = compute_precision(arrcoc_tokens, ground_truth)
        results['arrcoc']['recall'] = compute_recall(arrcoc_tokens, ground_truth)
        results['arrcoc']['token_count'] = tf.reduce_sum(arrcoc_tokens['num_tokens'])

    # Compute improvements
    results['improvements']['precision_gain'] = (
        results['arrcoc']['precision'] - results['automl']['precision']
    ) / results['automl']['precision']

    results['improvements']['token_reduction'] = (
        results['automl']['token_count'] - results['arrcoc']['token_count']
    ) / results['automl']['token_count']

    return results
```

**Cost Comparison:**
- AutoML training: 20 hours × $3.15 = $63
- Custom ARR-COC training: ~100 hours (including dev) × $3.24 = $324
- **Justification:** 4x higher precision + 50% fewer tokens = significant quality/efficiency gains

---

## Sources

**Source Documents:**
- [46-frozen-backbone-adapter-training.md](../practical-implementation/46-frozen-backbone-adapter-training.md)

**Web Research:**
- [AutoML in Vertex AI: Understanding the Relationship](https://promevo.com/blog/automl-in-vertex-ai) - Promevo (accessed 2025-11-16)
- [Hands-Off Machine Learning with Google AutoML](https://towardsdatascience.com/hands-off-machine-learning-with-google-automl-e63b079f09d1) - Towards Data Science (accessed 2025-11-16)
- [Vertex AI AutoML Beginner's Guide](https://docs.cloud.google.com/vertex-ai/docs/beginner/beginners-guide) - Google Cloud Documentation (accessed 2025-11-16)
- [AutoML Model Export TensorFlow ONNX](https://learn.microsoft.com/en-us/answers/questions/2244597/export-ml-model-to-onnx-format) - Microsoft Learn (accessed 2025-11-16)
- [Transfer Learning AutoML Warm Start](https://github.com/automl/transfer-hpo-framework) - GitHub AutoML Project (accessed 2025-11-16)
- [AWS Warm-Starting Hyperparameter Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html) - AWS Documentation (accessed 2025-11-16)
- [When to Graduate from AutoML](https://www.reddit.com/r/MachineLearning/comments/twxmk9/d_with_the_rise_of_automl_what_are_the_important/) - Reddit r/MachineLearning (accessed 2025-11-16)

**Additional References:**
- Vertex AI Pricing: https://cloud.google.com/vertex-ai/pricing
- TensorFlow SavedModel Format: https://www.tensorflow.org/guide/saved_model
- ONNX Model Format: https://onnx.ai/
- Vertex AI Vizier: https://cloud.google.com/vertex-ai/docs/vizier/overview

---

*Last updated: 2025-11-16*
*Knowledge domain: Vertex AI AutoML, Custom Training, Hybrid ML Strategies, Model Export, Transfer Learning*
