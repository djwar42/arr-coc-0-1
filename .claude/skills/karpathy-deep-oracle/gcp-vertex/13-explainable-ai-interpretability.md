# Vertex AI Explainable AI and Interpretability

## Overview

Vertex AI Explainable AI provides feature-based and example-based explanations to help understand ML model predictions. It integrates explanation methods directly into the training and serving pipeline, enabling post-hoc interpretability for vision, tabular, and text models without modifying model architecture.

**Core value proposition**: Transform black-box predictions into interpretable insights through attribution methods based on Shapley values and integrated gradients, supporting regulatory compliance, model debugging, and user trust.

## Explanation Methods

Vertex Explainable AI offers three primary feature-based attribution methods, all grounded in Shapley value theory from cooperative game theory.

### 1. Sampled Shapley Attribution

**Algorithm**: Approximates Shapley values by sampling feature coalitions rather than computing all possible combinations (which would be computationally intractable for high-dimensional inputs).

**How it works**:
- Generates random permutations of features
- For each feature, measures marginal contribution by comparing predictions with/without that feature
- Averages contributions across multiple samples to approximate true Shapley value
- Path count parameter controls precision (higher = more accurate but slower)

**Best for**:
- Non-differentiable models (tree-based, ensemble methods)
- Tabular data with categorical and numerical features
- Text classification (word-level attributions)

**Configuration**:
```python
from google.cloud.aiplatform.explain import SampledShapleyAttribution

sampled_shapley_config = SampledShapleyAttribution(
    path_count=50  # Number of random paths (10-100 typical range)
)
```

**Characteristics**:
- Model-agnostic (works with any black-box model)
- Computationally expensive (scales with feature count × path_count)
- Provides faithful attributions grounded in game theory
- Can handle feature interactions naturally

From [Medium: Vertex Explainable AI with Python](https://medium.com/@pysquad/vertex-explainable-ai-with-python-making-ai-decisions-understandable-4ba009965282) (accessed 2025-11-16):
> "Sampled Shapley approximates Shapley values for features that contribute to the label being predicted. A sampling approach is used to approximate the Shapley values rather than computing them exactly."

### 2. Integrated Gradients

**Algorithm**: Computes gradients of the model output with respect to input features along a path from a baseline input to the actual input.

**Mathematical foundation**:
```
Attribution(x) = (x - x') × ∫[0,1] ∂F(x' + α(x - x'))/∂x dα
```
Where:
- `x` = actual input
- `x'` = baseline input (e.g., all zeros, mean values, or blurred image)
- `F` = model prediction function
- `α` = interpolation parameter from 0 to 1

**How it works**:
1. Define baseline input (neutral/uninformative state)
2. Create linear path from baseline to actual input
3. Compute gradients at multiple points along path
4. Integrate (sum) gradients to get feature attributions
5. Scale by input difference from baseline

**Best for**:
- Differentiable models (neural networks, deep learning)
- Image classification and object detection
- Models with continuous input features

**Configuration**:
```python
from google.cloud.aiplatform.explain import IntegratedGradientsAttribution

ig_config = IntegratedGradientsAttribution(
    step_count=50,  # Number of interpolation steps (25-100 typical)
    smooth_grad_config={
        'noise_sigma': 0.1,  # Noise std dev for smoothing
        'noisy_sample_count': 50  # Number of noisy samples
    }
)
```

**Baseline selection strategies**:
- **Zero baseline**: All features set to 0 (simple but may not be meaningful)
- **Mean baseline**: Dataset mean values (better for normalized data)
- **Blurred baseline**: For images, use Gaussian blur to remove detail
- **Random baseline**: Sample from training distribution

**Approximation error metric**:
```python
error = |sum(attributions) - (prediction - baseline_prediction)|
```
Lower error = more faithful attribution. Increase `step_count` to reduce error.

From [Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/overview) (accessed 2025-11-16):
> "Integrated Gradients is a gradients-based method to efficiently compute feature attributions with the same axiomatic properties as Shapley values."

### 3. XRAI (eXplanation with Ranked Area Integrals)

**Algorithm**: Builds on Integrated Gradients for image models by segmenting the image into regions and ranking their importance.

**How it works**:
1. Over-segment image using algorithm like Felzenszwalb or SLIC (creates ~100-1000 segments)
2. Compute Integrated Gradients pixel-level attributions
3. Aggregate attributions within each segment
4. Iteratively merge segments based on attribution scores
5. Create saliency map showing most important regions

**Output**: Hierarchical ranking of image regions by importance, from most salient to least.

**Best for**:
- Computer vision models (image classification, object detection)
- Visual question answering
- Medical image analysis
- Any scenario requiring human-interpretable visual explanations

**Configuration**:
```python
from google.cloud.aiplatform.explain import XraiAttribution

xrai_config = XraiAttribution(
    step_count=50,  # IG interpolation steps
    smooth_grad_config={
        'noise_sigma': 0.15,
        'noisy_sample_count': 50
    }
)
```

**Advantages over pixel-level IG**:
- Produces coherent regions instead of noisy pixel attributions
- More interpretable for humans (segments align with objects)
- Reduces attribution artifacts
- Provides ranking of visual importance

From [arXiv:1906.02825](https://arxiv.org/abs/1906.02825) (accessed 2025-11-16):
> "XRAI is a region-based attribution method that builds upon integrated gradients. It assesses overlapping regions of the image to create a saliency map, which highlights relevant regions of the image to the model prediction."

## ExplanationMetadata Configuration

ExplanationMetadata defines how Vertex AI should interpret your model's inputs and outputs for explanation generation.

### Structure

```python
from google.cloud.aiplatform.explain import ExplanationMetadata

explanation_metadata = ExplanationMetadata(
    inputs={
        'image_input': {
            'input_tensor_name': 'images',  # Model input tensor
            'encoding': 'BAG_OF_FEATURES',  # For images
            'modality': 'image',
            'domain': [0, 255]  # Pixel value range
        },
        'numerical_feature': {
            'input_tensor_name': 'features',
            'encoding': 'IDENTITY',  # For numerical/tabular
            'index_feature_mapping': ['feature_1', 'feature_2', ...]
        }
    },
    outputs={
        'probability': {
            'output_tensor_name': 'predictions',
            'index_name_mapping': ['class_0', 'class_1', 'class_2']
        }
    }
)
```

### Input Configuration Fields

**encoding**: How features should be interpreted
- `IDENTITY`: Direct numerical values (tabular, embeddings)
- `BAG_OF_FEATURES`: Treat as collection of independent features (images, text tokens)
- `BAG_OF_FEATURES_SPARSE`: For sparse feature representations
- `INDICATOR`: Binary features (one-hot encoded)
- `COMBINED_EMBEDDING`: Multiple features combined into embedding
- `CONCAT_EMBEDDING`: Concatenated embeddings

**modality**: Type of input data
- `numeric`: Continuous numerical features
- `categorical`: Discrete categorical features
- `image`: Image tensors
- `text`: Text sequences

**input_baselines**: Optional baseline values for attribution
```python
'image_input': {
    'input_baselines': [[0.0] * num_pixels]  # Black image baseline
}
# OR
'tabular_input': {
    'input_baselines': [[mean_val_1, mean_val_2, ...]]  # Dataset means
}
```

**domain**: Valid range for input values (e.g., `[0, 255]` for images, `[-1, 1]` for normalized)

**index_feature_mapping**: Names for tabular features
```python
'features': {
    'index_feature_mapping': [
        'age', 'income', 'education_level', 'credit_score'
    ]
}
```

### Output Configuration Fields

**index_name_mapping**: Class labels for classification
```python
'predictions': {
    'index_name_mapping': [
        'cat', 'dog', 'bird', 'fish'
    ]
}
```

### Auto-inference

If you omit ExplanationMetadata, Vertex AI attempts to infer inputs/outputs from your model's SavedModel signature:
```python
# Vertex AI will automatically detect:
# - Input tensor names and shapes
# - Output tensor names
# - Encoding types based on tensor shapes
```

**When to use auto-inference**: Simple models with standard TensorFlow SavedModel exports.

**When to specify manually**: Custom models, multiple inputs/outputs, specific baseline requirements, or non-standard encoding.

From [Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations-feature-based) (accessed 2025-11-16):
> "ExplanationMetadata contains the inputs and outputs metadata of your model. If omitted, Vertex AI automatically infers the inputs and outputs from the model."

## Batch Explanation Jobs

Generate explanations for thousands or millions of predictions asynchronously using batch processing.

### Use Cases

1. **Dataset auditing**: Explain all predictions on validation/test set
2. **Model debugging**: Find systematic attribution patterns indicating bugs
3. **Compliance reporting**: Generate explanation artifacts for regulatory review
4. **Offline analysis**: Deep dive into model behavior without real-time constraints

### Creating Batch Explanation Jobs

```python
from google.cloud import aiplatform

# Initialize client
aiplatform.init(project='your-project', location='us-central1')

# Get model resource
model = aiplatform.Model('projects/PROJECT/locations/LOCATION/models/MODEL_ID')

# Create batch prediction job with explanations
batch_job = model.batch_predict(
    job_display_name='batch-explanations-job',

    # Input data source
    bigquery_source='bq://project.dataset.input_table',
    # OR
    gcs_source='gs://bucket/input_data/*.jsonl',

    # Output destination
    bigquery_destination_prefix='bq://project.dataset',
    # OR
    gcs_destination_prefix='gs://bucket/output/',

    # Enable explanations
    generate_explanation=True,

    # Machine type for workers
    machine_type='n1-standard-4',

    # Parallelism
    starting_replica_count=10,
    max_replica_count=50,

    # Explanation parameters (optional override)
    explanation_metadata=explanation_metadata,
    explanation_parameters=explanation_parameters
)

# Monitor job
batch_job.wait()
print(f"Job state: {batch_job.state}")
print(f"Output: {batch_job.output_info}")
```

### Input Format Options

**BigQuery**:
```sql
-- Input table must contain instances in JSON format
CREATE TABLE `project.dataset.input_table` AS
SELECT
    TO_JSON_STRING(STRUCT(
        feature1, feature2, feature3
    )) as instance
FROM `project.dataset.source_data`;
```

**Cloud Storage (JSONL)**:
```json
{"feature1": 1.2, "feature2": "value", "feature3": [0, 1, 0]}
{"feature1": 3.4, "feature2": "value2", "feature3": [1, 0, 1]}
```

### Output Format

**Predictions + Explanations**:
```json
{
    "instance": {"feature1": 1.2, "feature2": "value"},
    "prediction": [0.1, 0.7, 0.2],
    "explanations": {
        "attributions": [
            {
                "featureAttributions": {
                    "feature1": 0.45,
                    "feature2": 0.38,
                    "feature3": 0.17
                },
                "approximationError": 0.002
            }
        ]
    }
}
```

### Performance Optimization

**Batch size tuning**:
```python
# Larger batches = faster throughput, higher memory
batch_job = model.batch_predict(
    instances_format='jsonl',
    batch_size=64,  # Default 64, range 1-128
    ...
)
```

**Worker scaling**:
- `starting_replica_count`: Initial workers (start high for large jobs)
- `max_replica_count`: Maximum autoscaling limit
- Cost tradeoff: More workers = faster completion but higher cost

**Resource allocation**:
```python
# For explanation-heavy workloads
machine_type='n1-highmem-8'  # More memory for complex attributions
accelerator_type='NVIDIA_TESLA_T4'  # GPU for deep learning models
accelerator_count=1
```

From [Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction) (accessed 2025-11-16):
> "Batch prediction allows you to make predictions against a model using input from BigQuery or Cloud Storage. You can independently choose to output predictions to either a BigQuery table or Cloud Storage location."

## Visual Explanations (Heatmaps and Saliency Maps)

Transform attribution scores into visual overlays that highlight important regions in images.

### Generating Image Explanations

```python
from google.cloud import aiplatform
import numpy as np
import matplotlib.pyplot as plt

# Get endpoint
endpoint = aiplatform.Endpoint('projects/.../endpoints/...')

# Prepare image instance
image_data = preprocess_image('path/to/image.jpg')  # Returns numpy array

# Get prediction with explanation
response = endpoint.explain(
    instances=[{'image': image_data.tolist()}]
)

# Extract attribution values
attribution = response.explanations[0].attributions[0]
feature_attributions = attribution.feature_attributions

# Reshape to image dimensions (H, W, C)
attribution_map = np.array(feature_attributions['image']).reshape(224, 224, 3)

# Create heatmap visualization
def create_heatmap(image, attribution_map):
    # Sum across color channels
    attribution_sum = np.sum(np.abs(attribution_map), axis=2)

    # Normalize to [0, 1]
    attribution_norm = (attribution_sum - attribution_sum.min()) / (
        attribution_sum.max() - attribution_sum.min()
    )

    # Apply colormap
    heatmap = plt.cm.jet(attribution_norm)[:, :, :3]

    # Overlay on original image
    overlay = 0.6 * image + 0.4 * heatmap

    return overlay

overlay = create_heatmap(image_data, attribution_map)
plt.imshow(overlay)
plt.title(f"Prediction: {response.predictions[0]} (confidence: {max(response.predictions[0])})")
plt.show()
```

### XRAI Region Highlighting

```python
# XRAI returns ranked regions instead of pixel-level attributions
response = endpoint.explain(
    instances=[{'image': image_data.tolist()}],
    parameters={'xrai_config': {'step_count': 50}}
)

# Extract region attributions
regions = response.explanations[0].attributions[0]

# Visualize top-k most important regions
def visualize_xrai_regions(image, attributions, top_k=5):
    # Sort regions by attribution score
    sorted_regions = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Highlight top regions
    highlighted = image.copy()
    for i, (region_id, score) in enumerate(sorted_regions[:top_k]):
        # Draw bounding box or mask for region
        draw_region(highlighted, region_id, alpha=0.3, color=cmap(i))

    return highlighted
```

### Customizing Visualization

**Color schemes**:
- Red-blue diverging: Positive (red) vs negative (blue) attributions
- Jet colormap: Magnitude-based (hot = high importance)
- Green overlay: Positive attributions only

**Thresholding**:
```python
# Show only attributions above threshold
threshold = 0.3
attribution_filtered = np.where(
    attribution_norm > threshold,
    attribution_norm,
    0
)
```

**Multi-class visualization**:
```python
# Show different classes in different colors
for class_idx in range(num_classes):
    class_attribution = get_attribution_for_class(response, class_idx)
    overlay = create_heatmap(image, class_attribution)
    plt.subplot(1, num_classes, class_idx + 1)
    plt.imshow(overlay)
    plt.title(f"Class {class_idx}")
```

From [Medium: Explaining an Image Classification Model](https://medium.com/@yasmeen87151/explaining-an-image-classification-model-with-vertex-explainable-ai-9f61f2e6b72b) (accessed 2025-11-16):
> "This tutorial shows how to use Vertex Explainable AI to understand image classification predictions through visual attributions, creating saliency maps that highlight important regions."

## Tabular Explanations (Feature Importance Scores)

Generate feature-level attribution scores for structured data predictions.

### Getting Tabular Explanations

```python
# Deploy model with explanations enabled
model = aiplatform.Model.upload(
    display_name='tabular-model',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    explanation_metadata=ExplanationMetadata(
        inputs={
            'features': {
                'input_tensor_name': 'features',
                'encoding': 'IDENTITY',
                'index_feature_mapping': [
                    'age', 'income', 'credit_score', 'debt_ratio',
                    'employment_length', 'num_accounts'
                ]
            }
        },
        outputs={'prediction': {'output_tensor_name': 'probabilities'}}
    ),
    explanation_parameters=ExplanationParameters(
        sampled_shapley_attribution=SampledShapleyAttribution(path_count=50)
    )
)

endpoint = model.deploy(machine_type='n1-standard-4')

# Get explanation for single instance
instance = {
    'age': 35,
    'income': 75000,
    'credit_score': 720,
    'debt_ratio': 0.3,
    'employment_length': 8,
    'num_accounts': 5
}

response = endpoint.explain(instances=[instance])

# Extract feature attributions
attributions = response.explanations[0].attributions[0].feature_attributions
for feature, score in attributions.items():
    print(f"{feature}: {score:.4f}")
```

### Interpreting Attribution Scores

**Positive attribution**: Feature increases predicted probability of positive class
**Negative attribution**: Feature decreases predicted probability
**Magnitude**: Strength of feature's influence on prediction

Example output:
```
credit_score: 0.3421  # Strong positive influence
income: 0.1832        # Moderate positive influence
debt_ratio: -0.2154   # Moderate negative influence
age: 0.0621           # Weak positive influence
employment_length: 0.0893
num_accounts: -0.0412
```

### Visualization

**Bar chart of feature importance**:
```python
import pandas as pd
import seaborn as sns

# Create DataFrame from attributions
df = pd.DataFrame([
    {'feature': k, 'attribution': v}
    for k, v in attributions.items()
]).sort_values('attribution', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='attribution', y='feature', palette='coolwarm')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.title('Feature Attributions for Prediction')
plt.xlabel('Attribution Score')
plt.show()
```

### Aggregating Explanations Across Dataset

```python
# Collect explanations for all instances
all_attributions = []

for instance in validation_dataset:
    response = endpoint.explain(instances=[instance])
    attrs = response.explanations[0].attributions[0].feature_attributions
    all_attributions.append(attrs)

# Compute aggregate statistics
feature_means = pd.DataFrame(all_attributions).mean()
feature_stds = pd.DataFrame(all_attributions).std()

# Identify most important features globally
top_features = feature_means.abs().sort_values(ascending=False).head(10)
print("Top 10 most important features:")
print(top_features)
```

### Detecting Model Bias

```python
# Group explanations by sensitive attribute
male_attributions = [
    explain(instance)
    for instance in data if instance['gender'] == 'male'
]
female_attributions = [
    explain(instance)
    for instance in data if instance['gender'] == 'female'
]

# Compare average attributions
male_avg = pd.DataFrame(male_attributions).mean()
female_avg = pd.DataFrame(female_attributions).mean()

# Flag features with large discrepancies
bias_threshold = 0.1
biased_features = (male_avg - female_avg).abs() > bias_threshold
print(f"Potentially biased features: {biased_features[biased_features].index.tolist()}")
```

From [Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-explanations) (accessed 2025-11-16):
> "The attribution methods approximate the Shapley value, providing feature importance scores that indicate how much each feature contributed to the prediction."

## Model Cards for Compliance and Trust

Model Cards provide structured documentation about model training, evaluation, intended use, and limitations to support transparency and compliance.

### Creating Model Cards

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import Model
from google.cloud.aiplatform.explain import ModelCard

# Create model card
model_card = ModelCard(
    model_name='fraud-detection-v1',
    model_description='Binary classifier for credit card fraud detection',

    # Model details
    model_type='XGBoost Classifier',
    model_version='1.0',
    training_date='2024-01-15',

    # Intended use
    intended_use='Identify fraudulent credit card transactions in real-time',
    primary_uses=['Fraud detection', 'Risk assessment'],
    out_of_scope_uses=['Consumer credit scoring', 'Insurance underwriting'],

    # Training data
    training_data_description='Historical credit card transactions (Jan 2022 - Dec 2023)',
    training_data_size=1000000,
    data_preprocessing='Removed PII, normalized numerical features, one-hot encoded categories',

    # Evaluation
    evaluation_metrics={
        'precision': 0.92,
        'recall': 0.88,
        'f1_score': 0.90,
        'auc_roc': 0.94
    },
    evaluation_dataset_size=200000,

    # Ethical considerations
    sensitive_data=['Transaction location', 'Merchant category'],
    fairness_assessment='Tested for bias across geographic regions and merchant types',

    # Limitations
    limitations=[
        'May have reduced accuracy on novel fraud patterns',
        'Not validated for international transactions',
        'Performance degrades when fraud tactics evolve'
    ],

    # Regulatory compliance
    regulatory_requirements=['GDPR Article 22', 'FCRA Section 615'],
    explainability_type='Feature-based (Sampled Shapley)',

    # Contact
    model_owner='ml-team@company.com',
    license='Proprietary'
)

# Upload model with model card
model = aiplatform.Model.upload(
    display_name='fraud-detection-v1',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri=container_image,
    model_card=model_card
)
```

### Model Card Schema

```python
{
    "model_details": {
        "name": str,
        "overview": str,
        "version": str,
        "owners": [str],
        "license": str,
        "references": [str],
        "citations": [str]
    },
    "model_parameters": {
        "model_architecture": str,
        "data": {
            "train": {"name": str, "size": int, "description": str},
            "eval": {"name": str, "size": int},
            "test": {"name": str, "size": int}
        },
        "input_format": str,
        "output_format": str
    },
    "quantitative_analysis": {
        "performance_metrics": [
            {"type": str, "value": float, "threshold": float}
        ],
        "graphics": {
            "description": str,
            "collection": [{"name": str, "image": str}]
        }
    },
    "considerations": {
        "users": [str],
        "use_cases": [str],
        "limitations": [str],
        "tradeoffs": [str],
        "ethical_considerations": [
            {"name": str, "mitigation_strategy": str}
        ]
    }
}
```

### Compliance Mappings

**GDPR (Article 22 - Right to Explanation)**:
```python
model_card.regulatory_compliance['gdpr'] = {
    'article_22_compliance': True,
    'explanation_method': 'Sampled Shapley Attribution',
    'explanation_availability': 'On-demand via API',
    'human_review_process': 'Flagged predictions reviewed by fraud analysts'
}
```

**EU AI Act (High-Risk AI Systems)**:
```python
model_card.eu_ai_act = {
    'risk_category': 'High-risk',
    'conformity_assessment': 'Internal control',
    'technical_documentation': 'gs://bucket/technical-docs/',
    'risk_management_system': 'Continuous monitoring with drift detection',
    'data_governance': 'Training data versioned and lineage tracked',
    'transparency': 'Model card and explanations provided'
}
```

**SOC 2 Type II**:
```python
model_card.soc2_controls = {
    'cc6.1': 'Logical access controls via IAM',
    'cc7.2': 'Model monitoring and alerting',
    'cc8.1': 'Change management via MLOps pipeline',
    'a1.2': 'Encryption at rest and in transit'
}
```

### Programmatic Access

```python
# Retrieve model card
model = aiplatform.Model('projects/.../models/...')
card = model.get_model_card()

print(f"Model: {card.model_name}")
print(f"Intended use: {card.intended_use}")
print(f"Limitations: {card.limitations}")
print(f"Performance: {card.evaluation_metrics}")
```

### Version Control

```python
# Update model card for new version
updated_card = model.get_model_card()
updated_card.model_version = '1.1'
updated_card.training_date = '2024-06-15'
updated_card.evaluation_metrics['f1_score'] = 0.93
updated_card.changelog = 'Added temporal features, retrained on 2024 data'

model.update_model_card(updated_card)
```

From [Google Cloud Security & Compliance](https://cloud.google.com/security/compliance/soc-2) (accessed 2025-11-16):
> "Google Cloud undergoes regular third-party audits to certify individual products against SOC 2 standards, supporting customer compliance requirements."

## arr-coc-0-1 Attention Visualization with Explainable AI

Apply Vertex AI Explainable AI to visualize and interpret ARR-COC's relevance realization process, particularly the attending phase where token budgets are allocated.

### Explaining Relevance Allocation Decisions

```python
from google.cloud import aiplatform
from arr_coc.attending import AttentionAllocator

# Deploy ARR-COC model to Vertex AI with explanations
model = aiplatform.Model.upload(
    display_name='arr-coc-relevance-allocator',
    artifact_uri='gs://bucket/arr-coc-model/',
    serving_container_image_uri=custom_container,
    explanation_metadata=ExplanationMetadata(
        inputs={
            'image_patches': {
                'input_tensor_name': 'patches',
                'encoding': 'BAG_OF_FEATURES',
                'modality': 'image',
                'input_baselines': [[0.0] * (13 * 64)]  # 13 channels, 64 tokens baseline
            },
            'query_embedding': {
                'input_tensor_name': 'query',
                'encoding': 'IDENTITY',
                'modality': 'numeric'
            }
        },
        outputs={
            'token_budget': {
                'output_tensor_name': 'allocated_tokens',
                'index_name_mapping': [f'patch_{i}' for i in range(200)]  # K=200 patches
            }
        }
    ),
    explanation_parameters=ExplanationParameters(
        integrated_gradients_attribution=IntegratedGradientsAttribution(
            step_count=50,
            smooth_grad_config={'noise_sigma': 0.1, 'noisy_sample_count': 30}
        )
    )
)

endpoint = model.deploy(machine_type='n1-highmem-8', accelerator_type='NVIDIA_TESLA_T4')

# Explain relevance allocation for a query
image = load_image('path/to/image.jpg')
query = "Where is the cat?"

response = endpoint.explain(
    instances=[{
        'image_patches': arr_coc.extract_patches(image),  # [K, 13, 64]
        'query_embedding': arr_coc.encode_query(query)   # [768]
    }]
)

# Visualize which image features influenced token allocation
attributions = response.explanations[0].attributions[0]
patch_attributions = attributions.feature_attributions['image_patches']

# Map back to original image space
def visualize_relevance_attributions(image, patch_attrs, patch_grid):
    """Overlay patch-level attributions on original image"""
    H, W = image.shape[:2]
    h, w = patch_grid

    heatmap = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            patch_idx = i * w + j
            # Sum attribution across all channels and tokens for this patch
            heatmap[i, j] = np.sum(np.abs(patch_attrs[patch_idx]))

    # Resize to image dimensions
    heatmap_resized = resize(heatmap, (H, W))

    # Overlay
    overlay = 0.6 * image + 0.4 * plt.cm.jet(heatmap_resized)[:, :, :3]
    return overlay

overlay = visualize_relevance_attributions(
    image,
    patch_attributions,
    patch_grid=(10, 20)  # K=200 patches arranged 10x20
)
plt.imshow(overlay)
plt.title(f"Query: '{query}' - Relevance Attribution Heatmap")
plt.show()
```

### Explaining Opponent Processing

```python
# Get attributions for each tension axis
tensions = ['compress_particularize', 'exploit_explore', 'focus_diversify']

for tension in tensions:
    response = endpoint.explain(
        instances=[instance],
        parameters={
            'output_indices': [tension_to_output_idx[tension]]
        }
    )

    attrs = response.explanations[0].attributions[0]

    # Visualize which patches drove this tension's resolution
    visualize_tension_attributions(image, attrs, tension)
```

### Debugging LOD Allocation

```python
# Compare explanations for patches with different LOD allocations
high_lod_patches = [i for i, lod in enumerate(lod_allocation) if lod >= 350]
low_lod_patches = [i for i, lod in enumerate(lod_allocation) if lod <= 100]

# Get patch-specific attributions
high_lod_attrs = get_attributions_for_patches(response, high_lod_patches)
low_lod_attrs = get_attributions_for_patches(response, low_lod_patches)

# Identify discriminating features
discriminating_channels = []
for channel in range(13):  # 13-channel texture array
    high_contrib = np.mean([attrs[channel] for attrs in high_lod_attrs])
    low_contrib = np.mean([attrs[channel] for attrs in low_lod_attrs])

    if abs(high_contrib - low_contrib) > threshold:
        discriminating_channels.append({
            'channel': channel,
            'high_lod_contrib': high_contrib,
            'low_lod_contrib': low_contrib,
            'difference': high_contrib - low_contrib
        })

print("Channels driving high vs low LOD allocation:")
for ch in discriminating_channels:
    print(f"Channel {ch['channel']}: Δ = {ch['difference']:.4f}")
```

### Validating Vervaekean Principles

```python
# Test if relevance realization exhibits expected properties

# 1. Query-dependence: Same image, different queries should yield different attributions
query_1 = "Where is the red car?"
query_2 = "Where is the person?"

attrs_1 = get_attributions(image, query_1)
attrs_2 = get_attributions(image, query_2)

correlation = np.corrcoef(attrs_1.flatten(), attrs_2.flatten())[0, 1]
print(f"Attribution correlation (different queries): {correlation:.3f}")
# Expect: Low correlation (< 0.5) indicating query-dependent relevance

# 2. Transjective coupling: Attributions should involve both image and query
image_only_attrs = get_attributions(image, query="")  # Null query
query_only_attrs = get_attributions(blank_image, query)  # Blank image

full_attrs = get_attributions(image, query)

print(f"Image-only variance: {np.var(image_only_attrs):.4f}")
print(f"Query-only variance: {np.var(query_only_attrs):.4f}")
print(f"Full (transjective) variance: {np.var(full_attrs):.4f}")
# Expect: Full variance >> individual variances (emergence from coupling)

# 3. Opponent processing: Tension attributions should show balancing
compress_attrs = get_tension_attrs(instance, 'compress')
particularize_attrs = get_tension_attrs(instance, 'particularize')

balance_score = np.abs(np.mean(compress_attrs) + np.mean(particularize_attrs))
print(f"Compress-Particularize balance: {balance_score:.4f}")
# Expect: Small value (close to 0) indicating balanced opposition
```

### Generating Model Cards for ARR-COC

```python
arr_coc_card = ModelCard(
    model_name='ARR-COC-VIS',
    model_description='Adaptive Relevance Realization for Vision-Language Models using Vervaeke cognitive framework',

    model_type='Hybrid: Relevance scorers + Neural allocators + Qwen3-VL backbone',
    model_version='0.1-MVP',

    intended_use='Query-aware visual token allocation with variable LOD (64-400 tokens per patch)',
    primary_uses=[
        'Visual question answering with compute-efficient attention',
        'Object localization with dynamic resolution',
        'Multimodal reasoning with cognitive grounding'
    ],

    training_data_description='VQAv2, GQA, Visual Genome (patch-level relevance annotations)',

    evaluation_metrics={
        'vqa_accuracy': 0.78,
        'gqa_accuracy': 0.72,
        'avg_tokens_per_image': 18500,  # vs 100,800 for full resolution
        'compute_reduction': '82%'
    },

    explainability_type='Integrated Gradients for token allocation decisions',
    explainability_features=[
        'Patch-level relevance heatmaps',
        'Query-attribution coupling visualization',
        'Tension-specific feature importance',
        'LOD allocation debugging'
    ],

    limitations=[
        'Relevance scorers trained on English queries only',
        'Optimal LOD ranges (64-400) tuned for natural images',
        'Opponent processing may oscillate on ambiguous queries',
        'Transjective coupling untested on synthetic/abstract images'
    ],

    ethical_considerations=[
        {
            'name': 'Attention bias',
            'description': 'Model may allocate tokens inequitably across demographic groups',
            'mitigation': 'Monitor attribution distributions across protected attributes'
        },
        {
            'name': 'Query manipulation',
            'description': 'Adversarial queries could exploit relevance allocation',
            'mitigation': 'Robustness testing with paraphrased and adversarial queries'
        }
    ],

    model_owner='arr-coc-team@example.com',
    references=[
        'Vervaeke, J. (2019). Relevance Realization and the Cognitive Science of Wisdom',
        'Platonic Dialogue Part 46: ARR-COC MVP Architecture'
    ]
)
```

## Best Practices

### 1. Choose Appropriate Explanation Method

**Decision tree**:
- Differentiable model + tabular data → Integrated Gradients
- Non-differentiable model + tabular data → Sampled Shapley
- Image classification → XRAI (or Integrated Gradients for pixel-level)
- Text classification → Sampled Shapley (word-level attributions)

### 2. Baseline Selection

**For Integrated Gradients**:
- Images: Use blurred baseline (Gaussian blur σ=10) or black/white image
- Tabular: Use dataset mean values or median
- Text: Use zero embeddings or mask tokens

**Testing baselines**:
```python
baselines = [zero_baseline, mean_baseline, blurred_baseline]
errors = []

for baseline in baselines:
    response = explain(instance, baseline=baseline)
    error = response.approximation_error
    errors.append((baseline, error))

best_baseline = min(errors, key=lambda x: x[1])[0]
```

### 3. Validate Explanation Quality

**Check approximation error**:
```python
# For Integrated Gradients
if response.approximation_error > 0.05:
    # Increase step_count or adjust baseline
    warnings.warn(f"High approximation error: {response.approximation_error}")
```

**Sanity checks**:
- **Completeness**: `sum(attributions) ≈ prediction - baseline_prediction`
- **Sensitivity**: Changing important features should change prediction
- **Implementation invariance**: Explanations shouldn't change if model is rewritten equivalently

### 4. Performance Optimization

**Online explanations** (low latency required):
- Use lower `step_count` (25-50 for IG)
- Use lower `path_count` (10-30 for Sampled Shapley)
- Cache baseline values
- Deploy on accelerators (GPU/TPU)

**Batch explanations** (high throughput):
- Increase `step_count`/`path_count` for accuracy
- Use autoscaling workers
- Process in parallel

### 5. User-Facing Explanations

**Presentation guidelines**:
- Show top-5 most important features (avoid overwhelming users)
- Use natural language descriptions, not raw attribution scores
- Provide context: "This feature increased probability by X%"
- Include confidence intervals if available

**Example**:
```python
def format_explanation_for_user(attributions, prediction, top_k=5):
    # Sort by absolute attribution
    sorted_attrs = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    # Generate natural language
    explanation = f"Prediction: {prediction['class']} (confidence: {prediction['probability']:.1%})\n\n"
    explanation += "Key factors:\n"

    for feature, score in sorted_attrs:
        direction = "increased" if score > 0 else "decreased"
        magnitude = "strongly" if abs(score) > 0.3 else "moderately" if abs(score) > 0.1 else "slightly"
        explanation += f"• {feature}: {magnitude} {direction} prediction ({abs(score):.2f})\n"

    return explanation
```

### 6. Monitoring Explanations in Production

```python
# Log attribution distributions
import logging

def log_explanation_stats(response):
    attrs = response.explanations[0].attributions[0].feature_attributions

    logging.info({
        'prediction': response.predictions[0],
        'top_feature': max(attrs, key=attrs.get),
        'attribution_entropy': compute_entropy(attrs),
        'num_zero_attrs': sum(1 for v in attrs.values() if abs(v) < 0.01),
        'approximation_error': response.approximation_error
    })

# Alert on anomalies
if attribution_entropy < threshold:
    alert("Explanation concentrated on single feature - possible model issue")
```

## Troubleshooting

### Common Issues

**Issue**: High approximation error in Integrated Gradients
**Solution**: Increase `step_count` from 50 to 100, or adjust baseline

**Issue**: Explanations don't match intuition
**Solution**: Verify baseline is meaningful, check for label leakage, validate model itself

**Issue**: Sampled Shapley too slow
**Solution**: Reduce `path_count`, use batch explanations instead of online

**Issue**: XRAI produces noisy saliency maps
**Solution**: Increase `smooth_grad` sample count, adjust segmentation parameters

**Issue**: Batch explanation job OOM (out of memory)
**Solution**: Reduce batch size, use higher-memory machine type, enable GPU

## Sources

**Web Research:**
- [Google Cloud: Vertex AI Explainable AI Overview](https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/overview) (accessed 2025-11-16)
- [Google Cloud: Configure Feature-Based Explanations](https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations-feature-based) (accessed 2025-11-16)
- [Google Cloud: Improve Feature-Based Explanations](https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/improving-explanations) (accessed 2025-11-16)
- [Google Cloud: Tabular Classification Explanations](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/classification-explanations) (accessed 2025-11-16)
- [Google Cloud: Batch Predictions](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction) (accessed 2025-11-16)
- [Medium: Vertex Explainable AI with Python - Making AI Decisions Understandable](https://medium.com/@pysquad/vertex-explainable-ai-with-python-making-ai-decisions-understandable-4ba009965282) by PySquad (accessed 2025-11-16)
- [Medium: Explaining an Image Classification Model with Vertex Explainable AI](https://medium.com/@yasmeen87151/explaining-an-image-classification-model-with-vertex-explainable-ai-9f61f2e6b72b) by Yasmeen Begum (accessed 2025-11-16)
- [Medium: Can I Explain a Text Model with Vertex AI? Yes, You Can!](https://medium.com/google-cloud/can-i-explain-a-text-model-with-vertex-ai-yes-you-can-1326f0265f09) by Ivan Nardini (accessed 2025-11-16)
- [arXiv:1906.02825: XRAI: Better Attributions Through Regions](https://arxiv.org/abs/1906.02825) by Kapishnikov et al. (accessed 2025-11-16)
- [Google Research: XRAI - Better Attributions Through Regions](https://research.google/pubs/xrai-better-attributions-through-regions/) (accessed 2025-11-16)
- [Google Cloud: SOC 2 Compliance](https://cloud.google.com/security/compliance/soc-2) (accessed 2025-11-16)
- [Google Colab: Vertex AI Explaining Image Classification](https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/xai_image_classification_feature_attributions.ipynb) (accessed 2025-11-16)

**Additional References:**
- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS
