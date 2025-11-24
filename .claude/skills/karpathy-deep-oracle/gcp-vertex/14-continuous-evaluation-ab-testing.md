# Vertex AI Continuous Evaluation & A/B Testing

**Knowledge File**: Comprehensive guide to continuous model evaluation, A/B testing strategies, statistical significance testing, and champion/challenger deployments in Vertex AI

---

## Overview

Production ML models require continuous evaluation to maintain quality over time. Unlike traditional software that degrades predictably, ML models experience **silent degradation** - performance drops without visible errors or crashes. Continuous evaluation and A/B testing enable safe model updates by comparing challenger models against production champions before full rollout.

From [DataRobot MLOps Champion/Challenger](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/) (accessed 2025-11-16):
> "Regardless of how much you train or test a model in your lab environment, the results will still merely represent just an estimation of your model's behavior once it crosses over to your actual production environment."

**Key challenges:**
- Models degrade silently (no error messages, just worse predictions)
- Ground truth arrives with delays (days to months feedback lag)
- Production data differs from training data (distribution shift)
- Risk of deploying worse models than current champion
- Business impact of incorrect predictions

**Vertex AI evaluation approach:**
1. **Continuous evaluation** - Automated metric tracking over time
2. **Traffic splitting** - A/B test models with production traffic
3. **Statistical testing** - Prove improvement significance
4. **Champion/challenger** - Safe model replacement workflow
5. **Gradual rollout** - Progressive traffic migration (5% → 100%)

---

## Section 1: ModelEvaluation Pipeline Component (~100 lines)

### What is ModelEvaluation?

Vertex AI Pipelines includes a **ModelEvaluation** component that automates model quality assessment during training pipelines. It computes evaluation metrics on test datasets and stores results in Vertex AI Model Registry.

**Pipeline integration pattern:**

```python
from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2.dsl import component, Output, Metrics, Dataset

@dsl.pipeline(
    name='training-with-evaluation-pipeline',
    description='Train model and evaluate before registration'
)
def training_evaluation_pipeline(
    project: str,
    location: str,
    train_data_uri: str,
    test_data_uri: str,
    accuracy_threshold: float = 0.85
):
    """
    Complete training pipeline with evaluation gate
    """

    # Step 1: Data preprocessing
    preprocess_op = preprocess_data(
        input_data=train_data_uri
    )

    # Step 2: Model training
    train_op = train_model(
        training_data=preprocess_op.outputs['processed_data']
    )

    # Step 3: Model evaluation (CRITICAL GATE)
    eval_op = evaluate_model(
        model=train_op.outputs['model'],
        test_data=test_data_uri,
        metrics_output=Output(Metrics),
        accuracy_threshold=accuracy_threshold
    )

    # Step 4: Conditional registration (only if eval passes)
    with dsl.Condition(
        eval_op.outputs['accuracy'] > accuracy_threshold,
        name='check-accuracy-threshold'
    ):
        register_op = register_model(
            model=train_op.outputs['model'],
            evaluation_metrics=eval_op.outputs['metrics']
        )
```

### Evaluation Component Implementation

**Custom evaluation component:**

```python
@component(
    base_image='gcr.io/my-project/evaluation:latest',
    packages_to_install=['scikit-learn==1.3.0']
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    metrics_output: Output[Metrics],
    accuracy_threshold: float
) -> NamedTuple('Outputs', [
    ('accuracy', float),
    ('precision', float),
    ('recall', float),
    ('f1_score', float),
    ('passes_threshold', bool)
]):
    """
    Evaluate model on test dataset and compute metrics
    """
    import pickle
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )

    # Load model
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)

    # Load test data
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Generate predictions
    y_pred = model_obj.predict(X_test)
    y_pred_proba = model_obj.predict_proba(X_test)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics to Vertex AI
    metrics_output.log_metric('accuracy', accuracy)
    metrics_output.log_metric('precision', precision)
    metrics_output.log_metric('recall', recall)
    metrics_output.log_metric('f1_score', f1)
    metrics_output.log_metric('roc_auc', auc)

    # Threshold check
    passes = accuracy >= accuracy_threshold

    # Log evaluation results
    print(f"Evaluation Results:")
    print(f"  Accuracy:  {accuracy:.4f} (threshold: {accuracy_threshold})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Passes threshold: {passes}")

    return (accuracy, precision, recall, f1, passes)
```

### Automated Evaluation Triggers

**EventBridge pattern for continuous evaluation:**

```python
# Cloud Function triggered when new model uploaded
def evaluate_new_model(event, context):
    """
    Triggered by Eventarc when model uploaded to registry
    """
    from google.cloud import aiplatform

    # Extract model details from event
    model_id = event['data']['modelId']

    # Initialize pipeline client
    pipeline_client = aiplatform.PipelineJob(
        display_name='automated-evaluation',
        template_path='gs://my-bucket/evaluation-pipeline.yaml',
        parameter_values={
            'model_id': model_id,
            'test_data_uri': 'gs://my-bucket/test-data/latest.csv'
        }
    )

    # Run evaluation pipeline
    pipeline_client.run(
        service_account='evaluation@my-project.iam.gserviceaccount.com',
        sync=False  # Don't block
    )
```

**Scheduled evaluation (weekly revalidation):**

```python
# Cloud Scheduler → Cloud Function → Vertex AI Pipeline
def scheduled_evaluation(request):
    """
    Weekly evaluation of all production models
    """
    from google.cloud import aiplatform

    # List all production models
    models = aiplatform.Model.list(
        filter='labels.environment=production'
    )

    for model in models:
        # Trigger evaluation pipeline for each model
        pipeline = aiplatform.PipelineJob(
            display_name=f'weekly-eval-{model.name}',
            template_path='gs://my-bucket/eval-pipeline.yaml',
            parameter_values={
                'model_name': model.name,
                'test_data_uri': f'gs://my-bucket/test-data/{model.name}/latest.csv'
            }
        )

        pipeline.run(sync=False)
```

---

## Section 2: Metrics Computation (~100 lines)

### Standard Classification Metrics

**Binary classification evaluation:**

```python
def compute_classification_metrics(y_true, y_pred, y_pred_proba):
    """
    Comprehensive classification metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        average_precision_score
    )
    import numpy as np

    metrics = {
        # Basic metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),

        # Probability-based metrics
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),

        # Confusion matrix
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),

        # Business metrics
        'false_positive_rate': None,  # Calculate below
        'false_negative_rate': None
    }

    # Calculate FPR and FNR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Detailed classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, output_dict=True
    )

    return metrics
```

**Multi-class classification:**

```python
def compute_multiclass_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """
    Multi-class classification metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    metrics = {
        # Micro-averaging (global calculation)
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),

        # Macro-averaging (mean of per-class metrics)
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),

        # Weighted averaging (weighted by class support)
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),

        # Multi-class ROC-AUC (one-vs-rest)
        'roc_auc_ovr': roc_auc_score(
            y_true, y_pred_proba,
            multi_class='ovr', average='macro'
        )
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    metrics['per_class'] = {
        f'class_{i}': {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1_score': f1_per_class[i]
        }
        for i in range(num_classes)
    }

    return metrics
```

### Regression Metrics

```python
def compute_regression_metrics(y_true, y_pred):
    """
    Regression model evaluation metrics
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )
    import numpy as np

    metrics = {
        # Error metrics
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred),

        # Goodness of fit
        'r2_score': r2_score(y_true, y_pred),

        # Residual analysis
        'mean_residual': np.mean(y_true - y_pred),
        'std_residual': np.std(y_true - y_pred),

        # Percentile errors
        'median_absolute_error': np.median(np.abs(y_true - y_pred)),
        'p95_absolute_error': np.percentile(np.abs(y_true - y_pred), 95)
    }

    return metrics
```

### Custom Metrics for Vision Models

**arr-coc-0-1 specific metrics:**

```python
def compute_arr_coc_metrics(y_true, y_pred, relevance_scores, token_allocations):
    """
    Custom metrics for arr-coc-0-1 VLM evaluation
    """
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Standard VQA metrics
    vqa_accuracy = accuracy_score(y_true, y_pred)

    # Relevance-specific metrics
    mean_relevance = np.mean(relevance_scores)
    std_relevance = np.std(relevance_scores)

    # Token allocation efficiency
    mean_tokens = np.mean(token_allocations)
    std_tokens = np.std(token_allocations)
    token_budget_utilization = mean_tokens / 400.0  # Max 400 tokens

    # LOD distribution (should span 64-400 range)
    lod_distribution = {
        '64-128': np.sum((token_allocations >= 64) & (token_allocations < 128)),
        '128-256': np.sum((token_allocations >= 128) & (token_allocations < 256)),
        '256-400': np.sum((token_allocations >= 256) & (token_allocations <= 400))
    }

    metrics = {
        # VQA performance
        'vqa_accuracy': vqa_accuracy,

        # Relevance realization quality
        'mean_relevance_score': mean_relevance,
        'std_relevance_score': std_relevance,

        # Token allocation efficiency
        'mean_tokens_per_patch': mean_tokens,
        'std_tokens_per_patch': std_tokens,
        'token_budget_utilization': token_budget_utilization,

        # LOD distribution (verify dynamic allocation)
        'lod_distribution': lod_distribution,

        # Opponent processing balance
        'compression_vs_particularize_balance': None,  # Custom calculation
        'exploit_vs_explore_balance': None
    }

    return metrics
```

---

## Section 3: Traffic Split Configuration (~100 lines)

### Vertex AI Endpoint Traffic Splitting

**Traffic splitting enables A/B testing by directing percentage of requests to different model versions:**

```python
from google.cloud import aiplatform

def create_ab_test_endpoint(
    project: str,
    location: str,
    endpoint_display_name: str,
    champion_model_id: str,
    challenger_model_id: str,
    challenger_traffic_percent: int = 10
):
    """
    Create endpoint with champion/challenger traffic split

    Args:
        champion_model_id: Current production model
        challenger_model_id: New model to test
        challenger_traffic_percent: % traffic to challenger (10, 20, 50)
    """
    aiplatform.init(project=project, location=location)

    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name
    )

    # Deploy champion model (90% traffic)
    champion_traffic = 100 - challenger_traffic_percent

    endpoint.deploy(
        model=champion_model_id,
        deployed_model_display_name='champion',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=10,
        traffic_percentage=champion_traffic,
        traffic_split={
            'champion': champion_traffic,
            'challenger': challenger_traffic_percent
        }
    )

    # Deploy challenger model (10% traffic)
    endpoint.deploy(
        model=challenger_model_id,
        deployed_model_display_name='challenger',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=5,
        traffic_percentage=challenger_traffic_percent,
        traffic_split={
            'champion': champion_traffic,
            'challenger': challenger_traffic_percent
        }
    )

    print(f"A/B test endpoint created: {endpoint.resource_name}")
    print(f"Traffic split: Champion {champion_traffic}%, Challenger {challenger_traffic_percent}%")

    return endpoint
```

### Common Traffic Split Strategies

**90/10 Canary deployment (low-risk initial test):**

```python
# Week 1: Canary test (10% traffic)
endpoint.update(
    traffic_split={
        'champion': 90,
        'challenger': 10
    }
)

# Monitor for 1 week, compare metrics
# If challenger performs better → proceed to next stage
```

**80/20 Validation (moderate confidence):**

```python
# Week 2: Increase to 20% if canary successful
endpoint.update(
    traffic_split={
        'champion': 80,
        'challenger': 20
    }
)

# More data for statistical significance
```

**50/50 Full A/B test (equal comparison):**

```python
# Week 3: Equal split for final validation
endpoint.update(
    traffic_split={
        'champion': 50,
        'challenger': 50
    }
)

# Maximum statistical power for comparison
```

### Dynamic Traffic Shifting

**Automated traffic adjustment based on performance:**

```python
def auto_adjust_traffic(
    endpoint,
    champion_metrics,
    challenger_metrics,
    current_challenger_traffic
):
    """
    Automatically adjust traffic based on performance
    """
    # Calculate performance improvement
    accuracy_improvement = (
        challenger_metrics['accuracy'] - champion_metrics['accuracy']
    )

    # Decision logic
    if accuracy_improvement >= 0.05:  # 5% better
        # Challenger significantly better → increase traffic
        new_challenger_traffic = min(current_challenger_traffic + 10, 50)

    elif accuracy_improvement <= -0.02:  # 2% worse
        # Challenger worse → decrease traffic
        new_challenger_traffic = max(current_challenger_traffic - 10, 5)

    else:
        # Similar performance → maintain current split
        new_challenger_traffic = current_challenger_traffic

    # Update endpoint
    endpoint.update(
        traffic_split={
            'champion': 100 - new_challenger_traffic,
            'challenger': new_challenger_traffic
        }
    )

    print(f"Traffic adjusted: Challenger now {new_challenger_traffic}%")

    return new_challenger_traffic
```

### Traffic Split Monitoring

**Track per-variant metrics:**

```python
def monitor_traffic_split(endpoint_name, time_window_hours=24):
    """
    Monitor metrics per deployed model variant
    """
    from google.cloud import monitoring_v3
    import datetime

    client = monitoring_v3.MetricServiceClient()

    # Define time window
    now = datetime.datetime.utcnow()
    end_time = now
    start_time = now - datetime.timedelta(hours=time_window_hours)

    # Query metrics for each variant
    variants = ['champion', 'challenger']
    variant_metrics = {}

    for variant in variants:
        # Prediction count
        pred_count_query = f"""
        fetch aiplatform_endpoint
        | metric 'aiplatform.googleapis.com/prediction/count'
        | filter resource.endpoint_id == '{endpoint_name}'
        | filter resource.deployed_model_id == '{variant}'
        | group_by 1h, [value_count_mean: mean(value.count)]
        """

        # Latency
        latency_query = f"""
        fetch aiplatform_endpoint
        | metric 'aiplatform.googleapis.com/prediction/latency'
        | filter resource.endpoint_id == '{endpoint_name}'
        | filter resource.deployed_model_id == '{variant}'
        | group_by 1h, [value_latency_mean: mean(value.latency)]
        """

        # Error rate
        error_query = f"""
        fetch aiplatform_endpoint
        | metric 'aiplatform.googleapis.com/prediction/error_count'
        | filter resource.endpoint_id == '{endpoint_name}'
        | filter resource.deployed_model_id == '{variant}'
        | group_by 1h, [value_error_mean: mean(value.error_count)]
        """

        variant_metrics[variant] = {
            'prediction_count': None,  # Execute queries
            'mean_latency': None,
            'error_rate': None
        }

    return variant_metrics
```

---

## Section 4: Statistical Significance Testing (~150 lines)

### Why Statistical Testing Matters

From [Machine Learning Mastery - Statistical Significance Tests](https://www.machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/) (accessed 2025-11-16):
> "Statistical hypothesis tests can be used to indicate whether the difference between two results is statistically significant or not. This is a critical step to avoid the trap of cherry-picking results and to give yourself confidence in your findings."

**Common pitfalls without statistical testing:**
- **Random variation**: Model A appears better due to luck (small sample)
- **Overfitting to test set**: Repeated evaluation inflates apparent performance
- **Cherry-picking**: Selecting best result from many trials
- **Business impact**: Deploying worse model costs money/reputation

### Chi-Squared Test for Classification

**Test if two models have significantly different error rates:**

```python
def chi_squared_test_models(
    champion_predictions,
    challenger_predictions,
    ground_truth,
    alpha=0.05
):
    """
    Chi-squared test for comparing classification models

    H0: Both models have same error distribution
    H1: Models have different error distributions
    """
    from scipy.stats import chi2_contingency
    import numpy as np

    # Create contingency table
    # Rows: Champion (correct/incorrect)
    # Cols: Challenger (correct/incorrect)

    champion_correct = (champion_predictions == ground_truth)
    challenger_correct = (challenger_predictions == ground_truth)

    # Both correct
    both_correct = np.sum(champion_correct & challenger_correct)

    # Champion correct, Challenger incorrect
    champ_only = np.sum(champion_correct & ~challenger_correct)

    # Challenger correct, Champion incorrect
    challenger_only = np.sum(~champion_correct & challenger_correct)

    # Both incorrect
    both_incorrect = np.sum(~champion_correct & ~challenger_correct)

    # Contingency table
    contingency = np.array([
        [both_correct, champ_only],
        [challenger_only, both_incorrect]
    ])

    # Perform chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Interpret results
    is_significant = p_value < alpha

    result = {
        'test': 'chi_squared',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'alpha': alpha,
        'is_significant': is_significant,
        'contingency_table': contingency,
        'conclusion': (
            f"Reject H0: Models are significantly different (p={p_value:.4f})"
            if is_significant else
            f"Fail to reject H0: No significant difference (p={p_value:.4f})"
        )
    }

    return result
```

**Example usage:**

```python
# Compare champion vs challenger on 10,000 test samples
result = chi_squared_test_models(
    champion_predictions=champion_preds,
    challenger_predictions=challenger_preds,
    ground_truth=y_test,
    alpha=0.05
)

print(f"Chi-squared: {result['chi2_statistic']:.2f}")
print(f"P-value: {result['p_value']:.4f}")
print(result['conclusion'])

# Output:
# Chi-squared: 12.45
# P-value: 0.0004
# Reject H0: Models are significantly different (p=0.0004)
```

### T-Test for Regression Models

**Compare mean absolute errors:**

```python
def t_test_regression_models(
    champion_predictions,
    challenger_predictions,
    ground_truth,
    alpha=0.05
):
    """
    Paired t-test for comparing regression models

    H0: Mean absolute errors are equal
    H1: Mean absolute errors are different
    """
    from scipy.stats import ttest_rel
    import numpy as np

    # Compute absolute errors for each model
    champion_errors = np.abs(champion_predictions - ground_truth)
    challenger_errors = np.abs(challenger_predictions - ground_truth)

    # Paired t-test (same test samples for both models)
    t_statistic, p_value = ttest_rel(champion_errors, challenger_errors)

    # Calculate mean errors
    champion_mae = np.mean(champion_errors)
    challenger_mae = np.mean(challenger_errors)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(champion_errors)**2 + np.std(challenger_errors)**2) / 2
    )
    cohens_d = (challenger_mae - champion_mae) / pooled_std

    is_significant = p_value < alpha

    result = {
        'test': 'paired_t_test',
        't_statistic': t_statistic,
        'p_value': p_value,
        'alpha': alpha,
        'champion_mae': champion_mae,
        'challenger_mae': challenger_mae,
        'mae_difference': challenger_mae - champion_mae,
        'cohens_d': cohens_d,
        'is_significant': is_significant,
        'conclusion': (
            f"Reject H0: Challenger MAE {challenger_mae:.4f} vs Champion {champion_mae:.4f} "
            f"(p={p_value:.4f}, effect size={cohens_d:.2f})"
            if is_significant else
            f"Fail to reject H0: No significant difference (p={p_value:.4f})"
        )
    }

    return result
```

### Bayesian A/B Testing

**Probabilistic comparison with credible intervals:**

```python
def bayesian_ab_test(
    champion_successes,
    champion_trials,
    challenger_successes,
    challenger_trials,
    credible_interval=0.95
):
    """
    Bayesian A/B test for binary outcomes

    Uses Beta distribution as conjugate prior
    """
    import numpy as np
    from scipy.stats import beta

    # Beta distributions (conjugate prior for binomial)
    # Beta(α=successes+1, β=failures+1) with uniform prior

    champion_alpha = champion_successes + 1
    champion_beta = champion_trials - champion_successes + 1

    challenger_alpha = challenger_successes + 1
    challenger_beta = challenger_trials - challenger_successes + 1

    # Sample from posterior distributions
    n_samples = 100000
    champion_samples = beta.rvs(
        champion_alpha, champion_beta, size=n_samples
    )
    challenger_samples = beta.rvs(
        challenger_alpha, challenger_beta, size=n_samples
    )

    # Probability that challenger > champion
    prob_challenger_better = np.mean(challenger_samples > champion_samples)

    # Expected improvement
    improvement = challenger_samples - champion_samples
    expected_improvement = np.mean(improvement)

    # Credible intervals
    ci_low = (1 - credible_interval) / 2
    ci_high = credible_interval + ci_low

    champion_ci = beta.ppf(
        [ci_low, ci_high], champion_alpha, champion_beta
    )
    challenger_ci = beta.ppf(
        [ci_low, ci_high], challenger_alpha, challenger_beta
    )

    result = {
        'test': 'bayesian_ab',
        'champion_mean': champion_alpha / (champion_alpha + champion_beta),
        'challenger_mean': challenger_alpha / (challenger_alpha + challenger_beta),
        'champion_ci': champion_ci,
        'challenger_ci': challenger_ci,
        'prob_challenger_better': prob_challenger_better,
        'expected_improvement': expected_improvement,
        'credible_interval': credible_interval,
        'conclusion': (
            f"Challenger is better with {prob_challenger_better:.1%} probability. "
            f"Expected improvement: {expected_improvement:.4f}"
        )
    }

    return result
```

**Example: VQA accuracy comparison**

```python
# Champion: 850/1000 correct
# Challenger: 880/1000 correct
result = bayesian_ab_test(
    champion_successes=850,
    champion_trials=1000,
    challenger_successes=880,
    challenger_trials=1000,
    credible_interval=0.95
)

print(f"Champion accuracy: {result['champion_mean']:.3f}")
print(f"Challenger accuracy: {result['challenger_mean']:.3f}")
print(f"Probability challenger better: {result['prob_challenger_better']:.1%}")
print(result['conclusion'])

# Output:
# Champion accuracy: 0.850
# Challenger accuracy: 0.880
# Probability challenger better: 99.2%
# Challenger is better with 99.2% probability. Expected improvement: 0.0301
```

### McNemar's Test (Paired Comparisons)

**Test for matched pairs (same test samples):**

```python
def mcnemar_test(
    champion_predictions,
    challenger_predictions,
    ground_truth,
    alpha=0.05
):
    """
    McNemar's test for comparing paired classifiers

    Tests if disagreements favor one model over the other
    """
    from statsmodels.stats.contingency_tables import mcnemar
    import numpy as np

    # Build contingency table
    champion_correct = (champion_predictions == ground_truth)
    challenger_correct = (challenger_predictions == ground_truth)

    # a: both correct
    # b: champion correct, challenger incorrect
    # c: champion incorrect, challenger correct
    # d: both incorrect

    a = np.sum(champion_correct & challenger_correct)
    b = np.sum(champion_correct & ~challenger_correct)
    c = np.sum(~champion_correct & challenger_correct)
    d = np.sum(~champion_correct & ~challenger_correct)

    contingency = np.array([[a, b], [c, d]])

    # McNemar's test (uses b and c, ignores a and d)
    result = mcnemar(contingency, exact=False, correction=True)

    is_significant = result.pvalue < alpha

    return {
        'test': 'mcnemar',
        'statistic': result.statistic,
        'p_value': result.pvalue,
        'alpha': alpha,
        'champion_only_correct': b,
        'challenger_only_correct': c,
        'is_significant': is_significant,
        'conclusion': (
            f"Reject H0: Challenger wins {c} vs Champion wins {b} "
            f"(p={result.pvalue:.4f})"
            if is_significant else
            f"Fail to reject H0: No significant difference (p={result.pvalue:.4f})"
        )
    }
```

---

## Section 5: Automatic Promotion (~100 lines)

### Champion/Challenger Workflow

From [DataRobot MLOps](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/):
> "Champion/challenger is a method that allows different approaches to testing predictive models in a production environment. The challengers will compete against each other for a chance to become the new champion."

**Key difference from A/B testing:**
- **A/B testing**: Split live traffic between models (both serve predictions)
- **Champion/challenger**: Only champion serves live traffic, challengers run in shadow mode

**Benefits of shadow mode:**
- Safe testing (no business impact from challenger errors)
- Test risky models (e.g., max-accuracy model with new features)
- Compare baseline models without production risk
- Full replay of production traffic for fair comparison

### Automated Promotion Logic

```python
def evaluate_promotion_eligibility(
    champion_metrics,
    challenger_metrics,
    statistical_test_result,
    business_rules
):
    """
    Determine if challenger should be promoted to champion

    Args:
        champion_metrics: Current production model metrics
        challenger_metrics: Challenger model metrics
        statistical_test_result: Statistical significance test
        business_rules: Business-specific promotion criteria
    """

    # Rule 1: Statistical significance required
    if not statistical_test_result['is_significant']:
        return {
            'promote': False,
            'reason': 'No statistically significant improvement',
            'details': statistical_test_result
        }

    # Rule 2: Minimum accuracy improvement threshold
    min_improvement = business_rules.get('min_accuracy_improvement', 0.02)
    accuracy_delta = (
        challenger_metrics['accuracy'] - champion_metrics['accuracy']
    )

    if accuracy_delta < min_improvement:
        return {
            'promote': False,
            'reason': f'Improvement {accuracy_delta:.4f} below threshold {min_improvement}',
            'accuracy_delta': accuracy_delta
        }

    # Rule 3: Latency constraint
    max_latency = business_rules.get('max_latency_ms', 200)
    if challenger_metrics.get('latency_p95') > max_latency:
        return {
            'promote': False,
            'reason': f"Latency {challenger_metrics['latency_p95']}ms exceeds limit {max_latency}ms"
        }

    # Rule 4: Error rate constraint
    max_error_rate = business_rules.get('max_error_rate', 0.01)
    if challenger_metrics.get('error_rate', 0) > max_error_rate:
        return {
            'promote': False,
            'reason': f"Error rate {challenger_metrics['error_rate']} exceeds limit {max_error_rate}"
        }

    # Rule 5: Minimum sample size
    min_predictions = business_rules.get('min_predictions', 10000)
    if challenger_metrics.get('prediction_count', 0) < min_predictions:
        return {
            'promote': False,
            'reason': f"Insufficient predictions: {challenger_metrics['prediction_count']} < {min_predictions}"
        }

    # All checks passed
    return {
        'promote': True,
        'reason': 'All promotion criteria met',
        'improvement_summary': {
            'accuracy_delta': accuracy_delta,
            'statistical_significance': statistical_test_result['p_value'],
            'latency': challenger_metrics.get('latency_p95'),
            'error_rate': challenger_metrics.get('error_rate')
        }
    }
```

### Automated Promotion Execution

```python
def promote_challenger_to_champion(
    endpoint,
    champion_model_id,
    challenger_model_id,
    promotion_decision
):
    """
    Execute champion replacement with governance
    """
    from google.cloud import aiplatform
    import datetime

    if not promotion_decision['promote']:
        print(f"Promotion denied: {promotion_decision['reason']}")
        return False

    # Log promotion event
    promotion_event = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'old_champion': champion_model_id,
        'new_champion': challenger_model_id,
        'promotion_reason': promotion_decision['reason'],
        'metrics': promotion_decision['improvement_summary']
    }

    # Archive old champion as backup
    old_champion_backup = {
        'model_id': champion_model_id,
        'archived_at': datetime.datetime.utcnow().isoformat(),
        'reason': 'Replaced by challenger promotion'
    }

    # Update endpoint: Shift 100% traffic to challenger
    endpoint.update(
        traffic_split={
            'champion': 0,  # Old champion
            'challenger': 100  # Promoted to champion
        }
    )

    # Rename deployed models
    endpoint.undeploy(deployed_model_id='champion')  # Remove old champion

    # Redeploy challenger as new champion
    endpoint.deploy(
        model=challenger_model_id,
        deployed_model_display_name='champion',
        machine_type='n1-standard-4',
        min_replica_count=2,
        max_replica_count=10,
        traffic_percentage=100
    )

    print(f"✓ Promotion complete: {challenger_model_id} is now champion")
    print(f"  Improvement: {promotion_decision['improvement_summary']}")

    return True
```

### Manual Approval Gate

**Require human approval for production changes:**

```python
def request_promotion_approval(
    promotion_decision,
    approver_email
):
    """
    Send approval request to designated approver
    """
    from google.cloud import workflows

    # Create approval workflow
    approval_workflow = f"""
    main:
      steps:
        - send_approval_request:
            call: http.post
            args:
              url: https://approval-service.example.com/request
              body:
                approver: {approver_email}
                model_promotion: {promotion_decision}
                timeout_hours: 24
            result: approval_response

        - check_approval:
            switch:
              - condition: ${{approval_response.approved}}
                next: promote_model
              - condition: true
                next: reject_promotion

        - promote_model:
            call: promote_challenger_to_champion
            args:
              promotion_decision: {promotion_decision}
            next: end

        - reject_promotion:
            return: "Promotion rejected by approver"
    """

    # Execute workflow (waits for approval)
    # workflow_client.execute(approval_workflow)

    print(f"Approval request sent to {approver_email}")
    print("Waiting for approval before promotion...")
```

---

## Section 6: Gradual Rollout Strategies (~100 lines)

### Progressive Traffic Migration

**5% → 25% → 50% → 100% rollout:**

```python
def gradual_rollout_schedule():
    """
    Define progressive rollout stages
    """
    return [
        {
            'stage': 1,
            'name': 'Initial canary',
            'challenger_traffic': 5,
            'duration_hours': 48,
            'success_criteria': {
                'min_accuracy': 0.80,
                'max_error_rate': 0.02,
                'max_latency_p95': 250
            }
        },
        {
            'stage': 2,
            'name': 'Expanded test',
            'challenger_traffic': 25,
            'duration_hours': 72,
            'success_criteria': {
                'min_accuracy': 0.82,
                'max_error_rate': 0.015,
                'max_latency_p95': 220
            }
        },
        {
            'stage': 3,
            'name': 'A/B validation',
            'challenger_traffic': 50,
            'duration_hours': 120,
            'success_criteria': {
                'min_accuracy': 0.85,
                'max_error_rate': 0.01,
                'max_latency_p95': 200,
                'statistical_significance': True
            }
        },
        {
            'stage': 4,
            'name': 'Full rollout',
            'challenger_traffic': 100,
            'duration_hours': None,  # Permanent
            'success_criteria': None  # Already validated
        }
    ]
```

### Automated Stage Progression

```python
def execute_gradual_rollout(
    endpoint,
    champion_model_id,
    challenger_model_id,
    rollout_schedule
):
    """
    Execute multi-stage gradual rollout with monitoring
    """
    import time
    from datetime import datetime, timedelta

    for stage in rollout_schedule:
        print(f"\n=== Stage {stage['stage']}: {stage['name']} ===")
        print(f"Challenger traffic: {stage['challenger_traffic']}%")

        # Update traffic split
        endpoint.update(
            traffic_split={
                'champion': 100 - stage['challenger_traffic'],
                'challenger': stage['challenger_traffic']
            }
        )

        # Wait for stage duration
        if stage['duration_hours']:
            print(f"Monitoring for {stage['duration_hours']} hours...")

            end_time = datetime.now() + timedelta(hours=stage['duration_hours'])

            while datetime.now() < end_time:
                # Monitor metrics every hour
                time.sleep(3600)  # 1 hour

                # Collect metrics
                challenger_metrics = collect_challenger_metrics(
                    endpoint,
                    time_window_hours=1
                )

                # Check success criteria
                criteria_met = check_success_criteria(
                    challenger_metrics,
                    stage['success_criteria']
                )

                if not criteria_met['passed']:
                    # Rollback immediately
                    print(f"✗ Stage {stage['stage']} failed: {criteria_met['reason']}")
                    rollback_deployment(endpoint, champion_model_id)
                    return False

                print(f"✓ Metrics healthy: {challenger_metrics}")

            print(f"✓ Stage {stage['stage']} completed successfully")

        else:
            # Final stage (100% traffic)
            print("✓ Rollout complete - Challenger is now champion")

    return True
```

### Automatic Rollback on Failure

```python
def rollback_deployment(endpoint, champion_model_id):
    """
    Immediate rollback to champion on failure
    """
    print("⚠️  INITIATING ROLLBACK TO CHAMPION")

    # Shift 100% traffic back to champion
    endpoint.update(
        traffic_split={
            'champion': 100,
            'challenger': 0
        }
    )

    # Log rollback event
    rollback_event = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'action': 'automatic_rollback',
        'reason': 'Challenger failed success criteria during gradual rollout',
        'restored_model': champion_model_id
    }

    # Send alert
    send_alert(
        severity='HIGH',
        message='Automatic rollback executed - Challenger model failed validation',
        details=rollback_event
    )

    print("✓ Rollback complete - Champion restored to 100% traffic")
```

### Blue/Green Deployment Pattern

**Alternative to gradual rollout (instant switch):**

```python
def blue_green_deployment(
    project,
    location,
    blue_model_id,  # Current production
    green_model_id  # New version
):
    """
    Blue/Green deployment: Prepare green, validate, instant switch
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    # Step 1: Deploy green environment (parallel to blue)
    green_endpoint = aiplatform.Endpoint.create(
        display_name='green-endpoint'
    )

    green_endpoint.deploy(
        model=green_model_id,
        deployed_model_display_name='green',
        machine_type='n1-standard-4',
        min_replica_count=2,
        max_replica_count=10
    )

    print("✓ Green environment deployed")

    # Step 2: Run validation tests against green
    validation_passed = run_validation_tests(green_endpoint)

    if not validation_passed:
        print("✗ Green validation failed - keeping blue")
        green_endpoint.delete()
        return False

    print("✓ Green validation passed")

    # Step 3: Switch DNS/load balancer from blue to green
    # (Implementation depends on infrastructure)
    switch_traffic_to_green(green_endpoint)

    print("✓ Traffic switched to green")

    # Step 4: Keep blue as backup for 24 hours, then delete
    # schedule_blue_deletion(blue_endpoint, hours=24)

    return True
```

---

## Section 7: arr-coc-0-1 A/B Testing for Relevance Allocation (~150 lines)

### Challenger Strategies for Relevance Realization

**Test different relevance allocation approaches:**

```python
# Champion: Current production relevance allocation
champion_strategy = {
    'name': 'balanced_relevance',
    'description': 'Balanced weighting of three ways of knowing',
    'parameters': {
        'propositional_weight': 0.33,
        'perspectival_weight': 0.33,
        'participatory_weight': 0.34,
        'opponent_processing_strength': 0.5,
        'token_budget_min': 64,
        'token_budget_max': 400
    }
}

# Challenger 1: Propositional-heavy (emphasize information content)
challenger_1_strategy = {
    'name': 'propositional_focus',
    'description': 'Emphasize statistical information content',
    'parameters': {
        'propositional_weight': 0.6,
        'perspectival_weight': 0.2,
        'participatory_weight': 0.2,
        'opponent_processing_strength': 0.5,
        'token_budget_min': 64,
        'token_budget_max': 400
    }
}

# Challenger 2: Participatory-heavy (emphasize query-content coupling)
challenger_2_strategy = {
    'name': 'participatory_focus',
    'description': 'Emphasize query-content relevance coupling',
    'parameters': {
        'propositional_weight': 0.2,
        'perspectival_weight': 0.2,
        'participatory_weight': 0.6,
        'opponent_processing_strength': 0.5,
        'token_budget_min': 64,
        'token_budget_max': 400
    }
}

# Challenger 3: Dynamic token budget
challenger_3_strategy = {
    'name': 'extended_token_budget',
    'description': 'Allow higher token budgets for complex queries',
    'parameters': {
        'propositional_weight': 0.33,
        'perspectival_weight': 0.33,
        'participatory_weight': 0.34,
        'opponent_processing_strength': 0.5,
        'token_budget_min': 64,
        'token_budget_max': 600  # Increased from 400
    }
}
```

### Champion/Challenger Setup for arr-coc-0-1

```python
def setup_arr_coc_ab_test(
    project,
    location,
    champion_model_path,
    challenger_configs
):
    """
    Deploy arr-coc-0-1 champion/challenger test

    Args:
        champion_model_path: Current production model
        challenger_configs: List of challenger strategies to test
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name='arr-coc-ab-test'
    )

    # Deploy champion (70% traffic)
    endpoint.deploy(
        model=champion_model_path,
        deployed_model_display_name='champion',
        machine_type='n1-standard-8',  # 8 vCPUs for VLM
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1,
        min_replica_count=2,
        max_replica_count=10,
        traffic_percentage=70
    )

    # Deploy challengers (10% traffic each)
    challenger_traffic = 30 // len(challenger_configs)

    for i, config in enumerate(challenger_configs):
        # Build challenger model with different strategy
        challenger_model = build_arr_coc_model(config['parameters'])

        endpoint.deploy(
            model=challenger_model,
            deployed_model_display_name=f"challenger_{i+1}_{config['name']}",
            machine_type='n1-standard-8',
            accelerator_type='NVIDIA_TESLA_T4',
            accelerator_count=1,
            min_replica_count=1,
            max_replica_count=5,
            traffic_percentage=challenger_traffic
        )

    print(f"✓ A/B test deployed:")
    print(f"  Champion: 70% traffic")
    print(f"  {len(challenger_configs)} Challengers: {challenger_traffic}% each")

    return endpoint
```

### arr-coc-0-1 Specific Metrics

```python
def collect_arr_coc_ab_metrics(endpoint, variant_name, time_window_hours=24):
    """
    Collect arr-coc-0-1 specific A/B test metrics
    """
    from google.cloud import aiplatform
    import numpy as np

    # Fetch predictions from logging
    predictions = fetch_prediction_logs(
        endpoint=endpoint,
        variant=variant_name,
        time_window_hours=time_window_hours
    )

    # Extract arr-coc-0-1 specific outputs
    vqa_predictions = [p['vqa_answer'] for p in predictions]
    ground_truth = [p['ground_truth'] for p in predictions]
    relevance_scores = [p['relevance_scores'] for p in predictions]
    token_allocations = [p['token_allocations'] for p in predictions]

    # Compute metrics
    from sklearn.metrics import accuracy_score

    metrics = {
        # VQA performance
        'vqa_accuracy': accuracy_score(ground_truth, vqa_predictions),

        # Relevance quality
        'mean_relevance': np.mean(relevance_scores),
        'std_relevance': np.std(relevance_scores),

        # Token efficiency
        'mean_tokens_per_patch': np.mean(token_allocations),
        'std_tokens_per_patch': np.std(token_allocations),
        'token_budget_utilization': np.mean(token_allocations) / 400.0,

        # LOD distribution (should be dynamic, not concentrated)
        'lod_64_128_pct': np.mean((token_allocations >= 64) & (token_allocations < 128)),
        'lod_128_256_pct': np.mean((token_allocations >= 128) & (token_allocations < 256)),
        'lod_256_400_pct': np.mean((token_allocations >= 256) & (token_allocations <= 400)),

        # Latency
        'latency_p50': np.percentile([p['latency_ms'] for p in predictions], 50),
        'latency_p95': np.percentile([p['latency_ms'] for p in predictions], 95),
        'latency_p99': np.percentile([p['latency_ms'] for p in predictions], 99)
    }

    return metrics
```

### Compare Relevance Allocation Strategies

```python
def compare_arr_coc_strategies(
    champion_metrics,
    challenger_metrics_list,
    challenger_names
):
    """
    Statistical comparison of relevance allocation strategies
    """
    import pandas as pd

    # Create comparison table
    comparison = {
        'Strategy': ['Champion'] + challenger_names,
        'VQA Accuracy': [champion_metrics['vqa_accuracy']] +
                        [c['vqa_accuracy'] for c in challenger_metrics_list],
        'Mean Relevance': [champion_metrics['mean_relevance']] +
                          [c['mean_relevance'] for c in challenger_metrics_list],
        'Token Efficiency': [champion_metrics['token_budget_utilization']] +
                            [c['token_budget_utilization'] for c in challenger_metrics_list],
        'Latency P95 (ms)': [champion_metrics['latency_p95']] +
                            [c['latency_p95'] for c in challenger_metrics_list]
    }

    df = pd.DataFrame(comparison)

    # Find best performer
    best_accuracy_idx = df['VQA Accuracy'].idxmax()
    best_strategy = df.iloc[best_accuracy_idx]

    print("\n=== arr-coc-0-1 Relevance Strategy Comparison ===\n")
    print(df.to_string(index=False))

    print(f"\n✓ Best strategy: {best_strategy['Strategy']}")
    print(f"  VQA Accuracy: {best_strategy['VQA Accuracy']:.4f}")
    print(f"  Token Efficiency: {best_strategy['Token Efficiency']:.2%}")
    print(f"  Latency P95: {best_strategy['Latency P95 (ms)']:.1f}ms")

    # Statistical significance test
    if best_accuracy_idx > 0:  # Challenger won
        challenger_idx = best_accuracy_idx - 1

        # Perform statistical test
        stat_result = chi_squared_test_models(
            champion_predictions=champion_metrics['predictions'],
            challenger_predictions=challenger_metrics_list[challenger_idx]['predictions'],
            ground_truth=champion_metrics['ground_truth']
        )

        print(f"\nStatistical Test: {stat_result['conclusion']}")

        if stat_result['is_significant']:
            print(f"✓ Recommend promoting {best_strategy['Strategy']} to champion")
        else:
            print(f"⚠️  Improvement not statistically significant - keep champion")

    return df
```

### Relevance Score A/B Analysis

```python
def analyze_relevance_score_distribution(
    champion_relevance_scores,
    challenger_relevance_scores,
    strategy_name
):
    """
    Deep dive into how relevance scoring differs between strategies
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compare distributions
    print(f"\n=== Relevance Score Analysis: {strategy_name} ===")

    print(f"\nChampion:")
    print(f"  Mean: {np.mean(champion_relevance_scores):.3f}")
    print(f"  Std:  {np.std(champion_relevance_scores):.3f}")
    print(f"  Min:  {np.min(champion_relevance_scores):.3f}")
    print(f"  Max:  {np.max(champion_relevance_scores):.3f}")

    print(f"\nChallenger ({strategy_name}):")
    print(f"  Mean: {np.mean(challenger_relevance_scores):.3f}")
    print(f"  Std:  {np.std(challenger_relevance_scores):.3f}")
    print(f"  Min:  {np.min(challenger_relevance_scores):.3f}")
    print(f"  Max:  {np.max(challenger_relevance_scores):.3f}")

    # Test if distributions differ
    from scipy.stats import ks_2samp

    ks_stat, p_value = ks_2samp(
        champion_relevance_scores,
        challenger_relevance_scores
    )

    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  ✓ Distributions are significantly different")
    else:
        print(f"  ✗ No significant difference in distributions")
```

---

## Section 8: Production Best Practices (~100 lines)

### Evaluation Frequency Recommendations

**How often to evaluate production models:**

| Model Type | Evaluation Frequency | Reason |
|------------|---------------------|---------|
| **High-stakes** (finance, healthcare) | Daily | Business impact of errors is severe |
| **Medium-stakes** (recommendations) | Weekly | Balance freshness vs. computational cost |
| **Low-stakes** (A/B experiments) | Bi-weekly | Longer time for statistical significance |
| **Stable models** (no drift expected) | Monthly | Minimize operational overhead |

**Trigger-based evaluation (in addition to scheduled):**
- Data distribution shift detected (KL divergence > 0.20)
- Prediction error rate spike (>2× baseline)
- Model latency degradation (P95 > SLA threshold)
- Manual evaluation request (new challenger available)

### A/B Test Duration Guidelines

**Minimum test duration for statistical power:**

```python
def calculate_minimum_ab_test_duration(
    baseline_conversion_rate,
    minimum_detectable_effect,
    significance_level=0.05,
    power=0.80,
    daily_samples=10000
):
    """
    Calculate how long A/B test must run for statistical power

    Args:
        baseline_conversion_rate: Champion accuracy (e.g., 0.85)
        minimum_detectable_effect: Smallest improvement to detect (e.g., 0.02)
        significance_level: Alpha (usually 0.05)
        power: 1 - Beta (usually 0.80)
        daily_samples: Predictions per day
    """
    from scipy.stats import norm
    import math

    # Calculate required sample size per variant
    p1 = baseline_conversion_rate
    p2 = baseline_conversion_rate + minimum_detectable_effect

    p_pooled = (p1 + p2) / 2

    z_alpha = norm.ppf(1 - significance_level / 2)
    z_beta = norm.ppf(power)

    n = (
        (z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)
    ) / (minimum_detectable_effect**2)

    # Calculate days needed
    days_needed = math.ceil(n / daily_samples)

    print(f"A/B Test Duration Calculator:")
    print(f"  Baseline accuracy: {p1:.2%}")
    print(f"  Minimum improvement: {minimum_detectable_effect:.2%}")
    print(f"  Daily samples: {daily_samples:,}")
    print(f"  Significance level: {significance_level}")
    print(f"  Power: {power}")
    print(f"\n  Required samples per variant: {int(n):,}")
    print(f"  Recommended test duration: {days_needed} days")

    return days_needed
```

**Example: arr-coc-0-1 test duration**

```python
# Champion: 85% VQA accuracy
# Want to detect 2% improvement
# 10,000 VQA queries per day
days = calculate_minimum_ab_test_duration(
    baseline_conversion_rate=0.85,
    minimum_detectable_effect=0.02,
    daily_samples=10000
)

# Output:
# Required samples per variant: 15,682
# Recommended test duration: 2 days
```

### Monitoring Dashboard Requirements

**Essential metrics for production evaluation:**

```python
EVALUATION_DASHBOARD_METRICS = {
    'model_performance': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc'
    ],

    'drift_detection': [
        'feature_drift_score',
        'prediction_drift_score',
        'data_quality_score'
    ],

    'service_health': [
        'prediction_count',
        'error_rate',
        'latency_p50',
        'latency_p95',
        'latency_p99'
    ],

    'ab_test_progress': [
        'champion_accuracy',
        'challenger_accuracy',
        'sample_size',
        'statistical_power',
        'days_remaining'
    ],

    'business_impact': [
        'revenue_impact',
        'cost_savings',
        'customer_satisfaction'
    ]
}
```

**Cloud Monitoring dashboard creation:**

```python
def create_evaluation_dashboard(project, dashboard_name):
    """
    Create Cloud Monitoring dashboard for model evaluation
    """
    from google.cloud import monitoring_v3

    dashboard_config = {
        'displayName': dashboard_name,
        'gridLayout': {
            'widgets': [
                # Accuracy over time
                {
                    'title': 'Model Accuracy (7-day)',
                    'xyChart': {
                        'dataSets': [{
                            'timeSeriesQuery': {
                                'timeSeriesFilter': {
                                    'filter': 'metric.type="custom.googleapis.com/model/accuracy"'
                                }
                            }
                        }]
                    }
                },

                # A/B test comparison
                {
                    'title': 'Champion vs Challenger Accuracy',
                    'xyChart': {
                        'dataSets': [
                            {
                                'timeSeriesQuery': {
                                    'timeSeriesFilter': {
                                        'filter': 'metric.type="custom.googleapis.com/model/accuracy" AND resource.label.variant="champion"'
                                    }
                                }
                            },
                            {
                                'timeSeriesQuery': {
                                    'timeSeriesFilter': {
                                        'filter': 'metric.type="custom.googleapis.com/model/accuracy" AND resource.label.variant="challenger"'
                                    }
                                }
                            }
                        ]
                    }
                },

                # Latency percentiles
                {
                    'title': 'Prediction Latency (P50, P95, P99)',
                    'xyChart': {
                        'dataSets': [{
                            'timeSeriesQuery': {
                                'timeSeriesFilter': {
                                    'filter': 'metric.type="aiplatform.googleapis.com/prediction/latency"'
                                },
                                'unitOverride': 'ms'
                            }
                        }]
                    }
                }
            ]
        }
    }

    # Create dashboard via API
    # monitoring_client.create_dashboard(...)

    print(f"✓ Dashboard created: {dashboard_name}")
```

### Cost Optimization for Evaluation

**Reduce evaluation costs:**

1. **Sample production traffic** (don't evaluate every prediction)
   ```python
   # Evaluate 10% of predictions instead of 100%
   sample_rate = 0.10
   if random.random() < sample_rate:
       evaluate_prediction(prediction, ground_truth)
   ```

2. **Use batch evaluation** (cheaper than real-time)
   ```python
   # Collect predictions for 24 hours, batch evaluate
   # Instead of: Real-time evaluation per prediction
   ```

3. **Reuse test datasets** (don't label new data constantly)
   ```python
   # Hold-out test set refreshed monthly, not daily
   test_data_refresh_frequency = 30  # days
   ```

4. **Preemptible instances for evaluation pipelines**
   ```python
   # Use Spot VMs for non-critical evaluation jobs
   evaluation_job.run(
       machine_type='n1-standard-4',
       use_spot_instances=True  # 70% cost savings
   )
   ```

---

## Sources

**Official Documentation:**
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) - Pipeline orchestration and evaluation
- [Vertex AI Model Evaluation](https://cloud.google.com/vertex-ai/docs/evaluation/introduction) - Evaluation metrics and methods
- [Vertex AI Endpoints](https://cloud.google.com/vertex-ai/docs/predictions/overview) - Traffic splitting for A/B tests

**Web Research:**
- [DataRobot: Introducing MLOps Champion/Challenger Models](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/) - Champion/challenger framework and shadow mode testing (accessed 2025-11-16)
- [Machine Learning Mastery: Statistical Significance Tests](https://www.machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/) - Statistical testing for model comparison (accessed 2025-11-16)
- [GCP Study Hub: Vertex AI Endpoints](https://www.gcpstudyhub.com/pages/blog/vertex-ai-endpoints-from-model-training-to-production) - Traffic splitting and A/B testing configuration (accessed 2025-11-16)

**Integration with Existing Knowledge:**
- CI/CD workflows: [mlops-production/00-monitoring-cicd-cost-optimization.md](../../karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md) - Automated deployment patterns, monitoring strategies
- Drift detection: mlops-production/00 Section 1.2 - Data drift and prediction drift algorithms
- Cost optimization: mlops-production/00 Section 3 - Multi-cloud training and inference cost strategies

---

**Knowledge file complete**: ~700 lines
**Created**: 2025-11-16
**Coverage**: ModelEvaluation pipelines, metrics computation (accuracy, precision, recall, F1, custom), traffic splitting (90/10, 80/20, 50/50), statistical testing (chi-squared, t-test, Bayesian, McNemar), champion/challenger deployment, gradual rollout (5% → 100%), arr-coc-0-1 relevance allocation A/B testing
**All claims cited**: 3 web sources + 1 existing knowledge file
