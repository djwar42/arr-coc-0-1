# Statistical Testing & A/B Comparison for Gradio VLM Interfaces (2025)

## Overview

Statistical testing is critical for validating VLM model improvements and ensuring that observed performance differences are not due to random chance. This guide covers A/B testing patterns, statistical significance methods, metrics collection, and automated validation workflows specifically for Gradio-based vision-language model interfaces.

**Why Statistical Testing Matters for VLM Validation:**
- Prevents false conclusions from small sample sizes or random variance
- Quantifies confidence in model comparison results
- Enables data-driven checkpoint selection
- Supports reproducible research with proper effect size reporting
- Avoids common pitfalls like p-hacking and multiple comparison issues

## Section 1: Statistical Testing Fundamentals

### The Problem with P-Values Alone

**Common Pitfalls in VLM Evaluation:**
```python
# ❌ BAD: P-value only tells you IF there's a difference
def bad_comparison(model_a_scores, model_b_scores):
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(model_a_scores, model_b_scores)

    if p_value < 0.05:
        return "Model B is better!"  # Dangerous conclusion
    return "No difference"
```

**Problems:**
1. **P-value doesn't tell you HOW MUCH better** - could be 0.1% or 10% improvement
2. **Sample size dependency** - huge datasets make tiny differences "significant"
3. **Multiple comparison problem** - testing 20 metrics gives 1 false positive by chance
4. **Ignores practical significance** - statistically significant ≠ practically useful

**From** [Medium - Statistical Significance in A/B Testing: Beyond P-Hacking in 2025](https://medium.com/@tod01/testing-your-machine-learning-model-with-gradio-969c87ea03ab) (accessed 2025-10-31):
> "It provides a crucial measure of precision; a wide interval suggests uncertainty, even if the result is statistically significant."

### Effect Size vs Statistical Significance

**Always report both:**
```python
import numpy as np
from scipy import stats

def proper_comparison(model_a_scores, model_b_scores):
    """
    Complete statistical comparison with effect size
    """
    # Statistical significance (p-value)
    t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)

    # Effect size (Cohen's d)
    mean_diff = np.mean(model_b_scores) - np.mean(model_a_scores)
    pooled_std = np.sqrt((np.std(model_a_scores)**2 +
                          np.std(model_b_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std

    # Confidence interval (95%)
    ci_lower, ci_upper = stats.t.interval(
        0.95,
        len(model_a_scores) + len(model_b_scores) - 2,
        loc=mean_diff,
        scale=stats.sem(np.concatenate([model_a_scores, model_b_scores]))
    )

    return {
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d),
        'mean_difference': mean_diff,
        'ci_95': (ci_lower, ci_upper)
    }

def interpret_cohens_d(d):
    """Cohen's d interpretation guidelines"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

**Cohen's d Interpretation** (from [Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00863/full), accessed 2025-10-31):
- d < 0.2: Negligible effect
- 0.2 ≤ d < 0.5: Small effect
- 0.5 ≤ d < 0.8: Medium effect
- d ≥ 0.8: Large effect

### Multiple Comparison Correction

**When testing multiple metrics/models:**
```python
from statsmodels.stats.multitest import multipletests

def multiple_comparison_safe(models_scores, metric_names, alpha=0.05):
    """
    Proper handling of multiple comparisons
    """
    p_values = []
    comparisons = []

    # Collect all p-values
    baseline = models_scores[0]
    for i, model_scores in enumerate(models_scores[1:], 1):
        for metric_name in metric_names:
            _, p_val = stats.ttest_ind(
                baseline[metric_name],
                model_scores[metric_name]
            )
            p_values.append(p_val)
            comparisons.append(f"Model_{i}_vs_Baseline_{metric_name}")

    # Bonferroni correction (conservative)
    reject, p_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method='bonferroni'
    )

    # Benjamini-Hochberg (less conservative, controls FDR)
    reject_bh, p_corrected_bh, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method='fdr_bh'
    )

    return {
        'comparisons': comparisons,
        'raw_p_values': p_values,
        'bonferroni_corrected': p_corrected,
        'bonferroni_reject': reject,
        'bh_corrected': p_corrected_bh,
        'bh_reject': reject_bh
    }
```

**From** [ResearchGate - Statistical Significance Testing in ML Model Comparisons](https://www.researchgate.net/publication/392727623_Statistical_Significance_Testing_in_ML_Model_Comparisons_Beyond_p-values_and_t-tests) (accessed 2025-10-31):
> "This article advocates for a paradigm shift towards more robust and informative statistical approaches for ML model evaluation."

## Section 2: A/B Comparison Patterns for Gradio

### Side-by-Side Checkpoint Comparison UI

```python
import gradio as gr
import numpy as np
from collections import defaultdict

class CheckpointComparator:
    """
    A/B testing interface for VLM checkpoints
    """
    def __init__(self):
        self.results_buffer = defaultdict(lambda: {'A': [], 'B': []})

    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# VLM Checkpoint A/B Comparison")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model A (Checkpoint 1)")
                    image_a = gr.Image(type="pil", label="Input Image")
                    query_a = gr.Textbox(label="Query")
                    output_a = gr.Textbox(label="Model A Output", interactive=False)
                    confidence_a = gr.Number(label="Confidence Score", interactive=False)

                with gr.Column():
                    gr.Markdown("### Model B (Checkpoint 2)")
                    image_b = gr.Image(type="pil", label="Input Image (synced)")
                    query_b = gr.Textbox(label="Query (synced)")
                    output_b = gr.Textbox(label="Model B Output", interactive=False)
                    confidence_b = gr.Number(label="Confidence Score", interactive=False)

            # Difference highlighting
            with gr.Row():
                gr.Markdown("### Statistical Comparison")
                stat_output = gr.JSON(label="Current Statistics")

            # User feedback for subjective quality
            with gr.Row():
                gr.Markdown("**Which output is better?**")
                better_choice = gr.Radio(
                    choices=["Model A", "Model B", "Tie"],
                    label="Human Preference"
                )
                submit_feedback = gr.Button("Submit Feedback")

            # Run comparison button
            run_btn = gr.Button("Run Comparison", variant="primary")

            # Wire up events
            run_btn.click(
                fn=self.run_comparison,
                inputs=[image_a, query_a],
                outputs=[output_a, confidence_a, output_b, confidence_b, stat_output]
            )

            submit_feedback.click(
                fn=self.record_feedback,
                inputs=[better_choice, confidence_a, confidence_b],
                outputs=[stat_output]
            )

        return demo

    def run_comparison(self, image, query):
        """Run both models and return outputs"""
        # Model A inference
        output_a, conf_a = self.infer_model_a(image, query)

        # Model B inference
        output_b, conf_b = self.infer_model_b(image, query)

        # Store results
        self.results_buffer['confidence']['A'].append(conf_a)
        self.results_buffer['confidence']['B'].append(conf_b)

        # Calculate current statistics
        stats = self.calculate_statistics()

        return output_a, conf_a, output_b, conf_b, stats

    def record_feedback(self, choice, conf_a, conf_b):
        """Record human preference"""
        self.results_buffer['human_preference'][choice].append(1)

        # Update statistics
        return self.calculate_statistics()

    def calculate_statistics(self):
        """Calculate and return current statistical comparison"""
        if len(self.results_buffer['confidence']['A']) < 2:
            return {"message": "Need more samples (n >= 2)"}

        scores_a = np.array(self.results_buffer['confidence']['A'])
        scores_b = np.array(self.results_buffer['confidence']['B'])

        # T-test
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

        # Effect size
        mean_diff = np.mean(scores_b) - np.mean(scores_a)
        pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval
        sem = stats.sem(scores_b - scores_a)
        ci = stats.t.interval(0.95, len(scores_a)-1,
                             loc=mean_diff, scale=sem)

        return {
            'n_samples': len(scores_a),
            'model_a_mean': float(np.mean(scores_a)),
            'model_b_mean': float(np.mean(scores_b)),
            'mean_difference': float(mean_diff),
            'p_value': float(p_value),
            'statistically_significant': bool(p_value < 0.05),
            'cohens_d': float(cohens_d),
            'effect_size': interpret_cohens_d(cohens_d),
            'ci_95_lower': float(ci[0]),
            'ci_95_upper': float(ci[1]),
            'recommendation': self.get_recommendation(p_value, cohens_d)
        }

    def get_recommendation(self, p_value, cohens_d):
        """Provide actionable recommendation"""
        if p_value >= 0.05:
            return "No statistically significant difference detected"
        elif abs(cohens_d) < 0.2:
            return "Statistically significant but negligible practical difference"
        elif cohens_d > 0.5:
            return "Model B shows meaningful improvement - consider deploying"
        elif cohens_d < -0.5:
            return "Model A performs better - stick with current checkpoint"
        else:
            return "Small but significant difference - gather more data"
```

### Statistical Significance Indicators in UI

**Visual feedback for significance:**
```python
def create_significance_indicator(p_value, cohens_d):
    """
    Create visual indicator for statistical results
    """
    with gr.Row():
        # Traffic light indicator
        if p_value >= 0.05:
            color = "🔴"  # Red - no significant difference
            message = "Not statistically significant"
        elif abs(cohens_d) < 0.2:
            color = "🟡"  # Yellow - significant but small effect
            message = "Significant but small effect"
        else:
            color = "🟢"  # Green - significant with meaningful effect
            message = "Significant meaningful difference"

        gr.Markdown(f"## {color} {message}")

        # Detailed stats
        with gr.Accordion("Statistical Details", open=False):
            gr.Markdown(f"""
            - **P-value:** {p_value:.4f}
            - **Cohen's d:** {cohens_d:.3f}
            - **Effect size:** {interpret_cohens_d(cohens_d)}
            - **Interpretation:** {'Reject null hypothesis' if p_value < 0.05 else 'Fail to reject null hypothesis'}
            """)
```

**From** [Towards Data Science - Building a Modern Dashboard with Python and Gradio](https://towardsdatascience.com/building-a-modern-dashboard-with-python-and-gradio) (accessed 2025-10-31):
> "Gradio simplifies the development process by providing an intuitive framework that eliminates the complexities associated with building user interfaces from scratch."

## Section 3: Metrics Collection & Aggregation

### Session-Based Metric Tracking

```python
import json
import pandas as pd
from datetime import datetime

class MetricsCollector:
    """
    Collect and aggregate metrics across sessions
    """
    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = []

    def record_inference(self, model_name, input_data, output_data, metadata):
        """Record a single inference with all relevant metrics"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'model_name': model_name,
            'input_hash': hash(str(input_data)),
            'output': output_data,
            'metrics': {
                'accuracy': metadata.get('accuracy'),
                'confidence': metadata.get('confidence'),
                'latency_ms': metadata.get('latency_ms'),
                'tokens_used': metadata.get('tokens_used'),
            },
            'user_feedback': None  # Filled in later
        }
        self.metrics.append(record)
        return record

    def add_user_feedback(self, record_index, feedback):
        """Add user feedback to existing record"""
        if 0 <= record_index < len(self.metrics):
            self.metrics[record_index]['user_feedback'] = feedback

    def get_aggregated_stats(self, model_name=None):
        """Get aggregated statistics"""
        df = pd.DataFrame(self.metrics)

        if model_name:
            df = df[df['model_name'] == model_name]

        if len(df) == 0:
            return {}

        # Expand metrics dict into columns
        metrics_df = pd.DataFrame(df['metrics'].tolist())

        return {
            'count': len(df),
            'mean_confidence': metrics_df['confidence'].mean(),
            'std_confidence': metrics_df['confidence'].std(),
            'mean_latency_ms': metrics_df['latency_ms'].mean(),
            'median_latency_ms': metrics_df['latency_ms'].median(),
            'p95_latency_ms': metrics_df['latency_ms'].quantile(0.95),
            'total_tokens': metrics_df['tokens_used'].sum(),
            'positive_feedback_rate': self._calculate_feedback_rate(df, positive=True)
        }

    def _calculate_feedback_rate(self, df, positive=True):
        """Calculate positive/negative feedback rate"""
        feedback_records = df[df['user_feedback'].notna()]
        if len(feedback_records) == 0:
            return None

        if positive:
            return (feedback_records['user_feedback'] == 'positive').mean()
        else:
            return (feedback_records['user_feedback'] == 'negative').mean()

    def export_to_csv(self, filename):
        """Export metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
        return filename

    def export_to_json(self, filename):
        """Export metrics to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return filename
```

### Integration with Gradio State

```python
def create_metrics_tracking_ui():
    """
    Gradio UI with persistent metrics tracking
    """
    with gr.Blocks() as demo:
        # Initialize collector in Gradio State
        collector = gr.State(MetricsCollector())
        current_record_idx = gr.State(0)

        gr.Markdown("# VLM Testing with Metrics Tracking")

        with gr.Row():
            image_input = gr.Image(type="pil")
            query_input = gr.Textbox(label="Query")

        run_btn = gr.Button("Run Inference")

        with gr.Row():
            output_text = gr.Textbox(label="Model Output", interactive=False)
            confidence_score = gr.Number(label="Confidence", interactive=False)
            latency_display = gr.Number(label="Latency (ms)", interactive=False)

        # User feedback
        with gr.Row():
            feedback_btn_pos = gr.Button("👍 Good", variant="secondary")
            feedback_btn_neg = gr.Button("👎 Bad", variant="secondary")

        # Statistics display
        with gr.Accordion("Session Statistics", open=True):
            stats_display = gr.JSON(label="Aggregated Metrics")

        # Export buttons
        with gr.Row():
            export_csv_btn = gr.Button("Export CSV")
            export_json_btn = gr.Button("Export JSON")
            download_file = gr.File(label="Download")

        def run_inference(image, query, collector_state):
            """Run inference and record metrics"""
            import time
            start_time = time.time()

            # Simulate model inference
            output = f"Description: {query} (simulated)"
            confidence = 0.85

            latency = (time.time() - start_time) * 1000

            # Record metrics
            record = collector_state.record_inference(
                model_name="VLM-v2",
                input_data={'image': image, 'query': query},
                output_data=output,
                metadata={
                    'confidence': confidence,
                    'latency_ms': latency,
                    'tokens_used': 42
                }
            )

            # Get updated stats
            stats = collector_state.get_aggregated_stats()

            return output, confidence, latency, stats, len(collector_state.metrics) - 1

        def add_feedback(feedback_type, record_idx, collector_state):
            """Add user feedback to record"""
            collector_state.add_user_feedback(record_idx, feedback_type)
            stats = collector_state.get_aggregated_stats()
            return stats

        # Wire up events
        run_btn.click(
            fn=run_inference,
            inputs=[image_input, query_input, collector],
            outputs=[output_text, confidence_score, latency_display,
                    stats_display, current_record_idx]
        )

        feedback_btn_pos.click(
            fn=lambda idx, coll: add_feedback('positive', idx, coll),
            inputs=[current_record_idx, collector],
            outputs=[stats_display]
        )

        feedback_btn_neg.click(
            fn=lambda idx, coll: add_feedback('negative', idx, coll),
            inputs=[current_record_idx, collector],
            outputs=[stats_display]
        )

        export_csv_btn.click(
            fn=lambda coll: coll.export_to_csv(f'metrics_{coll.session_id}.csv'),
            inputs=[collector],
            outputs=[download_file]
        )

        export_json_btn.click(
            fn=lambda coll: coll.export_to_json(f'metrics_{coll.session_id}.json'),
            inputs=[collector],
            outputs=[download_file]
        )

    return demo
```

**From** [Langfuse - Open Source LLM Observability for Gradio](https://langfuse.com/integrations/other/gradio) (accessed 2025-10-31):
> "Each chat message belongs to a thread in the Gradio Chatbot which can be reset using clear. We implement a method that creates a session_id that is used globally."

## Section 4: Automated Validation Workflows

### Batch Evaluation with Statistical Analysis

```python
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    confidence: float
    latency_ms: float

class AutomatedValidator:
    """
    Automated batch evaluation with statistical testing
    """
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.results = {}

    def evaluate_model(self, model, model_name):
        """Run full evaluation on model"""
        accuracies = []
        confidences = []
        latencies = []

        for sample in self.test_dataset:
            import time
            start = time.time()

            prediction = model.predict(sample['image'], sample['query'])
            latency = (time.time() - start) * 1000

            # Calculate accuracy (exact match or similarity)
            accuracy = self._calculate_accuracy(
                prediction['output'],
                sample['ground_truth']
            )

            accuracies.append(accuracy)
            confidences.append(prediction['confidence'])
            latencies.append(latency)

        result = EvaluationResult(
            model_name=model_name,
            accuracy=np.mean(accuracies),
            confidence=np.mean(confidences),
            latency_ms=np.mean(latencies)
        )

        self.results[model_name] = {
            'result': result,
            'accuracies': accuracies,
            'confidences': confidences,
            'latencies': latencies
        }

        return result

    def compare_models(self, baseline_name, candidate_name, alpha=0.05):
        """
        Statistical comparison between two models
        """
        baseline = self.results[baseline_name]
        candidate = self.results[candidate_name]

        # Compare accuracy
        acc_comparison = self._statistical_test(
            baseline['accuracies'],
            candidate['accuracies'],
            metric_name='accuracy',
            alpha=alpha
        )

        # Compare confidence
        conf_comparison = self._statistical_test(
            baseline['confidences'],
            candidate['confidences'],
            metric_name='confidence',
            alpha=alpha
        )

        # Compare latency (lower is better, so flip sign)
        lat_comparison = self._statistical_test(
            baseline['latencies'],
            candidate['latencies'],
            metric_name='latency',
            alpha=alpha,
            lower_is_better=True
        )

        return {
            'baseline': baseline_name,
            'candidate': candidate_name,
            'accuracy': acc_comparison,
            'confidence': conf_comparison,
            'latency': lat_comparison,
            'overall_recommendation': self._make_recommendation(
                acc_comparison, conf_comparison, lat_comparison
            )
        }

    def _statistical_test(self, baseline_scores, candidate_scores,
                         metric_name, alpha=0.05, lower_is_better=False):
        """Perform statistical test with effect size"""
        from scipy import stats

        # T-test
        t_stat, p_value = stats.ttest_ind(candidate_scores, baseline_scores)

        # Effect size (Cohen's d)
        mean_diff = np.mean(candidate_scores) - np.mean(baseline_scores)
        if lower_is_better:
            mean_diff = -mean_diff  # Flip for latency

        pooled_std = np.sqrt((np.var(baseline_scores) +
                             np.var(candidate_scores)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval (bootstrap)
        ci_lower, ci_upper = self._bootstrap_ci(
            candidate_scores, baseline_scores, alpha=alpha
        )

        return {
            'metric': metric_name,
            'baseline_mean': float(np.mean(baseline_scores)),
            'candidate_mean': float(np.mean(candidate_scores)),
            'mean_difference': float(mean_diff),
            'p_value': float(p_value),
            'is_significant': bool(p_value < alpha),
            'cohens_d': float(cohens_d),
            'effect_size': interpret_cohens_d(cohens_d),
            'ci_95': (float(ci_lower), float(ci_upper)),
            'sample_size': len(baseline_scores)
        }

    def _bootstrap_ci(self, group1, group2, alpha=0.05, n_bootstrap=2000):
        """
        Bootstrap confidence interval for mean difference
        """
        differences = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            differences.append(np.mean(sample2) - np.mean(sample1))

        # Calculate percentile CI
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(differences, lower_percentile)
        ci_upper = np.percentile(differences, upper_percentile)

        return ci_lower, ci_upper

    def _make_recommendation(self, acc_comp, conf_comp, lat_comp):
        """Make deployment recommendation based on all metrics"""
        # Accuracy is most important
        if not acc_comp['is_significant']:
            return {
                'decision': 'KEEP_BASELINE',
                'reason': 'No significant accuracy improvement'
            }

        if acc_comp['cohens_d'] < 0.2:
            return {
                'decision': 'KEEP_BASELINE',
                'reason': 'Accuracy improvement too small to justify change'
            }

        # Check if latency degraded significantly
        if lat_comp['is_significant'] and lat_comp['cohens_d'] < -0.5:
            return {
                'decision': 'CAUTION',
                'reason': 'Accuracy improved but latency degraded significantly'
            }

        return {
            'decision': 'DEPLOY_CANDIDATE',
            'reason': f"Significant accuracy improvement (d={acc_comp['cohens_d']:.2f})"
        }
```

### Gradio UI for Automated Evaluation

```python
def create_automated_evaluation_ui():
    """
    UI for running automated evaluations
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Automated VLM Checkpoint Evaluation")

        with gr.Row():
            baseline_path = gr.Textbox(label="Baseline Checkpoint Path")
            candidate_path = gr.Textbox(label="Candidate Checkpoint Path")
            dataset_path = gr.Textbox(label="Test Dataset Path")

        run_eval_btn = gr.Button("Run Evaluation", variant="primary")

        # Progress indicator
        progress = gr.Textbox(label="Progress", interactive=False)

        # Results display
        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_json = gr.JSON(label="Evaluation Summary")
                recommendation = gr.Markdown()

            with gr.Tab("Detailed Statistics"):
                detailed_stats = gr.DataFrame(label="Per-Metric Comparison")

            with gr.Tab("Visualizations"):
                comparison_plot = gr.Plot(label="Distribution Comparison")

        def run_evaluation(baseline_path, candidate_path, dataset_path):
            """Run full automated evaluation"""
            # Load test dataset
            yield "Loading test dataset...", {}, "", None, None
            test_data = load_test_dataset(dataset_path)

            # Load models
            yield "Loading baseline model...", {}, "", None, None
            baseline_model = load_model(baseline_path)

            yield "Loading candidate model...", {}, "", None, None
            candidate_model = load_model(candidate_path)

            # Run evaluation
            validator = AutomatedValidator(test_data)

            yield "Evaluating baseline model...", {}, "", None, None
            baseline_result = validator.evaluate_model(baseline_model, "baseline")

            yield "Evaluating candidate model...", {}, "", None, None
            candidate_result = validator.evaluate_model(candidate_model, "candidate")

            # Statistical comparison
            yield "Running statistical tests...", {}, "", None, None
            comparison = validator.compare_models("baseline", "candidate")

            # Generate summary
            summary = {
                'baseline_accuracy': baseline_result.accuracy,
                'candidate_accuracy': candidate_result.accuracy,
                'comparison': comparison
            }

            # Create recommendation markdown
            rec = comparison['overall_recommendation']
            rec_md = f"""
            ## Recommendation: {rec['decision']}

            **Reason:** {rec['reason']}

            ### Key Findings:
            - Accuracy difference: {comparison['accuracy']['mean_difference']:.3f}
              (p={comparison['accuracy']['p_value']:.4f}, d={comparison['accuracy']['cohens_d']:.2f})
            - Effect size: {comparison['accuracy']['effect_size']}
            """

            # Create detailed stats table
            stats_df = pd.DataFrame([
                comparison['accuracy'],
                comparison['confidence'],
                comparison['latency']
            ])

            # Create plot
            plot = create_comparison_plot(validator.results)

            yield "Evaluation complete!", summary, rec_md, stats_df, plot

        run_eval_btn.click(
            fn=run_evaluation,
            inputs=[baseline_path, candidate_path, dataset_path],
            outputs=[progress, summary_json, recommendation,
                    detailed_stats, comparison_plot]
        )

    return demo

def create_comparison_plot(results):
    """Create visualization comparing model distributions"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric in zip(axes, ['accuracies', 'confidences', 'latencies']):
        baseline_data = results['baseline'][metric]
        candidate_data = results['candidate'][metric]

        ax.hist(baseline_data, alpha=0.5, label='Baseline', bins=20)
        ax.hist(candidate_data, alpha=0.5, label='Candidate', bins=20)
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title(f'{metric.capitalize()} Distribution')

    plt.tight_layout()
    return fig
```

**From** [Nature Scientific Reports - Evaluation metrics and statistical tests for machine learning](https://www.nature.com/articles/s41598-024-56706-x) (accessed 2025-10-31):
> "Our aim here is to introduce the most common metrics for binary and multi-class classification, regression, image segmentation, and object detection."

## Sources

**Web Research:**

1. **[Medium - Statistical Significance in A/B Testing: Beyond P-Hacking in 2025](https://medium.com/@nextechie/statistical-significance-in-a-b-testing-beyond-p-hacking-in-2025-26170077f587)** (accessed 2025-10-31)
   - Confidence intervals and precision measures
   - Common pitfalls in 2025 A/B testing

2. **[Towards Data Science - Building a Modern Dashboard with Python and Gradio](https://towardsdatascience.com/building-a-modern-dashboard-with-python-and-gradio/)** (accessed 2025-10-31)
   - Complete Gradio dashboard implementation
   - Metrics tracking and visualization patterns
   - Session-based data collection

3. **[Analytics Vidhya - Building an Interactive Data Dashboard Creation With Gradio](https://www.analyticsvidhya.com/blog/2025/07/interactive-data-dashboard/)** (accessed 2025-10-31)
   - Interactive dashboard patterns
   - Data aggregation workflows

4. **[Langfuse - Open Source LLM Observability for Gradio](https://langfuse.com/integrations/other/gradio)** (accessed 2025-10-31)
   - Session tracking implementation
   - Metrics collection patterns
   - User feedback integration

5. **[Frontiers in Psychology - Calculating and reporting effect sizes](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00863/full)** (accessed 2025-10-31)
   - Cohen's d calculation and interpretation
   - Bootstrap confidence intervals for effect sizes

6. **[ResearchGate - Statistical Significance Testing in ML Model Comparisons](https://www.researchgate.net/publication/392727623_Statistical_Significance_Testing_in_ML_Model_Comparisons_Beyond_p-values_and_t-tests)** (accessed 2025-10-31)
   - Beyond p-values and t-tests for ML evaluation
   - Robust statistical approaches

7. **[Nature Scientific Reports - Evaluation metrics and statistical tests for machine learning](https://www.nature.com/articles/s41598-024-56706-x)** (accessed 2025-10-31)
   - Common metrics for classification and regression
   - Statistical testing best practices

8. **[YouTube - Comparing SLMs and LLMs with similarity metrics (Julien Simon)](https://www.youtube.com/watch?v=79iuy2uKLpc)** (accessed 2025-10-31)
   - Model A/B testing with Gradio and Arcee Conductor
   - Similarity metrics implementation

**Additional References:**

9. **[Statsig - How confidence intervals empower better decisions](https://www.statsig.com/perspectives/confidence-intervals-better-decisions)** (accessed 2025-10-31)
   - Confidence intervals in ML model performance

10. **[Analytics Toolkit - A/B Testing Statistics: A Concise Guide](https://blog.analytics-toolkit.com/2022/a-b-testing-statistics-a-concise-guide/)** (accessed 2025-10-31)
    - Statistical foundations for A/B testing
    - Common statistical pitfalls
