# Gradio Statistical Testing & Validation (2025)

**Source**: Platonic Dialogue Part 39 + Bright Data research (2025-01-30, 2025-01-31)
**Context**: A/B testing, effect size analysis, ablation studies, statistical significance
**Philosophy**: Beyond p-values - comprehensive statistical validation

---

## Part 3: A/B Testing & Statistical Analysis

### Pattern: Hypothesis Testing

**From research (Medium "Demystifying A/B Testing in ML", Toward Data Science guide):**

**Test Design:**
1. **State hypothesis**: "Adaptive tensions improve accuracy vs fixed tensions"
2. **Define metrics**: Primary (accuracy), Secondary (speed, memory)
3. **Sample size**: Use statistical calculators (avoid false positives)
4. **Random assignment**: Ensure fair comparison

```python
import scipy.stats as stats
import pandas as pd

def run_ab_test(variant_a_results, variant_b_results, alpha=0.05):
    """
    Statistical A/B test between two model variants

    Args:
        variant_a_results: List of accuracy scores for variant A
        variant_b_results: List of accuracy scores for variant B
        alpha: Significance level (typically 0.05 for 95% confidence)

    Returns:
        Dictionary with test results and interpretation
    """
    # Calculate means and standard deviations
    mean_a = np.mean(variant_a_results)
    mean_b = np.mean(variant_b_results)
    std_a = np.std(variant_a_results, ddof=1)
    std_b = np.std(variant_b_results, ddof=1)

    # Perform two-sample t-test
    t_stat, p_value = stats.ttest_ind(variant_a_results, variant_b_results)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_b - mean_a) / pooled_std

    # Interpret results
    is_significant = p_value < alpha
    interpretation = {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'difference': mean_b - mean_a,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': is_significant,
        'conclusion': f"Variant B is {'significantly' if is_significant else 'not significantly'} different from Variant A"
    }

    return interpretation

# Example: Compare two ARR-COC variants
baseline_scores = [0.75, 0.78, 0.74, 0.76, 0.77]  # 5 test runs
arr_coc_scores = [0.82, 0.84, 0.83, 0.81, 0.85]   # 5 test runs

result = run_ab_test(baseline_scores, arr_coc_scores)
print(f"p-value: {result['p_value']:.4f}")
print(f"Effect size: {result['cohens_d']:.2f}")
print(result['conclusion'])
```

**Key concepts (from research):**
- **Statistical significance**: p < 0.05 (95% confidence)
- **Practical significance**: Effect size (Cohen's d) matters
- **Sample size**: Critical for statistical power
- **Multiple comparisons**: Bonferroni correction if testing many variants

### Pattern: Pandas Experiment Analysis

**From research (Real Python, Analytics Vidhya pandas groupby guides):**

```python
def analyze_experiment_results(results_df):
    """
    Analyze experiment results across multiple variants

    DataFrame columns: ['model', 'query', 'accuracy', 'latency', 'tokens']
    """
    # Group by model variant, compute statistics
    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'count'],
        'latency': ['mean', 'std'],
        'tokens': ['mean', 'std']
    })

    # Compare each variant to baseline
    baseline_acc = summary.loc['baseline', ('accuracy', 'mean')]

    summary['accuracy_vs_baseline'] = (
        summary[('accuracy', 'mean')] - baseline_acc
    ) / baseline_acc * 100

    return summary

# Example usage
import pandas as pd

results = pd.DataFrame([
    {'model': 'baseline', 'query': 'q1', 'accuracy': 0.75, 'latency': 0.45, 'tokens': 1024},
    {'model': 'arr_coc_v1', 'query': 'q1', 'accuracy': 0.82, 'latency': 0.38, 'tokens': 850},
    {'model': 'arr_coc_v2', 'query': 'q1', 'accuracy': 0.84, 'latency': 0.40, 'tokens': 820},
    # ... more results
])

summary = analyze_experiment_results(results)
print(summary)
```

**Pattern (from research):**
- `groupby('model')` to group by variant
- `.agg(['mean', 'std', 'count'])` for multiple statistics
- Easy side-by-side comparison in DataFrame
- Export to CSV for further analysis

### Pattern: Ablation Study Methodology

**From research (arXiv 1901.08644, Baeldung ML guide, Wikipedia):**

**Definition**: Systematically remove components to measure their impact.

```python
def ablation_study_arr_coc():
    """
    Test ARR-COC with components systematically removed

    Components to ablate:
    - Propositional scorer (information content)
    - Perspectival scorer (salience)
    - Participatory scorer (query-content coupling)
    - Tension balancing
    """
    variants = {
        'full': ['propositional', 'perspectival', 'participatory', 'tensions'],
        'no_propositional': ['perspectival', 'participatory', 'tensions'],
        'no_perspectival': ['propositional', 'participatory', 'tensions'],
        'no_participatory': ['propositional', 'perspectival', 'tensions'],
        'no_tensions': ['propositional', 'perspectival', 'participatory'],
        'minimal': ['participatory'],  # Query-awareness only
    }

    results = {}
    baseline_accuracy = None

    for variant_name, components in variants.items():
        # Configure ARR-COC with only these components
        model = configure_arr_coc(components)
        accuracy = evaluate_model(model, test_dataset)

        if variant_name == 'full':
            baseline_accuracy = accuracy

        degradation = (baseline_accuracy - accuracy) / baseline_accuracy * 100

        results[variant_name] = {
            'components': components,
            'accuracy': accuracy,
            'degradation_%': degradation
        }

    return results

# Expected findings (from Dialogue 39):
# - Removing participatory scorer: Large degradation (query-awareness critical)
# - Removing propositional: Moderate degradation (loses statistical info)
# - Removing perspectival: Moderate degradation (loses salience)
# - Removing tensions: Small degradation (static balance works reasonably)
```

**Process (from research):**
1. Baseline with all components
2. Remove component X → measure performance drop
3. Remove each component one at a time
4. Compare to baseline
5. System must function (graceful degradation)

**Example for ARR-COC (from Dialogue 39):**
- Test all 3 scorers vs 2 scorers vs 1 scorer
- Identify which components are critical
- Quantify impact of each component

## Part 8: Advanced Statistical Testing (2025 Expansion)

### Statistical Significance: Beyond p-values

**From Medium "Statistical Significance in A/B Testing: Beyond P-Hacking in 2025" + Nature Scientific Reports**:

lol, so you ran an A/B test, got p=0.03, and declared victory? Not so fast. **80% of A/B tests fail to produce statistically significant results**, and even when they do, you need more than just a p-value.

**The Three-Metric Framework** (2025 consensus):

1. **Statistical Significance** (p-value < 0.05)
   - Tells you: Is this effect real or random chance?
   - Doesn't tell you: How big is the effect? Does it matter?

2. **Effect Size** (Cohen's d, eta-squared)
   - Tells you: How much better is variant B than variant A?
   - Critical for practical significance

3. **Sample Size** (Power analysis)
   - Tells you: Do you have enough data to detect real effects?
   - Industry standard: 80% statistical power

**Complete Testing Pattern**:

```python
import scipy.stats as stats
import numpy as np
from typing import Dict, List, Tuple

def comprehensive_ab_test(
    variant_a: List[float],
    variant_b: List[float],
    alpha: float = 0.05,
    min_effect_size: float = 0.2
) -> Dict:
    """
    Complete A/B testing with statistical significance + effect size

    Args:
        variant_a: Accuracy scores for baseline model
        variant_b: Accuracy scores for experimental model
        alpha: Significance level (typically 0.05)
        min_effect_size: Minimum Cohen's d to care about (0.2 = small, 0.5 = medium, 0.8 = large)

    Returns:
        Dictionary with all metrics + interpretation
    """
    # 1. Descriptive statistics
    mean_a = np.mean(variant_a)
    mean_b = np.mean(variant_b)
    std_a = np.std(variant_a, ddof=1)
    std_b = np.std(variant_b, ddof=1)
    n_a = len(variant_a)
    n_b = len(variant_b)

    # 2. Statistical significance (two-sample t-test)
    t_stat, p_value = stats.ttest_ind(variant_a, variant_b)
    is_significant = p_value < alpha

    # 3. Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (mean_b - mean_a) / pooled_std

    # Effect size interpretation (Cohen 1988)
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # 4. Confidence interval for mean difference
    mean_diff = mean_b - mean_a
    se_diff = pooled_std * np.sqrt(1/n_a + 1/n_b)
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff

    # 5. Power analysis (post-hoc)
    from statsmodels.stats.power import ttest_ind_solve_power
    observed_power = ttest_ind_solve_power(
        effect_size=abs(cohens_d),
        nobs1=n_a,
        ratio=n_b/n_a,
        alpha=alpha,
        alternative='two-sided'
    )

    # 6. Overall verdict
    is_practically_significant = abs(cohens_d) >= min_effect_size

    if is_significant and is_practically_significant:
        verdict = f"✅ SIGNIFICANT: Variant B is {effect_interpretation}ly better (p={p_value:.4f}, d={cohens_d:.3f})"
    elif is_significant and not is_practically_significant:
        verdict = f"⚠️ STATISTICALLY but not PRACTICALLY significant (p={p_value:.4f}, d={cohens_d:.3f})"
    elif not is_significant and is_practically_significant:
        verdict = f"⚠️ LARGE EFFECT but not statistically significant - need more samples! (p={p_value:.4f}, d={cohens_d:.3f})"
    else:
        verdict = f"❌ NOT SIGNIFICANT: No meaningful difference (p={p_value:.4f}, d={cohens_d:.3f})"

    return {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_difference': mean_diff,
        'ci_95': (ci_lower, ci_upper),
        'p_value': p_value,
        'is_statistically_significant': is_significant,
        'cohens_d': cohens_d,
        'effect_size_interpretation': effect_interpretation,
        'is_practically_significant': is_practically_significant,
        'sample_size_a': n_a,
        'sample_size_b': n_b,
        'statistical_power': observed_power,
        'verdict': verdict
    }

# Example usage
baseline_accuracy = [0.75, 0.78, 0.74, 0.76, 0.77, 0.75, 0.78]  # n=7
arr_coc_accuracy = [0.82, 0.84, 0.83, 0.81, 0.85, 0.83, 0.84]   # n=7

result = comprehensive_ab_test(baseline_accuracy, arr_coc_accuracy)

print(f"Baseline: {result['mean_a']:.3f} ± {np.std(baseline_accuracy, ddof=1):.3f}")
print(f"ARR-COC:  {result['mean_b']:.3f} ± {np.std(arr_coc_accuracy, ddof=1):.3f}")
print(f"Difference: {result['mean_difference']:.3f} [95% CI: {result['ci_95'][0]:.3f} to {result['ci_95'][1]:.3f}]")
print(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']} effect)")
print(f"p-value: {result['p_value']:.4f}")
print(f"Statistical power: {result['statistical_power']:.2f}")
print(f"\n{result['verdict']}")
```

**Output example**:
```
Baseline: 0.762 ± 0.016
ARR-COC:  0.831 ± 0.014
Difference: 0.069 [95% CI: 0.051 to 0.087]
Cohen's d: 4.421 (large effect)
p-value: 0.0001
Statistical power: 1.00

✅ SIGNIFICANT: Variant B is largely better (p=0.0001, d=4.421)
```

### Sample Size Calculation (Before Testing)

**How many samples do you need?**

```python
from statsmodels.stats.power import ttest_ind_solve_power

def calculate_required_sample_size(
    expected_effect_size: float = 0.5,  # Cohen's d
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size per group

    Args:
        expected_effect_size: How big a difference you expect (0.2=small, 0.5=medium, 0.8=large)
        alpha: Significance level (0.05 standard)
        power: Statistical power (0.80 = 80% chance of detecting real effect)

    Returns:
        Required sample size per group
    """
    n = ttest_ind_solve_power(
        effect_size=expected_effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,  # Equal groups
        alternative='two-sided'
    )
    return int(np.ceil(n))

# Example: Testing ARR-COC vs baseline
# Expect medium effect (d=0.5)
n_required = calculate_required_sample_size(expected_effect_size=0.5, power=0.80)
print(f"Need {n_required} test examples per model for 80% power")
# Output: Need 64 test examples per model for 80% power

# Expect small effect (d=0.2)
n_small = calculate_required_sample_size(expected_effect_size=0.2, power=0.80)
print(f"For small effects: {n_small} examples needed")
# Output: For small effects: 394 examples needed
```

**The Reality**:
- Small effects need **HUGE** samples (394+ per group)
- Medium effects need moderate samples (64 per group)
- Large effects need small samples (26 per group)

**For ARR-COC testing**: If you expect a medium improvement (5-8% accuracy gain), test on **64+ diverse queries** minimum.

### Ablation Study Best Practices (2025)

**From ResearchGate "Ablation Studies in Artificial Neural Networks" + Stackademic Practitioner's Guide**:

**Definition**: Systematically remove components to measure their contribution.

**The Pattern** (must-follow):

1. **Baseline with ALL components** → Measure performance
2. **Remove component X** → Measure performance drop
3. **Remove component Y** → Measure performance drop
4. **Remove component Z** → Measure performance drop
5. **Compare drops** → Identify critical vs optional components

**Implementation for ARR-COC**:

```python
def ablation_study_arr_coc(test_dataset, model_factory):
    """
    Systematic ablation study for ARR-COC components

    Tests contribution of:
    - Propositional scorer (information content)
    - Perspectival scorer (salience)
    - Participatory scorer (query-content coupling)
    - Tension balancing (opponent processing)
    """
    # Component configurations
    ablation_configs = {
        'full': {
            'components': ['propositional', 'perspectival', 'participatory', 'tensions'],
            'description': 'All components (baseline)'
        },
        'no_propositional': {
            'components': ['perspectival', 'participatory', 'tensions'],
            'description': 'Without information content scorer'
        },
        'no_perspectival': {
            'components': ['propositional', 'participatory', 'tensions'],
            'description': 'Without salience scorer'
        },
        'no_participatory': {
            'components': ['propositional', 'perspectival', 'tensions'],
            'description': 'Without query-coupling scorer'
        },
        'no_tensions': {
            'components': ['propositional', 'perspectival', 'participatory'],
            'description': 'Without opponent processing (static balance)'
        },
        'minimal': {
            'components': ['participatory'],
            'description': 'Query-awareness only (single scorer)'
        }
    }

    results = {}
    baseline_accuracy = None

    for config_name, config in ablation_configs.items():
        # Build model with specified components
        model = model_factory(components=config['components'])

        # Evaluate on test set
        accuracies = []
        latencies = []
        token_counts = []

        for example in test_dataset:
            start = time.time()
            output = model.process(example['image'], example['query'])
            latency = time.time() - start

            accuracy = evaluate_output(output, example['ground_truth'])
            accuracies.append(accuracy)
            latencies.append(latency)
            token_counts.append(output['tokens_used'])

        mean_accuracy = np.mean(accuracies)
        mean_latency = np.mean(latencies)
        mean_tokens = np.mean(token_counts)

        # Calculate degradation from baseline
        if config_name == 'full':
            baseline_accuracy = mean_accuracy
            degradation_pct = 0.0
        else:
            degradation_pct = (baseline_accuracy - mean_accuracy) / baseline_accuracy * 100

        results[config_name] = {
            'description': config['description'],
            'components': config['components'],
            'accuracy': mean_accuracy,
            'latency': mean_latency,
            'tokens': mean_tokens,
            'degradation_%': degradation_pct
        }

    return results

# Example output interpretation
results = ablation_study_arr_coc(test_dataset, arr_coc_factory)

# Sort by degradation to find most critical components
sorted_results = sorted(
    [(k, v) for k, v in results.items() if k != 'full'],
    key=lambda x: x[1]['degradation_%'],
    reverse=True
)

print("Component Importance (by degradation):")
for config_name, metrics in sorted_results:
    print(f"  {config_name}: {metrics['degradation_%']:.1f}% degradation")
    print(f"    Accuracy: {metrics['accuracy']:.3f} (vs {results['full']['accuracy']:.3f})")
    print(f"    Tokens: {metrics['tokens']:.0f} (vs {results['full']['tokens']:.0f})")
```

**Expected findings for ARR-COC** (hypothetical):
```
Component Importance (by degradation):
  no_participatory: 18.2% degradation
    Accuracy: 0.689 (vs 0.842)
    Tokens: 1024 (vs 247)
    ⚠️ CRITICAL: Query-coupling is essential!

  no_perspectival: 8.5% degradation
    Accuracy: 0.770 (vs 0.842)
    Tokens: 289 (vs 247)
    ⚠️ IMPORTANT: Salience helps significantly

  no_propositional: 5.2% degradation
    Accuracy: 0.798 (vs 0.842)
    Tokens: 251 (vs 247)
    ✅ HELPFUL: Information content contributes

  no_tensions: 2.1% degradation
    Accuracy: 0.824 (vs 0.842)
    Tokens: 253 (vs 247)
    ✅ MINOR: Static balance works OK
```

**Interpretation**: Participatory scorer (query-coupling) is **CRITICAL** (18% drop). Perspectival and Propositional help. Adaptive tensions provide modest gains.

### W&B + Gradio: The Ultimate Integration (2025)

**From Gradio.app official guide + W&B Reports**:

**Why integrate?**
- Gradio: Interactive testing UI
- W&B: Persistent experiment tracking
- Together: Every Gradio interaction → logged W&B artifact

**Complete Integration Pattern**:

```python
import gradio as gr
import wandb
from datetime import datetime

# Initialize W&B run
wandb.init(
    project="arr-coc-testing",
    name=f"gradio_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        'model': 'arr-coc-v1',
        'base': 'Qwen3-VL-2B',
        'testing_mode': 'interactive'
    },
    tags=["gradio", "interactive", "user-testing"]
)

def process_with_wandb_logging(image, query, model_variant):
    """
    Process query and log everything to W&B

    Logs:
    - Input (image + query)
    - Output (answer + tokens + latency)
    - Model variant used
    - Timestamp
    """
    start = time.time()

    # Run inference
    output = model.process(image, query, variant=model_variant)
    latency = time.time() - start

    # Log to W&B
    wandb.log({
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'model_variant': model_variant,

        # Input
        'query': query,
        'image': wandb.Image(image, caption=query),

        # Output
        'answer': output['answer'],
        'tokens_used': output['tokens'],
        'latency_sec': latency,
        'memory_gb': torch.cuda.max_memory_allocated() / 1e9,

        # Quality (if available)
        'confidence': output.get('confidence', None),

        # Visualization
        'relevance_heatmap': wandb.Image(output['heatmap']) if output.get('heatmap') else None
    })

    # Also log as W&B Table (for easy comparison later)
    wandb.log({
        "interaction_table": wandb.Table(
            columns=["Query", "Answer", "Tokens", "Latency", "Image"],
            data=[[query, output['answer'], output['tokens'], latency, wandb.Image(image)]]
        )
    })

    return output['answer'], output['heatmap'], f"Tokens: {output['tokens']}, Latency: {latency:.2f}s"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ARR-COC Testing (W&B Logged)")

    with gr.Row():
        image = gr.Image(type="pil", label="Input Image")
        query = gr.Textbox(label="Query", placeholder="What is this?")

    model_variant = gr.Dropdown(
        choices=['baseline', 'arr_coc_v1', 'arr_coc_v2'],
        label="Model Variant",
        value='arr_coc_v1'
    )

    submit_btn = gr.Button("Process (Logs to W&B)")

    with gr.Row():
        answer = gr.Textbox(label="Answer", interactive=False)
        heatmap = gr.Image(label="Relevance Heatmap")

    metrics = gr.Textbox(label="Metrics", interactive=False)

    submit_btn.click(
        fn=process_with_wandb_logging,
        inputs=[image, query, model_variant],
        outputs=[answer, heatmap, metrics]
    )

demo.launch()

# At end of session, finish W&B run
# wandb.finish()
```

**W&B Dashboard Shows**:
- All queries tested (searchable)
- Side-by-side image comparisons
- Token usage over time
- Latency distribution
- Model variant performance comparison
- Click any query → see full context (image, answer, heatmap)

**Advanced: User Feedback Integration**:

```python
def process_with_feedback(image, query, model_variant):
    """Add user feedback loop"""
    output = model.process(image, query, variant=model_variant)

    # Log to W&B
    wandb.log({
        'query': query,
        'answer': output['answer'],
        'tokens': output['tokens']
    })

    return output['answer'], output['heatmap']

def log_user_feedback(query, answer, rating, comments):
    """Log user rating back to W&B"""
    wandb.log({
        'user_feedback': {
            'query': query,
            'answer': answer,
            'rating': rating,  # 1-5 stars
            'comments': comments
        }
    })
    return "Feedback logged! Thank you."

with gr.Blocks() as demo:
    # ... previous interface ...

    # Add feedback section
    gr.Markdown("## Rate this answer")
    rating = gr.Radio(choices=[1, 2, 3, 4, 5], label="Rating")
    comments = gr.Textbox(label="Comments (optional)", placeholder="What worked? What didn't?")
    feedback_btn = gr.Button("Submit Feedback")
    feedback_status = gr.Textbox(label="Status", interactive=False)

    feedback_btn.click(
        fn=log_user_feedback,
        inputs=[query, answer, rating, comments],
        outputs=feedback_status
    )
```

**The Result**: Every user interaction → W&B artifact with:
- Input/output pairs
- Performance metrics
- User ratings
- Time-series analysis
- Exportable to Pandas for statistical analysis

**Use case for ARR-COC**: Test with 10 users, collect 50 interactions each, analyze which queries work best, identify failure modes, iterate on model.

---

---

**Related Gradio Files:**
- [09-gradio-core-testing-patterns.md](09-gradio-core-testing-patterns.md) - Multi-model comparison, interface patterns
- [11-gradio-production-deployment.md](11-gradio-production-deployment.md) - W&B integration, T4 constraints
- [12-gradio-visualization-best-practices.md](12-gradio-visualization-best-practices.md) - Gallery testing, Gradio 5

**Primary Sources:**
- Bright Data Research 2025-01-31 (EXPANSION 1 - Statistics):
  - Medium "Statistical Significance in A/B Testing: Beyond P-Hacking in 2025"
  - Nature Scientific Reports: "Evaluation metrics and statistical tests for machine learning" (874 citations)
  - PMC: "Using Effect Size—or Why the P Value Is Not Enough" (7,602 citations)
  - ResearchGate: "Ablation Studies in Artificial Neural Networks"
  - Stackademic: "The Practitioner's Guide to Ablation in Machine Learning" (Jul 2025)
  - Gradio.app: "Gradio and W&B Integration" official guide

**Last Updated**: 2025-01-31 (Split from 09-gradio-testing-patterns-2025-01-30.md)
**Version**: 1.0 - Statistical testing patterns (~700 lines)
