# Statistical Hypothesis Testing

## Overview

Statistical hypothesis testing is the foundational framework for making evidence-based decisions under uncertainty in cognitive science, machine learning, and experimental research. It provides a rigorous method for determining whether observed data provide sufficient evidence to reject a null hypothesis in favor of an alternative hypothesis, controlling for random variation and quantifying the reliability of conclusions.

## Core Concepts

### The Null Hypothesis (H₀)

The null hypothesis represents the default position or status quo - typically that no effect exists, no difference is present, or that any observed pattern is due to chance alone. It is the hypothesis that the test seeks to provide evidence against.

**Key Properties:**
- Always includes an equality (=, ≤, or ≥)
- Represents the skeptical position
- Cannot be "proven" true, only rejected or failed to reject
- Example: H₀: μ₁ = μ₂ (two group means are equal)

### The Alternative Hypothesis (H₁ or Hₐ)

The alternative hypothesis represents the research claim or the effect being investigated. It contradicts the null hypothesis and is what researchers hope to find evidence for.

**Types:**
- **Two-tailed**: H₁: μ₁ ≠ μ₂ (difference in either direction)
- **One-tailed (greater)**: H₁: μ₁ > μ₂ (directional increase)
- **One-tailed (less)**: H₁: μ₁ < μ₂ (directional decrease)

### P-Values: The Evidence Metric

From [NIH Statistical Testing Review](https://www.ncbi.nlm.nih.gov/books/NBK557530/) (accessed 2025-11-16):

The p-value is the probability of obtaining test results at least as extreme as the observed results, **assuming the null hypothesis is true**. It quantifies how surprising the data are under the null hypothesis.

**Critical Interpretations:**
- **p < 0.05**: Conventionally considered statistically significant
- **p-value is NOT**: The probability that the null hypothesis is true
- **p-value is NOT**: The probability that results are due to chance alone
- **p-value IS**: The probability of the data (or more extreme) given H₀ is true

**Common Misconception:**
A p-value of 0.03 does NOT mean there is a 97% chance the effect is real. It means: "If there were truly no effect, we would observe data this extreme or more extreme only 3% of the time."

## Type I and Type II Errors

### Type I Error (False Positive)

**Definition**: Rejecting the null hypothesis when it is actually true.

From [Statistics By Jim - Bonferroni Correction](https://statisticsbyjim.com/hypothesis-testing/bonferroni-correction/) (accessed 2025-11-16):

The probability of a Type I error equals the significance level (α). For a single test at α = 0.05, there is a 5% chance of a false positive.

**Real-World Example:**
- Cognitive Science: Concluding a cognitive intervention improves memory when it actually has no effect
- Machine Learning: Declaring a model improvement significant when performance gains are random noise
- ARR-COC-0-1: Falsely concluding relevance allocation improves task performance when gains are spurious

**Control Mechanism**: Set significance threshold α (typically 0.05 or 0.01)

### Type II Error (False Negative)

**Definition**: Failing to reject the null hypothesis when it is actually false.

**Probability**: Denoted β (beta)
**Complement**: Statistical power = 1 - β

**Real-World Example:**
- Missing a true cognitive impairment in patient assessment
- Failing to detect genuine model performance improvement
- ARR-COC-0-1: Missing that relevance realization actually improves vision-language understanding

**Control Mechanism**: Increase sample size, increase effect size, or relax α

### The Tradeoff

From [NIH Type I and II Errors Review](https://www.ncbi.nlm.nih.gov/books/NBK557530/) (accessed 2025-11-16):

Reducing Type I errors (making α more stringent) inevitably increases Type II errors (reduces power). The optimal balance depends on the relative costs of false positives versus false negatives in your domain.

**Decision Matrix:**

```
                    Reality
                H₀ True        H₀ False
Decision  ┌─────────────────────────────────┐
Reject H₀ │ Type I Error   Correct Decision │
          │  (α)            (Power = 1-β)    │
          ├─────────────────────────────────┤
Fail to   │ Correct        Type II Error    │
Reject H₀ │  (1-α)          (β)             │
          └─────────────────────────────────┘
```

## Statistical Power Analysis

### What is Statistical Power?

**Definition**: The probability of correctly rejecting a false null hypothesis (detecting a true effect when it exists).

**Formula**: Power = 1 - β

From [Analytics Toolkit - Observed Power Guide](https://blog.analytics-toolkit.com/2024/comprehensive-guide-to-observed-power-post-hoc-power/) (accessed 2025-11-16):

> "The statistical power of a statistical test is defined as the probability of observing a p-value statistically significant at a certain threshold α if a true effect of a certain magnitude μ₁ is in fact present."

**Typical Target**: 80% or 90% power

**What 80% Power Means:**
If a true effect exists, you have an 80% chance of detecting it (rejecting H₀) in your experiment.

### Factors Affecting Power

1. **Sample Size (n)**: Larger n → Higher power
2. **Effect Size**: Larger true effect → Higher power
3. **Significance Level (α)**: Larger α → Higher power (but more Type I errors)
4. **Variance**: Lower variance → Higher power

### Pre-Test vs Post-Hoc Power

**Pre-Test Power Analysis** (Appropriate):
- Calculate required sample size before data collection
- Based on expected/minimum detectable effect size
- Guides experimental design

**Post-Hoc Power** (Problematic):
From [Analytics Toolkit Guide](https://blog.analytics-toolkit.com/2024/comprehensive-guide-to-observed-power-post-hoc-power/) (accessed 2025-11-16):

> "Observed power is almost always heavily biased vis-à-vis true power. It does not hone in on the true value with larger amounts of information; is not consistent, and its variance is far too large to be useful."

**Why Observed Power Fails:**
- Direct transformation of the p-value (adds no new information)
- Non-significant results always appear "underpowered"
- Biased estimate of true power
- Confuses pre-data probability with post-data evidence

## Multiple Comparisons Problem

### The Escalating Error Rate

From [Statistics By Jim](https://statisticsbyjim.com/hypothesis-testing/bonferroni-correction/) (accessed 2025-11-16):

When conducting multiple hypothesis tests, the family-wise error rate (FWER) - probability of making at least one Type I error - increases dramatically:

**Family-Wise Error Rate Formula**: FWER = 1 - (1 - α)^C

Where α is the significance level for a single test and C is the number of comparisons.

**Example** (α = 0.05):
```
Number of Tests    FWER
──────────────────────────
1                  5.0%
5                  22.6%
10                 40.1%
15                 53.7%
20                 64.2%
50                 92.3%
100                99.4%
```

**Consequence**: With 15 hypothesis tests at α = 0.05, there is a 54% chance of at least one false positive!

### Bonferroni Correction

**Purpose**: Control family-wise error rate to equal the original α level across all tests.

**Method**: Divide significance level by number of comparisons.

**Formula**: α_adjusted = α / n

**Example**:
- Original α = 0.05
- 5 hypothesis tests
- Adjusted α = 0.05 / 5 = 0.01
- Declare significance only if p ≤ 0.01

**Advantages:**
- Simple to calculate and apply
- Provides strong control of FWER
- Conservative approach (low false positive rate)

**Disadvantages:**
- Reduces statistical power (increases Type II errors)
- Too conservative with many tests or correlated tests
- May miss genuine effects

### Alternative Corrections

**Holm-Bonferroni Method:**
- Less conservative than Bonferroni
- Orders p-values and applies sequential testing
- Better power while maintaining FWER control

**Benjamini-Hochberg Procedure:**
- Controls False Discovery Rate (FDR) instead of FWER
- More powerful than Bonferroni
- Appropriate when some false positives are acceptable

**False Discovery Rate (FDR):**
- Expected proportion of false positives among rejected hypotheses
- More lenient than FWER control
- Common in genomics, neuroimaging, machine learning

## Hypothesis Testing in Machine Learning

### Model Validation Challenges

From [Nature Scientific Reports - ML Evaluation Metrics](https://www.nature.com/articles/s41598-024-56706-x) (accessed 2025-11-16):

Machine learning introduces unique hypothesis testing challenges:
- High-dimensional feature spaces
- Multiple models compared simultaneously
- Repeated testing on validation sets
- Data leakage risks
- Non-independence of tests

### Statistical Tests for ML Models

**Comparing Two Models:**
- **Paired t-test**: Compare performance on same data splits
- **McNemar's test**: Binary classification accuracy differences
- **Wilcoxon signed-rank**: Non-parametric alternative

**Comparing Multiple Models:**
- **ANOVA with post-hoc tests**: Compare mean performance
- **Friedman test**: Non-parametric alternative for multiple models
- **Bonferroni correction**: Control FWER across pairwise comparisons

### Cross-Validation and Hypothesis Testing

**Problem**: Repeated evaluation on overlapping data violates independence assumptions.

**Solutions:**
- **Nested cross-validation**: Separate tuning from testing
- **Corrected t-tests**: Account for overlap between folds
- **Permutation tests**: Non-parametric approach to significance

## Computational Implementation Considerations

### File 1: DeepSpeed ZeRO for Hypothesis Testing at Scale

**Distributed Hypothesis Testing:**
When conducting thousands of statistical tests (e.g., feature selection, ablation studies), memory-efficient parallel processing becomes critical:

- **ZeRO Stage 1**: Distribute optimizer states for large-scale permutation testing
- **ZeRO Stage 2**: Partition gradient computations for bootstrap resampling
- **ZeRO Stage 3**: Full model sharding for massive hypothesis test suites

**Use Case**: Testing 10,000+ feature importance hypotheses in deep learning requires distributed statistical computation.

### File 9: Kubernetes for Experimental Workflows

**Orchestrating Statistical Tests:**
- **Parallel Experiments**: Run multiple hypothesis tests concurrently
- **Resource Allocation**: GPU scheduling for computationally intensive tests
- **Reproducibility**: Containerized environments ensure consistent results
- **Batch Processing**: Execute test suites systematically

**Example**: ARR-COC-0-1 ablation studies across 100+ configurations require orchestrated parallel testing.

### File 13: AMD ROCm for Statistical Computing

**Hardware Acceleration:**
- **Permutation Tests**: GPU-accelerated resampling (1000+ permutations)
- **Bootstrap Confidence Intervals**: Parallel bootstrap on AMD MI300X
- **Monte Carlo Simulations**: Simulate null distributions efficiently

**Performance**: GPU acceleration reduces 10,000 permutation test from hours to minutes.

## ARR-COC-0-1: Hypothesis Testing for Relevance Realization (10%)

### Experimental Design for Token Allocation

**Primary Hypothesis:**
- **H₀**: Adaptive relevance realization provides no improvement over uniform token allocation
- **H₁**: Relevance-based allocation improves task performance metrics

**Test Statistics:**
- Task accuracy (classification, VQA)
- Inference efficiency (tokens per correct answer)
- Quality-speed tradeoff curves

### Multiple Comparison Scenarios

**Ablation Studies** (Require Correction):
1. Test each relevance scorer (Propositional, Perspectival, Participatory)
2. Compare opponent processing strategies
3. Evaluate LOD allocation policies (64-400 tokens)
4. Assess integration methods

**Number of Tests**: 3 scorers × 4 strategies × 5 LOD policies × 3 integration methods = 180 comparisons!

**Required Correction**:
- Bonferroni: α_adjusted = 0.05 / 180 = 0.00028
- Alternative: Benjamini-Hochberg FDR control at q = 0.05

### Power Analysis for ARR-COC-0-1

**Minimum Detectable Effect (MDE):**
What is the smallest performance improvement worth detecting?

**Example Calculation:**
- Baseline accuracy: 75%
- MDE: 2% improvement (77% accuracy)
- Target power: 80%
- Significance: α = 0.05
- **Required sample size**: ~1,200 evaluation instances

**Practical Implication**: Under-powered experiments (n < 500) risk missing genuine 2% improvements.

### Cross-Modal Validation Testing

**Challenge**: Testing relevance realization across multiple benchmarks (COCO, Visual Genome, TextVQA) introduces multiple comparison issues.

**Approach:**
1. **Primary Test**: Single pre-specified benchmark (e.g., COCO)
2. **Secondary Tests**: Other benchmarks as exploratory with Bonferroni correction
3. **Meta-Analysis**: Combine evidence across benchmarks with proper weighting

### Type I vs Type II Tradeoffs

**Conservative Approach** (Low α):
- Minimize false claims of relevance improving performance
- Risk: Miss subtle but genuine improvements
- Appropriate: Published research, production deployment

**Liberal Approach** (Higher α or FDR):
- Maximize detection of potential improvements
- Risk: False positive improvements waste resources
- Appropriate: Early exploration, hypothesis generation

## Best Practices

### Experimental Design

1. **Pre-Register Hypotheses**: Specify H₀, H₁, α, and analysis plan before data collection
2. **Power Analysis**: Calculate required sample size before starting
3. **One Primary Test**: Designate one main hypothesis to avoid multiple comparison issues
4. **Exploratory vs Confirmatory**: Clearly distinguish hypothesis-generating from hypothesis-testing analyses

### Reporting Standards

From [Statistics By Jim](https://statisticsbyjim.com/hypothesis-testing/bonferroni-correction/) (accessed 2025-11-16):

**Complete Reporting Includes:**
- Exact p-values (not just "p < 0.05")
- Effect sizes with confidence intervals
- Sample sizes
- Test statistics and degrees of freedom
- Multiple comparison corrections applied
- Power analysis (pre-test, not post-hoc)

**Example:**
> "We compared relevance allocation against uniform allocation using a paired t-test (n=1,500). Relevance allocation improved accuracy by 3.2% (95% CI: 1.8% to 4.6%), t(1499) = 4.52, p = 0.00001, Cohen's d = 0.39, power = 0.95."

### Common Pitfalls to Avoid

1. **P-Hacking**: Testing multiple hypotheses until finding p < 0.05
2. **HARKing**: Hypothesizing After Results are Known
3. **Data Peeking**: Repeatedly checking significance during data collection
4. **Post-Hoc Power**: Computing power after observing results
5. **Ignoring Multiple Comparisons**: Not correcting when conducting many tests
6. **Confusing Significance with Importance**: Small p-value ≠ large or meaningful effect

## Advanced Topics

### Sequential Testing

**Problem**: Fixed sample size wastes resources if effect is large or clearly absent.

**Solution**: Sequential probability ratio test (SPRT) or group sequential methods.

**Application**: Online A/B testing, adaptive clinical trials, real-time model evaluation.

### Bayesian Hypothesis Testing

**Alternative Framework**: Compute probability of hypotheses given data, not probability of data given hypotheses.

**Bayes Factor**: Ratio of evidence for H₁ vs H₀
- BF > 10: Strong evidence for H₁
- BF < 0.1: Strong evidence for H₀
- 0.33 < BF < 3: Weak evidence either way

**Advantages**: Direct probability statements, no multiple comparison problem, incorporates prior knowledge.

### Equivalence Testing

**Goal**: Show two conditions are *equivalent*, not just "not different."

**Method**: Two one-sided tests (TOST)
- Define equivalence bounds (e.g., ±5% difference)
- Reject if both bounds exclude null effect

**Application**: Showing compressed models perform equivalently to full models.

## Practical Examples

### Example 1: Cognitive Science Experiment

**Research Question**: Does cognitive training improve working memory?

**Design:**
- H₀: μ_training = μ_control
- H₁: μ_training > μ_control (one-tailed)
- α = 0.05, power = 0.80
- Expected effect: d = 0.5 (medium)
- Required n: 64 per group

**Analysis:**
- Independent samples t-test
- Observed: t(126) = 2.89, p = 0.002 (one-tailed)
- Effect size: d = 0.52, 95% CI: [0.19, 0.85]
- **Conclusion**: Reject H₀, training improves working memory

### Example 2: Machine Learning Model Comparison

**Research Question**: Does model architecture B outperform architecture A?

**Design:**
- H₀: Accuracy_A = Accuracy_B
- H₁: Accuracy_B > Accuracy_A
- α = 0.05, 10-fold cross-validation
- Paired t-test (same data splits)

**Results:**
```
Fold    Acc_A    Acc_B    Difference
────────────────────────────────────
1       0.82     0.85     +0.03
2       0.79     0.82     +0.03
3       0.84     0.86     +0.02
...
Mean    0.815    0.838    +0.023
```

**Analysis:**
- t(9) = 4.12, p = 0.0013 (one-tailed)
- Mean difference: 2.3%, 95% CI: [1.1%, 3.5%]
- **Conclusion**: Model B significantly outperforms Model A

### Example 3: Multiple Feature Ablations

**Research Question**: Which of 8 visual features are important for image classification?

**Design:**
- Test each feature removal independently
- 8 hypothesis tests require correction
- Bonferroni: α_adjusted = 0.05 / 8 = 0.00625

**Results:**
```
Feature      p-value    Significant (α=0.00625)
──────────────────────────────────────────────
Color        0.0001     Yes ✓
Texture      0.0034     Yes ✓
Edges        0.0089     No
Shape        0.0002     Yes ✓
Size         0.145      No
Position     0.0451     No
Contrast     0.0012     Yes ✓
Brightness   0.234      No
```

**Conclusion**: 4 of 8 features (Color, Texture, Shape, Contrast) are statistically significant predictors after Bonferroni correction.

## Sources

**Source Documents:**
None directly used (web research primary source).

**Web Research:**

- [Type I and Type II Errors and Statistical Power](https://www.ncbi.nlm.nih.gov/books/NBK557530/) - NIH StatPearls (accessed 2025-11-16): Comprehensive overview of error types, statistical power, and their role in hypothesis testing

- [What is the Bonferroni Correction and How to Use It](https://statisticsbyjim.com/hypothesis-testing/bonferroni-correction/) - Statistics By Jim (accessed 2025-11-16): Detailed explanation of multiple comparison correction, family-wise error rates, and when to use Bonferroni vs alternatives

- [A Comprehensive Guide to Observed Power (Post Hoc Power)](https://blog.analytics-toolkit.com/2024/comprehensive-guide-to-observed-power-post-hoc-power/) - Analytics Toolkit (accessed 2025-11-16): Critical analysis of observed power, pre-test vs post-hoc power analysis, and common misuses in statistical practice

- [Evaluation metrics and statistical tests for machine learning](https://www.nature.com/articles/s41598-024-56706-x) - Nature Scientific Reports (2024): Comprehensive review of statistical testing methods for ML model evaluation and validation

**Influential Files:**

- **File 1** (distributed-training/00-deepspeed-zero-optimizer.md): Memory-efficient parallel processing enables large-scale hypothesis testing suites with thousands of statistical tests

- **File 9** (orchestration/00-kubernetes-gpu-scheduling.md): Container orchestration for reproducible experimental workflows and parallel hypothesis test execution

- **File 13** (alternative-hardware/00-amd-rocm-ml.md): GPU acceleration for computationally intensive statistical procedures (permutation tests, bootstrap resampling, Monte Carlo simulation)

**Additional References:**

- Multiple hypothesis testing corrections: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR
- Sequential testing methods and adaptive experimental design
- Bayesian hypothesis testing and Bayes factors
- Statistical power analysis and sample size determination
- P-value interpretation and common misconceptions
