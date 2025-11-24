# Effect Sizes & Practical Significance

## Overview

Effect sizes quantify the magnitude of experimental findings independent of sample size, bridging the gap between statistical significance (p-values) and practical importance. While p-values answer "Is there an effect?", effect sizes answer "How large is the effect?" and "Does it matter in practice?" - critical questions for ARR-COC token allocation decisions where practical impact determines resource investment.

**Core principle**: Statistical significance ≠ practical significance. A tiny effect can be statistically significant with large n, while a large effect might not reach significance with small n. Effect sizes provide the missing piece: quantifying magnitude independent of sample size.

From [Mann et al., JACC Basic to Translational Science, 2024](https://www.jacc.org/doi/10.1016/j.jacbts.2024.01.008):
- "Effect size is a quantitative measure of the magnitude of the experimental effect... independent of sample size"
- Enables comparison across studies with different sample sizes and measurements
- Essential for meta-analyses and research synthesis

## Cohen's d: Standardized Mean Difference

### Definition and Calculation

Cohen's d measures the standardized difference between two group means, expressing the difference in standard deviation units. It's the most common effect size for comparing two groups.

**Formula**: d = (M₁ - M₂) / s

Where:
- M₁, M₂ = means of groups 1 and 2
- s = pooled standard deviation (or control group SD, or pretest SD depending on design)

From [Statistics By Jim](https://statisticsbyjim.com/basics/cohens-d/) (accessed 2025-11-16):
"Cohen's d characterizes the effect size by relating the mean difference to variability, similar to a signal-to-noise ratio. A large Cohen's d indicates the mean difference (effect size = signal) is large compared to the variability (noise)."

### Interpretation Guidelines

**Cohen's conventions** (Cohen, 1988):
- d = 0.2: Small effect (mean difference is 0.2 SD)
- d = 0.5: Medium effect (mean difference is 0.5 SD)
- d = 0.8: Large effect (mean difference is 0.8 SD)

**Critical caveat**: These are general guidelines. Practical significance depends on domain context. In medical research, d < 1.0 may not be clinically meaningful since it's within normal variation. In educational interventions, d = 0.2 might represent substantial progress.

From [Brydges & Baguley, Innovation in Aging, 2019](https://academic.oup.com/innovateage/article/3/4/igz036/5560156):
- Cohen's guidelines often overestimate effect sizes in gerontology
- Recommend domain-specific benchmarks: r = .10, .20, .30 (or d = 0.20, 0.40, 0.60) for aging research
- Field-specific norms more meaningful than universal standards

### When to Use Cohen's d

**Advantages (unitless measure)**:
1. Comparable across studies with different measurement scales
2. Useful when original units lack intrinsic meaning (psychological inventories)
3. Enables meta-analysis across heterogeneous measures
4. Standard in psychology literature

**Disadvantages (unitless measure)**:
1. Less intuitive than raw mean differences with units
2. Requires context to interpret practical significance
3. Loses connection to real-world quantities

From [Baguley, British Journal of Psychology, 2009](https://pubmed.ncbi.nlm.nih.gov/19017432/):
- "Being so disinterested in our variables that we do not care about their units can hardly be desirable" (Tukey, 1969)
- Prefer unstandardized effects (mean difference in original units) when units are meaningful
- Use standardized effects when comparing across different measures or when units lack inherent meaning

**Best practice**: Report both standardized (Cohen's d) and unstandardized (mean difference with units) effect sizes when possible.

## Correlation Coefficient r: Association Strength

### Definition and Properties

Pearson's r measures the strength and direction of linear relationships between two continuous variables. Unlike Cohen's d (for group differences), r quantifies association.

**Properties**:
- Range: -1 to +1
- Sign indicates direction (positive/negative correlation)
- Magnitude indicates strength (|r| closer to 1 = stronger)
- r² = proportion of variance explained (coefficient of determination)

From [Hedges & Schauer, Psychological Methods, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562970/):
"The standardized mean difference (sometimes called Cohen's d) is an effect size measure widely used to describe the outcomes of experiments. The correlation coefficient is its counterpart for correlational studies."

### Interpretation Guidelines

**Cohen's conventions for r**:
- r = .10: Small effect (1% variance explained)
- r = .30: Medium effect (9% variance explained)
- r = .50: Large effect (25% variance explained)

**Updated recommendations** (Brydges & Baguley, 2019):
- Gerontology: r = .10, .20, .30 for small/medium/large
- Many fields: Cohen's r thresholds too liberal

From [ResearchGate, 2024](https://www.researchgate.net/publication/378465566_R_effect_size_and_Generalized_Cohen_d_2024_with_name):
"Renewed thresholds for r effect size, parallel and comparable with Cohen's d, are 0.05 for 'very small', 0.09-0.10 for 'small', 0.15 for 'small-medium', 0.20 for 'medium', 0.30 for 'medium-large', and 0.40 for 'large'."

### Relationship to Cohen's d

**Approximate conversion** (for 2x2 design):
- d = 2r / √(1-r²)
- r = d / √(d² + 4)

This allows translating between group comparison (d) and correlation (r) frameworks, useful for meta-analyses combining different study designs.

## Eta-Squared (η²) and Partial Eta-Squared: ANOVA Effect Sizes

### Eta-Squared (η²)

Measures proportion of total variance in the dependent variable explained by the independent variable(s) in ANOVA.

**Formula**: η² = SS_effect / SS_total

Where:
- SS_effect = sum of squares for the effect
- SS_total = total sum of squares

**Interpretation**:
- η² = .01: Small effect (1% variance explained)
- η² = .06: Medium effect (6% variance explained)
- η² = .14: Large effect (14% variance explained)

From [Levine & Hullett, Human Communication Research, 2002](https://academic.oup.com/hcr/article-abstract/28/4/612/4331349):
"Eta squared (η²) is the most commonly reported estimate of effect size for the ANOVA. However, partial eta squared is often confused with eta squared, leading to misreporting."

### Partial Eta-Squared (ηp²)

Measures proportion of variance explained by an effect **after removing variance from other effects**. Default in SPSS and many software packages.

**Formula**: ηp² = SS_effect / (SS_effect + SS_error)

**Key difference**: Partial η² excludes variance from other IVs in the denominator, while η² uses total variance. This makes ηp² generally larger than η² in factorial designs.

From [National University LibGuides](https://resources.nu.edu/statsresources/eta):
"Partial eta squared is telling us how large of an effect the independent variable(s) had on the dependent variable."

**Critical reporting issue**: Many papers report ηp² but label it as η², inflating apparent effect sizes. Always specify which measure you're reporting.

### When to Use Each

- **η²**: Single-factor ANOVA, comparable to R² in regression
- **ηp²**: Multi-factor ANOVA, isolates each effect's unique contribution
- Both are bounded [0, 1], but ηp² ≥ η² always

## Confidence Intervals for Effect Sizes

### Why Confidence Intervals Matter

Effect sizes, like any statistic, have sampling uncertainty. Confidence intervals (CIs) quantify this precision, showing the range of plausible population effect sizes.

From [Boscardin et al., Perspectives on Medical Education, 2024](https://pmejournal.org/articles/10.5334/pme.1324):
"Effect sizes provide a measure of the magnitude of an effect, while confidence intervals provide the precision (level of certainty) around that estimate."

**Key advantages of CIs**:
1. Show precision of the estimate (narrow CI = more precise)
2. Reveal practical significance range (does the CI include trivial effects?)
3. Support estimation-based inference (not just reject/accept null)
4. Allow comparing effects across studies

### Interpreting Effect Size CIs

From [Elkins, Journal of Physiotherapy, 2024](https://www.sciencedirect.com/science/article/pii/S1836955324000869):
"A confidence interval from 0.25 to 0.55 can be interpreted as a small to medium effect, whereas a confidence interval from -0.8 to 0.2 can be interpreted as spanning from a large negative effect to a small positive effect."

**Example interpretations**:

| Effect Size Point Estimate | 95% CI | Interpretation |
|----------------------------|---------|----------------|
| d = 0.40 | [0.25, 0.55] | Consistently small-medium effect |
| d = 0.40 | [-0.10, 0.90] | Highly uncertain, wide range |
| d = 0.15 | [0.10, 0.20] | Precise but small effect |
| r = 0.45 | [0.30, 0.60] | Moderate-large, consistent |

**Critical insight**: A statistically significant effect with a CI including practically trivial values suggests limited practical importance, even with p < .05.

### Confidence Intervals vs P-values

From [Rovetta, PMC11814670, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11814670/):
"A 95% 'confidence' interval (95% CI) can be employed to assess the consistency between data and several hypotheses about the effect instead of the sole null hypothesis... p-values and confidence intervals are compatibility measures."

**Relationship**:
- If 95% CI excludes the null value (e.g., 0 for differences), p < .05
- If 95% CI includes the null value, p > .05
- But CIs provide richer information than binary significance

**Best practice**: Always report CIs alongside effect sizes. They convey both magnitude and precision in one measure.

## Statistical Significance vs Practical Significance

### The Critical Distinction

From [Scribbr](https://www.scribbr.com/statistics/effect-size/) (accessed 2025-11-16):
"While **statistical significance** shows that an effect exists in a study, **practical significance** shows that the effect is large enough to be meaningful in the real world. Statistical significance is denoted by p-values, whereas practical significance is represented by effect sizes."

**The sample size paradox**:
- Large n makes any tiny effect statistically significant (p < .05)
- Small n can miss large, important effects (p > .05)
- Effect sizes are independent of sample size - they quantify magnitude directly

**Example**: With n = 13,000 per group, a weight loss difference of 0.1 kg can be statistically significant (p = .01) but practically meaningless (Cohen's d = 0.015).

### Determining Practical Significance

**Domain-specific criteria** are essential. Generic rules of thumb are starting points, not endpoints.

From [Mann et al., JACC, 2024](https://www.jacc.org/doi/10.1016/j.jacbts.2024.01.008):
"Cohen suggested a convention to interpret the magnitude of the effect size, where d = 0.2 is a small effect, d = 0.5 is a moderate effect, and d = 0.8 is a large effect. However, these are arbitrary and should be considered in the context of the specific field."

**Questions for assessing practical significance**:
1. Is the effect size clinically/educationally/economically meaningful?
2. Does it exceed minimal clinically important difference (MCID)?
3. How does it compare to existing interventions?
4. What are the costs/risks relative to benefits?
5. Does the CI lower bound still indicate practical value?

**Example - Medical Context**:
- Blood pressure reduction: d = 0.3 might be clinically significant if sustained
- Depression inventory: d = 0.5 often considered minimum for clinical importance
- Mortality rate: Even small effects (d = 0.1) can be highly significant at population scale

From [learning-gate.com, 2024](https://learning-gate.com/index.php/2576-8484/article/view/2062):
"Effect sizes can be understood in the context of confidence intervals, which provide similar information to statistical significance tests but focus on the size of the effect and its precision rather than just whether it exists."

## Critical Effect Size Values

### Minimum Detectable Effect Sizes

From [Perugini et al., Psychological Science, 2025](https://journals.sagepub.com/doi/10.1177/25152459251335298):
"Critical-effect-size values represent the smallest detectable effect that can reach statistical significance given a specific sample size... Reporting critical-effect-size values provides transparency about what effects a study can and cannot detect."

**Definition**: Given sample size n, alpha level, and power (typically .80), what is the minimum effect size detectable?

**Example calculation**:
- Two-group t-test, n = 50 per group, α = .05, power = .80
- Minimum detectable d ≈ 0.57
- Any true effect d < 0.57 likely to be non-significant (underpowered)

**Implication**: If your study can only detect d = 0.57 or larger, but you find non-significant results, you cannot rule out effects in the range 0 < d < 0.57. The study lacks sensitivity for small-to-medium effects.

### Power Analysis and Effect Sizes

**Before data collection**: Choose expected effect size (from pilot data or literature) → determine required n for desired power.

**After data collection**: Report observed effect size with CI. Transparency about detectable effect range prevents misinterpretation of null findings.

From [Riesthuis & Garnier, 2024](https://garstats.wordpress.com/2024/09/13/riesthuis-ampps-2024/):
"Simulation-based power analyses using confidence intervals allow researchers to determine the sample size needed to detect the smallest effect size of interest with adequate precision."

## Pipeline Parallelism and Effect Size Computation (File 2)

**Connection to DeepSeek Pipeline Parallelism** (distributed-training/01-deepspeed-pipeline-parallelism.md):

Effect size calculations in large-scale experiments with multi-stage pipelines benefit from distributed computation strategies:

1. **Stage-wise computation**: Calculate effect sizes per pipeline stage (early layers vs late layers)
2. **Micro-batch aggregation**: Accumulate statistics across micro-batches for stable estimates
3. **Gradient accumulation analog**: Sum of squares accumulates like gradients for variance estimation
4. **Memory efficiency**: ZeRO-style partitioning for storing statistics across GPUs

**ARR-COC application**: When evaluating relevance scorer performance (propositional/perspectival/participatory), pipeline parallelism enables efficient effect size computation across patches:
- Partition image into K patches (pipeline stages)
- Compute relevance scores per patch (forward pass)
- Aggregate effect sizes comparing different allocation strategies
- Efficient memory use for large-scale ablation studies

## Kubeflow ML Pipelines for Effect Size Experiments (File 10)

**Connection to Kubeflow ML Pipelines** (orchestration/01-kubeflow-ml-pipelines.md):

Effect size research requires systematic experimentation across conditions. Kubeflow orchestrates:

1. **Experiment pipeline**: Data prep → Model training → Evaluation → Effect size calculation
2. **Hyperparameter sweeps**: Test token allocation (K=100 vs 200) → compute Cohen's d for performance difference
3. **Metadata tracking**: Store effect sizes, CIs, p-values in ML Metadata
4. **Visualization**: Kubeflow Pipelines UI displays effect size distributions
5. **Reproducibility**: Containerized pipelines ensure consistent effect size calculations

**ARR-COC experiments**:
```python
# Kubeflow pipeline for token allocation effect size
@dsl.pipeline(name='ARR-COC Effect Size Study')
def token_allocation_effect_size():
    # Baseline: uniform allocation
    baseline = train_model(allocation='uniform', K=200)

    # Treatment: relevance-based allocation
    treatment = train_model(allocation='relevance', K=200)

    # Compute effect size
    effect_size = compute_cohens_d(
        baseline.outputs['accuracy'],
        treatment.outputs['accuracy']
    )

    # CI estimation via bootstrap
    ci = bootstrap_ci(baseline, treatment, n_iterations=1000)
```

Benefits:
- Automated effect size reporting in MLOps pipelines
- Version-controlled experiment designs
- Scalable to hundreds of experimental conditions
- Integrates with Vertex AI for distributed execution

## Apple Metal Performance Primitives for Effect Size (File 14)

**Connection to Apple Metal ML** (alternative-hardware/01-apple-metal-ml.md):

Effect size calculations for on-device model evaluation leverage Metal Performance Shaders:

1. **GPU-accelerated statistics**: Metal computes means, variances, covariances for thousands of samples
2. **Real-time effect monitoring**: Track Cohen's d during on-device training (federated learning)
3. **Energy efficiency**: Low-power effect size computation on M4 (important for mobile experiments)
4. **Neural Engine integration**: Offload model inference to ANE, statistics to GPU

**ARR-COC on Apple Silicon**:
```swift
// Metal shader for variance computation (Cohen's d denominator)
kernel void compute_pooled_variance(
    device float* group1_scores [[buffer(0)]],
    device float* group2_scores [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Efficient parallel variance calculation
    float mean1 = compute_mean(group1_scores);
    float mean2 = compute_mean(group2_scores);
    float var1 = compute_variance(group1_scores, mean1);
    float var2 = compute_variance(group2_scores, mean2);

    // Pooled SD
    result[0] = sqrt((var1 + var2) / 2.0);
}
```

Applications:
- On-device A/B testing (relevance-based vs uniform allocation)
- Federated effect size aggregation across devices
- Privacy-preserving experimental analysis (local computation)
- Energy-efficient confidence interval estimation via bootstrapping

## ARR-COC-0-1: Token Allocation as Effect Size Optimization

### Practical Significance of Token Allocation

ARR-COC's variable LOD system (64-400 tokens per patch) is fundamentally about **maximizing practical significance** of allocated computational resources:

**The token allocation problem as effect size reasoning**:
- **Statistical question**: "Does allocating more tokens to high-relevance patches improve performance?" (p-value)
- **Effect size question**: "HOW MUCH does relevance-based allocation improve performance?" (Cohen's d, CI)
- **Practical question**: "Is the improvement worth the computational cost?" (practical significance)

From ARR-COC implementation (arr-coc-0-1/arr_coc/attending.py):
```python
class TokenBudgetAllocator:
    """
    Maps relevance scores to token budgets (64-400 range).

    Effect size reasoning:
    - High relevance patches: Large effect expected → allocate 400 tokens
    - Medium relevance: Medium effect → 200 tokens
    - Low relevance: Small/no effect → 64 tokens (minimal resources)
    """

    def allocate_budgets(self, relevance_scores, total_K=200):
        # Softmax converts relevance to allocation distribution
        # This is equivalent to: "invest computational resources
        # proportional to expected effect size"
        weights = softmax(relevance_scores)
        budgets = clip(weights * total_K, min=64, max=400)
        return budgets
```

### Measuring Allocation Effect Sizes

**Experimental design** for validating ARR-COC:

1. **Control group**: Uniform allocation (all patches get K/N tokens)
2. **Treatment group**: Relevance-based allocation (64-400 tokens)
3. **Effect size**: Cohen's d for accuracy difference
4. **Practical significance**: Is improvement worth computational overhead?

**Example results** (hypothetical):
```
Experimental Condition    | VQA Accuracy | Tokens Used | Compute Time
--------------------------|--------------|-------------|-------------
Uniform allocation        | 72.3%        | 200K        | 1.2s
Relevance-based (ARR-COC) | 76.8%        | 180K        | 1.0s

Effect size: d = 0.65 (medium-large effect)
95% CI for accuracy difference: [3.2%, 5.8%]
Practical significance: 4.5% accuracy gain + 10% speedup + 10% token reduction
```

**Interpretation**: Medium-large effect size (d = 0.65) with tight CI suggests robust practical benefit. The effect is not just statistically significant (p < .001) but practically meaningful - substantial accuracy improvement with fewer resources.

### Confidence Intervals for Allocation Strategies

ARR-COC experiments should report **confidence intervals** for performance differences, not just p-values:

**Bootstrap CI estimation**:
```python
def allocation_effect_ci(baseline_model, treatment_model, n_bootstrap=1000):
    """
    Compute 95% CI for Cohen's d comparing allocation strategies.
    """
    effect_sizes = []

    for _ in range(n_bootstrap):
        # Resample validation set with replacement
        baseline_acc = baseline_model.evaluate(resample(val_set))
        treatment_acc = treatment_model.evaluate(resample(val_set))

        # Compute effect size for this bootstrap sample
        pooled_sd = compute_pooled_sd(baseline_acc, treatment_acc)
        d = (treatment_acc - baseline_acc) / pooled_sd
        effect_sizes.append(d)

    # 95% CI from bootstrap distribution
    lower = np.percentile(effect_sizes, 2.5)
    upper = np.percentile(effect_sizes, 97.5)

    return lower, upper
```

**Reporting**: "Relevance-based allocation improved VQA accuracy by 4.5 percentage points (Cohen's d = 0.65, 95% CI [0.42, 0.88], p < .001). The effect size indicates a robust, practically significant improvement that persists across resampling."

### Minimal Practically Important Difference (MPID)

For ARR-COC, define **minimal practically important difference** based on application needs:

**Example MPID thresholds**:
- Medical VQA: 2% accuracy improvement (high stakes)
- General VQA: 1% accuracy improvement (competitive benchmarks)
- Educational applications: 3% improvement (justify deployment costs)

**Pre-registration**: Before experiments, specify:
1. Expected effect size (from pilot data)
2. MPID threshold
3. Sample size needed for power = .80 to detect MPID
4. Analysis plan (primary outcome, effect size measures)

This prevents p-hacking and ensures experiments can detect practically meaningful effects.

### Token Allocation as Resource-Rational Effect Size

ARR-COC embodies **resource-rational analysis** (Griffiths et al., Psychological Review, 2015): allocate limited resources (tokens) to maximize expected utility (relevance realization).

**Effect size framework**:
- **Relevance scorers** estimate potential effect size of each patch
- **Token allocation** invests resources proportional to expected effect
- **Opponent processing** balances effect size against computational cost
- **Validation** measures actual effect sizes to refine allocation policy

This creates a feedback loop:
1. Estimate effect sizes (relevance scores)
2. Allocate resources accordingly
3. Measure actual effect sizes (accuracy, F1, etc.)
4. Update allocation policy to maximize practical significance per token

The result: **Adaptive token allocation that optimizes practical significance**, not just statistical significance.

## Meta-Analysis and Effect Size Aggregation

### Why Effect Sizes Enable Meta-Analysis

Meta-analysis combines results from multiple studies to estimate overall effect size. This is only possible because effect sizes are standardized - d = 0.5 means the same thing across studies, even with different samples and measures.

From [Maher et al., CBE-Life Sciences Education, 2013](https://www.lifescied.org/doi/10.1187/cbe.13-04-0082):
"Effect size measures are a key complement to statistical significance testing when reporting quantitative research findings... they enable synthesis across studies in systematic reviews and meta-analyses."

**Meta-analytic workflow**:
1. Extract effect sizes (d or r) from each study
2. Weight by precision (inverse variance weighting)
3. Compute weighted mean effect size
4. Test for heterogeneity (are effects consistent across studies?)
5. Report summary effect with CI

**ARR-COC meta-analysis example**:
```
Study 1 (VQA): d = 0.58 [0.32, 0.84]
Study 2 (Visual Reasoning): d = 0.71 [0.45, 0.97]
Study 3 (Image Captioning): d = 0.42 [0.18, 0.66]

Meta-analytic summary: d = 0.57 [0.41, 0.73]
Interpretation: Relevance-based allocation shows consistent medium effect across tasks
```

### Publication Bias and Effect Sizes

**File-drawer problem**: Studies with small, non-significant effects less likely to be published. This biases meta-analyses toward larger effects.

From [Juniper Publishers, 2024](https://juniperpublishers.com/bboaj/BBOAJ.MS.ID.555819.php):
"It offers guidelines for reporting effect size, confidence intervals, and observed statistical power in conjunction with statistical significance... to address publication bias and improve reproducibility."

**Solutions**:
1. Pre-register studies (commit to publishing regardless of p-value)
2. Report effect sizes and CIs even for null results
3. Test for publication bias (funnel plots, trim-and-fill)
4. Encourage replication studies

ARR-COC experiments should:
- Report effect sizes for all ablations, not just "significant" ones
- Publish negative results (e.g., "texture features had minimal effect, d = 0.08")
- Share data and code for reproducible effect size calculations

## Power Analysis and Sample Size Determination

### Statistical Power

From [Scribbr](https://www.scribbr.com/statistics/effect-size/):
"In statistics, power refers to the likelihood of a hypothesis test detecting a true effect if there is one. A statistically powerful test is more likely to reject a false negative (a Type II error)."

**Power (1 - β)**: Probability of detecting an effect of size d, given α and n.

Typical convention: Power ≥ .80 (80% chance of detecting the effect if it exists).

### Sample Size Calculation

**Inputs**:
- Expected effect size (from pilot or literature)
- Alpha level (typically .05)
- Desired power (typically .80)

**Output**: Minimum sample size needed

**Example** (two-group comparison):
- Expected d = 0.5 (medium effect)
- α = .05, power = .80
- Required n ≈ 64 per group (128 total)

If you only collect n = 30 per group, power drops to ~.47 (less than 50% chance of detecting d = 0.5).

**Implication for ARR-COC**: When validating relevance-based allocation, ensure sufficient validation set size to detect meaningful effects with high power. Underpowered experiments risk missing true benefits.

### Post-hoc Power Analysis

**Warning**: Calculating power after seeing results is controversial. It's a function of the observed p-value and adds no new information.

From [Lenth, 2001](https://stat.uiowa.edu/sites/stat.uiowa.edu/files/techrep/tr378.pdf):
"Observed power is a function of the p-value... It is completely uninformative for interpreting the results of a study."

**Instead**: Report effect size with CI. The CI width conveys precision (analogous to power), and the effect size magnitude indicates practical significance.

## Reporting Guidelines

### APA Style Recommendations

From APA 7th edition guidelines:
1. Always report effect sizes for primary outcomes
2. Include confidence intervals for effect sizes when possible
3. Report exact p-values (not just p < .05)
4. Describe practical significance, not just statistical significance

**Example reporting**:
"Relevance-based token allocation (M = 76.8, SD = 4.2) significantly outperformed uniform allocation (M = 72.3, SD = 4.5), t(198) = 7.21, p < .001, d = 0.65, 95% CI [0.42, 0.88]. This represents a medium-to-large effect size, with the treatment group scoring 4.5 percentage points higher on average."

### Complete Effect Size Reporting Checklist

**For every primary analysis, report**:
- [ ] Effect size estimate (d, r, η², etc.)
- [ ] 95% confidence interval for effect size
- [ ] Sample sizes (n per group)
- [ ] Exact p-value
- [ ] Descriptive statistics (M, SD for each group)
- [ ] Practical significance interpretation

**Optional but recommended**:
- [ ] Power analysis (prospective) or CI width (retrospective)
- [ ] Comparison to domain-specific benchmarks
- [ ] Visualization (effect size plot with error bars)

## Sources

**Primary Research Articles**:
- Mann, D. L., et al. (2024). [Is it Time to Abandon the Use of P-Values in Early Phase Clinical Trials?](https://www.jacc.org/doi/10.1016/j.jacbts.2024.01.008) JACC: Basic to Translational Science. (accessed 2025-11-16)
- Hedges, L. V., & Schauer, J. M. (2024). [Interpretation of the Standardized Mean Difference Effect Size](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562970/). National Institutes of Health. (accessed 2025-11-16)
- Brydges, C. R., & Baguley, T. (2019). [Effect Size Guidelines, Sample Size Calculations, and Statistical Power in Gerontology](https://academic.oup.com/innovateage/article/3/4/igz036/5560156). Innovation in Aging, 3(4). (accessed 2025-11-16)
- Levine, T. R., & Hullett, C. R. (2002). [Eta Squared, Partial Eta Squared, and Misreporting of Effect Size](https://academic.oup.com/hcr/article-abstract/28/4/612/4331349). Human Communication Research, 28(4), 612-625. (accessed 2025-11-16)
- Baguley, T. (2009). [Standardized or simple effect size: what should be reported?](https://pubmed.ncbi.nlm.nih.gov/19017432/) British Journal of Psychology, 100(Pt 3), 603-17. (accessed 2025-11-16)

**Methodological Resources**:
- Rovetta, A. (2025). [p-Values and confidence intervals as compatibility measures](https://pmc.ncbi.nlm.nih.gov/articles/PMC11814670/). PMC. (accessed 2025-11-16)
- Boscardin, C. K., et al. (2024). [How to Use and Report on p-values](https://pmejournal.org/articles/10.5334/pme.1324). Perspectives on Medical Education. (accessed 2025-11-16)
- Elkins, M. (2024). [Appraisal Research Note: Interpreting confidence intervals](https://www.sciencedirect.com/science/article/pii/S1836955324000869). Journal of Physiotherapy. (accessed 2025-11-16)
- Perugini, M., et al. (2025). [The Benefits of Reporting Critical-Effect-Size Values](https://journals.sagepub.com/doi/10.1177/25152459251335298). Psychological Science. (accessed 2025-11-16)

**Educational Guides**:
- Scribbr. (2023). [What is Effect Size and Why Does It Matter?](https://www.scribbr.com/statistics/effect-size/) (accessed 2025-11-16)
- Statistics By Jim. [Cohens D: Definition, Using & Examples](https://statisticsbyjim.com/basics/cohens-d/) (accessed 2025-11-16)
- National University. [LibGuides: Partial Eta Squared](https://resources.nu.edu/statsresources/eta) (accessed 2025-11-16)

**2024 Research**:
- ResearchGate. (2024). [R effect size and Generalized Cohen d 2024](https://www.researchgate.net/publication/378465566_R_effect_size_and_Generalized_Cohen_d_2024_with_name) (accessed 2025-11-16)
- learning-gate.com. (2024). [The role of effect size and significance test in research design](https://learning-gate.com/index.php/2576-8484/article/view/2062) (accessed 2025-11-16)
- Riesthuis, P., & Garnier, M. (2024). [Simulation-Based Power Analyses for the Smallest Effect Size of Interest](https://garstats.wordpress.com/2024/09/13/riesthuis-ampps-2024/) (accessed 2025-11-16)

**Engineering Infrastructure References** (Karpathy Deep Oracle):
- File 2: distributed-training/01-deepspeed-pipeline-parallelism.md - Pipeline parallel computation for effect size statistics
- File 10: orchestration/01-kubeflow-ml-pipelines.md - ML experiment orchestration with automated effect size reporting
- File 14: alternative-hardware/01-apple-metal-ml.md - GPU-accelerated statistics on Apple Silicon

**ARR-COC Implementation**:
- arr-coc-0-1/arr_coc/attending.py - Token budget allocation based on relevance (practical significance optimization)
- arr-coc-0-1/arr_coc/balancing.py - Opponent processing (cost-benefit analysis)
- ARR-COC experimental validation (hypothetical studies illustrating effect size reporting)
