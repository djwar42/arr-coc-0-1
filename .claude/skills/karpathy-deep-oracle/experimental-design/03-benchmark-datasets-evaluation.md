# Benchmark Datasets & Statistical Evaluation

## Overview

Rigorous evaluation of vision-language models requires standardized benchmark datasets, proper statistical hypothesis testing, and careful correction for multiple comparisons. This document covers experimental design fundamentals, benchmark selection, statistical analysis methods, and best practices for evaluating model performance with scientific rigor.

**Key principle**: Statistical significance (p < 0.05) tells us if results are unlikely due to chance, but effect size tells us if results matter practically. Both are essential.

---

## Section 1: Experimental Design Fundamentals

### 1.1 Core Principles

**Three pillars of rigorous experimentation:**

1. **Control**: Systematic manipulation of independent variables while controlling confounds
2. **Randomization**: Random assignment to conditions reduces systematic bias
3. **Replication**: Multiple trials/subjects ensure results are not flukes

**Independent vs Dependent Variables:**

- **Independent Variable (IV)**: What you manipulate (e.g., token budget: 64, 200, 400)
- **Dependent Variable (DV)**: What you measure (e.g., VQA accuracy, inference latency)
- **Confounds**: Uncontrolled variables that could explain results (e.g., image resolution, batch size)

From [DataCamp: Hypothesis Testing](https://www.datacamp.com/tutorial/hypothesis-testing) (accessed 2025-11-14):
- Hypothesis testing evaluates claims about populations using sample data
- Proper experimental design ensures internal validity (causal claims) and external validity (generalizability)

### 1.2 Between-Subjects vs Within-Subjects Design

**Between-Subjects Design:**
- Different subjects (images/samples) in each condition
- Pros: No learning/fatigue effects, simpler analysis
- Cons: Requires larger sample size, more variability
- Example: Group A gets 64-token model, Group B gets 400-token model

**Within-Subjects Design:**
- Same subjects tested in all conditions
- Pros: Controls for individual differences, smaller sample needed
- Cons: Order effects (counterbalancing required), longer experiment
- Example: Same images evaluated by all token budgets

**For ARR-COC-0-1**: Within-subjects design preferred - test each image across token budgets (64, 200, 400) to control for image difficulty variation.

### 1.3 Factorial Designs

**Definition**: Manipulate multiple independent variables simultaneously to study interactions.

**2x3 Factorial Example** (ARR-COC-0-1):
- Factor A: Query Type (2 levels: object-focused, scene-focused)
- Factor B: Token Budget (3 levels: 64, 200, 400)
- Total conditions: 2 × 3 = 6

**Why factorial designs?**
- Detect **main effects**: Overall impact of each factor
- Detect **interactions**: Does token budget effect depend on query type?

From [Medium: ML Series Day 42](https://medium.com/@ebimsv/ml-series-day-42-statistical-tests-for-model-comparison-4f5cf63da74a) (accessed 2025-11-14):
- Factorial ANOVA analyzes multiple factors simultaneously
- Interaction effects reveal when one variable's impact depends on another variable's level

### 1.4 Sample Size and Statistical Power

**Statistical Power**: Probability of detecting a true effect (rejecting false null hypothesis)

**Power Analysis Components:**
1. **Effect Size**: How large is the difference? (Cohen's d, eta-squared)
2. **Alpha (α)**: Type I error rate (false positive), typically 0.05
3. **Power (1-β)**: Desired power, typically 0.80 (80% chance of detecting true effect)
4. **Sample Size (n)**: How many observations needed?

**Rule of thumb**:
- Small effects need n > 500
- Medium effects need n > 100
- Large effects need n > 30

**For ARR-COC-0-1 ablations**: Use existing benchmark sizes (VQA v2: 214k test images, GQA: 12k validation). Report achieved power post-hoc.

---

## Section 2: Vision-Language Benchmark Datasets

### 2.1 Major VQA Benchmarks

**VQA v2.0** (Visual Question Answering)
- **Size**: 1.1M questions on 200k COCO images
- **Task**: Answer natural language questions about images
- **Metric**: VQA accuracy (exact match with ground truth)
- **Strengths**: Large-scale, diverse questions, balanced dataset
- **Weaknesses**: Answers are often single words (limited reasoning evaluation)

From [arXiv: Open-ended VQA benchmarking](https://arxiv.org/abs/2402.07270) (accessed 2025-11-14):
- Novel VQA benchmark for text-generative vision-language models
- Allows granular evaluation beyond simple accuracy metrics

**GQA** (Visual Reasoning)
- **Size**: 22M questions on 113k images
- **Task**: Compositional question answering with structured reasoning
- **Metric**: GQA accuracy + consistency score
- **Strengths**: Tests compositional reasoning, spatial relationships
- **Weaknesses**: Synthetic questions (less natural than VQA v2)

**TextVQA** (Text Reading)
- **Size**: 45k questions on 28k images
- **Task**: Answer questions requiring OCR (reading text in images)
- **Metric**: Exact match accuracy
- **Use for ARR-COC-0-1**: Tests if attention allocation handles text-heavy regions

**NaturalBench** (Real-World Evaluation)

From [NeurIPS 2024: NaturalBench](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e69ff56d0ebff0752ff29caaddc25dd-Paper-Datasets_and_Benchmarks_Track.pdf) (accessed 2025-11-14):
- Evaluates VLMs on natural, challenging images
- Tests complex visio-linguistic reasoning beyond standard benchmarks

### 2.2 Benchmark Selection Criteria

**How to choose benchmarks for ARR-COC-0-1:**

1. **Task Alignment**: Does benchmark test relevance realization capabilities?
   - VQA v2: Tests query-driven attention
   - GQA: Tests compositional relevance across objects

2. **Scale**: Large enough for statistical power?
   - Minimum 1,000 samples per condition for robust statistics

3. **Diversity**: Covers multiple query types and visual scenarios?
   - Include object-focused, scene-focused, spatial, counting questions

4. **Established Baselines**: Can we compare to prior work?
   - VQA v2, GQA have extensive leaderboards

5. **Evaluation Metrics**: Supports effect size calculation?
   - Accuracy alone insufficient - need variance estimates

**Recommended for ARR-COC-0-1**:
- **Primary**: VQA v2 (query-driven relevance)
- **Secondary**: GQA (compositional reasoning), TextVQA (attention to text)
- **Ablation**: Subset of 5,000 images across all benchmarks for detailed token budget analysis

### 2.3 Dataset Splits and Cross-Validation

**Standard Split**:
- Train: 70-80% (for training/fine-tuning)
- Validation: 10-15% (for hyperparameter tuning)
- Test: 10-15% (held out for final evaluation, never touched during development)

**k-Fold Cross-Validation** (when data is limited):
- Split data into k folds (typically k=5 or k=10)
- Train on k-1 folds, validate on remaining fold
- Repeat k times, average results
- **Advantage**: Uses all data for both training and validation
- **Disadvantage**: Computationally expensive (k training runs)

**For ARR-COC-0-1**: Use standard benchmark splits (no cross-validation needed for large datasets like VQA v2).

---

## Section 3: Statistical Hypothesis Testing Framework

### 3.1 Null Hypothesis Testing Paradigm

**Framework**:
1. **Null Hypothesis (H₀)**: No effect exists (e.g., "Token budget has no impact on VQA accuracy")
2. **Alternative Hypothesis (H₁)**: Effect exists (e.g., "400-token budget improves accuracy over 64-token")
3. **Test Statistic**: Calculate from data (t, F, χ²)
4. **P-value**: Probability of observing data this extreme if H₀ is true
5. **Decision**: Reject H₀ if p < α (typically α = 0.05)

**Critical principle**: p < 0.05 does NOT mean "effect is important" - it means "effect is unlikely due to chance." Must also report effect size.

From [Towards Data Science: Statistical Tests](https://towardsdatascience.com/statistical-tests-t-test-andanova-674b242a5274/) (accessed 2025-11-14):
- t-test compares means between two groups
- ANOVA compares means among three or more groups
- Both test whether observed differences are statistically significant

### 3.2 t-Tests: Comparing Two Means

**Independent Samples t-test**:
- Compares two separate groups
- Example: Compare VQA accuracy of 64-token vs 400-token budget
- Assumptions: Normal distribution, equal variances (Levene's test)

**Formula**:
```
t = (mean₁ - mean₂) / SE_diff
SE_diff = sqrt(s₁²/n₁ + s₂²/n₂)
```

**Paired Samples t-test**:
- Compares same subjects in two conditions
- Example: Same images evaluated by two models
- More powerful than independent t-test (controls for image difficulty)

**One-Sample t-test**:
- Compares sample mean to known value
- Example: Does ARR-COC-0-1 exceed 70% accuracy threshold?

**For ARR-COC-0-1**:
- Use **paired t-test** when same images tested across token budgets
- Report t-statistic, degrees of freedom, p-value, Cohen's d

### 3.3 ANOVA: Comparing Multiple Groups

**One-Way ANOVA**:
- Compares 3+ groups on single factor
- Example: Compare VQA accuracy across token budgets (64, 200, 400)
- Null hypothesis: μ₁ = μ₂ = μ₃ (all means equal)

**F-statistic**:
```
F = MS_between / MS_within
MS_between = variance between group means
MS_within = variance within groups (error)
```

**Interpretation**:
- Large F → group means differ more than expected by chance
- p < 0.05 → at least one group differs significantly
- **But**: ANOVA doesn't tell you WHICH groups differ (need post-hoc tests)

From [Analytics Vidhya: 5 Statistical Tests](https://www.analyticsvidhya.com/blog/2024/07/statistical-tests-every-data-scientist-should-know/) (accessed 2025-11-14):
- ANOVA tests whether group means differ significantly
- Post-hoc tests (Tukey HSD, Bonferroni) identify which specific groups differ

**Factorial ANOVA** (2+ factors):
- Example: 2x3 ANOVA with Query Type × Token Budget
- Tests:
  - Main effect of Query Type
  - Main effect of Token Budget
  - Query Type × Token Budget interaction

**Repeated Measures ANOVA**:
- Within-subjects design (same images across conditions)
- Accounts for correlation between repeated measures
- More powerful than between-subjects ANOVA

**For ARR-COC-0-1**:
- Use **Repeated Measures ANOVA** for token budget ablation (same images across budgets)
- Report F-statistic, df, p-value, eta-squared (η²)

### 3.4 Non-Parametric Tests

**When to use**:
- Data violates normality assumption (Shapiro-Wilk test p < 0.05)
- Small sample sizes (n < 30)
- Ordinal data or ranks

**Mann-Whitney U Test** (non-parametric t-test):
- Compares two independent groups
- Based on ranks, not raw values
- Example: Compare median latency of two models

**Wilcoxon Signed-Rank Test** (non-parametric paired t-test):
- Compares two related groups
- Example: Paired image evaluations

**Kruskal-Wallis Test** (non-parametric ANOVA):
- Compares 3+ independent groups
- Example: Token budget comparison when accuracy is not normally distributed

**For ARR-COC-0-1**: First test normality. If violated, use non-parametric alternatives.

---

## Section 4: Effect Sizes and Practical Significance

### 4.1 Why Effect Size Matters

**Problem with p-values alone**:
- Large sample → tiny effects become "significant" (p < 0.05)
- Small sample → large effects may not reach significance
- p-value doesn't tell you HOW MUCH difference exists

**Solution**: Report effect size alongside p-value.

From [ResearchGate: On Comparable Effect Sizes 2024](https://www.researchgate.net/publication/379033915_On_comparable_effect_sizes_2024) (accessed 2025-11-14):
- Effect sizes quantify magnitude of differences between groups
- Essential for understanding practical significance beyond statistical significance

From [Statistics By Jim: Effect Sizes](https://statisticsbyjim.com/basics/effect-sizes-statistics/) (accessed 2025-11-14):
- Effect sizes measure strength of relationships independent of sample size
- Standardized effect sizes enable comparison across studies

### 4.2 Cohen's d (Standardized Mean Difference)

**Formula**:
```
d = (mean₁ - mean₂) / pooled_SD
```

**Interpretation** (Cohen's guidelines):
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

**Example**:
- 64-token accuracy: 68.5% (SD = 4.2)
- 400-token accuracy: 74.3% (SD = 3.8)
- d = (74.3 - 68.5) / 4.0 = 1.45 (very large effect)

**For ARR-COC-0-1**: Report Cohen's d for all pairwise comparisons of token budgets.

### 4.3 Eta-Squared (η²) for ANOVA

**Formula**:
```
η² = SS_between / SS_total
```

**Interpretation**:
- η² = proportion of variance explained by factor
- η² = 0.01: Small effect (1% of variance)
- η² = 0.06: Medium effect (6% of variance)
- η² = 0.14: Large effect (14% of variance)

From [Medium: Effect Size](https://medium.com/@TingyuZou/effect-size-68e915bc22f3) (accessed 2025-11-14):
- Cohen's d for t-tests, eta-squared for ANOVA, Pearson's r for correlations
- Each effect size measure appropriate for different statistical tests

**Partial Eta-Squared (ηₚ²)**:
- Accounts for other factors in factorial design
- ηₚ² = SS_effect / (SS_effect + SS_error)

**For ARR-COC-0-1**: Report eta-squared for ANOVA main effects and interactions.

### 4.4 Practical Significance Thresholds

**Academic guidelines are insufficient** - define domain-specific thresholds:

**For VQA accuracy**:
- Δ < 1%: Negligible (within measurement noise)
- Δ = 1-3%: Small but meaningful
- Δ = 3-5%: Moderate practical impact
- Δ > 5%: Large practical impact (publishable)

**For inference latency**:
- Δ < 10ms: Negligible for most applications
- Δ = 10-50ms: Noticeable in interactive systems
- Δ > 50ms: Major user experience impact

**Report both**:
- Statistical significance: "p < 0.001"
- Practical significance: "5.8% accuracy improvement (large effect, d = 1.45)"

---

## Section 5: Multiple Comparisons Correction

### 5.1 The Multiple Comparisons Problem

**Problem**: Running k independent tests, each with α = 0.05, inflates family-wise error rate (FWER).

**Family-Wise Error Rate**:
```
FWER = 1 - (1 - α)^k
```

**Example**:
- 10 comparisons at α = 0.05
- FWER = 1 - (0.95)^10 = 40.1% chance of at least one false positive

**When correction is needed**:
- Multiple post-hoc comparisons after ANOVA
- Testing multiple dependent variables
- Subgroup analyses

From [Columbia Public Health: False Discovery Rate](https://www.publichealth.columbia.edu/research/population-health-methods/false-discovery-rate) (accessed 2025-11-14):
- Traditional Bonferroni correction is overly conservative for many comparisons
- FDR provides less stringent control while maintaining interpretability

### 5.2 Bonferroni Correction (Conservative)

**Method**: Divide significance level by number of comparisons

```
α_corrected = α / k
```

**Example**:
- 6 pairwise comparisons (3 token budgets)
- α_corrected = 0.05 / 6 = 0.0083
- Reject H₀ only if p < 0.0083

**Pros**:
- Simple to apply
- Controls FWER exactly

**Cons**:
- Very conservative (reduces statistical power)
- Too strict when k is large (e.g., k > 10)

**For ARR-COC-0-1**: Use Bonferroni for small number of planned comparisons (≤ 10).

### 5.3 Holm-Bonferroni (Step-Down) Correction

**Method**: Sequential testing from smallest to largest p-value

1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Compare p₁ to α/k, p₂ to α/(k-1), ..., pₖ to α/1
3. Stop at first non-significant result

**Advantages over Bonferroni**:
- More powerful (less conservative)
- Still controls FWER

**For ARR-COC-0-1**: Preferred over standard Bonferroni when k > 5.

### 5.4 False Discovery Rate (FDR) - Benjamini-Hochberg

**Philosophy**: Control proportion of false discoveries rather than probability of any false discovery

**FDR**: Expected proportion of false positives among all discoveries
```
FDR = E[FP / (FP + TP)]
```

**Benjamini-Hochberg Procedure**:
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Find largest i where pᵢ ≤ (i/k) × α
3. Reject H₀ for all tests 1, 2, ..., i

**Example** (k=10, α=0.05):
```
Test  p-value  (i/k)×0.05  Reject?
1     0.001    0.005       Yes (0.001 < 0.005)
2     0.008    0.010       Yes (0.008 < 0.010)
3     0.012    0.015       Yes (0.012 < 0.015)
4     0.025    0.020       No  (0.025 > 0.020) → STOP
```

From [Wikipedia: False Discovery Rate](https://en.wikipedia.org/wiki/False_discovery_rate) (accessed 2025-11-14):
- FDR provides less stringent control than FWER methods (Bonferroni)
- Appropriate when some false positives are acceptable to gain statistical power

**When to use FDR vs Bonferroni**:
- **Bonferroni**: Exploratory research, few comparisons, must avoid any false positives
- **FDR**: Large-scale testing (e.g., genomics), willing to accept 5% false discoveries for more power

**For ARR-COC-0-1**: Use FDR when testing many ablations (e.g., 20+ hyperparameter configurations).

### 5.5 Permutation Tests

**Concept**: Generate null distribution by randomly permuting labels

**Procedure**:
1. Calculate test statistic on real data: t_obs
2. Randomly shuffle group labels N times (N = 1,000 - 10,000)
3. Calculate test statistic for each permutation: t_perm
4. p-value = proportion of permutations where |t_perm| ≥ |t_obs|

**Advantages**:
- No parametric assumptions (distribution-free)
- Can correct for multiple comparisons via permutation-based FDR
- Exact p-values (not asymptotic)

**Disadvantages**:
- Computationally expensive
- Requires exchangeability assumption

**For ARR-COC-0-1**: Use permutation tests when normality assumption is violated or sample sizes are unequal.

---

## Section 6: Post-Hoc Tests for ANOVA

### 6.1 Why Post-Hoc Tests?

**ANOVA tells you**: "At least one group differs"

**Post-hoc tests tell you**: "Which specific groups differ"

**Example**:
- ANOVA: F(2,297) = 15.6, p < 0.001 (significant)
- Post-hoc: 64 vs 400 tokens (p < 0.001), 200 vs 400 tokens (p = 0.08, n.s.)

### 6.2 Tukey's Honest Significant Difference (HSD)

**Method**: Pairwise comparisons controlling FWER

**Test statistic** (Studentized range):
```
q = (mean_i - mean_j) / SE
```

**Advantages**:
- Controls FWER exactly
- Balanced power across all comparisons
- Widely used and accepted

**Use when**: All pairwise comparisons of equal interest

**For ARR-COC-0-1**: Use Tukey HSD after repeated measures ANOVA for token budget comparisons.

### 6.3 Bonferroni Post-Hoc Test

**Method**: Apply Bonferroni correction to all pairwise t-tests

**Advantages**:
- Simple, flexible (can test any contrasts)
- Controls FWER

**Disadvantages**:
- More conservative than Tukey for all pairwise comparisons
- Power decreases rapidly with many comparisons

**Use when**: Only a few planned comparisons (not all pairwise)

### 6.4 Dunnett's Test

**Method**: Compare all groups to single control group

**Example**: Compare 200-token and 400-token to baseline 64-token model

**Advantages**:
- More powerful than Tukey when only control comparisons matter
- Controls FWER

**For ARR-COC-0-1**: Use Dunnett when 64-token budget is baseline and we only care about improvements.

### 6.5 Games-Howell (Unequal Variances)

**Method**: Post-hoc test for unequal variances (Welch's correction)

**Use when**:
- Levene's test shows unequal variances (p < 0.05)
- Sample sizes are unequal

**For ARR-COC-0-1**: Use if token budget groups have different variance (e.g., 400-token more variable due to complex allocation).

---

## Section 7: Reporting Statistical Results

### 7.1 APA Style Guidelines

**Format for t-test**:
"VQA accuracy was significantly higher for 400-token budget (M = 74.3, SD = 3.8) than 64-token budget (M = 68.5, SD = 4.2), t(99) = 8.45, p < 0.001, d = 1.45."

**Format for ANOVA**:
"A repeated measures ANOVA revealed a significant main effect of token budget on VQA accuracy, F(2,198) = 42.3, p < 0.001, ηₚ² = 0.30. Post-hoc Tukey tests showed all pairwise comparisons were significant (ps < 0.01)."

**Format for interaction**:
"A significant Query Type × Token Budget interaction emerged, F(2,396) = 5.67, p = 0.004, ηₚ² = 0.03, indicating that token budget effects were larger for scene-focused queries than object-focused queries."

### 7.2 Essential Reporting Elements

**Always include**:
1. **Descriptive statistics**: M, SD (or median, IQR for non-parametric)
2. **Test statistic**: t, F, χ², U, etc.
3. **Degrees of freedom**: df
4. **Exact p-value**: p = 0.023 (not p < 0.05, unless p < 0.001)
5. **Effect size**: d, η², r
6. **Confidence intervals**: 95% CI [lower, upper]

**Example table**:
```
Token Budget   M (SD)      95% CI
64 tokens      68.5 (4.2)  [67.7, 69.3]
200 tokens     72.1 (3.9)  [71.3, 72.9]
400 tokens     74.3 (3.8)  [73.5, 75.1]

Pairwise comparisons (Tukey HSD):
64 vs 200:  p < 0.001, d = 0.88
64 vs 400:  p < 0.001, d = 1.45
200 vs 400: p = 0.004, d = 0.57
```

### 7.3 Visualization Best Practices

**Bar plots with error bars**:
- Show means with 95% CI error bars (not SE or SD)
- Include individual data points (transparency if many)
- Label axes clearly with units

**Box plots**:
- Show distribution (median, quartiles, outliers)
- Useful for non-normal data

**Interaction plots**:
- For factorial designs
- Plot means with separate lines for each factor level

**Effect size plots**:
- Forest plots showing Cohen's d with 95% CI
- Helps visualize practical significance

From [NeurIPS 2024: ConvBench](https://papers.nips.cc/paper_files/paper/2024/file/b69396afc07a9ca3428d194f4db84c02-Paper-Datasets_and_Benchmarks_Track.pdf) (accessed 2025-11-14):
- Multi-turn conversation evaluation benchmark for vision-language models
- Comprehensive evaluation methodology with proper statistical reporting

---

## Section 8: ARR-COC-0-1 Statistical Validation Strategy

### 8.1 Experimental Design for Token Budget Ablation

**Research Questions**:
1. Does token budget (64, 200, 400) significantly affect VQA accuracy?
2. Does the effect size justify computational cost?
3. Does token budget interact with query type?

**Design**:
- **Type**: 2×3 Repeated Measures Factorial
- **Factor A**: Query Type (object-focused, scene-focused) - within-subjects
- **Factor B**: Token Budget (64, 200, 400) - within-subjects
- **Dependent Variable**: VQA accuracy (%)
- **Sample Size**: 5,000 images from VQA v2 validation set
- **Power**: >0.95 for medium effects (d ≥ 0.5)

**Counterbalancing**: Randomize order of token budgets across images to prevent order effects.

### 8.2 Primary Analysis Plan

**Step 1: Descriptive Statistics**
- Calculate M, SD, 95% CI for each token budget
- Check normality (Shapiro-Wilk test, Q-Q plots)
- Check sphericity for repeated measures (Mauchly's test)

**Step 2: Main Analysis**
- 2×3 Repeated Measures ANOVA
- If sphericity violated: Apply Greenhouse-Geisser correction
- Report F, df, p, ηₚ²

**Step 3: Post-Hoc Comparisons**
- Tukey HSD for all pairwise token budget comparisons
- Report p-values (Bonferroni-adjusted), Cohen's d

**Step 4: Effect Size Interpretation**
- Calculate Cohen's d for each comparison
- Assess practical significance:
  - Δ < 3%: Not practically significant
  - Δ = 3-5%: Moderate practical impact
  - Δ > 5%: Large practical impact

### 8.3 Secondary Analyses

**Query Type Interaction**:
- Test if token budget effects differ by query type
- Simple effects analysis if interaction is significant

**Inference Latency Trade-offs**:
- Correlate token budget with inference time
- Calculate efficiency metric: accuracy_gain / latency_cost

**Robustness Checks**:
- Repeat analysis on GQA benchmark
- Test if results hold across image complexity levels (simple vs complex scenes)

### 8.4 Reporting Template for ARR-COC-0-1 Paper

**Methods Section**:
"We evaluated ARR-COC-0-1 on VQA v2 validation set (n = 5,000 images) using a 2×3 repeated measures factorial design with Query Type (object-focused, scene-focused) and Token Budget (64, 200, 400) as within-subjects factors. We conducted repeated measures ANOVA with Greenhouse-Geisser correction if sphericity was violated. Post-hoc pairwise comparisons used Tukey HSD with Bonferroni correction. We reported partial eta-squared (ηₚ²) for ANOVA effects and Cohen's d for pairwise comparisons. Statistical analyses were conducted in Python using scipy.stats and statsmodels."

**Results Section**:
"Token budget significantly affected VQA accuracy, F(2,9998) = 127.4, p < 0.001, ηₚ² = 0.42 (large effect). Post-hoc Tukey tests revealed all pairwise comparisons were significant (ps < 0.001). The 400-token budget (M = 74.3%, SD = 3.8) outperformed the 64-token budget (M = 68.5%, SD = 4.2) by 5.8 percentage points (d = 1.45, very large effect). This represents a 8.5% relative improvement with practical significance for VQA applications. A significant Query Type × Token Budget interaction emerged, F(2,9998) = 8.23, p < 0.001, ηₚ² = 0.06, with larger token budget benefits for scene-focused queries (Δ = 7.2%) than object-focused queries (Δ = 4.4%)."

### 8.5 Benchmark Evaluation Protocol

**VQA v2 Evaluation**:
```python
# Pseudocode
for token_budget in [64, 200, 400]:
    for image in vqa_v2_val:
        prediction = model.predict(image, query, budget=token_budget)
        accuracy = exact_match(prediction, ground_truth)

    # Calculate statistics
    mean_acc = np.mean(accuracies)
    ci_95 = bootstrap_ci(accuracies, n_bootstrap=10000)

    # Store results
    results[token_budget] = {
        'mean': mean_acc,
        'ci_95': ci_95,
        'n': len(accuracies)
    }

# Statistical comparison
f_stat, p_value = repeated_measures_anova(results)
effect_size = partial_eta_squared(results)
post_hoc = tukey_hsd(results)
```

**GQA Evaluation** (compositional reasoning):
- Report GQA accuracy + consistency score
- Analyze performance on spatial, relational, and logical question types separately

**TextVQA Evaluation** (attention to text):
- Report exact match accuracy
- Analyze if attention allocation properly weights text regions

### 8.6 Statistical Analysis Checklist

**Before analysis**:
- [ ] Check for missing data (report handling strategy)
- [ ] Test normality assumptions (Shapiro-Wilk, Q-Q plots)
- [ ] Test homogeneity of variance (Levene's test)
- [ ] Test sphericity for repeated measures (Mauchly's test)

**During analysis**:
- [ ] Use appropriate test (parametric vs non-parametric)
- [ ] Apply corrections (Greenhouse-Geisser, Bonferroni)
- [ ] Calculate effect sizes (Cohen's d, eta-squared)
- [ ] Compute confidence intervals (95% CI)

**Reporting**:
- [ ] Include descriptive statistics (M, SD, 95% CI)
- [ ] Report exact p-values (not just p < 0.05)
- [ ] Report effect sizes with interpretation
- [ ] Include visualizations (plots with error bars)
- [ ] State practical significance explicitly

---

## Section 9: Advanced Topics

### 9.1 Bayesian Hypothesis Testing

**Alternative to frequentist null hypothesis testing**:

**Bayes Factor (BF₁₀)**:
- Ratio of evidence for H₁ vs H₀
- BF₁₀ = 10: Data are 10× more likely under H₁ than H₀

**Advantages**:
- Quantifies evidence for null (not just reject/fail to reject)
- Allows sequential testing (no inflation of Type I error)
- Incorporates prior beliefs

**Interpretation**:
- BF₁₀ = 1-3: Anecdotal evidence
- BF₁₀ = 3-10: Moderate evidence
- BF₁₀ = 10-30: Strong evidence
- BF₁₀ > 30: Very strong evidence

**For ARR-COC-0-1**: Consider Bayesian t-tests for small sample exploratory analyses where sequential testing is needed.

### 9.2 Equivalence Testing (TOST)

**Problem**: "Not significant" ≠ "no effect"

**Solution**: Test if effect is smaller than smallest effect size of interest (SESOI)

**Two One-Sided Tests (TOST)**:
1. Test if effect > -SESOI
2. Test if effect < +SESOI
3. If both p < 0.05, conclude equivalence

**Example**: Test if 200-token and 400-token budgets are equivalent within 1% accuracy

**For ARR-COC-0-1**: Use TOST to claim token budgets are "practically equivalent" (e.g., 200 vs 400 on simple images).

### 9.3 Meta-Analysis of Benchmark Results

**Combine results across multiple benchmarks**:
- VQA v2, GQA, TextVQA
- Calculate weighted average effect size
- Test for heterogeneity (Q-statistic)

**Fixed vs Random Effects**:
- Fixed: Assumes true effect is same across benchmarks
- Random: Allows true effect to vary across benchmarks

**For ARR-COC-0-1**: Meta-analyze token budget effects across 3 benchmarks to quantify generalizability.

### 9.4 Regression-Based Analysis

**Alternative to ANOVA**: Treat token budget as continuous predictor

**Linear Regression**:
```
Accuracy = β₀ + β₁(Token_Budget) + ε
```

**Advantages**:
- Can model non-linear relationships (polynomial regression)
- Can include continuous covariates (image complexity)
- Tests trend (linear, quadratic)

**Mixed Effects Regression**:
- Random intercepts for images (controls for image difficulty)
- Fixed effects for token budget
- More powerful than repeated measures ANOVA

**For ARR-COC-0-1**: Use mixed effects models to account for image-level and query-level variability simultaneously.

---

## Sources

### Web Research

**Statistical Hypothesis Testing:**
- [Towards Data Science: Statistical Tests](https://towardsdatascience.com/statistical-tests-t-test-andanova-674b242a5274/) - t-test and ANOVA fundamentals (accessed 2025-11-14)
- [DataCamp: Hypothesis Testing](https://www.datacamp.com/tutorial/hypothesis-testing) - Hypothesis testing framework (accessed 2025-11-14)
- [Medium: ML Series Day 42](https://medium.com/@ebimsv/ml-series-day-42-statistical-tests-for-model-comparison-4f5cf63da74a) - Statistical tests for model comparison (accessed 2025-11-14)
- [Analytics Vidhya: 5 Statistical Tests](https://www.analyticsvidhya.com/blog/2024/07/statistical-tests-every-data-scientist-should-know/) - Essential statistical tests for data science (accessed 2025-11-14)

**Effect Sizes:**
- [ResearchGate: On Comparable Effect Sizes 2024](https://www.researchgate.net/publication/379033915_On_comparable_effect_sizes_2024) - Effect size fundamentals (accessed 2025-11-14)
- [Statistics By Jim: Effect Sizes](https://statisticsbyjim.com/basics/effect-sizes-statistics/) - Cohen's d and practical significance (accessed 2025-11-14)
- [Medium: Effect Size by Tingyu Zou](https://medium.com/@TingyuZou/effect-size-68e915bc22f3) - When to use Cohen's d, eta-squared, Pearson's r (accessed 2025-11-14)

**Multiple Comparisons:**
- [Columbia Public Health: False Discovery Rate](https://www.publichealth.columbia.edu/research/population-health-methods/false-discovery-rate) - FDR vs Bonferroni correction (accessed 2025-11-14)
- [Wikipedia: False Discovery Rate](https://en.wikipedia.org/wiki/False_discovery_rate) - FDR methodology and interpretation (accessed 2025-11-14)

**Vision-Language Benchmarks:**
- [arXiv: Open-ended VQA Benchmarking (Ging et al., 2024)](https://arxiv.org/abs/2402.07270) - Novel VQA benchmark for text-generative VLMs (accessed 2025-11-14)
- [NeurIPS 2024: NaturalBench (Li et al.)](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e69ff56d0ebff0752ff29caaddc25dd-Paper-Datasets_and_Benchmarks_Track.pdf) - Natural challenging VQA evaluation (accessed 2025-11-14)
- [NeurIPS 2024: ConvBench (Liu et al.)](https://papers.nips.cc/paper_files/paper/2024/file/b69396afc07a9ca3428d194f4db84c02-Paper-Datasets_and_Benchmarks_Track.pdf) - Multi-turn conversation evaluation (accessed 2025-11-14)
- [ICLR 2024: Open-Ended VQA Benchmarking](https://proceedings.iclr.cc/paper_files/paper/2024/file/cb1c4782f159b55380b4584671c4fd88-Paper-Conference.pdf) - Granular VLM evaluation methodology (accessed 2025-11-14)

### Additional References

**Academic Papers:**
- Chen, S. Y., et al. (2017). "A general introduction to adjustment for multiple comparisons." *Journal of Thoracic Disease*, 9(6), 1725-1729.
- Maher, J. M., et al. (2013). "Effect Size Analysis in Quantitative Research." *CBE—Life Sciences Education*, 12(4), 608-617.

**Statistical Methodology:**
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: A practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

---

**Document Version**: 1.0
**Created**: 2025-11-14
**Lines**: ~700
**Related**: experimental-design/, research-methodology/
