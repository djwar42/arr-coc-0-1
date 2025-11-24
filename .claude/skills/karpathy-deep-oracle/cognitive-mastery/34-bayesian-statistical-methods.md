# Bayesian Statistical Methods

## Overview

Bayesian statistical methods provide a principled framework for scientific inference that quantifies uncertainty, incorporates prior knowledge, and makes probabilistic statements about parameters. Unlike frequentist methods that treat parameters as fixed unknowns, Bayesian statistics treats them as random variables with probability distributions, enabling direct probability statements about hypotheses and parameters.

**Core Philosophy**: Probability represents degree of belief, updated through evidence. Parameters have distributions. Inference combines prior knowledge with observed data to produce posterior beliefs.

**Key Innovation for ARR-COC-0-1**: Token allocation decisions can be evaluated using Bayesian model comparison - Bayes factors quantify evidence for different relevance allocation strategies, enabling principled comparison of competing hypotheses about optimal compression.

From existing knowledge ([06-bayesian-inference-deep.md](06-bayesian-inference-deep.md)):
- Bayesian inference uses all of probability theory, not just Bayes' theorem
- Posterior = Likelihood × Prior / Evidence
- Marginalization (total probability) is equally fundamental to Bayes' theorem

## 1. Bayesian t-Test: Probabilistic Group Comparisons

### The Bayesian Alternative to Frequentist t-Test

From [Model-averaged Bayesian t tests (Springer, 2024)](https://link.springer.com/article/10.3758/s13423-024-02590-5):

**"Bayesian t-tests simultaneously take into account models that assume different prior variances, providing robust inference when prior specification is uncertain."**

Key advantages:
- Direct probability statements: "Probability that μ₁ > μ₂ given data"
- Incorporates prior knowledge about effect sizes
- Handles small sample sizes more gracefully
- Quantifies evidence for null hypothesis (not just against it)

### Bayesian Independent Samples t-Test

**Model specification**:
```
Group 1: y₁ᵢ ~ N(μ₁, σ²)
Group 2: y₂ⱼ ~ N(μ₂, σ²)

Parameter of interest: δ = (μ₁ - μ₂) / σ  (effect size)

Prior on δ: Cauchy(0, r)  where r = scale parameter
```

**Two competing hypotheses**:
- H₀: δ = 0 (no difference)
- H₁: δ ~ Cauchy(0, r) (some difference exists)

**Inference**: Compute posterior distribution P(δ | data) and make probabilistic statements like:
- "There is a 97% probability that μ₁ > μ₂"
- "The effect size is between 0.3 and 0.8 with 95% probability"

### Model-Averaged Bayesian t-Tests

From [Maier et al. (2024)](https://link.springer.com/article/10.3758/s13423-024-02590-5):

**Challenge**: Prior specification (e.g., scale r in Cauchy prior) affects results.

**Solution**: Average over multiple priors with different scales:
```
P(δ | data) = Σᵢ P(δ | data, priorᵢ) × P(priorᵢ | data)
```

This **Bayesian model averaging** provides robust inference when uncertain about prior specification.

### ARR-COC-0-1 Application: Comparing Allocation Strategies

**Research question**: Does query-aware allocation outperform uniform allocation?

**Bayesian t-test setup**:
```
Group 1 (Query-aware): FID scores under ARR-COC allocation
Group 2 (Uniform): FID scores under uniform token allocation

H₀: No difference in performance
H₁: Query-aware allocation improves quality

Prior on effect size: Cauchy(0, 0.707) (medium effect)
```

**Advantages over frequentist t-test**:
- Direct statement: "95% probability that query-aware improves FID by 2-8 points"
- Can quantify evidence for null (no improvement) via BF₀₁
- Incorporates prior belief about expected effect sizes in VLM compression

## 2. Bayes Factors: Quantifying Evidential Support

### The Bayes Factor as Evidence Ratio

From [Tutorial on Bayesian t-Test (Wiley, 2025)](https://onlinelibrary.wiley.com/doi/10.1111/jan.70122):

**Definition**:
```
BF₁₀ = P(Data | H₁) / P(Data | H₀)

Where:
- P(Data | H₁) = marginal likelihood under alternative hypothesis
- P(Data | H₀) = marginal likelihood under null hypothesis
```

**Interpretation** (Jeffreys' scale):
```
BF₁₀ = 1-3:    Anecdotal evidence for H₁
BF₁₀ = 3-10:   Moderate evidence for H₁
BF₁₀ = 10-30:  Strong evidence for H₁
BF₁₀ = 30-100: Very strong evidence for H₁
BF₁₀ > 100:    Extreme evidence for H₁

BF₁₀ < 1:      Evidence favors H₀
BF₀₁ = 1/BF₁₀
```

### Computing Bayes Factors

**For simple models** (conjugate priors):
Analytical computation via marginal likelihoods.

**For complex models**:
- Bridge sampling
- Savage-Dickey density ratio (when H₀ is nested in H₁)
- Laplace approximation
- Importance sampling

From [Dudbridge (2024) - Empirical Bayes Factors](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297874):

**Empirical Bayes approach**: Estimate prior hyperparameters from data, then compute BF. Provides **test-based empirical Bayes factors** for standard tests (t-test, chi-squared, etc.).

### Bayes Factors vs p-Values

From [Tendeiro & Kiers (2024) - Diagnosing Misuse](https://journals.sagepub.com/doi/10.1177/25152459231213371):

**Critical differences**:

**p-value**:
- P(Data or more extreme | H₀ is true)
- Does NOT quantify evidence for H₀
- Depends on sampling intentions
- Cannot support null hypothesis

**Bayes Factor**:
- P(Data | H₁) / P(Data | H₀)
- Quantifies relative evidence for both hypotheses
- Independent of sampling intentions
- Can provide evidence FOR null hypothesis
- Directly answers: "Which hypothesis is better supported?"

**Common misinterpretation of p-values**: p = 0.05 does NOT mean 95% confidence in rejecting H₀. It means: "If H₀ were true, we'd see data this extreme 5% of the time."

### ARR-COC-0-1 Application: Model Selection for Relevance Scorers

**Research question**: Which relevance scorer best predicts human eye fixations?

**Competing models**:
1. **M₁ (Information)**: Propositional scorer only (Shannon entropy)
2. **M₂ (Salience)**: Perspectival scorer only (NeVa salience)
3. **M₃ (Coupling)**: Participatory scorer only (cross-attention)
4. **M₄ (Full)**: All three scorers combined

**Compute Bayes factors**:
```
BF₄₃ = P(Fixation data | M₄) / P(Fixation data | M₃)

If BF₄₃ = 25: "Strong evidence that combining all three scorers
               better predicts fixations than coupling alone"
```

**Advantage**: Can quantify evidence that simpler model (M₂) is sufficient, not just reject complex model.

## 3. Credible Intervals: Bayesian Uncertainty Quantification

### Definition and Interpretation

From [Statsig (2024) - Credible vs Confidence Intervals](https://www.statsig.com/perspectives/credible-vs-confidence-intervals):

**Credible interval**: Derived from posterior distribution P(θ | data).

**95% credible interval**: There is a 95% probability that the true parameter value lies within this range, given the observed data and prior beliefs.

**Contrast with confidence interval**:
- **Confidence interval**: "If we repeated sampling infinite times, 95% of constructed intervals would contain true parameter"
- **Credible interval**: "Given this data, there's 95% probability parameter is in this range"

**Intuitive interpretation**: Credible intervals provide the probabilistic statement most people THINK confidence intervals provide.

### Types of Credible Intervals

**Equal-tailed interval (ETI)**:
```
[θ₀.₀₂₅, θ₀.₉₇₅]  where θₚ is p-th percentile of posterior
```
2.5% probability in each tail.

**Highest density interval (HDI)**:
```
Shortest interval containing 95% of posterior mass
```

From [Edwards (2025) - Using HDI](https://www.sciencedirect.com/science/article/pii/S0165783625000633):

**"HDI can reduce perceived risk compared to ETI, especially for skewed posteriors. For fisheries management, HDI gives more conservative lower bounds on biomass estimates."**

**Example**: Posterior on biomass is right-skewed.
- ETI [0.4, 2.30]: Lower bound 0.4
- HDI [0.45, 2.15]: Lower bound 0.45 (more conservative)

### Constructing Credible Intervals

**Steps**:
1. Obtain posterior distribution P(θ | data) via:
   - Analytical solution (conjugate priors)
   - MCMC sampling (Stan, PyMC)
   - Variational inference (approximate)

2. Compute percentiles or density regions:
```python
# Equal-tailed interval
lower = np.percentile(posterior_samples, 2.5)
upper = np.percentile(posterior_samples, 97.5)

# Highest density interval (via PyMC or arviz)
import arviz as az
hdi = az.hdi(posterior_samples, hdi_prob=0.95)
```

3. Report: "95% credible interval: [lower, upper]"

### ARR-COC-0-1 Application: Uncertainty in Token Budget Optimization

**Research question**: What is the optimal K (number of patches allocated)?

**Bayesian approach**:
```
Data: FID scores for K ∈ {50, 100, 150, 200, 250}

Model: FID(K) = a + b/K + ε,  ε ~ N(0, σ²)

Priors:
  a ~ N(15, 5²)     # Expected minimum FID
  b ~ N(1000, 500²) # Token efficiency parameter
  σ ~ Exponential(1)

Posterior: P(a, b, σ | FID data)
```

**Compute optimal K**:
```
K_opt = argmin_K E[FID(K) | data]
```

**Quantify uncertainty**:
```
95% credible interval for K_opt: [180, 220]

Interpretation: "Given our data, there's 95% probability that
                 the optimal patch count is between 180 and 220"
```

**Advantage**: Direct probabilistic statement about optimal budget, not just point estimate.

## 4. Prior Specification and Sensitivity Analysis

### The Challenge of Prior Selection

From [Veenman et al. (2024) - Hierarchical Bayesian Modeling](https://link.springer.com/article/10.3758/s13428-023-02204-3):

**"Our introduction gives particular emphasis to prior specification and prior sensitivity, as well as to the calculation of Bayes factors for model comparisons."**

**Prior types**:

**Informative priors**: Strong beliefs based on domain knowledge
```python
# Example: Human foveal acuity
prior_cpd = Normal(60, 5)  # cycles per degree
# Justified by decades of psychophysics research
```

**Weakly informative priors**: Regularize without dominating
```python
# Example: Relevance scores bounded [0, 1]
prior_relevance = Beta(2, 2)  # Slight preference for 0.5
```

**Non-informative priors**: Maximum uncertainty (use cautiously)
```python
prior_uniform = Uniform(0, 1)
prior_jeffreys = ImproperPrior()  # Scale-invariant
```

### Prior Sensitivity Analysis

From [Sekulovski et al. (2024) - Sensitivity Analysis](https://advances.in/psychology/10.56296/aip00016/):

**Goal**: Assess robustness of conclusions to prior specification.

**Method**:
1. Specify multiple plausible priors (varying informativeness/location)
2. Compute posterior for each prior
3. Compare conclusions:
   - If consistent across priors → Robust inference
   - If varying → Prior-dependent, report sensitivity

**Example**:
```python
# Test three priors for effect size
priors = [
    Cauchy(0, 0.5),   # Narrow (skeptical)
    Cauchy(0, 0.707), # Medium (default)
    Cauchy(0, 1.0)    # Wide (optimistic)
]

for prior in priors:
    posterior = compute_posterior(data, prior)
    bf = compute_bayes_factor(posterior)
    print(f"Prior: {prior}, BF₁₀: {bf:.2f}")

# Output:
# Prior: Cauchy(0, 0.5),   BF₁₀: 8.3
# Prior: Cauchy(0, 0.707), BF₁₀: 5.2
# Prior: Cauchy(0, 1.0),   BF₁₀: 3.7

# Conclusion: Moderate-to-strong evidence across all priors
```

### Empirical Bayes: Data-Driven Priors

**Controversial but practical**: Estimate hyperparameters from data.

**Hierarchical empirical Bayes**:
```
Level 1 (observations): yᵢ ~ N(θᵢ, σ²)
Level 2 (parameters):    θᵢ ~ N(μ, τ²)
Level 3 (hyperparameters): Estimate μ, τ² from data
```

**Advantage**: Less subjective, leverages data structure.
**Disadvantage**: "Using data twice" - prior no longer independent of data.

From [van Geen et al. (2021) - Hierarchical Bayesian RL](https://www.sciencedirect.com/science/article/abs/pii/S0022249621000742):

**"Hierarchical Bayesian modeling provides an alternate method for deriving empirical priors in reinforcement learning contexts."**

### ARR-COC-0-1 Application: Prior Sensitivity for Relevance Thresholds

**Research question**: At what relevance score should we allocate maximum tokens?

**Prior specification**:
```
θ_threshold ~ Beta(α, β)

Test three priors:
1. Skeptical: Beta(2, 8)    # Prior belief: low relevance typical
2. Neutral:   Beta(5, 5)    # Uniform-ish
3. Optimistic: Beta(8, 2)   # Prior belief: high relevance common
```

**Sensitivity analysis**:
```python
results = []
for prior_params in [(2,8), (5,5), (8,2)]:
    posterior = fit_model(fixation_data, prior_params)
    threshold_95 = posterior.quantile(0.95)
    results.append(threshold_95)

# Results: [0.72, 0.78, 0.84]
# Conclusion: Threshold estimate varies by ~0.12 across priors
#             Moderately sensitive - report range
```

**Recommendation**: Report sensitivity and justify prior choice based on pilot data or literature.

## 5. Hierarchical Bayesian Models

### Motivation: Partial Pooling

From [Wesner (2024) - Hierarchical Bayesian Size Spectra](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14312):

**"Using hierarchical Bayesian models has the benefit of improving parameter estimates with partial pooling - borrowing strength across groups."**

**Three estimation strategies**:

**Complete pooling**: Ignore group structure (one model for all)
```
yᵢⱼ ~ N(μ, σ²)  for all groups j
```
Over-simplifies, loses group-specific information.

**No pooling**: Separate model per group
```
yᵢⱼ ~ N(μⱼ, σ²)  independent estimates
```
Over-fits small groups, doesn't share information.

**Partial pooling** (hierarchical Bayesian):
```
Level 1: yᵢⱼ ~ N(μⱼ, σ²)
Level 2: μⱼ ~ N(μ_global, τ²)
```
Shares information across groups while preserving group differences.

### Structure of Hierarchical Models

**General form**:
```
Observations: y ~ Likelihood(θ)
Parameters:   θ ~ Prior(ψ)
Hyperparameters: ψ ~ Hyperprior
```

**Example - Multi-site clinical trial**:
```
Level 1 (patients): Effect in site j: yᵢⱼ ~ N(θⱼ, σ²)
Level 2 (sites):    Site effects:      θⱼ ~ N(μ, τ²)
Level 3 (global):   Global mean:       μ ~ N(0, 10²)
                    Between-site var:  τ ~ Exponential(1)
```

**Shrinkage**: Small sites get "shrunk" toward global mean. Large sites less affected.

### Benefits of Hierarchical Modeling

1. **Improved estimates**: Borrowing strength across groups
2. **Handles missing data**: Can estimate parameters for groups with sparse data
3. **Natural regularization**: Automatic shrinkage prevents overfitting
4. **Uncertainty propagation**: Quantifies uncertainty at all levels

From [Zhao et al. (2024) - Hierarchical Bayesian Contrast Sensitivity](https://tvst.arvojournals.org/article.aspx?articleid=2802354):

**"Hierarchical Bayesian models enable advanced statistical inference on contrast sensitivity functions, pooling information across subjects while respecting individual differences."**

### ARR-COC-0-1 Application: Multi-Query Relevance Modeling

**Research question**: How does relevance realization vary across query types?

**Hierarchical model**:
```
Level 1 (images): Relevance score for image i, query type j
  r_ij ~ Beta(α_j * m_j, (1 - α_j) * m_j)

  where α_j = mean relevance, m_j = precision

Level 2 (query types): Query-specific parameters
  α_j ~ Beta(α_global * m_global, (1 - α_global) * m_global)
  m_j ~ Gamma(a, b)

Level 3 (global): Population-level hyperparameters
  α_global ~ Beta(5, 5)  # Weakly informative
  m_global ~ Gamma(2, 0.1)
```

**Benefits**:
- Query types with few examples (e.g., "spatial reasoning") borrow strength from other queries
- Quantifies between-query variability in relevance
- Enables prediction for new query types

**Inference**: Use Stan or PyMC for MCMC sampling of full posterior.

## 6. Bayesian Model Comparison

### Beyond Single Hypotheses

**Frequentist paradigm**: Test H₀ vs H₁.
**Bayesian paradigm**: Compare multiple models simultaneously.

**Model comparison via Bayes factors**:
```
BF₁₂ = P(Data | M₁) / P(Data | M₂)

For K models: Compute BF for all pairs, or use posterior model probabilities
```

### Posterior Model Probabilities

From [Elsemüller et al. (2024) - Deep Learning for BMC](https://psycnet.apa.org/record/2024-80637-001):

**Given K models with priors P(Mₖ)**:
```
P(Mₖ | Data) = P(Data | Mₖ) × P(Mₖ) / P(Data)

where P(Data) = Σⱼ P(Data | Mⱼ) × P(Mⱼ)
```

**Interpretation**: Direct probability that Mₖ is the true model, given data.

**Example**:
```
Models: M₁ (Linear), M₂ (Quadratic), M₃ (Cubic)
Equal priors: P(M₁) = P(M₂) = P(M₃) = 1/3

After observing data:
P(M₁ | Data) = 0.05  (unlikely)
P(M₂ | Data) = 0.82  (most probable)
P(M₃ | Data) = 0.13  (overfitting)

Conclusion: Quadratic model best supported
```

### Bayesian Model Averaging

**When uncertain about model**: Average predictions across models.

```
P(ŷ | Data) = Σₖ P(ŷ | Mₖ, Data) × P(Mₖ | Data)
```

**Advantage**: Accounts for model uncertainty in predictions.

From [Maier et al. (2024)](https://link.springer.com/article/10.3758/s13423-024-02590-5):

**"Model-averaged Bayesian t-tests provide robust inference when uncertain about prior specification by averaging over models with different priors."**

### ARR-COC-0-1 Application: Comparing Relevance Frameworks

**Research question**: Which cognitive framework best explains token allocation?

**Competing models**:
1. **M₁ (Information-theoretic)**: Allocate based on Shannon entropy alone
2. **M₂ (Salience-based)**: Allocate based on visual salience maps
3. **M₃ (Vervaekean)**: Full 3P framework (Propositional, Perspectival, Participatory)

**Model comparison**:
```python
# Compute marginal likelihoods
ml_1 = compute_marginal_likelihood(M1, fixation_data)
ml_2 = compute_marginal_likelihood(M2, fixation_data)
ml_3 = compute_marginal_likelihood(M3, fixation_data)

# Posterior model probabilities (equal priors)
p_1 = ml_1 / (ml_1 + ml_2 + ml_3)
p_2 = ml_2 / (ml_1 + ml_2 + ml_3)
p_3 = ml_3 / (ml_1 + ml_2 + ml_3)

# Output:
# P(M₁ | Data) = 0.08
# P(M₂ | Data) = 0.22
# P(M₃ | Data) = 0.70

# Conclusion: Strong evidence for Vervaekean framework
#             (70% probability it's the best model)
```

## 7. Practical Implementation Considerations

### Computational Methods for Bayesian Inference

**Analytical solutions** (rare):
- Conjugate priors only
- Simple models (Normal-Normal, Beta-Binomial, etc.)

**Markov Chain Monte Carlo (MCMC)**:
- Stan (HMC/NUTS - state of the art)
- PyMC (Python, user-friendly)
- JAGS (older, still used)

**Variational Inference** (approximate):
- Faster than MCMC
- Trade accuracy for speed
- Good for large datasets

**Example - Stan for ARR-COC relevance model**:
```stan
data {
  int<lower=0> N;           // Number of patches
  vector[N] relevance;      // Observed relevance scores
  vector[N] entropy;        // Shannon entropy
  vector[N] salience;       // NeVa salience
  vector[N] coupling;       // Cross-attention
}

parameters {
  real<lower=0> w_prop;     // Propositional weight
  real<lower=0> w_persp;    // Perspectival weight
  real<lower=0> w_part;     // Participatory weight
  real<lower=0> sigma;      // Noise
}

model {
  // Priors
  w_prop ~ exponential(1);
  w_persp ~ exponential(1);
  w_part ~ exponential(1);
  sigma ~ exponential(1);

  // Likelihood
  relevance ~ normal(w_prop * entropy +
                     w_persp * salience +
                     w_part * coupling,
                     sigma);
}

generated quantities {
  // Posterior predictive checks
  vector[N] relevance_rep;
  for (n in 1:N) {
    relevance_rep[n] = normal_rng(w_prop * entropy[n] +
                                   w_persp * salience[n] +
                                   w_part * coupling[n],
                                   sigma);
  }
}
```

### Diagnosing MCMC Convergence

**Essential diagnostics**:

1. **R-hat** (Gelman-Rubin diagnostic):
   - Compares within-chain and between-chain variance
   - R-hat < 1.01 indicates convergence
   - R-hat > 1.1 → Poor convergence, run longer

2. **Effective sample size (ESS)**:
   - Accounts for autocorrelation in MCMC chains
   - Want ESS > 400 for stable estimates
   - ESS < 100 → Increase iterations

3. **Trace plots**:
   - Visual inspection of chain mixing
   - Should look like "hairy caterpillar"
   - Trends or stuck chains → Problem

4. **Posterior predictive checks**:
   - Generate data from posterior
   - Compare to observed data
   - Large discrepancies → Model misspecification

```python
import arviz as az

# Fit model
trace = pm.sample(2000, tune=1000)

# Diagnostics
print(az.summary(trace))  # R-hat, ESS for all parameters
az.plot_trace(trace)      # Trace plots
az.plot_posterior(trace)  # Posterior distributions

# Posterior predictive check
ppc = pm.sample_posterior_predictive(trace)
az.plot_ppc(ppc)
```

### Reporting Bayesian Results

From [Tutorial on Bayesian t-Test (2025)](https://onlinelibrary.wiley.com/doi/10.1111/jan.70122):

**Essential elements**:
1. **Prior specification**: Justify choice, report sensitivity
2. **Posterior summary**: Mean, median, 95% credible interval
3. **Bayes factor**: Quantify evidence strength
4. **Model diagnostics**: R-hat, ESS, convergence checks
5. **Posterior predictive checks**: Assess model adequacy

**Example report**:
```
"We compared query-aware vs uniform token allocation using
a Bayesian independent samples t-test. We specified a Cauchy(0, 0.707)
prior on the standardized effect size (medium effect prior).

The posterior distribution for the effect size had mean δ = 0.52
(95% credible interval: [0.18, 0.87]), indicating query-aware
allocation improved FID scores by approximately half a standard
deviation. The Bayes factor BF₁₀ = 8.3 provides moderate evidence
for a positive effect.

All MCMC chains converged (R-hat < 1.01, ESS > 1000). Posterior
predictive checks showed good model fit to observed data."
```

## 8. Integration with Distributed Computing and VLM Engineering

### Tensor Parallelism for Bayesian MCMC (File 3: Megatron Tensor Parallel)

**Challenge**: Hierarchical Bayesian models with thousands of parameters (e.g., multi-query relevance model across 10,000 images).

**Solution**: Parallelize MCMC sampling across GPUs.

**Implementation**:
```python
# PyMC with GPU acceleration via JAX
import pymc as pm
import jax

with pm.Model() as hierarchical_model:
    # Define large hierarchical structure
    # ...

    # Use NUTS sampler with JAX backend
    trace = pm.sample(
        2000,
        tune=1000,
        chains=4,
        cores=4,
        nuts_sampler='numpyro',  # JAX-based sampler
        target_accept=0.95
    )
```

**Tensor parallelism benefit**: Split large parameter tensors (e.g., query-specific relevance parameters) across multiple GPUs, enabling Bayesian inference on models too large for single GPU memory.

### Ray for Distributed Bayesian Model Selection (File 11: Ray Distributed ML)

**Challenge**: Compare 100+ candidate relevance models via Bayes factors.

**Solution**: Parallelize marginal likelihood computation across Ray cluster.

**Implementation**:
```python
import ray
import pymc as pm

@ray.remote
def compute_marginal_likelihood(model_config, data):
    """Compute P(Data | Model) for single model."""
    with pm.Model() as model:
        # Build model from config
        build_model(model_config)

        # Compute marginal likelihood via bridge sampling
        trace = pm.sample(2000, return_inferencedata=False)
        ml = pm.compute_marginal_likelihood(trace)

    return ml

# Distribute across cluster
model_configs = generate_100_model_variants()
futures = [compute_marginal_likelihood.remote(config, data)
           for config in model_configs]

marginal_likelihoods = ray.get(futures)

# Compute posterior model probabilities
posterior_probs = softmax(marginal_likelihoods)
best_model = model_configs[np.argmax(posterior_probs)]
```

**Ray benefit**: Scale Bayesian model comparison to hundreds of competing hypotheses, exploring model space efficiently.

### Intel oneAPI for Bayesian Inference on CPU (File 15: Intel oneAPI ML)

**Use case**: Bayesian inference on non-GPU hardware (e.g., edge devices, cost-sensitive deployments).

**Implementation**:
```python
# PyMC with Intel's oneMKL backend
import os
os.environ['MKL_NUM_THREADS'] = '32'

import pymc as pm
import theano.tensor as tt

theano.config.blas.ldflags = '-lmkl_rt'  # Use Intel MKL

with pm.Model() as model:
    # Define Bayesian model
    # ...

    # Sample using optimized BLAS operations
    trace = pm.sample(2000, cores=32)  # Leverage all CPU cores
```

**oneAPI benefit**: Accelerate matrix operations in MCMC (e.g., Cholesky decompositions, matrix inversions) using Intel's optimized libraries, making CPU-based Bayesian inference competitive.

## 9. ARR-COC-0-1 Bayesian Workflow

### End-to-End Bayesian Analysis Pipeline

**Research goal**: Validate relevance realization framework using Bayesian methods.

**Step 1: Prior Elicitation**
```python
# Specify priors for relevance scorer weights
priors = {
    'w_propositional': pm.Exponential('w_prop', 1),
    'w_perspectival': pm.Exponential('w_persp', 1),
    'w_participatory': pm.Exponential('w_part', 1)
}

# Justification: Exponential(1) is weakly informative
# - Favors smaller weights (regularization)
# - Allows large weights if data supports
# - Mean = 1, which is reasonable scale for normalized scores
```

**Step 2: Model Specification**
```python
with pm.Model() as relevance_model:
    # Priors
    w_prop = pm.Exponential('w_prop', 1)
    w_persp = pm.Exponential('w_persp', 1)
    w_part = pm.Exponential('w_part', 1)
    sigma = pm.Exponential('sigma', 1)

    # Likelihood
    predicted_relevance = (w_prop * propositional_scores +
                          w_persp * perspectival_scores +
                          w_part * participatory_scores)

    observed_fixations = pm.Normal('obs',
                                   mu=predicted_relevance,
                                   sigma=sigma,
                                   observed=fixation_data)
```

**Step 3: MCMC Sampling**
```python
with relevance_model:
    trace = pm.sample(2000, tune=1000, chains=4,
                     target_accept=0.95,
                     return_inferencedata=True)
```

**Step 4: Convergence Diagnostics**
```python
print(az.summary(trace, round_to=3))
# Check R-hat < 1.01, ESS > 400 for all parameters
```

**Step 5: Posterior Analysis**
```python
# Extract posterior means
w_prop_mean = trace.posterior['w_prop'].mean()
w_persp_mean = trace.posterior['w_persp'].mean()
w_part_mean = trace.posterior['w_part'].mean()

# 95% credible intervals
print(az.hdi(trace, hdi_prob=0.95))

# Posterior predictive check
ppc = pm.sample_posterior_predictive(trace, model=relevance_model)
az.plot_ppc(ppc)
```

**Step 6: Model Comparison**
```python
# Compare full model vs reduced models
models = {
    'Full': relevance_model,
    'Prop_only': propositional_only_model,
    'Persp_only': perspectival_only_model,
    'Part_only': participatory_only_model
}

comparison = az.compare(models, ic='loo')  # LOO-CV
print(comparison)

# Interpret:
# - Lower LOO = better predictive performance
# - dLOO > 4: Substantial evidence for best model
# - Weight: Posterior model probability
```

**Step 7: Sensitivity Analysis**
```python
# Re-run with different priors
priors_skeptical = {'lam': 2}  # Exponential(2) - more regularization
priors_optimistic = {'lam': 0.5}  # Exponential(0.5) - less regularization

trace_skeptical = fit_model(priors_skeptical)
trace_optimistic = fit_model(priors_optimistic)

# Compare posterior weight estimates
compare_posteriors([trace, trace_skeptical, trace_optimistic])
```

**Step 8: Report Results**
```
"Bayesian analysis (N=500 images, 1200 fixations) revealed that
all three ways of knowing contribute to relevance realization:

Propositional (Shannon entropy): w = 0.34 [0.22, 0.47]
Perspectival (NeVa salience):    w = 0.41 [0.28, 0.55]
Participatory (Cross-attention): w = 0.51 [0.37, 0.66]

The full model substantially outperformed single-scorer models
(ΔLOO > 15 for all comparisons, BF > 100), providing strong
evidence for Vervaeke's multi-dimensional relevance framework.

Results were robust to prior specification (credible intervals
overlapped >90% across three tested priors)."
```

## 10. Limitations and Best Practices

### When Bayesian Methods Excel

**Advantages**:
- Small sample sizes with informative priors
- Need for direct probability statements
- Hierarchical/multi-level data structures
- Sequential data collection (updating posteriors)
- Quantifying evidence FOR null hypothesis

### When to Exercise Caution

**Challenges**:
- Computational cost (MCMC can be slow)
- Prior specification (subjective, requires justification)
- Model complexity (easy to overfit with flexible priors)
- Interpretation (requires understanding of Bayesian philosophy)

From [Campbell et al. (2024) - Point-Null Models](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Defining-a-Credible-Interval-Is-Not-Always-Possible-with-Point/10.1214/23-BA1397.full):

**"When using model-averaged posteriors with a 'point-null' model (e.g., H₀: δ = 0 exactly), Bayesian credible intervals may be undefined, unlike frequentist confidence intervals."**

**Problem**: Mixing continuous and discrete parameter spaces creates non-standard posteriors.

**Solution**: Use Bayes factors for point hypotheses, not credible intervals.

### Best Practices

1. **Always report priors**: Transparency is essential for reproducibility
2. **Conduct sensitivity analysis**: Test robustness to prior choices
3. **Check convergence**: R-hat, ESS, trace plots are mandatory
4. **Posterior predictive checks**: Validate model adequacy
5. **Avoid optional stopping issues**: Pre-register sample size or use sequential Bayes factors with correction
6. **Prefer weakly informative priors**: Unless strong domain knowledge justifies informative priors
7. **Report uncertainty**: Credible intervals, not just point estimates
8. **Use model comparison**: Compare multiple hypotheses, not just test one

From [Heuts et al. (2025) - Statistical Primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12036961/):

**"The Bayesian framework combines prior beliefs and currently obtained data (the likelihood), resulting in updated beliefs, also known as posterior distributions. This provides a natural way to incorporate prior knowledge and quantify uncertainty."**

## Summary: Bayesian Methods for ARR-COC-0-1

Bayesian statistical methods provide powerful tools for validating and optimizing relevance realization in VLMs:

1. **Bayesian t-tests**: Compare allocation strategies with direct probability statements
2. **Bayes factors**: Quantify evidence for competing relevance frameworks (M₁, M₂, M₃)
3. **Credible intervals**: Quantify uncertainty in optimal token budgets and threshold parameters
4. **Hierarchical models**: Pool information across query types while respecting individual differences
5. **Model comparison**: Evaluate Vervaekean 3P framework against alternatives
6. **Sensitivity analysis**: Ensure robustness of conclusions to prior specification

**Integration with distributed computing** (Files 3, 11, 15) enables scalable Bayesian inference for large VLM experiments, parallelizing MCMC across GPUs (tensor parallel), distributing model comparison across clusters (Ray), and accelerating CPU inference (Intel oneAPI).

**Key advantage**: Bayesian methods provide the most direct answers to scientific questions - "What is the probability that query-aware allocation improves quality?" rather than "Can we reject the hypothesis of no improvement?"

## Sources

**Existing Knowledge**:
- [06-bayesian-inference-deep.md](06-bayesian-inference-deep.md) - Bayesian inference foundations (Bayes' theorem, priors, posteriors)

**Web Research** (accessed 2025-11-16):
- [Model-averaged Bayesian t tests (Maier et al., 2024)](https://link.springer.com/article/10.3758/s13423-024-02590-5) - Springer
- [Tutorial on Bayesian t-Test (2025)](https://onlinelibrary.wiley.com/doi/10.1111/jan.70122) - Wiley Online Library
- [Empirical Bayes Factors (Dudbridge, 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297874) - PLOS ONE
- [Diagnosing Misuse of Bayes Factor (Tendeiro & Kiers, 2024)](https://journals.sagepub.com/doi/10.1177/25152459231213371) - Sage Journals
- [Credible vs Confidence Intervals (Statsig, 2024)](https://www.statsig.com/perspectives/credible-vs-confidence-intervals)
- [Using HDI (Edwards, 2025)](https://www.sciencedirect.com/science/article/pii/S0165783625000633) - ScienceDirect
- [Hierarchical Bayesian Modeling (Veenman et al., 2024)](https://link.springer.com/article/10.3758/s13428-023-02204-3) - Springer
- [Prior Sensitivity Analysis (Sekulovski et al., 2024)](https://advances.in/psychology/10.56296/aip00016/)
- [Hierarchical Bayesian RL (van Geen et al., 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0022249621000742) - ScienceDirect
- [Hierarchical Bayesian Size Spectra (Wesner, 2024)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14312) - BES Journals
- [Hierarchical Bayesian Contrast Sensitivity (Zhao et al., 2024)](https://tvst.arvojournals.org/article.aspx?articleid=2802354) - ARVO Journals
- [Deep Learning for BMC (Elsemüller et al., 2024)](https://psycnet.apa.org/record/2024-80637-001) - APA PsycNet
- [Point-Null Models (Campbell et al., 2024)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Defining-a-Credible-Interval-Is-Not-Always-Possible-with-Point/10.1214/23-BA1397.full) - Project Euclid
- [Statistical Primer (Heuts et al., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12036961/) - NIH PMC

**Influential Files** (Future expansion):
- File 3: distributed-training/02-megatron-lm-tensor-parallelism.md - Tensor parallel for large Bayesian models
- File 11: orchestration/02-ray-distributed-ml.md - Distributed Bayesian model comparison
- File 15: alternative-hardware/02-intel-oneapi-ml.md - CPU-accelerated Bayesian inference

**ARR-COC-0-1 Integration**:
- Bayesian model comparison for relevance frameworks
- Credible intervals for optimal token budgets
- Hierarchical modeling for multi-query experiments
- Prior sensitivity for threshold parameters
