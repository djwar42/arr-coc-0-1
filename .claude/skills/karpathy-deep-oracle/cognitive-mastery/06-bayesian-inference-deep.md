# Bayesian Inference Deep Dive

## Overview

Bayesian inference is a principled framework for updating beliefs about the world using probability theory. Unlike frequentist statistics that treats parameters as fixed unknowns, Bayesian inference treats them as random variables with probability distributions. This enables quantifying uncertainty, incorporating prior knowledge, and making predictions that account for what we don't know.

**Core Philosophy**: Probability represents degree of belief, not just frequency. As we observe data, we update our beliefs using Bayes' theorem.

**Key Innovation for ARR-COC-0-1**: Relevance realization can be formalized as Bayesian inference - visual patches with high information gain (likelihood ratio) receive more tokens (computational resources).

## 1. Bayes' Theorem: The Foundation

### The Core Equation

```
P(θ|D) = P(D|θ) × P(θ) / P(D)

Where:
- P(θ|D): Posterior - updated belief about parameters θ after seeing data D
- P(D|θ): Likelihood - probability of observing data D given parameters θ
- P(θ): Prior - initial belief about parameters θ before seeing data
- P(D): Evidence (marginal likelihood) - normalizing constant
```

### Beyond the Formula

From [Loredo & Wolpert (arXiv:2406.18905, 2024)](https://arxiv.org/abs/2406.18905):

**"Bayesian inference uses all of probability theory, not just Bayes's theorem."**

Critical insight: Many scientific hypotheses are **composite hypotheses** where evidence depends on nuisance parameters. The law of total probability (marginalization) is equally fundamental:

```
P(D) = ∫ P(D|θ) P(θ) dθ
```

This integral is often **intractable** - the computational bottleneck of Bayesian inference.

### Likelihood vs Probability

**Crucial distinction**:
- **Probability**: Function of data D with fixed parameters θ, integrates to 1 over D
- **Likelihood**: Function of parameters θ with fixed data D, does NOT integrate to 1 over θ

The likelihood tells us how probable the observed data is under different parameter values, but it's not a probability distribution over parameters.

## 2. Prior Distributions: Encoding Initial Knowledge

### Types of Priors

**Informative Priors**: Strong beliefs based on domain knowledge
```
Example: Human visual acuity
Prior on foveal resolution: N(60 cpd, 5²) cycles per degree
Justified by decades of psychophysics research
```

**Weakly Informative Priors**: Regularize without dominating
```
Example: ARR-COC relevance scores
Prior on attention weights: Beta(2, 2)
Favors balanced allocation but allows data to dominate
```

**Non-informative Priors**: Maximum uncertainty (use cautiously)
```
Uniform prior: P(θ) = constant
Jeffreys prior: P(θ) ∝ √|I(θ)| (invariant to reparameterization)
```

### Prior Selection Principles

1. **Physical constraints**: Parameters must satisfy domain constraints (e.g., probabilities in [0,1])
2. **Scale invariance**: Use priors invariant to measurement units when appropriate
3. **Exchangeability**: If observations are exchangeable, prior should respect symmetry
4. **Empirical Bayes**: Estimate hyperparameters from data (controversial but practical)

## 3. Conjugate Priors: Analytical Tractability

### Definition

A prior P(θ) is **conjugate** to likelihood P(D|θ) if the posterior P(θ|D) belongs to the same distributional family as the prior.

From [Towards Data Science (2022)](https://towardsdatascience.com/bayesian-conjugate-priors-simply-explained-747218be0f70):

**"Conjugate priors allow us to compute the posterior with simple addition rather than intractable integrals."**

### Common Conjugate Pairs

**Beta-Binomial** (most common):
```
Prior: θ ~ Beta(α, β)
Likelihood: k successes in n trials ~ Binomial(n, θ)
Posterior: θ ~ Beta(α + k, β + (n - k))

Update rule: Just add successes to α, failures to β!
```

**Gamma-Poisson**:
```
Prior: λ ~ Gamma(α, β)
Likelihood: Events ~ Poisson(λ)
Posterior: λ ~ Gamma(α + Σxᵢ, β + n)
```

**Gaussian-Gaussian** (unknown mean, known variance):
```
Prior: μ ~ N(μ₀, σ₀²)
Likelihood: X ~ N(μ, σ²)
Posterior: μ ~ N(μₙ, σₙ²)

Where:
μₙ = (σ²μ₀ + nσ₀²x̄) / (σ² + nσ₀²)
σₙ² = (σ²σ₀²) / (σ² + nσ₀²)
```

**Dirichlet-Multinomial**:
```
Prior: θ ~ Dirichlet(α₁, ..., αₖ)
Likelihood: Counts ~ Multinomial(n, θ)
Posterior: θ ~ Dirichlet(α₁ + n₁, ..., αₖ + nₖ)
```

### Limitations

- Only work for specific likelihood-prior combinations
- Real-world problems rarely fit these exact forms
- But: Useful as building blocks in hierarchical models

## 4. Posterior Inference: Beyond Point Estimates

### Full Posterior Distribution

Don't just find maximum a posteriori (MAP) estimate θ̂ = argmax P(θ|D). **Use the entire distribution**:

```
Uncertainty quantification:
- Mean: E[θ|D] = ∫ θ P(θ|D) dθ
- Variance: Var[θ|D] = ∫ (θ - E[θ|D])² P(θ|D) dθ
- Credible intervals: P(θₗ < θ < θᵤ | D) = 0.95
```

### Posterior Predictive Distribution

Predict new data y_new by marginalizing over parameter uncertainty:

```
P(y_new | D) = ∫ P(y_new | θ) P(θ | D) dθ
```

This accounts for both:
1. **Aleatoric uncertainty**: Inherent data randomness
2. **Epistemic uncertainty**: Parameter uncertainty

**ARR-COC-0-1 Application**: When allocating tokens to a new image patch, marginalize over uncertainty in relevance scorer parameters - don't just use point estimates!

## 5. Markov Chain Monte Carlo (MCMC): Sampling Complex Posteriors

### The Problem

For most real models, the posterior P(θ|D) is:
- High-dimensional (thousands to millions of parameters)
- Non-conjugate (no analytical form)
- Intractable normalizing constant P(D)

**Solution**: Draw samples from the posterior distribution without computing P(D).

### Metropolis-Hastings Algorithm

```python
# Initialize
θ_current = initial_guess

for iteration in range(num_samples):
    # Propose new state
    θ_proposed = proposal_distribution(θ_current)

    # Compute acceptance ratio
    ratio = P(D|θ_proposed) * P(θ_proposed) / (P(D|θ_current) * P(θ_current))

    # Accept or reject
    if random() < min(1, ratio):
        θ_current = θ_proposed  # Accept

    samples.append(θ_current)
```

**Key insight**: The normalizing constant P(D) cancels in the ratio! We only need unnormalized posterior.

### Gibbs Sampling

For multivariate posteriors, sample each parameter conditionally:

```python
for iteration in range(num_samples):
    for i in range(num_params):
        # Sample θᵢ given all other parameters
        θ[i] = sample_from(P(θᵢ | θ₋ᵢ, D))
```

Works when conditional distributions are tractable (often conjugate).

### Hamiltonian Monte Carlo (HMC)

Uses gradient information to propose distant states efficiently:

```python
# Treat θ as position, introduce momentum p
for iteration in range(num_samples):
    p = sample_momentum()

    # Simulate Hamiltonian dynamics
    θ_proposed, p_proposed = leapfrog_integration(θ, p, gradients)

    # Metropolis accept/reject on joint (θ, p)
    if random() < acceptance_probability:
        θ = θ_proposed
```

**Modern implementation**: [Stan](https://mc-stan.org/) uses No-U-Turn Sampler (NUTS), an adaptive HMC variant.

### Convergence Diagnostics

From [Reddy & Fairbanks (arXiv:2405.11179, 2024)](https://arxiv.org/abs/2405.11179) on accelerating MCMC:

**Critical metrics**:
1. **R-hat (Gelman-Rubin)**: Compare variance within vs between chains, should be < 1.01
2. **Effective sample size (ESS)**: Accounting for autocorrelation, need ESS > 1000 per parameter
3. **Trace plots**: Visual inspection for mixing and stationarity
4. **Autocorrelation**: Should decay to zero within reasonable lag

**Multilevel MCMC**: Use coarse approximations (like ML surrogates) for cheap proposals, refine with high-fidelity model. Can achieve 2x speedup.

## 6. Variational Inference: Optimization Alternative to MCMC

### Core Idea

Instead of sampling from P(θ|D), find a simpler distribution Q(θ) that approximates it.

**Minimize KL divergence**:
```
Q*(θ) = argmin_Q KL(Q(θ) || P(θ|D))
      = argmin_Q ∫ Q(θ) log(Q(θ)/P(θ|D)) dθ
```

### Evidence Lower Bound (ELBO)

Since P(D) is intractable, maximize the ELBO instead:

```
ELBO(Q) = E_Q[log P(D, θ)] - E_Q[log Q(θ)]
        = E_Q[log P(D|θ)] - KL(Q(θ) || P(θ))

log P(D) = ELBO(Q) + KL(Q(θ) || P(θ|D))
```

Since KL ≥ 0, ELBO is a lower bound on log evidence. Maximizing ELBO minimizes KL to posterior.

### Mean-Field Variational Bayes

Assume parameters are independent:
```
Q(θ) = ∏ᵢ Qᵢ(θᵢ)
```

**Coordinate ascent algorithm**:
```python
for iteration in range(max_iter):
    for i in range(num_params):
        # Update factor i
        log Qᵢ(θᵢ) = E_{Q₋ᵢ}[log P(θ, D)] + const
```

### Pros and Cons

**Advantages**:
- Faster than MCMC (optimization vs sampling)
- Scales to massive datasets (stochastic gradients)
- Deterministic (no MCMC convergence issues)

**Disadvantages**:
- Underestimates uncertainty (mean-field assumption)
- Can get stuck in local optima
- No theoretical guarantees on approximation quality

**When to use**: Large-scale problems where MCMC is too slow, and you can validate approximation quality on held-out data.

## 7. Laplace Approximation: Gaussian Posterior Approximation

### Method

Approximate posterior with Gaussian centered at MAP estimate:

```
1. Find MAP: θ̂ = argmax P(θ|D)
2. Compute Hessian: H = -∇²log P(θ|D)|_{θ=θ̂}
3. Approximate: P(θ|D) ≈ N(θ̂, H⁻¹)
```

**Works when**: Posterior is unimodal and roughly symmetric (common with lots of data).

### Connection to Neural Networks

Neural network weights θ learned via maximum likelihood ≈ MAP with uniform prior.

**Laplace approximation for uncertainty**:
```
After training:
1. Compute Hessian H at optimum θ̂
2. Uncertainty: Cov(θ) ≈ H⁻¹
3. Predictive uncertainty: Var[f(x)] ≈ ∇f(x)ᵀ H⁻¹ ∇f(x)
```

Computationally cheap but assumes Gaussian posterior (often poor assumption for NNs).

## 8. Hierarchical Bayesian Models: Sharing Statistical Strength

### Structure

Parameters at multiple levels:
```
Data level: y_ij ~ P(y | θ_i)
Group level: θ_i ~ P(θ | φ)
Population level: φ ~ P(φ)
```

**Example**: Image patches in ARR-COC-0-1
```
Patch relevance: r_i ~ N(μ_category, σ²)
Category mean: μ_category ~ N(μ_global, τ²)
Global mean: μ_global ~ N(0, 10²)
```

### Benefits

1. **Partial pooling**: Share information across groups (vs no pooling or complete pooling)
2. **Regularization**: Shrinks group estimates toward population mean
3. **Handling small samples**: Borrow strength from other groups
4. **Uncertainty propagation**: Account for uncertainty at all levels

### Inference

**Full Bayesian**: Joint posterior P(θ₁, ..., θₙ, φ | D) via MCMC
**Empirical Bayes**: Estimate φ from data, then infer θ_i | φ, D

## 9. Model Comparison and Selection

### Bayes Factors

Compare models M₁ and M₂:
```
BF = P(D|M₁) / P(D|M₂)
   = ∫ P(D|θ₁,M₁)P(θ₁|M₁)dθ₁ / ∫ P(D|θ₂,M₂)P(θ₂|M₂)dθ₂
```

**Interpretation**:
- BF > 10: Strong evidence for M₁
- BF > 100: Decisive evidence for M₁

**Automatic Occam's razor**: Complex models penalized by integrating over larger parameter space.

### Information Criteria

**Watanabe-Akaike Information Criterion (WAIC)**:
```
WAIC = -2(lppd - p_WAIC)

Where:
lppd = Σᵢ log E[P(yᵢ|θ)] (log pointwise predictive density)
p_WAIC = Σᵢ Var[log P(yᵢ|θ)] (effective number of parameters)
```

Estimates out-of-sample predictive accuracy. Lower is better.

**Leave-One-Out Cross-Validation (LOO-CV)**:
```
LOO = Σᵢ log P(yᵢ | D₋ᵢ)
```

Can be approximated efficiently from MCMC samples using Pareto-smoothed importance sampling (PSIS-LOO).

## 10. Practical Implementation: Tensor Parallelism for Bayesian Inference

**Influenced by**: File 3 - Megatron-LM Tensor Parallelism

### Distributing Posterior Sampling

Large Bayesian models (e.g., Bayesian neural networks) require distributed inference.

**Tensor parallel MCMC**:
```python
# Partition parameter vector θ across GPUs
θ_local = partition_parameters(θ, rank)

for iteration in range(num_samples):
    # Each GPU proposes update to its partition
    θ_proposed_local = propose_update(θ_local)

    # All-reduce to compute global likelihood
    log_likelihood_local = compute_local_likelihood(θ_proposed_local)
    log_likelihood = all_reduce(log_likelihood_local, op=SUM)

    # Accept/reject (synchronized across GPUs)
    if accept(log_likelihood):
        θ_local = θ_proposed_local
```

**Key challenge**: Proposals must be synchronized to maintain detailed balance.

### Parallel Variational Inference

**Data parallelism for ELBO optimization**:
```python
# Each GPU processes mini-batch
for batch in data_loader:
    # Local ELBO gradient
    grad_local = compute_elbo_gradient(batch, Q)

    # Aggregate gradients
    grad = all_reduce(grad_local, op=MEAN)

    # Update variational parameters
    Q.update(grad)
```

Scales linearly with number of GPUs for large datasets.

## 11. Production Serving: Triton for Bayesian Inference

**Influenced by**: File 7 - Triton Inference Server

### Serving Bayesian Predictions

Real-time Bayesian inference requires serving posterior predictive distributions, not just point predictions.

**Triton ensemble for Bayesian inference**:
```
1. Input processor (preprocessing)
2. Posterior sampler (MCMC or VI model)
3. Predictive distribution (marginalize over samples)
4. Output formatter (mean, variance, quantiles)
```

**Example configuration**:
```python
# Triton model.pbtxt for Bayesian CNN
name: "bayesian_cnn_ensemble"
platform: "ensemble"

input: {name: "image", data_type: TYPE_FP32, dims: [3, 224, 224]}
output: {
    name: "mean_prediction", data_type: TYPE_FP32, dims: [1000]
    name: "epistemic_uncertainty", data_type: TYPE_FP32, dims: [1000]
    name: "aleatoric_uncertainty", data_type: TYPE_FP32, dims: [1000]
}

ensemble_scheduling {
    step {
        model_name: "posterior_sampler"
        model_version: -1
        input_map {key: "input_image", value: "image"}
        output_map {key: "weight_samples", value: "posterior_samples"}
    }
    step {
        model_name: "predictive_distribution"
        model_version: -1
        input_map {
            key: "image", value: "image"
            key: "samples", value: "posterior_samples"
        }
        output_map {
            key: "mean", value: "mean_prediction"
            key: "epistemic_var", value: "epistemic_uncertainty"
            key: "aleatoric_var", value: "aleatoric_uncertainty"
        }
    }
}
```

**Batching strategy**: Sample multiple posterior draws in parallel, marginalize via batch operations.

### Uncertainty-Aware Caching

```python
# Cache predictions with uncertainty estimates
cache_key = hash(input_image)
cached = cache.get(cache_key)

if cached and cached.epistemic_uncertainty < threshold:
    return cached.mean_prediction
else:
    # High uncertainty: recompute with more samples
    prediction = bayesian_inference(input_image, num_samples=100)
    cache.set(cache_key, prediction)
    return prediction
```

## 12. Alternative Hardware: Intel oneAPI for Bayesian Workloads

**Influenced by**: File 15 - Intel oneAPI ML

### Optimizing MCMC on CPUs

Many Bayesian models have irregular memory access patterns poorly suited to GPUs.

**Intel oneDNN for Bayesian neural networks**:
```cpp
// Optimize matrix operations in Hamiltonian Monte Carlo
#include <oneapi/dnnl/dnnl.hpp>

// Gradient computation for leapfrog integration
auto gradient_primitive = dnnl::inner_product_forward::primitive_desc(
    engine, prop_kind::forward_inference,
    memory_desc({batch_size, input_dim}, dt::f32, tag::nc),
    memory_desc({input_dim, output_dim}, dt::f32, tag::nc),
    memory_desc({batch_size, output_dim}, dt::f32, tag::nc)
);
```

**SYCL for parallel tempering**:
```cpp
// Run multiple chains at different temperatures in parallel
queue.submit([&](handler& h) {
    h.parallel_for(range<1>(num_chains), [=](id<1> chain_id) {
        double temperature = temperatures[chain_id];

        // MCMC step at this temperature
        auto proposal = propose_state(states[chain_id]);
        double acceptance = compute_acceptance(proposal, temperature);

        if (random() < acceptance) {
            states[chain_id] = proposal;
        }
    });
});
```

**When to use CPU**: Small to medium models with complex likelihoods, where MCMC step cost dominates batch size benefits.

## 13. ARR-COC-0-1: Bayesian Relevance Realization (10%)

### Formulation

**Relevance as posterior probability**:
```
P(relevant | patch, query) ∝ P(patch | relevant, query) × P(relevant | query)

Where:
- P(patch | relevant, query): Likelihood from visual features
- P(relevant | query): Prior from query semantics
- P(relevant | patch, query): Posterior relevance
```

### Three Ways of Knowing as Bayesian Components

**1. Propositional (Information Scorer)**:
```
Prior: Patches with high entropy get higher base relevance
P(relevant) ∝ H(patch) = -Σ p(x) log p(x)

Bayesian interpretation: Maximum entropy principle
```

**2. Perspectival (Salience Scorer)**:
```
Likelihood: Salient features predict relevance
P(patch | relevant) ∝ exp(salience_score(patch))

Bayesian interpretation: Exponential family likelihood
```

**3. Participatory (Query Scorer)**:
```
Prior update: Query modulates relevance prior
P(relevant | query) ∝ cross_attention(patch, query)

Bayesian interpretation: Empirical Bayes prior from query
```

### Token Allocation as Bayesian Decision Theory

**Utility-theoretic token allocation**:
```python
# For each patch i
expected_utility[i] = 0
for num_tokens in [64, 128, 256, 400]:
    # Expected information gain (Bayesian utility)
    info_gain = entropy(P(relevant | patch)) - entropy(P(relevant | patch, num_tokens))

    # Cost: Computational resources
    cost = num_tokens * compute_per_token

    # Expected utility
    utility = info_gain - cost
    expected_utility[i] += P(num_tokens | patch) * utility

# Allocate tokens to maximize total expected utility
allocation = argmax_allocation(expected_utility, total_budget=200)
```

**Bayesian updating during inference**:
```python
# Initial relevance prior
relevance_prior = combine_scorers(info, salience, query)

# As we process patches, update posterior
for patch in sorted_by_relevance:
    # Likelihood from patch features
    likelihood = visual_encoder(patch)

    # Posterior relevance (Bayes' theorem)
    relevance_posterior = (likelihood * relevance_prior) / evidence

    # Allocate tokens based on posterior uncertainty
    tokens = allocate_by_uncertainty(relevance_posterior)

    # Refined features reduce uncertainty
    refined_features = detailed_encoder(patch, tokens)

    # Update prior for next patch (Bayesian updating)
    relevance_prior = update_prior(relevance_posterior, refined_features)
```

### Hierarchical Bayesian Relevance Model

```
Token allocation: t_i ~ Categorical(π(r_i))
Patch relevance: r_i ~ N(μ_query, σ²_patch)
Query influence: μ_query ~ N(μ_global, τ²)
Global mean: μ_global ~ N(0.5, 0.1²)

Where:
- r_i: Relevance score for patch i
- μ_query: Query-specific mean relevance
- μ_global: Dataset average relevance
```

This hierarchical structure enables:
1. **Sharing information** across patches within an image
2. **Adaptation** to different query types
3. **Regularization** via shrinkage to global mean

### Uncertainty-Driven Exploration

**Epistemic uncertainty** (parameter uncertainty) vs **aleatoric uncertainty** (inherent randomness):

```python
# Measure epistemic uncertainty in relevance scores
relevance_samples = []
for _ in range(num_posterior_samples):
    scorer_params = sample_from_posterior()
    relevance = score_patch(patch, scorer_params)
    relevance_samples.append(relevance)

epistemic_uncertainty = np.var(relevance_samples)

# Allocate more tokens to uncertain patches
if epistemic_uncertainty > threshold:
    tokens = 400  # High resolution for uncertain regions
else:
    tokens = 64   # Low resolution for confident predictions
```

**Exploration-exploitation tradeoff**: Thompson sampling for patch selection
```python
# Sample relevance from posterior
sampled_relevance = sample_relevance_posterior(patch)

# Select patches with highest sampled relevance
selected_patches = argsort(sampled_relevance)[:K]

# Process selected patches (exploitation with stochastic exploration)
```

## Sources

**Web Research**:
- [Loredo, T. J., & Wolpert, R. L. (2024). Bayesian inference: More than Bayes's theorem. arXiv:2406.18905](https://arxiv.org/abs/2406.18905) - Marginalization and composite hypotheses (accessed 2025-11-16)
- [Howell, E. (2022). Bayesian Conjugate Priors Simply Explained. Towards Data Science](https://towardsdatascience.com/bayesian-conjugate-priors-simply-explained-747218be0f70) - Beta-Binomial conjugacy derivation (accessed 2025-11-16)
- [Reddy, S., & Fairbanks, H. (2024). Accelerating Multilevel Markov Chain Monte Carlo Using Machine Learning Models. arXiv:2405.11179](https://arxiv.org/abs/2405.11179) - MCMC acceleration techniques (accessed 2025-11-16)

**Influential Files**:
- **File 3**: [Megatron-LM Tensor Parallelism](../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md) - Distributed posterior sampling across GPUs
- **File 7**: [Triton Inference Server](../karpathy/inference-optimization/02-triton-inference-server.md) - Serving Bayesian predictions with uncertainty quantification
- **File 15**: [Intel oneAPI ML](../karpathy/alternative-hardware/02-intel-oneapi-ml.md) - CPU-optimized MCMC for irregular memory patterns

**ARR-COC-0-1 Implementation**:
- Relevance realization formalized as Bayesian inference
- Token allocation as Bayesian decision theory
- Uncertainty-driven exploration via epistemic uncertainty
- Hierarchical Bayesian model for relevance across patches

**Related Oracle Knowledge**:
- [Free Energy Principle](00-free-energy-principle-foundations.md) - Free energy as negative log evidence
- [Precision & Attention](01-precision-attention-resource.md) - Precision as inverse variance in Bayesian inference
- [Predictive Processing](../cognitive-foundations/01-predictive-processing-hierarchical.md) - Hierarchical Bayesian brain hypothesis

**Additional References**:
- [Stan Development Team. Stan Modeling Language User's Guide](https://mc-stan.org/) - Modern probabilistic programming
- [Gelman, A., et al. Bayesian Data Analysis (3rd ed.)](http://www.stat.columbia.edu/~gelman/book/) - Comprehensive textbook
- [Murphy, K. P. Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/) - Modern ML perspective on Bayesian methods
