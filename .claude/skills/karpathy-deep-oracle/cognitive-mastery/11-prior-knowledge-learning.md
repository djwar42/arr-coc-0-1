# Prior Knowledge & Learning

## Overview

Learning in Bayesian systems is fundamentally the process of updating prior beliefs with new evidence to form posterior distributions. This cycle of belief updating—where today's posterior becomes tomorrow's prior—provides a principled framework for accumulating knowledge over time. Hierarchical Bayesian models extend this by learning priors themselves from data, while empirical Bayes methods estimate prior distributions from observed data rather than specifying them subjectively.

**Core Principle**: Learning as iterative belief updating through Bayes' rule, with priors encoding accumulated knowledge and posteriors becoming new priors as evidence accumulates.

From [Bayesian Brain & Probabilistic Inference](02-bayesian-brain-probabilistic.md):
- Prior beliefs shape perception and guide learning
- Brain updates priors through experience (synaptic plasticity)
- Hierarchical priors across multiple timescales

---

## Section 1: Hierarchical Bayesian Models Fundamentals

### What are Hierarchical Models?

**Hierarchical structure** in Bayesian models reflects nested levels of uncertainty:

```
Hyperpriors (population level)
    ↓
Priors (group level)
    ↓
Parameters (individual level)
    ↓
Data (observation level)
```

From [Prior Knowledge Elicitation: The Past, Present, and Future](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Prior-Knowledge-Elicitation-The-Past-Present-and-Future/10.1214/23-BA1381.full) (Mikkola et al. 2024):

> "Prior elicitation transforms domain knowledge of various kinds into well-defined prior distributions, and offers a solution to the prior specification problem."

**Key Insight**: Group-level data informs individual-level estimates through partial pooling—neither complete pooling (ignore individual differences) nor no pooling (treat individuals as independent).

### Why Hierarchical Models?

From [Hierarchical Bayesian models of reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0022249621000742) (van Geen & Gerraty 2021):

**Benefits**:
1. **Improved parameter estimation**: Regularization from group-level priors
2. **Better generalization**: Out-of-sample prediction accuracy
3. **Data efficiency**: Share statistical strength across individuals
4. **Uncertainty quantification**: Explicit modeling of multiple uncertainty levels

**Example**: Learning rates in reinforcement learning
- Individual-level: Each subject has their own learning rate α_i
- Group-level: Learning rates drawn from population distribution N(μ_α, σ_α)
- Hyperprior: Population mean μ_α and variance σ_α have their own priors

### Partial Pooling

**Complete pooling** (ignore individual differences):
```
All subjects share single parameter: α = 0.3
```

**No pooling** (maximum likelihood per individual):
```
Subject 1: α_1 = 0.85 (overfit to noise)
Subject 2: α_2 = 0.12 (overfit to noise)
Subject 3: α_3 = 0.45 (overfit to noise)
```

**Partial pooling** (hierarchical Bayesian):
```
Subject 1: α_1 = 0.42 (shrunk toward group mean μ_α = 0.35)
Subject 2: α_2 = 0.28 (shrunk toward group mean)
Subject 3: α_3 = 0.37 (shrunk toward group mean)
```

**Shrinkage amount** depends on:
- Data quantity (more data → less shrinkage)
- Uncertainty (high individual uncertainty → more shrinkage)
- Population variance (small σ_α → more shrinkage)

---

## Section 2: Empirical Bayes Methods

### Definition

**Empirical Bayes**: Estimate hyperparameters of prior distribution from observed data, then use this "empirical prior" for inference.

**Two-stage process**:
1. Estimate hyperparameters θ from marginal distribution of data
2. Use p(parameter | θ_estimated) as prior in Bayesian inference

From [Empirical Bayes methods in labor economics](https://www.sciencedirect.com/science/article/abs/pii/S1573446324000014) (Walters 2024):

> "Empirical Bayes methods are closely connected to machine learning approaches to model selection and regularization."

### Empirical Bayes vs Fully Bayesian

**Empirical Bayes**:
- Point estimates of hyperparameters from data
- Computationally efficient
- Underestimates uncertainty (treats hyperparameters as known)
- Good approximation when data abundant

**Fully Bayesian (Hierarchical)**:
- Full posterior over hyperparameters
- Propagates uncertainty correctly
- Computationally intensive (MCMC, variational inference)
- Better for small datasets

### Applications

**1. James-Stein Estimator** (classic example):
- Shrink multiple parameter estimates toward common mean
- Reduces mean squared error vs. maximum likelihood
- Empirical prior from group statistics

**2. Reinforcement Learning**:
From van Geen & Gerraty (2021):
- Estimate learning rate distribution from one dataset
- Use as prior for new participants
- **Issue**: Requires large separate dataset to estimate priors
- **Hierarchical alternative**: Learn group prior simultaneously with individual parameters

**3. Neural Network Priors**:
- Estimate weight distribution from related tasks
- Transfer learning through empirical priors
- Meta-learning as learning priors

---

## Section 3: Prior Elicitation Techniques

### Expert Knowledge Elicitation

From Mikkola et al. (2024):

**Methods for eliciting priors from experts**:

**1. Direct specification**:
- Expert states prior distribution parameters
- **Challenge**: Experts don't think in statistical terms
- **Solution**: Translate domain knowledge into distributional form

**2. Simulation-based elicitation**:
- Expert specifies constraints on observable quantities
- Algorithm finds prior distribution matching constraints
- **Example**: "I expect 70% of patients to respond to treatment" → Beta prior on response probability

**3. Quantile-based elicitation**:
- Expert provides median, quartiles, extremes
- Fit parametric distribution to match quantiles
- More intuitive than specifying mean/variance directly

**4. Predictive elicitation**:
- Expert predicts outcomes from hypothetical data
- Infer prior that would produce these predictions
- Aligns with how experts naturally reason

### Automated Prior Learning

**Meta-learning approaches**:
- Learn prior from multiple related tasks
- Prior captures task family structure
- Enables few-shot learning on new tasks

**Neural processes**:
- Neural networks that output distributions over functions
- Trained on task distributions
- Produce data-efficient priors for new tasks

From [Modeling rapid language learning](https://www.nature.com/articles/s41467-025-59957-y) (McCoy et al. 2025):

> "We show that learning from limited naturalistic data is possible with an approach that bridges the divide between two popular modeling traditions: Bayesian models and neural networks."

**Key idea**: Distill Bayesian priors into neural network weights through meta-training.

---

## Section 4: Learning as Updating Priors

### Bayesian Update Cycle

**Sequential learning** = iterative prior updating:

```
Prior_t=0 → Observe data_1 → Posterior_t=1
         ↓
Prior_t=1 = Posterior_t=1 → Observe data_2 → Posterior_t=2
                          ↓
Prior_t=2 = Posterior_t=2 → Observe data_3 → Posterior_t=3
                          ...
```

**Mathematical form**:
```
p(θ | data_1:t) ∝ p(data_t | θ) × p(θ | data_1:t-1)
                   [likelihood]   [prior from previous update]
```

### Prior Informativeness Over Time

**Weak prior** (little data):
- High uncertainty → posterior dominated by likelihood
- New data has strong influence
- Exploration favored

**Strong prior** (much data):
- Low uncertainty → posterior dominated by prior
- New data has weak influence unless contradictory
- Exploitation favored

**Learning rate analogy**:
- Effective learning rate = precision_likelihood / (precision_prior + precision_likelihood)
- As data accumulates, effective learning rate decreases
- Matches cognitive literature: learning slows with experience

### Computational Models

From [Bayesian brain hypothesis](../cognitive-foundations/02-bayesian-brain-probabilistic.md):

**Synaptic plasticity as Bayesian update**:
- Synaptic weights encode posterior beliefs
- Hebbian learning = likelihood update
- Synaptic consolidation = prior strengthening
- Sleep replay = offline posterior refinement

**Prediction error = surprise**:
```
Prediction error = actual - expected
                 = log p(observation | prior)
```

High prediction error → weak prior → large update
Low prediction error → strong prior → small update

---

## Section 5: FSDP for Hierarchical Models (File 4)

From [FSDP vs DeepSeek](../../karpathy/distributed-training/03-fsdp-vs-deepspeed.md):

### Distributed Hierarchical Inference

**Challenge**: Hierarchical Bayesian models have nested parameter dependencies
- Group parameters depend on hyperparameters
- Individual parameters depend on group parameters
- Cannot parallelize naively

**FSDP solution**:
- Shard parameter hierarchy across GPUs
- Maintain synchronization through all-gather/reduce-scatter
- Enable large-scale hierarchical models (millions of individuals)

**Memory efficiency**:
```
Traditional: Each GPU stores full hierarchy
FSDP: Shard across GPUs, reconstruct on demand

Example: 1M individuals × 10 parameters
Traditional: 10M parameters per GPU
FSDP: 10M / N_gpu parameters per GPU
```

**Use case**: Population-scale medical trials
- Individual patient parameters (treatment response)
- Hospital-level parameters (practice patterns)
- Region-level hyperparameters (demographics)
- National-level hyperpriors (healthcare policy effects)

**Implementation pattern**:
```python
# Hierarchical model sharded across GPUs
hyperprior_shard = FSDP(HyperpriorModule())  # Top level
group_prior_shard = FSDP(GroupPriorModule(hyperprior_shard))  # Middle
individual_param_shard = FSDP(IndividualModule(group_prior_shard))  # Bottom

# Synchronize updates bottom-up
individual_loss.backward()  # Gradients flow up hierarchy
optimizer.step()  # Update all levels coherently
```

---

## Section 6: ML Workload Patterns for Bayesian Inference (File 12)

From [ML Workload Patterns on K8s](../../karpathy/orchestration/03-ml-workload-patterns-k8s.md):

### Orchestrating Hierarchical Bayesian Workflows

**Workload characteristics**:
- **Long-running**: MCMC chains run for hours/days
- **Stateful**: Chain states must persist across restarts
- **Parallel**: Multiple chains for convergence diagnostics
- **Resource-variable**: Warm-up vs. sampling vs. diagnostics

**K8s patterns for Bayesian inference**:

**1. Job for single inference run**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hierarchical-inference
spec:
  template:
    spec:
      containers:
      - name: stan-sampler
        image: mc-stan:latest
        command: ["sample"]
        args: ["--chains=4", "--parallel_chains=4"]
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
```

**2. CronJob for periodic prior updates**:
```yaml
# Update population priors daily as new data arrives
apiVersion: batch/v1
kind: CronJob
metadata:
  name: prior-update
spec:
  schedule: "0 0 * * *"  # Daily at midnight
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: empirical-bayes-updater
            command: ["python", "update_priors.py"]
```

**3. StatefulSet for persistent chains**:
```yaml
# Maintain chain state for incremental learning
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mcmc-chains
spec:
  serviceName: "mcmc"
  replicas: 4  # 4 independent chains
  template:
    spec:
      containers:
      - name: chain-sampler
        volumeMounts:
        - name: chain-state
          mountPath: /data/chain
  volumeClaimTemplates:
  - metadata:
      name: chain-state
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

**Monitoring & diagnostics**:
- Track R-hat (convergence) across chains
- Monitor effective sample size (ESS)
- Alert on divergent transitions
- Visualize trace plots in real-time

---

## Section 7: TPU Programming for Probabilistic Models (File 16)

From [TPU Programming Fundamentals](../../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md):

### TPU Advantages for Bayesian Computation

**Why TPUs for probabilistic inference?**

**1. Matrix operations**:
- Bayesian inference = matrix-heavy (covariance, precision matrices)
- TPU MXUs (Matrix Multiplication Units) excel at this
- 128×128 bfloat16 matrix multiply per clock cycle

**2. Variational inference on TPUs**:
```python
# Optimize ELBO (Evidence Lower Bound)
def elbo_loss(params, data):
    q_mean, q_logvar = encoder(data, params)  # Variational posterior
    z_samples = sample_gaussian(q_mean, q_logvar)  # Monte Carlo samples
    reconstruction = decoder(z_samples, params)

    # KL divergence (prior is N(0,I))
    kl = 0.5 * jnp.sum(q_mean**2 + jnp.exp(q_logvar) - q_logvar - 1)

    # Reconstruction likelihood
    log_likelihood = jnp.sum(log_prob(data, reconstruction))

    return -(log_likelihood - kl)  # Negative ELBO

# TPU-optimized with pmap
elbo_grad = jax.pmap(jax.grad(elbo_loss))  # Parallel across TPU cores
```

**3. Hamiltonian Monte Carlo (HMC)**:
- Requires gradient computations (TPUs excel)
- Symplectic integration (matrix operations)
- Batch parallel chains across TPU cores

**Example: Hierarchical model on TPU**:
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def hierarchical_model(data, N_groups, N_individuals):
    # Hyperpriors
    mu_global = numpyro.sample('mu_global', dist.Normal(0, 10))
    sigma_global = numpyro.sample('sigma_global', dist.HalfNormal(5))

    # Group-level priors
    with numpyro.plate('groups', N_groups):
        mu_group = numpyro.sample('mu_group',
                                   dist.Normal(mu_global, sigma_global))
        sigma_group = numpyro.sample('sigma_group', dist.HalfNormal(2))

    # Individual-level parameters
    with numpyro.plate('individuals', N_individuals):
        theta = numpyro.sample('theta',
                               dist.Normal(mu_group[group_id], sigma_group[group_id]))

    # Likelihood
    with numpyro.plate('observations', len(data)):
        numpyro.sample('obs', dist.Normal(theta[individual_id], 1), obs=data)

# Run on TPU with 8 cores
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

devices = mesh_utils.create_device_mesh((8,))
sharding = PositionalSharding(devices)

# Parallel MCMC chains
mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(hierarchical_model),
    num_warmup=1000,
    num_samples=5000,
    num_chains=8,  # One chain per TPU core
    chain_method='parallel'
)
```

**Performance gains**:
- 10-50x faster than CPU for medium-sized models (10K-100K parameters)
- Scales linearly with TPU cores for embarrassingly parallel chains
- bfloat16 precision sufficient for most Bayesian inference

---

## Section 8: ARR-COC-0-1: Relevance Realization as Hierarchical Bayesian Learning (10%)

### Token Allocation as Prior Learning

**ARR-COC-0-1 implements hierarchical Bayesian principles implicitly**:

**Observation**: Token budget allocation (64-400 tokens per patch) adapts based on relevance scores.

**Bayesian interpretation**:
- **Prior**: Base allocation of 64 tokens (uniform prior over patch importance)
- **Likelihood**: Relevance scores from knowing.py (propositional, perspectival, participatory)
- **Posterior**: Updated token allocation (64-400 range)

**Learning connection**:
```python
# From attending.py (simplified)
def allocate_tokens(relevance_scores, total_budget=200):
    # Prior: Uniform allocation
    base_allocation = total_budget // num_patches  # e.g., 64 tokens

    # Likelihood: Relevance-weighted adjustment
    relevance_weight = softmax(relevance_scores)  # Normalize to distribution

    # Posterior: Weighted allocation
    allocation = clip(base_allocation * relevance_weight * scale_factor,
                      min=64, max=400)
    return allocation
```

**Hierarchical structure in ARR-COC**:

**Level 1: Query-level (hyperprior)**:
- Query type influences expected relevance distribution
- Object queries → expect localized relevance
- Scene queries → expect distributed relevance

**Level 2: Patch-level (group prior)**:
- Relevance scores across patches
- Spatial correlations (neighboring patches similar relevance)
- Drawn from query-specific prior

**Level 3: Token-level (individual parameters)**:
- Each token within allocated budget
- Represents specific visual feature
- Precision determined by parent patch relevance

### Empirical Prior from Training Data

**ARR-COC-0-1 learns priors during training**:

**Phase 1: Initial training (weak priors)**:
- Model explores wide range of allocations
- High variance in relevance scores
- Large update steps (high learning rate analog)

**Phase 2: Refinement (strengthening priors)**:
- Empirical distribution of "good allocations" emerges
- Variance decreases (tighter priors)
- Smaller updates (lower effective learning rate)

**Phase 3: Fine-tuning (strong priors)**:
- Task-specific allocation patterns encoded
- Low variance (strong priors)
- Resistant to outlier queries (prior dominates)

**Connection to human vision**:
- Foveal-peripheral hierarchy = hierarchical prior over spatial resolution
- Attention = dynamic prior updating based on task demands
- Saccade planning = prediction of where high relevance likely

### Quality Adapter as Prior Elicitation

From ARR-COC-0-1 architecture:

**Quality Adapter** (Section 8 from adapter.py):
- Maps 400-dimensional high-quality embeddings → 64-dimensional low-quality
- Acts as **learned prior** on visual feature distributions
- Trained to preserve task-relevant information under compression

**Bayesian view**:
```
High-quality (400-dim) = Full posterior
Low-quality (64-dim) = Prior with reduced uncertainty
Quality Adapter = Marginalization operator (integrating out irrelevant dimensions)
```

**Prior strength calibration**:
- Adapter learns which dimensions are "uninformative" (high prior variance)
- Which dimensions are "diagnostic" (low prior variance, preserve in compression)
- This is empirical Bayes: learn prior structure from data

**Relevance Realization = Hierarchical Inference**:
```
Query + Image → Relevance Scores (likelihood)
                ↓
Opponent Processing → Navigate tensions (prior constraints)
                ↓
Token Allocation → Posterior predictive (action)
                ↓
Compressed Features → New data for next layer (updated prior)
```

**Learning loop**:
1. Current allocation = prior belief about patch importance
2. Query + visual features = likelihood of relevance
3. Balanced allocation = posterior
4. Feedback from downstream task = update hyperparameters
5. Repeat (accumulate knowledge about relevance patterns)

**Why this matters**:
- ARR-COC doesn't just "attend" (mechanistic)
- It **infers** relevance through hierarchical Bayesian updating (cognitive)
- Opponent processing = balancing competing prior constraints
- Transjective knowing = likelihood-prior interaction (neither alone sufficient)

---

## Sources

**Source Documents:**
- [cognitive-foundations/02-bayesian-brain-probabilistic.md](../../cognitive-foundations/02-bayesian-brain-probabilistic.md) - Bayesian brain hypothesis, prior-likelihood integration

**Influential Files:**
- [File 4: FSDP vs DeepSpeed](../../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - Distributed hierarchical models
- [File 12: ML Workload Patterns](../../karpathy/orchestration/03-ml-workload-patterns-k8s.md) - K8s patterns for Bayesian workflows
- [File 16: TPU Programming](../../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md) - Hardware acceleration for probabilistic inference

**Web Research (accessed 2025-11-16):**
- [Prior Knowledge Elicitation: The Past, Present, and Future](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Prior-Knowledge-Elicitation-The-Past-Present-and-Future/10.1214/23-BA1381.full) - Mikkola et al. 2024, comprehensive review of prior elicitation methods
- [Hierarchical Bayesian models of reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0022249621000742) - van Geen & Gerraty 2021, empirical Bayes vs hierarchical comparison
- [Empirical Bayes methods in labor economics](https://www.sciencedirect.com/science/article/abs/pii/S1573446324000014) - Walters 2024, connection to ML regularization
- [Modeling rapid language learning](https://www.nature.com/articles/s41467-025-59957-y) - McCoy et al. 2025, Bayesian priors distilled into neural networks
- [A Hierarchical Bayesian Model of Adaptive Teaching](https://onlinelibrary.wiley.com/doi/10.1111/cogs.13477) - Chen et al. 2024, pedagogy as hierarchical inference
- [Refining Bayesian hierarchical MPT modeling](https://link.springer.com/article/10.3758/s13428-024-02370-y) - Sarafoglou et al. 2024, best practices for hierarchical models

**ARR-COC-0-1 Implementation:**
- knowing.py - Relevance scorers (likelihood computation)
- balancing.py - Opponent processing (prior constraints)
- attending.py - Token allocation (posterior inference)
- adapter.py - Quality adapter (learned prior over feature importance)
