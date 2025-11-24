# Psychophysics & Detection Theory: Computational Foundations of Perceptual Measurement

## Overview

Psychophysics is the quantitative science of relating physical stimuli to perceptual experience - the bridge between objective measurement and subjective sensation. Founded by Gustav Fechner in 1860, psychophysics provides the mathematical and experimental tools for measuring discrimination thresholds, sensitivity limits, and perceptual scaling functions. Detection theory extends this framework by separating true sensory sensitivity from response bias, enabling precise characterization of perceptual decision-making.

**Core Insight**: Perception is not a passive readout of physical input but an active inference process with measurable detection limits, discrimination thresholds, and systematic scaling relationships. Understanding these psychophysical laws is essential for validating computational vision models against human perception.

**Why This Matters for ARR-COC-0-1**: Token allocation policies must respect human psychophysical limits - just-noticeable differences in visual detail, detection thresholds for relevance, and perceptual scaling of compression quality. By grounding token budgets in Weber fractions and d' sensitivity measures, we ensure that model attention allocation produces perceptually meaningful differences.

From [A unified framework for perceived magnitude and discrimination](https://www.pnas.org/doi/10.1073/pnas.2312293121) (PNAS 2024, accessed 2025-11-16):
> "Weber's law of perceptual sensitivity can coexist with Stevens' power-law scaling of intensity ratings (for all exponents), resolving a 67-year paradox in psychophysics."

---

## Section 1: Weber's Law - The Constant Relative Threshold

### 1.1 Weber's Law Formulation

**Fundamental Principle**: The just-noticeable difference (JND) is proportional to stimulus magnitude.

**Mathematical Form**:
```
ΔI / I = k (Weber fraction)

Where:
- ΔI = Just-noticeable difference
- I = Standard stimulus intensity
- k = Weber fraction (constant for each modality)
```

**Example - Visual Brightness**:
```
If k = 0.08 for brightness (8% Weber fraction)

Baseline: 100 cd/m² → JND = 8 cd/m² (need 108 cd/m² to detect change)
Baseline: 500 cd/m² → JND = 40 cd/m² (need 540 cd/m² to detect change)
Baseline: 1000 cd/m² → JND = 80 cd/m² (need 1080 cd/m² to detect change)

The absolute JND scales linearly with intensity.
```

From [Weber's Law as the emergent phenomenon of choices](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1532069/full) (Frontiers in Neuroscience 2025, accessed 2025-11-16):
> "Weber's Law is one of the best-documented phenomena in psychophysics... computational modeling reveals it emerges from optimal resource allocation in neural coding."

### 1.2 Weber Fractions Across Modalities

**Vision**:
- Brightness: k ≈ 0.08 (8%)
- Line length: k ≈ 0.03 (3%)
- Contrast: k ≈ 0.01-0.02 (1-2%, very precise!)
- Spatial frequency: k ≈ 0.05 (5%)

**Audition**:
- Loudness: k ≈ 0.10 (10%)
- Pitch (frequency): k ≈ 0.003 (0.3%, extremely precise!)

**Tactile/Proprioception**:
- Weight discrimination: k ≈ 0.02 (2%)
- Line length (haptic): k ≈ 0.07 (7%)

**Taste/Smell**:
- Salt concentration: k ≈ 0.08 (8%)
- Odor intensity: k ≈ 0.10-0.20 (10-20%, less precise)

**Key Pattern**: Fine spatial/temporal discriminations (pitch, contrast) have smaller Weber fractions (better precision) than intensity judgments (loudness, brightness).

### 1.3 Violations and Modifications

**Near Absolute Threshold**:
Weber's law breaks down for very weak stimuli:
```
Modified Weber's Law:
ΔI / (I + I₀) = k

Where I₀ accounts for threshold effects (near-threshold constant offset)
```

**At Very High Intensities**:
Weber fraction increases due to:
- Sensory saturation
- Neural response compression
- Protective mechanisms (pain, glare)

**Suprathreshold Region**:
Weber's law holds best in the mid-range of perceptual experience (10× to 1000× above threshold).

From [Bayes vs. Weber: how to break a law of psychophysics](https://www.biorxiv.org/content/10.1101/2024.08.08.607196v1.full-text) (bioRxiv 2024, accessed 2025-11-16):
> "A classic tenet of psychophysics due to Weber is that human perceptual judgments are more variable for larger magnitudes. We show this variability pattern emerges from Bayesian inference with magnitude-dependent priors."

---

## Section 2: Stevens' Power Law - Perceptual Scaling Functions

### 2.1 Stevens' Power Law Formulation

**Fundamental Principle**: Perceived magnitude grows as a power function of physical intensity.

**Mathematical Form**:
```
Ψ = k × I^n

Where:
- Ψ = Perceived magnitude
- I = Physical intensity
- k = Scaling constant (arbitrary units)
- n = Exponent (characteristic of each modality)
```

### 2.2 Exponents by Modality

**Compressive Exponents (n < 1)** - Perceived growth slower than physical:
```
Brightness: n ≈ 0.33 (cube root)
  → 8× luminance increase → 2× perceived brightness

Loudness: n ≈ 0.67 (approximately 2/3 power)
  → 4× sound pressure → 2.5× perceived loudness

Smell (odor intensity): n ≈ 0.55
  → Rapid saturation at high concentrations

Taste (saltiness): n ≈ 0.60
  → Diminishing returns at high concentrations
```

**Expansive Exponents (n > 1)** - Perceived growth faster than physical:
```
Electric shock: n ≈ 3.5
  → Small increases in current → large increases in pain

Pain (thermal): n ≈ 1.0-3.5 (varies by modality)
  → Protective function: amplifies dangerous stimuli

Heaviness (lifted weight): n ≈ 1.45
  → Perceptual exaggeration for motor control
```

**Linear Exponents (n ≈ 1)** - Rare cases:
```
Line length: n ≈ 1.0 (veridical perception)
Numerosity (small sets): n ≈ 1.0 (accurate counting)
```

From [Two Neuroscience Laws Governing How We Sense the World](https://www.simonsfoundation.org/2024/06/17/two-neuroscience-laws-governing-how-we-sense-the-world-finally-united-after-67-years/) (Simons Foundation 2024, accessed 2025-11-16):
> "A new theoretical framework describes our ability to both absolutely and relatively gauge the properties of sensory inputs, unifying Weber's and Stevens' laws after 67 years."

### 2.3 Functional Significance of Exponents

**Why Compressive (n < 1)?**
- **Dynamic range expansion**: Represent vast luminance range (10^6:1) in limited neural response
- **Noise reduction**: Compress high-intensity noise (less perceptual impact)
- **Adaptation**: Prevent saturation at common environmental levels

**Why Expansive (n > 1)?**
- **Warning amplification**: Pain must be highly salient for survival
- **Fine control**: Weight perception exaggerated for precise motor planning
- **Threat detection**: Rapid escalation signals danger

**Biological Implementation**:
```
Neural Response = k × (Stimulus)^n

Implemented via:
- Adaptation (compressive)
- Gain control (divisive normalization)
- Contrast normalization
- Population coding with nonlinear transfer functions
```

---

## Section 3: Signal Detection Theory (SDT) - Separating Sensitivity from Bias

### 3.1 The Fundamental Insight

**Classical Problem**:
Detection is probabilistic, influenced by both:
1. **Sensitivity** - True ability to discriminate signal from noise
2. **Criterion** - Willingness to say "yes" (response bias)

Classical threshold models conflate these factors.

**SDT Solution**:
Separate sensitivity (d') from criterion (c) using distributions:

```
Noise distribution: N(0, 1)
Signal + Noise distribution: N(d', 1)

d' = separation between distributions (sensitivity)
c = criterion location (bias)
```

### 3.2 Core SDT Metrics

**d-prime (d') - Sensitivity**:
```
d' = Z(Hit Rate) - Z(False Alarm Rate)

Where Z = inverse cumulative standard normal

Example:
Hits: 80/100 (0.80)
False Alarms: 20/100 (0.20)

d' = Z(0.80) - Z(0.20)
   = 0.84 - (-0.84)
   = 1.68

Interpretation: Signal distribution is 1.68 standard deviations above noise.
```

**Criterion (c) - Response Bias**:
```
c = -0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]

c = 0: Neutral (equal evidence required)
c > 0: Conservative (prefer saying "no")
c < 0: Liberal (prefer saying "yes")
```

**Beta (β) - Likelihood Ratio**:
```
β = exp(c × d')

β = 1: Neutral
β > 1: Conservative
β < 1: Liberal

Beta represents the likelihood ratio at the criterion.
```

From [Signal Detection Theory Michael S. Landy](https://www.cns.nyu.edu/~david/courses/perceptionGrad/Readings/Landy-SDTchapter2024.pdf) (NYU 2024, accessed 2025-11-16):
> "Signal detection theory (SDT) is the primary performance model and data-analysis method for sensory experiments in audition, vision, etc."

### 3.3 ROC Curves - Characterizing Performance

**Receiver Operating Characteristic (ROC)**:
- Plot Hit Rate vs False Alarm Rate
- Each point represents a different criterion setting
- Area under curve (AUC) = probability correct in 2AFC

**ROC Curve Properties**:
```
AUC = Φ(d'/√2)

Where Φ is cumulative standard normal

AUC = 0.5 → d' = 0 (chance performance)
AUC = 0.75 → d' ≈ 1.35
AUC = 0.90 → d' ≈ 2.56
AUC = 1.0 → d' = ∞ (perfect discrimination)
```

**Isosensitivity Curves**:
All points on same ROC curve have equal d', but different criteria.

**Applications**:
- Medical diagnosis (radiologist detecting tumors)
- Security screening (TSA threat detection)
- Relevance detection (ARR-COC: is this region query-relevant?)

---

## Section 4: Psychometric Functions - Threshold Estimation

### 4.1 The Psychometric Function

**Definition**: Probability of detection/discrimination as function of stimulus intensity.

**Standard Form (Cumulative Gaussian)**:
```
Ψ(x) = γ + (1 - γ - λ) × Φ((x - α) / β)

Where:
- α = Threshold (50% point, accounting for guess/lapse)
- β = Slope (inverse of discriminability)
- γ = Guess rate (0.5 for 2AFC)
- λ = Lapse rate (inattention, motor errors)
- Φ = Cumulative Gaussian
```

**Alternative Forms**:
- Weibull (asymmetric, common in vision)
- Logistic (similar to Gaussian, easier computation)
- Quick (QUEST uses this)

### 4.2 Threshold Definitions

**Absolute Threshold**:
- Minimum stimulus intensity detectable 50% of time
- Depends on: modality, attention, adaptation, criterion

**Differential Threshold (JND)**:
- Minimum intensity difference detectable 75% of time (2AFC)
- Related to Weber fraction: JND = k × I

**Contrast Sensitivity Function (CSF)**:
- Inverse of contrast threshold vs spatial frequency
- Peaks at 3-5 cycles/degree for human vision
- Bandpass characteristic (low-pass + high-pass attenuation)

From [Psychophysical methods, or how to measure a threshold](https://academic.oup.com/book/9576/chapter/156603220) (Oxford 2024, accessed 2025-11-16):
> "Psychophysical methods are usually described in a historical context, starting with Weber, Fechner, and Stevens, but modern adaptive methods dramatically improve efficiency."

### 4.3 Adaptive Threshold Estimation

**QUEST (Quick Estimation by Sequential Testing)**:
```
Bayesian adaptive method:
1. Maintain posterior P(θ|data) for threshold θ
2. Select stimulus maximizing expected information gain
3. Update posterior after each response
4. Converges in 20-40 trials (vs 200+ for constant stimuli)
```

**Psi Method**:
- Estimates full psychometric function (threshold, slope, lapse)
- Uses entropy minimization
- Theoretically optimal efficiency

**2-Down-1-Up Staircase**:
- Decrease intensity after 2 correct, increase after 1 incorrect
- Converges to 70.7% correct point
- Simple, robust, widely used

---

## Section 5: Distributed Training Infrastructure for Psychophysical Experiments

**From [FSDP vs DeepSpeed](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - File 4 Influence**:

### 5.1 Scaling Psychophysical Model Training

**Why FSDP for Perceptual Models?**

Large-scale psychophysical experiments generate massive datasets:
- 30 subjects × 200 trials × 100 conditions = 600,000 data points
- Hierarchical Bayesian models with millions of parameters
- Full psychometric function estimation across stimulus space

**FSDP FULL_SHARD for Large Perceptual Models**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

# Hierarchical Bayesian psychometric model (5B parameters)
psychometric_model = HierarchicalBayesianPsycho(
    n_subjects=30,
    n_conditions=100,
    latent_dims=512
)

# FSDP sharding across 8 GPUs
fsdp_model = FullyShardedDataParallel(
    psychometric_model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    cpu_offload=CPUOffload(offload_params=True),  # Huge parameter space
)

# Train on psychophysical data
for epoch in range(n_epochs):
    for batch in psycho_dataloader:
        # Batch: stimulus intensities, subject responses, conditions
        logits = fsdp_model(batch['stimuli'])
        loss = psychometric_loss(logits, batch['responses'], batch['subjects'])
        loss.backward()
        optimizer.step()
```

**Memory Savings**:
```
Without FSDP: 5B params × 4 bytes × 4 (params+grads+optim) = 80GB per GPU
With FSDP (8 GPUs): 80GB / 8 = 10GB per GPU

Enables fitting large hierarchical models on modest clusters.
```

**From File 12 (ML Workload Patterns)**:

Production psychophysical pipelines require orchestration:
```yaml
# Kubernetes CronJob for nightly psychometric fitting
apiVersion: batch/v1
kind: CronJob
metadata:
  name: psychometric-fitting
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: fsdp-psycho
            image: psycho-fsdp:latest
            resources:
              limits:
                nvidia.com/gpu: 8  # Multi-GPU FSDP
            command: ["python", "fit_hierarchical_bayesian.py"]
            args: ["--data", "/data/today.csv", "--sharding", "FULL_SHARD"]
```

---

## Section 6: TPU-Optimized Psychometric Function Fitting

**From [TPU Programming Fundamentals](../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md) - File 16 Influence**:

### 6.1 Why TPUs for Psychophysics?

**Matrix Multiplication = Core Operation**:
```
Psychometric function fitting via maximum likelihood:

Gradient computation:
∇L = Σᵢ (yᵢ - Ψ(xᵢ|θ)) × ∇Ψ(xᵢ|θ)

Where Ψ Jacobian is dense matrix multiply:
∇Ψ = J × θ  (N_trials × N_params matrix-vector product)

For hierarchical models: massive Hessian computations (N_params × N_params)
```

**TPU TensorCore Advantage**:
```
MXU performs bf16[128,128] @ bf16[128,128] → f32[128,128] every 8 cycles

Psychometric Hessian (1000×1000) fits perfectly:
- 8 MXU tiles (128×128 each)
- Completed in ~64 cycles on TPU v5e
- vs thousands of cycles on CPU

Speedup: 50-100× for Hessian-based optimization (Newton-Raphson, Laplace approximation)
```

**JAX Implementation**:
```python
import jax
import jax.numpy as jnp
from jax import jit, vmap

@jit  # TPU compilation
def psychometric_weibull(x, alpha, beta, gamma, lambda_):
    """Vectorized psychometric function"""
    return gamma + (1 - gamma - lambda_) * (1 - jnp.exp(-(x/alpha)**beta))

@jit
def neg_log_likelihood(params, stimuli, responses):
    """Batched across subjects and trials"""
    alpha, beta, gamma, lambda_ = params
    p_correct = vmap(psychometric_weibull, in_axes=(0, None, None, None, None))(
        stimuli, alpha, beta, gamma, lambda_
    )
    return -jnp.sum(responses * jnp.log(p_correct) + (1-responses) * jnp.log(1-p_correct))

# TPU-optimized fitting (compiled to XLA)
from jax.example_libraries import optimizers

opt_init, opt_update, get_params = optimizers.adam(learning_rate=0.01)
opt_state = opt_init(initial_params)

@jit
def step(i, opt_state, stimuli, responses):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(neg_log_likelihood)(params, stimuli, responses)
    return opt_update(i, grads, opt_state), loss

# Runs entirely on TPU TensorCore + HBM
for i in range(1000):
    opt_state, loss = step(i, opt_state, stimuli_batch, responses_batch)
```

**Hierarchical Bayesian Extension**:
```python
# Multi-subject model: N_subjects × N_params
# Compiled to efficient TPU matmuls

@jit
def hierarchical_posterior(subject_params, population_params, data):
    """Population-level priors + subject-level likelihoods"""
    # Population: Gaussian prior over subject params
    mu_pop, sigma_pop = population_params

    # Subject likelihoods (batched across all subjects)
    subject_ll = vmap(neg_log_likelihood, in_axes=(0, 0, 0))(
        subject_params, data['stimuli'], data['responses']
    )

    # Prior: subjects ~ N(mu_pop, sigma_pop)
    prior = -0.5 * jnp.sum(((subject_params - mu_pop) / sigma_pop)**2)

    return jnp.sum(subject_ll) - prior  # Negative log posterior

# TPU handles entire 30-subject × 1000-param model in single batch
```

---

## Section 7: Real-Time Psychophysical Inference on Ray

**From [Ray Distributed ML](../karpathy/orchestration/02-ray-distributed-ml.md) - File 12 Influence**:

### 7.1 Online Adaptive Testing with Ray

**Use Case**: Real-time adaptive psychophysics (QUEST) across many simultaneous subjects.

**Ray Actors for Stateful QUEST**:
```python
import ray

@ray.remote
class QUESTActor:
    """Stateful Bayesian adaptive threshold tracker per subject"""
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.posterior = jnp.ones(100) / 100  # Uniform prior over thresholds
        self.trial_history = []

    def select_next_stimulus(self):
        """Choose stimulus maximizing expected information gain"""
        # Entropy-minimization (Psi method)
        entropies = []
        for candidate_stim in jnp.linspace(-5, 5, 50):
            # Expected posterior entropy after observing response
            expected_H = self._compute_expected_entropy(candidate_stim, self.posterior)
            entropies.append(expected_H)

        # Return stimulus with minimum expected entropy
        return jnp.linspace(-5, 5, 50)[jnp.argmin(jnp.array(entropies))]

    def update_posterior(self, stimulus, response):
        """Bayesian update after observing response"""
        likelihood = self._likelihood(stimulus, response, self.posterior)
        self.posterior = likelihood * self.posterior
        self.posterior /= jnp.sum(self.posterior)  # Normalize
        self.trial_history.append((stimulus, response))

    def get_threshold_estimate(self):
        """Current best estimate (posterior mode or mean)"""
        return jnp.argmax(self.posterior)

# Deploy QUEST actors across Ray cluster (one per subject)
quest_actors = [QUESTActor.remote(subject_id=i) for i in range(30)]

# Parallel adaptive testing
@ray.remote
def run_adaptive_session(quest_actor, n_trials=40):
    """Run full adaptive session for one subject"""
    for trial in range(n_trials):
        # Actor selects optimal next stimulus
        stim = ray.get(quest_actor.select_next_stimulus.remote())

        # Present stimulus, get response (simulated here)
        response = simulate_subject_response(stim)  # 0 or 1

        # Update posterior
        quest_actor.update_posterior.remote(stim, response)

    # Return final threshold estimate
    return ray.get(quest_actor.get_threshold_estimate.remote())

# Run 30 subjects in parallel (each converges in ~40 trials)
threshold_estimates = ray.get([
    run_adaptive_session.remote(actor) for actor in quest_actors
])

# Aggregate population statistics
population_mean = jnp.mean(jnp.array(threshold_estimates))
population_std = jnp.std(jnp.array(threshold_estimates))
```

**Why Ray > Traditional Multiprocessing**:
- **Stateful actors**: Each QUEST maintains Bayesian posterior across trials
- **Dynamic scheduling**: Subjects finish at different rates (Ray auto-balances)
- **Fault tolerance**: If one subject crashes, others continue
- **Real-time monitoring**: Ray dashboard shows per-subject progress

---

## Section 8: ARR-COC-0-1 Psychophysical Validation (10%)

### 8.1 Weber's Law in Token Allocation

**Research Question**: Do human subjects perceive token allocation differences according to Weber's law?

**Experimental Design**:
```
Method: 2-Alternative Forced Choice (2AFC) discrimination

Stimuli: ARR-COC image patches with varying token budgets
- Reference: 200 tokens
- Comparison: 200 + ΔT tokens (adaptive staircase)

Task: "Which image shows more detail in the [query-relevant region]?"

Procedure:
1. Present query: "Where is the dog?"
2. Show two ARR-COC renderings side-by-side:
   - Left: 200 tokens to dog region
   - Right: 200 + ΔT tokens to dog region
3. Subject chooses image with better detail
4. 2-down-1-up staircase → converges to 70.7% correct threshold

Expected Result:
Weber fraction k ≈ 0.10-0.15 (10-15%)
→ At 200 tokens, JND ≈ 20-30 tokens
→ At 400 tokens, JND ≈ 40-60 tokens

Validates: Token allocation increments must exceed perceptual JND to be meaningful.
```

**FSDP-Trained Discrimination Model**:
```python
# Train perceptual discrimination network on human judgments
# Predict: "Which ARR-COC rendering has higher perceived quality?"

discrimination_net = nn.Sequential(
    FeatureExtractor(),  # Shared CNN
    SiameseComparator(),  # Compare two ARR-COC outputs
    nn.Linear(512, 1),  # Binary: left vs right better
    nn.Sigmoid()
)

# FSDP for large feature extractors
fsdp_discriminator = FullyShardedDataParallel(
    discrimination_net,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)

# Train on human 2AFC data (10,000 trials × 30 subjects)
for batch in human_2afc_dataloader:
    # Batch: (left_img, right_img, subject_choice)
    pred = fsdp_discriminator(batch['left'], batch['right'])
    loss = bce_loss(pred, batch['choice'])
    loss.backward()
```

**Ray-Distributed Human Experiment**:
```python
# Deploy web-based 2AFC experiment using Ray Serve

@serve.deployment
class AdaptiveStaircaseServer:
    def __init__(self):
        self.staircases = {}  # subject_id -> QUESTActor

    async def get_trial(self, subject_id):
        """Return next optimal stimulus pair"""
        if subject_id not in self.staircases:
            self.staircases[subject_id] = QUESTActor.remote(subject_id)

        # QUEST selects ΔT (token difference)
        delta_T = await self.staircases[subject_id].select_next_stimulus.remote()

        # Generate ARR-COC renderings (200 vs 200+ΔT tokens)
        ref_img = arr_coc_render(query, image, tokens=200)
        comp_img = arr_coc_render(query, image, tokens=200+delta_T)

        return {
            'reference': ref_img,
            'comparison': comp_img,
            'delta_T': delta_T
        }

    async def submit_response(self, subject_id, choice):
        """Update QUEST posterior"""
        await self.staircases[subject_id].update_posterior.remote(...)

# Deploy to Ray cluster (handles 100+ concurrent subjects)
serve.run(AdaptiveStaircaseServer.bind())
```

### 8.2 Signal Detection Analysis of Relevance Judgments

**Research Question**: Can humans reliably detect query-relevant vs irrelevant regions? What is their d' sensitivity?

**Experimental Design**:
```
Method: Yes/No detection task

Stimuli: Image regions (500ms flash)
- Signal trials (100): Regions with high ARR-COC attention (top 25%)
- Noise trials (100): Regions with low ARR-COC attention (bottom 25%)

Task: "Is this region relevant to the query?"

Responses: Yes (detected relevance) / No (not relevant)

Analysis:
Hit Rate = P("Yes" | Signal)
False Alarm Rate = P("Yes" | Noise)

d' = Z(Hit Rate) - Z(False Alarm Rate)
c = -0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]

Expected Result:
- High sensitivity: d' > 2.0 (humans reliably detect ARR-COC relevance)
- Neutral criterion: c ≈ 0 (balanced yes/no responding)

Validates: ARR-COC attention maps align with human relevance perception.
```

**TPU-Optimized SDT Analysis**:
```python
# Fit hierarchical SDT model to 30 subjects' data
# JAX + TPU for fast Hessian computation

@jit
def sdt_model(subject_params, trials):
    """Hierarchical SDT: d' and c vary by subject and condition"""
    d_prime, criterion = subject_params

    # Probability of "yes" response
    # P(yes|signal) = Φ(d'/2 - c)
    # P(yes|noise) = Φ(-d'/2 - c)

    signal_trials = trials['is_signal']
    p_yes = jnp.where(
        signal_trials,
        jax.scipy.stats.norm.cdf(d_prime/2 - criterion),  # Hit rate
        jax.scipy.stats.norm.cdf(-d_prime/2 - criterion)  # FA rate
    )

    return p_yes

@jit
def hierarchical_sdt_loss(all_subject_params, population_params, all_trials):
    """Population-level priors + subject likelihoods"""
    mu_d, sigma_d, mu_c, sigma_c = population_params

    # Batched across subjects
    log_likelihoods = vmap(lambda p, t: jnp.sum(
        jnp.log(sdt_model(p, t))
    ), in_axes=(0, 0))(all_subject_params, all_trials)

    # Priors: d' ~ N(mu_d, sigma_d), c ~ N(mu_c, sigma_c)
    prior_d = -0.5 * jnp.sum(((all_subject_params[:, 0] - mu_d) / sigma_d)**2)
    prior_c = -0.5 * jnp.sum(((all_subject_params[:, 1] - mu_c) / sigma_c)**2)

    return -jnp.sum(log_likelihoods) - prior_d - prior_c

# TPU computes full 30-subject × 200-trial model in milliseconds
# Hessian (for confidence intervals) via JAX autodiff
hessian_fn = jax.hessian(hierarchical_sdt_loss)
H = hessian_fn(all_params, population_params, all_data)  # 62×62 matrix (TPU MXU)
standard_errors = jnp.sqrt(jnp.diag(jnp.linalg.inv(H)))
```

### 8.3 Stevens' Power Law for Quality Ratings

**Research Question**: How does perceived ARR-COC quality scale with token budget?

**Experimental Design**:
```
Method: Magnitude estimation

Stimuli: VQA questions answered by ARR-COC at varying token budgets
- Budgets: 64, 128, 200, 300, 400 tokens

Task: "Rate the overall image understanding quality (0-100 scale)"

Analysis:
Fit Stevens' power law:
Quality = k × (Tokens)^n

Expected Result:
Compressive exponent: n ≈ 0.4-0.6
→ Diminishing returns above ~250-300 tokens
→ Doubling tokens from 100→200 gives larger gain than 200→400

Informs: Optimal token budget allocation policy (maximal perceptual quality per token)
```

**FSDP Quality Regression**:
```python
# Large transformer trained to predict human quality ratings from ARR-COC features

quality_net = nn.Sequential(
    CLIP_Encoder(),  # Extract ARR-COC visual features
    nn.TransformerEncoder(num_layers=12, d_model=768),  # Contextual integration
    nn.Linear(768, 1),  # Predict quality rating (0-100)
)

# FSDP sharding (large transformer)
fsdp_quality_model = FullyShardedDataParallel(
    quality_net,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)

# Train on human magnitude estimation data
# Dataset: 40 images × 5 budgets × 30 subjects = 6000 ratings

for batch in quality_ratings_dataloader:
    # Batch: ARR-COC rendering, token budget, subject rating
    features = fsdp_quality_model.encode(batch['arr_coc_image'])
    predicted_quality = fsdp_quality_model.predict(features, batch['token_budget'])

    # Loss: MSE on human ratings
    loss = mse_loss(predicted_quality, batch['human_rating'])
    loss.backward()

# Fit power law to aggregated predictions
# Quality = k × Tokens^n
```

**Power Law Validation**:
```python
import scipy.optimize

def power_law(tokens, k, n):
    return k * (tokens ** n)

# Aggregate human ratings across subjects
mean_quality = quality_ratings.groupby('token_budget').mean()
token_budgets = [64, 128, 200, 300, 400]

# Fit power law via least squares
params, cov = scipy.optimize.curve_fit(
    power_law,
    token_budgets,
    mean_quality,
    p0=[10, 0.5]  # Initial guess: k=10, n=0.5
)

k_fit, n_fit = params
print(f"Fitted: Quality = {k_fit:.2f} × Tokens^{n_fit:.3f}")

# Example output:
# Quality = 8.3 × Tokens^0.42
# → Compressive exponent confirms diminishing returns
```

---

## References & Sources

**Classical Psychophysics**:
1. [Weber's Law as emergent phenomenon](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1532069/full) - Frontiers Neuroscience 2025
2. [Bayes vs. Weber: breaking psychophysics laws](https://www.biorxiv.org/content/10.1101/2024.08.08.607196v1.full-text) - bioRxiv 2024
3. [Unified framework for Weber and Stevens](https://www.pnas.org/doi/10.1073/pnas.2312293121) - PNAS 2024
4. [Two laws united after 67 years](https://www.simonsfoundation.org/2024/06/17/two-neuroscience-laws-governing-how-we-sense-the-world-finally-united-after-67-years/) - Simons Foundation 2024

**Modern Psychophysical Methods**:
5. [Signal Detection Theory (SDT) chapter](https://www.cns.nyu.edu/~david/courses/perceptionGrad/Readings/Landy-SDTchapter2024.pdf) - NYU 2024
6. [Psychophysical methods and thresholds](https://academic.oup.com/book/9576/chapter/156603220) - Oxford University Press
7. [Machine learning for psychophysics simulation](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202405789) - Wiley 2025

**Computational Implementation**:
8. [FSDP vs DeepSpeed comparison](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - Distributed training for large psychometric models
9. [Ray Distributed ML](../karpathy/orchestration/02-ray-distributed-ml.md) - Online adaptive testing infrastructure
10. [TPU Programming Fundamentals](../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md) - Accelerated Hessian computation for Bayesian fitting

**ARR-COC Connection**:
11. [Psychophysics & Human Studies](../karpathy/research-methodology/01-psychophysics-human-studies.md) - Foundational methods for perceptual validation

All web sources accessed 2025-11-16.

---

**Integration Summary**: This cognitive-mastery treatment extends foundational psychophysics with computational depth - distributed FSDP training for hierarchical Bayesian models, TPU-accelerated Hessian computation, and Ray-based real-time adaptive testing. For ARR-COC-0-1, psychophysical validation ensures token allocation policies respect human Weber fractions (10-15% JNDs), produce high d' relevance detection (>2.0), and follow compressive quality scaling (n≈0.4-0.6), grounding computational vision in human perceptual limits.
