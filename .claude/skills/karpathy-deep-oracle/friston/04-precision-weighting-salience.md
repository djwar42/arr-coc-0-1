# Precision Weighting & Salience

## Overview

Precision weighting is a fundamental mechanism in predictive processing whereby the brain modulates the influence of prediction errors based on their estimated reliability. In Friston's free energy framework, precision represents the inverse variance of probability distributions - essentially a measure of confidence or certainty. Attention, in this view, is reconceptualized as the optimization of precision expectations, determining which prediction errors should drive belief updates and which should be suppressed as noise.

**Core insight**: Salience emerges naturally from precision-weighted prediction errors. What we find salient is not determined solely by bottom-up stimulus properties, but by the unexpected - stimuli that generate large precision-weighted prediction errors demand attention and processing resources.

From [Precision weighting of cortical unsigned prediction error signals](https://www.nature.com/articles/s41380-020-0803-8) (Haarsma et al., 2021, *Molecular Psychiatry*):
> "Such prediction error signals update prior beliefs in a manner that is weighted by their associated precision, such that more is learned from prediction errors with higher precision."

---

## Section 1: Precision as Inverse Variance

### Mathematical Definition

**Precision** (pi) is defined as the inverse of variance:

```
pi = 1/sigma^2
```

Where:
- pi = precision (confidence in the estimate)
- sigma^2 = variance (spread of the distribution)

**High precision** = Low variance = Tight distribution = High confidence
**Low precision** = High variance = Spread distribution = Low confidence

### Precision in Probabilistic Inference

In Bayesian terms, precision determines how much weight to give different sources of information:

```python
# Bayesian update with precision weighting
posterior_mean = (prior_precision * prior_mean + likelihood_precision * observation) / (prior_precision + likelihood_precision)

# Precision determines relative weighting
weight_prior = prior_precision / (prior_precision + likelihood_precision)
weight_observation = likelihood_precision / (prior_precision + likelihood_precision)
```

**Key insight**: When sensory precision is high (clear signal), observations dominate. When prior precision is high (strong expectations), priors dominate.

### Precision vs Attention: The Equivalence

Feldman & Friston (2010) established the theoretical equivalence:

```
Attention = E[pi]  # Expected precision
```

**Attention is not separate from perception** - it IS the process of optimizing precision estimates:
- Attending to something = Increasing its expected precision
- Ignoring something = Decreasing its expected precision

From [Attention, uncertainty, and free-energy](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00215/full) (Feldman & Friston, 2010):
> "Attention can be understood as the process of optimizing expected precision. This optimization corresponds to increasing synaptic gain on units encoding prediction errors."

---

## Section 2: Attention as Expected Precision

### The Gain Control Mechanism

**Attention operates via gain modulation on prediction error units**:

```python
# Precision-weighted prediction error
weighted_PE = precision * prediction_error

# Neural response scales with precision
response = baseline + gain * weighted_PE

# Where gain is controlled by expected precision
gain = f(expected_precision)
```

**Effect on neural processing**:
- High precision -> High gain -> Amplified prediction errors -> Strong belief updates
- Low precision -> Low gain -> Suppressed prediction errors -> Weak updates

From [The effects of neural gain on attention and learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC3725201/) (Eldar et al., 2013):
> "Increased gain narrows attention by strengthening already strong neural representations at the expense of competing weaker representations."

### Synaptic Implementation

**Postsynaptic gain modulation** implements precision weighting:
- Precision increases postsynaptic excitability
- Same presynaptic input produces stronger postsynaptic response
- Implemented via voltage-dependent conductances

**NMDA receptors** are particularly important:
- Voltage-dependent (require both pre and post activity)
- Allow multiplicative gain control
- Basis for Hebbian plasticity

### Top-Down vs Bottom-Up Precision

**Bottom-up precision** (stimulus-driven):
- Determined by signal quality
- High contrast = high precision
- Clear signal = high precision

**Top-down precision** (expectation-driven):
- Determined by context and goals
- Task relevance = increased precision
- Uncertainty about stimulus = decreased precision

```python
# Combined precision
total_precision = bottom_up_precision + top_down_precision_modulation

# Attention = top-down modulation of expected precision
attention_effect = top_down_precision_modulation
```

---

## Section 3: Gain Control Mechanisms

### Neural Gain Control

**Gain control** refers to multiplicative scaling of neural responses:

```
output = gain * input
```

Different from additive effects (baseline shifts):
```
output = input + offset  # NOT gain control
```

### Contrast Gain Control

The canonical example of gain control in vision:

From [Attention and Contrast Gain Control](https://psycnet.apa.org/record/2005-02531-010) (Reynolds & Heeger, 2009):
> "Attention has co-opted the circuits that mediate contrast gain control - the dynamic calibration of neuronal responsiveness."

**Normalization model**:
```
R_i = (E_i * A_i) / (sigma + sum_j(E_j * A_j))
```

Where:
- R_i = response of neuron i
- E_i = excitatory drive
- A_i = attention field
- sigma = semisaturation constant
- Denominator = suppressive normalization

### Attention as Response Gain

**Attention increases the gain of responses to attended stimuli**:

```python
# Without attention
response = contrast_response_function(stimulus)

# With attention (multiplicative gain)
response = attention_gain * contrast_response_function(stimulus)

# Attention shifts the contrast-response function leftward
effective_contrast = attention_factor * physical_contrast
```

**Evidence from single-unit recordings**:
- V4 neurons show multiplicative scaling with attention (McAdams & Maunsell, 1999)
- MT neurons show both additive and multiplicative components (Treue & Martinez-Trujillo, 1999)

### Precision as the Gain Parameter

**In predictive coding, precision IS the gain**:

From [How Prediction Errors Shape Perception, Attention, and Motivation](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00548/full) (den Ouden et al., 2012):
> "By enhancing the precision of specific PEs, attention increases the weight that is put on these errors in subsequent inference and learning. This is equivalent to proposals of attention increasing synaptic gain (precision) of specific sensory neurons."

---

## Section 4: Salience = Precision-Weighted Prediction Error

### The Salience Computation

**Salience is not a separate computation** - it emerges from precision-weighted prediction errors:

```python
salience = precision * |prediction_error|

# High salience when:
# 1. Large prediction error (unexpected)
# 2. High precision (reliable signal)
```

**What makes something salient**:
- NOT bottom-up properties alone (contrast, color)
- NOT top-down goals alone (task relevance)
- The COMBINATION: unexpected + reliable + relevant

### Predictive Coding Account of Salience

From [How Prediction Errors Shape Perception, Attention, and Motivation](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00548/full) (den Ouden et al., 2012):
> "Salience arises quite naturally from predictive coding theories of neural processing, since the amplitude of the response a stimulus evokes is directly determined by how unexpected it is."

**Benefits of this account**:
1. Explains why expected stimuli (even high contrast) are less salient
2. Explains why unexpected low-contrast stimuli can be highly salient
3. Natural mechanism for surprise-driven attention

### Unsigned vs Signed Salience

**Unsigned salience** (absolute prediction error):
- Signals surprise magnitude only
- |expected - observed|
- Drives orienting/alerting

**Signed salience** (valenced prediction error):
- Signals surprise direction
- positive = better than expected
- negative = worse than expected
- Drives learning and motivation

```python
# Unsigned PE (sensory salience)
sensory_salience = abs(prediction - observation)

# Signed PE (reward/motivational salience)
reward_PE = reward_obtained - reward_expected  # Can be + or -
```

### The Alerting Function

**Salience triggers orienting and alerting**:

From [Dopamine in motivational control: rewarding, aversive, and alerting](https://www.cell.com/neuron/fulltext/S0896-6273(10)00901-2) (Bromberg-Martin et al., 2010):
> "Midbrain dopamine neurons show a fast (<100ms) phasic increase in firing in response to salient unexpected sensory stimuli."

**Circuit**:
Superior colliculus -> Substantia nigra pars compacta -> Striatum

This pathway allows salient stimuli to interrupt ongoing processing and redirect attention/behavior.

---

## Section 5: Resource Allocation Based on Precision

### Precision as Resource Allocation Signal

**The brain allocates computational resources proportional to precision**:

```python
def allocate_resources(signals, total_budget):
    """Allocate resources based on precision."""
    precisions = [estimate_precision(s) for s in signals]

    # Normalize to budget
    total_precision = sum(precisions)
    allocations = [(p / total_precision) * total_budget for p in precisions]

    return allocations
```

**Resource types allocated**:
- Neural firing rates (more spikes for precise signals)
- Synaptic plasticity (more learning from precise errors)
- Metabolic resources (more glucose/oxygen)
- Working memory capacity
- Processing time

### Optimal Resource Allocation

**Resource-rational perspective**: The brain optimally trades off accuracy vs computational cost.

From [Resource-rational analysis](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A) (Lieder & Griffiths, 2020):

```python
# Optimal resource allocation criterion
utility = information_gain - computational_cost

# Allocate where marginal gain exceeds marginal cost
allocate if: dI/dR > dC/dR
```

**Precision guides this allocation**:
- High-precision signals have high information gain per unit resource
- Low-precision signals have diminishing returns

### Hierarchical Precision Allocation

**Different levels of the hierarchy have different precision requirements**:

```
Sensory level: High precision (detailed, concrete)
    |
    v  Precision-weighted PEs propagate up
    |
Associative level: Medium precision (integrative)
    |
    v
Conceptual level: Low precision (abstract, flexible)
```

**Implication**: Resources should be differentially allocated across the hierarchy based on task demands.

---

## Section 6: Neural Implementation (Neuromodulators)

### Dopamine and Precision

**Dopamine encodes precision of beliefs about policies**:

From [The Dopaminergic Midbrain Encodes the Expected Certainty about Desired Outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4585497/) (Schwartenbeck et al., 2015):
> "We proposed a generic model based on active (Bayesian) inference wherein dopamine encodes the precision of beliefs about optimal policies."

**Dopamine's dual role**:
1. **Phasic dopamine**: Reward prediction errors (signed)
2. **Tonic dopamine**: Precision/confidence in action policies

```python
# Phasic DA: RPE signals
phasic_DA = reward - expected_reward

# Tonic DA: Precision of policy beliefs
tonic_DA = precision_of_optimal_policy
```

**Clinical implication**: Dopamine dysfunction -> aberrant precision -> psychosis

### Acetylcholine and Sensory Precision

From [Acetylcholine modulates the precision of prediction error in the auditory cortex](https://elifesciences.org/articles/91475) (Perez-Gonzalez et al., 2024):
> "Precision weighting of prediction errors, therefore, encodes the confidence or reliability afforded such errors."

**ACh function**:
- Increases sensory precision
- Enhances signal-to-noise ratio
- Sharpens neural tuning

```python
# ACh modulation of precision
sensory_precision = baseline_precision * (1 + ACh_release * ACh_sensitivity)

# High ACh -> Trust sensory input more
# Low ACh -> Trust predictions more
```

**Basal forebrain cholinergic system**:
- Receives top-down inputs about uncertainty
- Projects broadly to cortex
- Modulates cortical precision globally

### Norepinephrine and Network Gain

**Locus coeruleus-norepinephrine (LC-NE) system**:

From [The role of the LC-NE system in attention](https://www.sciencedirect.com/science/article/abs/pii/S0149763425002337) (Torres et al., 2025):
> "The LC-NE system regulates attention by optimizing neural gain and signal-to-noise ratio."

**NE function**:
- Global gain modulation
- Arousal and alertness
- Exploration-exploitation balance

```python
# High NE (phasic): Narrowed attention, high gain
# Low NE (tonic): Broad attention, low gain

gain = baseline_gain + NE_level * gain_sensitivity
```

### GABA and Precision of Predictions

**GABAergic inhibition implements predictions**:

From cortical microcircuit models:
- Interneurons encode predictions
- Inhibit prediction error units
- Reduce PE when predictions match input

```python
# PE unit activity
PE_activity = excitatory_input - inhibitory_prediction

# Where inhibitory prediction comes from GABAergic interneurons
inhibitory_prediction = GABA_weight * prediction_unit_activity
```

---

## Section 7: Mathematical Formulation

### Precision-Weighted Prediction Error

The core computation:

```python
# Precision-weighted PE
weighted_PE = pi * epsilon

# Where:
# pi = precision (inverse variance)
# epsilon = prediction error (observation - prediction)
```

### Bayesian Update with Precision

```python
# Posterior mean (precision-weighted combination)
mu_posterior = (pi_prior * mu_prior + pi_likelihood * observation) / (pi_prior + pi_likelihood)

# Posterior precision (sum of precisions)
pi_posterior = pi_prior + pi_likelihood
```

### Free Energy and Precision

In free energy terms:

```
F = -ln p(y|theta) + KL[q(theta)||p(theta)]
  = Prediction_error + Complexity
```

**Precision appears in the prediction error term**:

```
Prediction_error = (1/2) * pi * epsilon^2

# Precision scales the cost of prediction errors
# High precision: errors are costly
# Low precision: errors are cheap
```

### Hierarchical Message Passing with Precision

At each level of hierarchy:

```python
# Bottom-up: precision-weighted PE
message_up = pi_bottom * epsilon

# Top-down: predictions (expected values)
message_down = mu_top

# Update rule
mu_new = mu_old + learning_rate * pi * epsilon
```

**Precision-weighted message passing**:
1. Compute prediction error: epsilon = input - prediction
2. Weight by precision: weighted_PE = pi * epsilon
3. Update belief: mu += learning_rate * weighted_PE
4. Update precision: pi based on PE variance

### Attention in the Hierarchy

```python
# Attention modulates precision at specific levels
pi_attended = baseline_precision + attention_modulation

# This affects how much that level's PEs influence updates
weighted_PE = pi_attended * epsilon

# Result: attended information has more influence on inference
```

---

## Section 8: ARR-COC-0-1 Connection - Token Allocation AS Precision Weighting

### The Core Equivalence

**ARR-COC-0-1's token allocation directly implements precision weighting**:

```python
# In Friston's framework
weighted_PE = precision * prediction_error

# In ARR-COC-0-1
tokens_allocated = precision_score * base_tokens

# The mapping is direct:
# Precision score -> Token budget
# More tokens = More "computational gain" on that patch
```

### Precision Computation in ARR-COC

From the existing oracle knowledge (cognitive-mastery/01-precision-attention-resource.md):

```python
# knowing.py - Multiple precision sources
information_precision = shannon_entropy(patch)      # Propositional
perspectival_precision = archetypal_salience(patch) # Perspectival
participatory_precision = query_relevance(patch, query)  # Participatory

# Combined precision (balancing.py)
total_precision = balance_tensions(
    information_precision,
    perspectival_precision,
    participatory_precision
)

# Map to tokens (attending.py)
tokens = precision_to_token_budget(total_precision, min=64, max=400)
```

### Token Budget as Gain Control

**More tokens = Higher gain = More detailed encoding**:

```python
# In the brain
high_precision -> high_gain -> amplified_response -> detailed_encoding

# In ARR-COC-0-1
high_precision -> more_tokens -> more_capacity -> detailed_encoding
```

**The parallel**:
| Biological System | ARR-COC-0-1 |
|-------------------|-------------|
| Precision (1/variance) | Precision score (0-1) |
| Synaptic gain | Token allocation (64-400) |
| Neural firing rate | Embedding dimensionality |
| Metabolic resources | Compute resources |

### Salience-Driven Allocation

**Query-driven precision = Top-down precision modulation**:

```python
# Query creates expectations about relevance
query = "Find the red car in the parking lot"

# Patches matching query get precision boost
if patch.semantic_relevance(query) > threshold:
    top_down_precision_boost = high
else:
    top_down_precision_boost = low

# Combined with bottom-up precision
total_precision = bottom_up_entropy + top_down_precision_boost
```

This mirrors how attention in the brain modulates expected precision.

### Resource-Rational Token Allocation

**ARR-COC-0-1 implements resource-rational cognition**:

```python
# Total token budget = Computational constraint
budget_constraint: sum(tokens_per_patch) <= max_total_tokens

# Optimization objective
maximize: information_about_query
subject_to: budget_constraint

# Solution: Allocate proportional to precision
for patch in patches:
    tokens[patch] = (precision[patch] / sum_precisions) * total_budget
```

**This is optimal because**:
- High-precision patches have more information per token
- Low-precision patches have diminishing returns
- Budget forces trade-offs that precision optimally resolves

### Precision-Weighted Compression

**Token count determines compression fidelity**:

```python
# realizing.py - Precision determines compression
def compress_patch(patch, tokens):
    if tokens >= 300:  # High precision
        # Minimal compression, preserve details
        return encode_detailed(patch, tokens)
    elif tokens <= 100:  # Low precision
        # Aggressive compression, lose details
        return encode_coarse(patch, tokens)
    else:  # Medium precision
        return encode_moderate(patch, tokens)
```

**Analogy to foveal vision**:
| Biological | ARR-COC-0-1 |
|------------|-------------|
| Fovea (high acuity) | 400-token patches |
| Parafovea (medium) | 128-256 tokens |
| Periphery (low acuity) | 64-token patches |
| Saccades (reallocation) | Query-driven reallocation |

### Active Inference in VLMs

**ARR-COC-0-1 can be viewed as active inference**:

```python
# Expected free energy guides attention
EFE = expected_ambiguity - expected_information_gain

# Query reduces ambiguity about relevant regions
query_relevance -> reduces_EFE -> increases_precision -> more_tokens

# This implements epistemic foraging:
# Allocate precision (tokens) to reduce uncertainty about query
```

**The system minimizes expected free energy by**:
1. Allocating tokens to informative regions (reduces ambiguity)
2. Compressing uninformative regions (reduces computational cost)
3. Query-driven allocation (pragmatic value)

### Empirical Predictions

**Testable predictions from this framework**:

1. **Token allocation should correlate with human gaze patterns**
   - High-token regions should match fixation density
   - Test: Eye-tracking comparison

2. **Accuracy should scale with precision**
   - Questions about high-precision regions: High accuracy
   - Questions about low-precision regions: Lower accuracy
   - Test: VQA accuracy vs token allocation

3. **Query-driven reallocation should improve relevance**
   - Same image, different queries -> different allocations
   - Test: Compare allocations across queries

4. **Optimal budget exists**
   - Too few tokens: Insufficient precision
   - Too many tokens: Wasted computation
   - Test: Accuracy vs total budget curve

### Integration with Vervaeke's Framework

**Precision weighting implements relevance realization**:

- **Propositional precision** (entropy) = Information content
- **Perspectival precision** (salience) = Agent-specific relevance
- **Participatory precision** (query coupling) = Task relevance

**The balance of precisions IS relevance realization**:
```python
relevance = balance(propositional, perspectival, participatory)
# This balance determines what is salient
# What is salient gets tokens (resources)
# What gets tokens gets processed in detail
```

---

## Sources

### Source Documents

**Existing Oracle Knowledge**:
- [cognitive-mastery/01-precision-attention-resource.md](../cognitive-mastery/01-precision-attention-resource.md) - Foundation for precision-attention equivalence

### Web Research

**Precision-Weighted Prediction Errors**:
- Haarsma, J. et al. (2021). "Precision weighting of cortical unsigned prediction error signals." *Molecular Psychiatry*, 26, 4358-4366. https://www.nature.com/articles/s41380-020-0803-8 (accessed 2025-11-23)
- den Ouden, H.E.M. et al. (2012). "How Prediction Errors Shape Perception, Attention, and Motivation." *Frontiers in Psychology*, 3:548. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00548/full (accessed 2025-11-23)
- Clark, A. (2013). "The many faces of precision." *Frontiers in Psychology*, PMC3659294. https://pmc.ncbi.nlm.nih.gov/articles/PMC3659294/ (accessed 2025-11-23)

**Attention as Gain Control**:
- Eldar, E. et al. (2013). "The effects of neural gain on attention and learning." *Nature Neuroscience*, 16(8), 1146-1153. PMC3725201. https://pmc.ncbi.nlm.nih.gov/articles/PMC3725201/ (accessed 2025-11-23)
- Reynolds, J.H. & Heeger, D.J. (2009). "The normalization model of attention." *Neuron*, 61(2), 168-185.
- Smout, C.A. et al. (2019). "Attention promotes the neural encoding of prediction errors." *PLoS Biology*, 17(2), e2006812. PMC6411367. https://pmc.ncbi.nlm.nih.gov/articles/PMC6411367/ (accessed 2025-11-23)

**Dopamine and Precision**:
- Schwartenbeck, P. et al. (2015). "The Dopaminergic Midbrain Encodes the Expected Certainty about Desired Outcomes." *Cerebral Cortex*, 25(10), 3434-3445. PMC4585497. https://pmc.ncbi.nlm.nih.gov/articles/PMC4585497/ (accessed 2025-11-23)
- Bromberg-Martin, E.S. et al. (2010). "Dopamine in motivational control: rewarding, aversive, and alerting." *Neuron*, 68(5), 815-834.

**Neuromodulators and Precision**:
- Perez-Gonzalez, D. et al. (2024). "Acetylcholine modulates the precision of prediction error in the auditory cortex." *eLife*, 13:e91475. https://elifesciences.org/articles/91475 (accessed 2025-11-23)
- Torres, A.S. et al. (2025). "The role of the LC-NE system in attention: From cells, to circuits, to cognition." *Neuroscience & Biobehavioral Reviews*. https://www.sciencedirect.com/science/article/abs/pii/S0149763425002337 (accessed 2025-11-23)

**Theoretical Foundations**:
- Feldman, H. & Friston, K.J. (2010). "Attention, uncertainty, and free-energy." *Frontiers in Human Neuroscience*, 4:215. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00215/full (accessed 2025-11-23)
- Friston, K.J. (2009). "The free-energy principle: a rough guide to the brain?" *Trends in Cognitive Sciences*, 13(7), 293-301.
- Yon, D. et al. (2021). "Precision and the Bayesian brain." *Current Biology*, 31(17), R1026-R1032. https://www.sciencedirect.com/science/article/pii/S0960982221010344 (accessed 2025-11-23)

**Resource-Rational Analysis**:
- Lieder, F. & Griffiths, T.L. (2020). "Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources." *Behavioral and Brain Sciences*, 43, e1. https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A (accessed 2025-11-23)

### Additional References

- Kok, P. et al. (2012). "Attention reverses the effect of prediction in silencing sensory signals." *Cerebral Cortex*, 22(9), 2197-2206.
- Spratling, M.W. (2012). "Predictive coding as a model of the V1 saliency map hypothesis." *Neural Networks*, 26, 7-28.
- Hohwy, J. (2012). "Attention and conscious perception in the hypothesis testing brain." *Frontiers in Psychology*, 3:96.
