# Precision & Attention as Resource Allocation

## Overview

Precision-weighting mechanisms transform attention from a binary on/off process into a sophisticated resource allocation system that modulates the influence of sensory signals based on their estimated reliability. This framework, central to predictive processing and active inference theories, conceptualizes attention as expected precision—the brain's confidence in prediction errors that determine how sensory information updates internal models.

**Core principle**: Attention operates as gain control on precision-weighted prediction errors, allocating limited computational resources to reliable sensory signals while suppressing unreliable noise.

From [Precision weighting of cortical unsigned prediction error signals](https://www.nature.com/articles/s41380-020-0803-8) (Haarsma et al., 2021, *Molecular Psychiatry*):
> "Precision weighting of cortical prediction error signals is a key mechanism through which dopamine modulates inference and contributes to the pathogenesis of psychosis."

## Section 1: Precision-Weighting (Gain Control, Confidence Scaling)

### Expected Precision as Attention

**Precision** in predictive coding refers to the inverse variance (certainty) of a probability distribution. When the brain estimates sensory signals, it simultaneously estimates their precision—how reliable that estimate is.

**Attention as expected precision** (Feldman & Friston, 2010):
```
Attention = E[π]  # Expected precision
π = 1/σ²          # Precision (inverse variance)
```

Where:
- High precision → Low variance → Reliable signal → Increase gain
- Low precision → High variance → Unreliable signal → Decrease gain

**Gain modulation mechanism**:
```python
# Precision-weighted prediction error
weighted_PE = precision * prediction_error

# Update = precision-scaled error
belief_update = learning_rate * weighted_PE
```

**Effect on neural responses**:
- High-precision signals: Amplified via increased postsynaptic gain
- Low-precision signals: Suppressed via decreased gain
- Implements adaptive filtering of unreliable information

From [Acetylcholine modulates the precision of prediction error in the auditory cortex](https://elifesciences.org/articles/91475) (Pérez-González et al., 2024, *eLife*):
> "Precision weighting of prediction errors, therefore, encodes the confidence in the accuracy of the information (i.e. prediction errors) in guiding belief updating."

### Dopamine and Precision Encoding

**Dopamine's role in precision-weighting**:

From [The Dopaminergic Midbrain Encodes the Expected Certainty about Desired Outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4585497/) (Schwartenbeck et al., 2015):
> "We proposed a generic model based on active (Bayesian) inference wherein dopamine encodes the precision of beliefs about optimal policies."

**Key findings**:
1. **Phasic dopamine** → Expected precision of prediction errors
2. **Tonic dopamine** → Baseline precision (signal-to-noise ratio)
3. **DA dysfunction** → Aberrant precision → Psychosis (false salience)

**Computational implementation**:
- Dopamine scales prediction error precision (π)
- High DA → High precision → Strong belief updates
- Low DA → Low precision → Weak updates (cognitive inertia)

### Confidence Scaling Mechanisms

**Metacognitive precision** - Confidence in confidence:

From [Precision and False Perceptual Inference](https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2018.00039/full) (Parr & Friston, 2018):
> "Precision-weighting mechanism amplifies the influence of reliable prediction errors via gain modulation."

**Hierarchical precision cascades**:
```
Level 3: High-level concepts (low precision, abstract)
   ↓ Precision-weighted PEs ↓
Level 2: Mid-level features (medium precision)
   ↓ Precision-weighted PEs ↓
Level 1: Sensory input (high precision, concrete)
```

**Layer-specific precision allocation**:
- **Sensory layers**: High precision for clear stimuli, low for noisy
- **Associative layers**: Medium precision, integrate context
- **Executive layers**: Low precision, flexible hypothesis updates

### Acetylcholine and Sensory Precision

From [Acetylcholine modulates the precision of prediction error in the auditory cortex](https://elifesciences.org/articles/91475) (Pérez-González et al., 2024):

**Findings**:
- **Acetylcholine (ACh)** modulates precision of sensory prediction errors
- **Optogenetic ACh activation** → Increased precision → Sharper tuning
- **ACh as precision dial**: Adjusts signal-to-noise in sensory cortex

**Mechanism**:
```
ACh release → Precision increase → Enhanced PE weighting → Faster learning
```

**Implications for attention**:
- ACh not just "arousal" but precision modulation
- Attention = ACh-mediated precision enhancement
- Top-down control via basal forebrain → cortical ACh release

## Section 2: Attention as Expected Precision (Allocate Resources to Informative Signals)

### Information-Theoretic Foundation

**Precision tracks information content**:

High-precision signals carry more information (lower entropy):
```
Information = -log₂(P(signal))
Precision ∝ Information content
```

**Allocation rule**: Allocate attention to high-precision (informative) signals.

From [Minimizing Precision-Weighted Sensory Prediction Errors via Memory Creation, Selection, and Update](https://pmc.ncbi.nlm.nih.gov/articles/PMC6855676/) (Oh et al., 2019):
> "We proposed a new model of motor adaptation, which uses multiple precision-weighted prediction errors for memory creation, selection, and update."

**Motor adaptation example**:
- **High-precision error** (clear visual feedback) → Large update
- **Low-precision error** (proprioceptive noise) → Small update
- Adaptive weighting prevents noise-driven learning

### Expected Free Energy and Precision

**Active inference** (Friston): Agents minimize expected free energy (EFE)

```
EFE = Expected ambiguity - Expected information gain
        ↑                          ↑
   (imprecise signals)      (precise signals)
```

**Precision determines information gain**:
- Attending to precise signals → Reduces uncertainty → Lowers EFE
- Ignoring imprecise signals → Avoids noise → Prevents overfitting

**Policy selection**:
```python
# Choose actions that sample high-precision observations
policy_value = -EFE
             = information_gain - ambiguity
             = E[precision] * novelty - H[observations]
```

From [Active Inference: A Process Theory](https://activeinference.github.io/papers/process_theory.pdf) (Friston et al., 2017):
> "We have previously considered the intimate (monotonic) relationship between expected precision and expected utility in this context."

### Precision-Driven Exploration

**Exploration-exploitation tradeoff mediated by precision**:

**Exploit**: Sample high-precision, familiar sensory data
- Low uncertainty about outcomes
- High confidence in predictions
- Minimal expected ambiguity

**Explore**: Sample low-precision, novel sensory data
- High uncertainty → Potential information gain
- Low confidence in current model
- Epistemic value (resolve uncertainty)

**Precision modulates exploration**:
```python
if expected_precision_high:
    # Exploit: Trust current model
    action = argmax(expected_reward)
else:
    # Explore: Reduce uncertainty
    action = argmax(information_gain)
```

## Section 3: Resource-Rational Cognition (Bounded Rationality, Optimality)

### Resource-Rational Analysis Framework

From [Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A) (Lieder & Griffiths, 2020, *Behavioral and Brain Sciences*):

**Core tenets**:
1. **Computational constraints**: Cognition is bounded by time, energy, memory
2. **Optimal resource use**: Strategies adapted to computational costs
3. **Precision as resource**: Allocating precision = allocating computation

**Resource-rational decision making**:
```
Utility = Expected reward - Computational cost
```

**Attention allocation**:
```
Attend where: Information_gain / Cost > threshold
```

### Bounded Rationality

**Simon's bounded rationality** + **Kahneman & Tversky's heuristics** = Resource-rational framework

**Heuristics as resource-rational** (not "biases"):
- **Availability heuristic**: Sample high-precision (easily retrieved) memories
- **Representativeness**: Use prototypical (high-precision) features
- **Anchoring**: Default to high-confidence prior, adjust with precision-weighted evidence

**Computational costs**:
- **Time**: Limited processing cycles
- **Energy**: Metabolic cost of neural firing
- **Memory**: Working memory capacity (~4 items)
- **Attention**: Precision allocation bottleneck

From [Resource-rational decision making](https://www.sciencedirect.com/science/article/abs/pii/S2352154621000371) (Bhui & Gershman, 2021, *Current Opinion in Behavioral Sciences*):
> "This 'resource-rational' analysis connects psychology and neuroscience to ideas from engineering, economics, and machine learning."

### Optimality Under Constraints

**Precision allocation as optimization**:

**Objective**: Maximize expected utility under precision budget constraint

```python
def optimize_precision_allocation(signals, budget):
    """
    Allocate precision to maximize information gain.

    Constraint: sum(precision) <= budget
    """
    # Sort signals by information content
    sorted_signals = sort_by_entropy(signals)

    # Greedy allocation to most informative
    precision = np.zeros(len(signals))
    remaining_budget = budget

    for i, signal in enumerate(sorted_signals):
        if remaining_budget > 0:
            precision[i] = min(signal.max_precision, remaining_budget)
            remaining_budget -= precision[i]

    return precision
```

**Trade-offs**:
- **High precision everywhere**: Accurate but computationally expensive
- **Low precision everywhere**: Cheap but inaccurate
- **Optimal**: High precision on task-relevant, low on irrelevant

### Anytime Algorithms

**Resource-rational attention = anytime computation**:

From [Rational use of cognitive resources in human planning](https://escholarship.org/uc/item/11j0r4gj) (Callaway et al., 2022, *Nature Human Behaviour*):
> "Using the approach of resource-rational analysis, we can use the observed discrepancies to generate hypotheses about additional constraints."

**Anytime property**:
- Can return answer at any time
- Longer compute → Better answer
- Precision determines stopping criterion

**Example**: Visual search
```python
def visual_search_anytime(image, query, time_budget):
    precision = initial_precision
    best_match = None

    while time_remaining(time_budget):
        # Increase precision iteratively
        match = search_with_precision(image, query, precision)

        if confidence(match) > threshold:
            return match  # Early stopping

        precision += delta_precision  # Refine

    return best_match
```

## Section 4: Token Budget as Precision Allocation (64-400 Tokens)

### VLM Token Budgets

**Standard VLMs**: Fixed token allocation per patch
- CLIP: 256 tokens per patch
- BLIP-2: 32 query tokens (fixed)
- Qwen-VL: 256 tokens per image region

**Problem**: Wastes tokens on low-information regions

**ARR-COC-0-1 solution**: Dynamic 64-400 token allocation

### Precision Mapping to Token Count

**Precision → Token budget mapping**:

```python
def precision_to_tokens(precision_score, min_tokens=64, max_tokens=400):
    """
    Map precision score to token allocation.

    High precision → More tokens (detailed encoding)
    Low precision → Fewer tokens (compressed encoding)
    """
    normalized = (precision_score - min_precision) / (max_precision - min_precision)
    tokens = min_tokens + normalized * (max_tokens - min_tokens)
    return int(tokens)
```

**Precision sources** (from ARR-COC knowing.py):
1. **Propositional** (InformationScorer): Shannon entropy
2. **Perspectival** (PerspectivalScorer): Archetypal salience
3. **Participatory** (ParticipatoryScorer): Query-content coupling

**Combined precision**:
```python
total_precision = (
    w1 * information_precision +
    w2 * perspectival_precision +
    w3 * participatory_precision
)
```

### Budget Constraint Optimization

**Total token budget constraint**:
```
sum(tokens_per_patch) <= max_total_tokens
```

**If budget exceeded**: Normalize via divisive suppression
```python
if sum(tokens) > budget:
    scale_factor = budget / sum(tokens)
    tokens = tokens * scale_factor
```

**Implements normalization model** (Section 7 from attention-resource-allocation.md):
```
R_i = (E_i × A_i) / (σ + Σ_j E_j)
```

Where:
- E_i = Expected precision of patch i
- A_i = Attention field (query relevance)
- Denominator = Competitive normalization

## Section 5: Pipeline Stages (File 2: Attention Allocation Across Processing Stages)

### Attention Pipeline Architecture

**Multi-stage precision allocation**:

```
Stage 1: Feature Extraction
   ├─ Precision: High (sensory input)
   └─ Tokens: Full resolution (e.g., 256)

Stage 2: Feature Fusion
   ├─ Precision: Medium (cross-modal)
   └─ Tokens: Reduced (e.g., 128)

Stage 3: Reasoning
   ├─ Precision: Variable (task-dependent)
   └─ Tokens: Adaptive (64-400)

Stage 4: Output Generation
   ├─ Precision: Low (abstract)
   └─ Tokens: Minimal (16-32)
```

### Pipeline Parallelism for Precision

From [DeepSpeed Pipeline Parallelism](distributed-training/01-deepspeed-pipeline-parallelism.md):

**Micro-batching enables stage-wise precision**:

**Problem**: Different stages have different precision requirements
- Vision encoder: High precision (detailed features)
- Cross-attention: Medium precision (fusion)
- Language decoder: Low precision initially, high at output

**Solution**: Pipeline with heterogeneous precision
```python
# Stage 1: Vision encoding (FP16, high precision)
vision_features = vision_encoder(image, dtype=torch.float16)

# Stage 2: Cross-attention (FP16→BF16, medium precision)
fused_features = cross_attention(vision_features, text_embeddings, dtype=torch.bfloat16)

# Stage 3: Language generation (BF16, lower precision acceptable)
output = language_decoder(fused_features, dtype=torch.bfloat16)
```

**Gradient precision** varies by stage:
- Early stages: Low-precision gradients acceptable (less sensitive)
- Late stages: High-precision gradients needed (directly affect output)

### Asynchronous Precision Updates

**Challenge**: Updating precision estimates is costly

**Solution**: Async precision computation in pipeline

```python
# Pipeline stage with async precision update
class PrecisionAwarePipelineStage:
    def forward(self, x, precision=None):
        if precision is None:
            precision = self.cached_precision  # Use cached

        # Compute weighted features
        weighted = precision * x

        # Async: Update precision for next iteration
        self.precision_estimator.update_async(x)

        return weighted
```

**Benefits**:
- Precision computation doesn't block forward pass
- Amortizes precision cost across multiple batches
- Enables online precision adaptation

## Section 6: Serving Optimization (File 6: Dynamic Precision for Inference)

### TensorRT Dynamic Precision

From [TensorRT for VLM Deployment](inference-optimization/01-tensorrt-vlm-deployment.md):

**Mixed-precision inference**:
- FP32: High precision, slow (baseline)
- FP16: Medium precision, 2× faster
- INT8: Low precision, 4× faster
- FP8 (H100): Medium-low precision, 2× faster than FP16

**Query-aware precision selection**:

```python
def select_precision(query, image):
    """Choose precision level based on query requirements."""
    if is_detail_query(query):
        # "Count small objects" → Need high precision
        return torch.float16
    elif is_general_query(query):
        # "Describe scene" → Lower precision OK
        return torch.bfloat16
    else:
        # Unknown → Default to medium
        return torch.float16
```

### Adaptive Batch Precision

**Problem**: Batching requests with different precision needs

**Solution**: Split batch by precision requirement

```python
# Separate high/low precision requests
high_precision_batch = [req for req in batch if req.needs_high_precision]
low_precision_batch = [req for req in batch if not req.needs_high_precision]

# Process with appropriate precision
high_results = model.forward(high_precision_batch, dtype=torch.float16)
low_results = model.forward(low_precision_batch, dtype=torch.bfloat16)
```

**Dynamic precision** during inference:
```python
# Start with low precision
output = model.forward(input, precision='low')

if confidence(output) < threshold:
    # Retry with higher precision if uncertain
    output = model.forward(input, precision='high')
```

### Precision-Latency Trade-off

**Latency vs precision**:

| Precision | Latency (A100) | Accuracy | Use Case |
|-----------|----------------|----------|----------|
| FP32 | 16ms | 100% (baseline) | Research, debugging |
| FP16 | 8ms | 99.5% | Production (general) |
| BF16 | 8ms | 99.2% | Training-compatible |
| INT8 | 4ms | 97-99% | High-throughput serving |
| FP8 | 4ms (H100) | 98-99.5% | Latest hardware |

**Query-specific latency budgets**:
- Real-time (< 50ms): Use INT8/FP8, sacrifice precision
- Interactive (< 200ms): Use FP16, balanced
- Batch (> 1s): Use FP16/FP32, maximize accuracy

## Section 7: ML Pipelines (File 10: Kubeflow for Attention Experiments)

### Kubeflow for Precision Experiments

From [Kubeflow ML Pipelines](orchestration/01-kubeflow-ml-pipelines.md):

**Pipeline for precision allocation experiments**:

```python
@kfp.dsl.pipeline(
    name='precision-allocation-experiments',
    description='Test different precision allocation strategies'
)
def precision_experiment_pipeline(
    dataset_path: str,
    precision_strategies: list,
    token_budgets: list
):
    # Component 1: Prepare dataset
    data_task = prepare_data_op(dataset_path=dataset_path)

    # Component 2: Run experiments (grid search)
    for strategy in precision_strategies:
        for budget in token_budgets:
            experiment_task = run_precision_experiment_op(
                data=data_task.output,
                strategy=strategy,
                budget=budget
            )

            # Component 3: Evaluate
            eval_task = evaluate_op(
                model=experiment_task.outputs['model'],
                metrics=['accuracy', 'token_efficiency']
            )
```

**Precision strategies to test**:
1. **Uniform**: All patches 256 tokens (baseline)
2. **Entropy-based**: Allocate by information content
3. **Query-driven**: Allocate by query relevance
4. **Hybrid**: Combine entropy + query (ARR-COC)

### Distributed Precision Experiments

**Challenge**: Testing many precision configurations is expensive

**Solution**: Distributed hyperparameter search

```python
# Katib (Kubeflow) experiment for precision tuning
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: precision-allocation-tuning
spec:
  objective:
    type: maximize
    objectivemetricName: accuracy_per_token
  algorithm:
    algorithmName: bayesian
  parameters:
    - name: min_tokens
      parameterType: int
      feasibleSpace:
        min: "32"
        max: "128"
    - name: max_tokens
      parameterType: int
      feasibleSpace:
        min: "128"
        max: "512"
    - name: precision_weight_entropy
      parameterType: double
      feasibleSpace:
        min: "0.0"
        max: "1.0"
```

**Metrics to track**:
- **Accuracy**: Task performance
- **Token efficiency**: Accuracy / avg_tokens_used
- **Latency**: Inference time
- **Memory**: Peak GPU usage

### Reproducibility and Ablation

**Kubeflow ensures reproducibility** of precision experiments:

```python
# Track all precision configurations
experiment_metadata = {
    'precision_strategy': 'query_driven',
    'token_budget': {'min': 64, 'max': 400},
    'weights': {'entropy': 0.3, 'salience': 0.2, 'query': 0.5},
    'dataset': 'vqa_v2',
    'model': 'arr-coc-0-1-v1.0'
}
```

**Ablation studies**:
1. Remove entropy scorer → Performance drop?
2. Remove query scorer → Performance drop?
3. Fixed tokens vs dynamic → Efficiency gain?

## Section 8: ARR-COC-0-1: Token Allocation AS Precision Weighting (10%)

### Precision-Weighting in ARR-COC-0-1

**Direct implementation of precision-weighted prediction errors**:

```python
# knowing.py - Compute precision scores
information_precision = shannon_entropy(patch)  # Propositional
perspectival_precision = archetypal_salience(patch)  # Perspectival
participatory_precision = query_relevance(patch, query)  # Participatory

# balancing.py - Combine precisions (opponent processing)
total_precision = balance_tensions(
    information_precision,
    perspectival_precision,
    participatory_precision
)

# attending.py - Map precision to tokens
tokens = precision_to_token_budget(total_precision, min=64, max=400)
```

### Precision as Gain Control

**Token count = Gain on visual features**:

More tokens → More detailed encoding → Higher gain on prediction errors from that patch

```python
# realizing.py - Apply precision weighting
def compress_patch(patch, tokens):
    """
    Higher tokens = Higher precision = More detailed compression.
    Lower tokens = Lower precision = Lossy compression.
    """
    if tokens >= 300:
        # High precision: Minimal compression
        return encode_detailed(patch, tokens)
    elif tokens <= 100:
        # Low precision: Aggressive compression
        return encode_coarse(patch, tokens)
    else:
        # Medium precision: Balanced
        return encode_moderate(patch, tokens)
```

### Expected Precision and Active Inference

**ARR-COC implements active inference principles**:

**Expected precision** → **Expected token allocation**

```python
# Query creates expectations about precision
query = "Find small red car in parking lot"

# Expected high precision regions:
expected_high_precision = [
    'parking lot areas',
    'vehicle regions',
    'red color regions'
]

# Allocate tokens accordingly
for patch in image_patches:
    if patch.matches(expected_high_precision):
        tokens[patch] = 400  # High precision
    else:
        tokens[patch] = 64   # Low precision (background)
```

**Free energy minimization**:
- Allocating precision to query-relevant regions → Reduces expected ambiguity
- Compressing irrelevant regions → Reduces computational cost
- Optimal allocation → Minimizes expected free energy

### Resource-Rational Token Allocation

**ARR-COC as resource-rational cognition**:

**Bounded rationality**:
- Total token budget = Computational constraint
- Cannot encode everything at high detail
- Must allocate tokens strategically

**Optimality criterion**:
```
Maximize: Information gain from visual input
Subject to: sum(tokens_per_patch) <= total_budget
```

**Solution**: Allocate tokens proportional to expected precision

**Comparison to biological vision**:

| Biological | ARR-COC-0-1 |
|------------|-------------|
| Foveal vision (high acuity) | High-token patches (400 tokens) |
| Peripheral vision (low acuity) | Low-token patches (64 tokens) |
| Saccades (attention shifts) | Dynamic reallocation based on query |
| Precision-weighted integration | Token-weighted feature fusion |

**Empirical validation needed**:
1. Compare token allocation to human gaze patterns
2. Measure accuracy vs token budget curves
3. Test query-driven precision modulation
4. Ablate precision sources (entropy, salience, query)

## Sources

### Source Documents

**Vervaeke Framework**:
- [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) - Foundational attention theory

**Influential Files (PART 2 specification)**:
- [distributed-training/01-deepspeed-pipeline-parallelism.md](../karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md) - Pipeline precision allocation
- [inference-optimization/01-tensorrt-vlm-deployment.md](../karpathy/inference-optimization/01-tensorrt-vlm-deployment.md) - Dynamic precision serving
- [orchestration/01-kubeflow-ml-pipelines.md](../karpathy/orchestration/01-kubeflow-ml-pipelines.md) - Precision experiment orchestration

### Web Research

**Precision-Weighted Prediction Errors**:
- Haarsma, J. et al. (2021). "Precision weighting of cortical unsigned prediction error signals." *Molecular Psychiatry*, 26, 4358–4366. https://www.nature.com/articles/s41380-020-0803-8 (accessed 2025-11-16)
- Oh, Y. et al. (2019). "Minimizing Precision-Weighted Sensory Prediction Errors via Memory Creation, Selection, and Update." *Journal of Neuroscience*, 39(42), 8423–8435. PMC6855676. (accessed 2025-11-16)
- Pérez-González, D. et al. (2024). "Acetylcholine modulates the precision of prediction error in the auditory cortex." *eLife*, 13:e91475. https://elifesciences.org/articles/91475 (accessed 2025-11-16)

**Attention as Gain Control**:
- Nobre, A.C. et al. (2025). "How the brain shifts between external and internal attention." *Neuron*. https://www.sciencedirect.com/science/article/pii/S0896627325004714 (accessed 2025-11-16)
- DeYoe, E.A. et al. (2024). "Are neuronal mechanisms of attention universal across human cortex?" *Psychonomic Bulletin & Review*. https://link.springer.com/article/10.3758/s13423-024-02495-3 (accessed 2025-11-16)

**Active Inference and Expected Precision**:
- Friston, K. et al. (2016). "Active inference and learning." *Neuroscience & Biobehavioral Reviews*, 68, 862-879. PMC5167251. (accessed 2025-11-16)
- Friston, K. et al. (2017). "Active Inference: A Process Theory." https://activeinference.github.io/papers/process_theory.pdf (accessed 2025-11-16)
- Parr, T. & Friston, K. (2018). "Precision and False Perceptual Inference." *Frontiers in Integrative Neuroscience*, 12:39. https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2018.00039/full (accessed 2025-11-16)
- Schwartenbeck, P. et al. (2015). "The Dopaminergic Midbrain Encodes the Expected Certainty about Desired Outcomes." *Cerebral Cortex*, 25(10), 3434-3445. PMC4585497. (accessed 2025-11-16)

**Resource-Rational Analysis**:
- Lieder, F. & Griffiths, T.L. (2020). "Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources." *Behavioral and Brain Sciences*, 43, e1. https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A (accessed 2025-11-16)
- Bhui, R. & Gershman, S.J. (2021). "Resource-rational decision making." *Current Opinion in Behavioral Sciences*, 41, 15-21. https://www.sciencedirect.com/science/article/abs/pii/S2352154621000371 (accessed 2025-11-16)
- Callaway, F. et al. (2022). "Rational use of cognitive resources in human planning." *Nature Human Behaviour*, 6, 1112–1125. https://escholarship.org/uc/item/11j0r4gj (accessed 2025-11-16)

### Additional References

- Parr, T. et al. (2019). "Perceptual awareness and active inference." *Neuroscience of Consciousness*, 2019(1), niz012. https://academic.oup.com/nc/article/2019/1/niz012/5566576
- Limanowski, J. (2024). "The Many Roles of Precision in Action." *Entropy*, 26(9), 790. https://www.mdpi.com/1099-4300/26/9/790
