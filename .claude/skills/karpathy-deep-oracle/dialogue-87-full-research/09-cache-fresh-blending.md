# BATCH 9: Cache/Fresh Blending Research

## Thompson Sampling for Exploration-Exploitation

### The Multi-Armed Bandit Problem

Choose between:
- **Exploit:** Use best-known option
- **Explore:** Try uncertain options

### Thompson Sampling Algorithm

```python
class ThompsonSampling:
    def __init__(self, num_arms):
        # Beta distribution parameters
        self.alpha = np.ones(num_arms)  # Successes + 1
        self.beta = np.ones(num_arms)   # Failures + 1

    def select_arm(self):
        # Sample from posterior of each arm
        samples = [np.random.beta(a, b)
                  for a, b in zip(self.alpha, self.beta)]

        # Choose arm with highest sample
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### Why Thompson Sampling Works

1. **Natural exploration:** High uncertainty → high variance samples
2. **Automatic annealing:** As knowledge grows, variance shrinks
3. **Probability matching:** Plays arm in proportion to being best
4. **Theoretical guarantees:** Optimal regret bounds

## Cache vs Fresh Trade-off

### The Problem

- **Cache:** Fast, possibly stale
- **Fresh:** Accurate, slow to compute

### Staleness-Freshness Model

```python
class CacheFreshBlender:
    def __init__(self, cache_ttl=100, freshness_weight=0.5):
        self.cache_ttl = cache_ttl
        self.freshness_weight = freshness_weight
        self.cache = {}
        self.timestamps = {}

    def get(self, key, compute_fresh):
        if key in self.cache:
            age = current_time() - self.timestamps[key]
            staleness = age / self.cache_ttl

            # Blend based on staleness
            if staleness < 1.0:
                weight = self.freshness_weight * staleness
                cached = self.cache[key]

                if np.random.random() < weight:
                    # Compute fresh
                    fresh = compute_fresh()
                    self.cache[key] = fresh
                    self.timestamps[key] = current_time()
                    return fresh
                else:
                    return cached
            else:
                # Too stale, must refresh
                fresh = compute_fresh()
                self.cache[key] = fresh
                self.timestamps[key] = current_time()
                return fresh
        else:
            # Not in cache
            fresh = compute_fresh()
            self.cache[key] = fresh
            self.timestamps[key] = current_time()
            return fresh
```

## Adaptive Mixture Weight Learning

### The Ensemble Problem

Combine multiple models/predictions optimally:

```python
class AdaptiveMixture:
    def __init__(self, num_models):
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, model_predictions):
        weights = F.softmax(self.weights, dim=0)
        return sum(w * pred for w, pred in zip(weights, model_predictions))
```

### Bayesian Model Averaging

```python
def bayesian_average(predictions, likelihoods, priors):
    """
    predictions: [num_models, ...]
    likelihoods: P(data | model)
    priors: P(model)
    """
    # Posterior: P(model | data) ∝ P(data | model) × P(model)
    posteriors = likelihoods * priors
    posteriors = posteriors / posteriors.sum()

    # Weighted average
    return sum(p * pred for p, pred in zip(posteriors, predictions))
```

## Retrieval Augmented Generation (RAG)

### Blending Retrieved vs Generated

```python
class RAGBlender:
    def __init__(self, retriever, generator, blend_weight=0.3):
        self.retriever = retriever
        self.generator = generator
        self.blend_weight = blend_weight

    def forward(self, query):
        # Retrieve relevant documents
        retrieved = self.retriever(query)
        retrieval_answer = self.extract_answer(retrieved)

        # Generate answer
        generated_answer = self.generator(query)

        # Blend
        return self.blend_weight * retrieval_answer + \
               (1 - self.blend_weight) * generated_answer
```

### Adaptive RAG

```python
class AdaptiveRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.confidence_estimator = nn.Linear(hidden_dim, 1)

    def forward(self, query):
        # Get both
        retrieved = self.retriever(query)
        generated = self.generator(query)

        # Estimate confidence in retrieval
        retrieval_conf = self.confidence_estimator(retrieved).sigmoid()

        # Adaptive blend
        return retrieval_conf * retrieved + (1 - retrieval_conf) * generated
```

## Memory Consolidation Patterns

### Short-term vs Long-term Memory

```python
class MemoryConsolidation:
    def __init__(self, short_term_size, long_term_size):
        self.short_term = deque(maxlen=short_term_size)
        self.long_term = {}
        self.consolidation_threshold = 3

    def add(self, key, value):
        self.short_term.append((key, value))

        # Consolidate if seen multiple times
        if self.count_in_short_term(key) >= self.consolidation_threshold:
            self.long_term[key] = value

    def query(self, key):
        # Check long-term first (cached/consolidated)
        if key in self.long_term:
            return self.long_term[key]

        # Check short-term (fresh)
        for k, v in self.short_term:
            if k == key:
                return v

        return None
```

## Integration with Spicy Lentil

### Cache/Fresh for Cognitive Fingerprint

The user's fingerprint can be:
- **Cached:** Previous session's understanding
- **Fresh:** Current context updates

```python
class CognitiveCache:
    def __init__(self, user_id):
        self.cached_fingerprint = load_from_disk(user_id)
        self.current_session = []

    def get_fingerprint(self):
        # Blend cached (stable) with fresh (session)
        if self.current_session:
            session_features = process_session(self.current_session)
            blend_weight = compute_session_importance(self.current_session)
            return (1 - blend_weight) * self.cached_fingerprint + \
                   blend_weight * session_features
        else:
            return self.cached_fingerprint
```

### Thompson Sampling for Pathway Selection

Uncertain about which pathway to use? Sample!

```python
class PathwayBandit:
    def __init__(self, num_pathways=9):
        # Track success/failure per pathway
        self.alpha = np.ones(num_pathways)
        self.beta = np.ones(num_pathways)

    def select_pathways(self, k=4):
        # Sample from each pathway's posterior
        samples = [np.random.beta(a, b)
                  for a, b in zip(self.alpha, self.beta)]

        # Select top-k
        return np.argsort(samples)[-k:]

    def update(self, pathway, success):
        if success:
            self.alpha[pathway] += 1
        else:
            self.beta[pathway] += 1
```

### Saccade Planning as Exploration

```python
class SaccadeExplorer:
    """Thompson sampling for where to look next"""
    def __init__(self, num_regions):
        self.alpha = np.ones(num_regions)  # Information gained
        self.beta = np.ones(num_regions)   # Time wasted

    def plan_saccade(self):
        # Sample expected value of each region
        samples = [np.random.beta(a, b)
                  for a, b in zip(self.alpha, self.beta)]

        # Saccade to highest expected value region
        return np.argmax(samples)

    def update(self, region, info_gained):
        if info_gained > threshold:
            self.alpha[region] += 1
        else:
            self.beta[region] += 1
```

## Key Formulas

### Thompson Sampling Posterior
```
P(θ | data) ∝ P(data | θ) × P(θ)
```

### Beta Distribution (for binary outcomes)
```
Beta(α, β) where α = successes + 1, β = failures + 1
```

### Regret Bound
```
Regret(T) = O(√(KT log T))
```
Where K = number of arms, T = time steps

### Staleness Weight
```
w_fresh = min(1, age / ttl)
```

## Performance Insights

### Thompson Sampling Benefits

1. **Simple:** Easy to implement
2. **Optimal:** Near-optimal regret
3. **Adaptive:** Automatically balances explore/exploit
4. **Parallelizable:** Can batch samples

### Cache Benefits

- 10-1000x speedup for repeated queries
- Trade-off: staleness vs speed

### Ensemble Benefits

- Lower variance than single model
- More robust predictions

## Implementation Recommendations

1. **Start with Thompson:** Natural, principled exploration
2. **Cache aggressively:** But track staleness
3. **Adaptive blending:** Learn weights from feedback
4. **Monitor regret:** Track cumulative suboptimality

---

**Sources:**
- "A Tutorial on Thompson Sampling" - NeurIPS 2017
- "The Multi-Armed Bandit Problem and Its Solutions" - Lil'Log
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "An Information-Theoretic Approach to Cache Allocation"
