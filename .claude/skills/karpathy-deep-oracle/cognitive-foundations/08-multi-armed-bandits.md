# Multi-Armed Bandits: Exploration vs Exploitation

## Overview

The **multi-armed bandit (MAB)** problem is a classic framework for studying the exploration-exploitation tradeoff in sequential decision-making. Named after slot machines (one-armed bandits), it models scenarios where an agent must repeatedly choose between multiple options (arms) with unknown reward distributions, learning which arms are best while simultaneously maximizing cumulative reward.

**Core Challenge**: Balance between:
- **Exploration**: Try different arms to learn their reward distributions
- **Exploitation**: Pull the arm that currently appears best based on knowledge

From [Wikipedia - Exploration-exploitation dilemma](https://en.wikipedia.org/wiki/Exploration–exploitation_dilemma) (accessed 2025-11-14):
> "The exploration–exploitation dilemma, also known as the explore–exploit tradeoff, is a fundamental concept in decision-making that arises in many domains. It is depicted as the balancing act between two opposing strategies. Exploitation involves choosing the best option based on current knowledge of the system (which may be incomplete or misleading), while exploration involves trying out new options that may lead to better outcomes in the future."

**Relevance to ARR-COC-0-1**: Dynamic token allocation (64-400 tokens per patch) is fundamentally a bandit problem - allocate more tokens to patches that appear relevant (exploit) while exploring other patches that might become relevant.

---

## Section 1: The Multi-Armed Bandit Problem

### Formal Definition

At each time step t = 1, 2, ..., T:
1. Agent selects an arm a_t ∈ {1, 2, ..., K}
2. Environment reveals reward r_t ~ P_a(r) from that arm's distribution
3. Agent observes r_t and updates its beliefs

**Objective**: Maximize cumulative reward ∑_{t=1}^T r_t

**Key Properties**:
- K arms with unknown reward distributions
- Rewards are stochastic (sampled from probability distributions)
- Agent learns distributions through interaction
- No state transitions (unlike full RL)

### The Regret Framework

**Regret** measures how much reward is lost compared to always pulling the optimal arm:

R(T) = T·μ* - ∑_{t=1}^T μ_{a_t}

Where:
- μ* = max_a μ_a (best arm's mean reward)
- μ_{a_t} = mean reward of arm pulled at time t
- Lower regret = better algorithm

From [Cornell University - Regret Bounds for Sleeping Experts and Bandits](https://www.cs.cornell.edu/~rdk/papers/colt08.pdf) (accessed 2025-11-14):
> "We present optimal (up to a constant factor) algorithms for both the best expert and the multi-armed bandit versions with O(√(KT log K)) regret bounds."

**Regret Decomposition**:
- Regret from suboptimal arms = ∑_a Δ_a · E[N_a(T)]
- Where Δ_a = μ* - μ_a (gap from optimal)
- N_a(T) = number of times arm a pulled by time T

**Goal**: Design algorithms with **sublinear regret** O(√T) or O(log T)

---

## Section 2: Classical Bandit Algorithms

### Epsilon-Greedy

**Strategy**: With probability ε explore randomly, with probability 1-ε exploit best arm

```
At time t:
  With probability ε:
    Pull random arm
  With probability 1-ε:
    Pull arg max_a Q_t(a)  # estimated mean reward
```

**Properties**:
- Simple to implement
- Regret: O(T^(2/3)) with optimal ε decay
- Problem: Continues exploring suboptimal arms even when optimal is clear

**Variants**:
- Fixed ε: Constant exploration (linear regret)
- Decaying ε_t = 1/t: Reduces exploration over time
- Adaptive ε: Based on confidence intervals

### Upper Confidence Bound (UCB)

**Principle**: "Optimism in the face of uncertainty" - choose arms with highest upper confidence bound

**UCB1 Algorithm**:
```
Q_t(a) = μ_t(a) + √(2 ln t / N_t(a))
```

Where:
- μ_t(a) = empirical mean reward of arm a
- N_t(a) = number of times arm a pulled
- Confidence term decreases as arm is pulled more

**Key Insight**: UCB balances exploitation (μ_t) and exploration (confidence term) automatically without hyperparameters

From [SciTePress - A Study on Multi-Arm Bandit Problem with UCB and Thompson Sampling](https://www.scitepress.org/Papers/2024/129384/129384.pdf) (accessed 2025-11-14):
> "Upper Confidence Bound algorithm and Thompson Sampling algorithm are widely used for great performance. UCB strategically selects actions that balance between exploiting the current knowledge and exploring less-tried options."

**Regret Bound**: O(√(KT log T)) - near-optimal

**UCB Variants**:
- UCB-V: Uses variance estimates
- UCB-Tuned: Adaptive exploration
- KL-UCB: Uses KL-divergence for tighter bounds

### Thompson Sampling (Bayesian Approach)

**Principle**: Probability matching - sample arms proportional to probability they're optimal

**Algorithm**:
```
For each arm a:
  Maintain posterior distribution p(μ_a | data)
At time t:
  Sample θ_a ~ p(μ_a | data) for each arm
  Pull arm a_t = arg max_a θ_a
```

**For Bernoulli Bandits** (rewards in {0,1}):
- Prior: Beta(α_a, β_a) for each arm
- Update: Win → α_a += 1, Loss → β_a += 1
- Sample: θ_a ~ Beta(α_a, β_a)

**Properties**:
- Incorporates prior knowledge naturally
- Often outperforms UCB empirically
- Regret: O(√(KT)) - optimal
- Adapts exploration based on uncertainty

From [Performance Comparison and Analysis of UCB, ETC, Thompson Sampling](https://drpress.org/ojs/index.php/HSET/article/view/20588) (accessed 2025-11-14):
> "Thompson sampling often outperforms UCB and ε-greedy in practice. The Bayesian approach naturally incorporates uncertainty and adapts exploration strategies."

**Advantages**:
- No hyperparameters (unlike ε-greedy)
- Automatic uncertainty quantification
- Works with any prior/likelihood combination

---

## Section 3: Contextual Bandits

### Extension to Contextual Decision-Making

**Contextual Bandit**: At each time t, observe context x_t before choosing arm

Applications:
- Personalized recommendations (context = user features)
- Medical treatment selection (context = patient info)
- Online advertising (context = user/page features)

**Formal Setting**:
1. Observe context x_t ∈ X
2. Choose arm a_t ∈ A
3. Receive reward r_t ~ P(r | x_t, a_t)
4. Goal: Learn policy π: X → A

From [Kameleoon - Understanding contextual bandits](https://www.kameleoon.com/blog/contextual-bandits) (accessed 2025-11-14):
> "A contextual bandit is a machine learning approach that blends reinforcement learning principles with contextual insights to optimize decisions. Unlike traditional multi-armed bandits that treat all situations identically, contextual bandits use individual customer data to personalize decisions, asking 'what works best for this person, right now?'"

### LinUCB (Linear Contextual Bandits)

**Model**: Assume reward is linear in context-arm features
```
E[r_t | x_t, a] = x_t^T θ_a
```

**Algorithm**:
- Maintain estimate θ̂_a for each arm
- Use ridge regression with confidence intervals
- Choose: arg max_a (x_t^T θ̂_a + α·√(x_t^T A_a^(-1) x_t))

**Properties**:
- Handles high-dimensional contexts
- Regret: O(d√(T log T)) where d = feature dimension
- Used in news article recommendation (Yahoo, Google)

### Neural Contextual Bandits

**Modern Approach**: Use neural networks to model reward function

From [arXiv - Neural Contextual Bandits for Personalized Recommendation](https://dl.acm.org/doi/10.1145/3589335.3641241) (accessed 2025-11-14):
> "This tutorial investigates the contextual bandits as a powerful framework for personalized recommendations, using neural networks to handle complex, high-dimensional contexts."

**Architecture**:
- Neural network: f_θ(x, a) → predicted reward
- Uncertainty estimation via:
  - Ensemble methods
  - Bayesian neural networks
  - Bootstrap sampling

**Advantages**:
- Captures non-linear relationships
- Shares representations across arms
- Scales to millions of contexts/arms

---

## Section 4: Regret Bounds and Theoretical Analysis

### Lower Bounds (Fundamental Limits)

**Lai-Robbins Lower Bound** (1985):
- Any consistent algorithm must have:
  ```
  R(T) ≥ ∑_{a: Δ_a > 0} (Δ_a · log T) / KL(P_a || P*)
  ```
- Where KL is Kullback-Leibler divergence
- Depends on difficulty of distinguishing suboptimal arms

**Implications**:
- Logarithmic regret O(log T) is optimal for structured problems
- Square-root regret O(√T) is optimal for adversarial/unstructured problems
- Gap-dependent vs gap-independent bounds

From [MIT - High-Probability Regret Bounds for Bandit Online Linear Optimization](https://www.mit.edu/~rakhlin/papers/bandit_merged.pdf) (accessed 2025-11-14):
> "The present work closes the gap between full information and bandit online optimization against the adaptive adversary in terms of the growth of regret."

### Upper Bounds (Algorithm Performance)

**UCB Regret**:
- O(√(KT log T)) for worst-case
- O(∑_a (log T)/Δ_a) gap-dependent bound

**Thompson Sampling Regret**:
- O(√(KT)) Bayesian regret
- O(∑_a (log T)/Δ_a) frequentist bound

**Key Result**: Both UCB and Thompson Sampling are near-optimal

### Sample Complexity

**Question**: How many pulls needed to identify best arm with high probability?

**Fixed-Confidence**: Given δ, find arm within ε of optimal with probability 1-δ
- Sample complexity: O((K/ε²) log(1/δ))

**Fixed-Budget**: Given T pulls, minimize error probability
- Optimal allocation: Pull arms proportionally to 1/Δ_a²

From [JMLR - Information Capacity Regret Bounds for Bandits with Mediator Feedback](https://jmlr.org/papers/volume25/24-0227/24-0227.pdf) (accessed 2025-11-14):
> "We propose a method for generating simulated contextual bandit environments for personalization tasks, with information-theoretic regret bounds that depend on the capacity of the feedback mechanism."

---

## Section 5: Bayesian Bandits and Prior Knowledge

### Bayesian Framework

**Key Idea**: Maintain posterior distribution over arm parameters

**Bayes' Rule Update**:
```
p(θ | data) ∝ p(data | θ) · p(θ)
posterior = likelihood × prior
```

**Advantages**:
- Incorporates prior knowledge
- Natural uncertainty quantification
- Optimal exploration via posterior sampling

### Conjugate Priors for Common Distributions

**Bernoulli Rewards** (binary outcomes):
- Prior: Beta(α, β)
- Posterior after s successes, f failures: Beta(α+s, β+f)

**Gaussian Rewards** (continuous):
- Prior: Normal-Gamma on (μ, τ)
- Posterior: Conjugate normal-gamma update

**Poisson Rewards** (count data):
- Prior: Gamma(α, β)
- Posterior: Gamma(α+∑counts, β+n)

### Gittins Index

**Optimal Policy for Discounted Infinite-Horizon**:
- Each arm has a "Gittins index" G_a(state)
- Optimal to pull arm with highest index
- Index represents "calibrated fair reward"

**Properties**:
- Provably optimal for discounted reward
- Computationally tractable via dynamic programming
- Generalizes to restless bandits

---

## Section 6: Non-Stationary and Adversarial Bandits

### Non-Stationary Bandits

**Problem**: Reward distributions change over time
- Concept drift in recommendation systems
- Market dynamics in financial trading
- User preference evolution

**Solutions**:
- **Sliding window**: Only use recent observations
- **Discounted UCB**: Weight recent data more heavily
  ```
  μ_t(a) = ∑ γ^(t-s) r_s / ∑ γ^(t-s)
  ```
- **Change-point detection**: Detect distribution shifts

### Adversarial Bandits

**Setting**: Rewards chosen adversarially (worst-case)
- No stochastic assumptions
- Reward can depend on algorithm's past choices

**Exp3 Algorithm** (Exponential-weight algorithm for Exploration and Exploitation):
```
Probability of arm a: p_a = (1-γ) · exp(η·G_a)/Z + γ/K
```
- Maintains cumulative gain estimates G_a
- Exploration rate γ, learning rate η

**Regret**: O(√(KT log K)) - optimal for adversarial setting

From [AAAI - Regret Bounds for Batched Bandits](https://ojs.aaai.org/index.php/AAAI/article/view/16901) (accessed 2025-11-14):
> "We present simple algorithms for batched stochastic multi-armed bandit and batched stochastic linear bandit problems. We prove bounds for their expected regret in both adversarial and stochastic settings."

---

## Section 7: Applications and Practical Considerations

### A/B Testing and Online Experimentation

**Traditional A/B Testing**:
- Fixed allocation (50/50 split)
- Run until statistical significance
- Then deploy winner

**Bandit-Based Testing**:
- Adaptive allocation favors better variants
- Reduces opportunity cost during experiment
- Continuous learning and optimization

From [Optimizely - Contextual bandits: The next step in personalization](https://www.optimizely.com/insights/blog/contextual-bandits-in-personalization/) (accessed 2025-11-14):
> "Contextual bandits automatically learn which experiences work best for different audiences, providing valuable insights while maximizing conversions. Unlike traditional A/B tests with fixed allocation, bandits adapt in real-time."

### Recommendation Systems

**Problem**: Recommend items (articles, products, ads) to maximize engagement

**Bandit Formulation**:
- Arms = candidate items
- Context = user features, page context
- Reward = click, purchase, engagement time

**Deployed Systems**:
- Yahoo News: LinUCB for article recommendation
- Google: Neural bandits for ad selection
- Netflix: Contextual bandits for content ranking

### Resource Allocation

**Compute Allocation**:
- Arms = different compute configurations
- Reward = performance/cost ratio
- Explore hardware/software combinations

**Budget Allocation**:
- Marketing channels as arms
- Reward = conversion rate
- Optimize spending across channels

**Medical Treatment Selection**:
- Arms = treatment options
- Context = patient characteristics
- Reward = treatment outcome
- Ethical constraints on exploration

---

## Section 8: ARR-COC-0-1 Token Allocation as Multi-Armed Bandit

### Mapping ARR-COC-0-1 to Bandits

**The ARR-COC-0-1 Token Allocation Problem**:
- **Arms**: Different patches in the image (spatial locations)
- **Context**: Query + patch visual features + salience scores
- **Action**: Token budget allocation (64, 128, 192, 256, 320, 384, 400 tokens)
- **Reward**: Relevance to query (measured by three ways of knowing)

From [john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md](../source-documents/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Opponent Processing: Navigate tensions between competing constraints. Example: Compression ↔ Particularization. Dynamic balance, not fixed trade-off... [Balancing] → Navigate tensions (Compress↔Particularize, Exploit↔Explore, Focus↔Diversify)"

**Opponent Processing as Bandit Problem**:
- **Exploit**: Allocate more tokens to patches with high current relevance scores
- **Explore**: Try different token budgets on uncertain patches that might become relevant

### Dynamic Token Budget Optimization

**Contextual Bandit Formulation**:

**Context** x_t for patch p at time t:
- Query embedding q
- Patch visual features v_p
- Propositional score (Shannon entropy)
- Perspectival score (salience/attention)
- Participatory score (query-patch coupling)

**Arms** a ∈ {64, 128, 192, 256, 320, 384, 400}:
- Discrete token budget levels
- Higher budgets = more detailed encoding
- Lower budgets = more compression

**Reward** r(x_t, a):
- Post-hoc relevance assessment
- Could be measured by:
  - Downstream task performance
  - Human relevance judgments
  - Change in integrated relevance score after processing

**Policy** π(a | x_t):
- Neural network mapping context to token budget
- Trained via bandit feedback

### UCB for Patch Selection

**UCB Strategy for ARR-COC-0-1**:
```python
def select_patches_ucb(patches, query, t):
    scores = []
    for p in patches:
        # Exploitation: Current relevance estimate
        Q_p = estimate_relevance(p, query)

        # Exploration: Confidence bonus (inverse of visit count)
        confidence = sqrt(2 * log(t) / visit_count[p])

        # UCB score
        ucb_score = Q_p + confidence
        scores.append(ucb_score)

    # Allocate more tokens to high UCB patches
    return allocate_tokens_by_scores(patches, scores)
```

**Properties**:
- Patches with high relevance get more tokens (exploit)
- Patches rarely examined get exploration bonus
- Automatically balances compression vs particularization

### Thompson Sampling for Token Budget

**Bayesian Approach**:
```python
def thompson_sampling_tokens(patch, query):
    # Posterior over relevance for each token budget
    posteriors = {
        64: Beta(α_64, β_64),
        128: Beta(α_128, β_128),
        # ... other budgets
    }

    # Sample from each posterior
    samples = {k: p.sample() for k, p in posteriors.items()}

    # Choose budget with highest sample
    return max(samples.items(), key=lambda x: x[1])[0]
```

**Update After Observing Relevance**:
- High relevance → increase α for that budget
- Low relevance → increase β for that budget
- Posterior concentrates on effective budgets

### Multi-Level Bandit Problem

**Hierarchical Decision**:

1. **Patch-Level Bandit**: Which patches to encode?
   - Arms = patches in image
   - Explore different spatial regions

2. **Budget-Level Bandit**: How many tokens per patch?
   - Arms = token budgets {64, 128, ..., 400}
   - Explore compression-accuracy tradeoff

3. **Feature-Level Bandit**: Which features to extract?
   - Arms = different feature extractors (RGB, LAB, Sobel, etc.)
   - Explore different knowing dimensions

**Cascading Bandits**: Decisions are sequential and dependent

### Learned Relevance Allocation Policy

**Training with Bandit Feedback**:

**Offline Phase** (Supervised):
- Train on datasets with ground-truth relevance
- Learn initial policy π_θ(budget | patch, query)

**Online Phase** (Bandit Learning):
- Deploy policy and collect feedback
- Observe actual downstream performance
- Update policy via gradient bandit methods:
  ```
  ∇_θ J = E[∇_θ log π_θ(a|x) · (r - baseline)]
  ```

**Advantages**:
- Adapts to actual usage patterns
- Learns from implicit feedback (clicks, task success)
- No need for exhaustive labeled relevance data

### Exploration Strategies for ARR-COC-0-1

**Epsilon-Greedy Exploration**:
- With probability ε: Random token allocation
- With probability 1-ε: Use learned policy
- Decay ε over training

**Boltzmann Exploration**:
- Sample budgets proportional to exp(Q/τ)
- Temperature τ controls exploration
- Softer than epsilon-greedy

**Intrinsic Motivation** (Curiosity-Driven):
- Reward model prediction error
- Explore patches where relevance model is uncertain
- Similar to active learning

**Connection to Vervaekean Framework**:
From relevance realization concepts: The explore-exploit tension is precisely the "Exploit↔Explore" opponent process in Vervaeke's framework. The bandit algorithm realizes this tension by:
- Quantifying uncertainty (how much we know about each patch)
- Balancing immediate reward (exploit known relevant patches) vs long-term learning (explore uncertain patches)
- Adapting allocation based on transjective coupling between query and patches

---

## Section 9: Advanced Topics and Extensions

### Restless Bandits

**Problem**: Arm states evolve even when not pulled
- Stock prices change whether traded or not
- Product popularity evolves over time

**Whittle Index**:
- Generalization of Gittins index
- Approximately optimal for large K
- Tractable via relaxation

### Combinatorial Bandits

**Problem**: Choose subset of arms simultaneously
- Allocate budget across multiple patches
- Select features for multiple patches
- Combinatorial action space

**Semi-Bandit Feedback**:
- Observe reward for each selected arm
- More information than full bandit
- Enables better learning

### Dueling Bandits

**Setting**: Compare pairs of arms (relative feedback)
- User prefers A over B (no absolute rewards)
- Applications: Ranking, recommendation

From [PMLR - Regret Lower Bound and Optimal Algorithm in Dueling Bandit](https://proceedings.mlr.press/v40/Komiyama15.html) (accessed 2025-11-14):
> "The proposed algorithm is found to be the first one with a regret upper bound that matches the lower bound. Experimental comparisons of dueling bandit algorithms demonstrate superior performance."

### Bandits with Delayed Feedback

**Problem**: Reward observed after delay
- Medical trials (treatment effect delayed)
- Online ads (conversion happens later)

**Strategies**:
- Wait for feedback before updating
- Use optimistic estimates for pending rewards
- Model delay distribution

### Safe Exploration

**Constraint**: Never pull arms with reward below threshold
- Medical safety constraints
- Financial risk limits

**Approaches**:
- Conservative UCB (use lower confidence bound)
- Constrained Thompson sampling
- Safe exploration via Gaussian processes

---

## Sources

**Source Documents:**
- [john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md](../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md) - Opponent processing (Exploit↔Explore tension)

**Web Research:**
- [Wikipedia - Exploration-exploitation dilemma](https://en.wikipedia.org/wiki/Exploration–exploitation_dilemma) (accessed 2025-11-14) - Core concepts and decision theory
- [Hugging Face Deep RL Course - The Exploration/Exploitation trade-off](https://huggingface.co/learn/deep-rl-course/en/unit1/exp-exp-tradeoff) (accessed 2025-11-14) - RL perspective on exploration
- [SciTePress - A Study on Multi-Arm Bandit Problem with UCB and Thompson Sampling](https://www.scitepress.org/Papers/2024/129384/129384.pdf) (accessed 2025-11-14) - Algorithm comparison
- [Cornell University - Regret Bounds for Sleeping Experts and Bandits](https://www.cs.cornell.edu/~rdk/papers/colt08.pdf) (accessed 2025-11-14) - Theoretical regret bounds
- [MIT - High-Probability Regret Bounds for Bandit Online Linear Optimization](https://www.mit.edu/~rakhlin/papers/bandit_merged.pdf) (accessed 2025-11-14) - Advanced theory
- [PMLR - Regret Lower Bound and Optimal Algorithm in Dueling Bandit](https://proceedings.mlr.press/v40/Komiyama15.html) (accessed 2025-11-14) - Optimal algorithms
- [AAAI - Regret Bounds for Batched Bandits](https://ojs.aaai.org/index.php/AAAI/article/view/16901) (accessed 2025-11-14) - Batched settings
- [JMLR - Information Capacity Regret Bounds for Bandits with Mediator Feedback](https://jmlr.org/papers/volume25/24-0227/24-0227.pdf) (accessed 2025-11-14) - Information-theoretic bounds
- [Kameleoon - Understanding contextual bandits](https://www.kameleoon.com/blog/contextual-bandits) (accessed 2025-11-14) - Contextual bandits for personalization
- [ACM - Neural Contextual Bandits for Personalized Recommendation](https://dl.acm.org/doi/10.1145/3589335.3641241) (accessed 2025-11-14) - Neural approaches
- [Optimizely - Contextual bandits in personalization](https://www.optimizely.com/insights/blog/contextual-bandits-in-personalization/) (accessed 2025-11-14) - Real-world applications

**Additional Resources:**
- [GeeksforGeeks - Exploitation and Exploration in Machine Learning](https://www.geeksforgeeks.org/machine-learning/exploitation-and-exploration-in-machine-learning/) (accessed 2025-11-14)
- [Towards Data Science - An Overview of Contextual Bandits](https://towardsdatascience.com/an-overview-of-contextual-bandits-53ac3aa45034/) (accessed 2025-11-14)
