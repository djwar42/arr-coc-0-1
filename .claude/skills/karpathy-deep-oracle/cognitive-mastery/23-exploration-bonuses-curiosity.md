# Exploration Bonuses & Curiosity: Intrinsic Motivation for Discovery

## Overview

**Exploration bonuses** and **curiosity-driven learning** provide intrinsic rewards that motivate agents to discover novel states and acquire diverse skills without external task-specific rewards. This document covers count-based exploration, prediction-based curiosity, empowerment, and their integration with reinforcement learning for efficient discovery in sparse-reward environments.

**Core Principle**: Agents receive intrinsic rewards for visiting novel states or reducing prediction uncertainty, balancing exploration (discovering new knowledge) with exploitation (using known strategies).

**Relationship to ARR-COC-0-1**: Token allocation as an exploration problem where patches compete for computational resources. Curiosity bonuses guide attention to informative regions, empowerment maximizes future representational options.

---

## Section 1: Count-Based Exploration Methods

### 1.1 Visit Count Bonuses

**Core Idea**: Reward agent for visiting infrequently seen states.

From [GitHub - Intrinsic Reward Motivation](https://github.com/mhngu23/Intrinsic-Reward-Motivati-Reinforcement-Learning-Re-Implementation) (accessed 2025-11-16):
> "This method uses a count-based approach to reward the agent for visiting less frequent states. A bonus is given to the agent based on the inverse of the visit count."

**Classical Formulation**:
```
Intrinsic Reward = 1 / sqrt(N(s))
```
where `N(s)` = number of times state `s` visited.

**Multi-Armed Bandit Connection**:
From [Surprise-Adaptive Intrinsic Motivation](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_77.pdf) (accessed 2025-11-16):
> "The UCB algorithm adds a count-based exploration bonus to the current value estimate of an action before selecting the maximum valued arm."

UCB (Upper Confidence Bound):
```
UCB(a) = Q(a) + c * sqrt(log(t) / N(a))
```
where `N(a)` = times action `a` taken, `t` = total timesteps, `c` = exploration constant.

**Advantages**:
- Simple, interpretable
- Provably efficient in tabular MDPs
- Works well in discrete, low-dimensional spaces

**Limitations**:
- Doesn't scale to continuous/high-dimensional spaces
- Requires state discretization (loses precision)
- Counts can overflow in long episodes

### 1.2 Pseudo-Counts via Density Models

**Problem**: How to count visits in continuous state spaces?

From [Unifying Count-Based Exploration](https://dl.acm.org/doi/10.5555/3157096.3157262) (accessed 2025-11-16):
> "We apply our ideas to Atari 2600 games, providing sensible pseudo-counts from raw pixels. We transform these pseudo-counts into exploration bonuses."

**Method**: Train density model `ρ(s)` to estimate state visitation frequency.

**Pseudo-Count Formula**:
```
N'(s) = ρ(s) * (1 - ρ(s)) / (ρ'(s) - ρ(s))
```
where `ρ(s)` = density before visit, `ρ'(s)` = density after visit.

**Intrinsic Reward**:
```
r_intrinsic = β / sqrt(N'(s))
```

**Density Model Choices**:
- Context Tree Switching (CTS) for discrete observations
- PixelCNN for image observations
- Normalizing Flows for continuous states

**Application**: Atari games where raw pixels → pseudo-counts → exploration bonuses.

### 1.3 State Discretization Trade-offs

**Challenge**: Balancing granularity vs counts.

From [The Impact of Intrinsic Rewards](https://link.springer.com/article/10.1007/s00521-025-11340-0) (accessed 2025-11-16):
> "State Count performs best for low-dimensional observations, while Maximum Entropy is more robust."

**Discretization Strategies**:
1. **Grid-based**: Divide continuous space into uniform bins
2. **Adaptive**: Use k-means clustering, decision trees
3. **Learned**: Train encoder to map states → discrete codes

**Trade-off**:
- Fine discretization → sparse counts, slow exploration
- Coarse discretization → overlapping states, imprecise bonuses

---

## Section 2: Prediction-Based Curiosity

### 2.1 Forward Dynamics Prediction Error

**Core Idea**: Reward agent for encountering unpredictable states.

From [MIT Neural Computation - Intrinsic Rewards](https://direct.mit.edu/neco/article/36/9/1854/123686/Intrinsic-Rewards-for-Exploration-Without-Harm) (accessed 2025-11-16):
> "Prediction error curiosity rewards agents for discovering observations they cannot accurately predict. However, such agents may be attracted to stochastic processes."

**ICM (Intrinsic Curiosity Module)**:
```
Forward Model: s_t+1_pred = f(s_t, a_t)
Prediction Error: e_t = ||s_t+1 - s_t+1_pred||^2
Intrinsic Reward: r_intrinsic = η * e_t
```

**From [Curiosity-Driven Exploration (Medium)](https://medium.com/biased-algorithms/curiosity-driven-exploration-in-reinforcement-learning-dd3f7d263fce) (accessed 2025-11-16)**:
> "The agent assigns itself an intrinsic reward based on the prediction error — basically how far off its guess was from what actually happened."

**Advantages**:
- Works in high-dimensional spaces (images, sensory data)
- Naturally scales with state complexity
- Encourages systematic exploration

**The Noisy-TV Problem**:
From MIT paper:
> "An agent equipped with prediction error curiosity may be attracted to purely stochastic noise (white noise TV screen) because it's unpredictable."

**Solution**: Separate **aleatoric** (environmental randomness) from **epistemic** (model uncertainty).

### 2.2 Inverse Dynamics for Feature Learning

**Problem**: Raw state prediction error includes uncontrollable noise.

**ICM Solution**: Learn feature representation `φ(s)` through inverse dynamics.

```
Inverse Model: a_t_pred = g(φ(s_t), φ(s_t+1))
Loss: L_inverse = ||a_t - a_t_pred||^2
```

**Insight**: Features useful for predicting actions filter out uncontrollable noise.

**Forward Model in Feature Space**:
```
φ(s_t+1)_pred = f(φ(s_t), a_t)
Intrinsic Reward: r_intrinsic = ||φ(s_t+1) - φ(s_t+1)_pred||^2
```

**Why This Works**:
- Inverse model forces `φ` to retain only action-relevant information
- Stochastic TV pixels don't predict actions → filtered out
- Agent-controllable dynamics emphasized

### 2.3 Random Network Distillation (RND)

**Alternative**: Use prediction error of random fixed network.

From [Exploring Meta-Learned Curiosity](https://iclr-blogposts.github.io/2024/blog/exploring-meta-learned-curiosity-algorithms/) (accessed 2025-11-16):
> "Intrinsic rewards are usually predictive errors. For instance, an RL agent equipped with a world model uses prediction error as curiosity."

**RND Architecture**:
```
Fixed Random Network: f_target(s) (never trained)
Predictor Network: f_pred(s) (trained online)
Intrinsic Reward: r_intrinsic = ||f_target(s) - f_pred(s)||^2
```

**Why Random Targets?**:
- Novel states → large prediction error (high bonus)
- Familiar states → low prediction error (low bonus)
- No need for forward/inverse dynamics models

**Advantage**: Simpler than ICM, avoids noisy-TV problem naturally.

### 2.4 Curiosity in Vision Hierarchies

From [Curiosity-Driven Hierarchical Vision](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009245) (accessed 2025-11-16):
> "Mainstream approaches enhance exploration by deriving intrinsic rewards from prediction errors between states. By integrating hierarchical representations, exploration becomes more structured."

**Hierarchical Curiosity**:
```
Level 1 (low): Predict pixels → rewards low-level novelty
Level 2 (mid): Predict edges/textures → rewards structural patterns
Level 3 (high): Predict objects/concepts → rewards semantic novelty
```

**Benefit**: Different curiosity signals at different abstraction levels guide multi-scale exploration.

---

## Section 3: Empowerment as Intrinsic Motivation

### 3.1 Information-Theoretic Definition

From [PRX Life - Intrinsic Motivation in Dynamical Systems](https://journals.aps.org/prxlife/abstract/10.1103/PRXLife.2.033009) (accessed 2025-11-16):
> "Empowerment is defined as the mutual information between potential actions and subsequent future states. This corresponds to maximizing the diversity of future world states achievable as a result of chosen actions."

**Formal Definition**:
```
Empowerment(s) = max_{p(a)} I(A; S_future | s)
```
where `I(A; S_future)` = mutual information between action sequence `A` and resulting future state.

**Interpretation**:
- High empowerment → many diverse, controllable futures
- Agent seeks states where actions have maximal influence
- Not entropy of passive diffusion, but **intentional control**

From PRX Life:
> "It is not enough to have diverse end states, but these must have been induced by the actions. Variability only counts if it can be specifically caused by the agent."

### 3.2 Empowerment vs Entropy

**Key Distinction**:

| Measure | What It Captures |
|---------|------------------|
| **Entropy H(S_future)** | Diversity of all possible futures (including random noise) |
| **Empowerment I(A; S_future)** | Diversity specifically caused by agent's actions |

From PRX Life:
> "If only the end state entropy were important, an agent would be induced to seek out, say, staying in front of a white-noise TV screen. However, unless the 'pixels' are controllable by the agent, this white noise would not contribute to empowerment."

**Example: Inverted Pendulum**:
- **Up position**: High empowerment (kick left/right → diverse controllable futures)
- **Down position**: Low empowerment (gravity dominates, actions have limited effect)

### 3.3 Computing Empowerment via Channel Capacity

From PRX Life (Tiomkin et al., 2024):
> "In the linear-response regime, calculating empowerment reduces to computing the channel capacity of a linear Gaussian channel."

**For continuous systems**:
```
Empowerment = sum_i log(1 + λ_i * P_i / σ^2)
```
where:
- `λ_i` = singular values of sensitivity matrix (∂S_future/∂A)
- `P_i` = power allocated to channel `i` (water-filling)
- `σ^2` = observation noise variance

**Sensitivity Matrix**:
```
M = ∂S_future / ∂A
```
captures how actions propagate through system dynamics.

**Connection to Lyapunov Exponents**:
From PRX Life:
> "The logarithm of eigenvalues of M reduces to the usual characteristic Lyapunov exponents... we refer to the log-spectrum as the control Lyapunov exponents."

**Empowerment → seeks states with high sensitivity to control.**

### 3.4 Empowerment Maximization for Control

**Algorithm** (from PRX Life):
```
1. Current state: s_t
2. For each candidate action a:
   - Compute sensitivity matrix M(s_t, a)
   - Calculate empowerment via channel capacity
3. Select action with max expected future empowerment
4. Execute action, observe next state
5. Repeat
```

**Applications**:
- Balancing inverted pendula without reward
- Cart-pole swing-up
- Double pendulum stabilization

From PRX Life results:
> "The controller manages to balance inverted pendula without extrinsic rewards, and without fine-tuning the control strategy to the dynamical equations."

---

## Section 4: Exploration vs Exploitation Trade-off

### 4.1 The Fundamental Dilemma

From [Medium - Day 92: Exploration vs Exploitation](https://medium.com/@sebuzdugan/day-92-100-exploration-vs-exploitation-balancing-curiosity-and-control-95c697d52da6) (accessed 2025-11-16):
> "One of the most foundational challenges in reinforcement learning — the exploration vs. exploitation dilemma."

**Trade-off**:
- **Exploration**: Try new actions to discover better strategies (curiosity-driven)
- **Exploitation**: Use known good strategies to maximize immediate reward

**Without Exploration**: Agent gets stuck in local optima (e.g., always taking safe action).

**Without Exploitation**: Agent wastes time on random actions, never converges to good policy.

### 4.2 Curiosity as Exploration Driver

From [CDE: Curiosity-Driven Exploration](https://arxiv.org/pdf/2509.09675) (accessed 2025-11-16):
> "Efficient exploration is a central challenge in Reinforcement Learning, which aims to balance between exploration and exploitation. Curiosity-driven methods provide intrinsic rewards that guide exploration."

**Curiosity Bonus Effect**:
```
Total Reward = r_extrinsic + β * r_curiosity
```

**Early Training** (β high):
- Curiosity dominates → agent explores widely
- Discovers states, transitions, skills

**Late Training** (β low or annealed):
- Extrinsic reward dominates → agent exploits knowledge
- Converges to task-specific policy

**Adaptive β Schedule**:
```
β(t) = β_0 / (1 + α * t)
```
gradually shifts from exploration to exploitation.

### 4.3 Count-Based vs Prediction-Based Trade-offs

From [The Impact of Intrinsic Rewards](https://link.springer.com/article/10.1007/s00521-025-11340-0) (accessed 2025-11-16):
> "State Count performs best for low-dimensional observations, while Maximum Entropy is more robust. DIAYN (Diversity is All You Need) encourages skill discovery."

**Comparison**:

| Method | Best For | Weakness |
|--------|----------|----------|
| **Count-Based** | Discrete, low-dim states | Doesn't scale to continuous |
| **Prediction Error** | High-dim, continuous | Attracted to stochastic noise |
| **Empowerment** | Controllable dynamics | Computationally expensive |

**Recommendation**: Use prediction-based with learned features (ICM) or random targets (RND) for high-dimensional problems.

### 4.4 Meta-Learning Curiosity

From [Exploring Meta-Learned Curiosity](https://iclr-blogposts.github.io/2024/blog/exploring-meta-learned-curiosity-algorithms/) (accessed 2025-11-16):
> "Neuroscience shows the brain balances exploration and exploitation through meta-learning, which relies on repeating tasks, memory, and rewards."

**Meta-Learned Curiosity**:
- Train curiosity function itself via meta-RL
- Learn when to explore vs exploit across task distribution
- Adapt curiosity bonuses to task structure

**Example**: Agent learns "explore more in sparse-reward tasks, less in dense-reward tasks."

---

## Section 5: Engineering Implementations

### 5.1 FSDP for Distributed Curiosity

**Influenced by File 4**: [FSDP vs DeepSeek](../distributed-training/03-fsdp-vs-deepspeed.md)

**Challenge**: Computing curiosity bonuses for large-scale agents (vision transformers, LLMs).

**FSDP Strategy**:
```
Shard 1: Compute prediction errors for image patches 1-100
Shard 2: Compute prediction errors for image patches 101-200
...
All-Reduce: Aggregate intrinsic rewards across shards
```

**Benefit**: Scale curiosity to millions of states without memory bottleneck.

**Implementation**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

curiosity_model = FSDP(ForwardDynamicsModel(...))
predictions = curiosity_model(states, actions)
prediction_errors = (predictions - next_states).pow(2).mean(dim=-1)
intrinsic_rewards = prediction_errors  # distributed computation
```

### 5.2 torch.compile for Fast Curiosity Computation

**Influenced by File 8**: [torch.compile & AOT Inductor](../inference-optimization/03-torch-compile-aot-inductor.md)

**Problem**: Curiosity modules add overhead (forward model, inverse model, RND predictor).

**Solution**: Compile curiosity computation with `torch.compile`.

```python
@torch.compile
def compute_curiosity_bonus(states, actions, next_states):
    # Forward dynamics
    pred_next = forward_model(states, actions)
    pred_error = (pred_next - next_states).pow(2).mean(dim=-1)

    # Inverse dynamics (optional)
    pred_actions = inverse_model(states, next_states)
    inv_error = (pred_actions - actions).pow(2).mean(dim=-1)

    return pred_error + 0.1 * inv_error

# 2-3x speedup over eager mode
curiosity_rewards = compute_curiosity_bonus(s, a, s_next)
```

**Benefit**: Real-time curiosity for high-frequency control (e.g., robotics at 30 Hz).

### 5.3 TPU Deployment for Curiosity-Driven RL

**Influenced by File 16**: [TPU Programming Fundamentals](../alternative-hardware/03-tpu-programming-fundamentals.md)

**Use Case**: Train millions of curiosity-driven agents in parallel (OpenAI Five, AlphaStar scale).

**TPU Strategy**:
```
# Each TPU core runs 128 parallel environments
for core in tpu_cores:
    states = envs[core].reset()
    for step in range(episode_length):
        curiosity_bonuses = curiosity_model(states)  # on TPU
        actions = policy(states, curiosity_bonuses)  # on TPU
        states, rewards = envs.step(actions)
        total_rewards = rewards + beta * curiosity_bonuses
        # Update policy with total_rewards
```

**TPU Advantage**: Matrix multiply-heavy (prediction models) → 10x faster than GPU for large batches.

### 5.4 Production Deployment Patterns

**Influenced by**: Files 4, 8, 16 (FSDP, torch.compile, TPU)

**Real-World Curiosity Pipeline**:
```
1. Data Collection (distributed):
   - 1000 parallel envs on FSDP cluster
   - Collect (s, a, s') transitions with curiosity bonuses

2. Model Training (compiled):
   - torch.compile(policy_network)
   - torch.compile(curiosity_network)
   - Train on collected data

3. Deployment (TPU inference):
   - Serve policy on TPU for low-latency decisions
   - Compute curiosity bonuses on-device for robotics
```

**Monitoring**: Track exploration diversity via state visitation entropy.

---

## Section 6: Recent Advances (2024-2025)

### 6.1 LLM Reasoning with Count-Based Curiosity

From [arXiv - Motivating Exploration in LLM Reasoning](https://arxiv.org/html/2510.16614v2) (accessed 2025-11-16):
> "We study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards)."

**Method**:
- LLM generates reasoning traces (chain-of-thought)
- Count unique reasoning patterns
- Reward LLM for generating novel reasoning strategies

**Result**: Improves MATH, GSM8K performance by 3-7% via exploration of diverse solution paths.

### 6.2 Curiosity for Embodied AI

From [Curiosity-Driven Embodied Learning](https://www.sciencedirect.com/science/article/abs/pii/S1566253525009443) (accessed 2025-11-16):
> "The proposed curiosity map informs a dual-protocol exploration policy that strategically balances semantic curiosity exploitation, which focuses on meaningful discovery."

**Cross-Modal Curiosity**:
- Visual curiosity: Prediction error on image features
- Semantic curiosity: Prediction error on object labels/descriptions
- Joint curiosity: I(visual; semantic | actions)

**Application**: Robots learn object manipulation by exploring visual-semantic surprises.

### 6.3 Biological Inspiration: Extinction Burst

From [bioRxiv - Extinction Burst Explained by Curiosity](https://www.biorxiv.org/content/10.1101/2024.08.28.610088v1) (accessed 2025-11-16):
> "We built a reinforcement learning model incorporating curiosity, defined as expected reward prediction errors, and the model successfully reproduced extinction burst behavior."

**Extinction Burst**: When reward stops, animal initially increases response rate (explores alternatives).

**Curiosity Explanation**:
- Reward cessation → high prediction error
- Curiosity drives increased exploration
- Eventually prediction error reduces → behavior extinguishes

**Implication**: Curiosity bonuses explain non-monotonic learning curves in animals.

---

## Section 7: Continuous Evaluation & Debugging

**Influenced by**: Vertex AI continuous evaluation patterns

**Metrics for Curiosity-Driven Learning**:

```python
# State coverage
unique_states_visited = len(set(state_hashes))
coverage_ratio = unique_states_visited / total_possible_states

# Novelty over time
novelty_curve = [mean(curiosity_bonuses[t]) for t in timesteps]

# Exploration efficiency
reward_per_unique_state = total_extrinsic_reward / unique_states_visited
```

**Debugging Pathologies**:
1. **Curiosity Collapse**: All states become familiar → zero bonus
   - **Fix**: Increase network capacity, use RND instead of forward dynamics

2. **Noisy-TV Attraction**: Agent stuck at stochastic source
   - **Fix**: Use inverse dynamics features, or empowerment instead

3. **Local Exploration**: Agent revisits nearby states only
   - **Fix**: Increase β, use count-based instead of prediction-based

**Logging & Visualization**:
```python
wandb.log({
    "curiosity_bonus_mean": curiosity_bonuses.mean(),
    "state_visit_counts_hist": wandb.Histogram(visit_counts),
    "empowerment_landscape": wandb.Image(empowerment_heatmap)
})
```

---

## Section 8: ARR-COC-0-1 Integration (10%)

### 8.1 Token Allocation as Exploration Problem

**ARR-COC-0-1 Challenge**: Allocate 64-400 tokens per patch based on relevance.

**Framing as Exploration**:
```
State: Current visual patch features
Action: Token budget allocation (64, 128, 256, 400)
Reward (extrinsic): Query-relevant information gain
Reward (intrinsic): Curiosity about patch contents
```

**Why Curiosity Helps**:
- Early training: Explore diverse patch types (textures, edges, objects)
- Learn which patches generalize across queries
- Discover query-independent visual patterns

### 8.2 Prediction Error for Patch Novelty

**Implementation**:
```python
class PatchCuriosityScorer:
    def __init__(self):
        self.forward_model = PatchPredictor()  # predicts patch_{t+1} from patch_t

    def compute_curiosity(self, patch_features):
        # Predict next patch in sequence
        pred_next = self.forward_model(patch_features[:-1])
        prediction_error = (pred_next - patch_features[1:]).pow(2).mean()

        # Bonus for unpredictable patches
        curiosity_bonus = eta * prediction_error
        return curiosity_bonus
```

**Use Case**: Prioritize novel visual regions (unusual textures, rare objects) for detailed encoding.

### 8.3 Empowerment for Token Budget Flexibility

**Empowerment Interpretation**: How much future representational choice does allocating tokens to this patch provide?

```python
def empowerment_for_patch(patch, token_budget):
    # Measure: I(token_budget; future_query_success | patch)
    # High empowerment → token allocation strongly influences future query handling

    sensitivity_matrix = compute_gradient(
        future_accuracy,
        wrt=token_budget,
        at=patch
    )

    singular_values = torch.svd(sensitivity_matrix).S
    empowerment = (singular_values.log1p()).sum()  # channel capacity
    return empowerment
```

**Insight**: Patches with high empowerment (e.g., salient objects, query-relevant regions) get more tokens.

### 8.4 Exploration vs Exploitation in Training

**Training Schedule**:
```
Epoch 1-10: β_curiosity = 1.0
  - Explore all patch types equally
  - Learn general visual features

Epoch 11-30: β_curiosity = 0.5
  - Balance novelty with query relevance
  - Specialize to common query patterns

Epoch 31+: β_curiosity = 0.1
  - Exploit learned relevance scorers
  - Fine-tune token allocation policy
```

**Evaluation**: Compare curiosity-driven vs standard training on novel query types.

---

## Sources

**Source Documents:**
- None (this topic uses only web research)

**Web Research:**
- [MERCI: Count-Based Curiosity for LLMs](https://arxiv.org/html/2510.16614v2) - arXiv:2510.16614 (accessed 2025-11-16)
- [Intrinsic Rewards Without Harm](https://direct.mit.edu/neco/article/36/9/1854/123686) - MIT Neural Computation (accessed 2025-11-16)
- [PRX Life - Intrinsic Motivation in Dynamical Systems](https://journals.aps.org/prxlife/abstract/10.1103/PRXLife.2.033009) - Tiomkin et al. (2024), American Physical Society (accessed 2025-11-16)
- [Unifying Count-Based Exploration](https://dl.acm.org/doi/10.5555/3157096.3157262) - ACM Digital Library (accessed 2025-11-16)
- [The Impact of Intrinsic Rewards](https://link.springer.com/article/10.1007/s00521-025-11340-0) - Springer (accessed 2025-11-16)
- [GitHub - Intrinsic Reward Implementations](https://github.com/mhngu23/Intrinsic-Reward-Motivati-Reinforcement-Learning-Re-Implementation) - Implementation examples (accessed 2025-11-16)
- [Curiosity-Driven Exploration (Medium)](https://medium.com/biased-algorithms/curiosity-driven-exploration-in-reinforcement-learning-dd3f7d263fce) - Tutorial (accessed 2025-11-16)
- [CDE: Curiosity-Driven Exploration](https://arxiv.org/pdf/2509.09675) - arXiv:2509.09675 (accessed 2025-11-16)
- [Hierarchical Vision Curiosity](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009245) - ScienceDirect (accessed 2025-11-16)
- [Exploring Meta-Learned Curiosity](https://iclr-blogposts.github.io/2024/blog/exploring-meta-learned-curiosity-algorithms/) - ICLR Blogposts 2024 (accessed 2025-11-16)
- [Extinction Burst via Curiosity](https://www.biorxiv.org/content/10.1101/2024.08.28.610088v1) - bioRxiv (accessed 2025-11-16)
- [Embodied Cross-Modal Curiosity](https://www.sciencedirect.com/science/article/abs/pii/S1566253525009443) - ScienceDirect (accessed 2025-11-16)

**Technical References:**
- Files 4, 8, 16 from Karpathy Deep Oracle (FSDP, torch.compile, TPU programming)

**Additional References:**
- [Surprise-Adaptive Intrinsic Motivation](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_77.pdf) - RLJ 2024 (accessed 2025-11-16)
- [Medium - Exploration vs Exploitation](https://medium.com/@sebuzdugan/day-92-100-exploration-vs-exploitation-balancing-curiosity-and-control-95c697d52da6) - Sebastian Buzdugan (accessed 2025-11-16)
