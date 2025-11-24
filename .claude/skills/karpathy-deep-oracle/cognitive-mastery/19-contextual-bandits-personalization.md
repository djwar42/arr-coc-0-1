# Contextual Bandits & Personalization: Query-Aware Resource Allocation

## Overview

**Contextual bandits** extend the multi-armed bandit framework by incorporating contextual information (side information, features) to make personalized, context-dependent decisions. Unlike standard MAB where rewards depend only on the arm pulled, contextual bandits observe a context vector before each decision and select actions based on both historical rewards and current context.

**Core Innovation**: Context-aware action selection enables personalized allocation - different contexts (users, queries, states) receive different optimal actions.

From [Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation"](https://arxiv.org/abs/1003.0146) (2010):
> "In typical recommendation scenarios, we not only want to optimize overall reward, but also want to personalize recommendations for individual users. Contextual bandits provide a principled framework for this personalization."

**Relevance to ARR-COC-0-1**: Token allocation (64-400 tokens per patch) is fundamentally a contextual bandit problem - allocate more tokens to patches based on **context** (the query + image content), learning which contexts benefit most from high-resolution processing.

From [True Theta - LinUCB Tutorial](https://truetheta.io/concepts/reinforcement-learning/lin-ucb/) (accessed 2025-11-16):
> "When a user arrives at a website, they are represented as a feature vector. This may measure their gender, geography, age, device, or anything else considered relevant. Also, each recommendable article is represented with a feature vector. This may measure the article's topic, category, geography and date. Together, the user's and articles' features are considered the 'context' of the recommendation."

---

## Section 1: Contextual Bandit Problem Formulation (~100 lines)

### 1.1 Formal Definition

At each time step t = 1, 2, ..., T:

1. **Context Arrival**: Agent observes context x_t ∈ X (feature vector describing current state)
2. **Action Selection**: Based on context and history, agent selects action a_t ∈ A
3. **Reward Observation**: Agent receives reward r_t(a_t) ~ P(r|x_t, a_t)
4. **Update**: Agent updates policy using (x_t, a_t, r_t) triplet

**Key Properties**:
- **Context-dependent rewards**: E[r_t(a)|x_t] varies with context
- **Personalization**: Optimal action a*(x_t) = argmax_a E[r|x_t, a] depends on context
- **No state transitions**: Each decision is independent (unlike full RL)
- **Partial feedback**: Only observe reward for selected action

From [Neural Contextual Bandits Tutorial](https://www2024.thewebconf.org/docs/tutorial-slides/neural-contextual-bandits.pdf) (WWW 2024):
> "Contextual bandits bridge the gap between stateless multi-armed bandits and full reinforcement learning, providing a sweet spot for personalized decision-making in applications like recommendation, advertising, and content optimization."

### 1.2 Context Types

**User Context** (shared features):
- Demographics: age, gender, location
- Historical behavior: click history, purchase patterns
- Device/platform: mobile vs desktop, browser
- Temporal: time of day, day of week

**Action Context** (arm features):
- Article features: topic, length, author, freshness
- Product features: price, category, brand
- Ad features: advertiser, creative type, bid amount

**Hybrid Context**:
- User-action interactions: user affinity for topic
- Cross-features: user location × product availability
- Computed features: predicted CTR, similarity scores

From [Bietti et al., "A Contextual Bandit Bake-off"](https://arxiv.org/abs/1802.04064) (2021):
> "Real-world contextual bandit applications often involve hundreds to thousands of features combining user attributes, item characteristics, and interaction signals. Feature engineering is as critical as algorithm selection."

### 1.3 Comparison with MAB

| Aspect | Multi-Armed Bandit | Contextual Bandit |
|--------|-------------------|-------------------|
| **Context** | None | Feature vector x_t |
| **Optimal Action** | Fixed (same for all rounds) | Context-dependent a*(x_t) |
| **Personalization** | One-size-fits-all | Personalized per context |
| **Sample Efficiency** | Learn K arm means | Learn function f(x,a) → r |
| **Generalization** | None | Generalizes to new contexts |

---

## Section 2: LinUCB Algorithm (~120 lines)

### 2.1 Linear Contextual Bandits

**Assumption**: Expected reward is linear in context features:

E[r_t(a)|x_t,a] = x_t,a^T θ_a*

Where:
- x_t,a ∈ R^d: context features for action a at time t
- θ_a* ∈ R^d: unknown true parameter vector for action a

**LinUCB Algorithm** (Li et al. 2010):
```
For each action a:
  A_a = λI  (d×d identity matrix, regularization)
  b_a = 0_d (d-dimensional zero vector)

At round t:
  Observe context x_t,a for each action a

  For each action a:
    θ_a = A_a^(-1) b_a  (ridge regression estimate)
    UCB_a = x_t,a^T θ_a + α√(x_t,a^T A_a^(-1) x_t,a)

  Select a_t = argmax_a UCB_a
  Observe reward r_t

  Update selected action:
    A_a_t = A_a_t + x_t,a_t x_t,a_t^T
    b_a_t = b_a_t + r_t x_t,a_t
```

From [True Theta - LinUCB](https://truetheta.io/concepts/reinforcement-learning/lin-ucb/) (2024):
> "LinUCB essentially runs K online ridge regressions, which bring convenient Bayesian computations of confidence intervals. The upper bound can be computed analytically and updated in each round."

### 2.2 Upper Confidence Bound Strategy

**Optimism Under Uncertainty**: Select action with highest upper confidence bound

UCB_a(x) = μ̂_a(x) + α·σ̂_a(x)

Where:
- μ̂_a(x): estimated expected reward (exploitation)
- α·σ̂_a(x): confidence width (exploration)
- α: exploration parameter (tunable hyperparameter)

**Intuition**: "Be optimistic about uncertain actions"
- High μ̂: Action seems good from past data → exploit
- High σ̂: Action is uncertain → explore
- UCB balances both automatically

**Confidence Width Interpretation**:
- σ̂_a(x) = √(x^T A_a^(-1) x) measures uncertainty
- Small when A_a is large (many samples in direction x)
- Large when A_a is small (few samples in direction x)

From [Strategic Linear Contextual Bandits](https://proceedings.neurips.cc/paper_files/paper/2024/file/d390199c28b467315b454789b6584f19-Paper-Conference.pdf) (NeurIPS 2024):
> "The UCB algorithm balances exploitation (choosing actions with high estimated rewards) and exploration (reducing uncertainty about action values) through a single principled formula."

### 2.3 Regret Analysis

**Regret**: Cumulative difference from optimal policy

R(T) = Σ_{t=1}^T (r_t* - r_t)

Where r_t* = max_a E[r|x_t, a] is optimal reward in round t

**LinUCB Regret Bound**:
R(T) = O(d√(T log T))

**Interpretation**:
- Sublinear in T (average regret → 0)
- Scales with feature dimension d
- Much better than random: O(T)
- Better than ε-greedy: O(T^(2/3))

From [Abbasi-Yadkori et al., "Improved Algorithms for Linear Stochastic Bandits"](https://proceedings.neurips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf) (NIPS 2011):
> "Linear contextual bandits achieve near-optimal regret rates when the linearity assumption holds, with regret growing only logarithmically in the number of rounds for fixed dimensions."

---

## Section 3: Neural Contextual Bandits (~120 lines)

### 3.1 Beyond Linearity: Deep Representations

**Limitation of LinUCB**: Linear models cannot capture complex nonlinear relationships

**Neural Bandits**: Use neural networks to model E[r|x,a]

μ_θ(x,a) = f_θ(x,a)

Where f_θ is a neural network with parameters θ

From [Neural Contextual Bandits Tutorial](https://www2024.thewebconf.org/docs/tutorial-slides/neural-contextual-bandits.pdf) (WWW 2024):
> "Neural contextual bandits leverage the representation power of deep neural networks to transform raw contextual data into effective features for decision-making."

### 3.2 Neural LinUCB Architecture

**Two-Stage Approach**:

1. **Representation Learning**: Neural network φ_θ(x,a) → z (learned features)
2. **Linear UCB**: LinUCB on learned features z

Combined Model:
```
φ_θ: (x,a) → z ∈ R^d  (neural encoder)
μ(x,a) = z^T w_a      (linear head per action)
```

**Advantages**:
- Captures nonlinear patterns via φ_θ
- Maintains LinUCB uncertainty quantification
- End-to-end differentiable

From [DeepLinUCB Paper](https://dl.acm.org/doi/10.1145/3543873.3587684) (WWW 2023):
> "DeepLinUCB leverages the representation power of deep neural network to transform the raw features and applies a linear contextual bandit algorithm on the last hidden layer."

### 3.3 Thompson Sampling for Neural Bandits

**Alternative to UCB**: Bayesian approach with posterior sampling

**Algorithm**:
```
For round t:
  Sample θ_a ~ p(θ_a | D_t) for each action a
  Compute μ̃_a(x_t) = f_θ_a(x_t, a)
  Select a_t = argmax_a μ̃_a(x_t)
  Observe reward r_t
  Update posterior p(θ_a | D_t+1)
```

**Advantages**:
- Naturally handles uncertainty via posterior
- Often more sample-efficient than UCB
- Flexible (works with any likelihood model)

From [Multi-Objective Neural Bandits](https://www.ijcai.org/proceedings/2025/0547.pdf) (IJCAI 2025):
> "Thompson Sampling provides a principled Bayesian approach to exploration, sampling actions according to their probability of being optimal given the current posterior beliefs."

### 3.4 Neural Architecture Choices

**Input Encoding**:
- Concatenation: [x_user; x_action]
- Factorized: φ_user(x_user) ⊙ φ_action(x_action)
- Cross-features: φ(x_user, x_action) with attention

**Output Heads**:
- Shared backbone + action-specific heads
- Disjoint networks per action
- Single network with action as input

**Uncertainty Estimation**:
- Dropout approximation (MC-Dropout)
- Ensembles (bootstrap aggregation)
- Explicit variance networks

From [Bandit Sampling for Neural NAS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224004557) (Neurocomputing 2024):
> "Neural architecture design for bandits must balance expressiveness (capturing complex patterns) with tractable uncertainty quantification (enabling principled exploration)."

---

## Section 4: Personalization Applications (~100 lines)

### 4.1 Content Recommendation

**Problem**: Which article/video/product to recommend to user?

**Context**:
- User: demographics, history, real-time behavior
- Items: content type, popularity, freshness
- Environment: time, device, session position

**Objective**: Maximize clicks, watch time, purchases

From [Recommender Systems using LinUCB](https://medium.com/data-science/recommender-systems-using-linucb-a-contextual-multi-armed-bandit-approach-35a6f0eb6c4) (Medium 2019):
> "LinUCB provides a natural framework for cold-start recommendations: new users/items can immediately benefit from context features rather than waiting to accumulate sufficient interaction data."

**Case Study - News Recommendation** (Li et al. 2010):
- Yahoo! Front Page Today Module
- Context: user features (location, history) + article features (category, age)
- Result: 12.5% CTR improvement over random
- Key: Personalization beats one-size-fits-all

### 4.2 Online Advertising

**Problem**: Which ad to show to maximize revenue?

**Context**:
- User: browsing history, demographics, intent signals
- Ad: advertiser, bid, creative, landing page
- Page: content, position, competing ads

**Objective**: Maximize click-through-rate × bid (revenue)

From [Constrained Contextual Bandits for Limited-Budget Allocation](https://www.sciencedirect.com/science/article/abs/pii/S0952197623017426) (Engineering Applications of AI 2024):
> "Online advertising introduces budget constraints: advertisers have daily spending limits. Contextual bandits must balance exploration-exploitation while respecting these constraints through Lagrangian relaxation."

### 4.3 Personalized Pricing

**Problem**: What price to offer to maximize profit?

**Context**:
- Customer: purchase history, loyalty tier, browsing time
- Product: demand elasticity, inventory, seasonality
- Market: competitor prices, time-sensitive promotions

**Objective**: Maximize (price - cost) × P(purchase|price, context)

**Dynamic Pricing Example**:
- Airline ticket pricing: seat class × booking time × route demand
- E-commerce: personalized discounts based on cart abandonment probability
- Ride-sharing: surge pricing based on supply-demand imbalance

From [Deep RL for Hyper-Personalized Recommendations](https://www.shaped.ai/blog/deep-learning-for-hyper-personalized-recommendations) (Shaped.ai 2025):
> "Multi-armed bandit algorithms balance exploration (suggesting new genres or titles) and exploitation (leveraging known user preferences) to continuously refine personalization strategies."

### 4.4 Medical Treatment Assignment

**Problem**: Which treatment to prescribe for patient?

**Context**:
- Patient: demographics, genetics, medical history, comorbidities
- Treatment: mechanism, side effects, contraindications
- Outcome: efficacy for specific condition

**Objective**: Maximize treatment efficacy, minimize adverse events

**Ethical Considerations**:
- Exploration must respect patient safety
- Constraints: never assign dominated treatments
- Interpretability: physicians need explainable recommendations

From [Contextual Restless Bandits for Medical Applications](https://arxiv.org/abs/2403.15640) (2024):
> "Medical treatment assignment introduces complexities beyond standard bandits: patient states evolve over time (restless), actions have long-term effects, and ethical constraints require conservative exploration."

---

## Section 5: Query-Aware Relevance Allocation (~100 lines)

### 5.1 Contextual Bandits for Attention

**Problem**: How to allocate limited computational budget across inputs?

**Query-Aware Formulation**:
- Context x_t: query representation + input features
- Actions a: budget allocation strategies (e.g., token counts per patch)
- Reward r: task performance (accuracy, F1, retrieval quality)

**ARR-COC Connection**: Token allocation IS contextual bandit
- Context: query embedding + image patch features → relevance scores
- Actions: token budgets {64, 128, 256, 400} per patch
- Reward: downstream task accuracy

From [Query-Aware Resource Allocation](https://www.semanticscholar.org/paper/ca25e449e26edff475e60c38b9ac5014e329326b) (accessed 2025-11-16):
> "Buffer pool aware query scheduling demonstrates that resource allocation can be learned via deep reinforcement learning, adapting dynamically to query characteristics and system state."

### 5.2 Personalized Token Budgets

**Standard VLM**: Fixed token budget per image (e.g., 256 tokens)

**Contextual Bandit VLM**: Dynamic budget based on query + image
```
Context: [query_emb; image_patch_features]
Actions: token_budgets = [64, 128, 256, 400]
Reward: accuracy on query-image pair

Policy: a*(x) = argmax_{budget} E[accuracy | query, image, budget]
```

**Learning Strategy**:
1. Initialize: Uniform exploration across budgets
2. For each (query, image) pair:
   - Extract context features
   - Select budget via LinUCB/Neural Bandit
   - Process with selected budget
   - Observe accuracy reward
   - Update policy
3. Converge: High-relevance patches → high budget, low-relevance → low budget

**Expected Gains**:
- 30-50% compute savings (vs fixed high budget)
- Maintained or improved accuracy (vs fixed low budget)
- Automatic adaptation to query types

From [Semantic-Aware Resource Allocation](https://arxiv.org/abs/2406.07996) (arXiv 2024):
> "Semantic-aware resource allocation based on deep reinforcement learning enables flexible, context-dependent resource assignment that outperforms fixed allocation strategies across diverse workloads."

### 5.3 Multi-Patch Allocation as Combinatorial Bandit

**Extension**: Allocate budgets to multiple patches simultaneously

**Combinatorial Contextual Bandit**:
- Context: query + all patch features
- Actions: budget vector [b_1, ..., b_N] where b_i ∈ {64, 128, 256, 400}
- Constraint: Σ b_i ≤ total_budget
- Reward: accuracy

**Challenges**:
- Exponential action space: 4^N combinations
- Requires structured exploration
- Credit assignment: which patch budgets contributed to reward?

**Solution Approaches**:
- Greedy allocation: Sequential patch-by-patch decisions
- Linear decomposition: Assume additive rewards r = Σ r_i(b_i)
- Neural combinatorial optimization: Learn set-to-set mapping

From [Efficient Deep RL Resource Allocation](https://dl.acm.org/doi/abs/10.1007/s10515-021-00318-6) (Automated Software Engineering 2022):
> "Resource allocation in fog networks requires balancing multiple objectives simultaneously. Deep reinforcement learning provides a framework for learning these complex allocation policies from interaction data."

---

## Section 6: Advanced Techniques (~100 lines)

### 6.1 Hybrid Models: LinUCB + Neural Features

**Architecture**:
```
Raw features x → Neural Encoder φ_θ(x) → z → LinUCB(z)
```

**Advantages**:
- Neural network: Learn complex feature representations
- LinUCB: Fast uncertainty quantification, closed-form updates
- Best of both worlds: expressiveness + tractable exploration

From [DeepLinUCB](https://dl.acm.org/doi/10.1145/3543873.3587684) (WWW 2023):
> "The key innovation is maintaining a linear model on top of deep representations, enabling efficient exploration through analytically computed confidence bounds while leveraging deep learning's representation power."

### 6.2 Off-Policy Evaluation

**Problem**: Evaluate new policy without deploying it

**Logged Data**: Historical decisions from production policy
D = {(x_t, a_t, r_t, π_0(a_t|x_t))}

**Inverse Propensity Scoring (IPS)**:
```
V̂(π) = (1/T) Σ_t (π(a_t|x_t) / π_0(a_t|x_t)) r_t
```

**Challenges**:
- High variance when policies differ substantially
- Requires accurate logging of π_0(a|x)
- Variance reduction: capping, normalization, doubly robust estimators

From [Transfer Learning for Contextual Bandits](https://projecteuclid.org/journals/annals-of-statistics/volume-52/issue-1/Transfer-learning-for-contextual-multi-armed-bandits/10.1214/23-AOS2341.full) (Annals of Statistics 2024):
> "Off-policy evaluation enables safe policy improvement: we can assess new algorithms on historical data before deploying them, reducing risk of performance degradation in production systems."

### 6.3 Fairness in Personalization

**Problem**: Personalization can amplify biases

**Fairness Constraints**:
- **Demographic parity**: P(a|sensitive_feature=0) = P(a|sensitive_feature=1)
- **Equal opportunity**: P(reward=1|a, sensitive=0) = P(reward=1|a, sensitive=1)
- **Individual fairness**: Similar individuals receive similar treatment

From [Fairness in Federated Contextual Bandits](https://arxiv.org/abs/2402.03531) (2024):
> "Federated contextual bandits must balance personalization, fairness, and privacy. Differential privacy mechanisms ensure individual data protection while fairness constraints prevent discrimination."

### 6.4 Non-Stationarity & Concept Drift

**Challenge**: User preferences and content change over time

**Strategies**:
- **Sliding window**: Only use recent data (last N rounds)
- **Exponential forgetting**: Weight recent data more: w_t = exp(-λ(T-t))
- **Change detection**: Detect distribution shifts, reset when detected
- **Meta-learning**: Learn to adapt quickly to new distributions

From [Analysis of Deep RL Algorithms for Resource Allocation](https://www.mdpi.com/1424-8220/25/17/5286) (Sensors 2025):
> "Real-world systems exhibit non-stationarity: user behavior changes, content shifts, system dynamics evolve. Contextual bandit algorithms must incorporate adaptive mechanisms to track these changes."

---

## Section 7: Integration with FSDP, ML Workloads, TPU (~80 lines)

### 7.1 Distributed Training of Contextual Bandit Policies (File 4: FSDP)

**Challenge**: Large-scale contextual bandits with millions of features

From [karpathy/distributed-training/03-fsdp-vs-deepspeed.md](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md):
> "PyTorch FSDP enables training models that cannot fit on a single GPU by sharding parameters, gradients, and optimizer states across workers."

**Application to Neural Bandits**:
```python
# Neural contextual bandit with FSDP
from torch.distributed.fsdp import FullyShardedDataParallel

class NeuralBandit(nn.Module):
    def __init__(self, context_dim, action_dim, hidden_dim=512):
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(action_dim)
        ])

    def forward(self, context):
        z = self.encoder(context)
        return [head(z) for head in self.action_heads]

# Wrap with FSDP for distributed training
model = FullyShardedDataParallel(
    NeuralBandit(context_dim=10000, action_dim=1000),
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

**Why FSDP for Bandits**:
- Handle massive context dimensions (10K+ features)
- Scale to millions of actions (product catalogs)
- Train on billions of logged interactions

**Memory Savings**:
- Standard: Full model on each GPU → 32GB per GPU
- FSDP: Sharded model → 4GB per GPU (8-way sharding)

### 7.2 Production ML Workload Patterns (File 12: K8s ML Workloads)

From [karpathy/orchestration/03-ml-workload-patterns-k8s.md](../karpathy/orchestration/03-ml-workload-patterns-k8s.md):
> "ML workloads on Kubernetes require specialized scheduling patterns: batch jobs for training, recurring tasks for model updates, gang scheduling for distributed training."

**Contextual Bandit Training Pipeline**:
```yaml
# Daily model retraining job
apiVersion: batch/v1
kind: CronJob
metadata:
  name: bandit-model-retrain
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: contextual-bandit-trainer:latest
            command: ["python", "train.py"]
            args:
            - --data-path=/mnt/logs/yesterday
            - --model-output=/mnt/models/bandit-v1
            resources:
              requests:
                nvidia.com/gpu: 4
                memory: "64Gi"
              limits:
                nvidia.com/gpu: 4
                memory: "128Gi"
```

**Real-Time Serving Pattern**:
```yaml
# Bandit policy serving deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bandit-serving
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: server
        image: bandit-server:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
```

**Benefits**:
- Automated retraining: Model stays fresh
- Horizontal scaling: Handle high traffic
- Fault tolerance: Crashed pods restart automatically

### 7.3 TPU-Accelerated Bandit Training (File 16: TPU Fundamentals)

From [karpathy/alternative-hardware/03-tpu-programming-fundamentals.md](../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md):
> "TPUs are essentially matrix multiplication machines connected to fast memory. Understanding this simple model is crucial for writing efficient TPU code."

**Why TPUs for Contextual Bandits**:
- Batch matrix operations: Compute UCBs for all actions simultaneously
- High-throughput inference: Serve thousands of requests/second
- Cost-efficiency: Lower cost per FLOP than GPUs for matmul-heavy workloads

**LinUCB on TPU**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def linucb_select_action(contexts, A_inv_list, b_list, alpha):
    """
    Vectorized LinUCB action selection on TPU

    Args:
        contexts: [num_actions, context_dim]
        A_inv_list: [num_actions, context_dim, context_dim]
        b_list: [num_actions, context_dim]
        alpha: exploration parameter
    """
    # Compute all theta estimates: [num_actions, context_dim]
    theta_list = jnp.einsum('ijk,ik->ij', A_inv_list, b_list)

    # Compute all expected rewards: [num_actions]
    mu_list = jnp.einsum('ij,ij->i', contexts, theta_list)

    # Compute all uncertainties: [num_actions]
    # sigma^2 = x^T A^-1 x
    sigma_list = jnp.sqrt(
        jnp.einsum('ij,ijk,ik->i', contexts, A_inv_list, contexts)
    )

    # UCB = mu + alpha * sigma
    ucb_list = mu_list + alpha * sigma_list

    return jnp.argmax(ucb_list)

# Batch processing for high throughput
@jax.jit
def batch_linucb(batch_contexts, A_inv_list, b_list, alpha):
    """Process batch of contexts [batch_size, num_actions, context_dim]"""
    return jax.vmap(
        lambda ctx: linucb_select_action(ctx, A_inv_list, b_list, alpha)
    )(batch_contexts)
```

**TPU Performance**:
- Single TPU v5e: 5e13 bf16 FLOPs/s
- Batch UCB computation: Process 10K contexts in <1ms
- Matrix inversions via Cholesky: Leverage MXU efficiency

---

## Section 8: ARR-COC-0-1 Integration (~60 lines)

### 8.1 Token Allocation as Contextual Bandit

**ARR-COC-0-1 Formulation**:

**Context**:
```python
context = {
    'query': query_embedding,  # [768] CLIP text embedding
    'patch_features': patch_features,  # [N_patches, 1024] visual features
    'relevance_scores': {
        'propositional': info_entropy,      # Information content
        'perspectival': salience_map,       # Visual salience
        'participatory': query_alignment    # Query-content coupling
    }
}
```

**Actions**:
```python
actions = {
    'patch_i': token_budget_i,  # ∈ {64, 128, 256, 400}
    for i in range(num_patches)
}
```

**Reward**:
```python
reward = accuracy on downstream task (e.g., VQA, retrieval)
```

**Policy Learning**:
```python
class ARRCOCBandit:
    def __init__(self):
        # Contextual bandit per patch
        self.patch_bandits = [LinUCB(dim=768+1024+3) for _ in range(K)]

    def allocate_tokens(self, query, image_patches):
        contexts = self.extract_contexts(query, image_patches)
        budgets = []

        for i, ctx in enumerate(contexts):
            # Select budget via UCB
            budget = self.patch_bandits[i].select_action(ctx)
            budgets.append(budget)

        return budgets

    def update(self, query, image_patches, budgets, accuracy):
        contexts = self.extract_contexts(query, image_patches)

        for i, (ctx, budget) in enumerate(zip(contexts, budgets)):
            # Update each patch's bandit
            self.patch_bandits[i].update(ctx, budget, accuracy)
```

### 8.2 Query-Aware Relevance Learning

**Transjective Relevance via Contextual Bandits**:

Traditional VLM: Fixed budget allocation
ARR-COC: **Context-dependent** budget allocation

**Example Scenarios**:

**Query: "What color is the car?"**
→ Context favors color-salient regions
→ Bandit learns: High budget to car patches, low to background

**Query: "How many people are in the scene?"**
→ Context favors spatial coverage
→ Bandit learns: Moderate budget across all person-containing patches

**Query: "What is the text on the sign?"**
→ Context favors high-resolution text regions
→ Bandit learns: Maximum budget to sign patches, minimal to others

**Learning Curve**:
- Early: Uniform exploration across budgets
- Mid: Identifies query-pattern correlations (text queries → high res)
- Late: Converges to query-specific allocation policies

### 8.3 Integration with Opponent Processing

**Balancing Tensions via Bandit Feedback**:

From ARR-COC balancing.py:
```python
# Compress ↔ Particularize
compression_score = -budget_cost
particularize_score = detail_preservation

# Exploit ↔ Explore
exploit_score = expected_accuracy  # from bandit
explore_score = uncertainty        # from bandit

# Navigate tensions using bandit UCBs
tension_balance = lambda: (
    compression_score + particularize_score,
    exploit_score + alpha * explore_score
)
```

**Feedback Loop**:
1. **Knowing**: Measure 3Ps relevance → context features
2. **Balancing**: Navigate tensions → propose budget range
3. **Attending**: Contextual bandit → select specific budget from range
4. **Realizing**: Execute allocation → observe accuracy
5. **Update**: Bandit update → improve future allocations

**Result**: Adaptive, query-aware token allocation that maximizes relevance realization under resource constraints.

---

## Sources

**Source Documents:**
- [karpathy/distributed-training/03-fsdp-vs-deepspeed.md](../karpathy/distributed-training/03-fsdp-vs-deepspeed.md) - FSDP for large-scale neural bandits
- [karpathy/orchestration/03-ml-workload-patterns-k8s.md](../karpathy/orchestration/03-ml-workload-patterns-k8s.md) - Production ML patterns for bandit serving
- [karpathy/alternative-hardware/03-tpu-programming-fundamentals.md](../karpathy/alternative-hardware/03-tpu-programming-fundamentals.md) - TPU acceleration for bandit computation

**Web Research:**
- Li, Chu, Langford, Schapire. [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146). WWW 2010 (accessed 2025-11-16)
- [True Theta - A Reliable Contextual Bandit Algorithm: LinUCB](https://truetheta.io/concepts/reinforcement-learning/lin-ucb/) (accessed 2025-11-16)
- [Neural Contextual Bandits for Personalized Recommendation](https://www2024.thewebconf.org/docs/tutorial-slides/neural-contextual-bandits.pdf). WWW 2024 Tutorial (accessed 2025-11-16)
- Shi et al. [Deep Neural Network with LinUCB: A Contextual Bandit Approach](https://dl.acm.org/doi/10.1145/3543873.3587684). WWW 2023 (accessed 2025-11-16)
- [Strategic Linear Contextual Bandits](https://proceedings.neurips.cc/paper_files/paper/2024/file/d390199c28b467315b454789b6584f19-Paper-Conference.pdf). NeurIPS 2024 (accessed 2025-11-16)
- Abbasi-Yadkori, Pal, Szepesvari. [Improved Algorithms for Linear Stochastic Bandits](https://proceedings.neurips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf). NIPS 2011 (accessed 2025-11-16)
- Bietti, Agarwal, Langford. [A Contextual Bandit Bake-off](https://arxiv.org/abs/1802.04064). arXiv 2021 (accessed 2025-11-16)
- Solanki et al. [Fairness and Privacy Guarantees in Federated Contextual Bandits](https://arxiv.org/abs/2402.03531). arXiv 2024 (accessed 2025-11-16)
- Chen et al. [Contextual Restless Multi-Armed Bandits with Application to Resource Allocation](https://arxiv.org/abs/2403.15640). arXiv 2024 (accessed 2025-11-16)
- Cai et al. [Transfer Learning for Contextual Multi-Armed Bandits](https://projecteuclid.org/journals/annals-of-statistics/volume-52/issue-1/Transfer-learning-for-contextual-multi-armed-bandits/10.1214/23-AOS2341.full). Annals of Statistics 2024 (accessed 2025-11-16)
- Shao et al. [Semantic-Aware Resource Allocation Based on Deep Reinforcement Learning](https://arxiv.org/abs/2406.07996). arXiv 2024 (accessed 2025-11-16)
- Lakhan et al. [Efficient Deep-Reinforcement Learning Aware Resource Allocation](https://dl.acm.org/doi/abs/10.1007/s10515-021-00318-6). Automated Software Engineering 2022 (accessed 2025-11-16)
- Ali et al. [Analysis of Deep Reinforcement Learning Algorithms for Resource Allocation](https://www.mdpi.com/1424-8220/25/17/5286). Sensors 2025 (accessed 2025-11-16)
- [Shaped.ai - Deep Learning for Hyper-Personalized Recommendations](https://www.shaped.ai/blog/deep-learning-for-hyper-personalized-recommendations) (accessed 2025-11-16)

**Additional References:**
- [Medium - Recommender Systems using LinUCB](https://medium.com/data-science/recommender-systems-using-linucb-a-contextual-multi-armed-bandit-approach-35a6f0eb6c4) (accessed 2025-11-16)
- [Constrained Contextual Bandits for Limited-Budget Allocation](https://www.sciencedirect.com/science/article/abs/pii/S0952197623017426). Engineering Applications of AI 2024 (accessed 2025-11-16)
