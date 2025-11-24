# Reinforcement Learning Fundamentals

**Created**: 2025-11-14
**Part of**: Cognitive Foundations & Research Methodology expansion
**Connection to ARR-COC-0-1**: Procedural knowing through interaction, learning relevance allocation strategies

---

## Overview

Reinforcement Learning (RL) is a paradigm where agents learn optimal behavior through interaction with an environment. Unlike supervised learning (learning from labeled examples) or unsupervised learning (finding patterns in data), RL learns from trial-and-error using reward signals. This directly connects to Vervaeke's **procedural knowing** - learning HOW to act through experience.

**Core RL Loop**:
```
Agent → Action → Environment → (New State, Reward) → Agent
```

The agent learns a **policy** π that maps states to actions, aiming to maximize cumulative reward over time.

---

## 1. RL Fundamentals: The Markov Decision Process (MDP)

### MDP Components

**Markov Decision Process (S, A, P, R, γ)**:
- **S**: State space (possible situations the agent can be in)
- **A**: Action space (possible actions the agent can take)
- **P**: Transition probability P(s'|s,a) - probability of reaching state s' from state s after action a
- **R**: Reward function R(s,a,s') - immediate reward for transition
- **γ**: Discount factor (0 < γ ≤ 1) - importance of future vs immediate rewards

**Markov Property**: Future state depends only on current state and action, not on history.

### Return and Value Functions

**Return** G_t (cumulative discounted reward from time t):
```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... = Σ_{k=0}^∞ γ^k * r_{t+k}
```

**State Value Function** V^π(s) - expected return from state s following policy π:
```
V^π(s) = E_π[G_t | s_t = s]
```

**Action Value Function** Q^π(s,a) - expected return from state s, taking action a, then following π:
```
Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
```

**Bellman Equations** (recursive relationship):
```
V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*V^π(s')]
Q^π(s,a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*Σ_{a'} π(a'|s')*Q^π(s',a')]
```

### Optimal Policy

**Goal**: Find optimal policy π* that maximizes expected return:
```
π* = argmax_π V^π(s) for all s
```

**Optimal Value Functions**:
```
V*(s) = max_π V^π(s)
Q*(s,a) = max_π Q^π(s,a)
```

**Bellman Optimality Equation**:
```
V*(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*V*(s')]
Q*(s,a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*max_{a'} Q*(s',a')]
```

---

## 2. Value-Based Methods: Q-Learning and DQN

### Tabular Q-Learning

**Q-Learning Algorithm** (Watkins & Dayan, 1992):
```
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize s
    For each step:
        Choose action a using ε-greedy policy from Q
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α*[r + γ*max_a' Q(s',a') - Q(s,a)]
        s ← s'
```

**Key Properties**:
- **Off-policy**: Learns optimal Q* while following exploratory policy (ε-greedy)
- **Model-free**: Doesn't require knowledge of P(s'|s,a) or R
- **Temporal Difference (TD)**: Updates based on bootstrapped estimate (next Q value)
- **Converges** to Q* under suitable conditions (all states visited infinitely, learning rate decay)

**Epsilon-Greedy Exploration**:
```
With probability ε: select random action (explore)
With probability 1-ε: select a = argmax_a Q(s,a) (exploit)
```

### Deep Q-Networks (DQN)

**Problem with Tabular Q-Learning**: Doesn't scale to large/continuous state spaces (e.g., images).

**Solution**: Use neural network to approximate Q(s,a; θ).

**DQN Algorithm** (Mnih et al., 2015 - DeepMind Atari):

**Key Innovations**:

1. **Experience Replay**:
   - Store transitions (s, a, r, s') in replay buffer D
   - Sample random minibatches for training
   - Breaks correlation between consecutive samples
   - Improves data efficiency through reuse

2. **Target Network**:
   - Maintain separate target network Q(s,a; θ⁻) with older parameters
   - Update target network periodically (e.g., every 10k steps)
   - Stabilizes training by preventing moving target problem

**DQN Loss Function** (Huber loss):
```
L(θ) = E_{(s,a,r,s')~D} [(y - Q(s,a; θ))²]

where y = r + γ*max_a' Q(s',a'; θ⁻)  (target value using target network)
```

**Huber Loss** (robust to outliers):
```
L_δ(x) = { ½x²         if |x| ≤ δ
         { δ(|x| - ½δ)  otherwise
```

**DQN Training Loop**:
```python
# From PyTorch DQN tutorial
for episode in episodes:
    state = env.reset()
    for t in range(max_steps):
        # ε-greedy action selection
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax_a Q(state, a; θ)

        next_state, reward, done = env.step(action)

        # Store transition in replay buffer
        replay_buffer.push(state, action, reward, next_state)

        # Sample minibatch and optimize
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)

            # Compute Q(s,a) for sampled transitions
            q_values = Q_network(batch.states).gather(1, batch.actions)

            # Compute target: r + γ*max_a' Q(s',a'; θ⁻)
            with torch.no_grad():
                next_q_values = target_network(batch.next_states).max(1)[0]
                targets = batch.rewards + gamma * next_q_values

            # Compute loss and update
            loss = huber_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Soft update target network
        target_network = τ*Q_network + (1-τ)*target_network

        state = next_state
        if done:
            break
```

**DQN Extensions** (2015-2024):

- **Double DQN**: Use policy network for action selection, target network for value evaluation (reduces overestimation)
  ```
  y = r + γ*Q(s', argmax_a' Q(s',a'; θ); θ⁻)
  ```

- **Dueling DQN**: Separate value V(s) and advantage A(s,a) streams:
  ```
  Q(s,a; θ) = V(s; θ_v) + (A(s,a; θ_a) - mean_a A(s,a'; θ_a))
  ```

- **Prioritized Experience Replay**: Sample transitions based on TD error magnitude (learn from surprising transitions)

- **Rainbow DQN** (2017): Combines 6 extensions (Double, Dueling, Prioritized, Multi-step, Distributional, Noisy Nets)

---

## 3. Policy Gradient Methods: REINFORCE, A2C, PPO

### Policy Gradient Fundamentals

**Direct Policy Optimization**: Instead of learning Q(s,a) and deriving policy, directly parameterize policy π(a|s; θ) and optimize it.

**Objective**: Maximize expected return J(θ):
```
J(θ) = E_{τ~π_θ}[R(τ)]  where τ is trajectory (s_0, a_0, r_0, s_1, a_1, r_1, ...)
```

**Policy Gradient Theorem** (Sutton et al., 2000):
```
∇_θ J(θ) = E_{τ~π_θ}[Σ_t ∇_θ log π(a_t|s_t; θ) * G_t]
```

**Intuition**: Increase probability of actions that led to high return, decrease probability of actions with low return.

### REINFORCE Algorithm

**REINFORCE** (Williams, 1992) - Monte Carlo policy gradient:

```
For each episode:
    Generate trajectory τ = (s_0, a_0, r_0, ..., s_T) using π(a|s; θ)
    For t = 0 to T:
        G_t = Σ_{k=t}^T γ^{k-t} * r_k  (return from time t)
        θ ← θ + α*∇_θ log π(a_t|s_t; θ) * G_t
```

**Variance Reduction - Baseline**:
```
∇_θ J(θ) = E[Σ_t ∇_θ log π(a_t|s_t; θ) * (G_t - b(s_t))]
```

Common baseline: b(s_t) = V(s_t) (value function estimate)

**Problems**:
- High variance (Monte Carlo)
- Sample inefficient (on-policy, can't reuse old data)
- Unstable training

### Actor-Critic Architecture

**Idea**: Combine value-based and policy-based methods.

**Two Networks**:
- **Actor**: π(a|s; θ_actor) - policy network
- **Critic**: V(s; θ_critic) - value network (estimates baseline)

**Advantage Function** A(s,a):
```
A(s,a) = Q(s,a) - V(s)  (how much better is action a than average)
```

**TD Error as Advantage Estimate**:
```
δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  (1-step TD error approximates advantage)
```

**Actor-Critic Update**:
```
Critic update: minimize (r_t + γ*V(s_{t+1}; θ_critic) - V(s_t; θ_critic))²
Actor update: θ_actor ← θ_actor + α*∇_θ log π(a_t|s_t; θ_actor) * δ_t
```

### A3C (Asynchronous Advantage Actor-Critic)

**A3C** (Mnih et al., 2016 - DeepMind):

**Key Innovation**: Multiple parallel workers exploring environment simultaneously.

**Benefits**:
- Decorrelated experience (different workers in different states)
- Faster training (parallelism)
- No replay buffer needed (on-policy but decorrelated by parallel exploration)

**Architecture**:
```
Global Network (parameters θ_global)
↓
Worker 1, Worker 2, ..., Worker N (copies of network)
- Each worker interacts with separate environment copy
- Accumulates gradients locally for n steps
- Asynchronously updates global network
- Syncs parameters from global network
```

**A2C** (Advantage Actor-Critic): Synchronous version of A3C (all workers update together).

### PPO (Proximal Policy Optimization)

**PPO** (Schulman et al., 2017 - OpenAI):

**Problem**: Policy gradient methods are sensitive to step size. Too large → policy collapses; too small → slow learning.

**Solution**: Constrain policy updates to "trust region" - prevent policy from changing too much per update.

**PPO-Clip Objective**:
```
r_t(θ) = π(a_t|s_t; θ) / π(a_t|s_t; θ_old)  (probability ratio)

L^CLIP(θ) = E[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
```

**Intuition**:
- If advantage A_t > 0 (good action): increase probability, but not more than (1+ε) times
- If advantage A_t < 0 (bad action): decrease probability, but not more than (1-ε) times
- Prevents destructively large policy updates

**PPO Algorithm**:
```
For iteration in iterations:
    Collect trajectories using current policy π(θ_old)
    Compute advantages A_t for all timesteps

    For epoch in K epochs:  # Multiple epochs on same data
        For minibatch in minibatches:
            Compute PPO-Clip loss
            Update θ using gradient ascent

    θ_old ← θ
```

**Why PPO is Popular** (2024 standard):
- Simple to implement
- Sample efficient (reuses data for K epochs)
- Stable training
- Works well across diverse tasks
- Used in RLHF (ChatGPT training)

---

## 4. Model-Based vs Model-Free RL

### Model-Free RL

**Definition**: Learn value functions or policies directly from experience, without learning environment dynamics.

**Examples**: Q-Learning, DQN, REINFORCE, A3C, PPO

**Pros**:
- Simpler (no need to model complex environment)
- Works when environment dynamics are unknown or too complex

**Cons**:
- Sample inefficient (requires many environment interactions)
- Can't plan ahead (purely reactive)

### Model-Based RL

**Definition**: Learn model of environment dynamics P(s'|s,a) and/or R(s,a), then use it for planning or data generation.

**Components**:
1. **Model Learning**: Learn P(s'|s,a) from data (supervised learning problem)
2. **Planning**: Use learned model to simulate trajectories and evaluate actions

**Approaches**:

**1. Dyna-Q** (Sutton, 1991):
```
Real experience: Update Q(s,a) from real environment interaction
Simulated experience: Use learned model to generate fake transitions, update Q(s,a) from those too
```

**2. Model Predictive Control (MPC)**:
```
For each state s:
    Simulate K action sequences using learned model
    Choose action from sequence with highest predicted return
    Execute first action, replan at next state
```

**3. World Models / Dreamer** (Ha & Schmidhuber, 2018; Hafner et al., 2020):
- Learn latent world model (VAE-style)
- Train policy entirely in imagined rollouts
- Sample efficient (reuses data through model)

**Model-Based vs Model-Free Trade-offs**:

| Aspect | Model-Free | Model-Based |
|--------|-----------|-------------|
| Sample Efficiency | Low (needs many real interactions) | High (reuses data through model) |
| Computational Cost | Low (simple updates) | High (model training + planning) |
| Asymptotic Performance | Often better (no model bias) | Limited by model accuracy |
| Applicability | Works when model is hard to learn | Requires learnable dynamics |

**Hybrid Approaches** (2024):
- **Model-Based Value Expansion (MVE)**: Use model for multi-step backups in value learning
- **TD-MPC2** (2024): Combines MPC planning with learned world model
- **Dreamer v3** (2024): State-of-the-art model-based method

---

## 5. Exploration Strategies

**Exploration-Exploitation Dilemma**: Should agent try new actions (explore) or use known good actions (exploit)?

### Exploration Methods

**1. ε-Greedy**:
```
With probability ε: random action
With probability 1-ε: greedy action (argmax_a Q(s,a))
```
- Simple, widely used
- ε often annealed from high (0.9) to low (0.01) over training

**2. Boltzmann (Softmax) Exploration**:
```
π(a|s) ∝ exp(Q(s,a) / τ)  where τ is temperature
```
- High τ: nearly uniform (explore)
- Low τ: near-greedy (exploit)

**3. Upper Confidence Bound (UCB)**:
```
a = argmax_a [Q(s,a) + c*sqrt(ln(N(s)) / N(s,a))]
```
- Bonus for uncertain actions (visited less often)
- Optimistic exploration

**4. Thompson Sampling** (Bayesian):
```
Sample Q̃(s,a) from posterior distribution
Choose a = argmax_a Q̃(s,a)
```
- Maintains uncertainty estimates over Q values
- Probability of selecting action proportional to probability it's optimal

**5. Intrinsic Motivation / Curiosity**:
- **Prediction Error**: Reward agent for visiting states where model prediction error is high
- **Novelty**: Reward visiting rarely seen states
- **Empowerment**: Maximize mutual information between actions and future states

**6. Entropy Regularization** (policy gradients):
```
J(θ) = E[R(τ)] + β*H(π)  where H(π) = -E[log π(a|s)]
```
- Encourages stochastic policies (exploration)
- Used in SAC (Soft Actor-Critic)

---

## 6. Connection to Machine Learning

### RL vs Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| Feedback | Direct labels (y for every x) | Delayed rewards (sparse signal) |
| Data | i.i.d. dataset | Sequential, correlated experience |
| Objective | Minimize prediction error | Maximize cumulative reward |
| Exploration | None (all data provided) | Critical (must discover good behavior) |

### RL in Modern ML Systems

**1. RLHF (Reinforcement Learning from Human Feedback)**:
- Train reward model from human preferences
- Use PPO to fine-tune LLM to maximize learned reward
- Used in ChatGPT, Claude, Gemini

**2. Robotics**:
- Manipulation: Learn grasping, assembly, tool use
- Locomotion: Learn walking, running (Boston Dynamics, ANYmal)
- Sim-to-real transfer: Train in simulation, deploy on real robot

**3. Game Playing**:
- AlphaGo (2016): Defeated world Go champion using policy gradients + Monte Carlo Tree Search
- AlphaZero (2017): Mastered Chess, Shogi, Go from self-play
- AlphaStar (2019): StarCraft II at professional level
- OpenAI Five (2019): Dota 2 team strategies

**4. Recommendation Systems**:
- Model user interaction as MDP
- Policy: which content to recommend
- Reward: engagement, clicks, watch time

**5. Dialogue Systems**:
- Policy: what to say next
- Reward: task success, user satisfaction

---

## 7. Challenges and Frontiers (2024)

### Current Challenges

**1. Sample Efficiency**:
- Deep RL requires millions of environment interactions
- Expensive for real-world applications (robotics, healthcare)

**2. Generalization**:
- Policies often overfit to training environments
- Poor transfer to new scenarios

**3. Partial Observability**:
- Real-world often has hidden state (POMDP)
- Recurrent networks (LSTMs, Transformers) help but still challenging

**4. Credit Assignment**:
- Which action caused eventual success/failure?
- Sparse rewards make this harder

**5. Reward Design**:
- Specifying reward function is hard
- Reward hacking (agent exploits loopholes)

### Recent Advances (2024)

**1. Offline RL**:
- Learn from fixed dataset (no environment interaction)
- Conservative Q-Learning (CQL), Implicit Q-Learning (IQL)
- Important for real-world (can't afford random exploration)

**2. Multi-Task RL**:
- Single policy that solves multiple tasks
- Meta-RL: Learn to learn new tasks quickly

**3. Foundation Models for RL**:
- Pre-train on large offline datasets
- Fine-tune for downstream tasks
- Decision Transformer (treats RL as sequence modeling)

**4. Diffusion Models for RL**:
- Diffusion Policy (2023-2024): Model policy as diffusion process
- Better multi-modal action distributions

**5. LLM-guided RL**:
- Use LLMs for reward shaping, subgoal generation
- "Eureka" (NVIDIA 2024): GPT-4 writes reward functions

---

## 8. ARR-COC-0-1 as RL Training Problem

### Relevance Allocation as Procedural Knowing

**Connection to Vervaeke's Framework**:
- **Procedural Knowing** = Learning HOW to allocate tokens through experience
- RL provides formal framework for procedural learning
- Agent learns compression policy through trial-and-error

### Formulation: Token Allocation as MDP

**State** s_t:
- Visual features: image patches, query embedding
- Current relevance scores (propositional, perspectival, participatory)
- Available token budget

**Action** a_t:
- Token allocation decision: [64, 128, 256, 400] for each patch
- Or continuous: allocation weights for K=200 patches

**Reward** r_t:
- Task performance: downstream VQA accuracy
- Compression efficiency: -λ * total_tokens_used
- Balancing reward: -(deviation from opponent processing balance)

**Policy** π(a|s; θ):
- Current: Deterministic (TensionBalancer + AttentionAllocator)
- RL version: Stochastic policy network
  ```
  π(allocation|visual_features, query, relevance_scores; θ)
  ```

### RL Training Approaches for ARR-COC-0-1

**Option 1: Policy Gradient (PPO)**
```python
class TokenAllocationPolicy(nn.Module):
    def __init__(self):
        self.relevance_encoder = knowing.py  # Compute 3 ways
        self.policy_head = nn.Linear(hidden_dim, num_patches * num_levels)

    def forward(self, image, query):
        relevance_scores = self.relevance_encoder(image, query)
        logits = self.policy_head(relevance_scores)
        allocation_probs = softmax(logits.view(num_patches, num_levels))
        return allocation_probs

# Training loop
for batch in dataloader:
    # Collect trajectories
    allocation_probs = policy(image, query)
    allocations = sample(allocation_probs)  # Stochastic

    # Execute compression with sampled allocations
    compressed_features = realize(image, allocations)

    # Downstream task
    vqa_output = vlm(compressed_features, query)

    # Compute reward
    task_reward = accuracy(vqa_output, ground_truth)
    efficiency_penalty = -λ * total_tokens
    reward = task_reward + efficiency_penalty

    # PPO update
    advantage = compute_advantage(reward, baseline)
    ppo_loss = compute_ppo_clip_loss(allocation_probs, advantage)
    optimizer.step()
```

**Option 2: Q-Learning for Discrete Allocations**
```python
class AllocationQNetwork(nn.Module):
    def __init__(self):
        self.relevance_encoder = knowing.py
        self.q_head = nn.Linear(hidden_dim, num_patches * 4)  # 4 LOD levels

    def forward(self, image, query):
        relevance = self.relevance_encoder(image, query)
        q_values = self.q_head(relevance).view(num_patches, 4)
        return q_values  # Q(patch_i, LOD_level)

# Training: DQN with experience replay
for batch in dataloader:
    q_values = q_network(image, query)

    # ε-greedy: sometimes explore random allocations
    if random() < epsilon:
        allocations = random_allocation()
    else:
        allocations = argmax(q_values, dim=1)  # Best LOD per patch

    # Execute and measure reward
    compressed = realize(image, allocations)
    reward = evaluate(compressed, query, ground_truth)

    # Store in replay buffer
    replay_buffer.push(image, query, allocations, reward)

    # Train on minibatch
    optimize_dqn(replay_buffer)
```

**Option 3: Multi-Armed Bandit per Patch**

Simplest approach: Each patch is independent bandit problem.

```python
class PatchBandit:
    def __init__(self, num_arms=4):  # 4 LOD levels
        self.q_estimates = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)

    def select_arm(self, epsilon=0.1):
        if random() < epsilon:
            return random.choice([0,1,2,3])
        else:
            return argmax(self.q_estimates)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q_estimates[arm] += (reward - self.q_estimates[arm]) / self.counts[arm]

# Training
bandits = [PatchBandit() for _ in range(num_patches)]

for image, query in dataset:
    # Each patch chooses LOD independently
    allocations = [bandit.select_arm() for bandit in bandits]

    reward = evaluate(image, query, allocations)

    # Update all bandits with same global reward
    for bandit, allocation in zip(bandits, allocations):
        bandit.update(allocation, reward)
```

### Exploration-Exploitation in Token Allocation

**Exploit**: Use learned relevance patterns (high entropy → high LOD)

**Explore**: Try unexpected allocations:
- Low LOD for salient patches (test if salience alone is sufficient)
- High LOD for uniform patches (test if hidden detail matters)
- Discover non-obvious relevance patterns

**Tempering Dimension** (Vervaeke):
- Currently missing in ARR-COC-0-1
- RL naturally implements explore-exploit tempering
- Balances refining known strategies vs discovering new ones

### Reward Shaping for Relevance Realization

**Multi-Objective Reward**:
```
r = α*accuracy + β*compression_ratio + γ*balance_bonus

where:
- accuracy: downstream task performance (VQA, classification)
- compression_ratio: 1 - (actual_tokens / max_tokens)
- balance_bonus: reward for navigating opponent tensions
```

**Balance Bonus** (reward opponent processing):
```
compression_score = mean([patch_LOD < threshold for patch in patches])
particularization_score = mean([patch_LOD > threshold for patch in patches])

balance = 1 - |compression_score - particularization_score|
```

**Shaped Reward Example**:
```python
def compute_reward(allocations, vqa_accuracy):
    # Task performance
    task_reward = vqa_accuracy

    # Compression efficiency
    total_tokens = sum([LOD_to_tokens[lod] for lod in allocations])
    max_tokens = len(allocations) * 400
    efficiency_reward = (max_tokens - total_tokens) / max_tokens

    # Opponent balance
    low_detail_frac = (allocations < 128).mean()
    high_detail_frac = (allocations > 256).mean()
    balance_reward = 1 - abs(low_detail_frac - high_detail_frac)

    return 0.7*task_reward + 0.2*efficiency_reward + 0.1*balance_reward
```

---

## Sources

### Source Documents
- [john-vervaeke-oracle/papers/00-Vervaeke-2012-Primary-Paper-Analysis.md](../../john-vervaeke-oracle/papers/00-Vervaeke-2012-Primary-Paper-Analysis.md) - Procedural knowing, opponent processing

### Web Research

**RL Fundamentals & DQN**:
- [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - Comprehensive DQN implementation (accessed 2025-11-14)
- [Deep Q-Learning (DQN) - Mnih et al. 2015](https://arxiv.org/abs/1312.5602) - Original DeepMind Atari paper
- [Deep Reinforcement Learning Hands-On (2024)](https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae) - DQN extensions and improvements

**Policy Gradient Methods**:
- [The Definitive Guide to Policy Gradients (2024)](https://arxiv.org/pdf/2401.13662) - arXiv:2401.13662 (accessed 2025-11-14)
- [Policy Gradient Methods in RL - Towards Data Science (May 2024)](https://towardsdatascience.com/policy-gradient-methods-in-reinforcement-learning-31f8a9659398/)
- [REINFORCE Algorithm Tutorial (2024)](https://medium.com/@harshbhatt7585/reinforce-policy-gradient-8a5b2dfed76d) - Policy gradient basics

**Actor-Critic & PPO**:
- [Comparative Analysis of A3C and PPO (IEEE 2024)](https://ieeexplore.ieee.org/document/10703056/) - Modern comparison study
- [DQN VS PPO VS A2C (KDD 2024)](https://kdd2024.kdd.org/wp-content/uploads/2024/08/18-KDD-UC-de-la-Fuente.pdf) - Algorithm comparison research
- [PPO Implementation Analysis (2024)](https://www.researchgate.net/publication/384585837) - Practical PPO study

**Model-Based vs Model-Free**:
- [Model-Free vs Model-Based RL (2024)](https://arxiv.org/abs/2409.17896) - arXiv:2409.17896 (accessed 2025-11-14)
- [Unifying Model-Based and Model-Free RL (RLJ 2024)](https://rlj.cs.umass.edu/2024/papers/Paper37.html) - Hybrid approaches
- [Comparative Study of RL Approaches (2024)](https://www.sciencedirect.com/science/article/abs/pii/S2352710223010318) - Building energy optimization study

**Recent Advances**:
- [Reddit r/reinforcementlearning (2024)](https://www.reddit.com/r/reinforcementlearning/) - Community discussions on state-of-the-art methods
- [Stanford CS234 (2024)](https://www.youtube.com/watch?v=b_wvosA70f8) - Q-learning and DQN lecture

### Additional References
- Sutton & Barto - "Reinforcement Learning: An Introduction" (2nd ed., 2018) - The RL textbook
- Spinning Up in Deep RL (OpenAI) - Practical RL resource
- Stable Baselines3 - Standard RL implementation library

---

**Key Insight for ARR-COC-0-1**: RL provides computational framework for Vervaeke's **procedural knowing**. The current ARR-COC-0-1 uses deterministic rules (InformationScorer → TensionBalancer → AttentionAllocator). RL version would learn these allocation strategies through experience, discovering non-obvious relevance patterns that fixed rules might miss. The exploration-exploitation trade-off directly implements Vervaeke's "cognitive tempering" dimension - currently missing from our architecture.
