# Reinforcement Learning & Reward: Action Selection Through Return Maximization

## Overview

**Reinforcement learning (RL)** is a computational framework for learning optimal behavior through trial-and-error interaction with an environment. Unlike supervised learning (which requires labeled examples) or unsupervised learning (which finds patterns), RL agents learn by receiving scalar reward signals that indicate the desirability of their actions.

**Core Problem**: How should an agent select actions to maximize cumulative reward over time when actions have delayed consequences and the environment may be uncertain or partially observable?

From [Sutton & Barto, "Reinforcement Learning: An Introduction"](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) (2018):
> "Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them."

**Relevance to ARR-COC-0-1**: Token allocation is fundamentally a sequential decision problem - allocate tokens to patches to maximize relevance realization over a sequence of visual queries. RL provides the formal framework for learning optimal allocation policies.

From [Malekzadeh & Plataniotis, "Active Inference and Reinforcement Learning: A Unified Inference"](https://direct.mit.edu/neco/article/36/10/2073/124162/Active-Inference-and-Reinforcement-Learning-A) (Neural Computation, 2024):
> "Active inference (AIF) optimizes two complementary objective functions: variational free energy (VFE) and expected free energy (EFE). The VFE objective is analogous to reward maximization in RL, while EFE supplies information-seeking exploratory behavior."

---

## Section 1: Markov Decision Processes (~100 lines)

### 1.1 Formal Framework

A **Markov Decision Process (MDP)** is a tuple (S, A, P, R, γ) where:

- **S**: State space (what the agent observes)
- **A**: Action space (what the agent can do)
- **P(s'|s,a)**: Transition dynamics (how actions change states)
- **R(s,a)**: Reward function (immediate feedback)
- **γ ∈ [0,1]**: Discount factor (preference for immediate vs future rewards)

**Markov Property**: Future states depend only on current state and action, not on history.

P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)

**Policy π(a|s)**: Mapping from states to actions (stochastic or deterministic)

**Objective**: Find policy π* that maximizes expected cumulative discounted reward:

J(π) = E_π[∑_{t=0}^∞ γ^t R(s_t, a_t)]

From [Ghasemi et al., "Introduction to Reinforcement Learning"](https://arxiv.org/abs/2408.07712) (arXiv:2408.07712, 2024):
> "The discount factor γ controls the agent's planning horizon. γ → 0 makes the agent myopic (only immediate rewards matter), while γ → 1 makes the agent farsighted (all future rewards weighted equally)."

### 1.2 Exploration vs Exploitation

**Fundamental Tradeoff**:
- **Exploitation**: Choose actions known to yield high rewards
- **Exploration**: Try new actions to discover potentially better rewards

**Multi-Armed Bandit**: Simplified RL with single state, multiple actions
- ε-greedy: Exploit best action with probability (1-ε), explore randomly with ε
- Upper Confidence Bound (UCB): Balance mean reward with uncertainty
- Thompson Sampling: Bayesian approach sampling from posterior over action values

From [Forbes et al., "Potential-Based Reward Shaping For Intrinsic Motivation"](https://arxiv.org/abs/2402.07411) (AAMAS 2024):
> "Intrinsic motivation (IM) reward-shaping methods address exploration in sparse-reward environments by providing auxiliary rewards for novelty, curiosity, or uncertainty reduction. However, naive IM can alter the set of optimal policies."

### 1.3 Return and Value Functions

**Return G_t**: Cumulative discounted reward from time t onward

G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = ∑_{k=0}^∞ γ^k R_{t+k+1}

**State-Value Function V^π(s)**: Expected return starting from state s under policy π

V^π(s) = E_π[G_t | s_t = s]

**Action-Value Function Q^π(s,a)**: Expected return from state s, taking action a, then following π

Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]

**Bellman Equations** (consistency between values at successive time steps):

V^π(s) = ∑_a π(a|s) ∑_{s'} P(s'|s,a)[R(s,a) + γV^π(s')]

Q^π(s,a) = ∑_{s'} P(s'|s,a)[R(s,a) + γ∑_{a'} π(a'|s')Q^π(s',a')]

**Optimal Value Functions**:

V*(s) = max_π V^π(s)
Q*(s,a) = max_π Q^π(s,a)

Optimal policy π*(a|s) selects actions maximizing Q*(s,a).

---

## Section 2: Temporal Difference Learning (~120 lines)

### 2.1 Value-Based Methods

**Q-Learning** (Watkins 1989): Learn optimal action-value function Q*(s,a)

Update rule:
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

Where:
- α: Learning rate
- r: Observed reward
- s': Next state
- TD error: δ = r + γ max_{a'} Q(s',a') - Q(s,a)

**SARSA** (On-policy TD control):
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

Difference: SARSA uses actual next action a' from policy, Q-learning uses max over actions.

**Deep Q-Networks (DQN)**: Use neural networks to approximate Q(s,a) for high-dimensional states (Mnih et al., Nature 2015)
- Experience replay: Store (s,a,r,s') transitions, sample mini-batches for training
- Target network: Separate network for computing Q-targets, updated periodically
- Stabilizes learning in complex environments

From [Henry Wu, "Intro to RL: Monte Carlo to Policy Gradient"](https://medium.com/@hsinhungw/intro-to-reinforcement-learning-monte-carlo-to-policy-gradient-1c7ede4eed6e) (Medium, 2023):
> "Value functions estimate the expected return for the agent in a given state or performing a given action in a given state. By learning accurate value functions, the agent can make better decisions without needing to search through all possible future action sequences."

### 2.2 Advantage Functions

**Advantage A^π(s,a)**: How much better action a is than average under policy π

A^π(s,a) = Q^π(s,a) - V^π(s)

**Generalized Advantage Estimation (GAE)**: Bias-variance tradeoff for advantage estimation

A^{GAE}(s,a) = ∑_{t=0}^∞ (γλ)^t δ_t

Where δ_t = r_t + γV(s_{t+1}) - V(s_t) is TD error, λ ∈ [0,1] controls bias-variance.

**Benefits**:
- λ = 0: Low variance, high bias (one-step TD)
- λ = 1: High variance, low bias (Monte Carlo)
- λ ∈ (0,1): Balanced trade-off

Used in Actor-Critic methods (Section 3) for stable policy gradient estimation.

### 2.3 Function Approximation

For large or continuous state spaces, tabular Q-learning infeasible. Use parametric approximations:

**Linear approximation**:
Q(s,a; w) = w^T φ(s,a)

Where φ(s,a) is feature vector, w is weight vector.

**Neural network approximation**:
Q(s,a; θ) = Neural_Net(s,a; θ)

**Challenges**:
- **Deadly triad**: Function approximation + bootstrapping (TD) + off-policy learning → instability
- **Overestimation bias**: max operator in Q-learning systematically overestimates values
- **Catastrophic forgetting**: New experiences overwrite old knowledge

**Solutions**:
- Double Q-learning: Use two Q-functions to reduce overestimation
- Prioritized experience replay: Sample important transitions more frequently
- Dueling architecture: Separate value and advantage streams in network

---

## Section 3: Policy Gradient Methods (~120 lines)

### 3.1 Direct Policy Optimization

Instead of learning value function then deriving policy, **directly parameterize policy**:

π(a|s; θ) = Probability of action a in state s, with parameters θ

**Policy Gradient Theorem** (Sutton et al. 1999):

∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) Q^π(s,a)]

**Intuition**: Increase probability of actions with positive value, decrease probability of negative-value actions.

From [Sutton et al., "Policy Gradient Methods for Reinforcement Learning"](https://dl.acm.org/doi/10.5555/3009657.3009806) (NIPS 1999):
> "In policy gradient methods, the policy is explicitly represented by its own function approximator, independent of the value function. This allows the policy to be optimized directly for the objective of interest."

**REINFORCE Algorithm**:
```
For each episode:
  Generate trajectory (s_0, a_0, r_1, s_1, a_1, ..., s_T)
  For each step t:
    G_t = ∑_{k=t+1}^T γ^{k-t} r_k  (return from step t)
    θ ← θ + α γ^t G_t ∇_θ log π(a_t|s_t; θ)
```

**Variance Reduction**: Use baseline b(s) that doesn't depend on action:

∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) (Q^π(s,a) - b(s))]

Common baseline: V^π(s) (state-value function) → gradient proportional to advantage A^π(s,a).

### 3.2 Actor-Critic Methods

Combine value-based and policy-based approaches:

**Actor**: Policy π(a|s; θ) that selects actions
**Critic**: Value function V(s; w) or Q(s,a; w) that evaluates actions

**Advantage Actor-Critic (A2C)**:
```
Critic update: w ← w + α_w δ ∇_w V(s; w)
Where δ = r + γV(s'; w) - V(s; w)

Actor update: θ ← θ + α_θ δ ∇_θ log π(a|s; θ)
```

**A3C (Asynchronous A2C)**: Multiple parallel agents collect experience, update shared parameters asynchronously (Mnih et al. 2016)

**PPO (Proximal Policy Optimization)**: Constrain policy updates to trust region

L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where r_t(θ) = π(a|s;θ)/π(a|s;θ_old) is probability ratio, Â_t is advantage estimate.

**Benefits**: Prevents catastrophically large policy updates, improves training stability.

From [OpenReview, "Policy Gradient for Reinforcement Learning"](https://openreview.net/pdf?id=XsWA2y2TyI):
> "Policy gradient methods enable optimization of non-linear, non-convex objective functions directly, avoiding the indirection of value-based methods. This is particularly useful when optimal actions require continuous, high-dimensional outputs."

### 3.3 Deterministic Policy Gradient

For continuous action spaces, stochastic policies can be inefficient. **Deterministic Policy Gradient (DPG)**:

Policy: a = μ(s; θ) (deterministic)
∇_θ J(θ) = E[∇_θ μ(s; θ) ∇_a Q(s,a)|_{a=μ(s)}]

**DDPG (Deep DPG)**: Combines DPG with DQN techniques
- Replay buffer
- Target networks for actor and critic
- Ornstein-Uhlenbeck noise for exploration

**TD3 (Twin Delayed DDPG)**: Addresses overestimation in DDPG
- Two Q-networks, use minimum for updates
- Delayed policy updates (less frequent than Q-updates)
- Target policy smoothing (add noise to target actions)

**SAC (Soft Actor-Critic)**: Maximum entropy RL framework
- Objective: J(π) = ∑_t E[r_t + α H(π(·|s_t))]
- H(π) is policy entropy, α controls exploration
- Automatically tunes α during training

---

## Section 4: Reward Shaping & Intrinsic Motivation (~120 lines)

### 4.1 Reward Engineering Challenges

**Sparse Rewards**: Agent receives reward only at terminal states (e.g., winning game)
- Difficult to learn without dense feedback
- Random exploration unlikely to discover rewarding states

**Reward Hacking**: Agent exploits loopholes in reward function
- Example: Cleaning robot hides dirt under rug instead of removing it
- Reward specification is difficult, especially for complex real-world tasks

**Credit Assignment**: Which actions led to observed reward?
- Temporal credit assignment: Reward delayed many steps after critical action
- Structural credit assignment: Many agents/components contributed to outcome

### 4.2 Potential-Based Reward Shaping (PBRS)

**Shaped Reward**:
R'(s,a,s') = R(s,a,s') + F(s,s')

Where F(s,s') is shaping function.

**Potential-Based Shaping** (Ng et al. 1999):
F(s,s') = γΦ(s') - Φ(s)

Where Φ: S → ℝ is potential function.

**Theorem**: PBRS preserves optimal policy (doesn't change policy ranking).

**Proof Intuition**: Shaped cumulative reward differs from original by constant:

G'_t = ∑_{k=0}^∞ γ^k[R_{t+k+1} + γΦ(s_{t+k+2}) - Φ(s_{t+k+1})]
     = G_t + γΦ(s_{t+1}) - Φ(s_t)  (telescoping sum)

Since Φ(s_t) doesn't depend on action a_t, optimal policy unchanged.

From [Forbes et al., "Potential-Based Reward Shaping For Intrinsic Motivation"](https://arxiv.org/abs/2402.07411) (2024):
> "We present Potential-Based Intrinsic Motivation (PBIM), a method for converting IM rewards into a potential-based form that is useable without altering the set of optimal policies. Testing in MiniGrid DoorKey and Cliff Walking environments, we demonstrate that PBIM successfully prevents the agent from converging to a suboptimal policy."

### 4.3 Intrinsic Motivation Methods

**Count-Based Exploration**:
- Bonus reward r_intrinsic(s) = β/√(N(s))
- N(s): Visit count for state s
- Encourages visiting rare states

**Prediction-Based Bonuses**:
- **ICM (Intrinsic Curiosity Module)**: Reward prediction error of forward model
  - Learn f: (s,a) → ŝ' (predict next state)
  - r_intrinsic = ||s' - ŝ'||²
- **RND (Random Network Distillation)**: Train network to predict random features
  - Fixed random network: g(s)
  - Trained predictor: ĝ(s; θ)
  - r_intrinsic = ||g(s) - ĝ(s)||²
  - Prediction error high for novel states

**Empowerment**: Maximize mutual information I(A; S'|S) between actions and resulting states
- Encourages actions that give agent control over environment
- Information-theoretic formulation of exploratory behavior

**Connection to Active Inference**: These intrinsic motivations approximate information-seeking behavior formalized in expected free energy (Section 7).

### 4.4 Reward-Free RL

**Paradigm Shift**: Learn useful representations/skills without external reward, then rapidly adapt to downstream tasks.

**Self-Supervised RL**:
- Pre-training phase: Explore environment, learn state representations
- Fine-tuning phase: Optimize for specific task reward

**Skill Discovery**:
- Learn diverse behaviors (skills) in unsupervised manner
- DIAYN (Diversity Is All You Need): Maximize I(Skills; States) - H(Skills|Actions)
  - Learn distinguishable skills that visit different states

**Benefits**: Reduces need for reward engineering, enables rapid task adaptation.

---

## Section 5: Model-Based RL (~100 lines)

### 5.1 Learning World Models

**Model-Based vs Model-Free**:
- **Model-Free**: Learn policy/value function directly from experience (Q-learning, policy gradient)
- **Model-Based**: Learn model of environment dynamics, use for planning

**Transition Model**: P(s'|s,a) or f(s,a) → s'
**Reward Model**: R(s,a) or r(s,a,s')

**Sample Efficiency**: Model-based methods can be more sample-efficient
- Generate synthetic rollouts using learned model
- Plan multiple steps ahead without environment interaction

**Challenges**:
- Model errors compound over long horizons
- High-dimensional state spaces difficult to model accurately

### 5.2 Dyna Architecture

Integrate model learning and model-free RL (Sutton 1991):

```
Loop:
  (a) Direct RL: Real experience → update Q-function
  (b) Model Learning: Real experience → update model
  (c) Planning: Simulated experience from model → update Q-function
```

**Dyna-Q Algorithm**:
- Maintain tabular Q(s,a) and model M(s,a) → (r, s')
- After each real step: Update Q, update M
- Planning: Sample previously visited (s,a), generate (r,s') from M, update Q

**Benefits**: Combines strengths of model-free (robustness) and model-based (efficiency)

### 5.3 Planning with Learned Models

**MCTS (Monte Carlo Tree Search)**:
- Build search tree by simulating trajectories using model
- Selection: Navigate tree using UCB (balance exploration/exploitation)
- Expansion: Add new nodes
- Simulation: Rollout to terminal state
- Backpropagation: Update value estimates along path

**AlphaZero**: Combines MCTS with deep neural networks
- Neural network f(s) → (p, v) outputs policy prior p and value estimate v
- MCTS uses neural network to guide search
- Self-play generates training data
- Achieves superhuman performance in Go, chess, shogi

**MuZero**: Extends AlphaZero to environments without known rules
- Learns latent dynamics model: (s_t, a_t) → (r_{t+1}, s_{t+1})
- Plans in learned latent space, not raw observations
- Combines model-based planning with model-free value learning

---

## Section 6: Partial Observability (POMDPs) (~120 lines)

### 6.1 Hidden State Problem

**POMDP**: Partially Observable Markov Decision Process
- Agent doesn't observe true state s, only observation o
- Observation function: O(o|s,a)
- History: h_t = (o_0, a_0, o_1, a_1, ..., o_t)

**Belief State**: Probability distribution over possible states
b_t(s) = P(s_t = s | h_t)

**Optimal Policy**: Function of belief state π*(b)

**Challenges**:
- Belief space is continuous even for discrete state spaces
- Belief update computationally expensive
- Planning in belief space intractable for large problems

From [Malekzadeh & Plataniotis, "Active Inference and Reinforcement Learning"](https://direct.mit.edu/neco/article/36/10/2073/124162) (2024):
> "Many real-world problems involve partial or noisy observations, where agents cannot access complete and accurate information about the environment. These problems are commonly formulated as partially observable Markov decision processes (POMDPs)."

### 6.2 Recurrent Policies

**LSTM/GRU Policies**: Maintain hidden state h_t summarizing history
- π(a|o_t, h_t; θ)
- h_{t+1} = RNN(o_t, a_t, h_t)

**Advantages**:
- Implicit belief state representation
- End-to-end trainable with policy gradient
- Scales to high-dimensional observations (images)

**R2D2 (Recurrent DQN)**: Applies LSTM to DQN
- Stores episode chunks in replay buffer
- Burn-in period to initialize hidden state

**Limitations**:
- Opaque learned representations
- Difficult to interpret what agent "believes"
- May fail to capture long-term dependencies

### 6.3 State Estimation Approaches

**Explicit Belief Tracking**:
- Maintain particle filter or Kalman filter over states
- Update beliefs using Bayes rule
- Plan using belief as input to policy

**Variational Autoencoders (VAE)**:
- Encoder: q(z|o) maps observations to latent state z
- Decoder: p(o|z) reconstructs observations from latent
- Transition: p(z'|z,a) models dynamics in latent space

**World Models** (Ha & Schmidhuber 2018):
- Vision model (VAE): Compress observations to latent z
- Memory model (RNN): Predict next latent ẑ_{t+1} = RNN(z_t, a_t)
- Controller: Small network π(a|z,h) trained with evolutionary strategy

**Benefits**: Compact representation, interpretable latent space, fast training in imagination.

### 6.4 Information-Seeking Behavior

In POMDPs, **information has value** - actions that reveal hidden state enable better future decisions.

**POMCP (POMDP Monte Carlo Planning)**:
- MCTS extension for POMDPs
- Maintains belief particles at each node
- Selects actions balancing immediate reward and information gain

**Active Sensing**: Deliberately take actions to reduce uncertainty
- Example: Robot turning head to better localize object
- Formalized in expected free energy (EFE) in active inference (Section 7)

**Bayesian RL**: Maintain distribution over MDP parameters
- Exploration as information gathering about environment
- Optimal exploration = value of information

---

## Section 7: Active Inference as RL (~100 lines)

### 7.1 Expected Free Energy

**Active Inference** minimizes expected free energy (EFE):

G(π, τ) = E_π[ln Q(o_τ, s_τ) - ln P(o_τ, s_τ | π)]

Where:
- τ: Future time step
- Q(o,s): Recognition density (approximate posterior)
- P(o,s|π): Generative model

**Decomposition**:
EFE = Pragmatic value (reward) + Epistemic value (information gain)

From [Friston et al., "Reinforcement Learning or Active Inference?"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421) (PLOS One 2009):
> "We show that it is fairly simple to teach an agent complicated and adaptive behaviours using a free-energy formulation of perception. In this formulation, agents adjust their internal states to minimize the long-term average of surprise about sensory exchanges with the world."

### 7.2 RL-AIF Equivalence

**Pragmatic Value** ≈ **Expected Reward**:

E[R(s,a)] ≈ -E_π[D_KL(P(o_τ|s_τ, π) || P_preferred(o_τ))]

Where P_preferred(o_τ) encodes desired observations (goal states).

**Epistemic Value** ≈ **Information Gain**:

I(S_τ; O_τ | π) = Expected reduction in uncertainty about states

**Unified Objective**:
- RL: Maximize ∑_t E[r_t] (reward only)
- AIF: Minimize ∑_t EFE_t (reward + information)

From [Malekzadeh & Plataniotis](https://direct.mit.edu/neco/article/36/10/2073/124162) (2024):
> "AIF optimizes two complementary objective functions: variational free energy (VFE) and expected free energy (EFE). The VFE objective is analogous to reward maximization in RL, while EFE supplies information-seeking exploratory behavior."

### 7.3 Intrinsic Exploration

**Information-Seeking = Intrinsic Motivation**:

AIF naturally balances:
- **Exploitation**: Actions achieving preferred observations (reward-seeking)
- **Exploration**: Actions reducing uncertainty (information-seeking)

**Epistemic value encourages**:
- Visiting states where observations are informative
- Reducing posterior uncertainty over hidden states
- Resolving ambiguity in environment model

**Comparison to RL Exploration**:
- ε-greedy, UCB: Heuristic, not principled information-seeking
- RND, ICM: Learn prediction error as intrinsic reward
- AIF: Information gain emerges from free energy minimization

### 7.4 Reward-Free Capability

AIF agents can operate **without external reward specification**:

- Minimize surprise about sensory data (stay in expected states)
- Maximize model evidence (confirm predictions)
- Epistemic value drives exploratory behavior even without pragmatic goals

**Homeostatic Regulation**: Agent seeks states matching prior preferences
- Encoded in generative model P(o,s)
- No explicit reward function needed
- Behavior emerges from model structure

**Comparison**:
- RL: Requires carefully designed reward function
- AIF: Behavior emerges from generative model + free energy minimization
- Hybrid: Use AIF for exploration, RL rewards for specific task objectives

---

## Section 8: ARR-COC-0-1 Integration (~70 lines)

### 8.1 Token Allocation as RL

**State s**: Visual query + current image patches + LOD allocations
**Action a**: LOD assignment for next patch (64, 128, 256, 400 tokens)
**Reward r**: Relevance realization improvement (change in VQA accuracy, comprehension score)

**Policy π(LOD | query, patch, context)**:
- Input: Query embedding, patch features, spatial context
- Output: Distribution over LOD levels {64, 128, 256, 400}
- Parametrized by neural network (adapter in ARR-COC-0-1)

**Value Function V(s)**: Expected relevance improvement from current state
**Q-Function Q(s, LOD)**: Expected relevance of allocating specific LOD to patch

### 8.2 Reward Shaping for Relevance

**Base Reward**: Task performance (VQA accuracy, comprehension)

**Shaped Rewards** (intrinsic motivation for relevance):
1. **Information Gain**: IG(patch) = H(before) - H(after)
   - Entropy reduction from processing patch
   - Approximates epistemic value in AIF

2. **Salience Bonus**: r_salience(patch) = ∑_i w_i × scorer_i(patch)
   - Propositional: Shannon entropy of patch
   - Perspectival: Jungian archetypal salience
   - Participatory: Query-patch cross-attention

3. **Opponent Process Reward**: Balance tensions
   - Compress ↔ Particularize: Penalize extreme LODs
   - Exploit ↔ Explore: Bonus for trying under-sampled LODs
   - Focus ↔ Diversify: Reward spatial diversity in high-LOD allocations

**Potential-Based Formulation**:
Φ(state) = Expected relevance achievable from current state
F(s,s') = γΦ(s') - Φ(s)

Ensures shaped rewards preserve optimal allocation policy.

### 8.3 Active Inference for Exploration

**Expected Free Energy for Token Allocation**:

EFE(LOD | patch, query) = Pragmatic + Epistemic

**Pragmatic**: Expected relevance improvement (RL reward)
**Epistemic**: Expected information gain about hidden query semantics

**Information-Seeking Allocation**:
- High uncertainty patches → allocate more tokens (higher LOD)
- Ambiguous query semantics → explore multiple resolutions
- Known irrelevant regions → allocate minimal tokens (64 LOD)

**Comparison to Pure RL**:
- RL alone: Maximize task reward, may under-explore
- AIF addition: Intrinsic drive to resolve uncertainty
- Hybrid: Task performance + automatic exploration

### 8.4 Learning Allocation Policies

**Training Approaches**:

1. **Supervised Pre-training**: Human-labeled relevance maps → initial policy
2. **RL Fine-tuning**: Policy gradient (PPO) on VQA task reward
3. **AIF Integration**: EFE minimization for exploration bonus

**Distributed Training** (File 1: ZeRO optimizer):
- Large vision-language models require distributed training
- Shard optimizer states, gradients, parameters across GPUs
- Enables training complex allocation policies

**Inference Optimization** (File 5: TensorRT):
- Deploy learned policy for fast LOD selection
- Optimize for low-latency token allocation
- Critical for real-time visual question answering

**Orchestration** (File 9: Kubernetes GPU scheduling):
- Manage multi-GPU training jobs for policy learning
- Schedule allocation experiments across compute cluster
- Track policy performance metrics

### 8.5 Connections to AMD/Alternative Hardware (File 13)

**AMD MI300X for RL Training**:
- ROCm support for PyTorch RL libraries (Stable-Baselines3, RLlib)
- High memory bandwidth (5.3 TB/s) beneficial for experience replay buffers
- Cost-effective alternative to NVIDIA for large-scale policy training

**Mixed Precision Training**:
- FP16/BF16 for value/policy network forward passes
- FP32 for advantage computation (numerical stability)
- Accelerates training without sacrificing final performance

**Deployment Considerations**:
- Train policy on AMD MI300X cluster
- Deploy to NVIDIA edge devices (Jetson) for inference
- Cross-platform compatibility via ONNX export

---

## Sources

**Source Documents**: None directly referenced (cognitive-mastery files are research-based)

**Web Research**:

**Reinforcement Learning Foundations**:
- [Sutton & Barto, "Reinforcement Learning: An Introduction"](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) (2nd ed., 2018) - Canonical RL textbook (accessed 2025-11-16)
- [Ghasemi et al., "Introduction to Reinforcement Learning"](https://arxiv.org/abs/2408.07712) (arXiv:2408.07712, 2024) - Recent comprehensive survey
- [Henry Wu, "Intro to RL: Monte Carlo to Policy Gradient"](https://medium.com/@hsinhungw/intro-to-reinforcement-learning-monte-carlo-to-policy-gradient-1c7ede4eed6e) (Medium, 2023) - Accessible tutorial

**Policy Gradient Methods**:
- [Sutton et al., "Policy Gradient Methods for Reinforcement Learning"](https://dl.acm.org/doi/10.5555/3009657.3009806) (NIPS 1999) - Original policy gradient theorem paper
- [OpenReview, "Policy Gradient for Reinforcement Learning"](https://openreview.net/pdf?id=XsWA2y2TyI) - Modern survey
- [Towards Data Science, "Policy Based RL, the Easy Way"](https://towardsdatascience.com/policy-based-reinforcement-learning-the-easy-way-8de9a3356083) (2019) - Practical guide

**Reward Shaping & Intrinsic Motivation**:
- [Forbes et al., "Potential-Based Reward Shaping For Intrinsic Motivation"](https://arxiv.org/abs/2402.07411) (AAMAS 2024, arXiv:2402.07411) - PBIM method
- [Forbes et al., "Potential-Based Intrinsic Motivation: Preserving Optimality"](https://arxiv.org/abs/2410.12197) (arXiv:2410.12197, 2024) - Extended analysis

**Active Inference & RL**:
- [Friston et al., "Reinforcement Learning or Active Inference?"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421) (PLOS One, 2009) - Foundational AIF-RL connection
- [Malekzadeh & Plataniotis, "Active Inference and Reinforcement Learning: A Unified Inference"](https://direct.mit.edu/neco/article/36/10/2073/124162) (Neural Computation 36(10), 2024) - Continuous space POMDP unification
- [Reddit r/reinforcementlearning, "Active Inference Success Stories"](https://www.reddit.com/r/reinforcementlearning/comments/1fbu536/) (2024) - Practitioner discussion

**Additional References**:
- Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015) - DQN
- Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv:1707.06347, 2017) - PPO
- Ha & Schmidhuber, "World Models" (NeurIPS 2018) - Learned environment models
- Ng et al., "Policy Invariance Under Reward Transformations" (ICML 1999) - PBRS theory

**Karpathy Engineering Files** (Referenced):
- File 1: distributed-training/00-deepspeed-zero-optimizer.md
- File 5: inference-optimization/00-tensorrt-fundamentals.md
- File 9: orchestration/00-kubernetes-gpu-scheduling.md
- File 13: alternative-hardware/00-amd-rocm-ml.md
