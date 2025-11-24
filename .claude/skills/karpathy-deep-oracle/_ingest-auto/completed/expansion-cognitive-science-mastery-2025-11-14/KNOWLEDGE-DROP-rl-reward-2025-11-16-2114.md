# KNOWLEDGE DROP: Reinforcement Learning & Reward

**Created**: 2025-11-16 21:14
**Part**: PART 21 of expansion-cognitive-science-mastery-2025-11-14
**File**: cognitive-mastery/20-reinforcement-learning-reward.md
**Lines**: ~700 lines

---

## What Was Created

Comprehensive knowledge file on **reinforcement learning fundamentals**, covering:

1. **Markov Decision Processes** (~100 lines)
   - Formal MDP framework: states, actions, transitions, rewards, discount factors
   - Exploration vs exploitation tradeoff
   - Value functions (V, Q) and Bellman equations

2. **Temporal Difference Learning** (~120 lines)
   - Q-learning and SARSA algorithms
   - Deep Q-Networks (DQN, experience replay, target networks)
   - Advantage functions and Generalized Advantage Estimation (GAE)
   - Function approximation challenges (deadly triad, overestimation bias)

3. **Policy Gradient Methods** (~120 lines)
   - Direct policy optimization and policy gradient theorem
   - REINFORCE, Actor-Critic, A2C/A3C
   - PPO (Proximal Policy Optimization)
   - Deterministic policy gradients (DDPG, TD3, SAC)

4. **Reward Shaping & Intrinsic Motivation** (~120 lines)
   - Sparse reward challenges
   - Potential-Based Reward Shaping (PBRS) - preserves optimal policies
   - Intrinsic motivation: count-based, ICM, RND, empowerment
   - Reward-free RL and skill discovery

5. **Model-Based RL** (~100 lines)
   - Learning world models vs model-free approaches
   - Dyna architecture (integrate learning and planning)
   - MCTS, AlphaZero, MuZero

6. **Partial Observability (POMDPs)** (~120 lines)
   - Belief states and hidden state problem
   - Recurrent policies (LSTM/GRU, R2D2)
   - State estimation (VAE, World Models)
   - Information-seeking behavior

7. **Active Inference as RL** (~100 lines)
   - Expected free energy (EFE) decomposition
   - RL-AIF equivalence (pragmatic + epistemic value)
   - Intrinsic exploration from information-seeking
   - Reward-free capability

8. **ARR-COC-0-1 Integration** (~70 lines)
   - Token allocation as RL problem
   - Reward shaping for relevance (information gain, salience, opponent processing)
   - Active inference for exploration
   - Learning allocation policies with distributed training
   - AMD MI300X considerations for RL

---

## Key Research Sources

**Foundational**:
- Sutton & Barto (2018) - RL textbook standard
- Ghasemi et al. (arXiv:2408.07712, 2024) - Modern RL survey

**Policy Gradients**:
- Sutton et al. (NIPS 1999) - Policy gradient theorem
- Schulman et al. (2017) - PPO algorithm

**Reward Shaping**:
- Forbes et al. (AAMAS 2024, arXiv:2402.07411) - Potential-Based Intrinsic Motivation (PBIM)
- Ng et al. (ICML 1999) - PBRS theory

**Active Inference**:
- Friston et al. (PLOS One 2009) - "Reinforcement Learning or Active Inference?"
- Malekzadeh & Plataniotis (Neural Computation 2024) - Unified continuous space POMDP inference

**Classic Papers**:
- Mnih et al. (Nature 2015) - DQN
- Ha & Schmidhuber (NeurIPS 2018) - World Models

---

## Influenced By Files

**File 1** (distributed-training/00-deepspeed-zero-optimizer.md):
- Multi-GPU policy training for complex allocation networks
- ZeRO optimizer for large RL models

**File 5** (inference-optimization/00-tensorrt-fundamentals.md):
- Fast policy inference for real-time token allocation
- Optimize learned Q-functions for low-latency decisions

**File 9** (orchestration/00-kubernetes-gpu-scheduling.md):
- Manage distributed RL training experiments
- Schedule policy learning jobs across cluster

**File 13** (alternative-hardware/00-amd-rocm-ml.md):
- AMD MI300X for RL training (high memory for replay buffers)
- ROCm support for PyTorch RL libraries (Stable-Baselines3)
- Cost-effective alternative for large-scale policy optimization

---

## ARR-COC-0-1 Connections (~10%)

**Token Allocation = Sequential Decision Problem**:
- State: Query + patches + current LOD allocations
- Actions: Assign LOD {64, 128, 256, 400} to next patch
- Reward: Relevance realization improvement (VQA accuracy)

**Reward Shaping for Relevance**:
- Information gain: H(before) - H(after) from processing patch
- Salience bonus: Multi-scorer relevance (propositional, perspectival, participatory)
- Opponent processing: Balance compress↔particularize, exploit↔explore tensions
- Potential-based formulation preserves optimal allocation policy

**Active Inference Integration**:
- Expected free energy = Pragmatic (reward) + Epistemic (information gain)
- Information-seeking drives exploration: allocate high LOD to uncertain patches
- Hybrid approach: Task reward (VQA) + intrinsic curiosity (resolve ambiguity)

**Learning Pipeline**:
1. Supervised pre-training on relevance maps
2. RL fine-tuning with PPO on task rewards
3. AIF integration for automatic exploration
4. Distributed training with ZeRO (File 1)
5. Fast inference with TensorRT (File 5)
6. Orchestrated experiments (File 9)
7. AMD MI300X cluster training (File 13)

---

## Quality Markers

✓ Comprehensive 8-section structure (~700 lines)
✓ 20+ web sources cited with URLs and access dates
✓ Citations throughout (Sutton, Friston, Forbes, Malekzadeh)
✓ Formal mathematical definitions (MDP, Bellman, policy gradient)
✓ ARR-COC-0-1 integration section (~70 lines, 10% of content)
✓ All 4 influenced files explicitly referenced
✓ Practical algorithms (Q-learning, PPO, PBRS, EFE)
✓ Recent research (2024 papers on PBIM, AIF-RL unification)

---

## Novel Contributions

**Unified RL-AIF Perspective**:
- Connects traditional RL (reward maximization) with active inference (free energy minimization)
- Shows epistemic value = information gain = intrinsic exploration
- Explains how AIF naturally balances exploitation and exploration

**Potential-Based Intrinsic Motivation**:
- Recent breakthrough (Forbes et al. 2024): convert intrinsic rewards to PBRS form
- Preserves optimal policies while enabling complex exploration bonuses
- Directly applicable to ARR-COC relevance shaping

**POMDP Integration**:
- Partial observability crucial for real-world vision (never see full scene)
- Belief states + information-seeking actions
- Connects to ARR-COC uncertainty-driven token allocation

**ARR-COC as RL Problem**:
- Formalized token allocation as MDP/POMDP
- Designed reward shaping preserving optimality (PBRS)
- Integrated active inference for automatic exploration
- Practical training pipeline with modern distributed systems

---

## Completion Status

- [✓] Web research: 3 searches conducted
- [✓] Key sources scraped: Forbes PBIM, Malekzadeh AIF-RL
- [✓] Knowledge file created: 20-reinforcement-learning-reward.md
- [✓] 8 sections completed: MDP, TD, Policy Gradient, Reward Shaping, Model-Based, POMDP, AIF, ARR-COC
- [✓] Citations: 20+ sources with full URLs
- [✓] ARR-COC integration: Section 8 (~70 lines, 10%)
- [✓] Influenced files: 1, 5, 9, 13 explicitly cited
- [✓] KNOWLEDGE DROP created

**PART 21: SUCCESS** ✓
