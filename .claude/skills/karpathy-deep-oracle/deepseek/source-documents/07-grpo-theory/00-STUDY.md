# GRPO Theory - Study

**Source**: AI Engineering Academy (Theory Behind GRPO)
**Date Processed**: 2025-10-28
**Category**: Reinforcement Learning (GRPO Algorithm)

---

## 📝 TL;DR

**GRPO** (Group Relative Policy Optimization) is the RL algorithm that powers DeepSeek-R1's reasoning capabilities.

**Key innovation**: No value function network - uses **group average reward as baseline** instead.

**Why it matters**: Memory-efficient, works at large scale, enables reasoning emergence.

---

## 🎯 What is GRPO?

**Group Relative Policy Optimization** - RL algorithm for training LLMs on complex tasks (math, code, reasoning).

**vs PPO** (standard RL):
| Feature | PPO | GRPO |
|---------|-----|------|
| **Value function** | Requires separate network | No value function! |
| **Baseline** | Learned from value network | Group average reward |
| **Memory** | High (policy + value networks) | Low (policy only) |
| **Complexity** | More complex | Simpler |

**Big deal**: Eliminating value function saves **huge amounts of memory** at large scale.

---

## 🔧 How GRPO Works

### Step-by-Step Process

**1. Generate multiple responses**
For each prompt s_j, generate K_j different responses (e.g., K=8 answers)

**2. Score all responses**
Use reward model to score each response → get rewards R_j1, R_j2, ..., R_jK

**3. Calculate group average**
Mean reward: R̄_j = (R_j1 + R_j2 + ... + R_jK) / K_j

**4. Compute advantages**
For each response: A_jk = R_jk - R̄_j
(How much better/worse than group average?)

**5. Update policy**
Favor responses with positive advantage (A > 0)
Discourage responses with negative advantage (A < 0)

### Mathematical Formulation

**Loss function**:
```
L = - Σ Σ (π_θ(a_jk | s_j) / π_θ_old(a_jk | s_j)) * A_jk
    + β * KL(π_θ || π_θ_old)
```

**Components**:
- First term: Policy gradient with advantage weighting
- Second term: KL penalty (prevents too-large updates)
- β: Hyperparameter controlling update size

**The trick**: Using group average R̄_j as baseline eliminates need for value function.

---

## 💡 Why This Works

### Advantage Estimation Without Value Function

**Standard RL (PPO)**:
- Advantage = Reward - Value(state)
- Value network estimates future rewards
- Requires training separate network (expensive!)

**GRPO**:
- Advantage = Reward - Group Average
- Group average is empirical baseline
- No extra network needed

**Why group average works**:
- Reduces variance in advantage estimates
- Multiple samples per prompt → stable baseline
- Self-normalizing within each group

### Memory Savings

**PPO memory requirements**:
- Policy network: ~X GB
- Value network: ~X GB (similar size!)
- Total: ~2X GB

**GRPO memory requirements**:
- Policy network: ~X GB
- No value network: 0 GB
- Total: ~X GB

**At 671B scale (R1)**: This is the difference between "can train" and "can't train".

---

## 📊 Performance Results

### DeepSeek-R1 with GRPO

**R1-Zero** (pure GRPO, no SFT):
- AIME 2024: 15.6% → 71.0% (pass@1), 86.7% (majority voting)
- Matches o1-0912 performance

**R1** (GRPO + multi-stage):
- AIME 2024: 79.8%
- MATH-500: 97.3%
- Matches o1-1217

**Comparison**: GRPO matches/beats PPO on reasoning tasks while using less memory.

---

## 🔬 Technical Details

### Group Size (K)

**Trade-off**:
- Larger K → more stable baseline (lower variance)
- Larger K → more compute per update
- Typical: K = 4-16 depending on task

**DeepSeek's choice**: K varies by task, likely 8-16 for math/code.

### Reward Model

**For math/code**:
- Rule-based (answer correct = +1, wrong = 0)
- No learned reward model needed
- Simple and reliable

**For general tasks**:
- May need learned reward model
- GRPO still works, just need good rewards

### KL Penalty (β)

**Purpose**: Prevents policy from changing too much
**Effect**: Stability vs. learning speed trade-off
**Tuning**: Higher β = more stable, slower learning

---

## 💡 Key Insights (Karpathy's Take)

**On eliminating value function**:
- "You don't need it" - group average works fine
- Memory savings at 671B scale are massive
- Simpler is better - one less thing to tune

**On group-based advantages**:
- Multiple samples per prompt is the key
- Self-normalizing within group reduces variance
- Empirical baseline beats learned baseline

**On vs PPO**:
- PPO is fine, but GRPO is better for reasoning
- Value function overhead not worth it at large scale
- GRPO's simplicity is a feature, not a bug

**On reasoning emergence**:
- GRPO + good rewards = reasoning appears naturally
- No need for supervised reasoning examples
- R1-Zero proves it

---

## 🔗 Connections

**Used in**:
- DeepSeek-R1 (R1-Zero and R1)
- DeepSeekMath (first application)

**Connects to**:
- `reinforcement-learning/00-overview.md` - RL category overview
- [R1 Paper Study](../04-deepseek-r1-paper/00-STUDY.md) - Full R1 details

**Enables**:
- Reasoning emergence (R1-Zero)
- Memory-efficient training at 671B scale
- Competitive with o1 performance

---

## 📚 Comparison with Other Methods

### GRPO vs PPO
**PPO**: Standard RL, uses value function, higher memory
**GRPO**: No value function, group baseline, lower memory
**Winner**: GRPO for large-scale reasoning tasks

### GRPO vs DPO
**DPO**: Direct Preference Optimization, no RL loop
**GRPO**: Full RL with reward model
**Use case**: DPO for preferences, GRPO for reasoning

### GRPO vs RLHF (with PPO)
**RLHF**: General alignment (helpfulness, safety)
**GRPO**: Task-specific optimization (math, code)
**Both**: Can be used together (different stages)

---

## 🎯 Key Takeaways

1. **No value function needed** - Group average works as baseline
2. **Memory-efficient** - Critical for 671B scale training
3. **Reasoning emerges** - R1-Zero proves it (15.6% → 71%)
4. **Simpler than PPO** - Fewer hyperparameters, easier to tune
5. **Production-ready** - Powers R1 which matches o1

---

**Last Updated**: 2025-10-28
**Status**: Core algorithm study complete
**Note**: GRPO is the secret sauce behind R1's reasoning capabilities
