# KNOWLEDGE DROP: Goal-Conditioned Learning

**Created**: 2025-01-23 22:45
**Part**: 34/42
**Status**: Complete
**File**: ml-affordances/03-goal-conditioned.md (723 lines)

## What Was Ingested

**Core Topic**: Goal-conditioned reinforcement learning and hindsight experience replay

**Key Sources**:
- NeurIPS 2024: GCPO framework for on-policy goal-conditioned RL
- Andrychowicz et al. NeurIPS 2017: Original HER paper
- arXiv:2406.03870: GOOSE for safety-critical scenarios

## Key Concepts Captured

### 1. Goal Representations
- State-based goals: s_g
- Feature-based goals: φ(s_g)
- Distance-based goals: d_g
- Latent goal embeddings

### 2. Hindsight Experience Replay (HER)
- **Core idea**: Learn from failure by relabeling goals
- **Strategies**: Future, final, episode, random
- **Impact**: 10× fewer episodes to converge
- **Cost**: 5× storage overhead, still 2× more sample efficient

### 3. Goal-Conditioned Value Functions
- Q(s, a, g): Action value for goal g
- V(s, g): State value for goal g
- Universal Value Function Approximators (UVFA)
- Multi-head learning (value + success prediction)

## TRAIN STATION: Goal = Affordance = EFE = Planning

**The Unification**: All answer "What future states can I reach from here?"

**Four Equivalent Framings**:
1. **Goal-Conditioned RL**: π(a|s,g) - policy for goal g
2. **Affordance Detection**: Goal g is afforded if reachable
3. **Expected Free Energy**: EFE(s,a,g) - minimize energy to reach g
4. **Amortized Planning**: Policy IS learned planner

**Coffee Cup = Donut**:
```python
# These are THE SAME FUNCTION:
goal_policy(s, g)      # GCRL: action for goal
affordance_detect(s, g) # Gibson: can I reach g?
minimize_efe(s, g)     # Friston: min free energy
implicit_plan(s, g)    # Planning: first action of plan

# Different names, same computation!
```

## ML-HEAVY Implementation

**Complete HER Agent** (200+ lines):
- GoalConditionedActor/Critic networks
- HER replay buffer with k=4 relabeling
- Future strategy hindsight goals
- Soft target network updates
- Full training loop with success metrics

**Performance**:
- Storage: 5× overhead (k=4 hindsight goals)
- Sample efficiency: 2× better despite storage cost
- Convergence: 10× fewer episodes vs no HER

## ARR-COC Connection (10%)

**Goal-Conditioned Relevance**:
- Relevance scores conditioned on processing goal
- Different goals → different token selection
- QA goal: Select answer-relevant tokens
- Summary goal: Select key-point tokens

**Hindsight Relevance**:
- Learn from both intended and achieved goals
- "Tokens that led to output were relevant FOR that output"
- Even if output wasn't intended goal

**Token Allocation**:
- Budget allocation based on goal
- Same input, different processing per goal
- Goal = what future computation state to reach

## Code Highlights

**HER Relabeling**:
```python
# For each transition t:
for _ in range(k_hindsight):
    # Sample future state as goal
    future_t = np.random.randint(t, T)
    hindsight_goal = trajectory[future_t]['next_state']
    # Store: "If I wanted to reach THIS, I succeeded!"
```

**Goal-Conditioned Network**:
```python
class GoalConditionedPolicy(nn.Module):
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.policy(x)  # π(a | s, g)
```

**Unified Framework**:
```python
# One network, four interpretations
def universal_goal_function(state, goal):
    return network(state, goal)

# Call it whatever you want:
# - Goal-conditioned action
# - Affordance reachability
# - EFE minimization
# - Plan's first step
```

## What This Enables

**For Karpathy Oracle**:
1. Understanding goal-conditioned systems
2. HER as curriculum learning technique
3. Connection to affordances and active inference
4. Practical implementation patterns

**For ML Practice**:
- Sparse reward problems → dense via HER
- Multi-task learning via goals
- Transfer across goals
- Sample-efficient exploration

**For TRAIN STATIONS**:
- Goal ≡ Affordance connection
- RL ≡ Active Inference bridge
- Learning ≡ Planning unification

## Integration Points

**Connects to**:
- Affordance detection (PART 31)
- Expected free energy (Active Inference batch)
- World models (PART 35 - coming)
- Planning (active inference planning)

**ML Techniques**:
- Off-policy RL (HER requires off-policy)
- Replay buffers with augmentation
- Universal value functions
- Multi-task learning

## Statistics

- **Total lines**: 723
- **Code blocks**: 18
- **Major sections**: 6
- **Web sources**: 3
- **Key equations**: Q(s,a,g), V(s,g), EFE
- **TRAIN STATION**: Goals = Affordances = EFE = Planning
- **ARR-COC %**: 10%

## Next Steps

PART 35: World Models for Affordances
- Model-based RL with goals
- World models predict goal reachability
- Planning in learned latent space
- Dreamer architecture for goal-conditioned control
