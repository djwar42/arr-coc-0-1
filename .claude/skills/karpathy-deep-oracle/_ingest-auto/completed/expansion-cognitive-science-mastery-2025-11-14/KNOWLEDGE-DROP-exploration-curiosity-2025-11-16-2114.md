# KNOWLEDGE DROP: Exploration Bonuses & Curiosity

**Date**: 2025-11-16 21:14
**Part**: PART 24
**File Created**: `cognitive-mastery/23-exploration-bonuses-curiosity.md`
**Lines**: ~700

## What Was Created

Comprehensive knowledge file covering intrinsic motivation through exploration bonuses and curiosity-driven learning, including:

1. **Count-Based Exploration** - Visit counts, pseudo-counts, UCB algorithms
2. **Prediction-Based Curiosity** - Forward dynamics, ICM, RND, noisy-TV problem
3. **Empowerment** - Information-theoretic formulation, channel capacity, Lyapunov connections
4. **Exploration vs Exploitation** - Fundamental trade-off, meta-learning curiosity
5. **Engineering** - FSDP, torch.compile, TPU deployment for curiosity
6. **Recent Advances** - LLM reasoning, embodied AI, biological inspiration (2024-2025)
7. **ARR-COC-0-1 Integration** - Token allocation as exploration, patch novelty bonuses

## Key Research Findings

### Intrinsic Motivation Methods

**Count-Based (Classical)**:
- Visit count bonuses: `r = 1/sqrt(N(s))`
- UCB exploration: `UCB(a) = Q(a) + c*sqrt(log(t)/N(a))`
- Pseudo-counts for continuous spaces via density models

**Prediction-Based (Modern)**:
- ICM (Intrinsic Curiosity Module): Reward prediction errors
- RND (Random Network Distillation): Predict random fixed network
- Inverse dynamics filter out uncontrollable noise
- Solution to "noisy-TV problem"

**Empowerment (Information-Theoretic)**:
- Maximize `I(Actions; Future_States | current_state)`
- Not just entropy, but **controllable** diversity
- Computed via channel capacity in linear regime
- Connects to "controlled Lyapunov exponents"

### Major Insights

From **PRX Life (Tiomkin et al., 2024)**:
> "Empowerment is not just the entropy of a passive diffusion process, but of the subprocess that the agent can actively generate."

**Key Distinction**:
- **Entropy**: All possible futures (including random noise)
- **Empowerment**: Futures specifically caused by agent's actions

From **MIT Neural Computation**:
> "Prediction error curiosity rewards agents for discovering observations they cannot accurately predict. However, such agents may be attracted to stochastic processes."

**Solution**: Use inverse dynamics to learn features that filter controllable from uncontrollable.

### Real-World Applications

**LLM Reasoning (2024)**:
- MERCI: Count-based curiosity for chain-of-thought reasoning
- Rewards novel reasoning patterns
- 3-7% improvement on MATH, GSM8K benchmarks

**Robotics (2025)**:
- Cross-modal curiosity (visual + semantic)
- Dual-protocol exploration (novelty + semantic meaning)
- Embodied AI learning object manipulation

**Biological Models**:
- Extinction burst explained by curiosity-driven exploration
- High prediction error when reward stops → increased exploration
- Non-monotonic learning curves

## Technical Implementation

### Empowerment Computation (PRX Life)

**Linear Gaussian Channel Capacity**:
```
Empowerment = sum_i log(1 + λ_i * P_i / σ^2)
```
where `λ_i` = singular values of sensitivity matrix `∂S_future/∂A`.

**Algorithm**:
1. Compute sensitivity matrix from dynamics
2. SVD to get singular values
3. Water-filling for power allocation
4. Channel capacity = empowerment

**Result**: Balanced inverted pendula without task-specific reward.

### Production Deployment

**FSDP for Distributed Curiosity**:
- Shard curiosity computation across GPUs
- Scale to millions of states

**torch.compile for Speed**:
- 2-3x faster curiosity bonus computation
- Real-time for 30 Hz robotics

**TPU for Massive Parallelism**:
- Millions of parallel environments
- Matrix-heavy prediction models

## ARR-COC-0-1 Connections

### Token Allocation as Exploration

**Framing**:
- State: Patch features
- Action: Token budget (64-400)
- Intrinsic reward: Patch novelty/unpredictability

**Training Schedule**:
- Early: High β_curiosity → explore all patch types
- Mid: Balance novelty + query relevance
- Late: Low β_curiosity → exploit learned patterns

### Patch Curiosity Scoring

```python
# Prediction error for novel patches
pred_next = forward_model(patch_t)
curiosity = (pred_next - patch_{t+1}).pow(2).mean()

# Empowerment for representational flexibility
empowerment = I(token_budget; future_query_success | patch)
```

**Benefit**: Prioritize informative, diverse patches during training.

## Sources Used

**Key Papers**:
1. PRX Life - Tiomkin et al. (2024) - Empowerment in dynamical systems
2. MIT Neural Computation (2024) - Intrinsic rewards without harm
3. arXiv:2510.16614 - MERCI for LLM reasoning
4. arXiv:2509.09675 - CDE curiosity-driven exploration

**Technical Resources**:
- ACM DL - Unifying count-based exploration
- Springer - Impact of intrinsic rewards comparison
- bioRxiv - Extinction burst via curiosity
- GitHub implementations

**Total Web Sources**: 13 (all 2024-2025)

## Quality Checklist

- [x] Covers all required topics (count-based, prediction, empowerment, explore/exploit)
- [x] Cites 13+ web sources with URLs and access dates
- [x] References Files 4, 8, 16 (FSDP, torch.compile, TPU)
- [x] 10% ARR-COC-0-1 integration (Section 8)
- [x] Recent research (2024-2025)
- [x] Engineering implementations (production patterns)
- [x] ~700 lines of content

## Next Steps

PART 24 complete. Ready for oracle consolidation after all 42 parts finish.
