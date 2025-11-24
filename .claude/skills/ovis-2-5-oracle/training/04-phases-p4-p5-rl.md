# Phases P4-P5: RL Optimization

**Category**: Training
**Related**: [00-overview.md](00-overview.md), [../architecture/06-thinking-mode.md](../architecture/06-thinking-mode.md)

## Phase P4: DPO

**Method**: Direct Preference Optimization
**Data**: Preference pairs (chosen vs rejected)
**Size**: ~10M pairs

**Training**:
```yaml
learning_rate: 1e-5
loss: DPO + auxiliary NLL
```

**Goal**: Align with human preferences on reasoning quality.

## Phase P5: GRPO

**Method**: Group Relative Policy Optimization
**Trainable**: LLM only (vision frozen)
**Data**: Math problems with verifiable answers
**Size**: ~5M problems

**Training**:
```yaml
learning_rate: 5e-6
reward: Correct answer + reasoning quality
```

**Goal**: Optimize reasoning through RL.
