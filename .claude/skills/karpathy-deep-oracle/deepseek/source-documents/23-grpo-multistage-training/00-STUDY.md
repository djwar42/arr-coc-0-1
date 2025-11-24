# GRPO & Multi-Stage Training (Bavalpreet) - Study

**Source**: Medium (DeepSeek R1: Understanding GRPO and Multi-Stage Training by BavalpreetSinghh)
**Date Processed**: 2025-10-28
**Category**: Reinforcement Learning (GRPO + Training Pipeline)

---

## 📝 TL;DR

Detailed breakdown of R1's multi-stage training: cold-start SFT → RL with GRPO → rejection sampling → second SFT → final RL polish. GRPO uses group-relative rewards (no critic needed), making it simpler than PPO. Works.

---

## 🎯 Key Concepts

### Multi-Stage Pipeline (Deep Dive)

**Stage 1: Cold-Start SFT**
- ~4k CoT examples
- Teaches basic reasoning format
- Without this, RL struggles to find signal

**Stage 2: RL with GRPO**
- Rule-based rewards (correctness + format)
- Group-relative scoring (best output in group gets reward)
- Problem: Model becomes TOO focused on reasoning tasks
- Loses general capabilities (formatting, language consistency)

**Stage 3: Rejection Sampling**
- Generate multiple outputs from RL checkpoint
- Keep high-quality reasoning samples (~600k)
- Mix with general data from base model (~200k)
- Total: 800k diverse samples

**Stage 4: Second SFT**
- Train on combined 800k dataset
- Recovers general capabilities
- Maintains reasoning quality
- Creates balanced model

**Stage 5: Final RL Polish**
- One more RL pass to fine-tune
- Uses GRPO again
- Final model: DeepSeek-R1

### GRPO Algorithm

**Key Innovation**: No critic model needed

**Standard PPO**:
- Actor (policy) + Critic (value function)
- Critic estimates expected reward
- Requires training TWO models

**GRPO**:
- Only Actor (policy)
- Generate multiple outputs per prompt
- Rank outputs within group
- Best output gets positive reward, others negative
- No critic needed!

**Math**:
```
For prompt p, generate K outputs: {o1, o2, ..., oK}
Rewards: {r1, r2, ..., rK}
Group mean: r_mean = (r1 + r2 + ... + rK) / K

GRPO advantage: A_i = r_i - r_mean

Update policy based on A_i (relative to group, not absolute)
```

**Why This Works**:
- Relative rewards are more stable than absolute
- No need to train a critic
- Simpler to implement and tune
- Works well in practice

### Why Multi-Stage?

**Why not just RL?**: RL alone makes model too specialized, loses general capabilities

**Why not just SFT?**: SFT alone doesn't achieve strong reasoning (needs trial-and-error from RL)

**Solution**: Alternate between RL (improve reasoning) and SFT (recover general capabilities)

---

## 💡 Why This Matters

Shows the practical engineering behind R1. Not just "apply GRPO", but a careful multi-stage process that balances reasoning vs general capabilities.

Key insight: **RL makes models hyper-specialized**. You need to periodically "reset" with SFT to maintain balance.

---

## 🔧 Karpathy-Style Implementation Notes

**GRPO Pseudocode**:
```python
def grpo_train_step(policy, prompt, num_samples=4):
    # Generate multiple outputs
    outputs = [policy.generate(prompt) for _ in range(num_samples)]

    # Get rewards (e.g., correctness score)
    rewards = [reward_fn(output) for output in outputs]

    # Compute group mean
    mean_reward = sum(rewards) / len(rewards)

    # Relative advantages
    advantages = [r - mean_reward for r in rewards]

    # Update policy (standard policy gradient)
    for output, advantage in zip(outputs, advantages):
        loss = -advantage * log_prob(output | prompt)
        loss.backward()
```

**Multi-Stage Flow**:
```python
# Stage 1
model = base_model.finetune(cold_start_data)  # Few thousand CoT samples

# Stage 2
model = grpo_train(model, rl_data)  # Focus on reasoning

# Stage 3
rejection_samples = rejection_sample(model, prompts, threshold)
combined_data = rejection_samples + general_data

# Stage 4
model = model.finetune(combined_data)  # Recover general capabilities

# Stage 5
model = grpo_train(model, final_rl_data)  # Final polish

return model  # DeepSeek-R1!
```

---

## 🔗 Connections

- **04-deepseek-r1-paper**: Official R1 paper
- **07-grpo-theory**: GRPO algorithm theory
- **20-illustrated-grpo**: Visual GRPO guide
- **22-understanding-r1-christian**: Accessible R1 explainer

---

## 💭 Karpathy Take

This is basically a detailed playbook for training a reasoning model. The multi-stage pipeline is the key innovation - not GRPO itself (which is clever but fairly straightforward), but HOW you use it in stages while maintaining model balance.

The rejection sampling step is smart. You use your RL checkpoint to generate training data for itself. Classic self-improvement loop. Filter for quality, mix with general data so the model doesn't forget how to be a normal LLM, then retrain.

GRPO's "no critic" thing is nice from an engineering perspective. PPO requires training and maintaining a separate critic model, which is annoying. GRPO just generates multiple outputs and ranks them relative to each other. Simpler = better.

The cold-start SFT is crucial. Without it, RL flails around trying to discover what "good reasoning" looks like. With a few thousand examples, RL immediately knows "oh we want  <think>  tags with step-by-step logic" and can optimize from there.

Bottom line: Training reasoning models isn't magic. It's careful engineering - alternate between RL (optimize for task) and SFT (maintain general capabilities), use rejection sampling to create high-quality data, repeat until good. ¯\_(ツ)_/¯
