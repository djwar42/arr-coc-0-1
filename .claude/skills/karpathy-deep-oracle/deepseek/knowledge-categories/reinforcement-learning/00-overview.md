# Reinforcement Learning - Overview

**Category**: RL techniques for reasoning and post-training
**Source Documents**: DeepSeek-R1 paper, V3 technical report
**Date Created**: 2025-10-28

---

## 🎯 What This Category Covers

How DeepSeek uses RL to develop reasoning capabilities:
- **GRPO** - Group Relative Policy Optimization
- **Pure RL** - R1-Zero emerges reasoning without SFT
- **Multi-stage training** - Cold-start → RL → rejection sampling → RL
- **Distillation** - Transferring reasoning to smaller/general models

---

## 📊 Core Techniques

### 1. GRPO (Group Relative Policy Optimization)
**What**: RL algorithm for training reasoning models
**Why**: Simpler and more stable than PPO for reasoning tasks
**Impact**: R1 matches o1-1217, R1-Zero achieves 71% on AIME from pure RL

**How it works**:
- Generate multiple reasoning attempts for same problem
- Compare within group (relative reward)
- Update model to favor better attempts
- More stable than absolute rewards

**Source**: DeepSeek-R1
**Details**: [01-grpo-algorithm.md](01-grpo-algorithm.md)

### 2. Reasoning Emergence (R1-Zero)
**What**: Pure RL training without SFT - reasoning emerges naturally
**Why**: Tests if reasoning can develop without supervised examples
**Impact**: 15.6% → 71% on AIME 2024 (86.7% with majority voting)

**What emerges**:
- Chain-of-thought reasoning
- Self-verification ("let me check...")
- Error correction
- Structured thinking

**Source**: DeepSeek-R1
**Details**: [02-reasoning-emergence.md](02-reasoning-emergence.md)

### 3. Multi-Stage Training Pipeline
**What**: Staged approach to develop reasoning with clean output
**Why**: R1-Zero has messy output (language mixing, poor readability)
**Impact**: R1 matches o1-1217 with clean, readable reasoning

**Stages**:
1. **Cold-start SFT** - Teach output formatting (not reasoning)
2. **Reasoning-oriented RL** - Apply GRPO like R1-Zero
3. **Rejection sampling + SFT** - Stabilize with good examples
4. **RL for all scenarios** - Final polish on all prompt types

**Source**: DeepSeek-R1
**Details**: [03-multi-stage-training.md](03-multi-stage-training.md)

### 4. Reasoning Distillation
**What**: Transfer reasoning from R1 (671B MoE) to smaller/general models
**Why**: R1 is for reasoning, V3 is general - distillation combines both
**Impact**: V3 gains reasoning, 14B models beat 32B+ competitors

**Process**:
- Generate reasoning traces from R1
- Fine-tune target model on traces
- Model learns reasoning patterns

**Source**: DeepSeek-R1, V3 post-training
**Details**: [04-reasoning-distillation.md](04-reasoning-distillation.md)

---

## 🚀 The R1 Journey

### R1-Zero (Pure RL)
**Start**: DeepSeek-V3-Base (671B MoE)
**Training**: GRPO only, no SFT
**Result**: 71% AIME 2024 (from 15.6%)

**What emerged** (not explicitly taught):
- Chain-of-thought reasoning
- Self-verification loops
- Error detection and correction
- Mathematical reasoning patterns

**Problems**:
- Language mixing (Chinese + English randomly)
- Poor readability
- Verbose, hard to follow

**Lesson**: Reasoning emerges from RL, but formatting doesn't.

### R1 (Production Version)
**Improvements**:
- Cold-start SFT for formatting
- Multi-stage training for stability
- Matches o1-1217 on benchmarks
- Clean, readable output

**Performance**:
- **AIME 2024**: 79.8% (pass@1)
- **MATH-500**: 97.3%
- **GPQA Diamond**: 71.5%
- **Codeforces**: 96.3 percentile

**Trade-off**: More complex training, but production-ready.

---

## 💡 Key Insights (Karpathy's Take)

**On pure RL emergence**:
- Reasoning isn't something you need to teach with examples
- RL discovers reasoning patterns through trial and error
- This is genuinely surprising - R1-Zero proves it works

**On GRPO vs PPO**:
- Simpler is better for reasoning
- Group-relative rewards more stable than absolute
- PPO works too, but GRPO is easier to tune

**On multi-stage training**:
- Can't just "throw RL at it and hope"
- Each stage solves specific problem:
  - Cold-start: Format output
  - RL: Develop reasoning
  - Rejection sampling: Stabilize
  - Final RL: Polish everything

**On distillation**:
- Larger models find better reasoning paths
- Smaller models learn from these paths
- Distillation > training smaller model with RL directly

---

## 📈 Distillation Results

### R1 → DeepSeek-V3
**Process**: Distill reasoning from R1 into V3 during post-training
**Result**: V3 gains reasoning without being reasoning-specialized
**Benefit**: General model + reasoning capabilities

### R1 → Qwen/Llama (1.5B-70B)
**Process**: Fine-tune smaller dense models on R1 reasoning traces
**Results**:
- **14B model** beats QwQ-32B-Preview (SOTA open-source)
- **32B and 70B** set new records for dense reasoning models

**Key finding**: Distillation beats direct RL on smaller models.

---

## 🔧 Technical Details

### Reward Modeling
**For math/code**:
- Rule-based rewards (answer is right/wrong)
- No learned reward model needed
- Simpler and more reliable

**For general reasoning**:
- May need learned reward model
- R1 focuses on verifiable domains first

### Training Stability
**Challenges**:
- RL can be unstable (loss spikes, divergence)
- Reasoning tasks have sparse rewards

**Solutions**:
- Group relative rewards (GRPO)
- Careful hyperparameter tuning
- Multi-stage approach

---

## 🔗 Cross-References

**Connects to Model Architectures**:
- R1 uses V3-Base architecture (671B MoE)
- MoE enables economical RL training

**Connects to Training Efficiency**:
- R1 training is economical (only 5K GPU hours for post-training)
- GRPO is compute-efficient

**Connects to Codebases**:
- `deepseek/codebases/05-DeepSeek-V3/` - V3 distills from R1

**Connects to Source Documents**:
- [R1 Paper](../../source-documents/04-deepseek-r1-paper/00-STUDY.md)
- [V3 Technical Report](../../source-documents/01-deepseek-v3-technical-report/00-STUDY.md)

---

## 🎯 Key Takeaways

1. **Reasoning emerges from RL** - R1-Zero proves it
2. **GRPO is effective** - Simpler than PPO, works better for reasoning
3. **Multi-stage training essential** - Each stage addresses specific issue
4. **Distillation scales** - Transfer reasoning to smaller/general models
5. **Verifiable domains first** - Math/code easier than open-ended reasoning

---

**Last Updated**: 2025-10-28
**Status**: Active - core category for understanding DeepSeek's reasoning approach
