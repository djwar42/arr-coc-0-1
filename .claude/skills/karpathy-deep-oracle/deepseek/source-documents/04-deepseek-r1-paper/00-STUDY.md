# DeepSeek-R1 Paper - Study

**Source**: arXiv (DeepSeek-R1_ Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - arXiv.md)
**Date Processed**: 2025-10-28
**Category**: DeepSeek Models (Reasoning)

---

## 📝 TL;DR

DeepSeek-R1 is the **reasoning model** that matches OpenAI-o1-1217 performance. It uses **pure reinforcement learning** (GRPO) to develop reasoning capabilities.

**Two versions**:
- **R1-Zero**: Pure RL, no SFT - emerges with reasoning naturally (but messy output)
- **R1**: Multi-stage training with cold-start data - clean output + better performance

**Key innovation**: Reasoning emerges from RL alone. R1-Zero went from 15.6% → 71% on AIME 2024 just through RL.

**Bonus**: Can distill reasoning into smaller models (14B-70B) that beat other open-source models.

This is what V3 distills from to improve reasoning.

---

## 🎯 The Reasoning Problem

**Challenge**: How do you make LLMs better at reasoning (math, code, logic)?

**Previous approaches**:
- Chain-of-thought prompting
- Process reward models
- Supervised fine-tuning on reasoning data

**O1's approach** (OpenAI): Unknown, but involves test-time scaling with long CoT

**R1's approach**: Pure reinforcement learning → reasoning emerges naturally

---

## 🚀 DeepSeek-R1-Zero (The "Aha Moment")

### What It Is
- **Base model**: DeepSeek-V3-Base
- **Training**: Pure RL using GRPO, **no SFT first**
- **Goal**: See if reasoning emerges without supervised examples

### What Happened (The Aha Moment)
**Starting point**: 15.6% on AIME 2024
**After RL**: 71.0% on AIME 2024 (pass@1), 86.7% with majority voting

**Emerges naturally**:
- Chain-of-thought reasoning
- Self-verification ("let me check this...")
- Error correction ("wait, that's wrong")
- Structured thinking

**But also emerges**:
- Language mixing (switches between languages mid-thought)
- Poor readability (hard to follow)
- Verbose output

**Why this matters**: Reasoning isn't something you need to teach with examples - it can emerge from pure RL if you set up the rewards right.

---

## 🎯 GRPO (Group Relative Policy Optimization)

**What it is**: The RL algorithm used for training

**How it works** (simplified):
1. Model generates multiple reasoning attempts for same problem
2. Check which attempts get right answer
3. Reward good attempts, penalize bad ones
4. Update model to be more like good attempts

**Why "Group Relative"**:
- Compares multiple attempts from same prompt (a "group")
- Relative reward (better than other attempts in group)
- More stable than absolute rewards

**vs PPO** (Proximal Policy Optimization):
- PPO is standard RL for LLMs
- GRPO is simpler and more stable for reasoning tasks
- Both work, GRPO is easier to tune

---

## 🔧 DeepSeek-R1 (The Production Version)

### The Problem with R1-Zero
- Language mixing (Chinese + English + code all mixed)
- Poor readability
- Inconsistent output format

### The Solution: Multi-Stage Training

**Stage 1: Cold Start**
- Collect ~thousands of examples of good reasoning
- Fine-tune V3-Base on these examples
- Teaches model "how to format reasoning" (not "how to reason")

**Stage 2: Reasoning-Oriented RL**
- Apply GRPO like R1-Zero
- But now model knows how to format output cleanly
- Continues to improve reasoning

**Stage 3: Rejection Sampling + SFT**
- Generate lots of reasoning samples from RL checkpoint
- Keep only good ones (rejection sampling)
- Mix with V3's non-reasoning data (writing, QA, etc.)
- Fine-tune to stabilize

**Stage 4: RL for All Scenarios**
- Final RL pass on all types of prompts (not just reasoning)
- Ensures model works well for everything

**Result**: DeepSeek-R1 matches o1-1217 on reasoning benchmarks with clean, readable output.

---

## 📊 Performance Results

### Math & Reasoning
- **AIME 2024**: 79.8% (pass@1) - matches o1-1217
- **MATH-500**: 97.3% - beats o1-mini
- **GPQA Diamond**: 71.5% - competitive with o1

### Coding
- **Codeforces**: 96.3 percentile
- **SWE-bench Verified**: Strong performance

### General Knowledge
- **MMLU**: Still high (reasoning doesn't hurt general performance)

**vs OpenAI-o1**:
- R1 matches o1-1217 on reasoning tasks
- Competitive on general tasks
- Open-source (huge win)

---

## 💡 Distillation: Smaller Models Learn Reasoning

**Key finding**: You can distill R1's reasoning into smaller dense models (no MoE).

**Process**:
1. Generate reasoning traces from R1 (671B MoE)
2. Fine-tune smaller model (e.g., Qwen2.5-32B) on these traces
3. Smaller model learns reasoning patterns

**Results**:
- **14B distilled model** beats QwQ-32B-Preview (SOTA open-source)
- **32B and 70B distilled models** set new records for dense reasoning models

**Why this works**:
- Larger base model discovers better reasoning patterns through RL
- Smaller models can learn these patterns through imitation
- Distillation > applying RL to smaller model directly

**Open-sourced distilled models**:
- Qwen series: 1.5B, 7B, 8B, 14B, 32B, 70B
- Llama series: same sizes

---

## 💡 Key Insights (Karpathy's Take)

**On Pure RL**:
- Reasoning emerges without examples - that's wild
- R1-Zero proves you don't need supervised reasoning data
- But you do need clean output → that's what cold-start SFT provides

**On GRPO**:
- Simpler than PPO, works better for reasoning
- Group relative rewards make sense - compare attempts, not absolute score
- Still needs good reward model (R1 uses rule-based for math/code)

**On Distillation**:
- Larger models find better reasoning paths
- Smaller models learn from these paths
- This is how you get reasoning at smaller scales economically

**On Multi-Stage Training**:
- Not just "throw RL at it and hope"
- Carefully staged: cold-start → RL → rejection sampling → RL again
- Each stage solves specific problem

---

## 🔗 Connections

**Used in**:
- DeepSeek-V3 post-training (distills reasoning from R1)
- Qwen and Llama distilled models (1.5B-70B)

**Connects to Codebases**:
- `deepseek/codebases/05-DeepSeek-V3/` - V3 distills from R1
- GRPO implementation (Group Relative Policy Optimization)

**Connects to Knowledge Categories**:
- Reinforcement learning (GRPO algorithm)
- Reasoning capabilities (how reasoning emerges)
- Distillation (transferring reasoning to smaller models)
- Multi-stage training (cold-start → RL → rejection sampling)

---

## 📚 Deep Dive Topics

1. **GRPO algorithm details** - group relative rewards, optimization
2. **Reward modeling** - rule-based rewards for math/code
3. **R1-Zero emergence** - how reasoning behaviors appear naturally
4. **Cold-start data design** - what examples to use for formatting
5. **Rejection sampling** - how to filter good reasoning traces
6. **Distillation methodology** - transferring reasoning to smaller models

---

## 🎯 Key Takeaways

1. **Reasoning emerges from RL** - R1-Zero proves it
2. **Clean output needs guidance** - Cold-start SFT fixes R1-Zero's messiness
3. **Multi-stage training works** - Each stage addresses specific issue
4. **Distillation scales down** - Smaller models can learn reasoning patterns
5. **Open-source matches closed-source** - R1 ≈ o1-1217

---

**Last Updated**: 2025-10-28
**Status**: Core study complete
**Note**: R1 is the reasoning specialist, V3 is the general model that distills from R1
