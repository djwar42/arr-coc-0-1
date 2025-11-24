# Understanding DeepSeek R1 (Christian Houmann) - Study

**Source**: Christian B. B. Houmann Blog (bagerbach.com)
**Date Processed**: 2025-10-28
**Category**: DeepSeek Models (R1 Reasoning)

---

## 📝 TL;DR

R1 is DeepSeek's open-source reasoning model (MIT licensed!) that matches or beats OpenAI's o1 in benchmarks. Uses pure RL (R1-Zero) or multi-stage training (R1) to learn Chain-of-Thought reasoning. Costs $0.14-0.55/M input tokens vs o1's $15. Pretty wild.

---

## 🎯 Key Concepts

### R1 vs R1-Zero

**R1-Zero** (Pure RL):
- RL directly on DeepSeek-V3-Base
- NO supervised fine-tuning
- Reasoning emerges purely from RL
- Problem: Language mixing, formatting issues

**R1** (Multi-Stage):
- Cold-start SFT → RL → Rejection sampling → SFT again → Final RL
- Better readability and formatting
- Maintains reasoning quality
- Production-ready

### The Multi-Stage Pipeline

**Stage 1: Cold-Start Fine-Tuning**
- Few thousand CoT samples
- Gives RL a decent starting point
- Otherwise RL struggles to find signal

**Stage 2: First RL (GRPO)**
- Rule-based rewards for correctness + formatting
- Forces `<think>` tags for CoT
- Result: Strong reasoning, weak general capabilities

**Stage 3: Rejection Sampling + General Data**
- Collect 600k high-quality reasoning samples
- Add 200k general task samples
- Total: 800k diverse examples

**Stage 4: Second Fine-Tuning**
- Train on combined 800k samples
- Recovers general capabilities
- Maintains reasoning quality

**Stage 5: Final RL**
- Polish everything
- Final model: R1

### Inference-Time Scaling

**Traditional Scaling**: More data + more compute during training
**Reasoning Scaling**: More compute during inference (longer CoT)

R1 demonstrates that inference-time scaling works. The model literally "thinks" longer to solve harder problems.

### Cost Comparison

**DeepSeek R1**:
- Input: $0.14-0.55 per million tokens
- Output: $2.19 per million tokens

**OpenAI o1**:
- Input: $15 per million tokens
- Output: $60 per million tokens

~27-100x cheaper 🤯

---

## 💡 Why This Matters

**First open-source reasoning model** that actually matches closed models like o1. MIT licensed, fully transparent training methodology, and crazy cheap to run.

The multi-stage pipeline shows how to bootstrap reasoning:
1. Cold-start with small SFT dataset
2. Let RL discover reasoning patterns
3. Fix emergent problems (language mixing) with targeted SFT
4. Repeat

**Key insight**: You don't need massive supervised CoT datasets. A few thousand examples + RL is enough.

---

## 🔧 Karpathy-Style Implementation Notes

**The Cold-Start Problem**:
```python
# Don't do this (won't work):
base_model → pure_RL → ???

# Do this:
base_model → small_SFT(few_thousand_COT) → RL → works!
```

**Why cold-start matters**: RL needs *some* signal to know what "good reasoning" looks like. Without SFT cold-start, RL flails around and might never discover CoT patterns.

**Rejection Sampling**:
```python
# Generate multiple outputs, keep only the good ones
for prompt in dataset:
    outputs = [model.generate(prompt) for _ in range(N)]
    good_outputs = [o for o in outputs if reward(o) > threshold]
    sft_data.extend(good_outputs)
```

This creates high-quality training data from your RL checkpoint.

**Multi-Stage is Key**:
- Stage 1-2: Teach reasoning (accuracy focus)
- Stage 3-4: Recover general capabilities (don't lose what the base model knew)
- Stage 5: Final polish

---

## 🔗 Connections

- **04-deepseek-r1-paper**: Official technical paper (this is the accessible explainer)
- **07-grpo-theory**: GRPO algorithm explained
- **20-illustrated-grpo**: Visual guide to GRPO
- **01-deepseek-v3-technical-report**: R1 built on V3-Base

---

## 💭 Karpathy Take

Okay so this is actually a really nice writeup. Christian does a good job explaining the R1 pipeline without getting too deep in the weeds.

The big takeaway: **You don't need a giant supervised CoT dataset to teach reasoning**. A few thousand examples for cold-start, then let RL figure it out. The model discovers reasoning patterns on its own through trial and error.

R1-Zero is wild - pure RL with no SFT, and it still learns to reason. But yeah, you get weird behavior like language mixing (model literally switches languages mid-response because some concepts are easier to express in Chinese or whatever). R1 fixes that with the multi-stage pipeline.

The cost numbers are insane. $0.55/M input tokens vs $15/M for o1. That's not a typo. You can run R1 for lunch money.

And it's MIT licensed! You can literally download the weights and fine-tune it yourself. Compare that to o1 which is closed-source and you have to use through OpenAI's API.

The inference-time scaling thing is interesting too. Traditional scaling = throw more compute at training. Reasoning scaling = throw more compute at inference (let the model "think" longer). Both work, but reasoning scaling means you can make a smaller model perform way better by just giving it more time to think.

Pretty cool that they published the full pipeline with all the intermediate checkpoints and their problems. That's the kind of transparency that actually moves the field forward.

**tl;dr**: Open reasoning model that beats o1 on some benchmarks, costs ~50x less, MIT licensed, full training details published. This is what open AI looks like. ¯\_(ツ)_/¯
