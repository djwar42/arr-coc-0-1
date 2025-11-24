# Pure RL for Math Reasoning Deep Dive
**Enhanced**: 2025-10-29
**Sources**: RUG Bachelor Thesis, DeepSeek-R1-Zero documentation, RLVR research
**Category**: COMPREHENSIVE TECHNICAL ANALYSIS

---

## 🎯 Executive Summary

**The Bold Claim**: You can train a math reasoning model with ZERO supervised examples - just pure reinforcement learning.

**Traditional Pipeline**:
```
Pre-trained LLM → SFT (supervised examples) → RL fine-tuning → Reasoning model
```

**Pure RL Pipeline**:
```
Pre-trained LLM → RL (no supervision!) → Reasoning model
```

**Key Results**:
- DeepSeek-R1-Zero: Pure RL, no SFT, achieves strong reasoning (self-verification emerges!)
- Qwen2.5-Math: 36% → 73.6% on MATH500 with just 1 training example + RL
- GRPO on small models: 15-25% improvement on GSM8K without any supervised data

**Why This Matters**:
- No need for expensive human-labeled reasoning chains
- Emergent behaviors (self-correction, reflection) appear automatically
- Scales to domains where supervision is impossible (novel math problems)

---

## 🔬 Core Insight: Why Pure RL Works for Math

### The Verification Advantage

**Math has a unique property**: Verifiable correctness

```python
Problem: "What is 2 + 2?"
Answer: "4" → check(2 + 2 == 4) → ✅ Reward = +1
Answer: "5" → check(2 + 2 == 5) → ❌ Reward = -1
```

**Compare to other domains**:
- Creative writing: Who judges quality? (subjective)
- Translation: Requires human evaluation (expensive)
- **Math: Self-verifying** (can be programmatically checked)

This makes math ideal for pure RL:
```
Generate solution → Execute/verify → Reward → Update policy
```

No humans needed!

### The Cold Start Problem (Solved)

**Traditional belief**: "You need SFT to bootstrap RL"

**Why?**
- Random policy produces garbage
- Garbage solutions get no reward
- Model never learns anything (stuck in local minimum)

**Pure RL solution**: Large pre-trained models already have latent reasoning capability

**Example (Qwen2.5-Math-1.5B baseline)**:
```
Prompt: "Solve: 2x + 5 = 13"

Initial attempts (before RL):
- "x = 7" (wrong, but close!)
- "x = 8" (correct! but rare)
- "x = 3" (wrong)
- "x = 5" (wrong)

Success rate: ~5-10% (enough to bootstrap RL!)
```

**Key insight**: Pre-training gives you a weak base policy. RL amplifies the signal.

---

## 💻 Implementation: Pure RL Training Loop

### Algorithm: GRPO (Group Relative Policy Optimization)

**Standard RLHF**:
```
1. Sample N solutions per problem
2. Rank by absolute reward
3. Update policy toward top-k
```

**GRPO**:
```
1. Sample N solutions per problem
2. Rank WITHIN each problem (relative)
3. Update policy toward relative best
```

**Why relative ranking matters**:
- Hard problems: All solutions wrong → still learn from "least wrong"
- Easy problems: All solutions correct → learn from "most elegant"
- Normalizes difficulty (prevents easy problem bias)

### Pseudocode

```python
def pure_rl_training(model, problems, num_epochs=1000):
    """
    Train model with pure RL (no SFT)
    """
    for epoch in range(num_epochs):
        for problem in problems:
            # Sample N solutions
            solutions = [model.generate(problem) for _ in range(N)]

            # Verify each solution
            rewards = [verify_solution(problem, sol) for sol in solutions]

            # GRPO: Rank within this problem
            sorted_idx = np.argsort(rewards)  # worst to best

            # Compute advantages (relative to mean)
            mean_reward = np.mean(rewards)
            advantages = rewards - mean_reward

            # Update policy
            for i, sol in enumerate(solutions):
                # Higher advantage → reinforce more
                # Lower advantage → suppress
                log_prob = model.log_prob(sol | problem)
                loss = -advantages[i] * log_prob
                loss.backward()

            optimizer.step()
```

### Verification Function

**For GSM8K (grade school math)**:
```python
def verify_solution(problem, solution):
    """
    Extract final answer and check correctness
    """
    # Parse: "The answer is 42"
    predicted = extract_answer(solution)

    # Get ground truth
    ground_truth = problem.answer

    # Binary reward
    if predicted == ground_truth:
        return 1.0  # Correct!
    else:
        return -1.0  # Wrong
```

**For MATH (competition math)**:
```python
def verify_solution(problem, solution):
    """
    More sophisticated verification
    """
    # Extract final boxed answer
    predicted = extract_boxed_answer(solution)  # \boxed{42}

    # Symbolic math check
    try:
        # Parse as sympy expression
        pred_expr = sympify(predicted)
        gt_expr = sympify(problem.answer)

        # Check equivalence (not just string match)
        if simplify(pred_expr - gt_expr) == 0:
            return 1.0
        else:
            return -1.0
    except:
        return -1.0  # Parse error = wrong
```

---

## 📊 Experimental Results

### Experiment 1: DeepSeek-R1-Zero

**Setup**:
- Base model: DeepSeek-V3 (pre-trained, no SFT)
- Training: Pure GRPO on math/code problems
- Evaluation: AIME, MATH, GSM8K

**Results**:

| Benchmark | Base Model | R1-Zero (Pure RL) | R1 (RL + SFT) |
|-----------|------------|-------------------|---------------|
| GSM8K | 72.3% | 91.6% (+19.3) | 97.3% |
| MATH | 34.1% | 71.2% (+37.1) | 79.8% |
| AIME 2024 | 10% | 53% (+43) | 79% |

**Key Observations**:
1. Pure RL works! (huge improvements over base)
2. SFT still helps (+6-8% over pure RL)
3. Emergent behaviors appear in pure RL:
   - Self-verification: "Let me check: 2+2=4 ✓"
   - Reflection: "Wait, that doesn't look right..."
   - Chain-of-thought (without any CoT examples!)

### Experiment 2: 1-Shot RLVR (Qwen2.5-Math)

**Setup**:
- Model: Qwen2.5-Math-1.5B
- Training: RL with just 1 example!
- Method: Carefully select the single most informative example

**Results**:

| Training Data | MATH500 Accuracy |
|---------------|------------------|
| Base model | 36.0% |
| Random 1 example + RL | 48.2% |
| Best 1 example + RL | 73.6% (!!) |
| Full SFT (1000s examples) | 76.4% |

**Holy shit**: One example + RL ≈ thousands of supervised examples!

**The magic example** (simplified):
```
Problem: "A rectangle has perimeter 20 and area 24. What is its diagonal?"

Solution (step-by-step):
Let length = l, width = w
Perimeter: 2l + 2w = 20 → l + w = 10
Area: lw = 24
Substitute w = 10 - l: l(10-l) = 24
Expand: 10l - l² = 24 → l² - 10l + 24 = 0
Factor: (l-6)(l-4) = 0 → l = 6 or l = 4
If l = 6, then w = 4 (or vice versa)
Diagonal: √(6² + 4²) = √52 = 2√13

The answer is 2√13
```

**Why this example works**:
- Shows multi-step reasoning
- Demonstrates algebraic manipulation
- Has verification steps
- Teaches problem decomposition

RL then generalizes these patterns to new problems!

### Experiment 3: Small Model Pure RL (RUG Thesis)

**Setup**:
- Models: Qwen2-Math-1.5B, Qwen2.5-Math-1.5B
- Training: Pure GRPO on GSM8K subset
- No SFT, no examples

**Results**:

| Model | Baseline | After Pure RL | Improvement |
|-------|----------|---------------|-------------|
| Qwen2-Math-1.5B | 58.3% | 73.1% | +14.8% |
| Qwen2.5-Math-1.5B | 67.8% | 82.4% | +14.6% |

**Training efficiency**:
- Epochs: 3-5 (very few!)
- Time: ~12 hours on single A100
- Data: 1000 problems, 8 samples each

**Insight**: Pure RL is surprisingly sample-efficient for math

---

## 🧠 Emergent Behaviors

### Self-Verification

**Appears spontaneously** (never trained on examples with self-verification):

```
Problem: "What is 15% of 80?"

R1-Zero output:
Let me calculate 15% of 80.
15% = 15/100 = 0.15
0.15 × 80 = 12

Let me verify: 12/80 = 0.15 = 15% ✓
The answer is 12.
```

**Why it emerges**:
- RL rewards correct answers
- Self-verified solutions are more likely correct
- Model learns "if I verify, my reward is higher"
- Verification behavior gets reinforced

### Reflection and Self-Correction

**Example**:
```
Problem: "If x² = 16, what is x?"

Initial attempt:
x² = 16
x = 4
Wait, let me think about this more carefully.
x² = 16 means x·x = 16
This could be x = 4 or x = -4!
But the problem asks for "what is x", suggesting a single answer.
If no constraints given, both solutions are valid.
The answer is x = ±4.
```

**This is insane**: The model learned to doubt itself and reconsider!

### Chain-of-Thought Reasoning

**Appears without CoT training data**:

Pure RL models spontaneously generate reasoning chains:
- "First, let's..."
- "Now I need to..."
- "This means..."
- "Finally..."

**Why?** Intermediate reasoning steps increase probability of correct final answer.

RL discovers: Reasoning chains → higher reward → reinforcement

---

## 💡 Theoretical Explanation: Why Pure RL Works

### The Credit Assignment Problem

**Challenge**: How does RL know which part of the solution was good/bad?

**Sparse reward**:
```
Long solution (20 steps) → Final answer → Reward (+1 or -1)
Which of the 20 steps were good?
```

**Solution**: Self-consistency as implicit supervision

```
Sample 8 solutions:
- Solution 1: "42" (via method A)
- Solution 2: "42" (via method B)
- Solution 3: "42" (via method C)
- Solution 4: "17" (outlier)
- ...

Majority vote: "42" is probably correct
RL reinforces solutions 1, 2, 3 (even if we don't have ground truth!)
```

This is **self-training RL**: Use model's own consistency to generate pseudo-labels.

### Pre-training as Initialization

**Key insight**: Large models already have latent reasoning

Pre-training on web text includes:
- Math textbooks
- Solution explanations
- Worked examples

The model "knows how to reason" but doesn't "know when to reason"

**Pure RL's role**: Activate latent reasoning capability

```
Pre-training: "Here are patterns that sometimes appear (reasoning)"
Pure RL: "Use those patterns whenever you see math problems!"
```

---

## 🔧 Practical Considerations

### When Pure RL Works

✅ **Good fit**:
- Verifiable correctness (math, code, theorem proving)
- Pre-trained model has latent capability (>50B params)
- Expensive to collect supervised data
- Want emergent behaviors (self-verification)

❌ **Not ideal**:
- Subjective tasks (creative writing)
- No verification function (open-ended QA)
- Very small models (<1B params)
- Need fast convergence (SFT is faster)

### Hyperparameter Sensitivity

**Critical parameters**:

| Parameter | Value | Impact if wrong |
|-----------|-------|-----------------|
| Num samples (N) | 8-16 | Too low: noisy gradients; Too high: slow |
| Learning rate | 1e-6 | Too high: instability; Too low: slow |
| Group size | 8 | GRPO needs groups for relative ranking |
| KL penalty | 0.1 | Too high: no learning; Too low: mode collapse |

**Most sensitive**: Number of samples per problem (N)
- N=4: High variance, unstable
- N=8: Good balance
- N=16: Better, but 2x slower

### Training Tips

**1. Start with strong base model**:
```python
# Good: Math-specialized pre-trained model
base_model = "Qwen2.5-Math-1.5B"  # Already has math knowledge

# Bad: General pre-trained model
base_model = "GPT-2"  # Will struggle
```

**2. Verify your verification function**:
```python
# Common bugs
def bad_verify(problem, solution):
    # BUG: String matching (won't handle "42" vs "42.0")
    return solution.strip() == problem.answer.strip()

def good_verify(problem, solution):
    # Use symbolic math library
    return sympy.simplify(solution - problem.answer) == 0
```

**3. Use curriculum learning**:
```python
# Start with easy problems, gradually increase difficulty
problems = sort_by_difficulty(all_problems)
for epoch in range(num_epochs):
    threshold = min(0.1 * epoch, 1.0)  # 0% → 100% difficulty
    current_problems = [p for p in problems if p.difficulty < threshold]
    train_on(current_problems)
```

---

## 📈 Connection to DeepSeek-R1

**R1 uses two-stage approach**:

**Stage 1: R1-Zero (Pure RL)**:
```
DeepSeek-V3 (base) → GRPO (pure RL) → R1-Zero
```
Results: Strong reasoning, emergent self-verification

**Stage 2: R1 (RL + SFT)**:
```
R1-Zero → Cold-start SFT (few examples) → GRPO → R1
```
Results: Even stronger reasoning, more stable

**Why two stages?**
- Pure RL: Discovers general reasoning patterns
- SFT: Teaches specific output formats and edge cases
- Combined: Best of both worlds

**R1-Zero proves**: You don't NEED SFT (but it helps)

**Practical lesson**: Start with pure RL, add SFT only if needed

---

## 💭 Karpathy Take

**What's mind-blowing**:
- Self-verification just... appears? (never explicitly trained on it)
- One example + RL ≈ thousands of supervised examples (1-shot RLVR)
- Emergent behaviors are consistent across models (not flukes)
- Pre-training contains so much latent knowledge (RL just unlocks it)

**What's concerning**:
- Only works for verifiable domains (math, code)
- Requires strong base model (won't work on GPT-2)
- Hyperparameter sensitivity (easy to get garbage if wrong)
- Mode collapse risk (model repeats same wrong answer)

**Real talk**:
This is one of those "wait, that shouldn't work" results that actually works. Traditional ML wisdom: "RL needs supervision to bootstrap." Pure RL: "lol hold my beer ¯\_(ツ)_/¯"

The 1-shot result (36% → 73.6%) is particularly wild. If I didn't see the paper, I'd call BS. But it makes sense: that one example teaches "how to structure your reasoning", then RL optimizes "what reasoning to use".

**Emergent self-verification is the real story**: Nobody taught the model to say "let me verify". It discovered that verified answers get higher rewards, so it started verifying. This suggests our models understand reasoning way better than we think - they just need the right incentive structure.

**Missing piece**: Scaling laws for pure RL
- How does improvement scale with compute/data?
- At what model size does pure RL become viable?
- Can we predict when emergent behaviors appear?

**Would I use this?**
- New math reasoning model: Yes (proven approach)
- Domain where supervision is expensive: Yes (worth trying)
- General reasoning tasks: Maybe (if I can define verification)
- Domains without verification: No (need SFT)
- Rapid prototyping: No (SFT is faster for initial version)

**Connection to R1**:
R1-Zero is the purest demonstration of this technique. DeepSeek's two-stage approach (pure RL → SFT → RL) is smart: Get emergent behaviors from pure RL, then polish with SFT. Best of both worlds.

**Future**: This opens the door to training reasoning models in domains where we can't get supervised data (novel theorem proving, mathematical conjectures, etc.). If we can define verification, we can train with pure RL.

---

## 🔗 Cross-References

**Directly Related**:
- **04-deepseek-r1-paper**: R1-Zero uses pure RL (this is the foundation)
- **07-grpo-theory**: GRPO is the RL algorithm used
- **20-illustrated-grpo**: Visual explanation of GRPO

**Conceptually Related**:
- **82-mtp-planning-paper**: MTP for multi-step reasoning (complementary)
- **13-multi-token-prediction**: Different approach to improving reasoning

**RL Training**:
- Pure RL (this doc) → Emergent reasoning
- GRPO (doc 07) → The algorithm
- R1 implementation (doc 04) → Production deployment

---

## 📚 Key Insights

**Theoretical**:
- Pre-training stores latent reasoning → RL activates it
- Self-consistency enables self-supervision
- Emergent behaviors from simple reward signal

**Practical**:
- One good example + RL > 1000 random examples
- Math is ideal domain (verifiable rewards)
- Requires strong base model (>1B params)

**Surprising**:
- Self-verification emerges without training
- Reflection emerges without training
- Chain-of-thought emerges without training

---

## 📚 Further Reading

- RUG Thesis: [Pure RL for Math Reasoning](https://fse.studenttheses.ub.rug.nl/36470/)
- RLVR Paper: [One Example is Enough](https://arxiv.org/abs/2504.20571)
- DeepSeek-R1-Zero: Technical report section on pure RL
- Self-consistency paper: [Wang et al., ICLR 2023]

---

**Status**: ✅ Proven technique (multiple papers confirm)
**Bottom Line**: Pure RL works for math reasoning - no supervision needed, emergent behaviors included!
