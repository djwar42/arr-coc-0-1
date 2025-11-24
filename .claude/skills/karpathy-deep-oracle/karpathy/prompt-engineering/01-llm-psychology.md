# LLM Psychology: Understanding Token Simulators

**How LLMs think (and why they need your help)**

## Primary Sources

From [State of GPT](../../source-documents/karpathy/32-State of GPT _ BRK216HFS.md):
- Human vs LLM cognitive architecture
- "Token simulators" concept
- System 1 vs System 2 thinking
- Prompt engineering techniques

---

## The Core Problem: Different Brains

### Your Brain When Writing

Imagine writing this sentence:
> "California's population is 53 times that of Alaska"

**What actually happens in your head**:

1. **Self-awareness**: "I probably don't know these populations off the top of my head" → Aware of what you know/don't know
2. **Tool use**: Look up California population on Wikipedia → 39.2M
3. **More tool use**: Look up Alaska population → 0.74M
4. **More self-awareness**: "Dividing 39.2 by 0.74 is very unlikely to succeed, that's not the kind of thing I can do in my head"
5. **Calculator**: Punch in 39.2 / 0.74 = 53
6. **Reflection**: "Does 53 make sense? That's quite large... but California is the most populous state, so okay"
7. **Writing**: "California has 53x times greater..." → Delete, awkward phrasing
8. **Editing**: Try again, inspect, adjust
9. **Final**: "California's population is 53 times that of Alaska"

**Computational work**: Massive internal monologue, tool use, self-correction, reflection

---

### LLM "Brain" Processing Same Sentence

**From GPT's perspective**: Just a sequence of tokens

```
California | 's | population | is | 53 | times | that | of | Alaska
```

**Per token**:
- Same amount of compute (~80 layers)
- No tool use
- No self-correction
- No reflection
- No awareness of knowledge gaps

**The difference**:
> "These Transformers are just like token simulators so they don't know what they don't know like they just imitate the next token they don't know what they're good at or not good at they just tried their best to imitate the next token" — Karpathy

---

## The Token Simulator Paradigm

### What LLMs Actually Are

**Not**: Reasoning systems, knowledge databases, or intelligent agents

**Actually**: Statistical models that predict the next token based on previous tokens

**Training data impact**:
- "All of that internal dialogue is completely stripped"
- No record of the thinking process
- Just input → output artifacts

### Cognitive Limitations

**LLMs lack**:
- ❌ Self-knowledge ("I don't know" awareness)
- ❌ Inner monologue (separate reasoning stream)
- ❌ Tool use (by default)
- ❌ Reflection and self-correction
- ❌ Sanity checking
- ❌ Variable compute per token
- ❌ Ability to "try again" if stuck

**The rule**:
> "These Transformers in the process as they predict the next token just like you they can get unlucky and they can they could sample and not a very good token and they can go down sort of like a blind alley in terms of reasoning and so unlike you they cannot recover from that they are stuck with every single token they sample" — Karpathy

---

### Cognitive Advantages

**LLMs do have**:
- ✅ **Massive fact-based knowledge** across vast domains (billions of parameters)
- ✅ **Large and perfect working memory** (whatever fits in context window)
- ✅ **Lossless memory access** via self-attention (can instantly recall anything in context)

**But**: Fixed context window size (finite memory)

---

## System 1 vs System 2 Thinking

### Human Systems

**System 1**: Fast, automatic, unconscious
- Pattern matching
- Intuition
- Immediate responses

**System 2**: Slow, deliberate, conscious
- Planning
- Reflection
- Step-by-step reasoning

### LLM Default: Pure System 1

**Token sampling**: Fast, automatic, no planning

**Missing**: System 2 deliberation, unless you prompt for it

**The solution**:
> "I think to a large extent prompting is just making up for this sort of cognitive difference between these two kind of architectures" — Karpathy

---

## Making Up for Cognitive Differences

### Technique 1: Give Them Tokens to Think

**Problem**: "You can't expect the Transformer to make to to do too much reasoning per token"

**Solution**: Spread reasoning across more tokens

**Bad prompt**:
```
Q: What's 127 * 843?
A: [model has 1 token to answer] → Likely wrong
```

**Good prompt**:
```
Q: What's 127 * 843?
A: Let me work this out step by step.
First, 127 * 800 = 101,600
Then, 127 * 40 = 5,080
Then, 127 * 3 = 381
Adding: 101,600 + 5,080 + 381 = 107,061
```

**Why it works**:
> "These Transformers need tokens to think quote unquote I like to say sometimes" — Karpathy

---

### Technique 2: Few-Shot Prompting

**Show, don't just tell**

**Example**:
```
Q: What is 2 + 2?
A: Let me break this down:
   2 + 2 = 4

Q: What is 17 * 3?
A: Let me break this down:
   17 * 3 = 17 + 17 + 17 = 51

Q: [Your actual question]
A:
```

**Why**: "Give a few examples the Transformer will imitate that template and it will just end up working out better"

---

### Technique 3: "Let's Think Step by Step"

**Magic phrase**: Just add this to your prompt

**Effect**: Conditions the model into showing its work
- Less computational work per token
- More tokens for reasoning
- Higher success rate

**Example**:
```
Q: A farmer has 12 chickens and each chicken lays 3 eggs per day. How many eggs in a week?
A: Let's think step by step.
```

**Result**: Model shows intermediate calculations instead of guessing the answer

---

### Technique 4: Self-Consistency (Try Multiple Times)

**Human ability**: Start writing, delete, try again, pick best version

**LLM problem**: Gets "stuck" with every sampled token, can't backtrack

**Solution**: Sample multiple times, pick the best

**Process**:
1. Generate 5 different completions
2. Score them (via another model or heuristic)
3. Keep the best or do majority vote

**Example**: Math problems
- Sample 10 solutions
- Take majority answer
- Higher accuracy than single sample

---

### Technique 5: Self-Verification

**Surprising fact**: LLMs know when they've screwed up!

**Example**:
```
Prompt: Write a poem that does NOT rhyme
Model: [Writes a poem that rhymes]

Follow-up: Did you meet the assignment?
Model: No, I didn't actually meet the assignment.
       The poem I wrote rhymes. Let me try again...
```

**Why it happens**:
> "Especially for the bigger models like gpt4 you can just ask it did you meet the assignment and actually gpt4 knows very well that it did not meet the assignment it just kind of got unlucky in its sampling" — Karpathy

**Key insight**: Without prompting, it doesn't know to check!

---

## Advanced: Tree of Thoughts

**Concept**: Maintain multiple reasoning paths, score and prune

**Like AlphaGo for text**:
- AlphaGo had a policy for next stone
- But also did Monte Carlo tree search
- Evaluated multiple futures, kept best paths

**Tree of Thoughts**:
1. Generate multiple completions at each step
2. Score them along the way
3. Keep promising branches
4. Prune dead ends
5. Explore best paths deeper

**Implementation**: Not just a prompt! Python glue code + multiple API calls

**Example pseudocode**:
```python
# Generate 3 possible next steps
branches = [llm(prompt + step) for step in range(3)]

# Score each branch
scores = [evaluate(branch) for branch in branches]

# Keep top 2
best_branches = top_k(branches, scores, k=2)

# Expand further
for branch in best_branches:
    # Repeat: generate, score, prune
```

**Reference**: Paper from "just last week" (in 2023) — "This space is pretty quickly evolving"

---

## Chain of Thought Techniques

### React: Thought-Action-Observation

**Structure**: Interleave thinking and acting

```
Thought: I need to find California's population
Action: search("California population")
Observation: 39.2 million

Thought: Now I need Alaska's population
Action: search("Alaska population")
Observation: 0.74 million

Thought: Now I can calculate the ratio
Action: calculate(39.2 / 0.74)
Observation: 52.97

Thought: I'll round to 53 for clarity
Answer: California's population is 53 times that of Alaska
```

**Why**: Makes LLM's "thinking" explicit and checkable

---

### AutoGPT: Recursive Task Decomposition

**Concept**: LLM maintains task list, recursively breaks down tasks

**Status** (as of 2023):
> "I think got a lot of hype recently and I think but I think I still find it kind of inspirationally interesting... I don't think this currently works very well and I would not advise people to use it in Practical applications I just think it's something to generally take inspiration from" — Karpathy

**Why interesting**: Shows direction LLM systems are heading

**Why doesn't work yet**: Models not capable enough for reliable autonomous operation

---

## The Imitation Problem

### LLMs Don't Want to Succeed, They Want to Imitate

**Training data spectrum**:
- Student solutions (wrong)
- Average solutions (okay)
- Expert solutions (excellent)

**Problem**: LLM trained on all of it
- Can't inherently distinguish quality
- Will imitate whatever matches the prompt

**Solution**: Ask for quality!

**Bad prompt**:
```
Solve this physics problem: [problem]
```

**Good prompt**:
```
You are a leading expert in physics with years of experience.
Let's work this out in a step-by-step way to be sure we have the right answer.

Solve this physics problem: [problem]
```

**Why it works**: Conditions model on high-quality region of training data

---

### Finding the Right IQ

**Karpathy's advice**:
- ✅ "Pretend you have IQ 120" → Good, in-distribution
- ✅ "You are a leading expert" → Good conditioning
- ❌ "Pretend you have IQ 400" → Out-of-distribution, might trigger sci-fi roleplay

**The U-curve**: Too much can backfire
- Too little expertise: Bad solutions
- Right amount: Great solutions
- Too much claimed expertise: Model thinks it's a fictional character

---

## Tool Use: Offload What LLMs Suck At

### The Principle

**You know what you're bad at** → Use calculator for 7-digit multiplication

**LLMs don't know** → Will try to do mental arithmetic and fail

**Solution**: Tell them explicitly!

**Example prompt**:
```
You are not very good at mental arithmetic with large numbers.
Whenever you need to multiply or divide numbers larger than 100,
use the calculator tool like this: [CALC: expression]

Now solve this problem: What is 8,472 * 3,194?
```

**Tools you can give LLMs**:
- Calculator (arithmetic)
- Code interpreter (complex logic)
- Search (up-to-date info)
- Database queries (structured data)
- API calls (external services)

---

## Key Principles

### 1. LLMs Are Token Simulators
- Not reasoning engines
- Not knowledge bases
- Statistical next-token predictors

### 2. They Need Tokens to Think
- Can't do complex reasoning in one token
- Spread it across multiple tokens
- "Let's think step by step" is magic

### 3. They Don't Know What They Don't Know
- No self-awareness by default
- Tell them what they're bad at
- Give them tools for those tasks

### 4. They Can Get Unlucky
- Sample multiple times
- Pick the best
- Or ask them to verify their own work

### 5. They Want to Imitate, Not Excel
- Ask for quality explicitly
- "You are an expert"
- "Let's be sure we have the right answer"

### 6. They Have No System 2
- Unless you prompt for it
- Tree of thoughts, chain of thought
- Python glue code for complex reasoning

---

## Practical Implications

**For prompt engineering**:
- Show your work (few-shot examples)
- Ask for step-by-step reasoning
- Request verification
- Give tools for weak areas
- Condition on high quality

**For system design**:
- Don't expect one-shot answers to complex questions
- Build reasoning loops
- Implement verification steps
- Provide tool access
- Allow multiple attempts

**For realistic expectations**:
- LLMs are powerful but limited
- They need scaffolding for complex tasks
- Prompt engineering = cognitive prosthetics
- Good prompts compensate for architectural differences

---

## Related Content

- [Few-Shot Prompting](02-few-shot-techniques.md) - Examples and patterns
- [Tool Use](03-tool-integration.md) - Calculators, code, search
- [Chain of Thought](04-chain-of-thought.md) - Reasoning patterns
- [Realistic Expectations](../llm-applications/01-limitations.md) - What LLMs can't do
