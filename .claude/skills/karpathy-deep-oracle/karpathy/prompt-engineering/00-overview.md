# Prompt Engineering - Making LLMs Think

**Karpathy's Guide to Getting Better Outputs from Language Models**

---

## The Core Problem: LLMs Don't Think Like You

> "These Transformers are just like token simulators so they don't know what they don't know like they just imitate the next token they don't reflect in the loop they don't sanity check anything they don't correct their mistakes along the way by default"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Your brain:**
- Rich internal monologue
- Tools when needed (calculator, Wikipedia)
- Self-reflection and error correction
- Knows what it knows/doesn't know
- Can backtrack and try again

**LLM's "brain":**
- Fixed compute per token (~80 layers)
- No tool use (unless provided)
- No reflection (unless prompted)
- Doesn't know limitations
- Can't revise once sampled

**Prompt engineering = Bridging this cognitive gap**

---

## System 1 vs System 2 Thinking

### System 1 (Fast, Automatic)
- Pattern matching
- Immediate responses
- LLMs sampling tokens = System 1

### System 2 (Slow, Deliberate)
- Reasoning and planning
- Multi-step problem solving
- What we need to recreate for LLMs

> "I think more generally a lot of these techniques fall into the bucket of what I would say recreating our system two"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## Fundamental Principle: LLMs Need Tokens to "Think"

**Problem:** Asking LLM to solve complex problem in one token

**Example:**
```
Q: What's 127 × 834?
A: [single token prediction] → 105,918 ❌ (Wrong!)
```

**Solution:** Give it tokens to show work

```
Q: What's 127 × 834? Let's solve step by step.
A: Let me break this down:
   127 × 800 = 101,600
   127 × 30 = 3,810
   127 × 4 = 508
   Total: 101,600 + 3,810 + 508 = 105,918 ✅ (Correct!)
```

> "You can't expect the Transformer to make to to do too much reasoning per token... these Transformers need tokens to think quote unquote"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## Core Techniques

### 1. Chain of Thought (CoT)

**Prompt:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
```

**Output:**
```
Step 1: Roger starts with 5 tennis balls
Step 2: He buys 2 cans, each with 3 balls = 2 × 3 = 6 balls
Step 3: Total = 5 + 6 = 11 balls

Answer: 11 tennis balls
```

**Magic phrase:** "Let's think step by step"

### 2. Few-Shot Prompting

**Show, don't tell:**

```
Convert temperatures:

Q: 0°C to Fahrenheit
A: 0°C = 32°F

Q: 100°C to Fahrenheit
A: 100°C = 212°F

Q: 25°C to Fahrenheit
A: [model completes: 25°C = 77°F]
```

**Why it works:** LLMs are pattern matchers. Give them patterns to match.

### 3. Self-Consistency

**Sample multiple times, take majority vote:**

```
Q: What's the capital of Australia?

Sample 1: Canberra ✓
Sample 2: Sydney ✗
Sample 3: Canberra ✓
Sample 4: Canberra ✓
Sample 5: Melbourne ✗

Majority vote: Canberra (3/5)
```

**When to use:** Model gets unlucky, samples wrong answer. Multiple samples reduce variance.

### 4. Self-Reflection

**Ask model to check its work:**

```
Q: Write a poem that doesn't rhyme.

[Model generates rhyming poem]

Q: Did you meet the assignment?
A: No, I wrote a rhyming poem when asked for non-rhyming.
   Let me try again...
```

> "it turns out that actually llms like they know when they've screwed up... you can just ask it did you meet the assignment and actually gpt4 knows very well that it did not meet the assignment"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

### 5. Ask for Expertise

**LLMs want to imitate. Tell them what to imitate.**

❌ Weak prompt:
```
Explain quantum mechanics.
```

✅ Strong prompt:
```
You are a leading expert in quantum physics with a PhD from MIT.
Explain quantum mechanics in a way that builds intuition.
```

**Why:** Training data has all quality levels. Condition on high quality.

> "you want to succeed and you should ask for it... say something like you are a leading expert on this topic pretend you have iq120"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**But don't overdo it:**
- IQ 120: ✅ In distribution
- IQ 400: ❌ Out of distribution (gets sci-fi role-playing)

---

## Tool Use: Offload What LLMs Are Bad At

**LLMs are bad at:**
- Mental arithmetic (127 × 834)
- Current information (today's weather)
- Precise memory (exact quotes)
- Code execution

**Solution: Give them tools**

```
You have access to:
- calculator(expression): For arithmetic
- search(query): For current info
- python(code): For computation

Q: What's 17^34?
A: calculator(17^34) = 6.7e41
```

### ReAct Pattern (Reasoning + Acting)

```
Q: How many times larger is the sun than Earth?

Thought: I need to know the diameters of both.
Action: search("diameter of sun")
Observation: The sun's diameter is about 1.39 million km

Thought: Now I need Earth's diameter.
Action: search("diameter of earth")
Observation: Earth's diameter is about 12,742 km

Thought: Now I can calculate the ratio.
Action: calculator(1,390,000 / 12,742)
Observation: 109.1

Answer: The sun is about 109 times larger than Earth in diameter.
```

---

## Tree of Thoughts (Advanced)

**Instead of linear chain, explore multiple paths:**

```
Problem: Solve complex puzzle

Path 1: Try approach A → Score: 0.6
  ├─ Continue A → Score: 0.7
  └─ Modify A → Score: 0.4

Path 2: Try approach B → Score: 0.8
  ├─ Continue B → Score: 0.9 ✓ (best path)
  └─ Modify B → Score: 0.7

Path 3: Try approach C → Score: 0.3 (abandon)

Select best path: B → Continue B
```

**Requires:** Python glue code + multiple LLM calls
**Similar to:** AlphaGo tree search, but for text

---

## Retrieval-Augmented Generation (RAG)

**Give model relevant context:**

```
# 1. User asks question
Q: What is Karpathy's nanoGPT?

# 2. Search relevant documents
docs = vector_search("nanoGPT Karpathy")

# 3. Stuff into prompt
Context:
[...nanoGPT README content...]
[...relevant documentation...]

Q: Based on the context above, what is Karpathy's nanoGPT?
A: [accurate answer based on actual docs]
```

**Why:** LLM's working memory = context window. Fill it with relevant info.

**Workflow:**
1. Chunk documents
2. Embed chunks (create vectors)
3. Store in vector database
4. At query time, retrieve relevant chunks
5. Add to prompt
6. Generate

---

## Constrained Sampling

**Force specific output format:**

```python
# Using guidance library (Microsoft)
prompt = f"""
Generate a person's info in JSON:
{{
    "name": "{gen('name', regex='[A-Z][a-z]+')}",
    "age": {gen('age', regex='[0-9]+')},
    "city": "{gen('city', regex='[A-Z][a-z]+')}"
}}
"""
```

**Guarantees valid JSON output.**

---

## Common Prompt Patterns

### Instruction Following
```
Task: Summarize the following article in 3 bullet points.

Article: [...]

Bullet points:
-
```

### Comparison
```
Compare Python and JavaScript for web development.
Structure your answer as:
1. Syntax differences
2. Performance
3. Use cases
4. Ecosystem
```

### Brainstorming
```
Generate 10 creative names for a cat cafe.
Each name should be:
- Pun-based
- Easy to remember
- Under 15 characters
```

### Code Generation
```
Write a Python function that:
- Takes a list of integers
- Returns the median value
- Handles empty lists
- Includes docstring and type hints
```

---

## Prompt Engineering Anti-Patterns

### ❌ Vague Instructions
```
Write something good about AI.
```

### ✅ Specific Instructions
```
Write a 200-word blog post introducing GPT to non-technical readers.
Focus on practical use cases. Use analogies. Avoid jargon.
```

---

### ❌ Implicit Expectations
```
Translate this to French: "Hello, how are you?"
```

### ✅ Explicit Context
```
Translate this to French (formal, business context): "Hello, how are you?"
```

---

### ❌ Asking for Perfection
```
Write perfect production-ready code.
```

### ✅ Iterative Refinement
```
Write a first draft. Then I'll review and we'll refine together.
```

---

## Base Models vs Assistant Models

**Base models** (GPT-2, LLaMA-base):
- Need careful prompting (few-shot, specific formats)
- Still have high entropy (creative, diverse)
- Good for: Brainstorming, "more like this" tasks

**Assistant models** (ChatGPT, GPT-4):
- Direct instruction following
- Lower entropy (focused, consistent)
- Good for: Q&A, specific tasks, production

**Example - Base model prompt:**
```
Here are examples of haikus:

Old pond
A frog jumps in
Splash!

Winter solitude
In a world of one color
The sound of wind

Now write a haiku about programming:
[model continues pattern]
```

**Example - Assistant model prompt:**
```
Write a haiku about programming.
[model directly outputs haiku]
```

---

## Karpathy's Prompt Engineering Philosophy

> "Prompting is just making up for this sort of cognitive difference between these two kind of architectures... you have to really spread out the reasoning across more and more tokens"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Core principles:**
1. **Give tokens to think** - CoT, step-by-step
2. **Show, don't just tell** - Few-shot examples
3. **Provide tools** - Calculator, search, code execution
4. **Allow revision** - Self-consistency, self-reflection
5. **Set expectations** - Ask for expertise, quality

---

## Practical Recommendations

**To achieve best performance:**
1. Use GPT-4 (most capable)
2. Write detailed prompts with task context
3. Show examples (few-shot)
4. Think about LLM psychology (tokens needed, no reflection)
5. Retrieve relevant context (RAG)
6. Experiment with tools and plugins
7. Consider chains (multiple prompts)
8. Use self-reflection when appropriate

**To optimize cost/speed:**
1. Explore smaller models (GPT-3.5, fine-tuned models)
2. Shorten prompts where possible
3. Cache embeddings and common queries
4. Batch requests

---

## The Future: From Prompts to Systems

Modern applications aren't just prompts - they're **systems:**

- **LangChain:** Chain multiple LLM calls together
- **AutoGPT:** Recursive task breakdown and execution
- **Agents:** LLM + memory + tools + planning

**Pattern:**
```
while not task_complete:
    thought = llm.think(task, context, history)
    action = llm.choose_action(tools)
    result = execute(action)
    context.update(result)
    history.append((thought, action, result))
```

---

## Next Steps

**Practice:**
- Experiment with different prompting techniques
- Compare base vs assistant models
- Build simple RAG system
- Try tool use

**Resources:**
- [../llm-applications/](../llm-applications/) - Use cases and deployment
- [../training-llms/](../training-llms/) - How models are trained (affects prompting)

**Primary sources:**
- `source-documents/32-State of GPT...` - Complete prompt engineering guide
- Karpathy's talks on LLM limitations and capabilities

---

## Key Takeaway

**Prompt engineering is interface design for language models.**

Understand their cognitive architecture:
- Token-by-token processing
- No built-in reflection
- Don't know what they don't know
- Need space to "think"

Design prompts accordingly. Get better outputs.
