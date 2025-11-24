# LLM Applications - Use Cases, Limitations, and Best Practices

**Practical Guidance on Deploying Language Models Responsibly**

---

## Karpathy's Core Advice: Low-Stakes + Human Oversight

> "My recommendation right now is use llms in low stakes applications combine them always with human oversight use them as a source of inspiration and suggestions and think co-pilots instead of completely autonomous agents"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

**Guiding principles:**
- 🟢 **Low-stakes:** Not life-or-death decisions
- 🟢 **Human-in-the-loop:** Always have oversight
- 🟢 **Co-pilot mode:** Assist, don't replace
- 🔴 **Not autonomous:** Too unreliable currently

---

## Appropriate Use Cases

### ✅ Code Assistance (GitHub Copilot)
- **What:** Autocomplete code, suggest implementations
- **Why it works:** Low stakes, programmer reviews
- **Failure mode:** Buggy code → caught in review/testing

### ✅ Writing Assistance (Draft emails, docs)
- **What:** Generate first drafts, improve phrasing
- **Why it works:** Human edits and approves
- **Failure mode:** Poor writing → human catches it

### ✅ Brainstorming & Ideation
- **What:** Generate ideas, explore possibilities
- **Why it works:** Quantity over quality, human filters
- **Failure mode:** Bad ideas → ignored

### ✅ Summarization (Meeting notes, articles)
- **What:** Condense long texts
- **Why it works:** Human can verify against source
- **Failure mode:** Missing details → human adds them

### ✅ Question Answering (with verification)
- **What:** Answer questions from docs/FAQs
- **Why it works:** Human can fact-check
- **Failure mode:** Wrong answer → human corrects

### ✅ Creative Writing (Stories, marketing copy)
- **What:** Generate creative content
- **Why it works:** Subjective quality, human edits
- **Failure mode:** Bland content → rewrite

### ✅ Data Extraction (Structured info from text)
- **What:** Parse resumes, extract entities
- **Why it works:** Human reviews extractions
- **Failure mode:** Missed fields → caught in review

---

## Inappropriate Use Cases

### ❌ Medical Diagnosis
- **Why not:** Life-or-death stakes
- **Risk:** Hallucinations could harm patients
- **Alternative:** Assist doctors with research, not diagnose

### ❌ Legal Advice
- **Why not:** Legal consequences, accountability
- **Risk:** Fabricated case law (it happens!)
- **Alternative:** Assist lawyers with research, not advise clients

### ❌ Financial Trading (Autonomous)
- **Why not:** Large financial risk
- **Risk:** Unpredictable behavior, losses
- **Alternative:** Analysis tool with human final decision

### ❌ Critical Infrastructure
- **Why not:** Safety-critical systems
- **Risk:** Failure = catastrophic consequences
- **Alternative:** Human operators with AI assistance

### ❌ Autonomous Vehicles (Current models)
- **Why not:** Safety-critical, real-time
- **Risk:** Reasoning failures → accidents
- **Alternative:** Specialized, safety-certified systems

---

## Limitations (The Reality Check)

### 1. Hallucinations

**Problem:** LLMs confidently state false information

**Example:**
```
Q: Who won the 2023 Nobel Prize in Physics?
A: Dr. Jane Smith won for her work on quantum gravity.
[Completely fabricated - Jane Smith doesn't exist]
```

**Why it happens:** Model trained to predict plausible text, not truth

**Mitigation:**
- Fact-check critical info
- Use RAG (ground in real documents)
- Multiple sources verification

### 2. Knowledge Cutoffs

**Problem:** Training data has a date limit

**Example:**
```
GPT-3: Knowledge cutoff September 2021
→ Doesn't know about events after that date
```

**Mitigation:**
- Use RAG to inject current info
- Combine with search APIs
- Update models regularly

### 3. Reasoning Errors

**Problem:** LLMs can make logical mistakes

**Example:**
```
Q: If it takes 5 machines 5 minutes to make 5 widgets,
   how long does it take 100 machines to make 100 widgets?

Wrong answer: 100 minutes (linear scaling assumption)
Correct answer: 5 minutes (machines work in parallel)
```

**Mitigation:**
- Chain of thought prompting
- Multiple samples + verification
- Human review for critical reasoning

### 4. Biases

**Problem:** Training data contains societal biases

**Examples:**
- Gender stereotypes (doctors = male, nurses = female)
- Racial biases in language
- Cultural assumptions

**Mitigation:**
- Diverse training data
- RLHF to reduce harmful outputs
- Human review and filtering
- Fairness metrics

### 5. Context Window Limits

**Problem:** Can only see limited text (2K-128K tokens)

**Example:**
```
GPT-3: 2048 tokens ≈ 1500 words
→ Can't read full books or long reports
```

**Mitigation:**
- Summarization techniques
- Sliding window approaches
- RAG (retrieve relevant chunks)

### 6. Security Vulnerabilities

**Attacks:**
- **Prompt injection:** "Ignore previous instructions..."
- **Jailbreaking:** Circumvent safety guardrails
- **Data poisoning:** Manipulate training data
- **Model extraction:** Steal model via queries

**Mitigation:**
- Input sanitization
- Output filtering
- Rate limiting
- Defense-in-depth

---

## Cost-Performance Tradeoffs

### Model Selection Matrix

| Model | Cost | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **GPT-4** | $$$$ | Slow | Excellent | Critical tasks, complex reasoning |
| **GPT-3.5** | $$ | Fast | Good | General assistance, high volume |
| **Fine-tuned GPT-3.5** | $$ | Fast | Task-specific | Domain-specific applications |
| **Llama 65B** | $ (self-host) | Medium | Good | Privacy-sensitive, high volume |
| **Llama 13B** | $ | Fast | Okay | Simple tasks, resource-constrained |

**Decision factors:**
- Budget (API costs vs self-hosting)
- Latency requirements
- Quality bar
- Privacy/compliance needs
- Query volume

---

## Fine-Tuning vs Prompting

### When to Fine-Tune:

✅ **High volume, specific task**
- Example: Customer support for specific product
- Why: Cheaper per query after upfront cost

✅ **Consistent output format**
- Example: Always generate valid JSON
- Why: Hard to enforce via prompts alone

✅ **Domain-specific knowledge**
- Example: Medical terminology
- Why: General models lack specialized knowledge

✅ **Tone/style requirements**
- Example: Match brand voice
- Why: Hard to consistently prompt

### When to Use Prompting:

✅ **Low volume, varied tasks**
- Why: Fine-tuning overhead not worth it

✅ **Rapid iteration**
- Why: Prompts update instantly, fine-tuning takes hours/days

✅ **General knowledge tasks**
- Why: Base models already capable

✅ **Exploration phase**
- Why: Don't know requirements yet

---

## Deployment Patterns

### 1. API-based (OpenAI, Anthropic)

**Pros:**
- No infrastructure management
- Latest models automatically
- Scales effortlessly
- Pay per use

**Cons:**
- Recurring costs
- Data leaves your infrastructure
- Vendor lock-in
- Rate limits

**Best for:** Startups, experiments, low-medium volume

### 2. Self-Hosted Open Models

**Pros:**
- One-time GPU cost
- Full data control
- Customizable
- No per-query fees

**Cons:**
- Infrastructure management
- Model updates manual
- Scaling requires more GPUs
- Upfront costs

**Best for:** High volume, privacy-sensitive, enterprises

### 3. Hybrid

**Example:**
- GPT-4 for complex queries (5%)
- Self-hosted Llama for simple queries (95%)

**Best for:** Optimizing cost vs quality

---

## Monitoring & Evaluation

### Metrics to Track:

**Quality:**
- User ratings (thumbs up/down)
- Task success rate
- Human eval scores
- A/B test comparisons

**Performance:**
- Latency (P50, P95, P99)
- Throughput (queries/sec)
- Error rate
- Timeout rate

**Cost:**
- Tokens per query
- Cost per successful query
- Total monthly spend

**Safety:**
- Harmful content rate
- PII exposure
- Jailbreak attempts
- Hallucination frequency

### Continuous Improvement:

```
1. Collect user feedback
2. Log failures and edge cases
3. Improve prompts or add examples
4. Re-evaluate on test set
5. Deploy improved version
6. Repeat
```

---

## Practical Architecture

**Modern LLM application stack:**

```
User Interface
    ↓
API Gateway (rate limiting, auth)
    ↓
Prompt Router (task classification)
    ↓
    ├─→ Simple task → Small model (fast, cheap)
    ├─→ Complex task → Large model (slow, expensive)
    └─→ Specialized task → Fine-tuned model
    ↓
RAG System (optional)
    ├─ Vector Database (embeddings)
    └─ Document Store (original content)
    ↓
LLM (OpenAI API / Self-hosted)
    ↓
Output Filter (safety, validation)
    ↓
Logging & Monitoring
    ↓
User Response
```

---

## Responsible AI Checklist

Before deploying LLM application:

✅ **Identify risk level** (low/medium/high stakes)
✅ **Add human oversight** (especially for high stakes)
✅ **Test for biases** (run fairness evaluations)
✅ **Implement safety filters** (harmful content detection)
✅ **Set up monitoring** (track failures, user reports)
✅ **Plan for failures** (fallback behavior, error messages)
✅ **Comply with regulations** (GDPR, data privacy)
✅ **Document limitations** (set user expectations)
✅ **Enable user feedback** (thumbs up/down, reporting)
✅ **Regular audits** (review outputs, update filters)

---

## Karpathy's Deployment Philosophy

**Current state (2023):**
- Models are powerful but unreliable
- Work best as assistants, not autonomy
- Need human oversight for important tasks
- Cost-performance tradeoffs matter

**Near future:**
- Better reasoning (GPT-5, GPT-6, ...)
- Lower costs (competition, efficiency)
- More specialized models (medicine, law, etc.)
- Better safety mechanisms

**For now:**
> "Think co-pilots instead of completely autonomous agents that are just like performing a task somewhere it's just not clear that the models are there right now"
>
> — *Source: [source-documents/32-State of GPT...](../../source-documents/karpathy/32-State%20of%20GPT%20_%20BRK216HFS.md)*

---

## Real-World Examples

### GitHub Copilot (✅ Good use case)
- **Task:** Code completion
- **Stakes:** Low (programmer reviews)
- **Human oversight:** Always (dev approves suggestions)
- **Result:** Productivity boost, not replacement

### ChatGPT for Customer Support (✅ With caveats)
- **Task:** Answer common questions
- **Stakes:** Medium (customer satisfaction)
- **Human oversight:** Agent reviews before sending
- **Result:** Faster responses, quality maintained

### LLM for Hiring Decisions (❌ Bad use case)
- **Task:** Screen resumes
- **Stakes:** High (impacts livelihoods)
- **Risk:** Biases, errors, legal issues
- **Better:** Assist recruiters, don't replace

---

## Next Steps

**For builders:**
- Start small (low-stakes use case)
- Add human oversight
- Monitor carefully
- Iterate based on feedback

**For users:**
- Understand limitations
- Verify critical information
- Provide feedback (helps everyone)
- Use as co-pilot, not oracle

**Resources:**
- [../prompt-engineering/](../prompt-engineering/) - Get better outputs
- [../training-llms/](../training-llms/) - Understand how models work

**Primary sources:**
- `source-documents/32-State of GPT...` - Complete LLM overview
- Industry talks on practical deployment

---

## The Big Picture

**LLMs are powerful tools, not magic:**
- Incredible pattern matchers
- Great at generating plausible text
- Terrible at being reliable 100% of the time

**Use them wisely:**
- Low-stakes applications
- Human oversight always
- Co-pilot mode, not autopilot
- Continuous monitoring

**The future is bright, but we're not there yet.**
