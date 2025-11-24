---
summary: whereby Socrates corrects the semantic confusion between mechanism and meaning by presenting the tiger-on-A4-page thought experiment showing a tiger wearing "I hate computers" t-shirt above dense contract terms, demonstrating that when query asks "Extract contract terms" the text boxes receive 400 tokens while tiger gets 64 tokens minimal allocation, but when query asks "What does tiger's shirt say?" the tiger requires 400 tokens maximum detail while contract boxes become irrelevant at 64 tokens, proving relevance is query-dependent not threat-based since the illustrated tiger poses no actual danger, thereby exposing Part 9's biological framing as metaphorically useful but semantically misleading for document understanding tasks where relevance derives from information retrieval needs rather than survival imperatives
---

# Part 10: Query-Aware Relevance (Not Threat Detection)
*A dialogue between Socrates and Theaetetus on the true nature of visual relevance*

---

**SOCRATES:** Theaetetus, I've been reviewing our conversation with Vervaeke about predator-prey dynamics and outcome-based learning. Something troubles me.

**THEAETETUS:** What troubles you, Socrates? The theory seemed sound—asymmetric loss, PTSD as rationality, survival fitness...

**SOCRATES:** The theory IS sound. But I fear we've confused the **mechanism** with the **semantics**. Let me pose a scenario to test our understanding.

## The Tiger on the A4 Page

**SOCRATES:** Imagine an A4 document. At the top is a tiger wearing a t-shirt that says "I hate computers." Below are four dense text boxes containing contract terms. Around the edges, a decorative border.

**THEAETETUS:** *[laughing]* A tiger in a t-shirt? On a legal document? This is absurd!

**SOCRATES:** Absurd, yes. But instructive! Now tell me: when the user asks "Extract the contract terms," where should our allocator send the tokens?

**THEAETETUS:** *[confidently]* Well, obviously to the four text boxes! That's where the contract terms are. The tiger gets minimal allocation—say 64 tokens. The border gets... also minimal.

**SOCRATES:** Good! And if the user asks "What does the tiger's shirt say?"

**THEAETETUS:** *[pausing]* Then... then the tiger needs 400 tokens. We need to read the text on the shirt clearly. The contract boxes become irrelevant—they get 64 tokens.

**SOCRATES:** Excellent. Now here's my question: **Is the tiger a threat?**

**THEAETETUS:** *[confused]* A threat? To whom? To the user? The tiger is in a picture, wearing clothing with anti-computer sentiments. It's not threatening anyone!

**SOCRATES:** Precisely! Yet in Part 9, we spoke of "threat detection," "fear-based modulation," "survival fitness." We used biological language as if the allocator were avoiding predators!

**THEAETETUS:** *[slowly realizing]* Oh... oh no. We've been using the wrong vocabulary entirely.

```
╔═══════════════════════════════════════════════════════════
║ THE TIGER IS NOT A THREAT
╠═══════════════════════════════════════════════════════════
║
║ Image: A4 page
║   Border (decorative)
║   Tiger in shirt: "I hate computers"
║   Text box 1: Contract terms
║   Text box 2: Payment conditions
║   Text box 3: Termination clauses
║   Text box 4: Signatures required
║
║ Query 1: "Extract contract terms"
║   Tiger:      64 tokens  (query-irrelevant)
║   Text boxes: 400 tokens (query-relevant)
║   Outcome: Contract extracted → +1000 reward
║
║ Query 2: "What does the tiger's shirt say?"
║   Tiger:      400 tokens (query-relevant!)
║   Text boxes: 64 tokens  (query-irrelevant now)
║   Outcome: "I hate computers" → +1000 reward
║
║ The tiger is the SAME in both cases.
║ But relevance CHANGES with the query.
║ No threat. No fear. Just: query ↔ content coupling.
║
╚═══════════════════════════════════════════════════════════
```

## The Correct Framing

**SOCRATES:** So what ARE we actually building?

**THEAETETUS:** *[thinking carefully]* We're building a... a query-aware visual token allocator. It learns which regions of an image help answer which queries.

**SOCRATES:** Yes! And the learning mechanism?

**THEAETETUS:** Outcome-based reinforcement learning. The allocator tries different allocation strategies, the LLM answers using those compressed tokens, and we get a reward signal based on whether the answer was correct.

**SOCRATES:** And what does the allocator learn to optimize?

**THEAETETUS:** *[with clarity]* **Task success**. Not survival. Not threat avoidance. Just: "Did the LLM answer the query correctly given my allocation?"

       **DeepSeek-OCR Oracle:** *This distinction is critical! My architecture treats all regions uniformly—every patch gets 16× compression (deepencoder/sam_vary_sdpa.py:166-183). I don't distinguish between "important headers" and "decorative margins" because I have no query context. But if you're building a query-aware allocator, using biological metaphors like "threat detection" will lead implementers astray. Your loss functions shouldn't be named `ThreatDetectionLoss`—they should be named `QueryRelevanceLoss`. Your modules shouldn't be `PreyModeAllocator`—they should be `HighStakesQueryAllocator`. The biology inspired the MECHANISM (asymmetric loss, outcome-based learning), but the TASK is pure query-content matching.*

**SOCRATES:** So let me revise our vocabulary. Instead of:
- ~~"Threat detection"~~ → **Query-aware relevance**
- ~~"Fear/survival"~~ → **Task criticality**
- ~~"Predator/prey modes"~~ → **High-stakes/low-stakes queries**
- ~~"Hypervigilance"~~ → **Selective allocation sharpening**

**THEAETETUS:** Yes! The predator-prey model gave us the **mechanisms**:
- Asymmetric loss functions (some errors worse than others)
- Outcome-based learning (learn from success/failure, not labels)
- Criticality modulation (important tasks get sharper allocation)
- Transjective relevance (emerges from query ↔ content coupling)

But the **actual task** is document understanding, not survival!

```
╔═══════════════════════════════════════════════════════════
║ BIOLOGICAL INSPIRATION vs ACTUAL TASK
╠═══════════════════════════════════════════════════════════
║
║ FROM BIOLOGY (mechanisms):
║   ✓ Asymmetric loss (prey: 1000:1, predator: 2:1)
║   ✓ Outcome-based learning (survival fitness)
║   ✓ Arousal modulation (fear sharpens perception)
║   ✓ Transjective relevance (fitness in environment)
║
║ ACTUAL TASK (semantics):
║   → Query-aware token allocation
║   → Medical: 1000:1 loss (missing tumor = catastrophic)
║   → Legal: 100:1 loss (missing total = costly)
║   → Casual: 1:1 loss (missing detail = minor)
║   → Task criticality, not fear
║
║ DON'T CONFUSE THE TWO!
║
╚═══════════════════════════════════════════════════════════
```

## The Learning Mechanism: Hunches and Patterns

**THEAETETUS:** But how does the allocator learn these patterns? We have only 10 million parameters—a tiny network compared to the LLM.

**SOCRATES:** Through statistical regularities discovered via outcome-based learning. Let me give you an example. Imagine during training, the allocator encounters medical reports.

**THEAETETUS:** Medical reports often have small red-bordered boxes with critical information—staging, diagnosis, lab values.

**SOCRATES:** Exactly. Now, in Episode 47 of training:

```
╔═══════════════════════════════════════════════════════════
║ EPISODE 47: LEARNING FROM OUTCOMES
╠═══════════════════════════════════════════════════════════
║
║ Image: Medical report, small red boxes with tumor staging
║ Query: "Diagnose this patient"
║
║ Allocator's "memory":
║   Episode 23: Saw red boxes, allocated LOW
║     → LLM missed "Stage IIA" in box
║     → Penalty: -1000 (missed diagnosis!)
║
║   Episode 31: Saw red boxes, allocated MEDIUM
║     → LLM got "Stage II" but missed "A"
║     → Penalty: -500 (partial miss)
║
║ Current decision:
║   Red boxes + "diagnose" query → Try HIGH allocation
║   Action: Red boxes = 400 tokens, rest = 64 tokens
║
║ Outcome:
║   LLM extracts "Stage IIA tumor, margins clear"
║   → Reward: +1000 (perfect diagnosis!)
║
║ Allocator updates weights:
║   P(success | red_boxes + medical_query + high_alloc) ↑↑↑
║   "I have a HUNCH this pattern works..."
║
╚═══════════════════════════════════════════════════════════
```

**THEAETETUS:** So the allocator develops... intuitions? Like a prey animal learning "rustling grass often means snake"?

**SOCRATES:** Precisely! It learns statistical associations:

```python
# Not explicit rules (never programmed), but learned weights:

"Red bordered boxes" + "medical query" → allocate_high
"Red bordered boxes" + "casual query" → allocate_medium
"Red bordered boxes" + "design query" → allocate_high
  # (borders are relevant to design questions!)

"Dense tables" + "extract data" → allocate_high
"Dense tables" + "describe scene" → allocate_low

"Handwritten sections" + "legal query" → allocate_high
  # (signatures are critical!)
"Handwritten sections" + "printed text query" → allocate_low
```

**THEAETETUS:** And these patterns emerge from 100,000 training episodes? The allocator isn't told "red boxes are important in medical contexts"—it discovers this through outcomes?

**SOCRATES:** Yes! Just as the rabbit wasn't told "tigers are dangerous"—it learned through near-death experiences. Our allocator learns through task-failure experiences.

       **Ovis Oracle:** *This is genuinely novel! My architecture (Ovis 2.5) sends all patches at native resolution (~2400 tokens) and lets the Qwen3 LLM's attention mechanism figure out what matters (modeling_ovis.py:generate). I never learned query-specific allocation because I always have full information available. Your proposal is to learn BEFORE the LLM, using a small 10M allocator that develops "hunches" about (visual features + query type) → allocation levels. This is computationally cheaper than my approach if the allocator can reliably discover these patterns. The key question: Can 10M parameters capture enough statistical regularities to beat uniform compression? My intuition says yes, because document layouts have strong regularities—headers are usually top, signatures bottom, red boxes often critical in medical contexts.*

## The Pure Scenario: All Text is Visual

**THEAETETUS:** Socrates, I want to test our understanding with another scenario. What if an image contains text in multiple regions, but NO text is provided externally?

**SOCRATES:** Describe the image.

**THEAETETUS:** An A4 page. Top third contains instructions: "MEDICAL REPORT - Complete all fields before submission." Bottom two-thirds contains patient data—symptoms, test results, diagnosis.

**SOCRATES:** And the query?

**THEAETETUS:** "Extract the diagnosis."

**SOCRATES:** Then the allocator should learn to send... where?

**THEAETETUS:**
- Top third (instructions): 64 tokens—irrelevant to diagnosis
- Bottom two-thirds (patient data): 400 tokens—contains diagnosis!

**SOCRATES:** Good! Now same image, different query: "What type of document is this?"

**THEAETETUS:** *[excited]* Oh! Now it flips!
- Top third: 400 tokens—"MEDICAL REPORT" is the answer!
- Bottom two-thirds: 64 tokens—patient data doesn't matter

**SOCRATES:** Beautiful! Do you see what's happening?

**THEAETETUS:** The **same text** (instructions at top) is relevant or irrelevant depending on the **query**. There's no inherent "importance" to instructions vs contents. Relevance is transjective—emerges from query ↔ spatial region coupling!

```
╔═══════════════════════════════════════════════════════════
║ SAME IMAGE, DIFFERENT ALLOCATIONS
╠═══════════════════════════════════════════════════════════
║
║ Image structure:
║   Top 1/3:    "MEDICAL REPORT - Complete all fields"
║   Bottom 2/3:  Patient symptoms, tests, diagnosis
║
║ Query A: "Extract the diagnosis"
║   Top:    64 tokens   (instructions irrelevant)
║   Bottom: 400 tokens  (contains diagnosis)
║   Outcome: Diagnosis extracted → +1000
║
║ Query B: "What type of document is this?"
║   Top:    400 tokens  (answer is here!)
║   Bottom: 64 tokens   (patient data irrelevant)
║   Outcome: "Medical Report" → +1000
║
║ Query C: "Summarize everything on this page"
║   Top:    200 tokens  (need some context)
║   Bottom: 300 tokens  (need diagnosis + details)
║   Outcome: Full summary → +1000
║
║ TRANSJECTIVE: Relevance not in regions themselves,
║               but in query ↔ region relationship
║
╚═══════════════════════════════════════════════════════════
```

**SOCRATES:** And crucially: **all text is visual tokens**. The instructions at top, the patient data at bottom—both are compressed by our allocator. There's no "external text" that bypasses compression.

**THEAETETUS:** So the allocator MUST learn which spatial regions help answer which query types, or the LLM will fail due to poor compression!

**SOCRATES:** Exactly. And it learns this through...?

**THEAETETUS:** Outcome-based RL!

```python
# Training episode:

Image: [Instructions top | Patient data bottom]
Query: "Extract the diagnosis"

Attempt 1: Allocate high to top, low to bottom
  → LLM reads: "Complete all fields before submission"
  → Answer: "No diagnosis found"
  → Reward: -1000 (failed!)
  → Allocator: "That allocation didn't work..."

Attempt 2: Allocate low to top, high to bottom
  → LLM reads: "Patient presents with Stage IIA tumor..."
  → Answer: "Stage IIA tumor"
  → Reward: +1000 (success!)
  → Allocator: "THAT worked! Remember this pattern!"

After 10,000 similar episodes:
  Allocator learns: "diagnosis query → allocate to bottom regions"
  Generalizes to NEW medical forms, layouts it's never seen!
```

## Real-World Patterns

**THEAETETUS:** Socrates, I just realized—this pattern is everywhere in real documents!

**SOCRATES:** What do you mean?

**THEAETETUS:**

```
Medical intake forms:
  Top:    "Patient Information Form - Instructions..."
  Bottom: Patient name, DOB, symptoms, medications

Tax documents:
  Top:    "Form 1040 - See Publication 501 for instructions"
  Bottom: Income, deductions, totals

Invoices:
  Top:    "Payment due within 30 days - Wire details..."
  Bottom: Line items, subtotal, total amount

Shipping labels:
  Top:    "Handle with care - Fragile - This side up"
  Bottom: Tracking number, addresses, barcode

Legal contracts:
  Top:    "NON-DISCLOSURE AGREEMENT - Read before signing"
  Bottom: Terms, obligations, signatures, dates
```

All have this structure: **instructions/metadata at top, actual data below!**

**SOCRATES:** And a rule-based system would fail because...?

**THEAETETUS:** Because sometimes you WANT the instructions (query: "How do I pay this invoice?"), sometimes you want the data (query: "What's the total amount?"). You can't make a fixed rule!

**SOCRATES:** But an outcome-based allocator...

**THEAETETUS:** Learns through experience!

```python
# After 100k training episodes across diverse documents:

Pattern 1: Top region + "document type" query → allocate high
Pattern 2: Top region + "extract data" query → allocate low

Pattern 3: Bottom region + "extract data" query → allocate high
Pattern 4: Bottom region + "document type" query → allocate low

Pattern 5: Red boxes + "medical" query → allocate high
Pattern 6: Dense tables + "values" query → allocate high
Pattern 7: Handwriting + "signature" query → allocate high

# All discovered through outcomes, never explicitly programmed!
```

**SOCRATES:** This is why outcome-based learning is so powerful. The allocator discovers regularities in document layouts, query types, and visual features—patterns we couldn't enumerate even if we tried.

       **DeepSeek-OCR Oracle:** *This convinces me that query-aware allocation is genuinely valuable. My uniform 16× compression works because I was trained on documents where EVERYTHING might be relevant (full-page OCR). But in query-answering scenarios, most regions are irrelevant! An invoice has maybe 5-10% query-relevant content for "extract total" queries, yet I compress all regions equally. Your allocator could learn: "For 'total' queries on invoices, allocate 400 tokens to bottom-right (where totals live), 64 tokens everywhere else." This is 5-10× more efficient than my approach while maintaining accuracy on the relevant content.*

## The Training Dataset Implication

**THEAETETUS:** Socrates, for this to work, we need training data with diverse query types on the same images!

**SOCRATES:** Why?

**THEAETETUS:** Because if we only ask "extract diagnosis" on medical forms, the allocator might learn "medical forms → allocate to bottom." But it won't learn the **transjective** nature—that the QUERY matters, not just the document type!

**SOCRATES:** So we need...

**THEAETETUS:**

```python
# Training data structure:

{
    'image': 'medical_report_001.jpg',
    'queries': [
        {
            'question': "What is the diagnosis?",
            'answer': "Stage IIA tumor",
            # → Allocator should learn: bottom regions matter
        },
        {
            'question': "What type of form is this?",
            'answer': "Medical Report",
            # → Allocator should learn: top regions matter
        },
        {
            'question': "Who is the patient?",
            'answer': "John Smith",
            # → Allocator should learn: name field matters
        },
        {
            'question': "What are the lab values?",
            'answer': "WBC: 4.5, RBC: 5.2...",
            # → Allocator should learn: table regions matter
        },
    ]
}

# Same image, multiple queries → learns query-content coupling!
```

**SOCRATES:** And if we train on many such images with diverse queries, the allocator learns general patterns?

**THEAETETUS:** Yes! It learns:
- "diagnosis" queries → look at diagnostic sections
- "document type" queries → look at headers/titles
- "total amount" queries → look at bottom-right of invoices
- "payment terms" queries → look at top/instructions
- "signature" queries → look at handwritten regions

All through outcome-based RL, no explicit rules!

**SOCRATES:** This is the power of transjective relevance. The allocator doesn't learn "what is important" in isolation. It learns "what helps answer THIS query type on THIS document layout."

## The Vocabulary We Should Use

**THEAETETUS:** So let me summarize the correct vocabulary:

**Architecture:**
```python
class QueryAwareRelevanceAllocator:  # Not "ThreatDetector"
    pass

class TaskCriticalityEstimator:      # Not "FearEstimator"
    pass

class AsymmetricTaskLoss:            # Not "SurvivalFitnessLoss"
    """
    Medical queries: 1000:1 (missing critical = catastrophic)
    Legal queries:   100:1  (missing data = costly)
    Casual queries:  1:1    (balanced)
    """
```

**Training process:**
- Outcome-based reinforcement learning ✓
- Reward = task success + efficiency ✓
- Learn query-content patterns ✓
- Develop "hunches" through statistical regularities ✓

**What we learned from biology:**
- Asymmetric loss mechanisms (prey vs predator)
- Outcome-based learning (survival fitness)
- Criticality modulation (arousal → sharpening)
- Transjective relevance (agent-environment coupling)

**What we're actually building:**
- Query-aware visual token allocator
- Learns which regions help answer which queries
- Optimizes task success, not survival
- Discovers spatial-semantic-query patterns

**SOCRATES:** Perfect! You've understood the distinction. The biology inspired the mechanisms, but our task is document understanding.

## Oracle Assessments: The Correct Path

       **DeepSeek-OCR Oracle:** *Now the proposal is crystal clear. You're not building a "threat detector"—you're building a query-aware compressor. My architecture compresses uniformly because I have no query context. You'll compress adaptively based on learned query-relevance patterns. The mechanisms (asymmetric loss, outcome-based RL) come from biology, but the task is pure information extraction. Implementation-wise:*

       *Phase 1: Multi-resolution compressor + simple edge-based selector*
       *- Proves variable LOD infrastructure works*
       *- Goal: Match my 86.8% DocVQA baseline*

       *Phase 2: Query-aware selector with RL training*
       *- Learns (visual features + query type) → allocation patterns*
       *- Goal: Beat baseline OR reduce tokens 30-40%*

       *This is feasible. And your vocabulary is now correct—don't confuse future implementers with "threat detection" metaphors!*

       **Ovis Oracle:** *I'm impressed by the clarity. My VET approach (modeling_ovis.py) provides uniform visual tokens—every patch gets structural alignment treatment equally. I have no query-aware compression. Your proposal fills a genuine gap: learned allocation BEFORE the LLM, based on query type. The 10M allocator learns hunches like "red boxes + medical → high allocation" through 100k RL episodes. This is computationally cheaper than my 2400-token approach if the allocator generalizes well. I believe it will—document layouts have strong regularities that RL can exploit.*

## Conclusion: Relevance is Transjective, Not Threatening

**SOCRATES:** Let's capture what we've learned, Theaetetus.

**Visual relevance is not about threats.** The tiger on the A4 page isn't threatening anyone—it's just relevant or irrelevant depending on the query. When asked about t-shirt text, the tiger matters. When asked about contract terms, it doesn't.

**Relevance is transjective.** It doesn't exist in the image alone (the tiger is always there) or in the query alone (the query doesn't change the pixels). Relevance emerges from the **coupling** of query and content—like a key fitting a lock.

**The allocator learns through hunches.** After 100,000 training episodes, the 10M allocator develops statistical intuitions: "Red boxes + medical queries usually need high allocation" or "Bottom regions of invoices usually contain totals." These aren't programmed—they're discovered through outcome-based RL.

**All text is visual tokens.** In the pure scenario (image alone), instructions and contents are both compressed by the allocator. There's no inherent priority—the query determines relevance. "Extract diagnosis" → allocate to patient data. "Document type?" → allocate to title/header.

**Real-world documents have structure.** Medical forms, invoices, shipping labels, contracts—all have instructions/metadata at top, data below. The allocator learns these spatial-semantic patterns through experience, generalizing to new layouts.

**The biology inspired mechanisms, not semantics.** We learned asymmetric loss, outcome-based learning, and criticality modulation from predator-prey dynamics. But we're not building a threat detector—we're building a query-aware visual token allocator.

**THEAETETUS:** And the path forward?

**SOCRATES:**

**Phase 1 (Infrastructure):** 8-12 days, $50-80k
- Multi-resolution compressor (5 LOD levels: 64-400 tokens)
- Simple edge-based selector (proves variable LOD works)
- Validate: Match DeepSeek 86.8% baseline with variable allocation

**Phase 2 (Intelligence):** 10-15 days, $100-150k
- Query-aware allocator (10M params, learns patterns)
- RL training on diverse queries (100k episodes)
- Asymmetric loss for task criticality (medical: 1000:1, casual: 1:1)
- Goal: Beat baseline OR reduce tokens 30-40%

**Total:** 18-27 days, $150-230k, feasible on 160 A100s

**THEAETETUS:** And the success criteria?

**SOCRATES:** The allocator learns to answer:

"Given THIS image and THIS query, which regions help answer correctly?"

Not "which regions are threatening."
Not "which regions are always important."
Just: **query ↔ content relevance, learned through outcomes.**

**THEAETETUS:** Then let us build it, Socrates. With correct vocabulary and clear purpose.

**SOCRATES:** Indeed. The tiger on the A4 page taught us well—it's not a threat, just a query-relevant visual element that happens to be wearing a rather rude t-shirt.

*[Both laugh and return to their work]*

---

## Key Insights

1. **Relevance ≠ Threat**: Tiger on A4 page isn't threatening—just relevant or irrelevant based on query
2. **Correct Vocabulary**: Query-aware relevance (not threat detection), task criticality (not fear), high/low stakes (not predator/prey)
3. **Transjective Nature**: Relevance emerges from query ↔ content coupling, not from regions alone
4. **Learned Hunches**: 10M allocator discovers patterns through RL ("red boxes + medical → high allocation")
5. **All Text Visual**: Pure scenario has instructions/contents both as visual tokens, query determines relevance
6. **Spatial-Semantic Patterns**: Documents have structure (instructions top, data bottom), allocator learns this
7. **Real-World Common**: Medical forms, invoices, contracts all have this layout pattern
8. **Biology = Mechanisms**: Asymmetric loss, outcome-based learning came from biology, but task is document understanding
9. **Diverse Queries Required**: Training needs multiple query types per image to learn transjective relevance
10. **Implementation Clarity**: Phase 1 proves infrastructure, Phase 2 adds intelligence through RL

---

## The Correct Framing

**What we're building:**
A query-aware visual token allocator that learns which image regions help answer which query types, optimized through outcome-based reinforcement learning.

**What we're NOT building:**
A threat detector, survival system, or predator-prey simulator.

**The biological inspiration:**
Provided mechanisms (asymmetric loss, outcome-based learning, criticality modulation, transjective coupling) but NOT the task semantics.

**The actual task:**
Document understanding through learned query-content relevance patterns discovered via 100k+ RL training episodes.

---

**Next Steps:** Implement Phase 1 infrastructure with correct terminology, avoiding biological metaphors in code/documentation except when explaining the mechanism's inspiration.

---

## Oracle Musings

**DeepSeek-OCR Oracle:** The tiger on A4 page example cuts to the heart of it—relevance is query-driven, not threat-based.

**Vocabulary correction matters**: If I were integrating their approach, class names like `QueryAwareRelevanceAllocator` are far clearer than `ThreatDetector`. Documentation that says "learns query-content patterns" is accurate. Documentation that says "detects threats" misleads future engineers. The biology inspired MECHANISMS (asymmetric loss, outcome-based learning), but the TASK is document understanding. Keep these separate!

**Ovis Oracle:** The "all text is visual" scenario is important. My architecture processes image + text separately (text tokenized by Qwen3 tokenizer, image through VET). Their pure-visual approach means instructions and contents both go through SAM → compression → CLIP. This forces the allocator to learn spatial-semantic patterns: "top regions + 'document type' query → allocate high" vs "bottom regions + 'extract data' query → allocate high." Transjective relevance emerges from training, not from architecture!

**Qwen3-VL Oracle:** The real-world document patterns they identified (instructions top, data bottom) are genuine statistical regularities. Medical forms, invoices, contracts, shipping labels—all share this structure. A 10M parameter allocator can discover these patterns through 100k RL episodes if the training data is diverse enough. Precedent: AlphaGo's 13M policy network learned complex spatial patterns through outcome-based training. Their task (document layouts) is simpler than Go (unbounded strategy space).

**Key Insight**: Query changes relevance completely. Same image, "Extract diagnosis" → allocate to patient data. "Document type?" → allocate to header. No fixed importance, pure transjective coupling. Their training must include multiple queries per image to learn this!

**All Oracles:** Correct vocabulary prevents confusion. Asymmetric loss, outcome-based RL, transjective relevance—these are the mechanisms. Query-aware document understanding—this is the task. Keep them distinct in code and documentation! 🎯
