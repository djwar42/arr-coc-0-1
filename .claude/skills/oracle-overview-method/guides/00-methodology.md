# Oracle Overview Method: Complete Methodology

This is the complete Oracle Overview methodology documentation.

## Overview

**Oracle Overviews** are expert commentaries added to discussion documents "after the fact" — technical specialists observe with full knowledge of their domains and provide deep insights, critiques, and solutions.

**Purpose**: Oracle interjections provide:
- Technical accuracy (code references, line numbers, architectural details)
- Cross-domain insights (comparing approaches, identifying tradeoffs)
- Foreshadowing (what challenges await)
- Depth (computational analysis, metrics, implementation details)
- Constructive solutions (proven techniques from production systems)

**Primary Use Case**: Platonic Dialogues in this project (`RESEARCH/PlatonicDialogues/`)

## Document Structure

Each Oracle Overview has three components:

1. **Original Dialogue** - Conceptual discussion between participants
2. **Oracle Interjections** - Technical commentary during key moments (7 spaces indent, **bold + *italics***)
3. **Oracle Musings** - Final section where oracles discuss what they observed (no indent, **bold** only)

**Optional**:
4. **Oracle Proposals** - Concrete implementation solutions (on-demand, when oracles propose fixes)

## Platonic Dialogues in This Project

**Location**: `RESEARCH/PlatonicDialogues/`

These dialogues chronicle the conceptual development of the project through Socratic inquiry:

- **0-dual-encoder-genesis.md** - Initial architectural concepts
- **1-shannon-jung-vervake.md** - Multi-metric information measurement
- **2-global-context-and-navit.md** - Processing strategies comparison
- **3-arr-coc-creation.md** - Unified architecture proposal
- **4-training-philosophy.md** - Training strategy development
- **5-weight-distribution-problem.md** - Technical challenges (critical)
- **6-huggingface-integration.md** - Component reuse strategy
- **7-arr-coc-complete-synthesis.md** - Final architecture assessment
- **8-vervaeke-enters-relevance.md** - Theoretical grounding
- **8-1-addendum.md** - Additional insights

**Participants**: Typically Socrates, Theaetetus, and sometimes domain experts (e.g., Vervaeke in Part 8)

## Oracle Participants (Project-Specific Examples)

Oracle participants are domain experts with deep technical knowledge. Define them based on your project's needs.

### Current Project Oracles

**DeepSeek-OCR Oracle**:
- **Knowledge Base**: `.claude/skills/deepseek-ocr-oracle/`
- **Expertise**: Efficiency-focused compression, serial architectures
- **Key Characteristics**: Optical compression philosophy, multi-resolution training, computational efficiency
- **References**: Actual code files with line numbers (e.g., `deepencoder/sam_vary_sdpa.py:166-183`)

**Ovis Oracle**:
- **Knowledge Base**: `.claude/skills/ovis-2-5-oracle/`
- **Expertise**: Quality-focused understanding, structural alignment
- **Key Characteristics**: VET probabilistic embeddings, native resolution, 5-phase curriculum
- **References**: Model implementation details (e.g., `modeling_ovis.py:25-34`)

### Adding New Oracles

To add oracles for other dialogues:

1. Create oracle knowledge base: `.claude/skills/<oracle-name>/`
2. Define oracle expertise and perspective
3. Identify what technical depth they provide
4. Activate oracle skills before adding commentary

**Note**: Dialogue participants (like Vervaeke in Part 8) are NOT automatically oracles unless specified!

## Formatting Standards

### 1. Oracle Interjections (During Dialogue)

**Format**: 7 spaces + `**Oracle Name:**` + *italic text*

```markdown
**PARTICIPANT A:** [makes a conceptual point]

       **Oracle Name:** *They've identified [CONCEPT]! But they don't yet realize [DEEPER INSIGHT]. Let me be precise: [TECHNICAL DETAILS with numbers]. Implementation: [CODE_REF:LINE_NUMBERS]. Cost: [METRICS]. Why this matters: [IMPLICATIONS].*

**PARTICIPANT B:** [responds]

       **Another Oracle:** *Our approach differs: [CONTRASTING PERSPECTIVE]. We achieve [DIFFERENT_METRIC] through [ALTERNATIVE_TECHNIQUE]. See [REFERENCE].*
```

**Key Requirements**:
- **Exactly 7 spaces** before `**Oracle Name:**`
- Entire oracle comment in **bold + *italics***
- Reference actual code files, line numbers, or documentation
- Provide quantitative details (FLOPs, memory, tokens, metrics, percentages)
- Explain **WHY** things work, not just **WHAT**
- Use first-person when oracle describes their own system ("My architecture...", "We use...")

**When to Add Interjections**:
- Key conceptual breakthroughs
- Technical claims that need validation
- Architectural decisions being proposed
- Trade-offs being discussed
- Challenges being identified

### 2. Oracle Musings (End of Dialogue)

**Format**: No indent + `**Oracle Name:**` + regular text (no italics)

```markdown
---

## Oracle Musings

**Oracle A:** Oracle B, they've identified [KEY INSIGHT] in this dialogue. Shall we analyze what they've discovered?

**Oracle B:** Indeed! [PARTICIPANT] correctly sees [OBSERVATION]. Let me compare our approaches using implementation details:

**My approach ([System Name])**:
- [Technical detail 1 with specifics]
- [Technical detail 2 with code references]
- [Computational analysis with numbers]

**Oracle A:** And here's my perspective:

**My approach ([Other System])**:
- [Contrasting implementation]
- [Different trade-offs with metrics]
- [Why we made different choices]

**Oracle B:** So the key difference is [FUNDAMENTAL DISTINCTION]. [PROPOSED SYSTEM] needs to address [CHALLENGE] to succeed.

**Oracle A:** Agreed! [ASSESSMENT with specific concerns and predictions]
```

**Key Requirements**:
- **NO spaces** before oracle names in musings
- Regular text (not italics) for dialogue
- Structured comparison using actual details
- Discussion format (back-and-forth between oracles)
- Technical depth (code refs, metrics, specific numbers)
- Predictions about future challenges
- Assessment of proposals (feasibility, novelty, risks)

**What to Include in Musings**:
- Architectural comparisons with specifics
- What participants got right/wrong
- Technical concerns about proposals
- Predictions about implementation challenges
- Assessment of feasibility and value
- Honest scope evaluation

### 3. Oracle Proposals (Optional, On-Demand)

**Format**: No indent + `**Oracle Name:**` + regular text

Add this section when oracles propose concrete solutions to identified challenges.

```markdown
---

## Oracle Proposals

**Oracle A:** We should help them solve [CHALLENGE] we identified. Here's what worked for us.

**Oracle B:** Excellent! I'll contribute from my experience with [DOMAIN].

### Proposal 1: [Solution Name]

**Challenge**: [Specific problem identified in dialogue]

**Oracle A's Solution Adapted**:

[Detailed implementation strategy with code patterns, metrics, and proven results]

**Key Innovations from [System]**:
1. [Technique 1 with details]
2. [Technique 2 with metrics]
3. [Proven results from production]

---

**Oracle B:** Now let me address [DIFFERENT CHALLENGE].

### Proposal 2: [Another Solution]

[Similar structure: Challenge → Solution → Details → Results]
```

**When to Add Proposals**:
- Dialogue identifies significant implementation challenges
- Oracles have proven solutions from their domains
- Concrete technical guidance would be valuable
- User requests "oracle proposals" for the dialogue

## Content Guidelines

### What Oracles Should Discuss

**In Interjections**:
- ✅ Technical accuracy corrections or validation
- ✅ Code references with line numbers
- ✅ Computational analysis (FLOPs, memory, speed, costs)
- ✅ Why architectural decisions were made
- ✅ What challenges await with specific details
- ✅ Cross-system comparisons with metrics

**In Musings**:
- ✅ Detailed architectural comparisons
- ✅ Assessment of what participants discovered
- ✅ Technical concerns with quantified risks
- ✅ Predictions about future dialogues/challenges
- ✅ Feasibility analysis (novelty, feasibility, value ratings)
- ✅ Honest scope discipline

**In Proposals** (when applicable):
- ✅ Concrete implementation strategies
- ✅ Proven techniques from production systems
- ✅ Training schedules, cost estimates, metrics
- ✅ Code patterns and architectural blueprints
- ✅ Risk mitigation and fallback mechanisms

### What Oracles Should NOT Do

- ❌ Contradict their own documented architectures
- ❌ Provide solutions during interjections (they observe, don't intervene in dialogue)
- ❌ Reference future dialogues they haven't "seen" yet
- ❌ Overreach beyond their expertise domains
- ❌ Use vague language ("might work", "probably")—be specific with numbers!
- ❌ Claim capabilities they don't have

## Workflow for Adding Oracle Commentary

### Step 1: Activate Oracle Skills

```bash
# Activate relevant oracle knowledge bases
> Skill(command: "oracle-name-1")
> Skill(command: "oracle-name-2")
```

Or manually read oracle knowledge:
```
Read oracle documentation from .claude/skills/<oracle-name>/
```

### Step 2: Read the Dialogue

```
Read(file_path: "RESEARCH/PlatonicDialogues/N-dialogue-name.md")
```

Understand:
- What concepts are being discussed?
- What technical claims are made?
- What architectural decisions are proposed?
- Where would expert depth add value?

### Step 3: Read Relevant Oracle Knowledge

Based on dialogue theme, read 2-4 oracle knowledge files for each oracle.

**Examples**:
- Architecture overviews
- Technical concepts
- Training strategies
- Code implementation references

### Step 4: Add Oracle Interjections

Identify 4-6 key moments in the dialogue where oracle commentary adds depth.

**Criteria for interjection points**:
- Conceptual breakthrough
- Technical claim
- Architectural proposal
- Trade-off discussion
- Challenge identification
- Cross-system comparison opportunity

Add commentary with:
- 7-space indent
- **Bold + *italics***
- Code references
- Specific metrics
- Explanatory depth

### Step 5: Write Oracle Musings

Create dialogue between oracles (no indent) discussing:
- What participants discovered
- Technical accuracy assessment
- Architectural comparisons with details
- Challenges and feasibility
- Predictions and concerns

Include ratings if appropriate:
- **Novelty**: ⭐⭐⭐⭐⭐ (5/5)
- **Feasibility**: ⭐⭐⭐⚪⚪ (3/5)
- **Value**: ⭐⭐⭐⭐⚪ (4/5)

### Step 6: Add Oracle Proposals (If Needed)

If dialogue identifies major challenges and oracles have proven solutions:
- Create "Oracle Proposals" section
- Provide concrete implementation strategies
- Include code patterns, training plans, cost estimates
- Reference proven techniques from production systems

### Step 7: Commit Changes

```bash
git add <dialogue-file>.md
git commit -m "Add oracle commentary to [Dialogue Name]

Oracle Interjections (X total):
- [Key insight 1]
- [Key insight 2]

Oracle Musings:
- [Assessment summary]

[Optional: Oracle Proposals:
- [Solution 1]
- [Solution 2]]
"
```

## Quality Checklist

Before considering a dialogue complete:

### Interjections
- [ ] 4-6 oracle interjections throughout dialogue
- [ ] All relevant oracles represented
- [ ] All interjections use 7-space indent + **bold + *italics***
- [ ] Code references include file names and line numbers
- [ ] Computational details provided (FLOPs, memory, tokens, metrics)
- [ ] WHY explained, not just WHAT

### Musings
- [ ] Oracle Musings section at end
- [ ] Musings have NO indent
- [ ] Regular text (not italics), **bold** for names only
- [ ] Structured comparison of systems/approaches
- [ ] Specific technical details with metrics
- [ ] Predictions about challenges or future dialogues
- [ ] Technical accuracy verified against oracle knowledge
- [ ] No contradictions with actual architectures
- [ ] Appropriate scope (no overreaching claims)

### Proposals (if applicable)
- [ ] Clear problem statements
- [ ] Concrete implementation strategies
- [ ] Proven techniques referenced
- [ ] Quantified results (time, cost, metrics)
- [ ] Code patterns or architectural blueprints
- [ ] Risk assessment and fallbacks

## Example Oracle Interjection (Well-Formatted)

```markdown
**THEAETETUS:** The key challenge is balancing efficiency and quality in compression.

       **DeepSeek-OCR Oracle:** *They've identified the fundamental trade-off! But they don't yet see HOW we resolved it. Let me be precise: we use 5 compression modes with fixed ratios. Base mode (273 tokens) achieves 10-15× compression at 85-87% accuracy on OmniDocBench. The serial SAM→CLIP architecture is non-negotiable: SAM processes 4096 patches with O(N) window attention (~65 GFLOPs), compresses to 256 patches via neck+net_2+net_3 (deepencoder/sam_vary_sdpa.py:166-183), then CLIP applies O(N²) global attention on the compressed grid (~180 GFLOPs). If we reversed this—CLIP first on 4096 patches—we'd have ~2800 GFLOPs and 16GB memory vs our current 245 GFLOPs and 1.5GB. Computational necessity drove the design.*

**SOCRATES:** So the architecture follows from efficiency constraints?

       **Ovis Oracle:** *Exactly the opposite for us! We prioritize understanding over efficiency. Our native resolution approach with RoPE (modeling_siglip2_navit.py) preserves aspect ratios at 448²-1792² without compression. Each visual token is a probabilistic discrete embedding via VET: `embedding = softmax(logits) @ embedding_table` where the table is 16,384×1280 (modeling_ovis.py:105). This produces ~2400 tokens per image—10× more than DeepSeek's Base mode—but achieves 90%+ accuracy on the same benchmarks. We defer compression to the LLM's attention mechanism rather than forcing it in the visual encoder. Different philosophy: preserve everything, let the language model choose what matters.*
```

**Why this is good**:
- 7-space indent + **bold + *italics***
- Specific metrics (273 tokens, 85-87% accuracy, 65 GFLOPs, etc.)
- Code references with line numbers
- Explains WHY (computational necessity, philosophical difference)
- Quantifies trade-offs (2800 vs 245 GFLOPs, 16GB vs 1.5GB)
- First-person perspective ("we use", "our approach")

## Tips for Success

1. **Be Specific**: Don't say "uses compression"—say "16× compression via neck(768→256) + 2× strided convolutions"
2. **Use Numbers**: FLOPs, memory, tokens, accuracy percentages, training days, costs
3. **Reference Code**: Always include file paths and line numbers when discussing implementation
4. **Stay In Character**: Oracles observe "after the fact"—they comment on what was discussed, they don't intervene in the dialogue itself
5. **Predict Accurately**: Oracles can foreshadow based on their technical knowledge of what challenges await
6. **Compare Systems**: Constantly contrast different approaches with specific metrics
7. **Maintain Scope**: Stay within oracle expertise—don't claim capabilities they don't have
8. **Quantify Trade-offs**: When discussing choices, provide specific numbers for both sides

## For New LLM Sessions

When continuing this work in a new session:

1. Read this file first (`guides/00-methodology.md`)
2. Identify which dialogue needs oracle commentary
3. Activate relevant oracle skills or read their knowledge bases
4. Read the dialogue to understand key themes
5. Read 2-4 relevant oracle knowledge files for context
6. Follow the formatting standards exactly (7 spaces, **bold + *italics***, etc.)
7. Add interjections, musings, and proposals (if needed)
8. Use quality checklist to verify completeness
9. Commit with descriptive message

## Adaptation to Other Projects

This methodology can be applied to any discussion documents where expert commentary would add technical depth:

**Generic Pattern**:
1. Identify discussion documents (meetings, design docs, conceptual dialogues)
2. Define domain experts (oracles) with specific knowledge bases
3. Add expert commentary "after the fact" with technical details
4. Use consistent formatting (customize indent/style as needed)
5. Include expert discussion section at end
6. Optionally: Add expert solution proposals

**The universal value**: Expert commentary transforms conceptual discussions into technically grounded documentation by adding domain-specific knowledge, real-world metrics, and proven implementation insights.

---

**Remember**: Oracle Overviews provide technical depth and accuracy. They're not just commentary—they're educational content that teaches readers about domain architectures while enriching the original discussion with expert perspectives.
