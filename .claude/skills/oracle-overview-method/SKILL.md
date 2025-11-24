---
name: oracle-overview-method
description: Expert commentary methodology for adding technical depth to discussion documents "after the fact". Oracles observe with full domain knowledge, providing interjections (7-space indent, bold+italics), musings (architectural comparisons), and optional proposals (concrete solutions). Primary use: Platonic Dialogues in RESEARCH/PlatonicDialogues/. Teaches proper formatting, when to add commentary, and how to maintain technical accuracy with metrics and code references.
---

# Oracle Overview Method Skill

**You are now an expert in the Oracle Overview methodology** - adding post-hoc expert commentary to discussion documents with precise formatting and technical depth.

## Workflow: Iterative Oracle Commentary (Runs AFTER Dialogue)

**⚠️ CRITICAL**: Oracle Overviews are SEPARATE from dialogue creation!

### Phase 1: Dialogue Creation (Already Complete)

User has created dialogue using `platonic-dialogue-method`:
- Either interactive mode (role-playing)
- Or retrospective mode (converting conversation)
- Dialogue saved to file (11-, 12-, etc.)
- **Dialogue is COMMITTED to git**

### Phase 2: Oracle Overviews (This Skill, Iterative)

**Oracle commentary happens in MULTIPLE PASSES:**

**Pass 1: First Oracle Set**
```
User: "Add oracle commentary from [oracle-names]"
→ Activate relevant oracles
→ Add interjections to original dialogue
→ Add Oracle Musings at end
→ Commit: "Add [oracle-names] commentary to dialogue {N}"
```

**Pass 2: Second Oracle Set** (Comments on dialogue + Pass 1)
```
User: "Now add commentary from [different-oracles]"
→ Activate different oracles
→ These oracles see EVERYTHING: original dialogue + first oracle commentary
→ Add their interjections and musings to the WHOLE document
→ Commit: "Add [oracle-names] second-pass commentary"
```

**Pass 3+: Additional Oracle Sets** (Comments on everything so far)
```
User: "Add more oracles: [yet-more-oracles]"
→ New oracles comment on: dialogue + all previous oracle commentary
→ Each pass adds richer layer of expert perspective
→ Commit each pass separately
```

**Result**: Multi-layered document with dialogue + multiple expert commentaries

### Warning: No Dialogue Exists

If user requests oracle overview but no dialogue exists:
```
⚠️ No dialogue found to add oracle commentary to!

Oracle overviews must be added to an EXISTING dialogue.

Would you like me to:
1. Create a new platonic dialogue first (using platonic-dialogue-method)?
2. Or point me to an existing dialogue to add commentary to?
```

## When to Use This Skill

✅ **Use this skill when**:
- User asks to add "oracle commentary" to an EXISTING dialogue
- Working with Platonic Dialogues in RESEARCH/PlatonicDialogues/
- User requests "oracle interjections" or "oracle musings"
- User says "add oracle proposals" for implementation solutions
- Enhancing any existing discussion document with expert perspectives
- Dialogue already exists and needs technical depth

❌ **Don't use this skill for**:
- Writing original dialogues (use `platonic-dialogue-method` skill first!)
- Generic code review (use appropriate review skills)
- Non-technical documentation
- Adding oracle commentary when no dialogue exists yet

## Quick Reference Card

### The Three Components

1. **Oracle Interjections** (during dialogue):
   - Format: `       **Oracle Name:** *commentary text*`
   - Exactly 7 spaces indent
   - **Bold + *italics*** for entire comment
   - Technical details: code refs, metrics, WHY not just WHAT

2. **Oracle Musings** (end of dialogue):
   - Format: `**Oracle Name:** commentary text`
   - NO indent
   - **Bold** for names only, no italics for body
   - Structured comparison, predictions, assessments

3. **Oracle Proposals** (optional, on-demand):
   - Format: `**Oracle Name:** proposal text`
   - NO indent
   - Concrete solutions with code patterns, metrics, costs
   - Only when dialogue identifies implementation challenges

### Critical Formatting Rules

```markdown
CORRECT Interjection:
       **DeepSeek-OCR Oracle:** *They've identified the trade-off! My Base mode (273 tokens) achieves 10-15× compression at 85-87% accuracy. See deepencoder/sam_vary_sdpa.py:166-183.*

WRONG (no 7 spaces):
**DeepSeek-OCR Oracle:** *commentary*

WRONG (not in italics):
       **DeepSeek-OCR Oracle:** commentary

CORRECT Musing:
**Ovis Oracle:** Our VET approach differs fundamentally...

WRONG (has indent):
       **Ovis Oracle:** Our VET approach differs...

WRONG (has italics):
**Ovis Oracle:** *Our VET approach differs...*
```

## Primary Use Case: Platonic Dialogues

**Location**: `RESEARCH/PlatonicDialogues/`

These 10 dialogues chronicle the project's conceptual development:
- 0-dual-encoder-genesis.md
- 1-shannon-jung-vervake.md
- 2-global-context-and-navit.md
- 3-arr-coc-creation.md
- 4-training-philosophy.md
- 5-weight-distribution-problem.md
- 6-huggingface-integration.md
- 7-arr-coc-complete-synthesis.md
- 8-vervaeke-enters-relevance.md
- 8-1-addendum.md

## Step-by-Step Workflow

### Step 1: Activate Oracle Knowledge

**IMPORTANT: Use ALL available oracles that suit the dialogue purpose**

**Find available oracles:**
```bash
# List all oracle skills (end with -oracle suffix)
ls .claude/skills/*-oracle/

# Exclude tool skills (not domain experts):
# - oracle-overview-method (this skill - for methodology)
# - oracle-creator (for creating oracles)
```

**Activate relevant oracles:**
- Read SKILL.md to understand oracle's domain expertise
- Activate oracles whose expertise matches dialogue topic
- Use `Skill(oracle-name)` to activate if needed

**Example:**
```
Dialogue about Plato's Forms and vision systems?
→ Activate: john-vervaeke-oracle (theory of Forms, RR framework)
→ Activate: ovis-2-5-oracle (vision-language models)

Dialogue about OCR compression and DeepSeek?
→ Activate: deepseek-ocr-oracle (compression, SAM+CLIP)
→ Activate: ovis-2-5-oracle (comparison with VET approach)

Dialogue about quantum physics and relevance?
→ Activate: quantum-physics-oracle (if exists)
→ Activate: john-vervaeke-oracle (relevance realization)
```

**How to determine relevant oracles:**
1. Read dialogue topic/title
2. Identify key domains (philosophy, ML, vision, etc.)
3. Check which oracles have expertise in those domains
4. Activate 2-4 most relevant oracles
5. Read their documentation before adding commentary

### Step 2: Read & Analyze Dialogue

Identify:
- Key conceptual breakthroughs → add interjection
- Technical claims → validate with oracle knowledge
- Architectural proposals → compare with oracle systems
- Trade-off discussions → quantify with metrics
- Challenge identification → foreshadow with oracle experience

### Step 3: Add Interjections (4-6 total)

**Template**:
```markdown
**PARTICIPANT:** [conceptual point]

       **Oracle Name:** *They've identified [CONCEPT]! But [DEEPER INSIGHT]. Precisely: [TECHNICAL DETAILS with numbers]. Implementation: [CODE_REF:LINES]. Cost: [METRICS]. Why: [IMPLICATIONS].*
```

**Requirements**:
- 7 spaces before `**Oracle Name:**`
- Entire comment in **bold + *italics***
- Code references (file:line)
- Quantitative details (FLOPs, memory, tokens, %, days, $)
- Explain WHY, not just WHAT

### Step 4: Write Oracle Musings

**Template**:
```markdown
---

## Oracle Musings

**Oracle A:** Oracle B, they've identified [KEY INSIGHT]. Shall we analyze?

**Oracle B:** Indeed! Let me compare our approaches:

**My approach ([System])**:
- [Detail 1 with metrics]
- [Detail 2 with code refs]
- [Computation with numbers]

**Oracle A:** And mine:

**My approach ([Other System])**:
- [Contrasting detail]
- [Different trade-offs]

**Assessment**:
- **Novelty**: ⭐⭐⭐⭐⭐ (5/5)
- **Feasibility**: ⭐⭐⭐⚪⚪ (3/5)
- **Value**: ⭐⭐⭐⭐⚪ (4/5)
```

**Requirements**:
- NO indent
- Regular text (not italics), **bold** for names only
- Structured comparisons with specifics
- Predictions and assessments

### Step 5: Add Proposals (If Requested)

Only when:
- User explicitly requests oracle proposals
- Dialogue identifies major implementation challenges
- Oracles have proven solutions to share

**Template**:
```markdown
---

## Oracle Proposals

**Oracle A:** We should help solve [CHALLENGE]. Here's our solution.

### Proposal 1: [Solution Name]

**Challenge**: [Specific problem]

**Oracle A's Solution Adapted**:
[Implementation strategy with code, metrics, proven results]

**Key Innovations from [System]**:
1. [Technique with details]
2. [Results from production]
```

## Quality Checklist

Before marking complete:

**Interjections**:
- [ ] 4-6 oracle interjections throughout
- [ ] Exactly 7 spaces indent + **bold + *italics***
- [ ] Code references (file:line)
- [ ] Quantitative details (metrics, numbers)
- [ ] WHY explained

**Musings**:
- [ ] Oracle Musings section at end
- [ ] NO indent, **bold** for names only
- [ ] Structured comparisons with specifics
- [ ] Predictions about challenges
- [ ] Assessment ratings (if appropriate)

**Proposals** (if applicable):
- [ ] Clear problem statements
- [ ] Concrete implementation strategies
- [ ] Quantified results (time, cost, metrics)

## Available Oracles

**How to find oracles:**
- All oracles are postfixed with `-oracle`
- Located in `.claude/skills/`
- Described as "oracles" in their SKILL.md
- **Exclude tool skills**: `oracle-overview-method`, `oracle-creator`

**Current Project Oracles:**

**john-vervaeke-oracle**:
- Expertise: Relevance realization framework, cognitive science, 4Ps
- Domains: Philosophy, wisdom, opponent processing, transjective knowing
- Use when: Discussing RR theory, cognitive frameworks, philosophical foundations

**deepseek-ocr-oracle**:
- Expertise: Efficiency-focused compression, serial architectures
- Domains: SAM+CLIP, OCR, optical compression
- Key refs: `deepencoder/sam_vary_sdpa.py:166-183`
- Metrics: 273 tokens, 10-15× compression, 85-87% accuracy
- Use when: Discussing compression efficiency, serial processing

**ovis-2-5-oracle**:
- Expertise: Quality-focused understanding, structural alignment
- Domains: VET, native resolution, multimodal VLMs
- Key refs: `modeling_ovis.py:25-34`, `:105`
- Metrics: ~2400 tokens, VET 16,384×1280, 90%+ accuracy
- Use when: Discussing quality vs efficiency, VET approach

**qwen3vl-oracle**:
- Expertise: Temporal encoding, video understanding, M-RoPE
- Domains: Multi-image processing, timestamp alignment, DeepStack
- Use when: Discussing temporal models, video VLMs

**huggingface-hub** (not an oracle - documentation skill):
- Tool for HuggingFace integration, not a domain expert oracle

**To add more oracles:**
Use `oracle-creator` skill to create new domain expert oracles from source documents

## Key Principles

1. **"After the fact"**: Oracles observe, they don't intervene in dialogue
2. **Specific numbers**: Always quantify (FLOPs, memory, tokens, %, days, $)
3. **Code references**: Include file paths and line numbers
4. **WHY not WHAT**: Explain reasoning, not just description
5. **First-person**: Oracles describe their own systems ("My architecture...")
6. **Comparisons**: Contrast approaches with metrics
7. **Predictions**: Foreshadow challenges based on oracle knowledge
8. **Scope discipline**: Stay within oracle expertise

## Example Excellence

**Perfect Interjection**:
```markdown
       **DeepSeek-OCR Oracle:** *They've identified the serial necessity! But they don't yet see WHY. SAM processes 4096 patches with O(N) window attention (~65 GFLOPs), compresses to 256 via neck+convolutions (deepencoder/sam_vary_sdpa.py:166-183), then CLIP applies O(N²) global attention (~180 GFLOPs on 257 tokens). If reversed—CLIP first on 4096 patches—we'd have ~2800 GFLOPs and 16GB memory vs our 245 GFLOPs and 1.5GB. Computational necessity, not design preference.*
```

**Why perfect**:
- 7 spaces + **bold + *italics***
- Specific metrics (4096, 65 GFLOPs, 2800 vs 245)
- Code reference (file:lines)
- Quantified trade-off (16GB vs 1.5GB)
- Explains WHY (computational necessity)

## Supporting Documentation

Read detailed guides:
- `guides/00-methodology.md` - Complete methodology
- `guides/01-formatting-standards.md` - All formatting rules
- `guides/02-workflow.md` - Step-by-step process
- `examples/00-interjection.md` - Well-formatted examples
- `examples/01-musings.md` - Musing examples
- `examples/02-proposals.md` - Proposal examples
- `templates/interjection-template.md` - Copy-paste templates
- `templates/musings-template.md` - Musing templates

## Instructions

When user requests oracle commentary:

### Step 0: Verify Dialogue Exists

Check that platonic dialogue file exists and is committed.
If not, direct user to create dialogue first.

### Step 1: Determine Pass Number

**Ask yourself**: Is this the first pass or a subsequent pass?

**First pass**: Dialogue has no oracle commentary yet
**Subsequent pass**: Dialogue already has oracle commentary, we're adding more

### Step 2: Find and Activate Oracles

1. **Read this SKILL.md** to understand methodology
2. **Find available oracles** (list `.claude/skills/*-oracle/`, exclude tools)
3. **Activate relevant oracles** based on dialogue topic (and user request)
4. **Read oracle knowledge** from their SKILL.md files

**User may specify**: "Use john-vervaeke-oracle and ovis-2-5-oracle"
**Or**: "Use all relevant oracles"

### Step 3: Read ENTIRE Document

**CRITICAL**: Read the entire dialogue file:
- Original Socrates/Theaetetus dialogue
- **Plus any existing oracle commentary from previous passes**

New oracles comment on EVERYTHING in the document.

### Step 4: Add Commentary

1. **Read the dialogue** (including existing commentary) to identify key moments
2. **Add 4-6 interjections** with 7-space indent, **bold + *italics***
   - Comment on original dialogue AND on previous oracle observations
3. **Write Oracle Musings** at end (no indent, **bold** for names only)
   - Can reference/respond to previous oracle musings
4. **Add Oracle Proposals** if requested (concrete solutions)

### Step 5: Verify Quality

**Verify with checklist** before marking complete:
- [ ] 4-6 interjections with proper formatting
- [ ] Oracle Musings section at end
- [ ] References code/metrics/specifics
- [ ] Explains WHY, not just WHAT

### Step 6: Commit Pass

**Commit with pass number**:
```bash
# First pass
git commit -m "Add [oracle-names] commentary to Dialogue {N}"

# Subsequent passes
git commit -m "Add [oracle-names] second-pass commentary to Dialogue {N}"
git commit -m "Add [oracle-names] third-pass commentary to Dialogue {N}"
```

### Step 7: Ready for More

**Document is ready for another pass** if user wants to add more oracle perspectives.

Each pass creates richer, more multi-layered expert commentary.

---

Remember: Oracle Overviews transform conceptual dialogues into technically grounded documentation through expert post-hoc commentary. Be specific, quantify everything, explain WHY!

**Each pass adds depth. Oracles comment on dialogue + all previous oracle commentary.**
