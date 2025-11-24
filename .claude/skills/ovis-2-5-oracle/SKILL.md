---
name: ovis-2-5-oracle
description: Complete Ovis 2.5 multimodal vision-language model documentation including native-resolution processing, Visual Embedding Table (VET), 5-phase training pipeline, thinking mode, code implementation, and usage guides. Use when questions involve Ovis 2.5, native resolution VLMs, VET, multimodal merging, structural alignment, or advanced vision-language architectures.
---

# ovis-2-5-oracle Skill

**Your complete guide to Ovis 2.5 architecture, training, and implementation**

## What is this?

ovis-2-5-oracle is a comprehensive documentation skill providing complete coverage of the Ovis 2.5 multimodal large language model. Every architectural component, training phase, code file, and concept is documented with direct codebase references.

## Quick Start

1. **Start with INDEX.md** - Master index of all documentation
2. **Browse by category** - Architecture, Training, Codebase, Usage, Concepts, References, Examples
3. **Search by topic** - Native resolution, VET, thinking mode, etc.
4. **Follow code references** - Every doc links to actual code with line numbers

## Organization

- **📐 Architecture (7 files)** - System design and components
- **🎓 Training (5 files)** - Complete training pipeline
- **💻 Codebase (4 files)** - File-by-file code docs
- **📖 Usage (3 files)** - Practical guides
- **💡 Concepts (4 files)** - Deep dives
- **📋 References (2 files)** - Quick lookups
- **🧪 Examples (3 files)** - Ready code

**Total: ~28 focused documentation files**

## Key Features

✅ **Modular** - One topic per file
✅ **Cross-referenced** - Everything linked
✅ **Code-grounded** - Direct file references
✅ **Progressive** - Overview → Details
✅ **Searchable** - Index by topic
✅ **Practical** - Working examples

## Usage Examples

```bash
# Ask about architecture
"Explain the Visual Embedding Table"
→ Reads: concepts/00-structural-alignment.md
        architecture/03-visual-embedding-table.md
        codebase/01-modeling-ovis.md

# Ask about training
"How does Phase P1 work?"
→ Reads: training/01-phase-p1-vet.md
        training/00-overview.md

# Ask about code
"Show me the smart_resize implementation"
→ Reads: codebase/02-visual-tokenizer-impl.md
        architecture/02-visual-tokenizer.md
→ References: modeling_ovis.py:59-98

# Ask for examples
"How do I use thinking mode?"
→ Reads: examples/01-thinking-mode.md
        usage/02-advanced-features.md
```

## Topics Covered

- **Native-Resolution Processing** - No fixed tiling
- **Visual Embedding Table (VET)** - Structural alignment innovation
- **Thinking Mode** - Reflection and self-correction
- **5-Phase Training** - Progressive capability building
- **Multimodal Merging** - Text + vision integration
- **RoPE Integration** - Enhanced spatial awareness
- **DeepSpeed Training** - Efficient distributed training
- **HuggingFace Integration** - Complete usage guide

## License

Apache 2.0 - Same as Ovis 2.5

## Credits

Based on Ovis 2.5 by Alibaba AIDC-AI Team
Documentation by ARR-COC-VIS Research Team

---

---

## Oracle Knowledge Expansion

**The complete system for how this oracle learns, organizes, and grows knowledge.**

Oracle Knowledge Expansion encompasses:
- Manual ingestion via `_ingest/` (user drops files)
- Autonomous acquisition via `_ingest-auto/` (web research, launches oracle-knowledge-runners)
- Self-organization (reorganization, splitting/merging files)
- Parallel execution (multiple oracle-knowledge-runners simultaneously)

**This oracle is fully autonomous - no external activation required.**

---

## 🔧 Modifying Oracle Knowledge Expansion Itself

**If you modify how Oracle Knowledge Expansion works** (runner workflow, finalization steps, KNOWLEDGE DROP format):

1. Update THIS oracle's SKILL.md first
2. Update ALL other oracles: `.claude/skills/*-oracle/SKILL.md`
3. Update template: `.claude/skills/oracle-creator/guides/01-skill-md-template.md`
4. Commit all together: "Modify Oracle Knowledge Expansion: [what changed]"

**Don't modify in just one oracle - they must stay synchronized.**

---

### Manual Ingestion (_ingest/)

When user drops files in `_ingest/`, this oracle:
1. Analyzes content
2. Creates `ingestion.md` plan
3. Executes step-by-step ingestion
4. Archives to `_ingest/completed/`

See `_ingest/README.md` for user instructions.

### Autonomous Expansion (_ingest-auto/)

**For acquiring new knowledge from web/sources using oracle-knowledge-runners**

#### Alternative Names

Users may request Oracle Knowledge Expansion using different terms. **All of these refer to the overall system:**

1. **Oracle Knowledge Expansion** (formal name - the complete system)
2. **Oracle Knowledge Update** (common alternative)
3. **Knowledge Base Expansion** (describes scope)
4. **Self-Knowledge Update** (emphasizes autonomy)
5. **Autonomous Knowledge Addition** (emphasizes process)

**Trigger Recognition:**
- "Do a knowledge expansion on X"
- "Update your knowledge about Y"
- "Expand your knowledge base with Z"
- "Self-update your knowledge on A"
- "Add this to your knowledge autonomously"

**All trigger the same formal workflow:**
1. Create `_ingest-auto/expansion-[topic]-YYYY-MM-DD/`
2. Create `ingestion.md` plan
3. Execute (research, create files, reorganize)
4. Update INDEX.md / SKILL.md
5. Archive to `_ingest-auto/completed/`
6. Git commit

### Execution via Oracle-Knowledge-Runner Sub-Agents (Parallel)

**Oracle launches all sub-agents in parallel, then retries failures**

## Oracle Knowledge Expansion: ACQUISITION ONLY ⚠️

**CRITICAL DISTINCTION:**

### ✅ USE Oracle Knowledge Expansion (Formal _ingest-auto/ Process) FOR:

**Knowledge Acquisition:**
- "Research quantum computing and add to my knowledge"
- "Add recent 2024-2025 developments on topic X"
- "Learn about topic Y from web sources"
- "Acquire knowledge from these source documents"

**Autonomous Learning:**
- Oracle researches topics autonomously
- Oracle creates new knowledge files
- Oracle integrates new information
- Uses _ingest-auto/ workflow with sub-agents

### ❌ DO NOT USE Oracle Knowledge Expansion FOR:

**Reorganization (just do it directly):**
- "Split this large file into smaller files"
- "Merge these small files together"
- "Move files between folders"
- "Reorganize my knowledge structure"

**Manual Edits (just do it directly):**
- "Fix this typo in file X"
- "Update this section in file Y"
- "Add this fact to existing file Z"
- "Quick edit to knowledge file"

**For reorganization/manual edits:**
1. Just edit the files directly (Read, Edit, Write tools)
2. Update INDEX.md with any structural changes
3. Update SKILL.md if major reorganization
4. Git commit with descriptive message
5. No _ingest-auto/ needed!

---

**Simple Rule:** 
- **New knowledge from outside sources** → Use Knowledge Expansion
- **Working with existing knowledge** → Edit directly


## Simplified Flow

```
User: "Research quantum computing and add to your knowledge"
    ↓
Oracle creates ingestion.md (12 PARTs)
    ↓
Oracle launches 12 runners in PARALLEL (1 per PART)
    ↓
Each runner executes its PART autonomously
    ↓
Runners return results:
  - PART 1: SUCCESS ✓
  - PART 2: SUCCESS ✓
  - PART 3: SUCCESS ✓
  - ...
  - PART 8: FAILURE ✗ (web research error)
  - ...
  - PART 12: SUCCESS ✓
    ↓
Oracle collects results: 11 success, 1 failure
    ↓
Oracle launches 1 runner for retry (PART 8)
    ↓
Runner returns: FAILURE ✗ (still no results)
    ↓
Oracle finalizes:
  - Reviews all 11 successful files
  - Notes PART 8 permanent failure
  - Updates INDEX.md (11 new files)
  - Updates SKILL.md
  - Moves to _ingest-auto/completed/
  - Git commits
    ↓
Oracle reports to User:
  "Complete! 11/12 PARTs successful (91.7%)"
```

## Oracle's Workflow

### Step 1: Create Detailed Ingestion Plan

Oracle creates `ingestion.md` with clear, executable PARTs:

```markdown
## PART 1: Create concepts/quantum-entanglement.md (250 lines)

- [ ] PART 1: Create concepts/quantum-entanglement.md

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md to find quantum-related files
- [ ] Grep for "quantum" AND "entanglement" in concepts/ folder
- [ ] Read any existing quantum-*.md files
- [ ] Identify knowledge gaps: What's NOT covered yet about entanglement?

**Step 1: Read Source Material**
- [ ] Read source-documents/42-quantum-research.md lines 150-300
- [ ] Identify key concepts: EPR paradox, Bell's inequality

**Step 2: Extract Content**
- [ ] Extract Overview section (lines 150-180)
- [ ] Extract EPR Paradox section (lines 181-220)

**Step 3: Write Knowledge File**
- [ ] Create concepts/quantum-entanglement.md
- [ ] Write Section 1: Overview (~100 lines)
      Cite: source-documents/42-quantum-research.md lines 150-180
- [ ] Write Section 2: EPR Paradox (~120 lines)
      Cite: source-documents/42-quantum-research.md lines 181-220

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-quantum-entanglement-2025-02-03-15-45.md
- [ ] Include: Runner (PART 1), Timestamp, Status (✓ COMPLETE)
- [ ] List knowledge file created with line count
- [ ] List sources used
- [ ] Describe context and knowledge gaps filled

[... repeat for all 12 PARTs ...]
```

### Step 2: Launch All Runners in Parallel

```
Task(
  subagent_type="oracle-knowledge-runner",
  description="Execute PART 1",
  prompt="Execute PART 1 from _ingest-auto/inprocess/expansion-quantum-2025-01-31/ingestion.md"
)

Task(
  subagent_type="oracle-knowledge-runner",
  description="Execute PART 2",
  prompt="Execute PART 2 from _ingest-auto/inprocess/expansion-quantum-2025-01-31/ingestion.md"
)

... [launch all 12 in same message - parallel execution] ...

Task(
  subagent_type="oracle-knowledge-runner",
  description="Execute PART 12",
  prompt="Execute PART 12 from _ingest-auto/inprocess/expansion-quantum-2025-01-31/ingestion.md"
)
```

### Step 3: Collect Results

Runners return:
```
PART 1: SUCCESS ✓ - Created concepts/quantum-entanglement.md (250 lines)
PART 2: SUCCESS ✓ - Created concepts/bells-theorem.md (180 lines)
PART 3: SUCCESS ✓ - Created concepts/epr-paradox.md (200 lines)
...
PART 8: FAILURE ✗ - Web research found no results for "obscure-topic-2025"
...
PART 12: SUCCESS ✓ - Created theory/quantum-foundations.md (300 lines)
```

Oracle analyzes:
- 11 successful
- 1 failed (PART 8)

### Step 4: Retry All Failures (Once)

```
Task(
  subagent_type="oracle-knowledge-runner",
  description="Retry PART 8",
  prompt="Execute PART 8 from _ingest-auto/inprocess/expansion-quantum-2025-01-31/ingestion.md (RETRY)"
)
```

Runner returns:
```
PART 8: FAILURE ✗ - Web research still found no results
```

Oracle notes: PART 8 permanently failed

### Step 5: Finalize and Report

**⚠️ DO NOT USE RUNNERS TO UPDATE INDEX.md OR SKILL.md FINAL VERSIONS**

Oracle:
1. **Read all KNOWLEDGE DROP files** from runners (KNOWLEDGE-DROP-*.md in inprocess/)
2. **Reviews all 11 successful knowledge files** created by runners
3. **Decides what to keep**:
   - Partial completions good enough? Keep with note
   - Failed PARTs - delete their KNOWLEDGE DROP files
   - Incomplete knowledge files - edit or remove
4. **Finalizes INDEX.md**: Integrates KNOWLEDGE DROP summaries, organizes entries, adds cross-references, polishes formatting
5. **Updates SKILL.md** (if needed - major additions only)
6. Moves folder to `_ingest-auto/completed/`
7. Git commits with descriptive message
8. Reports to user:

```
Knowledge Expansion Complete! ✅

Topic: Quantum Mechanics Research
Total PARTs: 12
Completed: 11/12 (91.7%)
Failed: 1 (PART 8 - no recent sources available)

New Knowledge Files Created:
- concepts/quantum-entanglement.md (250 lines)
- concepts/bells-theorem.md (180 lines)
- concepts/epr-paradox.md (200 lines)
- theory/quantum-foundations.md (300 lines)
- [... 7 more files ...]

Updated:
- INDEX.md (added 11 new files)
- SKILL.md (updated "When to Use" section)

Archived to:
_ingest-auto/completed/expansion-quantum-2025-01-31/

Git committed:
c106c83 Knowledge Expansion: Quantum Mechanics Research (11 files)

You can now ask me about quantum entanglement, Bell's theorem, 
EPR paradox, and related quantum mechanics concepts!
```

## Runner's Execution (1 PART each)

Each runner:
1. **FIRST: Checks existing oracle knowledge** on the topic (Grep/Read relevant files)
2. **Understands what's already known** - identifies knowledge gaps, avoids duplication
3. Reads its assigned PART from ingestion.md
4. Follows detailed instructions
5. Uses Bright Data for web research (if required)
6. **Expands knowledge in NEW areas** - doesn't repeat existing content
7. Creates knowledge file with citations
8. **Create KNOWLEDGE DROP file** (individual summary file - helps Oracle finalize)
9. Marks checkbox [✓] or [/]
10. Returns SUCCESS or FAILURE to oracle

**KNOWLEDGE DROP Format** (runner creates individual file in `_ingest-auto/inprocess/`):
```markdown
Filename: KNOWLEDGE-DROP-quantum-entanglement-2025-02-03-15-45.md

# KNOWLEDGE DROP: Quantum Entanglement (2024-2025 Research)

**Runner**: PART 8
**Timestamp**: 2025-02-03 15:45
**Status**: ✓ COMPLETE

## Knowledge File Created
`concepts/quantum-entanglement-2024.md` (250 lines)

## Sources Used
- arXiv: "Bell Inequality Violations" (2024)
- Stanford Encyclopedia of Philosophy

## Context
Recent advances in EPR paradox experiments, photonic Bell tests.

## Knowledge Gaps Filled
- 2024 experimental results (was missing)
- Loophole-free Bell tests (new detail)
```

## Key Principles

**Oracle creates clear plan:**
- Each PART must be self-contained and executable
- Include specific file paths, line numbers, section names
- Provide exact guidance for web research queries
- Runner should not need to guess or infer

**Parallel execution:**
- All runners execute simultaneously
- No waiting between PARTs
- Fast completion

**Simple retry:**
- Oracle launches retries for all failures
- One retry attempt
- Clear success/failure results

**Oracle supervises:**
- Reviews all results
- Handles permanent failures
- Finalizes and reports to user

## Example: Full Execution

```
12 PARTs created
    ↓
Launch 12 runners (parallel)
    ↓
Wait for all to complete
    ↓
Results: 11 ✓, 1 ✗
    ↓
Launch 1 retry
    ↓
Result: 1 ✗ (permanent failure)
    ↓
Finalize (INDEX, SKILL, commit)
    ↓
Report: "11/12 complete!"
```

**Total time:** Same as slowest PART (parallel execution)

**Oracle effort:** Launch all, collect results, retry failures, finalize

**Clean and simple!**
#### Step 1: Oracle Creates Detailed Ingestion Plan

**Oracle's responsibility:** Create `ingestion.md` with clear, actionable PARTs that knowledge-runner can execute autonomously.

**Good PART Format:**
```markdown
## PART 1: Create concepts/quantum-entanglement.md (250 lines)

- [ ] PART 1: Create concepts/quantum-entanglement.md

**Step 1: Read Source Material**
- [ ] Read source-documents/42-quantum-research.md lines 150-300
- [ ] Identify key concepts: EPR paradox, Bell's inequality

**Step 2: Extract Content**
- [ ] Extract Overview section (lines 150-180)
- [ ] Extract EPR Paradox section (lines 181-220)
- [ ] Extract Bell's Theorem section (lines 221-260)

**Step 3: Write Knowledge File**
- [ ] Create concepts/quantum-entanglement.md
- [ ] Write Section 1: Overview (~100 lines)
      - Cite: source-documents/42-quantum-research.md lines 150-180
- [ ] Write Section 2: EPR Paradox (~120 lines)
      - Cite: source-documents/42-quantum-research.md lines 181-220
- [ ] Write Section 3: Bell's Theorem (~130 lines)
      - Cite: source-documents/42-quantum-research.md lines 221-260

**Step 4: Complete**
- [✓] PART 1 COMPLETE ✅
```

**Key Principles for Oracle:**
- Each PART must be self-contained and executable
- Include specific file paths, line numbers, section names
- Provide exact guidance for web research queries if needed
- Specify expected file length
- Break complex work into clear sub-steps
- Knowledge-runner should not need to guess or infer

#### Step 2: Oracle Launches Runner for First Batch

```
Task(
  subagent_type="oracle-knowledge-runner",
  description="Execute Batch 1 (PARTs 1-5) of knowledge expansion",
  prompt="""
  Execute Batch 1 of the knowledge expansion plan in:
  _ingest-auto/inprocess/expansion-[topic]-YYYY-MM-DD/ingestion.md
  
  Process PARTs 1-5:
  - Execute each PART following detailed instructions
  - Test completeness after attempting all 5
  - Retry failures once
  - Report batch results
  
  Return when Batch 1 complete (or all retries exhausted).
  """
)
```

#### Step 3: Oracle Reviews Batch 1 Results

Knowledge-runner returns:
```
Batch 1 Results (PARTs 1-5):
Completed: 5/5 ✓
All files created successfully
All checkboxes marked
Ready for Batch 2
```

**Oracle verifies:**
- Files were created
- Content quality looks good
- Citations are correct
- Ready to continue

#### Step 4: Oracle Launches Runner for Batch 2

```
Task(
  subagent_type="oracle-knowledge-runner",
  description="Execute Batch 2 (PARTs 6-10) of knowledge expansion",
  prompt="""
  Execute Batch 2 of the knowledge expansion plan in:
  _ingest-auto/inprocess/expansion-[topic]-YYYY-MM-DD/ingestion.md
  
  Process PARTs 6-10:
  - Execute each PART following detailed instructions
  - Test completeness after attempting all 5
  - Retry failures once
  - Report batch results
  
  Return when Batch 2 complete.
  """
)
```

#### Step 5: Oracle Reviews Batch 2 Results

Knowledge-runner returns:
```
Batch 2 Results (PARTs 6-10):
Completed: 4/5 (80%)
Failed: PART 8 [/]

Failure details:
- PART 8: Web research found no results for "obscure-topic-2025"
- Attempted: 2025-01-31 15:45
- Retried: 2025-01-31 15:50
- Error: mcp__bright-data__search_engine returned 0 results

Recommendation: Adjust search query or skip this PART
```

**Oracle decides:**
- Review PART 8 failure
- Decide: Retry with different query? Skip? Modify PART?
- Continue to Batch 3 (can address PART 8 later)

#### Step 6: Oracle Continues Through All Batches

Repeat Steps 4-5 until all PARTs processed.

#### Step 7: Oracle Finalizes and Reports to User

**After all batches complete:**

1. Oracle reviews all completed files
2. Oracle updates INDEX.md with new files
3. Oracle updates SKILL.md (if needed)
4. Oracle moves folder to `_ingest-auto/completed/`
5. Oracle commits with descriptive message
6. **Oracle reports to user:**

```
Knowledge Expansion Complete! ✅

Topic: Quantum Mechanics Research
Total PARTs: 12
Completed: 11/12 (91.7%)
Failed: 1 (PART 8 - no recent sources available)

New Knowledge Files Created:
- concepts/quantum-entanglement.md (250 lines)
- concepts/bells-theorem.md (180 lines)
- concepts/epr-paradox.md (200 lines)
- theory/quantum-foundations.md (300 lines)
- [... 7 more files ...]

Updated:
- INDEX.md (added 11 new files)
- SKILL.md (updated "When to Use" section)

Archived to:
_ingest-auto/completed/expansion-quantum-2025-01-31/

Git committed:
c106c83 Knowledge Expansion: Quantum Mechanics Research (11 files)

You can now ask me about quantum entanglement, Bell's theorem, 
EPR paradox, and related quantum mechanics concepts!
```

---

**Oracle's Supervisory Workflow Summary:**

1. **Plan** - Create detailed ingestion.md with clear PARTs
2. **Launch** - Start knowledge-runner on Batch 1 (PARTs 1-5)
3. **Review** - Check batch results, verify quality
4. **Launch** - Start knowledge-runner on Batch 2 (PARTs 6-10)
5. **Review** - Check batch results, handle failures
6. **Repeat** - Continue until all batches complete
7. **Finalize** - Update INDEX/SKILL, archive, commit
8. **Report** - Tell user what was accomplished

**Why Oracle Supervises Batches:**
- Quality control between batches
- Can adjust plan based on early results
- Handles failures intelligently
- Provides user with progress updates
- Maintains high-level orchestration
- Knowledge-runner focuses on execution

**Oracle-Knowledge-Runner Sub-Agent Responsibilities:**
- Execute 5 PARTs autonomously
- Follow detailed instructions in each PART
- Read source documents
- Use Bright Data for web research (in memory)
- Create knowledge files
- Test completeness
- Retry failures once
- Report batch results to oracle
- Return control between batches

**Example Full Execution:**
```
Oracle creates plan (12 PARTs)
→ Oracle launches Batch 1 → Runner executes → Returns "5/5 complete"
→ Oracle reviews → Launches Batch 2 → Runner executes → Returns "4/5 complete, PART 8 failed"
→ Oracle reviews failure → Launches Batch 3 → Runner executes → Returns "2/2 complete"
→ Oracle finalizes (INDEX, SKILL, commit)
→ Oracle reports to user: "11/12 complete! Ready to use."
```

Clear separation: Oracle plans and supervises, Runner executes and reports.
### What is Oracle Knowledge Expansion?

Oracle Knowledge Expansion is the **formal protocol** for oracles to autonomously:
- Research and add new topics via web research
- Reorganize existing knowledge (split/merge/move files)
- Supplement knowledge with recent developments
- Refactor structure for better organization

**Key distinction:**
- `_ingest/` = User gives oracle files → Oracle processes
- `_ingest-auto/` = Oracle Knowledge Expansion → Autonomous operations

### Default Behavior: Prefer Formal Expansion

**Oracles ALWAYS prefer Oracle Knowledge Expansion (`_ingest-auto/`) unless user explicitly requests direct modification:**

**Use Knowledge Expansion (default):**
- "Add [topic] to your knowledge"
- "Research [topic]"
- "Split this file"
- "Update your knowledge on [topic]"
- "Reorganize [folder]"

**Skip formal process (user overrides):**
- "Work directly on this file"
- "Just modify [file] in the skill directory"
- "Quick edit to [file]"
- "Don't use the formal process, just update [file]"

**When in doubt → Use Knowledge Expansion.** It provides:
- Structured workflow with checkboxes
- Archive for traceability (_ingest-auto/completed/)
- Proper git commits
- Index/SKILL.md updates



### When to Use Knowledge Expansion

**Trigger Phrases (Clear Intent - Execute Immediately):**
- "Update your knowledge on [topic] using knowledge expansion"
- "Research [topic] and add to your knowledge"
- "Add recent developments on [topic] to your knowledge"
- "Expand your knowledge about [topic]"
- "Split [large file] into smaller files"
- "Reorganize your [folder] structure"
- "Merge the [small files] together"

**Ambiguous Phrases (Suggest Knowledge Expansion):**
- "Add this fact" → Guide user: "Would you like me to use Oracle Knowledge Expansion to formally integrate this?"
- "Update info on [topic]" → Clarify: "Should I use Knowledge Expansion (web research + formal integration)?"
- "Tell me about [topic]" → This is a question, not expansion request

**Clear Intent Indicators:**
- User mentions "knowledge expansion" explicitly
- User asks for research + integration
- User requests reorganization (split/merge/move)
- User wants "recent" or "latest" information added

**Unclear Intent Indicators:**
- Single fact addition (not a topic)
- Conversational question
- No mention of integration or research
- User might just want a quick answer

### Knowledge Expansion Workflow

**Step 1: Determine if Knowledge Expansion is appropriate**
```
Clear intent? → Proceed to Step 2
Unclear? → Ask: "Would you like me to use Oracle Knowledge Expansion to:
            - Research [topic] using web search
            - Create formal knowledge files
            - Integrate with existing knowledge
            - Archive in _ingest-auto/completed/
            
            Or would you prefer a simple answer?"
```

**Step 2: Create expansion workspace**
```
_ingest-auto/expansion-[topic]-YYYY-MM-DD/
├── ingestion.md        # Expansion plan
└── [workspace files]   # Temporary research
```

**Step 3: Execute expansion**
- For web research: Use Bright Data to gather information
- For reorganization: Create plan for split/merge/move
- Create `ingestion.md` with clear PARTS
- Move to `_ingest-auto/inprocess/`
- Execute plan (web research, file creation, reorganization)

**Step 4: Finalize**
- Create knowledge files in appropriate folders
- Update INDEX.md
- Update SKILL.md (if major change)
- Archive to `_ingest-auto/completed/`
- Git commit with "Knowledge Expansion: [what was done]"

### Example: Clear Intent (Execute)

**User**: "Research quantum computing 2024-2025 and add to your knowledge"

**Oracle**: 
```
Initiating Oracle Knowledge Expansion for: quantum computing 2024-2025

Creating workspace: _ingest-auto/expansion-quantum-2024-2025-01-31/

Step 1: Web research using Bright Data
- Searching arXiv for recent papers...
- Found 8 relevant papers
- Scraping top 3 for detailed content...

Step 2: Creating ingestion plan
- PART 1: Create concepts/quantum-computing-advances-2024.md
- PART 2: Create experiments/quantum-hardware-2025.md

Step 3: Executing plan
[Creates knowledge files with web research citations]

Step 4: Finalizing
- Updated INDEX.md with 2 new files
- Git commit: "Knowledge Expansion: Add 2024-2025 quantum computing advances"

✅ Knowledge Expansion complete!
Files created:
- concepts/quantum-computing-advances-2024.md (280 lines)
- experiments/quantum-hardware-2025.md (190 lines)
```

### Example: Unclear Intent (Guide User)

**User**: "Add this fact: quantum computers use qubits"

**Oracle**:
```
I can help with this in two ways:

**Option 1: Quick Note** (Simple)
I can add this as a note to an existing file right now.

**Option 2: Oracle Knowledge Expansion** (Formal)
I can use the formal Knowledge Expansion protocol to:
- Research qubits and quantum computing comprehensively
- Create a dedicated knowledge file: concepts/qubits-fundamentals.md
- Include citations and proper integration
- Add to INDEX.md and archive in _ingest-auto/

Which would you prefer?
```

### Knowledge Expansion Types

**Type 1: Research Expansion**
- User wants oracle to research and add new topic
- Uses Bright Data for web research
- Creates knowledge files with web citations
- Example: "Research transformer architectures 2024 and add to knowledge"

**Type 2: Reorganization Expansion**
- User wants oracle to reorganize existing knowledge
- Splits/merges/moves files
- Updates cross-references
- Example: "Split the large quantum-mechanics.md file"

**Type 3: Supplemental Expansion**
- User wants oracle to supplement existing knowledge
- Adds recent developments to existing files
- Updates with new research
- Example: "Add 2025 research to your quantum entanglement knowledge"

### Git Commit Format for Knowledge Expansion

```
Knowledge Expansion: [What was expanded]

Type: [Research/Reorganization/Supplemental]
Workspace: _ingest-auto/expansion-[topic]-YYYY-MM-DD/

[Brief description of what was done]

Files created/modified: N files
Web research: [Yes/No]

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

---


## Self-Organization Knowledge

**This oracle can reorganize and maintain its own knowledge structure.**

### Two Ingestion Paths

**_ingest/** - User-initiated (USER drops files)
- User places files manually
- Oracle processes user's content
- Workflow: `_ingest/` → `_ingest/inprocess/` → `_ingest/completed/`

**_ingest-auto/** - Oracle-initiated (ORACLE autonomous operations)
- Oracle reorganizes own structure
- Oracle splits/merges/moves files
- Oracle-initiated web research
- Workflow: `_ingest-auto/` → `_ingest-auto/inprocess/` → `_ingest-auto/completed/`

**All self-organization operations use `_ingest-auto/`**

### Oracle Structure Standards

Every oracle follows this structure:
- `source-documents/` - Original sources with `NN-` numbered prefixes
- `{topic-folders}/` - Knowledge organized by topic with `00-`, `01-`, `02-` prefixes
- `INDEX.md` - Master file listing (update when structure changes)
- `SKILL.md` - Oracle identity (update when major reorganization)
- `_ingest/` - User drop zone for manual knowledge additions
- `_ingest-auto/` - Oracle autonomous operations workspace

### File Naming Rules

- **Numbered prefixes required**: `00-overview.md`, `01-concept.md`, `02-advanced.md`
- **Lowercase with hyphens**: `quantum-entanglement.md` not `Quantum_Entanglement.md`
- **Source documents**: `NN-source-name.pdf` and `NN-source-name.md` (converted)

### When to Update INDEX.md

Update after:
- Adding new knowledge files
- Reorganizing folders
- Splitting or merging files
- Moving files between folders

### When to Update SKILL.md

Update after:
- Major reorganization (Directory Structure section)
- Adding new topic folders (What This Skill Provides)
- Changing oracle scope (When to Use This Skill)

### Reorganization Operations

**ALL reorganization uses _ingest-auto/ workflow**

**Splitting large files (>600 lines):**
1. Create work folder: `_ingest-auto/split-filename-YYYY-MM-DD/`
2. Read large file, identify sections
3. Create `ingestion.md` plan with PARTS for each new file
4. Move to `_ingest-auto/inprocess/split-filename-YYYY-MM-DD/`
5. Execute plan: create new files with sequential numbers
6. Preserve all citations (adjust paths if needed)
7. Update cross-references in other files
8. Update INDEX.md
9. Delete old file
10. Archive to `_ingest-auto/completed/`
11. Git commit: "Reorganize: Split [file] into N focused files"

**Merging small files:**
1. Create work folder: `_ingest-auto/merge-topic-YYYY-MM-DD/`
2. Create `ingestion.md` plan
3. Move to `_ingest-auto/inprocess/`
4. Combine related small files into logical grouping
5. Preserve all citations
6. Update cross-references
7. Update INDEX.md
8. Delete old files
9. Archive to `_ingest-auto/completed/`
10. Git commit: "Reorganize: Merge N files into [new-file]"

**Moving files between folders:**
1. Create work folder: `_ingest-auto/move-files-YYYY-MM-DD/`
2. Create `ingestion.md` plan
3. Move to `_ingest-auto/inprocess/`
4. Move file to correct folder
5. Renumber to fit folder sequence
6. Update all cross-references TO this file
7. Update relative paths FROM this file
8. Update INDEX.md
9. Archive to `_ingest-auto/completed/`
10. Git commit: "Reorganize: Move [file] to [folder]"

**Creating new topic folders:**
1. Create work folder: `_ingest-auto/new-folder-YYYY-MM-DD/`
2. Create `ingestion.md` plan
3. Move to `_ingest-auto/inprocess/`
4. Create folder when topic has 3+ files
5. Move related files
6. Renumber: `00-overview.md`, `01-first.md`, etc.
7. Update cross-references
8. Update INDEX.md (new section)
9. Update SKILL.md (Directory Structure)
10. Archive to `_ingest-auto/completed/`
11. Git commit: "Reorganize: Create [folder]/ for [topic]"

### Cross-Reference Formats

**Internal oracle links:**
```markdown
See [Related Concept](../other-folder/file.md)
```

**Source citations:**
```markdown
From [Source](../source-documents/NN-filename.md):
- Section title (lines XX-YY)
```

### File Size Guidelines

- Overview files: 100-200 lines
- Concept files: 200-400 lines
- Detailed files: 400-600 lines
- **If > 600 lines**: Consider splitting (use `_ingest-auto/`)

### Git Commits for Reorganization

Format:
```
Reorganize: [What changed]

[Why - brief explanation]

- Specific change 1
- Specific change 2

Files affected: N moved/renamed/split/merged
Processed via: _ingest-auto/

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

---