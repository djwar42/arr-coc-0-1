---
name: oracle-creator
description: Meta-skill for creating domain expert oracle skills from source documents. Use when user wants to create a new oracle (e.g., "create vending-machine-oracle from PDFs"). Guides through document conversion, content organization, SKILL.md generation, and oracle deployment. (project)
---

# Oracle Creator

**A skill to create oracle skills** - Convert source documents into domain expert oracles.

## What This Skill Does

Creates self-contained oracle skills from user-provided source documents:

1. **Accepts** PDFs, markdown files, and other documents
2. **Converts** all to clean markdown format
3. **Organizes** content into numbered documentation structure
4. **Generates** comprehensive SKILL.md and INDEX.md
5. **Deploys** complete oracle to `.claude/skills/{oracle-name}/`

## What Oracle-Creator Does NOT Do

**❌ Oracle-creator does NOT handle knowledge updates for existing oracles**

Oracles are autonomous and update themselves.

Oracle-creator is ONLY for:
- ✅ Creating NEW oracles from scratch
- ✅ Converting source documents → initial oracle structure
- ✅ Generating initial SKILL.md and INDEX.md
- ✅ Setting up folder structure for new oracles

**Once an oracle is created, it maintains itself.**

### Oracle Knowledge Expansion (All Oracles)

Every oracle created by oracle-creator includes:
- `_ingest/` folder for manual knowledge addition
- `_ingest-auto/` folder for autonomous expansion
- Oracle Knowledge Expansion protocol in SKILL.md
- Ability to create ingestion.md plans and launch oracle-knowledge-runners
- Autonomous execution and self-organization

**Oracles do NOT call oracle-creator for updates.**

## When to Use This Skill

User says something like:
- "Create an oracle called **ancient-greek-philosophy-oracle** using these PDFs"
- "Make a **quantum-physics-oracle** from this documentation folder"
- "Build an **astronomy-oracle** expert skill from these papers"

## Oracle Concept

**What is an Oracle?**
- Domain expert embedded in Claude Code
- Self-contained knowledge base
- Can be person-focused (john-vervaeke-oracle) or topic-focused (car-engine-oracle)
- Includes all source materials converted to markdown
- Organized with numbered prefixes for clear navigation

**Oracle Characteristics:**
- ✅ Self-contained (no external dependencies)
- ✅ Preserves original sources (PDFs, etc.) + markdown conversions
- ✅ Comprehensive SKILL.md entry point
- ✅ Organized topic folders extracted from sources
- ✅ Cross-referenced with source document citations
- ✅ All extracts reference back to source-documents/
- ✅ **Dynamic learning** - Can expand knowledge on-the-fly using Bright Data

## Quick Start

### User Workflow

```bash
# 1. User provides source directory and specialization
"Create ancient-greek-philosophy-oracle from PDFs in ~/philosophy-texts/
 This oracle knows Plato, Aristotle, Socratic method, and Greek metaphysics"

# 2. Oracle-creator processes
→ Converts PDFs to markdown
→ Organizes into numbered structure
→ Generates SKILL.md with navigation
→ Creates INDEX.md with cross-references
→ Deploys to .claude/skills/ancient-greek-philosophy-oracle/

# 3. Oracle ready to use
"Explain Plato's theory of Forms" → ancient-greek-philosophy-oracle activates
```

## Oracle Creation Process

### Phase 1: Gather Requirements

**What to Ask User:**
1. **Oracle name** - e.g., "ancient-greek-philosophy-oracle" (lowercase, hyphens)
2. **Source directory** - Path to PDFs/markdown files
3. **Specialization** - What does this oracle know? (for SKILL.md description)
4. **Learning mode** - Seeking (can expand dynamically) or Static (fixed)?
   - **Default: Seeking** (oracle can grow over time)
   - **Static** if: historical period, versioned codebase, or user wants "unchanging"
5. **Additional sources** - Any other directories or files to include?

### Phase 2: Preserve and Convert Documents

**CRITICAL: Keep originals + create markdown versions**

**For each source document:**
1. **Copy original** to `source-documents/` with numbered prefix
2. **Convert to markdown** if not already markdown
3. **Keep both** original and markdown versions
4. **Deduplicate** - Remove duplicate MD files (same content, keep first)

**Output Structure:**
```
.claude/skills/{oracle-name}/
├── source-documents/         # ALL originals preserved + conversions
│   ├── 00-republic.pdf       # Original PDF (preserved)
│   ├── 00-republic.md        # Converted markdown (for reading)
│   ├── 01-symposium.pdf      # Original PDF (preserved)
│   ├── 01-symposium.md       # Converted markdown (for reading)
│   └── 02-ethics.md          # Already markdown (copied)
├── _ingest/                  # User drops new documents here
│   └── README.md             # Instructions for user
└── _ingest-auto/             # Temporary folder for auto-downloaded PDFs
                              # (cleaned after processing)
```

**Why both formats?**
- Originals for safe keeping and verification
- Markdown for oracle to read and extract from

**Ingestion Folders (REQUIRED on oracle creation):**

**_ingest/** - User-managed manual ingestion
- User drops new PDFs/markdown here for manual updates
- README.md explains usage
- User instructs oracle-creator to process
- Kept clean after ingestion

**_ingest-auto/** - Automated ingestion (dynamic learning)
- Web-downloaded PDFs saved here temporarily
- Analyzed and converted to markdown
- Files moved to proper locations in oracle structure
- **ALWAYS cleaned after processing** (should stay empty)

**Why underscore prefix?** Makes these special folders visually distinct and sort first.

These folders MUST be created during oracle deployment (Phase 6).

### Phase 3: Read Sources and Extract Topics

**CRITICAL: Read all markdown sources first, then organize**

**Step 3.1: Read All Source Documents**
- Read through all `source-documents/*.md` files
- Identify major themes, concepts, and topics
- Note relationships and groupings
- Plan logical topic organization

**Step 3.2: Codebase Commenting (If Present)**

**If source directory contains full codebases:**

1. **Copy codebase wholesale** to `source-codebases/{number}-{name}/`
2. **Add Claude's code comments** to major files (3-5 core files)
3. **Create INDEX.md inside each codebase** (UPPERCASE, inside the codebase folder)
4. **Create codebase overview** in `codebase/` folder

```python
# Copy entire codebase
shutil.copytree(
    source_codebase_dir,
    f"source-codebases/00-{codebase_name}"
)

# Add Claude's code comments to core files
# Top-level overview + major files only
core_files = identify_core_files(codebase)  # main.py, app.py, core modules
for file in core_files:
    add_claudes_code_comments(file)

# Create INDEX.md inside the codebase folder
create_codebase_index(f"source-codebases/00-{codebase_name}/INDEX.md")
```

**Codebase Structure (CRITICAL - No External Dependencies):**

```
source-codebases/
├── 00-nanoGPT/                    # Complete codebase copied wholesale
│   ├── INDEX.md                   # UPPERCASE, overview of this codebase
│   ├── train.py                   # Original files (may have Claude comments)
│   ├── model.py
│   ├── config/
│   └── ...
├── 01-llama2.c/                   # Another complete codebase
│   ├── INDEX.md                   # UPPERCASE, overview of this codebase
│   ├── run.c
│   └── ...
└── 02-overview.md                 # Optional: Overview markdown for a codebase
```

**INDEX.md Inside Each Codebase:**
- **Location**: `source-codebases/{number}-{name}/INDEX.md`
- **Naming**: UPPERCASE `INDEX.md`
- **Purpose**: Overview of this specific codebase
- **Content**: Architecture, key files, file listing, how to navigate
- **Created during**: Claude's code comments exploration

**Guidelines for codebase organization:**
- ✅ **DO**: Copy entire codebase as-is to source-codebases/
- ✅ **DO**: Create simple INDEX.md (UPPERCASE) with overview and folder structure
- ✅ **DO**: Include any required dependencies/submodules with the codebase
- ✅ **DO**: "Take everything with you" - no external dependencies
- ❌ **DON'T**: Add Claude's code comments by default (only on user request)
- ❌ **DON'T**: Go overboard with initial documentation
- ❌ **DON'T**: Link to external codebases that might disappear
- 🔄 **LATER**: User can request deeper exploration using 3-phase process

**Default INDEX.md Contents** (minimal overview):
- Brief description of codebase
- Directory structure tree
- Key files and their purposes
- Links to README and documentation folders
- Simple, straightforward, no deep analysis yet

**Deep Codebase Exploration (User-Requested Only):**

When user specifically asks for deep codebase analysis, use the **3-Phase Documentation Process**:

**Phase 1: High-Level Overview**
- Explore repository structure and identify major components
- Document architecture and key files
- Add high-level understanding to INDEX.md
- Update codebase/ folder with overview document

**Phase 2: Deep Dive with Claude's Code Comments**
- Add comprehensive `<claudes_code_comments>` to 3-5 major files
- Document function purposes, algorithms, optimizations
- Explain design patterns and code flows
- Include mathematical foundations where relevant

**Phase 3: Verification & Completion**
- Review all Phase 2 work for accuracy
- Verify code flow descriptions match execution
- Add comments to remaining important files
- Cross-reference related components

**Example 3-Phase Workflow**: See `.claude/skills/andrej-karpathy-oracle/source-codebases/CODEBASE-ADDITIONS.md` for complete methodology with examples of 12 codebases documented systematically.

**When to Use 3-Phase Process:**
- User explicitly requests "deep dive into codebase"
- User wants "comprehensive Claude's code comments"
- User asks for "detailed documentation of source code"
- Oracle needs deep understanding for answering complex questions

**Default Behavior**: Simple INDEX.md only, no deep commenting unless requested.

**Create codebase documentation folder:**
```
codebase/
├── 00-overview.md              # High-level architecture of ALL codebases
├── 01-nanoGPT-analysis.md      # Analysis of nanoGPT codebase
├── 02-llama2-analysis.md       # Analysis of llama2.c codebase
└── 03-file-index.md            # Complete file listing across ALL codebases
```

**"Take Everything With You" Policy:**

Oracles should be **self-contained with no external dependencies**:

✅ **DO bring with you:**
- Complete codebases → `source-codebases/`
- PDFs and papers → `source-documents/`
- Any submodules or dependencies required by codebases
- Images, diagrams, supplementary materials
- Related datasets (if reasonable size)

❌ **DON'T rely on external links when avoidable:**
- GitHub repos that might change or disappear
- External documentation that might move
- Web resources that aren't permanently archived

**When external references are acceptable:**
- Permanent scholarly resources (Stanford Encyclopedia, arXiv)
- Official documentation for widely-used tools
- Citations to published papers (with DOI)
- Web research supplements (with date accessed)

**Principle**: If you can easily "take it with you", do so. Oracles should work even if the internet disappeared tomorrow (except for dynamic learning updates).

**Step 3.3: Web Research Enrichment (Judicious)**

**Use Bright Data to supplement source material:**

```python
# For significant concepts that need more context
mcp__bright-data__search_engine(
    query="Plato theory of Forms scholarly articles"
)

# Scrape relevant articles
mcp__bright-data__scrape_as_markdown(
    url="https://plato.stanford.edu/entries/plato-forms/"
)
```

**Guidelines:**
- ✅ **DO**: Fill gaps in understanding of key concepts
- ✅ **DO**: Add scholarly context from reputable sources
- ✅ **DO**: Link to web sources with full citations
- ❌ **DON'T**: Go overboard - supplement, don't replace sources
- ❌ **DON'T**: Use unreliable sources
- ❌ **DON'T**: Add tangential information

**Cite web sources:**
```markdown
## Additional Research

From web research (Stanford Encyclopedia of Philosophy):
- [Plato's Theory of Forms](https://plato.stanford.edu/entries/plato-forms/)
- Key insight: {what you learned}
```

**Step 3.4: Create Topic-Based Organization**

Extract and organize content from sources (including web research + codebases):

```
.claude/skills/{oracle-name}/
├── SKILL.md                       # Main entry point (loaded on invocation)
├── source-documents/              # Originals + conversions
│   ├── 00-republic.pdf            # Original preserved
│   ├── 00-republic.md             # Converted for reading
│   ├── 01-symposium.pdf
│   ├── 01-symposium.md
│   └── ...
├── source-codebases/              # Full codebases (if present)
│   └── 00-example-project/        # Copied wholesale
│       ├── src/
│       ├── tests/
│       └── ...
├── codebase/                      # Codebase documentation (if present)
│   ├── INDEX.md                   # ⭐ Subfolder index (file list + keywords)
│   ├── 00-overview.md             # Architecture overview
│   ├── 01-core-module.md          # Major file commentary
│   └── 02-file-index.md           # Complete listing
├── {topic-folder1}/               # Extracted theme group
│   ├── INDEX.md                   # ⭐ Subfolder index (file list + keywords)
│   ├── 00-overview.md             # References source-documents/
│   ├── 01-concept-a.md            # References source-documents/
│   └── 02-concept-b.md            # References source-documents/
├── {topic-folder2}/               # Another theme group
│   ├── INDEX.md                   # ⭐ Subfolder index
│   ├── 00-introduction.md
│   └── 01-advanced.md
└── {topic-folder3}/
    ├── INDEX.md                   # ⭐ Subfolder index
    └── 00-summary.md
```

**⚠️ NO ROOT INDEX.md** - Each topic folder has its own INDEX.md instead!

### Distributed Index Architecture

**Each topic subfolder has its own INDEX.md - no root-level master index!**

**Flow:**
```
User question → SKILL.md routing → Read folder's INDEX.md → Read specific file
```

**Subfolder INDEX.md Format:**
```markdown
# [Folder Name] - Index

**[Brief description of this topic area]**

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-overview.md` | Introduction and navigation | basics, getting started |
| `01-core-concepts.md` | Fundamental concepts | architecture, fundamentals |

## Cross-References

Related folders: `../other-topic/`, `../related-area/`
```

**Why Distributed Indexes:**
- ✅ **Modular**: Each topic is self-contained
- ✅ **Scalable**: Add new topics without bloating root
- ✅ **Focused**: Only load what you need
- ✅ **Maintainable**: Update one folder without touching others

**Each extracted file must reference sources:**
```markdown
## Primary Sources

From [Republic](../source-documents/00-republic.md):
- Book V, discussion of Forms (lines 450-520)

From [Symposium](../source-documents/01-symposium.md):
- Diotima's speech on love (lines 200-350)

## Additional Research (Web Sources)

Web research to supplement understanding:
- [Stanford Encyclopedia - Plato's Forms](https://plato.stanford.edu/entries/plato-forms/)
  - Added context on historical interpretations
- [Internet Encyclopedia of Philosophy](https://iep.utm.edu/platonic-form/)
  - Contemporary scholarly perspectives
```

**Numbering Convention:**
- `00-` for overview/intro
- `01-`, `02-`, `03-` for sequential topics
- Lowercase, hyphens in names

### Phase 4: Generate SKILL.md

**Required Sections:**
1. **Metadata** (YAML frontmatter)
   - name: {oracle-name}
   - description: What it knows + when to use (max 1024 chars)

2. **What This Skill Provides**
   - Brief summary of oracle capabilities

3. **Directory Structure**
   - Tree view of all files

4. **When to Use This Skill**
   - Example questions → file mappings

5. **Detailed File Descriptions**
   - Brief summary of each file

6. **Quick Navigation**
   - Topic → file mappings

7. **Oracle Knowledge Expansion** (MANDATORY - SLIM SUMMARY + FULL INSTRUCTIONS FILE)

   **Structure for every oracle:**
   ```
   {oracle-name}/
   ├── SKILL.md                          # Slim summary (~30 lines)
   ├── oracle-knowledge-expansion/       # Full instructions folder
   │   └── full-expansion-instructions.md  # Complete 900+ line guide
   └── ...
   ```

   **In SKILL.md (slim summary):**
   - Strong trigger phrase detection ("expand knowledge", "research and add", etc.)
   - **⚠️ MUST READ pointer**: `oracle-knowledge-expansion/full-expansion-instructions.md`
   - Brief 20-line summary of what Knowledge Expansion is
   - Synchronization warning (update all oracles when modifying the system)

   **In oracle-knowledge-expansion/full-expansion-instructions.md (full guide):**
   - Complete parallel execution workflow
   - oracle-knowledge-runner sub-agent responsibilities
   - KNOWLEDGE DROP format
   - When to use vs when to edit directly
   - Self-organization capabilities
   - Detailed examples

   **Why this split:**
   - SKILL.md stays readable (~1,600 lines instead of 2,500)
   - Full instructions always available when needed
   - Easy to update (single file for full instructions)
   - Synchronization: Copy `full-expansion-instructions.md` to all oracles

10. **Oracle Self-Check** (RECOMMENDED)
    - Steps for oracle to verify its own structure health
    - Folder structure checks, file naming verification
    - Can be customized for special oracle structures

**See**: `guides/01-skill-md-template.md` for full template

### Phase 5: Generate Subfolder INDEX.md Files

**⚠️ NO ROOT INDEX.md** - Create INDEX.md inside each topic folder instead!

**For each topic folder, create INDEX.md with:**
1. **Description** - What this folder covers
2. **Files Table** - All files with descriptions and keywords
3. **Quick Start** - Where to begin
4. **Cross-References** - Related folders

**Example subfolder INDEX.md:**
```markdown
# Concepts - Index

**Core philosophical concepts extracted from source documents**

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-overview.md` | Introduction to key concepts | basics, overview |
| `01-forms.md` | Theory of Forms | metaphysics, Forms |
| `02-ethics.md` | Ethical framework | ethics, virtue |

## Cross-References

Related: `../applications/`, `../dialogues/`
```

### Phase 6: Deploy Oracle

**Deployment Checklist:**
- ✅ All source docs converted to MD
- ✅ SKILL.md has proper metadata (includes slim Knowledge Expansion summary)
- ✅ Each topic folder has INDEX.md (file list + keywords)
- ✅ Numbered prefixes on all docs
- ✅ Directory structure follows convention
- ✅ Oracle name is lowercase-hyphenated
- ✅ **_ingest/ folder created with README.md**
- ✅ **_ingest-auto/ folder created (empty)**
- ✅ **oracle-knowledge-expansion/ folder with full-expansion-instructions.md**

**Create Required Folders:**
```bash
# Create _ingest folder with README
mkdir -p .claude/skills/{oracle-name}/_ingest
cat > .claude/skills/{oracle-name}/_ingest/README.md << 'EOF'
Place new documents (PDFs, markdown, etc.) here, then ask {oracle-name} to ingest the knowledge.
EOF

# Create _ingest-auto folder with README (working directory)
mkdir -p .claude/skills/{oracle-name}/_ingest-auto
cat > .claude/skills/{oracle-name}/_ingest-auto/README.md << 'EOF'
# Auto-Ingestion Working Directory

Working directory for oracle knowledge expansion. Cleaned after processing.
EOF

# Create oracle-knowledge-expansion folder with full instructions
mkdir -p .claude/skills/{oracle-name}/oracle-knowledge-expansion

# Copy full instructions from karpathy-deep-oracle (the reference implementation)
cp .claude/skills/karpathy-deep-oracle/oracle-knowledge-expansion/full-expansion-instructions.md \
   .claude/skills/{oracle-name}/oracle-knowledge-expansion/
```

**Test Oracle:**
```bash
# Restart Claude Code to load new skill
# Ask a question in the oracle's domain
# Verify oracle activates and provides knowledge
```

## Oracle Naming Rules

**Good Oracle Names:**
- `ancient-greek-philosophy-oracle` ✅
- `quantum-physics-oracle` ✅
- `astronomy-oracle` ✅
- `john-vervaeke-oracle` ✅ (person-focused)

**Bad Oracle Names:**
- `GreekPhilosophyOracle` ❌ (camelCase)
- `greek_philosophy_oracle` ❌ (underscores)
- `gp-oracle` ❌ (unclear abbreviation)

## Example Oracles

**Existing Oracles in This Project:**

1. **john-vervaeke-oracle**
   - Person-focused
   - Academic papers + concepts
   - ~4,000 lines of RR knowledge

2. **ovis-2-5-oracle**
   - Topic-focused (ML model)
   - Architecture + training + code
   - ~42 files, fully indexed

3. **deepseek-ocr-oracle**
   - Topic-focused (ML model)
   - Vision-language model expertise

## File Organization Best Practices

**Use Numbered Prefixes:**
```
concepts/
├── 00-overview.md       # Always start with 00
├── 01-fundamentals.md
├── 02-advanced.md
└── 03-expert.md
```

**Why Numbered Prefixes?**
- Ensures consistent ordering
- Makes references predictable
- Easy to insert new topics
- Clear progression (intro → advanced)

**Cross-Reference Format:**
```markdown
[concept overview](concepts/00-overview.md)
[training guide](guides/01-training.md)
```

## Common Patterns

### Pattern 1: Academic Oracle (Person-Focused)

**Example**: `john-vervaeke-oracle`

**Structure:**
- `papers/` - Academic publications
- `concepts/` - Key ideas
- `{Name}-Application-Guide.md` - Project-specific

**When to Use**: Oracle about a person's work

### Pattern 2: Technical Oracle (Topic-Focused)

**Example**: `ovis-2-5-oracle`

**Structure:**
- `architecture/` - System design
- `codebase/` - Code documentation
- `usage/` - Practical guides
- `examples/` - Working code

**When to Use**: Oracle about a technology/tool

### Pattern 3: Domain Oracle (Knowledge-Focused)

**Example**: Hypothetical `quantum-physics-oracle`

**Structure:**
- `fundamentals/` - Core concepts
- `phenomena/` - Specific topics
- `experiments/` - Key experiments
- `applications/` - Real-world uses

**When to Use**: Oracle about a field of knowledge

## Content Organization Strategies

### Strategy: Read Sources → Extract Topic Folders

**Always follow this pattern:**

1. **Preserve originals in source-documents/**
   ```
   source-documents/
   ├── 00-paper1.pdf          # Original
   ├── 00-paper1.md           # Converted
   ├── 01-paper2.pdf
   ├── 01-paper2.md
   └── 02-notes.md            # Already markdown
   ```

2. **Read all markdown sources thoroughly**
   - Identify major themes
   - Note concept groupings
   - Plan topic organization

3. **Create topic folders with extracted content**
   ```
   {topic-folder1}/           # First major theme
   ├── 00-overview.md         # Extract + cite sources
   ├── 01-concept-a.md        # Extract + cite sources
   └── 02-concept-b.md        # Extract + cite sources

   {topic-folder2}/           # Second major theme
   ├── 00-introduction.md
   └── 01-details.md
   ```

4. **Every extracted file cites source documents**
   ```markdown
   ## Primary Sources

   From [Paper 1](../source-documents/00-paper1.md):
   - Section 2.1 on theory (lines 150-200)

   From [Paper 2](../source-documents/01-paper2.md):
   - Experimental validation (lines 300-450)
   ```

**Result**: Self-contained oracle with originals preserved and content logically organized with full traceability

## Tools and Resources

**This Skill Provides:**
- `guides/01-skill-md-template.md` - SKILL.md template
- `guides/02-subfolder-index-template.md` - Subfolder INDEX.md template
- `guides/03-creation-workflow.md` - Step-by-step workflow
- `templates/codebase-overview-template.md` - Codebase overview template
- `templates/codebase-file-analysis-template.md` - File analysis template
- `templates/oracle-dynamic-learning.md` - **Dynamic learning guidelines**
- `examples/00-ancient-greek-philosophy-oracle.md` - Complete example

**External Tools:**

**Bright Data MCP** (for web research enrichment):
- `mcp__bright-data__search_engine` - Search Google/Bing for scholarly articles
- `mcp__bright-data__scrape_as_markdown` - Convert web pages to markdown
- `mcp__bright-data__extract` - Extract structured data from complex pages
- `mcp__bright-data__search_engine_batch` - Multiple searches simultaneously

**PDF Tools:**
- PDF parsers (user-provided or built-in)

## Quality Checklist

Before deploying oracle:

**Structure:**
- [ ] SKILL.md has proper YAML metadata
- [ ] Each topic folder has INDEX.md with file list + keywords
- [ ] All docs use numbered prefixes
- [ ] Directory structure is clear
- [ ] Oracle name follows naming rules

**Content:**
- [ ] All source docs converted to MD
- [ ] No external file dependencies
- [ ] Cross-references use full paths
- [ ] File descriptions are accurate
- [ ] Navigation paths are clear

**Usability:**
- [ ] SKILL.md description explains WHEN to use
- [ ] Quick start section exists
- [ ] Example questions → file mappings
- [ ] Clear topic organization

## Workflow Summary

1. **User provides**: Source dir + oracle specialization
2. **Convert**: All docs to markdown
3. **Organize**: Numbered structure with categories
4. **Generate**: SKILL.md + subfolder INDEX.md files
5. **Deploy**: To `.claude/skills/{oracle-name}/`
6. **Test**: Restart Claude, ask questions

## Advanced Features

### Multi-Source Oracles

Combine multiple directories:
```bash
"Create ml-papers-oracle from:
 - ~/papers/vision/ (computer vision papers)
 - ~/papers/nlp/ (NLP papers)
 - ~/papers/rl/ (reinforcement learning)"
```

### Incremental Oracle Updates

Add new sources to existing oracle:
```bash
"Add these 3 new papers to quantum-physics-oracle:
 - bell-theorem-update.pdf
 - entanglement-experiments-2024.pdf
 - quantum-computing-review.pdf"
```

### Oracle Specialization Levels

**Narrow Oracle**: Single topic expert
- Example: `foveal-vision-oracle` (just human fovea)

**Broad Oracle**: Field expert
- Example: `computer-vision-oracle` (entire CV field)

**Meta Oracle**: Cross-domain expert
- Example: `cognitive-science-oracle` (multiple disciplines)

## Integration with Claude Code

**Oracles are Claude Code skills:**
- Auto-discovered at startup
- Activated by relevant questions
- Access via `Skill(oracle-name)`

**Usage Pattern:**
```
User: "Explain quantum entanglement"
Claude: [Activates quantum-physics-oracle]
        [Reads relevant concept docs]
        [Provides expert answer]
```

## File Reference

All oracle-creator documentation:

- **This file** (`SKILL.md`) - Main guide
- `INDEX.md` - Master index
- `guides/00-overview.md` - Detailed overview
- `guides/01-skill-md-template.md` - SKILL.md template
- `guides/02-index-template.md` - INDEX.md template
- `guides/03-creation-workflow.md` - Step-by-step process
- `templates/` - Ready-to-use templates
- `examples/00-ancient-greek-philosophy-oracle.md` - Complete example

---

**Last Updated**: 2025-10-28
**Status**: Complete meta-skill for oracle generation
**Version**: 1.0
