# Oracle Creator - Master Index

**Meta-skill for creating domain expert oracle skills from source documents**

## Quick Navigation

| Topic | File | Description |
|-------|------|-------------|
| **Main Entry** | [SKILL.md](SKILL.md) | Complete oracle-creator guide |
| **Overview** | [guides/00-overview.md](guides/00-overview.md) | What oracles are and why |
| **SKILL.md Template** | [guides/01-skill-md-template.md](guides/01-skill-md-template.md) | Template for oracle SKILL.md |
| **INDEX.md Template** | [guides/02-index-template.md](guides/02-index-template.md) | Template for oracle INDEX.md |
| **Creation Workflow** | [guides/03-creation-workflow.md](guides/03-creation-workflow.md) | Step-by-step process |
| **Example Oracle** | [examples/00-ancient-greek-philosophy-oracle.md](examples/00-ancient-greek-philosophy-oracle.md) | Complete example |

## Document Structure

```
.claude/skills/oracle-creator/
├── SKILL.md                           # Main entry point
├── INDEX.md                           # This file
├── guides/
│   ├── 00-overview.md                 # Oracle concept and philosophy
│   ├── 01-skill-md-template.md        # SKILL.md generation template
│   ├── 02-index-template.md           # INDEX.md generation template
│   └── 03-creation-workflow.md        # Step-by-step creation process
├── templates/
│   ├── skill-template.md              # Ready-to-use SKILL.md template
│   ├── index-template.md              # Ready-to-use INDEX.md template
│   └── concept-template.md            # Template for concept docs
└── examples/
    └── 00-ancient-greek-philosophy-oracle.md   # Complete oracle example
```

## How to Use Oracle Creator

### 1. User Request Pattern

```
"Create {oracle-name}-oracle from {source-directory}
 This oracle knows {specialization}"
```

**Examples:**
- "Create ancient-greek-philosophy-oracle from ~/texts/greek-philosophy/. Knows Plato, Aristotle, Socratic method"
- "Create quantum-physics-oracle from ~/papers/quantum/. Knows quantum mechanics, entanglement, Bell's theorem"
- "Create ml-model-oracle from ~/ml-project/ (includes full codebase). Knows PyTorch training pipeline and model architecture"

### 2. Creation Process

**Step 1: Requirements Gathering**
- Ask user for oracle name
- Ask user for source directory/files
- Ask user for specialization description

**Step 2: Document Conversion**
- Convert PDFs → markdown
- Copy existing markdown files
- **Copy codebases wholesale** (if present)
- Organize with numbered prefixes

**Step 3: Content Organization**
- **Add Claude's code comments** to codebases (top-level + core files only)
- **Web research enrichment** (judicious - fill gaps with reputable sources)
- Create category directories (concepts/, guides/, codebase/, etc.)
- Number files with `00-`, `01-`, `02-` prefixes
- Extract key concepts from sources

**Step 4: Generate Metadata**
- Create SKILL.md from template
- Create INDEX.md with cross-references
- Add navigation guides

**Step 5: Deploy and Test**
- Place in `.claude/skills/{oracle-name}/`
- Verify structure
- Commit to git

### 3. Verification

Run through checklist:
- [ ] SKILL.md has YAML metadata (name + description)
- [ ] INDEX.md cross-references all files
- [ ] All docs have numbered prefixes
- [ ] Source docs converted to markdown
- [ ] Oracle is self-contained
- [ ] Directory structure follows convention

## Common Oracle Patterns

### Academic Oracle (Person-Focused)

**Example**: `john-vervaeke-oracle`

```
├── SKILL.md
├── INDEX.md
├── papers/
│   ├── 00-primary-paper.md
│   ├── 01-secondary-paper.md
│   └── 02-thesis.md
├── concepts/
│   ├── 00-core-theory.md
│   └── 01-applications.md
└── {Name}-Application-Guide.md
```

**When**: Oracle about a person's work/theories

### Technical Oracle (Topic-Focused)

**Example**: `ovis-2-5-oracle`

```
├── SKILL.md
├── INDEX.md
├── architecture/
│   ├── 00-overview.md
│   └── 01-components.md
├── codebase/
│   ├── 00-structure.md
│   └── 01-implementation.md
├── usage/
│   └── 00-quickstart.md
└── examples/
    └── 00-basic.md
```

**When**: Oracle about a technology/tool/system

### Domain Oracle (Knowledge-Focused)

**Example**: `quantum-physics-oracle`

```
├── SKILL.md
├── INDEX.md
├── fundamentals/
│   ├── 00-overview.md
│   └── 01-principles.md
├── phenomena/
│   ├── 00-entanglement.md
│   └── 01-superposition.md
└── experiments/
    └── 00-bell-test.md
```

**When**: Oracle about a field of knowledge

## File Organization Rules

### Numbering Convention

**Always use numbered prefixes:**
- `00-` for overviews and introductions
- `01-`, `02-`, `03-` for sequential topics
- Numbers enforce logical ordering
- Makes cross-references predictable

**Example:**
```
concepts/
├── 00-overview.md          # Start here
├── 01-fundamentals.md      # Build on overview
├── 02-intermediate.md      # Build on fundamentals
└── 03-advanced.md          # Build on intermediate
```

### Naming Rules

**Files**: `{number}-{descriptive-name}.md`
- Lowercase only
- Hyphens (not underscores)
- Descriptive (not abbreviated)

**Directories**: `{category-name}/`
- Plural preferred (`concepts/` not `concept/`)
- Lowercase, hyphens

**Oracles**: `{topic}-oracle`
- Lowercase, hyphens
- Always ends with `-oracle`
- Clear, not abbreviated

### Cross-References

**Always use full relative paths:**
```markdown
[core concept](concepts/00-overview.md)
[training guide](guides/01-training.md)
[example code](examples/00-basic.md)
```

## Topic Index

### Oracle Creation

| Topic | File |
|-------|------|
| What are oracles? | [guides/00-overview.md](guides/00-overview.md) |
| Creation workflow | [guides/03-creation-workflow.md](guides/03-creation-workflow.md) |
| SKILL.md template | [guides/01-skill-md-template.md](guides/01-skill-md-template.md) |
| INDEX.md template | [guides/02-index-template.md](guides/02-index-template.md) |
| Example oracle | [examples/00-ancient-greek-philosophy-oracle.md](examples/00-ancient-greek-philosophy-oracle.md) |

### Templates

| Template | File |
|----------|------|
| SKILL.md | [templates/skill-template.md](templates/skill-template.md) |
| INDEX.md | [templates/index-template.md](templates/index-template.md) |
| Concept doc | [templates/concept-template.md](templates/concept-template.md) |
| Codebase overview | [templates/codebase-overview-template.md](templates/codebase-overview-template.md) |
| Codebase file analysis | [templates/codebase-file-analysis-template.md](templates/codebase-file-analysis-template.md) |

### Examples

| Example | File |
|---------|------|
| Ancient Greek Philosophy oracle | [examples/00-ancient-greek-philosophy-oracle.md](examples/00-ancient-greek-philosophy-oracle.md) |

## Quick Reference

### Oracle Name Rules

✅ **Good**:
- `ancient-greek-philosophy-oracle`
- `quantum-physics-oracle`
- `astronomy-oracle`

❌ **Bad**:
- `GreekPhilosophyOracle` (camelCase)
- `greek_philosophy_oracle` (underscores)
- `gp-oracle` (unclear abbreviation)

### Required Files

Every oracle must have:
1. `SKILL.md` - Entry point with YAML metadata
2. `INDEX.md` - Master index (optional but recommended)
3. Source documents - All converted to markdown
4. Numbered prefixes - On all documentation files

### SKILL.md Metadata

```yaml
---
name: {oracle-name}
description: {What it knows} + {When to use} (max 1024 chars)
---
```

**Description must include:**
1. What the oracle knows (domain expertise)
2. When to use it (trigger keywords)
3. Concise but comprehensive

## Workflow Cheat Sheet

```
1. User provides: name + sources + specialization
2. Convert: PDFs/docs → markdown
3. Organize: Numbered structure
4. Generate: SKILL.md + INDEX.md
5. Deploy: .claude/skills/{oracle-name}/
6. Test: Restart Claude, verify activation
```

## Example Usage

### Creating an Oracle

```
User: "Create ancient-greek-philosophy-oracle using PDFs in ~/philosophy-texts/
       This oracle knows Plato, Aristotle, Socratic method, and Greek metaphysics"

Oracle Creator:
1. Converts 5 PDFs to markdown
2. Organizes into:
   - source-documents/00-republic.md ... 04-metaphysics.md
   - concepts/00-overview.md, 01-theory-of-forms.md
   - guides/00-understanding-plato.md
3. Generates SKILL.md with metadata
4. Generates INDEX.md with cross-references
5. Deploys to .claude/skills/ancient-greek-philosophy-oracle/

Result: "ancient-greek-philosophy-oracle created! 5 source docs, 7 organized files.
         Ask questions like 'What is Plato's theory of Forms?' to test."
```

### Using a Created Oracle

```
User: "What is Plato's theory of Forms?"

Claude: [Activates ancient-greek-philosophy-oracle]
        [Reads concepts/01-theory-of-forms.md]
        [Reads source-documents/00-republic.md]

        "According to Plato's Republic, the Theory of Forms posits that...
        [detailed answer from oracle knowledge base]"
```

## Tips and Best Practices

### Content Organization

**Start Broad, Get Specific:**
```
00-overview.md        # High-level
01-fundamentals.md    # Core concepts
02-intermediate.md    # Build complexity
03-advanced.md        # Expert level
```

**Separate Concerns:**
- `source-documents/` - Original materials
- `concepts/` - Extracted ideas
- `guides/` - How-to instructions
- `examples/` - Practical code/demos

**Cross-Reference Heavily:**
- Link concepts to source documents
- Link guides to concepts
- Link examples to guides
- Create navigation web

### Quality Checks

**Before Deployment:**
1. All source docs are self-contained (no external links to local files)
2. SKILL.md description is under 1024 chars
3. INDEX.md covers all files
4. Numbered prefixes on all docs
5. Oracle name follows naming rules

**After Deployment:**
1. Restart Claude Code
2. Ask domain question
3. Verify oracle activates
4. Check answer quality
5. Update if needed

### Common Pitfalls

❌ **Don't**:
- Use absolute paths in cross-references
- Skip numbered prefixes
- Abbreviate oracle names
- Reference external files
- Forget YAML metadata

✅ **Do**:
- Use relative paths
- Number all docs
- Use descriptive names
- Include all sources
- Test before finalizing

## Related Skills

- `platonic-dialogue-method` - Creates Socratic dialogues
- `oracle-overview-method` - Adds expert commentary to docs
- `john-vervaeke-oracle` - Example academic oracle
- `ovis-2-5-oracle` - Example technical oracle

---

**Last Updated**: 2025-10-28
**Files**: 5 guides, 3 templates, 1 example
**Status**: Complete oracle-creator meta-skill
**Version**: 1.0
