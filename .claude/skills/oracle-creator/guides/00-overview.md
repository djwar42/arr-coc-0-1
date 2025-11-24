# Oracle Creator Overview

## What Are Oracles?

**Oracles** are domain expert skills embedded in Claude Code. They provide:

- **Deep knowledge** on specific topics
- **Self-contained** documentation with all sources
- **Structured navigation** for efficient knowledge retrieval
- **Context-aware activation** when relevant questions arise

## Oracle Philosophy

### Canonical Knowledge

Oracles embody the classical concept - sources of profound wisdom:
- **Ancient Greek Philosophy** - Socrates, Plato, Aristotle
- **Quantum Physics** - Fundamental nature of reality
- **Astronomy** - Celestial knowledge
- **Cognitive Science** - Understanding minds
- **Person-Focused** - Deep expertise on individual's work (e.g., john-vervaeke-oracle)

### Self-Contained Expertise

Every oracle is complete and independent:
- All source documents converted to markdown
- No external dependencies
- Organized for quick navigation
- Cross-referenced internally

## Oracle Types

### 1. Academic Oracle (Person-Focused)

**Example**: `john-vervaeke-oracle`

Knowledge about a person's work, theories, and contributions.

**Structure:**
```
john-vervaeke-oracle/
├── SKILL.md
├── INDEX.md
├── papers/
│   ├── 00-primary-paper.md
│   └── 01-secondary-paper.md
├── concepts/
│   ├── 00-relevance-realization.md
│   └── 01-transjective.md
└── Application-Guide.md
```

**When to Use**: Oracle about philosopher, scientist, researcher

### 2. Technical Oracle (Topic-Focused)

**Example**: `ovis-2-5-oracle`

Knowledge about a technology, tool, or system.

**Structure:**
```
ovis-2-5-oracle/
├── SKILL.md
├── INDEX.md
├── architecture/
│   ├── 00-overview.md
│   └── 01-components.md
├── codebase/
│   └── 00-structure.md
├── usage/
│   └── 00-quickstart.md
└── examples/
    └── 00-basic.md
```

**When to Use**: Oracle about ML model, framework, software

### 3. Domain Oracle (Knowledge-Focused)

**Example**: `ancient-greek-philosophy-oracle`

Knowledge about a field, discipline, or domain.

**Structure:**
```
ancient-greek-philosophy-oracle/
├── SKILL.md
├── INDEX.md
├── philosophers/
│   ├── 00-plato.md
│   └── 01-aristotle.md
├── concepts/
│   ├── 00-theory-of-forms.md
│   └── 01-metaphysics.md
└── texts/
    ├── 00-republic.md
    └── 01-nicomachean-ethics.md
```

**When to Use**: Oracle about field of knowledge

## Core Components

### 1. SKILL.md (Required)

**Purpose**: Entry point with metadata and complete guide

**Contains:**
- YAML frontmatter (name + description)
- What the oracle knows
- Directory structure
- When to use (question → file mappings)
- File descriptions
- Navigation guides

**Why Required**: Claude Code discovers skills via SKILL.md

### 2. INDEX.md (Recommended)

**Purpose**: Master index with cross-references

**Contains:**
- Quick navigation table
- Document structure tree
- Topic index
- Usage examples
- Cross-reference mappings

**Why Recommended**: Efficient knowledge retrieval

### 3. Source Documents

**Purpose**: Complete source material in markdown

**Location**: Usually `source-documents/` or organized by category

**Contains:**
- Converted PDFs
- Original markdown files
- Research papers
- Books/chapters
- Documentation

**Why Included**: Oracle is self-contained

### 4. Organized Content

**Purpose**: Extracted concepts and practical guides

**Typical Categories:**
- `concepts/` - Key ideas and theories
- `guides/` - How-to documentation
- `examples/` - Working code/demos
- `architecture/` - System design (technical oracles)
- `codebase/` - Code documentation (technical oracles)

**Why Organized**: Fast topic navigation

## Naming Conventions

### Oracle Names

**Pattern**: `{topic}-oracle`

**Rules:**
- Lowercase only
- Hyphens (not underscores)
- Descriptive (not abbreviated)
- Ends with `-oracle`

**Good:**
- `ancient-greek-philosophy-oracle` ✅
- `quantum-physics-oracle` ✅
- `john-vervaeke-oracle` ✅

**Bad:**
- `GreekPhilosophyOracle` ❌ (camelCase)
- `greek_philosophy_oracle` ❌ (underscores)
- `gp-oracle` ❌ (unclear)

### File Names

**Pattern**: `{number}-{descriptive-name}.md`

**Rules:**
- Start with `00-` for overviews
- Sequential numbers (`01-`, `02-`, etc.)
- Lowercase, hyphens
- Descriptive (not abbreviated)

**Examples:**
```
concepts/
├── 00-overview.md          # Always start here
├── 01-fundamentals.md
├── 02-advanced-topics.md
└── 03-expert-techniques.md
```

### Directory Names

**Pattern**: `{category}/`

**Rules:**
- Plural preferred
- Lowercase, hyphens
- Clear purpose

**Examples:**
- `concepts/` ✅
- `philosophers/` ✅
- `source-documents/` ✅

## File Organization

### Numbered Prefixes

**Why numbered prefixes?**
1. **Enforces order** - Logical progression
2. **Predictable** - Easy to reference
3. **Insertable** - Add topics between existing
4. **Clear path** - Intro → advanced

**Convention:**
- `00-` - Overviews and introductions
- `01-` - Fundamentals
- `02-` - Intermediate
- `03-` - Advanced
- `04+` - Expert/specialized

### Cross-References

**Always use relative paths:**
```markdown
[core concept](concepts/00-overview.md)
[advanced guide](guides/03-expert.md)
[example code](examples/00-basic.md)
```

**Never use:**
- Absolute paths (`/Users/...`)
- External links to local files
- Broken references

## Oracle Creation Process

### High-Level Workflow

1. **Gather** - Collect source documents
2. **Convert** - Transform to markdown
3. **Organize** - Structure with numbered prefixes
4. **Generate** - Create SKILL.md + INDEX.md
5. **Deploy** - Place in `.claude/skills/{oracle-name}/`
6. **Test** - Verify activation

### Detailed Steps

See `03-creation-workflow.md` for complete process.

## Quality Standards

### Structure Quality

- [ ] SKILL.md has YAML metadata
- [ ] All files use numbered prefixes
- [ ] Directory structure is clear
- [ ] Cross-references are valid
- [ ] Oracle name follows conventions

### Content Quality

- [ ] All sources converted to markdown
- [ ] No external dependencies
- [ ] Concepts are well-organized
- [ ] Navigation is intuitive
- [ ] Examples are practical

### Usability Quality

- [ ] SKILL.md explains WHEN to use
- [ ] INDEX.md covers all files
- [ ] Quick start exists
- [ ] Question → file mappings clear
- [ ] Cross-references work

## Integration with Claude Code

### Discovery

Claude Code auto-discovers skills at startup:
1. Scans `.claude/skills/`
2. Reads all `SKILL.md` files
3. Registers oracles with descriptions
4. Activates on relevant queries

### Activation

Oracles activate based on SKILL.md description:
```yaml
description: "Knows ancient Greek philosophy. Use when questions
involve Plato, Aristotle, Socratic method, theory of Forms..."
```

User asks: "What is Plato's theory of Forms?"
→ Claude activates `ancient-greek-philosophy-oracle`
→ Reads relevant concept files
→ Provides expert answer

### Usage

Access oracle knowledge:
```python
# In Claude Code
Skill('ancient-greek-philosophy-oracle')
# Activates oracle, makes knowledge available
```

## Best Practices

### Source Organization

**Strategy 1: By Document**
Use when sources are self-contained chapters/papers:
```
source-documents/
├── 00-republic.md
├── 01-symposium.md
└── 02-nicomachean-ethics.md
```

**Strategy 2: By Concept**
Use when extracting themes across sources:
```
concepts/
├── 00-ethics.md
├── 01-metaphysics.md
└── 02-epistemology.md
```

**Strategy 3: Hybrid**
Use for comprehensive oracles:
```
├── source-documents/    # Original works
├── concepts/            # Extracted themes
├── guides/              # Practical howtos
└── examples/            # Working demos
```

### Content Extraction

**Extract key concepts:**
- Identify recurring themes
- Group related ideas
- Create focused topic files
- Link back to sources

**Maintain traceability:**
- Note which source each concept came from
- Provide page numbers if available
- Link concepts to full source documents

### Navigation Design

**Multiple access paths:**
1. **SKILL.md** - High-level overview
2. **INDEX.md** - Topic-based lookup
3. **Cross-references** - Follow connections
4. **Quick navigation** - Direct topic → file

## Common Patterns

### Pattern: Progressive Depth

```
00-overview.md       # Big picture
01-fundamentals.md   # Core concepts
02-intermediate.md   # Build complexity
03-advanced.md       # Expert level
```

### Pattern: By Subdomain

```
physics/
├── 00-mechanics.md
├── 01-thermodynamics.md
└── 02-quantum.md

chemistry/
├── 00-organic.md
└── 01-inorganic.md
```

### Pattern: Theory + Practice

```
concepts/             # What it is
├── 00-theory.md

guides/               # How to use it
├── 00-getting-started.md

examples/             # Working code
└── 00-basic-example.md
```

## Examples in This Project

### john-vervaeke-oracle

**Type**: Academic (person-focused)
**Size**: ~4,000 lines
**Structure**:
- 6 academic paper analyses
- 4 concept overviews
- 1 project application guide

**Lessons**:
- Papers analyzed in depth
- Concepts extracted and enriched
- Project-specific applications

### ovis-2-5-oracle

**Type**: Technical (ML model)
**Size**: 42 files
**Structure**:
- Architecture documentation
- Training pipeline details
- Code implementation guide
- Usage examples

**Lessons**:
- Comprehensive file organization
- Clear numbered prefixes
- Strong INDEX.md navigation
- Code references with line numbers

### deepseek-ocr-oracle

**Type**: Technical (vision-language model)
**Structure**:
- Model architecture
- Training methodology
- Implementation details

**Lessons**:
- Focus on key innovations
- Clear architectural diagrams
- Code-to-concept mappings

## Future Directions

### Multi-Source Oracles

Combine multiple source directories:
```bash
"Create ml-foundations-oracle from:
 - ~/papers/computer-vision/
 - ~/papers/nlp/
 - ~/papers/reinforcement-learning/"
```

### Incremental Updates

Add new knowledge to existing oracle:
```bash
"Add these 3 papers to quantum-physics-oracle:
 - bell-theorem-2024.pdf
 - entanglement-experiments.pdf"
```

### Cross-Oracle References

Link between related oracles:
```markdown
See also: [john-vervaeke-oracle](../john-vervaeke-oracle/SKILL.md)
for cognitive science perspective.
```

## Summary

**Oracles are:**
- Domain expert skills
- Self-contained knowledge bases
- Structured for efficient retrieval
- Activated by relevant questions

**Creating oracles involves:**
1. Converting sources to markdown
2. Organizing with numbered structure
3. Generating SKILL.md + INDEX.md
4. Deploying to `.claude/skills/`

**Quality oracles have:**
- Clear SKILL.md with metadata
- Comprehensive INDEX.md
- All sources in markdown
- Intuitive navigation
- Proper naming conventions

---

**Next Steps:**
- Read `01-skill-md-template.md` for SKILL.md structure
- Read `02-index-template.md` for INDEX.md structure
- Read `03-creation-workflow.md` for step-by-step process
