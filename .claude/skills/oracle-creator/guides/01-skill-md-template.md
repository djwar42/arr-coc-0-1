# SKILL.md Template

Complete template for creating oracle SKILL.md files.

## Template Structure

```markdown
---
name: {oracle-name}
description: {Brief description of what oracle knows} + {When to use it} + {Key topics}. Use when questions involve {topic1}, {topic2}, {topic3}. (max 1024 chars)
---

# {Oracle Display Name}

## What This Skill Provides

Brief summary of oracle's knowledge domain and coverage.

**Example:**
> Complete knowledge base on ancient Greek philosophy including Plato's theory of Forms, Aristotle's metaphysics, Socratic method, and Pre-Socratic thought. Includes primary texts, concept analyses, and philosophical connections.

## Directory Structure

```
.claude/skills/{oracle-name}/
├── SKILL.md (this file)
├── INDEX.md
├── {category1}/
│   ├── 00-{file}.md
│   └── 01-{file}.md
├── {category2}/
│   └── 00-{file}.md
└── {category3}/
    └── 00-{file}.md
```

## When to Use This Skill

Map example questions to specific files:

**Example:**
- "What is Plato's theory of Forms?" → Read `concepts/00-theory-of-forms.md`
- "Explain Aristotle's metaphysics" → Read `concepts/01-aristotle-metaphysics.md`
- "How does the Socratic method work?" → Read `guides/00-socratic-method.md`

## Detailed File Descriptions

### {Category 1} Directory

#### 1. {File Name}
**Size**: {X} lines | **Content**: {Brief description}
- {Key point 1}
- {Key point 2}
- {Key point 3}

#### 2. {File Name}
**Size**: {X} lines | **Content**: {Brief description}
- {Key point 1}
- {Key point 2}

### {Category 2} Directory

[Repeat pattern above]

## Quick Navigation

**For {Topic A}** → `{path/to/file.md}`
**For {Topic B}** → `{path/to/file.md}`
**For {Topic C}** → `{path/to/file.md}`

## Key Insights Summary

Bullet points of most important concepts:

- **{Concept 1}**: {One-sentence explanation}
- **{Concept 2}**: {One-sentence explanation}
- **{Concept 3}**: {One-sentence explanation}

## File Access Patterns

When user asks:

**"{Example question 1}"**
→ Read `{file1.md}` first
→ Then `{file2.md}` for depth

**"{Example question 2}"**
→ Read `{file3.md}`
→ Complete answer there

## Integration with {Project Name}

If applicable, explain how oracle applies to current project.

**Example:**
> This oracle provides theoretical foundation for ARR-COC-VIS relevance realization implementation. See `Application-Guide.md` for direct module mappings.

## Citation & Attribution

How to cite oracle sources.

**Example:**
> When citing this knowledge base:
> Plato. (380 BCE). The Republic. Synthesized from multiple translations for ancient-greek-philosophy-oracle, 2025.

## Related Oracles

- `{related-oracle-1}` - {Brief connection}
- `{related-oracle-2}` - {Brief connection}

---

**Last Updated**: {Date}
**Status**: {Complete/In Progress}
**Version**: {X.Y}
**Total Content**: {X lines/pages of knowledge}
```

## Complete Example

Here's a complete SKILL.md for `ancient-greek-philosophy-oracle`:

```markdown
---
name: ancient-greek-philosophy-oracle
description: Comprehensive knowledge base on ancient Greek philosophy including Plato's theory of Forms, Aristotle's metaphysics, Socratic method, Pre-Socratic thought, and classical Greek ethics. Use when questions involve Plato, Aristotle, Socrates, Forms, teleology, virtue ethics, or Greek epistemology. Includes primary texts, concept analyses, and philosophical connections.
---

# Ancient Greek Philosophy Oracle

## What This Skill Provides

Complete knowledge base on ancient Greek philosophy including:

1. **Plato's Philosophy** - Theory of Forms, Republic, epistemology
2. **Aristotle's System** - Metaphysics, ethics, logic, natural philosophy
3. **Socratic Method** - Dialectic, maieutics, philosophical inquiry
4. **Pre-Socratics** - Heraclitus, Parmenides, Democritus
5. **Hellenistic Schools** - Stoicism, Epicureanism, Skepticism

## Directory Structure

```
.claude/skills/ancient-greek-philosophy-oracle/
├── SKILL.md (this file)
├── INDEX.md
├── philosophers/
│   ├── 00-socrates.md
│   ├── 01-plato.md
│   ├── 02-aristotle.md
│   └── 03-presocratics.md
├── concepts/
│   ├── 00-theory-of-forms.md
│   ├── 01-aristotelian-metaphysics.md
│   ├── 02-virtue-ethics.md
│   └── 03-epistemology.md
├── texts/
│   ├── 00-republic.md
│   ├── 01-symposium.md
│   ├── 02-nicomachean-ethics.md
│   └── 03-metaphysics.md
└── guides/
    ├── 00-understanding-plato.md
    └── 01-reading-aristotle.md
```

## When to Use This Skill

### Philosophical Concepts
- "What is Plato's theory of Forms?" → Read `concepts/00-theory-of-forms.md`
- "Explain Aristotle's four causes" → Read `concepts/01-aristotelian-metaphysics.md`
- "What is virtue ethics?" → Read `concepts/02-virtue-ethics.md`

### Philosophical Methods
- "How does the Socratic method work?" → Read `philosophers/00-socrates.md`
- "What is dialectic?" → Read `guides/00-understanding-plato.md`

### Primary Texts
- "Summarize Plato's Republic" → Read `texts/00-republic.md`
- "What does Aristotle say about happiness?" → Read `texts/02-nicomachean-ethics.md`

## Detailed File Descriptions

### Philosophers Directory

#### 1. socrates.md
**Size**: 350 lines | **Content**: Socrates' life, method, and philosophy
- Socratic method and dialectic
- Historical context in Athens
- Trial and death
- Influence on Plato

#### 2. plato.md
**Size**: 650 lines | **Content**: Plato's complete philosophical system
- Theory of Forms
- Political philosophy
- Epistemology and knowledge
- Soul and immortality
- Dialogues overview

#### 3. aristotle.md
**Size**: 800 lines | **Content**: Aristotle's systematic philosophy
- Metaphysics and ontology
- Four causes and teleology
- Ethics and virtue
- Logic and categories
- Natural philosophy

#### 4. presocratics.md
**Size**: 400 lines | **Content**: Early Greek philosophers
- Thales, Anaximander, Anaximenes
- Heraclitus and flux
- Parmenides and Being
- Democritus and atomism

### Concepts Directory

#### 1. theory-of-forms.md
**Size**: 500 lines | **Content**: Plato's most important doctrine
- Forms vs particulars
- Participation and imitation
- The Form of the Good
- Epistemological implications
- Criticisms and responses

#### 2. aristotelian-metaphysics.md
**Size**: 550 lines | **Content**: Aristotle's metaphysical system
- Substance and essence
- Matter and form (hylomorphism)
- Four causes
- Actuality and potentiality
- Prime mover

#### 3. virtue-ethics.md
**Size**: 450 lines | **Content**: Greek ethical theory
- Virtue (arete) and excellence
- Golden mean
- Eudaimonia (flourishing)
- Practical wisdom (phronesis)
- Cardinal virtues

#### 4. epistemology.md
**Size**: 400 lines | **Content**: Greek theories of knowledge
- Plato's divided line
- Allegory of the cave
- Aristotelian empiricism
- Knowledge vs opinion

### Primary Texts Directory

#### 1. republic.md
**Size**: 1200 lines | **Content**: Complete analysis of Plato's Republic
- Justice in the individual and state
- Three parts of the soul
- Philosopher kings
- Theory of Forms presented
- Allegory of the cave

#### 2. symposium.md
**Size**: 600 lines | **Content**: Plato's dialogue on love
- Speeches on Eros
- Ladder of love
- Beauty and the Good
- Socrates and Diotima

#### 3. nicomachean-ethics.md
**Size**: 1000 lines | **Content**: Aristotle's ethical masterwork
- Happiness as highest good
- Virtue ethics framework
- Golden mean
- Practical wisdom
- Friendship

#### 4. metaphysics.md
**Size**: 900 lines | **Content**: Aristotle's first philosophy
- Being qua being
- Substance theory
- Four causes explained
- Prime mover argument

### Guides Directory

#### 1. understanding-plato.md
**Size**: 350 lines | **Content**: How to read Plato
- Dialogue form explained
- Historical context
- Key themes to watch
- Recommended reading order

#### 2. reading-aristotle.md
**Size**: 300 lines | **Content**: Approach to Aristotle's treatises
- Systematic vs dialogical
- Technical terminology
- Logical structure
- Corpus overview

## Quick Navigation

**For Plato** → `philosophers/01-plato.md` or `concepts/00-theory-of-forms.md`
**For Aristotle** → `philosophers/02-aristotle.md` or `concepts/01-aristotelian-metaphysics.md`
**For Socrates** → `philosophers/00-socrates.md`
**For Ethics** → `concepts/02-virtue-ethics.md` or `texts/02-nicomachean-ethics.md`
**For Metaphysics** → `concepts/01-aristotelian-metaphysics.md` or `texts/03-metaphysics.md`

## Key Insights Summary

### Core Concepts

- **Theory of Forms**: Plato's doctrine that abstract Forms are more real than physical particulars
- **Four Causes**: Aristotle's explanation of change (material, formal, efficient, final)
- **Virtue Ethics**: Excellence (arete) achieved through cultivating virtues, leading to eudaimonia
- **Socratic Method**: Philosophical inquiry through questioning to expose contradictions
- **Hylomorphism**: Aristotle's view that things are matter-form composites

### Philosophical Contributions

- **Epistemology**: Distinction between knowledge (episteme) and opinion (doxa)
- **Ethics**: Virtue-based approach to the good life
- **Metaphysics**: Systematic accounts of being, substance, causation
- **Logic**: Aristotle's foundational work in syllogistic reasoning
- **Political Philosophy**: Plato's ideal state, Aristotle's analysis of constitutions

## File Access Patterns

When user asks:

**"What is Plato's theory of Forms?"**
→ Read `concepts/00-theory-of-forms.md` first (focused concept)
→ Then `philosophers/01-plato.md` for broader context
→ Then `texts/00-republic.md` for full presentation

**"Explain Aristotle's four causes"**
→ Read `concepts/01-aristotelian-metaphysics.md` (complete explanation)

**"What is virtue ethics?"**
→ Read `concepts/02-virtue-ethics.md` (concept overview)
→ Then `texts/02-nicomachean-ethics.md` (full Aristotelian treatment)

**"How does the Socratic method work?"**
→ Read `philosophers/00-socrates.md` (method explained)
→ Then `guides/00-understanding-plato.md` (method in practice)

## Citation & Attribution

When citing this knowledge base:

> Plato, Aristotle, et al. Ancient Greek Philosophy. Synthesized from primary texts and scholarship for ancient-greek-philosophy-oracle, 2025.

Primary sources:
- Plato (380 BCE). The Republic. Translated by various.
- Aristotle (350 BCE). Nicomachean Ethics. Translated by various.
- Aristotle (350 BCE). Metaphysics. Translated by various.

## Related Oracles

- `john-vervaeke-oracle` - Connects to virtue epistemology and wisdom
- `philosophy-of-mind-oracle` - Plato's theory of soul, Aristotle's De Anima

---

**Last Updated**: 2025-10-28
**Status**: Complete knowledge base
**Version**: 1.0
**Total Content**: ~8,000 lines across 15 files
**Coverage**: Plato, Aristotle, Socrates, Pre-Socratics, core concepts
```

## Key Elements

Every SKILL.md must have:

1. **YAML Frontmatter** - name and description
2. **What This Skill Provides** - Overview
3. **Directory Structure** - Visual tree
4. **When to Use** - Question → file mappings
5. **Detailed File Descriptions** - Every file explained
6. **Quick Navigation** - Topic → file shortcuts
7. **Key Insights** - Core concepts summarized

## Best Practices

### Description

**Include:**
- What oracle knows (domain)
- When to use (trigger keywords)
- Key topics covered

**Keep under 1024 characters**

### File Descriptions

**Format:**
```markdown
#### {Number}. {Filename}
**Size**: {Lines} | **Content**: {Brief description}
- {Key point 1}
- {Key point 2}
- {Key point 3}
```

### Navigation

**Multiple paths:**
- Quick navigation (direct links)
- When to use (question-based)
- File access patterns (workflows)
- Related oracles (connections)

---

## Oracle Knowledge Expansion Section (Add to All Oracles)

**Every oracle SKILL.md should include the complete Oracle Knowledge Expansion section.**

This section documents how the oracle learns, organizes, and grows knowledge autonomously.

**Standard placement**: After the main oracle documentation, before the end of file.

**Required subsections:**
1. Manual Ingestion (_ingest/)
2. Autonomous Expansion (_ingest-auto/)
3. Execution via Oracle-Knowledge-Runner Sub-Agents
4. Oracle Knowledge Expansion: ACQUISITION ONLY warning
5. Simplified Flow diagram
6. Oracle's Workflow (Steps 1-5)
7. Runner's Execution details
8. Key Principles

**Critical: Include synchronization warning**

After the "This oracle is fully autonomous" line and first `---`, insert:

```markdown
## 🔧 Modifying Oracle Knowledge Expansion Itself

**If you modify how Oracle Knowledge Expansion works** (runner workflow, finalization steps, KNOWLEDGE DROP format):

1. Update THIS oracle's SKILL.md first
2. Update ALL other oracles: `.claude/skills/*-oracle/SKILL.md`
3. Update template: `.claude/skills/oracle-creator/guides/01-skill-md-template.md`
4. Commit all together: "Modify Oracle Knowledge Expansion: [what changed]"

**Don't modify in just one oracle - they must stay synchronized.**

---
```

**See existing oracles for complete Oracle Knowledge Expansion section examples:**
- `.claude/skills/ovis-2-5-oracle/SKILL.md`
- `.claude/skills/deepseek-ocr-oracle/SKILL.md`

---

Use this template when creating new oracles. Adapt structure to fit your oracle's specific needs.
