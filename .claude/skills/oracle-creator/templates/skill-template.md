---
name: {oracle-name}
description: {What oracle knows} + {When to use} + {Key topics}. Use when questions involve {topic1}, {topic2}, {topic3}. (max 1024 chars total)
---

# {Oracle Display Name}

## What This Skill Provides

{Brief overview of oracle's knowledge and capabilities}

## Directory Structure

```
.claude/skills/{oracle-name}/
├── SKILL.md (this file)
├── INDEX.md
├── {category1}/
│   ├── 00-{file}.md
│   └── 01-{file}.md
└── {category2}/
    └── 00-{file}.md
```

## When to Use This Skill

Map example questions to files:

- "{Example question 1}" → Read `{path/to/file1.md}`
- "{Example question 2}" → Read `{path/to/file2.md}`
- "{Example question 3}" → Read `{path/to/file3.md}`

## Detailed File Descriptions

### {Category 1} Directory

#### 1. {filename}.md
**Size**: {N} lines | **Content**: {Brief description}
- {Key point 1}
- {Key point 2}
- {Key point 3}

#### 2. {filename}.md
**Size**: {N} lines | **Content**: {Brief description}
- {Key point 1}
- {Key point 2}

### {Category 2} Directory

[Repeat pattern above for all categories and files]

## Quick Navigation

**For {Topic A}** → `{path/to/file.md}`
**For {Topic B}** → `{path/to/file.md}`
**For {Topic C}** → `{path/to/file.md}`

## Key Insights Summary

- **{Concept 1}**: {One-sentence explanation}
- **{Concept 2}**: {One-sentence explanation}
- **{Concept 3}**: {One-sentence explanation}

## File Access Patterns

When user asks:

**"{Example question}"**
→ Read `{file1.md}` first
→ Then `{file2.md}` for depth

**"{Example question}"**
→ Read `{file3.md}`
→ Complete answer there

## Citation & Attribution

{How to cite oracle sources}

## Related Oracles

- `{related-oracle-1}` - {Brief connection}
- `{related-oracle-2}` - {Brief connection}

---

## Oracle Learning Mode

**This is a {SEEKING / STATIC} Oracle**

### {If SEEKING}

**✅ Seeking Oracle** - Can expand knowledge dynamically during conversations

See [oracle-dynamic-learning.md](../../oracle-creator/templates/oracle-dynamic-learning.md) for complete guidelines.

**Quick Guidelines:**

**Conservative expansion policy:**
- **Usually ZERO** expansions per conversation
- **Max 1-2** expansions per conversation
- **Only when**:
  1. User explicitly requests new knowledge
  2. User reveals critical gap in oracle knowledge
  3. Cannot respond adequately with existing knowledge

**How to expand:**
```python
# Judicious search in oracle's domain
mcp__bright-data__search_engine(
    query="site:.edu {oracle domain} {specific topic}"
)

# Create dynamic file: {parent-num}-{sub-num}-{topic}-{YYYYMMDD}.md
```

**Domain Boundaries**:
- **Core**: {Primary domain topics}
- **Adjacent**: {Related fields when connected}
- **Out of bounds**: {Unrelated topics}

### {If STATIC}

**🔒 Static Oracle** - Fixed knowledge base, no dynamic expansion

This oracle's knowledge is:
- **Fixed at creation** - {Reason: e.g., "Historical accuracy for 1950s period"}
- **Version-locked** - {Reason: e.g., "Codebase frozen at v2.1.0"}
- **Canonical** - {Reason: e.g., "Based solely on original texts"}
- **Unchanging by design** - {Reason: user requested fixed oracle}

To expand this oracle, use `oracle-creator` to add new source documents and regenerate.

---

## Oracle Self-Check

**IMPORTANT: You MUST activate oracle-creator BEFORE running self-check!**

⚠️ **MANDATORY FIRST STEP - Activate oracle-creator**

**DO NOT start self-check without this:**
```
Skill(oracle-creator)
```

**Why this is mandatory:**
- Oracle-creator contains the authoritative format and standards
- Self-check verifies against oracle-creator's specifications
- Without oracle-creator active, you don't know what to check for
- This is NOT optional - it's a hard requirement

**After activating oracle-creator, proceed with checks below:**

**Step 2: Check folder structure**
```bash
# Required folders:
✅ _ingest/README.md exists
✅ _ingest-auto/README.md exists
✅ source-documents/ exists (if applicable)
✅ source-codebases/ exists (if applicable)
✅ INDEX.md exists in root
✅ SKILL.md exists in root
```

**Step 3: Verify codebases (if present)**
```bash
# For each codebase in source-codebases/:
✅ Has INDEX.md (UPPERCASE) inside codebase folder
✅ No loose INDEX.md or overview.md files outside codebases
✅ Codebase folders numbered: 00-{name}/, 01-{name}/
```

**Step 4: Check for misplaced files**
```bash
# Look for strange .md files:
❌ No {name}-overview.md outside proper locations
❌ No index.md (lowercase) - should be INDEX.md (UPPERCASE)
❌ No unnumbered .md files in root
❌ No .git directories in source-codebases/ copies
```

**Step 5: Verify numbering conventions**
```bash
# All documentation files should have prefixes:
✅ concepts/00-*.md, 01-*.md, etc.
✅ architecture/00-*.md, 01-*.md, etc.
✅ source-documents/00-*.pdf, 00-*.md, etc.
✅ Dynamic additions: {parent}-{sub}-{topic}-{date}.md
```

**Step 6: Report findings**
```
✓ All checks passed
✗ Issues found: {list specific problems}
```

**When to run:**
- After oracle creation
- After adding new knowledge
- Periodically for maintenance
- When oracle behavior seems off

---

**Last Updated**: {Date}
**Status**: {Complete/In Progress}
**Version**: {X.Y}
**Total Content**: {X lines of knowledge}
**Learning Mode**: {Seeking (dynamic) / Static (fixed)}
