# Example Oracle: Ancient Greek Philosophy

This document shows what a complete oracle structure looks like using `ancient-greek-philosophy-oracle` as an example.

## User Request

```
User: "Create ancient-greek-philosophy-oracle from PDFs in ~/philosophy-texts/
       This oracle knows Plato, Aristotle, Socratic method, and Greek metaphysics.

       Sources:
       - republic.pdf (Plato's Republic)
       - symposium.pdf (Plato's Symposium)
       - nicomachean-ethics.pdf (Aristotle's Ethics)
       - metaphysics.pdf (Aristotle's Metaphysics)
       - presocratics.pdf (Pre-Socratic fragments)
       "
```

## Oracle Creator Response

### Phase 1: Requirements Confirmed

```
Creating: ancient-greek-philosophy-oracle
Sources: 5 PDFs from ~/philosophy-texts/
Specialization: Plato, Aristotle, Socratic method, Greek metaphysics

Proceeding with oracle creation...
```

### Phase 2: Preserve Originals and Convert

```
Copying original PDFs and converting to markdown...

✓ Copied: republic.pdf → source-documents/00-republic.pdf
✓ Converted: republic.pdf → source-documents/00-republic.md (1200 lines)

✓ Copied: symposium.pdf → source-documents/01-symposium.pdf
✓ Converted: symposium.pdf → source-documents/01-symposium.md (600 lines)

✓ Copied: nicomachean-ethics.pdf → source-documents/02-nicomachean-ethics.pdf
✓ Converted: nicomachean-ethics.pdf → source-documents/02-nicomachean-ethics.md (1000 lines)

✓ Copied: metaphysics.pdf → source-documents/03-metaphysics.pdf
✓ Converted: metaphysics.pdf → source-documents/03-metaphysics.md (900 lines)

✓ Copied: presocratics.pdf → source-documents/04-presocratics.pdf
✓ Converted: presocratics.pdf → source-documents/04-presocratics.md (400 lines)

Result: 5 original PDFs preserved + 5 markdown conversions
Total markdown: 4,100 lines across 5 source documents
```

### Phase 3: Read Sources and Extract Organization

```
Reading all source markdown files...

Reading source-documents/00-republic.md (1200 lines)...
- Found: Theory of Forms, justice, ideal state, philosopher kings
- Major theme: Political philosophy + metaphysics

Reading source-documents/01-symposium.md (600 lines)...
- Found: Theory of love, beauty, Forms
- Major theme: Love and ascent to Forms

Reading source-documents/02-nicomachean-ethics.md (1000 lines)...
- Found: Virtue, happiness, golden mean, practical wisdom
- Major theme: Ethical theory

Reading source-documents/03-metaphysics.md (900 lines)...
- Found: Being, substance, four causes, prime mover
- Major theme: Metaphysical foundations

Reading source-documents/04-presocratics.md (400 lines)...
- Found: Early philosophy, flux, atomism
- Major theme: Origins of Western thought

Analysis complete. Identified topic structure:
- philosophers/ (Thinker profiles)
- concepts/ (Major ideas across sources)
- texts/ (Deep analyses of specific works)
- guides/ (Practical reading approaches)

Creating directories and extracting content...
✓ philosophers/
✓ concepts/
✓ texts/
✓ guides/

Extracting concepts with source citations:
✓ concepts/00-theory-of-forms.md (500 lines)
  → Cites: 00-republic.md, 01-symposium.md
✓ concepts/01-aristotelian-metaphysics.md (550 lines)
  → Cites: 03-metaphysics.md, 02-nicomachean-ethics.md
✓ concepts/02-virtue-ethics.md (450 lines)
  → Cites: 02-nicomachean-ethics.md
✓ concepts/03-epistemology.md (400 lines)
  → Cites: 00-republic.md

Creating philosopher profiles with source citations:
✓ philosophers/00-socrates.md (350 lines)
  → Cites: 00-republic.md, 01-symposium.md
✓ philosophers/01-plato.md (650 lines)
  → Cites: 00-republic.md, 01-symposium.md
✓ philosophers/02-aristotle.md (800 lines)
  → Cites: 02-nicomachean-ethics.md, 03-metaphysics.md
✓ philosophers/03-presocratics.md (400 lines)
  → Cites: 04-presocratics.md

Creating text analyses with source citations:
✓ texts/00-republic.md (1200 lines)
  → Based on: source-documents/00-republic.md
✓ texts/01-symposium.md (600 lines)
  → Based on: source-documents/01-symposium.md
✓ texts/02-nicomachean-ethics.md (1000 lines)
  → Based on: source-documents/02-nicomachean-ethics.md
✓ texts/03-metaphysics.md (900 lines)
  → Based on: source-documents/03-metaphysics.md

Creating reading guides with source citations:
✓ guides/00-understanding-plato.md (350 lines)
  → Synthesizes: 00-republic.md, 01-symposium.md
✓ guides/01-reading-aristotle.md (300 lines)
  → Synthesizes: 02-nicomachean-ethics.md, 03-metaphysics.md

Verifying all files cite sources...
✓ All 15 extracted files include "Primary Sources" sections
✓ All files link back to source-documents/
```

### Phase 4: Generate Metadata

```
Generating SKILL.md...
- YAML frontmatter with oracle metadata
- Directory structure tree
- File descriptions for all 15 files
- Question → file mappings
- Navigation guides

Generating INDEX.md...
- Quick navigation table
- Topic index by category
- Cross-reference map
- Usage examples
- File summary with line counts
```

### Phase 5: Quality Assurance

```
Validating structure...
✓ All files have numbered prefixes
✓ Directory structure follows conventions
✓ Oracle name is valid

Validating cross-references...
✓ All markdown links resolve
✓ No broken references
✓ Cross-reference map accurate

Validating content...
✓ All files have valid markdown
✓ No encoding errors
✓ Reasonable file lengths
```

### Phase 6: Deployment

```
Oracle structure:
.claude/skills/ancient-greek-philosophy-oracle/
├── SKILL.md (400 lines)
├── INDEX.md (500 lines)
├── source-documents/              # ORIGINALS PRESERVED + CONVERSIONS
│   ├── 00-republic.pdf            # Original PDF (preserved)
│   ├── 00-republic.md             # Converted markdown (1200 lines)
│   ├── 01-symposium.pdf           # Original PDF (preserved)
│   ├── 01-symposium.md            # Converted markdown (600 lines)
│   ├── 02-nicomachean-ethics.pdf  # Original PDF (preserved)
│   ├── 02-nicomachean-ethics.md   # Converted markdown (1000 lines)
│   ├── 03-metaphysics.pdf         # Original PDF (preserved)
│   ├── 03-metaphysics.md          # Converted markdown (900 lines)
│   ├── 04-presocratics.pdf        # Original PDF (preserved)
│   └── 04-presocratics.md         # Converted markdown (400 lines)
├── philosophers/                   # EXTRACTED with source citations
│   ├── 00-socrates.md (350 lines) → cites source-documents/
│   ├── 01-plato.md (650 lines) → cites source-documents/
│   ├── 02-aristotle.md (800 lines) → cites source-documents/
│   └── 03-presocratics.md (400 lines) → cites source-documents/
├── concepts/                       # EXTRACTED with source citations
│   ├── 00-theory-of-forms.md (500 lines) → cites source-documents/
│   ├── 01-aristotelian-metaphysics.md (550 lines) → cites source-documents/
│   ├── 02-virtue-ethics.md (450 lines) → cites source-documents/
│   └── 03-epistemology.md (400 lines) → cites source-documents/
├── texts/                          # EXTRACTED with source citations
│   ├── 00-republic.md (1200 lines) → based on source-documents/00-republic.md
│   ├── 01-symposium.md (600 lines) → based on source-documents/01-symposium.md
│   ├── 02-nicomachean-ethics.md (1000 lines) → based on source-documents/02-*.md
│   └── 03-metaphysics.md (900 lines) → based on source-documents/03-metaphysics.md
└── guides/                         # EXTRACTED with source citations
    ├── 00-understanding-plato.md (350 lines) → synthesizes multiple sources
    └── 01-reading-aristotle.md (300 lines) → synthesizes multiple sources

Total: 25 files (15 extracted + 5 PDFs + 5 MD conversions)
Total markdown content: ~11,500 lines
Originals: 5 PDFs preserved for verification

Committing to git...
```

```bash
git add .claude/skills/ancient-greek-philosophy-oracle
git commit -m "Add ancient-greek-philosophy-oracle skill

- Converted 5 primary source PDFs to markdown
- Organized into philosophers, concepts, texts, guides
- Generated comprehensive SKILL.md (400 lines)
- Generated comprehensive INDEX.md (500 lines)
- Oracle knows Plato, Aristotle, Socratic method, Greek metaphysics
- Total: 20 files, ~11,500 lines of philosophical knowledge
"
```

### Phase 7: Success

```
✅ ancient-greek-philosophy-oracle created successfully!

Location: .claude/skills/ancient-greek-philosophy-oracle/
Files: 20 total
  - 5 source documents (4,100 lines)
  - 4 philosopher profiles (2,200 lines)
  - 4 concept analyses (1,900 lines)
  - 4 primary texts (3,700 lines)
  - 2 reading guides (650 lines)
  - SKILL.md + INDEX.md (900 lines)

Total: ~11,500 lines of ancient Greek philosophy knowledge

To use:
1. Restart Claude Code to load the skill
2. Ask questions like:
   - "What is Plato's theory of Forms?"
   - "Explain Aristotle's four causes"
   - "How does the Socratic method work?"
3. The oracle will activate automatically

Test it now? (Restart required)
```

## Example Usage After Creation

### User Question 1

```
User: "What is Plato's theory of Forms?"

Claude: [Activates ancient-greek-philosophy-oracle]
        [Reads concepts/00-theory-of-forms.md]
        [Reads philosophers/01-plato.md for context]

        "Plato's theory of Forms is his central metaphysical doctrine.
        According to the theory, abstract Forms (or Ideas) are more real
        than physical particulars we perceive...

        [Detailed explanation from oracle knowledge base]

        The theory appears most fully in Books 5-7 of the Republic,
        particularly in the divided line analogy and allegory of the cave."
```

### User Question 2

```
User: "Explain Aristotle's four causes"

Claude: [Activates ancient-greek-philosophy-oracle]
        [Reads concepts/01-aristotelian-metaphysics.md]

        "Aristotle's four causes are his explanation of how things come
        to be and why they are as they are:

        1. Material Cause: What something is made of
        2. Formal Cause: The essence or definition
        3. Efficient Cause: What brought it into being
        4. Final Cause: The purpose or end (telos)

        [Detailed explanation with examples from oracle knowledge]"
```

### User Question 3

```
User: "Compare Plato and Aristotle on metaphysics"

Claude: [Activates ancient-greek-philosophy-oracle]
        [Reads concepts/00-theory-of-forms.md]
        [Reads concepts/01-aristotelian-metaphysics.md]
        [Reads philosophers/01-plato.md]
        [Reads philosophers/02-aristotle.md]

        "Plato and Aristotle differ fundamentally on metaphysics:

        Plato's approach:
        - Forms are transcendent, separate from physical world
        - Particulars participate in or imitate Forms
        - True reality is the realm of Forms

        Aristotle's approach:
        - Form and matter are inseparable (hylomorphism)
        - No transcendent realm of Forms
        - Substances are the primary realities
        - Form is immanent in matter

        [Detailed comparison from oracle knowledge]"
```

## Key Takeaways

This example shows:

1. **Complete workflow** from user request to deployed oracle
2. **Content organization** into logical categories
3. **File structure** with numbered prefixes
4. **Metadata generation** (SKILL.md and INDEX.md)
5. **Quality assurance** steps
6. **Successful usage** after deployment

## Apply to Your Oracle

Use this pattern for any domain:
- Replace philosophy sources with your domain sources
- Adapt categories to your content structure
- Follow same numbering and organization principles
- Generate comprehensive SKILL.md and INDEX.md
- Test with domain-specific questions

---

**Example demonstrates**: Complete oracle creation workflow from start to finish
