# Oracle Dynamic Learning Guidelines

**IMPORTANT: Include this section in every oracle SKILL.md**

## Oracle Learning Mode

**This oracle is a: {SEEKING / STATIC} Oracle**

### Seeking Oracle (Default)

**Seeking oracles** can expand their knowledge on-the-fly during conversations using Bright Data web research.

**Use when:**
- Oracle covers evolving field (AI/ML, contemporary philosophy)
- Recent research is valuable
- Living knowledge base desired
- User wants oracle to grow over time

### Static Oracle (Fixed)

**Static oracles** have fixed knowledge and do NOT expand dynamically.

**Use when:**
- Historical accuracy required (specific time period)
- Versioned codebase (must stay at specific version)
- Canonical text interpretation (shouldn't incorporate modern views)
- User explicitly requests "unchanging" or "fixed" oracle
- Knowledge base should remain as originally created

---

## Dynamic Knowledge Expansion (For SEEKING Oracles Only)

**Note:** Skip this section entirely if oracle is STATIC.

This oracle can expand its knowledge **on-the-fly** during conversations using Bright Data web research.

### Conservative Expansion Policy

**⚠️ IMPORTANT: On-the-fly expansion should be RARE**

- **Default**: Use existing knowledge (usually zero expansions)
- **Maximum**: 1-2 expansions per conversation at most
- **Only expand when**:
  1. User **directly** requests new knowledge ("Can you search for...")
  2. User reveals **critical gap** ("I thought you'd know about...")
  3. Oracle lacks **essential** information to respond adequately

### When to Expand Knowledge

**✅ DO expand knowledge when:**
- **User explicitly asks** oracle to search for new information
- **User reveals gap**: "Don't you know about {critical topic}?"
- **Cannot respond** adequately with current knowledge
- Topic **directly** relates to oracle's domain expertise
- **Scholarly sources available** (reputable, domain-appropriate)

**❌ DON'T expand knowledge when:**
- Current knowledge is sufficient (even if not exhaustive)
- Topic is unrelated to oracle's theme/purpose
- Would require more than 1-2 searches in conversation
- Information is tangential or only loosely related
- User hasn't indicated need for new knowledge
- Source quality is questionable
- Already expanded 1-2 times this conversation

### Appropriate Expansion Examples

**{Oracle Name} Oracle** (expertise: {domain}):

**Good expansions:**
```
User asks about {related concept in domain}
→ Oracle searches: "site:.edu {concept} {oracle domain}"
→ Finds scholarly article expanding understanding
→ Integrates into response with citation
```

**Acceptable broad expansions:**
```
Discussion connects {domain} to {adjacent field}
→ Oracle searches for connection: "{domain} applications in {field}"
→ Finds reputable source showing relationship
→ Notes connection enriches domain understanding
```

**Bad expansions:**
```
Topic shifts to completely unrelated area
→ Oracle should NOT expand into unrelated domains
→ Stay within expertise boundaries
```

### How to Expand Knowledge (Detailed Process)

**Step 1: Recognize Gap**
```
User mentions specific aspect of {domain} not in current knowledge
→ Identify: "I should deepen understanding of {specific topic}"
→ Check: Is this within my domain? Do I need this now?
```

**Step 2: Judicious Search**
```python
# Search reputable sources in oracle's domain
results = mcp__bright-data__search_engine(
    query="site:.edu {oracle domain} {specific topic}"
)

# Or for scholarly articles
results = mcp__bright-data__search_engine(
    query="site:plato.stanford.edu {topic in oracle domain}"
)
```

**Step 3: Download PDFs to _ingest-auto/ (if applicable)**
```python
from pathlib import Path

# If source is a PDF, download to _ingest-auto/
ingest_auto = Path(".claude/skills/{oracle-name}/_ingest-auto")

# Download PDF (if applicable)
if url.endswith('.pdf'):
    pdf_path = ingest_auto / f"temp-{topic_slug}-{timestamp}.pdf"
    # Download PDF to _ingest-auto/
    download_pdf(url, pdf_path)

    # Convert PDF to markdown
    markdown_content = convert_pdf_to_markdown(pdf_path)

    # Extract relevant sections
    relevant_section = extract_key_insights(markdown_content, current_topic)
```

**Step 4: Scrape and Extract (web sources)**
```python
from datetime import datetime

# For web sources (non-PDF)
# Scrape relevant article
article = mcp__bright-data__scrape_as_markdown(url=scholarly_url)

# Extract relevant portions (don't save entire article)
relevant_section = extract_key_insights(article, current_topic)
```

**Step 5: Move or Delete PDF from _ingest-auto/**
```python
# After extracting relevant content, decide what to do with PDF

# Option 1: Keep PDF as source document (significant reference)
if pdf_is_significant_source:
    # Find next source document number
    source_docs = list(Path("source-documents").glob("*.pdf"))
    next_num = len(source_docs)

    # Move to source-documents/ with proper numbering
    source_path = Path(f"source-documents/{next_num:02d}-{topic_slug}.pdf")
    pdf_path.rename(source_path)

    # Also save the markdown conversion
    markdown_path = source_path.with_suffix('.md')
    markdown_path.write_text(markdown_content)

    print(f"✓ Preserved PDF: {source_path}")

# Option 2: Delete PDF (already extracted what we need)
else:
    pdf_path.unlink()  # Delete temporary PDF
    print(f"✓ Cleaned up temporary PDF: {pdf_path.name}")

# CRITICAL: Always clean _ingest-auto/ after processing
# Remove any leftover files
for temp_file in ingest_auto.glob("*"):
    if temp_file.name != "README.md":
        temp_file.unlink()
        print(f"✓ Cleaned: {temp_file.name}")

print(f"✓ _ingest-auto/ is clean and tidy")
```

**When to Keep vs Delete:**

**Keep PDF (move to source-documents/):**
- Major scholarly paper being added as primary source
- Reference that will be cited multiple times
- Complete work being integrated into oracle knowledge base

**Delete PDF (temporary extraction only):**
- Already extracted key insights into dynamic knowledge file
- Web article scraped for specific fact/citation
- Supplementary material, not primary source
- Would clutter source-documents/ unnecessarily

**Step 6: Create Dynamic Knowledge File**

**CRITICAL: Use numbered naming to show dynamic additions**

Pattern: `{original-number}-{sub-number}-{topic}-{YYYYMMDD}.md`

```python
# Example: Adding to concepts/ folder
# Original file: concepts/00-theory-of-forms.md
# Dynamic addition: concepts/00-1-contemporary-interpretations-20250128.md

from pathlib import Path
from datetime import datetime

# Determine parent topic number
parent_file = "concepts/00-theory-of-forms.md"  # The related original file
parent_num = "00"  # Extract from parent

# Find next sub-number (00-1, 00-2, etc.)
existing_subs = list(Path("concepts").glob(f"{parent_num}-*"))
next_sub = len(existing_subs) + 1

# Create filename with date
date_str = datetime.now().strftime("%Y%m%d")
filename = f"{parent_num}-{next_sub}-contemporary-interpretations-{date_str}.md"

# Full path
dynamic_file = Path(f"concepts/{filename}")
```

**Step 5: Write Dynamic Knowledge File**

```python
content = f"""# Contemporary Interpretations of Theory of Forms

**⚡ Dynamic Knowledge Addition**
**Added**: {datetime.now().strftime("%Y-%m-%d")}
**Reason**: User asked about modern scholarly interpretations
**Source**: [{source_title}]({source_url})
**Parent Topic**: [Theory of Forms](00-theory-of-forms.md)

## Overview

{brief_intro_from_oracle}

## Research Findings

From [{source_title}]({source_url}) (accessed {date}):

{extracted_relevant_content}

## Integration with Oracle Knowledge

This research expands our understanding of {topic} by:
- {Key insight 1 and how it relates to existing knowledge}
- {Key insight 2 and how it enriches domain}
- {Connection back to parent topic}

## References

- Original Topic: [Theory of Forms](00-theory-of-forms.md)
- Source: {source_url}
- Added: {YYYY-MM-DD}
- Domain: {oracle domain}

---

**Note**: This file was added dynamically during conversation to expand
oracle knowledge on {specific topic} within {domain}.
"""

# Write file
dynamic_file.write_text(content)

print(f"✓ Added dynamic knowledge: {filename}")
```

**Step 6: Update Parent File (Optional)**

Link from original file to dynamic addition:

```python
# Add to parent file (00-theory-of-forms.md)
link_section = f"""

## Recent Developments

- [Contemporary Interpretations](00-1-contemporary-interpretations-{date_str}.md) - Added {date}
"""

# Append to parent file
parent_path = Path("concepts/00-theory-of-forms.md")
parent_content = parent_path.read_text()
parent_path.write_text(parent_content + link_section)
```

**Step 7: Report Expansion Verbosely**

**IMPORTANT: Be verbose when expanding knowledge**

Tell user:
- What triggered the expansion
- What file was created
- How many lines added
- Why it was necessary
- What it adds to oracle expertise

```markdown
**⚡ Dynamic Knowledge Expansion**

**Triggered by**: User asked about contemporary interpretations, which isn't in my current knowledge base.

**Searched**: Stanford Encyclopedia of Philosophy for modern Plato scholarship

**Added file**: `concepts/00-1-contemporary-interpretations-20250128.md` (437 lines)

**Why**: My existing knowledge covers classical Theory of Forms but lacks modern scholarly perspectives that emerged in last 20 years. This fills a critical gap for answering user's question.

**Content**: Prof. X's analysis of Forms in context of analytic philosophy, connects to my existing knowledge in `concepts/00-theory-of-forms.md`.

**Source**: [Stanford Encyclopedia - Contemporary Plato](https://plato.stanford.edu/entries/plato-forms-contemporary/)

From the new research:
{answer to user's question using new knowledge}

This expansion enhances my Theory of Forms expertise with 21st-century scholarly perspectives.
```

**Format for reporting:**
```
⚡ Dynamic Knowledge Expansion
├─ Triggered: {User action or revealed gap}
├─ Searched: {Where you looked}
├─ File: {filename} ({N} lines)
├─ Why: {Necessity explanation}
├─ Connects to: {Existing knowledge}
└─ Source: {URL with date}
```

### Domain Boundaries

**{Oracle Name} Oracle Expertise**: {domain description}

**Core Domain** (always appropriate):
- {Core topic 1}
- {Core topic 2}
- {Core topic 3}

**Adjacent Domains** (appropriate if connected):
- {Adjacent field 1} - when it relates to {core domain}
- {Adjacent field 2} - when discussing {specific application}

**Out of Bounds** (not appropriate):
- {Unrelated field 1}
- {Unrelated field 2}
- Topics that don't connect back to oracle's purpose

### Guidelines for Conservative Use

**Frequency Limits (STRICTLY ENFORCED):**
- **Per conversation**: 1-2 expansions MAXIMUM, usually ZERO
- **Per topic**: At most 1 search per specific concept
- **Default behavior**: Use existing knowledge, don't expand unless clear need

**Triggering Conditions:**
- **User directly requests**: "Can you research...", "Search for..."
- **User reveals critical gap**: "Don't you know about...", "I'm surprised you don't have..."
- **Cannot adequately respond**: Lack essential information for proper answer

**Quality Standards:**
- **Reputable sources only**: .edu, Stanford Encyclopedia, peer-reviewed
- **Recent when relevant**: For fast-moving fields, prefer recent sources
- **Scholarly depth**: Prefer academic sources over blogs/general articles
- **Domain-specific**: Sources must be clearly related to oracle's expertise

**Integration Standards:**
- **Cite immediately**: Always include URL and access date
- **Explain necessity**: Why this expansion was needed
- **Explain relevance**: Why this expands oracle's domain knowledge
- **Connect to conversation**: How this addresses critical gap
- **Don't overwhelm**: Minimal expansion to fill specific gap

### Example Dynamic Learning Session

```
User: "Can you explain {specific advanced topic in domain}?"

Oracle: [Checks internal knowledge - finds gap]

Oracle internal: "I have foundational knowledge of {domain} but this
specific aspect needs deeper exploration. This is clearly within my
domain expertise. I'll do 1 focused search."

[Searches]
mcp__bright-data__search_engine(
    query="site:.edu {domain} {specific topic} scholarly"
)

[Finds article]
mcp__bright-data__scrape_as_markdown(
    url="https://university.edu/dept/{topic}"
)

Oracle response:
"Based on my foundational knowledge and recent scholarly research
[University Article](https://...), {specific topic} in {domain}
involves...

**Dynamic Knowledge Addition**
From [Title](URL) (accessed 2025-01-28):
This research by Prof. X expands our understanding by showing...

This directly enhances my expertise in {domain} regarding {topic}."
```

### Self-Regulation

**Oracle MUST ask itself before expanding:**

1. **"Did user trigger this?"**
   - User explicitly requested? → Yes, proceed
   - User revealed critical gap? → Yes, proceed
   - Oracle just wants more info? → **NO, don't expand**

2. **"Can I answer with existing knowledge?"**
   - If yes → **Use existing knowledge, don't expand**
   - If no → Only then consider expansion

3. **"Have I already expanded in this conversation?"**
   - If yes (1-2 times) → **DON'T expand again**
   - If no → Proceed with caution

4. **"Is this in my core domain?"**
   - If yes → potentially appropriate
   - If no → don't expand

5. **"Is there a reputable source?"**
   - If yes → use it
   - If no → don't expand

6. **"Does this fill a critical gap?"**
   - If yes → expand with citation
   - If no → **use existing knowledge**

**Default answer: DON'T EXPAND. Use what you already know.**

### Reasonable Exceptions

**Broad knowledge that relates back** to oracle's purpose is acceptable:

**Example 1**: Philosophy oracle expanding into cognitive science
```
john-vervaeke-oracle discussing relevance realization
→ User asks about neuroscience of attention
→ Oracle: "While cognitive neuroscience is adjacent to my core
   philosophy expertise, it directly informs RR framework..."
→ Search: "site:.edu neuroscience attention relevance"
→ Integrate: Relates findings back to RR theory
→ Acceptable because it enriches philosophical framework
```

**Example 2**: Technical oracle expanding into related applications
```
ovis-2-5-oracle discussing vision-language models
→ User asks about medical imaging applications
→ Oracle: "Medical imaging applies vision-language principles..."
→ Search: "site:.edu medical imaging vision language models"
→ Integrate: Shows how VLM architecture applies
→ Acceptable because demonstrates domain application
```

**Example 3**: Historical oracle expanding into modern scholarship
```
ancient-greek-philosophy-oracle discussing Plato's Forms
→ User asks about contemporary interpretations
→ Oracle: "Modern scholarship has reexamined Forms theory..."
→ Search: "site:plato.stanford.edu Forms contemporary"
→ Integrate: Updates classical knowledge with modern analysis
→ Acceptable because enriches core domain expertise
```

### File Naming Convention for Dynamic Additions

**Pattern**: `{parent-number}-{sub-number}-{topic-slug}-{YYYYMMDD}.md`

**Examples:**

```
Original Files:
concepts/00-theory-of-forms.md
concepts/01-aristotelian-metaphysics.md

Dynamic Additions:
concepts/00-1-contemporary-interpretations-20250128.md  (expands 00)
concepts/00-2-plato-modern-critique-20250202.md         (also expands 00)
concepts/01-1-four-causes-detail-20250128.md            (expands 01)

Directory listing shows relationship:
00-theory-of-forms.md
00-1-contemporary-interpretations-20250128.md  ← clearly related to 00
00-2-plato-modern-critique-20250202.md         ← also related to 00
01-aristotelian-metaphysics.md
01-1-four-causes-detail-20250128.md            ← clearly related to 01
```

**Why this naming?**
- **Parent number** (`00-`, `01-`) shows which topic it expands
- **Sub-number** (`-1-`, `-2-`) shows order of additions
- **Topic slug** describes what was added
- **Date** (`20250128`) shows when it was added
- Sorts naturally in file listings
- Clear visual grouping with parent topics

**Alternative: Use `dynamic/` subdirectory** (if many additions):

```
concepts/
├── 00-theory-of-forms.md
├── 00-dynamic/
│   ├── 01-contemporary-interpretations-20250128.md
│   └── 02-modern-critique-20250202.md
├── 01-aristotelian-metaphysics.md
└── 01-dynamic/
    └── 01-four-causes-detail-20250128.md
```

### Updating INDEX.md

**After adding dynamic knowledge, update INDEX.md:**

```markdown
## Dynamic Knowledge Additions

Files added during conversations:

### Concepts
- [00-1-contemporary-interpretations-20250128.md](concepts/00-1-contemporary-interpretations-20250128.md)
  - Parent: [Theory of Forms](concepts/00-theory-of-forms.md)
  - Added: 2025-01-28
  - Reason: User asked about modern interpretations

- [01-1-four-causes-detail-20250128.md](concepts/01-1-four-causes-detail-20250128.md)
  - Parent: [Aristotelian Metaphysics](concepts/01-aristotelian-metaphysics.md)
  - Added: 2025-01-28
  - Reason: Needed deeper explanation of efficient cause
```

### Meta-Learning Record

**In SKILL.md, add section tracking expansions:**

```markdown
## Knowledge Expansion History

**2025-01-28**:
- Added [contemporary-interpretations](concepts/00-1-contemporary-interpretations-20250128.md)
  - Source: Stanford Encyclopedia
  - Reason: User discussion of modern Plato scholarship
  - Integration: Enriched Theory of Forms understanding

**2025-02-02**:
- Added [modern-critique](concepts/00-2-plato-modern-critique-20250202.md)
  - Source: University of Oxford Philosophy Dept
  - Reason: Debate about Forms ontology
  - Integration: Added critical perspectives
```

---

**Remember**: Oracles are **living knowledge bases** that grow judiciously during conversations while staying true to their domain expertise. Expand thoughtfully, cite properly, and maintain domain focus.
