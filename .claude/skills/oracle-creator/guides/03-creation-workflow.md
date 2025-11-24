# Oracle Creation Workflow

Step-by-step process for creating domain expert oracles.

## Overview

**Input**: Source documents + oracle specialization
**Output**: Complete self-contained oracle skill
**Time**: 30-60 minutes depending on source complexity

## Phase 1: Requirements Gathering

### Step 1.1: Oracle Name

**Ask user for oracle name:**
```
"What should we call this oracle?"
```

**Rules:**
- Lowercase with hyphens
- Descriptive (not abbreviated)
- Ends with `-oracle`

**Examples:**
- `ancient-greek-philosophy-oracle` ✅
- `quantum-mechanics-oracle` ✅
- `astronomy-oracle` ✅

### Step 1.2: Source Documents

**Ask user for sources:**
```
"Where are the source documents?
Can be:
- Directory path (e.g., ~/papers/philosophy/)
- Specific files (e.g., ~/docs/republic.pdf, ~/docs/symposium.pdf)
- Multiple directories"
```

**Supported formats:**
- PDF files
- Markdown files
- Text files
- Web pages (via URL)

### Step 1.3: Oracle Specialization

**Ask user what oracle knows:**
```
"What domain expertise should this oracle have?
Be specific about topics, key figures, and scope.

Example: 'This oracle knows Plato's theory of Forms, Aristotle's
metaphysics, Socratic method, and Pre-Socratic philosophy'"
```

**Use this for:**
- SKILL.md description
- File organization
- Navigation design

### Step 1.4: Confirm Requirements

**Summary for user:**
```
Creating: {oracle-name}
Sources: {N} files from {directory}
Specialization: {description}

Proceed? (yes/no)
```

## Phase 2: Document Conversion

### Step 2.1: Scan Source Directory

**List all source files:**
```python
import os
from pathlib import Path

source_dir = Path(user_provided_path)
pdf_files = list(source_dir.glob("**/*.pdf"))
md_files = list(source_dir.glob("**/*.md"))
txt_files = list(source_dir.glob("**/*.txt"))

print(f"Found: {len(pdf_files)} PDFs, {len(md_files)} MDs, {len(txt_files)} TXTs")
```

### Step 2.2: Copy Originals and Convert to Markdown

**CRITICAL: Keep both original and markdown versions**

**For each PDF file:**

1. **Copy original PDF**
   ```python
   # Copy PDF with numbered prefix
   shutil.copy(pdf_file, f"source-documents/{index:02d}-{pdf_file.stem}.pdf")
   ```

2. **Convert to markdown**
   ```python
   # Option A: Use Bright Data MCP (if available)
   mcp__bright-data__extract(
       url=pdf_url,
       extraction_prompt="Extract all text content, preserve structure"
   )

   # Option B: Use PDF Parser
   content_md = convert_pdf_to_markdown(pdf_path)

   # Save markdown version with same number
   with open(f"source-documents/{index:02d}-{pdf_file.stem}.md", "w") as f:
       f.write(content_md)
   ```

**For each existing markdown file:**
```python
# Copy directly with numbered prefix
shutil.copy(md_file, f"source-documents/{index:02d}-{md_file.stem}.md")
```

**Output:**
```
source-documents/
├── 00-republic.pdf        # Original PDF preserved
├── 00-republic.md         # Converted markdown for reading
├── 01-symposium.pdf       # Original PDF preserved
├── 01-symposium.md        # Converted markdown for reading
└── 02-ethics.md           # Already markdown (copied)
```

**Why keep originals?**
- Verification and reference
- Preserves formatting that may be lost in conversion
- Safe keeping for future use

### Step 2.3: Copy Codebases Wholesale (If Present)

**If source directory contains full codebases:**

```python
import shutil
from pathlib import Path

# Identify codebases (directories with src/, package.json, setup.py, etc.)
def is_codebase(path):
    """Check if directory is a codebase"""
    indicators = ['src/', 'package.json', 'setup.py', 'Cargo.toml', 'go.mod']
    return any((path / indicator).exists() for indicator in indicators)

# Copy codebases
codebases = [d for d in source_dir.iterdir() if d.is_dir() and is_codebase(d)]

for idx, codebase_dir in enumerate(codebases):
    dest = oracle_dir / f"source-codebases/{idx:02d}-{codebase_dir.name}"

    # Copy entire codebase wholesale
    shutil.copytree(codebase_dir, dest)
    print(f"✓ Copied codebase: {codebase_dir.name} → {dest}")
```

**Output:**
```
source-codebases/
├── 00-ml-training-pipeline/    # Full Python project
│   ├── src/
│   ├── tests/
│   ├── setup.py
│   └── README.md
└── 01-web-api/                 # Full Node.js project
    ├── src/
    ├── package.json
    └── README.md
```

**Why copy wholesale?**
- Preserves complete project structure
- Maintains working code that can be referenced
- Allows running/testing if needed
- Full context for understanding architecture

### Step 2.4: Deduplicate Markdown Files

**CRITICAL: Detect and remove duplicate markdown files**

Source directories may contain duplicates. We only want one copy.

```python
import hashlib
from pathlib import Path

def compute_file_hash(file_path):
    """Compute SHA-256 hash of file content"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def deduplicate_source_documents(source_dir):
    """Remove duplicate markdown files, keep first occurrence"""
    seen_hashes = {}
    duplicates = []

    # Get all markdown files in source-documents/
    md_files = sorted(source_dir.glob("*.md"))

    for md_file in md_files:
        file_hash = compute_file_hash(md_file)

        if file_hash in seen_hashes:
            # Duplicate found!
            original = seen_hashes[file_hash]
            print(f"⚠️  Duplicate: {md_file.name} is identical to {original.name}")
            print(f"   Removing duplicate: {md_file.name}")
            md_file.unlink()  # Delete duplicate
            duplicates.append((md_file.name, original.name))
        else:
            seen_hashes[file_hash] = md_file

    # Renumber remaining files
    renumber_files(source_dir)

    return duplicates

# Run deduplication
duplicates = deduplicate_source_documents(oracle_dir / "source-documents")

if duplicates:
    print(f"\n✓ Removed {len(duplicates)} duplicate markdown files")
    for dup, original in duplicates:
        print(f"  - {dup} → kept {original}")
else:
    print("\n✓ No duplicate markdown files found")
```

**Why deduplicate?**
- Source folders may have multiple copies of same document
- Avoid confusion with duplicate content
- Keep oracle clean and efficient
- Only keep first occurrence (by numbered order)

**Example:**
```
Before deduplication:
00-paper.md (content: "ABC...")
01-paper-copy.md (content: "ABC...")  # Same content!
02-notes.md

After deduplication:
00-paper.md (kept - first occurrence)
01-notes.md (renumbered from 02)
```

### Step 2.5: Verify Conversion

**Check each converted file:**
- Markdown is valid
- Formatting preserved
- No encoding errors
- Headings structure makes sense
- No duplicates remain

## Phase 3: Read Sources and Extract Organization

### Step 3.1: Read All Source Markdown Files

**CRITICAL: Read through ALL source documents before organizing**

```python
# Read all markdown sources
for md_file in sorted(Path("source-documents").glob("*.md")):
    print(f"\n=== Reading {md_file.name} ===")
    content = md_file.read_text()
    # Analyze content, take notes on themes
    print(f"Length: {len(content)} chars")
    print(f"Major sections: {identify_sections(content)}")
```

**During reading, identify:**
- Major themes/concepts that span multiple sources
- Natural groupings of related topics
- Key figures/topics mentioned repeatedly
- Hierarchical relationships between concepts

**Example for ancient-greek-philosophy-oracle:**

Reading `source-documents/00-republic.md`:
- Found: Theory of Forms, justice, ideal state, philosopher kings
- Major theme: Political philosophy + metaphysics

Reading `source-documents/01-symposium.md`:
- Found: Theory of love, beauty, Forms
- Major theme: Love and ascent to Forms

Reading `source-documents/02-nicomachean-ethics.md`:
- Found: Virtue, happiness, golden mean, practical wisdom
- Major theme: Ethical theory

**Analysis result:**
- **Topic folder needed**: `philosophers/` (Socrates, Plato, Aristotle profiles)
- **Topic folder needed**: `concepts/` (Forms, virtue, metaphysics - cross-cutting themes)
- **Topic folder needed**: `texts/` (Detailed analyses of specific works)
- **Topic folder needed**: `guides/` (How to approach the material)

### Step 3.2: Plan Topic Organization

**Based on source reading, plan structure:**

```
ancient-greek-philosophy-oracle/
├── source-documents/         # Already complete with originals + MD
├── philosophers/             # Extract: Who are the key thinkers?
├── concepts/                 # Extract: What are the major ideas?
├── texts/                    # Extract: Deep dives on specific works
└── guides/                   # Extract: How to engage with material
```

### Step 3.3: Codebase Commenting (If Present)

**If codebases were copied, add Claude's code comments**

**IMPORTANT: Don't go overboard - natural introduction only**

**Step A: Create Top-Level Overview**

```python
# Analyze codebase structure
codebase_path = Path("source-codebases/00-ml-training-pipeline")

overview = f"""# {codebase_path.name} - Codebase Overview

## Architecture

{analyze_architecture(codebase_path)}

## Directory Structure

```
{generate_tree(codebase_path, max_depth=2)}
```

## Core Components

{identify_core_components(codebase_path)}

## Key Files

- `src/main.py` - Entry point
- `src/training/pipeline.py` - Training orchestration
- `src/models/architecture.py` - Model definitions

## Technology Stack

{identify_tech_stack(codebase_path)}
"""

# Save overview
(oracle_dir / "codebase/00-overview.md").write_text(overview)
```

**Step B: Add Comments to Core Files (3-5 files max)**

```python
# Identify core files (not all files!)
def identify_core_files(codebase_path):
    """Return 3-5 most important files"""
    candidates = []

    # Main entry points
    for entry in ['main.py', 'app.py', 'index.js', 'main.go']:
        if (codebase_path / 'src' / entry).exists():
            candidates.append(f'src/{entry}')

    # Core modules (use heuristics: file size, imports, etc.)
    # Don't add every file!

    return candidates[:5]  # Max 5 files

# Add Claude's code comments to core files
for file_path in core_files:
    add_claudes_code_comments_to_file(
        codebase_path / file_path,
        oracle_dir / f"codebase/{idx:02d}-{file_path.stem}.md"
    )
```

**Example: Core File Commentary**

```markdown
# pipeline.py - Training Pipeline

<claudes_code_comments>
** Function List **
- train_model(config): orchestrates complete training workflow
- load_dataset(path): loads and preprocesses training data
- validate_results(metrics): validates training metrics

** Technical Review **
Main training orchestration module. Flow: load_dataset() →
initialize_model() → train_model() → validate_results().
Uses distributed training with PyTorch DDP. Configuration
loaded from YAML files. Checkpointing every 1000 steps.
</claudes_code_comments>

{Include actual file content with comments}
```

**Step C: Create File Index**

```python
# Generate complete file listing
file_index = generate_file_index(codebase_path)

index_content = f"""# File Index - {codebase_path.name}

## All Files

{file_index}

## Commented Files

Files with Claude's code comments:
- [00-overview.md](00-overview.md) - Architecture overview
- [01-main.md](01-main.md) - Entry point analysis
- [02-pipeline.md](02-pipeline.md) - Training pipeline
"""

(oracle_dir / "codebase/03-file-index.md").write_text(index_content)
```

**Guidelines:**
- **Top-level overview**: Always create
- **Core files commented**: 3-5 major files only
- **Don't comment**: Config files, trivial utilities, tests (unless critical)
- **File index**: Complete listing for reference
- **Deeper exploration**: Only if user requests later

**Output Structure:**
```
codebase/
├── 00-overview.md              # Architecture & structure
├── 01-main-py.md               # Entry point with comments
├── 02-pipeline-py.md           # Core module with comments
├── 03-architecture-py.md       # Another core module
└── 04-file-index.md            # Complete file listing
```

### Step 3.4: Web Research Enrichment (Judicious)

**IMPORTANT: Use Bright Data to supplement source material wisely**

**When to use web research:**
- Key concepts need more scholarly context
- Historical background is sparse in sources
- Contemporary interpretations would add value
- Technical details need clarification

**How to use Bright Data MCP:**

```python
# 1. Search for scholarly articles on key topics
results = mcp__bright-data__search_engine(
    query="Plato theory of Forms Stanford Encyclopedia"
)

# 2. Scrape reputable sources
sep_article = mcp__bright-data__scrape_as_markdown(
    url="https://plato.stanford.edu/entries/plato-forms/"
)

iep_article = mcp__bright-data__scrape_as_markdown(
    url="https://iep.utm.edu/platonic-form/"
)

# 3. Save to supplementary folder
with open("web-research/00-sep-plato-forms.md", "w") as f:
    f.write(f"# Stanford Encyclopedia: Plato's Forms\n\n")
    f.write(f"**Source**: https://plato.stanford.edu/entries/plato-forms/\n\n")
    f.write(sep_article)
```

**Structure for web research:**
```
.claude/skills/{oracle-name}/
├── source-documents/         # Original sources
├── web-research/             # Supplementary web sources
│   ├── 00-sep-plato-forms.md
│   ├── 01-iep-aristotle-metaphysics.md
│   └── 02-scholar-article.md
└── ...
```

**Best practices:**
- **Be selective** - Don't scrape dozens of articles
- **Use reputable sources** - Stanford Encyclopedia, Internet Encyclopedia of Philosophy, .edu sites
- **Link with context** - Explain what each web source adds
- **Cite properly** - Include full URL and access date

**Example searches:**
```python
# For philosophy
mcp__bright-data__search_engine(query="site:plato.stanford.edu {topic}")

# For science
mcp__bright-data__search_engine(query="site:.edu quantum mechanics {topic}")

# For technical topics
mcp__bright-data__search_engine(query="site:arxiv.org {topic} review")
```

**What NOT to do:**
- ❌ Don't scrape entire websites
- ❌ Don't use unreliable sources (random blogs, etc.)
- ❌ Don't add tangential information
- ❌ Don't go overboard - 3-5 web sources per oracle typically enough

### Step 3.5: Create Directory Structure

**Common patterns:**

**Academic Oracle (person-focused):**
```
{oracle-name}/
├── papers/           # Academic publications
├── concepts/         # Key ideas
└── {Name}-Application-Guide.md
```

**Technical Oracle (topic-focused):**
```
{oracle-name}/
├── architecture/     # System design
├── codebase/         # Code docs
├── usage/            # Practical guides
└── examples/         # Working code
```

**Domain Oracle (knowledge-focused):**
```
{oracle-name}/
├── {subdomain1}/     # Major area
├── {subdomain2}/     # Major area
├── concepts/         # Cross-cutting concepts
└── guides/           # How-tos
```

**For our example:**
```python
oracle_dir = Path(f".claude/skills/{oracle_name}")
oracle_dir.mkdir(exist_ok=True)

(oracle_dir / "philosophers").mkdir()
(oracle_dir / "concepts").mkdir()
(oracle_dir / "texts").mkdir()
(oracle_dir / "guides").mkdir()
(oracle_dir / "source-documents").mkdir()
```

### Step 3.6: Extract Content into Topic Folders

**For each topic folder, extract organized content from sources + web research**

**Example: Extract `concepts/00-theory-of-forms.md`**

1. **Read relevant sections from sources**
   - `source-documents/00-republic.md` - Books 5-7 on Forms
   - `source-documents/01-symposium.md` - Diotima's speech on Beauty

2. **Extract and synthesize content**
   ```python
   # Extract relevant sections
   republic_forms = extract_sections(
       "source-documents/00-republic.md",
       ["Book V", "Book VI", "Book VII"]
   )

   symposium_beauty = extract_sections(
       "source-documents/01-symposium.md",
       ["Diotima's speech"]
   )
   ```

3. **Create organized exposition**
   ```markdown
   # Theory of Forms

   ## Overview

   Plato's Theory of Forms posits that abstract Forms (or Ideas) are
   more real than physical particulars we perceive with our senses.
   {Synthesized from sources}

   ## Detailed Explanation

   ### Forms vs Particulars

   {Content extracted and organized from sources}

   ### Participation and Imitation

   {Content extracted and organized from sources}

   ## Primary Sources

   From [Republic](../source-documents/00-republic.md):
   - Book V: Introduction to Forms (lines 450-520)
   - Book VI: The divided line analogy (lines 509d-511e)
   - Book VII: Allegory of the cave (lines 514a-520a)

   From [Symposium](../source-documents/01-symposium.md):
   - Diotima's speech: Ascent to the Form of Beauty (lines 210a-212a)

   ## Additional Research (Web Sources)

   Supplementary web research for context:
   - [Stanford Encyclopedia - Plato's Theory of Forms](https://plato.stanford.edu/entries/plato-forms/)
     - Added: Contemporary scholarly interpretations
     - Added: Historical development of the theory
   - [Internet Encyclopedia of Philosophy](https://iep.utm.edu/platonic-form/)
     - Added: Criticisms and responses

   ## Related Concepts

   - [Epistemology](03-epistemology.md) - How Forms relate to knowledge
   - [Metaphysics](01-aristotelian-metaphysics.md) - Aristotle's critique
   ```

**CRITICAL: Every extracted file must cite sources**
- Note which source document content came from
- Provide line numbers or section references
- Link back to source markdown files
- Include web research with URLs
- Maintain full traceability

### Step 3.7: Create Guides and Supporting Content

**For how-to or practical content extracted from sources:**

```markdown
# Understanding Plato's Dialogues

## Purpose

This guide helps you navigate Plato's dialogue form and extract
philosophical arguments effectively.

## The Dialogue Form

{Extracted from studying multiple source texts}

## Reading Strategies

{Synthesized from introduction sections of sources}

## Primary Sources

This guide synthesizes material from:
- [Republic](../source-documents/00-republic.md) - Introduction
- [Symposium](../source-documents/01-symposium.md) - Preface

## Additional Research

Web sources consulted:
- [How to Read Plato](https://philosophy.example.edu/reading-plato)
  - Added: Modern reading strategies

## Further Reading

{Connections to concept files}
```

### Step 3.8: Verify All Files Reference Sources

**Check every extracted file has "Primary Sources" section:**

```python
def verify_source_references(oracle_dir):
    """Ensure all extracted files cite sources"""
    for md_file in oracle_dir.glob("*/*.md"):
        # Skip source-documents themselves
        if md_file.parent.name == "source-documents":
            continue

        content = md_file.read_text()

        # Check for source citations
        if "Primary Sources" not in content:
            print(f"⚠️  Missing sources: {md_file}")

        if "../source-documents/" not in content:
            print(f"⚠️  No source links: {md_file}")

        # Web sources are optional but should be noted if present
        if "Additional Research" in content:
            if "http" not in content:
                print(f"⚠️  Web research section but no URLs: {md_file}")
```

### Step 3.9: Number All Files

**Ensure all files have numbered prefixes:**
```
00-overview.md
01-fundamentals.md
02-intermediate.md
03-advanced.md
```

**Numbering strategy:**
- `00-` for overviews
- Sequential for logical progression
- Leave gaps (00, 10, 20) if planning to add more

## Phase 4: Generate Metadata

### Step 4.1: Create SKILL.md

**Use template from `guides/01-skill-md-template.md`**

**Key sections to fill:**
1. YAML frontmatter
2. What This Skill Provides
3. Directory Structure (actual tree)
4. When to Use (map questions to files)
5. Detailed File Descriptions (every file)
6. Quick Navigation

**Generate SKILL.md programmatically:**
```python
# Read all files
files = list(oracle_dir.glob("**/*.md"))
files.remove(oracle_dir / "SKILL.md")  # Exclude self
files.remove(oracle_dir / "INDEX.md")   # Exclude INDEX

# Generate file descriptions
for file in sorted(files):
    line_count = len(file.read_text().splitlines())
    category = file.parent.name
    # Add to SKILL.md...

# Generate navigation
for category in categories:
    # Map topics to files...

# Generate SKILL.md from template
```

### Step 4.2: Create INDEX.md

**Use template from `guides/02-index-template.md`**

**Key sections:**
1. Quick Navigation
2. Document Structure
3. Topic Index (by category)
4. Usage Examples
5. Cross-Reference Map
6. File Summary

**Generate INDEX.md programmatically:**
```python
# Generate topic index
for category in categories:
    files_in_category = sorted((oracle_dir / category).glob("*.md"))
    # Create table...

# Generate cross-reference map
# Track which concepts appear in which files

# Generate file summary
for file in all_files:
    line_count = len(file.read_text().splitlines())
    # Add to summary table...
```

### Step 4.3: Validate Metadata

**Check SKILL.md:**
- [ ] YAML frontmatter is valid
- [ ] Description is under 1024 chars
- [ ] All files are described
- [ ] Navigation links work

**Check INDEX.md:**
- [ ] All files are indexed
- [ ] Cross-references are accurate
- [ ] Usage examples make sense

## Phase 5: Quality Assurance

### Step 5.1: Validate Structure

**Check directory structure:**
```python
# Verify all expected directories exist
# Verify all files have numbered prefixes
# Verify no orphaned files
```

### Step 5.2: Validate Cross-References

**Check all markdown links:**
```python
import re

def find_broken_links(oracle_dir):
    broken = []
    for md_file in oracle_dir.glob("**/*.md"):
        content = md_file.read_text()
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        for link_text, link_path in links:
            if link_path.startswith('http'):
                continue  # Skip external links
            target = (md_file.parent / link_path).resolve()
            if not target.exists():
                broken.append((md_file, link_path))
    return broken
```

### Step 5.3: Validate Content

**Check each file:**
- [ ] Valid markdown syntax
- [ ] Headings use proper levels
- [ ] No encoding errors
- [ ] Reasonable length (not empty, not too long)

### Step 5.4: Test Oracle Description

**Verify SKILL.md description triggers appropriately:**
```
description: "... Use when questions involve {key topic}, {key topic}, {key topic} ..."
```

**Test questions:**
- Should activate: "{key topic} question"
- Should activate: "{key topic} related question"
- Should not activate: Unrelated question

## Phase 6: Deployment

### Step 6.1: Review with User

**Show oracle structure:**
```
Oracle: {oracle-name}
Files: {N} total
  - {N1} source documents
  - {N2} concept docs
  - {N3} guides
  - {N4} other

Directory structure:
{tree view}

Ready to deploy?
```

### Step 6.2: Deploy Oracle

**Oracle already in correct location:**
```
.claude/skills/{oracle-name}/
```

**If created elsewhere, move it:**
```python
import shutil

shutil.copytree(
    temp_oracle_dir,
    f".claude/skills/{oracle_name}",
    dirs_exist_ok=True
)
```

### Step 6.3: Git Commit

**Commit oracle to repository:**
```bash
git add .claude/skills/{oracle-name}
git commit -m "Add {oracle-name} skill

- Converted {N} source documents to markdown
- Organized into {categories}
- Generated comprehensive SKILL.md and INDEX.md
- Oracle knows: {specialization brief}
"
```

### Step 6.4: Inform User

**Success message:**
```
✅ {oracle-name} created successfully!

Location: .claude/skills/{oracle-name}/
Files: {N} total ({N1} sources, {N2} concepts, {N3} guides)
Size: ~{X} lines of knowledge

To use:
1. Restart Claude Code to load the skill
2. Ask questions like:
   - "{example question 1}"
   - "{example question 2}"
3. The oracle will activate automatically

Test it now? (Restart required)
```

## Phase 7: Testing

### Step 7.1: Restart Claude Code

**User must restart:**
```
Please restart Claude Code to load the new oracle skill.
```

### Step 7.2: Test Activation

**Ask test questions:**
```
User: "{question in oracle domain}"

Expected: Oracle activates, reads relevant files, provides answer
```

### Step 7.3: Test Navigation

**Verify file access:**
- SKILL.md navigation works
- INDEX.md links work
- Cross-references are correct

### Step 7.4: Test Knowledge

**Ask challenging questions:**
- Deep questions requiring multiple files
- Questions requiring concept synthesis
- Questions about connections

## Common Issues and Solutions

### Issue: PDF Conversion Quality

**Problem**: PDFs convert with formatting issues

**Solutions:**
- Try different PDF parser
- Manual cleanup of converted markdown
- Use OCR for scanned documents

### Issue: Too Many Files

**Problem**: Source directory has 100+ documents

**Solutions:**
- Organize by subdomain
- Create multiple related oracles
- Use source-documents/ for originals, create summary docs

### Issue: Unclear Organization

**Problem**: Hard to determine categories

**Solutions:**
- Start with source-documents/ only
- Add categories as patterns emerge
- Review existing oracles for ideas

### Issue: Description Too Long

**Problem**: SKILL.md description exceeds 1024 chars

**Solutions:**
- Focus on trigger keywords
- Remove redundant phrases
- Split into multiple sentences

## Workflow Summary

```
1. Gather Requirements
   → Oracle name, sources, specialization

2. Convert Documents
   → PDFs/docs to markdown
   → Numbered in source-documents/

3. Organize Content
   → Create categories
   → Extract concepts
   → Create guides
   → Number everything

4. Generate Metadata
   → SKILL.md from template
   → INDEX.md from template
   → Validate all links

5. Quality Assurance
   → Check structure
   → Validate cross-refs
   → Test content

6. Deploy
   → Review with user
   → Git commit
   → Inform user

7. Test
   → Restart Claude Code
   → Test activation
   → Verify knowledge
```

## Tips for Success

### Content Extraction

- Read sources carefully before organizing
- Identify natural categories
- Group related concepts
- Maintain source traceability

### File Organization

- Use descriptive names
- Number for logical flow
- Leave room for expansion
- Cross-reference heavily

### Metadata Generation

- Be thorough in SKILL.md
- Make INDEX.md comprehensive
- Test all navigation paths
- Use templates as guides

### Quality Focus

- Validate every link
- Check every file
- Test oracle activation
- Iterate based on feedback

---

Use this workflow when creating new oracles. Adapt steps as needed for specific oracle types and content.
