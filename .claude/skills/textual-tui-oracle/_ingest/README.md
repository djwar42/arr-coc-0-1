# Manual Document Ingestion

Place new documents (PDFs, markdown, text files) here for manual ingestion into the textual-tui-oracle knowledge base.

## How to Use

1. **Drop files** in this folder:
   - PDFs from Textual documentation
   - Markdown files with examples
   - Text files with notes
   - Code snippets

2. **Instruct the oracle**:
   ```
   "Ingest the new documents in _ingest/"
   ```

3. **Oracle will**:
   - Activate oracle-creator automatically
   - Convert PDFs to markdown
   - Organize content into appropriate folders
   - Update INDEX.md and SKILL.md
   - Move processed files to source-documents/

## What to Add

**Useful additions:**
- Textual blog posts (releases, features)
- Community tutorials
- Widget documentation pages
- Advanced guide sections
- API reference pages
- Example applications
- Your own Textual notes/patterns

## Keep This Folder Clean

After ingestion, files are moved to appropriate locations. This folder should remain empty except during active processing.
