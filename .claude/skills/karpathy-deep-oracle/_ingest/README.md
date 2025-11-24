# Manual Ingestion Folder

This folder is for **manual** document ingestion. Drop PDF files, research papers, or source documents here for the oracle to process.

**Workflow:**
1. User drops files into `_ingest/`
2. Oracle analyzes documents
3. Converts to markdown knowledge
4. Moves to appropriate oracle locations
5. `_ingest/` cleaned up

**For automated knowledge expansion** (web research, etc.), the oracle uses `_ingest-auto/` instead.

## What to Put Here

✅ **PDF research papers** - Academic papers, technical reports
✅ **Documentation** - API docs, tutorials, guides
✅ **Articles** - Blog posts, technical articles
✅ **Code** - Source files to analyze

## What NOT to Put Here

❌ Random files - Keep it relevant to oracle expertise
❌ Large codebases - Use oracle knowledge expansion instead
❌ Temporary files - This is for permanent knowledge

**Keep this folder clean!** It should be empty except during active processing.
