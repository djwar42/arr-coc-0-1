# Auto-Ingestion Folder

This folder is for **automatic** knowledge ingestion during Oracle Knowledge Expansion (dynamic learning). The oracle downloads and processes web content here temporarily.

## Workflow

1. **User requests**: "Research DataTable widget and add documentation"
2. **Oracle downloads**: Web pages, docs saved to `_ingest-auto/`
3. **Oracle processes**: Converts to markdown, extracts knowledge
4. **Oracle organizes**: Files moved to proper oracle locations
5. **Folder cleaned**: `_ingest-auto/` returns to empty state

## Keep This Folder Clean

This folder should be **empty** except during active knowledge expansion. All files are temporary and get moved to their final locations.

## NOT for Manual Use

For **manual** document ingestion, use the `_ingest/` folder instead.

## Oracle Knowledge Expansion

This folder supports the Oracle Knowledge Expansion process where the oracle autonomously:
- Researches topics from web sources
- Downloads Textual documentation
- Extracts relevant information
- Organizes into knowledge structure
- Commits updates with descriptive messages

**Trigger with**: "Research [topic] and add to knowledge base"
