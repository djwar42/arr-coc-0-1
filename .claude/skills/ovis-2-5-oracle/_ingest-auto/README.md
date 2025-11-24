# Auto-Ingestion Folder

This folder is for **automatic** knowledge ingestion during dynamic learning. The oracle itself saves web-downloaded PDFs here temporarily during knowledge expansion.

**Workflow:**
1. Oracle downloads PDF during dynamic learning
2. PDF saved to _ingest-auto/
3. Oracle analyzes and converts to markdown
4. Files moved to proper oracle locations
5. _ingest-auto/ cleaned up

**Keep this folder clean and tidy** - it should be empty except during active processing.

For **manual** document ingestion, use the `_ingest/` folder instead.
