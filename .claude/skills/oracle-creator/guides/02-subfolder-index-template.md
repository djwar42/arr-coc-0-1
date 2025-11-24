# Subfolder INDEX.md Template

Template for creating topic subfolder INDEX.md files.

## Purpose

Each topic folder has its own INDEX.md that:
- Lists all files in the folder with descriptions and keywords
- Provides quick navigation within the topic
- Links to related folders
- Helps oracle find specific files by keyword

**⚠️ NO ROOT INDEX.md** - Oracles use distributed subfolder indexes instead!

## Template Structure

```markdown
# {Folder Name} - Index

**{Brief description of what this topic folder covers}**

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-overview.md` | Introduction and getting started | basics, overview, introduction |
| `01-core-concepts.md` | Fundamental concepts explained | fundamentals, core, theory |
| `02-practical-guide.md` | Hands-on implementation | practical, tutorial, howto |
| `03-advanced-topics.md` | Advanced techniques | advanced, optimization, production |

## Quick Start

Start with `00-overview.md` for orientation.

For practical implementation, see `02-practical-guide.md`.

## Cross-References

**Related folders:**
- `../related-topic/` - Related concepts
- `../applications/` - Practical applications
- `../source-documents/` - Original sources

**See also:**
- `../other-folder/specific-file.md` - Related deep-dive
```

## Complete Example

Example INDEX.md for `karpathy/gradio/` folder:

```markdown
# Gradio Development - Index

**Gradio as a development microscope for VLM testing and W&B integration**

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-core-testing-patterns.md` | Development microscope pattern | testing, debugging, inspection |
| `01-state-management.md` | Gradio state and memory | state, memory, session |
| `02-multi-model-comparison.md` | A/B testing checkpoints | comparison, checkpoints, evaluation |
| `03-lru-checkpoint-manager.md` | Memory-efficient loading | LRU, memory, GPU, T4, A100 |
| `10-wandb-integration-basics.md` | Basic W&B setup | wandb, logging, tracking |
| `11-wandb-huggingface-trainer.md` | HuggingFace Trainer + W&B | trainer, huggingface, integration |
| `12-wandb-pytorch-manual.md` | Manual PyTorch + W&B | pytorch, manual, training loop |

## Quick Start

Start with `00-core-testing-patterns.md` for the development microscope concept.

For W&B integration, see `10-wandb-integration-basics.md`.

## Cross-References

**Related folders:**
- `../practical-implementation/` - W&B Launch and automation
- `../vlm-research/` - Latest VLM techniques to test
- `../../source-codebases/karpathy/` - Reference implementations
```

## Keywords Best Practices

**Good keywords:**
- Specific technical terms: `PagedAttention`, `FlashAttention`, `KV-cache`
- Common synonyms: `optimization`, `efficiency`, `speed`
- Use cases: `production`, `deployment`, `inference`
- Framework names: `pytorch`, `huggingface`, `wandb`

**Bad keywords:**
- Too generic: `good`, `important`, `useful`
- Too specific: File names, line numbers
- Redundant: Keywords already in file name

## File Table Best Practices

1. **Order files by number prefix** (00, 01, 02...)
2. **Keep descriptions concise** (5-15 words)
3. **Include 3-5 keywords** per file
4. **Use lowercase keywords** for consistency
5. **Use backticks around file names** for clarity

## Cross-Reference Best Practices

1. **Always use relative paths** (`../folder/file.md`)
2. **Link to folders, not just files** (`../topic/` for general reference)
3. **Be specific when helpful** (`../source-documents/00-paper.md` for direct citation)
4. **Group related folders** under clear headings

## When to Create Subfolder INDEX.md

Create INDEX.md when a folder has:
- 3+ files (worth indexing)
- Files that serve different purposes (overview vs advanced)
- Content that benefits from keyword search

Don't create INDEX.md for:
- Single-file folders
- Source document folders (just list files directly)
- Temporary/work folders (`_ingest/`, `_ingest-auto/`)

## Integration with SKILL.md

SKILL.md routes to folders, not individual files:

```markdown
### When to Use This Oracle

"How do I build a Gradio interface for VLMs?"
→ **See**: `karpathy/gradio/` (check INDEX.md for specific file)

"W&B integration with HuggingFace Trainer?"
→ **See**: `karpathy/gradio/` (files 10-12 cover W&B integration)
```

Oracle workflow:
1. Read routing in SKILL.md
2. Read folder's INDEX.md
3. Find file by keyword match
4. Read specific knowledge file

## Minimal INDEX.md

For small folders (3-5 files), keep it simple:

```markdown
# CUDA Knowledge - Index

**CUDA programming and PyTorch compilation expertise**

## Files

| File | Description | Keywords |
|------|-------------|----------|
| `00-streams-concurrency.md` | Async execution | streams, async, overlap |
| `01-memory-management.md` | Unified memory | memory, pinned, UVM |
| `02-tensor-cores.md` | Tensor Core programming | WMMA, FP8, TF32 |

## Cross-References

Related: `../implementations/`, `../vllm-knowledge/`
```

## File Ordering

Number files by conceptual flow:
- `00-` Overview/introduction
- `01-03` Core concepts
- `04-09` Intermediate topics
- `10+` Advanced/specialized

Example:
```
00-overview.md           # Start here
01-fundamentals.md       # Basic concepts
02-architecture.md       # How it works
03-implementation.md     # How to use
10-advanced-patterns.md  # Advanced usage
11-optimization.md       # Performance tuning
```

---

Use this template when creating subfolder INDEX.md files. Every topic folder should have one!
