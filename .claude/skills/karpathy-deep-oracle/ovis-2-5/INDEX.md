# Ovis 2.5 Knowledge - Overview

**Total**: 38 documentation files across 7 topic folders

## What is Ovis 2.5?

Ovis 2.5 is a multimodal vision-language model featuring:
- **Native resolution processing** (no fixed image size)
- **Visual Embedding Table (VET)** - probabilistic token generation
- **5-phase training pipeline**
- **Thinking mode** for complex reasoning

## Folder Structure

```
ovis-2-5/
├── 00-overview.md              # This file
├── architecture/               # Model architecture (8 files)
├── codebase/                   # Implementation details (6 files)
├── concepts/                   # Key concepts (6 files)
├── examples/                   # Code examples (5 files)
├── references/                 # API reference (2 files)
├── training/                   # 5-phase pipeline (7 files)
└── usage/                      # Quickstart & fine-tuning (4 files)
```

## Topic Folders

| Folder | Files | Contents |
|--------|-------|----------|
| `architecture/` | 8 | NaViT vision, VET, Qwen3 LLM, merging, thinking mode |
| `codebase/` | 6 | modeling_ovis.py analysis, implementation details |
| `concepts/` | 6 | Structural alignment, probabilistic VTE, native resolution |
| `examples/` | 5 | Basic inference, thinking mode, multi-image, video |
| `references/` | 2 | API reference, model config |
| `training/` | 7 | P1-VET, P2-Multimodal, P3-Instruction, P4/P5-RL |
| `usage/` | 4 | Quickstart, HuggingFace, advanced features, fine-tuning |

## Quick Navigation

**Understanding Ovis:**
- Start → `architecture/00-overview.md`
- Key innovation → `architecture/03-visual-embedding-table.md`
- Core concept → `concepts/00-structural-alignment.md`

**Using Ovis:**
- Quick start → `usage/00-quickstart.md`
- Fine-tuning → `usage/03-fine-tuning.md`
- Examples → `examples/00-basic-inference.md`

**Deep dive:**
- Implementation → `codebase/01-modeling-ovis.md`
- Training pipeline → `training/00-overview.md`

## Cross-References

**Source code**: `../source-codebases/deepseek/14-Ovis-2-5/`
**Original oracle**: Still available at `ovis-2-5-oracle` skill
**Related VLMs**: DeepSeek-VL2, Qwen3-VL
