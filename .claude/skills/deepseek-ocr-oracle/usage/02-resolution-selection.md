# Resolution Mode Selection Guide

## Decision Tree

```
Is text very large/simple? (slides, posters)
    → Tiny (73 tokens)

Is it a book, report, or chart?
    → Small (111 tokens)

Standard document?
    → Base (273 tokens) [DEFAULT]

Dense text, formulas, fine print?
    → Large (421 tokens)

Ultra-high-res or multi-column?
    → Gundam (variable)
```

## By Document Type

| Type | Mode | Why |
|------|------|-----|
| Presentation slides | Tiny | Large fonts, minimal text |
| Books/novels | Small | Standard text, simple layout |
| Business reports | Small-Base | Mixed content |
| Academic papers | Base-Large | Formulas, citations, dense |
| Legal documents | Large | Fine print, precision needed |
| Scanned newspapers | Gundam | Multi-column, high detail |
| Technical diagrams | Large | Preserve fine details |

## By Quality Requirement

**Speed priority**: Tiny/Small (8× faster than Large)
**Quality priority**: Large/Gundam
**Balanced**: Base (recommended default)

## By GPU Memory

| VRAM | Max Mode | Notes |
|------|----------|-------|
| 8GB | Tiny/Small | Batch size 1-2 |
| 12GB | Base | Batch size 2 |
| 16GB | Large | Batch size 1-2 |
| 24GB+ | Gundam | Batch size 1 |

## Quick Test Strategy

1. Start with **Base** (default)
2. If quality insufficient → **Large**
3. If too slow → **Small**
4. If still problems → **Gundam** or **Tiny**

## Code Examples

```python
# Automatic (Base default)
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"])

# Explicit mode selection
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"],
                    base_size=640, image_size=640)  # Small mode
```

**See Also**:
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - Technical details
- [quick-start.md](quick-start.md) - Usage examples
