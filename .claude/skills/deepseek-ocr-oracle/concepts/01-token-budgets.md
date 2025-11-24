# Token Budgets: Why 73-421 Tokens?

## The Range

**Minimum**: 73 tokens (Tiny mode, 8×8 grid)
**Maximum**: 421 tokens (Large mode, 20×20 grid)
**Sweet spot**: 273 tokens (Base mode, 16×16 grid)

## Lower Bound (73 tokens)

**Why not less?**
- 8×8 = 64 patches minimum for text recognition
- Below this: Characters become unreadable
- Quality drops dramatically

**Use case**: Simple slides with large fonts only

## Upper Bound (421 tokens)

**Why not more?**
- Diminishing returns above 20×20 patches
- Computational cost increases quadratically
- Memory constraints
- Quality gains minimal

**Above 421**: Use Gundam mode (dynamic tiling)

## Sweet Spot (273 tokens)

**16×16 = 256 patches** + newlines = 273 tokens

**Why optimal?**
- Good quality for most documents
- Balanced compute/memory
- Empirically validated on benchmarks
- **Default choice** for DeepSeek-OCR

## Comparison with Other VLMs

| Model | Vision Tokens | Quality |
|-------|--------------|---------|
| Standard VLM | 7000+ | High |
| BLIP-2 | 32 (Q-Former) | Medium |
| **DeepSeek-OCR** | **73-421** | **High** |

**Key**: DeepSeek-OCR achieves high quality with far fewer tokens than standard VLMs!

## Formula

**Actual tokens** = Visual grid + newlines + separators

```
tokens = (grid_width + 1) × grid_height + 1
```

**Examples**:
- 8×8: (8+1) × 8 + 1 = 73
- 16×16: (16+1) × 16 + 1 = 273
- 20×20: (20+1) × 20 + 1 = 421

## Newlines Matter!

**Paper reports**: 64, 100, 256, 400 (grid sizes)
**Implementation**: 73, 111, 273, 421 (includes newlines)

**Why newlines?**
- Preserve 2D structure
- Help LLM understand layout
- Separate rows in token sequence

## Tuning for Your Use Case

**Speed critical** → Tiny (73) or Small (111)
**Quality critical** → Large (421) or Gundam
**General purpose** → Base (273)

**See Also**:
- [optical-compression.md](optical-compression.md) - How compression works
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - All modes
