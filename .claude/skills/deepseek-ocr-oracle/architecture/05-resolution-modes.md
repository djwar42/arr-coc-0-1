# Multi-Resolution Modes

## Overview

DeepSeek-OCR supports **variable token budgets** (73-421 tokens) through 5 resolution modes, all using the **same model weights**.

## Mode Table

| Mode | Resolution | Visual Grid | Actual Tokens* | Use Case |
|------|-----------|-------------|---------------|----------|
| **Tiny** | 512×512 | 8×8 | 73 | Simple slides, low-detail |
| **Small** | 640×640 | 10×10 | 111 | Books, reports, charts |
| **Base** | 1024×1024 | 16×16 | 273 | Standard documents (default) |
| **Large** | 1280×1280 | 20×20 | 421 | High-detail, dense text |
| **Gundam** | Dynamic | Variable | Variable | Ultra-high-res (tiled) |

*Actual tokens = visual_grid_tokens + newline_tokens + separators

## Token Calculation Formula

### Base Modes (Tiny/Small/Base/Large)

```python
tokens = (tokens_per_row + 1) × num_rows + 1

# Example: Base mode (16×16 grid)
tokens = (16 + 1) × 16 + 1 = 273
```

**Implementation**: `deepseek_ocr.py:61-106`

### Gundam Mode (Dynamic Tiling)

```python
# Local views (tiled)
local_tokens = (q*w + 1) × (q*h)

# Global view
global_tokens = base_mode_tokens

# Total
total = local_tokens + global_tokens

# Example: 2×3 tiling at 640×640
# q=10 (10×10 grid per 640×640 tile)
local = (10*2 + 1) × (10*3) = 21 × 30 = 630
global = 273 (1024×1024 base)
total = 903 tokens
```

## Mode Selection Guide

### By Document Type

| Document Type | Recommended Mode | Tokens | Rationale |
|--------------|------------------|--------|-----------|
| Presentation slides | Tiny | 73 | Large fonts, minimal text |
| Books/novels | Small | 111 | Standard text, simple layout |
| Reports | Small-Base | 111-273 | Mixed content |
| Academic papers | Base-Large | 273-421 | Dense text, formulas, diagrams |
| Scanned documents | Large | 421 | Preserve quality |
| Multi-column layouts | Gundam | Variable | Need high resolution |

### By Quality Requirements

**Speed priority** → Tiny/Small (8× faster than Large)
**Quality priority** → Large/Gundam
**Balanced** → Base (default, works well)

## Technical Details

### How Multi-Resolution Works

**Same model handles all resolutions!**

**Mechanism**: Positional encoding interpolation

```python
# Tiny (8×8)
pos_embed = interpolate(learned_pos_embed, size=(8, 8))

# Base (16×16)
pos_embed = interpolate(learned_pos_embed, size=(16, 16))

# Large (20×20)
pos_embed = interpolate(learned_pos_embed, size=(20, 20))
```

**Training**: All resolutions trained simultaneously with weighted sampling

### Gundam Mode (Dynamic)

**Purpose**: Ultra-high-resolution images

**Strategy**: Tile + global view

```
Original Image (3000×2000)
    ↓
Divide into tiles (e.g., 3×2 grid of 640×640)
    ↓
Process each tile (111 tokens × 6 = 666 tokens)
    ↓
Add global view (273 tokens)
    ↓
Total: 939 tokens
```

**When to use**:
- Scanned multi-page documents
- High-resolution charts
- Fine print preservation

## Performance Comparison

| Mode | Tokens | GPU Time (ms) | Memory (GB) | VRAM (GB) |
|------|--------|---------------|-------------|-----------|
| Tiny | 73 | 25 | 0.8 | 8 |
| Small | 111 | 35 | 1.0 | 10 |
| Base | 273 | 50 | 1.5 | 12 |
| Large | 421 | 80 | 2.0 | 16 |
| Gundam | ~900 | 150+ | 3.0+ | 20+ |

*Times on A100, batch size 1

## Code Examples

### Tiny Mode
```python
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"],
                    base_size=512, image_size=512, crop_mode=False)
```

### Base Mode (Default)
```python
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"],
                    base_size=1024, image_size=1024, crop_mode=False)
```

### Gundam Mode
```python
result = model.infer(tokenizer, prompt="<image>\nFree OCR.", images=["doc.jpg"],
                    base_size=1024, image_size=640, crop_mode=True)
```

## File References

- `deepseek_ocr.py:61-106` - Token calculation
- `process/image_process.py:45-120` - Dynamic tiling
- `config.py:17-30` - Mode definitions

**See Also**:
- [../usage/resolution-selection.md](../usage/resolution-selection.md) - How to choose
- [../concepts/token-budgets.md](../concepts/token-budgets.md) - Why these numbers?
