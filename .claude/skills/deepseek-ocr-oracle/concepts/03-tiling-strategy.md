# Tiling Strategy (Gundam Mode)

## Purpose

Handle **ultra-high-resolution** images that exceed standard mode capabilities.

## How It Works

```
Original Image (3000×2000)
    ↓
Divide into tiles
    ├─ Tile 1 (640×640) → 111 tokens
    ├─ Tile 2 (640×640) → 111 tokens
    ├─ Tile 3 (640×640) → 111 tokens
    ├─ ... (6 tiles total)
    └─ Global view (1024×1024) → 273 tokens
    ↓
Total: 6×111 + 273 = 939 tokens
```

## Dynamic Tiling

**Not uniform grid!** - Content-aware aspect ratio matching

**Algorithm** (in `process/image_process.py:45-120`):
1. Try different tile configurations (2×2, 2×3, 3×2, etc.)
2. Calculate aspect ratio deviation for each
3. Choose best match to original aspect ratio
4. Minimize padding/distortion

## Two Variants

### Gundam (Base + Small tiles)
- Global view: 1024×1024 (273 tokens)
- Local tiles: 640×640 (111 tokens each)
- **Use case**: Standard high-res documents

### Gundam-Master (Base + Base tiles)
- Global view: 1280×1280 (421 tokens)
- Local tiles: 1024×1024 (273 tokens each)
- **Use case**: Maximum quality, extreme detail

## When to Use

**✓ Use Gundam**:
- Multi-column layouts
- Scanned newspapers
- Technical drawings with fine details
- Documents > 2000px on any side

**✗ Don't use Gundam**:
- Standard documents (Base sufficient)
- Speed priority (Gundam is slower)
- Limited GPU memory

## Performance

**Speed**: 2-3× slower than Base (more tiles to process)
**Memory**: 2-3× more than Base
**Quality**: Best for ultra-high-res

## Code Example

```python
# Enable Gundam mode
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR.",
    images=["high_res_doc.jpg"],
    base_size=1024,      # Global view
    image_size=640,      # Tile size
    crop_mode=True       # Enable tiling!
)
```

## Token Budget Calculation

**Formula**: `(q*w + 1) × (q*h) + global_tokens`

Where:
- q = tiles per dimension (e.g., 10 for 640×640)
- w, h = number of tiles horizontally/vertically
- global_tokens = base mode tokens (273)

**Example** (2×3 tiling):
- Local: (10*2 + 1) × (10*3) = 21 × 30 = 630
- Global: 273
- **Total**: 903 tokens

**See Also**:
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - All modes
- [../code-reference/image-processing.md](../code-reference/image-processing.md) - Tiling implementation
