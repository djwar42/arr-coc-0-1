# Stage 3: Gundam-Master Fine-tuning

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 452-550

## Objective

Fine-tune for ultra-high-resolution documents (Gundam mode).

## Focus

**Gundam-Master mode**: 1024×1024 tiles (256 tokens each)

Compared to base Gundam: 640×640 tiles (100 tokens each)

**Higher quality** for extreme detail preservation

## Data

Subset of high-resolution documents:
- Academic papers with formulas
- Technical diagrams
- Multi-column layouts
- Fine-print documents

**Size**: ~10M samples

## Training

- **Duration**: 3 days on 160 A100 GPUs
- **Learning rate**: 1e-5 (very low, fine-tuning)
- **Focus**: Base model already good, just adapting

## Result

Gundam-Master mode for maximum quality ultra-high-res processing!

**See Also**:
- [../architecture/resolution-modes.md](../architecture/resolution-modes.md) - Gundam mode details
