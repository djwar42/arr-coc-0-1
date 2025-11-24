# Token Budget Comparison

**Token counts across different models and resolutions**

## Standard Configurations

### Base Models (224×224)

| Model | Patch Size | Tokens | Compression | Final Tokens |
|-------|------------|--------|-------------|--------------|
| ViT-B/16 | 16×16 | 196 | None | 196 |
| ViT-L/14 | 14×14 | 256 | None | 256 |

### VLM Standard (336×336)

| Model | Patch Size | Tokens | Compression | Final Tokens |
|-------|------------|--------|-------------|--------------|
| CLIP-ViT-L/14 | 14×14 | 576 | None | 576 |
| LLaVA-1.5 | 14×14 | 576 | None | 576 |
| DeepSeek-OCR | 14×14 | 576 | 16× | 36 |

### High-Resolution (672×672+)

| Model | Resolution | Slices | Tokens/Slice | Compression | Final Tokens |
|-------|------------|--------|--------------|-------------|--------------|
| LLaVA-UHD | 672×672 | 4 | 576 | 9× | 256 (4×64) |
| LLaVA-UHD | 672×1008 | 6 | 576 | 9× | 384 (6×64) |
| GPT-4V | Variable | N×M | 512² | Yes | Unknown |

## Token Budget Analysis

**Key insight**: Compression enables high-resolution processing within reasonable token budgets.

**Without compression**: 672×672 at 14×14 patches = 2,304 tokens
**With 9× compression**: 2,304 → 256 tokens (feasible for LLM)

## Primary Sources

- Model papers in `../source-documents/`

## Related Documents

- [00-approaches-compared.md](00-approaches-compared.md)
- [02-resolution-strategies.md](02-resolution-strategies.md)
- [../concepts/02-token-efficiency.md](../concepts/02-token-efficiency.md)
