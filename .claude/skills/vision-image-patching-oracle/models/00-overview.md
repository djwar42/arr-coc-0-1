# Model Patching Approaches Overview

**Comparison of image patching strategies across major vision-language models**

## Summary Table

| Model | Patch Size | Resolution | Slicing | Compression | Token Count |
|-------|------------|------------|---------|-------------|-------------|
| ViT | 16×16 | 224×224 | No | No | 196 |
| CLIP-ViT-L/14 | 14×14 | 336×336 | No | No | 576 |
| LLaVA-1.5 | 14×14 | 336×336 | No | No | 576 |
| LLaVA-UHD | 14×14 | 672×672 | Variable slices | 9× | 384 (6 slices × 64) |
| GPT-4V | Unknown | Variable | Fixed 512×512 | Yes (unknown ratio) | Unknown |
| Ovis | Native | Any | No (native) | No | Variable |
| DeepSeek-OCR | 14×14 | 336×336 | No | 16× | 36 |

## Approach Categories

### Fixed Patching (ViT, CLIP)
- Standard 14×14 or 16×16 patches
- Single resolution (224×224 or 336×336)
- No compression
- 196-576 tokens

### Native Resolution (LLaVA-UHD, Ovis)
- Variable-sized slicing
- Preserves aspect ratio
- Compression modules
- Adaptive token count

### Specialized Compression (DeepSeek-OCR)
- Domain-specific (documents)
- Extreme compression (16×)
- Optical compression architecture
- 36 tokens for 336×336

## Primary Sources

All source-documents in `../source-documents/`

## Related Documents

- [01-vit.md](01-vit.md)
- [02-llava-uhd.md](02-llava-uhd.md)
- [04-ovis.md](04-ovis.md)
- [05-deepseek-ocr.md](05-deepseek-ocr.md)
- [../comparisons/00-approaches-compared.md](../comparisons/00-approaches-compared.md)
