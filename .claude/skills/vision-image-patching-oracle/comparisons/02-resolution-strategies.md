# Resolution Strategy Comparison

**How different models handle variable resolutions**

## Strategy Categories

### 1. Fixed Resolution (ViT, Early LLaVA)

**Approach**: Resize all images to fixed size (e.g., 336×336)

**Pros**: Simple, predictable token count
**Cons**: Shape distortion, information loss

### 2. Variable Slicing (LLaVA-UHD)

**Approach**: Divide into adaptive slices based on aspect ratio

**Pros**: No distortion, native aspect ratio preserved
**Cons**: Variable token count, complex batching

### 3. Native Processing (Ovis)

**Approach**: Process full image without slicing, dynamic token allocation

**Pros**: No slice boundaries, global context
**Cons**: Limited to moderate resolutions

### 4. Specialized (DeepSeek-OCR)

**Approach**: Domain-specific compression for documents

**Pros**: Extreme efficiency (16×)
**Cons**: Optimized for specific content type

## Resolution Scaling

| Strategy | Max Resolution | Scaling Method | Token Growth |
|----------|----------------|----------------|--------------|
| Fixed | 336×336 | Resize | O(1) - constant |
| Variable Slicing | 2048×2048+ | Adaptive slices | O(n) - linear with compression |
| Native | 672×672 | Full image + VET | O(n) - content-dependent |
| Specialized | 336×336 | Optical compression | O(1) - 36 tokens always |

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)
- [07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)

## Related Documents

- [00-approaches-compared.md](00-approaches-compared.md)
- [01-token-budgets.md](01-token-budgets.md)
- [../architecture/03-native-resolution.md](../architecture/03-native-resolution.md)
