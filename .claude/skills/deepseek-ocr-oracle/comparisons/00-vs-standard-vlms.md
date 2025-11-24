# DeepSeek-OCR vs Standard VLMs

## Key Differences

| Aspect | Standard VLMs | DeepSeek-OCR |
|--------|--------------|--------------|
| **Vision tokens** | 7000+ | 73-421 |
| **Architecture** | Parallel encoders | Serial SAM→CLIP |
| **Compression** | Minimal | 16× spatial |
| **Resolution** | Fixed | Variable (5 modes) |
| **Purpose** | General vision | Text-focused OCR |

## Token Efficiency

**Standard VLM** (e.g., LLaVA, Qwen-VL):
- High-res grid: 7000+ vision tokens
- Fixed resolution
- General-purpose features

**DeepSeek-OCR**:
- Optical compression: 73-421 tokens
- Variable resolution
- Text-optimized features

**Result**: 10-20× fewer tokens with comparable quality!

## Architecture Comparison

### Standard VLM
```
Image → CLIP → Projector → LLM
```
- Single vision encoder
- High-res grid → many tokens
- General features

### DeepSeek-OCR
```
Image → SAM → Compress → CLIP → Projector → LLM
```
- Serial dual encoders
- Compression between stages
- Text-specialized features

## Performance Comparison

**Fox Benchmark** (200 pages):
- Standard VLM: ~1.4M vision tokens
- DeepSeek-OCR: ~54.6K vision tokens
- **Savings**: 96%!

**Quality**: Comparable or better despite fewer tokens

## When to Use Each

**Standard VLM**:
- General vision understanding
- Object detection
- Scene description
- Visual reasoning

**DeepSeek-OCR**:
- Document OCR
- Text extraction
- Chart parsing
- Formula recognition

**See Also**:
- [vs-arr-coc-vis.md](vs-arr-coc-vis.md) - Compression vs relevance
- [performance-metrics.md](performance-metrics.md) - Benchmarks
