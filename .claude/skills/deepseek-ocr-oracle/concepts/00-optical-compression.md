# Optical Compression

## The Core Innovation

**Idea**: A single image can represent text using FAR fewer tokens than digital text.

**Example**:
- Text: `"Hello World"` = 2-3 tokens in most tokenizers
- Image of text: Can represent entire paragraphs in 73-273 tokens
- **Result**: 10-20× compression ratios achieved

## Why Optical Compression Works

### 1. 2D Spatial Encoding

Text is inherently 2D when written, but LLMs process it as 1D sequences.

**Digital text**:
```
"This is line 1\nThis is line 2\nThis is line 3" → 15+ tokens
```

**Optical encoding**:
```
Image of 3 lines → ~111 tokens (Small mode)
```

The image preserves:
- Spatial layout
- Font information
- Structure (tables, columns)
- Visual context (diagrams, charts)

### 2. Human-Designed Compression

Written text is already optimized by humans:
- Clear fonts for readability
- Structured layouts
- High information density
- Redundancy reduced

**Result**: Images of text are pre-compressed by design.

### 3. Vision Encoders are Compression Experts

DeepEncoder (SAM + CLIP):
- SAM: Fine-grained local features (16× compression)
- CLIP: Semantic understanding
- Together: Preserve meaning while drastically reducing tokens

## Compression Ratios

| Text Length | Vision Tokens | Compression | Precision |
|-------------|--------------|-------------|-----------|
| 600-700 | 73 (Tiny) | 8.8× | 96.5% |
| 600-700 | 111 (Small) | 5.8× | 98.5% |
| 700-800 | 73 (Tiny) | 10.3× | 93.8% |
| 700-800 | 111 (Small) | 6.5× | 97.3% |
| 800-900 | 73 (Tiny) | 11.5× | 83.8% |
| 800-900 | 111 (Small) | 7.4× | 96.8% |
| 1200-1300 | 73 (Tiny) | 17.1× | 59.1% |
| 1200-1300 | 111 (Small) | 11.0× | 87.1% |

**Key Insight**: Higher compression → lower precision, but still very usable!

## Comparison: Text vs Optical

### Scenario: 1000-token document

**Standard VLM approach**:
```
Image → 7000+ vision tokens (high-res grid)
Text → 1000 text tokens
Total: 8000+ tokens
```

**DeepSeek-OCR approach**:
```
Image → 273 vision tokens (optical compression)
Text → 1000 text tokens (generated)
Total: 1273 tokens during inference
```

**Savings**: ~6× reduction in vision tokens!

## The Compression Pipeline

```
Original Image (1024×1024, millions of pixels)
    ↓
Patch Embedding (4096 tokens, 16×16 patches)
    ↓
SAM Window Attention (cheap processing)
    ↓
16× Spatial Compression (4096 → 256 patches)
    ↓
CLIP Global Attention (semantic extraction)
    ↓
Feature Fusion (SAM + CLIP)
    ↓
MLP Projection (to language space)
    ↓
Final: 273 vision tokens (with newlines)
    ↓
Language Model (decodes back to text)
```

**Net result**: ~14,000× compression (millions of pixels → 273 tokens)!

## Why Not Just OCR + LLM?

**Traditional approach**:
1. OCR engine extracts text (pytesseract, EasyOCR)
2. Text → LLM
3. LLM processes

**Problems**:
- OCR errors propagate
- Layout information lost
- Tables/charts require separate handling
- Multi-step pipeline (slow)

**DeepSeek-OCR approach**:
1. Image → Vision tokens (compressed)
2. Vision tokens → LLM decoding
3. Done

**Advantages**:
- End-to-end learning (no OCR errors)
- Layout preserved in vision tokens
- Charts/tables handled natively
- Single-step pipeline (fast)

## Optical vs Traditional Compression

### Traditional Image Compression (JPEG, PNG)
- Pixel-level compression
- Focus: Preserve visual appearance
- Lossy: Some visual quality lost
- **Goal**: Smaller file size

### Optical Compression (DeepSeek-OCR)
- Semantic-level compression
- Focus: Preserve meaning and text content
- Lossy: Visual appearance lost, meaning preserved
- **Goal**: Fewer tokens for LLM processing

**Key difference**: We don't care about reconstructing the image, only the text!

## The Memory/Quality Tradeoff

DeepSeek-OCR offers 5 compression levels:

| Mode | Tokens | Quality | Speed | Use Case |
|------|--------|---------|-------|----------|
| Tiny | 73 | Acceptable | Fastest | Simple slides |
| Small | 111 | Good | Fast | Books, reports |
| Base | 273 | Very Good | Medium | Standard docs |
| Large | 421 | Excellent | Slow | Dense text |
| Gundam | Variable | Best | Slowest | Ultra-high-res |

**User choice**: Quality vs speed, just like JPEG quality settings!

## Biological Parallel: Human Vision

Humans don't remember every pixel:
- Foveal vision: High detail where you look
- Peripheral vision: Low detail everywhere else
- **Compression**: Brain only stores salient information

DeepSeek-OCR similarly:
- SAM: Captures local details
- 16× compression: Reduces to essentials
- CLIP: Extracts semantic meaning
- **Result**: Minimal tokens, maximal information

## Forgetting as Compression

Over time, compress old contexts more aggressively:

```
Recent:    High-res (421 tokens) → Clear memory
1 hour:    Mid-res (273 tokens)  → Fading
1 day:     Low-res (111 tokens)  → Distant
1 week:    Tiny (73 tokens)      → Barely there
```

**Biological memory parallel**: Recent events are clear, old ones fade.

**Implementation**: Progressive resolution reduction for older contexts.

## Why 73-421 Tokens?

**Lower bound (73)**:
- Minimum grid for text recognition
- 8×8 patches (64) + newlines/separators
- Below this: Text becomes unreadable

**Upper bound (421)**:
- Diminishing returns above this
- 20×20 patches (400) + newlines
- Above this: Not much quality gain

**Sweet spot (273)**:
- 16×16 patches (256) + newlines
- Balance of quality and efficiency
- Empirically validated on benchmarks

## Production Impact

**Fox Benchmark** (200 pages):
- Traditional VLM: 7000+ tokens/page → 1.4M total
- DeepSeek-OCR: 273 tokens/page → 54.6K total
- **Savings**: 96% reduction!

**OmniDocBench** (diverse documents):
- Traditional: Varies, often 5000-10000 tokens/page
- DeepSeek-OCR: 73-421 tokens/page (adaptive)
- **Result**: State-of-art with minimal tokens

## Key Takeaways

1. **Optical compression is real**: 10-20× ratios achieved
2. **Quality-efficiency tradeoff**: User controls via resolution modes
3. **End-to-end learning**: No OCR errors
4. **Layout preserved**: 2D structure maintained
5. **Production-proven**: 200k+ pages/day on single A100

## File References

**Compression implementation**:
- SAM compression: `deepencoder/sam_vary_sdpa.py:166-183` (neck + conv layers)
- Token calculation: `deepseek_ocr.py:61-106`
- Preprocessing: `process/image_process.py`

**Paper evidence**:
- Compression ratios: Table 2 in DeepSeek-OCR paper
- Quality analysis: Figure 8 (precision vs compression)
- Benchmarks: Fox, OmniDocBench results

**See Also**:
- [token-budgets.md](token-budgets.md) - Why 73-421 tokens?
- [design-philosophy.md](design-philosophy.md) - Serial architecture rationale
- [forgetting.md](forgetting.md) - Progressive compression
- [../architecture/compression.md](../architecture/compression.md) - Technical details
