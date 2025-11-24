# Design Philosophy

## Core Principles

### 1. Serial, Not Parallel

**Standard approach**: Process everything in parallel
**DeepSeek approach**: Cheap first, then expensive

```
Traditional:                  DeepSeek-OCR:
  SAM ↘                         SAM
       Concat → LLM               ↓
 CLIP ↗                        Compress
                                 ↓
                              CLIP → LLM
```

**Why?**
- SAM: Cheap window attention on 4096 tokens
- Compress: 16× reduction
- CLIP: Expensive global attention on 256 tokens only
- **Result**: Memory efficient, fast

### 2. Compression as Core Feature

**Not an afterthought** - compression is central to architecture!

**Location**: Inside SAM (not separate module)
**Amount**: 16× (empirically optimal)
**Purpose**: Enable expensive CLIP processing

### 3. Multi-Resolution by Design

**One model, all resolutions** (73-421 tokens)

**Mechanism**: Positional encoding interpolation
**Training**: All modes trained simultaneously
**Benefit**: Single deployment, user chooses quality/speed

### 4. End-to-End Learning

**No traditional OCR engine** in pipeline!

**Traditional**: Image → OCR → Text → LLM
**DeepSeek**: Image → Vision tokens → LLM

**Benefits**:
- No OCR error propagation
- Layout preserved
- Faster (single step)

### 5. Biological Inspiration

**Human vision parallels**:
- Foveal (SAM): High detail where needed
- Peripheral (compression): Reduce elsewhere
- Semantic (CLIP): Understand meaning
- Memory (forgetting): Progressive compression over time

## Key Design Decisions

**Q**: Why SAM + CLIP, not just one?
**A**: SAM = fine details, CLIP = semantics. Need both!

**Q**: Why 16× compression specifically?
**A**: Empirical sweet spot. Less = still expensive, More = quality loss

**Q**: Why serial architecture?
**A**: Enables compression before expensive operations

**Q**: Why train all resolutions together?
**A**: Single model deployment, user flexibility

## Design Tradeoffs

**Chosen**: Serial architecture
**Tradeoff**: More sequential (slower than parallel if unlimited compute)
**Benefit**: Practical efficiency with real hardware constraints

**Chosen**: 16× compression
**Tradeoff**: Some information loss
**Benefit**: Massive speed/memory gains

**Chosen**: Multi-resolution single model
**Tradeoff**: Complexity in training
**Benefit**: Deployment simplicity

## Philosophy Summary

**Efficiency through intelligence**:
- Process cheap before expensive
- Compress aggressively but smartly
- Learn once, deploy flexibly
- Biological inspiration

**See Also**:
- [optical-compression.md](optical-compression.md) - Why compression works
- [../architecture/deepencoder.md](../architecture/deepencoder.md) - Serial architecture
