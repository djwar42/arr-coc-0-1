# Performance Metrics & Benchmarks

## Benchmark Results

### Fox Benchmark (200 pages)

| Metric | Standard VLM | DeepSeek-OCR | Improvement |
|--------|-------------|--------------|-------------|
| Vision tokens/page | 7000+ | 273 (Base) | 96% reduction |
| Total tokens | ~1.4M | 54.6K | 96% reduction |
| Quality (precision) | High | High | Comparable |
| Speed | Slow | Fast | 10-20× faster |

### OmniDocBench

**State-of-the-art** on document understanding tasks:
- Document OCR
- Chart parsing
- Formula recognition
- Layout analysis

**DeepSeek-OCR**: Best performance with minimal tokens!

## Compression Ratios

| Text Length | Mode | Tokens | Compression | Precision |
|-------------|------|--------|-------------|-----------|
| 600-700 | Tiny | 73 | 8.8× | 96.5% |
| 600-700 | Small | 111 | 5.8× | 98.5% |
| 700-800 | Tiny | 73 | 10.3× | 93.8% |
| 700-800 | Small | 111 | 6.5× | 97.3% |
| 800-900 | Small | 111 | 7.4× | 96.8% |
| 1200-1300 | Small | 111 | 11.0× | 87.1% |

**Key**: Higher compression → lower precision, but still usable!

## Production Throughput

**Single A100 GPU**:
- Base mode: 20k+ pages/day
- Small mode: 40k+ pages/day
- Large mode: 10k+ pages/day

**With vLLM**: 10-20× faster than HuggingFace Transformers

## Quality by Document Type

| Document | Mode | Edit Distance |
|----------|------|---------------|
| Slides | Tiny | 0.116 |
| Books | Small | 0.085 |
| Reports | Small | 0.079 |
| Textbooks | Base | 0.100 |
| Magazines | Large | 0.060 |
| Academic papers | Large | 0.155 |

**Lower is better** - Edit distance measures OCR errors

## Inference Speed (A100)

| Mode | Tokens | Time (ms) | Pages/sec |
|------|--------|-----------|-----------|
| Tiny | 73 | 25 | 40 |
| Small | 111 | 35 | 28 |
| Base | 273 | 50 | 20 |
| Large | 421 | 80 | 12 |

**Without generation time** - add ~10ms/token for output

## Memory Usage

| Mode | VRAM (GB) | Batch Size |
|------|-----------|------------|
| Tiny | 8 | 4 |
| Small | 10 | 4 |
| Base | 12 | 2 |
| Large | 16 | 1 |

## Cost Analysis

**Training cost**: ~$260k (17 days, 160 A100s)

**Inference cost** (single A100):
- Base mode: 20k pages/day
- At $2/hour: $48/day
- **Cost per page**: $0.0024

**Extremely cost-effective** for production!

## Comparison Summary

**vs Standard VLMs**: 96% fewer tokens, comparable quality
**vs Traditional OCR**: End-to-end, no error propagation
**Production**: 20k+ pages/day on single GPU

**See Also**:
- [vs-standard-vlms.md](vs-standard-vlms.md) - Architectural comparison
- DeepSeek-OCR paper - Full benchmark results
- `RESEARCH/DeepSeekOCR/ARCHITECTURE.md` - Performance analysis
