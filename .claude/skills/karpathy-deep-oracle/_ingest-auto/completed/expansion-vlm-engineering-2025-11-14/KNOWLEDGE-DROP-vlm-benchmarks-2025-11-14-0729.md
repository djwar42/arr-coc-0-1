# KNOWLEDGE DROP: VLM Benchmarks & Leaderboards

**Timestamp**: 2025-11-14 07:29
**Part**: 16 of 20
**Batch**: 4 (Evaluation & Benchmarking)
**File Created**: `vlm-engineering/15-vlm-benchmarks-leaderboards.md`
**Lines**: ~700
**Status**: ✓ Complete

## What Was Created

Comprehensive guide to VLM evaluation landscape covering:

1. **VQA Benchmarks** (VQAv2, GQA, OKVQA, TextVQA, VizWiz)
2. **Captioning Benchmarks** (COCO, Flickr30K, NoCaps)
3. **Multi-Task Benchmarks** (MMMU, MMBench, MME, SEED-Bench)
4. **Reasoning Benchmarks** (CLEVR, NLVR2, Winoground)
5. **Leaderboards** (Vision Arena, Open VLM Leaderboard)
6. **Benchmark Saturation Issues** (when benchmarks become too easy)
7. **Custom Benchmark Creation** (ARR-COC-0-1 evaluation protocol)
8. **Best Practices** (multi-benchmark evaluation, error analysis, efficiency metrics)

## Key Insights for ARR-COC-0-1

### 1. Benchmark Saturation Reality
- VQAv2 approaching 80% (human agreement ~83%)
- MMMU still challenging: GPT-4V only 56% overall
- Need to focus on HARD categories where models struggle (42% on MMMU hard)

### 2. Efficiency Metrics Matter
- Accuracy alone insufficient - must report tokens/FLOPs
- Pareto frontier analysis: accuracy vs computational cost
- ARR-COC-0-1's relevance allocation needs efficiency baselines

### 3. Multi-Benchmark Coverage Essential
- Single benchmark = dataset-specific performance
- Recommended suite: VQAv2, GQA, TextVQA, COCO, MMMU
- Zero-shot transfer tests generalization (VQAv2→GQA, VQAv2→VizWiz)

### 4. Leaderboard Insights
**Vision Arena** (human preference):
- Gemini 2.5 Pro leads (1249 Elo)
- Best open-source: Qwen3-VL-235B (1204 Elo, rank 12)
- 40-50 Elo gap between proprietary and open models

**Open VLM Leaderboard** (standardized benchmarks):
- Aggregates MMMU, MMBench, TextVQA, VQAv2, etc.
- Enables reproducible comparison for open models

### 5. Error Analysis Framework
From MMMU GPT-4V analysis:
- 40% perception errors (missed visual details)
- 30% knowledge errors (incorrect domain knowledge)
- 20% reasoning errors (logical mistakes)
- 10% other (ambiguity, annotation issues)

**For ARR-COC-0-1**: Track whether failures stem from relevance allocation (perception) vs downstream reasoning.

### 6. Image Type Heterogeneity
MMMU's 30 image types reveal specialization challenges:
- Diagrams (3,184 questions)
- Chemical structures (573) - models near random guessing
- Sheet music (335) - extremely challenging
- Medical images (272) - domain-specific

**ARR-COC-0-1 should adapt relevance strategies by image type** (diagrams need compress, photos need particularize).

### 7. ARR-COC-0-1 Benchmark Strategy

**Primary Evaluation**:
1. VQAv2 (general) - Target: ≥75%
2. GQA (compositional) - Target: ≥65%
3. TextVQA (OCR/high-res) - Target: ≥70%
4. COCO Captions - Target: ≥130 CIDEr

**Efficiency Evaluation**:
1. Token budget ablations (64, 144, 256, 400)
2. Pareto frontier vs fixed-budget baselines
3. Relevance IoU with human annotations (target ≥0.7)
4. Adaptive LOD correlation (target ≥0.8)

**Ablation Studies**:
- Remove opponent processing → measure compress/particularize impact
- Fixed LOD → measure adaptation value
- Remove relevance scoring → measure allocation quality
- Vary K (patches) → characterize K vs accuracy tradeoff

## Citations Included

**Primary Sources**:
- MMMU Benchmark (https://mmmu-benchmark.github.io/) - 11.5K college-level questions
- Vision Arena (https://lmarena.ai/leaderboard/vision) - 551K votes, 81 models
- Open VLM Leaderboard (HuggingFace) - Standardized benchmark aggregation

**Research**:
- Benchmark saturation study (gradientscience.org)
- VLM evaluation survey (arXiv:2501.02189v3)
- Community discussions on benchmark gaming (Reddit LocalLLaMA)

## Integration Points

**Connects to**:
- `12-vqa-evaluation-metrics.md` - VQA-specific evaluation details
- `13-captioning-multitask-evaluation.md` - Captioning metrics
- `14-ablation-studies-analysis.md` - Ablation methodology
- `16-vlm-inference-optimization.md` - Efficiency metrics alignment

**Influences**:
- Benchmark selection for ARR-COC-0-1 evaluation
- Efficiency metric definitions (accuracy per token)
- Error analysis categorization
- Custom benchmark design for relevance allocation quality

## Next Steps

**For Oracle Integration**:
1. Update INDEX.md with new file
2. Cross-reference with evaluation files (12, 13, 14)
3. Ensure ARR-COC-0-1 sections consistently reference this benchmark guide

**For ARR-COC-0-1 Development**:
1. Implement custom benchmark (relevance allocation IoU)
2. Run multi-benchmark evaluation (VQAv2, GQA, TextVQA, COCO)
3. Generate Pareto frontier plots (accuracy vs tokens)
4. Conduct error analysis (100+ failures categorized)

## Quality Checklist

- [✓] Comprehensive coverage of VLM benchmarks (VQA, captioning, multi-task, reasoning)
- [✓] Leaderboard analysis with current rankings (Nov 2025)
- [✓] Benchmark saturation discussion with research citations
- [✓] ARR-COC-0-1 custom benchmark protocol detailed
- [✓] Best practices for VLM evaluation
- [✓] Efficiency metrics emphasized (tokens, FLOPs, latency)
- [✓] Error analysis framework from MMMU study
- [✓] Web research citations with access dates
- [✓] Integration with ARR-COC-0-1 throughout (Section 8 in each benchmark)

**Total**: 720 lines of comprehensive VLM benchmarking knowledge for ARR-COC-0-1 evaluation strategy.
