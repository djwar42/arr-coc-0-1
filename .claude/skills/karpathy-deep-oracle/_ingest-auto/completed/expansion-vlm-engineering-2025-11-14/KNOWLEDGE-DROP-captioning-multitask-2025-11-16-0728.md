# KNOWLEDGE DROP: Captioning & Multi-Task Evaluation

**Date**: 2025-11-16 07:28
**Source**: PART 14 execution
**File Created**: `vlm-engineering/13-captioning-multitask-evaluation.md`

## Summary

Created comprehensive guide to VLM captioning and multi-task evaluation metrics (700+ lines). Covers all major caption metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE), MS-COCO evaluation protocol, visual grounding, image-text retrieval, and multi-task benchmarks (VLUE, MMBench, SEED-Bench).

## Key Content

### Section 1: Image Captioning Metrics (200 lines)
- **BLEU**: N-gram overlap, brevity penalty, geometric mean
- **METEOR**: Synonyms, stemming, recall-weighted F-score
- **ROUGE-L**: Longest common subsequence matching
- **CIDEr**: TF-IDF weighted consensus scoring (0.92 human correlation)
- **SPICE**: Semantic scene graph matching (tuple-based F1)
- Metric comparison table with strengths/weaknesses

### Section 2: MS-COCO Protocol (150 lines)
- 330k images, 5 reference captions per image
- Official evaluation server workflow
- Multi-reference evaluation logic (max vs average vs union)
- Corpus-level vs image-level metrics
- Common pitfalls (length bias, copying, hallucination)

### Section 3: Other Caption Benchmarks (100 lines)
- Flickr30K: 31k images, people activities
- NoCaps: Novel object generalization testing
- Conceptual Captions (CC3M, CC12M): Pre-training datasets

### Section 4: Visual Grounding (100 lines)
- RefCOCO family: IoU-based accuracy @ 0.5 threshold
- Visual Genome: Dense captioning
- GROOViST: Grounding in visual storytelling

### Section 5: Image-Text Retrieval (100 lines)
- Recall@K (K=1,5,10) for bidirectional retrieval
- Mean Reciprocal Rank (MRR)
- NDCG for multiple correct matches

### Section 6: Multi-Task Benchmarks (150 lines)
- **VLUE**: 8 tasks, unified VL understanding
- **MMBench**: 2,974 questions, 20 ability dimensions, circular evaluation
- **SEED-Bench**: 24k questions, 27 dimensions (spatial/temporal/reasoning)
- **LLaVA-Bench**: 90 pairs, GPT-4 as judge, instruction-following

### Section 7: ARR-COC-0-1 Evaluation (100 lines)
- Relevance-aware captioning: Query-conditioned generation
- Token budget efficiency: CIDEr per 100 tokens
- VQA + Captioning joint evaluation
- Grounding with opponent processing

### Section 8: Best Practices (100 lines)
- Reporting standards (dataset version, all 5 metrics, inference settings)
- Essential ablations (vision encoder, token budget, fusion method)
- Error analysis categories (hallucination, missing objects, wrong attributes)
- Human evaluation protocol (1-5 rating, inter-annotator agreement)

## Citations

**Primary Papers**:
- MS COCO Captions (Chen et al., 2015)
- BLEU (Papineni et al., 2002)
- METEOR (Banerjee & Lavie, 2005)
- ROUGE (Lin, 2004)
- CIDEr (Vedantam et al., 2015)
- SPICE (Anderson et al., 2016)

**Benchmarks**:
- VLUE (Zhou et al., 2022)
- MMBench (OpenCompass, 2024)
- SEED-Bench (Li et al., 2024)
- RefCOCO (Yu et al., 2016)
- GROOViST (Surikuchi et al., 2023)

**Web Resources**:
- LearnOpenCV VLM Evaluation Guide (accessed 2025-11-16)
- MS COCO Evaluation API (GitHub)
- Image Captioning Evaluation Survey (Li et al., 2024)

## Integration Notes

- Complements PART 13 (VQA evaluation metrics)
- Provides ARR-COC-0-1 caption evaluation framework
- Multi-task benchmarks enable holistic VLM assessment
- All metrics include equations and Python implementation references

## File Stats

- **Lines**: 716
- **Sections**: 8
- **Citations**: 15+ papers
- **Code Examples**: Evaluation protocols, metric formulas, ablation studies
