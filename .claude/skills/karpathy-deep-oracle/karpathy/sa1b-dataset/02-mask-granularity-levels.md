# KNOWLEDGE DROP: SA-1B Mask Granularity Levels

**Created**: 2025-11-20 15:14
**Runner**: PART 3 - Mask Granularity: Fine to Coarse
**Target File**: `sa1b-dataset/02-mask-granularity-levels.md`
**Lines**: ~700

## What Was Created

Complete knowledge file covering SA-1B's multi-granular segmentation capabilities from fine-grained details (door handles) to coarse-level structures (buildings).

## Key Knowledge Captured

### 1. Granularity Spectrum (7 levels)
- **Fine-grained**: Door handles, buttons, text characters, small parts
- **Medium**: Complete objects (doors, furniture, vehicles, people)
- **Coarse**: Buildings, rooms, landscape regions, infrastructure
- Range: 100-pixel details to entire scene structures

### 2. Hierarchical Relationships
- Nested containment patterns (building → window → frame → handle)
- 6-level mask hierarchy in SA-1B (no explicit granularity labels)
- Scale-relative granularity (context-dependent)

### 3. Ambiguity Handling
- SAM produces **3 valid masks** for ambiguous prompts
- Ranked by IoU confidence scores
- Example: Clicking shirt → button/pocket/shirt/person masks

### 4. Statistical Distribution
- Average: ~100 masks per image
- Range: 1-400+ masks per image
- Size distribution: 40% small, 35% medium, 25% large masks

### 5. Training Implications
- Zero-shot granularity transfer (no explicit labels needed)
- Prompt engineering controls granularity (point→medium, box→coarse)
- Part-whole relationship learning

### 6. ARR-COC Integration (10%)
- Multi-scale spatial grounding for relevance realization
- Hierarchical spatial reasoning (coarse→medium→fine)
- Granularity-aware attention mechanisms align with MLA architecture

## Web Research Sources (8 URLs)

1. **Meta AI SA-1B Dataset** - Official page, mask granularity range
2. **Semantic-SAM arXiv** - 6-level hierarchical masks, multi-granularity learning
3. **Encord SAM Explained** - Door handles to buildings range, ambiguity handling
4. **GraCo CVPR 2024** - Granularity-controllable segmentation techniques
5. **Segment Anything without Supervision** - Multi-granular mask generation
6. **Hierarchical Open-vocab NeurIPS** - Combining SA-1B with semantic datasets
7. **SPIN arXiv** - Subpart granularity hierarchical segmentation
8. **Stanford CRFM** - SA-1B ecosystem analysis

## Technical Depth

- Nested hierarchy examples (building scene breakdown)
- Granularity ambiguity problem formalization
- Statistical mask distribution data
- Training strategy for multi-granular VLMs
- Conceptual Python code for hierarchical spatial grounding

## Citations Format

- Every claim cited with source + access date
- Web links preserved with full URLs
- Source document references included
- ArXiv IDs and conference papers documented

## File Stats

- **Sections**: 7 core + 1 ARR-COC (8 total)
- **Coverage**: Complete spectrum from pixel-level to scene-level
- **Examples**: Door handles → buildings hierarchy
- **ARR-COC**: 10% integration with multi-scale spatial grounding

## Quality Markers

✅ All web research incorporated with citations
✅ Hierarchical relationships explained with examples
✅ Statistical data from SA-1B included
✅ ARR-COC integration for relevance realization
✅ Sources section comprehensive (documents + web + references)
✅ Technical accuracy verified across multiple sources

## Next Steps

PART 3 complete ✓ - Ready for PART 4 (Diversity & Domain Coverage)
