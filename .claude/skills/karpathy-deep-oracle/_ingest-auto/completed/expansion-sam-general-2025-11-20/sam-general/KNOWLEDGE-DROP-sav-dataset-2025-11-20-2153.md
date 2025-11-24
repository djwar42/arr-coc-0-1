# Knowledge Drop: SA-V Video Dataset

**Date:** 2025-11-20 21:53
**Runner:** expansion-sam-general
**File Created:** `17-sav-video-dataset.md`

## Summary

Created comprehensive documentation for SA-V (Segment Anything Video) dataset, the largest video segmentation dataset ever created by Meta AI for training SAM 2.

## Key Information Captured

### Dataset Scale
- **51,000+ videos** from 47 countries
- **643,000 masklets** (spatio-temporal masks)
- **35.5 million individual frame masks**
- 50x larger than previous video segmentation datasets

### Annotation Breakdown
- 191K manual annotations (human-annotated)
- 452K automatic annotations (model-generated, human-verified)
- Average ~12.6 masklets per video

### Data Engine Process
Three-phase annotation pipeline:
1. SAM-assisted annotation (6x faster than manual)
2. Semi-automatic with human verification
3. Fully automatic with quality control

### Benchmark Tasks
- Video Object Segmentation (VOS)
- Interactive Video Segmentation
- Semi-supervised VOS
- SAM 2 achieves 82.5 J&F on DAVIS 2017

## Sources Used

**Primary:**
- SAM 2 Paper (arXiv:2408.00714)
- Meta AI Dataset Page
- Meta AI Blog Post (July 29, 2024)

**Documentation:**
- Ultralytics SAM 2 Documentation
- Encord SAM 2 Analysis
- Kili Technology Deep Dive

## ARR-COC Integration Notes

- Data engine methodology applicable to multi-modal training
- Video-language understanding opportunities
- Temporal reasoning for foundation models
- ~10% of document dedicated to integration patterns

## Line Count

Total lines: ~700 (as specified)

## Sections Completed

1. SA-V Dataset Overview (~120 lines)
2. Video Collection (~100 lines)
3. Annotation Process (~120 lines)
4. Statistics (~100 lines)
5. Benchmark Tasks (~80 lines)
6. Download/Usage (~80 lines)
7. ARR-COC Integration (~70 lines)

## Quality Notes

- All major statistics cited with sources
- Code examples for data loading
- Comparison tables with prior datasets
- Complete citation information included
