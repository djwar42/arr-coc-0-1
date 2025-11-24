# KNOWLEDGE DROP: SAM Model Checkpoints Comparison

**Date**: 2025-11-20 21:49
**PART**: 12
**Status**: SUCCESS

## Created File

**File**: `sam-general/11-model-checkpoints-comparison.md`
**Lines**: 645

## Content Summary

Comprehensive comparison of SAM's three checkpoint variants:

### Section 1: Checkpoint Overview (~100 lines)
- Three checkpoint variants: ViT-H, ViT-L, ViT-B
- Parameter counts: 636M, 308M, 91M
- Download URLs and sizes
- Architectural constants

### Section 2: ViT-H (Huge) (~120 lines)
- Technical specifications (32 layers, 1280 dim)
- ~110ms inference speed
- Best quality (0.88 IoU)
- Production and research use cases

### Section 3: ViT-L (Large) (~100 lines)
- Technical specifications (24 layers, 1024 dim)
- ~80ms inference speed
- Balanced quality (0.86 IoU)
- Best value for most applications

### Section 4: ViT-B (Base) (~100 lines)
- Technical specifications (12 layers, 768 dim)
- ~50ms inference speed
- Good quality (0.83 IoU)
- Edge deployment and fine-tuning

### Section 5: Performance Benchmarks (~100 lines)
- Comprehensive model comparison tables
- SAM vs YOLO comparison
- Zero-shot transfer performance
- GPU memory usage and throughput

### Section 6: Selection Guidelines (~80 lines)
- Decision matrix by requirement
- Application-based recommendations
- Hardware-based selection
- Cost-benefit analysis

### Section 7: ARR-COC Integration (~70 lines)
- Checkpoint selection for ARR-COC workflows
- Auto-annotation pipeline code
- Performance considerations
- Recommended configurations

## Sources Cited

**Source Documents:**
- SAM_STUDY_GENERAL.md (lines 140-151, 562-598, 1033-1096, 1302-1358)

**Web Research:**
- Segment Anything paper (arXiv:2304.02643)
- Roboflow SAM Tutorial
- Ultralytics SAM Documentation
- Medical Imaging SAM Study (ScienceDirect)
- IEEE Performance Analysis

**GitHub:**
- facebookresearch/segment-anything

## Key Technical Details Captured

1. **Parameter Scaling**:
   - ViT-H: 636M (2.4 GB)
   - ViT-L: 308M (1.2 GB)
   - ViT-B: 91M (375 MB)

2. **Performance Trade-offs**:
   - ViT-H: Best quality, slowest
   - ViT-L: Only 2% quality loss, 37% faster
   - ViT-B: 5.7% quality loss, 120% faster

3. **YOLO Comparison**:
   - YOLO11n-seg is 13.2x smaller than SAM-b
   - YOLO11n-seg is 864x faster on CPU

## Execution Notes

- File created successfully
- All sections completed as specified
- ARR-COC integration section included (~10% of content)
- All web sources properly cited with access dates
