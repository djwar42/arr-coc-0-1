# KNOWLEDGE DROP: SA-1B Dataset Statistics (2025-11-20 15:25)

## What Was Created

**File**: `sa1b-dataset/01-statistics-scale.md` (~750 lines)

**Topic**: SA-1B dataset statistics—11M images, 1.1B masks, scale comparisons

## Key Knowledge Acquired

### Core Statistics (Massive Scale)
- **11M images**: High-resolution (averaging 1500×2250 pixels)
- **1.1B masks**: Largest segmentation dataset ever (6× larger than OpenImages v5)
- **~100 masks per image**: Comprehensive class-agnostic coverage
- **10TB uncompressed**: Distributed across 1000 tar files

### Mask Distribution Insights
- Average: ~100 masks per image
- Range: 1 to 400+ masks (scene-dependent)
- Right-skewed distribution (most images 50-150 masks)
- Multi-granularity: tiny parts → whole objects → scenes

### Scale Comparisons
- **73× more masks** than OpenImages v5 (1.1B vs 15M)
- **440× more masks** than LVIS (1.1B vs 2.5M)
- **12.5× more masks per image** than OpenImages (100 vs 8)
- **100× more masks per image** than COCO (~100 vs ~1-2)

### Image Specifications
- Resolution: 1500×2250 pixels (average, high-res)
- Format: JPEG, RGB color space
- Privacy: Faces and license plates blurred
- Licensing: Professionally sourced from third-party photo company

### Data Collection Scale
- Stage 1 (Manual): ~120K images, ~4M masks
- Stage 2 (Semi-auto): ~180K images, ~10M masks
- Stage 3 (Fully auto): ~10.7M images, ~1.086B masks
- Total: 11M images collected over months

### Storage & Computation
- Compressed: ~1TB (tar.gz)
- Uncompressed: ~10TB total
- Training: Requires 32-256 GPUs, days to weeks
- Distribution: 1000 tar files (~11K images each, ~10GB compressed)

## ARR-COC Integration (10%)

**Why SA-1B Scale Matters for ARR-COC:**

1. **Zero-Shot Spatial Grounding**: 1.1B masks enable foundation-level spatial reasoning
2. **Class-Agnostic Flexibility**: No category labels → flexible grounding for referring expressions
3. **Pre-Training Strategy**: Use SA-1B for spatial encoder pre-training before relevance tasks
4. **Scale Requirements**: Foundation models need massive data (1.1B masks proves the point)

**Potential Training Strategies:**
- Option A: Pre-train on SA-1B subset (1M images, 100M masks)
- Option B: Use pre-trained SAM encoder as frozen backbone
- Option C: Fine-tune SAM decoder on relevance-grounded segmentation

The **1.1B mask scale** demonstrates that **exhaustive spatial coverage** enables **zero-shot generalization**—a key lesson for ARR-COC's multimodal training.

## Sources Summary

- **11 web sources** (Meta AI, Stanford CRFM, Ultralytics, TensorFlow, arXiv, Medium, etc.)
- **1 source document** (SAM_DATASET_SA1B.md, 1123 lines)
- All links preserved with access dates (2025-11-20)
- Comprehensive citations for every statistic

## File Statistics

- **Total lines**: ~750 lines
- **Sections**: 8 (7 main + ARR-COC at 10%)
- **Tables**: 1 (dataset comparison table)
- **Citations**: 11+ web sources + 1 source document
- **ARR-COC content**: ~75 lines (10% of total)

---

**Status**: ✅ PART 2 COMPLETE (2025-11-20 15:25)
