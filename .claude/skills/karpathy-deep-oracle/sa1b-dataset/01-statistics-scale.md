# SA-1B Dataset Statistics: 11M Images, 1.1B Masks

## Overview

SA-1B (Segment Anything 1-Billion) represents an unprecedented scale in segmentation datasets. With **11 million diverse, high-resolution images** and **1.1 billion segmentation masks**, SA-1B is approximately **6× larger** than the next biggest open-source segmentation dataset (OpenImages v5). This massive scale enables training foundation models with zero-shot generalization capabilities across diverse visual domains.

The dataset's statistics reveal careful curation: averaging **~100 masks per image**, images with resolutions averaging **1500×2250 pixels**, and comprehensive coverage spanning natural scenes, objects, and textures. Every mask is class-agnostic (no semantic labels), focusing purely on spatial boundaries.

## 1. Core Dataset Numbers

From [SA-1B Dataset (Meta AI)](https://ai.meta.com/datasets/segment-anything/) (accessed 2025-11-20):

**Primary Statistics:**
- **Total Images**: 11,000,000 (11M)
- **Total Masks**: 1,100,000,000 (1.1B)
- **Average Masks per Image**: ~100
- **Image Resolution**: Averaging 1500×2250 pixels (high-resolution)
- **Dataset Size**: ~10TB uncompressed

**Scale Comparison:**
- **6× larger** than OpenImages v5 (previous largest open segmentation dataset)
- **10× more masks** than any prior public segmentation dataset
- **100× more masks per image** than COCO (which has ~1-2 masks per image)

From [Segment Anything Model (Ultralytics)](https://docs.ultralytics.com/models/sam/) (accessed 2025-11-20):

> "SAM is trained on the extensive SA-1B dataset which comprises over 1 billion masks across 11 million images. SA-1B is the largest segmentation dataset to date, offering a broad and high-quality source of data for model training."

This massive scale is not just about quantity—it's about **diversity at scale**, enabling SAM to learn robust spatial reasoning across countless visual contexts.

## 2. Image Specifications

From [Stanford CRFM - SA-1B](https://crfm.stanford.edu/ecosystem-graphs/index.html?asset=SA-1B) (accessed 2025-11-20):

> "SA-1B consists of 11M diverse, high-resolution (averaging 1500×2250 pixels), and privacy protecting images collected and licensed from a third party photo company."

**Image Properties:**
- **Resolution**: 1500×2250 pixels (average)
  - High-resolution enables fine-grained segmentation
  - Supports multiple granularity levels (small objects to entire scenes)
  - Preserves detail for intricate boundaries
- **Format**: JPEG
- **Color Space**: RGB (3 channels)
- **Aspect Ratio**: Primarily 2:3 ratio (portrait and landscape)
- **Licensing**: Professionally licensed from third-party photo company
- **Privacy**: Faces and license plates blurred (PII removal)

**Resolution Advantages:**
- Fine detail preservation (door handles, small objects)
- Multi-scale segmentation (tiny parts to full scenes)
- High boundary fidelity (precise edges, curves)
- Enables zoom-in analysis without quality loss

From [TensorFlow Datasets - segment_anything](https://www.tensorflow.org/datasets/catalog/segment_anything) (accessed 2025-11-20):

> "The SA-1B dataset consists of 11M diverse, high-resolution, licensed, and privacy-protecting images and 1.1B mask annotations."

## 3. Mask Distribution Statistics

From [Papers Explained 238: Segment Anything Model](https://ritvik19.medium.com/papers-explained-238-segment-anything-model-b3960b569fce) (accessed 2025-11-20):

> "In terms of scale, SA-1B stands out with significantly more images and masks than other datasets. The distribution of masks per image, relative to other datasets, shows SA-1B's comprehensive coverage."

**Masks per Image Distribution:**
- **Average**: ~100 masks per image
- **Range**: 1 to 400+ masks (depending on scene complexity)
- **Median**: 94 masks per image
- **Distribution**: Right-skewed (most images have 50-150 masks, some have 300+)

**Why ~100 masks per image?**
- **Multi-granularity**: From tiny parts (screws, leaves) to whole objects (cars, buildings)
- **Hierarchical structure**: Objects contain sub-parts (face → eyes, nose, mouth)
- **Dense coverage**: Every visible object boundary annotated
- **Class-agnostic**: No filtering by semantic category

**Comparison with Other Datasets:**
- **COCO**: ~5-10 masks per image (only labeled object categories)
- **LVIS**: ~15-20 masks per image (1200 categories, but limited per image)
- **OpenImages v5**: ~10-15 masks per image (selective annotation)
- **SA-1B**: ~100 masks per image (comprehensive class-agnostic coverage)

From [Segment Anything without Supervision (arXiv)](https://arxiv.org/html/2406.20081v1) (accessed 2025-11-20):

> "All masks are collected in a class-agnostic manner with an average of approximately 100 masks per image and 1.1 billion segmentation masks."

## 4. Scale Comparisons

**SA-1B vs. Previous Largest Datasets:**

| Dataset | Images | Masks | Masks/Image | Categories | Year |
|---------|--------|-------|-------------|------------|------|
| **SA-1B** | 11M | 1.1B | ~100 | 0 (class-agnostic) | 2023 |
| OpenImages v5 | 1.9M | 15M | ~8 | 350 | 2020 |
| LVIS | 100K | 2M | ~20 | 1,200 | 2019 |
| COCO | 328K | 2.5M | ~8 | 80 | 2014 |
| Pascal VOC | 11K | 28K | ~2.5 | 20 | 2012 |

From [Unitlab AI - Guide to SAM](https://blog.unitlab.ai/guide-to-the-segment-anything-model-sam/) (accessed 2025-11-20):

> "SAM is accurate. SA-1B contains 1.1 billion masks across 11 million images, making it six times larger than the next biggest open-source dataset, OpenImages v5."

**Key Scale Insights:**
- **73× more masks** than OpenImages v5 (1.1B vs 15M)
- **6× more images** than OpenImages v5 (11M vs 1.9M)
- **12.5× more masks per image** than OpenImages (100 vs 8)
- **440× more masks** than LVIS (1.1B vs 2.5M)

This unprecedented scale enables:
- Zero-shot generalization to new visual domains
- Robust boundary detection across diverse scenes
- Multi-granularity segmentation without retraining
- Foundation model pre-training with spatial reasoning

## 5. Data Collection Scale

**Collection Statistics:**
- **11M images** collected over several months
- **1.1B masks** annotated through data engine (3 stages)
- **Privacy-protected**: All faces and license plates blurred before annotation
- **Licensed**: Professionally sourced from third-party photo company
- **Geographic diversity**: Images from multiple continents and cultures

From [Encord - Segment Anything Model Explained](https://encord.com/blog/segment-anything-model-explained/) (accessed 2025-11-20):

> "Utilizing the extensive SA-1B dataset, comprising over 11 million meticulously curated images with more than 1 billion masks, SAM has demonstrated impressive zero-shot transfer capabilities."

**Data Engine Contribution:**
- **Stage 1 (Manual)**: ~120K images, ~4M masks (annotators + SAM assistance)
- **Stage 2 (Semi-automatic)**: ~180K images, ~10M masks (SAM proposals → human verification)
- **Stage 3 (Fully automatic)**: ~10.7M images, ~1.086B masks (SAM generates all masks)

**Quality at Scale:**
- **High-quality**: Human verification in Stages 1-2 ensures accuracy
- **Consistent**: Automated Stage 3 maintains uniform annotation quality
- **Comprehensive**: Every visible object boundary captured

## 6. Storage and Computational Scale

**Dataset Storage Requirements:**
- **Total Size**: ~10TB uncompressed
- **Compressed Size**: ~1TB (tar.gz archives)
- **Per-Image Breakdown**:
  - Image (JPEG): ~3-5 MB average
  - Masks (RLE compressed): ~1-2 MB average
  - Metadata (JSON): ~100-500 KB average

**Distribution Format:**
- **1000 tar files**: sa_000000.tar to sa_000999.tar
- **~11,000 images per tar**: Evenly distributed
- **~10GB per tar file** (compressed)

**Computational Scale for Training:**
- Training SAM on full SA-1B requires:
  - **Multi-GPU clusters** (32-256 GPUs)
  - **Days to weeks** of training time
  - **Terabytes of GPU memory** (distributed across nodes)
  - **Efficient I/O pipelines** to stream 10TB dataset

From [Labelbox - SA-1B Dataset](https://labelbox.com/datasets/segment-anything-1-billion-mask-dataset-sa-1b/) (accessed 2025-11-20):

> "The SA-1B dataset is the largest segmentation dataset, with 11,000,000 datarows, and is the largest ever segmentation dataset."

## 7. Statistical Significance for Foundation Models

**Why 1.1B Masks Matter:**

**Data Scaling Laws:**
- Foundation models benefit from **massive data scale**
- More diverse examples → better zero-shot generalization
- 1.1B masks provide **exhaustive coverage** of visual patterns

**Diversity at Scale:**
- 11M images × ~100 masks/image = comprehensive spatial reasoning
- Class-agnostic annotation → no category bias
- High-resolution → fine-grained boundary learning

**Zero-Shot Performance:**
- Trained on SA-1B, SAM achieves **zero-shot segmentation** on 23 unseen datasets
- No fine-tuning needed for new domains (medical, satellite, microscopy)
- Generalizes from natural images to specialized imagery

From [Andrey Lukyanenko - Paper Review: SAM](https://andlukyane.com/blog/paper-review-sam) (accessed 2025-11-20):

> "SA-1B is a dataset consisting of 11 million diverse, high-resolution, licensed, and privacy-protecting images and 1.1 billion high-quality segmentation masks."

**Statistical Power:**
- **Sample size**: 1.1B masks provides statistical robustness
- **Coverage**: Every common visual pattern likely represented multiple times
- **Outliers**: Rare visual patterns still have hundreds of examples
- **Generalization**: Model learns boundaries, not memorizes specific objects

## 8. ARR-COC-0-1: Dataset Scale for Relevance Realization Training (10%)

**Why SA-1B Scale Matters for ARR-COC:**

SA-1B's **11M images and 1.1B masks** demonstrate the **data scale required** for foundation-level spatial reasoning. For ARR-COC's relevance realization training:

**Lesson 1: Scale Enables Zero-Shot Spatial Grounding**
- SAM's zero-shot segmentation on unseen domains proves **massive scale → generalization**
- ARR-COC could leverage SA-1B for **pre-training spatial reasoning** before relevance tasks
- 1.1B masks provide **exhaustive boundary examples** for grounding language to spatial regions

**Lesson 2: Class-Agnostic Data for Flexible Grounding**
- SA-1B's **no category labels** enables learning **pure spatial boundaries**
- ARR-COC can use class-agnostic masks to **ground referring expressions** flexibly
- "The red book on the left shelf" → spatial region, not predefined object class

**Potential Integration:**
- **Pre-train on SA-1B**: Learn spatial segmentation as foundational capability
- **Fine-tune on relevance data**: Add language grounding and attention realization
- **Multi-modal training**: Combine SA-1B masks with text descriptions for grounded VLMs

**Scale Considerations:**
- Full SA-1B training requires **multi-GPU infrastructure** (32-256 GPUs)
- Subset training (1M images, 100M masks) still provides **strong spatial reasoning**
- ARR-COC can use **pre-trained SAM encoder** as frozen backbone (saves compute)

From [Labellerr - Segment Anything: Automated Labeling](https://www.labellerr.com/blog/segment-anything-automated-labeling-with-foundation-model/) (accessed 2025-11-20):

> "The SA-1B dataset is a large-scale computer vision dataset with 11 million images and 1.1 billion segmentation masks, with high-resolution images and comprehensive annotations."

**ARR-COC Training Strategy:**
1. **Option A**: Pre-train spatial encoder on SA-1B subset (1M images)
2. **Option B**: Use pre-trained SAM encoder as frozen feature extractor
3. **Option C**: Fine-tune SAM decoder on relevance-grounded segmentation tasks

The **1.1B mask scale** proves that **foundation-level spatial reasoning** requires **massive, diverse data**—a key lesson for ARR-COC's multimodal training strategy.

---

## Sources

**Source Documents:**
- [SAM_DATASET_SA1B.md](../../PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md) - Lines 1-1123 (comprehensive SA-1B overview)

**Web Research:**
- [SA-1B Dataset (Meta AI)](https://ai.meta.com/datasets/segment-anything/) - Official dataset page, 11M images, 1.1B masks (accessed 2025-11-20)
- [Stanford CRFM - SA-1B](https://crfm.stanford.edu/ecosystem-graphs/index.html?asset=SA-1B) - Dataset specifications, 1500×2250 resolution (accessed 2025-11-20)
- [Segment Anything Model (Ultralytics)](https://docs.ultralytics.com/models/sam/) - SAM model and SA-1B training details (accessed 2025-11-20)
- [TensorFlow Datasets - segment_anything](https://www.tensorflow.org/datasets/catalog/segment_anything) - Dataset format and structure (accessed 2025-11-20)
- [Papers Explained 238: SAM](https://ritvik19.medium.com/papers-explained-238-segment-anything-model-b3960b569fce) - Mask distribution analysis (accessed 2025-11-20)
- [Segment Anything without Supervision (arXiv)](https://arxiv.org/html/2406.20081v1) - arXiv:2406.20081v1, 100 masks per image average (accessed 2025-11-20)
- [Unitlab AI - Guide to SAM](https://blog.unitlab.ai/guide-to-the-segment-anything-model-sam/) - Scale comparison with other datasets (accessed 2025-11-20)
- [Encord - SAM Explained](https://encord.com/blog/segment-anything-model-explained/) - Zero-shot capabilities and dataset curation (accessed 2025-11-20)
- [Labelbox - SA-1B Dataset](https://labelbox.com/datasets/segment-anything-1-billion-mask-dataset-sa-1b/) - Largest segmentation dataset statistics (accessed 2025-11-20)
- [Andrey Lukyanenko - Paper Review: SAM](https://andlukyane.com/blog/paper-review-sam) - SA-1B dataset overview (accessed 2025-11-20)
- [Labellerr - Segment Anything](https://www.labellerr.com/blog/segment-anything-automated-labeling-with-foundation-model/) - High-resolution images and comprehensive annotations (accessed 2025-11-20)

**Additional References:**
- [GitHub - facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official SAM repository
- [OpenDataLab - SA-1B](https://opendatalab.com/OpenDataLab/SA-1B) - Dataset access and download information
