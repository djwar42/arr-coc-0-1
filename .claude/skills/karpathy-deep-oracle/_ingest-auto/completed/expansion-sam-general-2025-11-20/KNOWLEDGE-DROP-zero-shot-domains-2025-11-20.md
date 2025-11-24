# SAM: Zero-Shot Domain Transfer

**PART 29/42 - Zero-Shot Transfer Across 23 Diverse Domains**

**Date**: 2025-11-20
**Source**: SAM paper Section 7 (Zero-Shot Transfer Experiments)

---

## Zero-Shot Transfer Overview

**Definition**: SAM performs segmentation on entirely new domains/datasets without fine-tuning, using only the pre-trained model.

**Key Achievement**: Competitive or superior performance vs. fully supervised models on 23 datasets spanning 8 domain categories.

---

## 23 Datasets Evaluated

### 1. Objects (ADE20K, COCO, etc.)
- **ADE20K**: Indoor/outdoor scenes, 150 semantic categories
- **COCO**: Common objects in context
- **LVIS**: Large vocabulary instance segmentation (1,203 categories)

**SAM Performance**:
- ADE20K: 47.3 mIoU (center point prompt)
- COCO: Competitive with fully supervised ViTDet

### 2. Parts (PACO-LVIS, PartImageNet)
- **PACO-LVIS**: Object part segmentation
- **PartImageNet**: Fine-grained part annotations

**Challenge**: Requires precise boundaries between adjacent parts

**SAM Advantage**: Ambiguity-aware (multi-mask output handles uncertainty)

### 3. Medical Imaging
- **Breast ultrasound**: Tumor detection
- **Colonoscopy**: Polyp segmentation
- **Chest X-ray**: Lung/heart segmentation

**Transfer Gap**: Natural images → medical domain (different modality, limited training data)

**SAM Performance**: Comparable to task-specific models without medical fine-tuning

### 4. Remote Sensing
- **Satellite imagery**: Building/road detection
- **Aerial photography**: Land use classification

**Challenge**: Different resolution, spectral bands vs. RGB images

**SAM Capability**: Box prompts work well for geospatial object detection

### 5. Document Analysis
- **Receipt scanning**: Text region extraction
- **Form parsing**: Field boundary detection

**SAM Use**: Delineates text blocks without OCR-specific training

### 6. Microscopy
- **Cell segmentation**: Biological samples
- **Tissue analysis**: Histopathology slides

**Domain Shift**: Microscopic vs. macroscopic imagery

**SAM Robustness**: Prompt-guided segmentation adapts to cellular structures

### 7. Underwater Imagery
- **Marine biology**: Fish/coral segmentation
- **Underwater robotics**: Object detection

**Challenge**: Low visibility, color distortion, suspended particles

**SAM Generalization**: Handles degraded image quality via robust ViT-H encoder

### 8. Thermal Imaging
- **Infrared cameras**: Heat signature segmentation
- **Night vision**: Low-light object detection

**Modality Gap**: Thermal vs. RGB (no color information)

**SAM Transfer**: Texture/shape cues still enable segmentation

---

## Prompt Strategies for Zero-Shot Transfer

### 1. Center Point Prompt (Simplest)
- **Method**: Single click at object center
- **Use Case**: Quick object extraction, ambiguous boundaries
- **Performance**: 47.3 mIoU on ADE20K (baseline)

### 2. Box Prompt (Most Reliable)
- **Method**: Bounding box around target
- **Use Case**: Known object location, precise boundaries
- **Performance**: Higher precision than point prompts

### 3. Mask Prompt (Iterative Refinement)
- **Method**: Coarse mask → SAM refines boundaries
- **Use Case**: Complex shapes, iterative annotation
- **Workflow**: Human sketch → SAM correction → final mask

### 4. Multi-Prompt Combination
- **Method**: Points + boxes + mask (ensemble)
- **Use Case**: Maximum accuracy, challenging cases
- **Performance**: Best results but slower inference

---

## Domain Adaptation Without Fine-Tuning

### Why SAM Generalizes

**1. Massive Pretraining (SA-1B)**
- 1.1 billion masks, 11 million images
- Diverse object scales, lighting, occlusions
- Data engine ensures broad distribution coverage

**2. Promptable Interface**
- Flexible input (points/boxes/masks)
- Adapts to domain-specific annotation workflows
- No need for domain-specific architectures

**3. Ambiguity-Aware Design**
- Multi-mask output handles uncertain boundaries
- Confidence scores (IoU prediction) guide selection
- Graceful degradation on out-of-distribution data

**4. ViT-H Encoder Robustness**
- MAE pre-trained on ImageNet
- Learns robust features (texture, shape, context)
- 630M parameters capture rich visual patterns

---

## Comparison: SAM vs. Fully Supervised Models

### Medical Imaging (MedSAM Benchmark)

| Dataset | Fully Supervised | SAM (Zero-Shot) | Gap |
|---------|------------------|-----------------|-----|
| Polyp segmentation | 82.3 mIoU | 78.1 mIoU | -4.2 |
| Breast ultrasound | 76.5 Dice | 73.2 Dice | -3.3 |
| Chest X-ray | 89.1 mIoU | 85.7 mIoU | -3.4 |

**Insight**: SAM achieves 90-95% of supervised performance without medical training data!

### Remote Sensing (Satellite Imagery)

| Task | Supervised ViT | SAM (Zero-Shot) | Gap |
|------|----------------|-----------------|-----|
| Building detection | 71.2 F1 | 68.9 F1 | -2.3 |
| Road extraction | 78.5 IoU | 74.1 IoU | -4.4 |

**Insight**: SAM's box prompts enable geospatial object detection without GIS-specific training.

### Underwater Imagery

| Metric | Task-Specific CNN | SAM (Zero-Shot) | Gap |
|--------|-------------------|-----------------|-----|
| Fish segmentation | 65.3 mIoU | 61.8 mIoU | -3.5 |
| Coral detection | 58.7 mIoU | 54.2 mIoU | -4.5 |

**Challenge**: Underwater domain shift is larger (color distortion, turbidity)

**SAM Limitation**: RGB-trained model struggles with heavy color cast

---

## Failure Cases and Limitations

### 1. Extreme Domain Shift
- **Infrared/Thermal**: No RGB color cues
- **X-ray/CT scans**: 2D projections of 3D anatomy
- **Performance Drop**: 10-15% lower than supervised baselines

**Mitigation**: Fine-tune SAM on small domain-specific datasets (MedSAM approach)

### 2. Fine-Grained Boundaries
- **Adjacent objects**: Hard to separate without semantic understanding
- **Transparent/reflective surfaces**: Boundary ambiguity
- **Performance**: Multi-mask output helps but not perfect

**Workaround**: Use mask prompts for iterative refinement

### 3. Small Object Detection
- **Tiny objects** (<10 pixels): May be missed without precise prompts
- **High-resolution images**: Downsampling loses detail

**Solution**: Multi-scale inference, patch-based processing

### 4. Text/Semantic Reasoning
- **SAM lacks**: Language understanding, object relationships
- **Cannot**: "Segment all dogs" or "Find the largest tree"

**Requires**: External text encoder (SAM 3 addresses this with CLIP)

---

## ARR-COC Integration (5%)

### Relevance Realization and Zero-Shot Transfer

**Connection**: SAM's zero-shot ability mirrors ARR-COC's relevance generalization:

1. **Propositional Knowing**: SAM learns mask/no-mask boundary rules
2. **Perspectival Knowing**: Prompts shift attention to relevant image regions
3. **Participatory Knowing**: Iterative refinement (mask prompts) = co-creation

**ARR-COC Insight**: Zero-shot transfer = relevance realization across domains without explicit labels!

### Domain-Agnostic Relevance

**SAM's Prompts** = **ARR-COC Salience Landscape**:
- Point prompt → focal attention peak
- Box prompt → bounded relevance region
- Mask prompt → relevance template

**Hypothesis**: SAM's ViT-H encoder learns universal relevance features (edges, textures, gestalt) that transfer across domains.

---

## Practical Deployment Patterns

### 1. Interactive Annotation Tools
- **Workflow**: Point → SAM predicts → User refines → Export mask
- **Speed**: 10× faster than manual polygon tracing
- **Domains**: Medical imaging, satellite analysis, robotics

### 2. Data Augmentation Pipelines
- **Use**: Generate masks for semi-supervised learning
- **Quality**: SAM masks used as pseudo-labels for downstream tasks
- **Benefit**: Reduces manual annotation cost by 80%

### 3. Object-Centric Models
- **Integration**: SAM → extract objects → feed to classifiers/detectors
- **Example**: SAM segments proposals, CLIP classifies them
- **Result**: Open-vocabulary object detection

### 4. Robotic Manipulation
- **Application**: Grasp point selection via SAM segmentation
- **Challenge**: Real-time inference (30 FPS required)
- **Solution**: SAM-Fast (distilled model), TensorRT optimization

---

## Key Takeaways

1. **Zero-Shot = 90-95% Supervised**: SAM competes with fully trained models without fine-tuning
2. **23 Datasets, 8 Domains**: Broad evaluation across natural/medical/remote sensing/underwater
3. **Prompt Flexibility**: Points/boxes/masks enable domain-specific workflows
4. **Failure Cases**: Extreme shifts (thermal/X-ray), fine-grained boundaries, tiny objects
5. **ARR-COC**: Zero-shot transfer = relevance realization across domains

**Next**: PART 30 - Automatic Mask Generation (grid-based inference for dense segmentation)

---

**References**:
- Kirillov et al., "Segment Anything" (ICCV 2023) - Section 7
- MedSAM benchmark (arXiv:2304.12306)
- SAM zero-shot experiments (23 datasets)
