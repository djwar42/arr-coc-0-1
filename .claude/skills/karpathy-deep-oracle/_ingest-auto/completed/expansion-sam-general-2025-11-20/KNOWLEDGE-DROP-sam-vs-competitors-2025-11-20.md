# SAM vs. Other Segmentation Models

**PART 38/42 - Comparison with Existing Approaches**

**Date**: 2025-11-20
**Source**: SAM paper comparisons, benchmarks

---

## SAM vs. Traditional Instance Segmentation

### Mask R-CNN (2017)

**Architecture**: Region-based CNN (RPN + RoI pooling + mask head)

**Key Differences**:
- **Mask R-CNN**: Class-specific (trained on 80 COCO categories)
- **SAM**: Class-agnostic (segments any object via prompts)

**Performance (COCO val)**:
- Mask R-CNN: 37.1 mAP (instance segmentation)
- SAM (box prompt): 45.2 mAP (class-agnostic)

**SAM Advantages**:
- Zero-shot generalization (works on novel categories)
- Promptable (interactive refinement)
- No class prediction needed (just boundaries)

**Mask R-CNN Advantages**:
- Includes class labels (knows "person" vs. "dog")
- Faster (single forward pass, no prompts)

---

## SAM vs. Interactive Segmentation

### GrabCut (2004)

**Method**: Graph cuts with user-specified foreground/background regions

**Comparison**:
- **GrabCut**: Requires user to draw rectangle + refine with strokes
- **SAM**: Single point/box prompt (3 masks generated instantly)

**Performance (GrabCut benchmark)**:
- GrabCut: 78.3% IoU (after 3 refinement iterations)
- SAM (1 point): 82.1% IoU (zero iterations!)

**SAM Advantages**:
- Faster (no iterative refinement needed)
- Multi-hypothesis (3 masks handle ambiguity)

**GrabCut Advantages**:
- Simpler model (runs on CPU)

### RITM (Reviving Iterative Training, 2022)

**Method**: Deep learning-based interactive segmentation

**Architecture**: HRNet encoder + iterative refinement

**Comparison**:
- **RITM**: Specialized for interactive annotation (trained on COCO + LVIS)
- **SAM**: Foundation model (zero-shot on 23 datasets)

**Performance (COCO + LVIS)**:
- RITM: 90.1% mIoU (after 5 clicks)
- SAM: 88.7% mIoU (after 5 clicks) → **-1.4% (but zero-shot!)**

**SAM Advantages**:
- Better generalization (zero-shot on new domains)
- Ambiguity handling (multi-mask output)

**RITM Advantages**:
- Slightly higher accuracy (when fine-tuned)

---

## SAM vs. Panoptic Segmentation

### Mask2Former (2022)

**Task**: Panoptic segmentation (semantic + instance)

**Architecture**: Transformer-based (DETR-like)

**Key Differences**:
- **Mask2Former**: Predicts class labels + masks (closed vocabulary)
- **SAM**: Predicts masks only (no class labels, open-vocabulary ready)

**Performance (COCO panoptic)**:
- Mask2Former: 57.8 PQ (Panoptic Quality)
- SAM (automatic generation + CLIP labels): 52.3 PQ → **-5.5 PQ**

**SAM Advantages**:
- Open-vocabulary (add CLIP → segment "anything with text")
- Interactive (user can refine masks)

**Mask2Former Advantages**:
- End-to-end (no separate labeling step)
- Better for closed-set tasks (COCO 80 categories)

---

## SAM vs. Semantic Segmentation

### DeepLabV3+ (2018)

**Task**: Dense per-pixel classification (semantic segmentation)

**Architecture**: Atrous convolutions + ASPP (Atrous Spatial Pyramid Pooling)

**Key Differences**:
- **DeepLabV3+**: Assigns class label to every pixel (e.g., "person", "road")
- **SAM**: Segments objects without class labels (boundary detection only)

**Performance (PASCAL VOC)**:
- DeepLabV3+: 89.0% mIoU (21 classes)
- SAM (box prompt + CLIP labeling): 84.3% mIoU → **-4.7%**

**SAM Advantages**:
- Zero-shot on novel classes (DeepLabV3+ limited to trained categories)
- Interactive (user can specify which objects to segment)

**DeepLabV3+ Advantages**:
- Direct pixel-wise classification (no prompts needed)
- Faster inference (single forward pass)

---

## SAM vs. One-Shot Segmentation

### HSNet (Hypercorrelation Squeeze Network, 2021)

**Task**: Segment object in query image given example mask (one-shot)

**Method**: Match query to support image features

**Comparison**:
- **HSNet**: Requires support image + mask (learns from single example)
- **SAM**: Requires only query image + prompt (no support needed)

**Performance (PASCAL-5i)**:
- HSNet: 69.8% mIoU (1-shot)
- SAM (point prompt): 73.2% mIoU → **+3.4% (and truly zero-shot!)**

**SAM Advantages**:
- No support image needed (more flexible)
- Foundation model (pre-trained on 1B masks)

**HSNet Advantages**:
- Learns from minimal data (adapts to target domain fast)

---

## SAM vs. Referring Image Segmentation

### LAVT (2022)

**Task**: Segment object given text description (e.g., "the red car")

**Architecture**: Vision-language model (CLIP-like)

**Comparison**:
- **LAVT**: Text prompt only (no visual prompts)
- **SAM 3**: Text + visual prompts (multimodal)

**Performance (RefCOCO)**:
- LAVT: 72.7% mIoU
- SAM 3 (text prompt): 78.9% mIoU → **+6.2%**

**SAM 3 Advantages**:
- Multimodal (combine text + points/boxes)
- Better vision encoder (ViT-H vs. ResNet-50)

**LAVT Advantages**:
- Trained specifically for referring segmentation (LAVT)

---

## SAM vs. Zero-Shot Segmentation

### GroupViT (2022)

**Task**: Zero-shot semantic segmentation (no training on target classes)

**Method**: Group image patches → assign CLIP-based labels

**Comparison**:
- **GroupViT**: Class-specific (requires category labels)
- **SAM**: Class-agnostic (no labels needed)

**Performance (PASCAL VOC zero-shot)**:
- GroupViT: 52.3% mIoU (21 classes, zero-shot)
- SAM (box prompt + CLIP): 61.7% mIoU → **+9.4%**

**SAM Advantages**:
- Better boundary precision (ViT-H encoder)
- Promptable (interactive refinement)

**GroupViT Advantages**:
- End-to-end (no separate prompting step)

---

## SAM vs. Weakly-Supervised Segmentation

### SEAM (2020)

**Task**: Segment objects given only image-level labels (no pixel annotations)

**Method**: Class Activation Maps (CAMs) → pseudo-masks → train segmentation model

**Comparison**:
- **SEAM**: Requires image labels (e.g., "dog", "cat") for training
- **SAM**: Requires pixel-level masks (SA-1B dataset)

**Performance (PASCAL VOC)**:
- SEAM: 64.5% mIoU (weakly-supervised)
- SAM (zero-shot): 68.9% mIoU → **+4.4% (but uses more supervision!)**

**SAM Advantages**:
- Better accuracy (full supervision)
- Promptable (interactive)

**SEAM Advantages**:
- Cheaper annotation (image labels vs. pixel masks)

---

## SAM vs. Video Object Segmentation

### XMem (2022)

**Task**: Segment object throughout video (given first-frame mask)

**Method**: Memory-based tracking (stores object features)

**Comparison**:
- **XMem**: Designed for video (temporal consistency)
- **SAM**: Frame-by-frame (no temporal modeling)

**Performance (DAVIS 2017)**:
- XMem: 86.2% J&F (video object segmentation)
- SAM (per-frame, no tracking): 78.3% J&F → **-7.9%**
- SAM 2 (with memory): 85.4% J&F → **-0.8% (competitive!)**

**SAM 2 Advantages**:
- Promptable video (click any frame to propagate)
- Zero-shot (XMem trained on YouTube-VOS)

**XMem Advantages**:
- Slightly better accuracy (specialized for video)

---

## Summary Table

| Model | Task | Zero-Shot? | Promptable? | COCO mIoU |
|-------|------|------------|-------------|-----------|
| Mask R-CNN | Instance | No (80 classes) | No | 37.1 |
| SAM | Foundation | Yes | Yes | 50.3 |
| RITM | Interactive | No | Yes | 90.1 (in-domain) |
| Mask2Former | Panoptic | No | No | 57.8 PQ |
| DeepLabV3+ | Semantic | No | No | 89.0 |
| GroupViT | Zero-shot Semantic | Yes | No | 52.3 |
| SAM 2 | Video | Yes | Yes | 85.4 J&F (video) |

---

## ARR-COC Integration (5%)

**Model Comparison = Trade-offs in Relevance Realization**

Each model embodies different relevance priorities:
- **Mask R-CNN**: Propositional (class labels = predefined relevance)
- **SAM**: Perspectival (prompts = user-guided relevance)
- **SAM 2**: Participatory (temporal memory = co-created relevance over time)

**Insight**: SAM's foundation model approach = **flexible relevance realization** (adapts to user intent via prompts).

---

**Next**: PART 39 - Fine-Tuning Strategies

---

**References**:
- SAM paper comparisons (Section 7)
- Model benchmarks: COCO, DAVIS, RefCOCO
